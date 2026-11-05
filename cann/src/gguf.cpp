#include "strata/cann/gguf.hpp"

#include <cstring>
#include <fstream>
#include <stdexcept>

namespace strata::cann {
namespace {

enum class Kind : uint32_t {
    U8 = 0, I8 = 1, U16 = 2, I16 = 3, U32 = 4, I32 = 5, F32 = 6, Bool = 7,
    String = 8, Array = 9, U64 = 10, I64 = 11, F64 = 12,
};

struct Reader {
    std::ifstream in;
    explicit Reader(const std::string & path) : in(path, std::ios::binary) {
        if (!in) {
            throw std::runtime_error("gguf: cannot open " + path);
        }
    }

    template <typename T>
    T get() {
        T v{};
        in.read((char *) &v, sizeof(T));
        if (!in) {
            throw std::runtime_error("gguf: unexpected end of file");
        }
        return v;
    }

    std::string str() {
        const uint64_t n = get<uint64_t>();
        std::string s(n, '\0');
        in.read(s.data(), (std::streamsize) n);
        if (!in) {
            throw std::runtime_error("gguf: truncated string");
        }
        return s;
    }

    void skip_value(Kind k, int depth = 0) {
        switch (k) {
            case Kind::U8: case Kind::I8: case Kind::Bool: get<uint8_t>(); break;
            case Kind::U16: case Kind::I16: get<uint16_t>(); break;
            case Kind::U32: case Kind::I32: case Kind::F32: get<uint32_t>(); break;
            case Kind::U64: case Kind::I64: case Kind::F64: get<uint64_t>(); break;
            case Kind::String: str(); break;
            case Kind::Array: {
                const uint32_t elem = get<uint32_t>();
                const uint64_t count = get<uint64_t>();
                if (depth > 2) {
                    throw std::runtime_error("gguf: nested array too deep");
                }
                for (uint64_t i = 0; i < count; ++i) {
                    skip_value((Kind) elem, depth + 1);
                }
                break;
            }
            default:
                throw std::runtime_error("gguf: unknown value type " + std::to_string((int) k));
        }
    }
};

uint64_t product(const std::vector<uint64_t> & dims) {
    uint64_t n = 1;
    for (uint64_t d : dims) {
        n *= d;
    }
    return n;
}

} // namespace

uint64_t GgufTensor::elements() const {
    return product(dims);
}

Gguf::Gguf(const std::string & path) : path_(path) {
    Reader r(path);

    char magic[4] = { 0, 0, 0, 0 };
    r.in.read(magic, 4);
    if (std::memcmp(magic, "GGUF", 4) != 0) {
        throw std::runtime_error("gguf: bad magic");
    }
    version_ = r.get<uint32_t>();
    const uint64_t n_tensors = r.get<uint64_t>();
    const uint64_t n_kv = r.get<uint64_t>();

    for (uint64_t i = 0; i < n_kv; ++i) {
        const std::string key = r.str();
        const auto kind = (Kind) r.get<uint32_t>();
        switch (kind) {
            case Kind::String: strings_[key] = r.str(); break;
            case Kind::U8: ints_[key] = r.get<uint8_t>(); break;
            case Kind::I8: ints_[key] = r.get<int8_t>(); break;
            case Kind::U16: ints_[key] = r.get<uint16_t>(); break;
            case Kind::I16: ints_[key] = r.get<int16_t>(); break;
            case Kind::U32: ints_[key] = r.get<uint32_t>(); break;
            case Kind::I32: ints_[key] = r.get<int32_t>(); break;
            case Kind::U64: ints_[key] = (int64_t) r.get<uint64_t>(); break;
            case Kind::I64: ints_[key] = r.get<int64_t>(); break;
            // keep the value: truncating 1e-6 to an integer made every eps and every float
            // hyperparameter read back as 0, which is wrong without ever failing
            case Kind::F32: {
                const float f = r.get<float>();
                floats_[key] = (double) f;
                ints_[key] = (int64_t) f;
                break;
            }
            case Kind::F64: {
                const double f = r.get<double>();
                floats_[key] = f;
                ints_[key] = (int64_t) f;
                break;
            }
            case Kind::Bool: ints_[key] = r.get<uint8_t>() ? 1 : 0; break;
            default: r.skip_value(kind); break;
        }
    }

    tensors_.reserve((size_t) n_tensors);
    for (uint64_t i = 0; i < n_tensors; ++i) {
        GgufTensor t;
        t.name = r.str();
        const uint32_t n_dims = r.get<uint32_t>();
        t.dims.resize(n_dims);
        for (uint32_t d = 0; d < n_dims; ++d) {
            t.dims[d] = r.get<uint64_t>();
        }
        t.type = r.get<uint32_t>();
        t.offset = r.get<uint64_t>();
        tensors_.push_back(std::move(t));
    }

    uint64_t align = 32;
    const auto it = ints_.find("general.alignment");
    if (it != ints_.end() && it->second > 0) {
        align = (uint64_t) it->second;
    }
    const uint64_t here = (uint64_t) r.in.tellg();
    data_offset_ = (here + align - 1) / align * align;
}

std::string Gguf::arch() const {
    return metadata_str("general.architecture");
}

int64_t Gguf::metadata_int(const std::string & key) const {
    const auto it = ints_.find(key);
    return it == ints_.end() ? 0 : it->second;
}

double Gguf::metadata_flt(const std::string & key) const {
    const auto it = floats_.find(key);
    return it == floats_.end() ? 0.0 : it->second;
}

std::string Gguf::metadata_str(const std::string & key) const {
    const auto it = strings_.find(key);
    return it == strings_.end() ? std::string() : it->second;
}

const GgufTensor * Gguf::find(const std::string & name) const {
    for (const auto & t : tensors_) {
        if (t.name == name) {
            return &t;
        }
    }
    return nullptr;
}

std::vector<uint8_t> Gguf::tensor_bytes(const GgufTensor & t, uint64_t offset, uint64_t bytes) const {
    std::ifstream in(path_, std::ios::binary);
    if (!in) {
        throw std::runtime_error("gguf: cannot reopen " + path_);
    }
    in.seekg((std::streamoff) (data_offset_ + t.offset + offset));
    std::vector<uint8_t> out(bytes);
    in.read((char *) out.data(), (std::streamsize) bytes);
    if (!in) {
        throw std::runtime_error("gguf: short read for " + t.name);
    }
    return out;
}

} // namespace strata::cann
