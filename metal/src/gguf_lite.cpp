// metal/src/gguf_lite.cpp - see gguf_lite.h.
#include "gguf_lite.h"

#include "ref_quant.h"

#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

#include <cstdio>
#include <cstring>

namespace gguf {

namespace {
struct Reader {
    const uint8_t* p;
    const uint8_t* end;
    bool ok = true;
    template <class T> T get() {
        T v{};
        if (!ok || (size_t)(end - p) < sizeof(T)) { ok = false; return v; }   // after a failure every read is 0
        std::memcpy(&v, p, sizeof(T));
        p += sizeof(T);
        return v;
    }
    std::string str() {
        const uint64_t n = get<uint64_t>();
        if (!ok || n > (uint64_t)(end - p)) { ok = false; return {}; }
        std::string s((const char*)p, n);
        p += n;
        return s;
    }
};
// value types: 0 u8 1 i8 2 u16 3 i16 4 u32 5 i32 6 f32 7 bool 8 string 9 array 10 u64 11 i64 12 f64
size_t scalar_size(uint32_t t) {
    switch (t) {
        case 0: case 1: case 7: return 1;
        case 2: case 3: return 2;
        case 4: case 5: case 6: return 4;
        case 10: case 11: case 12: return 8;
    }
    return 0;
}
double scalar_num(Reader& r, uint32_t t) {
    switch (t) {
        case 0: return r.get<uint8_t>(); case 1: return r.get<int8_t>(); case 2: return r.get<uint16_t>();
        case 3: return r.get<int16_t>(); case 4: return r.get<uint32_t>(); case 5: return r.get<int32_t>();
        case 6: return r.get<float>(); case 7: return r.get<uint8_t>(); case 10: return (double)r.get<uint64_t>();
        case 11: return (double)r.get<int64_t>(); case 12: return r.get<double>();
    }
    r.ok = false;
    return 0;
}
}  // namespace

bool File::open(const std::string& p, std::string& err) {
    path = p;
    const int fd = ::open(p.c_str(), O_RDONLY);
    if (fd < 0) { err = "cannot open " + p; return false; }
    struct stat st;
    if (fstat(fd, &st) != 0 || st.st_size <= 0) { ::close(fd); err = "empty or unreadable file: " + p; return false; }
    file_bytes = (size_t)st.st_size;
    const size_t page = (size_t)getpagesize();
    map_bytes = (file_bytes + page - 1) / page * page;
    map = mmap(nullptr, map_bytes, PROT_READ, MAP_SHARED, fd, 0);
    ::close(fd);
    if (map == MAP_FAILED) { map = nullptr; err = "mmap failed: " + p; return false; }
    Reader r{(const uint8_t*)map, (const uint8_t*)map + file_bytes};
    if (r.get<uint32_t>() != 0x46554747u) { err = "not a GGUF file"; return false; }
    if (r.get<uint32_t>() != 3) { err = "GGUF version is not 3"; return false; }
    const uint64_t n_t = r.get<uint64_t>(), n_kv = r.get<uint64_t>();
    uint64_t align = 32;
    std::string bad;
    for (uint64_t i = 0; i < n_kv && r.ok; ++i) {
        const std::string key = r.str();
        const uint32_t t = r.get<uint32_t>();
        if (t == 8) meta_str[key] = r.str();
        else if (t == 9) {
            const uint32_t et = r.get<uint32_t>();
            const uint64_t n = r.get<uint64_t>();
            if (et == 8) for (uint64_t k = 0; k < n && r.ok; ++k) r.str();
            else {
                const size_t sz = scalar_size(et);   // a nested array or an unknown type is refused
                if (sz == 0) { bad = "GGUF array of type " + std::to_string(et) + ": " + key; r.ok = false; break; }
                if (n > (uint64_t)(r.end - r.p) / sz) { r.ok = false; break; }
                r.p += n * sz;
            }
        } else if (scalar_size(t) == 0) {
            bad = "unknown GGUF value type " + std::to_string(t) + ": " + key;
            r.ok = false;
        } else {
            const double v = scalar_num(r, t);
            meta_num[key] = v;
            // the spec's field is a uint32; out of that range (or NaN) it is 0
            if (key == "general.alignment") align = v >= 1 && v <= 4294967295.0 ? (uint64_t)v : 0;
        }
    }
    std::vector<Tensor> ts;
    for (uint64_t i = 0; i < n_t && r.ok; ++i) {
        Tensor t;
        t.name = r.str();
        const uint32_t nd = r.get<uint32_t>();
        if (nd > 4) { bad = "tensor has " + std::to_string(nd) + " dimensions: " + t.name; r.ok = false; break; }
        for (uint32_t k = 0; k < nd; ++k) t.shape.push_back(r.get<uint64_t>());
        t.type = (int)r.get<uint32_t>();
        t.offset = r.get<uint64_t>();
        ts.push_back(t);
    }
    if (!r.ok) { err = bad.empty() ? "truncated GGUF header" : bad; return false; }
    if (align == 0 || (align & (align - 1)) != 0) { err = "general.alignment is not a power of two"; return false; }
    const uint64_t pos = (uint64_t)(r.p - (const uint8_t*)map);
    const uint64_t data = (pos + align - 1) / align * align;
    for (Tensor& t : ts) {
        t.offset += data;
        uint64_t n = 1;
        for (uint64_t d : t.shape)
            if (__builtin_mul_overflow(n, d, &n)) { err = "tensor shape overflows: " + t.name; return false; }
        if (!ref::known_type(t.type)) {
            err = "unknown tensor type " + std::to_string(t.type) + ": " + t.name;
            return false;
        }
        const uint64_t be = ref::block_elems(t.type);
        const uint64_t bb = (uint64_t)ref::block_bytes(t.type);
        if (n / be > UINT64_MAX / bb) { err = "tensor size overflows: " + t.name; return false; }
        t.bytes = n / be * bb;
        if (t.offset < data || t.offset > file_bytes || t.bytes > file_bytes - t.offset) {
            err = "tensor lies outside the file: " + t.name;
            return false;
        }
        tensors[t.name] = t;
    }
    return true;
}

File::~File() {
    if (map) munmap(map, map_bytes);
}

}  // namespace gguf
