#include "strata/weights/safetensors.hpp"
#include "json_checked.hpp"
#include <algorithm>
#include <bit>
#include <cstdio>
#include <cstring>
#include <cstdlib>
#include <limits>
#include <set>
#ifdef _WIN32
#include <share.h>
#endif

namespace strata::weights {
using namespace detail;
namespace fs = std::filesystem;
namespace {
constexpr uint64_t kMaxHeader = 100000000, kMaxMetadata = 100000000;
constexpr size_t kStagingBytes = 8 * 1024 * 1024;
struct File {
    FILE* f = nullptr;
    uint64_t size = 0;
    explicit File(const fs::path& path) {
#ifdef _WIN32
        f = _wfsopen(path.c_str(), L"rb", _SH_DENYWR);
#else
        f = std::fopen(path.c_str(), "rb");
#endif
        require(f != nullptr, "cannot open read-only model file");
        std::setvbuf(f, nullptr, _IONBF, 0); // no hidden stdio read-ahead
#ifdef _WIN32
        const int seek = _fseeki64(f, 0, SEEK_END);
        const auto end = _ftelli64(f);
#else
        const int seek = fseeko(f, 0, SEEK_END);
        const auto end = ftello(f);
#endif
        if (seek != 0 || end < 0) { std::fclose(f); f = nullptr; throw std::runtime_error("cannot size file"); }
        size = static_cast<uint64_t>(end);
    }
    ~File() { if (f) std::fclose(f); }
    File(const File&) = delete;
    void read(uint64_t offset, std::span<uint8_t> dst) {
        require(offset <= size && dst.size() <= size - offset, "file read out of bounds");
        require(offset <= INT64_MAX, "file offset exceeds signed 64-bit seek");
#ifdef _WIN32
        const int rc = _fseeki64(f, static_cast<int64_t>(offset), SEEK_SET);
#else
        const int rc = fseeko(f, static_cast<off_t>(offset), SEEK_SET);
#endif
        require(rc == 0, "seek failed");
        require(dst.empty() || std::fread(dst.data(), 1, dst.size(), f) == dst.size(), "short file read");
    }
};
fs::path contained_file(const fs::path& root, const std::string& name) {
    // Basenames only. Refuse Windows drive/ADS/UNC syntax on every platform.
    require(!name.empty() && name != "." && name != ".." &&
            name.find_first_of("/\\:") == std::string::npos && name.find('\0') == std::string::npos &&
            name.back() != '.' && name.back() != ' ', "unsafe model filename: " + name);
    const auto resolved = fs::canonical(root / utf8_path(name));
    require(resolved.parent_path() == root, "model file escapes root (symlink/junction): " + name);
    require(fs::is_regular_file(resolved), "model source is not a regular file");
    return resolved;
}
DType dtype(const std::string& s) {
    for (int i = 0; i <= static_cast<int>(DType::F8_E5M2); ++i) {
        const auto d = static_cast<DType>(i);
        if (s == dtype_name(d)) return d;
    }
    throw std::runtime_error("unsupported safetensors dtype: " + s);
}
std::string read_text(File& f, uint64_t offset, uint64_t size) {
    std::string s(static_cast<size_t>(size), '\0');
    f.read(offset, {reinterpret_cast<uint8_t*>(s.data()), s.size()});
    return s;
}
} // namespace
const char* dtype_name(DType d) {
    constexpr const char* names[] = {"U8","I8","BOOL","I16","U16","I32","U32","I64","U64",
                                   "F16","BF16","F32","F64","F8_E4M3","F8_E5M2"};
    return names[static_cast<size_t>(d)];
}
uint64_t dtype_bytes(DType d) {
    constexpr uint64_t sizes[] = {1,1,1,2,2,4,4,8,8,2,2,4,8,1,1};
    return sizes[static_cast<size_t>(d)];
}
const char* family_name(Family f) {
    constexpr const char* names[] = {"dense", "expert", "ngram", "mtp", "vision"};
    return names[static_cast<size_t>(f)];
}
Family weight_family(const std::string& n) {
    if (n.starts_with("mtp.") || n.find(".mtp.") != n.npos) return Family::Mtp;
    if (n.find(".experts.") != n.npos) return Family::Expert;
    if (n.find("ngram") != n.npos) return Family::Ngram;
    if (n.starts_with("model.visual.")) return Family::Vision;
    return Family::Dense;
}
std::vector<uint8_t> WeightSource::read(const TensorDesc& t) {
    require(t.bytes <= std::numeric_limits<size_t>::max(), "tensor too large for address space");
    std::vector<uint8_t> bytes(static_cast<size_t>(t.bytes));
    const ReadRequest r{&t, 0, bytes};
    read_many({&r, 1});
    return bytes;
}
struct SafetensorsSource::Impl {
    fs::path root;
    std::map<std::string, TensorDesc> tensors;
    std::vector<ShardDesc> shards;
    std::vector<std::unique_ptr<File>> files;
    IoStats stats;
    bool experts_sealed = false;
    bool resident_sealed = false;
    std::string metadata(const std::string& name) {
        File f(contained_file(root, name));
        require(f.size <= kMaxMetadata, "metadata JSON exceeds 100 MB");
        stats.metadata_bytes += f.size;
        return read_text(f, 0, f.size);
    }
};
SafetensorsSource::SafetensorsSource(const fs::path& dir) : impl_(std::make_unique<Impl>()) {
    static_assert(std::endian::native == std::endian::little, "little-endian host required");
    auto& p = *impl_;
    p.root = fs::canonical(dir);
    const bool trace = std::getenv("STRATA_SAFETENSORS_TRACE") != nullptr;
    if (trace) std::fprintf(stderr, "safetensors: reading index\n");
    const auto index = parse(p.metadata("model.safetensors.index.json"));
    if (trace) std::fprintf(stderr, "safetensors: validating index\n");
    require(index.is_object() && index.contains("weight_map") && index.at("weight_map").is_object(),
            "index requires weight_map object");
    const auto& map = index.at("weight_map");
    require(!map.empty() && map.size() <= 1000000, "index tensor count out of range");
    std::set<std::string> names;
    for (const auto& [key, value] : map.items()) {
        require(!key.empty() && key != "__metadata__" && value.is_string(), "invalid weight_map entry");
        const auto name = value.get<std::string>();
        require(name.ends_with(".safetensors"), "index target must be a safetensors shard");
        names.insert(name);
    }
    // Check paths once per file, not once per tensor (~300,000 tensors).
    for (const auto& name : names) contained_file(p.root, name);
    for (const auto& e : fs::directory_iterator(p.root)) {
        if (e.path().extension() == ".safetensors") {
            const auto utf8 = e.path().filename().u8string();
            require(names.contains(std::string(utf8.begin(), utf8.end())), "unindexed safetensors file");
        }
    }
    for (const auto& name : names) {
        if (trace) std::fprintf(stderr, "safetensors: reading header %s\n", name.c_str());
        ShardDesc s;
        s.path = contained_file(p.root, name);
        auto file = std::make_unique<File>(s.path);
        s.file_bytes = file->size;
        require(s.file_bytes >= 8, "truncated safetensors length");
        std::array<uint8_t, 8> length{};
        file->read(0, length);
        std::memcpy(&s.header_bytes, length.data(), 8);
        require(s.header_bytes > 0 && s.header_bytes <= kMaxHeader && s.header_bytes <= s.file_bytes - 8,
                "invalid safetensors header length");
        s.data_offset = 8 + s.header_bytes;
        const auto text = read_text(*file, 8, s.header_bytes);
        require(text.front() == '{', "safetensors header must begin with object");
        const auto header = parse(text);
        require(header.is_object(), "header must be object");
        p.stats.header_bytes += s.data_offset;
        std::vector<std::pair<uint64_t, uint64_t>> ranges;
        for (const auto& [key, v] : header.items()) {
            if (key == "__metadata__") {
                require(v.is_object(), "__metadata__ must be string map");
                for (const auto& value : v) require(value.is_string(), "non-string __metadata__ value");
                continue;
            }
            require(map.contains(key) && map.at(key) == name, "header/index mismatch: " + key);
            require(v.is_object() && v.size() == 3 && v.contains("dtype") &&
                    v.contains("shape") && v.contains("data_offsets"), "invalid tensor descriptor: " + key);
            TensorDesc t;
            t.name = key;
            t.dtype = dtype(v.at("dtype").get<std::string>());
            t.family = weight_family(key);
            t.file_id = p.files.size();
            const auto& shape = v.at("shape");
            require(shape.is_array() && shape.size() <= 16, "invalid tensor rank");
            uint64_t elements = 1;
            for (const auto& dim : shape) {
                const auto d = uint64(dim);
                t.physical_shape.push_back(d);
                elements = mul(elements, d);
            }
            const auto& offsets = v.at("data_offsets");
            require(offsets.is_array() && offsets.size() == 2, "invalid data_offsets");
            const auto begin = uint64(offsets[0]), end = uint64(offsets[1]);
            require(begin <= end && end <= s.file_bytes - s.data_offset, "tensor offsets out of bounds");
            t.offset = add(s.data_offset, begin);
            t.bytes = end - begin;
            require(t.bytes == mul(elements, dtype_bytes(t.dtype)), "dtype/shape/byte mismatch: " + key);
            require(p.tensors.emplace(key, std::move(t)).second, "tensor repeated across shards");
            ranges.emplace_back(begin, end);
            ++s.tensor_count;
        }
        std::sort(ranges.begin(), ranges.end());
        uint64_t end = 0;
        for (const auto& r : ranges) {
            require(r.first == end, "tensor storage contains overlap or hole");
            end = r.second;
        }
        require(end == s.file_bytes - s.data_offset, "unindexed trailing data in shard");
        p.shards.push_back(s);
        p.files.push_back(std::move(file));
    }
    require(p.tensors.size() == map.size(), "index contains missing tensors");
    if (trace) std::fprintf(stderr, "safetensors: all headers validated\n");
}
SafetensorsSource::~SafetensorsSource() = default;
const TensorDesc& SafetensorsSource::tensor(const std::string& n) const {
    const auto it = impl_->tensors.find(n);
    require(it != impl_->tensors.end(), "missing tensor: " + n);
    return it->second;
}
const std::map<std::string, TensorDesc>& SafetensorsSource::tensors() const { return impl_->tensors; }
const std::vector<ShardDesc>& SafetensorsSource::shards() const { return impl_->shards; }
const IoStats& SafetensorsSource::io_stats() const { return impl_->stats; }
void SafetensorsSource::seal_expert_reads() { impl_->experts_sealed = true; }
void SafetensorsSource::seal_resident_reads() {
    require(impl_->experts_sealed, "expert residency must be sealed before releasing startup files");
    impl_->resident_sealed = true;
    impl_->files.clear();
}
std::string SafetensorsSource::metadata_text(const std::string& name) { return impl_->metadata(name); }
void SafetensorsSource::read_many(std::span<const ReadRequest> requests) {
    auto& p = *impl_;
    std::vector<const ReadRequest*> ordered;
    for (const auto& r : requests) {
        require(!p.resident_sealed, "resident weight source reads sealed after startup");
        require(r.tensor != nullptr, "null read descriptor");
        const auto& t = tensor(r.tensor->name);
        require(&t == r.tensor, "read descriptor does not belong to this source");
        require(!(p.experts_sealed && (t.family == Family::Expert || t.family == Family::Mtp)), "expert source reads sealed after residency");
        require(r.relative_offset <= t.bytes && r.destination.size() <= t.bytes - r.relative_offset,
                "tensor read out of bounds");
        if (!r.destination.empty()) ordered.push_back(&r);
    }
    std::sort(ordered.begin(), ordered.end(), [](auto a, auto b) {
        if (a->tensor->file_id != b->tensor->file_id) return a->tensor->file_id < b->tensor->file_id;
        return a->tensor->offset + a->relative_offset < b->tensor->offset + b->relative_offset;
    });
    std::vector<uint8_t> staging;
    for (size_t i = 0; i < ordered.size();) {
        const auto* first = ordered[i];
        const auto file = first->tensor->file_id;
        const auto family = first->tensor->family;
        const uint64_t begin = first->tensor->offset + first->relative_offset;
        uint64_t end = begin + first->destination.size();
        size_t j = i + 1;
        // Adjacent ranges of one family only; never read gaps through ngram data.
        while (j < ordered.size()) {
            const auto* r = ordered[j];
            const uint64_t rb = r->tensor->offset + r->relative_offset;
            if (r->tensor->file_id != file || r->tensor->family != family || rb != end ||
                end - begin + r->destination.size() > kStagingBytes) break;
            end += r->destination.size();
            ++j;
        }
        if (j == i + 1) {
            auto dst = first->destination;
            for (size_t at = 0; at < dst.size();) {
                const auto n = std::min(kStagingBytes, dst.size() - at);
                p.files[file]->read(begin + at, dst.subspan(at, n));
                ++p.stats.data_calls;
                p.stats.data_bytes[static_cast<size_t>(family)] += n;
                at += n;
            }
        } else {
            staging.resize(static_cast<size_t>(end - begin));
            p.stats.staging_peak_bytes = std::max<uint64_t>(p.stats.staging_peak_bytes, staging.size());
            p.files[file]->read(begin, staging);
            ++p.stats.data_calls;
            p.stats.data_bytes[static_cast<size_t>(family)] += staging.size();
            for (size_t k = i; k < j; ++k) {
                const auto* r = ordered[k];
                const auto at = r->tensor->offset + r->relative_offset - begin;
                std::memcpy(r->destination.data(), staging.data() + at, r->destination.size());
            }
        }
        i = j;
    }
}
} // namespace strata::weights
