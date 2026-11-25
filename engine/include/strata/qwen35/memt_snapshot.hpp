// include/strata/qwen35/memt_snapshot.hpp - the `memt-memory-v1` safetensors snapshot.
#pragma once

#include <algorithm>
#include <cerrno>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <map>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#if defined(_WIN32)
#include <io.h>
#else
#include <fcntl.h>
#include <sys/stat.h>
#include <unistd.h>
#endif

namespace strata::qwen35::memt {

inline constexpr const char* kSnapshotFormat = "memt-memory-v1";
inline constexpr std::size_t kSnapshotMaxHeader = std::size_t(1) << 30;

class SnapshotError : public std::runtime_error {
public:
    explicit SnapshotError(const std::string& what) : std::runtime_error(what) {}
};

/// The whole file as bytes.
using Snapshot = std::string;

struct SnapshotLayer {
    int layer = 0;
    std::vector<float> values;   // [1, heads, dim, dim] (the record's, already validated)
};

// ------------------------------------------------------------------------------------------------ JSON
namespace memt_detail {

struct Json {
    enum class Kind { Null, Bool, Int, Real, Str, Array, Object };
    Kind kind = Kind::Null;
    bool boolean = false;
    int64_t integer = 0;
    double real = 0.0;
    std::string text;
    std::vector<Json> items;
    std::map<std::string, Json> fields;   // std::map: iteration is the sorted order json.dumps writes

    bool is_int() const { return kind == Kind::Int; }
    bool is_str() const { return kind == Kind::Str; }
    bool is_array() const { return kind == Kind::Array; }
    bool is_object() const { return kind == Kind::Object; }

    const Json* get(const std::string& key) const {
        if (kind != Kind::Object) return nullptr;
        const auto it = fields.find(key);
        return it == fields.end() ? nullptr : &it->second;
    }
    const Json& at(const std::string& key) const {
        const Json* value = get(key);
        if (value == nullptr) throw SnapshotError("snapshot JSON is missing " + key);
        return *value;
    }
};

class JsonParser {
public:
    explicit JsonParser(const std::string& text) : text_(text) {}

    Json parse() {
        Json value = value_at(0);
        skip();
        if (pos_ != text_.size()) fail("trailing characters");
        return value;
    }

private:
    const std::string& text_;
    std::size_t pos_ = 0;

    [[noreturn]] void fail(const char* what) const {
        throw SnapshotError(std::string("invalid snapshot header: ") + what + " at byte " + std::to_string(pos_));
    }
    void skip() {
        while (pos_ < text_.size() && (text_[pos_] == ' ' || text_[pos_] == '\t' || text_[pos_] == '\n' ||
                                       text_[pos_] == '\r'))
            ++pos_;
    }
    char take() {
        if (pos_ >= text_.size()) fail("unexpected end");
        return text_[pos_++];
    }
    void expect(char c) {
        if (take() != c) fail("unexpected character");
    }
    bool starts(const char* word) const {
        const std::size_t n = std::strlen(word);
        return text_.compare(pos_, n, word) == 0;
    }
    void literal(const char* word) {
        if (!starts(word)) fail("unexpected literal");
        pos_ += std::strlen(word);
    }

    Json value_at(int depth) {
        if (depth > 8) fail("unexpected nesting");
        skip();
        if (pos_ >= text_.size()) fail("unexpected end");
        const char c = text_[pos_];
        if (c == '{') return object(depth);
        if (c == '[') return array(depth);
        if (c == '"') {
            Json value;
            value.kind = Json::Kind::Str;
            value.text = string();
            return value;
        }
        if (c == 't' || c == 'f') {
            Json value;
            value.kind = Json::Kind::Bool;
            if (c == 't') { literal("true"); value.boolean = true; }
            else { literal("false"); }
            return value;
        }
        if (c == 'n') {
            literal("null");
            return Json{};
        }
        return number();
    }

    Json object(int depth) {
        Json value;
        value.kind = Json::Kind::Object;
        expect('{');
        skip();
        if (pos_ < text_.size() && text_[pos_] == '}') { ++pos_; return value; }
        for (;;) {
            skip();
            std::string key = string();
            skip();
            expect(':');
            if (!value.fields.emplace(std::move(key), value_at(depth + 1)).second)
                fail("duplicate key");
            skip();
            const char c = take();
            if (c == '}') break;
            if (c != ',') fail("expected , or }");
        }
        return value;
    }

    Json array(int depth) {
        Json value;
        value.kind = Json::Kind::Array;
        expect('[');
        skip();
        if (pos_ < text_.size() && text_[pos_] == ']') { ++pos_; return value; }
        for (;;) {
            value.items.push_back(value_at(depth + 1));
            skip();
            const char c = take();
            if (c == ']') break;
            if (c != ',') fail("expected , or ]");
        }
        return value;
    }

    static int hex(char c) {
        if (c >= '0' && c <= '9') return c - '0';
        if (c >= 'a' && c <= 'f') return c - 'a' + 10;
        if (c >= 'A' && c <= 'F') return c - 'A' + 10;
        return -1;
    }

    static void put_utf8(std::string& out, unsigned cp) {
        if (cp < 0x80) {
            out.push_back((char) cp);
        } else if (cp < 0x800) {
            out.push_back((char) (0xc0 | (cp >> 6)));
            out.push_back((char) (0x80 | (cp & 0x3f)));
        } else {
            out.push_back((char) (0xe0 | (cp >> 12)));
            out.push_back((char) (0x80 | ((cp >> 6) & 0x3f)));
            out.push_back((char) (0x80 | (cp & 0x3f)));
        }
    }

    std::string string() {
        expect('"');
        std::string out;
        for (;;) {
            const char c = take();
            if (c == '"') return out;
            if ((unsigned char) c < 0x20) fail("control character in string");
            if (c != '\\') { out.push_back(c); continue; }
            const char e = take();
            switch (e) {
            case '"': out.push_back('"'); break;
            case '\\': out.push_back('\\'); break;
            case '/': out.push_back('/'); break;
            case 'b': out.push_back('\b'); break;
            case 'f': out.push_back('\f'); break;
            case 'n': out.push_back('\n'); break;
            case 'r': out.push_back('\r'); break;
            case 't': out.push_back('\t'); break;
            case 'u': {
                unsigned cp = 0;
                for (int i = 0; i < 4; ++i) {
                    const int h = hex(take());
                    if (h < 0) fail("bad \\u escape");
                    cp = (cp << 4) | (unsigned) h;
                }
                put_utf8(out, cp);
                break;
            }
            default: fail("bad escape");
            }
        }
    }

    Json number() {
        const std::size_t begin = pos_;
        if (pos_ < text_.size() && text_[pos_] == '-') ++pos_;
        bool digits = false;
        while (pos_ < text_.size() && text_[pos_] >= '0' && text_[pos_] <= '9') { ++pos_; digits = true; }
        bool integral = digits;
        if (pos_ < text_.size() && text_[pos_] == '.') {
            integral = false;
            ++pos_;
            while (pos_ < text_.size() && text_[pos_] >= '0' && text_[pos_] <= '9') ++pos_;
        }
        if (pos_ < text_.size() && (text_[pos_] == 'e' || text_[pos_] == 'E')) {
            integral = false;
            ++pos_;
            if (pos_ < text_.size() && (text_[pos_] == '+' || text_[pos_] == '-')) ++pos_;
            while (pos_ < text_.size() && text_[pos_] >= '0' && text_[pos_] <= '9') ++pos_;
        }
        if (!digits) fail("expected a number");
        const std::string raw = text_.substr(begin, pos_ - begin);
        Json value;
        if (integral) {
            value.kind = Json::Kind::Int;
            value.integer = std::strtoll(raw.c_str(), nullptr, 10);
        } else {
            value.kind = Json::Kind::Real;
            value.real = std::strtod(raw.c_str(), nullptr);
        }
        return value;
    }
};

inline std::string quote(const std::string& value) {
    std::string out = "\"";
    for (const char c : value) {
        switch (c) {
        case '"': out += "\\\""; break;
        case '\\': out += "\\\\"; break;
        case '\n': out += "\\n"; break;
        case '\r': out += "\\r"; break;
        case '\t': out += "\\t"; break;
        default:
            if ((unsigned char) c < 0x20) {
                char buf[8];
                std::snprintf(buf, sizeof buf, "\\u%04x", (unsigned) (unsigned char) c);
                out += buf;
            } else {
                out.push_back(c);
            }
        }
    }
    out.push_back('"');
    return out;
}

}  // namespace memt_detail

// ------------------------------------------------------------------------------------------------ encode / decode
/// The safetensors file as bytes.  `layers` and `data` are parallel (layer -> its
/// [1, heads, dim, dim] FP32 state on the host) and may be in any order.
inline Snapshot encode_snapshot(const std::string& identity, const std::vector<int>& layers,
                                const std::vector<const float*>& data, int64_t heads, int64_t dim) {
    if (identity.empty()) throw SnapshotError("snapshot identity must be nonempty");
    if (layers.size() != data.size()) throw SnapshotError("one state per layer is required");
    if (heads <= 0 || dim <= 0) throw SnapshotError("snapshot shapes must be positive");
    std::vector<int> order(layers.size());
    for (std::size_t i = 0; i < order.size(); ++i) order[i] = (int) i;
    std::sort(order.begin(), order.end(), [&](int a, int b) { return layers[(std::size_t) a] < layers[(std::size_t) b]; });
    for (std::size_t i = 1; i < order.size(); ++i)
        if (layers[(std::size_t) order[i]] == layers[(std::size_t) order[i - 1]])
            throw SnapshotError("duplicate memory layer " + std::to_string(layers[(std::size_t) order[i]]));
    const std::size_t count = (std::size_t) heads * (std::size_t) dim * (std::size_t) dim;
    for (std::size_t i = 0; i < data.size(); ++i) {
        if (data[i] == nullptr) throw SnapshotError("null memory state");
        for (std::size_t j = 0; j < count; ++j)
            if (!std::isfinite(data[i][j]))
                throw SnapshotError("memory." + std::to_string(layers[i]) + " contains non-finite values");
    }

    // The header json.dumps(sort_keys=True) writes: keys sorted as strings, so
    // memory.11 precedes memory.3; the data offsets follow the numeric order.
    std::vector<std::pair<std::string, std::size_t>> header_keys;   // (name, index in `order`)
    for (std::size_t k = 0; k < order.size(); ++k)
        header_keys.emplace_back("memory." + std::to_string(layers[(std::size_t) order[k]]), k);
    std::sort(header_keys.begin(), header_keys.end(),
              [](const std::pair<std::string, std::size_t>& a, const std::pair<std::string, std::size_t>& b) {
                  return a.first < b.first;
              });
    std::map<std::string, std::pair<std::size_t, std::size_t>> offsets;   // name -> [begin, end)
    std::size_t cursor = 0;
    for (std::size_t k = 0; k < order.size(); ++k) {
        const std::string name = "memory." + std::to_string(layers[(std::size_t) order[k]]);
        offsets[name] = {cursor, cursor + count * sizeof(float)};
        cursor += count * sizeof(float);
    }

    std::string header = "{\"__metadata__\": {\"format\": " + memt_detail::quote(kSnapshotFormat) +
                         ", \"identity\": " + memt_detail::quote(identity) + "}";
    for (const auto& key : header_keys) {
        const auto it = offsets.find(key.first);
        header += ", " + memt_detail::quote(key.first) + ": {\"data_offsets\": [" + std::to_string(it->second.first) +
                  ", " + std::to_string(it->second.second) + "], \"dtype\": \"F32\", \"shape\": [1, " +
                  std::to_string((long long) heads) + ", " + std::to_string((long long) dim) + ", " +
                  std::to_string((long long) dim) + "]}";
    }
    header += "}";
    header.append((8 - header.size() % 8) % 8, ' ');   // safetensors pads the header to 8 bytes

    Snapshot out;
    out.reserve(8 + header.size() + cursor);
    const uint64_t size = (uint64_t) header.size();
    out.append(reinterpret_cast<const char*>(&size), 8);
    out += header;
    for (const int index : order)
        out.append(reinterpret_cast<const char*>(data[(std::size_t) index]), count * sizeof(float));
    return out;
}

/// States `blob` holds, ascending by layer, validated against `identity` and the
/// geometry.  Refuses a different layer set, another dtype/shape, or non-finite
/// values: the same checks tools/memt_snapshot.py makes.
inline std::vector<SnapshotLayer> decode_snapshot(const Snapshot& blob, const std::string& identity, int64_t heads,
                                                   int64_t dim) {
    if (blob.size() < 9) throw SnapshotError("snapshot is shorter than a safetensors header");
    uint64_t header_size = 0;
    std::memcpy(&header_size, blob.data(), 8);
    if (header_size <= 1 || header_size > kSnapshotMaxHeader || 8 + header_size > blob.size())
        throw SnapshotError("invalid safetensors header size");
    const std::string header_text = blob.substr(8, (std::size_t) header_size);
    memt_detail::Json header = memt_detail::JsonParser(header_text).parse();
    if (!header.is_object()) throw SnapshotError("snapshot header is not an object");
    const memt_detail::Json* metadata = header.get("__metadata__");
    if (metadata == nullptr || !metadata->is_object()) throw SnapshotError("snapshot has no __metadata__");
    const memt_detail::Json* format = metadata->get("format");
    const memt_detail::Json* got_identity = metadata->get("identity");
    if (format == nullptr || !format->is_str() || format->text != kSnapshotFormat)
        throw SnapshotError("snapshot format is not " + std::string(kSnapshotFormat));
    if (got_identity == nullptr || !got_identity->is_str() || got_identity->text != identity)
        throw SnapshotError("snapshot identity mismatch");
    if (metadata->fields.size() != 2) throw SnapshotError("snapshot metadata has unexpected keys");

    const std::size_t start = 8 + (std::size_t) header_size;
    const std::size_t data_bytes = blob.size() - start;
    const std::size_t count = (std::size_t) heads * (std::size_t) dim * (std::size_t) dim;
    struct LayerRange { int layer; std::size_t begin, end; };
    std::vector<LayerRange> ranges;
    for (const auto& field : header.fields) {
        const std::string& name = field.first;
        const memt_detail::Json& info = field.second;
        if (name == "__metadata__") continue;
        if (name.rfind("memory.", 0) != 0 || name.size() == 7) throw SnapshotError("unexpected snapshot tensor " + name);
        for (std::size_t i = 7; i < name.size(); ++i)
            if (name[i] < '0' || name[i] > '9') throw SnapshotError("unexpected snapshot tensor " + name);
        char* name_end = nullptr;
        const long long parsed = std::strtoll(name.c_str() + 7, &name_end, 10);
        if (name_end == name.c_str() + 7 || *name_end != '\0' || parsed < 0 || parsed > 0x7fffffffLL)
            throw SnapshotError("memory layer index out of range in " + name);
        const int layer = (int) parsed;
        const memt_detail::Json* dtype = info.get("dtype");
        if (dtype == nullptr || !dtype->is_str() || dtype->text != "F32") throw SnapshotError(name + " must be F32");
        const memt_detail::Json* shape = info.get("shape");
        if (shape == nullptr || !shape->is_array() || shape->items.size() != 4)
            throw SnapshotError(name + " shape must be [1, H, D, D]");
        int64_t got[4];
        for (int i = 0; i < 4; ++i) {
            if (!shape->items[(std::size_t) i].is_int()) throw SnapshotError(name + " shape must be integers");
            got[i] = shape->items[(std::size_t) i].integer;
        }
        if (got[0] != 1 || got[1] != heads || got[2] != dim || got[3] != dim)
            throw SnapshotError(name + " has an unexpected shape");
        const memt_detail::Json* offsets = info.get("data_offsets");
        if (offsets == nullptr || !offsets->is_array() || offsets->items.size() != 2 ||
            !offsets->items[0].is_int() || !offsets->items[1].is_int())
            throw SnapshotError(name + " has no data_offsets");
        const int64_t begin = offsets->items[0].integer, end = offsets->items[1].integer;
        if (begin < 0 || end < begin || (uint64_t) end > data_bytes)
            throw SnapshotError(name + " has data outside the file");
        if ((uint64_t) (end - begin) != (uint64_t) count * 4u)
            throw SnapshotError(name + " has an unexpected byte length");
        for (const LayerRange& seen : ranges)
            if (seen.layer == layer) throw SnapshotError("duplicate " + name);
        ranges.push_back({layer, (std::size_t) begin, (std::size_t) end});
    }
    std::sort(ranges.begin(), ranges.end(), [](const LayerRange& a, const LayerRange& b) { return a.begin < b.begin; });
    std::size_t next_offset = 0;
    for (const LayerRange& range : ranges) {
        if (range.begin != next_offset) throw SnapshotError("snapshot tensor ranges overlap or leave a gap");
        next_offset = range.end;
    }
    if (next_offset != data_bytes) throw SnapshotError("unexpected data after the snapshot tensors");
    std::vector<SnapshotLayer> layers;
    layers.reserve(ranges.size());
    for (const LayerRange& range : ranges) {
        SnapshotLayer state;
        state.layer = range.layer;
        state.values.resize(count);
        std::memcpy(state.values.data(), blob.data() + start + range.begin, count * sizeof(float));
        for (const float value : state.values)
            if (!std::isfinite(value))
                throw SnapshotError("memory." + std::to_string(range.layer) + " contains non-finite values");
        layers.push_back(std::move(state));
    }
    std::sort(layers.begin(), layers.end(),
              [](const SnapshotLayer& a, const SnapshotLayer& b) { return a.layer < b.layer; });
    return layers;
}

// ------------------------------------------------------------------------------------------------ files
inline Snapshot load_snapshot(const std::string& path) {
    std::FILE* file = std::fopen(path.c_str(), "rb");
    if (file == nullptr) throw SnapshotError("cannot open " + path);
    Snapshot blob;
    char buffer[1 << 16];
    for (;;) {
        const std::size_t got = std::fread(buffer, 1, sizeof buffer, file);
        blob.append(buffer, got);
        if (got != sizeof buffer) break;
    }
    const bool failed = std::ferror(file) != 0;
    std::fclose(file);
    if (failed) throw SnapshotError("cannot read " + path);
    return blob;
}

inline void save_snapshot(const std::string& path, const Snapshot& blob, bool overwrite) {
    if (path.size() < 12 || path.compare(path.size() - 12, 12, ".safetensors") != 0)
        throw SnapshotError("snapshot names end in .safetensors");
#if !defined(_WIN32)
    {
        struct stat st {};
        if (lstat(path.c_str(), &st) == 0 && S_ISLNK(st.st_mode))
            throw SnapshotError("snapshot target must not be a symlink: " + path);
    }
#endif
    if (!overwrite) {
        std::FILE* existing = std::fopen(path.c_str(), "rb");
        if (existing != nullptr) {
            std::fclose(existing);
            throw SnapshotError("snapshot already exists: " + path);
        }
    }
    const std::string temporary = path + ".tmp." +
                                  std::to_string((long long) std::chrono::steady_clock::now().time_since_epoch().count());
#if defined(_WIN32)
    std::FILE* file = std::fopen(temporary.c_str(), "wbx");
#else
    const int fd = ::open(temporary.c_str(), O_WRONLY | O_CREAT | O_EXCL | O_CLOEXEC, 0600);
    std::FILE* file = fd < 0 ? nullptr : ::fdopen(fd, "wb");
    if (file == nullptr && fd >= 0) {
        ::close(fd);
        std::remove(temporary.c_str());
    }
#endif
    if (file == nullptr) throw SnapshotError("cannot write " + temporary);
    const std::size_t written = std::fwrite(blob.data(), 1, blob.size(), file);
    const bool flushed = std::fflush(file) == 0;
    bool synced = true;
#if defined(_WIN32)
    synced = _commit(_fileno(file)) == 0;
#else
    synced = fsync(fileno(file)) == 0;
#endif
    const bool closed = std::fclose(file) == 0;
    if (written != blob.size() || !flushed || !synced || !closed) {
        std::remove(temporary.c_str());
        throw SnapshotError("cannot write " + temporary);
    }
    if (overwrite) {
        std::error_code error;
        std::filesystem::rename(temporary, path, error);
        if (error) {
            std::remove(temporary.c_str());
            throw SnapshotError("cannot replace " + path);
        }
    } else {
#if defined(_WIN32)
        if (std::rename(temporary.c_str(), path.c_str()) != 0) {
            std::remove(temporary.c_str());
            throw SnapshotError("cannot create " + path);
        }
#else
        if (::link(temporary.c_str(), path.c_str()) != 0) {
            const int error = errno;
            std::remove(temporary.c_str());
            if (error == EEXIST) throw SnapshotError("snapshot already exists: " + path);
            throw SnapshotError("cannot create " + path);
        }
        std::remove(temporary.c_str());
#endif
    }
#if !defined(_WIN32)
    {   // the rename / link is durable only once its directory is synced
        const std::size_t slash = path.find_last_of('/');
        const std::string dir = slash == std::string::npos ? "." : path.substr(0, slash);
        const int fd = ::open(dir.c_str(), O_RDONLY);
        if (fd >= 0) {
            (void) fsync(fd);
            (void) close(fd);
        }
    }
#endif
}

}  // namespace strata::qwen35::memt
