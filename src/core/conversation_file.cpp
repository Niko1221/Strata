#include "strata/core/conversation_file.hpp"

#include <algorithm>
#include <cerrno>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <memory>
#include <string>
#include <system_error>
#ifdef _WIN32
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#include <malloc.h>
#else
#include <fcntl.h>
#include <unistd.h>
#endif

namespace strata::core {
namespace {
constexpr char kMagic[8] = {'S', 'T', 'R', 'S', 'E', 'S', 'S', '\x01'};
constexpr char kEnd[8] = {'S', 'T', 'R', 'S', 'E', 'N', 'D', '\x01'};
constexpr uint32_t kVersion = 1;
constexpr size_t kHeader = 64, kTrailer = 16;

constexpr uint64_t P1 = 0x9E3779B185EBCA87ull, P2 = 0xC2B2AE3D27D4EB4Full, P3 = 0x165667B19E3779F9ull,
                   P4 = 0x85EBCA77C2B2AE63ull, P5 = 0x27D4EB2F165667C5ull;
uint64_t rotl(uint64_t x, int r) { return (x << r) | (x >> (64 - r)); }
uint64_t round1(uint64_t acc, uint64_t w) { return rotl(acc + w * P2, 31) * P1; }
uint64_t load64(const uint8_t* p) { uint64_t v; std::memcpy(&v, p, 8); return v; }

struct FileCloser { void operator()(std::FILE* f) const { if (f) std::fclose(f); } };
using File = std::unique_ptr<std::FILE, FileCloser>;

// ---- file I/O in aligned 16 MiB blocks.  Linux: O_DIRECT when the filesystem takes it (no page cache, so a
// 1.2 GB session neither evicts other files nor is charged to the engine's memory cgroup), buffered I/O otherwise
// (tmpfs, or STRATA_SESSION_BUFFERED=1).  Windows and other systems: ordinary buffered I/O, same blocks.
constexpr size_t kAlign = 4096, kBlock = 16u << 20;
#ifdef _WIN32
struct AlignedFree { void operator()(uint8_t* p) const { _aligned_free(p); } };
using Aligned = std::unique_ptr<uint8_t, AlignedFree>;
Aligned aligned_block() { return Aligned(static_cast<uint8_t*>(_aligned_malloc(kBlock, kAlign))); }
#else
struct AlignedFree { void operator()(uint8_t* p) const { std::free(p); } };
using Aligned = std::unique_ptr<uint8_t, AlignedFree>;
Aligned aligned_block() { return Aligned(static_cast<uint8_t*>(std::aligned_alloc(kAlign, kBlock))); }
#endif

#ifdef _WIN32
// paths reach the engine as UTF-8 (the server writes its commands as UTF-8)
std::wstring wide(const std::string& s) {
    const int n = MultiByteToWideChar(CP_UTF8, 0, s.data(), (int) s.size(), nullptr, 0);
    if (n <= 0) return {};
    std::wstring w((size_t) n, L'\0');
    MultiByteToWideChar(CP_UTF8, 0, s.data(), (int) s.size(), w.data(), n);
    return w;
}

std::string last_error() {
    const DWORD code = GetLastError();
    char msg[256] = {};
    const DWORD n = FormatMessageA(FORMAT_MESSAGE_FROM_SYSTEM | FORMAT_MESSAGE_IGNORE_INSERTS, nullptr, code, 0, msg,
                                   sizeof msg, nullptr);
    std::string s(msg, n);
    while (!s.empty() && (s.back() == '\n' || s.back() == '\r' || s.back() == ' ' || s.back() == '.')) s.pop_back();
    return s.empty() ? "error " + std::to_string(code) : s;
}

class RawFile {
public:
    ~RawFile() { close(); }
    bool open_write(const std::string& path) {
        return open(path, GENERIC_WRITE, 0, CREATE_ALWAYS);
    }
    bool open_read(const std::string& path) {
        return open(path, GENERIC_READ, FILE_SHARE_READ, OPEN_EXISTING);
    }
    bool is_open() const { return h_ != INVALID_HANDLE_VALUE; }
    bool direct() const { return false; }
    bool write_all(const uint8_t* p, size_t n) {
        while (n) {
            DWORD w = 0;
            const DWORD want = (DWORD) std::min<size_t>(n, size_t(1) << 30);
            if (!WriteFile(h_, p, want, &w, nullptr) || w == 0) { error_ = last_error(); return false; }
            p += w; n -= w;
        }
        return true;
    }
    // bytes read, 0 at the end of the file, -1 on error
    long long read_some(uint8_t* p, size_t n) {
        DWORD r = 0;
        if (!ReadFile(h_, p, (DWORD) std::min<size_t>(n, size_t(1) << 30), &r, nullptr)) {
            error_ = last_error();
            return -1;
        }
        return (long long) r;
    }
    bool truncate(uint64_t n) {
        LARGE_INTEGER at{};
        at.QuadPart = (LONGLONG) n;
        if (SetFilePointerEx(h_, at, nullptr, FILE_BEGIN) && SetEndOfFile(h_)) return true;
        error_ = last_error();
        return false;
    }
    bool close() {
        if (h_ == INVALID_HANDLE_VALUE) return true;
        const bool ok = CloseHandle(h_) != 0;
        if (!ok) error_ = last_error();
        h_ = INVALID_HANDLE_VALUE;
        return ok;
    }
    const std::string& error() const { return error_; }
private:
    bool open(const std::string& path, DWORD access, DWORD share, DWORD disposition) {
        h_ = CreateFileW(wide(path).c_str(), access, share, nullptr, disposition,
                         FILE_ATTRIBUTE_NORMAL | FILE_FLAG_SEQUENTIAL_SCAN, nullptr);
        if (h_ == INVALID_HANDLE_VALUE) error_ = last_error();
        return h_ != INVALID_HANDLE_VALUE;
    }
    HANDLE h_ = INVALID_HANDLE_VALUE;
    std::string error_;
};

// std::rename does not replace an existing file on Windows: MoveFileExW does, atomically on one volume
bool replace_file(const std::string& from, const std::string& to, std::string& error) {
    if (MoveFileExW(wide(from).c_str(), wide(to).c_str(), MOVEFILE_REPLACE_EXISTING | MOVEFILE_WRITE_THROUGH))
        return true;
    error = last_error();
    return false;
}
void remove_file(const std::string& path) { DeleteFileW(wide(path).c_str()); }
std::filesystem::path fs_path(const std::string& path) { return std::filesystem::path(wide(path)); }
#else
#ifndef O_DIRECT
#define O_DIRECT 0            // not offered by this system: buffered I/O
#endif

bool buffered_forced() {
    const char* e = std::getenv("STRATA_SESSION_BUFFERED");
    return e != nullptr && e[0] == '1';
}

class RawFile {
public:
    ~RawFile() { close(); }
    bool open_write(const std::string& path) { return open(path, O_WRONLY | O_CREAT | O_TRUNC); }
    bool open_read(const std::string& path) { return open(path, O_RDONLY); }
    bool is_open() const { return fd_ >= 0; }
    bool direct() const { return direct_; }
    bool write_all(const uint8_t* p, size_t n) {
        while (n) {
            const ssize_t w = ::write(fd_, p, n);
            if (w < 0 && errno == EINTR) continue;
            if (w <= 0) { error_ = w < 0 ? std::strerror(errno) : "short write"; return false; }
            p += w; n -= (size_t) w;
        }
        return true;
    }
    // bytes read, 0 at the end of the file, -1 on error
    long long read_some(uint8_t* p, size_t n) {
        for (;;) {
            const ssize_t r = ::read(fd_, p, n);
            if (r < 0 && errno == EINTR) continue;
            if (r < 0) error_ = std::strerror(errno);
            return (long long) r;
        }
    }
    bool truncate(uint64_t n) {
        if (::ftruncate(fd_, (off_t) n) == 0) return true;
        error_ = std::strerror(errno);
        return false;
    }
    bool close() {
        if (fd_ < 0) return true;
        const bool ok = ::close(fd_) == 0;
        if (!ok) error_ = std::strerror(errno);
        fd_ = -1;
        return ok;
    }
    const std::string& error() const { return error_; }
private:
    bool open(const std::string& path, int flags) {
        const bool buffered = O_DIRECT == 0 || buffered_forced();
        fd_ = buffered ? -1 : ::open(path.c_str(), flags | O_DIRECT | O_CLOEXEC, 0644);
        direct_ = fd_ >= 0;
        // EINVAL: this filesystem refuses O_DIRECT (tmpfs): buffered I/O instead
        if (fd_ < 0 && (buffered || errno == EINVAL)) fd_ = ::open(path.c_str(), flags | O_CLOEXEC, 0644);
        if (fd_ < 0) error_ = std::strerror(errno);
        return fd_ >= 0;
    }
    int fd_ = -1;
    bool direct_ = false;
    std::string error_;
};

bool replace_file(const std::string& from, const std::string& to, std::string& error) {
    if (std::rename(from.c_str(), to.c_str()) == 0) return true;
    error = std::strerror(errno);
    return false;
}
void remove_file(const std::string& path) { std::remove(path.c_str()); }
std::filesystem::path fs_path(const std::string& path) { return std::filesystem::path(path); }
#endif

class FileSink {
public:
    explicit FileSink(const std::string& path) : buf_(aligned_block()) {
        if (buf_) f_.open_write(path);
    }
    bool is_open() const { return f_.is_open(); }
    std::string error() const { return buf_ ? f_.error() : std::string("out of memory for the write buffer"); }
    // room in the block for in-place production (avoids a staging copy)
    uint8_t* span(size_t& room) { room = kBlock - used_; return buf_.get() + used_; }
    bool commit(size_t n) { used_ += n; total_ += n; return used_ < kBlock || drain(); }
    bool write(const void* p, size_t n) {
        const uint8_t* s = static_cast<const uint8_t*>(p);
        while (n) {
            size_t room = 0;
            uint8_t* d = span(room);
            const size_t c = std::min(room, n);
            std::memcpy(d, s, c);
            if (!commit(c)) return false;
            s += c; n -= c;
        }
        return true;
    }
    // last block padded to the alignment for direct I/O, then the file cut back to its exact size
    bool finish() {
        if (!f_.is_open()) return false;
        const size_t padded = f_.direct() ? (used_ + kAlign - 1) / kAlign * kAlign : used_;
        std::memset(buf_.get() + used_, 0, padded - used_);
        bool ok = f_.write_all(buf_.get(), padded);
        ok = ok && (padded == used_ || f_.truncate(total_));
        ok = f_.close() && ok;
        return ok;
    }
private:
    bool drain() {
        if (!f_.write_all(buf_.get(), used_)) return false;
        used_ = 0;
        return true;
    }
    Aligned buf_;
    RawFile f_;
    size_t used_ = 0;
    uint64_t total_ = 0;
};

class FileSource {
public:
    explicit FileSource(const std::string& path) : buf_(aligned_block()) {
        if (buf_) f_.open_read(path);
    }
    bool is_open() const { return f_.is_open(); }
    std::string error() const { return buf_ ? f_.error() : std::string("out of memory for the read buffer"); }
    bool read(void* p, size_t n) {
        uint8_t* d = static_cast<uint8_t*>(p);
        while (n) {
            if (at_ == have_ && !fill()) return false;
            const size_t c = std::min(n, have_ - at_);
            std::memcpy(d, buf_.get() + at_, c);
            at_ += c; d += c; n -= c;
        }
        return true;
    }
private:
    bool fill() {
        at_ = have_ = 0;
        while (have_ < kBlock) {
            const long long r = f_.read_some(buf_.get() + have_, kBlock - have_);
            if (r < 0) return false;
            if (r == 0) break;
            have_ += (size_t) r;
            if (f_.direct() && have_ % kAlign) break;   // short aligned read: end of file
        }
        return have_ > 0;
    }
    Aligned buf_;
    RawFile f_;
    size_t at_ = 0, have_ = 0;
};

// ---- serialization: the same walk counts (f == nullptr) and writes
struct Out {
    FileSink* f = nullptr;
    SessionHasher* hash = nullptr;
    uint64_t n = 0;
    bool ok = true;
    void raw(const void* p, size_t c) {
        n += c;
        if (!f || !ok || !c) return;
        if (hash) hash->update(p, c);
        ok = f->write(p, c);
    }
    void u64(uint64_t v) { raw(&v, 8); }
    void i64(int64_t v) { raw(&v, 8); }
    template<class T> void vec(const std::vector<T>& v) { u64(v.size()); raw(v.data(), v.size() * sizeof(T)); }
    void buffer(const ConversationBuffer& b) {
        u64(b.size());
        b.visit(0, b.size(), [&](const uint8_t* p, size_t c, size_t) { raw(p, c); return ok; });
    }
    // a K/V part pulled from its owner straight into the file block (sizing pass: counted, never read)
    void source(const SessionKvSource& s, size_t part) {
        const size_t size = s.sizes[part];
        u64(size);
        if (!f) { n += size; return; }
        for (size_t at = 0; at < size && ok;) {
            size_t room = 0;
            uint8_t* d = f->span(room);
            const size_t c = std::min(room, size - at);
            if (!s.read || !s.read(part, at, d, c)) { ok = false; read_failed = true; return; }
            if (hash) hash->update(d, c);
            n += c;
            ok = f->commit(c);
            at += c;
        }
    }
    bool read_failed = false;
};

void put_kv_header(Out& o, int format, int64_t cells, int64_t heads, int64_t head_dim, int64_t page_size,
                   int64_t pooled_rows, int64_t idx_dim) {
    for (int64_t v : {int64_t(format), cells, heads, head_dim, page_size, pooled_rows, idx_dim}) o.i64(v);
}

void put_checkpoint(Out& o, const ConversationCheckpoint& c) {
    o.vec(c.ids);
    o.u64(c.imgs.size());
    for (const auto& k : c.imgs) { o.i64(k.start); o.u64(k.hash); }
    o.vec(c.gdn); o.vec(c.ple); o.vec(c.tails); o.vec(c.dead); o.vec(c.block_pos);
    o.u64(c.used);
}

void put_payload(Out& o, const SavedConversation& s, const std::vector<SessionKvSource>* sources = nullptr) {
    for (int64_t g : s.geometry) o.i64(g);
    o.i64(s.layer_lo); o.i64(s.layer_hi); o.u64(s.cvec ? 1 : 0);
    put_checkpoint(o, s.live);
    o.u64(s.checkpoints.size());
    for (const auto& c : s.checkpoints) put_checkpoint(o, c);
    if (sources) {
        o.u64(sources->size());
        for (const auto& k : *sources) {
            put_kv_header(o, k.format, k.cells, k.heads, k.head_dim, k.page_size, k.pooled_rows, k.idx_dim);
            for (size_t part = 0; part < 5; ++part) o.source(k, part);
        }
        return;
    }
    o.u64(s.kv.size());
    for (const auto& k : s.kv) {
        put_kv_header(o, k.format, k.cells, k.heads, k.head_dim, k.page_size, k.pooled_rows, k.idx_dim);
        o.buffer(k.k); o.buffer(k.v); o.buffer(k.k_scale); o.buffer(k.v_scale); o.buffer(k.pooled);
    }
}

// ---- parsing: every count is checked against the bytes left before anything is allocated
struct In {
    FileSource* f;
    SessionHasher hash;
    uint64_t left;
    std::string& error;
    bool fail(const std::string& m) { if (error.empty()) error = "session file: " + m; return false; }
    bool raw(void* p, size_t c) {
        if (c > left) return fail("payload ends early");
        if (c && !f->read(p, c)) return fail("read error");
        hash.update(p, c);
        left -= c;
        return true;
    }
    bool u64(uint64_t& v) { return raw(&v, 8); }
    bool i64(int64_t& v) { return raw(&v, 8); }
    bool count(uint64_t& n, size_t elem) {
        if (!u64(n)) return false;
        if (elem && n > left / elem) return fail("element count exceeds the payload");
        return true;
    }
    template<class T> bool vec(std::vector<T>& v) {
        uint64_t n = 0;
        if (!count(n, sizeof(T))) return false;
        v.resize((size_t) n);
        return raw(v.data(), (size_t) n * sizeof(T));
    }
    bool buffer(ConversationBuffer& b) {
        uint64_t n = 0;
        if (!count(n, 1)) return false;
        b = {};
        b.resize((size_t) n);
        return b.visit(0, b.size(), [&](uint8_t* p, size_t c, size_t) { return raw(p, c); });
    }
};

bool get_checkpoint(In& in, ConversationCheckpoint& c) {
    if (!in.vec(c.ids)) return false;
    uint64_t n = 0;
    if (!in.count(n, 16)) return false;
    c.imgs.resize((size_t) n);
    for (auto& k : c.imgs) if (!in.i64(k.start) || !in.u64(k.hash)) return false;
    return in.vec(c.gdn) && in.vec(c.ple) && in.vec(c.tails) && in.vec(c.dead) && in.vec(c.block_pos) &&
           in.u64(c.used);
}

bool get_payload(In& in, SavedConversation& s) {
    for (auto& g : s.geometry) if (!in.i64(g)) return false;
    uint64_t cvec = 0;
    if (!in.i64(s.layer_lo) || !in.i64(s.layer_hi) || !in.u64(cvec)) return false;
    if (cvec > 1) return in.fail("invalid cvec flag");
    s.cvec = cvec == 1;
    if (!get_checkpoint(in, s.live)) return false;
    uint64_t n = 0;
    // a checkpoint is at least 8 counts + used = 72 bytes; a K/V layer at least 7 + 5 = 96
    if (!in.count(n, 72)) return false;
    s.checkpoints.resize((size_t) n);
    for (auto& c : s.checkpoints) if (!get_checkpoint(in, c)) return false;
    if (!in.count(n, 96)) return false;
    s.kv.resize((size_t) n);
    for (auto& k : s.kv) {
        int64_t format = 0;
        if (!in.i64(format) || !in.i64(k.cells) || !in.i64(k.heads) || !in.i64(k.head_dim) ||
            !in.i64(k.page_size) || !in.i64(k.pooled_rows) || !in.i64(k.idx_dim)) return false;
        if (format < INT32_MIN || format > INT32_MAX) return in.fail("invalid K/V format");
        k.format = (int) format;
        if (!in.buffer(k.k) || !in.buffer(k.v) || !in.buffer(k.k_scale) || !in.buffer(k.v_scale) ||
            !in.buffer(k.pooled)) return false;
    }
    return true;
}

bool has_stage_parts(const SavedConversation& s) {
    if (!s.live.stage_parts.empty()) return true;
    for (const auto& c : s.checkpoints) if (!c.stage_parts.empty()) return true;
    return false;
}

void header_bytes(uint8_t* h, const SessionFileIdentity& id, uint64_t payload) {
    std::memset(h, 0, kHeader);
    std::memcpy(h, kMagic, 8);
    const uint32_t version = kVersion, size = kHeader;
    std::memcpy(h + 8, &version, 4);
    std::memcpy(h + 12, &size, 4);
    std::memcpy(h + 16, &id.model, 8);
    std::memcpy(h + 24, &id.config, 8);
    std::memcpy(h + 32, &payload, 8);
    const uint64_t hh = session_hash64(h, 56, 0);
    std::memcpy(h + 56, &hh, 8);
}
} // namespace

SessionHasher::SessionHasher(uint64_t seed)
    : lane_{seed + P1 + P2, seed + P2, seed, seed - P1}, pending_{} {}

void SessionHasher::block(const uint8_t* p) {
    for (int i = 0; i < 4; ++i) lane_[i] = round1(lane_[i], load64(p + 8 * i));
}

void SessionHasher::update(const void* data, size_t n) {
    const uint8_t* p = static_cast<const uint8_t*>(data);
    total_ += n;
    if (npending_) {
        const size_t take = std::min(n, sizeof pending_ - npending_);
        std::memcpy(pending_ + npending_, p, take);
        npending_ += take; p += take; n -= take;
        if (npending_ < sizeof pending_) return;
        block(pending_);
        npending_ = 0;
    }
    for (; n >= 32; p += 32, n -= 32) block(p);
    std::memcpy(pending_, p, n);
    npending_ = n;
}

uint64_t SessionHasher::digest() const {
    uint64_t h = rotl(lane_[0], 1) + rotl(lane_[1], 7) + rotl(lane_[2], 12) + rotl(lane_[3], 18);
    for (uint64_t l : lane_) { h ^= round1(0, l); h = h * P1 + P4; }
    h += total_;
    size_t i = 0;
    for (; i + 8 <= npending_; i += 8) { h ^= round1(0, load64(pending_ + i)); h = rotl(h, 27) * P1 + P4; }
    for (; i < npending_; ++i) { h ^= pending_[i] * P5; h = rotl(h, 11) * P1; }
    h ^= h >> 33; h *= P2; h ^= h >> 29; h *= P3; h ^= h >> 32;
    return h;
}

uint64_t session_hash64(const void* data, size_t n, uint64_t seed) {
    SessionHasher h(seed);
    h.update(data, n);
    return h.digest();
}

bool session_model_fingerprint(const std::vector<std::string>& paths, uint64_t& fingerprint, std::string& error) {
    namespace fs = std::filesystem;
    SessionHasher h(0x5354524154414d44ull);
    std::vector<uint8_t> buf(1u << 20);
    auto add_file = [&](const fs::path& p, const std::string& name) -> bool {
        std::error_code ec;
        const uint64_t size = fs::file_size(p, ec);
        if (ec) { error = "session fingerprint: " + p.string() + ": " + ec.message(); return false; }
        h.update(name.data(), name.size());
        h.update(&size, 8);
        File f(std::fopen(p.string().c_str(), "rb"));
        if (!f) { error = "session fingerprint: cannot open " + p.string(); return false; }
        const uint64_t head = std::min<uint64_t>(size, buf.size());
        if (std::fread(buf.data(), 1, head, f.get()) != head) { error = "session fingerprint: read " + p.string(); return false; }
        h.update(buf.data(), head);
        if (size > buf.size()) {
            const uint64_t tail = std::min<uint64_t>(size - head, buf.size());
            if (std::fseek(f.get(), -(long) tail, SEEK_END) != 0 ||
                std::fread(buf.data(), 1, tail, f.get()) != tail) { error = "session fingerprint: read " + p.string(); return false; }
            h.update(buf.data(), tail);
        }
        return true;
    };
    for (const auto& s : paths) {
        const fs::path p(s);
        std::error_code ec;
        if (fs::is_directory(p, ec)) {
            std::vector<fs::path> files;
            for (auto it = fs::recursive_directory_iterator(p, ec); !ec && it != fs::recursive_directory_iterator();
                 it.increment(ec))
                if (it->is_regular_file()) files.push_back(it->path());
            if (ec) { error = "session fingerprint: " + s + ": " + ec.message(); return false; }
            std::sort(files.begin(), files.end());
            for (const auto& f : files)
                if (!add_file(f, fs::relative(f, p).generic_string())) return false;
        } else if (!add_file(p, p.filename().string())) {
            return false;
        }
    }
    fingerprint = h.digest();
    return true;
}

namespace {
bool write_impl(const std::string& path, const SavedConversation& image, const std::vector<SessionKvSource>* sources,
                const SessionFileIdentity& id, size_t& bytes, std::string& error) {
    if (has_stage_parts(image)) { error = "session file: layer-split state cannot be saved"; return false; }
    if (sources && !image.kv.empty()) { error = "session file: K/V given both in the image and as sources"; return false; }
    Out sizing;
    put_payload(sizing, image, sources);
    const uint64_t payload = sizing.n;
    const std::string tmp = path + ".tmp";
    FileSink f(tmp);
    if (!f.is_open()) { error = "session file: cannot create " + tmp + ": " + f.error(); return false; }
    uint8_t h[kHeader];
    header_bytes(h, id, payload);
    bool ok = f.write(h, kHeader);
    SessionHasher hash(0);
    Out out{&f, &hash};
    if (ok) put_payload(out, image, sources);
    ok = ok && out.ok && out.n == payload;
    if (ok) {
        const uint64_t ph = hash.digest();
        ok = f.write(&ph, 8) && f.write(kEnd, 8);
    }
    ok = f.finish() && ok;
    std::string why = out.read_failed ? std::string("K/V source read") : f.error();
    if (ok && replace_file(tmp, path, why)) {
        bytes = kHeader + payload + kTrailer;
        return true;
    }
    error = "session file: writing " + path + " failed" + (why.empty() ? std::string() : ": " + why);
    remove_file(tmp);
    return false;
}
} // namespace

std::vector<ConversationCheckpoint> session_checkpoints_to_save(const std::vector<ConversationCheckpoint>& chain) {
    const ConversationCheckpoint* deepest = nullptr;
    for (const auto& c : chain)
        if (!deepest || c.ids.size() > deepest->ids.size()) deepest = &c;
    if (!deepest) return {};
    return {*deepest};
}

bool session_file_write(const std::string& path, const SavedConversation& meta, const std::vector<SessionKvSource>& kv,
                        const SessionFileIdentity& id, size_t& bytes, std::string& error) {
    return write_impl(path, meta, &kv, id, bytes, error);
}

bool session_file_write(const std::string& path, const SavedConversation& image, const SessionFileIdentity& id,
                        size_t& bytes, std::string& error) {
    return write_impl(path, image, nullptr, id, bytes, error);
}

bool session_file_read(const std::string& path, const SessionFileIdentity& id, SavedConversation& image,
                       size_t& bytes, std::string& error) {
    error.clear();
    std::error_code ec;
    const uint64_t size = std::filesystem::file_size(fs_path(path), ec);
    if (ec) { error = "session file: " + path + ": " + ec.message(); return false; }
    if (size < kHeader + kTrailer) { error = "session file: size " + std::to_string(size) + " is below the minimum"; return false; }
    FileSource f(path);
    if (!f.is_open()) { error = "session file: cannot open " + path + ": " + f.error(); return false; }
    uint8_t h[kHeader];
    if (!f.read(h, kHeader)) { error = "session file: header read error"; return false; }
    uint32_t version = 0, hsize = 0;
    uint64_t model = 0, config = 0, payload = 0, r0 = 0, r1 = 0, hh = 0;
    std::memcpy(&version, h + 8, 4); std::memcpy(&hsize, h + 12, 4);
    std::memcpy(&model, h + 16, 8); std::memcpy(&config, h + 24, 8); std::memcpy(&payload, h + 32, 8);
    std::memcpy(&r0, h + 40, 8); std::memcpy(&r1, h + 48, 8); std::memcpy(&hh, h + 56, 8);
    if (std::memcmp(h, kMagic, 8) != 0) { error = "session file: not a Strata session file (magic)"; return false; }
    if (hh != session_hash64(h, 56, 0)) { error = "session file: header checksum mismatch"; return false; }
    if (version != kVersion) { error = "session file: unsupported version " + std::to_string(version); return false; }
    if (hsize != kHeader || r0 || r1) { error = "session file: invalid header fields"; return false; }
    if (model != id.model) { error = "session file: saved with another model (model fingerprint differs)"; return false; }
    if (config != id.config) { error = "session file: saved with another engine configuration (config fingerprint differs)"; return false; }
    if (payload > size || size - kHeader - kTrailer != payload) {
        error = "session file: size " + std::to_string(size) + " does not match the header (" +
                std::to_string(kHeader + payload + kTrailer) + ")";
        return false;
    }
    SavedConversation parsed;
    In in{&f, SessionHasher(0), payload, error};
    if (!get_payload(in, parsed)) return false;
    if (in.left) { error = "session file: payload has " + std::to_string(in.left) + " unparsed bytes"; return false; }
    uint8_t t[kTrailer];
    if (!f.read(t, kTrailer)) { error = "session file: trailer read error"; return false; }
    uint64_t ph = 0;
    std::memcpy(&ph, t, 8);
    if (ph != in.hash.digest()) { error = "session file: payload checksum mismatch"; return false; }
    if (std::memcmp(t + 8, kEnd, 8) != 0) { error = "session file: bad end marker"; return false; }
    image = std::move(parsed);
    bytes = (size_t) size;
    return true;
}

} // namespace strata::core
