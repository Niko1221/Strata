// The spill directory's own format is only the sidecar: the conversation itself is an ordinary session file,
// written and read by conversation_file.cpp, so a spilled conversation is byte-for-byte what the slot save/restore
// API would write and the same model/config identity rules refuse a foreign file. The sidecar carries the token and
// image metadata (no K/V) so best() can match without reading the conversation.
#include "strata/core/conversation_spill.hpp"

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iterator>
#include <limits>
#include <new>
#include <sstream>
#include <stdexcept>
#include <system_error>

#if defined(_WIN32)
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#include <climits>
#else
#include <cerrno>
#include <fcntl.h>
#include <sys/stat.h>
#include <unistd.h>
#endif

namespace strata::core {
namespace {

constexpr char kMetaMagic[4] = {'S', 'C', 'S', 'M'};   // Strata conversation-spill metadata
constexpr uint32_t kMetaVersion = 2;                   // v1 (read for compatibility) has no last_used field
constexpr uint32_t kMetaVersionMin = 1;
constexpr size_t kMaxSidecarEntries = 256;

// The last-use stamp: wall-clock milliseconds, so the order survives a restart (a steady clock would start over).
uint64_t now_ms() {
    return (uint64_t) std::chrono::duration_cast<std::chrono::milliseconds>(
               std::chrono::system_clock::now().time_since_epoch()).count();
}

// The chain manifest beside a delta log: base file size, committed log length, and a hash over both.
constexpr char kChainMagic[8] = {'S', 'C', 'D', 'C', 'H', 'A', 'I', '\x01'};
constexpr uint64_t kChainVersion = 1;
constexpr uint64_t kChainFile = 40;                     // magic + version + base + committed + hash

#if defined(_WIN32)
// the session files' helpers (conversation_file.cpp, same anonymous-namespace shape): paths reach the engine as
// UTF-8 - invalid UTF-8 is refused, not replaced - and API errors become a message without trailing whitespace
bool wide(const std::string& s, std::wstring& w) {
    w.clear();
    if (s.empty()) return true;
    if (s.size() > (size_t) INT_MAX) return false;
    const int n = MultiByteToWideChar(CP_UTF8, MB_ERR_INVALID_CHARS, s.data(), (int) s.size(), nullptr, 0);
    if (n <= 0) return false;
    w.assign((size_t) n, L'\0');
    return MultiByteToWideChar(CP_UTF8, MB_ERR_INVALID_CHARS, s.data(), (int) s.size(), w.data(), n) == n;
}

std::string last_error(DWORD code = GetLastError()) {
    char msg[256] = {};
    const DWORD n = FormatMessageA(FORMAT_MESSAGE_FROM_SYSTEM | FORMAT_MESSAGE_IGNORE_INSERTS, nullptr, code, 0, msg,
                                   sizeof msg, nullptr);
    std::string s(msg, n);
    while (!s.empty() && (s.back() == '\n' || s.back() == '\r' || s.back() == ' ' || s.back() == '.')) s.pop_back();
    return s.empty() ? "error " + std::to_string(code) : s;
}
#endif

// A small length-delimited writer/reader for the sidecar, with a trailing hash of the payload (the same
// SessionHasher the session file uses) so a truncated or edited sidecar is refused rather than trusted.
class MetaWriter {
public:
    explicit MetaWriter(std::ostream& out) : out_(out) {}
    bool raw(const void* p, size_t n) {
        out_.write(static_cast<const char*>(p), (std::streamsize) n);
        if (!out_) return false;
        hash_.update(p, n);
        return true;
    }
    bool u64(uint64_t v) { return raw(&v, sizeof v); }
    bool i64(int64_t v) { return raw(&v, sizeof v); }
    bool finish() { const uint64_t d = hash_.digest(); return raw(&d, sizeof d); }
private:
    std::ostream& out_;
    SessionHasher hash_{0};
};

class MetaReader {
public:
    MetaReader(std::istream& in, uint64_t file_bytes) : in_(in), file_bytes_(file_bytes) {}
    uint64_t remaining() const { return consumed_ <= file_bytes_ && file_bytes_ - consumed_ >= 8 ? file_bytes_ - consumed_ - 8 : 0; }
    bool raw(void* p, size_t n) {
        if (n > remaining()) return false;
        in_.read(static_cast<char*>(p), (std::streamsize) n);
        if (in_.gcount() != (std::streamsize) n) return false;
        hash_.update(p, n);
        consumed_ += n;
        return true;
    }
    bool u64(uint64_t& v) { return raw(&v, sizeof v); }
    bool i64(int64_t& v) { return raw(&v, sizeof v); }
    bool finish() {
        if (consumed_ + 8 != file_bytes_) return false;
        uint64_t want = 0;
        in_.read(reinterpret_cast<char*>(&want), 8);
        if (in_.gcount() != 8) return false;
        return want == hash_.digest();
    }
private:
    std::istream& in_;
    uint64_t file_bytes_ = 0, consumed_ = 0;
    SessionHasher hash_{0};
};

bool valid_meta_chain(const ConversationCheckpoint& live, const std::vector<size_t>& checkpoint_lengths) {
    if (live.ids.empty()) return false;
    int64_t previous_image = -1;
    for (const auto& key : live.imgs) {
        if (key.start <= previous_image || key.start < 0 || (uint64_t) key.start >= live.ids.size()) return false;
        previous_image = key.start;
    }
    for (size_t length : checkpoint_lengths)
        if (length == 0 || length > live.ids.size()) return false;
    return true;
}

struct PathInfo {
    bool inspected = true;
    bool exists = false;
    bool regular = false;
    uint64_t bytes = 0;
};

// symlink_status is deliberate: cache artifacts must be regular files, and accounting must never follow a link to
// somebody else's file.  A non-regular path is reported as existing but has no target size attributed to it.
PathInfo inspect_path_nofollow(const std::filesystem::path& path) {
    PathInfo info;
    std::error_code ec;
    const auto status = std::filesystem::symlink_status(path, ec);
    if (ec == std::errc::no_such_file_or_directory || ec == std::errc::not_a_directory) {
        ec.clear();
        return info;
    }
    if (ec) { info.inspected = false; return info; }
    if (status.type() == std::filesystem::file_type::not_found) return info;
    info.exists = true;
    if (!std::filesystem::is_regular_file(status)) return info;
    info.regular = true;
    info.bytes = std::filesystem::file_size(path, ec);
    if (ec) { info.inspected = false; info.regular = false; info.bytes = 0; }
    return info;
}

bool read_chain_header(const std::filesystem::path& chain, uint64_t& base_size, uint64_t& committed,
                       std::string& error) {
    error.clear();
    const PathInfo info = inspect_path_nofollow(chain);
    if (!info.inspected || !info.regular || info.bytes != kChainFile) {
        error = "spill chain manifest is missing, non-regular or malformed";
        return false;
    }
    std::ifstream in(chain, std::ios::binary);
    if (!in) { error = "cannot open the spill chain manifest"; return false; }
    uint8_t buf[kChainFile];
    in.read(reinterpret_cast<char*>(buf), (std::streamsize) kChainFile);
    if (in.gcount() != (std::streamsize) kChainFile) { error = "truncated spill chain manifest"; return false; }
    if (std::memcmp(buf, kChainMagic, 8) != 0) { error = "unknown spill chain manifest"; return false; }
    uint64_t version = 0, hash = 0;
    std::memcpy(&version, buf + 8, 8);
    std::memcpy(&base_size, buf + 16, 8);
    std::memcpy(&committed, buf + 24, 8);
    std::memcpy(&hash, buf + 32, 8);
    if (version != kChainVersion || hash != session_hash64(buf, 32, 0) || committed == 0) {
        error = "spill chain manifest checksum or length mismatch";
        return false;
    }
    return true;
}

// ---- the recompute gate ledger (format documented in conversation_spill.hpp) ----
constexpr char kGateName[] = ".strata-cache-gate";
constexpr char kGateTempName[] = ".strata-cache-gate.tmp";
constexpr uint64_t kGateVersion = 1;
constexpr uint64_t kMaxGateLedgerBytes = 4u << 20;   // a ledger for 256 files and their batches is far smaller

// A token field: non-empty and free of whitespace and control characters, so a line never needs quoting.
bool gate_token_ok(const std::string& s) {
    if (s.empty()) return false;
    for (const unsigned char c : s)
        if (c <= ' ' || c == 0x7f) return false;
    return true;
}
bool gate_parse_u64(const std::string& s, int base, uint64_t& out) {
    if (s.empty() || s[0] == '-' || s[0] == '+') return false;   // no signs: a byte count or a fingerprint is unsigned
    errno = 0;
    char* end = nullptr;
    const unsigned long long v = std::strtoull(s.c_str(), &end, base);
    if (errno != 0 || end != s.c_str() + s.size()) return false;
    out = (uint64_t) v;
    return true;
}
bool gate_parse_i64(const std::string& s, int64_t& out) {
    if (s.empty()) return false;
    errno = 0;
    char* end = nullptr;
    const long long v = std::strtoll(s.c_str(), &end, 10);
    if (errno != 0 || end != s.c_str() + s.size()) return false;
    out = (int64_t) v;
    return true;
}
std::string gate_hex(uint64_t v) {
    char b[19];
    std::snprintf(b, sizeof b, "0x%016llx", (unsigned long long) v);
    return b;
}
// Split one line on single spaces; an empty field (two spaces, a leading or a trailing one) is kept and later
// rejected by the exact field-count check.
std::vector<std::string> gate_fields(const std::string& line) {
    std::vector<std::string> out;
    std::size_t at = 0;
    for (;;) {
        const std::size_t space = line.find(' ', at);
        if (space == std::string::npos) { out.push_back(line.substr(at)); break; }
        out.push_back(line.substr(at, space - at));
        at = space + 1;
    }
    return out;
}
// A reason as one token: empty is written as '-', whitespace is replaced so a writer never breaks the line.
std::string gate_reason_token(const std::string& reason) {
    if (reason.empty()) return "-";
    std::string s = reason;
    for (char& c : s)
        if ((unsigned char) c <= ' ') c = '_';
    return s;
}

} // namespace

bool ConversationSpillCache::read_sidecar(const std::filesystem::path& meta, Entry& entry, bool& other_identity,
                                           bool& stamped, std::string& error) const {
    other_identity = false;
    stamped = false;
    const PathInfo meta_info = inspect_path_nofollow(meta);
    const uint64_t file_bytes = meta_info.bytes;
    if (!meta_info.inspected || !meta_info.regular || file_bytes < 4 + 4 + 16 + 8) {
        error = "spill sidecar is missing, non-regular or too short";
        return false;
    }
    std::ifstream in(meta, std::ios::binary);
    if (!in) { error = "cannot open spill sidecar"; return false; }
    MetaReader r(in, file_bytes);
    char magic[4];
    uint64_t version = 0;
    uint64_t model = 0, config = 0;
    if (!r.raw(magic, 4) || std::memcmp(magic, kMetaMagic, 4) != 0 || !r.u64(version) ||
        version < kMetaVersionMin || version > kMetaVersion || !r.u64(model) || !r.u64(config)) {
        error = "unknown spill sidecar magic or version"; return false;
    }
    stamped = version >= kMetaVersion;   // the v2 sidecar carries the persisted last-use stamp
    entry.source_identity = {model, config};
    const bool foreign_config = config != identity_.config;
    // Another MODEL is never approved.  Another CONFIGURATION is parsed only when the gate's ledger holds a passed
    // batch for exactly this source -> this cache's identity and engine build; the per-entry row is checked after
    // the parse (its size and token count must match what the sidecar and file say).
    if (model != identity_.model || (foreign_config && !gate_identity_approved(entry.source_identity))) {
        other_identity = true;
        return false;
    }
    uint64_t cvec = 0;
    if (!r.u64(cvec) || cvec > 1) { error = "invalid cvec flag in sidecar"; return false; }
    entry.cvec = cvec != 0;
    uint64_t n = 0;
    if (!r.u64(n) || n > r.remaining() / 4 || n > (uint64_t) std::numeric_limits<size_t>::max()) {
        error = "invalid token count in sidecar"; return false;
    }
    try { entry.live_meta.ids.resize((size_t) n); } catch (const std::bad_alloc&) { error = "not enough RAM for sidecar"; return false; }
    for (auto& id : entry.live_meta.ids) { int64_t v = 0; if (!r.i64(v)) { error = "truncated sidecar ids"; return false; } id = (int32_t) v; }
    if (!r.u64(n) || n > r.remaining() / 16) { error = "invalid image count in sidecar"; return false; }
    try { entry.live_meta.imgs.resize((size_t) n); } catch (const std::bad_alloc&) { error = "not enough RAM for sidecar"; return false; }
    for (auto& k : entry.live_meta.imgs)
        if (!r.i64(k.start) || !r.u64(k.hash)) { error = "truncated sidecar images"; return false; }
    if (!r.u64(n) || n > kMaxSidecarEntries) { error = "invalid checkpoint count in sidecar"; return false; }
    entry.checkpoint_lengths.reserve((size_t) n);
    for (uint64_t i = 0; i < n; ++i) { uint64_t len = 0; if (!r.u64(len)) { error = "truncated sidecar checkpoints"; return false; } entry.checkpoint_lengths.push_back((size_t) len); }
    entry.last_used = 0;   // a v1 sidecar predates the stamp: the unknown age sorts as the oldest
    if (version >= 2 && !r.u64(entry.last_used)) { error = "truncated sidecar last-use stamp"; return false; }
    if (!r.finish()) { error = "spill sidecar checksum or length mismatch"; return false; }
    if (!valid_meta_chain(entry.live_meta, entry.checkpoint_lengths)) { error = "spill sidecar is not a valid prefix chain"; return false; }
    if (foreign_config) {
        const PathInfo session = inspect_path_nofollow(entry.session_path());
        if (!session.inspected || !session.regular ||
            !gate_entry_approved(entry.stem.filename().string(), entry.source_identity, session.bytes,
                                 (int64_t) entry.live_meta.ids.size())) {
            other_identity = true;
            return false;
        }
    }
    return true;
}

bool ConversationSpillCache::write_sidecar(const std::filesystem::path& meta, const Entry& entry, std::string& error,
                                           bool with_last_used) const {
    const std::filesystem::path temporary(meta.string() + ".tmp");
    std::ostringstream payload;
    {
        MetaWriter w(payload);
        // The sidecar records the identity the FILE carries, not this cache's: for a gate-approved entry they differ,
        // and stamping this cache's model/config would forge the file's identity on the next open (touch() rewrites
        // the sidecar on every hit), letting it be read after a restart without the ledger.  A new entry has
        // source_identity = identity_ (index_written), so its bytes are unchanged.
        if (!w.raw(kMetaMagic, 4) ||
            !w.u64(with_last_used ? kMetaVersion : kMetaVersionMin) || !w.u64(entry.source_identity.model) ||
            !w.u64(entry.source_identity.config) || !w.u64(entry.cvec ? 1 : 0) || !w.u64(entry.live_meta.ids.size())) {
            error = "cannot write spill sidecar"; return false;
        }
        for (int32_t id : entry.live_meta.ids) if (!w.i64(id)) { error = "cannot write spill sidecar ids"; return false; }
        if (!w.u64(entry.live_meta.imgs.size())) { error = "cannot write spill sidecar"; return false; }
        for (const auto& k : entry.live_meta.imgs) if (!w.i64(k.start) || !w.u64(k.hash)) { error = "cannot write spill sidecar images"; return false; }
        if (!w.u64(entry.checkpoint_lengths.size())) { error = "cannot write spill sidecar"; return false; }
        for (size_t len : entry.checkpoint_lengths) if (!w.u64(len)) { error = "cannot write spill sidecar checkpoints"; return false; }
        if (with_last_used && !w.u64(entry.last_used)) { error = "cannot write spill sidecar last-use stamp"; return false; }
        if (!w.finish()) { error = "cannot finish spill sidecar"; return false; }
    }
    const std::string bytes = payload.str();
    // The sidecar's token ids are decodable conversation content: the temp is created the way the session files
    // are (conversation_file.cpp RawFile::create_new) - exclusive, never following a link, owner-only (0600) -
    // not by truncating whatever a name happens to point at.
#if defined(_WIN32)
    // the session files' safe creation (conversation_file.cpp RawFile::create_new): CREATE_NEW never truncates an
    // existing file or follows a link - the path is converted UTF-8 -> UTF-16 first
    std::wstring w;
    if (!wide(temporary.string(), w)) { error = "cannot create spill sidecar: the path is not valid UTF-8"; return false; }
    const HANDLE h = CreateFileW(w.c_str(), GENERIC_WRITE, 0, nullptr, CREATE_NEW,
                                 FILE_ATTRIBUTE_NORMAL | FILE_FLAG_SEQUENTIAL_SCAN, nullptr);
    if (h == INVALID_HANDLE_VALUE) { error = "cannot create spill sidecar: " + last_error(); return false; }
    bool ok = true;
    size_t off = 0;
    while (ok && off < bytes.size()) {
        DWORD wrote = 0;
        const DWORD want = (DWORD) std::min<size_t>(bytes.size() - off, size_t(1) << 30);
        if (!WriteFile(h, bytes.data() + off, want, &wrote, nullptr) || wrote == 0) ok = false;
        else off += (size_t) wrote;
    }
    if (!ok) error = "cannot write spill sidecar: " + last_error();
    if (CloseHandle(h) == 0 && ok) { ok = false; error = "cannot write spill sidecar: " + last_error(); }
    if (!ok) { std::error_code rm; std::filesystem::remove(temporary, rm); return false; }
#else
    const int fd = ::open(temporary.c_str(), O_WRONLY | O_CREAT | O_EXCL | O_NOFOLLOW | O_CLOEXEC, 0600);
    if (fd < 0) { error = std::string("cannot create spill sidecar: ") + std::strerror(errno); return false; }
    size_t off = 0;
    while (off < bytes.size()) {
        const ssize_t n = ::write(fd, bytes.data() + off, bytes.size() - off);
        if (n > 0) { off += (size_t) n; continue; }
        if (n < 0 && errno == EINTR) continue;
        error = std::string("cannot write spill sidecar: ") + (n < 0 ? std::strerror(errno) : "short write");
        ::close(fd);
        std::error_code rm; std::filesystem::remove(temporary, rm);
        return false;
    }
    if (::close(fd) != 0) {
        error = std::string("cannot write spill sidecar: ") + std::strerror(errno);
        std::error_code rm; std::filesystem::remove(temporary, rm);
        return false;
    }
#endif
    std::error_code ec;
    std::filesystem::rename(temporary, meta, ec);
    if (ec) { std::filesystem::remove(temporary, ec); error = "cannot commit spill sidecar"; return false; }
    return true;
}

bool session_spill_gate_read(const std::filesystem::path& directory, std::vector<ConversationSpillGateBatch>& batches,
                             std::vector<ConversationSpillGateEntry>& entries, std::string& error) {
    batches.clear();
    entries.clear();
    error.clear();
    const std::filesystem::path path = directory / kGateName;
    const PathInfo info = inspect_path_nofollow(path);
    if (!info.inspected) { error = "recompute gate: cannot inspect " + path.string(); return false; }
    if (!info.regular) { error = "recompute gate: ledger is missing or not a regular file"; return false; }
    if (info.bytes > kMaxGateLedgerBytes) { error = "recompute gate: ledger is larger than the format allows"; return false; }
    std::ifstream in(path, std::ios::binary);
    if (!in) { error = "recompute gate: cannot open " + path.string(); return false; }
    const std::string text((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
    // A writer publishes by rename, so a wrong length or a missing final newline means a half-written or changed file.
    if (text.size() != info.bytes || text.empty() || text.back() != '\n') {
        error = "recompute gate: ledger is truncated or changed while it was read";
        return false;
    }
    // The checksum is the last line and covers every byte before it.
    const std::size_t last = text.size() - 1;
    const std::size_t previous = last == 0 ? std::string::npos : text.rfind('\n', last - 1);
    const std::size_t sum_at = previous == std::string::npos ? 0 : previous + 1;
    const std::vector<std::string> sum = gate_fields(text.substr(sum_at, last - sum_at));
    uint64_t want = 0;
    if (sum.size() != 2 || sum[0] != "sum" || !gate_parse_u64(sum[1], 16, want)) {
        error = "recompute gate: ledger has no checksum line";
        return false;
    }
    if (want != session_hash64(text.data(), sum_at, 0)) {
        error = "recompute gate: ledger checksum mismatch";
        return false;
    }
    std::istringstream body(text.substr(0, sum_at));
    std::string line;
    bool header = false;
    while (std::getline(body, line)) {
        const std::vector<std::string> f = gate_fields(line);
        if (!header) {
            if (f.size() != 2 || f[0] != "strata-cache-gate" ||
                f[1] != std::to_string((unsigned long long) kGateVersion)) {
                error = "recompute gate: unknown ledger header or version";
                return false;
            }
            header = true;
            continue;
        }
        if (!f.empty() && f[0] == "batch") {
            ConversationSpillGateBatch b;
            uint64_t time = 0;
            if (f.size() != 11 || !gate_token_ok(f[1]) || !gate_token_ok(f[6]) || !gate_token_ok(f[7]) ||
                !gate_parse_u64(f[2], 16, b.source.model) || !gate_parse_u64(f[3], 16, b.source.config) ||
                !gate_parse_u64(f[4], 16, b.target.model) || !gate_parse_u64(f[5], 16, b.target.config) ||
                !gate_parse_u64(f[9], 10, time) || (f[8] != "pass" && f[8] != "fail") ||
                (f[10] != "-" && !gate_token_ok(f[10]))) {
                error = "recompute gate: malformed batch line";
                return false;
            }
            b.id = f[1];
            b.engine_build = f[6];
            b.sample = f[7];
            b.passed = f[8] == "pass";
            b.verified_ms = time;
            b.reason = f[10] == "-" ? std::string() : f[10];
            batches.push_back(std::move(b));
        } else if (!f.empty() && f[0] == "entry") {
            ConversationSpillGateEntry e;
            int64_t tokens = 0;
            if (f.size() != 6 || !gate_token_ok(f[1]) || !gate_token_ok(f[2]) ||
                !gate_parse_u64(f[3], 10, e.file_bytes) || !gate_parse_i64(f[4], tokens) || tokens < 0 ||
                !gate_parse_u64(f[5], 10, e.last_used)) {
                error = "recompute gate: malformed entry line";
                return false;
            }
            e.stem = f[1];
            e.batch_id = f[2];
            e.tokens = tokens;
            entries.push_back(std::move(e));
        } else {
            error = "recompute gate: unknown ledger line";
            return false;
        }
    }
    if (!header) { error = "recompute gate: ledger has no header"; return false; }
    return true;
}

bool session_spill_gate_write(const std::filesystem::path& directory,
                              const std::vector<ConversationSpillGateBatch>& batches,
                              const std::vector<ConversationSpillGateEntry>& entries, std::string& error) {
    error.clear();
    std::string body = "strata-cache-gate " + std::to_string((unsigned long long) kGateVersion) + "\n";
    for (const auto& b : batches) {
        if (!gate_token_ok(b.id) || !gate_token_ok(b.engine_build) || !gate_token_ok(b.sample)) {
            error = "recompute gate: a batch id, engine build or sample is empty or contains whitespace";
            return false;
        }
        body += "batch " + b.id + " " + gate_hex(b.source.model) + " " + gate_hex(b.source.config) + " " +
                gate_hex(b.target.model) + " " + gate_hex(b.target.config) + " " + b.engine_build + " " +
                b.sample + " " + (b.passed ? "pass" : "fail") + " " + std::to_string(b.verified_ms) + " " +
                gate_reason_token(b.reason) + "\n";
    }
    for (const auto& e : entries) {
        if (!gate_token_ok(e.stem) || !gate_token_ok(e.batch_id) || e.tokens < 0) {
            error = "recompute gate: an entry stem or batch id is empty or contains whitespace, or its token count is negative";
            return false;
        }
        body += "entry " + e.stem + " " + e.batch_id + " " + std::to_string(e.file_bytes) + " " +
                std::to_string(e.tokens) + " " + std::to_string(e.last_used) + "\n";
    }
    char sum[32];
    std::snprintf(sum, sizeof sum, "sum %016llx\n", (unsigned long long) session_hash64(body.data(), body.size(), 0));
    const std::string bytes = body + sum;
    const std::filesystem::path temporary = directory / kGateTempName;
    const std::filesystem::path target = directory / kGateName;
    // A leftover temporary is a crashed writer's; remove it only when it is a regular file (never a link).
    const PathInfo stale = inspect_path_nofollow(temporary);
    if (!stale.inspected) { error = "recompute gate: cannot inspect the ledger temporary"; return false; }
    if (stale.exists) {
        if (!stale.regular) { error = "recompute gate: the ledger temporary path is not a regular file"; return false; }
        std::error_code rm;
        std::filesystem::remove(temporary, rm);
        if (rm) { error = "recompute gate: cannot remove a stale ledger temporary"; return false; }
    }
    // created like the sidecar's temp: exclusive, never following a link, owner-only (0600)
#if defined(_WIN32)
    std::wstring w;
    if (!wide(temporary.string(), w)) { error = "recompute gate: the path is not valid UTF-8"; return false; }
    const HANDLE h = CreateFileW(w.c_str(), GENERIC_WRITE, 0, nullptr, CREATE_NEW,
                                 FILE_ATTRIBUTE_NORMAL | FILE_FLAG_SEQUENTIAL_SCAN, nullptr);
    if (h == INVALID_HANDLE_VALUE) { error = "recompute gate: cannot create the ledger temporary: " + last_error(); return false; }
    bool ok = true;
    size_t off = 0;
    while (ok && off < bytes.size()) {
        DWORD wrote = 0;
        const DWORD want = (DWORD) std::min<size_t>(bytes.size() - off, size_t(1) << 30);
        if (!WriteFile(h, bytes.data() + off, want, &wrote, nullptr) || wrote == 0) ok = false;
        else off += (size_t) wrote;
    }
    if (!ok) error = "recompute gate: cannot write the ledger temporary: " + last_error();
    if (CloseHandle(h) == 0 && ok) { ok = false; error = "recompute gate: cannot close the ledger temporary: " + last_error(); }
    if (!ok) { std::error_code rm; std::filesystem::remove(temporary, rm); return false; }
#else
    const int fd = ::open(temporary.c_str(), O_WRONLY | O_CREAT | O_EXCL | O_NOFOLLOW | O_CLOEXEC, 0600);
    if (fd < 0) { error = std::string("recompute gate: cannot create the ledger temporary: ") + std::strerror(errno); return false; }
    size_t off = 0;
    while (off < bytes.size()) {
        const ssize_t n = ::write(fd, bytes.data() + off, bytes.size() - off);
        if (n > 0) { off += (size_t) n; continue; }
        if (n < 0 && errno == EINTR) continue;
        error = std::string("recompute gate: cannot write the ledger temporary: ") +
                (n < 0 ? std::strerror(errno) : "short write");
        ::close(fd);
        std::error_code rm;
        std::filesystem::remove(temporary, rm);
        return false;
    }
    if (::close(fd) != 0) {
        error = std::string("recompute gate: cannot close the ledger temporary: ") + std::strerror(errno);
        std::error_code rm;
        std::filesystem::remove(temporary, rm);
        return false;
    }
#endif
    std::error_code ec;
    std::filesystem::rename(temporary, target, ec);
    if (ec) {
        std::filesystem::remove(temporary, ec);
        error = "recompute gate: cannot commit the ledger";
        return false;
    }
    return true;
}

bool session_spill_gate_approves(const std::vector<ConversationSpillGateBatch>& batches,
                                 const std::vector<ConversationSpillGateEntry>& entries, const std::string& stem,
                                 const SessionFileIdentity& source, const SessionFileIdentity& target,
                                 const std::string& engine_build, uint64_t file_bytes, int64_t tokens) {
    if (!gate_token_ok(stem) || !gate_token_ok(engine_build)) return false;
    for (const auto& e : entries) {
        if (e.stem != stem || e.file_bytes != file_bytes || e.tokens != tokens) continue;
        for (const auto& b : batches) {
            if (b.id != e.batch_id || !b.passed) continue;
            if (b.source.model != source.model || b.source.config != source.config) continue;
            if (b.target.model != target.model || b.target.config != target.config) continue;
            if (b.engine_build != engine_build) continue;
            return true;
        }
    }
    return false;
}

bool ConversationSpillCache::open(const std::filesystem::path& directory, SessionFileIdentity identity,
                                  uint64_t budget_bytes, std::string& error, bool delta_chains,
                                  const ConversationSpillGateOptions& gate) {
    enabled_ = false;
    stale_files_wiped_ = 0;
    disk_evictions_ = 0;
    entries_.clear();
    bytes_ = 0;
    pinned_path_.clear();
    delta_ = delta_chains;
    gate_ = gate.enabled;
    gate_engine_build_ = gate.engine_build;
    gate_batches_.clear();
    gate_entries_.clear();
    gate_error_.clear();
    directory_ = directory;
    identity_ = identity;
    budget_ = budget_bytes;
    if (directory.empty() || !identity.model || !identity.config || !budget_) return true;
    try {
        std::error_code ec;
#if defined(_WIN32)
        std::filesystem::create_directories(directory_, ec);
#else
        // create the leaf private (0700): it holds the session files, which are owner-only (0600).  Only a
        // directory this engine creates is tightened; an existing one keeps its mode.
        bool made_leaf = ::mkdir(directory_.c_str(), 0700) == 0;
        if (!made_leaf && errno != EEXIST) {   // parents missing: the general path, then tighten the leaf it makes
            std::filesystem::create_directories(directory_, ec);
            made_leaf = !ec;
        }
        if (!ec && made_leaf && ::chmod(directory_.c_str(), 0700) != 0 && errno != EPERM)
            ec.assign(errno, std::generic_category());
#endif
        if (ec || !std::filesystem::is_directory(directory_, ec) || ec) {
            error = "cannot create or inspect spill directory " + directory_.string(); return false;
        }
        if (gate_) {
            // A missing, corrupt or half-written ledger approves nothing; it must never stop the same-identity cache.
            // The reason is kept for the caller's startup line, never printed here: the caller knows whether the gate
            // is worth a line at all.
            if (gate_engine_build_.empty()) {
                gate_error_ = "no engine build was given";
            } else if (!session_spill_gate_read(directory_, gate_batches_, gate_entries_, gate_error_)) {
                gate_batches_.clear();
                gate_entries_.clear();
            }
        }
        std::vector<std::filesystem::path> metas;
        for (std::filesystem::directory_iterator it(directory_, ec), end; !ec && it != end; it.increment(ec)) {
            const std::string name = it->path().filename().string();
            const bool ours = name.rfind("strata-conv-", 0) == 0 || name.rfind(".strata-conv-", 0) == 0;
            if (!ours) continue;
            const PathInfo info = inspect_path_nofollow(it->path());
            if (!info.inspected) { ec = std::make_error_code(std::errc::io_error); break; }
            // the session writer's hidden temp and our fixed sidecar/manifest temps are only removed when they are
            // regular files; a symlink is never followed or mistaken for one of our files.
            if (it->path().extension() == ".meta" && info.regular) metas.push_back(it->path());
            else if (it->path().extension() == ".tmp" && info.regular) {
                std::error_code rm;
                if (std::filesystem::remove(it->path(), rm) && !rm) ++stale_files_wiped_;
            }
        }
        if (ec) { error = "cannot scan spill directory " + directory_.string(); return false; }
        std::sort(metas.begin(), metas.end());
        size_t stale = stale_files_wiped_;
        size_t foreign = 0;
        uint64_t foreign_bytes = 0;
        auto initialize_unparsed_entry = [&](Entry& entry, const PathInfo& meta_info) {
            const PathInfo session = inspect_path_nofollow(entry.session_path());
            const PathInfo dlog = inspect_path_nofollow(entry.delta_path());
            const PathInfo chain = inspect_path_nofollow(entry.chain_path());
            entry.file_bytes = session.regular ? session.bytes : 0;
            entry.delta_accounted_bytes = dlog.regular ? dlog.bytes : 0;
            entry.meta_bytes = meta_info.regular ? meta_info.bytes : 0;
            entry.chain_file_bytes = chain.regular ? chain.bytes : 0;
            entry.has_chain = dlog.exists || chain.exists;
            entry.complete = false;
        };
        auto clean_or_retain = [&](Entry& entry) {
            uint64_t removed = 0;
            std::vector<std::string> left;
            if (remove_group(entry, removed, left)) {
                ++stale;
                return;
            }
            entry.complete = false;
            bytes_ += entry.total_bytes();
            std::fprintf(stderr, "strata serve: conversation cache: incomplete owned spill entry retained for cleanup: %s\n",
                         entry.stem.string().c_str());
            for (const auto& path : left)
                std::fprintf(stderr, "  retained spill artifact: %s\n", path.c_str());
            entries_.push_back(std::move(entry));
        };
        for (const auto& meta : metas) {
            Entry entry;
            entry.stem = meta;
            entry.stem.replace_extension();
            entry.prefix = entry.stem.filename().string().rfind("strata-conv-prefix-", 0) == 0;
            const PathInfo meta_info = inspect_path_nofollow(meta);
            bool other_identity = false;
            bool stamped = false;
            std::string parse_error;
            if (!read_sidecar(meta, entry, other_identity, stamped, parse_error)) {
                if (other_identity) {
                    // A foreign identity is kept, not indexed and outside this identity's budget. Count only regular
                    // files without following links so the warning never attributes an external target to this cache.
                    ++foreign;
                    for (const auto& path : {entry.session_path(), meta.string(), entry.delta_path(), entry.chain_path()}) {
                        const PathInfo info = inspect_path_nofollow(path);
                        if (info.regular) foreign_bytes += info.bytes;
                    }
                    continue;
                }
                initialize_unparsed_entry(entry, meta_info);
                clean_or_retain(entry);
                continue;
            }

            const PathInfo session = inspect_path_nofollow(entry.session_path());
            const PathInfo dlog = inspect_path_nofollow(entry.delta_path());
            const PathInfo chain = inspect_path_nofollow(entry.chain_path());
            if (!session.inspected || !dlog.inspected || !chain.inspected || !meta_info.regular) {
                initialize_unparsed_entry(entry, meta_info);
                clean_or_retain(entry);
                continue;
            }
            entry.file_bytes = session.regular ? session.bytes : 0;
            // v1 sidecars were never part of the baseline disk budget; only a stamped v2 sidecar is charged.
            entry.meta_bytes = stamped ? meta_info.bytes : 0;
            entry.has_chain = dlog.exists || chain.exists;
            entry.complete = session.regular;

            if (entry.has_chain) {
                uint64_t manifest_base = 0, manifest_committed = 0;
                std::string chain_error;
                const bool header_valid = chain.regular &&
                    read_chain_header(entry.chain_path(), manifest_base, manifest_committed, chain_error);
                if (header_valid) {
                    entry.delta_bytes = manifest_committed;
                    entry.delta_accounted_bytes = dlog.regular ? dlog.bytes : (dlog.exists ? manifest_committed : 0);
                    entry.chain_file_bytes = chain.bytes;
                } else {
                    entry.delta_accounted_bytes = dlog.regular ? dlog.bytes : 0;
                    entry.chain_file_bytes = chain.regular ? chain.bytes : 0;
                }
                uint64_t committed = 0;
                if (!entry.complete || !dlog.regular || !chain.regular ||
                    !read_chain(entry.chain_path(), entry.file_bytes, entry.delta_path(), committed, chain_error)) {
                    entry.complete = false;
                } else {
                    entry.delta_bytes = committed;
                    entry.delta_accounted_bytes = dlog.bytes;
                    entry.chain_file_bytes = chain.bytes;
                }
            }

            if (!entry.complete) {
                clean_or_retain(entry);
                continue;
            }
            // A new-format entry that cannot fit alone is cleaned up. A v1/plain legacy entry stays on the original
            // path: its sidecar is uncharged and the baseline eviction policy enforces the session-file budget.
            if (!(entry.has_chain && !delta_) && (entry.has_chain || entry.meta_bytes != 0) &&
                entry.total_bytes() > budget_) {
                clean_or_retain(entry);
                continue;
            }
            const uint64_t total = entry.total_bytes();
            entries_.push_back(std::move(entry));
            bytes_ += total;
            enforce_budget();
        }

        // Artifacts without a sidecar cannot be hits. Clean regular owned leftovers as one group; if an item cannot
        // be removed, retain an incomplete ledger entry so a later open/erase can report and retry it.
        std::vector<std::filesystem::path> orphan_stems;
        for (std::filesystem::directory_iterator it(directory_, ec), end; !ec && it != end; it.increment(ec)) {
            const std::string name = it->path().filename().string();
            if (name.rfind("strata-conv-", 0) != 0) continue;
            const auto ext = it->path().extension();
            if (ext != ".sess" && ext != ".dlog" && ext != ".dchain") continue;
            std::filesystem::path meta = it->path();
            meta.replace_extension(".meta");
            const PathInfo meta_info = inspect_path_nofollow(meta);
            if (!meta_info.inspected) { ec = std::make_error_code(std::errc::io_error); break; }
            if (meta_info.exists) continue;  // a symlink/invalid sidecar is not a plain snapshot either
            std::filesystem::path stem = it->path();
            stem.replace_extension();
            orphan_stems.push_back(std::move(stem));
        }
        if (ec) { error = "cannot scan spill directory " + directory_.string(); return false; }
        std::sort(orphan_stems.begin(), orphan_stems.end());
        orphan_stems.erase(std::unique(orphan_stems.begin(), orphan_stems.end()), orphan_stems.end());
        for (const auto& stem : orphan_stems) {
            Entry entry;
            entry.stem = stem;
            entry.prefix = stem.filename().string().rfind("strata-conv-prefix-", 0) == 0;
            entry.complete = false;
            const PathInfo session = inspect_path_nofollow(entry.session_path());
            const PathInfo dlog = inspect_path_nofollow(entry.delta_path());
            const PathInfo chain = inspect_path_nofollow(entry.chain_path());
            entry.file_bytes = session.regular ? session.bytes : 0;
            entry.delta_accounted_bytes = dlog.regular ? dlog.bytes : 0;
            entry.chain_file_bytes = chain.regular ? chain.bytes : 0;
            entry.has_chain = dlog.exists || chain.exists;
            clean_or_retain(entry);
        }
        if (foreign > 0)
            std::fprintf(stderr, "strata serve: conversation cache: WARNING: %zu spill file(s) (~%llu MiB) belong to "
                                 "another model/config identity: kept on disk, not indexed and outside the disk "
                                 "budget; remove them manually, e.g.\n"
                                 "  rm -f %s/strata-conv-*.sess %s/strata-conv-*.meta\n",
                         foreign, (unsigned long long) (foreign_bytes >> 20),
                         directory_.string().c_str(), directory_.string().c_str());
        enabled_ = true;
        enforce_budget();
        stale_files_wiped_ = stale;
        return true;
    } catch (const std::bad_alloc&) {
        entries_.clear(); bytes_ = 0; enabled_ = false;
        error = "not enough RAM to index spill directory"; return false;
    } catch (const std::exception& e) {
        entries_.clear(); bytes_ = 0; enabled_ = false;
        error = std::string("cannot index spill directory: ") + e.what(); return false;
    }
}

void ConversationSpillCache::adjust_bytes(uint64_t before, uint64_t after) {
    if (after >= before) {
        bytes_ += after - before;
        return;
    }
    const uint64_t debit = before - after;
    if (debit > bytes_) throw std::logic_error("spill disk accounting underflow");
    bytes_ -= debit;
}

bool ConversationSpillCache::entry_files_valid(const Entry& entry) const {
    if (!entry.complete) return false;
    const PathInfo session = inspect_path_nofollow(entry.session_path());
    const PathInfo meta = inspect_path_nofollow(entry.meta_path());
    if (!session.inspected || !session.regular || !meta.inspected || !meta.regular ||
        session.bytes != entry.file_bytes) return false;
    const PathInfo dlog = inspect_path_nofollow(entry.delta_path());
    const PathInfo chain = inspect_path_nofollow(entry.chain_path());
    if (!dlog.inspected || !chain.inspected) return false;
    if (!entry.has_chain) return !dlog.exists && !chain.exists;
    if (!dlog.regular || !chain.regular) return false;
    uint64_t committed = 0;
    std::string error;
    return read_chain(entry.chain_path(), entry.file_bytes, entry.delta_path(), committed, error) &&
           committed == entry.delta_bytes;
}

bool ConversationSpillCache::load(const std::string& path, SavedConversation& image, const SessionReadLimits& limits,
                                  std::string& error, SessionStatus* status) const {
    if (!enabled_) { error = "spill cache is disabled"; return false; }
    const size_t index = find(path);
    if (index >= entries_.size()) {
        error = "spill entry is missing, incomplete, or unsupported by this restore mode";
        return false;
    }
    if (entries_[index].has_chain) {
        error = "delta-chain entries require the streaming chain restore path";
        return false;
    }
    size_t bytes = 0;
    // A gate-approved entry keeps its own configuration: accept exactly that config for this read (the model was
    // checked when the entry was indexed).  Every other entry reads under this cache's identity, as before.
    SessionReadLimits accepted = limits;
    const SessionFileIdentity source = entries_[index].source_identity;
    if (source.config != identity_.config) accepted.accept_config = source.config;
    return session_file_read(path, identity_, image, bytes, error, accepted, status);
}

std::filesystem::path ConversationSpillCache::next_stem(bool prefix) {
    std::error_code ec;
    std::filesystem::path stem;
    do {
        stem = directory_ / ((prefix ? "strata-conv-prefix-" : "strata-conv-") + std::to_string(++serial_));
    } while (std::filesystem::exists(stem.string() + ".sess", ec) || std::filesystem::exists(stem.string() + ".meta", ec) ||
             std::filesystem::exists(stem.string() + ".dlog", ec) || std::filesystem::exists(stem.string() + ".dchain", ec));
    return stem;
}

// The file at <stem>.sess is written: check it against the disk limit, write its sidecar and index it.
bool ConversationSpillCache::index_written(const std::filesystem::path& stem, uint64_t file_bytes,
                                           const SavedConversation& meta, std::string& error, bool prefix) {
    std::error_code ec;
    const std::string session = stem.string() + ".sess";
    if (file_bytes > budget_) { std::filesystem::remove(session, ec); error = "snapshot exceeds the disk cache limit"; return false; }
    Entry entry;
    entry.stem = stem;
    entry.file_bytes = file_bytes;
    entry.source_identity = identity_;
    entry.last_used = delta_ ? now_ms() : 0;   // a save is the newest use; delta-off writes the baseline v1 sidecar,
                                               // which carries no stamp, so the entry stays at the unknown age
    entry.prefix = prefix;
    entry.cvec = meta.cvec;
    entry.live_meta.ids = meta.live.ids;
    entry.live_meta.imgs = meta.live.imgs;
    entry.checkpoint_lengths.reserve(meta.checkpoints.size());
    for (const auto& c : meta.checkpoints) entry.checkpoint_lengths.push_back(c.ids.size());
    if (!write_sidecar(stem.string() + ".meta", entry, error, delta_)) { std::filesystem::remove(session, ec); return false; }
    if (delta_) {
        const PathInfo meta = inspect_path_nofollow(entry.meta_path());
        if (!meta.inspected || !meta.regular) {
            std::filesystem::remove(session, ec);
            std::filesystem::remove(entry.meta_path(), ec);
            error = "cannot account the written spill sidecar";
            return false;
        }
        entry.meta_bytes = meta.bytes;
    }
    try {
        entries_.push_back(std::move(entry));
    } catch (const std::bad_alloc&) {
        std::filesystem::remove(session, ec); std::filesystem::remove(stem.string() + ".meta", ec);
        error = "not enough RAM to index conversation spill file"; return false;
    }
    bytes_ += entries_.back().total_bytes();
    enforce_budget();
    return true;
}

bool ConversationSpillCache::spill(const SavedConversation& image, std::string& error) {
    if (!enabled_) return false;
    if (image.live.ids.empty() || !image.stage_images.empty()) { error = "spill needs a single-GPU conversation"; return false; }
    std::error_code ec;
    const std::filesystem::path stem = next_stem();
    const std::string session = stem.string() + ".sess";
    size_t file_bytes = 0;
    SessionWriteOptions wo;
    wo.durable = true;
    SessionStatus st;
    if (!session_file_write(session, image, identity_, file_bytes, error, wo, &st)) {
        std::filesystem::remove(session, ec);
        return false;
    }
    return index_written(stem, file_bytes, image, error);
}

bool ConversationSpillCache::reserve_streamed(const SavedConversation& meta, ConversationSpillWrite& target,
                                              std::string& error, bool prefix) {
    target = {};
    if (!enabled_) return false;
    if (meta.live.ids.empty() || !meta.stage_images.empty() || !meta.kv.empty()) {
        error = "a streamed spill needs a single-GPU conversation without a host K/V image"; return false;
    }
    target.stem = next_stem(prefix);
    target.identity = identity_;
    target.prefix = prefix;
    target.valid = true;
    return true;
}

bool ConversationSpillCache::publish_streamed(const ConversationSpillWrite& target, const SavedConversation& meta,
                                              size_t bytes, std::string& error) {
    if (!enabled_ || !target.valid || target.identity.model != identity_.model ||
        target.identity.config != identity_.config) {
        error = "spill base reservation is no longer valid";
        return false;
    }
    return index_written(target.stem, bytes, meta, error, target.prefix);
}

bool ConversationSpillCache::spill_streamed(const SavedConversation& meta, const std::vector<SessionKvSource>& kv,
                                            size_t& bytes, std::string& error, bool durable, bool prefix,
                                            bool state_ref, bool state_codec_rht, SessionWriteTimings* timings,
                                            bool layout_v3) {
    bytes = 0;
    ConversationSpillWrite target;
    if (!reserve_streamed(meta, target, error, prefix)) return false;
    std::error_code ec;
    SessionWriteOptions wo;
    wo.durable = durable;
    wo.state_ref = state_ref;   // only a prefix save passes this on; ordinary saves and the delta writer never do
    wo.state_codec_rht = !prefix && state_codec_rht;
    wo.timings = timings;
    wo.layout_v3 = !prefix && layout_v3;   // the K/V-first layout is only for ordinary disk-only saves
    SessionStatus st;
    if (!session_file_write(target.session_path(), meta, kv, target.identity, bytes, error, wo, &st)) {
        std::filesystem::remove(target.session_path(), ec);
        return false;
    }
    return publish_streamed(target, meta, bytes, error);
}

bool ConversationSpillCache::has_prefix(const std::vector<int32_t>& ids, const std::vector<ConversationImageKey>& images,
                                        bool cvec) const {
    for (const Entry& e : entries_)
        if (e.prefix && e.complete && !(e.has_chain && !delta_) && entry_files_valid(e) && e.cvec == cvec &&
            e.live_meta.ids == ids && e.live_meta.imgs == images) return true;
    return false;
}

size_t ConversationSpillCache::prefix_count() const {
    size_t n = 0;
    for (const Entry& e : entries_) n += e.prefix ? 1 : 0;
    return n;
}

bool ConversationSpillCache::remove_group(Entry& entry, uint64_t& removed, std::vector<std::string>& left) {
    const uint64_t accounted_before = entry.total_bytes();
    removed = 0;
    left.clear();
    auto finish = [&](bool complete) {
        if (removed > accounted_before || accounted_before - removed != entry.total_bytes())
            throw std::logic_error("spill entry removal accounting mismatch");
        return complete;
    };
    auto remove_one = [&](const std::string& path, uint64_t& accounted) {
        const PathInfo before = inspect_path_nofollow(path);
        if (!before.inspected) { left.push_back(path); return false; }
        if (!before.exists) {
            removed += accounted;
            accounted = 0;
            return true;
        }
        std::error_code ec;
        std::filesystem::remove(path, ec);  // removes a symlink itself; never follows its target
        const PathInfo after = inspect_path_nofollow(path);
        if (after.inspected && !after.exists) {
            removed += accounted;
            accounted = 0;
            return true;
        }
        left.push_back(path);
        return false;
    };
    auto retain_or_forget = [&](const std::string& path, uint64_t& accounted) {
        const PathInfo info = inspect_path_nofollow(path);
        if (info.inspected && !info.exists) {
            removed += accounted;
            accounted = 0;
        } else {
            left.push_back(path);
        }
    };

    // Remove data first, but keep the manifest and sidecar until both base and log are gone: they are the ownership
    // ledger needed to recognize and account a survivor after a restart.
    const bool session_gone = remove_one(entry.session_path(), entry.file_bytes);
    const bool log_gone = remove_one(entry.delta_path(), entry.delta_accounted_bytes);
    if (!session_gone || !log_gone) {
        retain_or_forget(entry.chain_path(), entry.chain_file_bytes);
        retain_or_forget(entry.meta_path(), entry.meta_bytes);
        entry.complete = false;
        return finish(false);
    }
    const bool chain_gone = remove_one(entry.chain_path(), entry.chain_file_bytes);
    if (!chain_gone) {
        retain_or_forget(entry.meta_path(), entry.meta_bytes);
        entry.complete = false;
        return finish(false);
    }
    const bool meta_gone = remove_one(entry.meta_path(), entry.meta_bytes);
    entry.complete = meta_gone && left.empty();
    if (entry.complete) {
        entry.has_chain = false;
        entry.delta_bytes = 0;
    }
    return finish(entry.complete);
}

bool ConversationSpillCache::erase(const std::string& path, std::string& error) {
    auto found = std::find_if(entries_.begin(), entries_.end(),
                              [&](const Entry& entry) { return entry.session_path() == path; });
    if (found == entries_.end()) return true;
    const uint64_t before = found->total_bytes();
    uint64_t removed = 0;
    std::vector<std::string> left;
    const bool complete = remove_group(*found, removed, left);
    if (removed > before) throw std::logic_error("spill entry removal exceeds its accounted bytes");
    adjust_bytes(before, before - removed);
    if (pinned_path_ == path) pinned_path_.clear();
    if (complete) {
        entries_.erase(found);
        return true;
    }
    for (const auto& p : left) error += (error.empty() ? "" : ", ") + p;
    error = "some spill files could not be removed: " + error;
    return false;
}

void ConversationSpillCache::pin(const std::string& path) { pinned_path_ = path; }

void ConversationSpillCache::unpin(const std::string& path) {
    if (pinned_path_ == path) { pinned_path_.clear(); enforce_budget(); }
}

size_t ConversationSpillCache::drop_superseded(const std::vector<int32_t>& ids,
                                               const std::vector<ConversationImageKey>& images,
                                               const std::vector<ConversationCheckpoint>& checkpoints, bool cvec,
                                               const std::string& keep) {
    size_t dropped = 0;
    for (size_t i = 0; i < entries_.size();) {
        Entry& entry = entries_[i];
        size_t deepest = 0;
        for (size_t length : entry.checkpoint_lengths) deepest = std::max(deepest, length);
        auto same_saved_prefix = [&](size_t length, const std::vector<int32_t>& candidate_ids,
                                     const std::vector<ConversationImageKey>& candidate_images) {
            if (length == 0 || candidate_ids.size() != length || length > entry.live_meta.ids.size() ||
                !std::equal(entry.live_meta.ids.begin(), entry.live_meta.ids.begin() + (std::ptrdiff_t) length,
                            candidate_ids.begin())) return false;
            size_t a = 0, b = 0;
            while ((a < entry.live_meta.imgs.size() && entry.live_meta.imgs[a].start < (int64_t) length) ||
                   (b < candidate_images.size() && candidate_images[b].start < (int64_t) length)) {
                if (a == entry.live_meta.imgs.size() || entry.live_meta.imgs[a].start >= (int64_t) length ||
                    b == candidate_images.size() || candidate_images[b].start >= (int64_t) length ||
                    !(entry.live_meta.imgs[a] == candidate_images[b])) return false;
                ++a;
                ++b;
            }
            return true;
        };
        bool held = same_saved_prefix(deepest, ids, images);
        for (const auto& checkpoint : checkpoints)
            if (!held && same_saved_prefix(deepest, checkpoint.ids, checkpoint.imgs)) held = true;
        if (!entry.prefix && !(entry.complete && entry.has_chain && !delta_) && entry.cvec == cvec && deepest && held &&
            entry.session_path() != pinned_path_ && (keep.empty() || entry.session_path() != keep)) {
            const uint64_t before = entry.total_bytes();
            uint64_t removed = 0;
            std::vector<std::string> left;
            const bool complete = remove_group(entry, removed, left);
            if (removed > before) throw std::logic_error("spill entry removal exceeds its accounted bytes");
            adjust_bytes(before, before - removed);
            if (!complete) { ++i; continue; }
            entries_.erase(entries_.begin() + (std::ptrdiff_t) i);
            ++dropped;
            continue;
        }
        ++i;
    }
    return dropped;
}

void ConversationSpillCache::enforce_budget() {
    if (!delta_) {
        // the baseline policy: first-spilled first out, ordinary conversations before prefix entries, the pinned one
        // (being restored) held - an engine without --conversation-cache-disk-delta keeps the pre-feature eviction
        // order, and the removal takes a delta chain's files together when one is there
        for (int pass = 0; pass < 2; ++pass)
            for (size_t i = 0; (bytes_ > budget_ || entries_.size() > kMaxSidecarEntries) && i < entries_.size();) {
                if (entries_[i].session_path() == pinned_path_ || entries_[i].prefix != (pass == 1) ||
                    (entries_[i].complete && entries_[i].has_chain && !delta_)) { ++i; continue; }
                const uint64_t before = entries_[i].total_bytes();
                uint64_t removed = 0;
                std::vector<std::string> left;
                const bool complete = remove_group(entries_[i], removed, left);
                if (removed > before) throw std::logic_error("spill entry removal exceeds its accounted bytes");
                adjust_bytes(before, before - removed);
                if (!complete) { ++i; continue; }
                entries_.erase(entries_.begin() + (std::ptrdiff_t) i);
                ++disk_evictions_;
            }
        return;
    }
    // Least recently used first.  A save or a successful restore stamps an entry (persisted in its sidecar, so the
    // order survives a restart); a v1 sidecar's unknown age (0) sorts as the oldest.  Ordinary conversations come
    // before prefix entries - a prefix serves every new conversation - and the pinned one (being restored) is held.
    // An entry whose files cannot be removed (in use) is not retried within this call.
    std::vector<size_t> tried;
    for (int pass = 0; pass < 2; ++pass) {
        for (;;) {
            if (bytes_ <= budget_ && entries_.size() <= kMaxSidecarEntries) return;
            size_t victim = entries_.size();
            for (size_t i = 0; i < entries_.size(); ++i) {
                const Entry& e = entries_[i];
                if (e.prefix != (pass == 1) || e.session_path() == pinned_path_ ||
                    (e.complete && e.has_chain && !delta_)) continue;
                if (std::find(tried.begin(), tried.end(), i) != tried.end()) continue;
                if (victim == entries_.size() || e.last_used < entries_[victim].last_used) victim = i;
            }
            if (victim == entries_.size()) break;   // nothing left this pass may remove
            const uint64_t before = entries_[victim].total_bytes();
            uint64_t removed = 0;
            std::vector<std::string> left;
            const bool complete = remove_group(entries_[victim], removed, left);
            if (removed > before) throw std::logic_error("spill entry removal exceeds its accounted bytes");
            adjust_bytes(before, before - removed);
            if (!complete) {
                tried.push_back(victim);
                continue;
            }
            entries_.erase(entries_.begin() + (std::ptrdiff_t) victim);
            ++disk_evictions_;
            tried.clear();   // the indices shifted
        }
    }
}

bool ConversationSpillCache::read_chain(const std::filesystem::path& chain, uint64_t base_size,
                                        const std::filesystem::path& dlog, uint64_t& committed,
                                        std::string& error) const {
    uint64_t recorded_base = 0;
    if (!read_chain_header(chain, recorded_base, committed, error)) return false;
    if (recorded_base != base_size) { error = "spill chain base size mismatch"; return false; }
    const PathInfo log = inspect_path_nofollow(dlog);
    if (!log.inspected || !log.regular || committed > log.bytes) {
        error = "spill chain log is missing, non-regular or shorter than its manifest";
        return false;
    }
    return true;
}

bool ConversationSpillCache::write_chain(const std::filesystem::path& chain, uint64_t base_size, uint64_t committed,
                                         std::string& error) const {
    uint8_t buf[kChainFile];
    std::memcpy(buf, kChainMagic, 8);
    const uint64_t version = kChainVersion, base = base_size, at = committed;
    std::memcpy(buf + 8, &version, 8);
    std::memcpy(buf + 16, &base, 8);
    std::memcpy(buf + 24, &at, 8);
    const uint64_t hash = session_hash64(buf, 32, 0);
    std::memcpy(buf + 32, &hash, 8);
    const std::filesystem::path tmp(chain.string() + ".tmp");
    // Called only after exclusive creation succeeds; a create failure leaves this fixed path unowned.
    auto cleanup_owned_temp = [&]() {
        std::error_code cleanup_error;
        const bool removed = std::filesystem::remove(tmp, cleanup_error);
        if (cleanup_error) {
            error += "; cannot clean up owned temp " + tmp.string() + ": " + cleanup_error.message();
        } else if (!removed) {
            const PathInfo remaining = inspect_path_nofollow(tmp);
            if (!remaining.inspected || remaining.exists)
                error += "; cannot clean up owned temp " + tmp.string();
        }
    };
    // created the way the session files and the sidecar's temp are (conversation_file.cpp RawFile::create_new,
    // write_sidecar): exclusive, never following a link, owner-only (0600) - a pre-existing anything at the fixed
    // temp name, including a symlink to someone else's file, is refused rather than truncated
#if defined(_WIN32)
    std::wstring w;
    if (!wide(tmp.string(), w)) { error = "cannot create the spill chain manifest: the path is not valid UTF-8"; return false; }
    const HANDLE h = CreateFileW(w.c_str(), GENERIC_WRITE, 0, nullptr, CREATE_NEW,
                                 FILE_ATTRIBUTE_NORMAL | FILE_FLAG_SEQUENTIAL_SCAN, nullptr);
    if (h == INVALID_HANDLE_VALUE) { error = "cannot create the spill chain manifest: " + last_error(); return false; }
    DWORD wrote = 0;
    const BOOL write_ok = WriteFile(h, buf, (DWORD) kChainFile, &wrote, nullptr);
    const DWORD write_error = !write_ok ? GetLastError() :
                              wrote != (DWORD) kChainFile ? ERROR_WRITE_FAULT : ERROR_SUCCESS;
    const BOOL close_ok = CloseHandle(h);
    const DWORD close_error = close_ok ? ERROR_SUCCESS : GetLastError();
    if (write_error != ERROR_SUCCESS || close_error != ERROR_SUCCESS) {
        const DWORD why = write_error != ERROR_SUCCESS ? write_error : close_error;
        error = "cannot write the spill chain manifest: " + last_error(why);
        cleanup_owned_temp();
        return false;
    }
#else
    const int fd = ::open(tmp.c_str(), O_WRONLY | O_CREAT | O_EXCL | O_NOFOLLOW | O_CLOEXEC, 0600);
    if (fd < 0) { error = std::string("cannot create the spill chain manifest: ") + std::strerror(errno); return false; }
    size_t off = 0;
    while (off < kChainFile) {
        const ssize_t n = ::write(fd, buf + off, kChainFile - off);
        if (n > 0) { off += (size_t) n; continue; }
        if (n < 0 && errno == EINTR) continue;
        error = std::string("cannot write the spill chain manifest: ") + (n < 0 ? std::strerror(errno) : "short write");
        ::close(fd);
        cleanup_owned_temp();
        return false;
    }
    if (::close(fd) != 0) {
        error = std::string("cannot write the spill chain manifest: ") + std::strerror(errno);
        cleanup_owned_temp();
        return false;
    }
#endif
    std::error_code ec;
    std::filesystem::rename(tmp, chain, ec);
    if (ec) {
        error = "cannot commit the spill chain manifest: " + ec.message();
        cleanup_owned_temp();
        return false;
    }
    return true;
}

bool ConversationSpillCache::reserve_delta(size_t index, ConversationSpillDeltaWrite& target, std::string& error) {
    target = {};
    if (!enabled_ || index >= entries_.size()) { error = "spill cache is disabled"; return false; }
    if (!delta_) { error = "delta chains are off"; return false; }
    Entry& entry = entries_[index];
    if (entry.prefix) { error = "a prefix entry has no delta chain"; return false; }
    // An entry of another engine configuration is a gate-approved READ-ONLY file: its K/V was computed under that
    // config and a record appended here would carry this config's bytes behind them, so the chain would no longer be
    // the one the ledger approved.  Such a conversation is saved as a new file under this cache's identity instead.
    if (entry.source_identity.model != identity_.model || entry.source_identity.config != identity_.config) {
        error = "a gate-approved entry of another engine configuration is read-only";
        return false;
    }
    if (!entry.complete || !entry_files_valid(entry)) {
        error = "spill entry is incomplete or its files are not valid regular files";
        return false;
    }
    target.index = index;
    target.stem = entry.stem;
    target.identity = identity_;
    target.file_bytes = entry.file_bytes;
    target.delta_bytes = entry.delta_bytes;
    target.delta_accounted_bytes = entry.delta_accounted_bytes;
    target.had_chain = entry.has_chain;
    target.valid = true;
    return true;
}

bool ConversationSpillCache::publish_delta(const ConversationSpillDeltaWrite& target, const SavedConversation& meta,
                                           size_t written, bool write_succeeded, std::string& error) {
    if (!enabled_ || !target.valid || target.index >= entries_.size()) {
        error = "spill delta reservation is no longer valid";
        return false;
    }
    Entry& entry = entries_[target.index];
    auto same_reservation = [&]() {
        return entry.stem == target.stem && entry.file_bytes == target.file_bytes &&
               entry.delta_bytes == target.delta_bytes &&
               entry.delta_accounted_bytes == target.delta_accounted_bytes &&
               entry.has_chain == target.had_chain && entry.complete &&
               target.identity.model == identity_.model && target.identity.config == identity_.config;
    };
    auto account_log_after_attempt = [&](std::string& account_error) {
        if (!same_reservation()) {
            account_error = "spill delta reservation changed before publication";
            return false;
        }
        const uint64_t before = entry.total_bytes();
        const PathInfo log = inspect_path_nofollow(entry.delta_path());
        if (!log.inspected) {
            entry.complete = false;
        } else if (log.exists) {
            entry.has_chain = true;
            if (log.regular) entry.delta_accounted_bytes = log.bytes;
            else entry.complete = false;
        }
        if (!target.had_chain && entry.has_chain) entry.complete = false;
        if (target.had_chain) {
            uint64_t still_committed = 0;
            std::string verify_error;
            if (!read_chain(entry.chain_path(), entry.file_bytes, entry.delta_path(), still_committed, verify_error) ||
                still_committed != entry.delta_bytes) entry.complete = false;
        }
        adjust_bytes(before, entry.total_bytes());
        return true;
    };
    if (!write_succeeded) {
        std::string account_error;
        if (!account_log_after_attempt(account_error)) error += "; " + account_error;
        return false;
    }
    std::string account_error;
    if (!account_log_after_attempt(account_error)) { error = account_error; return false; }
    if (written > UINT64_MAX - entry.delta_bytes) {
        entry.complete = false;
        error = "delta record length overflows the committed chain";
        return false;
    }
    const uint64_t committed = entry.delta_bytes + written;
    if (!write_chain(entry.chain_path(), entry.file_bytes, committed, error)) return false;
    const PathInfo chain = inspect_path_nofollow(entry.chain_path());
    if (!chain.inspected || !chain.regular) {
        entry.has_chain = true;
        entry.complete = false;
        error = "published spill chain manifest is not a regular file";
        return false;
    }
    {
        const uint64_t before = entry.total_bytes();
        entry.has_chain = true;
        entry.delta_bytes = committed;
        entry.chain_file_bytes = chain.bytes;
        entry.complete = false;   // until the matching sidecar generation is published
        adjust_bytes(before, entry.total_bytes());
    }

    // The sidecar describes the state the chain now holds; the read pass double-checks its tail before restore.
    Entry updated = entry;
    updated.live_meta.ids = meta.live.ids;
    updated.live_meta.imgs = meta.live.imgs;
    updated.checkpoint_lengths.clear();
    for (const auto& c : meta.checkpoints) updated.checkpoint_lengths.push_back(c.ids.size());
    updated.last_used = now_ms();
    if (!write_sidecar(entry.meta_path(), updated, error, true)) return false;
    const PathInfo sidecar = inspect_path_nofollow(entry.meta_path());
    if (!sidecar.inspected || !sidecar.regular) {
        error = "published spill sidecar is not a regular file";
        return false;
    }
    {
        const uint64_t before = entry.total_bytes();
        entry.live_meta.ids = std::move(updated.live_meta.ids);
        entry.live_meta.imgs = std::move(updated.live_meta.imgs);
        entry.checkpoint_lengths = std::move(updated.checkpoint_lengths);
        entry.last_used = updated.last_used;
        entry.meta_bytes = sidecar.bytes;
        entry.complete = true;
        adjust_bytes(before, entry.total_bytes());
    }
    enforce_budget();
    return true;
}

bool ConversationSpillCache::spill_delta(size_t index, const SavedConversation& meta,
                                         const std::vector<SessionKvSource>& kv, int64_t first_token,
                                         size_t& bytes, std::string& error, bool durable, bool state_codec_rht,
                                         SessionWriteTimings* timings, bool layout_v3) {
    bytes = 0;
    try {
        ConversationSpillDeltaWrite target;
        if (!reserve_delta(index, target, error)) return false;
        SessionWriteOptions opt;
        opt.durable = durable;
        opt.state_codec_rht = state_codec_rht;
        opt.timings = timings;
        opt.layout_v3 = layout_v3;
        SessionStatus st;
        uint64_t written = 0;
        const bool ok = session_delta_write(target.delta_path(), target.delta_bytes, meta, kv, first_token,
                                            target.identity, written, error, opt, &st);
        if (!publish_delta(target, meta, (size_t) written, ok, error)) return false;
        bytes = (size_t) written;
        return true;
    } catch (const std::bad_alloc&) {
        error = "not enough RAM to index the delta record";
        return false;
    }
}

bool ConversationSpillCache::entry_source_identity(size_t index, SessionFileIdentity& out) const {
    if (index >= entries_.size()) return false;
    out = entries_[index].source_identity;
    return true;
}

size_t ConversationSpillCache::gate_approved_count() const {
    size_t n = 0;
    for (const Entry& entry : entries_)
        if (entry.source_identity.model != identity_.model || entry.source_identity.config != identity_.config) ++n;
    return n;
}

bool ConversationSpillCache::gate_identity_approved(const SessionFileIdentity& source) const {
    if (!gate_ || gate_engine_build_.empty()) return false;
    for (const auto& b : gate_batches_) {
        if (!b.passed || b.engine_build != gate_engine_build_) continue;
        if (b.source.model != source.model || b.source.config != source.config) continue;
        if (b.target.model != identity_.model || b.target.config != identity_.config) continue;
        return true;
    }
    return false;
}

bool ConversationSpillCache::gate_entry_approved(const std::string& stem, const SessionFileIdentity& source,
                                                 uint64_t file_bytes, int64_t tokens) const {
    return gate_ && session_spill_gate_approves(gate_batches_, gate_entries_, stem, source, identity_,
                                                gate_engine_build_, file_bytes, tokens);
}

size_t ConversationSpillCache::find(const std::string& path) const {
    for (size_t i = 0; i < entries_.size(); ++i) {
        const Entry& entry = entries_[i];
        if (entry.session_path() == path && entry.complete && !(entry.has_chain && !delta_) &&
            entry_files_valid(entry)) return i;
    }
    return entries_.size();
}

void ConversationSpillCache::touch(const std::string& path) {
    if (!enabled_ || !delta_) return;   // the baseline v1 sidecar carries no stamp to persist
    const size_t index = find(path);
    if (index >= entries_.size()) return;
    Entry& entry = entries_[index];
    const uint64_t stamp = now_ms();
    if (stamp <= entry.last_used) return;   // same millisecond or a backwards clock: nothing to record
    entry.last_used = stamp;
    std::string error;
    // Best effort: a failed save of the stamp keeps the in-memory order for this run and the next save re-stamps it;
    // the restore itself must not fail for a sidecar write.  touch() only runs with delta on, so the stamped v2
    // sidecar is written; the stamp's decimal growth can change the file by a few bytes - keep the budget accounting
    // exact.
    const uint64_t before = entry.total_bytes();
    write_sidecar(entry.meta_path(), entry, error, true);
    const PathInfo meta = inspect_path_nofollow(entry.meta_path());
    if (meta.inspected && meta.regular) entry.meta_bytes = meta.bytes;
    else entry.complete = false;
    if (entry.has_chain) {
        const PathInfo chain = inspect_path_nofollow(entry.chain_path());
        if (chain.inspected && chain.regular) entry.chain_file_bytes = chain.bytes;
        else entry.complete = false;
    }
    adjust_bytes(before, entry.total_bytes());
}

uint64_t ConversationSpillCache::entry_last_used(size_t index) const {
    return index < entries_.size() ? entries_[index].last_used : 0;
}

bool ConversationSpillCache::entry_info(size_t index, int64_t& tokens, int64_t& deepest, uint64_t& base_bytes,
                                        uint64_t& delta_bytes, bool& prefix, std::string& path) const {
    if (index >= entries_.size()) return false;
    const Entry& e = entries_[index];
    if (!e.complete || (e.has_chain && !delta_) || !entry_files_valid(e)) return false;
    tokens = (int64_t) e.live_meta.ids.size();
    deepest = 0;
    for (size_t n : e.checkpoint_lengths) deepest = std::max(deepest, (int64_t) n);
    base_bytes = e.file_bytes;
    delta_bytes = e.delta_bytes;
    prefix = e.prefix;
    path = e.session_path();
    return true;
}

bool ConversationSpillCache::entry_matches_prefix(size_t index, const std::vector<int32_t>& ids, size_t n,
                                                   const std::vector<ConversationImageKey>& images) const {
    if (index >= entries_.size() || !entries_[index].complete ||
        (entries_[index].has_chain && !delta_) || !entry_files_valid(entries_[index])) return false;
    const ConversationCheckpoint& c = entries_[index].live_meta;
    if (n == 0 || n > c.ids.size() || n > ids.size()) return false;
    if (!std::equal(c.ids.begin(), c.ids.begin() + (std::ptrdiff_t) n, ids.begin())) return false;
    size_t a = 0, b = 0;
    while ((a < c.imgs.size() && c.imgs[a].start < (int64_t) n) || (b < images.size() && images[b].start < (int64_t) n)) {
        if (a == c.imgs.size() || c.imgs[a].start >= (int64_t) n ||
            b == images.size() || images[b].start >= (int64_t) n || !(c.imgs[a] == images[b])) return false;
        ++a;
        ++b;
    }
    return true;
}

bool ConversationSpillCache::entry_chain_tail_matches(size_t index, const std::vector<int32_t>& tail_ids,
                                                      const std::vector<ConversationImageKey>& tail_imgs,
                                                      const std::vector<size_t>& tail_checkpoint_lengths) const {
    if (index >= entries_.size() || entries_[index].prefix || !entries_[index].has_chain ||
        !entries_[index].complete || !entry_files_valid(entries_[index])) return false;
    const Entry& e = entries_[index];
    return e.live_meta.ids == tail_ids && e.live_meta.imgs == tail_imgs &&
           e.checkpoint_lengths == tail_checkpoint_lengths;
}

} // namespace strata::core
