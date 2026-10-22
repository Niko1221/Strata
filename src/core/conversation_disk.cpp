// The conversation cache's disk tier (include/strata/core/conversation_disk.hpp, docs/CONVERSATION_CACHE_DISK.md).
// Stage files are session files (conversation_file.cpp); this file adds the per-conversation index file, the folder
// scan at start-up, prefix matching by token hashes and least-recently-used retention.
#include "strata/core/conversation_disk.hpp"

#include <algorithm>
#include <bit>
#include <cerrno>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <limits>
#include <new>
#include <random>
#include <string>
#include <system_error>
#ifndef _WIN32
#include <fcntl.h>
#include <unistd.h>
#endif

static_assert(std::endian::native == std::endian::little, "conversation disk index files are little-endian only");

namespace strata::core {
namespace {
namespace fs = std::filesystem;

constexpr char kIndexMagic[8] = {'S', 'T', 'R', 'C', 'D', 'X', '\x01', '\0'};
constexpr uint32_t kIndexVersion = 1;
constexpr uint64_t kIndexSeed = 0x5354524344583031ull;   // "STRCDX01"
constexpr uint64_t kIdsSeed = 0x5354524944533031ull;     // "STRIDS01"
constexpr uint64_t kLayoutSeed = 0x53545243444c3031ull;  // "STRCDL01"
constexpr size_t kIndexMaxBytes = 64u << 20;             // far above any real index (a few KiB)
constexpr uint64_t kMaxPoints = 4096, kMaxImages = 1u << 20, kMaxStages = 64;

uint64_t random_id() {
    static std::mt19937_64 rng{[] {
        std::random_device rd;
        return ((uint64_t) rd() << 32) ^ rd() ^ (uint64_t) std::chrono::steady_clock::now().time_since_epoch().count();
    }()};
    uint64_t v = 0;
    while (v == 0 || v == UINT64_MAX) v = rng();   // 0 is "none", UINT64_MAX the caller's "failed" mark
    return v;
}

std::string hex16(uint64_t v) {
    char b[17];
    std::snprintf(b, sizeof b, "%016llx", (unsigned long long) v);
    return b;
}

bool parse_hex16(const std::string& s, uint64_t& v) {
    if (s.size() != 16) return false;
    v = 0;
    for (char c : s) {
        int d = c >= '0' && c <= '9' ? c - '0' : c >= 'a' && c <= 'f' ? c - 'a' + 10 : -1;
        if (d < 0) return false;
        v = (v << 4) | (uint64_t) d;
    }
    return v != 0;
}

// ---- the index file: little-endian, fixed-width
//   magic[8] version:u32 reserved:u32 model:u64 config:u64 layout:u64 id:u64 used:u64 cvec:u64
//   points:u64 { tokens:i64 hash:u64 live:u64 imgs:u64 { start:i64 hash:u64 }* }*
//   files:u64 { bytes:u64 }*
//   hash:u64 (session_hash64 of every byte before it, seed kIndexSeed)
struct Writer {
    std::vector<uint8_t> b;
    void raw(const void* p, size_t n) { const uint8_t* q = (const uint8_t*) p; b.insert(b.end(), q, q + n); }
    void u64(uint64_t v) { raw(&v, 8); }
    void i64(int64_t v) { raw(&v, 8); }
};

struct Reader {
    const std::vector<uint8_t>& b;
    size_t at = 0;
    bool u64(uint64_t& v) {
        if (b.size() - at < 8) return false;
        std::memcpy(&v, b.data() + at, 8);
        at += 8;
        return true;
    }
    bool i64(int64_t& v) { uint64_t u = 0; if (!u64(u)) return false; v = (int64_t) u; return true; }
};

bool read_small(const fs::path& p, std::vector<uint8_t>& out) {
    std::error_code ec;
    const auto size = fs::file_size(p, ec);
    if (ec || size > kIndexMaxBytes) return false;
    std::FILE* f = std::fopen(p.string().c_str(), "rb");
    if (!f) return false;
    out.resize((size_t) size);
    const bool ok = std::fread(out.data(), 1, out.size(), f) == out.size();
    std::fclose(f);
    return ok;
}

// temporary name beside `path`, flushed, renamed over it, folder flushed (POSIX)
bool write_small(const fs::path& path, const std::vector<uint8_t>& data, bool durable, std::string& error) {
    const fs::path tmp = path.parent_path() / ("." + path.filename().string() + "." + hex16(random_id()) + ".tmp");
    std::FILE* f = std::fopen(tmp.string().c_str(), "wb");
    if (!f) { error = "conversation disk: cannot create " + tmp.string() + ": " + std::strerror(errno); return false; }
    bool ok = std::fwrite(data.data(), 1, data.size(), f) == data.size() && std::fflush(f) == 0;
#ifndef _WIN32
    if (ok && durable) ok = ::fsync(::fileno(f)) == 0;
#endif
    const int saved = errno;
    ok = std::fclose(f) == 0 && ok;
    std::error_code ec;
    if (!ok) {
        error = "conversation disk: cannot write " + tmp.string() + ": " + std::strerror(saved);
        fs::remove(tmp, ec);
        return false;
    }
    fs::rename(tmp, path, ec);
    if (ec) {
        error = "conversation disk: cannot rename " + tmp.string() + ": " + ec.message();
        fs::remove(tmp, ec);
        return false;
    }
#ifndef _WIN32
    if (durable) {
        const int d = ::open(path.parent_path().string().c_str(), O_RDONLY | O_DIRECTORY | O_CLOEXEC);
        if (d >= 0) { (void) ::fsync(d); ::close(d); }   // EINVAL on some filesystems: not a failure (as for SAVE)
    }
#endif
    return true;
}

// The pictures below `n` tokens, in prompt order (what conversation_prefix compares a checkpoint's against).
std::vector<ConversationImageKey> images_below(const std::vector<ConversationImageKey>& images, int64_t n) {
    std::vector<ConversationImageKey> out;
    for (const auto& im : images) if (im.start < n) out.push_back(im);
    return out;
}

ConversationDiskPoint point_of(const ConversationCheckpoint& c, bool live) {
    ConversationDiskPoint p;
    p.tokens = (int64_t) c.ids.size();
    p.hash = conversation_disk_ids_hash(c.ids.data(), c.ids.size());
    p.imgs = c.imgs;
    p.live = live;
    return p;
}

const ConversationDiskPoint* deepest_checkpoint(const std::vector<ConversationDiskPoint>& points) {
    const ConversationDiskPoint* d = nullptr;
    for (const auto& p : points)
        if (!p.live && (!d || p.tokens > d->tokens)) d = &p;
    return d;
}

bool same_point(const ConversationDiskPoint& a, const ConversationDiskPoint& b) {
    return a.tokens == b.tokens && a.hash == b.hash && a.imgs == b.imgs;
}
} // namespace

uint64_t ConversationDiskLayout::key() const {
    SessionHasher h(kLayoutSeed);
    const uint64_t n = stages.size();
    h.update(&n, 8);
    for (const auto& s : stages) { h.update(&s.first, 8); h.update(&s.second, 8); }
    return h.digest();
}

uint64_t conversation_disk_ids_hash(const int32_t* ids, size_t n) {
    return session_hash64(ids, n * sizeof(int32_t), kIdsSeed);
}

std::vector<ConversationDiskPoint> conversation_disk_points(const SavedConversation& image) {
    std::vector<ConversationDiskPoint> out;
    out.reserve(1 + image.checkpoints.size());
    out.push_back(point_of(image.live, true));
    for (const auto& c : image.checkpoints) out.push_back(point_of(c, false));
    return out;
}

std::string ConversationDiskTier::path_of(uint64_t id, const char* suffix) const {
    return (fs::path(folder_) / (hex16(id) + suffix)).string();
}

std::string ConversationDiskTier::stage_path(uint64_t id, size_t stage) const {
    return path_of(id, (".s" + std::to_string(stage) + ".strses").c_str());
}

const ConversationDiskTier::Entry* ConversationDiskTier::find(uint64_t id) const {
    for (const auto& e : entries_) if (e.id == id) return &e;
    return nullptr;
}

ConversationDiskTier::Entry* ConversationDiskTier::find_mut(uint64_t id) {
    for (auto& e : entries_) if (e.id == id) return &e;
    return nullptr;
}

bool ConversationDiskTier::write_index(const Entry& e, std::string& error) const {
    Writer w;
    w.raw(kIndexMagic, 8);
    const uint32_t version = kIndexVersion, reserved = 0;
    w.raw(&version, 4); w.raw(&reserved, 4);
    w.u64(identity_.model); w.u64(identity_.config); w.u64(layout_.key());
    w.u64(e.id); w.u64(e.used); w.u64(e.cvec ? 1 : 0);
    w.u64(e.points.size());
    for (const auto& p : e.points) {
        w.i64(p.tokens); w.u64(p.hash); w.u64(p.live ? 1 : 0);
        w.u64(p.imgs.size());
        for (const auto& im : p.imgs) { w.i64(im.start); w.u64(im.hash); }
    }
    w.u64(e.files.size());
    for (uint64_t f : e.files) w.u64(f);
    w.u64(session_hash64(w.b.data(), w.b.size(), kIndexSeed));
    return write_small(path_of(e.id, ".idx"), w.b, options_.durable, error);
}

void ConversationDiskTier::remove_files(uint64_t id, size_t stages) const {
    std::error_code ec;
    fs::remove(path_of(id, ".idx"), ec);   // the index first: a crash after it leaves orphans, never a broken entry
    for (size_t k = 0; k < stages; ++k) fs::remove(stage_path(id, k), ec);
}

bool ConversationDiskTier::open(const ConversationDiskOptions& options, const SessionFileIdentity& identity,
                                const ConversationDiskLayout& layout, std::string& error, OpenReport* report) {
    open_ = false;
    entries_.clear();
    bytes_ = clock_ = 0;
    try {
        open_ = open_impl(options, identity, layout, error, report);
    } catch (const std::exception& e) {   // a filesystem_error from a directory walk, a bad_alloc
        error = std::string("conversation disk: ") + e.what();
        entries_.clear();
        bytes_ = 0;
        open_ = false;
    }
    return open_;
}

bool ConversationDiskTier::open_impl(const ConversationDiskOptions& options, const SessionFileIdentity& identity,
                                     const ConversationDiskLayout& layout, std::string& error, OpenReport* report) {
    OpenReport rep;
    if (options.dir.empty()) { error = "conversation disk: no folder"; return false; }
    if (options.budget_bytes == 0) { error = "conversation disk: a zero byte budget"; return false; }
    if (layout.stages.empty() || layout.stages.size() > kMaxStages) { error = "conversation disk: invalid stage layout"; return false; }
    options_ = options;
    identity_ = identity;
    layout_ = layout;
    const std::string leaf = hex16(identity.model) + "-" + hex16(identity.config) + "-" + hex16(layout.key());
    folder_ = (fs::path(options.dir) / leaf).string();
    std::error_code ec;
    fs::create_directories(folder_, ec);
    if (ec || !fs::is_directory(folder_, ec)) {
        error = "conversation disk: cannot create " + folder_ + (ec ? ": " + ec.message() : std::string());
        return false;
    }
#ifndef _WIN32
    fs::permissions(folder_, fs::perms::owner_all, fs::perm_options::replace, ec);   // 0700: the user's own state
#endif
    // the other identities' and layouts' folders: reported, never touched (another engine may use them)
    for (const auto& d : fs::directory_iterator(options.dir)) {
        std::error_code e2;
        if (!d.is_directory(e2) || d.path().filename().string() == leaf) continue;
        ++rep.other_folders;
        for (const auto& f : fs::recursive_directory_iterator(d.path(), fs::directory_options::skip_permission_denied))
            if (f.is_regular_file(e2)) {
                const auto size = f.file_size(e2);
                if (!e2) rep.other_bytes += size;
            }
    }
    // pass 1: the index files
    std::vector<std::string> stray;
    for (const auto& d : fs::directory_iterator(folder_)) {
        std::error_code e2;
        const std::string name = d.path().filename().string();
        if (!d.is_regular_file(e2)) continue;
        if (!name.empty() && name[0] == '.' && name.size() > 4 && name.compare(name.size() - 4, 4, ".tmp") == 0) {
            stray.push_back(d.path().string());   // an unfinished write's temporary file
            continue;
        }
        if (name.size() != 20 || name.compare(16, 4, ".idx") != 0) continue;
        uint64_t id = 0;
        if (!parse_hex16(name.substr(0, 16), id)) { stray.push_back(d.path().string()); continue; }
        Entry e;
        if (!parse_index(d.path().string(), id, e)) {
            // not this runtime's, or damaged: its files go (the folder is this identity's and layout's alone)
            remove_files(id, layout.stages.size());
            ++rep.dropped;
            continue;
        }
        clock_ = std::max(clock_, e.used);
        bytes_ += e.bytes;
        entries_.push_back(std::move(e));
        ++rep.kept;
    }
    // pass 2: stage files no index names (a write that did not finish)
    for (const auto& d : fs::directory_iterator(folder_)) {
        std::error_code e2;
        const std::string name = d.path().filename().string();
        if (!d.is_regular_file(e2) || name.size() < 16 + 10 || name.compare(16, 2, ".s") != 0) continue;
        uint64_t id = 0;
        if (!parse_hex16(name.substr(0, 16), id) || find(id) == nullptr) stray.push_back(d.path().string());
    }
    for (const auto& p : stray) { std::error_code e4; if (fs::remove(p, e4)) ++rep.orphans; }
    std::sort(entries_.begin(), entries_.end(), [](const Entry& a, const Entry& b) { return a.used < b.used; });
    open_ = true;
    if (!make_room(0, error)) return false;   // (cannot fail with nothing incoming; a smaller budget evicts)
    if (report) *report = rep;
    return true;
}

bool ConversationDiskTier::parse_index(const std::string& path, uint64_t id, Entry& e) const {
    std::vector<uint8_t> b;
    if (!read_small(path, b) || b.size() < 16 + 8 * 8) return false;
    uint64_t stored = 0;
    std::memcpy(&stored, b.data() + b.size() - 8, 8);
    if (std::memcmp(b.data(), kIndexMagic, 8) != 0 || stored != session_hash64(b.data(), b.size() - 8, kIndexSeed))
        return false;
    uint32_t version = 0;
    std::memcpy(&version, b.data() + 8, 4);
    Reader r{b, 16};
    uint64_t model = 0, config = 0, lkey = 0, rid = 0, cvec = 0, np = 0, nf = 0;
    if (version != kIndexVersion || !r.u64(model) || !r.u64(config) || !r.u64(lkey) || !r.u64(rid) ||
        !r.u64(e.used) || !r.u64(cvec) || model != identity_.model || config != identity_.config ||
        lkey != layout_.key() || rid != id || cvec > 1 || !r.u64(np) || np < 1 || np > kMaxPoints)
        return false;
    e.id = id;
    e.cvec = cvec == 1;
    for (uint64_t i = 0; i < np; ++i) {
        ConversationDiskPoint p;
        uint64_t live = 0, ni = 0;
        if (!r.i64(p.tokens) || !r.u64(p.hash) || !r.u64(live) || live > 1 || !r.u64(ni) || ni > kMaxImages ||
            p.tokens <= 0 || ni > (b.size() - r.at) / 16)
            return false;
        p.live = live == 1;
        p.imgs.resize((size_t) ni);
        for (auto& k : p.imgs) if (!r.i64(k.start) || !r.u64(k.hash)) return false;
        e.points.push_back(std::move(p));
    }
    if (!r.u64(nf) || nf != layout_.stages.size()) return false;
    for (uint64_t k = 0; k < nf; ++k) {
        uint64_t size = 0;
        std::error_code ec;
        if (!r.u64(size)) return false;
        const auto have = fs::file_size(stage_path(id, (size_t) k), ec);
        if (ec || have != size) return false;   // a stage file missing or cut short
        e.files.push_back(size);
        e.bytes += size;
    }
    return r.at + 8 == b.size() && e.points.front().live;
}

bool ConversationDiskTier::make_room(uint64_t incoming, std::string& error, uint64_t keep) {
    if (incoming > options_.budget_bytes) {
        error = "conversation disk: the conversation (" + std::to_string(incoming >> 20) +
                " MiB) is larger than the disk budget";
        return false;
    }
    // the least recently used goes first; the conversation being read (load) and `keep` (just written) stay
    while (bytes_ > options_.budget_bytes - incoming || entries_.size() + (incoming ? 1 : 0) > options_.max_entries) {
        size_t v = entries_.size();
        for (size_t i = 0; i < entries_.size(); ++i)
            if (entries_[i].id != loading_ && entries_[i].id != keep && (v == entries_.size() || entries_[i].used < entries_[v].used))
                v = i;
        if (v == entries_.size()) break;
        drop(entries_[v].id);
        ++evictions_;
    }
    return true;
}

ConversationDiskTier::Match ConversationDiskTier::best_ids(const std::vector<int32_t>& prompt,
                                                           const std::vector<ConversationImageKey>& images,
                                                           bool cvec) const {
    Match best;
    if (!open_ || prompt.size() < 2) return best;
    // the lengths worth hashing: deeper than nothing, short of the prompt's last token, pictures as the prompt's
    std::vector<int64_t> lens;
    for (const auto& e : entries_) {
        if (e.cvec != cvec || e.failed) continue;
        for (const auto& p : e.points)
            if (p.tokens > 0 && (size_t) p.tokens < prompt.size() && images_below(images, p.tokens) == p.imgs)
                lens.push_back(p.tokens);
    }
    if (lens.empty()) return best;
    std::sort(lens.begin(), lens.end());
    lens.erase(std::unique(lens.begin(), lens.end()), lens.end());
    // one streaming pass over the prompt: the digest at each length equals the one-shot hash of that prefix
    std::vector<std::pair<int64_t, uint64_t>> at;
    at.reserve(lens.size());
    SessionHasher h(kIdsSeed);
    size_t done = 0;
    for (int64_t L : lens) {
        h.update(prompt.data() + done, ((size_t) L - done) * sizeof(int32_t));
        done = (size_t) L;
        at.emplace_back(L, h.digest());
    }
    auto hash_at = [&](int64_t L, uint64_t& out) {
        const auto it = std::lower_bound(at.begin(), at.end(), std::make_pair(L, uint64_t(0)),
                                         [](const auto& a, const auto& b) { return a.first < b.first; });
        if (it == at.end() || it->first != L) return false;
        out = it->second;
        return true;
    };
    // most recently used last: >= lets a later (more recent) entry win a tie
    for (const auto& e : entries_) {
        if (e.cvec != cvec || e.failed) continue;
        for (const auto& p : e.points) {
            uint64_t hp = 0;
            if (p.tokens < best.tokens || !hash_at(p.tokens, hp) || hp != p.hash) continue;
            if (images_below(images, p.tokens) != p.imgs) continue;
            if (p.tokens > best.tokens || (p.tokens == best.tokens && (p.live || !best.live)))
                best = {e.id, p.tokens, p.live};
        }
    }
    return best;
}

uint64_t ConversationDiskTier::holding(const SavedConversation& image) const {
    if (!open_) return 0;
    const auto points = conversation_disk_points(image);
    for (const auto& e : entries_)
        if (e.cvec == image.cvec && e.points.size() == points.size() &&
            std::equal(points.begin(), points.end(), e.points.begin(), same_point))
            return e.id;
    return 0;
}

size_t ConversationDiskTier::drop_superseded(const SavedConversation& image, uint64_t keep) {
    if (!open_) return 0;
    const auto points = conversation_disk_points(image);
    std::vector<uint64_t> gone;
    for (const auto& e : entries_) {
        if (e.cvec != image.cvec || e.id == loading_ || e.id == keep) continue;
        const ConversationDiskPoint* deepest = deepest_checkpoint(e.points);
        if (!deepest) continue;
        for (const auto& p : points)
            if (same_point(p, *deepest)) { gone.push_back(e.id); break; }
    }
    for (uint64_t id : gone) drop(id);
    return gone.size();
}

void ConversationDiskTier::mark_failed(uint64_t id) {
    if (Entry* e = find_mut(id)) e->failed = true;
}

bool ConversationDiskTier::drop(uint64_t id) {
    for (size_t i = 0; i < entries_.size(); ++i) {
        if (entries_[i].id != id) continue;
        remove_files(id, entries_[i].files.size());
        bytes_ -= std::min(bytes_, entries_[i].bytes);
        entries_.erase(entries_.begin() + (std::ptrdiff_t) i);
        return true;
    }
    return false;
}

size_t ConversationDiskTier::clear() {
    const size_t n = entries_.size();
    while (!entries_.empty()) drop(entries_.back().id);
    bytes_ = 0;
    return n;
}

bool ConversationDiskTier::spill(const SavedConversation& image, uint64_t& id, std::string& error,
                                 SessionStatus* status, const SessionProgress& progress, const SessionPhase& phase) {
    SessionStatus st;
    auto fail = [&](const std::string& why, SessionError kind = SessionError::invalid) {
        error = why;
        st.error = kind;
        if (status) *status = st;
        return false;
    };
    id = 0;
    if (!open_) return fail("conversation disk: off");
    const size_t n_files = 1 + image.stage_images.size();
    if (n_files != layout_.stages.size()) return fail("conversation disk: the image has another stage count");
    for (size_t k = 0; k < n_files; ++k) {
        const SavedConversation& part = k == 0 ? image : image.stage_images[k - 1];
        if (part.layer_lo != layout_.stages[k].first || part.layer_hi != layout_.stages[k].second)
            return fail("conversation disk: the image has another layer range");
        if (!part.stage_images.empty() && k != 0) return fail("conversation disk: nested stage images");
    }
    if (image.live.ids.empty()) return fail("conversation disk: an empty conversation");
    if (const uint64_t kept = holding(image)) {   // already there (written when idle, or read back unchanged)
        id = kept;
        if (Entry* e = find_mut(kept)) {
            e->used = ++clock_;
            std::string ignored;
            (void) write_index(*e, ignored);
            std::stable_sort(entries_.begin(), entries_.end(), [](const Entry& a, const Entry& b) { return a.used < b.used; });
        }
        if (status) *status = st;
        return true;
    }
    // nothing is deleted before the new copy is committed (a failed write keeps the older copies): the budget may
    // run over by this one conversation until make_room below
    if ((uint64_t) image.bytes() > options_.budget_bytes) {
        if (!make_room(image.bytes(), error)) { st.error = SessionError::storage; if (status) *status = st; return false; }
    }
    Entry e;
    e.id = random_id();
    while (find(e.id) != nullptr) e.id = random_id();
    e.cvec = image.cvec;
    e.points = conversation_disk_points(image);
    SessionWriteOptions wo;
    wo.min_free_bytes = options_.min_free_bytes;
    wo.durable = options_.durable;
    wo.progress = progress;
    wo.phase = phase;
    for (size_t k = 0; k < n_files; ++k) {
        const SavedConversation& part = k == 0 ? image : image.stage_images[k - 1];
        size_t written = 0;
        // stage 0's stage_images are not part of format v1's payload: each stage is its own file
        if (!session_file_write(stage_path(e.id, k), part, identity_, written, error, wo, &st)) {
            remove_files(e.id, n_files);
            if (status) *status = st;
            return false;
        }
        e.files.push_back(written);
        e.bytes += written;
    }
    e.used = ++clock_;
    if (!write_index(e, error)) {
        remove_files(e.id, n_files);
        return fail(error, SessionError::io);
    }
    bytes_ += e.bytes;
    id = e.id;
    entries_.push_back(std::move(e));
    ++writes_;
    drop_superseded(image, id);
    std::string ignored;
    (void) make_room(0, ignored, id);
    if (status) *status = st;
    return true;
}

bool ConversationDiskTier::load(uint64_t id, const std::vector<SessionReadLimits>& limits, SavedConversation& out,
                                std::string& error, SessionStatus* status) {
    SessionStatus st;
    Entry* e = find_mut(id);
    if (!open_ || e == nullptr) {
        error = "conversation disk: no such conversation";
        st.error = SessionError::invalid;
        if (status) *status = st;
        return false;
    }
    const size_t n_files = e->files.size();
    // while its files are read, the RAM preflight may evict parked conversations to the disk (a write that may
    // evict here too): this one is not the victim
    struct Loading {
        uint64_t& slot;
        ~Loading() { slot = 0; }
    } loading{loading_};
    loading_ = id;
    std::vector<SavedConversation> parts(n_files);
    for (size_t k = 0; k < n_files; ++k) {
        size_t read = 0;
        const SessionReadLimits none;
        if (!session_file_read(stage_path(id, k), identity_, parts[k], read, error, k < limits.size() ? limits[k] : none,
                               &st)) {
            if (st.error == SessionError::invalid) mark_failed(id);   // this run cannot use it: keep, do not offer again
            if (status) *status = st;
            return false;
        }
    }
    // (a write during the reads may have moved the entries)
    e = find_mut(id);
    if (e == nullptr) {
        error = "conversation disk: the conversation was dropped while it was read";
        st.error = SessionError::io;
        if (status) *status = st;
        return false;
    }
    // every stage holds the same conversation, the one the index describes
    const auto points = conversation_disk_points(parts[0]);
    bool same = parts[0].cvec == e->cvec && points.size() == e->points.size() &&
                std::equal(points.begin(), points.end(), e->points.begin(), same_point);
    for (size_t k = 0; same && k < n_files; ++k)
        same = parts[k].layer_lo == layout_.stages[k].first && parts[k].layer_hi == layout_.stages[k].second &&
               (k == 0 || (parts[k].live.ids == parts[0].live.ids &&
                           parts[k].checkpoints.size() == parts[0].checkpoints.size()));
    for (size_t k = 1; same && k < n_files; ++k)
        for (size_t c = 0; same && c < parts[0].checkpoints.size(); ++c)
            same = parts[k].checkpoints[c].ids == parts[0].checkpoints[c].ids;
    if (!same) {
        drop(id);
        error = "conversation disk: the files do not hold the conversation their index names";
        st.error = SessionError::invalid;
        if (status) *status = st;
        return false;
    }
    SavedConversation image = std::move(parts[0]);
    for (size_t k = 1; k < n_files; ++k) image.stage_images.push_back(std::move(parts[k]));
    image.disk_id = id;
    out = std::move(image);
    e->used = ++clock_;
    std::string ignored;
    (void) write_index(*e, ignored);   // the new LRU stamp; a failure only loses the order after a restart
    // keep entries_ least recently used first
    std::stable_sort(entries_.begin(), entries_.end(), [](const Entry& a, const Entry& b) { return a.used < b.used; });
    ++reads_;
    if (status) *status = st;
    return true;
}

} // namespace strata::core
