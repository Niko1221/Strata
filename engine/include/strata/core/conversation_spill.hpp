#pragma once
// include/strata/core/conversation_spill.hpp - the SSD tier of the conversation cache (phase P2 of the admission design,
// RESULTS-ENG-1116.md): where a parked conversation is written when RAM refuses it, and how much may go there.
//
// Ownership: every engine process writes into a folder of its own, `<root>/<pid>-<16 hex nonce>`, and holds a lock on
// `<that folder>/.lock` for its whole life (POSIX flock on an open file; Windows: the file opened with no sharing).  At start a
// process removes the folders under `<root>` whose lock nobody holds - and nothing else: only folders named exactly that way,
// never a link, never a file of the root, never another process's live folder.  Two engines on one data folder therefore
// never delete or overwrite each other's files (#6034 wiped every .conv file of a shared folder at start), and a crashed
// engine's files go at the next start.  The files themselves are session files (conversation_file.hpp: a temporary name,
// then a rename; bounded reads), so an entry exists only once it is complete.
//
// Budget: min(32 GiB, 25% of the free space), never below the free-space floor; a write reserves its estimate first and
// the reservation counts until the write ends (temporary files included).
//
// A layer split's conversation is one file per stage (`conv-N.session`, then `conv-N.s1.session` ...).
//
// Kept (--conversation-spill-persist on): the folder is `<root>/kept`, the same at every start and held by its lock (one engine at a
// time), and it is not removed at the end.  Beside each entry an index file (`conv-N.idx`) names the files' identity (the session
// files' model and settings fingerprints) and the tokens and pictures it matches on, so the next start takes the entries in without
// reading their files; entries of another identity, damaged indexes and files no index names are removed at that start.

#include "strata/core/conversation_cache.hpp"
#include "strata/core/conversation_file.hpp"

#include <algorithm>
#include <condition_variable>
#include <cstdint>
#include <cstring>
#include <deque>
#include <fstream>
#include <iterator>
#include <functional>
#include <mutex>
#include <optional>
#include <thread>
#include <utility>
#include <cstdio>
#include <filesystem>
#include <random>
#include <string>
#include <system_error>
#include <vector>

#if defined(_WIN32)
#ifndef NOMINMAX
#define NOMINMAX
#endif
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <windows.h>
#include <process.h>
#else
#include <fcntl.h>
#include <sys/file.h>
#include <unistd.h>
#if defined(__linux__)
#include <sys/resource.h>
#include <sys/syscall.h>
#endif
#endif

namespace strata::core {

/// The SSD tier's size: min(32 GiB, 25% of `free_bytes`), and never more than what keeps `floor_bytes` free.
inline uint64_t spill_default_budget(uint64_t free_bytes, uint64_t floor_bytes) {
    if (free_bytes <= floor_bytes) return 0;
    uint64_t b = free_bytes / 4;
    if (b > (uint64_t(32) << 30)) b = uint64_t(32) << 30;
    if (b > free_bytes - floor_bytes) b = free_bytes - floor_bytes;
    return b;
}

/// What the tier holds and has promised: `used` (written entries) + `reserved` (writes in flight) never pass `cap`.
struct SpillBudget {
    uint64_t cap = 0, used = 0, reserved = 0;
    bool reserve(uint64_t n) {
        if (n > cap || used + reserved > cap - n) return false;
        reserved += n;
        return true;
    }
    /// A write ended: its reservation goes; `written` bytes stay (0 when it failed).  A file larger than its reservation is
    /// the caller's to remove when used would pass the cap (fits_after).
    void finish(uint64_t reserved_n, uint64_t written) {
        reserved -= reserved_n < reserved ? reserved_n : reserved;
        used += written;
    }
    bool over() const { return used + reserved > cap; }
    void drop(uint64_t n) { used -= n < used ? n : used; }
};

namespace spill_detail {
inline bool is_namespace_name(const std::string& n) {
    const size_t dash = n.find('-');
    if (dash == std::string::npos || dash == 0 || n.size() != dash + 17) return false;
    for (size_t i = 0; i < dash; ++i) if (n[i] < '0' || n[i] > '9') return false;
    for (size_t i = dash + 1; i < n.size(); ++i)
        if (!((n[i] >= '0' && n[i] <= '9') || (n[i] >= 'a' && n[i] <= 'f'))) return false;
    return true;
}
#if defined(_WIN32)
using LockHandle = HANDLE;
inline LockHandle no_lock() { return INVALID_HANDLE_VALUE; }
inline LockHandle try_lock(const std::filesystem::path& p) {
    return CreateFileW(p.wstring().c_str(), GENERIC_READ | GENERIC_WRITE, 0, nullptr, OPEN_ALWAYS, FILE_ATTRIBUTE_NORMAL, nullptr);
}
inline void unlock(LockHandle h) { if (h != INVALID_HANDLE_VALUE) CloseHandle(h); }
inline unsigned long own_pid() { return (unsigned long) _getpid(); }
#else
using LockHandle = int;
inline LockHandle no_lock() { return -1; }
inline LockHandle try_lock(const std::filesystem::path& p) {
    const int fd = ::open(p.c_str(), O_RDWR | O_CREAT | O_CLOEXEC | O_NOFOLLOW, 0600);
    if (fd < 0) return -1;
    if (::flock(fd, LOCK_EX | LOCK_NB) != 0) { ::close(fd); return -1; }
    return fd;
}
inline void unlock(LockHandle fd) { if (fd >= 0) ::close(fd); }
inline unsigned long own_pid() { return (unsigned long) ::getpid(); }
#endif
/// The calling thread gives way (the SSD tier's background writer, P2 step 2c): on Linux nice +10 and the lowest best-effort I/O
/// level (not the idle class, which a steady stream of expert reads could starve - and a starved writer would hold the serve loop
/// at its next push); on Windows a below-normal thread priority; elsewhere nothing.  Best effort: a refusal changes nothing.
inline void lower_thread_priority() {
#if defined(_WIN32)
    (void) SetThreadPriority(GetCurrentThread(), THREAD_PRIORITY_BELOW_NORMAL);
#elif defined(__linux__)
    const long tid = ::syscall(SYS_gettid);
    (void) ::setpriority(PRIO_PROCESS, (id_t) tid, 10);
    constexpr int kWhoProcess = 1, kClassBestEffort = 2, kLowest = 7, kClassShift = 13;
    (void) ::syscall(SYS_ioprio_set, kWhoProcess, (int) tid, (kClassBestEffort << kClassShift) | kLowest);
#endif
}
}  // namespace spill_detail

namespace spill_detail {
// A kept entry's index file: native-endian fields (the session files refuse a big-endian build too), counts before arrays.
constexpr uint64_t kIndexMagic = 0x3258444943525453ull;   // "STRCIDX2"

struct IndexWriter {
    std::string b;
    template <class T> void put(T v) { b.append(reinterpret_cast<const char*>(&v), sizeof v); }
    void key(const ConversationCheckpoint& c) {
        put<uint32_t>((uint32_t) c.ids.size());
        b.append(reinterpret_cast<const char*>(c.ids.data()), c.ids.size() * sizeof(int32_t));
        put<uint32_t>((uint32_t) c.imgs.size());
        for (const auto& i : c.imgs) { put<int64_t>(i.start); put<uint64_t>(i.hash); }
    }
};

struct IndexReader {
    const std::string& b;
    size_t at = 0;
    bool ok = true;
    size_t left() const { return b.size() - at; }
    template <class T> T get() {
        T v{};
        if (left() < sizeof v) { ok = false; return v; }
        std::memcpy(&v, b.data() + at, sizeof v);
        at += sizeof v;
        return v;
    }
    bool key(ConversationCheckpoint& c) {
        const uint32_t n = get<uint32_t>();
        if (!ok || left() / sizeof(int32_t) < n) return ok = false;
        c.ids.resize(n);
        std::memcpy(c.ids.data(), b.data() + at, n * sizeof(int32_t));
        at += n * sizeof(int32_t);
        const uint32_t m = get<uint32_t>();
        if (!ok || left() / 16 < m) return ok = false;
        for (uint32_t i = 0; i < m; ++i) {
            const int64_t start = get<int64_t>();
            c.imgs.push_back({start, get<uint64_t>()});
        }
        return ok;
    }
};
}  // namespace spill_detail

/// Whether a conversation is worth writing to the SSD tier: reading it back (and writing it first) must cost less than reading its
/// prompt again.  A parked image has a fixed-size running state (about 236 MB on Qwen3.8-Flash-Next) whatever its length, so a short
/// conversation is cheaper to re-read than to round-trip through the disk, and a long one is not (l620, RTX PRO 5000: a 51-token
/// conversation re-read in about 220 ms against 120 ms written + 220 ms read back; an 8K one 1.6 s against about 0.5 s).  The speeds
/// are this engine's own, measured as it runs (moving averages; defaults until the first measurement: 2 tokens per ms - a mid-range
/// card's prompt speed - and 1 GB/s each way).  The prompt speed is learned only from passes of at least kMinTokens fresh tokens, so
/// a request's fixed costs do not make re-reading look slow.
struct SpillCost {
    static constexpr int64_t kMinTokens = 512;
    static constexpr double kAlpha = 0.3, kMargin = 1.25;
    double reread_tokens_per_ms = 2.0;
    double write_bytes_per_ms = 1.0e6, read_bytes_per_ms = 1.0e6;
    static void blend(double& avg, double sample) {
        if (sample > 0.0 && sample < 1e300) avg = avg * (1.0 - kAlpha) + sample * kAlpha;
    }
    void saw_prompt(int64_t tokens, double ms) { if (tokens >= kMinTokens && ms > 0.0) blend(reread_tokens_per_ms, (double) tokens / ms); }
    void saw_write(uint64_t bytes, double ms) { if (bytes > 0 && ms > 0.0) blend(write_bytes_per_ms, (double) bytes / ms); }
    void saw_read(uint64_t bytes, double ms) { if (bytes > 0 && ms > 0.0) blend(read_bytes_per_ms, (double) bytes / ms); }
    double reread_ms(int64_t tokens) const { return tokens > 0 ? (double) tokens / reread_tokens_per_ms : 0.0; }
    double round_trip_ms(uint64_t bytes) const { return (double) bytes / write_bytes_per_ms + (double) bytes / read_bytes_per_ms; }
    bool worth(int64_t tokens, uint64_t bytes) const { return reread_ms(tokens) > kMargin * round_trip_ms(bytes); }
};

/// This process's spill folder.  open() makes `<root>`, cleans the dead folders and takes its own; the destructor gives the
/// lock back and removes its own folder (an orderly end; after a crash the next start's clean_dead does it).
class SpillNamespace {
public:
    SpillNamespace() = default;
    SpillNamespace(const SpillNamespace&) = delete;
    SpillNamespace& operator=(const SpillNamespace&) = delete;
    ~SpillNamespace() { close(); }

    /// `kept`: the folder is `<root>/kept` (it may hold an earlier run's entries) and close() leaves it on the disk
    bool open(const std::filesystem::path& root, std::string& error, size_t* cleaned = nullptr, bool kept = false) {
        close();
        std::error_code ec;
        std::filesystem::create_directories(root, ec);
        if (ec || !std::filesystem::is_directory(root, ec) || std::filesystem::is_symlink(root, ec)) {
            error = "spill folder " + root.string() + " cannot be used" + (ec ? ": " + ec.message() : std::string());
            return false;
        }
        const size_t n = clean_dead(root);
        if (cleaned) *cleaned = n;
        std::random_device rd;
        char name[64];
        std::snprintf(name, sizeof name, "%lu-%08x%08x", spill_detail::own_pid(), (unsigned) rd(), (unsigned) rd());
        const std::filesystem::path dir = kept ? root / "kept" : root / name;
        const bool made = std::filesystem::create_directory(dir, ec);
        if (ec || (!made && !kept) || std::filesystem::is_symlink(dir, ec)) {
            error = "spill folder " + dir.string() + " cannot be made" + (ec ? ": " + ec.message() : std::string());
            return false;
        }
        lock_ = spill_detail::try_lock(dir / ".lock");
        if (lock_ == spill_detail::no_lock()) {
            if (!kept) std::filesystem::remove_all(dir, ec);
            error = "spill folder " + dir.string() + ": its lock cannot be taken" + (kept ? " (another engine keeps it)" : "");
            return false;
        }
        root_ = root;
        dir_ = dir;
        kept_ = kept;
        return true;
    }
    void close() {
        if (dir_.empty()) return;
        spill_detail::unlock(lock_);
        lock_ = spill_detail::no_lock();
        std::error_code ec;
        if (!kept_) std::filesystem::remove_all(dir_, ec);
        dir_.clear();
    }
    bool is_open() const { return !dir_.empty(); }
    bool kept() const { return kept_; }
    const std::filesystem::path& dir() const { return dir_; }
    /// the file of parked conversation `serial`, stage `part` (a session file; written to a temporary name first by session_file_write)
    std::filesystem::path file_for(uint64_t serial, size_t part = 0) const {
        return dir_ / ("conv-" + std::to_string(serial) + (part ? ".s" + std::to_string(part) : std::string()) + ".session");
    }
    /// a kept entry's index file
    std::filesystem::path index_for(uint64_t serial) const { return dir_ / ("conv-" + std::to_string(serial) + ".idx"); }

    /// Removes the folders under `root` that are named like a namespace, are real folders (not links) and whose lock can be
    /// taken - no process holds them.  Returns how many went.  Everything else under `root` is left alone.
    static size_t clean_dead(const std::filesystem::path& root) {
        size_t n = 0;
        std::error_code ec;
        for (std::filesystem::directory_iterator it(root, ec), end; !ec && it != end; it.increment(ec)) {
            const std::filesystem::path p = it->path();
            std::error_code e2;
            if (it->is_symlink(e2) || !it->is_directory(e2) || !spill_detail::is_namespace_name(p.filename().string())) continue;
            const spill_detail::LockHandle h = spill_detail::try_lock(p / ".lock");
            if (h == spill_detail::no_lock()) continue;   // a live engine holds it (or it cannot be checked): kept
            spill_detail::unlock(h);
            std::filesystem::remove_all(p, e2);
            if (!e2) ++n;
        }
        return n;
    }

private:
    std::filesystem::path root_, dir_;
    spill_detail::LockHandle lock_ = spill_detail::no_lock();
    bool kept_ = false;
};

/// The SSD tier's entries: parked conversations written as session files into this process's namespace, oldest first.
/// What stays in RAM per entry is only what matching needs: the token ids and picture keys of its live state and of each
/// checkpoint (no running state, no K/V).  put() deletes the oldest entries when the budget needs the room, and never
/// leaves a partial file behind; take() reads an entry back under the caller's read limits and removes it either way (a
/// file that does not read back is not tried again).
/// Thread-safe (P2 step 2c: a background writer puts while the serve loop matches and takes): the entry list and the budget are
/// under one lock, a file is written and read outside it, and an entry is reached by its serial (an index can shift when the
/// writer drops the oldest).  An entry is listed only once its file is complete.
class ConversationSpill {
public:
    struct Entry {
        uint64_t serial = 0;
        std::filesystem::path file;                      // the first stage's
        size_t parts = 1;                                // files: one per layer-split stage
        uint64_t bytes = 0;
        bool cvec = false;
        ConversationCheckpoint live;                     // ids and images only
        std::vector<ConversationCheckpoint> checkpoints;   // ids and images only
    };

    /// `keep`: the kept folder (--conversation-spill-persist), with the entries an earlier run left of this identity
    bool open(const std::filesystem::path& root, uint64_t cap, std::string& error, size_t* cleaned = nullptr,
              const SessionFileIdentity* keep = nullptr) {
        std::lock_guard<std::mutex> lk(mu_);
        entries_.clear();
        budget_ = SpillBudget{cap, 0, 0};
        if (!ns_.open(root, error, cleaned, keep != nullptr)) return false;
        if (keep) take_in_locked(*keep);
        return true;
    }
    bool kept() const { std::lock_guard<std::mutex> lk(mu_); return ns_.kept(); }
    bool enabled() const { std::lock_guard<std::mutex> lk(mu_); return enabled_locked(); }
    size_t size() const { std::lock_guard<std::mutex> lk(mu_); return entries_.size(); }
    uint64_t bytes() const { std::lock_guard<std::mutex> lk(mu_); return budget_.used; }
    SpillBudget budget() const { std::lock_guard<std::mutex> lk(mu_); return budget_; }
    Entry entry(size_t i) const { std::lock_guard<std::mutex> lk(mu_); return entries_.at(i); }
    /// the entry `serial` (a copy), if it is still listed
    std::optional<Entry> find(uint64_t serial) const {
        std::lock_guard<std::mutex> lk(mu_);
        const size_t i = index_locked(serial);
        return i == SIZE_MAX ? std::nullopt : std::optional<Entry>(entries_[i]);
    }
    std::filesystem::path dir() const { std::lock_guard<std::mutex> lk(mu_); return ns_.dir(); }

    /// Writes `image` as an entry (its serial in `*serial_out`).  False (with `error`) when the tier is off, the image is larger
    /// than the whole tier, or the write failed; nothing of this call is then left on the disk.  The file is written outside the
    /// lock, under a budget reserved for it.
    bool put(const SavedConversation& image, const SessionFileIdentity& id, std::string& error, uint64_t* serial_out = nullptr) {
        const size_t parts = 1 + image.stage_images.size();
        const uint64_t need = (uint64_t) image.bytes() + (uint64_t(parts) << 20);   // the payload and the files' own tables
        uint64_t serial = 0;
        bool kept = false;
        {
            std::lock_guard<std::mutex> lk(mu_);
            if (!enabled_locked()) { error = "the SSD tier is off"; return false; }
            if (need > budget_.cap) { error = "larger than the whole SSD tier"; return false; }
            while (!budget_.reserve(need)) {
                if (entries_.empty()) { error = "no room in the SSD tier"; return false; }
                drop_locked(0);
            }
            serial = next_serial_++;
            kept = ns_.kept();
        }
        Entry e;
        e.serial = serial; e.file = ns_.file_for(serial); e.parts = parts; e.cvec = image.cvec;
        e.live = stub(image.live);
        for (const auto& c : image.checkpoints) e.checkpoints.push_back(stub(c));
        bool ok = true;
        uint64_t written = 0;
        for (size_t k = 0; ok && k < parts; ++k) {   // (session_file_write writes the image it is given, not its stage images)
            size_t n = 0;
            ok = session_file_write(ns_.file_for(serial, k).string(), k ? image.stage_images[k - 1] : image, id, n, error);
            written += n;
        }
        e.bytes = written;
        if (ok && kept) ok = write_index(ns_.index_for(serial), e, id, error);   // after the files: it names complete ones
        if (!ok) remove_files(e);
        std::lock_guard<std::mutex> lk(mu_);
        budget_.finish(need, ok ? written : 0);
        if (!ok) return false;
        entries_.push_back(std::move(e));
        while (budget_.over() && entries_.size() > 1) drop_locked(0);   // (a file larger than its estimate)
        if (serial_out) *serial_out = serial;
        return true;
    }

    /// The entry and the prefix it gives `prompt`, by the RAM cache's rule (ConversationCache::best): its live state or a
    /// checkpoint, same pictures, same steering; ties prefer the newest.  Reach it by `serial` (find / take_serial).
    template <class Token>
    ConversationCache::Match best(const std::vector<Token>& prompt, const std::vector<ConversationImageKey>& images, bool cvec) const {
        std::lock_guard<std::mutex> lk(mu_);
        ConversationCache::Match m;
        for (size_t i = entries_.size(); i-- > 0;) {
            const Entry& e = entries_[i];
            if (e.cvec != cvec) continue;
            auto consider = [&](const ConversationCheckpoint& c, bool live) {
                const int64_t n = conversation_prefix(c, prompt, images);
                if (n > m.tokens) m = {i, n, live, e.serial};
            };
            consider(e.live, true);
            for (const auto& c : e.checkpoints) consider(c, false);
        }
        return m;
    }

    /// Reads entry `serial` back into `out` (the identity checked; the first stage's file bounded by `limits`, a later stage's by
    /// its RAM preflight only - the caller validates every stage image against its runtime) and removes the entry.  False: `out` is
    /// unchanged and the entry is gone (or was not there).  The files are read outside the lock: the entry leaves the list first.
    bool take_serial(uint64_t serial, const SessionFileIdentity& id, const SessionReadLimits& limits, SavedConversation& out,
                     std::string& error) {
        Entry e;
        {
            std::lock_guard<std::mutex> lk(mu_);
            const size_t i = index_locked(serial);
            if (i == SIZE_MAX) { error = "no such SSD entry"; return false; }
            e = entries_[i];
            budget_.drop(e.bytes);
            entries_.erase(entries_.begin() + (std::ptrdiff_t) i);
            if (ns_.kept()) { std::error_code ec; std::filesystem::remove(ns_.index_for(serial), ec); }   // (first: not taken in again)
        }
        SessionReadLimits stage_limits;
        stage_limits.admit = limits.admit;
        SavedConversation image;
        bool ok = true;
        for (size_t k = 0; ok && k < e.parts; ++k) {
            SavedConversation part;
            size_t read = 0;
            ok = session_file_read(ns_.file_for(serial, k).string(), id, part, read, error, k ? stage_limits : limits);
            if (!ok) break;
            if (k == 0) image = std::move(part);
            else image.stage_images.push_back(std::move(part));
        }
        remove_files(e);
        if (ok) out = std::move(image);
        return ok;
    }
    bool take(size_t i, const SessionFileIdentity& id, const SessionReadLimits& limits, SavedConversation& out, std::string& error) {
        uint64_t serial = 0;
        {
            std::lock_guard<std::mutex> lk(mu_);
            if (i >= entries_.size()) { error = "no such SSD entry"; return false; }
            serial = entries_[i].serial;
        }
        return take_serial(serial, id, limits, out, error);
    }

    void drop(size_t i) { std::lock_guard<std::mutex> lk(mu_); drop_locked(i); }
    /// Drops entry `serial` and its file; false when it was not listed.
    bool drop_serial(uint64_t serial) {
        std::lock_guard<std::mutex> lk(mu_);
        const size_t i = index_locked(serial);
        if (i == SIZE_MAX) return false;
        drop_locked(i);
        return true;
    }
    /// Drops every entry and keeps the tier open (#879: after non-finite logits no parked conversation is kept).  Returns how many.
    size_t clear() {
        std::lock_guard<std::mutex> lk(mu_);
        const size_t n = entries_.size();
        while (!entries_.empty()) drop_locked(entries_.size() - 1);
        return n;
    }
    /// the end: the entries go with their files, except in the kept folder, where they stay for the next start
    void close() {
        std::lock_guard<std::mutex> lk(mu_);
        if (!ns_.kept())
            while (!entries_.empty()) drop_locked(entries_.size() - 1);
        entries_.clear();
        ns_.close();
    }

private:
    bool enabled_locked() const { return ns_.is_open() && budget_.cap > 0; }
    size_t index_locked(uint64_t serial) const {
        for (size_t i = 0; i < entries_.size(); ++i)
            if (entries_[i].serial == serial) return i;
        return SIZE_MAX;
    }
    void drop_locked(size_t i) {
        remove_files(entries_.at(i));
        budget_.drop(entries_[i].bytes);
        entries_.erase(entries_.begin() + (std::ptrdiff_t) i);
    }
    /// an entry's index (first: a removed entry is never taken in again) and files
    void remove_files(const Entry& e) const {
        std::error_code ec;
        std::filesystem::remove(ns_.index_for(e.serial), ec);
        for (size_t k = 0; k < e.parts; ++k) std::filesystem::remove(ns_.file_for(e.serial, k), ec);
    }
    /// open() of the kept folder: the entries an earlier run left with this identity, oldest first, within the budget; every
    /// other file of the folder goes (another identity, a damaged index, files no index names, interrupted writes).  An index
    /// is written after its entry's files and removed before them, so an index names complete files
    void take_in_locked(const SessionFileIdentity& id) {
        std::vector<Entry> found;
        std::error_code ec;
        for (std::filesystem::directory_iterator it(ns_.dir(), ec), end; !ec && it != end; it.increment(ec)) {
            if (it->path().extension() != ".idx") continue;
            Entry e;
            if (read_index(it->path(), id, e)) {
                e.file = ns_.file_for(e.serial);
                found.push_back(std::move(e));
            }
        }
        std::sort(found.begin(), found.end(), [](const Entry& a, const Entry& b) { return a.serial < b.serial; });
        std::vector<std::filesystem::path> keep{ns_.dir() / ".lock"};
        for (auto& e : found) {
            for (size_t k = 0; k < e.parts; ++k) keep.push_back(ns_.file_for(e.serial, k));
            keep.push_back(ns_.index_for(e.serial));
            next_serial_ = std::max(next_serial_, e.serial + 1);
            budget_.used += e.bytes;
            entries_.push_back(std::move(e));
        }
        for (std::filesystem::directory_iterator it(ns_.dir(), ec), end; !ec && it != end; it.increment(ec))
            if (std::find(keep.begin(), keep.end(), it->path()) == keep.end()) { std::error_code e2; std::filesystem::remove_all(it->path(), e2); }
        while (budget_.over() && !entries_.empty()) drop_locked(0);
    }
    static bool write_index(const std::filesystem::path& path, const Entry& e, const SessionFileIdentity& id, std::string& error) {
        spill_detail::IndexWriter out;
        out.put(spill_detail::kIndexMagic);
        out.put(id.model);
        out.put(id.config);
        out.put<uint64_t>(e.serial);
        out.put<uint32_t>((uint32_t) e.parts);
        out.put<uint64_t>(e.bytes);
        out.put<uint8_t>(e.cvec ? 1 : 0);
        out.key(e.live);
        out.put<uint32_t>((uint32_t) e.checkpoints.size());
        for (const auto& c : e.checkpoints) out.key(c);
        std::filesystem::path temp = path;
        temp += ".tmp";
        bool written = false;
        {
            std::ofstream f(temp, std::ios::binary | std::ios::trunc);
            f.write(out.b.data(), (std::streamsize) out.b.size());
            written = bool(f);
        }
        std::error_code ec;
        if (written) std::filesystem::rename(temp, path, ec);   // a reader sees the whole index or none
        if (written && !ec) return true;
        error = written ? "cannot publish the index: " + ec.message() : "cannot write the index " + temp.string();
        std::filesystem::remove(temp, ec);
        return false;
    }
    static bool read_index(const std::filesystem::path& path, const SessionFileIdentity& id, Entry& e) {
        std::ifstream f(path, std::ios::binary);
        const std::string b((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
        spill_detail::IndexReader in{b};
        if (in.get<uint64_t>() != spill_detail::kIndexMagic || in.get<uint64_t>() != id.model || in.get<uint64_t>() != id.config)
            return false;
        e.serial = in.get<uint64_t>();
        e.parts = in.get<uint32_t>();
        e.bytes = in.get<uint64_t>();
        e.cvec = in.get<uint8_t>() != 0;
        if (!in.key(e.live)) return false;
        const uint32_t n = in.get<uint32_t>();
        for (uint32_t i = 0; in.ok && i < n; ++i) {
            ConversationCheckpoint c;
            if (in.key(c)) e.checkpoints.push_back(std::move(c));
        }
        return in.ok && in.left() == 0 && e.parts >= 1;
    }
    static ConversationCheckpoint stub(const ConversationCheckpoint& c) {
        ConversationCheckpoint s;
        s.ids = c.ids;
        s.imgs = c.imgs;
        return s;
    }
    mutable std::mutex mu_;
    SpillNamespace ns_;
    SpillBudget budget_;
    std::vector<Entry> entries_;
    uint64_t next_serial_ = 1;
};

/// The best prefix a parked image gives `prompt`, by the RAM cache's rule (its live state or a checkpoint, same pictures, same
/// steering): for the images the background writer still holds in RAM.  {tokens, live}; tokens 0 = none.
template <class Token>
std::pair<int64_t, bool> conversation_image_prefix(const SavedConversation& s, const std::vector<Token>& prompt,
                                                   const std::vector<ConversationImageKey>& images, bool cvec) {
    std::pair<int64_t, bool> m{0, false};
    if (s.cvec != cvec) return m;
    const int64_t l = conversation_prefix(s.live, prompt, images);
    if (l > m.first) m = {l, true};
    for (const auto& c : s.checkpoints) {
        const int64_t n = conversation_prefix(c, prompt, images);
        if (n > m.first) m = {n, false};
    }
    return m;
}

/// P2 step 2c: writes parked conversations to the SSD tier on its own thread, so a request's park does not wait for the disk.
/// At most one image is written at a time and one waits behind it (push() blocks past that): what the writer holds in RAM is at
/// most two images over the RAM tier's budget, and the physical-RAM gate sees it (it is still allocated).  A written image is not
/// freed on the writer thread: it is handed back (take_done) to be freed - or reused, it is still in RAM - on the caller's
/// thread, the one that owns the cache's allocators.  `write` runs on the writer thread and returns the SSD serial (0: failed).
template <class Item>
class BackgroundWriter {
public:
    struct Done {
        Item item;
        uint64_t serial = 0;   // its SSD entry; 0: the write failed (the image is all there is)
    };
    /// `on_start`: runs first on the writer thread (the serve loop lowers its priority there: spill_detail::lower_thread_priority)
    explicit BackgroundWriter(std::function<uint64_t(const Item&)> write, std::function<void()> on_start = {})
        : write_(std::move(write)), on_start_(std::move(on_start)), th_([this] { if (on_start_) on_start_(); run(); }) {}
    BackgroundWriter(const BackgroundWriter&) = delete;
    BackgroundWriter& operator=(const BackgroundWriter&) = delete;
    /// what is queued is not written (the namespace goes with the process); a write in flight finishes first
    ~BackgroundWriter() {
        {
            std::lock_guard<std::mutex> lk(mu_);
            stop_ = true;
            queue_.clear();
        }
        cv_.notify_all();
        th_.join();
    }
    void push(Item&& it) {
        std::unique_lock<std::mutex> lk(mu_);
        cv_.wait(lk, [&] { return queue_.empty() || stop_; });
        if (stop_) return;
        queue_.push_back(std::move(it));
        cv_.notify_all();
    }
    void wait_idle() {
        std::unique_lock<std::mutex> lk(mu_);
        cv_.wait(lk, [&] { return (queue_.empty() && !busy_) || stop_; });
    }
    bool idle() const { std::lock_guard<std::mutex> lk(mu_); return queue_.empty() && !busy_; }
    /// the written ones, to free or reuse on this thread
    std::vector<Done> take_done() {
        std::lock_guard<std::mutex> lk(mu_);
        std::vector<Done> out;
        out.swap(done_);
        return out;
    }
    /// the queued ones (not the one in flight) back, unwritten - #879 clears them with the cache
    std::vector<Item> take_queued() {
        std::lock_guard<std::mutex> lk(mu_);
        std::vector<Item> out;
        for (auto& q : queue_) out.push_back(std::move(q));
        queue_.clear();
        cv_.notify_all();
        return out;
    }
    /// f(const Item&) over every image not yet handed back - queued, in flight, written - under the lock (the writer only
    /// reads an image in flight, so reading it here as well is safe)
    template <class F>
    void for_each_held(F f) const {
        std::lock_guard<std::mutex> lk(mu_);
        for (const auto& q : queue_) f(q);
        if (inflight_) f(*inflight_);
        for (const auto& d : done_) f(d.item);
    }

private:
    void run() {
        std::unique_lock<std::mutex> lk(mu_);
        for (;;) {
            cv_.wait(lk, [&] { return stop_ || !queue_.empty(); });
            if (stop_) return;
            inflight_.emplace(std::move(queue_.front()));
            queue_.pop_front();
            busy_ = true;
            cv_.notify_all();   // (a push waiting for room)
            lk.unlock();
            uint64_t serial = 0;
            try { serial = write_(*inflight_); } catch (...) { serial = 0; }
            lk.lock();
            done_.push_back(Done{std::move(*inflight_), serial});
            inflight_.reset();
            busy_ = false;
            cv_.notify_all();
        }
    }
    std::function<uint64_t(const Item&)> write_;
    std::function<void()> on_start_;
    mutable std::mutex mu_;
    std::condition_variable cv_;
    std::deque<Item> queue_;
    std::optional<Item> inflight_;
    std::vector<Done> done_;
    bool busy_ = false, stop_ = false;
    std::thread th_;   // (last: it starts once the rest exists)
};

}  // namespace strata::core
