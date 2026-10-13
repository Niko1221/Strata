// src/core/conversation_disk.cpp - the disk tier of the parked conversations.  See the header.
#include "strata/core/conversation_disk.hpp"

#include <algorithm>
#include <chrono>
#include <cstdarg>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <random>
#include <set>
#include <system_error>
#include <utility>
#if defined(__linux__)
#include <sys/resource.h>
#include <sys/syscall.h>
#include <unistd.h>
#endif

namespace strata::core {
namespace {

namespace fs = std::filesystem;

// The second half of every file's identity: with this run's random number as the first, a file another run (or
// another tier) wrote is refused on read.
constexpr uint64_t kTierConfig = 0x5354524154414449ull;   // "STRATADI"

// ids and images only: the part of a checkpoint best() matches against
ConversationCheckpoint key_of(const ConversationCheckpoint& c) {
    ConversationCheckpoint k;
    k.ids = c.ids;
    k.imgs = c.imgs;
    return k;
}

// The payload is the conversation its index names: the same keys and control-vector state, in every stage.
bool holds(const SavedConversation& s, const ConversationCheckpoint& live,
           const std::vector<ConversationCheckpoint>& checkpoints, bool cvec) {
    auto same = [](const ConversationCheckpoint& x, const ConversationCheckpoint& y) {
        return x.ids == y.ids && x.imgs == y.imgs;
    };
    if (s.cvec != cvec || !same(s.live, live) || s.checkpoints.size() != checkpoints.size()) return false;
    for (size_t i = 0; i < checkpoints.size(); ++i)
        if (!same(s.checkpoints[i], checkpoints[i])) return false;
    for (const auto& stage : s.stage_images)
        if (!holds(stage, live, checkpoints, cvec)) return false;
    return true;
}

// Only the live state and the deepest checkpoint go to the disk (the session files' rule).  With a layer split each
// later stage keeps the same checkpoint (the split keeps the order); a stage whose list does not line up drops them all.
void keep_deepest_checkpoint(SavedConversation& img) {
    if (img.checkpoints.empty()) return;
    size_t d = 0;
    for (size_t i = 1; i < img.checkpoints.size(); ++i)
        if (img.checkpoints[i].ids.size() > img.checkpoints[d].ids.size()) d = i;
    bool aligned = true;
    for (const auto& s : img.stage_images)
        aligned = aligned && s.checkpoints.size() == img.checkpoints.size() && s.checkpoints[d].ids == img.checkpoints[d].ids;
    auto keep = [&](std::vector<ConversationCheckpoint>& v) {
        if (!aligned) { v.clear(); return; }
        ConversationCheckpoint c = std::move(v[d]);
        v.clear();
        v.push_back(std::move(c));
    };
    keep(img.checkpoints);
    for (auto& s : img.stage_images) keep(s.checkpoints);
}

// a file of this tier (also a temporary one session_file_write left behind: ".<name>.<hex>.tmp")
bool tier_file(const std::string& name) {
    const std::string prefix = ConversationDiskTier::file_prefix();
    return name.rfind(prefix, 0) == 0 || name.rfind("." + prefix, 0) == 0;
}

void remove_file(const std::string& path) {
    std::error_code ec;
    fs::remove(fs::path(path), ec);
}

// The index file of a persistent entry: native-endian fields (the session files refuse a big-endian build too),
// counts before arrays.
constexpr uint64_t kIndexMagic = 0x3158444943525453ull;   // "STRCIDX1"

struct IndexWriter {
    std::string b;
    template<class T> void put(T v) { b.append(reinterpret_cast<const char*>(&v), sizeof v); }
    void str(const std::string& s) { put<uint32_t>((uint32_t) s.size()); b += s; }
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
    template<class T> T get() {
        T v{};
        if (left() < sizeof v) { ok = false; return v; }
        std::memcpy(&v, b.data() + at, sizeof v);
        at += sizeof v;
        return v;
    }
    std::string str() {
        const uint32_t n = get<uint32_t>();
        if (!ok || left() < n) { ok = false; return {}; }
        at += n;
        return b.substr(at - n, n);
    }
    bool key(ConversationCheckpoint& c) {
        const uint32_t n = get<uint32_t>();
        if (!ok || left() / sizeof(int32_t) < n) return ok = false;
        c.ids.resize(n);
        std::memcpy(c.ids.data(), b.data() + at, n * sizeof(int32_t));
        at += n * sizeof(int32_t);
        const uint32_t m = get<uint32_t>();
        if (!ok || left() / 16 < m) return ok = false;
        for (uint32_t i = 0; i < m; ++i) c.imgs.push_back({get<int64_t>(), get<uint64_t>()});
        return ok;
    }
};

double mib(uint64_t bytes) { return double(bytes) / (1024.0 * 1024.0); }
double mbps(uint64_t bytes, double ms) { return ms > 0 ? double(bytes) / 1e6 / (ms / 1000.0) : 0.0; }
double ms_since(std::chrono::steady_clock::time_point t0) {
    return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
}

#if defined(__GNUC__)
std::string fmt(const char* f, ...) __attribute__((format(printf, 1, 2)));
#endif
std::string fmt(const char* f, ...) {
    char buf[512];
    va_list ap;
    va_start(ap, f);
    std::vsnprintf(buf, sizeof buf, f, ap);
    va_end(ap);
    return buf;
}

}  // namespace

ConversationDiskTier::ConversationDiskTier(ConversationDiskOptions options, std::function<Clock::time_point()> now)
    : options_(std::move(options)), now_(std::move(now)) {
    std::random_device rd;
    nonce_ = ((uint64_t) rd() << 32) ^ rd() ^ (uint64_t) std::chrono::system_clock::now().time_since_epoch().count();
}

ConversationDiskTier::~ConversationDiskTier() {
    if (open_ && options_.persist) wait_idle();   // what is queued belongs to the next start
    size_t dropped = 0;
    {
        std::lock_guard<std::mutex> lock(m_);
        stop_ = true;
        dropped = queue_.size();
        queue_.clear();   // not written: dropped, as the RAM cache would have
    }
    cv_.notify_all();
    if (thread_.joinable()) thread_.join();
    std::lock_guard<std::mutex> lock(m_);
    if (!open_) return;
    if (options_.persist) {
        log(fmt("end: %zu conversations kept on the disk for the next start (%.0f MiB)", entries_.size(), mib(bytes_)));
        return;
    }
    if (dropped || !entries_.empty())
        log(fmt("end: %zu queued dropped, %zu conversations removed from the disk", dropped, entries_.size()));
    while (!entries_.empty()) remove_locked(0, true);
}

SessionFileIdentity ConversationDiskTier::file_identity() const {
    return options_.persist ? options_.identity : SessionFileIdentity{nonce_, kTierConfig};
}

bool ConversationDiskTier::write_index(Entry& e, std::string& error) const {
    char name[96];
    std::snprintf(name, sizeof name, "%s%016llx-%llu.idx", file_prefix(), (unsigned long long) nonce_,
                  (unsigned long long) e.id);
    const fs::path path = fs::path(options_.dir) / name;
    const fs::path temp = fs::path(options_.dir) / ("." + std::string(name) + ".tmp");
    e.written_s = std::chrono::duration_cast<std::chrono::seconds>(
        std::chrono::system_clock::now().time_since_epoch()).count();
    const SessionFileIdentity id = file_identity();
    IndexWriter out;
    out.put(kIndexMagic);
    out.put(id.model);
    out.put(id.config);
    out.put<int64_t>(e.written_s);
    out.put<uint8_t>(e.cvec ? 1 : 0);
    out.put<uint64_t>(e.bytes);
    out.put<uint32_t>((uint32_t) e.files.size());
    for (const auto& f : e.files) out.str(fs::path(f).filename().string());
    out.key(e.live);
    out.put<uint32_t>((uint32_t) e.checkpoints.size());
    for (const auto& c : e.checkpoints) out.key(c);
    {
        std::ofstream f(temp, std::ios::binary | std::ios::trunc);
        f.write(out.b.data(), (std::streamsize) out.b.size());
        if (!f) {
            error = "cannot write the index " + temp.string();
            remove_file(temp.string());
            return false;
        }
    }
    std::error_code ec;
    fs::rename(temp, path, ec);   // a reader sees the whole index or none
    if (ec) {
        error = "cannot publish the index: " + ec.message();
        remove_file(temp.string());
        return false;
    }
    e.index = path.string();
    return true;
}

bool ConversationDiskTier::read_index(const std::string& path, Entry& e) const {
    std::ifstream f(fs::path(path), std::ios::binary);
    const std::string b((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
    IndexReader in{b};
    const SessionFileIdentity id = file_identity();
    if (in.get<uint64_t>() != kIndexMagic || in.get<uint64_t>() != id.model || in.get<uint64_t>() != id.config)
        return false;
    e.written_s = in.get<int64_t>();
    e.cvec = in.get<uint8_t>() != 0;
    e.bytes = in.get<uint64_t>();
    const uint32_t nfiles = in.get<uint32_t>();
    for (uint32_t i = 0; in.ok && i < nfiles; ++i) e.files.push_back((fs::path(options_.dir) / in.str()).string());
    if (!in.key(e.live)) return false;
    const uint32_t nck = in.get<uint32_t>();
    for (uint32_t i = 0; in.ok && i < nck; ++i) {
        ConversationCheckpoint c;
        if (in.key(c)) e.checkpoints.push_back(std::move(c));
    }
    if (!in.ok || in.left() != 0 || e.files.empty()) return false;
    std::error_code ec;
    for (const auto& file : e.files)
        if (!fs::is_regular_file(fs::path(file), ec)) return false;
    e.index = path;
    return true;
}

std::string ConversationDiskTier::scan_folder(size_t& stale, std::error_code& ec) {
    std::vector<Entry> found;
    std::vector<std::string> others;
    size_t refused = 0;
    for (fs::directory_iterator it(fs::path(options_.dir), ec), end; !ec && it != end; it.increment(ec)) {
        if (!it->is_regular_file(ec) || !tier_file(it->path().filename().string())) continue;
        Entry e;
        const bool index = it->path().extension() == ".idx";
        const bool taken = index && read_index(it->path().string(), e);
        if (taken) found.push_back(std::move(e));
        else others.push_back(it->path().string());
        refused += index && !taken;
    }
    // the files of the entries taken in stay; the rest goes (refused indexes and their files, interrupted writes)
    std::set<std::string> kept;
    for (const auto& e : found) kept.insert(e.files.begin(), e.files.end());
    for (const auto& p : others)
        if (!kept.count(p)) { remove_file(p); ++stale; }
    std::sort(found.begin(), found.end(), [](const Entry& a, const Entry& b) { return a.written_s < b.written_s; });
    for (auto& e : found) {
        e.id = ++next_id_;
        e.at = now_();
        bytes_ += e.bytes;
        entries_.push_back(std::move(e));
    }
    size_t over = 0;
    while (!entries_.empty() && bytes_ > options_.budget_bytes) {
        remove_locked(0, true);
        ++over;
    }
    return fmt("persistent (budget only): %zu conversations from earlier runs (%.0f MiB), %zu refused (another model, "
               "engine version or settings), %zu over the budget removed", entries_.size(), mib(bytes_), refused, over);
}

void ConversationDiskTier::log(const std::string& line) const {
    if (options_.log) options_.log(line);
    else std::fprintf(stderr, "strata serve: conversation disk: %s\n", line.c_str());
}

bool ConversationDiskTier::open(std::string& error) {
    open_ = false;
    if (options_.dir.empty() || options_.budget_bytes == 0) return true;   // off: nothing to open
    std::error_code ec;
    const fs::path dir(options_.dir);
    fs::create_directories(dir, ec);
    if (ec || !fs::is_directory(dir, ec)) {
        error = "cannot create the folder " + options_.dir + (ec ? ": " + ec.message() : std::string());
        return false;
    }
#ifndef _WIN32
    fs::permissions(dir, fs::perms::owner_all, fs::perm_options::replace, ec);   // the files hold conversations
    ec.clear();
#endif
    size_t stale = 0;
    const std::string taken_in = scan_folder(stale, ec);
    if (ec) {
        error = "cannot list the folder " + options_.dir + ": " + ec.message();
        return false;
    }
    open_ = true;
    thread_ = std::thread([this] { writer(); });
    const std::string kept = options_.persist ? taken_in : fmt("%lld min per conversation", (long long) (options_.ttl_s / 60));
    log(fmt("on: %s, budget %.0f MiB, %s; %zu stale files removed", options_.dir.c_str(), mib(options_.budget_bytes),
            kept.c_str(), stale));
    return true;
}

void ConversationDiskTier::remove_locked(size_t index, bool delete_files) {
    Entry& e = entries_.at(index);
    if (!e.index.empty()) remove_file(e.index);   // first, always: an entry gone from the list is not taken in again
    if (delete_files)
        for (const auto& f : e.files) remove_file(f);
    bytes_ -= e.bytes;
    entries_.erase(entries_.begin() + (std::ptrdiff_t) index);
}

size_t ConversationDiskTier::expire_locked() {
    if (!open_ || options_.ttl_s <= 0) return 0;
    const auto now = now_();
    const auto ttl = std::chrono::seconds(options_.ttl_s);
    size_t n = 0;
    while (!entries_.empty() && now - entries_.front().at >= ttl) {   // oldest first: the rest are younger
        remove_locked(0, true);
        ++n;
    }
    stats_.expired += n;
    return n;
}

size_t ConversationDiskTier::expire() {
    size_t n = 0;
    {
        std::lock_guard<std::mutex> lock(m_);
        n = expire_locked();
    }
    if (n) log(fmt("removed %zu conversation%s older than %lld min", n, n == 1 ? "" : "s",
                   (long long) (options_.ttl_s / 60)));
    return n;
}

bool ConversationDiskTier::put(SavedConversation&& image) {
    if (!open_) return false;
    Job job;
    job.image = std::move(image);
    keep_deepest_checkpoint(job.image);
    job.entry.live = key_of(job.image.live);
    for (const auto& c : job.image.checkpoints) job.entry.checkpoints.push_back(key_of(c));
    job.entry.cvec = job.image.cvec;
    job.entry.bytes = job.image.bytes();
    const size_t tokens = job.entry.live.ids.size();
    if (job.entry.bytes > options_.budget_bytes) {
        log(fmt("dropped %zu tokens: %.0f MiB is over the disk budget", tokens, mib(job.entry.bytes)));
        std::lock_guard<std::mutex> lock(m_);
        ++stats_.write_failed;
        return false;
    }
    size_t depth = 0;
    uint64_t queued_bytes = 0;
    const uint64_t estimate = job.entry.bytes;
    {
        std::lock_guard<std::mutex> lock(m_);
        job.entry.id = ++next_id_;
        ++stats_.queued;
        queue_.push_back(std::move(job));
        depth = queue_.size();
        for (const auto& j : queue_) queued_bytes += j.entry.bytes;
    }
    cv_.notify_all();
    log(fmt("queued %zu tokens (%.0f MiB in RAM until written; queue %zu, %.0f MiB)", tokens, mib(estimate), depth,
            mib(queued_bytes)));
    return true;
}

void ConversationDiskTier::writer() {
#if defined(__linux__)
    // below the serving threads: the write is background work, decode and prompt reading come first
    setpriority(PRIO_PROCESS, (id_t) syscall(SYS_gettid), 10);
#endif
    for (;;) {
        Job job;
        {
            std::unique_lock<std::mutex> lock(m_);
            cv_.wait(lock, [&] { return stop_ || !queue_.empty(); });
            if (stop_) return;
            job = std::move(queue_.front());
            queue_.pop_front();
            writing_ = true;
            writing_entry_ = job.entry;   // best() still sees it; take() waits for its file
        }
        write(job);
        {
            std::lock_guard<std::mutex> lock(m_);
            writing_ = false;
            writing_entry_ = Entry{};
        }
        cv_.notify_all();
    }
}

void ConversationDiskTier::write(Job& job) {
    if (options_.before_write) options_.before_write();
    const size_t tokens = job.entry.live.ids.size();
    size_t made_room = 0, expired = 0;
    {
        std::lock_guard<std::mutex> lock(m_);
        expired = expire_locked();
        while (!entries_.empty() && bytes_ + job.entry.bytes > options_.budget_bytes) {
            remove_locked(0, true);
            ++made_room;
        }
        stats_.evicted += made_room;
    }
    const auto t0 = std::chrono::steady_clock::now();
    // one session file per image: the first stage's, then each later stage's (the writer takes no nested images)
    std::vector<SavedConversation> parts = std::move(job.image.stage_images);
    job.image.stage_images.clear();
    const SessionFileIdentity id = file_identity();
    SessionWriteOptions wo;
    wo.min_free_bytes = options_.min_free_bytes;
    wo.durable = false;   // a cache: a file a power loss damages fails its hash check and is read again from the prompt
    Entry e = std::move(job.entry);
    e.bytes = 0;
    std::string error;
    auto write_one = [&](const SavedConversation& img, size_t k) -> bool {
        char name[96];
        std::snprintf(name, sizeof name, "%s%016llx-%llu-%zu.bin", file_prefix(), (unsigned long long) nonce_,
                      (unsigned long long) e.id, k);
        const std::string path = (fs::path(options_.dir) / name).string();
        size_t written = 0;
        if (!session_file_write(path, img, id, written, error, wo)) {
            remove_file(path);
            return false;
        }
        e.files.push_back(path);
        e.bytes += written;
        return true;
    };
    bool ok = write_one(job.image, 0);
    for (size_t k = 0; ok && k < parts.size(); ++k) ok = write_one(parts[k], k + 1);
    if (ok && options_.persist) ok = write_index(e, error);   // after the images: an index names complete files only
    job.image = SavedConversation{};   // the RAM goes back now, written or not
    parts.clear();
    const double ms = ms_since(t0);
    if (!ok) {
        for (const auto& f : e.files) remove_file(f);
        {
            std::lock_guard<std::mutex> lock(m_);
            ++stats_.write_failed;
        }
        log(fmt("dropped %zu tokens: the write failed after %.0f ms (%s)", tokens, ms, error.c_str()));
        return;
    }
    e.at = now_();
    const uint64_t b = e.bytes;
    const size_t nfiles = e.files.size();
    size_t on_disk = 0;
    uint64_t disk_bytes = 0;
    {
        std::lock_guard<std::mutex> lock(m_);
        bytes_ += e.bytes;
        entries_.push_back(std::move(e));
        ++stats_.written;
        stats_.bytes_written += b;
        stats_.write_ms += ms;
        on_disk = entries_.size();
        disk_bytes = bytes_;
    }
    log(fmt("wrote %zu tokens: %.0f MiB in %zu file%s, %.0f ms (%.0f MB/s); on disk %zu, %.0f MiB; %zu removed for "
            "the budget, %zu expired", tokens, mib(b), nfiles, nfiles == 1 ? "" : "s", ms, mbps(b, ms), on_disk,
            mib(disk_bytes), made_room, expired));
}

bool ConversationDiskTier::take(uint64_t id, int64_t tokens, SavedConversation& out, std::string& error,
                                const std::function<bool(uint64_t, std::string&)>& admit) {
    out = SavedConversation{};
    const auto t0 = std::chrono::steady_clock::now();
    std::unique_lock<std::mutex> lock(m_);
    if (!open_) { error = "the disk tier is off"; return false; }
    // still queued: back from RAM, no disk at all
    for (size_t i = 0; i < queue_.size(); ++i)
        if (queue_[i].entry.id == id) {
            out = std::move(queue_[i].image);
            const uint64_t b = queue_[i].entry.bytes;
            queue_.erase(queue_.begin() + (std::ptrdiff_t) i);
            ++stats_.ram_hits;
            stats_.tokens_back += (uint64_t) std::max<int64_t>(tokens, 0);
            lock.unlock();
            log(fmt("RAM hit: %lld tokens back from the write queue (%.0f MiB, no disk) in %.1f ms",
                    (long long) tokens, mib(b), ms_since(t0)));
            log(summary());
            return true;
        }
    // being written: its file is complete when the write ends
    double waited = 0;
    if (writing_ && writing_entry_.id == id) {
        const auto w0 = std::chrono::steady_clock::now();
        cv_.wait(lock, [&] { return !(writing_ && writing_entry_.id == id); });
        waited = ms_since(w0);
    }
    size_t index = entries_.size();
    for (size_t i = 0; i < entries_.size(); ++i)
        if (entries_[i].id == id) index = i;
    if (index == entries_.size()) {
        ++stats_.read_failed;
        error = "the entry is gone (its write failed, or it expired)";
        return false;
    }
    const Entry want = entries_[index];
    const std::vector<std::string>& files = want.files;
    remove_locked(index, false);   // the entry goes now; its files are deleted after the read, whatever it gives
    lock.unlock();
    const SessionFileIdentity fid = file_identity();
    SessionReadLimits limits;
    limits.admit = admit;
    uint64_t bytes = 0;
    bool ok = true;
    for (size_t k = 0; ok && k < files.size(); ++k) {
        SavedConversation part;
        size_t n = 0;
        ok = session_file_read(files[k], fid, part, n, error, limits);
        bytes += n;
        if (!ok) break;
        if (k == 0) out = std::move(part);
        else out.stage_images.push_back(std::move(part));
    }
    if (ok && !holds(out, want.live, want.checkpoints, want.cvec)) {
        ok = false;
        error = "the files hold another conversation than their index";
    }
    for (const auto& f : files) remove_file(f);
    const double ms = ms_since(t0);
    lock.lock();
    if (!ok) {
        ++stats_.read_failed;
        lock.unlock();
        out = SavedConversation{};
        log(fmt("read of %lld tokens failed after %.0f ms (%s)", (long long) tokens, ms, error.c_str()));
        return false;
    }
    ++stats_.disk_hits;
    stats_.bytes_read += bytes;
    stats_.read_ms += ms - waited;
    stats_.tokens_back += (uint64_t) std::max<int64_t>(tokens, 0);
    lock.unlock();
    log(fmt("disk hit: %lld tokens, %.0f MiB in %zu file%s, %.0f ms (%.0f MB/s)%s", (long long) tokens, mib(bytes),
            files.size(), files.size() == 1 ? "" : "s", ms - waited, mbps(bytes, ms - waited),
            waited > 0 ? fmt(", after %.0f ms waiting for its write", waited).c_str() : ""));
    log(summary());
    return true;
}

bool ConversationDiskTier::pending() const {
    std::lock_guard<std::mutex> lock(m_);
    return !queue_.empty() || writing_;
}

void ConversationDiskTier::wait_idle() {
    std::unique_lock<std::mutex> lock(m_);
    if (queue_.empty() && !writing_) return;
    const auto t0 = std::chrono::steady_clock::now();
    const size_t depth = queue_.size() + (writing_ ? 1 : 0);
    cv_.wait(lock, [&] { return queue_.empty() && !writing_; });
    const double ms = ms_since(t0);
    ++stats_.waits;
    stats_.wait_ms += ms;
    lock.unlock();
    log(fmt("waited %.0f ms for %zu queued write%s (their RAM was needed)", ms, depth, depth == 1 ? "" : "s"));
}

size_t ConversationDiskTier::size() const {
    std::lock_guard<std::mutex> lock(m_);
    return entries_.size();
}

uint64_t ConversationDiskTier::bytes() const {
    std::lock_guard<std::mutex> lock(m_);
    return bytes_;
}

ConversationDiskStats ConversationDiskTier::stats() const {
    std::lock_guard<std::mutex> lock(m_);
    return stats_;
}

size_t ConversationDiskTier::clear() {
    std::unique_lock<std::mutex> lock(m_);
    const size_t queued = queue_.size();
    queue_.clear();
    cv_.wait(lock, [&] { return !writing_; });
    const size_t n = queued + entries_.size();
    while (!entries_.empty()) remove_locked(0, true);
    lock.unlock();
    if (n) log(fmt("removed all %zu conversations (#879)", n));
    return n;
}

std::string ConversationDiskTier::summary() const {
    std::lock_guard<std::mutex> lock(m_);
    const auto& s = stats_;
    return fmt("totals: %zu queued, %zu written (%.0f MiB, %.0f MB/s), %zu RAM hits, %zu disk hits (%.0f MiB, "
               "%.0f MB/s), %llu prompt tokens given back, %zu waits for RAM (%.0f ms), %zu expired, "
               "%zu removed for the budget, %zu/%zu writes/reads failed; on disk %zu (%.0f MiB), queue %zu",
               s.queued, s.written, mib(s.bytes_written), mbps(s.bytes_written, s.write_ms), s.ram_hits, s.disk_hits,
               mib(s.bytes_read), mbps(s.bytes_read, s.read_ms), (unsigned long long) s.tokens_back, s.waits,
               s.wait_ms, s.expired, s.evicted, s.write_failed, s.read_failed, entries_.size(), mib(bytes_),
               queue_.size() + (writing_ ? 1 : 0));
}

}  // namespace strata::core
