// src/core/conversation_disk.cpp - the disk tier of the parked conversations.  See the header.
#include "strata/core/conversation_disk.hpp"

#include "strata/core/conversation_file.hpp"

#include <algorithm>
#include <chrono>
#include <cstdarg>
#include <cstdio>
#include <filesystem>
#include <random>
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
    if (open_ && (dropped || !entries_.empty()))
        log(fmt("end: %zu queued dropped, %zu conversations removed from the disk", dropped, entries_.size()));
    while (!entries_.empty()) remove_locked(0, true);
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
    for (fs::directory_iterator it(dir, ec), end; !ec && it != end; it.increment(ec))
        if (it->is_regular_file(ec) && tier_file(it->path().filename().string())) {
            std::error_code rm;
            if (fs::remove(it->path(), rm)) ++stale;
        }
    if (ec) {
        error = "cannot list the folder " + options_.dir + ": " + ec.message();
        return false;
    }
    open_ = true;
    thread_ = std::thread([this] { writer(); });
    log(fmt("on: %s, budget %.0f MiB, %lld min per conversation; %zu stale files removed", options_.dir.c_str(),
            mib(options_.budget_bytes), (long long) (options_.ttl_s / 60), stale));
    return true;
}

void ConversationDiskTier::remove_locked(size_t index, bool delete_files) {
    Entry& e = entries_.at(index);
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
    const SessionFileIdentity id{nonce_, kTierConfig};
    SessionWriteOptions wo;
    wo.min_free_bytes = options_.min_free_bytes;
    wo.durable = false;   // the files live and die with this process: no flush to survive a power loss
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
    const std::vector<std::string> files = entries_[index].files;
    remove_locked(index, false);   // the entry goes now; its files are deleted after the read, whatever it gives
    lock.unlock();
    const SessionFileIdentity fid{nonce_, kTierConfig};
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
               "%.0f MB/s), %llu prompt tokens given back, %zu waits for RAM (%.0f ms), %zu expired, %zu removed for "
               "the budget, %zu/%zu writes/reads failed; on disk %zu (%.0f MiB), queue %zu",
               s.queued, s.written, mib(s.bytes_written), mbps(s.bytes_written, s.write_ms), s.ram_hits, s.disk_hits,
               mib(s.bytes_read), mbps(s.bytes_read, s.read_ms), (unsigned long long) s.tokens_back, s.waits,
               s.wait_ms, s.expired, s.evicted, s.write_failed, s.read_failed, entries_.size(), mib(bytes_),
               queue_.size() + (writing_ ? 1 : 0));
}

}  // namespace strata::core
