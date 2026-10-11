#include "strata/core/disk_park_bridge.hpp"

#include <algorithm>
#include <cstdio>
#include <filesystem>
#include <system_error>

namespace strata::core {

const char* disk_park_kind_name(DiskParkKind kind) noexcept {
    switch (kind) {
        case DiskParkKind::StateImage: return "state";
        case DiskParkKind::QsaKv:      return "qsa";
        default:                       return "?";
    }
}

DiskParkBridge::DiskParkBridge(Options options) : options_(std::move(options)) {
    if (options_.base_path.empty()) return;   // ★ 关闭：和没有这一层完全一样

    namespace fs = std::filesystem;
    std::error_code ec;
    fs::create_directories(options_.base_path, ec);
    if (ec) {
        std::fprintf(stderr, "strata serve: disk park: cannot create %s: %s\n",
                     options_.base_path.c_str(), ec.message().c_str());
        return;
    }

    for (std::size_t k = 0; k < static_cast<std::size_t>(DiskParkKind::Count); ++k) {
        const std::size_t stride = options_.stride[k];
        if (stride == 0) continue;            // 该 family 关闭
        DiskParkStore::Options so;
        so.path = (fs::path(options_.base_path) /
                   (std::string("park-") + disk_park_kind_name(static_cast<DiskParkKind>(k)) + ".dat"))
                      .string();
        so.slot_size      = stride;
        so.capacity_bytes = options_.capacity_bytes[k];   // ★ 该 family 自己的预算
        so.verify_crc          = options_.verify_crc;
        so.defer_index_updates = true;        // 索引批量发布；退出前 flush
        try {
            families_[k].store  = std::make_unique<DiskParkStore>(so);
            families_[k].stride = stride;
        } catch (const std::exception& e) {
            // ★ 打不开就关掉这一层 —— 绝不能因此让引擎起不来
            std::fprintf(stderr, "strata serve: disk park: %s disabled: %s\n",
                         disk_park_kind_name(static_cast<DiskParkKind>(k)), e.what());
            families_[k].store.reset();
            families_[k].stride = 0;
        }
    }

    for (const auto& f : families_) enabled_ = enabled_ || (f.store != nullptr);
    if (!enabled_) return;

    workers_.reserve(kWorkerCount);
    for (std::size_t i = 0; i < kWorkerCount; ++i) workers_.emplace_back([this] { worker_loop(); });
}

DiskParkBridge::~DiskParkBridge() {
    {
        std::lock_guard lock(queue_mutex_);
        quit_ = true;
        queue_cv_.notify_all();
    }
    for (auto& t : workers_) if (t.joinable()) t.join();
    // ★ 析构不做 flush（DiskParkStore 的契约）；所有者必须先调 flush。
    for (auto& f : families_) if (f.store) f.store->flush_index();
}

void DiskParkBridge::worker_loop() {
    for (;;) {
        Job job;
        {
            std::unique_lock lock(queue_mutex_);
            queue_cv_.wait(lock, [this] { return quit_ || !queue_.empty(); });
            if (queue_.empty()) {
                if (quit_) return;
                continue;
            }
            job = std::move(queue_.front());
            queue_.pop_front();
            ++in_flight_;
            // ★ 出队后立刻通知：spill_wait 在同一个 CV 上等空位，
            //   不通知的话队列满时生产者只能干等到超时。
            queue_cv_.notify_one();
        }
        apply(job);
        {
            std::lock_guard lock(queue_mutex_);
            --in_flight_;
            if (queue_.empty() && in_flight_ == 0) idle_cv_.notify_all();
        }
    }
}

void DiskParkBridge::apply(Job& job) {
    if (!valid_kind(job.kind)) { finish(job, SpillStatus::Failed, false); return; }
    Family& f = family(job.kind);
    if (!f.store) { finish(job, SpillStatus::Failed, false); return; }

    const std::span<const std::byte> bytes =
        job.owned.empty() ? job.borrowed : std::span<const std::byte>(job.owned);

    if (job.probe) {
        finish(job, f.store->contains(job.id) ? SpillStatus::Present : SpillStatus::Missing, false);
        return;
    }
    if (bytes.size() != f.stride) { finish(job, SpillStatus::Failed, false); return; }

    const bool was_present = f.store->contains(job.id);
    std::vector<DiskParkIdentity> evicted;
    const bool ok = f.store->upsert_page(job.id, bytes, &evicted);

    {
        std::lock_guard lock(stats_mutex_);
        if (ok) {
            ++stats_.spills;
            stats_.spill_bytes += bytes.size();
            if (was_present) ++stats_.spill_dups;
            stats_.evicted_pages += evicted.size();
        } else {
            ++stats_.failures;
        }
    }
    f.store->flush_index();
    finish(job, ok ? SpillStatus::Written : SpillStatus::Failed, false);
}

void DiskParkBridge::finish(Job& job, SpillStatus status, bool) {
    if (job.completion) job.completion->status.store(status, std::memory_order_release);
}

// ---------------------------------------------------------------- 写

bool DiskParkBridge::spill_wait(const DiskParkIdentity& id, DiskParkKind kind,
                                std::span<const std::byte> bytes,
                                std::chrono::milliseconds timeout) {
    if (!enabled_ || !valid_kind(kind)) return false;
    Family& f = family(kind);
    if (!f.store || bytes.size() != f.stride) return false;

    Job job;
    job.id         = id;
    job.kind       = kind;
    job.owned.assign(bytes.begin(), bytes.end());   // ★ 拷贝入队
    job.completion = std::make_shared<SpillCompletion>();
    const SpillTicket ticket = job.completion;

    {
        std::unique_lock lock(queue_mutex_);
        if (!queue_cv_.wait_for(lock, timeout, [this] { return queue_.size() < kQueueCapacity; })) {
            std::lock_guard slock(stats_mutex_);
            ++stats_.queue_drops;
            return false;
        }
        queue_.push_back(std::move(job));
    }
    queue_cv_.notify_one();

    // ★ 真正等到落定：写前缀链不能留洞，调用方需要知道结果。
    const auto deadline = std::chrono::steady_clock::now() + timeout;
    for (;;) {
        const SpillStatus st = poll(ticket);
        if (st == SpillStatus::Written || st == SpillStatus::Present) return true;
        if (st == SpillStatus::Failed || st == SpillStatus::Missing)  return false;
        if (std::chrono::steady_clock::now() >= deadline) return false;
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
}

bool DiskParkBridge::spill_sync(const DiskParkIdentity& id, DiskParkKind kind,
                                std::span<const std::byte> bytes) {
    if (!enabled_ || !valid_kind(kind)) return false;
    Family& f = family(kind);
    if (!f.store || bytes.size() != f.stride) return false;

    const bool was_present = f.store->contains(id);
    std::vector<DiskParkIdentity> evicted;
    const bool ok = f.store->upsert_page(id, bytes, &evicted);
    {
        std::lock_guard lock(stats_mutex_);
        if (ok) {
            ++stats_.spills;
            stats_.spill_bytes += bytes.size();
            if (was_present) ++stats_.spill_dups;
            stats_.evicted_pages += evicted.size();
        } else {
            ++stats_.failures;
        }
    }
    f.store->flush_index();
    return ok;
}

SpillTicket DiskParkBridge::try_probe(const DiskParkIdentity& id, DiskParkKind kind) {
    if (!enabled_ || !valid_kind(kind) || !family(kind).store) return nullptr;
    Job job;
    job.id         = id;
    job.kind       = kind;
    job.probe      = true;
    job.completion = std::make_shared<SpillCompletion>();
    const SpillTicket ticket = job.completion;      // ★ 在 move 之前取出
    {
        std::lock_guard lock(queue_mutex_);
        if (queue_.size() >= kQueueCapacity) {
            std::lock_guard slock(stats_mutex_);
            ++stats_.queue_drops;
            return nullptr;
        }
        queue_.push_back(std::move(job));
    }
    queue_cv_.notify_one();
    return ticket;
}

SpillTicket DiskParkBridge::try_submit(const DiskParkIdentity& id, DiskParkKind kind,
                                       std::span<const std::byte> bytes) {
    if (!enabled_ || !valid_kind(kind)) return nullptr;
    Family& f = family(kind);
    if (!f.store || bytes.size() != f.stride) return nullptr;

    Job job;
    job.id       = id;
    job.kind     = kind;
    job.borrowed = bytes;                            // ★ 借用，调用方保证存活
    job.completion = std::make_shared<SpillCompletion>();
    const SpillTicket ticket = job.completion;
    {
        std::lock_guard lock(queue_mutex_);
        if (queue_.size() >= kQueueCapacity) {
            std::lock_guard slock(stats_mutex_);
            ++stats_.queue_drops;
            return nullptr;
        }
        queue_.push_back(std::move(job));
    }
    queue_cv_.notify_one();
    return ticket;
}

void DiskParkBridge::wait_idle() {
    std::unique_lock lock(queue_mutex_);
    idle_cv_.wait(lock, [this] { return queue_.empty() && in_flight_ == 0; });
}

// ---------------------------------------------------------------- 读

bool DiskParkBridge::restore(const DiskParkIdentity& id, DiskParkKind kind,
                             std::span<std::byte> destination) {
    if (!enabled_ || !valid_kind(kind)) return false;
    Family& f = family(kind);
    if (!f.store) return false;
    const bool ok = f.store->read_page(id, destination);
    std::lock_guard lock(stats_mutex_);
    if (ok) { ++stats_.restores; stats_.restore_bytes += destination.size(); }
    else    { ++stats_.restore_misses; }
    return ok;
}

bool DiskParkBridge::restore_many(std::span<const DiskParkIdentity> ids, DiskParkKind kind,
                                  std::span<std::byte> destination) {
    if (!enabled_ || !valid_kind(kind)) return false;
    Family& f = family(kind);
    if (!f.store) return false;
    if (destination.size() < ids.size() * f.stride) return false;

    bool all = true;
    for (std::size_t i = 0; i < ids.size(); ++i) {
        auto dst = destination.subspan(i * f.stride, f.stride);
        if (!f.store->read_page(ids[i], dst)) { all = false; break; }   // ★ 遇缺即停
    }
    std::lock_guard lock(stats_mutex_);
    if (all) { stats_.restores += ids.size(); stats_.restore_bytes += ids.size() * f.stride; }
    else     { ++stats_.restore_misses; }
    return all;
}

bool DiskParkBridge::drop(const DiskParkIdentity& id, DiskParkKind kind) {
    if (!enabled_ || !valid_kind(kind)) return false;
    Family& f = family(kind);
    return f.store && f.store->evict(id);
}

bool DiskParkBridge::contains(const DiskParkIdentity& id, DiskParkKind kind) const {
    if (!enabled_ || !valid_kind(kind)) return false;
    const Family& f = family(kind);
    return f.store && f.store->contains(id);
}

std::vector<std::uint32_t> DiskParkBridge::live_frontiers(DiskParkKind kind) const {
    std::vector<std::uint32_t> out;
    if (!enabled_ || !valid_kind(kind)) return out;
    const Family& f = family(kind);
    if (!f.store) return out;
    for (const DiskParkIdentity& id : f.store->live_identities()) out.push_back(id.frontier);
    std::sort(out.begin(), out.end());
    out.erase(std::unique(out.begin(), out.end()), out.end());
    return out;
}

std::vector<DiskParkIdentity> DiskParkBridge::live_identities(DiskParkKind kind) const {
    if (!enabled_ || !valid_kind(kind)) return {};
    const Family& f = family(kind);
    return f.store ? f.store->live_identities() : std::vector<DiskParkIdentity>{};
}

DiskParkStats DiskParkBridge::stats() const {
    std::lock_guard lock(stats_mutex_);
    return stats_;
}

std::uint32_t DiskParkBridge::slot_count(DiskParkKind kind) const {
    if (!valid_kind(kind)) return 0;
    const Family& f = family(kind);
    return f.store ? f.store->slot_count() : 0;
}

bool DiskParkBridge::index_rebuilt_from_scan(DiskParkKind kind) const {
    if (!valid_kind(kind)) return false;
    const Family& f = family(kind);
    return f.store && f.store->index_rebuilt_from_scan();
}

} // namespace strata::core
