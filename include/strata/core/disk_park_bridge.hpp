#pragma once

// 磁盘停放层的引擎门面（L3）：每个 family 一个 DiskParkStore，
// 一个有界写队列 + 若干写线程，以及恢复用的同步读。
//
// 设计参考 ~/ninfer/ninfer-all/src/core/disk_kv_bridge.{h,cpp}。
//
// ★ 贯穿全层的契约（必须始终成立）：
//   任何失败 —— 队列满、磁盘满、CRC 不符、页不存在、I/O 错误 —— 都只表现为
//   "没存下"或"读不到"，调用方【必须】回退到重算。这一层永远不改变引擎的结果。
//
// family 的划分：
//   StateImage = GDN 递推 + PLE 历史 + indexer 状态（每个 frontier 一份，整块）
//   QsaKv      = QSA 层的 K/V（按 token 块分页）

#include "strata/core/disk_park_store.hpp"

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <deque>
#include <memory>
#include <mutex>
#include <span>
#include <string>
#include <thread>
#include <vector>

namespace strata::core {

enum class DiskParkKind : std::uint8_t {
    StateImage = 0,
    QsaKv      = 1,
    Count      = 2,
};

const char* disk_park_kind_name(DiskParkKind kind) noexcept;

// 一次异步写入的完成状态。Written 表示"可读了"，不表示"已持久化到索引"。
enum class SpillStatus : std::uint8_t { Pending, Present, Missing, Written, Failed };

struct SpillCompletion {
    std::atomic<SpillStatus> status{SpillStatus::Pending};
};

using SpillTicket = std::shared_ptr<SpillCompletion>;

struct DiskParkStats {
    std::uint64_t spills         = 0;   // 成功写入的页数
    std::uint64_t spill_bytes    = 0;
    std::uint64_t spill_dups     = 0;   // 已存在、只刷新 LRU 的次数
    std::uint64_t restores       = 0;   // 成功读回的页数
    std::uint64_t restore_bytes  = 0;
    std::uint64_t restore_misses = 0;   // ★ 读不到（含 CRC 不符）
    std::uint64_t evicted_pages  = 0;   // 被 LRU 淘汰的页数
    std::uint64_t queue_drops    = 0;   // 队列满被拒的次数
    std::uint64_t failures       = 0;   // I/O 失败次数
};

class DiskParkBridge {
public:
    struct Options {
        std::string base_path;                       // 空 = 整个磁盘层关闭
        std::size_t stride[2]         = {0, 0};      // 每个 family 的页大小；0 = 该 family 关闭
        // ★ 每个 family 各自的磁盘预算（字节），下标同 DiskParkKind。
        //   不用比例切分：Strata 的 StateImage（@256K 约 4.1 GiB/份）比 QSA KV（2.11 GiB）还大，
        //   固定比例会给出反直觉的结果。调用方按实际需要各自算。
        std::size_t capacity_bytes[2] = {0, 0};
        bool verify_crc               = true;
    };

    explicit DiskParkBridge(Options options);
    ~DiskParkBridge();

    DiskParkBridge(const DiskParkBridge&)            = delete;
    DiskParkBridge& operator=(const DiskParkBridge&) = delete;

    // 是否启用（base_path 非空且至少一个 family 有 stride）。
    [[nodiscard]] bool enabled() const noexcept { return enabled_; }

    // ---- 写 ----
    // 把页拷进队列，最多等 timeout 腾出位置。
    // ★ 写一条前缀链时不能留洞（恢复遇到第一个缺失页就停），所以用等待版。
    // 已存在或成功入队返回 true。
    bool spill_wait(const DiskParkIdentity& id, DiskParkKind kind,
                    std::span<const std::byte> bytes, std::chrono::milliseconds timeout);

    // 在调用线程上直接写，不排队拷贝 —— 适合单块大对象（StateImage）。
    bool spill_sync(const DiskParkIdentity& id, DiskParkKind kind,
                    std::span<const std::byte> bytes);

    // 非阻塞探测与提交。nullptr 表示队列忙，稍后重试。
    // ★ 提交的字节是【借用】的：必须保持存活且不变，直到 ticket 离开 Pending（含取消）。
    [[nodiscard]] SpillTicket try_probe(const DiskParkIdentity& id, DiskParkKind kind);
    [[nodiscard]] SpillTicket try_submit(const DiskParkIdentity& id, DiskParkKind kind,
                                         std::span<const std::byte> bytes);

    [[nodiscard]] static SpillStatus poll(const SpillTicket& ticket) noexcept {
        return ticket->status.load(std::memory_order_acquire);
    }

    void wait_idle();

    // ---- 读 ----
    // 读一个页到 destination（family 的 stride 字节），在调用线程上完成。
    [[nodiscard]] bool restore(const DiskParkIdentity& id, DiskParkKind kind,
                              std::span<std::byte> destination);

    // 连续读多个页到 destination（ids.size() 个 stride）。任何一个缺失即返回 false。
    [[nodiscard]] bool restore_many(std::span<const DiskParkIdentity> ids, DiskParkKind kind,
                                    std::span<std::byte> destination);

    // 丢弃一个页（索引里有但读不出来时用，免得后续一直选中它）。
    bool drop(const DiskParkIdentity& id, DiskParkKind kind);

    [[nodiscard]] bool contains(const DiskParkIdentity& id, DiskParkKind kind) const;

    // 某个 family 里所有已存 StateImage 的 frontier —— ★ 恢复时的候选起点。
    [[nodiscard]] std::vector<std::uint32_t> live_frontiers(DiskParkKind kind) const;

    // ★ 更完整的视图：带摘要、tag 与 chunk 的全部身份。
    //   恢复时需要用它来"只挑属于本 profile 的条目"，并反查某个 frontier 的摘要。
    [[nodiscard]] std::vector<DiskParkIdentity> live_identities(DiskParkKind kind) const;

    [[nodiscard]] DiskParkStats stats() const;
    [[nodiscard]] std::uint32_t slot_count(DiskParkKind kind) const;
    [[nodiscard]] bool index_rebuilt_from_scan(DiskParkKind kind) const;

private:
    struct Family {
        std::unique_ptr<DiskParkStore> store;
        std::size_t stride = 0;
    };

    struct Job {
        DiskParkIdentity id;
        DiskParkKind kind = DiskParkKind::StateImage;
        std::vector<std::byte> owned;                 // 拷贝入队的字节
        std::span<const std::byte> borrowed;          // ★ 借用调用方的字节
        SpillTicket completion;
        bool probe = false;
        bool queued = false;
    };

    void worker_loop();
    void apply(Job& job);
    void finish(Job& job, SpillStatus status, bool evict_reported);

    [[nodiscard]] const Family& family(DiskParkKind kind) const {
        return families_[static_cast<std::size_t>(kind)];
    }
    [[nodiscard]] Family& family(DiskParkKind kind) {
        return families_[static_cast<std::size_t>(kind)];
    }
    [[nodiscard]] static bool valid_kind(DiskParkKind kind) noexcept {
        return static_cast<std::size_t>(kind) < static_cast<std::size_t>(DiskParkKind::Count);
    }

    // 参考实现用了 8 个写线程（NVMe 带宽需要多个流），队列 512。
    static constexpr std::size_t kQueueCapacity = 512;
    static constexpr std::size_t kWorkerCount   = 4;   // 停放场景比 ninfer 温和，4 个够

    Options options_;
    bool enabled_ = false;
    Family families_[static_cast<std::size_t>(DiskParkKind::Count)];

    mutable std::mutex stats_mutex_;
    mutable DiskParkStats stats_;

    std::mutex queue_mutex_;
    std::condition_variable queue_cv_;
    std::condition_variable idle_cv_;
    std::deque<Job> queue_;
    std::size_t in_flight_ = 0;
    bool quit_             = false;
    std::vector<std::thread> workers_;
};

} // namespace strata::core
