#pragma once

// 磁盘停放层（L3）的页存储 —— 把被淘汰的停放条目落到 NVMe 上。
//
// 设计参考 ~/ninfer/ninfer-all/src/core/disk_kv_store.{h,cpp}（见 ~/strata-patches/0001-DESIGN.md）。
//
// ★ 核心保证：这整个组件是【纯性能优化】，任何失败都必须退化成"当作没有"。
//   · CRC 不符 / magic 不符 / 读失败  → 读作 miss，调用方回退到重算
//   · 磁盘满 / 写失败                → 返回 false，调用方照常释放内存
//   · 索引丢失 / 损坏                → 扫描数据文件重建
//   ★ 引擎的结果永不依赖这个文件的正确性。
//
// 三个从参考实现学来的关键决定：
//   1. 用 positioned read/write 而不是 mmap —— 磁盘满时映射写会 SIGBUS，
//      而 pread/pwrite 只返回错误码，纯缓存可以把它变成"没存"。
//   2. 槽自描述（每个槽带 magic + 身份 + CRC），索引只是加速；丢了就扫描重建。
//   3. 槽在被读时引用计数 +1，淘汰跳过它 —— 否则读到一半被覆盖。

#include <cstddef>
#include <cstdint>
#include <mutex>
#include <optional>
#include <set>
#include <span>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

namespace strata::core {

// 一个页的身份。
//
// ★ 键是【前缀的纯函数】—— 不需要任何额外持久化元数据：
//   lo/hi   = 前缀 token 的 128-bit 滚动摘要（同一前缀在任何会话里都算出同一个值）
//   tag     = 执行 profile（KV 格式 / stride / rope / 上下文档位），
//             ★ 不同 profile 的字节布局不同，必须隔离，否则会静默读错
//   frontier= 这个页覆盖到的 token 位置
struct DiskParkIdentity {
    std::uint64_t lo       = 0;   // 前缀的 128-bit 滚动摘要
    std::uint64_t hi       = 0;
    std::uint32_t tag      = 0;   // 执行 profile（见 disk_park_digest.hpp）
    std::uint32_t frontier = 0;   // 覆盖到的 token 位置
    // ★ 分片号。一个停放条目的状态有 4 GiB 量级，装不进固定槽，所以切成
    //   chunk 0（清单）· 1..K（数据片）。chunk 让同一个 frontier 的多片互不覆盖。
    std::uint32_t chunk    = 0;

    friend constexpr bool operator==(const DiskParkIdentity&,
                                     const DiskParkIdentity&) noexcept = default;
};

struct DiskParkIdentityHash {
    std::size_t operator()(const DiskParkIdentity& id) const noexcept {
        std::uint64_t h = id.lo;
        h ^= id.hi + 0x9e3779b97f4a7c15ULL + (h << 6U) + (h >> 2U);
        h ^= std::uint64_t{id.tag} * 0xff51afd7ed558ccdULL;
        h ^= std::uint64_t{id.frontier} * 0xc4ceb9fe1a85ec53ULL;
        h ^= std::uint64_t{id.chunk} * 0x9E3779B97F4A7C15ULL;
        h ^= h >> 31U;
        h *= 0xff51afd7ed558ccdULL;
        h ^= h >> 33U;
        return static_cast<std::size_t>(h);
    }
};

// 一个页的存储。固定槽大小、LRU 淘汰、CRC 校验、索引可扫描重建。
class DiskParkStore {
public:
    struct Options {
        std::string path;              // 数据文件；不存在则创建，★ 绝不截断
        std::size_t slot_size   = 0;   // 每个页的有效负载大小（字节）
        std::size_t capacity_bytes = 0; // 磁盘预算；用于推导槽数
        std::uint32_t max_slots = 0;   // 为 0 时由 capacity_bytes 推导
        bool verify_crc         = true;
        // 为 true 时新页不立即发布到索引，等 flush_index()；淘汰始终立即。
        // 所有者必须在退出前 flush；析构函数不 flush。
        bool defer_index_updates = false;
        // ★ 价值感知淘汰（对应 ninfer 的 --value-aware-demote）：
        //   在【最久未用的一批】里挑 frontier 最小的丢
        //   （frontier 小 = 重读便宜 = 先丢；frontier 大 = 重建贵 = 留下）。
        //   false = 纯 LRU（最旧先丢）。两者都是【条目级】：
        //   同一个 (lo,hi,tag,frontier) 的所有 chunk 一起丢。
        bool value_aware_demote = false;
    };

    // 打开或创建数据文件，并加载（或扫描重建）索引。
    // 抛 std::invalid_argument（参数非法）或 std::runtime_error（I/O 错误、
    // 或数据文件是按不同几何布局写的）。
    explicit DiskParkStore(Options options);
    ~DiskParkStore();

    DiskParkStore(const DiskParkStore&)            = delete;
    DiskParkStore& operator=(const DiskParkStore&) = delete;

    [[nodiscard]] std::size_t slot_size() const noexcept { return options_.slot_size; }
    [[nodiscard]] std::uint32_t slot_count() const noexcept { return max_slots_; }
    [[nodiscard]] std::uint32_t live_slots() const;
    [[nodiscard]] std::size_t used_bytes() const;
    [[nodiscard]] const std::string& path() const noexcept { return options_.path; }
    // ★ 本次打开是否靠"扫描数据文件"重建了索引（即索引文件丢了/坏了）。
    [[nodiscard]] bool index_rebuilt_from_scan() const noexcept { return rebuilt_from_scan_; }

    // 插入页；已存在则只刷新 LRU 时间戳。满时先淘汰最久未用的一个
    // （被淘汰的身份写进 *evicted）。★ 当满且所有槽都在被读时返回 false。
    bool upsert_page(const DiskParkIdentity& id, std::span<const std::byte> bytes,
                     std::vector<DiskParkIdentity>* evicted = nullptr);

    // 读一个页到 destination（需 slot_size 字节）：先校验身份、再校验 CRC，
    // 并刷新 LRU。★ 拷贝和 CRC 在校锁之外做；被读期间该槽不会被淘汰。
    bool read_page(const DiskParkIdentity& id, std::span<std::byte> destination);

    [[nodiscard]] bool contains(const DiskParkIdentity& id) const;
    bool touch(const DiskParkIdentity& id);
    // 丢弃一个页（若它还活着且没在被读）。
    bool evict(const DiskParkIdentity& id);
    [[nodiscard]] std::vector<DiskParkIdentity> live_identities() const;

    // 发布挂起的索引变更（原子替换，不是持久性屏障）。
    void flush_index();

    static constexpr std::size_t kSlotHeaderSize = 48;
    // CRC-32，反射多项式 0xEDB88320（zlib 用的那个）。
    [[nodiscard]] static std::uint32_t crc32(std::span<const std::byte> data) noexcept;

private:
    struct Header {
        std::uint32_t magic    = 0;
        std::uint32_t reserved = 0;
        std::uint64_t last_used = 0;
        std::uint64_t lo        = 0;
        std::uint64_t hi        = 0;
        std::uint32_t tag       = 0;
        std::uint32_t frontier  = 0;
        std::uint32_t crc       = 0;
        std::uint32_t chunk     = 0;   // 取代原来的 pad，★ 结构大小仍是 48
        static constexpr std::uint32_t kMagic = 0x4E44564B;  // "NDVK"
    };
    static_assert(sizeof(Header) == kSlotHeaderSize);

    struct IndexRow {
        std::uint64_t lo        = 0;
        std::uint64_t hi        = 0;
        std::uint32_t tag       = 0;
        std::uint32_t frontier  = 0;
        std::uint32_t slot      = 0;
        std::uint32_t chunk     = 0;   // 取代原来的 pad
    };

    struct IndexHeader {
        std::uint32_t magic     = 0;
        std::uint32_t version   = 0;
        std::uint32_t count     = 0;
        std::uint32_t slot_size = 0;
        std::uint32_t max_slots = 0;
        std::uint32_t pad       = 0;
        std::uint64_t clock     = 0;
        static constexpr std::uint32_t kMagic = 0x4944564B;  // "IDVK"
    };

    [[nodiscard]] std::size_t slot_pitch() const noexcept;
    [[nodiscard]] std::uint64_t payload_offset(std::uint32_t slot) const noexcept;
    [[nodiscard]] static bool matches(const Header& header, const DiskParkIdentity& id) noexcept;

    void open_file(bool create);
    void close_file() noexcept;
    bool load_index();
    void rebuild_from_scan();
    void persist_index_locked();
    void clear_slot(std::uint32_t slot) noexcept;
    // 淘汰。调用者持有 mutex_。
    // ★ 价值感知：在【最久未用的那一批】里挑 frontier 最小的丢
    //   （frontier 小 = 重读便宜 = 先丢；frontier 大 = 重建贵 = 留下）。
    // ★ 条目级：同一个 (lo,hi,tag,frontier) 的所有 chunk 一起丢 ——
    //   半个条目留在盘上会让 find() 命中而 load() 失败。
    bool evict_cheapest_locked(std::vector<DiskParkIdentity>* evicted);
    // 纯 LRU（value_aware_demote = false 时用）。同样条目级。
    bool evict_lru_entry_locked(std::vector<DiskParkIdentity>* evicted);
    [[nodiscard]] bool write_payload(std::uint32_t slot, std::span<const std::byte> bytes) noexcept;
    [[nodiscard]] bool read_payload(std::uint32_t slot, std::span<std::byte> destination) noexcept;
    void lru_touch_locked(std::uint32_t slot, std::uint64_t stamp);
    void lru_erase_locked(std::uint32_t slot) noexcept;

    mutable std::mutex mutex_;
    Options options_;
    std::size_t file_bytes_  = 0;
    std::uint32_t max_slots_ = 0;
    std::uint64_t clock_     = 0;
    bool rebuilt_from_scan_  = false;
    bool index_dirty_        = false;
    // ★ 打开时数据文件是否已有内容。只有它非空、而索引又不可用，才算"重建"；
    // 全新创建的库不该被报成重建。
    bool file_had_content_   = false;
    int file_handle_         = -1;

    std::unordered_map<DiskParkIdentity, std::uint32_t, DiskParkIdentityHash> index_;
    std::vector<std::uint32_t> free_slots_;
    std::vector<std::uint32_t> readers_;
    // 活槽按 (LRU 时间戳, 槽号) 排序，淘汰直接取最冷的那个而不用在锁下读遍所有头。
    static constexpr std::uint64_t kNoStamp = ~std::uint64_t{0};
    std::set<std::pair<std::uint64_t, std::uint32_t>> lru_;
    std::vector<std::uint64_t> lru_stamps_;
};

} // namespace strata::core
