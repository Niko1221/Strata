#pragma once

// 停放条目的磁盘暂存策略层：切片、查找、取回。
//
// 它把 P1/P2/P3/P4-a 拼成一个可用的整体，但不【碰】ConversationCache ——
// 这样策略能在独立测试里验证，最后接进缓存时的改动可以很小。
//
// ★ 恢复的语义（与 ninfer 一致）：只取【能匹配的最长 frontier】，
//   任何一步失败都返回 nullopt，调用方回退到重算。
//   磁盘上的东西坏了、被删了、装不下 —— 引擎的结果都不受影响，只是慢。

#include "strata/core/disk_park_bridge.hpp"
#include "strata/core/disk_park_digest.hpp"
#include "strata/core/disk_park_image.hpp"

#include <cstdint>
#include <limits>
#include <memory>
#include <optional>
#include <span>
#include <string>
#include <vector>

namespace strata::core {

class DiskParkSpool {
public:
    struct Options {
        std::string path;                       // 空 = 整个层关闭
        DiskParkProfile profile;                // 进 tag
        std::size_t state_stride       = 0;     // StateImage 家族的槽大小；0 = 关闭该家族
        std::size_t kv_stride          = 0;     // QsaKv 家族的槽大小；0 = 关闭该家族
        std::size_t state_capacity_bytes = 0;
        std::size_t kv_capacity_bytes    = 0;
        bool verify_crc = true;
        // ★ 前缀去重：写入新快照时，把被它严格覆盖的旧条目从盘上丢掉。
        //   同一个会话在盘上留多份快照是纯浪费 —— find() 只选最长的匹配，
        //   短的永远不会被读。实测一个会话被写了 6 次（121K→155K token）。
        bool supersede_prefixes = true;
    };

    struct Stats {
        std::uint64_t stores = 0, store_failures = 0, store_bytes = 0;
        std::uint64_t loads = 0, load_failures = 0, load_bytes = 0;
        std::uint64_t find_hits = 0, find_misses = 0;
        std::uint64_t evicted = 0, dups = 0;
        // ★ 前缀去重：被新快照覆盖、因而从盘上丢掉的旧条目页数
        std::uint64_t superseded = 0;
    };

    explicit DiskParkSpool(Options options);
    ~DiskParkSpool();
    DiskParkSpool(const DiskParkSpool&) = delete;
    DiskParkSpool& operator=(const DiskParkSpool&) = delete;

    [[nodiscard]] bool enabled() const noexcept { return enabled_; }
    [[nodiscard]] const DiskParkProfile& profile() const noexcept { return options_.profile; }

    // 把一个停放条目落到磁盘。
    // frontier/ids/imgs 用于算键；image 是内容。
    // 返回写入的字节数；0 表示失败（★ 调用方照常释放内存，不影响正确性）。
    std::size_t store(const SavedConversation& image, std::int64_t frontier,
                      std::span<const std::int32_t> ids,
                      std::span<const std::uint64_t> image_hashes,
                      std::span<const std::int64_t> image_starts);

    // ★ 一次命中的完整身份。
    //   注意【不能只用 frontier 标识一个条目】—— 两个不同的对话完全可能有相同的
    //   token 数（消息长度一样），此时只有前缀摘要能区分它们。
    //   集成测试就抓过这个 bug：只按 frontier 查会读回另一个对话的状态。
    struct Found {
        std::int64_t frontier = 0;
        PrefixDigest digest{};
    };

    // 在磁盘上找【最长】的、其前缀摘要与给定 prompt 匹配的条目。
    [[nodiscard]] std::optional<Found> find(std::span<const std::int32_t> ids,
                                            std::span<const std::uint64_t> image_hashes,
                                            std::span<const std::int64_t> image_starts) const;

    // ★ 非 const：读回会累加 stats_（与 restore() 的语义一致）
    [[nodiscard]] std::optional<SavedConversation> load(const Found& found);

    bool drop(const Found& found);

    [[nodiscard]] std::vector<std::int64_t> frontiers() const;
    [[nodiscard]] Stats stats() const;
    void flush();

private:
    // chunk 0 保留给"该 frontier 是否存在"的探测页（内容为 8 字节总长度）。
    // chunk 1..K 是数据片；chunk 1 的前 16 字节是头（magic/version/总长）。
    static constexpr std::uint32_t kChunkMagic   = 0x314B4353U;  // "SCK1"
    static constexpr std::uint32_t kChunkVersion = 1U;
    static constexpr std::size_t   kChunkHeader  = 16;

    [[nodiscard]] DiskParkIdentity make_id(std::int64_t frontier, const PrefixDigest& d,
                                           std::uint32_t chunk) const noexcept;
    [[nodiscard]] PrefixDigest digest_at(std::int64_t frontier, std::span<const std::int32_t> ids,
                                        std::span<const std::uint64_t> image_hashes,
                                        std::span<const std::int64_t> image_starts) const;

    Options options_;
    bool enabled_ = false;
    std::unique_ptr<DiskParkBridge> bridge_;
    mutable Stats stats_;
};

} // namespace strata::core
