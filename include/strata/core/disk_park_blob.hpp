#pragma once

// 停放层【检查点状态流】：一个会话一个文件，只追加、按内容摘要去重。
//
// 为什么需要它（阶段 0 实测）：
//   一份 116,737 token 的快照里，**检查点链占 51.6%**（1468.7 MiB / 2847.4 MiB），
//   而链的 **91.7% 是每份检查点里的 GDN 状态**（112.2 MiB × 12，**每份都是完整拷贝**）。
//   现在每一轮换出都把这一整条链重写一遍 —— 那正是写放大的主体。
//
// 这个流的做法：
//   · 检查点状态 **只在第一次见到时写一次**，之后每轮只写"它在流里的区间"（32 字节）。
//   · 去重键 = (ids, imgs) 的 128-bit 滚动摘要 —— 与快照层同一套算法，
//     所以同一段前缀在任何时候算出的键都一样，不需要额外的索引文件。
//   · 只追加：旧字节永不移动，于是区间一旦记下就一直有效。
//
// ★ 失败语义与整层一致：任何失败都只表现为"这一层当作没有"，调用方回退到重算。
// ★ 本文件不碰 CUDA、不碰引擎状态：字节由调用方以回调给出/取走，可在无 GPU 环境完整测试。

#include <cstddef>
#include <cstdint>
#include <functional>
#include <map>
#include <span>
#include <string>
#include <vector>

namespace strata::core {

// 一段已落盘数据的区间。
struct DiskParkBlobRef {
    std::uint64_t offset = 0;   // 在 checkpoints.bin 里的字节偏移（0 = 无）
    std::uint64_t bytes  = 0;
    [[nodiscard]] bool present() const noexcept { return bytes != 0; }
    friend constexpr bool operator==(const DiskParkBlobRef&, const DiskParkBlobRef&) noexcept = default;
};

// 一个检查点状态的键：ids + imgs 的内容摘要。
struct DiskParkBlobKey {
    std::uint64_t lo = 0, hi = 0;
    friend constexpr bool operator==(const DiskParkBlobKey&, const DiskParkBlobKey&) noexcept = default;
    friend constexpr bool operator<(const DiskParkBlobKey& a, const DiskParkBlobKey& b) noexcept {
        return a.lo != b.lo ? a.lo < b.lo : a.hi < b.hi;
    }
};

class DiskParkBlobStream {
public:
    struct Options {
        std::string path;             // checkpoints.bin 的完整路径；空 = 关闭
        std::size_t max_tracked = 4096;   // 记住多少个键（超出就淘汰最久未用的）
    };

    struct Stats {
        std::uint64_t appends = 0;      // 真正写入的次数
        std::uint64_t dedup_hits = 0;   // 命中已有区间、因而【零写入】的次数
        std::uint64_t bytes = 0;        // 累计写入字节
        std::uint64_t evicted_keys = 0; // 因记忆上限被丢掉的键（下次会重写一份）
        std::uint64_t failures = 0;
    };

    // ★★ 每个条目在文件里【自带 key】（44 字节条目头：magic/version/key/长度）。
    //   为什么必须自带：去重靠一张"key → 区间"的表，而那张表只在内存里 ——
    //   引擎重启后表就空了，于是同一个检查点会被重写一遍，**去重等于没用**。
    //   把 key 写进条目头，open() 时扫一遍就能把表重建起来。
    //   （第一版没写 key，测试第 3 节直接把这个坑抓了出来。）

    // 填充回调：把【整份 blob】写进 dst（长度 n）。offset 是 blob 内偏移。
    using Fill = std::function<bool(std::uint64_t offset, std::uint8_t* dst, std::size_t n)>;
    // 取用回调：把 blob 的 [offset, offset+n) 交给调用方。
    using Read = std::function<bool(std::uint64_t offset, const std::uint8_t* src, std::size_t n)>;

    explicit DiskParkBlobStream(Options options);
    ~DiskParkBlobStream();
    DiskParkBlobStream(const DiskParkBlobStream&) = delete;
    DiskParkBlobStream& operator=(const DiskParkBlobStream&) = delete;

    [[nodiscard]] bool enabled() const noexcept { return fd_ >= 0; }
    [[nodiscard]] Stats stats() const { return stats_; }
    [[nodiscard]] std::uint64_t bytes_on_disk() const noexcept { return size_; }

    // 把一份 blob 放进去：命中已有键就直接返回旧区间（零写入）。
    // 未命中则追加整份（分块从 fill 取）。
    [[nodiscard]] DiskParkBlobRef put(const DiskParkBlobKey& key, std::uint64_t bytes, const Fill& fill);

    // 查一个键的区间（不写）。
    [[nodiscard]] DiskParkBlobRef find(const DiskParkBlobKey& key) const;

    // ★★★ P16（2026-10-11）：值不值得再存一份检查点。
    //   为什么需要：一份检查点固定 118,063,152 B（112.6 MiB），其中 99.65% 是【与对话
    //   长短无关】的 GDN 递归状态（gdn=117,669,888；ids=17,959 与 ids=134,683 两份逐字节
    //   同尺寸）；而本流是【整块内容寻址】—— 相邻两份 99.7% 的字节相同、键却不同 ⇒
    //   每轮整块重写 112.6 MiB，与对话多长无关，只与【轮数】成正比。
    //   判据沿用引擎里已有的那一套（generate.cpp 的 kMinSpillTokens）：
    //   "写盘的 SSD 磨损 > 省下的重读时间，不划算"。跳过之后，下次从更早的检查点挂载、
    //   把那一段按正常 prompt 读回填 —— 引擎本来就支持（"剩下的 N 个 token 由正常读回填"）。
    //   group 用来分组（一条会话一个组，用会话目录名）；min_advance <= 0 = 关掉闸门。
    //   min_frontier：绝对下限 —— 比它短的会话根本不值得有检查点（重读整段比写一份
    //   112.6 MiB 便宜得多）。与 KV spill 的 kMinSpillTokens 是同一条理由。
    [[nodiscard]] bool worth_storing(const std::string& group, std::int64_t frontier,
                                     std::int64_t min_advance,
                                     std::int64_t min_frontier = 0) const;
    // 真的写进去之后调用，记住这次的 frontier（只前进、不后退）。
    void note_stored(const std::string& group, std::int64_t frontier);

    // 读一个区间。
    [[nodiscard]] bool get(const DiskParkBlobRef& ref, const Read& sink) const;

private:
    struct Track { DiskParkBlobRef ref; std::uint64_t stamp = 0; };
    void trim();

    Options options_;
    int fd_ = -1;
    std::uint64_t size_ = 0;            // 文件当前长度
    std::uint64_t clock_ = 0;
    std::map<DiskParkBlobKey, Track> tracked_;
    std::map<std::string, std::int64_t> last_frontier_;   // P16：每条会话上次落盘的 frontier
    mutable Stats stats_{};
};

} // namespace strata::core
