#pragma once

// SavedConversation ⇄ 规范字节镜像（磁盘停放层要往固定槽里放的东西）。
//
// ★ 两条硬要求：
//   1. 【不依赖内存布局】—— 逐字段显式编解码，绝不用 memcpy(struct)。
//      上游加字段时我们宁可编译期报错，也不要静默读到错位数据。
//   2. 【失败必须干净】—— unpack 返回 nullopt，绝不抛、绝不产生半成品。
//      磁盘上的东西损坏是常态（断电、半写、被手工改），上层会回退到重算。
//
// 格式（全部小端）：
//   magic   u32 = "SPK1"
//   version u32 = 1
//   内容摘要 u64（见下）—— 用于在 unpack 时快速判断整体是否可信
//   ... 逐字段 ...
//
// ★ 运行期兼容性字段（geometry / layer_lo / layer_hi）必须保留：
//   SavedConversation 的注释写明 "restore requires the same"。

#include "strata/core/conversation_cache.hpp"

#include <cstddef>
#include <optional>
#include <span>
#include <vector>

namespace strata::core {

// ★ 输出汇：打包时按块回调，调用方可以边收边写（落盘 / 填页），
//   于是【不需要构造完整中间缓冲】。
//
//   为什么重要：@256K 的一份镜像有 4-6 GiB，而打包发生在【停放路径】上 ——
//   那里正是物理内存最紧的时候。用一个 vector 接整份会让瞬时需求翻倍。
class PackSink {
public:
    virtual ~PackSink() = default;
    // 接收 [data, data+n)。返回 false 表示放弃，pack 会立刻停止。
    virtual bool write(const void* data, std::size_t n) = 0;
};

// 流式打包。sink 收到的字节与 pack_saved_conversation() 的返回值完全相同。
// 返回 false 表示 sink 中途拒绝了（此时不应认为镜像完整）。
[[nodiscard]] bool pack_saved_conversation_into(const SavedConversation& image, PackSink& sink);

// ★ 输入源：对称于 PackSink。解包时按需拉取，调用方可以边读分片边解析，
//   于是【不需要把整份字节流先物化出来】。
//
//   为什么重要：早先的 load() 是"读全量 → 再解析"，峰值 = 2× 快照。
//   @512K 的一份快照 7.3 GiB → 峰值 14.6 GiB，而可用内存只有 8.9 GiB
//   → 磁盘上的东西根本读不回来（写进去了却用不上）。
//   流式解析把这个峰值降到 1×（解析出的 SavedConversation 本身）。
class PackSource {
public:
    virtual ~PackSource() = default;
    // 读 n 字节到 dst。返回 false 表示读失败（调用方应视为镜像损坏）。
    virtual bool read(void* dst, std::size_t n) = 0;
};

// 流式解包。expected_bytes 是镜像的总长（来自 chunk 0 的探测页）；
// 长度不符或任何读取失败都返回 nullopt。
// ★ 失败时必须保持干净：不抛、不产生半成品。
[[nodiscard]] std::optional<SavedConversation> unpack_saved_conversation_from(
    PackSource& source, std::uint64_t expected_bytes);

// 便捷版（测试与小对象用）：在内存里攒出完整字节流。
[[nodiscard]] std::vector<std::byte> pack_saved_conversation(const SavedConversation& image);

// 从字节流还原。★ 任何不一致（magic、版本、长度越界、尾部多余）都返回 nullopt。
[[nodiscard]] std::optional<SavedConversation> unpack_saved_conversation(std::span<const std::byte> bytes);

// 打包后的大小（用于预分配与"装不装得下"的判断），不实际分配。
[[nodiscard]] std::size_t packed_saved_conversation_bytes(const SavedConversation& image);

// ---------------------------------------------------------------- P7：只读字节拆解（诊断）
//
// ★ 为什么要有它：第十二节把写放大的账算成「KV 是大头」，但把一份快照按 part 拆开之后
//   到底是什么占了多少，**静态推不出来**（见 docs/停放层流式改造设计.md §2.3）。
//   这三个函数【只数不写】：不分配、不改状态、不碰磁盘，只回答「这些字节会落在哪一类」。
//
//   用途：STRATA_PARK_BREAKDOWN=1 时打一行拆解，量清
//     A) 检查点链到底多大（决定检查点增量流是不是收益主体）
//     D) live 那份 GDN 是否每轮都必须整写（决定稳态一轮是 0.5 MB 还是 118 MB）
struct ParkBreakdown {
    // kv[]（每个 ConversationKv 一行）
    std::uint64_t kv_k = 0, kv_v = 0, kv_scale = 0, kv_pooled = 0;
    // 检查点：live 与 checkpoints[] 分开，各自按 part
    std::uint64_t live_gdn = 0, live_ple = 0, live_tail = 0, live_dead = 0, live_pos = 0;
    std::uint64_t live_ids = 0, live_imgs = 0, live_parts = 0, live_slots = 0;
    std::uint64_t ck_gdn = 0, ck_ple = 0, ck_tail = 0, ck_dead = 0, ck_pos = 0;
    std::uint64_t ck_ids = 0, ck_imgs = 0, ck_parts = 0, ck_slots = 0;
    // 其余
    std::uint64_t stage_images = 0;   // 各 stage 的【净字节】（已减去上一层 live/kv 的重复计入）
    std::uint64_t metadata = 0;       // geometry + layer_lo/hi + 各长度字段 + cvec + 两个魔数
    std::uint64_t checkpoints = 0;    // 检查点【份数】
    std::uint64_t tokens = 0;         // live.ids.size()
    std::uint64_t total = 0;          // 与 packed_saved_conversation_bytes() 必须【逐字节相等】
    // ★ 层切分（stage_images 非空）时的探针。实测结论（本机 stage_images 恒空，故为 0）：
    //   counted_saved() 对每个嵌套 stage 只算 counted_saved(si)，
    //   而 put_saved() 写的是 `8 + put_saved(si)`，并且 si 自己的 checkpoints[]、
    //   cvec、它们的长度字段都【不在】counted_saved(si) 里。
    //   ⇒ packed_saved_conversation_bytes() 在层切分下【少算】（少算多少 = 这两数之差）。
    //   ★ 后果有限：spool 的探测页写的是这个数，load() 拿它当 expected_bytes，
    //     偏小只会让最后不满一页的尾部少读几个字节；而它又是分配尺寸，
    //     小 -> 缓冲区偏小 -> out.visit 越界返回 false -> 读失败，不至于静默错。
    //   本机单卡不触发；一旦启用层切分，这个数必须相等。
    std::uint64_t stage_counted = 0, stage_written = 0;
};
[[nodiscard]] ParkBreakdown park_breakdown(const SavedConversation& image);

// 只算一个检查点的字节（用于「第 i 份检查点多大」这种逐份输出）。
[[nodiscard]] std::uint64_t park_checkpoint_bytes(const ConversationCheckpoint& c);

} // namespace strata::core
