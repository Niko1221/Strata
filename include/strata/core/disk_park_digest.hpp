#pragma once

// 磁盘停放层的身份：前缀的滚动 128-bit 摘要 + 执行 profile 标签。
//
// ★ 核心性质：摘要只取决于【元素序列】，与计算的时机无关。
//   写盘时引擎从活跃序列算，恢复时从新请求的 prompt 算 —— 两者必然相同。
//   于是磁盘上不需要任何额外的持久化元数据（没有"索引键"要维护、要同步）。
//
// ★ 必须与 ConversationCache 的匹配语义严格一致
//   （include/strata/core/conversation_cache.hpp:167 conversation_prefix）：
//     · 前缀是检查点 ids 的【全部】元素，且必须【严格】短于 prompt
//     · 只计入 start < frontier 的图片，按 images 给出的顺序
//
// ★ profile 标签必须涵盖任何【改变 KV 字节布局】的配置：
//   KV 格式（int8 / q4_0 / k8v4 / fp16）· rope 设置 · 模型身份 · 上下文档位。
//   标签不同 → 身份不同 → 读不到别的配置的页（这是正确性要求，不是优化）。

#include <cstdint>
#include <span>
#include <string>
#include <vector>

namespace strata::core {

struct ConversationImageKey;   // conversation_cache.hpp

// 前缀的 128-bit 摘要。
struct PrefixDigest {
    std::uint64_t lo = 0;
    std::uint64_t hi = 0;
    friend constexpr bool operator==(const PrefixDigest&, const PrefixDigest&) noexcept = default;
};

// 滚动摘要构造器。
//
// ★★ API 只提供【一种】折入顺序 —— 有意如此。
//   摘要对折入顺序敏感，而"token 与图片谁先谁后"在语义上并不重要；
//   若放任调用方自由交替，写盘侧与恢复侧极易不一致（而症状是永远 miss）。
//   所以这里不暴露 push_token / push_image，只有 advance()。
//
// 每个 frontier 的规范折入顺序 = 交错的
//     token[0] · (start<1 的图) · token[1] · (start<2 的图) · …
class PrefixDigestBuilder {
public:
    PrefixDigestBuilder() { reset(); }

    void reset() noexcept;

    // 折入一个 token；随后自动折入所有 start < （已折入 token 数）的图片，按序。
    // ★ 这是唯一的推进方式。
    void advance(std::int32_t id, std::span<const std::uint64_t> image_hashes,
                 std::span<const std::int64_t> image_starts);

    [[nodiscard]] PrefixDigest value() const noexcept { return {lo_, hi_}; }

private:
    void mix_token(std::int32_t id) noexcept;
    void mix_image(std::int64_t start, std::uint64_t hash) noexcept;

    std::uint64_t lo_ = 0, hi_ = 0;
    std::int64_t tokens_ = 0;
    std::size_t image_cursor_ = 0;
};

// 从完整的 ids / imgs 直接算某个 frontier 的摘要（不增量）。
// frontier == ids.size() 时就是"整个 ids"的摘要。
// ★ 用于测试与"从 prompt 算"的路径。
[[nodiscard]] PrefixDigest prefix_digest(std::span<const std::int32_t> ids,
                                         std::span<const std::uint64_t> image_hashes,
                                         std::span<const std::int64_t> image_starts,
                                         std::int64_t frontier);

// ------------------------------------------------------------------ profile 标签

// 把"会改变 KV 字节布局"的配置规范化成一个 32-bit 标签。
struct DiskParkProfile {
    std::string model;          // 模型身份（文件名或指纹）
    std::string kv_format;      // "int8" / "q4_0" / "k8v4" / "fp16"
    std::int64_t max_context = 0;
    std::string rope;           // "none" / "yarn:2" / "yarn:4" …
    // §7.2 版本门：引擎版本（以及手动加的数值摘要）也进标签。
    //   0.1.41 写的页不许被 0.1.43 读，反过来也一样：版本不同 = 数值路径不同
    //   = 同一批 token 的 K/V 字节不同。以前 tag 里没有版本，跨版本静默命中。
    std::string engine;         // "0.1.43"，或 "0.1.43-tail0" 这类手动摘要键

    // FNV-1a 32 位，作用在规范化字符串上。
    // ★ 任何字段变化都会改变标签；不同标签的页永不互相读取。
    [[nodiscard]] std::uint32_t tag() const noexcept;

    // 规范化描述（供日志与诊断："tag 变了"时要能看出为什么）。
    [[nodiscard]] std::string describe() const;
};

} // namespace strata::core
