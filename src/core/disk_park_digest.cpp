#include "strata/core/disk_park_digest.hpp"

#include <algorithm>
#include <cstdio>

namespace strata::core {
namespace {

// 两个交错的 SplitMix64 流，构成 128-bit 状态。
// ★ 只用定长整数运算 —— 跨编译器/跨平台/跨优化级别都得到同一个值。
//   （摘要进磁盘文件名与身份，不确定性会让恢复永远 miss。）
constexpr std::uint64_t kSeedLo = 0x243F6A8885A308D3ULL;
constexpr std::uint64_t kSeedHi = 0x13198A2E03707344ULL;
constexpr std::uint64_t kGamma  = 0x9E3779B97F4A7C15ULL;

inline void mix128(std::uint64_t& lo, std::uint64_t& hi, std::uint64_t x) noexcept {
    lo += x + kGamma;
    std::uint64_t z = lo;
    z = (z ^ (z >> 30U)) * 0xBF58476D1CE4E5B9ULL;
    z = (z ^ (z >> 27U)) * 0x94D049BB133111EBULL;
    z ^= z >> 31U;

    hi += z + kGamma;
    std::uint64_t w = hi;
    w = (w ^ (w >> 31U)) * 0xD6E8FEB86659FD93ULL;
    w = (w ^ (w >> 29U)) * 0x94D049BB133111EBULL;
    w ^= w >> 32U;

    lo ^= w;
    hi ^= z;
}

} // namespace

void PrefixDigestBuilder::reset() noexcept {
    lo_ = kSeedLo;
    hi_ = kSeedHi;
    tokens_ = 0;
    image_cursor_ = 0;
}

void PrefixDigestBuilder::mix_token(std::int32_t id) noexcept {
    // 域分隔符：避免 token 与图片的字节串撞车
    mix128(lo_, hi_, 0x01ULL);
    mix128(lo_, hi_, static_cast<std::uint64_t>(static_cast<std::uint32_t>(id)));
}

void PrefixDigestBuilder::mix_image(std::int64_t start, std::uint64_t hash) noexcept {
    mix128(lo_, hi_, 0x02ULL);
    mix128(lo_, hi_, static_cast<std::uint64_t>(start));
    mix128(lo_, hi_, hash);
}

void PrefixDigestBuilder::advance(std::int32_t id, std::span<const std::uint64_t> image_hashes,
                                  std::span<const std::int64_t> image_starts) {
    mix_token(id);
    ++tokens_;
    const std::size_t n = std::min(image_hashes.size(), image_starts.size());
    while (image_cursor_ < n && image_starts[image_cursor_] < tokens_) {
        mix_image(image_starts[image_cursor_], image_hashes[image_cursor_]);
        ++image_cursor_;
    }
}

PrefixDigest prefix_digest(std::span<const std::int32_t> ids,
                           std::span<const std::uint64_t> image_hashes,
                           std::span<const std::int64_t> image_starts,
                           std::int64_t frontier) {
    PrefixDigestBuilder b;
    const std::int64_t n = std::min<std::int64_t>(frontier, static_cast<std::int64_t>(ids.size()));
    for (std::int64_t i = 0; i < n; ++i) b.advance(ids[static_cast<std::size_t>(i)], image_hashes, image_starts);
    return b.value();
}

// ------------------------------------------------------------------ profile

std::uint32_t DiskParkProfile::tag() const noexcept {
    const std::string s = describe();
    std::uint32_t h = 2166136261U;                     // FNV-1a 32
    for (unsigned char c : s) {
        h ^= static_cast<std::uint32_t>(c);
        h *= 16777619U;
    }
    return h;
}

std::string DiskParkProfile::describe() const {
    // ★ 顺序固定，字段用 '|' 分隔，值里不允许出现 '|'（由调用方保证）。
    std::string s;
    s.reserve(96);
    s += "v2|";        // v2 = 带引擎版本的字段表（§7.2）
    s += model.empty() ? "?" : model;
    s += '|';
    s += kv_format.empty() ? "?" : kv_format;
    s += '|';
    s += std::to_string(max_context);
    s += '|';
    s += rope.empty() ? "none" : rope;
    s += '|';
    s += engine.empty() ? "unversioned" : engine;   // §7.2：版本进标签
    return s;
}

} // namespace strata::core
