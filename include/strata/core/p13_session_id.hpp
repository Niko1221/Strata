#pragma once
// ★★★ P13：会话身份 —— "这个会话目录里的 K/V，到底是不是这一段 token 的？"
//
// 为什么必须有：取回侧原来按 `|cells − prompt 长度| <= 64` **猜**目录
// （P13 HANDOFF §5.4 的绕过办法）。猜错时会把【另一段长度相近的对话】的 K/V
// 装进这个请求 —— 而 `checks` 是按【本请求的前缀摘要】取的、是对的，
// 两者不一致还不会报错 ⇒ **静默的错误输出**。这是把开关打开之前的硬前置。
//
// 做法：
//   * 写侧：把会话自己的 token 序列落成 `<会话目录>/tokens.bin`
//     （每轮覆盖写；45k token 的会话 = 180 KB，可以忽略）；
//   * 读侧：拿它与本请求的 `ids` 求【最长公共前缀 lcp】，
//     只有 `lcp` 覆盖到我们真正要用的那些 cell 才允许取回那个目录。
//
// 为什么用 token 序列而不是摘要：摘要是按轮次记在 `DiskParkLogs::tracked_` 里的
// **内存**结构（`lookup()`/`watermark()` 重启即空），而取回场景恰恰就是重启之后；
// 而 token 序列是自包含的、精确的（没有摘要碰撞），读侧本来就有 `ids`。
//
// 文件布局（小端；本引擎只跑 x86_64）：
//   [magic 8B "P13TOK01"][count u64][int32 × count]
//
// 失败语义：文件缺失 / magic 不对 / 长度不够 ⇒ 返回 0 并写好 error。
// **0 必须被调用方当成"身份未验证"**，也就是"不许取回"，绝不是"随便用"。

#include "strata/core/conversation_cache.hpp"

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

namespace strata::core {

namespace p13_session_detail {
inline constexpr char kTokensMagic[8] = {'P', '1', '3', 'T', 'O', 'K', '0', '1'};
inline constexpr std::uint64_t kMaxTokens = (std::uint64_t) 1 << 32;
} // namespace p13_session_detail

// 落盘会话自己的 token 序列（覆盖写：临时文件 + rename ⇒ 读侧永远看不到半个文件）
inline bool p13_session_tokens_write(const std::string& path, const std::vector<std::int32_t>& tokens,
                                     std::string& error) {
    const std::uint64_t n = (std::uint64_t) tokens.size();
    if (n > p13_session_detail::kMaxTokens) { error = "p13 session id: too many tokens"; return false; }
    const std::string tmp = path + ".tmp";
    std::FILE* f = std::fopen(tmp.c_str(), "wb");
    if (f == nullptr) { error = "p13 session id: cannot create " + tmp; return false; }
    std::uint8_t h[16];
    std::memcpy(h, p13_session_detail::kTokensMagic, 8);
    std::memcpy(h + 8, &n, 8);
    bool ok = std::fwrite(h, 1, sizeof h, f) == sizeof h;
    if (ok && n > 0) ok = std::fwrite(tokens.data(), sizeof(std::int32_t), (std::size_t) n, f) == n;
    if (ok && std::fflush(f) != 0) ok = false;
    if (std::fclose(f) != 0) ok = false;
    if (!ok) {
        std::remove(tmp.c_str());
        error = "p13 session id: write failed";
        return false;
    }
    if (std::rename(tmp.c_str(), path.c_str()) != 0) {
        std::remove(tmp.c_str());
        error = "p13 session id: rename failed";
        return false;
    }
    return true;
}

// 读回会话自己的 token 序列（给压缩/审计工具用）。失败返回 false 并写好 error。
inline bool p13_session_tokens_read(const std::string& path, std::vector<std::int32_t>& tokens,
                                    std::string& error) {
    std::FILE* f = std::fopen(path.c_str(), "rb");
    if (f == nullptr) { error = "p13 session id: absent"; return false; }
    std::uint8_t h[16];
    std::uint64_t n = 0;
    bool ok = std::fread(h, 1, sizeof h, f) == sizeof h &&
              std::memcmp(h, p13_session_detail::kTokensMagic, 8) == 0;
    if (ok) {
        std::memcpy(&n, h + 8, 8);
        ok = n <= p13_session_detail::kMaxTokens;
    }
    tokens.clear();
    if (ok && n > 0) {
        tokens.assign((std::size_t) n, 0);
        ok = std::fread(tokens.data(), sizeof(std::int32_t), (std::size_t) n, f) == n;
    }
    std::fclose(f);
    if (!ok) { error = "p13 session id: corrupt tokens file"; tokens.clear(); return false; }
    return true;
}

// 最长公共前缀：会话自己的 token 序列 vs 本请求的 `ids`
// （`ids` 可以是 vector<int64_t> 或 vector<int32_t>：这里统一按 int64 比）。
// 返回 0 = 没有可用的身份信息（缺失/损坏）或第一个 token 就不一样。
template <class Ids>
inline std::size_t p13_session_tokens_lcp(const std::string& path, const Ids& ids, std::string& error) {
    std::FILE* f = std::fopen(path.c_str(), "rb");
    if (f == nullptr) { error = "p13 session id: absent"; return 0; }
    std::uint8_t h[16];
    std::uint64_t n = 0;
    bool ok = std::fread(h, 1, sizeof h, f) == sizeof h &&
              std::memcmp(h, p13_session_detail::kTokensMagic, 8) == 0;
    if (ok) {
        std::memcpy(&n, h + 8, 8);
        if (n > p13_session_detail::kMaxTokens) ok = false;
    }
    std::size_t lcp = 0;
    if (ok) {
        const std::size_t chunk = 4096;                    // 分块：既不假设文件大小，也不整段进内存
        std::vector<std::int32_t> buf(chunk);
        std::uint64_t at = 0;
        while (at < n && (std::size_t) at < ids.size()) {
            const std::uint64_t want64 =
                std::min<std::uint64_t>({ (std::uint64_t) chunk, n - at, (std::uint64_t) ids.size() - at });
            const std::size_t want = (std::size_t) want64;
            if (std::fread(buf.data(), sizeof(std::int32_t), want, f) != want) { ok = false; break; }
            for (std::size_t i = 0; i < want; ++i) {
                if ((std::int64_t) buf[i] != (std::int64_t) ids[(std::size_t) at + i]) {
                    std::fclose(f);
                    return lcp;                            // 分歧点即答案
                }
                ++lcp;
            }
            at += want;
        }
    }
    std::fclose(f);
    if (!ok) { error = "p13 session id: corrupt"; return 0; }
    return lcp;
}

// ---------------------------------------------------------------------------
// 检查点长度清单（P13 取回侧性能修复，2026-10-10）
//
// 为什么需要：取回侧原来为了找"blob 里有哪些检查点"，对 cut = ids.size(), …, 2
// **每一个**长度都重算一次 `prefix_digest` —— 那是 O(n²)（还要复制前缀）：
//   45k 会话 ≈ 10 s（刚好没踩到引擎 60 s 的"无进展"看门狗），
//   135k 会话 ≈ **88 s**（实测把引擎踩到自杀 ⇒ HTTP 503）。
//
// 修法：写侧每轮把"这一轮写进 blob 的检查点长度"记成一个文件，读侧只按这些长度去试
// ⇒ O(检查点个数 × n)（实测 12 × 1.3 ms ≈ 16 ms）。
//
// 布局（小端）：[magic 8B "P13CKP01"][count u32][int64 × count]
// 缺失/损坏 ⇒ 返回 false，调用方按"没有检查点"处理（⇒ 不取回，退化成冷读）——
// 宁可慢一次，也不要拿错状态。
// ---------------------------------------------------------------------------

namespace p13_session_detail {
inline constexpr char kCkptMagic[8] = {'P', '1', '3', 'C', 'K', 'P', '0', '1'};
} // namespace p13_session_detail

inline bool p13_ckpt_lengths_write(const std::string& path, const std::vector<std::int64_t>& lengths,
                                   std::string& error) {
    const std::string tmp = path + ".tmp";
    std::FILE* f = std::fopen(tmp.c_str(), "wb");
    if (f == nullptr) { error = "p13 ckpt lengths: cannot create " + tmp; return false; }
    std::uint8_t h[12];
    std::memcpy(h, p13_session_detail::kCkptMagic, 8);
    const std::uint32_t n = (std::uint32_t) lengths.size();
    std::memcpy(h + 8, &n, 4);
    bool ok = std::fwrite(h, 1, sizeof h, f) == sizeof h;
    if (ok && n > 0) ok = std::fwrite(lengths.data(), sizeof(std::int64_t), n, f) == n;
    if (ok && std::fflush(f) != 0) ok = false;
    if (std::fclose(f) != 0) ok = false;
    if (!ok) { std::remove(tmp.c_str()); error = "p13 ckpt lengths: write failed"; return false; }
    if (std::rename(tmp.c_str(), path.c_str()) != 0) {
        std::remove(tmp.c_str());
        error = "p13 ckpt lengths: rename failed";
        return false;
    }
    return true;
}

inline bool p13_ckpt_lengths_read(const std::string& path, std::vector<std::int64_t>& lengths,
                                  std::string& error) {
    lengths.clear();
    std::FILE* f = std::fopen(path.c_str(), "rb");
    if (f == nullptr) { error = "p13 ckpt lengths: absent"; return false; }
    std::uint8_t h[12];
    std::uint32_t n = 0;
    bool ok = std::fread(h, 1, sizeof h, f) == sizeof h &&
              std::memcmp(h, p13_session_detail::kCkptMagic, 8) == 0;
    if (ok) {
        std::memcpy(&n, h + 8, 4);
        if (n > 4096) ok = false;                       // 防御：检查点不可能这么多
    }
    if (ok && n > 0) {
        lengths.resize(n);
        ok = std::fread(lengths.data(), sizeof(std::int64_t), n, f) == n;
    }
    std::fclose(f);
    if (!ok) { lengths.clear(); error = "p13 ckpt lengths: corrupt"; return false; }
    return true;
}

} // namespace strata::core
