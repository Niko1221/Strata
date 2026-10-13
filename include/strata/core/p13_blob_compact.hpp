// ★★★ P13 停放层 GC（2026-10-10）：会话目录按【7 天】清扫 + 检查点 blob 压缩。
//
// 为什么需要：
//   · 会话目录（layerNN/kv.bin、pooled.bin、tokens.bin、ckpt_lengths.bin、mtp_kv.bin）
//     是每会话一份，删目录就干净了 —— 但没人删就会一直涨。
//   · `checkpoints.bin` 是【只追加 + 内容寻址 + 没有独立索引文件】的流：删掉会话目录
//     只会在里面留下孤儿记录（一份 118 MB）。压缩 = "把还活着的记录抄进新文件、改名"。
//
// 活键怎么来（不需要任何额外元数据）：
//   留下来的每个会话目录里有 `tokens.bin` + `ckpt_lengths.bin`，于是对每个长度 L
//   算一次 `prefix_digest(tokens[0..L))` —— 这就是它的键（与写侧同一套算法）。
//
// ★ 安全策略：【先验证、再提交】。
//   每个存活会话的【尾部】那条键必须在旧 blob 里找得到；找不到就整体放弃压缩
//   （说明我们算不出这个会话的键 —— 典型是带图片的会话：它的键含图片哈希，而
//   tokens.bin 里没有图片）。宁可 blob 继续长大，也绝不丢有用的记录。
//   dry_run = true 时只统计、不落盘。
#ifndef STRATA_CORE_P13_BLOB_COMPACT_HPP
#define STRATA_CORE_P13_BLOB_COMPACT_HPP

#include "strata/core/disk_park_blob.hpp"
#include "strata/core/disk_park_digest.hpp"
#include "strata/core/p13_session_id.hpp"

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <filesystem>
#include <string>
#include <sys/stat.h>
#include <unistd.h>
#include <vector>

namespace strata::core {

struct P13CompactStats {
    std::size_t session_dirs = 0;   // 扫到的会话目录总数
    std::size_t live_sessions = 0;  // 还在保留期内的
    std::size_t old_sessions = 0;   // 超过保留期的（会被删/已经该删）
    std::size_t keys_live = 0;      // 存活会话声明的键数
    std::size_t keys_found = 0;     // 其中在旧 blob 里真的找到的
    std::size_t records = 0;        // 抄过去的记录数
    std::size_t missing_tails = 0;  // 尾部键不在 blob 里的会话数（闸门跳过的那种，不阻塞压缩）
    std::uint64_t before = 0;       // 旧 blob 字节
    std::uint64_t after = 0;        // 新 blob 字节（dry-run 时 = 存活记录的净载荷）
    std::uint64_t payload = 0;      // 存活记录的净载荷（不含 blob 自己的头/索引）
    bool dry_run = true;
    bool wrote = false;
    std::uint32_t blob_tag = 0;     // 这次压缩认定的标签（0 = 文件没有版本门）
};

// 扫出（root/tag-*/*/）下的会话目录。root 为空 = 关闭。
// ★ tag != 0 ⇒ 只看那一个标签目录：blob 已经按标签分家，跨标签混键会把别的数值路径
//   写下的记录抄进这一份（§7.2）。
inline std::vector<std::string> p13_session_dirs(const std::string& root, std::uint32_t profile_tag = 0) {
    std::vector<std::string> out;
    std::error_code ec;
    if (root.empty() || !std::filesystem::is_directory(root, ec)) return out;
    for (const auto& tag : std::filesystem::directory_iterator(root, ec)) {
        if (!tag.is_directory()) continue;
        const std::string tname = tag.path().filename().string();
        if (tname.rfind("tag-", 0) != 0) continue;
        if (profile_tag != 0) {          // ★ 循环变量叫 tag（目录条目），别和参数混了
            unsigned long v = 0xFFFFFFFFUL;
            try { v = std::stoul(tname.substr(4), nullptr, 16); } catch (...) { v = 0xFFFFFFFFUL; }
            if (tname.size() != 12 || v != (unsigned long) profile_tag) continue;
        }
        for (const auto& sess : std::filesystem::directory_iterator(tag.path(), ec)) {
            if (sess.is_directory()) out.push_back(sess.path().string());
        }
    }
    std::sort(out.begin(), out.end());
    return out;
}

// 会话目录是否"老到该清"（mtime 早于 now - days*24h）。days <= 0 = 永不过期。
inline bool p13_session_expired(const std::string& dir, int days) {
    if (days <= 0) return false;
    std::error_code ec;
    const auto mt = std::filesystem::last_write_time(dir, ec);
    if (ec) return false;
    const auto age = std::filesystem::file_time_type::clock::now() - mt;
    return age > std::chrono::hours(24) * days;
}

// 一个会话声明的全部键（尾部那条单独标出来 —— 它是必须能验证的那条）
struct P13SessionKeys {
    std::vector<DiskParkBlobKey> keys;   // 全部（去重）
    DiskParkBlobKey tail{};              // 最长的那条（长度最大的记录）
    std::int64_t tail_len = 0;
    bool has_tail = false;
};

inline bool p13_session_keys(const std::string& dir, P13SessionKeys& out, std::string& error) {
    std::vector<std::int32_t> tokens;
    std::vector<std::int64_t> lens;
    if (!p13_session_tokens_read(dir + "/tokens.bin", tokens, error)) return false;
    if (!p13_ckpt_lengths_read(dir + "/ckpt_lengths.bin", lens, error)) return false;
    std::sort(lens.begin(), lens.end());
    lens.erase(std::unique(lens.begin(), lens.end()), lens.end());
    for (const std::int64_t L : lens) {
        if (L <= 0 || L > (std::int64_t) tokens.size()) continue;   // 长度越界 ⇒ 不算
        const PrefixDigest d = prefix_digest(tokens, {}, {}, L);
        const DiskParkBlobKey k{d.lo, d.hi};
        if (std::find_if(out.keys.begin(), out.keys.end(), [&](const DiskParkBlobKey& x) {
                return x.lo == k.lo && x.hi == k.hi; }) == out.keys.end())
            out.keys.push_back(k);
        if (!out.has_tail || L > (std::int64_t) out.tail_len) {
            out.tail = k;
            out.tail_len = L;
            out.has_tail = true;
        }
    }
    return true;
}

// 压缩。root = 停放层根目录（park-stream），blob_path = checkpoints.bin 的完整路径。
inline bool p13_compact_blob(const std::string& root, const std::string& blob_path, int days,
                             bool dry_run, P13CompactStats& out, std::string& error,
                             std::uint32_t tag = 0) {
    out = P13CompactStats{};
    out.dry_run = dry_run;
    if (tag == 0) tag = DiskParkBlobStream::file_tag(blob_path);   // 标签从文件头自己认
    out.blob_tag = tag;
    const std::vector<std::string> dirs = p13_session_dirs(root, tag);
    out.session_dirs = dirs.size();
    if (blob_path.empty()) { error = "p13 compact: empty blob path"; return false; }

    // ① 收集存活会话的键
    std::vector<DiskParkBlobKey> live;
    for (const std::string& dir : dirs) {
        if (p13_session_expired(dir, days)) { ++out.old_sessions; continue; }
        std::error_code ec;
        // 没有长度清单 = 这个会话没声明任何记录（可能只写了一半）⇒ 跳过，不阻塞压缩
        if (!std::filesystem::exists(dir + "/ckpt_lengths.bin", ec)) { ++out.live_sessions; continue; }
        if (!std::filesystem::exists(dir + "/tokens.bin", ec)) {
            error = "p13 compact: 无法验证会话 " + dir + "（有长度清单但没有 tokens.bin）⇒ 放弃压缩";
            return false;
        }
        P13SessionKeys sk;
        std::string kerr;
        if (!p13_session_keys(dir, sk, kerr)) {
            error = "p13 compact: 读会话 " + dir + " 失败 (" + kerr + ") ⇒ 放弃压缩";
            return false;
        }
        ++out.live_sessions;
        if (!sk.has_tail) continue;
        live.insert(live.end(), sk.keys.begin(), sk.keys.end());
        // 尾部键最后单独查（必须在旧 blob 里存在）
        live.push_back(sk.tail);
    }
    std::sort(live.begin(), live.end(), [](const DiskParkBlobKey& a, const DiskParkBlobKey& b) {
        return a.lo != b.lo ? a.lo < b.lo : a.hi < b.hi;
    });
    live.erase(std::unique(live.begin(), live.end(), [](const DiskParkBlobKey& a, const DiskParkBlobKey& b) {
                   return a.lo == b.lo && a.hi == b.hi;
               }),
               live.end());
    out.keys_live = live.size();

    // ② 打开旧 blob（不存在 = 没什么可压）
    DiskParkBlobStream::Options oo;
    oo.path = blob_path;
    oo.tag = tag;                                   // §7.2：标签不符 ⇒ 不读
    DiskParkBlobStream old_blob(oo);
    if (!old_blob.enabled()) {
        out.before = 0;
        return true;                      // 没有 blob：当作已经干净
    }
    out.before = old_blob.bytes_on_disk();

    // ③ 逐个键查旧 blob；尾部键必须存在
    std::vector<std::pair<DiskParkBlobKey, DiskParkBlobRef>> found;
    for (const DiskParkBlobKey& k : live) {
        const DiskParkBlobRef r = old_blob.find(k);
        if (r.present()) found.push_back({k, r});
    }
    out.keys_found = found.size();
    for (const std::string& dir : dirs) {
        if (p13_session_expired(dir, days)) continue;
        std::error_code ec;
        if (!std::filesystem::exists(dir + "/ckpt_lengths.bin", ec)) continue;
        P13SessionKeys sk;
        std::string kerr;
        if (!p13_session_keys(dir, sk, kerr) || !sk.has_tail) continue;
        const bool tail_ok = std::any_of(found.begin(), found.end(),
                                         [&](const std::pair<DiskParkBlobKey, DiskParkBlobRef>& f) {
                                             return f.first.lo == sk.tail.lo && f.first.hi == sk.tail.hi;
                                         });
        if (!tail_ok) {
            // ★ 2026-10-11：不再中止 —— 写侧的检查点【闸门】（P16：推进不足 8192 token、
            //   或会话短于 24000 token 就不落盘）会跳过一部分检查点，而 ckpt_lengths.bin
            //   记的是"本轮有哪些长度" ⇒ 最长那条可能压根没进过 blob。
            //   尾部不在 blob 里 = 这条会话没有需要保留的记录 ⇒ 跳过它、继续压。
            //   （原来这里 return false：任何一次闸门跳过都会让整个 blob 压不动。）
            ++out.missing_tails;
            continue;
        }
    }

    out.records = found.size();
    for (const auto& f : found) out.payload += f.second.bytes;
    out.after = out.payload;              // dry-run 的估算；真写完会被覆盖成实际大小
    if (dry_run) return true;

    // ④ 抄进新文件、改名（先写全再 rename ⇒ 崩了也不会留下半个 blob）
    //   ★★ 记下旧文件的属主/权限：压缩常由 root 跑（systemd timer），而【引擎】是以
    //      另一个用户跑的、需要 O_RDWR。不还回去就会把 blob 变成 root:root ⇒
    //      引擎打不开 ⇒ 取回拿不到运行状态（2026-10-10 实测事故，正因如此）。
    struct stat st_old {};
    const bool have_stat = (::stat(blob_path.c_str(), &st_old) == 0);
    const std::string tmp = blob_path + ".compact";
    std::remove(tmp.c_str());
    std::uint64_t real_after = 0;
    {
        DiskParkBlobStream::Options no;
        no.path = tmp;
        no.tag = tag;                               // 抄出来的新文件带上同一个标签
        DiskParkBlobStream new_blob(no);
        if (!new_blob.enabled()) { error = "p13 compact: 无法创建 " + tmp; return false; }
        for (const auto& f : found) {
            std::vector<std::uint8_t> buf((std::size_t) f.second.bytes);
            if (!old_blob.get(f.second, [&](std::uint64_t off, const std::uint8_t* src, std::size_t n) {
                    if (off + n > buf.size()) return false;
                    std::memcpy(buf.data() + (std::size_t) off, src, n);
                    return true;
                })) {
                error = "p13 compact: 读记录失败";
                return false;
            }
            const DiskParkBlobRef nr = new_blob.put(f.first, f.second.bytes,
                [&](std::uint64_t off, std::uint8_t* dst, std::size_t n) {
                    if (off + n > buf.size()) return false;
                    std::memcpy(dst, buf.data() + (std::size_t) off, n);
                    return true;
                });
            if (!nr.present()) { error = "p13 compact: 写记录失败"; return false; }
        }
        real_after = new_blob.bytes_on_disk();
    }   // ← 析构即关文件：必须先关掉再 rename
    if (have_stat) {
        if (::chmod(tmp.c_str(), st_old.st_mode & 07777) != 0)
            std::fprintf(stderr, "p13 compact: chmod 警告（blob 权限可能不对）\n");
        if (::chown(tmp.c_str(), st_old.st_uid, st_old.st_gid) != 0)
            std::fprintf(stderr, "p13 compact: chown 警告（blob 属主可能不对 -> 引擎可能打不开）\n");
    }
    std::error_code ec;
    std::filesystem::rename(tmp, blob_path, ec);
    if (ec) { error = "p13 compact: rename 失败: " + ec.message(); return false; }
    out.after = real_after;
    out.wrote = true;
    return true;
}

} // namespace strata::core

#endif
