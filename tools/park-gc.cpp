// park-gc —— 停放层 GC：删过期会话目录 + 按大小封顶 + 压缩检查点 blob
//
// 为什么需要这个工具：
//   会话目录（每会话一份 layerNN/kv.bin + tokens/pooled/ckpt_lengths/mtp_kv）只增不减。
//   实测 0.63 GB/h ≈ 15 GB/天；不清理一晚就能攒到 25 GB / 437 个目录（其中 360 个是空壳）。
//   仓库里【本来就有】GC 模块（include/strata/core/p13_blob_compact.hpp：p13_session_dirs /
//   p13_session_expired / p13_session_keys / p13_compact_blob），但 grep 全项目没有任何调用者
//   —— 本工具就是那个缺失的调用者。
//
// 三个动作：
//   ① 删掉 mtime 早于 days*24h 的会话目录（--days 控制；days <= 0 = 永不过期）
//      这是真正回收空间的那一步（p13_compact_blob 只在 stats 里【数】old_sessions，它自己并不删目录）
//   ② 【新增】大小闸门：会话目录 + checkpoints.bin 合计超过 --max-gib 时，
//      按 mtime 从最旧开始继续删，直到降到上限以下（--max-gib 0 = 关，默认关）
//   ③ 压缩 checkpoints.bin：只保留【存活会话还引用着的 + 同一 profile 标签的】记录
//      （p13_compact_blob，写 tmp → rename 覆盖）
//
//   为什么需要 ②：流式停放层（--disk-park-stream-dir）在引擎里【没有任何字节预算】
//   —— DiskParkLogs::Config 只有 capacity_cells（稀疏表观容量）/ max_open / max_tracked，
//   --disk-park-gib 只喂给快照池层（--disk-park-path，见 generate.cpp:8192-8193 →
//   disk_park_spool.cpp:25 → disk_park_store.cpp:433 的淘汰）。所以流式层的硬上限只能在这里做。
//   顺序说明：闸门在 ③ 之前跑，用的是【压缩前】的 blob 大小 —— 偏保守（可能多删一点），
//   好处是被删目录的记录马上会被 ③ 丢掉，不必等下一轮。
//   ★ 保护下限 --keep-newer-hours H（默认 6）：比 H 小时还新的目录【绝不删】，宁可超上限。
//     没有下限时，上限只要低于一天的写入量，闸门会一路啃到只剩当天、每天啃一遍新数据；
//     有下限 ⇒ 上限不够变成看得见的数字（"还超 X"），而不是一路删到没有。
//   ★ 计量口径：闸门按【实际分配的块】st_blocks×512 算，不是 file_size()（目录与 blob 同一个口径）。
//     停放层的 kv.bin 是稀疏文件（表观 2566 GiB / 实际 83 GB，约 3%），
//     用 file_size() 会把 64 GiB 的上限判成"要删 89% 的目录"（实测踩过）。
//
// 用法：
//   park-gc --root ~/Strata-data/park-stream                             # dry-run（默认，只统计）
//   park-gc --root ~/Strata-data/park-stream --apply                     # 真删真压
//   park-gc --root ~/Strata-data/park-stream --days 1 --apply            # 窗口收紧到 1 天
//   park-gc --root ... --days 1 --max-gib 64 --apply                     # 再加 64 GiB 硬上限
//   park-gc --root ... --days 1 --max-gib 64 --keep-newer-hours 12 --apply   # 闸门不碰 12 h 内的
//   park-gc --root ... --apply --no-blob                                 # 只删目录，不碰 blob
//   park-gc --root ... --tag AA00AEC7 --apply                            # 指定 profile 标签
//
// 安全性：
//   ① 删会话目录对运行中的引擎是安全的 —— 它下次要写那个目录时重开/重算，
//      这也是停放层一以贯之的失败语义（"任何失败都表现为当作没有"）。
//   ② 压 blob 会【改名覆盖】checkpoints.bin：引擎仍持有旧 inode 的 fd，
//      这之间它的写入会丢（重启后按新文件重建索引，不会写错）。
//      所以这一步最好在引擎空闲时、或重启之前执行。
//   ③ 大小闸门如果开始删【很新的】目录（< --warn-newer-hours，默认 6 小时），
//      说明上限低于一天的写入量 —— 会进入"刚停放就被删"的抖动，工具会打 WARNING。
//      这时该调大 --max-gib、关掉停放层、或降写入量，而不是让闸门一直啃当天的数据。
//
// 编译（★ 与 install_tag_gate.sh:103 保持一致：4 个 .cpp，都不碰 CUDA）：
//   g++ -std=c++20 -O2 -I ~/Strata/include -o ~/park-gc ~/Strata/tools/park-gc.cpp \
//       ~/Strata/src/core/disk_park_blob.cpp ~/Strata/src/core/disk_park_digest.cpp \
//       ~/Strata/src/core/disk_park_image.cpp

#include "strata/core/p13_blob_compact.hpp"

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <string>
#include <sys/stat.h>
#include <vector>

namespace fs = std::filesystem;

/// ★ 目录占用必须按【实际分配的块】计（st_blocks × 512），不能用 file_size()。
/// 为什么：停放层的 kv.bin 是【稀疏】的 —— 表观容量是 max_context × 每 cell 字节数
/// （实测：4225 个目录表观合计 2566 GiB，实际只占 83 GB，约 3%）。
/// 用 file_size() 当预算会放大 30 倍：实测 64 GiB 的上限会把 2932 个存活目录
/// 判成"2251 GiB、要删 2605 个"，而真实占用只有 89 GB、只需删掉约 25 GB。
/// 一个文件的【实际分配】字节：st_blocks×512；stat 失败才退回 file_size()。
/// 目录和 blob 都走这里 —— 口径不一致会让闸门多删（按 file_size 判稀疏文件会放大 30 倍）。
static std::uintmax_t file_alloc_bytes(const std::string& path) {
    struct stat st {};
    if (::stat(path.c_str(), &st) == 0) return (std::uintmax_t) st.st_blocks * 512;
    std::error_code ec;
    const std::uintmax_t n = (std::uintmax_t) fs::file_size(path, ec);
    return ec ? 0 : n;
}

static std::uintmax_t dir_bytes(const std::string& d) {
    std::uintmax_t n = 0;
    std::error_code ec;
    for (auto it = fs::recursive_directory_iterator(d, ec);
         it != fs::recursive_directory_iterator(); it.increment(ec)) {
        if (ec) break;
        std::error_code e2;
        if (!it->is_regular_file(e2)) continue;
        n += file_alloc_bytes(it->path().string());
    }
    return n;
}

/// 目录的【表观】大小（稀疏文件的名义容量），只用于显示，绝不参与预算判断。
static std::uintmax_t dir_apparent_bytes(const std::string& d) {
    std::uintmax_t n = 0;
    std::error_code ec;
    for (auto it = fs::recursive_directory_iterator(d, ec);
         it != fs::recursive_directory_iterator(); it.increment(ec)) {
        if (ec) break;
        std::error_code e2;
        if (it->is_regular_file(e2)) n += it->file_size(e2);
    }
    return n;
}

static std::string human(std::uintmax_t b) {
    char buf[64];
    if (b >= (std::uintmax_t) 1 << 30)
        std::snprintf(buf, sizeof buf, "%.2f GiB", (double) b / (double) ((std::uintmax_t) 1 << 30));
    else if (b >= (std::uintmax_t) 1 << 20)
        std::snprintf(buf, sizeof buf, "%.1f MiB", (double) b / (double) ((std::uintmax_t) 1 << 20));
    else
        std::snprintf(buf, sizeof buf, "%llu B", (unsigned long long) b);
    return buf;
}

/// 目录年龄（小时）。文件系统时钟与 system_clock 的基准可能不同，
/// 所以用 file_time_type::clock::now() 来算差，而不是 system_clock::now()。
static double age_hours(fs::file_time_type mt) {
    const auto d = fs::file_time_type::clock::now() - mt;
    return std::chrono::duration<double, std::ratio<3600>>(d).count();
}

struct Entry {
    std::string    path;
    std::uintmax_t bytes = 0;       // 实际分配的块
    std::uintmax_t apparent = 0;    // 表观（稀疏容量），仅显示
    fs::file_time_type mtime{};
    double         age_h = 0.0;
    bool           stale_mtime = false;   // ★ mtime 读不到：不当删除首选
};

int main(int argc, char** argv) {
    std::string root, blob, days_s = "7", max_gib_s = "0", warn_newer_s = "6", keep_newer_s = "6";
    bool apply = false, do_blob = true;
    std::uint32_t tag_opt = 0;                        // --tag：不给就从 blob 文件头自己认
    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        auto val = [&]() -> std::string { return i + 1 < argc ? std::string(argv[++i]) : std::string(); };
        if (a == "--root")        root = val();
        else if (a == "--blob")   blob = val();
        else if (a == "--days")   days_s = val();
        else if (a == "--max-gib") max_gib_s = val();
        else if (a == "--warn-newer-hours") warn_newer_s = val();
        else if (a == "--keep-newer-hours") keep_newer_s = val();
        else if (a == "--apply")  apply = true;
        else if (a == "--dry-run") apply = false;
        else if (a == "--no-blob") do_blob = false;
        else if (a == "--tag") { const std::string v = val(); tag_opt = (std::uint32_t) std::strtoul(v.c_str(), nullptr, 16); }
        else if (a == "-h" || a == "--help") {
            std::printf("用法: park-gc --root <park-stream> [--blob <checkpoints.bin>]"
                        " [--days N] [--max-gib N] [--keep-newer-hours H] [--warn-newer-hours H]"
                        " [--apply] [--no-blob] [--tag XXXXXXXX]\n"
                        "      （blob 现在带 profile 标签：不给 --tag 就从文件头认，"
                        "并且只压那一个标签的会话目录）\n"
                        "  --days N              删 mtime 早于 N 天的会话目录（<=0 = 永不过期，默认 7）\n"
                        "  --max-gib N           会话目录 + blob 合计封顶 N GiB，超了从最旧开始删（0 = 关，默认）\n"
                        "  --keep-newer-hours H  闸门【绝不删】比 H 小时还新的目录，宁可超上限（默认 6；0 = 关）\n"
                        "  --warn-newer-hours H  下限关掉后，删到比 H 小时还新的目录时打 WARNING（默认 6）\n");
            return 0;
        } else {
            std::fprintf(stderr, "未知参数: %s（--help 看用法）\n", a.c_str());
            return 2;
        }
    }
    if (root.empty()) { std::fprintf(stderr, "缺少 --root\n"); return 2; }
    const int days = std::atoi(days_s.c_str());
    const long long max_gib = std::atoll(max_gib_s.c_str());
    const double warn_newer_h = std::atof(warn_newer_s.c_str());
    const double keep_newer_h = std::atof(keep_newer_s.c_str());
    if (blob.empty()) blob = root + "/checkpoints.bin";

    std::printf("停放层 GC（%s）\n", apply ? "APPLY —— 会真删真压" : "dry-run —— 只统计");
    std::printf("  root : %s\n", root.c_str());
    std::printf("  blob : %s\n", blob.c_str());
    std::printf("  days : %d%s\n", days, days <= 0 ? "（<=0 = 永不过期）" : " 天");
    std::printf("  上限 : %s\n", max_gib > 0 ? (human((std::uintmax_t) max_gib << 30) + "（会话目录 + blob）").c_str()
                                            : "关（只按 --days）");
    std::printf("  下限 : %s\n", keep_newer_h > 0
                    ? (std::string("比 ") + std::to_string((long long) keep_newer_h) + " h 还新的目录不删").c_str()
                    : "关（闸门可以一直删到最新的那个）");

    // ---- ① 过期会话目录 ----
    const std::vector<std::string> dirs = strata::core::p13_session_dirs(root);
    std::size_t n_live = 0, n_old = 0, n_deleted = 0;
    std::uintmax_t bytes_old = 0, bytes_deleted = 0;
    std::vector<Entry> survivors;                       // 未过期，供 ② 用
    survivors.reserve(dirs.size());

    for (const std::string& d : dirs) {
        std::error_code mec;
        const auto mt = fs::last_write_time(d, mec);
        if (!strata::core::p13_session_expired(d, days)) {
            ++n_live;
            Entry e;
            e.path = d;
            e.bytes = dir_bytes(d);
            e.apparent = dir_apparent_bytes(d);
            e.mtime = mt;
            e.age_h = mec ? -1.0 : age_hours(mt);
            e.stale_mtime = (bool) mec;
            survivors.push_back(std::move(e));
            continue;
        }
        ++n_old;
        const std::uintmax_t b = dir_bytes(d);
        bytes_old += b;
        if (apply) {
            std::error_code ec;
            fs::remove_all(d, ec);
            if (!ec) { ++n_deleted; bytes_deleted += b; }
            else std::fprintf(stderr, "  删除失败 %s: %s\n", d.c_str(), ec.message().c_str());
        }
    }
    std::printf("  会话目录 : %zu 个（存活 %zu / 过期 %zu）\n", dirs.size(), n_live, n_old);
    if (apply)
        std::printf("  已删除   : %zu 个，回收 %s（过期目录共 %s）\n", n_deleted,
                    human(bytes_deleted).c_str(), human(bytes_old).c_str());
    else
        std::printf("  可回收   : %s（--apply 才真删）\n", human(bytes_old).c_str());

    // ---- ② 大小闸门（在 ③ 之前，用压缩前的 blob 大小，偏保守）----
    if (max_gib > 0) {
        const std::uintmax_t cap = (std::uintmax_t) max_gib << 30;
        const std::uintmax_t blob_now = file_alloc_bytes(blob);   // ★ 与目录同一个口径
        std::uintmax_t total = blob_now;
        std::uintmax_t apparent_total = blob_now;
        for (const Entry& e : survivors) { total += e.bytes; apparent_total += e.apparent; }

        std::printf("  闸门     : 上限 %s，当前实际 %s（存活目录 %s + blob %s；表观 %s）\n",
                    human(cap).c_str(), human(total).c_str(),
                    human(total - blob_now).c_str(), human(blob_now).c_str(),
                    human(apparent_total).c_str());
        std::printf("             口径：稀疏文件的【实际分配块】st_blocks×512，不是 file_size\n");

        if (total <= cap) {
            std::printf("             未超上限，闸门不动\n");
        } else {
            std::sort(survivors.begin(), survivors.end(),
                      [](const Entry& a, const Entry& b) {
                          if (a.stale_mtime != b.stale_mtime) return a.stale_mtime < b.stale_mtime;  // 读不到 mtime 的排最后
                          return a.mtime < b.mtime;                                                 // 最旧优先
                      });
            std::size_t g_n = 0, keep_n = 0, unknown_n = 0;
            std::uintmax_t g_bytes = 0, keep_bytes = 0, unknown_bytes = 0;
            double oldest_h = -1.0, newest_h = -1.0;
            for (const Entry& e : survivors) {
                if (total <= cap) break;
                if (e.stale_mtime) { ++unknown_n; unknown_bytes += e.bytes; continue; }   // ★ 不拿 stat 失败当删除首选
                if (keep_newer_h > 0 && e.age_h < keep_newer_h) { ++keep_n; keep_bytes += e.bytes; continue; }
                total -= e.bytes;
                ++g_n;
                g_bytes += e.bytes;
                if (oldest_h < 0) oldest_h = e.age_h;
                newest_h = e.age_h;
                if (apply) {
                    std::error_code ec;
                    fs::remove_all(e.path, ec);
                    if (ec) {
                        std::fprintf(stderr, "  闸门删除失败 %s: %s\n", e.path.c_str(), ec.message().c_str());
                        total += e.bytes;      // 没删掉就把账加回去
                        --g_n;
                        g_bytes -= e.bytes;
                    }
                }
            }
            std::printf("  闸门结果 : %s %zu 个，%s；最高龄 %s h，最新 %s h\n", apply ? "已删" : "将删",
                        g_n, human(g_bytes).c_str(),
                        oldest_h >= 0 ? std::to_string((long long) oldest_h).c_str() : "-",
                        newest_h >= 0 ? std::to_string((long long) newest_h).c_str() : "-");
            std::printf("             删除后合计 %s（上限 %s）\n", human(total).c_str(), human(cap).c_str());
            if (keep_n || unknown_n)
                std::printf("             没动的：%zu 个比 --keep-newer-hours 还新（%s）· %zu 个 mtime 读不到（%s）\n",
                            keep_n, human(keep_bytes).c_str(), unknown_n, human(unknown_bytes).c_str());
            if (total > cap)
                std::printf("  ⚠ 上限还差 %s —— 剩下的都在保护范围里。调大 --max-gib、调小 --keep-newer-hours，"
                            "或关掉停放层；闸门不会为了凑数字去啃刚停放的会话。\n", human(total - cap).c_str());
            else if (newest_h >= 0 && warn_newer_h > 0 && newest_h < warn_newer_h)
                std::printf("  ⚠ WARNING: 闸门不得不删 %.1f 小时前刚写的目录 —— "
                            "上限低于一天的写入量，会变成\"刚停放就被删\"的抖动。\n"
                            "             调大 --max-gib、关掉停放层，或降低写入量。\n", newest_h);
        }
    }

    // ---- ③ 检查点 blob ----
    if (do_blob) {
        strata::core::P13CompactStats st;
        std::string err;
        const std::uint32_t use_tag = tag_opt ? tag_opt
                                              : strata::core::DiskParkBlobStream::file_tag(blob);
        std::printf("  blob 标签: %08X%s\n", use_tag,
                    tag_opt ? "（--tag 指定）" : "（从 blob 文件头读到；0 = 没有版本门的老格式）");
        const bool ok = strata::core::p13_compact_blob(root, blob, days, /*dry_run=*/!apply, st, err, use_tag);
        if (!ok) {
            std::fprintf(stderr, "  ✗ blob 压缩失败: %s\n", err.c_str());
            return 1;
        }
        std::printf("  检查点 blob: 会话 %zu（存活 %zu / 过期 %zu）· 存活键 %zu（旧 blob 里找到 %zu）\n",
                    st.session_dirs, st.live_sessions, st.old_sessions, st.keys_live, st.keys_found);
        std::printf("              %s → %s（%zu 条记录）%s\n", human(st.before).c_str(),
                    human(apply ? st.after : st.payload).c_str(), st.records,
                    apply ? (st.wrote ? "  已写入" : "  未写入") : "  预计（dry-run）");
    } else {
        std::printf("  检查点 blob: 跳过（--no-blob）\n");
    }
    std::printf("%s\n", apply ? "完成。" : "dry-run 结束，未改动任何东西。");
    return 0;
}
