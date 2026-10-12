// park-gc —— 停放层 GC：删过期会话目录 + 压缩检查点 blob
//
// 为什么需要这个工具：
//   会话目录（每会话一份 layerNN/kv.bin + tokens/pooled/ckpt_lengths/mtp_kv）只增不减。
//   实测 0.63 GB/h ≈ 15 GB/天；不清理一晚就能攒到 25 GB / 437 个目录（其中 360 个是空壳）。
//   仓库里【本来就有】GC 模块（include/strata/core/p13_blob_compact.hpp：p13_session_dirs /
//   p13_session_expired / p13_session_keys / p13_compact_blob），但 grep 全项目没有任何调用者
//   —— 本工具就是那个缺失的调用者。
//
// 两个动作，都受 --days 控制（days <= 0 = 永不过期）：
//   ① 删掉 mtime 早于 days*24h 的会话目录   —— 这是真正回收空间的那一步
//      （p13_compact_blob 只在 stats 里【数】old_sessions，它自己并不删目录）
//   ② 压缩 checkpoints.bin：只保留【存活会话】还引用着的记录
//      （p13_compact_blob，写 tmp → rename 覆盖）
//
// 用法：
//   park-gc --root ~/Strata-data/park-stream                  # dry-run（默认，只统计）
//   park-gc --root ~/Strata-data/park-stream --apply          # 真删真压
//   park-gc --root ~/Strata-data/park-stream --days 1 --apply # 窗口收紧到 1 天
//   park-gc --root ... --apply --no-blob                      # 只删目录，不碰 blob
//
// 安全性：
//   ① 删会话目录对运行中的引擎是安全的 —— 它下次要写那个目录时重开/重算，
//      这也是停放层一以贯之的失败语义（"任何失败都表现为当作没有"）。
//   ② 压 blob 会【改名覆盖】checkpoints.bin：引擎仍持有旧 inode 的 fd，
//      这之间它的写入会丢（重启后按新文件重建索引，不会写错）。
//      所以这一步最好在引擎空闲时、或重启之前执行。
//
// 编译（只需 blob 那个 .cpp，都不碰 CUDA）：
//   g++ -std=c++20 -O2 -I ~/Strata/include -o ~/park-gc park-gc.cpp ~/Strata/src/core/disk_park_blob.cpp

#include "strata/core/p13_blob_compact.hpp"

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <string>
#include <vector>

namespace fs = std::filesystem;

static std::uintmax_t dir_bytes(const std::string& d) {
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

int main(int argc, char** argv) {
    std::string root, blob, days_s = "7";
    bool apply = false, do_blob = true;
    std::uint32_t tag_opt = 0;                        // --tag：不给就从 blob 文件头自己认
    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        auto val = [&]() -> std::string { return i + 1 < argc ? std::string(argv[++i]) : std::string(); };
        if (a == "--root")        root = val();
        else if (a == "--blob")   blob = val();
        else if (a == "--days")   days_s = val();
        else if (a == "--apply")  apply = true;
        else if (a == "--dry-run") apply = false;
        else if (a == "--no-blob") do_blob = false;
        else if (a == "--tag") { const std::string v = val(); tag_opt = (std::uint32_t) std::strtoul(v.c_str(), nullptr, 16); }
        else if (a == "-h" || a == "--help") {
            std::printf("用法: park-gc --root <park-stream> [--blob <checkpoints.bin>]"
                        " [--days N] [--apply] [--no-blob] [--tag XXXXXXXX]\n"
                        "      （blob 现在带 profile 标签：不给 --tag 就从文件头认，"
                        "并且只压那一个标签的会话目录）\n");
            return 0;
        } else {
            std::fprintf(stderr, "未知参数: %s（--help 看用法）\n", a.c_str());
            return 2;
        }
    }
    if (root.empty()) { std::fprintf(stderr, "缺少 --root\n"); return 2; }
    const int days = std::atoi(days_s.c_str());
    if (blob.empty()) blob = root + "/checkpoints.bin";

    std::printf("停放层 GC（%s）\n", apply ? "APPLY —— 会真删真压" : "dry-run —— 只统计");
    std::printf("  root : %s\n", root.c_str());
    std::printf("  blob : %s\n", blob.c_str());
    std::printf("  days : %d%s\n", days, days <= 0 ? "（<=0 = 永不过期）" : " 天");

    // ---- ① 过期会话目录 ----
    const std::vector<std::string> dirs = strata::core::p13_session_dirs(root);
    std::size_t n_live = 0, n_old = 0, n_deleted = 0;
    std::uintmax_t bytes_old = 0, bytes_deleted = 0;
    for (const std::string& d : dirs) {
        if (!strata::core::p13_session_expired(d, days)) { ++n_live; continue; }
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

    // ---- ② 检查点 blob ----
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
