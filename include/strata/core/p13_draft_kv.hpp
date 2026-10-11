#pragma once
// ★★★ P13：草稿（MTP）K/V 的落盘格式 —— 让"从逐层追加日志取回"这条路也能恢复草稿层。
//
// 为什么必须做：
//   逐层追加日志（P13）只存了【主模型每一层】的 K/V。草稿层的 K/V（`mtp.kv_state()`，
//   kv_mode==2 的 ring，host 池是它的权威副本）本来是 `conversation_snapshot_restore()`
//   顺带恢复的，而日志取回路径【跳过】了那个函数（否则快照会覆盖刚恢复的池）。
//   实测（2026-10-10 18:43，generate.cpp:10395 那块的第一行是 `if (incoming && ...)`，
//   而日志取回路径的入口条件就是 `!mem_hit` ⇒ `incoming` 恒为 nullptr ⇒ 那块是死代码）：
//   日志里【从来没有】出现过 "草稿 K/V 已恢复"，而 decode 只有 33-35 tok/s、
//   草稿接受率 1/8（冷读是 4/9、57 tok/s）。
//   ⇒ 草稿层从空/残留状态起步 ⇒ 草稿全被 verify 拒掉。
//
// 这里【不重新发明】读写：镜像的采集与写回都用上游已经验证过的那一对
//   `conversation_kv_save()` / `conversation_kv_restore()`（快照停放路径用的就是它们，
//   生产里"内存命中 → drafts 4/9"就是这条路的功劳）。
// 本文件只负责把 `ConversationKv`（纯 host 的 5 个缓冲区）写成一个文件、再原样读回来。
//
// ★★ 2026-10-10：改成**只落 ring 那一段**（写放大从 96% 降到 ~25%）。
//   草稿的恢复只用得上最后 `n_slots` 页（引擎自己 `kv_ring_restore` 只重填那段，
//   而且 prefill 本来就跳过 window 再也够不到的 cell）⇒ 整份 135.6 MB 里
//   只有 ~34.7 MB 是有人读的。文件里因此多了 `first_cell`：本文件只保存
//   cell [first_cell, cells) 这一段（每页 = page_size 个 cell 连续）。
//
// 文件布局（小端；本引擎只跑 x86_64）：
//   [magic 8B "P13MTP02"]
//   [upto u64]          采集时的 token 数
//   [first_cell u64]    本文件从这个 cell 开始（page 对齐）
//   [format i32][pad i32]
//   [cells u64][heads u64][head_dim u64][page_size u64][pooled_rows u64][idx_dim u64]
//   [sizes[5] u64]      k / v / k_scale / v_scale / pooled —— 每段是**切片**的长度
//   [k][v][k_scale][v_scale][pooled]   五个切片各自连续
//
// 失败语义：magic 不对 / 头读不全 / 尺寸不合理 ⇒ 返回 false 并把 error 写好。
// 调用方直接退化成"这次没有草稿 K/V"（与不做这个功能时完全一样），
// 【绝不】产生半个镜像 —— 半个镜像比没有更糟。
//
// 写是"临时文件 + rename"：读侧永远看不到半个文件。

#include "strata/core/conversation_cache.hpp"

#include <array>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <string>

namespace strata::core {

namespace p13_draft_detail {

inline constexpr std::size_t kHeaderBytes = 120;   // 比 01 版多 8 字节 first_cell
inline constexpr char kMagic[8] = {'P', '1', '3', 'M', 'T', 'P', '0', '2'};

inline void put_u64(std::uint8_t* p, std::uint64_t v) { std::memcpy(p, &v, 8); }
inline std::uint64_t get_u64(const std::uint8_t* p) {
    std::uint64_t v = 0;
    std::memcpy(&v, p, 8);
    return v;
}

// 顺序 = ConversationKv 的 k, v, k_scale, v_scale, pooled（与 conversation_snapshot.cpp 的 pools() 一致）
inline ConversationBuffer* part_at(ConversationKv& im, int i) {
    switch (i) {
        case 0: return &im.k;
        case 1: return &im.v;
        case 2: return &im.k_scale;
        case 3: return &im.v_scale;
        default: return &im.pooled;
    }
}
inline const ConversationBuffer* part_at(const ConversationKv& im, int i) {
    return part_at(const_cast<ConversationKv&>(im), i);
}

} // namespace p13_draft_detail

// 把一份草稿 K/V 镜像写到 `path`。失败返回 false（error 已写好），不留半个文件。
inline bool p13_draft_kv_write(const std::string& path, const ConversationKv& image,
                               std::int64_t upto, std::int64_t first_cell,
                               const std::array<std::uint64_t, 5>& slice_off,
                               const std::array<std::uint64_t, 5>& slice_len,
                               std::string& error) {
    using namespace p13_draft_detail;
    if (upto < 0) { error = "p13 draft kv: negative extent"; return false; }

    std::uint64_t sizes[5] = {};
    std::uint64_t total = kHeaderBytes;
    for (int i = 0; i < 5; ++i) {
        // ★ sizes[] 记的是【切片】长度；同时自检"切片必须落在缓冲区里"
        // ★ 只对【非空切片】做越界检查：q4 格式下 k_scale/v_scale 是空缓冲区，
        //   而 first_cell 是页偏移（>0）—— 对空 part 用 off+0 去比 0 会误判越界
        //   （实测被探针 m8 抓出来）。
        if (slice_len[i] != 0 &&
            slice_off[i] + slice_len[i] > (std::uint64_t) part_at(image, i)->size()) {
            error = "p13 draft kv: slice out of range";
            return false;
        }
        sizes[i] = slice_len[i];
        total += sizes[i];
    }
    if (total > ((std::uint64_t) 4 << 30)) { error = "p13 draft kv: image too large"; return false; }

    const std::string tmp = path + ".tmp";
    std::FILE* f = std::fopen(tmp.c_str(), "wb");
    if (f == nullptr) { error = "p13 draft kv: cannot create " + tmp; return false; }

    std::uint8_t h[kHeaderBytes];
    std::memset(h, 0, sizeof h);
    std::memcpy(h, kMagic, 8);
    put_u64(h + 8, (std::uint64_t) upto);
    put_u64(h + 16, (std::uint64_t) first_cell);
    const std::int32_t fmt = (std::int32_t) image.format;
    std::memcpy(h + 24, &fmt, 4);
    put_u64(h + 32, (std::uint64_t) image.cells);
    put_u64(h + 40, (std::uint64_t) image.heads);
    put_u64(h + 48, (std::uint64_t) image.head_dim);
    put_u64(h + 56, (std::uint64_t) image.page_size);
    put_u64(h + 64, (std::uint64_t) image.pooled_rows);
    put_u64(h + 72, (std::uint64_t) image.idx_dim);
    for (int i = 0; i < 5; ++i) put_u64(h + 80 + 8 * i, sizes[i]);

    bool ok = std::fwrite(h, 1, sizeof h, f) == sizeof h;
    for (int i = 0; ok && i < 5; ++i) {
        if (sizes[i] == 0) continue;
        const ConversationBuffer& b = *part_at(image, i);
        ok = b.visit((std::size_t) slice_off[i], (std::size_t) sizes[i],
                     [f](const std::uint8_t* p, std::size_t n, std::size_t) {
                         return std::fwrite(p, 1, n, f) == n;
                     });
    }
    if (ok && std::fflush(f) != 0) ok = false;
    if (std::fclose(f) != 0) ok = false;   // the close itself may report the write error
    if (!ok) {
        std::remove(tmp.c_str());
        error = "p13 draft kv: write failed";
        return false;
    }
    if (std::rename(tmp.c_str(), path.c_str()) != 0) {
        std::remove(tmp.c_str());
        error = "p13 draft kv: rename failed";
        return false;
    }
    return true;
}

// 读回一份草稿 K/V 镜像。失败返回 false（image / upto 不被改动），调用方按"没有"处理。
inline bool p13_draft_kv_read(const std::string& path, ConversationKv& image,
                              std::int64_t& upto, std::int64_t& first_cell,
                              std::array<std::uint64_t, 5>& slice_len,
                              std::string& error) {
    using namespace p13_draft_detail;
    std::FILE* f = std::fopen(path.c_str(), "rb");
    if (f == nullptr) { error = "p13 draft kv: absent"; return false; }

    std::uint8_t h[kHeaderBytes];
    bool ok = std::fread(h, 1, sizeof h, f) == sizeof h && std::memcmp(h, kMagic, 8) == 0;
    ConversationKv im;
    std::int64_t got_upto = 0, got_first = 0;
    std::uint64_t sizes[5] = {};
    if (ok) {
        got_upto = (std::int64_t) get_u64(h + 8);
        got_first = (std::int64_t) get_u64(h + 16);
        std::int32_t fmt = 0;
        std::memcpy(&fmt, h + 24, 4);
        im.format = fmt;
        im.cells = (std::int64_t) get_u64(h + 32);
        im.heads = (std::int64_t) get_u64(h + 40);
        im.head_dim = (std::int64_t) get_u64(h + 48);
        im.page_size = (std::int64_t) get_u64(h + 56);
        im.pooled_rows = (std::int64_t) get_u64(h + 64);
        im.idx_dim = (std::int64_t) get_u64(h + 72);
        std::uint64_t total = 0;
        for (int i = 0; i < 5; ++i) {
            sizes[i] = get_u64(h + 80 + 8 * i);
            if (sizes[i] > ((std::uint64_t) 1 << 32) || total > ((std::uint64_t) 4 << 30) - sizes[i]) ok = false;
            total += sizes[i];
        }
        if (ok && (got_upto < 0 || got_first < 0 || im.cells <= 0 || im.heads <= 0 ||
                   im.head_dim <= 0 || im.page_size <= 0 || got_first % im.page_size != 0 ||
                   got_first >= im.cells))
            ok = false;
    }

    std::array<std::uint64_t, 5> slice_len_local{};
    const std::uint64_t base = (std::uint64_t) kHeaderBytes;
    for (int i = 0; ok && i < 5; ++i) {
        slice_len_local[i] = sizes[i];
        if (sizes[i] == 0) continue;
        ConversationBuffer& b = *part_at(im, i);
        b.resize((std::size_t) sizes[i]);
        const std::uint64_t at = base + [&] {
            std::uint64_t o = 0;
            for (int k = 0; k < i; ++k) o += sizes[k];
            return o;
        }();
        ok = b.visit(0, b.size(), [&](std::uint8_t* p, std::size_t n, std::size_t off) {
            if (std::fseek(f, (long) (at + off), SEEK_SET) != 0) return false;
            return std::fread(p, 1, n, f) == n;
        });
    }
    std::fclose(f);
    if (!ok) { error = "p13 draft kv: corrupt image"; return false; }

    image = std::move(im);
    upto = got_upto;
    first_cell = got_first;
    slice_len = slice_len_local;
    return true;
}

} // namespace strata::core
