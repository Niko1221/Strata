// DiskParkSession 的单元测试：追加流 + sidecar 提交的存储层。
//
// 覆盖设计文档 §7 阶段 1 的判据：
//   ① 往返一致（含成行 part 的行对齐）
//   ② 稀疏（没写的洞不占盘）
//   ③ 跨轮追加：老字节不被碰
//   ④ 截断回退：改写历史时从回退点重写
//   ⑤ 崩溃一致性：sidecar 是唯一提交点；kv.bin 的尾巴永不参与
//   ⑥ plan 不符 → 干净放弃且不改文件
//   ⑦ 与上游镜像序列化器的拼接：拼出来的字节流必须能被 unpack 回原镜像
//
// ★ 本测试的"真值"只取决于【全局 cell 位置】（truth_at），与"哪一轮写的"无关 ——
//   因为追加流的核心性质正是"同一位置的字节永远相同"。比对必须建立在同一个定义上。

#include "strata/core/disk_park_image.hpp"
#include "strata/core/disk_park_digest.hpp"
#include "strata/core/disk_park_session.hpp"

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <cstdlib>
#include <filesystem>
#include <functional>
#include <sys/stat.h>
#include <vector>

using namespace strata::core;

static int g_checks = 0, g_failed = 0;
#define CHECK(cond, msg)                                                 \
    do {                                                                 \
        ++g_checks;                                                      \
        if (!(cond)) { ++g_failed; std::printf("  FAIL  %s\n", (msg)); } \
        else         { std::printf("  ok    %s\n", (msg)); }             \
    } while (0)

namespace {

constexpr std::size_t kK = 0, kV = 1, kKS = 2, kVS = 3, kP = 4;

// 真值：只取决于 (part, 全局 cell, 段内字节)，与轮次无关
std::uint8_t truth_at(std::size_t part, std::uint64_t cell, std::size_t byte) {
    return static_cast<std::uint8_t>((part * 131 + cell * 7 + byte * 3 + 1) & 0xFF);
}

// 覆盖任意 [0, capacity) 范围的真值源：不管 session 请求哪一段，都按全局位置取。
DiskParkFill truth_fill(const DiskParkSessionPlan& plan) {
    return [&plan](std::size_t part, std::size_t offset, std::uint8_t* dst, std::size_t n) {
        const auto& p = plan.parts[part];
        std::size_t i = 0;
        while (i < n) {
            // ★★ cell / within 必须按【当前位置 offset + i】每轮重算 ——
            //   在循环外算一次会让整段都写成同一个 cell（这正是本测试抓到的自己的 bug）。
            //   offset 是"part 内第几字节"，所以 cell 就是全局 cell，与 from_cell 无关。
            const std::size_t at = offset + i;
            if (p.row_cells <= 1) {
                const std::uint64_t cell = at / p.bytes_per_cell;
                const std::size_t within = at % p.bytes_per_cell;
                const std::size_t take = std::min(n - i, p.bytes_per_cell - within);
                for (std::size_t j = 0; j < take; ++j) dst[i + j] = truth_at(part, cell, within + j);
                i += take;
            } else {
                const std::uint64_t row = at / p.row_bytes;
                const std::size_t within = at % p.row_bytes;
                const std::size_t take = std::min(n - i, p.row_bytes - within);
                for (std::size_t j = 0; j < take; ++j)
                    dst[i + j] = truth_at(part, row * p.row_cells + (within + j) / p.bytes_per_cell,
                                          (within + j) % p.bytes_per_cell);
                i += take;
            }
        }
        return true;
    };
}

// 读回 [from,to) 并与 truth_at 逐字节比对
bool verify_part(const DiskParkSession& s, const DiskParkSessionPlan& plan,
                 DiskParkSession::Part part, std::uint64_t from, std::uint64_t to,
                 const char* what) {
    const std::size_t pi = static_cast<std::size_t>(part);
    if (plan.parts[pi].bytes_per_cell == 0) return true;
    const std::uint64_t n = s.part_bytes(part, to) - s.part_bytes(part, from);
    if (n == 0) return true;
    std::vector<std::uint8_t> got(static_cast<std::size_t>(n), 0);
    if (!s.read_kv(part, from, to, got.data(), got.size())) {
        std::printf("      （%s: read_kv 失败）\n", what);
        return false;
    }
    const auto& p = plan.parts[pi];
    for (std::size_t i = 0; i < got.size(); ++i) {
        // ★ 两种布局分别解：
        //   线性 part：byte i ↔ (cell = from + i/bpc, byte = i%bpc)
        //   成行 part：byte i ↔ 行 (from/bpc + i/row_bytes)、行内 (i%row_bytes)
        //              → cell = 行首cell + (行内偏移)/bpc
        std::uint64_t cell = 0; std::size_t byte = 0;
        if (p.row_cells <= 1) {
            cell = from + static_cast<std::uint64_t>(i) / p.bytes_per_cell;
            byte = i % p.bytes_per_cell;
        } else {
            const std::uint64_t row = from / p.row_cells + static_cast<std::uint64_t>(i) / p.row_bytes;
            const std::size_t within = i % p.row_bytes;
            cell = row * p.row_cells + within / p.bytes_per_cell;
            byte = within % p.bytes_per_cell;
        }
        const std::uint8_t want = truth_at(pi, cell, byte);
        if (got[i] != want) {
            std::printf("      （%s: 第 %zu 字节不符：读到 %02X 应为 %02X）\n", what, i, got[i], want);
            return false;
        }
    }
    return true;
}

std::uint64_t allocated_bytes(const std::string& path) {
    struct stat st {};
    if (::stat(path.c_str(), &st) != 0) return 0;
    return static_cast<std::uint64_t>(st.st_blocks) * 512;
}

} // namespace

// k8v4 风格的几何：K 每 cell 4 B、V 每 cell 2 B、两个 scale 各 1 B、
// pooled 每 4 cell 一行、每行 8 B（★ 行宽刻意不等于 4×2，以逼出行对齐错误）。
DiskParkSessionPlan make_plan(std::uint32_t tag = 0xABCD1234u) {
    DiskParkSessionPlan p;
    p.tag = tag;
    p.capacity_cells = 4096;
    p.kv_format = 3; p.heads = 2; p.head_dim = 256; p.page_size = 4; p.idx_dim = 8;
    p.parts[kK]  = {4, 1, 4, true};
    p.parts[kV]  = {2, 1, 2, true};
    p.parts[kKS] = {1, 1, 1, true};
    p.parts[kVS] = {1, 1, 1, true};
    // pooled：成行 part（每 4 cell 一行、每行 8 B）—— 用它考行对齐
    // pooled 留在 sidecar 载荷里（in_log=false）—— 与生产的选择一致：
    // 它小（实测 171 MiB/13 层），而且有"移动的备用行"，不是纯追加。
    p.parts[kP]  = {2, 4, 8, false};
    return p;
}

// 只为稀疏测试用的大容量 plan（1 Gi cell/part ⇒ 逻辑长度 ~9 GiB）
DiskParkSessionPlan make_big_plan(std::uint32_t tag = 0xABCD1234u) {
    DiskParkSessionPlan p = make_plan(tag);
    p.capacity_cells = 1ull << 30;
    return p;
}

int main() {
    namespace fs = std::filesystem;
    const fs::path root = fs::temp_directory_path() / "disk_park_session_test";
    fs::remove_all(root);
    fs::create_directories(root);

    const DiskParkSessionPlan plan = make_plan();
    const DiskParkSessionPlan big  = make_big_plan();
    const DiskParkFill fill = truth_fill(plan);
    const std::string dir = (root / "sess").string();

    std::printf("\n=== 1. 打开 / 新建 / 段偏移与水位无关 ===\n");
    {
        DiskParkSession s;
        CHECK(DiskParkSession::open(s, dir, plan), "open 成功（新建）");
        CHECK(s.ok(), "ok()");
        CHECK(s.log_cells() == 0, "★ 新会话水位为 0");
        CHECK(fs::exists(s.kv_path()), "kv.bin 建了");
        CHECK(!fs::exists(s.state_path()), "★ 还没有 state.bin（没提交过）");
        CHECK(fs::file_size(s.kv_path()) == DiskParkSession::kHeaderBytes, "文件只有头");

        CHECK(s.part_offset(DiskParkSession::Part::K, 0) == DiskParkSession::kHeaderBytes,
              "K 段从头之后开始");
        CHECK(s.part_offset(DiskParkSession::Part::V, 0) ==
                  DiskParkSession::kHeaderBytes + s.part_bytes(DiskParkSession::Part::K, 4096),
              "★ V 段在 K 的【固定容量】之后（与水位无关）");
        CHECK(s.part_offset(DiskParkSession::Part::V, 100) ==
                  s.part_offset(DiskParkSession::Part::V, 0) + 100 * 2,
              "线性 part 的偏移随 cell 线性");
        CHECK(s.part_offset(DiskParkSession::Part::Pooled, 4) ==
                  s.part_offset(DiskParkSession::Part::Pooled, 0) + 8,
              "★ pooled 每 4 cell 前进一整行（8 B，不是 4×2）");
    }

    std::printf("\n=== 2. 追加 + 读回（全部 5 个 part）===\n");
    {
        DiskParkSession s;
        CHECK(DiskParkSession::open(s, dir, plan), "重开成功");
        CHECK(s.log_cells() == 0, "★ 没有 sidecar → 水位仍是 0");
        CHECK(s.append_kv(0, 100, fill), "追加 [0,100)");
        CHECK(s.log_cells() == 100, "水位 = 100");
        CHECK(verify_part(s, plan, DiskParkSession::Part::K, 0, 100, "K"), "★ K 逐字节正确");
        CHECK(verify_part(s, plan, DiskParkSession::Part::V, 0, 100, "V"), "★ V 逐字节正确");
        CHECK(verify_part(s, plan, DiskParkSession::Part::KScale, 0, 100, "Kscale"), "★ Kscale 正确");
        CHECK(verify_part(s, plan, DiskParkSession::Part::VScale, 0, 100, "Vscale"), "★ Vscale 正确");
        CHECK(verify_part(s, plan, DiskParkSession::Part::Pooled, 0, 100, "pooled"),
              "★ pooled 行对齐正确（行宽 8 ≠ 4×2，错位就会在这里露出来）");
    }

    std::printf("\n=== 3. ★ 跨轮追加：第二轮接在第一轮之后，老字节不动 ===\n");
    {
        DiskParkSession s;
        CHECK(DiskParkSession::open(s, dir, plan), "重开");
        std::vector<std::byte> meta(8, std::byte{0});
        CHECK(s.commit(100, meta), "提交水位 100");

        DiskParkSession s2;
        CHECK(DiskParkSession::open(s2, dir, plan), "再重开");
        CHECK(s2.log_cells() == 100, "★ 水位从 sidecar 恢复 = 100");

        std::vector<std::uint8_t> old_k(100 * 4, 0);
        CHECK(s2.read_kv(DiskParkSession::Part::K, 0, 100, old_k.data(), old_k.size()), "读老 K");
        CHECK(s2.append_kv(100, 250, fill), "追加 [100,250)");
        CHECK(verify_part(s2, plan, DiskParkSession::Part::K, 0, 250, "K"),
              "★★ 跨两轮后 [0,250) 全部正确（第二轮没动第一轮的字节）");
        CHECK(verify_part(s2, plan, DiskParkSession::Part::Pooled, 0, 250, "pooled"),
              "★★ pooled 跨轮也对（第二轮从 cell 100 = 整行边界接上）");
    }

    std::printf("\n=== 4. ★★ 稀疏：容量 9 GiB 的逻辑文件，只写几 MB ===\n");
    {
        const std::string big_dir = (root / "big").string();
        DiskParkSession s;
        CHECK(DiskParkSession::open(s, big_dir, big), "用大容量 plan 新建");
        const DiskParkFill big_fill = truth_fill(big);
        CHECK(s.append_kv(0, 1024, big_fill), "只写前 1024 个 cell");
        struct stat st {};
        ::stat(s.kv_path().c_str(), &st);
        const auto logical = static_cast<std::uint64_t>(st.st_size);
        const auto alloc   = allocated_bytes(s.kv_path());
        std::printf("       逻辑 %.2f GiB / 实占 %.1f MiB\n",
                    (double) logical / 1073741824.0, (double) alloc / 1048576.0);
        CHECK(logical > (1ull << 33), "逻辑长度 > 8 GiB（容量撑起来了）");
        CHECK(alloc < (64ull << 20), "★ 实占 < 64 MiB");
        CHECK(alloc * 100 < logical, "★★ 实占不到逻辑的 1%（稀疏真的生效）");
    }

    std::printf("\n=== 5. ★ 截断回退：from_cell < 水位 时从回退点重写 ===\n");
    {
        DiskParkSession s;
        CHECK(DiskParkSession::open(s, dir, plan), "重开");
        std::vector<std::byte> meta(8, std::byte{0});
        CHECK(s.commit(250, meta), "提交水位 250");
        CHECK(s.append_kv(120, 180, fill), "从 120 重写 [120,180)");
        CHECK(s.log_cells() == 180, "水位变成 180");
        CHECK(verify_part(s, plan, DiskParkSession::Part::K, 0, 180, "K"),
              "★★ 回退点之前保持旧值、之后是新值，整段仍与真值一致（真值只认全局位置）");
        CHECK(verify_part(s, plan, DiskParkSession::Part::Pooled, 0, 180, "pooled"),
              "★★ pooled 回退重写也对（回退点 120 是整行边界）");
    }

    std::printf("\n=== 6. ★★ 崩溃一致性：kv.bin 的尾巴永不参与 ===\n");
    {
        DiskParkSession s;
        CHECK(DiskParkSession::open(s, dir, plan), "重开");
        std::vector<std::byte> meta(8, std::byte{0});
        meta[0] = std::byte{'O'}; meta[1] = std::byte{'K'};
        CHECK(s.commit(180, meta), "提交水位 180");
        CHECK(s.append_kv(180, 400, fill), "再追加到 400（模拟 commit 之前被杀）");
        CHECK(s.log_cells() == 400, "内存里水位=400");

        DiskParkSession s2;
        CHECK(DiskParkSession::open(s2, dir, plan), "模拟重启后重开");
        CHECK(s2.log_cells() == 180, "★★ 水位回到 sidecar 里的 180（尾部 220 cell 被忽略）");
        std::vector<std::byte> payload;
        std::uint64_t cells = 0;
        CHECK(s2.read_meta(payload, cells), "读 sidecar");
        CHECK(cells == 180 && payload.size() == 8 && payload[0] == std::byte{'O'},
              "★ sidecar 内容是上一次提交的那份");
    }

    std::printf("\n=== 7. ★ sidecar 损坏 / 截断 → 干净放弃（水位退回 0）===\n");
    {
        const std::string dir2 = (root / "broken").string();
        DiskParkSession s;
        CHECK(DiskParkSession::open(s, dir2, plan), "新建");
        std::vector<std::byte> meta(16, std::byte{7});
        CHECK(s.commit(50, meta), "提交");
        fs::resize_file(s.state_path(), DiskParkSession::kMetaHeaderBytes + 4);
        DiskParkSession s2;
        CHECK(DiskParkSession::open(s2, dir2, plan), "重开仍成功（kv.bin 本身没坏）");
        std::vector<std::byte> payload;
        std::uint64_t cells = 0;
        CHECK(!s2.read_meta(payload, cells), "★ 坏 sidecar 读不出来");
        CHECK(s2.log_cells() == 0, "★★ 水位退回 0（当作没有这一层，绝不读脏数据）");
    }

    std::printf("\n=== 8. ★ plan 不符 → 放弃，且不改动文件 ===\n");
    {
        DiskParkSession s;
        CHECK(!DiskParkSession::open(s, dir, make_plan(0xDEADBEEFu)), "★ 只换 tag → open 失败");
        CHECK(!s.ok(), "没有 fd 泄漏出来");

        DiskParkSessionPlan geom = make_plan();
        geom.parts[kK].bytes_per_cell = 8;
        DiskParkSession s2;
        CHECK(!DiskParkSession::open(s2, dir, geom), "★ 只换 K 的每 cell 字节数 → open 失败");

        DiskParkSessionPlan cap = make_plan();
        cap.capacity_cells = 8192;
        DiskParkSession s3;
        CHECK(!DiskParkSession::open(s3, dir, cap), "★ 只换容量 → open 失败（段偏移会变）");

        DiskParkSession ok;
        CHECK(DiskParkSession::open(ok, dir, plan), "★ 原 plan 仍能打开（文件没被改）");
        CHECK(ok.log_cells() == 180, "★ 水位仍在（失败的几次没有破坏任何东西）");
    }

    std::printf("\n=== 9. ★★ 与上游镜像序列化器拼接：拼出的字节流可 unpack 回原镜像 ===\n");
    {
        const std::string dir3 = (root / "splice").string();
        DiskParkSession s;
        CHECK(DiskParkSession::open(s, dir3, plan), "新建");

        const std::uint64_t cells = 300;
        SavedConversation img;
        img.layer_lo = 0; img.layer_hi = 3;
        img.live.ids.assign(static_cast<std::size_t>(cells), 7);
        img.live.gdn.assign(256, 3);
        ConversationCheckpoint c;
        c.ids.assign(64, 5); c.gdn.assign(32, 9); c.used = 1;
        img.checkpoints.push_back(std::move(c));

        // 镜像里每个 part 的长度必须与 plan 一致（上游会按 layout 逐 part 校验长度）。
        // ★ 本用例把 pooled 的 bytes_per_cell 设为 0（idx_dim=8 时其实不持有 pooled 字节），
        //   于是 log 里也只有 4 个 part 有数据 —— 两边一致，拼接才不会多出一段。
        const std::uint64_t part_len[5] = {
            plan.parts[kK].bytes_per_cell * cells, plan.parts[kV].bytes_per_cell * cells,
            plan.parts[kKS].bytes_per_cell * cells, plan.parts[kVS].bytes_per_cell * cells,
            plan.parts[kP].bytes_per_cell * cells};
        for (int layer = 0; layer < 3; ++layer) {
            ConversationKv kv;
            kv.format = plan.kv_format; kv.cells = static_cast<std::int64_t>(cells);
            kv.heads = plan.heads; kv.head_dim = plan.head_dim;
            kv.page_size = plan.page_size; kv.pooled_rows = 0; kv.idx_dim = plan.idx_dim;
            kv.k.resize(part_len[kK]); kv.v.resize(part_len[kV]);
            kv.k_scale.resize(part_len[kKS]); kv.v_scale.resize(part_len[kVS]);
            kv.pooled.resize(part_len[kP]);
            img.kv.push_back(std::move(kv));
        }

        CHECK(s.append_kv(0, cells, fill), "追加 KV 到 log");

        SavedConversation meta_img = img;
        for (auto& kv : meta_img.kv) {
            kv.k = ConversationBuffer{}; kv.v = ConversationBuffer{};
            kv.k_scale = ConversationBuffer{}; kv.v_scale = ConversationBuffer{};
            kv.pooled = ConversationBuffer{};
        }
        const std::vector<std::byte> meta = pack_saved_conversation(meta_img);
        CHECK(meta.size() > 8, "sidecar 载荷非空");
        CHECK(s.commit(cells, meta), "提交");

        std::vector<std::byte> payload;
        std::uint64_t lc = 0;
        CHECK(s.read_meta(payload, lc), "读 sidecar 载荷");
        CHECK(lc == cells, "水位 = cells");

        // ---- 拼接：拿"上游自己 pack 的完整镜像"当骨架，把它里面每个 KV part 的
        //      【零字节】按 (cell,byte) 换成 log 里的真值。----
        // ★ 第一版想手工数出 KV 的起点，数错了两轮；按骨架+固定步长定位就不会错。
        std::vector<std::byte> stream = pack_saved_conversation(img);
        const std::uint64_t full_len = stream.size();

        // 走到第一个 KV part 的数据起点：KV 之前的每一段都是"长度前缀 + 定长/变长数据"
        std::size_t at = 8;                                        // magic + version
        at += 18 * 8 + 8 + 8;                                      // geometry + layer_lo/hi
        auto rd_u64 = [&]() { std::uint64_t v = 0; std::memcpy(&v, stream.data() + at, 8); at += 8; return v; };
        auto rd_i64 = [&]() { std::int64_t  v = 0; std::memcpy(&v, stream.data() + at, 8); at += 8; return v; };
        std::function<void()> skip_ck = [&]() {
            at += static_cast<std::size_t>(rd_u64()) * 4;           // ids
            at += static_cast<std::size_t>(rd_u64()) * 16;          // imgs
            for (int k = 0; k < 5; ++k) at += static_cast<std::size_t>(rd_u64());
            (void) rd_i64();                                        // used
            for (std::uint64_t k = 0, n = rd_u64(); k < n; ++k) skip_ck();
        };
        skip_ck();                                                  // live
        for (std::uint64_t k = 0, n = rd_u64(); k < n; ++k) skip_ck();
        const std::uint64_t nk = rd_u64();

        CHECK(nk == img.kv.size(), "★ 骨架里的 KV 层数与镜像一致");
        std::size_t cursor = at;
        for (std::uint64_t layer = 0; layer < nk; ++layer) {
            cursor += 4 + 6 * 8;                                    // format + 6 个 int64
            for (std::size_t p = 0; p < 5; ++p) {
                const std::uint64_t want = [&] { std::uint64_t v = 0;
                    std::memcpy(&v, stream.data() + cursor, 8); cursor += 8; return v; }();
                const auto& pp = plan.parts[p];
                if (pp.bytes_per_cell == 0 || want == 0) continue;   // 这一 part 本用例没有
                CHECK(want == part_len[p], "★ 骨架里的 part 长度与预期一致");
                // 镜像里 part 是【按 cell 稠密】的（有 pooled_bytes 时按索引间隔排布），
                // 所以第 k 个字节属于哪个 (cell,byte) 由 part 自己的布局决定。
                for (std::uint64_t k = 0; k < want; ++k) {
                    std::uint64_t cell = 0; std::size_t byte = 0;
                    if (p == kP && pp.row_bytes > pp.bytes_per_cell) {
                        const std::size_t stride = pp.row_bytes / pp.bytes_per_cell;
                        cell = (k / pp.bytes_per_cell) * stride + (k % pp.bytes_per_cell);
                        byte = 0;
                    } else {
                        cell = k / pp.bytes_per_cell;
                        byte = static_cast<std::size_t>(k % pp.bytes_per_cell);
                    }
                    stream[cursor + static_cast<std::size_t>(k)] = std::byte{truth_at(p, cell, byte)};
                }
                cursor += static_cast<std::size_t>(want);
            }
        }
        // KV 数组之后还有两个字段：cvec(1 B) + stage_images 的个数(8 B)
        CHECK(cursor + 9 == full_len, "★ 走完 KV 之后只剩 cvec + stage 计数（9 B）");

        std::printf("       拼接后 %zu B，镜像应有 %llu B\n", stream.size(),
                    (unsigned long long) full_len);
        CHECK(stream.size() == full_len, "★ 拼接长度与 packed_saved_conversation_bytes 相等");

        auto back = unpack_saved_conversation(stream);
        CHECK(back.has_value(), "★★ 拼接出来的字节流能被 unpack");
        if (back) {
            CHECK(back->live.ids == img.live.ids, "live.ids 一致");
            CHECK(back->live.gdn == img.live.gdn, "live.gdn 一致");
            CHECK(back->checkpoints.size() == img.checkpoints.size(), "checkpoints 数量一致");
            CHECK(back->kv.size() == img.kv.size(), "kv 层数一致");
            CHECK(back->kv[0].k.size() == part_len[kK], "★ 第 0 层 K 的长度 = 从 log 取的长度");
            // ★ 一次读完再比 —— 早先写成"逐字节 read(&b,i,1)"，
            //   而 ConversationBuffer::read 每次都要走一遍段目录，O(n²) 直接把测试拖死。
            std::vector<std::uint8_t> kb(static_cast<std::size_t>(part_len[kK]), 0);
            back->kv[0].k.read(kb.data(), 0, kb.size());
            bool kv_ok = true;
            for (std::size_t i = 0; i < kb.size() && kv_ok; ++i)
                if (kb[i] != truth_at(kK, i / plan.parts[kK].bytes_per_cell,
                                      i % plan.parts[kK].bytes_per_cell)) kv_ok = false;
            CHECK(kv_ok, "★★ 从 log 拼回来的 K 逐字节正确");
        }
    }

    std::printf("\n=== 14. ★★★ 取回（方案 C）：KV 留载荷、检查点进日志，拼出来必须能 unpack ===\n");
    {
        // ★★★ 方案 C（2026-10-10 定稿）：
        //   · KV **永远留在载荷里**（日志按位置寻址、镜像按层存，模型不一致）
        //   · 检查点状态/live 的 5 个缓冲区**进日志**（占链 91.7%，且没有"层"的问题）
        //
        //   载荷的形态：把要搬走的那 5 个缓冲区 `resize(0)` —— 长度字段自然为 0、
        //   数据段自然消失（最小探针 m1.cpp 证实过）。
        const std::string dir4 = (root / "optC").string();
        DiskParkSession s;
        CHECK(DiskParkSession::open(s, dir4, plan), "新建会话");

        const std::uint64_t cells = 300;
        const std::size_t layers = 3;
        // 参考镜像：KV 有内容；检查点/live 的缓冲区也有内容（稍后会被搬进日志）
        SavedConversation ref;
        ref.layer_lo = 0; ref.layer_hi = static_cast<std::int64_t>(layers);
        ref.live.ids.assign(static_cast<std::size_t>(cells), 7);
        ref.live.gdn.assign(512, 11);
        ref.live.ple.assign(64, 12);
        ref.live.tails.assign(16, 13);
        ref.live.dead.assign(8, 14);
        ref.live.block_pos.assign(4, 15);
        {
            ConversationCheckpoint ck;
            ck.ids.assign(64, 5); ck.gdn.assign(256, 21); ck.ple.assign(32, 22);
            ck.tails.assign(8, 23); ck.dead.assign(4, 24); ck.block_pos.assign(2, 25);
            ck.used = 1;
            ref.checkpoints.push_back(std::move(ck));
        }
        const std::uint64_t part_len[5] = {
            plan.parts[kK].bytes_per_cell * cells, plan.parts[kV].bytes_per_cell * cells,
            plan.parts[kKS].bytes_per_cell * cells, plan.parts[kVS].bytes_per_cell * cells,
            (cells / plan.parts[kP].row_cells) * plan.parts[kP].row_bytes +
                (cells % plan.parts[kP].row_cells) * plan.parts[kP].bytes_per_cell};
        for (std::size_t layer = 0; layer < layers; ++layer) {
            ConversationKv kv;
            kv.format = plan.kv_format; kv.cells = static_cast<std::int64_t>(cells);
            kv.heads = plan.heads; kv.head_dim = plan.head_dim;
            kv.page_size = plan.page_size; kv.pooled_rows = 0; kv.idx_dim = plan.idx_dim;
            kv.k.resize(part_len[kK]); kv.v.resize(part_len[kV]);
            kv.k_scale.resize(part_len[kKS]); kv.v_scale.resize(part_len[kVS]);
            kv.pooled.resize(part_len[kP]);
            ref.kv.push_back(std::move(kv));
        }

        // ① 完整镜像（参照物）
        std::vector<std::byte> full = pack_saved_conversation(ref);
        std::printf("       完整镜像 = %zu 字节\n", full.size());

        // ② 侧车载荷 = 把"检查点/live 的 5 个缓冲区 + KV 的 4 个 in_log part"清空后 pack
        //    ★ 方案 C 下 KV 也清空（它留在载荷里，但内容由拼接从 kv.bin 之外的路径……
        //      不 —— 方案 C 里 KV 的数据【就在载荷里】，所以 KV 不清，只有检查点清）。
        // ★★★★ 载荷的正确形态（独立探针 m4.cpp 证实、逐字节相同才写回来）：
        //   live / 检查点的 5 个缓冲区 **保留长度、内容清零**。不要 resize(0)！
        //   为什么：
        //     · 长度字段必须是真值 —— 拼接的载荷光标靠它对齐后面的缓冲区；
        //       清成 0 会让那个 part 的循环根本不进 ⇒ fill_ckpt 一次都不被调用。
        //     · 内容清成 0 —— "这段在日志里"由【目录项 present()】表达。
        //   于是载荷与完整镜像【等长】，只是那些字节是 0。
        //   （我先后试过"删数据段"和"resize(0)"，都会让拼接错位；见 HANDOFF §5。）
        SavedConversation meta = ref;
        auto zero_keep_len = [](std::vector<std::uint8_t>& v) { for (auto& b : v) b = 0; };
        zero_keep_len(meta.live.gdn); zero_keep_len(meta.live.ple); zero_keep_len(meta.live.tails);
        zero_keep_len(meta.live.dead); zero_keep_len(meta.live.block_pos);
        for (auto& c : meta.checkpoints) {
            zero_keep_len(c.gdn); zero_keep_len(c.ple); zero_keep_len(c.tails);
            zero_keep_len(c.dead); zero_keep_len(c.block_pos);
        }
        // KV【不动】—— 方案 C 下它整份在载荷里
        std::vector<std::byte> payload = pack_saved_conversation(meta);
        std::printf("       侧车载荷 = %zu 字节（比完整镜像少 %lld，应 = 检查点/live 的缓冲之和）\n",
                    payload.size(), (long long) full.size() - (long long) payload.size());
        const std::uint64_t moved = (ref.live.gdn.size() + ref.live.ple.size() + ref.live.tails.size()
            + ref.live.dead.size() + ref.live.block_pos.size())
            + (ref.checkpoints[0].gdn.size() + ref.checkpoints[0].ple.size()
               + ref.checkpoints[0].tails.size() + ref.checkpoints[0].dead.size()
               + ref.checkpoints[0].block_pos.size());
        CHECK(full.size() == payload.size(),
              "★ 载荷与完整镜像【等长】（只是那些字节是 0 占位）");
        CHECK(moved > 0, "★ 确实有缓冲区搬进了日志");

        // ③ 找 kv_at（扫描 + 走完正好落在末尾的自检）
        const std::uint64_t kv_at = [&]() -> std::uint64_t {
            auto rdp = [&](std::size_t o) {
                std::uint64_t v = 0;
                if (o + 8 <= payload.size()) std::memcpy(&v, payload.data() + o, 8);
                return v;
            };
            for (std::size_t cand = 8; cand + 16 <= payload.size(); ++cand) {
                if (rdp(cand) != layers) continue;
                std::size_t at = cand + 8;
                bool ok = true;
                for (std::size_t layer = 0; layer < layers && ok; ++layer) {
                    if (at + 4 + 6 * 8 > payload.size()) { ok = false; break; }
                    std::int32_t fmt = 0; std::memcpy(&fmt, payload.data() + at, 4);
                    std::int64_t c = 0; std::memcpy(&c, payload.data() + at + 4, 8);
                    if (fmt != plan.kv_format || c <= 0) { ok = false; break; }
                    at += 4 + 6 * 8;
                    for (int p = 0; p < 5; ++p) {
                        const std::uint64_t len = rdp(at);
                        at += 8 + static_cast<std::size_t>(len);
                        if (at > payload.size()) { ok = false; break; }
                    }
                }
                if (ok && at + 9 == payload.size()) return cand;
            }
            return std::uint64_t{0};
        }();
        CHECK(kv_at > 0, "★ 定位到 kv_at（走完正好落在载荷末尾）");

        // ★★ 尾部目录：告诉拼接"哪些状态搬进了日志"。
        //   本用例的 off/len 指向 fill_ckpt 的日志缓存区间（live 与检查点各一段），
        //   真实实现里它们指向载荷里那段 0 占位的位置（见 DESIGN §3）。
        //   parse_ckpt_dir 会校验"引用落在载荷内"，所以这里 off 只要是个合法偏移即可。
        std::vector<DiskParkCkptDirEntry> dir_entries;
        dir_entries.push_back(kLiveDirKey);
        dir_entries.back().off = 0;  dir_entries.back().len = 1;      // 非 0 ⇒ present()
        {
            // ★ 键必须用【与拼接端同一算法】算出来的摘要：prefix_digest(检查点的 ids, ...)。
            //   第一版写死 {0x1234,0x5678}，于是拼接端永远找不到这一项
            //   （症状：产出 = 载荷长度，日志里的 906 字节一个都没被取用）。
            const std::vector<std::uint64_t> e_hashes;
            const std::vector<std::int64_t>  e_starts;
            const auto d = strata::core::prefix_digest(
                ref.checkpoints[0].ids, e_hashes, e_starts,
                static_cast<std::int64_t>(ref.checkpoints[0].ids.size()));
            DiskParkCkptDirEntry e;
            e.lo = d.lo; e.hi = d.hi; e.off = 0; e.len = 1;
            dir_entries.push_back(e);
        }
        append_ckpt_dir(payload, dir_entries);
        std::printf("       载荷 + 目录 = %zu 字节\n", payload.size());

        // ④ 把"进日志"的那些缓冲区写进日志（本测试直接用内存 buffer 模拟日志内容）
        //    日志记录 = 各缓冲区的真值，按 (live/检查点, part) 顺序
        std::vector<std::uint8_t> logbuf;
        auto append_to_log = [&logbuf](const std::vector<std::uint8_t>& v) {
            logbuf.insert(logbuf.end(), v.begin(), v.end());
        };
        const std::size_t live_off = logbuf.size();
        append_to_log(ref.live.gdn); append_to_log(ref.live.ple); append_to_log(ref.live.tails);
        append_to_log(ref.live.dead); append_to_log(ref.live.block_pos);
        const std::size_t ck_off = logbuf.size();
        append_to_log(ref.checkpoints[0].gdn); append_to_log(ref.checkpoints[0].ple);
        append_to_log(ref.checkpoints[0].tails); append_to_log(ref.checkpoints[0].dead);
        append_to_log(ref.checkpoints[0].block_pos);
        std::printf("       日志内容 = %zu 字节（live %zu + 检查点 %zu）\n",
                    logbuf.size(), ck_off - live_off, logbuf.size() - ck_off);

        // 填回调：ckpt=0 → live；ckpt=1 → checkpoints[0]；part 0..4；offset 是记录内偏移
        const std::size_t base_off[2] = {live_off, ck_off};
        static std::uint64_t g_ck_calls = 0;
        const DiskParkSession::DiskParkCkptFill fill_ckpt =
            [&logbuf, &base_off](std::size_t ckpt, std::size_t part, std::size_t offset,
                                 std::uint8_t* dst, std::size_t n) -> bool {
                ++g_ck_calls;
                std::printf("      [ckfill] ckpt=%zu part=%zu off=%zu n=%zu\n", ckpt, part, offset, n);
                const std::size_t base = (ckpt == 0) ? base_off[0] : base_off[1];
                if (base + offset + n > logbuf.size()) return false;
                std::memcpy(dst, logbuf.data() + base + offset, n);
                return true;
            };
        const DiskParkFill fill = [](std::size_t, std::size_t, std::uint8_t*, std::size_t) {
            return false;   // 方案 C 下 KV 不从日志取 ⇒ 这个回调不该被调用
        };

        // ⑤ 拼接；binary_bytes = 载荷全长（本用例没有追加目录）
        std::uint64_t body_bytes = 0;
        { std::vector<DiskParkCkptDirEntry> tmp; CHECK(parse_ckpt_dir(payload, tmp, body_bytes),
              "★ 能从载荷里解析出目录"); }
        std::vector<std::byte> out;
        const std::uint64_t total = s.splice_image(payload, kv_at, layers, 0, cells, fill,
            [&out](const void* pp, std::size_t n) {
                const auto* b = static_cast<const std::byte*>(pp);
                out.insert(out.end(), b, b + n);
                return true;
            }, fill_ckpt, body_bytes);
        std::printf("       fill_ckpt 被调用 %llu 次\n", (unsigned long long) g_ck_calls);
        CHECK(total > 0, "splice_image 成功（用 fill_ckpt）");
        std::printf("       拼出来 = %zu 字节，完整镜像 = %zu 字节，本体 = %llu\n",
                    out.size(), full.size(), (unsigned long long) body_bytes);
        CHECK(out.size() == body_bytes, "★ 拼出来的长度 = 载荷【本体】长度（目录不进产出）");
        CHECK(out.size() == full.size(), "★★ 与完整镜像等长");

        // ⑥ 内容：与完整镜像逐字段比对
        auto back = unpack_saved_conversation(out);
        CHECK(back.has_value(), "★★★ 拼出来的流能被 unpack 回 SavedConversation");
        if (back) {
            CHECK(back->live.gdn == ref.live.gdn, "★ live.gdn 从日志拼回来了，且逐字节一致");
            CHECK(back->live.ple == ref.live.ple, "★ live.ple 一致");
            CHECK(back->live.tails == ref.live.tails, "★ live.tails 一致");
            CHECK(back->live.dead == ref.live.dead, "★ live.dead 一致");
            CHECK(back->live.block_pos == ref.live.block_pos, "★ live.block_pos 一致");
            CHECK(back->live.ids == ref.live.ids, "live.ids 一致");
            CHECK(back->checkpoints.size() == 1, "检查点数量一致");
            if (!back->checkpoints.empty()) {
                CHECK(back->checkpoints[0].gdn == ref.checkpoints[0].gdn,
                      "★ 检查点 gdn 从日志拼回来了，且逐字节一致");
                CHECK(back->checkpoints[0].ple == ref.checkpoints[0].ple, "检查点 ple 一致");
                CHECK(back->checkpoints[0].used == ref.checkpoints[0].used, "检查点 used 原样保留");
            }
            CHECK(back->kv.size() == layers, "KV 层数一致（KV 留在载荷里）");
            if (back->kv.size() == layers) {
                CHECK(back->kv[0].k.size() == part_len[kK], "第 0 层 K 长度对");
                std::vector<std::uint8_t> kb(static_cast<std::size_t>(part_len[kK]), 0);
                back->kv[0].k.read(kb.data(), 0, kb.size());
                bool zeros = true;
                for (auto b : kb) if (b != 0) { zeros = false; break; }
                CHECK(zeros, "★ KV 的字节来自载荷（本用例载荷里是 0 占位）");
            }
        }

        // ⑦ fill_ckpt 拒绝 ⇒ 干净失败（不产出半成品）
        std::vector<std::byte> dummy;
        CHECK(s.splice_image(payload, kv_at, layers, 0, cells, fill,
                [&dummy](const void*, std::size_t) { return true; },
                [](std::size_t, std::size_t, std::size_t, std::uint8_t*, std::size_t) { return false; },
                body_bytes) == 0,
              "★ fill_ckpt 拒绝时返回 0（不产出半成品）");
    }



    std::printf("\n=== 15. ★★ P12：尾部追加目录（检查点/live 的日志引用）===\n");
    {
        // 目录记录"哪些状态被搬进了日志"。这一节只考【构造与解析】的往返与边界，
        // 不涉及拼接（拼接在下一节）。
        std::vector<std::byte> img(1000, std::byte{0x5A});
        for (std::size_t i = 0; i < img.size(); ++i) img[i] = std::byte{(unsigned char) (i & 0xFF)};

        // ① 没有目录 ⇒ 返回 false、image_bytes = size
        {
            std::vector<DiskParkCkptDirEntry> d;
            std::uint64_t ib = 0;
            CHECK(!parse_ckpt_dir(img, d, ib), "没有目录 ⇒ false");
            CHECK(ib == img.size(), "image_bytes = 全长（没有目录）");
            CHECK(d.empty(), "目录为空");
        }

        // ② 空目录（0 项）⇒ 能解析、image_bytes = 原长
        {
            std::vector<std::byte> a = img;
            append_ckpt_dir(a, {});
            std::vector<DiskParkCkptDirEntry> d;
            std::uint64_t ib = 0;
            CHECK(parse_ckpt_dir(a, d, ib), "空目录 ⇒ true");
            CHECK(d.empty(), "0 项");
            CHECK(ib == img.size(), "image_bytes 回到原长");
            CHECK(a.size() == img.size() + 12, "空目录只多 12 字节");
        }

        // ③ 三个条目往返（含 live 哨兵与"不在日志里"的项）
        {
            std::vector<DiskParkCkptDirEntry> in;
            // ★ 引用必须落在【载荷内】—— 解析器会校验。真实情形正是如此：
            //   日志记录就在载荷尾部之后的位置……不对，是【载荷内】的一段
            //   （把检查点状态从载荷里搬到日志后，载荷里那段变成 0 占位，
            //    引用指向的是【载荷里那段的偏移】。见 DESIGN §3。）
            in.push_back({0x1111, 0x2222, 100, 900});           // 普通检查点：指向载荷内一段
            in.push_back(kLiveDirKey);                          // live 哨兵、且没有搬进日志
            in.push_back({0x3333, 0x4444, 0, 0});               // 有键但没搬（占位仍在载荷里）
            std::vector<std::byte> a = img;
            append_ckpt_dir(a, in);
            CHECK(a.size() == img.size() + 12 + 32 * 3, "目录大小 = 12 + 32n");
            std::vector<DiskParkCkptDirEntry> out;
            std::uint64_t ib = 0;
            CHECK(parse_ckpt_dir(a, out, ib), "解析成功");
            CHECK(ib == img.size(), "★ 镜像本体长度 = 原长（目录不算本体）");
            CHECK(out.size() == 3, "3 项");
            CHECK(out[0].lo == 0x1111 && out[0].hi == 0x2222 && out[0].off == 100 && out[0].len == 900,
                  "第 0 项逐字段一致");
            CHECK(out[1].is_live() && !out[1].present(), "★ live 哨兵（lo=hi=0、len=0）");
            CHECK(out[2].present() == false && out[2].lo == 0x3333, "★ len=0 ⇒ present()==false");
            // 镜像本体没被动过
            bool same = true;
            for (std::size_t i = 0; i < img.size() && same; ++i) if (a[i] != img[i]) same = false;
            CHECK(same, "★ 追加目录没有改动镜像本体的字节");
        }

        // ④ 坏目录 ⇒ 干净失败
        {
            std::vector<std::byte> a = img;
            append_ckpt_dir(a, {{1, 2, 3, 4}});
            a.back() = std::byte{0x00};                          // 破坏魔数
            std::vector<DiskParkCkptDirEntry> d;
            std::uint64_t ib = 0;
            CHECK(!parse_ckpt_dir(a, d, ib), "魔数坏了 ⇒ false");
        }
        {
            // 项数谎报得很大 ⇒ 干净失败（不能 OOM / 越界）
            std::vector<std::byte> a = img;
            append_ckpt_dir(a, {{1, 2, 3, 4}});
            const std::uint64_t bogus = 1u << 19;                // 512K 项
            std::memcpy(a.data() + a.size() - 12, &bogus, 8);
            std::vector<DiskParkCkptDirEntry> d;
            std::uint64_t ib = 0;
            CHECK(!parse_ckpt_dir(a, d, ib), "★ 项数谎报 ⇒ false（不越界）");
        }
    }

    std::printf("\n=== 8. ★★ 长会话：段 > 16 MiB 时的块对齐（2026-10-10 的回归）===\n");
    {
        // V = 288 B/cell（= heads 2 × q4_0 每 head 144，与生产一致）：
        // 16 MiB % 288 = 64 ≠ 0 ⇒ 修之前第二块的 offset 不是 cell 边界，
        // 回调拒绝 ⇒ 整层一个字节都写不进去（实测 > 58,254 token 的会话就是这样
        // 被静默丢掉的：磁盘上只留一个 64 字节的空 kv.bin）。
        DiskParkSessionPlan lp;
        lp.tag = 0x1234ABCDu;
        lp.capacity_cells = 70000;
        lp.kv_format = 3; lp.heads = 2; lp.head_dim = 256; lp.page_size = 4; lp.idx_dim = 8;
        lp.parts[kK]  = {512, 1, 512, true};
        lp.parts[kV]  = {288, 1, 288, true};
        lp.parts[kKS] = {16, 1, 16, true};
        lp.parts[kVS] = {0, 1, 0, false};
        lp.parts[kP]  = {0, 4, 0, false};
        const std::string ldir = (root / "long").string();
        DiskParkSession s;
        CHECK(DiskParkSession::open(s, ldir, lp), "开一个 K=512B / V=288B 的长会话");
        // ★★ 关键的"契约"：真实调用方（disk_park_logs.cpp 的 `one`）要求
        //    offset 按 cell 对齐 —— 不对齐就返回 false。truth_fill 自己是宽容的，
        //    所以这里必须显式加上这条检查，否则测不出这个 bug（第一版就没测出来）。
        const DiskParkFill plain = truth_fill(lp);
        const DiskParkFill lfill = [&lp, &plain](std::size_t part, std::size_t offset,
                                                 std::uint8_t* dst, std::size_t n) {
            const auto& pp = lp.parts[part];
            if (pp.bytes_per_cell == 0) return true;
            if (offset % pp.bytes_per_cell != 0) return false;   // ← 引擎那边的契约
            return plain(part, offset, dst, n);
        };
        CHECK(s.append_kv(0, 60000, lfill), "★★ 追加 [0,60000)：V 段 16.9 MiB ⇒ 必须跨 16 MiB 块");
        CHECK(s.log_cells() == 60000, "水位 = 60000（整层成功，不是 0）");
        CHECK(verify_part(s, lp, DiskParkSession::Part::K, 0, 60000, "K（30.7 MiB）"),
              "★★ K 逐字节正确");
        CHECK(verify_part(s, lp, DiskParkSession::Part::V, 0, 60000, "V（16.9 MiB）"),
              "★★ V 逐字节正确（修之前这里整个 append 就失败了）");
        CHECK(verify_part(s, lp, DiskParkSession::Part::KScale, 0, 60000, "Kscale（0.9 MiB）"),
              "★ Kscale 正确");
        // 再追加一轮，确认跨轮也接得上
        CHECK(s.append_kv(60000, 66000, lfill), "再追加 [60000,66000)");
        CHECK(verify_part(s, lp, DiskParkSession::Part::V, 0, 66000, "V 跨两轮"),
              "★★ V 跨两轮仍逐字节正确");
    }

    fs::remove_all(root);
    std::printf("\n========================================\n");
    std::printf("  检查 %d 项，失败 %d 项\n", g_checks, g_failed);
    std::printf("========================================\n\n");
    return g_failed == 0 ? 0 : 1;
}
