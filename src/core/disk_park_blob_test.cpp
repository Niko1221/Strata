// DiskParkBlobStream 的单元测试：只追加 + 按内容去重。
//
// 这一层存在的唯一理由是【同一个检查点状态只写一次】，所以测试的重点就是：
//   · 第二次 put 同一个键 ⇒ 零写入（字节数不变），且返回同一个区间
//   · 跨"重启"（重新构造对象）读回来逐字节一致
//   · 记忆被淘汰之后行为仍然正确（只是多写一份）

#include "strata/core/disk_park_blob.hpp"

#include <cstdio>
#include <cstring>
#include <filesystem>
#include <map>
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

// 一份 blob 的"真值"：只取决于 (seed, 偏移)
std::uint8_t truth(std::uint8_t seed, std::uint64_t off) {
    return static_cast<std::uint8_t>((seed * 131 + off * 7 + 3) & 0xFF);
}
DiskParkBlobStream::Fill make_fill(std::uint8_t seed) {
    return [seed](std::uint64_t off, std::uint8_t* dst, std::size_t n) {
        for (std::size_t i = 0; i < n; ++i) dst[i] = truth(seed, off + i);
        return true;
    };
}
// 读回来逐字节比对
bool verify(const DiskParkBlobStream& s, const DiskParkBlobRef& ref, std::uint8_t seed) {
    std::vector<std::uint8_t> got(static_cast<std::size_t>(ref.bytes), 0);
    if (!s.get(ref, [&](std::uint64_t off, const std::uint8_t* src, std::size_t n) {
            std::memcpy(got.data() + off, src, n);
            return true;
        })) return false;
    for (std::size_t i = 0; i < got.size(); ++i)
        if (got[i] != truth(seed, i)) return false;
    return true;
}
DiskParkBlobKey key(std::uint64_t lo, std::uint64_t hi) { return DiskParkBlobKey{lo, hi}; }

} // namespace

int main() {
    namespace fs = std::filesystem;
    const fs::path root = fs::temp_directory_path() / "disk_park_blob_test";
    fs::remove_all(root);
    fs::create_directories(root);
    const std::string path = (root / "checkpoints.bin").string();

    std::printf("\n=== 1. 关闭状态（path 为空）===\n");
    {
        DiskParkBlobStream s({});
        CHECK(!s.enabled(), "未启用");
        CHECK(!s.put(key(1, 2), 100, make_fill(1)).present(), "put 返回空区间");
        CHECK(!s.find(key(1, 2)).present(), "find 为空");
    }

    std::printf("\n=== 2. 写入 + 读回逐字节一致 ===\n");
    {
        DiskParkBlobStream s({path, 64});
        CHECK(s.enabled(), "启用");
        const auto r1 = s.put(key(0xAAAA, 0xBBBB), 3u << 20, make_fill(7));
        CHECK(r1.present() && r1.bytes == (3u << 20), "写了一份 3 MiB");
        CHECK(verify(s, r1, 7), "★ 读回来逐字节一致（跨多个 1 MiB 分块）");
        CHECK(s.stats().appends == 1, "appends=1");
        CHECK(s.stats().dedup_hits == 0, "还没有去重命中");
    }

    std::printf("\n=== 3. ★★ 同一个键再 put ⇒ 零写入，且返回同一个区间 ===\n");
    {
        DiskParkBlobStream s({path, 64});
        const std::uint64_t before = s.bytes_on_disk();
        const auto again = s.put(key(0xAAAA, 0xBBBB), 3u << 20, make_fill(7));
        CHECK(again.present(), "返回了区间（不是空）");
        CHECK(s.bytes_on_disk() == before, "★★ 文件一字节都没长（dedup 生效）");
        CHECK(s.stats().dedup_hits == 1 && s.stats().appends == 0, "★ 记为命中、没有写入");
        CHECK(verify(s, again, 7), "★ 命中的区间读回来仍然正确");
        CHECK(s.find(key(0xAAAA, 0xBBBB)) == again, "find 与 put 返回一致");
    }

    std::printf("\n=== 4. ★ 不同的键 ⇒ 追加在末尾，老区间不受影响 ===\n");
    {
        DiskParkBlobStream s({path, 64});
        const auto old_ref = s.find(key(0xAAAA, 0xBBBB));
        const auto r2 = s.put(key(0xCCCC, 0xDDDD), 1u << 20, make_fill(9));
        CHECK(r2.present(), "新键写入成功");
        CHECK(r2.offset >= old_ref.offset + old_ref.bytes, "★ 新数据落在老数据之后（只追加）");
        CHECK(verify(s, old_ref, 7), "★ 老区间读回来仍然正确");
        CHECK(verify(s, r2, 9), "新区间也正确");
    }

    std::printf("\n=== 5. ★★ 跨重启：索引从文件里重建，去重仍然有效 ===\n");
    {
        // ★ 这一节是这次设计的转折点：第一版把"key → 区间"的表只放在内存里，
        //   重启后表空了 ⇒ 同一个检查点会被重写一遍 ⇒ **去重等于没用**。
        //   现在条目头自带 key，open() 扫一遍就能重建。
        DiskParkBlobStream fresh({path, 64});
        CHECK(fresh.enabled(), "重开成功");
        const auto r = fresh.find(key(0xAAAA, 0xBBBB));
        CHECK(r.present(), "★★ 重开后 find 仍能命中（索引从文件重建）");
        CHECK(verify(fresh, r, 7), "★★ 按区间读回来逐字节一致");

        // 再一次 put 同一个键 ⇒ 依然零写入
        const std::uint64_t before = fresh.bytes_on_disk();
        const auto again = fresh.put(key(0xAAAA, 0xBBBB), 3u << 20, make_fill(7));
        CHECK(again == r, "★★ 返回同一个区间");
        CHECK(fresh.bytes_on_disk() == before, "★★★ 跨重启之后去重依然生效（零写入）");
    }

    std::printf("\n=== 6. 记忆上限：淘汰最久未用的键，之后会重写一份 ===\n");
    {
        const std::string p2 = (root / "small.bin").string();
        DiskParkBlobStream s({p2, 2});                 // 只记 2 个键
        const auto a = s.put(key(1, 1), 4096, make_fill(1));
        const auto b = s.put(key(2, 2), 4096, make_fill(2));
        CHECK(a.present() && b.present(), "两份都写进去了");
        (void) s.find(key(1, 1));                        // 把 1 用一次（它变成最新）—— 要的是副作用
        const auto c = s.put(key(3, 3), 4096, make_fill(3));
        CHECK(c.present(), "第三份写入成功");
        CHECK(s.stats().evicted_keys >= 1, "★ 有键被淘汰出记忆");
        // 被淘汰的那个键再 put 会重写（文件变长）—— 有意的降级：多写，不错写
        const std::uint64_t before = s.bytes_on_disk();
        const auto a2 = s.put(key(1, 1), 4096, make_fill(1));
        CHECK(a2.present(), "重写成功");
        CHECK(s.bytes_on_disk() > before || s.stats().dedup_hits > 0,
              "★ 要么重写了、要么刚好还在记忆里（两种都正确）");
        // 但重开之后索引重建 ⇒ 那个键又命中了（这正是跨重启去重的意义）
        DiskParkBlobStream reopened({p2, 64});
        CHECK(reopened.find(key(1, 1)).present(), "★★ 重开后连被淘汰过的键也能重建出来");
        CHECK(verify(s, a2, 1), "★ 重写的那份内容正确");
    }

    std::printf("\n=== 7. 坏区间 / 坏回调 ⇒ 干净失败 ===\n");
    {
        DiskParkBlobStream s({path, 64});
        CHECK(!s.get(DiskParkBlobRef{0, 0}, [](std::uint64_t, const std::uint8_t*, std::size_t) { return true; }),
              "空区间 ⇒ false");
        CHECK(!s.get(DiskParkBlobRef{(std::uint64_t(1) << 40), 16}, [](std::uint64_t, const std::uint8_t*, std::size_t) { return true; }),
              "越界区间 ⇒ false");
        const auto r = s.find(key(0xAAAA, 0xBBBB));
        CHECK(!s.get(r, [](std::uint64_t, const std::uint8_t*, std::size_t) { return false; }),
              "回调拒绝 ⇒ false");
        CHECK(!s.put(key(5, 5), 100, [](std::uint64_t, std::uint8_t*, std::size_t) { return false; }).present(),
              "fill 拒绝 ⇒ 空区间");
    }

    std::printf("\n=== 8. ★★ P12 检查点状态往返：进日志、再按目录拼回来 ===\n");
    {
        // 这一节考的是【引擎写路径用它做的那件事】：
        //   检查点的 5 个缓冲区首尾相接进 blob → 记下 (摘要, 长度)
        //   → 另一端按同一个摘要取回、逐字节相等。
        // 用真实尺寸的一小份（gdn 512 / ple 64 / tails 16 / dead 8 / block_pos 4）。
        const std::size_t L[5] = {512, 64, 16, 8, 4};
        std::vector<std::uint8_t> state;
        for (int p2 = 0; p2 < 5; ++p2)
            for (std::size_t i = 0; i < L[p2]; ++i)
                state.push_back((std::uint8_t) ((p2 * 37 + i) & 0xFF));

        const std::string p3 = (root / "ckpt.bin").string();
        DiskParkBlobStream bs({p3, 64});
        CHECK(bs.enabled(), "blob 流打开");

        const DiskParkBlobKey k1{0x1111, 0x2222};
        const DiskParkBlobKey k2{0x3333, 0x4444};
        const auto r1 = bs.put(k1, state.size(), [&state](std::uint64_t off, std::uint8_t* dst, std::size_t n) {
            std::memcpy(dst, state.data() + off, n);
            return true;
        });
        CHECK(r1.present() && r1.bytes == state.size(), "第 0 个检查点写进去了");

        // ★ 去重：同一个摘要第二次 ⇒ 零写入（跨轮命中的基础）
        const std::uint64_t before = bs.bytes_on_disk();
        const auto r1b = bs.put(k1, state.size(), [&state](std::uint64_t off, std::uint8_t* dst, std::size_t n) {
            std::memcpy(dst, state.data() + off, n);
            return true;
        });
        CHECK(r1b == r1, "★ 同一个摘要再写 ⇒ 返回同一个区间");
        CHECK(bs.bytes_on_disk() == before, "★★ 零写入（内容寻址去重生效）");

        auto r2 = bs.put(k2, state.size(), [&state](std::uint64_t off, std::uint8_t* dst, std::size_t n) {
            std::memcpy(dst, state.data() + off, n);
            return true;
        });
        CHECK(r2.present(), "第 1 个检查点也写进去了");

        // ★ 跨"重启"取回：内容寻址 ⇒ 不需要额外的索引文件
        DiskParkBlobStream fresh({p3, 64});
        const auto back = fresh.find(k1);
        CHECK(back.present(), "★★ 重开后按摘要仍能找到（索引从文件重建）");
        std::vector<std::uint8_t> got(state.size(), 0);
        CHECK(fresh.get(back, [&got](std::uint64_t off, const std::uint8_t* src, std::size_t n) {
            std::memcpy(got.data() + off, src, n);
            return true;
        }), "取回成功");
        CHECK(got == state, "★★★ 逐字节一致");

        // ★ 按 5 个缓冲区的边界切开也一致（拼接端就是这么用的）
        std::size_t off = 0;
        bool parts_ok = true;
        for (int p2 = 0; p2 < 5 && parts_ok; ++p2) {
            for (std::size_t i = 0; i < L[p2]; ++i)
                if (got[off + i] != (std::uint8_t) ((p2 * 37 + i) & 0xFF)) { parts_ok = false; break; }
            off += L[p2];
        }
        CHECK(parts_ok && off == state.size(), "★ 按 5 个缓冲区的边界切开也逐字节一致");
    }

    // ==================================================================
    // ★★★ 6. P16 模拟：检查点落盘闸门（一份 112.6 MiB，值不值得再写）
    // ==================================================================
    std::printf("\n=== 6. ★★★ P16 模拟：检查点落盘闸门 ===\n");
    {
        DiskParkBlobStream s6({(root / "gate.bin").string(), 64});
        CHECK(s6.worth_storing("sess-A", 1000, 8192), "第一次（没有记录）⇒ 值得写");
        s6.note_stored("sess-A", 1000);
        CHECK(!s6.worth_storing("sess-A", 1500, 8192), "推进 500 < 8192 ⇒ 跳过");
        CHECK(!s6.worth_storing("sess-A", 9000, 8192), "推进 8000 < 8192 ⇒ 跳过");
        CHECK(s6.worth_storing("sess-A", 10000, 8192), "推进 9000 ≥ 8192 ⇒ 值得写");
        s6.note_stored("sess-A", 10000);
        CHECK(!s6.worth_storing("sess-A", 12000, 8192), "再推进 2000 ⇒ 跳过");
        CHECK(s6.worth_storing("sess-B", 1001, 8192), "★ 另一条会话互不影响");
        CHECK(s6.worth_storing("sess-A", 10001, 0), "min_advance=0 ⇒ 关掉闸门");
        CHECK(!s6.worth_storing("sess-C", 500, 8192, 24000), "★ 500 token 的短会话 ⇒ 不写");
        CHECK(!s6.worth_storing("sess-C", 23999, 8192, 24000), "★ 23999 ⇒ 仍不写");
        CHECK(s6.worth_storing("sess-C", 24000, 8192, 24000), "★ 24000 ⇒ 够长了，可以写");
        s6.note_stored("sess-A", 500);
        CHECK(!s6.worth_storing("sess-A", 8000, 8192), "★ 水位不倒退（500 不覆盖 10000）");
        // ---- 用实测那 27 轮的真实推进量做模拟（00:02:53–00:17:17 的主对话）----
        DiskParkBlobStream sim({(root / "sim.bin").string(), 64});
        const std::int64_t adv[27] = {188, 1069, 11509, 5945, 80, 233, 263, 692, 0, 203, 1286, 1488,
                                      158, 626, 578, 1366, 132, 1519, 208, 566, 11, 90, 251, 243, 314,
                                      3273, 6473};
        std::int64_t at = 0;
        int written = 0, skipped = 0;
        for (const std::int64_t d : adv) {
            at += d;
            if (sim.worth_storing("sess", at, 8192)) { sim.note_stored("sess", at); ++written; }
            else ++skipped;
        }
        std::printf("  → 实测那 27 轮：写 %d 份、闸门跳过 %d 份（112.6 MiB/份）\n", written, skipped);
        CHECK(written + skipped == 27, "27 轮全覆盖");
        CHECK(written <= 6, "★★★ 只写 ≤6 份（修复前 27 份）");
        CHECK(skipped >= 21, "★★★ 至少跳过 21 份 ⇒ 检查点写量降到 1/5 以下");
        // ---- 两个闸门合起来：实测那一轮里还有 9 个短会话（300~2000 token）----
        DiskParkBlobStream sim2({(root / "sim2.bin").string(), 64});
        const std::int64_t side[9] = {359, 1216, 2016, 1818, 443, 457, 486, 476, 638};
        int w2 = 0, s2 = 0;
        for (const std::int64_t t : side) {
            if (sim2.worth_storing("side", t, 8192, 24000)) { sim2.note_stored("side", t); ++w2; }
            else ++s2;
        }
        std::printf("  → 9 个短会话：写 %d 份、跳过 %d 份\n", w2, s2);
        CHECK(w2 == 0 && s2 == 9, "★★★ 9 个短会话一份检查点都不写（修复前 9 份 = 1.01 GiB）");
        // 主对话也带上绝对下限重跑一遍（前几轮 < 24000 的那几次也应当跳掉）
        DiskParkBlobStream sim3({(root / "sim3.bin").string(), 64});
        std::int64_t at3 = 0;
        int w3 = 0, s3 = 0;
        for (const std::int64_t d : adv) {
            at3 += d;
            if (sim3.worth_storing("sess", at3, 8192, 24000)) { sim3.note_stored("sess", at3); ++w3; }
            else ++s3;
        }
        std::printf("  → 主对话（含绝对下限）：写 %d 份、跳过 %d 份\n", w3, s3);
        CHECK(w3 + s3 == 27, "27 轮全覆盖");
        CHECK(w3 <= 3, "★★★ 主对话最多写 3 份");
    }

    fs::remove_all(root);
    std::printf("\n========================================\n");
    std::printf("\n=== 9. ★★★ §7.2 版本门：别的标签写下的 blob 读不到 ===\n");
    {
        const std::string pg = (root / "gated.bin").string();
        constexpr std::uint32_t TA = 0x11111111U, TB = 0x22222222U;
        const DiskParkBlobKey k{1234, 5678};
        auto fill = [](std::uint64_t off, std::uint8_t* dst, std::size_t n) {
            for (std::size_t i = 0; i < n; ++i) dst[i] = (std::uint8_t) (off + i);
            return true;
        };
        {   DiskParkBlobStream a({pg, 64, TA});
            CHECK(a.enabled(), "带标签的一份能开");
            CHECK(a.put(k, 4096, fill).present(), "标签 A 写进去了");
        }
        CHECK(DiskParkBlobStream::file_tag(pg) == TA, "★ file_tag 读回 A");
        {   DiskParkBlobStream b({pg, 64, TB});
            CHECK(b.enabled(), "标签不符 ⇒ 挪开老的、另起一份（这一层还活着）");
            CHECK(!b.find(k).present(), "★★ A 的那条记录 B 读不到");
            CHECK(std::filesystem::exists(pg + ".tag-11111111.stale"), "★ 老数据被挪开，没有被删");
        }
        CHECK(DiskParkBlobStream::file_tag(pg) == TB, "新的一份带的是当前标签");
        {   const std::string po = (root / "legacy.bin").string();
            std::FILE* f = std::fopen(po.c_str(), "wb");
            const unsigned char h[16] = {'S','K','B','1', 1,0,0,0, 0,0,0,0, 0,0,0,0};
            if (f) { std::fwrite(h, 1, 16, f); std::fclose(f); }
            DiskParkBlobStream s({po, 64, TA});
            CHECK(!s.enabled(), "★★ 老格式 SKB1（没有版本门）⇒ 整层不读");
        }
        {   const std::string pn = (root / "brand-new.bin").string();
            DiskParkBlobStream s({pn, 64, TA});
            CHECK(s.enabled() && s.tracked_entries() == 0, "新文件：建头 + 零条目");
        }
    }

    std::printf("  检查 %d 项，失败 %d 项\n", g_checks, g_failed);
    std::printf("========================================\n\n");
    return g_failed == 0 ? 0 : 1;
}
