#include "strata/core/disk_park_bridge.hpp"

#include <chrono>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <thread>
#include <vector>

using namespace strata::core;
using namespace std::chrono_literals;

static int g_checks = 0, g_failed = 0;
#define CHECK(cond, msg)                                                 \
    do {                                                                 \
        ++g_checks;                                                      \
        if (!(cond)) { ++g_failed; std::printf("  FAIL  %s\n", (msg)); } \
        else         { std::printf("  ok    %s\n", (msg)); }             \
    } while (0)

static std::vector<std::byte> pat(std::size_t n, std::uint8_t seed) {
    std::vector<std::byte> v(n);
    for (std::size_t i = 0; i < n; ++i)
        v[i] = static_cast<std::byte>((i * 131 + seed * 17) & 0xFF);
    return v;
}
static DiskParkIdentity id_of(std::uint64_t lo, std::uint32_t tag, std::uint32_t frontier) {
    return DiskParkIdentity{lo, lo * 0x9E3779B97F4A7C15ULL + 3, tag, frontier};
}

int main() {
    namespace fs = std::filesystem;
    const fs::path dir = fs::temp_directory_path() / "disk_park_bridge_test";
    fs::remove_all(dir);
    fs::create_directories(dir);

    const std::size_t kState = 32 * 1024, kKv = 16 * 1024;

    std::printf("\n=== 1. 关闭状态（空 base_path）= 与没有这一层完全一样 ===\n");
    {
        DiskParkBridge::Options o;   // base_path 为空
        DiskParkBridge b(o);
        CHECK(!b.enabled(), "未启用");
        std::vector<std::byte> out(kState);
        CHECK(!b.spill_sync(id_of(1, 1, 1), DiskParkKind::StateImage, pat(kState, 1)), "spill 返回 false");
        CHECK(!b.restore(id_of(1, 1, 1), DiskParkKind::StateImage, out), "restore 返回 false");
        CHECK(b.try_probe(id_of(1, 1, 1), DiskParkKind::StateImage) == nullptr, "probe 返回 nullptr");
        CHECK(b.try_submit(id_of(1, 1, 1), DiskParkKind::StateImage, pat(kState, 1)) == nullptr, "submit 返回 nullptr");
        CHECK(b.live_frontiers(DiskParkKind::StateImage).empty(), "没有候选");
        b.wait_idle();
    }

    std::printf("\n=== 2. 基本启用 ===\n");
    {
        DiskParkBridge::Options o;
        o.base_path      = (dir / "basic").string();
        o.stride[0]      = kState;
        o.stride[1]      = kKv;
        o.capacity_bytes[0] = kState * 4 + 4096 * 4;
        o.capacity_bytes[1] = kKv * 8 + 4096 * 8;
        DiskParkBridge b(o);
        CHECK(b.enabled(), "已启用");
        CHECK(b.slot_count(DiskParkKind::StateImage) >= 4, "StateImage 至少 4 槽");
        CHECK(b.slot_count(DiskParkKind::QsaKv) >= 8, "QsaKv 至少 8 槽");
        CHECK(!b.index_rebuilt_from_scan(DiskParkKind::StateImage), "首次创建没有索引可重建");

        auto a = pat(kState, 3);
        CHECK(b.spill_sync(id_of(10, 1, 1000), DiskParkKind::StateImage, a), "spill_sync 写入");
        CHECK(b.contains(id_of(10, 1, 1000), DiskParkKind::StateImage), "contains");
        std::vector<std::byte> out(kState);
        CHECK(b.restore(id_of(10, 1, 1000), DiskParkKind::StateImage, out), "restore 读回");
        CHECK(std::memcmp(out.data(), a.data(), kState) == 0, "内容一致");
        CHECK(!b.restore(id_of(11, 1, 1), DiskParkKind::StateImage, out), "不存在 → false");

        // ★ family 隔离
        CHECK(!b.contains(id_of(10, 1, 1000), DiskParkKind::QsaKv), "★ 同一个 id 在另一个 family 里读不到");
        CHECK(b.live_frontiers(DiskParkKind::StateImage).size() == 1, "live_frontiers 有 1 个");
        CHECK(b.live_frontiers(DiskParkKind::StateImage)[0] == 1000, "frontier == 1000");
    }

    std::printf("\n=== 3. ★ 跨重启（重开同一个目录）===\n");
    {
        DiskParkBridge::Options o;
        o.base_path      = (dir / "basic").string();
        o.stride[0]      = kState;
        o.stride[1]      = kKv;
        o.capacity_bytes[0] = kState * 4 + 4096 * 4;
        o.capacity_bytes[1] = kKv * 8 + 4096 * 8;
        DiskParkBridge b(o);
        CHECK(!b.index_rebuilt_from_scan(DiskParkKind::StateImage), "索引文件有效");
        CHECK(b.contains(id_of(10, 1, 1000), DiskParkKind::StateImage), "★ 重开后页还在");
        std::vector<std::byte> out(kState);
        CHECK(b.restore(id_of(10, 1, 1000), DiskParkKind::StateImage, out), "★ 重开后能读回");
        CHECK(std::memcmp(out.data(), pat(kState, 3).data(), kState) == 0, "内容一致");
    }

    std::printf("\n=== 4. ★ 索引删掉 → 扫描重建 ===\n");
    {
        std::remove((dir / "basic" / "park-state.dat.idx").c_str());
        DiskParkBridge::Options o;
        o.base_path      = (dir / "basic").string();
        o.stride[0]      = kState;
        o.stride[1]      = kKv;
        o.capacity_bytes[0] = kState * 4 + 4096 * 4;
        o.capacity_bytes[1] = kKv * 8 + 4096 * 8;
        DiskParkBridge b(o);
        CHECK(b.index_rebuilt_from_scan(DiskParkKind::StateImage), "★ 索引丢失 → 扫描重建");
        CHECK(b.contains(id_of(10, 1, 1000), DiskParkKind::StateImage), "★ 页被恢复");
    }

    std::printf("\n=== 5. 异步：spill_wait ===\n");
    {
        DiskParkBridge::Options o;
        o.base_path      = (dir / "async").string();
        o.stride[0]      = kState;
        o.capacity_bytes[0] = kState * 4 + 4096 * 4;
        DiskParkBridge b(o);
        auto a = pat(kState, 7);
        CHECK(b.spill_wait(id_of(20, 1, 200), DiskParkKind::StateImage, a, 2000ms), "spill_wait 成功");
        b.wait_idle();
        std::vector<std::byte> out(kState);
        CHECK(b.restore(id_of(20, 1, 200), DiskParkKind::StateImage, out), "读回");
        CHECK(std::memcmp(out.data(), a.data(), kState) == 0, "内容一致");
        const auto st = b.stats();
        CHECK(st.spills >= 1, "stats.spills >= 1");
        CHECK(st.spill_bytes >= kState, "stats.spill_bytes 累加");
    }

    std::printf("\n=== 6. 异步：try_submit（借用的字节）===\n");
    {
        DiskParkBridge::Options o;
        o.base_path      = (dir / "submit").string();
        o.stride[0]      = kState;
        o.capacity_bytes[0] = kState * 4 + 4096 * 4;
        DiskParkBridge b(o);
        auto a = pat(kState, 9);         // ★ 调用方持有，直到 ticket 离开 Pending
        SpillTicket t;
        for (int i = 0; i < 200 && !t; ++i) {
            t = b.try_submit(id_of(30, 1, 300), DiskParkKind::StateImage, a);
            if (!t) std::this_thread::sleep_for(1ms);
        }
        CHECK(t != nullptr, "try_submit 返回 ticket");
        if (t) {
            for (int i = 0; i < 2000 && b.poll(t) == SpillStatus::Pending; ++i)
                std::this_thread::sleep_for(1ms);
            CHECK(b.poll(t) == SpillStatus::Written, "★ 最终状态 Written");
        }
        std::vector<std::byte> out(kState);
        CHECK(b.restore(id_of(30, 1, 300), DiskParkKind::StateImage, out), "异步写入后可读");
        CHECK(std::memcmp(out.data(), a.data(), kState) == 0, "内容一致");
    }

    std::printf("\n=== 7. try_probe ===\n");
    {
        DiskParkBridge::Options o;
        o.base_path      = (dir / "submit").string();
        o.stride[0]      = kState;
        o.capacity_bytes[0] = kState * 4 + 4096 * 4;
        DiskParkBridge b(o);
        auto t = b.try_probe(id_of(30, 1, 300), DiskParkKind::StateImage);
        CHECK(t != nullptr, "probe 返回 ticket");
        if (t) {
            for (int i = 0; i < 2000 && b.poll(t) == SpillStatus::Pending; ++i)
                std::this_thread::sleep_for(1ms);
            CHECK(b.poll(t) == SpillStatus::Present, "★ 已存在 → Present");
        }
        auto t2 = b.try_probe(id_of(999, 1, 999), DiskParkKind::StateImage);
        if (t2) {
            for (int i = 0; i < 2000 && b.poll(t2) == SpillStatus::Pending; ++i)
                std::this_thread::sleep_for(1ms);
            CHECK(b.poll(t2) == SpillStatus::Missing, "★ 不存在 → Missing");
        }
    }

    std::printf("\n=== 8. restore_many：遇缺即停 ===\n");
    {
        DiskParkBridge::Options o;
        o.base_path      = (dir / "many").string();
        o.stride[1]      = kKv;                    // 只用 QsaKv
        o.capacity_bytes[1] = kKv * 8 + 4096 * 8;
        DiskParkBridge b(o);
        for (std::uint32_t i = 0; i < 5; ++i)
            CHECK(b.spill_sync(id_of(40 + i, 1, 100 * (i + 1)), DiskParkKind::QsaKv, pat(kKv, i)), "写入页");
        std::vector<DiskParkIdentity> ids{id_of(40, 1, 100), id_of(41, 1, 200), id_of(42, 1, 300)};
        std::vector<std::byte> dst(kKv * 3);
        CHECK(b.restore_many(ids, DiskParkKind::QsaKv, dst), "连续读 3 页");
        CHECK(std::memcmp(dst.data(), pat(kKv, 0).data(), kKv) == 0, "第 1 页正确");
        CHECK(std::memcmp(dst.data() + kKv * 2, pat(kKv, 2).data(), kKv) == 0, "第 3 页正确");

        std::vector<DiskParkIdentity> hole{id_of(40, 1, 100), id_of(777, 1, 999)};
        std::vector<std::byte> dst2(kKv * 2);
        CHECK(!b.restore_many(hole, DiskParkKind::QsaKv, dst2), "★ 中间缺一页 → 整体 false");
        CHECK(b.stats().restore_misses >= 1, "stats.restore_misses 累加");
    }

    std::printf("\n=== 9. LRU 淘汰通过桥接层 ===\n");
    {
        DiskParkBridge::Options o;
        o.base_path      = (dir / "lru").string();
        o.stride[1]      = kKv;
        o.capacity_bytes[1] = kKv * 4 + 4096 * 4;
        DiskParkBridge b(o);
        const std::uint32_t n = b.slot_count(DiskParkKind::QsaKv);
        for (std::uint32_t i = 0; i < n; ++i)
            b.spill_sync(id_of(50 + i, 1, i), DiskParkKind::QsaKv, pat(kKv, static_cast<std::uint8_t>(i)));
        std::vector<std::byte> out(kKv);
        (void)b.restore(id_of(50, 1, 0), DiskParkKind::QsaKv, out);   // 让 0 号变热
        CHECK(b.spill_sync(id_of(500, 1, 500), DiskParkKind::QsaKv, pat(kKv, 99)), "再写 → 触发淘汰");
        CHECK(b.contains(id_of(50, 1, 0), DiskParkKind::QsaKv), "★ 热页保住");
        CHECK(b.contains(id_of(500, 1, 500), DiskParkKind::QsaKv), "新页在");
        CHECK(b.stats().evicted_pages >= 1, "stats.evicted_pages >= 1");
    }

    std::printf("\n=== 10. drop ===\n");
    {
        DiskParkBridge::Options o;
        o.base_path      = (dir / "drop").string();
        o.stride[0]      = kState;
        o.capacity_bytes[0] = kState * 4 + 4096 * 4;
        DiskParkBridge b(o);
        CHECK(b.spill_sync(id_of(60, 1, 60), DiskParkKind::StateImage, pat(kState, 2)), "写入");
        CHECK(b.drop(id_of(60, 1, 60), DiskParkKind::StateImage), "drop");
        CHECK(!b.contains(id_of(60, 1, 60), DiskParkKind::StateImage), "drop 后不在了");
        std::vector<std::byte> out(kState);
        CHECK(!b.restore(id_of(60, 1, 60), DiskParkKind::StateImage, out), "drop 后读不到");
    }

    fs::remove_all(dir);
    std::printf("\n========================================\n");
    std::printf("  检查 %d 项，失败 %d 项\n", g_checks, g_failed);
    std::printf("========================================\n\n");
    return g_failed == 0 ? 0 : 1;
}
