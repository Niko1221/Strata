#include "strata/core/disk_park_store.hpp"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <random>
#include <vector>

using namespace strata::core;

static int g_checks = 0, g_failed = 0;
#define CHECK(cond, msg)                                                          \
    do {                                                                          \
        ++g_checks;                                                               \
        if (!(cond)) { ++g_failed; std::printf("  FAIL  %s\n", (msg)); }          \
        else         { std::printf("  ok    %s\n", (msg)); }                      \
    } while (0)

static std::vector<std::byte> pattern(std::size_t n, std::uint8_t seed) {
    std::vector<std::byte> v(n);
    for (std::size_t i = 0; i < n; ++i)
        v[i] = static_cast<std::byte>((i * 31 + seed * 7) & 0xFF);
    return v;
}

static DiskParkIdentity id_of(std::uint64_t lo, std::uint32_t tag, std::uint32_t frontier,
                              std::uint32_t chunk = 0) {
    return DiskParkIdentity{lo, lo * 0x9E3779B97F4A7C15ULL + 1, tag, frontier, chunk};
}

int main() {
    namespace fs = std::filesystem;
    const fs::path dir = fs::temp_directory_path() / "disk_park_store_test";
    fs::remove_all(dir);
    fs::create_directories(dir);

    const std::size_t kSlot = 64 * 1024;
    const std::size_t kCap  = kSlot * 8 + 4096 * 8;   // 约 8 个槽

    std::printf("\n=== 1. 基本写入 / 读取 ===\n");
    {
        DiskParkStore::Options o;
        o.path = (dir / "store.dat").string();
        o.slot_size = kSlot;
        o.capacity_bytes = kCap;
        DiskParkStore s(o);
        CHECK(s.slot_count() >= 8, "容量至少 8 个槽");
        CHECK(s.live_slots() == 0, "新库是空的");

        auto a = pattern(kSlot, 1);
        CHECK(s.upsert_page(id_of(100, 7, 1000), a), "写入页 A");
        CHECK(s.contains(id_of(100, 7, 1000)), "contains(A)");
        CHECK(s.live_slots() == 1, "live_slots == 1");

        std::vector<std::byte> out(kSlot);
        CHECK(s.read_page(id_of(100, 7, 1000), out), "读回 A");
        CHECK(std::memcmp(out.data(), a.data(), kSlot) == 0, "A 的内容一致");
        CHECK(!s.read_page(id_of(999, 7, 1), out), "不存在的页读作 miss");
        CHECK(!s.read_page(id_of(100, 8, 1000), out), "★ tag 不同也读不到（配置隔离）");
        // ★ chunk：同一个 frontier 的不同分片必须互不覆盖
        auto a1 = pattern(kSlot, 11), a2 = pattern(kSlot, 22);
        CHECK(s.upsert_page(id_of(100, 7, 1000, 1), a1), "写入分片 1");
        CHECK(s.upsert_page(id_of(100, 7, 1000, 2), a2), "写入分片 2");
        std::vector<std::byte> o1(kSlot), o2(kSlot);
        CHECK(s.read_page(id_of(100, 7, 1000, 1), o1) && std::memcmp(o1.data(), a1.data(), kSlot) == 0,
              "★ 分片 1 内容正确");
        CHECK(s.read_page(id_of(100, 7, 1000, 2), o2) && std::memcmp(o2.data(), a2.data(), kSlot) == 0,
              "★ 分片 2 内容正确（未被分片 1 覆盖）");
        CHECK(s.read_page(id_of(100, 7, 1000, 0), out) && std::memcmp(out.data(), a.data(), kSlot) == 0,
              "★ 分片 0（清单）仍完好");
        CHECK(s.live_slots() == 3, "3 个槽都活着（清单 + 2 个分片）");
    }

    std::printf("\n=== 2. 重开（索引命中）===\n");
    {
        DiskParkStore::Options o;
        o.path = (dir / "store.dat").string();
        o.slot_size = kSlot;
        o.capacity_bytes = kCap;
        DiskParkStore s(o);
        CHECK(!s.index_rebuilt_from_scan(), "索引文件有效，未触发扫描重建");
        CHECK(s.contains(id_of(100, 7, 1000)), "重开后 A 还在");
        std::vector<std::byte> out(kSlot);
        CHECK(s.read_page(id_of(100, 7, 1000), out), "重开后能读回 A");
        CHECK(std::memcmp(out.data(), pattern(kSlot, 1).data(), kSlot) == 0, "内容仍一致");
    }

    std::printf("\n=== 3. ★ 删掉索引 → 扫描重建（跨重启的关键）===\n");
    {
        std::remove((dir / "store.dat.idx").c_str());
        DiskParkStore::Options o;
        o.path = (dir / "store.dat").string();
        o.slot_size = kSlot;
        o.capacity_bytes = kCap;
        DiskParkStore s(o);
        CHECK(s.index_rebuilt_from_scan(), "★ 索引丢失 → 从数据文件扫描重建");
        CHECK(s.contains(id_of(100, 7, 1000)), "★ 扫描后 A 被恢复");
        std::vector<std::byte> out(kSlot);
        CHECK(s.read_page(id_of(100, 7, 1000), out), "扫描重建后能读回");
        CHECK(std::memcmp(out.data(), pattern(kSlot, 1).data(), kSlot) == 0, "内容仍一致");
    }

    std::printf("\n=== 4. LRU 淘汰 ===\n");
    {
        DiskParkStore::Options o;
        o.path = (dir / "lru.dat").string();
        o.slot_size = kSlot;
        o.capacity_bytes = kSlot * 4 + 4096 * 4;      // 约 4 个槽
        DiskParkStore s(o);
        const std::uint32_t n = s.slot_count();
        for (std::uint32_t i = 0; i < n; ++i)
            CHECK(s.upsert_page(id_of(i, 1, i), pattern(kSlot, static_cast<std::uint8_t>(i))), "填满");

        // 碰一下 0 号，让它变热
        CHECK(s.touch(id_of(0, 1, 0)), "touch 0 号（变热）");
        std::vector<DiskParkIdentity> ev;
        CHECK(s.upsert_page(id_of(1000, 1, 1000), pattern(kSlot, 9), &ev), "再写一个 → 触发淘汰");
        CHECK(ev.size() == 1, "淘汰了 1 个");
        CHECK(!(ev.size() == 1 && ev[0] == id_of(0, 1, 0)), "★ 热的那页没被淘汰");
        CHECK(s.contains(id_of(0, 1, 0)), "0 号仍在");
        if (ev.size() == 1) CHECK(!s.contains(ev[0]), "被淘汰的确实不在了");
        CHECK(s.live_slots() == n, "槽数不变（替换而非增长）");
    }

    std::printf("\n=== 5. ★ CRC：损坏的页读作 miss（不是脏数据）===\n");
    {
        DiskParkStore::Options o;
        o.path = (dir / "crc.dat").string();
        o.slot_size = kSlot;
        o.capacity_bytes = kSlot * 2 + 4096 * 2;
        {
            DiskParkStore s(o);
            CHECK(s.upsert_page(id_of(5, 1, 5), pattern(kSlot, 5)), "写入待损坏的页");
            s.flush_index();
        }
        // 直接在负载中间改一个字节
        std::FILE* f = std::fopen(o.path.c_str(), "r+b");
        CHECK(f != nullptr, "打开数据文件以便损坏");
        if (f) {
            const long off = static_cast<long>(DiskParkStore::kSlotHeaderSize) + 4096;
            std::fseek(f, off, SEEK_SET);
            const unsigned char bad = 0x5A;
            std::fwrite(&bad, 1, 1, f);
            std::fclose(f);
        }
        DiskParkStore s(o);
        std::vector<std::byte> out(kSlot);
        CHECK(!s.read_page(id_of(5, 1, 5), out), "★ CRC 不符 → 读作 miss");
        // 删掉索引再扫描：损坏的页应被扫描丢弃
        std::remove((std::string(o.path) + ".idx").c_str());
        DiskParkStore s2(o);
        CHECK(!s2.contains(id_of(5, 1, 5)), "★ 扫描重建时丢弃了损坏页");
    }

    std::printf("\n=== 6. 索引原子替换 / flush ===\n");
    {
        DiskParkStore::Options o;
        o.path = (dir / "flush.dat").string();
        o.slot_size = kSlot;
        o.capacity_bytes = kSlot * 4 + 4096 * 4;
        o.defer_index_updates = true;
        {
            DiskParkStore s(o);
            for (int i = 0; i < 3; ++i) CHECK(s.upsert_page(id_of(i, 2, i), pattern(kSlot, i)), "延迟索引写入");
            CHECK(s.live_slots() == 3, "内存索引已更新");
            s.flush_index();
        }
        DiskParkStore s(o);
        CHECK(!s.index_rebuilt_from_scan(), "flush 后索引文件有效");
        CHECK(s.live_slots() == 3, "flush 后 3 个页都在");
    }

    fs::remove_all(dir);
    std::printf("\n========================================\n");
    std::printf("  检查 %d 项，失败 %d 项\n", g_checks, g_failed);
    std::printf("========================================\n\n");
    return g_failed == 0 ? 0 : 1;
}
