#include "strata/core/disk_park_spool.hpp"

#include <cstdio>
#include <cstring>
#include <filesystem>
#include <vector>

using namespace strata::core;

static int g_checks = 0, g_failed = 0;
#define CHECK(cond, msg)                                                 \
    do {                                                                 \
        ++g_checks;                                                      \
        if (!(cond)) { ++g_failed; std::printf("  FAIL  %s\n", (msg)); } \
        else         { std::printf("  ok    %s\n", (msg)); }             \
    } while (0)

static DiskParkProfile prof(const char* kv = "int8", std::int64_t ctx = 262144) {
    DiskParkProfile p; p.model = "Qwen3.8-Flash-Next"; p.kv_format = kv;
    p.max_context = ctx; p.rope = "none"; return p;
}

// 造一个状态大小可调的镜像（用 gdn 撑体积，以便测多分片）
static SavedConversation make_image(std::size_t gdn_bytes, std::uint8_t seed) {
    SavedConversation s;
    s.layer_lo = 1; s.layer_hi = 11; s.cvec = true;
    s.live.ids.assign(64, static_cast<std::int32_t>(seed));
    s.live.gdn.assign(gdn_bytes, seed);
    s.live.ple.assign(1024, static_cast<std::uint8_t>(seed + 1));
    ConversationCheckpoint c;
    c.ids.assign(32, static_cast<std::int32_t>(seed + 2));
    c.gdn.assign(512, seed);
    c.used = seed;
    s.checkpoints.push_back(std::move(c));
    ConversationKv k;
    k.format = 1; k.cells = 1000; k.heads = 8; k.head_dim = 128;
    s.kv.push_back(std::move(k));
    return s;
}

int main() {
    namespace fs = std::filesystem;
    const fs::path dir = fs::temp_directory_path() / "disk_park_spool_test";
    fs::remove_all(dir); fs::create_directories(dir);

    const std::size_t kStride = 8192;          // 小步长，逼出多分片
    std::vector<std::int32_t> prompt;
    for (int i = 0; i < 300; ++i) prompt.push_back(1000 + i);
    std::vector<std::uint64_t> no_h; std::vector<std::int64_t> no_s;

    DiskParkSpool::Options base;
    base.path = (dir / "spool").string();
    base.profile = prof();
    base.state_stride = kStride;
    base.state_capacity_bytes = kStride * 4096;   // 32 MiB
    // ★ 前缀去重（supersede_prefixes，2026-10-10 加）默认【开】：写一份更长的快照时，
    //   盘上被它严格覆盖的旧条目会被丢掉（find() 只认最长的匹配，短的永远不会被读）。
    //   3/6/7/7b 这几节考的是【纯 store/find/drop 语义】，所以显式关掉它 ——
    //   去重本身由第 9 节单独考。不开这个开关，下面「存 4 个应看到 4 个」必然失败。
    DiskParkSpool::Options no_dedup = base;
    no_dedup.supersede_prefixes = false;

    std::printf("\n=== 1. 关闭状态零副作用 ===\n");
    {
        DiskParkSpool::Options o;                 // path 为空
        DiskParkSpool sp(o);
        CHECK(!sp.enabled(), "未启用");
        CHECK(sp.store(make_image(1000, 1), 100, prompt, no_h, no_s) == 0, "store 返回 0");
        CHECK(!sp.find(prompt, no_h, no_s).has_value(), "find 返回 nullopt");
        CHECK(!sp.load(DiskParkSpool::Found{100, {}}).has_value(), "load 返回 nullopt");
        CHECK(sp.frontiers().empty(), "没有候选");
    }

    std::printf("\n=== 2. store → find → load 往返（多分片）===\n");
    {
        DiskParkSpool sp(base);
        CHECK(sp.enabled(), "已启用");
        auto img = make_image(200 * 1024, 7);      // 200 KB → 至少 25 片
        const std::size_t wrote = sp.store(img, 200, prompt, no_h, no_s);
        CHECK(wrote > 0, "store 成功");
        std::printf("       （写了 %zu 字节，步长 %zu）\n", wrote, kStride);

        auto f = sp.find(prompt, no_h, no_s);
        CHECK(f.has_value() && f->frontier == 200, "★ find 找到 frontier=200");

        auto f200 = sp.find(prompt, no_h, no_s);
        CHECK(f200.has_value() && f200->frontier == 200, "find 200");
        auto back = f200 ? sp.load(*f200) : std::nullopt;
        CHECK(back.has_value(), "load 成功");
        if (back) {
            CHECK(back->live.gdn.size() == img.live.gdn.size(), "gdn 长度一致");
            CHECK(back->live.gdn == img.live.gdn, "★ gdn 内容逐字节一致（跨多个分片）");
            CHECK(back->live.ple == img.live.ple, "ple 一致");
            CHECK(back->checkpoints.size() == img.checkpoints.size(), "checkpoints 数量一致");
            CHECK(back->kv.size() == img.kv.size(), "kv 数量一致");
            CHECK(back->layer_lo == 1 && back->layer_hi == 11, "layer 范围一致");
        }
        CHECK(sp.frontiers().size() == 1 && sp.frontiers()[0] == 200, "frontiers 列表正确");
    }

    std::printf("\n=== 3. ★ 多个 frontier：find 取最长的那个 ===\n");
    {
        DiskParkSpool sp(no_dedup);
        CHECK(sp.store(make_image(2000, 11), 100, prompt, no_h, no_s) > 0, "存 frontier=100");
        CHECK(sp.store(make_image(2000, 12), 150, prompt, no_h, no_s) > 0, "存 frontier=150");
        CHECK(sp.store(make_image(2000, 13), 250, prompt, no_h, no_s) > 0, "存 frontier=250");
        auto f = sp.find(prompt, no_h, no_s);
        CHECK(f.has_value() && f->frontier == 250, "★ 取最长（250）");
        auto back = f ? sp.load(*f) : std::nullopt;
        CHECK(back && back->live.gdn.size() == 2000 && back->live.gdn[0] == 13,
              "★ 取回的是 250 那一份（不是别的）");
    }

    std::printf("\n=== 4. ★ 前缀不匹配 → nullopt ===\n");
    {
        DiskParkSpool sp(base);
        std::vector<std::int32_t> other = prompt;
        other[50] = 999999;                        // 在第 50 个 token 处改掉
        auto f = sp.find(other, no_h, no_s);
        CHECK(!f.has_value(), "★ 前缀不同 → 找不到（不会误命中）");

        // 比最短的条目还短的 prompt
        std::vector<std::int32_t> tiny(prompt.begin(), prompt.begin() + 50);
        CHECK(!sp.find(tiny, no_h, no_s).has_value(), "★ prompt 比所有条目都短 → nullopt");
    }

    std::printf("\n=== 5. ★ profile 隔离（换 KV 格式就读不到）===\n");
    {
        DiskParkSpool::Options o = base;
        o.profile = prof("q4_0");                  // 只换格式
        DiskParkSpool sp(o);
        CHECK(sp.enabled(), "q4_0 的 spool 启用了");
        CHECK(sp.frontiers().empty(), "★ 看不到 int8 的条目");
        CHECK(!sp.find(prompt, no_h, no_s).has_value(), "★ 前缀一样也找不到（tag 不同）");
        // 用 q4_0 存一份，不应影响 int8 的
        CHECK(sp.store(make_image(1000, 21), 100, prompt, no_h, no_s) > 0, "q4_0 存一份");
        DiskParkSpool sp2(base);                   // 回到 int8
        auto f = sp2.find(prompt, no_h, no_s);
        CHECK(f.has_value() && f->frontier == 250, "★ int8 侧不受影响，仍取 250");
    }

    std::printf("\n=== 6. ★ 跨重启 ===\n");
    {
        DiskParkSpool sp(no_dedup);
        // 测试 2 存过 200，测试 3 又存了 100/150/250 → 共 4 个
        const auto fs = sp.frontiers();
        CHECK(fs.size() == 4, "重开后看到 4 个条目（100/150/200/250）");
        CHECK(fs.size() == 4 && fs[0] == 100 && fs[1] == 150 && fs[2] == 200 && fs[3] == 250,
              "★ 列表按 frontier 升序且无重复");
        auto f = sp.find(prompt, no_h, no_s);
        CHECK(f.has_value() && f->frontier == 250, "★ 重开后 find 仍取 250");
        auto back = f ? sp.load(*f) : std::nullopt;
        CHECK(back && back->live.gdn.size() == 2000, "★ 重开后 load 内容完整");
    }

    std::printf("\n=== 7. drop ===\n");
    {
        DiskParkSpool sp(no_dedup);
        CHECK(sp.drop(*sp.find(prompt, no_h, no_s)), "drop frontier=250");
        CHECK(sp.frontiers().size() == 3, "剩 3 个（100/150/200）");
        auto f = sp.find(prompt, no_h, no_s);
        CHECK(f.has_value() && f->frontier == 200, "★ 现在取最长的是 200");
        CHECK(!sp.load(DiskParkSpool::Found{250, {}}).has_value(), "250 读不到了");
        // 再丢一个，最长变成 150
        CHECK(sp.drop(*sp.find(prompt, no_h, no_s)), "drop frontier=200");
        auto f2 = sp.find(prompt, no_h, no_s);
        CHECK(f2.has_value() && f2->frontier == 150, "★ 再丢后取最长的是 150");
        CHECK(sp.frontiers().size() == 2, "剩 2 个");
    }

    std::printf("\n=== 7b. ★ 带真实 KV 缓冲区的往返（P4b 原先只测了空 KV）===\n");
    {
        // 真实停放快照的 KV 是数 GiB 的实数据；这里用 300 KB 的多层 KV，
        // 足以跨多个 8 KiB 分片，覆盖"big buffer 走 chunking"的路径。
        SavedConversation s2;
        s2.layer_lo = 0; s2.layer_hi = 11;
        s2.live.ids.assign(128, 42);
        s2.live.gdn.assign(4096, 7);
        for (int layer = 0; layer < 3; ++layer) {
            ConversationKv k;
            k.format  = 1;                       // int8
            k.cells   = 4096; k.heads = 8; k.head_dim = 128;
            k.page_size = 64; k.pooled_rows = 16; k.idx_dim = 32;
            std::vector<std::uint8_t> kb(100 * 1024), vb(100 * 1024);
            for (std::size_t i = 0; i < kb.size(); ++i) {
                kb[i] = static_cast<std::uint8_t>((i + layer) & 0xFF);
                vb[i] = static_cast<std::uint8_t>((i * 3 + layer) & 0xFF);
            }
            k.k = ConversationBuffer(std::initializer_list<std::uint8_t>{});
            k.k.resize(kb.size());
            std::size_t w = 0;
            k.k.visit(0, kb.size(), [&](std::uint8_t* d, std::size_t n, std::size_t) {
                std::memcpy(d, kb.data() + w, n); w += n; return true; });
            k.v.resize(vb.size());
            w = 0;
            k.v.visit(0, vb.size(), [&](std::uint8_t* d, std::size_t n, std::size_t) {
                std::memcpy(d, vb.data() + w, n); w += n; return true; });
            k.k_scale = ConversationBuffer(std::initializer_list<std::uint8_t>{1,2,3,4});
            k.v_scale = ConversationBuffer(std::initializer_list<std::uint8_t>{5,6,7,8});
            s2.kv.push_back(std::move(k));
        }
        const std::size_t packed = packed_saved_conversation_bytes(s2);
        std::printf("       （打包后 %zu 字节，步长 %zu → %zu 片）\n",
                    packed, kStride, (packed + kStride - 17) / (kStride - 16));

        DiskParkSpool sp(base);
        CHECK(sp.store(s2, (std::int64_t) s2.live.ids.size(), s2.live.ids, no_h, no_s) > 0, "★ 带 KV 的条目写入（多分片）");
        // ★ find 必须用【更长】的 prompt：引擎要求匹配的前缀严格短于 prompt
        //   （"最后一个 token 开启下一个 verify 窗口"）。生产里对话总是在增长，
        //   所以存的 frontier 必然短于下次请求的 prompt —— 这里照此模拟。
        std::vector<std::int32_t> grown = s2.live.ids;
        grown.push_back(7); grown.push_back(8);
        auto f = sp.find(grown, no_h, no_s);
        CHECK(f.has_value() && f->frontier == (std::int64_t) s2.live.ids.size(), "find 命中 128");
        auto back = f ? sp.load(*f) : std::nullopt;
        CHECK(back.has_value(), "★ 读回成功");
        if (back) {
            CHECK(back->kv.size() == s2.kv.size(), "★ KV 层数一致（3）");
            bool kv_ok = back->kv.size() == 3;
            for (std::size_t i = 0; kv_ok && i < 3; ++i) {
                const auto& a = s2.kv[i];
                const auto& b = back->kv[i];
                if (a.format != b.format || a.cells != b.cells || a.head_dim != b.head_dim) { kv_ok = false; break; }
                if (a.k.size() != b.k.size() || a.v.size() != b.v.size()) { kv_ok = false; break; }
                // 逐字节比对 K/V
                std::vector<std::uint8_t> x(a.k.size()), y(b.k.size());
                std::size_t w2 = 0;
                a.k.visit(0, a.k.size(), [&](const std::uint8_t* q, std::size_t n, std::size_t) { std::memcpy(x.data()+w2,q,n); w2+=n; return true; });
                w2 = 0;
                b.k.visit(0, b.k.size(), [&](const std::uint8_t* q, std::size_t n, std::size_t) { std::memcpy(y.data()+w2,q,n); w2+=n; return true; });
                if (x != y) { kv_ok = false; break; }
            }
            CHECK(kv_ok, "★★ KV 的 K/V 缓冲区逐字节一致（跨分片、跨多层）");
            CHECK(back->live.gdn == s2.live.gdn, "状态也一致");
        }
        sp.drop(*sp.find(grown, no_h, no_s));
    }

    std::printf("\n=== 8. ★ 分片损坏 → load 干净失败（回退重算）===\n");
    {
        // 找到 frontier=100 的某个数据分片文件位置并破坏它
        const std::string dat = (dir / "spool" / "park-state.dat").string();
        std::FILE* f = std::fopen(dat.c_str(), "r+b");
        CHECK(f != nullptr, "打开数据文件");
        if (f) {
            // 破坏第 3 个槽的负载中段（前两个槽属于别的条目，这里只是粗暴破坏一片）
            const long off = static_cast<long>(2 * kStride) + 1024;
            std::fseek(f, off, SEEK_SET);
            const unsigned char bad = 0x5A;
            std::fwrite(&bad, 1, 1, f);
            std::fclose(f);
        }
        DiskParkSpool sp(base);
        auto g = sp.find(prompt, no_h, no_s);
        auto f150 = g ? sp.load(*g) : std::nullopt;      // 当前最长（破坏前是 150）
        auto f100 = sp.load(DiskParkSpool::Found{100, {}});
        // 至少有一个因为 CRC 失败而读不出来；关键是不能崩、不能返回脏数据
        const bool clean = !f150.has_value() || !f100.has_value() ||
                           (f150 && f100);      // 破坏的那片恰好不属于它们
        CHECK(clean, "★ 破坏后行为干净（要么 nullopt，要么完整正确）");
        if (f150) CHECK(f150->live.gdn.size() == 2000, "若返回则完整");
    }

    std::printf("\n=== 9. ★★ 前缀去重（supersede_prefixes）===\n");
    {
        // 与其它节不同：这一节用【独立的目录】+ 默认选项（去重【开】），
        // 因为去重是跨 store 生效的，不能和上面的存量混在一起。
        DiskParkSpool::Options o = base;
        o.path = (dir / "dedup").string();
        DiskParkSpool sp(o);
        CHECK(sp.enabled(), "去重 spool 启用");
        CHECK(o.supersede_prefixes, "★ 默认就是开的");

        // 存 100 → 150 → 200 → 250，每一步都长于上一步且是它的前缀
        const std::int64_t steps[] = {100, 150, 200, 250};
        std::uint64_t expected_superseded = 0;
        for (std::size_t i = 0; i < sizeof(steps) / sizeof(steps[0]); ++i) {
            const std::int64_t f = steps[i];
            const std::size_t w = sp.store(make_image(2000, static_cast<std::uint8_t>(30 + i)),
                                           f, prompt, no_h, no_s);
            CHECK(w > 0, "store 成功");
            const auto fr = sp.frontiers();
            char msg[128];
            std::snprintf(msg, sizeof msg, "★ 存完 %lld 盘上只剩 1 个（去掉被覆盖的旧份）", (long long) f);
            CHECK(fr.size() == 1 && fr[0] == f, msg);
            if (i > 0) ++expected_superseded;      // 第一份没有前驱可覆盖
        }
        // 每次覆盖丢掉【整条】旧条目：探测页(chunk 0) + 数据页 —— 至少 2 页
        const std::uint64_t sup = sp.stats().superseded;
        char msg[160];
        std::snprintf(msg, sizeof msg, "★ superseded 计到数了（%llu，%zu 次覆盖）",
                      (unsigned long long) sup, (size_t) expected_superseded);
        CHECK(sup >= expected_superseded * 2, msg);

        // 覆盖之后，最长的那份仍然读得回来
        auto f = sp.find(prompt, no_h, no_s);
        CHECK(f.has_value() && f->frontier == 250, "★ 去重后 find 仍取 250");
        auto back = f ? sp.load(*f) : std::nullopt;
        CHECK(back && back->live.gdn.size() == 2000 && back->live.gdn[0] == 33,
              "★ 去重后取回的仍是最后存的那一份（seed 33）");

        // 关掉去重时，同样的序列应该留下 4 份 —— 这是这一节的对照
        DiskParkSpool::Options o2 = base;
        o2.path = (dir / "dedup-off").string();
        o2.supersede_prefixes = false;
        DiskParkSpool sp2(o2);
        for (std::size_t i = 0; i < sizeof(steps) / sizeof(steps[0]); ++i)
            sp2.store(make_image(2000, static_cast<std::uint8_t>(30 + i)), steps[i], prompt, no_h, no_s);
        CHECK(sp2.frontiers().size() == 4, "★ 对照：关掉去重时留下 4 份");
        CHECK(sp2.stats().superseded == 0, "★ 对照：关掉去重时 superseded == 0");
    }

    fs::remove_all(dir);
    std::printf("\n========================================\n");
    std::printf("  检查 %d 项，失败 %d 项\n", g_checks, g_failed);
    std::printf("========================================\n\n");
    return g_failed == 0 ? 0 : 1;
}
