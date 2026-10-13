// DiskParkLogs 的单元测试：把「摘要 → 会话日志」这张表管对。
//
// 它要回答的核心问题是【追加流最容易错的那件事】：
//   同一条对话连续几轮，到底有没有落在【同一个文件】上、并且【从上一轮的水位续写】？
//   摘要每轮都在变（它描述的是"前 N 个 token"），所以这里必须靠一张表把
//   "上一轮的摘要" 映射到 "同一个目录" —— 这张表错了，盘上就会堆出 N 份互为前缀的快照，
//   而那正是整套改造要消灭的东西。

#include "strata/core/disk_park_digest.hpp"
#include "strata/core/disk_park_logs.hpp"

#include <algorithm>
#include <cstdio>
#include <filesystem>
#include <string>
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

// 与 disk_park_session_test 同一套几何：K 4 B/cell、V 2、两个 scale 1、pooled 每 4 cell 一行 8 B
constexpr std::uint64_t kCap = 4096;
DiskParkLogs::Config make_cfg(const std::string& root, std::size_t max_tracked = 4096) {
    DiskParkLogs::Config c;
    c.root = root;
    c.tag = 0xABCD1234u;
    c.kv_format = 3; c.heads = 2; c.head_dim = 256; c.page_size = 4; c.idx_dim = 8;
    c.idx_block = 4; c.capacity_cells = kCap;
    c.part_bytes_at_cap[kK]  = 4 * kCap;
    c.part_bytes_at_cap[kV]  = 2 * kCap;
    c.part_bytes_at_cap[kKS] = 1 * kCap;
    c.part_bytes_at_cap[kVS] = 1 * kCap;
    c.part_bytes_at_cap[kP]  = (kCap / 4) * 8;
    c.max_open = 4;
    c.max_tracked = max_tracked;
    return c;
}

// 真值：只取决于 (part, 全局 cell, 段内字节) —— 与轮次无关
std::uint8_t truth_at(std::size_t part, std::uint64_t cell, std::size_t byte) {
    return static_cast<std::uint8_t>((part * 131 + cell * 7 + byte * 3 + 1) & 0xFF);
}

DiskParkFill truth_fill(const DiskParkSessionPlan& plan) {
    return [&plan](std::size_t part, std::size_t offset, std::uint8_t* dst, std::size_t n) {
        const auto& p = plan.parts[part];
        std::size_t i = 0;
        while (i < n) {
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

// 用真实的摘要构造器算"前 n 个 token 的摘要" —— 与引擎里那条路径同一套算法
DiskParkLogDigest digest_of(const std::vector<std::int32_t>& ids, std::size_t n) {
    PrefixDigestBuilder b;
    const std::vector<std::uint64_t> no_h;
    const std::vector<std::int64_t> no_s;
    for (std::size_t i = 0; i < n; ++i) b.advance(ids[i], no_h, no_s);
    const PrefixDigest d = b.value();
    return {d.lo, d.hi};
}

std::size_t count_session_dirs(const std::string& root, std::uint32_t tag) {
    char tagbuf[16];
    std::snprintf(tagbuf, sizeof tagbuf, "%08X", tag);
    const std::filesystem::path p = std::filesystem::path(root) / ("tag-" + std::string(tagbuf));
    if (!std::filesystem::exists(p)) return 0;
    std::size_t n = 0;
    for (const auto& e : std::filesystem::directory_iterator(p))
        if (e.is_directory()) ++n;
    return n;
}

} // namespace

int main() {
    namespace fs = std::filesystem;
    const fs::path root = fs::temp_directory_path() / "disk_park_logs_test";
    fs::remove_all(root);
    fs::create_directories(root);

    std::vector<std::int32_t> ids(4096);
    for (std::size_t i = 0; i < ids.size(); ++i) ids[i] = 1000 + (std::int32_t) i;

    std::printf("\n=== 1. 关闭时零副作用（root 为空）===\n");
    {
        DiskParkLogs::Config c = make_cfg("");       // 空 = 关闭
        DiskParkLogs logs(c);
        CHECK(!logs.enabled(), "未启用");
        CHECK(logs.append({}, truth_fill(logs.plan())) == 0, "append 返回 0");
        CHECK(logs.watermark({}) == 0, "watermark 为 0");
        CHECK(logs.lookup({}).empty(), "lookup 为空");
    }

    std::printf("\n=== 2. 几何不自洽 → 整层关掉（宁可没有，也不按错布局写盘）===\n");
    {
        DiskParkLogs::Config c = make_cfg((root / "bad").string());
        c.part_bytes_at_cap[kK] = 4 * kCap + 1;      // 不整除
        DiskParkLogs logs(c);
        CHECK(!logs.enabled(), "★ K 的字节数不整除 → 关闭");
        DiskParkLogs::Config c2 = make_cfg((root / "bad2").string());
        c2.tag = 0;
        DiskParkLogs logs2(c2);
        CHECK(!logs2.enabled(), "tag 为 0 → 关闭");
    }

    const std::string rt = (root / "rt").string();
    const DiskParkLogs::Config cfg = make_cfg(rt);
    // ★ 注意 root 必须与 rt2 一致 —— 第一版写成 rt 了，于是"两个目录"的断言看着像失败，
    //   其实是两个配置指向了同一个目录（测试自己的错）
    const DiskParkLogs::Config cfg_small = make_cfg((root / "small").string(), /*max_tracked=*/2);

    std::printf("\n=== 3. ★★ 同一条对话三轮：必须落在同一个文件、从水位续写 ===\n");
    {
        DiskParkLogs logs(cfg);
        CHECK(logs.enabled(), "启用");
        const DiskParkFill fill = truth_fill(logs.plan());

        // 第 1 轮：1000 token（无前驱）
        DiskParkLogs::Request r1;
        r1.has_prev = false;
        r1.next = digest_of(ids, 1000);
        r1.to_tokens = 1000;
        const std::uint64_t w1 = logs.append(r1, fill);
        CHECK(w1 > 0, "第 1 轮写入成功");
        CHECK(logs.stats().fresh == 1, "记为新会话");
        const std::string dir1 = logs.lookup(r1.next);
        CHECK(!dir1.empty(), "第 1 轮之后记住了它的摘要");
        CHECK(logs.watermark(r1.next) == 1000, "水位 = 1000");

        // 第 2 轮：接在 1000 之后到 1600（prev = 第 1 轮的 next）
        DiskParkLogs::Request r2;
        r2.has_prev = true;
        r2.prev = r1.next;
        r2.next = digest_of(ids, 1600);
        r2.to_tokens = 1600;
        const std::uint64_t w2 = logs.append(r2, fill);
        CHECK(w2 > 0, "第 2 轮写入成功");
        CHECK(logs.stats().adopted == 1, "★★ 第 2 轮认领了上一轮的会话");
        CHECK(logs.lookup(r2.next) == dir1, "★★ 第 2 轮落在【同一个目录】");
        CHECK(logs.watermark(r2.next) == 1600, "水位推进到 1600");

        // 第 3 轮：到 2400
        DiskParkLogs::Request r3;
        r3.has_prev = true;
        r3.prev = r2.next;
        r3.next = digest_of(ids, 2400);
        r3.to_tokens = 2400;
        CHECK(logs.append(r3, fill) > 0, "第 3 轮写入成功");
        CHECK(logs.lookup(r3.next) == dir1, "★★ 第 3 轮还是同一个目录");
        CHECK(count_session_dirs(rt, cfg.tag) == 1,
              "★★★ 盘上只有 1 个会话目录（不是 3 份互为前缀的快照）");

        // 三轮的字节必须都在、且逐字节正确
        DiskParkSession s;
        // 不传 resume：没有 sidecar ⇒ 水位是 0（这一层只信提交点）
        CHECK(DiskParkSession::open(s, logs.dir_for(dir1), logs.plan()), "重开这个会话");
        CHECK(s.log_cells() == 0, "★ 没有 sidecar 且不给 resume ⇒ 水位 0（只信提交点）");
        // 传 resume：持有者的记忆就是水位
        DiskParkSession s2;
        CHECK(DiskParkSession::open(s2, logs.dir_for(dir1), logs.plan(), 2400), "带 resume 重开");
        CHECK(s2.log_cells() == 2400, "★ 给了 resume ⇒ 水位就是它（记忆兜住了）");
        for (std::size_t p = 0; p < 5; ++p) {
            const auto& pp = logs.plan().parts[p];
            if (pp.bytes_per_cell == 0) continue;
            const std::uint64_t n = s.part_bytes(static_cast<DiskParkSession::Part>(p), 2400);
            std::vector<std::uint8_t> got(static_cast<std::size_t>(n), 0);
            CHECK(s.read_kv(static_cast<DiskParkSession::Part>(p), 0, 2400, got.data(), got.size()),
                  "读整段");
            bool ok = true;
            for (std::size_t i = 0; i < got.size() && ok; ++i) {
                std::uint64_t cell; std::size_t byte;
                if (pp.row_cells <= 1) { cell = i / pp.bytes_per_cell; byte = i % pp.bytes_per_cell; }
                else {
                    const std::uint64_t row = i / pp.row_bytes;
                    const std::size_t within = i % pp.row_bytes;
                    cell = row * pp.row_cells + within / pp.bytes_per_cell;
                    byte = within % pp.bytes_per_cell;
                }
                if (got[i] != truth_at(p, cell, byte)) ok = false;
            }
            CHECK(ok, "★ 三轮拼起来逐字节正确（第 2、3 轮没破坏第 1 轮）");
        }
    }

    std::printf("\n=== 4. ★ 与上一轮无关的另一条对话 → 另一个目录 ===\n");
    {
        DiskParkLogs logs(cfg);
        const DiskParkFill fill = truth_fill(logs.plan());
        std::vector<std::int32_t> other = ids;
        other[500] = 999999;                          // 第 500 个 token 不同 = 另一条对话
        DiskParkLogs::Request r;
        r.has_prev = false;
        r.next = digest_of(other, 1200);
        r.to_tokens = 1200;
        CHECK(logs.append(r, fill) > 0, "写入成功");
        CHECK(count_session_dirs(rt, cfg.tag) == 2, "★ 现在是 2 个目录（互不覆盖）");
    }

    std::printf("\n=== 5. ★ 前驱认不出来（客户端改写了历史）→ 当新会话，不猜水位 ===\n");
    {
        DiskParkLogs logs(cfg);
        const DiskParkFill fill = truth_fill(logs.plan());
        std::vector<std::int32_t> edited = ids;
        edited[700] = 424242;
        DiskParkLogs::Request r;
        r.has_prev = true;
        r.prev = digest_of(edited, 900);              // 这个摘要从没写过
        r.next = digest_of(edited, 1500);
        r.to_tokens = 1500;
        CHECK(logs.append(r, fill) > 0, "写入成功");
        CHECK(logs.stats().fresh == 1 && logs.stats().adopted == 0, "★ 记为新会话");
        CHECK(logs.stats().writes == 1, "只写了 1 次");
        // 新会话的水位就是 1500（从 0 写起），不是 600
        CHECK(logs.watermark(r.next) == 1500, "水位 = 1500（从头写的）");
    }

    std::printf("\n=== 6. ★ 记忆有上限（max_tracked=2）时的行为 ===\n");
    {
        const std::string rt2 = (root / "small").string();
        DiskParkLogs logs(cfg_small);                 // max_tracked = 2
        const DiskParkFill fill = truth_fill(logs.plan());
        DiskParkLogDigest prev{};
        bool have_prev = false;
        std::uint64_t last = 0;
        for (int i = 1; i <= 5; ++i) {
            DiskParkLogs::Request r;
            r.has_prev = have_prev;
            r.prev = prev;
            r.to_tokens = 500ull * (std::uint64_t) i;
            r.next = digest_of(ids, static_cast<std::size_t>(r.to_tokens));
            CHECK(logs.append(r, fill) > 0, "写入成功");
            prev = r.next; have_prev = true;
            last = r.to_tokens;
        }
        CHECK(logs.watermark(prev) == last, "★ 最新那条的水位仍在");
        // 记忆只有 2 格 ⇒ 中间那些摘要会被挤掉；但每一轮用到的 prev 都会被 touch，
        // 所以整条【活跃链】始终落在同一个目录里（这正是淘汰规则必须是 LRU 的原因）
        CHECK(count_session_dirs(rt2, cfg.tag) == 1,
              "★★ 链式轮次即便记忆很小，也始终只有 1 个目录（prev 每轮都被 touch）");

        // ★ 记忆被淘汰之后：拿一个【很久以前】的摘要来续写，会当新会话（多写一份）
        //   —— 这正是设计里说的"宁可多写，不可写错"
        DiskParkLogs::Request stale;
        stale.has_prev = true;
        stale.prev = digest_of(ids, 500);             // 早已被挤掉
        stale.next = digest_of(ids, 900);
        stale.to_tokens = 900;
        CHECK(logs.append(stale, fill) > 0, "写入成功");
        CHECK(logs.stats().fresh >= 2, "★ 记为新会话（宁可多写一份）");
        CHECK(count_session_dirs(rt2, cfg.tag) == 2, "★ 于是多了一个目录 —— 这是有意的降级");
    }

    std::printf("\n=== 7. ★ 水位不倒退：to_tokens <= watermark 时什么都不写 ===\n");
    {
        DiskParkLogs logs(cfg);
        const DiskParkFill fill = truth_fill(logs.plan());
        DiskParkLogs::Request r;
        r.has_prev = false;
        r.next = digest_of(ids, 2000);
        r.to_tokens = 2000;
        CHECK(logs.append(r, fill) > 0, "先写到 2000");
        const std::uint64_t writes = logs.stats().writes;

        DiskParkLogs::Request back;
        back.has_prev = true;
        back.prev = r.next;
        back.next = digest_of(ids, 2000);
        back.to_tokens = 1500;                        // 倒退
        CHECK(logs.append(back, fill) == 0, "★ 倒退的请求不写");
        CHECK(logs.stats().writes == writes, "写次数没变");
    }

    std::printf("\n=== 8. ★★★ 逐层追加（引擎的真实形态：每层一个 host 池）===\n");
    {
        // 引擎的 K/V 是【每层一份】（layer.hpp: 每个 QsaState 有自己的 host 池），
        // 而"位置寻址 + 只追加"对【每一层各自】成立 ⇒ 每层一个 session。
        const std::string rt2 = (root / "layered").string();
        DiskParkLogs logs{make_cfg(rt2)};
        CHECK(logs.enabled(), "逐层日志打开");

        const std::size_t kLayers = 4;                 // 测试用 4 层（引擎是 12）
        // 真值只取决于 (层, part, cell, byte) —— 与"哪一轮写的"无关（核心不变量）
        auto truth_l = [](std::size_t L, std::size_t part, std::uint64_t cell, std::size_t byte) {
            return static_cast<std::uint8_t>((L * 131 + part * 37 + cell * 7 + byte * 3 + 1) & 0xFF);
        };
        const DiskParkLogs::DiskParkLayerFill fill =
            [&truth_l](std::size_t L, std::size_t part, std::size_t off,
                       std::uint8_t* dst, std::size_t n) {
                const std::size_t bpc[5] = {4, 2, 1, 1, 2};
                for (std::size_t i = 0; i < n; ++i) {
                    const std::size_t at = off + i;
                    dst[i] = truth_l(L, part, at / bpc[part], at % bpc[part]);
                }
                return true;
            };

        // ① 第一轮：0 → 120
        DiskParkLogs::Request r1;
        r1.has_prev = false;
        r1.next = {0x1111, 0x2222};
        r1.to_tokens = 120;
        std::string dir1;
        r1.out_dir = &dir1;
        const std::uint64_t w1 = logs.append_layers(r1, kLayers, fill);
        CHECK(w1 > 0, "第一轮逐层追加成功");
        CHECK(!dir1.empty(), "★ 拿到了会话目录名");
        std::printf("       （第一轮写了 %llu 字节 = %zu 层 × 4 个 part）\n",
                    (unsigned long long) w1, kLayers);

        // ② 第二轮：120 → 200（只写新增的 80 个 cell）
        DiskParkLogs::Request r2;
        r2.has_prev = true;
        r2.prev = r1.next;
        r2.next = {0x3333, 0x4444};
        r2.to_tokens = 200;
        std::string dir2;
        r2.out_dir = &dir2;
        const std::uint64_t w2 = logs.append_layers(r2, kLayers, fill);
        CHECK(w2 > 0, "第二轮逐层追加成功");
        CHECK(dir2 == dir1, "★★ 续写同一个会话目录（没有另开一份）");

        // ③ 逐层取回 [0,200)，与真值逐字节比对
        const std::size_t bpc[5] = {4, 2, 1, 1, 2};
        std::size_t checked = 0, bad = 0;
        std::size_t parts_seen = 0;
        const bool ok = logs.read_layers(dir1, kLayers, 0, 200,
            [&](std::size_t L, std::size_t part, std::size_t off,
                const std::uint8_t* src, std::size_t n) {
                ++parts_seen;
                for (std::size_t i = 0; i < n; ++i) {
                    const std::size_t at = off + i;
                    const std::uint8_t want = truth_l(L, part, at / bpc[part], at % bpc[part]);
                    ++checked;
                    if (src[i] != want) ++bad;
                }
                return true;
            });
        CHECK(ok, "逐层取回成功");
        std::printf("       （取回 %zu 字节，跨 %zu 个 (层,part)）\n", checked, parts_seen);
        CHECK(bad == 0, "★★★ 逐层取回的字节全部与真值相同（两轮追加后）");
        CHECK(checked > 0, "确实取到了数据");

        // ④ ★ 每层确实各有自己的目录（不是挤在一个文件里）
        std::size_t dirs = 0;
        for (std::size_t L = 0; L < kLayers; ++L) {
            const std::string lp = logs.dir_for_layer(dir1, L);
            if (fs::exists(fs::path(lp) / "kv.bin")) ++dirs;
        }
        CHECK(dirs == kLayers, "★ 每层各有一个 kv.bin（层数 = 目录数）");

        // ⑤ 水位：两轮之后应当记到 200
        CHECK(logs.watermark(r2.next) == 200, "第二轮后的水位 = 200");
        // ★★ 会话目录名 = 该会话【第一次建立时】的 next 的 hex，之后永不改变。
        //   （所以"第二轮写完后 name_for(r2.next) 仍等于 dir1"是【错的期望】——
        //     name_for 只是 hex 转换，不查表。见头文件里的澄清。）
        CHECK(logs.name_for(r1.next) == dir1, "★ name_for(第一次的 next) == append 用的目录名");
        // 要拿"某个摘要属于哪个会话"，正确途径是 lookup()（内存表）或 out_dir 出参
        CHECK(logs.lookup(r1.next) == dir1, "★ lookup(第一轮的 next) 指向同一个会话");
        CHECK(logs.lookup(r2.next) == dir1, "★★ lookup 顺着链：第二轮的 next 也在同一个会话里");
    }

    std::printf("\n=== 7. ★★ 重启后只补增量（水位以会话自己的 state.bin 为准）===\n");
    {
        const std::string rtr = (root / "restart").string();
        const DiskParkLogs::Config rc = make_cfg(rtr);
        const DiskParkLogDigest D1{0xAAAA, 0xBBBB};
        const DiskParkLogDigest Dother{0xCCCC, 0xDDDD};     // 内存里【查不到】的摘要
        const std::size_t kL = 4;
        std::uint64_t w1 = 0, w2 = 0;
        // append_layers 要的是【逐层】回调（layer, part, offset, dst, n）；
        // 测试里各层用同一套真值（与既有分层测试一致）。
        // ★ 必须【按值】接住 truth_fill 返回的临时对象，再【按值】捕获 ——
        //   第一版按引用捕获了一个临时 DiskParkFill ⇒ 悬垂引用 ⇒ 段错误（实测踩到）。
        const auto layer_fill = [](DiskParkFill plain) {
            return DiskParkLogs::DiskParkLayerFill(
                [plain](std::size_t, std::size_t part, std::size_t off, std::uint8_t* dst,
                        std::size_t n) { return plain(part, off, dst, n); });
        };
        {
            DiskParkLogs logsA(rc);
            const DiskParkLogs::DiskParkLayerFill fillA = layer_fill(truth_fill(logsA.plan()));
            DiskParkLogs::Request r{};
            r.has_prev = false;
            r.next = D1;
            r.to_tokens = 120;
            std::string dir;
            r.out_dir = &dir;
            w1 = logsA.append_layers(r, kL, fillA);
            logsA.commit_watermark(D1, 120);                // 水位落进 state.bin
            CHECK(w1 > 0 && !dir.empty(), "第 1 轮：写 120 个 cell 并提交水位");
        }
        {
            // ★ 模拟重启：新的 DiskParkLogs ⇒ tracked_ 是空的 ⇒ watermark(prev) = 0
            DiskParkLogs logsB(rc);
            const DiskParkLogs::DiskParkLayerFill fillB = layer_fill(truth_fill(logsB.plan()));
            DiskParkLogs::Request r{};
            r.has_prev = true;
            r.prev = Dother;
            r.next = D1;                                    // 目录名因此还是原来那个
            r.to_tokens = 200;
            std::string dir;
            r.out_dir = &dir;
            w2 = logsB.append_layers(r, kL, fillB);
            CHECK(w2 > 0, "重启后第 2 轮：追加成功");
            CHECK(w2 * 3 == w1 * 2,
                  "★★ 只补了增量（80 个 cell），没有把 200 个 cell 重写一遍（修之前这里是 5:3）");
            const std::size_t bpc[5] = {4, 2, 1, 1, 2};
            std::size_t bad = 0;
            const bool got = logsB.read_layers(dir, kL, 120, 200,
                [&](std::size_t L, std::size_t part, std::size_t off,
                    const std::uint8_t* src, std::size_t n) {
                    for (std::size_t i = 0; i < n; ++i) {
                        // ★ read_layers 回调的 offset 是【相对 from_cell(=120)】的
                        const std::size_t at = 120 * bpc[part] + off + i;
                        if (src[i] != truth_at(part, at / bpc[part], at % bpc[part])) ++bad;
                    }
                    return true;
                });
            CHECK(got, "读回 [120,200)");
            CHECK(bad == 0, "★★ 增量写的字节与真值逐字节一致");
        }
    }

    // ==================================================================
    // ★★★ 8. P15 模拟：内存续接（复用点是【检查点长度】，不是上一轮的 L）
    //   第 1 轮写到 1000。第 2 轮客户端带回来的前缀只复用 900（引擎挂在 900 这个
    //   检查点上），而 next=1600 的 hex 不是任何已存在的目录 ⇒ 摘要链必然 miss。
    //     A（不报别名，= 修复前）→ 新目录 + 全量 1600 个 cell
    //     B（报别名 digest(900)，= P15）→ 同一个目录 + 只补 [900,1600) = 700 个 cell
    //   ★ 判据用【相对】写量（第 2 轮 / 第 1 轮），不依赖每 cell 多少字节：
    //     整份重写 ⇒ 1600/1000 = 1.6×；只补增量 ⇒ 700/1000 = 0.7×。
    // ==================================================================
    std::printf("\n=== 8. ★★★ P15 模拟：内存续接只补增量（别名 frontier）===\n");
    {
        const std::size_t kL = 4;
        const auto layer_fill = [](DiskParkFill plain) {
            return DiskParkLogs::DiskParkLayerFill(
                [plain](std::size_t, std::size_t part, std::size_t off, std::uint8_t* dst,
                        std::size_t n) { return plain(part, off, dst, n); });
        };
        std::uint64_t a1 = 0, a2 = 0, b1 = 0, b2 = 0;
        std::size_t nA = 0, nB = 0;
        // ---- A：不报别名（修复前的行为）----
        {
            const std::string ra = (root / "p15-a").string();
            const DiskParkLogs::Config ca = make_cfg(ra);
            DiskParkLogs logs(ca);
            const DiskParkLogs::DiskParkLayerFill fill = layer_fill(truth_fill(logs.plan()));
            DiskParkLogs::Request r1;
            r1.has_prev = false; r1.next = digest_of(ids, 1000); r1.to_tokens = 1000;
            std::string d1; r1.out_dir = &d1;
            a1 = logs.append_layers(r1, kL, fill);
            DiskParkLogs::Request r2;
            r2.has_prev = true; r2.prev = digest_of(ids, 900);
            r2.next = digest_of(ids, 1600); r2.to_tokens = 1600;
            std::string d2; r2.out_dir = &d2;
            a2 = logs.append_layers(r2, kL, fill);
            CHECK(d2 != d1, "★ A 第 2 轮开了一个【新目录】（摘要链 miss）");
            CHECK(logs.stats().fresh == 2, "★ A 记了两次 fresh");
            nA = count_session_dirs(ra, ca.tag);
        }
        // ---- B：报别名 digest(900) = P15 ----
        {
            const std::string rb = (root / "p15-b").string();
            const DiskParkLogs::Config cb = make_cfg(rb);
            DiskParkLogs logs(cb);
            const DiskParkLogs::DiskParkLayerFill fill = layer_fill(truth_fill(logs.plan()));
            DiskParkLogs::Request r1;
            r1.has_prev = false; r1.next = digest_of(ids, 1000); r1.to_tokens = 1000;
            r1.aliases.push_back({digest_of(ids, 900), 900});      // ★ 本轮报出检查点长度
            std::string d1; r1.out_dir = &d1;
            b1 = logs.append_layers(r1, kL, fill);
            const std::string dir1 = d1;
            DiskParkLogs::Request r2;
            r2.has_prev = true; r2.prev = digest_of(ids, 900);
            r2.next = digest_of(ids, 1600); r2.to_tokens = 1600;
            std::string d2; r2.out_dir = &d2;
            b2 = logs.append_layers(r2, kL, fill);
            CHECK(d2 == dir1, "★★★ B 第 2 轮落在同一个目录（摘要链 miss 被别名救回）");
            CHECK(logs.stats().adopted == 1, "★★★ B 计一次 adopted（不是 fresh）");
            CHECK(logs.stats().fresh == 1, "★★ B 只开了 1 个目录");
            CHECK(logs.watermark(r2.next) == 1600, "水位推进到 1600");
            nB = count_session_dirs(rb, cb.tag);
            // ★ 逐字节验证：两轮拼起来必须与真值一致（增量写没把前面那段写坏）。
            //   多层会话在 dir/layerNN/ 下 ⇒ 用 read_layers，不能用 DiskParkSession::open(顶层)。
            const std::size_t bpc[5] = {4, 2, 1, 1, 2};
            std::size_t bad = 0;
            const bool got = logs.read_layers(dir1, kL, 0, 1600,
                [&](std::size_t, std::size_t part, std::size_t off,
                    const std::uint8_t* src, std::size_t n) {
                    for (std::size_t i = 0; i < n; ++i) {
                        const std::size_t at = off + i;      // from_cell = 0 ⇒ 段内偏移即全局偏移
                        if (src[i] != truth_at(part, at / bpc[part], at % bpc[part])) ++bad;
                    }
                    return true;
                });
            CHECK(got, "★★ B 读回 [0,1600)");
            CHECK(bad == 0, "★★★ B 逐字节正确：增量写接在前一轮后面（900+700=1600）");
        }
        CHECK(a1 == b1, "A/B 第 1 轮写量相同（同一场景）");
        CHECK(a2 == a1 * 16 / 10,
              "★★★ A（无别名）第 2 轮 = 1.6 × 第 1 轮 ⇒ 整份 1600 个 cell 重写 = 修复前");
        CHECK(b2 == b1 * 7 / 10,
              "★★★ B（P15）第 2 轮 = 0.7 × 第 1 轮 ⇒ 只补了 700 个 cell");
        CHECK(nA == 2, "★★★ A 盘上堆出 2 个互为前缀的目录");
        CHECK(nB == 1, "★★★ B 始终只有 1 个目录");
        CHECK(a2 > b2 * 2, "★★★ 同一场景：无别名写的量是有别名的 2 倍以上");
    }

    // ==================================================================
    // ★★★ 9. P18 模拟：逐层路径的会话目录淘汰（fd 无上限增长那个坑）
    //   逐层下一个会话目录 = 12 个 open_ 条目 = 12+ 个 fd，而淘汰只写在单层路径里
    //   ⇒ 实测 32 个目录 = 416 个 kv.bin fd，逼近 systemd 的 1024 软上限。
    //   这里把 max_open 压到 2，开 4 条互不相同的对话，断言：
    //     ① 打开着的【目录】数始终 ≤ 2（而不是 12 × 条数）
    //     ② 真的关掉过（closes > 0）
    //     ③ 被淘汰的目录还能重开续写，且逐字节正确（淘汰不能写坏数据）
    // ==================================================================
    std::printf("\n=== 9. ★★★ P18 模拟：逐层路径的会话目录淘汰 ===\n");
    {
        const std::size_t kL = 4;
        const std::string rp = (root / "p18").string();
        DiskParkLogs::Config c18 = make_cfg(rp);
        c18.max_open = 2;                       // 只允许同时开 2 个会话目录
        DiskParkLogs logs(c18);
        CHECK(logs.enabled(), "启用");
        const auto layer_fill = [](DiskParkFill plain) {
            return DiskParkLogs::DiskParkLayerFill(
                [plain](std::size_t, std::size_t part, std::size_t off, std::uint8_t* dst,
                        std::size_t n) { return plain(part, off, dst, n); });
        };
        const DiskParkLogs::DiskParkLayerFill fill = layer_fill(truth_fill(logs.plan()));
        std::string first_dir;
        for (int k = 0; k < 4; ++k) {
            std::vector<std::int32_t> other = ids;
            other[(std::size_t) (10 + k)] = 900000 + k;          // 4 条互不相同的对话
            const std::int64_t tk = 300 + 10 * (std::int64_t) k;
            DiskParkLogs::Request r;
            r.has_prev = false;
            r.next = digest_of(other, (std::size_t) tk);
            r.to_tokens = (std::uint64_t) tk;
            std::string d; r.out_dir = &d;
            CHECK(logs.append_layers(r, kL, fill) > 0, "写一条新对话");
            if (k == 0) first_dir = d;
            CHECK(logs.open_dirs() <= 2, "★★★ 打开着的会话目录 ≤ max_open(2)");
        }
        std::printf("  → open_dirs=%zu open_files=%zu closes=%llu\n", logs.open_dirs(),
                    logs.open_files(), (unsigned long long) logs.stats().closes);
        CHECK(logs.stats().closes > 0, "★★★ 真的关掉过会话（closes > 0）");
        // ★ 每个目录是 kL 层 + 1 条【用裸目录名】开的（append_layers 的 read_meta
        //   用它读 state.bin 的水位）⇒ 生产上实测正是 13 fd/目录（12 层 + 1）。
        CHECK(logs.open_files() <= 2 * (kL + 1), "★★★ 打开的文件数 ≤ 2 个目录 × (4 层 + 1)");
        // ③ 第 1 条对话的第 2 轮：它的 session 已被淘汰 ⇒ 必须能重开并续写
        std::vector<std::int32_t> c0 = ids;
        c0[10] = 900000;
        DiskParkLogs::Request r2;
        r2.has_prev = true;
        r2.prev = digest_of(c0, 300);
        r2.next = digest_of(c0, 360);
        r2.to_tokens = 360;
        std::string d2; r2.out_dir = &d2;
        CHECK(logs.append_layers(r2, kL, fill) > 0, "★ 被淘汰的目录还能续写（重开成功）");
        CHECK(d2 == first_dir, "★★ 续写落回同一个目录");
        const std::size_t bpc[5] = {4, 2, 1, 1, 2};
        std::size_t bad = 0;
        const bool got = logs.read_layers(first_dir, kL, 0, 360,
            [&](std::size_t, std::size_t part, std::size_t off,
                const std::uint8_t* src, std::size_t n) {
                for (std::size_t i = 0; i < n; ++i) {
                    const std::size_t at = off + i;
                    if (src[i] != truth_at(part, at / bpc[part], at % bpc[part])) ++bad;
                }
                return true;
            });
        CHECK(got, "读回被淘汰又重开的目录");
        CHECK(bad == 0, "★★★ 逐字节正确（淘汰+重开没有写坏数据）");
    }

    fs::remove_all(root);
    std::printf("\n========================================\n");
    std::printf("  检查 %d 项，失败 %d 项\n", g_checks, g_failed);
    std::printf("========================================\n\n");
    return g_failed == 0 ? 0 : 1;
}
