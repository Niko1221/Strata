// P4c 集成测试：真实 ConversationCache（打过补丁）+ 真实 DiskParkSpool。
//
// 验证的核心流程就是"先腾后取"：
//   缓存满了 → 淘汰的条目交给 spool 写盘 → 新请求命中磁盘上的旧前缀
//   → 先把常驻的全部腾出去（写盘）→ 再读回目标那一份 → 放回缓存
//
// ★ 用影子头文件的方式测试：p4/include 在 -I 里排在 ~/Strata/include 之前，
//   所以这里拿到的是打过补丁的 conversation_cache.hpp，而引擎源码树没被动过。

#include "strata/core/conversation_cache.hpp"
#include "strata/core/disk_park_spool.hpp"

#include <cstdio>
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

static DiskParkProfile prof() {
    DiskParkProfile p; p.model = "Qwen3.8-Flash-Next"; p.kv_format = "int8";
    p.max_context = 262144; p.rope = "none"; return p;
}

// 造一个"对话在 frontier 处的镜像"
static SavedConversation snap(const std::vector<std::int32_t>& ids, std::uint8_t seed) {
    SavedConversation s;
    s.layer_lo = 1; s.layer_hi = 11;
    s.live.ids = ids;
    s.live.gdn.assign(4096, seed);
    ConversationCheckpoint c;
    c.ids = ids;
    c.gdn.assign(1024, seed);
    s.checkpoints.push_back(std::move(c));
    return s;
}

int main() {
    namespace fs = std::filesystem;
    const fs::path dir = fs::temp_directory_path() / "disk_park_integration";
    fs::remove_all(dir); fs::create_directories(dir);

    DiskParkSpool::Options so;
    so.path = (dir / "spool").string();
    so.profile = prof();
    so.state_stride = 8192;
    so.state_capacity_bytes = 8192 * 8192;
    so.kv_stride = 0;                       // 本测试只用 StateImage 家族

    // 每条 8 KB gdn + 1 KB checkpoint ≈ 5 KB 打包后（gdn 压缩不了，就按原始算）
    const std::vector<std::int32_t> pA = {1,2,3,4,5,6,7,8,9,10};
    const std::vector<std::int32_t> pB = {1,2,3,4,5,6,7,8,9,11};
    const std::vector<std::int32_t> pCommon = {1,2,3,4,5};

    std::printf("\n=== 1. 关闭 spool 时：行为与改动前一致 ===\n");
    {
        ConversationCache cache(1 << 20, 2);          // 1 MiB, 2 槽
        CHECK(cache.enabled(), "缓存启用");
        std::vector<SavedConversation> ev;
        CHECK(cache.make_room(0, 0, &ev), "make_room 带出参仍返回 true");
        CHECK(ev.empty(), "空缓存淘汰 0 个");
        CHECK(cache.put(snap(pA, 1)), "放入 A");
        CHECK(cache.put(snap(pB, 2)), "放入 B");
        CHECK(cache.size() == 2, "2 个常驻");
        // 放第三个 → 淘汰最老的（A），并交出来
        std::vector<SavedConversation> out;
        CHECK(cache.make_room(0, 0, &out), "make_room");
        CHECK(out.size() == 1, "★ 淘汰的条目被交出来（而不是丢弃）");
        CHECK(!out.empty() && out[0].live.ids == pA, "★ 交出来的是最老的 A");
        CHECK(cache.size() == 1, "缓存剩 1 个");
    }

    std::printf("\n=== 2. ★ 完整流程：淘汰 → 写盘 → 新请求命中磁盘 → 先腾后取 ===\n");
    {
        DiskParkSpool spool(so);
        CHECK(spool.enabled(), "spool 启用");

        ConversationCache cache(1 << 20, 2);
        // 两个不同的对话
        CHECK(cache.put(snap(pA, 1)), "常驻 A");
        CHECK(cache.put(snap(pB, 2)), "常驻 B");

        // —— A 被顶掉（第三个对话来了）——
        std::vector<SavedConversation> evicted;
        cache.make_room(0, 0, &evicted);
        CHECK(evicted.size() == 1, "顶掉 1 个");
        // ★ 写盘（真实流程里由 generate.cpp 做）
        bool stored = false;
        for (auto& e : evicted) {
            const auto ids = e.live.ids;                    // 写盘前取出身份
            const std::int64_t frontier = static_cast<std::int64_t>(ids.size());
            stored = spool.store(e, frontier, ids, {}, {}) > 0 || stored;
        }
        CHECK(stored, "★ 被顶掉的 A 写进了磁盘");

        // 现在缓存里只剩 B（假设随后也被换走）
        std::vector<SavedConversation> rest;
        cache.drain(rest);
        CHECK(cache.size() == 0, "★ drain 把常驻的全部腾空");
        CHECK(rest.size() == 1, "腾出 1 个（B）");
        for (auto& e : rest) {
            const auto ids = e.live.ids;
            spool.store(e, static_cast<std::int64_t>(ids.size()), ids, {}, {});
        }

        // —— 新请求：A 的完整 prompt 回来了，末尾多一个新 token ——
        // ★ find() 与上游 best() 同一条约定：frontier 必须【严格】短于 prompt
        //   （最后一个 token 开启下一个 verify 窗口，见 disk_park_spool.cpp 的 find()）。
        //   所以查询用的 prompt 要比盘上的 frontier 至少长一个 token，等长必然 miss。
        auto extend = [](std::vector<std::int32_t> v) { v.push_back(99); return v; };
        const std::vector<std::int32_t> qA = extend(pA), qB = extend(pB);
        // 1) 缓存里没有 A
        auto m = cache.best(pA, {}, true);
        CHECK(m.tokens == 0, "缓存里没有 A 的前缀");
        // 2) 磁盘上有
        auto f = spool.find(qA, {}, {});
        CHECK(f.has_value() && f->frontier == (std::int64_t)pA.size(), "★ 磁盘上命中 A（frontier=10）");
        // 3) 取回并放回缓存
        if (f) {
            auto img = spool.load(*f);
            CHECK(img.has_value(), "★ 从磁盘读回 A");
            if (img) {
                CHECK(img->live.ids == pA, "读回的 ids 正确");
                CHECK(img->live.gdn.size() == 4096 && img->live.gdn[0] == 1, "★ 内容正确（是 A 不是 B）");
                CHECK(cache.put(std::move(*img)), "放回缓存");
                // ★ best() 要求前缀【严格】短于 prompt（最后一个 token 开启下一个 verify 窗口），
                //   所以这里用比 A 长两个 token 的 prompt，期望匹配满 A 的 10 个。
                std::vector<std::int32_t> pA_ext = pA;
                pA_ext.push_back(99); pA_ext.push_back(100);
                auto m2 = cache.best(pA_ext, {}, true);
                CHECK(m2.tokens == (int64_t)pA.size(), "★ 现在缓存里能匹配满 A 的前缀（10）");
            }
        }

        // —— B 的 prompt 仍应命中磁盘上的 B ——
        auto fb = spool.find(qB, {}, {});
        CHECK(fb.has_value() && fb->frontier == (std::int64_t)pB.size(), "★ 磁盘上也能命中 B");
        CHECK(fb.has_value() && f.has_value() && !(fb->digest == f->digest),
              "★ A 与 B 长度相同但摘要不同（这正是只按 frontier 查会出错的场景）");
        auto ib = spool.load(*fb);
        CHECK(ib && ib->live.gdn[0] == 2, "★ 读回的是 B（内容按 seed 区分）");

        // —— 一个从没见过的对话不应命中 ——
        std::vector<std::int32_t> pC = {9,9,9,9,9,9,9,9,9,9};
        CHECK(!spool.find(pC, {}, {}).has_value(), "★ 未知对话 → 不命中");
        // ★ 同一 frontier 上必须能区分不同对话（本节的两个条目长度相同）
        auto fa = spool.find(qA, {}, {});
        auto fbb = spool.find(qB, {}, {});
        CHECK(fa && fbb && !(fa->digest == fbb->digest), "★ 同长度不同对话的摘要不同");
    }

    std::printf("\n=== 3. ★ 部分前缀：只命中能匹配的最长 frontier ===\n");
    {
        ConversationCache cache(1 << 20, 2);
        CHECK(cache.put(snap(pCommon, 9)), "常驻公共前缀那一份");   // 只有 5 个 token
        std::vector<SavedConversation> ev;
        cache.drain(ev);                             // ★ 只有 1 个条目时 make_room 不会淘汰
        CHECK(ev.size() == 1, "取出（drain）");

        DiskParkSpool spool(so);
        for (auto& e : ev) {
            const auto ids = e.live.ids;
            spool.store(e, static_cast<std::int64_t>(ids.size()), ids, {}, {});
        }
        // ★ 用一个"只共享前 5 个 token、之后不同"的 prompt。
        //   （若直接用 pA，find 会取最长的 A（frontier=10）—— 那才是正确行为。）
        std::vector<std::int32_t> pA2 = {1,2,3,4,5, 77,78,79};
        auto f = spool.find(pA2, {}, {});
        CHECK(f.has_value() && f->frontier == 5, "★ 命中公共前缀（frontier=5），不是更长的那两个");
        auto img = spool.load(*f);
        CHECK(img && img->live.gdn[0] == 9, "读回的是公共前缀那一份（seed 9）");
        // 而带 A 全前缀的 prompt 应当命中更长的 A（frontier=10）
        std::vector<std::int32_t> qA = pA; qA.push_back(99);   // 同样要比 frontier 长一个 token
        auto fa2 = spool.find(qA, {}, {});
        CHECK(fa2.has_value() && fa2->frontier == 10, "★ pA 取最长的 10（不是 5）");
    }

    fs::remove_all(dir);
    std::printf("\n========================================\n");
    std::printf("  检查 %d 项，失败 %d 项\n", g_checks, g_failed);
    std::printf("========================================\n\n");
    return g_failed == 0 ? 0 : 1;
}
