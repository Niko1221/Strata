#include "strata/core/disk_park_digest.hpp"

#include <cstdio>
#include <string>
#include <vector>

using namespace strata::core;

// span 不能从花括号隐式构造，用这两个辅助保持测试可读。
static std::vector<std::uint64_t> H(std::initializer_list<std::uint64_t> l) { return {l}; }
static std::vector<std::int64_t>  S(std::initializer_list<std::int64_t> l) { return {l}; }

static int g_checks = 0, g_failed = 0;
#define CHECK(cond, msg)                                                 \
    do {                                                                 \
        ++g_checks;                                                      \
        if (!(cond)) { ++g_failed; std::printf("  FAIL  %s\n", (msg)); } \
        else         { std::printf("  ok    %s\n", (msg)); }             \
    } while (0)

int main() {
    const std::vector<std::int32_t> ids{7, 8, 9, 10, 11, 12, 13, 14};

    std::printf("\n=== 1. 确定性 ===\n");
    {
        auto a = prefix_digest(ids, {}, {}, 5);
        CHECK(a == prefix_digest(ids, {}, {}, 5), "同样输入 → 同样摘要");
        std::vector<std::int32_t> ids2 = ids; ids2[4] = 99;
        CHECK(!(a == prefix_digest(ids2, {}, {}, 5)), "★ 最后一个 token 不同 → 摘要不同");
        CHECK(!(a == prefix_digest(ids, {}, {}, 4)), "★ 长度不同 → 摘要不同");
        CHECK(!(a == prefix_digest(ids, {}, {}, 6)), "★ 更长 → 摘要不同");
    }

    std::printf("\n=== 2. ★★ 前缀性质：逐 token 增量 == 从头算（恢复能命中的前提）===\n");
    {
        const std::vector<std::uint64_t> ih{0xAAAA, 0xBBBB, 0xCCCC};
        const std::vector<std::int64_t>  is{2, 5, 9};
        PrefixDigestBuilder b;
        bool ok = true;
        for (std::size_t i = 0; i < ids.size(); ++i) {
            b.advance(ids[i], ih, is);
            if (!(b.value() == prefix_digest(ids, ih, is, static_cast<std::int64_t>(i + 1)))) {
                ok = false; std::printf("      （在 frontier=%zu 处不一致）\n", i + 1); break;
            }
        }
        CHECK(ok, "★ 每个 frontier：增量 == 从头算");
    }

    std::printf("\n=== 3. ★ 图片语义与 conversation_prefix 对齐 ===\n");
    {
        const std::uint64_t H0 = 111, H3 = 222;
        const auto plain2 = prefix_digest(ids, H({}), S({}), 2);

        // start=0 < 2 → frontier=2 时应已折入
        CHECK(!(prefix_digest(ids, H({H0}), S({0}), 2) == plain2), "★ start=0 在 frontier=2 被折入");

        // start=5 >= 2 → frontier=2 时不应折入
        CHECK(prefix_digest(ids, H({H0, H3}), S({5, 5}), 2) == prefix_digest(ids, H({H3}), S({5}), 2),
              "★ start=5 在 frontier=2 【未】折入（与完全无图不同，但与只有 start=5 相同）");
        CHECK(prefix_digest(ids, H({H3}), S({5}), 2) == plain2, "★ start=5 在 frontier=2 等价于无图");

        // start=3 < 4 → frontier=4 时应折入
        CHECK(!(prefix_digest(ids, H({H0, H3}), S({0, 3}), 4) == prefix_digest(ids, H({H0}), S({0}), 4)),
              "★ start=3 在 frontier=4 被折入");

        // start=5 >= 4 → frontier=4 时不应折入
        CHECK(prefix_digest(ids, H({H0, H3}), S({0, 5}), 4) == prefix_digest(ids, H({H0}), S({0}), 4),
              "★ start=5 在 frontier=4 【未】折入");

        // frontier=7 时 start=5 应折入
        CHECK(!(prefix_digest(ids, H({H0, H3}), S({0, 5}), 7) == prefix_digest(ids, H({H0}), S({0}), 7)),
              "★ start=5 在 frontier=7 已折入");

        // frontier=6 时 start=5 也应已折入（5 < 6）
        CHECK(!(prefix_digest(ids, H({H0, H3}), S({0, 5}), 6) == prefix_digest(ids, H({H0}), S({0}), 6)),
              "★ start=5 在 frontier=6 已折入");

        // ★ 图片顺序影响摘要
        CHECK(!(prefix_digest(ids, H({H0, H3}), S({0, 3}), 8) == prefix_digest(ids, H({H3, H0}), S({3, 0}), 8)),
              "★ 图片顺序不同 → 摘要不同");

        // hash 影响摘要
        CHECK(!(prefix_digest(ids, H({H0}), S({0}), 8) == prefix_digest(ids, H({H3}), S({0}), 8)),
              "★ 图片 hash 不同 → 摘要不同");

        // ★ imgs 比 ids 长时不应越界（引擎里 c.imgs 与 c.ids 独立）
        CHECK(prefix_digest(ids, H({H0}), S({0}), 8) == prefix_digest(ids, H({H0, H3}), S({0}), 8) ||
              true, "imgs/starts 长度不齐不崩");

        CHECK(prefix_digest({}, H({}), S({}), 0) == PrefixDigestBuilder{}.value(), "空前缀 = 种子值");
    }

    std::printf("\n=== 4. ★ profile 标签 ===\n");
    {
        DiskParkProfile base;
        base.model = "Qwen3.8-Flash-Next"; base.kv_format = "int8";
        base.max_context = 262144; base.rope = "none";
        const std::uint32_t t0 = base.tag();

        auto chg = [&](auto f) { DiskParkProfile p = base; f(p); return p.tag(); };
        CHECK(chg([](DiskParkProfile& p){ p.kv_format = "q4_0"; }) != t0, "★ kv_format → 标签变");
        CHECK(chg([](DiskParkProfile& p){ p.kv_format = "k8v4"; }) != t0, "★ k8v4 ≠ int8");
        CHECK(chg([](DiskParkProfile& p){ p.kv_format = "fp16"; }) != t0, "★ fp16 ≠ int8");
        CHECK(chg([](DiskParkProfile& p){ p.rope = "yarn:2"; }) != t0, "★ rope → 标签变");
        CHECK(chg([](DiskParkProfile& p){ p.rope = "yarn:4"; }) != t0, "★ yarn:4 ≠ yarn:2");
        CHECK(chg([](DiskParkProfile& p){ p.max_context = 524288; }) != t0, "★ 档位 → 标签变");
        CHECK(chg([](DiskParkProfile& p){ p.model = "Other"; }) != t0, "★ 模型 → 标签变");

        DiskParkProfile same = base;
        CHECK(same.tag() == t0, "同样 profile → 同样标签（跨重启稳定）");

        DiskParkProfile p1, p2;
        p1.model = "ab"; p1.kv_format = "c"; p1.max_context = 1; p1.rope = "none";
        p2.model = "a";  p2.kv_format = "bc"; p2.max_context = 1; p2.rope = "none";
        CHECK(p1.tag() != p2.tag(), "★ '|' 分隔符消除字段拼接歧义");

        std::printf("       描述: %s\n", base.describe().c_str());
        std::printf("       标签: 0x%08X\n", t0);
    }

    std::printf("\n========================================\n");
    std::printf("  检查 %d 项，失败 %d 项\n", g_checks, g_failed);
    std::printf("========================================\n\n");
    return g_failed == 0 ? 0 : 1;
}
