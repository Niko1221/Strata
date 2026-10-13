#include "strata/core/disk_park_image.hpp"

#include <cstdio>
#include <chrono>
#include <cstring>
#include <vector>

using namespace strata::core;

static int g_checks = 0, g_failed = 0;
#define CHECK(cond, msg)                                                 \
    do {                                                                 \
        ++g_checks;                                                      \
        if (!(cond)) { ++g_failed; std::printf("  FAIL  %s\n", (msg)); } \
        else         { std::printf("  ok    %s\n", (msg)); }             \
    } while (0)

static ConversationBuffer buf(std::initializer_list<std::uint8_t> v) { return ConversationBuffer(v); }

static bool eq_ckpt(const ConversationCheckpoint& a, const ConversationCheckpoint& b) {
    if (a.ids != b.ids || a.used != b.used) return false;
    if (a.imgs.size() != b.imgs.size()) return false;
    for (std::size_t i = 0; i < a.imgs.size(); ++i)
        if (!(a.imgs[i] == b.imgs[i])) return false;
    auto vv = [](const std::vector<std::uint8_t>& x, const std::vector<std::uint8_t>& y) { return x == y; };
    if (!vv(a.gdn, b.gdn) || !vv(a.ple, b.ple) || !vv(a.tails, b.tails) ||
        !vv(a.dead, b.dead) || !vv(a.block_pos, b.block_pos)) return false;
    if (a.stage_parts.size() != b.stage_parts.size()) return false;
    for (std::size_t i = 0; i < a.stage_parts.size(); ++i)
        if (!eq_ckpt(a.stage_parts[i], b.stage_parts[i])) return false;
    return true;
}
static bool eq_buf(const ConversationBuffer& a, const ConversationBuffer& b) {
    if (a.size() != b.size()) return false;
    std::vector<std::uint8_t> x(a.size()), y(b.size());
    std::size_t w = 0;
    a.visit(0, a.size(), [&](const std::uint8_t* p, std::size_t n, std::size_t) { std::memcpy(x.data()+w, p, n); w += n; return true; });
    w = 0;
    b.visit(0, b.size(), [&](const std::uint8_t* p, std::size_t n, std::size_t) { std::memcpy(y.data()+w, p, n); w += n; return true; });
    return x == y;
}
static bool eq_kv(const ConversationKv& a, const ConversationKv& b) {
    return a.format == b.format && a.cells == b.cells && a.heads == b.heads &&
           a.head_dim == b.head_dim && a.page_size == b.page_size &&
           a.pooled_rows == b.pooled_rows && a.idx_dim == b.idx_dim &&
           eq_buf(a.k, b.k) && eq_buf(a.v, b.v) && eq_buf(a.k_scale, b.k_scale) &&
           eq_buf(a.v_scale, b.v_scale) && eq_buf(a.pooled, b.pooled);
}
static bool eq_saved(const SavedConversation& a, const SavedConversation& b) {
    if (a.geometry != b.geometry || a.layer_lo != b.layer_lo || a.layer_hi != b.layer_hi) return false;
    if (a.cvec != b.cvec) return false;
    if (!eq_ckpt(a.live, b.live)) return false;
    if (a.checkpoints.size() != b.checkpoints.size()) return false;
    for (std::size_t i = 0; i < a.checkpoints.size(); ++i)
        if (!eq_ckpt(a.checkpoints[i], b.checkpoints[i])) return false;
    if (a.kv.size() != b.kv.size()) return false;
    for (std::size_t i = 0; i < a.kv.size(); ++i) if (!eq_kv(a.kv[i], b.kv[i])) return false;
    if (a.stage_images.size() != b.stage_images.size()) return false;
    for (std::size_t i = 0; i < a.stage_images.size(); ++i)
        if (!eq_saved(a.stage_images[i], b.stage_images[i])) return false;
    return true;
}

int main() {
    std::printf("\n=== 1. 空结构往返 ===\n");
    {
        SavedConversation s;
        auto bytes = pack_saved_conversation(s);
        auto back = unpack_saved_conversation(bytes);
        CHECK(back.has_value(), "空结构能往返");
        CHECK(back && eq_saved(s, *back), "内容一致");
        CHECK(bytes.size() == packed_saved_conversation_bytes(s), "预估大小 == 实际大小");
    }

    std::printf("\n=== 2. 带内容的往返 ===\n");
    {
        SavedConversation s;
        for (std::size_t i = 0; i < s.geometry.size(); ++i) s.geometry[i] = (std::int64_t)i * 7 - 3;
        s.layer_lo = 1; s.layer_hi = 11; s.cvec = false;

        s.live.ids = {100, 200, 300, 400};
        s.live.imgs.push_back({0, 0xDEADBEEF});
        s.live.imgs.push_back({3, 0x12345678});
        s.live.gdn = {1,2,3,4,5};
        s.live.ple = {9,8,7};
        s.live.tails = {0};
        s.live.dead = {0,1};
        s.live.block_pos = {2,3,4};
        s.live.used = 12345;
        ConversationCheckpoint sub;
        sub.ids = {7,7,7};
        sub.gdn = {42};
        s.live.stage_parts.push_back(sub);

        for (int i = 0; i < 3; ++i) {
            ConversationCheckpoint c;
            c.ids.assign(static_cast<std::size_t>(i + 1) * 10, i);
            c.imgs.push_back({(std::int64_t)i, 0xABCD0000ULL + (std::uint64_t)i});
            c.gdn.assign(static_cast<std::size_t>(i) * 100 + 1, static_cast<std::uint8_t>(i + 1));
            c.ple.assign(static_cast<std::size_t>(i) * 50 + 1, static_cast<std::uint8_t>(200 - i));
            c.used = static_cast<std::uint64_t>(i) * 1000;
            s.checkpoints.push_back(std::move(c));
        }

        for (int i = 0; i < 2; ++i) {
            ConversationKv k;
            k.format = i; k.cells = 100 + i; k.heads = 8; k.head_dim = 128;
            k.page_size = 16 * 1024 * 1024; k.pooled_rows = i * 3; k.idx_dim = 64;
            k.k = buf({1,2,3,4,5,6,7,8});
            k.v = buf({8,7,6,5});
            k.k_scale = buf({1,1});
            k.v_scale = buf({2,2});
            k.pooled = buf({});
            s.kv.push_back(std::move(k));
        }

        SavedConversation stage;
        stage.cvec = true;
        stage.layer_lo = 12; stage.layer_hi = 20;
        stage.live.ids = {555, 666};
        s.stage_images.push_back(std::move(stage));

        const std::size_t expected = packed_saved_conversation_bytes(s);
        auto bytes = pack_saved_conversation(s);
        CHECK(bytes.size() == expected, "预估大小 == 实际大小（复杂结构）");
        auto back = unpack_saved_conversation(bytes);
        CHECK(back.has_value(), "复杂结构能往返");
        CHECK(back && eq_saved(s, *back), "★ 逐字段一致（含 checkpoints / stage_parts / kv / stage_images）");
        CHECK(back && back->live.used == 12345, "used 字段保真");
        CHECK(back && back->geometry[17] == 17 * 7 - 3, "geometry 保真（restore requires the same）");
        CHECK(back && back->layer_lo == 1 && back->layer_hi == 11, "layer 范围保真");

        auto bytes2 = pack_saved_conversation(s);
        CHECK(bytes == bytes2, "★ 打包是确定性的（同样输入 → 同样字节）");
    }

    std::printf("\n=== 3. ★ 损坏必须干净失败（磁盘上的东西坏是常态）===\n");
    {
        SavedConversation s;
        s.live.ids = {1,2,3};
        s.live.gdn = {5,5,5};
        auto good = pack_saved_conversation(s);

        // 3a 截断到任意长度都应被拒（不能产生半成品）
        bool all_rejected = true;
        for (std::size_t cut = 0; cut < good.size(); ++cut) {
            std::vector<std::byte> t(good.begin(), good.begin() + static_cast<std::ptrdiff_t>(cut));
            if (unpack_saved_conversation(t).has_value()) { all_rejected = false; break; }
        }
        CHECK(all_rejected, "★ 任意截断 → nullopt（不产生半成品）");

        // 3b magic 坏
        auto bad = good;
        bad[0] = static_cast<std::byte>(0x00);
        CHECK(!unpack_saved_conversation(bad).has_value(), "★ magic 不符 → nullopt");

        // 3c 版本号坏
        bad = good;
        bad[4] = static_cast<std::byte>(99);
        CHECK(!unpack_saved_conversation(bad).has_value(), "★ 版本不符 → nullopt");

        // 3d 尾部多余字节
        auto extra = good;
        extra.push_back(static_cast<std::byte>(0));
        CHECK(!unpack_saved_conversation(extra).has_value(), "★ 尾部多余字节 → nullopt");

        // 3e 空输入
        CHECK(!unpack_saved_conversation({}).has_value(), "空输入 → nullopt");

        // 3f ★ 伪造巨量 count 不能拖进 OOM，必须快速返回
        // 布局: magic(4) version(4) geometry(18*8=144) layer_lo(8) layer_hi(8)
        //       → live.ids 的 count 在偏移 168
        constexpr std::size_t kIdsCountOff = 4 + 4 + 18 * 8 + 8 + 8;
        CHECK(good.size() > kIdsCountOff + 8, "样本足够长");
        std::vector<std::byte> evil = good;
        const std::uint64_t huge = 0xFFFFFFFFFFFFULL;
        std::memcpy(evil.data() + kIdsCountOff, &huge, sizeof huge);
        // 先确认这确实改到了 count（改之前能往返）
        CHECK(unpack_saved_conversation(good).has_value(), "未篡改的样本可往返");
        auto t0 = std::chrono::steady_clock::now();
        const bool rejected = !unpack_saved_conversation(evil).has_value();
        const auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(
                            std::chrono::steady_clock::now() - t0).count();
        CHECK(rejected, "★ 伪造巨量 count → nullopt");
        CHECK(ms < 200, "★ 且不慢（有长度上限检查，不会 OOM/长循环）");
    }

    std::printf("\n========================================\n");
    std::printf("  检查 %d 项，失败 %d 项\n", g_checks, g_failed);
    std::printf("========================================\n\n");
    return g_failed == 0 ? 0 : 1;
}
