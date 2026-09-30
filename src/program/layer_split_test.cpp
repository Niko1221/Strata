// src/program/layer_split_test.cpp - the layer split's preparation: what the host arena pins, and the prompt path's
// buffers lent by every card's expert cache.
//
//   1. the arena is pinned whole unless WDDM maps it into several contexts (then 8 GiB, as measured on an RTX 5080 +
//      RTX 3090); one GPU or Linux: no cap;
//   2. the slots a cache lends for `need` bytes are taken from its end: uniform slots round up, sized slots stop at
//      the first run from the end that holds `need`;
//   3. the bytes lent from a first slot to the end, for both layouts;
//   4. one card: --prefill auto takes the largest chunk whose buffers leave 128 slots and stay under the lend share;
//      a fixed chunk is halved until it fits; nothing fits: chunk 0;
//   5. several cards: ONE chunk for all of them (the hand-off rows are that size), the largest every card can lend -
//      a card with a small cache decides it - and each card lends its own slot count;
//   6. a card that cannot lend even the smallest chunk: no plan (every card allocates its own buffers).
//
// The placement (--layer-split auto), on a toy model of 8 layers x 4 experts whose profile interleaves the layers
// (rank r = layer r % 8, expert r / 8), so every layer's pairs are spread over the ranking:
//   7. a card's per-layer time from SMs x clock (0.33 ms on an RTX 5080, 0.69 on a 2080 Ti);
//   8. one card that holds every pair: its layers' time, nothing missed; half the pairs: the missed mass costs;
//   9. a cache fills hottest first and stops at the first pair that does not fit (as the engine's fill);
//  10. every later card adds one hand-off per window;
//  11. ample caches: the faster card takes every layer it can (the per-layer time decides);
//  12. equal cards, tight caches: the split point that lets both caches hold their layers' pairs;
//  13. three cards: every placement is tried; four or more: layers in proportion to speed;
//  14. a split is kept only when it is predicted faster than one card by more than the margin.
#include "strata/program/layer_split.hpp"

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <utility>
#include <vector>

namespace ls = strata::program::layer_split;

namespace {
int g_fail = 0;
void check(bool ok, const char* what) {
    std::printf("  %-74s %s\n", what, ok ? "ok" : "FAIL");
    if (!ok) ++g_fail;
}
bool near(double a, double b, double eps = 1e-9) { return std::fabs(a - b) <= eps; }
constexpr uint64_t MiB = 1ull << 20, GiB = 1ull << 30;
// a prompt path's device bytes for a chunk: 200 MiB fixed + 0.25 MiB a token (the shape of Prefill::bytes_needed)
uint64_t need(int64_t chunk) { return 200 * MiB + (uint64_t) chunk * (MiB / 4); }
// the toy model: 8 layers x 4 experts, the layers interleaved in the ranking, one byte a slot
constexpr int64_t kL = 8, kE = 4;
std::vector<std::pair<int32_t, int32_t>> toy_profile() {
    std::vector<std::pair<int32_t, int32_t>> p;
    for (int32_t r = 0; r < kL * kE; ++r) p.emplace_back(r % kL, r / kL);
    return p;
}
double toy_mass(int64_t n, double exp) {   // the share of the routed mass the n hottest ranks hold
    double a = 0, t = 0;
    for (int64_t r = 0; r < kL * kE; ++r) {
        const double m = std::pow((double) r + 1.0, -exp);
        t += m;
        if (r < n) a += m;
    }
    return a / t;
}
}  // namespace

int main() {
    std::printf("layer_split_test\n");
    {
        check(ls::arena_pin_cap(false, true) == 0, "one context, WDDM: the whole arena");
        check(ls::arena_pin_cap(false, false) == 0, "one context, Linux: the whole arena");
        check(ls::arena_pin_cap(true, true) == 8 * GiB, "several contexts under WDDM: 8 GiB");
        check(ls::arena_pin_cap(true, false) == 0, "several contexts, Linux: the whole arena");
    }
    {
        const ls::CacheSlots u{1000, 1000 * (int64_t) MiB, nullptr, (int64_t) MiB};   // uniform 1 MiB slots
        check(ls::slots_to_lend(u, 10 * MiB) == 10, "uniform slots: an exact multiple");
        check(ls::slots_to_lend(u, 10 * MiB + 1) == 11, "uniform slots: rounded up");
        check(ls::lent_bytes(u, 990) == 10 * MiB, "uniform slots: bytes from slot 990 to the end");
        // sized slots: 1 MiB each, the last four 2 MiB (offsets: each slot's start, then the end - as ExpertCache)
        std::vector<uint64_t> off;
        uint64_t at = 0;
        for (int i = 0; i < 100; ++i) { off.push_back(at); at += i >= 96 ? 2 * MiB : MiB; }
        off.push_back(at);
        const ls::CacheSlots s{100, (int64_t) at, off.data(), 2 * (int64_t) MiB};
        check(ls::slots_to_lend(s, 8 * MiB) == 4, "sized slots: the four 2 MiB slots hold 8 MiB");
        check(ls::slots_to_lend(s, 8 * MiB + 1) == 5, "sized slots: one more byte takes a fifth slot");
        check(ls::lent_bytes(s, 95) == 9 * MiB, "sized slots: bytes from slot 95 to the end");
        check(ls::slots_to_lend(s, 1000 * MiB) == 100, "sized slots: more than the cache holds: all of it");
    }
    {
        const ls::CacheSlots big{4000, 4000 * (int64_t) MiB, nullptr, (int64_t) MiB};
        const auto p = ls::plan_lend({big}, [](size_t, int64_t c) { return need(c); }, 8192, true, 90);
        check(p.chunk == 8192 && p.slots.size() == 1 && p.slots[0] == 2248, "one card, auto: 8192 (2248 of 4000 slots)");
        const ls::CacheSlots mid{3000, 3000 * (int64_t) MiB, nullptr, (int64_t) MiB};
        const auto q = ls::plan_lend({mid}, [](size_t, int64_t c) { return need(c); }, 8192, true, 60);
        check(q.chunk == 6144 && q.slots[0] == 1736, "one card, auto, 60% share: 6144 (1736 slots)");
        const auto f = ls::plan_lend({mid}, [](size_t, int64_t c) { return need(c); }, 16384, false, 90);
        check(f.chunk == 8192, "one card, fixed 16384: halved to 8192 (leaves 128 slots)");
        const ls::CacheSlots tiny{300, 300 * (int64_t) MiB, nullptr, (int64_t) MiB};
        const auto n = ls::plan_lend({tiny}, [](size_t, int64_t c) { return need(c); }, 8192, true, 90);
        check(n.chunk == 0 && n.slots.empty(), "one card, too small for any chunk: no plan");
    }
    {
        const ls::CacheSlots a{4000, 4000 * (int64_t) MiB, nullptr, (int64_t) MiB};
        const ls::CacheSlots b{2000, 2000 * (int64_t) MiB, nullptr, (int64_t) MiB};
        const auto p = ls::plan_lend({a, b}, [](size_t, int64_t c) { return need(c); }, 8192, true, 90);
        check(p.chunk == 6144 && p.slots.size() == 2, "two cards, auto: the smaller cache decides (6144, not 8192)");
        check(p.slots.size() == 2 && p.slots[0] == 1736 && p.slots[1] == 1736, "two cards, auto: each lends its slots");
        // the later card's prompt path needs more (its own buffers differ): per-stage need
        const auto q = ls::plan_lend({a, a}, [](size_t st, int64_t c) { return need(c) + (st == 1 ? 1024 * MiB : 0); },
                                     8192, true, 90);
        check(q.chunk == 8192 && q.slots.size() == 2 && q.slots[0] == 2248 && q.slots[1] == 3272,
              "two cards, per-stage need: slots differ");
        const ls::CacheSlots tiny{300, 300 * (int64_t) MiB, nullptr, (int64_t) MiB};
        const auto n = ls::plan_lend({a, tiny}, [](size_t, int64_t c) { return need(c); }, 8192, true, 90);
        check(n.chunk == 0 && n.slots.empty(), "two cards, one too small for any chunk: no plan");
        const auto e = ls::plan_lend({}, [](size_t, int64_t c) { return need(c); }, 8192, true, 90);
        check(e.chunk == 0 && e.slots.empty(), "no cards: no plan");
    }
    {
        check(near(ls::layer_ms_estimate(84, 2.617), 0.33, 1e-6), "per-layer time: RTX 5080 (84 SMs, 2.62 GHz) 0.33 ms");
        check(near(ls::layer_ms_estimate(68, 1.545), 0.6905, 1e-3), "per-layer time: RTX 2080 Ti (68 SMs, 1.55 GHz) 0.69 ms");
    }
    const std::vector<int64_t> ones((size_t) kL, 1);
    ls::Costs costs;
    costs.miss_ms = 100.0;
    costs.handoff_ms = 0.0;
    costs.mass_exp = 1.0;
    {
        const ls::Planner pl(kL, toy_profile(), ones, costs);
        const ls::Placement all = pl.predict({{100, 1.0}}, {});
        check(all.held == kL * kE && near(all.held_mass, 1.0) && near(all.ms, 8.0),
              "one card holding every pair: 8 layers x 1 ms, nothing missed");
        const ls::Placement half = pl.predict({{16, 1.0}}, {});
        check(half.held == 16 && near(half.held_mass, toy_mass(16, 1.0)) &&
                  near(half.ms, 8.0 + 100.0 * (1.0 - toy_mass(16, 1.0))),
              "one card holding half: the missed mass costs miss_ms per unit");
        ls::Costs flat = costs;
        flat.mass_exp = 0.5;
        const ls::Planner pf(kL, toy_profile(), ones, flat);
        check(pf.predict({{16, 1.0}}, {}).held_mass < half.held_mass, "a flatter routing curve: the same cache holds less");
    }
    {
        std::vector<int64_t> bytes = ones;
        bytes[1] = 5;   // layer 1's experts take 5 bytes: rank 1 does not fit a 3-byte cache after rank 0
        const ls::Planner pl(kL, toy_profile(), bytes, costs);
        check(pl.predict({{3, 1.0}}, {}).held == 1, "the fill stops at the first pair that does not fit");
    }
    {
        ls::Costs h = costs;
        h.handoff_ms = 0.5;
        const ls::Planner pl(kL, toy_profile(), ones, h);
        check(near(pl.predict({{100, 1.0}, {100, 1.0}}, {4}).ms, 8.5), "two cards: one hand-off per window");
        check(near(pl.predict({{100, 1.0}, {100, 1.0}, {100, 1.0}}, {3, 6}).ms, 9.0), "three cards: two hand-offs");
    }
    {
        const ls::Planner pl(kL, toy_profile(), ones, costs);
        const ls::Placement a = pl.best({{100, 0.5}, {100, 1.0}});
        check(a.at == std::vector<int64_t>{7} && near(a.ms, 4.5), "ample caches, faster first card: it takes 7 layers");
        const ls::Placement b = pl.best({{100, 1.0}, {100, 0.5}});
        check(b.at == std::vector<int64_t>{2} && near(b.ms, 5.0), "ample caches, faster second card: it takes 6 layers");
        const ls::Placement c = pl.best({{8, 1.0}, {24, 1.0}});
        check(c.at == std::vector<int64_t>{2} && c.held == kL * kE,
              "equal cards, tight caches: K=2 lets both hold their layers' pairs");
        const ls::Placement d = pl.best({{100, 0.5}, {100, 1.0}, {100, 1.0}});
        check(d.at == (std::vector<int64_t>{6, 7}) && near(d.ms, 5.0), "three cards: every placement tried");
        const ls::Placement e = pl.best({{100, 1.0}, {100, 1.0}, {100, 1.0}, {100, 1.0}});
        check(e.at == (std::vector<int64_t>{2, 4, 6}), "four equal cards: layers in proportion to speed");
    }
    {
        check(!ls::split_pays(60.0, 63.0, 0.05), "a split 5% faster or less: one card (margin 5%)");
        check(ls::split_pays(55.0, 63.0, 0.05), "a split 13% faster: kept");
        check(!ls::split_pays(63.0, 60.0, 0.0), "a split slower than one card: one card");
    }
    std::printf(g_fail ? "layer_split_test: %d FAILED\n" : "layer_split_test: all passed\n", g_fail);
    return g_fail ? 1 : 0;
}
