// src/program/layer_split_test.cpp - checks include/strata/program/layer_split.hpp.
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
constexpr uint64_t GiB = 1ull << 30;
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
        // a session of 10 bytes a layer: one card running all 8 layers has 100 - 80 = 20 bytes left for experts
        const ls::Planner pl(kL, toy_profile(), ones, costs, [](int64_t lb, int64_t le) { return 10 * (le - lb); });
        const ls::Placement one = pl.predict({{100, 1.0}}, {});
        check(one.held == 20 && near(one.ms, 8.0 + 100.0 * (1.0 - toy_mass(20, 1.0))),
              "the carve comes off the card's room: 20 of 32 pairs held");
        // 12 bytes a layer plus its 4 pairs: the faster card fits 6 layers in 100, so K=6, not 7 as without it
        const ls::Planner pc(kL, toy_profile(), ones, costs, [](int64_t lb, int64_t le) { return 12 * (le - lb); });
        const ls::Placement k7 = pc.predict({{100, 0.5}, {100, 1.0}}, {7});
        check(k7.held == 20 && k7.ms > 15.0, "the carve priced: K=7 holds 20 pairs (16 on the faster card)");
        const ls::Placement b = pc.best({{100, 0.5}, {100, 1.0}});
        check(b.at == std::vector<int64_t>{6} && b.held == kL * kE && near(b.ms, 5.0),
              "the carve priced: the faster card takes the 6 layers it can hold (K=6)");
    }
    std::printf(g_fail ? "layer_split_test: %d FAILED\n" : "layer_split_test: all passed\n", g_fail);
    return g_fail ? 1 : 0;
}
