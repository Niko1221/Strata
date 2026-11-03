#include "../src/expert_residency.hpp"

#include <cstdio>
#include <cstdlib>
#include <random>

using namespace fnx;

static void check(bool ok, const char* what) {
    if (!ok) { std::fprintf(stderr, "FAIL: %s\n", what); std::exit(1); }
}
template <typename F> static void refuses(F f) {
    try { f(); } catch (const std::runtime_error&) { return; }
    check(false, "impossible placement must fail");
}

int main() {
    // A cold one-layer pool must keep its hottest slots, even while a larger pool can still shrink.
    const std::vector<std::pair<size_t, size_t>> sizes{{3, 2}, {4, 3}, {3, 2}};
    const std::vector<double> score{10, 11, 12, 13, 0, 1, 2, 3, 14, 15, 16, 17};
    auto p = expert_residency(sizes, score, 4, 2, 0, 6);
    check(p.resident == std::vector<std::vector<char>>{{0,0,0,0}, {0,0,1,1}, {1,1,1,1}}, "rare pool floor and next cold pool");
    check(p.count == 6 && p.moved == 34, "extra counts only experts actually evicted");
    p = expert_residency(sizes, score, 4, 2, 8, 2);
    check(p.count == 4 && p.moved == 24, "byte target followed by extra evictions");
    p = expert_residency(sizes, score, 4, 2, 0, 0);
    check(p.count == 0, "zero request retains all experts");
    p = expert_residency(sizes, score, 4, 2, 44, 0);
    check(p.count == 8 && p.moved == 44, "exact capacity succeeds");
    refuses([&] { expert_residency(sizes, score, 4, 2, 45, 0); });
    refuses([&] { expert_residency(sizes, score, 4, 2, 44, 1); });
    refuses([&] { expert_residency(sizes, score, 4, 5, 0, 0); });

    // Equal total bytes do not make equal pools: gate/up and down strides both matter.
    p = expert_residency({{3,2}, {2,3}}, {0,0,0,0,0,0,0,0}, 4, 2, 0, 4);
    check(p.resident == std::vector<std::vector<char>>{{0,0,1,1}, {0,0,1,1}}, "pool strides and stable ties");

    // Prod-shaped 43/1/4 layer pools; the old global ordering empties the rare pool at this pressure.
    std::vector<std::pair<size_t, size_t>> prod(48, {1843200,1228800});
    prod[2] = {2252800,1740800};
    for (int l : {4,30,46,47}) prod[(size_t) l] = {1843200,1740800};
    std::vector<double> ranks(48 * 512);
    std::mt19937 rng(42);
    for (double& s : ranks) s = (double) rng() / rng.max();
    for (int e = 0; e < 512; ++e) ranks[2*512 + e] = -1 + e / 512.0;
    p = expert_residency(prod, ranks, 512, 320, 3.0 * (1ull << 30), 18253);
    check(std::count(p.resident[2].begin(), p.resident[2].end(), 1) == 320, "320 slots survive automatic VRAM eviction");
    for (int e = 0; e < 512; ++e) check(p.resident[2][(size_t)e] == (e >= 192), "rare pool keeps hottest experts");

    // A large explicit host budget must obey the same pool floor as VRAM-pressure evictions.
    p = expert_residency(prod, ranks, 512, 320, 59.0 * (1ull << 30), 0);
    check(p.moved >= 59.0 * (1ull << 30), "explicit RAM target reached");
    check(std::count(p.resident[2].begin(), p.resident[2].end(), 1) == 320, "explicit RAM target keeps rare pool floor");

    // Where no floor is hit, placement stays byte-for-byte the former global ranking, including ties.
    for (int run = 0; run < 1000; ++run) {
        std::vector<double> ranks(12);
        for (double& s : ranks) s = rng() % 5;
        std::vector<int> order(12);
        std::iota(order.begin(), order.end(), 0);
        std::stable_sort(order.begin(), order.end(), [&](int a, int b) { return ranks[a] < ranks[b]; });
        p = expert_residency(sizes, ranks, 4, 2, 0, 2);
        for (int i = 0; i < 12; ++i)
            check(p.resident[i/4][i%4] == (i != order[0] && i != order[1]), "unconstrained placement unchanged");
    }
    std::puts("expert_residency_test: PASS");
}
