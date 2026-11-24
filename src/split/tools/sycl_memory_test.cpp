#include "../src/sycl_memory.hpp"

#include <cstdio>
#include <cstdlib>
#include <cstdint>

using namespace fnx;

static void check(bool ok, const char* what) {
    if (!ok) { std::fprintf(stderr, "FAIL: %s\n", what); std::exit(1); }
}
template <typename F> static void refuses(F f) {
    try { f(); } catch (const std::runtime_error&) { return; }
    check(false, "impossible or invalid placement accepted");
}

int main() {
    const std::vector<std::vector<std::pair<size_t, size_t>>> sizes{
        {{3,2}, {4,3}, {3,2}}, {{6,4}, {8,6}, {6,4}}};
    const std::vector<double> scores{10,11,12,13,0,1,2,3,14,15,16,17};
    auto p = sycl_expert_placement(sizes, scores, 4, 2, {{68,0,0}, {136,0,0}});
    check(!p.evicted && p.device_bytes == std::vector<size_t>{68,136}, "exact fit keeps all experts");
    p = sycl_expert_placement(sizes, scores, 4, 2, {{68,0,0}, {100,14,18}});
    check(p.device_bytes == std::vector<size_t>{34,68} &&
          p.nonresident_bytes == std::vector<size_t>{34,68}, "tightest card controls shared placement");
    check(p.host_bytes == std::vector<size_t>{68,136}, "RAM budget includes resident experts' host copies");
    check(p.resident == std::vector<std::vector<char>>{{0,0,0,0}, {0,0,1,1}, {1,1,1,1}},
          "rare pool keeps its floor and hotter experts survive");
    refuses([&] { sycl_expert_placement(sizes, scores, 4, 2, {{23,0,0}, {136,0,0}}); });
    refuses([&] { sycl_expert_placement(sizes, scores, 4, 2, {{1,1,1}, {136,0,0}}); });
    refuses([&] { sycl_expert_placement(sizes, {0}, 4, 2, {{68,0,0}, {136,0,0}}); });
    refuses([&] {
        auto bad = scores;
        bad[0] = std::numeric_limits<double>::quiet_NaN();
        sycl_expert_placement(sizes, bad, 4, 2, {{68,0,0}, {136,0,0}});
    });
    refuses([&] {
        sycl_expert_placement({{{SIZE_MAX,1}}, {{1,1}}}, {0,1}, 2, 1, {{SIZE_MAX,0,0}, {8,0,0}});
    });
    // Equal first-card sizes must not merge distinct second-card pools.
    refuses([&] {
        sycl_expert_placement({{{1,1},{1,1}}, {{2,1},{1,2}}}, {0,0,0,0}, 2, 2,
                             {{8,0,0}, {11,0,0}});
    });
    std::puts("SYCL split memory placement: PASS");
}
