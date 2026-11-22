// engine/tests/core/riemann_rh_test.cpp
// Unit tests for RH-conditional primitives in strata/math/riemann_rh.hpp.
#include "strata/math/riemann_rh.hpp"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>

namespace {

void check(bool cond, const char* msg) {
    if (!cond) {
        std::fprintf(stderr, "FAIL: %s\n", msg);
        std::exit(1);
    }
}

}  // namespace

int main() {
    using namespace strata::math;

    // 1. Verify exact Liouville values lambda(n) = (-1)^{Omega(n)} for n = 1..16.
    const int expected_lambda[17] = {
        0,  // unused
        1, -1, -1, 1, -1, 1, -1, -1, 1, 1, -1, -1, -1, 1, 1, 1
    };
    for (int n = 1; n <= 16; ++n) {
        check(liouville_sign(n) == expected_lambda[n], "liouville_sign exact values 1..16");
    }

    // 2. Verify RH-consistent square-root cancellation |L(x)| <= 2 * sqrt(x) on 1..256.
    int partial = 0;
    for (int n = 1; n <= 256; ++n) {
        partial += (int) liouville_sign(n);
        check(std::abs(partial) <= 2.0 * std::sqrt((double) n) + 1.0, "Liouville partial sum sqrt bound");
    }

    // 3. Verify critical-line decay monotonicity and bounds.
    double prev = 0.0;
    for (uint64_t step : {0ull, 1ull, 4ull, 16ull, 64ull, 256ull, 1024ull}) {
        const double d = rh_critical_decay(step);
        check(d >= 0.50 && d <= 0.96, "rh_critical_decay bounds");
        check(d >= prev, "rh_critical_decay monotone");
        prev = d;
    }

    // 4. Verify von Koch hysteresis threshold scaling.
    const double t1 = rh_von_koch_threshold(10.0, 2.0, 1);
    const double t2 = rh_von_koch_threshold(10.0, 2.0, 100);
    check(t1 > 0.0 && t2 > 0.0 && t2 < t1, "rh_von_koch_threshold decreases with log(step)");

    // 5. Verify Zeta-Lagrangian allocator beats equal allocation on heterogeneous layers.
    std::vector<ZetaLayerSpec> specs = {
        {0, 4800000, 1.30, 1.15, 128},  // Q8_0 heavy CPU layer
        {1, 4800000, 1.30, 1.15, 128},
        {2, 4800000, 1.00, 1.15, 128},  // Q5_0 lighter CPU layer
        {3, 4800000, 1.00, 1.15, 128},
    };
    const size_t budget = 24ULL * 4800000ULL;
    size_t used = 0;
    const auto slots = zeta_lagrangian_allocate(specs, budget, &used);
    check(used == budget, "zeta_lagrangian_allocate uses full budget");
    check(slots[0] > slots[2] && slots[1] > slots[3], "heavier CPU layers get more VRAM slots");

    auto total_saved = [&](const std::vector<int>& s) {
        double sum = 0.0;
        for (size_t i = 0; i < specs.size(); ++i) {
            const double inv_z = 1.0 / truncated_zeta(specs[i].max_slots, specs[i].zeta_exponent);
            for (int k = 1; k <= s[i]; ++k) {
                sum += specs[i].cpu_miss_cost * std::pow((double) k, -specs[i].zeta_exponent) * inv_z;
            }
        }
        return sum;
    };
    const std::vector<int> equal_slots = {6, 6, 6, 6};
    check(total_saved(slots) > total_saved(equal_slots), "zeta_lagrangian_allocate strictly beats equal split");

    std::printf("riemann_rh_test: ALL PASS (zeta_saved=%.6f > equal_saved=%.6f, slots=[%d,%d,%d,%d])\n",
                total_saved(slots), total_saved(equal_slots), slots[0], slots[1], slots[2], slots[3]);
    return 0;
}
