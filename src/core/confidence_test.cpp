// src/core/confidence_test.cpp - the confidence monitor's math (no GPU, no model pack).
//
//   Built with -DSTRATA_BUILD_CONVERSATION_TESTS=ON; run the binary directly or `ctest -R confidence_test`.
#include "strata/core/confidence.hpp"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>

namespace {
int failures = 0;

void near(const char* what, double got, double want, double tol = 1e-6) {
    if (std::fabs(got - want) > tol) {
        std::printf("FAIL %s: got %.9f want %.9f\n", what, got, want);
        ++failures;
    }
}

void token_confidence_works() {
    const float flat[4] = {0, 0, 0, 0};                    // P = 1/4 each: T = log 4
    near("flat row (k=4)", strata::core::token_confidence(flat, 4, 4), std::log(4.0));
    const float peak[4] = {100, 0, 0, 0};                  // top-2: log P = ~0 and ~-100: T = 50
    near("peaked row (k=2)", strata::core::token_confidence(peak, 4, 2), 50.0, 1e-3);
    const float peak4[4] = {100, 0, 0, 0};                 // k clamped to the vocab: (0 - 100*3) / 4 negated = 75
    near("peaked row (k=8 clamped)", strata::core::token_confidence(peak4, 4, 8), 75.0, 1e-3);
    near("empty row", strata::core::token_confidence(nullptr, 0, 20), 0.0);
}

void pooling_and_slope_work() {
    // a sequence shorter than one window: its minimum, and a one-point fit has no slope
    std::vector<double> t = {3, 1, 2};
    auto pool = strata::core::pooled_bottom(t, 64, 8, 0.10);
    if (pool != std::vector<double>{1.0}) {
        std::printf("FAIL short sequence pooling\n");
        ++failures;
    }
    near("slope of one point", strata::core::fit_slope(pool), 0.0);
    // a strictly decreasing sequence with bottom = 100%: the pooled minima fall by 8 per stride
    std::vector<double> down;
    for (int i = 0; i < 100; ++i) down.push_back(100.0 - i);
    auto mins = strata::core::pooled_bottom(down, 64, 8, 1.0);     // [37, 29, 21, 13, 5]
    if (mins.size() != 5) {
        std::printf("FAIL pooled size: %zu\n", mins.size());
        ++failures;
    }
    near("pooled slope (falling)", strata::core::fit_slope(mins), -8.0);
    std::vector<double> up = {0, 1, 2, 3};
    near("rising slope", strata::core::fit_slope(up), 1.0);
    near("falling slope", strata::core::fit_slope({3, 2, 1, 0}), -1.0);
}

void summary_works() {
    // a flat line: mean log(2) (two equal tokens), slope 0
    std::vector<double> t(200, std::log(2.0));
    auto s = strata::core::confidence_summary(t);
    if (s.tokens != 200) {
        std::printf("FAIL summary tokens: %lld\n", (long long) s.tokens);
        ++failures;
    }
    near("summary mean", s.mean, std::log(2.0));
    near("summary slope (flat)", s.slope, 0.0);
    auto empty = strata::core::confidence_summary({});
    if (empty.tokens != 0 || empty.mean != 0.0 || empty.slope != 0.0) {
        std::printf("FAIL empty summary\n");
        ++failures;
    }
}
}  // namespace

int main() {
    token_confidence_works();
    pooling_and_slope_work();
    summary_works();
    if (failures) {
        std::printf("confidence_test: %d failures\n", failures);
        return 1;
    }
    std::printf("confidence_test: PASS\n");
    return 0;
}
