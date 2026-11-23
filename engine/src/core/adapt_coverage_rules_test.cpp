// src/core/adapt_coverage_rules_test.cpp - the mid-coverage adaptive defaults: on for CUDA, off for HIP, STRATA_ADAPT_MID=1/0 forces; no GPU needed.
#include "strata/core/adapt_coverage_rules.hpp"

#include <cstdio>
#include <initializer_list>

static int fails = 0;
#define CHECK(c) do { if (!(c)) { std::printf("FAIL line %d: %s\n", __LINE__, #c); ++fails; } } while (0)

int main() {
    using namespace strata::core;
#if defined(STRATA_USE_HIP)
    CHECK(!adapt_mid_default_on());
#else
    CHECK(adapt_mid_default_on());
#endif
    // HIP default (off): the 0.1.65 values (every 4, 96 swaps, decay 0.70) stay in every band unless STRATA_ADAPT_MID=1
    for (const double cov : {0.05, 0.15, 0.188, 0.325, 0.468, 0.65, 0.9}) {
        CHECK(!adapt_mid_coverage_applies(cov, nullptr, false));
        CHECK(!adapt_mid_coverage_applies(cov, "", false));
        CHECK(!adapt_mid_coverage_applies(cov, "0", false));
        CHECK(!adapt_mid_coverage_applies(cov, "11", false));
        CHECK(!adapt_mid_coverage_applies(cov, "on", false));
    }
    CHECK(adapt_mid_coverage_applies(0.15, "1", false));
    CHECK(adapt_mid_coverage_applies(0.188, "1", false));
    CHECK(adapt_mid_coverage_applies(0.325, "1", false));
    CHECK(adapt_mid_coverage_applies(0.468, "1", false));
    CHECK(adapt_mid_coverage_applies(0.65, "1", false));
    // CUDA default (on): the band 15%..65%, both ends included, nothing outside; STRATA_ADAPT_MID=0 turns it off
    CHECK(adapt_mid_coverage_applies(0.15, nullptr, true));
    CHECK(adapt_mid_coverage_applies(0.30, nullptr, true));
    CHECK(adapt_mid_coverage_applies(0.65, "", true));
    CHECK(adapt_mid_coverage_applies(0.30, "1", true));
    CHECK(!adapt_mid_coverage_applies(0.30, "0", true));
    CHECK(!adapt_mid_coverage_applies(0.149, nullptr, true));
    CHECK(!adapt_mid_coverage_applies(0.651, nullptr, true));
    CHECK(!adapt_mid_coverage_applies(0.0, "1", true));
    CHECK(!adapt_mid_coverage_applies(1.0, "1", true));
    if (fails == 0) std::printf("adapt_coverage_rules_test: OK\n");
    return fails == 0 ? 0 : 1;
}
