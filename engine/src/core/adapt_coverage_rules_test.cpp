// src/core/adapt_coverage_rules_test.cpp - the mid-coverage adaptive defaults are opt-in (STRATA_ADAPT_MID=1); no GPU needed.
#include "strata/core/adapt_coverage_rules.hpp"

#include <cstdio>
#include <initializer_list>

static int fails = 0;
#define CHECK(c) do { if (!(c)) { std::printf("FAIL line %d: %s\n", __LINE__, #c); ++fails; } } while (0)

int main() {
    using namespace strata::core;
    // not set / not "1": the 0.1.65 defaults (every 4, 96 swaps, decay 0.70) stay in every band
    for (const double cov : {0.05, 0.15, 0.188, 0.325, 0.468, 0.65, 0.9}) {
        CHECK(!adapt_mid_coverage_applies(cov, nullptr));
        CHECK(!adapt_mid_coverage_applies(cov, ""));
        CHECK(!adapt_mid_coverage_applies(cov, "0"));
        CHECK(!adapt_mid_coverage_applies(cov, "11"));
    }
    CHECK(!adapt_mid_coverage_enabled(nullptr));
    CHECK(!adapt_mid_coverage_enabled("0"));
    CHECK(!adapt_mid_coverage_enabled("on"));
    CHECK(adapt_mid_coverage_enabled("1"));
    // STRATA_ADAPT_MID=1: the 15%..65% band, both ends included, nothing outside it
    CHECK(adapt_mid_coverage_applies(0.15, "1"));
    CHECK(adapt_mid_coverage_applies(0.188, "1"));
    CHECK(adapt_mid_coverage_applies(0.325, "1"));
    CHECK(adapt_mid_coverage_applies(0.468, "1"));
    CHECK(adapt_mid_coverage_applies(0.65, "1"));
    CHECK(!adapt_mid_coverage_applies(0.149, "1"));
    CHECK(!adapt_mid_coverage_applies(0.651, "1"));
    CHECK(!adapt_mid_coverage_applies(0.0, "1"));
    CHECK(!adapt_mid_coverage_applies(1.0, "1"));
    if (fails == 0) std::printf("adapt_coverage_rules_test: OK\n");
    return fails == 0 ? 0 : 1;
}
