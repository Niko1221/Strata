// src/core/sm75_series_test.cpp - the Turing series is on by default since 0.2.23 (#7041, #7076): STRATA_SM75_PERF unset or non-zero is on, =0 is off.
#include "strata/core/sm75_series.hpp"

#include <cstdio>

static int fails = 0;
#define CHECK(c) do { if (!(c)) { std::printf("FAIL line %d: %s\n", __LINE__, #c); ++fails; } } while (0)

int main() {
    using strata::sm75_series_from;
    CHECK(sm75_series_from(nullptr));      // the default: on
    CHECK(sm75_series_from("1"));
    CHECK(sm75_series_from("2"));
    CHECK(!sm75_series_from("0"));         // the way back to the 0.2.22 behaviour
    CHECK(!sm75_series_from(""));
    // only a compute capability 7.5 card is ever affected
    CHECK(strata::is_sm75(7, 5));
    CHECK(!strata::is_sm75(7, 0));
    CHECK(!strata::is_sm75(8, 6));
    CHECK(!strata::is_sm75(12, 0));
    CHECK(!strata::sm75_default(8, 6));
    CHECK(!strata::sm75_default(8, 9));
    CHECK(!strata::sm75_switch("STRATA_SM75_TEST_UNSET", 8, 6));
    if (fails == 0) std::printf("sm75_series_test: ok\n");
    return fails == 0 ? 0 : 1;
}
