// src/core/dot_x4_policy_test.cpp - the 4-row dot kernels of the native expert path are opt-in (#6552).
#include "strata/quant/dot_x4_policy.hpp"

#include <cstdio>

static int fails = 0;
#define CHECK(c) do { if (!(c)) { std::printf("FAIL line %d: %s\n", __LINE__, #c); ++fails; } } while (0)

int main() {
    using strata::quant::dot_x4_enabled;
    CHECK(!dot_x4_enabled(nullptr, nullptr));    // the default: off
    CHECK(dot_x4_enabled("1", nullptr));          // STRATA_DOT_X4=1
    CHECK(!dot_x4_enabled("0", nullptr));
    CHECK(!dot_x4_enabled("", nullptr));
    CHECK(!dot_x4_enabled("11", nullptr));
    CHECK(!dot_x4_enabled("1", ""));              // STRATA_NO_DOT_X4 set (even empty) wins, as before
    CHECK(!dot_x4_enabled("1", "1"));
    CHECK(!dot_x4_enabled(nullptr, "1"));
    if (fails == 0) std::printf("dot_x4_policy_test: ok\n");
    return fails == 0 ? 0 : 1;
}
