// src/core/dot_x4_policy_test.cpp - which calls of the native expert path use the 4-row dot kernels (#6552, #7075): the default follows the token count.
#include "strata/quant/dot_x4_policy.hpp"

#include <cstdio>

static int fails = 0;
#define CHECK(c) do { if (!(c)) { std::printf("FAIL line %d: %s\n", __LINE__, #c); ++fails; } } while (0)

int main() {
    using strata::quant::dot_x4_enabled;
    using strata::quant::kDotX4MinTokens;
    // unset: off for one token (solo decode, cold rows), on from kDotX4MinTokens (batched windows, prompt rows)
    CHECK(!dot_x4_enabled(nullptr, nullptr, 1));
    CHECK(dot_x4_enabled(nullptr, nullptr, kDotX4MinTokens));
    CHECK(dot_x4_enabled(nullptr, nullptr, 2));
    CHECK(dot_x4_enabled(nullptr, nullptr, 64));
    CHECK(kDotX4MinTokens == 2);
    // STRATA_DOT_X4=1: on for every token count (the 0.2.21 / 0.2.22 opt-in)
    CHECK(dot_x4_enabled("1", nullptr, 1));
    CHECK(dot_x4_enabled("1", nullptr, 8));
    // STRATA_DOT_X4=0: off for every token count (exactly the 0.2.21 / 0.2.22 default)
    CHECK(!dot_x4_enabled("0", nullptr, 1));
    CHECK(!dot_x4_enabled("0", nullptr, 8));
    // anything else is "not set"
    CHECK(!dot_x4_enabled("", nullptr, 1));
    CHECK(dot_x4_enabled("", nullptr, 2));
    CHECK(!dot_x4_enabled("11", nullptr, 1));
    CHECK(dot_x4_enabled("11", nullptr, 4));
    // STRATA_NO_DOT_X4 set (even empty) wins over everything, as before
    CHECK(!dot_x4_enabled("1", "", 1));
    CHECK(!dot_x4_enabled("1", "1", 8));
    CHECK(!dot_x4_enabled(nullptr, "1", 8));
    CHECK(!dot_x4_enabled(nullptr, "", 64));
    // the mode, read once by the caller
    CHECK(strata::quant::dot_x4_mode(nullptr, nullptr) == -1);
    CHECK(strata::quant::dot_x4_mode("1", nullptr) == 1);
    CHECK(strata::quant::dot_x4_mode("0", nullptr) == 0);
    CHECK(strata::quant::dot_x4_mode("1", "1") == 0);
    if (fails == 0) std::printf("dot_x4_policy_test: ok\n");
    return fails == 0 ? 0 : 1;
}
