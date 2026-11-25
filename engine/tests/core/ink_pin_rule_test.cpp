// engine/tests/core/ink_pin_rule_test.cpp - the pinned host arena rule of Inkling 2 Mini (cap, floor, override, fallback). No GPU.
#include <cstdio>
#include <cstring>

#include "strata/core/ink_pin_rule.hpp"

using strata::ink_pin_rule;

static int fails = 0;
#define CHECK(c) do { if (!(c)) { std::printf("FAIL line %d: %s\n", __LINE__, #c); ++fails; } } while (0)

int main() {
    constexpr uint64_t GiB = 1ull << 30;
    // Windows, default: cap = min(16 GiB, 35% of the available RAM), and 8 GiB must stay available
    CHECK(ink_pin_rule(true, 8 * GiB, 64 * GiB, nullptr).pin);                    // 8 of 64: cap min(16, 22.4) = 16
    CHECK(ink_pin_rule(true, 16 * GiB, 64 * GiB, nullptr).pin);                   // exactly the cap
    CHECK(!ink_pin_rule(true, 16 * GiB + 1, 64 * GiB, nullptr).pin);              // over 16 GiB
    CHECK(ink_pin_rule(true, 8 * GiB, 24 * GiB, nullptr).pin);                    // 35% of 24 = 8.4
    CHECK(!ink_pin_rule(true, 9 * GiB, 24 * GiB, nullptr).pin);                   // over 35%
    CHECK(!ink_pin_rule(true, 8 * GiB, 20 * GiB, nullptr).pin);                   // 35% of 20 = 7: over
    CHECK(ink_pin_rule(true, 14 * GiB, 42 * GiB, nullptr).pin);                   // Q4_K_XL-like arena, 35% of 42 = 14.7, leaves 28
    CHECK(!ink_pin_rule(true, 14 * GiB, 39 * GiB, nullptr).pin);                  // 35% of 39 = 13.65: over
    CHECK(ink_pin_rule(true, 8 * GiB, 0, nullptr).pin);                           // available RAM unknown: up to 8 GiB
    CHECK(!ink_pin_rule(true, 8 * GiB + 1, 0, nullptr).pin);
    // the floor: with the override the 35% share no longer applies, the 8 GiB that must remain does
    CHECK(!ink_pin_rule(true, 10 * GiB, 17 * GiB, "16").pin);                     // leaves 7 GiB
    CHECK(ink_pin_rule(true, 10 * GiB, 18 * GiB, "16").pin);                      // leaves exactly 8 GiB
    // override: n GiB, 0 = off
    CHECK(!ink_pin_rule(true, 8 * GiB + 1, 64 * GiB, "8").pin);                   // lower cap than the default
    CHECK(ink_pin_rule(true, 8 * GiB, 64 * GiB, "8").pin);
    CHECK(ink_pin_rule(true, 20 * GiB, 64 * GiB, "24").pin);                      // higher cap than the default (35% of 64 = 22.4 would also allow: 20 < 22.4)
    CHECK(ink_pin_rule(true, 20 * GiB, 64 * GiB, "20").pin);
    CHECK(!ink_pin_rule(true, 1 * GiB, 64 * GiB, "0").pin);
    CHECK(!ink_pin_rule(false, 1 * GiB, 64 * GiB, "0").pin);
    CHECK(std::strstr(ink_pin_rule(true, 1 * GiB, 64 * GiB, "0").why, "off") != nullptr);
    CHECK(std::strstr(ink_pin_rule(true, 20 * GiB, 64 * GiB, nullptr).why, "16 GiB") != nullptr);
    CHECK(std::strstr(ink_pin_rule(true, 10 * GiB, 17 * GiB, nullptr).why, "35%") != nullptr || std::strstr(ink_pin_rule(true, 10 * GiB, 17 * GiB, nullptr).why, "8 GiB") != nullptr);
    // Linux keeps its rule: under 45% of the available RAM, no size cap, no floor
    CHECK(ink_pin_rule(false, 20 * GiB, 64 * GiB, nullptr).pin);                  // 31% of 64
    CHECK(ink_pin_rule(false, 28 * GiB, 64 * GiB, nullptr).pin);                  // 43.75%
    CHECK(!ink_pin_rule(false, 30 * GiB, 64 * GiB, nullptr).pin);                 // 46.9%
    CHECK(ink_pin_rule(false, 100 * GiB, 0, nullptr).pin);                        // unknown: always
    CHECK(ink_pin_rule(false, 15 * GiB, 20 * GiB - 1, nullptr).pin == false);     // 75%: no
    CHECK(!ink_pin_rule(false, 9 * GiB, 64 * GiB, "8").pin);                      // the override as a cap in GiB applies on Linux too
    CHECK(ink_pin_rule(false, 8 * GiB, 10 * GiB, "8").pin);                       // and replaces the 45% share (no floor on Linux)
    if (fails == 0) std::printf("ink_pin_rule_test: ok\n");
    return fails == 0 ? 0 : 1;
}
