// src/core/uma_cache_test.cpp - the RAM reserve of an integrated GPU's auto expert cache (no GPU needed).
#include "strata/core/uma_cache.hpp"

#include <cstdio>

static int fails = 0;
#define CHECK(c) do { if (!(c)) { std::printf("FAIL line %d: %s\n", __LINE__, #c); ++fails; } } while (0)

int main() {
    using namespace strata::core;
    constexpr uint64_t GiB = 1ull << 30;
    // the larger of 8 GiB and 8% of the RAM
    CHECK(uma_desktop_reserve_bytes(32 * GiB, nullptr) == 8 * GiB);
    CHECK(uma_desktop_reserve_bytes(64 * GiB, nullptr) == 8 * GiB);          // 8% = 5.1 GiB
    CHECK(uma_desktop_reserve_bytes(128 * GiB, nullptr) > 10 * GiB);         // 8% = 10.2 GiB
    CHECK(uma_desktop_reserve_bytes(128 * GiB, nullptr) < 11 * GiB);
    CHECK(uma_desktop_reserve_bytes(0, nullptr) == 8 * GiB);                 // RAM unknown: the floor
    // the person's own figure, 0 included
    CHECK(uma_desktop_reserve_bytes(128 * GiB, "0") == 0);
    CHECK(uma_desktop_reserve_bytes(128 * GiB, "4") == 4 * GiB);
    CHECK(uma_desktop_reserve_bytes(128 * GiB, "2.5") == 2 * GiB + GiB / 2);
    CHECK(uma_desktop_reserve_bytes(128 * GiB, "") > 10 * GiB);              // empty: not set
    CHECK(uma_desktop_reserve_bytes(64 * GiB, "x") == 8 * GiB);              // not a number: ignored
    CHECK(uma_desktop_reserve_bytes(64 * GiB, "-3") == 8 * GiB);
    // the room
    CHECK(uma_cache_room_bytes(100 * GiB, 10 * GiB) == 90 * GiB);
    CHECK(uma_cache_room_bytes(10 * GiB, 10 * GiB) == 0);
    CHECK(uma_cache_room_bytes(5 * GiB, 10 * GiB) == 0);
    CHECK(uma_cache_room_bytes(5 * GiB, 0) == 5 * GiB);
    // G16: the Windows HIP floor of the auto cache's reserve is for a discrete card of 15 GiB or more only
    CHECK(wddm_auto_reserve_mib(false, 20 * GiB, 700) == 2560);              // a 20 GB discrete card (#1376): the floor
    CHECK(wddm_auto_reserve_mib(false, 15 * GiB, 700) == 2560);              // 15 GiB: the floor (#1709)
    CHECK(wddm_auto_reserve_mib(false, 16 * GiB - 1, 700) == 2560);          // a 16 GB card reports just under 16
    CHECK(wddm_auto_reserve_mib(false, 8 * GiB, 700) == 700);                // a smaller card keeps its own sizing
    CHECK(wddm_auto_reserve_mib(false, 15 * GiB - 1, 700) == 700);
    CHECK(wddm_auto_reserve_mib(true, 94027ull << 20, 700) == 700);          // a Strix Halo (94,027 MiB pool): no floor
    CHECK(wddm_auto_reserve_mib(true, 20 * GiB, 700) == 700);                // integrated wins whatever the size
    CHECK(wddm_auto_reserve_mib(false, 20 * GiB, 3000) == 3000);             // a larger reserve is never lowered
    CHECK(wddm_auto_reserve_mib(true, 94027ull << 20, 3000) == 3000);
    if (fails == 0) std::printf("uma_cache_test: all checks passed\n");
    return fails == 0 ? 0 : 1;
}
