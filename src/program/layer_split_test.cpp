// src/program/layer_split_test.cpp - the layer split's preparation: what the host arena pins.
//
//   1. the arena is pinned whole unless WDDM maps it into several contexts (then 8 GiB, as measured on the 5080 +
//      3090 rig); one GPU or Linux: no cap.
#include "strata/program/layer_split.hpp"

#include <cstdint>
#include <cstdio>

namespace ls = strata::program::layer_split;

namespace {
int g_fail = 0;
void check(bool ok, const char* what) {
    std::printf("  %-74s %s\n", what, ok ? "ok" : "FAIL");
    if (!ok) ++g_fail;
}
constexpr uint64_t GiB = 1ull << 30;
}  // namespace

int main() {
    std::printf("layer_split_test\n");
    {
        check(ls::arena_pin_cap(false, true) == 0, "one context, WDDM: the whole arena");
        check(ls::arena_pin_cap(false, false) == 0, "one context, Linux: the whole arena");
        check(ls::arena_pin_cap(true, true) == 8 * GiB, "several contexts under WDDM: 8 GiB");
        check(ls::arena_pin_cap(true, false) == 0, "several contexts, Linux: the whole arena");
    }
    std::printf(g_fail ? "layer_split_test: %d FAILED\n" : "layer_split_test: all passed\n", g_fail);
    return g_fail ? 1 : 0;
}
