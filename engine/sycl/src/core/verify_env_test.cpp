// sycl/src/core/verify_env_test.cpp - #6250 / #4698: STRATA_VERIFY_NO_HOST=1 alone gets the device-built plan (strata::no_host_wants_device_plan).  Host only:
// no queue and no GPU.
//   verify_env_test
#include "strata/core/verify_env.hpp"

#include <cstdio>

static int g_fail = 0;
static void check(bool ok, const char* what) { std::printf("  %-100s %s\n", what, ok ? "ok" : "FAIL"); if (!ok) ++g_fail; }

int main() {
    using strata::no_host_wants_device_plan;
    check(no_host_wants_device_plan("1", nullptr), "NO_HOST=1, DEVICE_PLAN not set: the engine sets the device plan (#6250)");
    check(no_host_wants_device_plan("true", nullptr), "NO_HOST=true (any value but 0), DEVICE_PLAN not set: the device plan");
    check(!no_host_wants_device_plan("1", "1"), "both set: nothing to do");
    check(!no_host_wants_device_plan("1", "0"), "DEVICE_PLAN=0 set by the caller: kept");
    check(!no_host_wants_device_plan("0", nullptr), "NO_HOST=0 (an i915 card): nothing");
    check(!no_host_wants_device_plan("", nullptr), "NO_HOST empty: nothing");
    check(!no_host_wants_device_plan(nullptr, nullptr), "neither set: nothing");
    check(!no_host_wants_device_plan(nullptr, "1"), "DEVICE_PLAN alone: nothing");
    using strata::arena_refuses_no_host;
    check(arena_refuses_no_host("1", true, 24576, 9114), "#4698: NO_HOST=1, host arena, 9114 of 24576 experts in VRAM: refused");
    check(arena_refuses_no_host("true", true, 24576, 0), "#4698: NO_HOST=true (any value but 0), host arena, none in VRAM: refused");
    check(!arena_refuses_no_host("1", true, 24576, 24576), "#4698: every expert in VRAM: the arena is never read, not refused");
    check(!arena_refuses_no_host("0", true, 24576, 9114), "#4698: NO_HOST=0 (an i915 card, the arena's own mode): not refused");
    check(!arena_refuses_no_host("", true, 24576, 9114), "#4698: NO_HOST empty: not refused");
    check(!arena_refuses_no_host(nullptr, true, 24576, 9114), "#4698: NO_HOST not set: not refused");
    check(!arena_refuses_no_host("1", false, 24576, 9114), "#4698: --stream-experts (the USM mirror answers): not refused");
    std::printf("%s\n", g_fail ? "FAILED" : "all ok");
    return g_fail ? 1 : 0;
}
