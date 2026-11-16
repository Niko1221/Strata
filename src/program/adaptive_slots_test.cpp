// src/program/adaptive_slots_test.cpp - the host-side rules of --parallel-recycle (header-only, CPU).
//
//   2. a value that is not a number is -1 (the caller warns and leaves the feature off), and 0 stays off;
//   1. --parallel-recycle implies --vram-elastic where --vram-elastic can work, and nowhere else;
//   4. the grow target covers its own slot, the queued BGENs, the waiting BCO co-reads, and stops at the cap;
//   5. a co-read may only name a reserved slot, never merely one below the cap.
#include "strata/program/adaptive_slots.hpp"

#include <cstdio>

namespace as = strata::program::adaptive_slots;

namespace {
int g_fail = 0;
void check(bool ok, const char* what) {
    std::printf("  %-84s %s\n", what, ok ? "ok" : "FAIL");
    if (!ok) ++g_fail;
}
}  // namespace

int main() {
    std::printf("adaptive_slots_test\n");
    // 3. --vram-elastic's rule and the implication
    const bool no_remote[3] = {false, false, false};
    const bool one_remote[3] = {false, true, false};
    check(as::vram_elastic_veto(true, false, no_remote, true, false) == nullptr, "one NVIDIA GPU, serve, no helpers: elastic can be on");
    check(as::vram_elastic_veto(false, false, no_remote, true, false) != nullptr, "no --serve: refused");
    check(as::vram_elastic_veto(true, true, no_remote, true, false) != nullptr, "layer split: refused");
    check(as::vram_elastic_veto(true, false, one_remote, true, false) != nullptr, "helper cache: refused");
    check(as::vram_elastic_veto(true, false, no_remote, false, false) != nullptr, "--vram-segment-mib below 64: refused");
    check(as::vram_elastic_veto(true, false, no_remote, true, true) != nullptr, "HIP: refused");
    check(as::implies_vram_elastic(true, false, nullptr), "S>0, no --vram-elastic, no veto: implied");
    check(!as::implies_vram_elastic(false, false, nullptr), "S=0: not implied (off stays off)");
    check(!as::implies_vram_elastic(true, true, nullptr), "an explicit --vram-elastic is not 'implied'");
    check(!as::implies_vram_elastic(true, false, as::vram_elastic_veto(true, true, no_remote, true, false)),
          "S>0 on a layer split: the veto keeps the old refusal");

    // 4. the grow target
    check(as::grow_target(0, 0, 0, 8) == 1, "BGEN 0: one slot");
    check(as::grow_target(2, 4, 0, 8) == 4, "BGEN 2 with two queued: the whole burst in one grow");
    check(as::grow_target(0, 0, 6, 8) == 6, "a waiting BCO for slot 5 (a need of 6) grows for six slots");
    check(as::grow_target(1, 3, 5, 8) == 5, "the largest of BGEN, queued and co-read need wins");
    check(as::grow_target(7, 0, 0, 8) == 8, "never above the --batch cap");
    check(as::grow_target(0, 99, 99, 8) == 8, "co/BGEN needs are clamped to the cap too");

    // 5. the co-read guard
    check(as::co_slot_ok(0, 2) && as::co_slot_ok(1, 2), "a reserved slot is fine");
    check(!as::co_slot_ok(2, 2), "slot == reserved: no session");
    check(!as::co_slot_ok(-1, 2), "a negative slot is never fine");
    check(!as::co_slot_ok(5, 0), "after the idle shrink (reserved 0) no co-read slot exists");

    std::printf("%s\n", g_fail == 0 ? "adaptive_slots_test: all passed" : "adaptive_slots_test: FAILED");
    return g_fail == 0 ? 0 : 1;
}
