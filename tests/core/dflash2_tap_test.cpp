// tests/core/dflash2_tap_test.cpp - the DFlash2 tap-capture rule of record_window (src/core/verify.cpp): a layer's
// completed output is written into the drafter's device tap buffer when taps are on and the layer is one of the
// selected tap layers, in a single-sequence window and in a batch window alike.  A build without the DFlash2 tap
// kernels (Metal, SYCL) keeps a batch window uncaptured: the per-row fallback there feeds the STRATA_TAP_FILE
// research dump, which stays serial-only.
#include "strata/core/dflash2_tap.hpp"

#include <cstdio>
#include <string>
#include <vector>

using strata::core::dflash2_tap_capture;
using strata::core::dflash2_stage_taps_active;
using strata::core::dflash2_kernels_in_build;

namespace {
int g_fail = 0;
long g_checks = 0;
void check(bool ok, const std::string& what) {
    ++g_checks;
    if (ok) return;
    std::fprintf(stderr, "FAIL: %s\n", what.c_str());
    ++g_fail;
}
}  // namespace

int main() {
    // tap_slot_: layer -> slot in the tapped layers, -1 for an untapped layer, as Verifier::init fills it (5,15,25,35,45)
    std::vector<int> tap_slot(48, -1);
    for (int i = 0; i < 5; ++i) tap_slot[5 + 10 * i] = i;

    // taps off: nothing captured, either record kind, whatever the layer
    check(!dflash2_tap_capture(false, false, 15, tap_slot), "taps off: a single window captures nothing");
    check(!dflash2_tap_capture(false, true, 15, tap_slot), "taps off: a batch window captures nothing");
    check(!dflash2_tap_capture(false, false, 5, tap_slot), "taps off: even a selected layer captures nothing");

    // taps on, a layer not among the selected taps or outside tap_slot: no capture, either record kind
    check(!dflash2_tap_capture(true, false, 14, tap_slot), "taps on: an untapped layer captures nothing (single)");
    check(!dflash2_tap_capture(true, true, 14, tap_slot), "taps on: an untapped layer captures nothing (batch)");
    check(!dflash2_tap_capture(true, false, 48, tap_slot), "a layer at or past tap_slot's size captures nothing (single)");
    check(!dflash2_tap_capture(true, true, 48, tap_slot), "a layer at or past tap_slot's size captures nothing (batch)");
    check(!dflash2_tap_capture(true, false, -1, tap_slot), "layer -1 captures nothing (single)");
    check(!dflash2_tap_capture(true, true, -1, tap_slot), "layer -1 captures nothing (batch)");

    // taps on, selected layer: captured in both record kinds once the build has the DFlash2 tap kernels
    check(dflash2_tap_capture(true, false, 15, tap_slot), "taps on, selected layer: captured in a single window");
    for (int l : {5, 15, 25, 35, 45}) {
        check(dflash2_tap_capture(true, false, l, tap_slot, true), "every selected layer is captured in a single window (layer " + std::to_string(l) + ")");
        check(dflash2_tap_capture(true, true, l, tap_slot, true), "every selected layer is captured in a batch window (layer " + std::to_string(l) + ")");
    }

    // a build without the DFlash2 tap kernels: the single-sequence path stays (the research dump reads it); a batch
    // window stays uncaptured, so the STRATA_TAP_FILE path keeps serial-only semantics
    check(dflash2_tap_capture(true, false, 15, tap_slot, false), "no kernels in the build: a single window still captures (the file dump)");
    check(!dflash2_tap_capture(true, true, 15, tap_slot, false), "no kernels in the build: a batch window captures nothing (the file dump is serial-only)");

    // the default kernels_in_build follows the build's STRATA_HAVE_DFLASH2, so the expectation is written against the
    // same constant the header reads: the test says the same thing whether or not it is compiled in a CUDA/HIP build
    check(dflash2_tap_capture(true, true, 15, tap_slot) == dflash2_kernels_in_build,
          "the default follows the build: a batch window captures exactly when the build has the DFlash2 tap kernels");
    check(dflash2_tap_capture(true, false, 15, tap_slot), "the default follows the build: a single window is not affected");

    // layer-split stage tap activation (Verifier::init):
    check(!dflash2_stage_taps_active(false, false, 0, 48, 48, false, true), "no tap_req and no tap_file: inactive");
    check(dflash2_stage_taps_active(true, false, 0, 48, 48, false, true), "single-stage [0, n_layers): active without shared buffer");
    check(dflash2_stage_taps_active(false, true, 0, 48, 48, false, false), "single-stage tap_file: active even without kernels");
    check(!dflash2_stage_taps_active(true, false, 0, 24, 48, false, true), "partial stage [0, 24) without shared buffer: inactive");
    check(!dflash2_stage_taps_active(true, false, 24, 48, 48, false, true), "partial stage [24, 48) without shared buffer: inactive");
    check(dflash2_stage_taps_active(true, false, 0, 24, 48, true, true), "partial stage [0, 24) with shared buffer and kernels: active");
    check(dflash2_stage_taps_active(true, false, 24, 48, 48, true, true), "partial stage [24, 48) with shared buffer and kernels: active");
    check(!dflash2_stage_taps_active(true, false, 0, 24, 48, true, false), "partial stage with shared buffer but no kernels: inactive");
    check(!dflash2_stage_taps_active(false, true, 0, 24, 48, true, true), "partial stage with tap_file only (no tap_req): inactive");

    std::printf("dflash2_tap_test: %ld checks, %s\n", g_checks, g_fail ? "FAIL" : "PASS");
    return g_fail ? 1 : 0;
}
