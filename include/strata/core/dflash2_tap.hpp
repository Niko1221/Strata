// include/strata/core/dflash2_tap.hpp - the rule record_window applies when it decides whether a layer's completed
// output goes into the DFlash2 drafter's device tap buffer (Verifier::tap_dev_, src/core/verify.cpp).
#pragma once

#include <cstdint>
#include <vector>

namespace strata::core {

/// Whether this build compiles the DFlash2 tap kernel (strata::kernels::df2_tap_gr): the CUDA and HIP builds do,
/// the Metal and SYCL builds do not (the same test as the include of strata/kernels/dflash2.hpp).
inline constexpr bool dflash2_kernels_in_build =
#if defined(STRATA_HAVE_DFLASH2)
    true;
#else
    false;
#endif

/// `taps_on`: Verifier::tap_on_ (set_taps or STRATA_TAP_FILE), `batch_rec`: the recorded window is a batch window
/// (capture_batch).  The layer's completed output is captured when taps are on and the layer is one of the selected
/// tap layers (tap_slot[l] is its slot in the tapped layers, >= 0).  A batch record is not a reason to suppress the
/// capture: its rows are the window's rows in the same row-major order the tap_dev_ allocation sizes for
/// ((row * n_taps + slot) * n_embd).  Only where the build has no DFlash2 tap kernel does a batch record capture
/// nothing: its per-row fallback feeds the STRATA_TAP_FILE research dump, which stays serial-only.
inline bool dflash2_tap_capture(bool taps_on, bool batch_rec, int64_t l, const std::vector<int>& tap_slot,
                                bool kernels_in_build = dflash2_kernels_in_build) {
    if (!taps_on) return false;
    if (l < 0 || l >= (int64_t) tap_slot.size()) return false;
    if (tap_slot[(size_t) l] < 0) return false;
    return !batch_rec || kernels_in_build;
}

/// Whether `Verifier::init` turns on DFlash2 tap capture for a stage `[lb, le)` of a `n_layers`-layer model.
/// A single-card verifier (`lb == 0 && le == n_layers`) captures for `set_taps` (`has_tap_req`) or `STRATA_TAP_FILE`
/// (`has_tap_file`).  A layer-split stage (`lb > 0 || le < n_layers`) captures when `set_taps` gave it a shared
/// portable mapped buffer (`shared_tap_dev`): each stage writes only the tap slots whose layers fall inside its own
/// `[lb, le)` slice, and the last stage hands the completed row-major `[row][slot][n_embd]` buffer to the drafter.
inline bool dflash2_stage_taps_active(bool has_tap_req, bool has_tap_file, int64_t lb, int64_t le, int64_t n_layers,
                                      bool shared_tap_dev, bool kernels_in_build = dflash2_kernels_in_build) {
    if (n_layers <= 0 || lb < 0 || le <= lb || le > n_layers) return false;
    if (lb == 0 && le == n_layers) return has_tap_req || has_tap_file;
    return has_tap_req && shared_tap_dev && kernels_in_build;
}

/// Element offset into the row-major `[row][n_taps][n_embd]` tap buffer for `row` and tap `slot`.
inline size_t dflash2_tap_offset(int64_t row, int slot, size_t n_taps, int64_t n_embd) {
    if (row < 0 || slot < 0 || n_taps == 0 || n_embd <= 0) return 0;
    return ((size_t) row * n_taps + (size_t) slot) * (size_t) n_embd;
}

/// Starting row in `Prefill::tap_dev_` for a chunk at position `p0` whose tapped suffix begins at chunk offset `tap_t0`.
/// Single-stage prefill reuses row 0 for each chunk (`on_taps` consumes it before the next chunk starts); a layer-split
/// prefill indexes by `(p0 + tap_t0) - tap_first` so pipelined stages never overwrite an earlier chunk's taps in flight.
inline int64_t dflash2_prefill_tap_row0(int64_t p0, int64_t tap_t0, int64_t tap_first, bool shared_tap_dev) {
    if (!shared_tap_dev) return 0;
    const int64_t pos = p0 + tap_t0;
    return pos > tap_first ? pos - tap_first : 0;
}

}  // namespace strata::core
