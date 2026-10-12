// include/strata/program/kv_splice.hpp - the suffix-reuse K/V splice: a host rotation of K by a
// per-position RoPE delta over the unchanged tail.  V is untouched; the rope is content-invariant
// for V (V carries no position by itself - `include/strata/kernels/rope.hpp` rotates K alone).
//
// T3 (the unit test) needs a `cos`/`sin` table to drive `rope_neox_pair` (host-callable,
// `STRATA_ROPE_HD inline`, include/strata/kernels/rope.hpp:42).  The authoritative `build_rope_table`
// is a CUDA TU at src/kernels/cuda/rope.cu:48-58 and only its DECLARATION (`:47`) lives in the header;
// the body is pure `std::pow`/`std::cos`/`std::sin`, which means a CPU build cannot link the .cu without
// CUDA.  Per A2.2: T3 inlines a host C++ port of that body in this header.  `inline` means every TU
// gets the same instance and there is no link surface.
//
// `rotate_k_by_delta` walks a row of K in pairs `(i, i + n_rot/2)` (NEOX), reading cos/sin at the OLD
// position, RE-applying the rotation to the new position by reading the delta-table, and writing the
// result back in place.  The T3 test asserts the result equals the rotation applied at the NEW
// position from scratch, so any deviation in the cos/sin re-computation would surface.
#pragma once

#include "strata/kernels/rope.hpp"

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <vector>

namespace strata::program::suffix_reuse {

/// Host C++ port of `build_rope_table` (src/kernels/cuda/rope.cu:48-58) - pure `std::pow` / `std::cos`
/// / `std::sin`, float64 throughout in the reference's order (inv, then ang, then cos/sin).  `inline`
/// keeps the ODR happy across translation units that include this header.  Tables are `(max_pos, n_rot/2)`
/// row-major floats, matching the CUDA builder so the `rope_neox_pair` consumer agrees.
inline void build_rope_table_host(int n_rot, double theta, int max_pos, float* cos_tab, float* sin_tab) {
    const int half = n_rot / 2;
    for (int p = 0; p < max_pos; ++p) {
        for (int i = 0; i < half; ++i) {
            const double inv = std::pow(theta, -2.0 * (double) i / (double) n_rot);
            const double ang = (double) p * inv;
            cos_tab[(size_t) p * (size_t) half + (size_t) i] = (float) std::cos(ang);
            sin_tab[(size_t) p * (size_t) half + (size_t) i] = (float) std::sin(ang);
        }
    }
}

/// In-place NEOX partial rotation of one row of K.  `k` is one (head_dim,) row; `n_rot` is the
/// partial-rotation width (only the first `n_rot` of `head_dim` are touched, the tail passes through).
/// The host port of `rope_neox_pair` applies `(c, s)` at the chosen position; this wrapper chooses
/// (c, s) from the table using a per-row position offset.
///
/// `cos_tab` / `sin_tab` are `(max_pos, n_rot/2)` matching the CUDA builder.  `k` is a CACHED K row,
/// i.e. POST-RoPE at `old_pos` - the engine applies RoPE before the cell enters the cache.  We
/// must re-rotate by the DELTA angle `(new_pos - old_pos) * inv(i)` so `R(diag)·R(old)·x = R(new)·x`.
/// Reading the table at `new_pos` (the bug the F2 review caught) would apply `R(new)` on top of the
/// already-rotated cached row, giving `R(new)·R(old)·x = R(old+new)·x` - wrong.
///
/// `max_pos` is the table extent.  When `new_pos` overflows it (long suffix at a very deep position),
/// both `new_pos` and `delta` use a CLAMPED position so the table lookup stays in range - the cached
/// row's rotation at the clamped position approximates the extrapolated angle, which is the
/// pre-existing A2.2 convention the splice layer assumes.
inline void rotate_row_by_delta(float* k, int head_dim, int n_rot, int old_pos, int delta,
                                const float* cos_tab, const float* sin_tab, int max_pos) {
    const int half = n_rot / 2;
    const int new_pos = old_pos + delta;
    if (new_pos < 0) return;   // the rotation domain is nonnegative
    // F2 fix: rotation by the DELTA ANGLE, not by the full new-position angle.  The effective delta
    // is the offset between the clamped new position (where the table can land us) and old_pos:
    //   effective_delta = clamp(new_pos, 0, max_pos-1) - old_pos
    // For pair i the additional rotation angle is `effective_delta * inv(i)`.  When `new_pos` is in
    // range this is just `delta`; when it overflows the table the additional rotation is reduced
    // (capped at `max_pos-1 - old_pos`).  For delta>0 we read the table at the absolute effective
    // delta; for delta<0 we use `cos(-x)=cos(x), sin(-x)=-sin(x)` and flip the sign.
    const int new_pos_clamped = new_pos < max_pos ? new_pos : max_pos - 1;
    const int effective_delta = new_pos_clamped - old_pos;
    const int delta_abs = effective_delta < 0 ? -effective_delta : effective_delta;
    const int p_delta = delta_abs < max_pos ? delta_abs : max_pos - 1;
    const float s_sign = effective_delta < 0 ? -1.0f : 1.0f;
    for (int i = 0; i < half; ++i) {
        const float c = cos_tab[(size_t) p_delta * (size_t) half + (size_t) i];
        const float s = s_sign * sin_tab[(size_t) p_delta * (size_t) half + (size_t) i];
        float a = k[i], b = k[i + half];
        float oa, ob;
        strata::kernels::rope_neox_pair(a, b, c, s, oa, ob);
        k[i] = oa;
        k[i + half] = ob;
    }
    // the tail [n_rot, head_dim) passes through - `include/strata/kernels/rope.hpp:4`
    (void) head_dim;
}

/// Apply the splice rotation over a (rows, head_dim) K slab in place (the unchanged tail of the prompt).
/// `old_pos0` is the OLD prompt's position for row 0; `delta` is the per-token position offset
/// (uniform over a uniform edit; otherwise the caller iterates per-row).  `n_rot` matches the model.
inline void rotate_k_by_delta(float* k, int rows, int head_dim, int n_rot, int old_pos0, int delta,
                              const float* cos_tab, const float* sin_tab, int max_pos) {
    for (int r = 0; r < rows; ++r)
        rotate_row_by_delta(k + (size_t) r * (size_t) head_dim, head_dim, n_rot, old_pos0 + r, delta,
                            cos_tab, sin_tab, max_pos);
}

}  // namespace strata::program::suffix_reuse
