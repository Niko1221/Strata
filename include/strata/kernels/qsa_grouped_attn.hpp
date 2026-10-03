// include/strata/kernels/qsa_grouped_attn.hpp - V100 (sm_70) grouped verify attention over one window's rows.
//
// WHAT IT IS.  The verify window runs M = 2..8 rows (draft + verified tokens) through `qsa_decode_attn_batch`,
// whose FP32 kernel replays the decode path per row.  This kernel serves the whole group in ONE launch: one CTA
// per (split, KV head, token) walks that token's own top-k selection in 64-cell tiles on Volta WMMA m16n16k16
// (no ldmatrix; row-padded panels), each selected KV row is read once and shared by the token's 12 query heads,
// and the concurrent token CTAs' overlapping reads meet in L2.  Online softmax over the tiles, split-K partials
// and one combine kernel keep the shape of `qsa_decode_attn_batch`'s output contract.
//
// WHY NOT ONE SHARED TILE STREAM FOR ALL ROWS (the llama.cpp grouped verify shape): the rows carry their own
// top-k selections, so a shared cell stream computes row/cell pairs nobody selected.  Measured at M = 8, 32K
// context: the 8 x 2,051 selections union to 5,500 cells - 2.68x of extra QK/PV compute, which cost more than
// the shared KV reads saved (qsa_grouped_attn.cu has the numbers).  The rows are grouped by launch and by
// shared reads instead.
//
// WHO CALLS IT.  `qsa_decode_attn_batch` tries this first and keeps its FP32 path when it returns false.  The
// gate is cc < 80 (the V100 build; sm_80+ keeps the default path exactly), 2 <= n_q <= 8, the artifact's
// 24/2/256 geometry, and the KV format (all four).  n_q == 1 is never routed here.
// `STRATA_GROUPED_ATTN=0` forces the old path; `=1` forces this one on any card for A/B runs.
//
// NUMBERS.  FP32-level, not bitwise, the `qsa_prompt_attn.hpp` precedent.  Queries and probabilities enter the
// MMAs as exact hi+lo FP16 pairs (~22 bits); int8 K codes enter as exact FP16 with their per-64 scales folded
// into the score in FP32; FP16 pools enter as is.  What is left: the FP16 dequant of int8/q4 V (and q4 K) and
// the summation order.  `qsa_grouped_attn_parity` bounds it: against FP64 the new kernel's error stays under
// 1e-3 of the output scale (measured 2.2e-5 on fp16 KV, 3.2e-4 on int8 KV) and the new and old kernels' outputs
// agree to the same bound (measured 7e-6 / 3.3e-4).  `STRATA_GA_HILO=0` builds trade the hi+lo pairs for one
// FP16 cast each: ~1.4x faster, measured ~6e-4 of scale.  Deterministic run to run.
#pragma once

#include "strata/kernels/qsa_decode_attn.hpp"

#include <cstdint>

namespace strata::kernels {

/// Same arguments and output as `qsa_decode_attn_batch`; `scratch` (n_q times
/// `qsa_decode_attn_scratch_floats(cap, s)` floats) holds the split-K partial rows.  Returns false and launches
/// NOTHING when the device is not cc < 80, n_q is outside 2..8, the geometry is not 24/2/256 or the pools are
/// missing.
#if defined(STRATA_V100_OPT)
bool qsa_grouped_attn_batch(const float* q, const QsaAttnPools& pools, const int32_t* ids, const int32_t* steps,
                            int64_t cap, const QsaShapes& s, float* scratch, float* attn, int64_t n_q,
                            void* stream);
#else
// The V100 switch is off: the grouped kernel is not compiled and every caller stays on the FP32 path.
inline bool qsa_grouped_attn_batch(const float* q, const QsaAttnPools& pools, const int32_t* ids,
                                   const int32_t* steps, int64_t cap, const QsaShapes& s, float* scratch,
                                   float* attn, int64_t n_q, void* stream) {
    (void) q; (void) pools; (void) ids; (void) steps; (void) cap; (void) s; (void) scratch; (void) attn;
    (void) n_q; (void) stream;
    return false;
}
#endif

}  // namespace strata::kernels
