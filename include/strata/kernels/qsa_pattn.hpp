// include/strata/kernels/qsa_pattn.hpp - STRATA_PATTN_SM60: the prompt path's QSA attention on sm_6x, bit for bit
// what prefill.cpp gets today from qsa_decode_attn_batch (attn_chunk_kernel_pre75<1, 2> + attn_merge_kernel or
// attn_merge_v2_kernel, 32 queries a call) with int8 K/V.
//
// On sm_6x qsa_prompt_attn_batch returns false (no tensor cores) and the prompt reads every QSA layer's attention
// through the decode kernels: one CTA per (64-cell chunk, KV head, query), the partials of all 33 chunks written to
// scratch (817 KB a query) and merged by a second kernel.  The kernels here keep every operation of that path in its
// order - each lane's 8-dimension products, warp_sum's tree over the 32 lanes, the chunk softmax (warp_max / __expf /
// warp_sum pairs), the ascending fmaf value fold, the merge's chunk order (fmaxf, __expf(m - M), fmaf(l, w, L),
// fmaf(acc, w, acc), acc / L) - and change only where operands live and which thread does what.  The default
// (variant 0 = split code 1463):
//   - ps_score_x_kernel, one CTA per (chunk, KV head, query): q in registers (loaded once in a per-lane slot order),
//     the chunk's K staged with 16-byte loads, two cells at a time through one reduce-scatter whose 24 sums are
//     warp_sum's tree without reduce12's selects; writes the chunk's probabilities, rows, m and l to the scratch;
//   - ps_value_d4_kernel, one CTA per (KV head, query): four dims x 12 heads a thread, V four bytes a load, the chunks
//     in ascending order, each merged into the output as the merge kernel does (no partial accumulators, no merge
//     kernel).
// The caller's attention scratch holds the blocks of ~100 queries; a prompt chunk runs in batches of that size.
//
// Off unless STRATA_PATTN_SM60=1.  Variants other than the default are for pattn_bench.
#pragma once

#include "strata/kernels/qsa.hpp"
#include "strata/kernels/qsa_decode_attn.hpp"

#include <cstdint>

namespace strata::kernels {

/// STRATA_PATTN_SM60: 0 off (default); 1 on (the default variant).
int pattn_mode();

/// qsa_decode_attn_batch for all `n_q` queries of a prompt chunk in one call (int8 K/V, sm_6x, experimental build).
/// `scratch` / `scratch_floats`: the caller's attention scratch (prefill.cpp: attn_batch x qsa_decode_attn_scratch_floats).
/// variant: 0 = the default (1463); bench variants: 1 = per-chunk kernel (q in registers, K staged) + attn_merge_v2,
/// 2-9 = one fused persistent kernel (both passes in a CTA), 10-13 and 1000 + 100 S + 10 V + mask = split score / value
/// kernels (qsa_pattn.cu).  False: not applicable (nothing launched).
bool qsa_pattn_batch(const float* q, const QsaAttnPools& pools, const int32_t* ids, const int32_t* steps, int64_t cap,
                     const QsaShapes& s, float* scratch, uint64_t scratch_floats, float* attn, int64_t n_q,
                     void* stream, int variant = 0);

/// registers / shared memory / CTAs per SM of the kernels (pattn_bench's report)
void qsa_pattn_info(char* buf, int len);

}  // namespace strata::kernels
