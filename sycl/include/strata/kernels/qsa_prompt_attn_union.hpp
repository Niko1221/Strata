// SYCL port: the prompt attention (QSA, selected cells) over 8-query tiles on XMX - see qsa_prompt_attn_union.dp.cpp.
//
// The FP32 fallback (qsa_decode_attn_batch, 32 queries per launch) and the XMX v2 kernel (qsa_prompt_attn_xmx) read
// every query's own ~2,051 selected cells, once per query. Neighbouring prompt positions select many of the same
// cells, so here 8 neighbouring queries share one pass over the UNION of their selections:
//   pass 1, one work-group per 8-query tile: the sorted union of the 8 selections, each cell with the 8-bit set of
//           tile queries that selected it;
//   pass 2, one work-group per (tile, KV head): 64-cell chunks of the union, K and V dequantized once to FP16,
//           S^T = K Q^T and O += P V on joint_matrix (FP16 x FP16 -> FP32, 8x16x16, sub-group 16) for the tile's
//           96 rows (8 queries x 12 heads), an online softmax per row in which a cell the row's query did not select
//           is -inf - so every query attends to exactly its own selection.
// Not bitwise against the fallback: Q, K, V and P enter the products as FP16 (one rounding each) and the summation
// order is the tile's. INT8 K/V on a Battlemage card only; anything else returns false and the caller keeps its path.
#pragma once
#include "strata/kernels/qsa.hpp"
#include "strata/kernels/qsa_decode_attn.hpp"
#include <cstdint>
namespace strata::kernels {
/// The same contract as qsa_prompt_attn_batch; returns false (nothing launched) when the pools, shape, queue or device
/// are not handled, or its scratch (~80 MB per queue, allocated on first use) cannot be allocated.
bool qsa_prompt_attn_union(const float* q, const QsaAttnPools& pools, const int32_t* ids, const int32_t* steps, int64_t cap,
                           const QsaShapes& s, float* attn, int64_t n_q, void* stream);
/// STRATA_PROMPT_ATTN_UNION=1 (read once). Off by default.
bool qsa_prompt_attn_union_on();
}
