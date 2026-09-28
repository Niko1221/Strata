// include/strata/kernels/qsa_decode_attn.hpp - the QSA attention over each query's selected cells.
//
// The kernels read the KV POOLS directly through the page table and the selection ids (no gather copy) and serve
// all `n_head / n_head_kv` query heads that share a KV head from one read of each cell: a block per (KV head, query)
// walks the query's cells in tiles of 64 with an online softmax.  INT8 pools on the tensor cores: q and the
// probabilities as 24-bit fixed point (three 8-bit limbs) against the int8 codes, the integer sums exact, ~2.5e-6 of
// a head's largest output from FP64 on real prompts (FP32 kernels ~3e-6).  FP16 pools in FP32.
//
// The prompt path's queries fill the GPU by their number.  A verify window's 1-4 queries would leave most of it idle,
// so the decode splits each query's cells among up to 20 blocks (whole tiles, all queries' blocks in one wave) whose
// partial sums a merge adds with the usual log-sum-exp rescale.  Grids are sized by the capacity; the real count comes
// from `step[kStepWidth]` as for every other capturable QSA kernel.
#pragma once

#include "strata/kernels/qsa.hpp"

#include <cstdint>

namespace strata::kernels {

struct QsaAttnPools {
    const uint16_t* k_pool = nullptr;   ///< fp16 [page][kv_head][page_size][head_dim], or null when int8
    const uint16_t* v_pool = nullptr;
    const int8_t* k_q = nullptr;        ///< int8 codes, same layout
    const int8_t* v_q = nullptr;
    const uint16_t* k_scale = nullptr;  ///< fp16 [page][kv_head][page_size][head_dim / 64]
    const uint16_t* v_scale = nullptr;
    const int32_t* page_table = nullptr;
};

/// Scratch floats a query with `cap` selected cells needs: its splits' partial sums, maxima and sums.
uint64_t qsa_decode_attn_scratch_floats(int64_t cap, const QsaShapes& s);

void qsa_decode_attn_step(const float* q, const QsaAttnPools& pools, const int32_t* ids, const int32_t* step,
                          int64_t cap, const QsaShapes& s, float* scratch, float* attn, void* stream);

/// Plan v0.3 P5: `n_q` queries at once, each with its own selection: q [n_q, n_head, 256], ids [n_q, cap], steps
/// [n_q, kStepCount], attn [n_q, n_head, 256]; scratch is `n_q` times the single-query size.  With `gate` (the
/// query projection's rows [n_q, n_head, 2 * 256]: a head's queries, then its gate logits), attn is the gated output,
/// bitwise native_qsa_gate_apply's.
void qsa_decode_attn_batch(const float* q, const QsaAttnPools& pools, const int32_t* ids, const int32_t* steps,
                           int64_t cap, const QsaShapes& s, float* scratch, float* attn, int64_t n_q, void* stream,
                           const float* gate = nullptr);

/// The prompt path's form of qsa_decode_attn_batch (no scratch): a block per (KV head, query) walks all of the
/// query's cells, so no partial sums go through memory.
void qsa_prefill_attn(const float* q, const QsaAttnPools& pools, const int32_t* ids, const int32_t* steps, int64_t cap,
                      const QsaShapes& s, float* attn, int64_t n_q, void* stream);

}  // namespace strata::kernels
