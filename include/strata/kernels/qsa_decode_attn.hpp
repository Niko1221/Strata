// include/strata/kernels/qsa_decode_attn.hpp - plan v0.3 P3/P7: split-K decode attention over the selected cells.
//
// The first attention kernel runs one block per query head (24 blocks on a 48-SM part), reads each key row with one
// thread (uncoalesced) and walks every selected cell serially for the values; at a 4K context (2,051 selected
// cells) that costs ~0.3-0.5 ms per QSA layer.  This one reads the KV POOLS directly through the page table and the
// selection ids (no gather copy), in chunks of CHUNK cells per block, and serves all `n_head / n_head_kv` query
// heads that share a KV head from one read of the chunk:
//
//     grid (ceil(cap / CHUNK), n_head_kv):  s = q.k * scale for 12 heads x CHUNK cells, chunk max m and sum l,
//                                           acc = p . V  ->  partials
//     grid (n_head):                        merge the chunks with the usual log-sum-exp rescale
//
// FP16 pools or INT8 pools (codes + fp16 scale per 64 values, `kv_q8.hpp`).  The grid is sized by the capacity;
// the real count comes from `step[kStepWidth]` as for every other capturable QSA kernel.
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
    const uint8_t* k_q4 = nullptr;      ///< q4_0 block_q4_0 [page][kv_head][page_size][head_dim / 32 * 18]
    const uint8_t* v_q4 = nullptr;
    const int32_t* page_table = nullptr;
};

/// Batch windows: every query's own K/V pools - query z reads p[z] (z < n).  n = 0: the call's one `pools`.
inline constexpr int kAttnRowsMax = 8;
struct QsaAttnPoolsRows {
    QsaAttnPools p[kAttnRowsMax];
    int n = 0;
};

/// fuse-gdn (STRATA_FUSE_QSAWAVE): rows of a batch window that attend over different slots' pools at once - row
/// wave.row[z] of the window (its q, selection, step and scratch at the window's bases + row strides) with pools wave.p[z].
struct QsaAttnWave {
    QsaAttnPools p[kAttnRowsMax];
    int row[kAttnRowsMax] = {};
    int n = 0;
};

/// Scratch floats for `cap` selected cells: partial accumulators, maxima and sums.
uint64_t qsa_decode_attn_scratch_floats(int64_t cap, const QsaShapes& s);

void qsa_decode_attn_step(const float* q, const QsaAttnPools& pools, const int32_t* ids, const int32_t* step,
                          int64_t cap, const QsaShapes& s, float* scratch, float* attn, void* stream);

/// Plan v0.3 P5: `n_q` queries at once, each with its own selection: q [n_q, n_head, 256], ids [n_q, cap], steps
/// [n_q, kStepCount], attn [n_q, n_head, 256]; scratch is `n_q` times the single-query size.
void qsa_decode_attn_batch(const float* q, const QsaAttnPools& pools, const int32_t* ids, const int32_t* steps,
                           int64_t cap, const QsaShapes& s, float* scratch, float* attn, int64_t n_q, void* stream);

/// Up to this many queries of different slots in one qsa_decode_attn_slots call.
constexpr int kQsaAttnMaxSlots = 8;

/// One KV pool set per query of a qsa_decode_attn_slots call (passed by value as a kernel argument).
struct QsaAttnSlotPools {
    QsaAttnPools p[kQsaAttnMaxSlots];
};

/// qsa_decode_attn_batch for `n_q` queries that each attend over their OWN pools (`pools.p[i]`, one KV cache per batch
/// slot): one launch instead of one call per query, every block computing exactly what the per-query call computes.
/// The pools must all have one KV format. False (nothing launched) where the per-query call would run another kernel
/// (n_q > kQsaAttnMaxSlots, mixed formats, the pre-sm_75 / gfx103x or gfx906 variants): the caller loops instead.
bool qsa_decode_attn_slots(const float* q, const QsaAttnSlotPools& pools, const int32_t* ids, const int32_t* steps,
                           int64_t cap, const QsaShapes& s, float* scratch, float* attn, int64_t n_q, void* stream);
/// fuse-gdn (STRATA_FUSE_ATTNMERGE): qsa_decode_attn_batch's two passes apart, each the kernels qsa_decode_attn_batch
/// picks (the environment's): `chunks` writes the n_q queries' partials into `scratch` (one
/// qsa_decode_attn_scratch_floats(cap, s) stride each), `merge` turns them into `attn`.  A batch window runs the
/// chunks of its rows one by one (each between its row's resolve) and merges them all in one launch: per query the
/// same kernels on the same bytes as qsa_decode_attn_batch.
void qsa_decode_attn_batch_chunks(const float* q, const QsaAttnPools& pools, const int32_t* ids, const int32_t* steps,
                                  int64_t cap, const QsaShapes& s, float* scratch, int64_t n_q, void* stream);
/// fuse-gdn (STRATA_FUSE_QSAWAVE): qsa_decode_attn_batch_chunks for the wave's rows in one launch - per row the same
/// kernel on the same bytes.  Only where that kernel is HIP's attn_chunk_kernel_pre75 (gfx103x / gfx12); elsewhere it
/// launches nothing and returns false (the caller runs the per-row calls).
bool qsa_decode_attn_wave_chunks(const float* q, const QsaAttnWave& wave, const int32_t* ids, const int32_t* steps,
                                 int64_t cap, const QsaShapes& s, float* scratch, void* stream);
/// fuse-gdn (STRATA_FUSE_GATEMERGE): qsa_decode_attn_batch_merge, then native_qsa_gate_apply(attn, q_full, gated,
/// n_q * n_head, head_dim) in the same launch - the same bits in attn and gated.  HIP with the v2 merge only; elsewhere
/// it launches nothing and returns false.  `xq_out` non-null (STRATA_FUSE_GATEMERGE_Q8): also gated's q8_1 image, row r
/// at xq_out's row r - native_quantize_q8_1(gated, xq_out, n_head * head_dim, n_q)'s bytes.
bool qsa_decode_attn_batch_merge_gate(const int32_t* steps, int64_t cap, const QsaShapes& s, const float* scratch,
                                      float* attn, const float* q_full, float* gated, int64_t n_q, void* stream,
                                      void* xq_out = nullptr);
void qsa_decode_attn_batch_merge(const int32_t* steps, int64_t cap, const QsaShapes& s, const float* scratch, float* attn,
                                 int64_t n_q, void* stream);
/// For src/kernels/qsa_merge_daily_parity.cpp only: qsa_decode_attn_batch's two passes apart, whatever the environment
/// says.  `chunks_only` runs the chunk pass into `scratch` with `daily_chunk` 0: attn_chunk_kernel (upstream's default),
/// 1: the Daily's (attn_chunk_kernel_pre75, what STRATA_ATTN_DAILY=1 picks on CUDA).  `merge_only` merges those
/// partials with `v2` 0: attn_merge_kernel, 1: attn_merge_v2_kernel (STRATA_ATTN_MERGE_V2, on with STRATA_ATTN_DAILY).
/// Width must not exceed an unaligned cap (the chunk pass keeps its upstream contract); for aligned caps the grid
/// bounds widths above cap.  One input can go through every pair and the outputs be compared bit for bit.
void qsa_decode_attn_chunks_only(const float* q, const QsaAttnPools& pools, const int32_t* ids, const int32_t* steps,
                                 int64_t cap, const QsaShapes& s, float* scratch, int64_t n_q, int daily_chunk,
                                 void* stream);
void qsa_decode_attn_merge_only(const int32_t* steps, int64_t cap, const QsaShapes& s, const float* scratch,
                                float* attn, int64_t n_q, int v2, void* stream);
/// qsa_decode_attn_batch for up to kAttnRowsMax queries of DIFFERENT sequences (batch slots), query z attending
/// over rows.p[z] - one launch instead of one per query, the same per-query arithmetic (bit-identical outputs).
void qsa_decode_attn_batch_rows(const float* q, const QsaAttnPoolsRows& rows, const int32_t* ids, const int32_t* steps,
                                int64_t cap, const QsaShapes& s, float* scratch, float* attn, int64_t n_q, void* stream);

}  // namespace strata::kernels
