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

#include "strata/kernels/kv_q4.hpp"
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
    Q4Mode q4_mode = Q4Mode::kUniform;  ///< `--kv q4lm32`/`q4lm32s`: k_q4/v_q4 hold Max-Lloyd levels
    const int32_t* page_table = nullptr;
    /// WP-2a window (`--kv q4lm32w`): the exact fp16 side pool, rows [win_r sinks][win_s + win_b recent ring],
    /// [(row, kv_head, head_dim)] per tensor - no paging, always resident. A reader never indexes it by cell: it
    /// asks `qsa_win_slot` for the row (or -1 = read quantized), with `n` = step[kStepNKv]. The sink rows are
    /// cell-addressed (`cell < win_r`), written once and never moved. The recent rows are a ring,
    /// `row = win_r + cell % (win_s + win_b)`, and hold cell `c` only while `win_tag` says so: a rewind or a
    /// resume leaves a *later* cell in that slot, and the tag turns that into "read quantized" instead of
    /// "read the wrong cell". `win_b` is the largest append batch the ring must serve at once (a prompt chunk:
    /// every query of a batch has its own horizon, so the whole batch's window span is live simultaneously) -
    /// see STRATA_KV_WINDOW_BATCH. Null (and win_s/r/b <= 0) when the window is off.
    const uint16_t* k_win = nullptr;
    const uint16_t* v_win = nullptr;
    const int32_t* win_tag = nullptr;  ///< cell id of each recent ring row [(win_s + win_b)], or null (sinks only)
    int win_s = 0;    ///< recent exact cells per horizon (STRATA_KV_WINDOW_S, default 128)
    int win_r = 0;    ///< exact sink cells at the start (STRATA_KV_WINDOW_R, default 8)
    int win_b = 0;    ///< recent ring rows beyond win_s: the largest append batch (STRATA_KV_WINDOW_BATCH)
    // NOTE: no horizon field - readers take it from step[kStepNKv] (pos + 1, device-fresh and graph-safe;
    // qsa_step_fill derives all four step entries from one position). A host horizon would bake stale on
    // graph reuse.
};

/// Batch windows: every query's own K/V pools - query z reads p[z] (z < n).  n = 0: the call's one `pools`.
inline constexpr int kAttnRowsMax = 8;
struct QsaAttnPoolsRows {
    QsaAttnPools p[kAttnRowsMax];
    int n = 0;
};

#if defined(__CUDACC__) || defined(__HIPCC__)
/// True when the pools carry a live window: both side pools present, at least one exact cell, and - for the recent
/// rows - the tag that makes them readable (the sink rows need none, they never move).
inline __host__ __device__ bool qsa_win_on(const QsaAttnPools& p) {
    return p.k_win != nullptr && p.v_win != nullptr &&
           (p.win_r > 0 || (p.win_s > 0 && p.win_tag != nullptr));
}

/// Side-pool row that holds cell `c` under horizon `n` = step[kStepNKv], or -1 when the cell reads quantized.
/// A sink cell always hits its own row; a recent cell hits its ring row only while that row's tag still names `c`
/// (a rewind or a resume can leave a later cell there) - otherwise -1, and the caller reads the quantized pool.
inline __host__ __device__ long long qsa_win_slot(const QsaAttnPools& p, long long c, long long n) {
    if (c < (long long) p.win_r) return c;
    if (p.win_s <= 0 || c < n - (long long) p.win_s) return -1;
    const long long ring = (long long) p.win_s + (long long) p.win_b;
    if (ring <= 0 || p.win_tag == nullptr) return -1;
    const long long i = c % ring;
    if ((long long) p.win_tag[i] != c) return -1;
    return (long long) p.win_r + i;
}
#else
/// True when the pools carry a live window: both side pools present, at least one exact cell, and - for the recent
/// rows - the tag that makes them readable (the sink rows need none, they never move).
inline bool qsa_win_on(const QsaAttnPools& p) {
    return p.k_win != nullptr && p.v_win != nullptr &&
           (p.win_r > 0 || (p.win_s > 0 && p.win_tag != nullptr));
}

/// Side-pool row that holds cell `c` under horizon `n` = step[kStepNKv], or -1 when the cell reads quantized.
/// A sink cell always hits its own row; a recent cell hits its ring row only while that row's tag still names `c`
/// (a rewind or a resume can leave a later cell there) - otherwise -1, and the caller reads the quantized pool.
inline long long qsa_win_slot(const QsaAttnPools& p, long long c, long long n) {
    if (c < (long long) p.win_r) return c;
    if (p.win_s <= 0 || c < n - (long long) p.win_s) return -1;
    const long long ring = (long long) p.win_s + (long long) p.win_b;
    if (ring <= 0 || p.win_tag == nullptr) return -1;
    const long long i = c % ring;
    if ((long long) p.win_tag[i] != c) return -1;
    return (long long) p.win_r + i;
}
#endif

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
