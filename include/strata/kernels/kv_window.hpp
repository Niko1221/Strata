// include/strata/kernels/kv_window.hpp - FP16 sliding window + attention sinks over a quantized KV cache.
//
// SKVQ's tiering (arXiv 2405.06219) with the sink tokens of arXiv 2508.04257: the bulk of the cache stays in
// whatever format `--kv` chose, but the `sinks` cells at the head of the sequence and the `window` most recent
// cells are also kept in FP16, and every reader prefers that copy for those cells.  The exact copy costs a few
// hundred KB while the quantized bulk still costs hundreds of MB, so the error is removed exactly where the
// model attends most closely.
//
// At the serving geometry (12 QSA layers, 2 KV heads, head_dim 256): (8 sinks + 128 window + 16 slack) slots x
// 1 KiB (2 heads x 256 dims x 2 B) x 2 (K and V) = 304 KiB per layer, 3.56 MiB over 12 layers, against 1.69 GiB
// of q4_0 bulk at 262144 cells (576 B per cell and layer).
//
// The bulk pools stay COMPLETE.  Every cell is still written to k_q4/v_q4 (or k_q/v_q) by the append kernels as
// before, so snapshots, KV-streaming residency, ring modes and prompt checkpoints keep reading exactly the pools
// they always read; the window is a read-side overlay plus one small extra write per append.  A reader resolves a
// cell to a *row*, and rows of the overlay are numbered above the bulk pool's rows (see kv_window_row), so the
// single `row` a reader already computes carries the "this is FP16" decision.
//
// Enabling: STRATA_KV_WINDOW (cells, default 0 = off) and STRATA_KV_SINK (cells, default 8).  Off, every reader
// takes the path it took before this existed, bit for bit.
#pragma once

#include "strata/kernels/qsa.hpp"
#include "strata/kernels/qsa_decode_attn.hpp"   // QsaAttnPools, for the restore's bulk source

#include <cstdint>

#if defined(__CUDACC__) || defined(__HIPCC__)
#define STRATA_KV_WIN_FN __host__ __device__ __forceinline__
#else
#define STRATA_KV_WIN_FN inline
#endif

namespace strata::kernels {

/// Cells kept in FP16 at each end of the sequence.
struct KvWindowShape {
    int32_t window = 0;  ///< most recent cells kept in FP16 (0: the overlay is off)
    int32_t sinks = 0;   ///< cells at position < sinks kept in FP16 (attention sinks)
    bool on() const { return window > 0; }
};

/// One layer's FP16 overlay: (sinks + window) slots per KV head, layout [slot][kv_head][head_dim] fp16 - the row
/// addressing the fp16 pools already use, so a reader reads HD halves at (slot * n_head_kv + kvh) * HD.
/// Sinks own slots [0, sinks) for the cells at positions [0, sinks); the window ring owns [sinks, sinks + window)
/// for cell p at slot sinks + p % window.
struct KvWindowPools {
    uint16_t* k = nullptr;
    uint16_t* v = nullptr;
};

/// Ring slots beyond the window.  Speculative decoding appends a whole draft batch before it attends, so a cell at
/// position p can be overwritten by a cell at p + (draft length) while p is still inside the window that batch
/// reads; with window + slack slots a look-ahead write never lands on a slot a live window cell owns, and the
/// reader's membership test stays the plain `cell >= n_kv - window`.
inline constexpr int32_t kKvWindowSlack = 16;

STRATA_KV_WIN_FN int64_t kv_window_slots(const KvWindowShape& w) {
    return (int64_t) w.sinks + w.window + (w.window > 0 ? kKvWindowSlack : 0);
}

inline uint64_t kv_window_bytes(const QsaShapes& s, const KvWindowShape& w) {
    return (uint64_t) kv_window_slots(w) * (uint64_t) s.n_head_kv * (uint64_t) s.head_dim * sizeof(uint16_t) * 2;
}

/// The overlay slot of `cell`, or -1 when the cell is outside the sinks and the window.  `n_kv` is the number of
/// cells in the cache (step[kStepNKv]).  A sink always wins over the ring: for a sequence shorter than the window
/// the two regions cover the same cells, and the sink copy is the one that never moves.
STRATA_KV_WIN_FN int32_t kv_window_slot(int64_t cell, int64_t n_kv, int32_t window, int32_t sinks) {
    if (window <= 0 || cell < 0) return -1;
    if (cell < sinks) return (int32_t) cell;
    if (cell >= n_kv - window) return sinks + (int32_t) (cell % (window + kKvWindowSlack));
    return -1;
}

/// The row a reader should use for `cell` in `kvh`: below `bulk_rows` it is the pool row the page table gives
/// ((page * n_head_kv + kvh) * page_size + cell % page_size); at or above it, the overlay row
/// (bulk_rows + slot * n_head_kv + kvh), which reads FP16 from KvWindowPools instead of the quantized pool.
STRATA_KV_WIN_FN long long kv_window_row(int64_t cell, int64_t n_kv, int32_t window, int32_t sinks, int n_head_kv,
                                         int kvh, long long bulk_row, long long bulk_rows) {
    const int32_t slot = kv_window_slot(cell, n_kv, window, sinks);
    if (slot < 0 || bulk_rows <= 0) return bulk_row;
    return bulk_rows + (long long) slot * n_head_kv + kvh;
}

/// Copy the cell(s) an append just stored into the overlay, in FP16.  The values are the same (already rotated)
/// K/V the append quantized, so the overlay is the un-quantized original of exactly those cells.
///
/// `pos` and `n_kv` come from step[kStepPos]/step[kStepNKv] on the DEVICE, so the call site stays capturable.
void kv_window_write_step(const KvWindowPools& w, const KvWindowShape& shape, const int32_t* step, const float* kcur,
                          const float* vcur, const QsaShapes& s, void* stream);

/// The batched decode form: `n` steps at `step_stride`, K/V rows [n, n_head_kv, head_dim].
void kv_window_write_steps(const KvWindowPools& w, const KvWindowShape& shape, const int32_t* step, int step_stride,
                           int n, const float* kcur, const float* vcur, const QsaShapes& s, void* stream);

/// The prompt form: T consecutive (already rotated) cells from pos0, K/V [T, n_head_kv, head_dim].
void kv_window_write_batch(const KvWindowPools& w, const KvWindowShape& shape, int64_t pos0, int64_t T, const float* K,
                           const float* V, const QsaShapes& s, void* stream);

/// Rebuild the overlay from the quantized bulk (which is COMPLETE) for the cells at [0, sinks) and the last
/// `window` cells of `n_kv`.  Used after a snapshot or ring restore, where the overlay would otherwise hold the
/// previous conversation's cells: the restored window comes back at bulk precision, not exact, but correct.
/// `bulk` must address the authoritative identity copy (the host copy for a streamed state, the pools otherwise).
void kv_window_restore(const KvWindowPools& w, const KvWindowShape& shape, const QsaAttnPools& bulk, int kv_mode,
                       int64_t n_kv, const QsaShapes& s, void* stream);

}  // namespace strata::kernels
