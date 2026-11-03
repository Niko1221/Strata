// include/strata/kernels/kv_exact_window.hpp - the exact window over a compressed KV layer: the first `sinks` cells and
// the last `tail` cells of every layer are also kept in fp16, so the tokens that carry most of the attention mass
// are never read from the quantized pool. The attention kernel does not change: it already reads a gathered fp16
// scratch, so a window position is simply gathered from here instead of from the codes.
//
// WHY THIS EXISTS. A 3-bit KV cache is 4.5x the reconstruction error of 4-bit on the same vectors (measured:
// kv_octave_parity, MSE 0.0508 vs 0.0114 rotated). vLLM's Octave KV reports the same arithmetic on this exact
// model - Qwen3.8-Flash-Next costs +0.0395 nats at 3 bits, and +0.0163 with an fp16 window over the first 4 and
// last 64 tokens - because with sparse attention a few recent tokens carry 35-80% of the mass, and their error
// does not average out. The window is what makes the format usable, not an extra.
//
// LAYOUT. One pool per layer: `sinks` rows written once at positions 0..sinks-1, then a ring of `tail` rows for
// the newest positions. A row is [n_head_kv][head_dim] fp16 for K and for V. A per-row tag holds the position it
// currently holds (or -1), so a gather of a position older than the window reports itself as a miss instead of
// returning another token's keys.
#pragma once

#include "strata/kernels/qsa.hpp"

#include "strata/kernels/f16_bits.hpp"   // STRATA_HD

#include <cstdint>

namespace strata::kernels {

struct KvExactWindow {
    uint16_t* k = nullptr;      ///< [layers][sinks + tail][n_head_kv * head_dim]
    uint16_t* v = nullptr;      ///< same
    int32_t* tag = nullptr;     ///< [layers][sinks + tail] the position the row holds, or -1
    int layers = 0, sinks = 0, tail = 0, n_head_kv = 0, head_dim = 0, pos_stride = 0;   // pos_stride = [kStepPos]

    STRATA_HD bool present() const { return k != nullptr && v != nullptr && tag != nullptr; }
    STRATA_HD int rows() const { return sinks + tail; }
};

/// The row a position occupies, or -1 when it is not in the window at all: sinks keep rows 0..sinks-1, the newest
/// `tail` positions cycle through the rest.  A row is only the CANDIDATE - the caller still compares `tag[row]`.
STRATA_HD inline int kv_window_row(const KvExactWindow& w, long long pos) {
    if (pos < 0) return -1;
    if (pos < w.sinks) return (int) pos;
    if (w.tail <= 0) return -1;
    return w.sinks + (int) ((pos - w.sinks) % w.tail);
}

/// Bytes for one layer's window (K and V rows plus the tags).
inline uint64_t kv_exact_window_bytes_per_layer(const QsaShapes& s, int sinks, int tail) {
    const uint64_t row = (uint64_t) s.n_head_kv * (uint64_t) s.head_dim * 2u;
    return (uint64_t) (sinks + tail) * row * 2u + (uint64_t) (sinks + tail) * 4u;
}

/// Zero the tags (no rows are valid) and initialise the geometry. `pos_stride` is the append step record's stride
/// in int32s, the same record kv_append_* already reads kStepPos from.
void kv_exact_window_init(KvExactWindow& w, const QsaShapes& s, int layers, int sinks, int tail, int pos_stride, void* stream);

/// Record the cell at `step[kStepPos]` (with the raw (unrotated for oct3, rotated for q4_0) K/V the append kernel
/// gets). Sinks are written once; a ring row is overwritten and its tag replaced, which is what evicts the oldest.
void kv_exact_window_append(const KvExactWindow& w, int layer, const int32_t* step, const float* kcur, const float* vcur,
                      int n_cells, void* stream);

/// The prompt path: cells `pos0 .. pos0+n-1` of a contiguous K/V block [n][n_head_kv][head_dim].  `end` is the
/// position the prompt will end at (exclusive): a cell that is neither a sink nor within `tail` of `end` can never
/// be in the window once the prompt is complete, so it is skipped.  Only the last `tail` cells of a chunk are
/// written, which is also what keeps the tag stores race-free (one writer per row).
void kv_exact_window_append_range(const KvExactWindow& w, int layer, int64_t pos0, int64_t end, const float* kcur,
                      const float* vcur, int64_t n_cells, void* stream);

/// Gather `n` positions of `layer` into fp16 scratch [n][n_head_kv][head_dim].  A position the window does not hold
/// is left untouched, so a caller that gathers AFTER the quantized pool keeps the pool's value there; `hit` (may be
/// null) receives 1 for a gathered position and 0 for one it kept.
void kv_exact_window_gather(const KvExactWindow& w, int layer, const int32_t* positions, int64_t n, uint16_t* k_out,
                      uint16_t* v_out, int32_t* hit, void* stream);

}  // namespace strata::kernels
