// include/strata/prefill/moe_fp16tc.hpp - sm_70 (Volta) FP16 tensor-core routed experts for the prompt path.
//
// Volta's MMQ path uses CUDA-core int8 products for Q2_0 experts. This module uses FP16 tensor-core products
// with FP32 accumulation. The caller can read native expert pointers directly without gathering their weights.
// `blob[i]` contains gate then up ([2 n_ff, n_embd] native Q2_0), and `down[i]` contains [n_embd, n_ff].
// Weight storage must remain valid until the products finish. Stream-slot reuse must wait for the consumption
// event recorded after those products on the consuming stream.
//
// The activation codes and float scales are the same q8_1 values that MMQ uses. This kernel rounds each scale
// to FP16, then multiplies it by the exact signed code in FP16. Weight scales are already FP16. The operand
// rounding and FP32 accumulation differ from MMQ; outputs are not bit-identical. The numerical regression
// compares both paths with a double-precision reference and covers compact row offsets.
#pragma once

#include <cstddef>
#include <cstdint>

namespace strata::prefill::fp16tc {

/// Experts per launch (the blob/down pointer arrays travel in the kernel parameters).
constexpr int kMaxBatch = 16;

/// One layer's native Q2_0 geometry: n_embd 2560, n_ff 640, gu_row 720, d_row 180, up_off 460800.
struct Geom {
    int n_embd = 0, n_ff = 0;
    size_t gu_row = 0, d_row = 0;   ///< bytes per gate/up weight row and per down weight row (Q2_0: 18 B per 64)
    size_t up_off = 0;              ///< the up half's byte offset in `Batch::blob[i]` (gate rows start at 0)
};

/// A group of experts: `blob[i]` = gate||up [2 n_ff, n_embd] Q2_0, `down[i]` = down [n_embd, n_ff] Q2_0.
/// `max_rows` is the most rows one of them has (the launch grid); rows come from the `bounds` device array.
struct Batch {
    int n = 0;                      ///< experts, indices 0..n-1
    int max_rows = 0;               ///< max over the experts of (bounds[i+1] - bounds[i])
    /// gu() only: added to the absolute sorted output row. A compact caller passes the negated sub-group start,
    /// so its output begins at destination row zero while bounds and activation input rows remain absolute.
    int64_t gu_dst_row_base = 0;
    /// down() only: added to the activation row. A compact caller passes the negated group-relative sub-group
    /// start so its q8_1 input begins at row zero. Output rows remain dst_row_base + bounds[z].
    int64_t down_act_row_base = 0;
    const uint8_t* blob[kMaxBatch] = {};
    const uint8_t* down[kMaxBatch] = {};
};

#ifdef STRATA_PREFILL_FP16TC

/// This build has the kernels (the source was compiled into the CUDA MMQ library).
bool built();
/// This build has the kernels and the current device has compute capability 7.0 (Volta).
bool available();
/// The gathered matrices are this kernel's native Q2_0 form: both types Q2_0 (ggml 42), n_embd a multiple of 128 and
/// n_ff a multiple of 64 (the block writes 128 output columns), gu_row = 18 * n_embd / 64, d_row = 18 * n_ff / 64,
/// up_off = gu_row * n_ff, down_off = 2 * up_off.
bool geom_ok(int gu_type, int d_type, int64_t n_embd, int64_t n_ff, size_t gu_row, size_t d_row, size_t up_off,
             size_t down_off);

/// Gate/up of the batch's experts into `dst` ([rows][2 n_ff], gate at [0, n_ff), up at [n_ff, 2 n_ff) - the native,
/// non-interleaved order the caller's swiglu already expects).  Expert z reads the activation rows
/// [bounds[z], bounds[z+1]) of `xq` (the layer's q8_1, `xq_rows` rows) - absolute sorted rows, never shifted - and
/// writes dst rows [dst_row_base + gu_dst_row_base + bounds[z], ...).  dst_row_base is 0 for this product.
void gu(const Batch& b, const Geom& g, const int32_t* bounds, const void* xq, int64_t xq_rows, float* dst,
        int64_t ld_dst, void* stream);

/// Down of the batch's experts into `dst` ([rows][n_embd]): expert z reads `hq` rows
/// [down_act_row_base + bounds[z], down_act_row_base + bounds[z+1]) (the group's own q8_1, row 0 = the group's
/// first row; a compact sub-group passes the negated group-relative start so it reads its own [0, nr) rows) and
/// writes dst rows [dst_row_base + bounds[z], ...) - the output mapping is not shifted by down_act_row_base.
void down(const Batch& b, const Geom& g, const int32_t* bounds, const void* hq, int64_t hq_rows, float* dst,
          int64_t ld_dst, int64_t dst_row_base, void* stream);

#else  // !STRATA_PREFILL_FP16TC - the kernels are not in this build: no-ops so the caller needs no #ifdef.

inline bool built() { return false; }
inline bool available() { return false; }
inline bool geom_ok(int, int, int64_t, int64_t, size_t, size_t, size_t, size_t) { return false; }
inline void gu(const Batch&, const Geom&, const int32_t*, const void*, int64_t, float*, int64_t, void*) {}
inline void down(const Batch&, const Geom&, const int32_t*, const void*, int64_t, float*, int64_t, int64_t, void*) {}

#endif

}  // namespace strata::prefill::fp16tc
