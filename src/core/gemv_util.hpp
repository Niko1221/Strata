// src/core/gemv_util.hpp - the quantized-projection plumbing, shared by every architecture's layer file.
//
// These four things (`sform_of`, `plane_ptrs`, `gemv_quantized`, `project_bf16`) were file-local to
// `src/core/layer.cpp` while there was one architecture.  GLM-5.3-Flash's layer file needs exactly the same
// rules - which activation a tensor wants is a property of the TENSOR, and the plane layout is what the
// loader recorded - so they live here rather than being written a second time with a chance to drift.
//
// This is an INTERNAL header: it names `WeightRef` and two process-wide switches, so it is for `src/core`
// only and is not part of the engine's public surface.
#pragma once

#include "strata/core/weights.hpp"

#include "strata/kernels/s_gemv.hpp"

#include <cstdint>
#include <string>

namespace strata::core::gemv {

constexpr int Q8K_BYTES_PER_BLOCK = 292;
constexpr int Q8K_ELEMS_PER_BLOCK = 256;
/// `qwen4exp.attention.layer_norm_rms_epsilon`.  GLM's is 1e-5 and comes from the file (`ModelGeometry`),
/// so a GLM layer must NOT use this constant - it is here for the callers that predate the second family.
constexpr float RMS_EPS = 1e-6f;

/// A Q8_K buffer needs `n` a multiple of 256 and `n/256` blocks of 292 bytes.
uint64_t q8k_bytes(int64_t n);

/// `s_gemv_q8k` takes the canonical-form attributes; a `WeightRef` carries them, and a tensor that is NOT
/// quantized has none.  Returns false and names the tensor rather than building a form out of zeroes - which
/// would decode every code as `0 + bias` and produce a perfectly finite wrong answer.
bool sform_of(const WeightRef& r, strata::kernels::SForm& f, const std::string& name, std::string& err);

/// The three canonical planes of a quantized tensor, located INSIDE the loaded region.
struct Planes {
    const uint8_t* codes = nullptr;
    const float* scales = nullptr;
    const float* offset = nullptr;   ///< null when the form has none
};
bool plane_ptrs(const WeightRef& r, const std::string& name, Planes& out, std::string& err);

/// WHICH ACTIVATION A QUANTIZED WEIGHT WANTS, AND IT IS A PROPERTY OF THE **TENSOR**, NOT OF ITS ROLE.
///
///     code_bits == 2           ->  Q8_0,  via `quantize_q8_0` + `s2_gemv_q8`
///     anything else quantized  ->  Q8_K,  via `quantize_q8_K` + `s_gemv_q8k_split`
///
/// `x80` and `xq8k` are the two quantized images of the SAME activation; a caller produces both once and this
/// picks.  Producing only the one it thinks it needs is how the role-based assumption gets baked in again.
///
/// `w.native_data` (a projection served straight from the GGUF) takes the FP32 `x_f32` route instead, and
/// needs `w.native_q8_1` scratch.  `x_q8_1_ready` says a previous native projection already quantized this
/// same `x` into that scratch with no native projection in between, so the quantize can be skipped.
///
/// `ncols` is the CHUNKED-PREFILL column count: how many tokens share this weight in one call, 1..8.  GGUF
/// order puts columns contiguous, so `x_f32`, `x80` and `xq8k` are `ncols` consecutive `n_in`-wide columns and
/// `y` is `ncols` consecutive `n_out`-wide ones.  Because the columns are contiguous with no padding and every
/// `n_in` here is a whole number of quantisation blocks, `ncols` columns of `n_in` ARE one column of
/// `ncols * n_in` - so a caller produces the image with one `quantize_*(x, img, ncols * n_in)` and the
/// kernel-side block walk is unchanged.  Each column's output is bitwise what a `ncols == 1` call would give
/// (`native_mmvq_multi_exact`).  Only the native path takes `ncols > 1`; the canonical one loops.
bool gemv_quantized(const WeightRef& w, const Planes& p, const strata::kernels::SForm& f, const uint8_t* x80,
                    const uint8_t* xq8k, float* y, int64_t n_in, int64_t n_out, const std::string& name,
                    void* stream, std::string& err, const float* x_f32 = nullptr,
                    bool x_q8_1_ready = false, int ncols = 1);

/// The same projection over a CONTIGUOUS BAND of the weight's output rows, `[row0, row0 + rows)`.
///
/// This exists for absorbed MLA and nothing else.  `attn_k_b` is [256, 512, 64] - one 256x512 map PER HEAD,
/// folded into a 32768-row matrix - and the absorbed query is each head's own 256 values mapped into that
/// head's 512-wide latent.  The band for head `h` is therefore rows `[h*512, (h+1)*512)`, and the alternative
/// to slicing is a dedicated block-diagonal matvec kernel per quantization type, which is a decoder family
/// written a second time.
///
/// A ROW BAND IS EXACTLY A POINTER OFFSET because both the native and the canonical layouts store rows
/// contiguously.  The canonical side derives the per-row stride by dividing the plane sizes by `ne1` and
/// REFUSES if they do not divide - the assumption is checkable, so it is checked rather than trusted.
///
/// `x_f32`/`x80`/`xq8k` are the images of the `n_in`-wide input for THIS band; the caller quantizes the slice
/// it is about to project, because different bands read different inputs.
///
/// `ncols` is the same chunked-prefill column count as `gemv_quantized`'s and composes with the band the same
/// way: the band selects WEIGHT rows, `ncols` selects ACTIVATION columns.  The absorbed-MLA group path uses
/// both at once - one head's band, every token in the chunk - which is what turns 64 quantizes per token into
/// 64 per chunk.
bool project_rows(const WeightRef& w, const std::string& name, const float* x_f32, const uint8_t* x80,
                  const uint8_t* xq8k, float* y, int64_t n_in, int64_t row0, int64_t rows, void* stream,
                  std::string& err, int64_t ncols = 1);

/// Native BF16/F32 projections for the SSM gates, routing and the sparse indexer (`layer_set_native_bf16`).
extern bool native_bf16_projections;
/// The diagnostic short-context vector-attention adapter (`layer_set_native_flash_attn_short`).
extern bool native_flash_attn_short;

/// The row-split GEMVs' threads per row.  MEASURED, NOT ASSUMED, AND 64 IS NOT BETTER.
void project_bf16(const float* x, const uint16_t* x_bf16, const uint16_t* weights, float* out, int64_t n_in,
                  int64_t n_out, bool split, void* stream);

}  // namespace strata::core::gemv
