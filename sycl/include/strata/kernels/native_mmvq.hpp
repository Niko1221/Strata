#pragma once

#include <cstddef>
#include <cstdint>

namespace strata::kernels {

// Native GGUF Q2_0, Q4_0, Q5_0, Q8_0, Q3_K, Q4_K, Q5_K, Q6_K, IQ4_NL
// and IQ4_XS / CUDA Q8_1 adapters, pinned to llama.cpp
// 3cf03257f219afbe7334045ff7c6a06ac68c627d, sm_120 generic MMVQ.
// All pointers are device pointers, at least 4-byte aligned, with no overlap.
// All calls enqueue on the explicit non-null CUDA stream; no allocation or wait.
// The translation unit must use --use_fast_math, as the pinned CUDA oracle does.
//
// Shapes use GGUF order: n_in is the contiguous reduction dimension, n_out is
// the weight row count, and ncols is the activation column/token count, 1..8
// (plan v0.3 P3). Columns are contiguous: activation column j is x + j * n_in
// (its Q8_1 blocks at j * n_in / 32) and output column j is y + j * n_out. See
// native_mmvq_set_multi_exact for how ncols > 1 relates to ncols == 1.
// n_in must be a positive multiple of 32 for quantization/Q4_0/Q5_0/Q8_0/IQ4_NL,
// 64 for Q2_0, or 256 for the other formats. n_out must be positive. Weights remain
// unmodified row-major GGUF blocks: Q3_K=110, Q4_K=144, Q5_K=176, Q6_K=210 and
// IQ4_XS=136 bytes per 256 elements; Q2_0 is 18 bytes per 64 elements.
// Q4_0/IQ4_NL=18, Q5_0=22, Q8_0=34 bytes per 32 elements.
// Q8_1 scratch has 36 bytes per 32 elements, with no extra row padding here.
// Input floats must be finite, and their block scales/sums representable in FP16.
std::size_t native_q8_1_bytes(int n_in, int ncols = 1);

// Layout for ncols > 1. false: llama.cpp's generic multi-column table (upstream), equal to ncols == 1 to
// float rounding, speed not yet measured. true (default): the ncols == 1 layout, every column bitwise equal to a
// single-column call. Set before
// graph capture; captured graphs keep the kernels they captured.
void native_mmvq_set_multi_exact(bool exact);
void native_mmvq_set_q6k_wide(bool on);   // SYCL port: the 16-byte-load Q6_K kernel (default on)
void native_q6_k_mmvq_stride224(const void* weights, const void* x_q8_1, float* y, int n_in, int n_out, int ncols,
                                void* stream);   // SYCL port: experiment, Q6_K blocks at a 224-byte stride
bool native_mmvq_multi_exact();

// One quantization may serve multiple weight matrices sharing the same input.
// Q8_1 stores FP16 scale and FP16 warp sum of the ORIGINAL float inputs; it does
// not reconstruct that sum from the quantized integers.
void native_quantize_q8_1(const float* x, void* x_q8_1, int n_in, int ncols,
                          void* stream);

void native_q5_k_mmvq(const void* weights, const void* x_q8_1, float* y,
                      int n_in, int n_out, int ncols, void* stream);

// Convenience composition: caller owns scratch sized by native_q8_1_bytes.
void native_q5_k_f32(const void* weights, const float* x, void* scratch_q8_1,
                     float* y, int n_in, int n_out, int ncols, void* stream);

void native_q2_0_mmvq(const void* weights, const void* x_q8_1, float* y,
                      int n_in, int n_out, int ncols, void* stream);

void native_q2_0_f32(const void* weights, const float* x, void* scratch_q8_1,
                     float* y, int n_in, int n_out, int ncols, void* stream);

void native_q3_k_mmvq(const void* weights, const void* x_q8_1, float* y,
                      int n_in, int n_out, int ncols, void* stream);

void native_q3_k_f32(const void* weights, const float* x, void* scratch_q8_1,
                     float* y, int n_in, int n_out, int ncols, void* stream);

void native_iq4_xs_mmvq(const void* weights, const void* x_q8_1, float* y,
                       int n_in, int n_out, int ncols, void* stream);

void native_iq4_xs_f32(const void* weights, const float* x, void* scratch_q8_1,
                      float* y, int n_in, int n_out, int ncols, void* stream);

void native_q4_k_mmvq(const void* weights, const void* x_q8_1, float* y,
                      int n_in, int n_out, int ncols, void* stream);

void native_q4_k_f32(const void* weights, const float* x, void* scratch_q8_1,
                     float* y, int n_in, int n_out, int ncols, void* stream);

void native_q6_k_mmvq(const void* weights, const void* x_q8_1, float* y,
                      int n_in, int n_out, int ncols, void* stream);

void native_q6_k_f32(const void* weights, const float* x, void* scratch_q8_1,
                     float* y, int n_in, int n_out, int ncols, void* stream);

void native_q4_0_mmvq(const void* weights, const void* x_q8_1, float* y,
                       int n_in, int n_out, int ncols, void* stream);

void native_q4_0_f32(const void* weights, const float* x, void* scratch_q8_1,
                      float* y, int n_in, int n_out, int ncols, void* stream);

void native_q5_0_mmvq(const void* weights, const void* x_q8_1, float* y,
                       int n_in, int n_out, int ncols, void* stream);

void native_q5_0_f32(const void* weights, const float* x, void* scratch_q8_1,
                      float* y, int n_in, int n_out, int ncols, void* stream);

void native_q8_0_mmvq(const void* weights, const void* x_q8_1, float* y,
                       int n_in, int n_out, int ncols, void* stream);

void native_q8_0_f32(const void* weights, const float* x, void* scratch_q8_1,
                      float* y, int n_in, int n_out, int ncols, void* stream);

void native_iq4_nl_mmvq(const void* weights, const void* x_q8_1, float* y,
                       int n_in, int n_out, int ncols, void* stream);

void native_iq4_nl_f32(const void* weights, const float* x, void* scratch_q8_1,
                      float* y, int n_in, int n_out, int ncols, void* stream);

// Storage/dispatch helpers take stable GGML type IDs, avoiding a ggml runtime
// dependency in the engine: Q4_0=2, Q5_0=6, Q8_0=8, Q3_K=11, Q4_K=12, Q5_K=13,
// Q6_K=14, IQ4_NL=20, IQ4_XS=23, Q2_0=42. Unsupported IDs throw in the byte-count
// and launch helpers; only the
// capability query returns false.
bool native_mmvq_supported(int ggml_type) noexcept;
std::size_t native_mmvq_weight_bytes(int ggml_type, int n_in, int n_out);
void native_mmvq(int ggml_type, const void* weights, const void* x_q8_1, float* y,
                 int n_in, int n_out, int ncols, void* stream);

// exp 27/28 - P0 #2 avenue 1: pre-unpack Q6_K once so the decode skips the per-token
// 6-bit unpack/gather (the measured pipe cost).  A pre-unpacked block is 32 consecutive
// signed weight bytes with the two per-16 scales Q6_K keeps (fp32, matching the packed
// kernel's dsc0/dsc1 exactly), decoded by the same load+dp4a path the Q8_0 wide32 kernel
// uses.  The one-time Q6K->Q6U transform then lets native_mmvq(14) run ~2x faster decode.
struct Q6UBlock {
    float d0, d1;      // d * scales[2b], d * scales[2b+1] for the two 16-groups of this 32
    int8_t qs[32];     // signed 6-bit values, element-aligned (element 32b+i -> byte i)
};
// size of the pre-unpacked Q6_K buffer for an n_in x n_out matrix.
// NOTE: it maps the packed Q6_K type (14) to its pre-unpacked form; not a general type helper.
std::size_t native_mmvq_q6k_preunpack_bytes(int n_in, int n_out);
// device transform: Q6KBlock array (n_out * n_in/256) -> Q6UBlock array (n_out * n_in/32).
void native_q6k_preunpack(const void* weights, void* unpacked, int n_in, int n_out, void* stream);
// decode on a pre-unpacked Q6UBlock buffer (the no-bit-unpack path).
void native_mmvq_q6k_unpacked(const void* weights, const void* x_q8_1, float* y,
                              int n_in, int n_out, int ncols, void* stream);
// route native_mmvq(14)/native_q6_k_mmvq through the pre-unpacked decode when the packed
// pointer is registered and the feature is enabled.  Shared registry for the dense K-quants
// (each callsite checks its own packed pointer, so one map serves Q6_K and Q5_K).  Default on
// via native_dense; STRATA_MMVQ_PREUNPACK=0 opts out.
void native_mmvq_set_q6k_preunpack(bool enabled);
void native_mmvq_register_q6k_preunpack(const void* packed, const void* unpacked);
void native_mmvq_unregister_q6k_preunpack(const void* packed);
void native_mmvq_clear_q6k_preunpack();

// exp 30 (P0 #5): pre-unpacked Q5_K (ggml_type 13), the min-offset analog of Q6_K.  One Q5UBlock
// per 32 elements: dsc = d*sc[g], mn1 = mn*m[g] (fp32, the two halves of the d*val - m affine
// form), qs[32] = the unsigned 5-bit code (0..31) per element.  The decode is native
// q5_q8_dot_impl WITHOUT the per-element 5-bit gather: per half, dsc*dp4a(qs,u) - mn1*ones(u).
struct Q5UBlock {
    float dsc, mn1;
    int8_t qs[32];
};
std::size_t native_mmvq_q5k_preunpack_bytes(int n_in, int n_out);
void native_q5k_preunpack(const void* weights, void* unpacked, int n_in, int n_out, void* stream);
void native_mmvq_q5k_unpacked(const void* weights, const void* x_q8_1, float* y,
                              int n_in, int n_out, int ncols, void* stream);

} // namespace strata::kernels
