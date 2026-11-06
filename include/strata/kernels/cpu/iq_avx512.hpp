// include/strata/kernels/cpu/iq_avx512.hpp - plan v0.3 P6: AVX-512 multi-token dot products for the i-quant
// expert formats (IQ2_XXS, IQ2_XS, IQ2_S, IQ3_XXS, IQ3_S) against Q8_K activations (ggml's block_q8_K).
#pragma once

#include <cstddef>
#include <cstdint>

namespace strata::kernels::cpu {

bool iq512_supported(int ggml_type) noexcept;
/// Whether the grid lookups of these rows are gathers (the default where cpu_gather_fast(), STRATA_IQ_GATHER=1 / 0 force it): the
/// same words, so the same bits as the scalar lookups, and 1.3-2x faster where a gather is cheap (Sapphire Rapids).
bool iq512_gather_on() noexcept;
/// ff[t][r] = silu(gate_r . a[t]) * (up_r . a[t]), rows [r0, r1); gate rows at blob, up rows at blob + up_off.
void iq512_gu_rows(int ggml_type, const uint8_t* blob, size_t gu_row, size_t up_off, int n, const void* const* act,
                   int nt, float* const* ff, int r0, int r1);
/// out[t][r] = w_r . a[t], rows [r0, r1).
void iq512_rows(int ggml_type, const uint8_t* w, size_t row_bytes, int n, const void* const* act, int nt,
                float* const* out, int r0, int r1);

/// The DeepSeek-V4.1 CPU lane's rows (src/dsv41/ds4_gg_cpu.cpp): the same 64-value decode, summed in ggml's order, so a token's
/// dot is ggml-cpu's AVX2 vec_dot bit for bit at any number of tokens (1-6 in one pass, 7 and more as a prompt chunk's tiles).
/// The caller checks cpu_avx512_ok() (AVX-512 F/BW/VL/DQ/VNNI/VBMI).
bool iq512x_supported(int ggml_type) noexcept;
void iq512x_rows(int ggml_type, const uint8_t* w, size_t row_bytes, int n, const void* const* act, int nt,
                 float* const* out, int r0, int r1);

/// Singleton-equivalent accumulation for types 16, 17, 18, 21, 22, 23 (the linked ggml AVX2 order, one token or many);
/// caller checks the linked ggml ISA.  Used only for window-invariant native arithmetic.
void iq512_reference_gu_rows(int type, const uint8_t* blob, size_t gu_row, size_t up_off, int n,
                             const void* const* act, int nt, float* const* ff, int r0, int r1);

/// AVX-512 VNNI/VBMI multi-token down-projection rows for IQ4_NL (type 20) and Q2_0 (type 42) against Q8_0 activations,
/// bit-exact against ggml-cpu's AVX2 vec_dot / q2_0_q8_0_dot_avx2 for 1..N tokens.
void iq4nl512_down_rows(const uint8_t* w, size_t row_bytes, int n, const void* const* hq, int nt,
                        float* const* out, int r0, int r1);
void q2_512_down_rows(const uint8_t* w, size_t row_bytes, int n, const void* const* hq, int nt,
                      float* const* out, int r0, int r1);

/// AVX-512 VNNI multi-token K-quant rows: Q4_K (12), Q5_K (13), Q6_K (14), Q5_0 (6), Q5_1 (7), Q8_0 (8),
/// bit-exact against ggml-cpu's AVX2 vec_dot for 1..N tokens.
bool kq512_gu_supported(int ggml_type) noexcept;
bool kq512_supported(int ggml_type) noexcept;
void kq512_gu_rows(int ggml_type, const uint8_t* blob, size_t gu_row, size_t up_off, int n, const void* const* act,
                   int nt, float* const* ff, int r0, int r1);
void kq512_rows(int ggml_type, const uint8_t* w, size_t row_bytes, int n, const void* const* act, int nt,
                float* const* out, int r0, int r1);

/// AVX-512 Q8_K activation quantizer: byte-for-byte identical to quantize_row_q8_K_ref and q8k_quant_avx2.
void q8k_quant_avx512(const float* x, void* y, int64_t k);

/// AVX-512 BF16 row dots for router and shared/dense projections.
void bf16_rows_dot_512(const uint16_t* w, int rows, int cols, const float* x, float* out);
void bf16_rows_dot_multi_512(const uint16_t* w, int rows, int cols, const float* x, int nt, float* out);

}  // namespace strata::kernels::cpu
