// src/quant/sk_tiers.hpp - the per-ISA-tier copies of the S_K / SF4 / MX8 codecs (internal; never included by a public header).
// sk_codecs.cpp is compiled once per tier (sk_codecs_base.cpp, x86/sk_codecs_avx2.cpp, x86/sk_codecs_avx512.cpp) when STRATA_SK_X86_TIERS is defined,
// each into its own namespace; sk_dispatch.cpp defines the public strata::quant::sk functions and forwards to the tier the CPU can run.
// A test calls the tiers directly to compare them.
#pragma once

#include "strata/quant/sk_codecs.hpp"

#define STRATA_SK_TIER_API                                                                                                          \
    void quantize_q8_1_g64(const float* x, BlockQ8_1_G64* out, size_t n) noexcept;                                                  \
    void fwht64_f32(const float* in, float* out, size_t n) noexcept;                                                                \
    void dequant_sk_row(const uint8_t* raw, SkCode code, float* out, size_t n) noexcept;                                            \
    float dot_sk_q8_1_row(const uint8_t* raw, SkCode code, const BlockQ8_1_G64* x, size_t n) noexcept;                              \
    void gemm_sk_q8_1(const uint8_t* weights, SkCode code, const BlockQ8_1_G64* xs, float* out, size_t nrows, size_t ncols,        \
                      size_t m) noexcept;                                                                                            \
    void dequant_sf4_row(const uint8_t* packed_e2m1, const uint8_t* scales_e4m3, float outer_scale, float* out,                     \
                         size_t ncols) noexcept;                                                                                     \
    float dot_sf4_f32_row(const uint8_t* packed_e2m1, const uint8_t* scales_e4m3, float outer_scale, const float* x,               \
                          size_t ncols) noexcept;                                                                                    \
    void dequant_mx8_row(const uint8_t* weights_e4m3, const uint8_t* scales_ue8m0, float* out, size_t ncols) noexcept;              \
    float dot_mx8_f32_row(const uint8_t* weights_e4m3, const uint8_t* scales_ue8m0, const float* x, size_t ncols) noexcept;         \
    float dot_f32(const float* a, const float* b, size_t n) noexcept;                                                              \
    void dequant_paired2_sk_rows(const uint8_t* pair_raw, SkCode code, float* out_row0, float* out_row1, size_t ncols) noexcept;    \
    void gemv_paired2_sk_q8_1(const uint8_t* weights, SkCode code, const BlockQ8_1_G64* x, float* out, size_t nrows,               \
                              size_t ncols) noexcept;                                                                                \
    void gemm_paired2_sk_q8_1(const uint8_t* weights, SkCode code, const BlockQ8_1_G64* xs, float* out, size_t nrows,              \
                              size_t ncols, size_t m) noexcept;

namespace strata::quant::sk {

namespace base   { STRATA_SK_TIER_API }   // plain C++ (aarch64 builds do not use the tiers: NEON is the one compile-time tier there)
namespace avx2   { STRATA_SK_TIER_API }   // -mavx2 -mfma -mf16c
namespace avx512 { STRATA_SK_TIER_API }   // -mavx512f/bw/vl/dq/vnni + avx2: needs caps().avx512 && caps().avx512_vnni

}  // namespace strata::quant::sk
