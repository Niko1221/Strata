// engine/src/inkling2/ink_host_avx2.h - Runtime-dispatched AVX2+FMA host dot kernels for InkHybridEngine
// Compiled in a separate TU (ink_host_avx2.cpp) with -mavx2 -mfma -mf16c (/arch:AVX2) and ONLY called
// after strata::quant::caps().avx2 is verified at runtime (fixes #6618 SIGILL on non-AVX2 CPUs).
// MIT License - Copyright (c) 2026 Strata contributors
#pragma once

#include <cstdint>

namespace strata::ink::host_avx2 {

// Returns true if this binary includes the AVX2+FMA host TU AND the running CPU supports AVX2+FMA+F16C.
bool available() noexcept;

float dot_sf4_row(const uint8_t* p, const uint8_t* sc, float w_outer, const float* x, int cols,
                  const float* e4_lut) noexcept;
float dot_mx8_row(const uint8_t* w, const uint8_t* sc, const float* x, int cols,
                  const float* e4_lut, const float* u8_lut) noexcept;
float dot_sk_sub64(const uint8_t* pl, int fmt, int g, const int8_t* qs, float x_d, float x_s,
                   float eff, const int8_t (*s15_lut)[5]) noexcept;

// Whole S_K rows against `T` activation vectors in Q8_1_G64 blocks (80 bytes: qs[64], d, s; xq[t] points at token t's blocks).
// Row layout: n_sb super blocks of sb_bytes (f16 scale, 4 sub-scales, payload at +6), 4 groups of 64 per super block. The weights of a
// group are unpacked once for all T tokens. The float sums are added in the same order as the per-group function above, so the result
// is the same as calling dot_sk_sub64 group by group. fmt: 2, 3, 4, 5, 6, 8 are done with SIMD unpacking; other codes (S1_5)
// return false and the caller uses the per-group path.
bool dot_sk_row_multi(const uint8_t* row, int fmt, int sb_bytes, int n_sb, const uint8_t* const* xq, int T, float* out) noexcept;
// Down projection row pair: the first 18 groups of the pair row belong to output row 2r (activation blocks 0..17), the next 18 to
// row 2r+1 (activation blocks 0..17 again); out0[t], out1[t] receive the two sums.
bool dot_sk_pair_multi(const uint8_t* row, int fmt, int sb_bytes, const uint8_t* const* xq, int T, float* out0, float* out1) noexcept;

}  // namespace strata::ink::host_avx2
