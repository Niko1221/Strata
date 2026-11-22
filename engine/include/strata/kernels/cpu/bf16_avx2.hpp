// include/strata/kernels/cpu/bf16_avx2.hpp - multi-token BF16 weight x F32 activation expert rows in 256-bit AVX2+FMA lanes.
#pragma once

#include <cstddef>
#include <cstdint>

namespace strata::kernels::cpu {

/// Whether rows of `n` BF16 values fit the kernel (multiple of 16).
bool bf16_256_fits(int n);

/// Gate/up BF16 rows [r0, r1) against `nt` FP32 activations of `n` values, writing activated ff[t][r].
void bf16_256_gu_rows(const uint8_t* blob, size_t gu_row, size_t up_off, int n, const void* const* act, int nt,
                      float* const* ff, int r0, int r1, bool situ, float beta, float lb);

/// Plain BF16 rows [r0, r1) against `nt` FP32 activations of `n` values.
void bf16_256_rows(const uint8_t* w, size_t row_bytes, int n, const void* const* act, int nt, float* const* out,
                   int r0, int r1);

}  // namespace strata::kernels::cpu
