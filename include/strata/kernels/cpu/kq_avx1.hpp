// include/strata/kernels/cpu/kq_avx1.hpp - the AVX1 router dot for older CPUs (from the Strata_Dirigo fork).
//
// bf16_rows_dot_multi (kq_avx2.cpp) is compiled with -mavx2 and uses instructions an AVX-only CPU does
// not have (`vpmovzxwd`, `vfmadd`), so an AVX-only host cannot call it at all.  This is the same
// arithmetic on AVX1 instructions: 8 float lanes per token, one weight stream per row shared by all
// `nt` tokens, one horizontal sum per (row, token).  AVX1 has no FMA, so the multiply and the add stay
// separate; it has no 256-bit integer ops, so the bf16 -> f32 widening runs in 128-bit halves.
#pragma once

#include <cstddef>
#include <cstdint>

#include <cstdint>

namespace strata::kernels::cpu {

/// Mirrors bf16_rows_dot_multi's arithmetic (same accumulation shape, mul+add instead of FMA).
/// Requires cols % 8 == 0 and nt <= 8, exactly like the AVX2 kernel it stands in for.
void bf16_rows_dot_multi_avx1(const uint16_t* w, int rows, int cols, const float* x, int nt, float* out);

/// ff[t][r] = silu(gate_r . act[t]) * (up_r . act[t]) for Q4_K (12) gate/up rows: each super-block is
/// expanded once for the window, then every token runs the ggml-cpu AVX1 instruction sequence over the
/// expansion, so the rows are bit-identical to ggml_vec_dot_q4_K_q8_K for any nt (kq_avx1.cpp).
void kq_gu_rows_avx1(int ggml_type, const uint8_t* blob, size_t gu_row, size_t up_off, int n,
                     const void* const* act, int nt, float* const* ff, int r0, int r1);

}  // namespace strata::kernels::cpu
