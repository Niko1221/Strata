// src/kernels/cpu/q2_lut_kernels.hpp - internal to q2_lut.cpp and its two kernel units: the mirror layout and the
// kernels' entry points.  See include/strata/kernels/cpu/q2_lut.hpp.
#pragma once

#include "strata/kernels/cpu/expert.hpp"

#include <cstdint>

namespace strata::kernels::cpu::q2lut {

inline constexpr int TR = 16;              // rows per tile: one pshufb lane per row
inline constexpr int REC = 2 * TR + 256;   // one tile x one 64-value block: 16 fp16 scales, 16 index vectors
inline constexpr int TBL = 32;             // table bytes per column pair: an A plane and a B plane

// Mirror record (tile t, block b) at mir[(t * nblk + b) * REC]: bytes 0..31 the tile's 16 fp16 row scales, then
// index vector v = 0..15 at 32 + 16v covering the block's values 4v..4v+3: lane r = n(r, 2v) | n(r, 2v+1) << 4 with
// n(r, p) = code(2p) | code(2p+1) << 2.  Rows past the matrix are zero codes with a zero scale.  18 bytes per row
// and block, as in the GGUF rows.
//
// Tables, per token and column pair p (k / 2 pairs, TBL bytes each, 16-byte aligned): T + 762 = 32 A + B with
// T[n] = q[2p] * (n & 3) + q[2p+1] * (n >> 2); the A plane (A <= 47) then the B plane (B <= 31).

/// Rows [r0, r1) of a mirrored matrix (nblk blocks per row), nt tokens: q2_lut_avx1.cpp with F16C (Ivy Bridge),
/// q2_lut_snb.cpp without (Sandy Bridge).  Each row's result is the same whatever r0, r1 and nt.
void rows_f16c(const uint8_t* mir, int nblk, const uint8_t* const* tbl, const ActQ* const* a, int nt,
               float* const* out, int r0, int r1);
void rows_nof16c(const uint8_t* mir, int nblk, const uint8_t* const* tbl, const ActQ* const* a, int nt,
                 float* const* out, int r0, int r1);

/// One token's tables from its int8 activation q[0..k) (q2_lut_snb.cpp: needs only SSSE3).
void build_tables(const int8_t* q, int k, uint8_t* tbl);

}  // namespace strata::kernels::cpu::q2lut
