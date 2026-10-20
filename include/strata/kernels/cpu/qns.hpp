// include/strata/kernels/cpu/qns.hpp - Qwen Next S routed experts (trellis "qns" blobs, qns_blob.py) on AVX2.
//
// A blob row is a trellis path in a rotated basis: the input is rotated (x_rot = (H (x) Q)(signs * x)) and quantized
// per token to x_rot ~ step * (q0 + q1 / 254) (two int8 planes), the kernel computes per row the exact integer
// S = sum (level + 54) * (254 q0 + q1) = 254 C + F (C, F the two planes' sums of the GPU kernels), and finishes
//     y = scale * ((step / RMS) * (S - 54 * sum (254 q0 + q1)) / 254 + d0 * E + d1 * O),   d_j = -MEAN_j / RMS
// with E, O = step * sum (q0 + q1 / 254) over the even / odd columns. Same contract as the vLLM plugin's kernels.cu.
#pragma once

#include <cstddef>
#include <cstdint>

namespace strata::kernels::cpu::qns {

/// One token's quantized activation (4-byte aligned): a 16-byte header {float step; int32 sum of 254 q0 + q1;
/// float E; float O}, then int16 254 q0 + q1 of the rotated columns, each 4 columns 4g .. 4g + 3 stored as 4g, 4g + 2,
/// 4g + 1, 4g + 3.
inline constexpr size_t kActBytes = 16 + 2 * 2560;  ///< expert input x (2560)
inline constexpr size_t kHBytes = 16 + 2 * 640;     ///< down input h = silu(gate) * up (640)

/// The rotation factors (rotation.signs.N / rotation.q.N of the package, q row-major 5 x 5). Once, at load.
void set_rotation(const float* signs2560, const float* q2560, const float* signs640, const float* q640);
/// x (2560 floats) -> kActBytes.
void quant_act(const float* x, void* dst);
/// h (640 floats) -> kHBytes.
void quant_h(const float* h, void* dst);

/// ff[t][r] = silu(gate_r . x_t) * (up_r . x_t) for rows r in [r0, r1) of [0, 640), nt (1..8) tokens. Rows go in
/// blocks of 8: a range that cuts a block computes the whole block (split ranges at multiples of 8).
void gu_rows(const uint8_t* blob, const void* const* act, int nt, float* const* ff, int r0, int r1);
/// out[t][r] = down_r . h_t for rows r in [r0, r1) of [0, 2560), nt (1..8) tokens, blocks of 8 as gu_rows.
void down_rows(const uint8_t* blob, const void* const* hq, int nt, float* const* out, int r0, int r1);

/// The kernels' core, exposed for the exactness test: s[t][i] = S of row 8 b + i of the gate_up matrix (rows 0-639
/// gate, 640-1279 up; down = false) or of the down matrix, for act[0 .. nt).
void block_sums(const uint8_t* blob, bool down, int b, const void* const* act, int nt, int64_t (*s)[8]);

}  // namespace strata::kernels::cpu::qns
