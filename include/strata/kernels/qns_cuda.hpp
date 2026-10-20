// qns_cuda.hpp - Qwen Next S ("qns") routed experts on CUDA: one expert = one qns blob (qns_blob.py), every weight row a
// 2-bit-class trellis path in a rotated basis (FORMAT.md). Built with nvcc -O3 -arch=sm_86 (no NVRTC).
//
// Activations: a token's 2560-wide input is rotated (x_rot = R (signs * x), R = H_512 (x) Q_5) and quantized to two int8
// planes, x_rot ~ step (q0 + q1 / 254), in an opaque record of qns_act_bytes() bytes. Every expert row then finishes the
// exact int32 sums C = sum (level + 54) q0 and F = sum (level + 54) q1, so an entry's output does not depend on which
// entries share its group or call.
//
// Every entry point is CUDA-graph safe: no allocation, no host synchronization, launch shapes fixed by the arguments.
#pragma once

#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>

namespace strata::kernels {

/// The rotation factors (rotation.signs.N / rotation.q.N of the package; host float32 pointers: signs [2560], Q [5][5],
/// signs [640], Q [5][5]). Copied to device symbols of the current device: call once per device, before any other call.
void qns_cuda_set_rotation(const float* signs2560, const float* q2560, const float* signs640, const float* q640);

/// Bytes of one token's quantized 2560 activation (a multiple of 16).
size_t qns_act_bytes();

/// x [n_tok][2560] float32 (device) -> act [n_tok] records of qns_act_bytes() (the same as qns_dense_quant(.., 2560, null, ..)).
void qns_quant_act(const float* x, int64_t n_tok, void* act, cudaStream_t s);

/// Bytes of scratch qns_expert_grouped needs for `cap_entries` entries (gate/up outputs and the down activations).
size_t qns_grouped_scratch_bytes(int64_t cap_entries);

/// Grouped experts, native_expert_grouped's contract: group g's blob at device address grp_ptr[g] (256-byte aligned);
/// its entries [grp_start[g], grp_start[g+1]) read token ent_tok[e]'s record of `act` and write row ent_dst[e] of `out`
/// (2560 floats: the unweighted expert output). *n_groups (device) groups; entries index the scratch, so they must be
/// < cap_entries. `grid_groups` (1 .. cap_groups; 0 = cap_groups) groups run side by side, a block row each striding
/// over the rest; the results do not depend on it.
void qns_expert_grouped(const unsigned long long* grp_ptr, const int32_t* grp_start, const int32_t* n_groups,
                        const int32_t* ent_dst, const int32_t* ent_tok, int64_t cap_groups, int64_t cap_entries,
                        const void* act, void* scratch, float* out, cudaStream_t s, int64_t grid_groups);

/// One expert's gate/up in the original basis as fp16 [1280][2560], interleaved like iq_dequant_gu_f16: row 2r = gate
/// row r, row 2r+1 = up row r.
void qns_dequant_gu_f16(const uint8_t* blob, uint16_t* dst, cudaStream_t s);

/// One expert's down projection in the original basis as fp16 [2560][640].
void qns_dequant_d_f16(const uint8_t* blob, uint16_t* dst, cudaStream_t s);

// ---- Dense backbone matrices (GDN in_proj_qkv / in_proj_z / out_proj, attention q / k / v / o): the same trellis rows and
// integer contract, stored without a blob: words [rows/32][W][32] u32 (W = cols/64*T + 1, the expert layout) and f32
// scales [rows]. cols is 2560 (= 512 x 5) or 6144 (= 2048 x 3); rows a multiple of 32; T in 3..10. A 32-row-aligned
// slice of a matrix is a matrix (words + r0/32 * W * 32, scales + r0).
// perm (GDN out_proj only, else null): x_ckpt[c] = x[perm[c]], the engine's column order of checkpoint column c.

/// The 6144 rotation's factors (host float32: signs [6144], Q [3][3] row-major). Once per device, before any capture.
void qns_cuda_set_rotation_6144(const float* signs6144, const float* q6144);

/// Bytes of one token's quantized activation for a dense matrix of `cols` columns (a multiple of 16).
size_t qns_dense_act_bytes(int cols);

/// x [n_tok][cols] float32 (device, engine column order) -> act [n_tok] records of qns_dense_act_bytes(cols): gathered by
/// perm (when not null), rotated and quantized.
void qns_dense_quant(const float* x, int64_t n_tok, int cols, const int32_t* perm, void* act, cudaStream_t s);

/// y[t * rows + r] = row r . x_t for t < n_tok (1..8): each weight chunk is decoded once for all tokens. act is only read.
void qns_dense_gemv(const uint32_t* words, const float* scales, int rows, int cols, int T, const void* act, int n_tok,
                    float* y, cudaStream_t s);

/// dst [rows][cols] fp16 in the original basis and the engine's column order: dst[r][perm[c]] = w_ckpt[r][c] (dst[r][c]
/// without perm).
void qns_dense_dequant_f16(const uint32_t* words, const float* scales, int rows, int cols, int T, const int32_t* perm,
                           uint16_t* dst, cudaStream_t s);

}  // namespace strata::kernels
