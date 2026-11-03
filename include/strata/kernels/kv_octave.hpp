// include/strata/kernels/kv_octave.hpp - 3-bit KV codes on a Lloyd-Max grid (`--kv octave3`, the structure
// vLLM's Octave KV reports: random signs + Walsh-Hadamard rotation, then a fixed 8-level Gaussian codebook,
// packed 3 bits per value, read back without a dequantized copy).
//
// The rotation is the one q4_0 already ships (kv_q4.hpp), so K, V and the query all pass through fwht256 and
// <Hq, Hk> = <q, k> keeps the scores; the attention output is rotated back with H (self-inverse).
//
// 32 values per group: one fp16 scale plus three 32-bit bit-planes = 14 B, so 112 B per head per cell
// (256 values) against 144 for q4_0 and 528 for int8. A cell (2 KV heads, K and V) is 448 B against 816 for
// k8v4 (-45%) and 576 for q4_0 (-22%).
//
// THE CODEBOOK is the Lloyd-Max quantizer for N(0,1) at 3 bits: levels +-0.2451, +-0.7560, +-1.3439,
// +-2.1519, boundaries +-0.5005, +-1.0500, +-1.7479, distortion 0.0345 per coordinate. It is computed, not
// remembered (the 8-level Lloyd fixed point for N(0,1)), then held as int8 values in units of 1/32 --
// 8, 24, 43, 69 -- because that is what makes a code usable as a byte-indexed table entry (v_perm_b32) later.
// The decision thresholds below are the midpoints of the ROUNDED levels (16, 33.5, 56 in those units), so the
// quantizer is exactly nearest-neighbour for the table that is actually stored.
//
// THE SCALE keeps the projection: s = |x|^2 / (x . L), so <x, sL> = |x|^2 exactly. Octave KV reports that the
// least-squares scale (x.L / |L|^2) instead shrinks every vector, flattens the softmax and costs 2-3x the
// NLL, so this is the one to keep.
#pragma once

#include "strata/kernels/qsa.hpp"

#include <cstdint>

namespace strata::kernels {

inline constexpr int QK_OCT = 32;   // values per group

#pragma pack(push, 1)
/// 32 3-bit codes as three bit-planes: bit b of value j is bit j of word b (little-endian words).
struct block_oct3 {
    uint16_t d;        // scale (fp16 bits), multiplies the level table
    uint8_t qs[12];    // three uint32 bit-planes
};
#pragma pack(pop)

static_assert(sizeof(block_oct3) == 14, "block_oct3 must be 14 bytes");

/// Code -> level, in units of 1/32. Codes 0-3 are the positive half, 4-7 the negative one.
inline constexpr int8_t kOctLevel[8] = {8, 24, 43, 69, -8, -24, -43, -69};
/// Nearest-level thresholds on |x| * 32 (midpoints of the stored levels).
inline constexpr float kOctThreshold[3] = {16.0f, 33.5f, 56.0f};

inline uint64_t kv_oct_bytes_per_head(int head_dim) {
    return (uint64_t) (head_dim / QK_OCT) * sizeof(block_oct3);
}

/// Bytes per cell (one token, one layer): K and V of every KV head.
inline uint64_t kv_oct_bytes_per_cell(const QsaShapes& s) {
    return (uint64_t) s.n_head_kv * kv_oct_bytes_per_head((int) s.head_dim) * 2;
}

/// Quantize `n_groups` groups of QK_OCT values (row-major, ALREADY rotated by fwht256). One warp per group.
void kv_oct_quantize(const float* src, uint8_t* dst, int64_t n_groups, void* stream);

/// Dequantize the same groups back to fp32, still in the rotated frame.
void kv_oct_dequantize(const uint8_t* src, float* dst, int64_t n_groups, void* stream);

/// The engine's storage calls, one for one with kv_append_q4_* / kv_gather_q4_step: the same page table, the same
/// host copy for KV streaming, the same staging pool for the prompt path. K and V arrive ALREADY rotated by
/// fwht256, exactly as kv_append_q4's do; the host and stage pools are KvHostPools' k_q4/v_q4 byte arrays sized
/// with kv_oct_bytes_per_cell(). (KvHostPools is forward-declared here so this header stands on its own.)
struct KvHostPools;

void kv_append_oct_steps(uint8_t* k_oct, uint8_t* v_oct, const int32_t* page_table, const int32_t* step,
                         int step_stride, int n_steps, const float* kcur, const float* vcur, const QsaShapes& s,
                         void* stream, const KvHostPools* host = nullptr);
void kv_append_oct_step(uint8_t* k_oct, uint8_t* v_oct, const int32_t* page_table, const int32_t* step,
                        const float* kcur, const float* vcur, const QsaShapes& s, void* stream,
                        const KvHostPools* host = nullptr);
void kv_append_oct(uint8_t* k_oct, uint8_t* v_oct, const int32_t* page_table, int64_t pos0, int64_t T, const float* K,
                   const float* V, const QsaShapes& s, void* stream, const KvHostPools* host = nullptr,
                   const KvHostPools* stage = nullptr);
void kv_gather_oct_step(const uint8_t* k_oct, const uint8_t* v_oct, const int32_t* page_table, const int32_t* ids,
                        const int32_t* step, int64_t max_ids, const QsaShapes& s, uint16_t* k_scratch,
                        uint16_t* v_scratch, void* stream);

/// The host reference the GPU kernels are checked against: same pairwise summation tree (strides 16, 8, 4, 2, 1)
/// and the same constants, so the bytes match bit for bit.
void kv_oct_quantize_group_host(const float* x, block_oct3& blk);
void kv_oct_dequantize_group_host(const block_oct3& blk, float* x);

}  // namespace strata::kernels
