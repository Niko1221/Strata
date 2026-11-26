// src/kernels/cuda/tq_state.cuh - the trellis (APR) state window, shared by the kernels that read the trellis
// formats: iq_kernels.cu's tq_load / tq_step (the grouped kernels and the dequantizers) and i8_expert.cuh's tile
// kernel (STRATA_TQ_I8).  The two must decode the SAME bits: a step's state is read here once, in one place, so the
// int8 tile kernel is bitwise equal to the grouped kernels' decode (native_i8_parity checks the outputs).
//
// TQ2_T / TQK6 / TQK7 (engine/third_party/ggml/ggml-common-tq.h): 128 weights per block, {fp16 d; qs}.  qs is 32
// trellis steps of K bits (K = 8 / 6 / 7; TQ2_T byte-aligned, TQK bit-packed LSB-first, circular, step t's 16-bit
// window at bit (31 - t) * K; TQ2_T's state s_t = qs[(t + 31) % 32] << 8 | qs[t]).  Step t gives weights 4t..4t+3.
// Lane l of a block is its weights 16l..16l+15 = steps 4l..4l+3.
#pragma once

#include "ggml-common-tq.h"
#include "tq_lut_i8.cuh"

template<int TY> struct TqInfo;
template<> struct TqInfo<144> { using block_t = block_tq2_t; };   // byte-aligned, no K
template<> struct TqInfo<145> { using block_t = block_tqk6; static constexpr int K = 6; };
template<> struct TqInfo<146> { using block_t = block_tqk7; static constexpr int K = 7; };
template<> struct TqInfo<147> { using block_t = block_tqp5; static constexpr int K = 5; };
template<> struct TqInfo<148> { using block_t = block_tqp6; static constexpr int K = 6; };
template<> struct TqInfo<149> { using block_t = block_tqp7; static constexpr int K = 7; };

__device__ __forceinline__ float tq_h2f(uint32_t bits) { return __half2float(__ushort_as_half((unsigned short) bits)); }

// the four unscaled weights of the step with state s: (a.x, a.y, b.x, b.y), the fp16 codebook
__device__ __forceinline__ void tq_step(uint32_t s, float2& a, float2& b) {
    const uint32_t* lut = (const uint32_t*) tq_lut_f16;
    const uint32_t x = (s & TQ_STATE_MASK) * 0x9e3779b1u;
    const uint32_t pa = lut[x >> 21], pb = lut[(x >> 10) & 2047u];
    a = make_float2(tq_h2f(pa & 0xFFFFu), tq_h2f(pa >> 16));
    b = make_float2(tq_h2f(pb & 0xFFFFu), tq_h2f(pb >> 16));
}

// the four states of steps 4l..4l+3 of one TQ2_T block, from the block's bytes 4l..4l+3 and the byte before them
// (s_t = qs[(t + 31) % 32] << 8 | qs[t]): the arithmetic tq_states4 does after its loads, kept here so the int8 tile
// kernel's pack (i8_expert.cuh) extracts the SAME states from raw words it loaded earlier.
__device__ __forceinline__ void tq_states4_bytes(uint32_t b0, uint32_t b1, uint32_t b2, uint32_t b3, uint32_t prev, uint32_t s[4]) {
    s[0] = (prev << 8) | b0;
    s[1] = (b0 << 8) | b1;
    s[2] = (b1 << 8) | b2;
    s[3] = (b2 << 8) | b3;
}

// the four states of steps 4l..4l+3 of a bit-packed trellis block, from the 64-bit window that holds them at bit
// offset r (steps 4l..4l+3 lie at bits r + (3 - i) * K, i = 0..3): the arithmetic tq_states4 does after its loads.
template<int K>
__device__ __forceinline__ void tq_states4_bits(uint64_t v, int r, uint32_t s[4]) {
#pragma unroll
    for (int i = 0; i < 4; ++i) s[i] = (uint32_t) (v >> (r + (3 - i) * K)) & 0xFFFFu;
}

// the states of steps 4l..4l+3 of one block (qs is 2-byte aligned: every trellis block has an even size)
// Compute capability 6.0 (Tesla P100): the block's words are read with ld.global.nc (the weights are read-only; a generic
// pointer made these LD.E), and TQK6's lane window needs only three words (34 bits from a bit offset of 0 or 8), TQK7's
// four.  The same bits either way.
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ < 610
#define STRATA_TQ_LD16(p) ((uint32_t) __ldg((const unsigned short*) (p)))
#else
#define STRATA_TQ_LD16(p) ((uint32_t) *(p))
#endif
template<int TY>
__device__ __forceinline__ void tq_states4(const uint8_t* qs, int l, uint32_t s[4]) {
    const uint16_t* qs16 = (const uint16_t*) qs;
    if constexpr (TY == 144) {   // s_t = qs[(t + 31) % 32] << 8 | qs[t]
        const uint32_t w01 = STRATA_TQ_LD16(qs16 + 2 * l + 0), w23 = STRATA_TQ_LD16(qs16 + 2 * l + 1);
        const uint32_t prev = qs[(4 * l + 31) & 31];
        tq_states4_bytes(w01 & 0xFF, w01 >> 8, w23 & 0xFF, w23 >> 8, prev, s);
    } else {
        // step t's state: the 16 stream bits from bit (31 - t) * K, circular over 2K u16 words; steps 4l..4l+3 all
        // lie in the four words from (28 - 4l) * K >> 4
        constexpr int K = TqInfo<TY>::K, nwords = 2 * K;
        const int off3 = (28 - 4 * l) * K, wb = off3 >> 4, r = off3 & 15;
        const int wi1 = wb + 1 >= nwords ? wb + 1 - nwords : wb + 1;
        const int wi2 = wb + 2 >= nwords ? wb + 2 - nwords : wb + 2;
        const int wi3 = wb + 3 >= nwords ? wb + 3 - nwords : wb + 3;
        uint64_t v = (uint64_t) STRATA_TQ_LD16(qs16 + wb) | ((uint64_t) STRATA_TQ_LD16(qs16 + wi1) << 16) |
                     ((uint64_t) STRATA_TQ_LD16(qs16 + wi2) << 32);
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ < 610
        if constexpr (K == 6) {   // r is 0 or 8 here: bits r .. r + 33 < 48, the fourth word is never read
        } else
#endif
        v |= (uint64_t) STRATA_TQ_LD16(qs16 + wi3) << 48;
        tq_states4_bits<K>(v, r, s);
    }
}

// one step's four weights as one word of four int8, the int8 codebook (tq_lut_i8): the two 2-byte entries of the
// step, exactly the expression tq_load builds for its TqLane's q[i].  `lut` is tq_lut_i8 or its copy in shared
// memory (the tile kernel stages it, like the i-quant grids).
__device__ __forceinline__ uint32_t tq_i8_step4(uint32_t s, const uint16_t* lut) {
    const uint32_t x = (s & TQ_STATE_MASK) * 0x9e3779b1u;
    return (uint32_t) lut[x >> 21] | ((uint32_t) lut[(x >> 10) & 2047u] << 16);
}
