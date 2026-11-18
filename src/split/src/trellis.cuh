// The APR trellis expert types (ggml 144 TQ2_T, 145 TQK6, 146 TQK7: Agention's Gyro-M / Gyro-S files) for the split engine's
// decode dots. The block layouts and the 2048-point codebook are third_party/ggml/ggml-common-tq.h; the int8 codebook (one global
// scale) is the main engine's src/kernels/cuda/tq_lut_i8.cuh. tq_states4 / tq_load / tq_apply are the main engine's
// (src/kernels/cuda/iq_kernels.cu, "APR trellis types"), unchanged apart from the names, so a weight lane and its q8_1 activations
// give the same integer sum and the same float product as the main engine's vec_dot_tq_q8_1.
//
// A 128-value block is 8 lanes of 4 trellis steps (16 values). In the split's dot convention (qdot.cuh: vdot(v, y, kbx, iqs), y the
// q8_1 blocks aligned with block kbx's start) a call takes iqs = 4 l, l the lane: qk 128, qi 32 ints, vdr 4.
#pragma once

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cstdint>

#include "../../../third_party/ggml/ggml-common-tq.h"     // block_tq2_t / block_tqk6 / block_tqk7, tq_lut_f16, TQ_STATE_MASK
#include "../../kernels/cuda/tq_lut_i8.cuh"               // tq_lut_i8, TQ_LUT_I8_SCALE

namespace fnx {

template <int TY> struct TqInfo;
template <> struct TqInfo<144> { using block_t = block_tq2_t; static constexpr int bytes = 34; };   // byte-aligned, no K
template <> struct TqInfo<145> { using block_t = block_tqk6; static constexpr int K = 6; static constexpr int bytes = 26; };
template <> struct TqInfo<146> { using block_t = block_tqk7; static constexpr int K = 7; static constexpr int bytes = 30; };


__device__ __forceinline__ float tq_h2f(uint32_t bits) { return __half2float(__ushort_as_half((unsigned short) bits)); }

// the states of steps 4l..4l+3 of one block (qs is 2-byte aligned: every trellis block has an even size)
template <int TY>
__device__ __forceinline__ void tq_states4(const uint8_t* qs, int l, uint32_t s[4]) {
    const uint16_t* qs16 = (const uint16_t*) qs;
    if constexpr (TY == 144) {   // s_t = qs[(t + 31) % 32] << 8 | qs[t]
        const uint32_t w01 = qs16[2 * l + 0], w23 = qs16[2 * l + 1];
        const uint32_t prev = qs[(4 * l + 31) & 31];
        const uint32_t b0 = w01 & 0xFF, b1 = w01 >> 8, b2 = w23 & 0xFF, b3 = w23 >> 8;
        s[0] = (prev << 8) | b0;
        s[1] = (b0 << 8) | b1;
        s[2] = (b1 << 8) | b2;
        s[3] = (b2 << 8) | b3;
    } else {
        // step t's state: the 16 stream bits from bit (31 - t)*K, circular over 2K u16 words; steps 4l..4l+3 all lie in the four
        // words from (28 - 4l)*K >> 4
        constexpr int K = TqInfo<TY>::K, nwords = 2 * K;
        const int off3 = (28 - 4 * l) * K, wb = off3 >> 4, r = off3 & 15;
        const int wi1 = wb + 1 >= nwords ? wb + 1 - nwords : wb + 1;
        const int wi2 = wb + 2 >= nwords ? wb + 2 - nwords : wb + 2;
        const int wi3 = wb + 3 >= nwords ? wb + 3 - nwords : wb + 3;
        const uint64_t v = (uint64_t) qs16[wb] | ((uint64_t) qs16[wi1] << 16) | ((uint64_t) qs16[wi2] << 32) | ((uint64_t) qs16[wi3] << 48);
#pragma unroll
        for (int i = 0; i < 4; ++i) s[i] = (uint32_t) (v >> (r + (3 - i) * K)) & 0xFFFFu;
    }
}

// a lane's 16 int8 codebook values (4 ints of packed int8) and the block's scale: decoded once, applied to any number of rows
struct TqLane {
    int q[4];
    float d;
};
template <int TY>
__device__ __forceinline__ TqLane tq_load(const void* __restrict__ vbq, int kbx, int iqs) {
    const auto* b = (const typename TqInfo<TY>::block_t*) vbq + kbx;
    uint32_t s[4];
    tq_states4<TY>(b->qs, iqs / 4, s);
    TqLane r;
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        const uint32_t x = (s[i] & TQ_STATE_MASK) * 0x9e3779b1u;
        r.q[i] = (int) ((uint32_t) tq_lut_i8[x >> 21] | ((uint32_t) tq_lut_i8[(x >> 10) & 2047u] << 16));
    }
    r.d = tq_h2f(b->d);
    return r;
}
// the lane against the q8_1 blocks aligned with the weight block's start: lane l reads q8_1 block l / 2, ints 4 (l & 1) ..
__device__ __forceinline__ float tq_apply(const TqLane& r, const block_q8_1* __restrict__ bq8_1, int iqs) {
    const int l = iqs / 4;
    const block_q8_1* b8 = bq8_1 + l / 2;
    const int* q8 = (const int*) b8->qs + 4 * (l & 1);
    int sumi = 0;
#pragma unroll
    for (int i = 0; i < 4; ++i) sumi = qd_dp4a(r.q[i], q8[i], sumi);
    return r.d * (__low2float(b8->ds) * TQ_LUT_I8_SCALE) * (float) sumi;
}
template <int TY>
__device__ __forceinline__ float vec_dot_tq_q8_1(const void* __restrict__ vbq, const block_q8_1* __restrict__ bq8_1, const int& kbx,
                                                 const int& iqs) {
    return tq_apply(tq_load<TY>(vbq, kbx, iqs), bq8_1, iqs);
}

// the f16 weight the prompt path's tiles use for value j (0..127) of a block: f16(d * point), d and the point f16, the product in f32,
// as the main engine's dq_tq and ggml's to_float then a round to f16
template <int TY>
__device__ __forceinline__ void tq_step_f(uint32_t s, float d, float* y) {
    const uint32_t* lut = (const uint32_t*) tq_lut_f16;
    const uint32_t x = (s & TQ_STATE_MASK) * 0x9e3779b1u;
    const uint32_t pa = lut[x >> 21], pb = lut[(x >> 10) & 2047u];
    y[0] = d * tq_h2f(pa & 0xFFFFu);
    y[1] = d * tq_h2f(pa >> 16);
    y[2] = d * tq_h2f(pb & 0xFFFFu);
    y[3] = d * tq_h2f(pb >> 16);
}

}  // namespace fnx
