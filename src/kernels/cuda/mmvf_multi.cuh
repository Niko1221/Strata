// src/kernels/cuda/mmvf_multi.cuh - one block's row of the FP32-activation BF16 MMVF for up to 8 activation rows, for
// bf16_gemv_fp32_mmvf_multi (native_bf16.cu) and the verify window's router (native_router.cu); both translation
// units compile with --use_fast_math, the pinned contract's arithmetic.
//
// Adapted from llama.cpp 3cf03257f219afbe7334045ff7c6a06ac68c627d, ggml/src/ggml-cuda/{mmvf.cu,common.cuh}.
// MIT License, Copyright (c) 2023-2026 The ggml authors (see native_bf16.cu for the full notice).
#pragma once

#include "strata/kernels/bf16_bits.hpp"

#include <cuda_runtime.h>

#include <cstdint>

namespace strata::kernels {
namespace {

constexpr int kMmvfMaxRows = 8;

__device__ __forceinline__ float mmvf_warp_sum(float value) {
#pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1)
        value += __shfl_xor_sync(0xffffffffu, value, offset, 32);
    return value;
}

// `bf16_f32_mmvf_kernel` for up to kMmvfMaxRows activation rows (x: n_tok rows of n_in): each thread walks the same
// pairs in the same order for every row, and each row is reduced exactly as the single-row kernel reduces it; thread 0
// ends with the sums in acc[0 .. n_tok).
template <int BLOCK_SIZE>
__device__ __forceinline__ void mmvf_multi_row(const float* __restrict__ x, const uint16_t* __restrict__ row, int n_in,
                                               int n_tok, float (&acc)[kMmvfMaxRows]) {
    const int t = threadIdx.x;
    const uint32_t* weights2 = reinterpret_cast<const uint32_t*>(row);
    __shared__ float partials[kMmvfMaxRows][32];
    if constexpr (BLOCK_SIZE > 32) {
        if (t < 32)
            for (int k = 0; k < kMmvfMaxRows; ++k) partials[k][t] = 0.0f;
        __syncthreads();
    }
#pragma unroll
    for (int k = 0; k < kMmvfMaxRows; ++k) acc[k] = 0.0f;
    for (int pair = t; pair < n_in / 2; pair += BLOCK_SIZE) {
        const uint32_t weight = weights2[pair];
        const float w0 = f32_from_bf16((uint16_t) weight), w1 = f32_from_bf16((uint16_t) (weight >> 16));
#pragma unroll
        for (int k = 0; k < kMmvfMaxRows; ++k) {
            if (k >= n_tok) break;
            const float2 input = reinterpret_cast<const float2*>(x + (size_t) k * n_in)[pair];
            acc[k] = __fmaf_rn(w0, input.x, acc[k]);
            acc[k] = __fmaf_rn(w1, input.y, acc[k]);
        }
    }
#pragma unroll
    for (int k = 0; k < kMmvfMaxRows; ++k) {
        if (k >= n_tok) break;
        acc[k] = mmvf_warp_sum(acc[k]);
    }
    if constexpr (BLOCK_SIZE > 32) {
        if ((t & 31) == 0)
            for (int k = 0; k < n_tok; ++k) partials[k][t / 32] = acc[k];
        __syncthreads();
        if (t < 32) {
#pragma unroll
            for (int k = 0; k < kMmvfMaxRows; ++k) {
                if (k >= n_tok) break;
                acc[k] = mmvf_warp_sum(partials[k][t]);
            }
        }
    }
}

}  // namespace
}  // namespace strata::kernels
