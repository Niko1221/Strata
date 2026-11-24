// Adapted from llama.cpp 3cf03257f219afbe7334045ff7c6a06ac68c627d:
// ggml/src/ggml-cuda/{quantize.cu,vecdotq.cuh,mmvq.cu,common.cuh}
// and ggml/src/ggml-common.h.
//
// MIT License
// Copyright (c) 2023-2026 The ggml authors
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.

// Frozen Q4_K pair oracle from Strata 8d7bb980. Keep independent of the candidate.
// Baseline native_mmvq.cu SHA256: 3af8d882a5d28fbc88f356945c4ab1f85e8a657c58258dc84dc6d363dd7e0ec0
#pragma once
#include "strata/kernels/dp4a.hpp"
#include "strata/kernels/pdl.hpp"
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <cstdint>
namespace q4_reference {
using strata::kernels::pdl_trigger;
using strata::kernels::pdl_wait;
constexpr int QK=256, Q8K=32, QI=32, VDR=2, WARPS=4, WARP=32;
struct Q4KBlock { half2 dm; uint8_t scales[12]; uint8_t qs[128]; };
struct Q81Block { half2 ds; int8_t qs[32]; };
static_assert(sizeof(Q4KBlock)==144 && sizeof(Q81Block)==36);
__device__ __forceinline__ float warp_sum(float x) {
#pragma unroll
    for (int offset = WARP / 2; offset > 0; offset >>= 1) {
        x += __shfl_xor_sync(0xffffffff, x, offset, WARP);
    }
    return x;
}

// Exact pinned vec_dot_q4_K_q8_1_impl_vmmq expression and integer dot order.
__device__ __forceinline__ float q4_q8_dot_impl(
    const int* __restrict__ v, const int* __restrict__ u,
    const uint8_t* __restrict__ sc, const uint8_t* __restrict__ m, const half2& dm4,
    const float* __restrict__ d8) {
    float sumf_d = 0.0f;
    float sumf_m = 0.0f;
#pragma unroll
    for (int i = 0; i < 2; ++i) {
        const int v0i = (v[0] >> (4 * i)) & 0x0f0f0f0f;
        const int v1i = (v[1] >> (4 * i)) & 0x0f0f0f0f;
        const int dot1 = STRATA_DP4A(v1i, u[2 * i + 1], STRATA_DP4A(v0i, u[2 * i], 0));
        const int dot2 = STRATA_DP4A(0x01010101, u[2 * i + 1], STRATA_DP4A(0x01010101, u[2 * i], 0));
        sumf_d += d8[i] * (dot1 * sc[i]);
        sumf_m += d8[i] * (dot2 * m[i]);
    }
    const float2 dm4f = __half22float2(dm4);
    return dm4f.x * sumf_d - dm4f.y * sumf_m;
}

__device__ __forceinline__ float q4_q8_dot(const Q4KBlock* __restrict__ bq4,
                                          const Q81Block* __restrict__ bq8, int iqs) {
    int v[2];
    int u[4];
    float d8[2];
    const int bq8_offset = 2 * ((iqs / 2) / 4);
    const int* ql = reinterpret_cast<const int*>(bq4->qs + 16 * bq8_offset + 4 * ((iqs / 2) % 4));
    v[0] = ql[0];
    v[1] = ql[4];

    const uint16_t* scales = reinterpret_cast<const uint16_t*>(bq4->scales);
    const int j = bq8_offset / 2;
    const int jm = j & 1;
    const uint32_t s0 = scales[jm];
    const uint32_t s2 = scales[jm + 2];
    const uint32_t s4 = scales[jm + 4];
    const uint32_t hi = uint32_t(-int32_t(j >= 2));
    uint16_t aux[2];
    aux[0] = uint16_t(((s0 & 0x3f3f) & ~hi) |
                     ((((s4 >> 0) & 0x0f0f) | ((s0 & 0xc0c0) >> 2)) & hi));
    aux[1] = uint16_t(((s2 & 0x3f3f) & ~hi) |
                     ((((s4 >> 4) & 0x0f0f) | ((s2 & 0xc0c0) >> 2)) & hi));
    const uint8_t* sc = reinterpret_cast<const uint8_t*>(aux);
    const uint8_t* m = sc + 2;
#pragma unroll
    for (int i = 0; i < 2; ++i) {
        const Q81Block* bq8i = bq8 + bq8_offset + i;
        d8[i] = __low2float(bq8i->ds);
        const int* q8 = reinterpret_cast<const int*>(bq8i->qs) + ((iqs / 2) % 4);
        u[2 * i] = q8[0];
        u[2 * i + 1] = q8[4];
    }
    return q4_q8_dot_impl(v, u, sc, m, bq4->dm, d8);
}

template<bool SmallK, bool Pdl = false, bool Grouped = true, bool Pair = false, int RowsN = 0>
__launch_bounds__(WARPS * WARP, 1)
__global__ void reference_kernel(const Q4KBlock* __restrict__ w,
                                        const Q81Block* x_,
                                        float* y_, int n_in, int n_out,
                                        std::size_t w_group = 0, std::size_t x_group = 0,
                                        const int* w_index = nullptr,
                                        const Q4KBlock* __restrict__ w2 = nullptr, float* y2_ = nullptr,
                                        std::size_t w2_group = 0, unsigned x_blocks = 0) {
    const Q81Block* STRATA_PDL_RESTRICT x = x_;   // __restrict__ below sm_70 only (pdl.hpp, #1469)
    float* STRATA_PDL_RESTRICT y = y_;
    unsigned bx = blockIdx.x;
    if constexpr (Pair) {
        if (bx >= x_blocks) {   // the upper half of the grid is the second matrix
            bx -= x_blocks;
            w = w2;
            y = y2_;
            w_group = w2_group;
        }
    }
    if constexpr (Grouped) {
        x += blockIdx.y * x_group;                // pointer arithmetic only: safe before the wait
        y += blockIdx.y * std::size_t(n_out);
    }
    // PDL (pdl.hpp): let the next projection start loading its weights, then wait for the activations this
    // one reads. `w_index` IS NOT A WEIGHT - the routing kernel writes it and may still be running behind a
    // programmatic edge - so it is read after the wait like any other activation.
    constexpr int ROWS = RowsN > 0 ? RowsN : (SmallK ? WARPS : 1);
    static_assert(ROWS <= WARP, "the store below selects a row by lane, so ROWS cannot exceed a warp");
    constexpr int BLOCKS_PER_ITER = VDR * WARPS * WARP / QI;
    const int tid = WARP * int(threadIdx.y) + int(threadIdx.x);
    const int row0 = ROWS * int(bx);
    const int blocks_per_row = n_in / QK;
    const int kqs = VDR * (tid % (QI / VDR));   // hoisted: loop-invariant, and the prefetch wants it too
    float tmp[ROWS] = {};
    int kbx = tid / (QI / VDR);
    if constexpr (Pdl) {
        pdl_trigger();   // the next projection may start loading its own weights now
        pdl_wait();      // and from here every read sees what the kernel in front of this one wrote
    }
    if constexpr (Grouped) {
        if (w_index != nullptr && w_index[blockIdx.y] < 0) return;
    }
    if constexpr (Grouped)
        w += (w_index != nullptr ? (std::size_t) w_index[blockIdx.y] : (std::size_t) blockIdx.y) * w_group;
    for (; kbx < blocks_per_row; kbx += BLOCKS_PER_ITER) {
        const int kby = kbx * (QK / Q8K);
#pragma unroll
        for (int i = 0; i < ROWS; ++i) {
            // The source assumes allocator padding for partial row groups. This
            // guard preserves every valid row's math without an out-of-bounds read.
            if (row0 + i < n_out) {
                const std::size_t block = std::size_t(row0 + i) * blocks_per_row + kbx;
                tmp[i] += q4_q8_dot(w + block, x + kby, kqs);
            }
        }
    }
    __shared__ float partial[WARPS - 1][ROWS][WARP];
    if (threadIdx.y > 0) {
#pragma unroll
        for (int i = 0; i < ROWS; ++i) partial[threadIdx.y - 1][i][threadIdx.x] = tmp[i];
    }
    __syncthreads();
    if (threadIdx.y > 0) return;
#pragma unroll
    for (int i = 0; i < ROWS; ++i) {
#pragma unroll
        for (int l = 0; l < WARPS - 1; ++l) tmp[i] += partial[l][i][threadIdx.x];
        tmp[i] = warp_sum(tmp[i]);
        if (threadIdx.x == i && row0 + i < n_out) y[row0 + i] = tmp[i];
    }
}


template<int Rows>
void launch(const void* gate,std::size_t gs,const void* up,std::size_t us,
            const void* x,std::size_t xs,float* yg,float* yu,int ni,int no,int groups,
            cudaStream_t stream,const int32_t* ids) {
 const unsigned half=unsigned((std::size_t(no)+Rows-1)/Rows);
 strata::kernels::launch_pdl_pair(reference_kernel<true,true,true,true,Rows>,
     reference_kernel<true,false,true,true,Rows>,dim3(2*half,unsigned(groups)),dim3(WARP,WARPS),0,stream,
     static_cast<const Q4KBlock*>(gate),static_cast<const Q81Block*>(x),yg,ni,no,
     gs/144,xs/36,ids,static_cast<const Q4KBlock*>(up),yu,us/144,half);
}
}
