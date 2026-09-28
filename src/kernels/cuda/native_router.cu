// Adapted from topk-moe.cu/common.cuh in llama.cpp
// 3cf03257f219afbe7334045ff7c6a06ac68c627d; finite F32, 512-expert/10-output path.
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
#include "strata/kernels/native_router.hpp"
#include "strata/kernels/bf16_bits.hpp"
#include "hit_plan.cuh"
#include "mmvf_multi.cuh"
#include <cuda_runtime.h>
#include <atomic>
#include <cfloat>
#include <cstddef>
#include <cstdint>
#include <stdexcept>

namespace strata::kernels {
namespace {
std::atomic<bool> enabled{false};
__device__ __forceinline__ float warp_sum(float value) {
#pragma unroll
    for (int mask = 16; mask; mask >>= 1) value += __shfl_xor_sync(0xffffffffu, value, mask, 32);
    return value;
}
__device__ __forceinline__ float warp_max(float value) {
#pragma unroll
    for (int mask = 16; mask; mask >>= 1) value = fmaxf(value, __shfl_xor_sync(0xffffffffu, value, mask, 32));
    return value;
}
// A float as a uint32 in the same order (sign flipped for positives, all bits for negatives), and back.
__device__ __forceinline__ unsigned order_key(float f) {
    const unsigned b = __float_as_uint(f);
    return (b & 0x80000000u) ? ~b : (b | 0x80000000u);
}
__device__ __forceinline__ float order_key_float(unsigned k) {
    return __uint_as_float((k & 0x80000000u) ? (k & 0x7fffffffu) : ~k);
}
// One token, one warp: 512 logits -> the top 10 ids and weights; ids2 and weights2 take copies when given.  The logits
// load through the L2 (another block may have written them).  Each round takes the largest probability and, among
// equal ones, the lowest expert (a lane's own best, then two warp reductions); only the winning lane rescans.
__device__ __forceinline__ void route_token(const float* __restrict__ logits, int32_t* __restrict__ ids,
                                            float* __restrict__ weights, int32_t* ids2, float* weights2, int lane) {
    float values[16];
#pragma unroll
    for (int i = 0; i < 16; ++i) values[i] = __ldcg(logits + lane + i * 32);
    float maximum = -INFINITY;
#pragma unroll
    for (int i = 0; i < 16; ++i) maximum = max(maximum, values[i]);
    maximum = warp_max(maximum);
    float sum = 0.0f;
#pragma unroll
    for (int i = 0; i < 16; ++i) {
        values[i] = expf(values[i] - maximum);
        sum += values[i];
    }
    const float reciprocal = 1.0f / warp_sum(sum);
#pragma unroll
    for (int i = 0; i < 16; ++i) {
        values[i] *= reciprocal;
        if (__isnanf(values[i])) values[i] = -FLT_MAX;
    }
    float lbest = values[0];   // the lane's largest value, the lowest index among equal ones
    int li = 0;
#pragma unroll
    for (int i = 1; i < 16; ++i)
        if (values[i] > lbest) { lbest = values[i]; li = i; }
    float selected = 0.0f, selected_sum = 0.0f;
#pragma unroll 1
    for (int rank = 0; rank < 10; ++rank) {
        const unsigned mine = order_key(lbest);
        const unsigned top = __reduce_max_sync(0xffffffffu, mine);
        const int expert = __reduce_min_sync(0xffffffffu, mine == top ? lane + li * 32 : 0x7fffffff);
        const float best = order_key_float(top);
        if ((expert & 31) == lane) {
#pragma unroll
            for (int i = 0; i < 16; ++i)
                if (i == li) values[i] = -INFINITY;
            lbest = values[0];
            li = 0;
#pragma unroll
            for (int i = 1; i < 16; ++i)
                if (values[i] > lbest) { lbest = values[i]; li = i; }
            ids[rank] = expert;
            if (ids2 != nullptr) ids2[rank] = expert;
            // Deliberately accumulate by WINNING EXPERT lane, not output rank.
            // Multiple selected experts in one lane add in selection order.
            selected_sum += best;
        }
        if (rank == lane) selected = best;
    }
    selected_sum = max(warp_sum(selected_sum), 6.103515625e-5f);
    const float inverse_selected_sum = 1.0f / selected_sum;
    if (lane < 10) {
        const float wv = selected * inverse_selected_sum;
        weights[lane] = wv;
        if (weights2 != nullptr) weights2[lane] = wv;
    }
}
__launch_bounds__(256, 1)
__global__ void route(const float* __restrict__ logits, int32_t* __restrict__ ids,
                      float* __restrict__ weights) {
    // Preserve the pinned 32x8 block geometry; only row zero is active here.  Block b routes token b.
    if (threadIdx.y != 0) return;
    route_token(logits + blockIdx.x * 512, ids + blockIdx.x * 10, weights + blockIdx.x * 10, nullptr, nullptr,
                threadIdx.x);
}

// ================================ plan v0.3 P6: the verify window's router in one kernel ================================
//
// For T tokens: when the input rows have a mapped copy, the first blocks (scheduled first) only copy them, a 16-byte
// store (a full PCIe transaction) a thread, and fence; then a block a row of the logits (bf16_gemv_fp32_mmvf_multi's
// 256-thread block, mmvf_multi_row).  The last block to finish routes the tokens (`route_token`, a warp each); the
// routing warps, which wrote the mapped ids and weights, fence; then the ring's number, and the plan of the main
// GPU's hits (hit_plan_block).  In place of the gemv, the top 10, doorbell_publish and verify_hit_plan.
constexpr int VR_THREADS = 256;   // mmvf_block_size of n_embd a multiple of 512
template <int TT>
__global__ void __launch_bounds__(VR_THREADS) verify_router_kernel(VerifyRouterArgs a, int n_copy) {
    __shared__ bool s_last;
    const int t = (int) threadIdx.x, lane = t & 31, warp = t >> 5, b = (int) blockIdx.x;
    if (b < n_copy) {
        for (int i = b * VR_THREADS + t; i < TT * a.n_embd / 4; i += n_copy * VR_THREADS)
            reinterpret_cast<float4*>(a.x_out)[i] = reinterpret_cast<const float4*>(a.x)[i];
        __threadfence_system();   // this thread's writes before the block counts as done
    } else {
        const int row = b - n_copy;
        float acc[kMmvfMaxRows];
        mmvf_multi_row<VR_THREADS>(a.x, a.w + (size_t) row * a.n_embd, a.n_embd, TT, acc);
        if (t == 0)
            for (int k = 0; k < TT; ++k)
                a.logits[(size_t) k * a.n_expert + row] = a.bias ? acc[k] + a.bias[row] : acc[k];
        __threadfence();
    }
    __syncthreads();
    if (t == 0) s_last = atomicAdd(a.counter, 1u) == gridDim.x - 1;
    __syncthreads();
    if (!s_last) return;
    if (warp < TT) {
        route_token(a.logits + (size_t) warp * a.n_expert, a.ids + warp * 10, a.weights + warp * 10,
                    a.ids_out != nullptr ? a.ids_out + warp * 10 : nullptr,
                    a.w_out != nullptr ? a.w_out + warp * 10 : nullptr, lane);
        __threadfence_system();
    }
    __syncthreads();
    if (t == 0) {   // warp 0 routed and fenced
        *(volatile uint32_t*) a.seq = a.ring;
        *a.counter = 0u;   // for the next launch
    }
    if (a.plan != nullptr)
        hit_plan_block(a.ids, TT * 10, 10, a.res, a.n_expert, a.slot_ptr, a.plan, a.cap, a.ptr_off);
}
__global__ void __launch_bounds__(512) bias_update(const float* __restrict__ logits,
                                                   const float* __restrict__ predicted, int n_tok, float scale,
                                                   float* __restrict__ bias) {
    const int e = (int) threadIdx.x;
    float d = 0.0f;
    for (int t = 0; t < n_tok; ++t) d += logits[t * 512 + e] - predicted[t * 512 + e];
    bias[e] = fmaf(scale, d, bias[e]);
}
bool valid(const void* p, size_t bytes) {
    const auto address = reinterpret_cast<uintptr_t>(p);
    return p && address % 4 == 0 && bytes <= UINTPTR_MAX - address;
}
bool overlap(const void* a, size_t an, const void* b, size_t bn) {
    const auto ap = reinterpret_cast<uintptr_t>(a), bp = reinterpret_cast<uintptr_t>(b);
    return ap < bp + bn && bp < ap + an;
}
}
void native_router_set_enabled(bool value) { enabled.store(value, std::memory_order_relaxed); }
bool native_router_enabled() { return enabled.load(std::memory_order_relaxed); }
void native_router_top10(const float* logits, int32_t* ids, float* weights, void* stream) {
    if (!stream || !valid(logits, 512 * 4) || !valid(ids, 10 * 4) || !valid(weights, 10 * 4)
        || overlap(logits, 512 * 4, ids, 10 * 4) || overlap(logits, 512 * 4, weights, 10 * 4)
        || overlap(ids, 10 * 4, weights, 10 * 4))
        throw std::invalid_argument("native router requires a stream, aligned spans, and disjoint outputs");
    route<<<1, dim3(32, 8), 0, static_cast<cudaStream_t>(stream)>>>(logits, ids, weights);
    const auto error = cudaGetLastError();
    if (error != cudaSuccess) throw std::runtime_error(cudaGetErrorString(error));
}
void verify_router(const VerifyRouterArgs& a, void* stream) {
    if (!stream || a.n_tok < 1 || a.n_tok > 8 || a.n_expert != 512 || a.n_embd <= 0 || a.n_embd % 512 != 0 ||
        !a.x || !a.w || !a.logits || !a.ids || !a.weights || !a.seq || !a.counter || (a.plan && (!a.res || !a.slot_ptr)))
        throw std::invalid_argument("verify_router: 1..8 tokens, 512 experts, n_embd a multiple of 512, and buffers");
    const auto st = static_cast<cudaStream_t>(stream);
    const int n_copy = a.x_out != nullptr ? (a.n_tok * a.n_embd / 4 + VR_THREADS - 1) / VR_THREADS : 0;
    switch (a.n_tok) {
#define STRATA_VR(T) \
    case T: verify_router_kernel<T><<<unsigned(n_copy + a.n_expert), VR_THREADS, 0, st>>>(a, n_copy); break;
        STRATA_VR(1) STRATA_VR(2) STRATA_VR(3) STRATA_VR(4) STRATA_VR(5) STRATA_VR(6) STRATA_VR(7) STRATA_VR(8)
#undef STRATA_VR
    }
    const auto error = cudaGetLastError();
    if (error != cudaSuccess) throw std::runtime_error(cudaGetErrorString(error));
}
void verify_router_bias_update(const float* logits, const float* predicted, int n_tok, float* bias, void* stream) {
    if (!stream || n_tok < 1 || n_tok > 8 || !logits || !predicted || !bias)
        throw std::invalid_argument("verify_router_bias_update: 1..8 tokens and buffers");
    bias_update<<<1, 512, 0, static_cast<cudaStream_t>(stream)>>>(logits, predicted, n_tok, 0.125f / (float) n_tok,
                                                                   bias);
    const auto error = cudaGetLastError();
    if (error != cudaSuccess) throw std::runtime_error(cudaGetErrorString(error));
}
void native_router_top10_multi(const float* logits, int32_t* ids, float* weights, int n_tok, void* stream) {
    const size_t nl = size_t(n_tok) * 512 * 4, no = size_t(n_tok) * 10 * 4;
    if (!stream || n_tok < 1 || !valid(logits, nl) || !valid(ids, no) || !valid(weights, no)
        || overlap(logits, nl, ids, no) || overlap(logits, nl, weights, no) || overlap(ids, no, weights, no))
        throw std::invalid_argument("native router requires a stream, aligned spans, and disjoint outputs");
    route<<<unsigned(n_tok), dim3(32, 8), 0, static_cast<cudaStream_t>(stream)>>>(logits, ids, weights);
    const auto error = cudaGetLastError();
    if (error != cudaSuccess) throw std::runtime_error(cudaGetErrorString(error));
}
}
