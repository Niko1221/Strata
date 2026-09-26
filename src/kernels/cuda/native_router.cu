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
// One token, one warp: 512 logits -> the top 10 ids and weights; ids2 and weights2 take copies when given.  The logits
// load through the L2 (another block may have written them).
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
    float selected = 0.0f, selected_sum = 0.0f;
    for (int rank = 0; rank < 10; ++rank) {
        float best = values[0];
        int expert = lane;
#pragma unroll
        for (int i = 1; i < 16; ++i) {
            if (values[i] > best) { best = values[i]; expert = lane + i * 32; }
        }
#pragma unroll
        for (int mask = 16; mask; mask >>= 1) {
            const float other = __shfl_xor_sync(0xffffffffu, best, mask, 32);
            const int other_id = __shfl_xor_sync(0xffffffffu, expert, mask, 32);
            if (other > best || (other == best && other_id < expert)) { best = other; expert = other_id; }
        }
        if ((expert & 31) == lane) {
            values[expert / 32] = -INFINITY;
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
// For T tokens: a block a row of the logits (bf16_gemv_fp32_mmvf_multi's 256-thread block, mmvf_multi_row), the first
// blocks also copying the input rows to mapped memory, a 16-byte store (a full PCIe transaction) a thread; the last
// block to finish routes the tokens (`route`, a warp each), publishes their ids and weights, writes the ring's number
// and plans the main GPU's hits (hit_plan_block).  In place of the gemv, the top 10, doorbell_publish and
// verify_hit_plan.
constexpr int VR_THREADS = 256;   // mmvf_block_size of n_embd a multiple of 512
template <int TT>
__global__ void __launch_bounds__(VR_THREADS) verify_router_kernel(VerifyRouterArgs a) {
    __shared__ bool s_last;
    const int t = (int) threadIdx.x, lane = t & 31, warp = t >> 5;
    bool mapped = false;
    if (a.x_out != nullptr) {
        const int i = (int) blockIdx.x * VR_THREADS + t;
        if (i < TT * a.n_embd / 4) {
            reinterpret_cast<float4*>(a.x_out)[i] = reinterpret_cast<const float4*>(a.x)[i];
            mapped = true;
        }
    }
    float acc[kMmvfMaxRows];
    mmvf_multi_row<VR_THREADS>(a.x, a.w + (size_t) blockIdx.x * a.n_embd, a.n_embd, TT, acc);
    if (t == 0)
        for (int k = 0; k < TT; ++k) a.logits[(size_t) k * a.n_expert + blockIdx.x] = acc[k];
    if (mapped) __threadfence_system();   // this thread's writes before the block counts as done
    else __threadfence();
    __syncthreads();
    if (t == 0) s_last = atomicAdd(a.counter, 1u) == gridDim.x - 1;
    __syncthreads();
    if (!s_last) return;
    if (warp < TT)
        route_token(a.logits + (size_t) warp * a.n_expert, a.ids + warp * 10, a.weights + warp * 10,
                    a.ids_out != nullptr ? a.ids_out + warp * 10 : nullptr,
                    a.w_out != nullptr ? a.w_out + warp * 10 : nullptr, lane);
    __threadfence_system();
    __syncthreads();
    if (t == 0) {
        __threadfence_system();
        *(volatile uint32_t*) a.seq = a.ring;
        *a.counter = 0u;   // for the next launch
    }
    if (a.plan != nullptr)
        hit_plan_block(a.ids, TT * 10, 10, a.res, a.n_expert, a.slot_ptr, a.plan, a.cap, a.ptr_off);
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
    if (a.x_out != nullptr && a.n_tok * a.n_embd / 4 > a.n_expert * VR_THREADS)
        throw std::invalid_argument("verify_router: the blocks copy the input rows a float4 a thread");
    switch (a.n_tok) {
#define STRATA_VR(T) case T: verify_router_kernel<T><<<unsigned(a.n_expert), VR_THREADS, 0, st>>>(a); break;
        STRATA_VR(1) STRATA_VR(2) STRATA_VR(3) STRATA_VR(4) STRATA_VR(5) STRATA_VR(6) STRATA_VR(7) STRATA_VR(8)
#undef STRATA_VR
    }
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
