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
#include <cuda_runtime.h>
#include <atomic>
#include <cfloat>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
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
// STRATA_ROUTE_FAST (default on; =0 the old kernels): the router's lane exchanges as DPP moves on wave32 RDNA
// (row_xmask for the masks 1-8, v_permlanex16 for 16) instead of ds_bpermute through the LDS crossbar: the same
// values move (every lane of the warp is active at each exchange), ~1 us less a launch.  Elsewhere __shfl_xor_sync.
template <int M>
__device__ __forceinline__ int xor_lane(int v) {
#if defined(__HIPCC__) && (defined(__GFX10__) || defined(__GFX11__) || defined(__GFX12__))
    static_assert(M >= 1 && M <= 16, "a wave32 xor exchange");
    if constexpr (M == 16) return __builtin_amdgcn_permlanex16(v, v, 0x76543210u, 0xfedcba98u, false, false);
    else return __builtin_amdgcn_update_dpp(v, v, 0x160 + M, 0xf, 0xf, false);
#else
    return __shfl_xor_sync(0xffffffffu, v, M, 32);
#endif
}
template <bool F, int M>
__device__ __forceinline__ float xs(float v) {
    if constexpr (F) return __int_as_float(xor_lane<M>(__float_as_int(v)));
    else return __shfl_xor_sync(0xffffffffu, v, M, 32);
}
template <bool F, int M>
__device__ __forceinline__ int xs(int v) {
    if constexpr (F) return xor_lane<M>(v);
    else return __shfl_xor_sync(0xffffffffu, v, M, 32);
}
// warp_sum / warp_max / the top-1 exchange with the masks 16, 8, 4, 2, 1 in that order (the same trees)
template <bool F>
__device__ __forceinline__ float warp_sum_f(float value) {
    value += xs<F, 16>(value); value += xs<F, 8>(value); value += xs<F, 4>(value); value += xs<F, 2>(value);
    value += xs<F, 1>(value);
    return value;
}
template <bool F>
__device__ __forceinline__ float warp_max_f(float value) {
    value = fmaxf(value, xs<F, 16>(value)); value = fmaxf(value, xs<F, 8>(value)); value = fmaxf(value, xs<F, 4>(value));
    value = fmaxf(value, xs<F, 2>(value)); value = fmaxf(value, xs<F, 1>(value));
    return value;
}
template <bool F, int M>
__device__ __forceinline__ void top1_step(float& best, int& expert) {
    const float other = xs<F, M>(best);
    const int other_id = xs<F, M>(expert);
    if (other > best || (other == best && other_id < expert)) { best = other; expert = other_id; }
}
template <bool F>
__device__ __forceinline__ void top1_exchange(float& best, int& expert) {
    top1_step<F, 16>(best, expert); top1_step<F, 8>(best, expert); top1_step<F, 4>(best, expert);
    top1_step<F, 2>(best, expert); top1_step<F, 1>(best, expert);
}
template <bool F = false>
__launch_bounds__(256, 1)
__global__ void route(const float* __restrict__ logits, int32_t* __restrict__ ids,
                      float* __restrict__ weights) {
    // Preserve the pinned 32x8 block geometry; only row zero is active here.
    // blockIdx.x = the token (a multi-token launch; 0 for the single one)
    logits += (size_t) blockIdx.x * 512; ids += (size_t) blockIdx.x * 10; weights += (size_t) blockIdx.x * 10;
    if (threadIdx.y != 0) return;
    const int lane = threadIdx.x;
    float values[16];
#pragma unroll
    for (int i = 0; i < 16; ++i) values[i] = logits[lane + i * 32];
    __syncthreads();
    float maximum = -INFINITY;
#pragma unroll
    for (int i = 0; i < 16; ++i) maximum = max(maximum, values[i]);
    maximum = warp_max_f<F>(maximum);
    float sum = 0.0f;
#pragma unroll
    for (int i = 0; i < 16; ++i) {
        values[i] = expf(values[i] - maximum);
        sum += values[i];
    }
    const float reciprocal = 1.0f / warp_sum_f<F>(sum);
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
        top1_exchange<F>(best, expert);
        if ((expert & 31) == lane) {
            values[expert / 32] = -INFINITY;
            ids[rank] = expert;
            // Deliberately accumulate by WINNING EXPERT lane, not output rank.
            // Multiple selected experts in one lane add in selection order.
            selected_sum += best;
        }
        if (rank == lane) selected = best;
    }
    selected_sum = max(warp_sum_f<F>(selected_sum), 6.103515625e-5f);
    const float inverse_selected_sum = 1.0f / selected_sum;
    if (lane < 10) weights[lane] = selected * inverse_selected_sum;
}
template <bool F = false>
__launch_bounds__(256, 1)
__global__ void route_multi(const float* __restrict__ logits, int32_t* __restrict__ ids,
                            float* __restrict__ weights, int n_tok) {
    const int tk = (int) blockIdx.x * 8 + (int) threadIdx.y;
    if (tk >= n_tok) return;
    logits += (size_t) tk * 512; ids += (size_t) tk * 10; weights += (size_t) tk * 10;
    const int lane = threadIdx.x;
    float values[16];
#pragma unroll
    for (int i = 0; i < 16; ++i) values[i] = logits[lane + i * 32];
    __syncwarp();
    float maximum = -INFINITY;
#pragma unroll
    for (int i = 0; i < 16; ++i) maximum = max(maximum, values[i]);
    maximum = warp_max_f<F>(maximum);
    float sum = 0.0f;
#pragma unroll
    for (int i = 0; i < 16; ++i) {
        values[i] = expf(values[i] - maximum);
        sum += values[i];
    }
    const float reciprocal = 1.0f / warp_sum_f<F>(sum);
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
        top1_exchange<F>(best, expert);
        if ((expert & 31) == lane) {
            values[expert / 32] = -INFINITY;
            ids[rank] = expert;
            selected_sum += best;
        }
        if (rank == lane) selected = best;
    }
    selected_sum = max(warp_sum_f<F>(selected_sum), 6.103515625e-5f);
    const float inverse_selected_sum = 1.0f / selected_sum;
    if (lane < 10) weights[lane] = selected * inverse_selected_sum;
}
// STRATA_ROUTE_PUB: route_multi and the verify window's doorbell_publish_res in one launch (n_tok <= 8, one block).
// The routing is route_multi's body verbatim (a warp per token); after a block barrier the block publishes as
// doorbell_publish_res_kernel does: the ids, the MoE input when an expert is not resident, then the sequence number.
__launch_bounds__(256, 1)
__global__ void route_multi_publish_res(const float* __restrict__ logits_in, int32_t* __restrict__ ids_in,
                                        float* __restrict__ weights_in, int n_tok, const float* __restrict__ x,
                                        const int32_t* __restrict__ d_res, int n_expert, int n_x, float* x_out,
                                        int32_t* ids_out, uint32_t* seq) {
    const int tk = (int) blockIdx.x * 8 + (int) threadIdx.y;
    if (tk < n_tok) {
    const float* __restrict__ logits = logits_in + (size_t) tk * 512;
    int32_t* __restrict__ ids = ids_in + (size_t) tk * 10;
    float* __restrict__ weights = weights_in + (size_t) tk * 10;
    const int lane = threadIdx.x;
    float values[16];
#pragma unroll
    for (int i = 0; i < 16; ++i) values[i] = logits[lane + i * 32];
    __syncwarp();
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
            selected_sum += best;
        }
        if (rank == lane) selected = best;
    }
    selected_sum = max(warp_sum(selected_sum), 6.103515625e-5f);
    const float inverse_selected_sum = 1.0f / selected_sum;
    if (lane < 10) weights[lane] = selected * inverse_selected_sum;
    }
    // ---- the publish (doorbell_publish_res_kernel with 256 threads)
    __syncthreads();
    const int tid = (int) threadIdx.y * 32 + (int) threadIdx.x;
    const int k = n_tok * 10;
    int any_miss = 0;
    for (int i = tid; i < k; i += 256) {
        const int32_t id = ids_in[i];
        ids_out[i] = id;
        if (d_res == nullptr || id < 0 || id >= n_expert || d_res[id] < 0) any_miss = 1;
    }
    if (__syncthreads_or(any_miss)) {
        for (int i = tid; i < n_x; i += 256) x_out[i] = x[i];
        __threadfence_system();
    } else if (tid < k) {
        __threadfence_system();
    }
    __syncthreads();
    if (tid == 0) {
        *(volatile uint32_t*) seq = *(volatile uint32_t*) seq + 1u;
        __threadfence_system();
    }
}
// STRATA_ROUTE_FAST: route_multi_publish_res with the DPP exchanges and a faster publish, the same words.  The MoE
// input rows go to the host in 16-byte stores: for 5-8 rows each token's warp copies its row right after its routing,
// whatever the residency table says (a superset of the old publish, which copied every row when any routed expert
// was not resident); for 1-4 rows the whole block copies them after a barrier when any expert is not resident, as
// before.  (Copying only the rows of the tokens with a non-resident expert changed the drafts after a prompt: the
// host plans from its own table, and while the cache refills it can count an expert out that the device table still
// holds.)  The step number is stored (`seq_value`, the host's next one) rather than read over PCIe and incremented.
// x_out null: the router GEMV launch before this one already copied every row (bf16_gemv_fp32_mmvf_multi's x_pub).
template <bool F>
__launch_bounds__(256, 1)
__global__ void route_multi_publish_fast(const float* __restrict__ logits_in, int32_t* __restrict__ ids_in,
                                         float* __restrict__ weights_in, int n_tok, const float* __restrict__ x,
                                         const int32_t* __restrict__ d_res, int n_expert, int n_embd, float* x_out,
                                         int32_t* ids_out, uint32_t* seq, uint32_t seq_value) {
    __shared__ int s_miss[8];
    const int tk = (int) threadIdx.y;
    const int lane = threadIdx.x;
    const bool by_warp = n_tok > 4;
    bool wrote = false;
    if (tk < n_tok) {
    const float* __restrict__ logits = logits_in + (size_t) tk * 512;
    int32_t* __restrict__ ids = ids_in + (size_t) tk * 10;
    float* __restrict__ weights = weights_in + (size_t) tk * 10;
    float values[16];
#pragma unroll
    for (int i = 0; i < 16; ++i) values[i] = logits[lane + i * 32];
    __syncwarp();
    float maximum = -INFINITY;
#pragma unroll
    for (int i = 0; i < 16; ++i) maximum = max(maximum, values[i]);
    maximum = warp_max_f<F>(maximum);
    float sum = 0.0f;
#pragma unroll
    for (int i = 0; i < 16; ++i) {
        values[i] = expf(values[i] - maximum);
        sum += values[i];
    }
    const float reciprocal = 1.0f / warp_sum_f<F>(sum);
#pragma unroll
    for (int i = 0; i < 16; ++i) {
        values[i] *= reciprocal;
        if (__isnanf(values[i])) values[i] = -FLT_MAX;
    }
    float selected = 0.0f, selected_sum = 0.0f;
    int my_id = 0;
    for (int rank = 0; rank < 10; ++rank) {
        float best = values[0];
        int expert = lane;
#pragma unroll
        for (int i = 1; i < 16; ++i) {
            if (values[i] > best) { best = values[i]; expert = lane + i * 32; }
        }
        top1_exchange<F>(best, expert);
        if ((expert & 31) == lane) {
            values[expert / 32] = -INFINITY;
            ids[rank] = expert;
            selected_sum += best;
        }
        if (rank == lane) { selected = best; my_id = expert; }
    }
    selected_sum = max(warp_sum_f<F>(selected_sum), 6.103515625e-5f);
    const float inverse_selected_sum = 1.0f / selected_sum;
    if (lane < 10) weights[lane] = selected * inverse_selected_sum;
    // ---- this token's publish
    bool miss = false;
    if (lane < 10) {
        ids_out[tk * 10 + lane] = my_id;
        miss = d_res == nullptr || my_id < 0 || my_id >= n_expert || d_res[my_id] < 0;
        wrote = true;
    }
    const bool tmiss = __ballot_sync(0xffffffffu, miss) != 0u;
    if (lane == 0) s_miss[tk] = tmiss ? 1 : 0;
    if (by_warp && x_out != nullptr) {
        const float4* __restrict__ src = reinterpret_cast<const float4*>(x + (size_t) tk * n_embd);
        float4* dst = reinterpret_cast<float4*>(x_out + (size_t) tk * n_embd);
        for (int i = lane; i < n_embd / 4; i += 32) dst[i] = src[i];
        wrote = true;
    }
    }
    if (!by_warp && x_out != nullptr) {   // 1-4 rows: every row when any routed expert is not resident (the old rule)
        __syncthreads();
        int any = 0;
        for (int t = 0; t < n_tok; ++t) any |= s_miss[t];
        if (any) {
            const int tid = tk * 32 + lane, r4 = n_embd / 4;
            for (int i = tid; i < n_tok * r4; i += 256)
                reinterpret_cast<float4*>(x_out)[i] = reinterpret_cast<const float4*>(x)[i];
            wrote = true;
        }
    }
    if (wrote) __threadfence_system();
    __syncthreads();
    if (tk == 0 && lane == 0) {
        if (seq_value != 0u) *(volatile uint32_t*) seq = seq_value;
        else *(volatile uint32_t*) seq = *(volatile uint32_t*) seq + 1u;
        __threadfence_system();
    }
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
bool route_fast() {
    static const bool on = [] { const char* v = std::getenv("STRATA_ROUTE_FAST"); return v == nullptr || std::atoi(v) != 0; }();
    return on;
}
bool native_router_fast() { return route_fast(); }
void native_router_set_enabled(bool value) { enabled.store(value, std::memory_order_relaxed); }
bool native_router_enabled() { return enabled.load(std::memory_order_relaxed); }
void native_router_top10(const float* logits, int32_t* ids, float* weights, void* stream) {
    if (!stream || !valid(logits, 512 * 4) || !valid(ids, 10 * 4) || !valid(weights, 10 * 4)
        || overlap(logits, 512 * 4, ids, 10 * 4) || overlap(logits, 512 * 4, weights, 10 * 4)
        || overlap(ids, 10 * 4, weights, 10 * 4))
        throw std::invalid_argument("native router requires a stream, aligned spans, and disjoint outputs");
    if (route_fast()) route<true><<<1, dim3(32, 8), 0, static_cast<cudaStream_t>(stream)>>>(logits, ids, weights);
    else route<false><<<1, dim3(32, 8), 0, static_cast<cudaStream_t>(stream)>>>(logits, ids, weights);
    const auto error = cudaGetLastError();
    if (error != cudaSuccess) throw std::runtime_error(cudaGetErrorString(error));
}
void native_router_top10_multi(const float* logits, int32_t* ids, float* weights, int n_tok, void* stream) {
    if (!stream || n_tok < 1 || !valid(logits, (size_t) n_tok * 512 * 4) || !valid(ids, (size_t) n_tok * 10 * 4) ||
        !valid(weights, (size_t) n_tok * 10 * 4))
        throw std::invalid_argument("native router (multi) requires a stream and aligned [n,512]/[n,10] buffers");
    if (route_fast())
        route_multi<true><<<(unsigned) ((n_tok + 7) / 8), dim3(32, 8), 0, static_cast<cudaStream_t>(stream)>>>(logits, ids, weights, n_tok);
    else
        route_multi<false><<<(unsigned) ((n_tok + 7) / 8), dim3(32, 8), 0, static_cast<cudaStream_t>(stream)>>>(logits, ids, weights, n_tok);
    const auto error = cudaGetLastError();
    if (error != cudaSuccess) throw std::runtime_error(cudaGetErrorString(error));
}

void native_router_top10_multi_publish_res(const float* logits, int32_t* ids, float* weights, int n_tok, const float* x,
                                           const int32_t* d_res, int n_expert, int64_t n_x, float* x_out,
                                           int32_t* ids_out, uint32_t* d_seq, void* stream, uint32_t seq_value) {
    // x null: the rows are on the host already (the router GEMV copied them); the fast kernel only
    const bool x_done = x == nullptr;
    if (!stream || n_tok < 1 || n_tok > 8 || !valid(logits, (size_t) n_tok * 512 * 4) || !valid(ids, (size_t) n_tok * 10 * 4) ||
        !valid(weights, (size_t) n_tok * 10 * 4) || (!x_done && !x_out) || !ids_out || !d_seq)
        throw std::invalid_argument("native router (multi + publish) requires 1-8 rows, a stream and the publish buffers");
    const int64_t n_embd = n_x / n_tok;
    const bool fast = route_fast() && seq_value != 0u &&
                      (x_done || (n_embd * n_tok == n_x && n_embd % 4 == 0 && reinterpret_cast<uintptr_t>(x) % 16 == 0 &&
                                  reinterpret_cast<uintptr_t>(x_out) % 16 == 0));
    if (x_done && !fast) throw std::invalid_argument("native router (multi + publish): x null needs STRATA_ROUTE_FAST");
    if (fast)
        route_multi_publish_fast<true><<<1, dim3(32, 8), 0, static_cast<cudaStream_t>(stream)>>>(
            logits, ids, weights, n_tok, x, d_res, n_expert, (int) n_embd, x_done ? nullptr : x_out, ids_out, d_seq,
            seq_value);
    else
        route_multi_publish_res<<<1, dim3(32, 8), 0, static_cast<cudaStream_t>(stream)>>>(logits, ids, weights, n_tok, x, d_res,
                                                                                         n_expert, (int) n_x, x_out, ids_out, d_seq);
    const auto error = cudaGetLastError();
    if (error != cudaSuccess) throw std::runtime_error(cudaGetErrorString(error));
}

// STRATA_ROUTE_RESIDENT (EXPERIMENTAL, opt-in, changes the output): residency-biased routing for 512 x 10.
// After the top-10: for the ranks lo..hi a non-resident expert is replaced by the best resident expert not already
// picked when its logit is within `margin` of the replaced one; the weights are then the softmax of the selected
// logits (the router's renormalisation).  Tokens without a swap keep the router's exact bits.
// stats (device, 4 x uint64): [0] tail entries seen, [1] swaps, [2] non-resident entries before, [3] after.
//
// One warp per token.  This used to be one THREAD per token scanning all 512 experts (4 ranks x 512 x 12 scalar
// ops on up to 6 threads): 45 us per layer, 2.2 ms of a 19.4 ms window, which is exactly what the option saved in
// PCIe experts (~2.4 ms) - the swap cost as much as the misses it removed.  The ranks are still visited in order
// (a swap changes the set already picked), but the scan is coalesced (lane j reads f = j, j + 32, ...) and reduced
// within the warp.  The bits are unchanged: a strictly greater logit wins, the smallest expert index breaks ties,
// and the softmax is the same expression.
__global__ void route_resident_k(const float* __restrict__ logits, int32_t* __restrict__ ids, float* __restrict__ weights,
                                 const int32_t* __restrict__ res, int n_tok, float margin, int lo, int hi,
                                 unsigned long long* __restrict__ stats) {
    const int t = blockIdx.x;
    if (t >= n_tok) return;
    const int lane = (int) threadIdx.x;
    const float* l = logits + (size_t) t * 512;
    int32_t* id = ids + (size_t) t * 10;
    int32_t my[10];
    for (int r = 0; r < 10; ++r) my[r] = id[r];
    int swaps = 0, tail = 0, before = 0;
    if (lane == 0)
        for (int r = 0; r < 10; ++r) before += res[my[r]] < 0;
    for (int r = lo; r <= hi && r < 10; ++r) {
        const int e = my[r];
        if (res[e] >= 0) continue;                      // already resident: uniform across the warp
        if (lane == 0) ++tail;
        float bv = -INFINITY;
        int bi = -1;
        for (int f = lane; f < 512; f += 32) {
            if (res[f] < 0) continue;
            bool used = false;
            for (int q = 0; q < 10; ++q) used |= my[q] == f;
            if (used) continue;
            if (l[f] > bv) { bv = l[f]; bi = f; }
        }
        #pragma unroll
        for (int off = 16; off > 0; off >>= 1) {
            const float ov = __shfl_down_sync(0xffffffffu, bv, off);
            const int oi = __shfl_down_sync(0xffffffffu, bi, off);
            if (ov > bv || (ov == bv && oi >= 0 && (bi < 0 || oi < bi))) { bv = ov; bi = oi; }
        }
        if (lane == 0 && bi >= 0 && l[e] - bv <= margin) { my[r] = bi; ++swaps; }
        my[r] = __shfl_sync(0xffffffffu, my[r], 0);     // all lanes keep the same picked set
    }
    int after = 0;
    if (lane == 0)
        for (int r = 0; r < 10; ++r) after += res[my[r]] < 0;
    const int sw = __shfl_sync(0xffffffffu, swaps, 0);
    if (sw) {
        float m = -INFINITY;
        for (int r = 0; r < 10; ++r) m = fmaxf(m, l[my[r]]);
        float ex[10], sum = 0.0f;
        for (int r = 0; r < 10; ++r) { ex[r] = expf(l[my[r]] - m); sum += ex[r]; }
        for (int r = 0; r < 10; ++r) weights[(size_t) t * 10 + r] = ex[r] / sum;
    }
    if (lane == 0) {
        for (int r = 0; r < 10; ++r) id[r] = my[r];
        if (stats) {
            atomicAdd(stats + 0, (unsigned long long) tail);
            atomicAdd(stats + 1, (unsigned long long) sw);
            atomicAdd(stats + 2, (unsigned long long) before);
            atomicAdd(stats + 3, (unsigned long long) after);
        }
    }
}
void native_route_resident(const float* logits, int32_t* ids, float* weights, const int32_t* res_layer, int n_tok, float margin,
                           int rank_lo, int rank_hi, unsigned long long* stats, void* stream) {
    if (!stream || n_tok < 1 || !res_layer) throw std::invalid_argument("native_route_resident: bad arguments");
    route_resident_k<<<(unsigned) n_tok, 32, 0, static_cast<cudaStream_t>(stream)>>>(
        logits, ids, weights, res_layer, n_tok, margin, rank_lo, rank_hi, stats);
    const auto error = cudaGetLastError();
    if (error != cudaSuccess) throw std::runtime_error(cudaGetErrorString(error));
}
}
