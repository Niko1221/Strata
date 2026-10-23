// Arithmetic adapted from the MIT-licensed pinned ggml CUDA
// moe-weighted-reduction.cu at 3cf03257f219afbe7334045ff7c6a06ac68c627d.
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
#include "strata/kernels/native_moe.hpp"
#include "strata/kernels/dp4a.hpp"   // strata_spin_pause
#include <cuda_runtime.h>
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <stdexcept>

namespace strata::kernels {
namespace {
std::atomic<bool> enabled{false};
// GATED (S26 STRATA_LFUSE): `shared` is the shared expert's unscaled output and sg its raw gate logit per token;
// the row is scaled here as shared_expert_multi did (native_scalar_sigmoid_multi_kernel's expression, then one
// rounded multiply = scale_rows_kernel's `out *= g`), then added as before
template<bool ZADD = false, bool GATED = false>
__global__ void combine(const float* __restrict__ parts, const float* __restrict__ weights,
                        const float* __restrict__ shared, float* __restrict__ output,
                        int64_t n_embd, int k, const float* __restrict__ sg = nullptr) {
    // blockIdx.y = the token of a multi-token launch (0 for the single one)
    const int64_t tk = blockIdx.y;
    parts += tk * k * n_embd; weights += tk * k; if (shared) shared += tk * n_embd; output += tk * n_embd;
    const int64_t col = int64_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (col >= n_embd) return;
    // ZADD: a part is 0.0f + hit (the zeroed row plus moe_hit_add of the verify window), rounded as that add
    float p0 = parts[col];
    if (ZADD) p0 = 0.0f + p0;
    float sum = p0 * weights[0];
    for (int expert = 1; expert < k; ++expert) {
        float p = parts[int64_t(expert) * n_embd + col];
        if (ZADD) p = 0.0f + p;
        sum += p * weights[expert];
    }
    if constexpr (GATED) {
        // contraction off in this block only: the product is rounded on its own (scale_rows' store), then added
#pragma clang fp contract(off)
        const float g = __fdividef(1.0f, 1.0f + __expf(-sg[tk]));
        const float sh = shared[col] * g;
        sum += sh;
    } else {
        if (shared) sum += shared[col];
    }
    output[col] = sum;
}
__global__ void combine_k10_vec4(const float4* __restrict__ parts4, const float* __restrict__ weights,
                                 const float4* __restrict__ shared4, float4* __restrict__ output4,
                                 int64_t n4) {
    const int64_t tk = blockIdx.y;
    parts4 += tk * 10 * n4;
    weights += tk * 10;
    shared4 += tk * n4;
    output4 += tk * n4;
    const int64_t c4 = int64_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (c4 >= n4) return;
    const float w0 = __ldg(weights + 0);
    const float4 p0 = parts4[c4];
    float4 sum = make_float4(p0.x * w0, p0.y * w0, p0.z * w0, p0.w * w0);
#pragma unroll
    for (int expert = 1; expert < 10; ++expert) {
        const float w = __ldg(weights + expert);
        const float4 p = parts4[int64_t(expert) * n4 + c4];
        // the documented contract spelled out: the first product rounded, then one FMA per expert in order. Left to the
        // compiler's contraction, a build may fuse a different product (gfx1151: the float4 kernel then differed from the
        // scalar one, native_multi_parity)
        sum.x = fmaf(p.x, w, sum.x);
        sum.y = fmaf(p.y, w, sum.y);
        sum.z = fmaf(p.z, w, sum.z);
        sum.w = fmaf(p.w, w, sum.w);
    }
    const float4 sh = shared4[c4];
    sum.x += sh.x;
    sum.y += sh.y;
    sum.z += sh.z;
    sum.w += sh.w;
    output4[c4] = sum;
}
__global__ void combine_k10_vec4_hits(const float4* __restrict__ parts4, const float4* __restrict__ hits4,
                                      const int32_t* __restrict__ dst, const int32_t* __restrict__ count,
                                      const float* __restrict__ weights, const float4* __restrict__ shared4,
                                      float4* __restrict__ output4, int64_t n4) {
    const int64_t tk = blockIdx.y;
    __shared__ int hit_mask;
    if (threadIdx.x == 0) hit_mask = 0;
    __syncthreads();
    const int nh = *count;
    for (int h = threadIdx.x; h < nh; h += blockDim.x) {
        const int r = dst[h];
        if (r / 10 == (int) tk) atomicOr(&hit_mask, 1 << (r % 10));
    }
    __syncthreads();
    const int mask = hit_mask;
    parts4 += tk * 10 * n4;
    hits4 += tk * 10 * n4;
    weights += tk * 10;
    shared4 += tk * n4;
    output4 += tk * n4;
    const int64_t c4 = int64_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (c4 >= n4) return;
    const float w0 = __ldg(weights + 0);
    float4 p0 = parts4[c4];
    if (mask & 1) { const float4 hh = hits4[c4]; p0.x += hh.x; p0.y += hh.y; p0.z += hh.z; p0.w += hh.w; }
    float4 sum = make_float4(p0.x * w0, p0.y * w0, p0.z * w0, p0.w * w0);
#pragma unroll
    for (int expert = 1; expert < 10; ++expert) {
        const float w = __ldg(weights + expert);
        float4 p = parts4[int64_t(expert) * n4 + c4];
        if ((mask >> expert) & 1) { const float4 hh = hits4[int64_t(expert) * n4 + c4]; p.x += hh.x; p.y += hh.y; p.z += hh.z; p.w += hh.w; }
        sum.x = fmaf(p.x, w, sum.x);
        sum.y = fmaf(p.y, w, sum.y);
        sum.z = fmaf(p.z, w, sum.z);
        sum.w = fmaf(p.w, w, sum.w);
    }
    const float4 sh = shared4[c4];
    sum.x += sh.x;
    sum.y += sh.y;
    sum.z += sh.z;
    sum.w += sh.w;
    output4[c4] = sum;
}
// STRATA_FUSE_COMBINE: the CPU share's wait and copy folded into combine_k10_vec4_hits.  The parts are read where the
// copy put them from: the mapped rows `src4` (the pool's), after thread 0 of every block has seen *flag >= value.
//   !DP (copy_rows_from_mapped_after): a hit row's part is +0.0 (the copy wrote zeros there), the others the mapped row;
//   DP (wait_flag_ge_or + copy_or_zero_from_mapped): *skip == value - every part +0.0, nothing waited for or read -
//       else every part the mapped row.
// Then a hit's part gets its hit added and the combine is combine_k10_vec4_hits' (the same ops in the same order).
// GATED (STRATA_FUSE_SHGATE): `shared` is unscaled and sg[t] the raw gate logit: the row is scaled as
// sigmoid_scale_rows_vec4_kernel did (the same gate expression, one rounded multiply) before it is added.
template<bool DP, bool GATED>
__global__ void combine_k10_vec4_hits_mapped(const float4* src4, const float4* __restrict__ hits4,
                                             const int32_t* __restrict__ dst, const int32_t* __restrict__ count,
                                             const float* __restrict__ weights, const float4* __restrict__ shared4,
                                             float4* __restrict__ output4, int64_t n4, const volatile uint32_t* flag,
                                             uint32_t value, const uint32_t* skip, const float* __restrict__ sg) {
#pragma clang fp contract(off)
    const int64_t tk = blockIdx.y;
    __shared__ int hit_mask, zero;
    if (threadIdx.x == 0) {
        hit_mask = 0;
        const bool z = DP && *(const volatile uint32_t*) skip == value;
        zero = z;
        if (!z) {
            while (*flag < value) strata_spin_pause();
            __threadfence_system();
        }
    }
    __syncthreads();
    const int nh = *count;
    for (int h = threadIdx.x; h < nh; h += blockDim.x) {
        const int r = dst[h];
        if (r / 10 == (int) tk) atomicOr(&hit_mask, 1 << (r % 10));
    }
    __syncthreads();
    const int mask = hit_mask;
    const bool z = zero;
    src4 += tk * 10 * n4;
    hits4 += tk * 10 * n4;
    weights += tk * 10;
    shared4 += tk * n4;
    output4 += tk * n4;
    const int64_t c4 = int64_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (c4 >= n4) return;
    float4 p[10];
#pragma unroll
    for (int e = 0; e < 10; ++e) {   // every mapped load first, so they are in flight together
        const bool hit = (mask >> e) & 1;
        if (DP ? z : hit) p[e] = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
        else p[e] = src4[int64_t(e) * n4 + c4];
    }
    const float w0 = __ldg(weights + 0);
    if (mask & 1) { const float4 hh = hits4[c4]; p[0].x += hh.x; p[0].y += hh.y; p[0].z += hh.z; p[0].w += hh.w; }
    float4 sum = make_float4(p[0].x * w0, p[0].y * w0, p[0].z * w0, p[0].w * w0);
#pragma unroll
    for (int expert = 1; expert < 10; ++expert) {
        const float w = __ldg(weights + expert);
        float4 q = p[expert];
        if ((mask >> expert) & 1) { const float4 hh = hits4[int64_t(expert) * n4 + c4]; q.x += hh.x; q.y += hh.y; q.z += hh.z; q.w += hh.w; }
        sum.x = fmaf(q.x, w, sum.x);
        sum.y = fmaf(q.y, w, sum.y);
        sum.z = fmaf(q.z, w, sum.z);
        sum.w = fmaf(q.w, w, sum.w);
    }
    float4 sh = shared4[c4];
    if constexpr (GATED) {
        const float gt = __fdividef(1.0f, 1.0f + __expf(-__ldg(sg + tk)));
        sh.x *= gt;
        sh.y *= gt;
        sh.z *= gt;
        sh.w *= gt;
    }
    sum.x += sh.x;
    sum.y += sh.y;
    sum.z += sh.z;
    sum.w += sh.w;
    output4[c4] = sum;
}
bool valid_span(const void* p, size_t bytes) {
    const auto address = reinterpret_cast<uintptr_t>(p);
    return p && address % alignof(float) == 0 && bytes <= UINTPTR_MAX - address;
}
bool overlap(const void* a, size_t an, const void* b, size_t bn) {
    const auto ap = reinterpret_cast<uintptr_t>(a), bp = reinterpret_cast<uintptr_t>(b);
    return ap < bp + bn && bp < ap + an;
}
}
void native_moe_combine_set_enabled(bool value) { enabled.store(value, std::memory_order_relaxed); }
bool native_moe_combine_enabled() { return enabled.load(std::memory_order_relaxed); }
void native_moe_combine(const float* parts, const float* weights, const float* shared,
                        float* output, int64_t n_embd, int64_t k, void* stream) {
    if (!stream || n_embd <= 0 || n_embd > std::numeric_limits<int>::max() || k < 1 || k > 15)
        throw std::invalid_argument("native MoE combine requires a stream, positive width and 1..15 experts");
    const size_t row_bytes = size_t(n_embd) * sizeof(float);
    const size_t part_bytes = row_bytes * size_t(k), weight_bytes = size_t(k) * sizeof(float);
    if (!valid_span(parts, part_bytes) || !valid_span(weights, weight_bytes) || !valid_span(output, row_bytes)
            || (shared && !valid_span(shared, row_bytes))
            || overlap(output, row_bytes, parts, part_bytes)
            || overlap(output, row_bytes, weights, weight_bytes)
            || (shared && overlap(output, row_bytes, shared, row_bytes)))
        throw std::invalid_argument("native MoE combine requires aligned spans and disjoint output");
    if (k == 10 && shared != nullptr && (n_embd & 3) == 0 &&
        (((uintptr_t) parts | (uintptr_t) shared | (uintptr_t) output) & 15u) == 0) {
        const int64_t n4 = n_embd >> 2;
        combine_k10_vec4<<<unsigned((n4 + 127) / 128), 128, 0, static_cast<cudaStream_t>(stream)>>>(
            reinterpret_cast<const float4*>(parts), weights, reinterpret_cast<const float4*>(shared),
            reinterpret_cast<float4*>(output), n4);
    } else {
        combine<<<unsigned((n_embd + 255) / 256), 256, 0, static_cast<cudaStream_t>(stream)>>>(
            parts, weights, shared, output, n_embd, int(k));
    }
    const auto error = cudaGetLastError();
    if (error != cudaSuccess) throw std::runtime_error(cudaGetErrorString(error));
}
bool native_moe_combine_hits_supported(const float* parts, const float* shared, const float* output, int64_t n_embd, int64_t k, int n_tok) {
    return k == 10 && shared != nullptr && n_tok >= 1 && n_tok <= 8 && n_embd > 0 && (n_embd & 3) == 0 &&
           (((uintptr_t) parts | (uintptr_t) shared | (uintptr_t) output) & 15u) == 0;
}
void native_moe_combine_multi_hits_added(const float* parts, const float* hits, const int32_t* dst, const int32_t* count,
                                         const float* weights, const float* shared, float* output, int64_t n_embd, int64_t k,
                                         int n_tok, void* stream) {
    if (!stream || !native_moe_combine_hits_supported(parts, shared, output, n_embd, k, n_tok) || ((uintptr_t) hits & 15u) != 0)
        throw std::invalid_argument("native MoE combine (hits added) requires k = 10, aligned spans, 1..8 tokens");
    const int64_t n4 = n_embd >> 2;
    combine_k10_vec4_hits<<<dim3(unsigned((n4 + 127) / 128), (unsigned) n_tok), 128, 0, static_cast<cudaStream_t>(stream)>>>(
        reinterpret_cast<const float4*>(parts), reinterpret_cast<const float4*>(hits), dst, count, weights,
        reinterpret_cast<const float4*>(shared), reinterpret_cast<float4*>(output), n4);
    const auto error = cudaGetLastError();
    if (error != cudaSuccess) throw std::runtime_error(cudaGetErrorString(error));
}
void native_moe_combine_multi_hits_mapped(const float* mapped_parts, const float* hits, const int32_t* dst,
                                          const int32_t* count, const float* weights, const float* shared, float* output,
                                          int64_t n_embd, int64_t k, int n_tok, const uint32_t* flag, uint32_t value,
                                          const uint32_t* skip, const float* shared_gate, void* stream) {
    if (!stream || !flag || !native_moe_combine_hits_supported(mapped_parts, shared, output, n_embd, k, n_tok) ||
        ((uintptr_t) hits & 15u) != 0)
        throw std::invalid_argument("native MoE combine (mapped parts) requires k = 10, aligned spans, 1..8 tokens");
    const int64_t n4 = n_embd >> 2;
    const dim3 grid(unsigned((n4 + 127) / 128), (unsigned) n_tok);
    const auto st = static_cast<cudaStream_t>(stream);
#define STRATA_CMB(DP, G) combine_k10_vec4_hits_mapped<DP, G><<<grid, 128, 0, st>>>( \
        reinterpret_cast<const float4*>(mapped_parts), reinterpret_cast<const float4*>(hits), dst, count, weights, \
        reinterpret_cast<const float4*>(shared), reinterpret_cast<float4*>(output), n4, (const volatile uint32_t*) flag, value, skip, \
        shared_gate)
    if (skip != nullptr) { if (shared_gate != nullptr) STRATA_CMB(true, true); else STRATA_CMB(true, false); }
    else { if (shared_gate != nullptr) STRATA_CMB(false, true); else STRATA_CMB(false, false); }
#undef STRATA_CMB
    const auto error = cudaGetLastError();
    if (error != cudaSuccess) throw std::runtime_error(cudaGetErrorString(error));
}
void native_moe_combine_multi(const float* parts, const float* weights, const float* shared, float* output,
                              int64_t n_embd, int64_t k, int n_tok, void* stream) {
    if (!stream || n_embd <= 0 || k < 1 || k > 15 || n_tok < 1)
        throw std::invalid_argument("native MoE combine (multi) requires a stream, width, 1..15 experts, tokens");
    if (k == 10 && shared != nullptr && (n_embd & 3) == 0 &&
        (((uintptr_t) parts | (uintptr_t) shared | (uintptr_t) output) & 15u) == 0) {
        const int64_t n4 = n_embd >> 2;
        combine_k10_vec4<<<dim3(unsigned((n4 + 127) / 128), (unsigned) n_tok), 128, 0, static_cast<cudaStream_t>(stream)>>>(
            reinterpret_cast<const float4*>(parts), weights, reinterpret_cast<const float4*>(shared),
            reinterpret_cast<float4*>(output), n4);
    } else {
        combine<<<dim3(unsigned((n_embd + 255) / 256), (unsigned) n_tok), 256, 0, static_cast<cudaStream_t>(stream)>>>(
            parts, weights, shared, output, n_embd, int(k));
    }
    const auto error = cudaGetLastError();
    if (error != cudaSuccess) throw std::runtime_error(cudaGetErrorString(error));
}
void native_moe_combine_multi_hits_gated(const float* hits, const float* weights, const float* shared,
                                         const float* shared_gate, float* output, int64_t n_embd, int64_t k, int n_tok,
                                         void* stream) {
    if (!stream || n_embd <= 0 || k < 1 || k > 15 || n_tok < 1 || !shared || !shared_gate)
        throw std::invalid_argument("native MoE combine (hits, gated) requires a stream, width, 1..15 experts, tokens");
    combine<true, true><<<dim3(unsigned((n_embd + 255) / 256), (unsigned) n_tok), 256, 0, static_cast<cudaStream_t>(stream)>>>(
        hits, weights, shared, output, n_embd, int(k), shared_gate);
    const auto error = cudaGetLastError();
    if (error != cudaSuccess) throw std::runtime_error(cudaGetErrorString(error));
}
void native_moe_combine_multi_hits(const float* hits, const float* weights, const float* shared, float* output,
                                   int64_t n_embd, int64_t k, int n_tok, void* stream) {
    if (!stream || n_embd <= 0 || k < 1 || k > 15 || n_tok < 1)
        throw std::invalid_argument("native MoE combine (hits) requires a stream, width, 1..15 experts, tokens");
    combine<true><<<dim3(unsigned((n_embd + 255) / 256), (unsigned) n_tok), 256, 0, static_cast<cudaStream_t>(stream)>>>(
        hits, weights, shared, output, n_embd, int(k));
    const auto error = cudaGetLastError();
    if (error != cudaSuccess) throw std::runtime_error(cudaGetErrorString(error));
}
}
