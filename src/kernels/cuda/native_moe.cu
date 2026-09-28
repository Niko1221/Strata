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
#include <cuda_runtime.h>
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <stdexcept>

namespace strata::kernels {
namespace {
std::atomic<bool> enabled{false};
__global__ void combine(const float* __restrict__ parts, const float* __restrict__ weights,
                        const float* __restrict__ shared, float* __restrict__ output,
                        int64_t n_embd, int k) {
    const int64_t col = int64_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (col >= n_embd) return;
    float sum = parts[col] * weights[0];
    for (int expert = 1; expert < k; ++expert) {
        sum += parts[int64_t(expert) * n_embd + col] * weights[expert];
    }
    if (shared) sum += shared[col];
    output[col] = sum;
}
constexpr int kMaxK = 15;
// A shared expert's row times its scalar gate, in shared_expert_gate_rows' instructions (shared_expert.cu builds
// without fast math: nothing flushes to zero): sigmoid(logit) as __fdividef(1, 1 + __expf(-logit)), then the product.
// The explicit rounding keeps ptxas from fusing the product into the sum after it.
__device__ __forceinline__ float gated_shared(float row, float logit) {
    float e, d, g, y;
    const float one = 1.0f;
    asm("mul.rn.f32 %0, %1, 0fBFB8AA3B;" : "=f"(e) : "f"(logit));
    asm("ex2.approx.f32 %0, %1;" : "=f"(d) : "f"(e));
    asm("add.rn.f32 %0, %1, %2;" : "=f"(e) : "f"(d), "f"(one));
    asm("div.approx.f32 %0, %1, %2;" : "=f"(g) : "f"(one), "f"(e));
    asm("mul.rn.f32 %0, %1, %2;" : "=f"(y) : "f"(row), "f"(g));
    return y;
}
// One block row per token.  The rows a GPU computed are marked from the plan's entries first; each column then
// loads its k values (the host's over PCIe, all in flight at once) and sums them as `combine` does, then adds the
// shared expert's row, times its gate when `shared_gate` holds the gates' logits.
__global__ void gather_combine(const float* __restrict__ gpu_rows, const float* host_rows,
                               const int32_t* __restrict__ dst, const int32_t* __restrict__ count,
                               const int32_t* __restrict__ dst2, const int32_t* __restrict__ count2,
                               const int32_t* __restrict__ dst3, const int32_t* __restrict__ count3,
                               const float* __restrict__ weights, const float* __restrict__ shared,
                               const float* __restrict__ shared_gate, float* __restrict__ output, int64_t n_embd,
                               int k) {
    __shared__ unsigned on_gpu, as_is;   // bit j: row t*k + j is in gpu_rows, a hit / a row taken as it is
    const int t = blockIdx.y;
    if (threadIdx.x == 0) on_gpu = as_is = 0u;
    __syncthreads();
    const int c = *count, c2 = dst2 != nullptr ? *count2 : 0, c3 = dst3 != nullptr ? *count3 : 0;
    for (int i = threadIdx.x; i < c + c2 + c3; i += blockDim.x) {
        const int r = (i < c ? dst[i] : i < c + c2 ? dst2[i - c] : dst3[i - c - c2]) - t * k;
        if (r >= 0 && r < k) atomicOr(i < c + c2 ? &on_gpu : &as_is, 1u << r);
    }
    __syncthreads();
    const int64_t col = int64_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (col >= n_embd) return;
    const unsigned mask = on_gpu, same = as_is;
    const float* w = weights + int64_t(t) * k;
    float v[kMaxK];
#pragma unroll
    for (int j = 0; j < kMaxK; ++j) {
        if (j >= k) break;
        const int64_t at = (int64_t(t) * k + j) * n_embd + col;
        if ((same >> j) & 1u) {
            v[j] = gpu_rows[at];
        } else if ((mask >> j) & 1u) {
            const float h = gpu_rows[at];
            v[j] = __float_as_uint(h) == 0x80000000u ? 0.0f : h;   // 0 + h: the host's zeroed row plus the hit
        } else {
            v[j] = host_rows[at];
        }
    }
    float sum = v[0] * w[0];
#pragma unroll
    for (int j = 1; j < kMaxK; ++j) {
        if (j >= k) break;
        sum += v[j] * w[j];
    }
    if (shared) {
        const float s = shared[int64_t(t) * n_embd + col];
        sum += shared_gate ? gated_shared(s, shared_gate[t]) : s;
    }
    output[int64_t(t) * n_embd + col] = sum;
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
    combine<<<unsigned((n_embd + 255) / 256), 256, 0, static_cast<cudaStream_t>(stream)>>>(
        parts, weights, shared, output, n_embd, int(k));
    const auto error = cudaGetLastError();
    if (error != cudaSuccess) throw std::runtime_error(cudaGetErrorString(error));
}
void native_moe_gather_combine(const float* gpu_rows, const float* host_rows, const int32_t* dst,
                               const int32_t* count, const int32_t* dst2, const int32_t* count2, const int32_t* dst3,
                               const int32_t* count3, const float* weights, const float* shared,
                               const float* shared_gate, float* output, int64_t n_embd, int64_t k, int n_tok,
                               void* stream) {
    if (!stream || n_embd <= 0 || n_embd > std::numeric_limits<int>::max() || k < 1 || k > kMaxK || n_tok < 1 ||
        !gpu_rows || !host_rows || !dst || !count || (dst2 != nullptr && count2 == nullptr) ||
        (dst3 != nullptr && count3 == nullptr) || !weights || !output || (shared_gate != nullptr && !shared))
        throw std::invalid_argument("native MoE gather-combine requires a stream, positive width, 1..15 experts "
                                    "and its buffers");
    const dim3 grid(unsigned((n_embd + 255) / 256), unsigned(n_tok));
    gather_combine<<<grid, 256, 0, static_cast<cudaStream_t>(stream)>>>(gpu_rows, host_rows, dst, count, dst2, count2,
                                                                         dst3, count3, weights, shared, shared_gate,
                                                                         output, n_embd, int(k));
    const auto error = cudaGetLastError();
    if (error != cudaSuccess) throw std::runtime_error(cudaGetErrorString(error));
}
}
