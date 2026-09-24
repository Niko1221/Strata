// Arithmetic adapted from llama.cpp 3cf03257f219afbe7334045ff7c6a06ac68c627d:
// src/models/qwen4exp.cpp and ggml-cuda/{reduce_rows.cuh,sumrows.cu,unary.cu}.
// MIT License
// Copyright (c) 2023-2026 The ggml authors
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.

#include "strata/kernels/native_ple_postops.hpp"
#include "strata/kernels/native_gr_norm.hpp"
#include "strata/kernels/ngram.hpp"
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string>

namespace strata::kernels {
namespace {
constexpr int N = 2560, H = 4, D = N * H, HISTORY = 9;
__device__ float warp_sum(float x) {
    for (int offset = 16; offset; offset >>= 1) x += __shfl_xor_sync(0xffffffffu, x, offset);
    return x;
}
__global__ void gate_kernel(const float* key, const float* query, float* gate, float scale) {
    // SUM_ROWS selects 512 threads for four rows on the target GPU. Preserve
    // its eight partial lanes and materialized MUL rounding (never a dot FMA).
    float sums[8] = {};
#pragma unroll
    for (int j = 0; j < 8; ++j) {
        const int d = threadIdx.x + j * 512;
        const float p = d < N ? __fmul_rn(key[blockIdx.x * N + d], query[blockIdx.x * N + d]) : 0.0f;
        sums[j] += p;
    }
    float sum = 0;
#pragma unroll
    for (int j = 0; j < 8; ++j) sum += sums[j];
    __shared__ float partials[32];
    sum = warp_sum(sum);
    const int lane = threadIdx.x % 32;
    if (!lane) partials[threadIdx.x / 32] = sum;
    __syncthreads();
    sum = lane < 16 ? partials[lane] : 0.0f;
    sum = warp_sum(sum);
    if (threadIdx.x == 0) {
        const float s = __fmaf_rn(scale, sum, 0.0f); // ggml SCALE's zero bias
        const float mag = sqrtf(fmaxf(fabsf(s), 1e-6f));
        const float sign = float((s > 0.0f) - (s < 0.0f));
        gate[blockIdx.x] = 1.0f / (1.0f + expf(-__fmul_rn(sign, mag)));
    }
}
__global__ void broadcast_kernel(const float* value, const float* gate, float* gated) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < D) gated[i] = __fmul_rn(value[i % N], gate[i / N]);
}
__global__ void conv_residual_kernel(const float* history, const float* normalized,
                                    const uint16_t* weights, const float* hidden,
                                    const float* gated, float* conv, float* result) {
    const int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= D) return;
    float sum = 0;
#pragma unroll
    for (int k = 0; k < 4; ++k) {
        const float x = k == 3 ? normalized[c] : history[c * HISTORY + 3 * k];
        const float w = __half2float(__ushort_as_half(weights[c * 4 + k]));
        const float term = __fmul_rn(x, w);
        sum = k == 0 ? term : __fadd_rn(sum, term);
    }
    const float activation = sum / (1.0f + expf(-sum));
    conv[c] = activation;
    // Exact hidden/result alias is safe: each thread owns one element.
    result[c] = __fadd_rn(hidden[c], __fadd_rn(gated[c], activation));
}
struct Span { const void* p; size_t bytes; size_t alignment; };
bool overlaps(Span a, Span b) {
    const auto x = reinterpret_cast<uintptr_t>(a.p), y = reinterpret_cast<uintptr_t>(b.p);
    return x < y + b.bytes && y < x + a.bytes;
}
void validate(Span span) {
    const auto p = reinterpret_cast<uintptr_t>(span.p);
    if (!p || p % span.alignment || p > std::numeric_limits<uintptr_t>::max() - span.bytes)
        throw std::invalid_argument("native PLE postops require nonnull aligned bounded spans");
}
void launch_check() {
    const auto error = cudaGetLastError();
    if (error != cudaSuccess) throw std::runtime_error(std::string("native PLE postops launch: ") + cudaGetErrorString(error));
}
} // namespace

void native_ple_postops(const float* projected_key, const float* hidden,
                        const float* value, const float* history,
                        const PleWeights& w, const NativePlePostopsBuffers& b, void* stream) {
    if (!stream) throw std::invalid_argument("native PLE postops require an explicit stream");
    const Span inputs[] = {{projected_key,D*4,4}, {hidden,D*4,4}, {value,N*4,4},
        {history,HISTORY*D*4,4}, {w.norm_key,D*4,4}, {w.norm_query,D*4,4},
        {w.norm_conv,D*4,4}, {w.conv1d_f16,4*D*2,2}};
    const Span outputs[] = {{b.key,D*4,4}, {b.query,D*4,4}, {b.gate,H*4,4},
        {b.gated,D*4,4}, {b.normalized,D*4,4}, {b.conv,D*4,4}, {b.result,D*4,4}};
    for (const auto& span : inputs) validate(span);
    for (const auto& span : outputs) validate(span);
    for (size_t i = 0; i < 7; ++i) {
        for (size_t j = 0; j < 8; ++j)
            if (!(i == 6 && j == 1 && b.result == hidden) && overlaps(outputs[i], inputs[j]))
                throw std::invalid_argument("native PLE postops output overlaps an input or weight");
        for (size_t j = i + 1; j < 7; ++j)
            if (!(i == 1 && j == 4 && b.query == b.normalized) && overlaps(outputs[i], outputs[j]))
                throw std::invalid_argument("native PLE postops writable spans overlap");
    }
    native_gr_rms_norm_weighted(projected_key,w.norm_key,b.key,N,H,NG_RMS_EPS,stream);
    native_gr_rms_norm_weighted(hidden,w.norm_query,b.query,N,H,NG_RMS_EPS,stream);
    auto st = static_cast<cudaStream_t>(stream);
    gate_kernel<<<H,512,0,st>>>(b.key,b.query,b.gate,1.0f / std::sqrt(float(N)));
    broadcast_kernel<<<D/256,256,0,st>>>(value,b.gate,b.gated);
    launch_check();
    native_gr_rms_norm_weighted(b.gated,w.norm_conv,b.normalized,N,H,NG_RMS_EPS,stream);
    conv_residual_kernel<<<D/256,256,0,st>>>(history,b.normalized,w.conv1d_f16,hidden,b.gated,b.conv,b.result);
    launch_check();
}
} // namespace strata::kernels
