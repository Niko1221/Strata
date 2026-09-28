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
// n_tok tokens: token t's taps read the normalized rows of tokens t-9, t-6 and t-3 (history rows t, t+3 and t+6
// before the chunk while those tokens precede it) and its own.
__global__ void broadcast_tokens_kernel(const float* value, const float* gate, float* gated, int n_tok) {
    const size_t i = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i < size_t(n_tok) * D) gated[i] = __fmul_rn(value[i / D * N + i % N], gate[i / N]);
}
__global__ void conv_residual_tokens_kernel(const float* history, const float* normalized,
                                           const uint16_t* weights, const float* hidden,
                                           const float* gated, float* result, int n_tok) {
    const size_t i = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= size_t(n_tok) * D) return;
    const int t = int(i / D), c = int(i % D);
    float sum = 0;
#pragma unroll
    for (int k = 0; k < 4; ++k) {
        const int j = t - HISTORY + 3 * k;
        const float x = j >= 0 ? normalized[size_t(j) * D + c] : history[c * HISTORY + j + HISTORY];
        const float w = __half2float(__ushort_as_half(weights[c * 4 + k]));
        const float term = __fmul_rn(x, w);
        sum = k == 0 ? term : __fadd_rn(sum, term);
    }
    const float activation = sum / (1.0f + expf(-sum));
    result[i] = __fadd_rn(hidden[i], __fadd_rn(gated[i], activation));
}
// The history after the chunk: its last nine normalized rows, older rows moved up when the chunk is shorter; with
// `snaps`, the history after each of its tokens too.
__global__ void history_tokens_kernel(float* history, const float* normalized, int n_tok, float* snaps) {
    const int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= D) return;
    float old[HISTORY];
#pragma unroll
    for (int r = 0; r < HISTORY; ++r) old[r] = history[c * HISTORY + r];
    if (snaps != nullptr)
        for (int t = 1; t <= n_tok; ++t)
#pragma unroll
            for (int r = 0; r < HISTORY; ++r) {
                const int j = t - HISTORY + r;
                snaps[size_t(t - 1) * HISTORY * D + c * HISTORY + r] =
                    j >= 0 ? normalized[size_t(j) * D + c] : old[r + t];
            }
#pragma unroll
    for (int r = 0; r < HISTORY; ++r) {
        const int j = n_tok - HISTORY + r;
        history[c * HISTORY + r] = j >= 0 ? normalized[size_t(j) * D + c] : old[r + n_tok];
    }
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

void native_ple_postops_tokens(const float* projected_key, const float* hidden, const float* value, float* history,
                               const PleWeights& w, const NativePleTokensBuffers& b, int n_tok, void* stream,
                               float* snaps) {
    if (!stream || n_tok < 1) throw std::invalid_argument("native PLE postops: tokens need a stream and n_tok >= 1");
    const size_t rows = size_t(n_tok) * D * 4;
    const Span inputs[] = {{projected_key,rows,4}, {hidden,rows,4}, {value,size_t(n_tok)*N*4,4},
        {history,HISTORY*D*4,4}, {w.norm_key,D*4,4}, {w.norm_query,D*4,4},
        {w.norm_conv,D*4,4}, {w.conv1d_f16,4*D*2,2}};
    const Span outputs[] = {{b.key,rows,4}, {b.query,rows,4}, {b.gate,size_t(n_tok)*H*4,4},
        {b.gated,rows,4}, {b.normalized,rows,4}, {b.result,rows,4}};
    for (const auto& span : inputs) validate(span);
    for (const auto& span : outputs) validate(span);
    for (size_t i = 0; i < 6; ++i) {
        for (size_t j = 0; j < 8; ++j)
            if (!(i == 5 && j == 1 && b.result == hidden) && overlaps(outputs[i], inputs[j]))
                throw std::invalid_argument("native PLE postops output overlaps an input, the history or a weight");
        for (size_t j = i + 1; j < 6; ++j)
            if (!(i == 1 && j == 4 && b.query == b.normalized) && overlaps(outputs[i], outputs[j]))
                throw std::invalid_argument("native PLE postops writable spans overlap");
    }
    if (snaps != nullptr) {
        const Span s{snaps, size_t(n_tok) * HISTORY * D * 4, 4};
        validate(s);
        for (const auto& span : inputs)
            if (overlaps(s, span)) throw std::invalid_argument("native PLE postops: the snapshots overlap an input");
        for (const auto& span : outputs)
            if (overlaps(s, span)) throw std::invalid_argument("native PLE postops: the snapshots overlap an output");
    }
    const int R = H * n_tok;
    const unsigned blocks = unsigned((size_t(n_tok) * D + 255) / 256);
    auto st = static_cast<cudaStream_t>(stream);
    native_gr_rms_norm_weighted_repeat(projected_key,w.norm_key,b.key,N,R,H,NG_RMS_EPS,stream);
    native_gr_rms_norm_weighted_repeat(hidden,w.norm_query,b.query,N,R,H,NG_RMS_EPS,stream);
    gate_kernel<<<R,512,0,st>>>(b.key,b.query,b.gate,1.0f / std::sqrt(float(N)));
    broadcast_tokens_kernel<<<blocks,256,0,st>>>(value,b.gate,b.gated,n_tok);
    launch_check();
    native_gr_rms_norm_weighted_repeat(b.gated,w.norm_conv,b.normalized,N,R,H,NG_RMS_EPS,stream);
    conv_residual_tokens_kernel<<<blocks,256,0,st>>>(history,b.normalized,w.conv1d_f16,hidden,b.gated,b.result,n_tok);
    history_tokens_kernel<<<D/256,256,0,st>>>(history,b.normalized,n_tok,snaps);
    launch_check();
}
} // namespace strata::kernels
