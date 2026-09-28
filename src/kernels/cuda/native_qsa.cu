// Adapted from llama.cpp 3cf03257f219afbe7334045ff7c6a06ac68c627d:
// ggml/src/ggml-cuda/{norm.cu,common.cuh,unary.cu}.
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

#include "strata/kernels/native_qsa.hpp"
#include "strata/kernels/mrope.hpp"
#include <cuda_runtime.h>
#include <atomic>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string>

namespace strata::kernels {
namespace {
std::atomic<bool> enabled{false};
__device__ __forceinline__ float warp_sum(float value) {
#pragma unroll
    for (int offset = 16; offset; offset >>= 1)
        value += __shfl_xor_sync(0xffffffffu, value, offset, 32);
    return value;
}
template<int BlockSize>
__global__ void norm(const float* input, const float* __restrict__ gamma, float* output,
                     int n_cols, float epsilon) {
    const int tid = threadIdx.x;
    const std::size_t row_offset = std::size_t(blockIdx.x) * n_cols;
    input += row_offset; output += row_offset;
    float partial = 0.0f;
    for (std::size_t col = tid; col < std::size_t(n_cols); col += BlockSize) {
        const float x = input[col];
        partial += x * x;
    }
    __shared__ float sums[32];
    partial = warp_sum(partial);
    const int lane = tid % 32;
    if (lane == 0) sums[tid / 32] = partial;
    // All reads of input for the reduction precede this barrier. Afterwards,
    // each thread reads/writes only its own elements, allowing exact in-place use.
    __syncthreads();
    partial = lane < BlockSize / 32 ? sums[lane] : 0.0f;
    partial = warp_sum(partial);
    const float mean = partial / n_cols;
    const float scale = rsqrtf(mean + epsilon);
    for (std::size_t col = tid; col < std::size_t(n_cols); col += BlockSize)
        output[col] = scale * input[col] * gamma[col];
}
// `norm<256>`, then native_rope.cu's `apply` on its output, one block a row of at most 256 columns: the normalized
// row waits in shared memory.  Row r is read at input + r * in_stride.
__global__ void norm_rope(const float* input, int in_stride, const float* __restrict__ gamma, float* output,
                          int n_cols, float epsilon, int n_rot, float theta_scale, const int* __restrict__ positions,
                          const int32_t* __restrict__ mtab, int heads, int pos_stride) {
    constexpr int BlockSize = 256;
    const int tid = threadIdx.x, row = blockIdx.x;
    input += std::size_t(row) * in_stride;
    output += std::size_t(row) * n_cols;
    float partial = 0.0f;
    for (std::size_t col = tid; col < std::size_t(n_cols); col += BlockSize) {
        const float x = input[col];
        partial += x * x;
    }
    __shared__ float sums[32];
    __shared__ float values[BlockSize];
    partial = warp_sum(partial);
    const int lane = tid % 32;
    if (lane == 0) sums[tid / 32] = partial;
    __syncthreads();
    partial = lane < BlockSize / 32 ? sums[lane] : 0.0f;
    partial = warp_sum(partial);
    const float mean = partial / n_cols;
    const float scale = rsqrtf(mean + epsilon);
    for (std::size_t col = tid; col < std::size_t(n_cols); col += BlockSize)
        values[col] = scale * input[col] * gamma[col];
    __syncthreads();   // every read of input precedes this: output may be input
    const int pair = tid;
    if (pair >= n_cols / 2) return;
    if (pair >= n_rot / 2) {
        output[2 * pair] = values[2 * pair];
        output[2 * pair + 1] = values[2 * pair + 1];
        return;
    }
    const int position = positions[(row / heads) * pos_stride + row % heads];
    const float theta = mrope_pos(mtab, position, pair) * powf(theta_scale, float(pair));
    const float c = cosf(theta), s = sinf(theta);
    const float a = values[pair], b = values[pair + n_rot / 2];
    output[pair] = a * c - b * s;
    output[pair + n_rot / 2] = a * s + b * c;
}
__global__ void gate(const float* attn, const float* __restrict__ q_full, float* output,
                     int n_head, int head_dim) {
    const std::size_t i = std::size_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= std::size_t(n_head) * head_dim) return;
    const std::size_t head = i / head_dim, channel = i % head_dim;
    const float raw = q_full[head * 2 * head_dim + head_dim + channel];
    const float sigmoid = 1.0f / (1.0f + expf(-raw));
    output[i] = attn[i] * sigmoid;
}
std::size_t elements(int cols, int rows) {
    if (cols <= 0 || rows <= 0 || std::uint64_t(cols) * rows > std::uint64_t(std::numeric_limits<int>::max()))
        throw std::invalid_argument("native QSA requires positive bounded dimensions");
    return std::size_t(cols) * rows;
}
bool valid(const void* ptr, std::size_t bytes) {
    const auto address = reinterpret_cast<std::uintptr_t>(ptr);
    return ptr && address % 4 == 0 && bytes <= UINTPTR_MAX - address;
}
bool overlap(const void* a, std::size_t an, const void* b, std::size_t bn) {
    const auto ap = reinterpret_cast<std::uintptr_t>(a), bp = reinterpret_cast<std::uintptr_t>(b);
    return ap < bp + bn && bp < ap + an;
}
void buffers(const float* input, std::size_t in_bytes, const float* weight, std::size_t weight_bytes,
             float* output, void* stream) {
    if (!stream || !valid(input, in_bytes) || !valid(weight, weight_bytes) || !valid(output, in_bytes) ||
        overlap(input, in_bytes, weight, weight_bytes) || overlap(output, in_bytes, weight, weight_bytes) ||
        (input != output && overlap(input, in_bytes, output, in_bytes)))
        throw std::invalid_argument("native QSA requires a stream, aligned spans, and disjoint buffers or exact input/output alias");
}
void check_launch() {
    const auto result = cudaGetLastError();
    if (result != cudaSuccess)
        throw std::runtime_error(std::string("native QSA launch: ") + cudaGetErrorString(result));
}
} // namespace

void native_qsa_set_enabled(bool value) { enabled.store(value, std::memory_order_relaxed); }
bool native_qsa_enabled() { return enabled.load(std::memory_order_relaxed); }

void native_qsa_rms_norm_weighted(const float* input, const float* gamma, float* output,
                                  int n_cols, int n_rows, float epsilon, void* stream) {
    const auto count = elements(n_cols, n_rows);
    if (!std::isfinite(epsilon) || epsilon < 0.0f)
        throw std::invalid_argument("native QSA requires finite nonnegative epsilon");
    buffers(input, count * 4, gamma, std::size_t(n_cols) * 4, output, stream);
    if (n_cols < 1024)
        norm<256><<<unsigned(n_rows), 256, 0, static_cast<cudaStream_t>(stream)>>>(input, gamma, output, n_cols, epsilon);
    else
        norm<1024><<<unsigned(n_rows), 1024, 0, static_cast<cudaStream_t>(stream)>>>(input, gamma, output, n_cols, epsilon);
    check_launch();
}
void native_qsa_norm_rope_tokens(const float* input, int in_stride, const float* gamma, float* output, int n_cols,
                                 int n_rows, float epsilon, int n_rot, float freq_base, const int* positions, int heads,
                                 int pos_stride, void* stream) {
    const auto count = elements(n_cols, n_rows);
    if ((n_cols != 128 && n_cols != 256) || n_rot != 64 || in_stride < n_cols || heads < 1 || pos_stride < heads ||
        !std::isfinite(epsilon) || epsilon < 0.0f || !std::isfinite(freq_base) || freq_base <= 1.0f || !positions)
        throw std::invalid_argument("native QSA norm-rope requires 128 or 256 columns, 64 rotated, valid strides, "
                                    "epsilon and base");
    const std::size_t in_bytes = (std::size_t(n_rows - 1) * in_stride + n_cols) * 4;
    if (!stream || !valid(input, in_bytes) || !valid(gamma, std::size_t(n_cols) * 4) || !valid(output, count * 4) ||
        (input != output && overlap(input, in_bytes, output, count * 4)) ||
        (input == output && in_stride != n_cols) || overlap(output, count * 4, gamma, std::size_t(n_cols) * 4))
        throw std::invalid_argument("native QSA norm-rope requires a stream, aligned spans, and disjoint buffers or an "
                                    "exact input/output alias");
    // pinned host-side float powf, as native_rope_apply_tokens
    const float theta_scale = powf(freq_base, -2.0f / n_rot);
    norm_rope<<<unsigned(n_rows), 256, 0, static_cast<cudaStream_t>(stream)>>>(
        input, in_stride, gamma, output, n_cols, epsilon, n_rot, theta_scale, positions, mrope_table(), heads,
        pos_stride);
    check_launch();
}
void native_qsa_gate_apply(const float* attn, const float* q_full, float* output,
                           int n_head, int head_dim, void* stream) {
    const auto count = elements(head_dim, n_head);
    buffers(attn, count * 4, q_full, count * 8, output, stream);
    gate<<<unsigned((count + 255) / 256), 256, 0, static_cast<cudaStream_t>(stream)>>>(attn, q_full, output, n_head, head_dim);
    check_launch();
}
} // namespace strata::kernels
