// Adapted from llama.cpp 3cf03257f219afbe7334045ff7c6a06ac68c627d:
// src/models/qwen4exp.cpp; ggml/src/ggml-cuda/{set-rows.cu,norm.cu,rope.cu}.
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

#include "strata/kernels/native_qsa_indexer.hpp"
#include "strata/kernels/mrope.hpp"
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <atomic>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <stdexcept>

namespace strata::kernels {
namespace {
std::atomic<bool> enabled{false};
constexpr int D = 128, R = 4, ROT = 64, THREADS = 256;
__device__ float warp_sum(float x) {
#pragma unroll
    for (int offset = 16; offset; offset >>= 1)
        x += __shfl_xor_sync(0xffffffffu, x, offset);
    return x;
}
__global__ void append(const float* __restrict__ raw, const int32_t* __restrict__ pos_dev,
                        int pos_base, const float* __restrict__ gamma, float epsilon,
                        float* __restrict__ tail, float* __restrict__ dead,
                        float* __restrict__ pooled, int32_t* __restrict__ block_pos,
                        int max_cells, float theta_scale, const int32_t* __restrict__ mtab) {
    const int pos = *pos_dev, d = threadIdx.x;
    if (pos < 0 || pos >= max_cells) return;
    const int slot = pos % R;
    float incoming = 0.0f;
    if (d < D) {
        // SET_ROWS stores F16; GET_ROWS expands those exact values to F32.
        incoming = __half2float(__float2half_rn(raw[d]));
        if (slot < R - 1) tail[slot * D + d] = incoming;
    }
    if (pos != 0 && slot != R - 1) return;
    __shared__ float values[D];
    __shared__ float partials[32];
    float mean = 0.0f;
    if (d < D) {
        // The spare's four gather indices all name cell zero. Completed blocks
        // use chronological slices; each graph ADD materializes an F32 sum.
        float sum = pos == 0 ? incoming : tail[d];
#pragma unroll
        for (int j = 1; j < R; ++j)
            sum = __fadd_rn(sum, pos == 0 || j == R - 1 ? incoming : tail[j * D + d]);
        mean = __fmaf_rn(0.25f, sum, 0.0f); // SCALE includes a zero bias.
    }
    float square_sum = 0.0f;
    if (d < D) square_sum += mean * mean;
    square_sum = warp_sum(square_sum);
    const int lane = d % 32;
    if (lane == 0) partials[d / 32] = square_sum;
    __syncthreads();
    square_sum = lane < THREADS / 32 ? partials[lane] : 0.0f;
    square_sum = warp_sum(square_sum);
    const float scale = rsqrtf(square_sum / D + epsilon);
    if (d < D) values[d] = scale * mean * gamma[d];
    __syncthreads();
    if (d >= D) return;
    const int b = pos / R;
    const int rope_pos = pos == 0 ? 0 : pos_base + R * b;
    float y = values[d];
    if (d < ROT) {
        const int pair = d % (ROT / 2);
        const float theta = (pos == 0 ? 0 : mrope_pos(mtab, rope_pos, pair)) * powf(theta_scale, float(pair));
        const float c = cosf(theta), s = sinf(theta);
        const float a = values[pair], z = values[pair + ROT / 2];
        y = d < ROT / 2 ? a * c - z * s : a * s + z * c;
    }
    pooled[std::size_t(b) * D + d] = y;
    if (pos == 0) dead[d] = y;
    else pooled[std::size_t(b + 1) * D + d] = dead[d];
    if (d == 0 && pos != 0) *block_pos = rope_pos;
}
// `append` for a whole chunk of contiguous positions in two launches instead of one launch
// per token.  Block t does what the per-token kernel does at the token that COMPLETES a 4-cell block (slot 3) or at
// position 0 - same arithmetic in the same order, so the pooled keys are bit-identical.  The members of a block come
// from the chunk's own raw rows (rounded to F16 exactly as `tail` stores them) or, for positions before the chunk,
// from `tail` as it stood before the chunk.  What the per-token sequence would leave behind that later tokens
// overwrite is simply not written: position 0's pooled[0] (overwritten when block 0 completes in the same chunk) and
// every pooled[b+1] = dead placeholder but the last one; those, `block_pos` and the new `tail` are written by
// append_batch_finish after all blocks have read the old `tail`.
__global__ void append_batch(const float* __restrict__ raw, int64_t T, int pos0, int pos_base,
                             const float* __restrict__ gamma, float epsilon, const float* __restrict__ tail,
                             float* __restrict__ dead, float* __restrict__ pooled, int max_cells, float theta_scale,
                             const int32_t* __restrict__ mtab) {
    const int t = blockIdx.x, d = threadIdx.x;
    const int pos = pos0 + t;
    if (pos < 0 || pos >= max_cells) return;
    const int slot = pos % R;
    if (pos != 0 && slot != R - 1) return;
    // member j of this block: position pos - (R-1) + j
    auto member = [&](int j) -> float {
        const int tt = t - (R - 1) + j;
        return tt >= 0 ? __half2float(__float2half_rn(raw[(int64_t) tt * D + d])) : tail[j * D + d];
    };
    __shared__ float values[D];
    __shared__ float partials[32];
    float mean = 0.0f;
    if (d < D) {
        const float incoming = __half2float(__float2half_rn(raw[(int64_t) t * D + d]));
        float sum = pos == 0 ? incoming : member(0);
#pragma unroll
        for (int j = 1; j < R; ++j)
            sum = __fadd_rn(sum, pos == 0 || j == R - 1 ? incoming : member(j));
        mean = __fmaf_rn(0.25f, sum, 0.0f);
    }
    float square_sum = 0.0f;
    if (d < D) square_sum += mean * mean;
    square_sum = warp_sum(square_sum);
    const int lane = d % 32;
    if (lane == 0) partials[d / 32] = square_sum;
    __syncthreads();
    square_sum = lane < THREADS / 32 ? partials[lane] : 0.0f;
    square_sum = warp_sum(square_sum);
    const float scale = rsqrtf(square_sum / D + epsilon);
    if (d < D) values[d] = scale * mean * gamma[d];
    __syncthreads();
    if (d >= D) return;
    const int b = pos / R;
    const int rope_pos = pos == 0 ? 0 : pos_base + R * b;
    float y = values[d];
    if (d < ROT) {
        const int pair = d % (ROT / 2);
        const float theta = (pos == 0 ? 0 : mrope_pos(mtab, rope_pos, pair)) * powf(theta_scale, float(pair));
        const float c = cosf(theta), s = sinf(theta);
        const float a = values[pair], z = values[pair + ROT / 2];
        y = d < ROT / 2 ? a * c - z * s : a * s + z * c;
    }
    if (pos == 0) {
        dead[d] = y;
        if (pos0 + T - 1 < R - 1 || R - 1 >= max_cells) pooled[d] = y;   // block 0 does not complete in this chunk
    } else {
        pooled[std::size_t(b) * D + d] = y;
    }
}
// the chunk's end state: the last completed block's spare placeholder and position, and the tail (one block, D threads)
__global__ void append_batch_finish(const float* __restrict__ raw, int64_t T, int pos0, int pos_base,
                                    float* __restrict__ tail, const float* __restrict__ dead,
                                    float* __restrict__ pooled, int32_t* __restrict__ block_pos, int max_cells) {
    const int d = threadIdx.x;
    const int64_t last64 = (int64_t) pos0 + T - 1 < (int64_t) max_cells - 1 ? (int64_t) pos0 + T - 1 : (int64_t) max_cells - 1;
    const int last = (int) last64;   // last position appended
    if (last < pos0) return;
    // the last completing position (slot R-1, not 0) in [pos0, last]
    const int lc = (last + 1) / R * R - 1;
    if (lc >= pos0 && lc > 0) {
        const int b = lc / R;
        pooled[std::size_t(b + 1) * D + d] = dead[d];
        if (d == 0) *block_pos = pos_base + R * b;
    }
    // tail slot j: the last position p in [pos0, last] with p % R == j (j < R-1)
    for (int j = 0; j < R - 1; ++j) {
        int p = last - ((last % R) - j + R) % R;
        if (p >= pos0) tail[j * D + d] = __half2float(__float2half_rn(raw[(int64_t) (p - pos0) * D + d]));
    }
}
struct Span { const void* p; std::size_t n; };
void validate(Span s) {
    const auto p = reinterpret_cast<std::uintptr_t>(s.p);
    if (!p || p % 4 || s.n > UINTPTR_MAX - p)
        throw std::invalid_argument("native QSA indexer requires aligned bounded spans");
}
bool overlaps(Span a, Span b) {
    const auto x = reinterpret_cast<std::uintptr_t>(a.p), y = reinterpret_cast<std::uintptr_t>(b.p);
    return x < y + b.n && y < x + a.n;
}
} // namespace

void native_qsa_indexer_set_enabled(bool value) { enabled.store(value, std::memory_order_relaxed); }
bool native_qsa_indexer_enabled() { return enabled.load(std::memory_order_relaxed); }
void native_qsa_indexer_append(const float* raw, const int32_t* relative_pos_device, int32_t pos_base,
                               const float* gamma, float epsilon, const QsaIndexerBuffers& b,
                               const QsaShapes& s, int64_t max_cells, float freq_base, void* stream) {
    if (!stream || s.idx_dim != D || s.idx_block != R || s.n_rot != ROT ||
        max_cells < 1 || max_cells > INT32_MAX || pos_base < 0 || pos_base % R ||
        int64_t(pos_base) + max_cells > INT32_MAX || !std::isfinite(epsilon) || epsilon <= 0.0f ||
        !std::isfinite(freq_base) || freq_base <= 1.0f)
        throw std::invalid_argument("native QSA indexer requires fixed geometry, aligned position base, positive capacity/epsilon, valid frequency and explicit stream");
    const Span spans[] = {{raw,D*4},{relative_pos_device,4},{gamma,D*4},{b.tail,(R-1)*D*4},
        {b.dead,D*4},{b.pooled,std::size_t(max_cells/R+1)*D*4},{b.block_pos,4}};
    for (const auto& span : spans) validate(span);
    for (int i = 0; i < 7; ++i) for (int j = i + 1; j < 7; ++j)
        if (overlaps(spans[i], spans[j])) throw std::invalid_argument("native QSA indexer buffers overlap");
    const float theta_scale = powf(freq_base, -2.0f / ROT);
    append<<<1,THREADS,0,static_cast<cudaStream_t>(stream)>>>(raw,relative_pos_device,pos_base,gamma,epsilon,
        b.tail,b.dead,b.pooled,b.block_pos,int(max_cells),theta_scale,mrope_table());
    const auto error = cudaGetLastError();
    if (error != cudaSuccess) throw std::runtime_error(cudaGetErrorString(error));
}
void native_qsa_indexer_append_chunk(const float* raw, int64_t T, int64_t pos0, int32_t pos_base, const float* gamma,
                                     float epsilon, const QsaIndexerBuffers& b, const QsaShapes& s,
                                     int64_t max_cells, float freq_base, void* stream) {
    if (T <= 0) return;
    if (!stream || s.idx_dim != D || s.idx_block != R || s.n_rot != ROT || pos0 < 0 || pos0 + T > INT32_MAX ||
        max_cells < 1 || max_cells > INT32_MAX || pos_base < 0 || pos_base % R ||
        int64_t(pos_base) + max_cells > INT32_MAX || !std::isfinite(epsilon) || epsilon <= 0.0f ||
        !std::isfinite(freq_base) || freq_base <= 1.0f)
        throw std::invalid_argument("native QSA indexer (chunk) requires fixed geometry and valid positions");
    const Span spans[] = {{raw,std::size_t(T)*D*4},{gamma,D*4},{b.tail,(R-1)*D*4},
        {b.dead,D*4},{b.pooled,std::size_t(max_cells/R+1)*D*4},{b.block_pos,4}};
    for (const auto& span : spans) validate(span);
    for (int i = 0; i < 6; ++i) for (int j = i + 1; j < 6; ++j)
        if (overlaps(spans[i], spans[j])) throw std::invalid_argument("native QSA indexer buffers overlap");
    const float theta_scale = powf(freq_base, -2.0f / ROT);
    const auto st = static_cast<cudaStream_t>(stream);
    append_batch<<<(unsigned) T, THREADS, 0, st>>>(raw, T, (int) pos0, pos_base, gamma, epsilon, b.tail, b.dead,
                                                   b.pooled, int(max_cells), theta_scale, mrope_table());
    append_batch_finish<<<1, D, 0, st>>>(raw, T, (int) pos0, pos_base, b.tail, b.dead, b.pooled, b.block_pos,
                                         int(max_cells));
    const auto error = cudaGetLastError();
    if (error != cudaSuccess) throw std::runtime_error(cudaGetErrorString(error));
}
} // namespace strata::kernels
