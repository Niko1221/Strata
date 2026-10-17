// DFlash2 equations, independently expressed as direct CUDA/HIP kernels.
// Reference: llama.cpp src/models/dflash.cpp (a9d27ac), z-lab/dflash (MIT, 07ebd93).
#include "strata/kernels/dflash2.hpp"
#include <cuda_runtime.h>
#include <cmath>
#include <stdexcept>

namespace strata::kernels {
namespace {
__global__ void conv_kernel(const float* x, const float* dynamic, const float* base, float* out,
                            int rows, int hidden, int kernel, int group, int side, int block_rows) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= rows * hidden) return;
    const int row = i / hidden, channel = i % hidden, groups = hidden / group;
    float value = 0;
    const int limit = min(kernel, row % block_rows + 1);
    for (int tap = 0; tap < limit; ++tap) {
        const float weight = base[(side * kernel + tap) * hidden + channel] +
            dynamic[((row * 2 + side) * kernel + tap) * groups + channel / group];
        value += weight * x[(row - tap) * hidden + channel];
    }
    out[i] = value;
}
// The first implementation intentionally uses a serial scan per row. It is a correctness path,
// with no speed claim; replacing it by a parallel top-k requires a separate measured change.
__global__ void topk_kernel(const float* logits, int32_t* ids, float* unary, int vocab, int top_k) {
    if (threadIdx.x != 0) return;
    const int row = blockIdx.x;
    ids += row * top_k; unary += row * top_k; logits += (size_t) row * vocab;
    for (int k = 0; k < top_k; ++k) { ids[k] = -1; unary[k] = -INFINITY; }
    for (int token = 0; token < vocab; ++token) {
        const float score = logits[token];
        // Non-finite model outputs must not manufacture a token ID.
        if (!isfinite(score)) continue;
        int at = top_k;
        while (at > 0 && score > unary[at - 1]) --at;
        if (at == top_k) continue;
        for (int k = top_k - 1; k > at; --k) { ids[k] = ids[k - 1]; unary[k] = unary[k - 1]; }
        ids[at] = token; unary[at] = score;
    }
}
__global__ void select_kernel(const int32_t* ids, const float* unary, const float* hidden,
                              const float* prev, const float* next, const int32_t* anchor,
                              int32_t* out, float* confidence, int rows, int top_k, int rank) {
    if (threadIdx.x != 0 || blockIdx.x != 0) return;
    int predecessor = anchor[0];
    float scores[64];
    bool valid = true;
    for (int row = 1; row < rows; ++row) {
        int best = -1;
        for (int k = 0; k < top_k; ++k) {
            const int token = ids[row * top_k + k];
            float score = 0;
            if (!valid || token < 0) { scores[k] = -INFINITY; continue; }
            for (int j = 0; j < rank; ++j)
                score += (prev[(size_t) predecessor * rank + j] * hidden[row * rank + j]) *
                         next[(size_t) token * rank + j];
            score += unary[row * top_k + k];
            scores[k] = score;
            if (isfinite(score) && (best < 0 || score > scores[best])) best = k;
        }
        valid = best >= 0;
        if (!valid) { out[row - 1] = -1; if (confidence) confidence[row - 1] = 0; continue; }
        predecessor = ids[row * top_k + best];
        out[row - 1] = predecessor;
        if (confidence) {
            float sum = 0;
            for (int k = 0; k < top_k; ++k) sum += expf(scores[k] - scores[best]);
            confidence[row - 1] = 1.0f / sum;
        }
    }
}
}  // namespace

void dflash2_convolve(const float* x, const float* dynamic, const float* base, float* out,
                     int rows, int hidden, int kernel, int group, int side, int block_rows, void* stream) {
    if (rows <= 0 || hidden <= 0 || kernel <= 0 || group <= 0 || hidden % group ||
        block_rows <= 0 || rows % block_rows || side < 0 || side > 1 || x == out)
        throw std::invalid_argument("dflash2_convolve: invalid geometry or aliased output");
    conv_kernel<<<(rows * hidden + 255) / 256, 256, 0, (cudaStream_t) stream>>>(
        x, dynamic, base, out, rows, hidden, kernel, group, side, block_rows);
}
void dflash2_topk(const float* logits, int32_t* ids, float* unary, int rows, int vocab, int top_k, void* stream) {
    if (rows <= 0 || top_k < 1 || top_k > 64 || top_k > vocab)
        throw std::invalid_argument("dflash2_topk: invalid geometry");
    topk_kernel<<<rows, 1, 0, (cudaStream_t) stream>>>(logits, ids, unary, vocab, top_k);
}
void dflash2_select(const int32_t* ids, const float* unary, const float* hidden,
                    const float* prev, const float* next, const int32_t* anchor,
                    int32_t* out, float* confidence, int rows, int top_k, int rank, void* stream) {
    if (rows < 2 || top_k < 1 || top_k > 64 || rank < 1)
        throw std::invalid_argument("dflash2_select: invalid geometry");
    select_kernel<<<1, 1, 0, (cudaStream_t) stream>>>(
        ids, unary, hidden, prev, next, anchor, out, confidence, rows, top_k, rank);
}
}  // namespace strata::kernels
