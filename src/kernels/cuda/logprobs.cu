#include "strata/kernels/logprobs.hpp"

#include <cuda_runtime.h>

#include <cmath>

namespace strata::kernels {
namespace {

constexpr int kThreads = 128;
constexpr int kPerThread = 32;
constexpr int kSpan = kThreads * kPerThread;
constexpr int kMaxBlocks = 64;

__device__ __forceinline__ void take_best(float& score, int& id, float other, int other_id) {
    if (other > score || (other == score && other_id < id)) {
        score = other;
        id = other_id;
    }
}

__device__ __forceinline__ void warp_best(float& score, int& id) {
    for (int step = 16; step > 0; step >>= 1) {
        const float v = __shfl_down_sync(0xffffffffu, score, step);
        const int i = __shfl_down_sync(0xffffffffu, id, step);
        take_best(score, id, v, i);
    }
    score = __shfl_sync(0xffffffffu, score, 0);
    id = __shfl_sync(0xffffffffu, id, 0);
}

__device__ __forceinline__ float warp_max(float x) {
    for (int step = 16; step > 0; step >>= 1) x = fmaxf(x, __shfl_down_sync(0xffffffffu, x, step));
    return __shfl_sync(0xffffffffu, x, 0);
}

__device__ __forceinline__ double warp_sum(double x) {
    for (int step = 16; step > 0; step >>= 1) x += __shfl_down_sync(0xffffffffu, x, step);
    return __shfl_sync(0xffffffffu, x, 0);
}

// One block owns 4096 logits. It reads them once into registers; the stable
// local log-sum-exp and top-K selection share the same loads. The 61 blocks of
// a 248320-token vocabulary run across the GPU rather than serially on one SM.
__global__ void logprobs_part(const float* __restrict__ logits, int vocab, int blocks, int k,
                              float2* __restrict__ sums, int2* __restrict__ candidates) {
    const int row = blockIdx.y, block = blockIdx.x;
    const int tid = threadIdx.x, base = block * kSpan + tid;
    float values[kPerThread];
    float local_max = -INFINITY;
#pragma unroll
    for (int j = 0; j < kPerThread; ++j) {
        const int id = base + kThreads * j;
        const float v = id < vocab ? logits[(size_t) row * vocab + id] : -INFINITY;
        values[j] = v;
        local_max = fmaxf(local_max, v);
    }
    __shared__ float warp_mx[4];
    __shared__ double warp_sm[4];
    __shared__ float block_mx;
    __shared__ int chosen_id;
    __shared__ float warp_score[4];
    __shared__ int warp_id[4];
    const int warp = tid >> 5, lane = tid & 31;
    const float mx = warp_max(local_max);
    if (lane == 0) warp_mx[warp] = mx;
    __syncthreads();
    if (tid == 0) {
        float m = warp_mx[0];
        for (int w = 1; w < 4; ++w) m = fmaxf(m, warp_mx[w]);
        block_mx = m;
    }
    __syncthreads();
    double local_sum = 0.0;
#pragma unroll
    for (int j = 0; j < kPerThread; ++j)
        if (values[j] != -INFINITY) local_sum += exp((double) values[j] - (double) block_mx);
    const double sm = warp_sum(local_sum);
    if (lane == 0) warp_sm[warp] = sm;
    __syncthreads();
    if (tid == 0) {
        sums[(size_t) row * blocks + block] = make_float2(block_mx,
            (float) (warp_sm[0] + warp_sm[1] + warp_sm[2] + warp_sm[3]));
    }
    __syncthreads();

    for (int rank = 0; rank < k; ++rank) {
        float best = -INFINITY;
        int id = vocab;
#pragma unroll
        for (int j = 0; j < kPerThread; ++j) {
            const int candidate_id = base + kThreads * j;
            if (candidate_id < vocab) take_best(best, id, values[j], candidate_id);
        }
        warp_best(best, id);
        if (lane == 0) { warp_score[warp] = best; warp_id[warp] = id; }
        __syncthreads();
        if (tid == 0) {
            float b = warp_score[0];
            int i = warp_id[0];
            for (int w = 1; w < 4; ++w) take_best(b, i, warp_score[w], warp_id[w]);
            chosen_id = i;
            candidates[((size_t) row * blocks + block) * k + rank] = make_int2(i, __float_as_int(b));
        }
        __syncthreads();
        if (chosen_id >= base && chosen_id < base + kThreads * kPerThread &&
            (chosen_id - base) % kThreads == 0)
            values[(chosen_id - base) / kThreads] = -INFINITY;
        __syncthreads();
    }
}

// One warp combines all block sums and merges their sorted top-K lists. Only
// K+1 pairs per row leave the GPU, even for a 248320-token vocabulary.
__global__ void logprobs_finish(const float* __restrict__ logits, int vocab, int blocks, int k,
                                const int32_t* __restrict__ chosen, const float2* __restrict__ sums,
                                const int2* __restrict__ candidates, LogprobItem* __restrict__ out) {
    const int row = blockIdx.x, lane = threadIdx.x;
    float local_max = -INFINITY;
    for (int b = lane; b < blocks; b += 32)
        local_max = fmaxf(local_max, sums[(size_t) row * blocks + b].x);
    const float mx = warp_max(local_max);
    double local_sum = 0.0;
    for (int b = lane; b < blocks; b += 32) {
        const float2 part = sums[(size_t) row * blocks + b];
        if (part.y > 0.0f) local_sum += (double) part.y * exp((double) part.x - (double) mx);
    }
    const double log_sum = log(warp_sum(local_sum));
    const int picked = chosen[row];
    if (lane == 0) out[(size_t) row * (k + 1)] = {picked, (float) (((double) logits[(size_t) row * vocab + picked] - (double) mx) - log_sum)};

    // Each lane owns up to two sorted block lists. The first K elements of a
    // union must occur in the first K elements of one of its constituent lists.
    int cursor[2] = {0, 0};
    float head_score[2] = {-INFINITY, -INFINITY};
    int head_id[2] = {vocab, vocab};
    for (int j = 0; j < 2; ++j) {
        const int b = lane + 32 * j;
        if (b < blocks && k > 0) {
            const int2 p = candidates[((size_t) row * blocks + b) * k];
            head_id[j] = p.x;
            head_score[j] = __int_as_float(p.y);
        }
    }
    for (int rank = 0; rank < k; ++rank) {
        float score = head_score[0];
        int id = head_id[0];
        take_best(score, id, head_score[1], head_id[1]);
        warp_best(score, id);
        if (lane == 0) out[(size_t) row * (k + 1) + rank + 1] = {id, (float) (((double) score - (double) mx) - log_sum)};
        for (int j = 0; j < 2; ++j) {
            const int b = lane + 32 * j;
            if (b >= blocks || head_id[j] != id) continue;
            if (++cursor[j] < k) {
                const int2 p = candidates[((size_t) row * blocks + b) * k + cursor[j]];
                head_id[j] = p.x;
                head_score[j] = __int_as_float(p.y);
            } else {
                head_id[j] = vocab;
                head_score[j] = -INFINITY;
            }
        }
    }
}

}  // namespace

size_t logprobs_scratch_bytes(int rows, int vocab, int top_k) {
    if (rows < 1 || vocab < 1 || vocab > kSpan * kMaxBlocks || top_k < 0 ||
        top_k > kLogprobsMaxTop || top_k > vocab) return 0;
    const size_t blocks = ((size_t) vocab + kSpan - 1) / kSpan;
    return (size_t) rows * blocks * (sizeof(float2) + (size_t) top_k * sizeof(int2));
}

bool logprobs_extract(const float* logits, int rows, int vocab, const int32_t* chosen,
                      int top_k, void* scratch, LogprobItem* out, void* stream) {
    if (!logits || !chosen || !scratch || !out || !logprobs_scratch_bytes(rows, vocab, top_k)) return false;
    const int blocks = (vocab + kSpan - 1) / kSpan;
    auto* sums = static_cast<float2*>(scratch);
    auto* candidates = reinterpret_cast<int2*>(sums + (size_t) rows * blocks);
    logprobs_part<<<dim3(blocks, rows), kThreads, 0, (cudaStream_t) stream>>>(
        logits, vocab, blocks, top_k, sums, candidates);
    logprobs_finish<<<rows, 32, 0, (cudaStream_t) stream>>>(
        logits, vocab, blocks, top_k, chosen, sums, candidates, out);
    return cudaPeekAtLastError() == cudaSuccess;
}

}  // namespace strata::kernels
