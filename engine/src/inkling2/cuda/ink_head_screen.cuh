// engine/src/inkling2/cuda/ink_head_screen.cuh - INT8-screened exact top-128 of the head, one row (PERF-STEP lane, STRATA_INK_HEAD_SCREEN=1)
//
// MIT License
// Copyright (c) 2026 Niko1221 and the Strata contributors
//
// Plain decode needs, per step, the 128 best (id, softcapped logit) of the BF16 head's 200,058 logits and a finite flag. The full path reads the 883 MiB BF16 unembed
// (0.55 ms) and then runs a three-kernel top-k (0.08 ms). This path reads the 460 MB INT8 screening index of the unembed (the exact Cauchy-Schwarz index that already
// serves the drafting argmax) instead and produces THE SAME 128 keys, bit for bit:
//   1. approx dot a_i = sc_i * sum(q8_ik * h_k) (FP32 h), bounds lb_i = a_i - e_i*|h|, ub_i = a_i + e_i*|h|, with e_i the stored residual norm: |true_i - a_i| <= e_i*|h|.
//      A histogram of lb (65536 ordered-float bins) is built on the way.
//   2. T = the lower edge of the highest bin where the cumulative count reaches 128: at least 128 rows have lb >= T, so the 128th best true value is >= T,
//      and a row with ub < T cannot be in the top 128. (softcap is monotone, ties are decided by id as in the full path.)
//   3. candidates = rows with ub >= T (typically a few hundred); each gets its exact logit with the arithmetic of k_ink_gemv_multi_bf16<1,false> (same lane split,
//      same FMA order, same 16-lane butterfly) and the softcap of k_ink_softcap, so the value is the same float; key = ink_topk_key(value, id).
//   4. the keys are sorted with the top-k's own bitonic sort; the first 128 are the answer. More than 2048 candidates, no bin reaching 128 (NaN) or a failed
//      guard: the flag tells the caller to run the full path instead.
// The finite flag is taken from the screening dots at every 67th row (the full path looks at the logits there): they differ only when a logit overflows.
#pragma once

#include "ink_kernels.cuh"
#include "ink_topk.cuh"

namespace strata::ink {

constexpr int kHsBins = 65536;
constexpr int kHsCap = 2048;

__device__ __forceinline__ uint32_t hs_ordered(float v) {
    const uint32_t b = __float_as_uint(v);
    return b ^ ((b & 0x80000000u) ? 0xFFFFFFFFu : 0x80000000u);
}
__device__ __forceinline__ float hs_from_ordered(uint32_t o) {
    const uint32_t b = o ^ ((o & 0x80000000u) ? 0x80000000u : 0xFFFFFFFFu);
    return __uint_as_float(b);
}

// pass 1: the INT8 dot of every row against the FP32 h (copy of k_ink_screen_pass1_multiM<1> plus the histogram of the lower bounds)
__global__ void __launch_bounds__(256) k_ink_hs_pass1(const int8_t* __restrict__ q8, const float* __restrict__ sc, const float* __restrict__ eps, const float* __restrict__ h,
                                                      const float* __restrict__ h_norm, float* __restrict__ dot_hat, int* __restrict__ hist, int V) {
    const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
    const int row = blockIdx.x * 8 + warp;
    __shared__ alignas(16) float s_h[kInkHidden];
    for (int i = tid; i < kInkHidden / 4; i += 256) reinterpret_cast<float4*>(s_h)[i] = reinterpret_cast<const float4*>(h)[i];
    __syncthreads();
    if (row >= V) return;
    const uint4* q16 = reinterpret_cast<const uint4*>(q8 + (size_t)row * kInkHidden);
    float acc = 0.0f;
    for (int i = lane; i < kInkHidden / 16; i += 32) {
        const uint4 v = q16[i];
        const uint32_t u[4] = {v.x, v.y, v.z, v.w};
#pragma unroll
        for (int q = 0; q < 4; ++q) {
            const uint32_t p = u[q];
            const float4 xv = reinterpret_cast<const float4*>(s_h)[i * 4 + q];
            acc = fmaf((float)(int8_t)(p & 0xFFu), xv.x, acc);
            acc = fmaf((float)(int8_t)((p >> 8) & 0xFFu), xv.y, acc);
            acc = fmaf((float)(int8_t)((p >> 16) & 0xFFu), xv.z, acc);
            acc = fmaf((float)(int8_t)(p >> 24), xv.w, acc);
        }
    }
    const float d = warp_sum(acc) * sc[row];
    if (lane == 0) {
        dot_hat[row] = d;
        const float lb = d - eps[row] * h_norm[0];
        if (isfinite(lb)) atomicAdd(&hist[hs_ordered(lb) >> 16], 1);
    }
}

// the threshold: scan the histogram from the top, T = lower edge of the bin where 128 rows are reached; clears the histogram, resets the counters
__global__ void __launch_bounds__(1024) k_ink_hs_threshold(int* __restrict__ hist, float* __restrict__ d_T, int* __restrict__ cnt, int* __restrict__ flag, int* __restrict__ fin) {
    __shared__ int s_sum[1024];
    __shared__ int s_found;
    const int t = threadIdx.x;
    // thread t owns bins [(1023 - t) * 64, +64): thread 0 the top bins
    const int b0 = (1023 - t) * 64;
    int sum = 0;
    for (int i = 0; i < 64; ++i) sum += hist[b0 + i];
    s_sum[t] = sum;
    if (t == 0) { s_found = -1; }
    __syncthreads();
    // exclusive prefix over t (thread 0 first), serial in one warp's worth of steps: 1024 values by a simple two-level scan
    __shared__ int s_pre[1024];
    if (t == 0) {
        int run = 0;
        for (int i = 0; i < 1024; ++i) { s_pre[i] = run; run += s_sum[i]; if (s_found < 0 && run >= 128) s_found = i; }
        *cnt = 0;
        *fin = 1;
        *flag = (s_found < 0) ? 1 : 0;
    }
    __syncthreads();
    if (t == s_found) {
        int run = s_pre[t];
        for (int i = 63; i >= 0; --i) {
            run += hist[b0 + i];
            if (run >= 128) { *d_T = hs_from_ordered((uint32_t)(b0 + i) << 16); break; }
        }
    }
    for (int i = 0; i < 64; ++i) hist[b0 + i] = 0;
}

// candidates: ub >= T; the finite flag at every 67th row
__global__ void __launch_bounds__(256) k_ink_hs_cand(const float* __restrict__ dot_hat, const float* __restrict__ eps, const float* __restrict__ h_norm, const float* __restrict__ d_T,
                                                     int32_t* __restrict__ cand, int* __restrict__ cnt, int* __restrict__ fin, int V) {
    const int row = blockIdx.x * 256 + threadIdx.x;
    if (row >= V) return;
    const float d = dot_hat[row];
    if (row % 67 == 0 && !isfinite(d)) atomicAnd(fin, 0);
    if (d + eps[row] * h_norm[0] >= d_T[0]) {
        const int pos = atomicAdd(cnt, 1);
        if (pos < kHsCap) cand[pos] = row;
    }
}

// exact logits of the candidates: the arithmetic of k_ink_gemv_multi_bf16<1,false> (8 rows per 128-thread block, 16 lanes of 16 bytes per row, FMA order, butterfly) and k_ink_softcap
__global__ void __launch_bounds__(128) k_ink_hs_exact(const uint16_t* __restrict__ W, const float* __restrict__ h, const int32_t* __restrict__ cand, const int* __restrict__ cnt,
                                                      uint64_t* __restrict__ keys, float cap) {
    const int tid = threadIdx.x, sub = tid >> 4, lg = tid & 15;
    __shared__ alignas(16) float s_x[kInkHidden];
    for (int i = tid; i < kInkHidden / 4; i += 128) reinterpret_cast<float4*>(s_x)[i] = reinterpret_cast<const float4*>(h)[i];
    __syncthreads();
    const int n = min(cnt[0], kHsCap);
    for (int ci = blockIdx.x * 8 + sub; ci < n; ci += gridDim.x * 8) {
        const int row = cand[ci];
        const uint4* wr = reinterpret_cast<const uint4*>(W + (size_t)row * kInkHidden);
        float acc = 0.0f;
        for (int k8 = lg; k8 < kInkHidden / 8; k8 += 16) {
            const uint4 wv = wr[k8];
            const float2 w0 = bf16x2_to_f32x2(wv.x), w1 = bf16x2_to_f32x2(wv.y), w2 = bf16x2_to_f32x2(wv.z), w3 = bf16x2_to_f32x2(wv.w);
            const float4 x0 = reinterpret_cast<const float4*>(s_x)[k8 * 2];
            const float4 x1 = reinterpret_cast<const float4*>(s_x)[k8 * 2 + 1];
            acc = fmaf(w0.x, x0.x, acc); acc = fmaf(w0.y, x0.y, acc);
            acc = fmaf(w1.x, x0.z, acc); acc = fmaf(w1.y, x0.w, acc);
            acc = fmaf(w2.x, x1.x, acc); acc = fmaf(w2.y, x1.y, acc);
            acc = fmaf(w3.x, x1.z, acc); acc = fmaf(w3.y, x1.w, acc);
        }
        float v = acc;
#pragma unroll
        for (int o = 8; o >= 1; o >>= 1) v += __shfl_xor_sync(0xFFFFFFFFu, v, o);
        if (lg == 0) {
            const float val = cap * tanhf(v * (1.0f / cap));
            keys[ci] = ink_topk_key(val, row);
        }
    }
}

// sort the candidate keys, first 128 out; the flag says "run the full path" when there were too many candidates
__global__ void __launch_bounds__(kTopkThreads) k_ink_hs_sort(const uint64_t* __restrict__ keys, const int* __restrict__ cnt, int* __restrict__ flag, uint64_t* __restrict__ out) {
    __shared__ uint64_t s_key[kTopkSortN];
    const int n = cnt[0];
    for (int i = threadIdx.x; i < kTopkSortN; i += kTopkThreads) s_key[i] = (i < n && i < kHsCap) ? keys[i] : 0ull;
    __syncthreads();
    ink_topk_sort2048(s_key);
    for (int i = threadIdx.x; i < kTopkCand; i += kTopkThreads) out[i] = s_key[i];
    if (threadIdx.x == 0 && (n > kHsCap || n < kTopkCand)) *flag = 1;
}

struct HeadScreen {
    float* d_dot = nullptr;       // [V]
    int* d_hist = nullptr;        // [kHsBins], zero between steps
    float* d_T = nullptr;
    float* d_hn = nullptr;        // [1] |h|
    int32_t* d_cand = nullptr;    // [kHsCap]
    uint64_t* d_keys = nullptr;   // [kHsCap]
    int* d_misc = nullptr;        // cnt, flag, fin, scratch
    int* h_misc = nullptr;        // pinned copy of {flag, fin}
    int V = 0;
    bool ready = false;
    long calls = 0, fallbacks = 0, cand_sum = 0;

    bool ensure(int v) {
        if (ready) return V == v;
        V = v;
        if (cudaMalloc(&d_dot, (size_t)v * 4) != cudaSuccess || cudaMalloc(&d_hist, kHsBins * 4) != cudaSuccess || cudaMalloc(&d_T, 4) != cudaSuccess ||
            cudaMalloc(&d_hn, 4) != cudaSuccess || cudaMalloc(&d_cand, kHsCap * 4) != cudaSuccess || cudaMalloc(&d_keys, kHsCap * 8) != cudaSuccess ||
            cudaMalloc(&d_misc, 8 * 4) != cudaSuccess || cudaMallocHost(&h_misc, 8 * 4) != cudaSuccess) {
            (void)cudaGetLastError();
            free_all();
            return false;
        }
        cudaMemset(d_hist, 0, kHsBins * 4);
        cudaMemset(d_misc, 0, 8 * 4);
        ready = true;
        return true;
    }
    void free_all() {
        if (d_dot) cudaFree(d_dot);
        if (d_hist) cudaFree(d_hist);
        if (d_T) cudaFree(d_T);
        if (d_hn) cudaFree(d_hn);
        if (d_cand) cudaFree(d_cand);
        if (d_keys) cudaFree(d_keys);
        if (d_misc) cudaFree(d_misc);
        if (h_misc) cudaFreeHost(h_misc);
        d_dot = nullptr; d_hist = nullptr; d_T = nullptr; d_hn = nullptr; d_cand = nullptr; d_keys = nullptr; d_misc = nullptr; h_misc = nullptr;
        ready = false;
    }
};

}  // namespace strata::ink
