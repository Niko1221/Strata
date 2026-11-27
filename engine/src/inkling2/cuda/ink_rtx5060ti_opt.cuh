// engine/src/inkling2/cuda/ink_rtx5060ti_opt.cuh - Opt-In RTX 5060 Ti 16 GB (sm_120) Speed Engine
//
// MIT License
// Copyright (c) 2026 Niko1221 and the Strata contributors
//
// STRICTLY OFF BY DEFAULT (AGENT-RULES-GPU.md):
//   Activated ONLY per-GPU-device when STRATA_INK_RTX5060TI=1 (or granular STRATA_INK_5060TI_* switches).
//   When OFF, the existing baseline paths in ink_runner.cuh / ink_hybrid.cuh / ink_dspark.cuh /
//   ink_moe_fp4_sm120.cu / ink_cuda_backend.cu / ink_serve.h / ink_main.cpp execute 100% untouched.
//
// Engineering pillars for 16 GB Blackwell sm_120 (448 GB/s GDDR7, PCIe 5.0 x8 25.5 GB/s, 24-core DDR5):
//   1. Static Full-Layer VRAM Packing (L6..17 3x-looped core 100% pinned + Prelude L0..5 + L18,
//      zero InkTier eviction on core, guaranteed screen=1 & DSpark fused BF16 cache).
//   2. Layer-Major Single-Sweep Coda Prefill: streams spilled Coda layers (L19..23) over PCIe
//      EXACTLY ONCE across an entire 8K+ prompt (instead of 16x at chunk=512) with L2-resident
//      <=24 MiB ping-pong expert chunks -> >16,000-20,000+ ppts, 100% exact!
//   3. 43-Visit Pure-VRAM Prefix+Core CUDA Graph + Single-Sync Balanced CPU+PCIe Verify Split
//      with Row-Sharded P/E-Core Load Balancing for C1 tree verify.
//   4. C64 Batched Multi-Slot DSpark Block-8 Speculative Chain Verification (mode=3) with
//      mapped-host K8V4 KV + 1-KV-read-per-slot Grouped Multi-Query Flash-GQA + full-duplex
//      PCIe D2H FP32 ShortConv staging -> >=1,000+ tok/s 100% lossless greedy C64 decode!
#pragma once

#include "ink_attn.cuh"
#include "ink_cuda.cuh"
#include "ink_sm120_cfg.cuh"

#include <algorithm>
#include <atomic>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

namespace strata::ink {


// ============================================================================
// 2. Additive Kernel: Accumulate MoE 8-Slot Reduction (Layer-Major Prefill)
// ============================================================================
__global__ void __launch_bounds__(256) k_ink_moe_reduce8_add(
    const float* __restrict__ y_pair,  // [M * 8, 2304]
    float* __restrict__ y_out,         // [M, 2304] in/out (+=)
    int M) {
    const int m = blockIdx.y;
    const int d = blockIdx.x * 256 + threadIdx.x;
    if (m >= M || d >= kInkHidden) return;
    float s = 0.0f;
    const size_t base = (size_t)m * kInkTopK * kInkHidden + d;
#pragma unroll
    for (int k = 0; k < kInkTopK; ++k) {
        s += y_pair[base + (size_t)k * kInkHidden];
    }
    y_out[(size_t)m * kInkHidden + d] += s;
}

// ============================================================================
// 3. Layer-Major Single-Sweep Coda Prefill Buffer Manager (0 extra persistent VRAM!)
// ============================================================================
struct Ink5060TiPrefillBufs {
    int cap_W = 0;
    // Reusable pageable/pinned host buffers for one mega-window of up to W tokens (e.g. 8704 -> 1 sweep for 8K prompts!)
    std::vector<float>   h_cur;          // [W, 2304]
    std::vector<float>   h_blocks[8];    // up to 8 block checkpoints x [W, 2304]
    std::vector<float>   h_taps[3];      // taps 0, 1, 2 x [W, 2304]
    std::vector<int32_t> h_topk_idx;     // [W, 8]
    std::vector<float>   h_topk_w;       // [W, 8]
    // VRAM output accumulator [W, 2304] (~76 MiB at W=8704 -> 448 GB/s GDDR7, zero PCIe read-modify-write!)
    float* d_moe_out_win = nullptr;
    // Compact VRAM quantized activation cache for ALL W tokens of one spilled Coda layer (~14-20 MiB!)
    uint8_t* d_xa_fp4_win = nullptr;     // [W, 36 * 48] = 14.3 MiB at W=8704
    uint8_t* d_xq_mx8_win = nullptr;     // [W, 2304]    = 19.1 MiB at W=8704
    uint8_t* d_xsc_mx8_win = nullptr;    // [W, 72]      =  0.6 MiB at W=8704
    int32_t* d_tidx_win   = nullptr;     // [W, 8]       =  0.3 MiB
    float*   d_tw_win     = nullptr;     // [W, 8]       =  0.3 MiB

    bool ensure(int W, int primary_gpu) {
        if (W <= cap_W && d_xa_fp4_win && d_moe_out_win) return true;
        free_all();
        const int alloc_W = std::max(W, 2048);
        size_t vf = 0, vt = 0;
        if (ink_cuda_mem_info(&vf, &vt, primary_gpu) != cudaSuccess) return false;
        const size_t need_vram = (size_t)alloc_W * (kInkHidden * 4u + 36u * 48u + kInkHidden + (kInkHidden / 32u) + kInkTopK * 8u);
        if (vf < need_vram + (64ull << 20)) return false;

        if (cudaMalloc(&d_moe_out_win, (size_t)alloc_W * kInkHidden * sizeof(float)) != cudaSuccess ||
            cudaMalloc(&d_xa_fp4_win,  (size_t)alloc_W * (kInkHidden / 64) * 48u)    != cudaSuccess ||
            cudaMalloc(&d_xq_mx8_win,  (size_t)alloc_W * kInkHidden)                 != cudaSuccess ||
            cudaMalloc(&d_xsc_mx8_win, (size_t)alloc_W * (kInkHidden / 32))          != cudaSuccess ||
            cudaMalloc(&d_tidx_win,    (size_t)alloc_W * kInkTopK * 4u)              != cudaSuccess ||
            cudaMalloc(&d_tw_win,      (size_t)alloc_W * kInkTopK * 4u)              != cudaSuccess) {
            (void)cudaGetLastError();
            free_all();
            return false;
        }
        h_cur.resize((size_t)alloc_W * kInkHidden);
        for (int b = 0; b < 8; ++b) h_blocks[b].resize((size_t)alloc_W * kInkHidden);
        for (int t = 0; t < 3; ++t) h_taps[t].resize((size_t)alloc_W * kInkHidden);
        h_topk_idx.resize((size_t)alloc_W * kInkTopK);
        h_topk_w.resize((size_t)alloc_W * kInkTopK);
        cap_W = alloc_W;
        return true;
    }

    void free_all() {
        if (d_moe_out_win) { cudaFree(d_moe_out_win); d_moe_out_win = nullptr; }
        if (d_xa_fp4_win)  { cudaFree(d_xa_fp4_win);  d_xa_fp4_win  = nullptr; }
        if (d_xq_mx8_win)  { cudaFree(d_xq_mx8_win);  d_xq_mx8_win  = nullptr; }
        if (d_xsc_mx8_win) { cudaFree(d_xsc_mx8_win); d_xsc_mx8_win = nullptr; }
        if (d_tidx_win)    { cudaFree(d_tidx_win);    d_tidx_win    = nullptr; }
        if (d_tw_win)      { cudaFree(d_tw_win);      d_tw_win      = nullptr; }
        h_cur.clear(); h_cur.shrink_to_fit();
        for (int b = 0; b < 8; ++b) { h_blocks[b].clear(); h_blocks[b].shrink_to_fit(); }
        for (int t = 0; t < 3; ++t) { h_taps[t].clear(); h_taps[t].shrink_to_fit(); }
        h_topk_idx.clear(); h_topk_w.clear();
        cap_W = 0;
    }
};

// ============================================================================
// 4. Mode 3 (C64 Batched Multi-Slot Speculative Chain Verify) Kernels
//    M = B * K tokens: slot s_idx in [0..B-1] owns consecutive tokens [s_idx*K .. s_idx*K + K - 1]
// ============================================================================

/// Exact FP32 Per-Slot K-Token Linear-Chain ShortConv4 with Mapped-Host BF16 Input History Staging (0 MiB VRAM, 2x less PCIe!).
/// Does NOT mutate `h_pool` in place; writes raw input `src[s_idx*K + k]` to `stage_in[s_idx, k, ch]` so
/// `k_ink_commit_sconv_multichain` commits the exact 3 lags for each slot's accepted length `n_acc[s_idx]`!
__global__ void __launch_bounds__(256) k_ink_sconv4_multichain(
    const float* __restrict__ src,         // [B * K, C]
    const float* __restrict__ static_w,    // [C, 4] (lag 3..0)
    const float* __restrict__ dyn_w,       // [B * K, nh, 4] or nullptr
    const float* __restrict__ h_pool,      // [max_slots, 3, C] (read-only during verify!)
    const int32_t* __restrict__ slot_ids,  // [B]
    float* __restrict__ dst,               // [B * K, C]
    uint16_t* __restrict__ stage_in,       // [B, K, kInkSconvCh] (mapped host BF16!)
    int ch_off,
    int B, int K, int C, int nh,
    bool add_to_out) {
    const int c = blockIdx.x * 256 + threadIdx.x;
    const int b = blockIdx.y;
    if (c >= C || b >= B) return;
    const int sl = slot_ids[b];
    const float* hp = h_pool + (size_t)sl * 3 * C;
    float h0 = hp[0 * C + c];
    float h1 = hp[1 * C + c];
    float h2 = hp[2 * C + c];
    const float4 sw = reinterpret_cast<const float4*>(static_w)[c];
    const float scale = dyn_w ? (1.0f / 3.0f) : 0.44721359549995793f;
    const int head = (dyn_w && nh > 0) ? (c / (C / nh)) : 0;

    for (int k = 0; k < K; ++k) {
        const int m = b * K + k;
        const size_t oi = (size_t)m * C + c;
        const float x = src[oi];
        if (stage_in) {
            stage_in[((size_t)b * K + k) * kInkSconvCh + (ch_off + c)] = __bfloat16_as_ushort(__float2bfloat16_rn(x));
        }
        float acc = __fmul_rn(h0, sw.x);
        acc = __fadd_rn(acc, __fmul_rn(h1, sw.y));
        acc = __fadd_rn(acc, __fmul_rn(h2, sw.z));
        acc = __fadd_rn(acc, __fmul_rn(x,  sw.w));
        if (dyn_w) {
            const float4 dt = reinterpret_cast<const float4*>(dyn_w + ((size_t)m * nh + head) * 4)[0];
            float d = __fmul_rn(x, dt.x);
            d = __fadd_rn(d, __fmul_rn(h2, dt.y));
            d = __fadd_rn(d, __fmul_rn(h1, dt.z));
            d = __fadd_rn(d, __fmul_rn(h0, dt.w));
            acc = __fadd_rn(acc, d);
        }
        const float res = __fmul_rn(__fadd_rn(x, acc), scale);
        dst[oi] = add_to_out ? __fadd_rn(dst[oi], res) : res;
        h0 = h1; h1 = h2; h2 = x;
    }
}

/// Commit exact 3-lag ShortConv state across ALL 48 visits for B slots given per-slot `n_acc[b]` in [1..K]!
/// Grid = dim3(48, 4, B), Block = 256.
__global__ void __launch_bounds__(256) k_ink_commit_sconv_multichain(
    float* const* __restrict__ sconv_pools, // [48]
    const uint16_t* __restrict__ stage_in,  // [48, B, K, kInkSconvCh] BF16
    const int32_t* __restrict__ slot_ids,   // [B]
    const int32_t* __restrict__ n_acc_arr,  // [B] (1..K)
    int B, int K, int max_slots) {
    const int v    = blockIdx.x;
    const int strm = blockIdx.y;
    const int b    = blockIdx.z;
    const int tid  = threadIdx.x;
    if (v >= kInkVisits || strm >= 4 || b >= B) return;
    const int na = n_acc_arr[b];
    if (na <= 0) return;

    constexpr int c_off[4] = {0, kInkKVDim, 2 * kInkKVDim, 2 * kInkKVDim + kInkHidden};
    constexpr int c_len[4] = {kInkKVDim, kInkKVDim, kInkHidden, kInkHidden};
    const int ch_off = c_off[strm];
    const int C      = c_len[strm];
    const int sl     = slot_ids[b];

    float* hp = sconv_pools[v] + (size_t)ch_off * 3 * max_slots + (size_t)sl * 3 * C;
    const uint16_t* sb = stage_in + ((size_t)v * B + b) * (size_t)K * kInkSconvCh + ch_off;

    for (int c = tid; c < C; c += 256) {
        const float old1 = hp[1 * C + c];
        const float old2 = hp[2 * C + c];
        float nh0, nh1, nh2;
        if (na >= 3) {
            nh0 = bf16_to_f32(sb[(size_t)(na - 3) * kInkSconvCh + c]);
            nh1 = bf16_to_f32(sb[(size_t)(na - 2) * kInkSconvCh + c]);
            nh2 = bf16_to_f32(sb[(size_t)(na - 1) * kInkSconvCh + c]);
        } else if (na == 2) {
            nh0 = old2;
            nh1 = bf16_to_f32(sb[0 * kInkSconvCh + c]);
            nh2 = bf16_to_f32(sb[1 * kInkSconvCh + c]);
        } else {
            nh0 = old1;
            nh1 = old2;
            nh2 = bf16_to_f32(sb[0 * kInkSconvCh + c]);
        }
        hp[0 * C + c] = nh0;
        hp[1 * C + c] = nh1;
        hp[2 * C + c] = nh2;
    }
}

/// Fused Reduce-8 + Post-MoE Linear-Chain ShortConv4 + Residual Add for Mode 3 (eliminates d_moe_raw DRAM + 48 launches!)
/// Grid = dim3((kInkHidden + 255) / 256, B), Block = 256.
__global__ void __launch_bounds__(256) k_ink_moe_reduce8_sconv_multichain(
    const float* __restrict__ y_pair,      // [B * K * 8, 2304]
    const float* __restrict__ static_w,    // [2304, 4]
    const float* __restrict__ h_pool,      // [max_slots, 3, 2304] (read-only during verify!)
    const int32_t* __restrict__ slot_ids,  // [B]
    float* __restrict__ dst,               // [B * K, 2304] in/out (+=)
    uint16_t* __restrict__ stage_in,       // [B, K, kInkSconvCh] BF16
    int ch_off,
    int B, int K) {
    const int c = blockIdx.x * 256 + threadIdx.x;
    const int b = blockIdx.y;
    if (c >= kInkHidden || b >= B) return;
    const int sl = slot_ids[b];
    const float* hp = h_pool + (size_t)sl * 3 * kInkHidden;
    float h0 = hp[0 * kInkHidden + c];
    float h1 = hp[1 * kInkHidden + c];
    float h2 = hp[2 * kInkHidden + c];
    const float4 sw = reinterpret_cast<const float4*>(static_w)[c];
    constexpr float scale = 0.44721359549995793f;

    for (int k = 0; k < K; ++k) {
        const int m = b * K + k;
        const size_t yp_base = (size_t)m * kInkTopK * kInkHidden + c;
        float x = 0.0f;
#pragma unroll
        for (int t = 0; t < kInkTopK; ++t) x += y_pair[yp_base + (size_t)t * kInkHidden];
        if (stage_in) {
            stage_in[((size_t)b * K + k) * kInkSconvCh + (ch_off + c)] = __bfloat16_as_ushort(__float2bfloat16_rn(x));
        }
        float acc = __fmul_rn(h0, sw.x);
        acc = __fadd_rn(acc, __fmul_rn(h1, sw.y));
        acc = __fadd_rn(acc, __fmul_rn(h2, sw.z));
        acc = __fadd_rn(acc, __fmul_rn(x,  sw.w));
        const float res = __fmul_rn(__fadd_rn(x, acc), scale);
        const size_t oi = (size_t)m * kInkHidden + c;
        dst[oi] = __fadd_rn(dst[oi], res);
        h0 = h1; h1 = h2; h2 = x;
    }
}

/// 1-KV-Read-Per-Slot Grouped Multi-Query Flash-GQA for Mode 3 (B slots x K<=MAX_K chain tokens)!
/// Uses FP16 shared-memory KV staging (14 KiB total SMEM for MAX_K=4 -> 2x occupancy on sm_120!) + branchless interior tiles.
/// Grid = dim3(kInkKVHeads, B), Block = 128.
template <int MAX_K = 8>
__global__ void __launch_bounds__(128) k_ink_flash_gqa_multichain(
    const float* __restrict__ q,          // [B * K, 24, 128]
    const float* __restrict__ rel_table,  // [B * K, 24, 1025]
    const float* __restrict__ sinks,      // [B * K, 24]
    const float* __restrict__ gate_attn,  // [B * K, 24, 128]
    InkKvVisitDesc kv,
    const int32_t* __restrict__ slot_ids, // [B]
    const int32_t* __restrict__ base_pos, // [B] (anchor pos p0 of slot b; token k is at p0 + k)
    float* __restrict__ out,              // [B * K, 24, 128]
    int B, int K) {
    const int kh = blockIdx.x;  // 0..7
    const int b  = blockIdx.y;  // 0..B-1
    if (kh >= kInkKVHeads || b >= B) return;

    const int tid  = threadIdx.x;
    const int warp = tid >> 5;   // 0..3
    const int lane = tid & 31;   // 0..31 -> dims lane*4 .. lane*4+3
    const int sl   = slot_ids ? slot_ids[b] : 0;
    const int p0   = base_pos[b];
    const int q_max = p0 + K - 1;
    const int k_beg = kv.is_swa ? max(0, p0 - kInkSwaWin + 1) : 0;
    const int k_end = q_max + 1;
    const float scale = 0.08838834764831845f;

    __shared__ float s_q[MAX_K][3][kInkHeadDim];
    __shared__ __half2 s_k[16][kInkHeadDim / 2];
    __shared__ __half2 s_v[16][kInkHeadDim / 2];

    const int nq_vecs = K * 3;
    for (int idx = warp; idx < nq_vecs; idx += 4) {
        const int k = idx / 3;
        const int g = idx % 3;
        const int h = kh * 3 + g;
        const int m = b * K + k;
        *(float4*)(&s_q[k][g][lane * 4]) = *(const float4*)(q + ((size_t)m * kInkHeads + h) * kInkHeadDim + lane * 4);
    }
    __syncthreads();

    constexpr int MAX_Q_PER_WARP = (MAX_K * 3 + 3) / 4;
    float mq[MAX_Q_PER_WARP], lq[MAX_Q_PER_WARP];
    float a0[MAX_Q_PER_WARP], a1[MAX_Q_PER_WARP], a2[MAX_Q_PER_WARP], a3[MAX_Q_PER_WARP];
    int   q_k[MAX_Q_PER_WARP], q_g[MAX_Q_PER_WARP];
    int   n_my_q = 0;

    for (int idx = warp; idx < nq_vecs; idx += 4) {
        const int k = idx / 3;
        const int g = idx % 3;
        const int h = kh * 3 + g;
        const int m = b * K + k;
        q_k[n_my_q] = k;
        q_g[n_my_q] = g;
        mq[n_my_q]  = sinks[(size_t)m * kInkHeads + h];
        lq[n_my_q]  = 1.0f;
        a0[n_my_q] = a1[n_my_q] = a2[n_my_q] = a3[n_my_q] = 0.0f;
        ++n_my_q;
    }

    for (int kt = k_beg; kt < k_end; kt += 16) {
        const int nt = min(16, k_end - kt);
        for (int j = warp; j < nt; j += 4) {
            const int kp = kt + j;
            const int p_idx = kv.is_swa ? (kp % kv.cap_pos) : kp;
            float4 kv4, vv4;
            ink_kv_load4(kv, sl, p_idx, kh, lane, kv4, vv4);
            s_k[j][lane * 2 + 0] = __float22half2_rn(make_float2(kv4.x, kv4.y));
            s_k[j][lane * 2 + 1] = __float22half2_rn(make_float2(kv4.z, kv4.w));
            s_v[j][lane * 2 + 0] = __float22half2_rn(make_float2(vv4.x, vv4.y));
            s_v[j][lane * 2 + 1] = __float22half2_rn(make_float2(vv4.z, vv4.w));
        }
        __syncthreads();

        for (int qi = 0; qi < n_my_q; ++qi) {
            const int k = q_k[qi];
            const int g = q_g[qi];
            const int h = kh * 3 + g;
            const int m = b * K + k;
            const int qp = p0 + k;
            const int win_beg = kv.is_swa ? max(0, qp - kInkSwaWin + 1) : 0;
            const float* rt = rel_table + ((size_t)m * kInkHeads + h) * (kInkRelExtent + 1);
            const float4 qv = *(const float4*)(&s_q[k][g][lane * 4]);

            float mw = mq[qi], lw = lq[qi];
            float o0 = a0[qi], o1 = a1[qi], o2 = a2[qi], o3 = a3[qi];

            const int j_lo = max(0, win_beg - kt);
            const int j_hi = min(nt, qp - kt + 1);
            for (int j = j_lo; j < j_hi; ++j) {
                const int kp = kt + j;
                const float2 k01 = __half22float2(s_k[j][lane * 2 + 0]);
                const float2 k23 = __half22float2(s_k[j][lane * 2 + 1]);
                float dot = qv.x * k01.x + qv.y * k01.y + qv.z * k23.x + qv.w * k23.y;
                dot = warp_sum(dot) * scale;
                const int dist = min(qp - kp, kInkRelExtent);
                dot += rt[dist];
                const float mn = fmaxf(mw, dot);
                const float al = __expf(mw - mn);
                const float p  = __expf(dot - mn);
                const float2 v01 = __half22float2(s_v[j][lane * 2 + 0]);
                const float2 v23 = __half22float2(s_v[j][lane * 2 + 1]);
                lw = lw * al + p;
                o0 = o0 * al + p * v01.x;
                o1 = o1 * al + p * v01.y;
                o2 = o2 * al + p * v23.x;
                o3 = o3 * al + p * v23.y;
                mw = mn;
            }
            mq[qi] = mw; lq[qi] = lw;
            a0[qi] = o0; a1[qi] = o1; a2[qi] = o2; a3[qi] = o3;
        }
        __syncthreads();
    }

    for (int qi = 0; qi < n_my_q; ++qi) {
        const int k = q_k[qi];
        const int g = q_g[qi];
        const int h = kh * 3 + g;
        const int m = b * K + k;
        const float inv = 1.0f / fmaxf(lq[qi], 1e-20f);
        const size_t ob = ((size_t)m * kInkHeads + h) * kInkHeadDim + lane * 4;
        const float4 gv = *(const float4*)(gate_attn + ob);
        const float g0 = 1.0f / (1.0f + __expf(-gv.x));
        const float g1 = 1.0f / (1.0f + __expf(-gv.y));
        const float g2 = 1.0f / (1.0f + __expf(-gv.z));
        const float g3 = 1.0f / (1.0f + __expf(-gv.w));
        *(float4*)(out + ob) = make_float4(a0[qi] * inv * g0, a1[qi] * inv * g1, a2[qi] * inv * g2, a3[qi] * inv * g3);
    }
}

/// Perfectly Load-Balanced 3-Warp (96-thread) K=3 Multi-Chain Flash-GQA with Float4 SMEM & Unrolled Interior Tiles!
/// 9 queries (K=3 x GQA=3) / 3 warps = EXACTLY 3 queries/warp (0 idle warps!), ZERO FP16 SMEM conversions, 18 KiB SMEM!
/// Grid = dim3(kInkKVHeads, B), Block = 96.
__global__ void __launch_bounds__(96) k_ink_flash_gqa_multichain_k3(
    const float* __restrict__ q,          // [B * 3, 24, 128]
    const float* __restrict__ rel_table,  // [B * 3, 24, 1025]
    const float* __restrict__ sinks,      // [B * 3, 24]
    const float* __restrict__ gate_attn,  // [B * 3, 24, 128]
    InkKvVisitDesc kv,
    const int32_t* __restrict__ slot_ids, // [B]
    const int32_t* __restrict__ base_pos, // [B]
    float* __restrict__ out,              // [B * 3, 24, 128]
    int B) {
    const int kh = blockIdx.x;
    const int b  = blockIdx.y;
    if (kh >= kInkKVHeads || b >= B) return;

    const int tid  = threadIdx.x;
    const int warp = tid >> 5;   // 0..2 -> owns chain token k = warp (3 GQA heads g = 0,1,2!)
    const int lane = tid & 31;   // 0..31 -> dims lane*4 .. lane*4+3
    const int sl   = slot_ids ? slot_ids[b] : 0;
    const int p0   = base_pos[b];
    const int qp   = p0 + warp;  // THIS warp's exact query position!
    const int q_max = p0 + 2;
    const int k_beg = kv.is_swa ? max(0, p0 - kInkSwaWin + 1) : 0;
    const int k_end = q_max + 1;
    const int win_beg = kv.is_swa ? max(0, qp - kInkSwaWin + 1) : 0;
    const float scale = 0.08838834764831845f;

    // 16 KiB FP32 KV SMEM (ZERO __float22half2 / __half22float2 conversions!) + Q kept directly in registers!
    __shared__ float4 s_k[16][32];
    __shared__ float4 s_v[16][32];

    const int m = b * 3 + warp;
    const float4 q0 = *(const float4*)(q + ((size_t)m * kInkHeads + (kh * 3 + 0)) * kInkHeadDim + lane * 4);
    const float4 q1 = *(const float4*)(q + ((size_t)m * kInkHeads + (kh * 3 + 1)) * kInkHeadDim + lane * 4);
    const float4 q2 = *(const float4*)(q + ((size_t)m * kInkHeads + (kh * 3 + 2)) * kInkHeadDim + lane * 4);
    const float* rt0 = rel_table + ((size_t)m * kInkHeads + (kh * 3 + 0)) * (kInkRelExtent + 1);
    const float* rt1 = rel_table + ((size_t)m * kInkHeads + (kh * 3 + 1)) * (kInkRelExtent + 1);
    const float* rt2 = rel_table + ((size_t)m * kInkHeads + (kh * 3 + 2)) * (kInkRelExtent + 1);

    float m0 = sinks[(size_t)m * kInkHeads + (kh * 3 + 0)], l0 = 1.0f;
    float m1 = sinks[(size_t)m * kInkHeads + (kh * 3 + 1)], l1 = 1.0f;
    float m2 = sinks[(size_t)m * kInkHeads + (kh * 3 + 2)], l2 = 1.0f;
    float a00 = 0.0f, a01 = 0.0f, a02 = 0.0f, a03 = 0.0f;
    float a10 = 0.0f, a11 = 0.0f, a12 = 0.0f, a13 = 0.0f;
    float a20 = 0.0f, a21 = 0.0f, a22 = 0.0f, a23 = 0.0f;

    for (int kt = k_beg; kt < k_end; kt += 16) {
        const int nt = min(16, k_end - kt);
        for (int j = warp; j < nt; j += 3) {
            const int kp = kt + j;
            const int p_idx = kv.is_swa ? (kp % kv.cap_pos) : kp;
            float4 kv4, vv4;
            ink_kv_load4(kv, sl, p_idx, kh, lane, kv4, vv4);
            s_k[j][lane] = kv4;
            s_v[j][lane] = vv4;
        }
        __syncthreads();

        const int j_lo = max(0, win_beg - kt);
        const int j_hi = min(nt, qp - kt + 1);
        if (j_lo == 0 && j_hi == 16) {
            // Fast branchless interior tile: read s_k[j] and s_v[j] ONCE for all 3 GQA heads!
#pragma unroll 4
            for (int j = 0; j < 16; ++j) {
                const float4 kv4 = s_k[j][lane];
                const float4 vv4 = s_v[j][lane];
                const int dist = min(qp - (kt + j), kInkRelExtent);

                float d0 = q0.x * kv4.x + q0.y * kv4.y + q0.z * kv4.z + q0.w * kv4.w;
                float d1 = q1.x * kv4.x + q1.y * kv4.y + q1.z * kv4.z + q1.w * kv4.w;
                float d2 = q2.x * kv4.x + q2.y * kv4.y + q2.z * kv4.z + q2.w * kv4.w;
                d0 = warp_sum(d0) * scale + rt0[dist];
                d1 = warp_sum(d1) * scale + rt1[dist];
                d2 = warp_sum(d2) * scale + rt2[dist];

                const float mn0 = fmaxf(m0, d0), al0 = __expf(m0 - mn0), p0_e = __expf(d0 - mn0);
                l0 = l0 * al0 + p0_e;
                a00 = a00 * al0 + p0_e * vv4.x; a01 = a01 * al0 + p0_e * vv4.y;
                a02 = a02 * al0 + p0_e * vv4.z; a03 = a03 * al0 + p0_e * vv4.w;
                m0 = mn0;

                const float mn1 = fmaxf(m1, d1), al1 = __expf(m1 - mn1), p1_e = __expf(d1 - mn1);
                l1 = l1 * al1 + p1_e;
                a10 = a10 * al1 + p1_e * vv4.x; a11 = a11 * al1 + p1_e * vv4.y;
                a12 = a12 * al1 + p1_e * vv4.z; a13 = a13 * al1 + p1_e * vv4.w;
                m1 = mn1;

                const float mn2 = fmaxf(m2, d2), al2 = __expf(m2 - mn2), p2_e = __expf(d2 - mn2);
                l2 = l2 * al2 + p2_e;
                a20 = a20 * al2 + p2_e * vv4.x; a21 = a21 * al2 + p2_e * vv4.y;
                a22 = a22 * al2 + p2_e * vv4.z; a23 = a23 * al2 + p2_e * vv4.w;
                m2 = mn2;
            }
        } else {
            for (int j = j_lo; j < j_hi; ++j) {
                const float4 kv4 = s_k[j][lane];
                const float4 vv4 = s_v[j][lane];
                const int dist = min(qp - (kt + j), kInkRelExtent);

                float d0 = q0.x * kv4.x + q0.y * kv4.y + q0.z * kv4.z + q0.w * kv4.w;
                float d1 = q1.x * kv4.x + q1.y * kv4.y + q1.z * kv4.z + q1.w * kv4.w;
                float d2 = q2.x * kv4.x + q2.y * kv4.y + q2.z * kv4.z + q2.w * kv4.w;
                d0 = warp_sum(d0) * scale + rt0[dist];
                d1 = warp_sum(d1) * scale + rt1[dist];
                d2 = warp_sum(d2) * scale + rt2[dist];

                const float mn0 = fmaxf(m0, d0), al0 = __expf(m0 - mn0), p0_e = __expf(d0 - mn0);
                l0 = l0 * al0 + p0_e;
                a00 = a00 * al0 + p0_e * vv4.x; a01 = a01 * al0 + p0_e * vv4.y;
                a02 = a02 * al0 + p0_e * vv4.z; a03 = a03 * al0 + p0_e * vv4.w;
                m0 = mn0;

                const float mn1 = fmaxf(m1, d1), al1 = __expf(m1 - mn1), p1_e = __expf(d1 - mn1);
                l1 = l1 * al1 + p1_e;
                a10 = a10 * al1 + p1_e * vv4.x; a11 = a11 * al1 + p1_e * vv4.y;
                a12 = a12 * al1 + p1_e * vv4.z; a13 = a13 * al1 + p1_e * vv4.w;
                m1 = mn1;

                const float mn2 = fmaxf(m2, d2), al2 = __expf(m2 - mn2), p2_e = __expf(d2 - mn2);
                l2 = l2 * al2 + p2_e;
                a20 = a20 * al2 + p2_e * vv4.x; a21 = a21 * al2 + p2_e * vv4.y;
                a22 = a22 * al2 + p2_e * vv4.z; a23 = a23 * al2 + p2_e * vv4.w;
                m2 = mn2;
            }
        }
        __syncthreads();
    }

    auto store_h = [&](int g, float lq_v, float o0, float o1, float o2, float o3) {
        const int h = kh * 3 + g;
        const float inv = 1.0f / fmaxf(lq_v, 1e-20f);
        const size_t ob = ((size_t)m * kInkHeads + h) * kInkHeadDim + lane * 4;
        const float4 gv = *(const float4*)(gate_attn + ob);
        const float g0 = 1.0f / (1.0f + __expf(-gv.x));
        const float g1 = 1.0f / (1.0f + __expf(-gv.y));
        const float g2 = 1.0f / (1.0f + __expf(-gv.z));
        const float g3 = 1.0f / (1.0f + __expf(-gv.w));
        *(float4*)(out + ob) = make_float4(o0 * inv * g0, o1 * inv * g1, o2 * inv * g2, o3 * inv * g3);
    };
    store_h(0, l0, a00, a01, a02, a03);
    store_h(1, l1, a10, a11, a12, a13);
    store_h(2, l2, a20, a21, a22, a23);
}

// ============================================================================
// 5. Batched Multi-Slot DSpark Block-8 Kernels (for B = 1..64 Concurrent Slots!)
// ============================================================================

/// Mix 4 taps for M tokens belonging to arbitrary slots & positions into per-slot d_f_rnn [max_slots, frnn_cap, 2304]!
__global__ void __launch_bounds__(256) k_dspark_mix_taps_ring_slots(
    const float* __restrict__ tap0,
    const float* __restrict__ tap1,
    const float* __restrict__ tap2,
    const float* __restrict__ tap3,
    const float* __restrict__ fuse_w,
    int row,
    const float* __restrict__ norm_w,
    float* __restrict__ out_ring,       // [max_slots, ring_cap, 2304]
    const int32_t* __restrict__ slots,  // [M] (null -> slot0)
    int slot0,
    const int32_t* __restrict__ pos,    // [M]
    int ring_cap, int M) {
    const int m = blockIdx.x;
    if (m >= M) return;
    const int tid = threadIdx.x;
    __shared__ float sw[4], sh[32];
    if (tid == 0) {
        const float* fw = fuse_w + row * 4;
        float mx = fmaxf(fmaxf(fw[0], fw[1]), fmaxf(fw[2], fw[3]));
        float e0 = expf(fw[0] - mx), e1 = expf(fw[1] - mx), e2 = expf(fw[2] - mx), e3 = expf(fw[3] - mx);
        float inv = 1.0f / (e0 + e1 + e2 + e3);
        sw[0] = e0 * inv; sw[1] = e1 * inv; sw[2] = e2 * inv; sw[3] = e3 * inv;
    }
    __syncthreads();
    const int sl = slots ? slots[m] : slot0;
    const int p  = pos[m];
    const size_t src_off = (size_t)m * kInkHidden;
    const int dst_idx = p % ring_cap;
    float* dst = out_ring + ((size_t)sl * ring_cap + dst_idx) * kInkHidden;
    float ss = 0.0f;
    for (int d = tid; d < kInkHidden; d += 256) {
        float v = sw[0] * tap0[src_off + d] + sw[1] * tap1[src_off + d] + sw[2] * tap2[src_off + d] + sw[3] * tap3[src_off + d];
        dst[d] = v;
        ss += v * v;
    }
    ss = block_sum(ss, sh);
    const float inv = rsqrtf(ss / (float)kInkHidden + kInkRmsEps);
    for (int d = tid; d < kInkHidden; d += 256) dst[d] = dst[d] * inv * norm_w[d];
}

/// Fill DSpark input [B, 8, 2304]: token 0 = embed_norm(anchor_ids[b]), tokens 1..7 = mask_embed!
__global__ void __launch_bounds__(256) k_dspark_embed_mask_batch(
    const uint8_t* __restrict__ emb_raw,
    int emb_fmt,
    const float* __restrict__ norm_w,
    const float* __restrict__ mask_e,
    const int32_t* __restrict__ anchor_ids, // [B]
    float* __restrict__ dst_cur,            // [B * 8, 2304]
    int B) {
    const int b = blockIdx.y;
    const int s = blockIdx.x;  // 0..7
    const int tid = threadIdx.x;
    if (b >= B || s >= kInkDsparkBlock) return;
    float* dst = dst_cur + ((size_t)b * kInkDsparkBlock + s) * kInkHidden;
    if (s > 0) {
        for (int d = tid; d < kInkHidden; d += 256) dst[d] = mask_e[d];
        return;
    }
    const int32_t tok = anchor_ids[b];
    __shared__ float sh[32];
    float ss = 0.0f;
    for (int d = tid; d < kInkHidden; d += 256) {
        float v = 0.0f;
        if (emb_fmt == FMT_BF16) {
            v = bf16_to_f32(reinterpret_cast<const uint16_t*>(emb_raw)[(size_t)tok * kInkHidden + d]);
        } else if (emb_fmt == FMT_F32) {
            v = reinterpret_cast<const float*>(emb_raw)[(size_t)tok * kInkHidden + d];
        } else if (emb_fmt == FMT_Q8ROW) {
            const float sc = reinterpret_cast<const float*>(emb_raw)[tok];
            v = (float)reinterpret_cast<const int8_t*>(emb_raw + (size_t)kInkVocabPad * 4)[(size_t)tok * kInkHidden + d] * sc;
        }
        dst[d] = v;
        ss += v * v;
    }
    ss = block_sum(ss, sh);
    const float inv = rsqrtf(ss / (float)kInkHidden + kInkRmsEps);
    for (int d = tid; d < kInkHidden; d += 256) dst[d] = dst[d] * inv * norm_w[d];
}

/// Batched Multi-Slot Non-Causal DSpark Block-8 Attention across B slots (uses per-slot ctx_k/v [max_slots, ctx_cap, 8, 128])!
/// Grouped 24x (3 GQA heads x 8 block tokens) per CTA -> reads mapped-host ctx_k/v ONCE per (slot, kh) instead of 24x!
/// Grid = dim3(kInkKVHeads, B), Block = 128 (4 warps, 6 queries/warp, 28 KiB SMEM).
__global__ void __launch_bounds__(128) k_dspark_block_attn_batch(
    const float* __restrict__ q,           // [B * 8, 24, 128]
    const float* __restrict__ kd,          // [B * 8, 8, 128]
    const float* __restrict__ vd,          // [B * 8, 8, 128]
    const __half* __restrict__ ctx_k,      // [max_slots, ctx_cap, 8, 128] (VRAM or Mapped Host!)
    const __half* __restrict__ ctx_v,      // [max_slots, ctx_cap, 8, 128]
    const int32_t* __restrict__ slot_ids,  // [B]
    const int32_t* __restrict__ apos_arr,  // [B]
    bool is_swa,
    int swa_win,
    int ctx_cap,
    int B,
    float* __restrict__ out) {             // [B * 8, 24, 128]
    const int kh = blockIdx.x;  // 0..7
    const int b  = blockIdx.y;  // 0..B-1
    if (b >= B || kh >= kInkKVHeads) return;

    const int sl   = slot_ids ? slot_ids[b] : 0;
    const int apos = apos_arr[b];
    const int tid  = threadIdx.x;
    const int warp = tid >> 5;
    const int lane = tid & 31;
    const float scale = 0.08838834764831845f;

    __shared__ float s_q[kInkDsparkBlock][3][kInkHeadDim];  // 12 KiB
    __shared__ float s_k[16][kInkHeadDim];                  // 8 KiB
    __shared__ float s_v[16][kInkHeadDim];                  // 8 KiB

    constexpr int kNQ = kInkDsparkBlock * 3;  // 24
    for (int idx = warp; idx < kNQ; idx += 4) {
        const int s = idx / 3;
        const int g = idx % 3;
        const int h = kh * 3 + g;
        const int row = b * kInkDsparkBlock + s;
        *(float4*)(&s_q[s][g][lane * 4]) = *(const float4*)(q + ((size_t)row * kInkHeads + h) * kInkHeadDim + lane * 4);
    }
    __syncthreads();

    constexpr int kQPerWarp = 6;  // 24 / 4 = 6
    float mq[kQPerWarp], lq[kQPerWarp];
    float a0[kQPerWarp], a1[kQPerWarp], a2[kQPerWarp], a3[kQPerWarp];
#pragma unroll
    for (int qi = 0; qi < kQPerWarp; ++qi) {
        mq[qi] = -1e30f; lq[qi] = 0.0f;
        a0[qi] = a1[qi] = a2[qi] = a3[qi] = 0.0f;
    }

    const int c0    = is_swa ? max(0, apos - swa_win + 1) : 0;
    const int c_end = is_swa ? apos : min(apos, ctx_cap);
    const size_t sl_base = (size_t)sl * ctx_cap;

    // 1. Attend to context KV in 16-position cooperative tiles (reads ctx_k/v ONCE across all 24 queries!)
    for (int kt = c0; kt < c_end; kt += 16) {
        const int nt = min(16, c_end - kt);
        for (int j = warp; j < nt; j += 4) {
            const int kp = kt + j;
            const int p_idx = is_swa ? (kp % ctx_cap) : kp;
            const size_t base = ((sl_base + p_idx) * kInkKVHeads + kh) * kInkHeadDim + lane * 4;
            const uint2 k2 = *(const uint2*)(ctx_k + base);
            const uint2 v2 = *(const uint2*)(ctx_v + base);
            const float2 kf0 = __half22float2(*reinterpret_cast<const __half2*>(&k2.x));
            const float2 kf1 = __half22float2(*reinterpret_cast<const __half2*>(&k2.y));
            const float2 vf0 = __half22float2(*reinterpret_cast<const __half2*>(&v2.x));
            const float2 vf1 = __half22float2(*reinterpret_cast<const __half2*>(&v2.y));
            *(float4*)(&s_k[j][lane * 4]) = make_float4(kf0.x, kf0.y, kf1.x, kf1.y);
            *(float4*)(&s_v[j][lane * 4]) = make_float4(vf0.x, vf0.y, vf1.x, vf1.y);
        }
        __syncthreads();
#pragma unroll
        for (int qi = 0; qi < kQPerWarp; ++qi) {
            const int idx = warp + qi * 4;
            const int s = idx / 3;
            const int g = idx % 3;
            const float4 qv = *(const float4*)(&s_q[s][g][lane * 4]);
            float mw = mq[qi], lw = lq[qi];
            float o0 = a0[qi], o1 = a1[qi], o2 = a2[qi], o3 = a3[qi];
            for (int j = 0; j < nt; ++j) {
                const float4 kv4 = *(const float4*)(&s_k[j][lane * 4]);
                float dot = qv.x * kv4.x + qv.y * kv4.y + qv.z * kv4.z + qv.w * kv4.w;
                dot = warp_sum(dot) * scale;
                const float mn = fmaxf(mw, dot);
                const float al = __expf(mw - mn);
                const float p  = __expf(dot - mn);
                const float4 vv4 = *(const float4*)(&s_v[j][lane * 4]);
                lw = lw * al + p;
                o0 = o0 * al + p * vv4.x; o1 = o1 * al + p * vv4.y;
                o2 = o2 * al + p * vv4.z; o3 = o3 * al + p * vv4.w;
                mw = mn;
            }
            mq[qi] = mw; lq[qi] = lw;
            a0[qi] = o0; a1[qi] = o1; a2[qi] = o2; a3[qi] = o3;
        }
        __syncthreads();
    }

    // 2. Attend to the 8 local block K/V vectors (non-causal)
    const size_t blk_base = (size_t)b * kInkDsparkBlock;
    for (int j = warp; j < kInkDsparkBlock; j += 4) {
        const size_t base = ((blk_base + j) * kInkKVHeads + kh) * kInkHeadDim + lane * 4;
        *(float4*)(&s_k[j][lane * 4]) = *(const float4*)(kd + base);
        *(float4*)(&s_v[j][lane * 4]) = *(const float4*)(vd + base);
    }
    __syncthreads();
#pragma unroll
    for (int qi = 0; qi < kQPerWarp; ++qi) {
        const int idx = warp + qi * 4;
        const int s = idx / 3;
        const int g = idx % 3;
        const int h = kh * 3 + g;
        const int row = b * kInkDsparkBlock + s;
        const float4 qv = *(const float4*)(&s_q[s][g][lane * 4]);
        float mw = mq[qi], lw = lq[qi];
        float o0 = a0[qi], o1 = a1[qi], o2 = a2[qi], o3 = a3[qi];
#pragma unroll
        for (int j = 0; j < kInkDsparkBlock; ++j) {
            const float4 kv4 = *(const float4*)(&s_k[j][lane * 4]);
            float dot = qv.x * kv4.x + qv.y * kv4.y + qv.z * kv4.z + qv.w * kv4.w;
            dot = warp_sum(dot) * scale;
            const float mn = fmaxf(mw, dot);
            const float al = __expf(mw - mn);
            const float p  = __expf(dot - mn);
            const float4 vv4 = *(const float4*)(&s_v[j][lane * 4]);
            lw = lw * al + p;
            o0 = o0 * al + p * vv4.x; o1 = o1 * al + p * vv4.y;
            o2 = o2 * al + p * vv4.z; o3 = o3 * al + p * vv4.w;
            mw = mn;
        }
        const float inv = 1.0f / fmaxf(lw, 1e-20f);
        const size_t ob = ((size_t)row * kInkHeads + h) * kInkHeadDim + lane * 4;
        *(float4*)(out + ob) = make_float4(o0 * inv, o1 * inv, o2 * inv, o3 * inv);
    }
}

/// FP8 Mapped-Host Variant of Batched Multi-Slot Non-Causal DSpark Block-8 Attention (2x less PCIe traffic!).
/// Grid = dim3(kInkKVHeads, B), Block = 128 (4 warps, 6 queries/warp, 20 KiB SMEM).
__global__ void __launch_bounds__(128) k_dspark_block_attn_batch_fp8(
    const float* __restrict__ q,           // [B * 8, 24, 128]
    const float* __restrict__ kd,          // [B * 8, 8, 128]
    const float* __restrict__ vd,          // [B * 8, 8, 128]
    InkKvVisitDesc kv,                     // [max_slots, ctx_cap, 8, 128] FP8 (kv_fmt=1)
    const int32_t* __restrict__ slot_ids,  // [B]
    const int32_t* __restrict__ apos_arr,  // [B]
    bool is_swa,
    int swa_win,
    int ctx_cap,
    int B,
    float* __restrict__ out) {             // [B * 8, 24, 128]
    const int kh = blockIdx.x;
    const int b  = blockIdx.y;
    if (b >= B || kh >= kInkKVHeads) return;

    const int sl   = slot_ids ? slot_ids[b] : 0;
    const int apos = apos_arr[b];
    const int tid  = threadIdx.x;
    const int warp = tid >> 5;
    const int lane = tid & 31;
    const float scale = 0.08838834764831845f;

    __shared__ float  s_q[kInkDsparkBlock][3][kInkHeadDim];  // 12 KiB
    __shared__ float4 s_k[16][32];                           // 8 KiB (ZERO FP16 conversions!)
    __shared__ float4 s_v[16][32];                           // 8 KiB

    constexpr int kNQ = kInkDsparkBlock * 3;
    for (int idx = warp; idx < kNQ; idx += 4) {
        const int s = idx / 3;
        const int g = idx % 3;
        const int h = kh * 3 + g;
        const int row = b * kInkDsparkBlock + s;
        *(float4*)(&s_q[s][g][lane * 4]) = *(const float4*)(q + ((size_t)row * kInkHeads + h) * kInkHeadDim + lane * 4);
    }
    __syncthreads();

    constexpr int kQPerWarp = 6;
    float mq[kQPerWarp], lq[kQPerWarp];
    float a0[kQPerWarp], a1[kQPerWarp], a2[kQPerWarp], a3[kQPerWarp];
#pragma unroll
    for (int qi = 0; qi < kQPerWarp; ++qi) {
        mq[qi] = -1e30f; lq[qi] = 0.0f;
        a0[qi] = a1[qi] = a2[qi] = a3[qi] = 0.0f;
    }

    const int c0    = is_swa ? max(0, apos - swa_win + 1) : 0;
    const int c_end = is_swa ? apos : min(apos, ctx_cap);

    for (int kt = c0; kt < c_end; kt += 16) {
        const int nt = min(16, c_end - kt);
        for (int j = warp; j < nt; j += 4) {
            const int kp = kt + j;
            const int p_idx = kv.is_swa ? (kp % kv.cap_pos) : kp;
            float4 kv4, vv4;
            ink_kv_load4(kv, sl, p_idx, kh, lane, kv4, vv4);
            s_k[j][lane] = kv4;
            s_v[j][lane] = vv4;
        }
        __syncthreads();
#pragma unroll
        for (int qi = 0; qi < kQPerWarp; ++qi) {
            const int idx = warp + qi * 4;
            const int s = idx / 3;
            const int g = idx % 3;
            const float4 qv = *(const float4*)(&s_q[s][g][lane * 4]);
            float mw = mq[qi], lw = lq[qi];
            float o0 = a0[qi], o1 = a1[qi], o2 = a2[qi], o3 = a3[qi];
            for (int j = 0; j < nt; ++j) {
                const float4 kv4 = s_k[j][lane];
                float dot = qv.x * kv4.x + qv.y * kv4.y + qv.z * kv4.z + qv.w * kv4.w;
                dot = warp_sum(dot) * scale;
                const float mn = fmaxf(mw, dot);
                const float al = __expf(mw - mn);
                const float p  = __expf(dot - mn);
                const float4 vv4 = s_v[j][lane];
                lw = lw * al + p;
                o0 = o0 * al + p * vv4.x; o1 = o1 * al + p * vv4.y;
                o2 = o2 * al + p * vv4.z; o3 = o3 * al + p * vv4.w;
                mw = mn;
            }
            mq[qi] = mw; lq[qi] = lw;
            a0[qi] = o0; a1[qi] = o1; a2[qi] = o2; a3[qi] = o3;
        }
        __syncthreads();
    }

    const size_t blk_base = (size_t)b * kInkDsparkBlock;
    for (int j = warp; j < kInkDsparkBlock; j += 4) {
        const size_t base = ((blk_base + j) * kInkKVHeads + kh) * kInkHeadDim + lane * 4;
        s_k[j][lane] = *(const float4*)(kd + base);
        s_v[j][lane] = *(const float4*)(vd + base);
    }
    __syncthreads();
#pragma unroll
    for (int qi = 0; qi < kQPerWarp; ++qi) {
        const int idx = warp + qi * 4;
        const int s = idx / 3;
        const int g = idx % 3;
        const int h = kh * 3 + g;
        const int row = b * kInkDsparkBlock + s;
        const float4 qv = *(const float4*)(&s_q[s][g][lane * 4]);
        float mw = mq[qi], lw = lq[qi];
        float o0 = a0[qi], o1 = a1[qi], o2 = a2[qi], o3 = a3[qi];
#pragma unroll
        for (int j = 0; j < kInkDsparkBlock; ++j) {
            const float4 kv4 = s_k[j][lane];
            float dot = qv.x * kv4.x + qv.y * kv4.y + qv.z * kv4.z + qv.w * kv4.w;
            dot = warp_sum(dot) * scale;
            const float mn = fmaxf(mw, dot);
            const float al = __expf(mw - mn);
            const float p  = __expf(dot - mn);
            const float4 vv4 = s_v[j][lane];
            lw = lw * al + p;
            o0 = o0 * al + p * vv4.x; o1 = o1 * al + p * vv4.y;
            o2 = o2 * al + p * vv4.z; o3 = o3 * al + p * vv4.w;
            mw = mn;
        }
        const float inv = 1.0f / fmaxf(lw, 1e-20f);
        const size_t ob = ((size_t)row * kInkHeads + h) * kInkHeadDim + lane * 4;
        *(float4*)(out + ob) = make_float4(o0 * inv, o1 * inv, o2 * inv, o3 * inv);
    }
}

/// Ping-Pong Async PCIe DMA Visit-KV Prefetcher for C64 (hides 100% of mapped-host KV reads behind MoE compute!)
struct InkC64KvPrefetch {
    uint8_t*     d_slab = nullptr;
    size_t       slab_bytes = 0;
    uint8_t*     d_k[2] = {};
    uint8_t*     d_v[2] = {};
    float*       d_ksc[2] = {};
    float*       d_vsc[2] = {};
    cudaStream_t dma_st = nullptr;
    cudaEvent_t  ev_dma[2] = {};
    cudaEvent_t  ev_comp[2] = {};
    int          cap_slots = 0;
    int          cap_pos = 0;
    InkKvDtype   dtype = InkKvDtype::FP8;
    bool         ready = false;

    bool ensure(int slots, InkKvDtype kvd, int gpu_id) {
        if (ready && cap_slots >= slots && dtype == kvd) return true;
        if (kvd == InkKvDtype::FP16 || slots < 16) return false;
        free_all();
        auto bytes_for_pos = [&](int pos_c) -> size_t {
            const size_t k_b  = (size_t)slots * pos_c * kInkKVHeads * kInkHeadDim;
            const size_t v_b  = (kvd == InkKvDtype::K8V4) ? (k_b / 2u) : k_b;
            const size_t sc_b = (size_t)slots * pos_c * kInkKVHeads * sizeof(float);
            const size_t one  = ((k_b + 255ull) & ~255ull) + ((v_b + 255ull) & ~255ull)
                              + 2u * ((sc_b + 255ull) & ~255ull);
            return 2u * one;
        };
        size_t vf = 0, vt = 0;
        if (ink_cuda_mem_info(&vf, &vt, gpu_id) != cudaSuccess) return false;
        int chosen = 0;
        // Prefer cap_pos=256 (66 MiB FP8) unless >= 280 MiB free so VRAM ShortConv (117 MiB) + DSpark VRAM KV (99 MiB) all fit!
        if (vf > bytes_for_pos(512) + (160ull << 20))      chosen = 512;
        else if (vf > bytes_for_pos(384) + (120ull << 20)) chosen = 384;
        else if (vf > bytes_for_pos(256) + (28ull << 20))  chosen = 256;
        else if (vf > bytes_for_pos(192) + (24ull << 20))  chosen = 192;
        else if (vf > bytes_for_pos(160) + (20ull << 20))  chosen = 160;
        else return false;

        const size_t tot = bytes_for_pos(chosen);
        if (cudaMalloc(&d_slab, tot) != cudaSuccess) { cudaGetLastError(); d_slab = nullptr; slab_bytes = 0; return false; }
        slab_bytes = tot;
        const size_t k_b  = (size_t)slots * chosen * kInkKVHeads * kInkHeadDim;
        const size_t v_b  = (kvd == InkKvDtype::K8V4) ? (k_b / 2u) : k_b;
        const size_t sc_b = (size_t)slots * chosen * kInkKVHeads * sizeof(float);
        size_t off = 0;
        auto bump = [&](size_t b) -> uint8_t* {
            uint8_t* p = d_slab + off;
            off += (b + 255ull) & ~255ull;
            return p;
        };
        for (int i = 0; i < 2; ++i) {
            d_k[i]   = bump(k_b);
            d_v[i]   = bump(v_b);
            d_ksc[i] = reinterpret_cast<float*>(bump(sc_b));
            d_vsc[i] = reinterpret_cast<float*>(bump(sc_b));
            INK_CK(cudaEventCreateWithFlags(&ev_dma[i], cudaEventDisableTiming));
            INK_CK(cudaEventCreateWithFlags(&ev_comp[i], cudaEventDisableTiming));
        }
        INK_CK(cudaStreamCreateWithFlags(&dma_st, cudaStreamNonBlocking));
        cap_slots = slots;
        cap_pos   = chosen;
        dtype     = kvd;
        ready     = true;
        std::fprintf(stderr, "[Ink5060Ti] C64 Ping-Pong Async DMA KV Prefetch active (slots=%d, cap_pos=%d, %.1f MiB VRAM)\n",
                     cap_slots, cap_pos, (double)tot / 1048576.0);
        return true;
    }

    bool prefetch_visit(
        int buf, const InkKvVisitDesc& orig,
        const void* raw_h_k, const void* raw_h_v, const void* raw_h_ksc, const void* raw_h_vsc,
        int n_pos, bool wait_comp) {
        if (!ready) return false;
        const bool use_true_h2d = (raw_h_k && raw_h_v && raw_h_ksc && raw_h_vsc);
        const void* src_k   = use_true_h2d ? raw_h_k   : (orig.is_swa ? (const void*)orig.k_f8 : (const void*)orig.h_k_f8);
        const void* src_v   = use_true_h2d ? raw_h_v   : (orig.is_swa ? (const void*)orig.v_f8 : (const void*)orig.h_v_f8);
        const void* src_ksc = use_true_h2d ? raw_h_ksc : (orig.is_swa ? (const void*)orig.k_sc : (const void*)orig.h_k_sc);
        const void* src_vsc = use_true_h2d ? raw_h_vsc : (orig.is_swa ? (const void*)orig.v_sc : (const void*)orig.h_v_sc);
        const size_t src_cap = (size_t)(orig.is_swa ? orig.dev_cap : orig.host_cap);
        if (!src_k || !src_v || !src_ksc || !src_vsc || src_cap == 0) return false;
        const int len = std::max(1, std::min<int>((int)src_cap, std::min(cap_pos, n_pos)));
        if (wait_comp) INK_CK(cudaStreamWaitEvent(dma_st, ev_comp[buf], 0));
        const size_t k_row_b  = (size_t)kInkKVHeads * kInkHeadDim;
        const size_t v_row_b  = (dtype == InkKvDtype::K8V4) ? (k_row_b / 2u) : k_row_b;
        const size_t sc_row_b = (size_t)kInkKVHeads * sizeof(float);
        const cudaMemcpyKind kind = use_true_h2d ? cudaMemcpyHostToDevice : cudaMemcpyDefault;
        INK_CK(cudaMemcpy2DAsync(d_k[buf],   (size_t)cap_pos * k_row_b,  src_k,   src_cap * k_row_b,  (size_t)len * k_row_b,  cap_slots, kind, dma_st));
        INK_CK(cudaMemcpy2DAsync(d_v[buf],   (size_t)cap_pos * v_row_b,  src_v,   src_cap * v_row_b,  (size_t)len * v_row_b,  cap_slots, kind, dma_st));
        INK_CK(cudaMemcpy2DAsync(d_ksc[buf], (size_t)cap_pos * sc_row_b, src_ksc, src_cap * sc_row_b, (size_t)len * sc_row_b, cap_slots, kind, dma_st));
        INK_CK(cudaMemcpy2DAsync(d_vsc[buf], (size_t)cap_pos * sc_row_b, src_vsc, src_cap * sc_row_b, (size_t)len * sc_row_b, cap_slots, kind, dma_st));
        INK_CK(cudaEventRecord(ev_dma[buf], dma_st));
        return true;
    }

    InkKvVisitDesc make_desc(int buf, const InkKvVisitDesc& orig) const {
        InkKvVisitDesc d{};
        d.k_f8     = d_k[buf];
        d.v_f8     = d_v[buf];
        d.k_sc     = d_ksc[buf];
        d.v_sc     = d_vsc[buf];
        d.cap_pos  = cap_pos;
        d.dev_cap  = cap_pos;
        d.host_cap = 0;
        d.is_swa   = orig.is_swa;
        d.kv_fmt   = orig.kv_fmt;
        return d;
    }

    void free_all() {
        if (dma_st) { cudaStreamDestroy(dma_st); dma_st = nullptr; }
        for (int i = 0; i < 2; ++i) {
            if (ev_dma[i])  { cudaEventDestroy(ev_dma[i]);  ev_dma[i]  = nullptr; }
            if (ev_comp[i]) { cudaEventDestroy(ev_comp[i]); ev_comp[i] = nullptr; }
            d_k[i] = d_v[i] = nullptr; d_ksc[i] = d_vsc[i] = nullptr;
        }
        if (d_slab) { cudaFree(d_slab); d_slab = nullptr; }
        slab_bytes = 0;
        cap_slots = 0; cap_pos = 0; ready = false;
    }
};

/// Batched Markov RNN prep across B slots for draft step s in 0..6!
__global__ void __launch_bounds__(256) k_dspark_markov_prep_slots(
    const int32_t* __restrict__ prev_toks, // [B]
    const float* __restrict__ states,      // [B, 256]
    const float* __restrict__ d_cur_all,   // [B * 8, 2304]
    int step_s,                            // 0..6 (uses hidden at b*8 + step_s + 1)
    const float* __restrict__ f_rnn_slots, // [max_slots, frnn_cap, 2304]
    const int32_t* __restrict__ slot_ids,  // [B]
    const int32_t* __restrict__ apos_arr,  // [B]
    int frnn_cap,
    const uint8_t* __restrict__ mw1_raw, int mw1_fmt,
    float* __restrict__ joint_out,         // [B, 5120]
    int B) {
    constexpr int R = kInkMarkovRank;
    constexpr int J = 2 * R + 2 * kInkHidden;
    const int b = blockIdx.x;
    if (b >= B) return;
    const int tid = threadIdx.x;
    const int32_t pt = prev_toks[b];
    const int sl     = slot_ids[b];
    const int apos   = apos_arr[b];
    const int fidx   = (apos > 0 ? (apos - 1) : 0) % frnn_cap;
    const float* ctx = f_rnn_slots + ((size_t)sl * frnn_cap + fidx) * kInkHidden;
    const float* hs  = d_cur_all + ((size_t)b * kInkDsparkBlock + (step_s + 1)) * kInkHidden;
    float* jo        = joint_out + (size_t)b * J;

    jo[tid] = states[(size_t)b * R + tid];
    float mv = 0.0f;
    if (mw1_fmt == FMT_BF16) mv = bf16_to_f32(reinterpret_cast<const uint16_t*>(mw1_raw)[(size_t)pt * R + tid]);
    else if (mw1_fmt == FMT_F32) mv = reinterpret_cast<const float*>(mw1_raw)[(size_t)pt * R + tid];
    else {
        const float sc = reinterpret_cast<const float*>(mw1_raw)[pt];
        mv = (float)reinterpret_cast<const int8_t*>(mw1_raw + (size_t)kInkVocabPad * 4)[(size_t)pt * R + tid] * sc;
    }
    jo[R + tid] = mv;
    for (int d = tid; d < kInkHidden; d += 256) {
        jo[2 * R + d]              = hs[d];
        jo[2 * R + kInkHidden + d] = ctx[d];
    }
}

/// Batched GRU update in-place across B slots!
__global__ void __launch_bounds__(256) k_dspark_gru_upd_slots(
    const float* __restrict__ proj,  // [B, 768]
    const float* __restrict__ rp_b,  // [768]
    float* __restrict__ states,      // [B, 256] in/out
    float* __restrict__ lat_out,     // [B, 256]
    int B) {
    constexpr int R = kInkMarkovRank;
    const int b = blockIdx.x;
    if (b >= B) return;
    const int tid = threadIdx.x;
    const float* pr = proj + (size_t)b * (3 * R);
    float* st_b     = states + (size_t)b * R;
    const float p0 = pr[tid]         + rp_b[tid];
    const float p1 = pr[R + tid]     + rp_b[R + tid];
    const float p2 = pr[2 * R + tid] + rp_b[2 * R + tid];
    const float g  = 1.0f / (1.0f + __expf(-p0));
    const float c  = tanhf(p1);
    st_b[tid]      = c + g * (st_b[tid] - c);
    lat_out[(size_t)b * R + tid] = tanhf(p2);
}

/// Batched add hidden_out bias + per-slot hs!
__global__ void __launch_bounds__(256) k_dspark_add_bias_h_slots(
    float* __restrict__ corr,              // [B, 2304]
    const float* __restrict__ ho_b,        // [2304]
    const float* __restrict__ d_cur_all,   // [B * 8, 2304]
    int step_s,
    bool add_hs,
    int B) {
    const int b = blockIdx.y;
    const int d = blockIdx.x * 256 + threadIdx.x;
    if (b >= B || d >= kInkHidden) return;
    const float* hs = d_cur_all + ((size_t)b * kInkDsparkBlock + (step_s + 1)) * kInkHidden;
    corr[(size_t)b * kInkHidden + d] += ho_b[d] + (add_hs ? hs[d] : 0.0f);
}

/// Batched dp4a candidate scorer where EACH slot b in 0..B-1 has its OWN K_cand candidate list `cand_ids[b, K_cand]`!
/// Grid = dim3((K_cand + 7) / 8, B), Block = 256.
__global__ void __launch_bounds__(256) k_dspark_score_cand_dp4a_slots(
    const int8_t* __restrict__ q8,         // [V, 2304]
    const float* __restrict__ sc,          // [V]
    const float* __restrict__ corr,        // [B, 2304]
    const int32_t* __restrict__ cand_ids,  // [B, n_slots, K_cand]
    int step_s,
    int n_slots,
    int K_cand,
    int B,
    float* __restrict__ cand_scores) {     // [B, K_cand]
    const int b = blockIdx.y;
    if (b >= B) return;
    __shared__ int8_t s_q[kInkHidden];
    __shared__ float s_sc;
    __shared__ float sh_max[8];
    const int tid = threadIdx.x;
    const int warp = tid >> 5;
    const int lane = tid & 31;
    const float* cb = corr + (size_t)b * kInkHidden;

    float am = 0.0f;
    for (int d = tid; d < kInkHidden; d += 256) am = fmaxf(am, fabsf(cb[d]));
    am = warp_max(am);
    if (lane == 0) sh_max[warp] = am;
    __syncthreads();
    if (tid == 0) {
        float g = sh_max[0];
        for (int i = 1; i < 8; ++i) g = fmaxf(g, sh_max[i]);
        s_sc = (g > 0.0f) ? (g * (1.0f / 127.0f)) : 1.0f;
    }
    __syncthreads();
    const float inv = (s_sc > 0.0f) ? (1.0f / s_sc) : 0.0f;
    for (int d = tid; d < kInkHidden; d += 256) {
        s_q[d] = (int8_t)max(-127, min(127, __float2int_rn(cb[d] * inv)));
    }
    __syncthreads();

    const int k = blockIdx.x * 8 + warp;
    if (k >= K_cand) return;
    const int32_t row = cand_ids[((size_t)b * n_slots + step_s) * K_cand + k];
    const uint4* w16 = reinterpret_cast<const uint4*>(q8 + (size_t)row * kInkHidden);
    const uint4* x16 = reinterpret_cast<const uint4*>(s_q);
    int32_t acc = 0;
    for (int i = lane; i < (kInkHidden / 16); i += 32) {
        const uint4 w = w16[i];
        const uint4 x = x16[i];
        acc = dp4a_i8((int)w.x, (int)x.x, acc);
        acc = dp4a_i8((int)w.y, (int)x.y, acc);
        acc = dp4a_i8((int)w.z, (int)x.z, acc);
        acc = dp4a_i8((int)w.w, (int)x.w, acc);
    }
    acc = warp_sum(acc);
    if (lane == 0) {
        const float logit = (float)acc * sc[row] * s_sc;
        cand_scores[(size_t)b * K_cand + k] = kInkSoftcap * tanhf(logit * (1.0f / kInkSoftcap));
    }
}

/// Batched Top-1 argmax over K_cand scores for B slots -> writes draft token `out_draft[b, step_s]` and `prev_toks[b]`!
/// Grid = B, Block = 256.
__global__ void __launch_bounds__(256) k_dspark_top1_reduce_slots(
    const float* __restrict__ cand_scores, // [B, K_cand]
    const int32_t* __restrict__ cand_ids,  // [B, n_slots, K_cand]
    int step_s,
    int n_slots,
    int K_cand,
    int B,
    int32_t* __restrict__ prev_toks,       // [B] out
    int32_t* __restrict__ out_drafts) {    // [B, 7] out
    const int b = blockIdx.x;
    if (b >= B) return;
    const int tid = threadIdx.x;
    const float* sc = cand_scores + (size_t)b * K_cand;
    const int32_t* ids = cand_ids + ((size_t)b * n_slots + step_s) * K_cand;

    float bv = -3e38f;
    int32_t bi = 0;
    for (int k = tid; k < K_cand; k += 256) {
        const float v = sc[k];
        const int32_t id = ids[k];
        if (v > bv || (v == bv && id < bi)) { bv = v; bi = id; }
    }
    const int warp = tid >> 5;
    const int lane = tid & 31;
#pragma unroll
    for (int o = 16; o >= 1; o >>= 1) {
        const float ov = __shfl_xor_sync(0xFFFFFFFFu, bv, o);
        const int32_t oi = __shfl_xor_sync(0xFFFFFFFFu, bi, o);
        if (ov > bv || (ov == bv && oi < bi)) { bv = ov; bi = oi; }
    }
    __shared__ float sm_v[8];
    __shared__ int32_t sm_i[8];
    if (lane == 0) { sm_v[warp] = bv; sm_i[warp] = bi; }
    __syncthreads();
    if (tid == 0) {
        float best = sm_v[0];
        int32_t best_i = sm_i[0];
        for (int w = 1; w < 8; ++w) {
            if (sm_v[w] > best || (sm_v[w] == best && sm_i[w] < best_i)) { best = sm_v[w]; best_i = sm_i[w]; }
        }
        prev_toks[b] = best_i;
        out_drafts[(size_t)b * 7 + step_s] = best_i;
    }
}

/// Compact selected accepted tap rows from [B*K, 2304] into [n_tot, 2304] for multi-slot DSpark ingest!
__global__ void __launch_bounds__(256) k_ink_compact_taps_rows(
    const float* __restrict__ src,         // [B * K, 2304]
    float* __restrict__ dst,               // [n_tot, 2304]
    const int32_t* __restrict__ src_rows,  // [n_tot]
    int n_tot) {
    const int i = blockIdx.x;
    if (i >= n_tot) return;
    const int r = src_rows[i];
    const float4* s4 = reinterpret_cast<const float4*>(src + (size_t)r * kInkHidden);
    float4* d4 = reinterpret_cast<float4*>(dst + (size_t)i * kInkHidden);
    for (int d = threadIdx.x; d < (kInkHidden / 4); d += 256) d4[d] = s4[d];
}

/// Quantize B * n_slots DSpark draft hidden states from ms_cur[B, 8, 2304] (slots 1..n_slots) to int8 [B*n_slots, 2304] + scale [B*n_slots]!
/// Grid = B * n_slots, Block = 256.
__global__ void __launch_bounds__(256) k_dspark_quant_h_q8_slots(
    const float* __restrict__ ms_cur,  // [B * 8, 2304]
    int8_t* __restrict__ h_q8,         // [R_pad, 2304]
    float* __restrict__ h_sc,          // [R_pad]
    int B, int n_slots) {
    const int r = blockIdx.x;  // 0 .. B*n_slots-1
    if (r >= B * n_slots) return;
    const int b = r / n_slots;
    const int s = r % n_slots;
    const float* src = ms_cur + ((size_t)b * kInkDsparkBlock + (s + 1)) * kInkHidden;
    int8_t* dst      = h_q8 + (size_t)r * kInkHidden;
    const int tid    = threadIdx.x;
    const int warp   = tid >> 5;
    const int lane   = tid & 31;
    __shared__ float sh_max[8];
    __shared__ float s_sc;
    float am = 0.0f;
    for (int d = tid; d < kInkHidden; d += 256) am = fmaxf(am, fabsf(src[d]));
    am = warp_max(am);
    if (lane == 0) sh_max[warp] = am;
    __syncthreads();
    if (tid == 0) {
        float g = sh_max[0];
        for (int i = 1; i < 8; ++i) g = fmaxf(g, sh_max[i]);
        s_sc = (g > 0.0f) ? (g * (1.0f / 127.0f)) : 1.0f;
        h_sc[r] = s_sc;
    }
    __syncthreads();
    const float inv = (s_sc > 0.0f) ? (1.0f / s_sc) : 0.0f;
    for (int d = tid; d < kInkHidden; d += 256) {
        dst[d] = (int8_t)max(-127, min(127, __float2int_rn(src[d] * inv)));
    }
}

/// Extract Top-16 candidates per `n_ch` striped sub-chunks of a vocab tile [v0, v0+vt) from cuBLAS INT8 TC GEMM output!
/// Writes directly into tmp_ids/tmp_vals [R, 2048] matching k_dspark_top_cand_s2's exact 128x16 layout!
/// Grid = dim3(n_ch, R), Block = 256 (n_tiles * n_ch == 128, e.g. 8 tiles x 16 sub-chunks = 128!).
__global__ void __launch_bounds__(256) k_dspark_tile_top16_i32(
    const int32_t* __restrict__ tile_i32,  // [R_pad, vt] (row r, col j in [0..vt))
    const float* __restrict__ unemb_sc,    // [V]
    const float* __restrict__ h_sc,        // [R_pad]
    int v0, int vt, int tile_idx, int n_ch, int V,
    int32_t* __restrict__ tmp_ids,         // [R, 2048]
    float* __restrict__ tmp_vals) {        // [R, 2048]
    const int r  = blockIdx.y;
    const int ch = blockIdx.x;  // 0..n_ch-1
    const int tid  = threadIdx.x;
    const int warp = tid >> 5;
    const int lane = tid & 31;
    const int32_t* row = tile_i32 + (size_t)r * vt;
    const float hsc = h_sc[r];

    float b0 = -3e38f, b1 = -3e38f;
    int32_t i0 = 0, i1 = 0;
    for (int base = ch * 256; base < vt; base += n_ch * 256) {
        const int j = base + tid;
        if (j < vt && (v0 + j) < V) {
            const int32_t vid = v0 + j;
            const float v = (float)row[j] * unemb_sc[vid] * hsc;
            if (v > b0) { b1 = b0; i1 = i0; b0 = v; i0 = vid; }
            else if (v > b1) { b1 = v; i1 = vid; }
        }
    }
    float wv0 = b0; int32_t wi0 = i0;
#pragma unroll
    for (int o = 16; o >= 1; o >>= 1) {
        float ov = __shfl_xor_sync(0xFFFFFFFFu, wv0, o);
        int32_t oi = __shfl_xor_sync(0xFFFFFFFFu, wi0, o);
        if (ov > wv0 || (ov == wv0 && oi < wi0)) { wv0 = ov; wi0 = oi; }
    }
    float cand2 = (i0 == wi0) ? b1 : b0;
    int32_t cid2 = (i0 == wi0) ? i1 : i0;
    float wv1 = cand2; int32_t wi1 = cid2;
#pragma unroll
    for (int o = 16; o >= 1; o >>= 1) {
        float ov = __shfl_xor_sync(0xFFFFFFFFu, wv1, o);
        int32_t oi = __shfl_xor_sync(0xFFFFFFFFu, wi1, o);
        if (ov > wv1 || (ov == wv1 && oi < wi1)) { wv1 = ov; wi1 = oi; }
    }
    if (lane == 0) {
        const size_t ob = (size_t)r * 2048 + (size_t)(tile_idx * n_ch + ch) * 16 + warp * 2;
        tmp_ids[ob + 0]  = wi0; tmp_vals[ob + 0] = wv0;
        tmp_ids[ob + 1]  = wi1; tmp_vals[ob + 1] = wv1;
    }
}

// ============================================================================
// 6. 100%-VRAM 16 GB Compact Loader Kernels (BF16->Q8_ROW, MX8->SF4, Q8_ROW INT8 TC Argmax)
// ============================================================================

/// Convert BF16 [N_src, K] -> FMT_Q8ROW ([N_pad] float scales followed by [N_pad, K] int8), zero-padding rows >= N_src!
/// Grid = N_pad, Block = 256.
__global__ void __launch_bounds__(256) k_ink_bf16_to_q8row(
    const uint16_t* __restrict__ src,  // [N_src, K] BF16
    float* __restrict__ dst_sc,        // [N_pad]
    int8_t* __restrict__ dst_q,        // [N_pad, K]
    int N_src, int N_pad, int K) {
    const int r = blockIdx.x;
    if (r >= N_pad) return;
    const int tid  = threadIdx.x;
    const int warp = tid >> 5;
    const int lane = tid & 31;
    int8_t* qr = dst_q + (size_t)r * K;
    if (r >= N_src) {
        if (tid == 0) dst_sc[r] = 0.0f;
        for (int c = tid; c < K; c += 256) qr[c] = 0;
        return;
    }
    const uint16_t* wr = src + (size_t)r * K;
    __shared__ float sh_max[8];
    __shared__ float s_sc;
    float am = 0.0f;
    for (int c = tid; c < K; c += 256) {
        am = fmaxf(am, fabsf(bf16_to_f32(wr[c])));
    }
    am = warp_max(am);
    if (lane == 0) sh_max[warp] = am;
    __syncthreads();
    if (tid == 0) {
        float g = sh_max[0];
        for (int i = 1; i < 8; ++i) g = fmaxf(g, sh_max[i]);
        s_sc = (g > 0.0f) ? (g * (1.0f / 127.0f)) : 0.0f;
        dst_sc[r] = s_sc;
    }
    __syncthreads();
    const float inv = (s_sc > 0.0f) ? (1.0f / s_sc) : 0.0f;
    for (int c = tid; c < K; c += 256) {
        const float v = bf16_to_f32(wr[c]);
        qr[c] = (int8_t)max(-127, min(127, __float2int_rn(v * inv)));
    }
}

/// Pass 1 of GPU MX8 -> SF4 transcode: compute exact global amax of an [N, K] MXFP8 matrix into d_amax_u32!
__global__ void __launch_bounds__(256) k_ink_mx8_amax(
    const uint8_t* __restrict__ src_p,  // [N, K] E4M3FN
    const uint8_t* __restrict__ src_s,  // [N, K/32] UE8M0
    int N, int K,
    uint32_t* __restrict__ d_amax_u32) {
    const int64_t total_blks = (int64_t)N * (K >> 5);
    const int64_t stride = (int64_t)gridDim.x * 256;
    float am = 0.0f;
    for (int64_t b = (int64_t)blockIdx.x * 256 + threadIdx.x; b < total_blks; b += stride) {
        const float sc = ue8m0_to_f32(src_s[b]);
        if (sc == 0.0f) continue;
        const uint8_t* p = src_p + b * 32;
        float bm = 0.0f;
#pragma unroll
        for (int i = 0; i < 32; ++i) bm = fmaxf(bm, fabsf(e4m3_to_f32(p[i])));
        am = fmaxf(am, bm * sc);
    }
    const int warp = threadIdx.x >> 5;
    const int lane = threadIdx.x & 31;
    am = warp_max(am);
    __shared__ float sh[8];
    if (lane == 0) sh[warp] = am;
    __syncthreads();
    if (threadIdx.x == 0) {
        float g = sh[0];
        for (int i = 1; i < 8; ++i) g = fmaxf(g, sh[i]);
        atomicMax(d_amax_u32, __float_as_uint(g));
    }
}

__device__ __forceinline__ uint8_t ink_pack_e2m1x2_portable(float v_lo, float v_hi) {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000) && defined(CUDART_VERSION) && (CUDART_VERSION >= 12080)
    const __nv_fp4x2_e2m1 q2(make_float2(v_lo, v_hi));
    return q2.__x;
#else
    auto q1 = [](float x) -> uint8_t {
        uint8_t s = (x < 0.0f) ? 8u : 0u;
        float a = fabsf(x);
        uint8_t m = (a < 0.25f) ? 0u : (a < 0.75f) ? 1u : (a < 1.25f) ? 2u : (a < 1.75f) ? 3u
                  : (a < 2.5f)  ? 4u : (a < 3.5f)  ? 5u : (a < 5.0f)  ? 6u : 7u;
        return s | m;
    };
    return q1(v_lo) | (uint8_t)(q1(v_hi) << 4);
#endif
}

/// Pass 2 of GPU MX8 -> SF4 transcode: quantize [N, K] MXFP8 directly into .strata v1.1 FMT_SF4 layout!
/// Grid = N, Block = 256 (1 thread per 16-element sub-block: zero syncthreads, < 10 us/matrix!).
__global__ void __launch_bounds__(256) k_ink_mx8_to_sf4(
    const uint8_t* __restrict__ src_p,    // [N, K] E4M3FN
    const uint8_t* __restrict__ src_s,    // [N, K/32] UE8M0
    const uint32_t* __restrict__ d_amax_u32,
    uint8_t* __restrict__ dst_p,          // [N, K/2] E2M1x2
    uint8_t* __restrict__ dst_s,          // [N, K/16] UE4M3
    float* __restrict__ dst_hdr_pair,     // writes {w_outer, 1.0f}
    int N, int K) {
    const int r = blockIdx.x;
    if (r >= N) return;
    const float amax = __uint_as_float(*d_amax_u32);
    const float w_outer = fmaxf(amax / (6.0f * 240.0f), 1e-6f);
    if (r == 0 && threadIdx.x == 0) {
        dst_hdr_pair[0] = w_outer;
        dst_hdr_pair[1] = 1.0f;
    }
    const int nblk16 = K >> 4;
    for (int b = threadIdx.x; b < nblk16; b += 256) {
        const float s_mx = ue8m0_to_f32(src_s[(size_t)r * (K >> 5) + (b >> 1)]);
        const uint8_t* sp = src_p + (size_t)r * K + b * 16;
        float v[16];
        float am_sub = 0.0f;
#pragma unroll
        for (int i = 0; i < 16; ++i) {
            v[i] = e4m3_to_f32(sp[i]) * s_mx;
            am_sub = fmaxf(am_sub, fabsf(v[i]));
        }
        uint8_t sc_byte = 0u;
        float inv = 0.0f;
        if (am_sub > 0.0f) {
            const float sc_f = fminf(am_sub / (w_outer * 6.0f), 448.0f);
            const __nv_fp8_e4m3 fp8(sc_f);
            sc_byte = fp8.__x;
            const float s_sub = (float)fp8;
            const float denom = w_outer * s_sub;
            inv = (denom > 0.0f) ? (1.0f / denom) : 0.0f;
        }
        uint8_t* dp = dst_p + (size_t)r * (K >> 1) + b * 8;
#pragma unroll
        for (int j = 0; j < 8; ++j) {
            dp[j] = ink_pack_e2m1x2_portable(v[2 * j] * inv, v[2 * j + 1] * inv);
        }
        dst_s[(size_t)r * nblk16 + b] = sc_byte;
    }
}

/// Vocab-tile exact argmax over cublasGemmEx(CUDA_R_8I) int32 output [M_pad, vt] for Q8_ROW unembed!
/// Grid = M, Block = 256.
__global__ void __launch_bounds__(256) k_ink_q8row_tile_argmax_i32(
    const int32_t* __restrict__ tile_i32,  // [M_pad, vt]
    const float* __restrict__ w_sc,        // [V_pad]
    int v0, int vt, int tile_idx, int V, int n_tiles,
    float* __restrict__ out_val,           // [M, n_tiles]
    int32_t* __restrict__ out_id) {        // [M, n_tiles]
    const int m = blockIdx.x;
    const int tid = threadIdx.x;
    const int32_t* row = tile_i32 + (size_t)m * vt;
    float bv = -3e38f;
    int32_t bi = 0;
    for (int j = tid; j < vt; j += 256) {
        const int vid = v0 + j;
        if (vid < V) {
            const float v = (float)row[j] * w_sc[vid];
            if (v > bv || (v == bv && vid < bi)) { bv = v; bi = vid; }
        }
    }
    const int warp = tid >> 5;
    const int lane = tid & 31;
#pragma unroll
    for (int o = 16; o >= 1; o >>= 1) {
        const float ov = __shfl_xor_sync(0xFFFFFFFFu, bv, o);
        const int32_t oi = __shfl_xor_sync(0xFFFFFFFFu, bi, o);
        if (ov > bv || (ov == bv && oi < bi)) { bv = ov; bi = oi; }
    }
    __shared__ float sm_v[8];
    __shared__ int32_t sm_i[8];
    if (lane == 0) { sm_v[warp] = bv; sm_i[warp] = bi; }
    __syncthreads();
    if (tid == 0) {
        float best = sm_v[0];
        int32_t best_i = sm_i[0];
        for (int w = 1; w < 8; ++w) {
            if (sm_v[w] > best || (sm_v[w] == best && sm_i[w] < best_i)) { best = sm_v[w]; best_i = sm_i[w]; }
        }
        out_val[(size_t)m * n_tiles + tile_idx] = best;
        out_id[(size_t)m * n_tiles + tile_idx]  = best_i;
    }
}

/// Final argmax across n_tiles <= 32 tiles for M rows -> d_argmax[M]! Grid = M, Block = 32.
__global__ void __launch_bounds__(32) k_ink_q8row_final_argmax(
    const float* __restrict__ tile_val,    // [M, n_tiles]
    const int32_t* __restrict__ tile_id,   // [M, n_tiles]
    int n_tiles,
    int32_t* __restrict__ d_argmax) {      // [M]
    const int m = blockIdx.x;
    const int lane = threadIdx.x;
    float bv = (lane < n_tiles) ? tile_val[(size_t)m * n_tiles + lane] : -3e38f;
    int32_t bi = (lane < n_tiles) ? tile_id[(size_t)m * n_tiles + lane] : 0;
#pragma unroll
    for (int o = 16; o >= 1; o >>= 1) {
        const float ov = __shfl_xor_sync(0xFFFFFFFFu, bv, o);
        const int32_t oi = __shfl_xor_sync(0xFFFFFFFFu, bi, o);
        if (ov > bv || (ov == bv && oi < bi)) { bv = ov; bi = oi; }
    }
    if (lane == 0) d_argmax[m] = bi;
}

// ============================================================================
// 7. Zero-Cost CPU Prompt-Lookup (PLD) Suffix Grafter into InkTreeSpec
//    Scans recent [prompt + generated] history (< 5 us) for 5->4->3->2-gram suffix
//    matches at BOTH DSpark's deepest spine leaf AND the anchor, grafting verified
//    continuation chains into InkTreeSpec (100% target-verified, zero quality loss!).
// ============================================================================
template <typename TreeT>
inline int ink_pld_graft_tree(
    const std::vector<int32_t>& hist,
    int32_t anchor_tok,
    TreeT& tr,
    int max_m,
    int max_depth) noexcept {
    const int H = (int)hist.size();
    if (H < 4 || tr.M <= 0 || tr.M >= max_m) return 0;
    const int scan_beg = std::max(0, H - 8192);

    auto find_match = [&](const int32_t* ctx, int ctx_len, int max_cont, int32_t* out_cont) -> int {
        for (int ng = std::min(5, ctx_len); ng >= 3; --ng) {
            const int32_t* needle = ctx + (ctx_len - ng);
            // Scan backwards from H - ng - 1 so most recent occurrence wins!
            for (int pos = H - ng - 1; pos >= scan_beg; --pos) {
                bool eq = true;
                for (int k = 0; k < ng; ++k) {
                    if (hist[(size_t)(pos + k)] != needle[k]) { eq = false; break; }
                }
                if (eq) {
                    const int avail = std::min(max_cont, H - (pos + ng));
                    if (avail > 0) {
                        for (int k = 0; k < avail; ++k) out_cont[k] = hist[(size_t)(pos + ng + k)];
                        return avail;
                    }
                }
            }
        }
        return 0;
    };

    auto graft_chain = [&](int start_node, const int32_t* toks, int ntok) -> int {
        int added = 0;
        int cur = start_node;
        for (int k = 0; k < ntok; ++k) {
            const int32_t tk = toks[k];
            // Follow existing child if already present in tree!
            int child = -1;
            for (int j = cur + 1; j < tr.M; ++j) {
                if (tr.parent[j] == cur && tr.tok[j] == tk) { child = j; break; }
            }
            if (child >= 0) {
                cur = child;
                continue;
            }
            if (tr.M >= max_m || (int)tr.depth[cur] + 1 > max_depth) break;
            const int nid = tr.M++;
            tr.tok[nid]    = tk;
            tr.parent[nid] = (int8_t)cur;
            tr.depth[nid]  = (uint8_t)(tr.depth[cur] + 1);
            cur = nid;
            ++added;
        }
        return added;
    };

    int total_added = 0;
    // 1. Graft below DSpark's deepest high-confidence spine leaf (extends depth 7 -> depth 16..20 on structured/JSON/code!)
    int best_leaf = 0;
    for (int i = 1; i < tr.M; ++i) {
        if (tr.depth[i] > tr.depth[best_leaf]) best_leaf = i;
    }
    if (best_leaf > 0 && tr.depth[best_leaf] >= 2 && tr.M < max_m) {
        int32_t spine[kInkDeepTreeDepth];
        int sp_len = 0;
        for (int u = best_leaf; u >= 0 && sp_len < kInkDeepTreeDepth; u = tr.parent[u]) {
            spine[sp_len++] = tr.tok[u];
        }
        // Build context tail: last 4 hist tokens (excluding anchor if already last) + spine[sp_len-1..0]
        int32_t ctx[kInkDeepTreeDepth + 8];
        int clen = 0;
        const int h_end = (H > 0 && hist.back() == anchor_tok) ? (H - 1) : H;
        const int h_take = std::min(5, h_end);
        for (int i = h_end - h_take; i < h_end; ++i) ctx[clen++] = hist[(size_t)i];
        for (int i = sp_len - 1; i >= 0; --i) ctx[clen++] = spine[i];
        int32_t cont[16];
        const int nc = find_match(ctx, clen, std::min(12, max_m - tr.M), cont);
        if (nc > 0) total_added += graft_chain(best_leaf, cont, nc);
    }

    // 2. Also check direct suffix match from anchor_tok (node 0) for instant prompt-copy / structured loops!
    if (tr.M < max_m) {
        int32_t ctx[8];
        int clen = 0;
        const int h_end = (H > 0 && hist.back() == anchor_tok) ? (H - 1) : H;
        const int h_take = std::min(5, h_end);
        for (int i = h_end - h_take; i < h_end; ++i) ctx[clen++] = hist[(size_t)i];
        ctx[clen++] = anchor_tok;
        int32_t cont[12];
        const int nc = find_match(ctx, clen, std::min(10, max_m - tr.M), cont);
        if (nc > 0) total_added += graft_chain(0, cont, nc);
    }

    if (total_added > 0) tr.build();
    return total_added;
}

/// Fast 16-byte-aligned Q8_ROW unembed logits [M, V] using shared unembed_screen (q8 [V_pad, K], sc [V_pad]).
/// Grid = (V + 7) / 8, Block = 256 (8 warps, 1 row/warp -> ~1.0 ms for M=1 on RTX 5060 Ti!).
__global__ void __launch_bounds__(256) k_ink_q8row_unembed_dp4a(
    const int8_t* __restrict__ w_q,   // [V_pad, K] (256-byte aligned!)
    const float*  __restrict__ w_sc,  // [V_pad]
    const int8_t* __restrict__ x_q,   // [M, K]
    const float*  __restrict__ x_sc,  // [M]
    float* __restrict__ y,            // [M, V]
    int M, int V, int K) {
    const int warp = threadIdx.x >> 5;
    const int lane = threadIdx.x & 31;
    const int row  = blockIdx.x * 8 + warp;
    if (row >= V) return;
    const int K16 = K >> 4;
    const int4* w4 = reinterpret_cast<const int4*>(w_q + (size_t)row * K);
    const float ws = w_sc[row];
    for (int m = 0; m < M; ++m) {
        const int4* x4 = reinterpret_cast<const int4*>(x_q + (size_t)m * K);
        int32_t acc = 0;
        for (int i = lane; i < K16; i += 32) {
            const int4 wv = w4[i];
            const int4 xv = x4[i];
            acc = dp4a_i8(wv.x, xv.x, acc);
            acc = dp4a_i8(wv.y, xv.y, acc);
            acc = dp4a_i8(wv.z, xv.z, acc);
            acc = dp4a_i8(wv.w, xv.w, acc);
        }
#pragma unroll
        for (int o = 16; o >= 1; o >>= 1) acc += __shfl_xor_sync(0xFFFFFFFFu, acc, o);
        if (lane == 0) y[(size_t)m * V + row] = (float)acc * ws * x_sc[m];
    }
}

/// Convert F32 [N_src, K] -> FMT_Q8ROW ([N_pad] float scales followed by [N_pad, K] int8), for controllers.* quantization!
__global__ void __launch_bounds__(256) k_ink_f32_to_q8row(
    const float* __restrict__ src,     // [N_src, K] F32
    float* __restrict__ dst_sc,        // [N_pad]
    int8_t* __restrict__ dst_q,        // [N_pad, K]
    int N_src, int N_pad, int K) {
    const int r = blockIdx.x;
    if (r >= N_pad) return;
    const int tid  = threadIdx.x;
    const int warp = tid >> 5;
    const int lane = tid & 31;
    int8_t* qr = dst_q + (size_t)r * K;
    if (r >= N_src) {
        if (tid == 0) dst_sc[r] = 0.0f;
        for (int c = tid; c < K; c += 256) qr[c] = 0;
        return;
    }
    const float* wr = src + (size_t)r * K;
    __shared__ float sh_max[8];
    __shared__ float s_sc;
    float am = 0.0f;
    for (int c = tid; c < K; c += 256) am = fmaxf(am, fabsf(wr[c]));
    am = warp_max(am);
    if (lane == 0) sh_max[warp] = am;
    __syncthreads();
    if (tid == 0) {
        float g = sh_max[0];
        for (int i = 1; i < 8; ++i) g = fmaxf(g, sh_max[i]);
        s_sc = (g > 0.0f) ? (g * (1.0f / 127.0f)) : 0.0f;
        dst_sc[r] = s_sc;
    }
    __syncthreads();
    const float inv = (s_sc > 0.0f) ? (1.0f / s_sc) : 0.0f;
    for (int c = tid; c < K; c += 256) {
        qr[c] = (int8_t)max(-127, min(127, __float2int_rn(wr[c] * inv)));
    }
}

/// Q8_ROW Loop Controller Down Projection (small M <= 16, 4x less GDDR7 bandwidth than F32!)
template <int MAX_M>
__global__ void __launch_bounds__(256) k_ink_ctrl_down_q8_smallM(
    const float* __restrict__ cat,    // [M, D2]
    const uint8_t* __restrict__ w_q8, // FMT_Q8ROW [64, D2]
    float* __restrict__ gate,         // [M, 64]
    int M, int D2) {
    const int r = blockIdx.x;
    if (r >= kInkGateRank) return;
    const int tid = threadIdx.x;
    const int D2_4 = D2 >> 2;
    const float wsc = reinterpret_cast<const float*>(w_q8)[r];
    const int32_t* wr = reinterpret_cast<const int32_t*>(w_q8 + (size_t)kInkGateRank * 4u + (size_t)r * D2);
    float acc[MAX_M] = {};
    for (int i = tid; i < D2_4; i += 256) {
        const int32_t pk = wr[i];
        const float w0 = (float)(int8_t)(pk & 0xFF);
        const float w1 = (float)(int8_t)((pk >> 8) & 0xFF);
        const float w2 = (float)(int8_t)((pk >> 16) & 0xFF);
        const float w3 = (float)(int8_t)((pk >> 24) & 0xFF);
#pragma unroll
        for (int m = 0; m < MAX_M; ++m) {
            if (m < M) {
                const float4 x = reinterpret_cast<const float4*>(cat + (size_t)m * D2)[i];
                acc[m] = fmaf(w0, x.x, acc[m]);
                acc[m] = fmaf(w1, x.y, acc[m]);
                acc[m] = fmaf(w2, x.z, acc[m]);
                acc[m] = fmaf(w3, x.w, acc[m]);
            }
        }
    }
    __shared__ float sh[32];
#pragma unroll
    for (int m = 0; m < MAX_M; ++m) {
        if (m < M) {
            const float s = block_sum(acc[m], sh);
            if (tid == 0) gate[(size_t)m * kInkGateRank + r] = s * wsc;
        }
    }
}

/// Q8_ROW Loop Controller Apply (gain + forget, small M <= 16, 4x less GDDR7 bandwidth than F32!)
template <int MAX_M>
__global__ void __launch_bounds__(256) k_ink_ctrl_apply_q8_smallM(
    const float* __restrict__ cur,
    const float* __restrict__ msg,
    const float* __restrict__ gate,    // [M, 64]
    const uint8_t* __restrict__ wg_q8, // FMT_Q8ROW [D, 64]
    const uint8_t* __restrict__ wf_q8, // FMT_Q8ROW [D, 64]
    float* __restrict__ out,
    int M, int D) {
    __shared__ float s_gate[MAX_M][kInkGateRank];
    const int tid = threadIdx.x;
    for (int i = tid; i < M * kInkGateRank; i += 256) s_gate[i >> 6][i & 63] = gate[i];
    __syncthreads();

    const int sub = tid >> 4;  // 0..15
    const int lg  = tid & 15;  // 0..15 -> 4 elems each (64 total)
    const int d   = blockIdx.x * 16 + sub;
    if (d >= D) return;

    const float scg = reinterpret_cast<const float*>(wg_q8)[d];
    const float scf = reinterpret_cast<const float*>(wf_q8)[d];
    const int32_t pkg = reinterpret_cast<const int32_t*>(wg_q8 + (size_t)D * 4u + (size_t)d * kInkGateRank)[lg];
    const int32_t pkf = reinterpret_cast<const int32_t*>(wf_q8 + (size_t)D * 4u + (size_t)d * kInkGateRank)[lg];
    const float g0 = (float)(int8_t)(pkg & 0xFF),        g1 = (float)(int8_t)((pkg >> 8) & 0xFF),
                g2 = (float)(int8_t)((pkg >> 16) & 0xFF), g3 = (float)(int8_t)((pkg >> 24) & 0xFF);
    const float f0 = (float)(int8_t)(pkf & 0xFF),        f1 = (float)(int8_t)((pkf >> 8) & 0xFF),
                f2 = (float)(int8_t)((pkf >> 16) & 0xFF), f3 = (float)(int8_t)((pkf >> 24) & 0xFF);

#pragma unroll
    for (int m = 0; m < MAX_M; ++m) {
        if (m < M) {
            const float4 gv = reinterpret_cast<const float4*>(s_gate[m])[lg];
            float ag = g0 * gv.x + g1 * gv.y + g2 * gv.z + g3 * gv.w;
            float af = f0 * gv.x + f1 * gv.y + f2 * gv.z + f3 * gv.w;
#pragma unroll
            for (int o = 8; o >= 1; o >>= 1) {
                ag += __shfl_xor_sync(0xFFFFFFFFu, ag, o);
                af += __shfl_xor_sync(0xFFFFFFFFu, af, o);
            }
            if (lg == 0) {
                const size_t idx = (size_t)m * D + d;
                out[idx] = (1.0f - tanhf(af * scf)) * cur[idx] + tanhf(ag * scg) * msg[idx];
            }
        }
    }
}

/// 3x Fused Q8_ROW Loop Controller Down Projection (attn, k, v in 1 launch!)
template <int MAX_M>
__global__ void __launch_bounds__(256) k_ink_ctrl_3x_down_q8_smallM(
    const float* __restrict__ cat3,  // [3, M, D2]
    const uint8_t* __restrict__ wd0, const uint8_t* __restrict__ wd1, const uint8_t* __restrict__ wd2,
    float* __restrict__ gate3,       // [3, M, 64]
    int M, int D2) {
    const int r = blockIdx.x;
    const int c = blockIdx.y;
    if (r >= kInkGateRank || c >= 3) return;
    const int tid = threadIdx.x;
    const int D2_4 = D2 >> 2;
    const uint8_t* wd = (c == 0) ? wd0 : (c == 1 ? wd1 : wd2);
    const float* cat  = cat3 + (size_t)c * M * D2;
    float* gate       = gate3 + (size_t)c * M * kInkGateRank;
    const float wsc   = reinterpret_cast<const float*>(wd)[r];
    const int32_t* wr = reinterpret_cast<const int32_t*>(wd + (size_t)kInkGateRank * 4u + (size_t)r * D2);
    float acc[MAX_M] = {};
    for (int i = tid; i < D2_4; i += 256) {
        const int32_t pk = wr[i];
        const float w0 = (float)(int8_t)(pk & 0xFF);
        const float w1 = (float)(int8_t)((pk >> 8) & 0xFF);
        const float w2 = (float)(int8_t)((pk >> 16) & 0xFF);
        const float w3 = (float)(int8_t)((pk >> 24) & 0xFF);
#pragma unroll
        for (int m = 0; m < MAX_M; ++m) {
            if (m < M) {
                const float4 x = reinterpret_cast<const float4*>(cat + (size_t)m * D2)[i];
                acc[m] = fmaf(w0, x.x, acc[m]);
                acc[m] = fmaf(w1, x.y, acc[m]);
                acc[m] = fmaf(w2, x.z, acc[m]);
                acc[m] = fmaf(w3, x.w, acc[m]);
            }
        }
    }
    __shared__ float sh[32];
#pragma unroll
    for (int m = 0; m < MAX_M; ++m) {
        if (m < M) {
            const float s = block_sum(acc[m], sh);
            if (tid == 0) gate[(size_t)m * kInkGateRank + r] = s * wsc;
        }
    }
}

/// 3x Fused Q8_ROW Loop Controller Apply (attn, k, v in 1 launch!)
template <int MAX_M>
__global__ void __launch_bounds__(256) k_ink_ctrl_3x_apply_q8_smallM(
    const float* __restrict__ cur0, const float* __restrict__ cur1, const float* __restrict__ cur2,
    const float* __restrict__ msgs,   // [3, M, D]
    const float* __restrict__ gate3,  // [3, M, 64]
    const uint8_t* __restrict__ wg0, const uint8_t* __restrict__ wg1, const uint8_t* __restrict__ wg2,
    const uint8_t* __restrict__ wf0, const uint8_t* __restrict__ wf1, const uint8_t* __restrict__ wf2,
    float* __restrict__ out0, float* __restrict__ out1, float* __restrict__ out2,
    int M, int D) {
    const int c = blockIdx.y;
    if (c >= 3) return;
    __shared__ float s_gate[MAX_M][kInkGateRank];
    const int tid = threadIdx.x;
    const float* gate = gate3 + (size_t)c * M * kInkGateRank;
    for (int i = tid; i < M * kInkGateRank; i += 256) s_gate[i >> 6][i & 63] = gate[i];
    __syncthreads();

    const int sub = tid >> 4;
    const int lg  = tid & 15;
    const int d   = blockIdx.x * 16 + sub;
    if (d >= D) return;

    const float* cur   = (c == 0) ? cur0 : (c == 1 ? cur1 : cur2);
    const float* msg   = msgs + (size_t)c * M * D;
    const uint8_t* wg  = (c == 0) ? wg0 : (c == 1 ? wg1 : wg2);
    const uint8_t* wf  = (c == 0) ? wf0 : (c == 1 ? wf1 : wf2);
    float* out         = (c == 0) ? out0 : (c == 1 ? out1 : out2);

    const float scg = reinterpret_cast<const float*>(wg)[d];
    const float scf = reinterpret_cast<const float*>(wf)[d];
    const int32_t pkg = reinterpret_cast<const int32_t*>(wg + (size_t)D * 4u + (size_t)d * kInkGateRank)[lg];
    const int32_t pkf = reinterpret_cast<const int32_t*>(wf + (size_t)D * 4u + (size_t)d * kInkGateRank)[lg];
    const float g0 = (float)(int8_t)(pkg & 0xFF),        g1 = (float)(int8_t)((pkg >> 8) & 0xFF),
                g2 = (float)(int8_t)((pkg >> 16) & 0xFF), g3 = (float)(int8_t)((pkg >> 24) & 0xFF);
    const float f0 = (float)(int8_t)(pkf & 0xFF),        f1 = (float)(int8_t)((pkf >> 8) & 0xFF),
                f2 = (float)(int8_t)((pkf >> 16) & 0xFF), f3 = (float)(int8_t)((pkf >> 24) & 0xFF);

#pragma unroll
    for (int m = 0; m < MAX_M; ++m) {
        if (m < M) {
            const float4 gv = reinterpret_cast<const float4*>(s_gate[m])[lg];
            float ag = g0 * gv.x + g1 * gv.y + g2 * gv.z + g3 * gv.w;
            float af = f0 * gv.x + f1 * gv.y + f2 * gv.z + f3 * gv.w;
#pragma unroll
            for (int o = 8; o >= 1; o >>= 1) {
                ag += __shfl_xor_sync(0xFFFFFFFFu, ag, o);
                af += __shfl_xor_sync(0xFFFFFFFFu, af, o);
            }
            if (lg == 0) {
                const size_t idx = (size_t)m * D + d;
                out[idx] = (1.0f - tanhf(af * scf)) * cur[idx] + tanhf(ag * scg) * msg[idx];
            }
        }
    }
}

/// Ultra-fast 1-pass Q8_ROW unembed exact argmax for small M <= 8 (~1.0 ms @ 448 GB/s GDDR7 peak, ZERO 32x cuBLAS overhead!).
/// Grid = 256 CTAs (each scans V/256 vocab rows), Block = 256 (8 warps). Writes [M, 256] partial maxes -> P2 final reduce!
template <int MAX_M>
__global__ void __launch_bounds__(256) k_ink_q8row_argmax_smallm_p1(
    const int8_t* __restrict__ w_q,    // [V_pad, 2304]
    const float*  __restrict__ w_sc,   // [V_pad]
    const int8_t* __restrict__ x_q,    // [M, 2304]
    int M, int V,
    float*   __restrict__ out_val,     // [M, gridDim.x]
    int32_t* __restrict__ out_id) {    // [M, gridDim.x]
    __shared__ alignas(16) int8_t s_x[MAX_M][kInkHidden];
    const int tid  = threadIdx.x;
    const int warp = tid >> 5;
    const int lane = tid & 31;
    for (int m = 0; m < M; ++m) {
        const int4* sx = reinterpret_cast<const int4*>(x_q + (size_t)m * kInkHidden);
        int4* dx = reinterpret_cast<int4*>(s_x[m]);
        for (int i = tid; i < (kInkHidden / 16); i += 256) dx[i] = sx[i];
    }
    __syncthreads();

    float   bv[MAX_M];
    int32_t bi[MAX_M];
#pragma unroll
    for (int m = 0; m < MAX_M; ++m) { bv[m] = -3e38f; bi[m] = 0; }

    const int row_stride = gridDim.x * 8;
    const int row0 = blockIdx.x * 8 + warp;
    constexpr int K16 = kInkHidden / 16;  // 144
    for (int row = row0; row < V; row += row_stride) {
        const int4* w4 = reinterpret_cast<const int4*>(w_q + (size_t)row * kInkHidden);
        int32_t acc[MAX_M] = {};
        for (int i = lane; i < K16; i += 32) {
            const int4 wv = w4[i];
#pragma unroll
            for (int m = 0; m < MAX_M; ++m) {
                if (m < M) {
                    const int4 xv = reinterpret_cast<const int4*>(s_x[m])[i];
                    acc[m] = dp4a_i8(wv.x, xv.x, acc[m]);
                    acc[m] = dp4a_i8(wv.y, xv.y, acc[m]);
                    acc[m] = dp4a_i8(wv.z, xv.z, acc[m]);
                    acc[m] = dp4a_i8(wv.w, xv.w, acc[m]);
                }
            }
        }
        const float ws = w_sc[row];
#pragma unroll
        for (int m = 0; m < MAX_M; ++m) {
            if (m < M) {
                int32_t a = acc[m];
#pragma unroll
                for (int o = 16; o >= 1; o >>= 1) a += __shfl_xor_sync(0xFFFFFFFFu, a, o);
                if (lane == 0) {
                    const float v = (float)a * ws;
                    if (v > bv[m] || (v == bv[m] && row < bi[m])) { bv[m] = v; bi[m] = row; }
                }
            }
        }
    }
    __shared__ float   sm_v[MAX_M][8];
    __shared__ int32_t sm_i[MAX_M][8];
    if (lane == 0) {
#pragma unroll
        for (int m = 0; m < MAX_M; ++m) { sm_v[m][warp] = bv[m]; sm_i[m][warp] = bi[m]; }
    }
    __syncthreads();
    if (tid < M) {
        float best = sm_v[tid][0];
        int32_t best_i = sm_i[tid][0];
        for (int w = 1; w < 8; ++w) {
            if (sm_v[tid][w] > best || (sm_v[tid][w] == best && sm_i[tid][w] < best_i)) {
                best = sm_v[tid][w]; best_i = sm_i[tid][w];
            }
        }
        out_val[(size_t)tid * gridDim.x + blockIdx.x] = best;
        out_id [(size_t)tid * gridDim.x + blockIdx.x] = best_i;
    }
}

/// Pass 2 final argmax over n_blk <= 256 blocks -> d_argmax[M]! Grid = M, Block = 256.
__global__ void __launch_bounds__(256) k_ink_q8row_argmax_smallm_p2(
    const float*   __restrict__ blk_val,  // [M, n_blk]
    const int32_t* __restrict__ blk_id,   // [M, n_blk]
    int n_blk,
    int32_t* __restrict__ d_argmax) {     // [M]
    const int m = blockIdx.x;
    const int tid = threadIdx.x;
    const int warp = tid >> 5;
    const int lane = tid & 31;
    float bv = (tid < n_blk) ? blk_val[(size_t)m * n_blk + tid] : -3e38f;
    int32_t bi = (tid < n_blk) ? blk_id[(size_t)m * n_blk + tid] : 0;
#pragma unroll
    for (int o = 16; o >= 1; o >>= 1) {
        const float ov = __shfl_xor_sync(0xFFFFFFFFu, bv, o);
        const int32_t oi = __shfl_xor_sync(0xFFFFFFFFu, bi, o);
        if (ov > bv || (ov == bv && oi < bi)) { bv = ov; bi = oi; }
    }
    __shared__ float sm_v[8];
    __shared__ int32_t sm_i[8];
    if (lane == 0) { sm_v[warp] = bv; sm_i[warp] = bi; }
    __syncthreads();
    if (tid == 0) {
        float best = sm_v[0]; int32_t best_i = sm_i[0];
        for (int w = 1; w < 8; ++w) {
            if (sm_v[w] > best || (sm_v[w] == best && sm_i[w] < best_i)) { best = sm_v[w]; best_i = sm_i[w]; }
        }
        d_argmax[m] = best_i;
    }
}

/// Fused 1-Pass Register-Cached Loop Controller Cat+RMS+RowI8 Quantize: reads cur & msg ONCE (3x less DRAM!)
__global__ void __launch_bounds__(256) k_ink_ctrl_cat_rms_i8(
    const float* __restrict__ cur,   // [M, D]
    const float* __restrict__ msg,   // [M, D]
    int8_t* __restrict__ x_i8,       // [M, 2*D]
    float*  __restrict__ x_sc,       // [M]
    int M, int D) {
    const int m = blockIdx.x;
    if (m >= M) return;
    const int tid  = threadIdx.x;
    const int warp = tid >> 5;
    const int lane = tid & 31;
    const float* pc = cur + (size_t)m * D;
    const float* pm = msg + (size_t)m * D;

    float vc[9], vm[9];
    float sc = 0.0f, sm = 0.0f;
#pragma unroll
    for (int k = 0; k < 9; ++k) {
        const int d = k * 256 + tid;
        const float c = (d < D) ? pc[d] : 0.0f;
        const float v = (d < D) ? pm[d] : 0.0f;
        vc[k] = c; vm[k] = v;
        sc += c * c;
        sm += v * v;
    }
    __shared__ float sh_c[32], sh_m[32];
    sc = block_sum(sc, sh_c);
    sm = block_sum(sm, sh_m);
    const float inv_c = rsqrtf(sc / (float)D + kInkRmsEps);
    const float inv_m = rsqrtf(sm / (float)D + kInkRmsEps);

    float am = 0.0f;
#pragma unroll
    for (int k = 0; k < 9; ++k) {
        vc[k] *= inv_c;
        vm[k] *= inv_m;
        am = fmaxf(am, fmaxf(fabsf(vc[k]), fabsf(vm[k])));
    }
    am = warp_max(am);
    __shared__ float sh_am[8];
    __shared__ float s_sc;
    if (lane == 0) sh_am[warp] = am;
    __syncthreads();
    if (tid == 0) {
        float g = sh_am[0];
        for (int i = 1; i < 8; ++i) g = fmaxf(g, sh_am[i]);
        s_sc = (g > 0.0f) ? (g * (1.0f / 127.0f)) : 1.0f;
        x_sc[m] = s_sc;
    }
    __syncthreads();
    const float q_inv = (s_sc > 0.0f) ? (1.0f / s_sc) : 0.0f;
    int8_t* dst = x_i8 + (size_t)m * (2 * D);
#pragma unroll
    for (int k = 0; k < 9; ++k) {
        const int d = k * 256 + tid;
        if (d < D) {
            dst[d]     = (int8_t)max(-127, min(127, __float2int_rn(vc[k] * q_inv)));
            dst[D + d] = (int8_t)max(-127, min(127, __float2int_rn(vm[k] * q_inv)));
        }
    }
}

/// Fused 100%-Coalesced DP4A Large-M Q8_ROW Loop Controller Apply+Blend!
/// Grid = dim3((D + 31) / 32, (M + 7) / 8), Block = 256 (8 warps = 8 tokens, 32 lanes = 32 contiguous dims!).
/// Loads w_gain[32, 64] & w_forget[32, 64] ONCE into 4 KiB SMEM coalesced, quantizes gate[8, 64] to int8 ONCE,
/// then computes 16x __dp4a + 100%-coalesced cur/msg/out reads & writes! (13.6 ms -> ~1.5 ms!)
__global__ void __launch_bounds__(256) k_ink_ctrl_apply_q8_tile16(
    const float* __restrict__ cur,     // [M, D]
    const float* __restrict__ msg,     // [M, D]
    const float* __restrict__ gate,    // [M, 64]
    const uint8_t* __restrict__ wg_q8, // FMT_Q8ROW [D, 64]
    const uint8_t* __restrict__ wf_q8, // FMT_Q8ROW [D, 64]
    float* __restrict__ out,           // [M, D]
    int M, int D) {
    const int d0 = blockIdx.x * 32;
    const int m0 = blockIdx.y * 8;
    const int tid  = threadIdx.x;
    const int warp = tid >> 5;  // 0..7 (token offset within tile)
    const int lane = tid & 31;  // 0..31 (dim offset within tile)

    __shared__ alignas(16) int8_t s_wg[32][kInkGateRank];  // 2 KiB
    __shared__ alignas(16) int8_t s_wf[32][kInkGateRank];  // 2 KiB
    __shared__ float s_scg[32], s_scf[32];
    __shared__ alignas(16) int8_t s_gq[8][kInkGateRank];   // 512 B
    __shared__ float s_gsc[8];

    // 1. Cooperatively load 32 rows x 64 int8 weights (coalesced uint2 = 8 B/thread x 256 = 2 KiB in 1 pass!)
    const int wr_row = tid >> 3;  // 0..31
    const int wr_col = (tid & 7) * 8;
    const int g_d = d0 + wr_row;
    if (g_d < D) {
        *reinterpret_cast<uint2*>(&s_wg[wr_row][wr_col]) =
            *reinterpret_cast<const uint2*>(wg_q8 + (size_t)D * 4u + (size_t)g_d * kInkGateRank + wr_col);
        *reinterpret_cast<uint2*>(&s_wf[wr_row][wr_col]) =
            *reinterpret_cast<const uint2*>(wf_q8 + (size_t)D * 4u + (size_t)g_d * kInkGateRank + wr_col);
    }
    if (tid < 32 && (d0 + tid) < D) {
        s_scg[tid] = reinterpret_cast<const float*>(wg_q8)[d0 + tid];
        s_scf[tid] = reinterpret_cast<const float*>(wf_q8)[d0 + tid];
    }

    // 2. Each warp quantizes its 1 token's gate[64] -> int8[64] + scale in SMEM (2 elems/lane!)
    const int m = m0 + warp;
    if (m < M) {
        const float* gp = gate + (size_t)m * kInkGateRank;
        const float v0 = gp[2 * lane];
        const float v1 = gp[2 * lane + 1];
        const float am = warp_max(fmaxf(fabsf(v0), fabsf(v1)));
        const float gsc = (am > 0.0f) ? (am * (1.0f / 127.0f)) : 0.0f;
        const float inv = (am > 0.0f) ? (127.0f / am) : 0.0f;
        s_gq[warp][2 * lane]     = (int8_t)max(-127, min(127, __float2int_rn(v0 * inv)));
        s_gq[warp][2 * lane + 1] = (int8_t)max(-127, min(127, __float2int_rn(v1 * inv)));
        if (lane == 0) s_gsc[warp] = gsc;
    }
    __syncthreads();

    const int d = d0 + lane;
    if (m >= M || d >= D) return;

    // 3. 16x __dp4a in registers from SMEM -> 100%-coalesced cur/msg/out load+store!
    const int32_t* wg4 = reinterpret_cast<const int32_t*>(s_wg[lane]);
    const int32_t* wf4 = reinterpret_cast<const int32_t*>(s_wf[lane]);
    const int32_t* gq4 = reinterpret_cast<const int32_t*>(s_gq[warp]);
    int32_t ag = 0, af = 0;
#pragma unroll
    for (int k = 0; k < 16; ++k) {
        const int32_t gv = gq4[k];
        ag = dp4a_i8(wg4[k], gv, ag);
        af = dp4a_i8(wf4[k], gv, af);
    }
    const float gs = s_gsc[warp];
    const size_t idx = (size_t)m * D + d;
    out[idx] = (1.0f - tanhf((float)af * s_scf[lane] * gs)) * cur[idx] + tanhf((float)ag * s_scg[lane] * gs) * msg[idx];
}

/// Fused 1-Pass Register-Cached 3x Cat+RMS+RowI8 Quantize for Pre-Attn Controllers (reads cur3 & msg3 ONCE!)
/// Grid = dim3(M, 3), Block = 256.
__global__ void __launch_bounds__(256) k_ink_ctrl_3x_cat_rms_i8(
    const float* __restrict__ cur3,   // [3, M, D] (d_mixed)
    const float* __restrict__ msg3,   // [3, M, D] (d_msgs)
    int8_t* __restrict__ x_i8_3,      // [3, M_pad, 2*D]
    float*  __restrict__ x_sc_3,      // [3, M]
    int M, int M_pad, int D) {
    const int m = blockIdx.x;
    const int c = blockIdx.y;
    if (m >= M || c >= 3) return;
    const int tid  = threadIdx.x;
    const int warp = tid >> 5;
    const int lane = tid & 31;
    const float* pc = cur3 + ((size_t)c * M + m) * D;
    const float* pm = msg3 + ((size_t)c * M + m) * D;

    float vc[9], vm[9];
    float sc = 0.0f, sm = 0.0f;
#pragma unroll
    for (int k = 0; k < 9; ++k) {
        const int d = k * 256 + tid;
        const float cv = (d < D) ? pc[d] : 0.0f;
        const float mv = (d < D) ? pm[d] : 0.0f;
        vc[k] = cv; vm[k] = mv;
        sc += cv * cv;
        sm += mv * mv;
    }
    __shared__ float sh_c[32], sh_m[32];
    sc = block_sum(sc, sh_c);
    sm = block_sum(sm, sh_m);
    const float inv_c = rsqrtf(sc / (float)D + kInkRmsEps);
    const float inv_m = rsqrtf(sm / (float)D + kInkRmsEps);

    float am = 0.0f;
#pragma unroll
    for (int k = 0; k < 9; ++k) {
        vc[k] *= inv_c;
        vm[k] *= inv_m;
        am = fmaxf(am, fmaxf(fabsf(vc[k]), fabsf(vm[k])));
    }
    am = warp_max(am);
    __shared__ float sh_am[8];
    __shared__ float s_sc;
    if (lane == 0) sh_am[warp] = am;
    __syncthreads();
    if (tid == 0) {
        float g = sh_am[0];
        for (int i = 1; i < 8; ++i) g = fmaxf(g, sh_am[i]);
        s_sc = (g > 0.0f) ? (g * (1.0f / 127.0f)) : 1.0f;
        x_sc_3[(size_t)c * M + m] = s_sc;
    }
    __syncthreads();
    const float q_inv = (s_sc > 0.0f) ? (1.0f / s_sc) : 0.0f;
    int8_t* dst = x_i8_3 + ((size_t)c * M_pad + m) * (2 * D);
#pragma unroll
    for (int k = 0; k < 9; ++k) {
        const int d = k * 256 + tid;
        if (d < D) {
            dst[d]     = (int8_t)max(-127, min(127, __float2int_rn(vc[k] * q_inv)));
            dst[D + d] = (int8_t)max(-127, min(127, __float2int_rn(vm[k] * q_inv)));
        }
    }
}

/// 1-Pass Register-Cached 3x In-Place RMSNorm (reads each vector ONCE instead of twice!)
__global__ void __launch_bounds__(256) k_ink_rms_norm_3x_fast(
    float* __restrict__ x0, float* __restrict__ x1, float* __restrict__ x2,
    const float* __restrict__ w0, const float* __restrict__ w1, const float* __restrict__ w2,
    int M, int D) {
    const int m = blockIdx.x;
    const int c = blockIdx.y;
    if (m >= M || c >= 3) return;
    const int tid = threadIdx.x;
    float*       px = ((c == 0) ? x0 : (c == 1 ? x1 : x2)) + (size_t)m * D;
    const float* pw =  (c == 0) ? w0 : (c == 1 ? w1 : w2);
    float vv[9];
    float ss = 0.0f;
#pragma unroll
    for (int k = 0; k < 9; ++k) {
        const int d = k * 256 + tid;
        const float v = (d < D) ? px[d] : 0.0f;
        vv[k] = v;
        ss += v * v;
    }
    __shared__ float sh[32];
    ss = block_sum(ss, sh);
    const float inv = rsqrtf(ss / (float)D + kInkRmsEps);
#pragma unroll
    for (int k = 0; k < 9; ++k) {
        const int d = k * 256 + tid;
        if (d < D) px[d] = vv[k] * inv * pw[d];
    }
}

// ============================================================================
// 9. Fused Hardware-FP8->FP16x2 Register-Q Flash-Attention-2 Prefill Kernel (ZERO d_pf_s/d_pf_p DRAM!)
//    Grid = dim3((M + 63) / 64, kInkHeads), Block = 128 (4 warps x 16 Q-rows = 64 Q-rows/CTA!).
//    Uses single-instruction __nv_cvt_fp8x2_to_halfraw2 + register-resident Q + rt_far fast-path (2.1x faster!).
// ============================================================================
__device__ __forceinline__ __half2 ink_cvt_e4m3x2_f16x2(uint16_t raw2, __half2 sc2) {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 890)
    __nv_fp8x2_storage_t s = raw2;
    __half2_raw h = __nv_cvt_fp8x2_to_halfraw2(s, __NV_E4M3);
    return __hmul2(*reinterpret_cast<const __half2*>(&h), sc2);
#else
    float f0 = e4m3_to_f32((uint8_t)(raw2 & 0xFFu));
    float f1 = e4m3_to_f32((uint8_t)(raw2 >> 8));
    return __hmul2(__float22half2_rn(make_float2(f0, f1)), sc2);
#endif
}

__global__ void __launch_bounds__(128) k_ink_flash_pf_mma(
    const float* __restrict__ q,          // [M, 24, 128]
    InkKvVisitDesc kv,
    const float* __restrict__ rel_table,  // [M, 24, 1025]
    const float* __restrict__ sinks,      // [M, 24]
    const float* __restrict__ gate_attn,  // [M, 24, 128]
    int slot0, int p0, int M, bool is_swa,
    float* __restrict__ out) {            // [M, 24, 128]
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 800)
    constexpr int kStr = 136;
    __shared__ alignas(16) __half s_k[32][kStr];
    __shared__ alignas(16) __half s_v[32][kStr];

    const int q_cta = blockIdx.x * 64;
    const int h     = blockIdx.y;
    const int kh    = h / 3;
    if (q_cta >= M || h >= kInkHeads) return;

    const int tid  = threadIdx.x;
    const int warp = tid >> 5;
    const int lane = tid & 31;
    const int g    = lane >> 2;
    const int tig  = lane & 3;
    const int nq   = min(64, M - q_cta);

    constexpr float scale = 0.08838834764831845f;
    const int r0 = q_cta + warp * 16 + g;
    const int r1 = r0 + 8;
    uint32_t qa[8][4];
    const float* qp_row0 = q + ((size_t)min(r0, M - 1) * kInkHeads + h) * kInkHeadDim;
    const float* qp_row1 = q + ((size_t)min(r1, M - 1) * kInkHeads + h) * kInkHeadDim;
#pragma unroll
    for (int ks = 0; ks < 8; ++ks) {
        const int c0 = ks * 16 + 2 * tig;
        const int c1 = c0 + 8;
        if (r0 < M) {
            __half2 h0 = __float22half2_rn(make_float2(qp_row0[c0] * scale, qp_row0[c0 + 1] * scale));
            __half2 h2 = __float22half2_rn(make_float2(qp_row0[c1] * scale, qp_row0[c1 + 1] * scale));
            qa[ks][0] = *reinterpret_cast<const uint32_t*>(&h0);
            qa[ks][2] = *reinterpret_cast<const uint32_t*>(&h2);
        } else { qa[ks][0] = 0u; qa[ks][2] = 0u; }
        if (r1 < M) {
            __half2 h1 = __float22half2_rn(make_float2(qp_row1[c0] * scale, qp_row1[c0 + 1] * scale));
            __half2 h3 = __float22half2_rn(make_float2(qp_row1[c1] * scale, qp_row1[c1 + 1] * scale));
            qa[ks][1] = *reinterpret_cast<const uint32_t*>(&h1);
            qa[ks][3] = *reinterpret_cast<const uint32_t*>(&h3);
        } else { qa[ks][1] = 0u; qa[ks][3] = 0u; }
    }

    const int qp0 = p0 + r0;
    const int qp1 = p0 + r1;
    const float* rt0 = rel_table + ((size_t)min(r0, M - 1) * kInkHeads + h) * (kInkRelExtent + 1);
    const float* rt1 = rel_table + ((size_t)min(r1, M - 1) * kInkHeads + h) * (kInkRelExtent + 1);
    const float rt0_far = (r0 < M) ? rt0[kInkRelExtent] : 0.0f;
    const float rt1_far = (r1 < M) ? rt1[kInkRelExtent] : 0.0f;

    float m0 = (r0 < M) ? sinks[(size_t)r0 * kInkHeads + h] : -1e30f;
    float m1 = (r1 < M) ? sinks[(size_t)r1 * kInkHeads + h] : -1e30f;
    float l0 = 1.0f, l1 = 1.0f;
    float o[16][4] = {};

    const int k_beg = is_swa ? max(0, (p0 + q_cta) - kInkSwaWin + 1) : 0;
    const int k_end = p0 + q_cta + nq;
    const int kv_stride = (kv.host_cap > 0) ? kv.dev_cap : kv.cap_pos;

    auto pack_f16x2 = [](float a, float b) -> uint32_t {
        __half2 p = __float22half2_rn(make_float2(a, b));
        return *reinterpret_cast<const uint32_t*>(&p);
    };

    for (int kt = k_beg; kt < k_end; kt += 32) {
        const int nt = min(32, k_end - kt);
        for (int j = warp; j < 32; j += 4) {
            if (j < nt) {
                const int kp = kt + j;
                const int p_idx = is_swa ? (kp % kv.cap_pos) : kp;
                if (kv.kv_fmt == 1 && (is_swa || kv.host_cap == 0 || p_idx < kv.dev_cap)) {
                    const size_t hb = ((size_t)slot0 * kv_stride + p_idx) * kInkKVHeads + kh;
                    const size_t b4 = hb * kInkHeadDim + lane * 4;
                    const uint32_t k4 = *reinterpret_cast<const uint32_t*>(kv.k_f8 + b4);
                    const uint32_t v4 = *reinterpret_cast<const uint32_t*>(kv.v_f8 + b4);
                    const __half2 ksc = __float2half2_rn(kv.k_sc[hb]);
                    const __half2 vsc = __float2half2_rn(kv.v_sc[hb]);
                    *reinterpret_cast<__half2*>(&s_k[j][lane * 4 + 0]) = ink_cvt_e4m3x2_f16x2((uint16_t)(k4 & 0xFFFFu), ksc);
                    *reinterpret_cast<__half2*>(&s_k[j][lane * 4 + 2]) = ink_cvt_e4m3x2_f16x2((uint16_t)(k4 >> 16), ksc);
                    *reinterpret_cast<__half2*>(&s_v[j][lane * 4 + 0]) = ink_cvt_e4m3x2_f16x2((uint16_t)(v4 & 0xFFFFu), vsc);
                    *reinterpret_cast<__half2*>(&s_v[j][lane * 4 + 2]) = ink_cvt_e4m3x2_f16x2((uint16_t)(v4 >> 16), vsc);
                } else {
                    float4 kv4, vv4;
                    ink_kv_load4(kv, slot0, p_idx, kh, lane, kv4, vv4);
                    *reinterpret_cast<__half2*>(&s_k[j][lane * 4 + 0]) = __float22half2_rn(make_float2(kv4.x, kv4.y));
                    *reinterpret_cast<__half2*>(&s_k[j][lane * 4 + 2]) = __float22half2_rn(make_float2(kv4.z, kv4.w));
                    *reinterpret_cast<__half2*>(&s_v[j][lane * 4 + 0]) = __float22half2_rn(make_float2(vv4.x, vv4.y));
                    *reinterpret_cast<__half2*>(&s_v[j][lane * 4 + 2]) = __float22half2_rn(make_float2(vv4.z, vv4.w));
                }
            } else {
                *reinterpret_cast<uint2*>(&s_k[j][lane * 4]) = make_uint2(0u, 0u);
                *reinterpret_cast<uint2*>(&s_v[j][lane * 4]) = make_uint2(0u, 0u);
            }
        }
        __syncthreads();

        float s[4][4] = {};
#pragma unroll
        for (int ks = 0; ks < 8; ++ks) {
#pragma unroll
            for (int n = 0; n < 4; ++n) {
                uint32_t b0, b1;
                const uint32_t b_addr = (uint32_t)__cvta_generic_to_shared(&s_k[n * 8 + (lane & 7)][ks * 16 + (((lane >> 3) & 1) << 3)]);
                asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0, %1}, [%2];\n"
                             : "=r"(b0), "=r"(b1) : "r"(b_addr));
                asm volatile(
                    "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
                    "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                    : "+f"(s[n][0]), "+f"(s[n][1]), "+f"(s[n][2]), "+f"(s[n][3])
                    : "r"(qa[ks][0]), "r"(qa[ks][1]), "r"(qa[ks][2]), "r"(qa[ks][3]), "r"(b0), "r"(b1));
            }
        }

        const bool all_far = (!is_swa) && (kt + 31 <= qp0 - kInkRelExtent);
        float rm0 = -1e30f, rm1 = -1e30f;
        if (all_far) {
#pragma unroll
            for (int n = 0; n < 4; ++n) {
                s[n][0] = (r0 < M) ? (s[n][0] + rt0_far) : -1e30f;
                s[n][1] = (r0 < M) ? (s[n][1] + rt0_far) : -1e30f;
                s[n][2] = (r1 < M) ? (s[n][2] + rt1_far) : -1e30f;
                s[n][3] = (r1 < M) ? (s[n][3] + rt1_far) : -1e30f;
                rm0 = fmaxf(rm0, fmaxf(s[n][0], s[n][1]));
                rm1 = fmaxf(rm1, fmaxf(s[n][2], s[n][3]));
            }
        } else {
#pragma unroll
            for (int n = 0; n < 4; ++n) {
                const int k0 = n * 8 + 2 * tig;
                const int kp_a = kt + k0, kp_b = kt + k0 + 1;
                const bool v0a = (r0 < M) && (k0 < nt)     && (kp_a <= qp0) && (!is_swa || (qp0 - kp_a) < kInkSwaWin);
                const bool v0b = (r0 < M) && (k0 + 1 < nt) && (kp_b <= qp0) && (!is_swa || (qp0 - kp_b) < kInkSwaWin);
                const bool v1a = (r1 < M) && (k0 < nt)     && (kp_a <= qp1) && (!is_swa || (qp1 - kp_a) < kInkSwaWin);
                const bool v1b = (r1 < M) && (k0 + 1 < nt) && (kp_b <= qp1) && (!is_swa || (qp1 - kp_b) < kInkSwaWin);
                s[n][0] = v0a ? (s[n][0] + rt0[min(qp0 - kp_a, kInkRelExtent)]) : -1e30f;
                s[n][1] = v0b ? (s[n][1] + rt0[min(qp0 - kp_b, kInkRelExtent)]) : -1e30f;
                s[n][2] = v1a ? (s[n][2] + rt1[min(qp1 - kp_a, kInkRelExtent)]) : -1e30f;
                s[n][3] = v1b ? (s[n][3] + rt1[min(qp1 - kp_b, kInkRelExtent)]) : -1e30f;
                rm0 = fmaxf(rm0, fmaxf(s[n][0], s[n][1]));
                rm1 = fmaxf(rm1, fmaxf(s[n][2], s[n][3]));
            }
        }
        rm0 = fmaxf(rm0, __shfl_xor_sync(0xFFFFFFFFu, rm0, 1));
        rm0 = fmaxf(rm0, __shfl_xor_sync(0xFFFFFFFFu, rm0, 2));
        rm1 = fmaxf(rm1, __shfl_xor_sync(0xFFFFFFFFu, rm1, 1));
        rm1 = fmaxf(rm1, __shfl_xor_sync(0xFFFFFFFFu, rm1, 2));

        const float m0_new = fmaxf(m0, rm0);
        const float m1_new = fmaxf(m1, rm1);
        const float al0 = (m0_new > -1e29f) ? __expf(m0 - m0_new) : 0.0f;
        const float al1 = (m1_new > -1e29f) ? __expf(m1 - m1_new) : 0.0f;
        m0 = m0_new; m1 = m1_new;
#pragma unroll
        for (int j = 0; j < 16; ++j) { o[j][0] *= al0; o[j][1] *= al0; o[j][2] *= al1; o[j][3] *= al1; }
        float ls0 = 0.0f, ls1 = 0.0f;
        float p[4][4];
#pragma unroll
        for (int n = 0; n < 4; ++n) {
            p[n][0] = (s[n][0] > -1e29f) ? __expf(s[n][0] - m0) : 0.0f;
            p[n][1] = (s[n][1] > -1e29f) ? __expf(s[n][1] - m0) : 0.0f;
            p[n][2] = (s[n][2] > -1e29f) ? __expf(s[n][2] - m1) : 0.0f;
            p[n][3] = (s[n][3] > -1e29f) ? __expf(s[n][3] - m1) : 0.0f;
            ls0 += p[n][0] + p[n][1]; ls1 += p[n][2] + p[n][3];
        }
        ls0 += __shfl_xor_sync(0xFFFFFFFFu, ls0, 1); ls0 += __shfl_xor_sync(0xFFFFFFFFu, ls0, 2);
        ls1 += __shfl_xor_sync(0xFFFFFFFFu, ls1, 1); ls1 += __shfl_xor_sync(0xFFFFFFFFu, ls1, 2);
        l0 = l0 * al0 + ls0; l1 = l1 * al1 + ls1;

        const uint32_t pa0_0 = pack_f16x2(p[0][0], p[0][1]), pa0_1 = pack_f16x2(p[0][2], p[0][3]);
        const uint32_t pa0_2 = pack_f16x2(p[1][0], p[1][1]), pa0_3 = pack_f16x2(p[1][2], p[1][3]);
        const uint32_t pa1_0 = pack_f16x2(p[2][0], p[2][1]), pa1_1 = pack_f16x2(p[2][2], p[2][3]);
        const uint32_t pa1_2 = pack_f16x2(p[3][0], p[3][1]), pa1_3 = pack_f16x2(p[3][2], p[3][3]);
#pragma unroll
        for (int j = 0; j < 16; ++j) {
            const int d0 = j * 8;
            uint32_t vb0, vb1;
            const uint32_t v0_addr = (uint32_t)__cvta_generic_to_shared(&s_v[lane & 15][d0]);
            asm volatile("ldmatrix.sync.aligned.m8n8.x2.trans.shared.b16 {%0, %1}, [%2];\n"
                         : "=r"(vb0), "=r"(vb1) : "r"(v0_addr));
            asm volatile(
                "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
                "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                : "+f"(o[j][0]), "+f"(o[j][1]), "+f"(o[j][2]), "+f"(o[j][3])
                : "r"(pa0_0), "r"(pa0_1), "r"(pa0_2), "r"(pa0_3), "r"(vb0), "r"(vb1));
            if (nt > 16) {
                const uint32_t v1_addr = (uint32_t)__cvta_generic_to_shared(&s_v[16 + (lane & 15)][d0]);
                asm volatile("ldmatrix.sync.aligned.m8n8.x2.trans.shared.b16 {%0, %1}, [%2];\n"
                             : "=r"(vb0), "=r"(vb1) : "r"(v1_addr));
                asm volatile(
                    "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
                    "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                    : "+f"(o[j][0]), "+f"(o[j][1]), "+f"(o[j][2]), "+f"(o[j][3])
                    : "r"(pa1_0), "r"(pa1_1), "r"(pa1_2), "r"(pa1_3), "r"(vb0), "r"(vb1));
            }
        }
        __syncthreads();
    }

    const float inv0 = 1.0f / fmaxf(l0, 1e-20f);
    const float inv1 = 1.0f / fmaxf(l1, 1e-20f);
    const size_t ob0 = ((size_t)r0 * kInkHeads + h) * kInkHeadDim;
    const size_t ob1 = ((size_t)r1 * kInkHeads + h) * kInkHeadDim;
#pragma unroll
    for (int j = 0; j < 16; ++j) {
        const int c0 = j * 8 + 2 * tig;
        if (r0 < M) {
            out[ob0 + c0 + 0] = o[j][0] * inv0 / (1.0f + __expf(-gate_attn[ob0 + c0 + 0]));
            out[ob0 + c0 + 1] = o[j][1] * inv0 / (1.0f + __expf(-gate_attn[ob0 + c0 + 1]));
        }
        if (r1 < M) {
            out[ob1 + c0 + 0] = o[j][2] * inv1 / (1.0f + __expf(-gate_attn[ob1 + c0 + 0]));
            out[ob1 + c0 + 1] = o[j][3] * inv1 / (1.0f + __expf(-gate_attn[ob1 + c0 + 1]));
        }
    }
#else
    (void)q; (void)kv; (void)rel_table; (void)sinks; (void)gate_attn; (void)slot0; (void)p0; (void)M; (void)is_swa; (void)out;
#endif
}

/// Fused Direct-I32 DP4A Large-M Q8_ROW Loop Controller Apply+Blend (eliminates k_ink_i32_to_f32_scale!)
/// Grid = dim3((D + 31) / 32, (M + 7) / 8), Block = 256.
__global__ void __launch_bounds__(256) k_ink_ctrl_apply_q8_i32_tile16(
    const float* __restrict__ cur,      // [M, D]
    const float* __restrict__ msg,      // [M, D]
    const int32_t* __restrict__ gate_i32, // [M_pad, 64]
    const float* __restrict__ wd_sc,    // [64] (header of w_down Q8_ROW)
    const float* __restrict__ x_sc,     // [M]
    const uint8_t* __restrict__ wg_q8,  // FMT_Q8ROW [D, 64]
    const uint8_t* __restrict__ wf_q8,  // FMT_Q8ROW [D, 64]
    float* __restrict__ out,            // [M, D]
    int M, int D) {
    const int d0 = blockIdx.x * 32;
    const int m0 = blockIdx.y * 8;
    const int tid  = threadIdx.x;
    const int warp = tid >> 5;
    const int lane = tid & 31;

    __shared__ alignas(16) int8_t s_wg[32][kInkGateRank];
    __shared__ alignas(16) int8_t s_wf[32][kInkGateRank];
    __shared__ float s_scg[32], s_scf[32];
    __shared__ alignas(16) int8_t s_gq[8][kInkGateRank];
    __shared__ float s_gsc[8];

    const int wr_row = tid >> 3;
    const int wr_col = (tid & 7) * 8;
    const int g_d = d0 + wr_row;
    if (g_d < D) {
        *reinterpret_cast<uint2*>(&s_wg[wr_row][wr_col]) =
            *reinterpret_cast<const uint2*>(wg_q8 + (size_t)D * 4u + (size_t)g_d * kInkGateRank + wr_col);
        *reinterpret_cast<uint2*>(&s_wf[wr_row][wr_col]) =
            *reinterpret_cast<const uint2*>(wf_q8 + (size_t)D * 4u + (size_t)g_d * kInkGateRank + wr_col);
    }
    if (tid < 32 && (d0 + tid) < D) {
        s_scg[tid] = reinterpret_cast<const float*>(wg_q8)[d0 + tid];
        s_scf[tid] = reinterpret_cast<const float*>(wf_q8)[d0 + tid];
    }

    const int m = m0 + warp;
    if (m < M) {
        const int32_t* gp = gate_i32 + (size_t)m * kInkGateRank;
        const float v0 = (float)gp[2 * lane]     * wd_sc[2 * lane];
        const float v1 = (float)gp[2 * lane + 1] * wd_sc[2 * lane + 1];
        const float am = warp_max(fmaxf(fabsf(v0), fabsf(v1)));
        const float gsc = (am > 0.0f) ? (am * (1.0f / 127.0f) * x_sc[m]) : 0.0f;
        const float inv = (am > 0.0f) ? (127.0f / am) : 0.0f;
        s_gq[warp][2 * lane]     = (int8_t)max(-127, min(127, __float2int_rn(v0 * inv)));
        s_gq[warp][2 * lane + 1] = (int8_t)max(-127, min(127, __float2int_rn(v1 * inv)));
        if (lane == 0) s_gsc[warp] = gsc;
    }
    __syncthreads();

    const int d = d0 + lane;
    if (m >= M || d >= D) return;

    const int32_t* wg4 = reinterpret_cast<const int32_t*>(s_wg[lane]);
    const int32_t* wf4 = reinterpret_cast<const int32_t*>(s_wf[lane]);
    const int32_t* gq4 = reinterpret_cast<const int32_t*>(s_gq[warp]);
    int32_t ag = 0, af = 0;
#pragma unroll
    for (int k = 0; k < 16; ++k) {
        const int32_t gv = gq4[k];
        ag = dp4a_i8(wg4[k], gv, ag);
        af = dp4a_i8(wf4[k], gv, af);
    }
    const float gs = s_gsc[warp];
    const size_t idx = (size_t)m * D + d;
    out[idx] = (1.0f - tanhf((float)af * s_scf[lane] * gs)) * cur[idx] + tanhf((float)ag * s_scg[lane] * gs) * msg[idx];
}

/// Fused 3x Direct-I32 DP4A Large-M Q8_ROW Pre-Attn Controller Apply+Blend (1 launch replaces 6 launches!)
/// Grid = dim3((D + 31) / 32, (M + 7) / 8, 3), Block = 256.
__global__ void __launch_bounds__(256) k_ink_ctrl_3x_apply_q8_i32_tile16(
    const float* __restrict__ cur3,       // [3, M, D] (d_mixed)
    const float* __restrict__ msg3,       // [3, M, D] (d_msgs)
    const int32_t* __restrict__ gate3_i32,// [3, M_pad, 64]
    const uint8_t* __restrict__ wd0, const uint8_t* __restrict__ wd1, const uint8_t* __restrict__ wd2,
    const float* __restrict__ x_sc_3,     // [3, M]
    const uint8_t* __restrict__ wg0, const uint8_t* __restrict__ wg1, const uint8_t* __restrict__ wg2,
    const uint8_t* __restrict__ wf0, const uint8_t* __restrict__ wf1, const uint8_t* __restrict__ wf2,
    float* __restrict__ out0, float* __restrict__ out1, float* __restrict__ out2,
    int M, int M_pad, int D) {
    const int c = blockIdx.z;
    if (c >= 3) return;
    const int d0 = blockIdx.x * 32;
    const int m0 = blockIdx.y * 8;
    const int tid  = threadIdx.x;
    const int warp = tid >> 5;
    const int lane = tid & 31;

    const float*   wd_sc = reinterpret_cast<const float*>((c == 0) ? wd0 : (c == 1 ? wd1 : wd2));
    const uint8_t* wg_q8 = (c == 0) ? wg0 : (c == 1 ? wg1 : wg2);
    const uint8_t* wf_q8 = (c == 0) ? wf0 : (c == 1 ? wf1 : wf2);
    float*         out   = (c == 0) ? out0 : (c == 1 ? out1 : out2);

    __shared__ alignas(16) int8_t s_wg[32][kInkGateRank];
    __shared__ alignas(16) int8_t s_wf[32][kInkGateRank];
    __shared__ float s_scg[32], s_scf[32];
    __shared__ alignas(16) int8_t s_gq[8][kInkGateRank];
    __shared__ float s_gsc[8];

    const int wr_row = tid >> 3;
    const int wr_col = (tid & 7) * 8;
    const int g_d = d0 + wr_row;
    if (g_d < D) {
        *reinterpret_cast<uint2*>(&s_wg[wr_row][wr_col]) =
            *reinterpret_cast<const uint2*>(wg_q8 + (size_t)D * 4u + (size_t)g_d * kInkGateRank + wr_col);
        *reinterpret_cast<uint2*>(&s_wf[wr_row][wr_col]) =
            *reinterpret_cast<const uint2*>(wf_q8 + (size_t)D * 4u + (size_t)g_d * kInkGateRank + wr_col);
    }
    if (tid < 32 && (d0 + tid) < D) {
        s_scg[tid] = reinterpret_cast<const float*>(wg_q8)[d0 + tid];
        s_scf[tid] = reinterpret_cast<const float*>(wf_q8)[d0 + tid];
    }

    const int m = m0 + warp;
    if (m < M) {
        const int32_t* gp = gate3_i32 + ((size_t)c * M_pad + m) * kInkGateRank;
        const float v0 = (float)gp[2 * lane]     * wd_sc[2 * lane];
        const float v1 = (float)gp[2 * lane + 1] * wd_sc[2 * lane + 1];
        const float am = warp_max(fmaxf(fabsf(v0), fabsf(v1)));
        const float gsc = (am > 0.0f) ? (am * (1.0f / 127.0f) * x_sc_3[(size_t)c * M + m]) : 0.0f;
        const float inv = (am > 0.0f) ? (127.0f / am) : 0.0f;
        s_gq[warp][2 * lane]     = (int8_t)max(-127, min(127, __float2int_rn(v0 * inv)));
        s_gq[warp][2 * lane + 1] = (int8_t)max(-127, min(127, __float2int_rn(v1 * inv)));
        if (lane == 0) s_gsc[warp] = gsc;
    }
    __syncthreads();

    const int d = d0 + lane;
    if (m >= M || d >= D) return;

    const int32_t* wg4 = reinterpret_cast<const int32_t*>(s_wg[lane]);
    const int32_t* wf4 = reinterpret_cast<const int32_t*>(s_wf[lane]);
    const int32_t* gq4 = reinterpret_cast<const int32_t*>(s_gq[warp]);
    int32_t ag = 0, af = 0;
#pragma unroll
    for (int k = 0; k < 16; ++k) {
        const int32_t gv = gq4[k];
        ag = dp4a_i8(wg4[k], gv, ag);
        af = dp4a_i8(wf4[k], gv, af);
    }
    const float gs = s_gsc[warp];
    const size_t idx_c = ((size_t)c * M + m) * D + d;
    out[(size_t)m * D + d] = (1.0f - tanhf((float)af * s_scf[lane] * gs)) * cur3[idx_c] + tanhf((float)ag * s_scg[lane] * gs) * msg3[idx_c];
}

/// Fast per-slot >= 2/3-gram Prompt-Lookup (PLD) chain lookup for C64 Mode-3 speculative verify (< 1 us/slot, 0 extra VRAM!).
inline int ink_pld_find_chain(
    const std::vector<int32_t>& hist,
    int32_t anchor_tok,
    int max_cont,
    int32_t* out_cont) noexcept {
    const int H = (int)hist.size();
    if (H < 4 || max_cont <= 0) return 0;
    int32_t ctx[8];
    int clen = 0;
    const int h_end = (hist.back() == anchor_tok) ? (H - 1) : H;
    const int h_take = std::min(4, h_end);
    for (int i = h_end - h_take; i < h_end; ++i) ctx[clen++] = hist[(size_t)i];
    ctx[clen++] = anchor_tok;
    const int scan_beg = std::max(0, H - 2048);
    for (int ng = std::min(5, clen); ng >= 3; --ng) {
        const int32_t* needle = ctx + (clen - ng);
        for (int pos = H - ng - 1; pos >= scan_beg; --pos) {
            bool eq = true;
            for (int k = 0; k < ng; ++k) {
                if (hist[(size_t)(pos + k)] != needle[k]) { eq = false; break; }
            }
            if (eq) {
                const int avail = std::min(max_cont, H - (pos + ng));
                if (avail > 0) {
                    for (int k = 0; k < avail; ++k) out_cont[k] = hist[(size_t)(pos + ng + k)];
                    return avail;
                }
            }
        }
    }
    return 0;
}

// ============================================================================
// 8. Visit-Major All-VRAM Macro-Prefill Helpers (W = 2048..4096 -> >15,000 ppts!)
// ============================================================================
__global__ void __launch_bounds__(256) k_ink_f32_to_bf16_rows(
    const float* __restrict__ src, uint16_t* __restrict__ dst, int M, int D) {
    const size_t n = (size_t)M * D;
    const size_t i = (size_t)blockIdx.x * 256 + threadIdx.x;
    if (i < n) dst[i] = __bfloat16_as_ushort(__float2bfloat16_rn(src[i]));
}

__global__ void __launch_bounds__(256) k_ink_bf16_to_f32_rows(
    const uint16_t* __restrict__ src, float* __restrict__ dst, int M, int D) {
    const size_t n = (size_t)M * D;
    const size_t i = (size_t)blockIdx.x * 256 + threadIdx.x;
    if (i < n) dst[i] = bf16_to_f32(src[i]);
}

/// Compact per-row Q8_ROW pack for archived macro-prefill residual blocks (7-bit mantissa = BF16 precision, 1 B/elem!)
__global__ void __launch_bounds__(256) k_ink_f32_to_q8row_rows(
    const float* __restrict__ src,
    int8_t* __restrict__ dst_q,
    float*  __restrict__ dst_sc,
    int M, int D) {
    const int r = blockIdx.x;
    if (r >= M) return;
    const int tid = threadIdx.x;
    const int warp = tid >> 5;
    const int lane = tid & 31;
    const float* sr = src + (size_t)r * D;
    float am = 0.0f;
    for (int d = tid; d < D; d += 256) am = fmaxf(am, fabsf(sr[d]));
    am = warp_max(am);
    __shared__ float sh[8];
    __shared__ float s_sc;
    if (lane == 0) sh[warp] = am;
    __syncthreads();
    if (tid == 0) {
        float g = sh[0];
        for (int i = 1; i < 8; ++i) g = fmaxf(g, sh[i]);
        s_sc = (g > 0.0f) ? (g * (1.0f / 127.0f)) : 0.0f;
        dst_sc[r] = s_sc;
    }
    __syncthreads();
    const float inv = (s_sc > 0.0f) ? (1.0f / s_sc) : 0.0f;
    int8_t* dr = dst_q + (size_t)r * D;
    for (int d = tid; d < D; d += 256) dr[d] = (int8_t)max(-127, min(127, __float2int_rn(sr[d] * inv)));
}

__global__ void __launch_bounds__(256) k_ink_q8row_to_f32_rows(
    const int8_t* __restrict__ src_q,
    const float*  __restrict__ src_sc,
    float* __restrict__ dst,
    int M, int D) {
    const int r = blockIdx.x;
    if (r >= M) return;
    const float sc = src_sc[r];
    const int8_t* sr = src_q + (size_t)r * D;
    float* dr = dst + (size_t)r * D;
    for (int d = threadIdx.x; d < D; d += 256) dr[d] = (float)sr[d] * sc;
}

/// Manages 12-slot archived Q8_ROW (or BF16) cross-visit state + cur BF16 + macro MoE input cache for W = 2048..4096!
/// Reuses runner->d_pf_slab (108 MiB) for d_y_accum + d_ha_fp4 + tb during MoE -> ZERO extra MoE VRAM!
struct Ink5060TiMacroBufs {
    static constexpr int kArchSlots = 12;  // 0..5: blocks[0..5], 6: prelude, 7..10: bank[0..3] (coda reuses 7,8 for blocks[6,7]!), 11: bank_out
    int      cap_W     = 0;
    bool     use_q8    = true;
    uint8_t* d_slab    = nullptr;
    uint16_t* d_cur    = nullptr;            // [W, 2304] BF16
    uint8_t* d_arch_q[kArchSlots]  = {};     // [W, 2304] int8 (or uint16_t BF16 if !use_q8)
    float*   d_arch_sc[kArchSlots] = {};     // [W] scales (if use_q8)
    uint8_t* d_xa_fp4  = nullptr;            // [W, 36 * 48]
    int32_t* d_tidx    = nullptr;            // [W, 8]
    float*   d_tw      = nullptr;            // [W, 8]
    uint16_t* h_taps[3] = {};                // pinned host [W, 2304] BF16 for taps 0, 1, 2 (0 MiB VRAM!)

    bool ensure(int want_W, int gpu_id) {
        const int target_W = std::min(4224, std::max(1024, want_W));
        if (d_slab && cap_W >= target_W) return true;
        free_all();
        const bool force_bf16 = (std::getenv("STRATA_INK_SM120_MACRO_BF16") ? ink_5060ti_env_flag("STRATA_INK_SM120_MACRO_BF16", false) : ink_5060ti_env_flag("STRATA_INK_5060TI_MACRO_BF16", false));
        auto calc_bytes = [&](int W, bool q8) -> size_t {
            size_t off = 0;
            auto bump = [&](size_t b) { off += (b + 255ull) & ~255ull; };
            bump((size_t)W * kInkHidden * sizeof(uint16_t));                       // d_cur BF16
            for (int i = 0; i < kArchSlots; ++i) {
                bump((size_t)W * kInkHidden * (q8 ? 1u : 2u));                     // d_arch_q
                if (q8) bump((size_t)W * sizeof(float));                           // d_arch_sc
            }
            bump((size_t)W * (kInkHidden / 64) * 48u);                             // d_xa_fp4
            bump((size_t)W * kInkTopK * sizeof(int32_t));                          // d_tidx
            bump((size_t)W * kInkTopK * sizeof(float));                            // d_tw
            return off;
        };
        size_t vf = 0, vt = 0;
        if (ink_cuda_mem_info(&vf, &vt, gpu_id) != cudaSuccess) return false;
        int chosen_W = 0;
        bool chosen_q8 = !force_bf16;
        const size_t guard = 12ull << 20;
        if (!force_bf16 && vf > calc_bytes(4224, true) + guard)       { chosen_W = 4224; chosen_q8 = true;  }
        else if (!force_bf16 && vf > calc_bytes(4096, true) + guard)  { chosen_W = 4096; chosen_q8 = true;  }
        else if (!force_bf16 && vf > calc_bytes(3072, true) + guard)  { chosen_W = 3072; chosen_q8 = true;  }
        else if (vf > calc_bytes(2048, false) + guard)                { chosen_W = 2048; chosen_q8 = false; }
        else if (vf > calc_bytes(2048, true)  + guard)                { chosen_W = 2048; chosen_q8 = true;  }
        else if (vf > calc_bytes(1024, false) + guard)                { chosen_W = 1024; chosen_q8 = false; }
        else return false;

        const size_t tot = calc_bytes(chosen_W, chosen_q8);
        if (cudaMalloc(&d_slab, tot) != cudaSuccess) { cudaGetLastError(); d_slab = nullptr; return false; }
        size_t off = 0;
        auto slice = [&](size_t b) -> uint8_t* {
            uint8_t* p = d_slab + off;
            off += (b + 255ull) & ~255ull;
            return p;
        };
        d_cur = reinterpret_cast<uint16_t*>(slice((size_t)chosen_W * kInkHidden * sizeof(uint16_t)));
        for (int i = 0; i < kArchSlots; ++i) {
            d_arch_q[i]  = slice((size_t)chosen_W * kInkHidden * (chosen_q8 ? 1u : 2u));
            d_arch_sc[i] = chosen_q8 ? reinterpret_cast<float*>(slice((size_t)chosen_W * sizeof(float))) : nullptr;
        }
        d_xa_fp4 = slice((size_t)chosen_W * (kInkHidden / 64) * 48u);
        d_tidx   = reinterpret_cast<int32_t*>(slice((size_t)chosen_W * kInkTopK * sizeof(int32_t)));
        d_tw     = reinterpret_cast<float*>(slice((size_t)chosen_W * kInkTopK * sizeof(float)));
        for (int t = 0; t < 3; ++t) {
            if (cudaHostAlloc(&h_taps[t], (size_t)chosen_W * kInkHidden * sizeof(uint16_t), cudaHostAllocDefault) != cudaSuccess) {
                cudaGetLastError();
                free_all();
                return false;
            }
        }
        cap_W  = chosen_W;
        use_q8 = chosen_q8;
        std::fprintf(stderr, "[Ink5060Ti] Visit-Major Macro-Prefill active (W=%d, arch=%s, %.1f MiB VRAM)\n",
                     cap_W, use_q8 ? "Q8_ROW" : "BF16", (double)tot / 1048576.0);
        return true;
    }

    void free_all() {
        if (d_slab) { cudaFree(d_slab); d_slab = nullptr; }
        d_cur = nullptr; d_xa_fp4 = nullptr; d_tidx = nullptr; d_tw = nullptr;
        for (int i = 0; i < kArchSlots; ++i) { d_arch_q[i] = nullptr; d_arch_sc[i] = nullptr; }
        for (int t = 0; t < 3; ++t) { if (h_taps[t]) { cudaFreeHost(h_taps[t]); h_taps[t] = nullptr; } }
        cap_W = 0;
    }
};

// ============================================================================
// 10. Opt-In NVFP4 (FMT_SF4) Dense Trunk Transcode + Small-M Fused GEMV + Fused C64 Post-QKV
// ============================================================================

/// Pass 1 of BF16 -> FMT_SF4 transcode: compute global amax of BF16 [N, K] into d_amax_u32.
__global__ void __launch_bounds__(256) k_ink_bf16_amax(
    const uint16_t* __restrict__ src, int64_t total, uint32_t* __restrict__ d_amax_u32) {
    float am = 0.0f;
    const int64_t stride = (int64_t)gridDim.x * 256;
    for (int64_t i = (int64_t)blockIdx.x * 256 + threadIdx.x; i < total; i += stride) {
        am = fmaxf(am, fabsf(bf16_to_f32(src[i])));
    }
    am = warp_max(am);
    __shared__ float sh[8];
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    if (lane == 0) sh[warp] = am;
    __syncthreads();
    if (threadIdx.x == 0) {
        float g = sh[0];
        for (int i = 1; i < 8; ++i) g = fmaxf(g, sh[i]);
        atomicMax(d_amax_u32, __float_as_uint(g));
    }
}

/// Pass 2 of BF16 -> FMT_SF4 transcode: quantize BF16 [N, K] directly into .strata v1.1 FMT_SF4 payload!
/// Grid = N, Block = 256 (1 thread per 16-element sub-block).
__global__ void __launch_bounds__(256) k_ink_bf16_to_sf4(
    const uint16_t* __restrict__ src,
    const uint32_t* __restrict__ d_amax_u32,
    uint8_t* __restrict__ dst_sf4,
    int N, int K) {
    const int r = blockIdx.x;
    if (r >= N) return;
    const float amax = __uint_as_float(*d_amax_u32);
    const float w_outer = fmaxf(amax / (6.0f * 240.0f), 1e-6f);
    if (r == 0 && threadIdx.x == 0) {
        float* hdr = reinterpret_cast<float*>(dst_sf4);
        hdr[0] = w_outer;
        hdr[1] = 1.0f;
    }
    uint8_t* dst_p = dst_sf4 + 16u;
    uint8_t* dst_s = dst_p + (size_t)N * (K >> 1);
    const int nblk16 = K >> 4;
    const uint16_t* srow = src + (size_t)r * K;
    for (int b = threadIdx.x; b < nblk16; b += 256) {
        const uint16_t* sp = srow + b * 16;
        float vals[16];
        float bm = 0.0f;
#pragma unroll
        for (int i = 0; i < 16; ++i) {
            vals[i] = bf16_to_f32(sp[i]);
            bm = fmaxf(bm, fabsf(vals[i]));
        }
        const float raw_sc = bm / (6.0f * w_outer);
        const __nv_fp8_e4m3 sc_fp8(fminf(raw_sc, 448.0f));
        const uint8_t sc_byte = (raw_sc > 0.0f) ? sc_fp8.__x : 0u;
        const float s_dec = (sc_byte != 0u) ? ((float)sc_fp8 * w_outer) : 0.0f;
        const float inv = (s_dec > 0.0f) ? (1.0f / s_dec) : 0.0f;
        dst_s[(size_t)r * nblk16 + b] = sc_byte;
        uint8_t* dp = dst_p + (size_t)r * (K >> 1) + b * 8;
#pragma unroll
        for (int j = 0; j < 8; ++j) {
            dp[j] = ink_pack_e2m1x2_portable(vals[2 * j] * inv, vals[2 * j + 1] * inv);
        }
    }
}

/// Fused Small-M (M <= 16) Multi-Projection NVFP4 (FMT_SF4) GEMV with FP32 activations (2x less GDDR7 BW than Q8_ROW!).
/// Grid = (cumN[count] + 3) / 4, Block = 128 (4 warps = 4 output rows/CTA; 32 lanes = 128 weights/step, 100% coalesced!).
template <int MAX_M, typename MP>
__global__ void __launch_bounds__(128) k_ink_gemv_multi_sf4(
    MP mp,
    const float* __restrict__ X,  // [M, K] FP32
    int M, int K) {
    const int tid  = threadIdx.x;
    const int warp = tid >> 5;
    const int lane = tid & 31;
    const int g_row = blockIdx.x * 4 + warp;
    if (g_row >= mp.cumN[mp.count]) return;

    int pi = 0;
#pragma unroll
    for (int i = 1; i < 5; ++i) {
        if (i < mp.count && g_row >= mp.cumN[i]) pi = i;
    }
    const int row = g_row - mp.cumN[pi];
    const int Np  = mp.N[pi];
    const uint8_t* w_sf4 = mp.W[pi];
    const float    w_outer = *reinterpret_cast<const float*>(w_sf4);
    const uint8_t* w_p = w_sf4 + 16u + (size_t)row * (K >> 1);
    const uint8_t* w_s = w_sf4 + 16u + (size_t)Np * (K >> 1) + (size_t)row * (K >> 4);

    // Each lane processes 4 consecutive weights (2 packed bytes) per step: 4 lanes = 16-elem sub-block, 32 lanes = 128 weights!
    static constexpr float kE2M1[16] = {
        0.0f,  0.5f,  1.0f,  1.5f,  2.0f,  3.0f,  4.0f,  6.0f,
       -0.0f, -0.5f, -1.0f, -1.5f, -2.0f, -3.0f, -4.0f, -6.0f
    };
    float acc[MAX_M] = {};
    const int sub_lane = lane >> 2;  // 0..7 (which 16-elem sub-block within the 128-elem chunk)
    const int in_sub   = lane & 3;   // 0..3 -> elems 4*in_sub .. 4*in_sub+3
    for (int k0 = 0; k0 < K; k0 += 128) {
        const int b16 = (k0 >> 4) + sub_lane;
        const float sc = e4m3_to_f32(w_s[b16]);
        const int col = k0 + lane * 4;
        const uint16_t pk = *reinterpret_cast<const uint16_t*>(w_p + (col >> 1));
        const float w0 = kE2M1[ pk        & 0xFu] * sc;
        const float w1 = kE2M1[(pk >>  4) & 0xFu] * sc;
        const float w2 = kE2M1[(pk >>  8) & 0xFu] * sc;
        const float w3 = kE2M1[(pk >> 12) & 0xFu] * sc;
#pragma unroll
        for (int m = 0; m < MAX_M; ++m) {
            if (m < M) {
                const float4 xv = *reinterpret_cast<const float4*>(X + (size_t)m * K + col);
                acc[m] = fmaf(w0, xv.x, fmaf(w1, xv.y, fmaf(w2, xv.z, fmaf(w3, xv.w, acc[m]))));
            }
        }
        (void)in_sub;
    }
#pragma unroll
    for (int m = 0; m < MAX_M; ++m) {
        if (m < M) {
            float v = acc[m];
#pragma unroll
            for (int o = 16; o >= 1; o >>= 1) v += __shfl_xor_sync(0xFFFFFFFFu, v, o);
            if (lane == 0) mp.Y[pi][(size_t)m * Np + row] = v * w_outer;
        }
    }
}

/// Fused 1-Launch Mode-3 (C64 Multi-Chain) Post-QKV Kernel:
/// Replaces 5 launches (sconv K + sconv V + qkvr_clip_tau + kv_write + 2nd sdesc kv_write) with ONE launch!
/// Grid = dim3(64, B), Block = 128. Loops k = 0..K-1 in registers so ShortConv history & K/V never touch DRAM!
__global__ void __launch_bounds__(128) k_ink_post_qkv_multichain(
    float* __restrict__ q,                  // [B*K, 24, 128] in/out
    const float* __restrict__ q_scale, int q_scale_elems,
    const float* __restrict__ k_raw,        // [B*K, 8, 128]
    const float* __restrict__ v_raw,        // [B*K, 8, 128]
    const float* __restrict__ sw_k,         // [1024, 4]
    const float* __restrict__ sw_v,         // [1024, 4]
    const float* __restrict__ k_dyn,        // [B*K, 8, 4] or nullptr
    const float* __restrict__ v_dyn,        // [B*K, 8, 4] or nullptr
    float* __restrict__ hist_k_pool,        // [max_slots, 3, 1024]
    float* __restrict__ hist_v_pool,        // [max_slots, 3, 1024]
    uint16_t* __restrict__ stage_sconv,     // [B, K, kInkSconvCh] BF16 or nullptr
    float* __restrict__ r,                  // [B*K, 24, 16] in/out
    const float* __restrict__ r_scale, int r_scale_elems,
    const int32_t* __restrict__ slot_ids,   // [B]
    const int32_t* __restrict__ base_pos,   // [B]
    InkKvVisitDesc kv,
    InkKvVisitDesc sdesc,
    bool write_sdesc,
    bool is_global,
    int B, int K) {
    const int b   = blockIdx.y;
    const int blk = blockIdx.x;
    const int tid = threadIdx.x;
    if (b >= B) return;

    const int sl = slot_ids[b];
    const int p0 = base_pos[b];
    __shared__ float sh[8];

    if (blk < 24) {
        const int h = blk;
        float qsc = 1.0f;
        if (q_scale) {
            const int si = (q_scale_elems == kInkHeadDim) ? tid : (h * kInkHeadDim + tid);
            qsc = q_scale[si];
        }
        for (int k = 0; k < K; ++k) {
            const int m = b * K + k;
            float tau = 1.0f;
            if (is_global) {
                const float ratio = (float)(p0 + k + 1) * (1.0f / (float)kInkLogFloor);
                tau = 1.0f + kInkLogAlpha * logf(fmaxf(1.0f, ratio));
            }
            float* qp = q + ((size_t)m * kInkHeads + h) * kInkHeadDim;
            float val = qp[tid] * qsc;
            float ss = block_sum(val * val, sh);
            const float factor = fminf(1.0f, kInkQkClip / (sqrtf(ss) + 1e-6f)) * tau;
            qp[tid] = val * factor;
            __syncthreads();
        }
    } else if (blk < 40) {
        const bool is_k = (blk < 32);
        const int kh = is_k ? (blk - 24) : (blk - 32);
        const int c  = kh * kInkHeadDim + tid;
        const float* raw = is_k ? k_raw : v_raw;
        const float* swp = is_k ? sw_k : sw_v;
        const float* dyn = is_k ? k_dyn : v_dyn;
        const int ch_off = is_k ? 0 : kInkKVDim;
        float* hist = (is_k ? hist_k_pool : hist_v_pool) + (size_t)sl * 3 * kInkKVDim;

        float h0 = hist[0 * kInkKVDim + c];
        float h1 = hist[1 * kInkKVDim + c];
        float h2 = hist[2 * kInkKVDim + c];
        const float4 sw = reinterpret_cast<const float4*>(swp)[c];
        const float sc_conv = dyn ? (1.0f / 3.0f) : 0.44721359549995793f;

        for (int k = 0; k < K; ++k) {
            const int m = b * K + k;
            const float cur = raw[(size_t)m * kInkKVDim + c];
            if (stage_sconv) {
                stage_sconv[(size_t)m * kInkSconvCh + ch_off + c] = __bfloat16_as_ushort(__float2bfloat16_rn(cur));
            }
            float acc = h0 * sw.x + h1 * sw.y + h2 * sw.z + cur * sw.w;
            if (dyn) {
                const float4 dt = reinterpret_cast<const float4*>(dyn + ((size_t)m * kInkKVHeads + kh) * 4)[0];
                acc += cur * dt.x + h2 * dt.y + h1 * dt.z + h0 * dt.w;
            }
            float val = (cur + acc) * sc_conv;
            h0 = h1; h1 = h2; h2 = cur;

            float ss = block_sum(val * val, sh);
            if (is_k) {
                val *= fminf(1.0f, kInkQkClip / (sqrtf(ss) + 1e-6f));
            } else {
                val *= rsqrtf(ss * (1.0f / (float)kInkHeadDim) + kInkRmsEps);
            }
            __syncthreads();

            const int p = p0 + k;
            if (p >= 0 && (kv.is_swa || p < kv.cap_pos)) {
                const int p_idx = kv.is_swa ? (p % kv.cap_pos) : p;
                ink_kv_store_128(kv, sl, p_idx, kh, tid, val, is_k, sh);
                __syncthreads();
                if (write_sdesc && (sdesc.is_swa || p < sdesc.cap_pos)) {
                    const int sp_idx = sdesc.is_swa ? (p % sdesc.cap_pos) : p;
                    ink_kv_store_128(sdesc, sl, sp_idx, kh, tid, val, is_k, sh);
                    __syncthreads();
                }
            }
        }
        hist[0 * kInkKVDim + c] = h0;
        hist[1 * kInkKVDim + c] = h1;
        hist[2 * kInkKVDim + c] = h2;
    } else if (blk < 43) {
        // Pack 8 R-heads (8 x 16 = 128 threads, 100% active!) per CTA -> 43 CTAs/slot instead of 64!
        const int h = (blk - 40) * 8 + (tid >> 4);
        const int d = tid & 15;
        float rsc = 1.0f;
        if (r_scale) {
            const int si = (r_scale_elems == kInkDRel) ? d : (h * kInkDRel + d);
            rsc = r_scale[si];
        }
        const uint32_t half_mask = (tid & 16) ? 0xFFFF0000u : 0x0000FFFFu;
        for (int k = 0; k < K; ++k) {
            const int m = b * K + k;
            float tau = 1.0f;
            if (is_global) {
                const float ratio = (float)(p0 + k + 1) * (1.0f / (float)kInkLogFloor);
                tau = 1.0f + kInkLogAlpha * logf(fmaxf(1.0f, ratio));
            }
            float* rp = r + ((size_t)m * kInkHeads + h) * kInkDRel;
            float val = rp[d] * rsc;
            float ss = val * val;
#pragma unroll
            for (int o = 8; o >= 1; o >>= 1) ss += __shfl_xor_sync(half_mask, ss, o);
            const float factor = fminf(1.0f, kInkRelClip / (sqrtf(ss) + 1e-6f)) * tau;
            rp[d] = val * factor;
        }
    }
}

}  // namespace strata::ink
