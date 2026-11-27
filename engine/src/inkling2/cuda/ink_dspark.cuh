// engine/src/inkling2/cuda/ink_dspark.cuh - Persistent 100% On-GPU DSpark Block-8 / Tree-16 Recurrent Markov Drafter (< 0.35 ms!)
//
// MIT License
// Copyright (c) 2026 Niko1221 and the Strata contributors
//
// Implements Thinking Machines' co-trained 3-layer Block-8 DSpark drafter (embedded in .strata v1.1 Section 1):
//   - Taps [5, 23, 41, 48] mixed via softmax(fuse_weight[0..3]) + dflare_norm -> context K/V cache (SWA-1025, Global, SWA-1025) + f_rnn
//   - 3 non-causal block-attention layers (24 Q, 8 KV, head_dim=128, NeoX half-split RoPE theta=1e6, SwiGLU 9216)
//   - Fast base logits over B=8 slots via 230 MB 4-bit screening plane (or BF16/Q8_ROW) + 4-pass radix Top-K_cand (K_cand = 256/512)
//   - Persistent on-GPU 7-step markov_rank=256 RNN head scoring ONLY K_cand rows of unembed (fits 100% in L2 -> ~10 us/step!)
//   - Supports BOTH Block-8 linear drafting AND Tree-16 2-branch beam drafting with 100% lossless exact target verification!
#pragma once

#include "ink_runner.cuh"
#include "ink_dflash2.cuh"   // opt-in DFlash 2 drafter (STRATA_INK_DRAFTER=dflash2); every hook of it below starts with `if (df2)`

namespace strata::ink {

// ---- NeoX Half-Split RoPE (theta = 1e6) + Per-Head RMSNorm ----
__global__ void __launch_bounds__(128) k_dspark_rmsnorm_rope(
    float* __restrict__ x,               // [M, H, 128]
    const float* __restrict__ nw,        // [128] F32
    const int32_t* __restrict__ pos_arr, // [M]
    float theta,
    int M, int H) {
    const int m = blockIdx.y;
    const int h = blockIdx.x;
    const int d = threadIdx.x;  // 0..127
    if (m >= M || h >= H) return;
    __shared__ float s_x[kInkHeadDim];
    __shared__ float sh[8];
    float* xp = x + ((size_t)m * H + h) * kInkHeadDim;
    float v = xp[d];
    float ss = block_sum(v * v, sh);
    const float inv = rsqrtf(ss * (1.0f / (float)kInkHeadDim) + kInkRmsEps);
    s_x[d] = v * inv * nw[d];
    __syncthreads();

    // NeoX half-split RoPE: pairs (d, d + 64) for d in 0..63
    if (d < 64) {
        const float freq = 1.0f / powf(theta, (float)(2 * d) * (1.0f / (float)kInkHeadDim));
        const float ang = (float)pos_arr[m] * freq;
        float s_val, c_val;
        __sincosf(ang, &s_val, &c_val);
        const float x0 = s_x[d];
        const float x1 = s_x[d + 64];
        xp[d]      = x0 * c_val - x1 * s_val;
        xp[d + 64] = x0 * s_val + x1 * c_val;
    }
}

// Mix 4 target taps [4, M, 2304] with softmax(fuse_weight[row]) + RMSNorm(norm_w) -> out [M, 2304]
__global__ void __launch_bounds__(256) k_dspark_mix_taps(
    const float* __restrict__ tap0,
    const float* __restrict__ tap1,
    const float* __restrict__ tap2,
    const float* __restrict__ tap3,
    const float* __restrict__ fuse_w,  // [4, 4] F32
    int row,
    const float* __restrict__ norm_w,  // [2304] F32
    float* __restrict__ out,           // [M, 2304]
    int M) {
    const int m = blockIdx.x;
    if (m >= M) return;
    const int tid = threadIdx.x;
    __shared__ float sw[4];
    __shared__ float sh[32];
    if (tid == 0) {
        const float* fw = fuse_w + row * 4;
        float mx = fmaxf(fmaxf(fw[0], fw[1]), fmaxf(fw[2], fw[3]));
        float e0 = expf(fw[0] - mx), e1 = expf(fw[1] - mx), e2 = expf(fw[2] - mx), e3 = expf(fw[3] - mx);
        float inv = 1.0f / (e0 + e1 + e2 + e3);
        sw[0] = e0 * inv; sw[1] = e1 * inv; sw[2] = e2 * inv; sw[3] = e3 * inv;
    }
    __syncthreads();
    const size_t off = (size_t)m * kInkHidden;
    float* dst = out + off;
    float ss = 0.0f;
    for (int d = tid; d < kInkHidden; d += 256) {
        float v = sw[0] * tap0[off + d] + sw[1] * tap1[off + d] + sw[2] * tap2[off + d] + sw[3] * tap3[off + d];
        dst[d] = v;
        ss += v * v;
    }
    ss = block_sum(ss, sh);
    const float inv = rsqrtf(ss / (float)kInkHidden + kInkRmsEps);
    for (int d = tid; d < kInkHidden; d += 256) dst[d] = dst[d] * inv * norm_w[d];
}

// Ring-buffered variant for compact d_f_rnn [frnn_cap, 2304] (saves 2.4 GiB at 256K ctx!)
__global__ void __launch_bounds__(256) k_dspark_mix_taps_ring(
    const float* __restrict__ tap0,
    const float* __restrict__ tap1,
    const float* __restrict__ tap2,
    const float* __restrict__ tap3,
    const float* __restrict__ fuse_w,
    int row,
    const float* __restrict__ norm_w,
    float* __restrict__ out_ring,
    int p0, int ring_cap, int M) {
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
    const size_t src_off = (size_t)m * kInkHidden;
    const int dst_slot = (p0 + m) % ring_cap;
    float* dst = out_ring + (size_t)dst_slot * kInkHidden;
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

// Non-causal DSpark Block-8 Attention: attends to committed ctx [c0..apos-1] + all B=8 draft tokens [0..B-1]!
// Supports compact SWA-1025 ring buffers for layers 0 & 2 and elastic global KV for layer 1!
__global__ void __launch_bounds__(128) k_dspark_block_attn(
    const float* __restrict__ q,          // [B, 24, 128]
    const float* __restrict__ kd,         // [B, 8, 128]
    const float* __restrict__ vd,         // [B, 8, 128]
    const __half* __restrict__ ctx_k,     // [ctx_cap, 8, 128]
    const __half* __restrict__ ctx_v,     // [ctx_cap, 8, 128]
    const int32_t* __restrict__ apos_ptr, // [1] committed context length
    bool is_swa,
    int swa_win,                          // 1025
    int ctx_cap,
    int B,                                // 8
    float* __restrict__ out) {            // [B, 24, 128]
    const int b = blockIdx.y;
    const int h = blockIdx.x;  // 0..23
    if (b >= B || h >= kInkHeads) return;
    const int apos = *apos_ptr;

    const int tid = threadIdx.x;
    const int warp = tid >> 5;
    const int lane = tid & 31;
    const int kh = h / 3;
    const float scale = 0.08838834764831845f;

    const float4 qv = *(const float4*)(q + ((size_t)b * kInkHeads + h) * kInkHeadDim + lane * 4);
    const int c0 = is_swa ? max(0, apos - swa_win + 1) : 0;
    const int c_end = is_swa ? apos : min(apos, ctx_cap);

    float mw = -1e30f, lw = 0.0f;
    float a0 = 0.0f, a1 = 0.0f, a2 = 0.0f, a3 = 0.0f;

    auto attend = [&](float4 kv4, float4 vv4) {
        float dot = qv.x * kv4.x + qv.y * kv4.y + qv.z * kv4.z + qv.w * kv4.w;
        dot = warp_sum(dot) * scale;
        const float mn = fmaxf(mw, dot);
        const float al = __expf(mw - mn);
        const float p  = __expf(dot - mn);
        lw = lw * al + p;
        a0 = a0 * al + p * vv4.x; a1 = a1 * al + p * vv4.y;
        a2 = a2 * al + p * vv4.z; a3 = a3 * al + p * vv4.w;
        mw = mn;
    };

    for (int kp = c0 + warp; kp < c_end; kp += 4) {
        const int p_idx = is_swa ? (kp % ctx_cap) : kp;
        const size_t base = ((size_t)p_idx * kInkKVHeads + kh) * kInkHeadDim + lane * 4;
        const uint2 k2 = *(const uint2*)(ctx_k + base);
        const uint2 v2 = *(const uint2*)(ctx_v + base);
        const float2 kf0 = __half22float2(*reinterpret_cast<const __half2*>(&k2.x));
        const float2 kf1 = __half22float2(*reinterpret_cast<const __half2*>(&k2.y));
        const float2 vf0 = __half22float2(*reinterpret_cast<const __half2*>(&v2.x));
        const float2 vf1 = __half22float2(*reinterpret_cast<const __half2*>(&v2.y));
        attend(make_float4(kf0.x, kf0.y, kf1.x, kf1.y), make_float4(vf0.x, vf0.y, vf1.x, vf1.y));
    }
    for (int j = warp; j < B; j += 4) {
        const size_t base = ((size_t)j * kInkKVHeads + kh) * kInkHeadDim + lane * 4;
        attend(*(const float4*)(kd + base), *(const float4*)(vd + base));
    }

    __shared__ float sm_m[4], sm_l[4], sm_a[4][kInkHeadDim];
    if (lane == 0) { sm_m[warp] = mw; sm_l[warp] = lw; }
    sm_a[warp][lane * 4 + 0] = a0; sm_a[warp][lane * 4 + 1] = a1;
    sm_a[warp][lane * 4 + 2] = a2; sm_a[warp][lane * 4 + 3] = a3;
    __syncthreads();
    if (warp == 0) {
        float mg = fmaxf(fmaxf(sm_m[0], sm_m[1]), fmaxf(sm_m[2], sm_m[3]));
        float w0 = sm_l[0] > 0.0f ? __expf(sm_m[0] - mg) : 0.0f;
        float w1 = sm_l[1] > 0.0f ? __expf(sm_m[1] - mg) : 0.0f;
        float w2 = sm_l[2] > 0.0f ? __expf(sm_m[2] - mg) : 0.0f;
        float w3 = sm_l[3] > 0.0f ? __expf(sm_m[3] - mg) : 0.0f;
        const float inv = 1.0f / (sm_l[0] * w0 + sm_l[1] * w1 + sm_l[2] * w2 + sm_l[3] * w3);
        const size_t ob = ((size_t)b * kInkHeads + h) * kInkHeadDim + lane * 4;
        *(float4*)(out + ob) = make_float4(
            (sm_a[0][lane * 4 + 0] * w0 + sm_a[1][lane * 4 + 0] * w1 + sm_a[2][lane * 4 + 0] * w2 + sm_a[3][lane * 4 + 0] * w3) * inv,
            (sm_a[0][lane * 4 + 1] * w0 + sm_a[1][lane * 4 + 1] * w1 + sm_a[2][lane * 4 + 1] * w2 + sm_a[3][lane * 4 + 1] * w3) * inv,
            (sm_a[0][lane * 4 + 2] * w0 + sm_a[1][lane * 4 + 2] * w1 + sm_a[2][lane * 4 + 2] * w2 + sm_a[3][lane * 4 + 2] * w3) * inv,
            (sm_a[0][lane * 4 + 3] * w0 + sm_a[1][lane * 4 + 3] * w1 + sm_a[2][lane * 4 + 3] * w2 + sm_a[3][lane * 4 + 3] * w3) * inv);
    }
}

__global__ void __launch_bounds__(256) k_dspark_fill_mask(
    const float* __restrict__ mask_e,
    float* __restrict__ dst_cur,
    int B) {
    const int s = blockIdx.x + 1;
    if (s >= B) return;
    const int tid = threadIdx.x;
    const float4* src4 = reinterpret_cast<const float4*>(mask_e);
    float4* dst4 = reinterpret_cast<float4*>(dst_cur + (size_t)s * kInkHidden);
    for (int i = tid; i < (kInkHidden / 4); i += 256) dst4[i] = src4[i];
}

__global__ void __launch_bounds__(256) k_dspark_split_qkv(
    const float* __restrict__ qkv,  // [B, 5120]
    float* __restrict__ q,          // [B, 3072]
    float* __restrict__ k,          // [B, 1024]
    float* __restrict__ v,          // [B, 1024]
    int B) {
    constexpr int TOT = kInkQDim + 2 * kInkKVDim;  // 5120
    const int b = blockIdx.y;
    const int i = blockIdx.x * 256 + threadIdx.x;
    if (b >= B || i >= TOT) return;
    const float val = qkv[(size_t)b * TOT + i];
    if (i < kInkQDim) q[(size_t)b * kInkQDim + i] = val;
    else if (i < kInkQDim + kInkKVDim) k[(size_t)b * kInkKVDim + (i - kInkQDim)] = val;
    else v[(size_t)b * kInkKVDim + (i - kInkQDim - kInkKVDim)] = val;
}

__global__ void __launch_bounds__(256) k_dspark_swiglu_fused(
    const float* __restrict__ gu,   // [B, 2 * 9216]
    float* __restrict__ out,        // [B, 9216]
    int B) {
    const int b = blockIdx.y;
    const int i = blockIdx.x * 256 + threadIdx.x;
    if (b >= B || i >= kInkDsparkInter) return;
    const float* row = gu + (size_t)b * (2 * kInkDsparkInter);
    const float gv = row[i];
    out[(size_t)b * kInkDsparkInter + i] = (gv / (1.0f + __expf(-gv))) * row[kInkDsparkInter + i];
}

__global__ void __launch_bounds__(256) k_dspark_swiglu(const float* __restrict__ g, const float* __restrict__ u, float* __restrict__ out, size_t n) {
    const size_t i = (size_t)blockIdx.x * 256 + threadIdx.x;
    if (i < n) {
        const float gv = g[i];
        out[i] = (gv / (1.0f + __expf(-gv))) * u[i];
    }
}

// Fast 4-pass radix Top-K candidate extractor from base logits [7, V] -> cand_ids [7, K_cand] & cand_base [7, K_cand]
__device__ __forceinline__ uint32_t dspark_f2key(float f) {
    uint32_t b = __float_as_uint(f);
    return b ^ ((b >> 31) ? 0xFFFFFFFFu : 0x80000000u);
}

// ---- Ultra-Fast INT8 __dp4a Base Logits for DSpark 7 Slots (~0.30 ms on RTX 5090!) ----
__global__ void __launch_bounds__(256) k_dspark_quant_h_q8(
    const float* __restrict__ h,  // [7, 2304]
    int8_t* __restrict__ h_q8,    // [7, 2304]
    float* __restrict__ h_sc) {   // [7]
    const int s = blockIdx.x;
    const int tid = threadIdx.x;
    const float* hp = h + (size_t)s * kInkHidden;
    float am = 0.0f;
    for (int d = tid; d < kInkHidden; d += 256) am = fmaxf(am, fabsf(hp[d]));
    am = warp_max(am);
    __shared__ float sm[8];
    if ((tid & 31) == 0) sm[tid >> 5] = am;
    __syncthreads();
    am = fmaxf(fmaxf(fmaxf(sm[0], sm[1]), fmaxf(sm[2], sm[3])),
               fmaxf(fmaxf(sm[4], sm[5]), fmaxf(sm[6], sm[7])));
    const float sc = am * (1.0f / 127.0f);
    const float inv = sc > 0.0f ? (127.0f / am) : 0.0f;
    int8_t* dst = h_q8 + (size_t)s * kInkHidden;
    for (int d = tid; d < kInkHidden; d += 256) dst[d] = (int8_t)max(-127, min(127, __float2int_rn(hp[d] * inv)));
    if (tid == 0) h_sc[s] = sc;
}

// 128 threads/CTA (4 warps = 4 vocab rows), caches ALL 7 x 2304 int8 bytes in 16 KB shared memory, dp4a_i8!
__global__ void __launch_bounds__(128) k_dspark_screen_dp4a7(
    const int8_t* __restrict__ q8,    // [V, 2304]
    const float* __restrict__ sc,     // [V]
    const int8_t* __restrict__ h_q8,  // [7, 2304]
    const float* __restrict__ h_sc,   // [7]
    float* __restrict__ base_logits,  // [7, V]
    int V) {
    __shared__ alignas(16) uint32_t s_h[7][kInkHidden / 4];  // 16,128 B!
    const int tid = threadIdx.x;
    for (int m = 0; m < 7; ++m) {
        const uint4* src = reinterpret_cast<const uint4*>(h_q8 + (size_t)m * kInkHidden);
        uint4* dst = reinterpret_cast<uint4*>(s_h[m]);
        for (int i = tid; i < (kInkHidden / 16); i += 128) dst[i] = src[i];
    }
    __syncthreads();

    const int warp = tid >> 5;
    const int lane = tid & 31;
    const int row  = blockIdx.x * 4 + warp;
    if (row >= V) return;

    const uint4* q16 = reinterpret_cast<const uint4*>(q8 + (size_t)row * kInkHidden);
    int32_t acc[7] = {};
    for (int i = lane; i < (kInkHidden / 16); i += 32) {
        const uint4 v = q16[i];
        const int base = i * 4;
#pragma unroll
        for (int m = 0; m < 7; ++m) {
            const uint4 hv = *reinterpret_cast<const uint4*>(s_h[m] + base);
            int32_t a = acc[m];
            a = dp4a_i8(v.x, hv.x, a);
            a = dp4a_i8(v.y, hv.y, a);
            a = dp4a_i8(v.z, hv.z, a);
            a = dp4a_i8(v.w, hv.w, a);
            acc[m] = a;
        }
    }
    const float sc_r = sc[row];
#pragma unroll
    for (int m = 0; m < 7; ++m) {
        int32_t s = acc[m];
#pragma unroll
        for (int o = 16; o >= 1; o >>= 1) s += __shfl_xor_sync(0xFFFFFFFFu, s, o);
        if (lane == 0) base_logits[(size_t)m * V + row] = (float)s * sc_r * h_sc[m];
    }
}

// Zero-Emulated-dp4a Bank-Conflict-Free FP32-FMA Screener for Volta sm_70 (V100 15.7 TFLOPS) & Pascal sm_60 (P100)!
// Zero shared-memory LUT bank conflicts -> HBM2 bandwidth bound (~0.55 ms on V100 vs 4.9-6.7 ms emulated dp4a!)
__global__ void __launch_bounds__(128) k_dspark_screen_h2_7(
    const int8_t* __restrict__ q8,    // [V, 2304]
    const float* __restrict__ sc,     // [V]
    const int8_t* __restrict__ h_q8,  // [7, 2304]
    const float* __restrict__ h_sc,   // [7]
    float* __restrict__ base_logits,  // [7, V]
    int V) {
    __shared__ alignas(16) __half2 s_h[7][kInkHidden / 2];  // 32,256 B (linear read -> ZERO bank conflicts!)
    const int tid = threadIdx.x;
    for (int m = 0; m < 7; ++m) {
        const int8_t* src = h_q8 + (size_t)m * kInkHidden;
        for (int i = tid; i < (kInkHidden / 2); i += 128) {
            s_h[m][i] = __halves2half2(
                __float2half_rn((float)src[2 * i] * (1.0f / 127.0f)),
                __float2half_rn((float)src[2 * i + 1] * (1.0f / 127.0f)));
        }
    }
    __syncthreads();

    const int warp = tid >> 5;
    const int lane = tid & 31;
    const int row  = blockIdx.x * 4 + warp;
    if (row >= V) return;

    const uint4* q16 = reinterpret_cast<const uint4*>(q8 + (size_t)row * kInkHidden);
    __half2 acc[7];
#pragma unroll
    for (int m = 0; m < 7; ++m) acc[m] = __float2half2_rn(0.0f);

    for (int i = lane; i < (kInkHidden / 16); i += 32) {
        const uint4 v = q16[i];
        const __half2 w0 = __halves2half2(__int2half_rn((int)(int8_t)(v.x)),       __int2half_rn((int)(int8_t)(v.x >> 8)));
        const __half2 w1 = __halves2half2(__int2half_rn((int)(int8_t)(v.x >> 16)), __int2half_rn((int)(int8_t)(v.x >> 24)));
        const __half2 w2 = __halves2half2(__int2half_rn((int)(int8_t)(v.y)),       __int2half_rn((int)(int8_t)(v.y >> 8)));
        const __half2 w3 = __halves2half2(__int2half_rn((int)(int8_t)(v.y >> 16)), __int2half_rn((int)(int8_t)(v.y >> 24)));
        const __half2 w4 = __halves2half2(__int2half_rn((int)(int8_t)(v.z)),       __int2half_rn((int)(int8_t)(v.z >> 8)));
        const __half2 w5 = __halves2half2(__int2half_rn((int)(int8_t)(v.z >> 16)), __int2half_rn((int)(int8_t)(v.z >> 24)));
        const __half2 w6 = __halves2half2(__int2half_rn((int)(int8_t)(v.w)),       __int2half_rn((int)(int8_t)(v.w >> 8)));
        const __half2 w7 = __halves2half2(__int2half_rn((int)(int8_t)(v.w >> 16)), __int2half_rn((int)(int8_t)(v.w >> 24)));
        const int base = i * 8;
#pragma unroll
        for (int m = 0; m < 7; ++m) {
            const __half2* hp = &s_h[m][base];
            __half2 a = acc[m];
            a = __hfma2(w0, hp[0], a); a = __hfma2(w1, hp[1], a);
            a = __hfma2(w2, hp[2], a); a = __hfma2(w3, hp[3], a);
            a = __hfma2(w4, hp[4], a); a = __hfma2(w5, hp[5], a);
            a = __hfma2(w6, hp[6], a); a = __hfma2(w7, hp[7], a);
            acc[m] = a;
        }
    }
    const float sc_r = sc[row] * 127.0f;
#pragma unroll
    for (int m = 0; m < 7; ++m) {
        const float2 f2 = __half22float2(acc[m]);
        float s = warp_sum(f2.x + f2.y);
        if (lane == 0) base_logits[(size_t)m * V + row] = s * sc_r * h_sc[m];
    }
}

// Split fused kv_ctx [M, 2048] -> kc [M, 1024], vc [M, 1024]
__global__ void __launch_bounds__(256) k_dspark_split_kv(
    const float* __restrict__ kv_fused, float* __restrict__ kc, float* __restrict__ vc, int M) {
    const int m = blockIdx.y;
    const int c = blockIdx.x * 256 + threadIdx.x;
    if (m >= M || c >= 2 * kInkKVDim) return;
    const float v = kv_fused[(size_t)m * (2 * kInkKVDim) + c];
    if (c < kInkKVDim) kc[(size_t)m * kInkKVDim + c] = v;
    else               vc[(size_t)m * kInkKVDim + (c - kInkKVDim)] = v;
}

// ---- Ultra-Fast 2-Stage 896-CTA Parallel Top-512 Extractor (~0.03 ms on RTX 5090!) ----
// Stage 1: Grid = (128 striped chunks, 7 slots) = 896 CTAs across 170 SMs; each CTA finds Top-16 -> 2,048 pairs/slot!
__global__ void __launch_bounds__(256) k_dspark_top_cand_s1(
    const float* __restrict__ base_logits,  // [7, V]
    int V,
    int32_t* __restrict__ tmp_ids,          // [7, 128 * 16 = 2048]
    float* __restrict__ tmp_vals,           // [7, 2048]
    int ld = 0) {
    const int s = blockIdx.y;   // 0..6
    const int ch = blockIdx.x;  // 0..127
    const int tid = threadIdx.x;
    const int stride = (ld > 0) ? ld : V;
    const float* lg = base_logits + (size_t)s * stride;

    float b0 = -3e38f, b1 = -3e38f;
    int32_t i0 = 0, i1 = 0;
    for (int base = ch * 256; base < V; base += 128 * 256) {
        const int idx = base + tid;
        if (idx < V) {
            const float v = lg[idx];
            if (v > b0) { b1 = b0; i1 = i0; b0 = v; i0 = idx; }
            else if (v > b1) { b1 = v; i1 = idx; }
        }
    }
    const int warp = tid >> 5;
    const int lane = tid & 31;
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
        const size_t ob = (size_t)s * 2048 + ch * 16 + warp * 2;
        tmp_ids[ob + 0]  = wi0; tmp_vals[ob + 0] = wv0;
        tmp_ids[ob + 1]  = wi1; tmp_vals[ob + 1] = wv1;
    }
}

// Stage 2: 7 CTAs x 256 threads pick exact Top-K_cand (512) from the 2,048 Stage-1 candidates in shared memory (~3 us)!
__global__ void __launch_bounds__(256) k_dspark_top_cand_s2(
    const int32_t* __restrict__ tmp_ids,    // [7, 2048]
    const float* __restrict__ tmp_vals,     // [7, 2048]
    int K_cand,
    int32_t* __restrict__ cand_ids,         // [7, K_cand]
    float* __restrict__ cand_base) {        // [7, K_cand]
    __shared__ int32_t s_id[2048];
    __shared__ float   s_val[2048];
    __shared__ int hist[256];
    __shared__ uint32_t prefix_s, thr_s;
    __shared__ int need_s, cnt;
    const int s = blockIdx.x;
    const int tid = threadIdx.x;
    for (int i = tid; i < 2048; i += 256) {
        s_id[i]  = tmp_ids[(size_t)s * 2048 + i];
        s_val[i] = tmp_vals[(size_t)s * 2048 + i];
    }
    if (tid == 0) { prefix_s = 0; need_s = K_cand; }
    __syncthreads();
    for (int pass = 0; pass < 4; ++pass) {
        const int shift = 24 - 8 * pass;
        hist[tid] = 0;
        __syncthreads();
        const uint32_t pre = prefix_s;
        for (int i = tid; i < 2048; i += 256) {
            const uint32_t k = dspark_f2key(s_val[i]);
            if (pass == 0 || (k >> (shift + 8)) == (pre >> (shift + 8))) atomicAdd(&hist[(k >> shift) & 255u], 1);
        }
        __syncthreads();
        if (tid == 0) {
            int acc = 0, d = 255;
            const int need = need_s;
            for (; d > 0; --d) {
                if (acc + hist[d] >= need) break;
                acc += hist[d];
            }
            need_s = need - acc;
            prefix_s = pre | ((uint32_t)d << shift);
        }
        __syncthreads();
    }
    if (tid == 0) { thr_s = prefix_s; cnt = 0; }
    __syncthreads();
    const uint32_t thr = thr_s;
    int32_t* o_id = cand_ids + (size_t)s * K_cand;
    float* o_val = cand_base + (size_t)s * K_cand;
    for (int i = tid; i < 2048; i += 256) {
        if (dspark_f2key(s_val[i]) > thr) {
            int p = atomicAdd(&cnt, 1);
            if (p < K_cand) { o_id[p] = s_id[i]; o_val[p] = s_val[i]; }
        }
    }
    __syncthreads();
    if (cnt < K_cand) {
        for (int i = tid; i < 2048; i += 256) {
            if (dspark_f2key(s_val[i]) == thr) {
                int p = atomicAdd(&cnt, 1);
                if (p < K_cand) { o_id[p] = s_id[i]; o_val[p] = s_val[i]; }
            }
        }
    }
}

__global__ void __launch_bounds__(1024) k_dspark_top_cand(
    const float* __restrict__ base_logits,  // [7, V]
    int V, int K_cand,
    int32_t* __restrict__ cand_ids,         // [7, K_cand]
    float* __restrict__ cand_base) {        // [7, K_cand]
    __shared__ int hist[256];
    __shared__ uint32_t prefix_s, thr_s;
    __shared__ int need_s, cnt;
    const float* lg = base_logits + (size_t)blockIdx.x * V;
    int32_t* o_id = cand_ids + (size_t)blockIdx.x * K_cand;
    float* o_val = cand_base + (size_t)blockIdx.x * K_cand;
    if (threadIdx.x == 0) { prefix_s = 0; need_s = K_cand; }
    __syncthreads();
    for (int pass = 0; pass < 4; ++pass) {
        const int shift = 24 - 8 * pass;
        if (threadIdx.x < 256) hist[threadIdx.x] = 0;
        __syncthreads();
        const uint32_t pre = prefix_s;
        for (int i = threadIdx.x; i < V; i += 1024) {
            const uint32_t k = dspark_f2key(lg[i]);
            if (pass == 0 || (k >> (shift + 8)) == (pre >> (shift + 8))) atomicAdd(&hist[(k >> shift) & 255u], 1);
        }
        __syncthreads();
        if (threadIdx.x == 0) {
            int acc = 0, d = 255;
            const int need = need_s;
            for (; d > 0; --d) {
                if (acc + hist[d] >= need) break;
                acc += hist[d];
            }
            need_s = need - acc;
            prefix_s = pre | ((uint32_t)d << shift);
        }
        __syncthreads();
    }
    if (threadIdx.x == 0) { thr_s = prefix_s; cnt = 0; }
    __syncthreads();
    const uint32_t thr = thr_s;
    for (int i = threadIdx.x; i < V; i += 1024) {
        if (dspark_f2key(lg[i]) > thr) {
            int s = atomicAdd(&cnt, 1);
            if (s < K_cand) { o_id[s] = i; o_val[s] = lg[i]; }
        }
    }
    __syncthreads();
    if (cnt < K_cand) {
        for (int i = threadIdx.x; i < V; i += 1024) {
            if (dspark_f2key(lg[i]) == thr) {
                int s = atomicAdd(&cnt, 1);
                if (s < K_cand) { o_id[s] = i; o_val[s] = lg[i]; }
            }
        }
    }
}

// Multi-CTA High-Speed Markov RNN Helpers (84x faster than 1-CTA: ~5 us/step across 170 SMs!)
__global__ void __launch_bounds__(256) k_dspark_markov_prep(
    const int32_t* __restrict__ prev_tok_ptr,
    const float* __restrict__ state,      // [256]
    const float* __restrict__ hidden,     // [2304]
    const float* __restrict__ f_rnn_base, // [frnn_cap, 2304]
    const int32_t* __restrict__ apos_ptr, // [1]
    int frnn_cap,
    const uint8_t* __restrict__ mw1_raw, int mw1_fmt,
    float* __restrict__ joint_out) {      // [5120]
    constexpr int R = kInkMarkovRank;
    const int tid = threadIdx.x;
    const int32_t pt = *prev_tok_ptr;
    const int apos = *apos_ptr;
    const int fidx = (apos > 0 ? (apos - 1) : 0) % frnn_cap;
    const float* ctx = f_rnn_base + (size_t)fidx * kInkHidden;

    joint_out[tid] = state[tid];
    float mv = 0.0f;
    if (mw1_fmt == FMT_BF16) mv = bf16_to_f32(reinterpret_cast<const uint16_t*>(mw1_raw)[(size_t)pt * R + tid]);
    else if (mw1_fmt == FMT_F32) mv = reinterpret_cast<const float*>(mw1_raw)[(size_t)pt * R + tid];
    else {
        const float sc = reinterpret_cast<const float*>(mw1_raw)[pt];
        mv = (float)reinterpret_cast<const int8_t*>(mw1_raw + (size_t)kInkVocabPad * 4)[(size_t)pt * R + tid] * sc;
    }
    joint_out[R + tid] = mv;
    for (int d = tid; d < kInkHidden; d += 256) {
        joint_out[2 * R + d]              = hidden[d];
        joint_out[2 * R + kInkHidden + d] = ctx[d];
    }
}

__global__ void __launch_bounds__(256) k_dspark_markov_gru_upd(
    const float* __restrict__ proj,  // [768]
    const float* __restrict__ rp_b,  // [768]
    float* __restrict__ state,       // [256] in/out
    float* __restrict__ lat_out) {   // [256]
    constexpr int R = kInkMarkovRank;
    const int tid = threadIdx.x;
    const float p0 = proj[tid]         + rp_b[tid];
    const float p1 = proj[R + tid]     + rp_b[R + tid];
    const float p2 = proj[2 * R + tid] + rp_b[2 * R + tid];
    const float g = 1.0f / (1.0f + __expf(-p0));
    const float cand = tanhf(p1);
    state[tid] = cand + g * (state[tid] - cand);
    lat_out[tid] = tanhf(p2);
}

__global__ void __launch_bounds__(256) k_dspark_add_bias_and_h(
    float* __restrict__ corr,
    const float* __restrict__ ho_b,
    const float* __restrict__ hs,
    bool add_hs) {
    const int d = blockIdx.x * 256 + threadIdx.x;
    if (d < kInkHidden) corr[d] = corr[d] + ho_b[d] + (add_hs ? hs[d] : 0.0f);
}

// Multi-CTA L2-cached candidate scorer: Grid = (K_cand / 8 = 32 CTAs), 1 warp per candidate row -> ~4 us on RTX 5090!
__global__ void __launch_bounds__(256) k_dspark_score_cand_eval(
    const uint8_t* __restrict__ unemb_raw,
    int unemb_fmt,
    int unemb_rows,
    const float* __restrict__ corr,         // [2304]
    const int32_t* __restrict__ cand_ids,   // [K_cand]
    const float* __restrict__ cand_base,    // [K_cand] (nullable -> 0.0f)
    int K_cand,
    float* __restrict__ cand_scores) {      // [K_cand]
    __shared__ float s_corr[kInkHidden];
    const int tid = threadIdx.x;
    const int warp = tid >> 5;
    const int lane = tid & 31;
    for (int d = tid; d < kInkHidden; d += 256) s_corr[d] = corr[d];
    __syncthreads();

    const int k = blockIdx.x * 8 + warp;
    if (k >= K_cand) return;
    const int32_t row = cand_ids[k];
    float acc = 0.0f;
    if (unemb_fmt == FMT_BF16) {
        const uint4* wr = reinterpret_cast<const uint4*>(reinterpret_cast<const uint16_t*>(unemb_raw) + (size_t)row * kInkHidden);
        for (int k8 = lane; k8 < (kInkHidden / 8); k8 += 32) {
            const uint4 wv = wr[k8];
            const float2 w0 = bf16x2_to_f32x2(wv.x), w1 = bf16x2_to_f32x2(wv.y);
            const float2 w2 = bf16x2_to_f32x2(wv.z), w3 = bf16x2_to_f32x2(wv.w);
            const float4 x0 = *(const float4*)(s_corr + k8 * 8);
            const float4 x1 = *(const float4*)(s_corr + k8 * 8 + 4);
            acc = fmaf(w0.x, x0.x, acc); acc = fmaf(w0.y, x0.y, acc);
            acc = fmaf(w1.x, x0.z, acc); acc = fmaf(w1.y, x0.w, acc);
            acc = fmaf(w2.x, x1.x, acc); acc = fmaf(w2.y, x1.y, acc);
            acc = fmaf(w3.x, x1.z, acc); acc = fmaf(w3.y, x1.w, acc);
        }
    } else if (unemb_fmt == FMT_Q8ROW) {
        const float sc = reinterpret_cast<const float*>(unemb_raw)[row];
        const int8_t* qr = reinterpret_cast<const int8_t*>(unemb_raw + (size_t)unemb_rows * 4) + (size_t)row * kInkHidden;
        for (int k4 = lane; k4 < (kInkHidden / 4); k4 += 32) {
            const int32_t w4 = *(const int32_t*)(qr + k4 * 4);
            const float4 xv = *(const float4*)(s_corr + k4 * 4);
            acc = fmaf((float)(int8_t)(w4 & 0xFF), xv.x, acc);
            acc = fmaf((float)(int8_t)((w4 >> 8) & 0xFF), xv.y, acc);
            acc = fmaf((float)(int8_t)((w4 >> 16) & 0xFF), xv.z, acc);
            acc = fmaf((float)(int8_t)((w4 >> 24) & 0xFF), xv.w, acc);
        }
        acc *= sc;
    } else if (unemb_fmt == FMT_S6_K) {
        const int sb_b = ink_sb_bytes(FMT_S6_K);
        const uint8_t* rptr = unemb_raw + (size_t)row * 9 * sb_b;
        for (int d = lane; d < kInkHidden; d += 32) {
            const int s = d >> 8, idx = d & 255, g = idx >> 6;
            const uint8_t* blk = rptr + s * sb_b;
            const float eff = f16_to_f32(ld16_u(blk)) * ((float)blk[2 + g] * (1.0f / 255.0f));
            const uint8_t* pl = blk + 6;
            uint32_t lo = (pl[idx >> 1] >> ((idx & 1) * 4)) & 0x0Fu;
            uint32_t hi = (pl[128 + (idx >> 2)] >> ((idx & 3) * 2)) & 3u;
            acc = fmaf(((float)(lo | (hi << 4)) - 31.5f) * eff, s_corr[d], acc);
        }
    }
    acc = warp_sum(acc);
    if (lane == 0) {
        const float bv = cand_base ? cand_base[k] : 0.0f;
        cand_scores[k] = kInkSoftcap * tanhf((bv + acc) * (1.0f / kInkSoftcap));
    }
}

__global__ void __launch_bounds__(256) k_dspark_top2_reduce(
    const float* __restrict__ cand_scores,
    const int32_t* __restrict__ cand_ids,
    int K_cand,
    int32_t* __restrict__ out_top2_ids,
    float* __restrict__ out_top2_probs) {
    if (threadIdx.x != 0) return;
    float b1 = -3e38f, b2 = -3e38f;
    int32_t i1 = 0, i2 = 0;
    for (int k = 0; k < K_cand; ++k) {
        const float v = cand_scores[k];
        const int32_t id = cand_ids[k];
        if (v > b1 || (v == b1 && id < i1)) { b2 = b1; i2 = i1; b1 = v; i1 = id; }
        else if (v > b2 || (v == b2 && id < i2)) { b2 = v; i2 = id; }
    }
    float z = 0.0f;
    for (int k = 0; k < K_cand; ++k) z += expf(cand_scores[k] - b1);
    out_top2_ids[0] = i1;
    out_top2_ids[1] = i2;
    out_top2_probs[0] = 1.0f / z;
    out_top2_probs[1] = expf(b2 - b1) / z;
}

// ============================================================================
// True W=4 Multi-Branch On-GPU Markov Beam Tree Kernels (CUDA-Graph Capturable!)
// ============================================================================
constexpr int kInkMaxBeam = 4;
constexpr int kInkBeamPoolCap = 64;

__global__ void __launch_bounds__(256) k_dspark_beam_init(
    const int32_t* __restrict__ anchor_id,
    float* __restrict__ beam_state,       // [4, 256]
    int32_t* __restrict__ beam_prev_tok,  // [4]
    float* __restrict__ beam_logp,        // [4]
    int32_t* __restrict__ beam_node,      // [4]
    int32_t* __restrict__ pool_tok,       // [64]
    int8_t* __restrict__ pool_parent,     // [64]
    uint8_t* __restrict__ pool_depth,     // [64]
    float* __restrict__ pool_logp,        // [64]
    int32_t* __restrict__ pool_cnt) {     // [1]
    const int tid = threadIdx.x;
    for (int b = 0; b < kInkMaxBeam; ++b) beam_state[b * kInkMarkovRank + tid] = 0.0f;
    if (tid == 0) {
        const int32_t anc = *anchor_id;
        beam_prev_tok[0] = anc;
        beam_logp[0] = 0.0f;
        beam_node[0] = 0;
        for (int b = 1; b < kInkMaxBeam; ++b) {
            beam_prev_tok[b] = anc;
            beam_logp[b] = -1e30f;
            beam_node[b] = 0;
        }
        pool_tok[0] = anc;
        pool_parent[0] = -1;
        pool_depth[0] = 0;
        pool_logp[0] = 0.0f;
        *pool_cnt = 1;
    }
}

__global__ void __launch_bounds__(256) k_dspark_markov_prep_batch(
    const int32_t* __restrict__ prev_toks, // [nw]
    const float* __restrict__ states,      // [nw, 256]
    const float* __restrict__ hidden,      // [2304]
    const float* __restrict__ f_rnn_base,  // [frnn_cap, 2304]
    const int32_t* __restrict__ apos_ptr,  // [1]
    int frnn_cap,
    const uint8_t* __restrict__ mw1_raw, int mw1_fmt,
    float* __restrict__ joint_out,         // [nw, 5120]
    int nw) {
    constexpr int R = kInkMarkovRank;
    constexpr int J = 2 * R + 2 * kInkHidden;
    const int b = blockIdx.x;
    if (b >= nw) return;
    const int tid = threadIdx.x;
    const int32_t pt = prev_toks[b];
    const int apos = *apos_ptr;
    const int fidx = (apos > 0 ? (apos - 1) : 0) % frnn_cap;
    const float* ctx = f_rnn_base + (size_t)fidx * kInkHidden;
    float* jo = joint_out + (size_t)b * J;

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
        jo[2 * R + d]              = hidden[d];
        jo[2 * R + kInkHidden + d] = ctx[d];
    }
}

__global__ void __launch_bounds__(256) k_dspark_markov_gru_upd_batch(
    const float* __restrict__ proj,       // [nw, 768]
    const float* __restrict__ rp_b,       // [768]
    const float* __restrict__ state_in,   // [nw, 256]
    float* __restrict__ state_next,       // [nw, 256]
    float* __restrict__ lat_out,          // [nw, 256]
    int nw) {
    constexpr int R = kInkMarkovRank;
    const int b = blockIdx.x;
    if (b >= nw) return;
    const int tid = threadIdx.x;
    const float* pr = proj + (size_t)b * (3 * R);
    const float p0 = pr[tid]         + rp_b[tid];
    const float p1 = pr[R + tid]     + rp_b[R + tid];
    const float p2 = pr[2 * R + tid] + rp_b[2 * R + tid];
    const float g = 1.0f / (1.0f + __expf(-p0));
    const float cand = tanhf(p1);
    const float st_old = state_in[(size_t)b * R + tid];
    state_next[(size_t)b * R + tid] = cand + g * (st_old - cand);
    lat_out[(size_t)b * R + tid]    = tanhf(p2);
}

__global__ void __launch_bounds__(256) k_dspark_add_bias_and_h_batch(
    float* __restrict__ corr,       // [nw, 2304]
    const float* __restrict__ ho_b, // [2304]
    const float* __restrict__ hs,   // [2304]
    bool add_hs,
    int nw) {
    const int b = blockIdx.y;
    const int d = blockIdx.x * 256 + threadIdx.x;
    if (b >= nw || d >= kInkHidden) return;
    corr[(size_t)b * kInkHidden + d] += ho_b[d] + (add_hs ? hs[d] : 0.0f);
}

// Ultra-fast dp4a_i8 Batched Candidate Scorer: reads each K_cand INT8 unembed row ONCE for ALL nw <= 4 beams (~6 us!)
__global__ void __launch_bounds__(256) k_dspark_score_cand_dp4a_batch(
    const int8_t* __restrict__ q8,         // [V, 2304]
    const float* __restrict__ sc,          // [V]
    const float* __restrict__ corr,        // [nw, 2304]
    const int32_t* __restrict__ cand_ids,  // [K_cand]
    const float* __restrict__ cand_base,   // [K_cand] (nullable)
    int K_cand,
    int nw,
    float* __restrict__ cand_scores) {     // [nw, K_cand]
    __shared__ alignas(16) uint32_t s_c[kInkMaxBeam][kInkHidden / 4];  // 9,216 B!
    __shared__ float s_sc[kInkMaxBeam];
    __shared__ float sm[8];
    const int tid = threadIdx.x;
    const int warp = tid >> 5;
    const int lane = tid & 31;

    for (int b = 0; b < nw; ++b) {
        const float* cp = corr + (size_t)b * kInkHidden;
        float am = 0.0f;
        for (int d = tid; d < kInkHidden; d += 256) am = fmaxf(am, fabsf(cp[d]));
        am = warp_max(am);
        if (lane == 0) sm[warp] = am;
        __syncthreads();
        am = fmaxf(fmaxf(fmaxf(sm[0], sm[1]), fmaxf(sm[2], sm[3])),
                   fmaxf(fmaxf(sm[4], sm[5]), fmaxf(sm[6], sm[7])));
        const float csc = am * (1.0f / 127.0f);
        const float inv = csc > 0.0f ? (127.0f / am) : 0.0f;
        int8_t* dst = reinterpret_cast<int8_t*>(s_c[b]);
        for (int d = tid; d < kInkHidden; d += 256) dst[d] = (int8_t)max(-127, min(127, __float2int_rn(cp[d] * inv)));
        if (tid == 0) s_sc[b] = csc;
        __syncthreads();
    }

    const int k = blockIdx.x * 8 + warp;
    if (k >= K_cand) return;
    const int32_t row = cand_ids[k];
    const uint4* q16 = reinterpret_cast<const uint4*>(q8 + (size_t)row * kInkHidden);
    int32_t acc[kInkMaxBeam] = {};
    for (int i = lane; i < (kInkHidden / 16); i += 32) {
        const uint4 v = q16[i];
        const int base = i * 4;
#pragma unroll
        for (int b = 0; b < kInkMaxBeam; ++b) {
            if (b < nw) {
                const uint4 cv = *reinterpret_cast<const uint4*>(s_c[b] + base);
                int32_t a = acc[b];
                a = dp4a_i8(v.x, cv.x, a);
                a = dp4a_i8(v.y, cv.y, a);
                a = dp4a_i8(v.z, cv.z, a);
                a = dp4a_i8(v.w, cv.w, a);
                acc[b] = a;
            }
        }
    }
    const float sc_r = sc[row];
    const float bv = cand_base ? cand_base[k] : 0.0f;
#pragma unroll
    for (int b = 0; b < kInkMaxBeam; ++b) {
        if (b < nw) {
            int32_t s = acc[b];
#pragma unroll
            for (int o = 16; o >= 1; o >>= 1) s += __shfl_xor_sync(0xFFFFFFFFu, s, o);
            if (lane == 0) {
                const float val = bv + (float)s * sc_r * s_sc[b];
                cand_scores[(size_t)b * K_cand + k] = kInkSoftcap * tanhf(val * (1.0f / kInkSoftcap));
            }
        }
    }
}

// Parallel 256-thread Beam Expansion: exact log-softmax over K_cand -> top-4 children/beam -> record top-8 into pool + advance top-W beams!
__global__ void __launch_bounds__(256) k_dspark_beam_expand(
    const float* __restrict__ cand_scores,  // [nw, K_cand]
    const int32_t* __restrict__ cand_ids,   // [K_cand]
    const float* __restrict__ state_next,   // [nw, 256]
    int K_cand,
    int nw,
    int beam_w,
    int depth,
    float* __restrict__ state_out,          // [beam_w, 256]
    int32_t* __restrict__ beam_prev_tok,    // [beam_w]
    float* __restrict__ beam_logp,          // [beam_w] in/out
    int32_t* __restrict__ beam_node,        // [beam_w] in/out
    int32_t* __restrict__ pool_tok,         // [64]
    int8_t* __restrict__ pool_parent,       // [64]
    uint8_t* __restrict__ pool_depth,       // [64]
    float* __restrict__ pool_logp,          // [64]
    int32_t* __restrict__ pool_cnt) {       // [1]
    const int tid = threadIdx.x;
    const int warp = tid >> 5;
    const int lane = tid & 31;
    __shared__ float s_wval[8];
    __shared__ int32_t s_widx[8];
    __shared__ float sh_sum[32];
    __shared__ float s_mx, s_logz;
    __shared__ int32_t s_picked[4];

    __shared__ int32_t exp_tok[16];
    __shared__ float   exp_lp[16];
    __shared__ int32_t exp_pnode[16];
    __shared__ int32_t exp_pbeam[16];
    __shared__ int32_t win_beam[kInkMaxBeam];

    for (int b = 0; b < nw; ++b) {
        const float* sc = cand_scores + (size_t)b * K_cand;
        // 1. Max score
        float mv = -3e38f;
        for (int k = tid; k < K_cand; k += 256) mv = fmaxf(mv, sc[k]);
        mv = warp_max(mv);
        if (lane == 0) s_wval[warp] = mv;
        __syncthreads();
        if (tid == 0) {
            float g = s_wval[0];
            for (int w = 1; w < 8; ++w) g = fmaxf(g, s_wval[w]);
            s_mx = g;
        }
        __syncthreads();
        // 2. Partition function Z = sum exp(sc[k] - s_mx)
        const float mx_v = s_mx;
        float z = 0.0f;
        for (int k = tid; k < K_cand; k += 256) z += __expf(sc[k] - mx_v);
        z = block_sum(z, sh_sum);
        if (tid == 0) s_logz = mx_v + logf(fmaxf(z, 1e-30f));
        __syncthreads();
        const float logz = s_logz;
        const float parent_lp = beam_logp[b];
        const int32_t parent_nd = beam_node[b];

        // 3. Extract top-4 distinct candidates for beam b
        for (int r = 0; r < 4; ++r) {
            float bv = -3e38f;
            int32_t bk = -1;
            for (int k = tid; k < K_cand; k += 256) {
                bool skip = false;
                for (int j = 0; j < r; ++j) if (k == s_picked[j]) skip = true;
                if (!skip) {
                    const float v = sc[k];
                    if (v > bv || (v == bv && (bk < 0 || cand_ids[k] < cand_ids[bk]))) {
                        bv = v; bk = k;
                    }
                }
            }
#pragma unroll
            for (int o = 16; o >= 1; o >>= 1) {
                float ov = __shfl_xor_sync(0xFFFFFFFFu, bv, o);
                int32_t ok = __shfl_xor_sync(0xFFFFFFFFu, bk, o);
                if (ok >= 0 && (bk < 0 || ov > bv || (ov == bv && cand_ids[ok] < cand_ids[bk]))) {
                    bv = ov; bk = ok;
                }
            }
            if (lane == 0) { s_wval[warp] = bv; s_widx[warp] = bk; }
            __syncthreads();
            if (tid == 0) {
                float gv = -3e38f; int32_t gk = -1;
                for (int w = 0; w < 8; ++w) {
                    int32_t ok = s_widx[w];
                    float ov = s_wval[w];
                    if (ok >= 0 && (gk < 0 || ov > gv || (ov == gv && cand_ids[ok] < cand_ids[gk]))) {
                        gv = ov; gk = ok;
                    }
                }
                s_picked[r] = gk;
                const int idx = b * 4 + r;
                exp_tok[idx]   = (gk >= 0) ? cand_ids[gk] : 0;
                exp_lp[idx]    = (gk >= 0) ? (parent_lp + (gv - logz)) : -1e30f;
                exp_pnode[idx] = parent_nd;
                exp_pbeam[idx] = b;
            }
            __syncthreads();
        }
    }

    if (tid == 0) {
        const int n_exp = nw * 4;
        int ord[16];
        for (int i = 0; i < n_exp; ++i) ord[i] = i;
        // Insertion sort descending by joint log-prob
        for (int i = 1; i < n_exp; ++i) {
            int key = ord[i];
            int j = i - 1;
            while (j >= 0 && exp_lp[ord[j]] < exp_lp[key]) {
                ord[j + 1] = ord[j];
                --j;
            }
            ord[j + 1] = key;
        }
        const int base = *pool_cnt;
        const int k_rec = min(n_exp, min((depth == 1) ? 4 : 8, kInkBeamPoolCap - base));
        for (int j = 0; j < k_rec; ++j) {
            const int id = ord[j];
            const int dst = base + j;
            pool_tok[dst]    = exp_tok[id];
            pool_parent[dst] = (int8_t)exp_pnode[id];
            pool_depth[dst]  = (uint8_t)depth;
            pool_logp[dst]   = exp_lp[id];
        }
        *pool_cnt = base + k_rec;

        const int k_keep = min(beam_w, k_rec);
        for (int j = 0; j < beam_w; ++j) {
            if (j < k_keep) {
                const int id = ord[j];
                beam_prev_tok[j] = exp_tok[id];
                beam_logp[j]     = exp_lp[id];
                beam_node[j]     = base + j;
                win_beam[j]      = exp_pbeam[id];
            } else {
                beam_prev_tok[j] = exp_tok[ord[0]];
                beam_logp[j]     = -1e30f;
                beam_node[j]     = base;
                win_beam[j]      = exp_pbeam[ord[0]];
            }
        }
    }
    __syncthreads();
    for (int j = 0; j < beam_w; ++j) {
        const int pb = win_beam[j];
        state_out[(size_t)j * kInkMarkovRank + tid] = state_next[(size_t)pb * kInkMarkovRank + tid];
    }
}

// ============================================================================
// SpecAdapt: the draft window follows what the card measures, not a constant per card (PERF-LONGCTX; fast mode only, STRATA_INK_SM120_SPEC_ADAPT).
// The window-size model of draft_beam_tree picks the node count M that maximises  E[tokens](M) / cost(M)  with  cost(M) = 1 + R * (1 - (15/16)^M)  (R: the MoE share of a
// step, a constant of the card's table row). Here R and the drafter's calibration are estimated online from the steps the engine runs:
//   * R: a decayed least-squares fit of the measured step time  t = a + b * x,  x = 1 - (15/16)^M  (R = b / a): a card whose cold experts cost more per extra row (a hybrid
//     card, a long context) measures a larger R and gets a smaller window; a drafter whose rows are cheap gets a larger one;
//   * accept scale: the measured share of the predicted extra tokens (emitted - 1) / (E[tokens] - 1) that really came out, applied to the model's expectation;
//   * the plain step is the M = 1 bucket of the same model, so drafting switches itself off (the caller's plain margin allows it) when no window pays.
// The window changes with timing, so the exact form keeps the static model (the table's R).
// ============================================================================
struct SpecAdapt {
    bool on = false;
    double sw = 0, sx = 0, sy = 0, sxx = 0, sxy = 0;   // decayed sums of the fit
    double accept = 1.0;                                // EMA of the measured / predicted extra tokens
    long steps = 0;
    static double xof(int M) { return 1.0 - std::pow(15.0 / 16.0, (double)M); }
    void observe(int M, double ms, int n_emit, double pred_etok, int pred_m) {
        if (!on || M < 1 || ms <= 0.0 || ms > 2000.0) return;
        const double d = 0.985;
        sw *= d; sx *= d; sy *= d; sxx *= d; sxy *= d;
        const double x = xof(M);
        sw += 1; sx += x; sy += ms; sxx += x * x; sxy += x * ms;
        if (M == pred_m && M >= 3 && pred_etok > 1.05) {
            const double q = std::min(1.5, std::max(0.0, ((double)n_emit - 1.0) / (pred_etok - 1.0)));
            accept = 0.97 * accept + 0.03 * q;
        }
        ++steps;
    }
    // R for the window model; the prior (the table row's constant) until the fit has seen enough steps with different window sizes
    double ratio(double prior) const {
        if (!on || sw < 12.0) return prior;
        const double den = sw * sxx - sx * sx;
        if (den < 1e-3 * sw * sw) return prior;
        const double b = (sw * sxy - sx * sy) / den;
        const double a = (sy - b * sx) / sw;
        if (!(a > 0.5) || !(b > 0.0)) return prior;
        return std::min(60.0, std::max(0.5, b / a));
    }
    double accept_scale() const { return on ? std::min(1.2, std::max(0.25, accept)) : 1.0; }
};

// ============================================================================
// InkDSpark: On-GPU 3-Layer Block-8 / Tree-16 Recurrent Markov Drafter
// ============================================================================
struct InkDSpark {
    SpecAdapt adapt;   // the online window model (off unless the backend turns it on)
    InkCudaRunner& r;
    InkCudaModel& m;
    bool ready = false;
    int max_ctx = 4096;
    double last_pred_etok = 0.0;  // the last tree step: expected tokens (1 + sum of the kept paths' probabilities) and its node count
    int last_m = 0;
    int last_deepest_node = 0;
    float last_deepest_logp = -1e30f;
    int K_cand = 256;  // reduced-vocab L2-cached candidate pool for Markov RNN

    // Context KV cache for 3 DSpark layers (layers 0 & 2 = compact SWA-1025 rings; layer 1 = elastic global!) + compact f_rnn ring!
    int ctx_cap[3] = {0, 0, 0};
    int frnn_cap = 64;
    __half* d_ctx_k[3] = {};
    __half* d_ctx_v[3] = {};
    float*  d_f_rnn = nullptr;  // [frnn_cap, 2304] (fixed < 5 MiB regardless of max_ctx!)

    // Working buffers for Block-8 forward
    float *d_cur = nullptr, *d_h = nullptr, *d_q = nullptr, *d_k = nullptr, *d_v = nullptr;
    float *d_qkv_fused = nullptr, *d_gu_fused = nullptr;
    float *d_attn_out = nullptr, *d_proj = nullptr, *d_g = nullptr, *d_u = nullptr, *d_act = nullptr;
    float *d_ctx_tmp = nullptr, *d_kc_tmp = nullptr, *d_vc_tmp = nullptr, *d_kv_ctx_fused = nullptr;
    float *d_base_logits = nullptr;  // [7, V_unpadded] (slots 1..7)
    int32_t *d_cand_ids = nullptr;   // [7, K_cand]
    float   *d_cand_base = nullptr;  // [7, K_cand]
    int32_t *d_top_tmp_ids = nullptr;// [7, 1024]
    float   *d_top_tmp_vals = nullptr;// [7, 1024]
    int8_t  *d_h_q8 = nullptr;       // [7, 2304]
    float   *d_h_sc = nullptr;       // [7]
    float   *d_cand_scores = nullptr;// [K_cand]
    float   *d_rnn_state = nullptr;  // [256]
    float   *d_rnn_joint = nullptr;  // [5120]
    float   *d_rnn_proj  = nullptr;  // [768]
    float   *d_rnn_lat   = nullptr;  // [256]
    float   *d_corr = nullptr;       // [2304]
    int32_t *d_pos = nullptr;        // [max_ctx]
    int32_t *d_apos = nullptr;       // [1]
    int32_t *d_anchor_id = nullptr;  // [1]
    int32_t *d_draft_ids = nullptr;  // [8] (slot 0 = anchor, slots 1..7 = drafted)
    int32_t *d_top2_ids = nullptr;   // [7, 2]
    float   *d_top2_probs = nullptr; // [7, 2]
    int32_t *h_top2_ids = nullptr;   // pinned [7, 2]
    float   *h_top2_probs = nullptr; // pinned [7, 2]
    int32_t *h_meta_pin = nullptr;   // pinned [16] for async anchor/apos/pos
    int32_t *h_pos_pin = nullptr;    // pinned [max_batch] for zero-sync ingest_taps_from!
    cudaGraphExec_t draft_graph = nullptr;
    cudaGraphExec_t beam_graph = nullptr;
    int beam_graph_w = 0;

    // W=4 Multi-Branch Beam Tree buffers (~150 KB total!)
    float   *d_beam_state = nullptr;      // [4, 256]
    float   *d_beam_state_next = nullptr; // [4, 256]
    int32_t *d_beam_prev_tok = nullptr;   // [4]
    float   *d_beam_logp = nullptr;       // [4]
    int32_t *d_beam_node = nullptr;       // [4]
    float   *d_rnn_joint_b = nullptr;     // [4, 5120]
    float   *d_rnn_proj_b = nullptr;      // [4, 768]
    float   *d_rnn_lat_b = nullptr;       // [4, 256]
    float   *d_corr_b = nullptr;          // [4, 2304]
    float   *d_cand_scores_b = nullptr;   // [4, K_cand]
    int32_t *d_pool_tok = nullptr;        // [64]
    int8_t  *d_pool_parent = nullptr;     // [64]
    uint8_t *d_pool_depth = nullptr;      // [64]
    float   *d_pool_logp = nullptr;       // [64]
    int32_t *d_pool_cnt = nullptr;        // [1]
    int32_t *h_pool_tok = nullptr;        // pinned [64]
    int8_t  *h_pool_parent = nullptr;     // pinned [64]
    uint8_t *h_pool_depth = nullptr;      // pinned [64]
    float   *h_pool_logp = nullptr;       // pinned [64]
    int32_t *h_pool_cnt = nullptr;        // pinned [1]
    // Pre-dequantized BF16 (sm_80+) or FP16 (sm_70/75) cache of DSpark's 3 tiny layers (+ fused kv_ctx!)
    std::unordered_map<std::string, uint16_t*> bf16_cache;
    uint16_t* d_bf16_pool = nullptr;
    __half*   d_unembed_fp16 = nullptr;  // Optional FP16 unembed on V100 sm_70 (32GB) -> 0.67 ms TC screening!
    bool cache_is_fp16 = false;
    bool ctx_is_host[3] = {false, false, false};
    void* h_ctx_k_raw[3] = {nullptr, nullptr, nullptr};
    void* h_ctx_v_raw[3] = {nullptr, nullptr, nullptr};

    // Opt-in Multi-Slot Batched DSpark (STRATA_INK_RTX5060TI=1 + parallel >= 2):
    // Mapped-host per-slot context KV (0 MiB VRAM!) + tiny (~18 MiB) B_ch=16 batched scratch -> drafts 64 slots in ~8 ms!
    bool multislot_ready = false;
    int  ms_max_slots    = 0;
    int  ms_ctx_cap[3]   = {0, 0, 0};
    int  ms_frnn_cap     = 16;
    int  ms_k_cand       = 64;
    void*   ms_ctx_k_raw[3] = {};
    void*   ms_ctx_v_raw[3] = {};
    __half* ms_ctx_k[3]     = {};
    __half* ms_ctx_v[3]     = {};
    // Opt-in GDDR7 VRAM + Mapped-Host-Tail FP8 KV for Multi-Slot DSpark (eliminates 7 ms/step PCIe reads during C64 drafting!)
    uint8_t* ms_vram_kv_slab = nullptr;
    int      ms_vram_cap     = 0;
    int      ms_host_tail[3] = {0, 0, 0};
    InkKvVisitDesc ms_kv_desc[3] = {};
    void*    ms_tail_k_raw[3]   = {};
    void*    ms_tail_v_raw[3]   = {};
    void*    ms_tail_ksc_raw[3] = {};
    void*    ms_tail_vsc_raw[3] = {};
    float*  ms_f_rnn        = nullptr;  // [ms_max_slots, ms_frnn_cap, 2304]
    float  *ms_cur = nullptr, *ms_h = nullptr, *ms_q = nullptr, *ms_k = nullptr, *ms_v = nullptr;
    float  *ms_qkv_fused = nullptr, *ms_gu_fused = nullptr, *ms_attn_out = nullptr, *ms_proj = nullptr, *ms_act = nullptr;
    float  *ms_rnn_state = nullptr, *ms_rnn_joint = nullptr, *ms_rnn_proj = nullptr, *ms_rnn_lat = nullptr;
    float  *ms_corr = nullptr, *ms_cand_scores = nullptr;
    int32_t *ms_cand_ids = nullptr;     // [ms_max_slots, 7, ms_k_cand]
    float   *ms_cand_base = nullptr;    // [ms_max_slots, 7, ms_k_cand]
    int32_t *ms_slot_ids = nullptr, *ms_apos = nullptr, *ms_anchor_ids = nullptr, *ms_pos = nullptr;
    int32_t *ms_prev_toks = nullptr, *ms_out_drafts = nullptr;
    int32_t *ms_ing_slots = nullptr, *ms_ing_pos = nullptr, *ms_ing_rows = nullptr;
    float   *ms_taps_comp[4] = {};      // [r.max_batch, 2304] compact taps for multi-slot ingest
    int32_t *h_ms_drafts   = nullptr;   // pinned [ms_max_slots, 7]
    int32_t *h_ms_meta     = nullptr;   // pinned scratch for draft_block8_batched
    int32_t *h_ms_ing_meta = nullptr;   // 64-slot ring buffer for ingest_taps_slots/selected (zero async overwrite across 64 slots!)
    size_t   ms_ing_ring_stride = 0;
    size_t   ms_ing_ring_idx    = 0;
    uint8_t *ms_slab            = nullptr;
    // Vocab-tiled INT8 Tensor Core screener scratch (~18.7 MiB total; reads 441 MiB unembed_screen ONCE across all 64 slots!)
    int8_t  *ms_h_q8        = nullptr;  // [R_pad, 2304]
    float   *ms_h_sc        = nullptr;  // [R_pad]
    int32_t *ms_tile_i32    = nullptr;  // [R_pad, ms_vt_nom]
    int32_t *ms_top_tmp_ids = nullptr;  // [R_pad, 2048]
    float   *ms_top_tmp_vals= nullptr;  // [R_pad, 2048]
    int      ms_vt_nom      = 0;

    // Opt-in DFlash 2 drafter (STRATA_INK_DRAFTER=dflash2): when set, the entry points below hand over to it and DSpark's own buffers are not allocated.
    InkDFlash2* df2 = nullptr;

    InkDSpark(InkCudaRunner& runner) : r(runner), m(runner.m) {}

    bool init_dflash2(int ctx) {
        const std::string path = InkDFlash2::resolve_path(df2_model_path());
        if (path.empty()) {
            std::fprintf(stderr, "[InkDFlash2] DFlash 2 is the drafter, but no weights file was found (STRATA_INK_DFLASH2, or dflash2.safetensors next to "
                                 "the model)\n");
            return false;
        }
        int slots = std::max(1, r.max_slots);
        if (const char* e = std::getenv("STRATA_INK_DFLASH2_SLOTS")) slots = std::max(1, std::min(slots, std::atoi(e)));
        else {
            size_t vf_s = 0, vt_s = 0;
            (void)ink_cuda_mem_info(&vf_s, &vt_s);
            const bool small_card = m.all_sf4_vram || (vt_s > 0 && vt_s < (24ull << 30));
            slots = std::min(slots, small_card ? 16 : 32);   // rings are 42 MiB a slot; a 16 GB card keeps its memory for the experts, slots above 16 of it decode without a drafter; on a 32 GB card DFlash 2 won +12% at 32 slots and lost 7% at 64 (RTX 5090, fast), so batches of more than 32 decode plainly
        }
        auto d = std::make_unique<InkDFlash2>(r);
        std::string err;
        size_t vf0 = 0, vt0 = 0, vf1 = 0;
        (void)ink_cuda_mem_info(&vf0, &vt0);
        const auto t_load0 = std::chrono::steady_clock::now();
        if (!d->load(path, slots, err)) {
            std::fprintf(stderr, "[InkDFlash2] %s\n", err.c_str());
            d->free_all();
            return false;
        }
        (void)ink_cuda_mem_info(&vf1, &vt0);
        df2 = d.release();
        max_ctx = ctx;
        ready = true;
        // the 5060 Ti family's multi-user chain verify asks for this flag (it was DSpark's mapped-host multi-slot mode)
        if (ink_rtx5060ti_cfg(m.primary_gpu).c64_spec && r.max_slots >= 16 && m.all_sf4_vram) multislot_ready = true;
        std::fprintf(stderr, "[InkDFlash2] DFlash 2 drafter on: %s, %d slot(s), weights %.0f MiB + scratch %.0f MiB (GPU memory taken %.0f MiB), codebooks on the %s, %s block weights, loaded in %.2f s\n",
                     path.c_str(), df2->max_slots, (double)df2->bytes_weights / 1048576.0, (double)df2->bytes_scratch / 1048576.0,
                     (double)(vf0 > vf1 ? vf0 - vf1 : 0) / 1048576.0, df2->host_codebooks ? "host (mapped)" : "GPU", df2->var.attn == 0 ? "bf16" : (df2->var.attn == 1 && df2->var.ffn == 1 ? "Q8" : (df2->var.attn == 2 ? "4-bit groups" : (df2->var.ffn == 3 ? (df2->var.attn == 1 ? "mix Q8 + SF4" : "SF4") : "mixed"))),
                     std::chrono::duration<double>(std::chrono::steady_clock::now() - t_load0).count());
        return true;
    }

    void init(int ctx, int k_cand = 512, int init_glob_ctx = 0) {
        if (!m.man.has_dspark) return;
        if (InkDFlash2::requested()) {
            if (init_dflash2(ctx)) return;
        }
        if (!df2_dspark_selected()) {   // DFlash 2 is the only drafter (owner, 2026-11-26): no sidecar, or it could not start, means no drafter, not DSpark
            std::fprintf(stderr, "[InkDFlash2] this run has no drafter (DFlash 2 is the only drafter and it is not used here: no sidecar file, 2..15 slots on a card below 24 GiB, or it could not start; STRATA_INK_DRAFTER=dspark selects the dormant DSpark)\n");
            ready = false;
            return;
        }
        if (m.man.dspark.empty()) {
            std::fprintf(stderr, "[InkDSpark] the model file carries no DSpark weights; this run has no drafter\n");
            ready = false;
            return;
        }
        max_ctx = ctx;
        K_cand = m.all_sf4_vram ? std::min(k_cand, 128) : k_cand;
        const int swa_cap = std::min(max_ctx, m.man.c.dspark_swa_win + std::max(r.max_batch, 64));
        const int def_glob = rtx5090::dspark_opt_on() ? 32768
                           : ((ink_rtx5060ti_cfg(m.primary_gpu).c64_spec && r.max_slots >= 16) ? 256 : 8192);
        const int glob_init = (init_glob_ctx > 0) ? std::min(max_ctx, std::max(256, init_glob_ctx)) : std::min(max_ctx, def_glob);
        frnn_cap = std::max(64, r.max_batch + 32);

        // Pre-check free VRAM so DSpark NEVER crashes with CUDA error 2 (exit 3) on tight 4-16 GB GPUs!
        // Shrink initial global DSpark KV cap (8192 -> 2048 -> 512) before giving up (#6633)!
        int eff_glob_init = glob_init;
        {
            size_t vf = 0, vt = 0;
            if (ink_cuda_mem_info(&vf, &vt) == cudaSuccess) {
                auto calc_need = [&](int gcap) -> size_t {
                    size_t kv_need = 0;
                    for (int il = 0; il < 3; ++il) {
                        const int c = (m.man.c.dspark_is_swa[il] != 0) ? swa_cap : gcap;
                        kv_need += (size_t)c * kInkKVHeads * kInkHeadDim * 4;
                    }
                    const size_t fixed_ov = m.all_sf4_vram ? (20ull << 20) : (80ull << 20);
                    return kv_need + (size_t)frnn_cap * kInkHidden * 4 + fixed_ov;
                };
                const size_t shrink_pad = m.all_sf4_vram ? (24ull << 20) : (96ull << 20);
                const size_t min_pad    = m.all_sf4_vram ? (12ull << 20) : (64ull << 20);
                if (vf < calc_need(eff_glob_init) + shrink_pad && eff_glob_init > 2048) {
                    eff_glob_init = 2048;
                }
                if (vf < calc_need(eff_glob_init) + shrink_pad && eff_glob_init > 512) {
                    eff_glob_init = 512;
                }
                const size_t base_need = calc_need(eff_glob_init);
                if (vf < base_need + min_pad) {
                    std::fprintf(stderr, "[InkCUDA] Skipping DSpark drafter: tight VRAM (%.0f MiB free < %.0f MiB needed)\n",
                                 (double)vf / 1048576.0, (double)(base_need + min_pad) / 1048576.0);
                    ready = false;
                    return;
                }
            }
        }
        // In C64 (max_slots >= 16) multi-slot mode, skip unused single-slot KV/logits buffers (~36 MiB VRAM saved for C64 KV prefetch!)
        if (ink_rtx5060ti_cfg(m.primary_gpu).c64_spec && r.max_slots >= 16 && m.all_sf4_vram) {
            d_ctx_tmp      = dalloc<float>((size_t)r.max_batch * kInkHidden);
            d_kc_tmp       = dalloc<float>((size_t)r.max_batch * kInkKVDim);
            d_vc_tmp       = dalloc<float>((size_t)r.max_batch * kInkKVDim);
            d_kv_ctx_fused = dalloc<float>((size_t)r.max_batch * 2 * kInkKVDim);
            d_h_q8         = dalloc<int8_t>(7ull * kInkHidden);
            d_h_sc         = dalloc<float>(7);
            init_multislot(r.max_slots, ctx);
            ready = multislot_ready;
            return;
        }
        for (int il = 0; il < 3; ++il) {
            const bool is_swa_l = (m.man.c.dspark_is_swa[il] != 0);
            if (!is_swa_l && m.all_sf4_vram) {
                const int hcap = std::min(max_ctx, 32768);
                const size_t elems = (size_t)hcap * kInkKVHeads * kInkHeadDim;
                if (InkCudaRunner::alloc_mapped_host<__half>(elems, h_ctx_k_raw[il], d_ctx_k[il]) &&
                    InkCudaRunner::alloc_mapped_host<__half>(elems, h_ctx_v_raw[il], d_ctx_v[il])) {
                    ctx_cap[il] = hcap;
                    ctx_is_host[il] = true;
                    continue;
                }
                if (h_ctx_k_raw[il]) { cudaFreeHost(h_ctx_k_raw[il]); h_ctx_k_raw[il] = nullptr; }
                if (h_ctx_v_raw[il]) { cudaFreeHost(h_ctx_v_raw[il]); h_ctx_v_raw[il] = nullptr; }
                d_ctx_k[il] = nullptr; d_ctx_v[il] = nullptr;
            }
            ctx_is_host[il] = false;
            ctx_cap[il] = is_swa_l ? swa_cap : eff_glob_init;
            d_ctx_k[il] = dalloc<__half>((size_t)ctx_cap[il] * kInkKVHeads * kInkHeadDim);
            d_ctx_v[il] = dalloc<__half>((size_t)ctx_cap[il] * kInkKVHeads * kInkHeadDim);
        }
        d_f_rnn = dalloc<float>((size_t)frnn_cap * kInkHidden);

        const int B = kInkDsparkBlock;  // 8
        d_cur         = dalloc<float>(B * kInkHidden);
        d_h           = dalloc<float>(B * kInkHidden);
        d_q           = dalloc<float>(B * kInkQDim);
        d_k           = dalloc<float>(B * kInkKVDim);
        d_v           = dalloc<float>(B * kInkKVDim);
        d_qkv_fused   = dalloc<float>(B * (kInkQDim + 2 * kInkKVDim));
        d_gu_fused    = dalloc<float>(B * (2 * kInkDsparkInter));
        d_attn_out    = dalloc<float>(B * kInkQDim);
        d_proj        = dalloc<float>(B * kInkHidden);
        d_g           = dalloc<float>(B * kInkDsparkInter);
        d_u           = dalloc<float>(B * kInkDsparkInter);
        d_act         = dalloc<float>(B * kInkDsparkInter);
        d_ctx_tmp     = dalloc<float>((size_t)r.max_batch * kInkHidden);
        d_kc_tmp      = dalloc<float>((size_t)r.max_batch * kInkKVDim);
        d_vc_tmp      = dalloc<float>((size_t)r.max_batch * kInkKVDim);
        d_kv_ctx_fused= dalloc<float>((size_t)r.max_batch * 2 * kInkKVDim);
        d_base_logits = dalloc<float>(8ull * ((m.man.c.vocab_unpadded + 7) & ~7));
        d_cand_ids    = dalloc<int32_t>(7ull * K_cand);
        d_cand_base   = dalloc<float>(7ull * K_cand);
        d_top_tmp_ids = dalloc<int32_t>(7ull * 2048);
        d_top_tmp_vals= dalloc<float>(7ull * 2048);
        d_h_q8        = dalloc<int8_t>(7ull * kInkHidden);
        d_h_sc        = dalloc<float>(7);
        d_cand_scores = dalloc<float>(K_cand);
        d_rnn_state   = dalloc<float>(kInkMarkovRank);
        d_rnn_joint   = dalloc<float>(2 * kInkMarkovRank + 2 * kInkHidden);
        d_rnn_proj    = dalloc<float>(3 * kInkMarkovRank);
        d_rnn_lat     = dalloc<float>(kInkMarkovRank);
        d_corr        = dalloc<float>(kInkHidden);
        d_pos         = dalloc<int32_t>(std::max(r.max_batch, 16));
        d_apos        = dalloc<int32_t>(1);
        d_anchor_id   = dalloc<int32_t>(1);
        d_draft_ids   = dalloc<int32_t>(16);
        d_top2_ids    = dalloc<int32_t>(14);
        d_top2_probs  = dalloc<float>(14);
        h_top2_ids    = halloc_pinned<int32_t>(14);
        h_top2_probs  = halloc_pinned<float>(14);
        h_meta_pin    = halloc_pinned<int32_t>(16);
        h_pos_pin     = halloc_pinned<int32_t>(std::max(r.max_batch, 16));

        // Allocate W=4 Multi-Branch Beam Tree buffers (~150 KB total!)
        d_beam_state      = dalloc<float>(kInkMaxBeam * kInkMarkovRank);
        d_beam_state_next = dalloc<float>(kInkMaxBeam * kInkMarkovRank);
        d_beam_prev_tok   = dalloc<int32_t>(kInkMaxBeam);
        d_beam_logp       = dalloc<float>(kInkMaxBeam);
        d_beam_node       = dalloc<int32_t>(kInkMaxBeam);
        d_rnn_joint_b     = dalloc<float>((size_t)kInkMaxBeam * (2 * kInkMarkovRank + 2 * kInkHidden));
        d_rnn_proj_b      = dalloc<float>((size_t)kInkMaxBeam * 3 * kInkMarkovRank);
        d_rnn_lat_b       = dalloc<float>((size_t)kInkMaxBeam * kInkMarkovRank);
        d_corr_b          = dalloc<float>((size_t)kInkMaxBeam * kInkHidden);
        d_cand_scores_b   = dalloc<float>((size_t)kInkMaxBeam * K_cand);
        d_pool_tok        = dalloc<int32_t>(kInkBeamPoolCap);
        d_pool_parent     = dalloc<int8_t>(kInkBeamPoolCap);
        d_pool_depth      = dalloc<uint8_t>(kInkBeamPoolCap);
        d_pool_logp       = dalloc<float>(kInkBeamPoolCap);
        d_pool_cnt        = dalloc<int32_t>(1);
        h_pool_tok        = halloc_pinned<int32_t>(kInkBeamPoolCap);
        h_pool_parent     = halloc_pinned<int8_t>(kInkBeamPoolCap);
        h_pool_depth      = halloc_pinned<uint8_t>(kInkBeamPoolCap);
        h_pool_logp       = halloc_pinned<float>(kInkBeamPoolCap);
        h_pool_cnt        = halloc_pinned<int32_t>(1);

        // Pre-pack the 3 tiny DSpark layers into fused BF16 (sm_80+) or FP16 (sm_70/75) QKV, GU, KV_CTX, O, DOWN!
        // Collapses 21+6 cuBLAS calls -> 12+3 Tensor Core calls (< 0.35 ms on 5090 & V100!)
        size_t vfree = 0, vtot = 0;
        INK_CK(ink_cuda_mem_info(&vfree, &vtot));
        cache_is_fp16 = (m.sm_cc >= 70 && m.sm_cc < 80);
        auto load_to_tc = [&](const std::string& wn, uint16_t* dst_u16) {
            const auto& ti = m.tinfo(wn);
            const int N = (int)ti.rows(), K = (int)ti.cols();
            const uint8_t* W = m.d_trunk + ti.off;
            __nv_bfloat16* dst = reinterpret_cast<__nv_bfloat16*>(dst_u16);
            if (ti.fmt == FMT_BF16) {
                if (cache_is_fp16) {
                    const size_t nw = (size_t)N * K;
                    k_ink_bf16_to_fp16<<<(nw + 255) / 256, 256, 0, r.st>>>(
                        reinterpret_cast<const __nv_bfloat16*>(W), reinterpret_cast<__half*>(dst_u16), nw);
                    return;
                }
                INK_CK(cudaMemcpyAsync(dst, W, (size_t)N * K * 2, cudaMemcpyDeviceToDevice, r.st));
            } else {
                dim3 dg((K + 255) / 256, N);
                if (ti.fmt == FMT_Q8ROW)     k_ink_q8row_to_bf16<<<dg, 256, 0, r.st>>>(W, dst, N, K);
                else if (ti.fmt == FMT_S5_K) k_ink_sk_to_bf16<FMT_S5_K><<<dg, 256, 0, r.st>>>(W, dst, N, K);
                else if (ti.fmt == FMT_S6_K) k_ink_sk_to_bf16<FMT_S6_K><<<dg, 256, 0, r.st>>>(W, dst, N, K);
                else if (ti.fmt == FMT_S8_K) k_ink_sk_to_bf16<FMT_S8_K><<<dg, 256, 0, r.st>>>(W, dst, N, K);
                if (cache_is_fp16) {
                    const size_t nw = (size_t)N * K;
                    k_ink_bf16_to_fp16<<<(nw + 255) / 256, 256, 0, r.st>>>(dst, reinterpret_cast<__half*>(dst_u16), nw);
                }
            }
        };
        size_t tot_elems = 0;
        for (int il = 0; il < 3; ++il) {
            const std::string p = "dspark.layers." + std::to_string(il);
            tot_elems += (size_t)(kInkQDim + 2 * kInkKVDim) * kInkHidden;   // fused QKV [5120, 2304]
            tot_elems += (size_t)(2 * kInkDsparkInter) * kInkHidden;        // fused GU  [18432, 2304]
            tot_elems += (size_t)(2 * kInkKVDim) * kInkHidden;              // fused KV_CTX [2048, 2304]
            if (cache_is_fp16 || m.tinfo(p + ".self_attn.o_proj.weight").fmt != FMT_BF16) tot_elems += (size_t)kInkHidden * kInkQDim;
            if (cache_is_fp16 || m.tinfo(p + ".mlp.down_proj.weight").fmt != FMT_BF16)    tot_elems += (size_t)kInkHidden * kInkDsparkInter;
        }
        // Also pre-pack Markov RNN projections (~9 MB) so SQ-10/SQ-15 never dequantize them 7x/draft!
        const std::string rn_w = "dspark.markov_head.rnn_proj.weight";
        const std::string ho_w = "dspark.markov_head.hidden_out.weight";
        const bool pack_rn = (cache_is_fp16 || m.tinfo(rn_w).fmt != FMT_BF16);
        const bool pack_ho = (cache_is_fp16 || m.tinfo(ho_w).fmt != FMT_BF16);
        if (pack_rn) tot_elems += (size_t)m.tinfo(rn_w).rows() * m.tinfo(rn_w).cols();
        if (pack_ho) tot_elems += (size_t)m.tinfo(ho_w).rows() * m.tinfo(ho_w).cols();
        const size_t bf16_guard = ink_rtx5060ti_cfg(m.primary_gpu).pack ? (32ull << 20) : (768ull << 20);
        if (!m.all_sf4_vram && m.sm_cc >= 70 && vfree > tot_elems * 2 + bf16_guard) {
            d_bf16_pool = dalloc<uint16_t>(tot_elems);
            size_t cur = 0;
            for (int il = 0; il < 3; ++il) {
                const std::string p = "dspark.layers." + std::to_string(il);
                uint16_t* qkv = d_bf16_pool + cur;
                load_to_tc(p + ".self_attn.q_proj.weight", qkv);
                load_to_tc(p + ".self_attn.k_proj.weight", qkv + (size_t)kInkQDim * kInkHidden);
                load_to_tc(p + ".self_attn.v_proj.weight", qkv + (size_t)(kInkQDim + kInkKVDim) * kInkHidden);
                bf16_cache[p + ".qkv_fused"] = qkv;
                cur += (size_t)(kInkQDim + 2 * kInkKVDim) * kInkHidden;

                uint16_t* gu = d_bf16_pool + cur;
                load_to_tc(p + ".mlp.gate_proj.weight", gu);
                load_to_tc(p + ".mlp.up_proj.weight",   gu + (size_t)kInkDsparkInter * kInkHidden);
                bf16_cache[p + ".gu_fused"] = gu;
                cur += (size_t)(2 * kInkDsparkInter) * kInkHidden;

                uint16_t* kv_ctx = d_bf16_pool + cur;
                load_to_tc(p + ".self_attn.k_proj_ctx.weight", kv_ctx);
                load_to_tc(p + ".self_attn.v_proj_ctx.weight", kv_ctx + (size_t)kInkKVDim * kInkHidden);
                bf16_cache[p + ".kv_ctx_fused"] = kv_ctx;
                cur += (size_t)(2 * kInkKVDim) * kInkHidden;

                const std::string op = p + ".self_attn.o_proj.weight";
                if (cache_is_fp16 || m.tinfo(op).fmt != FMT_BF16) {
                    load_to_tc(op, d_bf16_pool + cur);
                    bf16_cache[op] = d_bf16_pool + cur;
                    cur += (size_t)kInkHidden * kInkQDim;
                }
                const std::string dp = p + ".mlp.down_proj.weight";
                if (cache_is_fp16 || m.tinfo(dp).fmt != FMT_BF16) {
                    load_to_tc(dp, d_bf16_pool + cur);
                    bf16_cache[dp] = d_bf16_pool + cur;
                    cur += (size_t)kInkHidden * kInkDsparkInter;
                }
            }
            if (pack_rn) {
                load_to_tc(rn_w, d_bf16_pool + cur);
                bf16_cache[rn_w] = d_bf16_pool + cur;
                cur += (size_t)m.tinfo(rn_w).rows() * m.tinfo(rn_w).cols();
            }
            if (pack_ho) {
                load_to_tc(ho_w, d_bf16_pool + cur);
                bf16_cache[ho_w] = d_bf16_pool + cur;
                cur += (size_t)m.tinfo(ho_w).rows() * m.tinfo(ho_w).cols();
            }
            INK_CK(cudaStreamSynchronize(r.st));
        }
        // On Volta sm_70 (V100 32GB, lacks hardware dp4a), pre-pack FP16 unembed [V_pad, 2304] (~604 MB) when >2 GiB free
        // Pad V to multiple of 8 so Volta sm_70 cuBLAS engages 125 TFLOPS FP16 Tensor Cores (~0.67 ms vs 4.6 ms)!
        if (m.sm_cc == 70 && m.unembed_screen.ready) {
            INK_CK(ink_cuda_mem_info(&vfree, &vtot));
            const int V_pad = (m.man.c.vocab_unpadded + 7) & ~7;
            const size_t u_elems = (size_t)V_pad * kInkHidden;
            if (vfree > u_elems * 2 + (2048ull << 20)) {
                d_unembed_fp16 = dalloc<__half>(u_elems);
                const auto& ui = m.tinfo("model.llm.unembed.weight");
                const uint8_t* W = m.d_trunk + ui.off;
                const int V_rows = std::min(V_pad, (int)ui.rows());
                if (ui.fmt == FMT_BF16) {
                    k_ink_bf16_to_fp16<<<((size_t)V_rows * kInkHidden + 255) / 256, 256, 0, r.st>>>(
                        reinterpret_cast<const __nv_bfloat16*>(W), d_unembed_fp16, (size_t)V_rows * kInkHidden);
                } else {
                    dim3 dg((kInkHidden + 255) / 256, V_rows);
                    __nv_bfloat16* dst_b = reinterpret_cast<__nv_bfloat16*>(d_unembed_fp16);
                    if (ui.fmt == FMT_S5_K)      k_ink_sk_to_bf16<FMT_S5_K><<<dg, 256, 0, r.st>>>(W, dst_b, V_rows, kInkHidden);
                    else if (ui.fmt == FMT_S6_K) k_ink_sk_to_bf16<FMT_S6_K><<<dg, 256, 0, r.st>>>(W, dst_b, V_rows, kInkHidden);
                    else if (ui.fmt == FMT_S8_K) k_ink_sk_to_bf16<FMT_S8_K><<<dg, 256, 0, r.st>>>(W, dst_b, V_rows, kInkHidden);
                    else                         k_ink_q8row_to_bf16<<<dg, 256, 0, r.st>>>(W, dst_b, V_rows, kInkHidden);
                    k_ink_bf16_to_fp16<<<((size_t)V_rows * kInkHidden + 255) / 256, 256, 0, r.st>>>(
                        dst_b, d_unembed_fp16, (size_t)V_rows * kInkHidden);
                }
                INK_CK(cudaStreamSynchronize(r.st));
            }
        }
        ready = true;
        if (ink_rtx5060ti_cfg(m.primary_gpu).c64_spec && r.max_slots >= 2) {
            init_multislot(r.max_slots, ctx);
        }
    }

    // Opt-in Multi-Slot Batched DSpark Initialization (0 MiB VRAM for 64-slot KV via mapped host RAM!)
    void init_multislot(int slots, int ctx) {
        if (multislot_ready || slots < 2 || !m.unembed_screen.ready) return;
        ms_max_slots = slots;
        ms_k_cand = 128;
        ms_frnn_cap = 4;
        const int swa_c = std::min(ctx, m.man.c.dspark_swa_win + 64);
        const int glb_c = std::min(ctx, 4096);
        bool ok = true;
        for (int il = 0; il < 3 && ok; ++il) {
            ms_ctx_cap[il] = (m.man.c.dspark_is_swa[il] != 0) ? swa_c : glb_c;
            const size_t elems = (size_t)ms_max_slots * ms_ctx_cap[il] * kInkKVHeads * kInkHeadDim;
            ok = InkCudaRunner::alloc_mapped_host<__half>(elems, ms_ctx_k_raw[il], ms_ctx_k[il])
              && InkCudaRunner::alloc_mapped_host<__half>(elems, ms_ctx_v_raw[il], ms_ctx_v[il]);
        }
        if (!ok) {
            free_multislot();
            return;
        }
        constexpr int kBCh = 32;
        constexpr int kMCh = kBCh * kInkDsparkBlock;  // 256
        try {
            const int V_u = (m.unembed_screen.V_pad > 0) ? m.unembed_screen.V_pad : m.man.c.vocab_unpadded;
            const int max_kd = std::max(3, ink_rtx5060ti_cfg(m.primary_gpu).c64_draft_k);
            const int R_pad = ((std::max(r.max_batch, ms_max_slots * max_kd) + 3) / 4) * 4;
            ms_vt_nom = ((V_u + 7) / 8 + 3) & ~3;
            const bool want_tc_scr = (m.sm_cc >= 75 && (V_u % 4 == 0) && ms_vt_nom > 0);
            const size_t ing_cap = (size_t)std::max(r.max_batch, ms_max_slots * 8);

            // Single 256-byte-aligned VRAM slab for all ~32 multi-slot buffers (eliminates ~55 MiB of 2 MiB cudaMalloc rounding!)
            size_t slab_off = 0;
            auto bind_ms = [&](int pass) {
                slab_off = 0;
                auto al = [&](auto*& ptr, size_t n) {
                    using T = typename std::remove_pointer<typename std::remove_reference<decltype(ptr)>::type>::type;
                    const size_t o = slab_off;
                    slab_off += (n * sizeof(T) + 255ull) & ~255ull;
                    if (pass == 1) ptr = reinterpret_cast<T*>(ms_slab + o);
                };
                al(ms_f_rnn,       (size_t)ms_max_slots * ms_frnn_cap * kInkHidden);
                al(ms_cur,         (size_t)ms_max_slots * kInkDsparkBlock * kInkHidden);
                al(ms_h,           (size_t)kMCh * kInkHidden);
                al(ms_q,           (size_t)kMCh * kInkQDim);
                al(ms_k,           (size_t)kMCh * kInkKVDim);
                al(ms_v,           (size_t)kMCh * kInkKVDim);
                al(ms_qkv_fused,   (size_t)kMCh * (kInkQDim + 2 * kInkKVDim));
                al(ms_gu_fused,    (size_t)kMCh * (2 * kInkDsparkInter));
                al(ms_attn_out,    (size_t)kMCh * kInkQDim);
                al(ms_proj,        (size_t)kMCh * kInkHidden);
                al(ms_act,         (size_t)kMCh * kInkDsparkInter);
                al(ms_rnn_state,   (size_t)ms_max_slots * kInkMarkovRank);
                al(ms_rnn_joint,   (size_t)ms_max_slots * (2 * kInkMarkovRank + 2 * kInkHidden));
                al(ms_rnn_proj,    (size_t)ms_max_slots * 3 * kInkMarkovRank);
                al(ms_rnn_lat,     (size_t)ms_max_slots * kInkMarkovRank);
                al(ms_corr,        (size_t)ms_max_slots * kInkHidden);
                al(ms_cand_scores, (size_t)ms_max_slots * ms_k_cand);
                al(ms_cand_ids,    (size_t)ms_max_slots * 7 * ms_k_cand);
                al(ms_cand_base,   (size_t)ms_max_slots * 7 * ms_k_cand);
                al(ms_slot_ids,    (size_t)ms_max_slots);
                al(ms_apos,        (size_t)ms_max_slots);
                al(ms_anchor_ids,  (size_t)ms_max_slots);
                al(ms_pos,         (size_t)kMCh);
                al(ms_prev_toks,   (size_t)ms_max_slots);
                al(ms_out_drafts,  (size_t)ms_max_slots * 7);
                al(ms_ing_slots,   ing_cap);
                al(ms_ing_pos,     ing_cap);
                al(ms_ing_rows,    ing_cap);
                for (int t = 0; t < 4; ++t) al(ms_taps_comp[t], (size_t)r.max_batch * kInkHidden);
                if (want_tc_scr) {
                    al(ms_h_q8,         (size_t)R_pad * kInkHidden);
                    al(ms_h_sc,         (size_t)R_pad);
                    al(ms_tile_i32,     (size_t)R_pad * ms_vt_nom);
                    al(ms_top_tmp_ids,  (size_t)R_pad * 2048);
                    al(ms_top_tmp_vals, (size_t)R_pad * 2048);
                }
            };
            bind_ms(0);
            ms_slab = dalloc<uint8_t>(slab_off);
            bind_ms(1);
            // Try placing positions [0..ms_vram_cap-1] of ALL 3 multi-slot DSpark layers in GDDR7 VRAM (K8V4, 75.0 MiB for 256 pos!)
            // with mapped host RAM tail for [ms_vram_cap..ctx-1]! Keeps BOTH DSpark KV & c64_kv_pf at cap_pos=256 with 50+ MiB free!
            if (m.all_sf4_vram && ctx <= 2048) {
                size_t vf_ds = 0, vt_ds = 0;
                if (ink_cuda_mem_info(&vf_ds, &vt_ds, m.primary_gpu) == cudaSuccess) {
                    const int base_kd_res = std::max(1, std::min(7, ink_rtx5060ti_cfg(m.primary_gpu).c64_draft_k));
                    const size_t sc_res = (size_t)kInkVisits * (size_t)ms_max_slots * (base_kd_res + 1) * kInkSconvCh * sizeof(uint16_t);
                    const size_t pf_res = 68ull << 20;  // reserve for c64_kv_pf (cap_pos=256 = 66 MiB)
                    const size_t keep_free = sc_res + pf_res + (44ull << 20);  // +44 MiB for argmax tile + 4-row logits + safety!
                    auto vram_bytes_for = [&](int dc) -> size_t {
                        const size_t kb = ((size_t)ms_max_slots * dc * kInkKVHeads * kInkHeadDim + 255ull) & ~255ull;
                        const size_t vb = ((size_t)ms_max_slots * dc * kInkKVHeads * (kInkHeadDim / 2) + 255ull) & ~255ull;
                        const size_t sb = ((size_t)ms_max_slots * dc * kInkKVHeads * sizeof(float) + 255ull) & ~255ull;
                        return 3u * (kb + vb + 2u * sb);
                    };
                    int chosen_dc = 0;
                    if (vf_ds > vram_bytes_for(256) + keep_free)      chosen_dc = 256;
                    else if (vf_ds > vram_bytes_for(192) + keep_free) chosen_dc = 192;
                    else if (vf_ds > vram_bytes_for(128) + keep_free) chosen_dc = 128;
                    if (chosen_dc > 0) {
                        const size_t tot_v = vram_bytes_for(chosen_dc);
                        if (cudaMalloc(&ms_vram_kv_slab, tot_v) == cudaSuccess) {
                            const size_t kb = ((size_t)ms_max_slots * chosen_dc * kInkKVHeads * kInkHeadDim + 255ull) & ~255ull;
                            const size_t vb = ((size_t)ms_max_slots * chosen_dc * kInkKVHeads * (kInkHeadDim / 2) + 255ull) & ~255ull;
                            const size_t sb = ((size_t)ms_max_slots * chosen_dc * kInkKVHeads * sizeof(float) + 255ull) & ~255ull;
                            size_t vo = 0;
                            bool tail_ok = true;
                            for (int il = 0; il < 3 && tail_ok; ++il) {
                                const int full_c = std::min(ctx, 2048);
                                const int htail  = std::max(0, full_c - chosen_dc);
                                ms_host_tail[il] = htail;
                                InkKvVisitDesc& d = ms_kv_desc[il];
                                d = {};
                                d.k_f8     = ms_vram_kv_slab + vo; vo += kb;
                                d.v_f8     = ms_vram_kv_slab + vo; vo += vb;
                                d.k_sc     = reinterpret_cast<float*>(ms_vram_kv_slab + vo); vo += sb;
                                d.v_sc     = reinterpret_cast<float*>(ms_vram_kv_slab + vo); vo += sb;
                                d.dev_cap  = chosen_dc;
                                d.host_cap = htail;
                                d.cap_pos  = chosen_dc + htail;
                                d.is_swa   = false;
                                d.kv_fmt   = (uint8_t)InkKvDtype::K8V4;
                                if (htail > 0) {
                                    const size_t he  = (size_t)ms_max_slots * htail * kInkKVHeads * kInkHeadDim;
                                    const size_t hve = he / 2u;
                                    const size_t hs  = (size_t)ms_max_slots * htail * kInkKVHeads;
                                    tail_ok = InkCudaRunner::alloc_mapped_host<uint8_t>(he,  ms_tail_k_raw[il],   d.h_k_f8)
                                           && InkCudaRunner::alloc_mapped_host<uint8_t>(hve, ms_tail_v_raw[il],   d.h_v_f8)
                                           && InkCudaRunner::alloc_mapped_host<float>(hs,    ms_tail_ksc_raw[il], d.h_k_sc)
                                           && InkCudaRunner::alloc_mapped_host<float>(hs,    ms_tail_vsc_raw[il], d.h_v_sc);
                                }
                            }
                            if (tail_ok) {
                                ms_vram_cap = chosen_dc;
                                // Free the unused FP16 mapped-host buffers to save ~1.6 GiB host RAM!
                                for (int il = 0; il < 3; ++il) {
                                    if (ms_ctx_k_raw[il]) { cudaFreeHost(ms_ctx_k_raw[il]); ms_ctx_k_raw[il] = nullptr; }
                                    if (ms_ctx_v_raw[il]) { cudaFreeHost(ms_ctx_v_raw[il]); ms_ctx_v_raw[il] = nullptr; }
                                    ms_ctx_k[il] = nullptr; ms_ctx_v[il] = nullptr;
                                }
                                std::fprintf(stderr, "[Ink5060Ti] Multi-slot DSpark VRAM K8V4 KV active (dev_cap=%d, %.1f MiB VRAM + host tail)\n",
                                             ms_vram_cap, (double)tot_v / 1048576.0);
                            } else {
                                cudaFree(ms_vram_kv_slab); ms_vram_kv_slab = nullptr;
                                for (int il = 0; il < 3; ++il) {
                                    if (ms_tail_k_raw[il])   { cudaFreeHost(ms_tail_k_raw[il]);   ms_tail_k_raw[il]   = nullptr; }
                                    if (ms_tail_v_raw[il])   { cudaFreeHost(ms_tail_v_raw[il]);   ms_tail_v_raw[il]   = nullptr; }
                                    if (ms_tail_ksc_raw[il]) { cudaFreeHost(ms_tail_ksc_raw[il]); ms_tail_ksc_raw[il] = nullptr; }
                                    if (ms_tail_vsc_raw[il]) { cudaFreeHost(ms_tail_vsc_raw[il]); ms_tail_vsc_raw[il] = nullptr; }
                                    ms_kv_desc[il] = {};
                                }
                            }
                        } else {
                            (void)cudaGetLastError();
                            ms_vram_kv_slab = nullptr;
                        }
                    }
                }
            }
            h_ms_drafts        = halloc_pinned<int32_t>((size_t)ms_max_slots * 7);
            h_ms_meta          = halloc_pinned<int32_t>(ing_cap * 3 + 256);
            ms_ing_ring_stride = ing_cap * 3 + 64;
            ms_ing_ring_idx    = 0;
            h_ms_ing_meta      = halloc_pinned<int32_t>(64ull * ms_ing_ring_stride);
            multislot_ready = true;
            std::fprintf(stderr, "[Ink5060Ti] Multi-slot batched DSpark ready (%d slots, %.1f MiB slab, SWA=%d / Glob=%d mapped host KV)\n",
                         ms_max_slots, (double)slab_off / 1048576.0, swa_c, glb_c);
        } catch (...) {
            (void)cudaGetLastError();
            free_multislot();
        }
    }

    // Ingest arbitrary-slot target taps [4, M, 2304] into multi-slot DSpark context KV + f_rnn!
    void ingest_taps_slots(float* const taps[4], const int32_t* h_slots, const int32_t* h_pos, int M) {
        if (df2) { if (M > 0) df2->ingest(taps, nullptr, h_slots, h_pos, M); return; }
        if (!multislot_ready || M <= 0) return;
        const int cap = std::min(M, r.max_batch);
        int32_t* ing_buf = h_ms_ing_meta + (ms_ing_ring_idx++ & 63ull) * ms_ing_ring_stride;
        std::memcpy(ing_buf, h_slots, (size_t)cap * 4);
        std::memcpy(ing_buf + cap, h_pos, (size_t)cap * 4);
        INK_CK(cudaMemcpyAsync(ms_ing_slots, ing_buf, (size_t)cap * 4, cudaMemcpyHostToDevice, r.st));
        INK_CK(cudaMemcpyAsync(ms_ing_pos,   ing_buf + cap, (size_t)cap * 4, cudaMemcpyHostToDevice, r.st));
        const float* fw = (const float*)m.trunk_ptr("dspark.fuse_weight");

        for (int il = 0; il < 3; ++il) {
            const std::string p = "dspark.layers." + std::to_string(il);
            k_dspark_mix_taps<<<cap, 256, 0, r.st>>>(
                taps[0], taps[1], taps[2], taps[3], fw, il,
                (const float*)m.trunk_ptr(p + ".dflare_norm.weight"), d_ctx_tmp, cap);
            auto it_kv = bf16_cache.find(p + ".kv_ctx_fused");
            if (it_kv != bf16_cache.end()) {
                dspark_gemm_bf16(it_kv->second, d_ctx_tmp, d_kv_ctx_fused, cap, 2 * kInkKVDim, kInkHidden, false);
                k_dspark_split_kv<<<dim3((2 * kInkKVDim + 255) / 256, cap), 256, 0, r.st>>>(d_kv_ctx_fused, d_kc_tmp, d_vc_tmp, cap);
            } else {
                r.linear(p + ".self_attn.k_proj_ctx.weight", d_ctx_tmp, d_kc_tmp, cap);
                r.linear(p + ".self_attn.v_proj_ctx.weight", d_ctx_tmp, d_vc_tmp, cap);
            }
            dim3 g(kInkKVHeads, cap);
            k_dspark_rmsnorm_rope<<<g, 128, 0, r.st>>>(
                d_kc_tmp, (const float*)m.trunk_ptr(p + ".self_attn.k_norm_ctx.weight"),
                ms_ing_pos, m.man.c.dspark_rope_theta, cap, kInkKVHeads);
            if (ms_vram_cap > 0) {
                k_ink_kv_write<<<g, 128, 0, r.st>>>(d_kc_tmp, d_vc_tmp, ms_ing_slots, 0, ms_ing_pos, ms_kv_desc[il], cap);
            } else {
                InkKvVisitDesc desc{};
                desc.k_f16   = ms_ctx_k[il];
                desc.v_f16   = ms_ctx_v[il];
                desc.cap_pos = ms_ctx_cap[il];
                desc.dev_cap = ms_ctx_cap[il];
                desc.host_cap = 0;
                desc.is_swa  = (m.man.c.dspark_is_swa[il] != 0);
                desc.kv_fmt  = 0;
                k_ink_kv_write<<<g, 128, 0, r.st>>>(d_kc_tmp, d_vc_tmp, ms_ing_slots, 0, ms_ing_pos, desc, cap);
            }
        }
        k_dspark_mix_taps_ring_slots<<<cap, 256, 0, r.st>>>(
            taps[0], taps[1], taps[2], taps[3], fw, 3,
            (const float*)m.trunk_ptr("dspark.markov_head.dflare_norm.weight"),
            ms_f_rnn, ms_ing_slots, 0, ms_ing_pos, ms_frnn_cap, cap);
    }

    // Compact selected rows `h_src_rows[0..n_tot-1]` from r.d_taps[0..3] and ingest into multi-slot DSpark!
    void ingest_taps_selected(const int32_t* h_src_rows, const int32_t* h_slots, const int32_t* h_pos, int n_tot) {
        if (df2) { if (n_tot > 0) df2->ingest(r.d_taps, h_src_rows, h_slots, h_pos, n_tot); return; }
        if (!multislot_ready || n_tot <= 0) return;
        const int cap = std::min(n_tot, r.max_batch);
        int32_t* row_buf = h_ms_ing_meta + (ms_ing_ring_idx++ & 63ull) * ms_ing_ring_stride;
        std::memcpy(row_buf, h_src_rows, (size_t)cap * 4);
        INK_CK(cudaMemcpyAsync(ms_ing_rows, row_buf, (size_t)cap * 4, cudaMemcpyHostToDevice, r.st));
        for (int t = 0; t < 4; ++t) {
            k_ink_compact_taps_rows<<<cap, 256, 0, r.st>>>(r.d_taps[t], ms_taps_comp[t], ms_ing_rows, cap);
        }
        ingest_taps_slots(ms_taps_comp, h_slots, h_pos, cap);
    }

    // Batched Multi-Slot Block-8 Drafting: drafts k_draft (1..7) tokens for each of B concurrent slots (B <= 64)!
    void draft_block8_batched(
        int B,
        const int32_t* h_slots,
        const int32_t* h_anchor_toks,
        const int32_t* h_apos,
        int32_t* out_drafts_7,
        int k_draft = 7) {
        if (df2) {
            std::vector<float> conf((size_t)std::max(1, B) * 7);
            if (B > 0 && !df2->draft(B, h_slots, h_anchor_toks, h_apos, out_drafts_7, conf.data()))
                for (int i = 0; i < B * 7; ++i) out_drafts_7[i] = h_anchor_toks[i / 7];
            return;
        }
        if (!multislot_ready || B <= 0) return;
        const int n_dr = std::max(1, std::min(7, k_draft));
        std::memcpy(h_ms_meta, h_slots, (size_t)B * 4);
        std::memcpy(h_ms_meta + B, h_anchor_toks, (size_t)B * 4);
        std::memcpy(h_ms_meta + 2 * B, h_apos, (size_t)B * 4);
        INK_CK(cudaMemcpyAsync(ms_slot_ids,   h_ms_meta,         (size_t)B * 4, cudaMemcpyHostToDevice, r.st));
        INK_CK(cudaMemcpyAsync(ms_anchor_ids, h_ms_meta + B,     (size_t)B * 4, cudaMemcpyHostToDevice, r.st));
        INK_CK(cudaMemcpyAsync(ms_apos,       h_ms_meta + 2 * B, (size_t)B * 4, cudaMemcpyHostToDevice, r.st));

        // 1. Embed anchors + mask tokens for all B slots -> ms_cur [B*8, 2304]
        const auto& ei = m.tinfo("model.llm.embed.weight");
        k_dspark_embed_mask_batch<<<dim3(kInkDsparkBlock, B), 256, 0, r.st>>>(
            m.embed_ptr(), ei.fmt,
            (const float*)m.trunk_ptr("model.llm.embed_norm.weight"),
            (const float*)m.trunk_ptr("dspark.mask_embed"),
            ms_anchor_ids, ms_cur, B);

        // 2. Run 3-layer Block-8 Transformer in chunks of B_ch = 32 slots (256 rows -> 2x fewer weight reads & full INT8 TC!)
        constexpr int kBCh = 32;
        const int V = m.man.c.vocab_unpadded;
        for (int b0 = 0; b0 < B; b0 += kBCh) {
            const int bc = std::min(kBCh, B - b0);
            const int M = bc * kInkDsparkBlock;
            float* cur_p = ms_cur + (size_t)b0 * kInkDsparkBlock * kInkHidden;
            int32_t* h_p_tmp = h_ms_meta + 3 * B + b0 * kInkDsparkBlock;
            for (int bi = 0; bi < bc; ++bi) {
                for (int s = 0; s < kInkDsparkBlock; ++s) h_p_tmp[bi * kInkDsparkBlock + s] = h_apos[b0 + bi] + s;
            }
            INK_CK(cudaMemcpyAsync(ms_pos, h_p_tmp, (size_t)M * 4, cudaMemcpyHostToDevice, r.st));

            for (int il = 0; il < 3; ++il) {
                const std::string p = "dspark.layers." + std::to_string(il);
                k_ink_rms_norm<<<M, 256, 0, r.st>>>(cur_p, (const float*)m.trunk_ptr(p + ".input_layernorm.weight"), ms_h, nullptr, M, kInkHidden);
                auto it_qkv = bf16_cache.find(p + ".qkv_fused");
                if (it_qkv != bf16_cache.end()) {
                    dspark_gemm_bf16(it_qkv->second, ms_h, ms_qkv_fused, M, kInkQDim + 2 * kInkKVDim, kInkHidden, false);
                    k_dspark_split_qkv<<<dim3((kInkQDim + 2 * kInkKVDim + 255) / 256, M), 256, 0, r.st>>>(ms_qkv_fused, ms_q, ms_k, ms_v, M);
                } else {
                    dspark_linear_b8(p + ".self_attn.q_proj.weight", ms_h, ms_q, M, false);
                    dspark_linear_b8(p + ".self_attn.k_proj.weight", ms_h, ms_k, M, true);
                    dspark_linear_b8(p + ".self_attn.v_proj.weight", ms_h, ms_v, M, true);
                }
                k_dspark_rmsnorm_rope<<<dim3(kInkHeads, M), 128, 0, r.st>>>(
                    ms_q, (const float*)m.trunk_ptr(p + ".self_attn.q_norm.weight"), ms_pos, m.man.c.dspark_rope_theta, M, kInkHeads);
                k_dspark_rmsnorm_rope<<<dim3(kInkKVHeads, M), 128, 0, r.st>>>(
                    ms_k, (const float*)m.trunk_ptr(p + ".self_attn.k_norm.weight"), ms_pos, m.man.c.dspark_rope_theta, M, kInkKVHeads);
                if (ms_vram_cap > 0) {
                    k_dspark_block_attn_batch_fp8<<<dim3(kInkKVHeads, bc), 128, 0, r.st>>>(
                        ms_q, ms_k, ms_v, ms_kv_desc[il],
                        ms_slot_ids + b0, ms_apos + b0,
                        m.man.c.dspark_is_swa[il] != 0, m.man.c.dspark_swa_win, ms_kv_desc[il].cap_pos, bc, ms_attn_out);
                } else {
                    k_dspark_block_attn_batch<<<dim3(kInkKVHeads, bc), 128, 0, r.st>>>(
                        ms_q, ms_k, ms_v, ms_ctx_k[il], ms_ctx_v[il],
                        ms_slot_ids + b0, ms_apos + b0,
                        m.man.c.dspark_is_swa[il] != 0, m.man.c.dspark_swa_win, ms_ctx_cap[il], bc, ms_attn_out);
                }
                dspark_linear_b8(p + ".self_attn.o_proj.weight", ms_attn_out, ms_proj, M, false);
                k_ink_add<<<(M * kInkHidden + 255) / 256, 256, 0, r.st>>>(cur_p, cur_p, ms_proj, M * kInkHidden);

                k_ink_rms_norm<<<M, 256, 0, r.st>>>(cur_p, (const float*)m.trunk_ptr(p + ".post_attention_layernorm.weight"), ms_h, nullptr, M, kInkHidden);
                auto it_gu = bf16_cache.find(p + ".gu_fused");
                if (it_gu != bf16_cache.end()) {
                    dspark_gemm_bf16(it_gu->second, ms_h, ms_gu_fused, M, 2 * kInkDsparkInter, kInkHidden, false);
                    k_dspark_swiglu_fused<<<dim3((kInkDsparkInter + 255) / 256, M), 256, 0, r.st>>>(ms_gu_fused, ms_act, M);
                } else {
                    dspark_linear_b8(p + ".mlp.gate_proj.weight", ms_h, ms_gu_fused, M, false);
                    dspark_linear_b8(p + ".mlp.up_proj.weight", ms_h, ms_gu_fused + (size_t)M * kInkDsparkInter, M, true);
                    k_dspark_swiglu<<<(M * kInkDsparkInter + 255) / 256, 256, 0, r.st>>>(ms_gu_fused, ms_gu_fused + (size_t)M * kInkDsparkInter, ms_act, M * kInkDsparkInter);
                }
                dspark_linear_b8(p + ".mlp.down_proj.weight", ms_act, ms_proj, M, false);
                k_ink_add<<<(M * kInkHidden + 255) / 256, 256, 0, r.st>>>(cur_p, cur_p, ms_proj, M * kInkHidden);
            }
            k_ink_rms_norm<<<M, 256, 0, r.st>>>(cur_p, (const float*)m.trunk_ptr("dspark.norm.weight"), cur_p, nullptr, M, kInkHidden);
        }

        // 3. Base logits screening + Top-ms_k_cand for ONLY B*n_dr active draft rows!
        const int V_gemm = (m.unembed_screen.V_pad > 0) ? m.unembed_screen.V_pad : V;
        if (ms_tile_i32 && (V_gemm % 4 == 0)) {
            const int R = B * n_dr;
            const int R_pad = ((R + 3) / 4) * 4;
            k_dspark_quant_h_q8_slots<<<R, 256, 0, r.st>>>(ms_cur, ms_h_q8, ms_h_sc, B, n_dr);
            if (R_pad > R) {
                INK_CK(cudaMemsetAsync(ms_h_q8 + (size_t)R * kInkHidden, 0, (size_t)(R_pad - R) * kInkHidden, r.st));
            }
            const int32_t ialpha = 1, ibeta = 0;
            for (int ti = 0; ti < 8; ++ti) {
                const int v0 = ti * ms_vt_nom;
                const int vt = std::max(0, std::min(ms_vt_nom, V_gemm - v0));
                if (vt > 0) {
                    INK_CUBLAS_CK(cublasGemmEx(
                        r.cublas, CUBLAS_OP_T, CUBLAS_OP_N,
                        vt, R_pad, kInkHidden, &ialpha,
                        m.unembed_screen.q8 + (size_t)v0 * kInkHidden, CUDA_R_8I, kInkHidden,
                        ms_h_q8, CUDA_R_8I, kInkHidden, &ibeta,
                        ms_tile_i32, CUDA_R_32I, vt,
                        CUBLAS_COMPUTE_32I, CUBLAS_GEMM_DEFAULT_TENSOR_OP));
                }
                k_dspark_tile_top16_i32<<<dim3(16, R), 256, 0, r.st>>>(
                    ms_tile_i32, m.unembed_screen.sc, ms_h_sc, v0, vt, ti, 16, V, ms_top_tmp_ids, ms_top_tmp_vals);
            }
            k_dspark_top_cand_s2<<<R, 256, 0, r.st>>>(ms_top_tmp_ids, ms_top_tmp_vals, ms_k_cand, ms_cand_ids, ms_cand_base);
        } else {
            for (int b = 0; b < B; ++b) {
                const float* hs7 = ms_cur + ((size_t)b * kInkDsparkBlock + 1) * kInkHidden;
                k_dspark_quant_h_q8<<<7, 256, 0, r.st>>>(hs7, d_h_q8, d_h_sc);
                k_dspark_screen_dp4a7<<<(V + 3) / 4, 128, 0, r.st>>>(
                    m.unembed_screen.q8, m.unembed_screen.sc, d_h_q8, d_h_sc, d_base_logits, V);
                k_dspark_top_cand_s1<<<dim3(128, 7), 256, 0, r.st>>>(d_base_logits, V, d_top_tmp_ids, d_top_tmp_vals, V);
                k_dspark_top_cand_s2<<<7, 256, 0, r.st>>>(
                    d_top_tmp_ids, d_top_tmp_vals, ms_k_cand,
                    ms_cand_ids + (size_t)b * 7 * ms_k_cand,
                    ms_cand_base + (size_t)b * 7 * ms_k_cand);
            }
        }

        // 4. Batched n_dr-Step On-GPU Markov RNN across ALL B slots simultaneously!
        const int cand_stride = (ms_tile_i32 && (V_gemm % 4 == 0)) ? n_dr : 7;
        INK_CK(cudaMemsetAsync(ms_rnn_state, 0, (size_t)B * kInkMarkovRank * 4, r.st));
        INK_CK(cudaMemcpyAsync(ms_prev_toks, ms_anchor_ids, (size_t)B * 4, cudaMemcpyDeviceToDevice, r.st));
        const auto& ti_mw1 = m.tinfo("dspark.markov_head.markov_w1.weight");
        const auto& ti_rpb = m.tinfo("dspark.markov_head.rnn_proj.bias");
        const auto& ti_hob = m.tinfo("dspark.markov_head.hidden_out.bias");

        for (int s = 0; s < n_dr; ++s) {
            k_dspark_markov_prep_slots<<<B, 256, 0, r.st>>>(
                ms_prev_toks, ms_rnn_state, ms_cur, s, ms_f_rnn, ms_slot_ids, ms_apos, ms_frnn_cap,
                m.d_trunk + ti_mw1.off, ti_mw1.fmt, ms_rnn_joint, B);
            dspark_linear_b8("dspark.markov_head.rnn_proj.weight", ms_rnn_joint, ms_rnn_proj, B);
            k_dspark_gru_upd_slots<<<B, 256, 0, r.st>>>(
                ms_rnn_proj, (const float*)(m.d_trunk + ti_rpb.off), ms_rnn_state, ms_rnn_lat, B);
            dspark_linear_b8("dspark.markov_head.hidden_out.weight", ms_rnn_lat, ms_corr, B);
            k_dspark_add_bias_h_slots<<<dim3((kInkHidden + 255) / 256, B), 256, 0, r.st>>>(
                ms_corr, (const float*)(m.d_trunk + ti_hob.off), ms_cur, s, true, B);
            k_dspark_score_cand_dp4a_slots<<<dim3((ms_k_cand + 7) / 8, B), 256, 0, r.st>>>(
                m.unembed_screen.q8, m.unembed_screen.sc, ms_corr, ms_cand_ids, s, cand_stride, ms_k_cand, B, ms_cand_scores);
            k_dspark_top1_reduce_slots<<<B, 256, 0, r.st>>>(
                ms_cand_scores, ms_cand_ids, s, cand_stride, ms_k_cand, B, ms_prev_toks, ms_out_drafts);
        }
        INK_CK(cudaMemcpyAsync(h_ms_drafts, ms_out_drafts, (size_t)B * 7 * 4, cudaMemcpyDeviceToHost, r.st));
        INK_CK(cudaStreamSynchronize(r.st));
        std::memcpy(out_drafts_7, h_ms_drafts, (size_t)B * 7 * 4);
    }

    void free_multislot() {
        const bool had_slab = (ms_slab != nullptr);
        if (ms_slab) { cudaFree(ms_slab); ms_slab = nullptr; }
        auto fr = [&](auto*& p) { if (had_slab) p = nullptr; else if (p) { cudaFree(p); p = nullptr; } };
        if (ms_vram_kv_slab) { cudaFree(ms_vram_kv_slab); ms_vram_kv_slab = nullptr; }
        ms_vram_cap = 0;
        for (int il = 0; il < 3; ++il) {
            if (ms_ctx_k_raw[il])    { cudaFreeHost(ms_ctx_k_raw[il]);    ms_ctx_k_raw[il]    = nullptr; }
            if (ms_ctx_v_raw[il])    { cudaFreeHost(ms_ctx_v_raw[il]);    ms_ctx_v_raw[il]    = nullptr; }
            if (ms_tail_k_raw[il])   { cudaFreeHost(ms_tail_k_raw[il]);   ms_tail_k_raw[il]   = nullptr; }
            if (ms_tail_v_raw[il])   { cudaFreeHost(ms_tail_v_raw[il]);   ms_tail_v_raw[il]   = nullptr; }
            if (ms_tail_ksc_raw[il]) { cudaFreeHost(ms_tail_ksc_raw[il]); ms_tail_ksc_raw[il] = nullptr; }
            if (ms_tail_vsc_raw[il]) { cudaFreeHost(ms_tail_vsc_raw[il]); ms_tail_vsc_raw[il] = nullptr; }
            ms_ctx_k[il] = nullptr; ms_ctx_v[il] = nullptr;
            ms_kv_desc[il] = {}; ms_host_tail[il] = 0;
        }
        fr(ms_f_rnn); fr(ms_cur); fr(ms_h); fr(ms_q); fr(ms_k); fr(ms_v);
        fr(ms_qkv_fused); fr(ms_gu_fused); fr(ms_attn_out); fr(ms_proj); fr(ms_act);
        fr(ms_rnn_state); fr(ms_rnn_joint); fr(ms_rnn_proj); fr(ms_rnn_lat);
        fr(ms_corr); fr(ms_cand_scores); fr(ms_cand_ids); fr(ms_cand_base);
        fr(ms_slot_ids); fr(ms_apos); fr(ms_anchor_ids); fr(ms_pos);
        fr(ms_prev_toks); fr(ms_out_drafts); fr(ms_ing_slots); fr(ms_ing_pos); fr(ms_ing_rows);
        for (int t = 0; t < 4; ++t) fr(ms_taps_comp[t]);
        if (h_ms_drafts)   { cudaFreeHost(h_ms_drafts);   h_ms_drafts   = nullptr; }
        if (h_ms_meta)     { cudaFreeHost(h_ms_meta);     h_ms_meta     = nullptr; }
        if (h_ms_ing_meta) { cudaFreeHost(h_ms_ing_meta); h_ms_ing_meta = nullptr; }
        fr(ms_h_q8); fr(ms_h_sc); fr(ms_tile_i32); fr(ms_top_tmp_ids); fr(ms_top_tmp_vals);
        ms_vt_nom = 0;
        multislot_ready = false;
    }

    // Fast cuBLAS BF16 (sm_80+) or FP16 (sm_70/75) Tensor Core linear for DSpark!
    void dspark_gemm_bf16(const uint16_t* W16, const float* X, float* Y, int B, int N, int K, bool reuse_x = false) {
        const float a = 1.0f, b = 0.0f;
        if (cache_is_fp16) {
            __half* x_fp16 = reinterpret_cast<__half*>(r.d_x_bf16);
            if (!reuse_x) {
                const size_t nx = (size_t)B * K;
                k_ink_f32_to_fp16<<<(nx + 255) / 256, 256, 0, r.st>>>(X, x_fp16, nx);
            }
            INK_CUBLAS_CK(cublasGemmEx(
                r.cublas, CUBLAS_OP_T, CUBLAS_OP_N, N, B, K, &a,
                W16, CUDA_R_16F, K, x_fp16, CUDA_R_16F, K, &b,
                Y, CUDA_R_32F, N, CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP));
        } else {
            if (!reuse_x) {
                const size_t nx = (size_t)B * K;
                k_ink_f32_to_bf16<<<((nx + 3) / 4 + 255) / 256, 256, 0, r.st>>>(X, r.d_x_bf16, nx);
            }
            INK_CUBLAS_CK(cublasGemmEx(
                r.cublas, CUBLAS_OP_T, CUBLAS_OP_N, N, B, K, &a,
                W16, CUDA_R_16BF, K, r.d_x_bf16, CUDA_R_16BF, K, &b,
                Y, CUDA_R_32F, N, CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP));
        }
    }

    void dspark_linear_b8(const std::string& wn, const float* X, float* Y, int B, bool reuse_x = false) {
        if (m.sm_cc >= 70) {
            const auto& ti = m.tinfo(wn);
            const int N = (int)ti.rows(), K = (int)ti.cols();
            auto it = bf16_cache.find(wn);
            const uint16_t* W16 = (it != bf16_cache.end()) ? it->second
                                : ((!cache_is_fp16 && ti.fmt == FMT_BF16) ? reinterpret_cast<const uint16_t*>(m.d_trunk + ti.off) : nullptr);
            if (W16) {
                dspark_gemm_bf16(W16, X, Y, B, N, K, reuse_x);
                return;
            }
        }
        r.linear(wn, X, Y, B, 0, reuse_x);
    }

    // Elastic growth for DSpark Global layer(s) (e.g. layer 1) as sequence length grows!
    bool ensure_ctx_cap(int needed_pos) {
        if (df2) return true;   // DFlash 2's rings are fixed-size sliding windows
        if (!ready || needed_pos <= 0) return true;
        const int target = std::min(max_ctx, std::max(needed_pos, ((needed_pos + 8191) / 8192) * 8192));
        bool grew = false;
        for (int il = 0; il < 3; ++il) {
            if (m.man.c.dspark_is_swa[il] != 0) continue;
            if (target <= ctx_cap[il]) continue;
            if (ctx_is_host[il]) {
                // the COMPACT drafter keeps this layer's context in mapped host memory with a fixed capacity: it cannot grow (the code below would cudaFree a mapped
                // device pointer), so the drafter stops here, as it does when the card is tight
                std::fprintf(stderr, "[InkDSpark] global ctx %d is fixed at %d tokens on mapped host memory, the prompt needs %d; disabling DSpark gracefully\n",
                             il, ctx_cap[il], target);
                ready = false;
                return false;
            }
            INK_CK(cudaStreamSynchronize(r.st));
            const size_t new_elems = (size_t)target * kInkKVHeads * kInkHeadDim;
            const size_t old_bytes = (size_t)ctx_cap[il] * kInkKVHeads * kInkHeadDim * sizeof(__half);
            __half *nk = nullptr, *nv = nullptr;
            if (cudaMalloc(&nk, new_elems * sizeof(__half)) != cudaSuccess ||
                cudaMalloc(&nv, new_elems * sizeof(__half)) != cudaSuccess) {
                (void)cudaGetLastError();
                if (nk) cudaFree(nk);
                if (nv) cudaFree(nv);
                std::fprintf(stderr, "[InkDSpark] Tight VRAM while growing global ctx %d -> %d; disabling DSpark gracefully\n",
                             ctx_cap[il], target);
                ready = false;
                return false;
            }
            INK_CK(cudaMemcpy(nk, d_ctx_k[il], old_bytes, cudaMemcpyDeviceToDevice));
            INK_CK(cudaMemcpy(nv, d_ctx_v[il], old_bytes, cudaMemcpyDeviceToDevice));
            cudaFree(d_ctx_k[il]);
            cudaFree(d_ctx_v[il]);
            d_ctx_k[il] = nk;
            d_ctx_v[il] = nv;
            ctx_cap[il] = target;
            grew = true;
        }
        if (grew) {
            if (draft_graph) { cudaGraphExecDestroy(draft_graph); draft_graph = nullptr; }
            if (beam_graph)  { cudaGraphExecDestroy(beam_graph);  beam_graph = nullptr; beam_graph_w = 0; }
        }
        return true;
    }

    // Ingest target taps [4, M, 2304] at positions [p0 .. p0 + M - 1] into DSpark context KV & f_rnn (ZERO host sync!)
    void ingest_taps_from(float* const taps[4], int p0, int M) {
        if (df2) { if (M > 0) df2->ingest_seq(taps, 0, p0, M); return; }
        if (!ready || M <= 0) return;
        if (!ensure_ctx_cap(p0 + M)) return;
        for (int i = 0; i < M; ++i) h_pos_pin[i] = p0 + i;
        INK_CK(cudaMemcpyAsync(d_pos, h_pos_pin, (size_t)M * 4, cudaMemcpyHostToDevice, r.st));
        const float* fw = (const float*)m.trunk_ptr("dspark.fuse_weight");

        for (int il = 0; il < 3; ++il) {
            const std::string p = "dspark.layers." + std::to_string(il);
            k_dspark_mix_taps<<<M, 256, 0, r.st>>>(
                taps[0], taps[1], taps[2], taps[3], fw, il,
                (const float*)m.trunk_ptr(p + ".dflare_norm.weight"), d_ctx_tmp, M);
            auto it_kv = bf16_cache.find(p + ".kv_ctx_fused");
            if (it_kv != bf16_cache.end()) {
                dspark_gemm_bf16(it_kv->second, d_ctx_tmp, d_kv_ctx_fused, M, 2 * kInkKVDim, kInkHidden, false);
                k_dspark_split_kv<<<dim3((2 * kInkKVDim + 255) / 256, M), 256, 0, r.st>>>(d_kv_ctx_fused, d_kc_tmp, d_vc_tmp, M);
            } else {
                r.linear(p + ".self_attn.k_proj_ctx.weight", d_ctx_tmp, d_kc_tmp, M);
                r.linear(p + ".self_attn.v_proj_ctx.weight", d_ctx_tmp, d_vc_tmp, M);
            }
            dim3 g(kInkKVHeads, M);
            k_dspark_rmsnorm_rope<<<g, 128, 0, r.st>>>(
                d_kc_tmp, (const float*)m.trunk_ptr(p + ".self_attn.k_norm_ctx.weight"),
                d_pos, m.man.c.dspark_rope_theta, M, kInkKVHeads);
            InkKvVisitDesc desc{};
            desc.k_f16 = d_ctx_k[il];
            desc.v_f16 = d_ctx_v[il];
            desc.cap_pos = ctx_cap[il];
            desc.is_swa = (m.man.c.dspark_is_swa[il] != 0);
            desc.kv_fmt = 0;
            k_ink_kv_write<<<g, 128, 0, r.st>>>(d_kc_tmp, d_vc_tmp, nullptr, 0, d_pos, desc, M);
        }
        k_dspark_mix_taps_ring<<<M, 256, 0, r.st>>>(
            taps[0], taps[1], taps[2], taps[3], fw, 3,
            (const float*)m.trunk_ptr("dspark.markov_head.dflare_norm.weight"),
            d_f_rnn, p0, frnn_cap, M);
    }

    void ingest_taps(int p0, int M) {
        ingest_taps_from(r.d_taps, p0, M);
    }

    // Shared 3-layer Block-8 DSpark transformer + Top-K_cand screening (captured in both draft_graph and beam_graph!)
    void emit_dspark_backbone() {
        const int B = kInkDsparkBlock;  // 8
        const auto& ei = m.tinfo("model.llm.embed.weight");
        k_ink_embed_norm<<<1, 256, 0, r.st>>>(
            m.embed_ptr(), ei.fmt,
            (const float*)m.trunk_ptr("model.llm.embed_norm.weight"),
            d_anchor_id, nullptr, nullptr, d_cur, 1, kInkHidden);
        const float* mask_e = (const float*)m.trunk_ptr("dspark.mask_embed");
        k_dspark_fill_mask<<<B - 1, 256, 0, r.st>>>(mask_e, d_cur, B);

        // 2. 3 non-causal block-attention layers (Fused QKV + Fused GU cuBLAS BF16 Tensor Cores -> < 0.85 ms total!)
        for (int il = 0; il < 3; ++il) {
            const std::string p = "dspark.layers." + std::to_string(il);
            k_ink_rms_norm<<<B, 256, 0, r.st>>>(d_cur, (const float*)m.trunk_ptr(p + ".input_layernorm.weight"), d_h, nullptr, B, kInkHidden);
            auto it_qkv = bf16_cache.find(p + ".qkv_fused");
            if (it_qkv != bf16_cache.end()) {
                dspark_gemm_bf16(it_qkv->second, d_h, d_qkv_fused, B, kInkQDim + 2 * kInkKVDim, kInkHidden, false);
                k_dspark_split_qkv<<<dim3((kInkQDim + 2 * kInkKVDim + 255) / 256, B), 256, 0, r.st>>>(d_qkv_fused, d_q, d_k, d_v, B);
            } else if (m.all_sf4_vram) {
                const std::string qkv_names[3] = {
                    p + ".self_attn.q_proj.weight", p + ".self_attn.k_proj.weight", p + ".self_attn.v_proj.weight"
                };
                float* qkv_ys[3] = { d_q, d_k, d_v };
                r.linear_multi(qkv_names, qkv_ys, 3, d_h, B);
            } else {
                dspark_linear_b8(p + ".self_attn.q_proj.weight", d_h, d_q, B, false);
                dspark_linear_b8(p + ".self_attn.k_proj.weight", d_h, d_k, B, true);
                dspark_linear_b8(p + ".self_attn.v_proj.weight", d_h, d_v, B, true);
            }
            k_dspark_rmsnorm_rope<<<dim3(kInkHeads, B), 128, 0, r.st>>>(
                d_q, (const float*)m.trunk_ptr(p + ".self_attn.q_norm.weight"), d_pos, m.man.c.dspark_rope_theta, B, kInkHeads);
            k_dspark_rmsnorm_rope<<<dim3(kInkKVHeads, B), 128, 0, r.st>>>(
                d_k, (const float*)m.trunk_ptr(p + ".self_attn.k_norm.weight"), d_pos, m.man.c.dspark_rope_theta, B, kInkKVHeads);
            if (ctx_is_host[il]) {
                k_dspark_block_attn_batch<<<dim3(kInkKVHeads, 1), 128, 0, r.st>>>(
                    d_q, d_k, d_v, d_ctx_k[il], d_ctx_v[il], nullptr, d_apos,
                    m.man.c.dspark_is_swa[il] != 0, m.man.c.dspark_swa_win, ctx_cap[il], 1, d_attn_out);
            } else {
                // fast only (TC_ATTN): the drafter's global layers read the whole committed context for every block row (4.5 ms a call at 128K, one block per head and row): the
                // tensor-core kernel in block mode (no bias, no sink, every row sees the whole block) reads each K/V tile once for the 8 rows
                bool tc_done = false;
                if (m.man.c.dspark_is_swa[il] == 0 && sm120_cfg(m.primary_gpu).tc_attn && tcattn::force_ref() != 0 && B <= 10 && B <= r.tree_cap) {
                    InkKvVisitDesc cd{};
                    cd.k_f16 = const_cast<__half*>(reinterpret_cast<const __half*>(d_ctx_k[il])); cd.v_f16 = const_cast<__half*>(reinterpret_cast<const __half*>(d_ctx_v[il]));
                    cd.cap_pos = ctx_cap[il]; cd.dev_cap = ctx_cap[il]; cd.host_cap = 0; cd.is_swa = false; cd.kv_fmt = 0;
                    constexpr int kBlk = tcattn::kFlagNoBias | tcattn::kFlagNoSink | tcattn::kFlagBlockTail;
                    tc_done = tcattn::launch(r.st, d_q, nullptr, nullptr, cd, nullptr, 0, d_apos, d_k, d_v, nullptr, nullptr, 0, r.d_attn_pm, r.d_attn_pl, r.d_attn_po, B, B, kBlk);
                    if (tc_done) tcattn::merge(r.st, r.d_attn_pm, r.d_attn_pl, r.d_attn_po, nullptr, d_attn_out, nullptr, d_apos, true, ctx_cap[il], B, B, kBlk);
                }
                if (!tc_done)
                k_dspark_block_attn<<<dim3(kInkHeads, B), 128, 0, r.st>>>(
                    d_q, d_k, d_v, d_ctx_k[il], d_ctx_v[il], d_apos,
                    m.man.c.dspark_is_swa[il] != 0, m.man.c.dspark_swa_win, ctx_cap[il], B, d_attn_out);
            }
            dspark_linear_b8(p + ".self_attn.o_proj.weight", d_attn_out, d_proj, B, false);
            k_ink_add<<<(B * kInkHidden + 255) / 256, 256, 0, r.st>>>(d_cur, d_cur, d_proj, B * kInkHidden);

            k_ink_rms_norm<<<B, 256, 0, r.st>>>(d_cur, (const float*)m.trunk_ptr(p + ".post_attention_layernorm.weight"), d_h, nullptr, B, kInkHidden);
            auto it_gu = bf16_cache.find(p + ".gu_fused");
            if (it_gu != bf16_cache.end()) {
                dspark_gemm_bf16(it_gu->second, d_h, d_gu_fused, B, 2 * kInkDsparkInter, kInkHidden, false);
                k_dspark_swiglu_fused<<<dim3((kInkDsparkInter + 255) / 256, B), 256, 0, r.st>>>(d_gu_fused, d_act, B);
            } else if (m.all_sf4_vram) {
                const std::string gu_names[2] = {
                    p + ".mlp.gate_proj.weight", p + ".mlp.up_proj.weight"
                };
                float* gu_ys[2] = { d_g, d_u };
                r.linear_multi(gu_names, gu_ys, 2, d_h, B);
                k_dspark_swiglu<<<(B * kInkDsparkInter + 255) / 256, 256, 0, r.st>>>(d_g, d_u, d_act, B * kInkDsparkInter);
            } else {
                dspark_linear_b8(p + ".mlp.gate_proj.weight", d_h, d_g, B, false);
                dspark_linear_b8(p + ".mlp.up_proj.weight", d_h, d_u, B, true);
                k_dspark_swiglu<<<(B * kInkDsparkInter + 255) / 256, 256, 0, r.st>>>(d_g, d_u, d_act, B * kInkDsparkInter);
            }
            dspark_linear_b8(p + ".mlp.down_proj.weight", d_act, d_proj, B, false);
            k_ink_add<<<(B * kInkHidden + 255) / 256, 256, 0, r.st>>>(d_cur, d_cur, d_proj, B * kInkHidden);
        }

        // 3. Final norm + ultra-fast dp4a INT8 base logits (~0.30 ms!) + 896-CTA 2-stage Top-512 (~0.03 ms!)
        k_ink_rms_norm<<<B, 256, 0, r.st>>>(d_cur, (const float*)m.trunk_ptr("dspark.norm.weight"), d_cur, nullptr, B, kInkHidden);
        const int V = m.man.c.vocab_unpadded;
        const float* base_logits_7 = d_base_logits;
        int base_ld = V;
        if (d_unembed_fp16) {
            const int V_pad = (V + 7) & ~7;
            const size_t nx = 8ull * kInkHidden;
            __half* x_fp16 = reinterpret_cast<__half*>(r.d_x_bf16);
            k_ink_f32_to_fp16<<<(nx + 255) / 256, 256, 0, r.st>>>(d_cur, x_fp16, nx);
            const float a = 1.0f, b = 0.0f;
            INK_CUBLAS_CK(cublasGemmEx(
                r.cublas, CUBLAS_OP_T, CUBLAS_OP_N, V_pad, 8, kInkHidden, &a,
                d_unembed_fp16, CUDA_R_16F, kInkHidden, x_fp16, CUDA_R_16F, kInkHidden, &b,
                d_base_logits, CUDA_R_32F, V_pad, CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP));
            base_logits_7 = d_base_logits + V_pad;
            base_ld = V_pad;
        } else if (m.unembed_screen.ready) {
            k_dspark_quant_h_q8<<<7, 256, 0, r.st>>>(d_cur + kInkHidden, d_h_q8, d_h_sc);
            if (m.sm_cc == 70 || m.sm_cc == 60) {
                k_dspark_screen_h2_7<<<(V + 3) / 4, 128, 0, r.st>>>(
                    m.unembed_screen.q8, m.unembed_screen.sc, d_h_q8, d_h_sc, d_base_logits, V);
            } else {
                k_dspark_screen_dp4a7<<<(V + 3) / 4, 128, 0, r.st>>>(
                    m.unembed_screen.q8, m.unembed_screen.sc, d_h_q8, d_h_sc, d_base_logits, V);
            }
        } else {
            r.linear("model.llm.unembed.weight", d_cur + kInkHidden, d_base_logits, 7, V);
        }
        k_dspark_top_cand_s1<<<dim3(128, 7), 256, 0, r.st>>>(base_logits_7, V, d_top_tmp_ids, d_top_tmp_vals, base_ld);
        k_dspark_top_cand_s2<<<7, 256, 0, r.st>>>(d_top_tmp_ids, d_top_tmp_vals, K_cand, d_cand_ids, d_cand_base);
    }

    // Emit full Block-8 draft graph body (100% CUDA-Graph capturable -> < 0.45 ms on RTX 5090!)
    void emit_draft_block8() {
        emit_dspark_backbone();

        // 4. Multi-CTA 7-step on-GPU Markov RNN loop (with pre-packed BF16 weights + dp4a_i8 candidate scoring!)
        INK_CK(cudaMemsetAsync(d_rnn_state, 0, kInkMarkovRank * 4, r.st));
        const auto& ti_mw1 = m.tinfo("dspark.markov_head.markov_w1.weight");
        const auto& ti_rpb = m.tinfo("dspark.markov_head.rnn_proj.bias");
        const auto& ti_hob = m.tinfo("dspark.markov_head.hidden_out.bias");
        const auto& ti_un  = m.tinfo("model.llm.unembed.weight");
        const bool use_dp4a_cand = (m.unembed_screen.ready && (m.sm_cc == 61 || m.sm_cc >= 75));

        const int32_t* prev_ptr = d_anchor_id;
        for (int s = 0; s < 7; ++s) {
            const float* hs = d_cur + (size_t)(s + 1) * kInkHidden;
            k_dspark_markov_prep<<<1, 256, 0, r.st>>>(
                prev_ptr, d_rnn_state, hs, d_f_rnn, d_apos, frnn_cap,
                m.d_trunk + ti_mw1.off, ti_mw1.fmt, d_rnn_joint);
            dspark_linear_b8("dspark.markov_head.rnn_proj.weight", d_rnn_joint, d_rnn_proj, 1);
            k_dspark_markov_gru_upd<<<1, 256, 0, r.st>>>(
                d_rnn_proj, (const float*)(m.d_trunk + ti_rpb.off), d_rnn_state, d_rnn_lat);
            dspark_linear_b8("dspark.markov_head.hidden_out.weight", d_rnn_lat, d_corr, 1);
            k_dspark_add_bias_and_h<<<(kInkHidden + 255) / 256, 256, 0, r.st>>>(
                d_corr, (const float*)(m.d_trunk + ti_hob.off), hs, m.unembed_screen.ready);
            if (use_dp4a_cand) {
                k_dspark_score_cand_dp4a_batch<<<(K_cand + 7) / 8, 256, 0, r.st>>>(
                    m.unembed_screen.q8, m.unembed_screen.sc, d_corr,
                    d_cand_ids + (size_t)s * K_cand, nullptr, K_cand, 1, d_cand_scores);
            } else {
                k_dspark_score_cand_eval<<<(K_cand + 7) / 8, 256, 0, r.st>>>(
                    m.d_trunk + ti_un.off, ti_un.fmt, (int)ti_un.rows(),
                    d_corr, d_cand_ids + (size_t)s * K_cand,
                    m.unembed_screen.ready ? nullptr : (d_cand_base + (size_t)s * K_cand),
                    K_cand, d_cand_scores);
            }
            k_dspark_top2_reduce<<<1, 256, 0, r.st>>>(
                d_cand_scores, d_cand_ids + (size_t)s * K_cand, K_cand,
                d_top2_ids + 2 * s, d_top2_probs + 2 * s);
            prev_ptr = d_top2_ids + 2 * s;
        }
    }

    // Emit True W=2..4 Multi-Branch On-GPU Markov Beam Tree Graph (captures ALL 7 beam steps in ONE CUDA Graph!)
    void emit_draft_beam_tree(int beam_w) {
        emit_dspark_backbone();

        const auto& ti_mw1 = m.tinfo("dspark.markov_head.markov_w1.weight");
        const auto& ti_rpb = m.tinfo("dspark.markov_head.rnn_proj.bias");
        const auto& ti_hob = m.tinfo("dspark.markov_head.hidden_out.bias");
        const auto& ti_un  = m.tinfo("model.llm.unembed.weight");
        const bool use_dp4a_cand = (m.unembed_screen.ready && (m.sm_cc == 61 || m.sm_cc >= 75));

        k_dspark_beam_init<<<1, 256, 0, r.st>>>(
            d_anchor_id, d_beam_state, d_beam_prev_tok, d_beam_logp, d_beam_node,
            d_pool_tok, d_pool_parent, d_pool_depth, d_pool_logp, d_pool_cnt);

        for (int s = 0; s < 7; ++s) {
            const int nw = (s == 0) ? 1 : beam_w;
            const float* hs = d_cur + (size_t)(s + 1) * kInkHidden;
            k_dspark_markov_prep_batch<<<nw, 256, 0, r.st>>>(
                d_beam_prev_tok, d_beam_state, hs, d_f_rnn, d_apos, frnn_cap,
                m.d_trunk + ti_mw1.off, ti_mw1.fmt, d_rnn_joint_b, nw);
            dspark_linear_b8("dspark.markov_head.rnn_proj.weight", d_rnn_joint_b, d_rnn_proj_b, nw);
            k_dspark_markov_gru_upd_batch<<<nw, 256, 0, r.st>>>(
                d_rnn_proj_b, (const float*)(m.d_trunk + ti_rpb.off),
                d_beam_state, d_beam_state_next, d_rnn_lat_b, nw);
            dspark_linear_b8("dspark.markov_head.hidden_out.weight", d_rnn_lat_b, d_corr_b, nw);
            k_dspark_add_bias_and_h_batch<<<dim3((kInkHidden + 255) / 256, nw), 256, 0, r.st>>>(
                d_corr_b, (const float*)(m.d_trunk + ti_hob.off), hs, m.unembed_screen.ready, nw);
            if (use_dp4a_cand) {
                k_dspark_score_cand_dp4a_batch<<<(K_cand + 7) / 8, 256, 0, r.st>>>(
                    m.unembed_screen.q8, m.unembed_screen.sc, d_corr_b,
                    d_cand_ids + (size_t)s * K_cand, nullptr, K_cand, nw, d_cand_scores_b);
            } else {
                for (int b = 0; b < nw; ++b) {
                    k_dspark_score_cand_eval<<<(K_cand + 7) / 8, 256, 0, r.st>>>(
                        m.d_trunk + ti_un.off, ti_un.fmt, (int)ti_un.rows(),
                        d_corr_b + (size_t)b * kInkHidden, d_cand_ids + (size_t)s * K_cand,
                        m.unembed_screen.ready ? nullptr : (d_cand_base + (size_t)s * K_cand),
                        K_cand, d_cand_scores_b + (size_t)b * K_cand);
                }
            }
            k_dspark_beam_expand<<<1, 256, 0, r.st>>>(
                d_cand_scores_b, d_cand_ids + (size_t)s * K_cand, d_beam_state_next,
                K_cand, nw, beam_w, s + 1,
                d_beam_state, d_beam_prev_tok, d_beam_logp, d_beam_node,
                d_pool_tok, d_pool_parent, d_pool_depth, d_pool_logp, d_pool_cnt);
        }
    }

    // Draft 7 tokens (Block-8) at custom RoPE position pos0 while attending committed ctx [0..ctx_apos-1]!
    void draft_block8_at(int32_t anchor_tok, int ctx_apos, int pos0, int32_t* out_ids_7, float* out_conf_7 = nullptr) {
        if (df2) {
            // a block that starts past the committed context (the 2-block chain) attends the context and its own rows; the gap in between is masked
            float conf[7];
            const int32_t slot0 = 0;
            const bool ok = df2->draft(1, &slot0, &anchor_tok, &pos0, out_ids_7, conf, &ctx_apos);
            for (int s = 0; s < 7; ++s) {
                if (!ok) out_ids_7[s] = anchor_tok;
                if (out_conf_7) out_conf_7[s] = ok ? conf[s] : 0.0f;
            }
            return;
        }
        const int B = kInkDsparkBlock;  // 8
        h_meta_pin[0] = anchor_tok;
        h_meta_pin[1] = ctx_apos;
        for (int i = 0; i < B; ++i) h_meta_pin[2 + i] = pos0 + i;
        INK_CK(cudaMemcpyAsync(d_anchor_id, h_meta_pin + 0, 4, cudaMemcpyHostToDevice, r.st));
        INK_CK(cudaMemcpyAsync(d_apos,      h_meta_pin + 1, 4, cudaMemcpyHostToDevice, r.st));
        INK_CK(cudaMemcpyAsync(d_pos,       h_meta_pin + 2, B * 4, cudaMemcpyHostToDevice, r.st));

        if (r.use_cuda_graph) {
            if (!draft_graph) {
                cudaGraph_t gr = nullptr;
                cudaError_t ce = cudaStreamBeginCapture(r.st, cudaStreamCaptureModeThreadLocal);
                if (ce == cudaSuccess) {
                    emit_draft_block8();
                    ce = cudaStreamEndCapture(r.st, &gr);
                    if (ce == cudaSuccess && gr) {
                        ce = cudaGraphInstantiate(&draft_graph, gr, nullptr, nullptr, 0);
                        cudaGraphDestroy(gr);
                    }
                }
                if (ce != cudaSuccess || !draft_graph) {
                    (void)cudaGetLastError();
                    if (draft_graph) { cudaGraphExecDestroy(draft_graph); draft_graph = nullptr; }
                    r.use_cuda_graph = false;
                    emit_draft_block8();
                } else {
                    INK_CK(cudaGraphLaunch(draft_graph, r.st));
                }
            } else {
                INK_CK(cudaGraphLaunch(draft_graph, r.st));
            }
        } else {
            emit_draft_block8();
        }
        INK_CK(cudaMemcpyAsync(h_top2_ids, d_top2_ids, 14 * 4, cudaMemcpyDeviceToHost, r.st));
        INK_CK(cudaMemcpyAsync(h_top2_probs, d_top2_probs, 14 * 4, cudaMemcpyDeviceToHost, r.st));
        INK_CK(cudaStreamSynchronize(r.st));
        for (int s = 0; s < 7; ++s) {
            out_ids_7[s] = h_top2_ids[2 * s];
            if (out_conf_7) out_conf_7[s] = h_top2_probs[2 * s];
        }
    }

    // Draft 7 tokens (Block-8) + Top-2 candidates per slot on GPU (CUDA Graph captured!)
    void draft_block8(int32_t anchor_tok, int apos, int32_t* out_ids_7, float* out_conf_7 = nullptr) {
        draft_block8_at(anchor_tok, apos, apos, out_ids_7, out_conf_7);
    }

    // DFlash 2 drafts one linear chain, so its "tree" is a path of the anchor and up to 7 drafts. The node count follows the same expected-tokens-per-cost rule
    // as the beam tree below (adaptive_moe_ratio > 0), with the chain's cumulative confidences as the prefix probabilities.
    // Cost of one more verify row, in plain steps, at context position apos: 0 where the K/V is in VRAM (a 24 GB card or more), else about 2.2e-5 per context token
    // (RTX 5060 Ti: 32K -> 0.7, 128K -> 1.3 measured). A step at or above 0.5 (about 25K tokens) is a plain step.
    double df2_ctx_cost(int apos) const {
        return (df2 && df2->host_codebooks) ? std::min(3.0, 2.2e-5 * (double)std::max(0, apos - 2048)) : 0.0;
    }
    bool df2_plain_ctx(int apos) const { return df2_ctx_cost(apos) >= 0.5; }

    void draft_chain_tree_df2(int32_t anchor_tok, int apos, int max_nodes, InkTreeSpec& out_tree, float adaptive_moe_ratio, float plain_margin) {
        max_nodes = std::max(2, std::min(r.tree_cap, max_nodes));
        if (df2_plain_ctx(apos)) {   // a plain step: no draft, no window (the ring keeps being fed by the ingest)
            out_tree = InkTreeSpec{};
            out_tree.M = 1;
            out_tree.tok[0] = anchor_tok;
            out_tree.parent[0] = -1;
            out_tree.depth[0] = 0;
            out_tree.build();
            last_pred_etok = 1.0;
            last_m = 1;
            last_deepest_node = 0;
            last_deepest_logp = -1e30f;
            return;
        }
        int32_t ids[7] = {};
        float conf[7] = {};
        const int32_t slot0 = 0;
        if (!df2->draft(1, &slot0, &anchor_tok, &apos, ids, conf))
            for (int s = 0; s < 7; ++s) { ids[s] = anchor_tok; conf[s] = 0.0f; }
        double pref_p[9];
        pref_p[0] = 1.0;
        double cum = 1.0;
        for (int i = 0; i < 7; ++i) {
            conf[i] = (float)std::pow(std::max(0.0, std::min(1.0, (double)conf[i])), df2->conf_pow);
            cum *= std::max(0.0, std::min(1.0, (double)conf[i]));
            pref_p[i + 1] = pref_p[i] + cum;
        }
        int want = std::min(max_nodes, 8);
        // On a card whose K/V lives (partly) in host memory every verify row reads the whole context again: a window row costs more the longer the context. Measured on the
        // RTX 5060 Ti (32K: a step at M = 8 took 3.3x a plain step, 128K: 8.6x): about 1.3e-5 of a plain step per row and context token, so wide windows are not worth it there.
        const double ctx_row_cost = df2_ctx_cost(apos);
        if (adaptive_moe_ratio > 0.0f || ctx_row_cost > 0.0) {
            double best_rate = -1.0;
            int best_m = std::min(rtx5090::dspark_opt_on() ? 4 : 3, 8);
            const double ratio = std::max(0.0f, adaptive_moe_ratio);
            for (int mb = 1; mb <= want; ++mb) {
                if (mb < 3 && plain_margin <= 0.0f && ctx_row_cost <= 0.1) continue;
                const double e_tok = pref_p[mb - 1];
                const double exp_frac = 1.0 - std::pow(15.0 / 16.0, (double)mb);
                double rate = e_tok / (1.0 + ratio * exp_frac + ctx_row_cost * (double)(mb - 1));
                if (mb < 3) rate *= (plain_margin > 0.0f ? (double)plain_margin : 1.0);
                if (rate > best_rate) { best_rate = rate; best_m = mb; }
            }
            want = best_m;
            last_pred_etok = pref_p[best_m - 1];
            last_m = best_m;
        }
        out_tree = InkTreeSpec{};
        for (int i = 0; i < want; ++i) {
            const int nid = out_tree.M++;
            out_tree.tok[nid] = (i == 0) ? anchor_tok : ids[i - 1];
            out_tree.parent[nid] = (i == 0) ? -1 : (int8_t)(i - 1);
            out_tree.depth[nid] = (uint8_t)i;
        }
        last_deepest_node = want - 1;
        // the log-probability of the chain to its end: the 2-block chain (fast mode, depth 7) asks for exp(it) >= 0.55
        last_deepest_logp = 0.0f;
        for (int i = 0; i + 1 < want; ++i) last_deepest_logp += std::log(std::max(1e-9f, conf[i]));
        out_tree.build();
    }

    // Draft True W=2..4 Multi-Branch Bayes-Optimal Connected Tree (M <= 16..32 nodes, CUDA Graph captured!)
    // When adaptive_moe_ratio > 0, dynamically picks M* in {4, 6, 8, 10, 12, 16} maximizing E[emit(M)] / (1 + ratio*(1-(15/16)^M))!
    void draft_beam_tree(int32_t anchor_tok, int apos, int beam_w, int max_nodes, InkTreeSpec& out_tree, float adaptive_moe_ratio = 0.0f, float plain_margin = 0.0f) {
        if (df2) { draft_chain_tree_df2(anchor_tok, apos, max_nodes, out_tree, adaptive_moe_ratio, plain_margin); return; }
        beam_w = std::max(1, std::min(kInkMaxBeam, beam_w));
        max_nodes = std::max(2, std::min(r.tree_cap, max_nodes));
        const int B = kInkDsparkBlock;  // 8
        h_meta_pin[0] = anchor_tok;
        h_meta_pin[1] = apos;
        for (int i = 0; i < B; ++i) h_meta_pin[2 + i] = apos + i;
        INK_CK(cudaMemcpyAsync(d_anchor_id, h_meta_pin + 0, 4, cudaMemcpyHostToDevice, r.st));
        INK_CK(cudaMemcpyAsync(d_apos,      h_meta_pin + 1, 4, cudaMemcpyHostToDevice, r.st));
        INK_CK(cudaMemcpyAsync(d_pos,       h_meta_pin + 2, B * 4, cudaMemcpyHostToDevice, r.st));

        if (r.use_cuda_graph) {
            if (!beam_graph || beam_graph_w != beam_w) {
                if (beam_graph) { cudaGraphExecDestroy(beam_graph); beam_graph = nullptr; }
                cudaGraph_t gr = nullptr;
                cudaError_t ce = cudaStreamBeginCapture(r.st, cudaStreamCaptureModeThreadLocal);
                if (ce == cudaSuccess) {
                    emit_draft_beam_tree(beam_w);
                    ce = cudaStreamEndCapture(r.st, &gr);
                    if (ce == cudaSuccess && gr) {
                        ce = cudaGraphInstantiate(&beam_graph, gr, nullptr, nullptr, 0);
                        cudaGraphDestroy(gr);
                    }
                }
                if (ce != cudaSuccess || !beam_graph) {
                    (void)cudaGetLastError();
                    if (beam_graph) { cudaGraphExecDestroy(beam_graph); beam_graph = nullptr; }
                    r.use_cuda_graph = false;
                    emit_draft_beam_tree(beam_w);
                } else {
                    beam_graph_w = beam_w;
                    cudaError_t le = cudaGraphLaunch(beam_graph, r.st);
                    if (le != cudaSuccess) {
                        cudaGetLastError();
                        cudaGraphExecDestroy(beam_graph); beam_graph = nullptr; beam_graph_w = 0;
                        emit_draft_beam_tree(beam_w);
                    }
                }
            } else {
                cudaError_t le = cudaGraphLaunch(beam_graph, r.st);
                if (le != cudaSuccess) {
                    cudaGetLastError();
                    cudaGraphExecDestroy(beam_graph); beam_graph = nullptr; beam_graph_w = 0;
                    emit_draft_beam_tree(beam_w);
                }
            }
        } else {
            emit_draft_beam_tree(beam_w);
        }
        INK_CK(cudaMemcpyAsync(h_pool_cnt,    d_pool_cnt,    4, cudaMemcpyDeviceToHost, r.st));
        INK_CK(cudaMemcpyAsync(h_pool_tok,    d_pool_tok,    kInkBeamPoolCap * 4, cudaMemcpyDeviceToHost, r.st));
        INK_CK(cudaMemcpyAsync(h_pool_parent, d_pool_parent, kInkBeamPoolCap, cudaMemcpyDeviceToHost, r.st));
        INK_CK(cudaMemcpyAsync(h_pool_depth,  d_pool_depth,  kInkBeamPoolCap, cudaMemcpyDeviceToHost, r.st));
        INK_CK(cudaMemcpyAsync(h_pool_logp,   d_pool_logp,   kInkBeamPoolCap * 4, cudaMemcpyDeviceToHost, r.st));
        INK_CK(cudaStreamSynchronize(r.st));

        const int cnt = std::min(kInkBeamPoolCap, std::max(1, *h_pool_cnt));
        int ord[kInkBeamPoolCap];
        int n_cand = cnt - 1;
        for (int i = 0; i < n_cand; ++i) ord[i] = i + 1;
        std::sort(ord, ord + n_cand, [&](int a, int b) {
            if (h_pool_logp[a] != h_pool_logp[b]) return h_pool_logp[a] > h_pool_logp[b];
            if (h_pool_depth[a] != h_pool_depth[b]) return h_pool_depth[a] < h_pool_depth[b];
            return a < b;
        });

        int want = std::min(max_nodes, cnt);
        if (adaptive_moe_ratio > 0.0f) {
            // Exact Sparse-MoE Bayes-Optimal Dynamic Tree Sizing across cached CUDA Graph buckets!
            // Three lists: the shipped one (switches off), the RTX 5090 one and the RTX 5060 Ti one (shipped + 20, 24).
            static constexpr int kBucketsBase[]  = {1, 2, 3, 4, 5, 6, 8, 10, 12, 16};
            static constexpr int kBucketsRtx[]   = {1, 2, 3, 4, 6, 8, 10, 12, 16, 20, 24};
            static constexpr int kBucketsSmall[] = {1, 2, 3, 4, 5, 6, 8, 10, 12, 16, 20, 24};
            const bool bk_rtx = rtx5090::dspark_opt_on();
            const bool bk_small = !bk_rtx && ink_rtx5060ti_cfg(m.primary_gpu).enabled;
            const int* buckets = bk_rtx ? kBucketsRtx : (bk_small ? kBucketsSmall : kBucketsBase);
            const int n_bkt = bk_rtx ? 11 : (bk_small ? 12 : 10);
            double pref_p[kInkBeamPoolCap + 1];
            pref_p[0] = 1.0;  // root always emits 1 token
            for (int i = 0; i < n_cand; ++i) {
                const float lp = h_pool_logp[ord[i]];
                pref_p[i + 1] = pref_p[i] + ((lp > -20.0f) ? std::exp((double)lp) : 0.0);
            }
            double best_rate = -1.0;
            int best_m = std::min(rtx5090::dspark_opt_on() ? 4 : 3, cnt);
            for (int bi = 0; bi < n_bkt; ++bi) {
                const int mb = buckets[bi];
                if (mb > max_nodes || mb > cnt) continue;
                if (mb < 3 && plain_margin <= 0.0f) continue;  // the 1 and 2 node trees exist only when the caller lets the step fall back to plain decoding
                const double e_tok = pref_p[mb - 1];
                const double exp_frac = 1.0 - std::pow(15.0 / 16.0, (double)mb);
                const double R_eff = adapt.on ? adapt.ratio((double)adaptive_moe_ratio) : (double)adaptive_moe_ratio;
                const double cost = 1.0 + R_eff * exp_frac;
                const double e_adj = adapt.on ? 1.0 + adapt.accept_scale() * (e_tok - 1.0) : e_tok;
                double rate = e_adj / cost;
                if (mb < 3) rate *= (double)plain_margin;
                if (rate > best_rate) { best_rate = rate; best_m = mb; }
            }
            want = best_m;
            last_pred_etok = pref_p[best_m - 1];
            last_m = best_m;
        }

        uint8_t keep[kInkBeamPoolCap] = {};
        keep[0] = 1;
        int n_kept = 1;
        const bool prune_lowp = m.all_sf4_vram;
        for (int i = 0; i < n_cand && n_kept < want; ++i) {
            const int u = ord[i];
            if (h_pool_logp[u] <= -1e20f) break;
            if (!keep[u]) {
                const int par = (int)h_pool_parent[u];
                if (par >= 0 && keep[par]) {
                    if (prune_lowp && (h_pool_logp[u] - h_pool_logp[par]) < -1.14f) continue;  // step prob < 0.32 -> skip
                    keep[u] = 1;
                    ++n_kept;
                }
            }
        }

        out_tree = InkTreeSpec{};
        int8_t remap[kInkBeamPoolCap];
        std::memset(remap, -1, sizeof(remap));
        last_deepest_node = 0;
        last_deepest_logp = -1e30f;
        int best_dep = 0;
        for (int p = 0; p < cnt; ++p) {
            if (!keep[p]) continue;
            const int nid = out_tree.M++;
            remap[p] = (int8_t)nid;
            out_tree.tok[nid] = h_pool_tok[p];
            out_tree.parent[nid] = (p == 0) ? -1 : remap[(int)h_pool_parent[p]];
            out_tree.depth[nid] = h_pool_depth[p];
            if ((int)h_pool_depth[p] > best_dep ||
                ((int)h_pool_depth[p] == best_dep && h_pool_logp[p] > last_deepest_logp)) {
                best_dep = (int)h_pool_depth[p];
                last_deepest_node = nid;
                last_deepest_logp = h_pool_logp[p];
            }
        }
        out_tree.build();
    }

    void free_all() {
        if (df2) { df2->free_all(); delete df2; df2 = nullptr; }
        free_multislot();
        if (draft_graph) { cudaGraphExecDestroy(draft_graph); draft_graph = nullptr; }
        if (beam_graph)  { cudaGraphExecDestroy(beam_graph);  beam_graph = nullptr; beam_graph_w = 0; }
        bf16_cache.clear();
        if (d_bf16_pool) { cudaFree(d_bf16_pool); d_bf16_pool = nullptr; }
        if (d_unembed_fp16) { cudaFree(d_unembed_fp16); d_unembed_fp16 = nullptr; }
        for (int il = 0; il < 3; ++il) {
            if (ctx_is_host[il]) {
                if (h_ctx_k_raw[il]) { cudaFreeHost(h_ctx_k_raw[il]); h_ctx_k_raw[il] = nullptr; }
                if (h_ctx_v_raw[il]) { cudaFreeHost(h_ctx_v_raw[il]); h_ctx_v_raw[il] = nullptr; }
                d_ctx_k[il] = nullptr; d_ctx_v[il] = nullptr;
                ctx_is_host[il] = false;
            } else {
                if (d_ctx_k[il]) { cudaFree(d_ctx_k[il]); d_ctx_k[il] = nullptr; }
                if (d_ctx_v[il]) { cudaFree(d_ctx_v[il]); d_ctx_v[il] = nullptr; }
            }
        }
        auto fr = [](auto*& p) { if (p) { cudaFree(p); p = nullptr; } };
        fr(d_f_rnn); fr(d_cur); fr(d_h); fr(d_q); fr(d_k); fr(d_v); fr(d_qkv_fused); fr(d_gu_fused); fr(d_attn_out);
        fr(d_proj); fr(d_g); fr(d_u); fr(d_act); fr(d_ctx_tmp); fr(d_kc_tmp); fr(d_vc_tmp); fr(d_kv_ctx_fused);
        fr(d_base_logits); fr(d_cand_ids); fr(d_cand_base); fr(d_top_tmp_ids); fr(d_top_tmp_vals); fr(d_h_q8); fr(d_h_sc); fr(d_cand_scores);
        fr(d_rnn_state); fr(d_rnn_joint); fr(d_rnn_proj); fr(d_rnn_lat);
        fr(d_corr); fr(d_pos); fr(d_apos); fr(d_anchor_id); fr(d_draft_ids);
        fr(d_top2_ids); fr(d_top2_probs);
        fr(d_beam_state); fr(d_beam_state_next); fr(d_beam_prev_tok); fr(d_beam_logp); fr(d_beam_node);
        fr(d_rnn_joint_b); fr(d_rnn_proj_b); fr(d_rnn_lat_b); fr(d_corr_b); fr(d_cand_scores_b);
        fr(d_pool_tok); fr(d_pool_parent); fr(d_pool_depth); fr(d_pool_logp); fr(d_pool_cnt);
        if (h_top2_ids) { cudaFreeHost(h_top2_ids); h_top2_ids = nullptr; }
        if (h_top2_probs) { cudaFreeHost(h_top2_probs); h_top2_probs = nullptr; }
        if (h_meta_pin) { cudaFreeHost(h_meta_pin); h_meta_pin = nullptr; }
        if (h_pos_pin) { cudaFreeHost(h_pos_pin); h_pos_pin = nullptr; }
        if (h_pool_tok) { cudaFreeHost(h_pool_tok); h_pool_tok = nullptr; }
        if (h_pool_parent) { cudaFreeHost(h_pool_parent); h_pool_parent = nullptr; }
        if (h_pool_depth) { cudaFreeHost(h_pool_depth); h_pool_depth = nullptr; }
        if (h_pool_logp) { cudaFreeHost(h_pool_logp); h_pool_logp = nullptr; }
        if (h_pool_cnt) { cudaFreeHost(h_pool_cnt); h_pool_cnt = nullptr; }
        ready = false;
    }
};

}  // namespace strata::ink
