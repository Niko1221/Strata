// engine/src/inkling2/cuda/ink_rtx5090_bspec_kernels.cuh - Low-level Mode-3 Batched Speculative & Fast Q8_ROW Argmax Kernels
// MIT License - Copyright (c) 2026 Niko1221 and the Strata contributors
#pragma once

#include <cstdint>
#include "ink_attn.cuh"
#include "ink_cuda.cuh"
#include "ink_kernels.cuh"
#include "ink_rtx5090_cfg.cuh"

namespace strata::ink::rtx5090 {

constexpr int kBspecMaxSlots = 64;
constexpr int kBspecMaxK     = 4;
constexpr int kBspecMaxM     = kBspecMaxSlots * kBspecMaxK; // 256
constexpr int kBspecKcand    = 128;
constexpr int kBspecV8       = 200056; // multiple of 8 for INT8 TC cublasGemmEx

// Fused Post-QKV for Mode 3: K/V ShortConv4 (intra-slot linear lag) + Q/K/V/R Clip + V RMSNorm + Global Tau + Stage (non-mutating!)
__global__ void __launch_bounds__(128) k_ink_post_qkv_fused_bspec(
    float* __restrict__ q,
    const float* __restrict__ q_scale, int q_scale_elems,
    const float* __restrict__ k_raw,
    const float* __restrict__ v_raw,
    const float* __restrict__ sw_k,
    const float* __restrict__ sw_v,
    const float* __restrict__ k_dyn,
    const float* __restrict__ v_dyn,
    const float* __restrict__ hist_k_pool,
    const float* __restrict__ hist_v_pool,
    float* __restrict__ k_out,
    float* __restrict__ v_out,
    float* __restrict__ r,
    const float* __restrict__ r_scale, int r_scale_elems,
    const int32_t* __restrict__ slots,
    const int32_t* __restrict__ pos_arr,
    __half* __restrict__ stage_k,
    __half* __restrict__ stage_v,
    float* __restrict__ stage_sconv,
    bool is_global,
    int K_slot,
    int M) {
    const int m = blockIdx.y;
    const int blk = blockIdx.x;
    const int tid = threadIdx.x;
    if (m >= M) return;

    float tau = 1.0f;
    if (is_global) {
        const float ratio = (float)(pos_arr[m] + 1) * (1.0f / (float)kInkLogFloor);
        tau = 1.0f + kInkLogAlpha * logf(fmaxf(1.0f, ratio));
    }

    __shared__ float sh[8];
    if (blk < 24) {
        const int h = blk;
        float* qp = q + ((size_t)m * kInkHeads + h) * kInkHeadDim;
        float val = qp[tid];
        if (q_scale) {
            const int si = (q_scale_elems == kInkHeadDim) ? tid : (h * kInkHeadDim + tid);
            val *= q_scale[si];
        }
        float ss = block_sum(val * val, sh);
        const float factor = fminf(1.0f, kInkQkClip / (sqrtf(ss) + 1e-6f)) * tau;
        qp[tid] = val * factor;
    } else if (blk < 40) {
        const bool is_k = (blk < 32);
        const int kh = is_k ? (blk - 24) : (blk - 32);
        const int c  = kh * kInkHeadDim + tid;
        const int sl = slots[m];
        const int k  = m % K_slot;
        const float* raw = is_k ? k_raw : v_raw;
        const float* swp = is_k ? sw_k : sw_v;
        const float* dyn = is_k ? k_dyn : v_dyn;
        const float* hist = (is_k ? hist_k_pool : hist_v_pool) + (size_t)sl * 3 * kInkKVDim;

        const float cur = raw[(size_t)m * kInkKVDim + c];
        auto fetch = [&](int lag) -> float {
            if (k >= lag) return raw[(size_t)(m - lag) * kInkKVDim + c];
            return hist[(size_t)(3 + k - lag) * kInkKVDim + c];
        };
        const float h2 = fetch(1), h1 = fetch(2), h0 = fetch(3);
        stage_sconv[(size_t)m * kInkSconvCh + (is_k ? 0 : kInkKVDim) + c] = cur;

        const float4 sw = reinterpret_cast<const float4*>(swp)[c];
        float acc = h0 * sw.x + h1 * sw.y + h2 * sw.z + cur * sw.w;
        const float sc_conv = dyn ? (1.0f / 3.0f) : 0.44721359549995793f;
        if (dyn) {
            const float4 dt = reinterpret_cast<const float4*>(dyn + ((size_t)m * kInkKVHeads + kh) * 4)[0];
            acc += cur * dt.x + h2 * dt.y + h1 * dt.z + h0 * dt.w;
        }
        float val = (cur + acc) * sc_conv;
        float ss = block_sum(val * val, sh);
        if (is_k) val *= fminf(1.0f, kInkQkClip / (sqrtf(ss) + 1e-6f));
        else      val *= rsqrtf(ss * (1.0f / (float)kInkHeadDim) + kInkRmsEps);
        const __half hv = __float2half_rn(fminf(fmaxf(val, -65504.0f), 65504.0f));
        (is_k ? k_out : v_out)[(size_t)m * kInkKVDim + c] = __half2float(hv);
        (is_k ? stage_k : stage_v)[(size_t)m * kInkKVDim + c] = hv;
    } else if (blk < 64) {
        if (tid < kInkDRel) {
            const int h = blk - 40;
            float* rp = r + ((size_t)m * kInkHeads + h) * kInkDRel;
            float val = rp[tid];
            if (r_scale) {
                const int si = (r_scale_elems == kInkDRel) ? tid : (h * kInkDRel + tid);
                val *= r_scale[si];
            }
            float ss = val * val;
#pragma unroll
            for (int o = 8; o >= 1; o >>= 1) ss += __shfl_xor_sync(0x0000FFFFu, ss, o);
            const float factor = fminf(1.0f, kInkRelClip / (sqrtf(ss) + 1e-6f)) * tau;
            rp[tid] = val * factor;
        }
    }
}

// Mode-3 Flash-GQA: attends committed slot KV [k_start..pos0-1] + intra-slot draft [0..k]!
__global__ void __launch_bounds__(128) k_ink_flash_gqa_bspec(
    const float* __restrict__ q,
    const float* __restrict__ rel_table,
    const float* __restrict__ sinks,
    const float* __restrict__ gate_attn,
    InkKvVisitDesc kv,
    const int32_t* __restrict__ slots,
    const int32_t* __restrict__ q_pos,
    const float* __restrict__ draft_k,
    const float* __restrict__ draft_v,
    float* __restrict__ out,
    int K_slot,
    int M) {
    const int kh = blockIdx.x;
    const int m  = blockIdx.y;
    if (kh >= kInkKVHeads || m >= M) return;

    const int tid  = threadIdx.x;
    const int warp = tid >> 5;
    const int lane = tid & 31;
    const int sl   = slots[m];
    const int qp   = q_pos[m];
    const int k    = m % K_slot;
    const int base_m = m - k;
    const int pos0 = qp - k;
    const float scale = 0.08838834764831845f;

    const int h0 = 3 * kh;
    const float4 qv0 = *(const float4*)(q + ((size_t)m * kInkHeads + h0 + 0) * kInkHeadDim + lane * 4);
    const float4 qv1 = *(const float4*)(q + ((size_t)m * kInkHeads + h0 + 1) * kInkHeadDim + lane * 4);
    const float4 qv2 = *(const float4*)(q + ((size_t)m * kInkHeads + h0 + 2) * kInkHeadDim + lane * 4);
    const float* rt0 = rel_table + ((size_t)m * kInkHeads + h0 + 0) * (kInkRelExtent + 1);
    const float* rt1 = rel_table + ((size_t)m * kInkHeads + h0 + 1) * (kInkRelExtent + 1);
    const float* rt2 = rel_table + ((size_t)m * kInkHeads + h0 + 2) * (kInkRelExtent + 1);

    const int k_start = kv.is_swa ? max(0, qp - kInkSwaWin + 1) : 0;
    const int k_end_safe = kv.is_swa ? pos0 : min(pos0, kv.cap_pos);

    float mw[3] = {
        (warp == 0) ? sinks[(size_t)m * kInkHeads + h0 + 0] : -1e30f,
        (warp == 0) ? sinks[(size_t)m * kInkHeads + h0 + 1] : -1e30f,
        (warp == 0) ? sinks[(size_t)m * kInkHeads + h0 + 2] : -1e30f
    };
    float lw[3] = { (warp == 0) ? 1.0f : 0.0f, (warp == 0) ? 1.0f : 0.0f, (warp == 0) ? 1.0f : 0.0f };
    float a[3][4] = {};

    auto attend = [&](const float4& kv4, const float4& vv4, int dist) {
        const float sc[3] = {
            warp_sum(qv0.x * kv4.x + qv0.y * kv4.y + qv0.z * kv4.z + qv0.w * kv4.w) * scale + rt0[dist],
            warp_sum(qv1.x * kv4.x + qv1.y * kv4.y + qv1.z * kv4.z + qv1.w * kv4.w) * scale + rt1[dist],
            warp_sum(qv2.x * kv4.x + qv2.y * kv4.y + qv2.z * kv4.z + qv2.w * kv4.w) * scale + rt2[dist]
        };
#pragma unroll
        for (int r = 0; r < 3; ++r) {
            const float mn = fmaxf(mw[r], sc[r]);
            const float al = __expf(mw[r] - mn);
            const float p  = __expf(sc[r] - mn);
            lw[r] = lw[r] * al + p;
            a[r][0] = a[r][0] * al + p * vv4.x;
            a[r][1] = a[r][1] * al + p * vv4.y;
            a[r][2] = a[r][2] * al + p * vv4.z;
            a[r][3] = a[r][3] * al + p * vv4.w;
            mw[r] = mn;
        }
    };

    for (int kp = k_start + warp; kp < k_end_safe; kp += 4) {
        const int p_idx = kv.is_swa ? (kp % kv.cap_pos) : kp;
        float4 kv4, vv4;
        ink_kv_load4(kv, sl, p_idx, kh, lane, kv4, vv4);
        const int dist = min(max(qp - kp, 0), kInkRelExtent);
        attend(kv4, vv4, dist);
    }
    for (int j = warp; j <= k; j += 4) {
        const int dm = base_m + j;
        const size_t off = ((size_t)dm * kInkKVHeads + kh) * kInkHeadDim + lane * 4;
        const float4 kv4 = *(const float4*)(draft_k + off);
        const float4 vv4 = *(const float4*)(draft_v + off);
        attend(kv4, vv4, k - j);
    }

    __shared__ float sm_m[4], sm_l[4], sm_a[4][kInkHeadDim];
#pragma unroll
    for (int r = 0; r < 3; ++r) {
        if (lane == 0) { sm_m[warp] = mw[r]; sm_l[warp] = lw[r]; }
        sm_a[warp][lane * 4 + 0] = a[r][0]; sm_a[warp][lane * 4 + 1] = a[r][1];
        sm_a[warp][lane * 4 + 2] = a[r][2]; sm_a[warp][lane * 4 + 3] = a[r][3];
        __syncthreads();
        if (warp == 0) {
            const float mg = fmaxf(fmaxf(sm_m[0], sm_m[1]), fmaxf(sm_m[2], sm_m[3]));
            const float w0 = sm_l[0] > 0.0f ? __expf(sm_m[0] - mg) : 0.0f;
            const float w1 = sm_l[1] > 0.0f ? __expf(sm_m[1] - mg) : 0.0f;
            const float w2 = sm_l[2] > 0.0f ? __expf(sm_m[2] - mg) : 0.0f;
            const float w3 = sm_l[3] > 0.0f ? __expf(sm_m[3] - mg) : 0.0f;
            const float inv = 1.0f / (sm_l[0] * w0 + sm_l[1] * w1 + sm_l[2] * w2 + sm_l[3] * w3);
            const size_t ob = ((size_t)m * kInkHeads + h0 + r) * kInkHeadDim + lane * 4;
            float o0 = (sm_a[0][lane * 4 + 0] * w0 + sm_a[1][lane * 4 + 0] * w1 + sm_a[2][lane * 4 + 0] * w2 + sm_a[3][lane * 4 + 0] * w3) * inv;
            float o1 = (sm_a[0][lane * 4 + 1] * w0 + sm_a[1][lane * 4 + 1] * w1 + sm_a[2][lane * 4 + 1] * w2 + sm_a[3][lane * 4 + 1] * w3) * inv;
            float o2 = (sm_a[0][lane * 4 + 2] * w0 + sm_a[1][lane * 4 + 2] * w1 + sm_a[2][lane * 4 + 2] * w2 + sm_a[3][lane * 4 + 2] * w3) * inv;
            float o3 = (sm_a[0][lane * 4 + 3] * w0 + sm_a[1][lane * 4 + 3] * w1 + sm_a[2][lane * 4 + 3] * w2 + sm_a[3][lane * 4 + 3] * w3) * inv;
            if (gate_attn) {
                const float4 gv = *(const float4*)(gate_attn + ob);
                o0 *= 1.0f / (1.0f + __expf(-gv.x)); o1 *= 1.0f / (1.0f + __expf(-gv.y));
                o2 *= 1.0f / (1.0f + __expf(-gv.z)); o3 *= 1.0f / (1.0f + __expf(-gv.w));
            }
            *(float4*)(out + ob) = make_float4(o0, o1, o2, o3);
        }
        __syncthreads();
    }
}

// Mode-3 Post-Attn / Post-MoE ShortConv4 + Stage (non-mutating!)
__global__ void __launch_bounds__(256) k_ink_sconv4_bspec(
    const float* __restrict__ src,
    const float* __restrict__ sw,
    const float* __restrict__ hist_base,
    const int32_t* __restrict__ slots,
    float* __restrict__ dst,
    float* __restrict__ stage_sconv,
    int ch_off, int M, int C, int K_slot, bool add_to_out) {
    const int m = blockIdx.y;
    const int c = blockIdx.x * 256 + threadIdx.x;
    if (m >= M || c >= C) return;
    const int sl = slots[m];
    const int k  = m % K_slot;
    const float* hist = hist_base + (size_t)sl * 3 * C;
    const float cur = src[(size_t)m * C + c];
    auto fetch = [&](int lag) -> float {
        if (k >= lag) return src[(size_t)(m - lag) * C + c];
        return hist[(size_t)(3 + k - lag) * C + c];
    };
    const float h2 = fetch(1), h1 = fetch(2), h0 = fetch(3);
    stage_sconv[(size_t)m * kInkSconvCh + ch_off + c] = cur;
    const float4 w = reinterpret_cast<const float4*>(sw)[c];
    const float y = (cur + h0 * w.x + h1 * w.y + h2 * w.z + cur * w.w) * 0.44721359549995793f;
    if (add_to_out) dst[(size_t)m * C + c] += y;
    else            dst[(size_t)m * C + c]  = y;
}

__global__ void __launch_bounds__(256) k_ink_commit_sconv_bspec(
    float* const* __restrict__ sconv_pools,
    const float* __restrict__ stage_sconv,
    const int32_t* __restrict__ slots_b,
    const int32_t* __restrict__ n_acc_b,
    int B, int K_slot, int max_slots, int m_cap) {
    const int v = blockIdx.x;
    const int b = blockIdx.y;
    if (v >= kInkVisits || b >= B) return;
    const int sl = slots_b[b];
    const int na = n_acc_b[b];
    if (na <= 0) return;
    const int base_m = b * K_slot;
    const float* st_v = stage_sconv + (size_t)v * m_cap * kInkSconvCh;
    constexpr int c_off[4] = {0, kInkKVDim, 2 * kInkKVDim, 2 * kInkKVDim + kInkHidden};
    constexpr int c_len[4] = {kInkKVDim, kInkKVDim, kInkHidden, kInkHidden};
    for (int strm = 0; strm < 4; ++strm) {
        const int off = c_off[strm];
        const int C   = c_len[strm];
        float* hist = sconv_pools[v] + (size_t)off * 3 * max_slots + (size_t)sl * 3 * C;
        for (int c = threadIdx.x; c < C; c += 256) {
            float nh0, nh1, nh2;
            if (na == 1) {
                nh0 = hist[1 * C + c];
                nh1 = hist[2 * C + c];
                nh2 = st_v[(size_t)(base_m + 0) * kInkSconvCh + off + c];
            } else if (na == 2) {
                nh0 = hist[2 * C + c];
                nh1 = st_v[(size_t)(base_m + 0) * kInkSconvCh + off + c];
                nh2 = st_v[(size_t)(base_m + 1) * kInkSconvCh + off + c];
            } else {
                nh0 = st_v[(size_t)(base_m + na - 3) * kInkSconvCh + off + c];
                nh1 = st_v[(size_t)(base_m + na - 2) * kInkSconvCh + off + c];
                nh2 = st_v[(size_t)(base_m + na - 1) * kInkSconvCh + off + c];
            }
            hist[0 * C + c] = nh0;
            hist[1 * C + c] = nh1;
            hist[2 * C + c] = nh2;
        }
    }
}

__global__ void __launch_bounds__(128) k_ink_commit_kv_bspec(
    const InkKvVisitDesc* __restrict__ kv_descs,
    const __half* __restrict__ stage_k,
    const __half* __restrict__ stage_v,
    const int32_t* __restrict__ slots_b,
    const int32_t* __restrict__ p0_b,
    const int32_t* __restrict__ n_acc_b,
    int B, int K_slot, int m_cap) {
    const int v  = blockIdx.x / kInkKVHeads;
    const int kh = blockIdx.x % kInkKVHeads;
    const int b  = blockIdx.y;
    const int d  = threadIdx.x;
    if (v >= kInkVisits || b >= B) return;
    const int sl = slots_b[b];
    const int p0 = p0_b[b];
    const int na = n_acc_b[b];
    const int base_m = b * K_slot;
    const InkKvVisitDesc kv = kv_descs[v];
    __shared__ float sk[4], sv[4];
    for (int j = 0; j < na; ++j) {
        const int p = p0 + j;
        if (p < 0 || (!kv.is_swa && p >= kv.cap_pos)) continue;
        const int p_idx = kv.is_swa ? (p % kv.cap_pos) : p;
        const size_t soff = ((size_t)v * m_cap + (base_m + j)) * kInkKVDim + (size_t)kh * kInkHeadDim + d;
        const float kv_val = __half2float(stage_k[soff]);
        const float vv_val = __half2float(stage_v[soff]);
        ink_kv_store_128(kv, sl, p_idx, kh, d, kv_val, true, sk);
        ink_kv_store_128(kv, sl, p_idx, kh, d, vv_val, false, sv);
    }
}

// ---- Fast INT8 TC Screened Exact Argmax Kernels (shared by C1 Q8_ROW & C64 Mode-3!) ----
__global__ void __launch_bounds__(256) k_ms_quant_h_q8(
    const float* __restrict__ h, int8_t* __restrict__ h_q8, float* __restrict__ h_sc, int M) {
    const int m = blockIdx.x;
    if (m >= M) return;
    const int tid = threadIdx.x;
    const float* hp = h + (size_t)m * kInkHidden;
    float am = 0.0f;
    for (int d = tid; d < kInkHidden; d += 256) am = fmaxf(am, fabsf(hp[d]));
    am = warp_max(am);
    __shared__ float sm[8];
    if ((tid & 31) == 0) sm[tid >> 5] = am;
    __syncthreads();
    am = fmaxf(fmaxf(fmaxf(sm[0], sm[1]), fmaxf(sm[2], sm[3])),
               fmaxf(fmaxf(sm[4], sm[5]), fmaxf(sm[6], sm[7])));
    const float sc = am * (1.0f / 127.0f);
    const float inv = (sc > 0.0f) ? (127.0f / am) : 0.0f;
    if (tid == 0) h_sc[m] = sc;
    int8_t* dst = h_q8 + (size_t)m * kInkHidden;
    for (int d = tid; d < kInkHidden; d += 256) {
        dst[d] = (int8_t)max(-127, min(127, __float2int_rn(hp[d] * inv)));
    }
}

__global__ void __launch_bounds__(256) k_ms_top_i32_s1(
    const int32_t* __restrict__ dots_i32, const float* __restrict__ w_sc, const float* __restrict__ h_sc,
    int V8, int32_t* __restrict__ tmp_ids, float* __restrict__ tmp_vals, int M) {
    const int m = blockIdx.y;
    const int ch = blockIdx.x;
    if (m >= M) return;
    const int tid = threadIdx.x;
    const int warp = tid >> 5;
    const int lane = tid & 31;
    const int chunk = (V8 + 63) / 64;
    const int beg = ch * chunk;
    const int end = min(V8, beg + chunk);
    const float hs = h_sc[m];
    const int32_t* row = dots_i32 + (size_t)m * V8;
    float bv = -3e38f;
    int32_t bi = -1;
    for (int v = beg + tid; v < end; v += 256) {
        const float val = (float)row[v] * w_sc[v] * hs;
        if (val > bv) { bv = val; bi = v; }
    }
#pragma unroll
    for (int o = 16; o >= 1; o >>= 1) {
        const float ov = __shfl_xor_sync(0xFFFFFFFFu, bv, o);
        const int32_t oi = __shfl_xor_sync(0xFFFFFFFFu, bi, o);
        if (oi >= 0 && (bi < 0 || ov > bv || (ov == bv && oi < bi))) { bv = ov; bi = oi; }
    }
    if (lane == 0) {
        const int out_idx = ch * 8 + warp;
        tmp_ids[(size_t)m * 512 + out_idx]  = bi;
        tmp_vals[(size_t)m * 512 + out_idx] = bv;
    }
}

__global__ void __launch_bounds__(256) k_ms_top_i32_s2(
    const int32_t* __restrict__ tmp_ids, const float* __restrict__ tmp_vals,
    int K_out, int32_t* __restrict__ out_ids, int32_t* __restrict__ out_cnt,
    int V_full, int V8, int M) {
    const int m = blockIdx.x;
    if (m >= M) return;
    const int tid = threadIdx.x;
    const int warp = tid >> 5;
    const int lane = tid & 31;
    __shared__ float sv[512];
    __shared__ int32_t si[512];
    __shared__ float wv[8];
    __shared__ int32_t wi[8];
    for (int i = tid; i < 512; i += 256) {
        sv[i] = tmp_vals[(size_t)m * 512 + i];
        si[i] = tmp_ids[(size_t)m * 512 + i];
    }
    __syncthreads();
    int32_t* dst = out_ids + (size_t)m * K_out;
    const int k_pick = out_cnt ? min(K_out - (V_full - V8), K_out) : K_out;
    for (int k = 0; k < k_pick; ++k) {
        float bv = -3e38f;
        int32_t bp = -1;
        for (int i = tid; i < 512; i += 256) {
            const float v = sv[i];
            if (v > bv) { bv = v; bp = i; }
        }
#pragma unroll
        for (int o = 16; o >= 1; o >>= 1) {
            const float ov = __shfl_xor_sync(0xFFFFFFFFu, bv, o);
            const int32_t op = __shfl_xor_sync(0xFFFFFFFFu, bp, o);
            if (op >= 0 && (bp < 0 || ov > bv)) { bv = ov; bp = op; }
        }
        if (lane == 0) { wv[warp] = bv; wi[warp] = bp; }
        __syncthreads();
        if (tid == 0) {
            float gv = -3e38f; int32_t gp = 0;
            for (int w = 0; w < 8; ++w) {
                if (wi[w] >= 0 && wv[w] > gv) { gv = wv[w]; gp = wi[w]; }
            }
            dst[k] = (si[gp] >= 0) ? si[gp] : 0;
            sv[gp] = -3e38f;
        }
        __syncthreads();
    }
    if (out_cnt && tid == 0) {
        int n = k_pick;
        for (int v = V8; v < V_full && n < K_out; ++v) dst[n++] = v;
        out_cnt[m] = n;
    }
}

__global__ void __launch_bounds__(256) k_ink_screen_eval_q8row(
    const uint8_t* __restrict__ W_q8row, int V_rows, const float* __restrict__ h,
    const int32_t* __restrict__ cand_ids, const int32_t* __restrict__ cand_cnt,
    int max_cand, int32_t* __restrict__ out_argmax, int M) {
    const int m = blockIdx.x;
    if (m >= M) return;
    const int n = min(cand_cnt[m], max_cand);
    const int warp = threadIdx.x >> 5;
    const int lane = threadIdx.x & 31;
    const float* hm = h + (size_t)m * kInkHidden;
    const float* w_sc = reinterpret_cast<const float*>(W_q8row);
    const int8_t* w_i8 = reinterpret_cast<const int8_t*>(W_q8row + (size_t)V_rows * sizeof(float));
    float best_v = -3e38f;
    int32_t best_i = 0x7FFFFFFF;
    for (int ci = warp; ci < n; ci += 8) {
        const int32_t row = cand_ids[(size_t)m * max_cand + ci];
        const uint4* q16 = reinterpret_cast<const uint4*>(w_i8 + (size_t)row * kInkHidden);
        float acc = 0.0f;
        for (int i = lane; i < (kInkHidden / 16); i += 32) {
            const uint4 v = q16[i];
            const uint32_t u[4] = {v.x, v.y, v.z, v.w};
            const int base = i * 16;
#pragma unroll
            for (int q = 0; q < 4; ++q) {
                const float4 xv = *reinterpret_cast<const float4*>(hm + base + q * 4);
                const uint32_t p = u[q];
                acc = fmaf((float)(int8_t)(p & 0xFFu), xv.x, acc);
                acc = fmaf((float)(int8_t)((p >> 8) & 0xFFu), xv.y, acc);
                acc = fmaf((float)(int8_t)((p >> 16) & 0xFFu), xv.z, acc);
                acc = fmaf((float)(int8_t)((p >> 24) & 0xFFu), xv.w, acc);
            }
        }
        const float val = warp_sum(acc) * w_sc[row];
        if (lane == 0 && (val > best_v || (val == best_v && row < best_i))) {
            best_v = val; best_i = row;
        }
    }
    __shared__ float sv[8];
    __shared__ int32_t si[8];
    if (lane == 0) { sv[warp] = best_v; si[warp] = best_i; }
    __syncthreads();
    if (threadIdx.x == 0) {
        float bv = sv[0]; int32_t bi = si[0];
        for (int w = 1; w < 8; ++w) {
            if (sv[w] > bv || (sv[w] == bv && si[w] < bi)) { bv = sv[w]; bi = si[w]; }
        }
        out_argmax[m] = bi;
    }
}

}  // namespace strata::ink::rtx5090
