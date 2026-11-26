// engine/src/inkling2/cuda/ink_rtx5090_bspec.cuh - Opt-in RTX 5090 Batched 64-Slot DSpark + Mode-3 Speculative Engine
// MIT License - Copyright (c) 2026 Niko1221 and the Strata contributors
// Included AFTER ink_dspark.cuh in ink_cuda_backend.cu. Strictly OFF by default!
#pragma once

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

#include "ink_dspark.cuh"
#include "ink_rtx5090_bspec_kernels.cuh"

namespace strata::ink::rtx5090 {

__global__ void __launch_bounds__(256) k_ms_mix_taps_indexed(
    const float* __restrict__ tap0, const float* __restrict__ tap1,
    const float* __restrict__ tap2, const float* __restrict__ tap3,
    const int32_t* __restrict__ src_idx, const float* __restrict__ fuse_w,
    int row, const float* __restrict__ norm_w, float* __restrict__ out, int N_tot) {
    const int i = blockIdx.x;
    if (i >= N_tot) return;
    const int tid = threadIdx.x;
    __shared__ float sw[4], sh[32];
    if (tid == 0) {
        const float* fw = fuse_w + row * 4;
        const float mx = fmaxf(fmaxf(fw[0], fw[1]), fmaxf(fw[2], fw[3]));
        const float e0 = expf(fw[0] - mx), e1 = expf(fw[1] - mx), e2 = expf(fw[2] - mx), e3 = expf(fw[3] - mx);
        const float inv = 1.0f / (e0 + e1 + e2 + e3);
        sw[0] = e0 * inv; sw[1] = e1 * inv; sw[2] = e2 * inv; sw[3] = e3 * inv;
    }
    __syncthreads();
    const int sm = src_idx ? src_idx[i] : i;
    const size_t soff = (size_t)sm * kInkHidden;
    float* dst = out + (size_t)i * kInkHidden;
    float ss = 0.0f;
    for (int d = tid; d < kInkHidden; d += 256) {
        const float v = sw[0] * tap0[soff + d] + sw[1] * tap1[soff + d] + sw[2] * tap2[soff + d] + sw[3] * tap3[soff + d];
        dst[d] = v;
        ss += v * v;
    }
    ss = block_sum(ss, sh);
    const float inv = rsqrtf(ss / (float)kInkHidden + kInkRmsEps);
    for (int d = tid; d < kInkHidden; d += 256) dst[d] = dst[d] * inv * norm_w[d];
}

__global__ void __launch_bounds__(256) k_ms_mix_frnn_indexed(
    const float* __restrict__ tap0, const float* __restrict__ tap1,
    const float* __restrict__ tap2, const float* __restrict__ tap3,
    const int32_t* __restrict__ src_idx, const int32_t* __restrict__ slots,
    int slot0, const int32_t* __restrict__ pos_arr, const float* __restrict__ fuse_w,
    const float* __restrict__ norm_w, float* __restrict__ frnn_pool, int N_tot) {
    const int i = blockIdx.x;
    if (i >= N_tot) return;
    const int tid = threadIdx.x;
    __shared__ float sw[4], sh[32];
    if (tid == 0) {
        const float* fw = fuse_w + 3 * 4;
        const float mx = fmaxf(fmaxf(fw[0], fw[1]), fmaxf(fw[2], fw[3]));
        const float e0 = expf(fw[0] - mx), e1 = expf(fw[1] - mx), e2 = expf(fw[2] - mx), e3 = expf(fw[3] - mx);
        const float inv = 1.0f / (e0 + e1 + e2 + e3);
        sw[0] = e0 * inv; sw[1] = e1 * inv; sw[2] = e2 * inv; sw[3] = e3 * inv;
    }
    __syncthreads();
    const int sm = src_idx ? src_idx[i] : i;
    const int sl = slots ? slots[i] : slot0;
    const int p  = pos_arr[i] & 7;
    const size_t soff = (size_t)sm * kInkHidden;
    float* dst = frnn_pool + ((size_t)sl * 8 + p) * kInkHidden;
    float ss = 0.0f;
    for (int d = tid; d < kInkHidden; d += 256) {
        const float v = sw[0] * tap0[soff + d] + sw[1] * tap1[soff + d] + sw[2] * tap2[soff + d] + sw[3] * tap3[soff + d];
        dst[d] = v;
        ss += v * v;
    }
    ss = block_sum(ss, sh);
    const float inv = rsqrtf(ss / (float)kInkHidden + kInkRmsEps);
    for (int d = tid; d < kInkHidden; d += 256) dst[d] = dst[d] * inv * norm_w[d];
}

__global__ void __launch_bounds__(256) k_ms_init_block8(
    const uint8_t* __restrict__ emb_w, int emb_fmt, const float* __restrict__ emb_nw,
    const float* __restrict__ mask_e, const int32_t* __restrict__ anchor_toks,
    float* __restrict__ out_cur, int B) {
    const int row = blockIdx.x;
    const int b   = row >> 3;
    const int k   = row & 7;
    if (b >= B) return;
    const int tid = threadIdx.x;
    float* dst = out_cur + (size_t)row * kInkHidden;
    if (k > 0) {
        const float4* src4 = reinterpret_cast<const float4*>(mask_e);
        float4* dst4 = reinterpret_cast<float4*>(dst);
        for (int i = tid; i < (kInkHidden / 4); i += 256) dst4[i] = src4[i];
        return;
    }
    const int32_t tok = anchor_toks[b];
    __shared__ float sh[32];
    float ss = 0.0f;
    for (int d = tid; d < kInkHidden; d += 256) {
        float v = 0.0f;
        if (emb_fmt == FMT_BF16) {
            v = bf16_to_f32(reinterpret_cast<const uint16_t*>(emb_w)[(size_t)tok * kInkHidden + d]);
        } else if (emb_fmt == FMT_F32) {
            v = reinterpret_cast<const float*>(emb_w)[(size_t)tok * kInkHidden + d];
        } else if (emb_fmt == FMT_Q8ROW) {
            const float sc = reinterpret_cast<const float*>(emb_w)[tok];
            v = (float)reinterpret_cast<const int8_t*>(emb_w + (size_t)kInkVocabPad * 4)[(size_t)tok * kInkHidden + d] * sc;
        } else if (emb_fmt == FMT_S6_K) {
            const int sb_b = ink_sb_bytes(FMT_S6_K);
            const int s = d >> 8, idx = d & 255, g = idx >> 6;
            const uint8_t* blk = emb_w + ((size_t)tok * 9 + s) * sb_b;
            const float eff = f16_to_f32(ld16_u(blk)) * ((float)blk[2 + g] * (1.0f / 255.0f));
            const uint8_t* pl = blk + 6;
            const uint32_t lo = (pl[idx >> 1] >> ((idx & 1) * 4)) & 0x0Fu;
            const uint32_t hi = (pl[128 + (idx >> 2)] >> ((idx & 3) * 2)) & 3u;
            v = ((float)(lo | (hi << 4)) - 31.5f) * eff;
        }
        dst[d] = v;
        ss += v * v;
    }
    ss = block_sum(ss, sh);
    const float inv = rsqrtf(ss / (float)kInkHidden + kInkRmsEps);
    for (int d = tid; d < kInkHidden; d += 256) dst[d] = dst[d] * inv * emb_nw[d];
}

__global__ void __launch_bounds__(128) k_ms_dspark_block_attn(
    const float* __restrict__ q, const float* __restrict__ kd, const float* __restrict__ vd,
    InkKvVisitDesc kv, const int32_t* __restrict__ slots_b, const int32_t* __restrict__ apos_b,
    int swa_win, int B, float* __restrict__ out) {
    const int row = blockIdx.y;
    const int h   = blockIdx.x;
    const int b   = row >> 3;
    if (b >= B || h >= kInkHeads) return;

    const int sl   = slots_b[b];
    const int apos = apos_b[b];
    const int tid  = threadIdx.x;
    const int warp = tid >> 5;
    const int lane = tid & 31;
    const int kh   = h / 3;
    const float scale = 0.08838834764831845f;

    const float4 qv = *(const float4*)(q + ((size_t)row * kInkHeads + h) * kInkHeadDim + lane * 4);
    const int win = kv.is_swa ? min(swa_win - 1, kv.cap_pos) : kv.cap_pos;
    const int c0  = max(0, apos - win);
    const int c_end = apos;

    float mw = -1e30f, lw = 0.0f;
    float a0 = 0.0f, a1 = 0.0f, a2 = 0.0f, a3 = 0.0f;
    auto attend = [&](const float4& kv4, const float4& vv4) {
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
        const int p_idx = kp % kv.cap_pos;
        float4 kv4, vv4;
        ink_kv_load4(kv, sl, p_idx, kh, lane, kv4, vv4);
        attend(kv4, vv4);
    }
    const int b0 = b << 3;
    for (int j = warp; j < 8; j += 4) {
        const size_t base = ((size_t)(b0 + j) * kInkKVHeads + kh) * kInkHeadDim + lane * 4;
        attend(*(const float4*)(kd + base), *(const float4*)(vd + base));
    }

    __shared__ float sm_m[4], sm_l[4], sm_a[4][kInkHeadDim];
    if (lane == 0) { sm_m[warp] = mw; sm_l[warp] = lw; }
    sm_a[warp][lane * 4 + 0] = a0; sm_a[warp][lane * 4 + 1] = a1;
    sm_a[warp][lane * 4 + 2] = a2; sm_a[warp][lane * 4 + 3] = a3;
    __syncthreads();
    if (warp == 0) {
        const float mg = fmaxf(fmaxf(sm_m[0], sm_m[1]), fmaxf(sm_m[2], sm_m[3]));
        const float w0 = sm_l[0] > 0.0f ? __expf(sm_m[0] - mg) : 0.0f;
        const float w1 = sm_l[1] > 0.0f ? __expf(sm_m[1] - mg) : 0.0f;
        const float w2 = sm_l[2] > 0.0f ? __expf(sm_m[2] - mg) : 0.0f;
        const float w3 = sm_l[3] > 0.0f ? __expf(sm_m[3] - mg) : 0.0f;
        const float inv = 1.0f / (sm_l[0] * w0 + sm_l[1] * w1 + sm_l[2] * w2 + sm_l[3] * w3);
        const size_t ob = ((size_t)row * kInkHeads + h) * kInkHeadDim + lane * 4;
        *(float4*)(out + ob) = make_float4(
            (sm_a[0][lane * 4 + 0] * w0 + sm_a[1][lane * 4 + 0] * w1 + sm_a[2][lane * 4 + 0] * w2 + sm_a[3][lane * 4 + 0] * w3) * inv,
            (sm_a[0][lane * 4 + 1] * w0 + sm_a[1][lane * 4 + 1] * w1 + sm_a[2][lane * 4 + 1] * w2 + sm_a[3][lane * 4 + 1] * w3) * inv,
            (sm_a[0][lane * 4 + 2] * w0 + sm_a[1][lane * 4 + 2] * w1 + sm_a[2][lane * 4 + 2] * w2 + sm_a[3][lane * 4 + 2] * w3) * inv,
            (sm_a[0][lane * 4 + 3] * w0 + sm_a[1][lane * 4 + 3] * w1 + sm_a[2][lane * 4 + 3] * w2 + sm_a[3][lane * 4 + 3] * w3) * inv);
    }
}

__global__ void __launch_bounds__(256) k_ms_gather_draft_h(
    const float* __restrict__ cur_b8, float* __restrict__ dst_h, int B, int K_draft) {
    const int idx = blockIdx.x;
    const int b = idx / K_draft;
    const int s = idx % K_draft;
    if (b >= B) return;
    const float4* sp = reinterpret_cast<const float4*>(cur_b8 + (size_t)(b * 8 + 1 + s) * kInkHidden);
    float4* dp = reinterpret_cast<float4*>(dst_h + (size_t)idx * kInkHidden);
    for (int i = threadIdx.x; i < (kInkHidden / 4); i += 256) dp[i] = sp[i];
}

__global__ void __launch_bounds__(256) k_ms_markov_prep(
    const int32_t* __restrict__ prev_toks, const float* __restrict__ states,
    const float* __restrict__ cur_b8, const float* __restrict__ frnn_pool,
    const int32_t* __restrict__ slots_b, const int32_t* __restrict__ apos_b,
    const uint8_t* __restrict__ mw1_raw, int mw1_fmt,
    float* __restrict__ joint_out, int B, int step_s) {
    constexpr int R = kInkMarkovRank;
    constexpr int J = 2 * R + 2 * kInkHidden;
    const int b = blockIdx.x;
    if (b >= B) return;
    const int tid = threadIdx.x;
    const int32_t pt = prev_toks[b];
    const int sl = slots_b[b];
    const int apos = apos_b[b];
    const int fidx = (apos > 0 ? (apos - 1) : 0) & 7;
    const float* hidden = cur_b8 + (size_t)(b * 8 + 1 + step_s) * kInkHidden;
    const float* ctx = frnn_pool + ((size_t)sl * 8 + fidx) * kInkHidden;
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

__global__ void __launch_bounds__(256) k_ms_add_bias_and_h(
    float* __restrict__ corr, const float* __restrict__ ho_b,
    const float* __restrict__ cur_b8, int B, int step_s) {
    const int b = blockIdx.y;
    const int d = blockIdx.x * 256 + threadIdx.x;
    if (b >= B || d >= kInkHidden) return;
    corr[(size_t)b * kInkHidden + d] += ho_b[d] + cur_b8[(size_t)(b * 8 + 1 + step_s) * kInkHidden + d];
}

__global__ void __launch_bounds__(256) k_ms_score_cand_dp4a(
    const int8_t* __restrict__ q8, const float* __restrict__ sc,
    const int8_t* __restrict__ corr_q8, const float* __restrict__ corr_sc,
    const int32_t* __restrict__ cand_ids, int K_cand, int K_draft, int step_s, int B,
    float* __restrict__ cand_scores) {
    const int b = blockIdx.y;
    const int warp = threadIdx.x >> 5;
    const int lane = threadIdx.x & 31;
    const int k = blockIdx.x * 8 + warp;
    if (b >= B || k >= K_cand) return;
    const int32_t row = cand_ids[((size_t)b * K_draft + step_s) * K_cand + k];
    const uint4* q16 = reinterpret_cast<const uint4*>(q8 + (size_t)row * kInkHidden);
    const uint4* c16 = reinterpret_cast<const uint4*>(corr_q8 + (size_t)b * kInkHidden);
    int32_t acc = 0;
    for (int i = lane; i < (kInkHidden / 16); i += 32) {
        const uint4 v = q16[i];
        const uint4 c = __ldg(c16 + i);
        acc = dp4a_i8(v.x, c.x, acc);
        acc = dp4a_i8(v.y, c.y, acc);
        acc = dp4a_i8(v.z, c.z, acc);
        acc = dp4a_i8(v.w, c.w, acc);
    }
#pragma unroll
    for (int o = 16; o >= 1; o >>= 1) acc += __shfl_xor_sync(0xFFFFFFFFu, acc, o);
    if (lane == 0) cand_scores[(size_t)b * K_cand + k] = (float)acc * sc[row] * corr_sc[b];
}

__global__ void __launch_bounds__(128) k_ms_cand_argmax(
    const float* __restrict__ cand_scores, const int32_t* __restrict__ cand_ids,
    int K_cand, int K_draft, int step_s, int B,
    int32_t* __restrict__ prev_toks, int32_t* __restrict__ draft_out) {
    const int b = blockIdx.x;
    if (b >= B) return;
    const int tid = threadIdx.x;
    float bv = -3e38f;
    int32_t bi = 0;
    if (tid < K_cand) {
        bv = cand_scores[(size_t)b * K_cand + tid];
        bi = cand_ids[((size_t)b * K_draft + step_s) * K_cand + tid];
    }
    const int warp = tid >> 5;
    const int lane = tid & 31;
#pragma unroll
    for (int o = 16; o >= 1; o >>= 1) {
        const float ov = __shfl_xor_sync(0xFFFFFFFFu, bv, o);
        const int32_t oi = __shfl_xor_sync(0xFFFFFFFFu, bi, o);
        if (ov > bv || (ov == bv && oi < bi)) { bv = ov; bi = oi; }
    }
    __shared__ float sv[4];
    __shared__ int32_t si[4];
    if (lane == 0) { sv[warp] = bv; si[warp] = bi; }
    __syncthreads();
    if (tid == 0) {
        float gv = sv[0]; int32_t gi = si[0];
        for (int w = 1; w < 4; ++w) {
            if (sv[w] > gv || (sv[w] == gv && si[w] < gi)) { gv = sv[w]; gi = si[w]; }
        }
        prev_toks[b] = gi;
        draft_out[(size_t)b * K_draft + step_s] = gi;
    }
}

struct Rtx5090BatchedSpec {
    bool ready = false;
    int max_slots = 0;
    int m_cap = kBspecMaxM;
    int kv_cap[3] = {0, 0, 0};

    __half* d_stage_k = nullptr;
    __half* d_stage_v = nullptr;
    float*  d_stage_sconv = nullptr;

    InkKvVisitDesc kv_desc[3]{};
    uint8_t* d_kv_k[3] = {};
    uint8_t* d_kv_v[3] = {};
    float*   d_kv_ksc[3] = {};
    float*   d_kv_vsc[3] = {};
    float*   d_frnn = nullptr;

    float*   d_cur = nullptr;
    float*   d_h = nullptr;
    float*   d_qkv = nullptr;
    float*   d_q = nullptr;
    float*   d_k = nullptr;
    float*   d_v = nullptr;
    float*   d_attn_out = nullptr;
    float*   d_proj = nullptr;
    float*   d_gu = nullptr;
    float*   d_act = nullptr;
    float*   d_draft_h = nullptr;
    int8_t*  d_h_q8 = nullptr;
    float*   d_h_sc = nullptr;
    int32_t* d_base_i32 = nullptr;
    int32_t* d_top_tmp_i = nullptr;
    float*   d_top_tmp_v = nullptr;
    int32_t* d_cand_ids = nullptr;
    int32_t* d_cand_cnt = nullptr;
    float*   d_rnn_st = nullptr;
    float*   d_rnn_st_n = nullptr;
    float*   d_rnn_joint = nullptr;
    float*   d_rnn_proj = nullptr;
    float*   d_rnn_lat = nullptr;
    float*   d_corr = nullptr;
    float*   d_cand_sc = nullptr;
    int32_t* d_prev_tok = nullptr;
    int32_t* d_draft_out = nullptr;
    int32_t* d_meta = nullptr;
    int32_t* h_meta = nullptr;
    int32_t* h_draft = nullptr;
    int32_t* h_argmax = nullptr;

    std::vector<int> slot_synced_pos;
    std::vector<std::vector<int32_t>> pld_hist_slots;
    int gov_plain_steps = 0;
    int gov_low_streak = 0;

    bool init(InkCudaRunner& r, int max_sl) {
        if (!dspark_opt_on() || exact_mode_on() || max_sl < 2 || max_sl > kBspecMaxSlots) return false;
        if (!r.m.man.has_dspark || r.m.ls_active) return false;
        const auto& ui = r.m.tinfo("model.llm.unembed.weight");
        if (ui.fmt != FMT_Q8ROW && !r.m.unembed_screen.ready) return false;
        if (r.max_batch < kBspecMaxM) return false;

        size_t vf = 0, vt = 0;
        if (ink_cuda_mem_info(&vf, &vt, r.m.primary_gpu) != cudaSuccess) return false;

        const int swa_cap  = (vf > (2200ull << 20)) ? 1025 : 512;
        const int glob_cap = (vf > (2400ull << 20)) ? 1024 : 512;
        const size_t kv_bytes =
            (size_t)max_sl * (2ull * swa_cap + glob_cap) * ((size_t)kInkKVDim * 2 + (size_t)kInkKVHeads * 8);
        const size_t stage_bytes =
            (size_t)kInkVisits * m_cap * ((size_t)2 * kInkKVDim * 2 + (size_t)kInkSconvCh * 4);
        const size_t work_bytes = (size_t)kBspecMaxM * kBspecV8 * 4 + (96ull << 20);
        const size_t need = kv_bytes + stage_bytes + work_bytes + (384ull << 20);
        if (vf < need) {
            std::fprintf(stderr, "[InkRTX5090] Skipping Mode-3 Batched Spec: %.0f MiB free < %.0f MiB needed\n",
                         (double)vf / 1048576.0, (double)need / 1048576.0);
            return false;
        }

        max_slots = max_sl;
        d_stage_k     = dalloc<__half>((size_t)kInkVisits * m_cap * kInkKVDim);
        d_stage_v     = dalloc<__half>((size_t)kInkVisits * m_cap * kInkKVDim);
        d_stage_sconv = dalloc<float>((size_t)kInkVisits * m_cap * kInkSconvCh);
        r.d_bspec_k     = d_stage_k;
        r.d_bspec_v     = d_stage_v;
        r.d_bspec_sconv = d_stage_sconv;
        r.bspec_m_cap   = m_cap;

        for (int il = 0; il < 3; ++il) {
            const bool is_swa = (r.m.man.c.dspark_is_swa[il] != 0);
            const int cap = is_swa ? swa_cap : glob_cap;
            kv_cap[il] = cap;
            d_kv_k[il]   = dalloc<uint8_t>((size_t)max_slots * cap * kInkKVDim);
            d_kv_v[il]   = dalloc<uint8_t>((size_t)max_slots * cap * kInkKVDim);
            d_kv_ksc[il] = dalloc<float>((size_t)max_slots * cap * kInkKVHeads);
            d_kv_vsc[il] = dalloc<float>((size_t)max_slots * cap * kInkKVHeads);
            kv_desc[il] = InkKvVisitDesc{};
            kv_desc[il].k_f8    = d_kv_k[il];
            kv_desc[il].v_f8    = d_kv_v[il];
            kv_desc[il].k_sc    = d_kv_ksc[il];
            kv_desc[il].v_sc    = d_kv_vsc[il];
            kv_desc[il].cap_pos = cap;
            kv_desc[il].dev_cap = cap;
            kv_desc[il].is_swa  = is_swa;
            kv_desc[il].kv_fmt  = 1;
        }
        d_frnn = dalloc<float>((size_t)max_slots * 8 * kInkHidden);

        const size_t N512 = (size_t)max_slots * 8;
        d_cur       = dalloc<float>(N512 * kInkHidden);
        d_h         = dalloc<float>(N512 * kInkHidden);
        d_qkv       = dalloc<float>(N512 * (kInkQDim + 2 * kInkKVDim));
        d_q         = dalloc<float>(N512 * kInkQDim);
        d_k         = dalloc<float>(N512 * kInkKVDim);
        d_v         = dalloc<float>(N512 * kInkKVDim);
        d_attn_out  = dalloc<float>(N512 * kInkQDim);
        d_proj      = dalloc<float>(N512 * kInkHidden);
        d_gu        = dalloc<float>(N512 * 2 * kInkDsparkInter);
        d_act       = dalloc<float>(N512 * kInkDsparkInter);
        d_draft_h   = dalloc<float>((size_t)m_cap * kInkHidden);
        d_h_q8      = dalloc<int8_t>((size_t)m_cap * kInkHidden);
        d_h_sc      = dalloc<float>(m_cap);
        d_base_i32  = dalloc<int32_t>((size_t)m_cap * kBspecV8);
        d_top_tmp_i = dalloc<int32_t>((size_t)m_cap * 512);
        d_top_tmp_v = dalloc<float>((size_t)m_cap * 512);
        d_cand_ids  = dalloc<int32_t>((size_t)m_cap * kBspecKcand);
        d_cand_cnt  = dalloc<int32_t>(m_cap);
        d_rnn_st    = dalloc<float>((size_t)max_slots * kInkMarkovRank);
        d_rnn_st_n  = dalloc<float>((size_t)max_slots * kInkMarkovRank);
        d_rnn_joint = dalloc<float>((size_t)max_slots * (2 * kInkMarkovRank + 2 * kInkHidden));
        d_rnn_proj  = dalloc<float>((size_t)max_slots * 3 * kInkMarkovRank);
        d_rnn_lat   = dalloc<float>((size_t)max_slots * kInkMarkovRank);
        d_corr      = dalloc<float>((size_t)max_slots * kInkHidden);
        d_cand_sc   = dalloc<float>((size_t)max_slots * kBspecKcand);
        d_prev_tok  = dalloc<int32_t>(max_slots);
        d_draft_out = dalloc<int32_t>(m_cap);
        d_meta      = dalloc<int32_t>(2048);
        h_meta      = halloc_pinned<int32_t>(2048);
        h_draft     = halloc_pinned<int32_t>(m_cap);
        h_argmax    = halloc_pinned<int32_t>(m_cap);
        slot_synced_pos.assign(max_slots, 0);
        pld_hist_slots.assign(max_slots, {});
        ready = true;
        std::fprintf(stderr, "[InkRTX5090] Mode-3 Batched Spec ready: max_slots=%d swa=%d glob=%d (~%.0f MiB)\n",
                     max_slots, swa_cap, glob_cap, (double)(kv_bytes + stage_bytes + work_bytes) / 1048576.0);
        return true;
    }

    void reset_slot(int sl) {
        if (sl >= 0 && sl < (int)slot_synced_pos.size()) {
            slot_synced_pos[sl] = 0;
            pld_hist_slots[sl].clear();
        }
    }

    void record_slot_tokens(int sl, int p0, const int32_t* toks, int n) {
        if (!ready || sl < 0 || sl >= max_slots || n <= 0) return;
        auto& h = pld_hist_slots[sl];
        if ((int)h.size() < p0 + n) h.resize((size_t)(p0 + n), -1);
        for (int i = 0; i < n; ++i) h[(size_t)(p0 + i)] = toks[i];
    }

    // Per-slot >= 3-gram PLD suffix override (0 extra VRAM, huge win on structured/code/chat batches!)
    int find_slot_pld(int sl, int apos, int k_want, int32_t* out_cont) const {
        if (sl < 0 || sl >= (int)pld_hist_slots.size() || apos < 4) return 0;
        const auto& h = pld_hist_slots[sl];
        if ((int)h.size() <= apos) return 0;
        const int lo = std::max(0, apos - 4096);
        const int32_t last_t = h[(size_t)apos];
        if (last_t < 0) return 0;
        const bool allow_2gram = (apos >= 2) && (last_t >= 256 || h[(size_t)(apos - 1)] >= 256);
        const int min_n = allow_2gram ? 2 : 3;
        for (int n = 4; n >= min_n; --n) {
            if (apos < n + 1) continue;
            for (int p = apos - 1; p >= lo + n - 1; --p) {
                if (h[(size_t)p] != last_t) continue;
                bool eq = true;
                for (int j = 1; j < n; ++j) {
                    if (h[(size_t)(p - j)] != h[(size_t)(apos - j)]) { eq = false; break; }
                }
                if (eq) {
                    int cnt = 0;
                    for (int k = 1; k <= k_want && (p + k) < apos; ++k) {
                        const int32_t ct = h[(size_t)(p + k)];
                        if (ct < 0) break;
                        out_cont[cnt++] = ct;
                    }
                    if (cnt > 0) return cnt;
                }
            }
        }
        return 0;
    }

    static void get_unemb_q8(const InkCudaModel& m, const int8_t*& w_i8, const float*& w_sc) {
        const auto& ui = m.tinfo("model.llm.unembed.weight");
        if (ui.fmt == FMT_Q8ROW) {
            w_sc = reinterpret_cast<const float*>(m.d_trunk + ui.off);
            w_i8 = reinterpret_cast<const int8_t*>(m.d_trunk + ui.off + (size_t)ui.rows() * sizeof(float));
        } else {
            w_sc = m.unembed_screen.sc;
            w_i8 = m.unembed_screen.q8;
        }
    }

    void ingest_slot_seq(InkDSpark& ds, int slot, int p0, int M) {
        if (!ready || !ds.ready || slot < 0 || slot >= max_slots || M <= 0) return;
        if (ds.df2) { ds.df2->ingest_seq(ds.r.d_taps, slot, p0, M); return; }   // DFlash 2 (STRATA_INK_DRAFTER=dflash2)
        auto& r = ds.r;
        auto& m = ds.m;
        const float* fw = (const float*)m.trunk_ptr("dspark.fuse_weight");
        for (int off = 0; off < M; off += m_cap) {
            const int cnt = std::min(m_cap, M - off);
            for (int i = 0; i < cnt; ++i) h_meta[i] = p0 + off + i;
            INK_CK(cudaMemcpyAsync(d_meta, h_meta, (size_t)cnt * 4, cudaMemcpyHostToDevice, r.st));
            float* t0 = r.d_taps[0] + (size_t)off * kInkHidden;
            float* t1 = r.d_taps[1] + (size_t)off * kInkHidden;
            float* t2 = r.d_taps[2] + (size_t)off * kInkHidden;
            float* t3 = r.d_taps[3] + (size_t)off * kInkHidden;
            for (int il = 0; il < 3; ++il) {
                const std::string p = "dspark.layers." + std::to_string(il);
                k_ms_mix_taps_indexed<<<cnt, 256, 0, r.st>>>(
                    t0, t1, t2, t3, nullptr, fw, il,
                    (const float*)m.trunk_ptr(p + ".dflare_norm.weight"), ds.d_ctx_tmp, cnt);
                auto it_kv = ds.bf16_cache.find(p + ".kv_ctx_fused");
                if (it_kv != ds.bf16_cache.end()) {
                    ds.dspark_gemm_bf16(it_kv->second, ds.d_ctx_tmp, ds.d_kv_ctx_fused, cnt, 2 * kInkKVDim, kInkHidden, false);
                    k_dspark_split_kv<<<dim3((2 * kInkKVDim + 255) / 256, cnt), 256, 0, r.st>>>(ds.d_kv_ctx_fused, ds.d_kc_tmp, ds.d_vc_tmp, cnt);
                } else {
                    r.linear(p + ".self_attn.k_proj_ctx.weight", ds.d_ctx_tmp, ds.d_kc_tmp, cnt);
                    r.linear(p + ".self_attn.v_proj_ctx.weight", ds.d_ctx_tmp, ds.d_vc_tmp, cnt);
                }
                dim3 g(kInkKVHeads, cnt);
                k_dspark_rmsnorm_rope<<<g, 128, 0, r.st>>>(
                    ds.d_kc_tmp, (const float*)m.trunk_ptr(p + ".self_attn.k_norm_ctx.weight"),
                    d_meta, m.man.c.dspark_rope_theta, cnt, kInkKVHeads);
                InkKvVisitDesc kd = kv_desc[il];
                kd.is_swa = true;
                k_ink_kv_write<<<g, 128, 0, r.st>>>(ds.d_kc_tmp, ds.d_vc_tmp, nullptr, slot, d_meta, kd, cnt);
            }
            k_ms_mix_frnn_indexed<<<cnt, 256, 0, r.st>>>(
                t0, t1, t2, t3, nullptr, nullptr, slot, d_meta, fw,
                (const float*)m.trunk_ptr("dspark.markov_head.dflare_norm.weight"), d_frnn, cnt);
            INK_CK(cudaStreamSynchronize(r.st));
        }
        slot_synced_pos[slot] = p0 + M;
    }

    void ingest_taps_batched(InkDSpark& ds, const int* slots_b, const int* p0_b, const int* n_acc_b, int B, int K_slot) {
        if (!ready || !ds.ready || B <= 0) return;
        auto& r = ds.r;
        auto& m = ds.m;
        int n_tot = 0;
        int32_t* h_src = h_meta;
        int32_t* h_sl  = h_meta + m_cap;
        int32_t* h_pos = h_meta + 2 * m_cap;
        for (int b = 0; b < B; ++b) {
            const int na = n_acc_b ? n_acc_b[b] : 1;
            const int sl = slots_b[b];
            for (int j = 0; j < na && n_tot < m_cap; ++j) {
                h_src[n_tot] = b * K_slot + j;
                h_sl[n_tot]  = sl;
                h_pos[n_tot] = p0_b[b] + j;
                ++n_tot;
            }
            if (sl >= 0 && sl < max_slots) slot_synced_pos[sl] = p0_b[b] + na;
        }
        if (n_tot <= 0) return;
        if (ds.df2) { ds.df2->ingest(r.d_taps, h_src, h_sl, h_pos, n_tot); return; }   // DFlash 2 (STRATA_INK_DRAFTER=dflash2)
        int32_t* d_src = d_meta;
        int32_t* d_sl  = d_meta + m_cap;
        int32_t* d_pos = d_meta + 2 * m_cap;
        INK_CK(cudaMemcpyAsync(d_src, h_src, (size_t)n_tot * 4, cudaMemcpyHostToDevice, r.st));
        INK_CK(cudaMemcpyAsync(d_sl,  h_sl,  (size_t)n_tot * 4, cudaMemcpyHostToDevice, r.st));
        INK_CK(cudaMemcpyAsync(d_pos, h_pos, (size_t)n_tot * 4, cudaMemcpyHostToDevice, r.st));

        const float* fw = (const float*)m.trunk_ptr("dspark.fuse_weight");
        for (int il = 0; il < 3; ++il) {
            const std::string p = "dspark.layers." + std::to_string(il);
            k_ms_mix_taps_indexed<<<n_tot, 256, 0, r.st>>>(
                r.d_taps[0], r.d_taps[1], r.d_taps[2], r.d_taps[3], d_src, fw, il,
                (const float*)m.trunk_ptr(p + ".dflare_norm.weight"), ds.d_ctx_tmp, n_tot);
            auto it_kv = ds.bf16_cache.find(p + ".kv_ctx_fused");
            if (it_kv != ds.bf16_cache.end()) {
                ds.dspark_gemm_bf16(it_kv->second, ds.d_ctx_tmp, ds.d_kv_ctx_fused, n_tot, 2 * kInkKVDim, kInkHidden, false);
                k_dspark_split_kv<<<dim3((2 * kInkKVDim + 255) / 256, n_tot), 256, 0, r.st>>>(ds.d_kv_ctx_fused, ds.d_kc_tmp, ds.d_vc_tmp, n_tot);
            } else {
                r.linear(p + ".self_attn.k_proj_ctx.weight", ds.d_ctx_tmp, ds.d_kc_tmp, n_tot);
                r.linear(p + ".self_attn.v_proj_ctx.weight", ds.d_ctx_tmp, ds.d_vc_tmp, n_tot);
            }
            dim3 g(kInkKVHeads, n_tot);
            k_dspark_rmsnorm_rope<<<g, 128, 0, r.st>>>(
                ds.d_kc_tmp, (const float*)m.trunk_ptr(p + ".self_attn.k_norm_ctx.weight"),
                d_pos, m.man.c.dspark_rope_theta, n_tot, kInkKVHeads);
            InkKvVisitDesc kd = kv_desc[il];
            kd.is_swa = true;
            k_ink_kv_write<<<g, 128, 0, r.st>>>(ds.d_kc_tmp, ds.d_vc_tmp, d_sl, 0, d_pos, kd, n_tot);
        }
        k_ms_mix_frnn_indexed<<<n_tot, 256, 0, r.st>>>(
            r.d_taps[0], r.d_taps[1], r.d_taps[2], r.d_taps[3], d_src, d_sl, 0, d_pos, fw,
            (const float*)m.trunk_ptr("dspark.markov_head.dflare_norm.weight"), d_frnn, n_tot);
    }

    void draft_batched(InkDSpark& ds, int B, int K_draft,
                       const int* slots_b, const int32_t* anchor_toks, const int* apos_b,
                       int32_t* h_out_draft) {
        if (ds.df2) {   // DFlash 2 (STRATA_INK_DRAFTER=dflash2): the first K_draft of its 7 chain tokens per slot
            std::vector<int32_t> ids((size_t)std::max(1, B) * 7);
            std::vector<float> conf((size_t)std::max(1, B) * 7);
            const bool ok = B > 0 && B <= ds.df2->max_slots && ds.df2->draft(B, slots_b, anchor_toks, apos_b, ids.data(), conf.data());
            for (int b = 0; b < B; ++b)
                for (int k = 0; k < K_draft; ++k) h_out_draft[b * K_draft + k] = ok ? ids[(size_t)b * 7 + k] : anchor_toks[b];
            return;
        }
        auto& r = ds.r;
        auto& m = ds.m;
        const int N_blk = B * 8;
        const int M_d   = B * K_draft;

        for (int b = 0; b < B; ++b) {
            h_meta[b]       = slots_b[b];
            h_meta[64 + b]  = apos_b[b];
            h_meta[128 + b] = anchor_toks[b];
            for (int k = 0; k < 8; ++k) h_meta[256 + b * 8 + k] = apos_b[b] + k;
        }
        INK_CK(cudaMemcpyAsync(d_meta, h_meta, (size_t)(256 + N_blk) * 4, cudaMemcpyHostToDevice, r.st));
        const int32_t* d_sl_b  = d_meta;
        const int32_t* d_ap_b  = d_meta + 64;
        const int32_t* d_anc_b = d_meta + 128;
        const int32_t* d_bpos  = d_meta + 256;

        const auto& ei = m.tinfo("model.llm.embed.weight");
        k_ms_init_block8<<<N_blk, 256, 0, r.st>>>(
            m.d_trunk + ei.off, ei.fmt,
            (const float*)m.trunk_ptr("model.llm.embed_norm.weight"),
            (const float*)m.trunk_ptr("dspark.mask_embed"),
            d_anc_b, d_cur, B);

        for (int il = 0; il < 3; ++il) {
            const std::string p = "dspark.layers." + std::to_string(il);
            k_ink_rms_norm<<<N_blk, 256, 0, r.st>>>(d_cur, (const float*)m.trunk_ptr(p + ".input_layernorm.weight"), d_h, nullptr, N_blk, kInkHidden);
            auto it_qkv = ds.bf16_cache.find(p + ".qkv_fused");
            if (it_qkv != ds.bf16_cache.end()) {
                ds.dspark_gemm_bf16(it_qkv->second, d_h, d_qkv, N_blk, kInkQDim + 2 * kInkKVDim, kInkHidden, false);
                k_dspark_split_qkv<<<dim3((kInkQDim + 2 * kInkKVDim + 255) / 256, N_blk), 256, 0, r.st>>>(d_qkv, d_q, d_k, d_v, N_blk);
            } else {
                ds.dspark_linear_b8(p + ".self_attn.q_proj.weight", d_h, d_q, N_blk, false);
                ds.dspark_linear_b8(p + ".self_attn.k_proj.weight", d_h, d_k, N_blk, true);
                ds.dspark_linear_b8(p + ".self_attn.v_proj.weight", d_h, d_v, N_blk, true);
            }
            k_dspark_rmsnorm_rope<<<dim3(kInkHeads, N_blk), 128, 0, r.st>>>(
                d_q, (const float*)m.trunk_ptr(p + ".self_attn.q_norm.weight"), d_bpos, m.man.c.dspark_rope_theta, N_blk, kInkHeads);
            k_dspark_rmsnorm_rope<<<dim3(kInkKVHeads, N_blk), 128, 0, r.st>>>(
                d_k, (const float*)m.trunk_ptr(p + ".self_attn.k_norm.weight"), d_bpos, m.man.c.dspark_rope_theta, N_blk, kInkKVHeads);
            k_ms_dspark_block_attn<<<dim3(kInkHeads, N_blk), 128, 0, r.st>>>(
                d_q, d_k, d_v, kv_desc[il], d_sl_b, d_ap_b, m.man.c.dspark_swa_win, B, d_attn_out);
            ds.dspark_linear_b8(p + ".self_attn.o_proj.weight", d_attn_out, d_proj, N_blk, false);
            k_ink_add<<<((size_t)N_blk * kInkHidden + 255) / 256, 256, 0, r.st>>>(d_cur, d_cur, d_proj, (size_t)N_blk * kInkHidden);

            k_ink_rms_norm<<<N_blk, 256, 0, r.st>>>(d_cur, (const float*)m.trunk_ptr(p + ".post_attention_layernorm.weight"), d_h, nullptr, N_blk, kInkHidden);
            auto it_gu = ds.bf16_cache.find(p + ".gu_fused");
            if (it_gu != ds.bf16_cache.end()) {
                ds.dspark_gemm_bf16(it_gu->second, d_h, d_gu, N_blk, 2 * kInkDsparkInter, kInkHidden, false);
                k_dspark_swiglu_fused<<<dim3((kInkDsparkInter + 255) / 256, N_blk), 256, 0, r.st>>>(d_gu, d_act, N_blk);
            } else {
                ds.dspark_linear_b8(p + ".mlp.gate_proj.weight", d_h, d_gu, N_blk, false);
                ds.dspark_linear_b8(p + ".mlp.up_proj.weight", d_h, d_gu + (size_t)N_blk * kInkDsparkInter, N_blk, true);
                k_dspark_swiglu<<<((size_t)N_blk * kInkDsparkInter + 255) / 256, 256, 0, r.st>>>(
                    d_gu, d_gu + (size_t)N_blk * kInkDsparkInter, d_act, (size_t)N_blk * kInkDsparkInter);
            }
            ds.dspark_linear_b8(p + ".mlp.down_proj.weight", d_act, d_proj, N_blk, false);
            k_ink_add<<<((size_t)N_blk * kInkHidden + 255) / 256, 256, 0, r.st>>>(d_cur, d_cur, d_proj, (size_t)N_blk * kInkHidden);
        }
        k_ink_rms_norm<<<N_blk, 256, 0, r.st>>>(d_cur, (const float*)m.trunk_ptr("dspark.norm.weight"), d_cur, nullptr, N_blk, kInkHidden);

        const int8_t* w_i8 = nullptr;
        const float*  w_sc = nullptr;
        get_unemb_q8(m, w_i8, w_sc);

        k_ms_gather_draft_h<<<M_d, 256, 0, r.st>>>(d_cur, d_draft_h, B, K_draft);
        k_ms_quant_h_q8<<<M_d, 256, 0, r.st>>>(d_draft_h, d_h_q8, d_h_sc, M_d);
        const int32_t alpha_i = 1, beta_i = 0;
        INK_CUBLAS_CK(cublasGemmEx(
            r.cublas, CUBLAS_OP_T, CUBLAS_OP_N,
            kBspecV8, M_d, kInkHidden, &alpha_i,
            w_i8, CUDA_R_8I, kInkHidden,
            d_h_q8, CUDA_R_8I, kInkHidden,
            &beta_i, d_base_i32, CUDA_R_32I, kBspecV8,
            CUBLAS_COMPUTE_32I, CUBLAS_GEMM_DEFAULT_TENSOR_OP));
        k_ms_top_i32_s1<<<dim3(64, M_d), 256, 0, r.st>>>(d_base_i32, w_sc, d_h_sc, kBspecV8, d_top_tmp_i, d_top_tmp_v, M_d);
        k_ms_top_i32_s2<<<M_d, 256, 0, r.st>>>(d_top_tmp_i, d_top_tmp_v, kBspecKcand, d_cand_ids, nullptr, m.man.c.vocab_unpadded, kBspecV8, M_d);

        INK_CK(cudaMemsetAsync(d_rnn_st, 0, (size_t)B * kInkMarkovRank * 4, r.st));
        INK_CK(cudaMemcpyAsync(d_prev_tok, d_anc_b, (size_t)B * 4, cudaMemcpyDeviceToDevice, r.st));
        const auto& ti_mw1 = m.tinfo("dspark.markov_head.markov_w1.weight");
        const auto& ti_rpb = m.tinfo("dspark.markov_head.rnn_proj.bias");
        const auto& ti_hob = m.tinfo("dspark.markov_head.hidden_out.bias");

        for (int s = 0; s < K_draft; ++s) {
            k_ms_markov_prep<<<B, 256, 0, r.st>>>(
                d_prev_tok, d_rnn_st, d_cur, d_frnn, d_sl_b, d_ap_b,
                m.d_trunk + ti_mw1.off, ti_mw1.fmt, d_rnn_joint, B, s);
            ds.dspark_linear_b8("dspark.markov_head.rnn_proj.weight", d_rnn_joint, d_rnn_proj, B);
            k_dspark_markov_gru_upd_batch<<<B, 256, 0, r.st>>>(
                d_rnn_proj, (const float*)(m.d_trunk + ti_rpb.off), d_rnn_st, d_rnn_st_n, d_rnn_lat, B);
            std::swap(d_rnn_st, d_rnn_st_n);
            ds.dspark_linear_b8("dspark.markov_head.hidden_out.weight", d_rnn_lat, d_corr, B);
            k_ms_add_bias_and_h<<<dim3((kInkHidden + 255) / 256, B), 256, 0, r.st>>>(
                d_corr, (const float*)(m.d_trunk + ti_hob.off), d_cur, B, s);
            k_ms_quant_h_q8<<<B, 256, 0, r.st>>>(d_corr, d_h_q8, d_h_sc, B);
            k_ms_score_cand_dp4a<<<dim3(kBspecKcand / 8, B), 256, 0, r.st>>>(
                w_i8, w_sc, d_h_q8, d_h_sc,
                d_cand_ids, kBspecKcand, K_draft, s, B, d_cand_sc);
            k_ms_cand_argmax<<<B, 128, 0, r.st>>>(
                d_cand_sc, d_cand_ids, kBspecKcand, K_draft, s, B, d_prev_tok, d_draft_out);
        }
        INK_CK(cudaMemcpyAsync(h_draft, d_draft_out, (size_t)M_d * 4, cudaMemcpyDeviceToHost, r.st));
        INK_CK(cudaStreamSynchronize(r.st));
        std::memcpy(h_out_draft, h_draft, (size_t)M_d * 4);
    }

    void head_argmax_fast(InkCudaRunner& r, int M, int32_t* out_ids) {
        auto& m = r.m;
        const auto& ui = m.tinfo("model.llm.unembed.weight");
        const bool is_q8row = (ui.fmt == FMT_Q8ROW);
        if ((is_q8row || (m.unembed_screen.ready && (ui.fmt == FMT_BF16 || ui.fmt == FMT_S6_K))) && M <= m_cap) {
            constexpr int kEvalK = 32;
            const int8_t* w_i8 = nullptr;
            const float*  w_sc = nullptr;
            get_unemb_q8(m, w_i8, w_sc);
            k_ms_quant_h_q8<<<M, 256, 0, r.st>>>(r.d_h_norm, d_h_q8, d_h_sc, M);
            const int32_t alpha_i = 1, beta_i = 0;
            INK_CUBLAS_CK(cublasGemmEx(
                r.cublas, CUBLAS_OP_T, CUBLAS_OP_N,
                kBspecV8, M, kInkHidden, &alpha_i,
                w_i8, CUDA_R_8I, kInkHidden,
                d_h_q8, CUDA_R_8I, kInkHidden,
                &beta_i, d_base_i32, CUDA_R_32I, kBspecV8,
                CUBLAS_COMPUTE_32I, CUBLAS_GEMM_DEFAULT_TENSOR_OP));
            k_ms_top_i32_s1<<<dim3(64, M), 256, 0, r.st>>>(d_base_i32, w_sc, d_h_sc, kBspecV8, d_top_tmp_i, d_top_tmp_v, M);
            k_ms_top_i32_s2<<<M, 256, 0, r.st>>>(d_top_tmp_i, d_top_tmp_v, kEvalK, d_cand_ids, d_cand_cnt, m.man.c.vocab_unpadded, kBspecV8, M);
            if (is_q8row) {
                k_ink_screen_eval_q8row<<<M, 256, 0, r.st>>>(
                    m.d_trunk + ui.off, (int)ui.rows(), r.d_h_norm, d_cand_ids, d_cand_cnt, kEvalK, r.d_argmax, M);
            } else if (ui.fmt == FMT_BF16) {
                k_ink_screen_eval_bf16<<<M, 256, 0, r.st>>>(
                    (const uint16_t*)(m.d_trunk + ui.off), r.d_h_norm, d_cand_ids, d_cand_cnt, kEvalK, r.d_argmax, M);
            } else {
                k_ink_screen_eval_s6k<<<M, 256, 0, r.st>>>(
                    m.d_trunk + ui.off, r.d_h_norm, d_cand_ids, d_cand_cnt, kEvalK, r.d_argmax, M);
            }
            INK_CK(cudaMemcpyAsync(h_argmax, r.d_argmax, (size_t)M * 4, cudaMemcpyDeviceToHost, r.st));
            INK_CK(cudaStreamSynchronize(r.st));
            std::memcpy(out_ids, h_argmax, (size_t)M * 4);
            return;
        }
        r.head_argmax_exact(M, out_ids);
    }

    void free_all() {
        auto fr = [](auto*& p) { if (p) { cudaFree(p); p = nullptr; } };
        fr(d_stage_k); fr(d_stage_v); fr(d_stage_sconv);
        for (int il = 0; il < 3; ++il) {
            fr(d_kv_k[il]); fr(d_kv_v[il]); fr(d_kv_ksc[il]); fr(d_kv_vsc[il]);
        }
        fr(d_frnn); fr(d_cur); fr(d_h); fr(d_qkv); fr(d_q); fr(d_k); fr(d_v);
        fr(d_attn_out); fr(d_proj); fr(d_gu); fr(d_act); fr(d_draft_h);
        fr(d_h_q8); fr(d_h_sc); fr(d_base_i32); fr(d_top_tmp_i); fr(d_top_tmp_v);
        fr(d_cand_ids); fr(d_cand_cnt); fr(d_rnn_st); fr(d_rnn_st_n);
        fr(d_rnn_joint); fr(d_rnn_proj); fr(d_rnn_lat); fr(d_corr); fr(d_cand_sc);
        fr(d_prev_tok); fr(d_draft_out); fr(d_meta);
        if (h_meta) { cudaFreeHost(h_meta); h_meta = nullptr; }
        if (h_draft) { cudaFreeHost(h_draft); h_draft = nullptr; }
        if (h_argmax) { cudaFreeHost(h_argmax); h_argmax = nullptr; }
        ready = false;
    }
};

}  // namespace strata::ink::rtx5090
