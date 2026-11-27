// ink_metal_dflash2.metal - Native Apple Silicon Metal Kernels for DFlash 2 Block-Diffusion Drafter
// Exact port of cuda/ink_dflash2_kernels.cuh:
//   - 5 non-causal SWA-2048 layers (32Q / 8KV x 128, SwiGLU 6912, RoPE theta=1e4)
//   - 2-tap group-16 dynamic causal convs (conv_pre + conv_post_add) along 8-row draft block
//   - 4 target taps [5, 23, 41, final_norm] -> fc [2304, 9216] -> hidden_norm -> per-layer K/V into 2112 ring
//   - Single-DRAM-pass Q8_ROW unembed screen -> exact Top-16 + exact full-vocab softcapped tail (m_t, l_t)
//   - Sequential 7-step rank-256 bilinear candidate selector with exact calibrated confidence
// MIT License - Copyright (c) 2026 Niko1221 and the Strata contributors
#include <metal_stdlib>
using namespace metal;

#define DF2_H 2304
#define DF2_Q 4096
#define DF2_KV 1024
#define DF2_FF 6912
#define DF2_DYN 576
#define DF2_RANK 256
#define DF2_TOPK 16
#define DF2_WIN 2048
#define DF2_RING 2112
#define DF2_CAP 30.0f
#define DF2_EPS 1e-6f

static inline float df2_bf16f(ushort b) { return as_type<float>(uint(b) << 16); }

static inline float4 df2_bf16x4(ushort4 b) {
    return float4(as_type<float>(uint(b.x) << 16), as_type<float>(uint(b.y) << 16),
                  as_type<float>(uint(b.z) << 16), as_type<float>(uint(b.w) << 16));
}

static inline float df2_softcap(float x, float cap) {
    return cap * tanh(clamp(x / cap, -15.0f, 15.0f));
}

static inline bool df2_better(float va, int ia, float vb, int ib) {
    return (va > vb) || (va == vb && ia < ib);
}

static inline float df2_tg_sum(float v, threadgroup float* sh, uint tid, uint nthr) {
    v = simd_sum(v);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if ((tid & 31u) == 0u) sh[tid >> 5] = v;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float r = 0.0f;
    const uint nw = (nthr + 31u) >> 5;
    for (uint i = 0; i < nw; ++i) r += sh[i];
    return r;
}

// ============================================================================
// 1. Init-Time Weight Quantizers (BF16 / S6_K -> Q8_ROW [N, K] char + float[N])
// ============================================================================
kernel void df2_bf16_to_q8row(
    device const ushort* src [[buffer(0)]],
    device char* dst_q       [[buffer(1)]],
    device float* dst_sc     [[buffer(2)]],
    constant int& N          [[buffer(3)]],
    constant int& K          [[buffer(4)]],
    uint r   [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]]) {
    if (int(r) >= N) return;
    threadgroup float sh[8];
    device const ushort* rp = src + (size_t)r * K;
    float mx = 0.0f;
    for (int k = int(tid); k < K; k += 256) mx = max(mx, fabs(df2_bf16f(rp[k])));
    mx = simd_max(mx);
    if ((tid & 31u) == 0u) sh[tid >> 5] = mx;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float gmx = sh[0];
    for (int i = 1; i < 8; ++i) gmx = max(gmx, sh[i]);
    const float sc = (gmx > 0.0f) ? (gmx / 127.0f) : 1.0f;
    const float inv = (gmx > 0.0f) ? (127.0f / gmx) : 0.0f;
    if (tid == 0u) dst_sc[r] = sc;
    device char* qp = dst_q + (size_t)r * K;
    for (int k = int(tid); k < K; k += 256) {
        qp[k] = char(clamp(int(rint(df2_bf16f(rp[k]) * inv)), -127, 127));
    }
}

// Quantize target unembed_w (Q8_ROW layout [V_pad floats][V_pad * 2304 chars]) -> split dst_q/dst_sc for V rows
kernel void df2_copy_q8row_unembed(
    device const uchar* W [[buffer(0)]],
    device char* dst_q    [[buffer(1)]],
    device float* dst_sc  [[buffer(2)]],
    constant int& V       [[buffer(3)]],
    constant int& V_pad   [[buffer(4)]],
    constant int& K       [[buffer(5)]],
    uint r   [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]]) {
    if (int(r) >= V) return;
    if (tid == 0u) dst_sc[r] = ((device const float*)W)[r];
    device const char* src = (device const char*)(W + (size_t)V_pad * 4ul) + (size_t)r * K;
    device char* dst = dst_q + (size_t)r * K;
    for (int k = int(tid); k < K; k += 256) dst[k] = src[k];
}

// Quantize target S6_K unembed [V, 2304] -> split Q8_ROW dst_q/dst_sc
kernel void df2_sk6_to_q8row(
    device const uchar* W [[buffer(0)]],
    device char* dst_q    [[buffer(1)]],
    device float* dst_sc  [[buffer(2)]],
    constant int& V       [[buffer(3)]],
    constant int& K       [[buffer(4)]],
    uint r   [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]]) {
    if (int(r) >= V) return;
    threadgroup float sh[8];
    const int nsb = K >> 8;
    const size_t rowb = (size_t)nsb * 198ul;
    device const uchar* rp = W + (size_t)r * rowb;
    float mx = 0.0f;
    for (int col = int(tid); col < K; col += 256) {
        const int sbi = col >> 8, j = col & 255, g = j >> 6;
        device const uchar* blk = rp + (size_t)sbi * 198ul;
        const ushort d16 = ushort(blk[0]) | (ushort(blk[1]) << 8);
        const float eff = float(as_type<half>(d16)) * (float(blk[2 + g]) * (1.0f / 255.0f));
        device const uchar* pl = blk + 6;
        const uint lo = (uint(pl[j >> 1]) >> ((j & 1) * 4)) & 15u;
        device const uchar* ph = pl + 128;   // (the pointer is formed first: Apple7 / Apple8 cannot select `pl[128 + (j >> 2)]`, #6608)
        const uint hi = (uint(ph[j >> 2]) >> ((j & 3) * 2)) & 3u;
        const float v = (float(int(lo | (hi << 4))) - 31.5f) * eff;
        mx = max(mx, fabs(v));
    }
    mx = simd_max(mx);
    if ((tid & 31u) == 0u) sh[tid >> 5] = mx;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float gmx = sh[0];
    for (int i = 1; i < 8; ++i) gmx = max(gmx, sh[i]);
    const float sc = (gmx > 0.0f) ? (gmx / 127.0f) : 1.0f;
    const float inv = (gmx > 0.0f) ? (127.0f / gmx) : 0.0f;
    if (tid == 0u) dst_sc[r] = sc;
    device char* qp = dst_q + (size_t)r * K;
    for (int col = int(tid); col < K; col += 256) {
        const int sbi = col >> 8, j = col & 255, g = j >> 6;
        device const uchar* blk = rp + (size_t)sbi * 198ul;
        const ushort d16 = ushort(blk[0]) | (ushort(blk[1]) << 8);
        const float eff = float(as_type<half>(d16)) * (float(blk[2 + g]) * (1.0f / 255.0f));
        device const uchar* pl = blk + 6;
        const uint lo = (uint(pl[j >> 1]) >> ((j & 1) * 4)) & 15u;
        device const uchar* ph = pl + 128;   // (the pointer is formed first: Apple7 / Apple8 cannot select `pl[128 + (j >> 2)]`, #6608)
        const uint hi = (uint(ph[j >> 2]) >> ((j & 3) * 2)) & 3u;
        const float v = (float(int(lo | (hi << 4))) - 31.5f) * eff;
        qp[col] = char(clamp(int(rint(v * inv)), -127, 127));
    }
}

// ============================================================================
// 2. Context Ingest: 4 Target Taps -> fc [2304, 9216] -> hidden_norm -> 5-Layer Ring K/V
// ============================================================================
// Pack 4 target taps (b_taps4[0..2, m, D] + b_h_norm[m, D]) for R rows -> out [R, 9216] F32
// grid (R, 1, 1), 256 threads
kernel void df2_pack_taps4(
    device const float* taps3  [[buffer(0)]],  // [3, M_stride_floats]
    device const float* h_norm [[buffer(1)]],  // [M, 2304]
    device float* out9216      [[buffer(2)]],  // [R, 9216]
    constant int& M_stride     [[buffer(3)]],  // M * 2304
    constant int& row0         [[buffer(4)]],  // starting token index in batch
    uint r   [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]]) {
    const int m = row0 + int(r);
    device float* dst = out9216 + (size_t)r * 9216ul;
    for (uint d = tid; d < DF2_H; d += 256u) {
        dst[0 * DF2_H + d] = taps3[0ul * (size_t)M_stride + (size_t)m * DF2_H + d];
        dst[1 * DF2_H + d] = taps3[1ul * (size_t)M_stride + (size_t)m * DF2_H + d];
        dst[2 * DF2_H + d] = taps3[2ul * (size_t)M_stride + (size_t)m * DF2_H + d];
        dst[3 * DF2_H + d] = h_norm[(size_t)m * DF2_H + d];
    }
}

// Store projected K/V [R, 2048] (first 1024=K, next 1024=V) into per-slot ring[slot][layer][pos % 2112][1024]
// Applies per-head RMSNorm(k_norm) + NeoX half-split RoPE(theta=1e4) to K!
// grid (R, 8, 1), 64 threads (each thread handles pair (d, d+64)!)
kernel void df2_ctx_kv_store(
    device const float* kv_in [[buffer(0)]],  // [R, 2048]
    device const float* knorm [[buffer(1)]],  // [128]
    device half* ring_k       [[buffer(2)]],  // [max_slots, 2112, 1024]
    device half* ring_v       [[buffer(3)]],  // [max_slots, 2112, 1024]
    device const int* slots   [[buffer(4)]],  // [R]
    device const int* pos_buf [[buffer(5)]],  // [R]
    constant int& row0        [[buffer(6)]],
    constant float& theta     [[buffer(7)]],
    uint2 tgp [[threadgroup_position_in_grid]],
    uint tid  [[thread_index_in_threadgroup]]) {
    threadgroup float sh[2];
    const int r = int(tgp.x), h = int(tgp.y);
    const int m = row0 + r;
    const int slot = slots[m];
    const int pos = pos_buf[m];
    const int cell = pos % DF2_RING;

    device const float* kp = kv_in + (size_t)r * 2048ul + (size_t)h * 128ul;
    device const float* vp = kv_in + (size_t)r * 2048ul + 1024ul + (size_t)h * 128ul;
    const float k0 = kp[tid];
    const float k1 = kp[tid + 64u];
    const float ss = k0 * k0 + k1 * k1;
    const float inv = rsqrt(df2_tg_sum(ss, sh, tid, 64u) * (1.0f / 128.0f) + DF2_EPS);
    const float x0 = k0 * inv * knorm[tid];
    const float x1 = k1 * inv * knorm[tid + 64u];
    const float ang = float(pos) * pow(theta, -float(2u * tid) * (1.0f / 128.0f));
    const float cs = cos(ang), sn = sin(ang);

    const size_t dst_base = ((size_t)slot * DF2_RING + (size_t)cell) * DF2_KV + (size_t)h * 128ul;
    ring_k[dst_base + tid]       = half(x0 * cs - x1 * sn);
    ring_k[dst_base + tid + 64u] = half(x1 * cs + x0 * sn);
    ring_v[dst_base + tid]       = half(vp[tid]);
    ring_v[dst_base + tid + 64u] = half(vp[tid + 64u]);
}

// ============================================================================
// 3. Block-8 Forward Ops (Init, RMSNorm, 2-Tap Group-16 Dynamic Convs, Q8/BF16 GEMV, QK RoPE, SWA-2048 Attn, SwiGLU)
// ============================================================================
kernel void df2_init_block_bf16(
    device const ushort* embed_w [[buffer(0)]],
    device const float* norm_w   [[buffer(1)]],
    device const float* mask_row [[buffer(2)]],
    device float* x_out          [[buffer(3)]],  // [8, 2304]
    constant int& anchor_tok     [[buffer(4)]],
    uint r   [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]]) {
    threadgroup float sh[8];
    device float* dst = x_out + (size_t)r * DF2_H;
    if (r == 0u) {
        device const ushort* row = embed_w + (size_t)anchor_tok * DF2_H;
        float ss = 0.0f;
        for (uint d = tid; d < DF2_H; d += 256u) {
            const float v = df2_bf16f(row[d]);
            ss += v * v;
        }
        const float inv = rsqrt(df2_tg_sum(ss, sh, tid, 256u) * (1.0f / float(DF2_H)) + DF2_EPS);
        for (uint d = tid; d < DF2_H; d += 256u) dst[d] = df2_bf16f(row[d]) * inv * norm_w[d];
    } else {
        for (uint d = tid; d < DF2_H; d += 256u) dst[d] = mask_row[d];
    }
}

kernel void df2_init_block_q8row(
    device const uchar* embed_w  [[buffer(0)]],
    device const float* norm_w   [[buffer(1)]],
    device const float* mask_row [[buffer(2)]],
    device float* x_out          [[buffer(3)]],  // [8, 2304]
    constant int& anchor_tok     [[buffer(4)]],
    constant int& V_pad          [[buffer(5)]],
    uint r   [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]]) {
    threadgroup float sh[8];
    device float* dst = x_out + (size_t)r * DF2_H;
    if (r == 0u) {
        const float sc = ((device const float*)embed_w)[anchor_tok];
        device const char* row = (device const char*)(embed_w + (size_t)V_pad * 4ul) + (size_t)anchor_tok * DF2_H;
        float ss = 0.0f;
        for (uint d = tid; d < DF2_H; d += 256u) {
            const float v = float(row[d]) * sc;
            ss += v * v;
        }
        const float inv = rsqrt(df2_tg_sum(ss, sh, tid, 256u) * (1.0f / float(DF2_H)) + DF2_EPS);
        for (uint d = tid; d < DF2_H; d += 256u) dst[d] = float(row[d]) * sc * inv * norm_w[d];
    } else {
        for (uint d = tid; d < DF2_H; d += 256u) dst[d] = mask_row[d];
    }
}

// RMSNorm [R, 2304]: grid (R, 1, 1), 256 threads
kernel void df2_rms_norm(
    device const float* x_in [[buffer(0)]],
    device const float* w    [[buffer(1)]],
    device float* y_out      [[buffer(2)]],
    uint r   [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]]) {
    threadgroup float sh[8];
    device const float* xp = x_in + (size_t)r * DF2_H;
    device float* yp = y_out + (size_t)r * DF2_H;
    float ss = 0.0f;
    for (uint d = tid; d < DF2_H; d += 256u) { const float v = xp[d]; ss += v * v; }
    const float inv = rsqrt(df2_tg_sum(ss, sh, tid, 256u) * (1.0f / float(DF2_H)) + DF2_EPS);
    for (uint d = tid; d < DF2_H; d += 256u) yp[d] = xp[d] * inv * w[d];
}

// 2-tap group-16 dynamic causal conv pre-sublayer: grid (8, 9, 1), 256 threads
kernel void df2_conv_pre(
    device const float* xn   [[buffer(0)]],  // [8, 2304]
    device const float* dyn  [[buffer(1)]],  // [8, 576]
    device const float* base [[buffer(2)]],  // [4, 2304]
    device float* c_out      [[buffer(3)]],  // [8, 2304]
    uint2 tgp [[threadgroup_position_in_grid]],
    uint tid  [[thread_index_in_threadgroup]]) {
    const int r = int(tgp.x);
    const int c = int(tgp.y) * 256 + int(tid);
    if (c >= DF2_H) return;
    const int t = r & 7;
    const int g = c >> 4;
    const float w0 = base[0 * DF2_H + c] + dyn[(size_t)r * DF2_DYN + g];
    float v = w0 * xn[(size_t)r * DF2_H + c];
    if (t > 0) {
        const float w1 = base[1 * DF2_H + c] + dyn[(size_t)r * DF2_DYN + 144 + g];
        v = fma(w1, xn[(size_t)(r - 1) * DF2_H + c], v);
    }
    c_out[(size_t)r * DF2_H + c] = v;
}

// 2-tap group-16 dynamic causal conv post-sublayer + residual add: grid (8, 9, 1), 256 threads
kernel void df2_conv_post_add(
    device float* x_io       [[buffer(0)]],  // [8, 2304] in-place
    device const float* z_in [[buffer(1)]],  // [8, 2304]
    device const float* dyn  [[buffer(2)]],  // [8, 576]
    device const float* base [[buffer(3)]],  // [4, 2304]
    uint2 tgp [[threadgroup_position_in_grid]],
    uint tid  [[thread_index_in_threadgroup]]) {
    const int r = int(tgp.x);
    const int c = int(tgp.y) * 256 + int(tid);
    if (c >= DF2_H) return;
    const int t = r & 7;
    const int g = c >> 4;
    const float w2 = base[2 * DF2_H + c] + dyn[(size_t)r * DF2_DYN + 288 + g];
    float v = w2 * z_in[(size_t)r * DF2_H + c];
    if (t > 0) {
        const float w3 = base[3 * DF2_H + c] + dyn[(size_t)r * DF2_DYN + 432 + g];
        v = fma(w3, z_in[(size_t)(r - 1) * DF2_H + c], v);
    }
    x_io[(size_t)r * DF2_H + c] += v;
}

// High-BW single-pass 8-row Q8_ROW GEMV (16 rows/TG = 2 rows/SG, vectorized char4x2 loads -> 558 GB/s!)
// grid (ceil(N / 16), 1, 1), 256 threads
kernel void df2_q8_mv8(
    device const char* W     [[buffer(0)]],
    device const float* w_sc [[buffer(1)]],
    device const float* X    [[buffer(2)]],  // [8, K]
    device float* Y          [[buffer(3)]],  // [8, N]
    constant int& N          [[buffer(4)]],
    constant int& K          [[buffer(5)]],
    uint tg  [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]]) {
    const uint sg = tid >> 5, lane = tid & 31u;
    const int row0 = int(tg) * 16 + int(sg) * 2;
    if (row0 >= N) return;
    const bool h1 = (row0 + 1) < N;
    device const char4* w0p = (device const char4*)(W + (size_t)row0 * K);
    device const char4* w1p = (device const char4*)(W + (size_t)(row0 + (h1 ? 1 : 0)) * K);
    const int K4 = K >> 2;
    float a0[8] = {0,0,0,0,0,0,0,0};
    float a1[8] = {0,0,0,0,0,0,0,0};
    for (int k = int(lane); k < K4; k += 32) {
        const float4 w0 = float4(w0p[k]);
        const float4 w1 = h1 ? float4(w1p[k]) : float4(0.0f);
        for (int i = 0; i < 8; ++i) {
            const float4 xv = ((device const float4*)(X + (size_t)i * K))[k];
            a0[i] += dot(w0, xv);
            if (h1) a1[i] += dot(w1, xv);
        }
    }
    const float sc0 = w_sc[row0];
    const float sc1 = h1 ? w_sc[row0 + 1] : 0.0f;
    for (int i = 0; i < 8; ++i) {
        const float v0 = simd_sum(a0[i]) * sc0;
        if (lane == 0u) Y[(size_t)i * N + row0] = v0;
        if (h1) {
            const float v1 = simd_sum(a1[i]) * sc1;
            if (lane == 0u) Y[(size_t)i * N + row0 + 1] = v1;
        }
    }
}

// 2D-parallel 8-row-per-TG BF16 GEMM/GEMV (for fc, wkv, conv_a, conv_f, sel_h): grid (ceil(N/16), ceil(M/8), 1), 256 threads
kernel void df2_bf16_mv_m(
    device const ushort* W [[buffer(0)]],
    device const float* X  [[buffer(1)]],  // [M, K]
    device float* Y        [[buffer(2)]],  // [M, N]
    constant int& N        [[buffer(3)]],
    constant int& K        [[buffer(4)]],
    constant int& M        [[buffer(5)]],
    uint2 tgp [[threadgroup_position_in_grid]],
    uint tid  [[thread_index_in_threadgroup]]) {
    const uint sg = tid >> 5, lane = tid & 31u;
    const int row0 = int(tgp.x) * 16 + int(sg) * 2;
    const int m0 = int(tgp.y) * 8;
    if (row0 >= N || m0 >= M) return;
    const int mm = min(8, M - m0);
    const bool h1 = (row0 + 1) < N;
    device const ushort4* w0p = (device const ushort4*)(W + (size_t)row0 * K);
    device const ushort4* w1p = (device const ushort4*)(W + (size_t)(row0 + (h1 ? 1 : 0)) * K);
    const int K4 = K >> 2;
    float a0[8] = {0,0,0,0,0,0,0,0};
    float a1[8] = {0,0,0,0,0,0,0,0};
    for (int k = int(lane); k < K4; k += 32) {
        const float4 w0 = df2_bf16x4(w0p[k]);
        const float4 w1 = h1 ? df2_bf16x4(w1p[k]) : float4(0.0f);
        for (int i = 0; i < mm; ++i) {
            const float4 xv = ((device const float4*)(X + (size_t)(m0 + i) * K))[k];
            a0[i] += dot(w0, xv);
            if (h1) a1[i] += dot(w1, xv);
        }
    }
    for (int i = 0; i < mm; ++i) {
        const float v0 = simd_sum(a0[i]);
        if (lane == 0u) Y[(size_t)(m0 + i) * N + row0] = v0;
        if (h1) {
            const float v1 = simd_sum(a1[i]);
            if (lane == 0u) Y[(size_t)(m0 + i) * N + row0 + 1] = v1;
        }
    }
}

// Per-head RMSNorm + NeoX half-split RoPE for block-8 Q [8, 32, 128] or K [8, 8, 128]
// grid (8, n_heads, 1), 64 threads (each thread owns pair (d, d+64))
kernel void df2_qk_norm_rope(
    device float* x_io       [[buffer(0)]],
    device const float* nw   [[buffer(1)]],  // [128]
    constant int& n_heads    [[buffer(2)]],
    constant int& stride_floats [[buffer(3)]],  // 4096 for Q, 2048 for K inside packed [K,V]
    constant int& pos0       [[buffer(4)]],
    constant float& theta    [[buffer(5)]],
    uint2 tgp [[threadgroup_position_in_grid]],
    uint tid  [[thread_index_in_threadgroup]]) {
    threadgroup float sh[2];
    const int t = int(tgp.x), h = int(tgp.y);
    device float* hp = x_io + (size_t)t * stride_floats + (size_t)h * 128ul;
    const float k0 = hp[tid], k1 = hp[tid + 64u];
    const float inv = rsqrt(df2_tg_sum(k0 * k0 + k1 * k1, sh, tid, 64u) * (1.0f / 128.0f) + DF2_EPS);
    const float x0 = k0 * inv * nw[tid];
    const float x1 = k1 * inv * nw[tid + 64u];
    const float ang = float(pos0 + t) * pow(theta, -float(2u * tid) * (1.0f / 128.0f));
    const float cs = cos(ang), sn = sin(ang);
    hp[tid]       = x0 * cs - x1 * sn;
    hp[tid + 64u] = x1 * cs + x0 * sn;
}

// Non-causal Block-8 SWA-2048 32:8 GQA Attention (4 Q heads per KV head, Split-K S_PARTS = 8!)
// Attends to cell in [max(lo, pos0+t-2047), pos0+7] where cell < ctx_hi || cell >= pos0
// grid (8, 8, 8) = (block_row, kv_head, split_k), 128 threads (4 SGs)
struct Df2AttnArgs {
    int slot;
    int pos0;
    int ctx_lo;
    int ctx_hi;
    float scale;
    int pad;
};

kernel void df2_block_attn_splitk(
    device const float* q_buf   [[buffer(0)]],  // [8, 32, 128]
    device const float* kv_blk  [[buffer(1)]],  // [8, 2048] (first 1024=K, next 1024=V)
    device const half* ring_k   [[buffer(2)]],  // [max_slots, 2112, 1024]
    device const half* ring_v   [[buffer(3)]],  // [max_slots, 2112, 1024]
    device float* part_o        [[buffer(4)]],  // [8, 32, 8, 128]
    device float* part_ml       [[buffer(5)]],  // [8, 32, 8, 2]
    constant Df2AttnArgs& A     [[buffer(6)]],
    uint3 tgp  [[threadgroup_position_in_grid]],
    uint sg    [[simdgroup_index_in_threadgroup]],
    uint lane  [[thread_index_in_simdgroup]]) {
    constexpr int S_PARTS = 8;
    threadgroup float sh_m[4][4];
    threadgroup float sh_l[4][4];
    threadgroup float sh_o[4][4 * 128];

    const int t = int(tgp.x), kh = int(tgp.y), sp = int(tgp.z);
    const int h0 = kh * 4;
    const float4 q0 = ((device const float4*)(q_buf + ((size_t)t * 32 + h0 + 0) * 128))[lane] * A.scale;
    const float4 q1 = ((device const float4*)(q_buf + ((size_t)t * 32 + h0 + 1) * 128))[lane] * A.scale;
    const float4 q2 = ((device const float4*)(q_buf + ((size_t)t * 32 + h0 + 2) * 128))[lane] * A.scale;
    const float4 q3 = ((device const float4*)(q_buf + ((size_t)t * 32 + h0 + 3) * 128))[lane] * A.scale;

    const int k_lo = max(A.ctx_lo, A.pos0 + t - (DF2_WIN - 1));
    const int k_hi = min(A.ctx_hi, A.pos0);
    const int k_tot = max(0, k_hi - k_lo);
    const int chunk = (k_tot + S_PARTS - 1) / S_PARTS;
    const int p_start = k_lo + sp * chunk;
    const int p_end = min(k_hi, p_start + chunk);

    float m0 = -1e30f, m1 = -1e30f, m2 = -1e30f, m3 = -1e30f;
    float l0 = 0.0f,   l1 = 0.0f,   l2 = 0.0f,   l3 = 0.0f;
    float4 o0 = float4(0.0f), o1 = float4(0.0f), o2 = float4(0.0f), o3 = float4(0.0f);

    const size_t slot_base = (size_t)A.slot * DF2_RING;
    for (int k = p_start + int(sg); k < p_end; k += 4) {
        const int cell = k % DF2_RING;
        const size_t off = (slot_base + (size_t)cell) * DF2_KV + (size_t)kh * 128ul;
        const float4 kv = float4(((device const half4*)(ring_k + off))[lane]);
        const float4 vv = float4(((device const half4*)(ring_v + off))[lane]);
        { const float sc = simd_sum(dot(q0, kv)); const float mn = max(m0, sc); const float c = exp(m0 - mn); const float p = exp(sc - mn); l0 = l0 * c + p; m0 = mn; o0 = o0 * c + p * vv; }
        { const float sc = simd_sum(dot(q1, kv)); const float mn = max(m1, sc); const float c = exp(m1 - mn); const float p = exp(sc - mn); l1 = l1 * c + p; m1 = mn; o1 = o1 * c + p * vv; }
        { const float sc = simd_sum(dot(q2, kv)); const float mn = max(m2, sc); const float c = exp(m2 - mn); const float p = exp(sc - mn); l2 = l2 * c + p; m2 = mn; o2 = o2 * c + p * vv; }
        { const float sc = simd_sum(dot(q3, kv)); const float mn = max(m3, sc); const float c = exp(m3 - mn); const float p = exp(sc - mn); l3 = l3 * c + p; m3 = mn; o3 = o3 * c + p * vv; }
    }

    // Split 0 also scores the 8 non-causal block keys [0..7]!
    if (sp == 0) {
        for (int bj = int(sg); bj < 8; bj += 4) {
            const float4 kv = ((device const float4*)(kv_blk + (size_t)bj * 2048ul + (size_t)kh * 128ul))[lane];
            const float4 vv = ((device const float4*)(kv_blk + (size_t)bj * 2048ul + 1024ul + (size_t)kh * 128ul))[lane];
            { const float sc = simd_sum(dot(q0, kv)); const float mn = max(m0, sc); const float c = exp(m0 - mn); const float p = exp(sc - mn); l0 = l0 * c + p; m0 = mn; o0 = o0 * c + p * vv; }
            { const float sc = simd_sum(dot(q1, kv)); const float mn = max(m1, sc); const float c = exp(m1 - mn); const float p = exp(sc - mn); l1 = l1 * c + p; m1 = mn; o1 = o1 * c + p * vv; }
            { const float sc = simd_sum(dot(q2, kv)); const float mn = max(m2, sc); const float c = exp(m2 - mn); const float p = exp(sc - mn); l2 = l2 * c + p; m2 = mn; o2 = o2 * c + p * vv; }
            { const float sc = simd_sum(dot(q3, kv)); const float mn = max(m3, sc); const float c = exp(m3 - mn); const float p = exp(sc - mn); l3 = l3 * c + p; m3 = mn; o3 = o3 * c + p * vv; }
        }
    }

    if (lane == 0u) {
        sh_m[0][sg] = m0; sh_l[0][sg] = l0;
        sh_m[1][sg] = m1; sh_l[1][sg] = l1;
        sh_m[2][sg] = m2; sh_l[2][sg] = l2;
        sh_m[3][sg] = m3; sh_l[3][sg] = l3;
    }
    *((threadgroup float4*)(sh_o[0] + sg * 128 + lane * 4)) = o0;
    *((threadgroup float4*)(sh_o[1] + sg * 128 + lane * 4)) = o1;
    *((threadgroup float4*)(sh_o[2] + sg * 128 + lane * 4)) = o2;
    *((threadgroup float4*)(sh_o[3] + sg * 128 + lane * 4)) = o3;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // All 4 SGs reduce their respective Q head in parallel!
    const int qi = int(sg);
    const int h = h0 + qi;
    const float gm = max(max(sh_m[qi][0], sh_m[qi][1]), max(sh_m[qi][2], sh_m[qi][3]));
    float gl = 0.0f;
    float4 go = float4(0.0f);
    if (gm > -1e29f) {
        for (int s = 0; s < 4; ++s) {
            const float w = (sh_m[qi][s] > -1e29f) ? exp(sh_m[qi][s] - gm) : 0.0f;
            gl += sh_l[qi][s] * w;
            go += (*((threadgroup float4*)(sh_o[qi] + s * 128 + lane * 4))) * w;
        }
    }
    const size_t pidx = ((size_t)t * 32 + h) * S_PARTS + sp;
    ((device float4*)(part_o + pidx * 128))[lane] = go;
    if (lane == 0u) {
        part_ml[pidx * 2 + 0] = gm;
        part_ml[pidx * 2 + 1] = gl;
    }
}

// Merge S_PARTS=8 splits -> attn_out [8, 32, 128]: grid (8, 32, 1), 128 threads
kernel void df2_block_attn_reduce(
    device const float* part_o  [[buffer(0)]],
    device const float* part_ml [[buffer(1)]],
    device float* attn_out      [[buffer(2)]],
    uint2 tgp [[threadgroup_position_in_grid]],
    uint tid  [[thread_index_in_threadgroup]]) {
    constexpr int S_PARTS = 8;
    const int t = int(tgp.x), h = int(tgp.y);
    const size_t base = ((size_t)t * 32 + h) * S_PARTS;
    float gm = -1e30f;
    for (int sp = 0; sp < S_PARTS; ++sp) gm = max(gm, part_ml[(base + sp) * 2 + 0]);
    float gl = 0.0f, acc = 0.0f;
    if (gm > -1e29f) {
        for (int sp = 0; sp < S_PARTS; ++sp) {
            const float pm = part_ml[(base + sp) * 2 + 0];
            const float pl = part_ml[(base + sp) * 2 + 1];
            if (pl > 0.0f && pm > -1e29f) {
                const float w = exp(pm - gm);
                gl += pl * w;
                acc = fma(part_o[(base + sp) * 128 + tid], w, acc);
            }
        }
    }
    attn_out[((size_t)t * 32 + h) * 128 + tid] = (gl > 0.0f) ? (acc / gl) : 0.0f;
}

// Standard SwiGLU: out[i] = silu(g[i]) * u[i]
kernel void df2_swiglu(
    device const float* g [[buffer(0)]],
    device const float* u [[buffer(1)]],
    device float* out     [[buffer(2)]],
    constant int& N       [[buffer(3)]],
    uint gid [[thread_position_in_grid]]) {
    if (int(gid) < N) {
        const float gv = g[gid];
        out[gid] = (gv / (1.0f + exp(-gv))) * u[gid];
    }
}

// ============================================================================
// 4. Single-DRAM-Pass Fused Q8 Unembed Screen -> Exact Top-16 + Full-Vocab Softcapped Tail
// ============================================================================
// Stage 1: grid (ceil(V / 64), 1, 1), 256 threads (8 SGs x 8 vocab rows/SG).
// Reads unembed_q8 ONCE across all 7 draft rows (x_hd[1..7, 2304])!
// Writes per-TG Top-16 (part_ci [7, nparts, 16], part_cv [7, nparts, 16]) and tail (part_tail [7, nparts, 2]).
kernel void df2_head_top16_part_q8(
    device const char* W_q     [[buffer(0)]],  // [V, 2304] int8
    device const float* W_sc   [[buffer(1)]],  // [V] float
    device const float* X_hd   [[buffer(2)]],  // [8, 2304] (draft rows are 1..7!)
    device int* part_ci        [[buffer(3)]],  // [7, nparts, 16]
    device float* part_cv      [[buffer(4)]],  // [7, nparts, 16]
    device float* part_tail    [[buffer(5)]],  // [7, nparts, 2]
    constant int& V            [[buffer(6)]],
    constant int& nparts       [[buffer(7)]],
    uint tg  [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]]) {
    threadgroup float sLog[7][64];
    const uint sg = tid >> 5, lane = tid & 31u;
    const int v_base = int(tg) * 64;
    const int r0 = v_base + int(sg) * 8;
    const int K4 = DF2_H >> 2;

    for (int sub = 0; sub < 8; sub += 2) {
        const int vr0 = r0 + sub;
        const int vr1 = vr0 + 1;
        const bool ok0 = (vr0 < V), ok1 = (vr1 < V);
        if (ok0) {
            device const char4* w0p = (device const char4*)(W_q + (size_t)vr0 * DF2_H);
            device const char4* w1p = ok1 ? (device const char4*)(W_q + (size_t)vr1 * DF2_H) : w0p;
            float a0[7] = {0,0,0,0,0,0,0};
            float a1[7] = {0,0,0,0,0,0,0};
            for (int k = int(lane); k < K4; k += 32) {
                const float4 w0 = float4(w0p[k]);
                const float4 w1 = ok1 ? float4(w1p[k]) : float4(0.0f);
                for (int d = 0; d < 7; ++d) {
                    const float4 xv = ((device const float4*)(X_hd + (size_t)(d + 1) * DF2_H))[k];
                    a0[d] += dot(w0, xv);
                    if (ok1) a1[d] += dot(w1, xv);
                }
            }
            const float sc0 = W_sc[vr0];
            const float sc1 = ok1 ? W_sc[vr1] : 0.0f;
            for (int d = 0; d < 7; ++d) {
                const float l0 = simd_sum(a0[d]) * sc0;
                const float l1 = ok1 ? (simd_sum(a1[d]) * sc1) : -INFINITY;
                if (lane == 0u) {
                    sLog[d][int(sg) * 8 + sub]     = l0;
                    sLog[d][int(sg) * 8 + sub + 1] = l1;
                }
            }
        } else {
            if (lane == 0u) {
                for (int d = 0; d < 7; ++d) {
                    sLog[d][int(sg) * 8 + sub]     = -INFINITY;
                    sLog[d][int(sg) * 8 + sub + 1] = -INFINITY;
                }
            }
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // 7 SGs (sg = 0..6) each independently extract exact Top-16 + softcapped tail over the 64 vocab rows!
    if (sg < 7u) {
        const int d = int(sg);
        const int i0 = v_base + int(lane);
        const int i1 = v_base + int(lane) + 32;
        float v0 = (i0 < V) ? sLog[d][lane]      : -INFINITY;
        float v1 = (i1 < V) ? sLog[d][lane + 32] : -INFINITY;
        int id0 = (i0 < V) ? i0 : 0x7fffffff;
        int id1 = (i1 < V) ? i1 : 0x7fffffff;

        const size_t cbase = ((size_t)d * (size_t)nparts + (size_t)tg) * 16ul;
        for (int k = 0; k < DF2_TOPK; ++k) {
            float bv = df2_better(v0, id0, v1, id1) ? v0 : v1;
            int bi   = df2_better(v0, id0, v1, id1) ? id0 : id1;
            for (ushort o = 16; o > 0; o >>= 1) {
                const float ov = simd_shuffle_xor(bv, o);
                const int oi   = simd_shuffle_xor(bi, o);
                if (df2_better(ov, oi, bv, bi)) { bv = ov; bi = oi; }
            }
            if (lane == 0u) {
                part_ci[cbase + k] = bi;
                part_cv[cbase + k] = bv;
            }
            if (id0 == bi) { v0 = -INFINITY; id0 = 0x7fffffff; }
            if (id1 == bi) { v1 = -INFINITY; id1 = 0x7fffffff; }
        }
        // Remaining 48 rows -> exact softcapped tail (m_t, l_t)
        const float c0 = (id0 < V) ? df2_softcap(v0, DF2_CAP) : -INFINITY;
        const float c1 = (id1 < V) ? df2_softcap(v1, DF2_CAP) : -INFINITY;
        const float m_t = simd_max(max(c0, c1));
        float l_t = 0.0f;
        if (isfinite(m_t)) {
            const float e0 = (id0 < V) ? exp(c0 - m_t) : 0.0f;
            const float e1 = (id1 < V) ? exp(c1 - m_t) : 0.0f;
            l_t = simd_sum(e0 + e1);
        }
        if (lane == 0u) {
            const size_t tbase = ((size_t)d * (size_t)nparts + (size_t)tg) * 2ul;
            part_tail[tbase + 0] = m_t;
            part_tail[tbase + 1] = l_t;
        }
    }
}

// Stage 2: Exact Global Top-16 + Softcapped Tail Reduce across nparts: grid (7, 1, 1), 256 threads (8 SGs)
// Writes global cand [7, 32] (first 16 = ids, next 16 = as_type<int>(raw_logit)) and tail [7, 2]
kernel void df2_head_top16_reduce(
    device const int* part_ci    [[buffer(0)]],  // [7, nparts, 16]
    device const float* part_cv  [[buffer(1)]],  // [7, nparts, 16]
    device const float* part_tail [[buffer(2)]], // [7, nparts, 2]
    device int* out_cand         [[buffer(3)]],  // [7, 32]
    device float* out_tail       [[buffer(4)]],  // [7, 2]
    constant int& nparts         [[buffer(5)]],
    uint d   [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]]) {
    threadgroup int sh_ci[8][16];
    threadgroup float sh_cv[8][16];
    threadgroup float sh_mt[8];
    threadgroup float sh_lt[8];

    const uint sg = tid >> 5, lane = tid & 31u;
    const size_t p_base = (size_t)d * (size_t)nparts;

    // Each of the 8 SGs maintains its own Top-16 (lanes 0..15 hold sorted top-16) + tail (m_t, l_t)
    float my_v = -INFINITY;
    int my_i = 0x7fffffff;
    float m_t = -INFINITY, l_t = 0.0f;

    auto merge_tail = [&](float mo, float lo) {
        if (!isfinite(mo) || lo <= 0.0f) return;
        if (!isfinite(m_t)) { m_t = mo; l_t = lo; }
        else if (mo > m_t)  { l_t = l_t * exp(m_t - mo) + lo; m_t = mo; }
        else                { l_t += lo * exp(mo - m_t); }
    };

    // Each thread in SG loads 16 candidates? Even cleaner: 1 thread per SG lane?
    // With 8 SGs, lane 0..15 hold the running Top-16; each part brings 16 new items in lanes 16..31!
    for (int p = int(sg); p < nparts; p += 8) {
        if (lane == 0u) {
            merge_tail(part_tail[(p_base + (size_t)p) * 2ul + 0], part_tail[(p_base + (size_t)p) * 2ul + 1]);
        }
        float v = (lane < 16u) ? my_v : part_cv[(p_base + (size_t)p) * 16ul + (lane - 16u)];
        int id  = (lane < 16u) ? my_i : part_ci[(p_base + (size_t)p) * 16ul + (lane - 16u)];

        // Extract top 16 of 32 items in this SG -> write into lanes 0..15
        float next_v = -INFINITY;
        int next_i = 0x7fffffff;
        for (int k = 0; k < 16; ++k) {
            float bv = v; int bi = id;
            for (ushort o = 16; o > 0; o >>= 1) {
                const float ov = simd_shuffle_xor(bv, o);
                const int oi   = simd_shuffle_xor(bi, o);
                if (df2_better(ov, oi, bv, bi)) { bv = ov; bi = oi; }
            }
            if (int(lane) == k) { next_v = bv; next_i = bi; }
            if (id == bi) { v = -INFINITY; id = 0x7fffffff; }
        }
        // Remaining 16 evicted candidates fold into this SG's tail!
        const float ev_c = (id < 0x7fffffff && isfinite(v)) ? df2_softcap(v, DF2_CAP) : -INFINITY;
        const float ev_m = simd_max(ev_c);
        float ev_l = 0.0f;
        if (isfinite(ev_m)) ev_l = simd_sum((id < 0x7fffffff && isfinite(v)) ? exp(ev_c - ev_m) : 0.0f);
        if (lane == 0u) merge_tail(ev_m, ev_l);

        my_v = next_v;
        my_i = next_i;
    }

    if (lane < 16u) {
        sh_ci[sg][lane] = my_i;
        sh_cv[sg][lane] = my_v;
    }
    if (lane == 0u) {
        sh_mt[sg] = m_t;
        sh_lt[sg] = l_t;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // SG 0 merges the 8 SG results (8 x 16 = 128 candidates + 8 tails)
    if (sg == 0u) {
        m_t = sh_mt[0]; l_t = sh_lt[0];
        if (lane == 0u) {
            for (int s = 1; s < 8; ++s) merge_tail(sh_mt[s], sh_lt[s]);
        }
        my_v = (lane < 16u) ? sh_cv[0][lane] : -INFINITY;
        my_i = (lane < 16u) ? sh_ci[0][lane] : 0x7fffffff;
        for (int s = 1; s < 8; ++s) {
            float v = (lane < 16u) ? my_v : sh_cv[s][lane - 16u];
            int id  = (lane < 16u) ? my_i : sh_ci[s][lane - 16u];
            float next_v = -INFINITY;
            int next_i = 0x7fffffff;
            for (int k = 0; k < 16; ++k) {
                float bv = v; int bi = id;
                for (ushort o = 16; o > 0; o >>= 1) {
                    const float ov = simd_shuffle_xor(bv, o);
                    const int oi   = simd_shuffle_xor(bi, o);
                    if (df2_better(ov, oi, bv, bi)) { bv = ov; bi = oi; }
                }
                if (int(lane) == k) { next_v = bv; next_i = bi; }
                if (id == bi) { v = -INFINITY; id = 0x7fffffff; }
            }
            const float ev_c = (id < 0x7fffffff && isfinite(v)) ? df2_softcap(v, DF2_CAP) : -INFINITY;
            const float ev_m = simd_max(ev_c);
            float ev_l = 0.0f;
            if (isfinite(ev_m)) ev_l = simd_sum((id < 0x7fffffff && isfinite(v)) ? exp(ev_c - ev_m) : 0.0f);
            if (lane == 0u) merge_tail(ev_m, ev_l);
            my_v = next_v;
            my_i = next_i;
        }
        if (lane < 16u) {
            out_cand[(size_t)d * 32ul + lane]      = my_i;
            out_cand[(size_t)d * 32ul + 16u + lane] = as_type<int>(my_v);
        }
        if (lane == 0u) {
            out_tail[(size_t)d * 2ul + 0] = m_t;
            out_tail[(size_t)d * 2ul + 1] = l_t;
        }
    }
}

// ============================================================================
// 5. Sequential 7-Step Rank-256 Bilinear Candidate Selector (exact port of k_df2_select)
//    grid (1, 1, 1), 512 threads (16 SGs x 32 lanes -> 1 SG per Top-16 candidate!)
// ============================================================================
kernel void df2_select_chain(
    device const int* cand      [[buffer(0)]],  // [7, 32] (0..15 ids, 16..31 raw logit bits)
    device const float* tail    [[buffer(1)]],  // [7, 2]
    device const float* gsel    [[buffer(2)]],  // [8, 256] (draft rows are 1..7)
    device const ushort* pred   [[buffer(3)]],  // [V, 256] BF16
    device const ushort* succ   [[buffer(4)]],  // [V, 256] BF16
    device int* ids_out         [[buffer(5)]],  // [7]
    device float* conf_out      [[buffer(6)]],  // [7]
    constant int& anchor_tok    [[buffer(7)]],
    constant int& n_drafts      [[buffer(8)]],
    uint tid [[thread_index_in_threadgroup]]) {
    threadgroup float pg[DF2_RANK];
    threadgroup float sc_sh[DF2_TOPK];
    threadgroup int prev_s;

    const uint sg = tid >> 5, lane = tid & 31u;
    if (tid == 0u) prev_s = anchor_tok;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (int d = 0; d < n_drafts; ++d) {
        const int pv = prev_s;
        if (tid < DF2_RANK) {
            pg[tid] = df2_bf16f(pred[(size_t)pv * DF2_RANK + tid]) * gsel[(size_t)(d + 1) * DF2_RANK + tid];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        const int c = cand[(size_t)d * 32ul + sg];
        float dt = 0.0f;
        if (c >= 0 && c < 0x7fffffff) {
            device const ushort4* sp = (device const ushort4*)(succ + (size_t)c * DF2_RANK);
            const uint k4 = lane * 2u;  // 32 lanes x 8 elems = 256!
            const float4 p0 = float4(pg[k4 * 4 + 0], pg[k4 * 4 + 1], pg[k4 * 4 + 2], pg[k4 * 4 + 3]);
            const float4 p1 = float4(pg[k4 * 4 + 4], pg[k4 * 4 + 5], pg[k4 * 4 + 6], pg[k4 * 4 + 7]);
            dt = dot(p0, df2_bf16x4(sp[k4])) + dot(p1, df2_bf16x4(sp[k4 + 1u]));
            dt = simd_sum(dt);
        }
        if (lane == 0u) {
            sc_sh[sg] = as_type<float>(cand[(size_t)d * 32ul + 16u + sg]) + dt;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        if (sg == 0u) {
            const float my_sc = (lane < DF2_TOPK) ? sc_sh[lane] : -INFINITY;
            float bv = my_sc;
            int bi   = (lane < DF2_TOPK) ? cand[(size_t)d * 32ul + lane] : 0x7fffffff;
            for (ushort o = 8; o > 0; o >>= 1) {
                const float ov = simd_shuffle_xor(bv, o);
                const int oi   = simd_shuffle_xor(bi, o);
                if (df2_better(ov, oi, bv, bi)) { bv = ov; bi = oi; }
            }
            const float top = df2_softcap(bv, DF2_CAP);
            float m_ref = top, tail_term = 0.0f;
            const float m_t = tail[2 * d];
            const float l_t = tail[2 * d + 1];
            if (isfinite(m_t) && l_t > 0.0f) {
                m_ref = max(top, m_t);
                tail_term = l_t * exp(m_t - m_ref);
            }
            const float my_cap = (lane < DF2_TOPK) ? df2_softcap(my_sc, DF2_CAP) : -INFINITY;
            const float z16 = simd_sum((lane < DF2_TOPK) ? exp(my_cap - m_ref) : 0.0f);
            if (lane == 0u) {
                ids_out[d]  = bi;
                conf_out[d] = exp(top - m_ref) / (z16 + tail_term);
                prev_s = bi;
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
}
