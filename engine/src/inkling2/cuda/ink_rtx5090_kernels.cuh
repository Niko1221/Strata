// engine/src/inkling2/cuda/ink_rtx5090_kernels.cuh - Opt-in RTX 5090 (sm_120) kernels for Prefill, C1, and C64
//
// MIT License
// Copyright (c) 2026 Niko1221 and the Strata contributors
//
// Placed NEXT TO the baseline kernels and dispatched ONLY when STRATA_INK_RTX5090=1 (or granular sub-switch)
// is enabled on an sm_120 GPU. Preserves exact greedy (temperature=0) parity with baseline.
#pragma once

#include "ink_cuda.cuh"
#include "ink_attn.cuh"
#include "ink_kernels.cuh"
#include "ink_moe_cuda.cuh"
#include "ink_rtx5090_cfg.cuh"

namespace strata::ink::rtx5090 {

#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 800)
__device__ __forceinline__ void mma_bf16_m16n8k16(
    float (&d)[4], uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3, uint32_t b0, uint32_t b1) {
    asm volatile(
        "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
        "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
        : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
        : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
}

__device__ __forceinline__ void mma_fp16_m16n8k16(
    float (&d)[4], uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3, uint32_t b0, uint32_t b1) {
    asm volatile(
        "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
        "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
        : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
        : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
}

__device__ __forceinline__ void mma_fp16_acc16_m16n8k16(
    uint32_t (&d)[2], uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3, uint32_t b0, uint32_t b1) {
    asm volatile(
        "mma.sync.aligned.m16n8k16.row.col.f16.f16.f16.f16 "
        "{%0, %1}, {%2, %3, %4, %5}, {%6, %7}, {%0, %1};\n"
        : "+r"(d[0]), "+r"(d[1])
        : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
}

__device__ __forceinline__ void ldmatrix_x4(uint32_t (&r)[4], const void* smem_ptr) {
    uint32_t addr = static_cast<uint32_t>(__cvta_generic_to_shared(smem_ptr));
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                 : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3])
                 : "r"(addr));
}

__device__ __forceinline__ void ldmatrix_x4_trans(uint32_t (&r)[4], const void* smem_ptr) {
    uint32_t addr = static_cast<uint32_t>(__cvta_generic_to_shared(smem_ptr));
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                 : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3])
                 : "r"(addr));
}

__device__ __forceinline__ void fa_cp16_async(void* smem_dst, const void* glob_src) {
    uint32_t addr = static_cast<uint32_t>(__cvta_generic_to_shared(smem_dst));
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" :: "r"(addr), "l"(glob_src));
}
__device__ __forceinline__ void fa_cp_commit() { asm volatile("cp.async.commit_group;\n" ::: "memory"); }
__device__ __forceinline__ void fa_cp_wait0()  { asm volatile("cp.async.wait_group 0;\n" ::: "memory"); }
__device__ __forceinline__ void fa_cp_wait1()  { asm volatile("cp.async.wait_group 1;\n" ::: "memory"); }
#endif

__device__ __forceinline__ uint32_t pack_bf16x2_rn(float a, float b) {
    const __nv_bfloat162 v = __floats2bfloat162_rn(a, b);
    return *reinterpret_cast<const uint32_t*>(&v);
}

__device__ __forceinline__ uint32_t pack_fp16x2_rn(float a, float b) {
    const __half2 v = __floats2half2_rn(a, b);
    return *reinterpret_cast<const uint32_t*>(&v);
}

// Transpose rel_proj [16, 1024] F32 -> [1024, 8] __half2 (32 KB/layer -> 100% L1/L2 resident, eliminates 19.35 GB d_rel_table!)
__global__ void __launch_bounds__(256) k_ink_transpose_rel_proj_h2(
    const float* __restrict__ rel_proj,   // [16, 1024]
    __half2* __restrict__ rel_proj_T) {   // [1024, 8]
    const int d = blockIdx.x * 256 + threadIdx.x;
    if (d >= kInkRelExtent) return;
#pragma unroll
    for (int j = 0; j < 8; ++j) {
        const float a = rel_proj[(size_t)(2 * j + 0) * kInkRelExtent + d];
        const float b = rel_proj[(size_t)(2 * j + 1) * kInkRelExtent + d];
        rel_proj_T[(size_t)d * 8 + j] = __floats2half2_rn(a, b);
    }
}

__device__ __forceinline__ float rel_dot16_h2(
    const __half2 (&rv)[8], const __half2* __restrict__ rp_T, int dist) {
    if (dist >= kInkRelExtent) return 0.0f;
    const uint4 p0 = __ldg(reinterpret_cast<const uint4*>(rp_T + (size_t)dist * 8));
    const uint4 p1 = __ldg(reinterpret_cast<const uint4*>(rp_T + (size_t)dist * 8 + 4));
    __half2 a0 = __hmul2(rv[0], *reinterpret_cast<const __half2*>(&p0.x));
    __half2 a1 = __hmul2(rv[1], *reinterpret_cast<const __half2*>(&p0.y));
    a0 = __hfma2(rv[2], *reinterpret_cast<const __half2*>(&p0.z), a0);
    a1 = __hfma2(rv[3], *reinterpret_cast<const __half2*>(&p0.w), a1);
    a0 = __hfma2(rv[4], *reinterpret_cast<const __half2*>(&p1.x), a0);
    a1 = __hfma2(rv[5], *reinterpret_cast<const __half2*>(&p1.y), a1);
    a0 = __hfma2(rv[6], *reinterpret_cast<const __half2*>(&p1.z), a0);
    a1 = __hfma2(rv[7], *reinterpret_cast<const __half2*>(&p1.w), a1);
    const float2 f = __half22float2(__hadd2(a0, a1));
    return f.x + f.y;
}

// ============================================================================
// 1. Pipelined cp.async + ldmatrix.x4.trans Blackwell FP16-TC FlashAttention-2 Prefill (v7: BN=32/64 + BF16_IO + Fused amax!)
//    Grid: dim3(24 Q heads, (M + 63) / 64), Block: 128 (4 warps = 4 Q-tiles of 16 rows)
//    - Direct cp.async.cg.16B from FP16 KV cache -> XOR-swizzled s_k[BN][128] & s_v[BN][128] (ZERO conversion!)
//    - BN=32 (32 KB smem, 3 CTAs/SM) for SWA; BN=64 (48 KB smem, 2 CTAs/SM) for Global (2x fewer s_q reloads & rescales!)
//    - BF16_IO=true: reads FP16 Q & R directly (zero F32->FP16!), reads BF16 gate_attn (2x less BW), fuses d_amax_out!
// ============================================================================
template <int BN = 32, bool BF16_IO = false>
__global__ void __launch_bounds__(128) k_ink_flash_gqa_pf_sm120(
    const void* __restrict__ q_in,        // [M, 24, 128] F32 (BF16_IO=false) or FP16 (BF16_IO=true)
    const void* __restrict__ r_in,        // [M, 24, 16] F32 (BF16_IO=false) or FP16 (BF16_IO=true)
    const __half2* __restrict__ rel_proj_T, // [1024, 8] __half2 (32 KB, L1/L2 resident!)
    const float* __restrict__ sinks,      // [M, 24]
    const void* __restrict__ gate_in,     // [M, 24, 128] F32 or BF16 (nullable)
    InkKvVisitDesc kv,
    int slot0,
    int p0,
    int M,
    float* __restrict__ out,              // [M, 24, 128]
    float* __restrict__ d_amax_out = nullptr) {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 800)
    constexpr int BM = 64;
    constexpr int D  = kInkHeadDim;       // 128
    constexpr int NK8  = BN / 8;          // 4 (BN=32) or 8 (BN=64)
    constexpr int NK16 = BN / 16;         // 2 (BN=32) or 4 (BN=64)
    constexpr float kScale = 0.08838834764831845f;

    const int h      = blockIdx.x;        // 0..23 Q head (3 GQA heads adjacent -> L2 KV sharing!)
    const int q_base = blockIdx.y * BM;
    if (q_base >= M || h >= kInkHeads) return;
    const int kh     = h / 3;             // 0..7 KV head

    const int tid   = threadIdx.x;        // 0..127
    const int warp  = tid >> 5;           // 0..3 (4 Q-tiles of 16 rows)
    const int lane  = tid & 31;           // 0..31
    const int g     = lane >> 2;          // 0..7
    const int tig   = lane & 3;           // 0..3

    const int qr0 = q_base + warp * 16 + g;
    const int qr1 = qr0 + 8;
    const bool vq0 = (qr0 < M);
    const bool vq1 = (qr1 < M);
    const int qp0 = p0 + qr0;
    const int qp1 = p0 + qr1;

    __shared__ alignas(16) uint16_t s_q[BM][D];
    __shared__ alignas(16) uint16_t s_k[BN][D];
    __shared__ alignas(16) uint16_t s_v[BN][D];

    auto swiz = [](int r, int c8) -> int { return (c8 ^ (r & 7)) * 8; };

    // 1. Load Q[64, 128] into XOR-swizzled FP16 s_q (each warp loads its 16 rows)
    for (int r = 0; r < 16; ++r) {
        const int sr = warp * 16 + r;
        const int qr = q_base + sr;
        const int c8 = lane >> 1;   // 0..15 (8-half chunks)
        const int sub4 = lane & 1;  // 0..1 (4-half subchunk)
        uint32_t* dst = reinterpret_cast<uint32_t*>(&s_q[sr][swiz(sr, c8) + sub4 * 4]);
        if (qr < M) {
            if constexpr (BF16_IO) {
                const uint2 qh = *reinterpret_cast<const uint2*>(
                    reinterpret_cast<const __half*>(q_in) + ((size_t)qr * kInkHeads + h) * D + lane * 4);
                dst[0] = qh.x;
                dst[1] = qh.y;
            } else {
                const float4 qv = *reinterpret_cast<const float4*>(
                    reinterpret_cast<const float*>(q_in) + ((size_t)qr * kInkHeads + h) * D + lane * 4);
                dst[0] = pack_fp16x2_rn(qv.x, qv.y);
                dst[1] = pack_fp16x2_rn(qv.z, qv.w);
            }
        } else {
            dst[0] = 0u; dst[1] = 0u;
        }
    }

    __half2 r0_h2[8], r1_h2[8];
    if (vq0) {
        if constexpr (BF16_IO) {
            const uint4* rp = reinterpret_cast<const uint4*>(
                reinterpret_cast<const __half2*>(r_in) + ((size_t)qr0 * kInkHeads + h) * 8);
            const uint4 v0 = rp[0], v1 = rp[1];
            r0_h2[0] = *reinterpret_cast<const __half2*>(&v0.x); r0_h2[1] = *reinterpret_cast<const __half2*>(&v0.y);
            r0_h2[2] = *reinterpret_cast<const __half2*>(&v0.z); r0_h2[3] = *reinterpret_cast<const __half2*>(&v0.w);
            r0_h2[4] = *reinterpret_cast<const __half2*>(&v1.x); r0_h2[5] = *reinterpret_cast<const __half2*>(&v1.y);
            r0_h2[6] = *reinterpret_cast<const __half2*>(&v1.z); r0_h2[7] = *reinterpret_cast<const __half2*>(&v1.w);
        } else {
            const float4* rp = reinterpret_cast<const float4*>(
                reinterpret_cast<const float*>(r_in) + ((size_t)qr0 * kInkHeads + h) * kInkDRel);
#pragma unroll
            for (int j = 0; j < 4; ++j) {
                const float4 v = rp[j];
                r0_h2[2 * j + 0] = __floats2half2_rn(v.x, v.y);
                r0_h2[2 * j + 1] = __floats2half2_rn(v.z, v.w);
            }
        }
    }
    if (vq1) {
        if constexpr (BF16_IO) {
            const uint4* rp = reinterpret_cast<const uint4*>(
                reinterpret_cast<const __half2*>(r_in) + ((size_t)qr1 * kInkHeads + h) * 8);
            const uint4 v0 = rp[0], v1 = rp[1];
            r1_h2[0] = *reinterpret_cast<const __half2*>(&v0.x); r1_h2[1] = *reinterpret_cast<const __half2*>(&v0.y);
            r1_h2[2] = *reinterpret_cast<const __half2*>(&v0.z); r1_h2[3] = *reinterpret_cast<const __half2*>(&v0.w);
            r1_h2[4] = *reinterpret_cast<const __half2*>(&v1.x); r1_h2[5] = *reinterpret_cast<const __half2*>(&v1.y);
            r1_h2[6] = *reinterpret_cast<const __half2*>(&v1.z); r1_h2[7] = *reinterpret_cast<const __half2*>(&v1.w);
        } else {
            const float4* rp = reinterpret_cast<const float4*>(
                reinterpret_cast<const float*>(r_in) + ((size_t)qr1 * kInkHeads + h) * kInkDRel);
#pragma unroll
            for (int j = 0; j < 4; ++j) {
                const float4 v = rp[j];
                r1_h2[2 * j + 0] = __floats2half2_rn(v.x, v.y);
                r1_h2[2 * j + 1] = __floats2half2_rn(v.z, v.w);
            }
        }
    }

    float m_row0 = vq0 ? sinks[(size_t)qr0 * kInkHeads + h] : -1e30f;
    float m_row1 = vq1 ? sinks[(size_t)qr1 * kInkHeads + h] : -1e30f;
    float l_row0 = 1.0f;
    float l_row1 = 1.0f;
    uint32_t acc_o[16][2];
#pragma unroll
    for (int jd = 0; jd < 16; ++jd) {
        acc_o[jd][0] = 0u;
        acc_o[jd][1] = 0u;
    }

    const int q_end = min(M, q_base + BM);
    const bool is_swa = kv.is_swa;
    const int k_beg = is_swa ? max(0, p0 + q_base - kInkSwaWin + 1) : 0;
    const int k_end = is_swa ? (p0 + q_end) : min(p0 + q_end, kv.cap_pos);
    const bool fast_fp16 = (kv.kv_fmt == 0 && kv.host_cap == 0);
    // FP8 E4M3 KV in VRAM (global and SWA visits): the K / V tile of the next step is read with 16-byte loads into registers while the scores of the
    // current one are computed, then converted the same way as the generic loader (e4m3 -> f32, x scale, round to FP16): bit-identical, 2.5x faster at 128K
#ifdef STRATA_PF_FP8_LEGACY   // test builds only: the old generic loader, to prove the new one bit-identical
    const bool fast_fp8 = false;
#else
    const bool fast_fp8 = (kv.kv_fmt == 1 && kv.host_cap == 0);
#endif
    const size_t kv_slot_base = (size_t)slot0 * kv.cap_pos;

    const int ld_r0 = tid >> 4;   // 0..7
    const int ld_c8 = tid & 15;   // 0..15

    auto load_kv_tile_async = [&](uint16_t (&s_dst)[BN][D], const __half* g_ptr, int kt, int n_k) {
        if (fast_fp16) {
#pragma unroll
            for (int rep = 0; rep < NK8; ++rep) {
                const int ki = rep * 8 + ld_r0;
                uint16_t* dptr = &s_dst[ki][swiz(ki, ld_c8)];
                if (ki < n_k) {
                    const int kp = kt + ki;
                    const int p_idx = is_swa ? (kp % kv.cap_pos) : kp;
                    const __half* sp = g_ptr + ((kv_slot_base + p_idx) * kInkKVHeads + kh) * D + ld_c8 * 8;
                    fa_cp16_async(dptr, sp);
                } else {
                    *reinterpret_cast<uint4*>(dptr) = make_uint4(0u, 0u, 0u, 0u);
                }
            }
            fa_cp_commit();
        }
    };

    auto load_kv_tile_gen = [&](uint16_t (&s_dst)[BN][D], bool is_k, int kt, int n_k) {
        const int d4 = lane;
        const int c8 = lane >> 1;
        const int sub4 = lane & 1;
        for (int ki = warp; ki < BN; ki += 4) {
            uint32_t* dptr = reinterpret_cast<uint32_t*>(&s_dst[ki][swiz(ki, c8) + sub4 * 4]);
            if (ki < n_k) {
                const int kp = kt + ki;
                const int p_idx = is_swa ? (kp % kv.cap_pos) : kp;
                const bool use_host = (!is_swa && kv.host_cap > 0 && p_idx >= kv.dev_cap);
                const int stride = use_host ? kv.host_cap : (kv.host_cap > 0 ? kv.dev_cap : kv.cap_pos);
                const int pos_l  = use_host ? (p_idx - kv.dev_cap) : p_idx;
                const size_t hbase = ((size_t)slot0 * stride + pos_l) * kInkKVHeads + kh;
                const size_t base  = hbase * D + (size_t)d4 * 4;
                if (kv.kv_fmt == 0) {
                    const __half* p = is_k ? (use_host ? kv.h_k_f16 : kv.k_f16) : (use_host ? kv.h_v_f16 : kv.v_f16);
                    const uint2 h2 = *reinterpret_cast<const uint2*>(p + base);
                    dptr[0] = h2.x; dptr[1] = h2.y;
                } else if (kv.kv_fmt == 1) {
                    const uint8_t* p = is_k ? (use_host ? kv.h_k_f8 : kv.k_f8) : (use_host ? kv.h_v_f8 : kv.v_f8);
                    const float sc = (is_k ? (use_host ? kv.h_k_sc : kv.k_sc) : (use_host ? kv.h_v_sc : kv.v_sc))[hbase];
                    const uint32_t u = *reinterpret_cast<const uint32_t*>(p + base);
                    const float2 f0 = e4m3x2_to_f32x2((uint16_t)(u & 0xFFFFu));
                    const float2 f1 = e4m3x2_to_f32x2((uint16_t)(u >> 16));
                    dptr[0] = pack_fp16x2_rn(f0.x * sc, f0.y * sc);
                    dptr[1] = pack_fp16x2_rn(f1.x * sc, f1.y * sc);
                } else if (kv.kv_fmt == 2 || is_k) {
                    const uint8_t* p = is_k ? (use_host ? kv.h_k_f8 : kv.k_f8) : (use_host ? kv.h_v_f8 : kv.v_f8);
                    const float sc = (is_k ? (use_host ? kv.h_k_sc : kv.k_sc) : (use_host ? kv.h_v_sc : kv.v_sc))[hbase];
                    const uint32_t u = *reinterpret_cast<const uint32_t*>(p + base);
                    dptr[0] = pack_fp16x2_rn((float)(int8_t)(u & 0xFFu) * sc, (float)(int8_t)((u >> 8) & 0xFFu) * sc);
                    dptr[1] = pack_fp16x2_rn((float)(int8_t)((u >> 16) & 0xFFu) * sc, (float)(int8_t)(u >> 24) * sc);
                } else {
                    const uint8_t* p = use_host ? kv.h_v_f8 : kv.v_f8;
                    const float sc = (use_host ? kv.h_v_sc : kv.v_sc)[hbase];
                    const uint16_t vp = *reinterpret_cast<const uint16_t*>(p + hbase * (D / 2) + (size_t)d4 * 2);
                    dptr[0] = pack_fp16x2_rn((float)((int)(vp & 0xFu) - 8) * sc, (float)((int)((vp >> 4) & 0xFu) - 8) * sc);
                    dptr[1] = pack_fp16x2_rn((float)((int)((vp >> 8) & 0xFu) - 8) * sc, (float)((int)((vp >> 12) & 0xFu) - 8) * sc);
                }
            } else {
                dptr[0] = 0u; dptr[1] = 0u;
            }
        }
    };

    constexpr int NP8 = BN / 16;   // 16-byte pieces per thread and tile: 8 threads per 128-byte row, 16 rows per pass
    uint4 raw_k8[NP8], raw_v8[NP8];
    float sc_k8[NP8], sc_v8[NP8];
    auto fp8_issue = [&](uint4 (&rr)[NP8], float (&rs)[NP8], bool is_k, int kt, int n_k) {
        const int piece = tid & 7;
#pragma unroll
        for (int ps = 0; ps < NP8; ++ps) {
            const int ki = ps * 16 + (tid >> 3);
            rr[ps] = make_uint4(0u, 0u, 0u, 0u);
            rs[ps] = 0.0f;
            if (ki < n_k) {
                const int kp = kt + ki;
                const int p_idx = is_swa ? (kp % kv.cap_pos) : kp;
                const size_t hbase = (kv_slot_base + p_idx) * kInkKVHeads + kh;
                rr[ps] = __ldcs(reinterpret_cast<const uint4*>((is_k ? kv.k_f8 : kv.v_f8) + hbase * D) + piece);
                rs[ps] = __ldg((is_k ? kv.k_sc : kv.v_sc) + hbase);
            }
        }
    };
    auto fp8_store = [&](uint16_t (&s_dst)[BN][D], const uint4 (&rr)[NP8], const float (&rs)[NP8]) {
        const int piece = tid & 7;
#pragma unroll
        for (int ps = 0; ps < NP8; ++ps) {
            const int ki = ps * 16 + (tid >> 3);
            const float sc = rs[ps];
            const uint32_t u[4] = {rr[ps].x, rr[ps].y, rr[ps].z, rr[ps].w};
            uint32_t w[8];
#pragma unroll
            for (int i = 0; i < 4; ++i) {
                const float2 f0 = e4m3x2_to_f32x2((uint16_t)(u[i] & 0xFFFFu));
                const float2 f1 = e4m3x2_to_f32x2((uint16_t)(u[i] >> 16));
                w[2 * i] = pack_fp16x2_rn(f0.x * sc, f0.y * sc);
                w[2 * i + 1] = pack_fp16x2_rn(f1.x * sc, f1.y * sc);
            }
            *reinterpret_cast<uint4*>(&s_dst[ki][swiz(ki, 2 * piece)]) = make_uint4(w[0], w[1], w[2], w[3]);
            *reinterpret_cast<uint4*>(&s_dst[ki][swiz(ki, 2 * piece + 1)]) = make_uint4(w[4], w[5], w[6], w[7]);
        }
    };

    if (k_beg < k_end) {
        const int n_k0 = min(BN, k_end - k_beg);
        if (fast_fp16) load_kv_tile_async(s_k, kv.k_f16, k_beg, n_k0);
        else if (fast_fp8) { fp8_issue(raw_k8, sc_k8, true, k_beg, n_k0); fp8_store(s_k, raw_k8, sc_k8); }
        else           load_kv_tile_gen(s_k, true, k_beg, n_k0);
    }

    for (int kt = k_beg; kt < k_end; kt += BN) {
        const int n_k = min(BN, k_end - kt);
        if (fast_fp16) fa_cp_wait0();
        __syncthreads();

        if (fast_fp16) load_kv_tile_async(s_v, kv.v_f16, kt, n_k);
        else if (fast_fp8) {
            fp8_issue(raw_v8, sc_v8, false, kt, n_k);
            if (kt + BN < k_end) fp8_issue(raw_k8, sc_k8, true, kt + BN, min(BN, k_end - (kt + BN)));
        }

        // 2. S[16, BN] = Q[16, 128] @ K[BN, 128]^T via ldmatrix.x4 + 419 TFLOPS FP16-Acc TC!
        uint32_t s_h2[NK8][2];
#pragma unroll
        for (int kn = 0; kn < NK8; ++kn) { s_h2[kn][0] = 0u; s_h2[kn][1] = 0u; }
        const int q_ld_r = warp * 16 + (lane & 15);
        const int q_ld_c = (lane >> 4) & 1;
        const int k_ld_r_in = ((lane >> 4) * 8) + (lane & 7);
        const int k_ld_c_in = (lane >> 3) & 1;
#pragma unroll
        for (int kd = 0; kd < 8; ++kd) {
            uint32_t a[4];
            ldmatrix_x4(a, &s_q[q_ld_r][swiz(q_ld_r, kd * 2 + q_ld_c)]);
#pragma unroll
            for (int k2 = 0; k2 < NK16; ++k2) {
                uint32_t b[4];
                const int kr = k2 * 16 + k_ld_r_in;
                ldmatrix_x4(b, &s_k[kr][swiz(kr, kd * 2 + k_ld_c_in)]);
                mma_fp16_acc16_m16n8k16(s_h2[2 * k2 + 0], a[0], a[1], a[2], a[3], b[0], b[1]);
                mma_fp16_acc16_m16n8k16(s_h2[2 * k2 + 1], a[0], a[1], a[2], a[3], b[2], b[3]);
            }
        }
        float s_reg[NK8][4];
#pragma unroll
        for (int kn = 0; kn < NK8; ++kn) {
            const float2 f0 = __half22float2(*reinterpret_cast<const __half2*>(&s_h2[kn][0]));
            const float2 f1 = __half22float2(*reinterpret_cast<const __half2*>(&s_h2[kn][1]));
            s_reg[kn][0] = f0.x; s_reg[kn][1] = f0.y;
            s_reg[kn][2] = f1.x; s_reg[kn][3] = f1.y;
        }

        // 3. RelPos + Mask + Online Softmax + Pack P -> FP16 rp[NK16][4]
        const bool all_far = (!is_swa && n_k == BN && (p0 + q_base - (kt + BN - 1)) >= kInkRelExtent);
        float tm0 = -1e30f, tm1 = -1e30f;
        if (all_far) {
#pragma unroll
            for (int kn = 0; kn < NK8; ++kn) {
                if (vq0) {
                    const float va = s_reg[kn][0] * kScale;
                    const float vb = s_reg[kn][1] * kScale;
                    s_reg[kn][0] = va; s_reg[kn][1] = vb;
                    tm0 = fmaxf(tm0, fmaxf(va, vb));
                } else { s_reg[kn][0] = -1e30f; s_reg[kn][1] = -1e30f; }
                if (vq1) {
                    const float va = s_reg[kn][2] * kScale;
                    const float vb = s_reg[kn][3] * kScale;
                    s_reg[kn][2] = va; s_reg[kn][3] = vb;
                    tm1 = fmaxf(tm1, fmaxf(va, vb));
                } else { s_reg[kn][2] = -1e30f; s_reg[kn][3] = -1e30f; }
            }
        } else {
#pragma unroll
            for (int kn = 0; kn < NK8; ++kn) {
                const int ki0 = kn * 8 + 2 * tig;
                const int kp_a = kt + ki0;
                const int kp_b = kp_a + 1;
                const bool in_a = (ki0 < n_k);
                const bool in_b = (ki0 + 1 < n_k);
                if (vq0) {
                    const int da = qp0 - kp_a, db = qp0 - kp_b;
                    const bool oka = in_a && (da >= 0) && (!is_swa || da < kInkSwaWin);
                    const bool okb = in_b && (db >= 0) && (!is_swa || db < kInkSwaWin);
                    const float va = oka ? (s_reg[kn][0] * kScale + rel_dot16_h2(r0_h2, rel_proj_T, da)) : -1e30f;
                    const float vb = okb ? (s_reg[kn][1] * kScale + rel_dot16_h2(r0_h2, rel_proj_T, db)) : -1e30f;
                    s_reg[kn][0] = va; s_reg[kn][1] = vb;
                    tm0 = fmaxf(tm0, fmaxf(va, vb));
                } else { s_reg[kn][0] = -1e30f; s_reg[kn][1] = -1e30f; }
                if (vq1) {
                    const int da = qp1 - kp_a, db = qp1 - kp_b;
                    const bool oka = in_a && (da >= 0) && (!is_swa || da < kInkSwaWin);
                    const bool okb = in_b && (db >= 0) && (!is_swa || db < kInkSwaWin);
                    const float va = oka ? (s_reg[kn][2] * kScale + rel_dot16_h2(r1_h2, rel_proj_T, da)) : -1e30f;
                    const float vb = okb ? (s_reg[kn][3] * kScale + rel_dot16_h2(r1_h2, rel_proj_T, db)) : -1e30f;
                    s_reg[kn][2] = va; s_reg[kn][3] = vb;
                    tm1 = fmaxf(tm1, fmaxf(va, vb));
                } else { s_reg[kn][2] = -1e30f; s_reg[kn][3] = -1e30f; }
            }
        }
        tm0 = fmaxf(tm0, __shfl_xor_sync(0xFFFFFFFFu, tm0, 1));
        tm0 = fmaxf(tm0, __shfl_xor_sync(0xFFFFFFFFu, tm0, 2));
        tm1 = fmaxf(tm1, __shfl_xor_sync(0xFFFFFFFFu, tm1, 1));
        tm1 = fmaxf(tm1, __shfl_xor_sync(0xFFFFFFFFu, tm1, 2));

        const float mn0 = fmaxf(m_row0, tm0);
        const float mn1 = fmaxf(m_row1, tm1);
        const float al0 = (m_row0 > -1e29f) ? __expf(m_row0 - mn0) : 0.0f;
        const float al1 = (m_row1 > -1e29f) ? __expf(m_row1 - mn1) : 0.0f;
        l_row0 *= al0;
        l_row1 *= al1;
        const __half2 h_al0 = __float2half2_rn(al0);
        const __half2 h_al1 = __float2half2_rn(al1);
#pragma unroll
        for (int jd = 0; jd < 16; ++jd) {
            __half2 o0 = __hmul2(*reinterpret_cast<const __half2*>(&acc_o[jd][0]), h_al0);
            __half2 o1 = __hmul2(*reinterpret_cast<const __half2*>(&acc_o[jd][1]), h_al1);
            acc_o[jd][0] = *reinterpret_cast<const uint32_t*>(&o0);
            acc_o[jd][1] = *reinterpret_cast<const uint32_t*>(&o1);
        }
        m_row0 = mn0;
        m_row1 = mn1;

        float tl0 = 0.0f, tl1 = 0.0f;
        uint32_t rp[NK16][4];
#pragma unroll
        for (int kk = 0; kk < NK16; ++kk) {
            const float p00 = (s_reg[2 * kk + 0][0] > -1e29f) ? __expf(s_reg[2 * kk + 0][0] - mn0) : 0.0f;
            const float p01 = (s_reg[2 * kk + 0][1] > -1e29f) ? __expf(s_reg[2 * kk + 0][1] - mn0) : 0.0f;
            const float p02 = (s_reg[2 * kk + 1][0] > -1e29f) ? __expf(s_reg[2 * kk + 1][0] - mn0) : 0.0f;
            const float p03 = (s_reg[2 * kk + 1][1] > -1e29f) ? __expf(s_reg[2 * kk + 1][1] - mn0) : 0.0f;
            tl0 += (p00 + p01) + (p02 + p03);

            const float p10 = (s_reg[2 * kk + 0][2] > -1e29f) ? __expf(s_reg[2 * kk + 0][2] - mn1) : 0.0f;
            const float p11 = (s_reg[2 * kk + 0][3] > -1e29f) ? __expf(s_reg[2 * kk + 0][3] - mn1) : 0.0f;
            const float p12 = (s_reg[2 * kk + 1][2] > -1e29f) ? __expf(s_reg[2 * kk + 1][2] - mn1) : 0.0f;
            const float p13 = (s_reg[2 * kk + 1][3] > -1e29f) ? __expf(s_reg[2 * kk + 1][3] - mn1) : 0.0f;
            tl1 += (p10 + p11) + (p12 + p13);

            rp[kk][0] = pack_fp16x2_rn(p00, p01);
            rp[kk][1] = pack_fp16x2_rn(p10, p11);
            rp[kk][2] = pack_fp16x2_rn(p02, p03);
            rp[kk][3] = pack_fp16x2_rn(p12, p13);
        }
        tl0 += __shfl_xor_sync(0xFFFFFFFFu, tl0, 1);
        tl0 += __shfl_xor_sync(0xFFFFFFFFu, tl0, 2);
        tl1 += __shfl_xor_sync(0xFFFFFFFFu, tl1, 1);
        tl1 += __shfl_xor_sync(0xFFFFFFFFu, tl1, 2);
        l_row0 += tl0;
        l_row1 += tl1;

        if (fast_fp16) fa_cp_wait0();
        else if (fast_fp8) fp8_store(s_v, raw_v8, sc_v8);
        else           load_kv_tile_gen(s_v, false, kt, n_k);
        __syncthreads();

        if (kt + BN < k_end) {
            const int n_kn = min(BN, k_end - (kt + BN));
            if (fast_fp16) load_kv_tile_async(s_k, kv.k_f16, kt + BN, n_kn);
            else if (fast_fp8) fp8_store(s_k, raw_k8, sc_k8);
            else           load_kv_tile_gen(s_k, true, kt + BN, n_kn);
        }

        // 4. acc_o[16, 128] += P[16, BN] @ V[BN, 128] via hardware ldmatrix.x4.trans + 419 TFLOPS FP16-Acc TC!
        const int v_ld_r_in = lane & 15;
        const int v_ld_c_in = (lane >> 4) & 1;
#pragma unroll
        for (int kk = 0; kk < NK16; ++kk) {
            const uint32_t a0 = rp[kk][0], a1 = rp[kk][1], a2 = rp[kk][2], a3 = rp[kk][3];
            const int vr = kk * 16 + v_ld_r_in;
#pragma unroll
            for (int j2 = 0; j2 < 8; ++j2) {
                uint32_t v[4];
                ldmatrix_x4_trans(v, &s_v[vr][swiz(vr, j2 * 2 + v_ld_c_in)]);
                mma_fp16_acc16_m16n8k16(acc_o[2 * j2 + 0], a0, a1, a2, a3, v[0], v[1]);
                mma_fp16_acc16_m16n8k16(acc_o[2 * j2 + 1], a0, a1, a2, a3, v[2], v[3]);
            }
        }
    }

    // 5. Epilogue: divide by l_row + fused sigmoid(gate_attn) + store + optional fused d_amax_out!
    const float inv0 = (l_row0 > 0.0f) ? (1.0f / l_row0) : 0.0f;
    const float inv1 = (l_row1 > 0.0f) ? (1.0f / l_row1) : 0.0f;
    float mx = 0.0f;
    if (vq0) {
        const size_t ob = ((size_t)qr0 * kInkHeads + h) * D;
#pragma unroll
        for (int jd = 0; jd < 16; ++jd) {
            const int c0 = jd * 8 + 2 * tig;
            const float2 f = __half22float2(*reinterpret_cast<const __half2*>(&acc_o[jd][0]));
            float v0 = f.x * inv0;
            float v1 = f.y * inv0;
            if (gate_in) {
                if constexpr (BF16_IO) {
                    const uint32_t g2 = *reinterpret_cast<const uint32_t*>(
                        reinterpret_cast<const __nv_bfloat16*>(gate_in) + ob + c0);
                    const float2 gf = bf16x2_to_f32x2(g2);
                    v0 *= 1.0f / (1.0f + __expf(-gf.x));
                    v1 *= 1.0f / (1.0f + __expf(-gf.y));
                } else {
                    const float* gf = reinterpret_cast<const float*>(gate_in) + ob + c0;
                    v0 *= 1.0f / (1.0f + __expf(-gf[0]));
                    v1 *= 1.0f / (1.0f + __expf(-gf[1]));
                }
            }
            out[ob + c0]     = v0;
            out[ob + c0 + 1] = v1;
            if (d_amax_out) mx = fmaxf(mx, fmaxf(fabsf(v0), fabsf(v1)));
        }
    }
    if (vq1) {
        const size_t ob = ((size_t)qr1 * kInkHeads + h) * D;
#pragma unroll
        for (int jd = 0; jd < 16; ++jd) {
            const int c0 = jd * 8 + 2 * tig;
            const float2 f = __half22float2(*reinterpret_cast<const __half2*>(&acc_o[jd][1]));
            float v0 = f.x * inv1;
            float v1 = f.y * inv1;
            if (gate_in) {
                if constexpr (BF16_IO) {
                    const uint32_t g2 = *reinterpret_cast<const uint32_t*>(
                        reinterpret_cast<const __nv_bfloat16*>(gate_in) + ob + c0);
                    const float2 gf = bf16x2_to_f32x2(g2);
                    v0 *= 1.0f / (1.0f + __expf(-gf.x));
                    v1 *= 1.0f / (1.0f + __expf(-gf.y));
                } else {
                    const float* gf = reinterpret_cast<const float*>(gate_in) + ob + c0;
                    v0 *= 1.0f / (1.0f + __expf(-gf[0]));
                    v1 *= 1.0f / (1.0f + __expf(-gf[1]));
                }
            }
            out[ob + c0]     = v0;
            out[ob + c0 + 1] = v1;
            if (d_amax_out) mx = fmaxf(mx, fmaxf(fabsf(v0), fabsf(v1)));
        }
    }
    if (d_amax_out) {
        mx = warp_max(mx);
        if (lane == 0) atomicMax(reinterpret_cast<uint32_t*>(d_amax_out), __float_as_uint(mx));
    }
#else
    (void)q_in; (void)r_in; (void)rel_proj_T; (void)sinks; (void)gate_in; (void)kv; (void)slot0; (void)p0; (void)M; (void)out; (void)d_amax_out;
#endif
}

__device__ __forceinline__ uint16_t rtx_f32x2_to_fp8x2(float a, float b) {
    return static_cast<uint16_t>(__nv_cvt_float2_to_fp8x2(make_float2(a, b), __NV_SATFINITE, __NV_E4M3));
}

// ============================================================================
// 1B. 100% Bit-Exact Zero-Spill Single-CTA-Per-Token Depth/Bank Mix (4x less GDDR7 traffic!)
//     When BF16_ST=true (prefill M>=1024): reads bst[*] & stp[0..S-2] as BF16, writes mixed/msgs as BF16 (2x less BW!)
// ============================================================================
template <bool BF16_ST = false>
__global__ void __launch_bounds__(256) k_ink_attn_res_3x_fused_1cta(
    DepthStatePtrs stp,
    const float* __restrict__ q_res, // [1 or 3, D]
    bool is_swa,
    DepthStatePtrs bst,
    const float* __restrict__ q_msg, // [4, D]
    float* __restrict__ mixed_out,   // [3, M, D] (BF16 if BF16_ST)
    float* __restrict__ msgs_out,    // [4, M, D] (BF16 if BF16_ST)
    float* __restrict__ cat3_out,    // [3, M, 2*D] (nullable)
    int M, int D,
    uint8_t* __restrict__ cat3_fp8_out = nullptr, // [3, M_pad, 2*D] FP8 (scale=0.125f, nullable)
    int M_pad = 0) {
    const int m = blockIdx.x;
    const int tid = threadIdx.x;
    const int D4 = D >> 2;
    if (m >= M) {
        if (cat3_fp8_out && m < M_pad) {
            for (int c = 0; c < 3; ++c) {
                uint32_t* dp = reinterpret_cast<uint32_t*>(cat3_fp8_out + ((size_t)c * M_pad + m) * (2 * D));
                for (int i = tid; i < 2 * D4; i += 256) dp[i] = 0u;
            }
        }
        return;
    }
    const int warp = tid >> 5;
    const int lane = tid & 31;
    const float inv_sqrt_d = rsqrtf((float)D);
    __shared__ float s_wred[8][5][8];   // [state][0=ss, 1..4=dot_c][warp]
    __shared__ float s_prob[4][8];
    __shared__ float s_sinv[8];
    __shared__ float sh[32];
    __shared__ alignas(16) uint32_t s_st[6][kInkHidden / 2];  // 27.6 KB BF16 cache -> ZERO GDDR7 re-read of bst & stp!

    // 1. Read bst ONCE for all 4 msg queries -> compute s_prob[0..3][s] (EXACT softmax * inv_rms!)
    const int B = bst.count;
    const bool use_smem_b = (cat3_fp8_out != nullptr && B <= 6 && D == kInkHidden);
    const float4* qm0 = reinterpret_cast<const float4*>(q_msg + 0 * (size_t)D);
    const float4* qm1 = reinterpret_cast<const float4*>(q_msg + 1 * (size_t)D);
    const float4* qm2 = reinterpret_cast<const float4*>(q_msg + 2 * (size_t)D);
    const float4* qm3 = reinterpret_cast<const float4*>(q_msg + 3 * (size_t)D);
    for (int s = 0; s < B; ++s) {
        float ss = 0.0f, d0 = 0.0f, d1 = 0.0f, d2 = 0.0f, d3 = 0.0f;
        if constexpr (BF16_ST) {
            const uint2* sp2 = reinterpret_cast<const uint2*>(reinterpret_cast<const __nv_bfloat16*>(bst.p[s]) + (size_t)m * D);
            for (int i = tid; i < D4; i += 256) {
                const uint2 u = sp2[i];
                if (use_smem_b) { s_st[s][2 * i + 0] = u.x; s_st[s][2 * i + 1] = u.y; }
                const float2 f0 = bf16x2_to_f32x2(u.x), f1 = bf16x2_to_f32x2(u.y);
                ss += f0.x * f0.x + f0.y * f0.y + f1.x * f1.x + f1.y * f1.y;
                const float4 q0 = qm0[i], q1 = qm1[i], q2 = qm2[i], q3 = qm3[i];
                d0 = fmaf(f0.x, q0.x, d0); d0 = fmaf(f0.y, q0.y, d0); d0 = fmaf(f1.x, q0.z, d0); d0 = fmaf(f1.y, q0.w, d0);
                d1 = fmaf(f0.x, q1.x, d1); d1 = fmaf(f0.y, q1.y, d1); d1 = fmaf(f1.x, q1.z, d1); d1 = fmaf(f1.y, q1.w, d1);
                d2 = fmaf(f0.x, q2.x, d2); d2 = fmaf(f0.y, q2.y, d2); d2 = fmaf(f1.x, q2.z, d2); d2 = fmaf(f1.y, q2.w, d2);
                d3 = fmaf(f0.x, q3.x, d3); d3 = fmaf(f0.y, q3.y, d3); d3 = fmaf(f1.x, q3.z, d3); d3 = fmaf(f1.y, q3.w, d3);
            }
        } else {
            const float4* sp4 = reinterpret_cast<const float4*>(bst.p[s] + (size_t)m * D);
            for (int i = tid; i < D4; i += 256) {
                const float4 sv = sp4[i];
                if (use_smem_b) {
                    s_st[s][2 * i + 0] = f32x2_to_bf16x2(sv.x, sv.y);
                    s_st[s][2 * i + 1] = f32x2_to_bf16x2(sv.z, sv.w);
                }
                ss += sv.x * sv.x + sv.y * sv.y + sv.z * sv.z + sv.w * sv.w;
                const float4 q0 = qm0[i], q1 = qm1[i], q2 = qm2[i], q3 = qm3[i];
                d0 = fmaf(sv.x, q0.x, d0); d0 = fmaf(sv.y, q0.y, d0); d0 = fmaf(sv.z, q0.z, d0); d0 = fmaf(sv.w, q0.w, d0);
                d1 = fmaf(sv.x, q1.x, d1); d1 = fmaf(sv.y, q1.y, d1); d1 = fmaf(sv.z, q1.z, d1); d1 = fmaf(sv.w, q1.w, d1);
                d2 = fmaf(sv.x, q2.x, d2); d2 = fmaf(sv.y, q2.y, d2); d2 = fmaf(sv.z, q2.z, d2); d2 = fmaf(sv.w, q2.w, d2);
                d3 = fmaf(sv.x, q3.x, d3); d3 = fmaf(sv.y, q3.y, d3); d3 = fmaf(sv.z, q3.z, d3); d3 = fmaf(sv.w, q3.w, d3);
            }
        }
        ss = warp_sum(ss); d0 = warp_sum(d0); d1 = warp_sum(d1); d2 = warp_sum(d2); d3 = warp_sum(d3);
        if (lane == 0) {
            s_wred[s][0][warp] = ss;
            s_wred[s][1][warp] = d0; s_wred[s][2][warp] = d1;
            s_wred[s][3][warp] = d2; s_wred[s][4][warp] = d3;
        }
    }
    __syncthreads();
    if (tid < B) {
        float ss = 0.0f;
#pragma unroll
        for (int w = 0; w < 8; ++w) ss += s_wred[tid][0][w];
        s_sinv[tid] = rsqrtf(ss / (float)D + kInkRmsEps);
    }
    __syncthreads();
    if (tid < 4) {
        const int c = tid;
        float mx = -1e30f;
        for (int s = 0; s < B; ++s) {
            float ds = 0.0f;
#pragma unroll
            for (int w = 0; w < 8; ++w) ds += s_wred[s][1 + c][w];
            const float sc = ds * s_sinv[s] * inv_sqrt_d;
            s_prob[c][s] = sc;
            mx = fmaxf(mx, sc);
        }
        float sum = 0.0f;
        for (int s = 0; s < B; ++s) {
            const float ex = expf(s_prob[c][s] - mx);
            s_prob[c][s] = ex;
            sum += ex;
        }
        const float inv_s = 1.0f / sum;
        for (int s = 0; s < B; ++s) s_prob[c][s] = s_prob[c][s] * inv_s * s_sinv[s];
    }
    __syncthreads();

    // Mix all 4 msgs from shared memory (ZERO GDDR7 re-read of bst!) + keep m0/m1/m2 in registers for FP8 packing!
    float4 r_m0[3], r_m1[3], r_m2[3];
    float ssm0 = 0.0f, ssm1 = 0.0f, ssm2 = 0.0f;
#pragma unroll
    for (int t = 0; t < 3; ++t) {
        const int i = tid + t * 256;
        if (i < D4) {
            float a0x=0,a0y=0,a0z=0,a0w=0, a1x=0,a1y=0,a1z=0,a1w=0, a2x=0,a2y=0,a2z=0,a2w=0, a3x=0,a3y=0,a3z=0,a3w=0;
            for (int s = 0; s < B; ++s) {
                float4 sv;
                if (use_smem_b) {
                    const float2 f0 = bf16x2_to_f32x2(s_st[s][2 * i + 0]);
                    const float2 f1 = bf16x2_to_f32x2(s_st[s][2 * i + 1]);
                    sv = make_float4(f0.x, f0.y, f1.x, f1.y);
                } else if constexpr (BF16_ST) {
                    const uint2 u = reinterpret_cast<const uint2*>(reinterpret_cast<const __nv_bfloat16*>(bst.p[s]) + (size_t)m * D)[i];
                    const float2 f0 = bf16x2_to_f32x2(u.x), f1 = bf16x2_to_f32x2(u.y);
                    sv = make_float4(f0.x, f0.y, f1.x, f1.y);
                } else {
                    sv = reinterpret_cast<const float4*>(bst.p[s] + (size_t)m * D)[i];
                }
                const float w0 = s_prob[0][s], w1 = s_prob[1][s], w2 = s_prob[2][s], w3 = s_prob[3][s];
                a0x=fmaf(w0,sv.x,a0x); a0y=fmaf(w0,sv.y,a0y); a0z=fmaf(w0,sv.z,a0z); a0w=fmaf(w0,sv.w,a0w);
                a1x=fmaf(w1,sv.x,a1x); a1y=fmaf(w1,sv.y,a1y); a1z=fmaf(w1,sv.z,a1z); a1w=fmaf(w1,sv.w,a1w);
                a2x=fmaf(w2,sv.x,a2x); a2y=fmaf(w2,sv.y,a2y); a2z=fmaf(w2,sv.z,a2z); a2w=fmaf(w2,sv.w,a2w);
                a3x=fmaf(w3,sv.x,a3x); a3y=fmaf(w3,sv.y,a3y); a3z=fmaf(w3,sv.z,a3z); a3w=fmaf(w3,sv.w,a3w);
            }
            const float4 mv0 = make_float4(a0x, a0y, a0z, a0w);
            const float4 mv1 = make_float4(a1x, a1y, a1z, a1w);
            const float4 mv2 = make_float4(a2x, a2y, a2z, a2w);
            if constexpr (BF16_ST) {
                __nv_bfloat16* mb = reinterpret_cast<__nv_bfloat16*>(msgs_out);
                reinterpret_cast<uint2*>(mb + ((size_t)0 * M + m) * D)[i] = make_uint2(f32x2_to_bf16x2(a0x, a0y), f32x2_to_bf16x2(a0z, a0w));
                reinterpret_cast<uint2*>(mb + ((size_t)1 * M + m) * D)[i] = make_uint2(f32x2_to_bf16x2(a1x, a1y), f32x2_to_bf16x2(a1z, a1w));
                reinterpret_cast<uint2*>(mb + ((size_t)2 * M + m) * D)[i] = make_uint2(f32x2_to_bf16x2(a2x, a2y), f32x2_to_bf16x2(a2z, a2w));
                reinterpret_cast<uint2*>(mb + ((size_t)3 * M + m) * D)[i] = make_uint2(f32x2_to_bf16x2(a3x, a3y), f32x2_to_bf16x2(a3z, a3w));
            } else {
                reinterpret_cast<float4*>(msgs_out + ((size_t)0 * M + m) * D)[i] = mv0;
                reinterpret_cast<float4*>(msgs_out + ((size_t)1 * M + m) * D)[i] = mv1;
                reinterpret_cast<float4*>(msgs_out + ((size_t)2 * M + m) * D)[i] = mv2;
                reinterpret_cast<float4*>(msgs_out + ((size_t)3 * M + m) * D)[i] = make_float4(a3x, a3y, a3z, a3w);
            }
            if (cat3_out || cat3_fp8_out) {
                r_m0[t] = mv0; r_m1[t] = mv1; r_m2[t] = mv2;
                ssm0 += a0x*a0x + a0y*a0y + a0z*a0z + a0w*a0w;
                ssm1 += a1x*a1x + a1y*a1y + a1z*a1z + a1w*a1w;
                ssm2 += a2x*a2x + a2y*a2y + a2z*a2z + a2w*a2w;
            }
        }
    }
    if (use_smem_b) __syncthreads();

    // 2. Read stp ONCE for 1 query (SWA) or 3 queries (Global)!
    const int S = stp.count;
    const bool use_smem_s = (cat3_fp8_out != nullptr && S <= 6 && D == kInkHidden);
    const int nq = is_swa ? 1 : 3;
    const float4* qr0 = reinterpret_cast<const float4*>(q_res + 0 * (size_t)D);
    const float4* qr1 = is_swa ? qr0 : reinterpret_cast<const float4*>(q_res + 1 * (size_t)D);
    const float4* qr2 = is_swa ? qr0 : reinterpret_cast<const float4*>(q_res + 2 * (size_t)D);
    for (int s = 0; s < S; ++s) {
        const bool is_bf16_s = BF16_ST && (s + 1 < S);
        float ss = 0.0f, d0 = 0.0f, d1 = 0.0f, d2 = 0.0f;
        if (is_bf16_s) {
            const uint2* sp2 = reinterpret_cast<const uint2*>(reinterpret_cast<const __nv_bfloat16*>(stp.p[s]) + (size_t)m * D);
            for (int i = tid; i < D4; i += 256) {
                const uint2 u = sp2[i];
                if (use_smem_s) { s_st[s][2 * i + 0] = u.x; s_st[s][2 * i + 1] = u.y; }
                const float2 f0 = bf16x2_to_f32x2(u.x), f1 = bf16x2_to_f32x2(u.y);
                ss += f0.x * f0.x + f0.y * f0.y + f1.x * f1.x + f1.y * f1.y;
                const float4 q0 = qr0[i];
                d0 = fmaf(f0.x, q0.x, d0); d0 = fmaf(f0.y, q0.y, d0); d0 = fmaf(f1.x, q0.z, d0); d0 = fmaf(f1.y, q0.w, d0);
                if (!is_swa) {
                    const float4 q1 = qr1[i], q2 = qr2[i];
                    d1 = fmaf(f0.x, q1.x, d1); d1 = fmaf(f0.y, q1.y, d1); d1 = fmaf(f1.x, q1.z, d1); d1 = fmaf(f1.y, q1.w, d1);
                    d2 = fmaf(f0.x, q2.x, d2); d2 = fmaf(f0.y, q2.y, d2); d2 = fmaf(f1.x, q2.z, d2); d2 = fmaf(f1.y, q2.w, d2);
                }
            }
        } else {
            const float4* sp4 = reinterpret_cast<const float4*>(stp.p[s] + (size_t)m * D);
            for (int i = tid; i < D4; i += 256) {
                const float4 sv = sp4[i];
                if (use_smem_s) {
                    s_st[s][2 * i + 0] = f32x2_to_bf16x2(sv.x, sv.y);
                    s_st[s][2 * i + 1] = f32x2_to_bf16x2(sv.z, sv.w);
                }
                ss += sv.x * sv.x + sv.y * sv.y + sv.z * sv.z + sv.w * sv.w;
                const float4 q0 = qr0[i];
                d0 = fmaf(sv.x, q0.x, d0); d0 = fmaf(sv.y, q0.y, d0); d0 = fmaf(sv.z, q0.z, d0); d0 = fmaf(sv.w, q0.w, d0);
                if (!is_swa) {
                    const float4 q1 = qr1[i], q2 = qr2[i];
                    d1 = fmaf(sv.x, q1.x, d1); d1 = fmaf(sv.y, q1.y, d1); d1 = fmaf(sv.z, q1.z, d1); d1 = fmaf(sv.w, q1.w, d1);
                    d2 = fmaf(sv.x, q2.x, d2); d2 = fmaf(sv.y, q2.y, d2); d2 = fmaf(sv.z, q2.z, d2); d2 = fmaf(sv.w, q2.w, d2);
                }
            }
        }
        ss = warp_sum(ss); d0 = warp_sum(d0);
        if (!is_swa) { d1 = warp_sum(d1); d2 = warp_sum(d2); }
        if (lane == 0) {
            s_wred[s][0][warp] = ss;
            s_wred[s][1][warp] = d0;
            if (!is_swa) { s_wred[s][2][warp] = d1; s_wred[s][3][warp] = d2; }
        }
    }
    __syncthreads();
    if (tid < S) {
        float ss = 0.0f;
#pragma unroll
        for (int w = 0; w < 8; ++w) ss += s_wred[tid][0][w];
        s_sinv[tid] = rsqrtf(ss / (float)D + kInkRmsEps);
    }
    __syncthreads();
    if (tid < nq) {
        const int c = tid;
        float mx = -1e30f;
        for (int s = 0; s < S; ++s) {
            float ds = 0.0f;
#pragma unroll
            for (int w = 0; w < 8; ++w) ds += s_wred[s][1 + c][w];
            const float sc = ds * s_sinv[s] * inv_sqrt_d;
            s_prob[c][s] = sc;
            mx = fmaxf(mx, sc);
        }
        float sum = 0.0f;
        for (int s = 0; s < S; ++s) {
            const float ex = expf(s_prob[c][s] - mx);
            s_prob[c][s] = ex;
            sum += ex;
        }
        const float inv_s = 1.0f / sum;
        for (int s = 0; s < S; ++s) s_prob[c][s] = s_prob[c][s] * inv_s * s_sinv[s];
    }
    __syncthreads();

    float4 r_s0[3], r_s1[3], r_s2[3];
    float ssc0 = 0.0f, ssc1 = 0.0f, ssc2 = 0.0f;
#pragma unroll
    for (int t = 0; t < 3; ++t) {
        const int i = tid + t * 256;
        if (i < D4) {
            float a0x=0,a0y=0,a0z=0,a0w=0, a1x=0,a1y=0,a1z=0,a1w=0, a2x=0,a2y=0,a2z=0,a2w=0;
            for (int s = 0; s < S; ++s) {
                float4 sv;
                if (use_smem_s) {
                    const float2 f0 = bf16x2_to_f32x2(s_st[s][2 * i + 0]);
                    const float2 f1 = bf16x2_to_f32x2(s_st[s][2 * i + 1]);
                    sv = make_float4(f0.x, f0.y, f1.x, f1.y);
                } else if (BF16_ST && (s + 1 < S)) {
                    const uint2 u = reinterpret_cast<const uint2*>(reinterpret_cast<const __nv_bfloat16*>(stp.p[s]) + (size_t)m * D)[i];
                    const float2 f0 = bf16x2_to_f32x2(u.x), f1 = bf16x2_to_f32x2(u.y);
                    sv = make_float4(f0.x, f0.y, f1.x, f1.y);
                } else {
                    sv = reinterpret_cast<const float4*>(stp.p[s] + (size_t)m * D)[i];
                }
                const float w0 = s_prob[0][s];
                a0x=fmaf(w0,sv.x,a0x); a0y=fmaf(w0,sv.y,a0y); a0z=fmaf(w0,sv.z,a0z); a0w=fmaf(w0,sv.w,a0w);
                if (!is_swa) {
                    const float w1 = s_prob[1][s], w2 = s_prob[2][s];
                    a1x=fmaf(w1,sv.x,a1x); a1y=fmaf(w1,sv.y,a1y); a1z=fmaf(w1,sv.z,a1z); a1w=fmaf(w1,sv.w,a1w);
                    a2x=fmaf(w2,sv.x,a2x); a2y=fmaf(w2,sv.y,a2y); a2z=fmaf(w2,sv.z,a2z); a2w=fmaf(w2,sv.w,a2w);
                }
            }
            const float4 v0 = make_float4(a0x, a0y, a0z, a0w);
            const float4 v1 = is_swa ? v0 : make_float4(a1x, a1y, a1z, a1w);
            const float4 v2 = is_swa ? v0 : make_float4(a2x, a2y, a2z, a2w);
            if constexpr (BF16_ST) {
                __nv_bfloat16* sb = reinterpret_cast<__nv_bfloat16*>(mixed_out);
                reinterpret_cast<uint2*>(sb + ((size_t)0 * M + m) * D)[i] = make_uint2(f32x2_to_bf16x2(v0.x, v0.y), f32x2_to_bf16x2(v0.z, v0.w));
                if (!is_swa || !cat3_fp8_out) {
                    reinterpret_cast<uint2*>(sb + ((size_t)1 * M + m) * D)[i] = make_uint2(f32x2_to_bf16x2(v1.x, v1.y), f32x2_to_bf16x2(v1.z, v1.w));
                    reinterpret_cast<uint2*>(sb + ((size_t)2 * M + m) * D)[i] = make_uint2(f32x2_to_bf16x2(v2.x, v2.y), f32x2_to_bf16x2(v2.z, v2.w));
                }
            } else {
                reinterpret_cast<float4*>(mixed_out + ((size_t)0 * M + m) * D)[i] = v0;
                if (!is_swa || !cat3_fp8_out) {
                    reinterpret_cast<float4*>(mixed_out + ((size_t)1 * M + m) * D)[i] = v1;
                    reinterpret_cast<float4*>(mixed_out + ((size_t)2 * M + m) * D)[i] = v2;
                }
            }
            if (cat3_out || cat3_fp8_out) {
                r_s0[t] = v0;
                ssc0 += v0.x*v0.x + v0.y*v0.y + v0.z*v0.z + v0.w*v0.w;
                if (!is_swa) {
                    r_s1[t] = v1; r_s2[t] = v2;
                    ssc1 += v1.x*v1.x + v1.y*v1.y + v1.z*v1.z + v1.w*v1.w;
                    ssc2 += v2.x*v2.x + v2.y*v2.y + v2.z*v2.z + v2.w*v2.w;
                }
            }
        }
    }
    if (cat3_out || cat3_fp8_out) {
        const float ic0 = rsqrtf(block_sum(ssc0, sh) / (float)D + kInkRmsEps);
        const float ic1 = is_swa ? ic0 : rsqrtf(block_sum(ssc1, sh) / (float)D + kInkRmsEps);
        const float ic2 = is_swa ? ic0 : rsqrtf(block_sum(ssc2, sh) / (float)D + kInkRmsEps);
        const float im0 = rsqrtf(block_sum(ssm0, sh) / (float)D + kInkRmsEps);
        const float im1 = rsqrtf(block_sum(ssm1, sh) / (float)D + kInkRmsEps);
        const float im2 = rsqrtf(block_sum(ssm2, sh) / (float)D + kInkRmsEps);
        if (cat3_fp8_out) {
            const float f_ic0 = ic0 * 8.0f, f_ic1 = ic1 * 8.0f, f_ic2 = ic2 * 8.0f;
            const float f_im0 = im0 * 8.0f, f_im1 = im1 * 8.0f, f_im2 = im2 * 8.0f;
            uint32_t* p0s = reinterpret_cast<uint32_t*>(cat3_fp8_out + ((size_t)0 * M_pad + m) * (2 * D));
            uint32_t* p0m = reinterpret_cast<uint32_t*>(cat3_fp8_out + ((size_t)0 * M_pad + m) * (2 * D) + D);
            uint32_t* p1s = reinterpret_cast<uint32_t*>(cat3_fp8_out + ((size_t)1 * M_pad + m) * (2 * D));
            uint32_t* p1m = reinterpret_cast<uint32_t*>(cat3_fp8_out + ((size_t)1 * M_pad + m) * (2 * D) + D);
            uint32_t* p2s = reinterpret_cast<uint32_t*>(cat3_fp8_out + ((size_t)2 * M_pad + m) * (2 * D));
            uint32_t* p2m = reinterpret_cast<uint32_t*>(cat3_fp8_out + ((size_t)2 * M_pad + m) * (2 * D) + D);
            auto pack4 = [](float4 v, float sc) -> uint32_t {
                const uint16_t lo = rtx_f32x2_to_fp8x2(v.x * sc, v.y * sc);
                const uint16_t hi = rtx_f32x2_to_fp8x2(v.z * sc, v.w * sc);
                return (uint32_t)lo | ((uint32_t)hi << 16);
            };
#pragma unroll
            for (int t = 0; t < 3; ++t) {
                const int i = tid + t * 256;
                if (i < D4) {
                    const uint32_t q_s0 = pack4(r_s0[t], f_ic0);
                    p0s[i] = q_s0;
                    p0m[i] = pack4(r_m0[t], f_im0);
                    p1s[i] = is_swa ? q_s0 : pack4(r_s1[t], f_ic1);
                    p1m[i] = pack4(r_m1[t], f_im1);
                    p2s[i] = is_swa ? q_s0 : pack4(r_s2[t], f_ic2);
                    p2m[i] = pack4(r_m2[t], f_im2);
                }
            }
        } else {
            float4* c0s = reinterpret_cast<float4*>(cat3_out + ((size_t)0 * M + m) * (2 * D));
            float4* c0m = reinterpret_cast<float4*>(cat3_out + ((size_t)0 * M + m) * (2 * D) + D);
            float4* c1s = reinterpret_cast<float4*>(cat3_out + ((size_t)1 * M + m) * (2 * D));
            float4* c1m = reinterpret_cast<float4*>(cat3_out + ((size_t)1 * M + m) * (2 * D) + D);
            float4* c2s = reinterpret_cast<float4*>(cat3_out + ((size_t)2 * M + m) * (2 * D));
            float4* c2m = reinterpret_cast<float4*>(cat3_out + ((size_t)2 * M + m) * (2 * D) + D);
#pragma unroll
            for (int t = 0; t < 3; ++t) {
                const int i = tid + t * 256;
                if (i < D4) {
                    const float4 s0 = r_s0[t], m0 = r_m0[t];
                    c0s[i] = make_float4(s0.x*ic0, s0.y*ic0, s0.z*ic0, s0.w*ic0);
                    c0m[i] = make_float4(m0.x*im0, m0.y*im0, m0.z*im0, m0.w*im0);
                    const float4 s1 = is_swa ? s0 : r_s1[t], m1 = r_m1[t];
                    c1s[i] = make_float4(s1.x*ic1, s1.y*ic1, s1.z*ic1, s1.w*ic1);
                    c1m[i] = make_float4(m1.x*im1, m1.y*im1, m1.z*im1, m1.w*im1);
                    const float4 s2 = is_swa ? s0 : r_s2[t], m2 = r_m2[t];
                    c2s[i] = make_float4(s2.x*ic2, s2.y*ic2, s2.z*ic2, s2.w*ic2);
                    c2m[i] = make_float4(m2.x*im2, m2.y*im2, m2.z*im2, m2.w*im2);
                }
            }
        }
    }
}

// ============================================================================
// 1C. Single-Pass Shared-Mem Depth Mix + RMSNorm + Optional Fused amax (2x less GDDR7 on non-looped visits!)
//     When BF16_ST=true: reads stp.p[0..S-2] as BF16 and stp.p[S-1] (d_cur) as FP32!
// ============================================================================
template <bool BF16_ST = false>
__global__ void __launch_bounds__(256) k_ink_attn_res_mix_1pass(
    DepthStatePtrs stp,
    const float* __restrict__ q_res,  // [nq, D]
    float* __restrict__ out,          // [nq, M, D]
    int nq, int M, int D,
    const float* __restrict__ nw0 = nullptr,
    const float* __restrict__ nw1 = nullptr,
    const float* __restrict__ nw2 = nullptr,
    float* __restrict__ d_amax = nullptr) { // [3] optional fused amax for FP8 dense_qkv!
    const int m = blockIdx.x;
    if (m >= M) return;
    const int tid = threadIdx.x;
    const int warp = tid >> 5;
    const int lane = tid & 31;
    const int D4 = D >> 2;
    const int S = stp.count;
    const float inv_sqrt_d = rsqrtf((float)D);
    __shared__ float s_wred[8][4][8];
    __shared__ float s_sinv[8];
    __shared__ float s_prob[3][8];
    __shared__ float sh[32];
    __shared__ alignas(16) uint32_t s_st[6][kInkHidden / 2];
    const bool use_smem = (S <= 6 && D == kInkHidden);

    const float4* q0 = reinterpret_cast<const float4*>(q_res + 0 * (size_t)D);
    const float4* q1 = (nq > 1) ? reinterpret_cast<const float4*>(q_res + 1 * (size_t)D) : q0;
    const float4* q2 = (nq > 2) ? reinterpret_cast<const float4*>(q_res + 2 * (size_t)D) : q0;
    for (int s = 0; s < S; ++s) {
        const bool is_bf16_s = BF16_ST && (s + 1 < S);
        float ss = 0.0f, d0 = 0.0f, d1 = 0.0f, d2 = 0.0f;
        if (is_bf16_s) {
            const uint2* sp2 = reinterpret_cast<const uint2*>(reinterpret_cast<const __nv_bfloat16*>(stp.p[s]) + (size_t)m * D);
            for (int i = tid; i < D4; i += 256) {
                const uint2 u = sp2[i];
                if (use_smem) { s_st[s][2 * i + 0] = u.x; s_st[s][2 * i + 1] = u.y; }
                const float2 f0 = bf16x2_to_f32x2(u.x), f1 = bf16x2_to_f32x2(u.y);
                ss += f0.x * f0.x + f0.y * f0.y + f1.x * f1.x + f1.y * f1.y;
                const float4 vq0 = q0[i];
                d0 = fmaf(f0.x, vq0.x, d0); d0 = fmaf(f0.y, vq0.y, d0); d0 = fmaf(f1.x, vq0.z, d0); d0 = fmaf(f1.y, vq0.w, d0);
                if (nq > 1) {
                    const float4 vq1 = q1[i];
                    d1 = fmaf(f0.x, vq1.x, d1); d1 = fmaf(f0.y, vq1.y, d1); d1 = fmaf(f1.x, vq1.z, d1); d1 = fmaf(f1.y, vq1.w, d1);
                }
                if (nq > 2) {
                    const float4 vq2 = q2[i];
                    d2 = fmaf(f0.x, vq2.x, d2); d2 = fmaf(f0.y, vq2.y, d2); d2 = fmaf(f1.x, vq2.z, d2); d2 = fmaf(f1.y, vq2.w, d2);
                }
            }
        } else {
            const float4* sp4 = reinterpret_cast<const float4*>(stp.p[s] + (size_t)m * D);
            for (int i = tid; i < D4; i += 256) {
                const float4 sv = sp4[i];
                if (use_smem) {
                    s_st[s][2 * i + 0] = f32x2_to_bf16x2(sv.x, sv.y);
                    s_st[s][2 * i + 1] = f32x2_to_bf16x2(sv.z, sv.w);
                }
                ss += sv.x * sv.x + sv.y * sv.y + sv.z * sv.z + sv.w * sv.w;
                const float4 vq0 = q0[i];
                d0 = fmaf(sv.x, vq0.x, d0); d0 = fmaf(sv.y, vq0.y, d0); d0 = fmaf(sv.z, vq0.z, d0); d0 = fmaf(sv.w, vq0.w, d0);
                if (nq > 1) {
                    const float4 vq1 = q1[i];
                    d1 = fmaf(sv.x, vq1.x, d1); d1 = fmaf(sv.y, vq1.y, d1); d1 = fmaf(sv.z, vq1.z, d1); d1 = fmaf(sv.w, vq1.w, d1);
                }
                if (nq > 2) {
                    const float4 vq2 = q2[i];
                    d2 = fmaf(sv.x, vq2.x, d2); d2 = fmaf(sv.y, vq2.y, d2); d2 = fmaf(sv.z, vq2.z, d2); d2 = fmaf(sv.w, vq2.w, d2);
                }
            }
        }
        ss = warp_sum(ss); d0 = warp_sum(d0);
        if (nq > 1) d1 = warp_sum(d1);
        if (nq > 2) d2 = warp_sum(d2);
        if (lane == 0) {
            s_wred[s][0][warp] = ss; s_wred[s][1][warp] = d0;
            if (nq > 1) s_wred[s][2][warp] = d1;
            if (nq > 2) s_wred[s][3][warp] = d2;
        }
    }
    __syncthreads();
    if (tid < S) {
        float ss = 0.0f;
#pragma unroll
        for (int w = 0; w < 8; ++w) ss += s_wred[tid][0][w];
        s_sinv[tid] = rsqrtf(ss / (float)D + kInkRmsEps);
    }
    __syncthreads();
    if (tid < nq) {
        const int c = tid;
        float mx = -1e30f;
        for (int s = 0; s < S; ++s) {
            float ds = 0.0f;
#pragma unroll
            for (int w = 0; w < 8; ++w) ds += s_wred[s][1 + c][w];
            const float sc = ds * s_sinv[s] * inv_sqrt_d;
            s_prob[c][s] = sc;
            mx = fmaxf(mx, sc);
        }
        float sum = 0.0f;
        for (int s = 0; s < S; ++s) { const float e = expf(s_prob[c][s] - mx); s_prob[c][s] = e; sum += e; }
        const float inv_s = 1.0f / sum;
        for (int s = 0; s < S; ++s) s_prob[c][s] = s_prob[c][s] * inv_s * s_sinv[s];
    }
    __syncthreads();

    const float* nws[3] = {nw0, nw1, nw2};
    for (int c = 0; c < nq; ++c) {
        float4 r_v[3];
        float ss = 0.0f;
#pragma unroll
        for (int t = 0; t < 3; ++t) {
            const int i = tid + t * 256;
            if (i < D4) {
                float ax = 0.0f, ay = 0.0f, az = 0.0f, aw = 0.0f;
                for (int s = 0; s < S; ++s) {
                    float4 sv;
                    if (use_smem) {
                        const float2 f0 = bf16x2_to_f32x2(s_st[s][2 * i + 0]);
                        const float2 f1 = bf16x2_to_f32x2(s_st[s][2 * i + 1]);
                        sv = make_float4(f0.x, f0.y, f1.x, f1.y);
                    } else if (BF16_ST && (s + 1 < S)) {
                        const uint2 u = reinterpret_cast<const uint2*>(reinterpret_cast<const __nv_bfloat16*>(stp.p[s]) + (size_t)m * D)[i];
                        const float2 f0 = bf16x2_to_f32x2(u.x), f1 = bf16x2_to_f32x2(u.y);
                        sv = make_float4(f0.x, f0.y, f1.x, f1.y);
                    } else {
                        sv = reinterpret_cast<const float4*>(stp.p[s] + (size_t)m * D)[i];
                    }
                    const float w = s_prob[c][s];
                    ax = fmaf(w, sv.x, ax); ay = fmaf(w, sv.y, ay); az = fmaf(w, sv.z, az); aw = fmaf(w, sv.w, aw);
                }
                r_v[t] = make_float4(ax, ay, az, aw);
                ss += ax * ax + ay * ay + az * az + aw * aw;
            }
        }
        float4* dst = reinterpret_cast<float4*>(out + ((size_t)c * M + m) * D);
        const float* nw = nws[c];
        const float inv = nw ? rsqrtf(block_sum(ss, sh) / (float)D + kInkRmsEps) : 1.0f;
        const float4* w4 = nw ? reinterpret_cast<const float4*>(nw) : nullptr;
        float mx = 0.0f;
#pragma unroll
        for (int t = 0; t < 3; ++t) {
            const int i = tid + t * 256;
            if (i < D4) {
                float4 v = r_v[t];
                if (w4) {
                    const float4 wv = w4[i];
                    v = make_float4(v.x * inv * wv.x, v.y * inv * wv.y, v.z * inv * wv.z, v.w * inv * wv.w);
                }
                dst[i] = v;
                if (d_amax) mx = fmaxf(mx, fmaxf(fmaxf(fabsf(v.x), fabsf(v.y)), fmaxf(fabsf(v.z), fabsf(v.w))));
            }
        }
        if (d_amax) {
            mx = warp_max(mx);
            if (lane == 0) sh[warp] = mx;
            __syncthreads();
            if (tid == 0) {
                float g = fmaxf(fmaxf(fmaxf(sh[0], sh[1]), fmaxf(sh[2], sh[3])),
                                fmaxf(fmaxf(sh[4], sh[5]), fmaxf(sh[6], sh[7])));
                atomicMax(reinterpret_cast<uint32_t*>(d_amax + c), __float_as_uint(g));
            }
            __syncthreads();
        }
    }
}

// ============================================================================
// 2. Fused Prefill MoE Reduce8 + 2D-Parallel ShortConv4 + Residual Add
//    Eliminates 19 MB d_moe_raw write+read per visit (1.8 GB/chunk saved!)
//    Grid: dim3((2304 + 255) / 256, (M + 15) / 16), Block: 256
// ============================================================================
__global__ void __launch_bounds__(256) k_ink_moe_reduce8_sconv_seq_par(
    const float* __restrict__ y_pair,     // [M, 8, 2304]
    const float* __restrict__ static_w,   // [2304, 4]
    const float* __restrict__ history,    // [3, 2304] (read-only!)
    float* __restrict__ cur_io,           // [M, 2304] in/out
    int M) {
    const int c  = blockIdx.x * 256 + threadIdx.x;
    const int m0 = blockIdx.y * 16;
    if (c >= kInkHidden || m0 >= M) return;
    const int m_end = min(M, m0 + 16);

    const float4 sw = reinterpret_cast<const float4*>(static_w)[c];
    constexpr float scale = 0.44721359549995793f;

    auto reduce_at = [&](int idx) -> float {
        if (idx < 0) return history[(size_t)(3 + idx) * kInkHidden + c];
        const float* yp = y_pair + (size_t)idx * kInkTopK * kInkHidden + c;
        float s = 0.0f;
#pragma unroll
        for (int k = 0; k < kInkTopK; ++k) s += yp[(size_t)k * kInkHidden];
        return s;
    };

    float h0 = reduce_at(m0 - 3);
    float h1 = reduce_at(m0 - 2);
    float h2 = reduce_at(m0 - 1);

    for (int m = m0; m < m_end; ++m) {
        const float cur = reduce_at(m);
        const float acc = h0 * sw.x + h1 * sw.y + h2 * sw.z + cur * sw.w;
        cur_io[(size_t)m * kInkHidden + c] += (cur + acc) * scale;
        h0 = h1; h1 = h2; h2 = cur;
    }
}

__global__ void __launch_bounds__(256) k_ink_moe_reduce8_save_hist(
    const float* __restrict__ y_pair,     // [M, 8, 2304]
    float* __restrict__ history,          // [3, 2304]
    int M) {
    const int c = blockIdx.x * 256 + threadIdx.x;
    if (c >= kInkHidden) return;
    for (int j = 0; j < 3; ++j) {
        const int idx = M - 3 + j;
        const float* yp = y_pair + (size_t)idx * kInkTopK * kInkHidden + c;
        float s = 0.0f;
#pragma unroll
        for (int k = 0; k < kInkTopK; ++k) s += yp[(size_t)k * kInkHidden];
        history[(size_t)j * kInkHidden + c] = s;
    }
}

// ============================================================================
// 3. Fused Prefill Q/K/V/R Clip + V RMSNorm + Global Tau + KV Write (1 launch!)
//    Eliminates separate k_ink_kv_write launch & d_k/d_v reload (48 visits!)
// ============================================================================
__global__ void __launch_bounds__(128) k_ink_qkvr_clip_tau_kv_write(
    float* __restrict__ q,               // [M, 24, 128]
    const float* __restrict__ q_scale, int q_scale_elems,
    const float* __restrict__ k_in,      // [M, 8, 128] (post-sconv K)
    const float* __restrict__ v_in,      // [M, 8, 128] (post-sconv V)
    float* __restrict__ r,               // [M, 24, 16]
    const float* __restrict__ r_scale, int r_scale_elems,
    const int32_t* __restrict__ pos_arr, // [M]
    int slot0,
    InkKvVisitDesc kv,
    bool is_global,
    int M) {
    const int m   = blockIdx.y;
    const int blk = blockIdx.x;  // 0..63 (24 Q + 8 K + 8 V + 24 R)
    const int tid = threadIdx.x; // 0..127
    if (m >= M) return;

    const int p = pos_arr[m];
    float tau = 1.0f;
    if (is_global) {
        const float ratio = (float)(p + 1) * (1.0f / (float)kInkLogFloor);
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
        qp[tid] = val * (fminf(1.0f, kInkQkClip / (sqrtf(ss) + 1e-6f)) * tau);
    } else if (blk < 40) {
        const bool is_k = (blk < 32);
        const int kh = is_k ? (blk - 24) : (blk - 32);
        const float* src = is_k ? k_in : v_in;
        float val = src[((size_t)m * kInkKVHeads + kh) * kInkHeadDim + tid];
        float ss = block_sum(val * val, sh);
        if (is_k) val *= fminf(1.0f, kInkQkClip / (sqrtf(ss) + 1e-6f));
        else      val *= rsqrtf(ss * (1.0f / (float)kInkHeadDim) + kInkRmsEps);
        if (p >= 0 && (kv.is_swa || p < kv.cap_pos)) {
            const int p_idx = kv.is_swa ? (p % kv.cap_pos) : p;
            ink_kv_store_128(kv, slot0, p_idx, kh, tid, val, is_k, sh);
        }
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
            rp[tid] = val * (fminf(1.0f, kInkRelClip / (sqrtf(ss) + 1e-6f)) * tau);
        }
    }
}

// ============================================================================
// 3B. Ultra-Fast Warp-Level 1-Launch Prefill Post-QKV Kernel (ZERO __syncthreads, ZERO d_k/d_v!)
//     Replaces 6 launches (sconv_k, save_hist_k, sconv_v, save_hist_v, qkvr_clip_tau, kv_write) with 2!
//     Grid: M tokens, Block: 256 (8 warps = 1 warp/KV-head, 3 Q/R heads/warp, 100% float4 coalesced!)
// ============================================================================
__global__ void __launch_bounds__(256) k_ink_post_qkv_prefill_sm120(
    float* __restrict__ q,               // [M, 24, 128] in/out
    const float* __restrict__ q_scale, int q_scale_elems,
    const float* __restrict__ k_raw,     // [M, 8, 128]
    const float* __restrict__ v_raw,     // [M, 8, 128]
    const float* __restrict__ sw_k,      // [1024, 4]
    const float* __restrict__ sw_v,      // [1024, 4]
    const float* __restrict__ k_dyn,     // [M, 8, 4] or nullptr
    const float* __restrict__ v_dyn,     // [M, 8, 4] or nullptr
    const float* __restrict__ hist_k,    // [3, 1024] read-only!
    const float* __restrict__ hist_v,    // [3, 1024] read-only!
    float* __restrict__ r,               // [M, 24, 16] in/out
    const float* __restrict__ r_scale, int r_scale_elems,
    int p0,
    int slot0,
    InkKvVisitDesc kv,
    bool is_global,
    int M) {
    const int m = blockIdx.x;
    if (m >= M) return;
    const int tid  = threadIdx.x;
    const int warp = tid >> 5;   // 0..7
    const int lane = tid & 31;   // 0..31 -> 4 dims/lane = 128 dims!
    const int p = p0 + m;
    float tau = 1.0f;
    if (is_global) {
        const float ratio = (float)(p + 1) * (1.0f / (float)kInkLogFloor);
        tau = 1.0f + kInkLogAlpha * logf(fmaxf(1.0f, ratio));
    }

    // 1. Q heads: 24 heads / 8 warps = 3 heads/warp (128-bit float4 coalesced!)
#pragma unroll
    for (int step = 0; step < 3; ++step) {
        const int h = warp + step * 8;
        float4* qp4 = reinterpret_cast<float4*>(q + ((size_t)m * kInkHeads + h) * kInkHeadDim);
        float4 qv = qp4[lane];
        if (q_scale) {
            const float4 qs = reinterpret_cast<const float4*>(
                q_scale + ((q_scale_elems == kInkHeadDim) ? 0 : h * kInkHeadDim))[lane];
            qv.x *= qs.x; qv.y *= qs.y; qv.z *= qs.z; qv.w *= qs.w;
        }
        const float ss = warp_sum(qv.x * qv.x + qv.y * qv.y + qv.z * qv.z + qv.w * qv.w);
        const float fac = fminf(1.0f, kInkQkClip / (sqrtf(ss) + 1e-6f)) * tau;
        qp4[lane] = make_float4(qv.x * fac, qv.y * fac, qv.z * fac, qv.w * fac);
    }

    // 2. K & V heads: 1 warp per KV head kh = warp (0..7) -> ShortConv4 + Clip/RMSNorm -> Direct FP16 KV write!
    const bool kv_ok = (p >= 0 && (kv.is_swa || p < kv.cap_pos));
    if (kv_ok) {
        const int kh = warp;
        const int p_idx = kv.is_swa ? (p % kv.cap_pos) : p;
        const int c4 = kh * 32 + lane;   // float4 index in [0..255]
        const size_t dst_base = (((size_t)slot0 * kv.cap_pos + p_idx) * kInkKVHeads + kh) * 32 + lane;

        auto do_kv = [&](const float* raw, const float* hist, const float* swp, const float* dyn,
                         __half* dst_f16, bool is_k) {
            auto load4 = [&](int idx) -> float4 {
                if (idx >= 0) return reinterpret_cast<const float4*>(raw + (size_t)idx * kInkKVDim)[c4];
                return reinterpret_cast<const float4*>(hist + (size_t)(3 + idx) * kInkKVDim)[c4];
            };
            const float4 cur = load4(m);
            const float4 h2  = load4(m - 1);
            const float4 h1  = load4(m - 2);
            const float4 h0  = load4(m - 3);
            const int c0 = c4 * 4;
            const float4 sw0 = reinterpret_cast<const float4*>(swp)[c0 + 0];
            const float4 sw1 = reinterpret_cast<const float4*>(swp)[c0 + 1];
            const float4 sw2 = reinterpret_cast<const float4*>(swp)[c0 + 2];
            const float4 sw3 = reinterpret_cast<const float4*>(swp)[c0 + 3];
            float a0 = h0.x * sw0.x + h1.x * sw0.y + h2.x * sw0.z + cur.x * sw0.w;
            float a1 = h0.y * sw1.x + h1.y * sw1.y + h2.y * sw1.z + cur.y * sw1.w;
            float a2 = h0.z * sw2.x + h1.z * sw2.y + h2.z * sw2.z + cur.z * sw2.w;
            float a3 = h0.w * sw3.x + h1.w * sw3.y + h2.w * sw3.z + cur.w * sw3.w;
            const float sc_conv = dyn ? (1.0f / 3.0f) : 0.44721359549995793f;
            if (dyn) {
                const float4 dt = reinterpret_cast<const float4*>(dyn + ((size_t)m * kInkKVHeads + kh) * 4)[0];
                a0 += cur.x * dt.x + h2.x * dt.y + h1.x * dt.z + h0.x * dt.w;
                a1 += cur.y * dt.x + h2.y * dt.y + h1.y * dt.z + h0.y * dt.w;
                a2 += cur.z * dt.x + h2.z * dt.y + h1.z * dt.z + h0.z * dt.w;
                a3 += cur.w * dt.x + h2.w * dt.y + h1.w * dt.z + h0.w * dt.w;
            }
            float v0 = (cur.x + a0) * sc_conv;
            float v1 = (cur.y + a1) * sc_conv;
            float v2 = (cur.z + a2) * sc_conv;
            float v3 = (cur.w + a3) * sc_conv;
            const float ss = warp_sum(v0 * v0 + v1 * v1 + v2 * v2 + v3 * v3);
            const float fac = is_k ? fminf(1.0f, kInkQkClip / (sqrtf(ss) + 1e-6f))
                                   : rsqrtf(ss * (1.0f / (float)kInkHeadDim) + kInkRmsEps);
            v0 = fminf(fmaxf(v0 * fac, -65504.0f), 65504.0f);
            v1 = fminf(fmaxf(v1 * fac, -65504.0f), 65504.0f);
            v2 = fminf(fmaxf(v2 * fac, -65504.0f), 65504.0f);
            v3 = fminf(fmaxf(v3 * fac, -65504.0f), 65504.0f);
            reinterpret_cast<uint2*>(dst_f16)[dst_base] = make_uint2(pack_fp16x2_rn(v0, v1), pack_fp16x2_rn(v2, v3));
        };
        do_kv(k_raw, hist_k, sw_k, k_dyn, kv.k_f16, true);
        do_kv(v_raw, hist_v, sw_v, v_dyn, kv.v_f16, false);
    }

    // 3. R heads: 24 heads / 8 warps = 3 heads/warp, lanes 0..15 (16 dims, zero sync!)
    if (lane < kInkDRel) {
#pragma unroll
        for (int step = 0; step < 3; ++step) {
            const int h = warp + step * 8;
            float* rp = r + ((size_t)m * kInkHeads + h) * kInkDRel;
            float val = rp[lane];
            if (r_scale) {
                const int si = (r_scale_elems == kInkDRel) ? lane : (h * kInkDRel + lane);
                val *= r_scale[si];
            }
            float ss = val * val;
#pragma unroll
            for (int o = 8; o >= 1; o >>= 1) ss += __shfl_xor_sync(0x0000FFFFu, ss, o);
            rp[lane] = val * (fminf(1.0f, kInkRelClip / (sqrtf(ss) + 1e-6f)) * tau);
        }
    }
}

__global__ void __launch_bounds__(256) k_ink_save_kv_hist_2x(
    const float* __restrict__ k_raw,
    const float* __restrict__ v_raw,
    float* __restrict__ hist_k,
    float* __restrict__ hist_v,
    int M) {
    const int c4 = blockIdx.x * 256 + threadIdx.x;
    if (c4 >= (kInkKVDim / 4)) return;
#pragma unroll
    for (int j = 0; j < 3; ++j) {
        const int src_m = M - 3 + j;
        reinterpret_cast<float4*>(hist_k + (size_t)j * kInkKVDim)[c4] =
            reinterpret_cast<const float4*>(k_raw + (size_t)src_m * kInkKVDim)[c4];
        reinterpret_cast<float4*>(hist_v + (size_t)j * kInkKVDim)[c4] =
            reinterpret_cast<const float4*>(v_raw + (size_t)src_m * kInkKVDim)[c4];
    }
}

// ============================================================================
// 3C. Zero-Copy BF16-IO Prefill Post-QKV & Post-WO ShortConv (2x less GDDR7, ZERO k_bf16_to_f32_fast!)
//     Reads BF16 Q/K/V/R from cublasLt directly; writes FP16 Q/R in-place for k_ink_flash_gqa_pf_sm120<BN, true>!
// ============================================================================
__global__ void __launch_bounds__(256) k_ink_post_qkv_prefill_sm120_bf16(
    __nv_bfloat16* __restrict__ q_io,     // [M, 24, 128] in: BF16, out: FP16 (in-place!)
    const float* __restrict__ q_scale, int q_scale_elems,
    const __nv_bfloat16* __restrict__ k_raw_bf16, // [M, 8, 128] BF16
    const __nv_bfloat16* __restrict__ v_raw_bf16, // [M, 8, 128] BF16
    const float* __restrict__ sw_k,
    const float* __restrict__ sw_v,
    const float* __restrict__ k_dyn,
    const float* __restrict__ v_dyn,
    const float* __restrict__ hist_k,
    const float* __restrict__ hist_v,
    __nv_bfloat16* __restrict__ r_io,     // [M, 24, 16] in: BF16, out: FP16 (in-place!)
    const float* __restrict__ r_scale, int r_scale_elems,
    int p0, int slot0, InkKvVisitDesc kv, bool is_global, int M) {
    const int m = blockIdx.x;
    if (m >= M) return;
    const int tid  = threadIdx.x;
    const int warp = tid >> 5;
    const int lane = tid & 31;
    const int p = p0 + m;
    float tau = 1.0f;
    if (is_global) {
        const float ratio = (float)(p + 1) * (1.0f / (float)kInkLogFloor);
        tau = 1.0f + kInkLogAlpha * logf(fmaxf(1.0f, ratio));
    }

    // 1. Q heads: BF16 in -> clip*tau -> FP16 out in-place!
#pragma unroll
    for (int step = 0; step < 3; ++step) {
        const int h = warp + step * 8;
        uint2* qp2 = reinterpret_cast<uint2*>(q_io + ((size_t)m * kInkHeads + h) * kInkHeadDim);
        const uint2 raw = qp2[lane];
        const float2 q0 = bf16x2_to_f32x2(raw.x);
        const float2 q1 = bf16x2_to_f32x2(raw.y);
        float4 qv = make_float4(q0.x, q0.y, q1.x, q1.y);
        if (q_scale) {
            const float4 qs = reinterpret_cast<const float4*>(
                q_scale + ((q_scale_elems == kInkHeadDim) ? 0 : h * kInkHeadDim))[lane];
            qv.x *= qs.x; qv.y *= qs.y; qv.z *= qs.z; qv.w *= qs.w;
        }
        const float ss = warp_sum(qv.x * qv.x + qv.y * qv.y + qv.z * qv.z + qv.w * qv.w);
        const float fac = fminf(1.0f, kInkQkClip / (sqrtf(ss) + 1e-6f)) * tau;
        qp2[lane] = make_uint2(
            pack_fp16x2_rn(fminf(fmaxf(qv.x * fac, -65504.0f), 65504.0f), fminf(fmaxf(qv.y * fac, -65504.0f), 65504.0f)),
            pack_fp16x2_rn(fminf(fmaxf(qv.z * fac, -65504.0f), 65504.0f), fminf(fmaxf(qv.w * fac, -65504.0f), 65504.0f)));
    }

    // 2. K & V heads: BF16 in -> ShortConv4 + Clip/RMSNorm -> FP16 KV directly!
    const bool kv_ok = (p >= 0 && (kv.is_swa || p < kv.cap_pos));
    if (kv_ok) {
        const int kh = warp;
        const int p_idx = kv.is_swa ? (p % kv.cap_pos) : p;
        const int c4 = kh * 32 + lane;
        const size_t dst_base = (((size_t)slot0 * kv.cap_pos + p_idx) * kInkKVHeads + kh) * 32 + lane;
        auto do_kv = [&](const __nv_bfloat16* raw, const float* hist, const float* swp, const float* dyn,
                         __half* dst_f16, bool is_k) {
            auto load4 = [&](int idx) -> float4 {
                if (idx >= 0) {
                    const uint2 u = reinterpret_cast<const uint2*>(raw + (size_t)idx * kInkKVDim)[c4];
                    const float2 f0 = bf16x2_to_f32x2(u.x), f1 = bf16x2_to_f32x2(u.y);
                    return make_float4(f0.x, f0.y, f1.x, f1.y);
                }
                return reinterpret_cast<const float4*>(hist + (size_t)(3 + idx) * kInkKVDim)[c4];
            };
            const float4 cur = load4(m), h2 = load4(m - 1), h1 = load4(m - 2), h0 = load4(m - 3);
            const int c0 = c4 * 4;
            const float4 sw0 = reinterpret_cast<const float4*>(swp)[c0 + 0];
            const float4 sw1 = reinterpret_cast<const float4*>(swp)[c0 + 1];
            const float4 sw2 = reinterpret_cast<const float4*>(swp)[c0 + 2];
            const float4 sw3 = reinterpret_cast<const float4*>(swp)[c0 + 3];
            float a0 = h0.x * sw0.x + h1.x * sw0.y + h2.x * sw0.z + cur.x * sw0.w;
            float a1 = h0.y * sw1.x + h1.y * sw1.y + h2.y * sw1.z + cur.y * sw1.w;
            float a2 = h0.z * sw2.x + h1.z * sw2.y + h2.z * sw2.z + cur.z * sw2.w;
            float a3 = h0.w * sw3.x + h1.w * sw3.y + h2.w * sw3.z + cur.w * sw3.w;
            const float sc_conv = dyn ? (1.0f / 3.0f) : 0.44721359549995793f;
            if (dyn) {
                const float4 dt = reinterpret_cast<const float4*>(dyn + ((size_t)m * kInkKVHeads + kh) * 4)[0];
                a0 += cur.x * dt.x + h2.x * dt.y + h1.x * dt.z + h0.x * dt.w;
                a1 += cur.y * dt.x + h2.y * dt.y + h1.y * dt.z + h0.y * dt.w;
                a2 += cur.z * dt.x + h2.z * dt.y + h1.z * dt.z + h0.z * dt.w;
                a3 += cur.w * dt.x + h2.w * dt.y + h1.w * dt.z + h0.w * dt.w;
            }
            float v0 = (cur.x + a0) * sc_conv, v1 = (cur.y + a1) * sc_conv;
            float v2 = (cur.z + a2) * sc_conv, v3 = (cur.w + a3) * sc_conv;
            const float ss = warp_sum(v0 * v0 + v1 * v1 + v2 * v2 + v3 * v3);
            const float fac = is_k ? fminf(1.0f, kInkQkClip / (sqrtf(ss) + 1e-6f))
                                   : rsqrtf(ss * (1.0f / (float)kInkHeadDim) + kInkRmsEps);
            v0 = fminf(fmaxf(v0 * fac, -65504.0f), 65504.0f);
            v1 = fminf(fmaxf(v1 * fac, -65504.0f), 65504.0f);
            v2 = fminf(fmaxf(v2 * fac, -65504.0f), 65504.0f);
            v3 = fminf(fmaxf(v3 * fac, -65504.0f), 65504.0f);
            reinterpret_cast<uint2*>(dst_f16)[dst_base] = make_uint2(pack_fp16x2_rn(v0, v1), pack_fp16x2_rn(v2, v3));
        };
        do_kv(k_raw_bf16, hist_k, sw_k, k_dyn, kv.k_f16, true);
        do_kv(v_raw_bf16, hist_v, sw_v, v_dyn, kv.v_f16, false);
    }

    // 3. R heads: BF16 in -> clip*tau -> FP16 out in-place!
    if (lane < kInkDRel) {
#pragma unroll
        for (int step = 0; step < 3; ++step) {
            const int h = warp + step * 8;
            uint16_t* rp = reinterpret_cast<uint16_t*>(r_io + ((size_t)m * kInkHeads + h) * kInkDRel);
            float val = bf16_to_f32(rp[lane]);
            if (r_scale) {
                const int si = (r_scale_elems == kInkDRel) ? lane : (h * kInkDRel + lane);
                val *= r_scale[si];
            }
            float ss = val * val;
#pragma unroll
            for (int o = 8; o >= 1; o >>= 1) ss += __shfl_xor_sync(0x0000FFFFu, ss, o);
            const float outv = fminf(fmaxf(val * (fminf(1.0f, kInkRelClip / (sqrtf(ss) + 1e-6f)) * tau), -65504.0f), 65504.0f);
            const __half hh = __float2half_rn(outv);
            rp[lane] = *reinterpret_cast<const uint16_t*>(&hh);
        }
    }
}

__global__ void __launch_bounds__(256) k_ink_save_kv_hist_2x_bf16(
    const __nv_bfloat16* __restrict__ k_raw_bf16,
    const __nv_bfloat16* __restrict__ v_raw_bf16,
    float* __restrict__ hist_k,
    float* __restrict__ hist_v,
    int M) {
    const int c4 = blockIdx.x * 256 + threadIdx.x;
    if (c4 >= (kInkKVDim / 4)) return;
#pragma unroll
    for (int j = 0; j < 3; ++j) {
        const int src_m = M - 3 + j;
        const uint2 uk = reinterpret_cast<const uint2*>(k_raw_bf16 + (size_t)src_m * kInkKVDim)[c4];
        const float2 fk0 = bf16x2_to_f32x2(uk.x), fk1 = bf16x2_to_f32x2(uk.y);
        reinterpret_cast<float4*>(hist_k + (size_t)j * kInkKVDim)[c4] = make_float4(fk0.x, fk0.y, fk1.x, fk1.y);
        const uint2 uv = reinterpret_cast<const uint2*>(v_raw_bf16 + (size_t)src_m * kInkKVDim)[c4];
        const float2 fv0 = bf16x2_to_f32x2(uv.x), fv1 = bf16x2_to_f32x2(uv.y);
        reinterpret_cast<float4*>(hist_v + (size_t)j * kInkKVDim)[c4] = make_float4(fv0.x, fv0.y, fv1.x, fv1.y);
    }
}

__global__ void __launch_bounds__(256) k_ink_sconv4_seq_par_bf16_in(
    const __nv_bfloat16* __restrict__ x_bf16, // [M, C]
    const float* __restrict__ static_w,       // [C, 4]
    const float* __restrict__ history,        // [3, C]
    float* __restrict__ out,                  // [M, C]
    int M, int C, bool add_to_out) {
    const int c  = blockIdx.x * 256 + threadIdx.x;
    const int m0 = blockIdx.y * 16;
    if (c >= C || m0 >= M) return;
    const int m_end = min(M, m0 + 16);
    const float4 sw = reinterpret_cast<const float4*>(static_w)[c];
    constexpr float scale = 0.44721359549995793f;
    const uint16_t* xb = reinterpret_cast<const uint16_t*>(x_bf16);
    auto fetch = [&](int idx) -> float {
        return idx >= 0 ? bf16_to_f32(xb[(size_t)idx * C + c]) : history[(size_t)(3 + idx) * C + c];
    };
    float h0 = fetch(m0 - 3), h1 = fetch(m0 - 2), h2 = fetch(m0 - 1);
    for (int m = m0; m < m_end; ++m) {
        const float cur = bf16_to_f32(xb[(size_t)m * C + c]);
        const float acc = h0 * sw.x + h1 * sw.y + h2 * sw.z + cur * sw.w;
        const float res = (cur + acc) * scale;
        const size_t oi = (size_t)m * C + c;
        out[oi] = add_to_out ? (out[oi] + res) : res;
        h0 = h1; h1 = h2; h2 = cur;
    }
}

__global__ void __launch_bounds__(256) k_ink_sconv4_save_hist_bf16_in(
    const __nv_bfloat16* __restrict__ x_bf16,
    float* __restrict__ history,
    int M, int C) {
    const int c = blockIdx.x * 256 + threadIdx.x;
    if (c >= C) return;
    const uint16_t* xb = reinterpret_cast<const uint16_t*>(x_bf16);
    history[0 * C + c] = bf16_to_f32(xb[(size_t)(M - 3) * C + c]);
    history[1 * C + c] = bf16_to_f32(xb[(size_t)(M - 2) * C + c]);
    history[2 * C + c] = bf16_to_f32(xb[(size_t)(M - 1) * C + c]);
}

// ============================================================================
// 3D. Wave 4 Deep-Tree Split-K Attention (stride kInkMaxTree=32, depth <= 31!) + Single-Pass M<=24 Screened Unembed
// ============================================================================
__global__ void __launch_bounds__(128) k_ink_flash_gqa_splitk_deep(
    const float* __restrict__ q,          // [M, 24, 128]
    const float* __restrict__ rel_table,  // [M, 24, 1025]
    const float* __restrict__ sinks,      // [M, 24]
    InkKvVisitDesc kv,
    const int32_t* __restrict__ slots,
    int slot0,
    const int32_t* __restrict__ q_pos,
    const float* __restrict__ tree_k,
    const float* __restrict__ tree_v,
    const uint8_t* __restrict__ tree_anc32, // [M, 32]
    const uint8_t* __restrict__ tree_dep,
    int tree_p0,
    float* __restrict__ part_m,
    float* __restrict__ part_l,
    float* __restrict__ part_o,
    int M,
    int n_splits = kInkAttnSplits) {
    const int kh = blockIdx.x;
    const int sp = blockIdx.y;
    const int m  = blockIdx.z;
    if (kh >= kInkKVHeads || sp >= n_splits || m >= M) return;

    const int tid  = threadIdx.x;
    const int warp = tid >> 5;
    const int lane = tid & 31;
    const int sl   = slots ? slots[m] : slot0;
    const int qp   = q_pos[m];
    const float scale = 0.08838834764831845f;

    const int h0 = 3 * kh;
    const float4 qv0 = *(const float4*)(q + ((size_t)m * kInkHeads + h0 + 0) * kInkHeadDim + lane * 4);
    const float4 qv1 = *(const float4*)(q + ((size_t)m * kInkHeads + h0 + 1) * kInkHeadDim + lane * 4);
    const float4 qv2 = *(const float4*)(q + ((size_t)m * kInkHeads + h0 + 2) * kInkHeadDim + lane * 4);
    const float* rt0 = rel_table + ((size_t)m * kInkHeads + h0 + 0) * (kInkRelExtent + 1);
    const float* rt1 = rel_table + ((size_t)m * kInkHeads + h0 + 1) * (kInkRelExtent + 1);
    const float* rt2 = rel_table + ((size_t)m * kInkHeads + h0 + 2) * (kInkRelExtent + 1);

    const int tp0 = tree_k ? q_pos[0] : tree_p0;
    const int k_start = kv.is_swa ? max(0, qp - kInkSwaWin + 1) : 0;
    const int k_end   = qp + 1;
    const int k_len   = max(0, k_end - k_start);
    const int per_sp  = (k_len + n_splits - 1) / n_splits;
    const int s_beg   = k_start + sp * per_sp;
    const int s_end   = min(k_end, s_beg + per_sp);

    float mw[3] = {
        (sp == 0 && warp == 0) ? sinks[(size_t)m * kInkHeads + h0 + 0] : -1e30f,
        (sp == 0 && warp == 0) ? sinks[(size_t)m * kInkHeads + h0 + 1] : -1e30f,
        (sp == 0 && warp == 0) ? sinks[(size_t)m * kInkHeads + h0 + 2] : -1e30f
    };
    float lw[3] = {
        (sp == 0 && warp == 0) ? 1.0f : 0.0f,
        (sp == 0 && warp == 0) ? 1.0f : 0.0f,
        (sp == 0 && warp == 0) ? 1.0f : 0.0f
    };
    float a[3][4] = {};

    auto attend3 = [&](float4 kv4, float4 vv4, int k_pos) {
        const int dist = min(max(qp - k_pos, 0), kInkRelExtent);
        float d0 = warp_sum(qv0.x * kv4.x + qv0.y * kv4.y + qv0.z * kv4.z + qv0.w * kv4.w) * scale + rt0[dist];
        float d1 = warp_sum(qv1.x * kv4.x + qv1.y * kv4.y + qv1.z * kv4.z + qv1.w * kv4.w) * scale + rt1[dist];
        float d2 = warp_sum(qv2.x * kv4.x + qv2.y * kv4.y + qv2.z * kv4.z + qv2.w * kv4.w) * scale + rt2[dist];
        const float sc[3] = {d0, d1, d2};
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

    for (int kp = s_beg + warp; kp < s_end; kp += 4) {
        float4 kv4, vv4;
        if (tree_k && kp >= tp0) {
            const int anc_node = (int)tree_anc32[m * kInkMaxTree + (kp - tp0)];
            const size_t tbase = ((size_t)anc_node * kInkKVHeads + kh) * kInkHeadDim + lane * 4;
            kv4 = *(const float4*)(tree_k + tbase);
            vv4 = *(const float4*)(tree_v + tbase);
        } else {
            if (!kv.is_swa && (kp < 0 || kp >= kv.cap_pos)) continue;
            const int p_idx = kv.is_swa ? (kp % kv.cap_pos) : kp;
            ink_kv_load4(kv, sl, p_idx, kh, lane, kv4, vv4);
        }
        attend3(kv4, vv4, kp);
    }

    __shared__ float sm_m[4], sm_l[4], sm_a[4][kInkHeadDim];
#pragma unroll
    for (int r = 0; r < 3; ++r) {
        if (lane == 0) { sm_m[warp] = mw[r]; sm_l[warp] = lw[r]; }
        sm_a[warp][lane * 4 + 0] = a[r][0]; sm_a[warp][lane * 4 + 1] = a[r][1];
        sm_a[warp][lane * 4 + 2] = a[r][2]; sm_a[warp][lane * 4 + 3] = a[r][3];
        __syncthreads();
        if (warp == 0) {
            float mg = fmaxf(fmaxf(sm_m[0], sm_m[1]), fmaxf(sm_m[2], sm_m[3]));
            float w0 = sm_l[0] > 0.0f ? __expf(sm_m[0] - mg) : 0.0f;
            float w1 = sm_l[1] > 0.0f ? __expf(sm_m[1] - mg) : 0.0f;
            float w2 = sm_l[2] > 0.0f ? __expf(sm_m[2] - mg) : 0.0f;
            float w3 = sm_l[3] > 0.0f ? __expf(sm_m[3] - mg) : 0.0f;
            const float lt = sm_l[0] * w0 + sm_l[1] * w1 + sm_l[2] * w2 + sm_l[3] * w3;
            const size_t idx_sp = ((size_t)m * kInkHeads + h0 + r) * n_splits + sp;
            if (lane == 0) { part_m[idx_sp] = mg; part_l[idx_sp] = lt; }
            *(float4*)(part_o + idx_sp * kInkHeadDim + lane * 4) = make_float4(
                sm_a[0][lane * 4 + 0] * w0 + sm_a[1][lane * 4 + 0] * w1 + sm_a[2][lane * 4 + 0] * w2 + sm_a[3][lane * 4 + 0] * w3,
                sm_a[0][lane * 4 + 1] * w0 + sm_a[1][lane * 4 + 1] * w1 + sm_a[2][lane * 4 + 1] * w2 + sm_a[3][lane * 4 + 1] * w3,
                sm_a[0][lane * 4 + 2] * w0 + sm_a[1][lane * 4 + 2] * w1 + sm_a[2][lane * 4 + 2] * w2 + sm_a[3][lane * 4 + 2] * w3,
                sm_a[0][lane * 4 + 3] * w0 + sm_a[1][lane * 4 + 3] * w1 + sm_a[2][lane * 4 + 3] * w2 + sm_a[3][lane * 4 + 3] * w3);
        }
        __syncthreads();
    }
}

// Single-Pass Exact Cauchy-Schwarz Screener for M <= 24 (reads 460 MB unembed plane ONCE -> 0.30 ms!)
template <int MAX_M>
__global__ void __launch_bounds__(256) k_ink_screen_pass1_multiM_rtx(
    const int8_t* __restrict__ q8,
    const float* __restrict__ sc,
    const float* __restrict__ eps,
    const float* __restrict__ h,
    const float* __restrict__ h_norm,
    float* __restrict__ dot_hat,
    int* __restrict__ max_lb_bits,
    int M, int V) {
    const int tid  = threadIdx.x;
    const int warp = tid >> 5;
    const int lane = tid & 31;
    const int row  = blockIdx.x * 8 + warp;

    const uint4* q16 = reinterpret_cast<const uint4*>(q8 + (size_t)row * kInkHidden);
    float acc[MAX_M] = {};
    constexpr int KT = (MAX_M <= 16) ? 288 : 192;
    static_assert(kInkHidden % KT == 0 && KT % 16 == 0, "KT divides 2304");
    __shared__ alignas(16) float s_h[MAX_M][KT];

    for (int k0 = 0; k0 < kInkHidden; k0 += KT) {
        for (int m = 0; m < M; ++m) {
            const float4* src4 = reinterpret_cast<const float4*>(h + (size_t)m * kInkHidden + k0);
            float4* dst4 = reinterpret_cast<float4*>(s_h[m]);
            for (int i = tid; i < (KT / 4); i += 256) dst4[i] = src4[i];
        }
        __syncthreads();
        if (row < V) {
            const uint4* qt = q16 + (k0 >> 4);
            for (int i = lane; i < (KT / 16); i += 32) {
                const uint4 v = qt[i];
                const uint32_t u[4] = {v.x, v.y, v.z, v.w};
#pragma unroll
                for (int q = 0; q < 4; ++q) {
                    const uint32_t p = u[q];
                    const float f0 = (float)(int8_t)(p & 0xFFu);
                    const float f1 = (float)(int8_t)((p >> 8) & 0xFFu);
                    const float f2 = (float)(int8_t)((p >> 16) & 0xFFu);
                    const float f3 = (float)(int8_t)(p >> 24);
                    const int idx4 = i * 4 + q;
#pragma unroll
                    for (int m = 0; m < MAX_M; ++m) {
                        if (m < M) {
                            const float4 xv = reinterpret_cast<const float4*>(s_h[m])[idx4];
                            acc[m] = fmaf(f0, xv.x, acc[m]); acc[m] = fmaf(f1, xv.y, acc[m]);
                            acc[m] = fmaf(f2, xv.z, acc[m]); acc[m] = fmaf(f3, xv.w, acc[m]);
                        }
                    }
                }
            }
        }
        if (k0 + KT < kInkHidden) __syncthreads();
    }
    if (row >= V) return;
    const float sc_r = sc[row];
    const float eps_r = eps[row];
#pragma unroll
    for (int m = 0; m < MAX_M; ++m) {
        if (m < M) {
            const float d = warp_sum(acc[m]) * sc_r;
            if (lane == 0) {
                dot_hat[(size_t)m * V + row] = d;
                if (max_lb_bits) {
                    const float lb = d - eps_r * h_norm[m];
                    uint32_t b = __float_as_uint(lb);
                    int ord = (int)(b ^ ((b >> 31) ? 0x7FFFFFFFu : 0u));
                    atomicMax(&max_lb_bits[m], ord);
                }
            }
        }
    }
}

// ============================================================================
// 4. Fused BF16 Loop Controller Kernels for M > 16 (Prefill & C64)
//    Replaces 3x TF32 GEMMs + 2x F32 buffers with BF16 cat_rms + 2 BF16 GEMMs (fused gain+forget)!
// ============================================================================
__global__ void __launch_bounds__(256) k_ink_ctrl_cat_rms_bf16(
    const float* __restrict__ cur,
    const float* __restrict__ msg,
    __nv_bfloat16* __restrict__ cat_bf16,  // [M, 2*D]
    int M, int D) {
    const int m = blockIdx.x;
    if (m >= M) return;
    const int tid = threadIdx.x;
    const int D4 = D >> 2;
    __shared__ float sh[32];
    const float4* cp4 = reinterpret_cast<const float4*>(cur + (size_t)m * D);
    const float4* mp4 = reinterpret_cast<const float4*>(msg + (size_t)m * D);
    float ssc = 0.0f, ssm = 0.0f;
    for (int i = tid; i < D4; i += 256) {
        const float4 cv = cp4[i], gv = mp4[i];
        ssc += cv.x * cv.x + cv.y * cv.y + cv.z * cv.z + cv.w * cv.w;
        ssm += gv.x * gv.x + gv.y * gv.y + gv.z * gv.z + gv.w * gv.w;
    }
    ssc = block_sum(ssc, sh);
    ssm = block_sum(ssm, sh);
    const float ic = rsqrtf(ssc / (float)D + kInkRmsEps);
    const float im = rsqrtf(ssm / (float)D + kInkRmsEps);
    uint32_t* d0 = reinterpret_cast<uint32_t*>(cat_bf16 + (size_t)m * (2 * D));
    uint32_t* d1 = reinterpret_cast<uint32_t*>(cat_bf16 + (size_t)m * (2 * D) + D);
    for (int i = tid; i < D4; i += 256) {
        const float4 cv = cp4[i], gv = mp4[i];
        d0[2 * i + 0] = pack_bf16x2_rn(cv.x * ic, cv.y * ic);
        d0[2 * i + 1] = pack_bf16x2_rn(cv.z * ic, cv.w * ic);
        d1[2 * i + 0] = pack_bf16x2_rn(gv.x * im, gv.y * im);
        d1[2 * i + 1] = pack_bf16x2_rn(gv.z * im, gv.w * im);
    }
}

__global__ void __launch_bounds__(256) k_ink_ctrl_blend_norm_fused_gf(
    const float* __restrict__ cur,
    const float* __restrict__ msg,
    const float* __restrict__ gf,      // [M, 2*D] where [0..D-1] = gain, [D..2D-1] = forget
    const float* __restrict__ norm_w,
    float* __restrict__ out,
    int M, int D,
    float* __restrict__ d_amax_out = nullptr) {
    const int m = blockIdx.x;
    if (m >= M) return;
    const int tid = threadIdx.x;
    const int D4 = D >> 2;
    __shared__ float sh[32];
    const float4* cp4 = reinterpret_cast<const float4*>(cur + (size_t)m * D);
    const float4* mp4 = reinterpret_cast<const float4*>(msg + (size_t)m * D);
    const float4* gp4 = reinterpret_cast<const float4*>(gf  + (size_t)m * (2 * D));
    const float4* fp4 = reinterpret_cast<const float4*>(gf  + (size_t)m * (2 * D) + D);
    float4* op4 = reinterpret_cast<float4*>(out + (size_t)m * D);
    float ss = 0.0f;
    for (int i = tid; i < D4; i += 256) {
        const float4 cv = cp4[i], mv = mp4[i], gv = gp4[i], fv = fp4[i];
        const float o0 = (1.0f - tanhf(fv.x)) * cv.x + tanhf(gv.x) * mv.x;
        const float o1 = (1.0f - tanhf(fv.y)) * cv.y + tanhf(gv.y) * mv.y;
        const float o2 = (1.0f - tanhf(fv.z)) * cv.z + tanhf(gv.z) * mv.z;
        const float o3 = (1.0f - tanhf(fv.w)) * cv.w + tanhf(gv.w) * mv.w;
        op4[i] = make_float4(o0, o1, o2, o3);
        ss += o0 * o0 + o1 * o1 + o2 * o2 + o3 * o3;
    }
    float mx = 0.0f;
    if (norm_w) {
        ss = block_sum(ss, sh);
        const float inv = rsqrtf(ss / (float)D + kInkRmsEps);
        const float4* w4 = reinterpret_cast<const float4*>(norm_w);
        for (int i = tid; i < D4; i += 256) {
            const float4 v = op4[i], wv = w4[i];
            const float r0 = v.x * inv * wv.x, r1 = v.y * inv * wv.y;
            const float r2 = v.z * inv * wv.z, r3 = v.w * inv * wv.w;
            op4[i] = make_float4(r0, r1, r2, r3);
            if (d_amax_out) mx = fmaxf(mx, fmaxf(fmaxf(fabsf(r0), fabsf(r1)), fmaxf(fabsf(r2), fabsf(r3))));
        }
    } else if (d_amax_out) {
        for (int i = tid; i < D4; i += 256) {
            const float4 v = op4[i];
            mx = fmaxf(mx, fmaxf(fmaxf(fabsf(v.x), fabsf(v.y)), fmaxf(fabsf(v.z), fabsf(v.w))));
        }
    }
    if (d_amax_out) {
        mx = warp_max(mx);
        if ((tid & 31) == 0) sh[tid >> 5] = mx;
        __syncthreads();
        if (tid == 0) {
            float g = fmaxf(fmaxf(fmaxf(sh[0], sh[1]), fmaxf(sh[2], sh[3])),
                            fmaxf(fmaxf(sh[4], sh[5]), fmaxf(sh[6], sh[7])));
            atomicMax(reinterpret_cast<uint32_t*>(d_amax_out), __float_as_uint(g));
        }
    }
}

// Split fused Q+R+Gate(+K+V) output [M, N_tot] into separate [M, N_i] buffers
__global__ void __launch_bounds__(256) k_ink_split_qkv_5(
    const float* __restrict__ fused,   // [M, 8576] = Q(3072) + R(384) + Gate(3072) + K(1024) + V(1024)
    float* __restrict__ q,
    float* __restrict__ r,
    float* __restrict__ gate,
    float* __restrict__ k,
    float* __restrict__ v,
    int M, bool has_kv) {
    constexpr int kRDim = kInkHeads * kInkDRel;  // 384
    const int tot = has_kv ? (kInkQDim + kRDim + kInkQDim + 2 * kInkKVDim)
                           : (kInkQDim + kRDim + kInkQDim);
    const int m = blockIdx.y;
    const int i = blockIdx.x * 256 + threadIdx.x;
    if (m >= M || i >= tot) return;
    const float val = fused[(size_t)m * tot + i];
    if (i < kInkQDim) {
        q[(size_t)m * kInkQDim + i] = val;
    } else if (i < kInkQDim + kRDim) {
        r[(size_t)m * kRDim + (i - kInkQDim)] = val;
    } else if (i < 2 * kInkQDim + kRDim) {
        gate[(size_t)m * kInkQDim + (i - kInkQDim - kRDim)] = val;
    } else if (i < 2 * kInkQDim + kRDim + kInkKVDim) {
        k[(size_t)m * kInkKVDim + (i - 2 * kInkQDim - kRDim)] = val;
    } else {
        v[(size_t)m * kInkKVDim + (i - 2 * kInkQDim - kRDim - kInkKVDim)] = val;
    }
}

// ============================================================================
// 5. High-BW C1 / Small-M (M <= 24) Kernels (100% Bit-Exact Greedy Parity!)
// ============================================================================

// Pre-convert [M, K] FP32 -> [M, K/2] __half2 ONCE per visit (saves 72x redundant conversions/expert!)
__global__ void __launch_bounds__(256) k_ink_f32_to_h2(
    const float* __restrict__ src,
    __half2* __restrict__ dst,
    size_t n_pairs) {
    INK_PDL_WAIT();   // PERF-STEP: first statement on every path (programmatic dependent launch)
    const size_t i = (size_t)blockIdx.x * 256 + threadIdx.x;
    if (i < n_pairs) {
        const float2 f = reinterpret_cast<const float2*>(src)[i];
        dst[i] = __floats2half2_rn(fminf(fmaxf(f.x, -65504.0f), 65504.0f),
                                   fminf(fmaxf(f.y, -65504.0f), 65504.0f));
    }
}

// High-Occupancy 8-feat/CTA (or 16-feat/CTA) SF4/MX8 Gate+Up+SITU with pre-converted x_h2 & MAX_M up to 16!
// Preserves EXACT dot16_h2_2acc & fmaf sum order as k_ink_moe_gu_sf4_mx8!
template <int MAX_M, int FEAT_PER_CTA = 8>
__global__ void __launch_bounds__(FEAT_PER_CTA * 8) k_ink_moe_gu_sf4_mx8_v2(
    const uint8_t* __restrict__ sec4_base,
    const InkExpertDev* __restrict__ experts,
    InkMoeTables tb,
    const __half2* __restrict__ x_h2,         // [M, 1152] pre-converted!
    float* __restrict__ h_f32) {              // [P, 1152]
    const int ai = blockIdx.y;
    if (ai >= *tb.nact) return;
    const int e = tb.active[ai];
    const int row0 = tb.off[e];
    const int n_total = tb.cnt[e];
    const InkExpertDev ed = experts[e];
    if (ed.tier > 1 || (ed.gu_code != FMT_SF4 && ed.gu_code != FMT_MX8)) return;

    const int tid  = threadIdx.x;
    const int sub  = tid >> 3;
    const int lg   = tid & 7;
    const int feat = blockIdx.x * FEAT_PER_CTA + sub;
    if (feat >= kInkInter) return;

    const uint8_t* body = sec4_base + ed.body_off;
    float w_outer = 1.0f;
    if (ed.gu_code == FMT_SF4) {
        const float2 h01 = *reinterpret_cast<const float2*>(body);
        w_outer = h01.x / h01.y;
    }
    __shared__ alignas(16) __half2 s_x[MAX_M][kInkHidden / 2];

    for (int off_m = 0; off_m < n_total; off_m += MAX_M) {
        const int n_e = min(MAX_M, n_total - off_m);
        for (int m = 0; m < n_e; ++m) {
            const uint4* src4 = reinterpret_cast<const uint4*>(x_h2 + (size_t)tb.tok_idx[row0 + off_m + m] * (kInkHidden / 2));
            uint4* dst4 = reinterpret_cast<uint4*>(s_x[m]);
            for (int i = tid; i < (kInkHidden / 8); i += FEAT_PER_CTA * 8) dst4[i] = src4[i];
        }
        __syncthreads();

        if (n_e == 1) {
            float ag0 = 0.0f, au0 = 0.0f;
            if (ed.gu_code == FMT_SF4) {
                constexpr size_t W_P_ROW = kInkHidden / 2;
                constexpr size_t W_S_ROW = kInkHidden / 16;
                const uint8_t* base_p = body + 16;
                const uint8_t* base_s = base_p + (size_t)(2 * kInkInter) * W_P_ROW;
                const uint4* gp16 = reinterpret_cast<const uint4*>(base_p + (size_t)(2 * feat) * W_P_ROW);
                const uint4* up16 = reinterpret_cast<const uint4*>(base_p + (size_t)(2 * feat + 1) * W_P_ROW);
                const uint16_t* gs2 = reinterpret_cast<const uint16_t*>(base_s + (size_t)(2 * feat) * W_S_ROW);
                const uint16_t* us2 = reinterpret_cast<const uint16_t*>(base_s + (size_t)(2 * feat + 1) * W_S_ROW);
#pragma unroll 3
                for (int i = lg; i < 72; i += 8) {
                    const uint4 gv = __ldg(gp16 + i), uv = __ldg(up16 + i);
                    const float2 sg = e4m3x2_to_f32x2(__ldg(gs2 + i));
                    const float2 su = e4m3x2_to_f32x2(__ldg(us2 + i));
                    __half2 gw[8], uw[8];
                    const __half2* xh0 = s_x[0] + i * 16;
                    sf4_decode16_h2(gv.x, gv.y, gw);
                    sf4_decode16_h2(uv.x, uv.y, uw);
                    ag0 = fmaf(dot16_h2_2acc(gw, xh0), sg.x, ag0);
                    au0 = fmaf(dot16_h2_2acc(uw, xh0), su.x, au0);
                    sf4_decode16_h2(gv.z, gv.w, gw);
                    sf4_decode16_h2(uv.z, uv.w, uw);
                    ag0 = fmaf(dot16_h2_2acc(gw, xh0 + 8), sg.y, ag0);
                    au0 = fmaf(dot16_h2_2acc(uw, xh0 + 8), su.y, au0);
                }
                ag0 *= w_outer; au0 *= w_outer;
            } else {
                constexpr size_t W_P_ROW = kInkHidden;
                constexpr size_t W_S_ROW = kInkHidden / 32;
                const uint8_t* base_p = body;
                const uint8_t* base_s = base_p + (size_t)(2 * kInkInter) * W_P_ROW;
                const uint4* gp16 = reinterpret_cast<const uint4*>(base_p + (size_t)(2 * feat) * W_P_ROW);
                const uint4* up16 = reinterpret_cast<const uint4*>(base_p + (size_t)(2 * feat + 1) * W_P_ROW);
                const uint8_t* gs = base_s + (size_t)(2 * feat) * W_S_ROW;
                const uint8_t* us = base_s + (size_t)(2 * feat + 1) * W_S_ROW;
#pragma unroll 4
                for (int i = lg; i < 144; i += 8) {
                    const float sg = ue8m0_to_f32(__ldg(gs + (i >> 1)));
                    const float su = ue8m0_to_f32(__ldg(us + (i >> 1)));
                    __half2 gw[8], uw[8];
                    mx8_decode16_h2(__ldg(gp16 + i), gw);
                    mx8_decode16_h2(__ldg(up16 + i), uw);
                    const __half2* xh = s_x[0] + i * 8;
                    ag0 = fmaf(dot16_h2_2acc(gw, xh), sg, ag0);
                    au0 = fmaf(dot16_h2_2acc(uw, xh), su, au0);
                }
            }
#pragma unroll
            for (int o = 4; o >= 1; o >>= 1) {
                ag0 += __shfl_xor_sync(0xFFFFFFFFu, ag0, o);
                au0 += __shfl_xor_sync(0xFFFFFFFFu, au0, o);
            }
            if (lg == 0) h_f32[(size_t)(row0 + off_m) * kInkInter + feat] = situ_f32(ag0, au0);
        } else {
            float ag[MAX_M] = {};
            float au[MAX_M] = {};
            if (ed.gu_code == FMT_SF4) {
                constexpr size_t W_P_ROW = kInkHidden / 2;
                constexpr size_t W_S_ROW = kInkHidden / 16;
                const uint8_t* base_p = body + 16;
                const uint8_t* base_s = base_p + (size_t)(2 * kInkInter) * W_P_ROW;
                const uint4* gp16 = reinterpret_cast<const uint4*>(base_p + (size_t)(2 * feat) * W_P_ROW);
                const uint4* up16 = reinterpret_cast<const uint4*>(base_p + (size_t)(2 * feat + 1) * W_P_ROW);
                const uint16_t* gs2 = reinterpret_cast<const uint16_t*>(base_s + (size_t)(2 * feat) * W_S_ROW);
                const uint16_t* us2 = reinterpret_cast<const uint16_t*>(base_s + (size_t)(2 * feat + 1) * W_S_ROW);
                for (int i = lg; i < 72; i += 8) {
                    const uint4 gv = __ldg(gp16 + i), uv = __ldg(up16 + i);
                    const float2 sg = e4m3x2_to_f32x2(__ldg(gs2 + i));
                    const float2 su = e4m3x2_to_f32x2(__ldg(us2 + i));
                    __half2 gw[8], uw[8];
                    sf4_decode16_h2(gv.x, gv.y, gw);
                    sf4_decode16_h2(uv.x, uv.y, uw);
#pragma unroll
                    for (int m = 0; m < MAX_M; ++m) {
                        if (m < n_e) {
                            const __half2* xh = s_x[m] + i * 16;
                            ag[m] = fmaf(dot16_h2_2acc(gw, xh), sg.x, ag[m]);
                            au[m] = fmaf(dot16_h2_2acc(uw, xh), su.x, au[m]);
                        }
                    }
                    sf4_decode16_h2(gv.z, gv.w, gw);
                    sf4_decode16_h2(uv.z, uv.w, uw);
#pragma unroll
                    for (int m = 0; m < MAX_M; ++m) {
                        if (m < n_e) {
                            const __half2* xh = s_x[m] + i * 16 + 8;
                            ag[m] = fmaf(dot16_h2_2acc(gw, xh), sg.y, ag[m]);
                            au[m] = fmaf(dot16_h2_2acc(uw, xh), su.y, au[m]);
                        }
                    }
                }
#pragma unroll
                for (int m = 0; m < MAX_M; ++m) { ag[m] *= w_outer; au[m] *= w_outer; }
            } else {
                constexpr size_t W_P_ROW = kInkHidden;
                constexpr size_t W_S_ROW = kInkHidden / 32;
                const uint8_t* base_p = body;
                const uint8_t* base_s = base_p + (size_t)(2 * kInkInter) * W_P_ROW;
                const uint4* gp16 = reinterpret_cast<const uint4*>(base_p + (size_t)(2 * feat) * W_P_ROW);
                const uint4* up16 = reinterpret_cast<const uint4*>(base_p + (size_t)(2 * feat + 1) * W_P_ROW);
                const uint8_t* gs = base_s + (size_t)(2 * feat) * W_S_ROW;
                const uint8_t* us = base_s + (size_t)(2 * feat + 1) * W_S_ROW;
                for (int i = lg; i < 144; i += 8) {
                    const float sg = ue8m0_to_f32(__ldg(gs + (i >> 1)));
                    const float su = ue8m0_to_f32(__ldg(us + (i >> 1)));
                    __half2 gw[8], uw[8];
                    mx8_decode16_h2(__ldg(gp16 + i), gw);
                    mx8_decode16_h2(__ldg(up16 + i), uw);
#pragma unroll
                    for (int m = 0; m < MAX_M; ++m) {
                        if (m < n_e) {
                            const __half2* xh = s_x[m] + i * 8;
                            ag[m] = fmaf(dot16_h2_2acc(gw, xh), sg, ag[m]);
                            au[m] = fmaf(dot16_h2_2acc(uw, xh), su, au[m]);
                        }
                    }
                }
            }
#pragma unroll
            for (int m = 0; m < MAX_M; ++m) {
                if (m < n_e) {
                    float gv = ag[m], uv = au[m];
#pragma unroll
                    for (int o = 4; o >= 1; o >>= 1) {
                        gv += __shfl_xor_sync(0xFFFFFFFFu, gv, o);
                        uv += __shfl_xor_sync(0xFFFFFFFFu, uv, o);
                    }
                    if (lg == 0) h_f32[(size_t)(row0 + off_m + m) * kInkInter + feat] = situ_f32(gv, uv);
                }
            }
        }
        if (off_m + MAX_M < n_total) __syncthreads();
    }
}

template <int MAX_M, int PAIRS_PER_CTA = 8>
__global__ void __launch_bounds__(PAIRS_PER_CTA * 8) k_ink_moe_dn_sf4_mx8_v2(
    const uint8_t* __restrict__ sec4_base,
    const InkExpertDev* __restrict__ experts,
    InkMoeTables tb,
    const __half2* __restrict__ h_h2,         // [P, 576] pre-converted!
    const float* __restrict__ topk_w,
    float* __restrict__ y_pair) {
    const int ai = blockIdx.y;
    if (ai >= *tb.nact) return;
    const int e = tb.active[ai];
    const int row0 = tb.off[e];
    const int n_total = tb.cnt[e];
    const InkExpertDev ed = experts[e];
    if (ed.tier > 1 || (ed.dn_code != FMT_SF4 && ed.dn_code != FMT_MX8)) return;

    const int tid    = threadIdx.x;
    const int sub    = tid >> 3;
    const int lg     = tid & 7;
    const int pair_r = blockIdx.x * PAIRS_PER_CTA + sub;
    if (pair_r >= kInkHidden / 2) return;

    const uint8_t* body = sec4_base + ed.body_off;
    float w_outer = 1.0f;
    if (ed.dn_code == FMT_SF4) {
        const float2 h23 = *reinterpret_cast<const float2*>(body + 8);
        w_outer = h23.x / h23.y;
    }
    __shared__ alignas(16) __half2 s_h[MAX_M][kInkInter / 2];

    for (int off_m = 0; off_m < n_total; off_m += MAX_M) {
        const int n_e = min(MAX_M, n_total - off_m);
        for (int m = 0; m < n_e; ++m) {
            const uint4* src4 = reinterpret_cast<const uint4*>(h_h2 + (size_t)(row0 + off_m + m) * (kInkInter / 2));
            uint4* dst4 = reinterpret_cast<uint4*>(s_h[m]);
            for (int i = tid; i < (kInkInter / 8); i += PAIRS_PER_CTA * 8) dst4[i] = src4[i];
        }
        __syncthreads();

        if (n_e == 1) {
            float y0 = 0.0f, y1 = 0.0f;
            if (ed.dn_code == FMT_SF4) {
                constexpr size_t W_P_ROW = kInkInter / 2;
                constexpr size_t W_S_ROW = kInkInter / 16;
                const uint8_t* base_p = body + ed.dn_off;
                const uint8_t* base_s = base_p + (size_t)kInkHidden * W_P_ROW;
                const uint4* p0 = reinterpret_cast<const uint4*>(base_p + (size_t)(2 * pair_r) * W_P_ROW);
                const uint4* p1 = reinterpret_cast<const uint4*>(base_p + (size_t)(2 * pair_r + 1) * W_P_ROW);
                const uint16_t* s0 = reinterpret_cast<const uint16_t*>(base_s + (size_t)(2 * pair_r) * W_S_ROW);
                const uint16_t* s1 = reinterpret_cast<const uint16_t*>(base_s + (size_t)(2 * pair_r + 1) * W_S_ROW);
#pragma unroll 3
                for (int i = lg; i < 36; i += 8) {
                    const uint4 v0 = __ldg(p0 + i), v1 = __ldg(p1 + i);
                    const float2 sc0 = e4m3x2_to_f32x2(__ldg(s0 + i));
                    const float2 sc1 = e4m3x2_to_f32x2(__ldg(s1 + i));
                    __half2 w0[8], w1[8];
                    const __half2* hh = s_h[0] + i * 16;
                    sf4_decode16_h2(v0.x, v0.y, w0);
                    sf4_decode16_h2(v1.x, v1.y, w1);
                    y0 = fmaf(dot16_h2_2acc(w0, hh), sc0.x, y0);
                    y1 = fmaf(dot16_h2_2acc(w1, hh), sc1.x, y1);
                    sf4_decode16_h2(v0.z, v0.w, w0);
                    sf4_decode16_h2(v1.z, v1.w, w1);
                    y0 = fmaf(dot16_h2_2acc(w0, hh + 8), sc0.y, y0);
                    y1 = fmaf(dot16_h2_2acc(w1, hh + 8), sc1.y, y1);
                }
                y0 *= w_outer; y1 *= w_outer;
            } else {
                constexpr size_t W_P_ROW = kInkInter;
                constexpr size_t W_S_ROW = kInkInter / 32;
                const uint8_t* base_p = body + ed.dn_off;
                const uint8_t* base_s = base_p + (size_t)kInkHidden * W_P_ROW;
                const uint4* p0 = reinterpret_cast<const uint4*>(base_p + (size_t)(2 * pair_r) * W_P_ROW);
                const uint4* p1 = reinterpret_cast<const uint4*>(base_p + (size_t)(2 * pair_r + 1) * W_P_ROW);
                const uint8_t* s0 = base_s + (size_t)(2 * pair_r) * W_S_ROW;
                const uint8_t* s1 = base_s + (size_t)(2 * pair_r + 1) * W_S_ROW;
#pragma unroll 4
                for (int i = lg; i < 72; i += 8) {
                    const float sc0 = ue8m0_to_f32(__ldg(s0 + (i >> 1)));
                    const float sc1 = ue8m0_to_f32(__ldg(s1 + (i >> 1)));
                    __half2 w0[8], w1[8];
                    mx8_decode16_h2(__ldg(p0 + i), w0);
                    mx8_decode16_h2(__ldg(p1 + i), w1);
                    const __half2* hh = s_h[0] + i * 8;
                    y0 = fmaf(dot16_h2_2acc(w0, hh), sc0, y0);
                    y1 = fmaf(dot16_h2_2acc(w1, hh), sc1, y1);
                }
            }
#pragma unroll
            for (int o = 4; o >= 1; o >>= 1) {
                y0 += __shfl_xor_sync(0xFFFFFFFFu, y0, o);
                y1 += __shfl_xor_sync(0xFFFFFFFFu, y1, o);
            }
            if (lg == 0) {
                const int pid = tb.pairs[row0 + off_m];
                const float rw = topk_w[pid];
                y_pair[(size_t)pid * kInkHidden + 2 * pair_r]     = y0 * rw;
                y_pair[(size_t)pid * kInkHidden + 2 * pair_r + 1] = y1 * rw;
            }
        } else {
            float a0[MAX_M] = {};
            float a1[MAX_M] = {};
            if (ed.dn_code == FMT_SF4) {
                constexpr size_t W_P_ROW = kInkInter / 2;
                constexpr size_t W_S_ROW = kInkInter / 16;
                const uint8_t* base_p = body + ed.dn_off;
                const uint8_t* base_s = base_p + (size_t)kInkHidden * W_P_ROW;
                const uint4* p0 = reinterpret_cast<const uint4*>(base_p + (size_t)(2 * pair_r) * W_P_ROW);
                const uint4* p1 = reinterpret_cast<const uint4*>(base_p + (size_t)(2 * pair_r + 1) * W_P_ROW);
                const uint16_t* s0 = reinterpret_cast<const uint16_t*>(base_s + (size_t)(2 * pair_r) * W_S_ROW);
                const uint16_t* s1 = reinterpret_cast<const uint16_t*>(base_s + (size_t)(2 * pair_r + 1) * W_S_ROW);
                for (int i = lg; i < 36; i += 8) {
                    const uint4 v0 = __ldg(p0 + i), v1 = __ldg(p1 + i);
                    const float2 sc0 = e4m3x2_to_f32x2(__ldg(s0 + i));
                    const float2 sc1 = e4m3x2_to_f32x2(__ldg(s1 + i));
                    __half2 w0[8], w1[8];
                    sf4_decode16_h2(v0.x, v0.y, w0);
                    sf4_decode16_h2(v1.x, v1.y, w1);
#pragma unroll
                    for (int m = 0; m < MAX_M; ++m) {
                        if (m < n_e) {
                            const __half2* hh = s_h[m] + i * 16;
                            a0[m] = fmaf(dot16_h2_2acc(w0, hh), sc0.x, a0[m]);
                            a1[m] = fmaf(dot16_h2_2acc(w1, hh), sc1.x, a1[m]);
                        }
                    }
                    sf4_decode16_h2(v0.z, v0.w, w0);
                    sf4_decode16_h2(v1.z, v1.w, w1);
#pragma unroll
                    for (int m = 0; m < MAX_M; ++m) {
                        if (m < n_e) {
                            const __half2* hh = s_h[m] + i * 16 + 8;
                            a0[m] = fmaf(dot16_h2_2acc(w0, hh), sc0.y, a0[m]);
                            a1[m] = fmaf(dot16_h2_2acc(w1, hh), sc1.y, a1[m]);
                        }
                    }
                }
#pragma unroll
                for (int m = 0; m < MAX_M; ++m) { a0[m] *= w_outer; a1[m] *= w_outer; }
            } else {
                constexpr size_t W_P_ROW = kInkInter;
                constexpr size_t W_S_ROW = kInkInter / 32;
                const uint8_t* base_p = body + ed.dn_off;
                const uint8_t* base_s = base_p + (size_t)kInkHidden * W_P_ROW;
                const uint4* p0 = reinterpret_cast<const uint4*>(base_p + (size_t)(2 * pair_r) * W_P_ROW);
                const uint4* p1 = reinterpret_cast<const uint4*>(base_p + (size_t)(2 * pair_r + 1) * W_P_ROW);
                const uint8_t* s0 = base_s + (size_t)(2 * pair_r) * W_S_ROW;
                const uint8_t* s1 = base_s + (size_t)(2 * pair_r + 1) * W_S_ROW;
                for (int i = lg; i < 72; i += 8) {
                    const float sc0 = ue8m0_to_f32(__ldg(s0 + (i >> 1)));
                    const float sc1 = ue8m0_to_f32(__ldg(s1 + (i >> 1)));
                    __half2 w0[8], w1[8];
                    mx8_decode16_h2(__ldg(p0 + i), w0);
                    mx8_decode16_h2(__ldg(p1 + i), w1);
#pragma unroll
                    for (int m = 0; m < MAX_M; ++m) {
                        if (m < n_e) {
                            const __half2* hh = s_h[m] + i * 8;
                            a0[m] = fmaf(dot16_h2_2acc(w0, hh), sc0, a0[m]);
                            a1[m] = fmaf(dot16_h2_2acc(w1, hh), sc1, a1[m]);
                        }
                    }
                }
            }
#pragma unroll
            for (int m = 0; m < MAX_M; ++m) {
                if (m < n_e) {
                    float y0 = a0[m], y1 = a1[m];
#pragma unroll
                    for (int o = 4; o >= 1; o >>= 1) {
                        y0 += __shfl_xor_sync(0xFFFFFFFFu, y0, o);
                        y1 += __shfl_xor_sync(0xFFFFFFFFu, y1, o);
                    }
                    if (lg == 0) {
                        const int pid = tb.pairs[row0 + off_m + m];
                        const float rw = topk_w[pid];
                        y_pair[(size_t)pid * kInkHidden + 2 * pair_r]     = y0 * rw;
                        y_pair[(size_t)pid * kInkHidden + 2 * pair_r + 1] = y1 * rw;
                    }
                }
            }
        }
        if (off_m + MAX_M < n_total) __syncthreads();
    }
}

// High-Wave 4-row/CTA (64 threads) 16-Lane Multi-Projection BF16 GEMV for M=1..24!
// 2x CTA count vs 8-row/CTA -> eliminates tail-wave stall on 170 SMs (`wo_ud`: 576 CTAs = 3.4 full waves)!
// Keeps IDENTICAL 16-lane k8 = lg + 16*t order & KT multiple of 128 -> 100% bit-exact!
template <int MAX_M>
__global__ void __launch_bounds__(64) k_ink_gemv_multi_bf16_v2(
    InkMultiProj mp,
    const float* __restrict__ X,
    int M, int K) {
    const int tid  = threadIdx.x;  // 0..63
    const int sub  = tid >> 4;     // 0..3 output rows/CTA
    const int lg   = tid & 15;     // 0..15 lanes/row (256B coalesced!)
    const int grow = blockIdx.x * 4 + sub;

    const uint8_t* Wp = nullptr;
    float* Yp = nullptr;
    int Np = 0, lrow = 0;
    const bool valid = ink_resolve_multi_proj(mp, grow, Wp, Yp, Np, lrow);
    const uint4* wr = valid ? reinterpret_cast<const uint4*>(reinterpret_cast<const uint16_t*>(Wp) + (size_t)lrow * K) : nullptr;
    float acc[MAX_M] = {};

    constexpr int KT = (MAX_M == 1) ? 2304 : ((MAX_M <= 8) ? 512 : ((MAX_M <= 16) ? 256 : 128));
    __shared__ alignas(16) float s_x[MAX_M][KT];
    for (int k0 = 0; k0 < K; k0 += KT) {
        const int kl = min(KT, K - k0);
        const int kl4 = kl >> 2;
        for (int m = 0; m < M; ++m) {
            const float4* sx = reinterpret_cast<const float4*>(X + (size_t)m * K + k0);
            float4* dx = reinterpret_cast<float4*>(s_x[m]);
            for (int i = tid; i < kl4; i += 64) dx[i] = sx[i];
        }
        __syncthreads();
        if (valid) {
            const int kl8 = kl >> 3;
            const uint4* wrt = wr + (k0 >> 3);
#pragma unroll 4
            for (int k8 = lg; k8 < kl8; k8 += 16) {
                const uint4 wv = __ldg(wrt + k8);
                const float2 w0 = bf16x2_to_f32x2(wv.x);
                const float2 w1 = bf16x2_to_f32x2(wv.y);
                const float2 w2 = bf16x2_to_f32x2(wv.z);
                const float2 w3 = bf16x2_to_f32x2(wv.w);
#pragma unroll
                for (int m = 0; m < MAX_M; ++m) {
                    if (m < M) {
                        const float4 x0 = reinterpret_cast<const float4*>(s_x[m])[k8 * 2];
                        const float4 x1 = reinterpret_cast<const float4*>(s_x[m])[k8 * 2 + 1];
                        acc[m] = fmaf(w0.x, x0.x, acc[m]); acc[m] = fmaf(w0.y, x0.y, acc[m]);
                        acc[m] = fmaf(w1.x, x0.z, acc[m]); acc[m] = fmaf(w1.y, x0.w, acc[m]);
                        acc[m] = fmaf(w2.x, x1.x, acc[m]); acc[m] = fmaf(w2.y, x1.y, acc[m]);
                        acc[m] = fmaf(w3.x, x1.z, acc[m]); acc[m] = fmaf(w3.y, x1.w, acc[m]);
                    }
                }
            }
        }
        if (k0 + KT < K) __syncthreads();
    }
    if (!valid) return;
#pragma unroll
    for (int m = 0; m < MAX_M; ++m) {
        if (m < M) {
            float v = acc[m];
#pragma unroll
            for (int o = 8; o >= 1; o >>= 1) v += __shfl_xor_sync(0xFFFFFFFFu, v, o);
            if (lg == 0) Yp[(size_t)m * Np + lrow] = v;
        }
    }
}

// High-Occupancy 4-way Split-K Router GEMV for small M (128 * 4 = 512 CTAs across 170 SMs vs 128!)
template <int MAX_M>
__global__ void __launch_bounds__(128) k_ink_router_gemv_splitk(
    const float* __restrict__ x,       // [M, 2304]
    const float* __restrict__ gate_w,  // [128, 2304]
    float* __restrict__ r_part,        // [4, M, 128]
    int M) {
    const int e  = blockIdx.x;  // 0..127
    const int sp = blockIdx.y;  // 0..3
    if (e >= kInkExperts || sp >= 4) return;
    const int tid = threadIdx.x;  // 0..127
    // 2304 / 4 = 576 floats = 144 float4s per split!
    constexpr int K4_SP = (kInkHidden / 4) / 4;  // 144
    const int base4 = sp * K4_SP;
    const float4* wr = reinterpret_cast<const float4*>(gate_w + (size_t)e * kInkHidden) + base4;
    float acc[MAX_M] = {};
    for (int i = tid; i < K4_SP; i += 128) {
        const float4 w = __ldg(wr + i);
#pragma unroll
        for (int m = 0; m < MAX_M; ++m) {
            if (m < M) {
                const float4 xv = reinterpret_cast<const float4*>(x + (size_t)m * kInkHidden)[base4 + i];
                acc[m] = fmaf(w.x, xv.x, acc[m]); acc[m] = fmaf(w.y, xv.y, acc[m]);
                acc[m] = fmaf(w.z, xv.z, acc[m]); acc[m] = fmaf(w.w, xv.w, acc[m]);
            }
        }
    }
    __shared__ float sh[8];
#pragma unroll
    for (int m = 0; m < MAX_M; ++m) {
        if (m < M) {
            float v = warp_sum(acc[m]);
            if ((tid & 31) == 0) sh[tid >> 5] = v;
            __syncthreads();
            if (tid == 0) r_part[((size_t)sp * M + m) * kInkExperts + e] = sh[0] + sh[1] + sh[2] + sh[3];
            __syncthreads();
        }
    }
}

__global__ void __launch_bounds__(128) k_ink_router_reduce_topk8(
    const float* __restrict__ r_part,  // [4, M, 128]
    const float* __restrict__ gate_b,  // [128]
    float* __restrict__ r_logits,      // [M, 128]
    int32_t* __restrict__ topk_idx,    // [M, 8]
    float* __restrict__ topk_w,        // [M, 8]
    float route_scale,
    int M) {
    const int m = blockIdx.x;
    if (m >= M) return;
    const int tid = threadIdx.x;  // 0..127 -> 1 expert/thread!
    __shared__ float s_prob[kInkExperts];
    __shared__ float s_score[kInkExperts];
    const float lg = r_part[((size_t)0 * M + m) * kInkExperts + tid]
                   + r_part[((size_t)1 * M + m) * kInkExperts + tid]
                   + r_part[((size_t)2 * M + m) * kInkExperts + tid]
                   + r_part[((size_t)3 * M + m) * kInkExperts + tid];
    r_logits[(size_t)m * kInkExperts + tid] = lg;
    const float p = 1.0f / (1.0f + __expf(-lg));
    s_prob[tid]  = p;
    s_score[tid] = p + gate_b[tid];
    __syncthreads();

    if (tid == 0) {
        int32_t idx[kInkTopK];
        float   w[kInkTopK];
        float   wsum = 0.0f;
        uint32_t used[4] = {0, 0, 0, 0};
        for (int k = 0; k < kInkTopK; ++k) {
            float bv = -1e30f;
            int bi = 0;
            for (int e = 0; e < kInkExperts; ++e) {
                if ((used[e >> 5] >> (e & 31)) & 1u) continue;
                const float v = s_score[e];
                if (v > bv) { bv = v; bi = e; }
            }
            used[bi >> 5] |= (1u << (bi & 31));
            idx[k] = bi;
            w[k]   = s_prob[bi];
            wsum  += w[k];
        }
        const float inv = (wsum > 0.0f) ? (route_scale / wsum) : 0.0f;
        for (int k = 0; k < kInkTopK; ++k) {
            topk_idx[(size_t)m * kInkTopK + k] = idx[k];
            topk_w[(size_t)m * kInkTopK + k]   = w[k] * inv;
        }
    }
}

// High-Occupancy 4-way Split-K Loop Controller Down for small M (64*4=256 CTAs for MLP, 64*4*3=768 for 3x Attn!)
template <int MAX_M, int N_CTRL>
__global__ void __launch_bounds__(128) k_ink_ctrl_down_splitk(
    const float* __restrict__ cat,    // [N_CTRL, M, 4608]
    const float* __restrict__ wd0, const float* __restrict__ wd1, const float* __restrict__ wd2,
    float* __restrict__ g_part,       // [4, N_CTRL, M, 64]
    int M) {
    const int r  = blockIdx.x;        // 0..63
    const int sp = blockIdx.y & 3;    // 0..3
    const int c  = blockIdx.y >> 2;   // 0..N_CTRL-1
    if (r >= kInkGateRank || c >= N_CTRL) return;
    const int tid = threadIdx.x;      // 0..127
    constexpr int D2 = 2 * kInkHidden;        // 4608
    constexpr int K4_SP = (D2 / 4) / 4;       // 288 float4s per split!
    const int base4 = sp * K4_SP;
    const float* wd = (c == 0) ? wd0 : (c == 1 ? wd1 : wd2);
    const float* cp = cat + (size_t)c * M * D2;
    const float4* wr = reinterpret_cast<const float4*>(wd + (size_t)r * D2) + base4;
    float acc[MAX_M] = {};
    for (int i = tid; i < K4_SP; i += 128) {
        const float4 w = __ldg(wr + i);
#pragma unroll
        for (int m = 0; m < MAX_M; ++m) {
            if (m < M) {
                const float4 x = reinterpret_cast<const float4*>(cp + (size_t)m * D2)[base4 + i];
                acc[m] = fmaf(w.x, x.x, acc[m]); acc[m] = fmaf(w.y, x.y, acc[m]);
                acc[m] = fmaf(w.z, x.z, acc[m]); acc[m] = fmaf(w.w, x.w, acc[m]);
            }
        }
    }
    __shared__ float sh[8];
#pragma unroll
    for (int m = 0; m < MAX_M; ++m) {
        if (m < M) {
            float v = warp_sum(acc[m]);
            if ((tid & 31) == 0) sh[tid >> 5] = v;
            __syncthreads();
            if (tid == 0) g_part[(((size_t)sp * N_CTRL + c) * M + m) * kInkGateRank + r] = sh[0] + sh[1] + sh[2] + sh[3];
            __syncthreads();
        }
    }
}

template <int N_CTRL>
__global__ void __launch_bounds__(64) k_ink_ctrl_down_splitk_reduce(
    const float* __restrict__ g_part, // [4, N_CTRL, M, 64]
    float* __restrict__ gate_out,     // [N_CTRL, M, 64]
    int M) {
    const int m = blockIdx.x;
    const int c = blockIdx.y;
    const int r = threadIdx.x;  // 0..63
    if (m >= M || c >= N_CTRL) return;
    const size_t stride = (size_t)N_CTRL * M * kInkGateRank;
    const size_t off = ((size_t)c * M + m) * kInkGateRank + r;
    gate_out[off] = g_part[off] + g_part[stride + off] + g_part[2 * stride + off] + g_part[3 * stride + off];
}

// ============================================================================
// 5B. Parallel Multi-CTA MoE Binning (replaces 1-CTA 2.5 ms k_ink_moe_bin at M>=256 -> 0.10 ms!)
//     AND Zero-Smem L1-Direct M=1..3 SF4/MX8 MoE GEMV (1,152 CTAs = 6.8 full waves on 170 SMs!)
// ============================================================================
__global__ void __launch_bounds__(256) k_ink_moe_bin_par_count(
    const InkExpertDev* __restrict__ ex,
    const int32_t* __restrict__ topk_idx,
    int P,
    int32_t* __restrict__ cnt) {
    __shared__ int s_cnt[128];
    const int tid = threadIdx.x;
    if (tid < 128) s_cnt[tid] = 0;
    __syncthreads();
    const int stride = gridDim.x * 256;
    for (int i = blockIdx.x * 256 + tid; i < P; i += stride) {
        const int e = topk_idx[i];
        if (ex[e].tier <= 1) atomicAdd(&s_cnt[e], 1);
    }
    __syncthreads();
    if (tid < 128 && s_cnt[tid] > 0) atomicAdd(&cnt[tid], s_cnt[tid]);
}

__global__ void __launch_bounds__(128) k_ink_moe_bin_par_prefix(
    int tile_rows,
    InkMoeTables tb,
    int32_t* __restrict__ fill_out) {
    __shared__ int s_off[128], s_act[128], s_tcnt[128];
    const int t = threadIdx.x;
    const int cv = tb.cnt[t];
    const int is_act = cv > 0 ? 1 : 0;
    const int nt = (cv + tile_rows - 1) / tile_rows;
    s_off[t] = cv; s_act[t] = is_act; s_tcnt[t] = nt;
    __syncthreads();
#pragma unroll
    for (int d = 1; d < 128; d <<= 1) {
        int a0 = 0, a1 = 0, a2 = 0;
        if (t >= d) { a0 = s_off[t - d]; a1 = s_act[t - d]; a2 = s_tcnt[t - d]; }
        __syncthreads();
        s_off[t] += a0; s_act[t] += a1; s_tcnt[t] += a2;
        __syncthreads();
    }
    const int off_e = s_off[t] - cv;
    tb.off[t] = off_e;
    fill_out[t] = off_e;
    if (t == 127) {
        tb.off[128] = s_off[127];
        *tb.nact = s_act[127];
        *tb.n_tiles = s_tcnt[127];
    }
    if (is_act) tb.active[s_act[t] - 1] = t;
    if (nt > 0) {
        const int tbase = s_tcnt[t] - nt;
        for (int ti = 0; ti < nt; ++ti) tb.tiles[tbase + ti] = make_int2(t, off_e + ti * tile_rows);
    }
}

__global__ void __launch_bounds__(256) k_ink_moe_bin_par_scatter(
    const InkExpertDev* __restrict__ ex,
    const int32_t* __restrict__ topk_idx,
    int P,
    int32_t* __restrict__ fill_io,
    InkMoeTables tb) {
    const int stride = gridDim.x * 256;
    for (int i = blockIdx.x * 256 + threadIdx.x; i < P; i += stride) {
        const int e = topk_idx[i];
        if (ex[e].tier > 1) continue;
        const int pos = atomicAdd(&fill_io[e], 1);
        tb.pairs[pos] = i;
        tb.tok_idx[pos] = i / kInkTopK;
    }
}

// High-Wave Zero-Smem L1-Direct M=1 SF4/MX8 Gate+Up+SITU (256 threads = 8 warps = 16 features/CTA x 16 lanes/feat, 576 CTAs!)
// Issues 2x upfront register-pipelined uint4 loads (4x uint4 = 64B in flight/thread = 1 KB/warp!) -> saturates GDDR7!
// PERF-STEP: the L2 prefetch of the routed expert's down matrix, issued by the gate+up CTAs of the same expert (72 CTAs per slot, each one slice, one bulk-prefetch
// instruction per 16 KiB): the down kernel that follows then reads it from L2. It reads nothing into registers and changes no result. pf_dn = 0 turns it off.
__device__ __forceinline__ void ink_l2_bulk_prefetch(const void* p, unsigned n) {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900) && defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ >= 12)
    asm volatile("cp.async.bulk.prefetch.L2.global [%0], %1;" ::"l"(reinterpret_cast<unsigned long long>(p)), "r"(n) : "memory");
#else
    (void)p; (void)n;
#endif
}
__device__ __forceinline__ void ink_moe_m1_prefetch_dn(const InkExpertDev& ed, const uint8_t* body) {
    const unsigned total = (ed.dn_code == FMT_SF4) ? (unsigned)(kInkHidden * (kInkInter / 2 + kInkInter / 16))
                                                   : (unsigned)(kInkHidden * (kInkInter + kInkInter / 32));
    const unsigned nb = gridDim.x;
    const unsigned per = (((total + nb - 1) / nb) + 15u) & ~15u;
    const unsigned off = blockIdx.x * per;
    if (off >= total) return;
    const unsigned len = (total - off) < per ? (total - off) : per;
    const uint8_t* src = body + ed.dn_off + off;
    for (unsigned o = 0; o < len; o += 16384u) ink_l2_bulk_prefetch(src + o, (len - o) < 16384u ? (len - o) : 16384u);
}

__global__ void __launch_bounds__(256) k_ink_moe_gu_m1_l1(
    const uint8_t* __restrict__ sec4_base,
    const InkExpertDev* __restrict__ experts,
    const int32_t* __restrict__ topk_idx,   // [8]
    const __half2* __restrict__ x_h2,       // [1152] L1-resident (4.6 KB!)
    __half2* __restrict__ h_h2_out,         // [8, 576] (output directly in __half2 for DN!)
    int pf_dn) {                            // PERF-STEP: 1 = prefetch the expert's down matrix into L2 from here
    INK_PDL_WAIT();   // PERF-STEP: first statement on every path (programmatic dependent launch)
    const int slot = blockIdx.y;            // 0..7
    const int tid  = threadIdx.x;           // 0..255
    const int sub  = tid >> 4;              // 0..15 features/CTA
    const int lg   = tid & 15;              // 0..15 lanes/feature (256B coalesced!)
    const int feat = blockIdx.x * 16 + sub;
    if (slot >= kInkTopK || feat >= kInkInter) return;

    const InkExpertDev ed = experts[topk_idx[slot]];
    if (ed.tier > 1 || (ed.gu_code != FMT_SF4 && ed.gu_code != FMT_MX8)) return;
    const uint8_t* body = sec4_base + ed.body_off;
    if (pf_dn != 0 && tid == 0 && (ed.dn_code == FMT_SF4 || ed.dn_code == FMT_MX8)) ink_moe_m1_prefetch_dn(ed, body);
    float ag = 0.0f, au = 0.0f;

    if (ed.gu_code == FMT_SF4) {
        const float2 h01 = *reinterpret_cast<const float2*>(body);
        const float w_outer = h01.x / h01.y;
        constexpr size_t W_P_ROW = kInkHidden / 2;
        constexpr size_t W_S_ROW = kInkHidden / 16;
        const uint8_t* base_p = body + 16;
        const uint8_t* base_s = base_p + (size_t)(2 * kInkInter) * W_P_ROW;
        const uint4* gp16 = reinterpret_cast<const uint4*>(base_p + (size_t)(2 * feat) * W_P_ROW);
        const uint4* up16 = reinterpret_cast<const uint4*>(base_p + (size_t)(2 * feat + 1) * W_P_ROW);
        const uint16_t* gs2 = reinterpret_cast<const uint16_t*>(base_s + (size_t)(2 * feat) * W_S_ROW);
        const uint16_t* us2 = reinterpret_cast<const uint16_t*>(base_s + (size_t)(2 * feat + 1) * W_S_ROW);

        auto proc_sf4 = [&](const uint4& gv, const uint4& uv, uint16_t gs_raw, uint16_t us_raw, int idx) {
            const float2 sg = e4m3x2_to_f32x2(gs_raw);
            const float2 su = e4m3x2_to_f32x2(us_raw);
            __half2 gw[8], uw[8];
            const __half2* xh0 = x_h2 + idx * 16;
            sf4_decode16_h2(gv.x, gv.y, gw);
            sf4_decode16_h2(uv.x, uv.y, uw);
            ag = fmaf(dot16_h2_2acc(gw, xh0), sg.x, ag);
            au = fmaf(dot16_h2_2acc(uw, xh0), su.x, au);
            sf4_decode16_h2(gv.z, gv.w, gw);
            sf4_decode16_h2(uv.z, uv.w, uw);
            ag = fmaf(dot16_h2_2acc(gw, xh0 + 8), sg.y, ag);
            au = fmaf(dot16_h2_2acc(uw, xh0 + 8), su.y, au);
        };

        // 72 uint4s / 16 lanes = 4 full waves (0, 16, 32, 48) + 8-lane tail (64..71) -> upfront 2-wave pipelining!
        const uint4 g0 = __ldg(gp16 + lg + 0),  u0 = __ldg(up16 + lg + 0);
        const uint4 g1 = __ldg(gp16 + lg + 16), u1 = __ldg(up16 + lg + 16);
        const uint16_t gs0 = __ldg(gs2 + lg + 0),  us0 = __ldg(us2 + lg + 0);
        const uint16_t gs1 = __ldg(gs2 + lg + 16), us1 = __ldg(us2 + lg + 16);
        const uint4 g2 = __ldg(gp16 + lg + 32), u2 = __ldg(up16 + lg + 32);
        const uint16_t gs2v = __ldg(gs2 + lg + 32), us2v = __ldg(us2 + lg + 32);
        proc_sf4(g0, u0, gs0, us0, lg + 0);
        const uint4 g3 = __ldg(gp16 + lg + 48), u3 = __ldg(up16 + lg + 48);
        const uint16_t gs3 = __ldg(gs2 + lg + 48), us3 = __ldg(us2 + lg + 48);
        proc_sf4(g1, u1, gs1, us1, lg + 16);
        const bool has4 = (lg < 8);
        const uint4 g4 = has4 ? __ldg(gp16 + lg + 64) : make_uint4(0u, 0u, 0u, 0u);
        const uint4 u4 = has4 ? __ldg(up16 + lg + 64) : make_uint4(0u, 0u, 0u, 0u);
        const uint16_t gs4 = has4 ? __ldg(gs2 + lg + 64) : 0u;
        const uint16_t us4 = has4 ? __ldg(us2 + lg + 64) : 0u;
        proc_sf4(g2, u2, gs2v, us2v, lg + 32);
        proc_sf4(g3, u3, gs3, us3, lg + 48);
        if (has4) proc_sf4(g4, u4, gs4, us4, lg + 64);
        ag *= w_outer; au *= w_outer;
    } else {
        constexpr size_t W_P_ROW = kInkHidden;
        constexpr size_t W_S_ROW = kInkHidden / 32;
        const uint8_t* base_p = body;
        const uint8_t* base_s = base_p + (size_t)(2 * kInkInter) * W_P_ROW;
        const uint4* gp16 = reinterpret_cast<const uint4*>(base_p + (size_t)(2 * feat) * W_P_ROW);
        const uint4* up16 = reinterpret_cast<const uint4*>(base_p + (size_t)(2 * feat + 1) * W_P_ROW);
        const uint8_t* gs = base_s + (size_t)(2 * feat) * W_S_ROW;
        const uint8_t* us = base_s + (size_t)(2 * feat + 1) * W_S_ROW;
#pragma unroll 3
        for (int i = lg; i < 144; i += 16) {
            const uint4 gv = __ldg(gp16 + i), uv = __ldg(up16 + i);
            const float sg = ue8m0_to_f32(__ldg(gs + (i >> 1)));
            const float su = ue8m0_to_f32(__ldg(us + (i >> 1)));
            __half2 gw[8], uw[8];
            mx8_decode16_h2(gv, gw);
            mx8_decode16_h2(uv, uw);
            const __half2* xh = x_h2 + i * 8;
            ag = fmaf(dot16_h2_2acc(gw, xh), sg, ag);
            au = fmaf(dot16_h2_2acc(uw, xh), su, au);
        }
    }
#pragma unroll
    for (int o = 8; o >= 1; o >>= 1) {
        ag += __shfl_xor_sync(0xFFFFFFFFu, ag, o);
        au += __shfl_xor_sync(0xFFFFFFFFu, au, o);
    }
    const float hv = situ_f32(ag, au);
    // Pack adjacent even/odd features (sub = 0,2,4..14 & 1,3,5..15 -> 16 lanes apart in same warp!) into __half2!
    const float hv_odd = __shfl_xor_sync(0xFFFFFFFFu, hv, 16);
    if (lg == 0 && (sub & 1) == 0) {
        h_h2_out[(size_t)slot * (kInkInter / 2) + (feat >> 1)] = __floats2half2_rn(hv, hv_odd);
    }
}

// High-Wave Zero-Smem L1-Direct M=1 SF4/MX8 Down (256 threads = 8 warps = 16 row-pairs/CTA x 16 lanes/pair, 576 CTAs!)
__global__ void __launch_bounds__(256) k_ink_moe_dn_m1_l1(
    const uint8_t* __restrict__ sec4_base,
    const InkExpertDev* __restrict__ experts,
    const int32_t* __restrict__ topk_idx,   // [8]
    const float* __restrict__ topk_w,       // [8]
    const __half2* __restrict__ h_h2,       // [8, 576] L1-resident!
    float* __restrict__ y_pair) { // [8, 2304]
    INK_PDL_WAIT();   // PERF-STEP: first statement on every path (programmatic dependent launch)
    const int slot   = blockIdx.y;          // 0..7
    const int tid    = threadIdx.x;
    const int sub    = tid >> 4;            // 0..15 row-pairs/CTA
    const int lg     = tid & 15;            // 0..15 lanes/pair (256B coalesced!)
    const int pair_r = blockIdx.x * 16 + sub;
    if (slot >= kInkTopK || pair_r >= kInkHidden / 2) return;

    const InkExpertDev ed = experts[topk_idx[slot]];
    if (ed.tier > 1 || (ed.dn_code != FMT_SF4 && ed.dn_code != FMT_MX8)) return;
    const uint8_t* body = sec4_base + ed.body_off;
    const __half2* hr = h_h2 + (size_t)slot * (kInkInter / 2);
    float y0 = 0.0f, y1 = 0.0f;

    if (ed.dn_code == FMT_SF4) {
        const float2 h23 = *reinterpret_cast<const float2*>(body + 8);
        const float w_outer = h23.x / h23.y;
        constexpr size_t W_P_ROW = kInkInter / 2;
        constexpr size_t W_S_ROW = kInkInter / 16;
        const uint8_t* base_p = body + ed.dn_off;
        const uint8_t* base_s = base_p + (size_t)kInkHidden * W_P_ROW;
        const uint4* p0 = reinterpret_cast<const uint4*>(base_p + (size_t)(2 * pair_r) * W_P_ROW);
        const uint4* p1 = reinterpret_cast<const uint4*>(base_p + (size_t)(2 * pair_r + 1) * W_P_ROW);
        const uint16_t* s0 = reinterpret_cast<const uint16_t*>(base_s + (size_t)(2 * pair_r) * W_S_ROW);
        const uint16_t* s1 = reinterpret_cast<const uint16_t*>(base_s + (size_t)(2 * pair_r + 1) * W_S_ROW);

        auto proc_dn = [&](const uint4& v0, const uint4& v1, uint16_t sr0, uint16_t sr1, int idx) {
            const float2 sc0 = e4m3x2_to_f32x2(sr0);
            const float2 sc1 = e4m3x2_to_f32x2(sr1);
            __half2 w0[8], w1[8];
            const __half2* hh = hr + idx * 16;
            sf4_decode16_h2(v0.x, v0.y, w0);
            sf4_decode16_h2(v1.x, v1.y, w1);
            y0 = fmaf(dot16_h2_2acc(w0, hh), sc0.x, y0);
            y1 = fmaf(dot16_h2_2acc(w1, hh), sc1.x, y1);
            sf4_decode16_h2(v0.z, v0.w, w0);
            sf4_decode16_h2(v1.z, v1.w, w1);
            y0 = fmaf(dot16_h2_2acc(w0, hh + 8), sc0.y, y0);
            y1 = fmaf(dot16_h2_2acc(w1, hh + 8), sc1.y, y1);
        };

        // 36 uint4s / 16 lanes = 2 full waves (0, 16) + 4-lane tail (32..35) -> issue ALL loads upfront!
        const uint4 a0 = __ldg(p0 + lg + 0),  b0 = __ldg(p1 + lg + 0);
        const uint4 a1 = __ldg(p0 + lg + 16), b1 = __ldg(p1 + lg + 16);
        const uint16_t sa0 = __ldg(s0 + lg + 0),  sb0 = __ldg(s1 + lg + 0);
        const uint16_t sa1 = __ldg(s0 + lg + 16), sb1 = __ldg(s1 + lg + 16);
        const bool has2 = (lg < 4);
        const uint4 a2 = has2 ? __ldg(p0 + lg + 32) : make_uint4(0u, 0u, 0u, 0u);
        const uint4 b2 = has2 ? __ldg(p1 + lg + 32) : make_uint4(0u, 0u, 0u, 0u);
        const uint16_t sa2 = has2 ? __ldg(s0 + lg + 32) : 0u;
        const uint16_t sb2 = has2 ? __ldg(s1 + lg + 32) : 0u;
        proc_dn(a0, b0, sa0, sb0, lg + 0);
        proc_dn(a1, b1, sa1, sb1, lg + 16);
        if (has2) proc_dn(a2, b2, sa2, sb2, lg + 32);
        y0 *= w_outer; y1 *= w_outer;
    } else {
        constexpr size_t W_P_ROW = kInkInter;
        constexpr size_t W_S_ROW = kInkInter / 32;
        const uint8_t* base_p = body + ed.dn_off;
        const uint8_t* base_s = base_p + (size_t)kInkHidden * W_P_ROW;
        const uint4* p0 = reinterpret_cast<const uint4*>(base_p + (size_t)(2 * pair_r) * W_P_ROW);
        const uint4* p1 = reinterpret_cast<const uint4*>(base_p + (size_t)(2 * pair_r + 1) * W_P_ROW);
        const uint8_t* s0 = base_s + (size_t)(2 * pair_r) * W_S_ROW;
        const uint8_t* s1 = base_s + (size_t)(2 * pair_r + 1) * W_S_ROW;
#pragma unroll 4
        for (int i = lg; i < 72; i += 16) {
            const uint4 v0 = __ldg(p0 + i), v1 = __ldg(p1 + i);
            const float sc0 = ue8m0_to_f32(__ldg(s0 + (i >> 1)));
            const float sc1 = ue8m0_to_f32(__ldg(s1 + (i >> 1)));
            __half2 w0[8], w1[8];
            mx8_decode16_h2(v0, w0);
            mx8_decode16_h2(v1, w1);
            const __half2* hh = hr + i * 8;
            y0 = fmaf(dot16_h2_2acc(w0, hh), sc0, y0);
            y1 = fmaf(dot16_h2_2acc(w1, hh), sc1, y1);
        }
    }
#pragma unroll
    for (int o = 8; o >= 1; o >>= 1) {
        y0 += __shfl_xor_sync(0xFFFFFFFFu, y0, o);
        y1 += __shfl_xor_sync(0xFFFFFFFFu, y1, o);
    }
    if (lg == 0) {
        const float rw = topk_w[slot];
        y_pair[(size_t)slot * kInkHidden + 2 * pair_r]     = y0 * rw;
        y_pair[(size_t)slot * kInkHidden + 2 * pair_r + 1] = y1 * rw;
    }
}

// ============================================================================
// 5C. Grouped Register-Pipelined SF4 MoE Kernels for M = 2..16 (ZERO cp.async, ZERO inner-loop barriers!)
//     Reads each active expert ONCE across all routed tokens (n_e <= 8 in shared mem) -> 3-4x faster than TR=16 TC!
// ============================================================================
__global__ void __launch_bounds__(256) k_moe_sf4_grouped_reg_gu(
    const uint8_t* __restrict__ sec4_base,
    const InkExpertDev* __restrict__ experts,
    InkMoeTables tb,
    const __half2* __restrict__ x_h2,      // [M, 1152] pre-converted!
    __half2* __restrict__ h_h2_out) {      // [P, 576] output directly in __half2 for DN!
    const int ai = blockIdx.y;
    if (ai >= *tb.nact) return;
    const int e = tb.active[ai];
    const int row0 = tb.off[e];
    const int n_total = tb.cnt[e];
    const InkExpertDev ed = experts[e];
    if (ed.tier > 1 || ed.gu_code != FMT_SF4) return;

    const int tid  = threadIdx.x;
    const int sub  = tid >> 4;             // 0..15 features/CTA
    const int lg   = tid & 15;             // 0..15 lanes/feature (256B coalesced!)
    const int feat = blockIdx.x * 16 + sub;
    const bool valid_feat = (feat < kInkInter);

    const uint8_t* body = sec4_base + ed.body_off;
    const float2 h01 = *reinterpret_cast<const float2*>(body);
    const float w_outer = h01.x / h01.y;
    constexpr size_t W_P_ROW = kInkHidden / 2;
    constexpr size_t W_S_ROW = kInkHidden / 16;
    const uint8_t* base_p = body + 16;
    const uint8_t* base_s = base_p + (size_t)(2 * kInkInter) * W_P_ROW;
    const uint4* gp16 = valid_feat ? reinterpret_cast<const uint4*>(base_p + (size_t)(2 * feat) * W_P_ROW) : nullptr;
    const uint4* up16 = valid_feat ? reinterpret_cast<const uint4*>(base_p + (size_t)(2 * feat + 1) * W_P_ROW) : nullptr;
    const uint16_t* gs2 = valid_feat ? reinterpret_cast<const uint16_t*>(base_s + (size_t)(2 * feat) * W_S_ROW) : nullptr;
    const uint16_t* us2 = valid_feat ? reinterpret_cast<const uint16_t*>(base_s + (size_t)(2 * feat + 1) * W_S_ROW) : nullptr;

    __shared__ alignas(16) __half2 s_x[8][kInkHidden / 2];  // 36 KB

    for (int off_m = 0; off_m < n_total; off_m += 8) {
        const int n_e = min(8, n_total - off_m);
        for (int m = 0; m < n_e; ++m) {
            const uint4* src4 = reinterpret_cast<const uint4*>(x_h2 + (size_t)tb.tok_idx[row0 + off_m + m] * (kInkHidden / 2));
            uint4* dst4 = reinterpret_cast<uint4*>(s_x[m]);
            for (int i = tid; i < (kInkHidden / 8); i += 256) dst4[i] = src4[i];
        }
        __syncthreads();

        if (valid_feat) {
            if (n_e == 1) {
                float ag = 0.0f, au = 0.0f;
                auto proc1 = [&](const uint4& gv, const uint4& uv, uint16_t gs_r, uint16_t us_r, int idx) {
                    const float2 sg = e4m3x2_to_f32x2(gs_r);
                    const float2 su = e4m3x2_to_f32x2(us_r);
                    __half2 gw[8], uw[8];
                    const __half2* xh0 = s_x[0] + idx * 16;
                    sf4_decode16_h2(gv.x, gv.y, gw);
                    sf4_decode16_h2(uv.x, uv.y, uw);
                    ag = fmaf(dot16_h2_2acc(gw, xh0), sg.x, ag);
                    au = fmaf(dot16_h2_2acc(uw, xh0), su.x, au);
                    sf4_decode16_h2(gv.z, gv.w, gw);
                    sf4_decode16_h2(uv.z, uv.w, uw);
                    ag = fmaf(dot16_h2_2acc(gw, xh0 + 8), sg.y, ag);
                    au = fmaf(dot16_h2_2acc(uw, xh0 + 8), su.y, au);
                };
                const uint4 g0 = __ldg(gp16 + lg + 0),  u0 = __ldg(up16 + lg + 0);
                const uint4 g1 = __ldg(gp16 + lg + 16), u1 = __ldg(up16 + lg + 16);
                const uint16_t gs0 = __ldg(gs2 + lg + 0),  us0 = __ldg(us2 + lg + 0);
                const uint16_t gs1 = __ldg(gs2 + lg + 16), us1 = __ldg(us2 + lg + 16);
                const uint4 g2 = __ldg(gp16 + lg + 32), u2 = __ldg(up16 + lg + 32);
                const uint16_t gs2v = __ldg(gs2 + lg + 32), us2v = __ldg(us2 + lg + 32);
                proc1(g0, u0, gs0, us0, lg + 0);
                const uint4 g3 = __ldg(gp16 + lg + 48), u3 = __ldg(up16 + lg + 48);
                const uint16_t gs3 = __ldg(gs2 + lg + 48), us3 = __ldg(us2 + lg + 48);
                proc1(g1, u1, gs1, us1, lg + 16);
                const bool has4 = (lg < 8);
                const uint4 g4 = has4 ? __ldg(gp16 + lg + 64) : make_uint4(0u, 0u, 0u, 0u);
                const uint4 u4 = has4 ? __ldg(up16 + lg + 64) : make_uint4(0u, 0u, 0u, 0u);
                const uint16_t gs4 = has4 ? __ldg(gs2 + lg + 64) : 0u;
                const uint16_t us4 = has4 ? __ldg(us2 + lg + 64) : 0u;
                proc1(g2, u2, gs2v, us2v, lg + 32);
                proc1(g3, u3, gs3, us3, lg + 48);
                if (has4) proc1(g4, u4, gs4, us4, lg + 64);
                ag *= w_outer; au *= w_outer;
#pragma unroll
                for (int o = 8; o >= 1; o >>= 1) {
                    ag += __shfl_xor_sync(0xFFFFFFFFu, ag, o);
                    au += __shfl_xor_sync(0xFFFFFFFFu, au, o);
                }
                const float hv = situ_f32(ag, au);
                const float hv_odd = __shfl_xor_sync(0xFFFFFFFFu, hv, 16);
                if (lg == 0 && (sub & 1) == 0) {
                    h_h2_out[(size_t)(row0 + off_m) * (kInkInter / 2) + (feat >> 1)] = __floats2half2_rn(hv, hv_odd);
                }
            } else if (n_e <= 4) {
                float ag[4] = {}, au[4] = {};
                for (int i = lg; i < 72; i += 16) {
                    const uint4 gv = __ldg(gp16 + i), uv = __ldg(up16 + i);
                    const float2 sg = e4m3x2_to_f32x2(__ldg(gs2 + i));
                    const float2 su = e4m3x2_to_f32x2(__ldg(us2 + i));
                    __half2 gw[8], uw[8];
                    sf4_decode16_h2(gv.x, gv.y, gw);
                    sf4_decode16_h2(uv.x, uv.y, uw);
#pragma unroll
                    for (int m = 0; m < 4; ++m) {
                        if (m < n_e) {
                            const __half2* xh = s_x[m] + i * 16;
                            ag[m] = fmaf(dot16_h2_2acc(gw, xh), sg.x, ag[m]);
                            au[m] = fmaf(dot16_h2_2acc(uw, xh), su.x, au[m]);
                        }
                    }
                    sf4_decode16_h2(gv.z, gv.w, gw);
                    sf4_decode16_h2(uv.z, uv.w, uw);
#pragma unroll
                    for (int m = 0; m < 4; ++m) {
                        if (m < n_e) {
                            const __half2* xh = s_x[m] + i * 16 + 8;
                            ag[m] = fmaf(dot16_h2_2acc(gw, xh), sg.y, ag[m]);
                            au[m] = fmaf(dot16_h2_2acc(uw, xh), su.y, au[m]);
                        }
                    }
                }
#pragma unroll
                for (int m = 0; m < 4; ++m) {
                    if (m < n_e) {
                        float gv = ag[m] * w_outer, uv = au[m] * w_outer;
#pragma unroll
                        for (int o = 8; o >= 1; o >>= 1) {
                            gv += __shfl_xor_sync(0xFFFFFFFFu, gv, o);
                            uv += __shfl_xor_sync(0xFFFFFFFFu, uv, o);
                        }
                        const float hv = situ_f32(gv, uv);
                        const float hv_odd = __shfl_xor_sync(0xFFFFFFFFu, hv, 16);
                        if (lg == 0 && (sub & 1) == 0) {
                            h_h2_out[(size_t)(row0 + off_m + m) * (kInkInter / 2) + (feat >> 1)] = __floats2half2_rn(hv, hv_odd);
                        }
                    }
                }
            } else {
                float ag[8] = {}, au[8] = {};
                for (int i = lg; i < 72; i += 16) {
                    const uint4 gv = __ldg(gp16 + i), uv = __ldg(up16 + i);
                    const float2 sg = e4m3x2_to_f32x2(__ldg(gs2 + i));
                    const float2 su = e4m3x2_to_f32x2(__ldg(us2 + i));
                    __half2 gw[8], uw[8];
                    sf4_decode16_h2(gv.x, gv.y, gw);
                    sf4_decode16_h2(uv.x, uv.y, uw);
#pragma unroll
                    for (int m = 0; m < 8; ++m) {
                        if (m < n_e) {
                            const __half2* xh = s_x[m] + i * 16;
                            ag[m] = fmaf(dot16_h2_2acc(gw, xh), sg.x, ag[m]);
                            au[m] = fmaf(dot16_h2_2acc(uw, xh), su.x, au[m]);
                        }
                    }
                    sf4_decode16_h2(gv.z, gv.w, gw);
                    sf4_decode16_h2(uv.z, uv.w, uw);
#pragma unroll
                    for (int m = 0; m < 8; ++m) {
                        if (m < n_e) {
                            const __half2* xh = s_x[m] + i * 16 + 8;
                            ag[m] = fmaf(dot16_h2_2acc(gw, xh), sg.y, ag[m]);
                            au[m] = fmaf(dot16_h2_2acc(uw, xh), su.y, au[m]);
                        }
                    }
                }
#pragma unroll
                for (int m = 0; m < 8; ++m) {
                    if (m < n_e) {
                        float gv = ag[m] * w_outer, uv = au[m] * w_outer;
#pragma unroll
                        for (int o = 8; o >= 1; o >>= 1) {
                            gv += __shfl_xor_sync(0xFFFFFFFFu, gv, o);
                            uv += __shfl_xor_sync(0xFFFFFFFFu, uv, o);
                        }
                        const float hv = situ_f32(gv, uv);
                        const float hv_odd = __shfl_xor_sync(0xFFFFFFFFu, hv, 16);
                        if (lg == 0 && (sub & 1) == 0) {
                            h_h2_out[(size_t)(row0 + off_m + m) * (kInkInter / 2) + (feat >> 1)] = __floats2half2_rn(hv, hv_odd);
                        }
                    }
                }
            }
        }
        if (off_m + 8 < n_total) __syncthreads();
    }
}

__global__ void __launch_bounds__(256) k_moe_sf4_grouped_reg_dn(
    const uint8_t* __restrict__ sec4_base,
    const InkExpertDev* __restrict__ experts,
    InkMoeTables tb,
    const __half2* __restrict__ h_h2,      // [P, 576]
    const float* __restrict__ topk_w,
    float* __restrict__ y_pair) {          // [P, 2304]
    const int ai = blockIdx.y;
    if (ai >= *tb.nact) return;
    const int e = tb.active[ai];
    const int row0 = tb.off[e];
    const int n_total = tb.cnt[e];
    const InkExpertDev ed = experts[e];
    if (ed.tier > 1 || ed.dn_code != FMT_SF4) return;

    const int tid    = threadIdx.x;
    const int sub    = tid >> 4;           // 0..15 row-pairs/CTA
    const int lg     = tid & 15;           // 0..15 lanes/pair
    const int pair_r = blockIdx.x * 16 + sub;
    const bool valid_r = (pair_r < kInkHidden / 2);

    const uint8_t* body = sec4_base + ed.body_off;
    const float2 h23 = *reinterpret_cast<const float2*>(body + 8);
    const float w_outer = h23.x / h23.y;
    constexpr size_t W_P_ROW = kInkInter / 2;
    constexpr size_t W_S_ROW = kInkInter / 16;
    const uint8_t* base_p = body + ed.dn_off;
    const uint8_t* base_s = base_p + (size_t)kInkHidden * W_P_ROW;
    const uint4* p0 = valid_r ? reinterpret_cast<const uint4*>(base_p + (size_t)(2 * pair_r) * W_P_ROW) : nullptr;
    const uint4* p1 = valid_r ? reinterpret_cast<const uint4*>(base_p + (size_t)(2 * pair_r + 1) * W_P_ROW) : nullptr;
    const uint16_t* s0 = valid_r ? reinterpret_cast<const uint16_t*>(base_s + (size_t)(2 * pair_r) * W_S_ROW) : nullptr;
    const uint16_t* s1 = valid_r ? reinterpret_cast<const uint16_t*>(base_s + (size_t)(2 * pair_r + 1) * W_S_ROW) : nullptr;

    __shared__ alignas(16) __half2 s_h[8][kInkInter / 2];  // 18 KB
    __shared__ float s_rw[8];
    __shared__ int32_t s_pair[8];

    for (int off_m = 0; off_m < n_total; off_m += 8) {
        const int n_e = min(8, n_total - off_m);
        if (tid < n_e) {
            const int pr = tb.pairs[row0 + off_m + tid];
            s_pair[tid] = pr;
            s_rw[tid]   = topk_w[pr] * w_outer;
        }
        for (int m = 0; m < n_e; ++m) {
            const uint4* src4 = reinterpret_cast<const uint4*>(h_h2 + (size_t)(row0 + off_m + m) * (kInkInter / 2));
            uint4* dst4 = reinterpret_cast<uint4*>(s_h[m]);
            for (int i = tid; i < (kInkInter / 8); i += 256) dst4[i] = src4[i];
        }
        __syncthreads();

        if (valid_r) {
            if (n_e == 1) {
                float y0 = 0.0f, y1 = 0.0f;
                auto proc1 = [&](const uint4& v0, const uint4& v1, uint16_t sr0, uint16_t sr1, int idx) {
                    const float2 sc0 = e4m3x2_to_f32x2(sr0);
                    const float2 sc1 = e4m3x2_to_f32x2(sr1);
                    __half2 w0[8], w1[8];
                    const __half2* hh = s_h[0] + idx * 16;
                    sf4_decode16_h2(v0.x, v0.y, w0);
                    sf4_decode16_h2(v1.x, v1.y, w1);
                    y0 = fmaf(dot16_h2_2acc(w0, hh), sc0.x, y0);
                    y1 = fmaf(dot16_h2_2acc(w1, hh), sc1.x, y1);
                    sf4_decode16_h2(v0.z, v0.w, w0);
                    sf4_decode16_h2(v1.z, v1.w, w1);
                    y0 = fmaf(dot16_h2_2acc(w0, hh + 8), sc0.y, y0);
                    y1 = fmaf(dot16_h2_2acc(w1, hh + 8), sc1.y, y1);
                };
                const uint4 a0 = __ldg(p0 + lg + 0),  b0 = __ldg(p1 + lg + 0);
                const uint4 a1 = __ldg(p0 + lg + 16), b1 = __ldg(p1 + lg + 16);
                const uint16_t sa0 = __ldg(s0 + lg + 0),  sb0 = __ldg(s1 + lg + 0);
                const uint16_t sa1 = __ldg(s0 + lg + 16), sb1 = __ldg(s1 + lg + 16);
                const bool has2 = (lg < 4);
                const uint4 a2 = has2 ? __ldg(p0 + lg + 32) : make_uint4(0u, 0u, 0u, 0u);
                const uint4 b2 = has2 ? __ldg(p1 + lg + 32) : make_uint4(0u, 0u, 0u, 0u);
                const uint16_t sa2 = has2 ? __ldg(s0 + lg + 32) : 0u;
                const uint16_t sb2 = has2 ? __ldg(s1 + lg + 32) : 0u;
                proc1(a0, b0, sa0, sb0, lg + 0);
                proc1(a1, b1, sa1, sb1, lg + 16);
                if (has2) proc1(a2, b2, sa2, sb2, lg + 32);
#pragma unroll
                for (int o = 8; o >= 1; o >>= 1) {
                    y0 += __shfl_xor_sync(0xFFFFFFFFu, y0, o);
                    y1 += __shfl_xor_sync(0xFFFFFFFFu, y1, o);
                }
                if (lg == 0) {
                    const float rw = s_rw[0];
                    const int pr = s_pair[0];
                    y_pair[(size_t)pr * kInkHidden + 2 * pair_r]     = y0 * rw;
                    y_pair[(size_t)pr * kInkHidden + 2 * pair_r + 1] = y1 * rw;
                }
            } else if (n_e <= 4) {
                float y0[4] = {}, y1[4] = {};
                for (int i = lg; i < 36; i += 16) {
                    const uint4 v0 = __ldg(p0 + i), v1 = __ldg(p1 + i);
                    const float2 sc0 = e4m3x2_to_f32x2(__ldg(s0 + i));
                    const float2 sc1 = e4m3x2_to_f32x2(__ldg(s1 + i));
                    __half2 w0[8], w1[8];
                    sf4_decode16_h2(v0.x, v0.y, w0);
                    sf4_decode16_h2(v1.x, v1.y, w1);
#pragma unroll
                    for (int m = 0; m < 4; ++m) {
                        if (m < n_e) {
                            const __half2* hh = s_h[m] + i * 16;
                            y0[m] = fmaf(dot16_h2_2acc(w0, hh), sc0.x, y0[m]);
                            y1[m] = fmaf(dot16_h2_2acc(w1, hh), sc1.x, y1[m]);
                        }
                    }
                    sf4_decode16_h2(v0.z, v0.w, w0);
                    sf4_decode16_h2(v1.z, v1.w, w1);
#pragma unroll
                    for (int m = 0; m < 4; ++m) {
                        if (m < n_e) {
                            const __half2* hh = s_h[m] + i * 16 + 8;
                            y0[m] = fmaf(dot16_h2_2acc(w0, hh), sc0.y, y0[m]);
                            y1[m] = fmaf(dot16_h2_2acc(w1, hh), sc1.y, y1[m]);
                        }
                    }
                }
#pragma unroll
                for (int m = 0; m < 4; ++m) {
                    if (m < n_e) {
                        float a0 = y0[m], a1 = y1[m];
#pragma unroll
                        for (int o = 8; o >= 1; o >>= 1) {
                            a0 += __shfl_xor_sync(0xFFFFFFFFu, a0, o);
                            a1 += __shfl_xor_sync(0xFFFFFFFFu, a1, o);
                        }
                        if (lg == 0) {
                            const float rw = s_rw[m];
                            const int pr = s_pair[m];
                            y_pair[(size_t)pr * kInkHidden + 2 * pair_r]     = a0 * rw;
                            y_pair[(size_t)pr * kInkHidden + 2 * pair_r + 1] = a1 * rw;
                        }
                    }
                }
            } else {
                float y0[8] = {}, y1[8] = {};
                for (int i = lg; i < 36; i += 16) {
                    const uint4 v0 = __ldg(p0 + i), v1 = __ldg(p1 + i);
                    const float2 sc0 = e4m3x2_to_f32x2(__ldg(s0 + i));
                    const float2 sc1 = e4m3x2_to_f32x2(__ldg(s1 + i));
                    __half2 w0[8], w1[8];
                    sf4_decode16_h2(v0.x, v0.y, w0);
                    sf4_decode16_h2(v1.x, v1.y, w1);
#pragma unroll
                    for (int m = 0; m < 8; ++m) {
                        if (m < n_e) {
                            const __half2* hh = s_h[m] + i * 16;
                            y0[m] = fmaf(dot16_h2_2acc(w0, hh), sc0.x, y0[m]);
                            y1[m] = fmaf(dot16_h2_2acc(w1, hh), sc1.x, y1[m]);
                        }
                    }
                    sf4_decode16_h2(v0.z, v0.w, w0);
                    sf4_decode16_h2(v1.z, v1.w, w1);
#pragma unroll
                    for (int m = 0; m < 8; ++m) {
                        if (m < n_e) {
                            const __half2* hh = s_h[m] + i * 16 + 8;
                            y0[m] = fmaf(dot16_h2_2acc(w0, hh), sc0.y, y0[m]);
                            y1[m] = fmaf(dot16_h2_2acc(w1, hh), sc1.y, y1[m]);
                        }
                    }
                }
#pragma unroll
                for (int m = 0; m < 8; ++m) {
                    if (m < n_e) {
                        float a0 = y0[m], a1 = y1[m];
#pragma unroll
                        for (int o = 8; o >= 1; o >>= 1) {
                            a0 += __shfl_xor_sync(0xFFFFFFFFu, a0, o);
                            a1 += __shfl_xor_sync(0xFFFFFFFFu, a1, o);
                        }
                        if (lg == 0) {
                            const float rw = s_rw[m];
                            const int pr = s_pair[m];
                            y_pair[(size_t)pr * kInkHidden + 2 * pair_r]     = a0 * rw;
                            y_pair[(size_t)pr * kInkHidden + 2 * pair_r + 1] = a1 * rw;
                        }
                    }
                }
            }
        }
        if (off_m + 8 < n_total) __syncthreads();
    }
}

// ============================================================================
// 5D. In-Place Startup MX8 -> SF4 Transcoder (Zero Permanent VRAM, Opt-In !EXACT Only!)
//     Converts the 4 single-pass MX8 layers (512 experts) into uniform SF4 (4.49 MB vs 8.22 MB -> -45% BW!)
// ============================================================================
__global__ void __launch_bounds__(256) k_mx8_mat_absmax(
    const uint8_t* __restrict__ src_p,   // [rows, cols] E4M3
    const uint8_t* __restrict__ src_s,   // [rows, cols/32] UE8M0
    int rows, int cols,
    float* __restrict__ d_amax) {
    const int nblk = rows * (cols >> 5);
    float mx = 0.0f;
    for (int blk = blockIdx.x * 8 + (threadIdx.x >> 5); blk < nblk; blk += gridDim.x * 8) {
        const int lane = threadIdx.x & 31;
        const float sc = ue8m0_to_f32(src_s[blk]);
        const float v = fabsf(e4m3_to_f32(src_p[(size_t)blk * 32 + lane])) * sc;
        mx = fmaxf(mx, v);
    }
    mx = warp_max(mx);
    __shared__ float sm[8];
    if ((threadIdx.x & 31) == 0) sm[threadIdx.x >> 5] = mx;
    __syncthreads();
    if (threadIdx.x == 0) {
        float g = fmaxf(fmaxf(fmaxf(sm[0], sm[1]), fmaxf(sm[2], sm[3])),
                        fmaxf(fmaxf(sm[4], sm[5]), fmaxf(sm[6], sm[7])));
        atomicMax(reinterpret_cast<uint32_t*>(d_amax), __float_as_uint(g));
    }
}

__global__ void __launch_bounds__(256) k_mx8_mat_to_sf4(
    const uint8_t* __restrict__ src_p,   // [rows, cols] E4M3
    const uint8_t* __restrict__ src_s,   // [rows, cols/32] UE8M0
    const float* __restrict__ d_amax,
    int rows, int cols,
    float* __restrict__ hdr_g_out,
    float* __restrict__ hdr_1_out,
    uint8_t* __restrict__ dst_p,         // [rows, cols/2] E2M1x2
    uint8_t* __restrict__ dst_s) {       // [rows, cols/16] E4M3
#if defined(CUDART_VERSION) && (CUDART_VERSION >= 12080)
    const float g = fmaxf((*d_amax) * (1.0f / (6.0f * 448.0f)), 1e-12f);
    if (blockIdx.x == 0 && threadIdx.x == 0) {
        *hdr_g_out = g;
        *hdr_1_out = 1.0f;
    }
    const int nsub = rows * (cols >> 4);
    const int tid = threadIdx.x;
    const int sub_in_cta = tid >> 3;  // 0..31 sub-blocks of 16 elements per CTA (8 lanes/sub-block -> 2 elems/lane!)
    const int lane8 = tid & 7;        // 0..7
    const int sb = blockIdx.x * 32 + sub_in_cta;
    if (sb >= nsub) return;

    const float sc_mx = ue8m0_to_f32(src_s[sb >> 1]);
    const uint8_t* sp = src_p + (size_t)sb * 16 + lane8 * 2;
    const float v0 = e4m3_to_f32(sp[0]) * sc_mx;
    const float v1 = e4m3_to_f32(sp[1]) * sc_mx;
    float am = fmaxf(fabsf(v0), fabsf(v1));
#pragma unroll
    for (int o = 4; o >= 1; o >>= 1) am = fmaxf(am, __shfl_xor_sync(0xFFFFFFFFu, am, o));

    const float sc_raw = fminf(am / (g * 6.0f), 448.0f);
    const __nv_fp8_e4m3 fp8_sc(sc_raw);
    const float sc_dec = (float)fp8_sc;
    const float denom = g * sc_dec;
    const float inv = (denom > 0.0f) ? (1.0f / denom) : 0.0f;
    const __nv_fp4x2_e2m1 q2(make_float2(v0 * inv, v1 * inv));
    dst_p[(size_t)sb * 8 + lane8] = q2.__x;
    if (lane8 == 0) dst_s[sb] = fp8_sc.__x;
#else
    (void)src_p; (void)src_s; (void)d_amax; (void)rows; (void)cols;
    (void)hdr_g_out; (void)hdr_1_out; (void)dst_p; (void)dst_s;
#endif
}

template <typename ModelT>
inline void transcode_mx8_to_sf4_inplace(ModelT& m, cudaStream_t st) {
#if defined(CUDART_VERSION) && (CUDART_VERSION >= 12080)
    if (!moe_opt_on() || exact_mode_on() || m.ls_active || m.ep2_active || m.hybrid.active || !m.d_sec4) return;
    static const bool tc_on = [] {
        const char* e = std::getenv("STRATA_INK_SM120_MX8_TO_SF4");
        return !e || e[0] != '0';
    }();
    if (!tc_on) return;

    int n_conv = 0;
    constexpr int kRowsGU = 2 * kInkInter;  // 2304
    constexpr int kColsGU = kInkHidden;     // 2304
    constexpr int kRowsDN = kInkHidden;     // 2304
    constexpr int kColsDN = kInkInter;      // 1152
    constexpr size_t kMx8GuBytes = (size_t)kRowsGU * kColsGU + (size_t)kRowsGU * (kColsGU / 32);
    constexpr size_t kMx8DnBytes = (size_t)kRowsDN * kColsDN + (size_t)kRowsDN * (kColsDN / 32);
    constexpr uint32_t kSf4DnOff = 16u + (uint32_t)((size_t)kRowsGU * (kColsGU / 2) + (size_t)kRowsGU * (kColsGU / 16));

    uint8_t* d_tmp = nullptr;
    float*   d_am  = nullptr;
    std::vector<InkExpertDev> h_exp(m.man.c.n_experts);

    for (int il = 0; il < m.man.c.n_phys; ++il) {
        if (!m.moe_info[il].has_mx8 || m.moe_info[il].has_sk || m.moe_info[il].has_gguf) continue;
        INK_CK(cudaMemcpy(h_exp.data(), m.d_experts[il], (size_t)m.man.c.n_experts * sizeof(InkExpertDev), cudaMemcpyDeviceToHost));
        bool any_l = false;
        for (int e = 0; e < m.man.c.n_experts; ++e) {
            if (h_exp[e].tier == 0 && h_exp[e].gu_code == FMT_MX8 && h_exp[e].dn_code == FMT_MX8) {
                if (!d_tmp) {
                    d_tmp = dalloc<uint8_t>(kMx8GuBytes + kMx8DnBytes + 4096);
                    d_am  = dalloc<float>(2);
                }
                uint8_t* dst_body = m.d_sec4 + h_exp[e].body_off;
                const size_t total_mx8 = (size_t)h_exp[e].dn_off + kMx8DnBytes;
                INK_CK(cudaMemcpyAsync(d_tmp, dst_body, total_mx8, cudaMemcpyDeviceToDevice, st));
                INK_CK(cudaMemsetAsync(d_am, 0, 2 * sizeof(float), st));

                // GU: MX8 -> SF4
                const uint8_t* gu_sp = d_tmp;
                const uint8_t* gu_ss = gu_sp + (size_t)kRowsGU * kColsGU;
                k_mx8_mat_absmax<<<256, 256, 0, st>>>(gu_sp, gu_ss, kRowsGU, kColsGU, d_am + 0);
                uint8_t* gu_dp = dst_body + 16;
                uint8_t* gu_ds = gu_dp + (size_t)kRowsGU * (kColsGU / 2);
                const int nsub_gu = kRowsGU * (kColsGU / 16);
                k_mx8_mat_to_sf4<<<(nsub_gu + 31) / 32, 256, 0, st>>>(
                    gu_sp, gu_ss, d_am + 0, kRowsGU, kColsGU,
                    reinterpret_cast<float*>(dst_body + 0), reinterpret_cast<float*>(dst_body + 4), gu_dp, gu_ds);

                // DN: MX8 -> SF4
                const uint8_t* dn_sp = d_tmp + h_exp[e].dn_off;
                const uint8_t* dn_ss = dn_sp + (size_t)kRowsDN * kColsDN;
                k_mx8_mat_absmax<<<256, 256, 0, st>>>(dn_sp, dn_ss, kRowsDN, kColsDN, d_am + 1);
                uint8_t* dn_dp = dst_body + kSf4DnOff;
                uint8_t* dn_ds = dn_dp + (size_t)kRowsDN * (kColsDN / 2);
                const int nsub_dn = kRowsDN * (kColsDN / 16);
                k_mx8_mat_to_sf4<<<(nsub_dn + 31) / 32, 256, 0, st>>>(
                    dn_sp, dn_ss, d_am + 1, kRowsDN, kColsDN,
                    reinterpret_cast<float*>(dst_body + 8), reinterpret_cast<float*>(dst_body + 12), dn_dp, dn_ds);

                h_exp[e].gu_code = FMT_SF4;
                h_exp[e].dn_code = FMT_SF4;
                h_exp[e].dn_off  = kSf4DnOff;
                any_l = true;
                ++n_conv;
            }
        }
        if (any_l) {
            INK_CK(cudaStreamSynchronize(st));
            INK_CK(cudaMemcpy(m.d_experts[il], h_exp.data(), (size_t)m.man.c.n_experts * sizeof(InkExpertDev), cudaMemcpyHostToDevice));
            m.moe_info[il].has_mx8 = false;
            m.moe_info[il].has_sf4 = true;
        }
    }
    if (d_tmp) cudaFree(d_tmp);
    if (d_am)  cudaFree(d_am);
    if (n_conv > 0) {
        std::fprintf(stderr, "[InkRTX5090] Transcoded %d MX8 experts -> uniform SF4 in-place (0 extra VRAM)\n", n_conv);
    }
#else
    (void)m; (void)st;
#endif
}

// Fused Split-K Merge + Dual F32/__half2 Output (eliminates 48 k_ink_f32_to_h2 launches before wo_ud!)
__global__ void __launch_bounds__(128) k_ink_flash_gqa_merge_h2(
    const float* __restrict__ part_m,
    const float* __restrict__ part_l,
    const float* __restrict__ part_o,
    const float* __restrict__ gate_attn,
    float* __restrict__ out,
    __half2* __restrict__ out_h2,
    int M, int n_splits) {
    const int h = blockIdx.x;
    const int m = blockIdx.y;
    const int d = threadIdx.x;
    if (h >= kInkHeads || m >= M || d >= kInkHeadDim) return;

    const size_t base_sp = ((size_t)m * kInkHeads + h) * n_splits;
    float mg = -1e30f;
    for (int sp = 0; sp < n_splits; ++sp) mg = fmaxf(mg, part_m[base_sp + sp]);
    float ls = 0.0f, acc = 0.0f;
    for (int sp = 0; sp < n_splits; ++sp) {
        const float lv = part_l[base_sp + sp];
        if (lv > 0.0f) {
            const float w = __expf(part_m[base_sp + sp] - mg);
            ls  += lv * w;
            acc += part_o[(base_sp + sp) * kInkHeadDim + d] * w;
        }
    }
    float res = (ls > 0.0f) ? (acc / ls) : 0.0f;
    const size_t ob = ((size_t)m * kInkHeads + h) * kInkHeadDim + d;
    if (gate_attn) res *= 1.0f / (1.0f + __expf(-gate_attn[ob]));
    out[ob] = res;
    if (out_h2) {
        const float odd = __shfl_xor_sync(0xFFFFFFFFu, res, 1);
        if ((d & 1) == 0) {
            out_h2[ob >> 1] = __floats2half2_rn(fminf(fmaxf(res, -65504.0f), 65504.0f),
                                                fminf(fmaxf(odd, -65504.0f), 65504.0f));
        }
    }
}

// High-BW FP16 Router GEMV for M <= 16 (2x less bandwidth than F32 + vectorized __hfma2 + 128 CTAs x 128 threads!)
template <int MAX_M>
__global__ void __launch_bounds__(128) k_ink_router_gemv_fp16(
    const float* __restrict__ x_f32,          // [M, 2304]
    const __half2* __restrict__ gate_w_h2,    // [128, 1152]
    float* __restrict__ r_logits,             // [M, 128]
    int M) {
    INK_PDL_WAIT();   // PERF-STEP: first statement on every path (programmatic dependent launch)
    const int e = blockIdx.x;
    if (e >= kInkExperts) return;
    const int tid  = threadIdx.x;
    const int warp = tid >> 5;
    const int lane = tid & 31;
    constexpr int K8 = kInkHidden / 8;  // 288 uint4s of __half2
    const uint4* wr = reinterpret_cast<const uint4*>(gate_w_h2 + (size_t)e * (kInkHidden / 2));
    __half2 acc0[MAX_M], acc1[MAX_M];
#pragma unroll
    for (int m = 0; m < MAX_M; ++m) {
        acc0[m] = __float2half2_rn(0.0f);
        acc1[m] = __float2half2_rn(0.0f);
    }
    for (int i = tid; i < K8; i += 128) {
        const uint4 wv = __ldg(wr + i);
        const __half2 w0 = *reinterpret_cast<const __half2*>(&wv.x);
        const __half2 w1 = *reinterpret_cast<const __half2*>(&wv.y);
        const __half2 w2 = *reinterpret_cast<const __half2*>(&wv.z);
        const __half2 w3 = *reinterpret_cast<const __half2*>(&wv.w);
#pragma unroll
        for (int m = 0; m < MAX_M; ++m) {
            if (m < M) {
                const float4* xp = reinterpret_cast<const float4*>(x_f32 + (size_t)m * kInkHidden + i * 8);
                const float4 x0 = xp[0], x1 = xp[1];
                acc0[m] = __hfma2(w0, __floats2half2_rn(x0.x, x0.y), acc0[m]);
                acc1[m] = __hfma2(w1, __floats2half2_rn(x0.z, x0.w), acc1[m]);
                acc0[m] = __hfma2(w2, __floats2half2_rn(x1.x, x1.y), acc0[m]);
                acc1[m] = __hfma2(w3, __floats2half2_rn(x1.z, x1.w), acc1[m]);
            }
        }
    }
    __shared__ float s_w[MAX_M][4];
#pragma unroll
    for (int m = 0; m < MAX_M; ++m) {
        if (m < M) {
            const float2 f0 = __half22float2(acc0[m]), f1 = __half22float2(acc1[m]);
            float s = warp_sum((f0.x + f0.y) + (f1.x + f1.y));
            if (lane == 0) s_w[m][warp] = s;
        }
    }
    __syncthreads();
    if (tid < M) {
        r_logits[(size_t)tid * kInkExperts + e] = s_w[tid][0] + s_w[tid][1] + s_w[tid][2] + s_w[tid][3];
    }
}

// ============================================================================
// 6. High-Level RTX 5090 MoE Launcher (Placed NEXT TO launch_ink_moe!)
// ============================================================================
inline void launch_ink_moe_rtx5090(
    cudaStream_t st,
    cublasHandle_t cublas,
    const uint8_t* sec4_base,
    const InkExpertDev* experts,
    const InkLayerMoeInfo& info,
    const float* gate_w,
    const float* gate_b,
    const float* x_f32,
    float* y_out,
    int M,
    InkMoeWorkspace& ws,
    float* d_scratch_f32,       // >= 4 * M * 128 floats for parallel bin fill (e.g. d_ctrl_gain)
    __half2* d_scratch_h2,      // >= (M * 1152 + M * 8 * 576) half2s for pre-converted small-M acts (e.g. d_x_bf16)
    float route_scale,
    bool force_exact_dec,
    bool run_reduce8,
    bool use_atomic_out = false,
    const __half2* w_router_h2 = nullptr) {
    // 1. Router (FP16 fast path for M<=16 under !EXACT; exact F32 baseline otherwise!)
    if (M <= 16 && w_router_h2 != nullptr && !rtx5090::exact_mode_on()) {
        if (M == 1)      k_ink_router_gemv_fp16<1><<<kInkExperts, 128, 0, st>>>(x_f32, w_router_h2, ws.r_logits, M);
        else if (M <= 4) k_ink_router_gemv_fp16<4><<<kInkExperts, 128, 0, st>>>(x_f32, w_router_h2, ws.r_logits, M);
        else if (M <= 8) k_ink_router_gemv_fp16<8><<<kInkExperts, 128, 0, st>>>(x_f32, w_router_h2, ws.r_logits, M);
        else             k_ink_router_gemv_fp16<16><<<kInkExperts, 128, 0, st>>>(x_f32, w_router_h2, ws.r_logits, M);
        k_ink_router_select_topk8<<<M, 32, 0, st>>>(ws.r_logits, gate_b, ws.topk_idx, ws.topk_w, route_scale, M, InkTailArgs{});   // tail skip (opt-in, f-tailskip) is not wired into the 5090 fast router path
        ink_moe_mark(0);
    } else {
        launch_ink_moe_router(st, cublas, gate_w, gate_b, x_f32, M, ws, route_scale, force_exact_dec);
    }

    static const bool m1_tc_on = [] {
        const char* e = std::getenv("STRATA_INK_SM120_M1_TC");
        return e && e[0] == '1';
    }();
    // 2a. Dedicated Zero-Bin Zero-Smem M=1 SF4/MX8 L1-Direct Kernel (or optional sm_120 NVFP4 TC via STRATA_INK_SM120_M1_TC=1!)
    if (M == 1 && !m1_tc_on && d_scratch_h2 != nullptr && !info.has_sk && !info.has_gguf && (info.has_sf4 || info.has_mx8)) {
        ink_moe_mark(1);
        __half2* x_h2 = d_scratch_h2;
        __half2* h_h2 = d_scratch_h2 + (kInkHidden / 2);
        k_ink_f32_to_h2<<<(kInkHidden / 2 + 255) / 256, 256, 0, st>>>(x_f32, x_h2, kInkHidden / 2);
        static const int m1_dn_pf = [] {   // PERF-STEP: STRATA_INK_M1_DNPF=1 (default 0 until measured)
            const char* e = std::getenv("STRATA_INK_M1_DNPF");
            return (e != nullptr && e[0] == '1') ? 1 : 0;
        }();
        k_ink_moe_gu_m1_l1<<<dim3(kInkInter / 16, kInkTopK), 256, 0, st>>>(sec4_base, experts, ws.topk_idx, x_h2, h_h2, m1_dn_pf);
        ink_moe_mark(2);
        k_ink_moe_dn_m1_l1<<<dim3((kInkHidden / 2) / 16, kInkTopK), 256, 0, st>>>(sec4_base, experts, ws.topk_idx, ws.topk_w, h_h2, ws.y_pair);
        ink_moe_mark(3);
    } else {
    // 2b. Bin + Dispatch SF4 & MX8 Experts (sm_120 NVFP4/MX8 TC for M>=2 — or M>=1 when M1_TC=1 — when !EXACT!)
    const bool use_sm120_tc = (sm120::available() && ((!force_exact_dec && M >= 24) || (!rtx5090::exact_mode_on() && M >= (m1_tc_on ? 1 : 2))));
    if (use_sm120_tc && !info.has_sk && !info.has_gguf && (info.has_sf4 || info.has_mx8)) {
        // tile size and kernel variants per batch range come from the card's table row (strata/core/ink_sm120_table.hpp)
        const Sm120MoeSel moe_sel = sm120_moe_sel(sm120_cfg_current(), M);
        const int tile_rows = moe_sel.tile_rows;
        const bool do_atomic = (use_atomic_out && tile_rows == 128 && y_out != nullptr);
        if (M >= 256 && d_scratch_f32 != nullptr) {
            const int P = M * kInkTopK;
            INK_CK(cudaMemsetAsync(ws.tb.cnt, 0, 128 * sizeof(int32_t), st));
            k_ink_moe_bin_par_count<<<32, 256, 0, st>>>(experts, ws.topk_idx, P, ws.tb.cnt);
            k_ink_moe_bin_par_prefix<<<1, 128, 0, st>>>(tile_rows, ws.tb, reinterpret_cast<int32_t*>(d_scratch_f32));
            k_ink_moe_bin_par_scatter<<<32, 256, 0, st>>>(experts, ws.topk_idx, P, reinterpret_cast<int32_t*>(d_scratch_f32), ws.tb);
        } else {
            k_ink_moe_bin<<<1, 128, 0, st>>>(experts, ws.topk_idx, M, tile_rows, ws.tb);
        }
        ink_moe_mark(1);
        const int mt = ink_moe_max_tiles(M, tile_rows);
        if (do_atomic) INK_CK(cudaMemsetAsync(y_out, 0, (size_t)M * kInkHidden * sizeof(float), st));
        if (info.has_sf4) {
            sm120::quantize_act_fp4(x_f32, M, kInkHidden, ws.xa_fp4, st, moe_sel.row_quant);
            sm120::moe_sf4_gu_tc(sec4_base, experts, ws.tb, ws.xa_fp4, ws.ha_fp4, mt, tile_rows, st, &moe_sel);
            ink_moe_mark(2);
            if (do_atomic) sm120::moe_sf4_dn_tc_atomic(sec4_base, experts, ws.tb, ws.ha_fp4, ws.topk_w, y_out, mt, st);
            else           sm120::moe_sf4_dn_tc(sec4_base, experts, ws.tb, ws.ha_fp4, ws.topk_w, ws.y_pair, mt, tile_rows, st, false, &moe_sel);
            ink_moe_mark(3);
        }
        if (info.has_mx8) {
            sm120::quantize_act_mx8(x_f32, M, kInkHidden, ws.xq_mx8, ws.xsc_mx8, st);
            sm120::moe_mx8_gu_tc(sec4_base, experts, ws.tb, ws.xq_mx8, ws.xsc_mx8, ws.hq_mx8, ws.hsc_mx8, mt, tile_rows, st, &moe_sel);
            ink_moe_mark(2);
            if (do_atomic) sm120::moe_mx8_dn_tc_atomic(sec4_base, experts, ws.tb, ws.hq_mx8, ws.hsc_mx8, ws.topk_w, y_out, mt, st);
            else           sm120::moe_mx8_dn_tc(sec4_base, experts, ws.tb, ws.hq_mx8, ws.hsc_mx8, ws.topk_w, ws.y_pair, mt, tile_rows, st, &moe_sel);
            ink_moe_mark(3);
        }
        if (do_atomic) return;
    } else {
        launch_ink_moe_dispatch_only(st, sec4_base, experts, info, x_f32, M, ws, force_exact_dec, false);
    }
    }

    if (run_reduce8) {
        dim3 rgrid((kInkHidden + 255) / 256, M);
        k_ink_moe_reduce8<<<rgrid, 256, 0, st>>>(ws.y_pair, y_out, M);
    }
}

}  // namespace strata::ink::rtx5090
