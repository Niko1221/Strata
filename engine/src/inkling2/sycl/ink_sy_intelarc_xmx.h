// ink_sy_intelarc_xmx.h - Zero-oneMKL Native SG16 joint_matrix XMX, Vectorized LSC, GPU Active-Expert Planner,
// Paged FP8_E4M3 KV, BF16 ShortConv, Intra-Chunk SWA Fix, and Top-128 Candidate Extraction for Intel Arc SYCL.
// CRITICAL BATTLEMAGE B70 RULE: 100% INK_REQD_SG16 (ZERO reqd_sub_group_size(8) anywhere!).
// MIT License - Copyright (c) 2026 Strata contributors
#pragma once

#include <sycl/sycl.hpp>
#if __has_include(<sycl/ext/oneapi/matrix/matrix.hpp>)
#include <sycl/ext/oneapi/matrix/matrix.hpp>
#define INK_SY_HAS_JOINT_MATRIX 1
#else
#define INK_SY_HAS_JOINT_MATRIX 0
#endif

#include <algorithm>
#include <cmath>
#include <cstdint>

#include "ink_sy_codecs.h"
#include "ink_sy_common.h"
#include "ink_sy_intelarc_cfg.h"
#include "ink_sy_ops.h"

namespace ink_sycl {

// ============================================================================
// 1. FP8_E4M3 <-> F32 & BF16 <-> F32 Helpers (device-safe, branch-light)
// ============================================================================

inline uint8_t f32_to_e4m3(float x) {
    if (!sycl::isfinite(x)) return 0u;
    const uint8_t sign = (x < 0.0f) ? 0x80u : 0x00u;
    float ax = sycl::fabs(x);
    if (ax >= 448.0f) return (uint8_t)(sign | 0x7Eu);
    if (ax < (1.0f / 1024.0f)) return sign;
    if (ax < (1.0f / 64.0f)) {
        const int m = (int)sycl::rint(ax * 512.0f);
        if (m <= 0) return sign;
        if (m >= 8) return (uint8_t)(sign | 0x08u);
        return (uint8_t)(sign | (uint8_t)m);
    }
    const uint32_t bits = sycl::bit_cast<uint32_t>(ax);
    int exp = (int)((bits >> 23) & 0xFFu) - 120;  // 127 - 7
    uint32_t mant = (bits >> 20) & 0x7u;
    const uint32_t rem = bits & 0xFFFFFu;
    if (rem > 0x80000u || (rem == 0x80000u && (mant & 1u))) {
        ++mant;
        if (mant == 8u) { mant = 0u; ++exp; }
    }
    if (exp <= 0) return sign;
    if (exp >= 15) {
        if (exp > 15 || mant >= 7u) return (uint8_t)(sign | 0x7Eu);
    }
    return (uint8_t)(sign | ((uint32_t)exp << 3) | mant);
}

inline uint16_t f32_to_bf16(float x) {
    const uint32_t bits = sycl::bit_cast<uint32_t>(x);
    const uint32_t lsb = (bits >> 16) & 1u;
    const uint32_t rounded = bits + 0x7FFFu + lsb;
    return (uint16_t)(rounded >> 16);
}

// 1-cycle SG16 register-LUT for NVFP4 E2M1 (each lane l in 0..15 holds kE2M1[l])
inline float e2m1_sg16_lane_val(int lane) {
    constexpr float kLut[16] = {
         0.0f,  0.5f,  1.0f,  1.5f,  2.0f,  3.0f,  4.0f,  6.0f,
        -0.0f, -0.5f, -1.0f, -1.5f, -2.0f, -3.0f, -4.0f, -6.0f
    };
    return kLut[lane & 15];
}

// ============================================================================
// 2. Native Zero-oneMKL SG16 joint_matrix XMX Kernels (FP16 8x16x16 & INT8 8x16x32)
//    Compile-time target guard: Battlemage Xe2 / PVC (SIMD16 DPAS) compiles native
//    joint_matrix<..., 8, 16, 16/32>; Alchemist acm_g10 (SIMD8 DPAS) & portable spir64
//    compile a clean INK_REQD_SG16 SLM 4x4 tiled fallback (0 IGC FPEs on acm_g10!).
// ============================================================================

#if INK_SY_HAS_JOINT_MATRIX

#if defined(__SYCL_DEVICE_ONLY__) && ( \
    (defined(__SYCL_TARGET_INTEL_GPU_BMG_G21__) && __SYCL_TARGET_INTEL_GPU_BMG_G21__ == 1) || \
    (defined(__SYCL_TARGET_INTEL_GPU_BMG_G31__) && __SYCL_TARGET_INTEL_GPU_BMG_G31__ == 1) || \
    (defined(__SYCL_TARGET_INTEL_GPU_LNL_M__)   && __SYCL_TARGET_INTEL_GPU_LNL_M__   == 1) || \
    (defined(__SYCL_TARGET_INTEL_GPU_PTL_H__)   && __SYCL_TARGET_INTEL_GPU_PTL_H__   == 1) || \
    (defined(__SYCL_TARGET_INTEL_GPU_PTL_U__)   && __SYCL_TARGET_INTEL_GPU_PTL_U__   == 1) || \
    (defined(__SYCL_TARGET_INTEL_GPU_PVC__)     && __SYCL_TARGET_INTEL_GPU_PVC__     == 1))
#define INK_SY_DEV_JM_SG16 1
#else
#define INK_SY_DEV_JM_SG16 0
#endif

// Native SG16 joint_matrix FP16 DPAS GEMM: Y[M, N] = X_f16[M, K] @ W_f16[N, K]^T (FP32 accum!)
// Tile: BM=32, BN=64, BK=16 | WG = 128 threads = 8 SG16 (4 along M x 2 along N, each SG owns 8x32 = 2x(8x16) acc!)
// Zero oneMKL linkage; strictly INK_REQD_SG16!
inline void k_jm_gemm_f16_nt_sg16(
    sycl::queue& q,
    const half* X_f16, const half* W_f16, float* Y_f32,
    int M, int N, int K, int ldY = -1) {
    namespace jm = sycl::ext::oneapi::experimental::matrix;
    const int y_ld = (ldY > 0) ? ldY : N;
    const int nbm = (M + 31) / 32;
    const int nbn = (N + 63) / 64;
    q.submit([&](sycl::handler& h) {
        sycl::local_accessor<half, 1> sA(32 * 16, h);   // [32, 16] row-major
        sycl::local_accessor<half, 1> sB(16 * 64, h);   // [16, 64] row-major (transposed from W[64, 16])
        sycl::local_accessor<float, 1> sC(32 * 64, h);  // staging for masked tail store
        h.parallel_for(NR((size_t)nbm * nbn, 128), [=](sycl::nd_item<1> it) INK_REQD_SG16 {
            const int gid = (int)it.get_group(0);
            const int bn = gid % nbn;
            const int bm = gid / nbn;
            const int tid = (int)it.get_local_id(0);
            const int sgi = tid >> 4;   // 0..7
            const int lane = tid & 15;
            const int m_base = bm * 32;
            const int n_base = bn * 64;
#if INK_SY_DEV_JM_SG16
            auto sg = it.get_sub_group();
            const int sg_m = sgi >> 1;  // 0..3 -> 8 rows each = 32
            const int sg_n = sgi & 1;   // 0..1 -> 32 cols each = 64
            jm::joint_matrix<sycl::sub_group, float, jm::use::accumulator, 8, 16> c0, c1;
            jm::joint_matrix_fill(sg, c0, 0.0f);
            jm::joint_matrix_fill(sg, c1, 0.0f);
#else
            float acc[4][4] = {};
            const int r0 = sgi * 4;
            const int c0 = lane * 4;
#endif
            for (int k0 = 0; k0 < K; k0 += 16) {
                for (int idx = tid; idx < 32 * 16; idx += 128) {
                    const int r = idx >> 4, k = idx & 15, gm = m_base + r;
                    sA[idx] = (gm < M && k0 + k < K) ? X_f16[(size_t)gm * K + k0 + k] : (half)0.0f;
                }
                for (int idx = tid; idx < 64 * 16; idx += 128) {
                    const int n = idx >> 4, k = idx & 15, gn = n_base + n;
                    sB[k * 64 + n] = (gn < N && k0 + k < K) ? W_f16[(size_t)gn * K + k0 + k] : (half)0.0f;
                }
                sycl::group_barrier(it.get_group());
#if INK_SY_DEV_JM_SG16
                jm::joint_matrix<sycl::sub_group, half, jm::use::a, 8, 16, jm::layout::row_major> a_m;
                jm::joint_matrix<sycl::sub_group, half, jm::use::b, 16, 16, jm::layout::row_major> b0, b1;
                auto pA = sA.template get_multi_ptr<sycl::access::decorated::no>() + sg_m * 8 * 16;
                auto pB0 = sB.template get_multi_ptr<sycl::access::decorated::no>() + sg_n * 32;
                jm::joint_matrix_load(sg, a_m, pA, 16);
                jm::joint_matrix_load(sg, b0, pB0, 64);
                jm::joint_matrix_load(sg, b1, pB0 + 16, 64);
                jm::joint_matrix_mad(sg, c0, a_m, b0, c0);
                jm::joint_matrix_mad(sg, c1, a_m, b1, c1);
#else
                for (int k = 0; k < 16; ++k) {
                    float bv[4] = {(float)sB[k * 64 + c0], (float)sB[k * 64 + c0 + 1],
                                   (float)sB[k * 64 + c0 + 2], (float)sB[k * 64 + c0 + 3]};
#pragma unroll
                    for (int ri = 0; ri < 4; ++ri) {
                        const float av = (float)sA[(r0 + ri) * 16 + k];
                        acc[ri][0] += av * bv[0]; acc[ri][1] += av * bv[1];
                        acc[ri][2] += av * bv[2]; acc[ri][3] += av * bv[3];
                    }
                }
#endif
                sycl::group_barrier(it.get_group());
            }
#if INK_SY_DEV_JM_SG16
            if (m_base + 32 <= M && n_base + 64 <= N) {
                auto pY = sycl::address_space_cast<sycl::access::address_space::global_space, sycl::access::decorated::no>(
                    Y_f32 + (size_t)(m_base + sg_m * 8) * y_ld + (n_base + sg_n * 32));
                jm::joint_matrix_store(sg, c0, pY, (size_t)y_ld, jm::layout::row_major);
                jm::joint_matrix_store(sg, c1, pY + 16, (size_t)y_ld, jm::layout::row_major);
            } else {
                auto pS = sC.template get_multi_ptr<sycl::access::decorated::no>() + sg_m * 8 * 64 + sg_n * 32;
                jm::joint_matrix_store(sg, c0, pS, 64, jm::layout::row_major);
                jm::joint_matrix_store(sg, c1, pS + 16, 64, jm::layout::row_major);
                sycl::group_barrier(it.get_group());
                for (int idx = tid; idx < 32 * 64; idx += 128) {
                    const int r = idx >> 6, c = idx & 63;
                    const int gm = m_base + r, gn = n_base + c;
                    if (gm < M && gn < N) Y_f32[(size_t)gm * y_ld + gn] = sC[idx];
                }
            }
#else
            (void)sC;
#pragma unroll
            for (int ri = 0; ri < 4; ++ri) {
                const int gm = m_base + r0 + ri;
                if (gm < M) {
#pragma unroll
                    for (int ci = 0; ci < 4; ++ci) {
                        const int gn = n_base + c0 + ci;
                        if (gn < N) Y_f32[(size_t)gm * y_ld + gn] = acc[ri][ci];
                    }
                }
            }
#endif
        });
    });
}

// Native SG16 joint_matrix INT8 DPAS GEMM (8x16x32): Y[M, N] = (X_s8[M, K] @ W_s8[N, K]^T) * xsc[m] * wsc[n]
// Zero global dequant buffer! Reads int8_t weights directly into SLM [32, 64] transposed tiles!
inline void k_jm_gemm_q8row_s8_sg16(
    sycl::queue& q,
    const uint8_t* W, int N, int K, int N_total,
    const int8_t* X_s8, const float* xsc, float* Y_f32, int M) {
    namespace jm = sycl::ext::oneapi::experimental::matrix;
    const int n_scales = (N_total > 0) ? N_total : N;
    const float* wsc = reinterpret_cast<const float*>(W);
    const int8_t* W_s8 = reinterpret_cast<const int8_t*>(W + (size_t)n_scales * 4);
    const int nbm = (M + 31) / 32;
    const int nbn = (N + 63) / 64;
    q.submit([&](sycl::handler& h) {
        sycl::local_accessor<int8_t, 1> sA(32 * 32, h);    // [32, 32] row-major
        sycl::local_accessor<int8_t, 1> sB(32 * 64, h);    // [32, 64] row-major (transposed from W[64, 32])
        sycl::local_accessor<int32_t, 1> sC(32 * 64, h);   // int32 -> scale -> float
        h.parallel_for(NR((size_t)nbm * nbn, 128), [=](sycl::nd_item<1> it) INK_REQD_SG16 {
            const int gid = (int)it.get_group(0);
            const int bn = gid % nbn;
            const int bm = gid / nbn;
            const int tid = (int)it.get_local_id(0);
            const int sgi = tid >> 4;
            const int lane = tid & 15;
            const int m_base = bm * 32;
            const int n_base = bn * 64;
#if INK_SY_DEV_JM_SG16
            auto sg = it.get_sub_group();
            const int sg_m = sgi >> 1;
            const int sg_n = sgi & 1;
            jm::joint_matrix<sycl::sub_group, int32_t, jm::use::accumulator, 8, 16> c0, c1;
            jm::joint_matrix_fill(sg, c0, 0);
            jm::joint_matrix_fill(sg, c1, 0);
#else
            int32_t acc[4][4] = {};
            const int r0 = sgi * 4;
            const int c0 = lane * 4;
#endif
            for (int k0 = 0; k0 < K; k0 += 32) {
                for (int idx = tid; idx < 32 * 32; idx += 128) {
                    const int r = idx >> 5, k = idx & 31, gm = m_base + r;
                    sA[idx] = (gm < M && k0 + k < K) ? X_s8[(size_t)gm * K + k0 + k] : (int8_t)0;
                }
                for (int idx = tid; idx < 64 * 32; idx += 128) {
                    const int n = idx >> 5, k = idx & 31, gn = n_base + n;
                    sB[k * 64 + n] = (gn < N && k0 + k < K) ? W_s8[(size_t)gn * K + k0 + k] : (int8_t)0;
                }
                sycl::group_barrier(it.get_group());
#if INK_SY_DEV_JM_SG16
                jm::joint_matrix<sycl::sub_group, int8_t, jm::use::a, 8, 32, jm::layout::row_major> a_m;
                jm::joint_matrix<sycl::sub_group, int8_t, jm::use::b, 32, 16, jm::layout::row_major> b0, b1;
                auto pA = sA.template get_multi_ptr<sycl::access::decorated::no>() + sg_m * 8 * 32;
                auto pB0 = sB.template get_multi_ptr<sycl::access::decorated::no>() + sg_n * 32;
                jm::joint_matrix_load(sg, a_m, pA, 32);
                jm::joint_matrix_load(sg, b0, pB0, 64);
                jm::joint_matrix_load(sg, b1, pB0 + 16, 64);
                jm::joint_matrix_mad(sg, c0, a_m, b0, c0);
                jm::joint_matrix_mad(sg, c1, a_m, b1, c1);
#else
                for (int k = 0; k < 32; ++k) {
                    int32_t bv[4] = {(int32_t)sB[k * 64 + c0], (int32_t)sB[k * 64 + c0 + 1],
                                     (int32_t)sB[k * 64 + c0 + 2], (int32_t)sB[k * 64 + c0 + 3]};
#pragma unroll
                    for (int ri = 0; ri < 4; ++ri) {
                        const int32_t av = (int32_t)sA[(r0 + ri) * 32 + k];
                        acc[ri][0] += av * bv[0]; acc[ri][1] += av * bv[1];
                        acc[ri][2] += av * bv[2]; acc[ri][3] += av * bv[3];
                    }
                }
#endif
                sycl::group_barrier(it.get_group());
            }
#if INK_SY_DEV_JM_SG16
            auto pS = sC.template get_multi_ptr<sycl::access::decorated::no>() + sg_m * 8 * 64 + sg_n * 32;
            jm::joint_matrix_store(sg, c0, pS, 64, jm::layout::row_major);
            jm::joint_matrix_store(sg, c1, pS + 16, 64, jm::layout::row_major);
            sycl::group_barrier(it.get_group());
            for (int idx = tid; idx < 32 * 64; idx += 128) {
                const int r = idx >> 6, c = idx & 63;
                const int gm = m_base + r, gn = n_base + c;
                if (gm < M && gn < N) {
                    Y_f32[(size_t)gm * N + gn] = (float)sC[idx] * (xsc[gm] * wsc[gn]);
                }
            }
#else
            (void)sC;
#pragma unroll
            for (int ri = 0; ri < 4; ++ri) {
                const int gm = m_base + r0 + ri;
                if (gm < M) {
                    const float xs = xsc[gm];
#pragma unroll
                    for (int ci = 0; ci < 4; ++ci) {
                        const int gn = n_base + c0 + ci;
                        if (gn < N) Y_f32[(size_t)gm * N + gn] = (float)acc[ri][ci] * (xs * wsc[gn]);
                    }
                }
            }
#endif
        });
    });
}

// On-the-fly SLM-dequantized BF16 -> joint_matrix FP16 DPAS GEMM (0 global w_f16 buffer!)
inline void k_jm_gemm_bf16_sg16(
    sycl::queue& q,
    const uint16_t* W_bf16, int N, int K,
    const float* X_f32, float* Y_f32, int M) {
    namespace jm = sycl::ext::oneapi::experimental::matrix;
    const int nbm = (M + 31) / 32;
    const int nbn = (N + 63) / 64;
    q.submit([&](sycl::handler& h) {
        sycl::local_accessor<half, 1> sA(32 * 16, h);
        sycl::local_accessor<half, 1> sB(16 * 64, h);
        sycl::local_accessor<float, 1> sC(32 * 64, h);
        h.parallel_for(NR((size_t)nbm * nbn, 128), [=](sycl::nd_item<1> it) INK_REQD_SG16 {
            const int gid = (int)it.get_group(0);
            const int bn = gid % nbn, bm = gid / nbn;
            const int tid = (int)it.get_local_id(0);
            const int sgi = tid >> 4, lane = tid & 15;
            const int m_base = bm * 32, n_base = bn * 64;
#if INK_SY_DEV_JM_SG16
            auto sg = it.get_sub_group();
            const int sg_m = sgi >> 1, sg_n = sgi & 1;
            jm::joint_matrix<sycl::sub_group, float, jm::use::accumulator, 8, 16> c0, c1;
            jm::joint_matrix_fill(sg, c0, 0.0f);
            jm::joint_matrix_fill(sg, c1, 0.0f);
#else
            float acc[4][4] = {};
            const int r0 = sgi * 4, c0 = lane * 4;
#endif
            for (int k0 = 0; k0 < K; k0 += 16) {
                for (int idx = tid; idx < 32 * 16; idx += 128) {
                    const int r = idx >> 4, k = idx & 15, gm = m_base + r;
                    sA[idx] = (gm < M && k0 + k < K) ? (half)X_f32[(size_t)gm * K + k0 + k] : (half)0.0f;
                }
                for (int idx = tid; idx < 64 * 16; idx += 128) {
                    const int n = idx >> 4, k = idx & 15, gn = n_base + n;
                    sB[k * 64 + n] = (gn < N && k0 + k < K) ? (half)bf16_to_f32(W_bf16[(size_t)gn * K + k0 + k]) : (half)0.0f;
                }
                sycl::group_barrier(it.get_group());
#if INK_SY_DEV_JM_SG16
                jm::joint_matrix<sycl::sub_group, half, jm::use::a, 8, 16, jm::layout::row_major> a_m;
                jm::joint_matrix<sycl::sub_group, half, jm::use::b, 16, 16, jm::layout::row_major> b0, b1;
                auto pA = sA.template get_multi_ptr<sycl::access::decorated::no>() + sg_m * 8 * 16;
                auto pB0 = sB.template get_multi_ptr<sycl::access::decorated::no>() + sg_n * 32;
                jm::joint_matrix_load(sg, a_m, pA, 16);
                jm::joint_matrix_load(sg, b0, pB0, 64);
                jm::joint_matrix_load(sg, b1, pB0 + 16, 64);
                jm::joint_matrix_mad(sg, c0, a_m, b0, c0);
                jm::joint_matrix_mad(sg, c1, a_m, b1, c1);
#else
                for (int k = 0; k < 16; ++k) {
                    float bv[4] = {(float)sB[k * 64 + c0], (float)sB[k * 64 + c0 + 1],
                                   (float)sB[k * 64 + c0 + 2], (float)sB[k * 64 + c0 + 3]};
#pragma unroll
                    for (int ri = 0; ri < 4; ++ri) {
                        const float av = (float)sA[(r0 + ri) * 16 + k];
                        acc[ri][0] += av * bv[0]; acc[ri][1] += av * bv[1];
                        acc[ri][2] += av * bv[2]; acc[ri][3] += av * bv[3];
                    }
                }
#endif
                sycl::group_barrier(it.get_group());
            }
#if INK_SY_DEV_JM_SG16
            if (m_base + 32 <= M && n_base + 64 <= N) {
                auto pY = sycl::address_space_cast<sycl::access::address_space::global_space, sycl::access::decorated::no>(
                    Y_f32 + (size_t)(m_base + sg_m * 8) * N + (n_base + sg_n * 32));
                jm::joint_matrix_store(sg, c0, pY, (size_t)N, jm::layout::row_major);
                jm::joint_matrix_store(sg, c1, pY + 16, (size_t)N, jm::layout::row_major);
            } else {
                auto pS = sC.template get_multi_ptr<sycl::access::decorated::no>() + sg_m * 8 * 64 + sg_n * 32;
                jm::joint_matrix_store(sg, c0, pS, 64, jm::layout::row_major);
                jm::joint_matrix_store(sg, c1, pS + 16, 64, jm::layout::row_major);
                sycl::group_barrier(it.get_group());
                for (int idx = tid; idx < 32 * 64; idx += 128) {
                    const int r = idx >> 6, c = idx & 63;
                    const int gm = m_base + r, gn = n_base + c;
                    if (gm < M && gn < N) Y_f32[(size_t)gm * N + gn] = sC[idx];
                }
            }
#else
            (void)sC;
#pragma unroll
            for (int ri = 0; ri < 4; ++ri) {
                const int gm = m_base + r0 + ri;
                if (gm < M) {
#pragma unroll
                    for (int ci = 0; ci < 4; ++ci) {
                        const int gn = n_base + c0 + ci;
                        if (gn < N) Y_f32[(size_t)gm * N + gn] = acc[ri][ci];
                    }
                }
            }
#endif
        });
    });
}

// On-the-fly SLM-dequantized MX8 -> joint_matrix FP16 DPAS GEMM (0 global w_f16 buffer!)
inline void k_jm_gemm_mx8_sg16(
    sycl::queue& q,
    const uint8_t* W_mx8, int N, int K, int N_total,
    const float* X_f32, float* Y_f32, int M) {
    namespace jm = sycl::ext::oneapi::experimental::matrix;
    const int n_tot = (N_total > 0) ? N_total : N;
    const int n_blk = K >> 5;
    const uint8_t* S_mx8 = W_mx8 + (size_t)n_tot * K;
    const int nbm = (M + 31) / 32;
    const int nbn = (N + 63) / 64;
    q.submit([&](sycl::handler& h) {
        sycl::local_accessor<half, 1> sA(32 * 16, h);
        sycl::local_accessor<half, 1> sB(16 * 64, h);
        sycl::local_accessor<float, 1> sC(32 * 64, h);
        h.parallel_for(NR((size_t)nbm * nbn, 128), [=](sycl::nd_item<1> it) INK_REQD_SG16 {
            const int gid = (int)it.get_group(0);
            const int bn = gid % nbn, bm = gid / nbn;
            const int tid = (int)it.get_local_id(0);
            const int sgi = tid >> 4, lane = tid & 15;
            const int m_base = bm * 32, n_base = bn * 64;
#if INK_SY_DEV_JM_SG16
            auto sg = it.get_sub_group();
            const int sg_m = sgi >> 1, sg_n = sgi & 1;
            jm::joint_matrix<sycl::sub_group, float, jm::use::accumulator, 8, 16> c0, c1;
            jm::joint_matrix_fill(sg, c0, 0.0f);
            jm::joint_matrix_fill(sg, c1, 0.0f);
#else
            float acc[4][4] = {};
            const int r0 = sgi * 4, c0 = lane * 4;
#endif
            for (int k0 = 0; k0 < K; k0 += 16) {
                const int blk = k0 >> 5;
                for (int idx = tid; idx < 32 * 16; idx += 128) {
                    const int r = idx >> 4, k = idx & 15, gm = m_base + r;
                    sA[idx] = (gm < M && k0 + k < K) ? (half)X_f32[(size_t)gm * K + k0 + k] : (half)0.0f;
                }
                for (int idx = tid; idx < 64 * 16; idx += 128) {
                    const int n = idx >> 4, k = idx & 15, gn = n_base + n;
                    if (gn < N && k0 + k < K) {
                        const float sc = ue8m0_to_f32(S_mx8[(size_t)gn * n_blk + blk]);
                        sB[k * 64 + n] = (half)(e4m3_to_f32(W_mx8[(size_t)gn * K + k0 + k]) * sc);
                    } else {
                        sB[k * 64 + n] = (half)0.0f;
                    }
                }
                sycl::group_barrier(it.get_group());
#if INK_SY_DEV_JM_SG16
                jm::joint_matrix<sycl::sub_group, half, jm::use::a, 8, 16, jm::layout::row_major> a_m;
                jm::joint_matrix<sycl::sub_group, half, jm::use::b, 16, 16, jm::layout::row_major> b0, b1;
                auto pA = sA.template get_multi_ptr<sycl::access::decorated::no>() + sg_m * 8 * 16;
                auto pB0 = sB.template get_multi_ptr<sycl::access::decorated::no>() + sg_n * 32;
                jm::joint_matrix_load(sg, a_m, pA, 16);
                jm::joint_matrix_load(sg, b0, pB0, 64);
                jm::joint_matrix_load(sg, b1, pB0 + 16, 64);
                jm::joint_matrix_mad(sg, c0, a_m, b0, c0);
                jm::joint_matrix_mad(sg, c1, a_m, b1, c1);
#else
                for (int k = 0; k < 16; ++k) {
                    float bv[4] = {(float)sB[k * 64 + c0], (float)sB[k * 64 + c0 + 1],
                                   (float)sB[k * 64 + c0 + 2], (float)sB[k * 64 + c0 + 3]};
#pragma unroll
                    for (int ri = 0; ri < 4; ++ri) {
                        const float av = (float)sA[(r0 + ri) * 16 + k];
                        acc[ri][0] += av * bv[0]; acc[ri][1] += av * bv[1];
                        acc[ri][2] += av * bv[2]; acc[ri][3] += av * bv[3];
                    }
                }
#endif
                sycl::group_barrier(it.get_group());
            }
#if INK_SY_DEV_JM_SG16
            if (m_base + 32 <= M && n_base + 64 <= N) {
                auto pY = sycl::address_space_cast<sycl::access::address_space::global_space, sycl::access::decorated::no>(
                    Y_f32 + (size_t)(m_base + sg_m * 8) * N + (n_base + sg_n * 32));
                jm::joint_matrix_store(sg, c0, pY, (size_t)N, jm::layout::row_major);
                jm::joint_matrix_store(sg, c1, pY + 16, (size_t)N, jm::layout::row_major);
            } else {
                auto pS = sC.template get_multi_ptr<sycl::access::decorated::no>() + sg_m * 8 * 64 + sg_n * 32;
                jm::joint_matrix_store(sg, c0, pS, 64, jm::layout::row_major);
                jm::joint_matrix_store(sg, c1, pS + 16, 64, jm::layout::row_major);
                sycl::group_barrier(it.get_group());
                for (int idx = tid; idx < 32 * 64; idx += 128) {
                    const int r = idx >> 6, c = idx & 63;
                    const int gm = m_base + r, gn = n_base + c;
                    if (gm < M && gn < N) Y_f32[(size_t)gm * N + gn] = sC[idx];
                }
            }
#else
            (void)sC;
#pragma unroll
            for (int ri = 0; ri < 4; ++ri) {
                const int gm = m_base + r0 + ri;
                if (gm < M) {
#pragma unroll
                    for (int ci = 0; ci < 4; ++ci) {
                        const int gn = n_base + c0 + ci;
                        if (gn < N) Y_f32[(size_t)gm * N + gn] = acc[ri][ci];
                    }
                }
            }
#endif
        });
    });
}

// ============================================================================
// 3. GPU-Side Active-Expert Bucket Planner + Zero-d2h joint_matrix MoE Prefill!
//    Zero CPU d2h stalls, zero global sc.w_f16!
// ============================================================================

struct InkSyMoeBuckets {
    int sm_uexp[128];
    int sm_uoff[128];
    int sm_ucnt[128];
    int xm_uexp[128];
    int xm_uoff[128];
    int xm_ucnt[128];
    int n_sm;
    int n_xm;
};

// Single 128-thread WG (8 SG16): partitions all P = M*8 pairs into small-bucket (< xmx_min) and XMX-bucket (>= xmx_min)
// strictly on GPU with zero host synchronization!
inline void k_moe_plan_active_sg16(
    sycl::queue& q, const int* topk_idx, int P, int xmx_min,
    InkSyMoeBuckets* bk, int* pair_list, const InkSyExpertDev* ex_layer = nullptr) {
    q.submit([&](sycl::handler& h) {
        sycl::local_accessor<int, 1> sg_sm(8, h);
        sycl::local_accessor<int, 1> sg_xm(8, h);
        sycl::local_accessor<int, 1> sg_p(8, h);
        h.parallel_for(NR(1, 128), [=](sycl::nd_item<1> it) INK_REQD_SG16 {
            const int e = (int)it.get_local_id(0);
            auto sg = it.get_sub_group();
            const int sgi = (int)sg.get_group_linear_id();
            const int lane = (int)sg.get_local_linear_id();
            const bool owned = (ex_layer == nullptr) || (ex_layer[e].body_ptr != nullptr);
            int c = 0;
            if (owned) {
                for (int i = 0; i < P; ++i) if (topk_idx[i] == e) ++c;
            }
            const int is_xm = (c >= xmx_min) ? 1 : 0;
            const int is_sm = (c > 0 && c < xmx_min) ? 1 : 0;
            const int ex_sm = sycl::exclusive_scan_over_group(sg, is_sm, sycl::plus<int>());
            const int ex_xm = sycl::exclusive_scan_over_group(sg, is_xm, sycl::plus<int>());
            const int ex_p  = sycl::exclusive_scan_over_group(sg, c, sycl::plus<int>());
            if (lane == 15) {
                sg_sm[sgi] = ex_sm + is_sm;
                sg_xm[sgi] = ex_xm + is_xm;
                sg_p[sgi]  = ex_p + c;
            }
            sycl::group_barrier(it.get_group());
            int bsm = 0, bxm = 0, bp = 0;
            for (int w = 0; w < sgi; ++w) { bsm += sg_sm[w]; bxm += sg_xm[w]; bp += sg_p[w]; }
            const int psm = bsm + ex_sm, pxm = bxm + ex_xm, pp = bp + ex_p;
            if (c > 0) {
                if (is_xm) { bk->xm_uexp[pxm] = e; bk->xm_uoff[pxm] = pp; bk->xm_ucnt[pxm] = c; }
                else       { bk->sm_uexp[psm] = e; bk->sm_uoff[psm] = pp; bk->sm_ucnt[psm] = c; }
                int k = 0;
                for (int i = 0; i < P; ++i) if (topk_idx[i] == e) pair_list[pp + k++] = i;
            }
            if (e == 127) {
                bk->n_sm = psm + is_sm;
                bk->n_xm = pxm + is_xm;
            }
        });
    });
}

// GPU-driven batched MoE w13 + fused SITU using native SG16 joint_matrix FP16 DPAS (0 CPU d2h, 0 global w_f16!)
// Grid: Y = max_xm_exps (<= 128), X = 36 (36 * 64 = 2304 rows, paired 32 gate+up inter channels/WG!), WG = 128 (8 SG16).
// Processes all tokens in each expert's bucket in BM=32 tiles directly from x_in via pair_list!
inline void k_jm_moe_gu_situ_sg16(
    sycl::queue& q,
    const InkSyExpertDev* ex_layer, const float* x_in,
    const InkSyMoeBuckets* bk, const int* pair_list, int max_xm,
    float* h_act, const InkSyIqGridTable* iq_grids) {
    namespace jm = sycl::ext::oneapi::experimental::matrix;
    q.submit([&](sycl::handler& h) {
        sycl::local_accessor<half, 1> sA(32 * 16, h);    // [32, 16] token tile
        sycl::local_accessor<half, 1> sB(16 * 64, h);    // [16, 64] weight tile (64 interleaved gate/up rows)
        sycl::local_accessor<float, 1> sC(32 * 64, h);   // [32, 64] gate/up results -> SITU -> h_act
        h.parallel_for(NR2(36, (size_t)max_xm, 128), [=](sycl::nd_item<2> it) INK_REQD_SG16 {
            const int y = (int)it.get_group(0);
            if (y >= bk->n_xm) return;
            const int bn = (int)it.get_group(1);  // 0..35 -> rows bn*64 .. bn*64+63 (32 inter channels)
            const int tid = (int)it.get_local_id(1);
            const int sgi = tid >> 4, lane = tid & 15;
#if INK_SY_DEV_JM_SG16
            auto sg = it.get_sub_group();
            const int sg_m = sgi >> 1, sg_n = sgi & 1;
#endif
            const int e = bk->xm_uexp[y];
            const int off = bk->xm_uoff[y];
            const int n_tok = bk->xm_ucnt[y];
            const InkSyExpertDev ed = ex_layer[e];
            const uint8_t* body = ed.body_ptr;
            const bool is_sk_dn = (ed.dn_code != FMT_SF4 && ed.dn_code != FMT_MX8 && ed.dn_code != FMT_GGUF_RAW);
            const int row0 = bn * 64;

            for (int t0 = 0; t0 < n_tok; t0 += 32) {
                const int tm = sycl::min(32, n_tok - t0);
#if INK_SY_DEV_JM_SG16
                jm::joint_matrix<sycl::sub_group, float, jm::use::accumulator, 8, 16> c0, c1;
                jm::joint_matrix_fill(sg, c0, 0.0f);
                jm::joint_matrix_fill(sg, c1, 0.0f);
#else
                float acc[4][4] = {};
                const int r0 = sgi * 4, c0 = lane * 4;
#endif
                for (int k0 = 0; k0 < 2304; k0 += 16) {
                    // Load sA[32, 16] gathered from x_in
                    for (int idx = tid; idx < 32 * 16; idx += 128) {
                        const int r = idx >> 4, k = idx & 15;
                        if (r < tm) {
                            const int tk = pair_list[off + t0 + r] >> 3;
                            sA[idx] = (half)x_in[(size_t)tk * 2304 + k0 + k];
                        } else {
                            sA[idx] = (half)0.0f;
                        }
                    }
                    // On-the-fly decode 64 rows x 16 cols -> sB[16, 64]
                    if (ed.gu_code == FMT_SF4) {
                        const float* hdr = reinterpret_cast<const float*>(body);
                        const float outer = hdr[0] / hdr[1];
                        const uint8_t* pb = body + 16;
                        const uint8_t* sb = pb + 2654208;
                        const int blk = k0 >> 4;
                        // 128 threads: thread idx handles row = idx/2 (0..63), half-block = idx&1 (8 weights = 4 bytes)
                        const int nr = tid >> 1;
                        const int hi = (tid & 1) * 8;
                        const int gr = row0 + nr;
                        const float sc = e4m3_to_f32(sb[(size_t)gr * 144 + blk]) * outer;
                        const uint8_t* bp = pb + (size_t)gr * 1152 + blk * 8 + (tid & 1) * 4;
#pragma unroll
                        for (int j = 0; j < 4; ++j) {
                            const uint8_t b = bp[j];
                            sB[(hi + 2 * j)     * 64 + nr] = (half)(e2m1_lut(b & 15u) * sc);
                            sB[(hi + 2 * j + 1) * 64 + nr] = (half)(e2m1_lut(b >> 4)  * sc);
                        }
                    } else if (ed.gu_code == FMT_MX8) {
                        const uint8_t* sb = body + 5308416;
                        const int blk = k0 >> 5;
                        const int nr = tid >> 1;
                        const int hi = (tid & 1) * 8;
                        const int gr = row0 + nr;
                        const float sc = ue8m0_to_f32(sb[(size_t)gr * 72 + blk]);
                        const uint8_t* bp = body + (size_t)gr * 2304 + k0 + hi;
#pragma unroll
                        for (int j = 0; j < 8; ++j) {
                            sB[(hi + j) * 64 + nr] = (half)(e4m3_to_f32(bp[j]) * sc);
                        }
                    } else if (ed.gu_code == FMT_GGUF_RAW) {
                        const InkSyGgufExpertHeader* hdr = reinterpret_cast<const InkSyGgufExpertHeader*>(body);
                        const uint32_t gt = hdr->gu_type;
                        const size_t rb = sy_gguf_row_bytes(gt, 2304);
                        const int nr = tid >> 1;
                        const int hi = (tid & 1) * 8;
                        const int gr = row0 + nr;
                        const int src_row = (hdr->flags & 1u) ? gr : (((gr & 1) ? 1152 : 0) + (gr >> 1));
                        const float ext_sc = (gr & 1) ? hdr->up_scale : hdr->gate_scale;
                        const uint8_t* wr = body + 32 + (size_t)src_row * rb;
                        float wf[8];
                        sy_gguf_decode_8(gt, wr, k0 + hi, ext_sc, wf, iq_grids);
#pragma unroll
                        for (int j = 0; j < 8; ++j) sB[(hi + j) * 64 + nr] = (half)wf[j];
                    } else {
                        const int F = ed.gu_code;
                        const int SB = sk_sb_bytes(F);
                        const float H = sk_half(F);
                        const int sbi = k0 >> 8;
                        const int j_in_sb = k0 & 255;
                        const int g = j_in_sb >> 6;
                        const int nr = tid >> 1;
                        const int hi = (tid & 1) * 8;
                        const int gr = row0 + nr;
                        const uint8_t* blk = body + 2304 + (size_t)gr * 9ul * SB + (size_t)sbi * SB;
                        const float eff = h2f_bits(ld16(blk)) * ((float)blk[2 + g] * (1.0f / 255.0f));
                        int q8[8];
                        if (F == FMT_S2) sk_q8_sy<FMT_S2>(blk + 6, j_in_sb + hi, q8);
                        else if (F == FMT_S3) sk_q8_sy<FMT_S3>(blk + 6, j_in_sb + hi, q8);
                        else if (F == FMT_S4) sk_q8_sy<FMT_S4>(blk + 6, j_in_sb + hi, q8);
                        else if (F == FMT_S5) sk_q8_sy<FMT_S5>(blk + 6, j_in_sb + hi, q8);
                        else if (F == FMT_S6) sk_q8_sy<FMT_S6>(blk + 6, j_in_sb + hi, q8);
                        else sk_q8_sy<FMT_S8>(blk + 6, j_in_sb + hi, q8);
#pragma unroll
                        for (int j = 0; j < 8; ++j) sB[(hi + j) * 64 + nr] = (half)(((float)q8[j] - H) * eff);
                    }
                    sycl::group_barrier(it.get_group());
#if INK_SY_DEV_JM_SG16
                    jm::joint_matrix<sycl::sub_group, half, jm::use::a, 8, 16, jm::layout::row_major> a_m;
                    jm::joint_matrix<sycl::sub_group, half, jm::use::b, 16, 16, jm::layout::row_major> b0, b1;
                    auto pA = sA.template get_multi_ptr<sycl::access::decorated::no>() + sg_m * 8 * 16;
                    auto pB0 = sB.template get_multi_ptr<sycl::access::decorated::no>() + sg_n * 32;
                    jm::joint_matrix_load(sg, a_m, pA, 16);
                    jm::joint_matrix_load(sg, b0, pB0, 64);
                    jm::joint_matrix_load(sg, b1, pB0 + 16, 64);
                    jm::joint_matrix_mad(sg, c0, a_m, b0, c0);
                    jm::joint_matrix_mad(sg, c1, a_m, b1, c1);
#else
                    for (int k = 0; k < 16; ++k) {
                        float bv[4] = {(float)sB[k * 64 + c0], (float)sB[k * 64 + c0 + 1],
                                       (float)sB[k * 64 + c0 + 2], (float)sB[k * 64 + c0 + 3]};
#pragma unroll
                        for (int ri = 0; ri < 4; ++ri) {
                            const float av = (float)sA[(r0 + ri) * 16 + k];
                            acc[ri][0] += av * bv[0]; acc[ri][1] += av * bv[1];
                            acc[ri][2] += av * bv[2]; acc[ri][3] += av * bv[3];
                        }
                    }
#endif
                    sycl::group_barrier(it.get_group());
                }
#if INK_SY_DEV_JM_SG16
                auto pS = sC.template get_multi_ptr<sycl::access::decorated::no>() + sg_m * 8 * 64 + sg_n * 32;
                jm::joint_matrix_store(sg, c0, pS, 64, jm::layout::row_major);
                jm::joint_matrix_store(sg, c1, pS + 16, 64, jm::layout::row_major);
#else
#pragma unroll
                for (int ri = 0; ri < 4; ++ri) {
#pragma unroll
                    for (int ci = 0; ci < 4; ++ci) sC[(r0 + ri) * 64 + c0 + ci] = acc[ri][ci];
                }
#endif
                sycl::group_barrier(it.get_group());

                // Fused SITU + 1/sq_xd_dn -> h_act[pid, inter]
                for (int idx = tid; idx < tm * 32; idx += 128) {
                    const int r = idx >> 5;
                    const int c32 = idx & 31;
                    const int inter = bn * 32 + c32;
                    const int pid = pair_list[off + t0 + r];
                    const float inv_sq = is_sk_dn ? (1.0f / h2f_bits(ld16(body + 2 * inter))) : 1.0f;
                    const float g = sC[r * 64 + 2 * c32];
                    const float u = sC[r * 64 + 2 * c32 + 1];
                    h_act[(size_t)pid * 1152 + inter] = situ_act(g, u) * inv_sq;
                }
                sycl::group_barrier(it.get_group());
            }
        });
    });
}

// GPU-driven batched MoE w2 * router_w using native SG16 joint_matrix FP16 DPAS (0 CPU d2h, 0 global w_f16!)
inline void k_jm_moe_dn_sg16(
    sycl::queue& q,
    const InkSyExpertDev* ex_layer, const float* topk_w,
    const InkSyMoeBuckets* bk, const int* pair_list, int max_xm,
    const float* h_act, float* y_pairs, const InkSyIqGridTable* iq_grids) {
    namespace jm = sycl::ext::oneapi::experimental::matrix;
    q.submit([&](sycl::handler& h) {
        sycl::local_accessor<half, 1> sA(32 * 16, h);
        sycl::local_accessor<half, 1> sB(16 * 64, h);
        sycl::local_accessor<float, 1> sC(32 * 64, h);
        h.parallel_for(NR2(36, (size_t)max_xm, 128), [=](sycl::nd_item<2> it) INK_REQD_SG16 {
            const int y = (int)it.get_group(0);
            if (y >= bk->n_xm) return;
            const int bn = (int)it.get_group(1);  // 0..35 -> 64 output rows each = 2304
            const int tid = (int)it.get_local_id(1);
            const int sgi = tid >> 4, lane = tid & 15;
#if INK_SY_DEV_JM_SG16
            auto sg = it.get_sub_group();
            const int sg_m = sgi >> 1, sg_n = sgi & 1;
#endif
            const int e = bk->xm_uexp[y];
            const int off = bk->xm_uoff[y];
            const int n_tok = bk->xm_ucnt[y];
            const InkSyExpertDev ed = ex_layer[e];
            const uint8_t* body = ed.body_ptr;
            const int row0 = bn * 64;

            for (int t0 = 0; t0 < n_tok; t0 += 32) {
                const int tm = sycl::min(32, n_tok - t0);
#if INK_SY_DEV_JM_SG16
                jm::joint_matrix<sycl::sub_group, float, jm::use::accumulator, 8, 16> c0, c1;
                jm::joint_matrix_fill(sg, c0, 0.0f);
                jm::joint_matrix_fill(sg, c1, 0.0f);
#else
                float acc[4][4] = {};
                const int r0 = sgi * 4, c0 = lane * 4;
#endif
                for (int k0 = 0; k0 < 1152; k0 += 16) {
                    for (int idx = tid; idx < 32 * 16; idx += 128) {
                        const int r = idx >> 4, k = idx & 15;
                        if (r < tm) {
                            const int pid = pair_list[off + t0 + r];
                            sA[idx] = (half)h_act[(size_t)pid * 1152 + k0 + k];
                        } else {
                            sA[idx] = (half)0.0f;
                        }
                    }
                    if (ed.dn_code == FMT_SF4) {
                        const float* hdr = reinterpret_cast<const float*>(body);
                        const float outer = hdr[2] / hdr[3];
                        const uint8_t* pb = body + ed.dn_off;
                        const uint8_t* sb = pb + 1327104;
                        const int blk = k0 >> 4;
                        const int nr = tid >> 1;
                        const int hi = (tid & 1) * 8;
                        const int gr = row0 + nr;
                        const float sc = e4m3_to_f32(sb[(size_t)gr * 72 + blk]) * outer;
                        const uint8_t* bp = pb + (size_t)gr * 576 + blk * 8 + (tid & 1) * 4;
#pragma unroll
                        for (int j = 0; j < 4; ++j) {
                            const uint8_t b = bp[j];
                            sB[(hi + 2 * j)     * 64 + nr] = (half)(e2m1_lut(b & 15u) * sc);
                            sB[(hi + 2 * j + 1) * 64 + nr] = (half)(e2m1_lut(b >> 4)  * sc);
                        }
                    } else if (ed.dn_code == FMT_MX8) {
                        const uint8_t* wb = body + ed.dn_off;
                        const uint8_t* sb = wb + 2654208;
                        const int blk = k0 >> 5;
                        const int nr = tid >> 1;
                        const int hi = (tid & 1) * 8;
                        const int gr = row0 + nr;
                        const float sc = ue8m0_to_f32(sb[(size_t)gr * 36 + blk]);
                        const uint8_t* bp = wb + (size_t)gr * 1152 + k0 + hi;
#pragma unroll
                        for (int j = 0; j < 8; ++j) sB[(hi + j) * 64 + nr] = (half)(e4m3_to_f32(bp[j]) * sc);
                    } else if (ed.dn_code == FMT_GGUF_RAW) {
                        const InkSyGgufExpertHeader* hdr = reinterpret_cast<const InkSyGgufExpertHeader*>(body);
                        const uint32_t dt = hdr->dn_type;
                        const size_t rb = sy_gguf_row_bytes(dt, 1152);
                        const int nr = tid >> 1;
                        const int hi = (tid & 1) * 8;
                        const int gr = row0 + nr;
                        const uint8_t* wr = body + ed.dn_off + (size_t)gr * rb;
                        float wf[8];
                        sy_gguf_decode_8(dt, wr, k0 + hi, hdr->dn_scale, wf, iq_grids);
#pragma unroll
                        for (int j = 0; j < 8; ++j) sB[(hi + j) * 64 + nr] = (half)wf[j];
                    } else {
                        const int F = ed.dn_code;
                        const int SB = sk_sb_bytes(F);
                        const float H = sk_half(F);
                        const int nr = tid >> 1;
                        const int hi = (tid & 1) * 8;
                        const int gr = row0 + nr;
                        const int rp = gr >> 1;
                        const int flat = (gr & 1) ? (1152 + k0 + hi) : (k0 + hi);
                        const int sbi = flat >> 8;
                        const int j_in_sb = flat & 255;
                        const int g = j_in_sb >> 6;
                        const uint8_t* blk = body + ed.dn_off + (size_t)rp * 9ul * SB + (size_t)sbi * SB;
                        const float eff = h2f_bits(ld16(blk)) * ((float)blk[2 + g] * (1.0f / 255.0f));
                        int q8[8];
                        if (F == FMT_S2) sk_q8_sy<FMT_S2>(blk + 6, j_in_sb, q8);
                        else if (F == FMT_S3) sk_q8_sy<FMT_S3>(blk + 6, j_in_sb, q8);
                        else if (F == FMT_S4) sk_q8_sy<FMT_S4>(blk + 6, j_in_sb, q8);
                        else if (F == FMT_S5) sk_q8_sy<FMT_S5>(blk + 6, j_in_sb, q8);
                        else if (F == FMT_S6) sk_q8_sy<FMT_S6>(blk + 6, j_in_sb, q8);
                        else sk_q8_sy<FMT_S8>(blk + 6, j_in_sb, q8);
#pragma unroll
                        for (int j = 0; j < 8; ++j) sB[(hi + j) * 64 + nr] = (half)(((float)q8[j] - H) * eff);
                    }
                    sycl::group_barrier(it.get_group());
#if INK_SY_DEV_JM_SG16
                    jm::joint_matrix<sycl::sub_group, half, jm::use::a, 8, 16, jm::layout::row_major> a_m;
                    jm::joint_matrix<sycl::sub_group, half, jm::use::b, 16, 16, jm::layout::row_major> b0, b1;
                    auto pA = sA.template get_multi_ptr<sycl::access::decorated::no>() + sg_m * 8 * 16;
                    auto pB0 = sB.template get_multi_ptr<sycl::access::decorated::no>() + sg_n * 32;
                    jm::joint_matrix_load(sg, a_m, pA, 16);
                    jm::joint_matrix_load(sg, b0, pB0, 64);
                    jm::joint_matrix_load(sg, b1, pB0 + 16, 64);
                    jm::joint_matrix_mad(sg, c0, a_m, b0, c0);
                    jm::joint_matrix_mad(sg, c1, a_m, b1, c1);
#else
                    for (int k = 0; k < 16; ++k) {
                        float bv[4] = {(float)sB[k * 64 + c0], (float)sB[k * 64 + c0 + 1],
                                       (float)sB[k * 64 + c0 + 2], (float)sB[k * 64 + c0 + 3]};
#pragma unroll
                        for (int ri = 0; ri < 4; ++ri) {
                            const float av = (float)sA[(r0 + ri) * 16 + k];
                            acc[ri][0] += av * bv[0]; acc[ri][1] += av * bv[1];
                            acc[ri][2] += av * bv[2]; acc[ri][3] += av * bv[3];
                        }
                    }
#endif
                    sycl::group_barrier(it.get_group());
                }
#if INK_SY_DEV_JM_SG16
                auto pS = sC.template get_multi_ptr<sycl::access::decorated::no>() + sg_m * 8 * 64 + sg_n * 32;
                jm::joint_matrix_store(sg, c0, pS, 64, jm::layout::row_major);
                jm::joint_matrix_store(sg, c1, pS + 16, 64, jm::layout::row_major);
#else
#pragma unroll
                for (int ri = 0; ri < 4; ++ri) {
#pragma unroll
                    for (int ci = 0; ci < 4; ++ci) sC[(r0 + ri) * 64 + c0 + ci] = acc[ri][ci];
                }
#endif
                sycl::group_barrier(it.get_group());

                for (int idx = tid; idx < tm * 64; idx += 128) {
                    const int r = idx >> 6, c = idx & 63;
                    const int pid = pair_list[off + t0 + r];
                    const float rw = topk_w[pid];
                    y_pairs[(size_t)pid * 2304 + row0 + c] = sC[idx] * rw;
                }
                sycl::group_barrier(it.get_group());
            }
        });
    });
}
#endif  // INK_SY_HAS_JOINT_MATRIX

// ============================================================================
// 4. Vectorized 16B LSC + 1-Cycle Register-LUT SF4 Decode & Q8_ROW dp4a Decode
// ============================================================================

inline void k_linear_q8row_vec_dp4a_sg16(
    const uint8_t* W, int N, int K, const int8_t* xq, const float* xsc, float* Y, int M, int N_total = -1) {
    const int n_scales = (N_total > 0) ? N_total : N;
    const float* scales = reinterpret_cast<const float*>(W);
    const sycl::uint4* qw16 = reinterpret_cast<const sycl::uint4*>(W + (size_t)n_scales * 4);
    const sycl::uint4* xq16 = reinterpret_cast<const sycl::uint4*>(xq);
    const int K16 = K >> 4;  // 16 int8s per 16B uint4 load
    Q().parallel_for(NR((N + 15) / 16, 256), [=](sycl::nd_item<1> it) INK_REQD_SG16 {
        const int sg = (int)it.get_local_id(0) >> 4, lane = (int)it.get_local_id(0) & 15;
        const int row = (int)it.get_group(0) * 16 + sg;
        if (row >= N) return;
        const float wsc = scales[row];
        const sycl::uint4* wp = qw16 + (size_t)row * K16;
        for (int m0 = 0; m0 < M; m0 += 4) {
            const int mm = sycl::min(4, M - m0);
            int32_t acc[4] = {0, 0, 0, 0};
            for (int k = lane; k < K16; k += 16) {
                const sycl::uint4 wv = wp[k];
                for (int i = 0; i < mm; ++i) {
                    const sycl::uint4 xv = xq16[(size_t)(m0 + i) * K16 + k];
                    int32_t a = acc[i];
                    a = dp4a_s8((int32_t)wv.x(), (int32_t)xv.x(), a);
                    a = dp4a_s8((int32_t)wv.y(), (int32_t)xv.y(), a);
                    a = dp4a_s8((int32_t)wv.z(), (int32_t)xv.z(), a);
                    a = dp4a_s8((int32_t)wv.w(), (int32_t)xv.w(), a);
                    acc[i] = a;
                }
            }
            for (int i = 0; i < mm; ++i) {
                const int32_t s = sycl::reduce_over_group(it.get_sub_group(), acc[i], sycl::plus<int32_t>());
                if (lane == 0) Y[(size_t)(m0 + i) * N + row] = (float)s * (wsc * xsc[m0 + i]);
            }
        }
    });
}

// ============================================================================
// 5. Phase 2 & 4: FP8_E4M3 KV, Intra-Chunk SWA Fix (>1024 tokens), BF16 ShortConv,
//    and Top-128 Candidate Extraction (zero 200058*4 host logits copy!)
// ============================================================================

// Store normalized K/V into BOTH optional chunk buffer (for SWA M > 1024 safety!) AND FP16/FP8 KV cache!
// When skip_ring_write != 0 (used with k_kv_commit_swa_ring_sg16 after attention), only writes k_chunk_f16/v_chunk_f16
// so the existing SWA ring is 100% preserved while earlier tokens in the chunk read k < chunk_pos0!
inline void k_kv_norm_store_ex_sg16(
    const float* k_in, const float* v_in,
    half* k_cache_f16, half* v_cache_f16,
    uint8_t* k_cache_fp8, uint8_t* v_cache_fp8,
    half* k_scale_fp8, half* v_scale_fp8,
    half* k_chunk_f16, half* v_chunk_f16,
    const int* slot_ids, const int* pos_buf,
    int M, int is_swa, int swa_win, int kv_stride, int use_fp8, int skip_ring_write = 0) {
    Q().parallel_for(NR2(INK_NKV, (size_t)M, 128), [=](sycl::nd_item<2> it) INK_REQD_SG16 {
        float* sh = lmem<float, 16>(it);
        const int m = (int)it.get_group(0), kh = (int)it.get_group(1), tid = (int)it.get_local_id(1);
        const int slot = slot_ids[m], pos = pos_buf[m];
        const int kpos = is_swa ? (pos & (swa_win - 1)) : pos;

        const float kv = k_in[((size_t)m * INK_NKV + kh) * INK_HD + tid];
        const float kss = wg_sum16(it, kv * kv, sh);
        const float kfac = sycl::fmin(1.0f, 22.627416997969522f / (sycl::sqrt(kss) + 1e-6f));
        const float k_norm = kv * kfac;

        const float vv = v_in[((size_t)m * INK_NKV + kh) * INK_HD + tid];
        const float vss = wg_sum16(it, vv * vv, sh);
        const float vfac = sycl::rsqrt(vss / (float)INK_HD + INK_RMS_EPS);
        const float v_norm = vv * vfac;

        if (k_chunk_f16 && v_chunk_f16) {
            const size_t coff = ((size_t)m * INK_NKV + kh) * INK_HD + tid;
            k_chunk_f16[coff] = (half)k_norm;
            v_chunk_f16[coff] = (half)v_norm;
            if (skip_ring_write) return;
            if (is_swa && M > swa_win && m < M - swa_win) return;
        }

        const size_t dst = (((size_t)slot * kv_stride + kpos) * INK_NKV + kh) * INK_HD + tid;
        if (!use_fp8) {
            k_cache_f16[dst] = (half)k_norm;
            v_cache_f16[dst] = (half)v_norm;
        } else {
            // Per-(pos, head) dynamic scale for lossless-grade FP8_E4M3 KV (max 448.0f -> scale = amax / 448.0f)
            auto sg = it.get_sub_group();
            float kam = sycl::reduce_over_group(sg, sycl::fabs(k_norm), sycl::maximum<float>());
            float vam = sycl::reduce_over_group(sg, sycl::fabs(v_norm), sycl::maximum<float>());
            const int w = (int)sg.get_group_linear_id(), l = (int)sg.get_local_linear_id();
            sycl::group_barrier(it.get_group());
            if (l == 0) { sh[w] = kam; sh[8 + w] = vam; }
            sycl::group_barrier(it.get_group());
            float kgm = (l < 8) ? sh[l] : 0.0f;
            float vgm = (l < 8) ? sh[8 + l] : 0.0f;
            kgm = sycl::reduce_over_group(sg, kgm, sycl::maximum<float>());
            vgm = sycl::reduce_over_group(sg, vgm, sycl::maximum<float>());
            const float ksc = (kgm > 0.0f) ? (kgm * (1.0f / 448.0f)) : 1.0f;
            const float vsc = (vgm > 0.0f) ? (vgm * (1.0f / 448.0f)) : 1.0f;
            const size_t soff = ((size_t)slot * kv_stride + kpos) * INK_NKV + kh;
            if (tid == 0) {
                k_scale_fp8[soff] = (half)ksc;
                v_scale_fp8[soff] = (half)vsc;
            }
            k_cache_fp8[dst] = f32_to_e4m3(k_norm / ksc);
            v_cache_fp8[dst] = f32_to_e4m3(v_norm / vsc);
        }
    });
}

// Post-attention commit of the tail min(M, swa_win) tokens from k_chunk_f16/v_chunk_f16 into the SWA ring cache!
inline void k_kv_commit_swa_ring_sg16(
    const half* k_chunk_f16, const half* v_chunk_f16,
    half* k_cache_f16, half* v_cache_f16,
    uint8_t* k_cache_fp8, uint8_t* v_cache_fp8,
    half* k_scale_fp8, half* v_scale_fp8,
    const int* slot_ids, const int* pos_buf,
    int M, int swa_win, int kv_stride, int use_fp8) {
    const int m0 = (M > swa_win) ? (M - swa_win) : 0;
    const int n_commit = M - m0;
    if (n_commit <= 0) return;
    Q().parallel_for(NR2(INK_NKV, (size_t)n_commit, 128), [=](sycl::nd_item<2> it) INK_REQD_SG16 {
        float* sh = lmem<float, 16>(it);
        const int m = m0 + (int)it.get_group(0);
        const int kh = (int)it.get_group(1), tid = (int)it.get_local_id(1);
        const int slot = slot_ids[m], pos = pos_buf[m];
        const int kpos = pos & (swa_win - 1);
        const size_t coff = ((size_t)m * INK_NKV + kh) * INK_HD + tid;
        const size_t dst = (((size_t)slot * kv_stride + kpos) * INK_NKV + kh) * INK_HD + tid;
        if (!use_fp8) {
            k_cache_f16[dst] = k_chunk_f16[coff];
            v_cache_f16[dst] = v_chunk_f16[coff];
        } else {
            const float k_norm = (float)k_chunk_f16[coff];
            const float v_norm = (float)v_chunk_f16[coff];
            auto sg = it.get_sub_group();
            float kam = sycl::reduce_over_group(sg, sycl::fabs(k_norm), sycl::maximum<float>());
            float vam = sycl::reduce_over_group(sg, sycl::fabs(v_norm), sycl::maximum<float>());
            const int w = (int)sg.get_group_linear_id(), l = (int)sg.get_local_linear_id();
            sycl::group_barrier(it.get_group());
            if (l == 0) { sh[w] = kam; sh[8 + w] = vam; }
            sycl::group_barrier(it.get_group());
            float kgm = (l < 8) ? sh[l] : 0.0f;
            float vgm = (l < 8) ? sh[8 + l] : 0.0f;
            kgm = sycl::reduce_over_group(sg, kgm, sycl::maximum<float>());
            vgm = sycl::reduce_over_group(sg, vgm, sycl::maximum<float>());
            const float ksc = (kgm > 0.0f) ? (kgm * (1.0f / 448.0f)) : 1.0f;
            const float vsc = (vgm > 0.0f) ? (vgm * (1.0f / 448.0f)) : 1.0f;
            const size_t soff = ((size_t)slot * kv_stride + kpos) * INK_NKV + kh;
            if (tid == 0) {
                k_scale_fp8[soff] = (half)ksc;
                v_scale_fp8[soff] = (half)vsc;
            }
            k_cache_fp8[dst] = f32_to_e4m3(k_norm / ksc);
            v_cache_fp8[dst] = f32_to_e4m3(v_norm / vsc);
        }
    });
}

// Unified Flash RelPos + Sink GQA Attention supporting:
//  - FP16 KV and FP8_E4M3 KV (with per-head scales)
//  - Intra-chunk SWA safety when M > 1024 (reads k >= chunk_pos0 directly from k_chunk_f16 / v_chunk_f16!)
inline void k_flash_relpos_gqa_sg16(
    const float* q_buf, const float* r_buf, const float* rel_proj_pad, const float* sinks_buf,
    const float* gate_buf,
    const half* k_cache_f16, const half* v_cache_f16,
    const uint8_t* k_cache_fp8, const uint8_t* v_cache_fp8,
    const half* k_scale_fp8, const half* v_scale_fp8,
    const half* k_chunk_f16, const half* v_chunk_f16, int chunk_pos0,
    const int* slot_ids, const int* pos_buf, float* attn_out,
    int M, int is_swa, int swa_win, int rel_extent, int kv_stride, int use_fp8) {
    const float scale = 1.0f / std::sqrt((float)INK_HD);
    Q().submit([&](sycl::handler& h) {
        sycl::local_accessor<float, 1> sh_m(8, h);
        sycl::local_accessor<float, 1> sh_l(8, h);
        sycl::local_accessor<float, 1> sh_o(8 * INK_HD, h);
        sycl::local_accessor<float, 1> sh_r(16, h);
        h.parallel_for(NR2(INK_NH, (size_t)M, 128), [=](sycl::nd_item<2> it) INK_REQD_SG16 {
            const int m = (int)it.get_group(0), head = (int)it.get_group(1);
            const int tid = (int)it.get_local_id(1);
            const int sg = tid >> 4, lane = tid & 15;
            const int kh = head / 3;
            const int slot = slot_ids[m], qpos = pos_buf[m];

            if (tid < 16) sh_r[tid] = r_buf[((size_t)m * INK_NH + head) * INK_DREL + tid];
            sycl::group_barrier(it.get_group());

            float qv[8], o_acc[8];
            const float* qp = q_buf + ((size_t)m * INK_NH + head) * INK_HD + lane * 8;
#pragma unroll
            for (int u = 0; u < 8; ++u) { qv[u] = qp[u] * scale; o_acc[u] = 0.0f; }

            const int k_start = is_swa ? sycl::max(0, qpos - swa_win + 1) : 0;
            const int k_end = qpos + 1;
            float m_max = (sg == 0) ? sinks_buf[(size_t)m * INK_NH + head] : -1e30f;
            float l_sum = (sg == 0) ? 1.0f : 0.0f;
            const size_t slot_base = (size_t)slot * kv_stride;
            const bool has_chunk = (k_chunk_f16 != nullptr && v_chunk_f16 != nullptr && chunk_pos0 >= 0);

            for (int k = k_start + sg; k < k_end; k += 8) {
                float dt = 0.0f;
                float vv[8];
                if (has_chunk && k >= chunk_pos0 && (k - chunk_pos0) < M) {
                    const size_t coff = ((size_t)(k - chunk_pos0) * INK_NKV + kh) * INK_HD + lane * 8;
#pragma unroll
                    for (int u = 0; u < 8; ++u) {
                        dt += qv[u] * (float)k_chunk_f16[coff + u];
                        vv[u] = (float)v_chunk_f16[coff + u];
                    }
                } else {
                    const int kidx = is_swa ? (k & (swa_win - 1)) : k;
                    const size_t kv_off = ((slot_base + kidx) * INK_NKV + kh) * INK_HD + lane * 8;
                    if (!use_fp8) {
#pragma unroll
                        for (int u = 0; u < 8; ++u) {
                            dt += qv[u] * (float)k_cache_f16[kv_off + u];
                            vv[u] = (float)v_cache_f16[kv_off + u];
                        }
                    } else {
                        const size_t soff = (slot_base + kidx) * INK_NKV + kh;
                        const float ksc = (float)k_scale_fp8[soff];
                        const float vsc = (float)v_scale_fp8[soff];
#pragma unroll
                        for (int u = 0; u < 8; ++u) {
                            dt += qv[u] * (e4m3_to_f32(k_cache_fp8[kv_off + u]) * ksc);
                            vv[u] = e4m3_to_f32(v_cache_fp8[kv_off + u]) * vsc;
                        }
                    }
                }
                float sc = sg_sum(it, dt);
                const int dist = sycl::min(qpos - k, rel_extent);
                sc += sg_sum(it, sh_r[lane] * rel_proj_pad[(size_t)lane * 1025 + dist]);

                const float mn = sycl::fmax(m_max, sc);
                const float corr = sycl::exp(m_max - mn);
                const float p = sycl::exp(sc - mn);
                l_sum = l_sum * corr + p;
                m_max = mn;
#pragma unroll
                for (int u = 0; u < 8; ++u) o_acc[u] = o_acc[u] * corr + p * vv[u];
            }

            if (lane == 0) { sh_m[sg] = m_max; sh_l[sg] = l_sum; }
#pragma unroll
            for (int u = 0; u < 8; ++u) sh_o[sg * INK_HD + lane * 8 + u] = o_acc[u];
            sycl::group_barrier(it.get_group());

            if (sg == 0) {
                float gm = -1e30f;
                for (int s = 0; s < 8; ++s) gm = sycl::fmax(gm, sh_m[s]);
                float gl = 0.0f, go[8] = {};
                for (int s = 0; s < 8; ++s) {
                    const float w = sycl::exp(sh_m[s] - gm);
                    gl += sh_l[s] * w;
#pragma unroll
                    for (int u = 0; u < 8; ++u) go[u] += sh_o[s * INK_HD + lane * 8 + u] * w;
                }
                const float il = 1.0f / gl;
                const float* gp = gate_buf + ((size_t)m * INK_NH + head) * INK_HD + lane * 8;
                float* op = attn_out + ((size_t)m * INK_NH + head) * INK_HD + lane * 8;
#pragma unroll
                for (int u = 0; u < 8; ++u) op[u] = (go[u] * il) / (1.0f + sycl::exp(-gp[u]));
            }
        });
    });
}

// BF16 ShortConv history kernel (Phase 2 FAST mode: halves ShortConv state footprint!)
inline void k_sconv4_multi_bf16_sg16(
    const float* x_in, const float* static_w, const float* dyn_taps, uint16_t* hist_pool,
    const int* active_slots, const int* slot_tok_off, const int* slot_tok_cnt, const int* slot_tok_idx,
    float* y_out, int n_active_slots, int C, int has_dyn,
    uint16_t* ring_out = nullptr, int save_ring = 0, int ring_cap = 16) {
    Q().parallel_for(NR2((C + 255) / 256, (size_t)n_active_slots, 256), [=](sycl::nd_item<2> it) INK_REQD_SG16 {
        const int si = (int)it.get_group(0);
        const int c = (int)it.get_group(1) * 256 + (int)it.get_local_id(1);
        if (c >= C) return;
        const int slot = active_slots[si], off = slot_tok_off[si], cnt = slot_tok_cnt[si];
        uint16_t* h = hist_pool + (size_t)slot * 3 * C;
        float h0 = bf16_to_f32(h[0 * C + c]), h1 = bf16_to_f32(h[1 * C + c]), h2 = bf16_to_f32(h[2 * C + c]);
        const float* sw = static_w + (size_t)c * 4;
        const float sw0 = sw[0], sw1 = sw[1], sw2 = sw[2], sw3 = sw[3];
        const float scale = has_dyn ? (1.0f / 3.0f) : 0.4472135954999579f;
        const int hd_idx = c >> 7;
        for (int k = 0; k < cnt; ++k) {
            const int m = slot_tok_idx[off + k];
            const float cur = x_in[(size_t)m * C + c];
            float acc = h0 * sw0 + h1 * sw1 + h2 * sw2 + cur * sw3;
            if (has_dyn) {
                const float* dt = dyn_taps + ((size_t)m * 8 + hd_idx) * 4;
                acc += cur * dt[0] + h2 * dt[1] + h1 * dt[2] + h0 * dt[3];
            }
            y_out[(size_t)m * C + c] = (cur + acc) * scale;
            h0 = h1; h1 = h2; h2 = cur;
            if (save_ring && ring_out && si == 0 && k < ring_cap) {
                uint16_t* rdst = ring_out + (size_t)k * 3 * C;
                rdst[0 * C + c] = f32_to_bf16(h0);
                rdst[1 * C + c] = f32_to_bf16(h1);
                rdst[2 * C + c] = f32_to_bf16(h2);
            }
        }
        h[0 * C + c] = f32_to_bf16(h0);
        h[1 * C + c] = f32_to_bf16(h1);
        h[2 * C + c] = f32_to_bf16(h2);
    });
}

// Exact Top-128 candidate extraction per row on GPU (matches InkBackend::forward_batch_cand contract!)
// 1 WG of 256 threads per row: 128 rounds of group max+min-tiebreak -> exact sorted top-128 + finite flag!
inline void k_top128_rows_sg16(
    const float* logits, int V, int M,
    float* out_val, int32_t* out_idx, uint8_t* out_finite) {
    Q().submit([&](sycl::handler& h) {
        sycl::local_accessor<float, 1> sh_v(256, h);
        sycl::local_accessor<int32_t, 1> sh_i(256, h);
        sycl::local_accessor<int32_t, 1> sh_sel(128, h);
        h.parallel_for(NR(M, 256), [=](sycl::nd_item<1> it) INK_REQD_SG16 {
            const int m = (int)it.get_group(0), tid = (int)it.get_local_id(0);
            auto grp = it.get_group();
            const float* row = logits + (size_t)m * V;

            // Check finiteness every 67th logit (exact match to finite_logits in ink_serve.h!)
            int fin_ok = 1;
            for (int i = tid * 67; i < V; i += 256 * 67) {
                if (!sycl::isfinite(row[i])) { fin_ok = 0; break; }
            }
            fin_ok = sycl::reduce_over_group(grp, fin_ok, sycl::minimum<int>());
            if (tid == 0) out_finite[m] = (uint8_t)fin_ok;

            // Each thread keeps its own top-16 local heap? Even simpler & 100% exact:
            // Scan in 1 pass to find per-thread top-2? No: 16-entry per-thread top-16 takes only 128B/thread,
            // wait: with 256 threads, merging 256x16 or running 128-round parallel select over 256 per-thread bests:
            // Better: keep top-16 per thread in private regs (16 floats + 16 ints = 32 regs, 0 spill!).
            // Actually V=200058 / 256 = 781 elems/thread. A single pass per thread finding top-4 is tiny,
            // OR maintain a 1-pass sorted top-8 per thread (covers up to 2048 candidates, always contains global top-128
            // unless >8 of the top-128 fall in the same stride-256 lane; to be 100% exact with ZERO edge cases,
            // maintain a mask of already-picked indices for each of 128 rounds or per-thread cursor).
            // Wait: 128 rounds over 781 elements would be 100k loads; instead: find global threshold or keep 128 rounds
            // over top-16 per thread (probability >16 of top-128 share mod 256 is < 10^-15, AND if any thread exhausts
            // its 16, it rescans its 781 elements!). That's 1 pass + 100% mathematically exact!
            constexpr int TK = 16;
            float lv[TK];
            int32_t li[TK];
#pragma unroll
            for (int k = 0; k < TK; ++k) { lv[k] = -1e38f; li[k] = V; }
            for (int i = tid; i < V; i += 256) {
                const float v = row[i];
                if (v > lv[TK - 1] || (v == lv[TK - 1] && i < li[TK - 1])) {
                    int j = TK - 1;
                    while (j > 0 && (v > lv[j - 1] || (v == lv[j - 1] && i < li[j - 1]))) {
                        lv[j] = lv[j - 1]; li[j] = li[j - 1]; --j;
                    }
                    lv[j] = v; li[j] = i;
                }
            }
            int cur = 0;
            const int K = sycl::min(128, V);
            for (int k = 0; k < K; ++k) {
                float cv = (cur < TK) ? lv[cur] : -1e38f;
                int32_t ci = (cur < TK) ? li[cur] : V;
                const float mx = sycl::reduce_over_group(grp, cv, sycl::maximum<float>());
                const int32_t cand = (cv == mx) ? ci : V;
                const int32_t bi = sycl::reduce_over_group(grp, cand, sycl::minimum<int32_t>());
                if (tid == 0) {
                    out_val[(size_t)m * 128 + k] = mx;
                    out_idx[(size_t)m * 128 + k] = bi;
                    sh_sel[k] = bi;
                }
                if (ci == bi && cur < TK) {
                    ++cur;
                    if (cur == TK && k + 1 < K) {
                        // Ultra-rare exact fallback: refill next TK from this thread's stride!
                        float prev_v = lv[TK - 1];
                        int32_t prev_i = li[TK - 1];
#pragma unroll
                        for (int r = 0; r < TK; ++r) { lv[r] = -1e38f; li[r] = V; }
                        for (int i = tid; i < V; i += 256) {
                            const float v = row[i];
                            if (v > prev_v || (v == prev_v && i <= prev_i)) continue;
                            if (v > lv[TK - 1] || (v == lv[TK - 1] && i < li[TK - 1])) {
                                int j = TK - 1;
                                while (j > 0 && (v > lv[j - 1] || (v == lv[j - 1] && i < li[j - 1]))) {
                                    lv[j] = lv[j - 1]; li[j] = li[j - 1]; --j;
                                }
                                lv[j] = v; li[j] = i;
                            }
                        }
                        cur = 0;
                    }
                }
                sycl::group_barrier(grp);
            }
            (void)sh_v; (void)sh_i; (void)sh_sel;
        });
    });
}

}  // namespace ink_sycl
