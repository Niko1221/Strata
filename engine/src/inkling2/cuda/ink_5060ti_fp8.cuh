// engine/src/inkling2/cuda/ink_5060ti_fp8.cuh - Opt-In RTX 5060 Ti 16 GB (sm_120) In-Place FP8_E4M3 Trunk,
// cublasLt FP8 TC, FP16-Acc FlashAttention-2 Prefill, and High-BW Small-M MoE Engine
//
// MIT License
// Copyright (c) 2026 Niko1221 and the Strata contributors
//
// STRICTLY OFF BY DEFAULT (activated ONLY when STRATA_INK_RTX5060TI=1 && !STRATA_INK_RTX5060TI_EXACT=1 on sm_120).
// Stores quantized FP8_E4M3 attention & controller weights IN-PLACE inside InkCudaModel::d_trunk (0 MiB duplicate
// weight pool -> saves ~53 MiB vs Q8_ROW on 16 GB cards!) and loads cublasLt dynamically via dlopen/dlsym (zero
// CMakeLists.txt changes).
#pragma once

#include <cublasLt.h>
#include "ink_cublaslt_dyn.cuh"
#include <cuda_fp8.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <unordered_map>

#ifndef _WIN32
#include <dlfcn.h>
#else
#include <windows.h>
#endif

#include "ink_cuda.cuh"
#include "ink_attn.cuh"
#include "ink_kernels.cuh"
#include "ink_moe_cuda.cuh"
#include "ink_rtx5060ti_opt.cuh"

namespace strata::ink::opt5060 {

// ============================================================================
// 1. Hardware Conversion & MMA Intrinsics (sm_89 / sm_120)
// ============================================================================
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 800)
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
#endif

__device__ __forceinline__ __half2 fp8x2_to_h2(uint16_t packed) {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 890)
    uint32_t h2_u32;
    asm("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(h2_u32) : "h"(packed));
    return *reinterpret_cast<const __half2*>(&h2_u32);
#else
    const __nv_fp8x2_storage_t s = packed;
    const __half2_raw r = __nv_cvt_fp8x2_to_halfraw2(s, __NV_E4M3);
    return *reinterpret_cast<const __half2*>(&r);
#endif
}

__device__ __forceinline__ uint16_t f32x2_to_fp8x2(float a, float b) {
    return static_cast<uint16_t>(__nv_cvt_float2_to_fp8x2(make_float2(a, b), __NV_SATFINITE, __NV_E4M3));
}

__device__ __forceinline__ uint32_t pack_fp16x2_rn(float a, float b) {
    const __half2 v = __floats2half2_rn(a, b);
    return *reinterpret_cast<const uint32_t*>(&v);
}

__device__ __forceinline__ void atomicMaxNonNegFloat(float* addr, float val) {
    atomicMax(reinterpret_cast<uint32_t*>(addr), __float_as_uint(val));
}

// ============================================================================
// 2. Startup & Runtime FP8_E4M3 Quantizers (BF16 / F32 -> FP8_E4M3)
// ============================================================================
__global__ void __launch_bounds__(256) k_fp8_absmax_bf16(
    const __nv_bfloat16* __restrict__ src, size_t n, float* __restrict__ d_amax) {
    float mx = 0.0f;
    const size_t n2 = n >> 1;
    const uint32_t* s2 = reinterpret_cast<const uint32_t*>(src);
    for (size_t i = (size_t)blockIdx.x * 256 + threadIdx.x; i < n2; i += (size_t)gridDim.x * 256) {
        const float2 f = bf16x2_to_f32x2(s2[i]);
        mx = fmaxf(mx, fmaxf(fabsf(f.x), fabsf(f.y)));
    }
    mx = warp_max(mx);
    __shared__ float sm[8];
    if ((threadIdx.x & 31) == 0) sm[threadIdx.x >> 5] = mx;
    __syncthreads();
    if (threadIdx.x == 0) {
        float g = fmaxf(fmaxf(fmaxf(sm[0], sm[1]), fmaxf(sm[2], sm[3])),
                        fmaxf(fmaxf(sm[4], sm[5]), fmaxf(sm[6], sm[7])));
        atomicMaxNonNegFloat(d_amax, g);
    }
}

__global__ void __launch_bounds__(256) k_fp8_absmax_f32(
    const float* __restrict__ src, size_t n, float* __restrict__ d_amax) {
    float mx = 0.0f;
    const size_t n4 = n >> 2;
    const float4* s4 = reinterpret_cast<const float4*>(src);
    for (size_t i = (size_t)blockIdx.x * 256 + threadIdx.x; i < n4; i += (size_t)gridDim.x * 256) {
        const float4 v = s4[i];
        mx = fmaxf(mx, fmaxf(fmaxf(fabsf(v.x), fabsf(v.y)), fmaxf(fabsf(v.z), fabsf(v.w))));
    }
    mx = warp_max(mx);
    __shared__ float sm[8];
    if ((threadIdx.x & 31) == 0) sm[threadIdx.x >> 5] = mx;
    __syncthreads();
    if (threadIdx.x == 0) {
        float g = fmaxf(fmaxf(fmaxf(sm[0], sm[1]), fmaxf(sm[2], sm[3])),
                        fmaxf(fmaxf(sm[4], sm[5]), fmaxf(sm[6], sm[7])));
        atomicMaxNonNegFloat(d_amax, g);
    }
}

__global__ void __launch_bounds__(1) k_fp8_finalize_scale(float* __restrict__ d_amax, float* __restrict__ d_scale) {
    *d_scale = fmaxf((*d_amax) * (1.0f / 448.0f), 1e-12f);
}

__global__ void __launch_bounds__(256) k_fp8_quant_bf16(
    const __nv_bfloat16* __restrict__ src, size_t n, const float* __restrict__ d_scale, uint8_t* __restrict__ dst) {
    const float inv = 1.0f / (*d_scale);
    const size_t n2 = n >> 1;
    const uint32_t* s2 = reinterpret_cast<const uint32_t*>(src);
    uint16_t* d2 = reinterpret_cast<uint16_t*>(dst);
    for (size_t i = (size_t)blockIdx.x * 256 + threadIdx.x; i < n2; i += (size_t)gridDim.x * 256) {
        const float2 f = bf16x2_to_f32x2(s2[i]);
        d2[i] = f32x2_to_fp8x2(f.x * inv, f.y * inv);
    }
}

__global__ void __launch_bounds__(256) k_fp8_quant_f32_padded(
    const float* __restrict__ src, size_t n_valid, size_t n_pad, const float* __restrict__ d_amax,
    float* __restrict__ d_scale, uint8_t* __restrict__ dst) {
    const float sc = fmaxf((*d_amax) * (1.0f / 448.0f), 1e-12f);
    if (blockIdx.x == 0 && threadIdx.x == 0) *d_scale = sc;
    const float inv = 1.0f / sc;
    const size_t np4 = n_pad >> 2;
    const size_t nv4 = n_valid >> 2;
    const float4* s4 = reinterpret_cast<const float4*>(src);
    uint32_t* d4 = reinterpret_cast<uint32_t*>(dst);
    for (size_t i = (size_t)blockIdx.x * 256 + threadIdx.x; i < np4; i += (size_t)gridDim.x * 256) {
        if (i < nv4) {
            const float4 v = s4[i];
            const uint16_t lo = f32x2_to_fp8x2(v.x * inv, v.y * inv);
            const uint16_t hi = f32x2_to_fp8x2(v.z * inv, v.w * inv);
            d4[i] = (uint32_t)lo | ((uint32_t)hi << 16);
        } else {
            d4[i] = 0u;
        }
    }
}

__global__ void __launch_bounds__(256) k_fp8_quant_bf16_padded(
    const __nv_bfloat16* __restrict__ src, size_t n_valid, size_t n_pad, const float* __restrict__ d_amax,
    float* __restrict__ d_scale, uint8_t* __restrict__ dst) {
    const float sc = fmaxf((*d_amax) * (1.0f / 448.0f), 1e-12f);
    if (blockIdx.x == 0 && threadIdx.x == 0) *d_scale = sc;
    const float inv = 1.0f / sc;
    const size_t np2 = n_pad >> 1;
    const size_t nv2 = n_valid >> 1;
    const uint32_t* s2 = reinterpret_cast<const uint32_t*>(src);
    uint16_t* d2 = reinterpret_cast<uint16_t*>(dst);
    for (size_t i = (size_t)blockIdx.x * 256 + threadIdx.x; i < np2; i += (size_t)gridDim.x * 256) {
        if (i < nv2) {
            const float2 f = bf16x2_to_f32x2(s2[i]);
            d2[i] = f32x2_to_fp8x2(f.x * inv, f.y * inv);
        } else {
            d2[i] = 0u;
        }
    }
}

__global__ void __launch_bounds__(1024) k_fp8_quant_f32_1cta_1024(
    const float* __restrict__ src, size_t n_valid, size_t n_pad,
    float* __restrict__ d_scale, uint8_t* __restrict__ dst) {
    const int tid = threadIdx.x;
    const size_t nv4 = n_valid >> 2, np4 = n_pad >> 2;
    const float4* s4 = reinterpret_cast<const float4*>(src);
    uint32_t* d4 = reinterpret_cast<uint32_t*>(dst);
    float mx = 0.0f;
    for (size_t i = tid; i < nv4; i += 1024) {
        const float4 v = s4[i];
        mx = fmaxf(mx, fmaxf(fmaxf(fabsf(v.x), fabsf(v.y)), fmaxf(fabsf(v.z), fabsf(v.w))));
    }
    mx = warp_max(mx);
    __shared__ float sm[32];
    if ((tid & 31) == 0) sm[tid >> 5] = mx;
    __syncthreads();
    if (tid < 32) {
        float g = warp_max(sm[tid]);
        if (tid == 0) {
            const float sc = fmaxf(g * (1.0f / 448.0f), 1e-12f);
            sm[0] = sc;
            *d_scale = sc;
        }
    }
    __syncthreads();
    const float inv = 1.0f / sm[0];
    for (size_t i = tid; i < np4; i += 1024) {
        if (i < nv4) {
            const float4 v = s4[i];
            const uint16_t lo = f32x2_to_fp8x2(v.x * inv, v.y * inv);
            const uint16_t hi = f32x2_to_fp8x2(v.z * inv, v.w * inv);
            d4[i] = (uint32_t)lo | ((uint32_t)hi << 16);
        } else {
            d4[i] = 0u;
        }
    }
}

__global__ void __launch_bounds__(256) k_fp8_quant_f32_1cta(
    const float* __restrict__ src, size_t n_valid, size_t n_pad,
    float* __restrict__ d_scale, uint8_t* __restrict__ dst) {
    const int tid = threadIdx.x;
    const size_t nv4 = n_valid >> 2, np4 = n_pad >> 2;
    const float4* s4 = reinterpret_cast<const float4*>(src);
    uint32_t* d4 = reinterpret_cast<uint32_t*>(dst);
    float mx = 0.0f;
    for (size_t i = tid; i < nv4; i += 256) {
        const float4 v = s4[i];
        mx = fmaxf(mx, fmaxf(fmaxf(fabsf(v.x), fabsf(v.y)), fmaxf(fabsf(v.z), fabsf(v.w))));
    }
    mx = warp_max(mx);
    __shared__ float sm[8];
    if ((tid & 31) == 0) sm[tid >> 5] = mx;
    __syncthreads();
    const float g = fmaxf(fmaxf(fmaxf(sm[0], sm[1]), fmaxf(sm[2], sm[3])),
                          fmaxf(fmaxf(sm[4], sm[5]), fmaxf(sm[6], sm[7])));
    const float sc = fmaxf(g * (1.0f / 448.0f), 1e-12f);
    if (tid == 0) *d_scale = sc;
    const float inv = 1.0f / sc;
    for (size_t i = tid; i < np4; i += 256) {
        if (i < nv4) {
            const float4 v = s4[i];
            const uint16_t lo = f32x2_to_fp8x2(v.x * inv, v.y * inv);
            const uint16_t hi = f32x2_to_fp8x2(v.z * inv, v.w * inv);
            d4[i] = (uint32_t)lo | ((uint32_t)hi << 16);
        } else {
            d4[i] = 0u;
        }
    }
}

__global__ void __launch_bounds__(256) k_fp8_quant_bf16_1cta(
    const __nv_bfloat16* __restrict__ src, size_t n_valid, size_t n_pad,
    float* __restrict__ d_scale, uint8_t* __restrict__ dst) {
    const int tid = threadIdx.x;
    const size_t nv2 = n_valid >> 1, np2 = n_pad >> 1;
    const uint32_t* s2 = reinterpret_cast<const uint32_t*>(src);
    uint16_t* d2 = reinterpret_cast<uint16_t*>(dst);
    float mx = 0.0f;
    for (size_t i = tid; i < nv2; i += 256) {
        const float2 f = bf16x2_to_f32x2(s2[i]);
        mx = fmaxf(mx, fmaxf(fabsf(f.x), fabsf(f.y)));
    }
    mx = warp_max(mx);
    __shared__ float sm[8];
    if ((tid & 31) == 0) sm[tid >> 5] = mx;
    __syncthreads();
    const float g = fmaxf(fmaxf(fmaxf(sm[0], sm[1]), fmaxf(sm[2], sm[3])),
                          fmaxf(fmaxf(sm[4], sm[5]), fmaxf(sm[6], sm[7])));
    const float sc = fmaxf(g * (1.0f / 448.0f), 1e-12f);
    if (tid == 0) *d_scale = sc;
    const float inv = 1.0f / sc;
    for (size_t i = tid; i < np2; i += 256) {
        if (i < nv2) {
            const float2 f = bf16x2_to_f32x2(s2[i]);
            d2[i] = f32x2_to_fp8x2(f.x * inv, f.y * inv);
        } else {
            d2[i] = 0u;
        }
    }
}

__global__ void __launch_bounds__(256) k_bf16_to_f32_fast(
    const __nv_bfloat16* __restrict__ src, float* __restrict__ dst, size_t n) {
    const size_t n2 = n >> 1;
    const uint32_t* s2 = reinterpret_cast<const uint32_t*>(src);
    float2* d2 = reinterpret_cast<float2*>(dst);
    for (size_t i = (size_t)blockIdx.x * 256 + threadIdx.x; i < n2; i += (size_t)gridDim.x * 256) {
        d2[i] = bf16x2_to_f32x2(s2[i]);
    }
}

__global__ void __launch_bounds__(256) k_f32_to_h2(
    const float* __restrict__ src, __half2* __restrict__ dst, size_t n_pairs) {
    const size_t i = (size_t)blockIdx.x * 256 + threadIdx.x;
    if (i < n_pairs) {
        const float2 f = reinterpret_cast<const float2*>(src)[i];
        dst[i] = __floats2half2_rn(fminf(fmaxf(f.x, -65504.0f), 65504.0f),
                                   fminf(fmaxf(f.y, -65504.0f), 65504.0f));
    }
}

// ============================================================================
// 3. Fused 1-CTA Depth/Bank Mix & Zero-F32-Intermediate FP8 Loop Controllers
// ============================================================================
__global__ void __launch_bounds__(256) k_ctrl_cat_rms_to_fp8(
    const float* __restrict__ cur, const float* __restrict__ msg,
    uint8_t* __restrict__ dst_fp8, int M, int M_pad, int D) {
    const int m = blockIdx.x;
    if (m >= M_pad) return;
    const int tid = threadIdx.x, D4 = D >> 2;
    uint32_t* d0 = reinterpret_cast<uint32_t*>(dst_fp8 + (size_t)m * (2 * D));
    uint32_t* d1 = reinterpret_cast<uint32_t*>(dst_fp8 + (size_t)m * (2 * D) + D);
    if (m >= M) {
        for (int i = tid; i < D4; i += 256) { d0[i] = 0u; d1[i] = 0u; }
        return;
    }
    __shared__ float sh[32];
    const float4* cp4 = reinterpret_cast<const float4*>(cur + (size_t)m * D);
    const float4* mp4 = reinterpret_cast<const float4*>(msg + (size_t)m * D);
    float ssc = 0.0f, ssm = 0.0f;
    for (int i = tid; i < D4; i += 256) {
        const float4 cv = cp4[i], gv = mp4[i];
        ssc += cv.x * cv.x + cv.y * cv.y + cv.z * cv.z + cv.w * cv.w;
        ssm += gv.x * gv.x + gv.y * gv.y + gv.z * gv.z + gv.w * gv.w;
    }
    const float ic = rsqrtf(block_sum(ssc, sh) / (float)D + kInkRmsEps) * 8.0f;
    const float im = rsqrtf(block_sum(ssm, sh) / (float)D + kInkRmsEps) * 8.0f;
    for (int i = tid; i < D4; i += 256) {
        const float4 cv = cp4[i], gv = mp4[i];
        d0[i] = (uint32_t)f32x2_to_fp8x2(cv.x * ic, cv.y * ic) | ((uint32_t)f32x2_to_fp8x2(cv.z * ic, cv.w * ic) << 16);
        d1[i] = (uint32_t)f32x2_to_fp8x2(gv.x * im, gv.y * im) | ((uint32_t)f32x2_to_fp8x2(gv.z * im, gv.w * im) << 16);
    }
}

__global__ void __launch_bounds__(256) k_attn_res_3x_fused_1cta(
    DepthStatePtrs stp, const float* __restrict__ q_res, bool is_swa,
    DepthStatePtrs bst, const float* __restrict__ q_msg,
    float* __restrict__ mixed_out, float* __restrict__ msgs_out,
    float* __restrict__ cat3_out, int M, int D,
    uint8_t* __restrict__ cat3_fp8_out = nullptr, int M_pad = 0) {
    const int m = blockIdx.x, tid = threadIdx.x, D4 = D >> 2;
    if (m >= M) {
        if (cat3_fp8_out && m < M_pad) {
            for (int c = 0; c < 3; ++c) {
                uint32_t* dp = reinterpret_cast<uint32_t*>(cat3_fp8_out + ((size_t)c * M_pad + m) * (2 * D));
                for (int i = tid; i < 2 * D4; i += 256) dp[i] = 0u;
            }
        }
        return;
    }
    const int warp = tid >> 5, lane = tid & 31;
    const float inv_sqrt_d = rsqrtf((float)D);
    __shared__ float s_wred[8][5][8];
    __shared__ float s_prob[4][8];
    __shared__ float s_sinv[8];
    __shared__ float sh[32];
    __shared__ alignas(16) uint32_t s_st[6][kInkHidden / 2];

    const int B = bst.count;
    const bool use_smem_b = (cat3_fp8_out != nullptr && B <= 6 && D == kInkHidden);
    const float4* qm0 = reinterpret_cast<const float4*>(q_msg + 0 * (size_t)D);
    const float4* qm1 = reinterpret_cast<const float4*>(q_msg + 1 * (size_t)D);
    const float4* qm2 = reinterpret_cast<const float4*>(q_msg + 2 * (size_t)D);
    const float4* qm3 = reinterpret_cast<const float4*>(q_msg + 3 * (size_t)D);
    for (int s = 0; s < B; ++s) {
        float ss = 0.0f, d0 = 0.0f, d1 = 0.0f, d2 = 0.0f, d3 = 0.0f;
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
        for (int s = 0; s < B; ++s) { const float ex = expf(s_prob[c][s] - mx); s_prob[c][s] = ex; sum += ex; }
        const float inv_s = 1.0f / sum;
        for (int s = 0; s < B; ++s) s_prob[c][s] = s_prob[c][s] * inv_s * s_sinv[s];
    }
    __syncthreads();

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
            reinterpret_cast<float4*>(msgs_out + ((size_t)0 * M + m) * D)[i] = mv0;
            reinterpret_cast<float4*>(msgs_out + ((size_t)1 * M + m) * D)[i] = mv1;
            reinterpret_cast<float4*>(msgs_out + ((size_t)2 * M + m) * D)[i] = mv2;
            reinterpret_cast<float4*>(msgs_out + ((size_t)3 * M + m) * D)[i] = make_float4(a3x, a3y, a3z, a3w);
            if (cat3_out || cat3_fp8_out) {
                r_m0[t] = mv0; r_m1[t] = mv1; r_m2[t] = mv2;
                ssm0 += a0x*a0x + a0y*a0y + a0z*a0z + a0w*a0w;
                ssm1 += a1x*a1x + a1y*a1y + a1z*a1z + a1w*a1w;
                ssm2 += a2x*a2x + a2y*a2y + a2z*a2z + a2w*a2w;
            }
        }
    }
    if (use_smem_b) __syncthreads();

    const int S = stp.count;
    const bool use_smem_s = (cat3_fp8_out != nullptr && S <= 6 && D == kInkHidden);
    const int nq = is_swa ? 1 : 3;
    const float4* qr0 = reinterpret_cast<const float4*>(q_res + 0 * (size_t)D);
    const float4* qr1 = is_swa ? qr0 : reinterpret_cast<const float4*>(q_res + 1 * (size_t)D);
    const float4* qr2 = is_swa ? qr0 : reinterpret_cast<const float4*>(q_res + 2 * (size_t)D);
    for (int s = 0; s < S; ++s) {
        float ss = 0.0f, d0 = 0.0f, d1 = 0.0f, d2 = 0.0f;
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
        for (int s = 0; s < S; ++s) { const float ex = expf(s_prob[c][s] - mx); s_prob[c][s] = ex; sum += ex; }
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
            const float4 sv0 = make_float4(a0x, a0y, a0z, a0w);
            const float4 sv1 = is_swa ? sv0 : make_float4(a1x, a1y, a1z, a1w);
            const float4 sv2 = is_swa ? sv0 : make_float4(a2x, a2y, a2z, a2w);
            reinterpret_cast<float4*>(mixed_out + ((size_t)0 * M + m) * D)[i] = sv0;
            if (!is_swa) {
                reinterpret_cast<float4*>(mixed_out + ((size_t)1 * M + m) * D)[i] = sv1;
                reinterpret_cast<float4*>(mixed_out + ((size_t)2 * M + m) * D)[i] = sv2;
            }
            if (cat3_out || cat3_fp8_out) {
                r_s0[t] = sv0; r_s1[t] = sv1; r_s2[t] = sv2;
                ssc0 += sv0.x*sv0.x + sv0.y*sv0.y + sv0.z*sv0.z + sv0.w*sv0.w;
                if (!is_swa) {
                    ssc1 += sv1.x*sv1.x + sv1.y*sv1.y + sv1.z*sv1.z + sv1.w*sv1.w;
                    ssc2 += sv2.x*sv2.x + sv2.y*sv2.y + sv2.z*sv2.z + sv2.w*sv2.w;
                }
            }
        }
    }
    if (!cat3_out && !cat3_fp8_out) return;

    const float ic0 = rsqrtf(block_sum(ssc0, sh) / (float)D + kInkRmsEps);
    const float ic1 = is_swa ? ic0 : rsqrtf(block_sum(ssc1, sh) / (float)D + kInkRmsEps);
    const float ic2 = is_swa ? ic0 : rsqrtf(block_sum(ssc2, sh) / (float)D + kInkRmsEps);
    const float im0 = rsqrtf(block_sum(ssm0, sh) / (float)D + kInkRmsEps);
    const float im1 = rsqrtf(block_sum(ssm1, sh) / (float)D + kInkRmsEps);
    const float im2 = rsqrtf(block_sum(ssm2, sh) / (float)D + kInkRmsEps);

    if (cat3_fp8_out) {
        const float sc0 = ic0 * 8.0f, sc1 = ic1 * 8.0f, sc2 = ic2 * 8.0f;
        const float sm0 = im0 * 8.0f, sm1 = im1 * 8.0f, sm2 = im2 * 8.0f;
        uint32_t* f0a = reinterpret_cast<uint32_t*>(cat3_fp8_out + ((size_t)0 * M_pad + m) * (2 * D));
        uint32_t* f0b = reinterpret_cast<uint32_t*>(cat3_fp8_out + ((size_t)0 * M_pad + m) * (2 * D) + D);
        uint32_t* f1a = reinterpret_cast<uint32_t*>(cat3_fp8_out + ((size_t)1 * M_pad + m) * (2 * D));
        uint32_t* f1b = reinterpret_cast<uint32_t*>(cat3_fp8_out + ((size_t)1 * M_pad + m) * (2 * D) + D);
        uint32_t* f2a = reinterpret_cast<uint32_t*>(cat3_fp8_out + ((size_t)2 * M_pad + m) * (2 * D));
        uint32_t* f2b = reinterpret_cast<uint32_t*>(cat3_fp8_out + ((size_t)2 * M_pad + m) * (2 * D) + D);
#pragma unroll
        for (int t = 0; t < 3; ++t) {
            const int i = tid + t * 256;
            if (i < D4) {
                const float4 c0 = r_s0[t], c1 = r_s1[t], c2 = r_s2[t];
                const float4 g0 = r_m0[t], g1 = r_m1[t], g2 = r_m2[t];
                f0a[i] = (uint32_t)f32x2_to_fp8x2(c0.x*sc0, c0.y*sc0) | ((uint32_t)f32x2_to_fp8x2(c0.z*sc0, c0.w*sc0) << 16);
                f0b[i] = (uint32_t)f32x2_to_fp8x2(g0.x*sm0, g0.y*sm0) | ((uint32_t)f32x2_to_fp8x2(g0.z*sm0, g0.w*sm0) << 16);
                f1a[i] = (uint32_t)f32x2_to_fp8x2(c1.x*sc1, c1.y*sc1) | ((uint32_t)f32x2_to_fp8x2(c1.z*sc1, c1.w*sc1) << 16);
                f1b[i] = (uint32_t)f32x2_to_fp8x2(g1.x*sm1, g1.y*sm1) | ((uint32_t)f32x2_to_fp8x2(g1.z*sm1, g1.w*sm1) << 16);
                f2a[i] = (uint32_t)f32x2_to_fp8x2(c2.x*sc2, c2.y*sc2) | ((uint32_t)f32x2_to_fp8x2(c2.z*sc2, c2.w*sc2) << 16);
                f2b[i] = (uint32_t)f32x2_to_fp8x2(g2.x*sm2, g2.y*sm2) | ((uint32_t)f32x2_to_fp8x2(g2.z*sm2, g2.w*sm2) << 16);
            }
        }
    } else {
        float4* c0a = reinterpret_cast<float4*>(cat3_out + ((size_t)0 * M + m) * (2 * D));
        float4* c0b = reinterpret_cast<float4*>(cat3_out + ((size_t)0 * M + m) * (2 * D) + D);
        float4* c1a = reinterpret_cast<float4*>(cat3_out + ((size_t)1 * M + m) * (2 * D));
        float4* c1b = reinterpret_cast<float4*>(cat3_out + ((size_t)1 * M + m) * (2 * D) + D);
        float4* c2a = reinterpret_cast<float4*>(cat3_out + ((size_t)2 * M + m) * (2 * D));
        float4* c2b = reinterpret_cast<float4*>(cat3_out + ((size_t)2 * M + m) * (2 * D) + D);
#pragma unroll
        for (int t = 0; t < 3; ++t) {
            const int i = tid + t * 256;
            if (i < D4) {
                const float4 c0 = r_s0[t], c1 = r_s1[t], c2 = r_s2[t];
                const float4 g0 = r_m0[t], g1 = r_m1[t], g2 = r_m2[t];
                c0a[i] = make_float4(c0.x*ic0, c0.y*ic0, c0.z*ic0, c0.w*ic0);
                c0b[i] = make_float4(g0.x*im0, g0.y*im0, g0.z*im0, g0.w*im0);
                c1a[i] = make_float4(c1.x*ic1, c1.y*ic1, c1.z*ic1, c1.w*ic1);
                c1b[i] = make_float4(g1.x*im1, g1.y*im1, g1.z*im1, g1.w*im1);
                c2a[i] = make_float4(c2.x*ic2, c2.y*ic2, c2.z*ic2, c2.w*ic2);
                c2b[i] = make_float4(g2.x*im2, g2.y*im2, g2.z*im2, g2.w*im2);
            }
        }
    }
}

__global__ void __launch_bounds__(256) k_mlp_res_to_fp8(
    DepthStatePtrs stp, const float* __restrict__ q_mlp, const float* __restrict__ msg,
    float* __restrict__ moe_out, uint8_t* __restrict__ dst_fp8, int M, int M_pad, int D) {
    const int m = blockIdx.x, tid = threadIdx.x, D4 = D >> 2;
    if (m >= M_pad) return;
    uint32_t* d0 = reinterpret_cast<uint32_t*>(dst_fp8 + (size_t)m * (2 * D));
    uint32_t* d1 = reinterpret_cast<uint32_t*>(dst_fp8 + (size_t)m * (2 * D) + D);
    if (m >= M) {
        for (int i = tid; i < D4; i += 256) { d0[i] = 0u; d1[i] = 0u; }
        return;
    }
    const int warp = tid >> 5, lane = tid & 31, S = stp.count;
    const float inv_sqrt_d = rsqrtf((float)D);
    __shared__ float s_wred[8][2][8], s_sinv[8], s_prob[8], sh[32];
    __shared__ alignas(16) uint32_t s_st[6][kInkHidden / 2];
    const bool use_smem = (S <= 6 && D == kInkHidden);
    const float4* q4 = reinterpret_cast<const float4*>(q_mlp);
    for (int s = 0; s < S; ++s) {
        float ss = 0.0f, dt = 0.0f;
        const float4* sp4 = reinterpret_cast<const float4*>(stp.p[s] + (size_t)m * D);
        for (int i = tid; i < D4; i += 256) {
            const float4 sv = sp4[i], qv = q4[i];
            if (use_smem) {
                s_st[s][2 * i + 0] = f32x2_to_bf16x2(sv.x, sv.y);
                s_st[s][2 * i + 1] = f32x2_to_bf16x2(sv.z, sv.w);
            }
            ss += sv.x * sv.x + sv.y * sv.y + sv.z * sv.z + sv.w * sv.w;
            dt = fmaf(sv.x, qv.x, dt); dt = fmaf(sv.y, qv.y, dt);
            dt = fmaf(sv.z, qv.z, dt); dt = fmaf(sv.w, qv.w, dt);
        }
        ss = warp_sum(ss); dt = warp_sum(dt);
        if (lane == 0) { s_wred[s][0][warp] = ss; s_wred[s][1][warp] = dt; }
    }
    __syncthreads();
    if (tid == 0) {
        float mx = -1e30f;
        for (int s = 0; s < S; ++s) {
            float ss = 0.0f, dt = 0.0f;
#pragma unroll
            for (int w = 0; w < 8; ++w) { ss += s_wred[s][0][w]; dt += s_wred[s][1][w]; }
            const float inv = rsqrtf(ss / (float)D + kInkRmsEps);
            s_sinv[s] = inv;
            const float sc = dt * inv * inv_sqrt_d;
            s_prob[s] = sc;
            mx = fmaxf(mx, sc);
        }
        float sum = 0.0f;
        for (int s = 0; s < S; ++s) { const float e = expf(s_prob[s] - mx); s_prob[s] = e; sum += e; }
        const float inv_s = 1.0f / sum;
        for (int s = 0; s < S; ++s) s_prob[s] = s_prob[s] * inv_s * s_sinv[s];
    }
    __syncthreads();

    float4 mo_reg[3], mp_reg[3];
    float ssc = 0.0f, ssm = 0.0f;
#pragma unroll
    for (int t = 0; t < 3; ++t) {
        const int i = tid + t * 256;
        if (i < D4) {
            float ax = 0, ay = 0, az = 0, aw = 0;
            for (int s = 0; s < S; ++s) {
                float4 sv;
                if (use_smem) {
                    const float2 f0 = bf16x2_to_f32x2(s_st[s][2 * i + 0]);
                    const float2 f1 = bf16x2_to_f32x2(s_st[s][2 * i + 1]);
                    sv = make_float4(f0.x, f0.y, f1.x, f1.y);
                } else {
                    sv = reinterpret_cast<const float4*>(stp.p[s] + (size_t)m * D)[i];
                }
                const float w = s_prob[s];
                ax = fmaf(w, sv.x, ax); ay = fmaf(w, sv.y, ay); az = fmaf(w, sv.z, az); aw = fmaf(w, sv.w, aw);
            }
            const float4 cv = make_float4(ax, ay, az, aw);
            const float4 mv = reinterpret_cast<const float4*>(msg + (size_t)m * D)[i];
            reinterpret_cast<float4*>(moe_out + (size_t)m * D)[i] = cv;
            mo_reg[t] = cv; mp_reg[t] = mv;
            ssc += ax * ax + ay * ay + az * az + aw * aw;
            ssm += mv.x * mv.x + mv.y * mv.y + mv.z * mv.z + mv.w * mv.w;
        }
    }
    const float ic = rsqrtf(block_sum(ssc, sh) / (float)D + kInkRmsEps) * 8.0f;
    const float im = rsqrtf(block_sum(ssm, sh) / (float)D + kInkRmsEps) * 8.0f;
#pragma unroll
    for (int t = 0; t < 3; ++t) {
        const int i = tid + t * 256;
        if (i < D4) {
            const float4 cv = mo_reg[t], gv = mp_reg[t];
            d0[i] = (uint32_t)f32x2_to_fp8x2(cv.x * ic, cv.y * ic) | ((uint32_t)f32x2_to_fp8x2(cv.z * ic, cv.w * ic) << 16);
            d1[i] = (uint32_t)f32x2_to_fp8x2(gv.x * im, gv.y * im) | ((uint32_t)f32x2_to_fp8x2(gv.z * im, gv.w * im) << 16);
        }
    }
}

__global__ void __launch_bounds__(256) k_ctrl_blend_norm_fused_gf(
    const float* __restrict__ cur, const float* __restrict__ msg,
    const float* __restrict__ gf, const float* __restrict__ norm_w,
    float* __restrict__ out, int M, int D, float* __restrict__ d_amax_out = nullptr) {
    const int m = blockIdx.x;
    if (m >= M) return;
    const int tid = threadIdx.x, D4 = D >> 2;
    __shared__ float sh[32];
    const float4* cp4 = reinterpret_cast<const float4*>(cur + (size_t)m * D);
    const float4* mp4 = reinterpret_cast<const float4*>(msg + (size_t)m * D);
    const float4* gp4 = reinterpret_cast<const float4*>(gf  + (size_t)m * (2 * D));
    const float4* fp4 = reinterpret_cast<const float4*>(gf  + (size_t)m * (2 * D) + D);
    float4* op4 = reinterpret_cast<float4*>(out + (size_t)m * D);
    float4 r_out[3];
    float ss = 0.0f;
#pragma unroll
    for (int t = 0; t < 3; ++t) {
        const int i = tid + t * 256;
        if (i < D4) {
            const float4 cv = cp4[i], mv = mp4[i], gv = gp4[i], fv = fp4[i];
            const float o0 = (1.0f - tanhf(fv.x)) * cv.x + tanhf(gv.x) * mv.x;
            const float o1 = (1.0f - tanhf(fv.y)) * cv.y + tanhf(gv.y) * mv.y;
            const float o2 = (1.0f - tanhf(fv.z)) * cv.z + tanhf(gv.z) * mv.z;
            const float o3 = (1.0f - tanhf(fv.w)) * cv.w + tanhf(gv.w) * mv.w;
            r_out[t] = make_float4(o0, o1, o2, o3);
            ss += o0 * o0 + o1 * o1 + o2 * o2 + o3 * o3;
        }
    }
    float mx = 0.0f;
    const float inv = norm_w ? rsqrtf(block_sum(ss, sh) / (float)D + kInkRmsEps) : 1.0f;
    const float4* w4 = norm_w ? reinterpret_cast<const float4*>(norm_w) : nullptr;
#pragma unroll
    for (int t = 0; t < 3; ++t) {
        const int i = tid + t * 256;
        if (i < D4) {
            float4 v = r_out[t];
            if (w4) {
                const float4 wv = w4[i];
                v = make_float4(v.x * inv * wv.x, v.y * inv * wv.y, v.z * inv * wv.z, v.w * inv * wv.w);
            }
            op4[i] = v;
            if (d_amax_out) mx = fmaxf(mx, fmaxf(fmaxf(fabsf(v.x), fabsf(v.y)), fmaxf(fabsf(v.z), fabsf(v.w))));
        }
    }
    if (d_amax_out) {
        mx = warp_max(mx);
        if ((tid & 31) == 0) sh[tid >> 5] = mx;
        __syncthreads();
        if (tid == 0) {
            float g = fmaxf(fmaxf(fmaxf(sh[0], sh[1]), fmaxf(sh[2], sh[3])),
                            fmaxf(fmaxf(sh[4], sh[5]), fmaxf(sh[6], sh[7])));
            atomicMaxNonNegFloat(d_amax_out, g);
        }
    }
}

__global__ void __launch_bounds__(256) k_ctrl_blend_norm_fused_gf_bf16(
    const float* __restrict__ cur, const float* __restrict__ msg,
    const __nv_bfloat16* __restrict__ gf_bf16, const float* __restrict__ norm_w,
    float* __restrict__ out, int M, int D, float* __restrict__ d_amax_out = nullptr) {
    const int m = blockIdx.x;
    if (m >= M) return;
    const int tid = threadIdx.x, D4 = D >> 2;
    __shared__ float sh[32];
    const uint2* gp4 = reinterpret_cast<const uint2*>(gf_bf16 + (size_t)m * (2 * D));
    const uint2* fp4 = reinterpret_cast<const uint2*>(gf_bf16 + (size_t)m * (2 * D) + D);
    float4* op4 = reinterpret_cast<float4*>(out + (size_t)m * D);
    float4 r_out[3];
    float ss = 0.0f;
#pragma unroll
    for (int t = 0; t < 3; ++t) {
        const int i = tid + t * 256;
        if (i < D4) {
            const float4 cv = reinterpret_cast<const float4*>(cur + (size_t)m * D)[i];
            const float4 mv = reinterpret_cast<const float4*>(msg + (size_t)m * D)[i];
            const uint2 g2 = gp4[i], f2 = fp4[i];
            const float2 g0 = bf16x2_to_f32x2(g2.x), g1 = bf16x2_to_f32x2(g2.y);
            const float2 f0 = bf16x2_to_f32x2(f2.x), f1 = bf16x2_to_f32x2(f2.y);
            const float o0 = (1.0f - tanhf(f0.x)) * cv.x + tanhf(g0.x) * mv.x;
            const float o1 = (1.0f - tanhf(f0.y)) * cv.y + tanhf(g0.y) * mv.y;
            const float o2 = (1.0f - tanhf(f1.x)) * cv.z + tanhf(g1.x) * mv.z;
            const float o3 = (1.0f - tanhf(f1.y)) * cv.w + tanhf(g1.y) * mv.w;
            r_out[t] = make_float4(o0, o1, o2, o3);
            ss += o0 * o0 + o1 * o1 + o2 * o2 + o3 * o3;
        }
    }
    float mx = 0.0f;
    const float inv = norm_w ? rsqrtf(block_sum(ss, sh) / (float)D + kInkRmsEps) : 1.0f;
    const float4* w4 = norm_w ? reinterpret_cast<const float4*>(norm_w) : nullptr;
#pragma unroll
    for (int t = 0; t < 3; ++t) {
        const int i = tid + t * 256;
        if (i < D4) {
            float4 v = r_out[t];
            if (w4) {
                const float4 wv = w4[i];
                v = make_float4(v.x * inv * wv.x, v.y * inv * wv.y, v.z * inv * wv.z, v.w * inv * wv.w);
            }
            op4[i] = v;
            if (d_amax_out) mx = fmaxf(mx, fmaxf(fmaxf(fabsf(v.x), fabsf(v.y)), fmaxf(fabsf(v.z), fabsf(v.w))));
        }
    }
    if (d_amax_out) {
        mx = warp_max(mx);
        if ((tid & 31) == 0) sh[tid >> 5] = mx;
        __syncthreads();
        if (tid == 0) {
            float g = fmaxf(fmaxf(fmaxf(sh[0], sh[1]), fmaxf(sh[2], sh[3])),
                            fmaxf(fmaxf(sh[4], sh[5]), fmaxf(sh[6], sh[7])));
            atomicMaxNonNegFloat(d_amax_out, g);
        }
    }
}

// ============================================================================
// 4. Small-M (M <= 16) High-BW FP8 Trunk, Controllers & Tiny-N Projections
// ============================================================================
struct InkMultiProjFp8 {
    const uint8_t* W[5];
    const float*   sc[5];
    const float*   X[5];
    float*         Y[5];
    int            N[5];
    int            cumN[6];
    int            count;
};

__global__ void __launch_bounds__(256) k_gemv_multi_fp8_m1_l1(
    InkMultiProjFp8 mp,
    const __half2* __restrict__ Xh_fallback,
    const __half2* __restrict__ Xh_k,
    const __half2* __restrict__ Xh_v,
    int K) {
    const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
    const int grow0 = blockIdx.x * 8, grow = grow0 + warp;
    if (grow >= mp.cumN[mp.count]) return;
    int pi = 0;
    while (pi + 1 < mp.count && grow0 >= mp.cumN[pi + 1]) ++pi;
    const __half2* Xh = (pi == 3 && Xh_k) ? Xh_k : ((pi == 4 && Xh_v) ? Xh_v : Xh_fallback);
    const int lrow = grow - mp.cumN[pi];
    const uint4* wr = reinterpret_cast<const uint4*>(mp.W[pi] + (size_t)lrow * K);
    const uint4* x4 = reinterpret_cast<const uint4*>(Xh);
    const float w_sc = *mp.sc[pi];
    const int k16_end = K >> 4;
    __half2 acc0 = __float2half2_rn(0.0f), acc1 = __float2half2_rn(0.0f);
    auto accum_u4 = [&](const uint4& wv, int idx16) {
        const uint4 xa = __ldg(x4 + idx16 * 2 + 0), xb = __ldg(x4 + idx16 * 2 + 1);
        acc0 = __hfma2(fp8x2_to_h2((uint16_t)(wv.x & 0xFFFFu)), *reinterpret_cast<const __half2*>(&xa.x), acc0);
        acc1 = __hfma2(fp8x2_to_h2((uint16_t)(wv.x >> 16)),     *reinterpret_cast<const __half2*>(&xa.y), acc1);
        acc0 = __hfma2(fp8x2_to_h2((uint16_t)(wv.y & 0xFFFFu)), *reinterpret_cast<const __half2*>(&xa.z), acc0);
        acc1 = __hfma2(fp8x2_to_h2((uint16_t)(wv.y >> 16)),     *reinterpret_cast<const __half2*>(&xa.w), acc1);
        acc0 = __hfma2(fp8x2_to_h2((uint16_t)(wv.z & 0xFFFFu)), *reinterpret_cast<const __half2*>(&xb.x), acc0);
        acc1 = __hfma2(fp8x2_to_h2((uint16_t)(wv.z >> 16)),     *reinterpret_cast<const __half2*>(&xb.y), acc1);
        acc0 = __hfma2(fp8x2_to_h2((uint16_t)(wv.w & 0xFFFFu)), *reinterpret_cast<const __half2*>(&xb.z), acc0);
        acc1 = __hfma2(fp8x2_to_h2((uint16_t)(wv.w >> 16)),     *reinterpret_cast<const __half2*>(&xb.w), acc1);
    };
    if (k16_end == 144) {
        const uint4 w0 = __ldg(wr + lane + 0), w1 = __ldg(wr + lane + 32);
        const uint4 w2 = __ldg(wr + lane + 64), w3 = __ldg(wr + lane + 96);
        const bool has4 = (lane < 16);
        const uint4 w4 = has4 ? __ldg(wr + lane + 128) : make_uint4(0u, 0u, 0u, 0u);
        accum_u4(w0, lane + 0); accum_u4(w1, lane + 32); accum_u4(w2, lane + 64); accum_u4(w3, lane + 96);
        if (has4) accum_u4(w4, lane + 128);
    } else if (k16_end == 192) {
        const uint4 w0 = __ldg(wr + lane + 0), w1 = __ldg(wr + lane + 32);
        const uint4 w2 = __ldg(wr + lane + 64), w3 = __ldg(wr + lane + 96);
        accum_u4(w0, lane + 0);
        const uint4 w4 = __ldg(wr + lane + 128);
        accum_u4(w1, lane + 32);
        const uint4 w5 = __ldg(wr + lane + 160);
        accum_u4(w2, lane + 64); accum_u4(w3, lane + 96); accum_u4(w4, lane + 128); accum_u4(w5, lane + 160);
    } else {
        for (int k16 = lane; k16 < k16_end; k16 += 32) accum_u4(__ldg(wr + k16), k16);
    }
    const float2 f0 = __half22float2(acc0), f1 = __half22float2(acc1);
    float s = (f0.x + f0.y) + (f1.x + f1.y);
#pragma unroll
    for (int o = 16; o >= 1; o >>= 1) s += __shfl_xor_sync(0xFFFFFFFFu, s, o);
    if (lane == 0) mp.Y[pi][lrow] = s * w_sc;
}

template <int MAX_M>
__global__ void __launch_bounds__(256) k_gemv_multi_fp8_smallM_l1(
    InkMultiProjFp8 mp,
    const __half2* __restrict__ Xh_q, const __half2* __restrict__ Xh_k, const __half2* __restrict__ Xh_v,
    int M, int K) {
    const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
    const int grow0 = blockIdx.x * 8, grow = grow0 + warp;
    if (grow >= mp.cumN[mp.count]) return;
    int pi = 0;
    while (pi + 1 < mp.count && grow0 >= mp.cumN[pi + 1]) ++pi;
    const __half2* Xh = (pi == 3 && Xh_k) ? Xh_k : ((pi == 4 && Xh_v) ? Xh_v : Xh_q);
    const int lrow = grow - mp.cumN[pi];
    const uint4* wr = reinterpret_cast<const uint4*>(mp.W[pi] + (size_t)lrow * K);
    const float w_sc = *mp.sc[pi];
    const int k16_end = K >> 4, k2 = K >> 1;
    __half2 acc0[MAX_M], acc1[MAX_M];
#pragma unroll
    for (int m = 0; m < MAX_M; ++m) { acc0[m] = __float2half2_rn(0.0f); acc1[m] = __float2half2_rn(0.0f); }
    for (int k16 = lane; k16 < k16_end; k16 += 32) {
        const uint4 wv = __ldg(wr + k16);
        const __half2 w0 = fp8x2_to_h2((uint16_t)(wv.x & 0xFFFFu)), w1 = fp8x2_to_h2((uint16_t)(wv.x >> 16));
        const __half2 w2 = fp8x2_to_h2((uint16_t)(wv.y & 0xFFFFu)), w3 = fp8x2_to_h2((uint16_t)(wv.y >> 16));
        const __half2 w4 = fp8x2_to_h2((uint16_t)(wv.z & 0xFFFFu)), w5 = fp8x2_to_h2((uint16_t)(wv.z >> 16));
        const __half2 w6 = fp8x2_to_h2((uint16_t)(wv.w & 0xFFFFu)), w7 = fp8x2_to_h2((uint16_t)(wv.w >> 16));
#pragma unroll
        for (int m = 0; m < MAX_M; ++m) {
            if (m < M) {
                const uint4* x4 = reinterpret_cast<const uint4*>(Xh + (size_t)m * k2);
                const uint4 xa = __ldg(x4 + k16 * 2 + 0), xb = __ldg(x4 + k16 * 2 + 1);
                acc0[m] = __hfma2(w0, *reinterpret_cast<const __half2*>(&xa.x), acc0[m]);
                acc1[m] = __hfma2(w1, *reinterpret_cast<const __half2*>(&xa.y), acc1[m]);
                acc0[m] = __hfma2(w2, *reinterpret_cast<const __half2*>(&xa.z), acc0[m]);
                acc1[m] = __hfma2(w3, *reinterpret_cast<const __half2*>(&xa.w), acc1[m]);
                acc0[m] = __hfma2(w4, *reinterpret_cast<const __half2*>(&xb.x), acc0[m]);
                acc1[m] = __hfma2(w5, *reinterpret_cast<const __half2*>(&xb.y), acc1[m]);
                acc0[m] = __hfma2(w6, *reinterpret_cast<const __half2*>(&xb.z), acc0[m]);
                acc1[m] = __hfma2(w7, *reinterpret_cast<const __half2*>(&xb.w), acc1[m]);
            }
        }
    }
    float* Yp = mp.Y[pi]; const int Np = mp.N[pi];
#pragma unroll
    for (int m = 0; m < MAX_M; ++m) {
        if (m < M) {
            const float2 f0 = __half22float2(acc0[m]), f1 = __half22float2(acc1[m]);
            float s = (f0.x + f0.y) + (f1.x + f1.y);
#pragma unroll
            for (int o = 16; o >= 1; o >>= 1) s += __shfl_xor_sync(0xFFFFFFFFu, s, o);
            if (lane == 0) Yp[(size_t)m * Np + lrow] = s * w_sc;
        }
    }
}

template <int TILE_M>
__global__ void __launch_bounds__(256) k_f32_sink_sconv_fused(
    const float* __restrict__ w_sink, const float* __restrict__ w_kd, const float* __restrict__ w_vd,
    const float* __restrict__ in_q, const float* __restrict__ in_k, const float* __restrict__ in_v,
    float* __restrict__ y_sink, float* __restrict__ y_kd, float* __restrict__ y_vd, int M) {
    const int row = blockIdx.x, m0 = blockIdx.y * TILE_M, tid = threadIdx.x;
    const float *W = nullptr, *X = nullptr; float* Y = nullptr;
    int lrow = 0, N = 0;
    if (row < kInkHeads) { W = w_sink; X = in_q; Y = y_sink; lrow = row; N = kInkHeads; }
    else if (row < kInkHeads + kInkKVHeads * 4) { if (!w_kd) return; W = w_kd; X = in_k; Y = y_kd; lrow = row - kInkHeads; N = kInkKVHeads * 4; }
    else { if (!w_vd) return; W = w_vd; X = in_v; Y = y_vd; lrow = row - (kInkHeads + kInkKVHeads * 4); N = kInkKVHeads * 4; }
    constexpr int D4 = kInkHidden / 4;
    const float4* wr = reinterpret_cast<const float4*>(W + (size_t)lrow * kInkHidden);
    float acc[TILE_M] = {};
    for (int i = tid; i < D4; i += 256) {
        const float4 wv = __ldg(wr + i);
#pragma unroll
        for (int tm = 0; tm < TILE_M; ++tm) {
            const int m = m0 + tm;
            if (m < M) {
                const float4 xv = reinterpret_cast<const float4*>(X + (size_t)m * kInkHidden)[i];
                acc[tm] = fmaf(wv.x, xv.x, fmaf(wv.y, xv.y, fmaf(wv.z, xv.z, fmaf(wv.w, xv.w, acc[tm]))));
            }
        }
    }
    __shared__ float sh[32];
#pragma unroll
    for (int tm = 0; tm < TILE_M; ++tm) {
        const int m = m0 + tm;
        if (m < M) {
            float s = block_sum(acc[tm], sh);
            if (tid == 0) Y[(size_t)m * N + lrow] = s;
        }
    }
}

__global__ void __launch_bounds__(256) k_split_rescale_qkv(
    const float* __restrict__ src, float* __restrict__ y_q, float* __restrict__ y_r, float* __restrict__ y_g,
    float* __restrict__ y_k, float* __restrict__ y_v,
    const float* __restrict__ sc0, const float* __restrict__ sc1, const float* __restrict__ sc2,
    const float* __restrict__ sc3, const float* __restrict__ sc4, int M, int N_tot) {
    const int m = blockIdx.y, c = blockIdx.x * 256 + threadIdx.x;
    if (m >= M || c >= N_tot) return;
    constexpr int N0 = kInkQDim, N1 = N0 + kInkHeads * kInkDRel, N2 = N1 + kInkQDim, N3 = N2 + kInkKVDim;
    const float v = src[(size_t)m * N_tot + c];
    if (c < N0) y_q[(size_t)m * kInkQDim + c] = v * (*sc0);
    else if (c < N1) y_r[(size_t)m * (kInkHeads * kInkDRel) + (c - N0)] = v * (*sc1);
    else if (c < N2) y_g[(size_t)m * kInkQDim + (c - N1)] = v * (*sc2);
    else if (c < N3) { if (y_k) y_k[(size_t)m * kInkKVDim + (c - N2)] = v * (*sc3); }
    else { if (y_v) y_v[(size_t)m * kInkKVDim + (c - N3)] = v * (*sc4); }
}

__global__ void __launch_bounds__(256) k_split_rescale_qkv_from_bf16(
    const __nv_bfloat16* __restrict__ src, float* __restrict__ y_q, float* __restrict__ y_r, float* __restrict__ y_g,
    float* __restrict__ y_k, float* __restrict__ y_v,
    const float* __restrict__ sc0, const float* __restrict__ sc1, const float* __restrict__ sc2,
    const float* __restrict__ sc3, const float* __restrict__ sc4, int M, int N_tot) {
    const int m = blockIdx.y, c2 = blockIdx.x * 256 + threadIdx.x;
    const int N2_tot = N_tot >> 1;
    if (m >= M || c2 >= N2_tot) return;
    constexpr int N0 = kInkQDim >> 1, N1 = N0 + ((kInkHeads * kInkDRel) >> 1), N2 = N1 + (kInkQDim >> 1), N3 = N2 + (kInkKVDim >> 1);
    const float2 f = bf16x2_to_f32x2(reinterpret_cast<const uint32_t*>(src)[(size_t)m * N2_tot + c2]);
    if (c2 < N0) { const float s = *sc0; reinterpret_cast<float2*>(y_q)[(size_t)m * (kInkQDim >> 1) + c2] = make_float2(f.x * s, f.y * s); }
    else if (c2 < N1) { const float s = *sc1; reinterpret_cast<float2*>(y_r)[(size_t)m * ((kInkHeads * kInkDRel) >> 1) + (c2 - N0)] = make_float2(f.x * s, f.y * s); }
    else if (c2 < N2) { const float s = *sc2; reinterpret_cast<float2*>(y_g)[(size_t)m * (kInkQDim >> 1) + (c2 - N1)] = make_float2(f.x * s, f.y * s); }
    else if (c2 < N3) { if (y_k) { const float s = *sc3; reinterpret_cast<float2*>(y_k)[(size_t)m * (kInkKVDim >> 1) + (c2 - N2)] = make_float2(f.x * s, f.y * s); } }
    else { if (y_v) { const float s = *sc4; reinterpret_cast<float2*>(y_v)[(size_t)m * (kInkKVDim >> 1) + (c2 - N3)] = make_float2(f.x * s, f.y * s); } }
}

template <int MAX_M>
__global__ void __launch_bounds__(256) k_ctrl_3x_down_fp8(
    const float* __restrict__ cat3,
    const uint8_t* __restrict__ wd0, const float* __restrict__ sd0,
    const uint8_t* __restrict__ wd1, const float* __restrict__ sd1,
    const uint8_t* __restrict__ wd2, const float* __restrict__ sd2,
    float* __restrict__ gate3, int M) {
    const int r = blockIdx.x, c = blockIdx.y;
    if (r >= kInkGateRank || c >= 3) return;
    constexpr int D2 = 2 * kInkHidden, D2_16 = D2 / 16;
    const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
    const uint8_t* wd = (c == 0) ? wd0 : (c == 1 ? wd1 : wd2);
    const float w_sc  = *((c == 0) ? sd0 : (c == 1 ? sd1 : sd2));
    const float* cat  = cat3 + (size_t)c * M * D2;
    const uint4* wr   = reinterpret_cast<const uint4*>(wd + (size_t)r * D2);
    __half2 acc0[MAX_M], acc1[MAX_M];
#pragma unroll
    for (int m = 0; m < MAX_M; ++m) { acc0[m] = __float2half2_rn(0.0f); acc1[m] = __float2half2_rn(0.0f); }
    for (int i = tid; i < D2_16; i += 256) {
        const uint4 wv = __ldg(wr + i);
        const __half2 w0 = fp8x2_to_h2((uint16_t)(wv.x & 0xFFFFu)), w1 = fp8x2_to_h2((uint16_t)(wv.x >> 16));
        const __half2 w2 = fp8x2_to_h2((uint16_t)(wv.y & 0xFFFFu)), w3 = fp8x2_to_h2((uint16_t)(wv.y >> 16));
        const __half2 w4 = fp8x2_to_h2((uint16_t)(wv.z & 0xFFFFu)), w5 = fp8x2_to_h2((uint16_t)(wv.z >> 16));
        const __half2 w6 = fp8x2_to_h2((uint16_t)(wv.w & 0xFFFFu)), w7 = fp8x2_to_h2((uint16_t)(wv.w >> 16));
#pragma unroll
        for (int m = 0; m < MAX_M; ++m) {
            if (m < M) {
                const float4* xp = reinterpret_cast<const float4*>(cat + (size_t)m * D2 + i * 16);
                const float4 x0 = xp[0], x1 = xp[1], x2 = xp[2], x3 = xp[3];
                acc0[m] = __hfma2(w0, __floats2half2_rn(x0.x, x0.y), acc0[m]);
                acc1[m] = __hfma2(w1, __floats2half2_rn(x0.z, x0.w), acc1[m]);
                acc0[m] = __hfma2(w2, __floats2half2_rn(x1.x, x1.y), acc0[m]);
                acc1[m] = __hfma2(w3, __floats2half2_rn(x1.z, x1.w), acc1[m]);
                acc0[m] = __hfma2(w4, __floats2half2_rn(x2.x, x2.y), acc0[m]);
                acc1[m] = __hfma2(w5, __floats2half2_rn(x2.z, x2.w), acc1[m]);
                acc0[m] = __hfma2(w6, __floats2half2_rn(x3.x, x3.y), acc0[m]);
                acc1[m] = __hfma2(w7, __floats2half2_rn(x3.z, x3.w), acc1[m]);
            }
        }
    }
    __shared__ float s_wr[MAX_M][8];
#pragma unroll
    for (int m = 0; m < MAX_M; ++m) {
        if (m < M) {
            const float2 f0 = __half22float2(acc0[m]), f1 = __half22float2(acc1[m]);
            float s = warp_sum((f0.x + f0.y) + (f1.x + f1.y));
            if (lane == 0) s_wr[m][warp] = s;
        }
    }
    __syncthreads();
    if (tid < M) {
        float s = s_wr[tid][0] + s_wr[tid][1] + s_wr[tid][2] + s_wr[tid][3]
                + s_wr[tid][4] + s_wr[tid][5] + s_wr[tid][6] + s_wr[tid][7];
        gate3[((size_t)c * M + tid) * kInkGateRank + r] = s * w_sc;
    }
}

template <int MAX_M>
__global__ void __launch_bounds__(256) k_ctrl_3x_apply_fp8(
    const float* __restrict__ cur0, const float* __restrict__ cur1, const float* __restrict__ cur2,
    const float* __restrict__ msgs, const float* __restrict__ gate3,
    const uint8_t* __restrict__ wgf0, const float* __restrict__ sgf0,
    const uint8_t* __restrict__ wgf1, const float* __restrict__ sgf1,
    const uint8_t* __restrict__ wgf2, const float* __restrict__ sgf2,
    float* __restrict__ out0, float* __restrict__ out1, float* __restrict__ out2, int M) {
    constexpr int D = kInkHidden;
    const int c = blockIdx.y;
    if (c >= 3) return;
    __shared__ float s_gate[MAX_M][kInkGateRank];
    const int tid = threadIdx.x;
    const float* gp = gate3 + (size_t)c * M * kInkGateRank;
    for (int i = tid; i < M * kInkGateRank; i += 256) s_gate[i >> 6][i & 63] = gp[i];
    __syncthreads();
    const int sub = tid >> 4, lg = tid & 15, d = blockIdx.x * 16 + sub;
    if (d >= D) return;
    const uint8_t* wgf = (c == 0) ? wgf0 : (c == 1 ? wgf1 : wgf2);
    const float w_sc   = *((c == 0) ? sgf0 : (c == 1 ? sgf1 : sgf2));
    const float* cur   = (c == 0) ? cur0 : (c == 1 ? cur1 : cur2);
    const float* msg   = msgs + (size_t)c * M * D;
    float* out         = (c == 0) ? out0 : (c == 1 ? out1 : out2);
    const uint32_t wg4 = *reinterpret_cast<const uint32_t*>(wgf + (size_t)d * kInkGateRank + lg * 4);
    const uint32_t wf4 = *reinterpret_cast<const uint32_t*>(wgf + (size_t)(D + d) * kInkGateRank + lg * 4);
    const float2 g0 = __half22float2(fp8x2_to_h2((uint16_t)(wg4 & 0xFFFFu))), g1 = __half22float2(fp8x2_to_h2((uint16_t)(wg4 >> 16)));
    const float2 f0 = __half22float2(fp8x2_to_h2((uint16_t)(wf4 & 0xFFFFu))), f1 = __half22float2(fp8x2_to_h2((uint16_t)(wf4 >> 16)));
#pragma unroll
    for (int m = 0; m < MAX_M; ++m) {
        if (m < M) {
            const float4 gv = reinterpret_cast<const float4*>(s_gate[m])[lg];
            float ag = g0.x * gv.x + g0.y * gv.y + g1.x * gv.z + g1.y * gv.w;
            float af = f0.x * gv.x + f0.y * gv.y + f1.x * gv.z + f1.y * gv.w;
#pragma unroll
            for (int o = 8; o >= 1; o >>= 1) { ag += __shfl_xor_sync(0xFFFFFFFFu, ag, o); af += __shfl_xor_sync(0xFFFFFFFFu, af, o); }
            if (lg == 0) {
                const size_t idx = (size_t)m * D + d;
                out[idx] = (1.0f - tanhf(af * w_sc)) * cur[idx] + tanhf(ag * w_sc) * msg[idx];
            }
        }
    }
}

template <int MAX_M>
__global__ void __launch_bounds__(256) k_ctrl_1x_down_fp8(
    const float* __restrict__ cat, const uint8_t* __restrict__ wd, const float* __restrict__ sd,
    float* __restrict__ gate, int M) {
    const int r = blockIdx.x;
    if (r >= kInkGateRank) return;
    constexpr int D2 = 2 * kInkHidden, D2_16 = D2 / 16;
    const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
    const float w_sc = *sd;
    const uint4* wr = reinterpret_cast<const uint4*>(wd + (size_t)r * D2);
    __half2 acc0[MAX_M], acc1[MAX_M];
#pragma unroll
    for (int m = 0; m < MAX_M; ++m) { acc0[m] = __float2half2_rn(0.0f); acc1[m] = __float2half2_rn(0.0f); }
    for (int i = tid; i < D2_16; i += 256) {
        const uint4 wv = __ldg(wr + i);
        const __half2 w0 = fp8x2_to_h2((uint16_t)(wv.x & 0xFFFFu)), w1 = fp8x2_to_h2((uint16_t)(wv.x >> 16));
        const __half2 w2 = fp8x2_to_h2((uint16_t)(wv.y & 0xFFFFu)), w3 = fp8x2_to_h2((uint16_t)(wv.y >> 16));
        const __half2 w4 = fp8x2_to_h2((uint16_t)(wv.z & 0xFFFFu)), w5 = fp8x2_to_h2((uint16_t)(wv.z >> 16));
        const __half2 w6 = fp8x2_to_h2((uint16_t)(wv.w & 0xFFFFu)), w7 = fp8x2_to_h2((uint16_t)(wv.w >> 16));
#pragma unroll
        for (int m = 0; m < MAX_M; ++m) {
            if (m < M) {
                const float4* xp = reinterpret_cast<const float4*>(cat + (size_t)m * D2 + i * 16);
                const float4 x0 = xp[0], x1 = xp[1], x2 = xp[2], x3 = xp[3];
                acc0[m] = __hfma2(w0, __floats2half2_rn(x0.x, x0.y), acc0[m]);
                acc1[m] = __hfma2(w1, __floats2half2_rn(x0.z, x0.w), acc1[m]);
                acc0[m] = __hfma2(w2, __floats2half2_rn(x1.x, x1.y), acc0[m]);
                acc1[m] = __hfma2(w3, __floats2half2_rn(x1.z, x1.w), acc1[m]);
                acc0[m] = __hfma2(w4, __floats2half2_rn(x2.x, x2.y), acc0[m]);
                acc1[m] = __hfma2(w5, __floats2half2_rn(x2.z, x2.w), acc1[m]);
                acc0[m] = __hfma2(w6, __floats2half2_rn(x3.x, x3.y), acc0[m]);
                acc1[m] = __hfma2(w7, __floats2half2_rn(x3.z, x3.w), acc1[m]);
            }
        }
    }
    __shared__ float s_wr[MAX_M][8];
#pragma unroll
    for (int m = 0; m < MAX_M; ++m) {
        if (m < M) {
            const float2 f0 = __half22float2(acc0[m]), f1 = __half22float2(acc1[m]);
            float s = warp_sum((f0.x + f0.y) + (f1.x + f1.y));
            if (lane == 0) s_wr[m][warp] = s;
        }
    }
    __syncthreads();
    if (tid < M) {
        float s = s_wr[tid][0] + s_wr[tid][1] + s_wr[tid][2] + s_wr[tid][3]
                + s_wr[tid][4] + s_wr[tid][5] + s_wr[tid][6] + s_wr[tid][7];
        gate[(size_t)tid * kInkGateRank + r] = s * w_sc;
    }
}

template <int MAX_M>
__global__ void __launch_bounds__(256) k_ctrl_1x_apply_fp8(
    const float* __restrict__ cur, const float* __restrict__ msg, const float* __restrict__ gate,
    const uint8_t* __restrict__ wgf, const float* __restrict__ sgf, float* __restrict__ out, int M) {
    constexpr int D = kInkHidden;
    __shared__ float s_gate[MAX_M][kInkGateRank];
    const int tid = threadIdx.x;
    for (int i = tid; i < M * kInkGateRank; i += 256) s_gate[i >> 6][i & 63] = gate[i];
    __syncthreads();
    const int sub = tid >> 4, lg = tid & 15, d = blockIdx.x * 16 + sub;
    if (d >= D) return;
    const float w_sc = *sgf;
    const uint32_t wg4 = *reinterpret_cast<const uint32_t*>(wgf + (size_t)d * kInkGateRank + lg * 4);
    const uint32_t wf4 = *reinterpret_cast<const uint32_t*>(wgf + (size_t)(D + d) * kInkGateRank + lg * 4);
    const float2 g0 = __half22float2(fp8x2_to_h2((uint16_t)(wg4 & 0xFFFFu))), g1 = __half22float2(fp8x2_to_h2((uint16_t)(wg4 >> 16)));
    const float2 f0 = __half22float2(fp8x2_to_h2((uint16_t)(wf4 & 0xFFFFu))), f1 = __half22float2(fp8x2_to_h2((uint16_t)(wf4 >> 16)));
#pragma unroll
    for (int m = 0; m < MAX_M; ++m) {
        if (m < M) {
            const float4 gv = reinterpret_cast<const float4*>(s_gate[m])[lg];
            float ag = g0.x * gv.x + g0.y * gv.y + g1.x * gv.z + g1.y * gv.w;
            float af = f0.x * gv.x + f0.y * gv.y + f1.x * gv.z + f1.y * gv.w;
#pragma unroll
            for (int o = 8; o >= 1; o >>= 1) { ag += __shfl_xor_sync(0xFFFFFFFFu, ag, o); af += __shfl_xor_sync(0xFFFFFFFFu, af, o); }
            if (lg == 0) {
                const size_t idx = (size_t)m * D + d;
                out[idx] = (1.0f - tanhf(af * w_sc)) * cur[idx] + tanhf(ag * w_sc) * msg[idx];
            }
        }
    }
}

__global__ void __launch_bounds__(256) k_rms_norm_3x_h2(
    float* __restrict__ x0, float* __restrict__ x1, float* __restrict__ x2,
    const float* __restrict__ w0, const float* __restrict__ w1, const float* __restrict__ w2,
    __half2* __restrict__ h2_out, int M, int D) {
    const int m = blockIdx.x, c = blockIdx.y;
    if (m >= M || c >= 3) return;
    float* xp = ((c == 0) ? x0 : (c == 1 ? x1 : x2)) + (size_t)m * D;
    const float* wp = (c == 0) ? w0 : (c == 1 ? w1 : w2);
    __half2* hp = h2_out + ((size_t)c * M + m) * (D >> 1);
    const int tid = threadIdx.x, D4 = D >> 2;
    float4* x4 = reinterpret_cast<float4*>(xp);
    const float4* w4 = reinterpret_cast<const float4*>(wp);
    __shared__ float sh[32];
    float4 r[3];
    float ss = 0.0f;
#pragma unroll
    for (int t = 0; t < 3; ++t) {
        const int i = tid + t * 256;
        if (i < D4) { const float4 v = x4[i]; r[t] = v; ss += v.x*v.x + v.y*v.y + v.z*v.z + v.w*v.w; }
    }
    const float inv = rsqrtf(block_sum(ss, sh) / (float)D + kInkRmsEps);
#pragma unroll
    for (int t = 0; t < 3; ++t) {
        const int i = tid + t * 256;
        if (i < D4) {
            const float4 v = r[t], wv = w4[i];
            const float a = v.x * inv * wv.x, b = v.y * inv * wv.y, cc = v.z * inv * wv.z, d = v.w * inv * wv.w;
            x4[i] = make_float4(a, b, cc, d);
            hp[2 * i + 0] = __floats2half2_rn(fminf(fmaxf(a, -65504.0f), 65504.0f), fminf(fmaxf(b, -65504.0f), 65504.0f));
            hp[2 * i + 1] = __floats2half2_rn(fminf(fmaxf(cc, -65504.0f), 65504.0f), fminf(fmaxf(d, -65504.0f), 65504.0f));
        }
    }
}

// ============================================================================
// 5. FP16-Acc Tensor Core FlashAttention-2 Prefill (BN=32 SWA / BN=64 Global)
//    + L1-Resident rel_proj_T (ZERO d_rel_table DRAM!) + Hardware FP8->FP16x2 KV Load!
// ============================================================================
__global__ void __launch_bounds__(256) k_transpose_rel_proj_h2(
    const float* __restrict__ rel_proj, __half2* __restrict__ rel_proj_T) {
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

template <int BN = 32>
__global__ void __launch_bounds__(128) k_flash_gqa_pf_fp16acc(
    const float* __restrict__ q_in,
    const float* __restrict__ r_in,
    const __half2* __restrict__ rel_proj_T,
    const float* __restrict__ sinks,
    const float* __restrict__ gate_in,
    InkKvVisitDesc kv, int slot0, int p0, int M,
    float* __restrict__ out) {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 800)
    constexpr int BM = 64, D = kInkHeadDim, NK8 = BN / 8, NK16 = BN / 16;
    constexpr float kScale = 0.08838834764831845f;
    const int h = blockIdx.x, q_base = blockIdx.y * BM;
    if (q_base >= M || h >= kInkHeads) return;
    const int kh = h / 3;
    const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31, g = lane >> 2, tig = lane & 3;
    const int qr0 = q_base + warp * 16 + g, qr1 = qr0 + 8;
    const bool vq0 = (qr0 < M), vq1 = (qr1 < M);
    const int qp0 = p0 + qr0, qp1 = p0 + qr1;

    __shared__ alignas(16) uint16_t s_q[BM][D];
    __shared__ alignas(16) uint16_t s_k[BN][D];
    __shared__ alignas(16) uint16_t s_v[BN][D];
    auto swiz = [](int r, int c8) -> int { return (c8 ^ (r & 7)) * 8; };

    for (int r = 0; r < 16; ++r) {
        const int sr = warp * 16 + r, qr = q_base + sr, c8 = lane >> 1, sub4 = lane & 1;
        uint32_t* dst = reinterpret_cast<uint32_t*>(&s_q[sr][swiz(sr, c8) + sub4 * 4]);
        if (qr < M) {
            const float4 qv = *reinterpret_cast<const float4*>(q_in + ((size_t)qr * kInkHeads + h) * D + lane * 4);
            dst[0] = pack_fp16x2_rn(qv.x, qv.y);
            dst[1] = pack_fp16x2_rn(qv.z, qv.w);
        } else { dst[0] = 0u; dst[1] = 0u; }
    }
    __half2 r0_h2[8], r1_h2[8];
    if (vq0) {
        const float4* rp = reinterpret_cast<const float4*>(r_in + ((size_t)qr0 * kInkHeads + h) * kInkDRel);
#pragma unroll
        for (int j = 0; j < 4; ++j) { const float4 v = rp[j]; r0_h2[2*j] = __floats2half2_rn(v.x, v.y); r0_h2[2*j+1] = __floats2half2_rn(v.z, v.w); }
    }
    if (vq1) {
        const float4* rp = reinterpret_cast<const float4*>(r_in + ((size_t)qr1 * kInkHeads + h) * kInkDRel);
#pragma unroll
        for (int j = 0; j < 4; ++j) { const float4 v = rp[j]; r1_h2[2*j] = __floats2half2_rn(v.x, v.y); r1_h2[2*j+1] = __floats2half2_rn(v.z, v.w); }
    }
    float m_row0 = vq0 ? sinks[(size_t)qr0 * kInkHeads + h] : -1e30f;
    float m_row1 = vq1 ? sinks[(size_t)qr1 * kInkHeads + h] : -1e30f;
    float l_row0 = 1.0f, l_row1 = 1.0f;
    uint32_t acc_o[16][2];
#pragma unroll
    for (int jd = 0; jd < 16; ++jd) { acc_o[jd][0] = 0u; acc_o[jd][1] = 0u; }

    const int q_end = min(M, q_base + BM);
    const bool is_swa = kv.is_swa;
    const int k_beg = is_swa ? max(0, p0 + q_base - kInkSwaWin + 1) : 0;
    const int k_end = is_swa ? (p0 + q_end) : min(p0 + q_end, kv.cap_pos);
    const bool fast_fp16 = (kv.kv_fmt == 0 && kv.host_cap == 0);
    const bool fast_fp8  = (kv.kv_fmt == 1 && kv.host_cap == 0);
    const size_t kv_slot_base = (size_t)slot0 * kv.cap_pos;
    const int ld_r0 = tid >> 4, ld_c8 = tid & 15;
    const int f8_r0 = tid >> 3, f8_c16 = tid & 7;  // 16 rows x 8 lanes/row (16 B/lane = 128 B/row coalesced!)

    auto load_kv_tile_async = [&](uint16_t (&s_dst)[BN][D], const __half* g_ptr, int kt, int n_k) {
        if (fast_fp16) {
#pragma unroll
            for (int rep = 0; rep < NK8; ++rep) {
                const int ki = rep * 8 + ld_r0;
                uint16_t* dptr = &s_dst[ki][swiz(ki, ld_c8)];
                if (ki < n_k) {
                    const int kp = kt + ki;
                    const int p_idx = is_swa ? (kp % kv.cap_pos) : kp;
                    fa_cp16_async(dptr, g_ptr + ((kv_slot_base + p_idx) * kInkKVHeads + kh) * D + ld_c8 * 8);
                } else { *reinterpret_cast<uint4*>(dptr) = make_uint4(0u, 0u, 0u, 0u); }
            }
            fa_cp_commit();
        }
    };
    auto load_kv_tile_fp8 = [&](uint16_t (&s_dst)[BN][D], const uint8_t* g_ptr, const float* sc_ptr, int kt, int n_k) {
        constexpr int NK16_R = BN / 16;
#pragma unroll
        for (int rep = 0; rep < NK16_R; ++rep) {
            const int ki = rep * 16 + f8_r0;
            uint4* dst0 = reinterpret_cast<uint4*>(&s_dst[ki][swiz(ki, f8_c16 * 2 + 0)]);
            uint4* dst1 = reinterpret_cast<uint4*>(&s_dst[ki][swiz(ki, f8_c16 * 2 + 1)]);
            if (ki < n_k) {
                const int kp = kt + ki;
                const int p_idx = is_swa ? (kp % kv.cap_pos) : kp;
                const size_t hbase = (kv_slot_base + p_idx) * kInkKVHeads + kh;
                const uint4 raw = __ldg(reinterpret_cast<const uint4*>(g_ptr + hbase * D + f8_c16 * 16));
                const __half2 sc2 = __float2half2_rn(__ldg(sc_ptr + hbase));
                const __half2 h0 = __hmul2(fp8x2_to_h2((uint16_t)(raw.x & 0xFFFFu)), sc2);
                const __half2 h1 = __hmul2(fp8x2_to_h2((uint16_t)(raw.x >> 16)),     sc2);
                const __half2 h2 = __hmul2(fp8x2_to_h2((uint16_t)(raw.y & 0xFFFFu)), sc2);
                const __half2 h3 = __hmul2(fp8x2_to_h2((uint16_t)(raw.y >> 16)),     sc2);
                const __half2 h4 = __hmul2(fp8x2_to_h2((uint16_t)(raw.z & 0xFFFFu)), sc2);
                const __half2 h5 = __hmul2(fp8x2_to_h2((uint16_t)(raw.z >> 16)),     sc2);
                const __half2 h6 = __hmul2(fp8x2_to_h2((uint16_t)(raw.w & 0xFFFFu)), sc2);
                const __half2 h7 = __hmul2(fp8x2_to_h2((uint16_t)(raw.w >> 16)),     sc2);
                *dst0 = make_uint4(*reinterpret_cast<const uint32_t*>(&h0), *reinterpret_cast<const uint32_t*>(&h1),
                                   *reinterpret_cast<const uint32_t*>(&h2), *reinterpret_cast<const uint32_t*>(&h3));
                *dst1 = make_uint4(*reinterpret_cast<const uint32_t*>(&h4), *reinterpret_cast<const uint32_t*>(&h5),
                                   *reinterpret_cast<const uint32_t*>(&h6), *reinterpret_cast<const uint32_t*>(&h7));
            } else {
                *dst0 = make_uint4(0u, 0u, 0u, 0u);
                *dst1 = make_uint4(0u, 0u, 0u, 0u);
            }
        }
    };
    auto load_kv_tile_gen = [&](uint16_t (&s_dst)[BN][D], bool is_k, int kt, int n_k) {
        const int d4 = lane, c8 = lane >> 1, sub4 = lane & 1;
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
                    const __half2 sc2 = __float2half2_rn((is_k ? (use_host ? kv.h_k_sc : kv.k_sc) : (use_host ? kv.h_v_sc : kv.v_sc))[hbase]);
                    const uint32_t u = *reinterpret_cast<const uint32_t*>(p + base);
                    const __half2 h0 = __hmul2(fp8x2_to_h2((uint16_t)(u & 0xFFFFu)), sc2);
                    const __half2 h1 = __hmul2(fp8x2_to_h2((uint16_t)(u >> 16)), sc2);
                    dptr[0] = *reinterpret_cast<const uint32_t*>(&h0);
                    dptr[1] = *reinterpret_cast<const uint32_t*>(&h1);
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
            } else { dptr[0] = 0u; dptr[1] = 0u; }
        }
    };

    if (k_beg < k_end) {
        const int n_k0 = min(BN, k_end - k_beg);
        if (fast_fp16)     load_kv_tile_async(s_k, kv.k_f16, k_beg, n_k0);
        else if (fast_fp8) load_kv_tile_fp8(s_k, kv.k_f8, kv.k_sc, k_beg, n_k0);
        else               load_kv_tile_gen(s_k, true, k_beg, n_k0);
    }
    const int q_ld_r = warp * 16 + (lane & 15), q_ld_c = (lane >> 4) & 1;
    const int k_ld_r_in = ((lane >> 4) * 8) + (lane & 7), k_ld_c_in = (lane >> 3) & 1;
    const int v_ld_r_in = lane & 15, v_ld_c_in = (lane >> 4) & 1;

    for (int kt = k_beg; kt < k_end; kt += BN) {
        const int n_k = min(BN, k_end - kt);
        if (fast_fp16) fa_cp_wait0();
        __syncthreads();
        if (fast_fp16) load_kv_tile_async(s_v, kv.v_f16, kt, n_k);

        uint32_t s_h2[NK8][2];
#pragma unroll
        for (int kn = 0; kn < NK8; ++kn) { s_h2[kn][0] = 0u; s_h2[kn][1] = 0u; }
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
            s_reg[kn][0] = f0.x; s_reg[kn][1] = f0.y; s_reg[kn][2] = f1.x; s_reg[kn][3] = f1.y;
        }

        const bool all_far = (!is_swa && n_k == BN && (p0 + q_base - (kt + BN - 1)) >= kInkRelExtent);
        float tm0 = -1e30f, tm1 = -1e30f;
        if (all_far) {
#pragma unroll
            for (int kn = 0; kn < NK8; ++kn) {
                if (vq0) { const float va = s_reg[kn][0]*kScale, vb = s_reg[kn][1]*kScale; s_reg[kn][0]=va; s_reg[kn][1]=vb; tm0=fmaxf(tm0,fmaxf(va,vb)); }
                else     { s_reg[kn][0]=-1e30f; s_reg[kn][1]=-1e30f; }
                if (vq1) { const float va = s_reg[kn][2]*kScale, vb = s_reg[kn][3]*kScale; s_reg[kn][2]=va; s_reg[kn][3]=vb; tm1=fmaxf(tm1,fmaxf(va,vb)); }
                else     { s_reg[kn][2]=-1e30f; s_reg[kn][3]=-1e30f; }
            }
        } else {
#pragma unroll
            for (int kn = 0; kn < NK8; ++kn) {
                const int ki0 = kn * 8 + 2 * tig, kp_a = kt + ki0, kp_b = kp_a + 1;
                const bool in_a = (ki0 < n_k), in_b = (ki0 + 1 < n_k);
                if (vq0) {
                    const int da = qp0 - kp_a, db = qp0 - kp_b;
                    const bool oka = in_a && (da >= 0) && (!is_swa || da < kInkSwaWin);
                    const bool okb = in_b && (db >= 0) && (!is_swa || db < kInkSwaWin);
                    const float va = oka ? (s_reg[kn][0] * kScale + rel_dot16_h2(r0_h2, rel_proj_T, da)) : -1e30f;
                    const float vb = okb ? (s_reg[kn][1] * kScale + rel_dot16_h2(r0_h2, rel_proj_T, db)) : -1e30f;
                    s_reg[kn][0] = va; s_reg[kn][1] = vb; tm0 = fmaxf(tm0, fmaxf(va, vb));
                } else { s_reg[kn][0] = -1e30f; s_reg[kn][1] = -1e30f; }
                if (vq1) {
                    const int da = qp1 - kp_a, db = qp1 - kp_b;
                    const bool oka = in_a && (da >= 0) && (!is_swa || da < kInkSwaWin);
                    const bool okb = in_b && (db >= 0) && (!is_swa || db < kInkSwaWin);
                    const float va = oka ? (s_reg[kn][2] * kScale + rel_dot16_h2(r1_h2, rel_proj_T, da)) : -1e30f;
                    const float vb = okb ? (s_reg[kn][3] * kScale + rel_dot16_h2(r1_h2, rel_proj_T, db)) : -1e30f;
                    s_reg[kn][2] = va; s_reg[kn][3] = vb; tm1 = fmaxf(tm1, fmaxf(va, vb));
                } else { s_reg[kn][2] = -1e30f; s_reg[kn][3] = -1e30f; }
            }
        }
        tm0 = fmaxf(tm0, __shfl_xor_sync(0xFFFFFFFFu, tm0, 1)); tm0 = fmaxf(tm0, __shfl_xor_sync(0xFFFFFFFFu, tm0, 2));
        tm1 = fmaxf(tm1, __shfl_xor_sync(0xFFFFFFFFu, tm1, 1)); tm1 = fmaxf(tm1, __shfl_xor_sync(0xFFFFFFFFu, tm1, 2));
        const float mn0 = fmaxf(m_row0, tm0), mn1 = fmaxf(m_row1, tm1);
        const float al0 = (m_row0 > -1e29f) ? __expf(m_row0 - mn0) : 0.0f;
        const float al1 = (m_row1 > -1e29f) ? __expf(m_row1 - mn1) : 0.0f;
        l_row0 *= al0; l_row1 *= al1;
        const __half2 h_al0 = __float2half2_rn(al0), h_al1 = __float2half2_rn(al1);
#pragma unroll
        for (int jd = 0; jd < 16; ++jd) {
            __half2 o0 = __hmul2(*reinterpret_cast<const __half2*>(&acc_o[jd][0]), h_al0);
            __half2 o1 = __hmul2(*reinterpret_cast<const __half2*>(&acc_o[jd][1]), h_al1);
            acc_o[jd][0] = *reinterpret_cast<const uint32_t*>(&o0);
            acc_o[jd][1] = *reinterpret_cast<const uint32_t*>(&o1);
        }
        m_row0 = mn0; m_row1 = mn1;

        float tl0 = 0.0f, tl1 = 0.0f;
        uint32_t rp[NK16][4];
#pragma unroll
        for (int kk = 0; kk < NK16; ++kk) {
            const float p00 = (s_reg[2*kk+0][0] > -1e29f) ? __expf(s_reg[2*kk+0][0] - mn0) : 0.0f;
            const float p01 = (s_reg[2*kk+0][1] > -1e29f) ? __expf(s_reg[2*kk+0][1] - mn0) : 0.0f;
            const float p02 = (s_reg[2*kk+1][0] > -1e29f) ? __expf(s_reg[2*kk+1][0] - mn0) : 0.0f;
            const float p03 = (s_reg[2*kk+1][1] > -1e29f) ? __expf(s_reg[2*kk+1][1] - mn0) : 0.0f;
            tl0 += (p00 + p01) + (p02 + p03);
            const float p10 = (s_reg[2*kk+0][2] > -1e29f) ? __expf(s_reg[2*kk+0][2] - mn1) : 0.0f;
            const float p11 = (s_reg[2*kk+0][3] > -1e29f) ? __expf(s_reg[2*kk+0][3] - mn1) : 0.0f;
            const float p12 = (s_reg[2*kk+1][2] > -1e29f) ? __expf(s_reg[2*kk+1][2] - mn1) : 0.0f;
            const float p13 = (s_reg[2*kk+1][3] > -1e29f) ? __expf(s_reg[2*kk+1][3] - mn1) : 0.0f;
            tl1 += (p10 + p11) + (p12 + p13);
            rp[kk][0] = pack_fp16x2_rn(p00, p01); rp[kk][1] = pack_fp16x2_rn(p10, p11);
            rp[kk][2] = pack_fp16x2_rn(p02, p03); rp[kk][3] = pack_fp16x2_rn(p12, p13);
        }
        tl0 += __shfl_xor_sync(0xFFFFFFFFu, tl0, 1); tl0 += __shfl_xor_sync(0xFFFFFFFFu, tl0, 2);
        tl1 += __shfl_xor_sync(0xFFFFFFFFu, tl1, 1); tl1 += __shfl_xor_sync(0xFFFFFFFFu, tl1, 2);
        l_row0 += tl0; l_row1 += tl1;

        if (fast_fp16)     fa_cp_wait0();
        else if (fast_fp8) load_kv_tile_fp8(s_v, kv.v_f8, kv.v_sc, kt, n_k);
        else               load_kv_tile_gen(s_v, false, kt, n_k);
        __syncthreads();
        if (kt + BN < k_end) {
            const int n_kn = min(BN, k_end - (kt + BN));
            if (fast_fp16)     load_kv_tile_async(s_k, kv.k_f16, kt + BN, n_kn);
            else if (fast_fp8) load_kv_tile_fp8(s_k, kv.k_f8, kv.k_sc, kt + BN, n_kn);
            else               load_kv_tile_gen(s_k, true, kt + BN, n_kn);
        }
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

    const float inv0 = (l_row0 > 0.0f) ? (1.0f / l_row0) : 0.0f;
    const float inv1 = (l_row1 > 0.0f) ? (1.0f / l_row1) : 0.0f;
    if (vq0) {
        const size_t ob = ((size_t)qr0 * kInkHeads + h) * D;
#pragma unroll
        for (int jd = 0; jd < 16; ++jd) {
            const int c0 = jd * 8 + 2 * tig;
            const float2 f = __half22float2(*reinterpret_cast<const __half2*>(&acc_o[jd][0]));
            float v0 = f.x * inv0, v1 = f.y * inv0;
            if (gate_in) { v0 *= 1.0f / (1.0f + __expf(-gate_in[ob + c0])); v1 *= 1.0f / (1.0f + __expf(-gate_in[ob + c0 + 1])); }
            out[ob + c0] = v0; out[ob + c0 + 1] = v1;
        }
    }
    if (vq1) {
        const size_t ob = ((size_t)qr1 * kInkHeads + h) * D;
#pragma unroll
        for (int jd = 0; jd < 16; ++jd) {
            const int c0 = jd * 8 + 2 * tig;
            const float2 f = __half22float2(*reinterpret_cast<const __half2*>(&acc_o[jd][1]));
            float v0 = f.x * inv1, v1 = f.y * inv1;
            if (gate_in) { v0 *= 1.0f / (1.0f + __expf(-gate_in[ob + c0])); v1 *= 1.0f / (1.0f + __expf(-gate_in[ob + c0 + 1])); }
            out[ob + c0] = v0; out[ob + c0 + 1] = v1;
        }
    }
#else
    (void)q_in; (void)r_in; (void)rel_proj_T; (void)sinks; (void)gate_in; (void)kv; (void)slot0; (void)p0; (void)M; (void)out;
#endif
}

// Fused Mode-3 (C64 Multi-Chain) Flash-GQA with L1-resident rel_proj_T (ZERO k_ink_rel_table launches + ZERO d_rel_table DRAM!)
// and direct hardware fp8x2_to_h2 KV load (skips FP8->FP32->FP16 roundtrip!).
template <int MAX_K = 4>
__global__ void __launch_bounds__(128) k_flash_gqa_multichain_fp8rel(
    const float* __restrict__ q,            // [B * K, 24, 128]
    const float* __restrict__ r_in,         // [B * K, 24, 16]
    const __half2* __restrict__ rel_proj_T, // [1024, 8] __half2
    const float* __restrict__ sinks,        // [B * K, 24]
    const float* __restrict__ gate_attn,    // [B * K, 24, 128]
    InkKvVisitDesc kv,
    const int32_t* __restrict__ slot_ids,   // [B]
    const int32_t* __restrict__ base_pos,   // [B]
    float* __restrict__ out,                // [B * K, 24, 128]
    int B, int K) {
    const int kh = blockIdx.x, b = blockIdx.y;
    if (kh >= kInkKVHeads || b >= B) return;
    const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
    const int sl = slot_ids ? slot_ids[b] : 0;
    const int p0 = base_pos[b];
    const int q_max = p0 + K - 1;
    const int k_beg = kv.is_swa ? max(0, p0 - kInkSwaWin + 1) : 0;
    const int k_end = q_max + 1;
    constexpr float kScale = 0.08838834764831845f;

    __shared__ float s_q[MAX_K][3][kInkHeadDim];
    __shared__ __half2 s_k[16][kInkHeadDim / 2];
    __shared__ __half2 s_v[16][kInkHeadDim / 2];

    const int nq_vecs = K * 3;
    for (int idx = warp; idx < nq_vecs; idx += 4) {
        const int k = idx / 3, g = idx % 3, h = kh * 3 + g, m = b * K + k;
        *(float4*)(&s_q[k][g][lane * 4]) = *(const float4*)(q + ((size_t)m * kInkHeads + h) * kInkHeadDim + lane * 4);
    }
    __syncthreads();

    constexpr int MAX_Q_PER_WARP = (MAX_K * 3 + 3) / 4;
    float mq[MAX_Q_PER_WARP], lq[MAX_Q_PER_WARP];
    float a0[MAX_Q_PER_WARP], a1[MAX_Q_PER_WARP], a2[MAX_Q_PER_WARP], a3[MAX_Q_PER_WARP];
    __half2 r_h2[MAX_Q_PER_WARP];
    int q_k[MAX_Q_PER_WARP], q_g[MAX_Q_PER_WARP];
    int n_my_q = 0;

    for (int idx = warp; idx < nq_vecs; idx += 4) {
        const int k = idx / 3, g = idx % 3, h = kh * 3 + g, m = b * K + k;
        q_k[n_my_q] = k; q_g[n_my_q] = g;
        mq[n_my_q] = sinks[(size_t)m * kInkHeads + h];
        lq[n_my_q] = 1.0f;
        a0[n_my_q] = a1[n_my_q] = a2[n_my_q] = a3[n_my_q] = 0.0f;
        if (lane < 8) {
            const float2 rv = *reinterpret_cast<const float2*>(r_in + ((size_t)m * kInkHeads + h) * kInkDRel + lane * 2);
            r_h2[n_my_q] = __floats2half2_rn(rv.x, rv.y);
        } else {
            r_h2[n_my_q] = __float2half2_rn(0.0f);
        }
        ++n_my_q;
    }

    const bool fast_fp8 = (kv.kv_fmt == 1 && kv.host_cap == 0);
    const size_t sl_base = (size_t)sl * kv.cap_pos;

    for (int kt = k_beg; kt < k_end; kt += 16) {
        const int nt = min(16, k_end - kt);
        if (fast_fp8) {
            for (int j = warp; j < nt; j += 4) {
                const int kp = kt + j;
                const int p_idx = kv.is_swa ? (kp % kv.cap_pos) : kp;
                const size_t hbase = (sl_base + p_idx) * kInkKVHeads + kh;
                const uint32_t ku = __ldg(reinterpret_cast<const uint32_t*>(kv.k_f8 + hbase * kInkHeadDim + lane * 4));
                const uint32_t vu = __ldg(reinterpret_cast<const uint32_t*>(kv.v_f8 + hbase * kInkHeadDim + lane * 4));
                const __half2 ksc2 = __float2half2_rn(__ldg(kv.k_sc + hbase));
                const __half2 vsc2 = __float2half2_rn(__ldg(kv.v_sc + hbase));
                s_k[j][lane * 2 + 0] = __hmul2(fp8x2_to_h2((uint16_t)(ku & 0xFFFFu)), ksc2);
                s_k[j][lane * 2 + 1] = __hmul2(fp8x2_to_h2((uint16_t)(ku >> 16)),     ksc2);
                s_v[j][lane * 2 + 0] = __hmul2(fp8x2_to_h2((uint16_t)(vu & 0xFFFFu)), vsc2);
                s_v[j][lane * 2 + 1] = __hmul2(fp8x2_to_h2((uint16_t)(vu >> 16)),     vsc2);
            }
        } else {
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
        }
        __syncthreads();

        for (int qi = 0; qi < n_my_q; ++qi) {
            const int k = q_k[qi], g = q_g[qi];
            const int qp = p0 + k;
            const int win_beg = kv.is_swa ? max(0, qp - kInkSwaWin + 1) : 0;
            const float4 qv = *(const float4*)(&s_q[k][g][lane * 4]);
            const __half2 rh = r_h2[qi];
            float mw = mq[qi], lw = lq[qi];
            float o0 = a0[qi], o1 = a1[qi], o2 = a2[qi], o3 = a3[qi];

            const int j_lo = max(0, win_beg - kt);
            const int j_hi = min(nt, qp - kt + 1);
            for (int j = j_lo; j < j_hi; ++j) {
                const int kp = kt + j;
                const float2 k01 = __half22float2(s_k[j][lane * 2 + 0]);
                const float2 k23 = __half22float2(s_k[j][lane * 2 + 1]);
                float dot = qv.x * k01.x + qv.y * k01.y + qv.z * k23.x + qv.w * k23.y;
                dot = warp_sum(dot) * kScale;
                const int dist = qp - kp;
                if (dist < kInkRelExtent) {
                    float rp_part = 0.0f;
                    if (lane < 8) {
                        const float2 pf = __half22float2(__hmul2(rh, __ldg(rel_proj_T + (size_t)dist * 8 + lane)));
                        rp_part = pf.x + pf.y;
                    }
                    rp_part += __shfl_xor_sync(0xFFFFFFFFu, rp_part, 1);
                    rp_part += __shfl_xor_sync(0xFFFFFFFFu, rp_part, 2);
                    rp_part += __shfl_xor_sync(0xFFFFFFFFu, rp_part, 4);
                    dot += __shfl_sync(0xFFFFFFFFu, rp_part, 0);
                }
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
        const int k = q_k[qi], g = q_g[qi], h = kh * 3 + g, m = b * K + k;
        const float inv = 1.0f / fmaxf(lq[qi], 1e-20f);
        const size_t ob = ((size_t)m * kInkHeads + h) * kInkHeadDim + lane * 4;
        const float4 gv = *(const float4*)(gate_attn + ob);
        *(float4*)(out + ob) = make_float4(
            a0[qi] * inv / (1.0f + __expf(-gv.x)),
            a1[qi] * inv / (1.0f + __expf(-gv.y)),
            a2[qi] * inv / (1.0f + __expf(-gv.z)),
            a3[qi] * inv / (1.0f + __expf(-gv.w)));
    }
}

// ============================================================================
// 6. Parallel Multi-CTA MoE Binning + High-BW Zero-Smem M=1 & Grouped M=2..16 SF4 MoE
// ============================================================================
__global__ void __launch_bounds__(256) k_moe_bin_par_count(
    const InkExpertDev* __restrict__ ex, const int32_t* __restrict__ topk_idx, int P, int32_t* __restrict__ cnt) {
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

__global__ void __launch_bounds__(128) k_moe_bin_par_prefix(
    int tile_rows, InkMoeTables tb, int32_t* __restrict__ fill_out) {
    __shared__ int s_off[128], s_act[128], s_tcnt[128];
    const int t = threadIdx.x, cv = tb.cnt[t], is_act = cv > 0 ? 1 : 0;
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
    tb.off[t] = off_e; fill_out[t] = off_e;
    if (t == 127) { tb.off[128] = s_off[127]; *tb.nact = s_act[127]; *tb.n_tiles = s_tcnt[127]; }
    if (is_act) tb.active[s_act[t] - 1] = t;
    if (nt > 0) {
        const int tbase = s_tcnt[t] - nt;
        for (int ti = 0; ti < nt; ++ti) tb.tiles[tbase + ti] = make_int2(t, off_e + ti * tile_rows);
    }
}

__global__ void __launch_bounds__(256) k_moe_bin_par_scatter(
    const InkExpertDev* __restrict__ ex, const int32_t* __restrict__ topk_idx, int P,
    int32_t* __restrict__ fill_io, InkMoeTables tb) {
    const int stride = gridDim.x * 256;
    for (int i = blockIdx.x * 256 + threadIdx.x; i < P; i += stride) {
        const int e = topk_idx[i];
        if (ex[e].tier > 1) continue;
        const int pos = atomicAdd(&fill_io[e], 1);
        tb.pairs[pos] = i;
        tb.tok_idx[pos] = i / kInkTopK;
    }
}

__global__ void __launch_bounds__(256) k_moe_gu_m1_l1(
    const uint8_t* __restrict__ sec4_base, const InkExpertDev* __restrict__ experts,
    const int32_t* __restrict__ topk_idx, const __half2* __restrict__ x_h2, __half2* __restrict__ h_h2_out) {
    const int slot = blockIdx.y, tid = threadIdx.x, sub = tid >> 4, lg = tid & 15;
    const int feat = blockIdx.x * 16 + sub;
    if (slot >= kInkTopK || feat >= kInkInter) return;
    const InkExpertDev ed = experts[topk_idx[slot]];
    if (ed.tier > 1 || ed.gu_code != FMT_SF4) return;
    const uint8_t* body = sec4_base + ed.body_off;
    const float2 h01 = *reinterpret_cast<const float2*>(body);
    const float w_outer = h01.x / h01.y;
    constexpr size_t W_P_ROW = kInkHidden / 2, W_S_ROW = kInkHidden / 16;
    const uint8_t* base_p = body + 16;
    const uint8_t* base_s = base_p + (size_t)(2 * kInkInter) * W_P_ROW;
    const uint4* gp16 = reinterpret_cast<const uint4*>(base_p + (size_t)(2 * feat) * W_P_ROW);
    const uint4* up16 = reinterpret_cast<const uint4*>(base_p + (size_t)(2 * feat + 1) * W_P_ROW);
    const uint16_t* gs2 = reinterpret_cast<const uint16_t*>(base_s + (size_t)(2 * feat) * W_S_ROW);
    const uint16_t* us2 = reinterpret_cast<const uint16_t*>(base_s + (size_t)(2 * feat + 1) * W_S_ROW);
    float ag = 0.0f, au = 0.0f;
    auto proc = [&](const uint4& gv, const uint4& uv, uint16_t gsr, uint16_t usr, int idx) {
        const float2 sg = e4m3x2_to_f32x2(gsr), su = e4m3x2_to_f32x2(usr);
        __half2 gw[8], uw[8];
        const __half2* xh0 = x_h2 + idx * 16;
        sf4_decode16_h2(gv.x, gv.y, gw); sf4_decode16_h2(uv.x, uv.y, uw);
        ag = fmaf(dot16_h2_2acc(gw, xh0), sg.x, ag); au = fmaf(dot16_h2_2acc(uw, xh0), su.x, au);
        sf4_decode16_h2(gv.z, gv.w, gw); sf4_decode16_h2(uv.z, uv.w, uw);
        ag = fmaf(dot16_h2_2acc(gw, xh0 + 8), sg.y, ag); au = fmaf(dot16_h2_2acc(uw, xh0 + 8), su.y, au);
    };
    const uint4 g0 = __ldg(gp16 + lg + 0), u0 = __ldg(up16 + lg + 0);
    const uint4 g1 = __ldg(gp16 + lg + 16), u1 = __ldg(up16 + lg + 16);
    const uint16_t gs0 = __ldg(gs2 + lg + 0), us0 = __ldg(us2 + lg + 0);
    const uint16_t gs1 = __ldg(gs2 + lg + 16), us1 = __ldg(us2 + lg + 16);
    const uint4 g2 = __ldg(gp16 + lg + 32), u2 = __ldg(up16 + lg + 32);
    const uint16_t gs2v = __ldg(gs2 + lg + 32), us2v = __ldg(us2 + lg + 32);
    proc(g0, u0, gs0, us0, lg + 0);
    const uint4 g3 = __ldg(gp16 + lg + 48), u3 = __ldg(up16 + lg + 48);
    const uint16_t gs3 = __ldg(gs2 + lg + 48), us3 = __ldg(us2 + lg + 48);
    proc(g1, u1, gs1, us1, lg + 16);
    const bool has4 = (lg < 8);
    const uint4 g4 = has4 ? __ldg(gp16 + lg + 64) : make_uint4(0u, 0u, 0u, 0u);
    const uint4 u4 = has4 ? __ldg(up16 + lg + 64) : make_uint4(0u, 0u, 0u, 0u);
    const uint16_t gs4 = has4 ? __ldg(gs2 + lg + 64) : 0u;
    const uint16_t us4 = has4 ? __ldg(us2 + lg + 64) : 0u;
    proc(g2, u2, gs2v, us2v, lg + 32); proc(g3, u3, gs3, us3, lg + 48);
    if (has4) proc(g4, u4, gs4, us4, lg + 64);
    ag *= w_outer; au *= w_outer;
#pragma unroll
    for (int o = 8; o >= 1; o >>= 1) { ag += __shfl_xor_sync(0xFFFFFFFFu, ag, o); au += __shfl_xor_sync(0xFFFFFFFFu, au, o); }
    const float hv = situ_f32(ag, au);
    const float hv_odd = __shfl_xor_sync(0xFFFFFFFFu, hv, 16);
    if (lg == 0 && (sub & 1) == 0) h_h2_out[(size_t)slot * (kInkInter / 2) + (feat >> 1)] = __floats2half2_rn(hv, hv_odd);
}

__global__ void __launch_bounds__(256) k_moe_dn_m1_l1(
    const uint8_t* __restrict__ sec4_base, const InkExpertDev* __restrict__ experts,
    const int32_t* __restrict__ topk_idx, const float* __restrict__ topk_w,
    const __half2* __restrict__ h_h2, float* __restrict__ y_pair) {
    const int slot = blockIdx.y, tid = threadIdx.x, sub = tid >> 4, lg = tid & 15;
    const int pair_r = blockIdx.x * 16 + sub;
    if (slot >= kInkTopK || pair_r >= kInkHidden / 2) return;
    const InkExpertDev ed = experts[topk_idx[slot]];
    if (ed.tier > 1 || ed.dn_code != FMT_SF4) return;
    const uint8_t* body = sec4_base + ed.body_off;
    const __half2* hr = h_h2 + (size_t)slot * (kInkInter / 2);
    const float2 h23 = *reinterpret_cast<const float2*>(body + 8);
    const float w_outer = h23.x / h23.y;
    constexpr size_t W_P_ROW = kInkInter / 2, W_S_ROW = kInkInter / 16;
    const uint8_t* base_p = body + ed.dn_off;
    const uint8_t* base_s = base_p + (size_t)kInkHidden * W_P_ROW;
    const uint4* p0 = reinterpret_cast<const uint4*>(base_p + (size_t)(2 * pair_r) * W_P_ROW);
    const uint4* p1 = reinterpret_cast<const uint4*>(base_p + (size_t)(2 * pair_r + 1) * W_P_ROW);
    const uint16_t* s0 = reinterpret_cast<const uint16_t*>(base_s + (size_t)(2 * pair_r) * W_S_ROW);
    const uint16_t* s1 = reinterpret_cast<const uint16_t*>(base_s + (size_t)(2 * pair_r + 1) * W_S_ROW);
    float y0 = 0.0f, y1 = 0.0f;
    auto proc = [&](const uint4& v0, const uint4& v1, uint16_t sr0, uint16_t sr1, int idx) {
        const float2 sc0 = e4m3x2_to_f32x2(sr0), sc1 = e4m3x2_to_f32x2(sr1);
        __half2 w0[8], w1[8];
        const __half2* hh = hr + idx * 16;
        sf4_decode16_h2(v0.x, v0.y, w0); sf4_decode16_h2(v1.x, v1.y, w1);
        y0 = fmaf(dot16_h2_2acc(w0, hh), sc0.x, y0); y1 = fmaf(dot16_h2_2acc(w1, hh), sc1.x, y1);
        sf4_decode16_h2(v0.z, v0.w, w0); sf4_decode16_h2(v1.z, v1.w, w1);
        y0 = fmaf(dot16_h2_2acc(w0, hh + 8), sc0.y, y0); y1 = fmaf(dot16_h2_2acc(w1, hh + 8), sc1.y, y1);
    };
    const uint4 a0 = __ldg(p0 + lg + 0), b0 = __ldg(p1 + lg + 0);
    const uint4 a1 = __ldg(p0 + lg + 16), b1 = __ldg(p1 + lg + 16);
    const uint16_t sa0 = __ldg(s0 + lg + 0), sb0 = __ldg(s1 + lg + 0);
    const uint16_t sa1 = __ldg(s0 + lg + 16), sb1 = __ldg(s1 + lg + 16);
    const bool has2 = (lg < 4);
    const uint4 a2 = has2 ? __ldg(p0 + lg + 32) : make_uint4(0u, 0u, 0u, 0u);
    const uint4 b2 = has2 ? __ldg(p1 + lg + 32) : make_uint4(0u, 0u, 0u, 0u);
    const uint16_t sa2 = has2 ? __ldg(s0 + lg + 32) : 0u;
    const uint16_t sb2 = has2 ? __ldg(s1 + lg + 32) : 0u;
    proc(a0, b0, sa0, sb0, lg + 0); proc(a1, b1, sa1, sb1, lg + 16);
    if (has2) proc(a2, b2, sa2, sb2, lg + 32);
    y0 *= w_outer; y1 *= w_outer;
#pragma unroll
    for (int o = 8; o >= 1; o >>= 1) { y0 += __shfl_xor_sync(0xFFFFFFFFu, y0, o); y1 += __shfl_xor_sync(0xFFFFFFFFu, y1, o); }
    if (lg == 0) {
        const float rw = topk_w[slot];
        y_pair[(size_t)slot * kInkHidden + 2 * pair_r]     = y0 * rw;
        y_pair[(size_t)slot * kInkHidden + 2 * pair_r + 1] = y1 * rw;
    }
}

__global__ void __launch_bounds__(256) k_moe_sf4_grouped_reg_gu(
    const uint8_t* __restrict__ sec4_base, const InkExpertDev* __restrict__ experts,
    InkMoeTables tb, const __half2* __restrict__ x_h2, __half2* __restrict__ h_h2_out) {
    const int ai = blockIdx.y;
    if (ai >= *tb.nact) return;
    const int e = tb.active[ai], row0 = tb.off[e], n_total = tb.cnt[e];
    const InkExpertDev ed = experts[e];
    if (ed.tier > 1 || ed.gu_code != FMT_SF4) return;
    const int tid = threadIdx.x, sub = tid >> 4, lg = tid & 15, feat = blockIdx.x * 16 + sub;
    const bool valid_feat = (feat < kInkInter);
    const uint8_t* body = sec4_base + ed.body_off;
    const float2 h01 = *reinterpret_cast<const float2*>(body);
    const float w_outer = h01.x / h01.y;
    constexpr size_t W_P_ROW = kInkHidden / 2, W_S_ROW = kInkHidden / 16;
    const uint8_t* base_p = body + 16;
    const uint8_t* base_s = base_p + (size_t)(2 * kInkInter) * W_P_ROW;
    const uint4* gp16 = valid_feat ? reinterpret_cast<const uint4*>(base_p + (size_t)(2 * feat) * W_P_ROW) : nullptr;
    const uint4* up16 = valid_feat ? reinterpret_cast<const uint4*>(base_p + (size_t)(2 * feat + 1) * W_P_ROW) : nullptr;
    const uint16_t* gs2 = valid_feat ? reinterpret_cast<const uint16_t*>(base_s + (size_t)(2 * feat) * W_S_ROW) : nullptr;
    const uint16_t* us2 = valid_feat ? reinterpret_cast<const uint16_t*>(base_s + (size_t)(2 * feat + 1) * W_S_ROW) : nullptr;
    __shared__ alignas(16) __half2 s_x[8][kInkHidden / 2];

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
                auto proc1 = [&](const uint4& gv, const uint4& uv, uint16_t gsr, uint16_t usr, int idx) {
                    const float2 sg = e4m3x2_to_f32x2(gsr), su = e4m3x2_to_f32x2(usr);
                    __half2 gw[8], uw[8];
                    const __half2* xh0 = s_x[0] + idx * 16;
                    sf4_decode16_h2(gv.x, gv.y, gw); sf4_decode16_h2(uv.x, uv.y, uw);
                    ag = fmaf(dot16_h2_2acc(gw, xh0), sg.x, ag); au = fmaf(dot16_h2_2acc(uw, xh0), su.x, au);
                    sf4_decode16_h2(gv.z, gv.w, gw); sf4_decode16_h2(uv.z, uv.w, uw);
                    ag = fmaf(dot16_h2_2acc(gw, xh0 + 8), sg.y, ag); au = fmaf(dot16_h2_2acc(uw, xh0 + 8), su.y, au);
                };
                const uint4 g0 = __ldg(gp16 + lg + 0), u0 = __ldg(up16 + lg + 0);
                const uint4 g1 = __ldg(gp16 + lg + 16), u1 = __ldg(up16 + lg + 16);
                const uint16_t gs0 = __ldg(gs2 + lg + 0), us0 = __ldg(us2 + lg + 0);
                const uint16_t gs1 = __ldg(gs2 + lg + 16), us1 = __ldg(us2 + lg + 16);
                const uint4 g2 = __ldg(gp16 + lg + 32), u2 = __ldg(up16 + lg + 32);
                const uint16_t gs2v = __ldg(gs2 + lg + 32), us2v = __ldg(us2 + lg + 32);
                proc1(g0, u0, gs0, us0, lg + 0);
                const uint4 g3 = __ldg(gp16 + lg + 48), u3 = __ldg(up16 + lg + 48);
                const uint16_t gs3 = __ldg(gs2 + lg + 48), us3 = __ldg(us2 + lg + 48);
                proc1(g1, u1, gs1, us1, lg + 16);
                const bool has4 = (lg < 8);
                const uint4 g4 = has4 ? __ldg(gp16 + lg + 64) : make_uint4(0u,0u,0u,0u);
                const uint4 u4 = has4 ? __ldg(up16 + lg + 64) : make_uint4(0u,0u,0u,0u);
                const uint16_t gs4 = has4 ? __ldg(gs2 + lg + 64) : 0u, us4 = has4 ? __ldg(us2 + lg + 64) : 0u;
                proc1(g2, u2, gs2v, us2v, lg + 32); proc1(g3, u3, gs3, us3, lg + 48);
                if (has4) proc1(g4, u4, gs4, us4, lg + 64);
                ag *= w_outer; au *= w_outer;
#pragma unroll
                for (int o = 8; o >= 1; o >>= 1) { ag += __shfl_xor_sync(0xFFFFFFFFu, ag, o); au += __shfl_xor_sync(0xFFFFFFFFu, au, o); }
                const float hv = situ_f32(ag, au), hv_odd = __shfl_xor_sync(0xFFFFFFFFu, hv, 16);
                if (lg == 0 && (sub & 1) == 0) h_h2_out[(size_t)(row0 + off_m) * (kInkInter / 2) + (feat >> 1)] = __floats2half2_rn(hv, hv_odd);
            } else if (n_e <= 4) {
                float ag[4] = {}, au[4] = {};
                for (int i = lg; i < 72; i += 16) {
                    const uint4 gv = __ldg(gp16 + i), uv = __ldg(up16 + i);
                    const float2 sg = e4m3x2_to_f32x2(__ldg(gs2 + i)), su = e4m3x2_to_f32x2(__ldg(us2 + i));
                    __half2 gw[8], uw[8];
                    sf4_decode16_h2(gv.x, gv.y, gw); sf4_decode16_h2(uv.x, uv.y, uw);
#pragma unroll
                    for (int m = 0; m < 4; ++m) if (m < n_e) { const __half2* xh = s_x[m] + i*16; ag[m] = fmaf(dot16_h2_2acc(gw, xh), sg.x, ag[m]); au[m] = fmaf(dot16_h2_2acc(uw, xh), su.x, au[m]); }
                    sf4_decode16_h2(gv.z, gv.w, gw); sf4_decode16_h2(uv.z, uv.w, uw);
#pragma unroll
                    for (int m = 0; m < 4; ++m) if (m < n_e) { const __half2* xh = s_x[m] + i*16 + 8; ag[m] = fmaf(dot16_h2_2acc(gw, xh), sg.y, ag[m]); au[m] = fmaf(dot16_h2_2acc(uw, xh), su.y, au[m]); }
                }
#pragma unroll
                for (int m = 0; m < 4; ++m) if (m < n_e) {
                    float gv = ag[m] * w_outer, uv = au[m] * w_outer;
#pragma unroll
                    for (int o = 8; o >= 1; o >>= 1) { gv += __shfl_xor_sync(0xFFFFFFFFu, gv, o); uv += __shfl_xor_sync(0xFFFFFFFFu, uv, o); }
                    const float hv = situ_f32(gv, uv), hv_odd = __shfl_xor_sync(0xFFFFFFFFu, hv, 16);
                    if (lg == 0 && (sub & 1) == 0) h_h2_out[(size_t)(row0 + off_m + m) * (kInkInter / 2) + (feat >> 1)] = __floats2half2_rn(hv, hv_odd);
                }
            } else {
                float ag[8] = {}, au[8] = {};
                for (int i = lg; i < 72; i += 16) {
                    const uint4 gv = __ldg(gp16 + i), uv = __ldg(up16 + i);
                    const float2 sg = e4m3x2_to_f32x2(__ldg(gs2 + i)), su = e4m3x2_to_f32x2(__ldg(us2 + i));
                    __half2 gw[8], uw[8];
                    sf4_decode16_h2(gv.x, gv.y, gw); sf4_decode16_h2(uv.x, uv.y, uw);
#pragma unroll
                    for (int m = 0; m < 8; ++m) if (m < n_e) { const __half2* xh = s_x[m] + i*16; ag[m] = fmaf(dot16_h2_2acc(gw, xh), sg.x, ag[m]); au[m] = fmaf(dot16_h2_2acc(uw, xh), su.x, au[m]); }
                    sf4_decode16_h2(gv.z, gv.w, gw); sf4_decode16_h2(uv.z, uv.w, uw);
#pragma unroll
                    for (int m = 0; m < 8; ++m) if (m < n_e) { const __half2* xh = s_x[m] + i*16 + 8; ag[m] = fmaf(dot16_h2_2acc(gw, xh), sg.y, ag[m]); au[m] = fmaf(dot16_h2_2acc(uw, xh), su.y, au[m]); }
                }
#pragma unroll
                for (int m = 0; m < 8; ++m) if (m < n_e) {
                    float gv = ag[m] * w_outer, uv = au[m] * w_outer;
#pragma unroll
                    for (int o = 8; o >= 1; o >>= 1) { gv += __shfl_xor_sync(0xFFFFFFFFu, gv, o); uv += __shfl_xor_sync(0xFFFFFFFFu, uv, o); }
                    const float hv = situ_f32(gv, uv), hv_odd = __shfl_xor_sync(0xFFFFFFFFu, hv, 16);
                    if (lg == 0 && (sub & 1) == 0) h_h2_out[(size_t)(row0 + off_m + m) * (kInkInter / 2) + (feat >> 1)] = __floats2half2_rn(hv, hv_odd);
                }
            }
        }
        if (off_m + 8 < n_total) __syncthreads();
    }
}

__global__ void __launch_bounds__(256) k_moe_sf4_grouped_reg_dn(
    const uint8_t* __restrict__ sec4_base, const InkExpertDev* __restrict__ experts,
    InkMoeTables tb, const __half2* __restrict__ h_h2, const float* __restrict__ topk_w, float* __restrict__ y_pair) {
    const int ai = blockIdx.y;
    if (ai >= *tb.nact) return;
    const int e = tb.active[ai], row0 = tb.off[e], n_total = tb.cnt[e];
    const InkExpertDev ed = experts[e];
    if (ed.tier > 1 || ed.dn_code != FMT_SF4) return;
    const int tid = threadIdx.x, sub = tid >> 4, lg = tid & 15, pair_r = blockIdx.x * 16 + sub;
    const bool valid_r = (pair_r < kInkHidden / 2);
    const uint8_t* body = sec4_base + ed.body_off;
    const float2 h23 = *reinterpret_cast<const float2*>(body + 8);
    const float w_outer = h23.x / h23.y;
    constexpr size_t W_P_ROW = kInkInter / 2, W_S_ROW = kInkInter / 16;
    const uint8_t* base_p = body + ed.dn_off;
    const uint8_t* base_s = base_p + (size_t)kInkHidden * W_P_ROW;
    const uint4* p0 = valid_r ? reinterpret_cast<const uint4*>(base_p + (size_t)(2 * pair_r) * W_P_ROW) : nullptr;
    const uint4* p1 = valid_r ? reinterpret_cast<const uint4*>(base_p + (size_t)(2 * pair_r + 1) * W_P_ROW) : nullptr;
    const uint16_t* s0 = valid_r ? reinterpret_cast<const uint16_t*>(base_s + (size_t)(2 * pair_r) * W_S_ROW) : nullptr;
    const uint16_t* s1 = valid_r ? reinterpret_cast<const uint16_t*>(base_s + (size_t)(2 * pair_r + 1) * W_S_ROW) : nullptr;
    __shared__ alignas(16) __half2 s_h[8][kInkInter / 2];
    __shared__ float s_rw[8];
    __shared__ int32_t s_pair[8];

    for (int off_m = 0; off_m < n_total; off_m += 8) {
        const int n_e = min(8, n_total - off_m);
        if (tid < n_e) { const int pr = tb.pairs[row0 + off_m + tid]; s_pair[tid] = pr; s_rw[tid] = topk_w[pr] * w_outer; }
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
                    const float2 sc0 = e4m3x2_to_f32x2(sr0), sc1 = e4m3x2_to_f32x2(sr1);
                    __half2 w0[8], w1[8];
                    const __half2* hh = s_h[0] + idx * 16;
                    sf4_decode16_h2(v0.x, v0.y, w0); sf4_decode16_h2(v1.x, v1.y, w1);
                    y0 = fmaf(dot16_h2_2acc(w0, hh), sc0.x, y0); y1 = fmaf(dot16_h2_2acc(w1, hh), sc1.x, y1);
                    sf4_decode16_h2(v0.z, v0.w, w0); sf4_decode16_h2(v1.z, v1.w, w1);
                    y0 = fmaf(dot16_h2_2acc(w0, hh + 8), sc0.y, y0); y1 = fmaf(dot16_h2_2acc(w1, hh + 8), sc1.y, y1);
                };
                const uint4 a0 = __ldg(p0 + lg + 0), b0 = __ldg(p1 + lg + 0);
                const uint4 a1 = __ldg(p0 + lg + 16), b1 = __ldg(p1 + lg + 16);
                const uint16_t sa0 = __ldg(s0 + lg + 0), sb0 = __ldg(s1 + lg + 0);
                const uint16_t sa1 = __ldg(s0 + lg + 16), sb1 = __ldg(s1 + lg + 16);
                const bool has2 = (lg < 4);
                const uint4 a2 = has2 ? __ldg(p0 + lg + 32) : make_uint4(0u,0u,0u,0u);
                const uint4 b2 = has2 ? __ldg(p1 + lg + 32) : make_uint4(0u,0u,0u,0u);
                const uint16_t sa2 = has2 ? __ldg(s0 + lg + 32) : 0u, sb2 = has2 ? __ldg(s1 + lg + 32) : 0u;
                proc1(a0, b0, sa0, sb0, lg + 0); proc1(a1, b1, sa1, sb1, lg + 16);
                if (has2) proc1(a2, b2, sa2, sb2, lg + 32);
#pragma unroll
                for (int o = 8; o >= 1; o >>= 1) { y0 += __shfl_xor_sync(0xFFFFFFFFu, y0, o); y1 += __shfl_xor_sync(0xFFFFFFFFu, y1, o); }
                if (lg == 0) {
                    const float rw = s_rw[0]; const int pr = s_pair[0];
                    y_pair[(size_t)pr * kInkHidden + 2 * pair_r]     = y0 * rw;
                    y_pair[(size_t)pr * kInkHidden + 2 * pair_r + 1] = y1 * rw;
                }
            } else if (n_e <= 4) {
                float y0[4] = {}, y1[4] = {};
                for (int i = lg; i < 36; i += 16) {
                    const uint4 v0 = __ldg(p0 + i), v1 = __ldg(p1 + i);
                    const float2 sc0 = e4m3x2_to_f32x2(__ldg(s0 + i)), sc1 = e4m3x2_to_f32x2(__ldg(s1 + i));
                    __half2 w0[8], w1[8];
                    sf4_decode16_h2(v0.x, v0.y, w0); sf4_decode16_h2(v1.x, v1.y, w1);
#pragma unroll
                    for (int m = 0; m < 4; ++m) if (m < n_e) { const __half2* hh = s_h[m] + i*16; y0[m] = fmaf(dot16_h2_2acc(w0, hh), sc0.x, y0[m]); y1[m] = fmaf(dot16_h2_2acc(w1, hh), sc1.x, y1[m]); }
                    sf4_decode16_h2(v0.z, v0.w, w0); sf4_decode16_h2(v1.z, v1.w, w1);
#pragma unroll
                    for (int m = 0; m < 4; ++m) if (m < n_e) { const __half2* hh = s_h[m] + i*16 + 8; y0[m] = fmaf(dot16_h2_2acc(w0, hh), sc0.y, y0[m]); y1[m] = fmaf(dot16_h2_2acc(w1, hh), sc1.y, y1[m]); }
                }
#pragma unroll
                for (int m = 0; m < 4; ++m) if (m < n_e) {
                    float a0 = y0[m], a1 = y1[m];
#pragma unroll
                    for (int o = 8; o >= 1; o >>= 1) { a0 += __shfl_xor_sync(0xFFFFFFFFu, a0, o); a1 += __shfl_xor_sync(0xFFFFFFFFu, a1, o); }
                    if (lg == 0) { const float rw = s_rw[m]; const int pr = s_pair[m]; y_pair[(size_t)pr * kInkHidden + 2 * pair_r] = a0 * rw; y_pair[(size_t)pr * kInkHidden + 2 * pair_r + 1] = a1 * rw; }
                }
            } else {
                float y0[8] = {}, y1[8] = {};
                for (int i = lg; i < 36; i += 16) {
                    const uint4 v0 = __ldg(p0 + i), v1 = __ldg(p1 + i);
                    const float2 sc0 = e4m3x2_to_f32x2(__ldg(s0 + i)), sc1 = e4m3x2_to_f32x2(__ldg(s1 + i));
                    __half2 w0[8], w1[8];
                    sf4_decode16_h2(v0.x, v0.y, w0); sf4_decode16_h2(v1.x, v1.y, w1);
#pragma unroll
                    for (int m = 0; m < 8; ++m) if (m < n_e) { const __half2* hh = s_h[m] + i*16; y0[m] = fmaf(dot16_h2_2acc(w0, hh), sc0.x, y0[m]); y1[m] = fmaf(dot16_h2_2acc(w1, hh), sc1.x, y1[m]); }
                    sf4_decode16_h2(v0.z, v0.w, w0); sf4_decode16_h2(v1.z, v1.w, w1);
#pragma unroll
                    for (int m = 0; m < 8; ++m) if (m < n_e) { const __half2* hh = s_h[m] + i*16 + 8; y0[m] = fmaf(dot16_h2_2acc(w0, hh), sc0.y, y0[m]); y1[m] = fmaf(dot16_h2_2acc(w1, hh), sc1.y, y1[m]); }
                }
#pragma unroll
                for (int m = 0; m < 8; ++m) if (m < n_e) {
                    float a0 = y0[m], a1 = y1[m];
#pragma unroll
                    for (int o = 8; o >= 1; o >>= 1) { a0 += __shfl_xor_sync(0xFFFFFFFFu, a0, o); a1 += __shfl_xor_sync(0xFFFFFFFFu, a1, o); }
                    if (lg == 0) { const float rw = s_rw[m]; const int pr = s_pair[m]; y_pair[(size_t)pr * kInkHidden + 2 * pair_r] = a0 * rw; y_pair[(size_t)pr * kInkHidden + 2 * pair_r + 1] = a1 * rw; }
                }
            }
        }
        if (off_m + 8 < n_total) __syncthreads();
    }
}

// ============================================================================
// 7. Zero-Extra-VRAM In-Place FP8 Trunk Descriptor (stored on InkCudaModel)
// ============================================================================
struct Ink5060TiFp8Trunk {
    bool ready = false;
    float* d_sc_pool = nullptr;          // 24*6 + 12*8 = 240 floats (< 1 KiB!)
    uint8_t* w_attn[24][6] = {};         // contiguous in d_trunk: 0=wq, 1=wr, 2=wgate, 3=wk, 4=wv, 5=wo
    float*   sc_attn[24][6] = {};
    uint8_t* w_ctrl_dn[12][4] = {};      // 0=attn, 1=k, 2=v, 3=mlp: [64, 4608] FP8
    float*   sc_ctrl_dn[12][4] = {};
    uint8_t* w_ctrl_gf[12][4] = {};      // fused [gain; forget] [4608, 64] FP8
    float*   sc_ctrl_gf[12][4] = {};
    void free_all() {
        if (d_sc_pool) { cudaFree(d_sc_pool); d_sc_pool = nullptr; }
        ready = false;
    }
};

// ============================================================================
// 8. Dynamic cublasLt FP8 Engine (binds to Ink5060TiFp8Trunk -> ~29 MiB scratch!)
// ============================================================================
struct Ink5060TiFp8Engine {
    using fn_create_t     = cublasStatus_t (*)(cublasLtHandle_t*);
    using fn_destroy_t    = cublasStatus_t (*)(cublasLtHandle_t);
    using fn_md_create_t  = cublasStatus_t (*)(cublasLtMatmulDesc_t*, cublasComputeType_t, cudaDataType_t);
    using fn_md_destroy_t = cublasStatus_t (*)(cublasLtMatmulDesc_t);
    using fn_md_setattr_t = cublasStatus_t (*)(cublasLtMatmulDesc_t, cublasLtMatmulDescAttributes_t, const void*, size_t);
    using fn_ml_create_t  = cublasStatus_t (*)(cublasLtMatrixLayout_t*, cudaDataType_t, uint64_t, uint64_t, int64_t);
    using fn_ml_destroy_t = cublasStatus_t (*)(cublasLtMatrixLayout_t);
    using fn_pr_create_t  = cublasStatus_t (*)(cublasLtMatmulPreference_t*);
    using fn_pr_destroy_t = cublasStatus_t (*)(cublasLtMatmulPreference_t);
    using fn_pr_setattr_t = cublasStatus_t (*)(cublasLtMatmulPreference_t, cublasLtMatmulPreferenceAttributes_t, const void*, size_t);
    using fn_get_heur_t   = cublasStatus_t (*)(cublasLtHandle_t, cublasLtMatmulDesc_t, cublasLtMatrixLayout_t, cublasLtMatrixLayout_t,
                                               cublasLtMatrixLayout_t, cublasLtMatrixLayout_t, cublasLtMatmulPreference_t, int,
                                               cublasLtMatmulHeuristicResult_t*, int*);
    using fn_matmul_t     = cublasStatus_t (*)(cublasLtHandle_t, cublasLtMatmulDesc_t, const void*, const void*, cublasLtMatrixLayout_t,
                                               const void*, cublasLtMatrixLayout_t, const void*, const void*, cublasLtMatrixLayout_t,
                                               void*, cublasLtMatrixLayout_t, const cublasLtMatmulAlgo_t*, void*, size_t, cudaStream_t);

    fn_create_t p_create = nullptr; fn_destroy_t p_destroy = nullptr;
    fn_md_create_t p_md_create = nullptr; fn_md_destroy_t p_md_destroy = nullptr; fn_md_setattr_t p_md_setattr = nullptr;
    fn_ml_create_t p_ml_create = nullptr; fn_ml_destroy_t p_ml_destroy = nullptr;
    fn_pr_create_t p_pr_create = nullptr; fn_pr_destroy_t p_pr_destroy = nullptr; fn_pr_setattr_t p_pr_setattr = nullptr;
    fn_get_heur_t p_get_heur = nullptr; fn_matmul_t p_matmul = nullptr;

    cublasLtHandle_t lt = nullptr;
    bool lt_ok = false;
    bool ready = false;
    const Ink5060TiFp8Trunk* tr = nullptr;

    uint8_t* d_slab = nullptr;
    uint8_t* d_ws = nullptr;
    size_t   ws_bytes = 8ull << 20;
    uint8_t* d_x_fp8 = nullptr;
    uint8_t* d_cat3_fp8 = nullptr;
    __nv_bfloat16* d_y_bf16 = nullptr;     // [max_m, 8576] BF16 for packed QKV & ctrl!
    float*   d_x_amax = nullptr;
    float*   d_x_sc = nullptr;
    float*   d_sc_inv8 = nullptr;
    float*   d_sc_one = nullptr;
    __half2* d_rel_proj_T = nullptr;       // [24, 1024, 8] __half2 (0.75 MiB!)
    __half2* d_x_h2 = nullptr;             // [3, 16, 1152] __half2 (0.21 MiB!)
    __half2* d_h_h2 = nullptr;             // [16 * 8, 576] __half2 (0.29 MiB!)
    int32_t* d_bin_fill = nullptr;         // [128]
    int max_m = 0;

    struct PlanEntry {
        cublasLtMatmulDesc_t md = nullptr;
        cublasLtMatrixLayout_t la = nullptr, lb = nullptr, lc = nullptr;
        cublasLtMatmulAlgo_t algo{};
        bool use_bf16_out = false;
        bool valid = false;
    };
    std::unordered_map<uint64_t, PlanEntry> plans;

    static uint64_t plan_key(int M, int N, int K, bool force_bf16 = false) {
        return ((uint64_t)(force_bf16 ? 1u : 0u) << 63) | ((uint64_t)(uint32_t)M << 32) | ((uint64_t)(uint32_t)N << 16) | (uint64_t)(uint32_t)K;
    }

    bool load_lt_symbols() {
        auto sym = [&](const char* name) -> void* { return ink_cublaslt_symbol(name); };
        p_create     = (fn_create_t)sym("cublasLtCreate");
        p_destroy    = (fn_destroy_t)sym("cublasLtDestroy");
        p_md_create  = (fn_md_create_t)sym("cublasLtMatmulDescCreate");
        p_md_destroy = (fn_md_destroy_t)sym("cublasLtMatmulDescDestroy");
        p_md_setattr = (fn_md_setattr_t)sym("cublasLtMatmulDescSetAttribute");
        p_ml_create  = (fn_ml_create_t)sym("cublasLtMatrixLayoutCreate");
        p_ml_destroy = (fn_ml_destroy_t)sym("cublasLtMatrixLayoutDestroy");
        p_pr_create  = (fn_pr_create_t)sym("cublasLtMatmulPreferenceCreate");
        p_pr_destroy = (fn_pr_destroy_t)sym("cublasLtMatmulPreferenceDestroy");
        p_pr_setattr = (fn_pr_setattr_t)sym("cublasLtMatmulPreferenceSetAttribute");
        p_get_heur   = (fn_get_heur_t)sym("cublasLtMatmulAlgoGetHeuristic");
        p_matmul     = (fn_matmul_t)sym("cublasLtMatmul");
        return p_create && p_destroy && p_md_create && p_md_destroy && p_md_setattr &&
               p_ml_create && p_ml_destroy && p_pr_create && p_pr_destroy && p_pr_setattr &&
               p_get_heur && p_matmul;
    }

    template <typename ModelT>
    void init(const ModelT& m, const Ink5060TiFp8Trunk& trunk_fp8, int max_batch, cudaStream_t st) {
        if (!trunk_fp8.ready || m.ls_active) return;
        tr = &trunk_fp8;
        max_m = ((max_batch + 31) & ~31);
        constexpr int kPackedQkvN = kInkQDim + kInkHeads * kInkDRel + kInkQDim + 2 * kInkKVDim; // 8576

        size_t off = 0;
        auto bind = [&](int pass) {
            off = 0;
            auto al = [&](auto*& ptr, size_t n) {
                using T = typename std::remove_pointer<typename std::remove_reference<decltype(ptr)>::type>::type;
                const size_t o = off;
                off += (n * sizeof(T) + 255ull) & ~255ull;
                if (pass == 1) ptr = reinterpret_cast<T*>(d_slab + o);
            };
            al(d_ws,         ws_bytes);
            al(d_x_fp8,      (size_t)max_m * (2 * kInkHidden));
            al(d_cat3_fp8,   3ull * (size_t)max_m * (2 * kInkHidden));
            al(d_y_bf16,     (size_t)max_m * kPackedQkvN);
            al(d_x_amax,     1);
            al(d_x_sc,       1);
            al(d_sc_inv8,    1);
            al(d_sc_one,     1);
            al(d_rel_proj_T, (size_t)m.man.c.n_phys * kInkRelExtent * 8u);
            al(d_x_h2,       3ull * 32ull * (kInkHidden / 2));
            al(d_h_h2,       32ull * kInkTopK * (kInkInter / 2));
            al(d_bin_fill,   128);
        };
        bind(0);
        if (cudaMalloc(&d_slab, off) != cudaSuccess) { cudaGetLastError(); d_slab = nullptr; return; }
        bind(1);

        const float h_inv8 = 0.125f, h_one = 1.0f;
        INK_CK(cudaMemcpyAsync(d_sc_inv8, &h_inv8, sizeof(float), cudaMemcpyHostToDevice, st));
        INK_CK(cudaMemcpyAsync(d_sc_one,  &h_one,  sizeof(float), cudaMemcpyHostToDevice, st));
        for (int ip = 0; ip < m.man.c.n_phys; ++ip) {
            const float* rp = (const float*)m.trunk_ptr("model.llm.layers." + std::to_string(ip) + ".attn.rel_logits_proj.proj");
            k_transpose_rel_proj_h2<<<(kInkRelExtent + 255) / 256, 256, 0, st>>>(rp, d_rel_proj_T + (size_t)ip * kInkRelExtent * 8u);
        }
        INK_CK(cudaStreamSynchronize(st));

        if (load_lt_symbols() && p_create(&lt) == CUBLAS_STATUS_SUCCESS) {
            lt_ok = true;
            static constexpr int kWarmM[] = {32, 64, 128, 192, 256};
            static constexpr int kWarmNK[][2] = {
                {kInkQDim, kInkHidden}, {kInkHeads * kInkDRel, kInkHidden},
                {kInkKVDim, kInkHidden}, {kInkHidden, kInkQDim},
                {kInkQDim + kInkHeads * kInkDRel + kInkQDim, kInkHidden},
                {kPackedQkvN, kInkHidden},
                {kInkGateRank, 2 * kInkHidden}, {2 * kInkHidden, kInkGateRank}
            };
            for (int wm : kWarmM) {
                if (wm > max_m) continue;
                for (const auto& nk : kWarmNK) {
                    (void)get_plan(wm, nk[0], nk[1], false);
                    (void)get_plan(wm, nk[0], nk[1], true);
                }
            }
        }
        ready = true;
        std::fprintf(stderr, "[Ink5060Ti] In-Place FP8 Engine active (lt_ok=%d, scratch=%.1f MiB, max_m=%d, 0 MiB weight dup)\n",
                     lt_ok ? 1 : 0, (double)off / 1048576.0, max_m);
    }

    const PlanEntry* get_plan(int M_pad, int N, int K, bool force_bf16 = false) {
        if (!lt_ok) return nullptr;
        const uint64_t key = plan_key(M_pad, N, K, force_bf16);
        auto it = plans.find(key);
        if (it != plans.end()) return it->second.valid ? &it->second : nullptr;

        PlanEntry pe{};
        auto try_build = [&](bool bf16_out) -> bool {
            cublasLtMatmulDesc_t md = nullptr;
            cublasLtMatrixLayout_t la = nullptr, lb = nullptr, lc = nullptr;
            cublasLtMatmulPreference_t pref = nullptr;
            if (p_md_create(&md, CUBLAS_COMPUTE_32F, CUDA_R_32F) != CUBLAS_STATUS_SUCCESS) return false;
            const cublasOperation_t opT = CUBLAS_OP_T, opN = CUBLAS_OP_N;
            const int8_t fast_acc = 1;
            p_md_setattr(md, CUBLASLT_MATMUL_DESC_TRANSA, &opT, sizeof(opT));
            p_md_setattr(md, CUBLASLT_MATMUL_DESC_TRANSB, &opN, sizeof(opN));
            p_md_setattr(md, CUBLASLT_MATMUL_DESC_FAST_ACCUM, &fast_acc, sizeof(fast_acc));
            p_md_setattr(md, CUBLASLT_MATMUL_DESC_A_SCALE_POINTER, &tr->d_sc_pool, sizeof(tr->d_sc_pool));
            p_md_setattr(md, CUBLASLT_MATMUL_DESC_B_SCALE_POINTER, &d_x_sc, sizeof(d_x_sc));
            const cudaDataType_t out_dt = bf16_out ? CUDA_R_16BF : CUDA_R_32F;
            if (p_ml_create(&la, CUDA_R_8F_E4M3, K, N, K) != CUBLAS_STATUS_SUCCESS ||
                p_ml_create(&lb, CUDA_R_8F_E4M3, K, M_pad, K) != CUBLAS_STATUS_SUCCESS ||
                p_ml_create(&lc, out_dt, N, M_pad, N) != CUBLAS_STATUS_SUCCESS ||
                p_pr_create(&pref) != CUBLAS_STATUS_SUCCESS) {
                if (la) p_ml_destroy(la); if (lb) p_ml_destroy(lb); if (lc) p_ml_destroy(lc);
                if (pref) p_pr_destroy(pref); p_md_destroy(md);
                return false;
            }
            p_pr_setattr(pref, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &ws_bytes, sizeof(ws_bytes));
            cublasLtMatmulHeuristicResult_t res{};
            int found = 0;
            cublasStatus_t hs = p_get_heur(lt, md, la, lb, lc, lc, pref, 1, &res, &found);
            p_pr_destroy(pref);
            if (hs != CUBLAS_STATUS_SUCCESS || found <= 0) {
                p_ml_destroy(la); p_ml_destroy(lb); p_ml_destroy(lc); p_md_destroy(md);
                return false;
            }
            pe.md = md; pe.la = la; pe.lb = lb; pe.lc = lc; pe.algo = res.algo; pe.use_bf16_out = bf16_out; pe.valid = true;
            return true;
        };
        if (force_bf16) try_build(true);
        else { if (!try_build(false)) try_build(true); }
        auto ins = plans.emplace(key, pe);
        return ins.first->second.valid ? &ins.first->second : nullptr;
    }

    void quantize_x(const float* X, int M, int M_pad, int K, cudaStream_t st) {
        const size_t nv = (size_t)M * K, np = (size_t)M_pad * K;
        if (np <= 4096)  { k_fp8_quant_f32_1cta<<<1, 256, 0, st>>>(X, nv, np, d_x_sc, d_x_fp8); return; }
        if (np <= 24576) { k_fp8_quant_f32_1cta_1024<<<1, 1024, 0, st>>>(X, nv, np, d_x_sc, d_x_fp8); return; }
        INK_CK(cudaMemsetAsync(d_x_amax, 0, 4, st));
        const unsigned g1 = (unsigned)std::min<size_t>(512, std::max<size_t>(1, (nv / 4 + 255) / 256));
        const unsigned g2 = (unsigned)std::min<size_t>(512, std::max<size_t>(1, (np / 4 + 255) / 256));
        k_fp8_absmax_f32<<<g1, 256, 0, st>>>(X, nv, d_x_amax);
        k_fp8_quant_f32_padded<<<g2, 256, 0, st>>>(X, nv, np, d_x_amax, d_x_sc, d_x_fp8);
    }

    void quantize_x_bf16(const __nv_bfloat16* X, int M, int M_pad, int K, cudaStream_t st) {
        const size_t nv = (size_t)M * K, np = (size_t)M_pad * K;
        if (np <= 4096) { k_fp8_quant_bf16_1cta<<<1, 256, 0, st>>>(X, nv, np, d_x_sc, d_x_fp8); return; }
        INK_CK(cudaMemsetAsync(d_x_amax, 0, 4, st));
        const unsigned g1 = (unsigned)std::min<size_t>(512, std::max<size_t>(1, (nv / 2 + 255) / 256));
        const unsigned g2 = (unsigned)std::min<size_t>(512, std::max<size_t>(1, (np / 2 + 255) / 256));
        k_fp8_absmax_bf16<<<g1, 256, 0, st>>>(X, nv, d_x_amax);
        k_fp8_quant_bf16_padded<<<g2, 256, 0, st>>>(X, nv, np, d_x_amax, d_x_sc, d_x_fp8);
    }

    bool run_ctrl_fp8(int ci, int ck, const float* cur, const float* msg, const float* norm_w,
                      float* out, float* d_gate_f32, int M, bool x_fp8_ready, cudaStream_t st,
                      const uint8_t* x_fp8_override = nullptr) {
        if (!ready || !lt_ok || ci < 0 || ci >= 12 || ck < 0 || ck >= 4) return false;
        const int M_pad = (M <= 32) ? 32 : ((M + 31) & ~31);
        if (M_pad > max_m) return false;
        constexpr int D = kInkHidden, D2 = 2 * D;
        const PlanEntry* pe1 = get_plan(M_pad, kInkGateRank, D2, false);
        const PlanEntry* pe2 = get_plan(M_pad, D2, kInkGateRank, true);
        if (!pe1 || !pe2 || !pe2->use_bf16_out) return false;

        const uint8_t* x1_ptr = x_fp8_override ? x_fp8_override : d_x_fp8;
        if (!x_fp8_ready && !x_fp8_override) {
            k_ctrl_cat_rms_to_fp8<<<M_pad, 256, 0, st>>>(cur, msg, d_x_fp8, M, M_pad, D);
        }
        const float alpha = 1.0f, beta = 0.0f;
        const float* sd1 = tr->sc_ctrl_dn[ci][ck];
        p_md_setattr(pe1->md, CUBLASLT_MATMUL_DESC_A_SCALE_POINTER, &sd1, sizeof(sd1));
        p_md_setattr(pe1->md, CUBLASLT_MATMUL_DESC_B_SCALE_POINTER, &d_sc_inv8, sizeof(d_sc_inv8));
        if (pe1->use_bf16_out) {
            if (p_matmul(lt, pe1->md, &alpha, tr->w_ctrl_dn[ci][ck], pe1->la, x1_ptr, pe1->lb,
                         &beta, d_y_bf16, pe1->lc, d_y_bf16, pe1->lc, &pe1->algo, d_ws, ws_bytes, st) != CUBLAS_STATUS_SUCCESS) return false;
            quantize_x_bf16(d_y_bf16, M, M_pad, kInkGateRank, st);
        } else {
            if (p_matmul(lt, pe1->md, &alpha, tr->w_ctrl_dn[ci][ck], pe1->la, x1_ptr, pe1->lb,
                         &beta, d_gate_f32, pe1->lc, d_gate_f32, pe1->lc, &pe1->algo, d_ws, ws_bytes, st) != CUBLAS_STATUS_SUCCESS) return false;
            quantize_x(d_gate_f32, M, M_pad, kInkGateRank, st);
        }
        const float* sd2 = tr->sc_ctrl_gf[ci][ck];
        p_md_setattr(pe2->md, CUBLASLT_MATMUL_DESC_A_SCALE_POINTER, &sd2, sizeof(sd2));
        p_md_setattr(pe2->md, CUBLASLT_MATMUL_DESC_B_SCALE_POINTER, &d_x_sc, sizeof(d_x_sc));
        if (p_matmul(lt, pe2->md, &alpha, tr->w_ctrl_gf[ci][ck], pe2->la, d_x_fp8, pe2->lb,
                     &beta, d_y_bf16, pe2->lc, d_y_bf16, pe2->lc, &pe2->algo, d_ws, ws_bytes, st) != CUBLAS_STATUS_SUCCESS) return false;
        k_ctrl_blend_norm_fused_gf_bf16<<<M, 256, 0, st>>>(cur, msg, d_y_bf16, norm_w, out, M, D, nullptr);
        return true;
    }

    bool gemm_fp8_packed_qkv(int ip, bool share_qkv, const float* in_q,
                             float* y_q, float* y_r, float* y_g, float* y_k, float* y_v,
                             int M, cudaStream_t st) {
        if (!ready || !lt_ok) return false;
        const int M_pad = (M <= 32) ? 32 : ((M + 31) & ~31);
        if (M_pad > max_m) return false;
        constexpr int kRDim = kInkHeads * kInkDRel;
        const int N_tot = share_qkv ? (kInkQDim + kRDim + kInkQDim + 2 * kInkKVDim) : (kInkQDim + kRDim + kInkQDim);
        const PlanEntry* pe = get_plan(M_pad, N_tot, kInkHidden, true);
        if (!pe || !pe->use_bf16_out) return false;
        quantize_x(in_q, M, M_pad, kInkHidden, st);
        p_md_setattr(pe->md, CUBLASLT_MATMUL_DESC_A_SCALE_POINTER, &d_sc_one, sizeof(d_sc_one));
        p_md_setattr(pe->md, CUBLASLT_MATMUL_DESC_B_SCALE_POINTER, &d_x_sc, sizeof(d_x_sc));
        const float alpha = 1.0f, beta = 0.0f;
        if (p_matmul(lt, pe->md, &alpha, tr->w_attn[ip][0], pe->la, d_x_fp8, pe->lb,
                     &beta, d_y_bf16, pe->lc, d_y_bf16, pe->lc, &pe->algo, d_ws, ws_bytes, st) != CUBLAS_STATUS_SUCCESS) return false;
        k_split_rescale_qkv_from_bf16<<<dim3(((N_tot >> 1) + 255) / 256, M), 256, 0, st>>>(
            d_y_bf16, y_q, y_r, y_g, share_qkv ? y_k : nullptr, share_qkv ? y_v : nullptr,
            tr->sc_attn[ip][0], tr->sc_attn[ip][1], tr->sc_attn[ip][2], tr->sc_attn[ip][3], tr->sc_attn[ip][4], M, N_tot);
        return true;
    }

    bool gemm_fp8_tn(const uint8_t* W_fp8, const float* w_sc, const float* X, float* Y,
                     int M, int N, int K, bool reuse_x, cudaStream_t st) {
        if (!ready || !lt_ok || (N & 15) != 0 || (K & 15) != 0) return false;
        const int M_pad = (M <= 32) ? 32 : ((M + 31) & ~31);
        if (M_pad > max_m) return false;
        const PlanEntry* pe = get_plan(M_pad, N, K, M != M_pad);
        if (!pe) return false;
        if (!reuse_x) quantize_x(X, M, M_pad, K, st);
        p_md_setattr(pe->md, CUBLASLT_MATMUL_DESC_A_SCALE_POINTER, &w_sc, sizeof(w_sc));
        p_md_setattr(pe->md, CUBLASLT_MATMUL_DESC_B_SCALE_POINTER, &d_x_sc, sizeof(d_x_sc));
        const float alpha = 1.0f, beta = 0.0f;
        if (!pe->use_bf16_out) {
            return p_matmul(lt, pe->md, &alpha, W_fp8, pe->la, d_x_fp8, pe->lb,
                            &beta, Y, pe->lc, Y, pe->lc, &pe->algo, d_ws, ws_bytes, st) == CUBLAS_STATUS_SUCCESS;
        }
        if (p_matmul(lt, pe->md, &alpha, W_fp8, pe->la, d_x_fp8, pe->lb,
                     &beta, d_y_bf16, pe->lc, d_y_bf16, pe->lc, &pe->algo, d_ws, ws_bytes, st) != CUBLAS_STATUS_SUCCESS) return false;
        const size_t n_out = (size_t)M * N;
        k_bf16_to_f32_fast<<<(unsigned)std::min<size_t>(1024, std::max<size_t>(1, (n_out / 2 + 255) / 256)), 256, 0, st>>>(d_y_bf16, Y, n_out);
        return true;
    }

    void free_all() {
        for (auto& kv : plans) {
            if (kv.second.la) p_ml_destroy(kv.second.la);
            if (kv.second.lb) p_ml_destroy(kv.second.lb);
            if (kv.second.lc) p_ml_destroy(kv.second.lc);
            if (kv.second.md) p_md_destroy(kv.second.md);
        }
        plans.clear();
        if (lt && p_destroy) { p_destroy(lt); lt = nullptr; }
        if (d_slab) { cudaFree(d_slab); d_slab = nullptr; }
        d_ws = nullptr; d_x_fp8 = nullptr; d_cat3_fp8 = nullptr; d_y_bf16 = nullptr;
        d_x_amax = nullptr; d_x_sc = nullptr; d_sc_inv8 = nullptr; d_sc_one = nullptr;
        d_rel_proj_T = nullptr; d_x_h2 = nullptr; d_h_h2 = nullptr; d_bin_fill = nullptr;
        lt_ok = false; ready = false; tr = nullptr;
    }
};

}  // namespace strata::ink::opt5060
