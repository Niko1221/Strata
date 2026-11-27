// engine/src/inkling2/cuda/ink_rtx5090_fp8.cuh - Opt-in RTX 5090 (sm_120) FP8_E4M3 Trunk & Controller Cache + Dynamic cublasLt FP8 TC
//
// MIT License
// Copyright (c) 2026 Niko1221 and the Strata contributors
//
// Strictly OFF by default (active only when STRATA_INK_RTX5090=1 && !STRATA_INK_RTX5090_EXACT=1 on sm_120).
// Loads cublasLt dynamically via dlsym/dlopen (zero CMakeLists.txt edits!) for 838 TFLOPS FP8 prefill/C64 GEMMs,
// and uses native sm_120 `cvt.rn.f16x2.e4m3x2` + __hfma2 GEMVs for 2x-4x less trunk/controller bandwidth at M<=24!
#pragma once

#include <cublasLt.h>
#include "ink_cublaslt_dyn.cuh"
#include <cuda_fp8.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <cstdint>
#include <cstdio>
#include <unordered_map>

#ifndef _WIN32
#include <dlfcn.h>
#else
#include <windows.h>
#endif

#include "ink_cuda.cuh"
#include "ink_kernels.cuh"
#include "ink_rtx5090_cfg.cuh"

namespace strata::ink::rtx5090 {

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

// Atomic max for non-negative float (IEEE-754 ordered as uint32)
__device__ __forceinline__ void atomicMaxNonNegFloat(float* addr, float val) {
    atomicMax(reinterpret_cast<uint32_t*>(addr), __float_as_uint(val));
}

// Pass 1: Compute per-tensor absmax of BF16 or F32 matrix -> d_scale[0] = max(amax / 448.0f, 1e-12f)
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
    const float am = *d_amax;
    *d_scale = fmaxf(am * (1.0f / 448.0f), 1e-12f);
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

__global__ void __launch_bounds__(256) k_fp8_absmax_q8row(
    const uint8_t* __restrict__ src, int rows, int cols, float* __restrict__ d_amax) {
    const float* row_sc = reinterpret_cast<const float*>(src);
    const int8_t* q8 = reinterpret_cast<const int8_t*>(src + (size_t)rows * sizeof(float));
    const size_t n4 = ((size_t)rows * cols) >> 2;
    const int cols4 = cols >> 2;
    const uint32_t* q4 = reinterpret_cast<const uint32_t*>(q8);
    float mx = 0.0f;
    for (size_t i = (size_t)blockIdx.x * 256 + threadIdx.x; i < n4; i += (size_t)gridDim.x * 256) {
        const int r = (int)(i / cols4);
        const float sc = row_sc[r];
        const uint32_t u = q4[i];
        const float v0 = (float)(int8_t)(u & 0xFFu) * sc;
        const float v1 = (float)(int8_t)((u >> 8) & 0xFFu) * sc;
        const float v2 = (float)(int8_t)((u >> 16) & 0xFFu) * sc;
        const float v3 = (float)(int8_t)(u >> 24) * sc;
        mx = fmaxf(mx, fmaxf(fmaxf(fabsf(v0), fabsf(v1)), fmaxf(fabsf(v2), fabsf(v3))));
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

__global__ void __launch_bounds__(256) k_fp8_quant_q8row(
    const uint8_t* __restrict__ src, int rows, int cols, const float* __restrict__ d_scale, uint8_t* __restrict__ dst) {
    const float inv = 1.0f / (*d_scale);
    const float* row_sc = reinterpret_cast<const float*>(src);
    const int8_t* q8 = reinterpret_cast<const int8_t*>(src + (size_t)rows * sizeof(float));
    const size_t n4 = ((size_t)rows * cols) >> 2;
    const int cols4 = cols >> 2;
    const uint32_t* q4 = reinterpret_cast<const uint32_t*>(q8);
    uint32_t* d4 = reinterpret_cast<uint32_t*>(dst);
    for (size_t i = (size_t)blockIdx.x * 256 + threadIdx.x; i < n4; i += (size_t)gridDim.x * 256) {
        const int r = (int)(i / cols4);
        const float sc = row_sc[r] * inv;
        const uint32_t u = q4[i];
        const uint16_t lo = f32x2_to_fp8x2((float)(int8_t)(u & 0xFFu) * sc, (float)(int8_t)((u >> 8) & 0xFFu) * sc);
        const uint16_t hi = f32x2_to_fp8x2((float)(int8_t)((u >> 16) & 0xFFu) * sc, (float)(int8_t)(u >> 24) * sc);
        d4[i] = (uint32_t)lo | ((uint32_t)hi << 16);
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

// Single-CTA 1024-thread exact absmax + quantize for medium matrices (M<=64, K<=3072 -> <=196K elems in 1 launch!)
__global__ void __launch_bounds__(1024) k_fp8_quant_f32_1cta_1024(
    const float* __restrict__ src, size_t n_valid, size_t n_pad,
    float* __restrict__ d_scale, uint8_t* __restrict__ dst) {
    const int tid = threadIdx.x;
    const size_t nv4 = n_valid >> 2;
    const size_t np4 = n_pad >> 2;
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

// Single-CTA exact absmax + quantize for small matrices (e.g., C64 gate [64, 64] = 4096 elems -> 1 launch!)
__global__ void __launch_bounds__(256) k_fp8_quant_bf16_1cta(
    const __nv_bfloat16* __restrict__ src, size_t n_valid, size_t n_pad,
    float* __restrict__ d_scale, uint8_t* __restrict__ dst) {
    const int tid = threadIdx.x;
    const size_t nv2 = n_valid >> 1;
    const size_t np2 = n_pad >> 1;
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

__global__ void __launch_bounds__(256) k_fp8_quant_f32_1cta(
    const float* __restrict__ src, size_t n_valid, size_t n_pad,
    float* __restrict__ d_scale, uint8_t* __restrict__ dst) {
    const int tid = threadIdx.x;
    const size_t nv4 = n_valid >> 2;
    const size_t np4 = n_pad >> 2;
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

// Zero-F32-Intermediate Controller CatRMS -> FP8 with exact power-of-two scale 0.125f (max=56 >= sqrt(2304)=48!)
__global__ void __launch_bounds__(256) k_ink_ctrl_cat_rms_to_fp8(
    const float* __restrict__ cur,
    const float* __restrict__ msg,
    uint8_t* __restrict__ dst_fp8,   // [M_pad, 2*D]
    int M, int M_pad, int D) {
    const int m = blockIdx.x;
    if (m >= M_pad) return;
    const int tid = threadIdx.x;
    const int D4 = D >> 2;
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
        const uint16_t c0 = f32x2_to_fp8x2(cv.x * ic, cv.y * ic);
        const uint16_t c1 = f32x2_to_fp8x2(cv.z * ic, cv.w * ic);
        const uint16_t m0 = f32x2_to_fp8x2(gv.x * im, gv.y * im);
        const uint16_t m1 = f32x2_to_fp8x2(gv.z * im, gv.w * im);
        d0[i] = (uint32_t)c0 | ((uint32_t)c1 << 16);
        d1[i] = (uint32_t)m0 | ((uint32_t)m1 << 16);
    }
}

// Fused Pre-MLP Depth Mix + CatRMS -> FP8 directly (1-pass shared-mem stp + register-cached mo/mp!)
// When BF16_ST=true (prefill M>=1024): reads stp.p[0..S-2] & msg as BF16 (2x less GDDR7!), writes moe_out as BF16!
template <bool BF16_ST = false>
__global__ void __launch_bounds__(256) k_ink_mlp_res_to_fp8(
    DepthStatePtrs stp,
    const float* __restrict__ q_mlp,
    const float* __restrict__ msg,
    float* __restrict__ moe_out,     // [M, D] (F32 if !BF16_ST, BF16-in-place if BF16_ST)
    uint8_t* __restrict__ dst_fp8,   // [M_pad, 2*D]
    int M, int M_pad, int D) {
    const int m = blockIdx.x;
    if (m >= M_pad) return;
    const int tid = threadIdx.x;
    const int D4 = D >> 2;
    uint32_t* d0 = reinterpret_cast<uint32_t*>(dst_fp8 + (size_t)m * (2 * D));
    uint32_t* d1 = reinterpret_cast<uint32_t*>(dst_fp8 + (size_t)m * (2 * D) + D);
    if (m >= M) {
        for (int i = tid; i < D4; i += 256) { d0[i] = 0u; d1[i] = 0u; }
        return;
    }
    const int warp = tid >> 5;
    const int lane = tid & 31;
    const int S = stp.count;
    const float inv_sqrt_d = rsqrtf((float)D);
    __shared__ float s_wred[8][2][8];
    __shared__ float s_sinv[8];
    __shared__ float s_prob[8];
    __shared__ float sh[32];
    __shared__ alignas(16) uint32_t s_st[6][kInkHidden / 2];  // 27.6 KB BF16 cache -> ZERO GDDR7 re-read of stp!
    const bool use_smem = (S <= 6 && D == kInkHidden);

    const float4* q4 = reinterpret_cast<const float4*>(q_mlp);
    for (int s = 0; s < S; ++s) {
        const bool is_bf16_s = BF16_ST && (s + 1 < S);
        float ss = 0.0f, dt = 0.0f;
        if (is_bf16_s) {
            const uint2* sp2 = reinterpret_cast<const uint2*>(reinterpret_cast<const __nv_bfloat16*>(stp.p[s]) + (size_t)m * D);
            for (int i = tid; i < D4; i += 256) {
                const uint2 u = sp2[i];
                if (use_smem) { s_st[s][2 * i + 0] = u.x; s_st[s][2 * i + 1] = u.y; }
                const float2 f0 = bf16x2_to_f32x2(u.x), f1 = bf16x2_to_f32x2(u.y);
                const float4 qv = q4[i];
                ss += f0.x * f0.x + f0.y * f0.y + f1.x * f1.x + f1.y * f1.y;
                dt = fmaf(f0.x, qv.x, dt); dt = fmaf(f0.y, qv.y, dt);
                dt = fmaf(f1.x, qv.z, dt); dt = fmaf(f1.y, qv.w, dt);
            }
        } else {
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

    float4 mo_reg[3], mp_reg[3];  // D4 = 576 <= 3 * 256 -> keep 100% in registers across block_sum!
    float ssc = 0.0f, ssm = 0.0f;
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
                const float w = s_prob[s];
                ax = fmaf(w, sv.x, ax); ay = fmaf(w, sv.y, ay); az = fmaf(w, sv.z, az); aw = fmaf(w, sv.w, aw);
            }
            const float4 cv = make_float4(ax, ay, az, aw);
            float4 mv;
            if constexpr (BF16_ST) {
                const uint2 u = reinterpret_cast<const uint2*>(reinterpret_cast<const __nv_bfloat16*>(msg) + (size_t)m * D)[i];
                const float2 m0 = bf16x2_to_f32x2(u.x), m1 = bf16x2_to_f32x2(u.y);
                mv = make_float4(m0.x, m0.y, m1.x, m1.y);
                reinterpret_cast<uint2*>(reinterpret_cast<__nv_bfloat16*>(moe_out) + (size_t)m * D)[i] =
                    make_uint2(f32x2_to_bf16x2(ax, ay), f32x2_to_bf16x2(az, aw));
            } else {
                mv = reinterpret_cast<const float4*>(msg + (size_t)m * D)[i];
                reinterpret_cast<float4*>(moe_out + (size_t)m * D)[i] = cv;
            }
            mo_reg[t] = cv;
            mp_reg[t] = mv;
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
            const uint16_t c0 = f32x2_to_fp8x2(cv.x * ic, cv.y * ic);
            const uint16_t c1 = f32x2_to_fp8x2(cv.z * ic, cv.w * ic);
            const uint16_t m0 = f32x2_to_fp8x2(gv.x * im, gv.y * im);
            const uint16_t m1 = f32x2_to_fp8x2(gv.z * im, gv.w * im);
            d0[i] = (uint32_t)c0 | ((uint32_t)c1 << 16);
            d1[i] = (uint32_t)m0 | ((uint32_t)m1 << 16);
        }
    }
}

// Direct BF16-GF Blend + RMSNorm + optional fused amax capture (supports BF16 cur/msg for 2x less GDDR7!)
template <bool BF16_CUR_MSG = false>
__global__ void __launch_bounds__(256) k_ink_ctrl_blend_norm_fused_gf_bf16(
    const float* __restrict__ cur,
    const float* __restrict__ msg,
    const __nv_bfloat16* __restrict__ gf_bf16, // [M, 2*D] BF16
    const float* __restrict__ norm_w,
    float* __restrict__ out,
    int M, int D,
    float* __restrict__ d_amax_out = nullptr) {
    const int m = blockIdx.x;
    if (m >= M) return;
    const int tid = threadIdx.x;
    const int D4 = D >> 2;
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
            float4 cv, mv;
            if constexpr (BF16_CUR_MSG) {
                const uint2 cu = reinterpret_cast<const uint2*>(reinterpret_cast<const __nv_bfloat16*>(cur) + (size_t)m * D)[i];
                const uint2 mu = reinterpret_cast<const uint2*>(reinterpret_cast<const __nv_bfloat16*>(msg) + (size_t)m * D)[i];
                const float2 c0 = bf16x2_to_f32x2(cu.x), c1 = bf16x2_to_f32x2(cu.y);
                const float2 m0 = bf16x2_to_f32x2(mu.x), m1 = bf16x2_to_f32x2(mu.y);
                cv = make_float4(c0.x, c0.y, c1.x, c1.y);
                mv = make_float4(m0.x, m0.y, m1.x, m1.y);
            } else {
                cv = reinterpret_cast<const float4*>(cur + (size_t)m * D)[i];
                mv = reinterpret_cast<const float4*>(msg + (size_t)m * D)[i];
            }
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

__global__ void __launch_bounds__(256) k_bf16_to_f32_fast(
    const __nv_bfloat16* __restrict__ src, float* __restrict__ dst, size_t n) {
    const size_t n2 = n >> 1;
    const uint32_t* s2 = reinterpret_cast<const uint32_t*>(src);
    float2* d2 = reinterpret_cast<float2*>(dst);
    for (size_t i = (size_t)blockIdx.x * 256 + threadIdx.x; i < n2; i += (size_t)gridDim.x * 256) {
        d2[i] = bf16x2_to_f32x2(s2[i]);
    }
}

// ============================================================================
// Small-M (M <= 3) High-Wave Multi-Projection FP8 GEMV (4 rows/CTA, 64 threads, per-proj X[5]!)
// Fuses ALL 5 QKV+R+Gate projections into 1 launch on ALL 48 visits (even looped layers!) & doubles wo_ud wave count!
// ============================================================================
struct InkMultiProjFp8 {
    const uint8_t* W[5];
    const float*   sc[5];   // device scalar [1] per projection
    const float*   X[5];    // optional per-projection input [M, K] (if nullptr, uses fallback X)
    float*         Y[5];
    int            N[5];
    int            cumN[6];
    int            count;
};

// Zero-Smem L1-Direct M=1 FP8 Multi-Proj GEMV (ZERO __syncthreads, ZERO redundant s_x conversions!)
// Zero-Smem L1-Direct M=1 FP8 Multi-Proj GEMV (256 threads = 8 warps = 8 rows/CTA, 32 lanes/row = 512B coalesced!)
// Issues upfront register-pipelined uint4 weight loads (4..6 in flight) with ZERO shared memory and ZERO barriers!
__global__ void __launch_bounds__(256) k_ink_gemv_multi_fp8_m1_l1(
    InkMultiProjFp8 mp,
    const __half2* __restrict__ Xh_fallback,  // [K/2]
    const __half2* __restrict__ Xh_k,         // [K/2] (for pi == 3)
    const __half2* __restrict__ Xh_v,         // [K/2] (for pi == 4)
    int K) {
    const int tid   = threadIdx.x;
    const int warp  = tid >> 5;    // 0..7 rows/CTA
    const int lane  = tid & 31;    // 0..31 lanes/row (512B coalesced!)
    const int grow0 = blockIdx.x * 8;
    const int grow  = grow0 + warp;
    if (grow >= mp.cumN[mp.count]) { INK_PDL_WAIT(); return; }   // PERF-STEP: the wait is on every path

    int pi = 0;
    while (pi + 1 < mp.count && grow0 >= mp.cumN[pi + 1]) ++pi;
    const __half2* Xh = (pi == 3 && Xh_k) ? Xh_k : ((pi == 4 && Xh_v) ? Xh_v : Xh_fallback);
    const int lrow = grow - mp.cumN[pi];
    const uint4* wr = reinterpret_cast<const uint4*>(mp.W[pi] + (size_t)lrow * K);
    const uint4* x4 = reinterpret_cast<const uint4*>(Xh);
    const float w_sc = *mp.sc[pi];
    const int k16_end = K >> 4;  // 144 (K=2304) or 192 (K=3072)

    // PERF-STEP: the weight row is a constant, so its loads are issued BEFORE the wait for the kernel in front (programmatic dependent launch): they run while
    // that kernel still finishes. Only the activations X (written by the kernel before) are read after INK_PDL_WAIT(). Same loads, same order of accumulation.
    uint4 w0 = make_uint4(0u, 0u, 0u, 0u), w1 = w0, w2 = w0, w3 = w0, w4 = w0, w5 = w0;
    bool has4 = false;
    if (k16_end == 144) {
        w0 = __ldg(wr + lane + 0);
        w1 = __ldg(wr + lane + 32);
        w2 = __ldg(wr + lane + 64);
        w3 = __ldg(wr + lane + 96);
        has4 = (lane < 16);
        if (has4) w4 = __ldg(wr + lane + 128);
    } else if (k16_end == 192) {
        w0 = __ldg(wr + lane + 0);
        w1 = __ldg(wr + lane + 32);
        w2 = __ldg(wr + lane + 64);
        w3 = __ldg(wr + lane + 96);
        w4 = __ldg(wr + lane + 128);
        w5 = __ldg(wr + lane + 160);
    }
    INK_PDL_WAIT();

    __half2 acc0 = __float2half2_rn(0.0f);
    __half2 acc1 = __float2half2_rn(0.0f);
    auto accum_u4 = [&](const uint4& wv, int idx16) {
        const uint4 xa = __ldg(x4 + idx16 * 2 + 0);
        const uint4 xb = __ldg(x4 + idx16 * 2 + 1);
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
        // K = 2304: 4 full 512B loads upfront + 16-lane tail (staged above, before the wait)
        accum_u4(w0, lane + 0);
        accum_u4(w1, lane + 32);
        accum_u4(w2, lane + 64);
        accum_u4(w3, lane + 96);
        if (has4) accum_u4(w4, lane + 128);
    } else if (k16_end == 192) {
        // K = 3072: 6 exact 512B loads (staged above, before the wait)
        accum_u4(w0, lane + 0);
        accum_u4(w1, lane + 32);
        accum_u4(w2, lane + 64);
        accum_u4(w3, lane + 96);
        accum_u4(w4, lane + 128);
        accum_u4(w5, lane + 160);
    } else {
        for (int k16 = lane; k16 < k16_end; k16 += 32) {
            accum_u4(__ldg(wr + k16), k16);
        }
    }
    const float2 f0 = __half22float2(acc0);
    const float2 f1 = __half22float2(acc1);
    float s = (f0.x + f0.y) + (f1.x + f1.y);
#pragma unroll
    for (int o = 16; o >= 1; o >>= 1) s += __shfl_xor_sync(0xFFFFFFFFu, s, o);
    if (lane == 0) mp.Y[pi][lrow] = s * w_sc;
}

// Zero-Smem L1-Direct Small-M (M=2..4) FP8 Multi-Proj GEMV (256 threads = 8 warps = 8 rows/CTA, 32 lanes/row!)
// Pre-converts X[M, K] -> __half2 once, then reads each weight row ONCE across all M tokens with ZERO barriers!
template <int MAX_M>
__global__ void __launch_bounds__(256) k_ink_gemv_multi_fp8_smallM_l1(
    InkMultiProjFp8 mp,
    const __half2* __restrict__ Xh_q,  // [M, K/2]
    const __half2* __restrict__ Xh_k,  // [M, K/2] (nullable -> Xh_q)
    const __half2* __restrict__ Xh_v,  // [M, K/2] (nullable -> Xh_q)
    int M, int K) {
    INK_PDL_WAIT();   // PERF-STEP: first statement on every path (programmatic dependent launch)
    const int tid   = threadIdx.x;
    const int warp  = tid >> 5;
    const int lane  = tid & 31;
    const int grow0 = blockIdx.x * 8;
    const int grow  = grow0 + warp;
    if (grow >= mp.cumN[mp.count]) return;

    int pi = 0;
    while (pi + 1 < mp.count && grow0 >= mp.cumN[pi + 1]) ++pi;
    const __half2* Xh = (pi == 3 && Xh_k) ? Xh_k : ((pi == 4 && Xh_v) ? Xh_v : Xh_q);
    const int lrow = grow - mp.cumN[pi];
    const uint4* wr = reinterpret_cast<const uint4*>(mp.W[pi] + (size_t)lrow * K);
    const float w_sc = *mp.sc[pi];
    const int k16_end = K >> 4;
    const int k2 = K >> 1;

    __half2 acc0[MAX_M], acc1[MAX_M];
#pragma unroll
    for (int m = 0; m < MAX_M; ++m) {
        acc0[m] = __float2half2_rn(0.0f);
        acc1[m] = __float2half2_rn(0.0f);
    }
    for (int k16 = lane; k16 < k16_end; k16 += 32) {
        const uint4 wv = __ldg(wr + k16);
        const __half2 w0 = fp8x2_to_h2((uint16_t)(wv.x & 0xFFFFu));
        const __half2 w1 = fp8x2_to_h2((uint16_t)(wv.x >> 16));
        const __half2 w2 = fp8x2_to_h2((uint16_t)(wv.y & 0xFFFFu));
        const __half2 w3 = fp8x2_to_h2((uint16_t)(wv.y >> 16));
        const __half2 w4 = fp8x2_to_h2((uint16_t)(wv.z & 0xFFFFu));
        const __half2 w5 = fp8x2_to_h2((uint16_t)(wv.z >> 16));
        const __half2 w6 = fp8x2_to_h2((uint16_t)(wv.w & 0xFFFFu));
        const __half2 w7 = fp8x2_to_h2((uint16_t)(wv.w >> 16));
#pragma unroll
        for (int m = 0; m < MAX_M; ++m) {
            if (m < M) {
                const uint4* x4 = reinterpret_cast<const uint4*>(Xh + (size_t)m * k2);
                const uint4 xa = __ldg(x4 + k16 * 2 + 0);
                const uint4 xb = __ldg(x4 + k16 * 2 + 1);
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
    float* Yp = mp.Y[pi];
    const int Np = mp.N[pi];
#pragma unroll
    for (int m = 0; m < MAX_M; ++m) {
        if (m < M) {
            const float2 f0 = __half22float2(acc0[m]);
            const float2 f1 = __half22float2(acc1[m]);
            float s = (f0.x + f0.y) + (f1.x + f1.y);
#pragma unroll
            for (int o = 16; o >= 1; o >>= 1) s += __shfl_xor_sync(0xFFFFFFFFu, s, o);
            if (lane == 0) Yp[(size_t)m * Np + lrow] = s * w_sc;
        }
    }
}

// Fused F32 Tiny-N Projection Kernel (attn_sink_proj [24,2304] + optional k_sconv/v_sconv weight_proj [32,2304] each!)
// Grid: dim3(has_kv_dyn ? 88 : 24, (M + TILE_M - 1) / TILE_M), Block: 256 -> eliminates 3-CTA stall + 72 launches!
template <int TILE_M>
__global__ void __launch_bounds__(256) k_ink_f32_sink_sconv_fused(
    const float* __restrict__ w_sink,  // [24, 2304]
    const float* __restrict__ w_kd,    // [32, 2304] (nullable)
    const float* __restrict__ w_vd,    // [32, 2304] (nullable)
    const float* __restrict__ in_q,    // [M, 2304]
    const float* __restrict__ in_k,    // [M, 2304]
    const float* __restrict__ in_v,    // [M, 2304]
    float* __restrict__ y_sink,        // [M, 24]
    float* __restrict__ y_kd,          // [M, 32]
    float* __restrict__ y_vd,          // [M, 32]
    int M) {
    INK_PDL_WAIT();   // PERF-STEP: first statement on every path (programmatic dependent launch)
    const int row = blockIdx.x;
    const int m0  = blockIdx.y * TILE_M;
    const int tid = threadIdx.x;
    const float* W = nullptr;
    const float* X = nullptr;
    float* Y = nullptr;
    int lrow = 0, N = 0;
    if (row < kInkHeads) {
        W = w_sink; X = in_q; Y = y_sink; lrow = row; N = kInkHeads;
    } else if (row < kInkHeads + kInkKVHeads * 4) {
        if (!w_kd) return;
        W = w_kd; X = in_k; Y = y_kd; lrow = row - kInkHeads; N = kInkKVHeads * 4;
    } else {
        if (!w_vd) return;
        W = w_vd; X = in_v; Y = y_vd; lrow = row - (kInkHeads + kInkKVHeads * 4); N = kInkKVHeads * 4;
    }
    constexpr int D4 = kInkHidden / 4; // 576
    const float4* wr = reinterpret_cast<const float4*>(W + (size_t)lrow * kInkHidden);
    float acc[TILE_M] = {};
    for (int i = tid; i < D4; i += 256) {
        const float4 wv = __ldg(wr + i);
#pragma unroll
        for (int tm = 0; tm < TILE_M; ++tm) {
            const int m = m0 + tm;
            if (m < M) {
                const float4 xv = reinterpret_cast<const float4*>(X + (size_t)m * kInkHidden)[i];
                acc[tm] = fmaf(wv.x, xv.x, acc[tm]);
                acc[tm] = fmaf(wv.y, xv.y, acc[tm]);
                acc[tm] = fmaf(wv.z, xv.z, acc[tm]);
                acc[tm] = fmaf(wv.w, xv.w, acc[tm]);
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

// L2-hot Split + Per-Tensor Rescale for Packed FP8 QKV+R+Gate GEMM (M = 5..64 -> 1 cublasLt call instead of 5!)
__global__ void __launch_bounds__(256) k_ink_split_rescale_qkv(
    const float* __restrict__ src,     // [M, N_tot] (6528 or 8576)
    float* __restrict__ y_q,           // [M, 3072]
    float* __restrict__ y_r,           // [M, 384]
    float* __restrict__ y_g,           // [M, 3072]
    float* __restrict__ y_k,           // [M, 1024] (nullable)
    float* __restrict__ y_v,           // [M, 1024] (nullable)
    const float* __restrict__ sc0, const float* __restrict__ sc1, const float* __restrict__ sc2,
    const float* __restrict__ sc3, const float* __restrict__ sc4,
    int M, int N_tot) {
    const int m = blockIdx.y;
    const int c = blockIdx.x * 256 + threadIdx.x;
    if (m >= M || c >= N_tot) return;
    constexpr int N0 = kInkQDim;
    constexpr int N1 = N0 + kInkHeads * kInkDRel;
    constexpr int N2 = N1 + kInkQDim;
    constexpr int N3 = N2 + kInkKVDim;
    const float v = src[(size_t)m * N_tot + c];
    if (c < N0) {
        y_q[(size_t)m * kInkQDim + c] = v * (*sc0);
    } else if (c < N1) {
        y_r[(size_t)m * (kInkHeads * kInkDRel) + (c - N0)] = v * (*sc1);
    } else if (c < N2) {
        y_g[(size_t)m * kInkQDim + (c - N1)] = v * (*sc2);
    } else if (c < N3) {
        if (y_k) y_k[(size_t)m * kInkKVDim + (c - N2)] = v * (*sc3);
    } else {
        if (y_v) y_v[(size_t)m * kInkKVDim + (c - N3)] = v * (*sc4);
    }
}

// Vectorized BF16-IO Split + Per-Tensor Rescale for Packed FP8 Prefill QKV+R+Gate GEMM (2 BF16s/thread -> 1,700 GB/s!)
__global__ void __launch_bounds__(256) k_ink_split_rescale_qkv_bf16(
    const __nv_bfloat16* __restrict__ src,  // [M, N_tot] BF16 (6528 or 8576)
    __nv_bfloat16* __restrict__ y_q,        // [M, 3072] BF16
    __nv_bfloat16* __restrict__ y_r,        // [M, 384]  BF16
    __nv_bfloat16* __restrict__ y_g,        // [M, 3072] BF16
    __nv_bfloat16* __restrict__ y_k,        // [M, 1024] BF16 (nullable)
    __nv_bfloat16* __restrict__ y_v,        // [M, 1024] BF16 (nullable)
    const float* __restrict__ sc0, const float* __restrict__ sc1, const float* __restrict__ sc2,
    const float* __restrict__ sc3, const float* __restrict__ sc4,
    int M, int N_tot) {
    const int m  = blockIdx.y;
    const int c2 = blockIdx.x * 256 + threadIdx.x;
    const int N2_tot = N_tot >> 1;
    if (m >= M || c2 >= N2_tot) return;
    constexpr int N0 = kInkQDim >> 1;
    constexpr int N1 = N0 + ((kInkHeads * kInkDRel) >> 1);
    constexpr int N2 = N1 + (kInkQDim >> 1);
    constexpr int N3 = N2 + (kInkKVDim >> 1);
    const uint32_t raw = reinterpret_cast<const uint32_t*>(src)[(size_t)m * N2_tot + c2];
    const float2 f = bf16x2_to_f32x2(raw);
    if (c2 < N0) {
        const float sc = *sc0;
        reinterpret_cast<uint32_t*>(y_q)[(size_t)m * (kInkQDim >> 1) + c2] = f32x2_to_bf16x2(f.x * sc, f.y * sc);
    } else if (c2 < N1) {
        const float sc = *sc1;
        reinterpret_cast<uint32_t*>(y_r)[(size_t)m * ((kInkHeads * kInkDRel) >> 1) + (c2 - N0)] = f32x2_to_bf16x2(f.x * sc, f.y * sc);
    } else if (c2 < N2) {
        const float sc = *sc2;
        reinterpret_cast<uint32_t*>(y_g)[(size_t)m * (kInkQDim >> 1) + (c2 - N1)] = f32x2_to_bf16x2(f.x * sc, f.y * sc);
    } else if (c2 < N3) {
        if (y_k) {
            const float sc = *sc3;
            reinterpret_cast<uint32_t*>(y_k)[(size_t)m * (kInkKVDim >> 1) + (c2 - N2)] = f32x2_to_bf16x2(f.x * sc, f.y * sc);
        }
    } else {
        if (y_v) {
            const float sc = *sc4;
            reinterpret_cast<uint32_t*>(y_v)[(size_t)m * (kInkKVDim >> 1) + (c2 - N3)] = f32x2_to_bf16x2(f.x * sc, f.y * sc);
        }
    }
}

template <int MAX_M>
__global__ void __launch_bounds__(64) k_ink_gemv_multi_fp8(
    InkMultiProjFp8 mp,
    const float* __restrict__ X_fallback,  // [M, K]
    int M, int K) {
    const int tid  = threadIdx.x; // 0..63
    const int sub  = tid >> 4;    // 0..3 output rows/CTA
    const int lg   = tid & 15;    // 0..15 lanes/row (16 bytes/lane = 256B coalesced!)
    const int grow0 = blockIdx.x * 4;
    const int grow  = grow0 + sub;

    // Every Inkling projection N[i] (3072, 384, 1024, 2304) is a multiple of 4 -> pi is uniform across the entire CTA!
    int pi = 0;
    const bool valid = (grow < mp.cumN[mp.count]);
    if (grow0 < mp.cumN[mp.count]) {
        while (pi + 1 < mp.count && grow0 >= mp.cumN[pi + 1]) ++pi;
    }
    const float* X = (mp.X[pi] != nullptr) ? mp.X[pi] : X_fallback;
    const int lrow = valid ? (grow - mp.cumN[pi]) : 0;
    const uint4* wr = valid ? reinterpret_cast<const uint4*>(mp.W[pi] + (size_t)lrow * K) : nullptr;
    const float w_sc = valid ? (*mp.sc[pi]) : 0.0f;

    float acc[MAX_M] = {};
    constexpr int KT = (MAX_M == 1) ? 2304 : ((MAX_M <= 4) ? 1024 : 512);
    __shared__ alignas(16) __half2 s_x[MAX_M][KT / 2];

    for (int k0 = 0; k0 < K; k0 += KT) {
        const int kl = min(KT, K - k0);
        const int kl2 = kl >> 1;
        for (int m = 0; m < M; ++m) {
            const float2* sx = reinterpret_cast<const float2*>(X + (size_t)m * K + k0);
            __half2* dx = s_x[m];
            for (int i = tid; i < kl2; i += 64) {
                const float2 f = sx[i];
                dx[i] = __floats2half2_rn(fminf(fmaxf(f.x, -65504.0f), 65504.0f),
                                          fminf(fmaxf(f.y, -65504.0f), 65504.0f));
            }
        }
        __syncthreads();

        if (valid) {
            const int kl16 = kl >> 4;
            const int base16 = k0 >> 4;
            for (int k16 = lg; k16 < kl16; k16 += 16) {
                const uint4 wv = __ldg(wr + base16 + k16);
                const __half2 w0 = fp8x2_to_h2((uint16_t)(wv.x & 0xFFFFu));
                const __half2 w1 = fp8x2_to_h2((uint16_t)(wv.x >> 16));
                const __half2 w2 = fp8x2_to_h2((uint16_t)(wv.y & 0xFFFFu));
                const __half2 w3 = fp8x2_to_h2((uint16_t)(wv.y >> 16));
                const __half2 w4 = fp8x2_to_h2((uint16_t)(wv.z & 0xFFFFu));
                const __half2 w5 = fp8x2_to_h2((uint16_t)(wv.z >> 16));
                const __half2 w6 = fp8x2_to_h2((uint16_t)(wv.w & 0xFFFFu));
                const __half2 w7 = fp8x2_to_h2((uint16_t)(wv.w >> 16));
#pragma unroll
                for (int m = 0; m < MAX_M; ++m) {
                    if (m < M) {
                        const __half2* xp = s_x[m] + k16 * 8;
                        __half2 p0 = __hmul2(w0, xp[0]);
                        __half2 p1 = __hmul2(w1, xp[1]);
                        p0 = __hfma2(w2, xp[2], p0);
                        p1 = __hfma2(w3, xp[3], p1);
                        p0 = __hfma2(w4, xp[4], p0);
                        p1 = __hfma2(w5, xp[5], p1);
                        p0 = __hfma2(w6, xp[6], p0);
                        p1 = __hfma2(w7, xp[7], p1);
                        const float2 f0 = __half22float2(p0);
                        const float2 f1 = __half22float2(p1);
                        acc[m] += (f0.x + f0.y) + (f1.x + f1.y);
                    }
                }
            }
        }
        if (k0 + KT < K) __syncthreads();
    }

    if (valid) {
        float* Yp = mp.Y[pi];
        const int Np = mp.N[pi];
#pragma unroll
        for (int m = 0; m < MAX_M; ++m) {
            if (m < M) {
                float s = acc[m];
#pragma unroll
                for (int o = 8; o >= 1; o >>= 1) s += __shfl_xor_sync(0xFFFFFFFFu, s, o);
                if (lg == 0) Yp[(size_t)m * Np + lrow] = s * w_sc;
            }
        }
    }
}

// ============================================================================
// Small-M (M <= 24) FP8 Loop Controllers (4x less bandwidth vs F32: 340 MB -> 85 MB!)
// Vectorized __half2 __hfma2 down-proj + single-barrier warp reduction across all M!
// ============================================================================
template <int MAX_M>
__global__ void __launch_bounds__(256) k_ink_ctrl_3x_down_fp8(
    const float* __restrict__ cat3,     // [3, M, 4608]
    const uint8_t* __restrict__ wd0, const float* __restrict__ sd0,
    const uint8_t* __restrict__ wd1, const float* __restrict__ sd1,
    const uint8_t* __restrict__ wd2, const float* __restrict__ sd2,
    float* __restrict__ gate3,          // [3, M, 64]
    int M) {
    INK_PDL_WAIT();   // PERF-STEP: first statement on every path (programmatic dependent launch)
    const int r = blockIdx.x;   // 0..63
    const int c = blockIdx.y;   // 0..2
    if (r >= kInkGateRank || c >= 3) return;
    constexpr int D2 = 2 * kInkHidden;  // 4608
    constexpr int D2_16 = D2 / 16;      // 288
    const int tid = threadIdx.x;
    const int warp = tid >> 5;
    const int lane = tid & 31;
    const uint8_t* wd = (c == 0) ? wd0 : (c == 1 ? wd1 : wd2);
    const float w_sc  = *((c == 0) ? sd0 : (c == 1 ? sd1 : sd2));
    const float* cat  = cat3 + (size_t)c * M * D2;
    const uint4* wr   = reinterpret_cast<const uint4*>(wd + (size_t)r * D2);
    __half2 acc0[MAX_M], acc1[MAX_M];
#pragma unroll
    for (int m = 0; m < MAX_M; ++m) {
        acc0[m] = __float2half2_rn(0.0f);
        acc1[m] = __float2half2_rn(0.0f);
    }
    for (int i = tid; i < D2_16; i += 256) {
        const uint4 wv = __ldg(wr + i);
        const __half2 w0 = fp8x2_to_h2((uint16_t)(wv.x & 0xFFFFu));
        const __half2 w1 = fp8x2_to_h2((uint16_t)(wv.x >> 16));
        const __half2 w2 = fp8x2_to_h2((uint16_t)(wv.y & 0xFFFFu));
        const __half2 w3 = fp8x2_to_h2((uint16_t)(wv.y >> 16));
        const __half2 w4 = fp8x2_to_h2((uint16_t)(wv.z & 0xFFFFu));
        const __half2 w5 = fp8x2_to_h2((uint16_t)(wv.z >> 16));
        const __half2 w6 = fp8x2_to_h2((uint16_t)(wv.w & 0xFFFFu));
        const __half2 w7 = fp8x2_to_h2((uint16_t)(wv.w >> 16));
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
__global__ void __launch_bounds__(256) k_ink_ctrl_3x_apply_fp8(
    const float* __restrict__ cur0, const float* __restrict__ cur1, const float* __restrict__ cur2,
    const float* __restrict__ msgs,    // [4, M, D]
    const float* __restrict__ gate3,   // [3, M, 64]
    const uint8_t* __restrict__ wgf0, const float* __restrict__ sgf0,
    const uint8_t* __restrict__ wgf1, const float* __restrict__ sgf1,
    const uint8_t* __restrict__ wgf2, const float* __restrict__ sgf2,
    float* __restrict__ out0, float* __restrict__ out1, float* __restrict__ out2,
    int M) {
    INK_PDL_WAIT();   // PERF-STEP: first statement on every path (programmatic dependent launch)
    constexpr int D = kInkHidden;
    const int c = blockIdx.y;  // 0..2
    if (c >= 3) return;
    __shared__ float s_gate[MAX_M][kInkGateRank];
    const int tid = threadIdx.x;
    const float* gp = gate3 + (size_t)c * M * kInkGateRank;
    for (int i = tid; i < M * kInkGateRank; i += 256) s_gate[i >> 6][i & 63] = gp[i];
    __syncthreads();

    const int sub = tid >> 4;  // 0..15 rows
    const int lg  = tid & 15;  // 0..15 -> 4 FP8 elements/lane
    const int d   = blockIdx.x * 16 + sub;
    if (d >= D) return;

    const uint8_t* wgf = (c == 0) ? wgf0 : (c == 1 ? wgf1 : wgf2);
    const float w_sc   = *((c == 0) ? sgf0 : (c == 1 ? sgf1 : sgf2));
    const float* cur   = (c == 0) ? cur0 : (c == 1 ? cur1 : cur2);
    const float* msg   = msgs + (size_t)c * M * D;
    float* out         = (c == 0) ? out0 : (c == 1 ? out1 : out2);

    const uint32_t wg4 = *reinterpret_cast<const uint32_t*>(wgf + (size_t)d * kInkGateRank + lg * 4);
    const uint32_t wf4 = *reinterpret_cast<const uint32_t*>(wgf + (size_t)(D + d) * kInkGateRank + lg * 4);
    const float2 g0 = __half22float2(fp8x2_to_h2((uint16_t)(wg4 & 0xFFFFu)));
    const float2 g1 = __half22float2(fp8x2_to_h2((uint16_t)(wg4 >> 16)));
    const float2 f0 = __half22float2(fp8x2_to_h2((uint16_t)(wf4 & 0xFFFFu)));
    const float2 f1 = __half22float2(fp8x2_to_h2((uint16_t)(wf4 >> 16)));

#pragma unroll
    for (int m = 0; m < MAX_M; ++m) {
        if (m < M) {
            const float4 gv = reinterpret_cast<const float4*>(s_gate[m])[lg];
            float ag = g0.x * gv.x + g0.y * gv.y + g1.x * gv.z + g1.y * gv.w;
            float af = f0.x * gv.x + f0.y * gv.y + f1.x * gv.z + f1.y * gv.w;
#pragma unroll
            for (int o = 8; o >= 1; o >>= 1) {
                ag += __shfl_xor_sync(0xFFFFFFFFu, ag, o);
                af += __shfl_xor_sync(0xFFFFFFFFu, af, o);
            }
            if (lg == 0) {
                const size_t idx = (size_t)m * D + d;
                out[idx] = (1.0f - tanhf(af * w_sc)) * cur[idx] + tanhf(ag * w_sc) * msg[idx];
            }
        }
    }
}

template <int MAX_M>
__global__ void __launch_bounds__(256) k_ink_ctrl_1x_down_fp8(
    const float* __restrict__ cat,      // [M, 4608]
    const uint8_t* __restrict__ wd, const float* __restrict__ sd,
    float* __restrict__ gate,           // [M, 64]
    int M) {
    INK_PDL_WAIT();   // PERF-STEP: first statement on every path (programmatic dependent launch)
    const int r = blockIdx.x;
    if (r >= kInkGateRank) return;
    constexpr int D2 = 2 * kInkHidden;
    constexpr int D2_16 = D2 / 16;
    const int tid = threadIdx.x;
    const int warp = tid >> 5;
    const int lane = tid & 31;
    const float w_sc = *sd;
    const uint4* wr = reinterpret_cast<const uint4*>(wd + (size_t)r * D2);
    __half2 acc0[MAX_M], acc1[MAX_M];
#pragma unroll
    for (int m = 0; m < MAX_M; ++m) {
        acc0[m] = __float2half2_rn(0.0f);
        acc1[m] = __float2half2_rn(0.0f);
    }
    for (int i = tid; i < D2_16; i += 256) {
        const uint4 wv = __ldg(wr + i);
        const __half2 w0 = fp8x2_to_h2((uint16_t)(wv.x & 0xFFFFu));
        const __half2 w1 = fp8x2_to_h2((uint16_t)(wv.x >> 16));
        const __half2 w2 = fp8x2_to_h2((uint16_t)(wv.y & 0xFFFFu));
        const __half2 w3 = fp8x2_to_h2((uint16_t)(wv.y >> 16));
        const __half2 w4 = fp8x2_to_h2((uint16_t)(wv.z & 0xFFFFu));
        const __half2 w5 = fp8x2_to_h2((uint16_t)(wv.z >> 16));
        const __half2 w6 = fp8x2_to_h2((uint16_t)(wv.w & 0xFFFFu));
        const __half2 w7 = fp8x2_to_h2((uint16_t)(wv.w >> 16));
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

// Fused 3x RMSNorm + __half2 dual output (eliminates 108 k_ink_f32_to_h2 launches/step for M <= 8!)
__global__ void __launch_bounds__(256) k_ink_rms_norm_3x_h2(
    float* __restrict__ x0, float* __restrict__ x1, float* __restrict__ x2,
    const float* __restrict__ w0, const float* __restrict__ w1, const float* __restrict__ w2,
    __half2* __restrict__ h2_out,  // [3, M, D/2] contiguous!
    int M, int D) {
    INK_PDL_WAIT();   // PERF-STEP: first statement on every path (programmatic dependent launch)
    const int m = blockIdx.x;
    const int c = blockIdx.y;
    if (m >= M || c >= 3) return;
    float* xp = ((c == 0) ? x0 : (c == 1 ? x1 : x2)) + (size_t)m * D;
    const float* wp = (c == 0) ? w0 : (c == 1 ? w1 : w2);
    __half2* hp = h2_out + ((size_t)c * M + m) * (D >> 1);
    const int tid = threadIdx.x;
    const int D4 = D >> 2;
    float4* x4 = reinterpret_cast<float4*>(xp);
    const float4* w4 = reinterpret_cast<const float4*>(wp);
    __shared__ float sh[32];
    float4 r[3];
    float ss = 0.0f;
#pragma unroll
    for (int t = 0; t < 3; ++t) {
        const int i = tid + t * 256;
        if (i < D4) {
            const float4 v = x4[i];
            r[t] = v;
            ss += v.x * v.x + v.y * v.y + v.z * v.z + v.w * v.w;
        }
    }
    const float inv = rsqrtf(block_sum(ss, sh) / (float)D + kInkRmsEps);
#pragma unroll
    for (int t = 0; t < 3; ++t) {
        const int i = tid + t * 256;
        if (i < D4) {
            const float4 v = r[t], wv = w4[i];
            const float a = v.x * inv * wv.x, b = v.y * inv * wv.y;
            const float cc = v.z * inv * wv.z, d = v.w * inv * wv.w;
            x4[i] = make_float4(a, b, cc, d);
            hp[2 * i + 0] = __floats2half2_rn(fminf(fmaxf(a, -65504.0f), 65504.0f), fminf(fmaxf(b, -65504.0f), 65504.0f));
            hp[2 * i + 1] = __floats2half2_rn(fminf(fmaxf(cc, -65504.0f), 65504.0f), fminf(fmaxf(d, -65504.0f), 65504.0f));
        }
    }
}

template <int MAX_M>
__global__ void __launch_bounds__(256) k_ink_ctrl_1x_apply_fp8(
    const float* __restrict__ cur,
    const float* __restrict__ msg,
    const float* __restrict__ gate,     // [M, 64]
    const uint8_t* __restrict__ wgf, const float* __restrict__ sgf,
    float* __restrict__ out,
    int M) {
    INK_PDL_WAIT();   // PERF-STEP: first statement on every path (programmatic dependent launch)
    constexpr int D = kInkHidden;
    __shared__ float s_gate[MAX_M][kInkGateRank];
    const int tid = threadIdx.x;
    for (int i = tid; i < M * kInkGateRank; i += 256) s_gate[i >> 6][i & 63] = gate[i];
    __syncthreads();
    const int sub = tid >> 4;
    const int lg  = tid & 15;
    const int d   = blockIdx.x * 16 + sub;
    if (d >= D) return;
    const float w_sc = *sgf;
    const uint32_t wg4 = *reinterpret_cast<const uint32_t*>(wgf + (size_t)d * kInkGateRank + lg * 4);
    const uint32_t wf4 = *reinterpret_cast<const uint32_t*>(wgf + (size_t)(D + d) * kInkGateRank + lg * 4);
    const float2 g0 = __half22float2(fp8x2_to_h2((uint16_t)(wg4 & 0xFFFFu)));
    const float2 g1 = __half22float2(fp8x2_to_h2((uint16_t)(wg4 >> 16)));
    const float2 f0 = __half22float2(fp8x2_to_h2((uint16_t)(wf4 & 0xFFFFu)));
    const float2 f1 = __half22float2(fp8x2_to_h2((uint16_t)(wf4 >> 16)));
#pragma unroll
    for (int m = 0; m < MAX_M; ++m) {
        if (m < M) {
            const float4 gv = reinterpret_cast<const float4*>(s_gate[m])[lg];
            float ag = g0.x * gv.x + g0.y * gv.y + g1.x * gv.z + g1.y * gv.w;
            float af = f0.x * gv.x + f0.y * gv.y + f1.x * gv.z + f1.y * gv.w;
#pragma unroll
            for (int o = 8; o >= 1; o >>= 1) {
                ag += __shfl_xor_sync(0xFFFFFFFFu, ag, o);
                af += __shfl_xor_sync(0xFFFFFFFFu, af, o);
            }
            if (lg == 0) {
                const size_t idx = (size_t)m * D + d;
                out[idx] = (1.0f - tanhf(af * w_sc)) * cur[idx] + tanhf(ag * w_sc) * msg[idx];
            }
        }
    }
}

// ============================================================================
// Dynamic cublasLt FP8 Engine (ZERO CMakeLists.txt changes!) + Runtime FP8 Trunk/Controller Cache
// ============================================================================
struct Rtx5090Fp8Engine {
    // Dynamic cublasLt function pointers
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

    fn_create_t     p_create = nullptr;
    fn_destroy_t    p_destroy = nullptr;
    fn_md_create_t  p_md_create = nullptr;
    fn_md_destroy_t p_md_destroy = nullptr;
    fn_md_setattr_t p_md_setattr = nullptr;
    fn_ml_create_t  p_ml_create = nullptr;
    fn_ml_destroy_t p_ml_destroy = nullptr;
    fn_pr_create_t  p_pr_create = nullptr;
    fn_pr_destroy_t p_pr_destroy = nullptr;
    fn_pr_setattr_t p_pr_setattr = nullptr;
    fn_get_heur_t   p_get_heur = nullptr;
    fn_matmul_t     p_matmul = nullptr;

    cublasLtHandle_t lt = nullptr;
    bool lt_ok = false;
    bool cache_ready = false;

    void*   d_ws = nullptr;
    size_t  ws_bytes = 32ull << 20;
    uint8_t* d_x_fp8 = nullptr;   // [max_batch, 4608]
    uint8_t* d_cat3_fp8 = nullptr;// [3, max_batch, 4608] pre-computed 3x pre-attn controller FP8 inputs!
    float*   d_x_amax = nullptr;  // [1]
    float*   d_x_sc = nullptr;    // [1]
    float*   d_sc_inv8 = nullptr; // [1] constant 0.125f for RMS-normalized CatRMS -> FP8!
    float*   d_sc_one = nullptr;  // [1] constant 1.0f for packed multi-scale FP8 QKV GEMM!
    float*   d_inp_amax = nullptr;// [4] fused amax for d_inp0/1/2 + attn_out
    __nv_bfloat16* d_y_bf16 = nullptr; // optional BF16 output fallback [max_batch, 4608]
    int max_m = 0;

    // Pre-quantized FP8 trunk weights:
    // Per layer ip (0..23): 0=wq(3072), 1=wr(384), 2=wgate(3072), 3=wk(1024), 4=wv(1024), 5=wo(2304x3072)
    uint8_t* d_w_pool = nullptr;
    float*   d_sc_pool = nullptr;
    uint8_t* w_attn[24][6] = {};
    float*   sc_attn[24][6] = {};
    // Per core layer ci (0..11), controller k (0=attn, 1=k, 2=v, 3=mlp): down [64, 4608], gf [4608, 64]
    uint8_t* w_ctrl_dn[12][4] = {};
    float*   sc_ctrl_dn[12][4] = {};
    uint8_t* w_ctrl_gf[12][4] = {};
    float*   sc_ctrl_gf[12][4] = {};
    // FP16 router gate weights [24, 128, 2304] (~13.5 MiB total -> 2x faster small-M router!)
    __half2* d_router_h2_pool = nullptr;
    __half2* w_router_h2[24]  = {};

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

    void quant_bf16_tensor(const __nv_bfloat16* src, size_t n, uint8_t* dst, float* d_sc, cudaStream_t st) {
        INK_CK(cudaMemsetAsync(d_x_amax, 0, 4, st));
        const unsigned grid = (unsigned)std::min<size_t>(512, std::max<size_t>(1, (n / 2 + 255) / 256));
        k_fp8_absmax_bf16<<<grid, 256, 0, st>>>(src, n, d_x_amax);
        k_fp8_finalize_scale<<<1, 1, 0, st>>>(d_x_amax, d_sc);
        k_fp8_quant_bf16<<<grid, 256, 0, st>>>(src, n, d_sc, dst);
    }

    void quant_q8row_tensor(const uint8_t* src, int rows, int cols, uint8_t* dst, float* d_sc, cudaStream_t st) {
        INK_CK(cudaMemsetAsync(d_x_amax, 0, 4, st));
        const size_t n4 = ((size_t)rows * cols) >> 2;
        const unsigned grid = (unsigned)std::min<size_t>(512, std::max<size_t>(1, (n4 + 255) / 256));
        k_fp8_absmax_q8row<<<grid, 256, 0, st>>>(src, rows, cols, d_x_amax);
        k_fp8_finalize_scale<<<1, 1, 0, st>>>(d_x_amax, d_sc);
        k_fp8_quant_q8row<<<grid, 256, 0, st>>>(src, rows, cols, d_sc, dst);
    }

    void quant_f32_tensor(const float* src, size_t n, uint8_t* dst, float* d_sc, cudaStream_t st) {
        INK_CK(cudaMemsetAsync(d_x_amax, 0, 4, st));
        const unsigned grid = (unsigned)std::min<size_t>(512, std::max<size_t>(1, (n / 4 + 255) / 256));
        k_fp8_absmax_f32<<<grid, 256, 0, st>>>(src, n, d_x_amax);
        k_fp8_quant_f32_padded<<<grid, 256, 0, st>>>(src, n, n, d_x_amax, d_sc, dst);
    }

    template <typename ModelT>
    void init(const ModelT& m, int max_batch, cudaStream_t st, size_t later_bytes = 0) {
        if (!fp8_trunk_on() || m.ls_active) return;
        size_t vf = 0, vt = 0;
        INK_CK(ink_cuda_mem_info(&vf, &vt, m.primary_gpu));
        // Total FP8 weight pool: 24 * (8576*2304 + 2304*3072) + 12 * 4 * (64*4608 + 4608*64) = 672 MB
        constexpr int kRDim = kInkHeads * kInkDRel; // 384
        constexpr size_t kLayerBytes =
            (size_t)(kInkQDim + kRDim + kInkQDim + 2 * kInkKVDim) * kInkHidden +
            (size_t)kInkHidden * kInkQDim;
        const int n_core = m.man.c.core_end - m.man.c.core_start;
        constexpr size_t kCtrlDnBytes = (size_t)kInkGateRank * (2 * kInkHidden);
        constexpr size_t kCtrlGfBytes = (size_t)(2 * kInkHidden) * kInkGateRank;
        const size_t total_w = (size_t)m.man.c.n_phys * kLayerBytes + (size_t)n_core * 4 * (kCtrlDnBytes + kCtrlGfBytes);
        // later_bytes: what the drafter (DFlash 2) and the crash margin still take after the runner; a card whose free memory cannot carry both the copy and them runs without the copy (the old path of the trunk)
        const size_t need_b = total_w + ws_bytes + (size_t)max_batch * 4608 * 6 + (512ull << 20) + later_bytes;
        if (vf < need_b) return;

        static const char* kAttnNames[6] = {
            ".attn.wq_du.weight", ".attn.wr_du.weight", ".attn.gate_attn.weight",
            ".attn.wk_dv.weight", ".attn.wv_dv.weight", ".attn.wo_ud.weight"
        };
        const int kAttnRows[6] = {kInkQDim, kRDim, kInkQDim, kInkKVDim, kInkKVDim, kInkHidden};
        const int kAttnCols[6] = {kInkHidden, kInkHidden, kInkHidden, kInkHidden, kInkHidden, kInkQDim};

        // Verify all attention weights are BF16 or Q8ROW (SQ-20 uses Q8ROW for wq/wk/wv/wo/gate_attn and BF16 for wr_du!)
        for (int ip = 0; ip < m.man.c.n_phys; ++ip) {
            const std::string p = "model.llm.layers." + std::to_string(ip);
            for (int k = 0; k < 6; ++k) {
                const uint8_t f = m.tinfo(p + kAttnNames[k]).fmt;
                if (f != FMT_BF16 && f != FMT_Q8ROW) return;
            }
        }

        max_m = ((max_batch + 31) & ~31);
        d_x_amax   = dalloc<float>(1);
        d_x_sc     = dalloc<float>(1);
        d_sc_inv8  = dalloc<float>(1);
        d_sc_one   = dalloc<float>(1);
        d_inp_amax = dalloc<float>(4);
        const float h_inv8 = 0.125f, h_one = 1.0f;
        INK_CK(cudaMemcpyAsync(d_sc_inv8, &h_inv8, sizeof(float), cudaMemcpyHostToDevice, st));
        INK_CK(cudaMemcpyAsync(d_sc_one,  &h_one,  sizeof(float), cudaMemcpyHostToDevice, st));
        d_x_fp8    = dalloc<uint8_t>((size_t)max_m * (2 * kInkHidden));
        d_cat3_fp8 = dalloc<uint8_t>(3ull * (size_t)max_m * (2 * kInkHidden));
        d_y_bf16 = dalloc<__nv_bfloat16>((size_t)max_m * (2 * kInkHidden));
        d_w_pool = dalloc<uint8_t>(total_w);
        const int n_sc = m.man.c.n_phys * 6 + n_core * 8;
        d_sc_pool = dalloc<float>(n_sc);

        size_t w_off = 0;
        int sc_i = 0;
        for (int ip = 0; ip < m.man.c.n_phys; ++ip) {
            const std::string p = "model.llm.layers." + std::to_string(ip);
            for (int k = 0; k < 6; ++k) {
                const std::string tn = p + kAttnNames[k];
                const auto& ti = m.tinfo(tn);
                const size_t elems = (size_t)kAttnRows[k] * kAttnCols[k];
                w_attn[ip][k]  = d_w_pool + w_off;
                sc_attn[ip][k] = d_sc_pool + sc_i++;
                if (ti.fmt == FMT_BF16) {
                    quant_bf16_tensor(reinterpret_cast<const __nv_bfloat16*>(m.trunk_ptr(tn)),
                                      elems, w_attn[ip][k], sc_attn[ip][k], st);
                } else {
                    quant_q8row_tensor(m.trunk_ptr(tn), kAttnRows[k], kAttnCols[k],
                                       w_attn[ip][k], sc_attn[ip][k], st);
                }
                w_off += elems;
            }
        }

        // Controllers: pack [w_gain; w_forget] temporarily into d_y_bf16 (F32 alias) or quantize directly
        float* tmp_gf = dalloc<float>(kCtrlGfBytes);
        static const char* kCN[4] = {"attn", "k", "v", "mlp"};
        for (int ci = 0; ci < n_core && ci < 12; ++ci) {
            const std::string p = "model.llm.layers." + std::to_string(m.man.c.core_start + ci);
            for (int k = 0; k < 4; ++k) {
                const std::string cp = p + ".controllers." + kCN[k];
                w_ctrl_dn[ci][k]  = d_w_pool + w_off;
                sc_ctrl_dn[ci][k] = d_sc_pool + sc_i++;
                quant_f32_tensor(reinterpret_cast<const float*>(m.trunk_ptr(cp + ".down_weight")),
                                 kCtrlDnBytes, w_ctrl_dn[ci][k], sc_ctrl_dn[ci][k], st);
                w_off += kCtrlDnBytes;

                const size_t half_gf = kCtrlGfBytes / 2;
                INK_CK(cudaMemcpyAsync(tmp_gf,           m.trunk_ptr(cp + ".gain_weight"),   half_gf * 4, cudaMemcpyDeviceToDevice, st));
                INK_CK(cudaMemcpyAsync(tmp_gf + half_gf, m.trunk_ptr(cp + ".forget_weight"), half_gf * 4, cudaMemcpyDeviceToDevice, st));
                w_ctrl_gf[ci][k]  = d_w_pool + w_off;
                sc_ctrl_gf[ci][k] = d_sc_pool + sc_i++;
                quant_f32_tensor(tmp_gf, kCtrlGfBytes, w_ctrl_gf[ci][k], sc_ctrl_gf[ci][k], st);
                w_off += kCtrlGfBytes;
            }
        }
        INK_CK(cudaStreamSynchronize(st));
        cudaFree(tmp_gf);

        // Pre-convert 24 router gate matrices [128, 2304] F32 -> __half2 (~13.5 MiB -> 2x less BW at M<=16!)
        {
            constexpr size_t kRPairs = (size_t)kInkExperts * (kInkHidden / 2);
            d_router_h2_pool = dalloc<__half2>((size_t)m.man.c.n_phys * kRPairs);
            for (int ip = 0; ip < m.man.c.n_phys; ++ip) {
                w_router_h2[ip] = d_router_h2_pool + (size_t)ip * kRPairs;
                const float* gw = (const float*)m.trunk_ptr("model.llm.layers." + std::to_string(ip) + ".mlp.gate.weight");
                k_ink_f32_to_h2<<<(kRPairs + 255) / 256, 256, 0, st>>>(gw, w_router_h2[ip], kRPairs);
            }
            INK_CK(cudaStreamSynchronize(st));
        }

        // Load cublasLt dynamically for 838 TFLOPS FP8 prefill & C64 GEMMs!
        if (load_lt_symbols() && p_create(&lt) == CUBLAS_STATUS_SUCCESS) {
            d_ws = dalloc<uint8_t>(ws_bytes);
            lt_ok = true;
            // Pre-warm plans for CUDA-Graph captured batched & tree shapes so AlgoGetHeuristic never runs inside stream capture!
            static constexpr int kWarmM[] = {32, 64, 128, 192, 256};
            static constexpr int kWarmNK[][2] = {
                {kInkQDim, kInkHidden}, {kInkHeads * kInkDRel, kInkHidden},
                {kInkKVDim, kInkHidden}, {kInkHidden, kInkQDim},
                {kInkQDim + kInkHeads * kInkDRel + kInkQDim, kInkHidden},
                {kInkQDim + kInkHeads * kInkDRel + kInkQDim + 2 * kInkKVDim, kInkHidden},
                {kInkGateRank, 2 * kInkHidden}, {2 * kInkHidden, kInkGateRank}
            };
            for (int wm : kWarmM) {
                if (wm > max_m) continue;
                for (const auto& nk : kWarmNK) (void)get_plan(wm, nk[0], nk[1]);
            }
            if (max_m >= 4096) {
                (void)get_plan(4096, kInkQDim + kInkHeads * kInkDRel + kInkQDim, kInkHidden, true);
                (void)get_plan(4096, kInkQDim + kInkHeads * kInkDRel + kInkQDim + 2 * kInkKVDim, kInkHidden, true);
            }
        }
        cache_ready = true;
        std::fprintf(stderr, "[InkRTX5090] fp8_cache=1 lt_ok=%d pool=%.1f MiB max_m=%d\n",
                     lt_ok ? 1 : 0, (double)total_w / (1024.0 * 1024.0), max_m);
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
            // Bind dummy non-null scale pointers for heuristic query (updated per call before matmul)
            p_md_setattr(md, CUBLASLT_MATMUL_DESC_A_SCALE_POINTER, &d_sc_pool, sizeof(d_sc_pool));
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
            pe.md = md; pe.la = la; pe.lb = lb; pe.lc = lc;
            pe.algo = res.algo; pe.use_bf16_out = bf16_out; pe.valid = true;
            return true;
        };

        if (force_bf16) {
            try_build(true);
        } else {
            // Prefer direct F32 output (zero extra copy!); fall back to BF16 output if needed
            if (!try_build(false)) try_build(true);
        }
        auto ins = plans.emplace(key, pe);
        return ins.first->second.valid ? &ins.first->second : nullptr;
    }

    void quantize_x(const float* X, int M, int M_pad, int K, cudaStream_t st) {
        const size_t nv = (size_t)M * K;
        const size_t np = (size_t)M_pad * K;
        if (np <= 16384) {
            k_fp8_quant_f32_1cta<<<1, 256, 0, st>>>(X, nv, np, d_x_sc, d_x_fp8);
            return;
        }
        if (np <= 81920) {
            k_fp8_quant_f32_1cta_1024<<<1, 1024, 0, st>>>(X, nv, np, d_x_sc, d_x_fp8);
            return;
        }
        INK_CK(cudaMemsetAsync(d_x_amax, 0, 4, st));
        const unsigned g1 = (unsigned)std::min<size_t>(512, std::max<size_t>(1, (nv / 4 + 255) / 256));
        const unsigned g2 = (unsigned)std::min<size_t>(512, std::max<size_t>(1, (np / 4 + 255) / 256));
        k_fp8_absmax_f32<<<g1, 256, 0, st>>>(X, nv, d_x_amax);
        k_fp8_quant_f32_padded<<<g2, 256, 0, st>>>(X, nv, np, d_x_amax, d_x_sc, d_x_fp8);
    }

    void quantize_x_bf16(const __nv_bfloat16* X, int M, int M_pad, int K, cudaStream_t st) {
        const size_t nv = (size_t)M * K;
        const size_t np = (size_t)M_pad * K;
        if (np <= 16384) {
            k_fp8_quant_bf16_1cta<<<1, 256, 0, st>>>(X, nv, np, d_x_sc, d_x_fp8);
            return;
        }
        INK_CK(cudaMemsetAsync(d_x_amax, 0, 4, st));
        const unsigned g1 = (unsigned)std::min<size_t>(512, std::max<size_t>(1, (nv / 2 + 255) / 256));
        const unsigned g2 = (unsigned)std::min<size_t>(512, std::max<size_t>(1, (np / 2 + 255) / 256));
        k_fp8_absmax_bf16<<<g1, 256, 0, st>>>(X, nv, d_x_amax);
        k_fp8_quant_bf16_padded<<<g2, 256, 0, st>>>(X, nv, np, d_x_amax, d_x_sc, d_x_fp8);
    }

    // Zero-F32-Intermediate Loop Controller + RMSNorm for M > 16 (Prefill & C64)!
    // Eliminates d_ctrl_cat F32 write+read+amax AND k_bf16_to_f32_fast gf_out write+read (~60 GB/chunk saved!).
    bool run_ctrl_fp8(int ci, int ck, const float* cur, const float* msg, const float* norm_w,
                      float* out, float* d_gate_f32, float* d_gf_f32_scratch,
                      int M, bool x_fp8_ready, cudaStream_t st,
                      const uint8_t* x_fp8_override = nullptr,
                      float* d_amax_out = nullptr,
                      bool bf16_cur_msg = false) {
        if (!cache_ready || !lt_ok || ci < 0 || ci >= 12 || ck < 0 || ck >= 4) return false;
        const int M_pad = (M <= 32) ? 32 : ((M + 31) & ~31);
        if (M_pad > max_m) return false;
        constexpr int D = kInkHidden;
        constexpr int D2 = 2 * D;
        const PlanEntry* pe1 = get_plan(M_pad, kInkGateRank, D2);
        const PlanEntry* pe2 = get_plan(M_pad, D2, kInkGateRank, /*force_bf16=*/bf16_cur_msg);
        if (!pe1 || !pe2) return false;

        const uint8_t* x1_ptr = x_fp8_override ? x_fp8_override : d_x_fp8;
        if (!x_fp8_ready && !x_fp8_override) {
            k_ink_ctrl_cat_rms_to_fp8<<<M_pad, 256, 0, st>>>(cur, msg, d_x_fp8, M, M_pad, D);
        }
        const float alpha = 1.0f, beta = 0.0f;
        // 1. gate[M, 64] = cat_fp8[M_pad, 4608] @ w_ctrl_dn^T
        const float* sd1 = sc_ctrl_dn[ci][ck];
        p_md_setattr(pe1->md, CUBLASLT_MATMUL_DESC_A_SCALE_POINTER, &sd1, sizeof(sd1));
        p_md_setattr(pe1->md, CUBLASLT_MATMUL_DESC_B_SCALE_POINTER, &d_sc_inv8, sizeof(d_sc_inv8));
        if (pe1->use_bf16_out) {
            if (p_matmul(lt, pe1->md, &alpha, w_ctrl_dn[ci][ck], pe1->la, x1_ptr, pe1->lb,
                         &beta, d_y_bf16, pe1->lc, d_y_bf16, pe1->lc, &pe1->algo, d_ws, ws_bytes, st) != CUBLAS_STATUS_SUCCESS) {
                return false;
            }
            quantize_x_bf16(d_y_bf16, M, M_pad, kInkGateRank, st);
        } else {
            if (p_matmul(lt, pe1->md, &alpha, w_ctrl_dn[ci][ck], pe1->la, x1_ptr, pe1->lb,
                         &beta, d_gate_f32, pe1->lc, d_gate_f32, pe1->lc, &pe1->algo, d_ws, ws_bytes, st) != CUBLAS_STATUS_SUCCESS) {
                return false;
            }
            quantize_x(d_gate_f32, M, M_pad, kInkGateRank, st);
        }

        // 2. gf[M, 4608] = gate_fp8[M_pad, 64] @ w_ctrl_gf^T -> direct blend+norm (ZERO F32 gf_out!)
        const float* sd2 = sc_ctrl_gf[ci][ck];
        p_md_setattr(pe2->md, CUBLASLT_MATMUL_DESC_A_SCALE_POINTER, &sd2, sizeof(sd2));
        p_md_setattr(pe2->md, CUBLASLT_MATMUL_DESC_B_SCALE_POINTER, &d_x_sc, sizeof(d_x_sc));
        if (pe2->use_bf16_out) {
            if (p_matmul(lt, pe2->md, &alpha, w_ctrl_gf[ci][ck], pe2->la, d_x_fp8, pe2->lb,
                         &beta, d_y_bf16, pe2->lc, d_y_bf16, pe2->lc, &pe2->algo, d_ws, ws_bytes, st) != CUBLAS_STATUS_SUCCESS) {
                return false;
            }
            if (bf16_cur_msg) k_ink_ctrl_blend_norm_fused_gf_bf16<true><<<M, 256, 0, st>>>(cur, msg, d_y_bf16, norm_w, out, M, D, d_amax_out);
            else              k_ink_ctrl_blend_norm_fused_gf_bf16<false><<<M, 256, 0, st>>>(cur, msg, d_y_bf16, norm_w, out, M, D, d_amax_out);
        } else {
            if (p_matmul(lt, pe2->md, &alpha, w_ctrl_gf[ci][ck], pe2->la, d_x_fp8, pe2->lb,
                         &beta, d_gf_f32_scratch, pe2->lc, d_gf_f32_scratch, pe2->lc, &pe2->algo, d_ws, ws_bytes, st) != CUBLAS_STATUS_SUCCESS) {
                return false;
            }
            k_ink_ctrl_blend_norm_fused_gf<<<M, 256, 0, st>>>(cur, msg, d_gf_f32_scratch, norm_w, out, M, D, d_amax_out);
        }
        return true;
    }

    // Run Y[M_pad, N] (BF16 directly!) = X[M, K] @ W_fp8[N, K]^T via cublasLt FP8 TC (ZERO k_bf16_to_f32_fast + 2x less GDDR7!)
    bool gemm_fp8_tn_bf16(const uint8_t* W_fp8, const float* w_sc, const float* X, __nv_bfloat16* Y_bf16,
                          int M, int N, int K, bool reuse_x, cudaStream_t st,
                          const float* pre_amax = nullptr) {
        if (!cache_ready || !lt_ok || (N & 15) != 0 || (K & 15) != 0) return false;
        const int M_pad = (M <= 32) ? 32 : ((M + 31) & ~31);
        if (M_pad > max_m) return false;
        const PlanEntry* pe = get_plan(M_pad, N, K, /*force_bf16=*/true);
        if (!pe || !pe->use_bf16_out) return false;

        if (!reuse_x) {
            if (pre_amax) {
                const size_t nv = (size_t)M * K;
                const size_t np = (size_t)M_pad * K;
                const unsigned g2 = (unsigned)std::min<size_t>(512, std::max<size_t>(1, (np / 4 + 255) / 256));
                k_fp8_quant_f32_padded<<<g2, 256, 0, st>>>(X, nv, np, pre_amax, d_x_sc, d_x_fp8);
            } else {
                quantize_x(X, M, M_pad, K, st);
            }
        }

        p_md_setattr(pe->md, CUBLASLT_MATMUL_DESC_A_SCALE_POINTER, &w_sc, sizeof(w_sc));
        p_md_setattr(pe->md, CUBLASLT_MATMUL_DESC_B_SCALE_POINTER, &d_x_sc, sizeof(d_x_sc));
        const float alpha = 1.0f, beta = 0.0f;
        const cublasStatus_t cs = p_matmul(
            lt, pe->md, &alpha,
            W_fp8, pe->la, d_x_fp8, pe->lb,
            &beta, Y_bf16, pe->lc, Y_bf16, pe->lc,
            &pe->algo, d_ws, ws_bytes, st);
        return cs == CUBLAS_STATUS_SUCCESS;
    }

    // Packed single-GEMM Q+R+Gate(+K+V) for M = 5..256 (uses contiguous w_attn[ip][0..4] in d_w_pool -> ZERO extra VRAM!)
    bool gemm_fp8_packed_qkv(int ip, bool share_qkv, const float* in_q,
                             float* d_scratch_f32,
                             float* y_q, float* y_r, float* y_g, float* y_k, float* y_v,
                             int M, cudaStream_t st, const float* pre_amax = nullptr) {
        if (!cache_ready || !lt_ok || !d_sc_one || !d_scratch_f32) return false;
        const int M_pad = (M <= 32) ? 32 : ((M + 31) & ~31);
        if (M_pad > max_m) return false;
        constexpr int kRDim = kInkHeads * kInkDRel;
        const int N_tot = share_qkv ? (kInkQDim + kRDim + kInkQDim + 2 * kInkKVDim)
                                    : (kInkQDim + kRDim + kInkQDim);
        const PlanEntry* pe = get_plan(M_pad, N_tot, kInkHidden);
        if (!pe || pe->use_bf16_out) return false;

        if (pre_amax) {
            const size_t nv = (size_t)M * kInkHidden;
            const size_t np = (size_t)M_pad * kInkHidden;
            const unsigned g2 = (unsigned)std::min<size_t>(512, std::max<size_t>(1, (np / 4 + 255) / 256));
            k_fp8_quant_f32_padded<<<g2, 256, 0, st>>>(in_q, nv, np, pre_amax, d_x_sc, d_x_fp8);
        } else {
            quantize_x(in_q, M, M_pad, kInkHidden, st);
        }
        p_md_setattr(pe->md, CUBLASLT_MATMUL_DESC_A_SCALE_POINTER, &d_sc_one, sizeof(d_sc_one));
        p_md_setattr(pe->md, CUBLASLT_MATMUL_DESC_B_SCALE_POINTER, &d_x_sc, sizeof(d_x_sc));
        const float alpha = 1.0f, beta = 0.0f;
        const cublasStatus_t cs = p_matmul(
            lt, pe->md, &alpha,
            w_attn[ip][0], pe->la, d_x_fp8, pe->lb,
            &beta, d_scratch_f32, pe->lc, d_scratch_f32, pe->lc,
            &pe->algo, d_ws, ws_bytes, st);
        if (cs != CUBLAS_STATUS_SUCCESS) return false;
        k_ink_split_rescale_qkv<<<dim3((N_tot + 255) / 256, M), 256, 0, st>>>(
            d_scratch_f32, y_q, y_r, y_g, share_qkv ? y_k : nullptr, share_qkv ? y_v : nullptr,
            sc_attn[ip][0], sc_attn[ip][1], sc_attn[ip][2], sc_attn[ip][3], sc_attn[ip][4], M, N_tot);
        return true;
    }

    // Packed single-GEMM BF16-IO Q+R+Gate(+K+V) for Prefill (M >= 128, uses d_ctrl_cat as [M_pad, 8576] BF16 -> 0 extra VRAM!)
    bool gemm_fp8_packed_qkv_bf16(int ip, bool share_qkv, const float* in_q,
                                  __nv_bfloat16* d_scratch_bf16,
                                  __nv_bfloat16* y_q, __nv_bfloat16* y_r, __nv_bfloat16* y_g,
                                  __nv_bfloat16* y_k, __nv_bfloat16* y_v,
                                  int M, cudaStream_t st, const float* pre_amax = nullptr) {
        if (!cache_ready || !lt_ok || !d_sc_one || !d_scratch_bf16) return false;
        const int M_pad = (M <= 32) ? 32 : ((M + 31) & ~31);
        if (M_pad > max_m) return false;
        constexpr int kRDim = kInkHeads * kInkDRel;
        const int N_tot = share_qkv ? (kInkQDim + kRDim + kInkQDim + 2 * kInkKVDim)
                                    : (kInkQDim + kRDim + kInkQDim);
        const PlanEntry* pe = get_plan(M_pad, N_tot, kInkHidden, /*force_bf16=*/true);
        if (!pe || !pe->use_bf16_out) return false;

        if (pre_amax) {
            const size_t nv = (size_t)M * kInkHidden;
            const size_t np = (size_t)M_pad * kInkHidden;
            const unsigned g2 = (unsigned)std::min<size_t>(512, std::max<size_t>(1, (np / 4 + 255) / 256));
            k_fp8_quant_f32_padded<<<g2, 256, 0, st>>>(in_q, nv, np, pre_amax, d_x_sc, d_x_fp8);
        } else {
            quantize_x(in_q, M, M_pad, kInkHidden, st);
        }
        p_md_setattr(pe->md, CUBLASLT_MATMUL_DESC_A_SCALE_POINTER, &d_sc_one, sizeof(d_sc_one));
        p_md_setattr(pe->md, CUBLASLT_MATMUL_DESC_B_SCALE_POINTER, &d_x_sc, sizeof(d_x_sc));
        const float alpha = 1.0f, beta = 0.0f;
        const cublasStatus_t cs = p_matmul(
            lt, pe->md, &alpha,
            w_attn[ip][0], pe->la, d_x_fp8, pe->lb,
            &beta, d_scratch_bf16, pe->lc, d_scratch_bf16, pe->lc,
            &pe->algo, d_ws, ws_bytes, st);
        if (cs != CUBLAS_STATUS_SUCCESS) return false;
        const int N2 = N_tot >> 1;
        k_ink_split_rescale_qkv_bf16<<<dim3((N2 + 255) / 256, M), 256, 0, st>>>(
            d_scratch_bf16, y_q, y_r, y_g, share_qkv ? y_k : nullptr, share_qkv ? y_v : nullptr,
            sc_attn[ip][0], sc_attn[ip][1], sc_attn[ip][2], sc_attn[ip][3], sc_attn[ip][4], M, N_tot);
        return true;
    }

    // Run Y[M, N] (F32) = X[M, K] @ W_fp8[N, K]^T via cublasLt FP8 TC!
    bool gemm_fp8_tn(const uint8_t* W_fp8, const float* w_sc, const float* X, float* Y,
                     int M, int N, int K, bool reuse_x, cudaStream_t st,
                     const float* pre_amax = nullptr) {
        if (!cache_ready || !lt_ok || (N & 15) != 0 || (K & 15) != 0) return false;
        const int M_pad = (M <= 32) ? 32 : ((M + 31) & ~31);
        if (M_pad > max_m) return false;
        const PlanEntry* pe = get_plan(M_pad, N, K);
        if (!pe) return false;

        if (!reuse_x) {
            if (pre_amax) {
                const size_t nv = (size_t)M * K;
                const size_t np = (size_t)M_pad * K;
                const unsigned g2 = (unsigned)std::min<size_t>(512, std::max<size_t>(1, (np / 4 + 255) / 256));
                k_fp8_quant_f32_padded<<<g2, 256, 0, st>>>(X, nv, np, pre_amax, d_x_sc, d_x_fp8);
            } else {
                quantize_x(X, M, M_pad, K, st);
            }
        }

        p_md_setattr(pe->md, CUBLASLT_MATMUL_DESC_A_SCALE_POINTER, &w_sc, sizeof(w_sc));
        p_md_setattr(pe->md, CUBLASLT_MATMUL_DESC_B_SCALE_POINTER, &d_x_sc, sizeof(d_x_sc));
        const float alpha = 1.0f, beta = 0.0f;
        if (!pe->use_bf16_out) {
            const cublasStatus_t cs = p_matmul(
                lt, pe->md, &alpha,
                W_fp8, pe->la, d_x_fp8, pe->lb,
                &beta, Y, pe->lc, Y, pe->lc,
                &pe->algo, d_ws, ws_bytes, st);
            return cs == CUBLAS_STATUS_SUCCESS;
        } else {
            if ((size_t)M_pad * N > (size_t)max_m * (2 * kInkHidden)) return false;
            const cublasStatus_t cs = p_matmul(
                lt, pe->md, &alpha,
                W_fp8, pe->la, d_x_fp8, pe->lb,
                &beta, d_y_bf16, pe->lc, d_y_bf16, pe->lc,
                &pe->algo, d_ws, ws_bytes, st);
            if (cs != CUBLAS_STATUS_SUCCESS) return false;
            const size_t n_out = (size_t)M * N;
            const unsigned g = (unsigned)std::min<size_t>(1024, std::max<size_t>(1, (n_out / 2 + 255) / 256));
            k_bf16_to_f32_fast<<<g, 256, 0, st>>>(d_y_bf16, Y, n_out);
            return true;
        }
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
        auto fr = [](auto*& p) { if (p) { cudaFree(p); p = nullptr; } };
        fr(d_ws); fr(d_x_fp8); fr(d_cat3_fp8); fr(d_x_amax); fr(d_x_sc); fr(d_sc_inv8); fr(d_sc_one); fr(d_inp_amax); fr(d_y_bf16); fr(d_w_pool); fr(d_sc_pool); fr(d_router_h2_pool);
        lt_ok = false;
        cache_ready = false;
    }
};

}  // namespace strata::ink::rtx5090
