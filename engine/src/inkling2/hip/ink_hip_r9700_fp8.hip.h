// engine/src/inkling2/hip/ink_hip_r9700_fp8.hip.h
// Opt-in AMD Radeon AI PRO R9700 (gfx1201, 32 GB) FP8 WMMA Trunk, FP8 WMMA Grouped/Persistent MoE,
// and Exact-Mode Non-Temporal (MALL-Bypass) MoE Kernels.
// Strictly compiled with #if guards so gfx1030/gfx1100/gfx1151/gfx1200/gfx1201 multi-arch builds stay clean.
// ZERO impact on default path when STRATA_INK_R9700 is OFF.
// MIT License - Copyright (c) 2026 Niko1221 and the Strata contributors.
#pragma once

#include "ink_hip_codecs.hip.h"

#include <type_traits>

namespace strata::inkling2::hip::r9700 {

using strata::inkling2::hip::bf16_to_f32;
using strata::inkling2::hip::e4m3fn_to_f32;
using strata::inkling2::hip::f32_to_e4m3fn;
using strata::inkling2::hip::fdot2_f32;
using strata::inkling2::hip::fv8;
using strata::inkling2::hip::fv8_zero;
using strata::inkling2::hip::GpuExpertDesc;
using strata::inkling2::hip::half_t;
using strata::inkling2::hip::hv2;
using strata::inkling2::hip::iv2;
using strata::inkling2::hip::sf4_e2m1_f16;
using strata::inkling2::hip::situ_f32;
using strata::inkling2::hip::sync_lds;
using strata::inkling2::hip::ue8m0_to_f32;
using strata::inkling2::hip::wave32_sum;

// =================================================================================================
// 1. Hardware Primitives: Non-Temporal Loads (MALL Bypass) & gfx12 Native FP8 WMMA (382.8 TFLOP/s)
// =================================================================================================

__device__ __forceinline__ uint4 load_u128_nt(const void* p) {
#if defined(__HIP_DEVICE_COMPILE__) && INK_GFX12
    const uint64_t* q = reinterpret_cast<const uint64_t*>(p);
    const uint64_t a = __builtin_nontemporal_load(q + 0);
    const uint64_t b = __builtin_nontemporal_load(q + 1);
    return make_uint4(static_cast<uint32_t>(a), static_cast<uint32_t>(a >> 32),
                      static_cast<uint32_t>(b), static_cast<uint32_t>(b >> 32));
#else
    return *reinterpret_cast<const uint4*>(p);
#endif
}

__device__ __forceinline__ uint2 load_u64_nt(const void* p) {
#if defined(__HIP_DEVICE_COMPILE__) && INK_GFX12
    const uint64_t a = __builtin_nontemporal_load(reinterpret_cast<const uint64_t*>(p));
    return make_uint2(static_cast<uint32_t>(a), static_cast<uint32_t>(a >> 32));
#else
    return *reinterpret_cast<const uint2*>(p);
#endif
}

#if defined(__HIP_DEVICE_COMPILE__) && defined(__gfx1201__)
#define INK_R9700_DEV 1
#else
#define INK_R9700_DEV 0
#endif

#if INK_R9700_DEV && __has_builtin(__builtin_amdgcn_wmma_f32_16x16x16_fp8_fp8_w32_gfx12)
#define INK_R9700_HAS_FP8_WMMA 1
__device__ __forceinline__ iv2 wmma_ld_fp8_k16(const uint8_t* p, int hi) {
    const uint2 v = *reinterpret_cast<const uint2*>(p + hi * 8);
    return __builtin_bit_cast(iv2, v);
}
__device__ __forceinline__ fv8 wmma_fp8_f32(iv2 a, iv2 b, fv8 c) {
    return __builtin_amdgcn_wmma_f32_16x16x16_fp8_fp8_w32_gfx12(a, b, c);
}
#else
#define INK_R9700_HAS_FP8_WMMA 0
#endif

// 2-cycle hardware v_perm_b32 LUT: converts 8 packed E2M1 nibbles (uint32_t) -> 8 exact FP8_E4M3FN bytes!
// E2M1 {0, 0.5, 1, 1.5, 2, 3, 4, 6} -> E4M3FN {0x00, 0x30, 0x38, 0x3C, 0x40, 0x44, 0x48, 0x4C}, sign bit 0x80.
__device__ __forceinline__ void sf4_u32_to_fp8x8(uint32_t raw4, uint32_t& lo4, uint32_t& hi4) {
    const uint32_t r1 = raw4 >> 16;
#if defined(__HIP_DEVICE_COMPILE__) && INK_GFX12
    constexpr uint32_t kLoLut = 0x3C383000u;
    constexpr uint32_t kHiLut = 0x4C484440u;
    const uint32_t m0 = (raw4 & 7u) | (((raw4 >> 4) & 7u) << 8) | (((raw4 >> 8) & 7u) << 16) | (((raw4 >> 12) & 7u) << 24);
    const uint32_t s0 = ((raw4 & 0x8u) << 4) | ((raw4 & 0x80u) << 8) | ((raw4 & 0x800u) << 12) | ((raw4 & 0x8000u) << 16);
    const uint32_t m1 = (r1 & 7u) | (((r1 >> 4) & 7u) << 8) | (((r1 >> 8) & 7u) << 16) | (((r1 >> 12) & 7u) << 24);
    const uint32_t s1 = ((r1 & 0x8u) << 4) | ((r1 & 0x80u) << 8) | ((r1 & 0x800u) << 12) | ((r1 & 0x8000u) << 16);
    lo4 = __builtin_amdgcn_perm(kHiLut, kLoLut, m0) | s0;
    hi4 = __builtin_amdgcn_perm(kHiLut, kLoLut, m1) | s1;
#else
    auto lut_byte = [](uint32_t nib) -> uint32_t {
        static constexpr uint8_t kT[8] = {0x00, 0x30, 0x38, 0x3C, 0x40, 0x44, 0x48, 0x4C};
        return static_cast<uint32_t>(kT[nib & 7u]) | ((nib & 8u) << 4);
    };
    lo4 = lut_byte(raw4 & 15u) | (lut_byte((raw4 >> 4) & 15u) << 8) |
          (lut_byte((raw4 >> 8) & 15u) << 16) | (lut_byte((raw4 >> 12) & 15u) << 24);
    hi4 = lut_byte(r1 & 15u) | (lut_byte((r1 >> 4) & 15u) << 8) |
          (lut_byte((r1 >> 8) & 15u) << 16) | (lut_byte((r1 >> 12) & 15u) << 24);
#endif
}

__device__ __forceinline__ uint32_t pack4_u8(uint8_t a, uint8_t b, uint8_t c, uint8_t d) {
    return static_cast<uint32_t>(a) | (static_cast<uint32_t>(b) << 8) |
           (static_cast<uint32_t>(c) << 16) | (static_cast<uint32_t>(d) << 24);
}

// =================================================================================================
// 2. EXACT-Mode Non-Temporal Direct MoE (100% Bit-Identical Arithmetic to k_moe_direct_*, MALL Bypass!)
//    Cold 19.4 GB expert reads stream via non-temporal loads so 64 MiB MALL retains hot trunk + KV!
// =================================================================================================

__global__ void __launch_bounds__(256) k_r9700_moe_direct_w13_situ_nt(
    const GpuExpertDesc* __restrict__ exps,
    const int32_t* __restrict__ topk_ids,
    const half_t* __restrict__ X_f16,
    half_t* __restrict__ act_out,
    int B, int top_k, int D, int inter, float beta, float lb) {
#if INK_R9700_DEV
    const int slot_k = blockIdx.y;
    if (slot_k >= B * top_k) return;
    const int b = slot_k / top_k;
    const int eid = topk_ids[slot_k];
    const GpuExpertDesc desc = exps[eid];
    if (!desc.body) return;

    const int tid = threadIdx.x;
    const int wave = tid >> 5;
    const int lane = tid & 31;
    const int j = blockIdx.x * 4 + (wave >> 1);
    const int row = (j << 1) | (wave & 1);
    const half_t* xrow = X_f16 + static_cast<size_t>(b) * D;
    float acc = 0.0f;

    __shared__ alignas(16) half_t sX[2304];
    __shared__ half_t s_e4m3[256];
    __shared__ uint32_t s_sf4_pair[256];
    __shared__ float s_sf4_sc[256];
    const bool use_sx = (D <= 2304 && (D & 7) == 0);
    if (use_sx) {
        const int d8 = D >> 3;
        const uint4* x8 = reinterpret_cast<const uint4*>(xrow);
        uint4* sx8 = reinterpret_cast<uint4*>(sX);
        for (int i = tid; i < d8; i += 256) sx8[i] = x8[i];
    }
    if (desc.gu_code == CODE_MX8) {
        s_e4m3[tid] = static_cast<half_t>(e4m3fn_to_f32(static_cast<uint8_t>(tid)));
    } else if (desc.gu_code == CODE_SF4) {
        const hv2 hp = {sf4_e2m1_f16(static_cast<uint32_t>(tid) & 15u), sf4_e2m1_f16(static_cast<uint32_t>(tid) >> 4)};
        s_sf4_pair[tid] = __builtin_bit_cast(uint32_t, hp);
        s_sf4_sc[tid] = e4m3fn_to_f32(static_cast<uint8_t>(tid));
    }
    __syncthreads();
    if (j >= inter) return;
    const half_t* xsrc = use_sx ? sX : xrow;

    if (desc.gu_code == CODE_SF4) {
        const float* hdr = reinterpret_cast<const float*>(desc.body);
        const float outer = hdr[0] / hdr[1];
        const uint8_t* packed = desc.body + 16;
        const uint8_t* scales = packed + static_cast<size_t>(2 * inter) * (D >> 1);
        const uint8_t* pr = packed + static_cast<size_t>(row) * (D >> 1);
        const uint8_t* sr = scales + static_cast<size_t>(row) * (D >> 4);
        const int nblk = D >> 4;
        for (int blk = lane; blk < nblk; blk += 32) {
            const float sc = s_sf4_sc[sr[blk]] * outer;
            const uint2 raw = load_u64_nt(pr + blk * 8);
            const uint4 x0 = *reinterpret_cast<const uint4*>(xsrc + (blk << 4));
            const uint4 x1 = *reinterpret_cast<const uint4*>(xsrc + (blk << 4) + 8);
            float d = fdot2_f32(s_sf4_pair[raw.x & 0xFFu], x0.x, 0.0f);
            d = fdot2_f32(s_sf4_pair[(raw.x >> 8) & 0xFFu], x0.y, d);
            d = fdot2_f32(s_sf4_pair[(raw.x >> 16) & 0xFFu], x0.z, d);
            d = fdot2_f32(s_sf4_pair[(raw.x >> 24) & 0xFFu], x0.w, d);
            d = fdot2_f32(s_sf4_pair[raw.y & 0xFFu], x1.x, d);
            d = fdot2_f32(s_sf4_pair[(raw.y >> 8) & 0xFFu], x1.y, d);
            d = fdot2_f32(s_sf4_pair[(raw.y >> 16) & 0xFFu], x1.z, d);
            d = fdot2_f32(s_sf4_pair[(raw.y >> 24) & 0xFFu], x1.w, d);
            acc = fmaf(d, sc, acc);
        }
    } else if (desc.gu_code == CODE_MX8) {
        const uint8_t* W = desc.body;
        const uint8_t* S = W + static_cast<size_t>(2 * inter) * D;
        const uint8_t* wr = W + static_cast<size_t>(row) * D;
        const uint8_t* sr = S + static_cast<size_t>(row) * (D >> 5);
        const int nblk = D >> 5;
        for (int blk = lane; blk < nblk; blk += 32) {
            const float sc = ue8m0_to_f32(sr[blk]);
            const uint8_t* wb = wr + (blk << 5);
            const uint4* xp = reinterpret_cast<const uint4*>(xsrc + (blk << 5));
            const uint4 wv0 = load_u128_nt(wb);
            const uint4 wv1 = load_u128_nt(wb + 16);
            const uint32_t ww[8] = {wv0.x, wv0.y, wv0.z, wv0.w, wv1.x, wv1.y, wv1.z, wv1.w};
            float d = 0.0f;
#pragma unroll
            for (int q = 0; q < 4; ++q) {
                const uint4 xv = xp[q];
                const uint32_t a = ww[q * 2 + 0];
                const uint32_t c = ww[q * 2 + 1];
                const hv2 h0 = {s_e4m3[a & 0xFFu], s_e4m3[(a >> 8) & 0xFFu]};
                const hv2 h1 = {s_e4m3[(a >> 16) & 0xFFu], s_e4m3[(a >> 24) & 0xFFu]};
                const hv2 h2 = {s_e4m3[c & 0xFFu], s_e4m3[(c >> 8) & 0xFFu]};
                const hv2 h3 = {s_e4m3[(c >> 16) & 0xFFu], s_e4m3[(c >> 24) & 0xFFu]};
                d = fdot2_f32(__builtin_bit_cast(uint32_t, h0), xv.x, d);
                d = fdot2_f32(__builtin_bit_cast(uint32_t, h1), xv.y, d);
                d = fdot2_f32(__builtin_bit_cast(uint32_t, h2), xv.z, d);
                d = fdot2_f32(__builtin_bit_cast(uint32_t, h3), xv.w, d);
            }
            acc = fmaf(d, sc, acc);
        }
    }

    const float val = wave32_sum(acc);
    __shared__ float s_gu[8];
    if (lane == 0) s_gu[wave] = val;
    __syncthreads();
    if (lane == 0 && (wave & 1) == 0) {
        const float g = s_gu[wave];
        const float u = s_gu[wave + 1];
        const float v = situ_f32(g, u, beta, lb);
        act_out[static_cast<size_t>(slot_k) * inter + j] = static_cast<half_t>(v);
    }
#else
    (void)exps; (void)topk_ids; (void)X_f16; (void)act_out; (void)B; (void)top_k; (void)D; (void)inter; (void)beta; (void)lb;
#endif
}

__global__ void __launch_bounds__(256) k_r9700_moe_direct_w2_par_nt(
    const GpuExpertDesc* __restrict__ exps,
    const int32_t* __restrict__ topk_ids,
    const half_t* __restrict__ act_in,
    float* __restrict__ y_part,
    int B, int top_k, int D, int inter) {
#if INK_R9700_DEV
    const int slot_k = blockIdx.y;
    if (slot_k >= B * top_k) return;
    const int eid = topk_ids[slot_k];
    const GpuExpertDesc desc = exps[eid];
    const int tid = threadIdx.x;
    const int wave = tid >> 5;
    const int lane = tid & 31;
    const int row = blockIdx.x * 8 + wave;
    if (!desc.body) {
        if (lane == 0 && row < D) y_part[static_cast<size_t>(slot_k) * D + row] = 0.0f;
        return;
    }

    const half_t* arow = act_in + static_cast<size_t>(slot_k) * inter;
    __shared__ alignas(16) half_t sA[1024];
    __shared__ half_t s_e4m3[256];
    __shared__ uint32_t s_sf4_pair[256];
    __shared__ float s_sf4_sc[256];
    const bool use_sa = (inter <= 1024 && (inter & 7) == 0);
    if (use_sa) {
        const int i8 = inter >> 3;
        const uint4* a8 = reinterpret_cast<const uint4*>(arow);
        uint4* sa8 = reinterpret_cast<uint4*>(sA);
        for (int i = tid; i < i8; i += 256) sa8[i] = a8[i];
    }
    if (desc.dn_code == CODE_MX8) {
        s_e4m3[tid] = static_cast<half_t>(e4m3fn_to_f32(static_cast<uint8_t>(tid)));
    } else if (desc.dn_code == CODE_SF4) {
        const hv2 hp = {sf4_e2m1_f16(static_cast<uint32_t>(tid) & 15u), sf4_e2m1_f16(static_cast<uint32_t>(tid) >> 4)};
        s_sf4_pair[tid] = __builtin_bit_cast(uint32_t, hp);
        s_sf4_sc[tid] = e4m3fn_to_f32(static_cast<uint8_t>(tid));
    }
    __syncthreads();
    if (row >= D) return;
    const half_t* asrc = use_sa ? sA : arow;
    float acc = 0.0f;

    if (desc.dn_code == CODE_SF4) {
        const float* hdr = reinterpret_cast<const float*>(desc.body);
        const float outer = hdr[2] / hdr[3];
        const uint8_t* packed = desc.body + desc.dn_offset;
        const uint8_t* scales = packed + static_cast<size_t>(D) * (inter >> 1);
        const uint8_t* pr = packed + static_cast<size_t>(row) * (inter >> 1);
        const uint8_t* sr = scales + static_cast<size_t>(row) * (inter >> 4);
        const int nblk = inter >> 4;
        for (int blk = lane; blk < nblk; blk += 32) {
            const float sc = s_sf4_sc[sr[blk]] * outer;
            const uint2 raw = load_u64_nt(pr + blk * 8);
            const uint4 x0 = *reinterpret_cast<const uint4*>(asrc + (blk << 4));
            const uint4 x1 = *reinterpret_cast<const uint4*>(asrc + (blk << 4) + 8);
            float d = fdot2_f32(s_sf4_pair[raw.x & 0xFFu], x0.x, 0.0f);
            d = fdot2_f32(s_sf4_pair[(raw.x >> 8) & 0xFFu], x0.y, d);
            d = fdot2_f32(s_sf4_pair[(raw.x >> 16) & 0xFFu], x0.z, d);
            d = fdot2_f32(s_sf4_pair[(raw.x >> 24) & 0xFFu], x0.w, d);
            d = fdot2_f32(s_sf4_pair[raw.y & 0xFFu], x1.x, d);
            d = fdot2_f32(s_sf4_pair[(raw.y >> 8) & 0xFFu], x1.y, d);
            d = fdot2_f32(s_sf4_pair[(raw.y >> 16) & 0xFFu], x1.z, d);
            d = fdot2_f32(s_sf4_pair[(raw.y >> 24) & 0xFFu], x1.w, d);
            acc = fmaf(d, sc, acc);
        }
    } else if (desc.dn_code == CODE_MX8) {
        const uint8_t* W = desc.body + desc.dn_offset;
        const uint8_t* S = W + static_cast<size_t>(D) * inter;
        const uint8_t* wr = W + static_cast<size_t>(row) * inter;
        const uint8_t* sr = S + static_cast<size_t>(row) * (inter >> 5);
        const int nblk = inter >> 5;
        for (int blk = lane; blk < nblk; blk += 32) {
            const float sc = ue8m0_to_f32(sr[blk]);
            const uint8_t* wb = wr + (blk << 5);
            const uint4* xp = reinterpret_cast<const uint4*>(asrc + (blk << 5));
            const uint4 wv0 = load_u128_nt(wb);
            const uint4 wv1 = load_u128_nt(wb + 16);
            const uint32_t ww[8] = {wv0.x, wv0.y, wv0.z, wv0.w, wv1.x, wv1.y, wv1.z, wv1.w};
            float d = 0.0f;
#pragma unroll
            for (int q = 0; q < 4; ++q) {
                const uint4 xv = xp[q];
                const uint32_t a = ww[q * 2 + 0];
                const uint32_t c = ww[q * 2 + 1];
                const hv2 h0 = {s_e4m3[a & 0xFFu], s_e4m3[(a >> 8) & 0xFFu]};
                const hv2 h1 = {s_e4m3[(a >> 16) & 0xFFu], s_e4m3[(a >> 24) & 0xFFu]};
                const hv2 h2 = {s_e4m3[c & 0xFFu], s_e4m3[(c >> 8) & 0xFFu]};
                const hv2 h3 = {s_e4m3[(c >> 16) & 0xFFu], s_e4m3[(c >> 24) & 0xFFu]};
                d = fdot2_f32(__builtin_bit_cast(uint32_t, h0), xv.x, d);
                d = fdot2_f32(__builtin_bit_cast(uint32_t, h1), xv.y, d);
                d = fdot2_f32(__builtin_bit_cast(uint32_t, h2), xv.z, d);
                d = fdot2_f32(__builtin_bit_cast(uint32_t, h3), xv.w, d);
            }
            acc = fmaf(d, sc, acc);
        }
    }

    const float yk = wave32_sum(acc);
    if (lane == 0) y_part[static_cast<size_t>(slot_k) * D + row] = yk;
#else
    (void)exps; (void)topk_ids; (void)act_in; (void)y_part; (void)B; (void)top_k; (void)D; (void)inter;
#endif
}

// =================================================================================================
// 3. FAST-Mode FP8_E4M3FN Quantizers (Row-wise for Dense Trunk, Per-16 for SF4, Per-32 for MX8)
// =================================================================================================

// One-time weight quantization at init: BF16 [N, K] -> FP8_E4M3FN [N, K] + float row_scale [N]
// Grid: (N,), Block: 256 (1 WG per output row)
__global__ void __launch_bounds__(256) k_r9700_quant_bf16_to_fp8_row(
    const uint16_t* __restrict__ W_bf16,
    uint8_t* __restrict__ W_fp8,
    float* __restrict__ W_scale,
    int N, int K) {
#if INK_R9700_DEV
    const int row = blockIdx.x;
    if (row >= N) return;
    const int tid = threadIdx.x;
    const int wave = tid >> 5;
    const int lane = tid & 31;
    const uint16_t* src = W_bf16 + static_cast<size_t>(row) * K;
    uint8_t* dst = W_fp8 + static_cast<size_t>(row) * K;

    float am = 0.0f;
    for (int k = tid; k < K; k += 256) {
        am = fmaxf(am, fabsf(bf16_to_f32(src[k])));
    }
    am = strata::inkling2::hip::wave32_max(am);
    __shared__ float s_m[8];
    if (lane == 0) s_m[wave] = am;
    __syncthreads();
    if (tid == 0) {
        float g = 0.0f;
        for (int w = 0; w < 8; ++w) g = fmaxf(g, s_m[w]);
        s_m[0] = g;
        W_scale[row] = g * (1.0f / 448.0f);
    }
    __syncthreads();
    const float g = s_m[0];
    const float inv = (g > 0.0f) ? (448.0f / g) : 0.0f;
    for (int k = tid; k < K; k += 256) {
        dst[k] = f32_to_e4m3fn(bf16_to_f32(src[k]) * inv);
    }
#else
    (void)W_bf16; (void)W_fp8; (void)W_scale; (void)N; (void)K;
#endif
}

// Fast activation quantization for Dense Trunk: FP32 [M, K] -> FP8_E4M3FN [M, K] + float row_scale [M]
// Grid: (M,), Block: 256 (K divisible by 4)
__global__ void __launch_bounds__(256) k_r9700_quant_f32_to_fp8_row(
    const float* __restrict__ X_f32,
    uint8_t* __restrict__ X_fp8,
    float* __restrict__ X_scale,
    int M, int K) {
#if INK_R9700_DEV
    const int row = blockIdx.x;
    if (row >= M) return;
    const int tid = threadIdx.x;
    const int wave = tid >> 5;
    const int lane = tid & 31;
    const int k4 = K >> 2;
    const float4* src4 = reinterpret_cast<const float4*>(X_f32 + static_cast<size_t>(row) * K);
    uint32_t* dst4 = reinterpret_cast<uint32_t*>(X_fp8 + static_cast<size_t>(row) * K);

    float am = 0.0f;
    for (int i = tid; i < k4; i += 256) {
        const float4 v = src4[i];
        am = fmaxf(am, fmaxf(fmaxf(fabsf(v.x), fabsf(v.y)), fmaxf(fabsf(v.z), fabsf(v.w))));
    }
    am = strata::inkling2::hip::wave32_max(am);
    __shared__ float s_m[8];
    if (lane == 0) s_m[wave] = am;
    __syncthreads();
    if (tid == 0) {
        float g = 0.0f;
        for (int w = 0; w < 8; ++w) g = fmaxf(g, s_m[w]);
        s_m[0] = g;
        X_scale[row] = g * (1.0f / 448.0f);
    }
    __syncthreads();
    const float g = s_m[0];
    const float inv = (g > 0.0f) ? (448.0f / g) : 0.0f;
    for (int i = tid; i < k4; i += 256) {
        const float4 v = src4[i];
        dst4[i] = pack4_u8(
            f32_to_e4m3fn(v.x * inv), f32_to_e4m3fn(v.y * inv),
            f32_to_e4m3fn(v.z * inv), f32_to_e4m3fn(v.w * inv));
    }
#else
    (void)X_f32; (void)X_fp8; (void)X_scale; (void)M; (void)K;
#endif
}

// Per-16 FP8 quantization for SF4 MoE: FP32 [M, K] -> FP8 [M, K] + float d16 [M, K/16]
// Grid: ((n_blks16 + 63) / 64,), Block: 256 (64 blocks of 16 per WG, 4 threads x float4 per block)
__global__ void __launch_bounds__(256) k_r9700_quant_f32_to_fp8_16(
    const float* __restrict__ src,
    uint8_t* __restrict__ dst_fp8,
    float* __restrict__ dst_d16,
    int n_blks16) {
#if INK_R9700_DEV
    const int blk = blockIdx.x * 64 + (threadIdx.x >> 2);
    const int sub = threadIdx.x & 3;
    if (blk >= n_blks16) return;
    const float4 v = *reinterpret_cast<const float4*>(src + static_cast<size_t>(blk) * 16 + sub * 4);
    float am = fmaxf(fmaxf(fabsf(v.x), fabsf(v.y)), fmaxf(fabsf(v.z), fabsf(v.w)));
    am = fmaxf(am, __shfl_xor(am, 1, 32));
    am = fmaxf(am, __shfl_xor(am, 2, 32));
    const float inv = (am > 0.0f) ? (448.0f / am) : 0.0f;
    *reinterpret_cast<uint32_t*>(dst_fp8 + static_cast<size_t>(blk) * 16 + sub * 4) = pack4_u8(
        f32_to_e4m3fn(v.x * inv), f32_to_e4m3fn(v.y * inv),
        f32_to_e4m3fn(v.z * inv), f32_to_e4m3fn(v.w * inv));
    if (sub == 0) dst_d16[blk] = am * (1.0f / 448.0f);
#else
    (void)src; (void)dst_fp8; (void)dst_d16; (void)n_blks16;
#endif
}

// Per-32 FP8 quantization for MX8 MoE (layers 20..23): FP32 [M, K] -> FP8 [M, K] + float d32 [M, K/32]
// Grid: ((n_blks32 + 31) / 32,), Block: 256 (32 blocks of 32 per WG, 8 threads x float4 per block)
__global__ void __launch_bounds__(256) k_r9700_quant_f32_to_fp8_32(
    const float* __restrict__ src,
    uint8_t* __restrict__ dst_fp8,
    float* __restrict__ dst_d32,
    int n_blks32) {
#if INK_R9700_DEV
    const int blk = blockIdx.x * 32 + (threadIdx.x >> 3);
    const int sub = threadIdx.x & 7;
    if (blk >= n_blks32) return;
    const float4 v = *reinterpret_cast<const float4*>(src + static_cast<size_t>(blk) * 32 + sub * 4);
    float am = fmaxf(fmaxf(fabsf(v.x), fabsf(v.y)), fmaxf(fabsf(v.z), fabsf(v.w)));
    am = fmaxf(am, __shfl_xor(am, 1, 32));
    am = fmaxf(am, __shfl_xor(am, 2, 32));
    am = fmaxf(am, __shfl_xor(am, 4, 32));
    const float inv = (am > 0.0f) ? (448.0f / am) : 0.0f;
    *reinterpret_cast<uint32_t*>(dst_fp8 + static_cast<size_t>(blk) * 32 + sub * 4) = pack4_u8(
        f32_to_e4m3fn(v.x * inv), f32_to_e4m3fn(v.y * inv),
        f32_to_e4m3fn(v.z * inv), f32_to_e4m3fn(v.w * inv));
    if (sub == 0) dst_d32[blk] = am * (1.0f / 448.0f);
#else
    (void)src; (void)dst_fp8; (void)dst_d32; (void)n_blks32;
#endif
}

// =================================================================================================
// 4. FAST-Mode Native FP8_E4M3FN Dense Trunk Linear (M=1..8 GEMV, M<=16 Skinny WMMA, M>16 128x128 WMMA)
// =================================================================================================

// M=1 Fused FP8 QKVRG + FP32 Sink Direct-Scatter GEMV (reads 1 B/weight FP8 instead of 2 B/weight BF16 -> 2x BW!)
// Grid: ((tot_n + ns + 7) / 8,), Block: 256 (8 waves = 8 rows)
__global__ void __launch_bounds__(256) k_r9700_fp8_gemv1_qkvrg(
    const uint8_t* __restrict__ W_fp8,
    const float* __restrict__ W_scale,
    const float* __restrict__ X_f32,
    float* __restrict__ q_out,
    float* __restrict__ r_out,
    float* __restrict__ gate_out,
    float* __restrict__ k_out,
    float* __restrict__ v_out,
    const float* __restrict__ sink_w_f32,
    float* __restrict__ sink_out,
    int q_dim, int r_dim, int kv_dim, int tot_n, int ns, int K) {
#if INK_R9700_DEV
    __shared__ half_t s_e4m3[256];
    __shared__ alignas(16) half_t sX[2304];
    const int tid = threadIdx.x;
    s_e4m3[tid] = static_cast<half_t>(e4m3fn_to_f32(static_cast<uint8_t>(tid)));
    const int k4 = K >> 2;
    const float4* x4 = reinterpret_cast<const float4*>(X_f32);
    uint2* sx2 = reinterpret_cast<uint2*>(sX);
    for (int i = tid; i < k4; i += 256) {
        const float4 v = x4[i];
        sx2[i] = make_uint2(strata::inkling2::hip::h2_pack(v.x, v.y),
                            strata::inkling2::hip::h2_pack(v.z, v.w));
    }
    __syncthreads();

    const int wave = tid >> 5;
    const int lane = tid & 31;
    const int row = blockIdx.x * 8 + wave;
    if (row >= tot_n + ns) return;

    if (row >= tot_n) {
        if (!sink_w_f32 || !sink_out) return;
        const int sr = row - tot_n;
        const float4* sw4 = reinterpret_cast<const float4*>(sink_w_f32 + static_cast<size_t>(sr) * K);
        float acc = 0.0f;
        for (int i = lane; i < k4; i += 32) {
            const float4 w = sw4[i];
            const float4 x = x4[i];
            acc = fmaf(w.x, x.x, fmaf(w.y, x.y, fmaf(w.z, x.z, fmaf(w.w, x.w, acc))));
        }
        const float s = wave32_sum(acc);
        if (lane == 0) sink_out[sr] = s;
        return;
    }

    const uint8_t* wr = W_fp8 + static_cast<size_t>(row) * K;
    const float wsc = W_scale[row];
    float acc = 0.0f;
    const int k16 = K >> 4;
    for (int i = lane; i < k16; i += 32) {
        const uint4 raw = *reinterpret_cast<const uint4*>(wr + (i << 4));
        const uint4 xv0 = *reinterpret_cast<const uint4*>(&sX[(i << 4) + 0]);
        const uint4 xv1 = *reinterpret_cast<const uint4*>(&sX[(i << 4) + 8]);
        const uint32_t ww[4] = {raw.x, raw.y, raw.z, raw.w};
        const uint32_t xx[8] = {xv0.x, xv0.y, xv0.z, xv0.w, xv1.x, xv1.y, xv1.z, xv1.w};
#pragma unroll
        for (int q = 0; q < 4; ++q) {
            const uint32_t a = ww[q];
            const hv2 h0 = {s_e4m3[a & 0xFFu], s_e4m3[(a >> 8) & 0xFFu]};
            const hv2 h1 = {s_e4m3[(a >> 16) & 0xFFu], s_e4m3[(a >> 24) & 0xFFu]};
            acc = fdot2_f32(__builtin_bit_cast(uint32_t, h0), xx[q * 2 + 0], acc);
            acc = fdot2_f32(__builtin_bit_cast(uint32_t, h1), xx[q * 2 + 1], acc);
        }
    }
    const float val = wave32_sum(acc) * wsc;
    if (lane == 0) {
        if (row < q_dim) q_out[row] = val;
        else if (row < q_dim + r_dim) r_out[row - q_dim] = val;
        else if (row < 2 * q_dim + r_dim) gate_out[row - q_dim - r_dim] = val;
        else if (k_out && row < 2 * q_dim + r_dim + kv_dim) k_out[row - 2 * q_dim - r_dim] = val;
        else if (v_out) v_out[row - 2 * q_dim - r_dim - kv_dim] = val;
    }
#else
    (void)W_fp8; (void)W_scale; (void)X_f32; (void)q_out; (void)r_out; (void)gate_out;
    (void)k_out; (void)v_out; (void)sink_w_f32; (void)sink_out;
    (void)q_dim; (void)r_dim; (void)kv_dim; (void)tot_n; (void)ns; (void)K;
#endif
}

// M=1..4 bandwidth-halved FP8 trunk GEMV (uint4 16-byte loads = 1 B/weight instead of 2 B/weight for BF16!)
// Grid: ((N + 7) / 8,), Block: 256 (8 waves = 8 rows)
template <int MT>
__global__ void __launch_bounds__(256) k_r9700_fp8_gemv_mr(
    const uint8_t* __restrict__ W_fp8,
    const float* __restrict__ W_scale,
    const float* __restrict__ X_f32,
    const float* __restrict__ bias,
    float* __restrict__ Y,
    int M, int N, int K) {
#if INK_R9700_DEV
    __shared__ half_t s_e4m3[256];
    __shared__ alignas(16) half_t sX[MT][2304];
    const int tid = threadIdx.x;
    s_e4m3[tid] = static_cast<half_t>(e4m3fn_to_f32(static_cast<uint8_t>(tid)));
    const bool use_sx = (K <= 2304 && (K & 15) == 0);
    if (use_sx) {
        const int k4 = K >> 2;
        for (int m = 0; m < M; ++m) {
            const float4* x4 = reinterpret_cast<const float4*>(X_f32 + static_cast<size_t>(m) * K);
            uint2* sx2 = reinterpret_cast<uint2*>(sX[m]);
            for (int i = tid; i < k4; i += 256) {
                const float4 v = x4[i];
                sx2[i] = make_uint2(strata::inkling2::hip::h2_pack(v.x, v.y),
                                    strata::inkling2::hip::h2_pack(v.z, v.w));
            }
        }
    }
    __syncthreads();

    const int wave = tid >> 5;
    const int lane = tid & 31;
    const int row = blockIdx.x * 8 + wave;
    if (row >= N) return;

    const uint8_t* wr = W_fp8 + static_cast<size_t>(row) * K;
    const float wsc = W_scale[row];
    float acc[MT];
#pragma unroll
    for (int m = 0; m < MT; ++m) acc[m] = 0.0f;

    if (use_sx) {
        const int k16 = K >> 4;
        for (int i = lane; i < k16; i += 32) {
            const uint4 raw = *reinterpret_cast<const uint4*>(wr + (i << 4));
            const uint32_t ww[4] = {raw.x, raw.y, raw.z, raw.w};
            uint32_t wpack[8];
#pragma unroll
            for (int q = 0; q < 4; ++q) {
                const uint32_t a = ww[q];
                const hv2 h0 = {s_e4m3[a & 0xFFu], s_e4m3[(a >> 8) & 0xFFu]};
                const hv2 h1 = {s_e4m3[(a >> 16) & 0xFFu], s_e4m3[(a >> 24) & 0xFFu]};
                wpack[q * 2 + 0] = __builtin_bit_cast(uint32_t, h0);
                wpack[q * 2 + 1] = __builtin_bit_cast(uint32_t, h1);
            }
#pragma unroll
            for (int m = 0; m < MT; ++m) {
                if (m < M) {
                    const uint4 xv0 = *reinterpret_cast<const uint4*>(&sX[m][(i << 4) + 0]);
                    const uint4 xv1 = *reinterpret_cast<const uint4*>(&sX[m][(i << 4) + 8]);
                    float d = fdot2_f32(wpack[0], xv0.x, acc[m]);
                    d = fdot2_f32(wpack[1], xv0.y, d);
                    d = fdot2_f32(wpack[2], xv0.z, d);
                    d = fdot2_f32(wpack[3], xv0.w, d);
                    d = fdot2_f32(wpack[4], xv1.x, d);
                    d = fdot2_f32(wpack[5], xv1.y, d);
                    d = fdot2_f32(wpack[6], xv1.z, d);
                    acc[m] = fdot2_f32(wpack[7], xv1.w, d);
                }
            }
        }
    } else {
        const int k8 = K >> 3;
        for (int i = lane; i < k8; i += 32) {
            const uint2 raw = *reinterpret_cast<const uint2*>(wr + i * 8);
            const hv2 h0 = {s_e4m3[raw.x & 0xFFu], s_e4m3[(raw.x >> 8) & 0xFFu]};
            const hv2 h1 = {s_e4m3[(raw.x >> 16) & 0xFFu], s_e4m3[(raw.x >> 24) & 0xFFu]};
            const hv2 h2 = {s_e4m3[raw.y & 0xFFu], s_e4m3[(raw.y >> 8) & 0xFFu]};
            const hv2 h3 = {s_e4m3[(raw.y >> 16) & 0xFFu], s_e4m3[(raw.y >> 24) & 0xFFu]};
            const uint32_t w0 = __builtin_bit_cast(uint32_t, h0);
            const uint32_t w1 = __builtin_bit_cast(uint32_t, h1);
            const uint32_t w2 = __builtin_bit_cast(uint32_t, h2);
            const uint32_t w3 = __builtin_bit_cast(uint32_t, h3);
#pragma unroll
            for (int m = 0; m < MT; ++m) {
                if (m < M) {
                    const float4 x0 = *reinterpret_cast<const float4*>(X_f32 + static_cast<size_t>(m) * K + i * 8);
                    const float4 x1 = *reinterpret_cast<const float4*>(X_f32 + static_cast<size_t>(m) * K + i * 8 + 4);
                    float d = fdot2_f32(w0, strata::inkling2::hip::h2_pack(x0.x, x0.y), acc[m]);
                    d = fdot2_f32(w1, strata::inkling2::hip::h2_pack(x0.z, x0.w), d);
                    d = fdot2_f32(w2, strata::inkling2::hip::h2_pack(x1.x, x1.y), d);
                    acc[m] = fdot2_f32(w3, strata::inkling2::hip::h2_pack(x1.z, x1.w), d);
                }
            }
        }
    }

    const float bv = bias ? bias[row] : 0.0f;
#pragma unroll
    for (int m = 0; m < MT; ++m) {
        const float sum = wave32_sum(acc[m]) * wsc + bv;
        if (lane == 0 && m < M) Y[static_cast<size_t>(m) * N + row] = sum;
    }
#else
    (void)W_fp8; (void)W_scale; (void)X_f32; (void)bias; (void)Y; (void)M; (void)N; (void)K;
#endif
}

// 0-bank-conflict XOR swizzle for 64-byte FP8 LDS rows (saves 25% LDS -> 2x WG/CU occupancy!)
__device__ __forceinline__ int swz64(int row, int b4) {
    return (row << 6) + (((b4 ^ ((row >> 1) & 3))) << 4);
}

// Fast float4 vectorized row*col scale + optional bias epilogue for R9700Fp8BlasLt (233 TFLOP/s FP8 GEMM!)
// Grid: (((N >> 2) + 255) / 256, M), Block: 256  (requires (N & 3) == 0)
__global__ void __launch_bounds__(256) k_r9700_apply_rc_scale(
    float* __restrict__ Y,
    const float* __restrict__ X_scale,
    const float* __restrict__ W_scale,
    const float* __restrict__ bias,
    int M, int N) {
#if INK_R9700_DEV
    const int n4 = N >> 2;
    const int c4 = blockIdx.x * 256 + threadIdx.x;
    const int m = blockIdx.y;
    if (c4 >= n4 || m >= M) return;
    const float xs = X_scale[m];
    float4* y4 = reinterpret_cast<float4*>(Y + static_cast<size_t>(m) * N);
    const float4 y = y4[c4];
    const float4 ws = reinterpret_cast<const float4*>(W_scale)[c4];
    float4 out = make_float4(y.x * (xs * ws.x), y.y * (xs * ws.y), y.z * (xs * ws.z), y.w * (xs * ws.w));
    if (bias) {
        const float4 bv = reinterpret_cast<const float4*>(bias)[c4];
        out.x += bv.x; out.y += bv.y; out.z += bv.z; out.w += bv.w;
    }
    y4[c4] = out;
#else
    (void)Y; (void)X_scale; (void)W_scale; (void)bias; (void)M; (void)N;
#endif
}

// Per-64 FP8 activation quantization (quarantines Goodfire d_2027=+32.4 outlier to 1 single BK=64 block!)
// Grid: ((n_blks64 + 15) / 16,), Block: 256 (16 blocks of 64 per WG, 16 threads x float4 per block)
__global__ void __launch_bounds__(256) k_r9700_quant_f32_to_fp8_64(
    const float* __restrict__ src,
    uint8_t* __restrict__ dst_fp8,
    float* __restrict__ dst_d64,
    int n_blks64) {
#if INK_R9700_DEV
    const int blk = blockIdx.x * 16 + (threadIdx.x >> 4);
    const int sub = threadIdx.x & 15;
    if (blk >= n_blks64) return;
    const float4 v = *reinterpret_cast<const float4*>(src + static_cast<size_t>(blk) * 64 + sub * 4);
    float am = fmaxf(fmaxf(fabsf(v.x), fabsf(v.y)), fmaxf(fabsf(v.z), fabsf(v.w)));
    am = fmaxf(am, __shfl_xor(am, 1, 32));
    am = fmaxf(am, __shfl_xor(am, 2, 32));
    am = fmaxf(am, __shfl_xor(am, 4, 32));
    am = fmaxf(am, __shfl_xor(am, 8, 32));
    const float inv = (am > 0.0f) ? (448.0f / am) : 0.0f;
    *reinterpret_cast<uint32_t*>(dst_fp8 + static_cast<size_t>(blk) * 64 + sub * 4) = pack4_u8(
        f32_to_e4m3fn(v.x * inv), f32_to_e4m3fn(v.y * inv),
        f32_to_e4m3fn(v.z * inv), f32_to_e4m3fn(v.w * inv));
    if (sub == 0) dst_d64[blk] = am * (1.0f / 448.0f);
#else
    (void)src; (void)dst_fp8; (void)dst_d64; (void)n_blks64;
#endif
}

// High-CU-Saturation Skinny Double-Buffered FP8 WMMA Dense Trunk GEMM for M=5..16 (BM=16, BN=32, BK=64)
// Uses per-64 block activation scales X_d64[M, K/64] to quarantine d_2027 outlier while saturating all 64 CUs!
// 6 KB LDS -> 10 WG/CU; 2 waves/WG (16 N-cols per wave).
// Grid: ((N + 31) / 32, (M + 15) / 16), Block: 64
__global__ void __launch_bounds__(64, 8) k_r9700_fp8_wmma_16x32(
    const uint8_t* __restrict__ W_fp8,
    const float* __restrict__ W_scale,
    const uint8_t* __restrict__ X_fp8,
    const float* __restrict__ X_d64,
    const float* __restrict__ bias,
    float* __restrict__ Y,
    int M, int N, int K) {
#if INK_R9700_HAS_FP8_WMMA
    constexpr int BM = 16, BN = 32, BK = 64;
    __shared__ alignas(16) uint8_t sA[2][BM * BK];   // 2,048 B
    __shared__ alignas(16) uint8_t sB[2][BN * BK];   // 4,096 B
    __shared__ float sXd[2][BM];                     //    64 B -> 6,208 B total!

    const int tid = threadIdx.x;
    const int wave = tid >> 5;
    const int lane = tid & 31;
    const int l16 = lane & 15;
    const int hi = lane >> 4;
    const int m_blk = blockIdx.y * BM;
    const int n_blk = blockIdx.x * BN;
    const int ra = tid >> 2, ca = tid & 3;  // 16 rows x 4 uint4s = 64 threads!
    const int nb64 = K >> 6;

    auto load_tile = [&](int k0, uint4& a0, float& xd, uint4& b0, uint4& b1) {
        a0 = (m_blk + ra < M && k0 + ca * 16 < K)
            ? *reinterpret_cast<const uint4*>(X_fp8 + static_cast<size_t>(m_blk + ra) * K + k0 + ca * 16) : make_uint4(0, 0, 0, 0);
        xd = (ca == 0 && m_blk + ra < M) ? X_d64[static_cast<size_t>(m_blk + ra) * nb64 + (k0 >> 6)] : 0.0f;
        b0 = (n_blk + ra +  0 < N && k0 + ca * 16 < K)
            ? *reinterpret_cast<const uint4*>(W_fp8 + static_cast<size_t>(n_blk + ra +  0) * K + k0 + ca * 16) : make_uint4(0, 0, 0, 0);
        b1 = (n_blk + ra + 16 < N && k0 + ca * 16 < K)
            ? *reinterpret_cast<const uint4*>(W_fp8 + static_cast<size_t>(n_blk + ra + 16) * K + k0 + ca * 16) : make_uint4(0, 0, 0, 0);
    };
    auto store_tile = [&](int buf, const uint4& a0, float xd, const uint4& b0, const uint4& b1) {
        *reinterpret_cast<uint4*>(&sA[buf][swz64(ra, ca)]) = a0;
        if (ca == 0) sXd[buf][ra] = xd;
        *reinterpret_cast<uint4*>(&sB[buf][swz64(ra +  0, ca)]) = b0;
        *reinterpret_cast<uint4*>(&sB[buf][swz64(ra + 16, ca)]) = b1;
    };

    uint4 pa0, pb0, pb1;
    float pxd;
    load_tile(0, pa0, pxd, pb0, pb1);
    store_tile(0, pa0, pxd, pb0, pb1);
    sync_lds();

    fv8 acc = fv8_zero();
    int buf = 0;
    for (int k0 = 0; k0 < K; k0 += BK) {
        const int kn = k0 + BK;
        if (kn < K) load_tile(kn, pa0, pxd, pb0, pb1);
        fv8 blk_acc = fv8_zero();
#pragma unroll
        for (int kk = 0; kk < 4; ++kk) {
            iv2 a = wmma_ld_fp8_k16(&sA[buf][swz64(l16, kk)], hi);
            iv2 b = wmma_ld_fp8_k16(&sB[buf][swz64(wave * 16 + l16, kk)], hi);
            blk_acc = wmma_fp8_f32(a, b, blk_acc);
        }
#pragma unroll
        for (int r = 0; r < 8; ++r) acc[r] = fmaf(blk_acc[r], sXd[buf][8 * hi + r], acc[r]);
        if (kn < K) {
            const int nb = buf ^ 1;
            store_tile(nb, pa0, pxd, pb0, pb1);
            sync_lds();
            buf = nb;
        }
    }

    const int gn = n_blk + wave * 16 + l16;
    if (gn < N) {
        const float wsc = W_scale[gn];
        const float bv = bias ? bias[gn] : 0.0f;
#pragma unroll
        for (int r = 0; r < 8; ++r) {
            const int gm = m_blk + 8 * hi + r;
            if (gm < M) Y[static_cast<size_t>(gm) * N + gn] = acc[r] * wsc + bv;
        }
    }
#else
    (void)W_fp8; (void)W_scale; (void)X_fp8; (void)X_d64; (void)bias; (void)Y; (void)M; (void)N; (void)K;
#endif
}

// Skinny Double-Buffered FP8 WMMA Dense Trunk GEMM for M=2..16 (BM=16, BN=128, BK=64, 18 KB LDS -> 3 WG/CU!)
// Grid: ((N + 127) / 128, (M + 15) / 16), Block: 128
__global__ void __launch_bounds__(128, 3) k_r9700_fp8_wmma_16x128(
    const uint8_t* __restrict__ W_fp8,
    const float* __restrict__ W_scale,
    const uint8_t* __restrict__ X_fp8,
    const float* __restrict__ X_scale,
    const float* __restrict__ bias,
    float* __restrict__ Y,
    int M, int N, int K) {
#if INK_R9700_HAS_FP8_WMMA
    constexpr int BM = 16, BN = 128, BK = 64;
    __shared__ alignas(16) uint8_t sA[2][BM * BK];   //  2,048 B
    __shared__ alignas(16) uint8_t sB[2][BN * BK];   // 16,384 B -> 18,432 B (3 WG/CU!)

    const int tid = threadIdx.x;
    const int wave = tid >> 5;
    const int lane = tid & 31;
    const int l16 = lane & 15;
    const int hi = lane >> 4;
    const int m_blk = blockIdx.y * BM;
    const int n_blk = blockIdx.x * BN;
    const int ra = tid >> 2, ca = tid & 3;

    auto load_tile = [&](int k0, uint4& a0, uint4& b0, uint4& b1, uint4& b2, uint4& b3) {
        a0 = (tid < 64 && m_blk + ra < M && k0 + ca * 16 < K)
            ? *reinterpret_cast<const uint4*>(X_fp8 + static_cast<size_t>(m_blk + ra) * K + k0 + ca * 16) : make_uint4(0, 0, 0, 0);
        b0 = (n_blk + ra +  0 < N && k0 + ca * 16 < K) ? *reinterpret_cast<const uint4*>(W_fp8 + static_cast<size_t>(n_blk + ra +  0) * K + k0 + ca * 16) : make_uint4(0, 0, 0, 0);
        b1 = (n_blk + ra + 32 < N && k0 + ca * 16 < K) ? *reinterpret_cast<const uint4*>(W_fp8 + static_cast<size_t>(n_blk + ra + 32) * K + k0 + ca * 16) : make_uint4(0, 0, 0, 0);
        b2 = (n_blk + ra + 64 < N && k0 + ca * 16 < K) ? *reinterpret_cast<const uint4*>(W_fp8 + static_cast<size_t>(n_blk + ra + 64) * K + k0 + ca * 16) : make_uint4(0, 0, 0, 0);
        b3 = (n_blk + ra + 96 < N && k0 + ca * 16 < K) ? *reinterpret_cast<const uint4*>(W_fp8 + static_cast<size_t>(n_blk + ra + 96) * K + k0 + ca * 16) : make_uint4(0, 0, 0, 0);
    };
    auto store_tile = [&](int buf, const uint4& a0, const uint4& b0, const uint4& b1, const uint4& b2, const uint4& b3) {
        if (tid < 64) *reinterpret_cast<uint4*>(&sA[buf][swz64(ra, ca)]) = a0;
        *reinterpret_cast<uint4*>(&sB[buf][swz64(ra +  0, ca)]) = b0;
        *reinterpret_cast<uint4*>(&sB[buf][swz64(ra + 32, ca)]) = b1;
        *reinterpret_cast<uint4*>(&sB[buf][swz64(ra + 64, ca)]) = b2;
        *reinterpret_cast<uint4*>(&sB[buf][swz64(ra + 96, ca)]) = b3;
    };

    uint4 pa0, pb0, pb1, pb2, pb3;
    load_tile(0, pa0, pb0, pb1, pb2, pb3);
    store_tile(0, pa0, pb0, pb1, pb2, pb3);
    sync_lds();

    fv8 acc[2] = {fv8_zero(), fv8_zero()};
    int buf = 0;
    for (int k0 = 0; k0 < K; k0 += BK) {
        const int kn = k0 + BK;
        if (kn < K) load_tile(kn, pa0, pb0, pb1, pb2, pb3);
#pragma unroll
        for (int kk = 0; kk < 4; ++kk) {
            iv2 a  = wmma_ld_fp8_k16(&sA[buf][swz64(l16, kk)], hi);
            iv2 b0 = wmma_ld_fp8_k16(&sB[buf][swz64(wave * 32 +  0 + l16, kk)], hi);
            iv2 b1 = wmma_ld_fp8_k16(&sB[buf][swz64(wave * 32 + 16 + l16, kk)], hi);
            acc[0] = wmma_fp8_f32(a, b0, acc[0]);
            acc[1] = wmma_fp8_f32(a, b1, acc[1]);
        }
        if (kn < K) {
            const int nb = buf ^ 1;
            store_tile(nb, pa0, pb0, pb1, pb2, pb3);
            sync_lds();
            buf = nb;
        }
    }

#pragma unroll
    for (int ni = 0; ni < 2; ++ni) {
        const int gn = n_blk + wave * 32 + ni * 16 + l16;
        if (gn < N) {
            const float wsc = W_scale[gn];
            const float bv = bias ? bias[gn] : 0.0f;
#pragma unroll
            for (int r = 0; r < 8; ++r) {
                const int gm = m_blk + 8 * hi + r;
                if (gm < M) Y[static_cast<size_t>(gm) * N + gn] = acc[ni][r] * (X_scale[gm] * wsc) + bv;
            }
        }
    }
#else
    (void)W_fp8; (void)W_scale; (void)X_fp8; (void)X_scale; (void)bias; (void)Y; (void)M; (void)N; (void)K;
#endif
}

// High-Throughput Double-Buffered FP8 WMMA Dense Trunk GEMM for M=17..127 (BM=64, BN=128, BK=64, 24 KB LDS -> 2 WG/CU!)
// Grid: ((N + 127) / 128, (M + 63) / 64), Block: 256
__global__ void __launch_bounds__(256, 2) k_r9700_fp8_wmma_64x128(
    const uint8_t* __restrict__ W_fp8,
    const float* __restrict__ W_scale,
    const uint8_t* __restrict__ X_fp8,
    const float* __restrict__ X_scale,
    const float* __restrict__ bias,
    float* __restrict__ Y,
    int M, int N, int K) {
#if INK_R9700_HAS_FP8_WMMA
    constexpr int BM = 64, BN = 128, BK = 64;
    __shared__ alignas(16) uint8_t sA[2][BM * BK];  //  8,192 B
    __shared__ alignas(16) uint8_t sB[2][BN * BK];  // 16,384 B -> 24,576 B (2 WG/CU!)

    const int tid = threadIdx.x;
    const int wave = tid >> 5;
    const int lane = tid & 31;
    const int l16 = lane & 15;
    const int hi = lane >> 4;
    const int wm = wave >> 1;
    const int wn = wave & 1;
    const int m_blk = blockIdx.y * BM;
    const int n_blk = blockIdx.x * BN;

    const int r0 = tid >> 2, c0 = tid & 3;
    const int r1 = (tid + 256) >> 2, c1 = (tid + 256) & 3;
    const int gm0 = m_blk + r0;
    const int gn0 = n_blk + r0, gn1 = n_blk + r1;

    auto load_tile = [&](int k0, uint4& a0, uint4& b0, uint4& b1) {
        a0 = (gm0 < M && k0 + c0 * 16 < K) ? *reinterpret_cast<const uint4*>(X_fp8 + static_cast<size_t>(gm0) * K + k0 + c0 * 16) : make_uint4(0, 0, 0, 0);
        b0 = (gn0 < N && k0 + c0 * 16 < K) ? *reinterpret_cast<const uint4*>(W_fp8 + static_cast<size_t>(gn0) * K + k0 + c0 * 16) : make_uint4(0, 0, 0, 0);
        b1 = (gn1 < N && k0 + c1 * 16 < K) ? *reinterpret_cast<const uint4*>(W_fp8 + static_cast<size_t>(gn1) * K + k0 + c1 * 16) : make_uint4(0, 0, 0, 0);
    };

    uint4 ra0, rb0, rb1;
    load_tile(0, ra0, rb0, rb1);
    *reinterpret_cast<uint4*>(&sA[0][swz64(r0, c0)]) = ra0;
    *reinterpret_cast<uint4*>(&sB[0][swz64(r0, c0)]) = rb0;
    *reinterpret_cast<uint4*>(&sB[0][swz64(r1, c1)]) = rb1;
    sync_lds();

    fv8 acc[4] = {fv8_zero(), fv8_zero(), fv8_zero(), fv8_zero()};
    int buf = 0;
    for (int k0 = 0; k0 < K; k0 += BK) {
        const int kn = k0 + BK;
        if (kn < K) load_tile(kn, ra0, rb0, rb1);
#pragma unroll
        for (int kk = 0; kk < 4; ++kk) {
            iv2 a = wmma_ld_fp8_k16(&sA[buf][swz64(wm * 16 + l16, kk)], hi);
#pragma unroll
            for (int ni = 0; ni < 4; ++ni) {
                iv2 b = wmma_ld_fp8_k16(&sB[buf][swz64(wn * 64 + ni * 16 + l16, kk)], hi);
                acc[ni] = wmma_fp8_f32(a, b, acc[ni]);
            }
        }
        if (kn < K) {
            const int nb = buf ^ 1;
            *reinterpret_cast<uint4*>(&sA[nb][swz64(r0, c0)]) = ra0;
            *reinterpret_cast<uint4*>(&sB[nb][swz64(r0, c0)]) = rb0;
            *reinterpret_cast<uint4*>(&sB[nb][swz64(r1, c1)]) = rb1;
            sync_lds();
            buf = nb;
        }
    }

    float xsc[8];
#pragma unroll
    for (int r = 0; r < 8; ++r) {
        const int gm = m_blk + wm * 16 + 8 * hi + r;
        xsc[r] = (gm < M) ? X_scale[gm] : 0.0f;
    }
#pragma unroll
    for (int ni = 0; ni < 4; ++ni) {
        const int gn = n_blk + wn * 64 + ni * 16 + l16;
        if (gn < N) {
            const float wsc = W_scale[gn];
            const float bv = bias ? bias[gn] : 0.0f;
#pragma unroll
            for (int r = 0; r < 8; ++r) {
                const int gm = m_blk + wm * 16 + 8 * hi + r;
                if (gm < M) Y[static_cast<size_t>(gm) * N + gn] = acc[ni][r] * (xsc[r] * wsc) + bv;
            }
        }
    }
#else
    (void)W_fp8; (void)W_scale; (void)X_fp8; (void)X_scale; (void)bias; (void)Y; (void)M; (void)N; (void)K;
#endif
}

// High-Throughput Double-Buffered FP8 WMMA Dense Trunk GEMM for M>=128 (BM=128, BN=128, BK=64)
// XOR-swizzled LD=64 -> EXACTLY 32,768 B LDS (2 WG/CU = 16 waves/CU!!) + 8-wide M-tile MALL swizzle!
// Grid: ((N + 127) / 128, (M + 127) / 128), Block: 256
__global__ void __launch_bounds__(256, 2) k_r9700_fp8_wmma_128x128(
    const uint8_t* __restrict__ W_fp8,
    const float* __restrict__ W_scale,
    const uint8_t* __restrict__ X_fp8,
    const float* __restrict__ X_scale,
    const float* __restrict__ bias,
    float* __restrict__ Y,
    int M, int N, int K) {
#if INK_R9700_HAS_FP8_WMMA
    constexpr int BM = 128, BN = 128, BK = 64;
    __shared__ alignas(16) uint8_t sA[2][BM * BK];  // 16,384 B
    __shared__ alignas(16) uint8_t sB[2][BN * BK];  // 16,384 B -> 32,768 B EXACTLY (2 WG/CU!)

    // 8-wide M-tile L2/MALL swizzle so concurrent WGs share W_fp8 tiles in 64 MiB MALL!
    constexpr int kGrpM = 8;
    const int gx = static_cast<int>(gridDim.x);
    const int gy = static_cast<int>(gridDim.y);
    const int lin = static_cast<int>(blockIdx.y) * gx + static_cast<int>(blockIdx.x);
    const int grp_sz = kGrpM * gx;
    const int grp_id = lin / grp_sz;
    const int first_m = grp_id * kGrpM;
    const int m_in_grp = min(kGrpM, gy - first_m);
    const int rem = lin - grp_id * grp_sz;
    const int bm_idx = first_m + (rem % m_in_grp);
    const int bn_idx = rem / m_in_grp;

    const int tid = threadIdx.x;
    const int wave = tid >> 5;
    const int lane = tid & 31;
    const int l16 = lane & 15;
    const int hi = lane >> 4;
    const int wm = wave >> 1;  // 0..3 (32 M/wave)
    const int wn = wave & 1;   // 0..1 (64 N/wave)
    const int m_blk = bm_idx * BM;
    const int n_blk = bn_idx * BN;

    const int r0 = tid >> 2, c0 = tid & 3;
    const int r1 = (tid + 256) >> 2, c1 = (tid + 256) & 3;
    const int gm0 = m_blk + r0, gm1 = m_blk + r1;
    const int gn0 = n_blk + r0, gn1 = n_blk + r1;

    auto load_tile = [&](int k0, uint4& a0, uint4& a1, uint4& b0, uint4& b1) {
        a0 = (gm0 < M && k0 + c0 * 16 < K) ? *reinterpret_cast<const uint4*>(X_fp8 + static_cast<size_t>(gm0) * K + k0 + c0 * 16) : make_uint4(0, 0, 0, 0);
        a1 = (gm1 < M && k0 + c1 * 16 < K) ? *reinterpret_cast<const uint4*>(X_fp8 + static_cast<size_t>(gm1) * K + k0 + c1 * 16) : make_uint4(0, 0, 0, 0);
        b0 = (gn0 < N && k0 + c0 * 16 < K) ? *reinterpret_cast<const uint4*>(W_fp8 + static_cast<size_t>(gn0) * K + k0 + c0 * 16) : make_uint4(0, 0, 0, 0);
        b1 = (gn1 < N && k0 + c1 * 16 < K) ? *reinterpret_cast<const uint4*>(W_fp8 + static_cast<size_t>(gn1) * K + k0 + c1 * 16) : make_uint4(0, 0, 0, 0);
    };

    uint4 ra0, ra1, rb0, rb1;
    load_tile(0, ra0, ra1, rb0, rb1);
    *reinterpret_cast<uint4*>(&sA[0][swz64(r0, c0)]) = ra0;
    *reinterpret_cast<uint4*>(&sA[0][swz64(r1, c1)]) = ra1;
    *reinterpret_cast<uint4*>(&sB[0][swz64(r0, c0)]) = rb0;
    *reinterpret_cast<uint4*>(&sB[0][swz64(r1, c1)]) = rb1;
    sync_lds();

    fv8 acc[2][4];
#pragma unroll
    for (int mi = 0; mi < 2; ++mi)
#pragma unroll
        for (int ni = 0; ni < 4; ++ni) acc[mi][ni] = fv8_zero();

    int buf = 0;
    for (int k0 = 0; k0 < K; k0 += BK) {
        const int kn = k0 + BK;
        if (kn < K) load_tile(kn, ra0, ra1, rb0, rb1);

#pragma unroll
        for (int kk = 0; kk < 4; ++kk) {
            iv2 a0 = wmma_ld_fp8_k16(&sA[buf][swz64(wm * 32 +  0 + l16, kk)], hi);
            iv2 a1 = wmma_ld_fp8_k16(&sA[buf][swz64(wm * 32 + 16 + l16, kk)], hi);
#pragma unroll
            for (int ni = 0; ni < 4; ++ni) {
                iv2 b = wmma_ld_fp8_k16(&sB[buf][swz64(wn * 64 + ni * 16 + l16, kk)], hi);
                acc[0][ni] = wmma_fp8_f32(a0, b, acc[0][ni]);
                acc[1][ni] = wmma_fp8_f32(a1, b, acc[1][ni]);
            }
        }

        if (kn < K) {
            const int nb = buf ^ 1;
            *reinterpret_cast<uint4*>(&sA[nb][swz64(r0, c0)]) = ra0;
            *reinterpret_cast<uint4*>(&sA[nb][swz64(r1, c1)]) = ra1;
            *reinterpret_cast<uint4*>(&sB[nb][swz64(r0, c0)]) = rb0;
            *reinterpret_cast<uint4*>(&sB[nb][swz64(r1, c1)]) = rb1;
            sync_lds();
            buf = nb;
        }
    }

    float xsc[2][8];
#pragma unroll
    for (int mi = 0; mi < 2; ++mi) {
#pragma unroll
        for (int r = 0; r < 8; ++r) {
            const int gm = m_blk + wm * 32 + mi * 16 + 8 * hi + r;
            xsc[mi][r] = (gm < M) ? X_scale[gm] : 0.0f;
        }
    }
#pragma unroll
    for (int ni = 0; ni < 4; ++ni) {
        const int gn = n_blk + wn * 64 + ni * 16 + l16;
        if (gn < N) {
            const float wsc = W_scale[gn];
            const float bv = bias ? bias[gn] : 0.0f;
#pragma unroll
            for (int mi = 0; mi < 2; ++mi) {
#pragma unroll
                for (int r = 0; r < 8; ++r) {
                    const int gm = m_blk + wm * 32 + mi * 16 + 8 * hi + r;
                    if (gm < M) {
                        Y[static_cast<size_t>(gm) * N + gn] = acc[mi][ni][r] * (xsc[mi][r] * wsc) + bv;
                    }
                }
            }
        }
    }
#else
    (void)W_fp8; (void)W_scale; (void)X_fp8; (void)X_scale; (void)bias; (void)Y; (void)M; (void)N; (void)K;
#endif
}

// =================================================================================================
// 5. FAST-Mode Double-Buffered XOR-Swizzled Persistent FP8 WMMA Grouped-MoE (BM=128 / BM=64 / BM=16!)
//    - Double-buffered sXq[2], sWq[2], sWs[2] with 0-bank-conflict XOR swizzle (1 barrier per K=64!)
//    - 256-entry s_e4m3_sc[256] LDS LUT (replaces 4 branchy e4m3fn_to_f32 per thread per K=64!)
//    - half_t sWs for SF4 (100% exact for all 256 e4m3fn scales; sf4_outer folded ONCE in epilogue!)
//    - float sWs for MX8 (preserves full 2^-127..2^127 ue8m0 exponent range!)
//    - Per-32 act_d32 in BM=64/128 epilogue + #pragma nounroll (shrinks code size 6x -> L1 I-cache hit!)
// =================================================================================================

struct MoeTile16 {
    int16_t eid;
    int16_t m_tile;  // t0 = m_tile * BM
};

template <int BM>
__global__ void __launch_bounds__(128) k_r9700_moe_build_tiles(
    const int32_t* __restrict__ exp_off,   // [129]
    MoeTile16* __restrict__ tiles,         // [max_tiles]
    int32_t* __restrict__ meta,            // [3]: [0]=n_tiles, [1]=cursor_w13(0), [2]=cursor_w2(0)
    int max_tiles) {
#if INK_R9700_DEV
    const int e = threadIdx.x;
    __shared__ int32_t s_nt[128];
    __shared__ int32_t s_off[129];
    const int m_e = exp_off[e + 1] - exp_off[e];
    const int nt = (m_e > 0) ? ((m_e + BM - 1) / BM) : 0;
    s_nt[e] = nt;
    __syncthreads();
    if (e == 0) {
        int acc = 0;
        for (int i = 0; i < 128; ++i) {
            s_off[i] = acc;
            acc += s_nt[i];
        }
        s_off[128] = acc;
        meta[0] = (acc < max_tiles) ? acc : max_tiles;
        meta[1] = 0;
        meta[2] = 0;
    }
    __syncthreads();
    const int base = s_off[e];
    for (int t = 0; t < nt; ++t) {
        if (base + t < max_tiles) {
            tiles[base + t] = MoeTile16{static_cast<int16_t>(e), static_cast<int16_t>(t)};
        }
    }
#else
    (void)exp_off; (void)tiles; (void)meta; (void)max_tiles;
#endif
}

// Persistent Double-Buffered FP8 WMMA w13 + fused SITU + per-32 FP8 quant (BM=64 or BM=128!)
// Grid: (128,), Block: 256 (8 waves: wm=0..3 owns 16*MI M-rows, wn=0..1 owns 64 N-cols = 32 inter!)
template <int CODEC_KIND, int BM>
__global__ void __launch_bounds__(256, (BM == 64) ? 2 : 1) k_r9700_moe_w13_situ_fp8_persist(
    const GpuExpertDesc* __restrict__ exps,
    const int32_t* __restrict__ exp_off,
    const int32_t* __restrict__ bucket_tok,
    const MoeTile16* __restrict__ tiles,
    int32_t* __restrict__ meta,            // [0]=n_tiles, [1]=atomic cursor
    const uint8_t* __restrict__ X_fp8,     // [B, D] row-quantized FP8
    const float* __restrict__ X_scale,     // [B] per-row float scale!
    uint8_t* __restrict__ act_fp8,         // [B * top_k, inter] fused FP8 output
    float* __restrict__ act_d32,           // [B * top_k, inter/32] per-32 float scale!
    int D, int inter, float beta, float lb) {
#if INK_R9700_HAS_FP8_WMMA
    constexpr int BN = 128, BK = 64, MI = BM / 64;
    constexpr int N_SC = (CODEC_KIND == 1) ? 4 : 2;
    using WsT = typename std::conditional<CODEC_KIND == 1, half_t, float>::type;
    const int N = 2 * inter;
    const int n_nb = (N + BN - 1) / BN;
    const int n_tiles = meta[0];
    const int total_work = n_tiles * n_nb;
    if (total_work <= 0) return;

    __shared__ alignas(16) uint8_t sXq[2][BM * BK];   //  8 KB (BM=64) or 16 KB (BM=128)
    __shared__ alignas(16) uint8_t sWq[2][BN * BK];   // 16 KB
    __shared__ WsT sWs[2][BN * N_SC];                 //  2 KB (half_t*512 for SF4, float*256 for MX8!)
    __shared__ half_t s_e4m3_sc[256];                 //  0.5 KB (initialized ONCE per WG!)
    __shared__ int32_t sTok[BM];
    __shared__ float sXsc[BM];
    __shared__ int32_t s_work_id;

    const int tid = threadIdx.x;
    if constexpr (CODEC_KIND == 1) {
        s_e4m3_sc[tid] = static_cast<half_t>(e4m3fn_to_f32(static_cast<uint8_t>(tid)));
        sync_lds();
    }

    const int wave = tid >> 5;
    const int lane = tid & 31;
    const int l16 = lane & 15;
    const int hi = lane >> 4;
    const int wm = wave >> 1;  // 0..3
    const int wn = wave & 1;   // 0..1 (64 N/wave = 32 inter/wave)
    const int nblk_x = (CODEC_KIND == 1) ? (D >> 4) : (D >> 5);
    const int nblk_inter = inter >> 5;
    const int r_x = tid >> 2, b4_x = tid & 3;
    const int r_w = tid >> 1, h_w = tid & 1;

    for (;;) {
        if (tid == 0) s_work_id = atomicAdd(&meta[1], 1);
        sync_lds();
        const int wid = s_work_id;
        if (wid >= total_work) break;

        const int tidx = wid / n_nb;
        const int n_blk = (wid - tidx * n_nb) * BN;
        const MoeTile16 mt = tiles[tidx];
        const int eid = mt.eid;
        const int beg = exp_off[eid];
        const int m_e = exp_off[eid + 1] - beg;
        const int t0 = static_cast<int>(mt.m_tile) * BM;
        const int tm = (m_e - t0 < BM) ? (m_e - t0) : BM;
        const GpuExpertDesc desc = exps[eid];
        if (!desc.body || tm <= 0) continue;

        if (tid < BM) {
            const int tok = (tid < tm) ? bucket_tok[beg + t0 + tid] : 0;
            sTok[tid] = tok;
            sXsc[tid] = (tid < tm) ? X_scale[tok] : 0.0f;
        }
        sync_lds();

        const float sf4_outer = (CODEC_KIND == 1)
            ? (reinterpret_cast<const float*>(desc.body)[0] / reinterpret_cast<const float*>(desc.body)[1]) : 1.0f;
        const uint8_t* sf4_p = desc.body + 16;
        const uint8_t* sf4_s = sf4_p + static_cast<size_t>(N) * (D >> 1);
        const uint8_t* mx8_w = desc.body;
        const uint8_t* mx8_s = mx8_w + static_cast<size_t>(N) * D;

        const int gn_w = n_blk + r_w;
        const int gn_s = n_blk + tid;

        auto prefetch_k = [&](int k0, uint4& rx0, uint4& rx1, uint4& rw0, uint4& rw1, uint32_t& rs) {
            rx0 = (r_x < tm)
                ? *reinterpret_cast<const uint4*>(X_fp8 + static_cast<size_t>(sTok[r_x]) * D + k0 + b4_x * 16)
                : make_uint4(0, 0, 0, 0);
            if constexpr (MI == 2) {
                rx1 = (r_x + 64 < tm)
                    ? *reinterpret_cast<const uint4*>(X_fp8 + static_cast<size_t>(sTok[r_x + 64]) * D + k0 + b4_x * 16)
                    : make_uint4(0, 0, 0, 0);
            } else {
                rx1 = make_uint4(0, 0, 0, 0);
            }
            if constexpr (CODEC_KIND == 1) {
                rw0 = (gn_w < N)
                    ? *reinterpret_cast<const uint4*>(sf4_p + static_cast<size_t>(gn_w) * (D >> 1) + (k0 >> 1) + h_w * 16)
                    : make_uint4(0, 0, 0, 0);
                rs = (tid < BN && gn_s < N)
                    ? *reinterpret_cast<const uint32_t*>(sf4_s + static_cast<size_t>(gn_s) * nblk_x + (k0 >> 4))
                    : 0u;
                rw1 = make_uint4(0, 0, 0, 0);
            } else {
                const int gn0 = n_blk + r_x, gn1 = n_blk + r_x + 64;
                rw0 = (gn0 < N) ? *reinterpret_cast<const uint4*>(mx8_w + static_cast<size_t>(gn0) * D + k0 + b4_x * 16) : make_uint4(0, 0, 0, 0);
                rw1 = (gn1 < N) ? *reinterpret_cast<const uint4*>(mx8_w + static_cast<size_t>(gn1) * D + k0 + b4_x * 16) : make_uint4(0, 0, 0, 0);
                rs = (tid < BN && gn_s < N)
                    ? static_cast<uint32_t>(*reinterpret_cast<const uint16_t*>(mx8_s + static_cast<size_t>(gn_s) * nblk_x + (k0 >> 5)))
                    : 0u;
            }
        };
        auto commit_k = [&](int buf, const uint4& rx0, const uint4& rx1, const uint4& rw0, const uint4& rw1, uint32_t rs) {
            *reinterpret_cast<uint4*>(&sXq[buf][swz64(r_x, b4_x)]) = rx0;
            if constexpr (MI == 2) {
                *reinterpret_cast<uint4*>(&sXq[buf][swz64(r_x + 64, b4_x)]) = rx1;
            }
            if constexpr (CODEC_KIND == 1) {
                uint32_t w0, w1, w2, w3, w4, w5, w6, w7;
                sf4_u32_to_fp8x8(rw0.x, w0, w1);
                sf4_u32_to_fp8x8(rw0.y, w2, w3);
                sf4_u32_to_fp8x8(rw0.z, w4, w5);
                sf4_u32_to_fp8x8(rw0.w, w6, w7);
                *reinterpret_cast<uint4*>(&sWq[buf][swz64(r_w, h_w * 2 + 0)]) = make_uint4(w0, w1, w2, w3);
                *reinterpret_cast<uint4*>(&sWq[buf][swz64(r_w, h_w * 2 + 1)]) = make_uint4(w4, w5, w6, w7);
                if (tid < BN) {
                    sWs[buf][tid * 4 + 0] = s_e4m3_sc[rs & 0xFFu];
                    sWs[buf][tid * 4 + 1] = s_e4m3_sc[(rs >> 8) & 0xFFu];
                    sWs[buf][tid * 4 + 2] = s_e4m3_sc[(rs >> 16) & 0xFFu];
                    sWs[buf][tid * 4 + 3] = s_e4m3_sc[(rs >> 24) & 0xFFu];
                }
            } else {
                *reinterpret_cast<uint4*>(&sWq[buf][swz64(r_x +  0, b4_x)]) = rw0;
                *reinterpret_cast<uint4*>(&sWq[buf][swz64(r_x + 64, b4_x)]) = rw1;
                if (tid < BN) {
                    sWs[buf][tid * 2 + 0] = ue8m0_to_f32(static_cast<uint8_t>(rs & 0xFFu));
                    sWs[buf][tid * 2 + 1] = ue8m0_to_f32(static_cast<uint8_t>((rs >> 8) & 0xFFu));
                }
            }
        };

        uint4 rx0_p, rx1_p, rw0_p, rw1_p;
        uint32_t rs_p;
        prefetch_k(0, rx0_p, rx1_p, rw0_p, rw1_p, rs_p);
        commit_k(0, rx0_p, rx1_p, rw0_p, rw1_p, rs_p);
        sync_lds();

        fv8 acc[MI][4];
#pragma unroll
        for (int mi = 0; mi < MI; ++mi)
#pragma unroll
            for (int ni = 0; ni < 4; ++ni) acc[mi][ni] = fv8_zero();

        int buf = 0;
        for (int k0 = 0; k0 < D; k0 += BK) {
            const int kn = k0 + BK;
            if (kn < D) prefetch_k(kn, rx0_p, rx1_p, rw0_p, rw1_p, rs_p);

            if (wm * (16 * MI) < tm) {
                if constexpr (CODEC_KIND == 1) {
#pragma unroll
                    for (int kk = 0; kk < 4; ++kk) {
                        iv2 a0 = wmma_ld_fp8_k16(&sXq[buf][swz64(wm * (16 * MI) + l16, kk)], hi);
                        iv2 b[4];
                        float ws[4];
#pragma unroll
                        for (int ni = 0; ni < 4; ++ni) {
                            const int n_row = wn * 64 + ni * 16 + l16;
                            b[ni] = wmma_ld_fp8_k16(&sWq[buf][swz64(n_row, kk)], hi);
                            ws[ni] = static_cast<float>(sWs[buf][n_row * 4 + kk]);
                        }
                        // Issue 4 independent FP8 WMMAs back-to-back to hide 32-cycle WMMA->FMA pipeline latency!
                        fv8 p0 = wmma_fp8_f32(a0, b[0], fv8_zero());
                        fv8 p1 = wmma_fp8_f32(a0, b[1], fv8_zero());
                        fv8 p2 = wmma_fp8_f32(a0, b[2], fv8_zero());
                        fv8 p3 = wmma_fp8_f32(a0, b[3], fv8_zero());
#pragma unroll
                        for (int r = 0; r < 8; ++r) {
                            acc[0][0][r] = fmaf(p0[r], ws[0], acc[0][0][r]);
                            acc[0][1][r] = fmaf(p1[r], ws[1], acc[0][1][r]);
                            acc[0][2][r] = fmaf(p2[r], ws[2], acc[0][2][r]);
                            acc[0][3][r] = fmaf(p3[r], ws[3], acc[0][3][r]);
                        }
                        if constexpr (MI == 2) {
                            iv2 a1 = wmma_ld_fp8_k16(&sXq[buf][swz64(wm * (16 * MI) + 16 + l16, kk)], hi);
                            fv8 q0 = wmma_fp8_f32(a1, b[0], fv8_zero());
                            fv8 q1 = wmma_fp8_f32(a1, b[1], fv8_zero());
                            fv8 q2 = wmma_fp8_f32(a1, b[2], fv8_zero());
                            fv8 q3 = wmma_fp8_f32(a1, b[3], fv8_zero());
#pragma unroll
                            for (int r = 0; r < 8; ++r) {
                                acc[1][0][r] = fmaf(q0[r], ws[0], acc[1][0][r]);
                                acc[1][1][r] = fmaf(q1[r], ws[1], acc[1][1][r]);
                                acc[1][2][r] = fmaf(q2[r], ws[2], acc[1][2][r]);
                                acc[1][3][r] = fmaf(q3[r], ws[3], acc[1][3][r]);
                            }
                        }
                    }
                } else {
#pragma unroll
                    for (int g2 = 0; g2 < 2; ++g2) {
                        fv8 p0[4] = {fv8_zero(), fv8_zero(), fv8_zero(), fv8_zero()};
                        fv8 p1[4] = {fv8_zero(), fv8_zero(), fv8_zero(), fv8_zero()};
#pragma unroll
                        for (int s = 0; s < 2; ++s) {
                            const int kk = g2 * 2 + s;
                            iv2 a0 = wmma_ld_fp8_k16(&sXq[buf][swz64(wm * (16 * MI) + l16, kk)], hi);
                            iv2 b[4];
#pragma unroll
                            for (int ni = 0; ni < 4; ++ni) {
                                b[ni] = wmma_ld_fp8_k16(&sWq[buf][swz64(wn * 64 + ni * 16 + l16, kk)], hi);
                            }
#pragma unroll
                            for (int ni = 0; ni < 4; ++ni) p0[ni] = wmma_fp8_f32(a0, b[ni], p0[ni]);
                            if constexpr (MI == 2) {
                                iv2 a1 = wmma_ld_fp8_k16(&sXq[buf][swz64(wm * (16 * MI) + 16 + l16, kk)], hi);
#pragma unroll
                                for (int ni = 0; ni < 4; ++ni) p1[ni] = wmma_fp8_f32(a1, b[ni], p1[ni]);
                            }
                        }
#pragma unroll
                        for (int ni = 0; ni < 4; ++ni) {
                            const float ws = static_cast<float>(sWs[buf][(wn * 64 + ni * 16 + l16) * 2 + g2]);
#pragma unroll
                            for (int r = 0; r < 8; ++r) acc[0][ni][r] = fmaf(p0[ni][r], ws, acc[0][ni][r]);
                            if constexpr (MI == 2) {
#pragma unroll
                                for (int r = 0; r < 8; ++r) acc[1][ni][r] = fmaf(p1[ni][r], ws, acc[1][ni][r]);
                            }
                        }
                    }
                }
            }
            if (kn < D) {
                const int nb = buf ^ 1;
                commit_k(nb, rx0_p, rx1_p, rw0_p, rw1_p, rs_p);
                sync_lds();
                buf = nb;
            }
        }

        // Compact Epilogue (#pragma nounroll shrinks code size 6x -> fits L1 I-cache!)
        // Per-32 act_d32 for BOTH SF4 & MX8 (ZERO cross-wave LDS sync!)
        if (wm * (16 * MI) < tm) {
            const bool is_even = ((lane & 1) == 0);
            const int gj_base = (n_blk + wn * 64) >> 1;  // 32-aligned!
            for (int mi = 0; mi < MI; ++mi) {
                const int m_off = wm * (16 * MI) + mi * 16;
                if (m_off >= tm) continue;
#pragma nounroll
                for (int r = 0; r < 8; ++r) {
                    const int lm = m_off + 8 * hi + r;
                    if (lm >= tm) continue;
                    const float xsc = sXsc[lm] * sf4_outer;
                    const size_t row_out = static_cast<size_t>(beg + t0 + lm);
                    float v[4];
#pragma unroll
                    for (int ni = 0; ni < 4; ++ni) {
                        const float g_val = acc[mi][ni][r] * xsc;
                        const float u_val = __shfl_xor(g_val, 1, 32);
                        v[ni] = is_even ? situ_f32(g_val, u_val, beta, lb) : 0.0f;
                    }
                    float am = fmaxf(fmaxf(fabsf(v[0]), fabsf(v[1])), fmaxf(fabsf(v[2]), fabsf(v[3])));
                    am = fmaxf(am, __shfl_xor(am, 2, 32));
                    am = fmaxf(am, __shfl_xor(am, 4, 32));
                    am = fmaxf(am, __shfl_xor(am, 8, 32));
                    const float inv = (am > 0.0f) ? (448.0f / am) : 0.0f;
                    if (is_even) {
                        for (int ni = 0; ni < 4; ++ni) {
                            const int j = gj_base + ni * 8 + (l16 >> 1);
                            if (j < inter) act_fp8[row_out * inter + j] = f32_to_e4m3fn(v[ni] * inv);
                        }
                        if (l16 == 0 && gj_base < inter) {
                            act_d32[row_out * nblk_inter + (gj_base >> 5)] = am * (1.0f / 448.0f);
                        }
                    }
                }
            }
        }
        sync_lds();
    }
#else
    (void)exps; (void)exp_off; (void)bucket_tok; (void)tiles; (void)meta;
    (void)X_fp8; (void)X_scale; (void)act_fp8; (void)act_d32; (void)D; (void)inter; (void)beta; (void)lb;
#endif
}

// Persistent Double-Buffered FP8 WMMA w2 (BM=64 or BM=128, consumes per-32 act_d32 -> 2x fewer sXd loads!)
// Grid: (128,), Block: 256
template <int CODEC_KIND, int BM>
__global__ void __launch_bounds__(256, (BM == 64) ? 2 : 1) k_r9700_moe_w2_fp8_persist(
    const GpuExpertDesc* __restrict__ exps,
    const int32_t* __restrict__ exp_off,
    const MoeTile16* __restrict__ tiles,
    int32_t* __restrict__ meta,            // [0]=n_tiles, [2]=atomic cursor
    const uint8_t* __restrict__ act_fp8,
    const float* __restrict__ act_d32,     // [B * top_k, inter/32]
    float* __restrict__ bucket_y,
    int D, int inter) {
#if INK_R9700_HAS_FP8_WMMA
    constexpr int BN = 128, BK = 64, MI = BM / 64;
    constexpr int N_SC = (CODEC_KIND == 1) ? 4 : 2;
    using WsT = typename std::conditional<CODEC_KIND == 1, half_t, float>::type;
    const int n_nb = (D + BN - 1) / BN;
    const int n_tiles = meta[0];
    const int total_work = n_tiles * n_nb;
    if (total_work <= 0) return;

    __shared__ alignas(16) uint8_t sXq[2][BM * BK];   //  8 KB (BM=64) or 16 KB (BM=128)
    __shared__ alignas(16) uint8_t sWq[2][BN * BK];   // 16 KB
    __shared__ float sXd[2][BM * 2];                  //  1 KB (BM=64) or  2 KB (BM=128)
    __shared__ WsT sWs[2][BN * N_SC];                 //  2 KB
    __shared__ half_t s_e4m3_sc[256];                 //  0.5 KB
    __shared__ int32_t s_work_id;                     // -> 27.5 KB (BM=64, 2 WG/CU!) or 36.5 KB (BM=128)

    const int tid = threadIdx.x;
    if constexpr (CODEC_KIND == 1) {
        s_e4m3_sc[tid] = static_cast<half_t>(e4m3fn_to_f32(static_cast<uint8_t>(tid)));
        sync_lds();
    }

    const int wave = tid >> 5;
    const int lane = tid & 31;
    const int l16 = lane & 15;
    const int hi = lane >> 4;
    const int wm = wave >> 1;
    const int wn = wave & 1;
    const int nblk_w = (CODEC_KIND == 1) ? (inter >> 4) : (inter >> 5);
    const int nblk_x32 = inter >> 5;
    const int r_x = tid >> 2, b4_x = tid & 3;
    const int r_w = tid >> 1, h_w = tid & 1;

    for (;;) {
        if (tid == 0) s_work_id = atomicAdd(&meta[2], 1);
        sync_lds();
        const int wid = s_work_id;
        if (wid >= total_work) break;

        const int tidx = wid / n_nb;
        const int n_blk = (wid - tidx * n_nb) * BN;
        const MoeTile16 mt = tiles[tidx];
        const int eid = mt.eid;
        const int beg = exp_off[eid];
        const int m_e = exp_off[eid + 1] - beg;
        const int t0 = static_cast<int>(mt.m_tile) * BM;
        const int tm = (m_e - t0 < BM) ? (m_e - t0) : BM;
        const GpuExpertDesc desc = exps[eid];
        if (!desc.body || tm <= 0) continue;

        const float sf4_outer = (CODEC_KIND == 1)
            ? (reinterpret_cast<const float*>(desc.body)[2] / reinterpret_cast<const float*>(desc.body)[3]) : 1.0f;
        const uint8_t* sf4_p = desc.body + desc.dn_offset;
        const uint8_t* sf4_s = sf4_p + static_cast<size_t>(D) * (inter >> 1);
        const uint8_t* mx8_w = desc.body + desc.dn_offset;
        const uint8_t* mx8_s = mx8_w + static_cast<size_t>(D) * inter;

        const int gn_w = n_blk + r_w;
        const int gn_s = n_blk + tid;

        auto prefetch_k = [&](int k0, uint4& rx0, uint4& rx1, float& rxd0, float& rxd1, uint4& rw0, uint4& rw1, uint32_t& rs) {
            if (r_x < tm) {
                const size_t row_i = static_cast<size_t>(beg + t0 + r_x);
                rx0 = *reinterpret_cast<const uint4*>(act_fp8 + row_i * inter + k0 + b4_x * 16);
                rxd0 = ((b4_x & 1) == 0) ? act_d32[row_i * nblk_x32 + (k0 >> 5) + (b4_x >> 1)] : 0.0f;
            } else {
                rx0 = make_uint4(0, 0, 0, 0); rxd0 = 0.0f;
            }
            if constexpr (MI == 2) {
                if (r_x + 64 < tm) {
                    const size_t row_i = static_cast<size_t>(beg + t0 + r_x + 64);
                    rx1 = *reinterpret_cast<const uint4*>(act_fp8 + row_i * inter + k0 + b4_x * 16);
                    rxd1 = ((b4_x & 1) == 0) ? act_d32[row_i * nblk_x32 + (k0 >> 5) + (b4_x >> 1)] : 0.0f;
                } else {
                    rx1 = make_uint4(0, 0, 0, 0); rxd1 = 0.0f;
                }
            } else {
                rx1 = make_uint4(0, 0, 0, 0); rxd1 = 0.0f;
            }
            if constexpr (CODEC_KIND == 1) {
                rw0 = (gn_w < D)
                    ? *reinterpret_cast<const uint4*>(sf4_p + static_cast<size_t>(gn_w) * (inter >> 1) + (k0 >> 1) + h_w * 16)
                    : make_uint4(0, 0, 0, 0);
                rs = (tid < BN && gn_s < D)
                    ? *reinterpret_cast<const uint32_t*>(sf4_s + static_cast<size_t>(gn_s) * nblk_w + (k0 >> 4))
                    : 0u;
                rw1 = make_uint4(0, 0, 0, 0);
            } else {
                const int gn0 = n_blk + r_x, gn1 = n_blk + r_x + 64;
                rw0 = (gn0 < D) ? *reinterpret_cast<const uint4*>(mx8_w + static_cast<size_t>(gn0) * inter + k0 + b4_x * 16) : make_uint4(0, 0, 0, 0);
                rw1 = (gn1 < D) ? *reinterpret_cast<const uint4*>(mx8_w + static_cast<size_t>(gn1) * inter + k0 + b4_x * 16) : make_uint4(0, 0, 0, 0);
                rs = (tid < BN && gn_s < D)
                    ? static_cast<uint32_t>(*reinterpret_cast<const uint16_t*>(mx8_s + static_cast<size_t>(gn_s) * nblk_w + (k0 >> 5)))
                    : 0u;
            }
        };
        auto commit_k = [&](int buf, const uint4& rx0, const uint4& rx1, float rxd0, float rxd1, const uint4& rw0, const uint4& rw1, uint32_t rs) {
            *reinterpret_cast<uint4*>(&sXq[buf][swz64(r_x, b4_x)]) = rx0;
            if ((b4_x & 1) == 0) sXd[buf][r_x * 2 + (b4_x >> 1)] = rxd0;
            if constexpr (MI == 2) {
                *reinterpret_cast<uint4*>(&sXq[buf][swz64(r_x + 64, b4_x)]) = rx1;
                if ((b4_x & 1) == 0) sXd[buf][(r_x + 64) * 2 + (b4_x >> 1)] = rxd1;
            }
            if constexpr (CODEC_KIND == 1) {
                uint32_t w0, w1, w2, w3, w4, w5, w6, w7;
                sf4_u32_to_fp8x8(rw0.x, w0, w1);
                sf4_u32_to_fp8x8(rw0.y, w2, w3);
                sf4_u32_to_fp8x8(rw0.z, w4, w5);
                sf4_u32_to_fp8x8(rw0.w, w6, w7);
                *reinterpret_cast<uint4*>(&sWq[buf][swz64(r_w, h_w * 2 + 0)]) = make_uint4(w0, w1, w2, w3);
                *reinterpret_cast<uint4*>(&sWq[buf][swz64(r_w, h_w * 2 + 1)]) = make_uint4(w4, w5, w6, w7);
                if (tid < BN) {
                    sWs[buf][tid * 4 + 0] = s_e4m3_sc[rs & 0xFFu];
                    sWs[buf][tid * 4 + 1] = s_e4m3_sc[(rs >> 8) & 0xFFu];
                    sWs[buf][tid * 4 + 2] = s_e4m3_sc[(rs >> 16) & 0xFFu];
                    sWs[buf][tid * 4 + 3] = s_e4m3_sc[(rs >> 24) & 0xFFu];
                }
            } else {
                *reinterpret_cast<uint4*>(&sWq[buf][swz64(r_x +  0, b4_x)]) = rw0;
                *reinterpret_cast<uint4*>(&sWq[buf][swz64(r_x + 64, b4_x)]) = rw1;
                if (tid < BN) {
                    sWs[buf][tid * 2 + 0] = ue8m0_to_f32(static_cast<uint8_t>(rs & 0xFFu));
                    sWs[buf][tid * 2 + 1] = ue8m0_to_f32(static_cast<uint8_t>((rs >> 8) & 0xFFu));
                }
            }
        };

        uint4 rx0_p, rx1_p, rw0_p, rw1_p;
        float rxd0_p, rxd1_p;
        uint32_t rs_p;
        prefetch_k(0, rx0_p, rx1_p, rxd0_p, rxd1_p, rw0_p, rw1_p, rs_p);
        commit_k(0, rx0_p, rx1_p, rxd0_p, rxd1_p, rw0_p, rw1_p, rs_p);
        sync_lds();

        fv8 acc[MI][4];
#pragma unroll
        for (int mi = 0; mi < MI; ++mi)
#pragma unroll
            for (int ni = 0; ni < 4; ++ni) acc[mi][ni] = fv8_zero();

        int buf = 0;
        for (int k0 = 0; k0 < inter; k0 += BK) {
            const int kn = k0 + BK;
            if (kn < inter) prefetch_k(kn, rx0_p, rx1_p, rxd0_p, rxd1_p, rw0_p, rw1_p, rs_p);

            if (wm * (16 * MI) < tm) {
#pragma unroll
                for (int g2 = 0; g2 < 2; ++g2) {
                    float xd0[8], xd1[8];
#pragma unroll
                    for (int r = 0; r < 8; ++r) {
                        xd0[r] = sXd[buf][(wm * (16 * MI) + 8 * hi + r) * 2 + g2];
                        if constexpr (MI == 2) xd1[r] = sXd[buf][(wm * (16 * MI) + 16 + 8 * hi + r) * 2 + g2];
                    }
                    if constexpr (CODEC_KIND == 1) {
#pragma unroll
                        for (int s = 0; s < 2; ++s) {
                            const int kk = g2 * 2 + s;
                            iv2 a0 = wmma_ld_fp8_k16(&sXq[buf][swz64(wm * (16 * MI) + l16, kk)], hi);
                            iv2 b[4];
                            float ws[4];
#pragma unroll
                            for (int ni = 0; ni < 4; ++ni) {
                                const int n_row = wn * 64 + ni * 16 + l16;
                                b[ni] = wmma_ld_fp8_k16(&sWq[buf][swz64(n_row, kk)], hi);
                                ws[ni] = static_cast<float>(sWs[buf][n_row * 4 + kk]);
                            }
                            fv8 p0 = wmma_fp8_f32(a0, b[0], fv8_zero());
                            fv8 p1 = wmma_fp8_f32(a0, b[1], fv8_zero());
                            fv8 p2 = wmma_fp8_f32(a0, b[2], fv8_zero());
                            fv8 p3 = wmma_fp8_f32(a0, b[3], fv8_zero());
#pragma unroll
                            for (int r = 0; r < 8; ++r) {
                                acc[0][0][r] = fmaf(p0[r], xd0[r] * ws[0], acc[0][0][r]);
                                acc[0][1][r] = fmaf(p1[r], xd0[r] * ws[1], acc[0][1][r]);
                                acc[0][2][r] = fmaf(p2[r], xd0[r] * ws[2], acc[0][2][r]);
                                acc[0][3][r] = fmaf(p3[r], xd0[r] * ws[3], acc[0][3][r]);
                            }
                            if constexpr (MI == 2) {
                                iv2 a1 = wmma_ld_fp8_k16(&sXq[buf][swz64(wm * (16 * MI) + 16 + l16, kk)], hi);
                                fv8 q0 = wmma_fp8_f32(a1, b[0], fv8_zero());
                                fv8 q1 = wmma_fp8_f32(a1, b[1], fv8_zero());
                                fv8 q2 = wmma_fp8_f32(a1, b[2], fv8_zero());
                                fv8 q3 = wmma_fp8_f32(a1, b[3], fv8_zero());
#pragma unroll
                                for (int r = 0; r < 8; ++r) {
                                    acc[1][0][r] = fmaf(q0[r], xd1[r] * ws[0], acc[1][0][r]);
                                    acc[1][1][r] = fmaf(q1[r], xd1[r] * ws[1], acc[1][1][r]);
                                    acc[1][2][r] = fmaf(q2[r], xd1[r] * ws[2], acc[1][2][r]);
                                    acc[1][3][r] = fmaf(q3[r], xd1[r] * ws[3], acc[1][3][r]);
                                }
                            }
                        }
                    } else {
                        fv8 p0[4] = {fv8_zero(), fv8_zero(), fv8_zero(), fv8_zero()};
                        fv8 p1[4] = {fv8_zero(), fv8_zero(), fv8_zero(), fv8_zero()};
#pragma unroll
                        for (int s = 0; s < 2; ++s) {
                            const int kk = g2 * 2 + s;
                            iv2 a0 = wmma_ld_fp8_k16(&sXq[buf][swz64(wm * (16 * MI) + l16, kk)], hi);
                            iv2 b[4];
#pragma unroll
                            for (int ni = 0; ni < 4; ++ni) {
                                b[ni] = wmma_ld_fp8_k16(&sWq[buf][swz64(wn * 64 + ni * 16 + l16, kk)], hi);
                            }
#pragma unroll
                            for (int ni = 0; ni < 4; ++ni) p0[ni] = wmma_fp8_f32(a0, b[ni], p0[ni]);
                            if constexpr (MI == 2) {
                                iv2 a1 = wmma_ld_fp8_k16(&sXq[buf][swz64(wm * (16 * MI) + 16 + l16, kk)], hi);
#pragma unroll
                                for (int ni = 0; ni < 4; ++ni) p1[ni] = wmma_fp8_f32(a1, b[ni], p1[ni]);
                            }
                        }
#pragma unroll
                        for (int ni = 0; ni < 4; ++ni) {
                            const float ws = static_cast<float>(sWs[buf][(wn * 64 + ni * 16 + l16) * 2 + g2]);
#pragma unroll
                            for (int r = 0; r < 8; ++r) acc[0][ni][r] = fmaf(p0[ni][r], xd0[r] * ws, acc[0][ni][r]);
                            if constexpr (MI == 2) {
#pragma unroll
                                for (int r = 0; r < 8; ++r) acc[1][ni][r] = fmaf(p1[ni][r], xd1[r] * ws, acc[1][ni][r]);
                            }
                        }
                    }
                }
            }
            if (kn < inter) {
                const int nb = buf ^ 1;
                commit_k(nb, rx0_p, rx1_p, rxd0_p, rxd1_p, rw0_p, rw1_p, rs_p);
                sync_lds();
                buf = nb;
            }
        }

        if (wm * (16 * MI) < tm) {
#pragma unroll
            for (int ni = 0; ni < 4; ++ni) {
                const int gn = n_blk + wn * 64 + ni * 16 + l16;
                if (gn < D) {
                    for (int mi = 0; mi < MI; ++mi) {
#pragma unroll
                        for (int r = 0; r < 8; ++r) {
                            const int lm = wm * (16 * MI) + mi * 16 + 8 * hi + r;
                            if (lm < tm) bucket_y[static_cast<size_t>(beg + t0 + lm) * D + gn] = acc[mi][ni][r] * sf4_outer;
                        }
                    }
                }
            }
        }
        sync_lds();
    }
#else
    (void)exps; (void)exp_off; (void)tiles; (void)meta;
    (void)act_fp8; (void)act_d32; (void)bucket_y; (void)D; (void)inter;
#endif
}

// =================================================================================================
// 6. FAST-Mode Double-Buffered Persistent Active-Tile BM=16, BN=128, BK=64 FP8 WMMA MoE (C64 Decode!)
//    - 21 KB LDS -> 3 WG/CU, 1 barrier per K=64, s_e4m3_sc[256] LUT, <=128 VGPRs!
// =================================================================================================
template <int CODEC_KIND>
__global__ void __launch_bounds__(128, 3) k_r9700_moe_w13_situ_fp8_bm16_persist(
    const GpuExpertDesc* __restrict__ exps,
    const int32_t* __restrict__ exp_off,
    const int32_t* __restrict__ bucket_tok,
    const MoeTile16* __restrict__ tiles,
    int32_t* __restrict__ meta,
    const uint8_t* __restrict__ X_fp8,
    const float* __restrict__ X_scale,
    uint8_t* __restrict__ act_fp8,
    float* __restrict__ act_d,
    int D, int inter, float beta, float lb) {
#if INK_R9700_HAS_FP8_WMMA
    constexpr int BM = 16, BN = 128, BK = 64;
    constexpr int N_SC = (CODEC_KIND == 1) ? 4 : 2;
    using WsT = typename std::conditional<CODEC_KIND == 1, half_t, float>::type;
    const int N = 2 * inter;
    const int n_nb = (N + BN - 1) / BN;
    const int n_tiles = meta[0];
    const int total_work = n_tiles * n_nb;
    if (total_work <= 0) return;

    __shared__ alignas(16) uint8_t sXq[2][BM * BK];   //  2 KB
    __shared__ alignas(16) uint8_t sWq[2][BN * BK];   // 16 KB
    __shared__ WsT sWs[2][BN * N_SC];                 //  2 KB
    __shared__ half_t s_e4m3_sc[256];                 //  0.5 KB
    __shared__ int32_t sTok[BM];
    __shared__ float sXsc[BM];
    __shared__ float s_am32[16 * 2];
    __shared__ int32_t s_work_id;                     // -> 20.7 KB (3 WG/CU!)

    const int tid = threadIdx.x;
    if constexpr (CODEC_KIND == 1) {
        s_e4m3_sc[tid]       = static_cast<half_t>(e4m3fn_to_f32(static_cast<uint8_t>(tid)));
        s_e4m3_sc[tid + 128] = static_cast<half_t>(e4m3fn_to_f32(static_cast<uint8_t>(tid + 128)));
        sync_lds();
    }

    const int wave = tid >> 5;
    const int lane = tid & 31;
    const int l16 = lane & 15;
    const int hi = lane >> 4;
    const int nblk_x = (CODEC_KIND == 1) ? (D >> 4) : (D >> 5);
    const int nblk_inter = (CODEC_KIND == 1) ? (inter >> 4) : (inter >> 5);

    for (;;) {
        if (tid == 0) s_work_id = atomicAdd(&meta[1], 1);
        sync_lds();
        const int wid = s_work_id;
        if (wid >= total_work) break;

        const int tidx = wid / n_nb;
        const int n_blk = (wid - tidx * n_nb) * BN;
        const MoeTile16 mt = tiles[tidx];
        const int eid = mt.eid;
        const int beg = exp_off[eid];
        const int m_e = exp_off[eid + 1] - beg;
        const int t0 = static_cast<int>(mt.m_tile) * BM;
        const int tm = (m_e - t0 < BM) ? (m_e - t0) : BM;
        const GpuExpertDesc desc = exps[eid];
        if (!desc.body || tm <= 0) continue;

        if (tid < BM) {
            const int tok = (tid < tm) ? bucket_tok[beg + t0 + tid] : 0;
            sTok[tid] = tok;
            sXsc[tid] = (tid < tm) ? X_scale[tok] : 0.0f;
        }
        sync_lds();

        const float sf4_outer = (CODEC_KIND == 1)
            ? (reinterpret_cast<const float*>(desc.body)[0] / reinterpret_cast<const float*>(desc.body)[1]) : 1.0f;
        const uint8_t* sf4_p = desc.body + 16;
        const uint8_t* sf4_s = sf4_p + static_cast<size_t>(N) * (D >> 1);
        const uint8_t* mx8_w = desc.body;
        const uint8_t* mx8_s = mx8_w + static_cast<size_t>(N) * D;

        const int r0_w = tid >> 1, h_w = tid & 1;
        const int r1_w = r0_w + 64;
        const int gn0_w = n_blk + r0_w, gn1_w = n_blk + r1_w;
        const int gn_s = n_blk + tid;

        auto prefetch_k16 = [&](int k0, uint4& rx, uint4& rw0, uint4& rw1, uint4& rw2, uint4& rw3, uint32_t& rs) {
            if (tid < 64) {
                const int r = tid >> 2, b4 = tid & 3;
                rx = (r < tm) ? *reinterpret_cast<const uint4*>(X_fp8 + static_cast<size_t>(sTok[r]) * D + k0 + b4 * 16) : make_uint4(0, 0, 0, 0);
            } else {
                rx = make_uint4(0, 0, 0, 0);
            }
            if constexpr (CODEC_KIND == 1) {
                rw0 = (gn0_w < N) ? *reinterpret_cast<const uint4*>(sf4_p + static_cast<size_t>(gn0_w) * (D >> 1) + (k0 >> 1) + h_w * 16) : make_uint4(0, 0, 0, 0);
                rw1 = (gn1_w < N) ? *reinterpret_cast<const uint4*>(sf4_p + static_cast<size_t>(gn1_w) * (D >> 1) + (k0 >> 1) + h_w * 16) : make_uint4(0, 0, 0, 0);
                rw2 = make_uint4(0, 0, 0, 0); rw3 = make_uint4(0, 0, 0, 0);
                rs = (gn_s < N) ? *reinterpret_cast<const uint32_t*>(sf4_s + static_cast<size_t>(gn_s) * nblk_x + (k0 >> 4)) : 0u;
            } else {
                const int r = tid >> 2, b4 = tid & 3;
                rw0 = (n_blk + r +  0 < N) ? *reinterpret_cast<const uint4*>(mx8_w + static_cast<size_t>(n_blk + r +  0) * D + k0 + b4 * 16) : make_uint4(0, 0, 0, 0);
                rw1 = (n_blk + r + 32 < N) ? *reinterpret_cast<const uint4*>(mx8_w + static_cast<size_t>(n_blk + r + 32) * D + k0 + b4 * 16) : make_uint4(0, 0, 0, 0);
                rw2 = (n_blk + r + 64 < N) ? *reinterpret_cast<const uint4*>(mx8_w + static_cast<size_t>(n_blk + r + 64) * D + k0 + b4 * 16) : make_uint4(0, 0, 0, 0);
                rw3 = (n_blk + r + 96 < N) ? *reinterpret_cast<const uint4*>(mx8_w + static_cast<size_t>(n_blk + r + 96) * D + k0 + b4 * 16) : make_uint4(0, 0, 0, 0);
                rs = (gn_s < N) ? static_cast<uint32_t>(*reinterpret_cast<const uint16_t*>(mx8_s + static_cast<size_t>(gn_s) * nblk_x + (k0 >> 5))) : 0u;
            }
        };
        auto commit_k16 = [&](int buf, const uint4& rx, const uint4& rw0, const uint4& rw1, const uint4& rw2, const uint4& rw3, uint32_t rs) {
            if (tid < 64) {
                const int r = tid >> 2, b4 = tid & 3;
                *reinterpret_cast<uint4*>(&sXq[buf][swz64(r, b4)]) = rx;
            }
            if constexpr (CODEC_KIND == 1) {
                uint32_t w0, w1, w2, w3, w4, w5, w6, w7;
                sf4_u32_to_fp8x8(rw0.x, w0, w1); sf4_u32_to_fp8x8(rw0.y, w2, w3);
                sf4_u32_to_fp8x8(rw0.z, w4, w5); sf4_u32_to_fp8x8(rw0.w, w6, w7);
                *reinterpret_cast<uint4*>(&sWq[buf][swz64(r0_w, h_w * 2 + 0)]) = make_uint4(w0, w1, w2, w3);
                *reinterpret_cast<uint4*>(&sWq[buf][swz64(r0_w, h_w * 2 + 1)]) = make_uint4(w4, w5, w6, w7);
                sf4_u32_to_fp8x8(rw1.x, w0, w1); sf4_u32_to_fp8x8(rw1.y, w2, w3);
                sf4_u32_to_fp8x8(rw1.z, w4, w5); sf4_u32_to_fp8x8(rw1.w, w6, w7);
                *reinterpret_cast<uint4*>(&sWq[buf][swz64(r1_w, h_w * 2 + 0)]) = make_uint4(w0, w1, w2, w3);
                *reinterpret_cast<uint4*>(&sWq[buf][swz64(r1_w, h_w * 2 + 1)]) = make_uint4(w4, w5, w6, w7);
                sWs[buf][tid * 4 + 0] = s_e4m3_sc[rs & 0xFFu];
                sWs[buf][tid * 4 + 1] = s_e4m3_sc[(rs >> 8) & 0xFFu];
                sWs[buf][tid * 4 + 2] = s_e4m3_sc[(rs >> 16) & 0xFFu];
                sWs[buf][tid * 4 + 3] = s_e4m3_sc[(rs >> 24) & 0xFFu];
            } else {
                const int r = tid >> 2, b4 = tid & 3;
                *reinterpret_cast<uint4*>(&sWq[buf][swz64(r +  0, b4)]) = rw0;
                *reinterpret_cast<uint4*>(&sWq[buf][swz64(r + 32, b4)]) = rw1;
                *reinterpret_cast<uint4*>(&sWq[buf][swz64(r + 64, b4)]) = rw2;
                *reinterpret_cast<uint4*>(&sWq[buf][swz64(r + 96, b4)]) = rw3;
                sWs[buf][tid * 2 + 0] = ue8m0_to_f32(static_cast<uint8_t>(rs & 0xFFu));
                sWs[buf][tid * 2 + 1] = ue8m0_to_f32(static_cast<uint8_t>((rs >> 8) & 0xFFu));
            }
        };

        uint4 rx_p, rw0_p, rw1_p, rw2_p, rw3_p;
        uint32_t rs_p;
        prefetch_k16(0, rx_p, rw0_p, rw1_p, rw2_p, rw3_p, rs_p);
        commit_k16(0, rx_p, rw0_p, rw1_p, rw2_p, rw3_p, rs_p);
        sync_lds();

        fv8 acc[2] = {fv8_zero(), fv8_zero()};
        int buf = 0;
        for (int k0 = 0; k0 < D; k0 += BK) {
            const int kn = k0 + BK;
            if (kn < D) prefetch_k16(kn, rx_p, rw0_p, rw1_p, rw2_p, rw3_p, rs_p);

            if constexpr (CODEC_KIND == 1) {
#pragma unroll
                for (int kk = 0; kk < 4; ++kk) {
                    iv2 a  = wmma_ld_fp8_k16(&sXq[buf][swz64(l16, kk)], hi);
                    iv2 b0 = wmma_ld_fp8_k16(&sWq[buf][swz64(wave * 32 +  0 + l16, kk)], hi);
                    iv2 b1 = wmma_ld_fp8_k16(&sWq[buf][swz64(wave * 32 + 16 + l16, kk)], hi);
                    const float ws0 = static_cast<float>(sWs[buf][(wave * 32 +  0 + l16) * 4 + kk]);
                    const float ws1 = static_cast<float>(sWs[buf][(wave * 32 + 16 + l16) * 4 + kk]);
                    fv8 p0 = wmma_fp8_f32(a, b0, fv8_zero());
                    fv8 p1 = wmma_fp8_f32(a, b1, fv8_zero());
#pragma unroll
                    for (int r = 0; r < 8; ++r) {
                        acc[0][r] = fmaf(p0[r], ws0, acc[0][r]);
                        acc[1][r] = fmaf(p1[r], ws1, acc[1][r]);
                    }
                }
            } else {
#pragma unroll
                for (int g2 = 0; g2 < 2; ++g2) {
                    fv8 p0 = fv8_zero(), p1 = fv8_zero();
#pragma unroll
                    for (int s = 0; s < 2; ++s) {
                        const int kk = g2 * 2 + s;
                        iv2 a  = wmma_ld_fp8_k16(&sXq[buf][swz64(l16, kk)], hi);
                        iv2 b0 = wmma_ld_fp8_k16(&sWq[buf][swz64(wave * 32 +  0 + l16, kk)], hi);
                        iv2 b1 = wmma_ld_fp8_k16(&sWq[buf][swz64(wave * 32 + 16 + l16, kk)], hi);
                        p0 = wmma_fp8_f32(a, b0, p0);
                        p1 = wmma_fp8_f32(a, b1, p1);
                    }
                    const float ws0 = static_cast<float>(sWs[buf][(wave * 32 +  0 + l16) * 2 + g2]);
                    const float ws1 = static_cast<float>(sWs[buf][(wave * 32 + 16 + l16) * 2 + g2]);
#pragma unroll
                    for (int r = 0; r < 8; ++r) {
                        acc[0][r] = fmaf(p0[r], ws0, acc[0][r]);
                        acc[1][r] = fmaf(p1[r], ws1, acc[1][r]);
                    }
                }
            }
            if (kn < D) {
                const int nb = buf ^ 1;
                commit_k16(nb, rx_p, rw0_p, rw1_p, rw2_p, rw3_p, rs_p);
                sync_lds();
                buf = nb;
            }
        }

        const bool is_even = ((lane & 1) == 0);
        const int gj_w = (n_blk + wave * 32) >> 1;
        if constexpr (CODEC_KIND == 1) {
#pragma nounroll
            for (int r = 0; r < 8; ++r) {
                const int lm = 8 * hi + r;
                if (lm >= tm) continue;
                const float xsc = sXsc[lm] * sf4_outer;
                const float g0 = acc[0][r] * xsc;
                const float u0 = __shfl_xor(g0, 1, 32);
                const float g1 = acc[1][r] * xsc;
                const float u1 = __shfl_xor(g1, 1, 32);
                const float v0 = is_even ? situ_f32(g0, u0, beta, lb) : 0.0f;
                const float v1 = is_even ? situ_f32(g1, u1, beta, lb) : 0.0f;
                float am = fmaxf(fabsf(v0), fabsf(v1));
                am = fmaxf(am, __shfl_xor(am, 2, 32));
                am = fmaxf(am, __shfl_xor(am, 4, 32));
                am = fmaxf(am, __shfl_xor(am, 8, 32));
                const float inv = (am > 0.0f) ? (448.0f / am) : 0.0f;
                const size_t row_out = static_cast<size_t>(beg + t0 + lm);
                if (is_even) {
                    const int j0 = gj_w + (l16 >> 1);
                    const int j1 = j0 + 8;
                    if (j0 < inter) act_fp8[row_out * inter + j0] = f32_to_e4m3fn(v0 * inv);
                    if (j1 < inter) act_fp8[row_out * inter + j1] = f32_to_e4m3fn(v1 * inv);
                    if (l16 == 0 && gj_w < inter) {
                        act_d[row_out * nblk_inter + (gj_w >> 4)] = am * (1.0f / 448.0f);
                    }
                }
            }
        } else {
            float v0r[8], v1r[8];
            const int pair = wave >> 1;
            const int half_w = wave & 1;
            if (tid < 32) s_am32[tid] = 0.0f;
            sync_lds();
#pragma nounroll
            for (int r = 0; r < 8; ++r) {
                const int lm = 8 * hi + r;
                const float xsc = (lm < tm) ? sXsc[lm] : 0.0f;
                const float g0 = acc[0][r] * xsc;
                const float u0 = __shfl_xor(g0, 1, 32);
                const float g1 = acc[1][r] * xsc;
                const float u1 = __shfl_xor(g1, 1, 32);
                v0r[r] = (is_even && lm < tm) ? situ_f32(g0, u0, beta, lb) : 0.0f;
                v1r[r] = (is_even && lm < tm) ? situ_f32(g1, u1, beta, lb) : 0.0f;
                float am = fmaxf(fabsf(v0r[r]), fabsf(v1r[r]));
                am = fmaxf(am, __shfl_xor(am, 2, 32));
                am = fmaxf(am, __shfl_xor(am, 4, 32));
                am = fmaxf(am, __shfl_xor(am, 8, 32));
                if (l16 == 0 && lm < tm && half_w == 0) s_am32[lm * 2 + pair] = am;
            }
            sync_lds();
#pragma nounroll
            for (int r = 0; r < 8; ++r) {
                const int lm = 8 * hi + r;
                float am = fmaxf(fabsf(v0r[r]), fabsf(v1r[r]));
                am = fmaxf(am, __shfl_xor(am, 2, 32));
                am = fmaxf(am, __shfl_xor(am, 4, 32));
                am = fmaxf(am, __shfl_xor(am, 8, 32));
                if (l16 == 0 && lm < tm && half_w == 1) s_am32[lm * 2 + pair] = fmaxf(s_am32[lm * 2 + pair], am);
            }
            sync_lds();
            const int gj_p = (n_blk + pair * 64) >> 1;
#pragma nounroll
            for (int r = 0; r < 8; ++r) {
                const int lm = 8 * hi + r;
                if (lm >= tm) continue;
                const float am = s_am32[lm * 2 + pair];
                const float inv = (am > 0.0f) ? (448.0f / am) : 0.0f;
                const size_t row_out = static_cast<size_t>(beg + t0 + lm);
                if (is_even) {
                    const int j0 = gj_w + (l16 >> 1);
                    const int j1 = j0 + 8;
                    if (j0 < inter) act_fp8[row_out * inter + j0] = f32_to_e4m3fn(v0r[r] * inv);
                    if (j1 < inter) act_fp8[row_out * inter + j1] = f32_to_e4m3fn(v1r[r] * inv);
                    if (half_w == 0 && l16 == 0 && gj_p < inter) {
                        act_d[row_out * nblk_inter + (gj_p >> 5)] = am * (1.0f / 448.0f);
                    }
                }
            }
        }
        sync_lds();
    }
#else
    (void)exps; (void)exp_off; (void)bucket_tok; (void)tiles; (void)meta;
    (void)X_fp8; (void)X_scale; (void)act_fp8; (void)act_d; (void)D; (void)inter; (void)beta; (void)lb;
#endif
}

template <int CODEC_KIND>
__global__ void __launch_bounds__(128, 3) k_r9700_moe_w2_fp8_bm16_persist(
    const GpuExpertDesc* __restrict__ exps,
    const int32_t* __restrict__ exp_off,
    const MoeTile16* __restrict__ tiles,
    int32_t* __restrict__ meta,
    const uint8_t* __restrict__ act_fp8,
    const float* __restrict__ act_d,
    float* __restrict__ bucket_y,
    int D, int inter) {
#if INK_R9700_HAS_FP8_WMMA
    constexpr int BM = 16, BN = 128, BK = 64;
    constexpr int N_SC = (CODEC_KIND == 1) ? 4 : 2;
    using WsT = typename std::conditional<CODEC_KIND == 1, half_t, float>::type;
    const int n_nb = (D + BN - 1) / BN;
    const int n_tiles = meta[0];
    const int total_work = n_tiles * n_nb;
    if (total_work <= 0) return;

    __shared__ alignas(16) uint8_t sXq[2][BM * BK];   //  2 KB
    __shared__ alignas(16) uint8_t sWq[2][BN * BK];   // 16 KB
    __shared__ float sXd[2][BM * N_SC];               //  0.5 KB
    __shared__ WsT sWs[2][BN * N_SC];                 //  2 KB
    __shared__ half_t s_e4m3_sc[256];                 //  0.5 KB
    __shared__ int32_t s_work_id;                     // -> 21.0 KB (3 WG/CU!)

    const int tid = threadIdx.x;
    if constexpr (CODEC_KIND == 1) {
        s_e4m3_sc[tid]       = static_cast<half_t>(e4m3fn_to_f32(static_cast<uint8_t>(tid)));
        s_e4m3_sc[tid + 128] = static_cast<half_t>(e4m3fn_to_f32(static_cast<uint8_t>(tid + 128)));
        sync_lds();
    }

    const int wave = tid >> 5;
    const int lane = tid & 31;
    const int l16 = lane & 15;
    const int hi = lane >> 4;
    const int nblk_x = (CODEC_KIND == 1) ? (inter >> 4) : (inter >> 5);

    for (;;) {
        if (tid == 0) s_work_id = atomicAdd(&meta[2], 1);
        sync_lds();
        const int wid = s_work_id;
        if (wid >= total_work) break;

        const int tidx = wid / n_nb;
        const int n_blk = (wid - tidx * n_nb) * BN;
        const MoeTile16 mt = tiles[tidx];
        const int eid = mt.eid;
        const int beg = exp_off[eid];
        const int m_e = exp_off[eid + 1] - beg;
        const int t0 = static_cast<int>(mt.m_tile) * BM;
        const int tm = (m_e - t0 < BM) ? (m_e - t0) : BM;
        const GpuExpertDesc desc = exps[eid];
        if (!desc.body || tm <= 0) continue;

        const float sf4_outer = (CODEC_KIND == 1)
            ? (reinterpret_cast<const float*>(desc.body)[2] / reinterpret_cast<const float*>(desc.body)[3]) : 1.0f;
        const uint8_t* sf4_p = desc.body + desc.dn_offset;
        const uint8_t* sf4_s = sf4_p + static_cast<size_t>(D) * (inter >> 1);
        const uint8_t* mx8_w = desc.body + desc.dn_offset;
        const uint8_t* mx8_s = mx8_w + static_cast<size_t>(D) * inter;

        const int r0_w = tid >> 1, h_w = tid & 1;
        const int r1_w = r0_w + 64;
        const int gn0_w = n_blk + r0_w, gn1_w = n_blk + r1_w;
        const int gn_s = n_blk + tid;

        auto prefetch_w2 = [&](int k0, uint4& rx, float& rxd, uint4& rw0, uint4& rw1, uint4& rw2, uint4& rw3, uint32_t& rs) {
            if (tid < 64) {
                const int r = tid >> 2, b4 = tid & 3;
                if (r < tm) {
                    const size_t row_i = static_cast<size_t>(beg + t0 + r);
                    rx = *reinterpret_cast<const uint4*>(act_fp8 + row_i * inter + k0 + b4 * 16);
                    if constexpr (CODEC_KIND == 1) rxd = act_d[row_i * nblk_x + (k0 >> 4) + b4];
                    else rxd = ((b4 & 1) == 0) ? act_d[row_i * nblk_x + (k0 >> 5) + (b4 >> 1)] : 0.0f;
                } else {
                    rx = make_uint4(0, 0, 0, 0); rxd = 0.0f;
                }
            } else {
                rx = make_uint4(0, 0, 0, 0); rxd = 0.0f;
            }
            if constexpr (CODEC_KIND == 1) {
                rw0 = (gn0_w < D) ? *reinterpret_cast<const uint4*>(sf4_p + static_cast<size_t>(gn0_w) * (inter >> 1) + (k0 >> 1) + h_w * 16) : make_uint4(0, 0, 0, 0);
                rw1 = (gn1_w < D) ? *reinterpret_cast<const uint4*>(sf4_p + static_cast<size_t>(gn1_w) * (inter >> 1) + (k0 >> 1) + h_w * 16) : make_uint4(0, 0, 0, 0);
                rw2 = make_uint4(0, 0, 0, 0); rw3 = make_uint4(0, 0, 0, 0);
                rs = (gn_s < D) ? *reinterpret_cast<const uint32_t*>(sf4_s + static_cast<size_t>(gn_s) * nblk_x + (k0 >> 4)) : 0u;
            } else {
                const int r = tid >> 2, b4 = tid & 3;
                rw0 = (n_blk + r +  0 < D) ? *reinterpret_cast<const uint4*>(mx8_w + static_cast<size_t>(n_blk + r +  0) * inter + k0 + b4 * 16) : make_uint4(0, 0, 0, 0);
                rw1 = (n_blk + r + 32 < D) ? *reinterpret_cast<const uint4*>(mx8_w + static_cast<size_t>(n_blk + r + 32) * inter + k0 + b4 * 16) : make_uint4(0, 0, 0, 0);
                rw2 = (n_blk + r + 64 < D) ? *reinterpret_cast<const uint4*>(mx8_w + static_cast<size_t>(n_blk + r + 64) * inter + k0 + b4 * 16) : make_uint4(0, 0, 0, 0);
                rw3 = (n_blk + r + 96 < D) ? *reinterpret_cast<const uint4*>(mx8_w + static_cast<size_t>(n_blk + r + 96) * inter + k0 + b4 * 16) : make_uint4(0, 0, 0, 0);
                rs = (gn_s < D) ? static_cast<uint32_t>(*reinterpret_cast<const uint16_t*>(mx8_s + static_cast<size_t>(gn_s) * nblk_x + (k0 >> 5))) : 0u;
            }
        };
        auto commit_w2 = [&](int buf, const uint4& rx, float rxd, const uint4& rw0, const uint4& rw1, const uint4& rw2, const uint4& rw3, uint32_t rs) {
            if (tid < 64) {
                const int r = tid >> 2, b4 = tid & 3;
                *reinterpret_cast<uint4*>(&sXq[buf][swz64(r, b4)]) = rx;
                if constexpr (CODEC_KIND == 1) sXd[buf][r * 4 + b4] = rxd;
                else if ((b4 & 1) == 0) sXd[buf][r * 2 + (b4 >> 1)] = rxd;
            }
            if constexpr (CODEC_KIND == 1) {
                uint32_t w0, w1, w2, w3, w4, w5, w6, w7;
                sf4_u32_to_fp8x8(rw0.x, w0, w1); sf4_u32_to_fp8x8(rw0.y, w2, w3);
                sf4_u32_to_fp8x8(rw0.z, w4, w5); sf4_u32_to_fp8x8(rw0.w, w6, w7);
                *reinterpret_cast<uint4*>(&sWq[buf][swz64(r0_w, h_w * 2 + 0)]) = make_uint4(w0, w1, w2, w3);
                *reinterpret_cast<uint4*>(&sWq[buf][swz64(r0_w, h_w * 2 + 1)]) = make_uint4(w4, w5, w6, w7);
                sf4_u32_to_fp8x8(rw1.x, w0, w1); sf4_u32_to_fp8x8(rw1.y, w2, w3);
                sf4_u32_to_fp8x8(rw1.z, w4, w5); sf4_u32_to_fp8x8(rw1.w, w6, w7);
                *reinterpret_cast<uint4*>(&sWq[buf][swz64(r1_w, h_w * 2 + 0)]) = make_uint4(w0, w1, w2, w3);
                *reinterpret_cast<uint4*>(&sWq[buf][swz64(r1_w, h_w * 2 + 1)]) = make_uint4(w4, w5, w6, w7);
                sWs[buf][tid * 4 + 0] = s_e4m3_sc[rs & 0xFFu];
                sWs[buf][tid * 4 + 1] = s_e4m3_sc[(rs >> 8) & 0xFFu];
                sWs[buf][tid * 4 + 2] = s_e4m3_sc[(rs >> 16) & 0xFFu];
                sWs[buf][tid * 4 + 3] = s_e4m3_sc[(rs >> 24) & 0xFFu];
            } else {
                const int r = tid >> 2, b4 = tid & 3;
                *reinterpret_cast<uint4*>(&sWq[buf][swz64(r +  0, b4)]) = rw0;
                *reinterpret_cast<uint4*>(&sWq[buf][swz64(r + 32, b4)]) = rw1;
                *reinterpret_cast<uint4*>(&sWq[buf][swz64(r + 64, b4)]) = rw2;
                *reinterpret_cast<uint4*>(&sWq[buf][swz64(r + 96, b4)]) = rw3;
                sWs[buf][tid * 2 + 0] = ue8m0_to_f32(static_cast<uint8_t>(rs & 0xFFu));
                sWs[buf][tid * 2 + 1] = ue8m0_to_f32(static_cast<uint8_t>((rs >> 8) & 0xFFu));
            }
        };

        uint4 rx_p, rw0_p, rw1_p, rw2_p, rw3_p;
        float rxd_p;
        uint32_t rs_p;
        prefetch_w2(0, rx_p, rxd_p, rw0_p, rw1_p, rw2_p, rw3_p, rs_p);
        commit_w2(0, rx_p, rxd_p, rw0_p, rw1_p, rw2_p, rw3_p, rs_p);
        sync_lds();

        fv8 acc[2] = {fv8_zero(), fv8_zero()};
        int buf = 0;
        for (int k0 = 0; k0 < inter; k0 += BK) {
            const int kn = k0 + BK;
            if (kn < inter) prefetch_w2(kn, rx_p, rxd_p, rw0_p, rw1_p, rw2_p, rw3_p, rs_p);

            if constexpr (CODEC_KIND == 1) {
#pragma unroll
                for (int kk = 0; kk < 4; ++kk) {
                    iv2 a  = wmma_ld_fp8_k16(&sXq[buf][swz64(l16, kk)], hi);
                    iv2 b0 = wmma_ld_fp8_k16(&sWq[buf][swz64(wave * 32 +  0 + l16, kk)], hi);
                    iv2 b1 = wmma_ld_fp8_k16(&sWq[buf][swz64(wave * 32 + 16 + l16, kk)], hi);
                    const float ws0 = static_cast<float>(sWs[buf][(wave * 32 +  0 + l16) * 4 + kk]);
                    const float ws1 = static_cast<float>(sWs[buf][(wave * 32 + 16 + l16) * 4 + kk]);
                    fv8 p0 = wmma_fp8_f32(a, b0, fv8_zero());
                    fv8 p1 = wmma_fp8_f32(a, b1, fv8_zero());
#pragma unroll
                    for (int r = 0; r < 8; ++r) {
                        const float xd = sXd[buf][(8 * hi + r) * 4 + kk];
                        acc[0][r] = fmaf(p0[r], xd * ws0, acc[0][r]);
                        acc[1][r] = fmaf(p1[r], xd * ws1, acc[1][r]);
                    }
                }
            } else {
#pragma unroll
                for (int g2 = 0; g2 < 2; ++g2) {
                    fv8 p0 = fv8_zero(), p1 = fv8_zero();
#pragma unroll
                    for (int s = 0; s < 2; ++s) {
                        const int kk = g2 * 2 + s;
                        iv2 a  = wmma_ld_fp8_k16(&sXq[buf][swz64(l16, kk)], hi);
                        iv2 b0 = wmma_ld_fp8_k16(&sWq[buf][swz64(wave * 32 +  0 + l16, kk)], hi);
                        iv2 b1 = wmma_ld_fp8_k16(&sWq[buf][swz64(wave * 32 + 16 + l16, kk)], hi);
                        p0 = wmma_fp8_f32(a, b0, p0);
                        p1 = wmma_fp8_f32(a, b1, p1);
                    }
                    const float ws0 = static_cast<float>(sWs[buf][(wave * 32 +  0 + l16) * 2 + g2]);
                    const float ws1 = static_cast<float>(sWs[buf][(wave * 32 + 16 + l16) * 2 + g2]);
#pragma unroll
                    for (int r = 0; r < 8; ++r) {
                        const float xd = sXd[buf][(8 * hi + r) * 2 + g2];
                        acc[0][r] = fmaf(p0[r], xd * ws0, acc[0][r]);
                        acc[1][r] = fmaf(p1[r], xd * ws1, acc[1][r]);
                    }
                }
            }
            if (kn < inter) {
                const int nb = buf ^ 1;
                commit_w2(nb, rx_p, rxd_p, rw0_p, rw1_p, rw2_p, rw3_p, rs_p);
                sync_lds();
                buf = nb;
            }
        }

#pragma unroll
        for (int ni = 0; ni < 2; ++ni) {
            const int gn = n_blk + wave * 32 + ni * 16 + l16;
            if (gn < D) {
#pragma unroll
                for (int r = 0; r < 8; ++r) {
                    const int lm = 8 * hi + r;
                    if (lm < tm) bucket_y[static_cast<size_t>(beg + t0 + lm) * D + gn] = acc[ni][r] * sf4_outer;
                }
            }
        }
        sync_lds();
    }
#else
    (void)exps; (void)exp_off; (void)tiles; (void)meta;
    (void)act_fp8; (void)act_d; (void)bucket_y; (void)D; (void)inter;
#endif
}

// =================================================================================================
// 6B. Exact-FP16 Persistent Active-Tile BM=16, BN=128, BK=64 WMMA MoE (C1 Tree Verify B=2..16!)
//     - Deduplicates ALL shared experts across B=2..16 tree nodes (ZERO duplicate DRAM reads!)
//     - ZERO empty-expert WGs, ZERO idle waves (all 4 waves own 32 N-cols!), coalesced uint4 SF4!
//     - 100% FP16 WMMA exactness (ZERO FP8 activation loss -> preserves 100% DSpark acceptance!)
// =================================================================================================
template <int CODEC_KIND>
__global__ void __launch_bounds__(128, 2) k_r9700_moe_w13_situ_f16_bm16_persist(
    const GpuExpertDesc* __restrict__ exps,
    const int32_t* __restrict__ exp_off,
    const int32_t* __restrict__ bucket_tok,
    const MoeTile16* __restrict__ tiles,
    int32_t* __restrict__ meta,            // [0]=n_tiles, [1]=cursor
    const half_t* __restrict__ X_f16,      // [B, D]
    half_t* __restrict__ bucket_act,       // [B * top_k, inter]
    int D, int inter, float beta, float lb) {
#if INK_R9700_DEV && INK_HAS_WMMA
    constexpr int BM = 16, BN = 128, BK = 64, LD = BK + 8;  // 72 half_t = 144 B/row
    const int N = 2 * inter;
    const int n_nb = (N + BN - 1) / BN;
    const int n_tiles = meta[0];
    const int total_work = n_tiles * n_nb;
    if (total_work <= 0) return;

    __shared__ alignas(16) half_t sX[BM * LD];     //  2,304 B
    __shared__ alignas(16) half_t sW[BN * LD];     // 18,432 B
    __shared__ uint32_t s_sf4_pair[256];           //  1,024 B (SF4)
    __shared__ float s_e4m3[256];                  //  1,024 B (SF4 scale or MX8 weight!)
    __shared__ int32_t sTok[BM];
    __shared__ int32_t s_work_id;                  // -> 22.8 KB (2 WG/CU!)

    const int tid = threadIdx.x;
    for (int i = tid; i < 256; i += 128) {
        if constexpr (CODEC_KIND == 1) {
            const hv2 hp = {sf4_e2m1_f16(static_cast<uint32_t>(i) & 15u), sf4_e2m1_f16(static_cast<uint32_t>(i) >> 4)};
            s_sf4_pair[i] = __builtin_bit_cast(uint32_t, hp);
            s_e4m3[i] = e4m3fn_to_f32(static_cast<uint8_t>(i)) * 0.0625f;
        } else {
            s_e4m3[i] = e4m3fn_to_f32(static_cast<uint8_t>(i));
        }
    }
    sync_lds();

    const int wave = tid >> 5;
    const int lane = tid & 31;
    const int l16 = lane & 15;
    const int hi = lane >> 4;
    (void)hi;
    const int nblk_x = (CODEC_KIND == 1) ? (D >> 4) : (D >> 5);

    for (;;) {
        if (tid == 0) s_work_id = atomicAdd(&meta[1], 1);
        sync_lds();
        const int wid = s_work_id;
        if (wid >= total_work) break;

        const int tidx = wid / n_nb;
        const int n_blk = (wid - tidx * n_nb) * BN;
        const MoeTile16 mt = tiles[tidx];
        const int eid = mt.eid;
        const int beg = exp_off[eid];
        const int m_e = exp_off[eid + 1] - beg;
        const int t0 = static_cast<int>(mt.m_tile) * BM;
        const int tm = (m_e - t0 < BM) ? (m_e - t0) : BM;
        const GpuExpertDesc desc = exps[eid];
        if (!desc.body || tm <= 0) continue;

        if (tid < BM) sTok[tid] = (tid < tm) ? bucket_tok[beg + t0 + tid] : 0;
        sync_lds();

        const float out_scale = (CODEC_KIND == 1)
            ? ((reinterpret_cast<const float*>(desc.body)[0] / reinterpret_cast<const float*>(desc.body)[1]) * 16.0f) : 1.0f;
        const uint8_t* sf4_p = desc.body + 16;
        const uint8_t* sf4_s = sf4_p + static_cast<size_t>(N) * (D >> 1);
        const uint8_t* mx8_w = desc.body;
        const uint8_t* mx8_s = mx8_w + static_cast<size_t>(N) * D;

        const int rx = tid >> 3, c8x = tid & 7;  // 16 rows x 8 uint4s = 128 threads -> 1 uint4/thread!
        const int r0_w = tid >> 1, h_w = tid & 1;
        const int r1_w = r0_w + 64;
        const int gn0_w = n_blk + r0_w, gn1_w = n_blk + r1_w;

        auto prefetch_f = [&](int k0, uint4& vx, uint4& w0, uint4& w1, uint4& w2, uint4& w3, uint32_t& s0, uint32_t& s1) {
            vx = (rx < tm) ? *reinterpret_cast<const uint4*>(X_f16 + static_cast<size_t>(sTok[rx]) * D + k0 + c8x * 8) : make_uint4(0, 0, 0, 0);
            if constexpr (CODEC_KIND == 1) {
                w0 = (gn0_w < N) ? *reinterpret_cast<const uint4*>(sf4_p + static_cast<size_t>(gn0_w) * (D >> 1) + (k0 >> 1) + h_w * 16) : make_uint4(0, 0, 0, 0);
                w1 = (gn1_w < N) ? *reinterpret_cast<const uint4*>(sf4_p + static_cast<size_t>(gn1_w) * (D >> 1) + (k0 >> 1) + h_w * 16) : make_uint4(0, 0, 0, 0);
                w2 = make_uint4(0, 0, 0, 0); w3 = make_uint4(0, 0, 0, 0);
                s0 = (gn0_w < N) ? static_cast<uint32_t>(*reinterpret_cast<const uint16_t*>(sf4_s + static_cast<size_t>(gn0_w) * nblk_x + (k0 >> 4) + h_w * 2)) : 0u;
                s1 = (gn1_w < N) ? static_cast<uint32_t>(*reinterpret_cast<const uint16_t*>(sf4_s + static_cast<size_t>(gn1_w) * nblk_x + (k0 >> 4) + h_w * 2)) : 0u;
            } else {
                const int r = tid >> 2, b4 = tid & 3;
                w0 = (n_blk + r +  0 < N) ? *reinterpret_cast<const uint4*>(mx8_w + static_cast<size_t>(n_blk + r +  0) * D + k0 + b4 * 16) : make_uint4(0, 0, 0, 0);
                w1 = (n_blk + r + 32 < N) ? *reinterpret_cast<const uint4*>(mx8_w + static_cast<size_t>(n_blk + r + 32) * D + k0 + b4 * 16) : make_uint4(0, 0, 0, 0);
                w2 = (n_blk + r + 64 < N) ? *reinterpret_cast<const uint4*>(mx8_w + static_cast<size_t>(n_blk + r + 64) * D + k0 + b4 * 16) : make_uint4(0, 0, 0, 0);
                w3 = (n_blk + r + 96 < N) ? *reinterpret_cast<const uint4*>(mx8_w + static_cast<size_t>(n_blk + r + 96) * D + k0 + b4 * 16) : make_uint4(0, 0, 0, 0);
                const int g2 = (k0 >> 5) + (b4 >> 1);
                s0 = ((n_blk + r +  0 < N) ? static_cast<uint32_t>(mx8_s[static_cast<size_t>(n_blk + r +  0) * nblk_x + g2]) : 0u) |
                     (((n_blk + r + 32 < N) ? static_cast<uint32_t>(mx8_s[static_cast<size_t>(n_blk + r + 32) * nblk_x + g2]) : 0u) << 8);
                s1 = ((n_blk + r + 64 < N) ? static_cast<uint32_t>(mx8_s[static_cast<size_t>(n_blk + r + 64) * nblk_x + g2]) : 0u) |
                     (((n_blk + r + 96 < N) ? static_cast<uint32_t>(mx8_s[static_cast<size_t>(n_blk + r + 96) * nblk_x + g2]) : 0u) << 8);
            }
        };
        auto commit_f = [&](const uint4& vx, const uint4& w0, const uint4& w1, const uint4& w2, const uint4& w3, uint32_t s0, uint32_t s1) {
            *reinterpret_cast<uint4*>(&sX[rx * LD + c8x * 8]) = vx;
            if constexpr (CODEC_KIND == 1) {
                auto dec32 = [&](int row, const uint4& raw, uint32_t sc16) {
                    const uint32_t sc_a = h2_splat(s_e4m3[sc16 & 0xFFu]);
                    const uint32_t sc_b = h2_splat(s_e4m3[(sc16 >> 8) & 0xFFu]);
                    half_t* dst = &sW[row * LD + h_w * 32];
                    const uint32_t ww[4] = {raw.x, raw.y, raw.z, raw.w};
#pragma unroll
                    for (int q = 0; q < 4; ++q) {
                        const uint32_t sc2 = (q < 2) ? sc_a : sc_b;
                        const uint32_t a = ww[q];
                        *reinterpret_cast<uint4*>(dst + q * 8) = make_uint4(
                            h2_mul(s_sf4_pair[a & 0xFFu], sc2),
                            h2_mul(s_sf4_pair[(a >> 8) & 0xFFu], sc2),
                            h2_mul(s_sf4_pair[(a >> 16) & 0xFFu], sc2),
                            h2_mul(s_sf4_pair[(a >> 24) & 0xFFu], sc2));
                    }
                };
                dec32(r0_w, w0, s0);
                dec32(r1_w, w1, s1);
            } else {
                const int r = tid >> 2, b4 = tid & 3;
                auto dec16 = [&](int row, const uint4& raw, uint8_t scb) {
                    const uint32_t sc2 = h2_splat(ue8m0_to_f32(scb));
                    half_t* dst = &sW[row * LD + b4 * 16];
                    const uint32_t ww[4] = {raw.x, raw.y, raw.z, raw.w};
#pragma unroll
                    for (int q = 0; q < 2; ++q) {
                        const uint32_t a0 = ww[q * 2 + 0], a1 = ww[q * 2 + 1];
                        const hv2 h0 = {static_cast<half_t>(s_e4m3[a0 & 0xFFu]), static_cast<half_t>(s_e4m3[(a0 >> 8) & 0xFFu])};
                        const hv2 h1 = {static_cast<half_t>(s_e4m3[(a0 >> 16) & 0xFFu]), static_cast<half_t>(s_e4m3[(a0 >> 24) & 0xFFu])};
                        const hv2 h2 = {static_cast<half_t>(s_e4m3[a1 & 0xFFu]), static_cast<half_t>(s_e4m3[(a1 >> 8) & 0xFFu])};
                        const hv2 h3 = {static_cast<half_t>(s_e4m3[(a1 >> 16) & 0xFFu]), static_cast<half_t>(s_e4m3[(a1 >> 24) & 0xFFu])};
                        *reinterpret_cast<uint4*>(dst + q * 8) = make_uint4(
                            h2_mul(__builtin_bit_cast(uint32_t, h0), sc2),
                            h2_mul(__builtin_bit_cast(uint32_t, h1), sc2),
                            h2_mul(__builtin_bit_cast(uint32_t, h2), sc2),
                            h2_mul(__builtin_bit_cast(uint32_t, h3), sc2));
                    }
                };
                dec16(r +  0, w0, static_cast<uint8_t>(s0 & 0xFFu));
                dec16(r + 32, w1, static_cast<uint8_t>((s0 >> 8) & 0xFFu));
                dec16(r + 64, w2, static_cast<uint8_t>(s1 & 0xFFu));
                dec16(r + 96, w3, static_cast<uint8_t>((s1 >> 8) & 0xFFu));
            }
        };

        uint4 vx_p, w0_p, w1_p, w2_p, w3_p;
        uint32_t s0_p, s1_p;
        prefetch_f(0, vx_p, w0_p, w1_p, w2_p, w3_p, s0_p, s1_p);
        commit_f(vx_p, w0_p, w1_p, w2_p, w3_p, s0_p, s1_p);
        sync_lds();

        fv8 acc[2] = {fv8_zero(), fv8_zero()};
        for (int k0 = 0; k0 < D; k0 += BK) {
            const int kn = k0 + BK;
            if (kn < D) prefetch_f(kn, vx_p, w0_p, w1_p, w2_p, w3_p, s0_p, s1_p);
#pragma unroll
            for (int kk = 0; kk < 4; ++kk) {
                wmma_h_t a  = wmma_ld_h(&sX[l16 * LD + kk * 16], hi);
                wmma_h_t b0 = wmma_ld_h(&sW[(wave * 32 +  0 + l16) * LD + kk * 16], hi);
                wmma_h_t b1 = wmma_ld_h(&sW[(wave * 32 + 16 + l16) * LD + kk * 16], hi);
                acc[0] = wmma_f16(a, b0, acc[0]);
                acc[1] = wmma_f16(a, b1, acc[1]);
            }
            sync_lds();
            if (kn < D) {
                commit_f(vx_p, w0_p, w1_p, w2_p, w3_p, s0_p, s1_p);
                sync_lds();
            }
        }

        const bool is_even = ((lane & 1) == 0);
#pragma unroll
        for (int ni = 0; ni < 2; ++ni) {
            const int gn = n_blk + wave * 32 + ni * 16 + wmma_acc_n(0, lane);
            const int gj = gn >> 1;
#pragma unroll
            for (int r = 0; r < 8; ++r) {
                const float g_val = acc[ni][r] * out_scale;
                const float u_val = __shfl_xor(g_val, 1, 32);
                const int lm = wmma_acc_m(r, lane);
                if (is_even && lm < tm && gj < inter) {
                    bucket_act[static_cast<size_t>(beg + t0 + lm) * inter + gj] =
                        static_cast<half_t>(situ_f32(g_val, u_val, beta, lb));
                }
            }
        }
        sync_lds();
    }
#else
    (void)exps; (void)exp_off; (void)bucket_tok; (void)tiles; (void)meta;
    (void)X_f16; (void)bucket_act; (void)D; (void)inter; (void)beta; (void)lb;
#endif
}

template <int CODEC_KIND>
__global__ void __launch_bounds__(128, 2) k_r9700_moe_w2_f16_bm16_persist(
    const GpuExpertDesc* __restrict__ exps,
    const int32_t* __restrict__ exp_off,
    const MoeTile16* __restrict__ tiles,
    int32_t* __restrict__ meta,            // [0]=n_tiles, [2]=cursor
    const half_t* __restrict__ bucket_act,
    float* __restrict__ bucket_y,
    int D, int inter) {
#if INK_R9700_DEV && INK_HAS_WMMA
    constexpr int BM = 16, BN = 128, BK = 64, LD = BK + 8;
    const int n_nb = (D + BN - 1) / BN;
    const int n_tiles = meta[0];
    const int total_work = n_tiles * n_nb;
    if (total_work <= 0) return;

    __shared__ alignas(16) half_t sX[BM * LD];
    __shared__ alignas(16) half_t sW[BN * LD];
    __shared__ uint32_t s_sf4_pair[256];
    __shared__ float s_e4m3[256];
    __shared__ int32_t s_work_id;

    const int tid = threadIdx.x;
    for (int i = tid; i < 256; i += 128) {
        if constexpr (CODEC_KIND == 1) {
            const hv2 hp = {sf4_e2m1_f16(static_cast<uint32_t>(i) & 15u), sf4_e2m1_f16(static_cast<uint32_t>(i) >> 4)};
            s_sf4_pair[i] = __builtin_bit_cast(uint32_t, hp);
            s_e4m3[i] = e4m3fn_to_f32(static_cast<uint8_t>(i)) * 0.0625f;
        } else {
            s_e4m3[i] = e4m3fn_to_f32(static_cast<uint8_t>(i));
        }
    }
    sync_lds();

    const int wave = tid >> 5;
    const int lane = tid & 31;
    const int l16 = lane & 15;
    const int hi = lane >> 4;
    (void)hi;
    const int nblk_x = (CODEC_KIND == 1) ? (inter >> 4) : (inter >> 5);

    for (;;) {
        if (tid == 0) s_work_id = atomicAdd(&meta[2], 1);
        sync_lds();
        const int wid = s_work_id;
        if (wid >= total_work) break;

        const int tidx = wid / n_nb;
        const int n_blk = (wid - tidx * n_nb) * BN;
        const MoeTile16 mt = tiles[tidx];
        const int eid = mt.eid;
        const int beg = exp_off[eid];
        const int m_e = exp_off[eid + 1] - beg;
        const int t0 = static_cast<int>(mt.m_tile) * BM;
        const int tm = (m_e - t0 < BM) ? (m_e - t0) : BM;
        const GpuExpertDesc desc = exps[eid];
        if (!desc.body || tm <= 0) continue;

        const float out_scale = (CODEC_KIND == 1)
            ? ((reinterpret_cast<const float*>(desc.body)[2] / reinterpret_cast<const float*>(desc.body)[3]) * 16.0f) : 1.0f;
        const uint8_t* sf4_p = desc.body + desc.dn_offset;
        const uint8_t* sf4_s = sf4_p + static_cast<size_t>(D) * (inter >> 1);
        const uint8_t* mx8_w = desc.body + desc.dn_offset;
        const uint8_t* mx8_s = mx8_w + static_cast<size_t>(D) * inter;

        const int rx = tid >> 3, c8x = tid & 7;
        const int r0_w = tid >> 1, h_w = tid & 1;
        const int r1_w = r0_w + 64;
        const int gn0_w = n_blk + r0_w, gn1_w = n_blk + r1_w;

        auto prefetch_f = [&](int k0, uint4& vx, uint4& w0, uint4& w1, uint4& w2, uint4& w3, uint32_t& s0, uint32_t& s1) {
            vx = (rx < tm) ? *reinterpret_cast<const uint4*>(bucket_act + static_cast<size_t>(beg + t0 + rx) * inter + k0 + c8x * 8) : make_uint4(0, 0, 0, 0);
            if constexpr (CODEC_KIND == 1) {
                w0 = (gn0_w < D) ? *reinterpret_cast<const uint4*>(sf4_p + static_cast<size_t>(gn0_w) * (inter >> 1) + (k0 >> 1) + h_w * 16) : make_uint4(0, 0, 0, 0);
                w1 = (gn1_w < D) ? *reinterpret_cast<const uint4*>(sf4_p + static_cast<size_t>(gn1_w) * (inter >> 1) + (k0 >> 1) + h_w * 16) : make_uint4(0, 0, 0, 0);
                w2 = make_uint4(0, 0, 0, 0); w3 = make_uint4(0, 0, 0, 0);
                s0 = (gn0_w < D) ? static_cast<uint32_t>(*reinterpret_cast<const uint16_t*>(sf4_s + static_cast<size_t>(gn0_w) * nblk_x + (k0 >> 4) + h_w * 2)) : 0u;
                s1 = (gn1_w < D) ? static_cast<uint32_t>(*reinterpret_cast<const uint16_t*>(sf4_s + static_cast<size_t>(gn1_w) * nblk_x + (k0 >> 4) + h_w * 2)) : 0u;
            } else {
                const int r = tid >> 2, b4 = tid & 3;
                w0 = (n_blk + r +  0 < D) ? *reinterpret_cast<const uint4*>(mx8_w + static_cast<size_t>(n_blk + r +  0) * inter + k0 + b4 * 16) : make_uint4(0, 0, 0, 0);
                w1 = (n_blk + r + 32 < D) ? *reinterpret_cast<const uint4*>(mx8_w + static_cast<size_t>(n_blk + r + 32) * inter + k0 + b4 * 16) : make_uint4(0, 0, 0, 0);
                w2 = (n_blk + r + 64 < D) ? *reinterpret_cast<const uint4*>(mx8_w + static_cast<size_t>(n_blk + r + 64) * inter + k0 + b4 * 16) : make_uint4(0, 0, 0, 0);
                w3 = (n_blk + r + 96 < D) ? *reinterpret_cast<const uint4*>(mx8_w + static_cast<size_t>(n_blk + r + 96) * inter + k0 + b4 * 16) : make_uint4(0, 0, 0, 0);
                const int g2 = (k0 >> 5) + (b4 >> 1);
                s0 = ((n_blk + r +  0 < D) ? static_cast<uint32_t>(mx8_s[static_cast<size_t>(n_blk + r +  0) * nblk_x + g2]) : 0u) |
                     (((n_blk + r + 32 < D) ? static_cast<uint32_t>(mx8_s[static_cast<size_t>(n_blk + r + 32) * nblk_x + g2]) : 0u) << 8);
                s1 = ((n_blk + r + 64 < D) ? static_cast<uint32_t>(mx8_s[static_cast<size_t>(n_blk + r + 64) * nblk_x + g2]) : 0u) |
                     (((n_blk + r + 96 < D) ? static_cast<uint32_t>(mx8_s[static_cast<size_t>(n_blk + r + 96) * nblk_x + g2]) : 0u) << 8);
            }
        };
        auto commit_f = [&](const uint4& vx, const uint4& w0, const uint4& w1, const uint4& w2, const uint4& w3, uint32_t s0, uint32_t s1) {
            *reinterpret_cast<uint4*>(&sX[rx * LD + c8x * 8]) = vx;
            if constexpr (CODEC_KIND == 1) {
                auto dec32 = [&](int row, const uint4& raw, uint32_t sc16) {
                    const uint32_t sc_a = h2_splat(s_e4m3[sc16 & 0xFFu]);
                    const uint32_t sc_b = h2_splat(s_e4m3[(sc16 >> 8) & 0xFFu]);
                    half_t* dst = &sW[row * LD + h_w * 32];
                    const uint32_t ww[4] = {raw.x, raw.y, raw.z, raw.w};
#pragma unroll
                    for (int q = 0; q < 4; ++q) {
                        const uint32_t sc2 = (q < 2) ? sc_a : sc_b;
                        const uint32_t a = ww[q];
                        *reinterpret_cast<uint4*>(dst + q * 8) = make_uint4(
                            h2_mul(s_sf4_pair[a & 0xFFu], sc2),
                            h2_mul(s_sf4_pair[(a >> 8) & 0xFFu], sc2),
                            h2_mul(s_sf4_pair[(a >> 16) & 0xFFu], sc2),
                            h2_mul(s_sf4_pair[(a >> 24) & 0xFFu], sc2));
                    }
                };
                dec32(r0_w, w0, s0);
                dec32(r1_w, w1, s1);
            } else {
                const int r = tid >> 2, b4 = tid & 3;
                auto dec16 = [&](int row, const uint4& raw, uint8_t scb) {
                    const uint32_t sc2 = h2_splat(ue8m0_to_f32(scb));
                    half_t* dst = &sW[row * LD + b4 * 16];
                    const uint32_t ww[4] = {raw.x, raw.y, raw.z, raw.w};
#pragma unroll
                    for (int q = 0; q < 2; ++q) {
                        const uint32_t a0 = ww[q * 2 + 0], a1 = ww[q * 2 + 1];
                        const hv2 h0 = {static_cast<half_t>(s_e4m3[a0 & 0xFFu]), static_cast<half_t>(s_e4m3[(a0 >> 8) & 0xFFu])};
                        const hv2 h1 = {static_cast<half_t>(s_e4m3[(a0 >> 16) & 0xFFu]), static_cast<half_t>(s_e4m3[(a0 >> 24) & 0xFFu])};
                        const hv2 h2 = {static_cast<half_t>(s_e4m3[a1 & 0xFFu]), static_cast<half_t>(s_e4m3[(a1 >> 8) & 0xFFu])};
                        const hv2 h3 = {static_cast<half_t>(s_e4m3[(a1 >> 16) & 0xFFu]), static_cast<half_t>(s_e4m3[(a1 >> 24) & 0xFFu])};
                        *reinterpret_cast<uint4*>(dst + q * 8) = make_uint4(
                            h2_mul(__builtin_bit_cast(uint32_t, h0), sc2),
                            h2_mul(__builtin_bit_cast(uint32_t, h1), sc2),
                            h2_mul(__builtin_bit_cast(uint32_t, h2), sc2),
                            h2_mul(__builtin_bit_cast(uint32_t, h3), sc2));
                    }
                };
                dec16(r +  0, w0, static_cast<uint8_t>(s0 & 0xFFu));
                dec16(r + 32, w1, static_cast<uint8_t>((s0 >> 8) & 0xFFu));
                dec16(r + 64, w2, static_cast<uint8_t>(s1 & 0xFFu));
                dec16(r + 96, w3, static_cast<uint8_t>((s1 >> 8) & 0xFFu));
            }
        };

        uint4 vx_p, w0_p, w1_p, w2_p, w3_p;
        uint32_t s0_p, s1_p;
        prefetch_f(0, vx_p, w0_p, w1_p, w2_p, w3_p, s0_p, s1_p);
        commit_f(vx_p, w0_p, w1_p, w2_p, w3_p, s0_p, s1_p);
        sync_lds();

        fv8 acc[2] = {fv8_zero(), fv8_zero()};
        for (int k0 = 0; k0 < inter; k0 += BK) {
            const int kn = k0 + BK;
            if (kn < inter) prefetch_f(kn, vx_p, w0_p, w1_p, w2_p, w3_p, s0_p, s1_p);
#pragma unroll
            for (int kk = 0; kk < 4; ++kk) {
                wmma_h_t a  = wmma_ld_h(&sX[l16 * LD + kk * 16], hi);
                wmma_h_t b0 = wmma_ld_h(&sW[(wave * 32 +  0 + l16) * LD + kk * 16], hi);
                wmma_h_t b1 = wmma_ld_h(&sW[(wave * 32 + 16 + l16) * LD + kk * 16], hi);
                acc[0] = wmma_f16(a, b0, acc[0]);
                acc[1] = wmma_f16(a, b1, acc[1]);
            }
            sync_lds();
            if (kn < inter) {
                commit_f(vx_p, w0_p, w1_p, w2_p, w3_p, s0_p, s1_p);
                sync_lds();
            }
        }

#pragma unroll
        for (int ni = 0; ni < 2; ++ni) {
            const int gn = n_blk + wave * 32 + ni * 16 + wmma_acc_n(0, lane);
            if (gn < D) {
#pragma unroll
                for (int r = 0; r < 8; ++r) {
                    const int lm = wmma_acc_m(r, lane);
                    if (lm < tm) bucket_y[static_cast<size_t>(beg + t0 + lm) * D + gn] = acc[ni][r] * out_scale;
                }
            }
        }
        sync_lds();
    }
#else
    (void)exps; (void)exp_off; (void)tiles; (void)meta;
    (void)bucket_act; (void)bucket_y; (void)D; (void)inter;
#endif
}

// =================================================================================================
// 7. Host Dispatchers
// =================================================================================================

inline void launch_r9700_fp8_linear(
    const uint8_t* W_fp8, const float* W_scale,
    const float* X_f32, uint8_t* sc_x_fp8, float* sc_x_scale,
    const float* bias, float* Y,
    int M, int N, int K, hipStream_t st) {
    if (M <= 4) {
        const dim3 grid((N + 7) / 8);
        if (M == 1)      k_r9700_fp8_gemv_mr<1><<<grid, 256, 0, st>>>(W_fp8, W_scale, X_f32, bias, Y, M, N, K);
        else if (M == 2) k_r9700_fp8_gemv_mr<2><<<grid, 256, 0, st>>>(W_fp8, W_scale, X_f32, bias, Y, M, N, K);
        else             k_r9700_fp8_gemv_mr<4><<<grid, 256, 0, st>>>(W_fp8, W_scale, X_f32, bias, Y, M, N, K);
        return;
    }
    if (M <= 16 && (K & 63) == 0) {
        const int nb64 = M * (K >> 6);
        k_r9700_quant_f32_to_fp8_64<<<(nb64 + 15) / 16, 256, 0, st>>>(X_f32, sc_x_fp8, sc_x_scale, nb64);
        const dim3 grid((N + 31) / 32, (M + 15) / 16);
        k_r9700_fp8_wmma_16x32<<<grid, 64, 0, st>>>(W_fp8, W_scale, sc_x_fp8, sc_x_scale, bias, Y, M, N, K);
    } else if (M <= 16) {
        k_r9700_quant_f32_to_fp8_row<<<M, 256, 0, st>>>(X_f32, sc_x_fp8, sc_x_scale, M, K);
        const dim3 grid((N + 127) / 128, (M + 15) / 16);
        k_r9700_fp8_wmma_16x128<<<grid, 128, 0, st>>>(W_fp8, W_scale, sc_x_fp8, sc_x_scale, bias, Y, M, N, K);
    } else if (M < 128) {
        k_r9700_quant_f32_to_fp8_row<<<M, 256, 0, st>>>(X_f32, sc_x_fp8, sc_x_scale, M, K);
        const dim3 grid((N + 127) / 128, (M + 63) / 64);
        k_r9700_fp8_wmma_64x128<<<grid, 256, 0, st>>>(W_fp8, W_scale, sc_x_fp8, sc_x_scale, bias, Y, M, N, K);
    } else {
        k_r9700_quant_f32_to_fp8_row<<<M, 256, 0, st>>>(X_f32, sc_x_fp8, sc_x_scale, M, K);
        const dim3 grid((N + 127) / 128, (M + 127) / 128);
        k_r9700_fp8_wmma_128x128<<<grid, 256, 0, st>>>(W_fp8, W_scale, sc_x_fp8, sc_x_scale, bias, Y, M, N, K);
    }
}

}  // namespace strata::inkling2::hip::r9700
