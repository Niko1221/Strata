// engine/src/inkling2/hip/ink_hip_df2.hip.h - DFlash 2 block-diffusion drafter for Inkling 2 Mini on the HIP backend (RDNA2 / RDNA3, wave32)
//
// MIT License
// Copyright (c) 2026 Niko1221 and the Strata contributors
//
// A port of the CUDA drafter (cuda/ink_dflash2.cuh, ink_dflash2_kernels.cuh) to HIP. The drafter follows tjm8874's DFlash 2 implementation for the checkpoint
// incoai/Inkling-2-Mini-DFlash2 (Apache-2.0); the selector and the top-16 extraction are adapted from it (MIT, notices/BASALT-MIT.txt).
//
// Same model as on CUDA: 5 non-causal sliding-window (2048) transformer layers (32 query / 8 KV heads of 128, SwiGLU 6912, RoPE theta 1e4), each of whose two
// sublayers is wrapped in 2-tap group-16 dynamic causal convolutions along the 8 rows of a block. Context: the target's 4 taps (visits 5, 23, 41 and the final
// normed state) -> fc (2304 x 9216) -> RMSNorm -> per layer K / V (K RMS-normed per head and roped) into a per-slot ring. Block: row 0 = the anchor token's
// embedding, rows 1..7 = the mask token's. Draft rows 1..7 are normed and scored against the target's own head (the unembed in whatever format the trunk
// keeps it): the exact top-16 and the full-vocabulary tail, then a rank-256 bilinear selector walks the 7 steps. A draft only picks what the target
// verifies: the output is the target's greedy argmax either way.
//
// Differences from the CUDA file: the matrices that cuBLAS multiplies there go through the backend's bf16 GEMM (hook), the 8-bit and 4-bit block weights use
// the portable dot4 (sdot4 on gfx1030/31/..., sudot4 on gfx11), the head scores the unembed row by row (hidden rows in LDS) and writes full fp32 logits of the 7 rows
// before the chunked top-16 (no separate int8 screen copy of the unembed: 460 MB less VRAM). SF4 block weights are not ported (the pinned bf16 checkpoint and the
// q8 / q4 sidecars load; an SF4 sidecar is refused). Nothing here changes what the target computes.
#pragma once

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <string>
#include <vector>

#include "../cuda/ink_dflash2_path.cuh"
#include "../ink_dflash2_mix.h"   // host decoder of the shipped pre-quantized (variant mix) file: SF4 -> bf16 (shared with the Metal and SYCL ports)
#include "../../image/vae/safetensors_lite.hpp"

namespace strata::ink {
using half_t = strata::inkling2::hip::half_t;
namespace hdf2 {
using namespace strata::inkling2::hip;
using namespace strata::quant::sk;

constexpr int kLayers = 5;
constexpr int kHidden = 2304;
constexpr int kHeads = 32;
constexpr int kKvHeads = 8;
constexpr int kHeadDim = 128;
constexpr int kQ = kHeads * kHeadDim;      // 4096
constexpr int kKV = kKvHeads * kHeadDim;   // 1024
constexpr int kFF = 6912;
constexpr int kDyn = 576;                  // 4 x 144 groups of the two 2-tap convolutions
constexpr int kGroup = 16;
constexpr int kGroups = kHidden / kGroup;  // 144
constexpr int kBlock = 8;
constexpr int kDraft = 7;
constexpr int kRank = 256;
constexpr int kTopK = 16;
constexpr int kWindow = 2048;
constexpr int kRingCap = kWindow + 64;
constexpr int kTargets = 4;
constexpr int kTileCells = 32;
constexpr int kMaxSplit = 16;
constexpr int kTopChunk = 4096;
constexpr float kRmsEps = 1e-6f;
constexpr int kGemvMR = 8;
constexpr int kKStride = 132;   // halves per K row in LDS

typedef _Float16 hd_h2 __attribute__((ext_vector_type(2)));

__device__ __forceinline__ float hd_bf(uint16_t u) { return bf16_to_f32(u); }
__device__ __forceinline__ float hd_softcap(float v, float cap) { return cap > 0.0f ? cap * tanhf(v / cap) : v; }

// block-wide sum for blocks of 32..256 threads (sh: 8 floats)
__device__ __forceinline__ float hd_block_sum(float v, float* sh) {
    v = wave32_sum(v);
    const int w = threadIdx.x >> 5, nw = (blockDim.x + 31) >> 5;
    if ((threadIdx.x & 31) == 0) sh[w] = v;
    __syncthreads();
    float t = 0.0f;
    for (int i = 0; i < nw; ++i) t += sh[i];
    __syncthreads();
    return t;
}

// ---- gathers, norms, quantization ------------------------------------------------------------------------------------------------

// fp32 [M][4 * 2304] holding the four tap rows rounded through bf16 (the CUDA gather stores bf16 for its GEMM)
__global__ void __launch_bounds__(256) k_hdf2_gather_taps(const float* __restrict__ t0, const float* __restrict__ t1, const float* __restrict__ t2,
                                                          const float* __restrict__ t3, int row0, int M, float* __restrict__ out) {
    const int i = blockIdx.x;
    if (i >= M) return;
    const int src = row0 + i;
    const float* tp[4] = {t0, t1, t2, t3};
    float* dst = out + (size_t)i * kTargets * kHidden;
#pragma unroll
    for (int t = 0; t < kTargets; ++t) {
        const float4* s4 = reinterpret_cast<const float4*>(tp[t] + (size_t)src * kHidden);
        float4* d4 = reinterpret_cast<float4*>(dst + t * kHidden);
        for (int d = threadIdx.x; d < kHidden / 4; d += 256) {
            const float4 v = s4[d];
            float4 o;
            o.x = hd_bf(f32_to_bf16(v.x));
            o.y = hd_bf(f32_to_bf16(v.y));
            o.z = hd_bf(f32_to_bf16(v.z));
            o.w = hd_bf(f32_to_bf16(v.w));
            d4[d] = o;
        }
    }
}

// out = x * rsqrt(mean(x^2) + eps) * w, rows of kHidden
__global__ void __launch_bounds__(256) k_hdf2_rms_norm(const float* __restrict__ x, const float* __restrict__ w, float* __restrict__ out, int rows) {
    const int r = blockIdx.x;
    if (r >= rows) return;
    __shared__ float sh[8];
    const float* xr = x + (size_t)r * kHidden;
    float ss = 0.0f;
    for (int c = threadIdx.x; c < kHidden; c += 256) ss += xr[c] * xr[c];
    ss = hd_block_sum(ss, sh);
    const float inv = rsqrtf(ss * (1.0f / (float)kHidden) + kRmsEps);
    for (int c = threadIdx.x; c < kHidden; c += 256) out[(size_t)r * kHidden + c] = xr[c] * inv * w[c];
}

// fp32 [rows][K] -> int8 codes + one fp32 scale per 32 values. One wave per 32-block, 8 blocks per CTA.
__global__ void __launch_bounds__(256) k_hdf2_quant_act(const float* __restrict__ x, int8_t* __restrict__ xq, float* __restrict__ xs, int rows, int K) {
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const int nb = K >> 5;
    const int b = blockIdx.x * 8 + warp;
    const int i = blockIdx.y;
    if (i >= rows || b >= nb) return;
    const float v = x[(size_t)i * K + (size_t)b * 32 + lane];
    const float am = wave32_max(fabsf(v));
    const float sc = am > 0.0f ? am * (1.0f / 127.0f) : 1.0f;
    const float inv = am > 0.0f ? 127.0f / am : 0.0f;
    xq[(size_t)i * K + (size_t)b * 32 + lane] = (int8_t)max(-127, min(127, __float2int_rn(v * inv)));
    if (lane == 0) xs[(size_t)i * nb + b] = sc;
}

// ---- weight quantization (bf16 checkpoint -> block weights) ------------------------------------------------------------------------
// Q8 row-scaled: one fp32 scale per row (amax / 127)
__global__ void __launch_bounds__(256) k_hdf2_bf16_to_q8row(const uint16_t* __restrict__ src, float* __restrict__ sc, int8_t* __restrict__ q, int N, int K) {
    const int n = blockIdx.x;
    if (n >= N) return;
    __shared__ float sh[8];
    float am = 0.0f;
    for (int k = threadIdx.x; k < K; k += 256) am = fmaxf(am, fabsf(hd_bf(src[(size_t)n * K + k])));
    am = wave32_max(am);
    if ((threadIdx.x & 31) == 0) sh[threadIdx.x >> 5] = am;
    __syncthreads();
    float g = sh[0];
    for (int i = 1; i < 8; ++i) g = fmaxf(g, sh[i]);
    const float s = g > 0.0f ? g / 127.0f : 1.0f;
    const float inv = g > 0.0f ? 127.0f / g : 0.0f;
    if (threadIdx.x == 0) sc[n] = s;
    for (int k = threadIdx.x; k < K; k += 256) q[(size_t)n * K + k] = (int8_t)max(-127, min(127, __float2int_rn(hd_bf(src[(size_t)n * K + k]) * inv)));
}

// 4-bit groups of 32 with an fp16 scale; byte j of a group = element j (low nibble) and j + 16 (high nibble)
__global__ void __launch_bounds__(256) k_hdf2_bf16_to_q4(const uint16_t* __restrict__ src, uint8_t* __restrict__ dst, half_t* __restrict__ sc, int N, int K) {
    const int n = blockIdx.x;
    if (n >= N) return;
    const int nb = K >> 5;
    for (int b = threadIdx.x; b < nb; b += 256) {
        const uint16_t* w = src + (size_t)n * K + (size_t)b * 32;
        float v[32];
        float am = 0.0f;
#pragma unroll
        for (int j = 0; j < 32; ++j) { v[j] = hd_bf(w[j]); am = fmaxf(am, fabsf(v[j])); }
        const float s = am > 0.0f ? am / 7.0f : 1.0f;
        const float inv = 1.0f / s;
        sc[(size_t)n * nb + b] = (half_t)s;
        uint8_t* d = dst + (size_t)n * (K / 2) + (size_t)b * 16;
#pragma unroll
        for (int j = 0; j < 16; ++j) {
            const int lo = max(-8, min(7, __float2int_rn(v[j] * inv)));
            const int hi = max(-8, min(7, __float2int_rn(v[j + 16] * inv)));
            d[j] = (uint8_t)((lo & 0xF) | ((hi & 0xF) << 4));
        }
    }
}

// ---- Q8 / 4-bit skinny GEMV: Y[m][n] = scale * sum_k W[n][k] * x[m][k], x as per-32-block int8 codes ------------------------------------
// One wave per output row, the 8 activation rows of a chunk in LDS. Block 256 = 8 output rows. Grid 1D: chunk (m / 8) varies fastest.
__global__ void __launch_bounds__(256) k_hdf2_gemv_q8(const int8_t* __restrict__ W, const float* __restrict__ wsc, const int8_t* __restrict__ xq,
                                                      const float* __restrict__ xs, float* __restrict__ Y, int M, int N, int K, int n_chunks) {
    extern __shared__ __align__(16) uint8_t g_smem[];
    const int nb = K >> 5;
    int8_t* sq = reinterpret_cast<int8_t*>(g_smem);
    float* ss = reinterpret_cast<float*>(g_smem + (size_t)kGemvMR * K);
    const int mc = blockIdx.x % n_chunks;
    const int rb = blockIdx.x / n_chunks;
    const int m0 = mc * kGemvMR;
    const int mrows = min(kGemvMR, M - m0);
    {
        const int4* src = reinterpret_cast<const int4*>(xq + (size_t)m0 * K);
        int4* dst = reinterpret_cast<int4*>(sq);
        const int n16 = mrows * K / 16;
        for (int i = threadIdx.x; i < n16; i += 256) dst[i] = src[i];
        const float* s2 = xs + (size_t)m0 * nb;
        for (int i = threadIdx.x; i < mrows * nb; i += 256) ss[i] = s2[i];
    }
    __syncthreads();
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const int n = rb * 8 + warp;
    if (n >= N) return;
    const uint4* w16 = reinterpret_cast<const uint4*>(W + (size_t)n * K);
    float acc[kGemvMR];
#pragma unroll
    for (int m = 0; m < kGemvMR; ++m) acc[m] = 0.0f;
    const int nc = K >> 4;
    for (int c = lane; c < nc; c += 32) {
        const uint4 w = w16[c];
#pragma unroll
        for (int m = 0; m < kGemvMR; ++m) {
            if (m < mrows) {
                const uint4 a = *reinterpret_cast<const uint4*>(sq + (size_t)m * K + c * 16);
                int p = sdot4_i32((int)w.x, (int)a.x, 0);
                p = sdot4_i32((int)w.y, (int)a.y, p);
                p = sdot4_i32((int)w.z, (int)a.z, p);
                p = sdot4_i32((int)w.w, (int)a.w, p);
                acc[m] += (float)p * ss[m * nb + (c >> 1)];
            }
        }
    }
    const float sc = wsc[n];
#pragma unroll
    for (int m = 0; m < kGemvMR; ++m) {
        const float s = wave32_sum(acc[m]);
        if (lane == 0 && m < mrows) Y[(size_t)(m0 + m) * N + n] = s * sc;
    }
}

__global__ void __launch_bounds__(256) k_hdf2_gemv_q4(const uint8_t* __restrict__ W, const half_t* __restrict__ wsc, const int8_t* __restrict__ xq,
                                                      const float* __restrict__ xs, float* __restrict__ Y, int M, int N, int K, int n_chunks) {
    extern __shared__ __align__(16) uint8_t g_smem[];
    const int nb = K >> 5;
    int8_t* sq = reinterpret_cast<int8_t*>(g_smem);
    float* ss = reinterpret_cast<float*>(g_smem + (size_t)kGemvMR * K);
    const int mc = blockIdx.x % n_chunks;
    const int rb = blockIdx.x / n_chunks;
    const int m0 = mc * kGemvMR;
    const int mrows = min(kGemvMR, M - m0);
    {
        const int4* src = reinterpret_cast<const int4*>(xq + (size_t)m0 * K);
        int4* dst = reinterpret_cast<int4*>(sq);
        const int n16 = mrows * K / 16;
        for (int i = threadIdx.x; i < n16; i += 256) dst[i] = src[i];
        const float* s2 = xs + (size_t)m0 * nb;
        for (int i = threadIdx.x; i < mrows * nb; i += 256) ss[i] = s2[i];
    }
    __syncthreads();
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const int n = rb * 8 + warp;
    if (n >= N) return;
    const uint4* w16 = reinterpret_cast<const uint4*>(W + (size_t)n * (K / 2));
    const half_t* wsr = wsc + (size_t)n * nb;
    float acc[kGemvMR];
#pragma unroll
    for (int m = 0; m < kGemvMR; ++m) acc[m] = 0.0f;
    for (int b = lane; b < nb; b += 32) {
        const uint4 w = w16[b];
        const float ws = (float)wsr[b] * (1.0f / 16.0f);
        const uint32_t lo0 = (w.x << 4) & 0xF0F0F0F0u, lo1 = (w.y << 4) & 0xF0F0F0F0u, lo2 = (w.z << 4) & 0xF0F0F0F0u, lo3 = (w.w << 4) & 0xF0F0F0F0u;
        const uint32_t hi0 = w.x & 0xF0F0F0F0u, hi1 = w.y & 0xF0F0F0F0u, hi2 = w.z & 0xF0F0F0F0u, hi3 = w.w & 0xF0F0F0F0u;
#pragma unroll
        for (int m = 0; m < kGemvMR; ++m) {
            if (m < mrows) {
                const uint4 a0 = *reinterpret_cast<const uint4*>(sq + (size_t)m * K + b * 32);
                const uint4 a1 = *reinterpret_cast<const uint4*>(sq + (size_t)m * K + b * 32 + 16);
                int p = sdot4_i32((int)lo0, (int)a0.x, 0);
                p = sdot4_i32((int)lo1, (int)a0.y, p);
                p = sdot4_i32((int)lo2, (int)a0.z, p);
                p = sdot4_i32((int)lo3, (int)a0.w, p);
                p = sdot4_i32((int)hi0, (int)a1.x, p);
                p = sdot4_i32((int)hi1, (int)a1.y, p);
                p = sdot4_i32((int)hi2, (int)a1.z, p);
                p = sdot4_i32((int)hi3, (int)a1.w, p);
                acc[m] += (float)p * ws * ss[m * nb + b];
            }
        }
    }
#pragma unroll
    for (int m = 0; m < kGemvMR; ++m) {
        const float s = wave32_sum(acc[m]);
        if (lane == 0 && m < mrows) Y[(size_t)(m0 + m) * N + n] = s;
    }
}

// ---- SF4 (the shipped pre-quantized file's feed-forward matrices): [N][K/2] bytes of two E2M1 (sign-magnitude, low nibble first), [N][K/16] UE4M3 block scales, one fp32 tensor scale ----
// value = e2m1 * ue4m3 * outer. E2M1 times 2 are the integers 0 1 2 3 4 6 8 12 (a 256-entry table maps a byte to two int8), dp4a-style dot with the int8 activations, then * 0.5 * outer.
__device__ __forceinline__ float hd_ue4m3(uint8_t c) {
    const int e = (c >> 3) & 15, m = c & 7;
    return e == 0 ? ldexpf((float)m, -9) : ldexpf(1.0f + (float)m * 0.125f, e - 7);
}
__global__ void __launch_bounds__(256) k_hdf2_gemv_sf4(const uint8_t* __restrict__ Wp, const uint8_t* __restrict__ Ws, float w_outer, const int8_t* __restrict__ xq,
                                                       const float* __restrict__ xs, float* __restrict__ Y, int M, int N, int K, int n_chunks) {
    extern __shared__ __align__(16) uint8_t g_smem[];
    __shared__ uint16_t lut[256];
    const int nb = K >> 5;
    int8_t* sq = reinterpret_cast<int8_t*>(g_smem);
    float* ss = reinterpret_cast<float*>(g_smem + (size_t)kGemvMR * K);
    {
        const int mag[8] = {0, 1, 2, 3, 4, 6, 8, 12};
        const int lo = threadIdx.x & 15, hi = threadIdx.x >> 4;
        const int vlo = (lo & 8) ? -mag[lo & 7] : mag[lo & 7];
        const int vhi = (hi & 8) ? -mag[hi & 7] : mag[hi & 7];
        lut[threadIdx.x] = (uint16_t)(((uint32_t)(vlo & 0xFF)) | ((uint32_t)(vhi & 0xFF) << 8));
    }
    const int mc = blockIdx.x % n_chunks;
    const int rb = blockIdx.x / n_chunks;
    const int m0 = mc * kGemvMR;
    const int mrows = min(kGemvMR, M - m0);
    {
        const int4* src = reinterpret_cast<const int4*>(xq + (size_t)m0 * K);
        int4* dst = reinterpret_cast<int4*>(sq);
        const int n16 = mrows * K / 16;
        for (int i = threadIdx.x; i < n16; i += 256) dst[i] = src[i];
        const float* s2 = xs + (size_t)m0 * nb;
        for (int i = threadIdx.x; i < mrows * nb; i += 256) ss[i] = s2[i];
    }
    __syncthreads();
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const int n = rb * 8 + warp;
    if (n >= N) return;
    const uint4* w16 = reinterpret_cast<const uint4*>(Wp + (size_t)n * (K / 2));
    const uint8_t* wsr = Ws + (size_t)n * (K / 16);
    float acc[kGemvMR];
#pragma unroll
    for (int m = 0; m < kGemvMR; ++m) acc[m] = 0.0f;
    for (int b = lane; b < nb; b += 32) {
        const uint4 w = w16[b];
        const float s0 = hd_ue4m3(wsr[2 * b]), s1 = hd_ue4m3(wsr[2 * b + 1]);
        const uint32_t wv[4] = {w.x, w.y, w.z, w.w};
        uint32_t d[8];
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            d[2 * j] = (uint32_t)lut[wv[j] & 0xFF] | ((uint32_t)lut[(wv[j] >> 8) & 0xFF] << 16);
            d[2 * j + 1] = (uint32_t)lut[(wv[j] >> 16) & 0xFF] | ((uint32_t)lut[(wv[j] >> 24) & 0xFF] << 16);
        }
#pragma unroll
        for (int m = 0; m < kGemvMR; ++m) {
            if (m < mrows) {
                const uint4 a0 = *reinterpret_cast<const uint4*>(sq + (size_t)m * K + b * 32);
                const uint4 a1 = *reinterpret_cast<const uint4*>(sq + (size_t)m * K + b * 32 + 16);
                int p0 = sdot4_i32((int)d[0], (int)a0.x, 0);
                p0 = sdot4_i32((int)d[1], (int)a0.y, p0);
                p0 = sdot4_i32((int)d[2], (int)a0.z, p0);
                p0 = sdot4_i32((int)d[3], (int)a0.w, p0);
                int p1 = sdot4_i32((int)d[4], (int)a1.x, 0);
                p1 = sdot4_i32((int)d[5], (int)a1.y, p1);
                p1 = sdot4_i32((int)d[6], (int)a1.z, p1);
                p1 = sdot4_i32((int)d[7], (int)a1.w, p1);
                acc[m] += ((float)p0 * s0 + (float)p1 * s1) * ss[m * nb + b];
            }
        }
    }
#pragma unroll
    for (int m = 0; m < kGemvMR; ++m) {
        const float s = wave32_sum(acc[m]);
        if (lane == 0 && m < mrows) Y[(size_t)(m0 + m) * N + n] = s * 0.5f * w_outer;
    }
}

// ---- convolutions ----------------------------------------------------------------------------------------------------------------

// out[r][c] = (b0[c] + dyn[g]) * z[r][c] + (t > 0 ? (b1[c] + dyn[144 + g]) * z[r-1][c] : 0); t = r % 8, g = c / 16
__global__ void __launch_bounds__(256) k_hdf2_conv_pre(const float* __restrict__ z, const float* __restrict__ dyn, const float* __restrict__ base,
                                                       float* __restrict__ out_f32, int rows) {
    const int r = blockIdx.x;
    if (r >= rows) return;
    const int t = r % kBlock;
    const float* dr = dyn + (size_t)r * kDyn;
    const float* zr = z + (size_t)r * kHidden;
    for (int c = threadIdx.x; c < kHidden; c += 256) {
        const int g = c / kGroup;
        float acc = (base[c] + dr[g]) * zr[c];
        if (t > 0) acc += (base[kHidden + c] + dr[kGroups + g]) * zr[c - kHidden];
        out_f32[(size_t)r * kHidden + c] = acc;
    }
}

// x[r][c] += (b2[c] + dyn[288 + g]) * z[r][c] + (t > 0 ? (b3[c] + dyn[432 + g]) * z[r-1][c] : 0)
__global__ void __launch_bounds__(256) k_hdf2_conv_post_add(const float* __restrict__ z, const float* __restrict__ dyn, const float* __restrict__ base,
                                                            float* __restrict__ x, int rows) {
    const int r = blockIdx.x;
    if (r >= rows) return;
    const int t = r % kBlock;
    const float* dr = dyn + (size_t)r * kDyn + 2 * kGroups;
    const float* b2 = base + 2 * kHidden;
    const float* zr = z + (size_t)r * kHidden;
    for (int c = threadIdx.x; c < kHidden; c += 256) {
        const int g = c / kGroup;
        float acc = (b2[c] + dr[g]) * zr[c];
        if (t > 0) acc += (b2[kHidden + c] + dr[kGroups + g]) * zr[c - kHidden];
        x[(size_t)r * kHidden + c] += acc;
    }
}

// ---- block inputs ----------------------------------------------------------------------------------------------------------------

// x[s * 8 + 0] = anchor_emb[s], x[s * 8 + t] = mask_row (t = 1..7)
__global__ void __launch_bounds__(256) k_hdf2_build_x(const float* __restrict__ anchor_emb, const float* __restrict__ mask_row, float* __restrict__ x, int B) {
    const int r = blockIdx.x;
    if (r >= B * kBlock) return;
    const int s = r / kBlock, t = r % kBlock;
    const float* src = t == 0 ? anchor_emb + (size_t)s * kHidden : mask_row;
    for (int d = threadIdx.x; d < kHidden; d += 256) x[(size_t)r * kHidden + d] = src[d];
}

// per-row positions of the block rows: pos0[s] + t (meta[s * 4 + 0] = slot, + 1 = pos0)
__global__ void k_hdf2_block_pos(const int32_t* __restrict__ meta, int B, int32_t* __restrict__ pos, int32_t* __restrict__ srow) {
    const int r = blockIdx.x * blockDim.x + threadIdx.x;
    if (r >= B * kBlock) return;
    pos[r] = meta[(r / kBlock) * 4 + 1] + (r % kBlock);
    srow[r] = meta[(r / kBlock) * 4 + 0];
}

__global__ void __launch_bounds__(256) k_hdf2_swiglu(const float* __restrict__ g, const float* __restrict__ u, float* __restrict__ out, size_t n) {
    const size_t i = (size_t)blockIdx.x * 256 + threadIdx.x;
    if (i < n) {
        const float gv = g[i];
        out[i] = (gv / (1.0f + __expf(-gv))) * u[i];
    }
}

// Per-head RMSNorm (weight [128] fp32) + NeoX half-split RoPE of x [M][H][128] in place; position of row m = pos[m].
__global__ void __launch_bounds__(128) k_hdf2_rmsnorm_rope(float* __restrict__ x, const float* __restrict__ nw, const int32_t* __restrict__ pos, float theta,
                                                           int M, int H) {
    const int m = blockIdx.y, h = blockIdx.x, d = threadIdx.x;
    if (m >= M || h >= H) return;
    __shared__ float s_x[kHeadDim];
    __shared__ float sh[8];
    float* xp = x + ((size_t)m * H + h) * kHeadDim;
    const float v = xp[d];
    const float ss = hd_block_sum(v * v, sh);
    const float inv = rsqrtf(ss * (1.0f / (float)kHeadDim) + kRmsEps);
    s_x[d] = v * inv * nw[d];
    __syncthreads();
    if (d < kHeadDim / 2) {
        const float freq = 1.0f / powf(theta, (float)(2 * d) * (1.0f / (float)kHeadDim));
        float sn, cs;
        sincosf((float)pos[m] * freq, &sn, &cs);
        const float x0 = s_x[d], x1 = s_x[d + kHeadDim / 2];
        xp[d] = x0 * cs - x1 * sn;
        xp[d + kHeadDim / 2] = x0 * sn + x1 * cs;
    }
}

// ---- ring writes -----------------------------------------------------------------------------------------------------------------

// rows [n][1024] fp32 K and V (K already normed and roped) -> fp16 ring cells of (slot, pos) in layer `layer`.
__global__ void __launch_bounds__(256) k_hdf2_ring_write(const float* __restrict__ kb, const float* __restrict__ vb, const int32_t* __restrict__ slots,
                                                         const int32_t* __restrict__ pos, int layer, half_t* const* __restrict__ rk_tab,
                                                         half_t* const* __restrict__ rv_tab, int n) {
    const int i = blockIdx.x;
    if (i >= n) return;
    const int s = slots[i];
    if (s < 0) return;
    half_t* rk = rk_tab[s];
    half_t* rv = rv_tab[s];
    if (!rk || !rv) return;
    const int cell = pos[i] % kRingCap;
    const size_t base = ((size_t)layer * kRingCap + cell) * kKV;
    for (int d = threadIdx.x; d < kKV; d += 256) {
        rk[base + d] = (half_t)kb[(size_t)i * kKV + d];
        rv[base + d] = (half_t)vb[(size_t)i * kKV + d];
    }
}

// ---- attention -------------------------------------------------------------------------------------------------------------------
// One CTA per (kv head, slot, split): the 4 query heads of the group x the 8 block rows = 32 query rows share every K/V cell they read.
// Row t of a block at position p sees cells [max(lo, p + t - 2047), p + 7] (not causal inside the block).
__global__ void __launch_bounds__(256) k_hdf2_attn_part(const float* __restrict__ qr, half_t* const* __restrict__ rk_tab, half_t* const* __restrict__ rv_tab,
                                                        const int32_t* __restrict__ meta, const int32_t* __restrict__ hi_ctx, int layer, int split,
                                                        float* __restrict__ part_m, float* __restrict__ part_l, float* __restrict__ part_o) {
    __shared__ __align__(16) float Qs[32][kHeadDim + 4];
    __shared__ __align__(16) half_t Ks[kTileCells][kKStride];
    __shared__ __align__(16) half_t Vs[kTileCells][kHeadDim];
    __shared__ float Ps[32][kTileCells + 1];
    const int h = blockIdx.x, b = blockIdx.y, sp = blockIdx.z;
    const int tid = threadIdx.x;
    const int slot = meta[b * 4 + 0];
    const int p = meta[b * 4 + 1];
    const int lo = meta[b * 4 + 2];
    const int hc = hi_ctx[b];
    const int kmin = max(lo, p - (kWindow - 1));
    const int kmax = p + kBlock - 1;
    const int ncell = kmax - kmin + 1;
    const int cs = (ncell + split - 1) / split;
    const int c_begin = kmin + sp * cs;
    const int c_end = min(kmax + 1, c_begin + cs);

    const float qscale = 0.08838834764831845f;
    for (int e = tid; e < 32 * kHeadDim; e += 256) {
        const int j = e / kHeadDim, d = e % kHeadDim;
        const int qh = j / kBlock, t = j % kBlock;
        Qs[j][d] = qr[(((size_t)(b * kBlock + t)) * kHeads + 4 * h + qh) * kHeadDim + d] * qscale;
    }
    const int j = tid >> 3;
    const int g = tid & 7;
    const int t_row = j % kBlock;
    const int win_lo = p + t_row - (kWindow - 1);
    float m_run = -3.0e38f, l_run = 0.0f;
    float acc[16];
#pragma unroll
    for (int i = 0; i < 16; ++i) acc[i] = 0.0f;
    const half_t* rk = rk_tab[slot];
    const half_t* rv = rv_tab[slot];
    const size_t slot_base = (size_t)layer * kRingCap;

    for (int c0 = c_begin; c0 < c_end; c0 += kTileCells) {
        __syncthreads();
        for (int e = tid; e < kTileCells * 16; e += 256) {
            const int cc = e >> 4, part = e & 15;
            const int cell = c0 + cc;
            int4 kv = make_int4(0, 0, 0, 0), vv = make_int4(0, 0, 0, 0);
            if (cell < c_end) {
                const size_t off = (slot_base + (size_t)(cell % kRingCap)) * kKV + (size_t)h * kHeadDim + part * 8;
                kv = *reinterpret_cast<const int4*>(rk + off);
                vv = *reinterpret_cast<const int4*>(rv + off);
            }
            uint32_t* kd = reinterpret_cast<uint32_t*>(&Ks[cc][part * 8]);
            kd[0] = (uint32_t)kv.x; kd[1] = (uint32_t)kv.y; kd[2] = (uint32_t)kv.z; kd[3] = (uint32_t)kv.w;
            *reinterpret_cast<int4*>(&Vs[cc][part * 8]) = vv;
        }
        __syncthreads();
        float s4[4];
#pragma unroll
        for (int i = 0; i < 4; ++i) s4[i] = 0.0f;
#pragma unroll 8
        for (int d = 0; d < kHeadDim; d += 2) {
            const float2 q2 = *reinterpret_cast<const float2*>(&Qs[j][d]);
#pragma unroll
            for (int i = 0; i < 4; ++i) {
                const hd_h2 k2 = *reinterpret_cast<const hd_h2*>(&Ks[g + 8 * i][d]);
                s4[i] = fmaf(q2.x, (float)k2[0], fmaf(q2.y, (float)k2[1], s4[i]));
            }
        }
        float tmax = -3.0e38f;
#pragma unroll
        for (int i = 0; i < 4; ++i) {
            const int cell = c0 + g + 8 * i;
            const bool ok = (cell < c_end) && (cell >= win_lo) && (cell < hc || cell >= p);
            s4[i] = ok ? s4[i] : -3.0e38f;
            tmax = fmaxf(tmax, s4[i]);
        }
#pragma unroll
        for (int o = 1; o < 8; o <<= 1) tmax = fmaxf(tmax, __shfl_xor(tmax, o, 32));
        const float m_new = fmaxf(m_run, tmax);
        const float alpha = (m_run > -1.0e38f) ? __expf(m_run - m_new) : 0.0f;
        float lsum = 0.0f;
#pragma unroll
        for (int i = 0; i < 4; ++i) {
            const float pv = (s4[i] > -1.0e38f) ? __expf(s4[i] - m_new) : 0.0f;
            Ps[j][g + 8 * i] = pv;
            lsum += pv;
        }
#pragma unroll
        for (int o = 1; o < 8; o <<= 1) lsum += __shfl_xor(lsum, o, 32);
        l_run = l_run * alpha + lsum;
        m_run = m_new;
#pragma unroll
        for (int i = 0; i < 16; ++i) acc[i] *= alpha;
        __syncthreads();
        for (int cc = 0; cc < kTileCells; ++cc) {
            const float pv = Ps[j][cc];
            if (pv != 0.0f) {
                const int4 v0 = *reinterpret_cast<const int4*>(&Vs[cc][g * 16]);
                const int4 v1 = *reinterpret_cast<const int4*>(&Vs[cc][g * 16 + 8]);
                const hd_h2* h0 = reinterpret_cast<const hd_h2*>(&v0);
                const hd_h2* h1 = reinterpret_cast<const hd_h2*>(&v1);
#pragma unroll
                for (int i = 0; i < 4; ++i) {
                    acc[2 * i] = fmaf(pv, (float)h0[i][0], acc[2 * i]);
                    acc[2 * i + 1] = fmaf(pv, (float)h0[i][1], acc[2 * i + 1]);
                    acc[8 + 2 * i] = fmaf(pv, (float)h1[i][0], acc[8 + 2 * i]);
                    acc[8 + 2 * i + 1] = fmaf(pv, (float)h1[i][1], acc[8 + 2 * i + 1]);
                }
            }
        }
    }
    const size_t pi = ((size_t)b * kKvHeads + h) * split + sp;
    if (g == 0) {
        part_m[pi * 32 + j] = m_run;
        part_l[pi * 32 + j] = l_run;
    }
    float* po = part_o + (pi * 32 + j) * kHeadDim + g * 16;
#pragma unroll
    for (int i = 0; i < 16; ++i) po[i] = acc[i];
}

// out[(b * 8 + t)][4096] <- softmax-merged partials; grid (8 kv heads, B, 32 rows), block 128 (one dim each)
__global__ void __launch_bounds__(128) k_hdf2_attn_combine(const float* __restrict__ part_m, const float* __restrict__ part_l, const float* __restrict__ part_o,
                                                           int split, float* __restrict__ out) {
    const int h = blockIdx.x, b = blockIdx.y, j = blockIdx.z, d = threadIdx.x;
    const size_t base = ((size_t)b * kKvHeads + h) * split;
    float m = -3.0e38f;
    for (int sp = 0; sp < split; ++sp) m = fmaxf(m, part_m[(base + sp) * 32 + j]);
    float l = 0.0f, o = 0.0f;
    for (int sp = 0; sp < split; ++sp) {
        const float ms = part_m[(base + sp) * 32 + j];
        if (ms > -1.0e38f) {
            const float w = __expf(ms - m);
            l += part_l[(base + sp) * 32 + j] * w;
            o += part_o[((base + sp) * 32 + j) * kHeadDim + d] * w;
        }
    }
    const int qh = j / kBlock, t = j % kBlock;
    out[((size_t)(b * kBlock + t)) * kQ + (size_t)(4 * h + qh) * kHeadDim + d] = l > 0.0f ? o / l : 0.0f;
}

// ---- head: logits of the 7 draft rows over the whole vocabulary, then top-16 and the tail, then the selector ---------------------------
__device__ __forceinline__ bool hd_better(float a, int ia, float b, int ib) { return a > b || (a == b && ia < ib); }
__device__ __forceinline__ bool hd_strict(float a, int ia, int ja, float b, int ib, int jb) { return a > b || (a == b && (ia < ib || (ia == ib && ja < jb))); }

// Raw logits (no softcap) of n (<= 7) hidden rows against the target's unembed [V][D=2304] in its own format. A persistent grid of waves; each wave takes vocabulary
// rows v = blockIdx.x * 8 + wave, + gridDim.x * 8, ... Writes logits[t * V + v].
__global__ void __launch_bounds__(256) k_hdf2_head_logits(const uint8_t* __restrict__ unemb_raw, int fmt, int vocab_rows, const float* __restrict__ hid, float* __restrict__ logits,
                                                          int n, int D, int V) {
    const int tid = threadIdx.x;
    const int wave = tid >> 5;
    const int lane = tid & 31;
    union Lds {
        half_t s_h[7 * 2304];
        uint16_t s_bf[7 * 2304];
    };
    __shared__ alignas(16) Lds u;
    const int tot = n * D;
    if (fmt == 102) {
        for (int idx = tid; idx < tot; idx += 256) u.s_bf[idx] = f32_to_bf16(hid[idx]);
    } else if (fmt != 101 && fmt != 100) {
        for (int idx = tid; idx < tot; idx += 256) u.s_h[idx] = (half_t)hid[idx];
    }
    __syncthreads();
    for (int v = blockIdx.x * 8 + wave; v < V; v += gridDim.x * 8) {
        float acc[7] = {0, 0, 0, 0, 0, 0, 0};
        if (fmt == 102) {
            const uint16_t* wr = reinterpret_cast<const uint16_t*>(unemb_raw) + (size_t)v * D;
            for (int k = lane * 8; k < D; k += 256) {
                const uint4 w8 = *reinterpret_cast<const uint4*>(wr + k);
                const float f0 = bf16_to_f32(w8.x & 0xFFFFu), f1 = bf16_to_f32(w8.x >> 16);
                const float f2 = bf16_to_f32(w8.y & 0xFFFFu), f3 = bf16_to_f32(w8.y >> 16);
                const float f4 = bf16_to_f32(w8.z & 0xFFFFu), f5 = bf16_to_f32(w8.z >> 16);
                const float f6 = bf16_to_f32(w8.w & 0xFFFFu), f7 = bf16_to_f32(w8.w >> 16);
#pragma unroll
                for (int t = 0; t < 7; ++t) {
                    if (t < n) {
                        const uint4 c8 = *reinterpret_cast<const uint4*>(u.s_bf + t * 2304 + k);
                        acc[t] = fmaf(f0, bf16_to_f32(c8.x & 0xFFFFu), fmaf(f1, bf16_to_f32(c8.x >> 16), fmaf(f2, bf16_to_f32(c8.y & 0xFFFFu), fmaf(f3, bf16_to_f32(c8.y >> 16), acc[t]))));
                        acc[t] = fmaf(f4, bf16_to_f32(c8.z & 0xFFFFu), fmaf(f5, bf16_to_f32(c8.z >> 16), fmaf(f6, bf16_to_f32(c8.w & 0xFFFFu), fmaf(f7, bf16_to_f32(c8.w >> 16), acc[t]))));
                    }
                }
            }
        } else if (fmt == 101) {
            const float* wr = reinterpret_cast<const float*>(unemb_raw) + (size_t)v * D;
            for (int k = lane * 4; k < D; k += 128) {
                const float4 wv = *reinterpret_cast<const float4*>(wr + k);
#pragma unroll
                for (int t = 0; t < 7; ++t) {
                    if (t < n) {
                        const float4 cv = *reinterpret_cast<const float4*>(hid + (size_t)t * D + k);
                        acc[t] = fmaf(wv.x, cv.x, fmaf(wv.y, cv.y, fmaf(wv.z, cv.z, fmaf(wv.w, cv.w, acc[t]))));
                    }
                }
            }
        } else if (fmt == 100) {
            const float sc = reinterpret_cast<const float*>(unemb_raw)[v];
            const int8_t* wr = reinterpret_cast<const int8_t*>(unemb_raw + (size_t)vocab_rows * 4) + (size_t)v * D;
            for (int k = lane * 4; k < D; k += 128) {
                const uint32_t w4 = *reinterpret_cast<const uint32_t*>(wr + k);
                const float f0 = (float)(int8_t)(w4 & 0xFFu), f1 = (float)(int8_t)((w4 >> 8) & 0xFFu);
                const float f2 = (float)(int8_t)((w4 >> 16) & 0xFFu), f3 = (float)(int8_t)((w4 >> 24) & 0xFFu);
#pragma unroll
                for (int t = 0; t < 7; ++t) {
                    if (t < n) {
                        const float4 cv = *reinterpret_cast<const float4*>(hid + (size_t)t * D + k);
                        acc[t] = fmaf(f0, cv.x, fmaf(f1, cv.y, fmaf(f2, cv.z, fmaf(f3, cv.w, acc[t]))));
                    }
                }
            }
#pragma unroll
            for (int t = 0; t < 7; ++t) acc[t] *= sc;
        } else {
            const int sb_b = (fmt == CODE_S6_K) ? 198 : sk_sb_bytes(fmt);
            const int nsb = D >> 8;
            const uint8_t* wr = unemb_raw + (size_t)v * nsb * sb_b;
            const int j0 = lane * 8;
            const int sub = j0 >> 6;
            for (int s = 0; s < nsb; ++s) {
                const uint8_t* blk = wr + (size_t)s * sb_b;
                const float eff = decode_sk_sub_eff_i8(blk, fmt, sub);
                const uint4 wv = decode_sk_8_unscaled_f16(blk + 6, fmt, j0);
                const int base = (s << 8) + j0;
#pragma unroll
                for (int t = 0; t < 7; ++t) {
                    if (t < n) {
                        const uint4 xv = *reinterpret_cast<const uint4*>(u.s_h + t * 2304 + base);
                        float d = fdot2_f32(wv.x, xv.x, 0.0f);
                        d = fdot2_f32(wv.y, xv.y, d);
                        d = fdot2_f32(wv.z, xv.z, d);
                        d = fdot2_f32(wv.w, xv.w, d);
                        acc[t] = fmaf(d, eff, acc[t]);
                    }
                }
            }
        }
#pragma unroll
        for (int t = 0; t < 7; ++t) {
            if (t < n) {
                const float z = wave32_sum(acc[t]);
                if (lane == 0) logits[(size_t)t * V + v] = z;
            }
        }
    }
}

// Stage 1 (adapted from tjm8874's top-16 part kernel, MIT): grid (chunks, rows), 256 threads = 8 waves. Each wave extracts its top-16 by shuffles; 128 threads
// rank the 128 wave winners. Also the softcapped chunk (max, sum exp) over every column.
__global__ void __launch_bounds__(256) k_hdf2_tile_part(const float* __restrict__ logits, int V, float cap, float2* __restrict__ part, float2* __restrict__ part_lse) {
    __shared__ float w_val[8 * kTopK];
    __shared__ int w_id[8 * kTopK];
    __shared__ float lse_sh[8];
    const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
    const int r = blockIdx.y, chunk = blockIdx.x;
    const int c0 = chunk * kTopChunk;
    const float* row = logits + (size_t)r * V;
    float vals[16];
    uint32_t alive = 0u;
    float tm = -INFINITY;
#pragma unroll
    for (int j = 0; j < 16; ++j) {
        const int col = c0 + j * 256 + tid;
        if (col < V) {
            const float v = row[col];
            vals[j] = v;
            alive |= (1u << j);
            tm = fmaxf(tm, hd_softcap(v, cap));
        } else {
            vals[j] = -INFINITY;
        }
    }
    for (int k = 0; k < kTopK; ++k) {
        float bv = -INFINITY;
        int bi = 0x7fffffff, bj = -1;
#pragma unroll
        for (int j = 0; j < 16; ++j) {
            if (alive & (1u << j)) {
                const int col = c0 + j * 256 + tid;
                if (hd_better(vals[j], col, bv, bi)) { bv = vals[j]; bi = col; bj = j; }
            }
        }
        int wl = lane, wj = bj;
#pragma unroll
        for (int o = 16; o > 0; o >>= 1) {
            const float ov = __shfl_xor(bv, o, 32);
            const int oi = __shfl_xor(bi, o, 32);
            const int ol = __shfl_xor(wl, o, 32);
            const int oj = __shfl_xor(wj, o, 32);
            if (hd_better(ov, oi, bv, bi)) { bv = ov; bi = oi; wl = ol; wj = oj; }
        }
        if (lane == wl && wj >= 0 && wj < 16) alive &= ~(1u << wj);
        if (lane == 0) { w_val[warp * kTopK + k] = bv; w_id[warp * kTopK + k] = bi; }
    }
    {
        const float wm = wave32_max(tm);
        if (lane == 0) lse_sh[warp] = wm;
    }
    __syncthreads();
    float2* dst = part + ((size_t)r * 64 + chunk) * kTopK;
    if (tid < 8 * kTopK) {
        const float mv = w_val[tid];
        const int mi = w_id[tid];
        int rk = 0;
        for (int j = 0; j < 8 * kTopK; ++j)
            if (hd_strict(w_val[j], w_id[j], j, mv, mi, tid)) ++rk;
        if (rk < kTopK) dst[rk] = make_float2(mv, __int_as_float(mi));
    }
    float cm = lse_sh[0];
#pragma unroll
    for (int w = 1; w < 8; ++w) cm = fmaxf(cm, lse_sh[w]);
    float tl = 0.0f;
    if (isfinite(cm)) {
#pragma unroll
        for (int j = 0; j < 16; ++j) {
            const int col = c0 + j * 256 + tid;
            if (col < V) tl += __expf(hd_softcap(vals[j], cap) - cm);
        }
    }
    const float wl2 = wave32_sum(tl);
    __syncthreads();
    if (lane == 0) lse_sh[warp] = wl2;
    __syncthreads();
    if (tid == 0) {
        float sl = 0.0f;
#pragma unroll
        for (int w = 0; w < 8; ++w) sl += lse_sh[w];
        part_lse[(size_t)r * 64 + chunk] = make_float2(cm, sl);
    }
}

// Stage 2: grid (rows), 512 threads. Merge the n_chunks x 16 chunk winners into the exact top-16 (ids + raw logits) and the tail: the softcapped
// full-vocabulary partition function without the 16 winners (max m, sum l).
__global__ void __launch_bounds__(512) k_hdf2_top16_merge(const float2* __restrict__ part, const float2* __restrict__ part_lse, int n_chunks, int V, float cap,
                                                          int32_t* __restrict__ cand_out, float* __restrict__ tail_out) {
    __shared__ float all_v[64 * kTopK];
    __shared__ int all_id[64 * kTopK];
    __shared__ float win_v[kTopK];
    __shared__ float m_all_sh, l_all_sh;
    const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
    const int r = blockIdx.x;
    const int n_cand = n_chunks * kTopK;
    const float2* src = part + (size_t)r * 64 * kTopK;
    int id0, id1;
    float v0, v1;
    auto load_slot = [&](int idx, float& vo, int& io) {
        if (idx < n_cand) {
            const float2 pr = src[idx];
            const int id = __float_as_int(pr.y);
            if (id >= 0 && id < V) { vo = pr.x; io = id; } else { vo = -INFINITY; io = 0x7fffffff; }
            all_v[idx] = vo;
            all_id[idx] = io;
        } else {
            vo = -INFINITY;
            io = 0x7fffffff;
        }
    };
    load_slot(tid, v0, id0);
    load_slot(tid + 512, v1, id1);
    if (warp == 15) {
        const float2* lr = part_lse + (size_t)r * 64;
        const int c0 = lane, c1 = lane + 32;
        const float2 p0 = c0 < n_chunks ? lr[c0] : make_float2(-INFINITY, 0.0f);
        const float2 p1 = c1 < n_chunks ? lr[c1] : make_float2(-INFINITY, 0.0f);
        const float m_all = wave32_max(fmaxf(p0.x, p1.x));
        float l_loc = 0.0f;
        if (isfinite(m_all)) {
            if (c0 < n_chunks && isfinite(p0.x)) l_loc += p0.y * __expf(p0.x - m_all);
            if (c1 < n_chunks && isfinite(p1.x)) l_loc += p1.y * __expf(p1.x - m_all);
        }
        const float l_all = wave32_sum(l_loc);
        if (lane == 0) { m_all_sh = m_all; l_all_sh = l_all; }
    }
    __syncthreads();
    int rk0 = 0, rk1 = 0;
    for (int j = 0; j < n_cand; ++j) {
        const float vj = all_v[j];
        const int ij = all_id[j];
        if (tid < n_cand && hd_strict(vj, ij, j, v0, id0, tid)) ++rk0;
        if (tid + 512 < n_cand && hd_strict(vj, ij, j, v1, id1, tid + 512)) ++rk1;
    }
    int32_t* out = cand_out + (size_t)r * 2 * kTopK;
    if (tid < n_cand && rk0 < kTopK) { out[rk0] = id0; out[kTopK + rk0] = __float_as_int(v0); win_v[rk0] = v0; }
    if (tid + 512 < n_cand && rk1 < kTopK) { out[rk1] = id1; out[kTopK + rk1] = __float_as_int(v1); win_v[rk1] = v1; }
    __syncthreads();
    if (warp == 0) {
        const float m_all = m_all_sh;
        const float u = lane < kTopK ? win_v[lane] : -INFINITY;
        const float term = (isfinite(u) && isfinite(m_all)) ? __expf(hd_softcap(u, cap) - m_all) : 0.0f;
        const float s16 = wave32_sum(term);
        if (lane == 0) {
            tail_out[2 * r] = m_all;
            tail_out[2 * r + 1] = fmaxf(0.0f, l_all_sh - s16);
        }
    }
}

// The bilinear candidate selector (adapted from tjm8874's select kernel, MIT): one CTA of 512 threads (16 waves = 16 candidates) per slot, the 7 draft steps one after
// the other: score(c) = raw(c) + sum_k pred[prev][k] * g[k] * succ[c][k]; the best candidate is the step's token and the next step's prev; conf = softmax probability of
// the winner over the softcapped scores plus the tail. pred / succ: bf16 [V][256]. meta[b * 4 + 3] = the anchor token (the first step's prev).
__global__ void __launch_bounds__(512) k_hdf2_select(const int32_t* __restrict__ meta, const int32_t* __restrict__ cand, const float* __restrict__ tail,
                                                     const float* __restrict__ g, const uint16_t* __restrict__ pred, const uint16_t* __restrict__ succ,
                                                     int n_drafts, int nv, float cap, int32_t* __restrict__ ids_out, float* __restrict__ conf_out) {
    __shared__ __align__(16) float pg[kRank];
    __shared__ float sc_sh[kTopK];
    __shared__ int32_t prev_s;
    const int s = blockIdx.x;
    const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
    for (int d = 0; d < n_drafts; ++d) {
        const int r = s * n_drafts + d;
        const int32_t prev = d == 0 ? meta[s * 4 + 3] : prev_s;
        if (tid < kRank) pg[tid] = (prev >= 0 && prev < nv) ? hd_bf(pred[(size_t)prev * kRank + tid]) * g[((size_t)s * kBlock + 1 + d) * kRank + tid] : 0.0f;
        __syncthreads();
        const int32_t id = cand[(size_t)r * 2 * kTopK + warp];
        float dot = 0.0f;
        if (id >= 0) {
            const uint4 sv = *(reinterpret_cast<const uint4*>(succ + (size_t)id * kRank) + lane);
            const float4 p0 = reinterpret_cast<const float4*>(pg)[2 * lane];
            const float4 p1 = reinterpret_cast<const float4*>(pg)[2 * lane + 1];
            dot += p0.x * hd_bf((uint16_t)(sv.x & 0xffffu));
            dot += p0.y * hd_bf((uint16_t)(sv.x >> 16));
            dot += p0.z * hd_bf((uint16_t)(sv.y & 0xffffu));
            dot += p0.w * hd_bf((uint16_t)(sv.y >> 16));
            dot += p1.x * hd_bf((uint16_t)(sv.z & 0xffffu));
            dot += p1.y * hd_bf((uint16_t)(sv.z >> 16));
            dot += p1.z * hd_bf((uint16_t)(sv.w & 0xffffu));
            dot += p1.w * hd_bf((uint16_t)(sv.w >> 16));
            dot = wave32_sum(dot);
        }
        if (lane == 0) sc_sh[warp] = __int_as_float(cand[(size_t)r * 2 * kTopK + kTopK + warp]) + dot;
        __syncthreads();
        if (warp == 0) {
            float bv = lane < kTopK ? sc_sh[lane] : -INFINITY;
            int32_t bi = lane < kTopK ? cand[(size_t)r * 2 * kTopK + lane] : 0x7fffffff;
            const float my_sc = bv;
#pragma unroll
            for (int o = 8; o > 0; o >>= 1) {
                const float ov = __shfl_xor(bv, o, 32);
                const int32_t oi = __shfl_xor(bi, o, 32);
                if (hd_better(ov, oi, bv, bi)) { bv = ov; bi = oi; }
            }
            const float top = hd_softcap(bv, cap);
            float m_ref = top, tail_term = 0.0f;
            const float m_t = tail[2 * r];
            const float l_t = tail[2 * r + 1];
            if (isfinite(m_t) && l_t > 0.0f) {
                m_ref = fmaxf(top, m_t);
                tail_term = l_t * __expf(m_t - m_ref);
            }
            const float my_cap = lane < kTopK ? hd_softcap(my_sc, cap) : -INFINITY;
            const float z16 = wave32_sum(lane < kTopK ? __expf(my_cap - m_ref) : 0.0f);
            if (lane == 0) {
                ids_out[s * n_drafts + d] = bi;
                conf_out[s * n_drafts + d] = __expf(top - m_ref) / (z16 + tail_term);
                prev_s = bi;
            }
        }
        __syncthreads();
    }
}

}  // namespace hdf2

// What the drafter needs from the backend (the backend fills it; nothing here reaches into its private state).
struct HipDf2Hooks {
    hipStream_t stream = nullptr;
    // Y[M][N] fp32 = X[M][K] fp32 * W[N][K]^T, W bf16 (the backend's bf16 GEMM: hipBLASLt where it has kernels, its own WMMA / plain kernels elsewhere)
    std::function<void(const uint16_t* W, const float* X, float* Y, int M, int N, int K)> gemm;
    // out[n][2304] fp32 = embed_norm(embedding of the n tokens in d_tok (device int32))
    std::function<void(const int32_t* d_tok, float* out, int n)> embed;
    const uint8_t* unemb = nullptr;   // the trunk's unembed in its own format
    int unemb_fmt = 0;
    int unemb_rows = 0;
    int vocab = 0;                    // unpadded vocabulary (200058)
    float softcap = 30.0f;
    bool small_vram = false;          // < 24 GB: codebooks in mapped host memory, rings lazily
};

struct InkHipDFlash2 {
    HipDf2Hooks h;
    bool ready = false;
    int V = 0;
    int max_slots = 1;
    float cap = 30.0f;
    float theta = 10000.0f;
    int mask_id = 200056;
    bool host_codebooks = false;
    double conf_pow = 1.0;
    Df2Variant var;
    size_t bytes_weights = 0, bytes_scratch = 0;
    static constexpr int kCh = 128;   // ingest rows per pass

    struct Layer {
        void *wq = nullptr, *wo = nullptr, *wg = nullptr, *wu = nullptr, *wd = nullptr;
        void *sq = nullptr, *so = nullptr, *sg = nullptr, *su = nullptr, *sd = nullptr;
        int kq = 1, ko = 1, kg = 1, ku = 1, kd = 1;   // 0 bf16, 1 Q8, 2 4-bit groups
        uint16_t *wkv = nullptr, *conv_a = nullptr, *conv_f = nullptr;
        float *base_a = nullptr, *base_f = nullptr, *attn_norm = nullptr, *post_norm = nullptr, *q_norm = nullptr, *k_norm = nullptr;
        float oq = 1, oo = 1, og = 1, ou = 1, od = 1;   // SF4 tensor scales (kind 3)
    } L[hdf2::kLayers];
    uint16_t *fc = nullptr, *sel_h = nullptr, *pred = nullptr, *succ = nullptr;
    float *hidden_norm = nullptr, *norm = nullptr, *mask_row = nullptr;
    void *pred_raw = nullptr, *succ_raw = nullptr;

    // per-slot rings (allocated at the first use of a slot)
    std::vector<half_t*> ring_k_h, ring_v_h;
    half_t** d_rk_tab = nullptr;
    half_t** d_rv_tab = nullptr;
    std::vector<int64_t> lo, hi;
    std::vector<char> slot_dead;

    // block scratch (one slot: 8 rows)
    int32_t *d_meta = nullptr, *d_pos = nullptr, *d_srow = nullptr, *d_tok = nullptr, *d_mask_tok = nullptr;
    float *d_anchor_emb = nullptr, *d_x = nullptr, *d_xn = nullptr, *d_c = nullptr, *d_dyn = nullptr, *d_q = nullptr, *d_kb = nullptr, *d_vb = nullptr;
    float *d_attn = nullptr, *d_z = nullptr, *d_g = nullptr, *d_u = nullptr, *d_act = nullptr, *d_hd = nullptr, *d_gsel = nullptr;
    int8_t* d_aq = nullptr;
    float* d_as = nullptr;
    float *p_m = nullptr, *p_l = nullptr, *p_o = nullptr;
    float* d_logits = nullptr;
    float2 *d_part = nullptr, *d_part_lse = nullptr;
    int32_t* d_cand = nullptr;
    float* d_tail = nullptr;
    int32_t* d_ids = nullptr;
    float* d_conf = nullptr;
    int32_t* h_meta = nullptr;   // pinned
    int32_t* h_ids = nullptr;
    float* h_conf = nullptr;
    // ingest scratch
    float *d_ix = nullptr, *d_ic = nullptr, *d_icn = nullptr, *d_ikc = nullptr, *d_ivc = nullptr;
    long n_drafts_run = 0, n_ingest_rows = 0;
    bool prof = std::getenv("STRATA_INK_DFLASH2_PROF") != nullptr;
    double prof_draft_ms = 0;

    std::vector<void*> owned;
    template <typename T>
    T* dal(size_t n, bool scratch) {
        void* p = nullptr;
        if (hipMalloc(&p, n * sizeof(T)) != hipSuccess) { (void)hipGetLastError(); alloc_fail = true; return nullptr; }
        (scratch ? bytes_scratch : bytes_weights) += n * sizeof(T);
        owned.push_back(p);
        return (T*)p;
    }
    bool alloc_fail = false;
    template <typename T>
    T* wal(size_t n) { return dal<T>(n, false); }
    template <typename T>
    T* sal(size_t n) { return dal<T>(n, true); }

#define HDF2_CK(x)                                                                                                       \
    do {                                                                                                                 \
        const hipError_t e__ = (x);                                                                                      \
        if (e__ != hipSuccess) { err = std::string("hip: ") + hipGetErrorString(e__) + " at " #x; return false; }        \
    } while (0)

    static bool requested() { return df2_requested(); }

    // ---- weights ----------------------------------------------------------------------------------------------------------
    bool load(const std::string& path, std::string& err) {
        using namespace hdf2;
        strata::image::StFile st;
        const std::string he = strata::image::st_read_header(path, st);
        if (!he.empty()) { err = "dflash2 file: " + he; return false; }
        V = h.vocab;
        cap = h.softcap;
        if (!h.unemb || V < 4096) { err = "the DFlash 2 drafter needs the trunk's unembed"; return false; }
        std::vector<unsigned char> host;
        auto read_t = [&](const std::string& name, size_t want_bytes, const char* dtype, bool prefix_ok) -> bool {
            auto it = st.tensors.find(name);
            if (it == st.tensors.end()) { err = "dflash2 file: tensor " + name + " is missing"; return false; }
            const size_t have = (size_t)(it->second.end - it->second.begin);
            if ((dtype && it->second.dtype != dtype) || (prefix_ok ? have < want_bytes : have != want_bytes)) {
                err = "dflash2 file: tensor " + name + " has another type or size";
                return false;
            }
            const std::string e = strata::image::st_read_tensor(path, st, it->second, host);
            if (!e.empty()) { err = "dflash2 file: " + e; return false; }
            return true;
        };
        auto up_bf16 = [&](const std::string& name, size_t n, uint16_t*& dst, size_t dst_off = 0, bool alloc = true, size_t alloc_n = 0) -> bool {
            if (!read_t(name, n * 2, "BF16", false)) return false;
            if (alloc) dst = wal<uint16_t>(alloc_n ? alloc_n : n);
            if (!dst) { err = "out of GPU memory for the DFlash 2 weights"; return false; }
            HDF2_CK(hipMemcpy(dst + dst_off, host.data(), n * 2, hipMemcpyHostToDevice));
            return true;
        };
        auto up_f32 = [&](const std::string& name, size_t n, float*& dst) -> bool {
            auto itf = st.tensors.find(name);
            std::vector<float> f(n);
            if (itf != st.tensors.end() && itf->second.dtype == "F32") {
                if (!read_t(name, n * 4, "F32", false)) return false;
                std::memcpy(f.data(), host.data(), n * 4);
            } else {
                if (!read_t(name, n * 2, "BF16", false)) return false;
                const uint16_t* b = reinterpret_cast<const uint16_t*>(host.data());
                for (size_t i = 0; i < n; ++i) { uint32_t u = (uint32_t)b[i] << 16; std::memcpy(&f[i], &u, 4); }
            }
            dst = wal<float>(n);
            if (!dst) { err = "out of GPU memory for the DFlash 2 weights"; return false; }
            HDF2_CK(hipMemcpy(dst, f.data(), n * 4, hipMemcpyHostToDevice));
            return true;
        };
        {
            var = df2_variant(h.small_vram);
            // HIP default: Q8 block weights on every card (333 MiB, no loss against bf16 that the acceptance shows); bf16 / q4 by STRATA_INK_DFLASH2_WQ
            if (!std::getenv("STRATA_INK_DFLASH2_WQ") && !std::getenv("STRATA_INK_DFLASH2_W4") && df2_file_variant(df2_resolve_path(df2_model_path())).empty())
            { var.attn = 1; var.ffn = h.small_vram ? 2 : 1; }   // cards below 24 GB: 8-bit attention, 4-bit feed-forward (-105 MiB); larger cards: Q8
            // SF4 block weights exist only in the shipped pre-quantized file (the loader below follows its tensors); from the bf16 checkpoint there is no SF4 conversion on HIP
            if ((var.attn == 3 || var.ffn == 3) && df2_file_variant(df2_resolve_path(df2_model_path())).empty()) {
                err = "HIP DFlash 2 has no SF4 conversion of the bf16 checkpoint: use STRATA_INK_DFLASH2_WQ=q8 | q4 | bf16, or the pre-quantized sidecar";
                return false;
            }
        }
        // STRATA_INK_DFLASH2_SF4=bf16: widen the file's SF4 matrices on the host to bf16 (shared decoder ink_dflash2_mix.h) instead of running them as SF4 (about 360 MiB more on the card)
        const bool sf4_widen = std::getenv("STRATA_INK_DFLASH2_SF4") && std::string(std::getenv("STRATA_INK_DFLASH2_SF4")) == "bf16";
        uint16_t* tmp = nullptr;   // the largest matrix, bf16, before it is quantized
        if (hipMalloc((void**)&tmp, (size_t)kFF * kHidden * 2) != hipSuccess) { err = "out of GPU memory for the DFlash 2 loader"; return false; }
        auto up_q = [&](const std::string& name, int N, int K, int& kind, void*& q, void*& sc, float& outer) -> bool {
            if (st.tensors.count(name + ".q")) {   // pre-quantized sidecar: the format follows the tensors
                const auto& ts = st.tensors.at(name + ".s");
                if (st.tensors.count(name + ".o")) {   // SF4: .q U8 [N*K/2], .s U8 [N*K/16], .o F32 [1]
                    const size_t qb = (size_t)N * K / 2, sb = (size_t)N * K / 16;
                    if (!read_t(name + ".o", 4, "F32", false)) return false;
                    std::memcpy(&outer, host.data(), 4);
                    if (sf4_widen) {   // exact bf16 of the SF4 values (host), the bf16 path of the GEMM hook
                        std::vector<unsigned char> qh;
                        if (!read_t(name + ".q", qb, "U8", false)) return false;
                        qh = host;
                        if (!read_t(name + ".s", sb, "U8", false)) return false;
                        std::vector<uint16_t> wide((size_t)N * K);
                        ink2::df2mix::sf4_to_bf16(qh.data(), host.data(), outer, N, K, wide.data());
                        kind = 0;
                        q = wal<uint16_t>((size_t)N * K);
                        sc = nullptr;
                        if (!q) { err = "out of GPU memory for the DFlash 2 weights"; return false; }
                        HDF2_CK(hipMemcpy(q, wide.data(), (size_t)N * K * 2, hipMemcpyHostToDevice));
                        return true;
                    }
                    kind = 3;
                    if (!read_t(name + ".q", qb, "U8", false)) return false;
                    q = wal<uint8_t>(qb);
                    if (!q) { err = "out of GPU memory for the DFlash 2 weights"; return false; }
                    HDF2_CK(hipMemcpy(q, host.data(), qb, hipMemcpyHostToDevice));
                    if (!read_t(name + ".s", sb, "U8", false)) return false;
                    sc = wal<uint8_t>(sb);
                    if (!sc) { err = "out of GPU memory for the DFlash 2 weights"; return false; }
                    HDF2_CK(hipMemcpy(sc, host.data(), sb, hipMemcpyHostToDevice));
                    return true;
                }
                kind = (ts.dtype == "F16") ? 2 : 1;
                const size_t qb = kind == 1 ? (size_t)N * K : (size_t)N * K / 2;
                const size_t sb = kind == 1 ? (size_t)N * 4 : (size_t)N * K / 16;
                if (!read_t(name + ".q", qb, kind == 1 ? "I8" : "U8", false)) return false;
                q = wal<uint8_t>(qb);
                if (!q) { err = "out of GPU memory for the DFlash 2 weights"; return false; }
                HDF2_CK(hipMemcpy(q, host.data(), qb, hipMemcpyHostToDevice));
                if (!read_t(name + ".s", sb, kind == 1 ? "F32" : "F16", false)) return false;
                sc = wal<uint8_t>(sb);
                if (!sc) { err = "out of GPU memory for the DFlash 2 weights"; return false; }
                HDF2_CK(hipMemcpy(sc, host.data(), sb, hipMemcpyHostToDevice));
                return true;
            }
            if (!read_t(name, (size_t)N * K * 2, "BF16", false)) return false;
            HDF2_CK(hipMemcpy(tmp, host.data(), (size_t)N * K * 2, hipMemcpyHostToDevice));
            if (kind == 0) {
                q = wal<uint16_t>((size_t)N * K);
                sc = nullptr;
                if (!q) { err = "out of GPU memory for the DFlash 2 weights"; return false; }
                HDF2_CK(hipMemcpy(q, host.data(), (size_t)N * K * 2, hipMemcpyHostToDevice));
            } else if (kind == 2) {
                q = wal<uint8_t>((size_t)N * K / 2);
                sc = wal<half_t>((size_t)N * (K / 32));
                if (!q || !sc) { err = "out of GPU memory for the DFlash 2 weights"; return false; }
                hdf2::k_hdf2_bf16_to_q4<<<N, 256, 0, h.stream>>>(tmp, (uint8_t*)q, (half_t*)sc, N, K);
            } else {
                q = wal<int8_t>((size_t)N * K);
                sc = wal<float>((size_t)N);
                if (!q || !sc) { err = "out of GPU memory for the DFlash 2 weights"; return false; }
                hdf2::k_hdf2_bf16_to_q8row<<<N, 256, 0, h.stream>>>(tmp, (float*)sc, (int8_t*)q, N, K);
            }
            HDF2_CK(hipStreamSynchronize(h.stream));
            return true;
        };
        bool ok = up_bf16("fc.weight", (size_t)kHidden * kTargets * kHidden, fc) && up_f32("hidden_norm.weight", kHidden, hidden_norm) &&
                  up_f32("norm.weight", kHidden, norm) && up_bf16("candidate_selector.hidden_projection.weight", (size_t)kRank * kHidden, sel_h);
        for (int l = 0; ok && l < kLayers; ++l) {
            Layer& y = L[l];
            const std::string p = "layers." + std::to_string(l) + ".";
            y.kq = y.ko = var.attn;
            y.kg = y.ku = y.kd = var.ffn;
            ok = up_q(p + "self_attn.q_proj.weight", kQ, kHidden, y.kq, y.wq, y.sq, y.oq) && up_q(p + "self_attn.o_proj.weight", kHidden, kQ, y.ko, y.wo, y.so, y.oo) &&
                 up_q(p + "mlp.gate_proj.weight", kFF, kHidden, y.kg, y.wg, y.sg, y.og) && up_q(p + "mlp.up_proj.weight", kFF, kHidden, y.ku, y.wu, y.su, y.ou) &&
                 up_q(p + "mlp.down_proj.weight", kHidden, kFF, y.kd, y.wd, y.sd, y.od) &&
                 up_bf16(p + "self_attn.k_proj.weight", (size_t)kKV * kHidden, y.wkv, 0, true, (size_t)2 * kKV * kHidden) &&
                 up_bf16(p + "self_attn.v_proj.weight", (size_t)kKV * kHidden, y.wkv, (size_t)kKV * kHidden, false) &&
                 up_bf16(p + "attention_conv.kernel_projection.weight", (size_t)kDyn * kHidden, y.conv_a) &&
                 up_bf16(p + "mlp_conv.kernel_projection.weight", (size_t)kDyn * kHidden, y.conv_f) &&
                 up_f32(p + "attention_conv.base_kernel", (size_t)4 * kHidden, y.base_a) && up_f32(p + "mlp_conv.base_kernel", (size_t)4 * kHidden, y.base_f) &&
                 up_f32(p + "input_layernorm.weight", kHidden, y.attn_norm) && up_f32(p + "post_attention_layernorm.weight", kHidden, y.post_norm) &&
                 up_f32(p + "self_attn.q_norm.weight", kHeadDim, y.q_norm) && up_f32(p + "self_attn.k_norm.weight", kHeadDim, y.k_norm);
        }
        (void)hipFree(tmp);
        if (!ok) return false;
        // selector codebooks: bf16 [201024][256]; the first V rows are used. Mapped host memory on cards below 24 GB (7 steps x 17 rows of 512 B per draft).
        host_codebooks = h.small_vram;
        if (const char* e = std::getenv("STRATA_INK_DFLASH2_CODEBOOK")) host_codebooks = (std::string(e) == "host");
        for (int which = 0; which < 2; ++which) {
            const std::string name = which == 0 ? "candidate_selector.predecessor_codebook" : "candidate_selector.successor_codebook";
            const size_t n = (size_t)V * kRank;
            if (!read_t(name, n * 2, "BF16", true)) return false;
            uint16_t*& dev = which == 0 ? pred : succ;
            void*& raw = which == 0 ? pred_raw : succ_raw;
            if (host_codebooks) {
                void* dp = nullptr;
                if (hipHostMalloc(&raw, n * 2, hipHostMallocMapped) != hipSuccess || hipHostGetDevicePointer(&dp, raw, 0) != hipSuccess) {
                    (void)hipGetLastError();
                    err = "pinned host memory for the DFlash 2 codebooks";
                    return false;
                }
                std::memcpy(raw, host.data(), n * 2);
                dev = (uint16_t*)dp;
            } else {
                dev = wal<uint16_t>(n);
                if (!dev) { err = "out of GPU memory for the DFlash 2 codebooks"; return false; }
                HDF2_CK(hipMemcpy(dev, host.data(), n * 2, hipMemcpyHostToDevice));
            }
        }
        mask_row = wal<float>(kHidden);
        d_mask_tok = sal<int32_t>(1);
        if (!mask_row || !d_mask_tok) { err = "out of GPU memory for the DFlash 2 weights"; return false; }
        HDF2_CK(hipMemcpy(d_mask_tok, &mask_id, 4, hipMemcpyHostToDevice));
        h.embed(d_mask_tok, mask_row, 1);
        HDF2_CK(hipStreamSynchronize(h.stream));
        return alloc_scratch(err);
    }

    bool alloc_scratch(std::string& err) {
        using namespace hdf2;
        const int R = kBlock, R7 = kDraft;
        d_meta = sal<int32_t>(16);
        d_pos = sal<int32_t>(R);
        d_srow = sal<int32_t>(R);
        d_tok = sal<int32_t>(1);
        d_anchor_emb = sal<float>(kHidden);
        d_x = sal<float>((size_t)R * kHidden);
        d_xn = sal<float>((size_t)R * kHidden);
        d_c = sal<float>((size_t)R * kHidden);
        d_dyn = sal<float>((size_t)R * kDyn);
        d_q = sal<float>((size_t)R * kQ);
        d_kb = sal<float>((size_t)R * kKV);
        d_vb = sal<float>((size_t)R * kKV);
        d_attn = sal<float>((size_t)R * kQ);
        d_z = sal<float>((size_t)R * kHidden);
        d_g = sal<float>((size_t)R * kFF);
        d_u = sal<float>((size_t)R * kFF);
        d_act = sal<float>((size_t)R * kFF);
        d_aq = sal<int8_t>((size_t)R * kFF);
        d_as = sal<float>((size_t)R * (kFF / 32));
        p_m = sal<float>((size_t)8 * kMaxSplit * 32 + 64);
        p_l = sal<float>((size_t)8 * kMaxSplit * 32 + 64);
        p_o = sal<float>((size_t)8 * kMaxSplit * 32 * kHeadDim + 64);
        d_hd = sal<float>((size_t)R * kHidden);
        d_gsel = sal<float>((size_t)R * kRank);
        d_logits = sal<float>((size_t)R7 * V);
        d_part = sal<float2>((size_t)R7 * 64 * kTopK);
        d_part_lse = sal<float2>((size_t)R7 * 64);
        d_cand = sal<int32_t>((size_t)R7 * 2 * kTopK);
        d_tail = sal<float>((size_t)R7 * 2);
        d_ids = sal<int32_t>(kDraft);
        d_conf = sal<float>(kDraft);
        d_ix = sal<float>((size_t)kCh * kTargets * kHidden);
        d_ic = sal<float>((size_t)kCh * kHidden);
        d_icn = sal<float>((size_t)kCh * kHidden);
        d_ikc = sal<float>((size_t)kCh * kKV);
        d_ivc = sal<float>((size_t)kCh * kKV);
        if (alloc_fail) { err = "out of GPU memory for the DFlash 2 scratch"; return false; }
        if (hipHostMalloc((void**)&h_meta, 64 * sizeof(int32_t), hipHostMallocDefault) != hipSuccess ||
            hipHostMalloc((void**)&h_ids, kDraft * sizeof(int32_t), hipHostMallocDefault) != hipSuccess ||
            hipHostMalloc((void**)&h_conf, kDraft * sizeof(float), hipHostMallocDefault) != hipSuccess) {
            err = "pinned host memory for the DFlash 2 drafter";
            return false;
        }
        ring_k_h.assign(max_slots, nullptr);
        ring_v_h.assign(max_slots, nullptr);
        slot_dead.assign(max_slots, 0);
        lo.assign(max_slots, 0);
        hi.assign(max_slots, 0);
        {
            half_t** t = nullptr;
            if (hipMalloc((void**)&t, (size_t)2 * max_slots * sizeof(half_t*)) != hipSuccess) { err = "DFlash 2 ring tables"; return false; }
            (void)hipMemset(t, 0, (size_t)2 * max_slots * sizeof(half_t*));
            d_rk_tab = t;
            d_rv_tab = t + max_slots;
        }
        for (int s = 0; s < std::min(max_slots, 1); ++s)
            if (!ensure_slot(s)) { err = "not enough GPU memory for the DFlash 2 context rings"; return false; }
        ready = true;
        return true;
    }

    // what the drafter holds on the card: weights, scratch and the rings taken so far
    size_t footprint_bytes() const {
        size_t n = bytes_weights + bytes_scratch;
        for (auto* p : ring_k_h) if (p) n += ring_bytes_per_slot();
        return n;
    }

    size_t ring_bytes_per_slot() const { return (size_t)2 * hdf2::kLayers * hdf2::kRingCap * hdf2::kKV * sizeof(half_t); }

    // the ring of a slot, taken at its first use (host side: call before a graph capture)
    bool ensure_slot(int s) {
        if (s < 0 || s >= max_slots || slot_dead[s]) return false;
        if (ring_k_h[s]) return true;
        const size_t one = (size_t)hdf2::kLayers * hdf2::kRingCap * hdf2::kKV * sizeof(half_t);
        half_t *k = nullptr, *v = nullptr;
        bool room = true;
        if (s >= 1) {
            size_t vf = 0, vt = 0;
            room = hipMemGetInfo(&vf, &vt) == hipSuccess && vf > 2 * one + (1024ull << 20);
        }
        if (!room || hipMalloc((void**)&k, one) != hipSuccess || hipMalloc((void**)&v, one) != hipSuccess) {
            (void)hipGetLastError();
            if (k) (void)hipFree(k);
            slot_dead[s] = 1;
            std::fprintf(stderr, "[InkHipDFlash2] no GPU memory for slot %d's context ring (%.0f MiB); that slot does not draft\n", s, (double)(2 * one) / 1048576.0);
            return false;
        }
        (void)hipMemset(k, 0, one);
        (void)hipMemset(v, 0, one);
        ring_k_h[s] = k;
        ring_v_h[s] = v;
        (void)hipMemcpy(d_rk_tab + s, &k, sizeof(k), hipMemcpyHostToDevice);
        (void)hipMemcpy(d_rv_tab + s, &v, sizeof(v), hipMemcpyHostToDevice);
        return true;
    }

    // ---- GEMM helper ----------------------------------------------------------------------------------------------------------
    // Y[M][N] fp32 = X[M][K] fp32 * W[N][K]^T; kind 0 bf16 (the backend's GEMM), 1 Q8 row-scaled, 2 4-bit groups (quant = false reuses the last quantized activation)
    void mv(const void* W, const void* wsc, int kind, float outer, const float* X, float* Y, int M, int N, int K, bool quant = true) {
        using namespace hdf2;
        if (kind == 0) {
            h.gemm((const uint16_t*)W, X, Y, M, N, K);
            return;
        }
        if (quant) k_hdf2_quant_act<<<dim3((K / 32 + 7) / 8, M), 256, 0, h.stream>>>(X, d_aq, d_as, M, K);
        const int nch = (M + kGemvMR - 1) / kGemvMR;
        const int nrb = (N + 7) / 8;
        const size_t sm = (size_t)9 * K + 16;
        if (kind == 3) k_hdf2_gemv_sf4<<<nrb * nch, 256, sm, h.stream>>>((const uint8_t*)W, (const uint8_t*)wsc, outer, d_aq, d_as, Y, M, N, K, nch);
        else if (kind == 2) k_hdf2_gemv_q4<<<nrb * nch, 256, sm, h.stream>>>((const uint8_t*)W, (const half_t*)wsc, d_aq, d_as, Y, M, N, K, nch);
        else k_hdf2_gemv_q8<<<nrb * nch, 256, sm, h.stream>>>((const int8_t*)W, (const float*)wsc, d_aq, d_as, Y, M, N, K, nch);
    }

    // ---- context ingest -------------------------------------------------------------------------------------------------------
    // Rows 0 .. B-1 of the four fp32 tap planes (device [B][2304]): the target's features at position d_pos[i] of slot d_slots[i] (device int32). Pure device work (graph-safe);
    // the slot rings must exist (ensure_slot) and note_ingest() must be called by the host for the same rows. Rows go in chunks of kCh, in order.
    void ingest_dev(const float* t0, const float* t1, const float* t2, const float* t3, const int32_t* d_slots, const int32_t* d_posv, int B) {
        using namespace hdf2;
        if (!ready || B <= 0) return;
        const bool ip = prof_stages && B >= 64;
        hipEvent_t ie[4] = {};
        if (ip) { for (auto& e : ie) (void)hipEventCreate(&e); }
        for (int c0 = 0; c0 < B; c0 += kCh) {
            const int cn = std::min(kCh, B - c0);
            if (ip) (void)hipEventRecord(ie[0], h.stream);
            k_hdf2_gather_taps<<<cn, 256, 0, h.stream>>>(t0, t1, t2, t3, c0, cn, d_ix);
            if (ip) (void)hipEventRecord(ie[1], h.stream);
            h.gemm(fc, d_ix, d_ic, cn, kHidden, kTargets * kHidden);
            if (ip) (void)hipEventRecord(ie[2], h.stream);
            k_hdf2_rms_norm<<<cn, 256, 0, h.stream>>>(d_ic, hidden_norm, d_icn, cn);
            for (int l = 0; l < kLayers; ++l) {
                h.gemm(L[l].wkv, d_icn, d_ikc, cn, kKV, kHidden);
                h.gemm(L[l].wkv + (size_t)kKV * kHidden, d_icn, d_ivc, cn, kKV, kHidden);
                k_hdf2_rmsnorm_rope<<<dim3(kKvHeads, cn), 128, 0, h.stream>>>(d_ikc, L[l].k_norm, d_posv + c0, theta, cn, kKvHeads);
                k_hdf2_ring_write<<<cn, 256, 0, h.stream>>>(d_ikc, d_ivc, d_slots + c0, d_posv + c0, l, d_rk_tab, d_rv_tab, cn);
            }
            if (ip) {
                (void)hipEventRecord(ie[3], h.stream);
                (void)hipEventSynchronize(ie[3]);
                float a = 0, b = 0, c = 0;
                (void)hipEventElapsedTime(&a, ie[0], ie[1]);
                (void)hipEventElapsedTime(&b, ie[1], ie[2]);
                (void)hipEventElapsedTime(&c, ie[2], ie[3]);
                ing_ms[0] += a; ing_ms[1] += b; ing_ms[2] += c; ing_rows += cn;
                if (++ing_calls % 8 == 0)
                    std::fprintf(stderr, "[InkHipDFlash2 ingest] %ld rows: ms per 128 rows: gather %.3f | fc gemm %.3f | norm + 5 layers of K/V %.3f\n", ing_rows,
                                 ing_ms[0] * 128.0 / ing_rows, ing_ms[1] * 128.0 / ing_rows, ing_ms[2] * 128.0 / ing_rows);
            }
        }
        if (ip) { for (auto& e : ie) (void)hipEventDestroy(e); }
    }
    double ing_ms[3] = {0, 0, 0};
    long ing_rows = 0, ing_calls = 0;

    // host bookkeeping of what was ingested (call once per forward with the batch's slots / positions, graph hit or not)
    void note_ingest(const int32_t* slots, const int32_t* pos, int n) {
        std::vector<int64_t> mx(max_slots, -1), mn(max_slots, INT64_MAX);
        for (int i = 0; i < n; ++i) {
            const int s = slots[i];
            if (s < 0 || s >= max_slots) continue;
            mx[s] = std::max<int64_t>(mx[s], pos[i]);
            mn[s] = std::min<int64_t>(mn[s], pos[i]);
        }
        constexpr int kKeep = hdf2::kRingCap - 16;
        for (int s = 0; s < max_slots; ++s) {
            if (mx[s] < 0) continue;
            // rows below mn that the ring already holds stay valid when the batch continues or re-starts inside them (a cached prompt prefix reused without a rollback call);
            // a gap above what was ingested, or a start below the oldest valid row, means nothing before mn is known
            if (mn[s] > hi[s] || mn[s] < lo[s]) lo[s] = mn[s];
            hi[s] = mx[s] + 1;
            lo[s] = std::max<int64_t>(lo[s], hi[s] - kKeep);
        }
        n_ingest_rows += n;
    }

    // The end of a request's prompt: its rows more than kKeep positions before it are never attended (the window is 2048), so a prefill batch skips them.
    std::vector<int64_t> skip_below;
    void hint_prompt(int slot, int prompt_len) {
        if (slot < 0 || slot >= max_slots) return;
        if (skip_below.size() < (size_t)max_slots) skip_below.assign(max_slots, 0);
        constexpr int kKeep = hdf2::kRingCap - 16;
        skip_below[slot] = prompt_len > kKeep ? (int64_t)prompt_len - kKeep : 0;
    }
    // the number of leading rows of a prefill batch that no draft can ever read (they stay skipped even when the batch is a prefix of several slots' rows: the scan stops at the first row that matters)
    int skip_prefix(const int32_t* slots, const int32_t* pos, int n) const {
        if (skip_below.empty()) return 0;
        int i = 0;
        while (i < n) {
            const int s = slots[i];
            if (s < 0 || s >= max_slots || skip_below[s] <= 0 || pos[i] >= skip_below[s]) break;
            ++i;
        }
        return i;
    }

    // rows at positions >= keep_len were ingested with a verify batch and then rejected: only [0, keep_len) is context
    void rollback(int slot, int64_t keep_len) {
        if (slot < 0 || slot >= max_slots) return;
        if (hi[slot] > keep_len) hi[slot] = keep_len;
        if (lo[slot] > hi[slot]) lo[slot] = hi[slot];
    }

    // the context of a slot is gone (the slot was reused or a conversation was restored): drafts see no context until the next ingest
    void reset_slot(int slot, int64_t pos) {
        if (slot < 0 || slot >= max_slots) return;
        lo[slot] = hi[slot] = pos;
        if (!skip_below.empty()) skip_below[slot] = 0;
    }

    // ---- the draft ------------------------------------------------------------------------------------------------------------
    // One slot: the anchor token at position pos0; out_ids / out_conf [7]. ctx_hi < 0: the context ends at pos0. Returns false when nothing was drafted.
    bool draft(int slot, int32_t anchor, int32_t pos0, int32_t* out_ids, float* out_conf, std::string& err) {
        using namespace hdf2;
        if (!ready || slot < 0 || slot >= max_slots || !ensure_slot(slot)) return false;
        if (prof) t_draft0 = std::chrono::steady_clock::now();
        const int64_t lo_eff0 = std::max<int64_t>(lo[slot], (int64_t)pos0 - (kWindow - 1));
        int64_t hc = std::min<int64_t>((int64_t)pos0, hi[slot]);
        int64_t lo_eff = lo_eff0 > pos0 ? pos0 : lo_eff0;
        h_meta[0] = slot;
        h_meta[1] = pos0;
        h_meta[2] = (int32_t)lo_eff;
        h_meta[3] = anchor;        // meta[b * 4 + 3]: the anchor token (the selector's first prev)
        h_meta[4] = anchor;        // meta[B * 4 + b]: the anchor tokens (B = 1): the embedding kernel reads them
        h_meta[5] = (int32_t)hc;   // meta[B * 5 + b]: where the context ends
        hipStream_t st = h.stream;
        if (hipMemcpyAsync(d_meta, h_meta, 6 * 4, hipMemcpyHostToDevice, st) != hipSuccess) { err = "dflash2 meta upload"; return false; }
        // The kernels read everything that changes per draft (slot, position, window, anchor) from d_meta, so one captured graph serves every draft. The first draft runs
        // directly (library first-use work must not happen inside a capture); a failed capture falls back to direct launches for good.
        if (graph_enabled && !prof_stages && n_drafts_run >= 1 && !graph_failed) {
            if (!g_draft) {
                hipGraph_t g = nullptr;
                bool ok = hipStreamBeginCapture(st, hipStreamCaptureModeThreadLocal) == hipSuccess;
                if (ok) {
                    emit_draft(st);
                    ok = hipStreamEndCapture(st, &g) == hipSuccess && g != nullptr;
                }
                if (ok) ok = hipGraphInstantiate(&g_draft, g, nullptr, nullptr, 0) == hipSuccess;
                if (g) (void)hipGraphDestroy(g);
                if (!ok) {
                    (void)hipGetLastError();
                    g_draft = nullptr;
                    graph_failed = true;
                    std::fprintf(stderr, "[InkHipDFlash2] the draft graph could not be captured: direct launches\n");
                    return draft_launch_direct(out_ids, out_conf, err);
                }
            }
            if (hipGraphLaunch(g_draft, st) != hipSuccess) { err = "dflash2 graph launch"; return false; }
        } else {
            emit_draft(st);
        }
        return draft_finish(out_ids, out_conf, err);
    }

    bool draft_launch_direct(int32_t* out_ids, float* out_conf, std::string& err) {
        emit_draft(h.stream);
        return draft_finish(out_ids, out_conf, err);
    }

    bool draft_finish(int32_t* out_ids, float* out_conf, std::string& err) {
        hipStream_t st = h.stream;
        (void)hipMemcpyAsync(h_ids, d_ids, hdf2::kDraft * 4, hipMemcpyDeviceToHost, st);
        (void)hipMemcpyAsync(h_conf, d_conf, hdf2::kDraft * 4, hipMemcpyDeviceToHost, st);
        const hipError_t se = hipStreamSynchronize(st);
        if (se != hipSuccess) { err = std::string("dflash2 draft: ") + hipGetErrorString(se); return false; }
        std::memcpy(out_ids, h_ids, hdf2::kDraft * 4);
        std::memcpy(out_conf, h_conf, hdf2::kDraft * 4);
        ++n_drafts_run;
        if (prof) {
            prof_draft_ms += std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t_draft0).count();
            stages_collect();
            if (n_drafts_run % 50 == 0) {
                std::fprintf(stderr, "[InkHipDFlash2 prof] %ld drafts, %.2f ms per draft (wall, incl. launch and sync), %ld ingest rows\n", n_drafts_run,
                             prof_draft_ms / (double)n_drafts_run, n_ingest_rows);
                if (prof_stages)
                    std::fprintf(stderr, "[InkHipDFlash2 stages] ms per draft: embed+attn L0 %.3f | ffn L0 %.3f | layers 1-4 %.3f | head logits %.3f | top16 %.3f | select %.3f\n",
                                 pst[1] / (double)n_drafts_run, pst[2] / (double)n_drafts_run, pst[3] / (double)n_drafts_run, pst[4] / (double)n_drafts_run,
                                 pst[5] / (double)n_drafts_run, pst[6] / (double)n_drafts_run);
            }
        }
        return true;
    }
    std::chrono::steady_clock::time_point t_draft0;

    void drop_graph() {
        if (g_draft) { (void)hipGraphExecDestroy(g_draft); g_draft = nullptr; }
    }

    hipGraphExec_t g_draft = nullptr;
    bool graph_enabled = !(std::getenv("STRATA_INK_DFLASH2_GRAPH") && std::getenv("STRATA_INK_DFLASH2_GRAPH")[0] == '0');
    bool graph_failed = false;

    // the whole block forward + head + selector on stream st (everything it needs per draft is in d_meta)
    void emit_draft(hipStream_t st) {
        using namespace hdf2;
        const int R = kBlock, R7 = kDraft;
        const int32_t* d_anchor = d_meta + 4;
        mark(0, st);
        h.embed(d_anchor, d_anchor_emb, 1);
        k_hdf2_build_x<<<R, 256, 0, st>>>(d_anchor_emb, mask_row, d_x, 1);
        k_hdf2_block_pos<<<1, 64, 0, st>>>(d_meta, 1, d_pos, d_srow);
        const int split = kMaxSplit;
        for (int l = 0; l < kLayers; ++l) {
            const Layer& y = L[l];
            // ---- attention sublayer
            k_hdf2_rms_norm<<<R, 256, 0, st>>>(d_x, y.attn_norm, d_xn, R);
            h.gemm(y.conv_a, d_xn, d_dyn, R, kDyn, kHidden);
            k_hdf2_conv_pre<<<R, 256, 0, st>>>(d_xn, d_dyn, y.base_a, d_c, R);
            mv(y.wq, y.sq, y.kq, y.oq, d_c, d_q, R, kQ, kHidden);
            h.gemm(y.wkv, d_c, d_kb, R, kKV, kHidden);
            h.gemm(y.wkv + (size_t)kKV * kHidden, d_c, d_vb, R, kKV, kHidden);
            k_hdf2_rmsnorm_rope<<<dim3(kHeads, R), 128, 0, st>>>(d_q, y.q_norm, d_pos, theta, R, kHeads);
            k_hdf2_rmsnorm_rope<<<dim3(kKvHeads, R), 128, 0, st>>>(d_kb, y.k_norm, d_pos, theta, R, kKvHeads);
            k_hdf2_ring_write<<<R, 256, 0, st>>>(d_kb, d_vb, d_srow, d_pos, l, d_rk_tab, d_rv_tab, R);
            k_hdf2_attn_part<<<dim3(kKvHeads, 1, split), 256, 0, st>>>(d_q, d_rk_tab, d_rv_tab, d_meta, d_meta + 5, l, split, p_m, p_l, p_o);
            k_hdf2_attn_combine<<<dim3(kKvHeads, 1, 32), 128, 0, st>>>(p_m, p_l, p_o, split, d_attn);
            mv(y.wo, y.so, y.ko, y.oo, d_attn, d_z, R, kHidden, kQ);
            k_hdf2_conv_post_add<<<R, 256, 0, st>>>(d_z, d_dyn, y.base_a, d_x, R);
            if (l == 0) mark(1, st);
            // ---- feed-forward sublayer
            k_hdf2_rms_norm<<<R, 256, 0, st>>>(d_x, y.post_norm, d_xn, R);
            h.gemm(y.conv_f, d_xn, d_dyn, R, kDyn, kHidden);
            k_hdf2_conv_pre<<<R, 256, 0, st>>>(d_xn, d_dyn, y.base_f, d_c, R);
            mv(y.wg, y.sg, y.kg, y.og, d_c, d_g, R, kFF, kHidden);
            mv(y.wu, y.su, y.ku, y.ou, d_c, d_u, R, kFF, kHidden, false);
            k_hdf2_swiglu<<<(unsigned)(((size_t)R * kFF + 255) / 256), 256, 0, st>>>(d_g, d_u, d_act, (size_t)R * kFF);
            mv(y.wd, y.sd, y.kd, y.od, d_act, d_z, R, kHidden, kFF);
            k_hdf2_conv_post_add<<<R, 256, 0, st>>>(d_z, d_dyn, y.base_f, d_x, R);
            if (l == 0) mark(2, st);
        }
        mark(3, st);
        // ---- head
        k_hdf2_rms_norm<<<R, 256, 0, st>>>(d_x, norm, d_hd, R);
        h.gemm(sel_h, d_hd, d_gsel, R, kRank, kHidden);
        const int head_blocks = head_blocks_n;
        k_hdf2_head_logits<<<head_blocks, 256, 0, st>>>(h.unemb, h.unemb_fmt, h.unemb_rows, d_hd + kHidden, d_logits, R7, kHidden, V);
        mark(4, st);
        const int nchunk = (V + kTopChunk - 1) / kTopChunk;
        k_hdf2_tile_part<<<dim3(nchunk, R7), 256, 0, st>>>(d_logits, V, cap, d_part, d_part_lse);
        k_hdf2_top16_merge<<<R7, 512, 0, st>>>(d_part, d_part_lse, nchunk, V, cap, d_cand, d_tail);
        mark(5, st);
        k_hdf2_select<<<1, 512, 0, st>>>(d_meta, d_cand, d_tail, d_gsel, pred, succ, kDraft, V, cap, d_ids, d_conf);
        mark(6, st);
    }

    // STRATA_INK_DFLASH2_PROF=2: GPU time of the stages of a draft (direct launches, no graph); printed with the draft timing
    bool prof_stages = [] { const char* e = std::getenv("STRATA_INK_DFLASH2_PROF"); return e && e[0] == '2'; }();
    hipEvent_t pev[8] = {};
    double pst[8] = {};
    int head_blocks_n = [] { const char* e = std::getenv("STRATA_INK_DFLASH2_HEAD_BLOCKS"); return e && std::atoi(e) > 0 ? std::atoi(e) : 768; }();
    void mark(int i, hipStream_t st) {
        if (!prof_stages) return;
        if (!pev[i]) (void)hipEventCreate(&pev[i]);
        (void)hipEventRecord(pev[i], st);
    }
    void stages_collect() {
        if (!prof_stages || !pev[0] || !pev[6]) return;
        for (int i = 1; i <= 6; ++i) {
            float ms = 0;
            if (hipEventElapsedTime(&ms, pev[i - 1], pev[i]) == hipSuccess) pst[i] += ms;
        }
    }

    void free_all() {
        drop_graph();
        for (auto& p : ring_k_h) if (p) (void)hipFree(p);
        for (auto& p : ring_v_h) if (p) (void)hipFree(p);
        ring_k_h.clear();
        ring_v_h.clear();
        if (d_rk_tab) (void)hipFree(d_rk_tab);
        d_rk_tab = d_rv_tab = nullptr;
        for (void* p : owned) (void)hipFree(p);
        owned.clear();
        if (pred_raw) { (void)hipHostFree(pred_raw); pred_raw = nullptr; }
        if (succ_raw) { (void)hipHostFree(succ_raw); succ_raw = nullptr; }
        if (h_meta) { (void)hipHostFree(h_meta); h_meta = nullptr; }
        if (h_ids) { (void)hipHostFree(h_ids); h_ids = nullptr; }
        if (h_conf) { (void)hipHostFree(h_conf); h_conf = nullptr; }
        ready = false;
    }
#undef HDF2_CK
};

}  // namespace strata::ink
