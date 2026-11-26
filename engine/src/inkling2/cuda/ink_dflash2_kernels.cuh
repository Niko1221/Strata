// engine/src/inkling2/cuda/ink_dflash2_kernels.cuh - kernels of the opt-in DFlash 2 block-diffusion drafter for Inkling 2 (see ink_dflash2.cuh)
//
// MIT License
// Copyright (c) 2026 Niko1221 and the Strata contributors
//
// The drafter's design (non-causal sliding-window block layers wrapped in 2-tap group-16 dynamic causal convolutions, a rank-256 bilinear
// candidate selector over the exact top-16 of the target's head, and the full-vocabulary tail used for the confidence) follows the checkpoint
// incoai/Inkling-2-Mini-DFlash2 (Apache-2.0) and tjm8874's implementation of it (MIT). The two kernels k_df2_tile_part and k_df2_select are adapted
// from that implementation, so its MIT notice is kept in notices/DFLASH2-TJM8874-MIT.txt.
//
// The kernels here are written for Strata's runner (fp32 activations, Q8 row-scaled weights, cuBLAS for the BF16 matrices). Nothing here changes what the target model computes: a draft only
// decides which tokens are verified, and the verification is the target's own greedy argmax.
#pragma once

#include "ink_runner.cuh"

namespace strata::ink {
namespace df2 {

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

__device__ __forceinline__ float df2_bf16f(uint16_t u) { return __uint_as_float((uint32_t)u << 16); }

__device__ __forceinline__ float df2_softcap(float v, float cap) {
    return cap > 0.0f ? cap * tanhf(v / cap) : v;
}

// ---- gathers & quantization ----------------------------------------------------------------------------------------------

// [M][4 * 2304] bf16 from four fp32 tap planes; row i reads plane row src_rows[i] (or i when src_rows is null).
__global__ void __launch_bounds__(256) k_df2_gather_taps(const float* __restrict__ t0, const float* __restrict__ t1,
                                                         const float* __restrict__ t2, const float* __restrict__ t3,
                                                         const int32_t* __restrict__ src_rows, int row0, int M,
                                                         __nv_bfloat16* __restrict__ out) {
    const int i = blockIdx.x;
    if (i >= M) return;
    const int src = src_rows ? src_rows[row0 + i] : (row0 + i);
    const float* tp[4] = {t0, t1, t2, t3};
    __nv_bfloat16* dst = out + (size_t)i * kTargets * kHidden;
#pragma unroll
    for (int t = 0; t < kTargets; ++t) {
        const float4* s4 = reinterpret_cast<const float4*>(tp[t] + (size_t)src * kHidden);
        for (int d = threadIdx.x; d < kHidden / 4; d += 256) {
            const float4 v = s4[d];
            __nv_bfloat162* d2 = reinterpret_cast<__nv_bfloat162*>(dst + t * kHidden + d * 4);
            d2[0] = __floats2bfloat162_rn(v.x, v.y);
            d2[1] = __floats2bfloat162_rn(v.z, v.w);
        }
    }
}

// fp32 [rows][K] -> int8 codes + one fp32 scale per 32 values. One warp per 32-block, 8 blocks per CTA. row_mode 1: row i is block row
// (i / 7) * 8 + 1 + i % 7 of the 8-row block buffer (the 7 draft rows of every slot).
__global__ void __launch_bounds__(256) k_df2_quant_act(const float* __restrict__ x, int8_t* __restrict__ xq, float* __restrict__ xs,
                                                       int rows, int K, int row_mode) {
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const int nb = K >> 5;
    const int b = blockIdx.x * 8 + warp;
    const int i = blockIdx.y;
    if (i >= rows || b >= nb) return;
    const int src_row = row_mode == 1 ? (i / kDraft) * kBlock + 1 + (i % kDraft) : i;
    const float v = x[(size_t)src_row * K + (size_t)b * 32 + lane];
    const float am = warp_max(fabsf(v));
    const float sc = am > 0.0f ? am * (1.0f / 127.0f) : 1.0f;
    const float inv = am > 0.0f ? 127.0f / am : 0.0f;
    xq[(size_t)i * K + (size_t)b * 32 + lane] = (int8_t)max(-127, min(127, __float2int_rn(v * inv)));
    if (lane == 0) xs[(size_t)i * nb + b] = sc;
}

// ---- Q8 row-scaled skinny GEMV: Y[m][n] = wsc[n] * sum_k W[n][k] * x[m][k], x as per-32-block int8 codes ---------------------
// One warp per output row, the 8 activation rows of a chunk in shared memory, dp4a. Block 256 = 8 output rows.
// Grid 1D: chunk (m / 8) varies fastest, so the CTAs that read the same weight rows run together (one DRAM read, the rest hit L2).
constexpr int kGemvMR = 8;
template <int MR>
__global__ void __launch_bounds__(256) k_df2_gemv_q8(const int8_t* __restrict__ W, const float* __restrict__ wsc,
                                                     const int8_t* __restrict__ xq, const float* __restrict__ xs,
                                                     float* __restrict__ Y, int M, int N, int K, int n_chunks) {
    extern __shared__ __align__(16) uint8_t g_smem[];
    const int nb = K >> 5;
    int8_t* sq = reinterpret_cast<int8_t*>(g_smem);                                   // [8][K]
    float* ss = reinterpret_cast<float*>(g_smem + (size_t)MR * K);               // [8][K/32]
    const int mc = blockIdx.x % n_chunks;
    const int rb = blockIdx.x / n_chunks;
    const int m0 = mc * MR;
    const int mrows = min(MR, M - m0);
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
    float acc[MR];
#pragma unroll
    for (int m = 0; m < MR; ++m) acc[m] = 0.0f;
    const int nc = K >> 4;
    for (int c = lane; c < nc; c += 32) {
        const uint4 w = w16[c];
#pragma unroll
        for (int m = 0; m < MR; ++m) {
            if (m < mrows) {
                const uint4 a = *reinterpret_cast<const uint4*>(sq + (size_t)m * K + c * 16);
                int p = dp4a_i8(w.x, a.x, 0);
                p = dp4a_i8(w.y, a.y, p);
                p = dp4a_i8(w.z, a.z, p);
                p = dp4a_i8(w.w, a.w, p);
                acc[m] += (float)p * ss[m * nb + (c >> 1)];
            }
        }
    }
    const float sc = wsc[n];
#pragma unroll
    for (int m = 0; m < MR; ++m) {
        const float s = warp_sum(acc[m]);
        if (lane == 0 && m < mrows) Y[(size_t)(m0 + m) * N + n] = s * sc;
    }
}


// ---- 4-bit weights (tight-VRAM cards): per-32-value groups, fp16 scale, byte j of a group = element j (low nibble) and j + 16 (high nibble) ------------
// W4 [N][K/2] bytes, wsc [N][K/32] half. The nibbles are used as the high half of an int8 (value * 16, the 1/16 is folded into the scale).
__global__ void __launch_bounds__(256) k_df2_bf16_to_q4(const uint16_t* __restrict__ src, uint8_t* __restrict__ dst, __half* __restrict__ sc, int N, int K) {
    const int n = blockIdx.x;
    if (n >= N) return;
    const int nb = K >> 5;
    for (int b = threadIdx.x; b < nb; b += 256) {
        const uint16_t* w = src + (size_t)n * K + (size_t)b * 32;
        float v[32];
        float am = 0.0f;
#pragma unroll
        for (int j = 0; j < 32; ++j) { v[j] = df2_bf16f(w[j]); am = fmaxf(am, fabsf(v[j])); }
        const float s = am > 0.0f ? am / 7.0f : 1.0f;
        const float inv = 1.0f / s;
        sc[(size_t)n * nb + b] = __float2half_rn(s);
        uint8_t* d = dst + (size_t)n * (K / 2) + (size_t)b * 16;
#pragma unroll
        for (int j = 0; j < 16; ++j) {
            const int lo = max(-8, min(7, __float2int_rn(v[j] * inv)));
            const int hi = max(-8, min(7, __float2int_rn(v[j + 16] * inv)));
            d[j] = (uint8_t)((lo & 0xF) | ((hi & 0xF) << 4));
        }
    }
}

// ---- the pre-quantized formats back to bf16 (a sidecar loaded on a big card; value as the GEMV kernels read it) --------------------------------------------------------------------------
__device__ __forceinline__ uint16_t df2_f32_to_bf16(float v) { return __bfloat16_as_ushort(__float2bfloat16_rn(v)); }

__global__ void __launch_bounds__(256) k_df2_q8row_to_bf16(const int8_t* __restrict__ q, const float* __restrict__ sc, uint16_t* __restrict__ out, int N, int K) {
    const size_t total = (size_t)N * K;
    for (size_t i = (size_t)blockIdx.x * 256 + threadIdx.x; i < total; i += (size_t)gridDim.x * 256) out[i] = df2_f32_to_bf16((float)q[i] * sc[i / (size_t)K]);
}

// 4-bit groups of 32 (low nibble = element j, high nibble = element j + 16, two's complement), half scale per 32: value = nibble * scale
__global__ void __launch_bounds__(256) k_df2_q4_to_bf16(const uint8_t* __restrict__ w, const __half* __restrict__ sc, uint16_t* __restrict__ out, int N, int K) {
    const size_t total = (size_t)N * K / 32;   // groups
    for (size_t g = (size_t)blockIdx.x * 256 + threadIdx.x; g < total; g += (size_t)gridDim.x * 256) {
        const float s = __half2float(sc[g]);
        const uint8_t* d = w + g * 16;
        uint16_t* o = out + g * 32;
        for (int j = 0; j < 16; ++j) {
            const int lo = (int)(d[j] & 0xF) - ((d[j] & 0x8) ? 16 : 0);
            const int hi = (int)(d[j] >> 4) - ((d[j] & 0x80) ? 16 : 0);
            o[j] = df2_f32_to_bf16((float)lo * s);
            o[j + 16] = df2_f32_to_bf16((float)hi * s);
        }
    }
}

// SF4: E2M1 x2 per byte (low nibble first), UE4M3 scale per 16, fp32 tensor scale: value = E2M1 * e4m3(scale) * w_outer
__global__ void __launch_bounds__(256) k_df2_sf4_to_bf16(const uint8_t* __restrict__ p, const uint8_t* __restrict__ s, float w_outer, uint16_t* __restrict__ out, int N, int K) {
    const float mag[8] = {0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f};
    const size_t blocks = (size_t)N * K / 16;
    for (size_t b = (size_t)blockIdx.x * 256 + threadIdx.x; b < blocks; b += (size_t)gridDim.x * 256) {
        const float sc = __half2float(__half(__nv_cvt_fp8_to_halfraw(s[b], __NV_E4M3))) * w_outer;
        const uint8_t* d = p + b * 8;
        uint16_t* o = out + b * 16;
        for (int j = 0; j < 8; ++j) {
            const uint8_t lo = d[j] & 0xF, hi = d[j] >> 4;
            o[2 * j] = df2_f32_to_bf16((lo & 8 ? -mag[lo & 7] : mag[lo & 7]) * sc);
            o[2 * j + 1] = df2_f32_to_bf16((hi & 8 ? -mag[hi & 7] : mag[hi & 7]) * sc);
        }
    }
}

template <int MR>
__global__ void __launch_bounds__(256) k_df2_gemv_q4(const uint8_t* __restrict__ W, const __half* __restrict__ wsc, const int8_t* __restrict__ xq,
                                                     const float* __restrict__ xs, float* __restrict__ Y, int M, int N, int K, int n_chunks) {
    extern __shared__ __align__(16) uint8_t g_smem[];
    const int nb = K >> 5;
    int8_t* sq = reinterpret_cast<int8_t*>(g_smem);
    float* ss = reinterpret_cast<float*>(g_smem + (size_t)MR * K);
    const int mc = blockIdx.x % n_chunks;
    const int rb = blockIdx.x / n_chunks;
    const int m0 = mc * MR;
    const int mrows = min(MR, M - m0);
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
    const __half* wsr = wsc + (size_t)n * nb;
    float acc[MR];
#pragma unroll
    for (int m = 0; m < MR; ++m) acc[m] = 0.0f;
    for (int b = lane; b < nb; b += 32) {
        const uint4 w = w16[b];
        const float ws = __half2float(wsr[b]) * (1.0f / 16.0f);
        const uint32_t lo0 = (w.x << 4) & 0xF0F0F0F0u, lo1 = (w.y << 4) & 0xF0F0F0F0u, lo2 = (w.z << 4) & 0xF0F0F0F0u, lo3 = (w.w << 4) & 0xF0F0F0F0u;
        const uint32_t hi0 = w.x & 0xF0F0F0F0u, hi1 = w.y & 0xF0F0F0F0u, hi2 = w.z & 0xF0F0F0F0u, hi3 = w.w & 0xF0F0F0F0u;
#pragma unroll
        for (int m = 0; m < MR; ++m) {
            if (m < mrows) {
                const uint4 a0 = *reinterpret_cast<const uint4*>(sq + (size_t)m * K + b * 32);
                const uint4 a1 = *reinterpret_cast<const uint4*>(sq + (size_t)m * K + b * 32 + 16);
                int p = dp4a_i8(lo0, a0.x, 0);
                p = dp4a_i8(lo1, a0.y, p);
                p = dp4a_i8(lo2, a0.z, p);
                p = dp4a_i8(lo3, a0.w, p);
                p = dp4a_i8(hi0, a1.x, p);
                p = dp4a_i8(hi1, a1.y, p);
                p = dp4a_i8(hi2, a1.z, p);
                p = dp4a_i8(hi3, a1.w, p);
                acc[m] += (float)p * ws * ss[m * nb + b];
            }
        }
    }
#pragma unroll
    for (int m = 0; m < MR; ++m) {
        const float s = warp_sum(acc[m]);
        if (lane == 0 && m < mrows) Y[(size_t)(m0 + m) * N + n] = s;
    }
}


// ---- SF4 (Strata's FMT_SF4 layout): [N][K/2] bytes of E2M1 x 2 (low nibble first), [N][K/16] UE4M3 block scales, one fp32 tensor scale (w_outer) ------------------------------
// value = E2M1(nibble) * e4m3(block scale) * w_outer. w_outer = amax / (6 * 448), block scale = amax_block / (6 * w_outer) rounded to E4M3.
__global__ void __launch_bounds__(256) k_df2_amax_bf16(const uint16_t* __restrict__ src, size_t n, uint32_t* __restrict__ out) {
    float am = 0.0f;
    for (size_t i = (size_t)blockIdx.x * 256 + threadIdx.x; i < n; i += (size_t)gridDim.x * 256) am = fmaxf(am, fabsf(df2_bf16f(src[i])));
    am = warp_max(am);
    __shared__ float sh[8];
    if ((threadIdx.x & 31) == 0) sh[threadIdx.x >> 5] = am;
    __syncthreads();
    if (threadIdx.x == 0) {
        float g = sh[0];
        for (int i = 1; i < 8; ++i) g = fmaxf(g, sh[i]);
        atomicMax(out, __float_as_uint(g));
    }
}

__global__ void __launch_bounds__(256) k_df2_bf16_to_sf4(const uint16_t* __restrict__ src, uint8_t* __restrict__ dst_p, uint8_t* __restrict__ dst_s,
                                                         float w_outer, int N, int K) {
    const int r = blockIdx.x;
    if (r >= N) return;
    const int nb = K >> 4;
    for (int b = threadIdx.x; b < nb; b += 256) {
        const uint16_t* w = src + (size_t)r * K + (size_t)b * 16;
        float v[16];
        float am = 0.0f;
#pragma unroll
        for (int i = 0; i < 16; ++i) { v[i] = df2_bf16f(w[i]); am = fmaxf(am, fabsf(v[i])); }
        uint8_t sc_byte = 0u;
        float inv = 0.0f;
        if (am > 0.0f) {
            const float sc_f = fminf(am / (w_outer * 6.0f), 448.0f);
            const __nv_fp8_e4m3 fp8(sc_f);
            sc_byte = fp8.__x;
            const float denom = w_outer * (float)fp8;
            inv = denom > 0.0f ? 1.0f / denom : 0.0f;
        }
        uint8_t* dp = dst_p + (size_t)r * (K >> 1) + (size_t)b * 8;
#pragma unroll
        for (int j = 0; j < 8; ++j) dp[j] = ink_pack_e2m1x2_portable(v[2 * j] * inv, v[2 * j + 1] * inv);
        dst_s[(size_t)r * nb + b] = sc_byte;
    }
}

// E2M1 values times 2 are the integers 0 1 2 3 4 6 8 12; two nibbles of a byte -> two int8 through a 256-entry table in shared memory, then dp4a with the int8 activations.
template <int MR>
__global__ void __launch_bounds__(256) k_df2_gemv_sf4(const uint8_t* __restrict__ Wp, const uint8_t* __restrict__ Ws, float w_outer, const int8_t* __restrict__ xq,
                                                      const float* __restrict__ xs, float* __restrict__ Y, int M, int N, int K, int n_chunks) {
    extern __shared__ __align__(16) uint8_t g_smem[];
    __shared__ uint16_t lut[256];
    const int nb = K >> 5;
    int8_t* sq = reinterpret_cast<int8_t*>(g_smem);
    float* ss = reinterpret_cast<float*>(g_smem + (size_t)MR * K);
    {
        const int mag[8] = {0, 1, 2, 3, 4, 6, 8, 12};
        const int lo = threadIdx.x & 15, hi = threadIdx.x >> 4;
        const int vlo = (lo & 8) ? -mag[lo & 7] : mag[lo & 7];
        const int vhi = (hi & 8) ? -mag[hi & 7] : mag[hi & 7];
        lut[threadIdx.x] = (uint16_t)(((uint32_t)(vlo & 0xFF)) | ((uint32_t)(vhi & 0xFF) << 8));
    }
    const int mc = blockIdx.x % n_chunks;
    const int rb = blockIdx.x / n_chunks;
    const int m0 = mc * MR;
    const int mrows = min(MR, M - m0);
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
    float acc[MR];
#pragma unroll
    for (int m = 0; m < MR; ++m) acc[m] = 0.0f;
    for (int b = lane; b < nb; b += 32) {
        const uint4 w = w16[b];
        __nv_fp8_e4m3 f0, f1;
        f0.__x = wsr[2 * b];
        f1.__x = wsr[2 * b + 1];
        const float s0 = (float)f0, s1 = (float)f1;
        const uint32_t wv[4] = {w.x, w.y, w.z, w.w};
        uint32_t d[8];
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            d[2 * j] = (uint32_t)lut[wv[j] & 0xFF] | ((uint32_t)lut[(wv[j] >> 8) & 0xFF] << 16);
            d[2 * j + 1] = (uint32_t)lut[(wv[j] >> 16) & 0xFF] | ((uint32_t)lut[(wv[j] >> 24) & 0xFF] << 16);
        }
#pragma unroll
        for (int m = 0; m < MR; ++m) {
            if (m < mrows) {
                const uint4 a0 = *reinterpret_cast<const uint4*>(sq + (size_t)m * K + b * 32);
                const uint4 a1 = *reinterpret_cast<const uint4*>(sq + (size_t)m * K + b * 32 + 16);
                int p0 = dp4a_i8(d[0], a0.x, 0);
                p0 = dp4a_i8(d[1], a0.y, p0);
                p0 = dp4a_i8(d[2], a0.z, p0);
                p0 = dp4a_i8(d[3], a0.w, p0);
                int p1 = dp4a_i8(d[4], a1.x, 0);
                p1 = dp4a_i8(d[5], a1.y, p1);
                p1 = dp4a_i8(d[6], a1.z, p1);
                p1 = dp4a_i8(d[7], a1.w, p1);
                acc[m] += ((float)p0 * s0 + (float)p1 * s1) * ss[m * nb + b];
            }
        }
    }
#pragma unroll
    for (int m = 0; m < MR; ++m) {
        const float s = warp_sum(acc[m]);
        if (lane == 0 && m < mrows) Y[(size_t)(m0 + m) * N + n] = s * 0.5f * w_outer;
    }
}

// ---- convolutions ------------------------------------------------------------------------------------------------------------

// out[r][c] = (b0[c] + dyn[g]) * z[r][c] + (t > 0 ? (b1[c] + dyn[144 + g]) * z[r-1][c] : 0); t = r % 8, g = c / 16. Optional bf16 copy.
__global__ void __launch_bounds__(256) k_df2_conv_pre(const float* __restrict__ z, const float* __restrict__ dyn,
                                                      const float* __restrict__ base, float* __restrict__ out_f32,
                                                      __nv_bfloat16* __restrict__ out_bf16, int rows) {
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
        if (out_bf16) out_bf16[(size_t)r * kHidden + c] = __float2bfloat16_rn(acc);
    }
}

// x[r][c] += (b2[c] + dyn[288 + g]) * z[r][c] + (t > 0 ? (b3[c] + dyn[432 + g]) * z[r-1][c] : 0)
__global__ void __launch_bounds__(256) k_df2_conv_post_add(const float* __restrict__ z, const float* __restrict__ dyn,
                                                           const float* __restrict__ base, float* __restrict__ x, int rows) {
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

// ---- block inputs --------------------------------------------------------------------------------------------------------------

// x[s * 8 + 0] = anchor_emb[s], x[s * 8 + t] = mask_row (t = 1..7)
__global__ void __launch_bounds__(256) k_df2_build_x(const float* __restrict__ anchor_emb, const float* __restrict__ mask_row,
                                                     float* __restrict__ x, int B) {
    const int r = blockIdx.x;
    if (r >= B * kBlock) return;
    const int s = r / kBlock, t = r % kBlock;
    const float* src = t == 0 ? anchor_emb + (size_t)s * kHidden : mask_row;
    for (int d = threadIdx.x; d < kHidden; d += 256) x[(size_t)r * kHidden + d] = src[d];
}

// per-row positions of the block rows: pos0[s] + t
__global__ void k_df2_block_pos(const int32_t* __restrict__ meta, int B, int32_t* __restrict__ pos, int32_t* __restrict__ srow) {
    const int r = blockIdx.x * blockDim.x + threadIdx.x;
    if (r >= B * kBlock) return;
    pos[r] = meta[(r / kBlock) * 4 + 1] + (r % kBlock);
    srow[r] = meta[(r / kBlock) * 4 + 0];
}

// silu(g) * u
__global__ void __launch_bounds__(256) k_df2_swiglu(const float* __restrict__ g, const float* __restrict__ u, float* __restrict__ out, size_t n) {
    const size_t i = (size_t)blockIdx.x * 256 + threadIdx.x;
    if (i < n) {
        const float gv = g[i];
        out[i] = (gv / (1.0f + __expf(-gv))) * u[i];
    }
}

// Per-head RMSNorm (weight [128] fp32) + NeoX half-split RoPE of x [M][H][128] in place; position of row m = pos[m].
__global__ void __launch_bounds__(128) k_df2_rmsnorm_rope(float* __restrict__ x, const float* __restrict__ nw, const int32_t* __restrict__ pos,
                                                          float theta, int M, int H) {
    const int m = blockIdx.y, h = blockIdx.x, d = threadIdx.x;
    if (m >= M || h >= H) return;
    __shared__ float s_x[kHeadDim];
    __shared__ float sh[8];
    float* xp = x + ((size_t)m * H + h) * kHeadDim;
    const float v = xp[d];
    const float ss = block_sum(v * v, sh);
    const float inv = rsqrtf(ss * (1.0f / (float)kHeadDim) + kInkRmsEps);
    s_x[d] = v * inv * nw[d];
    __syncthreads();
    if (d < kHeadDim / 2) {
        const float freq = 1.0f / powf(theta, (float)(2 * d) * (1.0f / (float)kHeadDim));
        float sn, cs;
        __sincosf((float)pos[m] * freq, &sn, &cs);
        const float x0 = s_x[d], x1 = s_x[d + kHeadDim / 2];
        xp[d] = x0 * cs - x1 * sn;
        xp[d + kHeadDim / 2] = x0 * sn + x1 * cs;
    }
}

// ---- ring writes ---------------------------------------------------------------------------------------------------------------

// rows [n][1024] fp32 K and V (K already normed and roped) -> fp16 ring cells of (slot, pos) in layer `layer`.
__global__ void __launch_bounds__(256) k_df2_ring_write(const float* __restrict__ kb, const float* __restrict__ vb,
                                                        const int32_t* __restrict__ slots, const int32_t* __restrict__ pos,
                                                        int layer, __half* const* __restrict__ rk_tab, __half* const* __restrict__ rv_tab, int n) {
    const int i = blockIdx.x;
    if (i >= n) return;
    const int s = slots[i];
    const int cell = pos[i] % kRingCap;
    __half* rk = rk_tab[s];
    __half* rv = rv_tab[s];
    const size_t base = ((size_t)layer * kRingCap + cell) * kKV;
    for (int d = threadIdx.x; d < kKV; d += 256) {
        rk[base + d] = __float2half_rn(kb[(size_t)i * kKV + d]);
        rv[base + d] = __float2half_rn(vb[(size_t)i * kKV + d]);
    }
}

// ---- attention -----------------------------------------------------------------------------------------------------------------
// One CTA per (kv head, slot, split): the 4 query heads of the group x the 8 block rows = 32 query rows share every K/V cell they read.
// Row t of a block at position p sees cells [max(lo, p + t - 2047), p + 7] (not causal inside the block). Partial results (running max, sum,
// unnormalized output) are combined by k_df2_attn_combine.
constexpr int kKStride = 132;   // halves per K row in shared memory (66 words: 8 cell groups hit 8 different banks)
__global__ void __launch_bounds__(256) k_df2_attn_part(const float* __restrict__ qr, __half* const* __restrict__ rk_tab,
                                                       __half* const* __restrict__ rv_tab, const int32_t* __restrict__ meta,
                                                       const int32_t* __restrict__ hi_ctx, int layer, int split, float* __restrict__ part_m,
                                                       float* __restrict__ part_l, float* __restrict__ part_o) {
    __shared__ __align__(16) float Qs[32][kHeadDim + 4];   // +4: the 4 rows of a warp read 4 different banks
    __shared__ __align__(16) __half Ks[kTileCells][kKStride];
    __shared__ __align__(16) __half Vs[kTileCells][kHeadDim];
    __shared__ float Ps[32][kTileCells + 1];
    const int h = blockIdx.x, b = blockIdx.y, sp = blockIdx.z;
    const int tid = threadIdx.x;
    const int slot = meta[b * 4 + 0];
    const int p = meta[b * 4 + 1];
    const int lo = meta[b * 4 + 2];
    const int hc = hi_ctx[b];   // the context ends at hc; cells hc .. p - 1 (a block that starts past it) hold nothing valid
    const int kmin = max(lo, p - (kWindow - 1));
    const int kmax = p + kBlock - 1;
    const int ncell = kmax - kmin + 1;
    const int cs = (ncell + split - 1) / split;
    const int c_begin = kmin + sp * cs;
    const int c_end = min(kmax + 1, c_begin + cs);

    // Q rows j = qh * 8 + t -> (row b * 8 + t, head 4 * h + qh), scaled by 1/sqrt(128)
    const float qscale = 0.08838834764831845f;
    for (int e = tid; e < 32 * kHeadDim; e += 256) {
        const int j = e / kHeadDim, d = e % kHeadDim;
        const int qh = j / kBlock, t = j % kBlock;
        Qs[j][d] = qr[(((size_t)(b * kBlock + t)) * kHeads + 4 * h + qh) * kHeadDim + d] * qscale;
    }
    const int j = tid >> 3;     // query row 0..31 of this thread (scores and output)
    const int g = tid & 7;      // cell group (scores: cells g + 8 i) / dim chunk (output: dims g * 16 ..)
    const int t_row = j % kBlock;
    const int win_lo = p + t_row - (kWindow - 1);   // first key position row j may see
    float m_run = -3.0e38f, l_run = 0.0f;
    float acc[16];
#pragma unroll
    for (int i = 0; i < 16; ++i) acc[i] = 0.0f;
    const __half* rk = rk_tab[slot];
    const __half* rv = rv_tab[slot];
    const size_t slot_base = (size_t)layer * kRingCap;

    for (int c0 = c_begin; c0 < c_end; c0 += kTileCells) {
        __syncthreads();
        // load K and V tiles (32 cells x 128 halves each = 512 int4 per tensor)
        for (int e = tid; e < kTileCells * 16; e += 256) {
            const int cc = e >> 4, part = e & 15;
            const int cell = c0 + cc;
            int4 kv = make_int4(0, 0, 0, 0), vv = make_int4(0, 0, 0, 0);
            if (cell < c_end) {
                const size_t off = (slot_base + (size_t)(cell % kRingCap)) * kKV + (size_t)h * kHeadDim + part * 8;
                kv = *reinterpret_cast<const int4*>(rk + off);
                vv = *reinterpret_cast<const int4*>(rv + off);
            }
            // K rows have a 132-half stride (not 16-byte aligned): store as 4 x 32-bit words
            uint32_t* kd = reinterpret_cast<uint32_t*>(&Ks[cc][part * 8]);
            kd[0] = (uint32_t)kv.x; kd[1] = (uint32_t)kv.y; kd[2] = (uint32_t)kv.z; kd[3] = (uint32_t)kv.w;
            *reinterpret_cast<int4*>(&Vs[cc][part * 8]) = vv;
        }
        __syncthreads();
        // scores for cells g, g + 8, g + 16, g + 24 of this tile
        float s4[4];
#pragma unroll
        for (int i = 0; i < 4; ++i) s4[i] = 0.0f;
#pragma unroll 8
        for (int d = 0; d < kHeadDim; d += 2) {
            const float2 q2 = *reinterpret_cast<const float2*>(&Qs[j][d]);
#pragma unroll
            for (int i = 0; i < 4; ++i) {
                const __half2 k2 = *reinterpret_cast<const __half2*>(&Ks[g + 8 * i][d]);
                const float2 kf = __half22float2(k2);
                s4[i] = fmaf(q2.x, kf.x, fmaf(q2.y, kf.y, s4[i]));
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
        for (int o = 1; o < 8; o <<= 1) tmax = fmaxf(tmax, __shfl_xor_sync(0xFFFFFFFFu, tmax, o));
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
        for (int o = 1; o < 8; o <<= 1) lsum += __shfl_xor_sync(0xFFFFFFFFu, lsum, o);
        l_run = l_run * alpha + lsum;
        m_run = m_new;
#pragma unroll
        for (int i = 0; i < 16; ++i) acc[i] *= alpha;
        __syncwarp();
        __syncthreads();
        // P V: this thread owns dims g * 16 .. g * 16 + 15 of row j
        for (int cc = 0; cc < kTileCells; ++cc) {
            const float pv = Ps[j][cc];
            if (pv != 0.0f) {
                const int4 v0 = *reinterpret_cast<const int4*>(&Vs[cc][g * 16]);
                const int4 v1 = *reinterpret_cast<const int4*>(&Vs[cc][g * 16 + 8]);
                const __half2* h0 = reinterpret_cast<const __half2*>(&v0);
                const __half2* h1 = reinterpret_cast<const __half2*>(&v1);
#pragma unroll
                for (int i = 0; i < 4; ++i) {
                    const float2 a = __half22float2(h0[i]);
                    const float2 c = __half22float2(h1[i]);
                    acc[2 * i] = fmaf(pv, a.x, acc[2 * i]);
                    acc[2 * i + 1] = fmaf(pv, a.y, acc[2 * i + 1]);
                    acc[8 + 2 * i] = fmaf(pv, c.x, acc[8 + 2 * i]);
                    acc[8 + 2 * i + 1] = fmaf(pv, c.y, acc[8 + 2 * i + 1]);
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
__global__ void __launch_bounds__(128) k_df2_attn_combine(const float* __restrict__ part_m, const float* __restrict__ part_l,
                                                          const float* __restrict__ part_o, int split, float* __restrict__ out) {
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

// ---- head: top-16 and the full-vocabulary tail from the int8 tile GEMM, then the selector ------------------------------------------
__device__ __forceinline__ bool df2_better(float a, int ia, float b, int ib) { return a > b || (a == b && ia < ib); }
__device__ __forceinline__ bool df2_strict(float a, int ia, int ja, float b, int ib, int jb) {
    return a > b || (a == b && (ia < ib || (ia == ib && ja < jb)));
}

// Stage 1 (adapted from tjm8874's top-16 part kernel, MIT): grid (chunks of this tile, rows), 256 threads = 8 warps. Each warp extracts its top-16
// by warp shuffles; 128 threads rank the 128 warp winners. Also the softcapped chunk (max, sum exp) over every column.
// tile_i32 [R_pad][vt] int8 dot products of the target head rows [v0, v0 + vt) with the quantized draft hidden of row r.
__global__ void __launch_bounds__(256) k_df2_tile_part(const int32_t* __restrict__ tile_i32, const float* __restrict__ head_sc,
                                                       const float* __restrict__ h_sc, int v0, int vt, int V, float cap,
                                                       int chunk_base, float2* __restrict__ part, float2* __restrict__ part_lse) {
    __shared__ float w_val[8 * kTopK];
    __shared__ int w_id[8 * kTopK];
    __shared__ float lse_sh[8];
    const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
    const int r = blockIdx.y, chunk = blockIdx.x;
    const int c0 = chunk * kTopChunk;
    const int32_t* row = tile_i32 + (size_t)r * vt;
    const float hs = h_sc[r];
    float vals[16];
    uint32_t alive = 0u;
    float tm = -INFINITY;
#pragma unroll
    for (int j = 0; j < 16; ++j) {
        const int col = c0 + j * 256 + tid;
        if (col < vt && (v0 + col) < V) {
            const float v = (float)row[col] * head_sc[v0 + col] * hs;
            vals[j] = v;
            alive |= (1u << j);
            tm = fmaxf(tm, df2_softcap(v, cap));
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
                const int col = v0 + c0 + j * 256 + tid;
                if (df2_better(vals[j], col, bv, bi)) { bv = vals[j]; bi = col; bj = j; }
            }
        }
        int wl = lane, wj = bj;
#pragma unroll
        for (int o = 16; o > 0; o >>= 1) {
            const float ov = __shfl_xor_sync(0xffffffffu, bv, o);
            const int oi = __shfl_xor_sync(0xffffffffu, bi, o);
            const int ol = __shfl_xor_sync(0xffffffffu, wl, o);
            const int oj = __shfl_xor_sync(0xffffffffu, wj, o);
            if (df2_better(ov, oi, bv, bi)) { bv = ov; bi = oi; wl = ol; wj = oj; }
        }
        if (lane == wl && wj >= 0 && wj < 16) alive &= ~(1u << wj);
        if (lane == 0) { w_val[warp * kTopK + k] = bv; w_id[warp * kTopK + k] = bi; }
    }
    {
        const float wm = warp_max(tm);
        if (lane == 0) lse_sh[warp] = wm;
    }
    __syncthreads();
    float2* dst = part + ((size_t)r * 64 + chunk_base + chunk) * kTopK;
    if (tid < 8 * kTopK) {
        const float mv = w_val[tid];
        const int mi = w_id[tid];
        int rk = 0;
        for (int j = 0; j < 8 * kTopK; ++j)
            if (df2_strict(w_val[j], w_id[j], j, mv, mi, tid)) ++rk;
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
            if (col < vt && (v0 + col) < V) tl += __expf(df2_softcap(vals[j], cap) - cm);
        }
    }
    const float wl2 = warp_sum(tl);
    __syncthreads();
    if (lane == 0) lse_sh[warp] = wl2;
    __syncthreads();
    if (tid == 0) {
        float sl = 0.0f;
#pragma unroll
        for (int w = 0; w < 8; ++w) sl += lse_sh[w];
        part_lse[(size_t)r * 64 + chunk_base + chunk] = make_float2(cm, sl);
    }
}

// Stage 2: grid (rows), 512 threads. Merge the n_chunks x 16 chunk winners into the exact top-16 (ids + raw logits) and the tail: the softcapped
// full-vocabulary partition function without the 16 winners (max m, sum l).
__global__ void __launch_bounds__(512) k_df2_top16_merge(const float2* __restrict__ part, const float2* __restrict__ part_lse,
                                                         int n_chunks, int V, float cap, int32_t* __restrict__ cand_out,
                                                         float* __restrict__ tail_out) {
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
        const float m_all = warp_max(fmaxf(p0.x, p1.x));
        float l_loc = 0.0f;
        if (isfinite(m_all)) {
            if (c0 < n_chunks && isfinite(p0.x)) l_loc += p0.y * __expf(p0.x - m_all);
            if (c1 < n_chunks && isfinite(p1.x)) l_loc += p1.y * __expf(p1.x - m_all);
        }
        const float l_all = warp_sum(l_loc);
        if (lane == 0) { m_all_sh = m_all; l_all_sh = l_all; }
    }
    __syncthreads();
    int rk0 = 0, rk1 = 0;
    for (int j = 0; j < n_cand; ++j) {
        const float vj = all_v[j];
        const int ij = all_id[j];
        if (tid < n_cand && df2_strict(vj, ij, j, v0, id0, tid)) ++rk0;
        if (tid + 512 < n_cand && df2_strict(vj, ij, j, v1, id1, tid + 512)) ++rk1;
    }
    int32_t* out = cand_out + (size_t)r * 2 * kTopK;
    if (tid < n_cand && rk0 < kTopK) { out[rk0] = id0; out[kTopK + rk0] = __float_as_int(v0); win_v[rk0] = v0; }
    if (tid + 512 < n_cand && rk1 < kTopK) { out[rk1] = id1; out[kTopK + rk1] = __float_as_int(v1); win_v[rk1] = v1; }
    __syncthreads();
    if (warp == 0) {
        const float m_all = m_all_sh;
        const float u = lane < kTopK ? win_v[lane] : -INFINITY;
        const float term = (isfinite(u) && isfinite(m_all)) ? __expf(df2_softcap(u, cap) - m_all) : 0.0f;
        const float s16 = warp_sum(term);
        if (lane == 0) {
            tail_out[2 * r] = m_all;
            tail_out[2 * r + 1] = fmaxf(0.0f, l_all_sh - s16);
        }
    }
}

// The bilinear candidate selector (adapted from tjm8874's select kernel, MIT): one CTA of 512 threads (16 warps = 16 candidates) per slot, the 7 draft
// steps one after the other: score(c) = raw(c) + sum_k pred[prev][k] * g[k] * succ[c][k]; the best candidate is the step's token and the next step's
// prev; conf = softmax probability of the winner over the softcapped scores plus the tail.
// pred / succ: bf16 [V][256]. meta[b * 4 + 3] = the anchor token (the first step's prev).
__global__ void __launch_bounds__(512) k_df2_select(const int32_t* __restrict__ meta, const int32_t* __restrict__ cand,
                                                    const float* __restrict__ tail, const float* __restrict__ g,
                                                    const uint16_t* __restrict__ pred, const uint16_t* __restrict__ succ, int n_drafts,
                                                    int nv, float cap, int32_t* __restrict__ ids_out, float* __restrict__ conf_out) {
    __shared__ __align__(16) float pg[kRank];
    __shared__ float sc_sh[kTopK];
    __shared__ int32_t prev_s;
    const int s = blockIdx.x;
    const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
    for (int d = 0; d < n_drafts; ++d) {
        const int r = s * n_drafts + d;
        const int32_t prev = d == 0 ? meta[s * 4 + 3] : prev_s;
        if (tid < kRank) pg[tid] = (prev >= 0 && prev < nv) ? df2_bf16f(pred[(size_t)prev * kRank + tid]) * g[((size_t)s * kBlock + 1 + d) * kRank + tid] : 0.0f;
        __syncthreads();
        const int32_t id = cand[(size_t)r * 2 * kTopK + warp];
        float dot = 0.0f;
        if (id >= 0) {
            const uint4 sv = *(reinterpret_cast<const uint4*>(succ + (size_t)id * kRank) + lane);
            const float4 p0 = reinterpret_cast<const float4*>(pg)[2 * lane];
            const float4 p1 = reinterpret_cast<const float4*>(pg)[2 * lane + 1];
            dot += p0.x * df2_bf16f((uint16_t)(sv.x & 0xffffu));
            dot += p0.y * df2_bf16f((uint16_t)(sv.x >> 16));
            dot += p0.z * df2_bf16f((uint16_t)(sv.y & 0xffffu));
            dot += p0.w * df2_bf16f((uint16_t)(sv.y >> 16));
            dot += p1.x * df2_bf16f((uint16_t)(sv.z & 0xffffu));
            dot += p1.y * df2_bf16f((uint16_t)(sv.z >> 16));
            dot += p1.z * df2_bf16f((uint16_t)(sv.w & 0xffffu));
            dot += p1.w * df2_bf16f((uint16_t)(sv.w >> 16));
            dot = warp_sum(dot);
        }
        if (lane == 0) sc_sh[warp] = __int_as_float(cand[(size_t)r * 2 * kTopK + kTopK + warp]) + dot;
        __syncthreads();
        if (warp == 0) {
            float bv = lane < kTopK ? sc_sh[lane] : -INFINITY;
            int32_t bi = lane < kTopK ? cand[(size_t)r * 2 * kTopK + lane] : 0x7fffffff;
            const float my_sc = bv;
#pragma unroll
            for (int o = 8; o > 0; o >>= 1) {
                const float ov = __shfl_xor_sync(0xffffffffu, bv, o);
                const int32_t oi = __shfl_xor_sync(0xffffffffu, bi, o);
                if (df2_better(ov, oi, bv, bi)) { bv = ov; bi = oi; }
            }
            const float top = df2_softcap(bv, cap);
            float m_ref = top, tail_term = 0.0f;
            const float m_t = tail[2 * r];
            const float l_t = tail[2 * r + 1];
            if (isfinite(m_t) && l_t > 0.0f) {
                m_ref = fmaxf(top, m_t);
                tail_term = l_t * __expf(m_t - m_ref);
            }
            const float my_cap = lane < kTopK ? df2_softcap(my_sc, cap) : -INFINITY;
            const float z16 = warp_sum(lane < kTopK ? __expf(my_cap - m_ref) : 0.0f);
            if (lane == 0) {
                ids_out[s * n_drafts + d] = bi;
                conf_out[s * n_drafts + d] = __expf(top - m_ref) / (z16 + tail_term);
                prev_s = bi;
            }
        }
        __syncthreads();
    }
}

// ---- NV-OLDER: the FP32 / dp4a form of what needs sm_80 elsewhere (V100, RTX 20, GTX 10, P100) -----------------------------------------------------------------------

// [M][4 * 2304] fp32 from four fp32 tap planes (the same gather as k_df2_gather_taps, without the bf16 rounding); row i reads plane row src_rows[i] (or i when src_rows is null).
__global__ void __launch_bounds__(256) k_df2_gather_taps_f32(const float* __restrict__ t0, const float* __restrict__ t1,
                                                             const float* __restrict__ t2, const float* __restrict__ t3,
                                                             const int32_t* __restrict__ src_rows, int row0, int M, float* __restrict__ out) {
    const int i = blockIdx.x;
    if (i >= M) return;
    const int src = src_rows ? src_rows[row0 + i] : (row0 + i);
    const float* tp[4] = {t0, t1, t2, t3};
    float* dst = out + (size_t)i * kTargets * kHidden;
#pragma unroll
    for (int t = 0; t < kTargets; ++t) {
        const float4* s4 = reinterpret_cast<const float4*>(tp[t] + (size_t)src * kHidden);
        float4* d4 = reinterpret_cast<float4*>(dst + t * kHidden);
        for (int d = threadIdx.x; d < kHidden / 4; d += 256) d4[d] = s4[d];
    }
}

// The head screen of the draft where the cuBLAS INT8 GEMM does not exist (sm_60): tile[c][v] = sum_k W[v][k] * h[c][k] (int8 x int8 -> int32) for v < vt, c < ncols, the layout and
// values of the GEMM the other cards call. One warp per weight row, the row read once per group of 8 columns; dp4a_i8 (the exact integer emulation on sm_60).
constexpr int kScreenCols = 8;
__global__ void __launch_bounds__(256) k_df2_screen_gemm(const int8_t* __restrict__ W, const int8_t* __restrict__ hq, int32_t* __restrict__ tile, int vt, int ncols) {
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const int v = blockIdx.x * 8 + warp;
    if (v >= vt) return;
    const uint4* w16 = reinterpret_cast<const uint4*>(W + (size_t)v * kHidden);
    constexpr int kNC = kHidden / 16;   // 144 chunks of 16 bytes
    for (int c0 = 0; c0 < ncols; c0 += kScreenCols) {
        int acc[kScreenCols];
#pragma unroll
        for (int j = 0; j < kScreenCols; ++j) acc[j] = 0;
        for (int c = lane; c < kNC; c += 32) {
            const uint4 w = w16[c];
#pragma unroll
            for (int j = 0; j < kScreenCols; ++j) {
                if (c0 + j < ncols) {
                    const uint4 a = *reinterpret_cast<const uint4*>(hq + (size_t)(c0 + j) * kHidden + (size_t)c * 16);
                    int p = dp4a_i8(w.x, a.x, 0);
                    p = dp4a_i8(w.y, a.y, p);
                    p = dp4a_i8(w.z, a.z, p);
                    p = dp4a_i8(w.w, a.w, p);
                    acc[j] += p;
                }
            }
        }
#pragma unroll
        for (int j = 0; j < kScreenCols; ++j) {
            int s = acc[j];
#pragma unroll
            for (int o = 16; o >= 1; o >>= 1) s += __shfl_xor_sync(0xFFFFFFFFu, s, o);
            if (lane == 0 && c0 + j < ncols) tile[(size_t)(c0 + j) * vt + v] = s;
        }
    }
}

}  // namespace df2
}  // namespace strata::ink
