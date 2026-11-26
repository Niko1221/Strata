// engine/src/inkling2/hip/ink_hip_strixhalo_megakernel.hip.h
// AMD Strix Halo (gfx1151, 20 WGPs / 40 CUs, 128 GB UMA, Windows 11 WDDM):
//   SF4 + Outlier-16 Dense Trunk (QKVRG + WO) + Uniform SF4 Coda Experts (L20..23):
//     - Explicitly protects Goodfire channel d_2027 (+32.4 RMS) + top-15 outlier channels in lossless BF16
//       with FP32 per-row outer scales while packing smooth channels into 4-bit E2M1 SF4 (~355 MiB vs 1.37 GiB BF16).
//     - Transcodes MX8 Coda experts (8.21 MiB -> 4.27 MiB) into uniform CODE_SF4 at load (FAST mode only).
//     - Strictly opt-in under STRATA_INK_STRIXHALO=1 (FAST mode only; ZERO effect when OFF or EXACT).
// MIT License - Copyright (c) 2026 Niko1221 and the Strata contributors.
#pragma once

#include "ink_hip_strixhalo_decode.hip.h"

namespace strata::inkling2::hip {

// =================================================================================================
// Part 1: SF4 (E2M1) + FP8_E4M3 Encoding & Top-16 Outlier-Split Trunk + Uniform SF4 Coda Experts
// =================================================================================================

__device__ __forceinline__ uint8_t f32_to_sf4_nibble(float v, float inv_sc) {
    const uint8_t sgn = (v < 0.0f) ? 8u : 0u;
    const float a = fabsf(v) * inv_sc;
    uint8_t mag = 0u;
    if (a >= 5.00f)      mag = 7u;  // 6.0
    else if (a >= 3.50f) mag = 6u;  // 4.0
    else if (a >= 2.50f) mag = 5u;  // 3.0
    else if (a >= 1.75f) mag = 4u;  // 2.0
    else if (a >= 1.25f) mag = 3u;  // 1.5
    else if (a >= 0.75f) mag = 2u;  // 1.0
    else if (a >= 0.25f) mag = 1u;  // 0.5
    return static_cast<uint8_t>(sgn | mag);
}

// Encode non-negative float x in [0, 448.0] -> FP8_E4M3FN byte (0x00..0x7E)
__device__ __forceinline__ uint8_t f32_to_e4m3_pos(float x) {
    if (!(x > 0.0f)) return 0u;
    if (x >= 448.0f) return 0x7Eu;
    const uint32_t bits = __float_as_uint(x);
    int exp = static_cast<int>((bits >> 23) & 0xFFu) - 127;
    uint32_t mant = bits & 0x7FFFFFu;
    if (exp < -9) return 0u;
    if (exp < -6) {
        // Subnormal E4M3: value = (m / 8) * 2^-6 = m * 2^-9
        const int shift = -6 - exp;
        uint32_t full = (1u << 23) | mant;
        uint32_t q = (full + (1u << (19 + shift))) >> (20 + shift);
        if (q > 7u) return 0x08u;  // smallest normal
        return static_cast<uint8_t>(q);
    }
    int e4 = exp + 7;
    // Round-to-nearest on top 3 mantissa bits (bits 22..20)
    uint32_t m3 = (mant + (1u << 19)) >> 20;
    if (m3 >= 8u) {
        m3 = 0u;
        ++e4;
    }
    if (e4 >= 15) {
        if (m3 >= 7u) return 0x7Eu;  // 0x7F is NaN in E4M3FN!
        e4 = 15;
    }
    return static_cast<uint8_t>((e4 << 3) | (m3 & 7u));
}

struct Sf4Outlier16Matrix {
    strata::hip_uma::GpuBuffer d_p;          // [N, K/2] uint8 packed SF4 nibbles
    strata::hip_uma::GpuBuffer d_s;          // [N, K/16] uint8 E4M3 per-16 scales
    strata::hip_uma::GpuBuffer d_row_outer;  // [N] float per-row outer scale (0 FP16 underflow!)
    strata::hip_uma::GpuBuffer d_out16;      // [N, 16] half_t exact FP16 outlier column weights
    int N = 0;
    int K = 0;
    bool ready = false;

    void release() noexcept {
        strata::hip_uma::free_gpu_buffer(d_p);
        strata::hip_uma::free_gpu_buffer(d_s);
        strata::hip_uma::free_gpu_buffer(d_row_outer);
        strata::hip_uma::free_gpu_buffer(d_out16);
        N = K = 0;
        ready = false;
    }
    ~Sf4Outlier16Matrix() { release(); }
};

// Accumulate column L2 energy of a BF16 [N, K] matrix into col_energy[K] (1 pass at init!)
// Grid: ((K + 255) / 256,), Block: 256
__global__ void __launch_bounds__(256) k_halo_accum_col_energy_bf16(
    const uint16_t* __restrict__ W_bf16,
    float* __restrict__ col_energy,
    int N, int K) {
#if defined(__HIP_DEVICE_COMPILE__) && !defined(__gfx1151__)
    return;
#endif

    const int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= K) return;
    float s = 0.0f;
    // Sample every 4th row for <1 ms init speed while capturing exact column magnitude hierarchy
    for (int n = 0; n < N; n += 4) {
        const float v = bf16_to_f32(W_bf16[static_cast<size_t>(n) * K + c]);
        s = fmaf(v, v, s);
    }
    col_energy[c] += s;
}

// Transcode BF16 [N, K] -> SF4 + Outlier-16 (1 block = 256 threads per row n -> <15 ms for all 24 layers!)
// Grid: (N,), Block: 256
__global__ void __launch_bounds__(256) k_quant_bf16_to_sf4_out16(
    const uint16_t* __restrict__ W_bf16,    // [N, K]
    const int32_t* __restrict__ cols16,     // [16] outlier channel indices
    uint8_t* __restrict__ W_p,              // [N, K/2]
    uint8_t* __restrict__ W_s,              // [N, K/16]
    float* __restrict__ row_outer_out,      // [N] FP32 outer scale (0 subnormal underflow!)
    uint16_t* __restrict__ W_out16,         // [N, 16] lossless BF16 outlier column weights
    int N, int K) {
#if defined(__HIP_DEVICE_COMPILE__) && !defined(__gfx1151__)
    return;
#endif

    const int n = blockIdx.x;
    if (n >= N) return;
    const int tid = threadIdx.x;
    const int wave = tid >> 5;
    const int lane = tid & 31;

    __shared__ int32_t s_cols[16];
    __shared__ float s_wmax[8];
    if (tid < 16) s_cols[tid] = cols16[tid];
    __syncthreads();

    const uint16_t* src = W_bf16 + static_cast<size_t>(n) * K;
    if (tid < 16) {
        const int c = s_cols[tid];
        W_out16[static_cast<size_t>(n) * 16 + tid] = (c >= 0 && c < K) ? src[c] : 0u;
    }

    auto is_outlier = [&](int col) -> bool {
#pragma unroll
        for (int i = 0; i < 16; ++i) {
            if (col == s_cols[i]) return true;
        }
        return false;
    };

    const int nblk = K >> 4;
    float local_max_blk = 0.0f;
    for (int b = tid; b < nblk; b += 256) {
        const int c0 = b << 4;
        float bmax = 0.0f;
#pragma unroll
        for (int i = 0; i < 16; ++i) {
            if (!is_outlier(c0 + i)) {
                bmax = fmaxf(bmax, fabsf(bf16_to_f32(src[c0 + i])));
            }
        }
        local_max_blk = fmaxf(local_max_blk, bmax);
    }
    for (int off = 16; off > 0; off >>= 1) local_max_blk = fmaxf(local_max_blk, __shfl_xor(local_max_blk, off, 32));
    if (lane == 0) s_wmax[wave] = local_max_blk;
    __syncthreads();
    if (tid == 0) {
        float gmax = 0.0f;
        for (int w = 0; w < 8; ++w) gmax = fmaxf(gmax, s_wmax[w]);
        const float ro = (gmax > 1e-12f) ? (gmax / (6.0f * 64.0f)) : 1.0f;
        s_wmax[0] = ro;
        row_outer_out[n] = ro;
    }
    __syncthreads();
    const float row_outer = s_wmax[0];
    const float inv_ro = (row_outer > 1e-30f) ? (1.0f / row_outer) : 0.0f;

    uint8_t* dp = W_p + static_cast<size_t>(n) * (K >> 1);
    uint8_t* ds = W_s + static_cast<size_t>(n) * nblk;
    for (int b = tid; b < nblk; b += 256) {
        const int c0 = b << 4;
        float vals[16];
        float bmax = 0.0f;
#pragma unroll
        for (int i = 0; i < 16; ++i) {
            const float v = is_outlier(c0 + i) ? 0.0f : bf16_to_f32(src[c0 + i]);
            vals[i] = v;
            bmax = fmaxf(bmax, fabsf(v));
        }
        const uint8_t e8 = f32_to_e4m3_pos((bmax * (1.0f / 6.0f)) * inv_ro);
        ds[b] = e8;
        const float eff_sc = e4m3fn_to_f32(e8) * row_outer;
        const float inv_sc = (eff_sc > 1e-30f) ? (1.0f / eff_sc) : 0.0f;
        uint8_t* pb = dp + b * 8;
#pragma unroll
        for (int i = 0; i < 8; ++i) {
            const uint8_t lo = f32_to_sf4_nibble(vals[2 * i + 0], inv_sc);
            const uint8_t hi = f32_to_sf4_nibble(vals[2 * i + 1], inv_sc);
            pb[i] = static_cast<uint8_t>((lo & 15u) | ((hi & 15u) << 4));
        }
    }
}

// Transcode an MX8 matrix [N, K] (W_e4m3[N, K] + S_ue8m0[N, K/32]) into standard CODE_SF4 layout!
// Pass 1: find max abs value across matrix -> writes outer = max_abs / (6 * 440) into hdr_pair[0..1].
// Grid: (N,), Block: 256
__global__ void __launch_bounds__(256) k_mx8_find_max_halo(
    const uint8_t* __restrict__ W_mx8,
    const uint8_t* __restrict__ S_mx8,
    uint32_t* __restrict__ max_bits_out,  // single uint32 zero-initialized
    int N, int K) {
#if defined(__HIP_DEVICE_COMPILE__) && !defined(__gfx1151__)
    return;
#endif

    const int n = blockIdx.x;
    if (n >= N) return;
    const int tid = threadIdx.x;
    const int nblk32 = K >> 5;
    const uint8_t* wr = W_mx8 + static_cast<size_t>(n) * K;
    const uint8_t* sr = S_mx8 + static_cast<size_t>(n) * nblk32;
    float mx = 0.0f;
    for (int b = tid; b < nblk32; b += 256) {
        const float sc = ue8m0_to_f32(sr[b]);
        const uint8_t* wb = wr + (b << 5);
        float bm = 0.0f;
#pragma unroll
        for (int i = 0; i < 32; ++i) bm = fmaxf(bm, fabsf(e4m3fn_to_f32(wb[i])));
        mx = fmaxf(mx, bm * sc);
    }
    for (int off = 16; off > 0; off >>= 1) mx = fmaxf(mx, __shfl_xor(mx, off, 32));
    if ((tid & 31) == 0 && mx > 0.0f) atomicMax(max_bits_out, __float_as_uint(mx));
}

// Pass 2: encode MX8 [N, K] -> SF4 packed [N, K/2] + E4M3 scales [N, K/16] + hdr_pair[0] = outer, hdr_pair[1] = 1.0f!
// Grid: (N,), Block: 256
__global__ void __launch_bounds__(256) k_mx8_to_sf4_halo(
    const uint8_t* __restrict__ W_mx8,
    const uint8_t* __restrict__ S_mx8,
    const uint32_t* __restrict__ max_bits_in,
    float* __restrict__ hdr_pair,
    uint8_t* __restrict__ sf4_p,
    uint8_t* __restrict__ sf4_s,
    int N, int K) {
#if defined(__HIP_DEVICE_COMPILE__) && !defined(__gfx1151__)
    return;
#endif

    const int n = blockIdx.x;
    if (n >= N) return;
    const int tid = threadIdx.x;
    const float gmax = __uint_as_float(*max_bits_in);
    const float outer = (gmax > 1e-12f) ? (gmax / (6.0f * 64.0f)) : 1.0f;
    if (n == 0 && tid == 0) {
        hdr_pair[0] = outer;
        hdr_pair[1] = 1.0f;
    }
    const float inv_outer = (outer > 1e-30f) ? (1.0f / outer) : 0.0f;
    const int nblk16 = K >> 4;
    const int nblk32 = K >> 5;
    const uint8_t* wr = W_mx8 + static_cast<size_t>(n) * K;
    const uint8_t* sr = S_mx8 + static_cast<size_t>(n) * nblk32;
    uint8_t* dp = sf4_p + static_cast<size_t>(n) * (K >> 1);
    uint8_t* ds = sf4_s + static_cast<size_t>(n) * nblk16;

    for (int b = tid; b < nblk16; b += 256) {
        const float sc32 = ue8m0_to_f32(sr[b >> 1]);
        const uint8_t* wb = wr + (b << 4);
        float vals[16];
        float bmax = 0.0f;
#pragma unroll
        for (int i = 0; i < 16; ++i) {
            const float v = e4m3fn_to_f32(wb[i]) * sc32;
            vals[i] = v;
            bmax = fmaxf(bmax, fabsf(v));
        }
        const uint8_t e8 = f32_to_e4m3_pos((bmax * (1.0f / 6.0f)) * inv_outer);
        ds[b] = e8;
        const float eff_sc = e4m3fn_to_f32(e8) * outer;
        const float inv_sc = (eff_sc > 1e-30f) ? (1.0f / eff_sc) : 0.0f;
        uint8_t* pb = dp + b * 8;
#pragma unroll
        for (int i = 0; i < 8; ++i) {
            const uint8_t lo = f32_to_sf4_nibble(vals[2 * i + 0], inv_sc);
            const uint8_t hi = f32_to_sf4_nibble(vals[2 * i + 1], inv_sc);
            pb[i] = static_cast<uint8_t>((lo & 15u) | ((hi & 15u) << 4));
        }
    }
}

// Standalone Outlier-16 split + FP16 smooth pack for 1 or 3 input tensors (used outside megakernel!)
// Grid: ((M * (K/4) + 255) / 256, n_br), Block: 256
__global__ void __launch_bounds__(256) k_halo_split_out16_f16(
    const float* __restrict__ X0,
    const float* __restrict__ X1,
    const float* __restrict__ X2,
    const int32_t* __restrict__ cols16,   // [16]
    half_t* __restrict__ X_smooth_f16,    // [n_br, M, K]
    float* __restrict__ X_out16,          // [n_br, M, 16]
    int M, int K, int n_br) {
#if defined(__HIP_DEVICE_COMPILE__) && !defined(__gfx1151__)
    return;
#endif

    const int br = blockIdx.y;
    if (br >= n_br) return;
    const float* X = (br == 0) ? X0 : ((br == 1) ? X1 : X2);
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    const int k4 = K >> 2;
    if (tid < M * 16) {
        const int m = tid >> 4;
        const int ci = tid & 15;
        const int col = cols16[ci];
        X_out16[(static_cast<size_t>(br) * M + m) * 16 + ci] =
            (col >= 0 && col < K) ? X[static_cast<size_t>(m) * K + col] : 0.0f;
    }
    if (tid >= M * k4) return;
    const int m = tid / k4;
    const int c4 = tid - m * k4;
    const int col0 = c4 << 2;
    float4 v = *reinterpret_cast<const float4*>(X + static_cast<size_t>(m) * K + col0);
#pragma unroll
    for (int i = 0; i < 16; ++i) {
        const int oc = cols16[i];
        if (oc == col0 + 0) v.x = 0.0f;
        else if (oc == col0 + 1) v.y = 0.0f;
        else if (oc == col0 + 2) v.z = 0.0f;
        else if (oc == col0 + 3) v.w = 0.0f;
    }
    *reinterpret_cast<uint2*>(X_smooth_f16 + (static_cast<size_t>(br) * M + m) * K + col0) =
        make_uint2(h2_pack(v.x, v.y), h2_pack(v.z, v.w));
}

// Standalone SF4 + Outlier-16 QKVRG + Direct Split + Optional FP32 Attn-Sink Proj (M in 1..16!)
// Grid: ((tot_n + n_sink + 7) / 8,), Block: 256
template <int MT, bool THREE_IN>
__global__ void __launch_bounds__(256) k_halo_sf4_out16_qkvrg_split(
    const uint8_t* __restrict__ W_p,
    const uint8_t* __restrict__ W_s,
    const float* __restrict__ row_outer,
    const uint16_t* __restrict__ W_out16,
    const half_t* __restrict__ X_f16,     // [THREE_IN ? 3 : 1, M, K]
    const float* __restrict__ X_out16,    // [THREE_IN ? 3 : 1, M, 16]
    const float* __restrict__ X0_f32,     // [M, K] (for FP32 W_sink)
    const float* __restrict__ W_sink,     // nullptr or [n_sink, K]
    float* __restrict__ q_out,
    float* __restrict__ r_out,
    float* __restrict__ g_out,
    float* __restrict__ k_out,
    float* __restrict__ v_out,
    float* __restrict__ sinks_out,
    int M, int q_dim, int r_dim, int kv_dim, int n_sink, int K) {
#if defined(__HIP_DEVICE_COMPILE__) && !defined(__gfx1151__)
    return;
#endif

    const int tid = threadIdx.x;
    const int wave = tid >> 5;
    const int lane = tid & 31;
    const int row = blockIdx.x * 8 + wave;
    const int o_r = q_dim;
    const int o_g = q_dim + r_dim;
    const int o_k = o_g + q_dim;
    const int o_v = o_k + kv_dim;
    const int tot_n = o_v + kv_dim;

    __shared__ uint32_t s_sf4_pair[256];
    __shared__ float s_sf4_sc[256];
    {
        const hv2 hp = {sf4_e2m1_f16(static_cast<uint32_t>(tid) & 15u), sf4_e2m1_f16(static_cast<uint32_t>(tid) >> 4)};
        s_sf4_pair[tid] = __builtin_bit_cast(uint32_t, hp);
        s_sf4_sc[tid] = e4m3fn_to_f32(static_cast<uint8_t>(tid));
    }
    __syncthreads();
    if (row >= tot_n + n_sink) return;

    float acc[MT];
#pragma unroll
    for (int m = 0; m < MT; ++m) acc[m] = 0.0f;

    if (row >= tot_n) {
        const int sr_idx = row - tot_n;
        const float* ws = W_sink + static_cast<size_t>(sr_idx) * K;
        constexpr int kB = 6;  // Pattern a (#682): 6x float4 loads in flight!
        int k = lane * 4;
        for (; k + 128 * (kB - 1) < K; k += 128 * kB) {
            float4 wv[kB];
#pragma unroll
            for (int b = 0; b < kB; ++b) wv[b] = *reinterpret_cast<const float4*>(ws + k + 128 * b);
#pragma unroll
            for (int b = 0; b < kB; ++b) {
                const int kk = k + 128 * b;
#pragma unroll
                for (int m = 0; m < MT; ++m) {
                    if (m < M) {
                        const float4 xv = *reinterpret_cast<const float4*>(X0_f32 + static_cast<size_t>(m) * K + kk);
                        acc[m] = fmaf(wv[b].x, xv.x, fmaf(wv[b].y, xv.y, fmaf(wv[b].z, xv.z, fmaf(wv[b].w, xv.w, acc[m]))));
                    }
                }
            }
        }
        for (; k < K; k += 128) {
            const float4 wv = *reinterpret_cast<const float4*>(ws + k);
#pragma unroll
            for (int m = 0; m < MT; ++m) {
                if (m < M) {
                    const float4 xv = *reinterpret_cast<const float4*>(X0_f32 + static_cast<size_t>(m) * K + k);
                    acc[m] = fmaf(wv.x, xv.x, fmaf(wv.y, xv.y, fmaf(wv.z, xv.z, fmaf(wv.w, xv.w, acc[m]))));
                }
            }
        }
#pragma unroll
        for (int m = 0; m < MT; ++m) {
            if (m < M) {
                const float v = wave32_sum(acc[m]);
                if (lane == 0) sinks_out[static_cast<size_t>(m) * n_sink + sr_idx] = v;
            }
        }
        return;
    }

    int br = 0;
    if constexpr (THREE_IN) br = (row < o_k) ? 0 : ((row < o_v) ? 1 : 2);
    const half_t* xb = X_f16 + static_cast<size_t>(br) * M * K;
    const float*  ob = X_out16 + static_cast<size_t>(br) * M * 16;

    const float ro = static_cast<float>(row_outer[row]);
    const int nblk = K >> 4;
    const uint8_t* pr = W_p + static_cast<size_t>(row) * (K >> 1);
    const uint8_t* sr = W_s + static_cast<size_t>(row) * nblk;

    for (int blk = lane; blk < nblk; blk += 32) {
        const float sc = s_sf4_sc[sr[blk]] * ro;
        const uint2 raw = *reinterpret_cast<const uint2*>(pr + blk * 8);
        const uint32_t w0 = s_sf4_pair[raw.x & 0xFFu];
        const uint32_t w1 = s_sf4_pair[(raw.x >> 8) & 0xFFu];
        const uint32_t w2 = s_sf4_pair[(raw.x >> 16) & 0xFFu];
        const uint32_t w3 = s_sf4_pair[(raw.x >> 24) & 0xFFu];
        const uint32_t w4 = s_sf4_pair[raw.y & 0xFFu];
        const uint32_t w5 = s_sf4_pair[(raw.y >> 8) & 0xFFu];
        const uint32_t w6 = s_sf4_pair[(raw.y >> 16) & 0xFFu];
        const uint32_t w7 = s_sf4_pair[(raw.y >> 24) & 0xFFu];
        const int col = blk << 4;
#pragma unroll
        for (int m = 0; m < MT; ++m) {
            if (m < M) {
                const half_t* xs = xb + static_cast<size_t>(m) * K + col;
                const uint4 x0 = *reinterpret_cast<const uint4*>(xs);
                const uint4 x1 = *reinterpret_cast<const uint4*>(xs + 8);
                float d = fdot2_f32(w0, x0.x, 0.0f);
                d = fdot2_f32(w1, x0.y, d);
                d = fdot2_f32(w2, x0.z, d);
                d = fdot2_f32(w3, x0.w, d);
                d = fdot2_f32(w4, x1.x, d);
                d = fdot2_f32(w5, x1.y, d);
                d = fdot2_f32(w6, x1.z, d);
                d = fdot2_f32(w7, x1.w, d);
                acc[m] = fmaf(d, sc, acc[m]);
            }
        }
    }

    // Add exact FP32 Outlier-16 contribution (lane 0..15 handles 1 outlier channel each -> zero extra sync!)
    if (lane < 16) {
        const float wo = bf16_to_f32(W_out16[static_cast<size_t>(row) * 16 + lane]);
#pragma unroll
        for (int m = 0; m < MT; ++m) {
            if (m < M) acc[m] = fmaf(wo, ob[static_cast<size_t>(m) * 16 + lane], acc[m]);
        }
    }

    float* dst_base = nullptr;
    int dst_stride = 0, dst_col = 0;
    if (row < o_r)      { dst_base = q_out; dst_stride = q_dim;  dst_col = row; }
    else if (row < o_g) { dst_base = r_out; dst_stride = r_dim;  dst_col = row - o_r; }
    else if (row < o_k) { dst_base = g_out; dst_stride = q_dim;  dst_col = row - o_g; }
    else if (row < o_v) { dst_base = k_out; dst_stride = kv_dim; dst_col = row - o_k; }
    else                { dst_base = v_out; dst_stride = kv_dim; dst_col = row - o_v; }

#pragma unroll
    for (int m = 0; m < MT; ++m) {
        if (m < M) {
            const float v = wave32_sum(acc[m]);
            if (lane == 0) dst_base[static_cast<size_t>(m) * dst_stride + dst_col] = v;
        }
    }
}

// Standalone SF4 + Outlier-16 GEMV for WO [D, q_dim] (M in 1..16!)
// Grid: ((N + 7) / 8,), Block: 256
template <int MT>
__global__ void __launch_bounds__(256) k_halo_sf4_out16_gemv(
    const uint8_t* __restrict__ W_p,
    const uint8_t* __restrict__ W_s,
    const float* __restrict__ row_outer,
    const uint16_t* __restrict__ W_out16,
    const half_t* __restrict__ X_f16,
    const float* __restrict__ X_out16,
    float* __restrict__ Y_out,
    int M, int N, int K) {
#if defined(__HIP_DEVICE_COMPILE__) && !defined(__gfx1151__)
    return;
#endif

    const int tid = threadIdx.x;
    const int wave = tid >> 5;
    const int lane = tid & 31;
    const int row = blockIdx.x * 8 + wave;

    __shared__ uint32_t s_sf4_pair[256];
    __shared__ float s_sf4_sc[256];
    {
        const hv2 hp = {sf4_e2m1_f16(static_cast<uint32_t>(tid) & 15u), sf4_e2m1_f16(static_cast<uint32_t>(tid) >> 4)};
        s_sf4_pair[tid] = __builtin_bit_cast(uint32_t, hp);
        s_sf4_sc[tid] = e4m3fn_to_f32(static_cast<uint8_t>(tid));
    }
    __syncthreads();
    if (row >= N) return;

    float acc[MT];
#pragma unroll
    for (int m = 0; m < MT; ++m) acc[m] = 0.0f;

    const float ro = static_cast<float>(row_outer[row]);
    const int nblk = K >> 4;
    const uint8_t* pr = W_p + static_cast<size_t>(row) * (K >> 1);
    const uint8_t* sr = W_s + static_cast<size_t>(row) * nblk;

    for (int blk = lane; blk < nblk; blk += 32) {
        const float sc = s_sf4_sc[sr[blk]] * ro;
        const uint2 raw = *reinterpret_cast<const uint2*>(pr + blk * 8);
        const uint32_t w0 = s_sf4_pair[raw.x & 0xFFu];
        const uint32_t w1 = s_sf4_pair[(raw.x >> 8) & 0xFFu];
        const uint32_t w2 = s_sf4_pair[(raw.x >> 16) & 0xFFu];
        const uint32_t w3 = s_sf4_pair[(raw.x >> 24) & 0xFFu];
        const uint32_t w4 = s_sf4_pair[raw.y & 0xFFu];
        const uint32_t w5 = s_sf4_pair[(raw.y >> 8) & 0xFFu];
        const uint32_t w6 = s_sf4_pair[(raw.y >> 16) & 0xFFu];
        const uint32_t w7 = s_sf4_pair[(raw.y >> 24) & 0xFFu];
        const int col = blk << 4;
#pragma unroll
        for (int m = 0; m < MT; ++m) {
            if (m < M) {
                const half_t* xs = X_f16 + static_cast<size_t>(m) * K + col;
                const uint4 x0 = *reinterpret_cast<const uint4*>(xs);
                const uint4 x1 = *reinterpret_cast<const uint4*>(xs + 8);
                float d = fdot2_f32(w0, x0.x, 0.0f);
                d = fdot2_f32(w1, x0.y, d);
                d = fdot2_f32(w2, x0.z, d);
                d = fdot2_f32(w3, x0.w, d);
                d = fdot2_f32(w4, x1.x, d);
                d = fdot2_f32(w5, x1.y, d);
                d = fdot2_f32(w6, x1.z, d);
                d = fdot2_f32(w7, x1.w, d);
                acc[m] = fmaf(d, sc, acc[m]);
            }
        }
    }
    if (lane < 16) {
        const float wo = bf16_to_f32(W_out16[static_cast<size_t>(row) * 16 + lane]);
#pragma unroll
        for (int m = 0; m < MT; ++m) {
            if (m < M) acc[m] = fmaf(wo, X_out16[static_cast<size_t>(m) * 16 + lane], acc[m]);
        }
    }
#pragma unroll
    for (int m = 0; m < MT; ++m) {
        if (m < M) {
            const float v = wave32_sum(acc[m]);
            if (lane == 0) Y_out[static_cast<size_t>(m) * N + row] = v;
        }
    }
}


// =================================================================================================
// StrixHaloState: Single self-contained opt-in state holder (nullptr on HipBackend when OFF!)
// =================================================================================================
struct StrixHaloState {
    StrixHaloCfg cfg{};
    UmaSlabArena slab_arena;
    WddmTdrSlicer tdr_slicer;
    UnembedScreenU4 screen_u4;
    Xdna2NpuProbe npu_probe;
    std::vector<Q8TrunkMatrix> q8_qkvrg;
    std::vector<Q8TrunkMatrix> q8_wo;
    std::vector<Sf4Outlier16Matrix> sf4_qkvrg;
    std::vector<Sf4Outlier16Matrix> sf4_wo;
    std::vector<strata::hip_uma::GpuBuffer> sf4_coda_exps;  // transcoded L20..23 MX8 -> uniform CODE_SF4
    std::vector<strata::hip_uma::GpuBuffer> rel_proj_f16;   // [n_phys] transposed [max_rel, 16] FP16
    std::vector<PldSlotHistory> pld;
    std::vector<float> slot_ewma_acc;                       // per-slot EWMA accepted tokens for #681 adaptive M

    // Wave 8 SF4+Outlier16 dense trunk device buffers
    strata::hip_uma::GpuBuffer d_qkv_cols16;    // [16] int32 (slot 0 = Goodfire d_2027!)
    strata::hip_uma::GpuBuffer d_wo_cols16;     // [16] int32
    strata::hip_uma::GpuBuffer d_sf4_x_f16;     // [3 * 16 * max(D, q_dim)] half_t
    strata::hip_uma::GpuBuffer d_sf4_x_o16;     // [3 * 16 * 16] float
    bool sf4_trunk_ready = false;

    // WS3A IU4 WMMA prefill scratch (grown on demand only when pf_iu4 is active)
    int iu4_scratch_B = 0;
    strata::hip_uma::GpuBuffer d_iu4_x_q4;
    strata::hip_uma::GpuBuffer d_iu4_x_d16;
    strata::hip_uma::GpuBuffer d_iu4_a_q4;
    strata::hip_uma::GpuBuffer d_iu4_a_d16;
    strata::hip_uma::GpuBuffer d_loop_act_scales;

    // WS4+WS5 Tree-16 / Pass-0 Prune / 1-Graph Commit / Stochastic Verify device buffers
    strata::hip_uma::GpuBuffer d_orig_node_map;
    strata::hip_uma::GpuBuffer d_draft_q;
    strata::hip_uma::GpuBuffer d_cum_prior;
    strata::hip_uma::GpuBuffer d_step_params;
    strata::hip_uma::GpuBuffer d_win_path;
    strata::hip_uma::GpuBuffer d_q8_xq;
    strata::hip_uma::GpuBuffer d_q8_xd;
    strata::hip_uma::GpuBuffer d_bspec_rec_ptrs;
    strata::hip_uma::GpuBuffer d_active_eids;

    // WS8 C64 batched spec adaptive acceptance governor
    float c64_ema_emit = 1.6f;
    int   c64_cool = 0;

    // Dedicated captured 1-Graph verify+walk+commit execs (indexed by M in 2..16, greedy vs sampled)
    hipGraphExec_t graph_greedy_exec[17]{};
    int greedy_slot[17]{};
    int greedy_surv[17]{};
    uint64_t greedy_kv_gen[17]{};
    int greedy_scratch_B[17]{};

    hipGraphExec_t graph_sample_exec[17]{};
    int sample_slot[17]{};
    int sample_surv[17]{};
    uint64_t sample_kv_gen[17]{};
    int sample_scratch_B[17]{};

    // Dedicated beam-draft graph using UnembedScreenU4
    hipGraphExec_t graph_beam_u4_exec = nullptr;
    int beam_u4_slot = -1;
    uint64_t beam_u4_gen = 0;

    // Captured 1-Graph tree ctx commit for WinPath16 (up to 16 tokens)
    hipGraphExec_t graph_ctx16_exec[17]{};
    uint64_t ctx16_gen[17]{};
    int ctx16_scratch_B[17]{};

    StrixHaloState() {
        for (int i = 0; i <= 16; ++i) {
            greedy_slot[i] = -1;
            sample_slot[i] = -1;
        }
    }

    void invalidate_graphs() noexcept {
        for (int i = 0; i <= 16; ++i) {
            if (graph_greedy_exec[i]) { (void)hipGraphExecDestroy(graph_greedy_exec[i]); graph_greedy_exec[i] = nullptr; }
            greedy_slot[i] = -1; greedy_surv[i] = 0; greedy_kv_gen[i] = 0; greedy_scratch_B[i] = 0;
            if (graph_sample_exec[i]) { (void)hipGraphExecDestroy(graph_sample_exec[i]); graph_sample_exec[i] = nullptr; }
            sample_slot[i] = -1; sample_surv[i] = 0; sample_kv_gen[i] = 0; sample_scratch_B[i] = 0;
            if (graph_ctx16_exec[i]) { (void)hipGraphExecDestroy(graph_ctx16_exec[i]); graph_ctx16_exec[i] = nullptr; }
            ctx16_gen[i] = 0; ctx16_scratch_B[i] = 0;
        }
        if (graph_beam_u4_exec) {
            (void)hipGraphExecDestroy(graph_beam_u4_exec);
            graph_beam_u4_exec = nullptr;
        }
        beam_u4_slot = -1;
        beam_u4_gen = 0;
    }

    void release() noexcept {
        invalidate_graphs();
        slab_arena.release();
        tdr_slicer.release();
        screen_u4.release();
        npu_probe.release();
        for (auto& q : q8_qkvrg) q.release();
        for (auto& q : q8_wo) q.release();
        for (auto& s : sf4_qkvrg) s.release();
        for (auto& s : sf4_wo) s.release();
        for (auto& b : sf4_coda_exps) strata::hip_uma::free_gpu_buffer(b);
        for (auto& b : rel_proj_f16) strata::hip_uma::free_gpu_buffer(b);
        q8_qkvrg.clear();
        q8_wo.clear();
        sf4_qkvrg.clear();
        sf4_wo.clear();
        sf4_coda_exps.clear();
        rel_proj_f16.clear();
        pld.clear();
        slot_ewma_acc.clear();
        strata::hip_uma::free_gpu_buffer(d_qkv_cols16);
        strata::hip_uma::free_gpu_buffer(d_wo_cols16);
        strata::hip_uma::free_gpu_buffer(d_sf4_x_f16);
        strata::hip_uma::free_gpu_buffer(d_sf4_x_o16);
        sf4_trunk_ready = false;
        strata::hip_uma::free_gpu_buffer(d_iu4_x_q4);
        strata::hip_uma::free_gpu_buffer(d_iu4_x_d16);
        strata::hip_uma::free_gpu_buffer(d_iu4_a_q4);
        strata::hip_uma::free_gpu_buffer(d_iu4_a_d16);
        strata::hip_uma::free_gpu_buffer(d_loop_act_scales);
        strata::hip_uma::free_gpu_buffer(d_orig_node_map);
        strata::hip_uma::free_gpu_buffer(d_draft_q);
        strata::hip_uma::free_gpu_buffer(d_cum_prior);
        strata::hip_uma::free_gpu_buffer(d_step_params);
        strata::hip_uma::free_gpu_buffer(d_win_path);
        strata::hip_uma::free_gpu_buffer(d_q8_xq);
        strata::hip_uma::free_gpu_buffer(d_q8_xd);
        strata::hip_uma::free_gpu_buffer(d_bspec_rec_ptrs);
        strata::hip_uma::free_gpu_buffer(d_active_eids);
        iu4_scratch_B = 0;
        c64_ema_emit = 1.6f;
        c64_cool = 0;
    }

    ~StrixHaloState() { release(); }
};

}  // namespace strata::inkling2::hip
