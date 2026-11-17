#pragma once
#include <cuda_fp16.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>
#include <cmath>
#include <cstdint>

namespace strata::qwen38 {

/// Normalized 8-element Spin(8) Clifford/Hadamard orthogonal transform (H_8^T H_8 = I_8).
/// Preserves inner products: dot(H_8 x, H_8 y) == dot(x, y) while flattening outliers.
__device__ __forceinline__ void spin8_h8_inplace(float v[8]) {
    constexpr float kInvSqrt8 = 0.35355339059327376f; // 1 / sqrt(8)
    // Stage 1: stride 1
    float a0 = v[0] + v[1], a1 = v[0] - v[1];
    float a2 = v[2] + v[3], a3 = v[2] - v[3];
    float a4 = v[4] + v[5], a5 = v[4] - v[5];
    float a6 = v[6] + v[7], a7 = v[6] - v[7];
    // Stage 2: stride 2
    float b0 = a0 + a2, b2 = a0 - a2;
    float b1 = a1 + a3, b3 = a1 - a3;
    float b4 = a4 + a6, b6 = a4 - a6;
    float b5 = a5 + a7, b7 = a5 - a7;
    // Stage 3: stride 4 + normalize
    v[0] = (b0 + b4) * kInvSqrt8;
    v[4] = (b0 - b4) * kInvSqrt8;
    v[1] = (b1 + b5) * kInvSqrt8;
    v[5] = (b1 - b5) * kInvSqrt8;
    v[2] = (b2 + b6) * kInvSqrt8;
    v[6] = (b2 - b6) * kInvSqrt8;
    v[3] = (b3 + b7) * kInvSqrt8;
    v[7] = (b3 - b7) * kInvSqrt8;
}

/// 1. Batch KV write kernel: Partial RoPE (64 dims) + Spin(8) rotation + k8v4 packing.
/// Grid: (n_tokens, n_kv_heads), Block: 32 threads (each thread owns 8 elements of 256-dim head).
/// k_out: [n_kv_heads][ctx][256] int8 + [n_kv_heads][ctx] fp16 scale
/// v_out: [n_kv_heads][ctx][128] packed uint8 (2x int4) + [n_kv_heads][ctx] fp16 scale
__global__ void __launch_bounds__(32)
spin8_kv_write_batch_kernel(
    const float* __restrict__ k_in,
    const float* __restrict__ v_in,
    int8_t* __restrict__ k_q8_out,
    half* __restrict__ k_scale_out,
    uint8_t* __restrict__ v_q4_out,
    half* __restrict__ v_scale_out,
    int pos_start,
    int n_tokens,
    int n_kv_heads,
    int ctx,
    float rope_base
) {
    const int tok = blockIdx.x;
    const int kh  = blockIdx.y;
    const int lane = threadIdx.x; // 0..31 -> 8 elems each = 256
    if (tok >= n_tokens || kh >= n_kv_heads) return;

    const int pos = pos_start + tok;
    const size_t in_off = (static_cast<size_t>(tok) * n_kv_heads + kh) * 256 + lane * 8;

    float kv[8], vv[8];
    #pragma unroll
    for (int i = 0; i < 8; ++i) {
        kv[i] = k_in[in_off + i];
        vv[i] = v_in[in_off + i];
    }

    // Partial RoPE on first 64 dims (lanes 0..7) using pair rotation (0..31 paired with 32..63)
    __shared__ float rope_buf[64];
    if (lane < 8) {
        #pragma unroll
        for (int i = 0; i < 8; ++i) rope_buf[lane * 8 + i] = kv[i];
    }
    __syncthreads();
    if (lane < 8) {
        #pragma unroll
        for (int i = 0; i < 8; ++i) {
            const int d = lane * 8 + i;
            const int p = d % 32;
            const float angle = static_cast<float>(pos) * powf(rope_base, -static_cast<float>(p) / 32.0f);
            float sn, cs;
            sincosf(angle, &sn, &cs);
            kv[i] = (d < 32) ? (rope_buf[d] * cs - rope_buf[d + 32] * sn)
                             : (rope_buf[d - 32] * sn + rope_buf[d] * cs);
        }
    }

    // Spin(8) orthogonal rotation flattens outliers across every 8-dim manifold block
    spin8_h8_inplace(kv);
    spin8_h8_inplace(vv);

    // Warp-reduce max abs for K (int8) and V (int4)
    float kmax = 0.0f, vmax = 0.0f;
    #pragma unroll
    for (int i = 0; i < 8; ++i) {
        kmax = fmaxf(kmax, fabsf(kv[i]));
        vmax = fmaxf(vmax, fabsf(vv[i]));
    }
    #pragma unroll
    for (int mask = 16; mask > 0; mask >>= 1) {
        kmax = fmaxf(kmax, __shfl_xor_sync(0xffffffffu, kmax, mask));
        vmax = fmaxf(vmax, __shfl_xor_sync(0xffffffffu, vmax, mask));
    }

    const float k_sc = (kmax > 0.0f) ? (kmax / 127.0f) : 1.0f;
    const float v_sc = (vmax > 0.0f) ? (vmax / 7.0f)   : 1.0f;
    const float k_inv = 1.0f / k_sc;
    const float v_inv = 1.0f / v_sc;

    const size_t cache_cell = static_cast<size_t>(kh) * ctx + pos;
    if (lane == 0) {
        k_scale_out[cache_cell] = __float2half(k_sc);
        v_scale_out[cache_cell] = __float2half(v_sc);
    }

    const size_t k_out_off = cache_cell * 256 + lane * 8;
    const size_t v_out_off = cache_cell * 128 + lane * 4;
    #pragma unroll
    for (int i = 0; i < 8; ++i) {
        int qk = __float2int_rn(kv[i] * k_inv);
        qk = max(-127, min(127, qk));
        k_q8_out[k_out_off + i] = static_cast<int8_t>(qk);
    }
    #pragma unroll
    for (int i = 0; i < 4; ++i) {
        int q0 = __float2int_rn(vv[i * 2 + 0] * v_inv);
        int q1 = __float2int_rn(vv[i * 2 + 1] * v_inv);
        q0 = max(-7, min(7, q0)) + 8;
        q1 = max(-7, min(7, q1)) + 8;
        v_q4_out[v_out_off + i] = static_cast<uint8_t>((q0 & 0x0F) | ((q1 & 0x0F) << 4));
    }
}

/// 2. Query preconditioning kernel: Partial RoPE + Spin(8) rotation on Q.
/// Allows decode/prefill attention to dot against rotated K without inverse transform.
__global__ void __launch_bounds__(32)
spin8_q_precondition_kernel(
    const float* __restrict__ q_in,
    float* __restrict__ q_out,
    int pos_start,
    int n_tokens,
    int n_q_heads,
    float rope_base
) {
    const int tok  = blockIdx.x;
    const int qh   = blockIdx.y;
    const int lane = threadIdx.x;
    if (tok >= n_tokens || qh >= n_q_heads) return;

    const int pos = pos_start + tok;
    const size_t off = (static_cast<size_t>(tok) * n_q_heads + qh) * 256 + lane * 8;

    float qv[8];
    #pragma unroll
    for (int i = 0; i < 8; ++i) qv[i] = q_in[off + i];

    __shared__ float rope_buf[64];
    if (lane < 8) {
        #pragma unroll
        for (int i = 0; i < 8; ++i) rope_buf[lane * 8 + i] = qv[i];
    }
    __syncthreads();
    if (lane < 8) {
        #pragma unroll
        for (int i = 0; i < 8; ++i) {
            const int d = lane * 8 + i;
            const int p = d % 32;
            const float angle = static_cast<float>(pos) * powf(rope_base, -static_cast<float>(p) / 32.0f);
            float sn, cs;
            sincosf(angle, &sn, &cs);
            qv[i] = (d < 32) ? (rope_buf[d] * cs - rope_buf[d + 32] * sn)
                             : (rope_buf[d - 32] * sn + rope_buf[d] * cs);
        }
    }

    spin8_h8_inplace(qv);
    #pragma unroll
    for (int i = 0; i < 8; ++i) q_out[off + i] = qv[i];
}

/// 3. SM120 FP8 Hadamard-Spin(8) quantization kernel.
/// Rotates rows in 8-element blocks and quantizes to __nv_fp8_e4m3 with per-row scale.
/// Grid: (rows), Block: 256 threads (each thread processes chunks of 8 elements).
__global__ void __launch_bounds__(256)
spin8_fp8_hadamard_quant_kernel(
    const half* __restrict__ src_fp16,
    __nv_fp8_e4m3* __restrict__ dst_fp8,
    float* __restrict__ row_scales,
    int rows,
    int K
) {
    const int row = blockIdx.x;
    if (row >= rows) return;
    const int n_blocks8 = K / 8;
    const half* row_in = src_fp16 + static_cast<size_t>(row) * K;
    __nv_fp8_e4m3* row_out = dst_fp8 + static_cast<size_t>(row) * K;

    // Pass 1: find row-wise max abs after Spin(8) H_8 rotation
    float local_max = 0.0f;
    for (int b = threadIdx.x; b < n_blocks8; b += blockDim.x) {
        float v[8];
        const int col = b * 8;
        #pragma unroll
        for (int i = 0; i < 8; ++i) v[i] = __half2float(row_in[col + i]);
        spin8_h8_inplace(v);
        #pragma unroll
        for (int i = 0; i < 8; ++i) local_max = fmaxf(local_max, fabsf(v[i]));
    }
    __shared__ float warp_max[8];
    for (int m = 16; m > 0; m >>= 1) local_max = fmaxf(local_max, __shfl_xor_sync(0xffffffffu, local_max, m));
    if ((threadIdx.x & 31) == 0) warp_max[threadIdx.x >> 5] = local_max;
    __syncthreads();
    __shared__ float r_scale;
    if (threadIdx.x == 0) {
        float gmax = 0.0f;
        for (int i = 0; i < 8; ++i) gmax = fmaxf(gmax, warp_max[i]);
        // FP8 E4M3 max normal is 448.0f
        r_scale = (gmax > 0.0f) ? (gmax / 448.0f) : 1.0f;
        row_scales[row] = r_scale;
    }
    __syncthreads();

    const float inv_sc = 1.0f / r_scale;
    // Pass 2: rotate & quantize to FP8 E4M3
    for (int b = threadIdx.x; b < n_blocks8; b += blockDim.x) {
        float v[8];
        const int col = b * 8;
        #pragma unroll
        for (int i = 0; i < 8; ++i) v[i] = __half2float(row_in[col + i]);
        spin8_h8_inplace(v);
        #pragma unroll
        for (int i = 0; i < 8; ++i) {
            row_out[col + i] = __nv_fp8_e4m3(v[i] * inv_sc);
        }
    }
}

} // namespace strata::qwen38
