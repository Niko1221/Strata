// engine/src/inkling2/hip/ink_hip_rdna2_gemm.hip.h
// Dense BF16 weight x fp32 activation GEMM for the cards without matrix cores (RDNA2, gfx103x): Y[M][N] = X[M][K] * W[N][K]^T (+ bias), fdot2 on fp16 pairs, fp32 accumulation.
// 64 x 64 output tile per 256-thread block, 4 x 4 per thread, BK = 32, 16-byte LDS reads, global loads of the next k tile in flight during the multiply. The old kernel
// (k_bf16_fdot2_tiled<16, 32>, 2 x 2 per thread, element-wise loads) read every weight once per 16 activation rows and spent one LDS read per fdot2: on a RX 6700 XT the
// attention projections of a 256-token prefill chunk ran at 1.3 TFLOPs.
// The sum of every output is accumulated in the same order as in the old kernel (k pairs 0, 1, 2, ... in sequence, fdot2 with fp32 accumulate), so the results are bit-identical.
// MIT License - Copyright (c) 2026 Niko1221 and the Strata contributors.
#pragma once

#include "ink_hip_common.hip.h"

namespace strata::inkling2::hip {

// requires K % 32 == 0 (the caller checks); any M, N
__global__ void __launch_bounds__(256) k_bf16_fdot2_tiled64(
    const uint16_t* __restrict__ W,
    const float* __restrict__ X,
    const float* __restrict__ bias,
    float* __restrict__ Y,
    int M, int N, int K) {
    constexpr int BK = 32, LDS_STRIDE = 40;   // halves per LDS row (80 B: 16-byte aligned, and 8 consecutive rows hit 8 different bank groups)
    __shared__ alignas(16) half_t sX[64 * LDS_STRIDE];
    __shared__ alignas(16) half_t sW[64 * LDS_STRIDE];
    const int tid = threadIdx.x;
    const int tx = tid & 15, ty = tid >> 4;
    const int m_blk = blockIdx.y * 64, n_blk = blockIdx.x * 64;

    // global -> register staging: X: 2 float4 per thread, W: 1 uint4 (8 bf16) per thread
    const int xr0 = tid >> 3, xc = (tid & 7) * 4;          // rows xr0 and xr0 + 32 of the X tile
    const int wr = tid >> 2, wc = (tid & 3) * 8;
    const bool xv0 = (m_blk + xr0) < M, xv1 = (m_blk + xr0 + 32) < M, wv = (n_blk + wr) < N;
    const float* xp0 = X + static_cast<size_t>(m_blk + xr0) * K + xc;
    const float* xp1 = X + static_cast<size_t>(m_blk + xr0 + 32) * K + xc;
    const uint16_t* wp = W + static_cast<size_t>(n_blk + wr) * K + wc;
    float4 gx0 = make_float4(0, 0, 0, 0), gx1 = gx0;
    uint4 gw = make_uint4(0, 0, 0, 0);
    if (xv0) gx0 = *reinterpret_cast<const float4*>(xp0);
    if (xv1) gx1 = *reinterpret_cast<const float4*>(xp1);
    if (wv) gw = *reinterpret_cast<const uint4*>(wp);

    float acc[4][4];
#pragma unroll
    for (int i = 0; i < 4; ++i)
#pragma unroll
        for (int j = 0; j < 4; ++j) acc[i][j] = 0.0f;

    for (int k0 = 0; k0 < K; k0 += BK) {
        // registers -> LDS (fp32 -> fp16 for X, bf16 -> fp16 for W)
        *reinterpret_cast<uint2*>(&sX[xr0 * LDS_STRIDE + xc]) = make_uint2(h2_pack(gx0.x, gx0.y), h2_pack(gx0.z, gx0.w));
        *reinterpret_cast<uint2*>(&sX[(xr0 + 32) * LDS_STRIDE + xc]) = make_uint2(h2_pack(gx1.x, gx1.y), h2_pack(gx1.z, gx1.w));
        *reinterpret_cast<uint4*>(&sW[wr * LDS_STRIDE + wc]) = make_uint4(
            h2_pack(bf16_to_f32(static_cast<uint16_t>(gw.x & 0xFFFFu)), bf16_to_f32(static_cast<uint16_t>(gw.x >> 16))),
            h2_pack(bf16_to_f32(static_cast<uint16_t>(gw.y & 0xFFFFu)), bf16_to_f32(static_cast<uint16_t>(gw.y >> 16))),
            h2_pack(bf16_to_f32(static_cast<uint16_t>(gw.z & 0xFFFFu)), bf16_to_f32(static_cast<uint16_t>(gw.z >> 16))),
            h2_pack(bf16_to_f32(static_cast<uint16_t>(gw.w & 0xFFFFu)), bf16_to_f32(static_cast<uint16_t>(gw.w >> 16))));
        __syncthreads();
        // the next tile's global loads, in flight while this tile is multiplied
        if (k0 + BK < K) {
            if (xv0) gx0 = *reinterpret_cast<const float4*>(xp0 + k0 + BK);
            if (xv1) gx1 = *reinterpret_cast<const float4*>(xp1 + k0 + BK);
            if (wv) gw = *reinterpret_cast<const uint4*>(wp + k0 + BK);
        }
#pragma unroll
        for (int q = 0; q < BK / 8; ++q) {
            uint4 xa[4], wa[4];
#pragma unroll
            for (int i = 0; i < 4; ++i) xa[i] = *reinterpret_cast<const uint4*>(&sX[(ty + 16 * i) * LDS_STRIDE + q * 8]);
#pragma unroll
            for (int j = 0; j < 4; ++j) wa[j] = *reinterpret_cast<const uint4*>(&sW[(tx + 16 * j) * LDS_STRIDE + q * 8]);
#pragma unroll
            for (int i = 0; i < 4; ++i)
#pragma unroll
                for (int j = 0; j < 4; ++j) {
                    float d = acc[i][j];
                    d = fdot2_f32(xa[i].x, wa[j].x, d);
                    d = fdot2_f32(xa[i].y, wa[j].y, d);
                    d = fdot2_f32(xa[i].z, wa[j].z, d);
                    d = fdot2_f32(xa[i].w, wa[j].w, d);
                    acc[i][j] = d;
                }
        }
        __syncthreads();
    }
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        const int gm = m_blk + ty + 16 * i;
        if (gm < M) {
#pragma unroll
            for (int j = 0; j < 4; ++j) {
                const int gn = n_blk + tx + 16 * j;
                if (gn < N) Y[static_cast<size_t>(gm) * N + gn] = acc[i][j] + (bias ? bias[gn] : 0.0f);
            }
        }
    }
}

}  // namespace strata::inkling2::hip
