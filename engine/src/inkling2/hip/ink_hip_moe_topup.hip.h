// engine/src/inkling2/hip/ink_hip_moe_topup.hip.h
// Grouped-MoE kernels for small prefill chunks (64 <= B <= STRATA_INK_TOPUP_MAXB rows), e.g. the few hundred
// tokens an agent turn appends to a cached conversation.
//
// The B >= 64 kernels k_moe_grouped_{w13_situ,w2}_wmma_sf4_i8_128 (ink_hip_codecs.hip.h) lay 8 waves out as
// 4 row strips x 2 column halves. With 64..256 rows an expert gets ~4..16 rows, so only the 2 waves of strip 0
// compute; the other 6 only load. These variants use BM = 32, BN = 128 and split the columns over all 8 waves.
// LDS layout, fragment loads, accumulation order and epilogues are unchanged, so every stored value is
// bit-identical to the BM = 128 kernels. Grid z (row tiles of 32) fans hot experts out like m_tiles.
// Switch: STRATA_INK_TOPUP_MOE=1 (default off), STRATA_INK_TOPUP_MAXB (default 256); SF4 int8 layers only.
// MIT License - Copyright (c) 2026 Niko1221 and the Strata contributors.
#pragma once

#include "ink_hip_ops.hip.h"

namespace strata::inkling2::hip {

constexpr int kTopupBM = 32;

// w13 + SITU: X_q8 [B, D] int8 per-16 quantised activations with X_d16 [B, D / 16] scales -> bucket_act [B * top_k, inter]
__global__ void __launch_bounds__(256) k_moe_grouped_w13_situ_wmma_sf4_i8_n8(
    const GpuExpertDesc* __restrict__ exps,
    const int32_t* __restrict__ exp_off,
    const int32_t* __restrict__ bucket_tok,
    const int8_t* __restrict__ X_q8,
    const float* __restrict__ X_d16,
    half_t* __restrict__ bucket_act,
    int D, int inter, float beta, float lb) {
#if INK_HAS_WMMA
    const int eid = blockIdx.y;
    const int beg = exp_off[eid];
    const int end = exp_off[eid + 1];
    const int m_e = end - beg;
    constexpr int BM = kTopupBM, BN = 128, BK = 64, LDI = BK + 16;
    if (static_cast<int>(blockIdx.z) * BM >= m_e) return;
    const GpuExpertDesc desc = exps[eid];
    if (!desc.body) return;

    __shared__ alignas(16) int8_t sXq[BM * LDI];   //  2,560 B
    __shared__ alignas(16) int8_t sWq[BN * LDI];   // 10,240 B
    __shared__ float sXd[BM * 4];                  //    512 B
    __shared__ float sWs[BN * 4];                  //  2,048 B
    __shared__ int32_t sTok[BM];                   //    128 B -> 15.1 KB

    const int tid = threadIdx.x;
    const int wave = tid >> 5;   // 0..7: columns wave * 16 .. wave * 16 + 15 of the 128-column tile
    const int lane = tid & 31;
    const int l16 = lane & 15;
    const int hi = lane >> 4;
    (void)hi;
    const int N = 2 * inter;
    const int n_blk = blockIdx.x * BN;
    const float half_outer = 0.5f * (reinterpret_cast<const float*>(desc.body)[0] / reinterpret_cast<const float*>(desc.body)[1]);
    const uint8_t* sf4_p = desc.body + 16;
    const uint8_t* sf4_s = sf4_p + static_cast<size_t>(N) * (D >> 1);
    const int nblk16 = D >> 4;
    const int t_stride = gridDim.z * BM;

    for (int t0 = blockIdx.z * BM; t0 < m_e; t0 += t_stride) {
        const int tm = (m_e - t0 < BM) ? (m_e - t0) : BM;
        if (tid < BM) sTok[tid] = (tid < tm) ? bucket_tok[beg + t0 + tid] : 0;
        sync_lds();

        fv8 acc[2];
        acc[0] = fv8_zero();
        acc[1] = fv8_zero();

        for (int k0 = 0; k0 < D; k0 += BK) {
            const int kb16 = k0 >> 4;
            if (tid < BM * 4) {
                const int r = tid >> 2, b4 = tid & 3;
                if (r < tm) {
                    const int tok = sTok[r];
                    *reinterpret_cast<int4*>(&sXq[r * LDI + b4 * 16]) =
                        *reinterpret_cast<const int4*>(X_q8 + static_cast<size_t>(tok) * D + k0 + b4 * 16);
                    sXd[r * 4 + b4] = X_d16[static_cast<size_t>(tok) * nblk16 + kb16 + b4];
                } else {
                    *reinterpret_cast<int4*>(&sXq[r * LDI + b4 * 16]) = make_int4(0, 0, 0, 0);
                    sXd[r * 4 + b4] = 0.0f;
                }
            }
#pragma unroll
            for (int iter = 0; iter < 2; ++iter) {
                const int idx = iter * 256 + tid;
                const int r = idx >> 2, b4 = idx & 3;
                const int gn = n_blk + r;
                const int gb = kb16 + b4;
                if (gn < N) {
                    sWs[r * 4 + b4] = e4m3fn_to_f32(sf4_s[static_cast<size_t>(gn) * nblk16 + gb]) * half_outer;
                    const uint2 raw = *reinterpret_cast<const uint2*>(sf4_p + static_cast<size_t>(gn) * (D >> 1) + gb * 8);
                    int4 p4;
                    p4.x = pack4_i8(sf4_e2m1_i8_x2(raw.x & 15u), sf4_e2m1_i8_x2((raw.x >> 4) & 15u),
                                    sf4_e2m1_i8_x2((raw.x >> 8) & 15u), sf4_e2m1_i8_x2((raw.x >> 12) & 15u));
                    p4.y = pack4_i8(sf4_e2m1_i8_x2((raw.x >> 16) & 15u), sf4_e2m1_i8_x2((raw.x >> 20) & 15u),
                                    sf4_e2m1_i8_x2((raw.x >> 24) & 15u), sf4_e2m1_i8_x2((raw.x >> 28) & 15u));
                    p4.z = pack4_i8(sf4_e2m1_i8_x2(raw.y & 15u), sf4_e2m1_i8_x2((raw.y >> 4) & 15u),
                                    sf4_e2m1_i8_x2((raw.y >> 8) & 15u), sf4_e2m1_i8_x2((raw.y >> 12) & 15u));
                    p4.w = pack4_i8(sf4_e2m1_i8_x2((raw.y >> 16) & 15u), sf4_e2m1_i8_x2((raw.y >> 20) & 15u),
                                    sf4_e2m1_i8_x2((raw.y >> 24) & 15u), sf4_e2m1_i8_x2((raw.y >> 28) & 15u));
                    *reinterpret_cast<int4*>(&sWq[r * LDI + b4 * 16]) = p4;
                } else {
                    *reinterpret_cast<int4*>(&sWq[r * LDI + b4 * 16]) = make_int4(0, 0, 0, 0);
                    sWs[r * 4 + b4] = 0.0f;
                }
            }
            sync_lds();

            {
                const bool do_m1 = (16 < tm);
                const int col16 = wmma_acc_n(0, lane);
#pragma unroll
                for (int kk = 0; kk < 4; ++kk) {
#if INK_GFX12
                    iv2 b0 = wmma_ld_i8_k16(&sWq[(wave * 16 + l16) * LDI + kk * 16], hi);
                    iv2 a0 = wmma_ld_i8_k16(&sXq[l16 * LDI + kk * 16], hi);
                    iv8 i00 = wmma_i8_k16(a0, b0, iv8_zero());
#elif INK_GFX11
                    iv4 b0 = wmma_ld_i8_k16_g11(&sWq[(wave * 16 + l16) * LDI + kk * 16]);
                    iv4 a0 = wmma_ld_i8_k16_g11(&sXq[l16 * LDI + kk * 16]);
                    iv8 i00 = wmma_i8_k16_g11(a0, b0, iv8_zero());
#endif
                    const float ws0 = sWs[(wave * 16 + col16) * 4 + kk];
#pragma unroll
                    for (int r = 0; r < 8; ++r) {
                        const float xd0 = sXd[wmma_acc_m(r, lane) * 4 + kk];
                        acc[0][r] = fmaf(static_cast<float>(i00[r]), xd0 * ws0, acc[0][r]);
                    }
                    if (do_m1) {
#if INK_GFX12
                        iv2 a1 = wmma_ld_i8_k16(&sXq[(16 + l16) * LDI + kk * 16], hi);
                        iv8 i10 = wmma_i8_k16(a1, b0, iv8_zero());
#elif INK_GFX11
                        iv4 a1 = wmma_ld_i8_k16_g11(&sXq[(16 + l16) * LDI + kk * 16]);
                        iv8 i10 = wmma_i8_k16_g11(a1, b0, iv8_zero());
#endif
#pragma unroll
                        for (int r = 0; r < 8; ++r) {
                            const float xd1 = sXd[(16 + wmma_acc_m(r, lane)) * 4 + kk];
                            acc[1][r] = fmaf(static_cast<float>(i10[r]), xd1 * ws0, acc[1][r]);
                        }
                    }
                }
            }
            sync_lds();
        }

        {
            const bool is_even = ((lane & 1) == 0);
            const int gn = n_blk + wave * 16 + wmma_acc_n(0, lane);
            const int gj = gn >> 1;
#pragma unroll
            for (int mi = 0; mi < 2; ++mi) {
                if (mi * 16 >= tm) continue;
#pragma unroll
                for (int r = 0; r < 8; ++r) {
                    const float g_val = acc[mi][r];
                    const float u_val = __shfl_xor(g_val, 1, 32);
                    const int lm = mi * 16 + wmma_acc_m(r, lane);
                    if (is_even && lm < tm && gj < inter) {
                        bucket_act[static_cast<size_t>(beg + t0 + lm) * inter + gj] =
                            static_cast<half_t>(situ_f32(g_val, u_val, beta, lb));
                    }
                }
            }
        }
        sync_lds();   // sTok is rewritten by the next tile of this block
    }
#else
    (void)exps; (void)exp_off; (void)bucket_tok; (void)X_q8; (void)X_d16; (void)bucket_act; (void)D; (void)inter; (void)beta; (void)lb;
#endif
}

// w2: act_q8 [B * top_k, inter] int8 per-16 with act_d16 [B * top_k, inter / 16] -> bucket_y [B * top_k, D] f32
__global__ void __launch_bounds__(256) k_moe_grouped_w2_wmma_sf4_i8_n8(
    const GpuExpertDesc* __restrict__ exps,
    const int32_t* __restrict__ exp_off,
    const int8_t* __restrict__ act_q8,
    const float* __restrict__ act_d16,
    float* __restrict__ bucket_y,
    int D, int inter) {
#if INK_HAS_WMMA
    const int eid = blockIdx.y;
    const int beg = exp_off[eid];
    const int end = exp_off[eid + 1];
    const int m_e = end - beg;
    constexpr int BM = kTopupBM, BN = 128, BK = 64, LDI = BK + 16;
    if (static_cast<int>(blockIdx.z) * BM >= m_e) return;
    const GpuExpertDesc desc = exps[eid];
    if (!desc.body) return;

    __shared__ alignas(16) int8_t sXq[BM * LDI];
    __shared__ alignas(16) int8_t sWq[BN * LDI];
    __shared__ float sXd[BM * 4];
    __shared__ float sWs[BN * 4];

    const int tid = threadIdx.x;
    const int wave = tid >> 5;
    const int lane = tid & 31;
    const int l16 = lane & 15;
    const int hi = lane >> 4;
    (void)hi;
    const int n_blk = blockIdx.x * BN;
    const float half_outer = 0.5f * (reinterpret_cast<const float*>(desc.body)[2] / reinterpret_cast<const float*>(desc.body)[3]);
    const uint8_t* sf4_p = desc.body + desc.dn_offset;
    const uint8_t* sf4_s = sf4_p + static_cast<size_t>(D) * (inter >> 1);
    const int nblk16 = inter >> 4;
    const int t_stride = gridDim.z * BM;

    for (int t0 = blockIdx.z * BM; t0 < m_e; t0 += t_stride) {
        const int tm = (m_e - t0 < BM) ? (m_e - t0) : BM;
        fv8 acc[2];
        acc[0] = fv8_zero();
        acc[1] = fv8_zero();

        for (int k0 = 0; k0 < inter; k0 += BK) {
            const int kb16 = k0 >> 4;
            if (tid < BM * 4) {
                const int r = tid >> 2, b4 = tid & 3;
                if (r < tm) {
                    const size_t row_i = static_cast<size_t>(beg + t0 + r);
                    *reinterpret_cast<int4*>(&sXq[r * LDI + b4 * 16]) =
                        *reinterpret_cast<const int4*>(act_q8 + row_i * inter + k0 + b4 * 16);
                    sXd[r * 4 + b4] = act_d16[row_i * nblk16 + kb16 + b4];
                } else {
                    *reinterpret_cast<int4*>(&sXq[r * LDI + b4 * 16]) = make_int4(0, 0, 0, 0);
                    sXd[r * 4 + b4] = 0.0f;
                }
            }
#pragma unroll
            for (int iter = 0; iter < 2; ++iter) {
                const int idx = iter * 256 + tid;
                const int r = idx >> 2, b4 = idx & 3;
                const int gn = n_blk + r;
                const int gb = kb16 + b4;
                if (gn < D) {
                    sWs[r * 4 + b4] = e4m3fn_to_f32(sf4_s[static_cast<size_t>(gn) * nblk16 + gb]) * half_outer;
                    const uint2 raw = *reinterpret_cast<const uint2*>(sf4_p + static_cast<size_t>(gn) * (inter >> 1) + gb * 8);
                    int4 p4;
                    p4.x = pack4_i8(sf4_e2m1_i8_x2(raw.x & 15u), sf4_e2m1_i8_x2((raw.x >> 4) & 15u),
                                    sf4_e2m1_i8_x2((raw.x >> 8) & 15u), sf4_e2m1_i8_x2((raw.x >> 12) & 15u));
                    p4.y = pack4_i8(sf4_e2m1_i8_x2((raw.x >> 16) & 15u), sf4_e2m1_i8_x2((raw.x >> 20) & 15u),
                                    sf4_e2m1_i8_x2((raw.x >> 24) & 15u), sf4_e2m1_i8_x2((raw.x >> 28) & 15u));
                    p4.z = pack4_i8(sf4_e2m1_i8_x2(raw.y & 15u), sf4_e2m1_i8_x2((raw.y >> 4) & 15u),
                                    sf4_e2m1_i8_x2((raw.y >> 8) & 15u), sf4_e2m1_i8_x2((raw.y >> 12) & 15u));
                    p4.w = pack4_i8(sf4_e2m1_i8_x2((raw.y >> 16) & 15u), sf4_e2m1_i8_x2((raw.y >> 20) & 15u),
                                    sf4_e2m1_i8_x2((raw.y >> 24) & 15u), sf4_e2m1_i8_x2((raw.y >> 28) & 15u));
                    *reinterpret_cast<int4*>(&sWq[r * LDI + b4 * 16]) = p4;
                } else {
                    *reinterpret_cast<int4*>(&sWq[r * LDI + b4 * 16]) = make_int4(0, 0, 0, 0);
                    sWs[r * 4 + b4] = 0.0f;
                }
            }
            sync_lds();

            {
                const bool do_m1 = (16 < tm);
                const int col16 = wmma_acc_n(0, lane);
#pragma unroll
                for (int kk = 0; kk < 4; ++kk) {
#if INK_GFX12
                    iv2 b0 = wmma_ld_i8_k16(&sWq[(wave * 16 + l16) * LDI + kk * 16], hi);
                    iv2 a0 = wmma_ld_i8_k16(&sXq[l16 * LDI + kk * 16], hi);
                    iv8 i00 = wmma_i8_k16(a0, b0, iv8_zero());
#elif INK_GFX11
                    iv4 b0 = wmma_ld_i8_k16_g11(&sWq[(wave * 16 + l16) * LDI + kk * 16]);
                    iv4 a0 = wmma_ld_i8_k16_g11(&sXq[l16 * LDI + kk * 16]);
                    iv8 i00 = wmma_i8_k16_g11(a0, b0, iv8_zero());
#endif
                    const float ws0 = sWs[(wave * 16 + col16) * 4 + kk];
#pragma unroll
                    for (int r = 0; r < 8; ++r) {
                        const float xd0 = sXd[wmma_acc_m(r, lane) * 4 + kk];
                        acc[0][r] = fmaf(static_cast<float>(i00[r]), xd0 * ws0, acc[0][r]);
                    }
                    if (do_m1) {
#if INK_GFX12
                        iv2 a1 = wmma_ld_i8_k16(&sXq[(16 + l16) * LDI + kk * 16], hi);
                        iv8 i10 = wmma_i8_k16(a1, b0, iv8_zero());
#elif INK_GFX11
                        iv4 a1 = wmma_ld_i8_k16_g11(&sXq[(16 + l16) * LDI + kk * 16]);
                        iv8 i10 = wmma_i8_k16_g11(a1, b0, iv8_zero());
#endif
#pragma unroll
                        for (int r = 0; r < 8; ++r) {
                            const float xd1 = sXd[(16 + wmma_acc_m(r, lane)) * 4 + kk];
                            acc[1][r] = fmaf(static_cast<float>(i10[r]), xd1 * ws0, acc[1][r]);
                        }
                    }
                }
            }
            sync_lds();
        }

        {
            const int gn = n_blk + wave * 16 + wmma_acc_n(0, lane);
            if (gn < D) {
#pragma unroll
                for (int mi = 0; mi < 2; ++mi) {
                    if (mi * 16 >= tm) continue;
#pragma unroll
                    for (int r = 0; r < 8; ++r) {
                        const int lm = mi * 16 + wmma_acc_m(r, lane);
                        if (lm < tm) bucket_y[static_cast<size_t>(beg + t0 + lm) * D + gn] = acc[mi][r];
                    }
                }
            }
        }
    }
#else
    (void)exps; (void)exp_off; (void)act_q8; (void)act_d16; (void)bucket_y; (void)D; (void)inter;
#endif
}

// Grid z for the n8 kernels: row tiles of 32 per expert, hot experts fan out over up to 8 blocks (as m_tiles_128).
inline int moe_topup_ztiles(int B) {
    const int z = (B + kTopupBM - 1) / kTopupBM;
    return z < 1 ? 1 : (z > 8 ? 8 : z);
}

}  // namespace strata::inkling2::hip
