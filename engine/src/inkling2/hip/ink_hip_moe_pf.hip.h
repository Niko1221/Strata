// engine/src/inkling2/hip/ink_hip_moe_pf.hip.h
// Grouped-MoE prefill kernels v2 for the SF4 (NVFP4) int8 path of Inkling 2 Mini on RDNA4, bit-identical to
// k_moe_grouped_{w13_situ,w2}_wmma_sf4_i8_128 (ink_hip_codecs.hip.h).
//
// Same arithmetic per stored element: the int8 activation codes and the SF4 -> doubled-int8 weight codes are the same
// values, the int32 WMMA dot over each 16-wide k block is exact, and each output element accumulates
//     acc = fmaf(float(dot16), xd * ws, acc)     (k0 outer, kk inner, from 0.0f)
// in the same order; the SITU epilogue (__shfl_xor partner = the up column) and the f16 / f32 stores are the same
// expressions. What changes is only how the work is laid out and fed:
//   - BM in {32, 64} rows x BN = 128 columns per block, all 8 waves busy: WR = BM / 32 row groups of 32 rows
//     x WC = 8 / WR column groups of 128 / WC columns (at 512-2048-row chunks an expert holds 32-128 rows, so one
//     decoded weight tile serves all of them);
//   - double-buffered LDS with register prefetch: the next k step's activation / weight bytes are loaded into registers
//     before the current step's WMMA and written to the other LDS buffer after it, one barrier per k step instead of two;
//   - the per-16 activation scales are stored transposed (sXdT[kk][row]), so a lane reads the 8 scales of its 8
//     accumulator rows with two ds_read_b128 instead of eight ds_read_b32;
//   - SF4 nibbles -> doubled int8 through v_perm_b32 byte selects into two 8-entry half tables (sf4_codes8, below)
//     instead of a 16-entry LUT per nibble; same int8 values {0, +-1, +-2, +-3, +-4, +-6, +-8, +-12} (nibble 8 = -0 -> 0).
// Switch (backend): STRATA_INK_PF_MOE = 64 (default on RDNA4) | 32 | 0 = off (the _128 kernels).
// Unit test: ink_hip_moe_pf_test.hip.cpp (bitwise vs the _128 kernels, B 64..2048, 4 routings, + timing).
// MIT License - Copyright (c) 2026 Niko1221 and the Strata contributors.
#pragma once

#include "ink_hip_codecs.hip.h"

namespace strata::inkling2::hip {

// ---- SF4 nibble word -> doubled-int8 codes with V_PERM_B32 byte selects ----
// The codebook is 2 x the e2m1 value: index 0..7 = {0, 1, 2, 3, 4, 6, 8, 12}, 8..15 = the same negated (8 = -0 -> 0),
// i.e. sf4_e2m1_i8_x2 (ink_hip_common.hip.h). V_PERM_B32 picks each result byte from the 8-byte pool {S0:S1} by a
// selector byte (0..3 = S1 bytes, 4..7 = S0 bytes), so one perm looks up 4 bytes in an 8-entry table. Each half of
// the codebook is one 8-byte pool; a third perm takes byte i from the positive or the negative lookup by bit 3.
namespace sf4dec {
constexpr int8_t kCode[16] = {0, 1, 2, 3, 4, 6, 8, 12, 0, -1, -2, -3, -4, -6, -8, -12};
// table word: codebook entries first..first+3 as bytes 0..3
constexpr uint32_t tword(int first) {
    uint32_t v = 0;
    for (int i = 0; i < 4; ++i) v |= static_cast<uint32_t>(static_cast<uint8_t>(kCode[first + i])) << (8 * i);
    return v;
}
constexpr uint32_t kPosLo = tword(0), kPosHi = tword(4), kNegLo = tword(8), kNegHi = tword(12);
constexpr uint32_t kIdx = 0x03020100u;   // selector "byte i of S1" for i = 0..3
}  // namespace sf4dec

#if defined(__HIP_DEVICE_COMPILE__) && __has_builtin(__builtin_amdgcn_perm)
// q: one nibble in bits 0-3 of each byte (bits 4-7 are ignored) -> the 4 codes, byte for byte
__device__ __forceinline__ uint32_t sf4_codes4(uint32_t q) {
    const uint32_t mag = q & 0x07070707u;                                   // 3-bit index into a half table
    const uint32_t pos = __builtin_amdgcn_perm(sf4dec::kPosHi, sf4dec::kPosLo, mag);
    const uint32_t neg = __builtin_amdgcn_perm(sf4dec::kNegHi, sf4dec::kNegLo, mag);
    const uint32_t pick = ((q >> 1) & 0x04040404u) | sf4dec::kIdx;          // bit 3 set -> byte i of neg (S0)
    return __builtin_amdgcn_perm(neg, pos, pick);
}
// w: 8 nibbles (nibble j = k position j) -> lo = codes 0..3, hi = codes 4..7 (byte b = code b)
__device__ __forceinline__ void sf4_codes8(uint32_t w, int32_t& lo, int32_t& hi) {
    const uint32_t ev = sf4_codes4(w);        // nibbles 0, 2, 4, 6 in bytes 0..3
    const uint32_t od = sf4_codes4(w >> 4);   // nibbles 1, 3, 5, 7 in bytes 0..3
    // k order: ev0 od0 ev1 od1 | ev2 od2 ev3 od3 (ev = S1 bytes 0..3, od = S0 bytes 4..7)
    lo = static_cast<int32_t>(__builtin_amdgcn_perm(od, ev, 0x05010400u));
    hi = static_cast<int32_t>(__builtin_amdgcn_perm(od, ev, 0x07030602u));
}
#else
__device__ __forceinline__ void sf4_codes8(uint32_t w, int32_t& lo, int32_t& hi) {
    int c[8];
#pragma unroll
    for (int j = 0; j < 8; ++j) c[j] = sf4dec::kCode[(w >> (4 * j)) & 15u];
    lo = pack4_i8(c[0], c[1], c[2], c[3]);
    hi = pack4_i8(c[4], c[5], c[6], c[7]);
}
#endif
// ---- end of the SF4 decode ----

// W13 = true: X = X_q8 [B, D] int8 per-16 (X_d16 [B, D / 16]) gathered through bucket_tok -> SITU -> out_act [B * top_k, inter] f16
// W13 = false: X = act_q8 [B * top_k, inter] int8 per-16 (act_d16 [B * top_k, inter / 16]) -> out_y [B * top_k, D] f32
// Grid: (ceil(N / 128), n_experts, z row tiles of BM), 256 threads. Needs K % 64 == 0 (checked by the dispatch).
template <int BM, bool W13>
__global__ void __launch_bounds__(256) k_moe_pf_sf4_i8(
    const GpuExpertDesc* __restrict__ exps,
    const int32_t* __restrict__ exp_off,
    const int32_t* __restrict__ bucket_tok,
    const int8_t* __restrict__ Xq,
    const float* __restrict__ Xd,
    half_t* __restrict__ out_act,
    float* __restrict__ out_y,
    int D, int inter, float beta, float lb) {
#if INK_HAS_WMMA && INK_GFX12
    static_assert(BM == 32 || BM == 64, "BM");
    constexpr int BN = 128, BK = 64, LDI = BK + 16;
    constexpr int WR = BM / 32;        // row groups of 32 rows
    constexpr int WC = 8 / WR;         // column groups
    constexpr int CS = 8 / WC;         // 16-column strips per wave (BN / 16 / WC)
    constexpr int NX = (BM * 4 + 255) / 256;
    const int eid = blockIdx.y;
    const int beg = exp_off[eid];
    const int end = exp_off[eid + 1];
    const int m_e = end - beg;
    if (static_cast<int>(blockIdx.z) * BM >= m_e) return;
    const GpuExpertDesc desc = exps[eid];
    if (!desc.body) return;

    __shared__ alignas(16) int8_t sXq[2][BM * LDI];
    __shared__ alignas(16) int8_t sWq[2][BN * LDI];
    __shared__ alignas(16) float sXdT[2][4 * BM];
    __shared__ float sWs[2][BN * 4];
    __shared__ int32_t sTok[BM];

    const int tid = threadIdx.x;
    const int wave = tid >> 5;
    const int lane = tid & 31;
    const int l16 = lane & 15;
    const int hi = lane >> 4;
    const int wr = wave / WC;
    const int wc = wave % WC;
    const int K = W13 ? D : inter;
    const int N = W13 ? 2 * inter : D;
    const int n_blk = blockIdx.x * BN;
    const float half_outer = W13
        ? 0.5f * (reinterpret_cast<const float*>(desc.body)[0] / reinterpret_cast<const float*>(desc.body)[1])
        : 0.5f * (reinterpret_cast<const float*>(desc.body)[2] / reinterpret_cast<const float*>(desc.body)[3]);
    const uint8_t* sf4_p = W13 ? (desc.body + 16) : (desc.body + desc.dn_offset);
    const uint8_t* sf4_s = sf4_p + static_cast<size_t>(N) * (K >> 1);
    const int nblk16 = K >> 4;
    const int t_stride = gridDim.z * BM;
    const int nk = K / BK;
    const int col16 = wmma_acc_n(0, lane);

    for (int t0 = blockIdx.z * BM; t0 < m_e; t0 += t_stride) {
        const int tm = (m_e - t0 < BM) ? (m_e - t0) : BM;
        if (W13) {
            for (int i = tid; i < BM; i += 256) sTok[i] = (i < tm) ? bucket_tok[beg + t0 + i] : 0;
            sync_lds();
        }

        int4 xr[NX];
        float xdr[NX];
        uint2 wraw[2];
        uint32_t wsb[2];
        auto gload = [&](int k0) __attribute__((always_inline)) {
            const int kb16 = k0 >> 4;
#pragma unroll
            for (int j = 0; j < NX; ++j) {
                const int idx = j * 256 + tid;
                const int r = idx >> 2, b4 = idx & 3;
                if (idx < BM * 4 && r < tm) {
                    const size_t row = W13 ? static_cast<size_t>(sTok[r]) : static_cast<size_t>(beg + t0 + r);
                    xr[j] = *reinterpret_cast<const int4*>(Xq + row * K + k0 + b4 * 16);
                    xdr[j] = Xd[row * nblk16 + kb16 + b4];
                } else {
                    xr[j] = make_int4(0, 0, 0, 0);
                    xdr[j] = 0.0f;
                }
            }
#pragma unroll
            for (int it = 0; it < 2; ++it) {
                const int idx = it * 256 + tid;
                const int r = idx >> 2, b4 = idx & 3;
                const int gn = n_blk + r;
                const int gb = kb16 + b4;
                if (gn < N) {
                    wsb[it] = sf4_s[static_cast<size_t>(gn) * nblk16 + gb];
                    wraw[it] = *reinterpret_cast<const uint2*>(sf4_p + static_cast<size_t>(gn) * (K >> 1) + gb * 8);
                } else {
                    wsb[it] = 0u;
                    wraw[it] = make_uint2(0u, 0u);
                }
            }
        };
        auto lstore = [&](int buf) __attribute__((always_inline)) {
#pragma unroll
            for (int j = 0; j < NX; ++j) {
                const int idx = j * 256 + tid;
                if (idx < BM * 4) {
                    const int r = idx >> 2, b4 = idx & 3;
                    *reinterpret_cast<int4*>(&sXq[buf][r * LDI + b4 * 16]) = xr[j];
                    sXdT[buf][b4 * BM + r] = xdr[j];
                }
            }
#pragma unroll
            for (int it = 0; it < 2; ++it) {
                const int idx = it * 256 + tid;
                const int r = idx >> 2, b4 = idx & 3;
                const int gn = n_blk + r;
                if (gn < N) {
                    sWs[buf][r * 4 + b4] = e4m3fn_to_f32(static_cast<uint8_t>(wsb[it])) * half_outer;
                    int4 p4;
                    sf4_codes8(wraw[it].x, p4.x, p4.y);
                    sf4_codes8(wraw[it].y, p4.z, p4.w);
                    *reinterpret_cast<int4*>(&sWq[buf][r * LDI + b4 * 16]) = p4;
                } else {
                    *reinterpret_cast<int4*>(&sWq[buf][r * LDI + b4 * 16]) = make_int4(0, 0, 0, 0);
                    sWs[buf][r * 4 + b4] = 0.0f;
                }
            }
        };

        fv8 acc[2][CS];
#pragma unroll
        for (int mi = 0; mi < 2; ++mi)
#pragma unroll
            for (int ni = 0; ni < CS; ++ni) acc[mi][ni] = fv8_zero();

        const int row0 = wr * 32;
        const bool wave_busy = row0 < tm;          // wave-uniform
        const bool do_m1 = (row0 + 16 < tm);       // wave-uniform
        gload(0);
        lstore(0);
        sync_lds();
        for (int ks = 0; ks < nk; ++ks) {
            const int buf = ks & 1;
            if (ks + 1 < nk) gload((ks + 1) * BK);
            if (wave_busy) {
#pragma unroll
                for (int kk = 0; kk < 4; ++kk) {
                    iv2 bw[CS];
                    float ws[CS];
#pragma unroll
                    for (int ni = 0; ni < CS; ++ni) {
                        bw[ni] = wmma_ld_i8_k16(&sWq[buf][(wc * 16 * CS + ni * 16 + l16) * LDI + kk * 16], hi);
                        ws[ni] = sWs[buf][(wc * 16 * CS + ni * 16 + col16) * 4 + kk];
                    }
                    {
                        const iv2 a0 = wmma_ld_i8_k16(&sXq[buf][(row0 + l16) * LDI + kk * 16], hi);
                        const float4 xa = *reinterpret_cast<const float4*>(&sXdT[buf][kk * BM + row0 + 8 * hi]);
                        const float4 xb = *reinterpret_cast<const float4*>(&sXdT[buf][kk * BM + row0 + 8 * hi + 4]);
                        const float xd[8] = {xa.x, xa.y, xa.z, xa.w, xb.x, xb.y, xb.z, xb.w};
#pragma unroll
                        for (int ni = 0; ni < CS; ++ni) {
                            const iv8 i0 = wmma_i8_k16(a0, bw[ni], iv8_zero());
#pragma unroll
                            for (int r = 0; r < 8; ++r)
                                acc[0][ni][r] = fmaf(static_cast<float>(i0[r]), xd[r] * ws[ni], acc[0][ni][r]);
                        }
                    }
                    if (do_m1) {
                        const iv2 a1 = wmma_ld_i8_k16(&sXq[buf][(row0 + 16 + l16) * LDI + kk * 16], hi);
                        const float4 xa = *reinterpret_cast<const float4*>(&sXdT[buf][kk * BM + row0 + 16 + 8 * hi]);
                        const float4 xb = *reinterpret_cast<const float4*>(&sXdT[buf][kk * BM + row0 + 16 + 8 * hi + 4]);
                        const float xd[8] = {xa.x, xa.y, xa.z, xa.w, xb.x, xb.y, xb.z, xb.w};
#pragma unroll
                        for (int ni = 0; ni < CS; ++ni) {
                            const iv8 i1 = wmma_i8_k16(a1, bw[ni], iv8_zero());
#pragma unroll
                            for (int r = 0; r < 8; ++r)
                                acc[1][ni][r] = fmaf(static_cast<float>(i1[r]), xd[r] * ws[ni], acc[1][ni][r]);
                        }
                    }
                }
            }
            if (ks + 1 < nk) lstore(buf ^ 1);
            sync_lds();
        }

        if (wave_busy) {
#pragma unroll
            for (int mi = 0; mi < 2; ++mi) {
                if (row0 + mi * 16 >= tm) continue;
#pragma unroll
                for (int ni = 0; ni < CS; ++ni) {
                    const int gn = n_blk + wc * 16 * CS + ni * 16 + col16;
                    if constexpr (W13) {
                        const bool is_even = ((lane & 1) == 0);
                        const int gj = gn >> 1;
#pragma unroll
                        for (int r = 0; r < 8; ++r) {
                            const float g_val = acc[mi][ni][r];
                            const float u_val = __shfl_xor(g_val, 1, 32);
                            const int lm = row0 + mi * 16 + wmma_acc_m(r, lane);
                            if (is_even && lm < tm && gj < inter) {
                                out_act[static_cast<size_t>(beg + t0 + lm) * inter + gj] =
                                    static_cast<half_t>(situ_f32(g_val, u_val, beta, lb));
                            }
                        }
                    } else {
                        if (gn < D) {
#pragma unroll
                            for (int r = 0; r < 8; ++r) {
                                const int lm = row0 + mi * 16 + wmma_acc_m(r, lane);
                                if (lm < tm) out_y[static_cast<size_t>(beg + t0 + lm) * D + gn] = acc[mi][ni][r];
                            }
                        }
                    }
                }
            }
        }
        if (W13) sync_lds();   // sTok is rewritten by the next row tile of this block
    }
#else
    (void)exps; (void)exp_off; (void)bucket_tok; (void)Xq; (void)Xd; (void)out_act; (void)out_y; (void)D; (void)inter;
    (void)beta; (void)lb;
#endif
}

// Grid z of the v2 kernels: row tiles of BM per expert, hot experts fan out over up to 8 blocks (as the _128 grid).
inline int moe_pf_ztiles(int B, int bm) {
    const int z = (B + bm - 1) / bm;
    return z < 1 ? 1 : (z > 8 ? 8 : z);
}

// Launch w13 (+ SITU) or w2 of one layer with the v2 kernels, BM = bm (32 or 64); same operands as the _128 launches.
inline void launch_moe_pf_sf4_i8(int bm, int B, int n_experts, int D, int inter, float beta, float lb,
                                 const GpuExpertDesc* exps, const int32_t* off, const int32_t* btk,
                                 const int8_t* xq, const float* xd, half_t* act, bool w13,
                                 const int8_t* aq, const float* ad, float* by, hipStream_t st) {
    const int zt = moe_pf_ztiles(B, bm);
    if (w13) {
        const dim3 g((2 * inter + 127) / 128, n_experts, zt);
        if (bm == 32) k_moe_pf_sf4_i8<32, true><<<g, 256, 0, st>>>(exps, off, btk, xq, xd, act, nullptr, D, inter, beta, lb);
        else k_moe_pf_sf4_i8<64, true><<<g, 256, 0, st>>>(exps, off, btk, xq, xd, act, nullptr, D, inter, beta, lb);
    } else {
        const dim3 g((D + 127) / 128, n_experts, zt);
        if (bm == 32) k_moe_pf_sf4_i8<32, false><<<g, 256, 0, st>>>(exps, off, nullptr, aq, ad, nullptr, by, D, inter, 0.f, 0.f);
        else k_moe_pf_sf4_i8<64, false><<<g, 256, 0, st>>>(exps, off, nullptr, aq, ad, nullptr, by, D, inter, 0.f, 0.f);
    }
}

}  // namespace strata::inkling2::hip
