// ink_sy_codecs.h - Intel Arc SYCL SG16 Linear & MoE Kernels for Inkling-2-Mini (.strata v1.1)
// CRITICAL BATTLEMAGE B70 RULE: ZERO reqd_sub_group_size(8)! Every kernel strictly uses INK_REQD_SG16!
// Features:
//   - Deterministic dp4a (`sycl::ext::oneapi::dot_acc`) Q8_ROW linear + Q8_1 activation quantization
//   - Register-LUT SF4 (lossless NVFP4) + MX8 (lossless MXFP8) + S_K (S2_K..S8_K + paired 2-row w2 SB_0..SB_8)
//   - SG16 FP16 tiled GEMM + optional oneMKL XMX GEMM for prefill & c=32/64 batching
//   - GPU-side MoE binning (ink_sy_moe_plan) + fused w13 SITU(4.0, 25.0) + w2 + reduce8
// 100% MIT clean-room implementation.
#pragma once

#include <algorithm>
#include <cstdint>

#ifdef INK_SYCL_USE_ONEMKL
#include <oneapi/mkl/blas.hpp>
#endif

#include "ink_sy_common.h"
#if __has_include(<sycl/ext/oneapi/experimental/group_load_store.hpp>)
#include <sycl/ext/oneapi/experimental/group_load_store.hpp>
#define INK_HAS_GROUP_LOAD 1
#endif

namespace ink_sycl {

template <int F>
inline void sk_q8_sy(const uint8_t* pl, int j, int* q) {
    if constexpr (F == FMT_S2) {
        const uint8_t b0 = pl[j >> 2], b1 = pl[(j >> 2) + 1];
        q[0] = b0 & 3; q[1] = (b0 >> 2) & 3; q[2] = (b0 >> 4) & 3; q[3] = (b0 >> 6) & 3;
        q[4] = b1 & 3; q[5] = (b1 >> 2) & 3; q[6] = (b1 >> 4) & 3; q[7] = (b1 >> 6) & 3;
    } else if constexpr (F == FMT_S3) {
        const uint8_t b0 = pl[j >> 2], b1 = pl[(j >> 2) + 1], h = pl[64 + (j >> 3)];
        q[0] = (b0 & 3) | (((h >> 0) & 1) << 2); q[1] = ((b0 >> 2) & 3) | (((h >> 1) & 1) << 2);
        q[2] = ((b0 >> 4) & 3) | (((h >> 2) & 1) << 2); q[3] = ((b0 >> 6) & 3) | (((h >> 3) & 1) << 2);
        q[4] = (b1 & 3) | (((h >> 4) & 1) << 2); q[5] = ((b1 >> 2) & 3) | (((h >> 5) & 1) << 2);
        q[6] = ((b1 >> 4) & 3) | (((h >> 6) & 1) << 2); q[7] = ((b1 >> 6) & 3) | (((h >> 7) & 1) << 2);
    } else if constexpr (F == FMT_S4) {
        const uint8_t b0 = pl[(j >> 1) + 0], b1 = pl[(j >> 1) + 1], b2 = pl[(j >> 1) + 2], b3 = pl[(j >> 1) + 3];
        q[0] = b0 & 15; q[1] = b0 >> 4; q[2] = b1 & 15; q[3] = b1 >> 4;
        q[4] = b2 & 15; q[5] = b2 >> 4; q[6] = b3 & 15; q[7] = b3 >> 4;
    } else if constexpr (F == FMT_S5) {
        const uint8_t b0 = pl[(j >> 1) + 0], b1 = pl[(j >> 1) + 1], b2 = pl[(j >> 1) + 2], b3 = pl[(j >> 1) + 3];
        const uint8_t h = pl[128 + (j >> 3)];
        q[0] = (b0 & 15) | (((h >> 0) & 1) << 4); q[1] = (b0 >> 4) | (((h >> 1) & 1) << 4);
        q[2] = (b1 & 15) | (((h >> 2) & 1) << 4); q[3] = (b1 >> 4) | (((h >> 3) & 1) << 4);
        q[4] = (b2 & 15) | (((h >> 4) & 1) << 4); q[5] = (b2 >> 4) | (((h >> 5) & 1) << 4);
        q[6] = (b3 & 15) | (((h >> 6) & 1) << 4); q[7] = (b3 >> 4) | (((h >> 7) & 1) << 4);
    } else if constexpr (F == FMT_S6) {
        const uint8_t b0 = pl[(j >> 1) + 0], b1 = pl[(j >> 1) + 1], b2 = pl[(j >> 1) + 2], b3 = pl[(j >> 1) + 3];
        const uint8_t h0 = pl[128 + (j >> 2)], h1 = pl[128 + (j >> 2) + 1];
        q[0] = (b0 & 15) | (((h0 >> 0) & 3) << 4); q[1] = (b0 >> 4) | (((h0 >> 2) & 3) << 4);
        q[2] = (b1 & 15) | (((h0 >> 4) & 3) << 4); q[3] = (b1 >> 4) | (((h0 >> 6) & 3) << 4);
        q[4] = (b2 & 15) | (((h1 >> 0) & 3) << 4); q[5] = (b2 >> 4) | (((h1 >> 2) & 3) << 4);
        q[6] = (b3 & 15) | (((h1 >> 4) & 3) << 4); q[7] = (b3 >> 4) | (((h1 >> 6) & 3) << 4);
    } else {
        const int8_t* qp = reinterpret_cast<const int8_t*>(pl + j);
        for (int i = 0; i < 8; ++i) q[i] = (int)qp[i];
    }
}

// ============================================================================
// 0B. Native Lossless Unsloth GGUF_RAW (code 20) SG16 Decoders (All 14 GGML Types)
//     Priority: Q4_K(12), Q5_K(13), Q6_K(14), Q8_0(8), IQ4_XS(23), IQ3_XXS(18)
//     Plus: IQ4_NL(20), Q4_0(2), Q3_K(11), Q2_K(10), MXFP4(39), BF16(30), F16(1), F32(0)
// ============================================================================
struct InkSyGgufExpertHeader {
    uint32_t magic;      // 0x47475546 ('GGUF')
    uint32_t gu_type;
    uint32_t dn_type;
    uint32_t flags;      // bit 0 = interleaved (2j=gate, 2j+1=up)
    float gate_scale;
    float up_scale;
    float dn_scale;
    uint32_t reserved;
};

inline size_t sy_gguf_row_bytes(uint32_t t, int K) {
    switch (t) {
        case 0:  return (size_t)K * 4u;
        case 1:  return (size_t)K * 2u;
        case 2:  return (size_t)(K >> 5) * 18u;
        case 8:  return (size_t)(K >> 5) * 34u;
        case 10: return (size_t)(K >> 8) * 84u;
        case 11: return (size_t)(K >> 8) * 110u;
        case 12: return (size_t)(K >> 8) * 144u;
        case 13: return (size_t)(K >> 8) * 176u;
        case 14: return (size_t)(K >> 8) * 210u;
        case 18: return (size_t)(K >> 8) * 98u;
        case 20: return (size_t)(K >> 5) * 18u;
        case 23: return (size_t)(K >> 8) * 136u;
        case 30: return (size_t)K * 2u;
        case 39: return (size_t)(K >> 5) * 17u;
        default: return 0u;
    }
}

inline uint32_t sy_ld32(const uint8_t* p) {
    return (uint32_t)p[0] | ((uint32_t)p[1] << 8) | ((uint32_t)p[2] << 16) | ((uint32_t)p[3] << 24);
}

inline float sy_iq4_nl(uint32_t i) {
    constexpr float kLut[16] = {
        -127.0f, -104.0f, -83.0f, -65.0f, -49.0f, -35.0f, -22.0f, -10.0f,
           1.0f,   13.0f,  25.0f,  38.0f,  53.0f,  69.0f,  89.0f, 113.0f
    };
    return kLut[i & 15u];
}

inline uint8_t sy_iq3_xxs_sign(uint32_t i) {
    constexpr uint8_t kSigns[128] = {
          0, 129, 130,   3, 132,   5,   6, 135, 136,   9,  10, 139,  12, 141, 142,  15,
        144,  17,  18, 147,  20, 149, 150,  23,  24, 153, 154,  27, 156,  29,  30, 159,
        160,  33,  34, 163,  36, 165, 166,  39,  40, 169, 170,  43, 172,  45,  46, 175,
         48, 177, 178,  51, 180,  53,  54, 183, 184,  57,  58, 187,  60, 189, 190,  63,
        192,  65,  66, 195,  68, 197, 198,  71,  72, 201, 202,  75, 204,  77,  78, 207,
         80, 209, 210,  83, 212,  85,  86, 215, 216,  89,  90, 219,  92, 221, 222,  95,
         96, 225, 226,  99, 228, 101, 102, 231, 232, 105, 106, 235, 108, 237, 238, 111,
        240, 113, 114, 243, 116, 245, 246, 119, 120, 249, 250, 123, 252, 125, 126, 255
    };
    return kSigns[i & 127u];
}

inline uint32_t sy_iq3_xxs_grid(uint32_t i) {
    constexpr uint32_t kGrid[256] = {
        0x00000000u, 0x00000002u, 0x00000200u, 0x00000202u, 0x00020000u, 0x00020002u, 0x00020200u, 0x00020202u,
        0x02000000u, 0x02000002u, 0x02000200u, 0x02000202u, 0x02020000u, 0x02020002u, 0x02020200u, 0x02020202u,
        0x00000004u, 0x00000204u, 0x00020004u, 0x00020204u, 0x02000004u, 0x02000204u, 0x02020004u, 0x02020204u,
        0x00000400u, 0x00000402u, 0x00000404u, 0x00020400u, 0x00020402u, 0x00020404u, 0x02000400u, 0x02000402u,
        0x02000404u, 0x02020400u, 0x02020402u, 0x02020404u, 0x00040000u, 0x00040002u, 0x00040004u, 0x00040200u,
        0x00040202u, 0x00040204u, 0x00040400u, 0x00040402u, 0x00040404u, 0x02040000u, 0x02040002u, 0x02040004u,
        0x02040200u, 0x02040202u, 0x02040204u, 0x02040400u, 0x02040402u, 0x02040404u, 0x04000000u, 0x04000002u,
        0x04000004u, 0x04000200u, 0x04000202u, 0x04000204u, 0x04000400u, 0x04000402u, 0x04000404u, 0x04020000u,
        0x04020002u, 0x04020004u, 0x04020200u, 0x04020202u, 0x04020204u, 0x04020400u, 0x04020402u, 0x04020404u,
        0x04040000u, 0x04040002u, 0x04040004u, 0x04040200u, 0x04040202u, 0x04040204u, 0x04040400u, 0x04040402u,
        0x04040404u, 0x00000006u, 0x00000206u, 0x00000406u, 0x00020006u, 0x00020206u, 0x00020406u, 0x00040006u,
        0x00040206u, 0x00040406u, 0x02000006u, 0x02000206u, 0x02000406u, 0x02020006u, 0x02020206u, 0x02020406u,
        0x02040006u, 0x02040206u, 0x02040406u, 0x04000006u, 0x04000206u, 0x04000406u, 0x04020006u, 0x04020206u,
        0x04020406u, 0x04040006u, 0x04040206u, 0x04040406u, 0x00000600u, 0x00000602u, 0x00000604u, 0x00000606u,
        0x00020600u, 0x00020602u, 0x00020604u, 0x00020606u, 0x00040600u, 0x00040602u, 0x00040604u, 0x00040606u,
        0x02000600u, 0x02000602u, 0x02000604u, 0x02000606u, 0x02020600u, 0x02020602u, 0x02020604u, 0x02020606u,
        0x02040600u, 0x02040602u, 0x02040604u, 0x02040606u, 0x04000600u, 0x04000602u, 0x04000604u, 0x04000606u,
        0x04020600u, 0x04020602u, 0x04020604u, 0x04020606u, 0x04040600u, 0x04040602u, 0x04040604u, 0x04040606u,
        0x00060000u, 0x00060002u, 0x00060004u, 0x00060006u, 0x00060200u, 0x00060202u, 0x00060204u, 0x00060206u,
        0x00060400u, 0x00060402u, 0x00060404u, 0x00060406u, 0x00060600u, 0x00060602u, 0x00060604u, 0x00060606u,
        0x02060000u, 0x02060002u, 0x02060004u, 0x02060006u, 0x02060200u, 0x02060202u, 0x02060204u, 0x02060206u,
        0x02060400u, 0x02060402u, 0x02060404u, 0x02060406u, 0x02060600u, 0x02060602u, 0x02060604u, 0x02060606u,
        0x04060000u, 0x04060002u, 0x04060004u, 0x04060006u, 0x04060200u, 0x04060202u, 0x04060204u, 0x04060206u,
        0x04060400u, 0x04060402u, 0x04060404u, 0x04060406u, 0x04060600u, 0x04060602u, 0x04060604u, 0x04060606u,
        0x06000000u, 0x06000002u, 0x06000004u, 0x06000006u, 0x06000200u, 0x06000202u, 0x06000204u, 0x06000206u,
        0x06000400u, 0x06000402u, 0x06000404u, 0x06000406u, 0x06000600u, 0x06000602u, 0x06000604u, 0x06000606u,
        0x06020000u, 0x06020002u, 0x06020004u, 0x06020006u, 0x06020200u, 0x06020202u, 0x06020204u, 0x06020206u,
        0x06020400u, 0x06020402u, 0x06020404u, 0x06020406u, 0x06020600u, 0x06020602u, 0x06020604u, 0x06020606u,
        0x06040000u, 0x06040002u, 0x06040004u, 0x06040006u, 0x06040200u, 0x06040202u, 0x06040204u, 0x06040206u,
        0x06040400u, 0x06040402u, 0x06040404u, 0x06040406u, 0x06040600u, 0x06040602u, 0x06040604u, 0x06040606u,
        0x06060000u, 0x06060002u, 0x06060004u, 0x06060006u, 0x06060200u, 0x06060202u, 0x06060204u, 0x06060206u,
        0x06060400u, 0x06060402u, 0x06060404u, 0x06060406u, 0x06060600u, 0x06060602u, 0x06060604u, 0x06060606u
    };
    return kGrid[i & 255u];
}

inline int sy_gguf_q3k_scale(const uint8_t* sc, int is) {
    const uint32_t a0 = sy_ld32(sc + 0);
    const uint32_t a1 = sy_ld32(sc + 4);
    const uint32_t a2 = sy_ld32(sc + 8);
    constexpr uint32_t km1 = 0x03030303u, km2 = 0x0f0f0f0fu;
    const uint32_t u0 = (a0 & km2) | (((a2 >> 0) & km1) << 4);
    const uint32_t u1 = (a1 & km2) | (((a2 >> 2) & km1) << 4);
    const uint32_t u2 = ((a0 >> 4) & km2) | (((a2 >> 4) & km1) << 4);
    const uint32_t u3 = ((a1 >> 4) & km2) | (((a2 >> 6) & km1) << 4);
    const uint32_t w = (is < 4) ? u0 : (is < 8) ? u1 : (is < 12) ? u2 : u3;
    return (int)((w >> ((is & 3) * 8)) & 0xFFu) - 32;
}

inline void sy_gguf_decode_8(uint32_t t, const uint8_t* row, int col, float ext_sc, float* out) {
    if (t == 12u) {  // Q4_K (144 B / 256)
        const uint8_t* b = row + (size_t)(col >> 8) * 144u;
        const float d = h2f_bits(ld16(b)) * ext_sc;
        const float dmin = h2f_bits(ld16(b + 2)) * ext_sc;
        const int j = col & 255, g = j >> 6, is_hi = (j >> 5) & 1, is = g * 2 + is_hi;
        const uint8_t* sc = b + 4;
        uint32_t s, m;
        if (is < 4) { s = sc[is] & 63u; m = sc[is + 4] & 63u; }
        else { s = (sc[is + 4] & 0xFu) | ((sc[is - 4] >> 6) << 4); m = (sc[is + 4] >> 4) | ((sc[is] >> 6) << 4); }
        const float d1 = d * (float)s, m1 = dmin * (float)m;
        const uint8_t* qs = b + 16 + g * 32 + (j & 31);
        for (int i = 0; i < 8; ++i) {
            const uint32_t q = is_hi ? (qs[i] >> 4) : (qs[i] & 0xFu);
            out[i] = d1 * (float)q - m1;
        }
    } else if (t == 13u) {  // Q5_K (176 B / 256)
        const uint8_t* b = row + (size_t)(col >> 8) * 176u;
        const float d = h2f_bits(ld16(b)) * ext_sc;
        const float dmin = h2f_bits(ld16(b + 2)) * ext_sc;
        const int j = col & 255, g = j >> 6, is_hi = (j >> 5) & 1, is = g * 2 + is_hi;
        const uint8_t* sc = b + 4;
        uint32_t s, m;
        if (is < 4) { s = sc[is] & 63u; m = sc[is + 4] & 63u; }
        else { s = (sc[is + 4] & 0xFu) | ((sc[is - 4] >> 6) << 4); m = (sc[is + 4] >> 4) | ((sc[is] >> 6) << 4); }
        const float d1 = d * (float)s, m1 = dmin * (float)m;
        const uint8_t* qh = b + 16 + (j & 31);
        const uint8_t* qs = b + 48 + g * 32 + (j & 31);
        const uint32_t hmask = 1u << is;
        for (int i = 0; i < 8; ++i) {
            const uint32_t q = (is_hi ? (qs[i] >> 4) : (qs[i] & 0xFu)) | ((qh[i] & hmask) ? 16u : 0u);
            out[i] = d1 * (float)q - m1;
        }
    } else if (t == 14u) {  // Q6_K (210 B / 256)
        const uint8_t* b = row + (size_t)(col >> 8) * 210u;
        const float d = h2f_bits(ld16(b + 208)) * ext_sc;
        const int j = col & 255, h = j >> 7, q4 = (j >> 5) & 3, l0 = j & 31;
        const uint8_t* ql = b + h * 64 + ((q4 & 1) ? 32 : 0) + l0;
        const uint8_t* qh = b + 128 + h * 32 + l0;
        const float sc = d * (float)((int8_t)b[192 + h * 8 + q4 * 2 + (l0 >> 4)]);
        const int sh_ql = (q4 >= 2) ? 4 : 0;
        const int sh_qh = q4 * 2;
        for (int i = 0; i < 8; ++i) {
            const int q = (int)(((ql[i] >> sh_ql) & 0xFu) | (((qh[i] >> sh_qh) & 3u) << 4)) - 32;
            out[i] = sc * (float)q;
        }
    } else if (t == 8u) {  // Q8_0 (34 B / 32)
        const uint8_t* b = row + (size_t)(col >> 5) * 34u;
        const float d = h2f_bits(ld16(b)) * ext_sc;
        const int8_t* qs = reinterpret_cast<const int8_t*>(b + 2 + (col & 31));
        for (int i = 0; i < 8; ++i) out[i] = d * (float)qs[i];
    } else if (t == 23u) {  // IQ4_XS (136 B / 256)
        const uint8_t* b = row + (size_t)(col >> 8) * 136u;
        const float d = h2f_bits(ld16(b)) * ext_sc;
        const uint16_t sh = ld16(b + 2);
        const int j = col & 255, ib = j >> 5, l = j & 31;
        const int ls = (int)(((b[4 + (ib >> 1)] >> ((ib & 1) * 4)) & 0xFu) | (((sh >> (2 * ib)) & 3u) << 4)) - 32;
        const float dl = d * (float)ls;
        const uint8_t* qs = b + 8 + ib * 16 + (l & 15);
        const int is_hi = (l >= 16);
        for (int i = 0; i < 8; ++i) {
            out[i] = dl * sy_iq4_nl(is_hi ? (qs[i] >> 4) : (qs[i] & 0xFu));
        }
    } else if (t == 18u) {  // IQ3_XXS (98 B / 256)
        const uint8_t* b = row + (size_t)(col >> 8) * 98u;
        const float d = h2f_bits(ld16(b)) * ext_sc;
        const int j = col & 255, ib = j >> 5, l = (j & 31) >> 3;
        const uint32_t aux = sy_ld32(b + 66 + ib * 4);
        const float db = d * (0.5f + (float)(aux >> 28)) * 0.5f;
        const uint8_t signs = sy_iq3_xxs_sign((aux >> (7 * l)) & 127u);
        const uint32_t g0 = sy_iq3_xxs_grid(b[2 + ib * 8 + l * 2 + 0]);
        const uint32_t g1 = sy_iq3_xxs_grid(b[2 + ib * 8 + l * 2 + 1]);
        for (int i = 0; i < 4; ++i) {
            const float v0 = (float)((g0 >> (i * 8)) & 0xFFu) * db;
            const float v1 = (float)((g1 >> (i * 8)) & 0xFFu) * db;
            out[i]     = (signs & (1u << i)) ? -v0 : v0;
            out[i + 4] = (signs & (1u << (i + 4))) ? -v1 : v1;
        }
    } else if (t == 20u) {  // IQ4_NL (18 B / 32)
        const uint8_t* b = row + (size_t)(col >> 5) * 18u;
        const float d = h2f_bits(ld16(b)) * ext_sc;
        const int l = col & 31, is_hi = (l >= 16);
        const uint8_t* qs = b + 2 + (l & 15);
        for (int i = 0; i < 8; ++i) out[i] = d * sy_iq4_nl(is_hi ? (qs[i] >> 4) : (qs[i] & 0xFu));
    } else if (t == 2u) {  // Q4_0 (18 B / 32)
        const uint8_t* b = row + (size_t)(col >> 5) * 18u;
        const float d = h2f_bits(ld16(b)) * ext_sc;
        const int l = col & 31, is_hi = (l >= 16);
        const uint8_t* qs = b + 2 + (l & 15);
        for (int i = 0; i < 8; ++i) {
            const int q = (int)(is_hi ? (qs[i] >> 4) : (qs[i] & 0xFu)) - 8;
            out[i] = d * (float)q;
        }
    } else if (t == 11u) {  // Q3_K (110 B / 256)
        const uint8_t* b = row + (size_t)(col >> 8) * 110u;
        const float d_all = h2f_bits(ld16(b + 108)) * ext_sc;
        const int j = col & 255, h = j >> 7, q4 = (j >> 5) & 3, l0 = j & 31;
        const float dl = d_all * (float)sy_gguf_q3k_scale(b + 96, h * 8 + q4 * 2 + (l0 >> 4));
        const uint8_t* qs = b + 32 + h * 32 + l0;
        const uint8_t* hm = b + l0;
        const int shift = q4 * 2;
        const uint32_t mbit = 1u << (h * 4 + q4);
        for (int i = 0; i < 8; ++i) {
            const int q = (int)((qs[i] >> shift) & 3u) - ((hm[i] & mbit) ? 0 : 4);
            out[i] = dl * (float)q;
        }
    } else if (t == 10u) {  // Q2_K (84 B / 256)
        const uint8_t* b = row + (size_t)(col >> 8) * 84u;
        const float d = h2f_bits(ld16(b + 80)) * ext_sc;
        const float dmin = h2f_bits(ld16(b + 82)) * ext_sc;
        const int j = col & 255, h = j >> 7, q4 = (j >> 5) & 3, l0 = j & 31;
        const uint8_t scb = b[h * 8 + q4 * 2 + (l0 >> 4)];
        const float dl = d * (float)(scb & 0xFu), ml = dmin * (float)(scb >> 4);
        const uint8_t* qs = b + 16 + h * 32 + l0;
        const int shift = q4 * 2;
        for (int i = 0; i < 8; ++i) out[i] = dl * (float)((qs[i] >> shift) & 3u) - ml;
    } else if (t == 39u) {  // MXFP4 (17 B / 32)
        const uint8_t* b = row + (size_t)(col >> 5) * 17u;
        const float d = ue8m0_to_f32(b[0]) * 2.0f * ext_sc;
        const int l = col & 31, is_hi = (l >= 16);
        const uint8_t* qs = b + 1 + (l & 15);
        for (int i = 0; i < 8; ++i) out[i] = d * e2m1_lut(is_hi ? (qs[i] >> 4) : (qs[i] & 0xFu));
    } else if (t == 30u) {  // BF16
        const uint8_t* p = row + (size_t)col * 2u;
        for (int i = 0; i < 8; ++i) out[i] = bf16_to_f32(ld16(p + i * 2)) * ext_sc;
    } else if (t == 1u) {   // F16
        const uint8_t* p = row + (size_t)col * 2u;
        for (int i = 0; i < 8; ++i) out[i] = h2f_bits(ld16(p + i * 2)) * ext_sc;
    } else {                // F32
        const float* p = reinterpret_cast<const float*>(row) + col;
        for (int i = 0; i < 8; ++i) out[i] = p[i] * ext_sc;
    }
}

// ============================================================================
// 1. Dense Trunk & DSpark Linear Kernels (SG16: BF16, Q8_ROW w/ dp4a, F32, S_K)
// ============================================================================

#ifdef INK_HAS_GROUP_LOAD
// A striped subgroup fetch loads consecutive K steps for all 16 lanes.
// Use integer packets so the native block read preserves float bit patterns.
template <bool BF16>
inline void k_streaming_gemv(const std::conditional_t<BF16, uint16_t, float>* weights, int n, int k,
                             const float* x, float* y) {
    namespace ex = sycl::ext::oneapi::experimental;
    constexpr int rows = BF16 ? 4 : 1;
    Q().parallel_for(NR((n + rows - 1) / rows, rows * 16), [=](sycl::nd_item<1> it) INK_REQD_SG16 {
        const int row = it.get_global_id(0) / 16, lane = it.get_local_id(0) % 16;
        if (row >= n)
            return;
        auto sg = it.get_sub_group();
        float acc = 0.f;
        const auto prop = ex::properties{ex::data_placement_striped, ex::full_group};
        if constexpr (!BF16) {
            for (int k0 = 0; k0 < k; k0 += 128) {
                sycl::vec<uint32_t, 8> w, a;
                ex::group_load(sg, reinterpret_cast<const uint32_t*>(weights + (size_t)row * k + k0), w,
                               prop);
                ex::group_load(sg, reinterpret_cast<const uint32_t*>(x + k0), a, prop);
#pragma unroll
                for (int j = 0; j < 8; ++j)
                    acc += sycl::bit_cast<float>(static_cast<uint32_t>(w[j])) *
                           sycl::bit_cast<float>(static_cast<uint32_t>(a[j]));
            }
        } else {
            for (int k0 = 0; k0 < k; k0 += 256) {
                sycl::vec<uint64_t, 4> w;
                ex::group_load(sg, reinterpret_cast<const uint64_t*>(weights + (size_t)row * k + k0), w,
                               prop);
#pragma unroll
                for (int j = 0; j < 4; ++j) {
                    auto a = *reinterpret_cast<const sycl::vec<float, 4>*>(x + k0 + 64 * j + lane * 4);
                    auto b = sycl::bit_cast<sycl::vec<uint16_t, 4>>(static_cast<uint64_t>(w[j]));
                    acc += bf16_to_f32(b[0]) * a[0] + bf16_to_f32(b[1]) * a[1] + bf16_to_f32(b[2]) * a[2] +
                           bf16_to_f32(b[3]) * a[3];
                }
            }
        }
        const float sum = sg_sum(it, acc);
        if (lane == 0)
            y[row] = sum;
    });
}
#endif

#ifdef INK_HAS_GROUP_LOAD
// Every tree row uses the same matrix. Load one striped BF16 packet per
// subgroup and carry all node accumulators in registers across K.
template<int Nodes>
inline void k_tree_bf16_projection(const uint16_t* weights, const float* x, float* y, int n, int k) {
    namespace ex = sycl::ext::oneapi::experimental;
    Q().parallel_for(NR((n + 3) / 4, 64), [=](sycl::nd_item<1> it) INK_REQD_SG16 {
        const int row = it.get_global_id(0) / 16, lane = it.get_local_id(0) % 16;
        if (row >= n) return;
        const auto sg = it.get_sub_group();
        const auto prop = ex::properties{ex::data_placement_striped, ex::full_group};
        float sum[Nodes]{};
        for (int k0 = 0; k0 < k; k0 += 256) {
            sycl::vec<uint64_t, 4> packet;
            ex::group_load(sg, reinterpret_cast<const uint64_t*>(weights + (size_t)row * k + k0), packet, prop);
#pragma unroll
            for (int j = 0; j < 4; ++j) {
                const auto w = sycl::bit_cast<sycl::vec<uint16_t, 4>>(static_cast<uint64_t>(packet[j]));
#pragma unroll
                for (int m = 0; m < Nodes; ++m) {
                    const auto a = *reinterpret_cast<const sycl::vec<float, 4>*>(x + m * k + k0 + 64 * j + lane * 4);
                    sum[m] += bf16_to_f32(w[0]) * a[0] + bf16_to_f32(w[1]) * a[1] +
                              bf16_to_f32(w[2]) * a[2] + bf16_to_f32(w[3]) * a[3];
                }
            }
        }
#pragma unroll
        for (int m = 0; m < Nodes; ++m) {
            const float v = sg_sum(it, sum[m]);
            if (!lane) y[(size_t)m * n + row] = v;
        }
    });
}

inline bool tree_bf16_projection(const InkTensorInfo& weight, const float* x, float* y, int nodes,
                                 int output_rows = -1) {
    if (weight.fmt != FMT_BF16 || weight.cols() % 256 ||
        reinterpret_cast<uintptr_t>(weight.dev_ptr) % 16 || reinterpret_cast<uintptr_t>(x) % 16) return false;
    auto run = [&](auto count) {
        k_tree_bf16_projection<decltype(count)::value>(reinterpret_cast<const uint16_t*>(weight.dev_ptr),
                                                       x, y, output_rows > 0 ? output_rows : weight.rows(),
                                                       weight.cols());
    };
    switch (nodes) {
        case 2: run(std::integral_constant<int, 2>{}); break;
        case 3: run(std::integral_constant<int, 3>{}); break;
        case 4: run(std::integral_constant<int, 4>{}); break;
        case 5: run(std::integral_constant<int, 5>{}); break;
        case 6: run(std::integral_constant<int, 6>{}); break;
        case 8: run(std::integral_constant<int, 8>{}); break;
        case 10: run(std::integral_constant<int, 10>{}); break;
        case 12: run(std::integral_constant<int, 12>{}); break;
        default: return false;
    }
    return true;
}
#endif

// BF16 SG16 GEMV (16 SGs = 256 threads/WG -> 16 rows/WG)
inline void k_linear_bf16_sg16(const uint16_t* W, int N, int K, const float* X, float* Y, int M) {
#ifdef INK_HAS_GROUP_LOAD
    static const bool stream = ink_bmg_dense();
    if (M == 1 && K % 256 == 0 && stream &&
        reinterpret_cast<uintptr_t>(W) % 16 == 0 && reinterpret_cast<uintptr_t>(X) % 16 == 0) {
        k_streaming_gemv<true>(W, N, K, X, Y);
        return;
    }
#endif
    Q().parallel_for(NR((N + 15) / 16, 256), [=](sycl::nd_item<1> it) INK_REQD_SG16 {
        const int sg = (int)it.get_local_id(0) >> 4;
        const int lane = (int)it.get_local_id(0) & 15;
        const int row = (int)it.get_group(0) * 16 + sg;
        if (row >= N) return;
        const uint16_t* wp = W + (size_t)row * K;
        for (int m0 = 0; m0 < M; m0 += 4) {
            const int mm = sycl::min(4, M - m0);
            float a[4] = {0, 0, 0, 0};
            for (int k = lane * 4; k < K; k += 64) {
                const float w0 = bf16_to_f32(wp[k + 0]), w1 = bf16_to_f32(wp[k + 1]);
                const float w2 = bf16_to_f32(wp[k + 2]), w3 = bf16_to_f32(wp[k + 3]);
                for (int i = 0; i < mm; ++i) {
                    const float* xp = X + (size_t)(m0 + i) * K + k;
                    a[i] += w0 * xp[0] + w1 * xp[1] + w2 * xp[2] + w3 * xp[3];
                }
            }
            for (int i = 0; i < mm; ++i) {
                const float v = sg_sum(it, a[i]);
                if (lane == 0) Y[(size_t)(m0 + i) * N + row] = v;
            }
        }
    });
}

// BF16 SG16 Tiled GEMM for M >= 16 (BM=32, BN=64, BK=16, 128 threads = 8 SG16)
inline void k_gemm_bf16_sg16(const uint16_t* W, int N, int K, const float* X, float* Y, int M) {
    const int nbm = (M + 31) / 32, nbn = (N + 63) / 64;
    Q().submit([&](sycl::handler& h) {
        sycl::local_accessor<float, 1> sA(32 * 17, h);
        sycl::local_accessor<float, 1> sB(64 * 17, h);
        h.parallel_for(NR((size_t)nbm * nbn, 128), [=](sycl::nd_item<1> it) INK_REQD_SG16 {
            const int gid = (int)it.get_group(0);
            const int bn = gid % nbn, bm = gid / nbn;
            const int tid = (int)it.get_local_id(0);
            const int tm = (tid >> 4) * 4, tn = (tid & 15) * 4;
            float c[4][4] = {};
            for (int k0 = 0; k0 < K; k0 += 16) {
                for (int idx = tid; idx < 32 * 16; idx += 128) {
                    const int r = idx >> 4, k = idx & 15, gm = bm * 32 + r;
                    sA[r * 17 + k] = (gm < M && k0 + k < K) ? X[(size_t)gm * K + k0 + k] : 0.f;
                }
                for (int idx = tid; idx < 64 * 16; idx += 128) {
                    const int r = idx >> 4, k = idx & 15, gn = bn * 64 + r;
                    sB[r * 17 + k] = (gn < N && k0 + k < K) ? bf16_to_f32(W[(size_t)gn * K + k0 + k]) : 0.f;
                }
                sycl::group_barrier(it.get_group());
#pragma unroll
                for (int k = 0; k < 16; ++k) {
                    const float a0 = sA[(tm + 0) * 17 + k], a1 = sA[(tm + 1) * 17 + k];
                    const float a2 = sA[(tm + 2) * 17 + k], a3 = sA[(tm + 3) * 17 + k];
                    const float b0 = sB[(tn + 0) * 17 + k], b1 = sB[(tn + 1) * 17 + k];
                    const float b2 = sB[(tn + 2) * 17 + k], b3 = sB[(tn + 3) * 17 + k];
                    c[0][0] += a0 * b0; c[0][1] += a0 * b1; c[0][2] += a0 * b2; c[0][3] += a0 * b3;
                    c[1][0] += a1 * b0; c[1][1] += a1 * b1; c[1][2] += a1 * b2; c[1][3] += a1 * b3;
                    c[2][0] += a2 * b0; c[2][1] += a2 * b1; c[2][2] += a2 * b2; c[2][3] += a2 * b3;
                    c[3][0] += a3 * b0; c[3][1] += a3 * b1; c[3][2] += a3 * b2; c[3][3] += a3 * b3;
                }
                sycl::group_barrier(it.get_group());
            }
            for (int r = 0; r < 4; ++r) {
                const int gm = bm * 32 + tm + r;
                if (gm < M) {
                    for (int j = 0; j < 4; ++j) {
                        const int gn = bn * 64 + tn + j;
                        if (gn < N) Y[(size_t)gm * N + gn] = c[r][j];
                    }
                }
            }
        });
    });
}

// Quantize X [M, K] -> Q8_ROW [M*K int8] + [M float scales] for deterministic dp4a dot_acc!
inline void k_quantize_x_q8_sg16(const float* X, int M, int K, int8_t* xq, float* xsc) {
    Q().parallel_for(NR(M, 256), [=](sycl::nd_item<1> it) INK_REQD_SG16 {
        float* sh = lmem<float, 16>(it);
        const int m = (int)it.get_group(0), tid = (int)it.get_local_id(0);
        const float* xp = X + (size_t)m * K;
        float amax = 0.0f;
        for (int k = tid; k < K; k += 256) amax = sycl::fmax(amax, sycl::fabs(xp[k]));
        auto sg = it.get_sub_group();
        amax = sycl::reduce_over_group(sg, amax, sycl::maximum<float>());
        const int w = (int)sg.get_group_linear_id(), l = (int)sg.get_local_linear_id();
        sycl::group_barrier(it.get_group());
        if (l == 0) sh[w] = amax;
        sycl::group_barrier(it.get_group());
        float gmax = (l < 16) ? sh[l] : 0.0f;
        gmax = sycl::reduce_over_group(sg, gmax, sycl::maximum<float>());
        const float d = gmax * (1.0f / 127.0f);
        const float inv = gmax > 0.0f ? (127.0f / gmax) : 0.0f;
        if (tid == 0) xsc[m] = d;
        int8_t* dst = xq + (size_t)m * K;
        for (int k = tid; k < K; k += 256) dst[k] = (int8_t)sycl::clamp((int)sycl::rint(xp[k] * inv), -127, 127);
    });
}

// Q8_ROW SG16 dp4a GEMV (deterministic dot_acc on int8x4!)
inline void k_linear_q8row_dp4a_sg16(const uint8_t* W, int N, int K, const int8_t* xq, const float* xsc, float* Y, int M, int N_total = -1) {
    const int n_scales = (N_total > 0) ? N_total : N;
    const float* scales = reinterpret_cast<const float*>(W);
    const int32_t* qw4 = reinterpret_cast<const int32_t*>(W + (size_t)n_scales * 4);
    const int32_t* xq4 = reinterpret_cast<const int32_t*>(xq);
    const int K4 = K >> 2;
    Q().parallel_for(NR((N + 15) / 16, 256), [=](sycl::nd_item<1> it) INK_REQD_SG16 {
        const int sg = (int)it.get_local_id(0) >> 4, lane = (int)it.get_local_id(0) & 15;
        const int row = (int)it.get_group(0) * 16 + sg;
        if (row >= N) return;
        const float wsc = scales[row];
        const int32_t* wp = qw4 + (size_t)row * K4;
        for (int m0 = 0; m0 < M; m0 += 4) {
            const int mm = sycl::min(4, M - m0);
            int32_t acc[4] = {0, 0, 0, 0};
            for (int k = lane; k < K4; k += 16) {
                const int32_t wv = wp[k];
                for (int i = 0; i < mm; ++i) {
                    acc[i] = dp4a_s8(wv, xq4[(size_t)(m0 + i) * K4 + k], acc[i]);
                }
            }
            for (int i = 0; i < mm; ++i) {
                const int32_t s = sycl::reduce_over_group(it.get_sub_group(), acc[i], sycl::plus<int32_t>());
                if (lane == 0) Y[(size_t)(m0 + i) * N + row] = (float)s * (wsc * xsc[m0 + i]);
            }
        }
    });
}

// Q8_ROW SG16 exact FP32-act fallback & tiled GEMM for M >= 16
inline void k_gemm_q8row_sg16(const uint8_t* W, int N, int K, const float* X, float* Y, int M, int N_total = -1) {
    const int nbm = (M + 31) / 32, nbn = (N + 63) / 64;
    const int n_scales = (N_total > 0) ? N_total : N;
    const float* scales = reinterpret_cast<const float*>(W);
    const int8_t* qw = reinterpret_cast<const int8_t*>(W + (size_t)n_scales * 4);
    Q().submit([&](sycl::handler& h) {
        sycl::local_accessor<float, 1> sA(32 * 17, h);
        sycl::local_accessor<float, 1> sB(64 * 17, h);
        h.parallel_for(NR((size_t)nbm * nbn, 128), [=](sycl::nd_item<1> it) INK_REQD_SG16 {
            const int gid = (int)it.get_group(0), bn = gid % nbn, bm = gid / nbn, tid = (int)it.get_local_id(0);
            const int tm = (tid >> 4) * 4, tn = (tid & 15) * 4;
            float c[4][4] = {};
            for (int k0 = 0; k0 < K; k0 += 16) {
                for (int idx = tid; idx < 32 * 16; idx += 128) {
                    const int r = idx >> 4, k = idx & 15, gm = bm * 32 + r;
                    sA[r * 17 + k] = (gm < M && k0 + k < K) ? X[(size_t)gm * K + k0 + k] : 0.f;
                }
                for (int idx = tid; idx < 64 * 16; idx += 128) {
                    const int r = idx >> 4, k = idx & 15, gn = bn * 64 + r;
                    sB[r * 17 + k] = (gn < N && k0 + k < K) ? (float)qw[(size_t)gn * K + k0 + k] : 0.f;
                }
                sycl::group_barrier(it.get_group());
#pragma unroll
                for (int k = 0; k < 16; ++k) {
                    const float a0 = sA[(tm + 0) * 17 + k], a1 = sA[(tm + 1) * 17 + k];
                    const float a2 = sA[(tm + 2) * 17 + k], a3 = sA[(tm + 3) * 17 + k];
                    const float b0 = sB[(tn + 0) * 17 + k], b1 = sB[(tn + 1) * 17 + k];
                    const float b2 = sB[(tn + 2) * 17 + k], b3 = sB[(tn + 3) * 17 + k];
                    c[0][0] += a0 * b0; c[0][1] += a0 * b1; c[0][2] += a0 * b2; c[0][3] += a0 * b3;
                    c[1][0] += a1 * b0; c[1][1] += a1 * b1; c[1][2] += a1 * b2; c[1][3] += a1 * b3;
                    c[2][0] += a2 * b0; c[2][1] += a2 * b1; c[2][2] += a2 * b2; c[2][3] += a2 * b3;
                    c[3][0] += a3 * b0; c[3][1] += a3 * b1; c[3][2] += a3 * b2; c[3][3] += a3 * b3;
                }
                sycl::group_barrier(it.get_group());
            }
            for (int r = 0; r < 4; ++r) {
                const int gm = bm * 32 + tm + r;
                if (gm < M) {
                    for (int j = 0; j < 4; ++j) {
                        const int gn = bn * 64 + tn + j;
                        if (gn < N) Y[(size_t)gm * N + gn] = c[r][j] * scales[gn];
                    }
                }
            }
        });
    });
}

#ifdef INK_HAS_GROUP_LOAD
// A subgroup reuses four token rows across its adjacent output channels.
// Striped loads preserve the original order within each FP32 dot.
template<int Outputs, int Subgroups>
inline void k_f32_prefill_pairs(const float* weights, int n, int k, const float* x, float* y, int m) {
    namespace ex = sycl::ext::oneapi::experimental;
    constexpr int tokens = 4, packet = 8;
    Q().parallel_for(NR2(n / (Outputs * Subgroups), (m + tokens - 1) / tokens, 16 * Subgroups),
                     [=](sycl::nd_item<2> it) INK_REQD_SG16 {
        const auto sg = it.get_sub_group();
        const int row = (it.get_group(1) * Subgroups + sg.get_group_linear_id()) * Outputs;
        const int first = it.get_group(0) * tokens;
        const auto prop = ex::properties{ex::data_placement_striped, ex::full_group};
        float acc[tokens][Outputs]{};
        for (int k0 = 0; k0 < k; k0 += 16 * packet) {
            sycl::vec<uint32_t, packet> weights_vec[Outputs];
#pragma unroll
            for (int o = 0; o < Outputs; ++o)
                ex::group_load(sg, reinterpret_cast<const uint32_t*>(weights + size_t(row + o) * k + k0),
                               weights_vec[o], prop);
#pragma unroll
            for (int t = 0; t < tokens; ++t) {
                sycl::vec<uint32_t, packet> values;
                const int tok = sycl::min(first + t, m - 1);
                ex::group_load(sg, reinterpret_cast<const uint32_t*>(x + size_t(tok) * k + k0), values, prop);
#pragma unroll
                for (int j = 0; j < packet; ++j) {
                    const float a = sycl::bit_cast<float>(static_cast<uint32_t>(values[j]));
#pragma unroll
                    for (int o = 0; o < Outputs; ++o)
                        acc[t][o] += sycl::bit_cast<float>(static_cast<uint32_t>(weights_vec[o][j])) * a;
                }
            }
        }
        for (int t = 0; t < tokens && first + t < m; ++t)
#pragma unroll
            for (int o = 0; o < Outputs; ++o) {
                const float value = sg_sum(it, acc[t][o]);
                if (!sg.get_local_linear_id()) y[size_t(first + t) * n + row + o] = value;
            }
    });
}
#endif

// One output row of an F32 linear by one sub-group (16 lanes, lane l sums the products of k = l, l + 16, ... in ascending k, then sg_sum). The loads of U
// consecutive products of a lane are issued together before the U multiply-adds, which run in the same order as before, so the value is bit for bit
// the one of the plain loop; what changes is that a row no longer waits for one memory round trip per product.
template <int U>
inline void f32_row_dot_sg16(const sycl::nd_item<1>& it, const float* wp, const float* X, float* Y, int N, int K, int M, int row, int lane) {
    for (int m0 = 0; m0 < M; m0 += 4) {
        const int mm = sycl::min(4, M - m0);
        float a[4] = {0, 0, 0, 0};
        for (int k0 = lane; k0 < K; k0 += 16 * U) {
            float w[U], x[4][U];
#pragma unroll
            for (int u = 0; u < U; ++u) {
                const int k = k0 + 16 * u;
                if (k < K) {
                    w[u] = wp[k];
                    for (int i = 0; i < mm; ++i) x[i][u] = X[(size_t)(m0 + i) * K + k];
                }
            }
#pragma unroll
            for (int u = 0; u < U; ++u) {
                const int k = k0 + 16 * u;
                if (k < K) {
                    for (int i = 0; i < mm; ++i) a[i] += w[u] * x[i][u];
                }
            }
        }
        for (int i = 0; i < mm; ++i) {
            const float v = sg_sum(it, a[i]);
            if (lane == 0) Y[(size_t)(m0 + i) * N + row] = v;
        }
    }
}

// Three independent F32 linears of the same shape in one launch (the attn / k / v loop controllers of a visit): 3 * N work-groups instead of three
// launches of N.
inline void k_linear_f32x3_sg16(const float* W0, const float* W1, const float* W2, int N, int K,
                                const float* X0, const float* X1, const float* X2, float* Y0, float* Y1, float* Y2, int M) {
    Q().parallel_for(NR((size_t)N * 3, 16), [=](sycl::nd_item<1> it) INK_REQD_SG16 {
        const int g = (int)it.get_group(0), lane = (int)it.get_local_id(0);
        const int mat = g / N, row = g % N;
        const float* W = (mat == 0) ? W0 : (mat == 1 ? W1 : W2);
        const float* X = (mat == 0) ? X0 : (mat == 1 ? X1 : X2);
        float* Y = (mat == 0) ? Y0 : (mat == 1 ? Y1 : Y2);
        f32_row_dot_sg16<8>(it, W + (size_t)row * K, X, Y, N, K, M, row, lane);
    });
}

inline bool use_f32_prefill_pairs(const float* w, int n, int k, const float* x, int m) {
#ifdef INK_HAS_GROUP_LOAD
    static const bool enabled = ink_bmg_dense();
    return enabled && m >= 32 && n <= 128 && n % 8 == 0 && k % 128 == 0 &&
           reinterpret_cast<uintptr_t>(w) % 16 == 0 && reinterpret_cast<uintptr_t>(x) % 16 == 0;
#else
    (void)w; (void)n; (void)k; (void)x; (void)m;
    return false;
#endif
}

// F32 SG16 Linear (exact FP32 for routers, controllers, dynamic sconv weight_proj, attn_sink_proj)
// Tree verification has only a few rows. Give each (node, output) dot its own
// work-group while keeping the same K traversal and SG16 sum as the regular projection.
inline void k_tree_linear_f32_sg16(const float* weights, int outputs, int inputs,
                                   const float* x, float* y, int nodes) {
    Q().parallel_for(NR((size_t)nodes * outputs, 16), [=](sycl::nd_item<1> it) INK_REQD_SG16 {
        const int dot = it.get_group(0), lane = it.get_local_id(0);
        const float* w = weights + (size_t)(dot % outputs) * inputs;
        const float* row = x + (size_t)(dot / outputs) * inputs;
        float acc = 0.0f;
        for (int k = lane; k < inputs; k += 16) acc += w[k] * row[k];
        const float sum = sg_sum(it, acc);
        if (lane == 0) y[dot] = sum;
    });
}

#ifdef INK_HAS_GROUP_LOAD
// The lane visits the same K indices as the scalar SG16 dot, with the
// loads issued in striped packets of eight before those additions.
inline void k_tree_linear_f32_striped(const float* weights, int outputs, int inputs,
                                      const float* x, float* y, int nodes) {
    namespace ex = sycl::ext::oneapi::experimental;
    constexpr int subgroups = 4;
    const int dots = nodes * outputs;
    Q().parallel_for(NR((dots + subgroups - 1) / subgroups, subgroups * 16), [=](sycl::nd_item<1> it) INK_REQD_SG16 {
        const auto sg = it.get_sub_group();
        const int dot = it.get_group(0) * subgroups + sg.get_group_linear_id();
        if (dot >= dots) return;
        const auto* w = reinterpret_cast<const uint32_t*>(weights + size_t(dot % outputs) * inputs);
        const auto* a = reinterpret_cast<const uint32_t*>(x + size_t(dot / outputs) * inputs);
        const auto properties = ex::properties{ex::data_placement_striped, ex::full_group};
        float sum = 0.f;
        for (int k = 0; k < inputs; k += 128) {
            sycl::vec<uint32_t, 8> wp, xp;
            ex::group_load(sg, w + k, wp, properties);
            ex::group_load(sg, a + k, xp, properties);
#pragma unroll
            for (int j = 0; j < 8; ++j) {
                sum += sycl::bit_cast<float>(static_cast<uint32_t>(wp[j])) *
                       sycl::bit_cast<float>(static_cast<uint32_t>(xp[j]));
            }
        }
        const float value = sg_sum(it, sum);
        if (sg.get_local_linear_id() == 0) y[dot] = value;
    });
}
#endif

inline void k_linear_f32_sg16(const float* W, int N, int K, const float* X, float* Y, int M) {
#ifdef INK_HAS_GROUP_LOAD
    static const bool stream = ink_bmg_dense();
    if (M == 1 && N <= 128 && K % 128 == 0 && stream &&
        reinterpret_cast<uintptr_t>(W) % 16 == 0 && reinterpret_cast<uintptr_t>(X) % 16 == 0) {
        k_streaming_gemv<false>(W, N, K, X, Y);
        return;
    }
    if (use_f32_prefill_pairs(W, N, K, X, M)) {
        if (M >= 512)
            k_f32_prefill_pairs<4, 2>(W, N, K, X, Y, M);
        else
            k_f32_prefill_pairs<2, 4>(W, N, K, X, Y, M);
        return;
    }
#endif
    if (N <= 1024) {
        // Few rows (the loop controllers, the router, the dynamic conv taps): one sub-group per row and one row per work-group, so the rows
        // spread over the whole GPU instead of N/16 work-groups; the arithmetic of a row is the code below, bit for bit.
        Q().parallel_for(NR((size_t)N, 16), [=](sycl::nd_item<1> it) INK_REQD_SG16 {
            const int row = (int)it.get_group(0), lane = (int)it.get_local_id(0);
            f32_row_dot_sg16<8>(it, W + (size_t)row * K, X, Y, N, K, M, row, lane);
        });
        return;
    }
    Q().parallel_for(NR((N + 15) / 16, 256), [=](sycl::nd_item<1> it) INK_REQD_SG16 {
        const int sg = (int)it.get_local_id(0) >> 4, lane = (int)it.get_local_id(0) & 15;
        const int row = (int)it.get_group(0) * 16 + sg;
        if (row >= N) return;
        const float* wp = W + (size_t)row * K;
        for (int m0 = 0; m0 < M; m0 += 4) {
            const int mm = sycl::min(4, M - m0);
            float a[4] = {0, 0, 0, 0};
            for (int k = lane; k < K; k += 16) {
                const float w = wp[k];
                for (int i = 0; i < mm; ++i) a[i] += w * X[(size_t)(m0 + i) * K + k];
            }
            for (int i = 0; i < mm; ++i) {
                const float v = sg_sum(it, a[i]);
                if (lane == 0) Y[(size_t)(m0 + i) * N + row] = v;
            }
        }
    });
}

// S_K SG16 Linear (S5_K DSpark, S6_K unembed)
template <int F>
inline void k_linear_sk_sg16(const uint8_t* W, int N, int K, const float* X, float* Y, int M) {
    constexpr int SB = sk_sb_bytes(F);
    constexpr float H = sk_half(F);
    const int nsb = K >> 8;
    const size_t rowb = (size_t)nsb * SB;
    Q().parallel_for(NR((N + 15) / 16, 256), [=](sycl::nd_item<1> it) INK_REQD_SG16 {
        const int sg = (int)it.get_local_id(0) >> 4, lane = (int)it.get_local_id(0) & 15;
        const int row = (int)it.get_group(0) * 16 + sg;
        if (row >= N) return;
        const uint8_t* rp = W + (size_t)row * rowb;
        for (int m0 = 0; m0 < M; m0 += 4) {
            const int mm = sycl::min(4, M - m0);
            float acc[4] = {0, 0, 0, 0};
            for (int unit = lane; unit < nsb * 32; unit += 16) {
                const int sbi = unit >> 5, u = unit & 31, g = u >> 3, j = (g << 6) + ((u & 7) << 3);
                const uint8_t* blk = rp + (size_t)sbi * SB;
                const float eff = h2f_bits(ld16(blk)) * ((float)blk[2 + g] * (1.0f / 255.0f));
                int q[8];
                sk_q8_sy<F>(blk + 6, j, q);
                float w[8];
                for (int t = 0; t < 8; ++t) w[t] = ((float)q[t] - H) * eff;
                for (int i = 0; i < mm; ++i) {
                    const float* xp = X + (size_t)(m0 + i) * K + sbi * 256 + j;
                    float s = 0.0f;
                    for (int t = 0; t < 8; ++t) s += w[t] * xp[t];
                    acc[i] += s;
                }
            }
            for (int i = 0; i < mm; ++i) {
                const float v = sg_sum(it, acc[i]);
                if (lane == 0) Y[(size_t)(m0 + i) * N + row] = v;
            }
        }
    });
}

// ============================================================================
// 2. Inkling-2-Mini MoE Kernels (SG16: SF4, MX8, S2_K..S8_K + Paired 2-Row w2)
// ============================================================================

// GPU-side expert binning (1 WG of 128 threads = 8 SG16, 1 thread per expert e in 0..127)
// When ex_layer != nullptr, only bins experts owned by this GPU (ex_layer[e].body_ptr != nullptr) for Dual-GPU EP!
inline void k_moe_plan_sg16_q(sycl::queue& q, const int* topk_idx, int P, int* uexp, int* uoff, int* ucnt, int* pair_list, int* n_uniq_dev,
                              const InkSyExpertDev* ex_layer = nullptr) {
    q.submit([&](sycl::handler& h) {
        sycl::local_accessor<int, 1> sg_u(8, h);
        sycl::local_accessor<int, 1> sg_p(8, h);
        h.parallel_for(NR(1, 128), [=](sycl::nd_item<1> it) INK_REQD_SG16 {
            const int e = (int)it.get_local_id(0);
            auto sg = it.get_sub_group();
            const int sgi = (int)sg.get_group_linear_id(), lane = (int)sg.get_local_linear_id();
            const bool owned = (ex_layer == nullptr) || (ex_layer[e].body_ptr != nullptr);
            int c = 0;
            if (owned) {
                for (int i = 0; i < P; ++i) if (topk_idx[i] == e) ++c;
            }
            const int has = c > 0 ? 1 : 0;
            const int ex_u = sycl::exclusive_scan_over_group(sg, has, sycl::plus<int>());
            const int ex_p = sycl::exclusive_scan_over_group(sg, c, sycl::plus<int>());
            if (lane == 15) { sg_u[sgi] = ex_u + has; sg_p[sgi] = ex_p + c; }
            sycl::group_barrier(it.get_group());
            int bu = 0, bp = 0;
            for (int w = 0; w < sgi; ++w) { bu += sg_u[w]; bp += sg_p[w]; }
            const int pu = bu + ex_u, pp = bp + ex_p;
            if (has) {
                uexp[pu] = e; uoff[pu] = pp; ucnt[pu] = c;
                int k = 0;
                for (int i = 0; i < P; ++i) if (topk_idx[i] == e) pair_list[pp + k++] = i;
            }
            if (e == 127) n_uniq_dev[0] = pu + has;
        });
    });
}

inline void k_moe_plan_sg16(const int* topk_idx, int P, int* uexp, int* uoff, int* ucnt, int* pair_list, int* n_uniq_dev) {
    k_moe_plan_sg16_q(Q(), topk_idx, P, uexp, uoff, ucnt, pair_list, n_uniq_dev, nullptr);
}

// Fused w13 + SITU(4.0, 25.0) + 1/sq_xd_dn (grid: 72 x max_uniq, 256 threads = 16 SG16 -> 16 inter channels/WG)
inline void k_moe_gu_situ_sg16_q(
    sycl::queue& q,
    const InkSyExpertDev* ex_layer, const float* x_in,
    const int* uexp, const int* uoff, const int* ucnt, const int* pair_list, const int* n_uniq_dev,
    int max_uniq, float* h_act) {
    // 1152 / 16 = 72 WGs along X, max_uniq along Y
    q.parallel_for(NR2(72, (size_t)max_uniq, 256), [=](sycl::nd_item<2> it) INK_REQD_SG16 {
        const int y = (int)it.get_group(0);
        if (y >= n_uniq_dev[0]) return;
        const int sg = (int)it.get_local_id(1) >> 4, lane = (int)it.get_local_id(1) & 15;
        const int r = (int)it.get_group(1) * 16 + sg;
        if (r >= 1152) return;

        const int e = uexp[y], off = uoff[y], n = ucnt[y];
        const InkSyExpertDev ed = ex_layer[e];
        const uint8_t* body = ed.body_ptr;
        const bool is_sk_dn = (ed.dn_code != FMT_SF4 && ed.dn_code != FMT_MX8 && ed.dn_code != FMT_GGUF_RAW);
        const float inv_sq = is_sk_dn ? (1.0f / h2f_bits(ld16(body + 2 * r))) : 1.0f;

        for (int t0 = 0; t0 < n; t0 += 4) {
            const int mm = sycl::min(4, n - t0);
            int pid[4], tk[4];
            for (int i = 0; i < mm; ++i) { pid[i] = pair_list[off + t0 + i]; tk[i] = pid[i] >> 3; }
            float ag[4] = {0, 0, 0, 0}, au[4] = {0, 0, 0, 0};

            if (ed.gu_code == FMT_SF4) {
                const float* hdr = reinterpret_cast<const float*>(body);
                const float outer = hdr[0] / hdr[1];
                const uint8_t* pb = body + 16;
                const uint8_t* sb = pb + 2654208;
                const uint8_t* pg = pb + (size_t)(2 * r) * 1152;
                const uint8_t* sg_p = sb + (size_t)(2 * r) * 144;
                const uint8_t* pu = pb + (size_t)(2 * r + 1) * 1152;
                const uint8_t* su_p = sb + (size_t)(2 * r + 1) * 144;
                for (int b = lane; b < 144; b += 16) {
                    const float scg = e4m3_to_f32(sg_p[b]) * outer;
                    const float scu = e4m3_to_f32(su_p[b]) * outer;
                    const uint8_t* bg = pg + b * 8;
                    const uint8_t* bu = pu + b * 8;
                    float wg[16], wu[16];
                    for (int j = 0; j < 8; ++j) {
                        wg[2*j] = e2m1_lut(bg[j] & 15u); wg[2*j+1] = e2m1_lut(bg[j] >> 4);
                        wu[2*j] = e2m1_lut(bu[j] & 15u); wu[2*j+1] = e2m1_lut(bu[j] >> 4);
                    }
                    for (int i = 0; i < mm; ++i) {
                        const float* xp = x_in + (size_t)tk[i] * 2304 + b * 16;
                        float dg = 0.0f, du = 0.0f;
                        for (int j = 0; j < 16; ++j) { dg += wg[j] * xp[j]; du += wu[j] * xp[j]; }
                        ag[i] += dg * scg; au[i] += du * scu;
                    }
                }
            } else if (ed.gu_code == FMT_MX8) {
                const uint8_t* wb = body;
                const uint8_t* sb = body + 5308416;
                const uint8_t* wg = wb + (size_t)(2 * r) * 2304;
                const uint8_t* sg_p = sb + (size_t)(2 * r) * 72;
                const uint8_t* wu = wb + (size_t)(2 * r + 1) * 2304;
                const uint8_t* su_p = sb + (size_t)(2 * r + 1) * 72;
                for (int b = lane; b < 72; b += 16) {
                    const float scg = ue8m0_to_f32(sg_p[b]);
                    const float scu = ue8m0_to_f32(su_p[b]);
                    const uint8_t* bg = wg + b * 32;
                    const uint8_t* bu = wu + b * 32;
                    for (int i = 0; i < mm; ++i) {
                        const float* xp = x_in + (size_t)tk[i] * 2304 + b * 32;
                        float dg = 0.0f, du = 0.0f;
                        for (int j = 0; j < 32; ++j) { dg += e4m3_to_f32(bg[j]) * xp[j]; du += e4m3_to_f32(bu[j]) * xp[j]; }
                        ag[i] += dg * scg; au[i] += du * scu;
                    }
                }
            } else if (ed.gu_code == FMT_GGUF_RAW) {
                const InkSyGgufExpertHeader* hdr = reinterpret_cast<const InkSyGgufExpertHeader*>(body);
                const uint32_t gt = hdr->gu_type;
                const size_t rb = sy_gguf_row_bytes(gt, 2304);
                const bool ilv = (hdr->flags & 1u) != 0;
                const uint8_t* rg = body + 32 + (size_t)(ilv ? (2 * r) : r) * rb;
                const uint8_t* ru = body + 32 + (size_t)(ilv ? (2 * r + 1) : (1152 + r)) * rb;
                for (int unit = lane; unit < 288; unit += 16) {
                    const int col = unit << 3;
                    float wg[8], wu[8];
                    sy_gguf_decode_8(gt, rg, col, hdr->gate_scale, wg);
                    sy_gguf_decode_8(gt, ru, col, hdr->up_scale, wu);
                    for (int i = 0; i < mm; ++i) {
                        const float* xp = x_in + (size_t)tk[i] * 2304 + col;
                        float dg = 0.0f, du = 0.0f;
                        for (int t = 0; t < 8; ++t) { dg += wg[t] * xp[t]; du += wu[t] * xp[t]; }
                        ag[i] += dg; au[i] += du;
                    }
                }
            } else {
                const int F = ed.gu_code;
                const int SB = sk_sb_bytes(F);
                const float H = sk_half(F);
                const size_t rowb = 9ul * SB;
                const uint8_t* gu = body + 2304;
                const uint8_t* rg = gu + (size_t)(2 * r) * rowb;
                const uint8_t* ru = gu + (size_t)(2 * r + 1) * rowb;
                for (int unit = lane; unit < 288; unit += 16) {
                    const int sbi = unit >> 5, u = unit & 31, g = u >> 3, j = (g << 6) + ((u & 7) << 3);
                    const uint8_t* bg = rg + (size_t)sbi * SB;
                    const uint8_t* bu = ru + (size_t)sbi * SB;
                    const float effg = h2f_bits(ld16(bg)) * ((float)bg[2 + g] * (1.0f / 255.0f));
                    const float effu = h2f_bits(ld16(bu)) * ((float)bu[2 + g] * (1.0f / 255.0f));
                    int qg[8], qu[8];
                    if (F == FMT_S2) { sk_q8_sy<FMT_S2>(bg + 6, j, qg); sk_q8_sy<FMT_S2>(bu + 6, j, qu); }
                    else if (F == FMT_S3) { sk_q8_sy<FMT_S3>(bg + 6, j, qg); sk_q8_sy<FMT_S3>(bu + 6, j, qu); }
                    else if (F == FMT_S4) { sk_q8_sy<FMT_S4>(bg + 6, j, qg); sk_q8_sy<FMT_S4>(bu + 6, j, qu); }
                    else if (F == FMT_S5) { sk_q8_sy<FMT_S5>(bg + 6, j, qg); sk_q8_sy<FMT_S5>(bu + 6, j, qu); }
                    else if (F == FMT_S6) { sk_q8_sy<FMT_S6>(bg + 6, j, qg); sk_q8_sy<FMT_S6>(bu + 6, j, qu); }
                    else { sk_q8_sy<FMT_S8>(bg + 6, j, qg); sk_q8_sy<FMT_S8>(bu + 6, j, qu); }
                    for (int i = 0; i < mm; ++i) {
                        const float* xp = x_in + (size_t)tk[i] * 2304 + sbi * 256 + j;
                        float dg = 0.0f, du = 0.0f;
                        for (int t = 0; t < 8; ++t) { dg += ((float)qg[t] - H) * xp[t]; du += ((float)qu[t] - H) * xp[t]; }
                        ag[i] += dg * effg; au[i] += du * effu;
                    }
                }
            }

            for (int i = 0; i < mm; ++i) {
                const float g = sg_sum(it, ag[i]), u = sg_sum(it, au[i]);
                if (lane == 0) h_act[(size_t)pid[i] * 1152 + r] = situ_act(g, u) * inv_sq;
            }
        }
    });
}

// Down projection w2 (SF4 / MX8 / paired 2-row S_K SB_0..SB_8) * router weight -> y_pairs [P, 2304]
// Grid: 72 x max_uniq, 256 threads = 16 SG16 (each SG handles 1 row-pair = 2 output rows!)
inline void k_moe_dn_sg16_q(
    sycl::queue& q,
    const InkSyExpertDev* ex_layer, const float* topk_w,
    const int* uexp, const int* uoff, const int* ucnt, const int* pair_list, const int* n_uniq_dev,
    int max_uniq, const float* h_act, float* y_pairs) {
    q.parallel_for(NR2(72, (size_t)max_uniq, 256), [=](sycl::nd_item<2> it) INK_REQD_SG16 {
        const int yy = (int)it.get_group(0);
        if (yy >= n_uniq_dev[0]) return;
        const int sg = (int)it.get_local_id(1) >> 4, lane = (int)it.get_local_id(1) & 15;
        const int rp = (int)it.get_group(1) * 16 + sg;
        if (rp >= 1152) return;

        const int e = uexp[yy], off = uoff[yy], n = ucnt[yy];
        const InkSyExpertDev ed = ex_layer[e];
        const uint8_t* body = ed.body_ptr;

        for (int t0 = 0; t0 < n; t0 += 4) {
            const int mm = sycl::min(4, n - t0);
            int pid[4];
            for (int i = 0; i < mm; ++i) pid[i] = pair_list[off + t0 + i];
            float a0[4] = {0, 0, 0, 0}, a1[4] = {0, 0, 0, 0};

            if (ed.dn_code == FMT_SF4) {
                const float* hdr = reinterpret_cast<const float*>(body);
                const float outer = hdr[2] / hdr[3];
                const uint8_t* pb = body + ed.dn_off;
                const uint8_t* sb = pb + 1327104;
                const uint8_t* p0 = pb + (size_t)(2 * rp) * 576;
                const uint8_t* s0 = sb + (size_t)(2 * rp) * 72;
                const uint8_t* p1 = pb + (size_t)(2 * rp + 1) * 576;
                const uint8_t* s1 = sb + (size_t)(2 * rp + 1) * 72;
                for (int b = lane; b < 72; b += 16) {
                    const float sc0 = e4m3_to_f32(s0[b]) * outer;
                    const float sc1 = e4m3_to_f32(s1[b]) * outer;
                    const uint8_t* b0 = p0 + b * 8;
                    const uint8_t* b1 = p1 + b * 8;
                    float w0[16], w1[16];
                    for (int j = 0; j < 8; ++j) {
                        w0[2*j] = e2m1_lut(b0[j] & 15u); w0[2*j+1] = e2m1_lut(b0[j] >> 4);
                        w1[2*j] = e2m1_lut(b1[j] & 15u); w1[2*j+1] = e2m1_lut(b1[j] >> 4);
                    }
                    for (int i = 0; i < mm; ++i) {
                        const float* xp = h_act + (size_t)pid[i] * 1152 + b * 16;
                        float d0 = 0.0f, d1 = 0.0f;
                        for (int j = 0; j < 16; ++j) { d0 += w0[j] * xp[j]; d1 += w1[j] * xp[j]; }
                        a0[i] += d0 * sc0; a1[i] += d1 * sc1;
                    }
                }
            } else if (ed.dn_code == FMT_MX8) {
                const uint8_t* wb = body + ed.dn_off;
                const uint8_t* sb = wb + 2654208;
                const uint8_t* w0 = wb + (size_t)(2 * rp) * 1152;
                const uint8_t* s0 = sb + (size_t)(2 * rp) * 36;
                const uint8_t* w1 = wb + (size_t)(2 * rp + 1) * 1152;
                const uint8_t* s1 = sb + (size_t)(2 * rp + 1) * 36;
                for (int b = lane; b < 36; b += 16) {
                    const float sc0 = ue8m0_to_f32(s0[b]);
                    const float sc1 = ue8m0_to_f32(s1[b]);
                    const uint8_t* b0 = w0 + b * 32;
                    const uint8_t* b1 = w1 + b * 32;
                    for (int i = 0; i < mm; ++i) {
                        const float* xp = h_act + (size_t)pid[i] * 1152 + b * 32;
                        float d0 = 0.0f, d1 = 0.0f;
                        for (int j = 0; j < 32; ++j) { d0 += e4m3_to_f32(b0[j]) * xp[j]; d1 += e4m3_to_f32(b1[j]) * xp[j]; }
                        a0[i] += d0 * sc0; a1[i] += d1 * sc1;
                    }
                }
            } else if (ed.dn_code == FMT_GGUF_RAW) {
                const InkSyGgufExpertHeader* hdr = reinterpret_cast<const InkSyGgufExpertHeader*>(body);
                const uint32_t dt = hdr->dn_type;
                const size_t rb = sy_gguf_row_bytes(dt, 1152);
                const uint8_t* r0 = body + ed.dn_off + (size_t)(2 * rp) * rb;
                const uint8_t* r1 = body + ed.dn_off + (size_t)(2 * rp + 1) * rb;
                for (int unit = lane; unit < 144; unit += 16) {
                    const int col = unit << 3;
                    float w0[8], w1[8];
                    sy_gguf_decode_8(dt, r0, col, hdr->dn_scale, w0);
                    sy_gguf_decode_8(dt, r1, col, hdr->dn_scale, w1);
                    for (int i = 0; i < mm; ++i) {
                        const float* xp = h_act + (size_t)pid[i] * 1152 + col;
                        float d0 = 0.0f, d1 = 0.0f;
                        for (int t = 0; t < 8; ++t) { d0 += w0[t] * xp[t]; d1 += w1[t] * xp[t]; }
                        a0[i] += d0; a1[i] += d1;
                    }
                }
            } else {
                const int F = ed.dn_code;
                const int SB = sk_sb_bytes(F);
                const float H = sk_half(F);
                const uint8_t* pr = body + ed.dn_off + (size_t)rp * 9ul * SB;
                for (int unit = lane; unit < 288; unit += 16) {
                    const int sbi = unit >> 5, u = unit & 31, g = u >> 3, j = (g << 6) + ((u & 7) << 3);
                    const uint8_t* blk = pr + (size_t)sbi * SB;
                    const float eff = h2f_bits(ld16(blk)) * ((float)blk[2 + g] * (1.0f / 255.0f));
                    int q[8];
                    if (F == FMT_S2) sk_q8_sy<FMT_S2>(blk + 6, j, q);
                    else if (F == FMT_S3) sk_q8_sy<FMT_S3>(blk + 6, j, q);
                    else if (F == FMT_S4) sk_q8_sy<FMT_S4>(blk + 6, j, q);
                    else if (F == FMT_S5) sk_q8_sy<FMT_S5>(blk + 6, j, q);
                    else if (F == FMT_S6) sk_q8_sy<FMT_S6>(blk + 6, j, q);
                    else sk_q8_sy<FMT_S8>(blk + 6, j, q);
                    const bool is_r1 = (unit >= 144);
                    const int col = (is_r1 ? (unit - 144) : unit) << 3;
                    for (int i = 0; i < mm; ++i) {
                        const float* xp = h_act + (size_t)pid[i] * 1152 + col;
                        float s = 0.0f;
                        for (int t = 0; t < 8; ++t) s += ((float)q[t] - H) * xp[t];
                        if (is_r1) a1[i] += s * eff; else a0[i] += s * eff;
                    }
                }
            }

            for (int i = 0; i < mm; ++i) {
                const float v0 = sg_sum(it, a0[i]), v1 = sg_sum(it, a1[i]);
                if (lane == 0) {
                    const float rw = topk_w[pid[i]];
                    y_pairs[(size_t)pid[i] * 2304 + 2 * rp] = v0 * rw;
                    y_pairs[(size_t)pid[i] * 2304 + 2 * rp + 1] = v1 * rw;
                }
            }
        }
    });
}

inline void k_moe_gu_situ_sg16(
    const InkSyExpertDev* ex_layer, const float* x_in,
    const int* uexp, const int* uoff, const int* ucnt, const int* pair_list, const int* n_uniq_dev,
    int max_uniq, float* h_act) {
    k_moe_gu_situ_sg16_q(Q(), ex_layer, x_in, uexp, uoff, ucnt, pair_list, n_uniq_dev, max_uniq, h_act);
}

inline void k_moe_dn_sg16(
    const InkSyExpertDev* ex_layer, const float* topk_w,
    const int* uexp, const int* uoff, const int* ucnt, const int* pair_list, const int* n_uniq_dev,
    int max_uniq, const float* h_act, float* y_pairs) {
    k_moe_dn_sg16_q(Q(), ex_layer, topk_w, uexp, uoff, ucnt, pair_list, n_uniq_dev, max_uniq, h_act, y_pairs);
}

// SQ-20 stores SF4 blocks as 8 bytes per 16 weights and MX8 blocks as
// 32 bytes per 32 weights. Keep a whole block in registers for its dot.
template <bool SF4> inline float sq20_block_dot(const uint8_t* payload, const float* x) {
    constexpr int block = SF4 ? 16 : 32;
    float acc = 0.0f;
#pragma unroll
    for (int i = 0; i < block; i += 8) {
        const auto a = *reinterpret_cast<const sycl::vec<float, 8>*>(x + i);
        if constexpr (SF4) {
            const auto w = *reinterpret_cast<const sycl::vec<uint8_t, 4>*>(payload + i / 2);
#pragma unroll
            for (int j = 0; j < 4; ++j) {
                acc += e2m1_lut(w[j] & 15) * a[2 * j];
                acc += e2m1_lut(w[j] >> 4) * a[2 * j + 1];
            }
        } else {
            const auto w = *reinterpret_cast<const sycl::vec<uint8_t, 8>*>(payload + i);
#pragma unroll
            for (int j = 0; j < 8; ++j)
                acc += e4m3_to_f32(w[j]) * a[j];
        }
    }
    return acc;
}

template <bool Down, bool SF4>
inline void sq20_dot_pair(const InkSyExpertDev& ed, int pair, int lane, const float* x, float& first,
                          float& second) {
    constexpr int K = Down ? 1152 : 2304, block = SF4 ? 16 : 32;
    constexpr int bytes = SF4 ? 8 : 32, blocks = K / block;
    const uint8_t* w = ed.body_ptr + (Down ? ed.dn_off : (SF4 ? 16 : 0));
    const uint8_t* scales = w + 2304 * blocks * bytes;
    float outer = 1.0f;
    if constexpr (SF4) {
        const float* hdr = reinterpret_cast<const float*>(ed.body_ptr);
        outer = hdr[Down ? 2 : 0] / hdr[Down ? 3 : 1];
    }
    for (int b = lane; b < blocks; b += 16) {
        const int i0 = 2 * pair * blocks + b, i1 = i0 + blocks;
        const float sc0 = SF4 ? e4m3_to_f32(scales[i0]) * outer : ue8m0_to_f32(scales[i0]);
        const float sc1 = SF4 ? e4m3_to_f32(scales[i1]) * outer : ue8m0_to_f32(scales[i1]);
        first += sq20_block_dot<SF4>(w + (size_t)i0 * bytes, x + b * block) * sc0;
        second += sq20_block_dot<SF4>(w + (size_t)i1 * bytes, x + b * block) * sc1;
    }
}

template <bool Down>
inline void k_sq20_expert_gemv(const InkSyExpertDev* experts, const int* topk, const float* rw,
                               const float* x, float* out) {
    constexpr int rows_per_group = 8;
    Q().parallel_for(NR2(1152 / rows_per_group, 8, 16 * rows_per_group),
                     [=](sycl::nd_item<2> it) INK_REQD_SG16 {
                         const int rank = it.get_group(0),
                                   pair = it.get_group(1) * rows_per_group + it.get_local_id(1) / 16;
                         const int lane = it.get_local_id(1) % 16;
                         const auto ed = experts[topk[rank]];
                         const float* input = Down ? x + rank * 1152 : x;
                         float a = 0.0f, b = 0.0f;
                         if ((Down ? ed.dn_code : ed.gu_code) == FMT_SF4)
                             sq20_dot_pair<Down, true>(ed, pair, lane, input, a, b);
                         else
                             sq20_dot_pair<Down, false>(ed, pair, lane, input, a, b);
                         a = sg_sum(it, a);
                         b = sg_sum(it, b);
                         if (lane == 0) {
                             if constexpr (Down) {
                                 out[rank * 2304 + 2 * pair] = a * rw[rank];
                                 out[rank * 2304 + 2 * pair + 1] = b * rw[rank];
                             } else
                                 out[rank * 1152 + pair] = situ_act(a, b);
                         }
                     });
}

// Both adjacent SF4 output rows reuse a single activation vector. Each lane
// accumulates the original blocks in order before the SG16 reduction.
template <bool Down>
inline void k_sq20_sf4_pair(const InkSyExpertDev* experts, const int* ids, const float* route_weights,
                            const float* x, float* out) {
    constexpr int K = Down ? 1152 : 2304, blocks = K / 16;
    Q().parallel_for(NR2(144, 8, 128), [=](sycl::nd_item<2> it) INK_REQD_SG16 {
        const int rank = it.get_group(0), lid = it.get_local_id(1), lane = lid % 16;
        const int pair = it.get_group(1) * 8 + lid / 16;
        const auto ed = experts[ids[rank]];
        const float* hdr = reinterpret_cast<const float*>(ed.body_ptr);
        const float outer = hdr[Down ? 2 : 0] / hdr[Down ? 3 : 1];
        const uint8_t* payload = ed.body_ptr + (Down ? ed.dn_off : 16);
        const uint8_t* scales = payload + 2304 * blocks * 8;
        const float* input = Down ? x + rank * K : x;
        float sums[2]{};
        for (int b = lane; b < blocks; b += 16) {
            auto a = *reinterpret_cast<const sycl::vec<float, 16>*>(input + b * 16);
#pragma unroll
            for (int p = 0; p < 2; ++p) {
                const int i = (2 * pair + p) * blocks + b;
                auto bytes = *reinterpret_cast<const sycl::vec<uint8_t, 8>*>(payload + i * 8);
                float dot = 0.f;
#pragma unroll
                for (int j = 0; j < 8; ++j) {
                    dot += e2m1_lut(bytes[j] & 15) * a[2 * j];
                    dot += e2m1_lut(bytes[j] >> 4) * a[2 * j + 1];
                }
                sums[p] += dot * (e4m3_to_f32(scales[i]) * outer);
            }
        }
        const float a = sg_sum(it, sums[0]), b = sg_sum(it, sums[1]);
        if (!lane) {
            if constexpr (Down) {
                out[rank * 2304 + 2 * pair] = a * route_weights[rank];
                out[rank * 2304 + 2 * pair + 1] = b * route_weights[rank];
            } else {
                out[rank * 1152 + pair] = situ_act(a, b);
            }
        }
    });
}

// Store the two exact BF16 bit patterns for a compressed SF4 byte together.
// One lookup recovers the adjacent coefficients, including signed zeros.
inline constexpr auto sf4_byte_bf16 = [] {
    constexpr uint16_t magnitude[] = {0, 0x3f00, 0x3f80, 0x3fc0, 0x4000, 0x4040, 0x4080, 0x40c0};
    std::array<uint32_t, 256> pairs{};
    for (uint32_t i = 0; i < pairs.size(); ++i) {
        const uint32_t lo = magnitude[i & 7] | ((i & 8) << 12);
        const uint32_t hi = magnitude[(i >> 4) & 7] | ((i & 128) << 8);
        pairs[i] = lo | (hi << 16);
    }
    return pairs;
}();

// A nibble's magnitude is at most six and is a multiple of one half.
inline float sf4_weight_from_nibble(uint32_t nibble) {
    constexpr uint32_t doubled_magnitudes = 0xc8643210u;
    const unsigned shift = (nibble & 7u) * 4u;
    const float magnitude = float((doubled_magnitudes >> shift) & 15u) * 0.5f;
    const uint32_t sign = (nibble & 8u) << 28;
    return sycl::bit_cast<float>(sycl::bit_cast<uint32_t>(magnitude) | sign);
}

// All non-reserved E4M3 coefficients fit exactly in FP32. Construct
// the normal values directly and restore the sign, including signed zero.
inline float mx8_coefficient_f32(uint32_t code) {
    const uint32_t magnitude = code & 127u;
    const uint32_t normal = 0x3c000000u + (magnitude << 20);
    const float subnormal = float(magnitude & 7u) * (1.f / 512.f);
    const uint32_t bits = magnitude < 8 ? sycl::bit_cast<uint32_t>(subnormal) : normal;
    return sycl::bit_cast<float>(magnitude == 127 ? 0u : bits | ((code & 128u) << 24));
}

// Both output rows share each
// activation load. Keep block sums in FP32 before applying their separate scales.
template <bool Down, bool SF4, bool Cached = false, int FixedRows = 0, bool MX8Only = false, bool TreeSF4 = false>
inline void sq20_small_pairs(sycl::nd_item<2> it, const InkSyExpertDev &ed, const int *order, int n,
                             const float *w, const float *x, float *out,
                             const float* factors = nullptr, int output_tile = 0) {
    constexpr int K = Down ? 1152 : 2304, width = SF4 ? 16 : 32;
    constexpr int blocks = K / width, bytes = SF4 ? 8 : 32;
    constexpr int pairs_per_group = (Cached || TreeSF4) ? 8 : 16;
    constexpr int rows_per_tile = FixedRows ? FixedRows : (Cached ? 8 : 4);
    const int pair = (Cached ? output_tile : int(it.get_group(1))) * pairs_per_group + it.get_local_id(1) / 16, lane = it.get_local_id(1) % 16;
    const uint8_t *data = ed.body_ptr + (Down ? ed.dn_off : (SF4 ? 16 : 0));
    const uint8_t *scales = data + 2304 * blocks * bytes;
    const float *hdr = reinterpret_cast<const float *>(ed.body_ptr);
    const float outer = SF4 && !Cached ? hdr[Down ? 2 : 0] / hdr[Down ? 3 : 1] : 1.f;
    for (int r0 = 0; r0 < n; r0 += rows_per_tile) {
        const int rows = FixedRows ? FixedRows : sycl::min(rows_per_tile, n - r0);
        const float *input[rows_per_tile];
        int pid[rows_per_tile];
        for (int r = 0; r < rows; ++r) {
            pid[r] = order[r0 + r];
            input[r] = x + (Down ? pid[r] : pid[r] / 8) * K;
        }
        float a[rows_per_tile]{}, b[rows_per_tile]{};
        for (int block = lane; block < blocks; block += 16) {
            const int i0 = 2 * pair * blocks + block, i1 = i0 + blocks;
            const float s0 = Cached ? factors[scales[i0]] : SF4 ? e4m3_to_f32(scales[i0]) * outer : ue8m0_to_f32(scales[i0]);
            const float s1 = Cached ? factors[scales[i1]] : SF4 ? e4m3_to_f32(scales[i1]) * outer : ue8m0_to_f32(scales[i1]);
            float v0[width], v1[width];
            if constexpr (SF4) {
                const auto p = *reinterpret_cast<const sycl::vec<uint8_t, 8> *>(data + (size_t)i0 * bytes);
                const auto q = *reinterpret_cast<const sycl::vec<uint8_t, 8> *>(data + (size_t)i1 * bytes);
#pragma unroll
                for (int j = 0; j < 8; ++j) {
                    if constexpr (Cached) {
                        v0[2 * j] = sf4_weight_from_nibble(p[j]);
                        v0[2 * j + 1] = sf4_weight_from_nibble(p[j] >> 4);
                        v1[2 * j] = sf4_weight_from_nibble(q[j]);
                        v1[2 * j + 1] = sf4_weight_from_nibble(q[j] >> 4);
                    } else if constexpr (TreeSF4) {
                        const uint32_t u = sf4_byte_bf16[p[j]], v = sf4_byte_bf16[q[j]];
                        v0[2 * j] = sycl::bit_cast<float>(u << 16);
                        v0[2 * j + 1] = sycl::bit_cast<float>(u & 0xffff0000u);
                        v1[2 * j] = sycl::bit_cast<float>(v << 16);
                        v1[2 * j + 1] = sycl::bit_cast<float>(v & 0xffff0000u);
                    } else {
                        v0[2 * j] = e2m1_lut(p[j] & 15);
                        v0[2 * j + 1] = e2m1_lut(p[j] >> 4);
                        v1[2 * j] = e2m1_lut(q[j] & 15);
                        v1[2 * j + 1] = e2m1_lut(q[j] >> 4);
                    }
                }
            } else {
#pragma unroll
                for (int j = 0; j < 32; j += 8) {
                    const auto p =
                        *reinterpret_cast<const sycl::vec<uint8_t, 8> *>(data + (size_t)i0 * bytes + j);
                    const auto q =
                        *reinterpret_cast<const sycl::vec<uint8_t, 8> *>(data + (size_t)i1 * bytes + j);
#pragma unroll
                    for (int t = 0; t < 8; ++t) {
                        if constexpr (MX8Only) {
                            v0[j + t] = mx8_coefficient_f32(p[t]);
                            v1[j + t] = mx8_coefficient_f32(q[t]);
                        } else {
                            v0[j + t] = e4m3_to_f32(p[t]);
                            v1[j + t] = e4m3_to_f32(q[t]);
                        }
                    }
                }
            }
            for (int r = 0; r < rows; ++r) {
                float a0 = 0.f, b0 = 0.f;
#pragma unroll
                for (int j = 0; j < width; j += 8) {
                    auto vec = *reinterpret_cast<const sycl::vec<float, 8> *>(input[r] + block * width + j);
#pragma unroll
                    for (int t = 0; t < 8; t++) {
                        a0 += v0[j + t] * vec[t];
                        b0 += v1[j + t] * vec[t];
                    }
                }
                a[r] += a0 * s0;
                b[r] += b0 * s1;
            }
        }
        for (int r = 0; r < rows; ++r) {
            const float v = sg_sum(it, a[r]), u = sg_sum(it, b[r]);
            if (!lane) {
                if constexpr (Down) {
                    out[pid[r] * 2304 + 2 * pair] = v * w[pid[r]];
                    out[pid[r] * 2304 + 2 * pair + 1] = u * w[pid[r]];
                } else
                    out[pid[r] * 1152 + pair] = situ_act(v, u);
            }
        }
    }
}
constexpr int sf4_scale_codes = 256;
constexpr int sf4_expert_scales = 2 * sf4_scale_codes;

// Form both projection tables on the device, using the same FP32 division
// and multiplication as the uncached compressed dot.
inline void k_sf4_build_scales(const InkSyExpertDev* experts, int count, float* factors) {
    Q().parallel_for(NR(count, sf4_scale_codes), [=](sycl::nd_item<1> it) {
        const int expert = it.get_group(0), code = it.get_local_id(0);
        const auto ed = experts[expert];
        if (!ed.body_ptr || ed.gu_code != FMT_SF4 || ed.dn_code != FMT_SF4) return;
        const float* h = reinterpret_cast<const float*>(ed.body_ptr);
        float* table = factors + size_t(expert) * sf4_expert_scales + code;
        const float value = e4m3_to_f32(code);
        table[0] = value * (h[0] / h[1]);
        table[sf4_scale_codes] = value * (h[2] / h[3]);
    });
}

template<bool Down>
inline void k_sf4_cached(const InkSyExpertDev* experts, const int* ids, const int* offs,
                              const int* counts, const int* order, const int* num, int limit,
                              const float* weights, const float* x, float* out, const float* factors) {
    constexpr int tiles = 1152 / 8, stripe = 8;
    const int groups = std::min(limit, 16);
    Q().parallel_for(NR2(tiles, groups, 128), [=](sycl::nd_item<2> it) INK_REQD_SG16 {
        const int active = *num;
        // Interleave eight output tiles from each active expert. Stride
        // through the complete plan even when all routed experts are unique.
        for (int i = it.get_group(0) * tiles + it.get_group(1); i < active * tiles; i += groups * tiles) {
            const int run = i / stripe, group = run % active, tile = (run / active) * stripe + i % stripe;
            const int expert = ids[group];
            const float* table = factors + size_t(expert) * sf4_expert_scales + (Down ? sf4_scale_codes : 0);
            const auto* pairs = order + offs[group];
            const int rows = counts[group];
            if (rows == 1)
                sq20_small_pairs<Down, true, true, 1>(it, experts[expert], pairs, rows, weights, x, out, table, tile);
            else
                sq20_small_pairs<Down, true, true>(it, experts[expert], pairs, rows, weights, x, out, table, tile);
        }
    });
}

template <bool Down>
inline void k_tree_sf4_packed(const InkSyExpertDev* experts, const int* ids, const int* offs,
                              const int* counts, const int* order, const int* num, int limit,
                              const float* weights, const float* x, float* out) {
    constexpr int pairs_per_group = 8;
    Q().parallel_for(NR2(1152 / pairs_per_group, limit, pairs_per_group * 16), [=](sycl::nd_item<2> it) INK_REQD_SG16 {
        const int group = it.get_group(0);
        if (group < *num)
            sq20_small_pairs<Down, true, false, 0, false, true>(it, experts[ids[group]], order + offs[group], counts[group], weights, x, out);
    });
}

template <bool Down>
inline void k_sq20_small_pairs(const InkSyExpertDev *experts, const int *exp, const int *off,
                               const int *count, const int *order, const int *num, int limit, const float *w,
                               const float *x, float *out) {
    Q().parallel_for(NR2(72, limit, 256), [=](sycl::nd_item<2> it) INK_REQD_SG16 {
        const int g = it.get_group(0);
        if (g >= *num)
            return;
        const auto ed = experts[exp[g]];
        if ((Down ? ed.dn_code : ed.gu_code) == FMT_SF4)
            sq20_small_pairs<Down, true>(it, ed, order + off[g], count[g], w, x, out);
        else
            sq20_small_pairs<Down, false>(it, ed, order + off[g], count[g], w, x, out);
    });
}

inline void sq20_small_moe(const InkSyExpertDev *experts, const int *exp, const int *off, const int *count,
                           const int *order, const int *num, int limit, const float *w, const float *x,
                           float *h, float *yp) {
    k_sq20_small_pairs<false>(experts, exp, off, count, order, num, limit, w, x, h);
    k_sq20_small_pairs<true>(experts, exp, off, count, order, num, limit, w, h, yp);
}

template <bool Down>
inline void k_sq20_small_pairs_mx8(const InkSyExpertDev* experts, const int* ids, const int* offsets,
                                    const int* counts, const int* order, const int* active, int limit,
                                    const float* weights, const float* x, float* out) {
    Q().parallel_for(NR2(72, limit, 256), [=](sycl::nd_item<2> it) INK_REQD_SG16 {
        const int group = it.get_group(0);
        if (group >= *active) return;
        sq20_small_pairs<Down, false, false, 0, true>(
            it, experts[ids[group]], order + offsets[group], counts[group], weights, x, out);
    });
}

inline void sq20_small_moe_mx8(const InkSyExpertDev* experts, const int* ids, const int* offsets,
                                const int* counts, const int* order, const int* active, int limit,
                                const float* weights, const float* x, float* hidden, float* y) {
    k_sq20_small_pairs_mx8<false>(experts, ids, offsets, counts, order, active, limit, weights, x, hidden);
    k_sq20_small_pairs_mx8<true>(experts, ids, offsets, counts, order, active, limit, weights, hidden, y);
}

// For a single resident token, top-k rank is already the pair index.

inline void k_moe_reduce8_sg16_q(sycl::queue& q, const float* y_pairs, float* moe_out, int M,
                                 const InkSyExpertDev* ex_layer = nullptr, const int* topk_idx = nullptr) {
    q.parallel_for(NR2(9, (size_t)M, 256), [=](sycl::nd_item<2> it) INK_REQD_SG16 {
        const int m = (int)it.get_group(0);
        const int d = (int)it.get_group(1) * 256 + (int)it.get_local_id(1);
        if (d >= 2304) return;
        float s = 0.0f;
        for (int k = 0; k < 8; ++k) {
            if (ex_layer && topk_idx) {
                const int e = topk_idx[m * 8 + k];
                if (e < 0 || e >= 128 || ex_layer[e].body_ptr == nullptr) continue;
            }
            s += y_pairs[((size_t)m * 8 + k) * 2304 + d];
        }
        moe_out[(size_t)m * 2304 + d] = s;
    });
}

// Consume packed GEMM outputs directly, keeping the original expert rank
// order and the separate FP32 rounding of the route weight multiplication.
inline void k_moe_reduce_packed(const float* packed, const float* small,
                                const float* weight, const int* packed_row,
                                float* out, int rows) {
    Q().parallel_for(NR2(9, rows, 256), [=](sycl::nd_item<2> it) INK_REQD_SG16 {
#pragma clang fp contract(off)
        constexpr int width = 2304;
        const int row = it.get_group(0);
        const int col = it.get_global_id(1);
        float sum = 0.0f;
#pragma unroll
        for (int rank = 0; rank < 8; ++rank) {
            const int pid = row * 8 + rank;
            const int source = packed_row[pid];
            const float value = source >= 0 ? packed[size_t(source) * width + col] * weight[pid]
                                            : small[size_t(pid) * width + col];
            sum += value;
        }
        out[size_t(row) * width + col] = sum;
    });
}

inline void k_moe_reduce_packed_vector(const float* packed, const float* small, const float* weight,
                                       const int* packed_row, float* out, int rows) {
    constexpr int width = 2304, threads = 64, values = 4;
    Q().parallel_for(NR2(width / (threads * values), rows, threads), [=](sycl::nd_item<2> it) INK_REQD_SG16 {
#pragma clang fp contract(off)
        const int row = it.get_group(0), col = it.get_global_id(1) * values;
        sycl::vec<float, values> sum(0.0f);
#pragma unroll
        for (int rank = 0; rank < 8; ++rank) {
            const int pair = row * 8 + rank, source = packed_row[pair];
            const size_t offset = size_t(source < 0 ? pair : source) * width + col;
            const auto v = *reinterpret_cast<const sycl::vec<float, values>*>((source < 0 ? small : packed) + offset);
            sum += source < 0 ? v : v * weight[pair];
        }
        *reinterpret_cast<sycl::vec<float, values>*>(out + size_t(row) * width + col) = sum;
    });
}

inline void k_moe_reduce8_sg16(const float* y_pairs, float* moe_out, int M) {
    k_moe_reduce8_sg16_q(Q(), y_pairs, moe_out, M, nullptr, nullptr);
}

}  // namespace ink_sycl
