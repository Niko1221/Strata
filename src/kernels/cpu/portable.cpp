// src/kernels/cpu/portable.cpp - CPU kernel implementations on non-x86 builds (Arm64: Apple M1-M5,
// Qualcomm Snapdragon X1/X2 Oryon, NVIDIA Grace Neoverse-V2 / DGX Spark GB10 / Vera).
//
// Provides:
//  - NEON Q8_K activation quantizer (q8k_quant_avx2 entry point on Arm64: bit-exact against quantize_row_q8_K_ref,
//    replacing ggml-cpu/arch/arm/quants.c's scalar placeholder);
//  - NEON Q8_1 activation quantizer (act_quant_q8_1: hardware FCVTAS round-half-away-from-zero);
//  - 4-token-tiled NEON / FEAT_BF16 router lookahead dot (bf16_rows_dot_multi);
//  - Fused Gate+Up and multi-token (1..8 tokens) Arm64 NEON + FEAT_DotProd (SDOT) expert kernels for:
//      * I-quants: IQ2_XXS (16), IQ2_XS (17), IQ3_XXS (18), IQ3_S (21), IQ2_S (22), IQ4_XS (23), with vectorized
//        CNT (vcnt_u8) odd-parity sign expansion eliminating 8 scalar keven_signs_q2xs table loads per 64 weights;
//      * Down-projections: IQ4_NL (20) and Q2_0 (42);
//      * K-quants: Q4_K (12), Q5_K (13), Q6_K (14), Q5_0 (6), Q5_1 (7), Q8_0 (8).
//  - Unreachable stubs for x86-only AVX/AVX-512 entry points so non-LTO Arm64 links cleanly.
#include "strata/kernels/cpu/expert.hpp"
#include "strata/kernels/cpu/expert_layout.hpp"
#include "strata/kernels/cpu/iq_avx2.hpp"
#include "strata/kernels/cpu/iq_avx512.hpp"
#include "strata/kernels/cpu/kq_avx1.hpp"
#include "strata/kernels/cpu/kq_avx2.hpp"
#include "strata/kernels/cpu/iq_avx1.hpp"
#include "strata/kernels/cpu/nvfp4_avx2.hpp"
#include "strata/kernels/cpu/nvfp4_avx512.hpp"

#ifndef GGML_COMMON_DECL_CPP
#define GGML_COMMON_DECL_CPP
#endif
#ifndef GGML_COMMON_IMPL_CPP
#define GGML_COMMON_IMPL_CPP
#endif
#include "ggml-common.h"

#include <atomic>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>

#if defined(STRATA_ARM_NEON_SIM)
#include "arm_neon_sim.hpp"
#define STRATA_HAVE_ARM_NEON 1
#define STRATA_HAVE_ARM_DOTPROD 1
#elif defined(__ARM_NEON) || defined(__ARM_NEON__) || defined(_M_ARM64)
#include <arm_neon.h>
#define STRATA_HAVE_ARM_NEON 1
#if defined(__ARM_FEATURE_DOTPROD) || defined(_M_ARM64)
#define STRATA_HAVE_ARM_DOTPROD 1
#endif
#endif

namespace strata::kernels::cpu {
#if defined(STRATA_PORTABLE_NAMESPACE)
namespace STRATA_PORTABLE_NAMESPACE {
#endif

namespace {

#if defined(_MSC_VER)
#define STRATA_ARM_INLINE __forceinline
#else
#define STRATA_ARM_INLINE __attribute__((always_inline)) inline
#endif

std::atomic<bool> oracle_q8_0{false};

[[noreturn]] void unsupported(const char* what) {
    std::fprintf(stderr,
                 "strata: %s is x86-only (AVX2/AVX-512) and this build has no portable version.\n"
                 "        On this CPU use a native GGUF pack (IQ2_XS, IQ3_S, IQ4_NL, Q4_K, ...).\n",
                 what);
    std::abort();
}

STRATA_ARM_INLINE uint16_t u16(const void* p) { uint16_t v; std::memcpy(&v, p, 2); return v; }
STRATA_ARM_INLINE uint32_t u32(const void* p) { uint32_t v; std::memcpy(&v, p, 4); return v; }
STRATA_ARM_INLINE uint64_t u64(const void* p) { uint64_t v; std::memcpy(&v, p, 8); return v; }

STRATA_ARM_INLINE float bf16f(uint16_t h) {
    const uint32_t b = (uint32_t) h << 16;
    float f;
    std::memcpy(&f, &b, 4);
    return f;
}

STRATA_ARM_INLINE float h2f(uint16_t h) {
#if defined(__ARM_FEATURE_FP16_SCALAR_ARITHMETIC) && !defined(STRATA_ARM_NEON_SIM)
    __fp16 fh;
    std::memcpy(&fh, &h, 2);
    return (float) fh;
#else
    const uint32_t w = (uint32_t) h << 16;
    const uint32_t sign = w & 0x80000000u;
    const uint32_t two_w = w + w;
    const uint32_t exp_offset = 0xE0u << 23;
    const float exp_scale = 0x1.0p-112f;
    const uint32_t norm_bits = (two_w >> 4) + exp_offset;
    float norm_val;
    std::memcpy(&norm_val, &norm_bits, 4);
    norm_val *= exp_scale;
    const uint32_t magic_mask = 126u << 23;
    const float magic_bias = 0.5f;
    const uint32_t denorm_bits = (two_w >> 17) | magic_mask;
    float denorm_val;
    std::memcpy(&denorm_val, &denorm_bits, 4);
    denorm_val -= magic_bias;
    const uint32_t denorm_cutoff = 1u << 27;
    uint32_t res_bits;
    if (two_w < denorm_cutoff) {
        std::memcpy(&res_bits, &denorm_val, 4);
    } else {
        std::memcpy(&res_bits, &norm_val, 4);
    }
    res_bits |= sign;
    float res;
    std::memcpy(&res, &res_bits, 4);
    return res;
#endif
}

}  // namespace

void act_quant_q8_1(const float* x, int n, ActQ& a) {
    if (oracle_q8_0.load(std::memory_order_relaxed)) unsupported("the Q8_0 oracle activation quantizer");
    a.nchunks = n / QKA;
    for (int k = 0; k < a.nchunks; ++k) {
        const float* xb = x + k * QKA;
        int8_t* q = a.q + k * QKA;
#if defined(STRATA_HAVE_ARM_NEON)
        float32x4_t v0 = vld1q_f32(xb + 0),  v1 = vld1q_f32(xb + 4);
        float32x4_t v2 = vld1q_f32(xb + 8),  v3 = vld1q_f32(xb + 12);
        float32x4_t v4 = vld1q_f32(xb + 16), v5 = vld1q_f32(xb + 20);
        float32x4_t v6 = vld1q_f32(xb + 24), v7 = vld1q_f32(xb + 28);
        float32x4_t m0 = vmaxnmq_f32(vabsq_f32(v0), vabsq_f32(v1));
        float32x4_t m1 = vmaxnmq_f32(vabsq_f32(v2), vabsq_f32(v3));
        float32x4_t m2 = vmaxnmq_f32(vabsq_f32(v4), vabsq_f32(v5));
        float32x4_t m3 = vmaxnmq_f32(vabsq_f32(v6), vabsq_f32(v7));
        const float amax = vmaxnmvq_f32(vmaxnmq_f32(vmaxnmq_f32(m0, m1), vmaxnmq_f32(m2, m3)));
        const float s = amax > 0.f ? amax / 127.f : 0.f;
        const float inv = s > 0.f ? 1.f / s : 0.f;
        const int32x4_t cmin = vdupq_n_s32(-127), cmax = vdupq_n_s32(127);
        auto q4 = [&](float32x4_t v) {
            return vminq_s32(vmaxq_s32(vcvtaq_s32_f32(vmulq_n_f32(v, inv)), cmin), cmax);
        };
        const int32x4_t i0 = q4(v0), i1 = q4(v1), i2 = q4(v2), i3 = q4(v3);
        const int32x4_t i4 = q4(v4), i5 = q4(v5), i6 = q4(v6), i7 = q4(v7);
        vst1q_s8(q + 0,  vcombine_s8(vqmovn_s16(vcombine_s16(vqmovn_s32(i0), vqmovn_s32(i1))),
                                     vqmovn_s16(vcombine_s16(vqmovn_s32(i2), vqmovn_s32(i3)))));
        vst1q_s8(q + 16, vcombine_s8(vqmovn_s16(vcombine_s16(vqmovn_s32(i4), vqmovn_s32(i5))),
                                     vqmovn_s16(vcombine_s16(vqmovn_s32(i6), vqmovn_s32(i7)))));
        const int32x4_t s0123 = vaddq_s32(vaddq_s32(i0, i1), vaddq_s32(i2, i3));
        const int32x4_t s4567 = vaddq_s32(vaddq_s32(i4, i5), vaddq_s32(i6, i7));
        const int32_t sum = vaddvq_s32(vaddq_s32(s0123, s4567));
        a.scale[k] = s;
        a.sum[k] = sum;
        a.hx[k] = s * (float) sum;
#else
        float amax = 0.f;
        for (int j = 0; j < QKA; ++j) amax = std::fmax(amax, std::fabs(xb[j]));
        const float s = amax > 0.f ? amax / 127.f : 0.f;
        const float inv = s > 0.f ? 1.f / s : 0.f;
        int32_t sum = 0;
        for (int j = 0; j < QKA; ++j) {
            const float t = xb[j] * inv;
            const float r = t + (t >= 0.f ? 0.5f : -0.5f);
            int v = (int) r;
            v = v < -127 ? -127 : (v > 127 ? 127 : v);
            q[j] = (int8_t) v;
            sum += v;
        }
        a.scale[k] = s;
        a.sum[k] = sum;
        a.hx[k] = s * (float) sum;
#endif
    }
}

void act_quant_q8_1_avx2(const float* x, int n, ActQ& a) { act_quant_q8_1(x, n, a); }

// Bit-exact NEON Q8_K activation quantizer (matches quantize_row_q8_K_ref byte-for-byte, including NaN/Inf/ties).
void q8k_quant_avx2(const float* x, void* vy, int64_t k) {
    block_q8_K* y = (block_q8_K*) vy;
    const int64_t nb = k / QK_K;
#if defined(STRATA_HAVE_ARM_NEON)
    const float32x4_t magic = vdupq_n_f32(12582912.f);
    const int32x4_t mant = vdupq_n_s32(0x007fffff), off = vdupq_n_s32(0x00400000), c127 = vdupq_n_s32(127);
    for (int64_t i = 0; i < nb; ++i, x += QK_K) {
        float32x4_t m0 = vdupq_n_f32(0.f), m1 = vdupq_n_f32(0.f), m2 = vdupq_n_f32(0.f), m3 = vdupq_n_f32(0.f);
        for (int j = 0; j < QK_K; j += 16) {
            m0 = vmaxnmq_f32(vabsq_f32(vld1q_f32(x + j + 0)),  m0);
            m1 = vmaxnmq_f32(vabsq_f32(vld1q_f32(x + j + 4)),  m1);
            m2 = vmaxnmq_f32(vabsq_f32(vld1q_f32(x + j + 8)),  m2);
            m3 = vmaxnmq_f32(vabsq_f32(vld1q_f32(x + j + 12)), m3);
        }
        const float amax = vmaxnmvq_f32(vmaxnmq_f32(vmaxnmq_f32(m0, m1), vmaxnmq_f32(m2, m3)));
        if (!amax) {
            y[i].d = 0.f;
            std::memset(y[i].qs, 0, QK_K);
            continue;
        }
        float mx = 0.f;
        const float32x4_t va = vdupq_n_f32(amax);
        for (int j = 0; j < QK_K; j += 4) {
            const uint32x4_t cmp = vceqq_f32(vabsq_f32(vld1q_f32(x + j)), va);
            if (vmaxvq_u32(cmp) != 0) {
                for (int l = 0; l < 4; ++l) {
                    if (std::fabs(x[j + l]) == amax) {
                        mx = x[j + l];
                        break;
                    }
                }
                break;
            }
        }
        const float iscale = -127.f / mx;
        for (int j = 0; j < QK_K; j += 16) {
            auto step4 = [&](const float* p) {
                float32x4_t prod = vmulq_n_f32(vld1q_f32(p), iscale);
#if defined(__GNUC__) && !defined(__clang__) && !defined(STRATA_ARM_NEON_SIM)
                __asm__("" : "+w"(prod));
#endif
                const float32x4_t t = vaddq_f32(prod, magic);
                const int32x4_t v = vminq_s32(vsubq_s32(vandq_s32(vreinterpretq_s32_f32(t), mant), off), c127);
                return vshrq_n_s32(vshlq_n_s32(v, 24), 24);
            };
            const int32x4_t v0 = step4(x + j + 0),  v1 = step4(x + j + 4);
            const int32x4_t v2 = step4(x + j + 8),  v3 = step4(x + j + 12);
            vst1q_s8(y[i].qs + j, vcombine_s8(vmovn_s16(vcombine_s16(vmovn_s32(v0), vmovn_s32(v1))),
                                              vmovn_s16(vcombine_s16(vmovn_s32(v2), vmovn_s32(v3)))));
            y[i].bsums[j / 16] = (int16_t) vaddvq_s32(vaddq_s32(vaddq_s32(v0, v1), vaddq_s32(v2, v3)));
        }
        y[i].d = 1.f / iscale;
    }
#else
    for (int64_t i = 0; i < nb; ++i, x += QK_K) {
        float amax = 0.f, max = 0.f;
        for (int j = 0; j < QK_K; ++j) {
            float ax = std::fabs(x[j]);
            if (ax > amax) { amax = ax; max = x[j]; }
        }
        if (!amax) {
            y[i].d = 0.f;
            std::memset(y[i].qs, 0, QK_K);
            continue;
        }
        const float iscale = -127.f / max;
        for (int j = 0; j < QK_K; ++j) {
            int v = (int) std::nearbyint(iscale * x[j]);
            y[i].qs[j] = (int8_t) (v < 127 ? v : 127);
        }
        for (int j = 0; j < QK_K / 16; ++j) {
            int sum = 0;
            for (int ii = 0; ii < 16; ++ii) sum += y[i].qs[j * 16 + ii];
            y[i].bsums[j] = (int16_t) sum;
        }
        y[i].d = 1.f / iscale;
    }
#endif
}

#if defined(STRATA_HAVE_ARM_DOTPROD)
namespace {

// Vectorized Arm NEON CNT (vcnt_u8) odd-parity sign generator for IQ2_XXS, IQ2_XS, and IQ3_XXS:
// Given 8 7-bit sign indices s7[0..7], bit 7 of byte k is (popcount(s7[k]) & 1), and byte b (0..7) of sign group k
// is -1 if bit b is set, else +1 (matching keven_signs_q2xs[128] without any L1 table loads).
STRATA_ARM_INLINE void arm_signs4_from_idx7(const uint8_t s7[8], int8x16_t out[4]) {
    static const uint8_t kRep01[16] = {0,0,0,0,0,0,0,0, 1,1,1,1,1,1,1,1};
    static const uint8_t kRep23[16] = {2,2,2,2,2,2,2,2, 3,3,3,3,3,3,3,3};
    static const uint8_t kRep45[16] = {4,4,4,4,4,4,4,4, 5,5,5,5,5,5,5,5};
    static const uint8_t kRep67[16] = {6,6,6,6,6,6,6,6, 7,7,7,7,7,7,7,7};
    static const uint8_t kBits[16]  = {1,2,4,8,16,32,64,128, 1,2,4,8,16,32,64,128};
    const uint8x8_t v7 = vld1_u8(s7);
    const uint8x8_t par = vshl_n_u8(vand_u8(vcnt_u8(v7), vdup_n_u8(1)), 7);
    const uint8x8_t sb = vorr_u8(v7, par);
    const uint8x16_t sb16 = vcombine_u8(sb, sb);
    const uint8x16_t bm = vld1q_u8(kBits);
    const int8x16_t one = vdupq_n_s8(1);
    out[0] = vorrq_s8(vreinterpretq_s8_u8(vtstq_u8(vqtbl1q_u8(sb16, vld1q_u8(kRep01)), bm)), one);
    out[1] = vorrq_s8(vreinterpretq_s8_u8(vtstq_u8(vqtbl1q_u8(sb16, vld1q_u8(kRep23)), bm)), one);
    out[2] = vorrq_s8(vreinterpretq_s8_u8(vtstq_u8(vqtbl1q_u8(sb16, vld1q_u8(kRep45)), bm)), one);
    out[3] = vorrq_s8(vreinterpretq_s8_u8(vtstq_u8(vqtbl1q_u8(sb16, vld1q_u8(kRep67)), bm)), one);
}

STRATA_ARM_INLINE void arm_signs4_from_raw64(uint64_t raw, int8x16_t out[4]) {
    static const uint8_t kRep01[16] = {0,0,0,0,0,0,0,0, 1,1,1,1,1,1,1,1};
    static const uint8_t kRep23[16] = {2,2,2,2,2,2,2,2, 3,3,3,3,3,3,3,3};
    static const uint8_t kRep45[16] = {4,4,4,4,4,4,4,4, 5,5,5,5,5,5,5,5};
    static const uint8_t kRep67[16] = {6,6,6,6,6,6,6,6, 7,7,7,7,7,7,7,7};
    static const uint8_t kBits[16]  = {1,2,4,8,16,32,64,128, 1,2,4,8,16,32,64,128};
    uint8_t sb_arr[8];
    std::memcpy(sb_arr, &raw, 8);
    const uint8x8_t sb = vld1_u8(sb_arr);
    const uint8x16_t sb16 = vcombine_u8(sb, sb);
    const uint8x16_t bm = vld1q_u8(kBits);
    const int8x16_t one = vdupq_n_s8(1);
    out[0] = vorrq_s8(vreinterpretq_s8_u8(vtstq_u8(vqtbl1q_u8(sb16, vld1q_u8(kRep01)), bm)), one);
    out[1] = vorrq_s8(vreinterpretq_s8_u8(vtstq_u8(vqtbl1q_u8(sb16, vld1q_u8(kRep23)), bm)), one);
    out[2] = vorrq_s8(vreinterpretq_s8_u8(vtstq_u8(vqtbl1q_u8(sb16, vld1q_u8(kRep45)), bm)), one);
    out[3] = vorrq_s8(vreinterpretq_s8_u8(vtstq_u8(vqtbl1q_u8(sb16, vld1q_u8(kRep67)), bm)), one);
}

STRATA_ARM_INLINE int8x16_t load_grid64_pair(const uint64_t* grid, uint32_t i0, uint32_t i1) {
    return vcombine_s8(vld1_s8((const int8_t*) &grid[i0]), vld1_s8((const int8_t*) &grid[i1]));
}

STRATA_ARM_INLINE int8x16_t load_grid32_quad(const uint32_t* grid, uint32_t i0, uint32_t i1, uint32_t i2, uint32_t i3) {
    uint32_t tmp[4] = {grid[i0], grid[i1], grid[i2], grid[i3]};
    int8x16_t r;
    std::memcpy(&r, tmp, 16);
    return r;
}

template <int TY> struct ArmIqFmt;

template <> struct ArmIqFmt<16> {   // IQ2_XXS
    static constexpr int bytes = 66;
    static constexpr float K = 0.125f;
    static STRATA_ARM_INLINE void decode(const uint8_t* b, int j, int8x16_t w[4], int32_t sc[4]) {
        const uint8_t* q = b + 2 + 16 * j;
        const uint32_t a0 = u32(q + 0), a1 = u32(q + 4), a2 = u32(q + 8), a3 = u32(q + 12);
        const uint8_t s7[8] = {
            (uint8_t) ((a1 >> 0) & 127), (uint8_t) ((a1 >> 7) & 127), (uint8_t) ((a1 >> 14) & 127), (uint8_t) ((a1 >> 21) & 127),
            (uint8_t) ((a3 >> 0) & 127), (uint8_t) ((a3 >> 7) & 127), (uint8_t) ((a3 >> 14) & 127), (uint8_t) ((a3 >> 21) & 127),
        };
        int8x16_t sgn[4];
        arm_signs4_from_idx7(s7, sgn);
        w[0] = vmulq_s8(load_grid64_pair(iq2xxs_grid, q[0], q[1]), sgn[0]);
        w[1] = vmulq_s8(load_grid64_pair(iq2xxs_grid, q[2], q[3]), sgn[1]);
        w[2] = vmulq_s8(load_grid64_pair(iq2xxs_grid, q[8], q[9]), sgn[2]);
        w[3] = vmulq_s8(load_grid64_pair(iq2xxs_grid, q[10], q[11]), sgn[3]);
        const int32_t s0 = 2 * (int32_t) (a1 >> 28) + 1;
        const int32_t s1 = 2 * (int32_t) (a3 >> 28) + 1;
        sc[0] = s0; sc[1] = s0; sc[2] = s1; sc[3] = s1;
        (void) a0; (void) a2;
    }
};

template <> struct ArmIqFmt<17> {   // IQ2_XS
    static constexpr int bytes = 74;
    static constexpr float K = 0.125f;
    static STRATA_ARM_INLINE void decode(const uint8_t* b, int j, int8x16_t w[4], int32_t sc[4]) {
        const uint8_t* q = b + 2 + 16 * j;
        uint16_t q2[8];
        std::memcpy(q2, q, 16);
        const uint8_t s7[8] = {
            (uint8_t) (q2[0] >> 9), (uint8_t) (q2[1] >> 9), (uint8_t) (q2[2] >> 9), (uint8_t) (q2[3] >> 9),
            (uint8_t) (q2[4] >> 9), (uint8_t) (q2[5] >> 9), (uint8_t) (q2[6] >> 9), (uint8_t) (q2[7] >> 9),
        };
        int8x16_t sgn[4];
        arm_signs4_from_idx7(s7, sgn);
        w[0] = vmulq_s8(load_grid64_pair(iq2xs_grid, q2[0] & 511, q2[1] & 511), sgn[0]);
        w[1] = vmulq_s8(load_grid64_pair(iq2xs_grid, q2[2] & 511, q2[3] & 511), sgn[1]);
        w[2] = vmulq_s8(load_grid64_pair(iq2xs_grid, q2[4] & 511, q2[5] & 511), sgn[2]);
        w[3] = vmulq_s8(load_grid64_pair(iq2xs_grid, q2[6] & 511, q2[7] & 511), sgn[3]);
        const uint8_t sb0 = b[66 + 2 * j], sb1 = b[66 + 2 * j + 1];
        sc[0] = 2 * (sb0 & 15) + 1;
        sc[1] = 2 * (sb0 >> 4) + 1;
        sc[2] = 2 * (sb1 & 15) + 1;
        sc[3] = 2 * (sb1 >> 4) + 1;
    }
};

template <> struct ArmIqFmt<18> {   // IQ3_XXS
    static constexpr int bytes = 98;
    static constexpr float K = 0.25f;
    static STRATA_ARM_INLINE void decode(const uint8_t* b, int j, int8x16_t w[4], int32_t sc[4]) {
        const uint8_t* q = b + 2 + 16 * j;
        const uint32_t a0 = u32(b + 66 + 8 * j), a1 = u32(b + 70 + 8 * j);
        const uint8_t s7[8] = {
            (uint8_t) ((a0 >> 0) & 127), (uint8_t) ((a0 >> 7) & 127), (uint8_t) ((a0 >> 14) & 127), (uint8_t) ((a0 >> 21) & 127),
            (uint8_t) ((a1 >> 0) & 127), (uint8_t) ((a1 >> 7) & 127), (uint8_t) ((a1 >> 14) & 127), (uint8_t) ((a1 >> 21) & 127),
        };
        int8x16_t sgn[4];
        arm_signs4_from_idx7(s7, sgn);
        w[0] = vmulq_s8(load_grid32_quad(iq3xxs_grid, q[0],  q[1],  q[2],  q[3]),  sgn[0]);
        w[1] = vmulq_s8(load_grid32_quad(iq3xxs_grid, q[4],  q[5],  q[6],  q[7]),  sgn[1]);
        w[2] = vmulq_s8(load_grid32_quad(iq3xxs_grid, q[8],  q[9],  q[10], q[11]), sgn[2]);
        w[3] = vmulq_s8(load_grid32_quad(iq3xxs_grid, q[12], q[13], q[14], q[15]), sgn[3]);
        const int32_t s0 = 2 * (int32_t) (a0 >> 28) + 1;
        const int32_t s1 = 2 * (int32_t) (a1 >> 28) + 1;
        sc[0] = s0; sc[1] = s0; sc[2] = s1; sc[3] = s1;
    }
};

template <> struct ArmIqFmt<21> {   // IQ3_S
    static constexpr int bytes = 110;
    static constexpr float K = 1.0f;
    static STRATA_ARM_INLINE void decode(const uint8_t* b, int j, int8x16_t w[4], int32_t sc[4]) {
        const uint8_t* q = b + 2 + 16 * j;
        const uint32_t h0 = b[66 + 2 * j], h1 = b[66 + 2 * j + 1];
        int8x16_t sgn[4];
        arm_signs4_from_raw64(u64(b + 74 + 8 * j), sgn);
        w[0] = vmulq_s8(load_grid32_quad(iq3s_grid, q[0]  | ((h0 << 8) & 0x100), q[1]  | ((h0 << 7) & 0x100),
                                                    q[2]  | ((h0 << 6) & 0x100), q[3]  | ((h0 << 5) & 0x100)), sgn[0]);
        w[1] = vmulq_s8(load_grid32_quad(iq3s_grid, q[4]  | ((h0 << 4) & 0x100), q[5]  | ((h0 << 3) & 0x100),
                                                    q[6]  | ((h0 << 2) & 0x100), q[7]  | ((h0 << 1) & 0x100)), sgn[1]);
        w[2] = vmulq_s8(load_grid32_quad(iq3s_grid, q[8]  | ((h1 << 8) & 0x100), q[9]  | ((h1 << 7) & 0x100),
                                                    q[10] | ((h1 << 6) & 0x100), q[11] | ((h1 << 5) & 0x100)), sgn[2]);
        w[3] = vmulq_s8(load_grid32_quad(iq3s_grid, q[12] | ((h1 << 4) & 0x100), q[13] | ((h1 << 3) & 0x100),
                                                    q[14] | ((h1 << 2) & 0x100), q[15] | ((h1 << 1) & 0x100)), sgn[3]);
        const uint8_t s = b[106 + j];
        const int32_t sa = 2 * (s & 15) + 1, sb = 2 * (s >> 4) + 1;
        sc[0] = sa; sc[1] = sa; sc[2] = sb; sc[3] = sb;
    }
};

template <> struct ArmIqFmt<22> {   // IQ2_S
    static constexpr int bytes = 82;
    static constexpr float K = 0.125f;
    static STRATA_ARM_INLINE void decode(const uint8_t* b, int j, int8x16_t w[4], int32_t sc[4]) {
        const uint8_t* qs = b + 2 + 8 * j;
        const uint32_t h0 = b[66 + 2 * j], h1 = b[66 + 2 * j + 1];
        int8x16_t sgn[4];
        arm_signs4_from_raw64(u64(b + 34 + 8 * j), sgn);
        w[0] = vmulq_s8(load_grid64_pair(iq2s_grid, qs[0] | ((h0 << 8) & 0x300), qs[1] | ((h0 << 6) & 0x300)), sgn[0]);
        w[1] = vmulq_s8(load_grid64_pair(iq2s_grid, qs[2] | ((h0 << 4) & 0x300), qs[3] | ((h0 << 2) & 0x300)), sgn[1]);
        w[2] = vmulq_s8(load_grid64_pair(iq2s_grid, qs[4] | ((h1 << 8) & 0x300), qs[5] | ((h1 << 6) & 0x300)), sgn[2]);
        w[3] = vmulq_s8(load_grid64_pair(iq2s_grid, qs[6] | ((h1 << 4) & 0x300), qs[7] | ((h1 << 2) & 0x300)), sgn[3]);
        const uint8_t s0 = b[74 + 2 * j], s1 = b[74 + 2 * j + 1];
        sc[0] = 2 * (s0 & 15) + 1;
        sc[1] = 2 * (s0 >> 4) + 1;
        sc[2] = 2 * (s1 & 15) + 1;
        sc[3] = 2 * (s1 >> 4) + 1;
    }
};

template <> struct ArmIqFmt<23> {   // IQ4_XS
    static constexpr int bytes = 136;
    static constexpr float K = 1.0f;
    static STRATA_ARM_INLINE void decode(const uint8_t* b, int j, int8x16_t w[4], int32_t sc[4]) {
        const int8x16_t lut = vld1q_s8(kvalues_iq4nl);
        const uint8x16_t m4 = vdupq_n_u8(0x0f);
        const uint8x16_t b0 = vld1q_u8(b + 8 + 32 * j);
        const uint8x16_t b1 = vld1q_u8(b + 24 + 32 * j);
        w[0] = vqtbl1q_s8(lut, vandq_u8(b0, m4));
        w[1] = vqtbl1q_s8(lut, vshrq_n_u8(b0, 4));
        w[2] = vqtbl1q_s8(lut, vandq_u8(b1, m4));
        w[3] = vqtbl1q_s8(lut, vshrq_n_u8(b1, 4));
        const uint8_t sl = b[4 + j];
        const uint16_t sh = (uint16_t) (u16(b + 2) >> (4 * j));
        const int32_t ls0 = ((sl & 0x0f) | ((sh << 4) & 0x30)) - 32;
        const int32_t ls1 = ((sl >> 4)   | ((sh << 2) & 0x30)) - 32;
        sc[0] = ls0; sc[1] = ls0; sc[2] = ls1; sc[3] = ls1;
    }
};

template <int TY, int NT>
STRATA_ARM_INLINE void arm_iq_gu_dot(const uint8_t* gr, const uint8_t* ur, int nb, const block_q8_K* const* y,
                                     float* g_out, float* u_out) {
    float sum_g[NT] = {}, sum_u[NT] = {};
    const int32x4_t zero = vdupq_n_s32(0);
    for (int i = 0; i < nb; ++i) {
        const uint8_t* gb = gr + (size_t) i * ArmIqFmt<TY>::bytes;
        const uint8_t* ub = ur + (size_t) i * ArmIqFmt<TY>::bytes;
        const float dxg = h2f(u16(gb)), dxu = h2f(u16(ub));
        int32_t bi_g[NT] = {}, bi_u[NT] = {};
        for (int j = 0; j < 4; ++j) {
            int8x16_t wg[4], wu[4];
            int32_t scg[4], scu[4];
            ArmIqFmt<TY>::decode(gb, j, wg, scg);
            ArmIqFmt<TY>::decode(ub, j, wu, scu);
            for (int t = 0; t < NT; ++t) {
                const int8_t* q8 = y[t][i].qs + 64 * j;
                const int8x16_t y0 = vld1q_s8(q8 + 0),  y1 = vld1q_s8(q8 + 16);
                const int8x16_t y2 = vld1q_s8(q8 + 32), y3 = vld1q_s8(q8 + 48);
                bi_g[t] += vaddvq_s32(vdotq_s32(zero, wg[0], y0)) * scg[0] +
                           vaddvq_s32(vdotq_s32(zero, wg[1], y1)) * scg[1] +
                           vaddvq_s32(vdotq_s32(zero, wg[2], y2)) * scg[2] +
                           vaddvq_s32(vdotq_s32(zero, wg[3], y3)) * scg[3];
                bi_u[t] += vaddvq_s32(vdotq_s32(zero, wu[0], y0)) * scu[0] +
                           vaddvq_s32(vdotq_s32(zero, wu[1], y1)) * scu[1] +
                           vaddvq_s32(vdotq_s32(zero, wu[2], y2)) * scu[2] +
                           vaddvq_s32(vdotq_s32(zero, wu[3], y3)) * scu[3];
            }
        }
        for (int t = 0; t < NT; ++t) {
            const float yd = y[t][i].d;
            sum_g[t] += (dxg * yd) * (float) bi_g[t];
            sum_u[t] += (dxu * yd) * (float) bi_u[t];
        }
    }
    for (int t = 0; t < NT; ++t) {
        g_out[t] = ArmIqFmt<TY>::K * sum_g[t];
        u_out[t] = ArmIqFmt<TY>::K * sum_u[t];
    }
}

template <int TY, int NT>
STRATA_ARM_INLINE void arm_iq_row_dot(const uint8_t* row, int nb, const block_q8_K* const* y, float* res) {
    float sumf[NT] = {};
    const int32x4_t zero = vdupq_n_s32(0);
    for (int i = 0; i < nb; ++i) {
        const uint8_t* blk = row + (size_t) i * ArmIqFmt<TY>::bytes;
        const float dx = h2f(u16(blk));
        int32_t bi[NT] = {};
        for (int j = 0; j < 4; ++j) {
            int8x16_t w[4];
            int32_t sc[4];
            ArmIqFmt<TY>::decode(blk, j, w, sc);
            for (int t = 0; t < NT; ++t) {
                const int8_t* q8 = y[t][i].qs + 64 * j;
                bi[t] += vaddvq_s32(vdotq_s32(zero, w[0], vld1q_s8(q8 + 0)))  * sc[0] +
                         vaddvq_s32(vdotq_s32(zero, w[1], vld1q_s8(q8 + 16))) * sc[1] +
                         vaddvq_s32(vdotq_s32(zero, w[2], vld1q_s8(q8 + 32))) * sc[2] +
                         vaddvq_s32(vdotq_s32(zero, w[3], vld1q_s8(q8 + 48))) * sc[3];
            }
        }
        for (int t = 0; t < NT; ++t) sumf[t] += (dx * y[t][i].d) * (float) bi[t];
    }
    for (int t = 0; t < NT; ++t) res[t] = ArmIqFmt<TY>::K * sumf[t];
}

template <int TY, int NT>
void arm_gu_rows_nt(const uint8_t* blob, size_t gu_row, size_t up_off, int n, const void* const* act,
                    float* const* ff, int r0, int r1) {
    const block_q8_K* y[NT];
    for (int t = 0; t < NT; ++t) y[t] = (const block_q8_K*) act[t];
    const int nb = n / QK_K;
    float g[NT], u[NT];
    for (int r = r0; r < r1; ++r) {
        arm_iq_gu_dot<TY, NT>(blob + (size_t) r * gu_row, blob + up_off + (size_t) r * gu_row, nb, y, g, u);
        for (int t = 0; t < NT; ++t) ff[t][r] = (g[t] / (1.f + std::exp(-g[t]))) * u[t];
    }
}

template <int TY>
void arm_gu_rows_dispatch(int nt, const uint8_t* blob, size_t gu_row, size_t up_off, int n, const void* const* act,
                          float* const* ff, int r0, int r1) {
    for (int t0 = 0; t0 < nt; t0 += 4) {
        const int m = nt - t0 < 4 ? nt - t0 : 4;
        switch (m) {
            case 1: arm_gu_rows_nt<TY, 1>(blob, gu_row, up_off, n, act + t0, ff + t0, r0, r1); break;
            case 2: arm_gu_rows_nt<TY, 2>(blob, gu_row, up_off, n, act + t0, ff + t0, r0, r1); break;
            case 3: arm_gu_rows_nt<TY, 3>(blob, gu_row, up_off, n, act + t0, ff + t0, r0, r1); break;
            default: arm_gu_rows_nt<TY, 4>(blob, gu_row, up_off, n, act + t0, ff + t0, r0, r1); break;
        }
    }
}

template <int TY, int NT>
void arm_dot_rows_nt(const uint8_t* w, size_t row_bytes, int n, const void* const* act, float* const* out, int r0, int r1) {
    const block_q8_K* y[NT];
    for (int t = 0; t < NT; ++t) y[t] = (const block_q8_K*) act[t];
    const int nb = n / QK_K;
    float res[NT];
    for (int r = r0; r < r1; ++r) {
        arm_iq_row_dot<TY, NT>(w + (size_t) r * row_bytes, nb, y, res);
        for (int t = 0; t < NT; ++t) out[t][r] = res[t];
    }
}

template <int TY>
void arm_dot_rows_dispatch(int nt, const uint8_t* w, size_t row_bytes, int n, const void* const* act,
                           float* const* out, int r0, int r1) {
    for (int t0 = 0; t0 < nt; t0 += 4) {
        const int m = nt - t0 < 4 ? nt - t0 : 4;
        switch (m) {
            case 1: arm_dot_rows_nt<TY, 1>(w, row_bytes, n, act + t0, out + t0, r0, r1); break;
            case 2: arm_dot_rows_nt<TY, 2>(w, row_bytes, n, act + t0, out + t0, r0, r1); break;
            case 3: arm_dot_rows_nt<TY, 3>(w, row_bytes, n, act + t0, out + t0, r0, r1); break;
            default: arm_dot_rows_nt<TY, 4>(w, row_bytes, n, act + t0, out + t0, r0, r1); break;
        }
    }
}

// IQ4_NL (20) multi-token down rows on Arm64 NEON + SDOT
template <int NT>
void arm_iq4nl_rows_nt(const uint8_t* w, size_t row_bytes, int n, const void* const* hq, float* const* out, int r0, int r1) {
    const block_q8_0* y[NT];
    for (int t = 0; t < NT; ++t) y[t] = (const block_q8_0*) hq[t];
    const int nb = n / QK4_NL;
    const int8x16_t values = vld1q_s8(kvalues_iq4nl);
    const uint8x16_t m4b = vdupq_n_u8(0x0f);
    const int32x4_t zero = vdupq_n_s32(0);
    for (int r = r0; r < r1; ++r) {
        const block_iq4_nl* x = (const block_iq4_nl*) (w + (size_t) r * row_bytes);
        float sumf[NT] = {};
        int ib = 0;
        for (; ib + 1 < nb; ib += 2) {
            const uint8x16_t b0 = vld1q_u8(x[ib + 0].qs);
            const uint8x16_t b1 = vld1q_u8(x[ib + 1].qs);
            const int8x16_t q0l = vqtbl1q_s8(values, vandq_u8(b0, m4b));
            const int8x16_t q0h = vqtbl1q_s8(values, vshrq_n_u8(b0, 4));
            const int8x16_t q1l = vqtbl1q_s8(values, vandq_u8(b1, m4b));
            const int8x16_t q1h = vqtbl1q_s8(values, vshrq_n_u8(b1, 4));
            const float dx0 = h2f(x[ib + 0].d), dx1 = h2f(x[ib + 1].d);
            for (int t = 0; t < NT; ++t) {
                const int32x4_t p0 = vdotq_s32(vdotq_s32(zero, q0l, vld1q_s8(y[t][ib + 0].qs)),
                                               q0h, vld1q_s8(y[t][ib + 0].qs + 16));
                const int32x4_t p1 = vdotq_s32(vdotq_s32(zero, q1l, vld1q_s8(y[t][ib + 1].qs)),
                                               q1h, vld1q_s8(y[t][ib + 1].qs + 16));
                sumf[t] += dx0 * h2f(y[t][ib + 0].d) * (float) vaddvq_s32(p0) +
                           dx1 * h2f(y[t][ib + 1].d) * (float) vaddvq_s32(p1);
            }
        }
        for (; ib < nb; ++ib) {
            const float dx = h2f(x[ib].d);
            for (int t = 0; t < NT; ++t) {
                int s0 = 0, s1 = 0;
                for (int j = 0; j < QK4_NL / 2; ++j) {
                    s0 += y[t][ib].qs[j] * kvalues_iq4nl[x[ib].qs[j] & 0xf];
                    s1 += y[t][ib].qs[j + QK4_NL / 2] * kvalues_iq4nl[x[ib].qs[j] >> 4];
                }
                sumf[t] += (dx * h2f(y[t][ib].d)) * (float) (s0 + s1);
            }
        }
        for (int t = 0; t < NT; ++t) out[t][r] = sumf[t];
    }
}

// Q2_0 (42) multi-token down rows on Arm64 NEON + SDOT (matches ggml_vec_dot_q2_0_q8_0 on Arm64)
template <int NT>
void arm_q2_rows_nt(const uint8_t* w, size_t row_bytes, int nblocks, const void* const* hq, float* const* out, int r0, int r1) {
    static const uint8_t tbl_idx_lo[16] = {0,0,0,0, 1,1,1,1, 2,2,2,2, 3,3,3,3};
    static const uint8_t tbl_idx_hi[16] = {4,4,4,4, 5,5,5,5, 6,6,6,6, 7,7,7,7};
    static const int8_t  shift_vals[16] = {0,-2,-4,-6, 0,-2,-4,-6, 0,-2,-4,-6, 0,-2,-4,-6};
    const uint8x16_t idx_lo = vld1q_u8(tbl_idx_lo), idx_hi = vld1q_u8(tbl_idx_hi);
    const int8x16_t  shifts = vld1q_s8(shift_vals), one = vdupq_n_s8(1);
    const uint8x16_t mask2  = vdupq_n_u8(0x03);
    const int32x4_t  zero   = vdupq_n_s32(0);
    const block_q8_0* y[NT];
    for (int t = 0; t < NT; ++t) y[t] = (const block_q8_0*) hq[t];
    for (int r = r0; r < r1; ++r) {
        const uint8_t* row = w + (size_t) r * row_bytes;
        float32x4_t sumv[NT];
        for (int t = 0; t < NT; ++t) sumv[t] = vdupq_n_f32(0.f);
        for (int i = 0; i < nblocks; ++i) {
            const uint8_t* blk = row + (size_t) i * 18;
            const float d0 = h2f(u16(blk));
            for (int k = 0; k < 2; ++k) {
                const uint8x8_t raw = vld1_u8(blk + 2 + k * 8);
                const uint8x16_t raw16 = vcombine_u8(raw, raw);
                const int8x16_t qv0 = vsubq_s8(vreinterpretq_s8_u8(vandq_u8(vshlq_u8(vqtbl1q_u8(raw16, idx_lo), shifts), mask2)), one);
                const int8x16_t qv1 = vsubq_s8(vreinterpretq_s8_u8(vandq_u8(vshlq_u8(vqtbl1q_u8(raw16, idx_hi), shifts), mask2)), one);
                for (int t = 0; t < NT; ++t) {
                    const block_q8_0& yb = y[t][i * 2 + k];
                    const float d1 = h2f(yb.d);
                    const int32x4_t p0 = vdotq_s32(zero, qv0, vld1q_s8(yb.qs));
                    const int32x4_t p1 = vdotq_s32(p0,   qv1, vld1q_s8(yb.qs + 16));
                    sumv[t] = vmlaq_n_f32(sumv[t], vcvtq_f32_s32(p1), d0 * d1);
                }
            }
        }
        for (int t = 0; t < NT; ++t) out[t][r] = vaddvq_f32(sumv[t]);
    }
}

}  // namespace
#endif  // STRATA_HAVE_ARM_DOTPROD

#if defined(STRATA_HAVE_ARM_DOTPROD)
namespace {

STRATA_ARM_INLINE void arm_unpack_k4_scales(const uint8_t* sc, uint8_t scales[8], uint8_t mins[8]) {
    constexpr uint32_t kmask1 = 0x3f3f3f3f, kmask2 = 0x0f0f0f0f, kmask3 = 0x03030303;
    const uint32_t u0 = u32(sc), u1 = u32(sc + 4), u2 = u32(sc + 8);
    const uint32_t sc_lo = u0 & kmask1;
    const uint32_t sc_hi = (u2 & kmask2) | (((u0 >> 6) & kmask3) << 4);
    const uint32_t min_lo = u1 & kmask1;
    const uint32_t min_hi = ((u2 >> 4) & kmask2) | (((u1 >> 6) & kmask3) << 4);
    std::memcpy(scales + 0, &sc_lo, 4);
    std::memcpy(scales + 4, &sc_hi, 4);
    std::memcpy(mins + 0, &min_lo, 4);
    std::memcpy(mins + 4, &min_hi, 4);
}

template <int NT>
STRATA_ARM_INLINE void arm_dot_q4_k(const block_q4_K* x, int nb, const block_q8_K* const* y, float* out) {
    const uint8x16_t m4b = vdupq_n_u8(0x0f);
    const int32x4_t zero = vdupq_n_s32(0);
    float sumf[NT] = {};
    for (int i = 0; i < nb; ++i) {
        const uint8_t* xb = (const uint8_t*) &x[i];
        const float dx = h2f(u16(xb)), dminx = h2f(u16(xb + 2));
        uint8_t sc[8], mn[8];
        arm_unpack_k4_scales(x[i].scales, sc, mn);
        const int16x8_t mins = vreinterpretq_s16_u16(vmovl_u8(vld1_u8(mn)));
        int32_t sumi1[NT] = {}, sumi2[NT] = {};
        for (int j = 0; j < 4; ++j) {
            const uint8x16_t b0 = vld1q_u8(x[i].qs + 32 * j);
            const uint8x16_t b1 = vld1q_u8(x[i].qs + 32 * j + 16);
            const int8x16_t q0l = vreinterpretq_s8_u8(vandq_u8(b0, m4b));
            const int8x16_t q1l = vreinterpretq_s8_u8(vandq_u8(b1, m4b));
            const int8x16_t q0h = vreinterpretq_s8_u8(vshrq_n_u8(b0, 4));
            const int8x16_t q1h = vreinterpretq_s8_u8(vshrq_n_u8(b1, 4));
            const int32_t s_lo = sc[2 * j + 0], s_hi = sc[2 * j + 1];
            for (int t = 0; t < NT; ++t) {
                const int8_t* q8 = y[t][i].qs + 64 * j;
                const int32x4_t p1 = vdotq_s32(vdotq_s32(zero, q0l, vld1q_s8(q8 + 0)),  q1l, vld1q_s8(q8 + 16));
                const int32x4_t p2 = vdotq_s32(vdotq_s32(zero, q0h, vld1q_s8(q8 + 32)), q1h, vld1q_s8(q8 + 48));
                sumi1[t] += vaddvq_s32(p1) * s_lo;
                sumi2[t] += vaddvq_s32(p2) * s_hi;
            }
        }
        for (int t = 0; t < NT; ++t) {
            const float yd = y[t][i].d;
            const int16x8_t q8sums = vpaddq_s16(vld1q_s16(y[t][i].bsums), vld1q_s16(y[t][i].bsums + 8));
            const int32x4_t prod = vaddq_s32(vmull_s16(vget_low_s16(q8sums), vget_low_s16(mins)),
                                             vmull_s16(vget_high_s16(q8sums), vget_high_s16(mins)));
            sumf[t] -= (yd * dminx) * (float) vaddvq_s32(prod);
            sumf[t] += (yd * dx) * (float) (sumi1[t] + sumi2[t]);
        }
    }
    for (int t = 0; t < NT; ++t) out[t] = sumf[t];
}

template <int NT>
STRATA_ARM_INLINE void arm_dot_q5_k(const block_q5_K* x, int nb, const block_q8_K* const* y, float* out) {
    const uint8x16_t m4b = vdupq_n_u8(0x0f), mone = vdupq_n_u8(1), mtwo = vdupq_n_u8(2);
    const int32x4_t zero = vdupq_n_s32(0);
    float sumf[NT] = {};
    for (int i = 0; i < nb; ++i) {
        const uint8_t* xb = (const uint8_t*) &x[i];
        const float dx = h2f(u16(xb)), dminx = h2f(u16(xb + 2));
        uint8_t sc[8], mn[8];
        arm_unpack_k4_scales(x[i].scales, sc, mn);
        const int16x8_t mins = vreinterpretq_s16_u16(vmovl_u8(vld1_u8(mn)));
        uint8x16_t qh0 = vld1q_u8(x[i].qh + 0), qh1 = vld1q_u8(x[i].qh + 16);
        int32_t sumi[NT] = {};
        for (int j = 0; j < 4; ++j) {
            const uint8x16_t b0 = vld1q_u8(x[i].qs + 32 * j);
            const uint8x16_t b1 = vld1q_u8(x[i].qs + 32 * j + 16);
            const uint8x16_t h0 = vshlq_n_u8(vandq_u8(mone, qh0), 4);
            const uint8x16_t h1 = vshlq_n_u8(vandq_u8(mone, qh1), 4);
            const uint8x16_t h2 = vshlq_n_u8(vandq_u8(mtwo, qh0), 3);
            const uint8x16_t h3 = vshlq_n_u8(vandq_u8(mtwo, qh1), 3);
            qh0 = vshrq_n_u8(qh0, 2);
            qh1 = vshrq_n_u8(qh1, 2);
            const int8x16_t q0l = vreinterpretq_s8_u8(vorrq_u8(vandq_u8(b0, m4b), h0));
            const int8x16_t q1l = vreinterpretq_s8_u8(vorrq_u8(vandq_u8(b1, m4b), h1));
            const int8x16_t q0h = vreinterpretq_s8_u8(vorrq_u8(vshrq_n_u8(b0, 4), h2));
            const int8x16_t q1h = vreinterpretq_s8_u8(vorrq_u8(vshrq_n_u8(b1, 4), h3));
            const int32_t s_lo = sc[2 * j + 0], s_hi = sc[2 * j + 1];
            for (int t = 0; t < NT; ++t) {
                const int8_t* q8 = y[t][i].qs + 64 * j;
                sumi[t] += vaddvq_s32(vdotq_s32(vdotq_s32(zero, q0l, vld1q_s8(q8 + 0)),  q1l, vld1q_s8(q8 + 16))) * s_lo;
                sumi[t] += vaddvq_s32(vdotq_s32(vdotq_s32(zero, q0h, vld1q_s8(q8 + 32)), q1h, vld1q_s8(q8 + 48))) * s_hi;
            }
        }
        for (int t = 0; t < NT; ++t) {
            const float yd = y[t][i].d;
            const int16x8_t q8sums = vpaddq_s16(vld1q_s16(y[t][i].bsums), vld1q_s16(y[t][i].bsums + 8));
            const int32x4_t prod = vaddq_s32(vmull_s16(vget_low_s16(q8sums), vget_low_s16(mins)),
                                             vmull_s16(vget_high_s16(q8sums), vget_high_s16(mins)));
            sumf[t] += (yd * dx) * (float) sumi[t] - (yd * dminx) * (float) vaddvq_s32(prod);
        }
    }
    for (int t = 0; t < NT; ++t) out[t] = sumf[t];
}

template <int NT>
STRATA_ARM_INLINE void arm_dot_q6_k(const block_q6_K* x, int nb, const block_q8_K* const* y, float* out) {
    const uint8x16_t m4b = vdupq_n_u8(0x0f), m3 = vdupq_n_u8(3);
    const int32x4_t zero = vdupq_n_s32(0);
    float sumf[NT] = {};
    for (int i = 0; i < nb; ++i) {
        const float dx = h2f(x[i].d);
        const int8x16_t sc8 = vld1q_s8(x[i].scales);
        const int16x8_t sc_lo = vmovl_s8(vget_low_s8(sc8)), sc_hi = vmovl_s8(vget_high_s8(sc8));
        int32_t isum[NT] = {};
        for (int j = 0; j < 2; ++j) {
            const uint8x16_t qh0 = vld1q_u8(x[i].qh + 32 * j + 0), qh1 = vld1q_u8(x[i].qh + 32 * j + 16);
            const uint8x16_t ql0 = vld1q_u8(x[i].ql + 64 * j + 0), ql1 = vld1q_u8(x[i].ql + 64 * j + 16);
            const uint8x16_t ql2 = vld1q_u8(x[i].ql + 64 * j + 32), ql3 = vld1q_u8(x[i].ql + 64 * j + 48);
            const int8x16_t a0 = vreinterpretq_s8_u8(vorrq_u8(vandq_u8(ql0, m4b), vshlq_n_u8(vandq_u8(m3, qh0), 4)));
            const int8x16_t a1 = vreinterpretq_s8_u8(vorrq_u8(vandq_u8(ql1, m4b), vshlq_n_u8(vandq_u8(m3, qh1), 4)));
            const int8x16_t a2 = vreinterpretq_s8_u8(vorrq_u8(vandq_u8(ql2, m4b), vshlq_n_u8(vandq_u8(m3, vshrq_n_u8(qh0, 2)), 4)));
            const int8x16_t a3 = vreinterpretq_s8_u8(vorrq_u8(vandq_u8(ql3, m4b), vshlq_n_u8(vandq_u8(m3, vshrq_n_u8(qh1, 2)), 4)));
            const int8x16_t b0 = vreinterpretq_s8_u8(vorrq_u8(vshrq_n_u8(ql0, 4), vshlq_n_u8(vandq_u8(m3, vshrq_n_u8(qh0, 4)), 4)));
            const int8x16_t b1 = vreinterpretq_s8_u8(vorrq_u8(vshrq_n_u8(ql1, 4), vshlq_n_u8(vandq_u8(m3, vshrq_n_u8(qh1, 4)), 4)));
            const int8x16_t b2 = vreinterpretq_s8_u8(vorrq_u8(vshrq_n_u8(ql2, 4), vshlq_n_u8(vandq_u8(m3, vshrq_n_u8(qh0, 6)), 4)));
            const int8x16_t b3 = vreinterpretq_s8_u8(vorrq_u8(vshrq_n_u8(ql3, 4), vshlq_n_u8(vandq_u8(m3, vshrq_n_u8(qh1, 6)), 4)));
            const int8_t* sc = x[i].scales + 8 * j;
            for (int t = 0; t < NT; ++t) {
                const int8_t* q8 = y[t][i].qs + 128 * j;
                isum[t] += vaddvq_s32(vdotq_s32(zero, a0, vld1q_s8(q8 + 0)))   * sc[0] +
                           vaddvq_s32(vdotq_s32(zero, a1, vld1q_s8(q8 + 16)))  * sc[1] +
                           vaddvq_s32(vdotq_s32(zero, a2, vld1q_s8(q8 + 32)))  * sc[2] +
                           vaddvq_s32(vdotq_s32(zero, a3, vld1q_s8(q8 + 48)))  * sc[3] +
                           vaddvq_s32(vdotq_s32(zero, b0, vld1q_s8(q8 + 64)))  * sc[4] +
                           vaddvq_s32(vdotq_s32(zero, b1, vld1q_s8(q8 + 80)))  * sc[5] +
                           vaddvq_s32(vdotq_s32(zero, b2, vld1q_s8(q8 + 96)))  * sc[6] +
                           vaddvq_s32(vdotq_s32(zero, b3, vld1q_s8(q8 + 112))) * sc[7];
            }
        }
        for (int t = 0; t < NT; ++t) {
            const int16x8_t q8s0 = vld1q_s16(y[t][i].bsums + 0), q8s1 = vld1q_s16(y[t][i].bsums + 8);
            const int32x4_t prod = vaddq_s32(vaddq_s32(vmull_s16(vget_low_s16(q8s0), vget_low_s16(sc_lo)),
                                                       vmull_s16(vget_high_s16(q8s0), vget_high_s16(sc_lo))),
                                             vaddq_s32(vmull_s16(vget_low_s16(q8s1), vget_low_s16(sc_hi)),
                                                       vmull_s16(vget_high_s16(q8s1), vget_high_s16(sc_hi))));
            sumf[t] += (dx * y[t][i].d) * (float) (isum[t] - 32 * vaddvq_s32(prod));
        }
    }
    for (int t = 0; t < NT; ++t) out[t] = sumf[t];
}

template <int NT>
STRATA_ARM_INLINE void arm_dot_q8_0(const block_q8_0* x, int nb, const block_q8_0* const* y, float* out) {
    const int32x4_t zero = vdupq_n_s32(0);
    float32x4_t sumv0[NT], sumv1[NT];
    for (int t = 0; t < NT; ++t) { sumv0[t] = vdupq_n_f32(0.f); sumv1[t] = vdupq_n_f32(0.f); }
    int ib = 0;
    for (; ib + 1 < nb; ib += 2) {
        const int8x16_t x0_0 = vld1q_s8(x[ib + 0].qs), x0_1 = vld1q_s8(x[ib + 0].qs + 16);
        const int8x16_t x1_0 = vld1q_s8(x[ib + 1].qs), x1_1 = vld1q_s8(x[ib + 1].qs + 16);
        const float dx0 = h2f(x[ib + 0].d), dx1 = h2f(x[ib + 1].d);
        for (int t = 0; t < NT; ++t) {
            const int32x4_t p0 = vaddq_s32(vdotq_s32(zero, x0_0, vld1q_s8(y[t][ib + 0].qs)),
                                           vdotq_s32(zero, x0_1, vld1q_s8(y[t][ib + 0].qs + 16)));
            const int32x4_t p1 = vaddq_s32(vdotq_s32(zero, x1_0, vld1q_s8(y[t][ib + 1].qs)),
                                           vdotq_s32(zero, x1_1, vld1q_s8(y[t][ib + 1].qs + 16)));
            sumv0[t] = vmlaq_n_f32(sumv0[t], vcvtq_f32_s32(p0), dx0 * h2f(y[t][ib + 0].d));
            sumv1[t] = vmlaq_n_f32(sumv1[t], vcvtq_f32_s32(p1), dx1 * h2f(y[t][ib + 1].d));
        }
    }
    for (int t = 0; t < NT; ++t) {
        float s = vaddvq_f32(sumv0[t]) + vaddvq_f32(sumv1[t]);
        for (int k = ib; k < nb; ++k) {
            int sumi = 0;
            for (int j = 0; j < QK8_0; ++j) sumi += x[k].qs[j] * y[t][k].qs[j];
            s += (float) sumi * (h2f(x[k].d) * h2f(y[t][k].d));
        }
        out[t] = s;
    }
}

STRATA_ARM_INLINE void expand_q5_high_bits(uint32_t qh, bool invert_shift4, int8x16_t& lo, int8x16_t& hi) {
    static const uint8_t kRepL[16] = {0,0,0,0,0,0,0,0, 1,1,1,1,1,1,1,1};
    static const uint8_t kRepH[16] = {2,2,2,2,2,2,2,2, 3,3,3,3,3,3,3,3};
    static const uint8_t kBits[16] = {1,2,4,8,16,32,64,128, 1,2,4,8,16,32,64,128};
    uint8_t b4[8] = {};
    std::memcpy(b4, &qh, 4);
    const uint8x8_t v = vld1_u8(b4);
    const uint8x16_t v16 = vcombine_u8(v, v);
    const uint8x16_t bm = vld1q_u8(kBits);
    const uint8x16_t m10 = vdupq_n_u8(0x10);
    const uint8x16_t t0 = vtstq_u8(vqtbl1q_u8(v16, vld1q_u8(kRepL)), bm);
    const uint8x16_t t1 = vtstq_u8(vqtbl1q_u8(v16, vld1q_u8(kRepH)), bm);
    if (invert_shift4) {
        lo = vreinterpretq_s8_u8(vandq_u8(vceqq_u8(t0, vdupq_n_u8(0)), m10));
        hi = vreinterpretq_s8_u8(vandq_u8(vceqq_u8(t1, vdupq_n_u8(0)), m10));
    } else {
        lo = vreinterpretq_s8_u8(vandq_u8(t0, m10));
        hi = vreinterpretq_s8_u8(vandq_u8(t1, m10));
    }
}

template <int NT>
STRATA_ARM_INLINE void arm_dot_q5_0(const block_q5_0* x, int nb, const block_q8_0* const* y, float* out) {
    const uint8x16_t m4b = vdupq_n_u8(0x0f);
    const int32x4_t zero = vdupq_n_s32(0);
    float32x4_t sumv0[NT], sumv1[NT];
    for (int t = 0; t < NT; ++t) { sumv0[t] = vdupq_n_f32(0.f); sumv1[t] = vdupq_n_f32(0.f); }
    int ib = 0;
    for (; ib + 1 < nb; ib += 2) {
        int8x16_t qhl0, qhh0, qhl1, qhh1;
        expand_q5_high_bits(u32(x[ib + 0].qh), true, qhl0, qhh0);
        expand_q5_high_bits(u32(x[ib + 1].qh), true, qhl1, qhh1);
        const uint8x16_t v0 = vld1q_u8(x[ib + 0].qs), v1 = vld1q_u8(x[ib + 1].qs);
        const int8x16_t x0l = vsubq_s8(vreinterpretq_s8_u8(vandq_u8(v0, m4b)), qhl0);
        const int8x16_t x0h = vsubq_s8(vreinterpretq_s8_u8(vshrq_n_u8(v0, 4)), qhh0);
        const int8x16_t x1l = vsubq_s8(vreinterpretq_s8_u8(vandq_u8(v1, m4b)), qhl1);
        const int8x16_t x1h = vsubq_s8(vreinterpretq_s8_u8(vshrq_n_u8(v1, 4)), qhh1);
        const float dx0 = h2f(x[ib + 0].d), dx1 = h2f(x[ib + 1].d);
        for (int t = 0; t < NT; ++t) {
            const int32x4_t p0 = vaddq_s32(vdotq_s32(zero, x0l, vld1q_s8(y[t][ib + 0].qs)),
                                           vdotq_s32(zero, x0h, vld1q_s8(y[t][ib + 0].qs + 16)));
            const int32x4_t p1 = vaddq_s32(vdotq_s32(zero, x1l, vld1q_s8(y[t][ib + 1].qs)),
                                           vdotq_s32(zero, x1h, vld1q_s8(y[t][ib + 1].qs + 16)));
            sumv0[t] = vmlaq_n_f32(sumv0[t], vcvtq_f32_s32(p0), dx0 * h2f(y[t][ib + 0].d));
            sumv1[t] = vmlaq_n_f32(sumv1[t], vcvtq_f32_s32(p1), dx1 * h2f(y[t][ib + 1].d));
        }
    }
    for (int t = 0; t < NT; ++t) {
        float s = vaddvq_f32(sumv0[t]) + vaddvq_f32(sumv1[t]);
        for (int k = ib; k < nb; ++k) {
            const uint32_t qh = u32(x[k].qh);
            int s0 = 0, s1 = 0;
            for (int j = 0; j < QK5_0 / 2; ++j) {
                const uint8_t xh0 = ((qh & (1u << j)) >> j) << 4;
                const uint8_t xh1 = ((qh & (1u << (j + 16))) >> (j + 12));
                s0 += (int32_t) (int8_t) (((x[k].qs[j] & 0x0f) | xh0) - 16) * y[t][k].qs[j];
                s1 += (int32_t) (int8_t) (((x[k].qs[j] >> 4) | xh1) - 16) * y[t][k].qs[j + QK5_0 / 2];
            }
            s += (h2f(x[k].d) * h2f(y[t][k].d)) * (float) (s0 + s1);
        }
        out[t] = s;
    }
}

template <int NT>
STRATA_ARM_INLINE void arm_dot_q5_1(const block_q5_1* x, int nb, const block_q8_1* const* y, float* out) {
    const uint8x16_t m4b = vdupq_n_u8(0x0f);
    const int32x4_t zero = vdupq_n_s32(0);
    float32x4_t sumv0[NT], sumv1[NT];
    float summs0[NT] = {}, summs1[NT] = {};
    for (int t = 0; t < NT; ++t) { sumv0[t] = vdupq_n_f32(0.f); sumv1[t] = vdupq_n_f32(0.f); }
    int ib = 0;
    for (; ib + 1 < nb; ib += 2) {
        int8x16_t qhl0, qhh0, qhl1, qhh1;
        expand_q5_high_bits(u32(x[ib + 0].qh), false, qhl0, qhh0);
        expand_q5_high_bits(u32(x[ib + 1].qh), false, qhl1, qhh1);
        const uint8x16_t v0 = vld1q_u8(x[ib + 0].qs), v1 = vld1q_u8(x[ib + 1].qs);
        const int8x16_t x0l = vorrq_s8(vreinterpretq_s8_u8(vandq_u8(v0, m4b)), qhl0);
        const int8x16_t x0h = vorrq_s8(vreinterpretq_s8_u8(vshrq_n_u8(v0, 4)), qhh0);
        const int8x16_t x1l = vorrq_s8(vreinterpretq_s8_u8(vandq_u8(v1, m4b)), qhl1);
        const int8x16_t x1h = vorrq_s8(vreinterpretq_s8_u8(vshrq_n_u8(v1, 4)), qhh1);
        const uint8_t* xb0 = (const uint8_t*) &x[ib + 0];
        const uint8_t* xb1 = (const uint8_t*) &x[ib + 1];
        const float dx0 = h2f(u16(xb0)), mx0 = h2f(u16(xb0 + 2));
        const float dx1 = h2f(u16(xb1)), mx1 = h2f(u16(xb1 + 2));
        for (int t = 0; t < NT; ++t) {
            const uint8_t* yb0 = (const uint8_t*) &y[t][ib + 0];
            const uint8_t* yb1 = (const uint8_t*) &y[t][ib + 1];
            summs0[t] += mx0 * h2f(u16(yb0 + 2));
            summs1[t] += mx1 * h2f(u16(yb1 + 2));
            const int32x4_t p0 = vaddq_s32(vdotq_s32(zero, x0l, vld1q_s8(y[t][ib + 0].qs)),
                                           vdotq_s32(zero, x0h, vld1q_s8(y[t][ib + 0].qs + 16)));
            const int32x4_t p1 = vaddq_s32(vdotq_s32(zero, x1l, vld1q_s8(y[t][ib + 1].qs)),
                                           vdotq_s32(zero, x1h, vld1q_s8(y[t][ib + 1].qs + 16)));
            sumv0[t] = vmlaq_n_f32(sumv0[t], vcvtq_f32_s32(p0), dx0 * h2f(u16(yb0)));
            sumv1[t] = vmlaq_n_f32(sumv1[t], vcvtq_f32_s32(p1), dx1 * h2f(u16(yb1)));
        }
    }
    for (int t = 0; t < NT; ++t) {
        float s = vaddvq_f32(sumv0[t]) + vaddvq_f32(sumv1[t]) + summs0[t] + summs1[t];
        for (int k = ib; k < nb; ++k) {
            const uint8_t* xbk = (const uint8_t*) &x[k];
            const uint8_t* ybk = (const uint8_t*) &y[t][k];
            const uint32_t qh = u32(x[k].qh);
            int s0 = 0, s1 = 0;
            for (int j = 0; j < QK5_1 / 2; ++j) {
                const uint8_t xh0 = ((qh >> j) << 4) & 0x10;
                const uint8_t xh1 = (qh >> (j + 12)) & 0x10;
                s0 += ((x[k].qs[j] & 0x0f) | xh0) * y[t][k].qs[j];
                s1 += ((x[k].qs[j] >> 4) | xh1) * y[t][k].qs[j + QK5_1 / 2];
            }
            s += (h2f(u16(xbk)) * h2f(u16(ybk))) * (float) (s0 + s1) + h2f(u16(xbk + 2)) * h2f(u16(ybk + 2));
        }
        out[t] = s;
    }
}

template <int NT>
void arm_kq_row_nt(int type, const uint8_t* w, int n, const void* const* act, float* out) {
    switch (type) {
        case 12: arm_dot_q4_k<NT>((const block_q4_K*) w, n / QK_K, (const block_q8_K* const*) act, out); break;
        case 13: arm_dot_q5_k<NT>((const block_q5_K*) w, n / QK_K, (const block_q8_K* const*) act, out); break;
        case 14: arm_dot_q6_k<NT>((const block_q6_K*) w, n / QK_K, (const block_q8_K* const*) act, out); break;
        case 6:  arm_dot_q5_0<NT>((const block_q5_0*) w, n / QK5_0, (const block_q8_0* const*) act, out); break;
        case 7:  arm_dot_q5_1<NT>((const block_q5_1*) w, n / QK5_1, (const block_q8_1* const*) act, out); break;
        case 8:  arm_dot_q8_0<NT>((const block_q8_0*) w, n / QK8_0, (const block_q8_0* const*) act, out); break;
        default: break;
    }
}

void arm_kq_row(int type, const uint8_t* w, int n, const void* const* act, int nt, float* out) {
    switch (nt) {
        case 1: arm_kq_row_nt<1>(type, w, n, act, out); break;
        case 2: arm_kq_row_nt<2>(type, w, n, act, out); break;
        case 3: arm_kq_row_nt<3>(type, w, n, act, out); break;
        default: arm_kq_row_nt<4>(type, w, n, act, out); break;
    }
}

template <int NT>
STRATA_ARM_INLINE void arm_gu_q4_k(const block_q4_K* xg, const block_q4_K* xu, int nb, const block_q8_K* const* y,
                                   float* out_g, float* out_u) {
    const uint8x16_t m4b = vdupq_n_u8(0x0f);
    const int32x4_t zero = vdupq_n_s32(0);
    float sum_g[NT] = {}, sum_u[NT] = {};
    for (int i = 0; i < nb; ++i) {
        const uint8_t* xbg = (const uint8_t*) &xg[i];
        const uint8_t* xbu = (const uint8_t*) &xu[i];
        const float dxg = h2f(u16(xbg)), dminxg = h2f(u16(xbg + 2));
        const float dxu = h2f(u16(xbu)), dminxu = h2f(u16(xbu + 2));
        uint8_t scg[8], mng[8], scu[8], mnu[8];
        arm_unpack_k4_scales(xg[i].scales, scg, mng);
        arm_unpack_k4_scales(xu[i].scales, scu, mnu);
        const int16x8_t mins_g = vreinterpretq_s16_u16(vmovl_u8(vld1_u8(mng)));
        const int16x8_t mins_u = vreinterpretq_s16_u16(vmovl_u8(vld1_u8(mnu)));
        int32_t sg1[NT] = {}, sg2[NT] = {}, su1[NT] = {}, su2[NT] = {};
        for (int j = 0; j < 4; ++j) {
            const uint8x16_t gb0 = vld1q_u8(xg[i].qs + 32 * j), gb1 = vld1q_u8(xg[i].qs + 32 * j + 16);
            const uint8x16_t ub0 = vld1q_u8(xu[i].qs + 32 * j), ub1 = vld1q_u8(xu[i].qs + 32 * j + 16);
            const int8x16_t g0l = vreinterpretq_s8_u8(vandq_u8(gb0, m4b)), g1l = vreinterpretq_s8_u8(vandq_u8(gb1, m4b));
            const int8x16_t g0h = vreinterpretq_s8_u8(vshrq_n_u8(gb0, 4)), g1h = vreinterpretq_s8_u8(vshrq_n_u8(gb1, 4));
            const int8x16_t u0l = vreinterpretq_s8_u8(vandq_u8(ub0, m4b)), u1l = vreinterpretq_s8_u8(vandq_u8(ub1, m4b));
            const int8x16_t u0h = vreinterpretq_s8_u8(vshrq_n_u8(ub0, 4)), u1h = vreinterpretq_s8_u8(vshrq_n_u8(ub1, 4));
            const int32_t sgl = scg[2 * j + 0], sgh = scg[2 * j + 1];
            const int32_t sul = scu[2 * j + 0], suh = scu[2 * j + 1];
            for (int t = 0; t < NT; ++t) {
                const int8_t* q8 = y[t][i].qs + 64 * j;
                const int8x16_t y0 = vld1q_s8(q8 + 0),  y1 = vld1q_s8(q8 + 16);
                const int8x16_t y2 = vld1q_s8(q8 + 32), y3 = vld1q_s8(q8 + 48);
                sg1[t] += vaddvq_s32(vdotq_s32(vdotq_s32(zero, g0l, y0), g1l, y1)) * sgl;
                sg2[t] += vaddvq_s32(vdotq_s32(vdotq_s32(zero, g0h, y2), g1h, y3)) * sgh;
                su1[t] += vaddvq_s32(vdotq_s32(vdotq_s32(zero, u0l, y0), u1l, y1)) * sul;
                su2[t] += vaddvq_s32(vdotq_s32(vdotq_s32(zero, u0h, y2), u1h, y3)) * suh;
            }
        }
        for (int t = 0; t < NT; ++t) {
            const float yd = y[t][i].d;
            const int16x8_t q8sums = vpaddq_s16(vld1q_s16(y[t][i].bsums), vld1q_s16(y[t][i].bsums + 8));
            const int32x4_t pg = vaddq_s32(vmull_s16(vget_low_s16(q8sums), vget_low_s16(mins_g)),
                                           vmull_s16(vget_high_s16(q8sums), vget_high_s16(mins_g)));
            const int32x4_t pu = vaddq_s32(vmull_s16(vget_low_s16(q8sums), vget_low_s16(mins_u)),
                                           vmull_s16(vget_high_s16(q8sums), vget_high_s16(mins_u)));
            sum_g[t] += (yd * dxg) * (float) (sg1[t] + sg2[t]) - (yd * dminxg) * (float) vaddvq_s32(pg);
            sum_u[t] += (yd * dxu) * (float) (su1[t] + su2[t]) - (yd * dminxu) * (float) vaddvq_s32(pu);
        }
    }
    for (int t = 0; t < NT; ++t) { out_g[t] = sum_g[t]; out_u[t] = sum_u[t]; }
}

template <int NT>
void arm_kq_gu_row_nt(int type, const uint8_t* g_ptr, const uint8_t* u_ptr, int n, const void* const* act,
                      float* g_out, float* u_out) {
    if (type == 12) {
        arm_gu_q4_k<NT>((const block_q4_K*) g_ptr, (const block_q4_K*) u_ptr, n / QK_K,
                        (const block_q8_K* const*) act, g_out, u_out);
        return;
    }
    arm_kq_row_nt<NT>(type, g_ptr, n, act, g_out);
    arm_kq_row_nt<NT>(type, u_ptr, n, act, u_out);
}

void arm_kq_gu_row(int type, const uint8_t* g_ptr, const uint8_t* u_ptr, int n, const void* const* act, int nt,
                   float* g_out, float* u_out) {
    switch (nt) {
        case 1: arm_kq_gu_row_nt<1>(type, g_ptr, u_ptr, n, act, g_out, u_out); break;
        case 2: arm_kq_gu_row_nt<2>(type, g_ptr, u_ptr, n, act, g_out, u_out); break;
        case 3: arm_kq_gu_row_nt<3>(type, g_ptr, u_ptr, n, act, g_out, u_out); break;
        default: arm_kq_gu_row_nt<4>(type, g_ptr, u_ptr, n, act, g_out, u_out); break;
    }
}

}  // namespace
#endif  // STRATA_HAVE_ARM_DOTPROD

void s2_expert_vnni(const uint8_t*, const float*, float*, ExpertScratch&) { unsupported("the Q2_0 expert kernel"); }
void s2_expert_vnni_q(const uint8_t*, const ActQ&, float*, ExpertScratch&) { unsupported("the Q2_0 expert kernel"); }
void s2_expert_gu_rows(const uint8_t*, const ActQ&, float*, int, int) { unsupported("the Q2_0 expert kernel"); }
void s2_expert_down_rows(const uint8_t*, const ActQ&, float*, int, int) { unsupported("the Q2_0 expert kernel"); }
void s2_expert_gu_rows_multi(const uint8_t*, const ActQ* const*, int, float* const*, int, int) {
    unsupported("the Q2_0 expert kernel");
}
void s2_expert_down_rows_multi(const uint8_t*, const ActQ* const*, int, float* const*, int, int) {
    unsupported("the Q2_0 expert kernel");
}
void s2_expert_vnni_multi(const uint8_t*, const ActQ* const*, int, float* const*, ExpertScratchMulti&) {
    unsupported("the Q2_0 expert kernel");
}
void q2_0_gguf_rows_multi(const uint8_t*, size_t, int, const ActQ* const*, int, float* const*, int, int) {
    unsupported("the Q2_0 GGUF row kernel");
}
bool q2_bitplane_enabled() { return false; }

void q2_0_gguf_rows_multi_avx2(const uint8_t* w, size_t row_bytes, int nblocks, const ActQ* const* a, int nt,
                               float* const* out, int r0, int r1) {
#if defined(STRATA_HAVE_ARM_DOTPROD)
    const void* const* hq = (const void* const*) a;
    for (int t0 = 0; t0 < nt; t0 += 4) {
        const int m = nt - t0 < 4 ? nt - t0 : 4;
        switch (m) {
            case 1: arm_q2_rows_nt<1>(w, row_bytes, nblocks, hq + t0, out + t0, r0, r1); break;
            case 2: arm_q2_rows_nt<2>(w, row_bytes, nblocks, hq + t0, out + t0, r0, r1); break;
            case 3: arm_q2_rows_nt<3>(w, row_bytes, nblocks, hq + t0, out + t0, r0, r1); break;
            default: arm_q2_rows_nt<4>(w, row_bytes, nblocks, hq + t0, out + t0, r0, r1); break;
        }
    }
#else
    (void) w; (void) row_bytes; (void) nblocks; (void) a; (void) nt; (void) out; (void) r0; (void) r1;
    unsupported("the Q2_0 GGUF row kernel");
#endif
}

void q2_0_gguf_rows_multi_avx2_legacy(const uint8_t* w, size_t row_bytes, int nblocks, const ActQ* const* a, int nt,
                                      float* const* out, int r0, int r1) {
    q2_0_gguf_rows_multi_avx2(w, row_bytes, nblocks, a, nt, out, r0, r1);
}
void q2_0_gguf_rows_multi_avx2_v(bool, const uint8_t* w, size_t row_bytes, int nblocks, const ActQ* const* a, int nt,
                                 float* const* out, int r0, int r1) {
    q2_0_gguf_rows_multi_avx2(w, row_bytes, nblocks, a, nt, out, r0, r1);
}
void s2_expert_scalar(const uint8_t*, const float*, float*, bool) { unsupported("the Q2_0 scalar oracle"); }

void iq512_gu_rows(int, const uint8_t*, size_t, size_t, int, const void* const*, int, float* const*, int, int) {
    unsupported("the AVX-512 i-quant kernel");
}
void iq512_reference_gu_rows(int, const uint8_t*, size_t, size_t, int, const void* const*, int, float* const*, int, int) {
    unsupported("the AVX-512 i-quant reference kernel");
}
void iq512_rows(int, const uint8_t*, size_t, int, const void* const*, int, float* const*, int, int) {
    unsupported("the AVX-512 i-quant kernel");
}
bool iq512_gather_on() noexcept { return false; }
bool iq512x_supported(int) noexcept { return false; }
void iq512x_rows(int, const uint8_t*, size_t, int, const void* const*, int, float* const*, int, int) {
    unsupported("the AVX-512 exact i-quant kernel");
}
void iq4nl512_down_rows(const uint8_t*, size_t, int, const void* const*, int, float* const*, int, int) {
    unsupported("the AVX-512 IQ4_NL kernel");
}
void q2_512_down_rows(const uint8_t*, size_t, int, const void* const*, int, float* const*, int, int) {
    unsupported("the AVX-512 Q2_0 kernel");
}
void kq512_gu_rows(int, const uint8_t*, size_t, size_t, int, const void* const*, int, float* const*, int, int) {
    unsupported("the AVX-512 K-quant kernel");
}
void kq512_rows(int, const uint8_t*, size_t, int, const void* const*, int, float* const*, int, int) {
    unsupported("the AVX-512 K-quant kernel");
}
void q8k_quant_avx512(const float*, void*, int64_t) {
    unsupported("the AVX-512 Q8_K quantizer");
}
void bf16_rows_dot_512(const uint16_t*, int, int, const float*, float*) {
    unsupported("the AVX-512 BF16 router dot");
}
void bf16_rows_dot_multi_512(const uint16_t*, int, int, const float*, int, float*) {
    unsupported("the AVX-512 BF16 router dot");
}

bool iq256x_supported(int type) noexcept {
#if defined(STRATA_HAVE_ARM_DOTPROD)
    return type == 16 || type == 17 || type == 18 || type == 21 || type == 22 || type == 23;
#else
    (void) type;
    return false;
#endif
}

void iq256_gu_rows(int type, const uint8_t* blob, size_t gu_row, size_t up_off, int n, const void* const* act, int nt,
                   float* const* ff, int r0, int r1) {
#if defined(STRATA_HAVE_ARM_DOTPROD)
    switch (type) {
        case 16: arm_gu_rows_dispatch<16>(nt, blob, gu_row, up_off, n, act, ff, r0, r1); break;
        case 17: arm_gu_rows_dispatch<17>(nt, blob, gu_row, up_off, n, act, ff, r0, r1); break;
        case 18: arm_gu_rows_dispatch<18>(nt, blob, gu_row, up_off, n, act, ff, r0, r1); break;
        case 21: arm_gu_rows_dispatch<21>(nt, blob, gu_row, up_off, n, act, ff, r0, r1); break;
        case 22: arm_gu_rows_dispatch<22>(nt, blob, gu_row, up_off, n, act, ff, r0, r1); break;
        case 23: arm_gu_rows_dispatch<23>(nt, blob, gu_row, up_off, n, act, ff, r0, r1); break;
        default: break;
    }
#else
    (void) type; (void) blob; (void) gu_row; (void) up_off; (void) n; (void) act; (void) nt; (void) ff; (void) r0; (void) r1;
    unsupported("the i-quant gate/up kernel");
#endif
}

void iq256_rows(int type, const uint8_t* w, size_t row_bytes, int n, const void* const* act, int nt, float* const* out,
                int r0, int r1) {
#if defined(STRATA_HAVE_ARM_DOTPROD)
    switch (type) {
        case 16: arm_dot_rows_dispatch<16>(nt, w, row_bytes, n, act, out, r0, r1); break;
        case 17: arm_dot_rows_dispatch<17>(nt, w, row_bytes, n, act, out, r0, r1); break;
        case 18: arm_dot_rows_dispatch<18>(nt, w, row_bytes, n, act, out, r0, r1); break;
        case 21: arm_dot_rows_dispatch<21>(nt, w, row_bytes, n, act, out, r0, r1); break;
        case 22: arm_dot_rows_dispatch<22>(nt, w, row_bytes, n, act, out, r0, r1); break;
        case 23: arm_dot_rows_dispatch<23>(nt, w, row_bytes, n, act, out, r0, r1); break;
        default: break;
    }
#else
    (void) type; (void) w; (void) row_bytes; (void) n; (void) act; (void) nt; (void) out; (void) r0; (void) r1;
    unsupported("the i-quant row kernel");
#endif
}

void iq256x_rows(int type, const uint8_t* w, size_t row_bytes, int n, const void* const* act, int nt, float* const* out,
                 int r0, int r1) {
    iq256_rows(type, w, row_bytes, n, act, nt, out, r0, r1);
}

int iq256_variant() noexcept { return 0; }
int iq256_variants() noexcept { return 1; }
void iq256_gu_rows_v(int, int type, const uint8_t* blob, size_t gu_row, size_t up_off, int n, const void* const* act,
                     int nt, float* const* ff, int r0, int r1) {
    iq256_gu_rows(type, blob, gu_row, up_off, n, act, nt, ff, r0, r1);
}
void iq256_rows_v(int, int type, const uint8_t* w, size_t row_bytes, int n, const void* const* act, int nt,
                  float* const* out, int r0, int r1) {
    iq256_rows(type, w, row_bytes, n, act, nt, out, r0, r1);
}

void iq4nl256_down_rows(const uint8_t* w, size_t row_bytes, int n, const void* const* hq, int nt, float* const* out,
                        int r0, int r1) {
#if defined(STRATA_HAVE_ARM_DOTPROD)
    for (int t0 = 0; t0 < nt; t0 += 4) {
        const int m = nt - t0 < 4 ? nt - t0 : 4;
        switch (m) {
            case 1: arm_iq4nl_rows_nt<1>(w, row_bytes, n, hq + t0, out + t0, r0, r1); break;
            case 2: arm_iq4nl_rows_nt<2>(w, row_bytes, n, hq + t0, out + t0, r0, r1); break;
            case 3: arm_iq4nl_rows_nt<3>(w, row_bytes, n, hq + t0, out + t0, r0, r1); break;
            default: arm_iq4nl_rows_nt<4>(w, row_bytes, n, hq + t0, out + t0, r0, r1); break;
        }
    }
#else
    (void) w; (void) row_bytes; (void) n; (void) hq; (void) nt; (void) out; (void) r0; (void) r1;
    unsupported("the IQ4_NL down kernel");
#endif
}

void iq4nl256_down_rows_v(int, const uint8_t* w, size_t row_bytes, int n, const void* const* hq, int nt,
                          float* const* out, int r0, int r1) {
    iq4nl256_down_rows(w, row_bytes, n, hq, nt, out, r0, r1);
}

void kq256_gu_rows(int type, const uint8_t* blob, size_t gu_row, size_t up_off, int n, const void* const* act, int nt,
                   float* const* ff, int r0, int r1) {
#if defined(STRATA_HAVE_ARM_DOTPROD)
    for (int t0 = 0; t0 < nt; t0 += 4) {
        const int m = nt - t0 < 4 ? nt - t0 : 4;
        float g[4], u[4];
        for (int r = r0; r < r1; ++r) {
            arm_kq_gu_row(type, blob + (size_t) r * gu_row, blob + up_off + (size_t) r * gu_row, n, act + t0, m, g, u);
            for (int t = 0; t < m; ++t) ff[t0 + t][r] = (g[t] / (1.f + std::exp(-g[t]))) * u[t];
        }
    }
#else
    (void) type; (void) blob; (void) gu_row; (void) up_off; (void) n; (void) act; (void) nt; (void) ff; (void) r0; (void) r1;
    unsupported("the K-quant gate/up kernel");
#endif
}

void kq256_rows(int type, const uint8_t* w, size_t row_bytes, int n, const void* const* act, int nt, float* const* out,
                int r0, int r1) {
#if defined(STRATA_HAVE_ARM_DOTPROD)
    for (int t0 = 0; t0 < nt; t0 += 4) {
        const int m = nt - t0 < 4 ? nt - t0 : 4;
        float s[4];
        for (int r = r0; r < r1; ++r) {
            arm_kq_row(type, w + (size_t) r * row_bytes, n, act + t0, m, s);
            for (int t = 0; t < m; ++t) out[t0 + t][r] = s[t];
        }
    }
#else
    (void) type; (void) w; (void) row_bytes; (void) n; (void) act; (void) nt; (void) out; (void) r0; (void) r1;
    unsupported("the K-quant row kernel");
#endif
}

void bf16_rows_dot_multi_avx1(const uint16_t*, int, int, const float*, int, float*) {
    unsupported("the AVX1 router dot");
}

void bf16_rows_dot(const uint16_t* w, int rows, int cols, const float* x, float* out) {
    bf16_rows_dot_multi(w, rows, cols, x, 1, out);
}

// 4-token-tiled NEON BF16 router dot: loads each BF16 weight row once per 4 tokens instead of once per token.
void bf16_rows_dot_multi(const uint16_t* w, int rows, int cols, const float* x, int nt, float* out) {
    for (int t0 = 0; t0 < nt; t0 += 4) {
        const int m = nt - t0 < 4 ? nt - t0 : 4;
        const float* x0 = x + (size_t) (t0 + 0) * cols;
        const float* x1 = m > 1 ? x + (size_t) (t0 + 1) * cols : x0;
        const float* x2 = m > 2 ? x + (size_t) (t0 + 2) * cols : x0;
        const float* x3 = m > 3 ? x + (size_t) (t0 + 3) * cols : x0;
        for (int r = 0; r < rows; ++r) {
            const uint16_t* wr = w + (size_t) r * (size_t) cols;
            int c = 0;
            float s0 = 0.f, s1 = 0.f, s2 = 0.f, s3 = 0.f;
#if defined(STRATA_HAVE_ARM_NEON)
            float32x4_t a0_0 = vdupq_n_f32(0.f), a0_1 = vdupq_n_f32(0.f);
            float32x4_t a1_0 = vdupq_n_f32(0.f), a1_1 = vdupq_n_f32(0.f);
            float32x4_t a2_0 = vdupq_n_f32(0.f), a2_1 = vdupq_n_f32(0.f);
            float32x4_t a3_0 = vdupq_n_f32(0.f), a3_1 = vdupq_n_f32(0.f);
            for (; c + 8 <= cols; c += 8) {
                const uint16x8_t h = vld1q_u16(wr + c);
                const float32x4_t w0 = vreinterpretq_f32_u32(vshll_n_u16(vget_low_u16(h), 16));
                const float32x4_t w1 = vreinterpretq_f32_u32(vshll_n_u16(vget_high_u16(h), 16));
                a0_0 = vfmaq_f32(a0_0, w0, vld1q_f32(x0 + c));
                a0_1 = vfmaq_f32(a0_1, w1, vld1q_f32(x0 + c + 4));
                if (m > 1) {
                    a1_0 = vfmaq_f32(a1_0, w0, vld1q_f32(x1 + c));
                    a1_1 = vfmaq_f32(a1_1, w1, vld1q_f32(x1 + c + 4));
                }
                if (m > 2) {
                    a2_0 = vfmaq_f32(a2_0, w0, vld1q_f32(x2 + c));
                    a2_1 = vfmaq_f32(a2_1, w1, vld1q_f32(x2 + c + 4));
                }
                if (m > 3) {
                    a3_0 = vfmaq_f32(a3_0, w0, vld1q_f32(x3 + c));
                    a3_1 = vfmaq_f32(a3_1, w1, vld1q_f32(x3 + c + 4));
                }
            }
            s0 = vaddvq_f32(vaddq_f32(a0_0, a0_1));
            if (m > 1) s1 = vaddvq_f32(vaddq_f32(a1_0, a1_1));
            if (m > 2) s2 = vaddvq_f32(vaddq_f32(a2_0, a2_1));
            if (m > 3) s3 = vaddvq_f32(vaddq_f32(a3_0, a3_1));
#endif
            for (; c < cols; ++c) {
                const float wf = bf16f(wr[c]);
                s0 += wf * x0[c];
                if (m > 1) s1 += wf * x1[c];
                if (m > 2) s2 += wf * x2[c];
                if (m > 3) s3 += wf * x3[c];
            }
            out[(size_t) (t0 + 0) * rows + r] = s0;
            if (m > 1) out[(size_t) (t0 + 1) * rows + r] = s1;
            if (m > 2) out[(size_t) (t0 + 2) * rows + r] = s2;
            if (m > 3) out[(size_t) (t0 + 3) * rows + r] = s3;
        }
    }
}

[[noreturn]] static void cpu_kernel_unreachable(const char* name) {
    std::fprintf(stderr, "strata: %s is an x86 kernel and cannot run on this CPU\n", name);
    std::abort();
}
void act_quant_q8_1_avx1(const float*, int, ActQ&) { cpu_kernel_unreachable("act_quant_q8_1_avx1"); }
void iq128_down_rows(int, const uint8_t*, size_t, int, const void* const*, int, float* const*, int, int) { cpu_kernel_unreachable("iq128_down_rows"); }
bool iq128_down_supported(int) noexcept { return false; }
void iq128_gu_rows(int, const uint8_t*, size_t, size_t, int, const void* const*, int, float* const*, int, int) { cpu_kernel_unreachable("iq128_gu_rows"); }
bool iq128_supported(int) noexcept { return false; }
void iq256_gu_rows_exact_one(int, const uint8_t*, size_t, size_t, int, const void*, float*, int, int) { cpu_kernel_unreachable("iq256_gu_rows_exact_one"); }
bool nvfp4_256_fits(int) { return false; }
void nvfp4_256_gu_rows(const uint8_t*, size_t, size_t, int, const void* const*, int, float* const*, int, int, float, float) { cpu_kernel_unreachable("nvfp4_256_gu_rows"); }
void nvfp4_256_rows(const uint8_t*, size_t, int, const void* const*, int, float* const*, int, int, float) { cpu_kernel_unreachable("nvfp4_256_rows"); }
bool nvfp4_512_fits(int) { return false; }
void nvfp4_512_gu_rows(const uint8_t*, size_t, size_t, int, const void* const*, int, float* const*, int, int, float, float) { cpu_kernel_unreachable("nvfp4_512_gu_rows"); }
void nvfp4_512_rows(const uint8_t*, size_t, int, const void* const*, int, float* const*, int, int, float) { cpu_kernel_unreachable("nvfp4_512_rows"); }
void q2_0_gguf_rows_multi_avx1_legacy(const uint8_t*, size_t, int, const ActQ* const*, int, float* const*, int, int) { cpu_kernel_unreachable("q2_0_gguf_rows_multi_avx1_legacy"); }
void q2_0_gguf_rows_multi_avx1(const uint8_t*, size_t, int, const ActQ* const*, int, float* const*, int, int) { cpu_kernel_unreachable("q2_0_gguf_rows_multi_avx1"); }
void q8k_quant_avx1(const float*, void*, int64_t) { cpu_kernel_unreachable("q8k_quant_avx1"); }

#if defined(STRATA_PORTABLE_NAMESPACE)
}  // namespace STRATA_PORTABLE_NAMESPACE
#endif
}  // namespace strata::kernels::cpu
