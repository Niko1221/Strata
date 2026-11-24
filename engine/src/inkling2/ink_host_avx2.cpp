// engine/src/inkling2/ink_host_avx2.cpp - Per-TU AVX2+FMA host dot kernels for InkHybridEngine
// Compiled ONLY with -mavx2 -mfma -mf16c (MSVC /arch:AVX2) when STRATA_ISA_FLOOR allows AVX2;
// guarded at runtime by strata::quant::caps().avx2 so non-AVX2 CPUs (e.g. Ivy Bridge Xeon E5 v2, #6618)
// never execute an AVX2/FMA instruction.
// MIT License - Copyright (c) 2026 Strata contributors
#include "ink_host_avx2.h"
#include "strata/quant/isa.hpp"

#include <cmath>
#include <cstring>

#if defined(INK_BUILD_HOST_AVX2) && defined(__AVX2__)
#include <immintrin.h>
#define INK_TU_HAS_AVX2 1
#endif

namespace strata::ink::host_avx2 {

bool available() noexcept {
#if defined(INK_TU_HAS_AVX2)
    static const bool ok = strata::quant::caps().avx2;
    return ok;
#else
    return false;
#endif
}

#if defined(INK_TU_HAS_AVX2)

namespace {

// Format constants matching InkFmt in ink_cuda.cuh
constexpr int kFmtS1_5 = 1;
constexpr int kFmtS2_K = 2;
constexpr int kFmtS3_K = 3;
constexpr int kFmtS4_K = 4;
constexpr int kFmtS5_K = 5;
constexpr int kFmtS6_K = 6;
constexpr int kFmtS8_K = 8;

// The closing arithmetic of one S_K group, written out so that every compiler does the same thing (GCC's default contraction gave exactly this
// form: the activation scale times the integer sum fused with the offset term, then the weight scales multiplied in): the result of a group is
// the same in the per-group function and in the whole-row kernels below.
inline float sk_term(float eff, float x_d, int32_t isum, float cst, float x_s) noexcept {
    return eff * std::fma(x_d, (float)isum, -(cst * x_s));
}
// a + b with no fusing of an earlier multiply into the add
inline float add_nofuse(float a, float b) noexcept {
    return _mm_cvtss_f32(_mm_add_ss(_mm_set_ss(a), _mm_set_ss(b)));
}

inline float avx2_hsum_ps(__m256 v) noexcept {
    __m128 hi = _mm256_extractf128_ps(v, 1);
    __m128 lo = _mm256_castps256_ps128(v);
    __m128 s = _mm_add_ps(lo, hi);
    s = _mm_add_ps(s, _mm_movehl_ps(s, s));
    s = _mm_add_ss(s, _mm_shuffle_ps(s, s, 1));
    return _mm_cvtss_f32(s);
}

inline int32_t avx2_hsum_epi32(__m256i v) noexcept {
    __m128i hi = _mm256_extracti128_si256(v, 1);
    __m128i lo = _mm256_castsi256_si128(v);
    __m128i s = _mm_add_epi32(lo, hi);
    s = _mm_add_epi32(s, _mm_shuffle_epi32(s, 0x4E));
    s = _mm_add_epi32(s, _mm_shuffle_epi32(s, 0xB1));
    return _mm_cvtsi128_si32(s);
}

inline int32_t avx2_dot_u8_i8_64(__m256i u0, __m256i u1, const int8_t* qs) noexcept {
    const __m256i ones = _mm256_set1_epi16(1);
    const __m256i q0 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(qs));
    const __m256i q1 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(qs + 32));
    const __m256i p0 = _mm256_madd_epi16(_mm256_maddubs_epi16(u0, q0), ones);
    const __m256i p1 = _mm256_madd_epi16(_mm256_maddubs_epi16(u1, q1), ones);
    return avx2_hsum_epi32(_mm256_add_epi32(p0, p1));
}

inline void s15_unpack64_local(const uint8_t* pl, int g, int8_t out[64], const int8_t (*lut)[5]) noexcept {
    const int base = g * 64;
    int i = 0;
    int byte_idx = base / 5;
    int rem = base % 5;
    if (rem != 0) {
        const int8_t* t = lut[pl[byte_idx++]];
        for (int k = rem; k < 5 && i < 64; ++k) out[i++] = t[k];
    }
    while (i + 5 <= 64) {
        const int8_t* t = lut[pl[byte_idx++]];
        out[i + 0] = t[0]; out[i + 1] = t[1]; out[i + 2] = t[2]; out[i + 3] = t[3]; out[i + 4] = t[4];
        i += 5;
    }
    if (i < 64) {
        const int8_t* t = lut[pl[byte_idx]];
        for (int k = 0; i < 64; ++k) out[i++] = t[k];
    }
}

}  // namespace

float dot_sf4_row(const uint8_t* p, const uint8_t* sc, float w_outer, const float* x, int cols,
                  const float* e4_lut) noexcept {
    const int nb = cols >> 4;
    const __m128i lut2x  = _mm_setr_epi8(0, 1, 2, 3, 4, 6, 8, 12, 0, -1, -2, -3, -4, -6, -8, -12);
    const __m128i mask0f = _mm_set1_epi8(0x0F);
    __m256 vsum = _mm256_setzero_ps();
    for (int b = 0; b < nb; ++b) {
        const __m128i raw = _mm_loadl_epi64(reinterpret_cast<const __m128i*>(p + b * 8));
        const __m128i lo  = _mm_and_si128(raw, mask0f);
        const __m128i hi  = _mm_and_si128(_mm_srli_epi16(raw, 4), mask0f);
        const __m128i nib = _mm_unpacklo_epi8(lo, hi);
        const __m128i w8  = _mm_shuffle_epi8(lut2x, nib);
        const __m256 wf0  = _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(w8));
        const __m256 wf1  = _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(_mm_srli_si128(w8, 8)));
        const float* xb   = x + b * 16;
        const __m256 prod = _mm256_fmadd_ps(wf0, _mm256_loadu_ps(xb), _mm256_mul_ps(wf1, _mm256_loadu_ps(xb + 8)));
        vsum = _mm256_fmadd_ps(prod, _mm256_set1_ps(0.5f * e4_lut[sc[b]]), vsum);
    }
    return avx2_hsum_ps(vsum) * w_outer;
}

float dot_mx8_row(const uint8_t* w, const uint8_t* sc, const float* x, int cols,
                  const float* e4_lut, const float* u8_lut) noexcept {
    const int nb = cols >> 5;
    __m256 vsum = _mm256_setzero_ps();
    for (int b = 0; b < nb; ++b) {
        const uint8_t* wb = w + b * 32;
        const float* xb = x + b * 32;
        __m256 acc = _mm256_setzero_ps();
        for (int k = 0; k < 32; k += 8) {
            const __m256 wf = _mm256_setr_ps(
                e4_lut[wb[k + 0]], e4_lut[wb[k + 1]], e4_lut[wb[k + 2]], e4_lut[wb[k + 3]],
                e4_lut[wb[k + 4]], e4_lut[wb[k + 5]], e4_lut[wb[k + 6]], e4_lut[wb[k + 7]]);
            acc = _mm256_fmadd_ps(wf, _mm256_loadu_ps(xb + k), acc);
        }
        vsum = _mm256_fmadd_ps(acc, _mm256_set1_ps(u8_lut[sc[b]]), vsum);
    }
    return avx2_hsum_ps(vsum);
}

float dot_sk_sub64(const uint8_t* pl, int fmt, int g, const int8_t* qs, float x_d, float x_s,
                   float eff, const int8_t (*s15_lut)[5]) noexcept {
    int32_t isum = 0;
    if (fmt == kFmtS1_5) {
        alignas(32) int8_t tr[64];
        s15_unpack64_local(pl, g, tr, s15_lut);
        const __m256i bias = _mm256_set1_epi8(1);
        const __m256i u0 = _mm256_add_epi8(_mm256_load_si256(reinterpret_cast<const __m256i*>(tr)), bias);
        const __m256i u1 = _mm256_add_epi8(_mm256_load_si256(reinterpret_cast<const __m256i*>(tr + 32)), bias);
        isum = avx2_dot_u8_i8_64(u0, u1, qs);
        return eff * (x_d * (float)isum - x_s);
    } else if (fmt == kFmtS4_K) {
        const uint8_t* p = pl + g * 32;
        const __m128i m0f = _mm_set1_epi8(0x0F);
        const __m128i v0 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(p));
        const __m128i v1 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(p + 16));
        const __m128i lo0 = _mm_and_si128(v0, m0f), hi0 = _mm_and_si128(_mm_srli_epi16(v0, 4), m0f);
        const __m128i lo1 = _mm_and_si128(v1, m0f), hi1 = _mm_and_si128(_mm_srli_epi16(v1, 4), m0f);
        const __m256i u0 = _mm256_set_m128i(_mm_unpackhi_epi8(lo0, hi0), _mm_unpacklo_epi8(lo0, hi0));
        const __m256i u1 = _mm256_set_m128i(_mm_unpackhi_epi8(lo1, hi1), _mm_unpacklo_epi8(lo1, hi1));
        isum = avx2_dot_u8_i8_64(u0, u1, qs);
        return sk_term(eff, x_d, isum, 7.5f, x_s);
    } else if (fmt == kFmtS8_K) {
        const int8_t* p = reinterpret_cast<const int8_t*>(pl + g * 64);
        const __m256i p0 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(p));
        const __m256i p1 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(p + 32));
        const __m256i q0 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(qs));
        const __m256i q1 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(qs + 32));
        const __m256i ones = _mm256_set1_epi16(1);
        const __m256i a0 = _mm256_madd_epi16(_mm256_maddubs_epi16(_mm256_abs_epi8(p0), _mm256_sign_epi8(q0, p0)), ones);
        const __m256i a1 = _mm256_madd_epi16(_mm256_maddubs_epi16(_mm256_abs_epi8(p1), _mm256_sign_epi8(q1, p1)), ones);
        isum = avx2_hsum_epi32(_mm256_add_epi32(a0, a1));
        return eff * x_d * (float)isum;
    } else {
        alignas(32) uint8_t u[64];
        if (fmt == kFmtS2_K) {
            const uint8_t* p = pl + g * 16;
            for (int i = 0; i < 16; ++i) {
                const uint8_t b = p[i];
                u[4 * i + 0] = (uint8_t)(b & 3u);
                u[4 * i + 1] = (uint8_t)((b >> 2) & 3u);
                u[4 * i + 2] = (uint8_t)((b >> 4) & 3u);
                u[4 * i + 3] = (uint8_t)(b >> 6);
            }
            isum = avx2_dot_u8_i8_64(_mm256_load_si256(reinterpret_cast<const __m256i*>(u)),
                                     _mm256_load_si256(reinterpret_cast<const __m256i*>(u + 32)), qs);
            return sk_term(eff, x_d, isum, 1.5f, x_s);
        } else if (fmt == kFmtS3_K) {
            const uint8_t* p = pl + g * 16;
            const uint8_t* hp = pl + 64 + g * 8;
            for (int i = 0; i < 8; ++i) {
                const uint8_t b0 = p[2 * i], b1 = p[2 * i + 1], h = hp[i];
                u[8 * i + 0] = (uint8_t)((b0 & 3u) | ((h & 1u) << 2));
                u[8 * i + 1] = (uint8_t)(((b0 >> 2) & 3u) | (((h >> 1) & 1u) << 2));
                u[8 * i + 2] = (uint8_t)(((b0 >> 4) & 3u) | (((h >> 2) & 1u) << 2));
                u[8 * i + 3] = (uint8_t)((b0 >> 6) | (((h >> 3) & 1u) << 2));
                u[8 * i + 4] = (uint8_t)((b1 & 3u) | (((h >> 4) & 1u) << 2));
                u[8 * i + 5] = (uint8_t)(((b1 >> 2) & 3u) | (((h >> 5) & 1u) << 2));
                u[8 * i + 6] = (uint8_t)(((b1 >> 4) & 3u) | (((h >> 6) & 1u) << 2));
                u[8 * i + 7] = (uint8_t)((b1 >> 6) | (((h >> 7) & 1u) << 2));
            }
            isum = avx2_dot_u8_i8_64(_mm256_load_si256(reinterpret_cast<const __m256i*>(u)),
                                     _mm256_load_si256(reinterpret_cast<const __m256i*>(u + 32)), qs);
            return sk_term(eff, x_d, isum, 3.5f, x_s);
        } else if (fmt == kFmtS5_K) {
            const uint8_t* p = pl + g * 32;
            const uint8_t* hp = pl + 128 + g * 8;
            for (int i = 0; i < 8; ++i) {
                const uint8_t h = hp[i];
                for (int k = 0; k < 4; ++k) {
                    const uint8_t b = p[4 * i + k];
                    u[8 * i + 2 * k + 0] = (uint8_t)((b & 0xFu) | (((h >> (2 * k)) & 1u) << 4));
                    u[8 * i + 2 * k + 1] = (uint8_t)((b >> 4) | (((h >> (2 * k + 1)) & 1u) << 4));
                }
            }
            isum = avx2_dot_u8_i8_64(_mm256_load_si256(reinterpret_cast<const __m256i*>(u)),
                                     _mm256_load_si256(reinterpret_cast<const __m256i*>(u + 32)), qs);
            return sk_term(eff, x_d, isum, 15.5f, x_s);
        } else if (fmt == kFmtS6_K) {
            const uint8_t* p = pl + g * 32;
            const uint8_t* hp = pl + 128 + g * 16;
            for (int i = 0; i < 16; ++i) {
                const uint8_t h = hp[i], b0 = p[2 * i], b1 = p[2 * i + 1];
                u[4 * i + 0] = (uint8_t)((b0 & 0xFu) | ((h & 3u) << 4));
                u[4 * i + 1] = (uint8_t)((b0 >> 4) | (((h >> 2) & 3u) << 4));
                u[4 * i + 2] = (uint8_t)((b1 & 0xFu) | (((h >> 4) & 3u) << 4));
                u[4 * i + 3] = (uint8_t)((b1 >> 4) | ((h >> 6) << 4));
            }
            isum = avx2_dot_u8_i8_64(_mm256_load_si256(reinterpret_cast<const __m256i*>(u)),
                                     _mm256_load_si256(reinterpret_cast<const __m256i*>(u + 32)), qs);
            return sk_term(eff, x_d, isum, 31.5f, x_s);
        }
    }
    return 0.0f;
}

// ---- whole-row S_K kernels: SIMD unpack of the 2/3/5/6-bit payloads (the per-group scalar unpack above is ~8x slower), weights unpacked once for T tokens ----
namespace {

// 16 bytes holding four 2-bit fields each -> 64 bytes: out[4*i + k] = (b[i] >> 2k) & 3
inline void unpack2_x16(__m128i v, __m128i o[4]) noexcept {
    const __m128i m3 = _mm_set1_epi8(3);
    const __m128i t0 = _mm_and_si128(v, m3);
    const __m128i t1 = _mm_and_si128(_mm_srli_epi16(v, 2), m3);
    const __m128i t2 = _mm_and_si128(_mm_srli_epi16(v, 4), m3);
    const __m128i t3 = _mm_and_si128(_mm_srli_epi16(v, 6), m3);
    const __m128i a_lo = _mm_unpacklo_epi8(t0, t1), a_hi = _mm_unpackhi_epi8(t0, t1);
    const __m128i b_lo = _mm_unpacklo_epi8(t2, t3), b_hi = _mm_unpackhi_epi8(t2, t3);
    o[0] = _mm_unpacklo_epi16(a_lo, b_lo);
    o[1] = _mm_unpackhi_epi16(a_lo, b_lo);
    o[2] = _mm_unpacklo_epi16(a_hi, b_hi);
    o[3] = _mm_unpackhi_epi16(a_hi, b_hi);
}

// 8 bytes of bits -> 64 bytes: out[n] = value if bit (n & 7) of byte (n >> 3) is set, else 0
inline void expand_bits_64(const uint8_t* h8, int value, __m256i& r0, __m256i& r1) noexcept {
    int32_t w0, w1;
    std::memcpy(&w0, h8, 4);
    std::memcpy(&w1, h8 + 4, 4);
    const __m256i mask = _mm256_setr_epi8(1, 2, 4, 8, 16, 32, 64, (char)128, 1, 2, 4, 8, 16, 32, 64, (char)128,
                                          1, 2, 4, 8, 16, 32, 64, (char)128, 1, 2, 4, 8, 16, 32, 64, (char)128);
    const __m256i idx = _mm256_setr_epi8(0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1,
                                         2, 2, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 3, 3);
    const __m256i val = _mm256_set1_epi8((char)value);
    const __m256i s0 = _mm256_shuffle_epi8(_mm256_set1_epi32(w0), idx);
    const __m256i s1 = _mm256_shuffle_epi8(_mm256_set1_epi32(w1), idx);
    r0 = _mm256_and_si256(_mm256_cmpeq_epi8(_mm256_and_si256(s0, mask), mask), val);
    r1 = _mm256_and_si256(_mm256_cmpeq_epi8(_mm256_and_si256(s1, mask), mask), val);
}

inline void nibbles_64(const uint8_t* p, __m256i& u0, __m256i& u1) noexcept {   // 32 bytes -> 64 bytes: out[2m] = lo(p[m]), out[2m+1] = hi(p[m])
    const __m128i m0f = _mm_set1_epi8(0x0F);
    const __m128i v0 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(p));
    const __m128i v1 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(p + 16));
    const __m128i lo0 = _mm_and_si128(v0, m0f), hi0 = _mm_and_si128(_mm_srli_epi16(v0, 4), m0f);
    const __m128i lo1 = _mm_and_si128(v1, m0f), hi1 = _mm_and_si128(_mm_srli_epi16(v1, 4), m0f);
    u0 = _mm256_set_m128i(_mm_unpackhi_epi8(lo0, hi0), _mm_unpacklo_epi8(lo0, hi0));
    u1 = _mm256_set_m128i(_mm_unpackhi_epi8(lo1, hi1), _mm_unpacklo_epi8(lo1, hi1));
}

// The unsigned weights (0 .. 2^bits-1) of group g of an S_K payload as 64 bytes; `cst` = the offset subtracted by the formula (half range).
inline bool sk_unpack_group(int fmt, const uint8_t* pl, int g, __m256i& u0, __m256i& u1, float& cst) noexcept {
    if (fmt == kFmtS2_K) {
        __m128i o[4];
        unpack2_x16(_mm_loadu_si128(reinterpret_cast<const __m128i*>(pl + g * 16)), o);
        u0 = _mm256_set_m128i(o[1], o[0]);
        u1 = _mm256_set_m128i(o[3], o[2]);
        cst = 1.5f;
        return true;
    } else if (fmt == kFmtS3_K) {
        __m128i o[4];
        unpack2_x16(_mm_loadu_si128(reinterpret_cast<const __m128i*>(pl + g * 16)), o);
        __m256i h0, h1;
        expand_bits_64(pl + 64 + g * 8, 4, h0, h1);
        u0 = _mm256_or_si256(_mm256_set_m128i(o[1], o[0]), h0);
        u1 = _mm256_or_si256(_mm256_set_m128i(o[3], o[2]), h1);
        cst = 3.5f;
        return true;
    } else if (fmt == kFmtS4_K) {
        nibbles_64(pl + g * 32, u0, u1);
        cst = 7.5f;
        return true;
    } else if (fmt == kFmtS5_K) {
        nibbles_64(pl + g * 32, u0, u1);
        __m256i h0, h1;
        expand_bits_64(pl + 128 + g * 8, 16, h0, h1);
        u0 = _mm256_or_si256(u0, h0);
        u1 = _mm256_or_si256(u1, h1);
        cst = 15.5f;
        return true;
    } else if (fmt == kFmtS6_K) {
        nibbles_64(pl + g * 32, u0, u1);
        __m128i o[4];
        unpack2_x16(_mm_loadu_si128(reinterpret_cast<const __m128i*>(pl + 128 + g * 16)), o);
        u0 = _mm256_or_si256(u0, _mm256_slli_epi16(_mm256_set_m128i(o[1], o[0]), 4));
        u1 = _mm256_or_si256(u1, _mm256_slli_epi16(_mm256_set_m128i(o[3], o[2]), 4));
        cst = 31.5f;
        return true;
    }
    return false;
}

inline float f16_scale(const uint8_t* b) noexcept {
    uint16_t h;
    std::memcpy(&h, b, 2);
    return _mm_cvtss_f32(_mm_cvtph_ps(_mm_cvtsi32_si128((int)h)));
}

// groups [0, n_groups) of a row given as a list of super blocks, accumulated into acc[t]; xg0 = index of the activation block of the first group
inline bool sk_accumulate(const uint8_t* row, int fmt, int sb_bytes, int sb_first, int sb_count, int xg0,
                          const uint8_t* const* xq, int T, float* acc) noexcept {
    for (int s = 0; s < sb_count; ++s) {
        const uint8_t* b = row + (size_t)(sb_first + s) * sb_bytes;
        const float Dsc = f16_scale(b);
        for (int sg = 0; sg < 4; ++sg) {
            __m256i u0, u1;
            float cst;
            if (!sk_unpack_group(fmt, b + 6, sg, u0, u1, cst)) return false;
            const float eff = Dsc * ((float)b[2 + sg] * (1.0f / 255.0f));
            const int xg = xg0 + s * 4 + sg;
            for (int t = 0; t < T; ++t) {
                const uint8_t* xb = xq[t] + (size_t)xg * 80;
                float x_d, x_s;
                std::memcpy(&x_d, xb + 64, 4);
                std::memcpy(&x_s, xb + 68, 4);
                const int32_t isum = avx2_dot_u8_i8_64(u0, u1, reinterpret_cast<const int8_t*>(xb));
                acc[t] = add_nofuse(acc[t], sk_term(eff, x_d, isum, cst, x_s));
            }
        }
    }
    return true;
}

}  // namespace

bool dot_sk_row_multi(const uint8_t* row, int fmt, int sb_bytes, int n_sb, const uint8_t* const* xq, int T, float* out) noexcept {
    if (T < 1 || T > 16 || fmt == kFmtS8_K || fmt == kFmtS1_5) return false;
    float acc[16];
    for (int t = 0; t < T; ++t) acc[t] = 0.0f;
    if (!sk_accumulate(row, fmt, sb_bytes, 0, n_sb, 0, xq, T, acc)) return false;
    for (int t = 0; t < T; ++t) out[t] = acc[t];
    return true;
}

bool dot_sk_pair_multi(const uint8_t* row, int fmt, int sb_bytes, const uint8_t* const* xq, int T, float* out0, float* out1) noexcept {
    if (T < 1 || T > 16 || fmt == kFmtS8_K || fmt == kFmtS1_5) return false;
    // 9 super blocks x 4 groups = 36 groups: groups 0..17 -> row 2r (activation blocks 0..17), groups 18..35 -> row 2r+1 (blocks 0..17).
    // The 18 groups of a row are 4.5 super blocks: the middle super block (index 4) is split between the two rows.
    float a0[16], a1[16];
    for (int t = 0; t < T; ++t) { a0[t] = 0.0f; a1[t] = 0.0f; }
    for (int s = 0; s < 9; ++s) {
        const uint8_t* b = row + (size_t)s * sb_bytes;
        const float Dsc = f16_scale(b);
        for (int sg = 0; sg < 4; ++sg) {
            const int sub_g = s * 4 + sg;
            __m256i u0, u1;
            float cst;
            if (!sk_unpack_group(fmt, b + 6, sg, u0, u1, cst)) return false;
            const float eff = Dsc * ((float)b[2 + sg] * (1.0f / 255.0f));
            const int xg = sub_g < 18 ? sub_g : sub_g - 18;
            float* acc = sub_g < 18 ? a0 : a1;
            for (int t = 0; t < T; ++t) {
                const uint8_t* xb = xq[t] + (size_t)xg * 80;
                float x_d, x_s;
                std::memcpy(&x_d, xb + 64, 4);
                std::memcpy(&x_s, xb + 68, 4);
                const int32_t isum = avx2_dot_u8_i8_64(u0, u1, reinterpret_cast<const int8_t*>(xb));
                acc[t] = add_nofuse(acc[t], sk_term(eff, x_d, isum, cst, x_s));
            }
        }
    }
    for (int t = 0; t < T; ++t) { out0[t] = a0[t]; out1[t] = a1[t]; }
    return true;
}

#else

bool dot_sk_row_multi(const uint8_t*, int, int, int, const uint8_t* const*, int, float*) noexcept { return false; }
bool dot_sk_pair_multi(const uint8_t*, int, int, const uint8_t* const*, int, float*, float*) noexcept { return false; }
float dot_sf4_row(const uint8_t*, const uint8_t*, float, const float*, int, const float*) noexcept { return 0.0f; }
float dot_mx8_row(const uint8_t*, const uint8_t*, const float*, int, const float*, const float*) noexcept { return 0.0f; }
float dot_sk_sub64(const uint8_t*, int, int, const int8_t*, float, float, float, const int8_t (*)[5]) noexcept { return 0.0f; }

#endif

}  // namespace strata::ink::host_avx2
