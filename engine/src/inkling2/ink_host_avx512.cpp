// engine/src/inkling2/ink_host_avx512.cpp - Per-TU AVX-512 host dot kernels for InkHybridEngine (SF4, MX8, S1.5..S8_K sub-groups)
// Compiled ONLY with -mavx512f -mavx512bw -mavx512vl -mavx512dq (+ -mavx2 -mfma -mf16c; MSVC /arch:AVX512) when STRATA_ISA_FLOOR allows;
// guarded at runtime by strata::quant::caps().avx512 so a CPU without AVX-512 never executes one of these instructions.
// No VNNI: u8 x s8 goes through VPMADDUBSW + VPMADDWD, which is exact for these ranges (u <= 63 for S6, |p| <= 128 for S8), so the integer sums
// equal the AVX2 / plain C++ ones and the S_K results equal the AVX2 ones to the last bit (same float order). SF4 and MX8 sum in 16 lanes, so
// they differ from AVX2 by float rounding only.
// MIT License - Copyright (c) 2026 Strata contributors
// Nothing in this file runs unless available() (ink_host_avx512_gate.cpp, a baseline-flag TU: a compiler may use AVX-512 anywhere in a TU built
// with these flags, so even the gate must not live here) said yes.
#include "ink_host_avx512.h"

#include <cstring>

#if defined(INK_BUILD_HOST_AVX512) && defined(__AVX512F__) && defined(__AVX512BW__) && defined(__AVX512VL__) && defined(__AVX512DQ__)
#include <immintrin.h>
#define INK_TU_HAS_AVX512 1
#endif

namespace strata::ink::host_avx512 {

#if defined(INK_TU_HAS_AVX512)

namespace {

// Format constants matching InkFmt in ink_cuda.cuh
constexpr int kFmtS1_5 = 1;
constexpr int kFmtS2_K = 2;
constexpr int kFmtS3_K = 3;
constexpr int kFmtS4_K = 4;
constexpr int kFmtS5_K = 5;
constexpr int kFmtS6_K = 6;
constexpr int kFmtS8_K = 8;

// sum of 64 u8 x s8 products (u8 <= 127 per pair-safe range, see the header comment)
inline int32_t dot_u8_i8_64(__m512i u, const int8_t* qs) noexcept {
    const __m512i q = _mm512_loadu_si512(qs);
    const __m512i p = _mm512_madd_epi16(_mm512_maddubs_epi16(u, q), _mm512_set1_epi16(1));
    return _mm512_reduce_add_epi32(p);
}

// 16 bytes of 2-bit fields -> 64 u8 (element 4i + k = bits 2k..2k+1 of byte i)
inline __m512i unpack_2bit_16(const uint8_t* p) noexcept {
    const __m512i w = _mm512_cvtepu8_epi32(_mm_loadu_si128(reinterpret_cast<const __m128i*>(p)));
    const __m512i m3 = _mm512_set1_epi32(3);
    const __m512i q0 = _mm512_and_si512(w, m3);
    const __m512i q1 = _mm512_and_si512(_mm512_srli_epi32(w, 2), m3);
    const __m512i q2 = _mm512_and_si512(_mm512_srli_epi32(w, 4), m3);
    const __m512i q3 = _mm512_srli_epi32(w, 6);
    return _mm512_or_si512(_mm512_or_si512(q0, _mm512_slli_epi32(q1, 8)),
                           _mm512_or_si512(_mm512_slli_epi32(q2, 16), _mm512_slli_epi32(q3, 24)));
}

// 32 bytes of nibbles -> 64 u8 (element 2i = low nibble of byte i, 2i + 1 = high nibble)
inline __m512i unpack_nibbles_32(const uint8_t* p) noexcept {
    const __m512i w = _mm512_cvtepu8_epi16(_mm256_loadu_si256(reinterpret_cast<const __m256i*>(p)));
    return _mm512_or_si512(_mm512_and_si512(w, _mm512_set1_epi16(0x000F)),
                           _mm512_slli_epi16(_mm512_srli_epi16(w, 4), 8));
}

// 8 bytes = 64 flag bits (element j <- bit j) -> `value` in the flagged bytes, 0 elsewhere
inline __m512i flag_bytes_8(const uint8_t* p, int value) noexcept {
    uint64_t m;
    std::memcpy(&m, p, 8);
    return _mm512_maskz_mov_epi8(static_cast<__mmask64>(m), _mm512_set1_epi8(static_cast<char>(value)));
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

// 16 F8_E4M3FN codes -> 16 floats, exactly the table of e4m3_lut_h (0x7F / 0xFF, the NaN codes, read as 0)
inline __m512 e4m3_to_ps(__m128i codes) noexcept {
    const __m512i c = _mm512_cvtepu8_epi32(codes);
    const __m512i sign = _mm512_slli_epi32(_mm512_and_si512(c, _mm512_set1_epi32(0x80)), 24);
    const __m512i e = _mm512_and_si512(_mm512_srli_epi32(c, 3), _mm512_set1_epi32(15));
    const __m512i m = _mm512_and_si512(c, _mm512_set1_epi32(7));
    const __m512 normal = _mm512_castsi512_ps(_mm512_or_si512(_mm512_slli_epi32(_mm512_add_epi32(e, _mm512_set1_epi32(120)), 23),
                                                              _mm512_slli_epi32(m, 20)));
    const __m512 sub = _mm512_mul_ps(_mm512_cvtepi32_ps(m), _mm512_set1_ps(1.0f / 512.0f));
    __m512 v = _mm512_mask_blend_ps(_mm512_cmpeq_epi32_mask(e, _mm512_setzero_si512()), normal, sub);
    const __mmask16 nan = _mm512_cmpeq_epi32_mask(_mm512_and_si512(c, _mm512_set1_epi32(0x7F)), _mm512_set1_epi32(0x7F));
    v = _mm512_mask_blend_ps(nan, v, _mm512_setzero_ps());
    return _mm512_castsi512_ps(_mm512_or_si512(_mm512_castps_si512(v), sign));
}

}  // namespace

float dot_sf4_row(const uint8_t* p, const uint8_t* sc, float w_outer, const float* x, int cols,
                  const float* e4_lut) noexcept {
    const int nb = cols >> 4;
    const __m128i lut2x  = _mm_setr_epi8(0, 1, 2, 3, 4, 6, 8, 12, 0, -1, -2, -3, -4, -6, -8, -12);
    const __m128i mask0f = _mm_set1_epi8(0x0F);
    __m512 v0 = _mm512_setzero_ps();
    __m512 v1 = _mm512_setzero_ps();
    int b = 0;
    for (; b + 2 <= nb; b += 2) {  // two 16-weight blocks per 16 bytes
        const __m128i raw = _mm_loadu_si128(reinterpret_cast<const __m128i*>(p + b * 8));
        const __m128i lo  = _mm_and_si128(raw, mask0f);
        const __m128i hi  = _mm_and_si128(_mm_srli_epi16(raw, 4), mask0f);
        const __m128i wa  = _mm_shuffle_epi8(lut2x, _mm_unpacklo_epi8(lo, hi));
        const __m128i wb  = _mm_shuffle_epi8(lut2x, _mm_unpackhi_epi8(lo, hi));
        const __m512 fa = _mm512_cvtepi32_ps(_mm512_cvtepi8_epi32(wa));
        const __m512 fb = _mm512_cvtepi32_ps(_mm512_cvtepi8_epi32(wb));
        const float* xb = x + b * 16;
        v0 = _mm512_fmadd_ps(_mm512_mul_ps(fa, _mm512_loadu_ps(xb)),      _mm512_set1_ps(0.5f * e4_lut[sc[b]]),     v0);
        v1 = _mm512_fmadd_ps(_mm512_mul_ps(fb, _mm512_loadu_ps(xb + 16)), _mm512_set1_ps(0.5f * e4_lut[sc[b + 1]]), v1);
    }
    for (; b < nb; ++b) {
        const __m128i raw = _mm_loadl_epi64(reinterpret_cast<const __m128i*>(p + b * 8));
        const __m128i lo  = _mm_and_si128(raw, mask0f);
        const __m128i hi  = _mm_and_si128(_mm_srli_epi16(raw, 4), mask0f);
        const __m128i w8  = _mm_shuffle_epi8(lut2x, _mm_unpacklo_epi8(lo, hi));
        const __m512 f = _mm512_cvtepi32_ps(_mm512_cvtepi8_epi32(w8));
        v0 = _mm512_fmadd_ps(_mm512_mul_ps(f, _mm512_loadu_ps(x + b * 16)), _mm512_set1_ps(0.5f * e4_lut[sc[b]]), v0);
    }
    return _mm512_reduce_add_ps(_mm512_add_ps(v0, v1)) * w_outer;
}

float dot_mx8_row(const uint8_t* w, const uint8_t* sc, const float* x, int cols,
                  const float* /*e4_lut*/, const float* u8_lut) noexcept {
    const int nb = cols >> 5;
    __m512 v0 = _mm512_setzero_ps();
    __m512 v1 = _mm512_setzero_ps();
    for (int b = 0; b < nb; ++b) {
        const uint8_t* wb = w + b * 32;
        const float* xb = x + b * 32;
        const __m512 w0 = e4m3_to_ps(_mm_loadu_si128(reinterpret_cast<const __m128i*>(wb)));
        const __m512 w1 = e4m3_to_ps(_mm_loadu_si128(reinterpret_cast<const __m128i*>(wb + 16)));
        const __m512 acc = _mm512_fmadd_ps(w1, _mm512_loadu_ps(xb + 16), _mm512_mul_ps(w0, _mm512_loadu_ps(xb)));
        const __m512 s = _mm512_set1_ps(u8_lut[sc[b]]);
        if (b & 1) v1 = _mm512_fmadd_ps(acc, s, v1);
        else       v0 = _mm512_fmadd_ps(acc, s, v0);
    }
    return _mm512_reduce_add_ps(_mm512_add_ps(v0, v1));
}

float dot_sk_sub64(const uint8_t* pl, int fmt, int g, const int8_t* qs, float x_d, float x_s,
                   float eff, const int8_t (*s15_lut)[5]) noexcept {
    int32_t isum = 0;
    switch (fmt) {
    case kFmtS1_5: {
        alignas(64) int8_t tr[64];
        s15_unpack64_local(pl, g, tr, s15_lut);
        const __m512i u = _mm512_add_epi8(_mm512_load_si512(tr), _mm512_set1_epi8(1));
        isum = dot_u8_i8_64(u, qs);
        return eff * (x_d * (float)isum - x_s);
    }
    case kFmtS2_K:
        isum = dot_u8_i8_64(unpack_2bit_16(pl + g * 16), qs);
        return eff * (x_d * (float)isum - 1.5f * x_s);
    case kFmtS3_K:
        isum = dot_u8_i8_64(_mm512_or_si512(unpack_2bit_16(pl + g * 16), flag_bytes_8(pl + 64 + g * 8, 4)), qs);
        return eff * (x_d * (float)isum - 3.5f * x_s);
    case kFmtS4_K:
        isum = dot_u8_i8_64(unpack_nibbles_32(pl + g * 32), qs);
        return eff * (x_d * (float)isum - 7.5f * x_s);
    case kFmtS5_K:
        isum = dot_u8_i8_64(_mm512_or_si512(unpack_nibbles_32(pl + g * 32), flag_bytes_8(pl + 128 + g * 8, 16)), qs);
        return eff * (x_d * (float)isum - 15.5f * x_s);
    case kFmtS6_K: {
        const __m512i hi = _mm512_slli_epi32(unpack_2bit_16(pl + 128 + g * 16), 4);  // fields are 0..3 in every byte: no cross-byte bits
        isum = dot_u8_i8_64(_mm512_or_si512(unpack_nibbles_32(pl + g * 32), hi), qs);
        return eff * (x_d * (float)isum - 31.5f * x_s);
    }
    case kFmtS8_K: {
        // 256-bit |p| x sign(q, p): the S8_K rows are read at memory speed, and a 512-bit reduction per sub-group costs more than it saves (measured)
        const int8_t* p = reinterpret_cast<const int8_t*>(pl + g * 64);
        const __m256i p0 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(p));
        const __m256i p1 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(p + 32));
        const __m256i q0 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(qs));
        const __m256i q1 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(qs + 32));
        const __m256i ones = _mm256_set1_epi16(1);
        const __m256i a0 = _mm256_madd_epi16(_mm256_maddubs_epi16(_mm256_abs_epi8(p0), _mm256_sign_epi8(q0, p0)), ones);
        const __m256i a1 = _mm256_madd_epi16(_mm256_maddubs_epi16(_mm256_abs_epi8(p1), _mm256_sign_epi8(q1, p1)), ones);
        const __m256i s = _mm256_add_epi32(a0, a1);
        __m128i h = _mm_add_epi32(_mm256_castsi256_si128(s), _mm256_extracti128_si256(s, 1));
        h = _mm_add_epi32(h, _mm_shuffle_epi32(h, 0x4E));
        h = _mm_add_epi32(h, _mm_shuffle_epi32(h, 0xB1));
        isum = _mm_cvtsi128_si32(h);
        return eff * x_d * (float)isum;
    }
    default:
        return 0.0f;
    }
}

#else

float dot_sf4_row(const uint8_t*, const uint8_t*, float, const float*, int, const float*) noexcept { return 0.0f; }
float dot_mx8_row(const uint8_t*, const uint8_t*, const float*, int, const float*, const float*) noexcept { return 0.0f; }
float dot_sk_sub64(const uint8_t*, int, int, const int8_t*, float, float, float, const int8_t (*)[5]) noexcept { return 0.0f; }

#endif

}  // namespace strata::ink::host_avx512
