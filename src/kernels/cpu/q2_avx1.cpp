// src/kernels/cpu/q2_avx1.cpp - the Q2_0 expert rows for CPUs with AVX but without AVX2 (Intel Sandy / Ivy Bridge);
// the router dot for those CPUs is kq_avx1.cpp.
//
// Compiled with AVX only (SSSE3 integer ops, no FMA), so nothing here can fault on those CPUs; reached only when
// cpu_avx1_ok() says yes and cpu_avx2_ok() says no.  The Q2_0 rows give the scalar reference's bits
// (q2_0_gguf_rows_multi_scalar): the integer dot of a 32-value chunk is exact in any order, and the float steps are
// the scalar ones in the scalar order - acc += (d * scale) * dot, corr += d * hx, out = acc - corr.
#include "strata/kernels/cpu/expert.hpp"

#include <immintrin.h>

#include <cstring>

namespace strata::kernels::cpu {
namespace {

// IEEE half -> float, exact (the same values as the scalar reference's ldexp form, without the calls)
inline float h2f(const uint8_t* p) {
    uint16_t h;
    std::memcpy(&h, p, 2);
    const uint32_t sign = (uint32_t) (h & 0x8000) << 16, e = (h >> 10) & 0x1f, m = h & 0x3ff;
    float f;
    if (e == 0) {
        f = (float) m * 5.9604644775390625e-8f;          // m * 2^-24: exact (m < 2^10)
        uint32_t b;
        std::memcpy(&b, &f, 4);
        b |= sign;
        std::memcpy(&f, &b, 4);
        return f;
    }
    const uint32_t b = sign | (e == 0x1f ? 0x7f800000u | (m << 13) : ((e + 112) << 23) | (m << 13));
    std::memcpy(&f, &b, 4);
    return f;
}

inline int hsum_i32(__m128i v) {
    v = _mm_add_epi32(v, _mm_shuffle_epi32(v, _MM_SHUFFLE(1, 0, 3, 2)));
    v = _mm_add_epi32(v, _mm_shuffle_epi32(v, _MM_SHUFFLE(2, 3, 0, 1)));
    return _mm_cvtsi128_si32(v);
}

}  // namespace

void q2_0_gguf_rows_multi_avx1(const uint8_t* w, size_t row_bytes, int nblocks, const ActQ* const* a, int nt,
                               float* const* out, int r0, int r1) {
    const __m128i m3 = _mm_set1_epi8(3), ones = _mm_set1_epi16(1);
    for (int t0 = 0; t0 < nt; t0 += MAXT) {
        const int m = nt - t0 < MAXT ? nt - t0 : MAXT;
        for (int r = r0; r < r1; ++r) {
            const uint8_t* row = w + (size_t) r * row_bytes;
            float acc[MAXT], corr[MAXT];
            for (int t = 0; t < m; ++t) { acc[t] = 0.0f; corr[t] = 0.0f; }
            for (int b = 0; b < nblocks; ++b) {
                const uint8_t* blk = row + (size_t) b * 18;
                const float d = h2f(blk);
                // 16 code bytes (value i in byte i/4, bits 2*(i%4)) -> codes 0..63 in value order, 16 per register
                const __m128i x = _mm_loadu_si128((const __m128i*) (blk + 2));
                const __m128i c0 = _mm_and_si128(x, m3), c1 = _mm_and_si128(_mm_srli_epi16(x, 2), m3);
                const __m128i c2 = _mm_and_si128(_mm_srli_epi16(x, 4), m3), c3 = _mm_and_si128(_mm_srli_epi16(x, 6), m3);
                const __m128i a0 = _mm_unpacklo_epi8(c0, c1), a1 = _mm_unpacklo_epi8(c2, c3);
                const __m128i b0 = _mm_unpackhi_epi8(c0, c1), b1 = _mm_unpackhi_epi8(c2, c3);
                const __m128i v0 = _mm_unpacklo_epi16(a0, a1), v1 = _mm_unpackhi_epi16(a0, a1);   // values 0..31
                const __m128i v2 = _mm_unpacklo_epi16(b0, b1), v3 = _mm_unpackhi_epi16(b0, b1);   // values 32..63
                for (int t = 0; t < m; ++t) {
                    const ActQ& q = *a[t0 + t];
                    const int8_t* qb = q.q + b * 64;
                    // codes 0..3 x int8 -127..127: a pair sums to at most 762, so maddubs never saturates
                    const __m128i s0 = _mm_add_epi32(
                        _mm_madd_epi16(_mm_maddubs_epi16(v0, _mm_loadu_si128((const __m128i*) qb)), ones),
                        _mm_madd_epi16(_mm_maddubs_epi16(v1, _mm_loadu_si128((const __m128i*) (qb + 16))), ones));
                    const __m128i s1 = _mm_add_epi32(
                        _mm_madd_epi16(_mm_maddubs_epi16(v2, _mm_loadu_si128((const __m128i*) (qb + 32))), ones),
                        _mm_madd_epi16(_mm_maddubs_epi16(v3, _mm_loadu_si128((const __m128i*) (qb + 48))), ones));
                    acc[t] += (d * q.scale[2 * b]) * (float) hsum_i32(s0);
                    corr[t] += d * q.hx[2 * b];
                    acc[t] += (d * q.scale[2 * b + 1]) * (float) hsum_i32(s1);
                    corr[t] += d * q.hx[2 * b + 1];
                }
            }
            for (int t = 0; t < m; ++t) out[t0 + t][r] = acc[t] - corr[t];
        }
    }
}

}  // namespace strata::kernels::cpu
