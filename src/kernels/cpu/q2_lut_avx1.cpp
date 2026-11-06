// src/kernels/cpu/q2_lut_avx1.cpp - the table-lookup Q2_0 rows (include/strata/kernels/cpu/q2_lut.hpp), built
// -msse4.2 -mavx -mf16c -mno-avx2 -mno-fma for Ivy Bridge.  q2_lut_snb.cpp includes this file again with
// STRATA_Q2_LUT_SNB defined, built without F16C, for Sandy Bridge: the same kernel converting the fp16 row scales in
// software, as rows_nof16c, plus build_tables (which needs no F16C, so it exists once, there).
//
// Per tile of 16 rows and 64-value block, per 32-value activation chunk, 16 lookups (2 per index vector, 8 index
// vectors): pshufb the A plane and the B plane of the column pair's table with the 16 rows' 4-bit indices, sum four
// lookups in bytes (A <= 188, B <= 124: no overflow), then punpck + pmaddubsw(.., [1, 32]) gives int16 B + 32A per
// row.  pshufb, punpck and paddb issue on ports 1/5; the regroup's pmaddubsw and the widening (pmaddwd against
// [1, 0] / [0, 1]: even rows / odd rows) on port 0, so the three ports share the work (a Sandy Bridge E5-2603:
// 7.4 products per cycle, against 2.4 for q2_avx1.cpp's multiply-add rows).  The int16 accumulators start at
// -16 * 762, the table offset of 16 lookups, so it cancels in integers (|sum| <= 12368).
//
// A row's result depends only on its own lanes: the same float operations in the same order whatever tile group
// (1 or 2 tiles), token group (1 or 4 tokens) or row range it is computed in.
#include "q2_lut_kernels.hpp"

#include <immintrin.h>

#include <cstddef>
#include <cstring>

#if defined(__AVX2__) || defined(__FMA__)
#error "q2_lut_avx1.cpp targets Sandy / Ivy Bridge: build it without AVX2 and FMA (see CMakeLists.txt)"
#endif
#if defined(__GNUC__) && (defined(STRATA_Q2_LUT_SNB) == defined(__F16C__))
#error "q2_lut_avx1.cpp: F16C for the Ivy Bridge build, none for the Sandy Bridge one (STRATA_Q2_LUT_SNB)"
#endif

#if defined(__GNUC__) || defined(__clang__)
// Keeps an accumulator chain in registers: without it GCC reassociates the 16 lookups of a chunk into a tree and
// spills the partial sums (~95 stack moves per block).
#define Q2LUT_KEEP(x, y) asm("" : "+x"(x), "+x"(y))
#define Q2LUT_UNROLL(n) _Pragma(#n)
#else
#define Q2LUT_KEEP(x, y) (void) 0
#define Q2LUT_UNROLL(n)
#endif

#ifdef STRATA_Q2_LUT_SNB
#define Q2LUT_ROWS rows_nof16c
#else
#define Q2LUT_ROWS rows_f16c
#endif

namespace strata::kernels::cpu::q2lut {
namespace {

// Four rows' fp16 scales at p as floats.  Without F16C: (h & 0x7fff) << 13 puts the half's exponent and mantissa in
// float position, a float 2^112 times too small; times 2^112 is exact for normal and subnormal halves alike (row
// scales are never inf / NaN), and the sign goes back separately.
inline __m128 load_h4(const uint8_t* p) {
#ifdef STRATA_Q2_LUT_SNB
    const __m128i h = _mm_unpacklo_epi16(_mm_loadl_epi64((const __m128i*) p), _mm_setzero_si128());
    const __m128i mag = _mm_slli_epi32(_mm_and_si128(h, _mm_set1_epi32(0x7fff)), 13);
    const __m128i sign = _mm_slli_epi32(_mm_and_si128(h, _mm_set1_epi32(0x8000)), 16);
    const __m128 f = _mm_mul_ps(_mm_castsi128_ps(mag), _mm_castsi128_ps(_mm_set1_epi32(0x77800000)));   // 2^112
    return _mm_or_ps(f, _mm_castsi128_ps(sign));
#else
    return _mm_cvtph_ps(_mm_loadl_epi64((const __m128i*) p));
#endif
}

// The int32 vectors after the even / odd widening hold rows [0,2,4,6] [1,3,5,7] [8,10,12,14] [9,11,13,15]: the
// scales of rows 4j..4j+3 in that order.
inline __m128 even_odd(__m128 lo, __m128 hi, bool odd) {
    return odd ? _mm_shuffle_ps(lo, hi, _MM_SHUFFLE(3, 1, 3, 1)) : _mm_shuffle_ps(lo, hi, _MM_SHUFFLE(2, 0, 2, 0));
}

// One lookup of column pair `tp` for TL tiles: the A and B bytes into ab[l][0..1] (assigned on a group's first).
template <int TL>
inline void look(const uint8_t* tp, const __m128i* n, __m128i (*ab)[2], bool first) {
    const __m128i ta = _mm_load_si128((const __m128i*) tp);
    const __m128i tb = _mm_load_si128((const __m128i*) (tp + 16));
    for (int l = 0; l < TL; ++l) {
        const __m128i xa = _mm_shuffle_epi8(ta, n[l]);
        const __m128i xb = _mm_shuffle_epi8(tb, n[l]);
        ab[l][0] = first ? xa : _mm_add_epi8(ab[l][0], xa);
        ab[l][1] = first ? xb : _mm_add_epi8(ab[l][1], xb);
        Q2LUT_KEEP(ab[l][0], ab[l][1]);
    }
}

// One group of 4 lookups (2 index vectors x 2 nibbles), then the regroup into the int16 accumulators.  A macro, not
// a lambda: a lambda here cost the unrolled 1-token kernel 10% on Sandy Bridge (GCC 13).
#define Q2LUT_GROUP(g)                                                                                            \
    do {                                                                                                          \
        __m128i ab[NT][TL][2];                                                                                    \
        Q2LUT_UNROLL(GCC unroll 2)                                                                                \
        for (int u = 0; u < 2; ++u) {                                                                             \
            const int v = 8 * h + 2 * (g) + u;                                                                    \
            __m128i n0[TL], n1[TL];                                                                               \
            for (int l = 0; l < TL; ++l) {                                                                        \
                const __m128i idx = _mm_load_si128((const __m128i*) (rec[l] + 2 * TR + 16 * v));                  \
                n0[l] = _mm_and_si128(idx, m4);                                                                   \
                n1[l] = _mm_and_si128(_mm_srli_epi16(idx, 4), m4);                                                \
            }                                                                                                     \
            for (int t = 0; t < NT; ++t) {                                                                        \
                const uint8_t* tp = tbl[t] + (size_t) (b * 32 + 2 * v) * TBL;                                     \
                look<TL>(tp, n0, ab[t], u == 0);                                                                  \
                look<TL>(tp + TBL, n1, ab[t], false);                                                             \
            }                                                                                                     \
        }                                                                                                         \
        for (int t = 0; t < NT; ++t)                                                                              \
            for (int l = 0; l < TL; ++l) {                                                                        \
                const __m128i lo = _mm_maddubs_epi16(_mm_unpacklo_epi8(ab[t][l][1], ab[t][l][0]), w132);          \
                const __m128i hi = _mm_maddubs_epi16(_mm_unpackhi_epi8(ab[t][l][1], ab[t][l][0]), w132);          \
                acc[t][l][0] = _mm_add_epi16(acc[t][l][0], lo);                                                   \
                acc[t][l][1] = _mm_add_epi16(acc[t][l][1], hi);                                                   \
                Q2LUT_KEEP(acc[t][l][0], acc[t][l][1]);                                                           \
            }                                                                                                     \
    } while (0)

// Tiles [tile0, tile1) (a multiple of TL of them), NT tokens; writes only rows in [r0, r1).
template <int NT, int TL>
void tiles(const uint8_t* mir, int nblk, const uint8_t* const* tbl, const ActQ* const* a, float* const* out,
           int tile0, int tile1, int r0, int r1) {
    const __m128i m4 = _mm_set1_epi8(0x0F);
    const __m128i w132 = _mm_set1_epi16(32 << 8 | 1);   // pmaddubsw bytes [B, A] -> B + 32A
    const __m128i ev = _mm_set1_epi32(1), od = _mm_set1_epi32(1 << 16);
    for (int tile = tile0; tile < tile1; tile += TL) {
        __m128 res[NT][TL][4];
        for (int t = 0; t < NT; ++t)
            for (int l = 0; l < TL; ++l)
                for (int j = 0; j < 4; ++j) res[t][l][j] = _mm_setzero_ps();
        for (int b = 0; b < nblk; ++b) {
            const uint8_t* rec[TL];
            for (int l = 0; l < TL; ++l) rec[l] = mir + ((size_t) (tile + l) * nblk + b) * REC;
            __m128 blk[NT][TL][4];   // sum over the block's two chunks of the chunk scale times the integer sum
            for (int t = 0; t < NT; ++t)
                for (int l = 0; l < TL; ++l)
                    for (int j = 0; j < 4; ++j) blk[t][l][j] = _mm_setzero_ps();
            Q2LUT_UNROLL(GCC unroll 2)
            for (int h = 0; h < 2; ++h) {
                __m128i acc[NT][TL][2];   // int16 B + 32A, rows 0..7 and 8..15, from -16 * 762
                for (int t = 0; t < NT; ++t)
                    for (int l = 0; l < TL; ++l) acc[t][l][0] = acc[t][l][1] = _mm_set1_epi16(-16 * 762);
                Q2LUT_UNROLL(GCC unroll 4)
                for (int g = 0; g < 4; ++g) Q2LUT_GROUP(g);
                for (int t = 0; t < NT; ++t) {
                    const __m128 s = _mm_set1_ps(a[t]->scale[2 * b + h]);
                    for (int l = 0; l < TL; ++l)
                        for (int j = 0; j < 2; ++j) {
                            const __m128i x = acc[t][l][j];
                            blk[t][l][2 * j] =
                                _mm_add_ps(blk[t][l][2 * j], _mm_mul_ps(_mm_cvtepi32_ps(_mm_madd_epi16(x, ev)), s));
                            blk[t][l][2 * j + 1] =
                                _mm_add_ps(blk[t][l][2 * j + 1], _mm_mul_ps(_mm_cvtepi32_ps(_mm_madd_epi16(x, od)), s));
                        }
                }
            }
            for (int l = 0; l < TL; ++l) {
                const __m128 d0 = load_h4(rec[l]), d1 = load_h4(rec[l] + 8);
                const __m128 d2 = load_h4(rec[l] + 16), d3 = load_h4(rec[l] + 24);
                const __m128 d[4] = {even_odd(d0, d1, false), even_odd(d0, d1, true), even_odd(d2, d3, false),
                                     even_odd(d2, d3, true)};
                for (int t = 0; t < NT; ++t) {
                    const __m128 c = _mm_set1_ps(a[t]->hx[2 * b] + a[t]->hx[2 * b + 1]);
                    for (int j = 0; j < 4; ++j)
                        res[t][l][j] = _mm_add_ps(res[t][l][j], _mm_mul_ps(d[j], _mm_sub_ps(blk[t][l][j], c)));
                }
            }
        }
        static constexpr int order[TR] = {0, 2, 4, 6, 1, 3, 5, 7, 8, 10, 12, 14, 9, 11, 13, 15};
        for (int t = 0; t < NT; ++t)
            for (int l = 0; l < TL; ++l) {
                alignas(16) float r[TR];
                for (int j = 0; j < 4; ++j) _mm_store_ps(r + 4 * j, res[t][l][j]);
                const int row0 = (tile + l) * TR;
                for (int i = 0; i < TR; ++i) {
                    const int row = row0 + order[i];
                    if (row >= r0 && row < r1) out[t][row] = r[i];
                }
            }
    }
}

}  // namespace

void Q2LUT_ROWS(const uint8_t* mir, int nblk, const uint8_t* const* tbl, const ActQ* const* a, int nt,
                float* const* out, int r0, int r1) {
    if (r0 >= r1) return;
    const int t0 = r0 / TR, t1 = (r1 + TR - 1) / TR;
    const int even = t0 + (t1 - t0) / 2 * 2;
    int t = 0;
    for (; t + 4 <= nt; t += 4) tiles<4, 1>(mir, nblk, tbl + t, a + t, out + t, t0, t1, r0, r1);
    for (; t < nt; ++t) {
        if (even > t0) tiles<1, 2>(mir, nblk, tbl + t, a + t, out + t, t0, even, r0, r1);
        if (even < t1) tiles<1, 1>(mir, nblk, tbl + t, a + t, out + t, even, t1, r0, r1);
    }
}

#ifdef STRATA_Q2_LUT_SNB
void build_tables(const int8_t* q, int k, uint8_t* tbl) {
    const __m128i c0 = _mm_setr_epi16(0, 1, 2, 3, 0, 1, 2, 3);
    const __m128i c1a = _mm_setr_epi16(0, 0, 0, 0, 1, 1, 1, 1), c1b = _mm_setr_epi16(2, 2, 2, 2, 3, 3, 3, 3);
    const __m128i off = _mm_set1_epi16(762), m31 = _mm_set1_epi16(31);
    for (int p = 0; p < k / 2; ++p) {
        const __m128i x0 = _mm_add_epi16(_mm_mullo_epi16(_mm_set1_epi16(q[2 * p]), c0), off);
        const __m128i q1 = _mm_set1_epi16(q[2 * p + 1]);
        const __m128i ta = _mm_add_epi16(x0, _mm_mullo_epi16(q1, c1a));   // T + 762, entries 0..7
        const __m128i tb = _mm_add_epi16(x0, _mm_mullo_epi16(q1, c1b));   // entries 8..15
        _mm_store_si128((__m128i*) (tbl + TBL * p), _mm_packus_epi16(_mm_srli_epi16(ta, 5), _mm_srli_epi16(tb, 5)));
        _mm_store_si128((__m128i*) (tbl + TBL * p + 16),
                        _mm_packus_epi16(_mm_and_si128(ta, m31), _mm_and_si128(tb, m31)));
    }
}
#endif

}  // namespace strata::kernels::cpu::q2lut
