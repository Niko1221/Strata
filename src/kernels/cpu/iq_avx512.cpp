// src/kernels/cpu/iq_avx512.cpp - plan v0.3 P6: the i-quant expert rows in 512-bit lanes, several tokens at once.
//
// ggml-cpu's x86 dot products for these formats are AVX2 and single-token: every token re-decodes the weights
// and a sign is applied with five instructions per 32 values.  Here 64 values are decoded once (8 or 16 grid
// lookups, one 64-bit sign mask, one vector of scales) and every token applies them with a masked subtract, a
// `maddubs` and a `madd`.  The arithmetic is ggml's (ggml-cpu/quants.c, the `_generic` references): integer sums
// per block, times d_x * d_y * the format's constant - only the order of the float additions differs.
//
// Formats: IQ2_XXS (16), IQ2_XS (17), IQ3_XXS (18), IQ1_S (19), IQ3_S (21), IQ2_S (22), IQ1_M (29).
#include "strata/kernels/cpu/iq_avx512.hpp"

#define GGML_COMMON_DECL_CPP
#define GGML_COMMON_IMPL_CPP
#include "ggml-common.h"

#include <immintrin.h>

#include <cmath>
#include <cstdlib>
#include <cstring>

namespace strata::kernels::cpu {
namespace {

inline float h2f(uint16_t h) { return _mm_cvtss_f32(_mm_cvtph_ps(_mm_cvtsi32_si128((int) h))); }
inline uint32_t u32(const uint8_t* p) { uint32_t v; std::memcpy(&v, p, 4); return v; }
inline uint16_t u16(const uint8_t* p) { uint16_t v; std::memcpy(&v, p, 2); return v; }
inline uint64_t u64(const uint8_t* p) { uint64_t v; std::memcpy(&v, p, 8); return v; }

// E-2: the IQ3 grids by one AVX-512 gather (the same words, the same results) - measured 3-5% SLOWER than the scalar
// lookups on a Ryzen 5 7600 (Zen 4 gathers are microcoded), so opt-in: STRATA_IQ_GATHER=1
const bool gather = std::getenv("STRATA_IQ_GATHER") != nullptr;
// E-2: software prefetch distance in bytes (STRATA_IQ_PREFETCH; 0 = off): -2-3% gate/up time on IQ3_S decode
// (the rows are read at ~30 GB/s by 6 cores, near what DDR5 with 4 KB pages gives: the pool is memory-bound)
const int prefetch_ahead = [] {
    const char* v = std::getenv("STRATA_IQ_PREFETCH");
    return v ? std::atoi(v) : 2048;
}();

// lanes 0-7 -> s0, 8-15 -> s1, 16-23 -> s2, 24-31 -> s3 (int16 lanes of a maddubs result: two values each)
inline __m512i scales4(int s0, int s1, int s2, int s3) {
    const __m512i idx = _mm512_set_epi16(3, 3, 3, 3, 3, 3, 3, 3, 2, 2, 2, 2, 2, 2, 2, 2, 1, 1, 1, 1, 1, 1, 1, 1,
                                         0, 0, 0, 0, 0, 0, 0, 0);
    const uint64_t packed = (uint64_t) (uint16_t) s0 | ((uint64_t) (uint16_t) s1 << 16) |
                            ((uint64_t) (uint16_t) s2 << 32) | ((uint64_t) (uint16_t) s3 << 48);
    return _mm512_permutexvar_epi16(idx, _mm512_set1_epi64((long long) packed));
}

// ---- per format: 64 values (chunk j of a 256-value block) -> grid magnitudes, sign mask, scales
template <int TY> struct Fmt;

template <> struct Fmt<16> {   // IQ2_XXS: d, qs[32] u16
    static constexpr int bytes = 66;
    static constexpr float K = 0.125f;
    static inline void decode(const uint8_t* b, int j, __m512i& g, __mmask64& m, __m512i& sc) {
        const uint8_t* q = b + 2 + 16 * j;
        const uint32_t a0 = u32(q), a1 = u32(q + 4), b0 = u32(q + 8), b1 = u32(q + 12);
        g = _mm512_set_epi64((long long) iq2xxs_grid[b0 >> 24], (long long) iq2xxs_grid[(b0 >> 16) & 255],
                             (long long) iq2xxs_grid[(b0 >> 8) & 255], (long long) iq2xxs_grid[b0 & 255],
                             (long long) iq2xxs_grid[a0 >> 24], (long long) iq2xxs_grid[(a0 >> 16) & 255],
                             (long long) iq2xxs_grid[(a0 >> 8) & 255], (long long) iq2xxs_grid[a0 & 255]);
        const uint64_t s = (uint64_t) ksigns_iq2xs[a1 & 127] | ((uint64_t) ksigns_iq2xs[(a1 >> 7) & 127] << 8) |
                           ((uint64_t) ksigns_iq2xs[(a1 >> 14) & 127] << 16) | ((uint64_t) ksigns_iq2xs[(a1 >> 21) & 127] << 24) |
                           ((uint64_t) ksigns_iq2xs[b1 & 127] << 32) | ((uint64_t) ksigns_iq2xs[(b1 >> 7) & 127] << 40) |
                           ((uint64_t) ksigns_iq2xs[(b1 >> 14) & 127] << 48) | ((uint64_t) ksigns_iq2xs[(b1 >> 21) & 127] << 56);
        m = _cvtu64_mask64(s);
        const int sa = 2 * (int) (a1 >> 28) + 1, sb = 2 * (int) (b1 >> 28) + 1;
        sc = scales4(sa, sa, sb, sb);
    }
};

template <> struct Fmt<17> {   // IQ2_XS: d, qs[32] u16 (9-bit grid index + 7-bit sign index), scales[8]
    static constexpr int bytes = 74;
    static constexpr float K = 0.125f;
    static inline void decode(const uint8_t* b, int j, __m512i& g, __mmask64& m, __m512i& sc) {
        const uint8_t* q = b + 2 + 16 * j;
        uint16_t v[8];
        std::memcpy(v, q, 16);
        g = _mm512_set_epi64((long long) iq2xs_grid[v[7] & 511], (long long) iq2xs_grid[v[6] & 511],
                             (long long) iq2xs_grid[v[5] & 511], (long long) iq2xs_grid[v[4] & 511],
                             (long long) iq2xs_grid[v[3] & 511], (long long) iq2xs_grid[v[2] & 511],
                             (long long) iq2xs_grid[v[1] & 511], (long long) iq2xs_grid[v[0] & 511]);
        uint64_t s = 0;
        for (int l = 0; l < 8; ++l) s |= (uint64_t) ksigns_iq2xs[v[l] >> 9] << (8 * l);
        m = _cvtu64_mask64(s);
        const uint8_t s0 = b[66 + 2 * j], s1 = b[66 + 2 * j + 1];
        sc = scales4(2 * (s0 & 15) + 1, 2 * (s0 >> 4) + 1, 2 * (s1 & 15) + 1, 2 * (s1 >> 4) + 1);
    }
};

template <> struct Fmt<22> {   // IQ2_S: d, qs[64] (32 grid bytes, 32 sign bytes), qh[8], scales[8]
    static constexpr int bytes = 82;
    static constexpr float K = 0.125f;
    static inline void decode(const uint8_t* b, int j, __m512i& g, __mmask64& m, __m512i& sc) {
        const uint8_t* qs = b + 2 + 8 * j;
        const uint8_t h0 = b[66 + 2 * j], h1 = b[66 + 2 * j + 1];
        g = _mm512_set_epi64((long long) iq2s_grid[qs[7] | ((h1 << 2) & 0x300)], (long long) iq2s_grid[qs[6] | ((h1 << 4) & 0x300)],
                             (long long) iq2s_grid[qs[5] | ((h1 << 6) & 0x300)], (long long) iq2s_grid[qs[4] | ((h1 << 8) & 0x300)],
                             (long long) iq2s_grid[qs[3] | ((h0 << 2) & 0x300)], (long long) iq2s_grid[qs[2] | ((h0 << 4) & 0x300)],
                             (long long) iq2s_grid[qs[1] | ((h0 << 6) & 0x300)], (long long) iq2s_grid[qs[0] | ((h0 << 8) & 0x300)]);
        m = _cvtu64_mask64(u64(b + 2 + 32 + 8 * j));
        const uint8_t s0 = b[74 + 2 * j], s1 = b[74 + 2 * j + 1];
        sc = scales4(2 * (s0 & 15) + 1, 2 * (s0 >> 4) + 1, 2 * (s1 & 15) + 1, 2 * (s1 >> 4) + 1);
    }
};

template <> struct Fmt<18> {   // IQ3_XXS: d, qs[64] grid bytes, 8 x u32 (4 x 7-bit sign index + 4-bit scale)
    static constexpr int bytes = 98;
    static constexpr float K = 0.25f;
    static inline void decode(const uint8_t* b, int j, __m512i& g, __mmask64& m, __m512i& sc) {
        const uint8_t* q = b + 2 + 16 * j;
        // E-2: one gather of the 16 grid words (the scalar lookups assembled with set_epi32 cost ~30 uops)
        g = gather ? _mm512_i32gather_epi32(_mm512_cvtepu8_epi32(_mm_loadu_si128((const __m128i*) q)), iq3xxs_grid, 4)
                   : _mm512_set_epi32((int) iq3xxs_grid[q[15]], (int) iq3xxs_grid[q[14]], (int) iq3xxs_grid[q[13]], (int) iq3xxs_grid[q[12]],
                             (int) iq3xxs_grid[q[11]], (int) iq3xxs_grid[q[10]], (int) iq3xxs_grid[q[9]], (int) iq3xxs_grid[q[8]],
                             (int) iq3xxs_grid[q[7]], (int) iq3xxs_grid[q[6]], (int) iq3xxs_grid[q[5]], (int) iq3xxs_grid[q[4]],
                             (int) iq3xxs_grid[q[3]], (int) iq3xxs_grid[q[2]], (int) iq3xxs_grid[q[1]], (int) iq3xxs_grid[q[0]]);
        const uint32_t a = u32(b + 2 + 64 + 8 * j), c = u32(b + 2 + 64 + 8 * j + 4);
        const uint64_t s = (uint64_t) ksigns_iq2xs[a & 127] | ((uint64_t) ksigns_iq2xs[(a >> 7) & 127] << 8) |
                           ((uint64_t) ksigns_iq2xs[(a >> 14) & 127] << 16) | ((uint64_t) ksigns_iq2xs[(a >> 21) & 127] << 24) |
                           ((uint64_t) ksigns_iq2xs[c & 127] << 32) | ((uint64_t) ksigns_iq2xs[(c >> 7) & 127] << 40) |
                           ((uint64_t) ksigns_iq2xs[(c >> 14) & 127] << 48) | ((uint64_t) ksigns_iq2xs[(c >> 21) & 127] << 56);
        m = _cvtu64_mask64(s);
        const int sa = 2 * (int) (a >> 28) + 1, sb = 2 * (int) (c >> 28) + 1;
        sc = scales4(sa, sa, sb, sb);
    }
};

template <> struct Fmt<21> {   // IQ3_S: d, qs[64], qh[8], signs[32], scales[4]
    static constexpr int bytes = 110;
    static constexpr float K = 1.0f;
    static inline void decode(const uint8_t* b, int j, __m512i& g, __mmask64& m, __m512i& sc) {
        const uint8_t* q = b + 2 + 16 * j;
        const uint32_t h0 = b[66 + 2 * j], h1 = b[66 + 2 * j + 1];
        if (gather) {   // E-2: the 9-bit indices (the high bit from qh under a mask), one gather
            __m512i idx = _mm512_cvtepu8_epi32(_mm_loadu_si128((const __m128i*) q));
            idx = _mm512_mask_add_epi32(idx, (__mmask16) (h0 | (h1 << 8)), idx, _mm512_set1_epi32(256));
            g = _mm512_i32gather_epi32(idx, iq3s_grid, 4);
        } else {
#define G3(k, h, kk) (int) iq3s_grid[q[k] | (((h >> kk) & 1) << 8)]
        g = _mm512_set_epi32(G3(15, h1, 7), G3(14, h1, 6), G3(13, h1, 5), G3(12, h1, 4),
                             G3(11, h1, 3), G3(10, h1, 2), G3(9, h1, 1), G3(8, h1, 0),
                             G3(7, h0, 7), G3(6, h0, 6), G3(5, h0, 5), G3(4, h0, 4),
                             G3(3, h0, 3), G3(2, h0, 2), G3(1, h0, 1), G3(0, h0, 0));
#undef G3
        }
        m = _cvtu64_mask64(u64(b + 74 + 8 * j));
        const uint8_t s = b[106 + j];
        const int sa = 2 * (s & 15) + 1, sb = 2 * (s >> 4) + 1;
        sc = scales4(sa, sa, sb, sb);
    }
};

inline void rows_prefetch(const uint8_t* blk) {
    if (prefetch_ahead <= 0) return;
    const uint8_t* ahead = blk + (size_t) prefetch_ahead;
    _mm_prefetch((const char*) ahead, _MM_HINT_T0);
    _mm_prefetch((const char*) ahead + 64, _MM_HINT_T0);
}

// ---- IQ1_S (19): 64 values (two 32-value sub-blocks) per pass.  Each sub-block is four 8-value grid entries
// (iq1s_grid) plus one `qh` word: a 4-bit scale selector (bits 12-14) and a sign bit (15) for the -1 +/- 1/8
// delta every value carries.  The grid half is the same maddubs/madd pair as the other formats (the grid bytes
// are 0/-1/+1, so ggml's mul_add sign trick applies); the delta half is a scalar per sub-block, ls * delta *
// (sum of the sub-block's activations), which q8_K precomputes in `bsums` as groups of 16.
// Ported from ggml's ggml_vec_dot_iq1_s_q8_K (arch/x86/quants.c); only the float-addition order differs.
template <int NT>
inline void row_dot_iq1s(const uint8_t* row, int nblocks, const block_q8_K* const* y, float* res) {
    __m512 accf[NT];
    float accd[NT];
    for (int t = 0; t < NT; ++t) { accf[t] = _mm512_setzero_ps(); accd[t] = 0.f; }
    for (int i = 0; i < nblocks; ++i) {
        const uint8_t* blk = row + (size_t) i * 50;
        rows_prefetch(blk);
        __m512i acci[NT];
        float sd[NT];
        for (int t = 0; t < NT; ++t) { acci[t] = _mm512_setzero_si512(); sd[t] = 0.f; }
        for (int j = 0; j < 4; ++j) {
            const uint16_t qh0 = u16(blk + 34 + 4 * j);       // sub-block 2j
            const uint16_t qh1 = u16(blk + 34 + 4 * j + 2);   // sub-block 2j+1
            const uint8_t* qs0 = blk + 2 + 8 * j;
            const uint8_t* qs1 = blk + 2 + 8 * j + 4;
            const __m512i g = _mm512_set_epi64(
                (long long) iq1s_grid[qs1[3] | ((qh1 >> 1) & 0x700)],
                (long long) iq1s_grid[qs1[2] | ((qh1 << 2) & 0x700)],
                (long long) iq1s_grid[qs1[1] | ((qh1 << 5) & 0x700)],
                (long long) iq1s_grid[qs1[0] | ((qh1 << 8) & 0x700)],
                (long long) iq1s_grid[qs0[3] | ((qh0 >> 1) & 0x700)],
                (long long) iq1s_grid[qs0[2] | ((qh0 << 2) & 0x700)],
                (long long) iq1s_grid[qs0[1] | ((qh0 << 5) & 0x700)],
                (long long) iq1s_grid[qs0[0] | ((qh0 << 8) & 0x700)]);
            const int lsv0 = 2 * ((qh0 >> 12) & 7) + 1, lsv1 = 2 * ((qh1 >> 12) & 7) + 1;
            const __m512i sc = scales4(lsv0, lsv0, lsv1, lsv1);
            const float sgn0 = (qh0 & 0x8000) ? -1.f : 1.f;
            const float sgn1 = (qh1 & 0x8000) ? -1.f : 1.f;
            for (int t = 0; t < NT; ++t) {
                const __m512i yv = _mm512_loadu_si512((const void*) (y[t][i].qs + 64 * j));
                // iq1s_grid holds signed values in {-1,0,1}: ggml's mul_add_epi8 trick (|g| as the maddubs
                // operand, q8's sign taken from g) - a direct maddubs would read -1 as 255.
                const __m512i ga = _mm512_abs_epi8(g);
                const __m512i ys = _mm512_mask_sub_epi8(yv, _mm512_movepi8_mask(g), _mm512_setzero_si512(), yv);
                acci[t] = _mm512_add_epi32(acci[t], _mm512_madd_epi16(_mm512_maddubs_epi16(ga, ys), sc));
                sd[t] += sgn0 * (float) lsv0 * (float) (y[t][i].bsums[4 * j] + y[t][i].bsums[4 * j + 1]) +
                         sgn1 * (float) lsv1 * (float) (y[t][i].bsums[4 * j + 2] + y[t][i].bsums[4 * j + 3]);
            }
        }
        const float dx = h2f(u16(blk));
        for (int t = 0; t < NT; ++t) {
            const float dd = dx * y[t][i].d;
            accf[t] = _mm512_fmadd_ps(_mm512_set1_ps(dd), _mm512_cvtepi32_ps(acci[t]), accf[t]);
            accd[t] += dd * sd[t];
        }
    }
    for (int t = 0; t < NT; ++t) res[t] = _mm512_reduce_add_ps(accf[t]) + IQ1S_DELTA * accd[t];
}

// ---- IQ1_M (29): 64 values (two 32-value sub-blocks) per pass; qs[32] grid indices, qh[16] high bits + delta
// signs, scales[8] packed 3-bit sub-scales.  Each sub-block splits into two 16-value halves with their own scale
// (ls1 from the low 6 bits of scales[ib/2], ls2 from the bits above), and each 8-value group carries the qh delta
// sign.  Ported from ggml's ggml_vec_dot_iq1_m_q8_K (arch/x86/quants.c); the shared f16 scale is assembled once
// per block from the top nibbles of the four scale words.
template <int NT>
inline void row_dot_iq1m(const uint8_t* row, int nblocks, const block_q8_K* const* y, float* res) {
    const __m512i one8 = _mm512_set1_epi8(1);
    __m512 accf[NT];
    float accd[NT];
    for (int t = 0; t < NT; ++t) { accf[t] = _mm512_setzero_ps(); accd[t] = 0.f; }
    for (int i = 0; i < nblocks; ++i) {
        const uint8_t* blk = row + (size_t) i * 56;
        rows_prefetch(blk);
        const uint8_t* qh = blk + 32;
        const uint16_t* sc = (const uint16_t*) (blk + 48);
        const uint16_t su = (uint16_t) ((sc[0] >> 12) | ((sc[1] >> 8) & 0x00f0) | ((sc[2] >> 4) & 0x0f00) |
                                        (sc[3] & 0xf000));
        const float dscale = h2f(su);
        __m512i acci[NT], acc2[NT];
        for (int t = 0; t < NT; ++t) { acci[t] = _mm512_setzero_si512(); acc2[t] = _mm512_setzero_si512(); }
        for (int j = 0; j < 4; ++j) {
            const int ib0 = 2 * j, ib1 = 2 * j + 1;
            const uint8_t qh0a = qh[2 * ib0], qh0b = qh[2 * ib0 + 1];
            const uint8_t qh1a = qh[2 * ib1], qh1b = qh[2 * ib1 + 1];
            const uint8_t* qs0 = blk + 4 * ib0;
            const uint8_t* qs1 = blk + 4 * ib1;
            const __m512i g = _mm512_set_epi64(
                (long long) iq1s_grid[qs1[3] | ((qh1b << 4) & 0x700)],
                (long long) iq1s_grid[qs1[2] | ((qh1b << 8) & 0x700)],
                (long long) iq1s_grid[qs1[1] | ((qh1a << 4) & 0x700)],
                (long long) iq1s_grid[qs1[0] | ((qh1a << 8) & 0x700)],
                (long long) iq1s_grid[qs0[3] | ((qh0b << 4) & 0x700)],
                (long long) iq1s_grid[qs0[2] | ((qh0b << 8) & 0x700)],
                (long long) iq1s_grid[qs0[1] | ((qh0a << 4) & 0x700)],
                (long long) iq1s_grid[qs0[0] | ((qh0a << 8) & 0x700)]);
            const __m512i delta = _mm512_set_epi64(
                (long long) (qh1b & 0x80 ? 0xffffffffffffffffULL : 0x0101010101010101ULL),
                (long long) (qh1b & 0x08 ? 0xffffffffffffffffULL : 0x0101010101010101ULL),
                (long long) (qh1a & 0x80 ? 0xffffffffffffffffULL : 0x0101010101010101ULL),
                (long long) (qh1a & 0x08 ? 0xffffffffffffffffULL : 0x0101010101010101ULL),
                (long long) (qh0b & 0x80 ? 0xffffffffffffffffULL : 0x0101010101010101ULL),
                (long long) (qh0b & 0x08 ? 0xffffffffffffffffULL : 0x0101010101010101ULL),
                (long long) (qh0a & 0x80 ? 0xffffffffffffffffULL : 0x0101010101010101ULL),
                (long long) (qh0a & 0x08 ? 0xffffffffffffffffULL : 0x0101010101010101ULL));
            const int ls0a = 2 * ((sc[ib0 / 2] >> (6 * (ib0 % 2) + 0)) & 0x7) + 1;
            const int ls0b = 2 * ((sc[ib0 / 2] >> (6 * (ib0 % 2) + 3)) & 0x7) + 1;
            const int ls1a = 2 * ((sc[ib1 / 2] >> (6 * (ib1 % 2) + 0)) & 0x7) + 1;
            const int ls1b = 2 * ((sc[ib1 / 2] >> (6 * (ib1 % 2) + 3)) & 0x7) + 1;
            const __m512i scv = scales4(ls0a, ls0b, ls1a, ls1b);
            for (int t = 0; t < NT; ++t) {
                const __m512i yv = _mm512_loadu_si512((const void*) (y[t][i].qs + 64 * j));
                const __m512i ga = _mm512_abs_epi8(g);
                const __m512i ys = _mm512_mask_sub_epi8(yv, _mm512_movepi8_mask(g), _mm512_setzero_si512(), yv);
                acci[t] = _mm512_add_epi32(acci[t], _mm512_madd_epi16(_mm512_maddubs_epi16(ga, ys), scv));
                const __m512i yd = _mm512_mask_sub_epi8(yv, _mm512_movepi8_mask(delta), _mm512_setzero_si512(), yv);
                acc2[t] = _mm512_add_epi32(acc2[t], _mm512_madd_epi16(_mm512_maddubs_epi16(one8, yd), scv));
            }
        }
        for (int t = 0; t < NT; ++t) {
            const float dd = dscale * y[t][i].d;
            accf[t] = _mm512_fmadd_ps(_mm512_set1_ps(dd), _mm512_cvtepi32_ps(acci[t]), accf[t]);
            accd[t] += dd * _mm512_reduce_add_ps(_mm512_cvtepi32_ps(acc2[t]));
        }
    }
    for (int t = 0; t < NT; ++t) res[t] = _mm512_reduce_add_ps(accf[t]) + IQ1M_DELTA * accd[t];
}

template <int TY, int NT>
inline void row_dot(const uint8_t* row, int nblocks, const block_q8_K* const* y, float* res) {
    __m512 accf[NT];
    for (int t = 0; t < NT; ++t) accf[t] = _mm512_setzero_ps();
    const __m512i zero = _mm512_setzero_si512();
    for (int i = 0; i < nblocks; ++i) {
        const uint8_t* blk = row + (size_t) i * Fmt<TY>::bytes;
        // E-2: the weight rows stream from DRAM (4 KB pages when large pages are refused): ask for the bytes a few
        // blocks ahead (the next row's start included) before the decode needs them
        if (prefetch_ahead > 0) {
            const uint8_t* ahead = blk + (size_t) prefetch_ahead;
            _mm_prefetch((const char*) ahead, _MM_HINT_T0);
            _mm_prefetch((const char*) ahead + 64, _MM_HINT_T0);
        }
        __m512i acci[NT];
        for (int t = 0; t < NT; ++t) acci[t] = _mm512_setzero_si512();
        for (int j = 0; j < 4; ++j) {
            __m512i g, sc;
            __mmask64 m;
            Fmt<TY>::decode(blk, j, g, m, sc);
            for (int t = 0; t < NT; ++t) {
                const __m512i yv = _mm512_loadu_si512((const void*) (y[t][i].qs + 64 * j));
                const __m512i ys = _mm512_mask_sub_epi8(yv, m, zero, yv);
                acci[t] = _mm512_add_epi32(acci[t], _mm512_madd_epi16(_mm512_maddubs_epi16(g, ys), sc));
            }
        }
        const float dx = h2f(u16(blk)) * Fmt<TY>::K;
        for (int t = 0; t < NT; ++t)
            accf[t] = _mm512_fmadd_ps(_mm512_set1_ps(dx * y[t][i].d), _mm512_cvtepi32_ps(acci[t]), accf[t]);
    }
    for (int t = 0; t < NT; ++t) res[t] = _mm512_reduce_add_ps(accf[t]);
}

template <int TY, int NT>
inline void row_dot_any(const uint8_t* row, int nblocks, const block_q8_K* const* y, float* res) {
    if constexpr (TY == 19) row_dot_iq1s<NT>(row, nblocks, y, res);
    else if constexpr (TY == 29) row_dot_iq1m<NT>(row, nblocks, y, res);
    else                    row_dot<TY, NT>(row, nblocks, y, res);
}

template <int TY, int NT>
void gu_rows(const uint8_t* blob, size_t gu_row, size_t up_off, int n, const void* const* act, float* const* ff,
             int r0, int r1) {
    const block_q8_K* y[NT];
    for (int t = 0; t < NT; ++t) y[t] = (const block_q8_K*) act[t];
    const int nb = n / QK_K;
    float g[NT], u[NT];
    for (int r = r0; r < r1; ++r) {
        row_dot_any<TY, NT>(blob + (size_t) r * gu_row, nb, y, g);
        row_dot_any<TY, NT>(blob + up_off + (size_t) r * gu_row, nb, y, u);
        for (int t = 0; t < NT; ++t) ff[t][r] = (g[t] / (1.f + std::exp(-g[t]))) * u[t];
    }
}

template <int TY, int NT>
void dot_rows(const uint8_t* w, size_t row_bytes, int n, const void* const* act, float* const* out, int r0, int r1) {
    const block_q8_K* y[NT];
    for (int t = 0; t < NT; ++t) y[t] = (const block_q8_K*) act[t];
    float res[NT];
    for (int r = r0; r < r1; ++r) {
        row_dot_any<TY, NT>(w + (size_t) r * row_bytes, n / QK_K, y, res);
        for (int t = 0; t < NT; ++t) out[t][r] = res[t];
    }
}

template <int TY>
void gu_rows_nt(int nt, const uint8_t* blob, size_t gu_row, size_t up_off, int n, const void* const* act,
                float* const* ff, int r0, int r1) {
    switch (nt) {
        case 1: gu_rows<TY, 1>(blob, gu_row, up_off, n, act, ff, r0, r1); break;
        case 2: gu_rows<TY, 2>(blob, gu_row, up_off, n, act, ff, r0, r1); break;
        case 3: gu_rows<TY, 3>(blob, gu_row, up_off, n, act, ff, r0, r1); break;
        case 4: gu_rows<TY, 4>(blob, gu_row, up_off, n, act, ff, r0, r1); break;
        case 5: gu_rows<TY, 5>(blob, gu_row, up_off, n, act, ff, r0, r1); break;
        case 6: gu_rows<TY, 6>(blob, gu_row, up_off, n, act, ff, r0, r1); break;
        case 7: gu_rows<TY, 7>(blob, gu_row, up_off, n, act, ff, r0, r1); break;
        default: gu_rows<TY, 8>(blob, gu_row, up_off, n, act, ff, r0, r1); break;
    }
}

template <int TY>
void dot_rows_nt(int nt, const uint8_t* w, size_t row_bytes, int n, const void* const* act, float* const* out, int r0,
                 int r1) {
    switch (nt) {
        case 1: dot_rows<TY, 1>(w, row_bytes, n, act, out, r0, r1); break;
        case 2: dot_rows<TY, 2>(w, row_bytes, n, act, out, r0, r1); break;
        case 3: dot_rows<TY, 3>(w, row_bytes, n, act, out, r0, r1); break;
        case 4: dot_rows<TY, 4>(w, row_bytes, n, act, out, r0, r1); break;
        default: for (int t0 = 0; t0 < nt; t0 += 4) {
            const int k = nt - t0 < 4 ? nt - t0 : 4;
            dot_rows_nt<TY>(k, w, row_bytes, n, act + t0, out + t0, r0, r1);
        }
    }
}

}  // namespace

bool iq512_supported(int type) noexcept {
    return type == 16 || type == 17 || type == 18 || type == 19 || type == 21 || type == 22 || type == 29;
}

void iq512_gu_rows(int type, const uint8_t* blob, size_t gu_row, size_t up_off, int n, const void* const* act, int nt,
                   float* const* ff, int r0, int r1) {
    switch (type) {
        case 16: gu_rows_nt<16>(nt, blob, gu_row, up_off, n, act, ff, r0, r1); break;
        case 17: gu_rows_nt<17>(nt, blob, gu_row, up_off, n, act, ff, r0, r1); break;
        case 18: gu_rows_nt<18>(nt, blob, gu_row, up_off, n, act, ff, r0, r1); break;
        case 19: gu_rows_nt<19>(nt, blob, gu_row, up_off, n, act, ff, r0, r1); break;
        case 21: gu_rows_nt<21>(nt, blob, gu_row, up_off, n, act, ff, r0, r1); break;
        case 22: gu_rows_nt<22>(nt, blob, gu_row, up_off, n, act, ff, r0, r1); break;
        case 29: gu_rows_nt<29>(nt, blob, gu_row, up_off, n, act, ff, r0, r1); break;
        default: break;
    }
}

void iq512_rows(int type, const uint8_t* w, size_t row_bytes, int n, const void* const* act, int nt, float* const* out,
                int r0, int r1) {
    switch (type) {
        case 16: dot_rows_nt<16>(nt, w, row_bytes, n, act, out, r0, r1); break;
        case 17: dot_rows_nt<17>(nt, w, row_bytes, n, act, out, r0, r1); break;
        case 18: dot_rows_nt<18>(nt, w, row_bytes, n, act, out, r0, r1); break;
        case 19: dot_rows_nt<19>(nt, w, row_bytes, n, act, out, r0, r1); break;
        case 21: dot_rows_nt<21>(nt, w, row_bytes, n, act, out, r0, r1); break;
        case 22: dot_rows_nt<22>(nt, w, row_bytes, n, act, out, r0, r1); break;
        case 29: dot_rows_nt<29>(nt, w, row_bytes, n, act, out, r0, r1); break;
        default: break;
    }
}

}  // namespace strata::kernels::cpu
