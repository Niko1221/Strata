// src/kernels/cpu/iq_avx1.cpp - the i-quant expert rows for CPUs without AVX2 (STRATA_ISA_FLOOR=avx:
// Sandy/Ivy Bridge, Xeon E5 v1/v2, Bulldozer).  See iq_avx1.hpp.
//
// iq_avx2.cpp's multi-token scheme in 128-bit lanes: each 32-value chunk is decoded once per verify window
// (grid lookups, one sign vector, one scale vector) and every token applies it with a load, a sign, a
// maddubs, a madd and an add - all SSSE3/SSE2, all available since Core 2.  The 256-bit integer ops of the
// AVX2 kernel (pshufb/maddubs/sign_epi8 in __m256i, vpgather, vpdpbusd) have no AVX1 form and no FMA exists
// there, so the float accumulation is a plain SSE2 mul+add.  Without these kernels such a CPU runs every
// expert on ggml-cpu's scalar `_generic` dots (every x86 i-quant dot is #if defined(__AVX2__)).
//
// The arithmetic is ggml's (ggml-cpu/quants.c, the `_generic` references) - only the order of the float
// additions differs; the integer sums are exact.  iq_avx1_parity checks: rows vs ggml-cpu's dot rel <= 1e-5,
// and a multi-token row bit for bit the same kernel's single-token row for that token (#152).
//
// Formats: gate/up IQ3_XXS (18), IQ3_S (21), IQ2_S (22), IQ4_XS (23); down IQ4_NL (20) and Q2_0 (42, the
// fork's type).  IQ2_XXS/IQ2_XS have no layer in the packs that need this and stay on ggml-cpu's dot.
//
// This file is compiled for SSE4.2 + AVX with AVX2/FMA/F16C explicitly off (CMakeLists.txt), and the engine
// calls it only behind cpu_avx1_ok().  No static constructors at namespace scope (#391): the tables below
// are constexpr, like iq_avx2.cpp's.
#include "strata/kernels/cpu/iq_avx1.hpp"

#include "ggml.h"

#define GGML_COMMON_DECL_CPP
#define GGML_COMMON_IMPL_CPP
#include "ggml-common.h"

#include <immintrin.h>

#include <cmath>
#include <cstdlib>
#include <cstring>
#include <vector>

#if !defined(__AVX__)
#error "iq_avx1.cpp must be compiled with AVX enabled (see the per-source flags in CMakeLists.txt)"
#endif
#if defined(__FMA__) || defined(__F16C__)
#error "iq_avx1.cpp must be compiled without FMA/F16C: it runs on CPUs (Sandy/Ivy Bridge) that lack them"
#endif

namespace strata::kernels::cpu {
namespace {

// fp16 -> fp32 without F16C.  The normal-exponent case (every real quantized scale) is branchless:
// (E<<10|M)<<13 + 0x38000000 == (E+112)<<23 | M<<13, the exact fp32 bits.  Zero, subnormal, inf and nan
// keep expert.cpp's full converter (bit-identical, off the hot path).
inline float h2f(uint16_t h) {
    const uint32_t e = h & 0x7C00u;
    if (e && e != 0x7C00u) {
        const uint32_t f = ((uint32_t) (h & 0x7FFFu) << 13) + 0x38000000u | (uint32_t) (h & 0x8000u) << 16;
        float out;
        std::memcpy(&out, &f, 4);
        return out;
    }
    const uint32_t sign = (uint32_t) (h >> 15) & 1u;
    uint32_t exp = (h >> 10) & 0x1Fu, man = h & 0x3FFu, f;
    if (exp == 0) {
        if (man == 0) {
            f = sign << 31;
        } else {
            exp = 127 - 15 + 1;
            while (!(man & 0x400u)) { man <<= 1; --exp; }
            man &= 0x3FFu;
            f = (sign << 31) | (exp << 23) | (man << 13);
        }
    } else if (exp == 31) {
        f = (sign << 31) | 0x7F800000u | (man << 13);
    } else {
        f = (sign << 31) | ((exp - 15 + 127) << 23) | (man << 13);
    }
    float out;
    std::memcpy(&out, &f, 4);
    return out;
}

inline uint32_t u32(const uint8_t* p) { uint32_t v; std::memcpy(&v, p, 4); return v; }
inline uint16_t u16(const uint8_t* p) { uint16_t v; std::memcpy(&v, p, 2); return v; }
inline uint64_t u64(const uint8_t* p) { uint64_t v; std::memcpy(&v, p, 8); return v; }

// The sign-vector pattern of iq_avx2.cpp's sgn_vec_at in 128-bit: 16 values from two sign bytes.
// bits = the two bytes broadcast over the 16 lanes, sel = bit k of each byte per lane, -> 16 bytes of
// -1 (bit set) / +1.  ggml's bit_selector pattern, same bits.
inline __m128i sgn16(uint32_t w, int byte_pair) {   // byte_pair 0: bytes 0-1 (values 0-15), 1: bytes 2-3
    const __m128i bits = _mm_shuffle_epi8(_mm_set1_epi32((int) w),
                                          byte_pair ? _mm_setr_epi8(2, 2, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 3, 3)
                                                    : _mm_setr_epi8(0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1));
    const __m128i sel = _mm_setr_epi8(1, 2, 4, 8, 16, 32, 64, (char) 0x80, 1, 2, 4, 8, 16, 32, 64, (char) 0x80);
    return _mm_or_si128(_mm_cmpeq_epi8(_mm_and_si128(bits, sel), sel), _mm_set1_epi8(1));
}

// keven_signs_q2xs as a constexpr table (iq_avx2.cpp's EvenSigns; byte k = 0xFF when bit k of
// ksigns_iq2xs[i] is set, 0x01 otherwise).  constexpr, not a runtime constructor (#391).
struct EvenSigns {
    uint64_t v[128];
    constexpr EvenSigns() : v{} {
        for (int i = 0; i < 128; ++i) {
            int par = 0;
            for (int k = 0; k < 7; ++k) par ^= (i >> k) & 1;
            const int s = i | (par << 7);          // == ksigns_iq2xs[i]
            uint64_t r = 0;
            for (int k = 0; k < 8; ++k) r |= (uint64_t) (((s >> k) & 1) ? 0xFF : 0x01) << (8 * k);
            v[i] = r;
        }
    }
};
static constexpr EvenSigns even_signs{};
static_assert(even_signs.v[0] == 0x0101010101010101ull && even_signs.v[1] == 0xFF010101010101FFull,
              "keven_signs_q2xs: byte k = 0xFF when bit k of ksigns_iq2xs[i] is set");

// The high index bits of a 32-value half spread one per byte (iq_avx2.cpp's HiSpread): punpcklbw puts them
// next to the low index bytes, so every grid index is a 16-bit load.  IQ3_S: byte k = bit k (the 9th bit of
// index k).  IQ2_S: byte k = bits 2k,2k+1 (bits 8-9 of index k), k < 4.
struct HiSpread {
    uint64_t iq3s[256];
    uint64_t iq2s[256];
    constexpr HiSpread() : iq3s{}, iq2s{} {
        for (int h = 0; h < 256; ++h) {
            uint64_t r3 = 0, r2 = 0;
            for (int k = 0; k < 8; ++k) r3 |= (uint64_t) ((h >> k) & 1) << (8 * k);
            for (int k = 0; k < 4; ++k) r2 |= (uint64_t) ((h >> (2 * k)) & 3) << (8 * k);
            iq3s[h] = r3;
            iq2s[h] = r2;
        }
    }
};
static constexpr HiSpread hi_spread{};
static_assert(hi_spread.iq3s[0x81] == 0x0100000000000001ull && hi_spread.iq2s[0xE4] == 0x0000000003020100ull,
              "hi_spread: one high-bit group per index byte");

// The low index bytes interleaved with the spread high bits: eight 16-bit grid indices in one register.
// (iq_avx2.cpp's grid_indices stores them to memory; on Ivy Bridge that store -> narrow-load round trip
// costs a forwarding stall per index, so the decoders below read the lanes with _mm_extract_epi16 instead.)
inline __m128i grid_idx(const uint8_t* q, uint64_t hi) {
    return _mm_unpacklo_epi8(_mm_loadl_epi64((const __m128i*) q), _mm_cvtsi64_si128((long long) hi));
}

inline float hsum4(__m128 v) {
    __m128 s = _mm_add_ps(v, _mm_movehl_ps(v, v));
    s = _mm_add_ss(s, _mm_movehdup_ps(s));
    return _mm_cvtss_f32(s);
}

// E-2 prefetching, the same knob as the AVX-2/512 kernels: STRATA_IQ_PREFETCH is the distance in bytes.
int prefetch_distance() {
    static const int d = [] {
        const char* v = std::getenv("STRATA_IQ_PREFETCH");
        return v ? std::atoi(v) : 2048;
    }();
    return d;
}

inline void rows_ahead(const uint8_t* blk, int pf) {
    if (pf <= 0) return;
    _mm_prefetch((const char*) blk + pf, _MM_HINT_T0);
    _mm_prefetch((const char*) blk + pf + 64, _MM_HINT_T0);
}

// ---- per format: one 32-value half (values 64*j + 32*half .. +31) -> grid magnitudes, sign vector and
// scales as TWO 16-value halves (g0/g1, s0/s1, c0/c1).
template <int TY> struct Fmt128;

template <> struct Fmt128<18> {   // IQ3_XXS: d, qs[64] grid bytes, 8 x u32 (4 x 7-bit sign index + 4-bit scale)
    static constexpr int bytes = 98;
    static constexpr float K = 0.25f;
    static inline void decode(const uint8_t* b, int j, int half,
                              __m128i& g0, __m128i& g1, __m128i& s0, __m128i& s1, __m128i& c0, __m128i& c1) {
        const uint8_t* q = b + 2 + 16 * j + 8 * half;
        g0 = _mm_set_epi32((int) iq3xxs_grid[q[3]], (int) iq3xxs_grid[q[2]],
                           (int) iq3xxs_grid[q[1]], (int) iq3xxs_grid[q[0]]);
        g1 = _mm_set_epi32((int) iq3xxs_grid[q[7]], (int) iq3xxs_grid[q[6]],
                           (int) iq3xxs_grid[q[5]], (int) iq3xxs_grid[q[4]]);
        const uint32_t w = u32(b + 2 + 64 + 8 * j + 4 * half);
        s0 = _mm_set_epi64x((long long) even_signs.v[(w >> 7) & 127], (long long) even_signs.v[w & 127]);
        s1 = _mm_set_epi64x((long long) even_signs.v[(w >> 21) & 127], (long long) even_signs.v[(w >> 14) & 127]);
        c0 = c1 = _mm_set1_epi16((short) (2 * (int) (w >> 28) + 1));
    }
};

template <> struct Fmt128<21> {   // IQ3_S: d, qs[64], qh[8], signs[32], scales[4]
    static constexpr int bytes = 110;
    static constexpr float K = 1.0f;
    static inline void decode(const uint8_t* b, int j, int half,
                              __m128i& g0, __m128i& g1, __m128i& s0, __m128i& s1, __m128i& c0, __m128i& c1) {
        const uint8_t* q = b + 2 + 16 * j + 8 * half;
        const __m128i idx = grid_idx(q, hi_spread.iq3s[b[66 + 2 * j + half]]);
#define G3(k) (int) iq3s_grid[_mm_extract_epi16(idx, k)]
        g0 = _mm_set_epi32(G3(3), G3(2), G3(1), G3(0));
        g1 = _mm_set_epi32(G3(7), G3(6), G3(5), G3(4));
#undef G3
        const uint32_t sg = u32(b + 74 + 8 * j + 4 * half);
        s0 = sgn16(sg, 0);
        s1 = sgn16(sg, 1);
        const uint8_t s = b[106 + j];
        c0 = c1 = _mm_set1_epi16((short) (2 * (int) (half ? s >> 4 : s & 15) + 1));
    }
};

template <> struct Fmt128<22> {   // IQ2_S: d, qs[64] (32 grid bytes, 32 sign bytes), qh[8], scales[8]
    static constexpr int bytes = 82;
    static constexpr float K = 0.125f;
    static inline void decode(const uint8_t* b, int j, int half,
                              __m128i& g0, __m128i& g1, __m128i& s0, __m128i& s1, __m128i& c0, __m128i& c1) {
        // only the first 4 index bytes are this half's (the load's rest stays inside the block)
        const __m128i idx = grid_idx(b + 2 + 8 * j + 4 * half, hi_spread.iq2s[b[66 + 2 * j + half]]);
        g0 = _mm_set_epi64x((long long) iq2s_grid[_mm_extract_epi16(idx, 1)],
                            (long long) iq2s_grid[_mm_extract_epi16(idx, 0)]);
        g1 = _mm_set_epi64x((long long) iq2s_grid[_mm_extract_epi16(idx, 3)],
                            (long long) iq2s_grid[_mm_extract_epi16(idx, 2)]);
        const uint32_t sg = u32(b + 2 + 32 + 8 * j + 4 * half);
        s0 = sgn16(sg, 0);
        s1 = sgn16(sg, 1);
        const uint8_t sb = b[74 + 2 * j + half];
        c0 = _mm_set1_epi16((short) (2 * (sb & 15) + 1));
        c1 = _mm_set1_epi16((short) (2 * (sb >> 4) + 1));
    }
};

template <> struct Fmt128<23> {   // IQ4_XS: d, scales_h, scales_l[4], qs[128] - 136 B, 8 signed sub-scales
    // ggml's ggml_vec_dot_iq4_xs_q8_K: the same 16-value codebook as IQ4_NL, one 6-bit scale per 32-value
    // sub-block read as two nibbles of scales_l plus two bits of scales_h, used SIGNED as (ls - 32): the
    // scale folds into the int16 operand of madd_epi16.  The codebook sign rides on sign_epi8, |w| is the
    // unsigned maddubs operand (a pair is at most 2 * 127 * 128, no saturation).
    static constexpr int bytes = 136;
    static constexpr float K = 1.0f;
    static inline void decode(const uint8_t* b, int j, int half,
                              __m128i& g0, __m128i& g1, __m128i& s0, __m128i& s1, __m128i& c0, __m128i& c1) {
        const int H = 2 * j + half;
        const __m128i values = _mm_loadu_si128((const __m128i*) kvalues_iq4nl);
        const __m128i bits = _mm_loadu_si128((const __m128i*) (b + 8 + 16 * H));
        const __m128i m4 = _mm_set1_epi8(0x0f);
        const __m128i v0 = _mm_shuffle_epi8(values, _mm_and_si128(bits, m4));
        const __m128i v1 = _mm_shuffle_epi8(values, _mm_and_si128(_mm_srli_epi16(bits, 4), m4));
        g0 = _mm_sign_epi8(v0, v0);
        g1 = _mm_sign_epi8(v1, v1);
        s0 = _mm_sign_epi8(_mm_set1_epi8(1), v0);
        s1 = _mm_sign_epi8(_mm_set1_epi8(1), v1);
        const int ls = ((b[4 + (H >> 1)] >> (4 * (H & 1))) & 0xf) | (((u16(b + 2) >> (2 * H)) & 3) << 4);
        c0 = c1 = _mm_set1_epi16((short) (ls - 32));
    }
};

// acc + madd(maddubs(g, sign(y, sgn)), sc): the AVX2 kernel's per-token application in 128-bit.
// (The scale cannot fold into the grid byte: iq3xxs_grid max 62 and iq2s_grid max 43, and c = 2*is+1 <= 31,
// so g*c reaches 1922 - far past the 255/127 maddubs operand range.  The int16 madd stage stays.)
inline __m128i madd_add(__m128i acc, __m128i g, __m128i sgn, __m128i yv, __m128i sc) {
    return _mm_add_epi32(acc, _mm_madd_epi16(_mm_maddubs_epi16(g, _mm_sign_epi8(yv, sgn)), sc));
}

// ---- the row kernels (iq_avx2_rows.inl's row_dot in 128-bit lanes)
// R rows of the SAME expert through one block loop: the per-half decode chains (index byte -> grid table ->
// madd) are ~20 cycles of pure latency and at NT=1 there is nothing to overlap, so the kernel runs latency-
// bound at ~1/4 of its issue rate.  Two rows interleave independent chains in the same registers and roughly
// double the throughput; the per-row integer sums and the float stage are unchanged, bit for bit.
template <int TY, int NT, int R>
inline void row_dot_r(const uint8_t* const* rows, int nblocks, const block_q8_K* const* y, float* res) {
    const int pf = prefetch_distance();
    __m128 accf[R][NT];
    for (int rr = 0; rr < R; ++rr)
        for (int t = 0; t < NT; ++t) accf[rr][t] = _mm_setzero_ps();
    for (int i = 0; i < nblocks; ++i) {
        __m128i acci[R][NT][2];
        for (int rr = 0; rr < R; ++rr)
            for (int t = 0; t < NT; ++t) acci[rr][t][0] = acci[rr][t][1] = _mm_setzero_si128();
        for (int rr = 0; rr < R; ++rr) rows_ahead(rows[rr] + (size_t) i * Fmt128<TY>::bytes, pf);
        for (int j = 0; j < 4; ++j) {
            for (int half = 0; half < 2; ++half) {
                __m128i g0[R], g1[R], s0[R], s1[R], c0[R], c1[R];
                for (int rr = 0; rr < R; ++rr) {
                    const uint8_t* blk = rows[rr] + (size_t) i * Fmt128<TY>::bytes;
                    Fmt128<TY>::decode(blk, j, half, g0[rr], g1[rr], s0[rr], s1[rr], c0[rr], c1[rr]);
                }
                const int off = 64 * j + 32 * half;
                for (int rr = 0; rr < R; ++rr) {
                    for (int t = 0; t < NT; ++t) {
                        const uint8_t* q8 = (const uint8_t*) (y[t][i].qs + off);
                        acci[rr][t][0] = madd_add(acci[rr][t][0], g0[rr], s0[rr], _mm_loadu_si128((const __m128i*) q8), c0[rr]);
                        acci[rr][t][1] = madd_add(acci[rr][t][1], g1[rr], s1[rr], _mm_loadu_si128((const __m128i*) (q8 + 16)), c1[rr]);
                    }
                }
            }
        }
        for (int rr = 0; rr < R; ++rr) {
            const uint8_t* blk = rows[rr] + (size_t) i * Fmt128<TY>::bytes;
            const float dx = h2f(u16(blk)) * Fmt128<TY>::K;
            for (int t = 0; t < NT; ++t) {
                // one cvt instead of two: the int32 sum is exact (|lane| <= 2 * 8 * 15 * 127 * 31 < 2^20 for
                // the grid formats, < 2^23 for IQ4_XS), so cvt(a+b) has the same bits as cvt(a)+cvt(b).
                const __m128 f = _mm_cvtepi32_ps(_mm_add_epi32(acci[rr][t][0], acci[rr][t][1]));
                accf[rr][t] = _mm_add_ps(accf[rr][t], _mm_mul_ps(_mm_set1_ps(dx * y[t][i].d), f));
            }
        }
    }
    for (int rr = 0; rr < R; ++rr)
        for (int t = 0; t < NT; ++t) res[rr * NT + t] = hsum4(accf[rr][t]);
}

template <int TY, int NT>
inline void row_dot(const uint8_t* row, int nblocks, const block_q8_K* const* y, float* res) {
    const uint8_t* rows[1] = {row};   // R=1 form: one row, same code, same bits
    row_dot_r<TY, NT, 1>(rows, nblocks, y, res);
}

template <int TY, int NT>
inline void gu_rows(const uint8_t* blob, size_t gu_row, size_t up_off, int n, const void* const* act,
                    float* const* ff, int r0, int r1) {
    const block_q8_K* y[NT];
    for (int t = 0; t < NT; ++t) y[t] = (const block_q8_K*) act[t];
    const int nb = n / QK_K;
    float g[NT], u[NT];
    // two rows per pass at one token (that is where the decode is latency-bound); at NT >= 2 the tokens
    // already fill the issue window and a second row pair only adds register pressure (measured neutral).
    // (A four-row fused gate+up pass was measured: it spills on 16 XMM registers and loses 10-20%.)
    int r = r0;
    if (NT == 1) {
        float g2[2 * NT], u2[2 * NT];
        for (; r + 1 < r1; r += 2) {
            const uint8_t* rows[2] = {blob + (size_t) r * gu_row, blob + (size_t) (r + 1) * gu_row};
            row_dot_r<TY, NT, 2>(rows, nb, y, g2);
            const uint8_t* urows[2] = {blob + up_off + (size_t) r * gu_row, blob + up_off + (size_t) (r + 1) * gu_row};
            row_dot_r<TY, NT, 2>(urows, nb, y, u2);
            for (int t = 0; t < NT; ++t) {
                const float gr = g2[0 * NT + t], ur = u2[0 * NT + t];
                ff[t][r] = (gr / (1.f + std::exp(-gr))) * ur;
                const float gr1 = g2[1 * NT + t], ur1 = u2[1 * NT + t];
                ff[t][r + 1] = (gr1 / (1.f + std::exp(-gr1))) * ur1;
            }
        }
    }
    for (; r < r1; ++r) {
        row_dot<TY, NT>(blob + (size_t) r * gu_row, nb, y, g);
        row_dot<TY, NT>(blob + up_off + (size_t) r * gu_row, nb, y, u);
        for (int t = 0; t < NT; ++t) ff[t][r] = (g[t] / (1.f + std::exp(-g[t]))) * u[t];
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

// ---- IQ4_NL (20) down rows against Q8_0.  Unlike the other i-quants, ggml-cpu HAS a hand-written __AVX__
// path for this type (two blocks per iteration, 256-bit float accumulation, no FMA so mul+add), and it beat
// the first 128-bit kernel here.  The row loop takes the same two-block shape, but decodes each block pair
// ONCE and lets every token share it - which ggml's per-token dot cannot - and keeps the int32 madd chain
// (exact) with a single 256-bit cvt per pair.
template <int NT>
void iq4nl_rows(const uint8_t* w, size_t row_bytes, int n, const block_q8_0* const* y, float* const* out,
                int r0, int r1) {
    const __m128i values = _mm_loadu_si128((const __m128i*) kvalues_iq4nl);
    const __m128i m4b = _mm_set1_epi8(0x0f);
    const __m128i ones = _mm_set1_epi16(1);
    const int nb = n / QK4_NL;
    const int pf = prefetch_distance();
    for (int r = r0; r < r1; ++r) {
        const uint8_t* row = w + (size_t) r * row_bytes;
        __m256 acc[NT];
        for (int t = 0; t < NT; ++t) acc[t] = _mm256_setzero_ps();
        int ib = 0;
        for (; ib + 1 < nb; ib += 2) {
            const uint8_t* blk0 = row + (size_t) ib * sizeof(block_iq4_nl);
            const uint8_t* blk1 = blk0 + sizeof(block_iq4_nl);
            rows_ahead(blk0, pf);
            const __m128i bits0 = _mm_loadu_si128((const __m128i*) (blk0 + 2));
            const __m128i bits1 = _mm_loadu_si128((const __m128i*) (blk1 + 2));
            const __m128i v00 = _mm_shuffle_epi8(values, _mm_and_si128(bits0, m4b));
            const __m128i v01 = _mm_shuffle_epi8(values, _mm_and_si128(_mm_srli_epi16(bits0, 4), m4b));
            const __m128i v10 = _mm_shuffle_epi8(values, _mm_and_si128(bits1, m4b));
            const __m128i v11 = _mm_shuffle_epi8(values, _mm_and_si128(_mm_srli_epi16(bits1, 4), m4b));
            const __m128i a00 = _mm_sign_epi8(v00, v00), a01 = _mm_sign_epi8(v01, v01);   // |w|: the unsigned maddubs operand
            const __m128i a10 = _mm_sign_epi8(v10, v10), a11 = _mm_sign_epi8(v11, v11);
            const float dx0 = h2f(u16(blk0)), dx1 = h2f(u16(blk1));
            for (int t = 0; t < NT; ++t) {
                const uint8_t* q8a = (const uint8_t*) y[t][ib].qs;
                const uint8_t* q8b = (const uint8_t*) y[t][ib + 1].qs;
                const __m128i ya0 = _mm_loadu_si128((const __m128i*) q8a);
                const __m128i ya1 = _mm_loadu_si128((const __m128i*) (q8a + 16));
                const __m128i yb0 = _mm_loadu_si128((const __m128i*) q8b);
                const __m128i yb1 = _mm_loadu_si128((const __m128i*) (q8b + 16));
                const __m128i pa = _mm_add_epi32(
                    _mm_madd_epi16(_mm_maddubs_epi16(a00, _mm_sign_epi8(ya0, v00)), ones),
                    _mm_madd_epi16(_mm_maddubs_epi16(a01, _mm_sign_epi8(ya1, v01)), ones));
                const __m128i pb = _mm_add_epi32(
                    _mm_madd_epi16(_mm_maddubs_epi16(a10, _mm_sign_epi8(yb0, v10)), ones),
                    _mm_madd_epi16(_mm_maddubs_epi16(a11, _mm_sign_epi8(yb1, v11)), ones));
                const __m256 p = _mm256_cvtepi32_ps(_mm256_castps_si256(_mm256_set_m128(_mm_castsi128_ps(pb), _mm_castsi128_ps(pa))));
                const __m256 dd = _mm256_set_m128(_mm_set1_ps(dx1 * h2f(y[t][ib + 1].d)),
                                                  _mm_set1_ps(dx0 * h2f(y[t][ib].d)));
                acc[t] = _mm256_add_ps(acc[t], _mm256_mul_ps(dd, p));
            }
        }
        for (; ib < nb; ++ib) {           // odd tail, one block
            const uint8_t* blk = row + (size_t) ib * sizeof(block_iq4_nl);
            const __m128i bits = _mm_loadu_si128((const __m128i*) (blk + 2));
            const __m128i v0 = _mm_shuffle_epi8(values, _mm_and_si128(bits, m4b));
            const __m128i v1 = _mm_shuffle_epi8(values, _mm_and_si128(_mm_srli_epi16(bits, 4), m4b));
            const __m128i aq0 = _mm_sign_epi8(v0, v0), aq1 = _mm_sign_epi8(v1, v1);
            const float dx = h2f(u16(blk));
            for (int t = 0; t < NT; ++t) {
                const uint8_t* q8 = (const uint8_t*) y[t][ib].qs;
                const __m128i p0 = _mm_madd_epi16(_mm_maddubs_epi16(aq0, _mm_sign_epi8(_mm_loadu_si128((const __m128i*) q8), v0)), ones);
                const __m128i p1 = _mm_madd_epi16(_mm_maddubs_epi16(aq1, _mm_sign_epi8(_mm_loadu_si128((const __m128i*) (q8 + 16)), v1)), ones);
                const __m128 f = _mm_cvtepi32_ps(_mm_add_epi32(p0, p1));   // exact int sum, one cvt
                acc[t] = _mm256_add_ps(acc[t], _mm256_castps128_ps256(_mm_mul_ps(_mm_set1_ps(dx * h2f(y[t][ib].d)), f)));
            }
        }
        for (int t = 0; t < NT; ++t) {
            const __m128 s = _mm_add_ps(_mm256_castps256_ps128(acc[t]), _mm256_extractf128_ps(acc[t], 1));
            out[t][r] = hsum4(s);
        }
    }
}

// ---- Q2_0 (42) down rows against Q8_0: ggml-cpu has only the scalar generic dot for this type on x86.
// Codes {0,1,2,3} map to {c-1}: unpack to unsigned code bytes, dot them with maddubs, and subtract the
// activation's pair sums - all integer, so the per-block sum is EXACTLY the generic dot's sumi_block.
inline void q2_unpack(__m128i q, __m128i& k0, __m128i& k1) {
    const __m128i m3 = _mm_set1_epi8(3);
    const __m128i c0 = _mm_and_si128(q, m3);
    const __m128i c1 = _mm_and_si128(_mm_srli_epi16(q, 2), m3);
    const __m128i c2 = _mm_and_si128(_mm_srli_epi16(q, 4), m3);
    const __m128i c3 = _mm_and_si128(_mm_srli_epi32(q, 6), m3);
    const __m128i t01 = _mm_unpacklo_epi8(c0, c1);
    const __m128i t23 = _mm_unpacklo_epi8(c2, c3);
    k0 = _mm_unpacklo_epi16(t01, t23);   // codes 0..15
    k1 = _mm_unpackhi_epi16(t01, t23);   // codes 16..31
}

template <int NT>
void q2_0_rows(const uint8_t* w, size_t row_bytes, int n, const block_q8_0* const* y, float* const* out,
               int r0, int r1) {
    const __m128i ones = _mm_set1_epi16(1);
    const int nb = n / QK2_0;
    const int pf = prefetch_distance();
    // sum(q) per lane depends only on the activations, not the row: compute the per-lane y-sum vectors once
    // per call and reuse them across every row (the kernel runs thousands of rows against the same NT token
    // blocks).  Lane m of yv[t][i][k] = sum of the 8 k-half-0 and 8 k-half-1 activation values that p0/p1
    // lane m covers - so the int32 subtraction below is the same integer the per-row form computed.
    std::vector<__m128i> yv((size_t) NT * nb * 2);   // [t][i][k]
    for (int t = 0; t < NT; ++t) {
        for (int i = 0; i < nb; ++i) {
            for (int k = 0; k < 2; ++k) {
                const uint8_t* q8 = (const uint8_t*) y[t][i * 2 + k].qs;
                const __m128i s0 = _mm_madd_epi16(_mm_maddubs_epi16(_mm_set1_epi8(1), _mm_loadu_si128((const __m128i*) q8)), ones);
                const __m128i s1 = _mm_madd_epi16(_mm_maddubs_epi16(_mm_set1_epi8(1), _mm_loadu_si128((const __m128i*) (q8 + 16))), ones);
                yv[((size_t) t * nb + i) * 2 + k] = _mm_add_epi32(s0, s1);
            }
        }
    }
    // two rows per pass at one token (independent unpack/dot chains, same bits)
    int r = r0;
    if (NT == 1 && r1 - r0 >= 2) {
        for (; r + 1 < r1; r += 2) {
            const uint8_t* rowA = w + (size_t) r * row_bytes;
            const uint8_t* rowB = w + (size_t) (r + 1) * row_bytes;
            __m128 accA = _mm_setzero_ps(), accB = _mm_setzero_ps();
            for (int i = 0; i < nb; ++i) {
                const uint8_t* blka = rowA + (size_t) i * sizeof(block_q2_0);
                const uint8_t* blkb = rowB + (size_t) i * sizeof(block_q2_0);
                rows_ahead(blka, pf);
                rows_ahead(blkb, pf);
                const float dxa = h2f(u16(blka)), dxb = h2f(u16(blkb));
                for (int k = 0; k < 2; ++k) {
                    __m128i k0a, k1a, k0b, k1b;
                    q2_unpack(_mm_cvtsi64_si128((long long) u64(blka + 2 + 8 * k)), k0a, k1a);
                    q2_unpack(_mm_cvtsi64_si128((long long) u64(blkb + 2 + 8 * k)), k0b, k1b);
                    const uint8_t* q8 = (const uint8_t*) y[0][i * 2 + k].qs;
                    const __m128i y0 = _mm_loadu_si128((const __m128i*) q8);
                    const __m128i y1 = _mm_loadu_si128((const __m128i*) (q8 + 16));
                    const __m128i yv01 = yv[(size_t) i * 2 + k];   // NT == 1: token 0, [i][k]
                    const __m128i qa = _mm_sub_epi32(_mm_add_epi32(
                        _mm_madd_epi16(_mm_maddubs_epi16(k0a, y0), ones),
                        _mm_madd_epi16(_mm_maddubs_epi16(k1a, y1), ones)), yv01);
                    const __m128i qb = _mm_sub_epi32(_mm_add_epi32(
                        _mm_madd_epi16(_mm_maddubs_epi16(k0b, y0), ones),
                        _mm_madd_epi16(_mm_maddubs_epi16(k1b, y1), ones)), yv01);
                    const float dy = h2f(y[0][i * 2 + k].d);
                    accA = _mm_add_ps(accA, _mm_mul_ps(_mm_set1_ps(dxa * dy), _mm_cvtepi32_ps(qa)));
                    accB = _mm_add_ps(accB, _mm_mul_ps(_mm_set1_ps(dxb * dy), _mm_cvtepi32_ps(qb)));
                }
            }
            out[0][r] = hsum4(accA);
            out[0][r + 1] = hsum4(accB);
        }
    }
    for (; r < r1; ++r) {
        const uint8_t* row = w + (size_t) r * row_bytes;
        __m128 accf[NT];
        for (int t = 0; t < NT; ++t) accf[t] = _mm_setzero_ps();
        for (int i = 0; i < nb; ++i) {
            const uint8_t* blk = row + (size_t) i * sizeof(block_q2_0);
            rows_ahead(blk, pf);
            const float dx = h2f(u16(blk));
            const uint8_t* qs = blk + 2;
            for (int k = 0; k < 2; ++k) {
                // 8 code bytes -> 32 unsigned codes in order (c0..c3 = the four 2-bit groups of each byte)
                __m128i k0, k1;
                q2_unpack(_mm_cvtsi64_si128((long long) u64(qs + 8 * k)), k0, k1);
                for (int t = 0; t < NT; ++t) {
                    const uint8_t* q8 = (const uint8_t*) y[t][i * 2 + k].qs;
                    const __m128i y0 = _mm_loadu_si128((const __m128i*) q8);
                    const __m128i y1 = _mm_loadu_si128((const __m128i*) (q8 + 16));
                    // (c-1)*qy: c*qy at int16 level, minus the precomputed per-lane sum(qy) at int32 level
                    // (same integer as the per-row subtraction, all values exact in int32)
                    const __m128i p0 = _mm_madd_epi16(_mm_maddubs_epi16(k0, y0), ones);
                    const __m128i p1 = _mm_madd_epi16(_mm_maddubs_epi16(k1, y1), ones);
                    const __m128i q = _mm_sub_epi32(_mm_add_epi32(p0, p1), yv[((size_t) t * nb + i) * 2 + k]);
                    const __m128 f = _mm_cvtepi32_ps(q);
                    accf[t] = _mm_add_ps(accf[t], _mm_mul_ps(_mm_set1_ps(dx * h2f(y[t][i * 2 + k].d)), f));
                }
            }
        }
        for (int t = 0; t < NT; ++t) out[t][r] = hsum4(accf[t]);
    }
}

}  // namespace

bool iq128_supported(int type) noexcept {
    // IQ4_XS (23) is deliberately NOT here: its ggml generic dot auto-vectorizes well and beats the 128-bit
    // grid kernel (0.7-1.0x in iq_avx1_parity --bench on an E5-2470 v2); the generic stays for that format's
    // gate/up rows.  Its down rows (IQ4_NL/Q2_0) still take iq128_down_rows.  The kernel itself stays built
    // and parity-tested.
    return type == 18 || type == 21 || type == 22;
}

bool iq128_down_supported(int type) noexcept {
    // IQ4_NL (20) is deliberately NOT here: unlike the other i-quants, ggml-cpu has a hand-written __AVX__
    // dot for it (two blocks per iteration, 256-bit float accumulation) and it measures 1.15-1.2x against
    // the 128-bit kernel on an E5-2470 v2 even with the decode shared across tokens.  The generic stays for
    // IQ4_NL down rows; the kernel stays built and parity-tested.  Q2_0 (42) has no x86 dot in ggml-cpu at
    // all (scalar generic) and stays on iq128_down_rows at 4.2x.
    return type == 42;
}

void iq128_gu_rows(int type, const uint8_t* blob, size_t gu_row, size_t up_off, int n, const void* const* act,
                   int nt, float* const* ff, int r0, int r1) {
    switch (type) {
        case 18: gu_rows_nt<18>(nt, blob, gu_row, up_off, n, act, ff, r0, r1); break;
        case 21: gu_rows_nt<21>(nt, blob, gu_row, up_off, n, act, ff, r0, r1); break;
        case 22: gu_rows_nt<22>(nt, blob, gu_row, up_off, n, act, ff, r0, r1); break;
        case 23: gu_rows_nt<23>(nt, blob, gu_row, up_off, n, act, ff, r0, r1); break;
        default: break;
    }
}

void iq128_down_rows(int type, const uint8_t* w, size_t row_bytes, int n, const void* const* hq, int nt,
                     float* const* out, int r0, int r1) {
    const block_q8_0* y[8];
    for (int t = 0; t < nt && t < 8; ++t) y[t] = (const block_q8_0*) hq[t];
    if (type == 20) {
        for (int t0 = 0; t0 < nt; t0 += 4) {
            const int k = nt - t0 < 4 ? nt - t0 : 4;
            switch (k) {
                case 1: iq4nl_rows<1>(w, row_bytes, n, y + t0, out + t0, r0, r1); break;
                case 2: iq4nl_rows<2>(w, row_bytes, n, y + t0, out + t0, r0, r1); break;
                case 3: iq4nl_rows<3>(w, row_bytes, n, y + t0, out + t0, r0, r1); break;
                default: iq4nl_rows<4>(w, row_bytes, n, y + t0, out + t0, r0, r1); break;
            }
        }
        return;
    }
    // Q2_0: the same 4-at-a-time chunking (NT is capped at 8 by the pool; prefill chunks larger than 8
    // loop here, the per-token rows identical either way).
    for (int t0 = 0; t0 < nt; t0 += 4) {
        const int k = nt - t0 < 4 ? nt - t0 : 4;
        switch (k) {
            case 1: q2_0_rows<1>(w, row_bytes, n, y + t0, out + t0, r0, r1); break;
            case 2: q2_0_rows<2>(w, row_bytes, n, y + t0, out + t0, r0, r1); break;
            case 3: q2_0_rows<3>(w, row_bytes, n, y + t0, out + t0, r0, r1); break;
            default: q2_0_rows<4>(w, row_bytes, n, y + t0, out + t0, r0, r1); break;
        }
    }
}

// ggml's quantize_row_q8_K_ref in SSE4.2/AVX1: q8k_quant_avx2's arithmetic, 128-bit lanes (its packs need no
// lane-crossing permute at 128-bit).  The same per-value operations as the scalar reference - one IEEE multiply
// by the same iscale, the same 1.5*2^23 rounding add, the same clamp - so the bytes are identical
// (q8k_quant_parity checks it), including the NaN and degenerate-block behavior: `ax > amax` is false for a NaN
// and max_ps returns its second operand for a NaN first one, and a zero block leaves bsums as the reference does.
void q8k_quant_avx1(const float* x, void* vy, int64_t k) {
    block_q8_K* y = (block_q8_K*) vy;
    const int64_t nb = k / QK_K;
    const __m128 absm = _mm_castsi128_ps(_mm_set1_epi32(0x7fffffff));
    const __m128 magic = _mm_set1_ps(12582912.f);
    const __m128i mant = _mm_set1_epi32(0x007fffff), off = _mm_set1_epi32(0x00400000);
    const __m128i c127 = _mm_set1_epi32(127);
    for (int64_t i = 0; i < nb; ++i, x += QK_K) {
        __m128 m = _mm_setzero_ps();
        for (int j = 0; j < QK_K; j += 4) m = _mm_max_ps(_mm_and_ps(_mm_loadu_ps(x + j), absm), m);
        __m128 h = _mm_max_ps(m, _mm_movehl_ps(m, m));
        h = _mm_max_ss(h, _mm_movehdup_ps(h));
        const float amax = _mm_cvtss_f32(h);
        if (!amax) {
            y[i].d = 0;
            std::memset(y[i].qs, 0, QK_K);
            continue;
        }
        float mx = 0.f;
        const __m128 va = _mm_set1_ps(amax);
        for (int j = 0; j < QK_K; j += 4) {
            const int msk = _mm_movemask_ps(_mm_cmpeq_ps(_mm_and_ps(_mm_loadu_ps(x + j), absm), va));
            if (msk) {
                int b = 0;
                while (!((msk >> b) & 1)) ++b;
                mx = x[j + b];
                break;
            }
        }
        const float iscale = -127.f / mx;
        const __m128 vis = _mm_set1_ps(iscale);
        for (int j = 0; j < QK_K; j += 16) {
            __m128i v[4];
            for (int q = 0; q < 4; ++q) {
                __m128 p = _mm_mul_ps(vis, _mm_loadu_ps(x + j + 4 * q));
#if defined(__GNUC__) && !defined(__clang__)
                // GCC fuses a multiply and an add into an FMA by default (-ffp-contract=fast); the reference
                // rounds the product before the add, so the product must stay a separate value (no FMA under
                // -mavx, but the barrier keeps the contract if the file is ever built with a newer floor)
                __asm__("" : "+x"(p));
#endif
                const __m128 t = _mm_add_ps(p, magic);
                v[q] = _mm_min_epi32(_mm_sub_epi32(_mm_and_si128(_mm_castps_si128(t), mant), off), c127);
                // the reference stores MIN(127, v) into an int8 (keeps the low byte), and its bsums add those
                // bytes: the same here, so even a degenerate block (iscale overflowing to inf) gives the same bytes
                v[q] = _mm_srai_epi32(_mm_slli_epi32(v[q], 24), 24);
            }
            __m128i s = _mm_add_epi32(_mm_add_epi32(v[0], v[1]), _mm_add_epi32(v[2], v[3]));
            s = _mm_add_epi32(s, _mm_shuffle_epi32(s, 0x4E));
            s = _mm_add_epi32(s, _mm_shuffle_epi32(s, 0xB1));
            y[i].bsums[j / 16] = (int16_t) _mm_cvtsi128_si32(s);
            _mm_storeu_si128((__m128i*) (y[i].qs + j),
                             _mm_packs_epi16(_mm_packs_epi32(v[0], v[1]), _mm_packs_epi32(v[2], v[3])));
        }
        y[i].d = 1 / iscale;
    }
}

}  // namespace strata::kernels::cpu
