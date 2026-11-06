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

// sgn16() as a 256-entry byte -> 8 sign bytes table (2 KB, L1-resident): byte b -> 0xFF in byte k where bit k
// of b is set, 0x01 elsewhere.  Same bits as sgn16's and/cmpeq/or, two loads instead of three vector ops.
struct Sign8 {
    uint64_t v[256];
    constexpr Sign8() : v{} {
        for (int i = 0; i < 256; ++i) {
            uint64_t r = 0;
            for (int k = 0; k < 8; ++k) r |= (uint64_t) (((i >> k) & 1) ? 0xFF : 0x01) << (8 * k);
            v[i] = r;
        }
    }
};
static constexpr Sign8 sign8{};
static_assert(sign8.v[0] == 0x0101010101010101ull && sign8.v[1] == 0x01010101010101FFull,
              "sign8: byte k = 0xFF when bit k of i is set");

// The sub-block scale (2k+1, k = 0..15) pre-broadcast over the 8 int16 lanes: one aligned 128-bit load on the
// load ports instead of vmovd + vpshufd + two vpshufb.  vpshufb is port 1 only on Ivy Bridge and port 1
// measures 61 % busy in this loop, ports 2/3 (loads) 36 %.  16 entries x 16 B = 256 B, 4 cache lines.
struct KScale16 {
    alignas(16) int16_t v[16][8];
    constexpr KScale16() : v{} {
        for (int k = 0; k < 16; ++k)
            for (int l = 0; l < 8; ++l) v[k][l] = (int16_t) (2 * k + 1);
    }
};
static constexpr KScale16 kscale16{};
static_assert(kscale16.v[0][0] == 1 && kscale16.v[15][7] == 31, "kscale16[k] == _mm_set1_epi16(2k+1)");

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
        c0 = c1 = _mm_load_si128((const __m128i*) kscale16.v[w >> 28]);
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
        c0 = c1 = _mm_load_si128((const __m128i*) kscale16.v[half ? s >> 4 : s & 15]);
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
        const uint8_t* sp = b + 2 + 32 + 8 * j + 4 * half;   // sgn16(sg,0) == [sign8[sp[0]], sign8[sp[1]]]
        s0 = _mm_set_epi64x((long long) sign8.v[sp[1]], (long long) sign8.v[sp[0]]);
        s1 = _mm_set_epi64x((long long) sign8.v[sp[3]], (long long) sign8.v[sp[2]]);
        const uint8_t sb = b[74 + 2 * j + half];
        c0 = _mm_load_si128((const __m128i*) kscale16.v[sb & 15]);
        c1 = _mm_load_si128((const __m128i*) kscale16.v[sb >> 4]);
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

// The grid tables pack four int8 magnitudes per uint32 entry in this ggml (GGML_TABLE_BEGIN(uint32_t, ...)).
constexpr int grid_byte_max(const uint32_t* t, int n) {
    int m = 0;
    for (int i = 0; i < n; ++i)
        for (int s = 0; s < 4; ++s) { const int b = (int) ((t[i] >> (8 * s)) & 0xFF); if (b > m) m = b; }
    return m;
}

// madd_add_fold is madd_add with the two 16-value halves of a 32-value sub-block folded into one int16 vector
// before the scale multiply: when c0 == c1 == c, madd(d0,c) + madd(d1,c) == madd(d0+d1, c) exactly, so one
// pmaddwd (ports 0/1 only) is replaced by one paddw (ports 0, 1 and 5).  Bit for bit the same int32 sum, as
// long as d0+d1 stays inside int16: each half is |d| <= 2*grid_max*127, so the pair stays under 4*127*max.
// Guarded against the tables themselves, not against a hand-written constant, because the IQ3_XXS margin is
// only 4% (31496 vs 32767) and a table change would otherwise turn this into silent saturation.
// The margin here is only 4%: this ggml packs four int8 magnitudes per uint32 table entry, iq3xxs_grid
// reaches 62 (4*127*62 = 31496 < 32768) and iq3s_grid reaches 15 (7620).  iq_avx1_parity scans both tables and
// fails if that ever stops holding, because the failure mode would be silent saturation.
inline __m128i madd_add_fold(__m128i acc, __m128i g0, __m128i g1, __m128i sgn0, __m128i sgn1,
                             __m128i y0, __m128i y1, __m128i sc) {
    const __m128i d = _mm_add_epi16(_mm_maddubs_epi16(g0, _mm_sign_epi8(y0, sgn0)),
                                    _mm_maddubs_epi16(g1, _mm_sign_epi8(y1, sgn1)));
    return _mm_add_epi32(acc, _mm_madd_epi16(d, sc));
}

// ---- the row kernels (iq_avx2_rows.inl's row_dot in 128-bit lanes)
// R rows of the SAME expert through one block loop: the per-half decode chains (index byte -> grid table ->
// madd) are ~20 cycles of pure latency and at NT=1 there is nothing to overlap, so the kernel runs latency-
// bound at ~1/4 of its issue rate.  Two rows interleave independent chains in the same registers and roughly
// double the throughput; the per-row integer sums and the float stage are unchanged, bit for bit.
template <int TY, int NT, int R, bool HOIST>
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
                // At one token the activation bytes are the same for every row in the pair and for both
                // projections, so load them once per unit instead of once per row.  Only for NT == 1: with more
                // tokens the hoisted vectors themselves (2 per token) push the 16 XMM registers over and spill.
                __m128i yv[NT][2];
                if constexpr (HOIST) {
                    const uint8_t* q8 = (const uint8_t*) (y[0][i].qs + off);
                    yv[0][0] = _mm_loadu_si128((const __m128i*) q8);
                    yv[0][1] = _mm_loadu_si128((const __m128i*) (q8 + 16));
                }
                // c0 == c1 in IQ3_XXS and IQ3_S (one 4-bit scale per 32 values): fold the halves, one pmaddwd
                // instead of two.  IQ2_S (two scales) and IQ4_XS (grid up to 127, would saturate) stay on the
                // two-accumulator path - and the fold needs both halves' decode, so it is per row, not shared.
                if constexpr (TY == 18 || TY == 21) {
                    for (int rr = 0; rr < R; ++rr) {
                        for (int t = 0; t < NT; ++t) {
                            const uint8_t* q8 = (const uint8_t*) (y[t][i].qs + off);
                            if constexpr (HOIST) {
                                acci[rr][t][0] = madd_add_fold(acci[rr][t][0], g0[rr], g1[rr], s0[rr], s1[rr],
                                                               yv[0][0], yv[0][1], c0[rr]);
                            } else {
                                acci[rr][t][0] = madd_add_fold(acci[rr][t][0], g0[rr], g1[rr], s0[rr], s1[rr],
                                                               _mm_loadu_si128((const __m128i*) q8),
                                                               _mm_loadu_si128((const __m128i*) (q8 + 16)), c0[rr]);
                            }
                        }
                    }
                    continue;
                }
                for (int rr = 0; rr < R; ++rr) {
                    for (int t = 0; t < NT; ++t) {
                        const uint8_t* q8 = (const uint8_t*) (y[t][i].qs + off);
                        if constexpr (HOIST) {
                            acci[rr][t][0] = madd_add(acci[rr][t][0], g0[rr], s0[rr], yv[0][0], c0[rr]);
                            acci[rr][t][1] = madd_add(acci[rr][t][1], g1[rr], s1[rr], yv[0][1], c1[rr]);
                        } else {
                            acci[rr][t][0] = madd_add(acci[rr][t][0], g0[rr], s0[rr], _mm_loadu_si128((const __m128i*) q8), c0[rr]);
                            acci[rr][t][1] = madd_add(acci[rr][t][1], g1[rr], s1[rr], _mm_loadu_si128((const __m128i*) (q8 + 16)), c1[rr]);
                        }
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
    row_dot_r<TY, NT, 1, false>(rows, nblocks, y, res);
}

template <int TY, int NT, int MODE>
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
    constexpr bool hoist = (MODE != 0 && NT == 1);
    if (NT == 1 && MODE == 3) {
        // four chains: two rows' gate and up in one pass (R = 4). 8 decoded vectors + 8 accumulators against
        // 16 XMM registers, so this measures what the spills really cost at one token.
        float g4[4 * NT];
        for (; r + 1 < r1; r += 2) {
            const uint8_t* rows[4] = {blob + (size_t) r * gu_row, blob + up_off + (size_t) r * gu_row,
                                      blob + (size_t) (r + 1) * gu_row, blob + up_off + (size_t) (r + 1) * gu_row};
            row_dot_r<TY, NT, 4, true>(rows, nb, y, g4);
            for (int t = 0; t < NT; ++t) {
                const float g0 = g4[0 * NT + t], g1 = g4[2 * NT + t];
                ff[t][r] = (g0 / (1.f + std::exp(-g0))) * g4[1 * NT + t];
                ff[t][r + 1] = (g1 / (1.f + std::exp(-g1))) * g4[3 * NT + t];
            }
        }
    }
    if (NT == 1 && MODE == 1) {
        float g2[2 * NT];
        // Pair ONE gate row with ONE up row (two independent chains, the second row's pointers are the same
        // stride), instead of two gate rows then two up rows: the two projections share the activation bytes,
        // so the hoisted loads above serve both, and one call replaces two.
        for (; r < r1; ++r) {
            const uint8_t* rows[2] = {blob + (size_t) r * gu_row, blob + up_off + (size_t) r * gu_row};
            row_dot_r<TY, NT, 2, hoist>(rows, nb, y, g2);   // g2[0] = gate, g2[1] = up
            for (int t = 0; t < NT; ++t) {
                const float gr = g2[0 * NT + t];
                ff[t][r] = (gr / (1.f + std::exp(-gr))) * g2[1 * NT + t];
            }
        }
    }
    for (; r < r1; ++r) {
        row_dot<TY, NT>(blob + (size_t) r * gu_row, nb, y, g);
        row_dot<TY, NT>(blob + up_off + (size_t) r * gu_row, nb, y, u);
        for (int t = 0; t < NT; ++t) ff[t][r] = (g[t] / (1.f + std::exp(-g[t]))) * u[t];
    }
}


// MODE 4 (measurement): IQ2_S at one token, the 32-value unit decoded in two 16-value phases so fewer vectors
// are live at once - 3 per row per phase (g, s, c) instead of 6 - which is what round 7 claims lets R = 2 run
// with no spills. The index build stays once per 32 values (their sample decoded each half independently and
// paid for grid_idx twice). Same products, same accumulators, so the bits cannot move.
static inline void iq2s_dec16(__m128i idx, const uint8_t* sp, uint8_t sb, int which,
                              __m128i& g, __m128i& s, __m128i& c) {
    g = _mm_set_epi64x((long long) iq2s_grid[_mm_extract_epi16(idx, 2 * which + 1)],
                       (long long) iq2s_grid[_mm_extract_epi16(idx, 2 * which)]);
    s = _mm_set_epi64x((long long) sign8.v[sp[2 * which + 1]], (long long) sign8.v[sp[2 * which]]);
    c = _mm_load_si128((const __m128i*) kscale16.v[which ? (sb >> 4) : (sb & 15)]);
}

template <int TY>
inline void gu_rows_phased(const uint8_t* blob, size_t gu_row, size_t up_off, int n, const void* const* act,
                           float* const* ff, int r0, int r1) {
    const block_q8_K* y0 = (const block_q8_K*) act[0];
    const int nb = n / QK_K;
    for (int r = r0; r < r1; ++r) {
        const uint8_t* gbase = blob + (size_t) r * gu_row;
        const uint8_t* ubase = blob + up_off + (size_t) r * gu_row;
        __m128 fg = _mm_setzero_ps(), fu = _mm_setzero_ps();
        for (int i = 0; i < nb; ++i) {
            const uint8_t* gb = gbase + (size_t) i * Fmt128<TY>::bytes;
            const uint8_t* ub = ubase + (size_t) i * Fmt128<TY>::bytes;
            const uint8_t* q8 = (const uint8_t*) y0[i].qs;
            __m128i ag0 = _mm_setzero_si128(), ag1 = _mm_setzero_si128();
            __m128i au0 = _mm_setzero_si128(), au1 = _mm_setzero_si128();
            for (int j = 0; j < 4; ++j) {
                for (int half = 0; half < 2; ++half) {
                    const int off = 64 * j + 32 * half;
                    const __m128i ig = grid_idx(gb + 2 + 8 * j + 4 * half, hi_spread.iq2s[gb[66 + 2 * j + half]]);
                    const __m128i iu = grid_idx(ub + 2 + 8 * j + 4 * half, hi_spread.iq2s[ub[66 + 2 * j + half]]);
                    const uint8_t* spg = gb + 2 + 32 + 8 * j + 4 * half;
                    const uint8_t* spu = ub + 2 + 32 + 8 * j + 4 * half;
                    const uint8_t sbg = gb[74 + 2 * j + half], sbu = ub[74 + 2 * j + half];
                    const __m128i yv0 = _mm_loadu_si128((const __m128i*) (q8 + off));
                    __m128i g, s, c;
                    iq2s_dec16(ig, spg, sbg, 0, g, s, c); ag0 = madd_add(ag0, g, s, yv0, c);
                    iq2s_dec16(iu, spu, sbu, 0, g, s, c); au0 = madd_add(au0, g, s, yv0, c);
                    const __m128i yv1 = _mm_loadu_si128((const __m128i*) (q8 + off + 16));
                    iq2s_dec16(ig, spg, sbg, 1, g, s, c); ag1 = madd_add(ag1, g, s, yv1, c);
                    iq2s_dec16(iu, spu, sbu, 1, g, s, c); au1 = madd_add(au1, g, s, yv1, c);
                }
            }
            const float dy = y0[i].d;
            fg = _mm_add_ps(fg, _mm_mul_ps(_mm_set1_ps(h2f(u16(gb)) * Fmt128<TY>::K * dy),
                                           _mm_cvtepi32_ps(_mm_add_epi32(ag0, ag1))));
            fu = _mm_add_ps(fu, _mm_mul_ps(_mm_set1_ps(h2f(u16(ub)) * Fmt128<TY>::K * dy),
                                           _mm_cvtepi32_ps(_mm_add_epi32(au0, au1))));
        }
        float hg = hsum4(fg), hu = hsum4(fu);
        ff[0][r] = (hg / (1.f + std::exp(-hg))) * hu;
    }
}


// MODE 5 (measurement): the same phasing applied to IQ3_S, which is a folded format.  The fold sums both halves'
// int16 products before ONE pmaddwd; a 16-value phase cannot do that, so this pays one extra pmaddwd and one
// extra accumulator per 32 values to get the live set down to 3 + 2 per row.  Whether the registers are worth
// more than the fold is exactly what this measures.
static inline void iq3s_dec16(__m128i idx, uint32_t sg, uint8_t sc, int half, int which,
                              __m128i& g, __m128i& s, __m128i& c) {
    const int m = 4 * which;
#define G4(o) (int) iq3s_grid[_mm_extract_epi16(idx, m + (o))]
    g = _mm_set_epi32(G4(3), G4(2), G4(1), G4(0));
#undef G4
    s = sgn16(sg, which);
    c = _mm_load_si128((const __m128i*) kscale16.v[half ? sc >> 4 : sc & 15]);
}

template <int TY>
inline void gu_rows_phased30(const uint8_t* blob, size_t gu_row, size_t up_off, int n, const void* const* act,
                             float* const* ff, int r0, int r1) {
    const block_q8_K* y0 = (const block_q8_K*) act[0];
    const int nb = n / QK_K;
    for (int r = r0; r < r1; ++r) {
        const uint8_t* gbase = blob + (size_t) r * gu_row;
        const uint8_t* ubase = blob + up_off + (size_t) r * gu_row;
        __m128 fg = _mm_setzero_ps(), fu = _mm_setzero_ps();
        for (int i = 0; i < nb; ++i) {
            const uint8_t* gb = gbase + (size_t) i * Fmt128<TY>::bytes;
            const uint8_t* ub = ubase + (size_t) i * Fmt128<TY>::bytes;
            const uint8_t* q8 = (const uint8_t*) y0[i].qs;
            __m128i ag0 = _mm_setzero_si128(), ag1 = _mm_setzero_si128();
            __m128i au0 = _mm_setzero_si128(), au1 = _mm_setzero_si128();
            for (int j = 0; j < 4; ++j) {
                for (int half = 0; half < 2; ++half) {
                    const int off = 64 * j + 32 * half;
                    const __m128i ig = grid_idx(gb + 2 + 16 * j + 8 * half, hi_spread.iq3s[gb[66 + 2 * j + half]]);
                    const __m128i iu = grid_idx(ub + 2 + 16 * j + 8 * half, hi_spread.iq3s[ub[66 + 2 * j + half]]);
                    const uint32_t sgs = u32(gb + 74 + 8 * j + 4 * half), sgsu = u32(ub + 74 + 8 * j + 4 * half);
                    const uint8_t sbg = gb[106 + j], sbu = ub[106 + j];
                    __m128i g, s, c;
                    const __m128i yv0 = _mm_loadu_si128((const __m128i*) (q8 + off));
                    iq3s_dec16(ig, sgs, sbg, half, 0, g, s, c); ag0 = madd_add(ag0, g, s, yv0, c);
                    iq3s_dec16(iu, sgsu, sbu, half, 0, g, s, c); au0 = madd_add(au0, g, s, yv0, c);
                    const __m128i yv1 = _mm_loadu_si128((const __m128i*) (q8 + off + 16));
                    iq3s_dec16(ig, sgs, sbg, half, 1, g, s, c); ag1 = madd_add(ag1, g, s, yv1, c);
                    iq3s_dec16(iu, sgsu, sbu, half, 1, g, s, c); au1 = madd_add(au1, g, s, yv1, c);
                }
            }
            const float dy = y0[i].d;
            fg = _mm_add_ps(fg, _mm_mul_ps(_mm_set1_ps(h2f(u16(gb)) * Fmt128<TY>::K * dy),
                                           _mm_cvtepi32_ps(_mm_add_epi32(ag0, ag1))));
            fu = _mm_add_ps(fu, _mm_mul_ps(_mm_set1_ps(h2f(u16(ub)) * Fmt128<TY>::K * dy),
                                           _mm_cvtepi32_ps(_mm_add_epi32(au0, au1))));
        }
        float hg = hsum4(fg), hu = hsum4(fu);
        ff[0][r] = (hg / (1.f + std::exp(-hg))) * hu;
    }
}


// MODE 6 (measurement): the phased IQ2_S shape at TWO tokens.  At one token the phasing won because the live set
// hit the 16-register wall; at two tokens the shipped path amortises one decode over two tokens and needs fewer
// registers per row, so the register argument may not apply.  This says whether the phasing generalises or is a
// one-token-only fix.
template <int TY, int NT>
inline void gu_rows_phased2(const uint8_t* blob, size_t gu_row, size_t up_off, int n, const void* const* act,
                            float* const* ff, int r0, int r1) {
    const block_q8_K* y[NT];
    for (int t = 0; t < NT; ++t) y[t] = (const block_q8_K*) act[t];
    const int nb = n / QK_K;
    for (int r = r0; r < r1; ++r) {
        const uint8_t* gbase = blob + (size_t) r * gu_row;
        const uint8_t* ubase = blob + up_off + (size_t) r * gu_row;
        __m128 fg[NT], fu[NT];
        for (int t = 0; t < NT; ++t) fg[t] = fu[t] = _mm_setzero_ps();
        for (int i = 0; i < nb; ++i) {
            const uint8_t* gb = gbase + (size_t) i * Fmt128<TY>::bytes;
            const uint8_t* ub = ubase + (size_t) i * Fmt128<TY>::bytes;
            __m128i ag[NT][2], au[NT][2];
            for (int t = 0; t < NT; ++t)
                for (int w = 0; w < 2; ++w) ag[t][w] = au[t][w] = _mm_setzero_si128();
            for (int j = 0; j < 4; ++j) {
                for (int half = 0; half < 2; ++half) {
                    const int off = 64 * j + 32 * half;
                    const __m128i ig = grid_idx(gb + 2 + 8 * j + 4 * half, hi_spread.iq2s[gb[66 + 2 * j + half]]);
                    const __m128i iu = grid_idx(ub + 2 + 8 * j + 4 * half, hi_spread.iq2s[ub[66 + 2 * j + half]]);
                    const uint8_t* spg = gb + 2 + 32 + 8 * j + 4 * half;
                    const uint8_t* spu = ub + 2 + 32 + 8 * j + 4 * half;
                    const uint8_t sbg = gb[74 + 2 * j + half], sbu = ub[74 + 2 * j + half];
                    __m128i g, sg_, c;
                    for (int which = 0; which < 2; ++which) {
                        iq2s_dec16(ig, spg, sbg, which, g, sg_, c);
                        for (int t = 0; t < NT; ++t) {
                            const __m128i yv = _mm_loadu_si128((const __m128i*) (y[t][i].qs + off + 16 * which));
                            ag[t][which] = madd_add(ag[t][which], g, sg_, yv, c);
                        }
                        iq2s_dec16(iu, spu, sbu, which, g, sg_, c);
                        for (int t = 0; t < NT; ++t) {
                            const __m128i yv = _mm_loadu_si128((const __m128i*) (y[t][i].qs + off + 16 * which));
                            au[t][which] = madd_add(au[t][which], g, sg_, yv, c);
                        }
                    }
                }
            }
            const float dg = h2f(u16(gb)) * Fmt128<TY>::K, du = h2f(u16(ub)) * Fmt128<TY>::K;
            for (int t = 0; t < NT; ++t) {
                const float dy = y[t][i].d;
                fg[t] = _mm_add_ps(fg[t], _mm_mul_ps(_mm_set1_ps(dg * dy), _mm_cvtepi32_ps(_mm_add_epi32(ag[t][0], ag[t][1]))));
                fu[t] = _mm_add_ps(fu[t], _mm_mul_ps(_mm_set1_ps(du * dy), _mm_cvtepi32_ps(_mm_add_epi32(au[t][0], au[t][1]))));
            }
        }
        for (int t = 0; t < NT; ++t) {
            const float hg = hsum4(fg[t]), hu = hsum4(fu[t]);
            ff[t][r] = (hg / (1.f + std::exp(-hg))) * hu;
        }
    }
}


// MODE 7 (measurement): phasing a FOLDED format while KEEPING the fold.  Mode 5 un-folded and that was my error:
// the fold is only an int16 add of the two halves' partial products before ONE pmaddwd, so a phase can build that
// int16 sum incrementally.  Live per row per phase: g, s, c + the int16 partial + one int32 accumulator = 5, the
// same as the phased IQ2_S shape, with no extra pmaddwd and bit-identical by construction (same ops, same order).
static inline void iq3x_dec16(int ty, const uint8_t* b, int j, int half, int which, __m128i& g, __m128i& s,
                              __m128i& c) {
    if (ty == 18) {                                        // IQ3_XXS: byte grid indices, u32 sign+scale word
        const uint8_t* q = b + 2 + 16 * j + 8 * half + 4 * which;
#define GX(o) (int) iq3xxs_grid[q[o]]
        g = _mm_set_epi32(GX(3), GX(2), GX(1), GX(0));
#undef GX
        const uint32_t w = u32(b + 2 + 64 + 8 * j + 4 * half);
        const int a = which ? ((w >> 14) & 127) : (w & 127), e2 = which ? ((w >> 21) & 127) : ((w >> 7) & 127);
        s = _mm_set_epi64x((long long) even_signs.v[e2], (long long) even_signs.v[a]);
        c = _mm_load_si128((const __m128i*) kscale16.v[w >> 28]);
    } else {                                               // IQ3_S: 9-bit indices, sign word, shared sub-scale
        const __m128i idx = grid_idx(b + 2 + 16 * j + 8 * half, hi_spread.iq3s[b[66 + 2 * j + half]]);
        const int m = 4 * which;
#define GS(o) (int) iq3s_grid[_mm_extract_epi16(idx, m + (o))]
        g = _mm_set_epi32(GS(3), GS(2), GS(1), GS(0));
#undef GS
        s = sgn16(u32(b + 74 + 8 * j + 4 * half), which);
        const uint8_t sc = b[106 + j];
        c = _mm_load_si128((const __m128i*) kscale16.v[half ? sc >> 4 : sc & 15]);
    }
}

template <int TY>
inline void gu_rows_phased31(const uint8_t* blob, size_t gu_row, size_t up_off, int n, const void* const* act,
                             float* const* ff, int r0, int r1) {
    const block_q8_K* y0 = (const block_q8_K*) act[0];
    const int nb = n / QK_K;
    for (int r = r0; r < r1; ++r) {
        const uint8_t* gbase = blob + (size_t) r * gu_row;
        const uint8_t* ubase = blob + up_off + (size_t) r * gu_row;
        __m128 fg = _mm_setzero_ps(), fu = _mm_setzero_ps();
        for (int i = 0; i < nb; ++i) {
            const uint8_t* gb = gbase + (size_t) i * Fmt128<TY>::bytes;
            const uint8_t* ub = ubase + (size_t) i * Fmt128<TY>::bytes;
            const uint8_t* q8 = (const uint8_t*) y0[i].qs;
            __m128i ag = _mm_setzero_si128(), au = _mm_setzero_si128();
            for (int j = 0; j < 4; ++j) {
                for (int half = 0; half < 2; ++half) {
                    const int off = 64 * j + 32 * half;
                    __m128i pg = _mm_setzero_si128(), pu = _mm_setzero_si128(), g, sg_, c, ct;
                    iq3x_dec16(TY, gb, j, half, 0, g, sg_, c); ct = c;
                    pg = _mm_maddubs_epi16(g, _mm_sign_epi8(_mm_loadu_si128((const __m128i*) (q8 + off)), sg_));
                    iq3x_dec16(TY, ub, j, half, 0, g, sg_, c);
                    pu = _mm_maddubs_epi16(g, _mm_sign_epi8(_mm_loadu_si128((const __m128i*) (q8 + off)), sg_));
                    iq3x_dec16(TY, gb, j, half, 1, g, sg_, c);
                    pg = _mm_add_epi16(pg, _mm_maddubs_epi16(g, _mm_sign_epi8(_mm_loadu_si128((const __m128i*) (q8 + off + 16)), sg_)));
                    iq3x_dec16(TY, ub, j, half, 1, g, sg_, c);
                    pu = _mm_add_epi16(pu, _mm_maddubs_epi16(g, _mm_sign_epi8(_mm_loadu_si128((const __m128i*) (q8 + off + 16)), sg_)));
                    ag = _mm_add_epi32(ag, _mm_madd_epi16(pg, ct));
                    au = _mm_add_epi32(au, _mm_madd_epi16(pu, c));
                }
            }
            const float dy = y0[i].d;
            fg = _mm_add_ps(fg, _mm_mul_ps(_mm_set1_ps(h2f(u16(gb)) * Fmt128<TY>::K * dy), _mm_cvtepi32_ps(ag)));
            fu = _mm_add_ps(fu, _mm_mul_ps(_mm_set1_ps(h2f(u16(ub)) * Fmt128<TY>::K * dy), _mm_cvtepi32_ps(au)));
        }
        float hg = hsum4(fg), hu = hsum4(fu);
        ff[0][r] = (hg / (1.f + std::exp(-hg))) * hu;
    }
}

// MODES 10 / 11 / 12 (measurement): the two shipped phased shapes with the two changes the stall census points
// at, each alone and combined.  The census (service stopped, pinned core, all counters 100% enabled): 2.98 uops
// issued per cycle of 4 IDQ slots, only 0.121 undelivered, 2.61 instructions retired per cycle - the loop is FED
// and WAITING, so the limit is chains and ports, not instruction count.  Both changes below are bit-identical by
// construction: they move loads and reassociate EXACT integer adds; no value, no rounding and no float grouping
// moves anywhere.
//   PK - read the index / sign bytes that sit next to each other as one 32-bit load and extract with shifts,
//        instead of one byte load each (IQ3_XXS reads FOUR byte loads per grid vector, IQ2_S two per sign
//        vector, per row per 16 values).  Same table entries, same order; it removes load-port uops, which are
//        2 per cycle on this core, from a loop that the census says is waiting downstream of the decoder.
//   SP - split the int32 accumulator chain by half of the 32-value unit and merge at the end of the unit,
//        instead of chaining every unit through one accumulator.  The chain is one add per step deep and the two
//        halves are independent; int32 addition here is exact (max |lane| = 4 * 127 * 127 = 64516, times 8 units
//        = 516128, far inside int32), so the merged value is bit for bit what the chained form produces.
static inline void iq3xxs_dec16_pk(const uint8_t* b, int j, int half, int which, __m128i& g, __m128i& s, __m128i& c) {
    const uint32_t qi = u32(b + 2 + 16 * j + 8 * half + 4 * which);        // four grid-index bytes: one load
#define GXB(o) (int) iq3xxs_grid[(qi >> (8 * (o))) & 0xFF]
    g = _mm_set_epi32(GXB(3), GXB(2), GXB(1), GXB(0));
#undef GXB
    const uint32_t w = u32(b + 2 + 64 + 8 * j + 4 * half);                 // sign + sub-scale word: one load (as shipped)
    const int a = which ? ((w >> 14) & 127) : (w & 127), e2 = which ? ((w >> 21) & 127) : ((w >> 7) & 127);
    s = _mm_set_epi64x((long long) even_signs.v[e2], (long long) even_signs.v[a]);
    c = _mm_load_si128((const __m128i*) kscale16.v[w >> 28]);
}

static inline void iq2s_dec16_pk(__m128i idx, uint32_t sw, uint8_t sb, int which,
                                 __m128i& g, __m128i& s, __m128i& c) {
    g = _mm_set_epi64x((long long) iq2s_grid[_mm_extract_epi16(idx, 2 * which + 1)],
                       (long long) iq2s_grid[_mm_extract_epi16(idx, 2 * which)]);
    s = _mm_set_epi64x((long long) sign8.v[(sw >> (8 * (2 * which + 1))) & 0xFF],   // four sign bytes: one load
                       (long long) sign8.v[(sw >> (8 * (2 * which))) & 0xFF]);
    c = _mm_load_si128((const __m128i*) kscale16.v[which ? (sb >> 4) : (sb & 15)]);
}

// the IQ3 phased-with-fold shape (mode 7) as a template over the two changes
template <int TY, bool PK, bool SP>
inline void gu_rows_phased32(const uint8_t* blob, size_t gu_row, size_t up_off, int n, const void* const* act,
                             float* const* ff, int r0, int r1) {
    const block_q8_K* y0 = (const block_q8_K*) act[0];
    const int nb = n / QK_K;
    for (int r = r0; r < r1; ++r) {
        const uint8_t* gbase = blob + (size_t) r * gu_row;
        const uint8_t* ubase = blob + up_off + (size_t) r * gu_row;
        __m128 fg = _mm_setzero_ps(), fu = _mm_setzero_ps();
        for (int i = 0; i < nb; ++i) {
            const uint8_t* gb = gbase + (size_t) i * Fmt128<TY>::bytes;
            const uint8_t* ub = ubase + (size_t) i * Fmt128<TY>::bytes;
            const uint8_t* q8 = (const uint8_t*) y0[i].qs;
            __m128i ag = _mm_setzero_si128(), au = _mm_setzero_si128();
            __m128i agb = _mm_setzero_si128(), aub = _mm_setzero_si128();
            for (int j = 0; j < 4; ++j) {
                for (int half = 0; half < 2; ++half) {
                    const int off = 64 * j + 32 * half;
                    __m128i pg = _mm_setzero_si128(), pu = _mm_setzero_si128(), g, sg_, c, ct;
                    if (PK && TY == 18) iq3xxs_dec16_pk(gb, j, half, 0, g, sg_, c); else iq3x_dec16(TY, gb, j, half, 0, g, sg_, c);
                    ct = c;
                    pg = _mm_maddubs_epi16(g, _mm_sign_epi8(_mm_loadu_si128((const __m128i*) (q8 + off)), sg_));
                    iq3x_dec16(TY, ub, j, half, 0, g, sg_, c);
                    pu = _mm_maddubs_epi16(g, _mm_sign_epi8(_mm_loadu_si128((const __m128i*) (q8 + off)), sg_));
                    if (PK && TY == 18) iq3xxs_dec16_pk(gb, j, half, 1, g, sg_, c); else iq3x_dec16(TY, gb, j, half, 1, g, sg_, c);
                    pg = _mm_add_epi16(pg, _mm_maddubs_epi16(g, _mm_sign_epi8(_mm_loadu_si128((const __m128i*) (q8 + off + 16)), sg_)));
                    iq3x_dec16(TY, ub, j, half, 1, g, sg_, c);
                    pu = _mm_add_epi16(pu, _mm_maddubs_epi16(g, _mm_sign_epi8(_mm_loadu_si128((const __m128i*) (q8 + off + 16)), sg_)));
                    if (SP && half) { agb = _mm_add_epi32(agb, _mm_madd_epi16(pg, ct)); aub = _mm_add_epi32(aub, _mm_madd_epi16(pu, c)); }
                    else { ag = _mm_add_epi32(ag, _mm_madd_epi16(pg, ct)); au = _mm_add_epi32(au, _mm_madd_epi16(pu, c)); }
                }
            }
            if (SP) { ag = _mm_add_epi32(ag, agb); au = _mm_add_epi32(au, aub); }
            const float dy = y0[i].d;
            fg = _mm_add_ps(fg, _mm_mul_ps(_mm_set1_ps(h2f(u16(gb)) * Fmt128<TY>::K * dy), _mm_cvtepi32_ps(ag)));
            fu = _mm_add_ps(fu, _mm_mul_ps(_mm_set1_ps(h2f(u16(ub)) * Fmt128<TY>::K * dy), _mm_cvtepi32_ps(au)));
        }
        float hg = hsum4(fg), hu = hsum4(fu);
        ff[0][r] = (hg / (1.f + std::exp(-hg))) * hu;
    }
}

// the IQ2_S phased shape (mode 4) over the same two changes
template <int TY, bool PK, bool SP>
inline void gu_rows_phased33(const uint8_t* blob, size_t gu_row, size_t up_off, int n, const void* const* act,
                             float* const* ff, int r0, int r1) {
    const block_q8_K* y0 = (const block_q8_K*) act[0];
    const int nb = n / QK_K;
    for (int r = r0; r < r1; ++r) {
        const uint8_t* gbase = blob + (size_t) r * gu_row;
        const uint8_t* ubase = blob + up_off + (size_t) r * gu_row;
        __m128 fg = _mm_setzero_ps(), fu = _mm_setzero_ps();
        for (int i = 0; i < nb; ++i) {
            const uint8_t* gb = gbase + (size_t) i * Fmt128<TY>::bytes;
            const uint8_t* ub = ubase + (size_t) i * Fmt128<TY>::bytes;
            const uint8_t* q8 = (const uint8_t*) y0[i].qs;
            __m128i ag0 = _mm_setzero_si128(), ag1 = _mm_setzero_si128();
            __m128i ag0b = _mm_setzero_si128(), ag1b = _mm_setzero_si128();
            __m128i au0 = _mm_setzero_si128(), au1 = _mm_setzero_si128();
            __m128i au0b = _mm_setzero_si128(), au1b = _mm_setzero_si128();
            for (int j = 0; j < 4; ++j) {
                for (int half = 0; half < 2; ++half) {
                    const int off = 64 * j + 32 * half;
                    const __m128i ig = grid_idx(gb + 2 + 8 * j + 4 * half, hi_spread.iq2s[gb[66 + 2 * j + half]]);
                    const __m128i iu = grid_idx(ub + 2 + 8 * j + 4 * half, hi_spread.iq2s[ub[66 + 2 * j + half]]);
                    const uint8_t sbg = gb[74 + 2 * j + half], sbu = ub[74 + 2 * j + half];
                    const __m128i yv0 = _mm_loadu_si128((const __m128i*) (q8 + off));
                    const __m128i yv1 = _mm_loadu_si128((const __m128i*) (q8 + off + 16));
                    __m128i g, s, c;
                    if (PK) {
                        const uint32_t swg = u32(gb + 2 + 32 + 8 * j + 4 * half), swu = u32(ub + 2 + 32 + 8 * j + 4 * half);
                        iq2s_dec16_pk(ig, swg, sbg, 0, g, s, c); ag0 = madd_add(ag0, g, s, yv0, c);
                        iq2s_dec16_pk(iu, swu, sbu, 0, g, s, c); au0 = madd_add(au0, g, s, yv0, c);
                        iq2s_dec16_pk(ig, swg, sbg, 1, g, s, c); ag1 = madd_add(ag1, g, s, yv1, c);
                        iq2s_dec16_pk(iu, swu, sbu, 1, g, s, c); au1 = madd_add(au1, g, s, yv1, c);
                    } else {
                        const uint8_t* spg = gb + 2 + 32 + 8 * j + 4 * half;
                        const uint8_t* spu = ub + 2 + 32 + 8 * j + 4 * half;
                        iq2s_dec16(ig, spg, sbg, 0, g, s, c); ag0 = madd_add(ag0, g, s, yv0, c);
                        iq2s_dec16(iu, spu, sbu, 0, g, s, c); au0 = madd_add(au0, g, s, yv0, c);
                        iq2s_dec16(ig, spg, sbg, 1, g, s, c); ag1 = madd_add(ag1, g, s, yv1, c);
                        iq2s_dec16(iu, spu, sbu, 1, g, s, c); au1 = madd_add(au1, g, s, yv1, c);
                    }
                }
            }
            // the 16-value phases already split the chain in two; SP splits each of them once more (by half of
            // the 32 values), which costs two live registers per row - the register cliff says that is the risk.
            if (SP) { ag0 = _mm_add_epi32(ag0, ag0b); ag1 = _mm_add_epi32(ag1, ag1b);
                      au0 = _mm_add_epi32(au0, au0b); au1 = _mm_add_epi32(au1, au1b); }
            const float dy = y0[i].d;
            fg = _mm_add_ps(fg, _mm_mul_ps(_mm_set1_ps(h2f(u16(gb)) * Fmt128<TY>::K * dy),
                                           _mm_cvtepi32_ps(_mm_add_epi32(ag0, ag1))));
            fu = _mm_add_ps(fu, _mm_mul_ps(_mm_set1_ps(h2f(u16(ub)) * Fmt128<TY>::K * dy),
                                           _mm_cvtepi32_ps(_mm_add_epi32(au0, au1))));
        }
        float hg = hsum4(fg), hu = hsum4(fu);
        ff[0][r] = (hg / (1.f + std::exp(-hg))) * hu;
    }
}


// MODE 16 (measurement): IQ3_S phased-with-fold with the grid index build issued one sub-half early.
// Mode 13 (IQ2_S) showed the loop waits on index LATENCY, not on ports. The 9-bit IQ3_S index needs
// grid_idx() = load 8 blob bytes + unpack with the spread byte, and its result feeds 4 vpextrw + 4 table loads:
// ~7 cycles of latency sitting in front of the multiplies. Same shape as mode 7, only the index build for the
// next sub-half is issued right after the current sub-half's multiplies, so the chain overlaps. Values, products
// and accumulation order are unchanged -> bit-identical by construction. IQ3_XXS is left alone: its indices are
// plain byte loads and pre-issuing them was already measured dead (mode 10).
static inline void iq3s_dec16_idx(const uint8_t* b, int j, int half, int which, __m128i idx,
                                  __m128i& g, __m128i& s, __m128i& c) {
    const int m = 4 * which;
#define GS(o) (int) iq3s_grid[_mm_extract_epi16(idx, m + (o))]
    g = _mm_set_epi32(GS(3), GS(2), GS(1), GS(0));
#undef GS
    s = sgn16(u32(b + 74 + 8 * j + 4 * half), which);
    const uint8_t sc = b[106 + j];
    c = _mm_load_si128((const __m128i*) kscale16.v[half ? sc >> 4 : sc & 15]);
}

template <int TY>
inline void gu_rows_iq3s_pipe(const uint8_t* blob, size_t gu_row, size_t up_off, int n, const void* const* act,
                              float* const* ff, int r0, int r1) {
    const block_q8_K* y0 = (const block_q8_K*) act[0];
    const int nb = n / QK_K;
    for (int r = r0; r < r1; ++r) {
        const uint8_t* gbase = blob + (size_t) r * gu_row;
        const uint8_t* ubase = blob + up_off + (size_t) r * gu_row;
        __m128 fg = _mm_setzero_ps(), fu = _mm_setzero_ps();
        for (int i = 0; i < nb; ++i) {
            const uint8_t* gb = gbase + (size_t) i * Fmt128<TY>::bytes;
            const uint8_t* ub = ubase + (size_t) i * Fmt128<TY>::bytes;
            const uint8_t* q8 = (const uint8_t*) y0[i].qs;
            __m128i ag = _mm_setzero_si128(), au = _mm_setzero_si128();
            __m128i ig = grid_idx(gb + 2, hi_spread.iq3s[gb[66]]);
            __m128i iu = grid_idx(ub + 2, hi_spread.iq3s[ub[66]]);
            for (int j = 0; j < 4; ++j) {
                for (int half = 0; half < 2; ++half) {
                    const int off = 64 * j + 32 * half;
                    __m128i pg = _mm_setzero_si128(), pu = _mm_setzero_si128(), g, sg_, c, ct;
                    iq3s_dec16_idx(gb, j, half, 0, ig, g, sg_, c); ct = c;
                    pg = _mm_maddubs_epi16(g, _mm_sign_epi8(_mm_loadu_si128((const __m128i*) (q8 + off)), sg_));
                    iq3s_dec16_idx(ub, j, half, 0, iu, g, sg_, c);
                    pu = _mm_maddubs_epi16(g, _mm_sign_epi8(_mm_loadu_si128((const __m128i*) (q8 + off)), sg_));
                    iq3s_dec16_idx(gb, j, half, 1, ig, g, sg_, c);
                    pg = _mm_add_epi16(pg, _mm_maddubs_epi16(g, _mm_sign_epi8(_mm_loadu_si128((const __m128i*) (q8 + off + 16)), sg_)));
                    iq3s_dec16_idx(ub, j, half, 1, iu, g, sg_, c);
                    pu = _mm_add_epi16(pu, _mm_maddubs_epi16(g, _mm_sign_epi8(_mm_loadu_si128((const __m128i*) (q8 + off + 16)), sg_)));
                    ag = _mm_add_epi32(ag, _mm_madd_epi16(pg, ct));
                    au = _mm_add_epi32(au, _mm_madd_epi16(pu, c));
                    const int j2 = j + half, h2 = half ^ 1;   // the next sub-half's indices, now
                    if (j != 3 || half != 1) {
                        ig = grid_idx(gb + 2 + 16 * j2 + 8 * h2, hi_spread.iq3s[gb[66 + 2 * j2 + h2]]);
                        iu = grid_idx(ub + 2 + 16 * j2 + 8 * h2, hi_spread.iq3s[ub[66 + 2 * j2 + h2]]);
                    }
                }
            }
            const float dy = y0[i].d;
            fg = _mm_add_ps(fg, _mm_mul_ps(_mm_set1_ps(h2f(u16(gb)) * Fmt128<TY>::K * dy), _mm_cvtepi32_ps(ag)));
            fu = _mm_add_ps(fu, _mm_mul_ps(_mm_set1_ps(h2f(u16(ub)) * Fmt128<TY>::K * dy), _mm_cvtepi32_ps(au)));
        }
        float hg = hsum4(fg), hu = hsum4(fu);
        ff[0][r] = (hg / (1.f + std::exp(-hg))) * hu;
    }
}


// MODE 16 for IQ3_XXS (17 of 48 layers - the widest reach left).  Same pre-issue, adapted to a format that has no
// index ASSEMBLY step: IQ3_XXS reads four plain grid-index bytes plus one sign+scale word per 16 values, so what can
// move early is the LOADS - the two 32-bit index words and the sign/scale word of the next sub-half are read a
// sub-half before they are used, while the iq3xxs_grid lookups stay at the point of use.  That is a different
// transformation from mode 10 (which changed HOW the bytes load - one 32-bit load - dead) and from mode 17 (which
// moved the whole codebook lookup across the boundary - 0.944 clean).  Same table entries in the same order, so the
// rows are bit-identical by construction; the parity tool memcmp-checks them anyway.
// MEASURED DEAD, NOT DISPATCHED (2026-10-19, 4 trials x min-of-5 alternating, engine live, both IQ3_XXS rungs):
// -7.55% pooled (0/8 positive, 90% CI [-9.4, -5.7]); the run's control (mode 17) landed on its clean margin, so the
// ratio is sound.  The mechanism does not transfer: IQ2_S and IQ3_S win their +2.5% / +3.9% by hoisting the
// grid_idx() SHUFFLE CHAIN that assembles packed indices, and IQ3_XXS has no such chain - its indices are plain
// adjacent bytes, so hoisting them only buys live registers and a guarded branch.  Kept as a diagnostic mode so the
// next base change can re-read the number in 20 seconds instead of re-implementing it.
static inline void iq3xxs_dec16_pre(uint32_t q0, uint32_t q1, uint32_t w, int which,
                                    __m128i& g, __m128i& s, __m128i& c) {
    const uint32_t q = which ? q1 : q0;
#define GXP(o) (int) iq3xxs_grid[(q >> (8 * (o))) & 0xFF]
    g = _mm_set_epi32(GXP(3), GXP(2), GXP(1), GXP(0));
#undef GXP
    const int a = which ? ((w >> 14) & 127) : (w & 127), e2 = which ? ((w >> 21) & 127) : ((w >> 7) & 127);
    s = _mm_set_epi64x((long long) even_signs.v[e2], (long long) even_signs.v[a]);
    c = _mm_load_si128((const __m128i*) kscale16.v[w >> 28]);
}

template <int TY>
inline void gu_rows_iq3xxs_pipe(const uint8_t* blob, size_t gu_row, size_t up_off, int n, const void* const* act,
                                float* const* ff, int r0, int r1) {
    const block_q8_K* y0 = (const block_q8_K*) act[0];
    const int nb = n / QK_K;
    for (int r = r0; r < r1; ++r) {
        const uint8_t* gbase = blob + (size_t) r * gu_row;
        const uint8_t* ubase = blob + up_off + (size_t) r * gu_row;
        __m128 fg = _mm_setzero_ps(), fu = _mm_setzero_ps();
        for (int i = 0; i < nb; ++i) {
            const uint8_t* gb = gbase + (size_t) i * Fmt128<TY>::bytes;
            const uint8_t* ub = ubase + (size_t) i * Fmt128<TY>::bytes;
            const uint8_t* q8 = (const uint8_t*) y0[i].qs;
            __m128i ag = _mm_setzero_si128(), au = _mm_setzero_si128();
            uint32_t qg0 = u32(gb + 2), qg1 = u32(gb + 6), wg = u32(gb + 66);
            uint32_t qu0 = u32(ub + 2), qu1 = u32(ub + 6), wu = u32(ub + 66);
            for (int j = 0; j < 4; ++j) {
                for (int half = 0; half < 2; ++half) {
                    const int off = 64 * j + 32 * half;
                    __m128i pg = _mm_setzero_si128(), pu = _mm_setzero_si128(), g, sg_, c, ct;
                    iq3xxs_dec16_pre(qg0, qg1, wg, 0, g, sg_, c); ct = c;
                    pg = _mm_maddubs_epi16(g, _mm_sign_epi8(_mm_loadu_si128((const __m128i*) (q8 + off)), sg_));
                    iq3xxs_dec16_pre(qu0, qu1, wu, 0, g, sg_, c);
                    pu = _mm_maddubs_epi16(g, _mm_sign_epi8(_mm_loadu_si128((const __m128i*) (q8 + off)), sg_));
                    iq3xxs_dec16_pre(qg0, qg1, wg, 1, g, sg_, c);
                    pg = _mm_add_epi16(pg, _mm_maddubs_epi16(g, _mm_sign_epi8(_mm_loadu_si128((const __m128i*) (q8 + off + 16)), sg_)));
                    iq3xxs_dec16_pre(qu0, qu1, wu, 1, g, sg_, c);
                    pu = _mm_add_epi16(pu, _mm_maddubs_epi16(g, _mm_sign_epi8(_mm_loadu_si128((const __m128i*) (q8 + off + 16)), sg_)));
                    ag = _mm_add_epi32(ag, _mm_madd_epi16(pg, ct));
                    au = _mm_add_epi32(au, _mm_madd_epi16(pu, c));
                    const int j2 = j + half, h2 = half ^ 1;   // the next sub-half's index bytes, now
                    if (j != 3 || half != 1) {
                        qg0 = u32(gb + 2 + 16 * j2 + 8 * h2); qg1 = u32(gb + 6 + 16 * j2 + 8 * h2);
                        wg  = u32(gb + 66 + 8 * j2 + 4 * h2);
                        qu0 = u32(ub + 2 + 16 * j2 + 8 * h2); qu1 = u32(ub + 6 + 16 * j2 + 8 * h2);
                        wu  = u32(ub + 66 + 8 * j2 + 4 * h2);
                    }
                }
            }
            const float dy = y0[i].d;
            fg = _mm_add_ps(fg, _mm_mul_ps(_mm_set1_ps(h2f(u16(gb)) * Fmt128<TY>::K * dy), _mm_cvtepi32_ps(ag)));
            fu = _mm_add_ps(fu, _mm_mul_ps(_mm_set1_ps(h2f(u16(ub)) * Fmt128<TY>::K * dy), _mm_cvtepi32_ps(au)));
        }
        float hg = hsum4(fg), hu = hsum4(fu);
        ff[0][r] = (hg / (1.f + std::exp(-hg))) * hu;
    }
}

// MODES 17 / 18 / 19 (round 10, report-10 proposals 1-3).  MEASURED DEAD, NOT USED (clean, service stopped,
// n=12 each, all three bit-exact): 17 = 0.944 (0/12), 19 = 0.925 (0/12), 18 = 1.003 (7/12) - and 18 is measured
// against the same pre-phasing baseline as the shipped mode 13, which reads 1.040 (12/12) in the same run, so the
// codebook pre-decode is ~3.6% slower than what production already runs.  The predicted +3.5-6.0% never appeared:
// the 5-cycle codebook L1 load is NOT the unhidden latency in front of vpmaddubsw.  Do not re-propose moving loads
// earlier; the remaining cycles live in the vector assembly and the pmaddubsw -> pmaddwd -> int32 add chain.  Round 9 showed the loop waits on the latency chain in
// front of vpmaddubsw, and pre-issuing the index BUILD paid +2.5% on IQ2_S.  These go one stage further: they
// pre-decode the CODEBOOK loads (the 5-cycle L1 hit on iq3xxs_grid / iq2s_grid / iq3s_grid) across the sub-block
// boundary, so the grid vectors for the next phase are already in registers when its multiply is issued.  Each is
// the shipped shape with the decode moved earlier - same values, same products, same accumulator order, so the
// bits cannot move.  The register cost is 2 extra live XMM per row carried across the boundary.
template <int TY>
void gu_rows_iq3xxs_gpipe(const uint8_t* blob, size_t gu_row, size_t up_off, int n, const void* const* act,
                          float* const* ff, int r0, int r1) {
    const block_q8_K* y0 = (const block_q8_K*) act[0];
    const int nb = n / QK_K;
    const int pf = prefetch_distance();
    for (int r = r0; r < r1; ++r) {
        const uint8_t* gbase = blob + (size_t) r * gu_row;
        const uint8_t* ubase = blob + up_off + (size_t) r * gu_row;
        __m128 fg = _mm_setzero_ps(), fu = _mm_setzero_ps();
        for (int i = 0; i < nb; ++i) {
            const uint8_t* gb = gbase + (size_t) i * Fmt128<TY>::bytes;
            const uint8_t* ub = ubase + (size_t) i * Fmt128<TY>::bytes;
            rows_ahead(gb, pf); rows_ahead(ub, pf);
            const uint8_t* q8 = (const uint8_t*) y0[i].qs;
            __m128i ag = _mm_setzero_si128(), au = _mm_setzero_si128();
#define GSET4(q) _mm_set_epi32((int) iq3xxs_grid[(q)[3]], (int) iq3xxs_grid[(q)[2]], \
                               (int) iq3xxs_grid[(q)[1]], (int) iq3xxs_grid[(q)[0]])
            __m128i gg0 = GSET4(gb + 2), gu0 = GSET4(ub + 2);
            for (int h = 0; h < 8; ++h) {
                const int off = 32 * h;
                const uint32_t wg = u32(gb + 66 + 4 * h), wu = u32(ub + 66 + 4 * h);
                const __m128i cg = _mm_load_si128((const __m128i*) kscale16.v[wg >> 28]);
                const __m128i cu = _mm_load_si128((const __m128i*) kscale16.v[wu >> 28]);
                const __m128i sg0 = _mm_set_epi64x((long long) even_signs.v[(wg >> 7) & 127],
                                                   (long long) even_signs.v[wg & 127]);
                const __m128i su0 = _mm_set_epi64x((long long) even_signs.v[(wu >> 7) & 127],
                                                   (long long) even_signs.v[wu & 127]);
                const __m128i yv0 = _mm_loadu_si128((const __m128i*) (q8 + off));
                __m128i pg = _mm_maddubs_epi16(gg0, _mm_sign_epi8(yv0, sg0));
                __m128i pu = _mm_maddubs_epi16(gu0, _mm_sign_epi8(yv0, su0));
                const __m128i gg1 = GSET4(gb + 6 + 8 * h), gu1 = GSET4(ub + 6 + 8 * h);
                const __m128i sg1 = _mm_set_epi64x((long long) even_signs.v[(wg >> 21) & 127],
                                                   (long long) even_signs.v[(wg >> 14) & 127]);
                const __m128i su1 = _mm_set_epi64x((long long) even_signs.v[(wu >> 21) & 127],
                                                   (long long) even_signs.v[(wu >> 14) & 127]);
                const __m128i yv1 = _mm_loadu_si128((const __m128i*) (q8 + off + 16));
                pg = _mm_add_epi16(pg, _mm_maddubs_epi16(gg1, _mm_sign_epi8(yv1, sg1)));
                pu = _mm_add_epi16(pu, _mm_maddubs_epi16(gu1, _mm_sign_epi8(yv1, su1)));
                ag = _mm_add_epi32(ag, _mm_madd_epi16(pg, cg));
                au = _mm_add_epi32(au, _mm_madd_epi16(pu, cu));
                if (h + 1 < 8) { gg0 = GSET4(gb + 2 + 8 * (h + 1)); gu0 = GSET4(ub + 2 + 8 * (h + 1)); }
            }
#undef GSET4
            const float dy = y0[i].d;
            fg = _mm_add_ps(fg, _mm_mul_ps(_mm_set1_ps(h2f(u16(gb)) * Fmt128<TY>::K * dy), _mm_cvtepi32_ps(ag)));
            fu = _mm_add_ps(fu, _mm_mul_ps(_mm_set1_ps(h2f(u16(ub)) * Fmt128<TY>::K * dy), _mm_cvtepi32_ps(au)));
        }
        float hg = hsum4(fg), hu = hsum4(fu);
        ff[0][r] = (hg / (1.f + std::exp(-hg))) * hu;
    }
}

template <int TY>
void gu_rows_iq2s_gpipe(const uint8_t* blob, size_t gu_row, size_t up_off, int n, const void* const* act,
                        float* const* ff, int r0, int r1) {
    const block_q8_K* y0 = (const block_q8_K*) act[0];
    const int nb = n / QK_K;
    const int pf = prefetch_distance();
    for (int r = r0; r < r1; ++r) {
        const uint8_t* gbase = blob + (size_t) r * gu_row;
        const uint8_t* ubase = blob + up_off + (size_t) r * gu_row;
        __m128 fg = _mm_setzero_ps(), fu = _mm_setzero_ps();
        for (int i = 0; i < nb; ++i) {
            const uint8_t* gb = gbase + (size_t) i * Fmt128<TY>::bytes;
            const uint8_t* ub = ubase + (size_t) i * Fmt128<TY>::bytes;
            rows_ahead(gb, pf); rows_ahead(ub, pf);
            const uint8_t* q8 = (const uint8_t*) y0[i].qs;
            __m128i ag0 = _mm_setzero_si128(), ag1 = _mm_setzero_si128();
            __m128i au0 = _mm_setzero_si128(), au1 = _mm_setzero_si128();
#define GSET2(v, k) _mm_set_epi64x((long long) iq2s_grid[_mm_extract_epi16(v, k + 1)], \
                                   (long long) iq2s_grid[_mm_extract_epi16(v, k)])
            __m128i ig = grid_idx(gb + 2, hi_spread.iq2s[gb[66]]);
            __m128i iu = grid_idx(ub + 2, hi_spread.iq2s[ub[66]]);
            __m128i gg0 = GSET2(ig, 0), gu0 = GSET2(iu, 0);
            for (int h = 0; h < 8; ++h) {
                const int off = 32 * h;
                const uint8_t* spg = gb + 34 + 4 * h;
                const uint8_t* spu = ub + 34 + 4 * h;
                const uint8_t sbg = gb[74 + h], sbu = ub[74 + h];
                const __m128i yv0 = _mm_loadu_si128((const __m128i*) (q8 + off));
                const __m128i sg0 = _mm_set_epi64x((long long) sign8.v[spg[1]], (long long) sign8.v[spg[0]]);
                const __m128i su0 = _mm_set_epi64x((long long) sign8.v[spu[1]], (long long) sign8.v[spu[0]]);
                ag0 = madd_add(ag0, gg0, sg0, yv0, _mm_load_si128((const __m128i*) kscale16.v[sbg & 15]));
                au0 = madd_add(au0, gu0, su0, yv0, _mm_load_si128((const __m128i*) kscale16.v[sbu & 15]));
                const __m128i gg1 = GSET2(ig, 2), gu1 = GSET2(iu, 2);
                const __m128i yv1 = _mm_loadu_si128((const __m128i*) (q8 + off + 16));
                const __m128i sg1 = _mm_set_epi64x((long long) sign8.v[spg[3]], (long long) sign8.v[spg[2]]);
                const __m128i su1 = _mm_set_epi64x((long long) sign8.v[spu[3]], (long long) sign8.v[spu[2]]);
                ag1 = madd_add(ag1, gg1, sg1, yv1, _mm_load_si128((const __m128i*) kscale16.v[sbg >> 4]));
                au1 = madd_add(au1, gu1, su1, yv1, _mm_load_si128((const __m128i*) kscale16.v[sbu >> 4]));
                if (h + 1 < 8) {
                    ig = grid_idx(gb + 2 + 4 * (h + 1), hi_spread.iq2s[gb[66 + h + 1]]);
                    iu = grid_idx(ub + 2 + 4 * (h + 1), hi_spread.iq2s[ub[66 + h + 1]]);
                    gg0 = GSET2(ig, 0); gu0 = GSET2(iu, 0);
                }
            }
#undef GSET2
            const float dy = y0[i].d;
            fg = _mm_add_ps(fg, _mm_mul_ps(_mm_set1_ps(h2f(u16(gb)) * Fmt128<TY>::K * dy),
                                           _mm_cvtepi32_ps(_mm_add_epi32(ag0, ag1))));
            fu = _mm_add_ps(fu, _mm_mul_ps(_mm_set1_ps(h2f(u16(ub)) * Fmt128<TY>::K * dy),
                                           _mm_cvtepi32_ps(_mm_add_epi32(au0, au1))));
        }
        float hg = hsum4(fg), hu = hsum4(fu);
        ff[0][r] = (hg / (1.f + std::exp(-hg))) * hu;
    }
}

template <int TY>
void gu_rows_iq3s_dpipe(const uint8_t* blob, size_t gu_row, size_t up_off, int n, const void* const* act,
                        float* const* ff, int r0, int r1) {
    const block_q8_K* y0 = (const block_q8_K*) act[0];
    const int nb = n / QK_K;
    const int pf = prefetch_distance();
    for (int r = r0; r < r1; ++r) {
        const uint8_t* gbase = blob + (size_t) r * gu_row;
        const uint8_t* ubase = blob + up_off + (size_t) r * gu_row;
        __m128 fg = _mm_setzero_ps(), fu = _mm_setzero_ps();
        for (int i = 0; i < nb; ++i) {
            const uint8_t* gb = gbase + (size_t) i * Fmt128<TY>::bytes;
            const uint8_t* ub = ubase + (size_t) i * Fmt128<TY>::bytes;
            rows_ahead(gb, pf); rows_ahead(ub, pf);
            const uint8_t* q8 = (const uint8_t*) y0[i].qs;
            __m128i ag = _mm_setzero_si128(), au = _mm_setzero_si128();
#define GSETW(v, m) _mm_set_epi32((int) iq3s_grid[_mm_extract_epi16(v, (m) + 3)], \
                                  (int) iq3s_grid[_mm_extract_epi16(v, (m) + 2)], \
                                  (int) iq3s_grid[_mm_extract_epi16(v, (m) + 1)], \
                                  (int) iq3s_grid[_mm_extract_epi16(v, (m) + 0)])
            __m128i ig = grid_idx(gb + 2, hi_spread.iq3s[gb[66]]);
            __m128i iu = grid_idx(ub + 2, hi_spread.iq3s[ub[66]]);
            for (int h = 0; h < 8; ++h) {
                const int j = h >> 1, half = h & 1, off = 32 * h;
                const uint32_t sgs = u32(gb + 74 + 4 * h), sgsu = u32(ub + 74 + 4 * h);
                const uint8_t scg = gb[106 + j], scu = ub[106 + j];
                const __m128i cg = _mm_load_si128((const __m128i*) kscale16.v[half ? scg >> 4 : scg & 15]);
                const __m128i cu = _mm_load_si128((const __m128i*) kscale16.v[half ? scu >> 4 : scu & 15]);
                const __m128i yv0 = _mm_loadu_si128((const __m128i*) (q8 + off));
                __m128i pg = _mm_maddubs_epi16(GSETW(ig, 0), _mm_sign_epi8(yv0, sgn16(sgs, 0)));
                __m128i pu = _mm_maddubs_epi16(GSETW(iu, 0), _mm_sign_epi8(yv0, sgn16(sgsu, 0)));
                const __m128i yv1 = _mm_loadu_si128((const __m128i*) (q8 + off + 16));
                pg = _mm_add_epi16(pg, _mm_maddubs_epi16(GSETW(ig, 4), _mm_sign_epi8(yv1, sgn16(sgs, 1))));
                pu = _mm_add_epi16(pu, _mm_maddubs_epi16(GSETW(iu, 4), _mm_sign_epi8(yv1, sgn16(sgsu, 1))));
                ag = _mm_add_epi32(ag, _mm_madd_epi16(pg, cg));
                au = _mm_add_epi32(au, _mm_madd_epi16(pu, cu));
                if (h + 1 < 8) {
                    const int j2 = (h + 1) >> 1, h2 = (h + 1) & 1;
                    ig = grid_idx(gb + 2 + 16 * j2 + 8 * h2, hi_spread.iq3s[gb[66 + h + 1]]);
                    iu = grid_idx(ub + 2 + 16 * j2 + 8 * h2, hi_spread.iq3s[ub[66 + h + 1]]);
                }
            }
#undef GSETW
            const float dy = y0[i].d;
            fg = _mm_add_ps(fg, _mm_mul_ps(_mm_set1_ps(h2f(u16(gb)) * Fmt128<TY>::K * dy), _mm_cvtepi32_ps(ag)));
            fu = _mm_add_ps(fu, _mm_mul_ps(_mm_set1_ps(h2f(u16(ub)) * Fmt128<TY>::K * dy), _mm_cvtepi32_ps(au)));
        }
        float hg = hsum4(fg), hu = hsum4(fu);
        ff[0][r] = (hg / (1.f + std::exp(-hg))) * hu;
    }
}


// MODES 13 / 14 / 15 (measurement, round 9).  The report's port census says the phased loop carries 16 port-0
// uops per 32-value unit (8 vpmaddubsw + 8 vpmaddwd + the 8 vpextrw that move grid indices out of XMM), and that
// on Ivy vpextrw is port 0 only, so 8 of those slots are "permanently reserved".  Two of these proposals change
// WHEN work issues, one changes WHERE it issues:
//   13  software-pipelined index pre-issue (report-9 proposal 1): sub-block h+1's indices are built while
//       sub-block h's multiplies retire.  Same uops, same ports - it can only pay if the stall is index LATENCY,
//       not port-0 throughput, and note the census's own 16.0-cycle port-0 floor is already above the 14.1-14.5
//       it predicts for this variant, so one of the two is wrong.  Measure, do not argue.
//   14  deduplicated grid_idx (report-9 proposal 2): the shipped iq3x_dec16 recomputes grid_idx for which=0 and
//       which=1 from the same 8 bytes; this builds it once.  Only pays if GCC does not already CSE the pair -
//       the instruction census says 32 -> 16 vpunpcklbw, so verify in the disassembly, not in the source.
//   15  scalar index extraction (this repo's answer to the port-0 tax): grid_idx is only
//       unpacklo(q_bytes, hi_bytes) and the only use of the result is vpextrw + a table load.  Read the four
//       index bytes as one u32 into a GPR, OR in the spread byte's bits, and load the two 8-byte grid entries
//       with movq/pinsrq-from-memory: the 8 vpextrw per unit leave port 0 entirely (they become scalar shifts on
//       ports 0/1/2/3/6 and load uops on 2/3).  Same table entries, same products, same order -> bit-identical.
template <int TY>
inline void gu_rows_iq2s_pipelined(const uint8_t* blob, size_t gu_row, size_t up_off, int n, const void* const* act,
                                   float* const* ff, int r0, int r1) {
    const block_q8_K* y0 = (const block_q8_K*) act[0];
    const int nb = n / QK_K;
    const int pf = prefetch_distance();
    for (int r = r0; r < r1; ++r) {
        const uint8_t* gbase = blob + (size_t) r * gu_row;
        const uint8_t* ubase = blob + up_off + (size_t) r * gu_row;
        __m128 fg = _mm_setzero_ps(), fu = _mm_setzero_ps();
        for (int i = 0; i < nb; ++i) {
            const uint8_t* gb = gbase + (size_t) i * Fmt128<TY>::bytes;
            const uint8_t* ub = ubase + (size_t) i * Fmt128<TY>::bytes;
            rows_ahead(gb, pf); rows_ahead(ub, pf);
            const uint8_t* q8 = (const uint8_t*) y0[i].qs;
            __m128i ag0 = _mm_setzero_si128(), ag1 = _mm_setzero_si128();
            __m128i au0 = _mm_setzero_si128(), au1 = _mm_setzero_si128();
            __m128i ig = grid_idx(gb + 2, hi_spread.iq2s[gb[66]]);
            __m128i iu = grid_idx(ub + 2, hi_spread.iq2s[ub[66]]);
            for (int h = 0; h < 8; ++h) {
                const int off = 32 * h;
                const uint8_t* spg = gb + 34 + 4 * h;
                const uint8_t* spu = ub + 34 + 4 * h;
                const uint8_t sbg = gb[74 + h], sbu = ub[74 + h];
                const __m128i yv0 = _mm_loadu_si128((const __m128i*) (q8 + off));
                __m128i g, s, c;
                iq2s_dec16(ig, spg, sbg, 0, g, s, c); ag0 = madd_add(ag0, g, s, yv0, c);
                iq2s_dec16(iu, spu, sbu, 0, g, s, c); au0 = madd_add(au0, g, s, yv0, c);
                const __m128i yv1 = _mm_loadu_si128((const __m128i*) (q8 + off + 16));
                iq2s_dec16(ig, spg, sbg, 1, g, s, c); ag1 = madd_add(ag1, g, s, yv1, c);
                iq2s_dec16(iu, spu, sbu, 1, g, s, c); au1 = madd_add(au1, g, s, yv1, c);
                if (h + 1 < 8) {
                    ig = grid_idx(gb + 2 + 4 * (h + 1), hi_spread.iq2s[gb[66 + h + 1]]);
                    iu = grid_idx(ub + 2 + 4 * (h + 1), hi_spread.iq2s[ub[66 + h + 1]]);
                }
            }
            const float dy = y0[i].d;
            fg = _mm_add_ps(fg, _mm_mul_ps(_mm_set1_ps(h2f(u16(gb)) * Fmt128<TY>::K * dy),
                                           _mm_cvtepi32_ps(_mm_add_epi32(ag0, ag1))));
            fu = _mm_add_ps(fu, _mm_mul_ps(_mm_set1_ps(h2f(u16(ub)) * Fmt128<TY>::K * dy),
                                           _mm_cvtepi32_ps(_mm_add_epi32(au0, au1))));
        }
        float hg = hsum4(fg), hu = hsum4(fu);
        ff[0][r] = (hg / (1.f + std::exp(-hg))) * hu;
    }
}

// the four index bytes of one IQ2_S sub-block as scalars: index = byte | (spread_byte << 8), exactly what
// unpacklo + vpextrw produce, with no vector index build and no vpextrw at all.
static inline __m128i iq2s_gr2(const uint8_t* q, uint64_t hi, int which) {
    const uint32_t qb = u32(q);
    const int k = 2 * which;
    const long long a = (long long) iq2s_grid[(int) ((qb >> (8 * k)) & 0xFF) | (int) (((hi >> (8 * k)) & 0xFF) << 8)];
    const long long b = (long long) iq2s_grid[(int) ((qb >> (8 * (k + 1))) & 0xFF) | (int) (((hi >> (8 * (k + 1))) & 0xFF) << 8)];
    return _mm_set_epi64x(b, a);
}

// MODE 20 (measurement): the codebook replaced by its OWN ARITHMETIC.  New fact: every byte of iq2s_grid is an
// EXACT affine function of the 10 index bits - byte_k(i) = (K_k + sum_i c_ki * bit_i(i)) >> 9 for all 1024 entries,
// every byte position, zero error (verified at first use below; the shape falls back to the table if it ever fails).
// The same holds for iq3xxs_grid and iq3s_grid.  So the 8 KB grid table carries no information: it is arithmetic we
// pay a load for.  No 3-bit shortcut exists (every subset searched), so the arithmetic is carried as two 5-bit half
// tables (2 KB total, permanently L1-hot) plus add, exact >>9 and two packs per magnitude group.
// This is NOT fewer uops - it is more.  The question it asks is the one the stall census makes interesting: does
// taking the random 8 KB grid load off ports 2/3 (which have spare capacity) and the vpextrw/vpinsrq assembly off
// ports 0/5 pay for the extra ALU on ports 0/1, which are the saturated ones?  Census says no.  Measure it.
// MEASURED, contended box, n=6, bit-exact: 0.46-0.49x - the arithmetic form costs TWICE the time of the table form
// (1.08-1.20 ms/expert vs 0.49-0.54).  The reason is structural, not tuning: the partial sums need 15-bit lanes
// (coefficients reach 442, numerators 512*43) while pmaddubsw needs 8-bit lanes, so the decode must widen to int32,
// do 4 loads + 4 adds + 4 shifts, then narrow through two packus - and packus is port 5, srai is port 0, both the
// saturated ports.  The table gives the same bytes in one load plus a set_epi64x.  On this core the LOOKUP is not an
// approximation of the arithmetic, it is a 2.1x faster expression of it.  Do not retry the affine decode in any form.
struct Q2Affine {
    alignas(16) int32_t lo[2][32][4], hi[2][32][4];
    bool ok;
};
static const Q2Affine& q2aff() {
    static Q2Affine t{};
    static const bool init = [&] {
        int K[8], C[8][10];
        for (int k = 0; k < 8; ++k) {
            K[k] = 512 * (int) ((iq2s_grid[0] >> (8 * k)) & 0xFF);
            for (int i = 0; i < 10; ++i)
                C[k][i] = (int) ((iq2s_grid[1u << i] >> (8 * k)) & 0xFF) - (int) ((iq2s_grid[0] >> (8 * k)) & 0xFF);
        }
        t.ok = true;
        for (int idx = 0; idx < 1024 && t.ok; ++idx)
            for (int k = 0; k < 8; ++k) {
                int v = K[k];
                for (int i = 0; i < 10; ++i) if ((idx >> i) & 1) v += C[k][i];
                if (v != 512 * (int) ((iq2s_grid[idx] >> (8 * k)) & 0xFF)) { t.ok = false; break; }
            }
        for (int k = 0; k < 8; ++k)
            for (int h = 0; h < 32; ++h) {
                int a = K[k], b = 0;
                for (int i = 0; i < 5; ++i) if ((h >> i) & 1) a += C[k][i];
                for (int i = 5; i < 10; ++i) if ((h >> (i - 5)) & 1) b += C[k][i];
                t.lo[k / 4][h][k % 4] = a;
                t.hi[k / 4][h][k % 4] = b;
            }
        return true;
    }();
    (void) init;
    return t;
}

static inline void iq2s_dec16_aff(__m128i idx, const uint8_t* sp, uint8_t sb, int which,
                                  __m128i& g, __m128i& s, __m128i& c) {
    const Q2Affine& T = q2aff();
    if (!T.ok) { iq2s_dec16(idx, sp, sb, which, g, s, c); return; }
    const int i0 = _mm_extract_epi16(idx, 2 * which), i1 = _mm_extract_epi16(idx, 2 * which + 1);
    const __m128i a = _mm_srai_epi32(_mm_add_epi32(_mm_load_si128((const __m128i*) T.lo[0][i0 & 31]),
                                                   _mm_load_si128((const __m128i*) T.hi[0][i0 >> 5])), 9);
    const __m128i b = _mm_srai_epi32(_mm_add_epi32(_mm_load_si128((const __m128i*) T.lo[1][i0 & 31]),
                                                   _mm_load_si128((const __m128i*) T.hi[1][i0 >> 5])), 9);
    const __m128i d = _mm_srai_epi32(_mm_add_epi32(_mm_load_si128((const __m128i*) T.lo[0][i1 & 31]),
                                                   _mm_load_si128((const __m128i*) T.hi[0][i1 >> 5])), 9);
    const __m128i e = _mm_srai_epi32(_mm_add_epi32(_mm_load_si128((const __m128i*) T.lo[1][i1 & 31]),
                                                   _mm_load_si128((const __m128i*) T.hi[1][i1 >> 5])), 9);
    g = _mm_packus_epi16(_mm_packus_epi32(a, b), _mm_packus_epi32(d, e));   // mag 0..7 then 8..15, as set_epi64x gives
    s = _mm_set_epi64x((long long) sign8.v[sp[1]], (long long) sign8.v[sp[0]]);
    c = _mm_load_si128((const __m128i*) kscale16.v[which ? (sb >> 4) : (sb & 15)]);
}

// the shipped mode-13 IQ2_S shape with the grid lookup swapped for its affine arithmetic; all else identical
template <int TY>
inline void gu_rows_iq2s_aff(const uint8_t* blob, size_t gu_row, size_t up_off, int n, const void* const* act,
                             float* const* ff, int r0, int r1) {
    const block_q8_K* y0 = (const block_q8_K*) act[0];
    const int nb = n / QK_K;
    const int pf = prefetch_distance();
    for (int r = r0; r < r1; ++r) {
        const uint8_t* gbase = blob + (size_t) r * gu_row;
        const uint8_t* ubase = blob + up_off + (size_t) r * gu_row;
        __m128 fg = _mm_setzero_ps(), fu = _mm_setzero_ps();
        for (int i = 0; i < nb; ++i) {
            const uint8_t* gb = gbase + (size_t) i * Fmt128<TY>::bytes;
            const uint8_t* ub = ubase + (size_t) i * Fmt128<TY>::bytes;
            rows_ahead(gb, pf); rows_ahead(ub, pf);
            const uint8_t* q8 = (const uint8_t*) y0[i].qs;
            __m128i ag0 = _mm_setzero_si128(), ag1 = _mm_setzero_si128();
            __m128i au0 = _mm_setzero_si128(), au1 = _mm_setzero_si128();
            __m128i ig = grid_idx(gb + 2, hi_spread.iq2s[gb[66]]);
            __m128i iu = grid_idx(ub + 2, hi_spread.iq2s[ub[66]]);
            for (int h = 0; h < 8; ++h) {
                const int off = 32 * h;
                const uint8_t* spg = gb + 34 + 4 * h;
                const uint8_t* spu = ub + 34 + 4 * h;
                const uint8_t sbg = gb[74 + h], sbu = ub[74 + h];
                const __m128i yv0 = _mm_loadu_si128((const __m128i*) (q8 + off));
                __m128i g, s, c;
                iq2s_dec16_aff(ig, spg, sbg, 0, g, s, c); ag0 = madd_add(ag0, g, s, yv0, c);
                iq2s_dec16_aff(iu, spu, sbu, 0, g, s, c); au0 = madd_add(au0, g, s, yv0, c);
                const __m128i yv1 = _mm_loadu_si128((const __m128i*) (q8 + off + 16));
                iq2s_dec16_aff(ig, spg, sbg, 1, g, s, c); ag1 = madd_add(ag1, g, s, yv1, c);
                iq2s_dec16_aff(iu, spu, sbu, 1, g, s, c); au1 = madd_add(au1, g, s, yv1, c);
                if (h + 1 < 8) {
                    ig = grid_idx(gb + 2 + 4 * (h + 1), hi_spread.iq2s[gb[66 + h + 1]]);
                    iu = grid_idx(ub + 2 + 4 * (h + 1), hi_spread.iq2s[ub[66 + h + 1]]);
                }
            }
            const float dy = y0[i].d;
            fg = _mm_add_ps(fg, _mm_mul_ps(_mm_set1_ps(h2f(u16(gb)) * Fmt128<TY>::K * dy),
                                           _mm_cvtepi32_ps(_mm_add_epi32(ag0, ag1))));
            fu = _mm_add_ps(fu, _mm_mul_ps(_mm_set1_ps(h2f(u16(ub)) * Fmt128<TY>::K * dy),
                                           _mm_cvtepi32_ps(_mm_add_epi32(au0, au1))));
        }
        float hg = hsum4(fg), hu = hsum4(fu);
        ff[0][r] = (hg / (1.f + std::exp(-hg))) * hu;
    }
}

template <int TY>
inline void gu_rows_iq2s_scalar(const uint8_t* blob, size_t gu_row, size_t up_off, int n, const void* const* act,
                                float* const* ff, int r0, int r1) {
    const block_q8_K* y0 = (const block_q8_K*) act[0];
    const int nb = n / QK_K;
    const int pf = prefetch_distance();
    for (int r = r0; r < r1; ++r) {
        const uint8_t* gbase = blob + (size_t) r * gu_row;
        const uint8_t* ubase = blob + up_off + (size_t) r * gu_row;
        __m128 fg = _mm_setzero_ps(), fu = _mm_setzero_ps();
        for (int i = 0; i < nb; ++i) {
            const uint8_t* gb = gbase + (size_t) i * Fmt128<TY>::bytes;
            const uint8_t* ub = ubase + (size_t) i * Fmt128<TY>::bytes;
            rows_ahead(gb, pf); rows_ahead(ub, pf);
            const uint8_t* q8 = (const uint8_t*) y0[i].qs;
            __m128i ag0 = _mm_setzero_si128(), ag1 = _mm_setzero_si128();
            __m128i au0 = _mm_setzero_si128(), au1 = _mm_setzero_si128();
            for (int h = 0; h < 8; ++h) {
                const int off = 32 * h;
                const uint8_t* spg = gb + 34 + 4 * h;
                const uint8_t* spu = ub + 34 + 4 * h;
                const uint8_t sbg = gb[74 + h], sbu = ub[74 + h];
                const uint64_t hg = hi_spread.iq2s[gb[66 + h]], hu = hi_spread.iq2s[ub[66 + h]];
                const __m128i yv0 = _mm_loadu_si128((const __m128i*) (q8 + off));
                __m128i s, c;
                s = _mm_set_epi64x((long long) sign8.v[spg[1]], (long long) sign8.v[spg[0]]);
                c = _mm_load_si128((const __m128i*) kscale16.v[sbg & 15]);
                ag0 = madd_add(ag0, iq2s_gr2(gb + 2 + 4 * h, hg, 0), s, yv0, c);
                s = _mm_set_epi64x((long long) sign8.v[spu[1]], (long long) sign8.v[spu[0]]);
                c = _mm_load_si128((const __m128i*) kscale16.v[sbu & 15]);
                au0 = madd_add(au0, iq2s_gr2(ub + 2 + 4 * h, hu, 0), s, yv0, c);
                const __m128i yv1 = _mm_loadu_si128((const __m128i*) (q8 + off + 16));
                s = _mm_set_epi64x((long long) sign8.v[spg[3]], (long long) sign8.v[spg[2]]);
                c = _mm_load_si128((const __m128i*) kscale16.v[sbg >> 4]);
                ag1 = madd_add(ag1, iq2s_gr2(gb + 2 + 4 * h, hg, 1), s, yv1, c);
                s = _mm_set_epi64x((long long) sign8.v[spu[3]], (long long) sign8.v[spu[2]]);
                c = _mm_load_si128((const __m128i*) kscale16.v[sbu >> 4]);
                au1 = madd_add(au1, iq2s_gr2(ub + 2 + 4 * h, hu, 1), s, yv1, c);
            }
            const float dy = y0[i].d;
            fg = _mm_add_ps(fg, _mm_mul_ps(_mm_set1_ps(h2f(u16(gb)) * Fmt128<TY>::K * dy),
                                           _mm_cvtepi32_ps(_mm_add_epi32(ag0, ag1))));
            fu = _mm_add_ps(fu, _mm_mul_ps(_mm_set1_ps(h2f(u16(ub)) * Fmt128<TY>::K * dy),
                                           _mm_cvtepi32_ps(_mm_add_epi32(au0, au1))));
        }
        float hg2 = hsum4(fg), hu2 = hsum4(fu);
        ff[0][r] = (hg2 / (1.f + std::exp(-hg2))) * hu2;
    }
}

template <int TY>
inline void gu_rows_iq3s_dedup(const uint8_t* blob, size_t gu_row, size_t up_off, int n, const void* const* act,
                               float* const* ff, int r0, int r1) {
    const block_q8_K* y0 = (const block_q8_K*) act[0];
    const int nb = n / QK_K;
    const int pf = prefetch_distance();
    for (int r = r0; r < r1; ++r) {
        const uint8_t* gbase = blob + (size_t) r * gu_row;
        const uint8_t* ubase = blob + up_off + (size_t) r * gu_row;
        __m128 fg = _mm_setzero_ps(), fu = _mm_setzero_ps();
        for (int i = 0; i < nb; ++i) {
            const uint8_t* gb = gbase + (size_t) i * Fmt128<TY>::bytes;
            const uint8_t* ub = ubase + (size_t) i * Fmt128<TY>::bytes;
            rows_ahead(gb, pf); rows_ahead(ub, pf);
            const uint8_t* q8 = (const uint8_t*) y0[i].qs;
            __m128i ag = _mm_setzero_si128(), au = _mm_setzero_si128();
            for (int h = 0; h < 8; ++h) {
                const int j = h >> 1, half = h & 1, off = 32 * h;
                const __m128i ig = grid_idx(gb + 2 + 16 * j + 8 * half, hi_spread.iq3s[gb[66 + 2 * j + half]]);
                const __m128i iu = grid_idx(ub + 2 + 16 * j + 8 * half, hi_spread.iq3s[ub[66 + 2 * j + half]]);
                const uint32_t sgs = u32(gb + 74 + 8 * j + 4 * half), sgsu = u32(ub + 74 + 8 * j + 4 * half);
                const uint8_t scg = gb[106 + j], scu = ub[106 + j];
                const __m128i cg = _mm_load_si128((const __m128i*) kscale16.v[half ? scg >> 4 : scg & 15]);
                const __m128i cu = _mm_load_si128((const __m128i*) kscale16.v[half ? scu >> 4 : scu & 15]);
#define GSET(v, m) _mm_set_epi32((int) iq3s_grid[_mm_extract_epi16(v, m + 3)], (int) iq3s_grid[_mm_extract_epi16(v, m + 2)], \
                                 (int) iq3s_grid[_mm_extract_epi16(v, m + 1)], (int) iq3s_grid[_mm_extract_epi16(v, m + 0)])
                const __m128i yv0 = _mm_loadu_si128((const __m128i*) (q8 + off));
                __m128i pg = _mm_maddubs_epi16(GSET(ig, 0), _mm_sign_epi8(yv0, sgn16(sgs, 0)));
                __m128i pu = _mm_maddubs_epi16(GSET(iu, 0), _mm_sign_epi8(yv0, sgn16(sgsu, 0)));
                const __m128i yv1 = _mm_loadu_si128((const __m128i*) (q8 + off + 16));
                pg = _mm_add_epi16(pg, _mm_maddubs_epi16(GSET(ig, 4), _mm_sign_epi8(yv1, sgn16(sgs, 1))));
                pu = _mm_add_epi16(pu, _mm_maddubs_epi16(GSET(iu, 4), _mm_sign_epi8(yv1, sgn16(sgsu, 1))));
#undef GSET
                ag = _mm_add_epi32(ag, _mm_madd_epi16(pg, cg));
                au = _mm_add_epi32(au, _mm_madd_epi16(pu, cu));
            }
            const float dy = y0[i].d;
            fg = _mm_add_ps(fg, _mm_mul_ps(_mm_set1_ps(h2f(u16(gb)) * Fmt128<TY>::K * dy), _mm_cvtepi32_ps(ag)));
            fu = _mm_add_ps(fu, _mm_mul_ps(_mm_set1_ps(h2f(u16(ub)) * Fmt128<TY>::K * dy), _mm_cvtepi32_ps(au)));
        }
        float hg = hsum4(fg), hu = hsum4(fu);
        ff[0][r] = (hg / (1.f + std::exp(-hg))) * hu;
    }
}


// MODE 8 (measurement, report-8 proposal A): the phased IQ2_S shape with both rows' weights decoded and held live
// together, so the two rows' L1 table lookups issue in parallel, and the sub-scale left as a memory operand for
// vpmaddwd rather than a live register.  Census 4 live vectors per row.  Per-row arithmetic and its order unchanged.
static inline __m128i madd_add_c(__m128i acc, __m128i g, __m128i s, __m128i yv, const int8_t* c) {
    return _mm_add_epi32(acc, _mm_madd_epi16(_mm_maddubs_epi16(g, _mm_sign_epi8(yv, s)),
                                             _mm_load_si128((const __m128i*) c)));
}

template <int TY>
inline void gu_rows_dualstream(const uint8_t* blob, size_t gu_row, size_t up_off, int n, const void* const* act,
                               float* const* ff, int r0, int r1) {
    const block_q8_K* y0 = (const block_q8_K*) act[0];
    const int nb = n / QK_K;
    for (int r = r0; r < r1; ++r) {
        const uint8_t* gbase = blob + (size_t) r * gu_row;
        const uint8_t* ubase = blob + up_off + (size_t) r * gu_row;
        __m128 fg = _mm_setzero_ps(), fu = _mm_setzero_ps();
        for (int i = 0; i < nb; ++i) {
            const uint8_t* gb = gbase + (size_t) i * Fmt128<TY>::bytes;
            const uint8_t* ub = ubase + (size_t) i * Fmt128<TY>::bytes;
            const uint8_t* q8 = (const uint8_t*) y0[i].qs;
            __m128i ag0 = _mm_setzero_si128(), ag1 = _mm_setzero_si128();
            __m128i au0 = _mm_setzero_si128(), au1 = _mm_setzero_si128();
            for (int j = 0; j < 4; ++j) {
                for (int half = 0; half < 2; ++half) {
                    const int off = 64 * j + 32 * half;
                    const __m128i ig = grid_idx(gb + 2 + 8 * j + 4 * half, hi_spread.iq2s[gb[66 + 2 * j + half]]);
                    const __m128i iu = grid_idx(ub + 2 + 8 * j + 4 * half, hi_spread.iq2s[ub[66 + 2 * j + half]]);
                    const uint8_t* spg = gb + 2 + 32 + 8 * j + 4 * half;
                    const uint8_t* spu = ub + 2 + 32 + 8 * j + 4 * half;
                    const uint8_t sbg = gb[74 + 2 * j + half], sbu = ub[74 + 2 * j + half];
                    for (int which = 0; which < 2; ++which) {
                        const int8_t* cg = (const int8_t*) kscale16.v[which ? (sbg >> 4) : (sbg & 15)];
                        const int8_t* cu = (const int8_t*) kscale16.v[which ? (sbu >> 4) : (sbu & 15)];
                        const __m128i yv = _mm_loadu_si128((const __m128i*) (q8 + off + 16 * which));
                        const __m128i gg = _mm_set_epi64x((long long) iq2s_grid[_mm_extract_epi16(ig, 2 * which + 1)],
                                                          (long long) iq2s_grid[_mm_extract_epi16(ig, 2 * which)]);
                        const __m128i gu = _mm_set_epi64x((long long) iq2s_grid[_mm_extract_epi16(iu, 2 * which + 1)],
                                                          (long long) iq2s_grid[_mm_extract_epi16(iu, 2 * which)]);
                        const __m128i sg = _mm_set_epi64x((long long) sign8.v[spg[2 * which + 1]],
                                                          (long long) sign8.v[spg[2 * which]]);
                        const __m128i su = _mm_set_epi64x((long long) sign8.v[spu[2 * which + 1]],
                                                          (long long) sign8.v[spu[2 * which]]);
                        if (which == 0) {
                            ag0 = madd_add_c(ag0, gg, sg, yv, cg);
                            au0 = madd_add_c(au0, gu, su, yv, cu);
                        } else {
                            ag1 = madd_add_c(ag1, gg, sg, yv, cg);
                            au1 = madd_add_c(au1, gu, su, yv, cu);
                        }
                    }
                }
            }
            const float dy = y0[i].d;
            fg = _mm_add_ps(fg, _mm_mul_ps(_mm_set1_ps(h2f(u16(gb)) * Fmt128<TY>::K * dy),
                                           _mm_cvtepi32_ps(_mm_add_epi32(ag0, ag1))));
            fu = _mm_add_ps(fu, _mm_mul_ps(_mm_set1_ps(h2f(u16(ub)) * Fmt128<TY>::K * dy),
                                           _mm_cvtepi32_ps(_mm_add_epi32(au0, au1))));
        }
        float hg = hsum4(fg), hu = hsum4(fu);
        ff[0][r] = (hg / (1.f + std::exp(-hg))) * hu;
    }
}

template <int TY> void gu_rows_nt(int nt, const uint8_t* blob, size_t gu_row, size_t up_off, int n,
               const void* const* act, float* const* ff, int r0, int r1);

// MODE at one token: 0 = two gate rows then two up rows, no hoist (the path shipped until now),
// 1 = one gate row paired with one up row, activation loads hoisted (the candidate), 2 = the old pairing
// with the loads hoisted.  shortcut: kept only while the A/B decides between them; delete the losers.
template <int TY>
void gu_rows_nt_mode(int nt, const uint8_t* blob, size_t gu_row, size_t up_off, int n, const void* const* act,
                     float* const* ff, int r0, int r1, int mode) {
    if (nt == 1) {
        switch (mode) {
            case 0: gu_rows<TY, 1, 0>(blob, gu_row, up_off, n, act, ff, r0, r1); return;
            case 2: gu_rows<TY, 1, 2>(blob, gu_row, up_off, n, act, ff, r0, r1); return;
            case 3: gu_rows<TY, 1, 3>(blob, gu_row, up_off, n, act, ff, r0, r1); return;
            case 4: gu_rows_phased<TY>(blob, gu_row, up_off, n, act, ff, r0, r1); return;
            case 5: gu_rows_phased30<TY>(blob, gu_row, up_off, n, act, ff, r0, r1); return;
            case 6: gu_rows_phased2<TY, 2>(blob, gu_row, up_off, n, act, ff, r0, r1); return;
            case 7: gu_rows_phased31<TY>(blob, gu_row, up_off, n, act, ff, r0, r1); return;
            case 8: gu_rows_dualstream<TY>(blob, gu_row, up_off, n, act, ff, r0, r1); return;
            case 10:   // packed byte loads (PK)
                if (TY == 18 || TY == 21) gu_rows_phased32<TY, true, false>(blob, gu_row, up_off, n, act, ff, r0, r1);
                else if (TY == 22) gu_rows_phased33<TY, true, false>(blob, gu_row, up_off, n, act, ff, r0, r1);
                else gu_rows<TY, 1, 1>(blob, gu_row, up_off, n, act, ff, r0, r1);
                return;
            case 11:   // split accumulator chain (SP)
                if (TY == 18 || TY == 21) gu_rows_phased32<TY, false, true>(blob, gu_row, up_off, n, act, ff, r0, r1);
                else if (TY == 22) gu_rows_phased33<TY, false, true>(blob, gu_row, up_off, n, act, ff, r0, r1);
                else gu_rows<TY, 1, 1>(blob, gu_row, up_off, n, act, ff, r0, r1);
                return;
            case 20:   // affine codebook arithmetic in place of the grid table (IQ2_S)
                if (TY == 22) gu_rows_iq2s_aff<TY>(blob, gu_row, up_off, n, act, ff, r0, r1);
                else if (TY == 18 || TY == 21) gu_rows_phased31<TY>(blob, gu_row, up_off, n, act, ff, r0, r1);
                else gu_rows<TY, 1, 1>(blob, gu_row, up_off, n, act, ff, r0, r1);
                return;
            case 17:   // round-10 P1: IQ3_XXS codebook loads pre-decoded across the sub-block boundary
                if (TY == 18) gu_rows_iq3xxs_gpipe<TY>(blob, gu_row, up_off, n, act, ff, r0, r1);
                else if (TY == 21) gu_rows_phased31<TY>(blob, gu_row, up_off, n, act, ff, r0, r1);
                else if (TY == 22) gu_rows_iq2s_pipelined<TY>(blob, gu_row, up_off, n, act, ff, r0, r1);
                else gu_rows<TY, 1, 1>(blob, gu_row, up_off, n, act, ff, r0, r1);
                return;
            case 18:   // round-10 P2: IQ2_S codebook loads pre-decoded (mode 13, one stage further)
                if (TY == 22) gu_rows_iq2s_gpipe<TY>(blob, gu_row, up_off, n, act, ff, r0, r1);
                else if (TY == 18 || TY == 21) gu_rows_phased31<TY>(blob, gu_row, up_off, n, act, ff, r0, r1);
                else gu_rows<TY, 1, 1>(blob, gu_row, up_off, n, act, ff, r0, r1);
                return;
            case 19:   // round-10 P3: IQ3_S dedup + index pre-issue
                if (TY == 21) gu_rows_iq3s_dpipe<TY>(blob, gu_row, up_off, n, act, ff, r0, r1);
                else if (TY == 18) gu_rows_phased31<TY>(blob, gu_row, up_off, n, act, ff, r0, r1);
                else if (TY == 22) gu_rows_iq2s_pipelined<TY>(blob, gu_row, up_off, n, act, ff, r0, r1);
                else gu_rows<TY, 1, 1>(blob, gu_row, up_off, n, act, ff, r0, r1);
                return;
            case 16:   // mode 13's mechanism: index build / index loads one sub-half early
                if (TY == 21) gu_rows_iq3s_pipe<TY>(blob, gu_row, up_off, n, act, ff, r0, r1);
                else if (TY == 18) gu_rows_iq3xxs_pipe<TY>(blob, gu_row, up_off, n, act, ff, r0, r1);
                else if (TY == 22) gu_rows_phased<TY>(blob, gu_row, up_off, n, act, ff, r0, r1);
                else gu_rows<TY, 1, 1>(blob, gu_row, up_off, n, act, ff, r0, r1);
                return;
            case 13:   // report-9 proposal 1: software-pipelined index pre-issue (IQ2_S)
                if (TY == 22) gu_rows_iq2s_pipelined<TY>(blob, gu_row, up_off, n, act, ff, r0, r1);
                else if (TY == 18 || TY == 21) gu_rows_phased31<TY>(blob, gu_row, up_off, n, act, ff, r0, r1);
                else gu_rows<TY, 1, 1>(blob, gu_row, up_off, n, act, ff, r0, r1);
                return;
            case 14:   // report-9 proposal 2: deduplicated grid_idx (IQ3_S / IQ3_XXS)
                if (TY == 21) gu_rows_iq3s_dedup<TY>(blob, gu_row, up_off, n, act, ff, r0, r1);
                else if (TY == 18) gu_rows_phased31<TY>(blob, gu_row, up_off, n, act, ff, r0, r1);
                else if (TY == 22) gu_rows_phased<TY>(blob, gu_row, up_off, n, act, ff, r0, r1);
                else gu_rows<TY, 1, 1>(blob, gu_row, up_off, n, act, ff, r0, r1);
                return;
            case 15:   // this repo's answer: take the 8 vpextrw per unit off port 0 (IQ2_S)
                if (TY == 22) gu_rows_iq2s_scalar<TY>(blob, gu_row, up_off, n, act, ff, r0, r1);
                else if (TY == 18 || TY == 21) gu_rows_phased31<TY>(blob, gu_row, up_off, n, act, ff, r0, r1);
                else gu_rows<TY, 1, 1>(blob, gu_row, up_off, n, act, ff, r0, r1);
                return;
            case 12:   // both
                if (TY == 18 || TY == 21) gu_rows_phased32<TY, true, true>(blob, gu_row, up_off, n, act, ff, r0, r1);
                else if (TY == 22) gu_rows_phased33<TY, true, true>(blob, gu_row, up_off, n, act, ff, r0, r1);
                else gu_rows<TY, 1, 1>(blob, gu_row, up_off, n, act, ff, r0, r1);
                return;
            default: gu_rows<TY, 1, 1>(blob, gu_row, up_off, n, act, ff, r0, r1); return;
        }
    }
    gu_rows_nt<TY>(nt, blob, gu_row, up_off, n, act, ff, r0, r1);
}

template <int TY>
void gu_rows_nt(int nt, const uint8_t* blob, size_t gu_row, size_t up_off, int n, const void* const* act,
                float* const* ff, int r0, int r1) {
    switch (nt) {
        case 1: gu_rows<TY, 1, 1>(blob, gu_row, up_off, n, act, ff, r0, r1); break;
        case 2: gu_rows<TY, 2, 1>(blob, gu_row, up_off, n, act, ff, r0, r1); break;
        case 3: gu_rows<TY, 3, 1>(blob, gu_row, up_off, n, act, ff, r0, r1); break;
        case 4: gu_rows<TY, 4, 1>(blob, gu_row, up_off, n, act, ff, r0, r1); break;
        case 5: gu_rows<TY, 5, 1>(blob, gu_row, up_off, n, act, ff, r0, r1); break;
        case 6: gu_rows<TY, 6, 1>(blob, gu_row, up_off, n, act, ff, r0, r1); break;
        case 7: gu_rows<TY, 7, 1>(blob, gu_row, up_off, n, act, ff, r0, r1); break;
        default: gu_rows<TY, 8, 1>(blob, gu_row, up_off, n, act, ff, r0, r1); break;
    }
}

// ---- IQ4_NL (20) down rows against Q8_0.  Unlike the other i-quants, ggml-cpu HAS a hand-written __AVX__
// path for this type (two blocks per iteration, 256-bit float accumulation, no FMA so mul+add), and it beat
// the first 128-bit kernel here.  The row loop takes the same two-block shape, but decodes each block pair
// ONCE and lets every token share it - which ggml's per-token dot cannot - and keeps the int32 madd chain
// (exact) with a single 256-bit cvt per pair.
template <int NT, bool DERIVE>
void iq4nl_rows(const uint8_t* w, size_t row_bytes, int n, const block_q8_0* const* y, float* const* out,
                int r0, int r1) {
    const __m128i values = _mm_loadu_si128((const __m128i*) kvalues_iq4nl);
    const __m128i m4b = _mm_set1_epi8(0x0f);
    const __m128i ones = _mm_set1_epi16(1);
    const int nb = n / QK4_NL;
    const int pf = prefetch_distance();

        // The activation scales y[t][ib].d converted from fp16 once per (token, block) instead of once per
    // (row, token, block): the row loop ran h2f() on the same nb values for every row of the expert.
    // std::vector, not float yd[NT][nb]: a VLA is not standard C++ and MSVC (which builds this file too)
    // refuses it.  One small allocation per call, against ~0.6 ms of work in the loop below.
    std::vector<float> yd((size_t) NT * nb);
    for (int t = 0; t < NT; ++t)
        for (int ib = 0; ib < nb; ++ib) yd[(size_t) t * nb + ib] = h2f(y[t][ib].d);

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
            __m128i a00, a01, a10, a11;      // |w|: the unsigned maddubs operand
            if (!DERIVE) {                   // DERIVE: keep only v* live and take |w| with pabsb at the use site
                a00 = _mm_sign_epi8(v00, v00); a01 = _mm_sign_epi8(v01, v01);
                a10 = _mm_sign_epi8(v10, v10); a11 = _mm_sign_epi8(v11, v11);
            }
            const float dx0 = h2f(u16(blk0)), dx1 = h2f(u16(blk1));
            for (int t = 0; t < NT; ++t) {
                const uint8_t* q8a = (const uint8_t*) y[t][ib].qs;
                const uint8_t* q8b = (const uint8_t*) y[t][ib + 1].qs;
                const __m128i ya0 = _mm_loadu_si128((const __m128i*) q8a);
                const __m128i ya1 = _mm_loadu_si128((const __m128i*) (q8a + 16));
                const __m128i yb0 = _mm_loadu_si128((const __m128i*) q8b);
                const __m128i yb1 = _mm_loadu_si128((const __m128i*) (q8b + 16));
                const __m128i pa = _mm_add_epi32(
                    _mm_madd_epi16(_mm_maddubs_epi16(DERIVE ? _mm_abs_epi8(v00) : a00, _mm_sign_epi8(ya0, v00)), ones),
                    _mm_madd_epi16(_mm_maddubs_epi16(DERIVE ? _mm_abs_epi8(v01) : a01, _mm_sign_epi8(ya1, v01)), ones));
                const __m128i pb = _mm_add_epi32(
                    _mm_madd_epi16(_mm_maddubs_epi16(DERIVE ? _mm_abs_epi8(v10) : a10, _mm_sign_epi8(yb0, v10)), ones),
                    _mm_madd_epi16(_mm_maddubs_epi16(DERIVE ? _mm_abs_epi8(v11) : a11, _mm_sign_epi8(yb1, v11)), ones));
                const __m256 p = _mm256_cvtepi32_ps(_mm256_castps_si256(_mm256_set_m128(_mm_castsi128_ps(pb), _mm_castsi128_ps(pa))));
                const __m256 dd = _mm256_set_m128(_mm_set1_ps(dx1 * yd[(size_t) t * nb + ib + 1]),
                                                  _mm_set1_ps(dx0 * yd[(size_t) t * nb + ib]));
                acc[t] = _mm256_add_ps(acc[t], _mm256_mul_ps(dd, p));
            }
        }
        for (; ib < nb; ++ib) {           // odd tail, one block
            const uint8_t* blk = row + (size_t) ib * sizeof(block_iq4_nl);
            const __m128i bits = _mm_loadu_si128((const __m128i*) (blk + 2));
            const __m128i v0 = _mm_shuffle_epi8(values, _mm_and_si128(bits, m4b));
            const __m128i v1 = _mm_shuffle_epi8(values, _mm_and_si128(_mm_srli_epi16(bits, 4), m4b));
            const __m128i aq0 = DERIVE ? _mm_abs_epi8(v0) : _mm_sign_epi8(v0, v0);
            const __m128i aq1 = DERIVE ? _mm_abs_epi8(v1) : _mm_sign_epi8(v1, v1);
            const float dx = h2f(u16(blk));
            for (int t = 0; t < NT; ++t) {
                const uint8_t* q8 = (const uint8_t*) y[t][ib].qs;
                const __m128i p0 = _mm_madd_epi16(_mm_maddubs_epi16(aq0, _mm_sign_epi8(_mm_loadu_si128((const __m128i*) q8), v0)), ones);
                const __m128i p1 = _mm_madd_epi16(_mm_maddubs_epi16(aq1, _mm_sign_epi8(_mm_loadu_si128((const __m128i*) (q8 + 16)), v1)), ones);
                const __m128 f = _mm_cvtepi32_ps(_mm_add_epi32(p0, p1));   // exact int sum, one cvt
                acc[t] = _mm256_add_ps(acc[t], _mm256_castps128_ps256(_mm_mul_ps(_mm_set1_ps(dx * yd[(size_t) t * nb + ib]), f)));
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
// Upstream AVX2 unpack trick ported to 128-bit SSE4.2: expands 8 code bytes into two 16-byte code vectors
// using cvtepu8_epi32 and word shifts/or, eliminating 4 unpack shuffles on Port 5.
inline void q2_unpack_128(const uint8_t* qs, __m128i& k0, __m128i& k1) {
    const __m128i m3 = _mm_set1_epi32(0x03030303);
    __m128i v0 = _mm_cvtepu8_epi32(_mm_cvtsi32_si128(*(const int*) qs));
    v0 = _mm_or_si128(v0, _mm_slli_epi32(v0, 12));
    v0 = _mm_or_si128(v0, _mm_slli_epi32(v0,  6));
    k0 = _mm_and_si128(v0, m3);

    __m128i v1 = _mm_cvtepu8_epi32(_mm_cvtsi32_si128(*(const int*) (qs + 4)));
    v1 = _mm_or_si128(v1, _mm_slli_epi32(v1, 12));
    v1 = _mm_or_si128(v1, _mm_slli_epi32(v1,  6));
    k1 = _mm_and_si128(v1, m3);
}

template <int NT>
void q2_0_rows(const uint8_t* w, size_t row_bytes, int n, const block_q8_0* const* y, float* const* out,
               int r0, int r1) {
    const __m128i ones = _mm_set1_epi16(1);
    const int nb = n / QK2_0;
    const int pf = prefetch_distance();

    // Precompute activation scales outside row loop: eliminates 51,200 redundant h2f() calls.
    std::vector<float> yd((size_t) NT * nb * 2);
    for (int t = 0; t < NT; ++t) {
        for (int b = 0; b < nb * 2; ++b) {
            yd[(size_t) t * nb * 2 + b] = h2f(y[t][b].d);
        }
    }

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
                    q2_unpack_128(blka + 2 + 8 * k, k0a, k1a);
                    q2_unpack_128(blkb + 2 + 8 * k, k0b, k1b);
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
                    const float dy = yd[(size_t) i * 2 + k];
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
                q2_unpack_128(qs + 8 * k, k0, k1);
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
                    accf[t] = _mm_add_ps(accf[t], _mm_mul_ps(_mm_set1_ps(dx * yd[(size_t) t * nb * 2 + i * 2 + k]), f));
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
    // IQ4_NL (20) is back: with the activation scales converted once per (token, block) and the block decode
    // shared across the window, iq4nl_rows beats ggml's hand-written __AVX__ dot - measured on an E5-2470 v2
    // (iq_avx1_parity --bench, min of 3, 2026-10-18): 1.17x at nt=1, 1.40x at nt=2, 1.48x at nt=4.  Q2_0 (42)
    // stays on iq128_down_rows: 5-7x ggml's scalar generic, and 1.2-1.75x better than this file was before the
    // same two fixes landed there.
    return type == 20 || type == 42;
}

void iq128_gu_rows(int type, const uint8_t* blob, size_t gu_row, size_t up_off, int n, const void* const* act,
                   int nt, float* const* ff, int r0, int r1) {
    // The one-token pairing of one gate row with one up row is measured per format (iq_avx1_parity, min of 4
    // alternating, 7 samples): IQ3_S +4.9% (7/7 samples), IQ2_S +2.6% (6/7), IQ3_XXS flat, IQ4_XS -23%.  So the
    // mode is chosen per type, not globally, and every mode writes bit for bit the same rows.
    switch (type) {
        case 18: gu_rows_nt_mode<18>(nt, blob, gu_row, up_off, n, act, ff, r0, r1, 7); break;   // phased, fold kept
        case 21: gu_rows_nt_mode<21>(nt, blob, gu_row, up_off, n, act, ff, r0, r1, 16); break;  // phased, fold kept, index build one sub-half early.  Re-measured on the 0.2.0 base: +3.85% (7/8 samples, 4 trials x min-of-5 alternating, both IQ3_S rungs); it was only +1.4% on the 0.1.45 base, which is why mode 13 shipped for IQ2_S and this waited.
        case 22: gu_rows_nt_mode<22>(nt, blob, gu_row, up_off, n, act, ff, r0, r1, 13); break;  // phased 16-value decode, index build one sub-block early (+2.5% at nt=1, 12/12 clean samples)
        case 23: gu_rows_nt_mode<23>(nt, blob, gu_row, up_off, n, act, ff, r0, r1, 0); break;
        default: break;
    }
}

void iq128_gu_rows_mode(int type, const uint8_t* blob, size_t gu_row, size_t up_off, int n, const void* const* act,
                        int nt, float* const* ff, int r0, int r1, int mode) {
    switch (type) {
        case 18: gu_rows_nt_mode<18>(nt, blob, gu_row, up_off, n, act, ff, r0, r1, mode); break;
        case 21: gu_rows_nt_mode<21>(nt, blob, gu_row, up_off, n, act, ff, r0, r1, mode); break;
        case 22: gu_rows_nt_mode<22>(nt, blob, gu_row, up_off, n, act, ff, r0, r1, mode); break;
        case 23: gu_rows_nt_mode<23>(nt, blob, gu_row, up_off, n, act, ff, r0, r1, mode); break;
        default: std::abort();
    }
}

// mode 1 = IQ4_NL at one token with |w| derived by pabsb instead of kept live (a register-census change).
// MEASURED, NOT USED: 12 paired samples at nt=1 gave mean 0.983x, 3/12 positive (0.941x to 1.026x).  The gate/up
// kernels won by shrinking the live set; this one does not, so the down loop is not register-limited the same way -
// it carries one accumulator per token and the 2x block unroll is what supplies its instruction-level parallelism.
// MODE 9 (measurement, report-8 proposal B): two IQ4_NL down rows in flight sharing the activation loads.  The
// per-row arithmetic is a copy of iq4nl_rows' - same 2-block unroll, block ib in lanes 0..3 and ib+1 in lanes 4..7,
// one cvt / one mul / one add per pair of blocks - because any other float grouping changes the bits.
void iq4nl_down_pair_nt1(const uint8_t* w, size_t row_bytes, int n, const block_q8_0* const* y,
                         float* const* out, int r0, int r1) {
    const int nb = n / QK4_NL;
    const block_q8_0* y0 = y[0];
    if (nb & 1) { iq4nl_rows<1, false>(w, row_bytes, n, &y0, out, r0, r1); return; }   // keep the shipped grouping
    const __m128i values = _mm_loadu_si128((const __m128i*) kvalues_iq4nl);
    const __m128i m4b = _mm_set1_epi8(0x0f);
    const __m128i ones = _mm_set1_epi16(1);
    const int pf = prefetch_distance();
    std::vector<float> yd(nb);
    for (int ib = 0; ib < nb; ++ib) yd[ib] = h2f(y0[ib].d);
    int r = r0;
    for (; r + 1 < r1; r += 2) {
        const uint8_t* rowA = w + (size_t) r * row_bytes;
        const uint8_t* rowB = w + (size_t) (r + 1) * row_bytes;
        __m256 accA = _mm256_setzero_ps(), accB = _mm256_setzero_ps();
        for (int ib = 0; ib < nb; ib += 2) {
            const uint8_t* A0 = rowA + (size_t) ib * sizeof(block_iq4_nl);
            const uint8_t* A1 = A0 + sizeof(block_iq4_nl);
            const uint8_t* B0 = rowB + (size_t) ib * sizeof(block_iq4_nl);
            const uint8_t* B1 = B0 + sizeof(block_iq4_nl);
            rows_ahead(A0, pf); rows_ahead(B0, pf);
            const uint8_t* q8 = (const uint8_t*) y0[ib].qs;
            const uint8_t* q8b = (const uint8_t*) y0[ib + 1].qs;
            const __m128i ya0 = _mm_loadu_si128((const __m128i*) q8);
            const __m128i ya1 = _mm_loadu_si128((const __m128i*) (q8 + 16));
            const __m128i yb0 = _mm_loadu_si128((const __m128i*) q8b);
            const __m128i yb1 = _mm_loadu_si128((const __m128i*) (q8b + 16));
            const __m128i tA0 = _mm_loadu_si128((const __m128i*) (A0 + 2));
            const __m128i tA1 = _mm_loadu_si128((const __m128i*) (A1 + 2));
            const __m128i tB0 = _mm_loadu_si128((const __m128i*) (B0 + 2));
            const __m128i tB1 = _mm_loadu_si128((const __m128i*) (B1 + 2));
            const __m128i vA0 = _mm_shuffle_epi8(values, _mm_and_si128(tA0, m4b));
            const __m128i vA1 = _mm_shuffle_epi8(values, _mm_and_si128(_mm_srli_epi16(tA0, 4), m4b));
            const __m128i vA2 = _mm_shuffle_epi8(values, _mm_and_si128(tA1, m4b));
            const __m128i vA3 = _mm_shuffle_epi8(values, _mm_and_si128(_mm_srli_epi16(tA1, 4), m4b));
            const __m128i aA0 = _mm_sign_epi8(vA0, vA0), aA1 = _mm_sign_epi8(vA1, vA1);
            const __m128i aA2 = _mm_sign_epi8(vA2, vA2), aA3 = _mm_sign_epi8(vA3, vA3);
            const __m128i pA = _mm_add_epi32(
                _mm_madd_epi16(_mm_maddubs_epi16(aA0, _mm_sign_epi8(ya0, vA0)), ones),
                _mm_madd_epi16(_mm_maddubs_epi16(aA1, _mm_sign_epi8(ya1, vA1)), ones));
            const __m128i qA = _mm_add_epi32(
                _mm_madd_epi16(_mm_maddubs_epi16(aA2, _mm_sign_epi8(yb0, vA2)), ones),
                _mm_madd_epi16(_mm_maddubs_epi16(aA3, _mm_sign_epi8(yb1, vA3)), ones));
            accA = _mm256_add_ps(accA, _mm256_mul_ps(
                _mm256_set_m128(_mm_set1_ps(h2f(u16(A1)) * yd[ib + 1]), _mm_set1_ps(h2f(u16(A0)) * yd[ib])),
                _mm256_cvtepi32_ps(_mm256_castps_si256(_mm256_set_m128(_mm_castsi128_ps(qA), _mm_castsi128_ps(pA))))));
            const __m128i vB0 = _mm_shuffle_epi8(values, _mm_and_si128(tB0, m4b));
            const __m128i vB1 = _mm_shuffle_epi8(values, _mm_and_si128(_mm_srli_epi16(tB0, 4), m4b));
            const __m128i vB2 = _mm_shuffle_epi8(values, _mm_and_si128(tB1, m4b));
            const __m128i vB3 = _mm_shuffle_epi8(values, _mm_and_si128(_mm_srli_epi16(tB1, 4), m4b));
            const __m128i aB0 = _mm_sign_epi8(vB0, vB0), aB1 = _mm_sign_epi8(vB1, vB1);
            const __m128i aB2 = _mm_sign_epi8(vB2, vB2), aB3 = _mm_sign_epi8(vB3, vB3);
            const __m128i pB = _mm_add_epi32(
                _mm_madd_epi16(_mm_maddubs_epi16(aB0, _mm_sign_epi8(ya0, vB0)), ones),
                _mm_madd_epi16(_mm_maddubs_epi16(aB1, _mm_sign_epi8(ya1, vB1)), ones));
            const __m128i qB = _mm_add_epi32(
                _mm_madd_epi16(_mm_maddubs_epi16(aB2, _mm_sign_epi8(yb0, vB2)), ones),
                _mm_madd_epi16(_mm_maddubs_epi16(aB3, _mm_sign_epi8(yb1, vB3)), ones));
            accB = _mm256_add_ps(accB, _mm256_mul_ps(
                _mm256_set_m128(_mm_set1_ps(h2f(u16(B1)) * yd[ib + 1]), _mm_set1_ps(h2f(u16(B0)) * yd[ib])),
                _mm256_cvtepi32_ps(_mm256_castps_si256(_mm256_set_m128(_mm_castsi128_ps(qB), _mm_castsi128_ps(pB))))));
        }
        out[0][r] = hsum4(_mm_add_ps(_mm256_castps256_ps128(accA), _mm256_extractf128_ps(accA, 1)));
        out[0][r + 1] = hsum4(_mm_add_ps(_mm256_castps256_ps128(accB), _mm256_extractf128_ps(accB, 1)));
    }
    for (; r < r1; ++r) iq4nl_rows<1, false>(w, row_bytes, n, &y0, out, r, r + 1);
}

void iq128_down_rows_mode(int type, const uint8_t* w, size_t row_bytes, int n, const void* const* hq, int nt,
                          float* const* out, int r0, int r1, int mode) {
    const block_q8_0* y0 = (const block_q8_0*) hq[0];
    if (mode == 1 && type == 20 && nt == 1) { iq4nl_rows<1, true>(w, row_bytes, n, &y0, out, r0, r1); return; }
    if (mode == 9 && type == 20 && nt == 1) { iq4nl_down_pair_nt1(w, row_bytes, n, &y0, out, r0, r1); return; }
    iq128_down_rows(type, w, row_bytes, n, hq, nt, out, r0, r1);
}

void iq128_down_rows(int type, const uint8_t* w, size_t row_bytes, int n, const void* const* hq, int nt,
                     float* const* out, int r0, int r1) {
    const block_q8_0* y[8];
    for (int t = 0; t < nt && t < 8; ++t) y[t] = (const block_q8_0*) hq[t];
    if (type == 20) {
        // IQ4_NL down at one token runs two rows in flight sharing the activation loads (mode 9 shape): measured
        // 24/24 samples positive, medians 1.058-1.066 across the four IQ4_NL rungs.  Other token counts keep the
        // 4-token chunking, where one decode already feeds four tokens and there is nothing to share.
        if (nt == 1) { iq4nl_down_pair_nt1(w, row_bytes, n, y, out, r0, r1); return; }
        for (int t0 = 0; t0 < nt; t0 += 4) {
            const int k = nt - t0 < 4 ? nt - t0 : 4;
            switch (k) {
                case 1: iq4nl_rows<1,false>(w, row_bytes, n, y + t0, out + t0, r0, r1); break;
                case 2: iq4nl_rows<2,false>(w, row_bytes, n, y + t0, out + t0, r0, r1); break;
                case 3: iq4nl_rows<3,false>(w, row_bytes, n, y + t0, out + t0, r0, r1); break;
                default: iq4nl_rows<4,false>(w, row_bytes, n, y + t0, out + t0, r0, r1); break;
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
