// src/kernels/cpu/s2_expert_avx1.cpp - the CANONICAL Q2_0 expert path for CPUs with AVX but
// no AVX2, no FMA3, no F16C and no AVX-512 (Sandy Bridge / Westmere-era Xeons, e.g. Xeon E5-2600 v1).
//
// WHY THIS IS THE FILE THAT MATTERS
// ---------------------------------
// q2_avx1.cpp (above) covered the NATIVE pack layout, reached through `q2_rows_any`.
// But `pool.cpp` - the actual CPU worker loop, i.e. the main expert compute path - does not use it.  The pool
// calls `s2_expert_vnni_q`, `s2_expert_gu_rows`, `s2_expert_down_rows` and their `_multi` forms, and EVERY one
// of those is defined in `expert.cpp`, a translation unit compiled with `/arch:AVX512`.  `generate.cpp` gates
// on that with `cpu_require_expert_support()` and exits.  So without this file a canonical Q2_0 pack - which
// is what the recommended Q2_0 / IQ2_XS sizes actually produce - cannot run here at all, however good the
// native kernel is.
//
// THE TWO LAYOUTS, and why this one is different
// ----------------------------------------------
//   CANONICAL (here): codes and scales in SEPARATE arrays.  A row is `nblocks` 16-byte code blocks, and the
//                     fp16 scales live in their own array at 2 bytes per block.
//   NATIVE (q2_avx1.cpp): ggml-style INTERLEAVED 18-byte blocks (2-byte fp16 scale then 16 code bytes).
// The arithmetic is identical; only where the scale sits differs.  Mixing them up reads 720 bytes out of a
// 640-byte row, which is a fault, not a wrong answer - so the two pointer arguments here are named to make
// the split obvious at every call site.
//
// THE CONTRACT, which is easy to get subtly wrong
// ----------------------------------------------
// One Q2_0 weight block is QK=64 weights sharing one fp16 scale `d`.  One activation chunk is QKA=32
// elements with its OWN scale, so a WEIGHT BLOCK SPANS TWO ACTIVATION CHUNKS with two different scales.
// Using one scale per weight block is the natural-looking mistake and is called out in expert.hpp.  Hence
// the two `dot4` pairs per block and the two `scale[2*b]` / `scale[2*b+1]` reads.
//
// ACCURACY.  The per-block int32 dot is the same widening-then-pairwise-sum as the AVX-512 kernel, in the
// same order, so it is bit-identical.  Only the lane grouping and the final FP32 reduction differ, which is
// why the parity check is a tolerance (1e-3, the same number P2.S3 uses for VNNI-vs-oracle) rather than
// bitwise.
//
// ISA: 128-bit integer ops, 256-bit FLOAT only.  Every ymm use is a float operation, so this runs on a CPU
// whose only vector ISA is AVX.  No FMA3 (`_mm_add_ps(_mm_mul_ps(...))` instead) and no F16C (`h2f` below is
// a software decode, because `vcvtph2ps` raises #UD on the target).
#include "strata/kernels/cpu/expert.hpp"

#include <immintrin.h>

#include <cmath>
#include <cstring>

namespace strata::kernels::cpu {
namespace {

/// fp16 -> fp32 in software (no F16C on this target).  Called once per 64-weight block.
inline float h2f_avx1(const uint8_t* p) {
    uint16_t h;
    std::memcpy(&h, p, 2);
    const uint32_t sign = (uint32_t) (h & 0x8000u) << 16;
    const uint32_t exp = (h >> 10) & 0x1Fu;
    const uint32_t man = h & 0x3FFu;
    uint32_t bits;
    if (exp == 0) {
        if (man == 0) {
            bits = sign;
        } else {
            uint32_t m = man;
            int e = -1;
            do { m <<= 1; ++e; } while (!(m & 0x400u));
            bits = sign | ((uint32_t) (127 - 15 - e) << 23) | ((m & 0x3FFu) << 13);
        }
    } else if (exp == 31) {
        bits = sign | 0x7F800000u | (man << 13);
    } else {
        bits = sign | ((exp - 15 + 127) << 23) | (man << 13);
    }
    float f;
    std::memcpy(&f, &bits, 4);
    return f;
}

/// 16 bytes of 2-bit codes -> 64 codes in value order, as four 16-byte vectors (codes 0-15, 16-31, 32-47,
/// 48-63).  Code i is byte i/4 at bit offset 2*(i%4).
inline void unpack64_sse(const uint8_t* codes, __m128i& v0, __m128i& v1, __m128i& v2, __m128i& v3) {
    const __m128i b = _mm_loadu_si128((const __m128i*) codes);
    const __m128i m3 = _mm_set1_epi8(3);
    const __m128i c0 = _mm_and_si128(b, m3);
    const __m128i c1 = _mm_and_si128(_mm_srli_epi16(b, 2), m3);
    const __m128i c2 = _mm_and_si128(_mm_srli_epi16(b, 4), m3);
    const __m128i c3 = _mm_and_si128(_mm_srli_epi16(b, 6), m3);
    const __m128i a0 = _mm_unpacklo_epi8(c0, c1);
    const __m128i a1 = _mm_unpacklo_epi8(c2, c3);
    const __m128i b0 = _mm_unpackhi_epi8(c0, c1);
    const __m128i b1 = _mm_unpackhi_epi8(c2, c3);
    v0 = _mm_unpacklo_epi16(a0, a1);
    v1 = _mm_unpackhi_epi16(a0, a1);
    v2 = _mm_unpacklo_epi16(b0, b1);
    v3 = _mm_unpackhi_epi16(b0, b1);
}

/// 16 codes against 16 int8 activations -> four int32 partial sums.  The lanes partition the 16 products
/// and the caller reduces all of them, so which product lands where does not matter - only the total.
inline __m128i dot4(const __m128i& codes, const int8_t* q, __m128i ones) {
    return _mm_madd_epi16(_mm_maddubs_epi16(codes, _mm_loadu_si128((const __m128i*) q)), ones);
}

inline int32_t hsum_epi32(__m128i v) {
    __m128i s = _mm_add_epi32(v, _mm_shuffle_epi32(v, _MM_SHUFFLE(1, 0, 3, 2)));
    s = _mm_add_epi32(s, _mm_shuffle_epi32(s, _MM_SHUFFLE(2, 3, 0, 1)));
    return _mm_cvtsi128_si32(s);
}

inline float hsum_ps(__m128 v) {
    __m128 h = _mm_add_ps(v, _mm_movehl_ps(v, v));
    h = _mm_add_ss(h, _mm_shuffle_ps(h, h, _MM_SHUFFLE(1, 1, 1, 1)));
    return _mm_cvtss_f32(h);
}

/// One canonical row against ONE activation.  `row_codes` and `row_scales` are SEPARATE arrays; that split
/// is the whole difference from the native kernel.
inline float row_dot_avx1(const uint8_t* row_codes, const uint8_t* row_scales, const ActQ& a, int nblocks) {
    const __m128i ones = _mm_set1_epi16(1);
    __m128 acc = _mm_setzero_ps();
    float corr = 0.f;
    for (int b = 0; b < nblocks; ++b) {
        const float d = h2f_avx1(row_scales + 2 * b);
        __m128i v0, v1, v2, v3;
        unpack64_sse(row_codes + (size_t) b * 16, v0, v1, v2, v3);
        const int8_t* q = a.q + (size_t) b * 64;
        const __m128i s0 = _mm_add_epi32(dot4(v0, q, ones), dot4(v1, q + 16, ones));
        const __m128i s1 = _mm_add_epi32(dot4(v2, q + 32, ones), dot4(v3, q + 48, ones));
        acc = _mm_add_ps(acc, _mm_mul_ps(_mm_set1_ps(d * a.scale[2 * b]), _mm_cvtepi32_ps(s0)));
        acc = _mm_add_ps(acc, _mm_mul_ps(_mm_set1_ps(d * a.scale[2 * b + 1]), _mm_cvtepi32_ps(s1)));
        corr += d * (a.hx[2 * b] + a.hx[2 * b + 1]);
    }
    return hsum_ps(acc) - corr;
}

/// The same row against NT activations, with the codes unpacked ONCE for all of them.  This is the whole
/// point of the multi-token form: a speculative verify window routes ~1.5 tokens to each expert, so the
/// unpack cost is shared instead of paid per token.  Each token keeps its own float accumulator, so every
/// token's result is identical to calling the single-token path on it alone.
template <int NT>
inline void row_dot_multi_avx1(const uint8_t* row_codes, const uint8_t* row_scales, const ActQ* const* a,
                               int nblocks, float* res) {
    const __m128i ones = _mm_set1_epi16(1);
    __m128 acc[NT];
    float corr[NT];
    for (int t = 0; t < NT; ++t) {
        acc[t] = _mm_setzero_ps();
        corr[t] = 0.f;
    }
    for (int b = 0; b < nblocks; ++b) {
        const float d = h2f_avx1(row_scales + 2 * b);
        __m128i v0, v1, v2, v3;
        unpack64_sse(row_codes + (size_t) b * 16, v0, v1, v2, v3);
        for (int t = 0; t < NT; ++t) {
            const int8_t* q = a[t]->q + (size_t) b * 64;
            const __m128i s0 = _mm_add_epi32(dot4(v0, q, ones), dot4(v1, q + 16, ones));
            const __m128i s1 = _mm_add_epi32(dot4(v2, q + 32, ones), dot4(v3, q + 48, ones));
            acc[t] = _mm_add_ps(acc[t], _mm_mul_ps(_mm_set1_ps(d * a[t]->scale[2 * b]), _mm_cvtepi32_ps(s0)));
            acc[t] = _mm_add_ps(acc[t], _mm_mul_ps(_mm_set1_ps(d * a[t]->scale[2 * b + 1]), _mm_cvtepi32_ps(s1)));
            corr[t] += d * (a[t]->hx[2 * b] + a[t]->hx[2 * b + 1]);
        }
    }
    for (int t = 0; t < NT; ++t) res[t] = hsum_ps(acc[t]) - corr[t];
}

template <int NT>
void gu_rows_multi_avx1(const uint8_t* blob, const ActQ* const* a1, float* const* ff, int r0, int r1) {
    float g[NT], u[NT];
    for (int r = r0; r < r1; ++r) {
        row_dot_multi_avx1<NT>(blob + O_GU_CODES + (size_t) (2 * r) * ROW_GU,
                               blob + O_GU_SCALES + (size_t) (2 * r) * SC_GU * 2, a1, SC_GU, g);
        row_dot_multi_avx1<NT>(blob + O_GU_CODES + (size_t) (2 * r + 1) * ROW_GU,
                               blob + O_GU_SCALES + (size_t) (2 * r + 1) * SC_GU * 2, a1, SC_GU, u);
        // SiLU on the GATE, times up - the reading docs/semantics.md records, and the one that is wrong the
        // other way round in a way that still produces a plausible number.
        for (int t = 0; t < NT; ++t) ff[t][r] = (g[t] / (1.f + std::exp(-g[t]))) * u[t];
    }
}

template <int NT>
void down_rows_multi_avx1(const uint8_t* blob, const ActQ* const* a2, float* const* out, int r0, int r1) {
    float o[NT];
    for (int r = r0; r < r1; ++r) {
        row_dot_multi_avx1<NT>(blob + O_D_CODES + (size_t) r * ROW_D, blob + O_D_SCALES + (size_t) r * SC_D * 2,
                               a2, SC_D, o);
        for (int t = 0; t < NT; ++t) out[t][r] = o[t];
    }
}

/// Dispatch on the token count, batching four at a time past that, exactly as the AVX-512 path does.
#define STRATA_S2_GU_DISPATCH(NT)                                                                    \
    case NT:                                                                                         \
        gu_rows_multi_avx1<NT>(blob, a1, ff, r0, r1);                                                \
        break;

#define STRATA_S2_DOWN_DISPATCH(NT)                                                                  \
    case NT:                                                                                         \
        down_rows_multi_avx1<NT>(blob, a2, out, r0, r1);                                            \
        break;

}  // namespace

void s2_expert_vnni_q_avx1(const uint8_t* blob, const ActQ& a1, float* out, ExpertScratch& ws) {
    for (int r = 0; r < FF; ++r) {
        const float g = row_dot_avx1(blob + O_GU_CODES + (size_t) (2 * r) * ROW_GU,
                                     blob + O_GU_SCALES + (size_t) (2 * r) * SC_GU * 2, a1, SC_GU);
        const float u = row_dot_avx1(blob + O_GU_CODES + (size_t) (2 * r + 1) * ROW_GU,
                                     blob + O_GU_SCALES + (size_t) (2 * r + 1) * SC_GU * 2, a1, SC_GU);
        ws.ff[r] = (g / (1.f + std::exp(-g))) * u;
    }
    act_quant_q8_1_avx1(ws.ff, FF, ws.a2);
    for (int r = 0; r < H; ++r)
        out[r] = row_dot_avx1(blob + O_D_CODES + (size_t) r * ROW_D, blob + O_D_SCALES + (size_t) r * SC_D * 2,
                              ws.a2, SC_D);
}

void s2_expert_gu_rows_avx1(const uint8_t* blob, const ActQ& a1, float* ff, int r0, int r1) {
    for (int r = r0; r < r1; ++r) {
        const float g = row_dot_avx1(blob + O_GU_CODES + (size_t) (2 * r) * ROW_GU,
                                     blob + O_GU_SCALES + (size_t) (2 * r) * SC_GU * 2, a1, SC_GU);
        const float u = row_dot_avx1(blob + O_GU_CODES + (size_t) (2 * r + 1) * ROW_GU,
                                     blob + O_GU_SCALES + (size_t) (2 * r + 1) * SC_GU * 2, a1, SC_GU);
        ff[r] = (g / (1.f + std::exp(-g))) * u;
    }
}

void s2_expert_down_rows_avx1(const uint8_t* blob, const ActQ& a2, float* out, int r0, int r1) {
    for (int r = r0; r < r1; ++r)
        out[r] = row_dot_avx1(blob + O_D_CODES + (size_t) r * ROW_D, blob + O_D_SCALES + (size_t) r * SC_D * 2, a2,
                              SC_D);
}

void s2_expert_gu_rows_multi_avx1(const uint8_t* blob, const ActQ* const* a1, int n_tokens, float* const* ff,
                                  int r0, int r1) {
    switch (n_tokens) {
        STRATA_S2_GU_DISPATCH(1)
        STRATA_S2_GU_DISPATCH(2)
        STRATA_S2_GU_DISPATCH(3)
        STRATA_S2_GU_DISPATCH(4)
        STRATA_S2_GU_DISPATCH(5)
        STRATA_S2_GU_DISPATCH(6)
        STRATA_S2_GU_DISPATCH(7)
        default:
            for (int t0 = 0; t0 < n_tokens; t0 += 4) {
                const int k = n_tokens - t0 < 4 ? n_tokens - t0 : 4;
                s2_expert_gu_rows_multi_avx1(blob, a1 + t0, k, ff + t0, r0, r1);
            }
    }
}

void s2_expert_down_rows_multi_avx1(const uint8_t* blob, const ActQ* const* a2, int n_tokens, float* const* out,
                                    int r0, int r1) {
    switch (n_tokens) {
        STRATA_S2_DOWN_DISPATCH(1)
        STRATA_S2_DOWN_DISPATCH(2)
        STRATA_S2_DOWN_DISPATCH(3)
        STRATA_S2_DOWN_DISPATCH(4)
        STRATA_S2_DOWN_DISPATCH(5)
        STRATA_S2_DOWN_DISPATCH(6)
        STRATA_S2_DOWN_DISPATCH(7)
        default:
            for (int t0 = 0; t0 < n_tokens; t0 += 4) {
                const int k = n_tokens - t0 < 4 ? n_tokens - t0 : 4;
                s2_expert_down_rows_multi_avx1(blob, a2 + t0, k, out + t0, r0, r1);
            }
    }
}

void s2_expert_vnni_multi_avx1(const uint8_t* blob, const ActQ* const* a1, int n_tokens, float* const* out,
                               ExpertScratchMulti& ws) {
    // MAXT is the engine's verify-window bound; beyond it, fall back to one full expert per token.
    if (n_tokens > MAXT) {
        for (int t = 0; t < n_tokens; ++t) s2_expert_vnni_q_avx1(blob, *a1[t], out[t], ws.single);
        return;
    }
    // Gate/up first, all tokens, so the codes are unpacked once per block for the whole window; then
    // requantize the intermediate; then the down rows.  The two phases cannot be fused: the down rows
    // need the quantized intermediate, which only exists once every gate/up row is done.
    float* ff[MAXT];
    for (int t = 0; t < n_tokens; ++t) ff[t] = ws.ff[t];
    s2_expert_gu_rows_multi_avx1(blob, a1, n_tokens, ff, 0, FF);

    const ActQ* a2[MAXT];
    for (int t = 0; t < n_tokens; ++t) {
        act_quant_q8_1_avx1(ws.ff[t], FF, ws.a2[t]);
        a2[t] = &ws.a2[t];
    }
    s2_expert_down_rows_multi_avx1(blob, a2, n_tokens, out, 0, H);
}

}  // namespace strata::kernels::cpu

#undef STRATA_S2_GU_DISPATCH
#undef STRATA_S2_DOWN_DISPATCH
