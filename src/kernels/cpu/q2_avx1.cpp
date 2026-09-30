// src/kernels/cpu/q2_avx1.cpp - LOCAL PORT (Z620): the Q2_0 expert rows and the activation quantizer for CPUs
// with AVX but no AVX2 and no FMA3 (Sandy Bridge / Westmere-era Xeons, i.e. Xeon E5-2600 v1).
//
// WHY THIS FILE EXISTS
// --------------------
// Upstream has exactly two Q2_0 CPU paths: the AVX-512 kernels in `expert.cpp` and the AVX2 ones in
// `q2_avx2.cpp`, and `q2_rows_any()` picks between them with NO AVX2 feature test at all:
//
//     if (cpu_avx512_ok()) q2_0_gguf_rows_multi(...);
//     else                 q2_0_gguf_rows_multi_avx2(...);      // <- SIGILL on a pre-AVX2 CPU
//
// so a Sandy Bridge either traps on the first expert row or is refused at startup by
// `cpu_require_expert_support()`. This file is the missing third rung.
//
// WHAT THE TARGET CPU ACTUALLY HAS  (measured, not assumed - see the note below)
// ---------------------------------------------------------------------------
// An HP Z620 with a Xeon E5-2680, family 6 model 45 stepping 7 = Sandy Bridge-E. Probed on the machine:
//
//     SSE3 / SSSE3 / SSE4.1 / SSE4.2 / AVX : present
//     FMA3 (leaf 1 ECX 12)                 : ABSENT   -> vfmadd*  raises #UD
//     F16C (leaf 1 ECX 29)                 : ABSENT   -> vcvtph2ps raises #UD
//     AVX2, AVX-512                        : ABSENT
//
// FMA3 and F16C arrived with IVY BRIDGE (model 62), one generation AFTER Sandy Bridge. Assuming
// otherwise is the trap here: a kernel that reaches for `_mm_fmadd_ps` or `_mm_cvtph_ps` compiles
// cleanly with -mfma -mf16c and then dies with an illegal instruction on exactly the machine it was
// written for. So this file uses NEITHER, and `cpu_avx1_ok()` correspondingly does not require them.
//
// WHAT IS USED
//   * AVX      256-bit float (the row accumulator stays a float vector)
//   * SSSE3    `_mm_shuffle_epi8` in the 2-bit code unpack
//   * SSE4.1   `_mm_maddubs_epi16` / `_mm_madd_epi16`, the int8 dot; `_mm_min_epi32`/`_mm_max_epi32`
//   * 128-bit integer ops only - every 256-bit use is FLOAT, which plain AVX allows. A 256-bit integer
//     operation would need AVX2 and would trap, so the int8 dot is 16 lanes wide rather than 32.
//
// THE TWO PLACES THIS DEPARTS FROM THE AVX2 KERNEL, both forced by the above:
//   1. `h2f` is a software fp16 decode instead of `_mm_cvtph_ps`.  It is called once per 64-weight
//      block, so the branchy version costs little against the dot it feeds.
//   2. The accumulation is `_mm_add_ps(_mm_mul_ps(...))` instead of `_mm_fmadd_ps`, which costs one
//      extra instruction per block and, more importantly, is the only legal option without FMA3.
//
// ACCURACY.  The per-block int32 dot is the same widening-then-pairwise-sum the AVX2 kernel performs, in
// the same order, so it is bit-identical; only the lane grouping and the final FP32 reduction differ.
// That makes the parity tolerance 1e-3 (P2.S3's own number for VNNI-vs-oracle) rather than bitwise.
//
// The per-file compile flag is what keeps this safe: a TU built for AVX only cannot emit AVX2, FMA3 or
// AVX-512 anywhere, so even a wrong dispatch answer costs speed instead of trapping. That is the same
// property `expert.cpp` and `q2_avx2.cpp` already rely on.
#include "strata/kernels/cpu/expert.hpp"

#include <immintrin.h>

#include <cmath>
#include <cstring>

namespace strata::kernels::cpu {
namespace {

/// fp16 -> fp32 in software, because F16C is not available on this target.  Bit manipulation rather
/// than `ldexp` so it is a handful of predictable instructions; the common case (a normal, non-zero
/// exponent) is the single branch at the bottom.
inline float h2f(const uint8_t* p) {
    uint16_t h;
    std::memcpy(&h, p, 2);
    const uint32_t sign = (uint32_t) (h & 0x8000u) << 16;
    const uint32_t exp = (h >> 10) & 0x1Fu;
    const uint32_t man = h & 0x3FFu;
    uint32_t bits;
    if (exp == 0) {
        if (man == 0) {
            bits = sign;
        } else {                                    // subnormal: normalise into a float exponent
            uint32_t m = man;
            int e = -1;
            do { m <<= 1; ++e; } while (!(m & 0x400u));
            bits = sign | ((uint32_t) (127 - 15 - e) << 23) | ((m & 0x3FFu) << 13);
        }
    } else if (exp == 31) {
        bits = sign | 0x7F800000u | (man << 13);   // inf / nan
    } else {
        bits = sign | ((exp - 15 + 127) << 23) | (man << 13);
    }
    float f;
    std::memcpy(&f, &bits, 4);
    return f;
}

/// 16 bytes of 2-bit codes -> the 64 codes in value order, as FOUR 16-byte vectors.
/// Code i is byte i/4 at bit offset 2*(i%4), so the shift/and ladder is the AVX2 file's, with the lane
/// count halved: c0..c3 hold codes 0-15, 16-31, 32-47, 48-63.
inline void unpack64_sse(const uint8_t* codes, __m128i& v0, __m128i& v1, __m128i& v2, __m128i& v3) {
    const __m128i b = _mm_loadu_si128((const __m128i*) codes);
    const __m128i m3 = _mm_set1_epi8(3);
    const __m128i c0 = _mm_and_si128(b, m3);
    const __m128i c1 = _mm_and_si128(_mm_srli_epi16(b, 2), m3);
    const __m128i c2 = _mm_and_si128(_mm_srli_epi16(b, 4), m3);
    const __m128i c3 = _mm_and_si128(_mm_srli_epi16(b, 6), m3);
    // Interleave with 8/16-wide moves instead of the AVX2 file's `_mm256_set_m128i`.
    const __m128i a0 = _mm_unpacklo_epi8(c0, c1);        // codes 0..7   and 16..23
    const __m128i a1 = _mm_unpacklo_epi8(c2, c3);        // codes 8..15  and 24..31
    const __m128i b0 = _mm_unpackhi_epi8(c0, c1);        // codes 32..39 and 48..55
    const __m128i b1 = _mm_unpackhi_epi8(c2, c3);        // codes 40..47 and 56..63
    v0 = _mm_unpacklo_epi16(a0, a1);                     // codes 0..15
    v1 = _mm_unpackhi_epi16(a0, a1);                     // codes 16..31
    v2 = _mm_unpacklo_epi16(b0, b1);                     // codes 32..47
    v3 = _mm_unpackhi_epi16(b0, b1);                     // codes 48..63
}

/// 16 codes against 16 int8 activations -> four int32 partial sums, one per lane.
/// `maddubs` widens the unsigned-byte products to int16, `madd` sums adjacent pairs against 1.  The
/// lanes are a partition of the 16 products, and the caller reduces all of them at the end, so WHICH
/// product lands in which lane does not matter - only the total does.
inline __m128i dot4(const __m128i& codes, const int8_t* q, __m128i ones) {
    return _mm_madd_epi16(_mm_maddubs_epi16(codes, _mm_loadu_si128((const __m128i*) q)), ones);
}

/// Sum the four int32 lanes down to one.
inline int32_t hsum_epi32(__m128i v) {
    __m128i s = _mm_add_epi32(v, _mm_shuffle_epi32(v, _MM_SHUFFLE(1, 0, 3, 2)));
    s = _mm_add_epi32(s, _mm_shuffle_epi32(s, _MM_SHUFFLE(2, 3, 0, 1)));
    return _mm_cvtsi128_si32(s);
}

/// Sum the four float lanes down to one: [a0 a1 a2 a3] -> a0+a1+a2+a3, added into lane 0.
inline float hsum_ps(__m128 v) {
    __m128 h = _mm_add_ps(v, _mm_movehl_ps(v, v));          // [a0+a2, a1+a3, .., ..]
    h = _mm_add_ss(h, _mm_shuffle_ps(h, h, _MM_SHUFFLE(1, 1, 1, 1)));
    return _mm_cvtss_f32(h);
}

/// Horizontal MAX over the four float lanes.  Distinct from `hsum_ps` on purpose: the quantizer needs
/// the max of |x| over a chunk, and reducing it with an add is silently wrong - it inflates the scale by
/// roughly 4x and shifts every code, which is what an early revision of this file did.
inline float hmax_ps(__m128 v) {
    __m128 h = _mm_max_ps(v, _mm_movehl_ps(v, v));          // [max(a0,a2), max(a1,a3), .., ..]
    h = _mm_max_ss(h, _mm_shuffle_ps(h, h, _MM_SHUFFLE(1, 1, 1, 1)));
    return _mm_cvtss_f32(h);
}

template <int NT>
inline void row_multi(const uint8_t* row, const ActQ* const* a, int nblocks, float* res) {
    // The accumulator stays a FLOAT VECTOR and is reduced ONCE at the end of the row, which is the
    // single most important property here: reducing per 64-weight block made the AVX-512 kernel
    // compute-bound at 21 GB/s instead of DRAM-bound at 32.
    __m128 acc[NT];
    float corr[NT];
    const __m128i ones = _mm_set1_epi16(1);
    for (int t = 0; t < NT; ++t) {
        acc[t] = _mm_setzero_ps();
        corr[t] = 0.f;
    }
    for (int b = 0; b < nblocks; ++b) {
        const uint8_t* blk = row + (size_t) b * 18;         // native layout: 2 fp16 scale + 16 code bytes
        const float d = h2f(blk);
        __m128i v0, v1, v2, v3;
        unpack64_sse(blk + 2, v0, v1, v2, v3);
        for (int t = 0; t < NT; ++t) {
            const int8_t* q = a[t]->q + b * 64;
            // 64 weights, but the activation is chunked at QKA=32, so a weight block spans TWO chunks
            // with two different scales. One scale per weight block is the natural-looking mistake.
            const __m128i s0 = _mm_add_epi32(dot4(v0, q, ones), dot4(v1, q + 16, ones));
            const __m128i s1 = _mm_add_epi32(dot4(v2, q + 32, ones), dot4(v3, q + 48, ones));
            // mul+add, not FMA: this target has no FMA3.
            acc[t] = _mm_add_ps(acc[t], _mm_mul_ps(_mm_set1_ps(d * a[t]->scale[2 * b]), _mm_cvtepi32_ps(s0)));
            acc[t] = _mm_add_ps(acc[t], _mm_mul_ps(_mm_set1_ps(d * a[t]->scale[2 * b + 1]), _mm_cvtepi32_ps(s1)));
            corr[t] += d * (a[t]->hx[2 * b] + a[t]->hx[2 * b + 1]);
        }
    }
    for (int t = 0; t < NT; ++t) res[t] = hsum_ps(acc[t]) - corr[t];
}

template <int NT>
void rows(const uint8_t* w, size_t row_bytes, int nblocks, const ActQ* const* a, float* const* out, int r0, int r1) {
    float res[NT];
    for (int r = r0; r < r1; ++r) {
        row_multi<NT>(w + (size_t) r * row_bytes, a, nblocks, res);
        for (int t = 0; t < NT; ++t) out[t][r] = res[t];
    }
}

}  // namespace

void q2_0_gguf_rows_multi_avx1(const uint8_t* w, size_t row_bytes, int nblocks, const ActQ* const* a, int nt,
                               float* const* out, int r0, int r1) {
    switch (nt) {
        case 1: rows<1>(w, row_bytes, nblocks, a, out, r0, r1); break;
        case 2: rows<2>(w, row_bytes, nblocks, a, out, r0, r1); break;
        case 3: rows<3>(w, row_bytes, nblocks, a, out, r0, r1); break;
        case 4: rows<4>(w, row_bytes, nblocks, a, out, r0, r1); break;
        default:
            // Same four-token batching the AVX2 file uses.
            for (int t0 = 0; t0 < nt; t0 += 4) {
                const int k = nt - t0 < 4 ? nt - t0 : 4;
                q2_0_gguf_rows_multi_avx1(w, row_bytes, nblocks, a + t0, k, out + t0, r0, r1);
            }
    }
}

/// The activation quantizer, the scalar rule bit for bit.  QKA=32 elements per chunk: scale = amax/127,
/// codes by half-away-from-zero rounding, clamped to [-127, 127], and the `hx` correction term.
void act_quant_q8_1_avx1(const float* x, int n, ActQ& a) {
    a.nchunks = n / QKA;
    const __m128 absmask = _mm_castsi128_ps(_mm_set1_epi32(0x7fffffff));
    const __m128 half = _mm_set1_ps(0.5f), mhalf = _mm_set1_ps(-0.5f), zero = _mm_setzero_ps();
    const __m128i lo = _mm_set1_epi32(-127), hi = _mm_set1_epi32(127);
    for (int k = 0; k < a.nchunks; ++k) {
        const float* xb = x + k * QKA;
        __m128 v[8];
        __m128 m = _mm_setzero_ps();
        for (int i = 0; i < 8; ++i) {
            v[i] = _mm_loadu_ps(xb + 4 * i);
            m = _mm_max_ps(m, _mm_and_ps(v[i], absmask));
        }
        const float amax = hmax_ps(m);
        const float s = amax > 0.f ? amax / 127.f : 0.f;
        const float inv = s > 0.f ? 1.f / s : 0.f;
        const __m128 vinv = _mm_set1_ps(inv);
        __m128i sum = _mm_setzero_si128();
        alignas(16) int32_t qi[QKA];
        for (int i = 0; i < 8; ++i) {
            const __m128 t = _mm_mul_ps(v[i], vinv);
            const __m128 r = _mm_add_ps(t, _mm_blendv_ps(mhalf, half, _mm_cmp_ps(t, zero, _CMP_GE_OQ)));
            __m128i q = _mm_cvttps_epi32(r);
            q = _mm_min_epi32(_mm_max_epi32(q, lo), hi);
            sum = _mm_add_epi32(sum, q);
            _mm_storeu_si128((__m128i*) (qi + 4 * i), q);
        }
        for (int j = 0; j < QKA; ++j) a.q[k * QKA + j] = (int8_t) qi[j];
        const int32_t total = hsum_epi32(sum);
        a.scale[k] = s;
        a.sum[k] = total;
        a.hx[k] = s * (float) total;
    }
}

}  // namespace strata::kernels::cpu
