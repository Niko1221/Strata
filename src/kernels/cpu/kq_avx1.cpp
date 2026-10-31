// src/kernels/cpu/kq_avx1.cpp - the AVX1 router dot for older CPUs (see kq_avx1.hpp).  From rwkeyes' Strata_Dirigo
// fork (MIT), its AVX1 floor.
//
// Measured on the hardware this exists for (Xeon E5-2665, 2x8 cores, AVX only) for one layer's router
// [512 experts x 5120 embeddings]: the scalar std::fma fallback it replaces took 362 ms/layer, a scalar
// mul+add 4.9 ms, and this kernel 2.6 ms - 137x, with no extra memory (the weights stay bf16).
//
// Compiled for AVX (/arch:AVX, -mavx; CMakeLists.txt sets that per source); it is only ever called behind
// cpu_avx1_ok(), so no CPU that lacks AVX can reach it.
#include "strata/kernels/cpu/kq_avx1.hpp"

#include "ggml.h"        // ggml_fp16_to_fp32
#define GGML_COMMON_DECL_CPP
#define GGML_COMMON_IMPL_CPP
#include "ggml-common.h" // block_q4_K / block_q8_K / QK_K: a multi-inclusion template, and the layout must be
                          // ggml's own - a re-typed struct with a wrong sizeof silently mis-advances the rows.
                          // Same three lines as kq_avx2.cpp:16 and iq_avx1.cpp:25.

#include <immintrin.h>

#include <cmath>
#include <cstring>

#if !defined(__AVX__)
#error "kq_avx1.cpp must be compiled with AVX enabled (see the per-source flags in CMakeLists.txt)"
#endif

namespace strata::kernels::cpu {
namespace {

// 8 floats -> one float, the same reduction order the AVX2 kernel's hsum_float_8 uses.
inline float hsum_float_8_avx1(__m256 v) {
    __m128 lo = _mm256_castps256_ps128(v);
    __m128 hi = _mm256_extractf128_ps(v, 1);
    lo = _mm_add_ps(lo, hi);
    lo = _mm_add_ps(lo, _mm_movehl_ps(lo, lo));
    lo = _mm_add_ss(lo, _mm_shuffle_ps(lo, lo, 0x55));
    return _mm_cvtss_f32(lo);
}

// 8 bf16 -> 8 f32: a bf16 is the top half of its f32, so widen in 128-bit halves (AVX1 has no
// 256-bit integer ops) and shift each 32-bit lane into the high half.
inline __m256 bf16x8_to_f32x8_avx1(const uint16_t* p) {
    const __m128i v = _mm_loadu_si128((const __m128i*) p);
    const __m128i z = _mm_setzero_si128();
    const __m128 lo = _mm_castsi128_ps(_mm_slli_epi32(_mm_unpacklo_epi16(v, z), 16));
    const __m128 hi = _mm_castsi128_ps(_mm_slli_epi32(_mm_unpackhi_epi16(v, z), 16));
    return _mm256_insertf128_ps(_mm256_castps128_ps256(lo), hi, 1);
}

}  // namespace

void bf16_rows_dot_multi_avx1(const uint16_t* w, int rows, int cols, const float* x, int nt, float* out) {
    // TOKENS ARE WALKED IN BLOCKS OF 8.  The accumulator is 8 wide and the old code trusted a comment that said
    // "nt <= 8", while expert_source.cpp passes `nt = n_tok_` - the whole batch.  A prefill chunk (the engine sizes
    // its own chunks near 17k) would have written thousands of __m256 past acc[8]: stack corruption on the AVX1
    // router, which is the live path on every CPU without AVX2.  It survived only because this call is the expert
    // PREFETCH estimate and its callers happened to hand it small windows - a contract enforced by luck.
    // Chunking is numerically identical for nt <= 8 (same rows, same order of additions per token).
    for (int t0 = 0; t0 < nt; t0 += 8) {
        const int m = nt - t0 < 8 ? nt - t0 : 8;
    for (int r = 0; r < rows; ++r) {
        const uint16_t* wr = w + (size_t) r * (size_t) cols;
        __m256 acc[8];
        for (int t = 0; t < m; ++t) acc[t] = _mm256_setzero_ps();
        int c = 0;
        for (; c + 8 <= cols; c += 8) {
            const __m256 wf = bf16x8_to_f32x8_avx1(wr + c);
            for (int t = 0; t < m; ++t) {
                const __m256 xv = _mm256_loadu_ps(x + (size_t) (t0 + t) * (size_t) cols + c);
                acc[t] = _mm256_add_ps(acc[t], _mm256_mul_ps(wf, xv));   // no FMA on AVX1: mul, then add
            }
        }
        for (int t = 0; t < m; ++t) {
            float s = hsum_float_8_avx1(acc[t]);
            for (; c < cols; ++c) {                                      // cols % 8 != 0 tail (cols % 8 == 0 today)
                const uint32_t bits = (uint32_t) wr[c] << 16;
                float wf; std::memcpy(&wf, &bits, sizeof wf);
                s += wf * x[(size_t) (t0 + t) * (size_t) cols + c];
            }
            out[(size_t) (t0 + t) * (size_t) rows + r] = s;
        }
    }
    }
}

// ---- Q4_K gate/up rows, multi-token, on top of ggml's own AVX1 rung (gu_type 12).
//
// WHY.  kq_cpu_bench on this E5-2470 v2: Q4_K 32.3 / 20.5 / 11.0 GB/s at 1/2/4 tokens, where q8_0 keeps
// 33.4 / 32.0 / 21.4.  At nt=1 every format sits at the DRAM ceiling, so nothing is left there; the loss is
// multi-token, and it is the per-block work the AVX1 rung redoes per token: the utmp scale/min dance, the two
// scale shuffles per 64 values, and the nibble expansion.  kq_cpu_bench's own `q4_K/q5_1` line says the weights
// are read once either way, so this is ALU, not bandwidth.
//
// HOW, and why it is bit-exact rather than merely close.  The per-token instruction sequence below is
// ggml_vec_dot_q4_K_q8_K's `#elif defined __AVX__` branch (arch/x86/quants.c:2122-2204) - same ops, same order,
// same lane mapping: maddubs -> madd(scale) -> add_epi32 into the same sumi_0/sumi_1 lane pair, the same
// hadd/madd mins path into acc_m, the same `acc += d * cvt(sumi)` per super-block, and the same final
// `hsum_float_8(acc) + hsum4(acc_m)`.  Only work that does not depend on the token moves out of the loop: the
// nibble expansion (into a 32-byte-aligned stack buffer, replayed per token) and the two scale shuffle vectors.
// Same bits in, same instruction sequence, so the same bits out - per token, whatever nt is.
//
// The first attempt at this idea was portable C++ that decoded once into plain bytes and dotted with scalars.
// It measured 8x SLOWER than this rung (0.36 vs 2.92 ms/expert at nt=1) because it gave up maddubs - two byte
// products per instruction.  Avoiding the re-decode is worth tens of percent; losing the SIMD byte dot costs an
// order of magnitude.  Hence: keep the SIMD, hoist the decode.
//
// Compiled for AVX (see CMakeLists.txt per-source flags), gated on cpu_avx1_ok() and !cpu_avx2_ok(): the SSE4.1
// intrinsics used here are inside what -mavx enables, and they are the same ones the rung itself uses.
namespace {

/// One super-block, expanded once per window: 4 chunks x {q4l_0, q4h_0, q4l_1, q4h_1} of nibbles-as-bytes,
/// in the order the rung consumes them, plus the two scale shuffle vectors of each chunk and the mins vector.
/// 256 + 128 + 16 bytes, read back from L1 once per token.
struct KQ128Block {
    int8_t nib[4][4][16];              // j: low nibbles of bytes 0-15, high of 0-15, low of 16-31, high of 16-31
    int16_t sl[4][8], sh[4][8];        // scale_l / scale_h as _mm_shuffle_epi8 produced them
    int16_t mins[8];
    float xd, xdmin;                   // the block's fp16 halves, widened once (ggml widens them per token)
};

/// The fp16 pair a Q4_K super-block starts with, read at byte offsets: ggml-common.h declares {d, dmin} inside a
/// union and under -std=c++17 (`__STRICT_ANSI__`) that aggregate is named rather than anonymous, so `x.d` does
/// not exist.  kq_avx2.cpp:35 does the same thing for the same reason.
inline float block_half_at(const block_q4_K* x, int which) {
    ggml_half hh;
    std::memcpy(&hh, (const uint8_t*) x + which * sizeof(ggml_half), sizeof hh);
    return ggml_fp16_to_fp32(hh);
}

/// ggml_vec_dot_q4_K_q8_K_generic's bit dance for the 8 sub-block scales and mins, copied verbatim so the
/// values are identical bit for bit and not merely equivalent (arch/x86/quants.c:2049-2055 and 2137-2143).
inline void q4k_scales_mins(const block_q4_K& x, int16_t sl[4][8], int16_t sh[4][8], int16_t mins_out[8]) {
    static const uint32_t kmask1 = 0x3f3f3f3f, kmask2 = 0x0f0f0f0f, kmask3 = 0x03030303;
    uint32_t utmp[4];
    std::memcpy(utmp, x.scales, 12);
    utmp[3] = ((utmp[2] >> 4) & kmask2) | (((utmp[1] >> 6) & kmask3) << 4);
    const uint32_t uaux = utmp[1] & kmask1;
    utmp[1] = (utmp[2] & kmask2) | (((utmp[0] >> 6) & kmask3) << 4);
    utmp[2] = uaux;
    utmp[0] &= kmask1;

    const __m128i utmps = _mm_set_epi32(utmp[3], utmp[2], utmp[1], utmp[0]);
    const __m128i scales = _mm_cvtepu8_epi16(utmps);                       // 8 int16 scale lanes
    const __m128i minsv  = _mm_cvtepu8_epi16(_mm_unpackhi_epi64(utmps, utmps));
    _mm_storeu_si128((__m128i*) mins_out, minsv);

    // The rung picks scale bytes with an incrementing shuffle mask: bytes 0,2,4,.. then 1,3,5,.. of the packed
    // 8-bit scales, broadcast into 16-bit lanes.  Same masks, same results - done once per block here.
    const __m128i m2 = _mm_set1_epi8(0x2);   // the reference bumps BOTH bytes of the mask
    __m128i shuffle = _mm_set1_epi16(0x0100);
    for (int j = 0; j < 4; ++j) {
        _mm_storeu_si128((__m128i*) sl[j], _mm_shuffle_epi8(scales, shuffle));
        shuffle = _mm_add_epi16(shuffle, m2);
        _mm_storeu_si128((__m128i*) sh[j], _mm_shuffle_epi8(scales, shuffle));
        shuffle = _mm_add_epi16(shuffle, m2);
    }
}

/// Expand one super-block's nibbles into bytes, exactly as the rung's `_mm_and_si128` / `_mm_srli_epi16` pair
/// leaves them, in the order the rung feeds them to maddubs.
inline void q4k_nibbles(const block_q4_K& x, int8_t nib[4][4][16]) {
    const __m128i m4 = _mm_set1_epi8(0xF);
    const uint8_t* q4 = x.qs;
    for (int j = 0; j < 4; ++j) {
        const __m128i b0 = _mm_loadu_si128((const __m128i*) (q4));          // bytes 0-15
        const __m128i b1 = _mm_loadu_si128((const __m128i*) (q4 + 16));     // bytes 16-31
        _mm_storeu_si128((__m128i*) nib[j][0], _mm_and_si128(b0, m4));                          // q4l_0
        _mm_storeu_si128((__m128i*) nib[j][1], _mm_and_si128(_mm_srli_epi16(b0, 4), m4));       // q4h_0
        _mm_storeu_si128((__m128i*) nib[j][2], _mm_and_si128(b1, m4));                          // q4l_1
        _mm_storeu_si128((__m128i*) nib[j][3], _mm_and_si128(_mm_srli_epi16(b1, 4), m4));       // q4h_1
        q4 += 32;
    }
}

/// `out[t] = row . act[t]` for every token of the window: the block is expanded once, then each token runs the
/// rung's own instruction sequence over the expansion.
void q4k_dots_multi_avx1(const uint8_t* row, int nb, const block_q8_K* const* y, int nt, float* out) {
    alignas(32) KQ128Block b;
    __m256 acc[8];
    __m128 acc_m[8];
    for (int t = 0; t < nt; ++t) {
        acc[t] = _mm256_setzero_ps();
        acc_m[t] = _mm_setzero_ps();
    }
    for (int i = 0; i < nb; ++i) {
        block_q4_K x;
        std::memcpy(&x, row + (size_t) i * sizeof(block_q4_K), sizeof x);
        q4k_scales_mins(x, b.sl, b.sh, b.mins);            // once for all tokens
        q4k_nibbles(x, b.nib);                             // once for all tokens
        const float xd = block_half_at(&x, 0), xdmin = block_half_at(&x, 1);
        const __m128i minsv = _mm_loadu_si128((const __m128i*) b.mins);
        for (int t = 0; t < nt; ++t) {
            const block_q8_K& yq = y[t][i];
            const float d = yq.d * xd, dmin = -yq.d * xdmin;    // the rung's signs, kept as it writes them

            // mins . q8sums -> 4 int32 -> acc_m, the rung's hadd/madd path (depends on the token's bsums).
            const __m128i q8sums_0 = _mm_loadu_si128((const __m128i*) &yq.bsums[0]);
            const __m128i q8sums_1 = _mm_loadu_si128((const __m128i*) &yq.bsums[8]);
            const __m128i q8s = _mm_hadd_epi16(q8sums_0, q8sums_1);
            const __m128i prod = _mm_madd_epi16(minsv, q8s);
            acc_m[t] = _mm_add_ps(_mm_mul_ps(_mm_set1_ps(dmin), _mm_cvtepi32_ps(prod)), acc_m[t]);

            __m128i sumi_0 = _mm_setzero_si128(), sumi_1 = _mm_setzero_si128();
            const int8_t* q8 = yq.qs;
            for (int j = 0; j < 4; ++j) {
                const __m128i sl = _mm_loadu_si128((const __m128i*) b.sl[j]);
                const __m128i sh = _mm_loadu_si128((const __m128i*) b.sh[j]);
                __m128i p = _mm_maddubs_epi16(_mm_loadu_si128((const __m128i*) b.nib[j][0]),
                                              _mm_loadu_si128((const __m128i*) q8)); q8 += 16;
                sumi_0 = _mm_add_epi32(sumi_0, _mm_madd_epi16(sl, p));
                p = _mm_maddubs_epi16(_mm_loadu_si128((const __m128i*) b.nib[j][2]),
                                      _mm_loadu_si128((const __m128i*) q8)); q8 += 16;
                sumi_1 = _mm_add_epi32(sumi_1, _mm_madd_epi16(sl, p));
                p = _mm_maddubs_epi16(_mm_loadu_si128((const __m128i*) b.nib[j][1]),
                                      _mm_loadu_si128((const __m128i*) q8)); q8 += 16;
                sumi_0 = _mm_add_epi32(sumi_0, _mm_madd_epi16(sh, p));
                p = _mm_maddubs_epi16(_mm_loadu_si128((const __m128i*) b.nib[j][3]),
                                      _mm_loadu_si128((const __m128i*) q8)); q8 += 16;
                sumi_1 = _mm_add_epi32(sumi_1, _mm_madd_epi16(sh, p));
            }
            const __m256i sumi = _mm256_insertf128_si256(_mm256_castsi128_si256(sumi_0), sumi_1, 1);
            acc[t] = _mm256_add_ps(_mm256_mul_ps(_mm256_set1_ps(d), _mm256_cvtepi32_ps(sumi)), acc[t]);
        }
    }
    for (int t = 0; t < nt; ++t) {
        acc_m[t] = _mm_add_ps(acc_m[t], _mm_movehl_ps(acc_m[t], acc_m[t]));
        acc_m[t] = _mm_add_ss(acc_m[t], _mm_movehdup_ps(acc_m[t]));
        out[t] = hsum_float_8_avx1(acc[t]) + _mm_cvtss_f32(acc_m[t]);   // the rung's order: acc, then acc_m
    }
}

}  // namespace

void kq_gu_rows_avx1(int ggml_type, const uint8_t* blob, size_t gu_row, size_t up_off, int n,
                     const void* const* act, int nt, float* const* ff, int r0, int r1) {
    (void) ggml_type;                                        // Q4_K only; the caller's gate says so
    const int nb = n / QK_K;
    const block_q8_K* y[8];                                  // 8 float lanes x 2 accumulator kinds per token
    float g[8], u[8];
    for (int t0 = 0; t0 < nt; t0 += 8) {
        const int m = nt - t0 < 8 ? nt - t0 : 8;
        for (int t = 0; t < m; ++t) y[t] = (const block_q8_K*) act[t0 + t];
        for (int r = r0; r < r1; ++r) {
            q4k_dots_multi_avx1(blob + (size_t) r * gu_row, nb, y, m, g);
            q4k_dots_multi_avx1(blob + up_off + (size_t) r * gu_row, nb, y, m, u);
            for (int t = 0; t < m; ++t) {
                ff[t0 + t][r] = (g[t] / (1.f + std::exp(-g[t]))) * u[t];   // the fallback's silu, unaltered
            }
        }
    }
}

}  // namespace strata::kernels::cpu
