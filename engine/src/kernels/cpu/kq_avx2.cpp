// src/kernels/cpu/kq_avx2.cpp - Unsloth UD-Q4_K_XL's expert rows (Q4_K gate/up, Q5_1 and Q8_0 down) for several
// tokens at once, BIT-EXACT against ggml-cpu's own dot products.
//
// ggml-cpu is built for AVX2 here (STRATA_PORTABLE: GGML_AVX2=ON, GGML_AVX512=OFF), so its vec_dot for these formats
// is the __AVX2__ branch of ggml-cpu/arch/x86/quants.c: ggml_vec_dot_q4_K_q8_K, ggml_vec_dot_q5_1_q8_1,
// ggml_vec_dot_q8_0_q8_0.  The functions below are those branches with the loop over tokens moved inside the loop
// over blocks: everything that depends only on the weights (the 4-bit unpacking, the 6-bit scales and mins, the
// fifth bits, the absolute values of the Q8_0 weights, the fp16 scales) is done once per block, and every token then
// runs exactly ggml's integer chain and ggml's float operations in ggml's order - so each token's result is ggml's
// to the bit (native_expert_parity checks it), whatever the number of tokens.  Unlike the i-quant kernels (#152)
// the group size therefore never changes an answer.
//
// The gain is the weight-side work and the weight bytes, read once per verify window instead of once per token.
#include "strata/kernels/cpu/kq_avx2.hpp"

#define GGML_COMMON_DECL_CPP
#define GGML_COMMON_IMPL_CPP
#include "ggml-common.h"

#include <immintrin.h>

#include <cmath>
#include <cstring>

namespace strata::kernels::cpu {
namespace {

constexpr int MAXT = 8;   // the verify window's tokens per group (kVerifyMaxT)

inline float h2f(ggml_half h) {
    uint16_t u;
    std::memcpy(&u, &h, 2);
    return _mm_cvtss_f32(_mm_cvtph_ps(_mm_cvtsi32_si128((int) u)));
}
// the fp16 pair {d, dmin} / {d, m} / {d, s} a block starts with (a union in ggml-common.h's C++ declarations)
inline float half_at(const void* block, int i) {
    ggml_half h;
    std::memcpy(&h, (const uint8_t*) block + 2 * i, 2);
    return h2f(h);
}

// ---- ggml's helpers (ggml-cpu/arch/x86/quants.c, the AVX2 / non-VNNI forms), verbatim
inline float hsum_float_8(const __m256 x) {
    __m128 res = _mm256_extractf128_ps(x, 1);
    res = _mm_add_ps(res, _mm256_castps256_ps128(x));
    res = _mm_add_ps(res, _mm_movehl_ps(res, res));
    res = _mm_add_ss(res, _mm_movehdup_ps(res));
    return _mm_cvtss_f32(res);
}
inline __m256i bytes_from_bits_32(const uint8_t* x) {
    uint32_t x32;
    std::memcpy(&x32, x, sizeof(uint32_t));
    const __m256i shuf_mask = _mm256_set_epi64x(0x0303030303030303, 0x0202020202020202, 0x0101010101010101,
                                                0x0000000000000000);
    __m256i bytes = _mm256_shuffle_epi8(_mm256_set1_epi32((int) x32), shuf_mask);
    const __m256i bit_mask = _mm256_set1_epi64x(0x7fbfdfeff7fbfdfe);
    bytes = _mm256_or_si256(bytes, bit_mask);
    return _mm256_cmpeq_epi8(bytes, _mm256_set1_epi64x(-1));
}
inline __m256i bytes_from_nibbles_32(const uint8_t* rsi) {
    const __m128i tmp = _mm_loadu_si128((const __m128i*) rsi);
    const __m256i bytes = _mm256_insertf128_si256(_mm256_castsi128_si256(tmp), _mm_srli_epi16(tmp, 4), 1);
    const __m256i lowMask = _mm256_set1_epi8(0xF);
    return _mm256_and_si256(lowMask, bytes);
}
inline __m256 sum_i16_pairs_float(const __m256i x) {
    const __m256i ones = _mm256_set1_epi16(1);
    return _mm256_cvtepi32_ps(_mm256_madd_epi16(ones, x));
}
inline __m256 mul_sum_us8_pairs_float(const __m256i ax, const __m256i sy) {
    return sum_i16_pairs_float(_mm256_maddubs_epi16(ax, sy));
}
alignas(32) const uint8_t k_shuffle[256] = {
     0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1,
     2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3,
     4, 5, 4, 5, 4, 5, 4, 5, 4, 5, 4, 5, 4, 5, 4, 5, 4, 5, 4, 5, 4, 5, 4, 5, 4, 5, 4, 5, 4, 5, 4, 5,
     6, 7, 6, 7, 6, 7, 6, 7, 6, 7, 6, 7, 6, 7, 6, 7, 6, 7, 6, 7, 6, 7, 6, 7, 6, 7, 6, 7, 6, 7, 6, 7,
     8, 9, 8, 9, 8, 9, 8, 9, 8, 9, 8, 9, 8, 9, 8, 9, 8, 9, 8, 9, 8, 9, 8, 9, 8, 9, 8, 9, 8, 9, 8, 9,
    10,11,10,11,10,11,10,11,10,11,10,11,10,11,10,11,10,11,10,11,10,11,10,11,10,11,10,11,10,11,10,11,
    12,13,12,13,12,13,12,13,12,13,12,13,12,13,12,13,12,13,12,13,12,13,12,13,12,13,12,13,12,13,12,13,
    14,15,14,15,14,15,14,15,14,15,14,15,14,15,14,15,14,15,14,15,14,15,14,15,14,15,14,15,14,15,14,15};
inline __m256i get_scale_shuffle_k4(int i) { return _mm256_loadu_si256((const __m256i*) k_shuffle + i); }

// ---- Q4_K x Q8_K (ggml_vec_dot_q4_K_q8_K, __AVX2__)
void q4k_dot(const block_q4_K* x, int nb, const void* const* act, int nt, float* res) {
    static const uint32_t kmask1 = 0x3f3f3f3f, kmask2 = 0x0f0f0f0f, kmask3 = 0x03030303;
    const __m256i m4 = _mm256_set1_epi8(0xF);
    __m256 acc[MAXT];
    __m128 acc_m[MAXT];
    for (int t = 0; t < nt; ++t) { acc[t] = _mm256_setzero_ps(); acc_m[t] = _mm_setzero_ps(); }
    for (int i = 0; i < nb; ++i) {
        const float xd = half_at(&x[i], 0), xdmin = half_at(&x[i], 1);
        uint32_t utmp[4];
        std::memcpy(utmp, x[i].scales, 12);
        utmp[3] = ((utmp[2] >> 4) & kmask2) | (((utmp[1] >> 6) & kmask3) << 4);
        const uint32_t uaux = utmp[1] & kmask1;
        utmp[1] = (utmp[2] & kmask2) | (((utmp[0] >> 6) & kmask3) << 4);
        utmp[2] = uaux;
        utmp[0] &= kmask1;
        const __m256i mins_and_scales = _mm256_cvtepu8_epi16(_mm_set_epi32((int) utmp[3], (int) utmp[2],
                                                                           (int) utmp[1], (int) utmp[0]));
        const __m128i mins = _mm256_extracti128_si256(mins_and_scales, 1);
        const __m128i sc128 = _mm256_extracti128_si256(mins_and_scales, 0);
        const __m256i scales = _mm256_insertf128_si256(_mm256_castsi128_si256(sc128), sc128, 1);
        __m256i scale_l[4], scale_h[4], q4l[4], q4h[4];
        const uint8_t* q4 = x[i].qs;
        for (int j = 0; j < QK_K / 64; ++j) {
            scale_l[j] = _mm256_shuffle_epi8(scales, get_scale_shuffle_k4(2 * j + 0));
            scale_h[j] = _mm256_shuffle_epi8(scales, get_scale_shuffle_k4(2 * j + 1));
            const __m256i q4bits = _mm256_loadu_si256((const __m256i*) q4);
            q4 += 32;
            q4l[j] = _mm256_and_si256(q4bits, m4);
            q4h[j] = _mm256_and_si256(_mm256_srli_epi16(q4bits, 4), m4);
        }
        for (int t = 0; t < nt; ++t) {
            const block_q8_K* y = (const block_q8_K*) act[t] + i;
            const float d = y->d * xd;
            const float dmin = -y->d * xdmin;
            const int8_t* q8 = y->qs;
            const __m256i q8sums = _mm256_loadu_si256((const __m256i*) y->bsums);
            const __m128i q8s = _mm_hadd_epi16(_mm256_extracti128_si256(q8sums, 0), _mm256_extracti128_si256(q8sums, 1));
            const __m128i prod = _mm_madd_epi16(mins, q8s);
            acc_m[t] = _mm_fmadd_ps(_mm_set1_ps(dmin), _mm_cvtepi32_ps(prod), acc_m[t]);
            __m256i sumi = _mm256_setzero_si256();
            for (int j = 0; j < QK_K / 64; ++j) {
                const __m256i q8l = _mm256_loadu_si256((const __m256i*) q8);
                q8 += 32;
                __m256i p16l = _mm256_maddubs_epi16(q4l[j], q8l);
                p16l = _mm256_madd_epi16(scale_l[j], p16l);
                const __m256i q8h = _mm256_loadu_si256((const __m256i*) q8);
                q8 += 32;
                __m256i p16h = _mm256_maddubs_epi16(q4h[j], q8h);
                p16h = _mm256_madd_epi16(scale_h[j], p16h);
                sumi = _mm256_add_epi32(sumi, _mm256_add_epi32(p16l, p16h));
            }
            acc[t] = _mm256_fmadd_ps(_mm256_set1_ps(d), _mm256_cvtepi32_ps(sumi), acc[t]);
        }
    }
    for (int t = 0; t < nt; ++t) {
        __m128 m = acc_m[t];
        m = _mm_add_ps(m, _mm_movehl_ps(m, m));
        m = _mm_add_ss(m, _mm_movehdup_ps(m));
        res[t] = hsum_float_8(acc[t]) + _mm_cvtss_f32(m);
    }
}

// ---- Q5_1 x Q8_1 (ggml_vec_dot_q5_1_q8_1, __AVX2__)
template <int NT>
void q5_1_dot(const block_q5_1* x, int nb, const void* const* act, float* res) {
    __m256 acc[NT];
    float summs[NT];
    for (int t = 0; t < NT; ++t) { acc[t] = _mm256_setzero_ps(); summs[t] = 0.0f; }
    for (int ib = 0; ib < nb; ++ib) {
        const __m256 dx = _mm256_set1_ps(half_at(&x[ib], 0));
        const float xm = half_at(&x[ib], 1);
        __m256i qx = bytes_from_nibbles_32(x[ib].qs);
        __m256i bxhi = bytes_from_bits_32(x[ib].qh);
        bxhi = _mm256_and_si256(bxhi, _mm256_set1_epi8(0x10));
        qx = _mm256_or_si256(qx, bxhi);
        for (int t = 0; t < NT; ++t) {
            const block_q8_1* y = (const block_q8_1*) act[t] + ib;
            summs[t] += xm * half_at(y, 1);
            const __m256 dy = _mm256_set1_ps(half_at(y, 0));
            const __m256i qy = _mm256_loadu_si256((const __m256i*) y->qs);
            const __m256 q = mul_sum_us8_pairs_float(qx, qy);
            acc[t] = _mm256_fmadd_ps(q, _mm256_mul_ps(dx, dy), acc[t]);
        }
    }
    for (int t = 0; t < NT; ++t) res[t] = hsum_float_8(acc[t]) + summs[t];
}

// ---- Q8_0 x Q8_0 (ggml_vec_dot_q8_0_q8_0, __AVX2__; mul_sum_i8_pairs_float without AVX-VNNI-INT8)
template <int NT>
void q8_0_dot(const block_q8_0* x, int nb, const void* const* act, float* res) {
    __m256 acc[NT];
    for (int t = 0; t < NT; ++t) acc[t] = _mm256_setzero_ps();
    for (int ib = 0; ib < nb; ++ib) {
        const float xd = h2f(x[ib].d);
        const __m256i qx = _mm256_loadu_si256((const __m256i*) x[ib].qs);
        const __m256i ax = _mm256_sign_epi8(qx, qx);
        for (int t = 0; t < NT; ++t) {
            const block_q8_0* y = (const block_q8_0*) act[t] + ib;
            const __m256 d = _mm256_set1_ps(xd * h2f(y->d));
            const __m256i qy = _mm256_loadu_si256((const __m256i*) y->qs);
            const __m256i sy = _mm256_sign_epi8(qy, qx);
            const __m256 q = mul_sum_us8_pairs_float(ax, sy);
            acc[t] = _mm256_fmadd_ps(d, q, acc[t]);
        }
    }
    for (int t = 0; t < NT; ++t) res[t] = hsum_float_8(acc[t]);
}

void q5_1_dot_nt(const block_q5_1* x, int nb, const void* const* act, int nt, float* res) {
    for (int t0 = 0; t0 < nt; t0 += 4) {
        const int k = nt - t0 < 4 ? nt - t0 : 4;
        switch (k) {
            case 1: q5_1_dot<1>(x, nb, act + t0, res + t0); break;
            case 2: q5_1_dot<2>(x, nb, act + t0, res + t0); break;
            case 3: q5_1_dot<3>(x, nb, act + t0, res + t0); break;
            default: q5_1_dot<4>(x, nb, act + t0, res + t0); break;
        }
    }
}

void q8_0_dot_nt(const block_q8_0* x, int nb, const void* const* act, int nt, float* res) {
    for (int t0 = 0; t0 < nt; t0 += 4) {
        const int k = nt - t0 < 4 ? nt - t0 : 4;
        switch (k) {
            case 1: q8_0_dot<1>(x, nb, act + t0, res + t0); break;
            case 2: q8_0_dot<2>(x, nb, act + t0, res + t0); break;
            case 3: q8_0_dot<3>(x, nb, act + t0, res + t0); break;
            default: q8_0_dot<4>(x, nb, act + t0, res + t0); break;
        }
    }
}

void dot_rows(int type, const uint8_t* row, int n, const void* const* act, int nt, float* res) {
    switch (type) {
        case 12: q4k_dot((const block_q4_K*) row, n / QK_K, act, nt, res); break;
        case 7: q5_1_dot_nt((const block_q5_1*) row, n / QK5_1, act, nt, res); break;
        case 8: q8_0_dot_nt((const block_q8_0*) row, n / QK8_0, act, nt, res); break;
        default: break;
    }
}

}  // namespace

void bf16_rows_dot_multi(const uint16_t* w, int rows, int cols, const float* x, int nt, float* out) {
    for (int t0 = 0; t0 < nt; t0 += 4) {
        const int m = nt - t0 < 4 ? nt - t0 : 4;
        const float* x0 = x + (size_t) t0 * cols;
        const float* x1 = x0 + cols;
        const float* x2 = x1 + cols;
        const float* x3 = x2 + cols;
        for (int r = 0; r < rows; ++r) {
            const uint16_t* wr = w + (size_t) r * (size_t) cols;
            __m256 a0 = _mm256_setzero_ps(), a1 = _mm256_setzero_ps(), a2 = _mm256_setzero_ps(),
                   a3 = _mm256_setzero_ps();
            for (int c = 0; c + 8 <= cols; c += 8) {
                const __m256 wf = _mm256_castsi256_ps(
                    _mm256_slli_epi32(_mm256_cvtepu16_epi32(_mm_loadu_si128((const __m128i*) (wr + c))), 16));
                a0 = _mm256_fmadd_ps(wf, _mm256_loadu_ps(x0 + c), a0);
                if (m > 1) a1 = _mm256_fmadd_ps(wf, _mm256_loadu_ps(x1 + c), a1);
                if (m > 2) a2 = _mm256_fmadd_ps(wf, _mm256_loadu_ps(x2 + c), a2);
                if (m > 3) a3 = _mm256_fmadd_ps(wf, _mm256_loadu_ps(x3 + c), a3);
            }
            out[(size_t) t0 * rows + r] = hsum_float_8(a0);
            if (m > 1) out[(size_t) (t0 + 1) * rows + r] = hsum_float_8(a1);
            if (m > 2) out[(size_t) (t0 + 2) * rows + r] = hsum_float_8(a2);
            if (m > 3) out[(size_t) (t0 + 3) * rows + r] = hsum_float_8(a3);
        }
    }
}

void bf16_rows_dot(const uint16_t* w, int rows, int cols, const float* x, float* out) {
    for (int r = 0; r < rows; ++r) {
        const uint16_t* wr = w + (size_t) r * (size_t) cols;
        __m256 acc = _mm256_setzero_ps();
        int c = 0;
        for (; c + 8 <= cols; c += 8) {
            const __m256i h = _mm256_cvtepu16_epi32(_mm_loadu_si128((const __m128i*) (wr + c)));
            acc = _mm256_fmadd_ps(_mm256_castsi256_ps(_mm256_slli_epi32(h, 16)), _mm256_loadu_ps(x + c), acc);
        }
        float s = hsum_float_8(acc);
        for (; c < cols; ++c) {
            const uint32_t b = (uint32_t) wr[c] << 16;
            float f;
            std::memcpy(&f, &b, 4);
            s += f * x[c];
        }
        out[r] = s;
    }
}

void kq256_gu_rows(int type, const uint8_t* blob, size_t gu_row, size_t up_off, int n, const void* const* act, int nt,
                   float* const* ff, int r0, int r1) {
    for (int t0 = 0; t0 < nt; t0 += MAXT) {
        const int m = nt - t0 < MAXT ? nt - t0 : MAXT;
        float g[MAXT], u[MAXT];
        for (int r = r0; r < r1; ++r) {
            dot_rows(type, blob + (size_t) r * gu_row, n, act + t0, m, g);
            dot_rows(type, blob + up_off + (size_t) r * gu_row, n, act + t0, m, u);
            // native_gu_rows' own SwiGLU expression: the same bits as the per-token ggml path
            for (int t = 0; t < m; ++t) ff[t0 + t][r] = (g[t] / (1.f + std::exp(-g[t]))) * u[t];
        }
    }
}

// Padé [7/6] rational approximation of tanh(x) for fused SITU activation.
static inline float kq_fast_tanh_f32(float x) noexcept {
    if (x > 5.0f) return 1.0f;
    if (x < -5.0f) return -1.0f;
    const float x2 = x * x;
    const float num = x * (135135.0f + x2 * (17325.0f + x2 * (378.0f + x2)));
    const float den = 135135.0f + x2 * (62370.0f + x2 * (3150.0f + 28.0f * x2));
    return num / den;
}

// Evolved T==1 zero-spill fused gate+up Q4_K x Q8_K:
// Hoists y[i].bsums -> q8s_pre[i] outside the row loop while keeping 1-row register footprint (<= 14 YMM, 0 stack spills).
// Bit-exact (0.00e+00 max diff) against single-row AVX2 dot.
void kq256_gu_t1_q4k(const uint8_t* blob, size_t gu_row, size_t up_off, int n, const void* act0,
                     float* ff0, int r0, int r1, bool situ, float situ_beta, float situ_linear_beta) {
    static const uint32_t kmask1 = 0x3f3f3f3f, kmask2 = 0x0f0f0f0f, kmask3 = 0x03030303;
    const __m256i m4 = _mm256_set1_epi8(0xF);
    const int nb = n / QK_K;
    const block_q8_K* y = (const block_q8_K*) act0;
    const float inv_beta = situ ? (1.0f / situ_beta) : 0.0f;
    const float inv_lb = (situ && situ_linear_beta > 0.f) ? (1.0f / situ_linear_beta) : 0.0f;

    // Precompute per-superblock q8s (depends only on activation y, not on weight row r).
    __m128i q8s_pre[64];
    const int nb_pre = nb <= 64 ? nb : 64;
    for (int i = 0; i < nb_pre; ++i) {
        const __m256i q8sums = _mm256_loadu_si256((const __m256i*) y[i].bsums);
        q8s_pre[i] = _mm_hadd_epi16(_mm256_extracti128_si256(q8sums, 0), _mm256_extracti128_si256(q8sums, 1));
    }

    auto hsum4 = [](__m128 m) {
        m = _mm_add_ps(m, _mm_movehl_ps(m, m));
        m = _mm_add_ss(m, _mm_movehdup_ps(m));
        return _mm_cvtss_f32(m);
    };

    for (int r = r0; r < r1; ++r) {
        const block_q4_K* xg = (const block_q4_K*) (blob + (size_t) r * gu_row);
        const block_q4_K* xu = (const block_q4_K*) (blob + up_off + (size_t) r * gu_row);
        if (r + 2 < r1) {
            _mm_prefetch((const char*) (blob + (size_t) (r + 2) * gu_row), _MM_HINT_T0);
            _mm_prefetch((const char*) (blob + up_off + (size_t) (r + 2) * gu_row), _MM_HINT_T0);
        }
        __m256 acc_g = _mm256_setzero_ps(), acc_u = _mm256_setzero_ps();
        __m128 acc_mg = _mm_setzero_ps(), acc_mu = _mm_setzero_ps();
        for (int i = 0; i < nb; ++i) {
            const float yd = y[i].d;
            __m128i q8s;
            if (i < 64) {
                q8s = q8s_pre[i];
            } else {
                const __m256i q8sums = _mm256_loadu_si256((const __m256i*) y[i].bsums);
                q8s = _mm_hadd_epi16(_mm256_extracti128_si256(q8sums, 0), _mm256_extracti128_si256(q8sums, 1));
            }
            __m256i q8l[4], q8h[4];
            const int8_t* q8 = y[i].qs;
            for (int j = 0; j < 4; ++j) {
                q8l[j] = _mm256_loadu_si256((const __m256i*) q8); q8 += 32;
                q8h[j] = _mm256_loadu_si256((const __m256i*) q8); q8 += 32;
            }
            const block_q4_K* bpair[2] = { &xg[i], &xu[i] };
            __m256* apair[2] = { &acc_g, &acc_u };
            __m128* mpair[2] = { &acc_mg, &acc_mu };
            for (int k = 0; k < 2; ++k) {
                const block_q4_K& bx = *bpair[k];
                const float d = yd * half_at(&bx, 0);
                const float dmin = -yd * half_at(&bx, 1);
                uint32_t utmp[4];
                std::memcpy(utmp, bx.scales, 12);
                utmp[3] = ((utmp[2] >> 4) & kmask2) | (((utmp[1] >> 6) & kmask3) << 4);
                const uint32_t uaux = utmp[1] & kmask1;
                utmp[1] = (utmp[2] & kmask2) | (((utmp[0] >> 6) & kmask3) << 4);
                utmp[2] = uaux;
                utmp[0] &= kmask1;
                const __m256i ms = _mm256_cvtepu8_epi16(_mm_set_epi32((int) utmp[3], (int) utmp[2], (int) utmp[1], (int) utmp[0]));
                const __m128i mins = _mm256_extracti128_si256(ms, 1);
                const __m128i sc128 = _mm256_extracti128_si256(ms, 0);
                const __m256i scales = _mm256_insertf128_si256(_mm256_castsi128_si256(sc128), sc128, 1);
                *mpair[k] = _mm_fmadd_ps(_mm_set1_ps(dmin), _mm_cvtepi32_ps(_mm_madd_epi16(mins, q8s)), *mpair[k]);
                __m256i sumi = _mm256_setzero_si256();
                const uint8_t* q4 = bx.qs;
                for (int j = 0; j < 4; ++j) {
                    const __m256i sl = _mm256_shuffle_epi8(scales, get_scale_shuffle_k4(2 * j + 0));
                    const __m256i sh = _mm256_shuffle_epi8(scales, get_scale_shuffle_k4(2 * j + 1));
                    const __m256i q4b = _mm256_loadu_si256((const __m256i*) q4); q4 += 32;
                    const __m256i p16l = _mm256_madd_epi16(sl, _mm256_maddubs_epi16(_mm256_and_si256(q4b, m4), q8l[j]));
                    const __m256i p16h = _mm256_madd_epi16(sh, _mm256_maddubs_epi16(_mm256_and_si256(_mm256_srli_epi16(q4b, 4), m4), q8h[j]));
                    sumi = _mm256_add_epi32(sumi, _mm256_add_epi32(p16l, p16h));
                }
                *apair[k] = _mm256_fmadd_ps(_mm256_set1_ps(d), _mm256_cvtepi32_ps(sumi), *apair[k]);
            }
        }
        const float gs = hsum_float_8(acc_g) + hsum4(acc_mg);
        const float us = hsum_float_8(acc_u) + hsum4(acc_mu);
        if (situ) {
            const float tg = situ_beta * kq_fast_tanh_f32(gs * inv_beta);
            const float tu = situ_linear_beta > 0.f ? situ_linear_beta * kq_fast_tanh_f32(us * inv_lb) : us;
            const float sig = 0.5f * (1.0f + kq_fast_tanh_f32(0.5f * gs));
            ff0[r] = (tg * sig) * tu;
        } else {
            ff0[r] = (gs / (1.f + std::exp(-gs))) * us;
        }
    }
}

void kq256_rows(int type, const uint8_t* w, size_t row_bytes, int n, const void* const* act, int nt, float* const* out,
                int r0, int r1) {
    for (int t0 = 0; t0 < nt; t0 += MAXT) {
        const int m = nt - t0 < MAXT ? nt - t0 : MAXT;
        float s[MAXT];
        for (int r = r0; r < r1; ++r) {
            dot_rows(type, w + (size_t) r * row_bytes, n, act + t0, m, s);
            for (int t = 0; t < m; ++t) out[t0 + t][r] = s[t];
        }
    }
}

// Evolved Gen-3 2-Row Unrolled Algebraic Unsigned Q5_0 x Q8_0 (T==1):
// sum((q_i - 16) * y_i) = sum(q_i * y_i) - 16 * sum(y_i), sharing Q8_0 activation loads across 2 rows.
void kq256_down_t1_q5_0(const uint8_t* w, size_t row_bytes, int n, const void* hq0,
                        float* out0, int r0, int r1) {
    const int nb = n / QK5_0;
    const block_q8_0* y = (const block_q8_0*) hq0;
    float yd[256];
    float ysum16[256];
    const int lim = nb < 256 ? nb : 256;
    for (int ib = 0; ib < lim; ++ib) {
        const float d = h2f(y[ib].d);
        yd[ib] = d;
        int s = 0;
        for (int i = 0; i < 32; ++i) s += (int) y[ib].qs[i];
        ysum16[ib] = -16.0f * d * (float) s;
    }
    const __m256i m16 = _mm256_set1_epi8(0x10);
    int r = r0;
    for (; r + 1 < r1; r += 2) {
        const block_q5_0* x0 = (const block_q5_0*) (w + (size_t) r * row_bytes);
        const block_q5_0* x1 = (const block_q5_0*) (w + (size_t) (r + 1) * row_bytes);
        if (r + 3 < r1) {
            _mm_prefetch((const char*) (w + (size_t) (r + 2) * row_bytes), _MM_HINT_T0);
            _mm_prefetch((const char*) (w + (size_t) (r + 3) * row_bytes), _MM_HINT_T0);
        }
        __m256 acc0 = _mm256_setzero_ps(), acc1 = _mm256_setzero_ps();
        float bias0 = 0.0f, bias1 = 0.0f;
        for (int ib = 0; ib < lim; ++ib) {
            const float ydi = yd[ib];
            const float ysi = ysum16[ib];
            const __m256i qy = _mm256_loadu_si256((const __m256i*) y[ib].qs);

            const float dx0 = h2f(x0[ib].d);
            bias0 = std::fmaf(dx0, ysi, bias0);
            __m256i qx0 = _mm256_or_si256(bytes_from_nibbles_32(x0[ib].qs),
                                          _mm256_and_si256(bytes_from_bits_32(x0[ib].qh), m16));
            acc0 = _mm256_fmadd_ps(_mm256_set1_ps(dx0 * ydi), mul_sum_us8_pairs_float(qx0, qy), acc0);

            const float dx1 = h2f(x1[ib].d);
            bias1 = std::fmaf(dx1, ysi, bias1);
            __m256i qx1 = _mm256_or_si256(bytes_from_nibbles_32(x1[ib].qs),
                                          _mm256_and_si256(bytes_from_bits_32(x1[ib].qh), m16));
            acc1 = _mm256_fmadd_ps(_mm256_set1_ps(dx1 * ydi), mul_sum_us8_pairs_float(qx1, qy), acc1);
        }
        out0[r]     = hsum_float_8(acc0) + bias0;
        out0[r + 1] = hsum_float_8(acc1) + bias1;
    }
    for (; r < r1; ++r) {
        const block_q5_0* x = (const block_q5_0*) (w + (size_t) r * row_bytes);
        __m256 acc = _mm256_setzero_ps();
        float bias = 0.0f;
        for (int ib = 0; ib < lim; ++ib) {
            const float dx = h2f(x[ib].d);
            bias = std::fmaf(dx, ysum16[ib], bias);
            __m256i qx = _mm256_or_si256(bytes_from_nibbles_32(x[ib].qs),
                                         _mm256_and_si256(bytes_from_bits_32(x[ib].qh), m16));
            const __m256i qy = _mm256_loadu_si256((const __m256i*) y[ib].qs);
            acc = _mm256_fmadd_ps(_mm256_set1_ps(dx * yd[ib]), mul_sum_us8_pairs_float(qx, qy), acc);
        }
        out0[r] = hsum_float_8(acc) + bias;
    }
}

// 2-Row Unrolled Q8_0 x Q8_0 (T==1): precomputes yd[ib] once and shares qy across 2 weight rows.
// Bit-exact against ggml_vec_dot_q8_0_q8_0 (__AVX2__).
void kq256_down_t1_q8_0(const uint8_t* w, size_t row_bytes, int n, const void* hq0,
                        float* out0, int r0, int r1) {
    const int nb = n / QK8_0;
    const block_q8_0* y = (const block_q8_0*) hq0;
    float yd[256];
    const int lim = nb < 256 ? nb : 256;
    for (int ib = 0; ib < lim; ++ib) yd[ib] = h2f(y[ib].d);
    int r = r0;
    for (; r + 1 < r1; r += 2) {
        const block_q8_0* x0 = (const block_q8_0*) (w + (size_t) r * row_bytes);
        const block_q8_0* x1 = (const block_q8_0*) (w + (size_t) (r + 1) * row_bytes);
        if (r + 3 < r1) {
            _mm_prefetch((const char*) (w + (size_t) (r + 2) * row_bytes), _MM_HINT_T0);
            _mm_prefetch((const char*) (w + (size_t) (r + 3) * row_bytes), _MM_HINT_T0);
        }
        __m256 acc0 = _mm256_setzero_ps(), acc1 = _mm256_setzero_ps();
        for (int ib = 0; ib < lim; ++ib) {
            const float ydi = yd[ib];
            const __m256i qy = _mm256_loadu_si256((const __m256i*) y[ib].qs);

            const __m256i qx0 = _mm256_loadu_si256((const __m256i*) x0[ib].qs);
            const __m256i ax0 = _mm256_sign_epi8(qx0, qx0);
            const __m256i sy0 = _mm256_sign_epi8(qy, qx0);
            acc0 = _mm256_fmadd_ps(_mm256_set1_ps(h2f(x0[ib].d) * ydi), mul_sum_us8_pairs_float(ax0, sy0), acc0);

            const __m256i qx1 = _mm256_loadu_si256((const __m256i*) x1[ib].qs);
            const __m256i ax1 = _mm256_sign_epi8(qx1, qx1);
            const __m256i sy1 = _mm256_sign_epi8(qy, qx1);
            acc1 = _mm256_fmadd_ps(_mm256_set1_ps(h2f(x1[ib].d) * ydi), mul_sum_us8_pairs_float(ax1, sy1), acc1);
        }
        out0[r]     = hsum_float_8(acc0);
        out0[r + 1] = hsum_float_8(acc1);
    }
    for (; r < r1; ++r) {
        const block_q8_0* x = (const block_q8_0*) (w + (size_t) r * row_bytes);
        __m256 acc = _mm256_setzero_ps();
        for (int ib = 0; ib < lim; ++ib) {
            const __m256i qx = _mm256_loadu_si256((const __m256i*) x[ib].qs);
            const __m256i qy = _mm256_loadu_si256((const __m256i*) y[ib].qs);
            const __m256i ax = _mm256_sign_epi8(qx, qx);
            const __m256i sy = _mm256_sign_epi8(qy, qx);
            acc = _mm256_fmadd_ps(_mm256_set1_ps(h2f(x[ib].d) * yd[ib]), mul_sum_us8_pairs_float(ax, sy), acc);
        }
        out0[r] = hsum_float_8(acc);
    }
}

}  // namespace strata::kernels::cpu
