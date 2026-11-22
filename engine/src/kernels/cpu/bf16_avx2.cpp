// src/kernels/cpu/bf16_avx2.cpp - multi-token BF16 weight x F32 activation expert rows (AVX2 + FMA).
#include "strata/kernels/cpu/bf16_avx2.hpp"

#include <immintrin.h>

#include <cmath>
#include <cstdlib>

namespace strata::kernels::cpu {
namespace {

inline int bf16_prefetch_distance() {
    static const int v = [] {
        const char* e = std::getenv("STRATA_BF16_PREFETCH");
        return e ? std::atoi(e) : 2048;
    }();
    return v;
}

inline float hsum8(const __m256 x) {
    __m128 res = _mm256_extractf128_ps(x, 1);
    res = _mm_add_ps(res, _mm256_castps256_ps128(x));
    res = _mm_add_ps(res, _mm_movehl_ps(res, res));
    res = _mm_add_ss(res, _mm_movehdup_ps(res));
    return _mm_cvtss_f32(res);
}

constexpr int kMaxW = 8;

template <int NT>
inline void row_dot(const uint8_t* row, const float* const* act, int n, float* res) {
    __m256 acc0[NT], acc1[NT];
    for (int t = 0; t < NT; ++t) {
        acc0[t] = _mm256_setzero_ps();
        acc1[t] = _mm256_setzero_ps();
    }
    const int pf = bf16_prefetch_distance();
    int k = 0;
    for (; k + 16 <= n; k += 16) {
        const uint16_t* wr = (const uint16_t*) row + k;
        if (pf > 0) _mm_prefetch((const char*) wr + pf, _MM_HINT_T0);
        const __m128i w0_16 = _mm_loadu_si128((const __m128i*) wr);
        const __m128i w1_16 = _mm_loadu_si128((const __m128i*) (wr + 8));
        const __m256 w0 = _mm256_castsi256_ps(_mm256_slli_epi32(_mm256_cvtepu16_epi32(w0_16), 16));
        const __m256 w1 = _mm256_castsi256_ps(_mm256_slli_epi32(_mm256_cvtepu16_epi32(w1_16), 16));
        for (int t = 0; t < NT; ++t) {
            const float* a = act[t] + k;
            acc0[t] = _mm256_fmadd_ps(w0, _mm256_loadu_ps(a), acc0[t]);
            acc1[t] = _mm256_fmadd_ps(w1, _mm256_loadu_ps(a + 8), acc1[t]);
        }
    }
    for (int t = 0; t < NT; ++t) res[t] = hsum8(_mm256_add_ps(acc0[t], acc1[t]));
}

template <int NT, bool SITU>
void gu_rows(const uint8_t* blob, size_t gu_row, size_t up_off, int n, const float* const* act, float* const* ff,
             int r0, int r1, float beta, float lb) {
    float g[NT], u[NT];
    for (int r = r0; r < r1; ++r) {
        row_dot<NT>(blob + (size_t) r * gu_row, act, n, g);
        row_dot<NT>(blob + up_off + (size_t) r * gu_row, act, n, u);
        for (int t = 0; t < NT; ++t) {
            if constexpr (SITU) {
                const float tg = beta * std::tanh(g[t] / beta);
                const float tu = lb > 0.f ? lb * std::tanh(u[t] / lb) : u[t];
                ff[t][r] = (tg / (1.f + std::exp(-g[t]))) * tu;
            } else {
                ff[t][r] = (g[t] / (1.f + std::exp(-g[t]))) * u[t];
            }
        }
    }
}

template <int NT>
void dot_rows(const uint8_t* w, size_t row_bytes, int n, const float* const* act, float* const* out, int r0, int r1) {
    float res[NT];
    for (int r = r0; r < r1; ++r) {
        row_dot<NT>(w + (size_t) r * row_bytes, act, n, res);
        for (int t = 0; t < NT; ++t) out[t][r] = res[t];
    }
}

template <typename F>
inline void slices(int nt, F&& run) {
    const int k = (nt + kMaxW - 1) / kMaxW;
    for (int i = 0, t0 = 0; i < k; ++i) {
        const int w = nt / k + (i < nt % k ? 1 : 0);
        run(t0, w);
        t0 += w;
    }
}

}  // namespace

bool bf16_256_fits(int n) { return n > 0 && (n % 16) == 0; }

void bf16_256_gu_rows(const uint8_t* blob, size_t gu_row, size_t up_off, int n, const void* const* act, int nt,
                      float* const* ff, int r0, int r1, bool situ, float beta, float lb) {
    slices(nt, [&](int t0, int w) {
        const float* const* a = reinterpret_cast<const float* const*>(act + t0);
        float* const* f = ff + t0;
        if (situ) {
            switch (w) {
                case 1: gu_rows<1, true>(blob, gu_row, up_off, n, a, f, r0, r1, beta, lb); break;
                case 2: gu_rows<2, true>(blob, gu_row, up_off, n, a, f, r0, r1, beta, lb); break;
                case 3: gu_rows<3, true>(blob, gu_row, up_off, n, a, f, r0, r1, beta, lb); break;
                case 4: gu_rows<4, true>(blob, gu_row, up_off, n, a, f, r0, r1, beta, lb); break;
                case 5: gu_rows<5, true>(blob, gu_row, up_off, n, a, f, r0, r1, beta, lb); break;
                case 6: gu_rows<6, true>(blob, gu_row, up_off, n, a, f, r0, r1, beta, lb); break;
                case 7: gu_rows<7, true>(blob, gu_row, up_off, n, a, f, r0, r1, beta, lb); break;
                default: gu_rows<8, true>(blob, gu_row, up_off, n, a, f, r0, r1, beta, lb); break;
            }
        } else {
            switch (w) {
                case 1: gu_rows<1, false>(blob, gu_row, up_off, n, a, f, r0, r1, beta, lb); break;
                case 2: gu_rows<2, false>(blob, gu_row, up_off, n, a, f, r0, r1, beta, lb); break;
                case 3: gu_rows<3, false>(blob, gu_row, up_off, n, a, f, r0, r1, beta, lb); break;
                case 4: gu_rows<4, false>(blob, gu_row, up_off, n, a, f, r0, r1, beta, lb); break;
                case 5: gu_rows<5, false>(blob, gu_row, up_off, n, a, f, r0, r1, beta, lb); break;
                case 6: gu_rows<6, false>(blob, gu_row, up_off, n, a, f, r0, r1, beta, lb); break;
                case 7: gu_rows<7, false>(blob, gu_row, up_off, n, a, f, r0, r1, beta, lb); break;
                default: gu_rows<8, false>(blob, gu_row, up_off, n, a, f, r0, r1, beta, lb); break;
            }
        }
    });
}

void bf16_256_rows(const uint8_t* w, size_t row_bytes, int n, const void* const* act, int nt, float* const* out,
                   int r0, int r1) {
    slices(nt, [&](int t0, int k) {
        const float* const* a = reinterpret_cast<const float* const*>(act + t0);
        float* const* o = out + t0;
        switch (k) {
            case 1: dot_rows<1>(w, row_bytes, n, a, o, r0, r1); break;
            case 2: dot_rows<2>(w, row_bytes, n, a, o, r0, r1); break;
            case 3: dot_rows<3>(w, row_bytes, n, a, o, r0, r1); break;
            case 4: dot_rows<4>(w, row_bytes, n, a, o, r0, r1); break;
            case 5: dot_rows<5>(w, row_bytes, n, a, o, r0, r1); break;
            case 6: dot_rows<6>(w, row_bytes, n, a, o, r0, r1); break;
            case 7: dot_rows<7>(w, row_bytes, n, a, o, r0, r1); break;
            default: dot_rows<8>(w, row_bytes, n, a, o, r0, r1); break;
        }
    });
}

}  // namespace strata::kernels::cpu
