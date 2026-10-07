// src/glm/kernels.cpp - the CPU kernels of the GLM-5.3 forward pass.  See the header for the formats.
//
// Built with AVX2 + FMA (the engine's floor, as the rest of Strata); the scalar `*_ref` functions are the
// definitions the tests compare against.
#include "strata/glm/kernels.hpp"

#include <immintrin.h>

#include <algorithm>
#include <cmath>
#include <cstring>

#if defined(_MSC_VER)
#include <intrin.h>
#endif

namespace strata::glm {
namespace {

inline float hsum(__m256 v) {
    __m128 lo = _mm256_castps256_ps128(v), hi = _mm256_extractf128_ps(v, 1);
    lo = _mm_add_ps(lo, hi);
    __m128 sh = _mm_movehdup_ps(lo);
    lo = _mm_add_ps(lo, sh);
    sh = _mm_movehl_ps(sh, lo);
    return _mm_cvtss_f32(_mm_add_ss(lo, sh));
}

/// The 32 code bytes of one group as 8 x 8 floats in `Act::xp` order: w[0..3] are the low nibbles (even elements
/// 0, 2, .., 62), w[4..7] the high nibbles (odd elements).  Codes, not values: the -8 is folded out by the caller.
inline void unpack_group(const uint8_t* c, __m256 w[8]) {
    const __m256i b = _mm256_loadu_si256((const __m256i*) c);
    const __m256i m = _mm256_set1_epi8(0x0F);
    const __m256i lo = _mm256_and_si256(b, m);
    const __m256i hi = _mm256_and_si256(_mm256_srli_epi16(b, 4), m);
    const __m128i l0 = _mm256_castsi256_si128(lo), l1 = _mm256_extracti128_si256(lo, 1);
    const __m128i h0 = _mm256_castsi256_si128(hi), h1 = _mm256_extracti128_si256(hi, 1);
    w[0] = _mm256_cvtepi32_ps(_mm256_cvtepu8_epi32(l0));
    w[1] = _mm256_cvtepi32_ps(_mm256_cvtepu8_epi32(_mm_srli_si128(l0, 8)));
    w[2] = _mm256_cvtepi32_ps(_mm256_cvtepu8_epi32(l1));
    w[3] = _mm256_cvtepi32_ps(_mm256_cvtepu8_epi32(_mm_srli_si128(l1, 8)));
    w[4] = _mm256_cvtepi32_ps(_mm256_cvtepu8_epi32(h0));
    w[5] = _mm256_cvtepi32_ps(_mm256_cvtepu8_epi32(_mm_srli_si128(h0, 8)));
    w[6] = _mm256_cvtepi32_ps(_mm256_cvtepu8_epi32(h1));
    w[7] = _mm256_cvtepi32_ps(_mm256_cvtepu8_epi32(_mm_srli_si128(h1, 8)));
}

inline __m256 group_dot(const __m256 w[8], const float* xp) {
    __m256 a0 = _mm256_mul_ps(w[0], _mm256_loadu_ps(xp + 0));
    __m256 a1 = _mm256_mul_ps(w[1], _mm256_loadu_ps(xp + 8));
    a0 = _mm256_fmadd_ps(w[2], _mm256_loadu_ps(xp + 16), a0);
    a1 = _mm256_fmadd_ps(w[3], _mm256_loadu_ps(xp + 24), a1);
    a0 = _mm256_fmadd_ps(w[4], _mm256_loadu_ps(xp + 32), a0);
    a1 = _mm256_fmadd_ps(w[5], _mm256_loadu_ps(xp + 40), a1);
    a0 = _mm256_fmadd_ps(w[6], _mm256_loadu_ps(xp + 48), a0);
    a1 = _mm256_fmadd_ps(w[7], _mm256_loadu_ps(xp + 56), a1);
    return _mm256_add_ps(a0, a1);
}

}  // namespace

bool cpu_has_avx2() {
#if defined(_MSC_VER)
    int r[4];
    __cpuid(r, 0);
    if (r[0] < 7) return false;
    __cpuidex(r, 7, 0);
    const bool avx2 = (r[1] & (1 << 5)) != 0;
    __cpuid(r, 1);
    const bool fma = (r[2] & (1 << 12)) != 0;
    return avx2 && fma;
#else
    return __builtin_cpu_supports("avx2") && __builtin_cpu_supports("fma");
#endif
}

void Act::prepare(const float* x, int S_, int I_) {
    S = S_;
    I = I_;
    const int ng = I / kGroup;
    xp.resize((size_t) S * I);
    gsum.resize((size_t) S * ng);
    for (int s = 0; s < S; ++s) {
        const float* xs = x + (size_t) s * I;
        float* d = xp.data() + (size_t) s * I;
        for (int g = 0; g < ng; ++g) {
            const float* xg = xs + g * kGroup;
            float* dg = d + g * kGroup;
            float sum = 0.0f;
            for (int j = 0; j < kGroup / 2; ++j) {
                dg[j] = xg[2 * j];
                dg[kGroup / 2 + j] = xg[2 * j + 1];
                sum += xg[2 * j] + xg[2 * j + 1];
            }
            gsum[(size_t) s * ng + g] = sum;
        }
    }
}

void q4_rows(const Q4& W, const Act& a, int r0, int r1, float* y, int ldy, int ybase) {
    const int ng = W.I / kGroup;
    const int S = a.S;
    if (S == 1) {
        const float* xp = a.xp.data();
        const float* gs = a.gsum.data();
        for (int r = r0; r < r1; ++r) {
            const uint8_t* c = W.row_codes(r);
            const float* sc = W.row_scales(r);
            __m256 acc = _mm256_setzero_ps();
            float bias = 0.0f;
            for (int g = 0; g < ng; ++g) {
                __m256 w[8];
                unpack_group(c + g * 32, w);
                acc = _mm256_fmadd_ps(group_dot(w, xp + g * kGroup), _mm256_set1_ps(sc[g]), acc);
                bias += sc[g] * gs[g];
            }
            y[r - ybase] = hsum(acc) - 8.0f * bias;
        }
        return;
    }
    // several activations: two rows at a time are unpacked once into exact float weights ((code - 8) * scale, in
    // the activation's permuted order), then 2 rows x 4 activations run with 8 independent accumulators.  That fits
    // AVX2's 16 registers (an earlier 16-activation version spilled its accumulators) and costs 6 loads per 8 FMAs.
    static thread_local std::vector<float> wbuf;
    wbuf.resize((size_t) 2 * W.I);
    const __m256 eight = _mm256_set1_ps(8.0f);
    for (int r = r0; r < r1; r += 2) {
        const int nrow = std::min(2, r1 - r);
        for (int j = 0; j < 2; ++j) {
            const int rr = r + std::min(j, nrow - 1);   // an odd last row is unpacked twice (its copy is unused)
            const uint8_t* c = W.row_codes(rr);
            const float* sc = W.row_scales(rr);
            float* wd = wbuf.data() + (size_t) j * W.I;
            for (int g = 0; g < ng; ++g) {
                __m256 w[8];
                unpack_group(c + g * 32, w);
                const __m256 s8 = _mm256_set1_ps(sc[g]);
                for (int q = 0; q < 8; ++q) _mm256_storeu_ps(wd + g * kGroup + 8 * q, _mm256_mul_ps(_mm256_sub_ps(w[q], eight), s8));
            }
        }
        const float* w0 = wbuf.data();
        const float* w1 = wbuf.data() + W.I;
        for (int s0 = 0; s0 < S; s0 += 4) {
            const int sn = std::min(4, S - s0);
            const float* xs[4];
            for (int k = 0; k < 4; ++k) xs[k] = a.xp.data() + (size_t) (s0 + std::min(k, sn - 1)) * W.I;
            __m256 a0[4], a1[4];
            for (int k = 0; k < 4; ++k) a0[k] = a1[k] = _mm256_setzero_ps();
            for (int i = 0; i < W.I; i += 8) {
                const __m256 v0 = _mm256_loadu_ps(w0 + i), v1 = _mm256_loadu_ps(w1 + i);
                for (int k = 0; k < 4; ++k) {
                    const __m256 x = _mm256_loadu_ps(xs[k] + i);
                    a0[k] = _mm256_fmadd_ps(v0, x, a0[k]);
                    a1[k] = _mm256_fmadd_ps(v1, x, a1[k]);
                }
            }
            for (int k = 0; k < sn; ++k) {
                y[(size_t) (s0 + k) * ldy + (r - ybase)] = hsum(a0[k]);
                if (nrow > 1) y[(size_t) (s0 + k) * ldy + (r + 1 - ybase)] = hsum(a1[k]);
            }
        }
    }
}

void q4_gemm(Pool& pool, const Q4& W, const Act& a, float* y) {
    const int64_t grain = std::max<int64_t>(4, W.O / (int64_t) (pool.size() * 8));
    pool.parallel_for(W.O, grain, [&](int64_t b, int64_t e) { q4_rows(W, a, (int) b, (int) e, y, W.O); });
}

void q4_gemm(Pool& pool, const Q4& W, const float* x, int S, float* y) {
    static thread_local Act a;
    a.prepare(x, S, W.I);
    q4_gemm(pool, W, a, y);
}

float dot_f32(const float* a, const float* b, int n) {
    __m256 a0 = _mm256_setzero_ps(), a1 = _mm256_setzero_ps();
    int i = 0;
    for (; i + 16 <= n; i += 16) {
        a0 = _mm256_fmadd_ps(_mm256_loadu_ps(a + i), _mm256_loadu_ps(b + i), a0);
        a1 = _mm256_fmadd_ps(_mm256_loadu_ps(a + i + 8), _mm256_loadu_ps(b + i + 8), a1);
    }
    float s = hsum(_mm256_add_ps(a0, a1));
    for (; i < n; ++i) s += a[i] * b[i];
    return s;
}

void axpy_f32(float* y, float a, const float* x, int n) {
    const __m256 a8 = _mm256_set1_ps(a);
    int i = 0;
    for (; i + 8 <= n; i += 8) _mm256_storeu_ps(y + i, _mm256_fmadd_ps(a8, _mm256_loadu_ps(x + i), _mm256_loadu_ps(y + i)));
    for (; i < n; ++i) y[i] += a * x[i];
}

void q4_rows_t(const Q4& W, int r0, int n, const float* coef, float* out) {
    const int ng = W.I / kGroup;
    static thread_local std::vector<float> tmp;
    tmp.assign(W.I, 0.0f);
    float bias[64] = {0};   // per group: sum_r coef * scale (the -8 term), I <= 4096
    for (int k = 0; k < n; ++k) {
        const uint8_t* c = W.row_codes(r0 + k);
        const float* sc = W.row_scales(r0 + k);
        for (int g = 0; g < ng; ++g) {
            const float f = coef[k] * sc[g];
            if (f == 0.0f) continue;
            bias[g] += f;
            __m256 w[8];
            unpack_group(c + g * 32, w);
            const __m256 f8 = _mm256_set1_ps(f);
            float* t = tmp.data() + g * kGroup;
            for (int q = 0; q < 8; ++q) _mm256_storeu_ps(t + 8 * q, _mm256_fmadd_ps(w[q], f8, _mm256_loadu_ps(t + 8 * q)));
        }
    }
    for (int g = 0; g < ng; ++g) {
        const float* t = tmp.data() + g * kGroup;
        float* o = out + g * kGroup;
        const float b = 8.0f * bias[g];
        for (int j = 0; j < kGroup / 2; ++j) {
            o[2 * j] += t[j] - b;
            o[2 * j + 1] += t[kGroup / 2 + j] - b;
        }
    }
}

void q4_dequant_rows(const Q4& W, int r0, int n, float* out) {
    const int ng = W.I / kGroup;
    for (int k = 0; k < n; ++k) {
        const uint8_t* c = W.row_codes(r0 + k);
        const float* sc = W.row_scales(r0 + k);
        float* o = out + (size_t) k * W.I;
        for (int g = 0; g < ng; ++g) {
            const float s = sc[g];
            for (int j = 0; j < kGroup / 2; ++j) {
                const uint8_t b = c[g * 32 + j];
                o[g * kGroup + 2 * j] = (float) ((int) (b & 15) - 8) * s;
                o[g * kGroup + 2 * j + 1] = (float) ((int) (b >> 4) - 8) * s;
            }
        }
    }
}

void mla_head_prompt(const float* Wk, const float* Wv, const float* q, int qstride, const float* kv, int kvl, int nr,
                     int nope, int vh, int S, int pos0, float scale, float* ctx, int cstride) {
    constexpr int B = 4;
    const int row = kvl + nr;
    static thread_local std::vector<float> qabs, sc, clat;
    qabs.resize((size_t) B * kvl);
    clat.resize((size_t) B * kvl);
    for (int s0 = 0; s0 < S; s0 += B) {
        const int m = std::min(B, S - s0);
        const float* qs[B];
        int nt[B];
        for (int i = 0; i < B; ++i) {
            const int ii = std::min(i, m - 1);   // a short last block repeats its last query (results unused)
            qs[i] = q + (size_t) (s0 + ii) * qstride;
            nt[i] = pos0 + s0 + ii + 1;
        }
        const int ntmax = nt[m - 1];
        // qabs_i = W_k^T q_nope,i: each Wk row is loaded once for the 4 queries
        for (int c = 0; c < kvl; c += 8) {
            __m256 a[B] = {_mm256_setzero_ps(), _mm256_setzero_ps(), _mm256_setzero_ps(), _mm256_setzero_ps()};
            for (int d = 0; d < nope; ++d) {
                const __m256 w = _mm256_loadu_ps(Wk + (size_t) d * kvl + c);
                for (int i = 0; i < B; ++i) a[i] = _mm256_fmadd_ps(_mm256_set1_ps(qs[i][d]), w, a[i]);
            }
            for (int i = 0; i < B; ++i) _mm256_storeu_ps(&qabs[(size_t) i * kvl + c], a[i]);
        }
        // scores: each cache row is loaded once for the 4 queries
        sc.resize((size_t) B * ntmax);
        for (int t = 0; t < ntmax; ++t) {
            const float* k = kv + (size_t) t * row;
            __m256 a[B] = {_mm256_setzero_ps(), _mm256_setzero_ps(), _mm256_setzero_ps(), _mm256_setzero_ps()};
            for (int c = 0; c < kvl; c += 8) {
                const __m256 kk = _mm256_loadu_ps(k + c);
                for (int i = 0; i < B; ++i) a[i] = _mm256_fmadd_ps(_mm256_loadu_ps(&qabs[(size_t) i * kvl + c]), kk, a[i]);
            }
            for (int c = 0; c < nr; c += 8) {
                const __m256 kk = _mm256_loadu_ps(k + kvl + c);
                for (int i = 0; i < B; ++i) a[i] = _mm256_fmadd_ps(_mm256_loadu_ps(qs[i] + nope + c), kk, a[i]);
            }
            for (int i = 0; i < B; ++i) sc[(size_t) i * ntmax + t] = hsum(a[i]) * scale;
        }
        for (int i = 0; i < B; ++i) {
            float* p = &sc[(size_t) i * ntmax];
            softmax_inplace(p, nt[i]);
            for (int t = nt[i]; t < ntmax; ++t) p[t] = 0.0f;   // causal: later rows do not exist for query i
        }
        // clat_i = sum_t p_i[t] L_t, 16 latent columns at a time
        for (int c = 0; c < kvl; c += 16) {
            __m256 a[B][2];
            for (int i = 0; i < B; ++i) a[i][0] = a[i][1] = _mm256_setzero_ps();
            for (int t = 0; t < ntmax; ++t) {
                const float* l = kv + (size_t) t * row + c;
                const __m256 l0 = _mm256_loadu_ps(l), l1 = _mm256_loadu_ps(l + 8);
                for (int i = 0; i < B; ++i) {
                    const __m256 p = _mm256_set1_ps(sc[(size_t) i * ntmax + t]);
                    a[i][0] = _mm256_fmadd_ps(p, l0, a[i][0]);
                    a[i][1] = _mm256_fmadd_ps(p, l1, a[i][1]);
                }
            }
            for (int i = 0; i < B; ++i) {
                _mm256_storeu_ps(&clat[(size_t) i * kvl + c], a[i][0]);
                _mm256_storeu_ps(&clat[(size_t) i * kvl + c + 8], a[i][1]);
            }
        }
        // ctx_i = W_v clat_i: each Wv row is loaded once for the 4 queries
        for (int r = 0; r < vh; ++r) {
            const float* w = Wv + (size_t) r * kvl;
            __m256 a[B] = {_mm256_setzero_ps(), _mm256_setzero_ps(), _mm256_setzero_ps(), _mm256_setzero_ps()};
            for (int c = 0; c < kvl; c += 8) {
                const __m256 ww = _mm256_loadu_ps(w + c);
                for (int i = 0; i < B; ++i) a[i] = _mm256_fmadd_ps(ww, _mm256_loadu_ps(&clat[(size_t) i * kvl + c]), a[i]);
            }
            for (int i = 0; i < m; ++i) ctx[(size_t) (s0 + i) * cstride + r] = hsum(a[i]);
        }
    }
}

void q8r_gemv(Pool& pool, const Q8R& W, const float* x, float* y) {
    pool.parallel_for(W.O, 256, [&](int64_t b, int64_t e) {
        for (int64_t r = b; r < e; ++r) {
            const uint8_t* c = W.codes + (size_t) r * W.I;
            __m256 a0 = _mm256_setzero_ps(), a1 = _mm256_setzero_ps();
            int i = 0;
            for (; i + 16 <= W.I; i += 16) {
                const __m128i v = _mm_loadu_si128((const __m128i*) (c + i));
                const __m256 w0 = _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(v));
                const __m256 w1 = _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(_mm_srli_si128(v, 8)));
                a0 = _mm256_fmadd_ps(w0, _mm256_loadu_ps(x + i), a0);
                a1 = _mm256_fmadd_ps(w1, _mm256_loadu_ps(x + i + 8), a1);
            }
            float s = hsum(_mm256_add_ps(a0, a1));
            for (; i < W.I; ++i) s += (float) (int8_t) c[i] * x[i];
            y[r] = W.scales[r] * s;
        }
    });
}

void q8r_row(const Q8R& W, int r, float* x) {
    const uint8_t* c = W.codes + (size_t) r * W.I;
    const float s = W.scales[r];
    for (int i = 0; i < W.I; ++i) x[i] = (float) (int8_t) c[i] * s;
}

float q4_weight(const Q4& W, int r, int i) {
    const uint8_t byte = W.row_codes(r)[i / 2];
    const int code = (i & 1) ? (byte >> 4) : (byte & 0x0F);
    return (float) (code - 8) * W.row_scales(r)[i / kGroup];
}

void q4_rows_ref(const Q4& W, const float* x, int S, int r0, int r1, float* y, int ldy) {
    for (int s = 0; s < S; ++s)
        for (int r = r0; r < r1; ++r) {
            double acc = 0.0;
            for (int i = 0; i < W.I; ++i) acc += (double) q4_weight(W, r, i) * x[(size_t) s * W.I + i];
            y[(size_t) s * ldy + r] = (float) acc;
        }
}

void rmsnorm(float* out, const float* x, const float* w, int n, float eps) {
    double ms = 0.0;
    for (int i = 0; i < n; ++i) ms += (double) x[i] * x[i];
    const float r = 1.0f / std::sqrt((float) (ms / n) + eps);
    for (int i = 0; i < n; ++i) out[i] = x[i] * r * w[i];
}

void softmax_inplace(float* x, int n) {
    float m = -1e30f;
    for (int i = 0; i < n; ++i) m = std::max(m, x[i]);
    float s = 0.0f;
    for (int i = 0; i < n; ++i) { x[i] = std::exp(x[i] - m); s += x[i]; }
    const float inv = 1.0f / s;
    for (int i = 0; i < n; ++i) x[i] *= inv;
}

void Rope::init(int n_, double theta) {
    n = n_;
    inv.resize(n / 2);
    for (int j = 0; j < n / 2; ++j) inv[j] = std::pow((float) theta, -2.0f * (float) j / (float) n);
}

void Rope::apply(float* v, int pos) const {
    const int half = n / 2;
    float in[512];
    std::memcpy(in, v, (size_t) n * sizeof(float));
    for (int j = 0; j < half; ++j) {
        const float ang = (float) pos * inv[j];
        const float cs = std::cos(ang), sn = std::sin(ang);
        const float a = in[2 * j], b = in[2 * j + 1];
        v[j] = a * cs - b * sn;
        v[half + j] = b * cs + a * sn;
    }
}

}  // namespace strata::glm
