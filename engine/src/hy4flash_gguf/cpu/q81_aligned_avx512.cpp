#include "q81_aligned.hpp"
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <immintrin.h>
namespace strata::hy4_flash::cpu::q81_aligned {
namespace {
using B = uint8_t;
using I = __m512i;
using F = __m512;
constexpr int N = mantissa_count;
static unsigned bits(float x) {
    unsigned u;
    memcpy(&u, &x, 4);
    return u;
}
static float fp(unsigned u) {
    float x;
    memcpy(&x, &u, 4);
    return x;
}
static float recip(const uint32_t* table, float x) {
    unsigned u = bits(x);
    return fp(table[2 * N + (u & 0x7fffff)] - ((u & 0x7f800000) - 0x3f800000));
}
static float expgpu(const uint32_t* table, float g) {
    float x = g * 1.4426950408889634f;
    int e = (int)x;
    float frac = x - e;
    unsigned u = bits(frac);
    unsigned key = (u >> 31) ? ~u : (u ^ 0x80000000);
    unsigned i = std::min(2 * N - 1u, bits(std::exp2(frac)) - 0x3f000000);
    while (i && key < table[i])
        --i;
    while (i + 1 < 2 * N && key >= table[i + 1])
        ++i;
    return fp(0x3f000000 + i + (unsigned(e) << 23));
}
static void scalar_activate(const uint32_t* table, const float* g, const float* u, float* h, int n) {
    for (int i = 0; i < n; i++) {
        float a = std::min(g[i], 10.f), b = std::clamp(u[i], -10.f, 10.f);
        float e = (-a * 1.44269504f >= 128.f) ? INFINITY : expgpu(table, -a);
        h[i] = ((e > 0x1p126f ? 0.f : recip(table, 1.f + e)) * a) * b;
    }
}
extern "C" __m512 _ZGVeN16v_exp2f(__m512);
void activate(const uint32_t* table, const float* g, const float* u, float* h, int n) {
    const I zero = _mm512_setzero_si512(), one = _mm512_set1_epi32(1), sign = _mm512_set1_epi32(0x80000000),
            max_idx = _mm512_set1_epi32(2 * N - 1), bias = _mm512_set1_epi32(0x3f000000);
    int j = 0;
    for (; j + 16 <= n; j += 16) {
        F a = _mm512_min_ps(_mm512_loadu_ps(g + j), _mm512_set1_ps(10));
        F v = _mm512_max_ps(_mm512_set1_ps(-10), _mm512_min_ps(_mm512_set1_ps(10), _mm512_loadu_ps(u + j)));
        F x = _mm512_mul_ps(a, _mm512_set1_ps(-1.44269504f));
        if (_mm512_cmp_ps_mask(x, _mm512_set1_ps(125), _CMP_NLT_UQ)) {
            scalar_activate(table, g + j, u + j, h + j, 16);
            continue;
        }
        I e = _mm512_cvttps_epi32(x);
        F frac = _mm512_sub_ps(x, _mm512_cvtepi32_ps(e));
        I val = _mm512_castps_si512(frac);
        I key = _mm512_xor_si512(val, _mm512_or_si512(_mm512_srai_epi32(val, 31), sign));
        I idx = _mm512_min_epu32(max_idx, _mm512_sub_epi32(_mm512_castps_si512(_ZGVeN16v_exp2f(frac)), bias));
        __mmask16 change;
        do {
            I bound = _mm512_i32gather_epi32(idx, table, 4);
            change = _mm512_cmplt_epu32_mask(key, bound) & _mm512_cmpneq_epi32_mask(idx, zero);
            idx = _mm512_mask_sub_epi32(idx, change, idx, one);
        } while (change);
        do {
            __mmask16 valid = _mm512_cmplt_epu32_mask(idx, max_idx);
            I bound = _mm512_mask_i32gather_epi32(zero, valid, idx, table + 1, 4);
            change = valid & _mm512_cmpge_epu32_mask(key, bound);
            idx = _mm512_mask_add_epi32(idx, change, idx, one);
        } while (change);
        F expv = _mm512_castsi512_ps(_mm512_add_epi32(_mm512_add_epi32(idx, bias), _mm512_slli_epi32(e, 23)));
        I denom = _mm512_castps_si512(_mm512_add_ps(_mm512_set1_ps(1), expv));
        I inv = _mm512_i32gather_epi32(_mm512_and_si512(denom, _mm512_set1_epi32(0x7fffff)), table + 2 * N, 4);
        I adj = _mm512_sub_epi32(_mm512_and_si512(denom, _mm512_set1_epi32(0x7f800000)),
                                 _mm512_set1_epi32(0x3f800000));
        _mm512_storeu_ps(h + j,
                         _mm512_mul_ps(_mm512_mul_ps(a, _mm512_castsi512_ps(_mm512_sub_epi32(inv, adj))), v));
    }
    if (j < n)
        scalar_activate(table, g + j, u + j, h + j, n - j);
}
static float half(const B* p) {
    uint16_t h;
    memcpy(&h, p, 2);
    return _cvtsh_ss(h);
}
// Packed activation lanes follow the CUDA dot-product group order.
struct PackedActivation {
    I vals[2][2], sum[2];
    F scale[2];
};
static void pack(PackedActivation* out, const B* x, int typ) {
    I one = _mm512_set1_epi8(1);
    for (int i = 0; i < 2; i++) {
        alignas(64) int group[16], ofs[16];
        for (int j = 0; j < 16; j++) {
            group[j] = 36 * (typ == 12 ? j / 4 * 2 + i : j / 8 * 4 + j % 8 / 4 + 2 * i);
            ofs[j] = group[j] + 4 + ((j % 4) * (typ == 12 ? 4 : 8));
        }
        I idx = _mm512_load_si512(group), off = _mm512_load_si512(ofs);
        for (int b = 0; b < 6; b++) {
            auto ptr = x + b * 288;
            I d = _mm512_i32gather_epi32(idx, ptr, 1);
            out[b].scale[i] = _mm512_cvtph_ps(_mm512_cvtepi32_epi16(d));
            I a = _mm512_i32gather_epi32(off, ptr, 1),
              v = _mm512_i32gather_epi32(off, ptr + (typ == 12 ? 16 : 4), 1);
            out[b].vals[i][0] = a;
            out[b].vals[i][1] = v;
            out[b].sum[i] = _mm512_dpbusd_epi32(_mm512_dpbusd_epi32(_mm512_setzero_si512(), one, a), one, v);
        }
    }
}
static I ints(int x) { return _mm512_set1_epi32(x); }
static I coef(int a, int b, int c, int d) {
    return _mm512_setr_epi32(a, a, a, a, b, b, b, b, c, c, c, c, d, d, d, d);
}
template <int typ>
static inline __attribute__((always_inline)) F dot_block(const B* w, const PackedActivation& x) {
    I zero = ints(0), mask = ints(0x0f0f0f0f);
    F d = _mm512_setzero_ps(), m = d;
    if (typ == 12) {
        I ix = _mm512_setr_epi32(0, 1, 2, 3, 8, 9, 10, 11, 16, 17, 18, 19, 24, 25, 26, 27);
        I a = _mm512_loadu_si512(w + 16), b = _mm512_loadu_si512(w + 80);
        I vv[2] = {_mm512_permutex2var_epi32(a, ix, b),
                   _mm512_permutex2var_epi32(a, _mm512_add_epi32(ix, ints(4)), b)};
        unsigned x0, x1, x2;
        memcpy(&x0, w + 4, 4);
        memcpy(&x1, w + 8, 4);
        memcpy(&x2, w + 12, 4);
        unsigned sc0 = x0 & 0x3f3f3f3f, sc1 = (x2 & 0x0f0f0f0f) | ((x0 >> 2) & 0x30303030);
        unsigned mn0 = x1 & 0x3f3f3f3f, mn1 = ((x2 >> 4) & 0x0f0f0f0f) | ((x1 >> 2) & 0x30303030);
        auto expand = [](unsigned lo, unsigned hi) {
            return _mm512_cvtepu8_epi32(_mm_cvtsi64_si128((uint64_t(hi) << 32) | lo));
        };
        I scales = expand(sc0, sc1), mins = expand(mn0, mn1), p0 = coef(0, 2, 4, 6);

        for (int i = 0; i < 2; i++) {
            I acc = zero;
            for (int k = 0; k < 2; k++) {
                I z = i ? _mm512_srli_epi16(vv[k], 4) : vv[k];
                acc = _mm512_dpbusd_epi32(acc, _mm512_and_si512(z, mask), x.vals[i][k]);
            }
            I idx = _mm512_add_epi32(p0, ints(i));
            I sc = _mm512_permutexvar_epi32(idx, scales), mn = _mm512_permutexvar_epi32(idx, mins);
            d = _mm512_fmadd_ps(x.scale[i], _mm512_cvtepi32_ps(_mm512_mullo_epi32(sc, acc)), d);
            m = _mm512_fmadd_ps(x.scale[i], _mm512_cvtepi32_ps(_mm512_mullo_epi32(mn, x.sum[i])), m);
        }
        return _mm512_fmsub_ps(_mm512_set1_ps(half(w)), d, _mm512_mul_ps(_mm512_set1_ps(half(w + 2)), m));
    }
    I ix = _mm512_setr_epi32(0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30);
    I lo0 = _mm512_loadu_si512(w), lo1 = _mm512_loadu_si512(w + 64), hh = _mm512_loadu_si512(w + 128);
    I hix = _mm512_setr_epi32(0, 2, 4, 6, 0, 2, 4, 6, 8, 10, 12, 14, 8, 10, 12, 14);
    for (int i = 0; i < 2; i++) {
        I acc = zero;
        for (int k = 0; k < 2; k++) {
            I lo = _mm512_permutex2var_epi32(lo0, _mm512_add_epi32(ix, ints(k)), lo1);
            I hi = _mm512_permutexvar_epi32(_mm512_add_epi32(hix, ints(k)), hh);
            I sh = coef(0, 2, 0, 2);
            if (i) {
                lo = _mm512_srli_epi16(lo, 4);
                sh = _mm512_add_epi32(sh, ints(4));
            }
            I q = _mm512_or_si512(
                _mm512_and_si512(lo, mask),
                _mm512_slli_epi16(_mm512_and_si512(_mm512_srlv_epi32(hi, sh), ints(0x03030303)), 4));
            acc = _mm512_dpbusd_epi32(acc, q, x.vals[i][k]);
        }
        I idx = _mm512_setr_epi32(0, 0, 1, 1, 2, 2, 3, 3, 8, 8, 9, 9, 10, 10, 11, 11);
        I sv = _mm512_permutexvar_epi32(_mm512_add_epi32(idx, ints(4 * i)),
                                        _mm512_cvtepi8_epi32(_mm_loadu_si128((const __m128i*)(w + 192))));
        acc = _mm512_sub_epi32(acc, _mm512_slli_epi32(x.sum[i], 5));
        d = _mm512_fmadd_ps(x.scale[i], _mm512_cvtepi32_ps(_mm512_mullo_epi32(sv, acc)), d);
    }
    return _mm512_mul_ps(d, _mm512_set1_ps(half(w + 208)));
}
template <int T> inline void rows(const B* w, const PackedActivation* packed, float* y, int nr) {
    constexpr int stride = T == 12 ? 144 : 210;
    for (int r = 0; r < nr; r++) {
        // Keep the even/odd superblock accumulators and the CUDA reduction tree.
        F even = _mm512_setzero_ps(), odd = even;
        for (int b = 0; b < 6; b += 2) {
            F a = dot_block<T>(w, packed[b]), v = dot_block<T>(w + stride, packed[b + 1]);
            even = _mm512_add_ps(even, a);
            odd = _mm512_add_ps(odd, v);
            w += 2 * stride;
        }
        F z = _mm512_add_ps(even, odd);
        __m256 a = _mm256_add_ps(_mm512_castps512_ps256(z), _mm512_extractf32x8_ps(z, 1));
        __m128 b = _mm_add_ps(_mm256_castps256_ps128(a), _mm256_extractf128_ps(a, 1));
        b = _mm_add_ps(b, _mm_movehl_ps(b, b));
        y[r] = _mm_cvtss_f32(_mm_add_ss(b, _mm_shuffle_ps(b, b, 0x55)));
    }
}
} // namespace

void down(int typ, const B* w, const B* x, float* y, int nr) {
    PackedActivation a[6];
    pack(a, x, typ);
    if (typ == 12)
        rows<12>(w, a, y, nr);
    else
        rows<14>(w, a, y, nr);
}
void gate_up(const uint32_t* table, const B* gate, const B* up, const B* activation, float* out, int count) {
    PackedActivation packed[6];
    pack(packed, activation, 12);
    float gate_values[1536];
    float up_values[1536];
    rows<12>(gate, packed, gate_values, count);
    rows<12>(up, packed, up_values, count);
    activate(table, gate_values, up_values, out, count);
}

void quantize(const uint32_t* table, const float* x, void* out, int n) {
    // CUDA fast math flushes FP32 subnormal inputs and results at each operation.
    const auto previous_mode = _mm_getcsr();
    const auto cuda_mode = previous_mode | _MM_FLUSH_ZERO_ON | _MM_DENORMALS_ZERO_ON;
    if (previous_mode != cuda_mode) _mm_setcsr(cuda_mode);
    auto q = static_cast<B*>(out);
    for (int i = 0; i < n; i += 32, x += 32, q += 36) {
        const F lo = _mm512_loadu_ps(x);
        const F hi = _mm512_loadu_ps(x + 16);
        const float max_abs = _mm512_reduce_max_ps(_mm512_max_ps(_mm512_abs_ps(lo), _mm512_abs_ps(hi)));

        // Match the GPU's 32-lane reduction tree before rounding the sum to FP16.
        const F sum16 = _mm512_add_ps(lo, hi);
        const __m256 sum8 = _mm256_add_ps(_mm512_castps512_ps256(sum16), _mm512_extractf32x8_ps(sum16, 1));
        const __m128 sum4 = _mm_add_ps(_mm256_castps256_ps128(sum8), _mm256_extractf128_ps(sum8, 1));
        const __m128 sum2 = _mm_add_ps(sum4, _mm_movehl_ps(sum4, sum4));
        const float sum = _mm_cvtss_f32(_mm_add_ss(sum2, _mm_shuffle_ps(sum2, sum2, 0x55)));

        const float d = std::min(65504.f, max_abs * (1.f / 127.f));
        const uint16_t head[2] = {_cvtss_sh(d, 0), _cvtss_sh(std::clamp(sum, -65504.f, 65504.f), 0)};
        memcpy(q, head, sizeof(head));
        if (d == 0.f) {
            for (int j = 0; j < 32; ++j)
                q[4 + j] = static_cast<int8_t>(x[j] > 0.f ? 127 : x[j] < 0.f ? -127 : 0);
            continue;
        }
        const float inv = recip(table, d);
        for (int j = 0; j < 32; j += 16) {
            F v = _mm512_mul_ps(_mm512_loadu_ps(x + j), _mm512_set1_ps(inv));
            F abs = _mm512_abs_ps(v);
            F val = _mm512_add_round_ps(abs, _mm512_set1_ps(.5f), _MM_FROUND_TO_ZERO | _MM_FROUND_NO_EXC);
            I s = _mm512_or_si512(_mm512_castps_si512(val),
                                  _mm512_and_si512(_mm512_castps_si512(v), ints(0x80000000)));
            I z = _mm512_cvttps_epi32(_mm512_max_ps(
                _mm512_set1_ps(-127.f), _mm512_min_ps(_mm512_set1_ps(127.f), _mm512_castsi512_ps(s))));
            _mm_storeu_si128((__m128i*)(q + 4 + j), _mm512_cvtepi32_epi8(z));
        }
    }
    if (previous_mode != cuda_mode) _mm_setcsr(previous_mode);
}

} // namespace strata::hy4_flash::cpu::q81_aligned
