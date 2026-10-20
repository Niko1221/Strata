// src/kernels/cpu/qns_avx2.cpp - qns.hpp on AVX2 + FMA (Haswell / Zen 2 and later; no AVX-512).
//
// Rows are stored in groups of 32 as [group][word][lane] u32 words (qns_blob.py), so word k of 8 consecutive rows is
// one 32-byte load and a vector lane is a row. A chunk = 32 steps = 64 columns = T words, and with the next word every
// state of the chunk is a 16-bit window at a compile-time bit offset. Per pair of steps: two fmix32 (vpmulld), both
// hashes' bytes 0, 1 folded into one dword, level + 54 of the 4 bytes by two vpshufb nibble lookups, split into the
// even and odd columns as int16 and multiplied by the token's int16 254 q0 + q1 of the same columns (vpmaddwd).
// Zen 2 (EPYC 7H12) per 8 rows x 2 steps (32 weights): ~14 cycles for one token, +2.3 per more token sharing the expert
// (a decoded chunk is reused). The floor is the one integer-multiply pipe: 4 vpmulld (2 cycles each) + 2 vpmaddwd.
#include "strata/kernels/cpu/qns.hpp"

#include <immintrin.h>

#include <algorithm>
#include <array>
#include <cassert>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <utility>

namespace strata::kernels::cpu::qns {
namespace {

constexpr float kMean0 = 1.5769348f, kMean1 = 1.4910126f, kRms = 19.183672f;

struct ActHeader {
    float step;
    int32_t sum;      // sum of 254 q0 + q1 over all columns
    float even, odd;  // E, O
};
static_assert(sizeof(ActHeader) == 16);

// One input width N = 5 n: the signs, and the 5 x 5 factor Q as 9 diagonals over 8 consecutive columns from a column
// r (mod 5): diag[r][d + 4][l] = Q[o][o + d] with o = (r + l) % 5 (0 outside Q), so (I (x) Q) y at columns j .. j + 7 is
// the sum over d of diag[j % 5][d + 4] * y[j + d .. j + d + 7].
template <int N>
struct Rotation {
    alignas(32) float signs[N];
    alignas(32) float diag[5][9][8];
};
Rotation<2560> g_rot2560;
Rotation<640> g_rot640;

template <int N>
void set(Rotation<N>& rot, const float* signs, const float* q) {
    std::memcpy(rot.signs, signs, sizeof(rot.signs));
    for (int r = 0; r < 5; ++r)
        for (int d = -4; d <= 4; ++d)
            for (int l = 0; l < 8; ++l) {
                const int o = (r + l) % 5, c = o + d;
                rot.diag[r][d + 4][l] = c >= 0 && c < 5 ? q[o * 5 + c] : 0.0f;
            }
}

// Walsh-Hadamard butterflies of stride 5, 10, 20 for 8 columns from j: which lanes are the upper half of their pair.
// The pattern depends on j % 40, i.e. on (j / 8) % 5.
constexpr std::array<int32_t, 3 * 5 * 8> kUpper = [] {
    std::array<int32_t, 3 * 5 * 8> t{};
    for (int s = 0; s < 3; ++s)
        for (int k = 0; k < 5; ++k)
            for (int l = 0; l < 8; ++l) t[(s * 5 + k) * 8 + l] = (8 * k + l) % (10 << s) >= (5 << s) ? -1 : 0;
    return t;
}();

// x_rot = (H_{N / 5} (x) Q)(signs * x) / sqrt(N / 5), quantized into dst (layout in qns.hpp). Every float operation is
// the plain scalar recipe's, in its order (Q's sum over c = 0 .. 4 from 0, butterflies a + b and a - b, |v * norm|,
// v * norm * inv), so the bytes do not depend on the vectorization. That needs mul and add kept apart: GCC fuses them
// into FMAs depending on its tuning, Clang and MSVC do not across intrinsics.
#if defined(__GNUC__) && !defined(__clang__)
#pragma GCC push_options
#pragma GCC optimize("fp-contract=off")
#endif
template <int N>
void quantize(const float* x, const Rotation<N>& rot, void* dst) {
    constexpr int pad = 24;  // the diagonals read columns j - 4 .. j + 11, the small butterflies j - 20 .. j + 27
    alignas(32) float buf_a[pad + N + pad], buf_b[pad + N + pad];
    std::memset(buf_a, 0, pad * sizeof(float));  // the diagonals multiply the pads by 0: they must be finite
    std::memset(buf_a + pad + N, 0, pad * sizeof(float));
    std::memset(buf_b, 0, pad * sizeof(float));
    std::memset(buf_b + pad + N, 0, pad * sizeof(float));
    float* a = buf_a + pad;
    float* b = buf_b + pad;
    for (int j = 0; j < N; j += 8) _mm256_store_ps(a + j, _mm256_mul_ps(_mm256_load_ps(rot.signs + j), _mm256_loadu_ps(x + j)));
    for (int j = 0; j < N; j += 32) {  // b = (I (x) Q) a, four sums at once (each is a chain of 9 dependent adds)
        const auto term = [&](int k, int d) {
            return _mm256_mul_ps(_mm256_load_ps(rot.diag[(j + 8 * k) % 5][d + 4]), _mm256_loadu_ps(a + j + 8 * k + d));
        };
        __m256 acc0 = _mm256_setzero_ps(), acc1 = acc0, acc2 = acc0, acc3 = acc0;
        for (int d = -4; d <= 4; ++d) {
            acc0 = _mm256_add_ps(acc0, term(0, d));
            acc1 = _mm256_add_ps(acc1, term(1, d));
            acc2 = _mm256_add_ps(acc2, term(2, d));
            acc3 = _mm256_add_ps(acc3, term(3, d));
        }
        _mm256_store_ps(b + j, acc0);
        _mm256_store_ps(b + j + 8, acc1);
        _mm256_store_ps(b + j + 16, acc2);
        _mm256_store_ps(b + j + 24, acc3);
    }
    // Walsh-Hadamard over i of column 5 i + o: strides 5, 10, 20 from one buffer into the other (b -> a -> b -> a), as
    // a + b (lower lanes) and a - b computed from the column one stride below (upper lanes); 40 and up in place in a
    for (int s = 0; s < 3; ++s) {
        const int len = 5 << s;
        const float* from = s == 1 ? a : b;
        float* to = s == 1 ? b : a;
        for (int j = 0; j < N; j += 8) {
            const __m256 here = _mm256_load_ps(from + j);
            const __m256 sum = _mm256_add_ps(here, _mm256_loadu_ps(from + j + len));
            const __m256 diff = _mm256_sub_ps(_mm256_loadu_ps(from + j - len), here);
            const __m256i upper = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(kUpper.data() + (s * 5 + (j / 8) % 5) * 8));
            _mm256_store_ps(to + j, _mm256_blendv_ps(sum, diff, _mm256_castsi256_ps(upper)));
        }
    }
    static_assert(N == 640 || N == 2560, "strides 40 .. N / 2 go two per pass: an even count");
    for (int len = 40; len < N; len *= 4)  // strides len and 2 len per pass (the same sums as one stride at a time)
        for (int i = 0; i < N; i += 4 * len)
            for (int j = i; j < i + len; j += 8) {
                float* p = a + j;
                const __m256 x0 = _mm256_load_ps(p), x1 = _mm256_load_ps(p + len);
                const __m256 x2 = _mm256_load_ps(p + 2 * len), x3 = _mm256_load_ps(p + 3 * len);
                const __m256 y0 = _mm256_add_ps(x0, x1), y1 = _mm256_sub_ps(x0, x1);
                const __m256 y2 = _mm256_add_ps(x2, x3), y3 = _mm256_sub_ps(x2, x3);
                _mm256_store_ps(p, _mm256_add_ps(y0, y2));
                _mm256_store_ps(p + len, _mm256_add_ps(y1, y3));
                _mm256_store_ps(p + 2 * len, _mm256_sub_ps(y0, y2));
                _mm256_store_ps(p + 3 * len, _mm256_sub_ps(y1, y3));
            }
    const __m256 norm = _mm256_set1_ps(1.0f / std::sqrt(float(N / 5)));
    const __m256 magnitude = _mm256_castsi256_ps(_mm256_set1_epi32(0x7FFFFFFF));
    __m256 max8 = _mm256_setzero_ps();
    for (int j = 0; j < N; j += 8) max8 = _mm256_max_ps(max8, _mm256_and_ps(_mm256_mul_ps(_mm256_load_ps(a + j), norm), magnitude));
    alignas(32) float lanes[8];
    _mm256_store_ps(lanes, max8);
    const float max = *std::max_element(lanes, lanes + 8);
    const float step = max > 0.0f ? max / 127.0f : 1.0f;
    const __m256 inv = _mm256_set1_ps(1.0f / step), k254 = _mm256_set1_ps(254.0f);
    const auto plane = [&](__m256 v) {  // clamp(nearbyint(v), -127, 127)
        const __m256 r = _mm256_round_ps(v, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
        return _mm256_min_ps(_mm256_max_ps(r, _mm256_set1_ps(-127.0f)), _mm256_set1_ps(127.0f));
    };
    const auto q16 = [&](int j) {  // 254 q0 + q1 of columns j .. j + 7 (exact in float)
        const __m256 scaled = _mm256_mul_ps(_mm256_mul_ps(_mm256_load_ps(a + j), norm), inv);
        const __m256 q0 = plane(scaled), q1 = plane(_mm256_mul_ps(_mm256_sub_ps(scaled, q0), k254));
        return _mm256_cvtps_epi32(_mm256_add_ps(_mm256_mul_ps(q0, k254), q1));
    };
    ActHeader* h = static_cast<ActHeader*>(dst);
    __m256i sums = _mm256_setzero_si256();  // lane l: columns = l (mod 8), so even lanes hold the even columns
    for (int j = 0; j < N; j += 16) {
        const __m256i lo = q16(j), hi = q16(j + 8);
        sums = _mm256_add_epi32(sums, _mm256_add_epi32(lo, hi));
        // per 4 columns the order 0, 2, 1, 3 (the kernel's even and odd level words), then int16
        const __m256i packed = _mm256_packs_epi32(_mm256_shuffle_epi32(lo, _MM_SHUFFLE(3, 1, 2, 0)),
                                                  _mm256_shuffle_epi32(hi, _MM_SHUFFLE(3, 1, 2, 0)));
        _mm256_storeu_si256(reinterpret_cast<__m256i*>(reinterpret_cast<int16_t*>(h + 1) + j),
                            _mm256_permute4x64_epi64(packed, _MM_SHUFFLE(3, 1, 2, 0)));
    }
    alignas(32) int32_t s[8];
    _mm256_store_si256(reinterpret_cast<__m256i*>(s), sums);
    const int even = s[0] + s[2] + s[4] + s[6], odd = s[1] + s[3] + s[5] + s[7];
    *h = {step, even + odd, step * float(even) / 254.0f, step * float(odd) / 254.0f};
}
#if defined(__GNUC__) && !defined(__clang__)
#pragma GCC pop_options
#endif

// The 16-bit state at bit B of the chunk whose words k = 0 .. T of the 8 lanes are at w + 32 k.
template <int B>
inline __m256i state(const uint32_t* w) {
    constexpr int k = B / 32, s = B % 32;
    const __m256i mask = _mm256_set1_epi32(0xFFFF);
    const __m256i lo = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(w + 32 * k));
    if constexpr (s <= 16) {
        return _mm256_and_si256(_mm256_srli_epi32(lo, 16 - s), mask);
    } else {
        const __m256i hi = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(w + 32 * (k + 1)));
        return _mm256_and_si256(_mm256_or_si256(_mm256_slli_epi32(lo, s - 16), _mm256_srli_epi32(hi, 48 - s)), mask);
    }
}

// level + 54 of byte b = kNibbles[b & 15] + kNibbles[32 + (b >> 4)]: 8 * (sum of b's 2-bit fields) + ((3 * b) & 15).
constexpr std::array<int8_t, 64> kNibbles = [] {
    std::array<int8_t, 64> t{};
    for (int i = 0; i < 32; ++i) {
        const int n = i % 16, fields = (n & 3) + (n >> 2);
        t[i] = int8_t(8 * fields + ((3 * n) & 15));
        t[32 + i] = int8_t(8 * fields);
    }
    return t;
}();

// x0, x1 = fmix32 of steps 2g, 2g + 1 before its last x ^= x >> 16 -> level + 54 of columns 4g .. 4g + 3 as int16
// pairs: out[0] = columns {4g, 4g + 2}, out[1] = {4g + 1, 4g + 3} (the activation's pair order).
inline void pair_levels(__m256i x0, __m256i x1, __m256i* out) {
    // the hashes' low halves lo16 ^ hi16 of both in one dword: {x0 lo ^ x0 hi, x1 hi ^ x1 lo}
    const __m256i swap16 = _mm256_setr_epi8(2, 3, 0, 1, 6, 7, 4, 5, 10, 11, 8, 9, 14, 15, 12, 13,
                                            2, 3, 0, 1, 6, 7, 4, 5, 10, 11, 8, 9, 14, 15, 12, 13);
    const __m256i h = _mm256_xor_si256(_mm256_blend_epi16(x0, x1, 0xAA),
                                       _mm256_shuffle_epi8(_mm256_blend_epi16(x1, x0, 0xAA), swap16));
    const __m256i nibble = _mm256_set1_epi8(0x0F);
    const __m256i low = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(kNibbles.data()));
    const __m256i high = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(kNibbles.data() + 32));
    const __m256i l = _mm256_add_epi8(_mm256_shuffle_epi8(low, _mm256_and_si256(h, nibble)),
                                      _mm256_shuffle_epi8(high, _mm256_and_si256(_mm256_srli_epi32(h, 4), nibble)));
    out[0] = _mm256_and_si256(l, _mm256_set1_epi32(0x00FF00FF));
    out[1] = _mm256_srli_epi16(l, 8);
}

// Steps 4G .. 4G + 3 of a chunk: use(lv, G) with lv[0 .. 3] = pair_levels of pairs 2G, 2G + 1. The four hashes go stage
// by stage: one step alone is a ~25-cycle dependency chain and the compiler keeps source order (10% faster).
template <int T, int G, class Use>
inline void quad_levels(const uint32_t* w, Use& use) {
    __m256i x0 = state<4 * G * T>(w), x1 = state<(4 * G + 1) * T>(w), x2 = state<(4 * G + 2) * T>(w),
            x3 = state<(4 * G + 3) * T>(w);
    const auto each = [&](auto f) { x0 = f(x0), x1 = f(x1), x2 = f(x2), x3 = f(x3); };
    each([](__m256i x) { return _mm256_add_epi32(_mm256_mullo_epi32(x, _mm256_set1_epi32(int(0xCFCCB83Fu))), _mm256_set1_epi32(0x584B4AA3)); });
    each([](__m256i x) { return _mm256_xor_si256(x, _mm256_srli_epi32(x, 16)); });
    each([](__m256i x) { return _mm256_mullo_epi32(x, _mm256_set1_epi32(int(0x85EBCA6Bu))); });
    __m256i lv[4];
    pair_levels(x0, x1, lv);
    pair_levels(x2, x3, lv + 2);
    use(lv, G);
}

template <int T, class Use, int... G>
inline void decode_chunk(const uint32_t* w, Use& use, std::integer_sequence<int, G...>) {
    (quad_levels<T, G>(w, use), ...);
}

// acc, acc2 += levels lv[0 .. 3] . the token's activation pairs a[0 .. 3]. |acc + acc2| <= 64 * 109 * 32385 < 2^31
// over a chunk.
inline void dot4(const __m256i* lv, const int32_t* a, __m256i& acc, __m256i& acc2) {
    acc = _mm256_add_epi32(acc, _mm256_madd_epi16(lv[0], _mm256_set1_epi32(a[0])));
    acc2 = _mm256_add_epi32(acc2, _mm256_madd_epi16(lv[1], _mm256_set1_epi32(a[1])));
    acc = _mm256_add_epi32(acc, _mm256_madd_epi16(lv[2], _mm256_set1_epi32(a[2])));
    acc2 = _mm256_add_epi32(acc2, _mm256_madd_epi16(lv[3], _mm256_set1_epi32(a[3])));
}

// s[t][i] for the 8 rows whose lanes start at w (word k of chunk c at w + 32 (c T + k)).
template <int T>
void block_sums_t(const uint32_t* w, int chunks, const void* const* act, int nt, int64_t (*s)[8]) {
    __m256i lo[8], hi[8];  // per token, int64 sums of lanes 0-3 and 4-7
    for (int t = 0; t < nt; ++t) lo[t] = hi[t] = _mm256_setzero_si256();
    const auto pairs = [&](int t, int c) { return reinterpret_cast<const int32_t*>(static_cast<const ActHeader*>(act[t]) + 1) + 32 * c; };
    const auto add = [&](int t, __m256i acc) {
        lo[t] = _mm256_add_epi64(lo[t], _mm256_cvtepi32_epi64(_mm256_castsi256_si128(acc)));
        hi[t] = _mm256_add_epi64(hi[t], _mm256_cvtepi32_epi64(_mm256_extracti128_si256(acc, 1)));
    };
    for (int c = 0; c < chunks; ++c) {
        __m256i acc = _mm256_setzero_si256(), acc2 = _mm256_setzero_si256();
        if (nt == 1) {  // the levels go straight into the dot product (10% faster than through lv)
            auto dot = [&](const __m256i* l, int g) { dot4(l, pairs(0, c) + 4 * g, acc, acc2); };
            decode_chunk<T>(w + 32 * T * c, dot, std::make_integer_sequence<int, 8>());
            add(0, _mm256_add_epi32(acc, acc2));
            continue;
        }
        __m256i lv[32];
        auto keep = [&](const __m256i* l, int g) { std::memcpy(lv + 4 * g, l, 4 * sizeof(__m256i)); };
        decode_chunk<T>(w + 32 * T * c, keep, std::make_integer_sequence<int, 8>());
        for (int t = 0; t < nt; ++t) {
            acc = acc2 = _mm256_setzero_si256();
            for (int g = 0; g < 8; ++g) dot4(lv + 4 * g, pairs(t, c) + 4 * g, acc, acc2);
            add(t, _mm256_add_epi32(acc, acc2));
        }
    }
    for (int t = 0; t < nt; ++t) {
        _mm256_storeu_si256(reinterpret_cast<__m256i*>(s[t]), lo[t]);
        _mm256_storeu_si256(reinterpret_cast<__m256i*>(s[t] + 4), hi[t]);
    }
}

struct Matrix {
    const uint32_t* words;
    const float* scales;
    int bits, chunks;
};

// Header u32[16]: magic, T_gu, T_d, W_gu, W_d, byte offsets of gu words, d words, gu scales, d scales (qns_blob.py).
Matrix matrix(const uint8_t* blob, bool down) {
    const uint32_t* h = reinterpret_cast<const uint32_t*>(blob);
    assert(h[0] == 0x31534E51u);
    const int bits = int(h[down ? 2 : 1]), chunks = down ? 10 : 40;
    assert(int(h[down ? 4 : 3]) == chunks * bits + 1);
    return {reinterpret_cast<const uint32_t*>(blob + h[down ? 6 : 5]),
            reinterpret_cast<const float*>(blob + h[down ? 8 : 7]), bits, chunks};
}

// y = scale * ((step / RMS) * ((C - 54 sum q0) + (F - 54 sum q1) / 254) + d0 E + d1 O), with S = 254 C + F.
float finish(int64_t s, const void* act, float scale) {
    const ActHeader& a = *static_cast<const ActHeader*>(act);
    const float dot = float(double(s - 54 * int64_t(a.sum)) / 254.0);
    return scale * (a.step / kRms * dot - kMean0 / kRms * a.even - kMean1 / kRms * a.odd);
}

}  // namespace

void set_rotation(const float* signs2560, const float* q2560, const float* signs640, const float* q640) {
    set(g_rot2560, signs2560, q2560);
    set(g_rot640, signs640, q640);
}

void quant_act(const float* x, void* dst) { quantize<2560>(x, g_rot2560, dst); }

void quant_h(const float* h, void* dst) { quantize<640>(h, g_rot640, dst); }

void block_sums(const uint8_t* blob, bool down, int b, const void* const* act, int nt, int64_t (*s)[8]) {
    assert(1 <= nt && nt <= 8);
    const Matrix m = matrix(blob, down);
    const uint32_t* w = m.words + size_t(b / 4) * (m.chunks * m.bits + 1) * 32 + 8 * (b % 4);
    switch (m.bits) {
        case 3: return block_sums_t<3>(w, m.chunks, act, nt, s);
        case 4: return block_sums_t<4>(w, m.chunks, act, nt, s);
        case 5: return block_sums_t<5>(w, m.chunks, act, nt, s);
        case 6: return block_sums_t<6>(w, m.chunks, act, nt, s);
        case 7: return block_sums_t<7>(w, m.chunks, act, nt, s);
        case 8: return block_sums_t<8>(w, m.chunks, act, nt, s);
        case 9: return block_sums_t<9>(w, m.chunks, act, nt, s);
        case 10: return block_sums_t<10>(w, m.chunks, act, nt, s);
    }
    std::abort();  // T outside 3..10: not a qns blob
}

void gu_rows(const uint8_t* blob, const void* const* act, int nt, float* const* ff, int r0, int r1) {
    assert(0 <= r0 && r0 < r1 && r1 <= 640);
    const float* scales = matrix(blob, false).scales;
    for (int b = r0 / 8; 8 * b < r1; ++b) {
        int64_t gate[8][8], up[8][8];
        block_sums(blob, false, b, act, nt, gate);
        block_sums(blob, false, 80 + b, act, nt, up);
        for (int t = 0; t < nt; ++t)
            for (int r = std::max(r0, 8 * b); r < std::min(r1, 8 * b + 8); ++r) {
                const float g = finish(gate[t][r % 8], act[t], scales[r]);
                ff[t][r] = g / (1.0f + std::exp(-g)) * finish(up[t][r % 8], act[t], scales[640 + r]);
            }
    }
}

void down_rows(const uint8_t* blob, const void* const* hq, int nt, float* const* out, int r0, int r1) {
    assert(0 <= r0 && r0 < r1 && r1 <= 2560);
    const float* scales = matrix(blob, true).scales;
    for (int b = r0 / 8; 8 * b < r1; ++b) {
        int64_t s[8][8];
        block_sums(blob, true, b, hq, nt, s);
        for (int t = 0; t < nt; ++t)
            for (int r = std::max(r0, 8 * b); r < std::min(r1, 8 * b + 8); ++r)
                out[t][r] = finish(s[t][r % 8], hq[t], scales[r]);
    }
}

}  // namespace strata::kernels::cpu::qns
