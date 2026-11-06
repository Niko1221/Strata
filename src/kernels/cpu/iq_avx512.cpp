// src/kernels/cpu/iq_avx512.cpp - plan v0.3 P6: the i-quant expert rows in 512-bit lanes, several tokens at once.
//
// ggml-cpu's x86 dot products for these formats are AVX2 and single-token: every token re-decodes the weights
// and a sign is applied with five instructions per 32 values.  Here 64 values are decoded once (8 or 16 grid
// lookups, one 64-bit sign mask, one vector of scales) and every token applies them with a masked subtract, a
// `maddubs` and a `madd`.  The arithmetic is ggml's (ggml-cpu/quants.c, the `_generic` references): integer sums
// per block, times d_x * d_y * the format's constant - only the order of the float additions differs.
//
// Formats: IQ2_XXS (16), IQ2_XS (17), IQ3_XXS (18), IQ3_S (21), IQ2_S (22).  IQ1_M stays on ggml-cpu (no shipped
// model has IQ1_M expert rows: the 'Coder IQ1_M' pack's gate/up are IQ2_S / IQ3_XXS / IQ3_S).
#include "strata/kernels/cpu/iq_avx512.hpp"
#include "strata/kernels/cpu/expert_layout.hpp"

#define GGML_COMMON_DECL_CPP
#define GGML_COMMON_IMPL_CPP
#include "ggml-common.h"

#include <immintrin.h>
#if defined(_MSC_VER)
#include <intrin.h>
#else
#include <cpuid.h>
#endif

#include <cmath>
#include <cstdlib>
#include <cstring>
#include <vector>

namespace strata::kernels::cpu {
namespace {

inline float h2f(uint16_t h) { return _mm_cvtss_f32(_mm_cvtph_ps(_mm_cvtsi32_si128((int) h))); }
inline uint32_t u32(const uint8_t* p) { uint32_t v; std::memcpy(&v, p, 4); return v; }
inline uint16_t u16(const uint8_t* p) { uint16_t v; std::memcpy(&v, p, 2); return v; }
inline uint64_t u64(const uint8_t* p) { uint64_t v; std::memcpy(&v, p, 8); return v; }

// E-2: the IQ3 grids by one AVX-512 gather (the same words, the same results) - measured 3-5% SLOWER than the scalar
// lookups on a Ryzen 5 7600 (Zen 4 gathers are microcoded), so opt-in: STRATA_IQ_GATHER=1
// 0.2.5: DEFAULT on where gathers are fast (Intel server and HEDT cores with AVX-512: cpu_gather_fast). Off on Zen 4
// (microcoded) and on Zen 5 until STRATA_IQ_GATHER=1.  STRATA_IQ_GATHER=0 forces the scalar loads.  The same words are
// loaded, so the results are bit for bit the scalar decode's.
const bool gather = [] {
    const char* v = std::getenv("STRATA_IQ_GATHER");
    // Default stays the Intel gather probe. Zen 5 is off until STRATA_IQ_GATHER=1.
    return v != nullptr && v[0] != '\0' ? std::atoi(v) != 0 : cpu_gather_fast();
}();
inline __mmask64 sign_mask_from_idx(__m256i si) {
    const __m256i par = _mm256_slli_epi32(_mm256_and_si256(_mm256_popcnt_epi32(si), _mm256_set1_epi32(1)), 7);
    return _cvtu64_mask64((uint64_t) _mm_cvtsi128_si64(_mm256_cvtepi32_epi8(_mm256_or_si256(si, par))));
}
inline __m256i sign_idx_xxs(uint32_t lo4, uint32_t hi4) {   // four 7-bit fields of each of two words -> eight dwords
    const __m256i w = _mm256_inserti128_si256(_mm256_castsi128_si256(_mm_set1_epi32((int) lo4)), _mm_set1_epi32((int) hi4), 1);
    return _mm256_and_si256(_mm256_srlv_epi32(w, _mm256_setr_epi32(0, 7, 14, 21, 0, 7, 14, 21)), _mm256_set1_epi32(127));
}
// the grid lookups by one 64-bit gather of eight entries (IQ2_XXS / IQ2_XS / IQ2_S): `idx` = eight 32-bit indices
inline __m512i gather64(__m256i idx, const void* grid) { return _mm512_i32gather_epi64(idx, grid, 8); }
// E-2: software prefetch distance in bytes (STRATA_IQ_PREFETCH; 0 = off): -2-3% gate/up time on IQ3_S decode
// (the rows are read at ~30 GB/s by 6 cores, near what DDR5 with 4 KB pages gives: the pool is memory-bound)
const int prefetch_ahead = [] {
    const char* v = std::getenv("STRATA_IQ_PREFETCH");
    return v ? std::atoi(v) : 2048;
}();

// lanes 0-7 -> s0, 8-15 -> s1, 16-23 -> s2, 24-31 -> s3 (int16 lanes of a maddubs result: two values each)
inline __m512i scales4(int s0, int s1, int s2, int s3) {
    const __m512i idx = _mm512_set_epi16(3, 3, 3, 3, 3, 3, 3, 3, 2, 2, 2, 2, 2, 2, 2, 2, 1, 1, 1, 1, 1, 1, 1, 1,
                                         0, 0, 0, 0, 0, 0, 0, 0);
    const uint64_t packed = (uint64_t) (uint16_t) s0 | ((uint64_t) (uint16_t) s1 << 16) |
                            ((uint64_t) (uint16_t) s2 << 32) | ((uint64_t) (uint16_t) s3 << 48);
    return _mm512_permutexvar_epi16(idx, _mm512_set1_epi64((long long) packed));
}

// lanes 0-15 -> sa, 16-31 -> sb (two 32-value halves of a 64-value step: IQ2_XXS, IQ3_XXS, IQ3_S)
inline __m512i scales2(int sa, int sb) {
    return _mm512_inserti64x4(_mm512_castsi256_si512(_mm256_set1_epi16((short) sa)),
                              _mm256_set1_epi16((short) sb), 1);
}

// ---- per format: 64 values (chunk j of a 256-value block) -> grid magnitudes, sign mask, scales
template <int TY> struct Fmt;

template <> struct Fmt<16> {   // IQ2_XXS: d, qs[32] u16
    static constexpr int bytes = 66;
    static constexpr float K = 0.125f;
    static inline void decode(const uint8_t* b, int j, __m512i& g, __mmask64& m, __m512i& sc) {
        const uint8_t* q = b + 2 + 16 * j;
        const uint32_t a0 = u32(q), a1 = u32(q + 4), b0 = u32(q + 8), b1 = u32(q + 12);
        g = gather ? gather64(_mm256_cvtepu8_epi32(_mm_cvtsi64_si128((long long) ((uint64_t) a0 | ((uint64_t) b0 << 32)))), iq2xxs_grid)
                   : _mm512_set_epi64((long long) iq2xxs_grid[b0 >> 24], (long long) iq2xxs_grid[(b0 >> 16) & 255],
                             (long long) iq2xxs_grid[(b0 >> 8) & 255], (long long) iq2xxs_grid[b0 & 255],
                             (long long) iq2xxs_grid[a0 >> 24], (long long) iq2xxs_grid[(a0 >> 16) & 255],
                             (long long) iq2xxs_grid[(a0 >> 8) & 255], (long long) iq2xxs_grid[a0 & 255]);
        const uint64_t s = (uint64_t) ksigns_iq2xs[a1 & 127] | ((uint64_t) ksigns_iq2xs[(a1 >> 7) & 127] << 8) |
                           ((uint64_t) ksigns_iq2xs[(a1 >> 14) & 127] << 16) | ((uint64_t) ksigns_iq2xs[(a1 >> 21) & 127] << 24) |
                           ((uint64_t) ksigns_iq2xs[b1 & 127] << 32) | ((uint64_t) ksigns_iq2xs[(b1 >> 7) & 127] << 40) |
                           ((uint64_t) ksigns_iq2xs[(b1 >> 14) & 127] << 48) | ((uint64_t) ksigns_iq2xs[(b1 >> 21) & 127] << 56);
        m = _cvtu64_mask64(s);
        const int sa = 2 * (int) (a1 >> 28) + 1, sb = 2 * (int) (b1 >> 28) + 1;
        sc = scales2(sa, sb);
    }
};

template <> struct Fmt<17> {   // IQ2_XS: d, qs[32] u16 (9-bit grid index + 7-bit sign index), scales[8]
    static constexpr int bytes = 74;
    static constexpr float K = 0.125f;
    static inline void decode(const uint8_t* b, int j, __m512i& g, __mmask64& m, __m512i& sc) {
        const uint8_t* q = b + 2 + 16 * j;
        uint16_t v[8];
        std::memcpy(v, q, 16);
        g = gather ? gather64(_mm256_and_si256(_mm256_cvtepu16_epi32(_mm_loadu_si128((const __m128i*) q)), _mm256_set1_epi32(511)), iq2xs_grid)
                   : _mm512_set_epi64((long long) iq2xs_grid[v[7] & 511], (long long) iq2xs_grid[v[6] & 511],
                             (long long) iq2xs_grid[v[5] & 511], (long long) iq2xs_grid[v[4] & 511],
                             (long long) iq2xs_grid[v[3] & 511], (long long) iq2xs_grid[v[2] & 511],
                             (long long) iq2xs_grid[v[1] & 511], (long long) iq2xs_grid[v[0] & 511]);
        uint64_t s = 0;
        for (int l = 0; l < 8; ++l) s |= (uint64_t) ksigns_iq2xs[v[l] >> 9] << (8 * l);
        m = _cvtu64_mask64(s);
        const uint8_t s0 = b[66 + 2 * j], s1 = b[66 + 2 * j + 1];
        sc = scales4(2 * (s0 & 15) + 1, 2 * (s0 >> 4) + 1, 2 * (s1 & 15) + 1, 2 * (s1 >> 4) + 1);
    }
};

template <> struct Fmt<22> {   // IQ2_S: d, qs[64] (32 grid bytes, 32 sign bytes), qh[8], scales[8]
    static constexpr int bytes = 82;
    static constexpr float K = 0.125f;
    static inline void decode(const uint8_t* b, int j, __m512i& g, __mmask64& m, __m512i& sc) {
        const uint8_t* qs = b + 2 + 8 * j;
        const uint8_t h0 = b[66 + 2 * j], h1 = b[66 + 2 * j + 1];
        if (gather) {   // the 10-bit indices: the byte plus two high bits from qh (h0 for lanes 0-3, h1 for 4-7), one 64-bit gather
            const __m256i hi = _mm256_and_si256(_mm256_sllv_epi32(_mm256_setr_epi32(h0, h0, h0, h0, h1, h1, h1, h1),
                                                                   _mm256_setr_epi32(8, 6, 4, 2, 8, 6, 4, 2)), _mm256_set1_epi32(0x300));
            g = gather64(_mm256_or_si256(_mm256_cvtepu8_epi32(_mm_loadl_epi64((const __m128i*) qs)), hi), iq2s_grid);
        } else
        g = _mm512_set_epi64((long long) iq2s_grid[qs[7] | ((h1 << 2) & 0x300)], (long long) iq2s_grid[qs[6] | ((h1 << 4) & 0x300)],
                             (long long) iq2s_grid[qs[5] | ((h1 << 6) & 0x300)], (long long) iq2s_grid[qs[4] | ((h1 << 8) & 0x300)],
                             (long long) iq2s_grid[qs[3] | ((h0 << 2) & 0x300)], (long long) iq2s_grid[qs[2] | ((h0 << 4) & 0x300)],
                             (long long) iq2s_grid[qs[1] | ((h0 << 6) & 0x300)], (long long) iq2s_grid[qs[0] | ((h0 << 8) & 0x300)]);
        m = _cvtu64_mask64(u64(b + 2 + 32 + 8 * j));
        const uint8_t s0 = b[74 + 2 * j], s1 = b[74 + 2 * j + 1];
        sc = scales4(2 * (s0 & 15) + 1, 2 * (s0 >> 4) + 1, 2 * (s1 & 15) + 1, 2 * (s1 >> 4) + 1);
    }
};

template <> struct Fmt<18> {   // IQ3_XXS: d, qs[64] grid bytes, 8 x u32 (4 x 7-bit sign index + 4-bit scale)
    static constexpr int bytes = 98;
    static constexpr float K = 0.25f;
    static inline void decode(const uint8_t* b, int j, __m512i& g, __mmask64& m, __m512i& sc) {
        const uint8_t* q = b + 2 + 16 * j;
        // E-2: one gather of the 16 grid words (on Intel gather-fast CPUs; on Zen 5, 16-lane 32-bit gather is slower than scalar loads)
        g = gather ? _mm512_i32gather_epi32(_mm512_cvtepu8_epi32(_mm_loadu_si128((const __m128i*) q)), iq3xxs_grid, 4)
                   : _mm512_set_epi32((int) iq3xxs_grid[q[15]], (int) iq3xxs_grid[q[14]], (int) iq3xxs_grid[q[13]], (int) iq3xxs_grid[q[12]],
                             (int) iq3xxs_grid[q[11]], (int) iq3xxs_grid[q[10]], (int) iq3xxs_grid[q[9]], (int) iq3xxs_grid[q[8]],
                             (int) iq3xxs_grid[q[7]], (int) iq3xxs_grid[q[6]], (int) iq3xxs_grid[q[5]], (int) iq3xxs_grid[q[4]],
                             (int) iq3xxs_grid[q[3]], (int) iq3xxs_grid[q[2]], (int) iq3xxs_grid[q[1]], (int) iq3xxs_grid[q[0]]);
        const uint32_t a = u32(b + 2 + 64 + 8 * j), c = u32(b + 2 + 64 + 8 * j + 4);
        const uint64_t s = (uint64_t) ksigns_iq2xs[a & 127] | ((uint64_t) ksigns_iq2xs[(a >> 7) & 127] << 8) |
                           ((uint64_t) ksigns_iq2xs[(a >> 14) & 127] << 16) | ((uint64_t) ksigns_iq2xs[(a >> 21) & 127] << 24) |
                           ((uint64_t) ksigns_iq2xs[c & 127] << 32) | ((uint64_t) ksigns_iq2xs[(c >> 7) & 127] << 40) |
                           ((uint64_t) ksigns_iq2xs[(c >> 14) & 127] << 48) | ((uint64_t) ksigns_iq2xs[(c >> 21) & 127] << 56);
        m = _cvtu64_mask64(s);
        const int sa = 2 * (int) (a >> 28) + 1, sb = 2 * (int) (c >> 28) + 1;
        sc = scales2(sa, sb);
    }
};

template <> struct Fmt<21> {   // IQ3_S: d, qs[64], qh[8], signs[32], scales[4]
    static constexpr int bytes = 110;
    static constexpr float K = 1.0f;
    static inline void decode(const uint8_t* b, int j, __m512i& g, __mmask64& m, __m512i& sc) {
        const uint8_t* q = b + 2 + 16 * j;
        const uint32_t h0 = b[66 + 2 * j], h1 = b[66 + 2 * j + 1];
        if (gather) {   // E-2: the 9-bit indices (the high bit from qh under a mask), one gather
            __m512i idx = _mm512_cvtepu8_epi32(_mm_loadu_si128((const __m128i*) q));
            idx = _mm512_mask_add_epi32(idx, (__mmask16) (h0 | (h1 << 8)), idx, _mm512_set1_epi32(256));
            g = _mm512_i32gather_epi32(idx, iq3s_grid, 4);
        } else {
            const __m256i lo = _mm256_cvtepu8_epi16(_mm_loadu_si128((const __m128i*) q));
            alignas(32) uint16_t sp[16];
            _mm256_store_si256((__m256i*) sp,
                               _mm256_mask_add_epi16(lo, (__mmask16) (h0 | (h1 << 8)), lo, _mm256_set1_epi16(256)));
            g = _mm512_set_epi32((int) iq3s_grid[sp[15]], (int) iq3s_grid[sp[14]], (int) iq3s_grid[sp[13]], (int) iq3s_grid[sp[12]],
                                 (int) iq3s_grid[sp[11]], (int) iq3s_grid[sp[10]], (int) iq3s_grid[sp[9]],  (int) iq3s_grid[sp[8]],
                                 (int) iq3s_grid[sp[7]],  (int) iq3s_grid[sp[6]],  (int) iq3s_grid[sp[5]],  (int) iq3s_grid[sp[4]],
                                 (int) iq3s_grid[sp[3]],  (int) iq3s_grid[sp[2]],  (int) iq3s_grid[sp[1]],  (int) iq3s_grid[sp[0]]);
        }
        m = _cvtu64_mask64(u64(b + 74 + 8 * j));
        const uint8_t s = b[106 + j];
        const int sa = 2 * (s & 15) + 1, sb = 2 * (s >> 4) + 1;
        sc = scales2(sa, sb);
    }
};

template <bool Singleton> struct RowAccumulator;
template <> struct RowAccumulator<false> { using Type = __m512; };
template <> struct RowAccumulator<true> { using Type = __m256; };

template <bool VNNI>
inline __m512i madd_add512(__m512i acc, __m512i p16, __m512i sc) {
    if constexpr (VNNI) return _mm512_dpwssd_epi32(acc, p16, sc);
    else return _mm512_add_epi32(acc, _mm512_madd_epi16(p16, sc));
}

template <int TY, int NT, bool Singleton = false, bool VNNI = true>
inline void row_dot(const uint8_t* row, int nblocks, const block_q8_K* const* y, float* res) {
    // Decode weights once for the group; select the accumulation order at compile time.
    using Acc = typename RowAccumulator<Singleton>::Type;
    Acc accf[NT];
    for (int t = 0; t < NT; ++t) {
        if constexpr (Singleton) accf[t] = _mm256_setzero_ps();
        else accf[t] = _mm512_setzero_ps();
    }
    const __m512i zero = _mm512_setzero_si512();
    constexpr int A = NT <= 2 ? 2 : 1;
    for (int i = 0; i < nblocks; ++i) {
        const uint8_t* blk = row + (size_t) i * Fmt<TY>::bytes;
        // E-2: the weight rows stream from DRAM (4 KB pages when large pages are refused): ask for the bytes a few
        // blocks ahead (the next row's start included) before the decode needs them
        if (prefetch_ahead > 0) {
            const uint8_t* ahead = blk + (size_t) prefetch_ahead;
            _mm_prefetch((const char*) ahead, _MM_HINT_T0);
            _mm_prefetch((const char*) ahead + 64, _MM_HINT_T0);
        }
        __m512i acci[NT][A];
        for (int t = 0; t < NT; ++t)
            for (int a = 0; a < A; ++a) acci[t][a] = _mm512_setzero_si512();
        for (int j = 0; j < 4; ++j) {
            __m512i g, sc;
            __mmask64 m;
            Fmt<TY>::decode(blk, j, g, m, sc);
            for (int t = 0; t < NT; ++t) {
                const __m512i yv = _mm512_loadu_si512((const void*) (y[t][i].qs + 64 * j));
                const __m512i ys = _mm512_mask_sub_epi8(yv, m, zero, yv);
                acci[t][j % A] = madd_add512<VNNI>(acci[t][j % A], _mm512_maddubs_epi16(g, ys), sc);
            }
        }
        if constexpr (A == 2) {
            for (int t = 0; t < NT; ++t) acci[t][0] = _mm512_add_epi32(acci[t][0], acci[t][1]);
        }
        if constexpr (Singleton) {
            const float dx = h2f(u16(blk));
            for (int t = 0; t < NT; ++t) {
                // Preserve ggml AVX2's eight integer lanes BEFORE floating conversion.
                const __m256i folded = _mm256_add_epi32(_mm512_castsi512_si256(acci[t][0]),
                                                       _mm512_extracti64x4_epi64(acci[t][0], 1));
                accf[t] = _mm256_fmadd_ps(_mm256_set1_ps(dx * y[t][i].d),
                                         _mm256_cvtepi32_ps(folded), accf[t]);
            }
        } else {
            const float dx = h2f(u16(blk)) * Fmt<TY>::K;
            for (int t = 0; t < NT; ++t)
                accf[t] = _mm512_fmadd_ps(_mm512_set1_ps(dx * y[t][i].d), _mm512_cvtepi32_ps(acci[t][0]), accf[t]);
        }
    }
    if constexpr (Singleton) {
        for (int t = 0; t < NT; ++t) {
            __m128 sum = _mm256_extractf128_ps(accf[t], 1);
            sum = _mm_add_ps(sum, _mm256_castps256_ps128(accf[t]));
            sum = _mm_add_ps(sum, _mm_movehl_ps(sum, sum));
            sum = _mm_add_ss(sum, _mm_movehdup_ps(sum));
            res[t] = Fmt<TY>::K * _mm_cvtss_f32(sum);
        }
    } else {
        for (int t = 0; t < NT; ++t) res[t] = _mm512_reduce_add_ps(accf[t]);
    }
}

// Fused gate + up block dot for NT <= 4: each block's activation y[t][i] is loaded once from L1D for both gate and up,
// and the independent gate/up integer accumulators break the vpdpwssd latency chain while keeping bit-for-bit parity.
template <int TY, int NT, bool Singleton = false>
inline void row_dot_gu(const uint8_t* row_g, const uint8_t* row_u, int nblocks, const block_q8_K* const* y,
                       float* g_out, float* u_out) {
    using Acc = typename RowAccumulator<Singleton>::Type;
    Acc accf_g[NT], accf_u[NT];
    for (int t = 0; t < NT; ++t) {
        if constexpr (Singleton) {
            accf_g[t] = _mm256_setzero_ps();
            accf_u[t] = _mm256_setzero_ps();
        } else {
            accf_g[t] = _mm512_setzero_ps();
            accf_u[t] = _mm512_setzero_ps();
        }
    }
    const __m512i zero = _mm512_setzero_si512();
    for (int i = 0; i < nblocks; ++i) {
        const uint8_t* blk_g = row_g + (size_t) i * Fmt<TY>::bytes;
        const uint8_t* blk_u = row_u + (size_t) i * Fmt<TY>::bytes;
        if (prefetch_ahead > 0) {
            const uint8_t* ahead_g = blk_g + (size_t) prefetch_ahead;
            const uint8_t* ahead_u = blk_u + (size_t) prefetch_ahead;
            _mm_prefetch((const char*) ahead_g, _MM_HINT_T0);
            _mm_prefetch((const char*) ahead_g + 64, _MM_HINT_T0);
            _mm_prefetch((const char*) ahead_u, _MM_HINT_T0);
            _mm_prefetch((const char*) ahead_u + 64, _MM_HINT_T0);
        }
        __m512i acci_g[NT], acci_u[NT];
        for (int t = 0; t < NT; ++t) {
            acci_g[t] = _mm512_setzero_si512();
            acci_u[t] = _mm512_setzero_si512();
        }
        for (int j = 0; j < 4; ++j) {
            __m512i gg, scg, gu, scu;
            __mmask64 mg, mu;
            Fmt<TY>::decode(blk_g, j, gg, mg, scg);
            Fmt<TY>::decode(blk_u, j, gu, mu, scu);
            for (int t = 0; t < NT; ++t) {
                const __m512i yv = _mm512_loadu_si512((const void*) (y[t][i].qs + 64 * j));
                const __m512i ysg = _mm512_mask_sub_epi8(yv, mg, zero, yv);
                const __m512i ysu = _mm512_mask_sub_epi8(yv, mu, zero, yv);
                acci_g[t] = _mm512_dpwssd_epi32(acci_g[t], _mm512_maddubs_epi16(gg, ysg), scg);
                acci_u[t] = _mm512_dpwssd_epi32(acci_u[t], _mm512_maddubs_epi16(gu, ysu), scu);
            }
        }
        if constexpr (Singleton) {
            const float dxg = h2f(u16(blk_g));
            const float dxu = h2f(u16(blk_u));
            for (int t = 0; t < NT; ++t) {
                const float dy = y[t][i].d;
                const __m256i fg = _mm256_add_epi32(_mm512_castsi512_si256(acci_g[t]),
                                                    _mm512_extracti64x4_epi64(acci_g[t], 1));
                const __m256i fu = _mm256_add_epi32(_mm512_castsi512_si256(acci_u[t]),
                                                    _mm512_extracti64x4_epi64(acci_u[t], 1));
                accf_g[t] = _mm256_fmadd_ps(_mm256_set1_ps(dxg * dy), _mm256_cvtepi32_ps(fg), accf_g[t]);
                accf_u[t] = _mm256_fmadd_ps(_mm256_set1_ps(dxu * dy), _mm256_cvtepi32_ps(fu), accf_u[t]);
            }
        } else {
            const float dxg = h2f(u16(blk_g)) * Fmt<TY>::K;
            const float dxu = h2f(u16(blk_u)) * Fmt<TY>::K;
            for (int t = 0; t < NT; ++t) {
                const float dy = y[t][i].d;
                accf_g[t] = _mm512_fmadd_ps(_mm512_set1_ps(dxg * dy), _mm512_cvtepi32_ps(acci_g[t]), accf_g[t]);
                accf_u[t] = _mm512_fmadd_ps(_mm512_set1_ps(dxu * dy), _mm512_cvtepi32_ps(acci_u[t]), accf_u[t]);
            }
        }
    }
    if constexpr (Singleton) {
        for (int t = 0; t < NT; ++t) {
            auto hsum = [](__m256 x) {
                __m128 sum = _mm256_extractf128_ps(x, 1);
                sum = _mm_add_ps(sum, _mm256_castps256_ps128(x));
                sum = _mm_add_ps(sum, _mm_movehl_ps(sum, sum));
                sum = _mm_add_ss(sum, _mm_movehdup_ps(sum));
                return _mm_cvtss_f32(sum);
            };
            g_out[t] = Fmt<TY>::K * hsum(accf_g[t]);
            u_out[t] = Fmt<TY>::K * hsum(accf_u[t]);
        }
    } else {
        for (int t = 0; t < NT; ++t) {
            g_out[t] = _mm512_reduce_add_ps(accf_g[t]);
            u_out[t] = _mm512_reduce_add_ps(accf_u[t]);
        }
    }
}

const bool fused_gu = vcache_pool_opt_in();

template <int TY, int NT, bool Singleton = false>
void gu_rows(const uint8_t* blob, size_t gu_row, size_t up_off, int n, const void* const* act, float* const* ff,
             int r0, int r1) {
    const block_q8_K* y[NT];
    for (int t = 0; t < NT; ++t) y[t] = (const block_q8_K*) act[t];
    const int nb = n / QK_K;
    float g[NT], u[NT];
    if constexpr (NT <= 4) {
        if (fused_gu) {
            for (int r = r0; r < r1; ++r) {
                row_dot_gu<TY, NT, Singleton>(blob + (size_t) r * gu_row, blob + up_off + (size_t) r * gu_row, nb, y, g, u);
                for (int t = 0; t < NT; ++t) ff[t][r] = (g[t] / (1.f + std::exp(-g[t]))) * u[t];
            }
            return;
        }
    }
    for (int r = r0; r < r1; ++r) {
        row_dot<TY, NT, Singleton, true>(blob + (size_t) r * gu_row, nb, y, g);
        row_dot<TY, NT, Singleton, true>(blob + up_off + (size_t) r * gu_row, nb, y, u);
        for (int t = 0; t < NT; ++t) ff[t][r] = (g[t] / (1.f + std::exp(-g[t]))) * u[t];
    }
}

template <int TY, int NT, bool VNNI>
void dot_rows(const uint8_t* w, size_t row_bytes, int n, const void* const* act, float* const* out, int r0, int r1) {
    const block_q8_K* y[NT];
    for (int t = 0; t < NT; ++t) y[t] = (const block_q8_K*) act[t];
    float res[NT];
    for (int r = r0; r < r1; ++r) {
        row_dot<TY, NT, false, VNNI>(w + (size_t) r * row_bytes, n / QK_K, y, res);
        for (int t = 0; t < NT; ++t) out[t][r] = res[t];
    }
}

template <int TY, bool Singleton = false>
void gu_rows_nt(int nt, const uint8_t* blob, size_t gu_row, size_t up_off, int n, const void* const* act,
                float* const* ff, int r0, int r1) {
    switch (nt) {
        case 1: gu_rows<TY, 1, Singleton>(blob, gu_row, up_off, n, act, ff, r0, r1); break;
        case 2: gu_rows<TY, 2, Singleton>(blob, gu_row, up_off, n, act, ff, r0, r1); break;
        case 3: gu_rows<TY, 3, Singleton>(blob, gu_row, up_off, n, act, ff, r0, r1); break;
        case 4: gu_rows<TY, 4, Singleton>(blob, gu_row, up_off, n, act, ff, r0, r1); break;
        case 5: gu_rows<TY, 5, Singleton>(blob, gu_row, up_off, n, act, ff, r0, r1); break;
        case 6: gu_rows<TY, 6, Singleton>(blob, gu_row, up_off, n, act, ff, r0, r1); break;
        case 7: gu_rows<TY, 7, Singleton>(blob, gu_row, up_off, n, act, ff, r0, r1); break;
        default: gu_rows<TY, 8, Singleton>(blob, gu_row, up_off, n, act, ff, r0, r1); break;
    }
}

template <int TY>
void dot_rows_nt(int nt, const uint8_t* w, size_t row_bytes, int n, const void* const* act, float* const* out, int r0,
                 int r1) {
    const bool vnni = cpu_avx512_ok();
    switch (nt) {
        case 1: if (vnni) dot_rows<TY, 1, true>(w, row_bytes, n, act, out, r0, r1); else dot_rows<TY, 1, false>(w, row_bytes, n, act, out, r0, r1); break;
        case 2: if (vnni) dot_rows<TY, 2, true>(w, row_bytes, n, act, out, r0, r1); else dot_rows<TY, 2, false>(w, row_bytes, n, act, out, r0, r1); break;
        case 3: if (vnni) dot_rows<TY, 3, true>(w, row_bytes, n, act, out, r0, r1); else dot_rows<TY, 3, false>(w, row_bytes, n, act, out, r0, r1); break;
        case 4: if (vnni) dot_rows<TY, 4, true>(w, row_bytes, n, act, out, r0, r1); else dot_rows<TY, 4, false>(w, row_bytes, n, act, out, r0, r1); break;
        default: for (int t0 = 0; t0 < nt; t0 += 4) {
            const int k = nt - t0 < 4 ? nt - t0 : 4;
            dot_rows_nt<TY>(k, w, row_bytes, n, act + t0, out + t0, r0, r1);
        }
    }
}


template <int TY> struct FmtX : Fmt<TY> {};
template <> struct FmtX<16> : Fmt<16> {
    static inline void decode(const uint8_t* b, int j, __m512i& g, __mmask64& m, __m512i& sc) {
        const uint8_t* q = b + 2 + 16 * j;
        const uint32_t a0 = u32(q), a1 = u32(q + 4), b0 = u32(q + 8), b1 = u32(q + 12);
        g = gather ? gather64(_mm256_cvtepu8_epi32(_mm_cvtsi64_si128((long long) ((uint64_t) a0 | ((uint64_t) b0 << 32)))), iq2xxs_grid)
                   : _mm512_set_epi64((long long) iq2xxs_grid[b0 >> 24], (long long) iq2xxs_grid[(b0 >> 16) & 255],
                             (long long) iq2xxs_grid[(b0 >> 8) & 255], (long long) iq2xxs_grid[b0 & 255],
                             (long long) iq2xxs_grid[a0 >> 24], (long long) iq2xxs_grid[(a0 >> 16) & 255],
                             (long long) iq2xxs_grid[(a0 >> 8) & 255], (long long) iq2xxs_grid[a0 & 255]);
        m = sign_mask_from_idx(sign_idx_xxs(a1, b1));
        const int sa = 2 * (int) (a1 >> 28) + 1, sb = 2 * (int) (b1 >> 28) + 1;
        sc = scales2(sa, sb);
    }
};
template <> struct FmtX<17> : Fmt<17> {
    static inline void decode(const uint8_t* b, int j, __m512i& g, __mmask64& m, __m512i& sc) {
        const uint8_t* q = b + 2 + 16 * j;
        uint16_t v[8];
        std::memcpy(v, q, 16);
        g = gather ? gather64(_mm256_and_si256(_mm256_cvtepu16_epi32(_mm_loadu_si128((const __m128i*) q)), _mm256_set1_epi32(511)), iq2xs_grid)
                   : _mm512_set_epi64((long long) iq2xs_grid[v[7] & 511], (long long) iq2xs_grid[v[6] & 511],
                             (long long) iq2xs_grid[v[5] & 511], (long long) iq2xs_grid[v[4] & 511],
                             (long long) iq2xs_grid[v[3] & 511], (long long) iq2xs_grid[v[2] & 511],
                             (long long) iq2xs_grid[v[1] & 511], (long long) iq2xs_grid[v[0] & 511]);
        m = sign_mask_from_idx(_mm256_srli_epi32(_mm256_cvtepu16_epi32(_mm_loadu_si128((const __m128i*) q)), 9));
        const uint8_t s0 = b[66 + 2 * j], s1 = b[66 + 2 * j + 1];
        sc = scales4(2 * (s0 & 15) + 1, 2 * (s0 >> 4) + 1, 2 * (s1 & 15) + 1, 2 * (s1 >> 4) + 1);
    }
};
template <> struct FmtX<18> : Fmt<18> {
    static inline void decode(const uint8_t* b, int j, __m512i& g, __mmask64& m, __m512i& sc) {
        const uint8_t* q = b + 2 + 16 * j;
        g = gather ? _mm512_i32gather_epi32(_mm512_cvtepu8_epi32(_mm_loadu_si128((const __m128i*) q)), iq3xxs_grid, 4)
                   : _mm512_set_epi32((int) iq3xxs_grid[q[15]], (int) iq3xxs_grid[q[14]], (int) iq3xxs_grid[q[13]], (int) iq3xxs_grid[q[12]],
                             (int) iq3xxs_grid[q[11]], (int) iq3xxs_grid[q[10]], (int) iq3xxs_grid[q[9]], (int) iq3xxs_grid[q[8]],
                             (int) iq3xxs_grid[q[7]], (int) iq3xxs_grid[q[6]], (int) iq3xxs_grid[q[5]], (int) iq3xxs_grid[q[4]],
                             (int) iq3xxs_grid[q[3]], (int) iq3xxs_grid[q[2]], (int) iq3xxs_grid[q[1]], (int) iq3xxs_grid[q[0]]);
        const uint32_t a = u32(b + 2 + 64 + 8 * j), c = u32(b + 2 + 64 + 8 * j + 4);
        m = sign_mask_from_idx(sign_idx_xxs(a, c));
        const int sa = 2 * (int) (a >> 28) + 1, sb = 2 * (int) (c >> 28) + 1;
        sc = scales2(sa, sb);
    }
};

// ---- the DeepSeek CPU lane's exact rows (iq512x_rows): the same 64-value decode, but summed as ggml's AVX2 dots sum, so a
// token's row is ggml-cpu's vec_dot bit for bit at any number of tokens: the integer sums of a block fold to ggml's eight lanes
// (low half = the even 32-value chunk, high half = the odd one), the block scale is f16(d) * y.d, eight float lanes accumulate
// across blocks, and K * ggml's hsum comes last.  vpdpwssd is the madd and the add in one.
inline __m256i fold256(__m512i a) { return _mm256_add_epi32(_mm512_castsi512_si256(a), _mm512_extracti64x4_epi64(a, 1)); }

inline float hsum8_ggml(__m256 x) {
    __m128 s = _mm_add_ps(_mm256_extractf128_ps(x, 1), _mm256_castps256_ps128(x));
    s = _mm_add_ps(s, _mm_movehl_ps(s, s));
    s = _mm_add_ss(s, _mm_movehdup_ps(s));
    return _mm_cvtss_f32(s);
}

template <int TY, int NT>
inline void row_dot_x(const uint8_t* row, int nblocks, const block_q8_K* const* y, float* res) {
    __m256 accf[NT];
    for (int t = 0; t < NT; ++t) accf[t] = _mm256_setzero_ps();
    const __m512i zero = _mm512_setzero_si512();
    constexpr int A = NT <= 2 ? 2 : 1;
    for (int i = 0; i < nblocks; ++i) {
        const uint8_t* blk = row + (size_t) i * FmtX<TY>::bytes;
        if (prefetch_ahead > 0) {
            const uint8_t* ahead = blk + (size_t) prefetch_ahead;
            _mm_prefetch((const char*) ahead, _MM_HINT_T0);
            _mm_prefetch((const char*) ahead + 64, _MM_HINT_T0);
        }
        __m512i acci[NT][A];
        for (int t = 0; t < NT; ++t)
            for (int a = 0; a < A; ++a) acci[t][a] = _mm512_setzero_si512();
        for (int j = 0; j < 4; ++j) {
            __m512i g, sc;
            __mmask64 m;
            FmtX<TY>::decode(blk, j, g, m, sc);
            for (int t = 0; t < NT; ++t) {
                const __m512i yv = _mm512_loadu_si512((const void*) (y[t][i].qs + 64 * j));
                const __m512i ys = _mm512_mask_sub_epi8(yv, m, zero, yv);
                acci[t][j % A] = _mm512_dpwssd_epi32(acci[t][j % A], _mm512_maddubs_epi16(g, ys), sc);
            }
        }
        if constexpr (A == 2) {
            for (int t = 0; t < NT; ++t) acci[t][0] = _mm512_add_epi32(acci[t][0], acci[t][1]);
        }
        const float dx = h2f(u16(blk));
        for (int t = 0; t < NT; ++t)
            accf[t] = _mm256_fmadd_ps(_mm256_set1_ps(dx * y[t][i].d), _mm256_cvtepi32_ps(fold256(acci[t][0])), accf[t]);
    }
    for (int t = 0; t < NT; ++t) res[t] = FmtX<TY>::K * hsum8_ggml(accf[t]);
}

template <int TY, int NT>
void dot_rows_x(const uint8_t* w, size_t row_bytes, int n, const void* const* act, float* const* out, int r0, int r1) {
    const block_q8_K* y[NT];
    for (int t = 0; t < NT; ++t) y[t] = (const block_q8_K*) act[t];
    float res[NT];
    for (int r = r0; r < r1; ++r) {
        row_dot_x<TY, NT>(w + (size_t) r * row_bytes, n / QK_K, y, res);
        for (int t = 0; t < NT; ++t) out[t][r] = res[t];
    }
}

// many tokens an expert (the prompt): signed int8 weights decoded once per block, activations as u8 (q8 + 128, once a call),
// sum(w * q) = sum(w * (q + 128)) - 128 * sum(w) with the second sum once a weight block; tokens in tiles of 16 and rows in
// groups of 8 so the tile's slice of a block and the group's accumulators stay in L1 (the AVX2 form's layout)
template <int TY>
void dot_rows_wide(const uint8_t* w, size_t row_bytes, int n, const void* const* act, int T, float* const* out, int r0, int r1) {
    constexpr int kTile = 16, kRows = 8;
    const int nb = n / QK_K;
    static thread_local std::vector<uint8_t> ubuf;
    static thread_local std::vector<float> accb;
    ubuf.resize((size_t) (T < kTile ? T : kTile) * n);
    accb.resize((size_t) kRows * kTile * 8);
    const __m512i zero = _mm512_setzero_si512(), one8 = _mm512_set1_epi8(1), flip = _mm512_set1_epi8((char) 0x80);
    for (int t0 = 0; t0 < T; t0 += kTile) {
        const int tt = T - t0 < kTile ? T - t0 : kTile;
        const block_q8_K* y[kTile];
        for (int t = 0; t < tt; ++t) {
            y[t] = (const block_q8_K*) act[t0 + t];
            uint8_t* u = ubuf.data() + (size_t) t * n;
            for (int i = 0; i < nb; ++i)
                for (int j = 0; j < 4; ++j)
                    _mm512_storeu_si512((void*) (u + (size_t) i * 256 + 64 * j),
                                        _mm512_xor_si512(_mm512_loadu_si512((const void*) (y[t][i].qs + 64 * j)), flip));
        }
        for (int rg = r0; rg < r1; rg += kRows) {
            const int rr = r1 - rg < kRows ? r1 - rg : kRows;
            for (int k = 0; k < rr * tt; ++k) _mm256_storeu_ps(&accb[(size_t) k * 8], _mm256_setzero_ps());
            for (int i = 0; i < nb; ++i)
                for (int q = 0; q < rr; ++q) {
                    const uint8_t* blk = w + (size_t) (rg + q) * row_bytes + (size_t) i * FmtX<TY>::bytes;
                    if (prefetch_ahead > 0) {
                        _mm_prefetch((const char*) blk + prefetch_ahead, _MM_HINT_T0);
                        _mm_prefetch((const char*) blk + prefetch_ahead + 64, _MM_HINT_T0);
                    }
                    __m512i w8[4], sc[4];
                    __m512i corr = zero;
                    for (int j = 0; j < 4; ++j) {
                        __m512i g, s;
                        __mmask64 m;
                        FmtX<TY>::decode(blk, j, g, m, s);
                        w8[j] = _mm512_mask_sub_epi8(g, m, zero, g);
                        sc[j] = s;
                        corr = _mm512_dpwssd_epi32(corr, _mm512_maddubs_epi16(one8, w8[j]), s);
                    }
                    const __m256i corr128 = _mm256_slli_epi32(fold256(corr), 7);
                    const float dx = h2f(u16(blk));
                    float* accq = &accb[(size_t) q * tt * 8];
                    for (int t = 0; t < tt; ++t) {
                        const uint8_t* u = ubuf.data() + (size_t) t * n + (size_t) i * 256;
                        __m512i a = zero;
                        for (int j = 0; j < 4; ++j)
                            a = _mm512_dpwssd_epi32(a, _mm512_maddubs_epi16(_mm512_loadu_si512((const void*) (u + 64 * j)), w8[j]), sc[j]);
                        const __m256i s = _mm256_sub_epi32(fold256(a), corr128);
                        float* ap = accq + (size_t) t * 8;
                        _mm256_storeu_ps(ap, _mm256_fmadd_ps(_mm256_set1_ps(dx * y[t][i].d), _mm256_cvtepi32_ps(s), _mm256_loadu_ps(ap)));
                    }
                }
            for (int q = 0; q < rr; ++q)
                for (int t = 0; t < tt; ++t)
                    out[t0 + t][rg + q] = FmtX<TY>::K * hsum8_ggml(_mm256_loadu_ps(&accb[((size_t) q * tt + t) * 8]));
        }
    }
}

template <int TY>
void dot_rows_x_nt(int nt, const uint8_t* w, size_t row_bytes, int n, const void* const* act, float* const* out, int r0, int r1) {
    switch (nt) {
        case 1: dot_rows_x<TY, 1>(w, row_bytes, n, act, out, r0, r1); break;
        case 2: dot_rows_x<TY, 2>(w, row_bytes, n, act, out, r0, r1); break;
        case 3: dot_rows_x<TY, 3>(w, row_bytes, n, act, out, r0, r1); break;
        case 4: dot_rows_x<TY, 4>(w, row_bytes, n, act, out, r0, r1); break;
        case 5: dot_rows_x<TY, 5>(w, row_bytes, n, act, out, r0, r1); break;
        case 6: dot_rows_x<TY, 6>(w, row_bytes, n, act, out, r0, r1); break;
        default: dot_rows_wide<TY>(w, row_bytes, n, act, nt, out, r0, r1); break;
    }
}


// ---- IQ1_M (29), AVX-512, 1..6 tokens: ggml's dot lane for lane.  The grid indices come out of two pdep (as ggml's BMI2 path), the
// +-1 delta signs out of one pext into an 8-bit mask a 64-value step; the signed grid bytes are split into |g| and a negative mask;
// the delta dot is the pair sums of the activation (made once a call) times +-scale.  d is f16 in the top nibbles of the scales.
inline void iq1m_pre(const block_q8_K* y, int nb, int16_t* p) {
    const __m256i one8 = _mm256_set1_epi8(1);
    for (int i = 0; i < nb; ++i)
        for (int c = 0; c < 8; ++c)
            _mm256_storeu_si256((__m256i*) (p + ((size_t) i * 8 + c) * 16),
                                _mm256_maddubs_epi16(one8, _mm256_loadu_si256((const __m256i*) (y[i].qs + 32 * c))));
}

template <int NT>
void rows_m(const uint8_t* w, size_t rb, int n, const block_q8_K* const* y, const int16_t* const* P, float* const* out, int r0, int r1) {
    const int nb = n / QK_K;
    const __m256i mask = _mm256_set1_epi16(0x7), mone = _mm256_set1_epi16(1), mtwo8 = _mm256_set1_epi8(2);
    const __m256i scales_shift = _mm256_set_epi64x(9, 3, 6, 0);
    const __m512i zero = _mm512_setzero_si512();
    // the 32 scale words a 64-weight step (8 each of chunks 4 it .. 4 it + 3) out of the 16 decoded scales (their order: 4 e + j for
    // shift e, word j): one vpermw a step instead of two vpshufb + adds + an insert
    __m512i sidx[4];
    for (int it = 0; it < 4; ++it)
        sidx[it] = _mm512_set_epi16(12 + it, 12 + it, 12 + it, 12 + it, 12 + it, 12 + it, 12 + it, 12 + it, 4 + it, 4 + it, 4 + it, 4 + it, 4 + it, 4 + it,
                                    4 + it, 4 + it, 8 + it, 8 + it, 8 + it, 8 + it, 8 + it, 8 + it, 8 + it, 8 + it, it, it, it, it, it, it, it, it);
    for (int r = r0; r < r1; ++r) {
        __m256 a1[NT], a2[NT];
        for (int t = 0; t < NT; ++t) a1[t] = a2[t] = _mm256_setzero_ps();
        const uint8_t* row = w + (size_t) r * rb;
        for (int i = 0; i < nb; ++i) {
            const uint8_t* blk = row + (size_t) i * 56;
            const uint8_t* qs = blk;
            const uint8_t* qh = blk + 32;
            const uint8_t* scb = blk + 48;
            if (prefetch_ahead > 0) {
                _mm_prefetch((const char*) blk + prefetch_ahead, _MM_HINT_T0);
                _mm_prefetch((const char*) blk + prefetch_ahead + 64, _MM_HINT_T0);
            }
            const uint16_t sc0 = u16(scb), sc1 = u16(scb + 2), sc2 = u16(scb + 4), sc3 = u16(scb + 6);
            const uint16_t d16 = (uint16_t) ((sc0 >> 12) | ((sc1 >> 8) & 0x00f0) | ((sc2 >> 4) & 0x0f00) | (sc3 & 0xf000));
            __m256i scales = _mm256_set1_epi64x((long long) u64(scb));
            scales = _mm256_srlv_epi64(scales, scales_shift);
            scales = _mm256_add_epi16(_mm256_slli_epi16(_mm256_and_si256(scales, mask), 1), mone);
            const __m512i sctab = _mm512_castsi256_si512(scales);
            __m512i s1[NT], s2[NT];
            for (int t = 0; t < NT; ++t) s1[t] = s2[t] = zero;
            for (int it = 0; it < 4; ++it) {
                const uint32_t qh32 = u32(qh + 4 * it);
                const uint64_t p1 = _pdep_u64(u32(qs + 8 * it), 0x00ff00ff00ff00ffULL) | _pdep_u64(qh32 & 0x7777, 0x0f000f000f000f00ULL);
                const uint64_t p2 = _pdep_u64(u32(qs + 8 * it + 4), 0x00ff00ff00ff00ffULL) | _pdep_u64((qh32 >> 16) & 0x7777, 0x0f000f000f000f00ULL);
                uint16_t ix[8];
                std::memcpy(ix, &p1, 8);
                std::memcpy(ix + 4, &p2, 8);
                const __m512i g = _mm512_set_epi64((long long) iq1s_grid[ix[7]], (long long) iq1s_grid[ix[6]], (long long) iq1s_grid[ix[5]],
                                                   (long long) iq1s_grid[ix[4]], (long long) iq1s_grid[ix[3]], (long long) iq1s_grid[ix[2]],
                                                   (long long) iq1s_grid[ix[1]], (long long) iq1s_grid[ix[0]]);
                const __m512i sc = _mm512_permutexvar_epi16(sidx[it], sctab);
                // the delta sign of each 8-weight group (bit 3 of its qh nibble) negates that group's 4 scale words
                const __mmask32 dneg = (__mmask32) (_pdep_u32(_pext_u32(qh32, 0x88888888u), 0x11111111u) * 15u);
                const __m512i ssc = _mm512_mask_sub_epi16(sc, dneg, zero, sc);
                const __m512i ag = _mm512_abs_epi8(g);
                const __mmask64 gneg = _mm512_movepi8_mask(g);
                for (int t = 0; t < NT; ++t) {
                    const __m512i q8 = _mm512_loadu_si512((const void*) (y[t][i].qs + 64 * it));
                    const __m512i pp = _mm512_loadu_si512((const void*) (P[t] + ((size_t) i * 8 + 2 * it) * 16));
                    const __m512i ys = _mm512_mask_sub_epi8(q8, gneg, zero, q8);
                    s1[t] = _mm512_dpwssd_epi32(s1[t], _mm512_maddubs_epi16(ag, ys), sc);
                    s2[t] = _mm512_dpwssd_epi32(s2[t], pp, ssc);
                }
            }
            const float dx = h2f(d16);
            for (int t = 0; t < NT; ++t) {
                const __m256 d = _mm256_set1_ps(y[t][i].d * dx);
                a1[t] = _mm256_fmadd_ps(d, _mm256_cvtepi32_ps(fold256(s1[t])), a1[t]);
                a2[t] = _mm256_fmadd_ps(d, _mm256_cvtepi32_ps(fold256(s2[t])), a2[t]);
            }
        }
        for (int t = 0; t < NT; ++t) out[t][r] = hsum8_ggml(a1[t]) + 0.125f * hsum8_ggml(a2[t]);
    }
}

void rows_m_nt(int nt, const uint8_t* w, size_t rb, int n, const void* const* act, float* const* out, int r0, int r1) {
    const int nb = n / QK_K;
    static thread_local std::vector<int16_t> pre;
    for (int t0 = 0; t0 < nt; t0 += 6) {
        const int k = nt - t0 < 6 ? nt - t0 : 6;
        const block_q8_K* y[6];
        const int16_t* P[6];
        float* o[6];
        pre.resize((size_t) k * n);
        for (int t = 0; t < k; ++t) {
            y[t] = (const block_q8_K*) act[t0 + t];
            iq1m_pre(y[t], nb, pre.data() + (size_t) t * n);
            P[t] = pre.data() + (size_t) t * n;
            o[t] = out[t0 + t];
        }
        switch (k) {
            case 1: rows_m<1>(w, rb, n, y, P, o, r0, r1); break;
            case 2: rows_m<2>(w, rb, n, y, P, o, r0, r1); break;
            case 3: rows_m<3>(w, rb, n, y, P, o, r0, r1); break;
            case 4: rows_m<4>(w, rb, n, y, P, o, r0, r1); break;
            case 5: rows_m<5>(w, rb, n, y, P, o, r0, r1); break;
            default: rows_m<6>(w, rb, n, y, P, o, r0, r1); break;
        }
    }
}

// ---- IQ4_NL (type 20) down rows in 512-bit AVX-512-VNNI lanes: two 32-value blocks (64 weights) decoded per step with
// _mm512_shuffle_epi8 and accumulated with _mm512_dpbusd_epi32.  The low 256-bit half of `acc[t]` holds ggml's `accum1`
// (even blocks) and the high 256-bit half holds `accum2` (odd blocks), so every token's output is bit-for-bit identical
// to ggml_vec_dot_iq4_nl_q8_0 and iq4nl_reference_rows at any group width (1..8 tokens).
template <int NT>
void iq4nl512_rows(const uint8_t* w, size_t row_bytes, int n, const void* const* hq, float* const* out, int r0, int r1) {
    alignas(16) static constexpr uint8_t kAbsValuesIq4nl[16] = {
        127, 104, 83, 65, 49, 35, 22, 10, 1, 13, 25, 38, 53, 69, 89, 113
    };
    const __m512i abs_values = _mm512_broadcast_i32x4(_mm_loadu_si128((const __m128i*) kAbsValuesIq4nl));
    const __m512i m4 = _mm512_set1_epi8(0x0f);
    const __m512i bit3 = _mm512_set1_epi8(0x08);
    const __m512i zero = _mm512_setzero_si512();
    const int nb = n / QK4_NL;
    for (int r = r0; r < r1; ++r) {
        const auto* x = (const block_iq4_nl*) (w + (size_t) r * row_bytes);
        __m512 acc[NT];
        for (int t = 0; t < NT; ++t) acc[t] = _mm512_setzero_ps();
        int ib = 0;
        for (; ib + 1 < nb; ib += 2) {
            if (prefetch_ahead > 0 && (ib & 7) == 0) {
                const char* pf = (const char*) &x[ib] + prefetch_ahead;
                _mm_prefetch(pf, _MM_HINT_T0);
                _mm_prefetch(pf + 64, _MM_HINT_T0);
            }
            const __m256i b0 = _mm256_broadcastsi128_si256(_mm_loadu_si128((const __m128i*) x[ib].qs));
            const __m256i b1 = _mm256_broadcastsi128_si256(_mm_loadu_si128((const __m128i*) x[ib + 1].qs));
            const __m512i b01 = _mm512_inserti64x4(_mm512_castsi256_si512(b0), b1, 1);
            const __m512i nib = _mm512_and_si512(_mm512_mask_srli_epi16(b01, 0xFF00FF00u, b01, 4), m4);
            const __m512i aq = _mm512_shuffle_epi8(abs_values, nib);
            const __mmask64 neg = _mm512_testn_epi8_mask(nib, bit3);
            const float d0 = h2f(u16((const uint8_t*) &x[ib].d));
            const float d1 = h2f(u16((const uint8_t*) &x[ib + 1].d));
            for (int t = 0; t < NT; ++t) {
                const auto* y = (const block_q8_0*) hq[t];
                const __m256i y0 = _mm256_loadu_si256((const __m256i*) y[ib].qs);
                const __m256i y1 = _mm256_loadu_si256((const __m256i*) y[ib + 1].qs);
                const __m512i y01 = _mm512_inserti64x4(_mm512_castsi256_si512(y0), y1, 1);
                const __m512i ys = _mm512_mask_sub_epi8(y01, neg, zero, y01);
                const __m512i p01 = _mm512_dpbusd_epi32(zero, aq, ys);
                const float s0 = h2f(u16((const uint8_t*) &y[ib].d)) * d0;
                const float s1 = h2f(u16((const uint8_t*) &y[ib + 1].d)) * d1;
                const __m512 sd = _mm512_insertf32x8(_mm512_castps256_ps512(_mm256_set1_ps(s0)), _mm256_set1_ps(s1), 1);
                acc[t] = _mm512_fmadd_ps(sd, _mm512_cvtepi32_ps(p01), acc[t]);
            }
        }
        for (int t = 0; t < NT; ++t) {
            const __m256 sum256 = _mm256_add_ps(_mm512_castps512_ps256(acc[t]), _mm512_extractf32x8_ps(acc[t], 1));
            float s = hsum8_ggml(sum256);
            if (ib < nb) {
                const auto* y = (const block_q8_0*) hq[t];
                int a = 0, b = 0;
                for (int j = 0; j < QK4_NL / 2; ++j) {
                    a += kvalues_iq4nl[x[ib].qs[j] & 15] * y[ib].qs[j];
                    b += kvalues_iq4nl[x[ib].qs[j] >> 4] * y[ib].qs[j + QK4_NL / 2];
                }
                const float d = h2f(u16((const uint8_t*) &y[ib].d)) * h2f(u16((const uint8_t*) &x[ib].d));
                s += d * (a + b);
            }
            out[t][r] = s;
        }
    }
}

void hadamard512_impl(float* x, int n, int block, const float* signs) {
    if (signs) {
        for (int i = 0; i < n; i += 16)
            _mm512_storeu_ps(x + i, _mm512_mul_ps(_mm512_loadu_ps(x + i), _mm512_loadu_ps(signs + i)));
    }
    const __m512 vscale = _mm512_set1_ps(1.0f / std::sqrt((float) block));
    for (int b0 = 0; b0 < n; b0 += block) {
        float* v = x + b0;
        // In-register butterflies for len = 1, 2, 4, 8 on each 16-float ZMM vector:
        for (int i = 0; i < block; i += 16) {
            __m512 w = _mm512_loadu_ps(v + i);
            // len = 1: pairs (0,1), (2,3), ...
            const __m512 a1 = _mm512_moveldup_ps(w), c1 = _mm512_movehdup_ps(w);
            w = _mm512_mask_sub_ps(_mm512_add_ps(a1, c1), (__mmask16) 0xAAAAu, a1, c1);
            // len = 2: pairs (0..1, 2..3), (4..5, 6..7), ...
            const __m512 a2 = _mm512_permute_ps(w, 0x44), c2 = _mm512_permute_ps(w, 0xEE);
            w = _mm512_mask_sub_ps(_mm512_add_ps(a2, c2), (__mmask16) 0xCCCCu, a2, c2);
            // len = 4: pairs (0..3, 4..7), (8..11, 12..15)
            const __m512 a4 = _mm512_castpd_ps(_mm512_permutex_pd(_mm512_castps_pd(w), 0x44));
            const __m512 c4 = _mm512_castpd_ps(_mm512_permutex_pd(_mm512_castps_pd(w), 0xEE));
            w = _mm512_mask_sub_ps(_mm512_add_ps(a4, c4), (__mmask16) 0xF0F0u, a4, c4);
            // len = 8: pairs (0..7, 8..15)
            const __m512 a8 = _mm512_shuffle_f32x4(w, w, 0x44);
            const __m512 c8 = _mm512_shuffle_f32x4(w, w, 0xEE);
            w = _mm512_mask_sub_ps(_mm512_add_ps(a8, c8), (__mmask16) 0xFF00u, a8, c8);
            _mm512_storeu_ps(v + i, w);
        }
        for (int len = 16; len < block; len <<= 1)
            for (int i = 0; i < block; i += len << 1)
                for (int j = i; j < i + len; j += 16) {
                    const __m512 a = _mm512_loadu_ps(v + j);
                    const __m512 c = _mm512_loadu_ps(v + j + len);
                    _mm512_storeu_ps(v + j, _mm512_add_ps(a, c));
                    _mm512_storeu_ps(v + j + len, _mm512_sub_ps(a, c));
                }
        for (int i = 0; i < block; i += 16)
            _mm512_storeu_ps(v + i, _mm512_mul_ps(_mm512_loadu_ps(v + i), vscale));
    }
}

}  // namespace

void iq512_gu_rows(int type, const uint8_t* blob, size_t gu_row, size_t up_off, int n, const void* const* act, int nt,
                   float* const* ff, int r0, int r1) {
    switch (type) {
        case 16: gu_rows_nt<16>(nt, blob, gu_row, up_off, n, act, ff, r0, r1); break;
        case 17: gu_rows_nt<17>(nt, blob, gu_row, up_off, n, act, ff, r0, r1); break;
        case 18: gu_rows_nt<18>(nt, blob, gu_row, up_off, n, act, ff, r0, r1); break;
        case 21: gu_rows_nt<21>(nt, blob, gu_row, up_off, n, act, ff, r0, r1); break;
        case 22: gu_rows_nt<22>(nt, blob, gu_row, up_off, n, act, ff, r0, r1); break;
        default: break;
    }
}

// The linked ggml AVX2 singleton order, for the formats validated against its production dot.
void iq512_reference_gu_rows(int type, const uint8_t* blob, size_t gu_row, size_t up_off, int n,
                             const void* const* act, int nt, float* const* ff, int r0, int r1) {
    switch (type) {
        case 18: gu_rows_nt<18, true>(nt, blob, gu_row, up_off, n, act, ff, r0, r1); break;
        case 21: gu_rows_nt<21, true>(nt, blob, gu_row, up_off, n, act, ff, r0, r1); break;
        case 22: gu_rows_nt<22, true>(nt, blob, gu_row, up_off, n, act, ff, r0, r1); break;
    }
}

void iq512_rows(int type, const uint8_t* w, size_t row_bytes, int n, const void* const* act, int nt, float* const* out,
                int r0, int r1) {
    switch (type) {
        case 16: dot_rows_nt<16>(nt, w, row_bytes, n, act, out, r0, r1); break;
        case 17: dot_rows_nt<17>(nt, w, row_bytes, n, act, out, r0, r1); break;
        case 18: dot_rows_nt<18>(nt, w, row_bytes, n, act, out, r0, r1); break;
        case 21: dot_rows_nt<21>(nt, w, row_bytes, n, act, out, r0, r1); break;
        case 22: dot_rows_nt<22>(nt, w, row_bytes, n, act, out, r0, r1); break;
        default: break;
    }
}

static bool has_vpopcnt() {
    static const bool ok = [] {
#if defined(_MSC_VER)
        int r[4];
        __cpuidex(r, 7, 0);
        return ((r[2] >> 14) & 1) != 0;   // AVX512_VPOPCNTDQ
#else
        unsigned a, b, c, d;
        __cpuid_count(7, 0, a, b, c, d);
        return ((c >> 14) & 1u) != 0;
#endif
    }();
    return ok;
}

bool iq512_gather_on() noexcept { return gather; }

bool iq512x_supported(int type) noexcept {
    return has_vpopcnt() && (type == 16 || type == 17 || type == 18 || type == 21 || type == 22 || type == 29);
}

void iq512x_rows(int type, const uint8_t* w, size_t row_bytes, int n, const void* const* act, int nt, float* const* out,
                 int r0, int r1) {
    switch (type) {
        case 16: dot_rows_x_nt<16>(nt, w, row_bytes, n, act, out, r0, r1); break;
        case 17: dot_rows_x_nt<17>(nt, w, row_bytes, n, act, out, r0, r1); break;
        case 18: dot_rows_x_nt<18>(nt, w, row_bytes, n, act, out, r0, r1); break;
        case 21: dot_rows_x_nt<21>(nt, w, row_bytes, n, act, out, r0, r1); break;
        case 22: dot_rows_x_nt<22>(nt, w, row_bytes, n, act, out, r0, r1); break;
        case 29: rows_m_nt(nt, w, row_bytes, n, act, out, r0, r1); break;
        default: break;
    }
}

void iq4nl512_down_rows(const uint8_t* w, size_t row_bytes, int n, const void* const* hq, int nt, float* const* out,
                        int r0, int r1) {
    for (int at = 0; at < nt; at += 4) {
        const int k = nt - at < 4 ? nt - at : 4;
        switch (k) {
            case 1: iq4nl512_rows<1>(w, row_bytes, n, hq + at, out + at, r0, r1); break;
            case 2: iq4nl512_rows<2>(w, row_bytes, n, hq + at, out + at, r0, r1); break;
            case 3: iq4nl512_rows<3>(w, row_bytes, n, hq + at, out + at, r0, r1); break;
            default: iq4nl512_rows<4>(w, row_bytes, n, hq + at, out + at, r0, r1); break;
        }
    }
}

void hadamard_rotate_avx512(float* x, int n, int block, const float* signs) {
    hadamard512_impl(x, n, block, signs);
}

}  // namespace strata::kernels::cpu
