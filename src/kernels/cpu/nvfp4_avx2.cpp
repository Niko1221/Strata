// src/kernels/cpu/nvfp4_avx2.cpp - see include/strata/kernels/cpu/nvfp4_avx2.hpp.
//
// Per 64-value block, once: its E2M1 values in element order as two 32-byte registers (ggml's reordering: sub-block
// s keeps value j in the low nibble of qs[8 s + j] and value j + 8 in the high one), their magnitudes, and the four
// sub-block scales. Per token, as ggml_vec_dot_nvfp4_q8_0's AVX2 path: vpsignb moves the sign onto the activation,
// maddubs + maddwd give 8 int32 lanes (four per 16-value sub-block), one FMA per 32 values applies
// ue4m3(d[s]) * d_q8, and hsum_float_8's order sums the lanes.
//
// Needs AVX2, FMA and F16C - what cpu_avx2_ok() checks before anything here is called.  Where the CPU also has
// AVX-VNNI (cpu_avxvnni_ok: Intel Alder Lake, Sapphire Rapids and later, AMD Zen 4 and later - the 256-bit copies of the instruction
// the AVX-512 rows already use), the row kernels run a second time from the same source with vpdpbusd standing for
// the maddubs + maddwd pair; see nvfp4_avx2_rows.inl.  Same bits, and nvfp4_cpu_parity checks both forms.
#include "strata/kernels/cpu/nvfp4_avx2.hpp"
#include "strata/kernels/cpu/expert_layout.hpp"   // cpu_avxvnni_ok()

#define GGML_COMMON_DECL_CPP
#define GGML_COMMON_IMPL_CPP
#include "ggml-common.h"

#include <immintrin.h>

#include <cmath>
#include <cstdlib>
#include <cstring>

// STRATA_AVXVNNI (CMake: the compiler has the AVX-VNNI intrinsics), as in q2_avx2.cpp / iq_avx2.cpp
#if !defined(STRATA_AVXVNNI)
#define STRATA_AVXVNNI 0
#endif
#if STRATA_AVXVNNI && (defined(__GNUC__) || defined(__clang__))
#define STRATA_AVXVNNI_FN __attribute__((target("avxvnni")))
#else
#define STRATA_AVXVNNI_FN
#endif

namespace strata::kernels::cpu {
namespace {

// A compile-time table, not a constructor: this file is compiled for a wide ISA, and a namespace-scope object with a
// constructor runs at program start on EVERY CPU - the compiler vectorises the loop with AVX/FMA, and a CPU without them
// (Westmere, Sandy Bridge, ...) died with an illegal instruction before main (#2324).  Powers of two: exact.
constexpr float pow2i(int e) {
    float r = 1.0f;
    for (; e > 0; --e) r *= 2.0f;
    for (; e < 0; ++e) r *= 0.5f;
    return r;
}
struct Ue4m3Table {   // ggml_ue4m3_to_fp32 (ggml-impl.h): halved for the doubled kvalues_fp4, 0x7F read as 0
    float v[256];
    constexpr Ue4m3Table() : v() {
        for (int x = 0; x < 256; ++x) {
            if (x == 0 || x == 0x7F) { v[x] = 0.0f; continue; }
            const int e = (x >> 3) & 0xF, m = x & 0x7;
            const float raw = e == 0 ? (float) m * pow2i(-9) : (1.0f + (float) m / 8.0f) * pow2i(e - 7);
            v[x] = raw * 0.5f;
        }
    }
};
constexpr Ue4m3Table kUe4m3;

// software prefetch distance in bytes (STRATA_NVFP4_PREFETCH, 0 = off), as the AVX-512 rows
// (read on first use, behind the CPU check: a static initialiser here would run on every CPU, see above)
inline int prefetch_distance() {
    static const int v = [] {
        const char* e = std::getenv("STRATA_NVFP4_PREFETCH");
        return e ? std::atoi(e) : 2048;
    }();
    return v;
}

// STRATA_NO_NVFP4_VNNI=1 keeps the AVX2 forms on a CPU that has AVX-VNNI, for A/B runs and parity of THIS kernel
// alone: the engine-wide STRATA_NO_AVXVNNI takes the Q2_0 and i-quant copies down with it, so a token-rate delta
// measured with it says nothing about these rows.  Read on first use, behind the CPU check, like the prefetch
// distance (#2324).
inline bool vnni_env_allows() {
    static const bool on = [] {
        const char* e = std::getenv("STRATA_NO_NVFP4_VNNI");
        return e == nullptr || std::atoi(e) == 0;
    }();
    return on;
}

inline float h2f(uint16_t h) { return _mm_cvtss_f32(_mm_cvtph_ps(_mm_cvtsi32_si128((int) h))); }

// sub-blocks 2 h and 2 h + 1 (16 bytes of qs) as their 32 signed doubled E2M1 values, in element order
inline __m256i values32(const uint8_t* qs, const __m128i kv) {
    const __m128i q = _mm_loadu_si128((const __m128i*) qs);
    const __m128i m4 = _mm_set1_epi8(0x0F);
    const __m128i lo = _mm_shuffle_epi8(kv, _mm_and_si128(q, m4));
    const __m128i hi = _mm_shuffle_epi8(kv, _mm_and_si128(_mm_srli_epi16(q, 4), m4));
    return _mm256_set_m128i(_mm_unpackhi_epi64(lo, hi), _mm_unpacklo_epi64(lo, hi));
}

inline float hsum8(const __m256 x) {   // ggml's hsum_float_8
    __m128 res = _mm256_extractf128_ps(x, 1);
    res = _mm_add_ps(res, _mm256_castps256_ps128(x));
    res = _mm_add_ps(res, _mm_movehl_ps(res, res));
    res = _mm_add_ss(res, _mm_movehdup_ps(res));
    return _mm_cvtss_f32(res);
}

constexpr int kMaxBlocks = 64;   // rows of up to 4096 values (n_embd 2560)
// tokens per pass: the pool's MAXT, so a window is one pass. Spilled accumulators cost less than another pass over
// the weights: one expert's gate+up, 8 tokens, 0.415 ms in passes of 4 + 4, 0.352 in one of 8 (5 6 7 alike)
constexpr int kMaxW = 8;

struct Acts {   // per token and block, the two Q8_0 blocks' scales; built once per call, reused by every row
    int nt = 0, nb = 0;
    const block_q8_0* y[kMaxW] = {};
    float dy[kMaxW * kMaxBlocks * 2];
    Acts(const void* const* act, int nt_, int nb_) : nt(nt_), nb(nb_) {
        for (int t = 0; t < nt; ++t) {
            y[t] = (const block_q8_0*) act[t];
            for (int ib = 0; ib < 2 * nb; ++ib) dy[(size_t) t * 2 * nb + ib] = h2f(y[t][ib].d);
        }
    }
};

// ---- the row kernels, twice: the AVX2 form and the AVX-VNNI one (nvfp4_avx2_rows.inl).  Both are copies of
// ggml's AVX2 arithmetic; the VNNI copy is compiled under target("avxvnni") so the plain copy keeps the file's
// -mavx2, and each namespace is internal, so neither reaches a caller as a weak / COMDAT copy (#795).
namespace plain {
#define STRATA_ROWS_VNNI 0
#define STRATA_ROWS_FN
#include "nvfp4_avx2_rows.inl"
}  // namespace plain
#if STRATA_AVXVNNI
namespace vnni {
#define STRATA_ROWS_VNNI 1
#define STRATA_ROWS_FN STRATA_AVXVNNI_FN
#include "nvfp4_avx2_rows.inl"
}  // namespace vnni
#endif

// windows wider than kMaxW (not from the pool) run in balanced slices of at most kMaxW
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

bool nvfp4_256_vnni_ok() { return STRATA_AVXVNNI && cpu_avxvnni_ok() && vnni_env_allows(); }

void nvfp4_256_gu_rows_v(bool vnni_rows, const uint8_t* blob, size_t gu_row, size_t up_off, int n,
                         const void* const* act, int nt, float* const* ff, int r0, int r1, float s_gate, float s_up) {
    slices(nt, [&](int t0, int w) {
        const Acts a(act + t0, w, n / QK_NVFP4);
        float* const* f = ff + t0;
#if STRATA_AVXVNNI
        if (vnni_rows) {
            vnni::gu_rows_nt(blob, gu_row, up_off, a, w, f, r0, r1, s_gate, s_up);
            return;
        }
#endif
        (void) vnni_rows;
        plain::gu_rows_nt(blob, gu_row, up_off, a, w, f, r0, r1, s_gate, s_up);
    });
}

void nvfp4_256_rows_v(bool vnni_rows, const uint8_t* w, size_t row_bytes, int n, const void* const* act, int nt,
                      float* const* out, int r0, int r1, float scale) {
    slices(nt, [&](int t0, int k) {
        const Acts a(act + t0, k, n / QK_NVFP4);
        float* const* o = out + t0;
#if STRATA_AVXVNNI
        if (vnni_rows) {
            vnni::dot_rows_nt(w, row_bytes, a, k, o, r0, r1, scale);
            return;
        }
#endif
        (void) vnni_rows;
        plain::dot_rows_nt(w, row_bytes, a, k, o, r0, r1, scale);
    });
}

bool nvfp4_256_fits(int n) { return n % QK_NVFP4 == 0 && n / QK_NVFP4 <= kMaxBlocks; }

void nvfp4_256_gu_rows(const uint8_t* blob, size_t gu_row, size_t up_off, int n, const void* const* act, int nt,
                       float* const* ff, int r0, int r1, float s_gate, float s_up) {
    nvfp4_256_gu_rows_v(nvfp4_256_vnni_ok(), blob, gu_row, up_off, n, act, nt, ff, r0, r1, s_gate, s_up);
}

void nvfp4_256_rows(const uint8_t* w, size_t row_bytes, int n, const void* const* act, int nt, float* const* out,
                    int r0, int r1, float scale) {
    nvfp4_256_rows_v(nvfp4_256_vnni_ok(), w, row_bytes, n, act, nt, out, r0, r1, scale);
}

}  // namespace strata::kernels::cpu
