// src/kernels/cpu/iq_avx1_parity.cpp - the AVX1 expert kernels (iq_avx1.cpp) against ggml-cpu's dots and
// against themselves at every token count.  No GPU, no model: random blocks (every grid index and sign
// pattern occurs) and random activations quantized the way the engine quantizes them.
//
//     iq_avx1_parity            IQ3_XXS/IQ3_S/IQ2_S/IQ4_XS gate/up rows and IQ4_NL/Q2_0 down rows:
//                               every row bit for bit the same kernel's one-token row (#152), and rel <= 1e-5
//                               against ggml-cpu's vec_dot (the scalar _generic the engine runs without these).
//     iq_avx1_parity --bench [--mb 128] [--nt 1,2,4] [--cpu N]
//                               ms per expert on one thread: ggml-cpu's per-token dot vs the iq128 kernel over
//                               --mb MB of expert blobs (more than any L3: the weights stream from DRAM as in
//                               decode), for the nt in --nt.
//
// Runs on any CPU with AVX (the kernels are compiled for SSE4.2 + AVX); on an AVX2 machine it checks the
// older-CPU path exactly as an AVX-only machine runs it (the engine's dispatch gate is separate: this calls
// the kernels directly).
#include "strata/kernels/cpu/iq_avx1.hpp"
#include "strata/kernels/cpu/kq_avx1.hpp"
#include "strata/kernels/cpu/native_expert.hpp"
#include "strata/kernels/cpu/expert_layout.hpp"

#define GGML_COMMON_DECL_CPP
#define GGML_COMMON_IMPL_CPP
#include "ggml-common.h"

#include "ggml.h"
#include "ggml-cpu.h"
#include <immintrin.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <map>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#if !defined(_WIN32)
#include <sched.h>
#endif

namespace cpu = strata::kernels::cpu;
using Clock = std::chrono::steady_clock;

namespace {

constexpr int kH = 2560, kFF = 640;   // the experts' geometry: n_embd, expert width (as iq_avx2_parity)
constexpr int kMaxT = 8;

uint64_t mix(uint64_t& s) {   // splitmix64
    uint64_t z = (s += 0x9E3779B97F4A7C15ull);
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ull;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBull;
    return z ^ (z >> 31);
}

const ggml_type_traits_cpu* traits_cpu(int type) { return ggml_get_type_traits_cpu((ggml_type) type); }

/// The gate/up kernel for the type under test: the AVX1 K-quant rows for Q4_K (12) at nt >= 2, the
/// i-quant rows otherwise.  It must pick what native_gu_rows picks, or the bench measures a path the
/// engine never runs.
void gu_rows(int gu, const uint8_t* blob, size_t gu_row, size_t up_off, int n, const void* const* act,
             int nt, float** ff, int r0, int r1) {
    const int kq_min = std::getenv("STRATA_KQ128_ANY") ? 2 : 6;   // keep in step with native_gu_rows' gate (nt >= 6 since the nt=4 A/B; a stale 4 here made the nt=2/4 lines measure a path the engine never runs)
    if (gu == 12 && nt >= kq_min) { cpu::kq_gu_rows_avx1(gu, blob, gu_row, up_off, n, act, nt, ff, r0, r1); return; }
    if (gu == 12) {
        // What native_gu_rows does for Q4_K at nt=1 on this box: ggml-cpu's own AVX1 dot, per token.  The i-quant
        // rung below does not know the type, so calling it here would check the kernel against uninitialized
        // memory instead of against the engine's real reference.
        const ggml_vec_dot_t dot = traits_cpu(gu)->vec_dot;
        for (int t = 0; t < nt; ++t) {
            for (int r = r0; r < r1; ++r) {
                float g = 0.f, u = 0.f;
                dot(n, &g, 0, blob + (size_t) r * gu_row, 0, act[t], 0, 1);
                dot(n, &u, 0, blob + up_off + (size_t) r * gu_row, 0, act[t], 0, 1);
                ff[t][r] = (g / (1.f + std::exp(-g))) * u;
            }
        }
        return;
    }
    cpu::iq128_gu_rows(gu, blob, gu_row, up_off, n, act, nt, ff, r0, r1);
}

// random bytes, then a sane fp16 scale at the front of every block (iq_avx2_parity's)
void fill_blocks(uint8_t* p, size_t bytes, int type, uint64_t& seed) {
    for (size_t i = 0; i < bytes; i += 8) {
        const uint64_t v = mix(seed);
        std::memcpy(p + i, &v, std::min<size_t>(8, bytes - i));
    }
    const size_t bs = ggml_type_size((ggml_type) type);
    for (size_t o = 0; o + bs <= bytes; o += bs) {
        const float d = (0.5f + (float) (mix(seed) >> 40) * (1.f / 16777216.f)) * 1e-3f;
        const ggml_fp16_t h = ggml_fp32_to_fp16(d);
        std::memcpy(p + o, &h, 2);
        if (type == 3 || type == 7 || type == 12) {
            // Q4_1, Q5_1 and Q4_K carry a second fp16 (the min / dmin) in the block header.  Left as the random
            // bytes, half its exponent patterns decode to NaN or Inf, and then ggml-cpu's own dot returns NaN
            // too - the check compares NaN against NaN and fails whatever the kernel under test does (measured:
            // every q4_K row reported "rel nan").  Same fix as the scale above, one line per pair type.
            const ggml_fp16_t hm = ggml_fp32_to_fp16(d * 0.5f);
            std::memcpy(p + o + 2, &hm, 2);
        }
    }
}

void fill_blob(uint8_t* blob, const cpu::NativeFmt& f, uint64_t& seed) {
    fill_blocks(blob, f.down_off, f.gu_type, seed);                        // the gate and up rows
    fill_blocks(blob + f.down_off, f.bytes - f.down_off, f.d_type, seed);  // the down rows
}

struct Case {
    int gu, dn;
    const char* name;
};

const Case kCases[] = {
    {18, 20, "iq3_xxs/iq4_nl"}, {21, 20, "iq3_s/iq4_nl"}, {22, 20, "iq2_s/iq4_nl"}, {23, 20, "iq4_xs/iq4_nl"},
    {18, 42, "iq3_xxs/q2_0"},   {21, 42, "iq3_s/q2_0"},   {22, 42, "iq2_s/q2_0"},   {23, 42, "iq4_xs/q2_0"},
  {12, 20, "q4_K/iq4_nl"},   // UD/AP-Q4_K_XL gate/up: the shape this box decodes
};

// one expert, nt tokens of quantized activations, and the reference rows
struct Fixture {
    cpu::NativeFmt f;
    std::vector<uint8_t> blob;
    std::vector<std::vector<uint8_t>> act, hq;          // per token
    std::vector<std::vector<float>> x, h;
    std::vector<float> ff_ref, out_ref;                 // [t][row]
};

void make_fixture(Fixture& fx, int gu, int dn, int nt, uint64_t& seed) {
    std::string err;
    if (!cpu::native_fmt(gu, dn, kH, kFF, fx.f, err)) { std::fprintf(stderr, "native_fmt: %s\n", err.c_str()); std::exit(1); }
    fx.blob.resize(fx.f.bytes);
    fill_blob(fx.blob.data(), fx.f, seed);
    fx.act.resize(nt); fx.hq.resize(nt); fx.x.resize(nt); fx.h.resize(nt);
    for (int t = 0; t < nt; ++t) {
        fx.act[t].resize(fx.f.act_bytes); fx.hq[t].resize(fx.f.h_bytes);
        fx.x[t].resize(kH); fx.h[t].resize(kFF);
        for (int i = 0; i < kH; ++i) fx.x[t][i] = (float) ((mix(seed) >> 40) * 0x1p-24 * 2.0 - 1.0);
        for (int i = 0; i < kFF; ++i) fx.h[t][i] = (float) ((mix(seed) >> 40) * 0x1p-24 * 2.0 - 1.0);
        cpu::native_quant_act(fx.f, fx.x[t].data(), fx.act[t].data());
        cpu::native_quant_h(fx.f, fx.h[t].data(), fx.hq[t].data());
    }
    // reference rows: ggml-cpu's own dots, per token (what the engine runs without the iq128 kernels)
    const ggml_vec_dot_t gu_dot = traits_cpu(gu)->vec_dot;
    const ggml_vec_dot_t dn_dot = traits_cpu(dn)->vec_dot;
    fx.ff_ref.resize((size_t) nt * kFF); fx.out_ref.resize((size_t) nt * kH);
    for (int t = 0; t < nt; ++t) {
        for (int r = 0; r < kFF; ++r) {
            float g = 0.f, u = 0.f;
            gu_dot(kH, &g, 0, fx.blob.data() + (size_t) r * fx.f.gu_row, 0, fx.act[t].data(), 0, 1);
            gu_dot(kH, &u, 0, fx.blob.data() + fx.f.up_off + (size_t) r * fx.f.gu_row, 0, fx.act[t].data(), 0, 1);
            fx.ff_ref[(size_t) t * kFF + r] = (g / (1.f + std::exp(-g))) * u;
        }
        for (int r = 0; r < kH; ++r) {
            float s = 0.f;
            dn_dot(kFF, &s, 0, fx.blob.data() + fx.f.down_off + (size_t) r * fx.f.d_row, 0, fx.hq[t].data(), 0, 1);
            fx.out_ref[(size_t) t * kH + r] = s;
        }
    }
}

int failures = 0;

// FNV-1a over every row any kernel produced in this run.  The checks above are self-consistency (the same
// kernel at nt and at nt=1) plus rel <= 1e-5 against ggml-cpu - they do not say a new kernel agrees with the
// kernel it replaces to the bit, and the seeds are fixed, so two builds printing the same fingerprint do.
uint64_t row_fp = 1469598103934665603ull;
std::map<std::string, uint64_t> rung_fp;   // per format+side, so a change can be pinned to one rung
std::map<std::string, std::pair<uint64_t, double>> vs_ggml;   // per rung: rows differing bitwise, worst rel
uint64_t fp_of(const char* s, const std::vector<float>& v) {
    uint64_t h = 1469598103934665603ull;
    for (const char* p = s; *p; ++p) { h ^= (unsigned char) *p; h *= 1099511628211ull; }
    for (float f : v) {
        uint32_t b;
        std::memcpy(&b, &f, sizeof b);
        h ^= b & 0xFF; h *= 1099511628211ull;
        h ^= (b >> 8) & 0xFF; h *= 1099511628211ull;
        h ^= (b >> 16) & 0xFF; h *= 1099511628211ull;
        h ^= (b >> 24) & 0xFF; h *= 1099511628211ull;
    }
    return h;
}

void check(const char* what, const std::vector<float>& got, const std::vector<float>& ref, bool exact) {
    const uint64_t h = fp_of(what, got);
    row_fp ^= h; row_fp *= 1099511628211ull;
    std::string key = std::string(what);
    size_t sp = key.find(" nt=");
    if (sp != std::string::npos) key.erase(sp);
    rung_fp[key] ^= h;
    if (exact) {   // bit-exact (iq_avx2_parity's rows_differ)
        size_t d = 0;
        for (size_t i = 0; i < ref.size(); ++i) d += std::memcmp(&got[i], &ref[i], sizeof(float)) != 0;
        if (d) { std::printf("  FAIL %s: %zu of %zu rows differ\n", what, d, ref.size()); ++failures; }
        return;
    }
    // the repo's bar (iq_avx2_parity's rel): aggregate sum|diff| / sum|ref| <= 1e-5
    double num = 0, den = 0;
    for (size_t i = 0; i < ref.size(); ++i) {
        num += std::fabs((double) got[i] - (double) ref[i]);
        den += std::fabs((double) ref[i]);
    }
    const double r = num / (den + 1e-30);
    size_t diff = 0;
    for (size_t i = 0; i < ref.size(); ++i) diff += std::memcmp(&got[i], &ref[i], sizeof(float)) != 0;
    std::string k = std::string(what);
    size_t kp = k.find(" nt=");
    if (kp != std::string::npos) k.erase(kp);
    vs_ggml[k].first += diff;
    if (r > vs_ggml[k].second) vs_ggml[k].second = r;
    if (!(r <= 1e-5)) { std::printf("  FAIL %s: rel %.3e > 1e-5 vs ggml-cpu\n", what, r); ++failures; }
}

// ---- PROTOTYPE (benchmark only, not the engine): expert blocks with the pack-time bit packing undone.
// IQ2_S  (type 22): the 10-bit grid indices already merged into 16-bit words, so the decode stops building
//                   them from qs bytes + a hi_spread broadcast + punpcklbw + four pextrw.
// IQ3_XXS (type 18): its grid indices are plain bytes already (nothing to merge), but one 32-bit word per
//                   32 values carries four 7-bit sign indices plus a 4-bit sub-scale - the unpack is three
//                   shifts and three masks; store them as bytes instead.
// IQ3_S  (type 21): both - 9-bit grid indices merged to 16-bit words, and its four sign bytes stored where
//                   the decode used to run sgn16 (shuffle + and + cmpeq + or) over a loaded u32.
// The engine is untouched: this lives in the parity tool, which synthesises blobs in memory, so there is no
// RAM risk and no restart. If it wins, the real version is an offline repack in the pack build
// (tools/iq_pack.py) plus a second gate/up blob pointer in NativeFmt - NOT a runtime repack cache: the GPU
// tier keeps reading the GGUF, so ExpertCache::fill_slot* is unaffected.
// shortcut: duplicates the float stage and the tables of iq_avx1.cpp; delete it once the engine has rungs.

struct Rp22 {                                    // 106 bytes: one repacked 256-value block
    uint8_t  d[2]; uint16_t idx[32]; uint8_t sign[32]; uint8_t scale[8];
};
struct Rp18 {                                    // 106 bytes
    uint8_t  d[2]; uint8_t q[64]; uint8_t sign[32]; uint8_t scale[8];
};
struct Rp21 {                                    // 170 bytes
    uint8_t  d[2]; uint16_t idx[64]; uint8_t sign[32]; uint8_t scale[8];
};
static_assert(sizeof(Rp22) == 106 && sizeof(Rp18) == 106 && sizeof(Rp21) == 170,
              "repacked block sizes");

struct RpSign8 { uint64_t v[256]; RpSign8() : v() {
    for (unsigned i = 0; i < 256; ++i) { uint64_t w = 0;
        for (int k = 0; k < 8; ++k) w |= (uint64_t) (((i >> k) & 1u) ? 0xFFu : 1u) << (8 * k);
        v[i] = w; } } };
struct RpEvenSigns { uint64_t v[128]; RpEvenSigns() : v() {
    for (int i = 0; i < 128; ++i) { int par = 0;
        for (int k = 0; k < 7; ++k) par ^= (i >> k) & 1;
        const int s = i | (par << 7);            // == ksigns_iq2xs[i]
        uint64_t r = 0;
        for (int k = 0; k < 8; ++k) r |= (uint64_t) (((s >> k) & 1) ? 0xFF : 0x01) << (8 * k);
        v[i] = r; } } };
struct RpK16 { alignas(16) int16_t v[16][16]; RpK16() : v() {
    for (int k = 0; k < 16; ++k) for (int l = 0; l < 16; ++l) v[k][l] = (int16_t) (2 * k + 1); } };
static const RpSign8 rp_sign8{};
static const RpEvenSigns rp_even_signs{};
static const RpK16 rp_k16{};

// fp16 -> fp32, the same bits as iq_avx1.cpp's h2f (fp16 -> fp32 is exact, so any correct conversion matches)
float h2f_local(uint16_t h) {
    const uint32_t e = h & 0x7C00u;
    if (e && e != 0x7C00u) {
        const uint32_t f = ((uint32_t) (h & 0x7FFFu) << 13) + 0x38000000u | (uint32_t) (h & 0x8000u) << 16;
        float out; std::memcpy(&out, &f, 4); return out;
    }
    const uint32_t sign = (uint32_t) (h >> 15) & 1u;
    uint32_t exp = (h >> 10) & 0x1Fu, man = h & 0x3FFu, f;
    if (exp == 0) {
        if (man == 0) { f = sign << 31; }
        else {
            exp = 127 - 15 + 1;
            while (!(man & 0x400u)) { man <<= 1; --exp; }
            man &= 0x3FFu;
            f = (sign << 31) | (exp << 23) | (man << 13);
        }
    } else if (exp == 31) { f = (sign << 31) | 0x7F800000u | (man << 13); }
    else { f = (sign << 31) | ((exp - 15 + 127) << 23) | (man << 13); }
    float out; std::memcpy(&out, &f, 4); return out;
}

// IQ2_S: the 2 high index bits of each of 4 indices live in one qh byte (round 5's repacker, verified
// against hi_spread.iq2s: (hq&3)<<8, (hq&0x0C)<<6, (hq&0x30)<<4, (hq&0xC0)<<2)
struct F22 {
    static constexpr int bytes = 82, groups = 8;
    static constexpr float K = 0.125f;
    static constexpr bool fold = false;          // two sub-block scales per 32 values: no folding
    static void repack(const uint8_t* src, Rp22* dst) {
        std::memcpy(dst->d, src, 2);
        const uint8_t* qs = src + 2, * qh = src + 66;
        for (int h = 0; h < 8; ++h) {
            const int j = h >> 1, half = h & 1;
            const uint8_t hq = qh[2 * j + half];
            const uint8_t* q = qs + 8 * j + 4 * half;
            dst->idx[4 * h + 0] = (uint16_t) (q[0] | ((hq & 0x03) << 8));
            dst->idx[4 * h + 1] = (uint16_t) (q[1] | ((hq & 0x0C) << 6));
            dst->idx[4 * h + 2] = (uint16_t) (q[2] | ((hq & 0x30) << 4));
            dst->idx[4 * h + 3] = (uint16_t) (q[3] | ((hq & 0xC0) << 2));
        }
        std::memcpy(dst->sign,  src + 34, 32);
        std::memcpy(dst->scale, src + 74, 8);
    }
    static inline void decode(const Rp22* b, int h, __m128i& g0, __m128i& g1,
                              __m128i& s0, __m128i& s1, __m128i& c0, __m128i& c1) {
        const uint16_t* ii = b->idx + 4 * h;
        g0 = _mm_set_epi64x((long long) iq2s_grid[ii[1]], (long long) iq2s_grid[ii[0]]);
        g1 = _mm_set_epi64x((long long) iq2s_grid[ii[3]], (long long) iq2s_grid[ii[2]]);
        const uint8_t* sp = b->sign + 4 * h;
        s0 = _mm_set_epi64x((long long) rp_sign8.v[sp[1]], (long long) rp_sign8.v[sp[0]]);
        s1 = _mm_set_epi64x((long long) rp_sign8.v[sp[3]], (long long) rp_sign8.v[sp[2]]);
        const uint8_t sb = b->scale[h];
        c0 = _mm_load_si128((const __m128i*) rp_k16.v[sb & 15]);
        c1 = _mm_load_si128((const __m128i*) rp_k16.v[sb >> 4]);
    }
};

// IQ3_XXS: byte grid indices stay byte indices; the packed sign/scale word becomes bytes
struct F18 {
    static constexpr int bytes = 98, groups = 8;
    static constexpr float K = 0.25f;
    static constexpr bool fold = true;           // one sub-scale per 32 values (the shipped fold relies on it)
    static void repack(const uint8_t* src, Rp18* dst) {
        std::memcpy(dst->d, src, 2);
        std::memcpy(dst->q, src + 2, 64);
        for (int h = 0; h < 8; ++h) {
            uint32_t w;
            std::memcpy(&w, src + 66 + 4 * h, 4);
            dst->sign[4 * h + 0] = (uint8_t) (w & 0x7Fu);
            dst->sign[4 * h + 1] = (uint8_t) ((w >> 7) & 0x7Fu);
            dst->sign[4 * h + 2] = (uint8_t) ((w >> 14) & 0x7Fu);
            dst->sign[4 * h + 3] = (uint8_t) ((w >> 21) & 0x7Fu);
            dst->scale[h] = (uint8_t) (w >> 28);
        }
    }
    static inline void decode(const Rp18* b, int h, __m128i& g0, __m128i& g1,
                              __m128i& s0, __m128i& s1, __m128i& c0, __m128i& c1) {
        const uint8_t* q = b->q + 8 * h;
        g0 = _mm_set_epi32((int) iq3xxs_grid[q[3]], (int) iq3xxs_grid[q[2]],
                           (int) iq3xxs_grid[q[1]], (int) iq3xxs_grid[q[0]]);
        g1 = _mm_set_epi32((int) iq3xxs_grid[q[7]], (int) iq3xxs_grid[q[6]],
                           (int) iq3xxs_grid[q[5]], (int) iq3xxs_grid[q[4]]);
        const uint8_t* sp = b->sign + 4 * h;
        s0 = _mm_set_epi64x((long long) rp_even_signs.v[sp[1]], (long long) rp_even_signs.v[sp[0]]);
        s1 = _mm_set_epi64x((long long) rp_even_signs.v[sp[3]], (long long) rp_even_signs.v[sp[2]]);
        c0 = c1 = _mm_load_si128((const __m128i*) rp_k16.v[b->scale[h]]);
    }
};

// IQ3_S: 9-bit grid indices (qs byte + 1 bit from qh) merged to words, sign bytes stored unpacked,
// the shared sub-scale split per half
struct F21 {
    static constexpr int bytes = 110, groups = 8;
    static constexpr float K = 1.0f;
    static constexpr bool fold = true;
    static void repack(const uint8_t* src, Rp21* dst) {
        std::memcpy(dst->d, src, 2);
        const uint8_t* qs = src + 2, * qh = src + 66;
        for (int h = 0; h < 8; ++h) {
            const uint8_t hb = qh[2 * (h >> 1) + (h & 1)];
            const uint8_t* q = qs + 8 * h;
            for (int k = 0; k < 8; ++k)
                dst->idx[8 * h + k] = (uint16_t) (q[k] | (((hb >> k) & 1u) << 8));
        }
        std::memcpy(dst->sign, src + 74, 32);   // four sign bytes per 32 values, the order sgn16 used
        for (int h = 0; h < 8; ++h) {
            const uint8_t s = src[106 + (h >> 1)];
            dst->scale[h] = (uint8_t) ((h & 1) ? (s >> 4) : (s & 15));
        }
    }
    static inline void decode(const Rp21* b, int h, __m128i& g0, __m128i& g1,
                              __m128i& s0, __m128i& s1, __m128i& c0, __m128i& c1) {
        const uint16_t* ii = b->idx + 8 * h;
        g0 = _mm_set_epi32((int) iq3s_grid[ii[3]], (int) iq3s_grid[ii[2]],
                           (int) iq3s_grid[ii[1]], (int) iq3s_grid[ii[0]]);
        g1 = _mm_set_epi32((int) iq3s_grid[ii[7]], (int) iq3s_grid[ii[6]],
                           (int) iq3s_grid[ii[5]], (int) iq3s_grid[ii[4]]);
        const uint8_t* sp = b->sign + 4 * h;
        s0 = _mm_set_epi64x((long long) rp_sign8.v[sp[1]], (long long) rp_sign8.v[sp[0]]);
        s1 = _mm_set_epi64x((long long) rp_sign8.v[sp[3]], (long long) rp_sign8.v[sp[2]]);
        c0 = c1 = _mm_load_si128((const __m128i*) rp_k16.v[b->scale[h]]);
    }
};

// one repacked row: the mirror of iq_avx1.cpp's row_dot_r<TY, NT, 1> - same accumulators, same order, and
// gate/up are separate calls exactly as gu_rows does them, so register pressure matches production
template <int NT, class B, class F>
void rp_row_dot(const B* row, int nb, const block_q8_K* const* y, float* res) {
    __m128 accf[NT];
    for (int t = 0; t < NT; ++t) accf[t] = _mm_setzero_ps();
    for (int i = 0; i < nb; ++i) {
        __m128i a0[NT], a1[NT];
        for (int t = 0; t < NT; ++t) a0[t] = a1[t] = _mm_setzero_si128();
        const B* blk = row + i;
        for (int h = 0; h < F::groups; ++h) {
            __m128i g0, g1, s0, s1, c0, c1;
            F::decode(blk, h, g0, g1, s0, s1, c0, c1);
            const int off = 32 * h;
            for (int t = 0; t < NT; ++t) {
                const uint8_t* q8 = (const uint8_t*) y[t][i].qs + off;
                if constexpr (F::fold) {
                    const __m128i d = _mm_add_epi16(_mm_maddubs_epi16(g0, _mm_sign_epi8(_mm_loadu_si128((const __m128i*) q8), s0)),
                                                    _mm_maddubs_epi16(g1, _mm_sign_epi8(_mm_loadu_si128((const __m128i*) (q8 + 16)), s1)));
                    a0[t] = _mm_add_epi32(a0[t], _mm_madd_epi16(d, c0));
                } else {
                    a0[t] = _mm_add_epi32(a0[t], _mm_madd_epi16(_mm_maddubs_epi16(g0, _mm_sign_epi8(_mm_loadu_si128((const __m128i*) q8), s0)), c0));
                    a1[t] = _mm_add_epi32(a1[t], _mm_madd_epi16(_mm_maddubs_epi16(g1, _mm_sign_epi8(_mm_loadu_si128((const __m128i*) (q8 + 16)), s1)), c1));
                }
            }
        }
        const uint16_t dh = (uint16_t) (blk->d[0] | (blk->d[1] << 8));
        const float dx = h2f_local(dh) * F::K;
        for (int t = 0; t < NT; ++t) {
            const __m128 f = _mm_cvtepi32_ps(_mm_add_epi32(a0[t], a1[t]));
            accf[t] = _mm_add_ps(accf[t], _mm_mul_ps(_mm_set1_ps(dx * y[t][i].d), f));
        }
    }
    for (int t = 0; t < NT; ++t) {
        __m128 s = _mm_add_ps(accf[t], _mm_movehl_ps(accf[t], accf[t]));
        s = _mm_add_ss(s, _mm_movehdup_ps(s));
        res[t] = _mm_cvtss_f32(s);
    }
}

template <int NT, class B, class F>
void rp_gu_rows(const B* gate, const B* up, int n, const void* const* act, float* const* ff, int nrow) {
    const block_q8_K* y[NT];
    const int nb = n / QK_K;
    for (int t = 0; t < NT; ++t) y[t] = (const block_q8_K*) act[t];
    for (int r = 0; r < nrow; ++r) {
        float g[NT], u[NT];
        rp_row_dot<NT, B, F>(gate + (size_t) r * nb, nb, y, g);
        rp_row_dot<NT, B, F>(up + (size_t) r * nb, nb, y, u);
        for (int t = 0; t < NT; ++t) ff[t][r] = (g[t] / (1.f + std::exp(-g[t]))) * u[t];
    }
}

template <class B, class F>
void rp_gu_nt(int nt, const B* g, const B* u, int n, const void* const* act, float* const* ff, int nrow) {
    switch (nt) {
        case 1: rp_gu_rows<1, B, F>(g, u, n, act, ff, nrow); break;
        case 2: rp_gu_rows<2, B, F>(g, u, n, act, ff, nrow); break;
        case 3: rp_gu_rows<3, B, F>(g, u, n, act, ff, nrow); break;
        case 4: rp_gu_rows<4, B, F>(g, u, n, act, ff, nrow); break;
        case 5: rp_gu_rows<5, B, F>(g, u, n, act, ff, nrow); break;
        case 6: rp_gu_rows<6, B, F>(g, u, n, act, ff, nrow); break;
        case 7: rp_gu_rows<7, B, F>(g, u, n, act, ff, nrow); break;
        default: rp_gu_rows<8, B, F>(g, u, n, act, ff, nrow); break;
    }
}

template <class B, class F>
std::vector<B> rp_build(const std::vector<uint8_t>& blob, size_t per, size_t gu_row, size_t up_off,
                        int nexp, int nb, int nrow) {
    std::vector<B> out((size_t) nexp * 2 * nrow * nb);
    for (int e = 0; e < nexp; ++e) {
        const uint8_t* src = blob.data() + (size_t) e * per;
        B* g = out.data() + (size_t) e * 2 * nrow * nb;
        for (int r = 0; r < nrow; ++r)
            for (int i = 0; i < nb; ++i) {
                F::repack(src + (size_t) r * gu_row + (size_t) i * F::bytes, g + (size_t) r * nb + i);
                F::repack(src + up_off + (size_t) r * gu_row + (size_t) i * F::bytes,
                          g + (size_t) nrow * nb + (size_t) r * nb + i);
            }
    }
    return out;
}

// Round 6 Q5: the same bytes, grouped per 32-value unit instead of per field, so the loop's three reads per
// unit come from one window. Identical size (106 / 170 B) - the question is cache lines, not memory.
// The unit header is unaligned by design; scalar uint16 loads handle that on Ivy Bridge.
struct Rp22u {                                        // 106 bytes
    uint8_t d[2];
    struct { uint16_t idx[4]; uint8_t sign[4]; uint8_t scale; } __attribute__((packed)) u[8];
};
struct Rp21u {                                        // 170 bytes
    uint8_t d[2];
    struct { uint16_t idx[8]; uint8_t sign[4]; uint8_t scale; } __attribute__((packed)) u[8];
};
static_assert(sizeof(Rp22u) == 106 && sizeof(Rp21u) == 170, "unit-contiguous repacked sizes");

struct F22u {
    static constexpr int bytes = 82, groups = 8;
    static constexpr float K = 0.125f;
    static constexpr bool fold = false;
    static void repack(const uint8_t* src, Rp22u* dst) {
        std::memcpy(dst->d, src, 2);
        const uint8_t* qs = src + 2, * qh = src + 66, * sg = src + 34;
        for (int h = 0; h < 8; ++h) {
            const int j = h >> 1, half = h & 1;
            const uint8_t hq = qh[2 * j + half];
            const uint8_t* q = qs + 8 * j + 4 * half;
            dst->u[h].idx[0] = (uint16_t) (q[0] | ((hq & 0x03) << 8));
            dst->u[h].idx[1] = (uint16_t) (q[1] | ((hq & 0x0C) << 6));
            dst->u[h].idx[2] = (uint16_t) (q[2] | ((hq & 0x30) << 4));
            dst->u[h].idx[3] = (uint16_t) (q[3] | ((hq & 0xC0) << 2));
            for (int k = 0; k < 4; ++k) dst->u[h].sign[k] = sg[4 * h + k];
            dst->u[h].scale = src[74 + h];
        }
    }
    static inline void decode(const Rp22u* b, int h, __m128i& g0, __m128i& g1,
                              __m128i& s0, __m128i& s1, __m128i& c0, __m128i& c1) {
        const uint16_t* ii = b->u[h].idx;
        g0 = _mm_set_epi64x((long long) iq2s_grid[ii[1]], (long long) iq2s_grid[ii[0]]);
        g1 = _mm_set_epi64x((long long) iq2s_grid[ii[3]], (long long) iq2s_grid[ii[2]]);
        const uint8_t* sp = b->u[h].sign;
        s0 = _mm_set_epi64x((long long) rp_sign8.v[sp[1]], (long long) rp_sign8.v[sp[0]]);
        s1 = _mm_set_epi64x((long long) rp_sign8.v[sp[3]], (long long) rp_sign8.v[sp[2]]);
        const uint8_t sb = b->u[h].scale;
        c0 = _mm_load_si128((const __m128i*) rp_k16.v[sb & 15]);
        c1 = _mm_load_si128((const __m128i*) rp_k16.v[sb >> 4]);
    }
};

struct F21u {
    static constexpr int bytes = 110, groups = 8;
    static constexpr float K = 1.0f;
    static constexpr bool fold = true;
    static void repack(const uint8_t* src, Rp21u* dst) {
        std::memcpy(dst->d, src, 2);
        const uint8_t* qs = src + 2, * qh = src + 66;
        for (int h = 0; h < 8; ++h) {
            const uint8_t hb = qh[2 * (h >> 1) + (h & 1)];
            const uint8_t* q = qs + 8 * h;
            for (int k = 0; k < 8; ++k) dst->u[h].idx[k] = (uint16_t) (q[k] | (((hb >> k) & 1u) << 8));
            for (int k = 0; k < 4; ++k) dst->u[h].sign[k] = src[74 + 4 * h + k];
            const uint8_t s = src[106 + (h >> 1)];
            dst->u[h].scale = (uint8_t) ((h & 1) ? (s >> 4) : (s & 15));
        }
    }
    static inline void decode(const Rp21u* b, int h, __m128i& g0, __m128i& g1,
                              __m128i& s0, __m128i& s1, __m128i& c0, __m128i& c1) {
        const uint16_t* ii = b->u[h].idx;
        g0 = _mm_set_epi32((int) iq3s_grid[ii[3]], (int) iq3s_grid[ii[2]],
                           (int) iq3s_grid[ii[1]], (int) iq3s_grid[ii[0]]);
        g1 = _mm_set_epi32((int) iq3s_grid[ii[7]], (int) iq3s_grid[ii[6]],
                           (int) iq3s_grid[ii[5]], (int) iq3s_grid[ii[4]]);
        const uint8_t* sp = b->u[h].sign;
        s0 = _mm_set_epi64x((long long) rp_sign8.v[sp[1]], (long long) rp_sign8.v[sp[0]]);
        s1 = _mm_set_epi64x((long long) rp_sign8.v[sp[3]], (long long) rp_sign8.v[sp[2]]);
        c0 = c1 = _mm_load_si128((const __m128i*) rp_k16.v[b->u[h].scale]);
    }
};

// ---- the checks
void run_checks() {
    for (const Case& c : kCases) {
        uint64_t seed = 0x5EED1234u + (uint64_t) c.gu * 1000 + (uint64_t) c.dn;
        Fixture fx;
        make_fixture(fx, c.gu, c.dn, kMaxT, seed);
        const void* actp[kMaxT]; const void* hqp[kMaxT];
        std::vector<float> ff[kMaxT], out[kMaxT];
        float* ffp[kMaxT]; float* outp[kMaxT];
        for (int t = 0; t < kMaxT; ++t) {
            actp[t] = fx.act[t].data(); hqp[t] = fx.hq[t].data();
            ff[t].resize(kFF); out[t].resize(kH);
            ffp[t] = ff[t].data(); outp[t] = out[t].data();
        }
        // gate/up: nt = 1..8, every width bit-identical to that token's nt = 1 row (#152), all rel <= 1e-5 vs ggml
        std::vector<std::vector<float>> ff1(kMaxT), out1(kMaxT);
        // PROTOTYPE: repacked gate/up for the three formats whose decode builds grid indices or unpacks
        // packed sign/scale fields (IQ2_S 20 layers, IQ3_XXS 17, IQ3_S 10; IQ4_XS is codebook-based, skip)
        const size_t rphalf = (size_t) kFF * (size_t) (kH / QK_K);
        std::vector<Rp22> rb22; std::vector<Rp18> rb18; std::vector<Rp21> rb21;
        if (c.gu == 22) rb22 = rp_build<Rp22, F22>(fx.blob, fx.f.bytes, fx.f.gu_row, fx.f.up_off, 1, kH / QK_K, kFF);
        else if (c.gu == 18) rb18 = rp_build<Rp18, F18>(fx.blob, fx.f.bytes, fx.f.gu_row, fx.f.up_off, 1, kH / QK_K, kFF);
        else if (c.gu == 21) rb21 = rp_build<Rp21, F21>(fx.blob, fx.f.bytes, fx.f.gu_row, fx.f.up_off, 1, kH / QK_K, kFF);
        const bool rp_on = (c.gu == 22 || c.gu == 18 || c.gu == 21);
        std::vector<Rp22u> rb22u; std::vector<Rp21u> rb21u;
        if (c.gu == 22) rb22u = rp_build<Rp22u, F22u>(fx.blob, fx.f.bytes, fx.f.gu_row, fx.f.up_off, 1, kH / QK_K, kFF);
        else if (c.gu == 21) rb21u = rp_build<Rp21u, F21u>(fx.blob, fx.f.bytes, fx.f.gu_row, fx.f.up_off, 1, kH / QK_K, kFF);
        const bool rpu_on = (c.gu == 22 || c.gu == 21);
        auto rp_run_unit = [&](int nt, float* const* dst) {
            if (c.gu == 22) rp_gu_nt<Rp22u, F22u>(nt, rb22u.data(), rb22u.data() + rphalf, kH, actp, dst, kFF);
            else rp_gu_nt<Rp21u, F21u>(nt, rb21u.data(), rb21u.data() + rphalf, kH, actp, dst, kFF);
        };
        auto rp_run = [&](int nt, float* const* dst) {
            if (c.gu == 22) rp_gu_nt<Rp22, F22>(nt, rb22.data(), rb22.data() + rphalf, kH, actp, dst, kFF);
            else if (c.gu == 18) rp_gu_nt<Rp18, F18>(nt, rb18.data(), rb18.data() + rphalf, kH, actp, dst, kFF);
            else rp_gu_nt<Rp21, F21>(nt, rb21.data(), rb21.data() + rphalf, kH, actp, dst, kFF);
        };
        for (int nt = 1; nt <= kMaxT; ++nt) {
            gu_rows(c.gu, fx.blob.data(), fx.f.gu_row, fx.f.up_off, kH, actp, nt, ffp, 0, kFF);
            if (rp_on) {
                // the bar for a layout change: bit for bit the rows the deployed kernel writes, no re-baseline
                std::vector<std::vector<float>> ffr(kMaxT, std::vector<float>(kFF)), ffru(kMaxT, std::vector<float>(kFF));
                float* ffrp[kMaxT];
                float* ffrpu[kMaxT];
                for (int t = 0; t < kMaxT; ++t) { ffrp[t] = ffr[t].data(); ffrpu[t] = ffru[t].data(); }
                rp_run(nt, ffrp);
                if (rpu_on) rp_run_unit(nt, ffrpu);
                // exact=true below: every row compared with memcmp, so 0 failures means the repacked kernel
                // writes bit for bit what the deployed kernel writes. Say it once, where an operator reads it.
                if (nt == kMaxT) std::printf("  %-16s gu repacked: bit-identical to the deployed kernel at nt=1..%d%s\n",
                                             c.name, kMaxT, rpu_on ? " (both layouts)" : "");
                for (int t = 0; t < nt; ++t) {
                    char what[96];
                    std::snprintf(what, sizeof what, "%s gu rp vs current nt=%d token %d", c.name, nt, t);
                    check(what, ffr[t], ff[t], true);
                    if (rpu_on) {
                        std::snprintf(what, sizeof what, "%s gu rp-unit vs current nt=%d token %d", c.name, nt, t);
                        check(what, ffru[t], ff[t], true);
                    }
                }
            }
            if (nt == 2 && c.gu == 22) {   // the two-token phased shape against the shipped two-token path
                std::vector<std::vector<float>> ff6(kMaxT, std::vector<float>(kFF));
                float* ff6p[kMaxT];
                for (int t = 0; t < kMaxT; ++t) ff6p[t] = ff6[t].data();
                cpu::iq128_gu_rows_mode(c.gu, fx.blob.data(), fx.f.gu_row, fx.f.up_off, kH, actp, nt, ff6p, 0, kFF, 6);
                for (int t = 0; t < nt; ++t) {
                    char what[96];
                    std::snprintf(what, sizeof what, "%s gu phased nt=2 token %d", c.name, t);
                    check(what, ff6[t], ff[t], true);
                }
            }
            if (nt == 1 && (c.gu == 18 || c.gu == 21 || c.gu == 22 || c.gu == 23)) {
                // every row pairing must write bit for bit what the shipped one writes (exact = memcmp)
                std::vector<std::vector<float>> ffm(kMaxT, std::vector<float>(kFF));
                float* ffmp[kMaxT];
                for (int t = 0; t < kMaxT; ++t) ffmp[t] = ffm[t].data();
                for (int mode : {0, 2, 3, 4, 5, 7, 8, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20}) {
                    if (mode == 4 && c.gu != 22) continue;   // the phased shapes read one format's layout
                    if (mode == 8 && c.gu != 22) continue;
                    if (mode == 5 && c.gu != 21) continue;
                    if (mode == 7 && (c.gu != 21 && c.gu != 18)) continue;
                    if ((mode == 10 || mode == 11 || mode == 12) && c.gu != 18 && c.gu != 21 && c.gu != 22) continue;
                    if ((mode == 13 || mode == 15) && c.gu != 22) continue;   // IQ2_S-only shapes
                    if (mode == 14 && c.gu != 21) continue;                     // the dedup shape reads IQ3_S's layout
                    if (mode == 16 && c.gu != 21 && c.gu != 18) continue;   // index pre-issue: IQ3_S, and IQ3_XXS from round 13
                    if (mode == 17 && c.gu != 18) continue;                     // the round-10 shapes are format-specific
                    if (mode == 18 && c.gu != 22) continue;
                    if (mode == 19 && c.gu != 21) continue;
                    if (mode == 20 && c.gu != 22) continue;   // the affine-codebook shape reads IQ2_S's layout
                    if (mode == 6 && (c.gu != 22 || nt != 1)) continue;   // two-token shape: checked at nt=2 below
                    cpu::iq128_gu_rows_mode(c.gu, fx.blob.data(), fx.f.gu_row, fx.f.up_off, kH, actp, nt, ffmp, 0, kFF, mode);
                    char what[96];
                    std::snprintf(what, sizeof what, "%s gu pairing mode=%d nt=1", c.name, mode);
                    check(what, ffm[0], ff[0], true);
                }
            }
            for (int t = 0; t < nt; ++t) {
                char what[96];
                std::snprintf(what, sizeof what, "%s gu nt=%d vs ggml", c.name, nt);
                check(what, ff[t], std::vector<float>(fx.ff_ref.begin() + (size_t) t * kFF,
                                                      fx.ff_ref.begin() + (size_t) (t + 1) * kFF), false);
                if (nt == 1) ff1[t] = ff[t];
                else {
                    std::snprintf(what, sizeof what, "%s gu nt=%d token %d vs its nt=1 row", c.name, nt, t);
                    check(what, ff[t], ff1[t], true);
                }
            }
        }
        // down
        for (int nt = 1; nt <= kMaxT; ++nt) {
            cpu::iq128_down_rows(c.dn, fx.blob.data() + fx.f.down_off, fx.f.d_row, kFF, hqp, nt, outp, 0, kH);
            if (nt == 1 && c.dn == 20) {   // the 2-row down shape against the shipped down path
                std::vector<std::vector<float>> ot9(kMaxT, std::vector<float>(kH));
                float* o9p[kMaxT];
                for (int t = 0; t < kMaxT; ++t) o9p[t] = ot9[t].data();
                cpu::iq128_down_rows_mode(c.dn, fx.blob.data() + fx.f.down_off, fx.f.d_row, kFF, hqp, nt, o9p, 0, kH, 9);
                char w9[96];
                std::snprintf(w9, sizeof w9, "%s dn pair2 nt=1", c.name);
                check(w9, ot9[0], out[0], true);
            }
            if (nt == 1 && c.dn == 20) {   // the pabsb down shape against the shipped down path
                std::vector<std::vector<float>> otv(kMaxT, std::vector<float>(kH));
                float* o1p[kMaxT];
                for (int t = 0; t < kMaxT; ++t) o1p[t] = otv[t].data();
                cpu::iq128_down_rows_mode(c.dn, fx.blob.data() + fx.f.down_off, fx.f.d_row, kFF, hqp, nt, o1p, 0, kH, 1);
                char what[96];
                std::snprintf(what, sizeof what, "%s dn pabsb nt=1", c.name);
                check(what, otv[0], out[0], true);
                (void) out1;
            }

            for (int t = 0; t < nt; ++t) {
                char what[96];
                std::snprintf(what, sizeof what, "%s down nt=%d vs ggml", c.name, nt);
                check(what, out[t], std::vector<float>(fx.out_ref.begin() + (size_t) t * kH,
                                                      fx.out_ref.begin() + (size_t) (t + 1) * kH), false);
                if (nt == 1) out1[t] = out[t];
                else {
                    std::snprintf(what, sizeof what, "%s down nt=%d token %d vs its nt=1 row", c.name, nt, t);
                    check(what, out[t], out1[t], true);
                }
            }
        }
        std::printf("  %-16s checked (gu %d, down %d)\n", c.name, c.gu, c.dn);
    }
}

// ---- the bench: ms per expert, the weights streamed from DRAM
void run_bench(int mb, const std::vector<int>& nts) {
    for (const Case& c : kCases) {
        uint64_t seed = 0xBEEF0000u + (uint64_t) c.gu * 1000 + (uint64_t) c.dn;
        Fixture fx;
        make_fixture(fx, c.gu, c.dn, 1, seed);
        const size_t per = fx.f.bytes;
        const int nexp = (int) ((size_t) mb * 1024 * 1024 / per);
        std::vector<uint8_t> blob((size_t) nexp * per);
        for (int e = 0; e < nexp; ++e) std::memcpy(blob.data() + (size_t) e * per, fx.blob.data(), per);
        // vary the blocks a little so the run is not one blob repeated (the same bytes hit the same caches)
        for (int e = 1; e < nexp; ++e) {
            uint64_t s2 = seed + (uint64_t) e * 7919;
            const size_t o = (size_t) e * per;
            for (size_t i = 0; i + 8 <= per; i += 128) {
                const uint64_t v = mix(s2);
                std::memcpy(blob.data() + o + i, &v, 8);   // keep block scales (first 2 bytes of each block)
            }
        }
        for (int nt : nts) {
            if (nt > kMaxT) continue;
            const void* actp[kMaxT]; const void* hqp[kMaxT];
            float* ffp[kMaxT]; float* outp[kMaxT];
            std::vector<std::vector<float>> ff(kMaxT, std::vector<float>(kFF)), out(kMaxT, std::vector<float>(kH));
            Fixture fx8;   // nt tokens of activations
            uint64_t s8 = seed + 0x1234;
            make_fixture(fx8, c.gu, c.dn, nt, s8);
            for (int t = 0; t < nt; ++t) {
                actp[t] = fx8.act[t].data(); hqp[t] = fx8.hq[t].data();
                ffp[t] = ff[t].data(); outp[t] = out[t].data();
            }
            const ggml_vec_dot_t gu_dot = traits_cpu(c.gu)->vec_dot, dn_dot = traits_cpu(c.dn)->vec_dot;
            // ggml-cpu's path (what the engine runs today on this CPU): per-token dots over every row
            double ms_ggml_gu = 0, ms_ggml_dn = 0;
            auto t0 = Clock::now();
            for (int e = 0; e < nexp; ++e) {
                for (int r = 0; r < kFF; ++r) {
                    for (int t = 0; t < nt; ++t) {
                        float g = 0.f, u = 0.f;
                        gu_dot(kH, &g, 0, blob.data() + (size_t) e * per + (size_t) r * fx.f.gu_row, 0, (const uint8_t*) actp[t], 0, 1);
                        gu_dot(kH, &u, 0, blob.data() + (size_t) e * per + fx.f.up_off + (size_t) r * fx.f.gu_row, 0, (const uint8_t*) actp[t], 0, 1);
                        ff[t][r] = (g / (1.f + std::exp(-g))) * u;
                    }
                }
            }
            ms_ggml_gu = std::chrono::duration<double, std::milli>(Clock::now() - t0).count();
            t0 = Clock::now();
            for (int e = 0; e < nexp; ++e) {
                for (int r = 0; r < kH; ++r) {
                    for (int t = 0; t < nt; ++t) {
                        float s = 0.f;
                        dn_dot(kFF, &s, 0, blob.data() + (size_t) e * per + fx.f.down_off + (size_t) r * fx.f.d_row, 0, (const uint8_t*) hqp[t], 0, 1);
                        out[t][r] = s;
                    }
                }
            }
            const double ms_ggml = ms_ggml_gu + (ms_ggml_dn = std::chrono::duration<double, std::milli>(Clock::now() - t0).count());
            double ms_iq_gu = 0, ms_iq_dn = 0;
            t0 = Clock::now();
            for (int e = 0; e < nexp; ++e) {
                gu_rows(c.gu, blob.data() + (size_t) e * per, fx.f.gu_row, fx.f.up_off, kH, actp, nt, ffp, 0, kFF);
            }
            ms_iq_gu = std::chrono::duration<double, std::milli>(Clock::now() - t0).count();
            t0 = Clock::now();
            for (int e = 0; e < nexp; ++e) {
                cpu::iq128_down_rows(c.dn, blob.data() + (size_t) e * per + fx.f.down_off, fx.f.d_row, kFF, hqp, nt, outp, 0, kH);
            }
            const double ms_iq = ms_iq_gu + (ms_iq_dn = std::chrono::duration<double, std::milli>(Clock::now() - t0).count());
            // (the mode-9 "down pair2" A/B of an earlier round is gone: iq128_down_rows took that shape at nt=1, so
            // both sides called iq4nl_down_pair_nt1 and the line printed a self-comparison as if it were a result.)
            if (nt == 1 && c.dn == 20) {   // pabsb down vs shipped down
                double bs = 1e30, bp = 1e30;
                for (int rep = 0; rep < 7; ++rep) {
                    t0 = Clock::now();
                    for (int e = 0; e < nexp; ++e)
                        cpu::iq128_down_rows(c.dn, blob.data() + (size_t) e * per + fx.f.down_off, fx.f.d_row, kFF, hqp, nt, outp, 0, kH);
                    bs = std::min(bs, std::chrono::duration<double, std::milli>(Clock::now() - t0).count());
                    t0 = Clock::now();
                    for (int e = 0; e < nexp; ++e)
                        cpu::iq128_down_rows_mode(c.dn, blob.data() + (size_t) e * per + fx.f.down_off, fx.f.d_row, kFF, hqp, nt, outp, 0, kH, 1);
                    bp = std::min(bp, std::chrono::duration<double, std::milli>(Clock::now() - t0).count());
                }
                std::printf("  %-16s nt=1  down pabsb %.3f vs shipped %.3f = %.3fx (min of 7 alternating)\n",
                            c.name, bp / nexp, bs / nexp, bs / bp);
            }
            if (nt == 2 && c.gu == 22) {   // does the phased shape generalise to two tokens?
                double bs = 1e30, bp = 1e30;
                for (int rep = 0; rep < 5; ++rep) {
                    t0 = Clock::now();
                    for (int e = 0; e < nexp; ++e)
                        gu_rows(c.gu, blob.data() + (size_t) e * per, fx.f.gu_row, fx.f.up_off, kH, actp, nt, ffp, 0, kFF);
                    bs = std::min(bs, std::chrono::duration<double, std::milli>(Clock::now() - t0).count());
                    t0 = Clock::now();
                    for (int e = 0; e < nexp; ++e)
                        cpu::iq128_gu_rows_mode(c.gu, blob.data() + (size_t) e * per, fx.f.gu_row, fx.f.up_off, kH, actp, nt, ffp, 0, kFF, 6);
                    bp = std::min(bp, std::chrono::duration<double, std::milli>(Clock::now() - t0).count());
                }
                std::printf("  %-16s nt=2  phased at 2 tokens: shipped %.3f vs phased %.3f = %.3fx (min of 5 alternating)\n",
                            c.name, bs / nexp, bp / nexp, bs / bp);
            }
            if (nt == 1 && (c.gu == 18 || c.gu == 21 || c.gu == 22 || c.gu == 23)) {   // row pairing / hoist A/B
                // NOTE: the mode-format guards below are a second copy of the list in the parity pass (~line 590).
                // They drift: mode 16 was opened to IQ3_XXS in the parity pass and silently stayed IQ3_S-only here,
                // so the first bench run printed no iq3_xxs mode=16 line at all.  Edit both when adding a mode.
            double b[21] = {1e30, 1e30, 1e30, 1e30, 1e30, 1e30, 1e30, 1e30, 1e30, 1e30, 1e30, 1e30, 1e30, 1e30, 1e30, 1e30, 1e30, 1e30, 1e30, 1e30, 1e30};
                for (int rep = 0; rep < 5; ++rep)
                    for (int mode = 0; mode < 21; ++mode) {
                        if (mode == 8 && c.gu != 22) continue;
                        if (mode == 4 && c.gu != 22) continue;
                        if (mode == 5 && c.gu != 21) continue;
                        if (mode == 6) continue;          // two-token shape, measured at nt=2
                    if (mode == 7 && (c.gu != 21 && c.gu != 18)) continue;
                    if (mode == 6) continue;   // two-token shape: measured at nt=2
                    if ((mode == 10 || mode == 11 || mode == 12) && c.gu != 18 && c.gu != 21 && c.gu != 22) continue;
                        if ((mode == 13 || mode == 15) && c.gu != 22) continue;
                        if (mode == 14 && c.gu != 21) continue;
                        if (mode == 16 && c.gu != 21 && c.gu != 18) continue;
                        if (mode == 17 && c.gu != 18) continue;
                        if (mode == 18 && c.gu != 22) continue;
                        if (mode == 19 && c.gu != 21) continue;
                        if (mode == 20 && c.gu != 22) continue;   // the affine-codebook shape reads IQ2_S's layout
                        t0 = Clock::now();
                        for (int e = 0; e < nexp; ++e)
                            cpu::iq128_gu_rows_mode(c.gu, blob.data() + (size_t) e * per, fx.f.gu_row, fx.f.up_off,
                                                    kH, actp, nt, ffp, 0, kFF, mode);
                        b[mode] = std::min(b[mode], std::chrono::duration<double, std::milli>(Clock::now() - t0).count());
                    }
                std::printf("  %-16s nt=1  pairing old %.3f | pair+hoist %.3f (%.3fx) | hoist %.3f (%.3fx) | R=4 %.3f (%.3fx)",
                            c.name, b[0] / nexp, b[1] / nexp, b[0] / b[1], b[2] / nexp, b[0] / b[2], b[3] / nexp, b[0] / b[3]);
                // the phased16 slot printed b[4] (mode 4, skipped unless IQ2_S): 1e30/nexp garbage.  It means mode 16,
                // and mode 16 only runs for IQ3_S/IQ3_XXS rows - print it only when it was measured.
                if (c.gu == 21 || c.gu == 18)
                    std::printf(" | phased16 %.3f (%.3fx vs old, %.3fx vs pre-campaign pairing)", b[16] / nexp, b[0] / b[16], b[1] / b[16]);
                std::printf(" min of 5 alternating\n");
                if (c.gu == 21)
                    std::printf("  %-16s nt=1  IQ3_S phased-unfolded %.3f vs pre-campaign pairing %.3f = %.3fx (min of 5 alternating)\n",
                                c.name, b[5] / nexp, b[1] / nexp, b[1] / b[5]);
                if (c.gu == 21 || c.gu == 18)
                    std::printf("  %-16s nt=1  IQ3 phased-with-fold %.3f vs pre-campaign pairing %.3f = %.3fx (min of 5 alternating)\n",
                                c.name, b[7] / nexp, b[1] / nexp, b[1] / b[7]);
                if (c.gu == 22)
                    std::printf("  %-16s nt=1  IQ2_S dual-stream %.3f vs shipped mode 13 %.3f = %.3fx (min of 5 alternating)\n",
                                c.name, b[8] / nexp, b[13] / nexp, b[13] / b[8]);
                if (c.gu == 18 || c.gu == 21 || c.gu == 22) {
                    // 'shipped' must be the path production actually dispatches, or every candidate on this line is
                    // measured against the wrong baseline: it used to read (c.gu == 22) ? b[4] : b[7], which went
                    // stale twice - mode 13 for IQ2_S (+3.7% over b[4]) and mode 16 for IQ3_S (+3.9% over b[7]).
                    // An IQ2_S candidate measured against b[4] looks promotable when it is really 3.7% behind.
                    // (b[1] in the older lines above stays what it always was: the pre-campaign shipped pairing.)
                    const double ship = (c.gu == 22) ? b[13] : ((c.gu == 21) ? b[16] : b[7]);
                    std::printf("  %-16s nt=1  packed-loads %.3f (%.3fx) | split-acc %.3f (%.3fx) | both %.3f (%.3fx)  vs shipped %.3f (min of 5 alternating)\n",
                                c.name, b[10] / nexp, ship / b[10], b[11] / nexp, ship / b[11], b[12] / nexp, ship / b[12], ship / nexp);
                    // one line per candidate: a shared multi-column line shifts its values when modes are added
                    for (int m : {13, 14, 15, 16, 17, 18, 19, 20})
                        if (b[m] < 1e29)
                            std::printf("  %-16s nt=1  mode=%-2d %s %.3f (%.3fx) vs shipped %.3f (min of 5 alternating)\n",
                                        c.name, m, m == 13 ? "iq2s-index-pipeline" : m == 14 ? "iq3s-dedup-grid"
                                                             : m == 15 ? "iq2s-scalar-index" : m == 16 ? (c.gu == 18 ? "iq3xxs-index-pipe" : "iq3s-index-pipeline")
                                                          : m == 17 ? "iq3xxs-codebook-pipe" : m == 18 ? "iq2s-codebook-pipe" : m == 19 ? "iq3s-dedup-pipe" : "iq2s-affine-codebook",
                                        b[m] / nexp, ship / b[m], ship / nexp);
                }
            }
            double ms_rp_gu = 0, ms_iq_gu_best = 0, ms_rpu_gu = 0;
            const size_t rphalf = (size_t) kFF * (size_t) (kH / QK_K);
            std::vector<Rp22> rb22; std::vector<Rp18> rb18; std::vector<Rp21> rb21;
            std::vector<Rp22u> rb22u; std::vector<Rp21u> rb21u;
            if (c.gu == 22) rb22 = rp_build<Rp22, F22>(blob, per, fx.f.gu_row, fx.f.up_off, nexp, kH / QK_K, kFF);
            else if (c.gu == 18) rb18 = rp_build<Rp18, F18>(blob, per, fx.f.gu_row, fx.f.up_off, nexp, kH / QK_K, kFF);
            else if (c.gu == 21) rb21 = rp_build<Rp21, F21>(blob, per, fx.f.gu_row, fx.f.up_off, nexp, kH / QK_K, kFF);
            if (c.gu == 22) rb22u = rp_build<Rp22u, F22u>(blob, per, fx.f.gu_row, fx.f.up_off, nexp, kH / QK_K, kFF);
            else if (c.gu == 21) rb21u = rp_build<Rp21u, F21u>(blob, per, fx.f.gu_row, fx.f.up_off, nexp, kH / QK_K, kFF);
            if (c.gu == 22 || c.gu == 18 || c.gu == 21) {   // PROTOTYPE: same weights, fields unpacked at pack time.
                // Alternate the two variants and take the best of 3 - a single shot wanders by 2x at this size.
                auto rp_one = [&](int e) {
                    if (c.gu == 22) { Rp22* g = rb22.data() + (size_t) e * 2 * rphalf; rp_gu_nt<Rp22, F22>(nt, g, g + rphalf, kH, actp, ffp, kFF); }
                    else if (c.gu == 18) { Rp18* g = rb18.data() + (size_t) e * 2 * rphalf; rp_gu_nt<Rp18, F18>(nt, g, g + rphalf, kH, actp, ffp, kFF); }
                    else { Rp21* g = rb21.data() + (size_t) e * 2 * rphalf; rp_gu_nt<Rp21, F21>(nt, g, g + rphalf, kH, actp, ffp, kFF); }
                };
                auto rpu_one = [&](int e) {
                    if (c.gu == 22) { Rp22u* g = rb22u.data() + (size_t) e * 2 * rphalf; rp_gu_nt<Rp22u, F22u>(nt, g, g + rphalf, kH, actp, ffp, kFF); }
                    else { Rp21u* g = rb21u.data() + (size_t) e * 2 * rphalf; rp_gu_nt<Rp21u, F21u>(nt, g, g + rphalf, kH, actp, ffp, kFF); }
                };
                double best_iq = 1e30, best_rp = 1e30, best_rpu = 1e30;
                for (int rep = 0; rep < 3; ++rep) {
                    t0 = Clock::now();
                    for (int e = 0; e < nexp; ++e)
                        gu_rows(c.gu, blob.data() + (size_t) e * per, fx.f.gu_row, fx.f.up_off, kH, actp, nt, ffp, 0, kFF);
                    best_iq = std::min(best_iq, std::chrono::duration<double, std::milli>(Clock::now() - t0).count());
                    t0 = Clock::now();
                    for (int e = 0; e < nexp; ++e) rp_one(e);
                    best_rp = std::min(best_rp, std::chrono::duration<double, std::milli>(Clock::now() - t0).count());
                    if (c.gu == 22 || c.gu == 21) {
                        t0 = Clock::now();
                        for (int e = 0; e < nexp; ++e) rpu_one(e);
                        best_rpu = std::min(best_rpu, std::chrono::duration<double, std::milli>(Clock::now() - t0).count());
                    }
                }
                ms_iq_gu_best = best_iq / nexp; ms_rp_gu = best_rp / nexp;
                if (c.gu == 22 || c.gu == 21) ms_rpu_gu = best_rpu / nexp;   // 18 has no unit layout: 1e30 would print
            }
            std::printf("  %-16s nt=%d  ggml %8.2f ms/expert   iq128 %6.2f ms/expert   %.1fx   [gu %.2f vs %.2f = %.2fx  dn %.2f vs %.2f = %.2fx]\n",
                        c.name, nt, ms_ggml / nexp, ms_iq / nexp, ms_ggml / ms_iq,
                        ms_ggml_gu / nexp, ms_iq_gu / nexp, ms_ggml_gu / ms_iq_gu,
                        ms_ggml_dn / nexp, ms_iq_dn / nexp, ms_ggml_dn / ms_iq_dn);
            if (ms_rp_gu > 0)
                std::printf("  %-16s nt=%d  gu  current %.3f  vs  REPACKED %.3f ms/expert  = %.3fx (best of 3, alternating)\n",
                            c.name, nt, ms_iq_gu_best, ms_rp_gu, ms_iq_gu_best / ms_rp_gu);
            if (ms_rpu_gu > 0)
                std::printf("  %-16s nt=%d  gu  REPACKED %.3f  vs  REPACKED-UNIT %.3f ms/expert  = %.3fx  (vs current %.3fx)\n",
                            c.name, nt, ms_rp_gu, ms_rpu_gu, ms_rp_gu / ms_rpu_gu, ms_iq_gu_best / ms_rpu_gu);
        }
    }
}

}  // namespace

int main(int argc, char** argv) {
    bool bench = false;
    int mb = 128, cpu_pin = -1;
    std::vector<int> nts = {1, 2, 4};
    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        if (a == "--bench") bench = true;
        else if (a == "--mb" && i + 1 < argc) mb = std::atoi(argv[++i]);
        else if (a == "--cpu" && i + 1 < argc) cpu_pin = std::atoi(argv[++i]);
        else if (a == "--nt" && i + 1 < argc) {
            nts.clear();
            const char* p = argv[++i];
            while (*p) { nts.push_back(std::atoi(p)); while (*p && *p != ',') ++p; if (*p == ',') ++p; }
        } else {
            std::printf("usage: iq_avx1_parity [--bench [--mb N] [--nt 1,2,4] [--cpu N]]\n");
            return 2;
        }
    }
#if !defined(_WIN32)
    if (cpu_pin >= 0) {
        cpu_set_t set;
        CPU_ZERO(&set);
        CPU_SET(cpu_pin, &set);
        if (sched_setaffinity(0, sizeof set, &set) != 0) std::fprintf(stderr, "pinning to CPU %d failed\n", cpu_pin);
    }
#endif
    // The IQ3_XXS/IQ3_S fold adds two int16 partial dots before the scale multiply, so 4*127*grid_max has to
    // stay inside int16.  Scan the tables: they pack four magnitudes per uint32 entry in this ggml.
    auto gmax = [](const uint32_t* tab, int n) { int m = 0;
        for (int i = 0; i < n; ++i) for (int s = 0; s < 4; ++s) m = std::max(m, (int) ((tab[i] >> (8 * s)) & 0xFF));
        return m; };
    for (auto pr : { std::pair{"iq3xxs_grid", gmax(iq3xxs_grid, 256)}, std::pair{"iq3s_grid", gmax(iq3s_grid, 512)} })
        if (4 * 127 * pr.second >= 32768) {
            std::printf("  FAIL %s: max magnitude %d, the folded halves would saturate int16\n", pr.first, pr.second);
            ++failures;
        }
    if (!cpu::cpu_avx1_ok()) { std::printf("iq_avx1_parity: no AVX here, the kernels would not run\n"); return 0; }
    std::printf("iq_avx1_parity: %s\n", cpu::cpu_avx1_ok() ? "AVX1 kernels checked (128-bit lanes)" : "");
    if (bench) { run_bench(mb, nts); return failures != 0; }
    run_checks();
    for (const auto& kv : vs_ggml) std::printf("iq_avx1_parity: vs ggml-cpu %-24s %6llu rows differ bitwise, worst rel %.3e\n", kv.first.c_str(), (unsigned long long) kv.second.first, kv.second.second);
    for (const auto& kv : rung_fp) std::printf("iq_avx1_parity: rung fingerprint %-22s %016llx\n", kv.first.c_str(), (unsigned long long) kv.second);
    std::printf("iq_avx1_parity: row fingerprint %016llx\n", (unsigned long long) row_fp);
    std::printf(failures ? "iq_avx1_parity: %d FAILURES\n" : "iq_avx1_parity: all rows match\n", failures);
    return failures != 0;
}
