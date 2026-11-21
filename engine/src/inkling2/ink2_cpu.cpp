// src/inkling2/ink2_cpu.cpp - see ink2_cpu.hpp.
//
// Expert slot layout (ink2_engine.cu): [w13 | s13 | w2 | s2].  NVFP4 (layers < mx_first): rows of K/2 bytes, value 2j
// in the low nibble of byte j, then one e4m3 scale per 16 columns per row; the dot product times a per-expert factor.
// MXFP8: rows of K e4m3 bytes, then one ue8m0 exponent per 32 columns per row.  w13 rows: 2j gate j, 2j + 1 up j.
//
// Kernels: AVX2 + FMA + F16C (any x86-64 since Haswell / Zen) and AVX-512 variants, chosen at run time.
//
// int8 activations (NVFP4 layers, opt-in: INK2_CPU_ACT=int8): the activations (x, then the SwiGLU output) quantized
// to int8 per 16 values (scale amax / 127), stored per chunk of C values (64 with AVX-512 VNNI, 32 otherwise) as
// [even values | odd values], so a row's low nibbles meet the first half and its high nibbles the second with no
// interleave.  The codes go through pshufb as doubled int8 + 12 (0..24: unsigned, for vpdpbusd / vpmaddubsw); each
// int32 lane (4 products) starts at -12 * (its 4 activations), which takes the +12 back; then per lane float(sum) *
// weight block scale (the e4m3 byte placed into an fp16, value * 2^-8) * activation block scale (* 128: the codes' 0.5
// and the 2^8) into fp32 accumulators.  KL 6e-4 against the resident run (fp32: 1e-4), and since the fp32 kernel's
// prefetch and vpermps decode no faster (one 1-job visit 65-67 us either way, fable/nh.cpp): not the default.
//
// fp32 path (default; MXFP8 layers always): NVFP4 codes through pshufb as doubled int8 (the 0.5 folded into the
// scale), MXFP8 bytes through fp16 (the GPU kernels' placement: the bits shifted into an fp16, exact, value * 2^-8,
// folded into the scale).  The element values are the GPU's exactly; only the order of the fp32 sums differs.
#include "ink2_cpu.hpp"

#if !defined(__linux__) || !(defined(__x86_64__) || defined(__i386__))
// CBA: Linux x86 only (thread pinning, NUMA placement through sysfs, GCC target attributes); elsewhere --cpu-threads
// is refused (isa() == nullptr) and the experts' host memory is plain memory
#include <cstdlib>
namespace ink2cpu {
int nodes() { return 1; }
void* node_alloc(size_t bytes, int) { return std::malloc(bytes); }
void node_free(void* p, size_t) { std::free(p); }
const char* isa() { return nullptr; }
void start(const Setup&) {}
void stop() {}
}  // namespace ink2cpu
#else

#if defined(__GNUC__) && !defined(__clang__)
#pragma GCC diagnostic ignored "-Wmaybe-uninitialized"   // GCC 12's avx512fintrin.h (_mm512_undefined_*)
#pragma GCC diagnostic ignored "-Wuninitialized"
#endif
#include <immintrin.h>
#include <pthread.h>
#include <sched.h>
#include <sys/mman.h>
#include <sys/syscall.h>
#include <unistd.h>
#include <cerrno>
#include <time.h>

#include <atomic>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <new>
#include <sstream>
#include <string>
#include <thread>
#include <vector>

namespace ink2cpu {
namespace {

// ------------------------------------------------------------------------------------------------ topology
std::vector<int> parse_list(const std::string& s) {   // "0-23,48-71"
    std::vector<int> out;
    std::stringstream ss(s);
    std::string part;
    while (std::getline(ss, part, ',')) {
        if (part.empty()) continue;
        const size_t d = part.find('-');
        const int a = std::atoi(part.c_str()), b = d == std::string::npos ? a : std::atoi(part.c_str() + d + 1);
        for (int i = a; i <= b; ++i) out.push_back(i);
    }
    return out;
}
std::string read_line(const std::string& path) {
    std::ifstream f(path);
    std::string s;
    std::getline(f, s);
    return s;
}
// STRATA_NUMA_POOL=0 (Strata's switch for its own NUMA split): NUMA placement off, one pool over every node's cores,
// one doorbell, expert memory wherever the kernel puts it
bool numa_off() {
    const char* e = std::getenv("STRATA_NUMA_POOL");
    return e && std::string(e) == "0";
}
// physical cores of a node (the first thread of each core), in order; NUMA off: of the whole machine
std::vector<int> node_cores(int node) {
    std::vector<int> cpus = parse_list(read_line(numa_off() ? std::string("/sys/devices/system/cpu/online")
                                                           : "/sys/devices/system/node/node" + std::to_string(node) + "/cpulist"));
    if (cpus.empty() && node == 0) {
        const int n = (int) std::thread::hardware_concurrency();
        for (int i = 0; i < n; ++i) cpus.push_back(i);
    }
    std::vector<int> cores;
    for (int c : cpus) {
        const std::vector<int> sib = parse_list(read_line("/sys/devices/system/cpu/cpu" + std::to_string(c) +
                                                          "/topology/thread_siblings_list"));
        if (sib.empty() || sib[0] == c) cores.push_back(c);
    }
    return cores;
}
void pin(int cpu) {
    cpu_set_t s;
    CPU_ZERO(&s);
    CPU_SET(cpu, &s);
    pthread_setaffinity_np(pthread_self(), sizeof s, &s);
}
void nap_us(long us) {
    timespec t{0, us * 1000};
    nanosleep(&t, nullptr);
}

// ------------------------------------------------------------------------------------------------ tables
float E4H[256];    // e4m3 scale * 0.5 (the doubled NVFP4 codes)
float MXS[256];    // 2^(s - 127) * 2^8 (the fp16 placement's 2^-8)
alignas(16) const int8_t E2M1X2[16] = {0, 1, 2, 3, 4, 6, 8, 12, 0, -1, -2, -3, -4, -6, -8, -12};
alignas(64) const float E2M1X2F[16] = {0, 1, 2, 3, 4, 6, 8, 12, -0.0f, -1, -2, -3, -4, -6, -8, -12};   // as fp32
alignas(16) const uint8_t E2M1X2U[16] = {12, 13, 14, 15, 16, 18, 20, 24, 12, 11, 10, 9, 8, 6, 4, 0};   // + 12
float e4m3_decode(unsigned v) {   // as the GPU's (NaN for 0x7F / 0xFF)
    const int e = (v >> 3) & 15, m = v & 7;
    float r = e == 0 ? (float) m * 0.001953125f : std::ldexp(1.0f + (float) m * 0.125f, e - 7);
    if (e == 15 && m == 7) r = NAN;
    return (v & 0x80) ? -r : r;
}

// ------------------------------------------------------------------------------------------------ fp32 kernels
#define T2 __attribute__((target("avx2,fma,f16c")))
#define T5 __attribute__((target("avx512f,avx512bw,avx512vl,avx2,fma,f16c")))
#define T5V __attribute__((target("avx512f,avx512bw,avx512vl,avx512vnni,avx2,fma,f16c")))

T2 inline float hsum256(__m256 a) {
    const __m128 h = _mm_add_ps(_mm256_castps256_ps128(a), _mm256_extractf128_ps(a, 1));
    const __m128 q = _mm_add_ps(h, _mm_movehl_ps(h, h));
    return _mm_cvtss_f32(_mm_add_ss(q, _mm_movehdup_ps(q)));
}
// 16 bytes = 32 NVFP4 values -> their doubled codes in element order: a = values 0-15, b = 16-31
T2 inline void nv_codes(const uint8_t* p, __m128i lut, __m128i& a, __m128i& b) {
    const __m128i q = _mm_loadu_si128((const __m128i*) p), m = _mm_set1_epi8(0x0F);
    const __m128i lo = _mm_and_si128(q, m), hi = _mm_and_si128(_mm_srli_epi16(q, 4), m);
    a = _mm_shuffle_epi8(lut, _mm_unpacklo_epi8(lo, hi));
    b = _mm_shuffle_epi8(lut, _mm_unpackhi_epi8(lo, hi));
}
// e4m3 bytes (zero-extended in 16-bit lanes) -> the fp16 bits of value * 2^-8
T2 inline __m128i e4m3_h8(__m128i w) {
    return _mm_or_si128(_mm_slli_epi16(_mm_and_si128(w, _mm_set1_epi16(0x80)), 8),
                        _mm_slli_epi16(_mm_and_si128(w, _mm_set1_epi16(0x7F)), 7));
}
T5 inline __m256i e4m3_h16(__m256i w) {
    return _mm256_or_si256(_mm256_slli_epi16(_mm256_and_si256(w, _mm256_set1_epi16(0x80)), 8),
                           _mm256_slli_epi16(_mm256_and_si256(w, _mm256_set1_epi16(0x7F)), 7));
}

constexpr int R = 4;   // rows per step (the activation loads shared)
// INK2_CPU_OPT (bit mask, default all): 1 vpermps NVFP4 kernel, 2 software prefetch, 4 SwiGLU fused into gate_up with h
// written by non-temporal stores, 8 the waiting threads keep AVX-512 warm, 16 start() pins its caller
int g_opt = 31;
// Software prefetch of the next step's rows (R rows ahead: codes, and the scale bytes).  A thread's share of an expert
// is ~100 rows read once: the hardware prefetchers restart at every row group (4 streams + 4 scale streams, 18 lines
// each).  One Cascade Lake core, cold slices (k1.cpp): 3.5 -> 3.9 GB/s (= the L2-resident rate); Zen 3 AVX2: 6.2 -> 7.5.
#define PF_ROWS(w, s, sdiv, step) \
    if (pf && (b & (step - 1)) == 0) { \
        _mm_prefetch((const char*) (w) + pf, _MM_HINT_T0); \
        if ((b & (8 * step - 1)) == 0) _mm_prefetch((const char*) (s) + pf / sdiv, _MM_HINT_T0); \
    }

// y[r] = factor * W[r] . x for r in [r0, r1)
T2 void nv_rows_avx2(const uint8_t* W, int N, int K, const float* x, float factor, float* y, int r0, int r1) {
    const uint8_t* S = W + (size_t) N * (K / 2);
    const size_t pf = g_opt & 2 ? (size_t) R * (K / 2) : 0;
    const __m128i lut = _mm_load_si128((const __m128i*) E2M1X2);
    for (int r = r0; r < r1; r += R) {
        const int nr = r1 - r < R ? r1 - r : R;
        __m256 acc[R];
        for (int i = 0; i < R; ++i) acc[i] = _mm256_setzero_ps();
        for (int b = 0; b < K / 32; ++b) {
            const __m256 x0 = _mm256_loadu_ps(x + 32 * b), x1 = _mm256_loadu_ps(x + 32 * b + 8),
                         x2 = _mm256_loadu_ps(x + 32 * b + 16), x3 = _mm256_loadu_ps(x + 32 * b + 24);
            for (int i = 0; i < nr; ++i) {
                const uint8_t* w = W + (size_t) (r + i) * (K / 2) + 16 * b;
                const uint8_t* s = S + (size_t) (r + i) * (K / 16) + 2 * b;
                PF_ROWS(w, s, 8, 4)
                __m128i a, c;
                nv_codes(w, lut, a, c);
                const __m256 v0 = _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(a)),
                             v1 = _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(_mm_srli_si128(a, 8))),
                             v2 = _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(c)),
                             v3 = _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(_mm_srli_si128(c, 8)));
                const __m256 p0 = _mm256_fmadd_ps(v1, x1, _mm256_mul_ps(v0, x0));
                const __m256 p1 = _mm256_fmadd_ps(v3, x3, _mm256_mul_ps(v2, x2));
                acc[i] = _mm256_fmadd_ps(p0, _mm256_set1_ps(E4H[s[0]]), acc[i]);
                acc[i] = _mm256_fmadd_ps(p1, _mm256_set1_ps(E4H[s[1]]), acc[i]);
            }
        }
        for (int i = 0; i < nr; ++i) y[r + i] = factor * hsum256(acc[i]);
    }
}
T5 void nv_rows_512(const uint8_t* W, int N, int K, const float* x, float factor, float* y, int r0, int r1) {
    const uint8_t* S = W + (size_t) N * (K / 2);
    const size_t pf = g_opt & 2 ? (size_t) R * (K / 2) : 0;
    const __m128i lut = _mm_load_si128((const __m128i*) E2M1X2);
    for (int r = r0; r < r1; r += R) {
        const int nr = r1 - r < R ? r1 - r : R;
        __m512 acc[R];
        for (int i = 0; i < R; ++i) acc[i] = _mm512_setzero_ps();
        for (int b = 0; b < K / 32; ++b) {
            const __m512 x0 = _mm512_loadu_ps(x + 32 * b), x1 = _mm512_loadu_ps(x + 32 * b + 16);
            for (int i = 0; i < nr; ++i) {
                const uint8_t* w = W + (size_t) (r + i) * (K / 2) + 16 * b;
                const uint8_t* s = S + (size_t) (r + i) * (K / 16) + 2 * b;
                PF_ROWS(w, s, 8, 4)
                __m128i a, c;
                nv_codes(w, lut, a, c);
                const __m512 v0 = _mm512_cvtepi32_ps(_mm512_cvtepi8_epi32(a)), v1 = _mm512_cvtepi32_ps(_mm512_cvtepi8_epi32(c));
                acc[i] = _mm512_fmadd_ps(_mm512_mul_ps(v0, x0), _mm512_set1_ps(E4H[s[0]]), acc[i]);
                acc[i] = _mm512_fmadd_ps(_mm512_mul_ps(v1, x1), _mm512_set1_ps(E4H[s[1]]), acc[i]);
            }
        }
        for (int i = 0; i < nr; ++i) y[r + i] = factor * _mm512_reduce_add_ps(acc[i]);
    }
}
// The sums of nv_rows_512 (same element values, same order: bit-identical results) with the codes decoded by one
// vpermps: 8 bytes zero-extended to 8 qwords, q | q << 28 puts the low nibble in the low 4 bits of the even dword and
// the high nibble in those of the odd one, and vpermps reads only those 4 bits of an index.  Cascade Lake, one core,
// L2-resident (k1.cpp): 690 uops per KB on ports 0 + 5 against 862, and the core stays in the AVX-512 "light" licence
// (2.39 GHz all-core bin against 1.99): 6.05 GB/s against 4.03.
T5 void nv_rows_perm(const uint8_t* W, int N, int K, const float* x, float factor, float* y, int r0, int r1) {
    const uint8_t* S = W + (size_t) N * (K / 2);
    const size_t pf = g_opt & 2 ? (size_t) R * (K / 2) : 0;
    const __m512 tab = _mm512_load_ps(E2M1X2F);
    for (int r = r0; r < r1; r += R) {
        const int nr = r1 - r < R ? r1 - r : R;
        __m512 acc[R];
        for (int i = 0; i < R; ++i) acc[i] = _mm512_setzero_ps();
        for (int b = 0; b < K / 32; ++b) {
            const __m512 x0 = _mm512_loadu_ps(x + 32 * b), x1 = _mm512_loadu_ps(x + 32 * b + 16);
            for (int i = 0; i < nr; ++i) {
                const uint8_t* w = W + (size_t) (r + i) * (K / 2) + 16 * b;
                const uint8_t* s = S + (size_t) (r + i) * (K / 16) + 2 * b;
                PF_ROWS(w, s, 8, 4)
                const __m512i q0 = _mm512_cvtepu8_epi64(_mm_loadl_epi64((const __m128i*) w)),
                              q1 = _mm512_cvtepu8_epi64(_mm_loadl_epi64((const __m128i*) (w + 8)));
                const __m512 v0 = _mm512_permutexvar_ps(_mm512_or_si512(q0, _mm512_slli_epi64(q0, 28)), tab),
                             v1 = _mm512_permutexvar_ps(_mm512_or_si512(q1, _mm512_slli_epi64(q1, 28)), tab);
                acc[i] = _mm512_fmadd_ps(_mm512_mul_ps(v0, x0), _mm512_set1_ps(E4H[s[0]]), acc[i]);
                acc[i] = _mm512_fmadd_ps(_mm512_mul_ps(v1, x1), _mm512_set1_ps(E4H[s[1]]), acc[i]);
            }
        }
        for (int i = 0; i < nr; ++i) y[r + i] = factor * _mm512_reduce_add_ps(acc[i]);
    }
}
T2 void mx_rows_avx2(const uint8_t* W, int N, int K, const float* x, float* y, int r0, int r1) {
    const uint8_t* S = W + (size_t) N * K;
    for (int r = r0; r < r1; r += R) {
        const int nr = r1 - r < R ? r1 - r : R;
        __m256 acc[R];
        for (int i = 0; i < R; ++i) acc[i] = _mm256_setzero_ps();
        for (int b = 0; b < K / 32; ++b) {
            const __m256 x0 = _mm256_loadu_ps(x + 32 * b), x1 = _mm256_loadu_ps(x + 32 * b + 8),
                         x2 = _mm256_loadu_ps(x + 32 * b + 16), x3 = _mm256_loadu_ps(x + 32 * b + 24);
            for (int i = 0; i < nr; ++i) {
                const uint8_t* w = W + (size_t) (r + i) * K + 32 * b;
                const __m128i q0 = _mm_loadu_si128((const __m128i*) w), q1 = _mm_loadu_si128((const __m128i*) (w + 16));
                const __m256 v0 = _mm256_cvtph_ps(e4m3_h8(_mm_cvtepu8_epi16(q0))),
                             v1 = _mm256_cvtph_ps(e4m3_h8(_mm_cvtepu8_epi16(_mm_srli_si128(q0, 8)))),
                             v2 = _mm256_cvtph_ps(e4m3_h8(_mm_cvtepu8_epi16(q1))),
                             v3 = _mm256_cvtph_ps(e4m3_h8(_mm_cvtepu8_epi16(_mm_srli_si128(q1, 8))));
                const __m256 p = _mm256_fmadd_ps(v3, x3, _mm256_fmadd_ps(v2, x2, _mm256_fmadd_ps(v1, x1, _mm256_mul_ps(v0, x0))));
                acc[i] = _mm256_fmadd_ps(p, _mm256_set1_ps(MXS[S[(size_t) (r + i) * (K / 32) + b]]), acc[i]);
            }
        }
        for (int i = 0; i < nr; ++i) y[r + i] = hsum256(acc[i]);
    }
}
T5 void mx_rows_512(const uint8_t* W, int N, int K, const float* x, float* y, int r0, int r1) {
    const uint8_t* S = W + (size_t) N * K;
    const size_t pf = g_opt & 2 ? (size_t) R * K : 0;
    for (int r = r0; r < r1; r += R) {
        const int nr = r1 - r < R ? r1 - r : R;
        __m512 acc[R];
        for (int i = 0; i < R; ++i) acc[i] = _mm512_setzero_ps();
        for (int b = 0; b < K / 32; ++b) {
            const __m512 x0 = _mm512_loadu_ps(x + 32 * b), x1 = _mm512_loadu_ps(x + 32 * b + 16);
            for (int i = 0; i < nr; ++i) {
                const uint8_t* w = W + (size_t) (r + i) * K + 32 * b;
                PF_ROWS(w, S + (size_t) (r + i) * (K / 32) + b, 32, 2)
                const __m256i q = _mm256_cvtepu8_epi16(_mm_loadu_si128((const __m128i*) w)),
                              q2 = _mm256_cvtepu8_epi16(_mm_loadu_si128((const __m128i*) (w + 16)));
                const __m512 v0 = _mm512_cvtph_ps(e4m3_h16(q)), v1 = _mm512_cvtph_ps(e4m3_h16(q2));
                const __m512 p = _mm512_fmadd_ps(v1, x1, _mm512_mul_ps(v0, x0));
                acc[i] = _mm512_fmadd_ps(p, _mm512_set1_ps(MXS[S[(size_t) (r + i) * (K / 32) + b]]), acc[i]);
            }
        }
        for (int i = 0; i < nr; ++i) y[r + i] = _mm512_reduce_add_ps(acc[i]);
    }
}

// ------------------------------------------------------------------------------------------------ int8 activations
struct QAct {               // K activations in the q8 kernels' form (see the top)
    const int8_t* q;        // [K]   per chunk of C: even values, then odd values
    const int32_t* init;    // [K/4] per lane (4 q): -12 * their sum
    const float* xs;        // [K/4] per lane: its block's scale * 128
};

// 16 activations (block b of the vector) -> int8 at their places in the chunk layout, and the block's scale d[b]
T2 void q8_block(const float* x, int b, int C, int8_t* q, float* d) {
    const __m256 a = _mm256_loadu_ps(x), c = _mm256_loadu_ps(x + 8), sgn = _mm256_set1_ps(-0.0f);
    const __m256 m = _mm256_max_ps(_mm256_andnot_ps(sgn, a), _mm256_andnot_ps(sgn, c));
    __m128 h = _mm_max_ps(_mm256_castps256_ps128(m), _mm256_extractf128_ps(m, 1));
    h = _mm_max_ps(h, _mm_movehl_ps(h, h));
    h = _mm_max_ss(h, _mm_movehdup_ps(h));
    const float am = _mm_cvtss_f32(h);
    d[b] = am / 127.0f;
    const __m256 id = _mm256_set1_ps(am > 0 ? 127.0f / am : 0.0f);
    const __m256i ia = _mm256_cvtps_epi32(_mm256_mul_ps(a, id)), ic = _mm256_cvtps_epi32(_mm256_mul_ps(c, id));
    // packs work per 128-bit half: [a0-3 c0-3 | a4-7 c4-7] -> [a0-3 a4-7 c0-3 c4-7] -> 16 int8 in order
    const __m256i p16 = _mm256_permute4x64_epi64(_mm256_packs_epi32(ia, ic), 0xD8);
    const __m128i p8 = _mm_packs_epi16(_mm256_castsi256_si128(p16), _mm256_extracti128_si256(p16, 1));
    const __m128i eo = _mm_shuffle_epi8(p8, _mm_setr_epi8(0, 2, 4, 6, 8, 10, 12, 14, 1, 3, 5, 7, 9, 11, 13, 15));
    int8_t* base = q + (size_t) (b / (C / 16)) * C + (b % (C / 16)) * 8;
    _mm_storel_epi64((__m128i*) base, eo);
    _mm_storel_epi64((__m128i*) (base + C / 2), _mm_unpackhi_epi64(eo, eo));
}
// the lanes' starts and scales of K quantized activations (q, block scales d)
T2 void q8_lanes(const int8_t* q, const float* d, int K, int C, int32_t* init, float* xs) {
    const __m256i one8 = _mm256_set1_epi8(1), m12 = _mm256_set1_epi16(-12);
    for (int i = 0; i < K; i += 32)
        _mm256_storeu_si256((__m256i*) (init + i / 4),
                            _mm256_madd_epi16(_mm256_maddubs_epi16(one8, _mm256_loadu_si256((const __m256i*) (q + i))), m12));
    // a chunk's lanes: its even half then its odd half, each two lanes per block
    for (int c = 0, i = 0; c < K / C; ++c)
        for (int half = 0; half < 2; ++half)
            for (int b = 0; b < C / 16; ++b, i += 2) xs[i] = xs[i + 1] = 128.0f * d[c * (C / 16) + b];
}

// CBA: the e4m3 block scales' sign bit is dropped (NVFP4 block scales are amax / 6 / global: non-negative)
T2 void q8_rows_avx2(const uint8_t* W, int N, int K, QAct X, float factor, float* y, int r0, int r1) {
    const uint8_t* S = W + (size_t) N * (K / 2);
    const __m256i lut = _mm256_broadcastsi128_si256(_mm_load_si128((const __m128i*) E2M1X2U));
    const __m128i m4 = _mm_set1_epi8(0x0F), sm = _mm_set1_epi16(0x7F00);
    const __m256i ones = _mm256_set1_epi16(1);
    // word w of the 8 lane scales: high byte = scale byte (w & 3) >> 1, low byte zero
    const __m128i sctl = _mm_setr_epi8(-1, 0, -1, 0, -1, 1, -1, 1, -1, 0, -1, 0, -1, 1, -1, 1);
    for (int r = r0; r < r1; r += R) {
        const int nr = r1 - r < R ? r1 - r : R;
        __m256 acc[R];
        for (int i = 0; i < R; ++i) acc[i] = _mm256_setzero_ps();
        for (int c = 0; c < K / 32; ++c) {
            const __m256i xq = _mm256_loadu_si256((const __m256i*) (X.q + 32 * c));
            const __m256i in = _mm256_loadu_si256((const __m256i*) (X.init + 8 * c));
            const __m256 xs = _mm256_loadu_ps(X.xs + 8 * c);
            for (int i = 0; i < nr; ++i) {
                const uint8_t* w = W + (size_t) (r + i) * (K / 2) + 16 * c;
                uint16_t s2;
                std::memcpy(&s2, S + (size_t) (r + i) * (K / 16) + 2 * c, 2);
                const __m128i q = _mm_loadu_si128((const __m128i*) w);
                const __m128i lo = _mm_and_si128(q, m4), hi = _mm_and_si128(_mm_srli_epi16(q, 4), m4);
                const __m256i wq = _mm256_shuffle_epi8(lut, _mm256_set_m128i(hi, lo));
                const __m256i isum = _mm256_add_epi32(_mm256_madd_epi16(_mm256_maddubs_epi16(wq, xq), ones), in);
                const __m128i sh = _mm_srli_epi16(_mm_and_si128(_mm_shuffle_epi8(_mm_set1_epi16((short) s2), sctl), sm), 1);
                const __m256 sc = _mm256_mul_ps(_mm256_cvtph_ps(sh), xs);
                acc[i] = _mm256_fmadd_ps(_mm256_cvtepi32_ps(isum), sc, acc[i]);
            }
        }
        for (int i = 0; i < nr; ++i) y[r + i] = factor * hsum256(acc[i]);
    }
}
T5V void q8_rows_vnni(const uint8_t* W, int N, int K, QAct X, float factor, float* y, int r0, int r1) {
    const uint8_t* S = W + (size_t) N * (K / 2);
    const __m512i lut = _mm512_broadcast_i32x4(_mm_load_si128((const __m128i*) E2M1X2U));
    const size_t pf = g_opt & 2 ? (size_t) R * (K / 2) : 0;
    const __m256i m4 = _mm256_set1_epi8(0x0F), sm = _mm256_set1_epi16(0x7F00);
    // word w of the 16 lane scales (per 128-bit half, words 0-7 / 8-15): high byte = scale byte (w & 7) >> 1
    const __m256i sctl = _mm256_setr_epi8(-1, 0, -1, 0, -1, 1, -1, 1, -1, 2, -1, 2, -1, 3, -1, 3,
                                          -1, 0, -1, 0, -1, 1, -1, 1, -1, 2, -1, 2, -1, 3, -1, 3);
    for (int r = r0; r < r1; r += R) {
        const int nr = r1 - r < R ? r1 - r : R;
        __m512 acc[R];
        for (int i = 0; i < R; ++i) acc[i] = _mm512_setzero_ps();
        for (int b = 0; b < K / 64; ++b) {
            const int c = b;
            const __m512i xq = _mm512_loadu_si512(X.q + 64 * c);
            const __m512i in = _mm512_loadu_si512(X.init + 16 * c);
            const __m512 xs = _mm512_loadu_ps(X.xs + 16 * c);
            for (int i = 0; i < nr; ++i) {
                const uint8_t* w = W + (size_t) (r + i) * (K / 2) + 32 * c;
                PF_ROWS(w, S + (size_t) (r + i) * (K / 16) + 4 * c, 8, 2)
                int s4;
                std::memcpy(&s4, S + (size_t) (r + i) * (K / 16) + 4 * c, 4);
                const __m256i q = _mm256_loadu_si256((const __m256i*) w);
                const __m256i lo = _mm256_and_si256(q, m4), hi = _mm256_and_si256(_mm256_srli_epi16(q, 4), m4);
                const __m512i wq = _mm512_shuffle_epi8(lut, _mm512_inserti64x4(_mm512_castsi256_si512(lo), hi, 1));
                const __m512i isum = _mm512_dpbusd_epi32(in, wq, xq);
                const __m256i sh = _mm256_srli_epi16(_mm256_and_si256(_mm256_shuffle_epi8(_mm256_set1_epi32(s4), sctl), sm), 1);
                const __m512 sc = _mm512_mul_ps(_mm512_cvtph_ps(sh), xs);
                acc[i] = _mm512_fmadd_ps(_mm512_cvtepi32_ps(isum), sc, acc[i]);
            }
        }
        for (int i = 0; i < nr; ++i) y[r + i] = factor * _mm512_reduce_add_ps(acc[i]);
    }
}

int g_isa = 0;          // fp32 kernels: 2 avx2, 5 avx512
bool g_vnni = false;    // q8 kernels: AVX-512 VNNI (chunks of 64), else AVX2 (chunks of 32)
bool g_q8 = false;      // NVFP4 layers with int8 activations (INK2_CPU_ACT=int8)
int g_C = 32;
void rows(bool nv, const uint8_t* W, int N, int K, const float* x, float factor, float* y, int r0, int r1) {
    if (r0 >= r1) return;
    if (nv) (g_isa == 5 ? (g_opt & 1 ? nv_rows_perm : nv_rows_512) : nv_rows_avx2)(W, N, K, x, factor, y, r0, r1);
    else (g_isa == 5 ? mx_rows_512 : mx_rows_avx2)(W, N, K, x, y, r0, r1);
}
void q8_rows(const uint8_t* W, int N, int K, QAct X, float factor, float* y, int r0, int r1) {
    if (r0 < r1) (g_vnni ? q8_rows_vnni : q8_rows_avx2)(W, N, K, X, factor, y, r0, r1);
}

// ------------------------------------------------------------------------------------------------ pools
Setup S;
inline float swiglu1(float g, float u) {   // as the GPU's
    return (std::tanh(g / S.gate_cap) * S.gate_cap) * (1.0f / (1.0f + std::exp(-g))) * (std::tanh(u / S.up_cap) * S.up_cap);
}

struct Barrier {   // spinning, sense-reversing
    alignas(64) std::atomic<int> count{0};
    alignas(64) std::atomic<int> sense{0};
    int n = 1;
    void wait(int& local) {
        local ^= 1;
        if (count.fetch_add(1, std::memory_order_acq_rel) == n - 1) {
            count.store(0, std::memory_order_relaxed);
            sense.store(local, std::memory_order_release);
        } else {
            while (sense.load(std::memory_order_acquire) != local) _mm_pause();
        }
    }
};

constexpr int K8 = BELL_K, FFX = 1152;   // FFX: FF bound of the buffers (start() checks)
struct alignas(64) Node {
    int id = 0;
    std::vector<int> cpus;
    alignas(64) std::atomic<unsigned> gen{0};   // the sequence number the workers are to run
    Barrier bar;
    // the node's jobs (published by its master before gen)
    int nj = 0, layer = 0;
    int bj[K8], ex[K8];                         // bell job index, expert
    // node-local; written by one thread (x) or by slices (the rest), read by all
    float* x = nullptr;                         // [H]          fp32 path
    int8_t* xq = nullptr;                       // [H]          q8 path: x quantized, and its block scales
    float* xd = nullptr;                        // [H/16]
    float* gu = nullptr;                        // [K8][2 FF]
    float* hh = nullptr;                        // [K8][FF]     fp32 path
    int8_t* hq = nullptr;                       // [K8][FF]     q8 path
    float* hd = nullptr;                        // [K8][FF/16]
    alignas(64) double ph[7] = {};              // trace: the master's ms at its phase ends
    std::vector<std::thread> th;
};
struct Priv {   // a pool thread's copies of the visit's inputs in the kernels' forms (heap, first touched by the thread)
    alignas(64) float x[BELL_H];
    alignas(64) float h[K8 * FFX];
    alignas(64) int8_t qx[BELL_H];
    alignas(64) int32_t ix[BELL_H / 4];
    alignas(64) float sx[BELL_H / 4];
    alignas(64) int8_t qh[K8 * FFX];
    alignas(64) int32_t ih[K8 * FFX / 4];
    alignas(64) float sh[K8 * FFX / 4];
    alignas(64) float warm[16] = {};   // warm512's scratch
};
std::vector<Node*> g_nodes;
std::atomic<bool> g_stop{false};
std::atomic<int> g_ready{0};   // threads started (start() returns once all are: a ring before a thread read its
                               // starting point was taken for that starting point, never acked: the GPU's wait trapped)
bool g_trace = false;
constexpr double IDLE_MS = 20.0;   // spinning this long without work, then naps of NAP_US
constexpr long NAP_US = 100;

double now_ms() {
    timespec t;
    clock_gettime(CLOCK_MONOTONIC, &t);
    return t.tv_sec * 1e3 + t.tv_nsec / 1e6;
}
// a share of n items in units of `align`
void share(int n, int t, int T, int align, int& a, int& b) {
    const int units = (n + align - 1) / align;
    a = std::min(n, (int) ((long) units * t / T) * align);
    b = std::min(n, (int) ((long) units * (t + 1) / T) * align);
}
// a copy of bytes the node's other threads wrote, each thread starting at its own place: the whole pool reading the
// same lines in the same order crawled (8.8 KB in 22 threads: 17 us; staggered: 7.6)
T2 void copy_rot(void* dst, const void* src, size_t bytes, int t, int T) {
    const size_t n = bytes / 32, o = n * t / T;
    __m256i* d = (__m256i*) dst;
    const __m256i* s = (const __m256i*) src;
    for (size_t i = o; i < n; ++i) _mm256_storeu_si256(d + i, _mm256_loadu_si256(s + i));
    for (size_t i = 0; i < o; ++i) _mm256_storeu_si256(d + i, _mm256_loadu_si256(s + i));
    std::memcpy((char*) dst + n * 32, (const char*) src + n * 32, bytes % 32);
}
inline void mark(Node& N, int t, int i) { if (!t && g_trace) N.ph[i] = now_ms(); }

void node_work(Node& N, Priv& P, int t, int& sense) {
    const int T = (int) N.cpus.size(), l = N.layer, H = S.h, FF = S.ff, nj = N.nj, C = g_C;
    const bool nv = l < S.mx_first, q8 = nv && g_q8, fuse = (g_opt & 4) && !q8;
    const size_t partA = nv ? S.nv_partA : S.mx_partA;
    int a, b;
    // private copies of the input: threads reading a buffer others write near hit an 8x slowdown (cpunv.cpp)
    if (q8) {
        copy_rot(P.qx, N.xq, H, t, T);
        q8_lanes(P.qx, N.xd, H, C, P.ix, P.sx);
    } else {
        copy_rot(P.x, N.x, (size_t) H * 4, t, T);
    }
    mark(N, t, 0);
    share(2 * FF, t, T, R, a, b);
    for (int j = 0; j < nj; ++j) {
        const int key = l * S.ne + N.ex[j];
        float* gu = N.gu + (size_t) j * 2 * FF;
        if (q8) q8_rows(S.ex[key], 2 * FF, H, {P.qx, P.ix, P.sx}, S.f13[key], gu, a, b);
        else rows(nv, S.ex[key], 2 * FF, H, P.x, nv ? S.f13[key] : 1.0f, gu, a, b);
        if (fuse) {
            // SwiGLU of the thread's own outputs right away (rows 2i and 2i + 1 are both in its share: R is even), one
            // barrier less.  h by non-temporal stores: every thread of the node reads all of it next, and lines left
            // modified in 22 cores' caches made those 22 copies of 4.6 KB cost ~8 us (bcast2.cpp: 12 us against
            // 0.5 us when the writers bypass their caches)
            float* hh = N.hh + (size_t) j * FF;
            for (int i = a / 2; i < b / 2; ++i) {
                const float v = swiglu1(gu[2 * i], gu[2 * i + 1]);
                int bits;
                std::memcpy(&bits, &v, 4);
                _mm_stream_si32((int*) (hh + i), bits);
            }
        }
    }
    mark(N, t, 1);
    if (fuse) {
        _mm_sfence();
        mark(N, t, 2);
    } else {
        N.bar.wait(sense);
        mark(N, t, 2);
        share(nj * FF / 16, t, T, 1, a, b);   // SwiGLU per block of 16 outputs (q8: quantized as a block)
        for (int blk = a; blk < b; ++blk) {
            alignas(32) float h[16];
            const float* gu = N.gu + 32 * (size_t) blk;
            for (int i = 0; i < 16; ++i) h[i] = swiglu1(gu[2 * i], gu[2 * i + 1]);
            if (q8) q8_block(h, blk, C, N.hq, N.hd);
            else std::memcpy(N.hh + 16 * (size_t) blk, h, sizeof h);
        }
    }
    N.bar.wait(sense);
    mark(N, t, 3);
    if (q8) {
        copy_rot(P.qh, N.hq, (size_t) nj * FF, t, T);
        q8_lanes(P.qh, N.hd, nj * FF, C, P.ih, P.sh);
    } else {
        copy_rot(P.h, N.hh, (size_t) nj * FF * 4, t, T);
    }
    mark(N, t, 4);
    share(H, t, T, R, a, b);
    for (int j = 0; j < nj; ++j) {
        const int key = l * S.ne + N.ex[j];
        float* y = S.bell[N.id]->y[N.bj[j]];
        if (q8) q8_rows(S.ex[key] + partA, H, FF, {P.qh + (size_t) j * FF, P.ih + (size_t) j * FF / 4, P.sh + (size_t) j * FF / 4},
                        S.f2[key], y, a, b);
        else rows(nv, S.ex[key] + partA, H, FF, P.h + (size_t) j * FF, nv ? S.f2[key] : 1.0f, y, a, b);
    }
    mark(N, t, 5);
    N.bar.wait(sense);
    mark(N, t, 6);
}

// A core that has run no 512-bit code for ~1 ms powers its upper vector lanes down and drops its AVX-512 licence; the
// next visit then starts slowly (nh.cpp: a 1-job visit after a gap of 1 ms or more takes +50 us, 58 -> 110).  The
// waiting threads run a few 512-bit FMAs every WARM_SPINS pauses (~0.1 ms) so that a visit always starts warm.
constexpr unsigned WARM_SPINS = 1024;
T5 __attribute__((noinline)) void warm512(float* p) {
    __m512 a = _mm512_loadu_ps(p), b = _mm512_set1_ps(0.999f);
    for (int i = 0; i < 8; ++i) a = _mm512_fmadd_ps(a, b, b);
    _mm512_storeu_ps(p, a);
}
inline void spin(Priv* P, unsigned& n) {
    _mm_pause();
    if ((g_opt & 8) && g_isa == 5 && ++n % WARM_SPINS == 0) warm512(P->warm);
}

void worker(Node* N, int t) {
    pin(N->cpus[t]);
    Priv* P = new Priv;
    int sense = 0;
    unsigned seen = 0, spins = 0;   // gen starts at 0
    g_ready.fetch_add(1);
    double last = now_ms();
    while (!g_stop.load(std::memory_order_relaxed)) {
        const unsigned g = N->gen.load(std::memory_order_acquire);
        if (g == seen) {
            if (now_ms() - last > IDLE_MS) nap_us(NAP_US);
            else spin(P, spins);
            continue;
        }
        seen = g;
        node_work(*N, *P, t, sense);
        last = now_ms();
    }
    delete P;
}

void master(Node* N) {
    pin(N->cpus[0]);
    Priv* P = new Priv;
    InkBell* B = S.bell[N->id];   // the node's own doorbell (local memory)
    int sense = 0;
    unsigned seen = 0, spins = 0;   // the bell is zeroed before start(): ring 0 = nothing yet
    g_ready.fetch_add(1);
    double last = now_ms();
    while (!g_stop.load(std::memory_order_relaxed)) {
        const unsigned seq = B->ring;
        if (seq == seen) {
            if (now_ms() - last > IDLE_MS) nap_us(NAP_US);
            else spin(P, spins);
            continue;
        }
        seen = seq;
        const double t_ring = now_ms();
        std::atomic_thread_fence(std::memory_order_acquire);
        const int n = B->n, l = B->layer;
        int nj = 0;
        for (int j = 0; j < n && j < K8; ++j) {
            const int e = B->e[j];
            if (S.home[l * S.ne + e] != N->id) continue;
            N->bj[nj] = j;
            N->ex[nj] = e;
            ++nj;
        }
        if (nj) {
            N->nj = nj;
            N->layer = l;
            const float* x = (const float*) B->x;
            if (l < S.mx_first && g_q8)
                for (int b = 0; b < S.h / 16; ++b) q8_block(x + 16 * b, b, g_C, N->xq, N->xd);
            else
                std::memcpy(N->x, x, (size_t) S.h * 4);
            N->gen.store(seq, std::memory_order_release);
            node_work(*N, *P, 0, sense);
        }
        std::atomic_thread_fence(std::memory_order_seq_cst);
        B->ack[N->id] = seq;
        last = now_ms();
        if (nj) {   // the node's time per job for the GPU's PCIe share (the ring seen -> the ack, over 16 visits)
            const float us = (float) (1e3 * (last - t_ring) / nj);
            B->us_job = B->us_job > 0.0f ? B->us_job + (us - B->us_job) / 16.0f : us;
        }
        if (g_trace) {   // per node, every 2000 rings: busy visits, us from the ring seen to the ack by jobs per visit,
                         // and the master's phases per busy visit
            static thread_local double cnt[K8 + 1], sum[K8 + 1], ph[7], rings = 0, worst = 0;
            const double d = last - t_ring;
            ++rings;
            if (nj) {
                cnt[nj] += 1;
                sum[nj] += d;
                worst = d > worst ? d : worst;
                for (int i = 0; i < 7; ++i) ph[i] += N->ph[i] - (i ? N->ph[i - 1] : t_ring);
            }
            if (rings == 2000) {
                double busy = 0;
                for (int i = 1; i <= K8; ++i) busy += cnt[i];
                busy = std::max(busy, 1.0);
                char line[512];
                int len = std::snprintf(line, sizeof line, "[cpu] node %d: %.0f busy of 2000, max %.0f us; us by jobs:", N->id,
                                        busy, 1e3 * worst);
                for (int i = 1; i <= K8; ++i)
                    if (cnt[i]) len += std::snprintf(line + len, sizeof line - len, " %d:%.0f(%.0f)", i, 1e3 * sum[i] / cnt[i], cnt[i]);
                len += std::snprintf(line + len, sizeof line - len, "; phases (in, gate_up, wait, swiglu, h, down, wait)");
                for (double v : ph) len += std::snprintf(line + len, sizeof line - len, " %.1f", 1e3 * v / busy);
                std::fprintf(stderr, "%s\n", line);
                for (int i = 0; i <= K8; ++i) cnt[i] = sum[i] = 0;
                for (double& v : ph) v = 0;
                rings = worst = 0;
            }
        }
    }
    delete P;
}

}  // namespace

// ------------------------------------------------------------------------------------------------ API
int nodes() {
    int n = 0;
    while (!read_line("/sys/devices/system/node/node" + std::to_string(n) + "/cpulist").empty()) ++n;
    if (numa_off()) return 1;
    if (const char* e = std::getenv("INK2_CPU_NODES")) n = std::min(n, std::max(1, std::atoi(e)));   // 1: node 0 only
    return std::min(n ? n : 1, 2);   // CBA: two nodes at most (InkBell::ack, Setup::bell); more = widen those arrays
}

void* node_alloc(size_t bytes, int node) {
    void* p = mmap(nullptr, bytes, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    if (p == MAP_FAILED) return nullptr;
    madvise(p, bytes, MADV_HUGEPAGE);
    if (numa_off()) return p;   // first touched by whoever writes it first
    // MPOL_BIND before the first touch: with the default policy a node short of free pages (page cache: the box had 3.7
    // of 95 GB free on node 0) silently takes the other node's pages (measured 2.2 GB local / 12.7 GB remote, the pool
    // then reads across UPI); bound, the kernel reclaims the node's own cache instead
    unsigned long mask[2] = {0, 0};
    mask[node / 64] = 1UL << (node % 64);
    if (syscall(SYS_mbind, p, bytes, 2 /* MPOL_BIND */, mask, 128, 0) != 0) {
        static bool said = false;
        if (!said) std::fprintf(stderr, "cpu experts: mbind to node %d failed (%s), placement by first touch only\n", node,
                                std::strerror(errno));
        said = true;
    }
    const std::vector<int> cores = node_cores(node);
    // first touch from threads pinned on the node: the kernel places each page (2 MiB with THP) there
    const int nt = std::max(1, std::min<int>(8, (int) cores.size()));
    std::vector<std::thread> th;
    for (int t = 0; t < nt; ++t)
        th.emplace_back([&, t] {
            if (!cores.empty()) pin(cores[t % cores.size()]);
            const size_t a = bytes / nt * t, b = t == nt - 1 ? bytes : bytes / nt * (t + 1);
            for (size_t i = a & ~(size_t) 4095; i < b; i += 4096) static_cast<volatile uint8_t*>(p)[i] = 0;
        });
    for (auto& x : th) x.join();
    return p;
}
void node_free(void* p, size_t bytes) { if (p) munmap(p, bytes); }

const char* isa() {
    __builtin_cpu_init();
    if (__builtin_cpu_supports("avx512f") && __builtin_cpu_supports("avx512bw") && __builtin_cpu_supports("avx512vl"))
        return "avx512";
    if (__builtin_cpu_supports("avx2") && __builtin_cpu_supports("fma") && __builtin_cpu_supports("f16c")) return "avx2";
    return nullptr;
}

void start(const Setup& s) {
    S = s;
    for (int i = 0; i < 256; ++i) {
        E4H[i] = 0.5f * e4m3_decode(i);
        MXS[i] = std::exp2f((float) i - 119.0f);
    }
    const char* k = isa();
    const bool avx2_only = std::getenv("INK2_CPU_AVX2") != nullptr;
    g_isa = k && std::string(k) == "avx512" && !avx2_only ? 5 : 2;
    g_vnni = g_isa == 5 && __builtin_cpu_supports("avx512vnni");
    g_C = g_vnni ? 64 : 32;
    const char* act = std::getenv("INK2_CPU_ACT");
    g_q8 = act && std::string(act) == "int8" && S.h % 64 == 0 && S.ff % 64 == 0 && S.h <= BELL_H && S.ff <= FFX;
    g_trace = std::getenv("INK2_CPU_TRACE") != nullptr;
    if (const char* e = std::getenv("INK2_CPU_OPT")) g_opt = std::atoi(e);
    const int nn = nodes();
    g_stop = false;
    for (int n = 0; n < nn; ++n) {
        // the node's control block (gen, barrier, job list: polled by all its threads) on its own node: from the main
        // thread's heap it could sit across the inter-socket link, every barrier then a remote round trip
        Node* N = new (node_alloc(sizeof(Node), n)) Node;
        N->id = n;
        std::vector<int> cores = node_cores(n);
        // node 0 keeps its first core for the engine's own thread (graph launches, the serve loop)
        if (n == 0 && cores.size() > 1) {
            if (g_opt & 16) {
                // ... and the caller is put there (both hyperthreads): left to the scheduler it sat on the second
                // hyperthread of a pool core (seen: cpu 78, pool thread 6 of node 1), whose thread then ran at half
                // speed and held every barrier of its node ~30 us
                cpu_set_t set;
                CPU_ZERO(&set);
                for (int c : parse_list(read_line("/sys/devices/system/cpu/cpu" + std::to_string(cores[0]) + "/topology/thread_siblings_list")))
                    CPU_SET(c, &set);
                sched_setaffinity(0, sizeof set, &set);   // CBA: for good: threads the caller starts later inherit it
            }
            cores.erase(cores.begin());
        }
        if ((int) cores.size() > S.threads) cores.resize(S.threads);
        N->cpus = cores;
        N->bar.n = (int) cores.size();
        auto alloc = [&](size_t bytes) { return node_alloc(bytes, n); };
        N->x = (float*) alloc((size_t) BELL_H * 4);
        N->xq = (int8_t*) alloc(BELL_H);
        N->xd = (float*) alloc((size_t) BELL_H / 16 * 4);
        N->gu = (float*) alloc((size_t) K8 * 2 * FFX * 4);
        N->hh = (float*) alloc((size_t) K8 * FFX * 4);
        N->hq = (int8_t*) alloc((size_t) K8 * FFX);
        N->hd = (float*) alloc((size_t) K8 * FFX / 16 * 4);
        g_nodes.push_back(N);
    }
    g_ready = 0;
    int total = 0;
    for (Node* N : g_nodes) {
        N->th.emplace_back(master, N);
        for (int t = 1; t < (int) N->cpus.size(); ++t) N->th.emplace_back(worker, N, t);
        total += (int) N->cpus.size();
    }
    while (g_ready.load() < total) std::this_thread::yield();
    std::fprintf(stderr, "cpu experts: %s, NVFP4 activations %s, %d node(s)%s, threads per node:",
                 g_isa == 5 ? "avx512" : "avx2", g_q8 ? (g_vnni ? "int8 (VNNI)" : "int8") : "fp32", nn,
                 numa_off() ? " (NUMA placement off: STRATA_NUMA_POOL=0)" : "");
    for (Node* N : g_nodes) std::fprintf(stderr, " %zu", N->cpus.size());
    std::fprintf(stderr, "\n");
}

void stop() {
    g_stop = true;
    for (Node* N : g_nodes) {
        for (auto& t : N->th) t.join();
        N->~Node();   // CBA: the node buffers stay mapped until exit
        node_free(N, sizeof(Node));
    }
    g_nodes.clear();
}

}  // namespace ink2cpu
#endif
