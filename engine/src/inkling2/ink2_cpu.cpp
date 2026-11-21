// src/inkling2/ink2_cpu.cpp - see ink2_cpu.hpp.
//
// Expert slot layout (ink2_engine.cu): [w13 | s13 | w2 | s2].  NVFP4 (layers < mx_first): rows of K/2 bytes, value 2j
// in the low nibble of byte j, then one e4m3 scale per 16 columns per row; the dot product times a per-expert factor.
// MXFP8: rows of K e4m3 bytes, then one ue8m0 exponent per 32 columns per row.  w13 rows: 2j gate j, 2j + 1 up j.
//
// Kernels: AVX2 + FMA + F16C (any x86-64 since Haswell / Zen) and an AVX-512 variant, chosen at run time.  NVFP4 codes
// go through pshufb as doubled int8 (0, 1, 2, 3, 4, 6, 8, 12, signed; the 0.5 is folded into the scale), MXFP8 bytes
// through fp16 (the GPU kernels' placement: the bits shifted into an fp16, exact, value * 2^-8, folded into the scale).
// The element values are the GPU's exactly; only the order of the fp32 sums differs.
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
void run_one(int, int, const float*, float*) {}
}  // namespace ink2cpu
#else

#if defined(__GNUC__) && !defined(__clang__)
#pragma GCC diagnostic ignored "-Wmaybe-uninitialized"   // GCC 12's avx512fintrin.h (_mm512_undefined_*)
#endif
#include <immintrin.h>
#include <pthread.h>
#include <sched.h>
#include <sys/mman.h>
#include <time.h>

#include <atomic>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
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
// physical cores of a node (the first thread of each core), in order
std::vector<int> node_cores(int node) {
    std::vector<int> cpus = parse_list(read_line("/sys/devices/system/node/node" + std::to_string(node) + "/cpulist"));
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
float e4m3_decode(unsigned v) {   // as the GPU's (NaN for 0x7F / 0xFF)
    const int e = (v >> 3) & 15, m = v & 7;
    float r = e == 0 ? (float) m * 0.001953125f : std::ldexp(1.0f + (float) m * 0.125f, e - 7);
    if (e == 15 && m == 7) r = NAN;
    return (v & 0x80) ? -r : r;
}
void init_tables() {
    for (int i = 0; i < 256; ++i) {
        E4H[i] = 0.5f * e4m3_decode(i);
        MXS[i] = std::exp2f((float) i - 119.0f);
    }
}

// ------------------------------------------------------------------------------------------------ kernels
#define T2 __attribute__((target("avx2,fma,f16c")))
#define T5 __attribute__((target("avx512f,avx512bw,avx512vl,avx2,fma,f16c")))

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

constexpr int R = 4;   // rows per step (the x loads shared)

// y[r] = factor * W[r] . x for r in [r0, r1)
T2 void nv_rows_avx2(const uint8_t* W, int N, int K, const float* x, float factor, float* y, int r0, int r1) {
    const uint8_t* S = W + (size_t) N * (K / 2);
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
    for (int r = r0; r < r1; r += R) {
        const int nr = r1 - r < R ? r1 - r : R;
        __m512 acc[R];
        for (int i = 0; i < R; ++i) acc[i] = _mm512_setzero_ps();
        for (int b = 0; b < K / 32; ++b) {
            const __m512 x0 = _mm512_loadu_ps(x + 32 * b), x1 = _mm512_loadu_ps(x + 32 * b + 16);
            for (int i = 0; i < nr; ++i) {
                const uint8_t* w = W + (size_t) (r + i) * K + 32 * b;
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

int g_isa = 0;   // 2 avx2, 5 avx512
void rows(bool nv, const uint8_t* W, int N, int K, const float* x, float factor, float* y, int r0, int r1) {
    if (r0 >= r1) return;
    if (nv) (g_isa == 5 ? nv_rows_512 : nv_rows_avx2)(W, N, K, x, factor, y, r0, r1);
    else (g_isa == 5 ? mx_rows_512 : mx_rows_avx2)(W, N, K, x, y, r0, r1);
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

constexpr int K8 = BELL_K;
struct alignas(64) Node {
    int id = 0;
    std::vector<int> cpus;
    alignas(64) std::atomic<unsigned> gen{0};   // the sequence number the workers are to run
    Barrier bar;
    // the node's jobs (published by its master before gen)
    int nj = 0, layer = 0;
    int bj[K8], ex[K8];                         // bell job index, expert
    float* x = nullptr;                         // [H]          node-local copies
    float* gu = nullptr;                        // [K8][2 FF]
    float* hh = nullptr;                        // [K8][FF]
    std::vector<std::thread> th;
};
std::vector<Node*> g_nodes;
std::atomic<bool> g_stop{false};
std::atomic<int> g_ready{0};   // threads started (start() returns once all are: a ring before a thread read its
                               // starting point was taken for that starting point, never acked: the GPU's wait trapped)
constexpr double IDLE_MS = 20.0;   // spinning this long without work, then naps of NAP_US
constexpr long NAP_US = 100;

double now_ms() {
    timespec t;
    clock_gettime(CLOCK_MONOTONIC, &t);
    return t.tv_sec * 1e3 + t.tv_nsec / 1e6;
}
// a share of n items in units of `align` (rows: 16 = a 64-byte cache line of fp32 outputs)
void share(int n, int t, int T, int align, int& a, int& b) {
    const int units = (n + align - 1) / align;
    a = std::min(n, (int) ((long) units * t / T) * align);
    b = std::min(n, (int) ((long) units * (t + 1) / T) * align);
}

void node_work(Node& N, int t, int& sense) {
    const int T = (int) N.cpus.size(), l = N.layer, H = S.h, FF = S.ff;
    const bool nv = l < S.mx_first;
    const size_t partA = nv ? S.nv_partA : S.mx_partA;
    int a, b;
    // private copy of the input: threads reading a buffer others write near hit an 8x slowdown (cpunv.cpp)
    alignas(64) static thread_local float xp[BELL_H];
    std::memcpy(xp, N.x, H * 4);
    share(2 * FF, t, T, 16, a, b);
    for (int j = 0; j < N.nj; ++j) {
        const int key = l * S.ne + N.ex[j];
        rows(nv, S.ex[key], 2 * FF, H, xp, nv ? S.f13[key] : 1.0f, N.gu + (size_t) j * 2 * FF, a, b);
    }
    N.bar.wait(sense);
    share(N.nj * FF, t, T, 16, a, b);
    for (int i = a; i < b; ++i) N.hh[i] = swiglu1(N.gu[2 * i], N.gu[2 * i + 1]);
    N.bar.wait(sense);
    alignas(64) static thread_local float hp[BELL_K * 1152];
    std::memcpy(hp, N.hh, (size_t) N.nj * FF * 4);
    share(H, t, T, 16, a, b);
    for (int j = 0; j < N.nj; ++j) {
        const int key = l * S.ne + N.ex[j];
        rows(nv, S.ex[key] + partA, H, FF, hp + (size_t) j * FF, nv ? S.f2[key] : 1.0f, S.bell->y[N.bj[j]], a, b);
    }
    N.bar.wait(sense);
}

void worker(Node* N, int t) {
    pin(N->cpus[t]);
    int sense = 0;
    unsigned seen = 0;   // gen starts at 0
    g_ready.fetch_add(1);
    double last = now_ms();
    while (!g_stop.load(std::memory_order_relaxed)) {
        const unsigned g = N->gen.load(std::memory_order_acquire);
        if (g == seen) {
            if (now_ms() - last > IDLE_MS) nap_us(NAP_US);
            else _mm_pause();
            continue;
        }
        seen = g;
        node_work(*N, t, sense);
        last = now_ms();
    }
}

void master(Node* N) {
    pin(N->cpus[0]);
    InkBell* B = S.bell;
    int sense = 0;
    unsigned seen = 0;   // the bell is zeroed before start(): ring 0 = nothing yet
    g_ready.fetch_add(1);
    double last = now_ms();
    while (!g_stop.load(std::memory_order_relaxed)) {
        const unsigned seq = B->ring;
        if (seq == seen) {
            if (now_ms() - last > IDLE_MS) nap_us(NAP_US);
            else _mm_pause();
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
            std::memcpy(N->x, (const void*) B->x, (size_t) S.h * 4);
            N->gen.store(seq, std::memory_order_release);
            node_work(*N, 0, sense);
        }
        std::atomic_thread_fence(std::memory_order_seq_cst);
        B->ack[N->id] = seq;
        last = now_ms();
        static const bool trace = std::getenv("INK2_CPU_TRACE") != nullptr;
        if (trace) {   // per node: visits, jobs, time from the ring seen to the ack (busy visits), every 2000 rings
            static thread_local double busy_ms = 0, busy_n = 0, jobs = 0, rings = 0, worst = 0;
            const double d = last - t_ring;
            ++rings;
            if (nj) { busy_ms += d; ++busy_n; jobs += nj; worst = d > worst ? d : worst; }
            if (rings == 2000) {
                std::fprintf(stderr, "[cpu] node %d: %.0f rings, %.0f busy, %.2f jobs per busy ring, %.1f us per busy ring (max %.0f)\n",
                             N->id, rings, busy_n, jobs / std::max(busy_n, 1.0), 1e3 * busy_ms / std::max(busy_n, 1.0), 1e3 * worst);
                busy_ms = busy_n = jobs = rings = worst = 0;
            }
        }
    }
}

}  // namespace

// ------------------------------------------------------------------------------------------------ API
int nodes() {
    int n = 0;
    while (!read_line("/sys/devices/system/node/node" + std::to_string(n) + "/cpulist").empty()) ++n;
    return n ? n : 1;
}

void* node_alloc(size_t bytes, int node) {
    void* p = mmap(nullptr, bytes, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    if (p == MAP_FAILED) return nullptr;
    madvise(p, bytes, MADV_HUGEPAGE);
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
    init_tables();
    const char* k = isa();
    g_isa = k && std::string(k) == "avx512" && !std::getenv("INK2_CPU_AVX2") ? 5 : 2;
    if (S.threads <= 0) return;   // the kernels only (run_one)
    const int nn = nodes();
    g_stop = false;
    for (int n = 0; n < nn; ++n) {
        Node* N = new Node;
        N->id = n;
        std::vector<int> cores = node_cores(n);
        // node 0 keeps its first core for the engine's own thread (graph launches, the serve loop)
        if (n == 0 && cores.size() > 1) cores.erase(cores.begin());
        if ((int) cores.size() > S.threads) cores.resize(S.threads);
        N->cpus = cores;
        N->bar.n = (int) cores.size();
        auto alloc = [&](size_t floats) { return static_cast<float*>(node_alloc(floats * 4, n)); };
        N->x = alloc(BELL_H);
        N->gu = alloc((size_t) K8 * 2 * S.ff);
        N->hh = alloc((size_t) K8 * S.ff);
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
    std::fprintf(stderr, "cpu experts: %s, %d node(s), threads per node:", g_isa == 5 ? "avx512" : "avx2", nn);
    for (Node* N : g_nodes) std::fprintf(stderr, " %zu", N->cpus.size());
    std::fprintf(stderr, "\n");
}

void stop() {
    g_stop = true;
    for (Node* N : g_nodes) {
        for (auto& t : N->th) t.join();
        delete N;   // CBA: the node buffers stay mapped until exit
    }
    g_nodes.clear();
}

void run_one(int l, int e, const float* x, float* y) {
    if (!g_isa) { init_tables(); const char* k = isa(); g_isa = k && std::string(k) == "avx512" ? 5 : 2; }
    const bool nv = l < S.mx_first;
    const int key = l * S.ne + e, H = S.h, FF = S.ff;
    std::vector<float> gu(2 * FF), hh(FF);
    rows(nv, S.ex[key], 2 * FF, H, x, nv ? S.f13[key] : 1.0f, gu.data(), 0, 2 * FF);
    for (int i = 0; i < FF; ++i) hh[i] = swiglu1(gu[2 * i], gu[2 * i + 1]);
    rows(nv, S.ex[key] + (nv ? S.nv_partA : S.mx_partA), H, FF, hh.data(), nv ? S.f2[key] : 1.0f, y, 0, H);
}

}  // namespace ink2cpu
#endif
