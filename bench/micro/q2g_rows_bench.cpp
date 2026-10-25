// bench/micro/q2g_rows_bench.cpp - q2_0_gguf_rows_multi vs its optimized variant.
//
// Calls the two exported kernels from src/kernels/cpu/expert.cpp on identical buffers, checks the outputs
// bit for bit, then times them warm and with the L3 flushed.  The kernel TU gets the engine's flags and the
// bench does not: the ISA guard scans the linked binary, and wide-ISA instructions belong only in the kernel
// translation units.
//
//   clang++ -std=c++17 -O3 -Iinclude -c bench/micro/q2g_rows_bench.cpp -o q2g_rows_bench.o
//   clang++ -std=c++17 -O3 -mavx512f -mavx512bw -mavx512vl -mavx512dq -mavx512vnni -mavx512vbmi -mf16c -mfma \
//       -Iinclude -c src/kernels/cpu/expert.cpp -o expert.o
//   clang++ q2g_rows_bench.o expert.o -o q2g_bench
//
// The same commands build on x86-64 Linux with g++ or clang++ (name the output without .exe and add -pthread
// when linking, for the affinity call).  The CPU check and the AVX-512 kernel keep the harness on x86-64.  The
// wall clock and thread pinning use the platform's API; where no TSC is available the cycle counter is the wall
// clock.
//
// --validate-only runs the full sweep and skips timing.  A CPU without the kernels' AVX-512 features exits 77
// (the CTest skip code) instead of crashing.
#if defined(__linux__) && !defined(_GNU_SOURCE)
#define _GNU_SOURCE 1
#endif
#include "strata/kernels/cpu/expert.hpp"

#if defined(_WIN32)
#define NOMINMAX
#include <windows.h>
#include <malloc.h>
#endif
#if defined(_MSC_VER)
#include <immintrin.h>
#include <intrin.h>
#elif defined(__x86_64__) || defined(__i386__)
#include <cpuid.h>
#endif

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <thread>
#include <vector>

#if defined(__linux__)
#include <pthread.h>
#include <sched.h>
#endif

namespace cpu = strata::kernels::cpu;

// expert.cpp also holds the whole-expert kernels; this bench links only that TU, so the one flag they read
// (defined in expert_layout.cpp) is stubbed here.  The Q2_0 row kernels never read it.
namespace strata::kernels::cpu {
bool expert_oracle_q8_0_enabled() { return false; }
}  // namespace strata::kernels::cpu

// The kernels this bench calls are AVX-512: F, BW, VL, VBMI and VNNI, the set the engine's cpu_avx512_ok()
// tests, plus the OSXSAVE/XCR0 state that must hold before the instructions are legal.  STRATA_FORCE_AVX2 and
// STRATA_FORCE_ISA narrow it exactly as the engine does.  Without any of it the run is a skip, not a crash.
// This is a copy of cpu_avx512_ok() (src/kernels/cpu/expert_layout.cpp) because the bench does not link that
// file; a change to the test there has to be made here too.
static bool host_has_avx512() {
    if (const char* f = std::getenv("STRATA_FORCE_AVX2"); f != nullptr && f[0] == '1') return false;
    if (const char* f = std::getenv("STRATA_FORCE_ISA"); f != nullptr && f[0] != '\0') {
        const std::string v(f);
        if (v == "avx2" || v == "avx" || v == "sse" || v == "sse4.2" || v == "none") return false;
    }
#if defined(_MSC_VER)
    int x[4];
    __cpuid(x, 0);
    if (x[0] < 7) return false;
    __cpuid(x, 1);
    if (!((x[2] >> 27) & 1)) return false;          // OSXSAVE
    if ((_xgetbv(0) & 0xE6) != 0xE6) return false;  // the OS saves the AVX-512 state
    __cpuidex(x, 7, 0);
    return ((x[1] >> 16) & 1) && ((x[1] >> 30) & 1) && ((x[1] >> 31) & 1) && ((x[2] >> 11) & 1) &&
           ((x[2] >> 1) & 1);
#elif defined(__x86_64__) || defined(__i386__)
    unsigned r[4] = {0, 0, 0, 0};
    __cpuid_count(0, 0, r[0], r[1], r[2], r[3]);
    if (r[0] < 7) return false;
    __cpuid_count(1, 0, r[0], r[1], r[2], r[3]);
    if (!((r[2] >> 27) & 1u)) return false;          // OSXSAVE
    unsigned lo = 0, hi = 0;
    __asm__ volatile("xgetbv" : "=a"(lo), "=d"(hi) : "c"(0));
    if (((((unsigned long long) hi << 32) | lo) & 0xE6) != 0xE6) return false;
    __cpuid_count(7, 0, r[0], r[1], r[2], r[3]);
    return ((r[1] >> 16) & 1u) && ((r[1] >> 30) & 1u) && ((r[1] >> 31) & 1u) && ((r[2] >> 11) & 1u) &&
           ((r[2] >> 1) & 1u);
#else
    return false;
#endif
}

using RowFn = void (*)(const uint8_t*, size_t, int, const cpu::ActQ* const*, int, float* const*, int, int);

static uint64_t g_rng = 0x9E3779B97F4A7C15ull;

static uint32_t next_u32() {
    g_rng ^= g_rng << 13;
    g_rng ^= g_rng >> 7;
    g_rng ^= g_rng << 17;
    return (uint32_t) (g_rng >> 32);
}

static void* xalloc(size_t n) {
#if defined(_WIN32)
    void* p = _aligned_malloc(n, 64);
#else
    void* p = nullptr;
    if (posix_memalign(&p, 64, n) != 0) p = nullptr;
#endif
    if (p == nullptr) {
        std::fprintf(stderr, "out of memory for %zu bytes\n", n);
        std::exit(2);
    }
    std::memset(p, 0, n);
    return p;
}

static void xfree(void* p) {
#if defined(_WIN32)
    _aligned_free(p);
#else
    std::free(p);
#endif
}

// Random code bytes; the two scale bytes of every 18-byte block get a finite normal fp16 so the arithmetic
// stays finite and a mismatch is a real mismatch rather than a NaN payload difference.
static void fill_weights(uint8_t* w, size_t bytes) {
    for (size_t i = 0; i < bytes; ++i) w[i] = (uint8_t) next_u32();
    for (size_t off = 0; off + 2 <= bytes; off += 18) {
        const uint16_t h = (uint16_t) (0x2400u | (next_u32() & 0x0FFFu));   // 0.015625 .. 0.03125
        std::memcpy(w + off, &h, 2);
    }
}

static void fill_act(cpu::ActQ& a, uint32_t seed) {
    g_rng = 0x9E3779B97F4A7C15ull ^ ((uint64_t) seed * 0xD1B54A32D192ED03ull);
    for (int i = 0; i < cpu::H; ++i) a.q[i] = (int8_t) (next_u32() & 0xFFu);
    for (int c = 0; c < cpu::MAXC; ++c) {
        a.scale[c] = 0.5f + (float) (next_u32() % 100u) * 0.01f;
        a.sum[c] = (int32_t) (next_u32() % 2001u) - 1000;
        a.hx[c] = a.scale[c] * (float) a.sum[c];
    }
    a.nchunks = cpu::MAXC;
    a.bp_pairs = 0;
}

#if defined(_WIN32)
static void sleep_ms(int ms) { Sleep(ms); }
static double now_s() {
    static LARGE_INTEGER freq = [] {
        LARGE_INTEGER f;
        QueryPerformanceFrequency(&f);
        return f;
    }();
    LARGE_INTEGER c;
    QueryPerformanceCounter(&c);
    return (double) c.QuadPart / (double) freq.QuadPart;
}
#else
static void sleep_ms(int ms) { std::this_thread::sleep_for(std::chrono::milliseconds(ms)); }
static double now_s() {
    return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count();
}
#endif

// The TSC where the compiler and architecture provide it, the wall clock (nanoseconds) otherwise.  Only
// differences are used, so either is a valid tick source for the cycle columns.
#if (defined(_MSC_VER) || defined(__clang__) || defined(__GNUC__)) && (defined(_M_X64) || defined(_M_IX86) || \
                                                                      defined(__x86_64__) || defined(__i386__))
#define Q2G_TSC 1
#else
#define Q2G_TSC 0
#endif

static uint64_t cycles_now() {
#if Q2G_TSC && defined(_MSC_VER)
    return __rdtsc();
#elif Q2G_TSC
    return __builtin_ia32_rdtsc();
#else
    return (uint64_t) (now_s() * 1e9);
#endif
}

// TSC ticks per second, measured against the wall clock over 50 ms.
static double tsc_hz() {
    const double q0 = now_s();
    const uint64_t t0 = cycles_now();
    sleep_ms(50);
    const uint64_t t1 = cycles_now();
    const double q1 = now_s();
    return (double) (t1 - t0) / (q1 - q0);
}

struct Stats {
    double med_ns = 0, p95_ns = 0, med_cyc = 0;
};

// Per-repetition v2/original ratio: the median plus the lowest and highest rep, so run-to-run spread is
// visible rather than hidden behind two independently-taken medians.
struct Ratio {
    double med = 0, lo = 0, hi = 0;
};

static double pct(std::vector<double> v, double p) {
    std::sort(v.begin(), v.end());
    const size_t k = (size_t) std::ceil(p * (double) v.size()) - 1;
    return v[std::min(k, v.size() - 1)];
}

static Ratio ratio_stats(const std::vector<double>& a, const std::vector<double>& b) {
    std::vector<double> r;
    for (size_t i = 0; i < a.size(); ++i) r.push_back(b[i] / a[i]);
    std::sort(r.begin(), r.end());
    Ratio out;
    out.med = pct(r, 0.5);
    out.lo = r.front();
    out.hi = r.back();
    return out;
}

// Alternates the two functions rep by rep so clock drift and thermal state are shared.
static void time_pair(RowFn f0, RowFn f1, const uint8_t* w, size_t rb, int nb, const cpu::ActQ* const* a, int nt,
                      float* const* out, int rows, int iters, Stats& s0, Stats& s1, Ratio& ratio) {
    for (int i = 0; i < 3; ++i) {
        f0(w, rb, nb, a, nt, out, 0, rows);
        f1(w, rb, nb, a, nt, out, 0, rows);
    }
    std::vector<double> n0, n1, c0, c1;
    for (int rep = 0; rep < 9; ++rep) {
        {
            const double q0 = now_s();
            const uint64_t t0 = cycles_now();
            for (int i = 0; i < iters; ++i) f0(w, rb, nb, a, nt, out, 0, rows);
            const uint64_t t1 = cycles_now();
            const double q1 = now_s();
            n0.push_back((q1 - q0) * 1e9 / iters);
            c0.push_back((double) (t1 - t0) / iters);
        }
        {
            const double q0 = now_s();
            const uint64_t t0 = cycles_now();
            for (int i = 0; i < iters; ++i) f1(w, rb, nb, a, nt, out, 0, rows);
            const uint64_t t1 = cycles_now();
            const double q1 = now_s();
            n1.push_back((q1 - q0) * 1e9 / iters);
            c1.push_back((double) (t1 - t0) / iters);
        }
    }
    s0.med_ns = pct(n0, 0.5);
    s0.p95_ns = pct(n0, 0.95);
    s0.med_cyc = pct(c0, 0.5);
    s1.med_ns = pct(n1, 0.5);
    s1.p95_ns = pct(n1, 0.95);
    s1.med_cyc = pct(c1, 0.5);
    ratio = ratio_stats(n0, n1);
}

// One 64-byte line of the flush buffer.  memcpy, not a cast: a uint64_t lvalue on a uint8_t buffer is a
// strict-aliasing violation that a compiler at -O2 may use to drop the read.
static uint64_t load_line(const uint8_t* p) {
    uint64_t v;
    std::memcpy(&v, p, sizeof v);
    return v;
}

// One call per repetition, each preceded by a read of a buffer larger than the 96 MB L3.
static void time_pair_cold(RowFn f0, RowFn f1, const uint8_t* w, size_t rb, int nb, const cpu::ActQ* const* a,
                           int nt, float* const* out, int rows, uint8_t* flush, size_t flush_bytes, Stats& s0,
                           Stats& s1, Ratio& ratio) {
    std::vector<double> n0, n1;
    volatile uint64_t sink = 0;
    for (int i = 0; i < 2; ++i) {
        f0(w, rb, nb, a, nt, out, 0, rows);
        f1(w, rb, nb, a, nt, out, 0, rows);
    }
    for (int rep = 0; rep < 21; ++rep) {
        {
            uint64_t acc = 0;
            for (size_t i = 0; i < flush_bytes; i += 64) acc += load_line(flush + i);
            sink += acc;
            const double q0 = now_s();
            f0(w, rb, nb, a, nt, out, 0, rows);
            const double q1 = now_s();
            n0.push_back((q1 - q0) * 1e9);
        }
        {
            uint64_t acc = 0;
            for (size_t i = 0; i < flush_bytes; i += 64) acc += load_line(flush + i);
            sink += acc;
            const double q0 = now_s();
            f1(w, rb, nb, a, nt, out, 0, rows);
            const double q1 = now_s();
            n1.push_back((q1 - q0) * 1e9);
        }
    }
    s0.med_ns = pct(n0, 0.5);
    s0.p95_ns = pct(n0, 0.95);
    s1.med_ns = pct(n1, 0.5);
    s1.p95_ns = pct(n1, 0.95);
    ratio = ratio_stats(n0, n1);
    (void) sink;
}

struct Case {
    int nt;
    int nblocks;
    int rows;
};

static bool validate(const Case& c, const uint8_t* w, size_t rb, const cpu::ActQ* const* a, float* const* o0,
                     float* const* o1) {
    for (int t = 0; t < c.nt; ++t) {
        std::memset(o0[t], 0xCD, (size_t) c.rows * sizeof(float));
        std::memset(o1[t], 0xCD, (size_t) c.rows * sizeof(float));
    }
    cpu::q2_0_gguf_rows_multi(w, rb, c.nblocks, a, c.nt, o0, 0, c.rows);
    cpu::q2_0_gguf_rows_multi_v2(w, rb, c.nblocks, a, c.nt, o1, 0, c.rows);
    for (int t = 0; t < c.nt; ++t) {
        if (std::memcmp(o0[t], o1[t], (size_t) c.rows * sizeof(float)) != 0) {
            for (int r = 0; r < c.rows; ++r) {
                uint32_t b0, b1;
                std::memcpy(&b0, &o0[t][r], 4);
                std::memcpy(&b1, &o1[t][r], 4);
                if (b0 != b1) {
                    std::fprintf(stderr, "MISMATCH nt=%d nb=%d row=%d t=%d: %08x vs %08x (%.9g vs %.9g)\n", c.nt,
                                 c.nblocks, r, t, b0, b1, o0[t][r], o1[t][r]);
                    return false;
                }
            }
        }
    }
    return true;
}

int main(int argc, char** argv) {
    const bool validate_only = argc > 1 && std::strcmp(argv[1], "--validate-only") == 0;

    if (!host_has_avx512()) {
        std::fprintf(stderr, "skip: this CPU has no AVX-512 F/BW/VL/VBMI/VNNI (or STRATA_FORCE_AVX2 / "
                             "STRATA_FORCE_ISA says not to use it)\n");
        return 77;
    }

    bool pinned_ok = false;
#if defined(_WIN32)
    SetPriorityClass(GetCurrentProcess(), HIGH_PRIORITY_CLASS);
    SetThreadPriority(GetCurrentThread(), THREAD_PRIORITY_HIGHEST);
    pinned_ok = SetThreadAffinityMask(GetCurrentThread(), (DWORD_PTR) 1 << 1) != 0;
#elif defined(__linux__)
    cpu_set_t set;
    CPU_ZERO(&set);
    CPU_SET(1, &set);
    pinned_ok = pthread_setaffinity_np(pthread_self(), sizeof(set), &set) == 0;
#endif
    if (!pinned_ok) std::fprintf(stderr, "warning: could not pin thread\n");

    const double ghz = tsc_hz() / 1e9;
#if !Q2G_TSC
    std::fprintf(stderr, "note: no TSC on this target; cycles are steady_clock nanoseconds\n");
#endif
    std::printf("tsc_ghz=%.4f pinned_core=%s\n", ghz, pinned_ok ? "1" : "none");

    // 128 MiB, larger than the 96 MB L3 of the 9800X3D.
    const size_t flush_bytes = (size_t) 128 << 20;
    uint8_t* flush = (uint8_t*) xalloc(flush_bytes);

    // ---- validation sweep: every NT, every nblocks 1..40, rows 7 (odd, so row starts walk all alignments)
    int checked = 0, failed = 0;
    for (int nt = 1; nt <= 8; ++nt) {
        for (int nb = 1; nb <= 40; ++nb) {
            const size_t rb = (size_t) nb * 18;
            const int rows = 7;
            uint8_t* w = (uint8_t*) xalloc(rb * rows);
            fill_weights(w, rb * rows);
            cpu::ActQ acts[8];
            const cpu::ActQ* ap[8];
            float* o0[8];
            float* o1[8];
            for (int t = 0; t < nt; ++t) {
                fill_act(acts[t], (uint32_t) (nt * 1000 + nb * 10 + t));
                ap[t] = &acts[t];
                o0[t] = (float*) xalloc((size_t) rows * sizeof(float));
                o1[t] = (float*) xalloc((size_t) rows * sizeof(float));
            }
            const Case c{nt, nb, rows};
            if (!validate(c, w, rb, ap, o0, o1)) {
                ++failed;
            } else {
                ++checked;
            }
            // A row range that does not start at 0.
            const int r0 = 3;
            cpu::q2_0_gguf_rows_multi(w, rb, nb, ap, nt, o0, r0, rows);
            cpu::q2_0_gguf_rows_multi_v2(w, rb, nb, ap, nt, o1, r0, rows);
            for (int t = 0; t < nt; ++t)
                if (std::memcmp(o0[t] + r0, o1[t] + r0, (size_t) (rows - r0) * sizeof(float)) != 0) {
                    std::fprintf(stderr, "RANGE MISMATCH nt=%d nb=%d t=%d\n", nt, nb, t);
                    ++failed;
                }
            for (int t = 0; t < nt; ++t) {
                xfree(o0[t]);
                xfree(o1[t]);
            }
            xfree(w);
        }
    }
    std::printf("validation: %d cases bit-exact, %d failed\n", checked, failed);
    xfree(flush);
    if (failed != 0) return 1;

    if (validate_only) {
        std::printf("validate-only: done\n");
        return 0;
    }

    // ---- timing: real geometry (FF=640 -> 10 blocks; H=2560 -> 40 blocks) and the full-group/tail edge cases
    flush = (uint8_t*) xalloc(flush_bytes);
    const Case cases[] = {
        {8, 10, 640},   {1, 10, 640},   {4, 10, 640},   {8, 40, 1280},  {1, 40, 1280},  {4, 40, 1280},
        {8, 8, 2048},   {1, 8, 2048},   {8, 1, 4096},   {1, 1, 4096},   {8, 7, 1024},   {1, 7, 1024},
        {8, 9, 1024},   {1, 9, 1024},   {8, 16, 512},   {1, 16, 512},
    };
    std::printf("%-4s %-4s %-5s %10s %10s %10s %10s %7s %16s %10s %10s\n", "nt", "nb", "rows", "orig_med",
                "v2_med", "orig_p95", "v2_p95", "ratio", "ratio_lo..hi", "orig_GB/s", "v2_GB/s");
    for (const Case& c : cases) {
        const size_t rb = (size_t) c.nblocks * 18;
        uint8_t* w = (uint8_t*) xalloc(rb * (size_t) c.rows);
        fill_weights(w, rb * (size_t) c.rows);
        cpu::ActQ acts[8];
        const cpu::ActQ* ap[8];
        float* out[8];
        float* vout[8];
        for (int t = 0; t < c.nt; ++t) {
            fill_act(acts[t], (uint32_t) (c.nt * 77 + c.nblocks * 13 + t));
            ap[t] = &acts[t];
            out[t] = (float*) xalloc((size_t) c.rows * sizeof(float));
            vout[t] = (float*) xalloc((size_t) c.rows * sizeof(float));
        }
        if (!validate(c, w, rb, ap, out, vout)) return 1;
        for (int t = 0; t < c.nt; ++t) xfree(vout[t]);
        const double bytes = (double) rb * (double) c.rows;
        const double ops = (double) c.rows * (double) c.nt * (double) c.nblocks;
        int iters = (int) (2000000.0 / ((double) c.rows * (double) c.nblocks));
        iters = std::max(1, std::min(iters, 2000));
        Stats s0, s1;
        Ratio ratio;
        time_pair(cpu::q2_0_gguf_rows_multi, cpu::q2_0_gguf_rows_multi_v2, w, rb, c.nblocks, ap, c.nt, out,
                  c.rows, iters, s0, s1, ratio);
        std::printf("%-4d %-4d %-5d %10.1f %10.1f %10.1f %10.1f %7.3f %7.3f..%-6.3f %10.1f %10.1f  cyc/op %.2f..%.2f\n",
                    c.nt, c.nblocks, c.rows, s0.med_ns, s1.med_ns, s0.p95_ns, s1.p95_ns, ratio.med, ratio.lo,
                    ratio.hi, bytes / s0.med_ns, bytes / s1.med_ns, s0.med_cyc / ops, s1.med_cyc / ops);
        for (int t = 0; t < c.nt; ++t) xfree(out[t]);
        xfree(w);
    }

    // ---- cold: L3 flushed before every call
    std::printf("\ncold (L3 flushed with %zu MiB between calls)\n", flush_bytes >> 20);
    std::printf("%-4s %-4s %-5s %10s %10s %10s %10s %7s %16s %10s %10s\n", "nt", "nb", "rows", "orig_med",
                "v2_med", "orig_p95", "v2_p95", "ratio", "ratio_lo..hi", "orig_GB/s", "v2_GB/s");
    for (const Case& c : cases) {
        if (!((c.nblocks == 10 && (c.nt == 1 || c.nt == 8)) || (c.nblocks == 40 && (c.nt == 1 || c.nt == 8)))) continue;
        const size_t rb = (size_t) c.nblocks * 18;
        uint8_t* w = (uint8_t*) xalloc(rb * (size_t) c.rows);
        fill_weights(w, rb * (size_t) c.rows);
        cpu::ActQ acts[8];
        const cpu::ActQ* ap[8];
        float* out[8];
        for (int t = 0; t < c.nt; ++t) {
            fill_act(acts[t], (uint32_t) (c.nt * 31 + c.nblocks * 7 + t));
            ap[t] = &acts[t];
            out[t] = (float*) xalloc((size_t) c.rows * sizeof(float));
        }
        const double bytes = (double) rb * (double) c.rows;
        Stats s0, s1;
        Ratio ratio;
        time_pair_cold(cpu::q2_0_gguf_rows_multi, cpu::q2_0_gguf_rows_multi_v2, w, rb, c.nblocks, ap, c.nt, out,
                       c.rows, flush, flush_bytes, s0, s1, ratio);
        std::printf("%-4d %-4d %-5d %10.1f %10.1f %10.1f %10.1f %7.3f %7.3f..%-6.3f %10.1f %10.1f\n", c.nt,
                    c.nblocks, c.rows, s0.med_ns, s1.med_ns, s0.p95_ns, s1.p95_ns, ratio.med, ratio.lo, ratio.hi,
                    bytes / s0.med_ns, bytes / s1.med_ns);
        for (int t = 0; t < c.nt; ++t) xfree(out[t]);
        xfree(w);
    }

    xfree(flush);
    std::printf("done\n");
    return 0;
}
