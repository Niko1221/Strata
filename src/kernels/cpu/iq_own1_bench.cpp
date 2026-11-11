// src/kernels/cpu/iq_own1_bench.cpp - STRATA_CPU_IQ_OWN1: the one-token IQ2_XXS / IQ2_XS rows (iq_own1_rows) against
// ggml-cpu's vec_dot, the dot they replace.  No GPU, no model.
//
//   iq_own1_bench                        parity: bit for bit against ggml-cpu's dot, 1M+ random rows per type (several
//                                        widths, including a block count that is not a multiple of 8); exit 1 on a difference
//   iq_own1_bench --bench [--threads N] [--mb M] [--reps R] [--type 16|17|29]
//                                        ns per 256-weight block from cache on one thread (rows that stay in L2), and GB/s of
//                                        weights from DRAM on N pinned threads (each streams its own M MiB, first-touched by
//                                        itself); ggml's dot, Strata's own multi-token rows at one token (iq256x_rows), these rows
//
// Skips (exit 77) on a CPU or a ggml-cpu build without AVX2 + FMA + F16C, or with AVX-512 (the rows match ggml's AVX2 dot).
#include "strata/kernels/cpu/expert_layout.hpp"
#include "strata/kernels/cpu/iq_avx2.hpp"

#include "ggml.h"
#include "ggml-cpu.h"

#include <algorithm>
#include <atomic>
#include <chrono>
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

namespace {

uint64_t g_seed = 0x9E3779B97F4A7C15ull;
uint64_t next_u64() {
    g_seed ^= g_seed << 13;
    g_seed ^= g_seed >> 7;
    g_seed ^= g_seed << 17;
    return g_seed;
}

// random bytes, the block scale a sane f16 (as ds4_gg_cpu.cpp own_ok): every index, sign and sub-scale pattern occurs
void fill_rows(std::vector<uint8_t>& w, int type) {
    const size_t bb = ggml_type_size((ggml_type) type);
    for (size_t o = 0; o + 8 <= w.size(); o += 8) {
        const uint64_t v = next_u64();
        std::memcpy(&w[o], &v, 8);
    }
    for (size_t o = 0; o < w.size(); o += bb) {
        const ggml_fp16_t d = ggml_fp32_to_fp16(0.0005f + 0.0005f * (float) (next_u64() % 64));
        if (type == 29) {   // IQ1_M: its f16 sits in the top nibbles of the four scale words (bytes 48..55)
            uint16_t sc[4];
            std::memcpy(sc, &w[o + 48], 8);
            for (int k = 0; k < 4; ++k) sc[k] = (uint16_t) ((sc[k] & 0x0fff) | ((d >> (4 * k)) & 0xf) << 12);
            std::memcpy(&w[o + 48], sc, 8);
        } else {
            std::memcpy(&w[o], &d, sizeof d);
        }
    }
}

void make_act(std::vector<uint8_t>& a, int n) {
    const ggml_type_traits_cpu* q8 = ggml_get_type_traits_cpu(GGML_TYPE_Q8_K);
    std::vector<float> x((size_t) n);
    const int kind = (int) (next_u64() % 3);
    for (auto& v : x) {
        const float u = ((float) (next_u64() % 20001) - 10000.0f) * 1e-4f;
        v = kind == 0 ? u : kind == 1 ? u * u * u * 8.0f : (next_u64() % 16 == 0 ? u * 30.0f : u);   // flat, peaked, outliers
    }
    a.assign(ggml_row_size(GGML_TYPE_Q8_K, n), 0);
    q8->from_float(x.data(), a.data(), n);
}

int parity() {
    int bad = 0;
    for (int type : {16, 17, 29}) {
        const ggml_type_traits_cpu* tc = ggml_get_type_traits_cpu((ggml_type) type);
        uint64_t rows_done = 0, diffs = 0;
        const int widths[] = {256, 512, 1536, 2048, 2304, 2560, 4096, 7168};   // 1..28 blocks; 6 and 9 are not multiples of 8
        for (int n : widths) {
            const size_t rb = ggml_row_size((ggml_type) type, n);
            const int rows = 4096;
            std::vector<uint8_t> w(rb * rows), a;
            std::vector<float> o0(rows), o1(rows);
            const int batches = n <= 512 ? 64 : 24;
            for (int b = 0; b < batches; ++b) {
                fill_rows(w, type);
                make_act(a, n);
                for (int r = 0; r < rows; ++r) tc->vec_dot(n, &o0[(size_t) r], 0, w.data() + (size_t) r * rb, 0, a.data(), 0, 1);
                cpu::iq_own1_rows(type, w.data(), rb, n, a.data(), o1.data(), 0, rows);
                for (int r = 0; r < rows; ++r)
                    if (std::memcmp(&o0[(size_t) r], &o1[(size_t) r], sizeof(float)) != 0) {
                        if (diffs < 5)
                            std::fprintf(stderr, "type %d n %d row %d: ggml %.9g, own1 %.9g\n", type, n, r, o0[(size_t) r],
                                         o1[(size_t) r]);
                        ++diffs;
                    }
                rows_done += rows;
            }
        }
        std::printf("iq_own1 parity type %d (%s): %llu rows, %llu differ from ggml-cpu's vec_dot\n", type,
                    ggml_type_name((ggml_type) type), (unsigned long long) rows_done, (unsigned long long) diffs);
        if (diffs != 0) bad = 1;
    }
    return bad;
}

bool pin(int core) {
#if defined(__linux__)
    cpu_set_t s;
    CPU_ZERO(&s);
    CPU_SET(core, &s);
    return pthread_setaffinity_np(pthread_self(), sizeof s, &s) == 0;
#else
    (void) core;
    return false;
#endif
}

enum Impl { kGgml = 0, kStrataRows = 1, kOwn1 = 2 };
const char* impl_name(int i) { return i == kGgml ? "ggml vec_dot" : i == kStrataRows ? "iq256x_rows nt=1" : "iq_own1_rows"; }

void run(int impl, int type, const uint8_t* w, size_t rb, int n, const void* act, float* out, int rows) {
    if (impl == kGgml) {
        const ggml_type_traits_cpu* tc = ggml_get_type_traits_cpu((ggml_type) type);
        for (int r = 0; r < rows; ++r) tc->vec_dot(n, &out[r], 0, w + (size_t) r * rb, 0, act, 0, 1);
    } else if (impl == kStrataRows) {
        const void* a[1] = {act};
        float* o[1] = {out};
        cpu::iq256x_rows(type, w, rb, n, a, 1, o, 0, rows);
    } else {
        cpu::iq_own1_rows(type, w, rb, n, act, out, 0, rows);
    }
}

std::vector<int> g_types = {16, 17, 29};

void bench(int threads, int mb, int reps) {
    // the expert shapes of the DeepSeek V4.1 Flash CPU lane (gate/up 2048 x 7168) and a Qwen-like 640 x 2560
    for (int n : {7168, 2560}) {
        for (int type : g_types) {
            const size_t rb = ggml_row_size((ggml_type) type, n);
            const int nb = n / 256;
            // ---- one thread, from cache: 128 KiB of rows (L2 on Broadwell is 256 KiB)
            pin(0);
            {
                const int rows = (int) std::max<size_t>(1, (128u << 10) / rb);
                std::vector<uint8_t> w(rb * (size_t) rows), a;
                fill_rows(w, type);
                make_act(a, n);
                std::vector<float> out((size_t) rows);
                double best[3] = {1e30, 1e30, 1e30};
                for (int rep = 0; rep < reps; ++rep)
                    for (int impl = 0; impl < 3; ++impl) {
                        const int loops = 400;
                        run(impl, type, w.data(), rb, n, a.data(), out.data(), rows);
                        const auto t0 = std::chrono::steady_clock::now();
                        for (int l = 0; l < loops; ++l) run(impl, type, w.data(), rb, n, a.data(), out.data(), rows);
                        const double s = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
                        best[impl] = std::min(best[impl], s * 1e9 / ((double) loops * rows * nb));
                    }
                std::printf("bench %-8s n %5d  1 thread, cache: %-17s %6.2f ns/block | %-17s %6.2f | %-17s %6.2f  "
                            "(own1 vs ggml %+.1f %%)\n",
                            ggml_type_name((ggml_type) type), n, impl_name(0), best[0], impl_name(1), best[1], impl_name(2),
                            best[2], 100.0 * (best[0] / best[2] - 1.0));
            }
            // ---- N threads, from DRAM: each thread its own rows, first-touched (so node-local) by itself
            {
                const int rows = (int) std::max<size_t>(1, ((size_t) mb << 20) / rb);
                std::vector<std::vector<uint8_t>> ws((size_t) threads);
                std::vector<std::vector<float>> outs((size_t) threads);
                std::vector<uint8_t> a;
                make_act(a, n);
                std::atomic<int> phase{0}, ready{0};
                double best[3] = {1e30, 1e30, 1e30};
                std::vector<std::thread> th;
                std::atomic<int> go_impl{-1}, done{0};
                std::atomic<bool> stop{false};
                for (int t = 0; t < threads; ++t)
                    th.emplace_back([&, t] {
                        pin(t);
                        ws[(size_t) t].assign(rb * (size_t) rows, 0);
                        uint64_t s = 0x1234567ull + (uint64_t) t * 0x9E3779B97F4A7C15ull;
                        auto& w = ws[(size_t) t];
                        for (size_t o = 0; o + 8 <= w.size(); o += 8) {
                            s ^= s << 13; s ^= s >> 7; s ^= s << 17;
                            std::memcpy(&w[o], &s, 8);
                        }
                        const size_t bb = ggml_type_size((ggml_type) type);
                        const ggml_fp16_t d = ggml_fp32_to_fp16(0.001f);
                        for (size_t o = 0; o < w.size(); o += bb) {
                            if (type == 29) {
                                uint16_t sc[4];
                                std::memcpy(sc, &w[o + 48], 8);
                                for (int k = 0; k < 4; ++k) sc[k] = (uint16_t) ((sc[k] & 0x0fff) | ((d >> (4 * k)) & 0xf) << 12);
                                std::memcpy(&w[o + 48], sc, 8);
                            } else {
                                std::memcpy(&w[o], &d, sizeof d);
                            }
                        }
                        outs[(size_t) t].assign((size_t) rows, 0.f);
                        ++ready;
                        int seen = 0;
                        for (;;) {
                            while (phase.load() == seen && !stop.load()) std::this_thread::yield();
                            if (stop.load()) return;
                            seen = phase.load();
                            run(go_impl.load(), type, w.data(), rb, n, a.data(), outs[(size_t) t].data(), rows);
                            ++done;
                        }
                    });
                while (ready.load() < threads) std::this_thread::yield();
                for (int rep = 0; rep < reps; ++rep)
                    for (int impl = 0; impl < 3; ++impl) {
                        go_impl = impl;
                        done = 0;
                        const auto t0 = std::chrono::steady_clock::now();
                        ++phase;
                        while (done.load() < threads) std::this_thread::yield();
                        const double sec = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
                        best[impl] = std::min(best[impl], sec);
                    }
                stop = true;
                for (auto& x : th) x.join();
                const double bytes = (double) rb * rows * threads;
                std::printf("bench %-8s n %5d %2d threads, DRAM %4d MiB each: %-17s %6.1f GB/s | %-17s %6.1f | %-17s %6.1f  "
                            "(own1 vs ggml %+.1f %%; %.2f / %.2f / %.2f ns/block per thread)\n",
                            ggml_type_name((ggml_type) type), n, threads, mb, impl_name(0), bytes / best[0] / 1e9, impl_name(1),
                            bytes / best[1] / 1e9, impl_name(2), bytes / best[2] / 1e9, 100.0 * (best[0] / best[2] - 1.0),
                            best[0] * 1e9 / ((double) rows * nb), best[1] * 1e9 / ((double) rows * nb),
                            best[2] * 1e9 / ((double) rows * nb));
            }
        }
    }
}

}  // namespace

int main(int argc, char** argv) {
    bool do_bench = false;
    int threads = 44, mb = 32, reps = 5;
    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        const bool more = i + 1 < argc;
        if (a == "--bench") do_bench = true;
        else if (a == "--threads" && more) threads = std::max(1, std::atoi(argv[++i]));
        else if (a == "--mb" && more) mb = std::max(1, std::atoi(argv[++i]));
        else if (a == "--reps" && more) reps = std::max(1, std::atoi(argv[++i]));
        else if (a == "--type" && more) g_types = {std::atoi(argv[++i])};
        else {
            std::fprintf(stderr, "usage: iq_own1_bench [--bench [--threads N] [--mb M] [--reps R] [--type T]]\n");
            return 2;
        }
    }
    if (!cpu::cpu_avx2_ok() || cpu::cpu_avx512_ok() || cpu::cpu_avx512bw_ok() || !ggml_cpu_has_avx2() || !ggml_cpu_has_fma() ||
        !ggml_cpu_has_f16c() || ggml_cpu_has_avx512()) {
        std::printf("iq_own1_bench: skipped (needs AVX2 + FMA + F16C and no AVX-512, in the CPU and in ggml-cpu)\n");
        return 77;
    }
    ggml_cpu_init();
    if (do_bench) {
        bench(threads, mb, reps);
        return 0;
    }
    const int bad = parity();
    std::printf("iq_own1_bench: %s\n", bad ? "FAILED" : "passed");
    return bad;
}
