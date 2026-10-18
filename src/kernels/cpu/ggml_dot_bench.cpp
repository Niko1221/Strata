// src/kernels/cpu/ggml_dot_bench.cpp - ggml-cpu's own dot products per weight format: one weight row against one
// activation row, the arithmetic the CPU pool runs for a native expert's rows.  No GPU, no model.  Made for
// STRATA_GGML_PDEP (CMakeLists.txt): ggml-cpu's IQ1_S / IQ1_M dots build their grid indices with BMI2 pdep where
// __BMI2__ is defined, and AMD Zen 1 / Zen 2 run pdep in microcode; every other format is the control, its code does
// not depend on it.  Per format: --rows rows of 4096 random weights quantized by ggml (valid scales, as a file has),
// every thread sweeping all of them, best of --reps timed runs of at least --min-ms each.  Prints ns per 256 weights
// and the weight bytes per second per thread, and a checksum of every output's bits from one single-thread sweep:
// two builds with the same checksum computed the same results, bit for bit.  tools/bench_ggml_pdep.py builds the bench
// with and without pdep and puts the two side by side.
//
//     ggml_dot_bench [--types iq1_s,iq1_m,iq2_xxs,...] [--rows 256] [--threads 1] [--reps 5] [--min-ms 200]
#include "strata/kernels/cpu/expert_layout.hpp"

#include "ggml.h"
#include "ggml-cpu.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <string>
#include <thread>
#include <vector>

namespace cpu = strata::kernels::cpu;

namespace {

constexpr int64_t kN = 4096;   // weights per row: a multiple of every block size (32, 256)
using Clock = std::chrono::steady_clock;

int type_of(const std::string& s) {
    for (int t = 0; t < GGML_TYPE_COUNT; ++t) {
        const char* n = ggml_type_name((ggml_type) t);
        if (n != nullptr && s == n) return t;
    }
    return -1;
}

std::vector<std::string> split(const std::string& s) {
    std::vector<std::string> v;
    size_t from = 0;
    for (;;) {
        const size_t c = s.find(',', from);
        v.push_back(s.substr(from, c == std::string::npos ? std::string::npos : c - from));
        if (c == std::string::npos) break;
        from = c + 1;
    }
    return v;
}

// rows x kN floats into `type`: ggml_quantize_chunk, not the CPU traits' from_float, which the i-quants do not have
void quantize(int type, const float* src, uint8_t* dst, int64_t rows) {
    const std::vector<float> imatrix((size_t) kN, 1.0f);
    ggml_quantize_chunk((ggml_type) type, src, dst, 0, rows, kN,
                        ggml_quantize_requires_imatrix((ggml_type) type) ? imatrix.data() : nullptr);
}

}  // namespace

int main(int argc, char** argv) {
    setvbuf(stdout, nullptr, _IONBF, 0);
    std::vector<std::string> types = split("iq1_s,iq1_m,iq2_xxs,iq2_xs,iq2_s,iq3_xxs,iq3_s,iq4_nl,iq4_xs,q2_K,q4_K");
    int rows = 256, threads = 1, reps = 5, min_ms = 200;
    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        const bool more = i + 1 < argc;
        if (a == "--types" && more) types = split(argv[++i]);
        else if (a == "--rows" && more) rows = std::max(1, std::atoi(argv[++i]));
        else if (a == "--threads" && more) threads = std::max(1, std::atoi(argv[++i]));
        else if (a == "--reps" && more) reps = std::max(1, std::atoi(argv[++i]));
        else if (a == "--min-ms" && more) min_ms = std::max(1, std::atoi(argv[++i]));
        else {
            std::fprintf(stderr, "usage: ggml_dot_bench [--types iq1_s,iq1_m,...] [--rows 256] [--threads 1] [--reps 5] "
                                 "[--min-ms 200]\n");
            return 2;
        }
    }
    ggml_cpu_init();
    std::printf("ggml_dot_bench: %s; ggml-cpu BMI2 (pdep in the IQ1_S / IQ1_M dots): %s; %d rows x %lld weights, "
                "%d thread(s), best of %d runs of >= %d ms\n", cpu::cpu_name().c_str(), ggml_cpu_has_bmi2() ? "yes" : "no",
                rows, (long long) kN, threads, reps, min_ms);
    std::printf("%-8s %9s %12s %10s  %s\n", "type", "B/256", "ns/256", "GB/s", "checksum");
    for (const std::string& name : types) {
        const int t = type_of(name);
        const auto* tr = t < 0 ? nullptr : ggml_get_type_traits_cpu((ggml_type) t);
        if (tr == nullptr || tr->vec_dot == nullptr) {
            std::printf("%-8s: %s\n", name.c_str(), t < 0 ? "unknown type" : "no ggml-cpu dot");
            continue;
        }
        const ggml_type vt = tr->vec_dot_type;
        const size_t row_bytes = ggml_row_size((ggml_type) t, kN), act_bytes = ggml_row_size(vt, kN);
        ggml_quantize_init((ggml_type) t);   // the i-quants' quantizers need their grids first
        std::vector<uint8_t> w((size_t) rows * row_bytes), act(act_bytes);
        {
            std::mt19937 rng(1234 + t);
            std::normal_distribution<float> nd(0.f, 0.02f);
            std::vector<float> f((size_t) rows * kN);
            for (auto& v : f) v = nd(rng);
            quantize(t, f.data(), w.data(), rows);
            std::normal_distribution<float> na(0.f, 1.f);
            std::vector<float> x((size_t) kN);
            for (auto& v : x) v = na(rng);
            const auto* vtr = ggml_get_type_traits_cpu(vt);
            if (vtr != nullptr && vtr->from_float != nullptr) vtr->from_float(x.data(), act.data(), kN);
            else quantize(vt, x.data(), act.data(), 1);
        }
        // the checksum: FNV-1a over every output's bits, one thread, one sweep
        uint64_t sum = 1469598103934665603ull;
        for (int r = 0; r < rows; ++r) {
            float out = 0.f;
            tr->vec_dot((int) kN, &out, 0, w.data() + (size_t) r * row_bytes, 0, act.data(), 0, 1);
            uint32_t bits;
            std::memcpy(&bits, &out, sizeof bits);
            for (int k = 0; k < 4; ++k) sum = (sum ^ ((bits >> (8 * k)) & 0xff)) * 1099511628211ull;
        }
        // timed: every thread sweeps every row `sweeps` times; the wall time of the slowest thread counts
        auto run = [&](int sweeps) -> double {
            std::atomic<int> ready{0};
            std::atomic<bool> go{false};
            std::vector<std::thread> th;
            std::vector<float> sink((size_t) threads * 16, 0.f);
            Clock::time_point t0;
            for (int k = 0; k < threads; ++k)
                th.emplace_back([&, k] {
                    ready.fetch_add(1);
                    while (!go.load(std::memory_order_acquire)) {}
                    float acc = 0.f;
                    for (int s = 0; s < sweeps; ++s)
                        for (int r = 0; r < rows; ++r) {
                            float out;
                            tr->vec_dot((int) kN, &out, 0, w.data() + (size_t) r * row_bytes, 0, act.data(), 0, 1);
                            acc += out;
                        }
                    sink[(size_t) k * 16] = acc;
                });
            while (ready.load() < threads) {}
            t0 = Clock::now();
            go.store(true, std::memory_order_release);
            for (auto& x : th) x.join();
            const double ms = std::chrono::duration<double, std::milli>(Clock::now() - t0).count();
            volatile float keep = sink[0];
            (void) keep;
            return ms;
        };
        const double one = std::max(0.01, run(1));
        const int sweeps = std::max(1, (int) (min_ms / one) + 1);
        double best = 1e30;
        for (int k = 0; k < reps; ++k) best = std::min(best, run(sweeps));
        const double per256 = (double) sweeps * rows * (kN / 256);
        const double ns = best * 1e6 / per256;
        const double gbs = (double) sweeps * rows * row_bytes / (best * 1e-3) / 1e9;
        std::printf("%-8s %9.1f %12.1f %10.2f  %016llx\n", name.c_str(), (double) ggml_row_size((ggml_type) t, 256), ns, gbs,
                    (unsigned long long) sum);
    }
    return 0;
}
