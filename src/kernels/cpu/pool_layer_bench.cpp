// src/kernels/cpu/pool_layer_bench.cpp - the CPU expert pool at decode's batch size, without the engine: the real
// ExpertPool (pinned workers, the host draining too) runs run_split_multi_native on one layer's worth of experts at a
// time - E distinct experts drawn from --mb MB of blobs (far more than any L3), each with 1 token or, with --p2, 2 -
// as the verify window hands it a layer.  Prints GB/s over the calls (the engine's "GB/s over the rows phases") and
// the pool's own gate/up / quantize / down split, so a gap to kq_cpu_bench's streaming rate is the pool's batching,
// not the kernels.
//
//     pool_layer_bench [--pair iq2_s/iq4_nl] [--workers 7] [--experts 6,7] [--p2 0.25] [--layers 4000] [--mb 4096]
//                      [--tasks 0]
#include "strata/kernels/cpu/expert_layout.hpp"
#include "strata/kernels/cpu/native_expert.hpp"
#include "strata/kernels/cpu/pool.hpp"

#include "ggml.h"
#include "ggml-cpu.h"

#include <pthread.h>
#include <sched.h>

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <random>
#include <string>
#include <vector>

namespace cpu = strata::kernels::cpu;

namespace {

constexpr int64_t kH = 2560, kFF = 640;

int type_of(const std::string& s) {
    for (int t = 0; t < GGML_TYPE_COUNT; ++t) {
        const char* n = ggml_type_name((ggml_type) t);
        if (n != nullptr && s == n) return t;
    }
    return -1;
}

std::vector<int> ints(const std::string& s) {
    std::vector<int> v;
    size_t from = 0;
    for (;;) {
        const size_t c = s.find(',', from);
        v.push_back(std::atoi(s.substr(from, c == std::string::npos ? std::string::npos : c - from).c_str()));
        if (c == std::string::npos) break;
        from = c + 1;
    }
    return v;
}

void make_blob(const cpu::NativeFmt& f, uint8_t* blob, std::mt19937& rng) {
    std::normal_distribution<float> nd(0.f, 0.02f);
    std::vector<float> row((size_t) std::max(kH, kFF));
    for (int64_t r = 0; r < 2 * kFF; ++r) {
        for (int64_t i = 0; i < kH; ++i) row[(size_t) i] = nd(rng);
        ggml_quantize_chunk((ggml_type) f.gu_type, row.data(), blob + (size_t) r * f.gu_row, 0, 1, kH, nullptr);
    }
    for (int64_t r = 0; r < kH; ++r) {
        for (int64_t i = 0; i < kFF; ++i) row[(size_t) i] = nd(rng);
        ggml_quantize_chunk((ggml_type) f.d_type, row.data(), blob + f.down_off + (size_t) r * f.d_row, 0, 1, kFF, nullptr);
    }
}

}  // namespace

int main(int argc, char** argv) {
    setvbuf(stdout, nullptr, _IONBF, 0);
    std::string pair = "iq2_s/iq4_nl";
    int workers = 7, layers = 4000, mb = 4096, tasks = 0;
    std::vector<int> experts = {6, 7};
    double p2 = 0.25;
    for (int i = 1; i + 1 < argc; i += 2) {
        const std::string a = argv[i], v = argv[i + 1];
        if (a == "--pair") pair = v;
        else if (a == "--workers") workers = std::atoi(v.c_str());
        else if (a == "--experts") experts = ints(v);
        else if (a == "--p2") p2 = std::atof(v.c_str());
        else if (a == "--layers") layers = std::atoi(v.c_str());
        else if (a == "--mb") mb = std::atoi(v.c_str());
        else if (a == "--tasks") tasks = std::atoi(v.c_str());
        else { std::fprintf(stderr, "pool_layer_bench: unknown %s\n", a.c_str()); return 2; }
    }
    ggml_cpu_init();
    const size_t slash = pair.find('/');
    const int gt = type_of(pair.substr(0, slash)), dt = type_of(pair.substr(slash + 1));
    cpu::NativeFmt f;
    std::string err;
    if (gt < 0 || dt < 0 || !cpu::native_fmt(gt, dt, kH, kFF, f, err)) { std::printf("%s: %s\n", pair.c_str(), err.c_str()); return 2; }
    ggml_quantize_init((ggml_type) gt);
    ggml_quantize_init((ggml_type) dt);
    const size_t E = std::max<size_t>(64, ((size_t) mb << 20) / f.bytes);
    std::unique_ptr<uint8_t[]> mem(new uint8_t[E * f.bytes + 4096]);
    uint8_t* blobs = (uint8_t*) (((uintptr_t) mem.get() + 4095) & ~(uintptr_t) 4095);
    {
        std::mt19937 rng(1234 + gt * 64 + dt);
        const size_t U = std::min<size_t>(16, E);
        for (size_t e = 0; e < U; ++e) make_blob(f, blobs + e * f.bytes, rng);
        for (size_t e = U; e < E; ++e) std::memcpy(blobs + e * f.bytes, blobs + (e % U) * f.bytes, f.bytes);
    }
    // two tokens' activations, quantized as the engine does; every job's outputs in its own buffer
    std::mt19937 rng(77);
    std::normal_distribution<float> nd(0.f, 1.f);
    std::vector<std::vector<uint8_t>> xa(2, std::vector<uint8_t>(cpu::kNativeActBytes));
    for (int t = 0; t < 2; ++t) {
        std::vector<float> x(kH);
        for (auto& v : x) v = nd(rng);
        cpu::native_quant_act(f, x.data(), xa[(size_t) t].data());
    }
    const int emax = *std::max_element(experts.begin(), experts.end());
    std::vector<float> outb((size_t) emax * 2 * kH);
    std::vector<cpu::ExpertJobMulti> jobs((size_t) emax);

    cpu::ExpertPool pool(workers, /*pin=*/true, /*host_works=*/true, cpu::PoolAffinity::All, tasks);
    {   // then the host thread on CPU 0, as the engine's --host-core first (before the pool, the workers would inherit
        // the one-CPU mask and all run on CPU 0)
        cpu_set_t cs;
        CPU_ZERO(&cs);
        CPU_SET(0, &cs);
        pthread_setaffinity_np(pthread_self(), sizeof cs, &cs);
    }
    std::printf("pool_layer_bench: %s, %zu experts of %.3f MB (%.0f MB), %d workers + host, experts/layer %s, p(2 tokens) %.2f\n",
                pair.c_str(), E, f.bytes / 1e6, E * (double) f.bytes / 1e6, pool.workers(),
                [&] { std::string s; for (int e : experts) s += (s.empty() ? "" : ",") + std::to_string(e); return s; }().c_str(), p2);
    std::uniform_int_distribution<size_t> pick(0, E - 1);
    std::uniform_real_distribution<double> u01(0, 1);
    auto run = [&](int n_layers, double& bytes, int& routed, int& distinct) {
        bytes = 0; routed = 0; distinct = 0;
        const auto t0 = std::chrono::steady_clock::now();
        for (int l = 0; l < n_layers; ++l) {
            const int ne = experts[(size_t) l % experts.size()];
            for (int e = 0; e < ne; ++e) {
                cpu::ExpertJobMulti& j = jobs[(size_t) e];
                j = cpu::ExpertJobMulti{};
                j.blob = blobs + pick(rng) * f.bytes;
                j.nt = u01(rng) < p2 ? 2 : 1;
                for (int t = 0; t < j.nt; ++t) {
                    j.nact[t] = xa[(size_t) t].data();
                    j.out[t] = &outb[((size_t) e * 2 + t) * kH];
                }
                routed += j.nt;
            }
            distinct += ne;
            bytes += ne * (double) f.bytes;
            pool.run_split_multi_native(f, jobs.data(), ne);
        }
        return std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    };
    double bytes;
    int routed, distinct;
    run(200, bytes, routed, distinct);   // warm-up: the workers spinning, the tables cached
    pool.ms_multi_gu = pool.ms_multi_q = pool.ms_multi_down = 0;
    const double s = run(layers, bytes, routed, distinct);
    std::printf("  %d layers: %.2f distinct / %.2f routed experts a layer, %.3f ms a layer, %.1f GB/s over the calls\n", layers,
                distinct / (double) layers, routed / (double) layers, s * 1e3 / layers, bytes / s / 1e9);
    std::printf("  pool: gate/up %.3f  quantize %.3f  down %.3f ms a layer (%.1f GB/s over gate/up + down)\n",
                pool.ms_multi_gu / layers, pool.ms_multi_q / layers, pool.ms_multi_down / layers,
                bytes / ((pool.ms_multi_gu + pool.ms_multi_down) / 1e3) / 1e9);
    {   // the outputs of a fixed set of layers (fresh rng), hashed after every call: equal across STRATA_POOL_FUSED
        std::mt19937 r2(4242);
        std::uniform_int_distribution<size_t> pk(0, E - 1);
        uint64_t h = 1469598103934665603ull;
        for (int l = 0; l < 300; ++l) {
            const int ne = experts[(size_t) l % experts.size()];
            std::fill(outb.begin(), outb.end(), 0.f);
            for (int e = 0; e < ne; ++e) {
                cpu::ExpertJobMulti& j = jobs[(size_t) e];
                j = cpu::ExpertJobMulti{};
                j.blob = blobs + pk(r2) * f.bytes;
                j.nt = (r2() & 3) == 0 ? 2 : 1;
                for (int t = 0; t < j.nt; ++t) { j.nact[t] = xa[(size_t) t].data(); j.out[t] = &outb[((size_t) e * 2 + t) * kH]; }
            }
            pool.run_split_multi_native(f, jobs.data(), ne);
            for (float v : outb) { uint32_t u; std::memcpy(&u, &v, 4); h = (h ^ u) * 1099511628211ull; }
        }
        std::printf("  output hash over 300 fixed layers: %016llx (fused layers so far %lld)\n", (unsigned long long) h,
                    (long long) pool.multi_fused);
    }
    double wp, dr, rp;
    pool.phase_ms(wp, dr, rp);
    std::printf("  pool phases (cumulative, warm-up included): wait-park %.1f  drain %.1f  re-park %.1f ms\n", wp, dr, rp);
    return 0;
}
