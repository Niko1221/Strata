// src/nemotron_h/nh_cpu_bench.cpp - how fast the CPU experts read their weights (no GPU): a decode-like load (one
// token, 22 distinct experts per layer over the file's MoE layers, every expert once per pass so nothing is cached),
// in GB/s of expert bytes, for a list of thread counts; plus the RAM's plain read bandwidth as the ceiling.
//   nh_cpu_bench --model <first shard> [--threads 6,12] [--passes 3]
#include "strata/artifact/gguf_reader.hpp"
#include "strata/nemotron_h/cpu_experts.hpp"

#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <string>
#include <thread>
#include <vector>

using Clock = std::chrono::steady_clock;
static double secs(Clock::time_point t) { return std::chrono::duration<double>(Clock::now() - t).count(); }

int main(int argc, char** argv) {
    std::string model, th = "6,12";
    int passes = 3;
    for (int i = 1; i + 1 < argc; i += 2) {
        const std::string a = argv[i], v = argv[i + 1];
        if (a == "--model") model = v;
        else if (a == "--threads") th = v;
        else if (a == "--passes") passes = std::atoi(v.c_str());
    }
    const strata::GgufModel gm = strata::GgufModel::open(model);
    struct L { strata::nemotron_h::CpuFmt f; const uint8_t *up, *dn; size_t ub, db; int E; };
    std::vector<L> layers;
    for (int l = 0; l < 88; ++l) {
        size_t su = 0, sd = 0;
        const auto* tu = gm.find("blk." + std::to_string(l) + ".ffn_up_exps.weight", &su);
        const auto* td = gm.find("blk." + std::to_string(l) + ".ffn_down_exps.weight", &sd);
        if (!tu) continue;
        L y;
        std::string err;
        strata::nemotron_h::cpu_fmt((int) tu->type, (int) td->type, (int64_t) tu->shape[0], (int64_t) tu->shape[1], y.f, err);
        y.up = gm.shard(su).tensor_data(*tu);
        y.dn = gm.shard(sd).tensor_data(*td);
        y.ub = (size_t) y.f.n_ff * y.f.up_row;
        y.db = (size_t) y.f.n_embd * y.f.d_row;
        y.E = (int) tu->shape[2];
        layers.push_back(y);
    }
    // RAM ceiling: parallel read of 8 GiB of the mapped experts (in page cache)
    {
        const size_t per = (size_t) 1 << 30;
        std::vector<std::thread> ts;
        std::atomic<uint64_t> sink{0};
        const auto t0 = Clock::now();
        for (int k = 0; k < 6; ++k)
            ts.emplace_back([&, k] {
                const uint64_t* p = (const uint64_t*) layers[(size_t) k].up;
                uint64_t s = 0;
                for (size_t i = 0; i < per / 8; i += 8) s += p[i] ^ p[i + 1] ^ p[i + 2] ^ p[i + 3] ^ p[i + 4] ^ p[i + 5] ^ p[i + 6] ^ p[i + 7];
                sink += s;
            });
        for (auto& t : ts) t.join();
        std::printf("RAM read, 6 threads: %.1f GB/s\n", 6.0 * per / secs(t0) / 1e9);
    }
    std::mt19937 rng(1);
    std::vector<float> x(1024);
    for (auto& v : x) v = (float) (rng() % 1000) / 1000.f - 0.5f;
    std::vector<float> out(22 * 1024);
    for (const char* p = th.c_str(); *p;) {
        const int n = std::atoi(p);
        while (*p && *p != ',') ++p;
        if (*p) ++p;
        strata::nemotron_h::CpuExperts pool(n);
        for (int ps = 0; ps < passes; ++ps) {
            size_t bytes = 0;
            const auto t0 = Clock::now();
            for (const L& y : layers) {
                std::vector<strata::nemotron_h::CpuJob> jobs;
                for (int k = 0; k < 22; ++k) {
                    const int e = (int) (rng() % (unsigned) y.E);
                    jobs.push_back({y.up + (size_t) e * y.ub, y.dn + (size_t) e * y.db, 0, out.data() + (size_t) k * 1024});
                    bytes += y.ub + y.db;
                }
                pool.run(y.f, x.data(), 1, jobs.data(), (int) jobs.size());
            }
            const double s = secs(t0);
            std::printf("threads %2d pass %d: %zu layers x 22 experts, %.2f GB in %.1f ms -> %.1f GB/s (%.1f tok/s if all on CPU)\n",
                        n, ps, layers.size(), bytes / 1e9, s * 1e3, bytes / s / 1e9, 1.0 / s);
        }
    }
    return 0;
}
