// src/kernels/exl3_moe_layer_check.cpp - a full layer's EXL3 MoE (10 routed experts) on the GPU vs the CPU
// reference, on REAL weights.  Needs the downloaded model; not a ctest.
//
// Usage: exl3_moe_layer_check <model_dir> <layer> <n_experts> [k]
#include "strata/artifact/exl3_model.hpp"
#include "strata/kernels/exl3_experts.hpp"
#include "strata/kernels/cpu/exl3.hpp"

#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <random>
#include <vector>

using namespace strata;
using namespace strata::kernels;

static float h2f(uint16_t h) {
    unsigned sign = (unsigned)(h >> 15) << 31, exp = (h >> 10) & 0x1F, man = h & 0x3FF, f;
    if (exp == 0) { if (!man) f = sign; else { exp = 127 - 15 + 1; while (!(man & 0x400)) { man <<= 1; --exp; } man &= 0x3FF; f = sign | (exp << 23) | (man << 13); } }
    else if (exp == 0x1F) f = sign | 0x7F800000u | (man << 13);
    else f = sign | ((exp + 127 - 15) << 23) | (man << 13);
    float o; std::memcpy(&o, &f, 4); return o;
}
static uint16_t to_h(float v) {
    unsigned f; std::memcpy(&f, &v, 4);
    unsigned sign = (f >> 16) & 0x8000u, m = f & 0x7FFFFFu; int e = (int)((f >> 23) & 0xFF) - 127 + 15;
    if (e <= 0) return (uint16_t) sign;
    if (e >= 0x1F) return (uint16_t)(sign | 0x7C00u);
    return (uint16_t)(sign | (e << 10) | (m >> 13));
}

int main(int argc, char** argv) {
    if (argc < 4) { std::fprintf(stderr, "usage: %s <model_dir> <layer> <n_experts> [k]\n", argv[0]); return 2; }
    try {
        std::string dir = argv[1];
        int layer = std::stoi(argv[2]), n_experts = std::stoi(argv[3]), k = argc > 4 ? std::stoi(argv[4]) : 10;
        Exl3Model m(dir);
        Exl3ExpertStore store(dir, layer, n_experts, nullptr);
        std::printf("store: layer %d, %d experts, %.1f MiB on GPU\n", layer, store.n_experts(),
                    (double) store.bytes() / (1 << 20));

        std::mt19937 rng(7);
        std::uniform_real_distribution<float> sc(-1.0f, 1.0f);
        std::vector<uint16_t> x(2560);
        for (auto& v : x) v = to_h(sc(rng) * 0.3f);
        std::vector<int> ids(k);
        std::vector<float> w(k);
        float wsum = 0;
        for (int i = 0; i < k; ++i) { ids[i] = i; w[i] = std::fabs(sc(rng)) + 0.05f; wsum += w[i]; }
        for (auto& v : w) v /= wsum;

        // GPU (x and out are device buffers)
        uint16_t* d_x = nullptr; uint16_t* d_out = nullptr;
        if (cudaMalloc(&d_x, (size_t) 2560 * 2) != cudaSuccess || cudaMalloc(&d_out, (size_t) 2560 * 2) != cudaSuccess) {
            std::fprintf(stderr, "cudaMalloc failed\n"); return 1;
        }
        (void) cudaMemcpy(d_x, x.data(), (size_t) 2560 * 2, cudaMemcpyHostToDevice);
        std::printf("running GPU...\n"); std::fflush(stdout);
        store.run(d_x, ids.data(), w.data(), k, d_out, nullptr);
        if (cudaDeviceSynchronize() != cudaSuccess) { std::fprintf(stderr, "cuda sync failed: %s\n", cudaGetErrorString(cudaGetLastError())); return 1; }
        std::vector<uint16_t> out_gpu(2560);
        (void) cudaMemcpy(out_gpu.data(), d_out, (size_t) 2560 * 2, cudaMemcpyDeviceToHost);
        (void) cudaFree(d_x); (void) cudaFree(d_out);
        std::printf("GPU run returned\n"); std::fflush(stdout);

        // ---- timing: expert FFN compute throughput (one layer, k experts) ----
        {
            (void) cudaDeviceSynchronize();
            const int iters = 200;
            auto t0 = std::chrono::high_resolution_clock::now();
            for (int it = 0; it < iters; ++it) store.run(d_x, ids.data(), w.data(), k, d_out, nullptr);
            (void) cudaDeviceSynchronize();
            auto t1 = std::chrono::high_resolution_clock::now();
            double ms = std::chrono::duration<double, std::milli>(t1 - t0).count() / iters;
            std::printf("expert FFN: %.3f ms/layer (k=%d experts) -> ~%.1f tok/s expert-compute-bound (x48 layers)\n",
                        ms, k, 1000.0 / (ms * 48.0));
        }
        // CPU reference
        std::printf("running CPU...\n"); std::fflush(stdout);
        std::vector<float> acc(2560, 0.0f);
        const std::string prefix = "model.language_model.layers." + std::to_string(layer) + ".mlp.experts.";
        std::vector<float> g(640), u(640), d(2560);
        std::vector<uint16_t> h(640);
        for (int i = 0; i < k; ++i) {
            std::string b = prefix + std::to_string(ids[i]);
            Exl3Linear G = m.linear(b + ".gate_proj"), U = m.linear(b + ".up_proj"), D = m.linear(b + ".down_proj");
            cpu::exl3_folded_gemv(G.trellis, G.ki, G.nj, G.bits, (cpu::Exl3Codebook) G.cb, G.suh, G.svh, x.data(), 1, g.data());
            cpu::exl3_folded_gemv(U.trellis, U.ki, U.nj, U.bits, (cpu::Exl3Codebook) U.cb, U.suh, U.svh, x.data(), 1, u.data());
            for (int j = 0; j < 640; ++j) { float gv = g[j]; h[j] = to_h((gv / (1.0f + std::exp(-gv))) * u[j]); }
            cpu::exl3_folded_gemv(D.trellis, D.ki, D.nj, D.bits, (cpu::Exl3Codebook) D.cb, D.suh, D.svh, h.data(), 1, d.data());
            for (int j = 0; j < 2560; ++j) acc[j] += w[i] * d[j];
        }

        double num = 0, den = 0, maxd = 0;
        for (int j = 0; j < 2560; ++j) { double dd = (double) h2f(out_gpu[j]) - acc[j]; num += dd * dd; den += (double) acc[j] * acc[j]; maxd = std::max(maxd, std::fabs(dd)); }
        double rel = std::sqrt(num) / (std::sqrt(den) + 1e-30);
        std::printf("layer %d MoE (%d experts): rel=%.3e max=%.3e %s\n", layer, k, rel, maxd, rel < 2e-2 ? "OK" : "FAIL");
        return rel < 2e-2 ? 0 : 1;
    } catch (const std::exception& e) {
        std::fprintf(stderr, "error: %s\n", e.what());
        return 1;
    }
}
