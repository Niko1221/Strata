// src/kernels/exl3_gpu_parity.cpp - GPU EXL3 reconstruct against the CPU reference (docs/EXL3.md).
//
// Synthetic trellises, no model: the GPU kernel must reproduce strata::kernels::cpu::exl3_reconstruct_weight.
#include <cuda_runtime.h>

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <random>
#include <vector>

#include "strata/kernels/exl3.hpp"
#include "strata/kernels/cpu/exl3.hpp"

namespace {

uint16_t to_h(float v) {
    unsigned f; std::memcpy(&f, &v, 4);
    unsigned sign = (f >> 16) & 0x8000u, m = f & 0x7FFFFFu; int e = (int)((f >> 23) & 0xFF) - 127 + 15;
    if (e <= 0) return (uint16_t)sign;
    if (e >= 0x1F) return (uint16_t)(sign | 0x7C00u);
    return (uint16_t)(sign | (e << 10) | (m >> 13));
}
float h2f(uint16_t h) {
    unsigned sign = (unsigned)(h >> 15) << 31, exp = (h >> 10) & 0x1F, man = h & 0x3FF, f;
    if (exp == 0) { if (!man) f = sign; else { exp = 127 - 15 + 1; while (!(man & 0x400)) { man <<= 1; --exp; } man &= 0x3FF; f = sign | (exp << 23) | (man << 13); } }
    else if (exp == 0x1F) f = sign | 0x7F800000u | (man << 13);
    else f = sign | ((exp + 127 - 15) << 23) | (man << 13);
    float o; std::memcpy(&o, &f, 4); return o;
}

int run(int ki, int nj, int bits, int cb, uint32_t seed) {
    int k = ki * 16, n = nj * 16, words = 256 * bits / 16;
    std::mt19937 rng(seed);
    std::vector<uint16_t> trellis((size_t)ki * nj * words);
    for (auto& x : trellis) x = (uint16_t)(rng() & 0xFFFF);
    std::vector<uint16_t> suh(k), svh(n);
    std::uniform_real_distribution<float> sc(-1.0f, 1.0f);
    for (auto& x : suh) x = to_h(sc(rng) * 0.5f);
    for (auto& x : svh) x = to_h(sc(rng) * 0.5f);

    std::vector<uint16_t> w_cpu((size_t)k * n);
    strata::kernels::cpu::exl3_reconstruct_weight(trellis.data(), ki, nj, bits,
                                                 (strata::kernels::cpu::Exl3Codebook)cb,
                                                 suh.data(), svh.data(), w_cpu.data());

    uint16_t *d_tr = nullptr, *d_suh = nullptr, *d_svh = nullptr, *d_out = nullptr;
    bool ok = true;
    auto chk = [&](cudaError_t e, const char* what) { if (e != cudaSuccess) { std::printf("CUDA %s: %s\n", what, cudaGetErrorString(e)); ok = false; } };
    chk(cudaMalloc(&d_tr, trellis.size() * 2), "tr");
    chk(cudaMalloc(&d_suh, (size_t)k * 2), "suh");
    chk(cudaMalloc(&d_svh, (size_t)n * 2), "svh");
    chk(cudaMalloc(&d_out, (size_t)k * n * 2), "out");
    chk(cudaMemcpy(d_tr, trellis.data(), trellis.size() * 2, cudaMemcpyHostToDevice), "ctr");
    chk(cudaMemcpy(d_suh, suh.data(), (size_t)k * 2, cudaMemcpyHostToDevice), "csuh");
    chk(cudaMemcpy(d_svh, svh.data(), (size_t)n * 2, cudaMemcpyHostToDevice), "csvh");
    strata::kernels::exl3_reconstruct_weight(d_tr, ki, nj, bits, cb, d_suh, d_svh, d_out, nullptr);
    chk(cudaGetLastError(), "launch");
    std::vector<uint16_t> w_gpu((size_t)k * n);
    chk(cudaMemcpy(w_gpu.data(), d_out, (size_t)k * n * 2, cudaMemcpyDeviceToHost), "cout");
    (void)cudaFree(d_tr); (void)cudaFree(d_suh); (void)cudaFree(d_svh); (void)cudaFree(d_out);
    if (!ok) return 1;

    double num = 0, den = 0, maxd = 0; size_t mism = 0, used = 0;
    for (size_t i = 0; i < w_gpu.size(); ++i) {
        double a = h2f(w_gpu[i]), b = h2f(w_cpu[i]);
        bool af = std::isfinite(a), bf = std::isfinite(b);
        if (af != bf) { ++mism; continue; }
        if (!af) continue;
        ++used; double d = a - b; num += d * d; den += b * b; maxd = std::max(maxd, std::fabs(d));
    }
    double rel = std::sqrt(num) / (std::sqrt(den) + 1e-30);
    std::printf("cb=%d %dx%d K=%d: rel=%.3e max_abs=%.3e finiteness_mismatch=%zu\n", cb, k, n, bits, rel, maxd, mism);
    return (rel < 5e-3 && mism == 0) ? 0 : 1;
}

}  // namespace

int main() {
    int fails = 0;
    fails += run(8, 8, 3, 2, 1);
    fails += run(40, 160, 3, 2, 2);      // the real expert down_proj shape
    fails += run(16, 24, 4, 1, 3);       // mcg
    fails += run(16, 16, 2, 0, 4);       // 3inst
    // Every released EXL3 branch uses integer K only; cover every integer bitrate.
    for (int K = 1; K <= 8; ++K) fails += run(8, 8, K, 2, 100 + K);
    std::printf("%s\n", fails ? "FAIL" : "OK");
    return fails ? 1 : 0;
}
