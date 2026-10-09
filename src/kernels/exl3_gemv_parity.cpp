// src/kernels/exl3_gemv_parity.cpp - GPU fused EXL3 decode-GEMV against the CPU reference (docs/EXL3.md).
//
// y = H(x . suh) @ W_hat . H . svh, without materializing W.  Reference: x @ reconstruct_weight(...).
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
    std::vector<uint16_t> suh(k), svh(n), x(k);
    std::uniform_real_distribution<float> sc(-1.0f, 1.0f);
    for (auto& v : suh) v = to_h(sc(rng) * 0.5f);
    for (auto& v : svh) v = to_h(sc(rng) * 0.5f);
    for (auto& v : x) v = to_h(sc(rng) * 0.3f);

    // reference: x @ reconstruct_weight
    std::vector<uint16_t> w((size_t)k * n);
    strata::kernels::cpu::exl3_reconstruct_weight(trellis.data(), ki, nj, bits,
                                                 (strata::kernels::cpu::Exl3Codebook)cb,
                                                 suh.data(), svh.data(), w.data());
    std::vector<double> ref(n, 0.0);
    for (int j = 0; j < n; ++j) {
        double acc = 0;
        for (int i = 0; i < k; ++i) acc += (double)h2f(x[i]) * h2f(w[(size_t)i * n + j]);
        ref[j] = acc;
    }

    uint16_t *d_x = nullptr, *d_suh = nullptr, *d_svh = nullptr, *d_tr = nullptr, *d_y = nullptr;
    bool ok = true;
    auto chk = [&](cudaError_t e, const char* what) { if (e != cudaSuccess) { std::printf("CUDA %s: %s\n", what, cudaGetErrorString(e)); ok = false; } };
    chk(cudaMalloc(&d_x, (size_t)k * 2), "x");
    chk(cudaMalloc(&d_suh, (size_t)k * 2), "suh");
    chk(cudaMalloc(&d_svh, (size_t)n * 2), "svh");
    chk(cudaMalloc(&d_tr, trellis.size() * 2), "tr");
    chk(cudaMalloc(&d_y, (size_t)n * 2), "y");
    chk(cudaMemcpy(d_x, x.data(), (size_t)k * 2, cudaMemcpyHostToDevice), "cx");
    chk(cudaMemcpy(d_suh, suh.data(), (size_t)k * 2, cudaMemcpyHostToDevice), "csuh");
    chk(cudaMemcpy(d_svh, svh.data(), (size_t)n * 2, cudaMemcpyHostToDevice), "csvh");
    chk(cudaMemcpy(d_tr, trellis.data(), trellis.size() * 2, cudaMemcpyHostToDevice), "ctr");
    strata::kernels::exl3_gemv(d_x, d_suh, d_svh, d_tr, ki, nj, bits, cb, d_y, nullptr);
    chk(cudaGetLastError(), "launch");
    std::vector<uint16_t> y(n);
    chk(cudaMemcpy(y.data(), d_y, (size_t)n * 2, cudaMemcpyDeviceToHost), "cy");
    (void)cudaFree(d_x); (void)cudaFree(d_suh); (void)cudaFree(d_svh); (void)cudaFree(d_tr); (void)cudaFree(d_y);
    if (!ok) return 1;

    double num = 0, den = 0, maxd = 0;
    for (int j = 0; j < n; ++j) { double a = h2f(y[j]); double d = a - ref[j]; num += d * d; den += ref[j] * ref[j]; maxd = std::max(maxd, std::fabs(d)); }
    double rel = std::sqrt(num) / (std::sqrt(den) + 1e-30);
    std::printf("gemv cb=%d %dx%d K=%d: rel=%.3e max_abs=%.3e\n", cb, k, n, bits, rel, maxd);
    return rel < 1e-2 ? 0 : 1;
}

}  // namespace

int main() {
    int fails = 0;
    fails += run(8, 8, 3, 2, 11);
    fails += run(40, 160, 3, 2, 12);     // real expert shape
    fails += run(16, 24, 4, 1, 13);      // mcg
    fails += run(16, 16, 2, 0, 14);      // 3inst
    // Every released EXL3 branch uses integer K only (2.05: K=2/4, 3.05: 3/5, 4.05: 4/6, 5.05: 5/6/7,
    // 6.05: 6/8).  Cover every integer bitrate, and the real expert / attention shapes at those K.
    for (int K = 1; K <= 8; ++K) fails += run(8, 8, K, 2, 100 + K);          // mul1, every integer bitrate
    for (int K = 2; K <= 6; ++K) fails += run(40, 160, K, 2, 200 + K);       // expert down (640x2560)
    fails += run(160, 640, 5, 2, 305);                                       // attn in_proj_qkv (2560x10240), one K
    std::printf("%s\n", fails ? "FAIL" : "OK");
    return fails ? 1 : 0;
}
