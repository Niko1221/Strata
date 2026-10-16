// src/kernels/exl3_gpu_model_check.cpp - run the GPU EXL3 GEMV on REAL expert weights from the model and
// compare to the CPU reference.  This is the validation that the fused kernel works on the actual file, not
// just synthetic fixtures.  Needs the downloaded model; not a ctest.
//
// Usage: exl3_gpu_model_check <model_dir> <base_name> [base_name2 ...]
#include "strata/artifact/exl3_model.hpp"
#include "strata/kernels/exl3.hpp"
#include "strata/kernels/cpu/exl3.hpp"

#include <cuda_runtime.h>

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <random>
#include <string>
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

static int check(const Exl3Model& m, const std::string& base, uint32_t seed) {
    Exl3Linear L = m.linear(base);
    int k = L.in(), n = L.out(), words = 256 * L.bits / 16;
    int sb = 0, vb = 0;
    for (int i = 0; i < k; ++i) if (!std::isfinite(h2f(L.suh[i]))) ++sb;
    for (int j = 0; j < n; ++j) if (!std::isfinite(h2f(L.svh[j]))) ++vb;
    std::printf("     suh_bad=%d/%d svh_bad=%d/%d suh[0..3]=%.4g %.4g %.4g %.4g svh[0..3]=%.4g %.4g %.4g %.4g\n",
                sb, k, vb, n, h2f(L.suh[0]), h2f(L.suh[1]), h2f(L.suh[2]), h2f(L.suh[3]),
                h2f(L.svh[0]), h2f(L.svh[1]), h2f(L.svh[2]), h2f(L.svh[3]));
    std::mt19937 rng(seed);
    std::uniform_real_distribution<float> sc(-1.0f, 1.0f);
    std::vector<uint16_t> x(k);
    for (auto& v : x) v = to_h(sc(rng) * 0.3f);
    { int xb = 0; for (auto v : x) if (!std::isfinite(h2f(v))) ++xb; std::printf("     x nonfinite=%d/%d x[0..3]=%.4g %.4g %.4g %.4g\n", xb, k, h2f(x[0]), h2f(x[1]), h2f(x[2]), h2f(x[3])); }

    std::vector<uint16_t> what((size_t) k * n);
    cpu::exl3_decode_weight_hat(L.trellis, L.ki, L.nj, L.bits, (cpu::Exl3Codebook) L.cb, what.data());
    int wb = 0; double wmax = 0;
    for (size_t i = 0; i < what.size(); ++i) { float f = h2f(what[i]); if (!std::isfinite(f)) ++wb; else wmax = std::max(wmax, (double) std::fabs(f)); }
    std::printf("     W_hat nonfinite=%d/%zu maxabs=%.4g\n", wb, what.size(), wmax);

    // CPU reference
    std::vector<float> y_cpu(n);
    cpu::exl3_folded_gemv(L.trellis, L.ki, L.nj, L.bits, (cpu::Exl3Codebook) L.cb, L.suh, L.svh,
                          x.data(), 1, y_cpu.data());

    // GPU
    uint16_t *d_tr = nullptr, *d_suh = nullptr, *d_svh = nullptr, *d_x = nullptr, *d_y = nullptr;
    bool ok = true;
    auto chk = [&](cudaError_t e, const char* w) { if (e != cudaSuccess) { std::printf("CUDA %s: %s\n", w, cudaGetErrorString(e)); ok = false; } };
    size_t tr_bytes = (size_t) L.ki * L.nj * words * 2;
    chk(cudaMalloc(&d_tr, tr_bytes), "tr");
    chk(cudaMalloc(&d_suh, (size_t) k * 2), "suh");
    chk(cudaMalloc(&d_svh, (size_t) n * 2), "svh");
    chk(cudaMalloc(&d_x, (size_t) k * 2), "x");
    chk(cudaMalloc(&d_y, (size_t) n * 2), "y");
    chk(cudaMemcpy(d_tr, L.trellis, tr_bytes, cudaMemcpyHostToDevice), "ctr");
    chk(cudaMemcpy(d_suh, L.suh, (size_t) k * 2, cudaMemcpyHostToDevice), "csuh");
    chk(cudaMemcpy(d_svh, L.svh, (size_t) n * 2, cudaMemcpyHostToDevice), "csvh");
    chk(cudaMemcpy(d_x, x.data(), (size_t) k * 2, cudaMemcpyHostToDevice), "cx");
    exl3_gemv(d_x, d_suh, d_svh, d_tr, L.ki, L.nj, L.bits, L.cb, d_y, nullptr);
    chk(cudaGetLastError(), "launch");
    std::vector<uint16_t> yg(n);
    chk(cudaMemcpy(yg.data(), d_y, (size_t) n * 2, cudaMemcpyDeviceToHost), "cy");
    {   // f32 activation path (exl3_gemv_f32), the one gemv_quantized will use
        std::vector<float> xf(k);
        for (int i = 0; i < k; ++i) xf[i] = h2f(x[i]);
        float* d_x32 = nullptr; float* d_y32 = nullptr;
        chk(cudaMalloc(&d_x32, (size_t) k * 4), "x32"); chk(cudaMalloc(&d_y32, (size_t) n * 4), "y32");
        chk(cudaMemcpy(d_x32, xf.data(), (size_t) k * 4, cudaMemcpyHostToDevice), "cx32");
        exl3_gemv_f32(d_x32, d_suh, d_svh, d_tr, L.ki, L.nj, L.bits, L.cb, d_y32, nullptr);
        chk(cudaGetLastError(), "launch32");
        std::vector<float> y32(n);
        chk(cudaMemcpy(y32.data(), d_y32, (size_t) n * 4, cudaMemcpyDeviceToHost), "cy32");
        (void) cudaFree(d_x32); (void) cudaFree(d_y32);
        double n2 = 0, d2 = 0;
        for (int j = 0; j < n; ++j) { double dd = y32[j] - y_cpu[j]; n2 += dd * dd; d2 += (double) y_cpu[j] * y_cpu[j]; }
        std::printf("     f32 path rel=%.3e\n", std::sqrt(n2) / (std::sqrt(d2) + 1e-30));
    }
    (void) cudaFree(d_tr); (void) cudaFree(d_suh); (void) cudaFree(d_svh); (void) cudaFree(d_x); (void) cudaFree(d_y);
    if (!ok) return 1;

    int inf_g = 0, inf_c = 0;
    for (int j = 0; j < n; ++j) { if (!std::isfinite(h2f(yg[j]))) ++inf_g; if (!std::isfinite(y_cpu[j])) ++inf_c; }
    double num = 0, den = 0, maxd = 0;
    for (int j = 0; j < n; ++j) {
        double d = (double) h2f(yg[j]) - y_cpu[j];
        num += d * d; den += (double) y_cpu[j] * y_cpu[j]; maxd = std::max(maxd, std::fabs(d));
    }
    std::printf("     nonfinite gpu=%d cpu=%d | yg[0..3]=%.4g %.4g %.4g %.4g | yc[0..3]=%.4g %.4g %.4g %.4g\n",
                inf_g, inf_c, h2f(yg[0]), h2f(yg[1]), h2f(yg[2]), h2f(yg[3]), y_cpu[0], y_cpu[1], y_cpu[2], y_cpu[3]);
    double rel = std::sqrt(num) / (std::sqrt(den) + 1e-30);
    std::printf("  %-72s in=%d out=%d K=%d cb=%d  rel=%.3e max=%.3e %s\n", base.c_str(), k, n, L.bits, L.cb,
                rel, maxd, rel < 1e-2 ? "OK" : "FAIL");
    return rel < 1e-2 ? 0 : 1;
}

static int check_ffn(const Exl3Model& m, const std::string& prefix, uint32_t seed) {
    Exl3Linear G = m.linear(prefix + ".gate_proj");
    Exl3Linear U = m.linear(prefix + ".up_proj");
    Exl3Linear D = m.linear(prefix + ".down_proj");
    int k = G.in(), ff = G.out(), n = D.out();
    std::mt19937 rng(seed);
    std::uniform_real_distribution<float> sc(-1.0f, 1.0f);
    std::vector<uint16_t> x(k);
    for (auto& v : x) v = to_h(sc(rng) * 0.3f);

    // CPU reference: gate, up, silu*up (fp16), down, weighted.
    std::vector<float> g(ff), u(ff), d(n);
    cpu::exl3_folded_gemv(G.trellis, G.ki, G.nj, G.bits, (cpu::Exl3Codebook) G.cb, G.suh, G.svh, x.data(), 1, g.data());
    cpu::exl3_folded_gemv(U.trellis, U.ki, U.nj, U.bits, (cpu::Exl3Codebook) U.cb, U.suh, U.svh, x.data(), 1, u.data());
    std::vector<uint16_t> h(ff);
    for (int i = 0; i < ff; ++i) { float gv = g[i]; h[i] = to_h((gv / (1.0f + std::exp(-gv))) * u[i]); }
    cpu::exl3_folded_gemv(D.trellis, D.ki, D.nj, D.bits, (cpu::Exl3Codebook) D.cb, D.suh, D.svh, h.data(), 1, d.data());
    float w = 0.7f;

    Exl3Mat gm{G.trellis, G.suh, G.svh, G.ki, G.nj, G.bits, G.cb};
    Exl3Mat um{U.trellis, U.suh, U.svh, U.ki, U.nj, U.bits, U.cb};
    Exl3Mat dm{D.trellis, D.suh, D.svh, D.ki, D.nj, D.bits, D.cb};
    uint16_t *d_tr = nullptr, *d_suh = nullptr, *d_svh = nullptr, *d_x = nullptr, *d_y = nullptr;
    bool ok = true;
    auto chk = [&](cudaError_t e, const char* w2) { if (e != cudaSuccess) { std::printf("CUDA %s: %s\n", w2, cudaGetErrorString(e)); ok = false; } };
    size_t tr_bytes = (size_t) G.ki * G.nj * (256 * G.bits / 16) * 2;
    chk(cudaMalloc(&d_tr, tr_bytes * 3), "tr");
    chk(cudaMalloc(&d_suh, (size_t)(k + k + ff) * 2), "suh");
    chk(cudaMalloc(&d_svh, (size_t)(ff + ff + n) * 2), "svh");
    chk(cudaMalloc(&d_x, (size_t) k * 2), "x");
    chk(cudaMalloc(&d_y, (size_t) n * 2), "y");
    uint16_t* tr = d_tr; uint16_t* suh = d_suh; uint16_t* svh = d_svh;
    auto up_tr = [&](const Exl3Linear& L, Exl3Mat& M) {
        size_t tb = (size_t) L.ki * L.nj * (256 * L.bits / 16) * 2;
        chk(cudaMemcpy(tr, L.trellis, tb, cudaMemcpyHostToDevice), "tr");
        chk(cudaMemcpy(suh, L.suh, (size_t) L.in() * 2, cudaMemcpyHostToDevice), "suh");
        chk(cudaMemcpy(svh, L.svh, (size_t) L.out() * 2, cudaMemcpyHostToDevice), "svh");
        M.trellis = tr; M.suh = suh; M.svh = svh;
        tr += tb / 2; suh += L.in(); svh += L.out();
    };
    up_tr(G, gm); up_tr(U, um); up_tr(D, dm);
    chk(cudaMemcpy(d_x, x.data(), (size_t) k * 2, cudaMemcpyHostToDevice), "cx");
    exl3_moe_ffn(&gm, &um, &dm, &w, 1, d_x, d_y, nullptr);
    chk(cudaGetLastError(), "launch");
    std::vector<uint16_t> yg(n);
    chk(cudaMemcpy(yg.data(), d_y, (size_t) n * 2, cudaMemcpyDeviceToHost), "cy");
    (void) cudaFree(d_tr); (void) cudaFree(d_suh); (void) cudaFree(d_svh); (void) cudaFree(d_x); (void) cudaFree(d_y);
    if (!ok) return 1;
    double num = 0, den = 0, maxd = 0;
    for (int j = 0; j < n; ++j) { double dd = (double) h2f(yg[j]) - w * d[j]; num += dd * dd; den += (double) w * d[j] * w * d[j]; maxd = std::max(maxd, std::fabs(dd)); }
    double rel = std::sqrt(num) / (std::sqrt(den) + 1e-30);
    std::printf("  FFN %-60s k=%d ff=%d n=%d rel=%.3e max=%.3e %s\n", prefix.c_str(), k, ff, n, rel, maxd,
                rel < 1e-2 ? "OK" : "FAIL");
    return rel < 1e-2 ? 0 : 1;
}

int main(int argc, char** argv) {    if (argc < 3) { std::fprintf(stderr, "usage: %s <model_dir> <base_name> [...]\n", argv[0]); return 2; }
    try {
        Exl3Model m(argv[1]);
        int fails = 0;
        if (std::string(argv[2]) == "--ffn") {
            for (int i = 3; i < argc; ++i) fails += check_ffn(m, argv[i], (uint32_t) i);
        } else {
            for (int i = 2; i < argc; ++i) fails += check(m, argv[i], (uint32_t) i);
        }
        std::printf("%s\n", fails ? "FAIL" : "OK");
        return fails ? 1 : 0;
    } catch (const std::exception& e) {
        std::fprintf(stderr, "error: %s\n", e.what());
        return 1;
    }
}
