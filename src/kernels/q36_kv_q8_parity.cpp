// src/kernels/q36_kv_q8_parity.cpp - Qwen3.6 int8 KV primitive round-trip vs a host reference (GPU, no model).
//
// Quantizes random [HD] vectors (varying magnitude, with occasional all-zero groups) and round-trips them
// through q36_kv_q8_quantize / q36_kv_q8_dequantize, checking:
//   1. the int8 codes and fp16 scales are BITWISE equal to the host reference (shared f16/quantize helpers);
//   2. the fp32 round-trip error is within half a quantization step plus the fp16 scale-rounding slack.
#include "strata/kernels/f16_bits.hpp"
#include "strata/kernels/q36_kv_q8.hpp"

#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <vector>

namespace k = strata::kernels;

namespace {
int g_fail = 0;
void ck(cudaError_t e, const char* w) {
    if (e != cudaSuccess) { std::fprintf(stderr, "%s: %s\n", w, cudaGetErrorString(e)); std::exit(2); }
}
template <typename T> T* dalloc(size_t n) {
    T* p = nullptr;
    ck(cudaMalloc(&p, n * sizeof(T) + 64), "malloc");
    ck(cudaMemset(p, 0, n * sizeof(T) + 64), "memset");
    return p;
}
}  // namespace

int main() {
    const int HD = 256, G = HD / k::Q36_KV_Q8_GROUP;
    std::mt19937 rng(19);
    std::normal_distribution<float> nd(0.f, 1.f);
    int8_t* dq = dalloc<int8_t>(HD);
    uint16_t* ds = dalloc<uint16_t>(G);
    float* dx = dalloc<float>(HD);
    float* dy = dalloc<float>(HD);

    double worst_steps = 0.0;
    for (int trial = 0; trial < 200; ++trial) {
        std::vector<float> x(HD);
        const float mag = (trial % 7 == 0) ? 1e-3f : (trial % 11 == 0 ? 40.f : 1.f);
        for (auto& v : x) v = nd(rng) * mag;
        if (trial % 13 == 0) std::fill(x.begin(), x.begin() + k::Q36_KV_Q8_GROUP, 0.f);  // an all-zero group
        // Host reference through the same scale/quantize helpers the device kernel uses.
        std::vector<int8_t> hq(HD);
        std::vector<uint16_t> hs(G);
        for (int g = 0; g < G; ++g) {
            float amax = 0.f;
            for (int t = 0; t < k::Q36_KV_Q8_GROUP; ++t)
                amax = std::fmax(amax, std::fabs(x[g * k::Q36_KV_Q8_GROUP + t]));
            hs[g] = k::q36_kv_q8_scale_bits(amax);
            for (int t = 0; t < k::Q36_KV_Q8_GROUP; ++t)
                hq[g * k::Q36_KV_Q8_GROUP + t] = k::q36_kv_q8_quantize(x[g * k::Q36_KV_Q8_GROUP + t], hs[g]);
        }
        ck(cudaMemcpy(dx, x.data(), HD * 4, cudaMemcpyHostToDevice), "x");
        k::q36_kv_q8_quantize(dq, ds, dx, HD, nullptr);
        k::q36_kv_q8_dequantize(dy, dq, ds, HD, nullptr);
        ck(cudaDeviceSynchronize(), "roundtrip");
        std::vector<int8_t> aq(HD);
        std::vector<uint16_t> as(G);
        std::vector<float> ay(HD);
        ck(cudaMemcpy(aq.data(), dq, HD, cudaMemcpyDeviceToHost), "q");
        ck(cudaMemcpy(as.data(), ds, G * 2, cudaMemcpyDeviceToHost), "s");
        ck(cudaMemcpy(ay.data(), dy, HD * 4, cudaMemcpyDeviceToHost), "y");
        for (int g = 0; g < G; ++g) {
            if (as[g] != hs[g]) {
                if (g_fail < 5) std::fprintf(stderr, "FAIL: scale g %d got %04x want %04x\n", g, as[g], hs[g]);
                ++g_fail;
            }
            for (int t = 0; t < k::Q36_KV_Q8_GROUP; ++t) {
                const int i = g * k::Q36_KV_Q8_GROUP + t;
                if (aq[i] != hq[i]) {
                    if (g_fail < 5) std::fprintf(stderr, "FAIL: code %d got %d want %d\n", i, (int) aq[i], (int) hq[i]);
                    ++g_fail;
                }
                const float sf = k::f32_from_f16(hs[g]);
                const double err = std::fabs((double) ay[i] - (double) x[i]) / (sf > 0 ? sf : 1.0);
                worst_steps = std::max(worst_steps, err);
            }
        }
    }
    // |x - q*sf| <= sf/2 from round-to-nearest, except when the fp16 scale rounds DOWN and the group max clamps
    // at 127: that error is at most amax*2^-11 ~= 127*sf/2048 = 0.062 sf (127/2048 steps).  Never larger than 0.5.
    if (!(worst_steps <= 0.5 + 127.0 / 2048.0 + 1e-3)) {
        std::fprintf(stderr, "FAIL: round-trip error %.3f quantization steps\n", worst_steps);
        ++g_fail;
    }
    std::printf("q36_kv_q8_parity: %s (worst round-trip error %.3f quantization steps)\n", g_fail ? "FAILED" : "OK",
                worst_steps);
    return g_fail ? 1 : 0;
}
