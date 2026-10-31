// src/kernels/mmvf_unr_hash.cpp - FNV-1a of the router-shaped bf16 reads (2560 inputs, 512 outputs; single-column and 1..8-column multi),
// for comparing STRATA_SMALL_UNR=0 against =1 (run twice, compare the printed hashes: they must be equal).
#include "strata/kernels/bf16_gemv.hpp"

#include <cuda_runtime.h>

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <random>
#include <vector>

static uint16_t bf16(float x) {
    uint32_t u;
    std::memcpy(&u, &x, 4);
    return (uint16_t) ((u + 0x7fff + ((u >> 16) & 1)) >> 16);
}

int main() {
    const int n_in = 2560, n_out = 512, TM = 8;
    std::mt19937 rng(11);
    std::normal_distribution<float> nd(0.f, 1.f);
    std::vector<float> x((size_t) TM * n_in);
    std::vector<uint16_t> w((size_t) n_out * n_in);
    for (auto& v : x) v = nd(rng);
    for (auto& v : w) v = bf16(0.05f * nd(rng));
    float *dx, *dy;
    uint16_t* dw;
    cudaMalloc((void**) &dx, x.size() * 4);
    cudaMalloc((void**) &dw, w.size() * 2);
    cudaMalloc((void**) &dy, (size_t) TM * n_out * 4);
    cudaMemcpy(dx, x.data(), x.size() * 4, cudaMemcpyHostToDevice);
    cudaMemcpy(dw, w.data(), w.size() * 2, cudaMemcpyHostToDevice);
    std::vector<float> y((size_t) TM * n_out);
    auto fnv = [&](int rows) {
        unsigned long long h = 1469598103934665603ull;
        for (size_t i = 0; i < (size_t) rows * n_out * 4; ++i) h = (h ^ ((const unsigned char*) y.data())[i]) * 1099511628211ull;
        return h;
    };
    cudaMemset(dy, 0xff, (size_t) TM * n_out * 4);
    strata::kernels::bf16_gemv_fp32_mmvf(dx, dw, dy, n_in, n_out, nullptr);
    cudaDeviceSynchronize();
    cudaMemcpy(y.data(), dy, (size_t) n_out * 4, cudaMemcpyDeviceToHost);
    std::printf("single   %016llx\n", fnv(1));
    for (int T = 1; T <= TM; ++T) {
        cudaMemset(dy, 0xff, (size_t) TM * n_out * 4);
        strata::kernels::bf16_gemv_fp32_mmvf_multi(dx, n_in, dw, dy, n_out, n_in, n_out, T, nullptr);
        cudaDeviceSynchronize();
        cudaMemcpy(y.data(), dy, (size_t) T * n_out * 4, cudaMemcpyDeviceToHost);
        std::printf("multi T%d %016llx\n", T, fnv(T));
    }
    return 0;
}
