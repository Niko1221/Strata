// src/prefill/gemm_parity.cpp - checks the prefill BF16 matrix layout against a scalar reference.
#include "strata/prefill/gemm.hpp"

#include <cuda_runtime.h>

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

namespace {

void check(cudaError_t error, const char* what) {
    if (error != cudaSuccess) {
        std::fprintf(stderr, "%s: %s\n", what, cudaGetErrorString(error));
        std::exit(1);
    }
}

uint16_t bf16(float value) {
    uint32_t bits = 0;
    std::memcpy(&bits, &value, sizeof(bits));
    return static_cast<uint16_t>(bits >> 16);
}

float fp32(uint16_t value) {
    uint32_t bits = static_cast<uint32_t>(value) << 16;
    float result = 0.0f;
    std::memcpy(&result, &bits, sizeof(result));
    return result;
}

}  // namespace

int main() {
    // Deliberately non-square and not tile-aligned: swapping the shared-memory K/N indices must fail this case.
    constexpr int T = 19, N = 23, K = 37, LDY = 29;
    std::vector<uint16_t> x(T * K), w(N * K);
    std::vector<float> initial(T * LDY), expected(T * LDY), actual(T * LDY);
    for (int t = 0; t < T; ++t) {
        for (int k = 0; k < K; ++k) x[t * K + k] = bf16(static_cast<float>(((t * 7 + k * 3) % 17) - 8) / 4.0f);
    }
    for (int n = 0; n < N; ++n) {
        for (int k = 0; k < K; ++k) w[n * K + k] = bf16(static_cast<float>(((n * 5 - k * 2) % 19) - 9) / 8.0f);
    }
    for (int t = 0; t < T; ++t) {
        for (int n = 0; n < LDY; ++n) initial[t * LDY + n] = static_cast<float>((t + n) % 11) / 16.0f;
    }
    expected = initial;
    for (int t = 0; t < T; ++t) {
        for (int n = 0; n < N; ++n) {
            float sum = 0.0f;
            for (int k = 0; k < K; ++k) sum = std::fma(fp32(x[t * K + k]), fp32(w[n * K + k]), sum);
            expected[t * LDY + n] += sum;
        }
    }

    uint16_t *dx = nullptr, *dw = nullptr;
    float* dy = nullptr;
    check(cudaMalloc(&dx, x.size() * sizeof(uint16_t)), "cudaMalloc X");
    check(cudaMalloc(&dw, w.size() * sizeof(uint16_t)), "cudaMalloc W");
    check(cudaMalloc(&dy, initial.size() * sizeof(float)), "cudaMalloc Y");
    check(cudaMemcpy(dx, x.data(), x.size() * sizeof(uint16_t), cudaMemcpyHostToDevice), "copy X");
    check(cudaMemcpy(dw, w.data(), w.size() * sizeof(uint16_t), cudaMemcpyHostToDevice), "copy W");
    check(cudaMemcpy(dy, initial.data(), initial.size() * sizeof(float), cudaMemcpyHostToDevice), "copy Y");

    std::string error;
    strata::prefill::Gemm gemm;
    if (!gemm.init(nullptr, 0, error)) {
        std::fprintf(stderr, "%s\n", error.c_str());
        return 1;
    }
    gemm.bf16(dx, dw, dy, T, N, K, LDY, 1.0f);
    check(cudaDeviceSynchronize(), "BF16 GEMM");
    check(cudaMemcpy(actual.data(), dy, actual.size() * sizeof(float), cudaMemcpyDeviceToHost), "copy result");

    int mismatches = 0;
    float largest = 0.0f;
    for (int t = 0; t < T; ++t) {
        for (int n = 0; n < N; ++n) {
            const float delta = std::fabs(actual[t * LDY + n] - expected[t * LDY + n]);
            if (delta > 1e-5f) ++mismatches;
            largest = std::fmax(largest, delta);
        }
        for (int n = N; n < LDY; ++n) {
            if (actual[t * LDY + n] != initial[t * LDY + n]) ++mismatches;
        }
    }
    std::printf("prefill BF16 GEMM: %dx%dx%d, mismatches=%d, max_abs_error=%.9g\n", T, N, K, mismatches,
                static_cast<double>(largest));

    cudaFree(dx);
    cudaFree(dw);
    cudaFree(dy);
    return mismatches ? 1 : 0;
}
