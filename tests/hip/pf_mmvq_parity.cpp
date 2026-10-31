// tests/hip/pf_mmvq_parity.cpp - Gemm::native_mmvq (STRATA_PF_MMVQ=1): the prompt path's quantized dense projections of a
// small chunk through decode's native MMVQ (8 token columns per call) against a host double product on the same
// dequantized weights and the same FP16 activations.  Activations go through Q8_1, so the bar is a rounding-level one
// per token row; one row is scaled x1000.  Also: the shapes it must decline leave Y untouched.
#include "strata/prefill/gemm.hpp"

#include "ggml.h"

#include <cuda_runtime.h>

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <string>
#include <vector>

namespace {
constexpr int64_t N = 640, K = 2560, MAX_T = 512;
void ck(cudaError_t e, const char* w) {
    if (e != cudaSuccess) { std::fprintf(stderr, "%s: %s\n", w, cudaGetErrorString(e)); std::exit(2); }
}
int fails = 0;
void expect(bool ok, const std::string& what) {
    if (!ok) { std::printf("FAIL: %s\n", what.c_str()); ++fails; }
}
}  // namespace

int main() {
    setenv("STRATA_PF_MMVQ", "1", 1);
    setenv("STRATA_PF_MMVQ_MAX_T", std::to_string(MAX_T).c_str(), 1);
    int ndev = 0;
    if (cudaGetDeviceCount(&ndev) != cudaSuccess || ndev == 0) { std::printf("SKIP: no GPU\n"); return 77; }
    cudaStream_t st = nullptr;
    ck(cudaStreamCreate(&st), "stream");
    strata::prefill::Gemm gm;
    std::string err;
    if (!gm.init(st, N * K, err)) { std::printf("FAIL: init: %s\n", err.c_str()); return 1; }

    std::mt19937 rng(7);
    std::normal_distribution<float> nd(0.0f, 1.0f);
    const ggml_type types[] = {GGML_TYPE_Q4_K, GGML_TYPE_Q6_K, GGML_TYPE_IQ4_XS, GGML_TYPE_Q8_0};
    const int64_t Ts[] = {1, 7, 8, 129, 512};

    float* dY = nullptr;
    uint16_t* dX = nullptr;
    void* dW = nullptr;
    ck(cudaMalloc(&dY, (size_t) (MAX_T + 1) * (N + 16) * 4), "malloc Y");
    ck(cudaMalloc(&dX, (size_t) (MAX_T + 1) * K * 2), "malloc X");
    std::vector<float> W((size_t) N * K);
    for (float& v : W) v = 0.05f * nd(rng);

    for (ggml_type type : types) {
        const size_t wbytes = ggml_row_size(type, K) * N;
        std::vector<uint8_t> blocks(wbytes);
        ggml_quantize_chunk(type, W.data(), blocks.data(), 0, N, K, nullptr);
        std::vector<float> Wd((size_t) N * K);
        ggml_get_type_traits(type)->to_float(blocks.data(), Wd.data(), N * K);
        if (dW) cudaFree(dW);
        ck(cudaMalloc(&dW, wbytes), "malloc W");
        ck(cudaMemcpy(dW, blocks.data(), wbytes, cudaMemcpyHostToDevice), "upload W");

        for (int64_t T : Ts) {
            std::vector<uint16_t> Xh((size_t) T * K);
            std::vector<double> Xr((size_t) T * K);
            for (int64_t t = 0; t < T; ++t)
                for (int64_t k = 0; k < K; ++k) {
                    const float v = nd(rng) * (t == T / 2 ? 1000.0f : 1.0f);
                    Xh[t * K + k] = ggml_fp32_to_fp16(v);
                    Xr[t * K + k] = ggml_fp16_to_fp32(Xh[t * K + k]);
                }
            ck(cudaMemcpy(dX, Xh.data(), Xh.size() * 2, cudaMemcpyHostToDevice), "upload X");
            ck(cudaMemset(dY, 0xff, (size_t) T * N * 4), "fill Y");
            const bool took = gm.native_mmvq(dX, type, dW, dY, T, N, K, 0, 0.0f, 0);
            ck(cudaStreamSynchronize(st), "sync");
            const std::string tag = std::string(ggml_type_name(type)) + " T=" + std::to_string(T);
            expect(took, tag + ": the path was declined");
            if (!took) continue;
            std::vector<float> Y((size_t) T * N);
            ck(cudaMemcpy(Y.data(), dY, Y.size() * 4, cudaMemcpyDeviceToHost), "download Y");
            double worst_rms = 0.0, worst_max = 0.0;
            for (int64_t t = 0; t < T; ++t) {
                double e2 = 0, r2 = 0, emax = 0, rmax = 0;
                for (int64_t n = 0; n < N; ++n) {
                    double ref = 0;
                    for (int64_t k = 0; k < K; ++k) ref += Xr[t * K + k] * Wd[n * K + k];
                    const double d = (double) Y[t * N + n] - ref;
                    e2 += d * d; r2 += ref * ref;
                    emax = std::fmax(emax, std::fabs(d)); rmax = std::fmax(rmax, std::fabs(ref));
                }
                worst_rms = std::fmax(worst_rms, std::sqrt(e2 / std::fmax(r2, 1e-30)));
                worst_max = std::fmax(worst_max, emax / std::fmax(rmax, 1e-30));
            }
            std::printf("%-14s rel RMS %.2e  max %.2e\n", tag.c_str(), worst_rms, worst_max);
            expect(std::isfinite(worst_rms) && worst_rms < 2e-2, tag + ": rel RMS too large");
            expect(std::isfinite(worst_max) && worst_max < 5e-2, tag + ": max diff too large");
        }

        // the shapes the path must decline, leaving Y as it was
        const uint32_t marker = 0x7fc00001u;
        std::vector<uint32_t> fill((size_t) 8 * (N + 16), marker);
        auto declines = [&](const std::string& what, int64_t T, int64_t k, int64_t ldy, float beta) {
            ck(cudaMemcpy(dY, fill.data(), fill.size() * 4, cudaMemcpyHostToDevice), "fill Y");
            const bool took = gm.native_mmvq(dX, type, dW, dY, T, N, k, ldy, beta, 0);
            ck(cudaStreamSynchronize(st), "sync");
            std::vector<uint32_t> back(fill.size());
            ck(cudaMemcpy(back.data(), dY, back.size() * 4, cudaMemcpyDeviceToHost), "download Y");
            expect(!took, std::string(ggml_type_name(type)) + ": took " + what);
            expect(back == fill, std::string(ggml_type_name(type)) + ": wrote Y on " + what);
        };
        declines("ldy = N + 16", 8, K, N + 16, 0.0f);
        declines("beta = 1", 8, K, 0, 1.0f);
        declines("T above STRATA_PF_MMVQ_MAX_T", MAX_T + 1, K, 0, 0.0f);
        if (type != GGML_TYPE_Q8_0) declines("K = 640 (not a 256 multiple)", 8, 640, 0, 0.0f);
    }
    // the workspace is allocated once at its fixed size: a larger chunk later must not grow (free + cudaMalloc, a device
    // sync mid-prompt) - device memory stays the same between a 1-token and a 512-token call on a fresh Gemm
    {
        strata::prefill::Gemm g2;
        if (!g2.init(st, N * K, err)) { std::printf("FAIL: init 2: %s\n", err.c_str()); return 1; }
        size_t f0 = 0, f1 = 0, tot = 0;
        expect(g2.native_mmvq(dX, GGML_TYPE_Q8_0, dW, dY, 1, N, K, 0, 0.0f, 0), "workspace: T=1 declined");
        ck(cudaStreamSynchronize(st), "sync");
        ck(cudaMemGetInfo(&f0, &tot), "meminfo");
        expect(g2.native_mmvq(dX, GGML_TYPE_Q8_0, dW, dY, MAX_T, N, K, 0, 0.0f, 0), "workspace: T=512 declined");
        ck(cudaStreamSynchronize(st), "sync");
        ck(cudaMemGetInfo(&f1, &tot), "meminfo");
        std::printf("workspace: free %zu -> %zu MiB\n", f0 >> 20, f1 >> 20);
        expect(f0 == f1, "workspace grew between a 1-token and a 512-token call");
    }
    // a type without a native MMVQ kernel (BF16) is declined
    expect(!gm.native_mmvq(dX, GGML_TYPE_BF16, dW, dY, 8, N, K, 0, 0.0f, 0), "BF16: taken");
    std::printf(fails ? "FAILED: %d\n" : "PASS\n", fails);
    return fails ? 1 : 0;
}
