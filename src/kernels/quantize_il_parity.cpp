// src/kernels/quantize_il_parity.cpp - native_quantize_q8_1_il (quantize + interleave in one launch, for the 2-4 column
// projection inputs) against native_quantize_q8_1 followed by native_q8_1_interleave, byte for byte: the plain q8_1
// blocks and the interleaved copy must both be identical.
//
//     build/quantize_il_parity
//
// Inputs: random normals per block, plus blocks of exact zeros, huge (1e4) and tiny (1e-30) magnitudes, so the
// scale path (d = amax / 127, q8_1_finite) and the zero-amax path are both exercised. Shapes cover the window's
// projection inputs (N, the GDN value width, NH*HD). Exit status 1 on any byte difference.
#include "strata/kernels/native_mmvq.hpp"

#include <cuda_runtime.h>

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <random>
#include <vector>

namespace {

bool check(cudaError_t e, const char* what) {
    if (e != cudaSuccess) {
        std::fprintf(stderr, "quantize_il_parity: %s: %s\n", what, cudaGetErrorString(e));
        return false;
    }
    return true;
}

// one case: n_in x ncols inputs, both paths, compare the full buffers (both zeroed first)
bool run_case(int n_in, int ncols, unsigned seed, cudaStream_t s) {
    using namespace strata::kernels;
    const std::size_t xf_n = std::size_t(n_in) * std::size_t(ncols);
    std::vector<float> h(xf_n);
    std::mt19937 rng(seed);
    std::normal_distribution<float> nd(0.0f, 1.0f);
    for (std::size_t i = 0; i < xf_n; ++i) {
        const std::size_t blk = i / 32;
        switch (blk % 8) {
            case 0: h[i] = 0.0f; break;                       // an all-zero block: amax 0
            case 1: h[i] = nd(rng) * 1.0e4f; break;           // huge
            case 2: h[i] = nd(rng) * 1.0e-30f; break;         // tiny
            default: h[i] = nd(rng); break;
        }
    }
    const std::size_t qb = native_q8_1_bytes(n_in, ncols);
    const std::size_t ib = native_q8_1_il_bytes(n_in, ncols);
    float* x = nullptr;
    void *xq_a = nullptr, *xq_b = nullptr, *xil_a = nullptr, *xil_b = nullptr;
    bool ok = check(cudaMalloc(&x, xf_n * sizeof(float)), "malloc x") && check(cudaMalloc(&xq_a, qb), "malloc xq_a") &&
              check(cudaMalloc(&xq_b, qb), "malloc xq_b") && check(cudaMalloc(&xil_a, ib), "malloc xil_a") &&
              check(cudaMalloc(&xil_b, ib), "malloc xil_b");
    if (!ok) return false;
    ok = check(cudaMemcpy(x, h.data(), xf_n * sizeof(float), cudaMemcpyHostToDevice), "copy x") &&
         check(cudaMemset(xq_a, 0, qb), "zero xq_a") && check(cudaMemset(xq_b, 0, qb), "zero xq_b") &&
         check(cudaMemset(xil_a, 0, ib), "zero xil_a") && check(cudaMemset(xil_b, 0, ib), "zero xil_b");
    if (ok) {
        // path A: the pair. Path B: the fused launch.
        native_quantize_q8_1(x, xq_a, n_in, ncols, s);
        native_q8_1_interleave(xq_a, xil_a, n_in, ncols, s);
        native_quantize_q8_1_il(x, xq_b, xil_b, n_in, ncols, s);
        ok = check(cudaStreamSynchronize(s), "sync");
    }
    std::vector<std::uint8_t> qa(qb), qbv(qb), ia(ib), ib_v(ib);
    if (ok) {
        ok = check(cudaMemcpy(qa.data(), xq_a, qb, cudaMemcpyDeviceToHost), "copy xq_a") &&
             check(cudaMemcpy(qbv.data(), xq_b, qb, cudaMemcpyDeviceToHost), "copy xq_b") &&
             check(cudaMemcpy(ia.data(), xil_a, ib, cudaMemcpyDeviceToHost), "copy xil_a") &&
             check(cudaMemcpy(ib_v.data(), xil_b, ib, cudaMemcpyDeviceToHost), "copy xil_b");
    }
    (void) cudaFree(x);
    (void) cudaFree(xq_a);
    (void) cudaFree(xq_b);
    (void) cudaFree(xil_a);
    (void) cudaFree(xil_b);
    if (!ok) return false;
    std::size_t dq = 0, di = 0;
    for (std::size_t i = 0; i < qb; ++i) dq += qa[i] != qbv[i];
    for (std::size_t i = 0; i < ib; ++i) di += ia[i] != ib_v[i];
    const bool same = dq == 0 && di == 0;
    std::printf("%s n_in=%d ncols=%d seed=%u  plain-q8_1 bytes differing=%zu/%zu  il bytes differing=%zu/%zu\n",
                same ? "PASS" : "FAIL", n_in, ncols, seed, dq, qb, di, ib);
    return same;
}

}  // namespace

int main() {
    cudaStream_t s = nullptr;
    if (!check(cudaStreamCreate(&s), "stream")) return 1;
    bool all = true;
    const int shapes[] = {32, 2560, 4096, 8192, 10240};   // N, the GDN value width, NH*HD class widths
    for (int n_in : shapes)
        for (int ncols = 2; ncols <= 4; ++ncols)
            for (unsigned seed : {1u, 2u, 3u}) all = run_case(n_in, ncols, seed, s) && all;
    (void) cudaStreamDestroy(s);
    std::printf("%s\n", all ? "ALL PASS" : "FAILURES");
    return all ? 0 : 1;
}
