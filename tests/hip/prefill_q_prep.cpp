// tests/hip/prefill_q_prep.cpp - the prompt path's q preparation as one kernel (q_prep, STRATA_QPREP=1) against the three
// kernels it replaces (split_q, rms_rows over the 24 q heads, rope): every float must be BITWISE equal, over chunk
// lengths around the block sizes, from a nonzero start position.  Synthetic q_full [T, 24, 512] (q | gate per head).
#include "strata/prefill/kernels.hpp"

#include <cuda_runtime.h>

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <vector>

namespace {
void ck(cudaError_t e, const char* w) {
    if (e != cudaSuccess) { std::fprintf(stderr, "%s: %s\n", w, cudaGetErrorString(e)); std::exit(2); }
}
template <typename T> T* dev(const std::vector<T>& h) {
    T* d = nullptr;
    ck(cudaMalloc(&d, h.size() * sizeof(T)), "malloc");
    ck(cudaMemcpy(d, h.data(), h.size() * sizeof(T), cudaMemcpyHostToDevice), "upload");
    return d;
}
}  // namespace

int main() {
    namespace pf = strata::prefill;
    constexpr int HEADS = 24, DIM = 256;
    std::mt19937 rng(7);
    std::normal_distribution<float> nd(0.f, 3.f);
    std::uniform_real_distribution<float> ud(0.5f, 1.5f);
    const float eps = 1e-6f;
    int failures = 0;
    for (const int64_t T : {1, 5, 64, 1000, 4089}) {
        for (const int64_t pos0 : {0, 4089, 100000}) {
            std::vector<float> full((size_t) T * HEADS * 2 * DIM), w(DIM);
            for (auto& x : full) x = nd(rng);
            for (auto& x : w) x = ud(rng);
            const float* dfull = dev(full);
            const float* dw = dev(w);
            float *q_ref = nullptr, *q_new = nullptr;
            const size_t n = (size_t) T * HEADS * DIM;
            ck(cudaMalloc(&q_ref, n * 4), "malloc");
            ck(cudaMalloc(&q_new, n * 4), "malloc");
            const auto sc = strata::kernels::rope_scaling();
            pf::split_q(dfull, q_ref, T, nullptr);
            pf::rms_rows(q_ref, dw, T * HEADS, DIM, DIM, eps, nullptr);
            pf::rope(q_ref, T, HEADS, DIM, HEADS * DIM, pos0, sc, nullptr);
            pf::q_prep(dfull, dw, q_new, T, eps, pos0, sc, nullptr);
            ck(cudaDeviceSynchronize(), "run");
            std::vector<float> a(n), b(n);
            ck(cudaMemcpy(a.data(), q_ref, n * 4, cudaMemcpyDeviceToHost), "download");
            ck(cudaMemcpy(b.data(), q_new, n * 4, cudaMemcpyDeviceToHost), "download");
            size_t diff = 0;
            for (size_t i = 0; i < n; ++i) diff += std::memcmp(&a[i], &b[i], 4) != 0;
            failures += diff != 0;
            std::printf("%s T %lld pos0 %lld: %zu of %zu values differ\n", diff ? "FAIL" : "PASS", (long long) T,
                        (long long) pos0, diff, n);
            cudaFree((void*) dfull); cudaFree((void*) dw); cudaFree(q_ref); cudaFree(q_new);
        }
    }
    std::printf("hip_prefill_q_prep: %s\n", failures ? "FAIL" : "PASS");
    return failures ? 1 : 0;
}
