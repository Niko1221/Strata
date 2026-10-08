// prefill_combine_write_test - strata::prefill::moe_combine_write_norm_rs (the MoE half's combine and the
// hyper-connection write with the next half's norm in one kernel) against moe_combine + gr_write_norm_rs, bit for bit:
// R, the row scales and the BF16 image (hi and lo), at the plain and a padded token stride.  Synthetic, no model.
// Exit 77 without a CUDA device.
#include "strata/prefill/kernels.hpp"

#include <cuda_runtime.h>

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <initializer_list>
#include <random>
#include <vector>

namespace sp = strata::prefill;

namespace {
constexpr int64_t N = 2560, HC = 4, D = N * HC, K = 10;

template <class T> T* dev(const std::vector<T>& h) {
    T* d = nullptr;
    cudaMalloc(&d, h.size() * sizeof(T));
    cudaMemcpy(d, h.data(), h.size() * sizeof(T), cudaMemcpyHostToDevice);
    return d;
}
template <class T> std::vector<T> host(const T* d, size_t n) {
    std::vector<T> h(n);
    cudaMemcpy(h.data(), d, n * sizeof(T), cudaMemcpyDeviceToHost);
    return h;
}

int run_case(int64_t T, int64_t ldx, bool lo, uint32_t seed) {
    std::mt19937 rng(seed);
    std::normal_distribution<float> nd(0.f, 1.f);
    std::uniform_real_distribution<float> u01(0.f, 1.f);
    const int64_t rows = T * K + 37;   // Dm has more rows than the tokens route (a CPU share's tail, padding)
    std::vector<float> Dm((size_t) (rows * N)), shared((size_t) (T * N)), sg((size_t) T), w((size_t) (T * K)),
        R((size_t) (T * D)), inj((size_t) (T * HC)), wn((size_t) D);
    for (auto& v : Dm) v = nd(rng) * 0.3f;
    for (auto& v : shared) v = nd(rng) * 0.1f;
    for (auto& v : sg) v = nd(rng);
    for (auto& v : w) v = u01(rng);
    for (auto& v : R) v = nd(rng) * 4.f;
    for (auto& v : inj) v = nd(rng);
    for (auto& v : wn) v = 0.5f + u01(rng);
    std::vector<int32_t> slot((size_t) (T * K));
    for (int64_t i = 0; i < T * K; ++i) slot[(size_t) i] = (int32_t) ((i * 7919 + 13) % (T * K));   // a permutation
    float *dDm = dev(Dm), *dsh = dev(shared), *dsg = dev(sg), *dw = dev(w), *dinj = dev(inj), *dwn = dev(wn);
    int32_t* dslot = dev(slot);
    float *dR0 = dev(R), *dR1 = dev(R), *dbo = nullptr, *drs0 = nullptr, *drs1 = nullptr;
    uint16_t *dx0 = nullptr, *dx1 = nullptr, *dl0 = nullptr, *dl1 = nullptr;
    const size_t nx = (size_t) (T * ldx);
    cudaMalloc(&dbo, (size_t) (T * N) * 4);
    cudaMalloc(&drs0, (size_t) (T * HC) * 4); cudaMalloc(&drs1, (size_t) (T * HC) * 4);
    cudaMalloc(&dx0, nx * 2); cudaMalloc(&dx1, nx * 2); cudaMalloc(&dl0, nx * 2); cudaMalloc(&dl1, nx * 2);
    for (uint16_t* p : {dx0, dx1, dl0, dl1}) cudaMemset(p, 0x5a, nx * 2);
    sp::moe_combine(dDm, dslot, dw, dsh, dsg, dbo, T, nullptr);
    sp::gr_write_norm_rs(dR0, dbo, dinj, HC, dwn, 1e-6f, drs0, dx0, T, nullptr, lo ? dl0 : nullptr, ldx);
    const bool took = sp::moe_combine_write_norm_rs(dDm, dslot, dw, dsh, dsg, dR1, dinj, HC, dwn, 1e-6f, drs1, dx1, T,
                                                    nullptr, lo ? dl1 : nullptr, ldx);
    const cudaError_t e = cudaDeviceSynchronize();
    if (e != cudaSuccess || !took) {
        std::printf("  T %lld: %s\n", (long long) T, e != cudaSuccess ? cudaGetErrorString(e) : "not taken");
        return 1;
    }
    const auto R0 = host(dR0, (size_t) (T * D)), R1 = host(dR1, (size_t) (T * D));
    const auto rs0 = host(drs0, (size_t) (T * HC)), rs1 = host(drs1, (size_t) (T * HC));
    const auto x0 = host(dx0, nx), x1 = host(dx1, nx), l0 = host(dl0, nx), l1 = host(dl1, nx);
    const bool same = std::memcmp(R0.data(), R1.data(), R0.size() * 4) == 0 &&
                      std::memcmp(rs0.data(), rs1.data(), rs0.size() * 4) == 0 &&
                      std::memcmp(x0.data(), x1.data(), nx * 2) == 0 && std::memcmp(l0.data(), l1.data(), nx * 2) == 0;
    std::printf("  T %5lld ldx %lld%s: %s\n", (long long) T, (long long) ldx, lo ? " lo" : "",
                same ? "R, rs, xn16, xn16_lo identical" : "DIFFER");
    for (void* p : std::initializer_list<void*>{dDm, dsh, dsg, dw, dinj, dwn, dslot, dR0, dR1, dbo, drs0, drs1, dx0, dx1,
                                                dl0, dl1})
        cudaFree(p);
    return same ? 0 : 1;
}

}  // namespace

int main() {
    int ndev = 0;
    if (cudaGetDeviceCount(&ndev) != cudaSuccess || ndev == 0) {
        std::printf("prefill_combine_write_test: no CUDA device (skipped)\n");
        return 77;
    }
    int fails = 0;
    fails += run_case(1, D, false, 1);
    fails += run_case(333, D, true, 2);
    fails += run_case(1000, D, false, 3);
    fails += run_case(517, D + 64, true, 4);   // the padded token stride (hc_pad)
    std::printf("prefill_combine_write_test: %s\n", fails ? "FAILED" : "ok");
    return fails ? 1 : 0;
}
