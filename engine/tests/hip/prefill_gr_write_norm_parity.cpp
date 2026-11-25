// tests/hip/prefill_gr_write_norm_parity.cpp - #6776: gr_write_norm_rs (the fused hyper-connection write-back + the next
// half's norm row scale and BF16 images) must give the same bits as the two kernels it replaces, gr_write then
// gr_norm_rs: R, rs, the BF16 image and its low half, in the BF16 and the FP16 image. Its vector (float4) path changed the
// order of the sum of squares once, rs differed in the last bit and a few answers flipped (7 of 300 right to wrong).
// Shapes cover the vector path (aligned, ldx = 10240 / 10304) and the scalar fallback (ldx odd), with and without the low
// image, and T from 1 to a few thousand tokens so that a rare last-bit difference shows.
//   gr_write_norm_parity [--bench]    exit 0 pass, 1 mismatch, 77 no GPU
#include "strata/prefill/kernels.hpp"

#include <cuda_runtime.h>

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <vector>

namespace {
constexpr int N = 2560, HC = 4, D = N * HC;

template <class T> struct Dev {
    T* p = nullptr;
    size_t n = 0;
    bool alloc(size_t cnt) {
        n = cnt;
        return cudaMalloc((void**) &p, cnt * sizeof(T)) == cudaSuccess;
    }
    void up(const std::vector<T>& h) { cudaMemcpy(p, h.data(), h.size() * sizeof(T), cudaMemcpyHostToDevice); }
    std::vector<T> down() const {
        std::vector<T> h(n);
        cudaMemcpy(h.data(), p, n * sizeof(T), cudaMemcpyDeviceToHost);
        return h;
    }
    ~Dev() { if (p) cudaFree(p); }
};

// returns the number of differing words
template <class T> size_t diff(const std::vector<T>& a, const std::vector<T>& b, const char* what, int64_t T_, int ldx, int lo, int f16) {
    size_t bad = 0, first = (size_t) -1;
    for (size_t i = 0; i < a.size(); ++i)
        if (std::memcmp(&a[i], &b[i], sizeof(T)) != 0) { if (first == (size_t) -1) first = i; ++bad; }
    if (bad)
        std::printf("MISMATCH %s T=%lld ldx=%d lo=%d f16=%d: %zu of %zu words differ, first at %zu\n", what, (long long) T_, ldx, lo, f16,
                    bad, a.size(), first);
    return bad;
}
}  // namespace

int main(int argc, char** argv) {
    const bool bench = argc > 1 && std::strcmp(argv[1], "--bench") == 0;
    int ndev = 0;
    if (cudaGetDeviceCount(&ndev) != cudaSuccess || ndev == 0) { std::printf("SKIP: no GPU\n"); return 77; }
    cudaStream_t s;
    cudaStreamCreate(&s);
    std::mt19937 rng(6776);
    std::normal_distribution<float> nd(0.f, 1.f);
    size_t total_bad = 0;
    struct Cfg { int64_t T; int ldx; bool lo, f16; };
    const Cfg cfgs[] = {{1, D, false, false},    {7, D, true, false},     {64, D, false, true},   {1000, D, true, true},
                        {4097, D, false, false}, {4097, D + 64, true, false}, {257, D + 1, false, false}, {257, D + 1, true, true}};
    for (const Cfg& c : cfgs) {
        const int64_t T = c.T;
        std::vector<float> R0((size_t) T * D), bo((size_t) T * N), inj((size_t) T * HC), w((size_t) HC * N);
        for (auto& v : R0) v = nd(rng) * (1.0f + 3.0f * (rng() % 7 == 0));
        for (auto& v : bo) v = nd(rng);
        for (auto& v : inj) v = nd(rng) * 2.0f;
        for (auto& v : w) v = 0.5f + 0.2f * nd(rng);
        Dev<float> dRa, dRb, dbo, dinj, dw, drsa, drsb;
        const size_t xn_n = (size_t) (T - 1) * c.ldx + D;
        Dev<uint16_t> dxa, dxb, dla, dlb;
        bool ok = dRa.alloc(R0.size()) && dRb.alloc(R0.size()) && dbo.alloc(bo.size()) && dinj.alloc(inj.size()) && dw.alloc(w.size()) &&
                  drsa.alloc((size_t) T * HC) && drsb.alloc((size_t) T * HC) && dxa.alloc(xn_n) && dxb.alloc(xn_n) && dla.alloc(xn_n) &&
                  dlb.alloc(xn_n);
        if (!ok) { std::printf("FAIL: allocation\n"); return 2; }
        dRa.up(R0); dRb.up(R0); dbo.up(bo); dinj.up(inj); dw.up(w);
        cudaMemset(dxa.p, 0, xn_n * 2); cudaMemset(dxb.p, 0, xn_n * 2); cudaMemset(dla.p, 0, xn_n * 2); cudaMemset(dlb.p, 0, xn_n * 2);
        cudaMemset(drsa.p, 0, (size_t) T * HC * 4); cudaMemset(drsb.p, 0, (size_t) T * HC * 4);
        // reference: the two kernels
        strata::prefill::gr_write(dRa.p, dbo.p, dinj.p, HC, T, s);
        strata::prefill::gr_norm_rs(dRa.p, dw.p, 1e-6f, drsa.p, dxa.p, T, s, c.lo ? dla.p : nullptr, c.ldx, c.f16);
        // fused
        strata::prefill::gr_write_norm_rs(dRb.p, dbo.p, dinj.p, HC, dw.p, 1e-6f, drsb.p, dxb.p, T, s, c.lo ? dlb.p : nullptr, c.ldx, c.f16);
        cudaStreamSynchronize(s);
        const cudaError_t e = cudaGetLastError();
        if (e != cudaSuccess) { std::printf("FAIL: %s\n", cudaGetErrorString(e)); return 2; }
        size_t bad = diff(dRa.down(), dRb.down(), "R", T, c.ldx, c.lo, c.f16) + diff(drsa.down(), drsb.down(), "rs", T, c.ldx, c.lo, c.f16) +
                     diff(dxa.down(), dxb.down(), "xn16", T, c.ldx, c.lo, c.f16);
        if (c.lo) bad += diff(dla.down(), dlb.down(), "xn16_lo", T, c.ldx, c.lo, c.f16);
        std::printf("T=%-5lld ldx=%-6d lo=%d f16=%d  %s\n", (long long) T, c.ldx, (int) c.lo, (int) c.f16, bad ? "DIFFERS" : "identical");
        total_bad += bad;
        if (bench && T == 4097 && c.ldx == D) {
            cudaEvent_t e0, e1;
            cudaEventCreate(&e0); cudaEventCreate(&e1);
            float best_f = 1e30f, best_u = 1e30f;
            for (int r = 0; r < 20; ++r) {
                float ms = 0;
                dRb.up(R0);
                cudaEventRecord(e0, s);
                strata::prefill::gr_write_norm_rs(dRb.p, dbo.p, dinj.p, HC, dw.p, 1e-6f, drsb.p, dxb.p, T, s, nullptr, c.ldx, false);
                cudaEventRecord(e1, s); cudaEventSynchronize(e1); cudaEventElapsedTime(&ms, e0, e1);
                best_f = ms < best_f ? ms : best_f;
                dRa.up(R0);
                cudaEventRecord(e0, s);
                strata::prefill::gr_write(dRa.p, dbo.p, dinj.p, HC, T, s);
                strata::prefill::gr_norm_rs(dRa.p, dw.p, 1e-6f, drsa.p, dxa.p, T, s, nullptr, c.ldx, false);
                cudaEventRecord(e1, s); cudaEventSynchronize(e1); cudaEventElapsedTime(&ms, e0, e1);
                best_u = ms < best_u ? ms : best_u;
            }
            std::printf("bench T=%lld: fused %.3f ms, unfused pair %.3f ms (best of 20)\n", (long long) T, best_f, best_u);
        }
    }
    std::printf(total_bad ? "RESULT: DIFFERS (%zu words)\n" : "RESULT: identical\n", total_bad);
    return total_bad ? 1 : 0;
}
