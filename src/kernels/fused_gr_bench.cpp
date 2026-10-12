// src/kernels/fused_gr_bench.cpp - fused_gr_read_multi: the AMD fast kernels (STRATA_GR_FAST) against the old
// ones, bit for bit on every output (lo, rs, inject, mixed, the in-place R), and each one's time.
//
//     build/fused_gr_bench [iters] [T min] [T max] [cold sets]
// With "cold sets" > 1 it also times the read with the weights rotating over that many copies (a layer's weights are
// cold in the engine) and prints the per-stage split (norm / down / up) from the GPU stamps.
#include "strata/kernels/fused_gr.hpp"
#include "strata/kernels/verify_kernels.hpp"

#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <vector>

namespace K = strata::kernels;

static uint16_t bf16(float x) {
    uint32_t u;
    std::memcpy(&u, &x, 4);
    return (uint16_t) ((u + 0x7fff + ((u >> 16) & 1)) >> 16);
}

int main(int argc, char** argv) {
    setvbuf(stdout, nullptr, _IONBF, 0);
    const int iters = argc > 1 ? std::atoi(argv[1]) : 500;
    const int t_lo = argc > 2 ? std::atoi(argv[2]) : 1, t_hi = argc > 3 ? std::atoi(argv[3]) : 6;
    const int cold_sets = argc > 4 ? std::atoi(argv[4]) : 0;
    const int N = 2560, HC = 4, D = N * HC, LR = 320, TM = K::kFusedGrMaxT;
    std::mt19937 rng(7);
    std::normal_distribution<float> nd(0.f, 1.f);
    auto fill = [&](std::vector<float>& v, float sc) { for (auto& x : v) x = sc * nd(rng); };
    std::vector<float> R((size_t) TM * D), bo((size_t) TM * N), inj((size_t) TM * HC), wn(D);
    fill(R, 1.0f); fill(bo, 0.5f); fill(inj, 1.0f); fill(wn, 0.3f);
    for (auto& x : wn) x += 1.0f;
    std::vector<uint16_t> wd((size_t) LR * D), wu((size_t) D * LR), wi((size_t) HC * D);
    for (auto& x : wd) x = bf16(0.02f * nd(rng));
    for (auto& x : wu) x = bf16(0.05f * nd(rng));
    for (auto& x : wi) x = bf16(0.02f * nd(rng));
    float *dR, *dbo, *dinj, *dwn, *dxn, *dlo, *drs, *dio, *dmix;
    uint16_t *dwd, *dwu, *dwi;
    cudaMalloc((void**) &dR, R.size() * 4);
    cudaMalloc((void**) &dbo, bo.size() * 4);
    cudaMalloc((void**) &dinj, inj.size() * 4);
    cudaMalloc((void**) &dwn, wn.size() * 4);
    cudaMalloc((void**) &dxn, (size_t) TM * D * 4);
    cudaMalloc((void**) &dlo, (size_t) TM * LR * 4);
    cudaMalloc((void**) &drs, (size_t) TM * HC * 4);
    cudaMalloc((void**) &dio, (size_t) TM * HC * 4);
    cudaMalloc((void**) &dmix, (size_t) TM * N * 4);
    cudaMalloc((void**) &dwd, wd.size() * 2);
    cudaMalloc((void**) &dwu, wu.size() * 2);
    cudaMalloc((void**) &dwi, wi.size() * 2);
    cudaMemcpy(dbo, bo.data(), bo.size() * 4, cudaMemcpyHostToDevice);
    cudaMemcpy(dinj, inj.data(), inj.size() * 4, cudaMemcpyHostToDevice);
    cudaMemcpy(dwn, wn.data(), wn.size() * 4, cudaMemcpyHostToDevice);
    cudaMemcpy(dwd, wd.data(), wd.size() * 2, cudaMemcpyHostToDevice);
    cudaMemcpy(dwu, wu.data(), wu.size() * 2, cudaMemcpyHostToDevice);
    cudaMemcpy(dwi, wi.data(), wi.size() * 2, cudaMemcpyHostToDevice);
    cudaStream_t s;
    cudaStreamCreate(&s);
    cudaEvent_t e0, e1;
    cudaEventCreate(&e0);
    cudaEventCreate(&e1);
    int failures = 0;
    for (int T = t_lo; T <= t_hi; ++T)
        for (int apply = 0; apply < 2; ++apply)
            for (int inject = 0; inject < 2; ++inject) {
                K::FusedGrArgs a[TM];
                for (int t = 0; t < T; ++t) {
                    a[t].R = dR + (size_t) t * D; a[t].R_out = dR + (size_t) t * D; a[t].apply = apply;
                    a[t].bo_prev = dbo + (size_t) t * N; a[t].inj_prev = dinj + (size_t) t * HC;
                    a[t].w_norm = dwn; a[t].w_down = dwd; a[t].w_up = dwu; a[t].w_inject = inject ? dwi : nullptr;
                    a[t].lo = dlo + (size_t) t * LR; a[t].rs = drs + (size_t) t * HC;
                    a[t].inject_out = dio + (size_t) t * HC; a[t].mixed = dmix + (size_t) t * N;
                }
                std::vector<float> out[2];
                for (int f = 0; f < 2; ++f) {
                    cudaMemcpy(dR, R.data(), R.size() * 4, cudaMemcpyHostToDevice);
                    cudaMemset(dlo, 0xff, (size_t) TM * LR * 4);
                    cudaMemset(drs, 0xff, (size_t) TM * HC * 4);
                    cudaMemset(dio, 0xff, (size_t) TM * HC * 4);
                    cudaMemset(dmix, 0xff, (size_t) TM * N * 4);
                    K::fused_gr_set_fast(f);
                    K::fused_gr_read_multi(a, T, dxn, s);
                    cudaStreamSynchronize(s);
                    auto& o = out[f];
                    o.resize((size_t) TM * (D + LR + HC + HC + N));
                    float* p = o.data();
                    cudaMemcpy(p, dR, (size_t) TM * D * 4, cudaMemcpyDeviceToHost); p += (size_t) TM * D;
                    cudaMemcpy(p, dlo, (size_t) TM * LR * 4, cudaMemcpyDeviceToHost); p += (size_t) TM * LR;
                    cudaMemcpy(p, drs, (size_t) TM * HC * 4, cudaMemcpyDeviceToHost); p += (size_t) TM * HC;
                    cudaMemcpy(p, dio, (size_t) TM * HC * 4, cudaMemcpyDeviceToHost); p += (size_t) TM * HC;
                    cudaMemcpy(p, dmix, (size_t) TM * N * 4, cudaMemcpyDeviceToHost);
                }
                const bool same = std::memcmp(out[0].data(), out[1].data(), out[0].size() * 4) == 0;
                double us[2] = {1e30, 1e30};
                for (int round = 0; round < 3; ++round)
                    for (int f = 0; f < 2; ++f) {
                        K::fused_gr_set_fast(f);
                        for (int i = 0; i < 20; ++i) K::fused_gr_read_multi(a, T, dxn, s);
                        cudaEventRecord(e0, s);
                        for (int i = 0; i < iters; ++i) K::fused_gr_read_multi(a, T, dxn, s);
                        cudaEventRecord(e1, s);
                        cudaEventSynchronize(e1);
                        float ms = 0;
                        cudaEventElapsedTime(&ms, e0, e1);
                        us[f] = std::fmin(us[f], 1e3 * ms / iters);
                    }
                unsigned long long h = 1469598103934665603ull;   // FNV-1a of the outputs: compare across runs (env switches)
                for (size_t b = 0; b < out[0].size() * 4; ++b) h = (h ^ ((const unsigned char*) out[0].data())[b]) * 1099511628211ull;
                std::printf("T %d apply %d inject %d | old %6.1f us  fast %6.1f us | %s | hash %016llx\n", T, apply, inject,
                            us[0], us[1], same ? "bitwise equal" : "DIFFERS", h);
                if (!same) ++failures;
            }
    if (cold_sets > 1) {
        // the engine reads a different layer's weights every time: rotate over `cold_sets` copies (13 MB each)
        std::vector<uint16_t*> cwd(cold_sets), cwu(cold_sets), cwi(cold_sets);
        std::vector<float*> cwn(cold_sets);
        for (int c = 0; c < cold_sets; ++c) {
            cudaMalloc((void**) &cwd[c], wd.size() * 2); cudaMemcpy(cwd[c], wd.data(), wd.size() * 2, cudaMemcpyHostToDevice);
            cudaMalloc((void**) &cwu[c], wu.size() * 2); cudaMemcpy(cwu[c], wu.data(), wu.size() * 2, cudaMemcpyHostToDevice);
            cudaMalloc((void**) &cwi[c], wi.size() * 2); cudaMemcpy(cwi[c], wi.data(), wi.size() * 2, cudaMemcpyHostToDevice);
            cudaMalloc((void**) &cwn[c], wn.size() * 4); cudaMemcpy(cwn[c], wn.data(), wn.size() * 4, cudaMemcpyHostToDevice);
        }
        unsigned long long* dst;
        cudaMalloc((void**) &dst, (size_t) iters * 4 * 8);
        std::vector<unsigned long long> hst((size_t) iters * 4);
        std::vector<K::FusedGrArgs> as((size_t) cold_sets * 8);
        for (int T = t_lo; T <= t_hi; ++T) for (int f = 0; f < 2; ++f) {
            K::fused_gr_set_fast(f);
            for (int c = 0; c < cold_sets; ++c)
                for (int t = 0; t < T; ++t) {
                    K::FusedGrArgs& a = as[(size_t) c * 8 + t];
                    a = K::FusedGrArgs();
                    a.R = dR + (size_t) t * D; a.R_out = dR + (size_t) t * D; a.apply = 1;
                    a.bo_prev = dbo + (size_t) t * N; a.inj_prev = dinj + (size_t) t * HC;
                    a.w_norm = cwn[c]; a.w_down = cwd[c]; a.w_up = cwu[c]; a.w_inject = cwi[c];
                    a.lo = dlo + (size_t) t * LR; a.rs = drs + (size_t) t * HC;
                    a.inject_out = dio + (size_t) t * HC; a.mixed = dmix + (size_t) t * N;
                }
            for (int rep = 0; rep < 2; ++rep) {
                for (int i = 0; i < iters; ++i) {
                    K::gpu_stamp(dst, i * 4, s);
                    K::fused_gr_read_multi(&as[(size_t) (i % cold_sets) * 8], T, dxn, s, dst, i * 4 + 1);
                    K::gpu_stamp(dst, i * 4 + 3, s);
                }
                cudaStreamSynchronize(s);
            }
            cudaEventRecord(e0, s);
            for (int i = 0; i < iters; ++i) K::fused_gr_read_multi(&as[(size_t) (i % cold_sets) * 8], T, dxn, s);
            cudaEventRecord(e1, s);
            cudaEventSynchronize(e1);
            float ms = 0;
            cudaEventElapsedTime(&ms, e0, e1);
            cudaMemcpy(hst.data(), dst, hst.size() * 8, cudaMemcpyDeviceToHost);
            // stamps: 0 start, 1 after norm, 2 after down, 3 after up (a ~1-2 us stamp kernel between each)
            std::vector<double> n_, d_, u_;
            for (int i = 1; i < iters; ++i) {
                const unsigned long long* p = &hst[(size_t) i * 4];
                n_.push_back((double) (p[1] - p[0])); d_.push_back((double) (p[2] - p[1])); u_.push_back((double) (p[3] - p[2]));
            }
            auto med = [](std::vector<double>& v) { std::sort(v.begin(), v.end()); return v[v.size() / 2] / 1e3; };
            std::printf("COLD(%d sets) fast %d T %d | read %6.1f us (events) | stamps median: norm %5.1f down %5.1f up %5.1f us (incl. a stamp kernel each)\n",
                        cold_sets, f, T, 1e3 * ms / iters, med(n_), med(d_), med(u_));
        }
    }
    std::printf("fused_gr_bench: %d differing\n", failures);
    return failures ? 1 : 0;
}
