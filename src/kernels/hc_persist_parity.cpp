// src/kernels/hc_persist_parity.cpp - the STRATA_HC_PERSIST=1 hyper-connection read (one persistent kernel) against the
// default multi read (fused_gr_read_multi).
//
// Real geometry (n_embd 2560, hc 4, hc_lr 320), every T = 1..8, the pending write applied and the inject rows on, as the
// verifier calls it. The persistent read sums in another order, so it is bounded, not bitwise: every output (the mixed
// row, the inject outputs, the written R) must be finite and within 1e-4 of the default read relative to its largest
// magnitude, and two runs of the persistent read must agree bit for bit. Where the card does not hold the 80 blocks at
// once (or on CUDA, where the read is not built) the read reports that it did not run and the test passes as a skip.
// `--off`: with STRATA_HC_PERSIST unset the persistent read must never run (the default path is untouched).
// `--fault`: STRATA_HC_PERSIST_FAULT=2 makes the second run-time read give up at its first barrier: that read's mixed
// rows must all be non-finite (the window's logits then are, and the engine ends the request, #879), the poll must count
// it, and the third read must be correct again (the kernel cleared its barrier: one abort does not poison the path).
// GPU, synthetic, no model.
#include "strata/kernels/fused_gr.hpp"

#include <cuda_runtime.h>

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <vector>

namespace k = strata::kernels;

namespace {
constexpr int N = 2560, HC = 4, D = N * HC, LR = 320, TM = 8;
void ck(cudaError_t e, const char* w) {
    if (e != cudaSuccess) {
        std::fprintf(stderr, "%s: %s\n", w, cudaGetErrorString(e));
        std::exit(2);
    }
}
uint16_t bf16(float x) {
    uint32_t u;
    std::memcpy(&u, &x, 4);
    return (uint16_t) ((u + 0x7fff + ((u >> 16) & 1)) >> 16);
}
template <typename T>
T* up(const std::vector<T>& h) {
    void* p = nullptr;
    ck(cudaMalloc(&p, h.size() * sizeof(T)), "malloc");
    ck(cudaMemcpy(p, h.data(), h.size() * sizeof(T), cudaMemcpyHostToDevice), "h2d");
    return (T*) p;
}
template <typename T>
T* dev(size_t n) {
    void* p = nullptr;
    ck(cudaMalloc(&p, n * sizeof(T)), "malloc");
    return (T*) p;
}
}  // namespace

int main(int argc, char** argv) {
    const bool off = argc > 1 && std::strcmp(argv[1], "--off") == 0;
    const bool fault = argc > 1 && std::strcmp(argv[1], "--fault") == 0;
    if (off) unsetenv("STRATA_HC_PERSIST");
    else setenv("STRATA_HC_PERSIST", "1", 1);
    if (fault) setenv("STRATA_HC_PERSIST_FAULT", "2", 1);
    k::fused_gr_check();
    k::fused_gr_persist_init();

    std::mt19937 rng(7);
    std::normal_distribution<float> nd(0.f, 1.f);
    std::vector<float> R((size_t) TM * D), bo((size_t) TM * N), inj((size_t) TM * HC), wn(D);
    for (auto& x : R) x = nd(rng);
    for (auto& x : bo) x = 0.5f * nd(rng);
    for (auto& x : inj) x = nd(rng);
    for (auto& x : wn) x = 1.f + 0.3f * nd(rng);
    std::vector<uint16_t> wd((size_t) LR * D), wu((size_t) D * LR), wi((size_t) HC * D);
    for (auto& x : wd) x = bf16(0.02f * nd(rng));
    for (auto& x : wu) x = bf16(0.05f * nd(rng));
    for (auto& x : wi) x = bf16(0.02f * nd(rng));
    float* dR = dev<float>(R.size());
    float* dbo = up(bo);
    float* dinj = up(inj);
    float* dwn = up(wn);
    uint16_t* dwd = up(wd);
    uint16_t* dwu = up(wu);
    uint16_t* dwi = up(wi);
    float* dxn = dev<float>((size_t) TM * D);
    float* dlo = dev<float>((size_t) TM * LR);
    float* drs = dev<float>((size_t) TM * HC);
    float* dio = dev<float>((size_t) TM * HC);
    float* dmix = dev<float>((size_t) TM * N);

    if (fault) {   // read 1 correct, read 2 gives up (non-finite, counted), read 3 correct again
        constexpr int T = 4;
        k::FusedGrArgs a[TM];
        for (int t = 0; t < T; ++t) {
            a[t] = k::FusedGrArgs{};
            a[t].R = dR + (size_t) t * D; a[t].R_out = dR + (size_t) t * D; a[t].apply = true;
            a[t].bo_prev = dbo + (size_t) t * N; a[t].inj_prev = dinj + (size_t) t * HC;
            a[t].w_norm = dwn; a[t].w_down = dwd; a[t].w_up = dwu; a[t].w_inject = dwi; a[t].eps = 1e-6f;
            a[t].lo = dlo + (size_t) t * LR; a[t].rs = drs + (size_t) t * HC;
            a[t].inject_out = dio + (size_t) t * HC; a[t].mixed = dmix + (size_t) t * N;
        }
        auto read = [&](bool persist, std::vector<float>& mix) -> bool {
            ck(cudaMemcpy(dR, R.data(), R.size() * 4, cudaMemcpyHostToDevice), "R");
            ck(cudaMemset(dmix, 0, (size_t) TM * N * 4), "mix");   // zeros, not 0xff (that is a NaN already)
            ck(cudaDeviceSynchronize(), "sync");
            bool ran_p = false;
            if (persist) ran_p = k::fused_gr_read_persist(a, T, nullptr);
            else k::fused_gr_read_multi(a, T, dxn, nullptr);
            ck(cudaDeviceSynchronize(), "read");
            mix.resize((size_t) T * N);
            ck(cudaMemcpy(mix.data(), dmix, (size_t) T * N * 4, cudaMemcpyDeviceToHost), "d2h mix");
            return persist ? ran_p : true;
        };
        std::vector<float> ref, r1, r2, r3;
        read(false, ref);
        if (!read(true, r1)) { std::printf("SKIP: the persistent read is not used on this card\n"); return 0; }
        read(true, r2);
        const unsigned aborts = k::fused_gr_persist_poll();
        read(true, r3);
        auto rel = [&](const std::vector<float>& x) {
            double mref = 0.0, md = 0.0;
            for (size_t i = 0; i < x.size(); ++i) {
                if (!std::isfinite(x[i])) return 1e30;
                mref = std::fmax(mref, std::fabs((double) ref[i]));
                md = std::fmax(md, std::fabs((double) x[i] - ref[i]));
            }
            return md / (mref > 0 ? mref : 1.0);
        };
        size_t nonfinite = 0;
        for (float v : r2) nonfinite += !std::isfinite(v);
        const double e1 = rel(r1), e3 = rel(r3);
        const bool ok = e1 < 1e-4 && nonfinite == r2.size() && aborts == 1 && e3 < 1e-4;
        std::printf("read 1: max relative difference %.3g; read 2 (forced abort): %zu of %zu mixed values non-finite, "
                    "%u abort(s) counted; read 3: max relative difference %.3g  %s\n", e1, nonfinite, r2.size(), aborts,
                    e3, ok ? "PASS" : "FAIL");
        return ok ? 0 : 1;
    }
    int bad = 0, ran = 0;
    for (int T = 1; T <= TM; ++T) {
        k::FusedGrArgs a[TM];
        for (int t = 0; t < T; ++t) {
            a[t] = k::FusedGrArgs{};
            a[t].R = dR + (size_t) t * D;
            a[t].R_out = dR + (size_t) t * D;
            a[t].apply = true;
            a[t].bo_prev = dbo + (size_t) t * N;
            a[t].inj_prev = dinj + (size_t) t * HC;
            a[t].w_norm = dwn;
            a[t].w_down = dwd;
            a[t].w_up = dwu;
            a[t].w_inject = dwi;
            a[t].eps = 1e-6f;
            a[t].lo = dlo + (size_t) t * LR;
            a[t].rs = drs + (size_t) t * HC;
            a[t].inject_out = dio + (size_t) t * HC;
            a[t].mixed = dmix + (size_t) t * N;
        }
        // 0: the default read, 1 and 2: the persistent read twice; each output is mixed | inject | R
        std::vector<float> out[3];
        bool used = true;
        for (int w = 0; w < 3; ++w) {
            ck(cudaMemcpy(dR, R.data(), R.size() * 4, cudaMemcpyHostToDevice), "R");
            ck(cudaMemset(dmix, 0xff, (size_t) TM * N * 4), "mix");
            ck(cudaMemset(dio, 0xff, (size_t) TM * HC * 4), "io");
            ck(cudaDeviceSynchronize(), "sync");
            if (w == 0) k::fused_gr_read_multi(a, T, dxn, nullptr);
            else used = k::fused_gr_read_persist(a, T, nullptr) && used;
            ck(cudaDeviceSynchronize(), "read");
            out[w].resize((size_t) T * (N + HC + D));
            ck(cudaMemcpy(out[w].data(), dmix, (size_t) T * N * 4, cudaMemcpyDeviceToHost), "d2h mix");
            ck(cudaMemcpy(out[w].data() + (size_t) T * N, dio, (size_t) T * HC * 4, cudaMemcpyDeviceToHost), "d2h io");
            ck(cudaMemcpy(out[w].data() + (size_t) T * (N + HC), dR, (size_t) T * D * 4, cudaMemcpyDeviceToHost), "d2h R");
            if (w == 1 && !used) break;
        }
        if (off) {
            if (used) {
                std::printf("T=%d: the persistent read ran with STRATA_HC_PERSIST unset  FAIL\n", T);
                ++bad;
            }
            continue;
        }
        if (!used) {
            std::printf("T=%d: the persistent read did not run on this card (skip)\n", T);
            continue;
        }
        ++ran;
        double mref = 0.0, md = 0.0;
        bool finite = true;
        for (size_t i = 0; i < out[0].size(); ++i) {
            finite = finite && std::isfinite(out[1][i]);
            mref = std::fmax(mref, std::fabs((double) out[0][i]));
            md = std::fmax(md, std::fabs((double) out[1][i] - out[0][i]));
        }
        const double rel = md / (mref > 0 ? mref : 1.0);
        const bool same = std::memcmp(out[1].data(), out[2].data(), out[1].size() * 4) == 0;
        const bool ok = finite && rel < 1e-4 && same;
        std::printf("T=%d: max relative difference %.3g against the default read, %s, two runs %s  %s\n", T, rel,
                    finite ? "finite" : "NON-FINITE", same ? "bitwise equal" : "DIFFER", ok ? "ok" : "FAIL");
        if (!ok) ++bad;
    }
    if (off) std::printf("%s: STRATA_HC_PERSIST unset, the persistent read never ran\n", bad ? "FAIL" : "PASS");
    else if (ran == 0) std::printf("SKIP: the persistent read is not used on this card\n");
    else std::printf("%s: %d token counts checked\n", bad ? "FAIL" : "PASS", ran);
    return bad ? 1 : 0;
}
