// src/kernels/gdn_defer_bench.cpp - fuse-gdn STRATA_FUSE_GDNDEFER: a batch window's GDN recurrence with the last
// window's kept rows committed first (gdn_commit_layers per slot, then the window's 36 gdn_step_norm_groups) against
// the same rows deferred into the window (gdn_pend_copy_layers per slot, then the window's kernels with pend_*).
//
//     build/gdn_defer_bench [reps] [groups] [rows per group] [kept rows]
//
// Qwen3.8-Flash-Next's shapes (h_k 16, h_v 48, 128 x 128 state, conv channels 10240, 36 GDN layers per slot: 113 MB of
// state each).  Parity: one sequence of each from the same states - every layer's state (the recurrence part), y and
// q8_1 image compared bit for bit (exit 1 on a difference).  Time: the median of CUDA events around each sequence.
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

namespace {
bool ck(cudaError_t e, const char* what) {
    if (e == cudaSuccess) return true;
    std::printf("CUDA: %s: %s\n", what, cudaGetErrorString(e));
    std::exit(1);
}
float* dev(size_t n, std::mt19937& rng, float lo, float hi, bool normal) {
    std::vector<float> h(n);
    std::normal_distribution<float> nd(0.f, hi);
    std::uniform_real_distribution<float> ud(lo, hi);
    for (auto& v : h) v = normal ? nd(rng) : ud(rng);
    float* d = nullptr;
    ck(cudaMalloc(&d, n * 4), "malloc");
    ck(cudaMemcpy(d, h.data(), n * 4, cudaMemcpyHostToDevice), "copy");
    return d;
}
}  // namespace

int main(int argc, char** argv) {
    const int reps = argc > 1 ? std::max(5, std::atoi(argv[1])) : 60;
    const int G = argc > 2 ? std::min(8, std::max(1, std::atoi(argv[2]))) : 2;
    const int R = argc > 3 ? std::min(4, std::max(1, std::atoi(argv[3]))) : 2;
    const int P = argc > 4 ? std::min(R, std::max(0, std::atoi(argv[4]))) : 2;
    const int h_k = 16, h_v = 48, S = 128, C = 10240, L = 36, VD = S * h_v, MT = 8, rows = G * R;
    const size_t SF = (size_t) S * h_v * S, st = SF + (size_t) C * 3;   // a layer: recurrence + conv history
    const int PM = MT;
    const size_t PL = (size_t) PM * (C + 2 * h_v);
    std::mt19937 rng(11);
    float* state0 = dev(st * L * G, rng, 0, 0.05f, true);
    float* stA = nullptr; float* stB = nullptr;
    ck(cudaMalloc(&stA, st * L * G * 4), "stA");
    ck(cudaMalloc(&stB, st * L * G * 4), "stB");
    // the last window's rows (per layer MT rows: h_L_, gate_L_, beta_L_) and this window's (one layer's buffers)
    float* hp = dev((size_t) L * MT * C, rng, 0, 0.1f, true);
    float* qp = dev((size_t) L * MT * C, rng, 0, 0.1f, true);   // its pre-conv rows (the conv commit's)
    float* gp = dev((size_t) L * MT * h_v, rng, -0.15f, -0.05f, false);
    float* bp = dev((size_t) L * MT * h_v, rng, 0.2f, 0.8f, false);
    float* h = dev((size_t) rows * C, rng, 0, 0.1f, true);
    float* gate = dev((size_t) rows * h_v, rng, -0.15f, -0.05f, false);
    float* beta = dev((size_t) rows * h_v, rng, 0.2f, 0.8f, false);
    float* z = dev((size_t) rows * VD, rng, 0, 1.0f, true);
    float* gamma = dev(S, rng, 0, 1.0f, true);
    float* pend = nullptr;
    ck(cudaMalloc(&pend, PL * L * G * 4), "pend");
    int32_t* words = nullptr;   // per group: n (the old commit's), 0 (its recurrence under DEFER), P (pending)
    ck(cudaMalloc(&words, (size_t) G * 3 * 4), "words");
    std::vector<int32_t> hw;
    for (int g = 0; g < G; ++g) { hw.push_back(P); hw.push_back(0); hw.push_back(P); }
    ck(cudaMemcpy(words, hw.data(), hw.size() * 4, cudaMemcpyHostToDevice), "words");
    const size_t yn = (size_t) rows * VD, xqb = (size_t) rows * (VD / 32) * 36;
    float *y = nullptr, *ys[2] = {};
    uint8_t* xq = nullptr;
    ck(cudaMalloc(&y, yn * 4), "y");
    ck(cudaMalloc(&xq, xqb), "xq");
    for (auto& p : ys) ck(cudaMalloc(&p, yn * 4 * L), "ys");
    std::vector<uint8_t> xqs[2];
    for (auto& v : xqs) v.resize(xqb * L);
    cudaStream_t s;
    ck(cudaStreamCreate(&s), "stream");
    // one window: [the commit of the last one's kept rows] + the 36 layers' recurrence (keep = record y and q8_1)
    auto window = [&](int v, float* state, bool keep) {
        for (int g = 0; g < G; ++g) {
            float* sg = state + (size_t) g * L * st;
            const size_t f = (size_t) g * R;
            if (v == 0) {
                if (P > 0 && !K::gdn_commit_layers(sg, st, SF, qp + f * C, (int64_t) MT * C, hp + f * C, (int64_t) MT * C,
                                                   gp + f * h_v, bp + f * h_v, (int64_t) MT * h_v, C, h_k, h_v, R, L,
                                                   words + g * 3, s)) {
                    std::printf("gdn_commit_layers declined\n");
                    std::exit(1);
                }
            } else {
                // DEFER: the conv history still commits now (gdn_commit_layers with n_state = 0), the rows go to pend
                if (P > 0 && !K::gdn_commit_layers(sg, st, SF, qp + f * C, (int64_t) MT * C, hp + f * C, (int64_t) MT * C,
                                                   gp + f * h_v, bp + f * h_v, (int64_t) MT * h_v, C, h_k, h_v, R, L,
                                                   words + g * 3, s, words + g * 3 + 1))
                    std::exit(1);
                K::gdn_pend_copy_layers(pend + (size_t) g * L * PL, PL, PM, hp + f * C, (int64_t) MT * C, gp + f * h_v,
                                        bp + f * h_v, (int64_t) MT * h_v, C, h_v, L, words + g * 3 + 2, s);
            }
        }
        for (int l = 0; l < L; ++l) {
            K::GdnGroups gr;
            gr.n = G;
            for (int g = 0; g < G; ++g) {
                gr.state[g] = state + ((size_t) g * L + l) * st;
                gr.first[g] = g * R;
                gr.ntok[g] = R;
                if (v == 1) {
                    const float* pl = pend + ((size_t) g * L + l) * PL;
                    gr.pend_h[g] = pl;
                    gr.pend_g[g] = pl + (size_t) PM * C;
                    gr.pend_b[g] = pl + (size_t) PM * C + (size_t) PM * h_v;
                    gr.pend_n[g] = words + g * 3 + 2;
                }
            }
            K::gdn_step_norm_groups(gr, h, C, gate, beta, z, gamma, 1e-6f, y, h_k, h_v, false, s, xq);
            if (keep) {
                cudaMemcpyAsync(ys[v] + (size_t) l * yn, y, yn * 4, cudaMemcpyDeviceToDevice, s);
                cudaMemcpyAsync(xqs[v].data() + (size_t) l * xqb, xq, xqb, cudaMemcpyDeviceToHost, s);
            }
        }
    };
    // parity: one window each from state0
    for (int v = 0; v < 2; ++v) {
        float* sv = v == 0 ? stA : stB;
        ck(cudaMemcpy(sv, state0, st * L * G * 4, cudaMemcpyDeviceToDevice), "state0");
        window(v, sv, true);
        ck(cudaStreamSynchronize(s), "parity window");
    }
    std::vector<uint32_t> a(st * L * G), b(st * L * G);
    ck(cudaMemcpy(a.data(), stA, a.size() * 4, cudaMemcpyDeviceToHost), "a");
    ck(cudaMemcpy(b.data(), stB, b.size() * 4, cudaMemcpyDeviceToHost), "b");
    long long sdiff = 0, cdiff = 0, smoved = 0;
    std::vector<uint32_t> s0(st * L * G);
    ck(cudaMemcpy(s0.data(), state0, s0.size() * 4, cudaMemcpyDeviceToHost), "s0");
    for (size_t i = 0; i < a.size(); ++i) {
        const bool rec = i % st < SF;
        (rec ? sdiff : cdiff) += a[i] != b[i];
        smoved += rec && a[i] != s0[i];
    }
    std::vector<uint32_t> ya(yn * L), yb(yn * L);
    ck(cudaMemcpy(ya.data(), ys[0], ya.size() * 4, cudaMemcpyDeviceToHost), "ya");
    ck(cudaMemcpy(yb.data(), ys[1], yb.size() * 4, cudaMemcpyDeviceToHost), "yb");
    long long ydiff = 0, nonfinite = 0;
    for (size_t i = 0; i < ya.size(); ++i) {
        ydiff += ya[i] != yb[i];
        float f;
        std::memcpy(&f, &ya[i], 4);
        nonfinite += !std::isfinite(f);
    }
    const bool xsame = std::memcmp(xqs[0].data(), xqs[1].data(), xqs[0].size()) == 0;
    // time: whole windows, each variant on its own states
    cudaEvent_t e0, e1;
    cudaEventCreate(&e0);
    cudaEventCreate(&e1);
    double med[2] = {};
    for (int v = 0; v < 2; ++v) {
        float* sv = v == 0 ? stA : stB;
        window(v, sv, false);
        std::vector<float> ms;
        for (int i = 0; i < reps; ++i) {
            cudaEventRecord(e0, s);
            window(v, sv, false);
            cudaEventRecord(e1, s);
            cudaEventSynchronize(e1);
            float m = 0;
            cudaEventElapsedTime(&m, e0, e1);
            ms.push_back(m);
        }
        std::sort(ms.begin(), ms.end());
        med[v] = ms[ms.size() / 2] * 1e3;
    }
    std::printf("gdn_defer_bench: %d groups x %d rows, %d rows kept, %d layers, %d reps\n", G, R, P, L, reps);
    std::printf("  commit, then the window   %8.1f us\n", med[0]);
    std::printf("  STRATA_FUSE_GDNDEFER      %8.1f us  (%+.1f us)\n", med[1], med[1] - med[0]);
    std::printf("  state: %lld recurrence / %lld conv values differ (%lld moved by the commit); y: %lld of %zu differ, "
                "%lld non-finite; q8_1 image %s\n", sdiff, cdiff, smoved, ydiff, ya.size(), nonfinite,
                xsame ? "the same" : "DIFFERS");
    const bool ok = sdiff == 0 && cdiff == 0 && ydiff == 0 && xsame && nonfinite == 0 && (P == 0 || smoved > 0);
    std::printf("gdn_defer_bench: %s\n", ok ? "ok" : "FAILED");
    return ok ? 0 : 1;
}
