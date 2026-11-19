// The router on a non-finite gate input: route() and route_publish() must give valid, distinct picks (no read past the
// expert tables) whatever the logits hold, and the same picks as the host's top-k when they are finite.  Also times both
// (DS4_ROUTE_TIMING=1) so a change to the kernels can be compared with an earlier build of this file.
#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#include "../../src/dsv41/ds4_kernels.cuh"

using namespace strata::ds4;

#define CK(x)                                                                                    \
    do {                                                                                         \
        cudaError_t e_ = (x);                                                                    \
        if (e_ != cudaSuccess) {                                                                 \
            std::fprintf(stderr, "%s:%d %s\n", __FILE__, __LINE__, cudaGetErrorString(e_));      \
            std::exit(2);                                                                        \
        }                                                                                        \
    } while (0)

static int fails = 0;
static void check(bool ok, const char* what) {
    std::printf("%s %s\n", ok ? "ok  " : "FAIL", what);
    if (!ok) ++fails;
}

// n routed experts: 384 (every set so far) and 192 (the ISTA-DAS Coder: half of each layer's experts removed).  The entries past n hold a score no real
// expert has (30, and a bias of 5), so a router that read one would pick it and the checks below would fail.
static int run(int n) {
    std::printf("-- %d experts\n", n);
    const int k = 6, T = 16;
    std::vector<float> bias(1024, 5.0f), logits((size_t) T * 1024, 30.0f);
    for (int i = 0; i < n; ++i) bias[(size_t) i] = 0.01f * (float) ((i * 37) % 17);
    std::vector<unsigned long long> vram(1024, 0ull);
    float *d_logits, *d_bias, *d_w;
    int* d_idx;
    unsigned long long *d_vram, *d_plan;
    CK(cudaMalloc(&d_logits, logits.size() * 4));
    CK(cudaMalloc(&d_bias, 1024 * 4));
    CK(cudaMalloc(&d_w, (size_t) T * kPicks * 4));
    CK(cudaMalloc(&d_idx, (size_t) T * kPicks * 4));
    CK(cudaMalloc(&d_vram, 1024 * 8));
    CK(cudaMalloc(&d_plan, (size_t) T * kPicks * 8));
    for (int i = 0; i < n; i += 2) vram[(size_t) i] = 0x1000ull + (unsigned long long) i;
    CK(cudaMemcpy(d_bias, bias.data(), 1024 * 4, cudaMemcpyHostToDevice));
    CK(cudaMemcpy(d_vram, vram.data(), 1024 * 8, cudaMemcpyHostToDevice));

    const float nan = std::nanf(""), inf = INFINITY;
    // DS4_ROUTE_FINITE_ONLY=1: skip the non-finite cases (an old build faults in them, and the context with it) to get its timing
    const int modes = std::getenv("DS4_ROUTE_FINITE_ONLY") != nullptr ? 1 : 6;
    for (int mode = 0; mode < modes; ++mode) {
        const char* names[] = {"finite", "all NaN", "+inf everywhere", "-inf everywhere", "half NaN", "fewer than k finite"};
        unsigned s = 12345u;
        for (int t = 0; t < T; ++t)
            for (int i = 0; i < 1024; ++i) {
                s = s * 1664525u + 1013904223u;
                float v = ((float) (s >> 8) / (float) (1 << 24) - 0.5f) * 8.0f;
                if (i >= n) v = 30.0f;   // (the poison; the NaN / inf modes below overwrite every entry, as before)
                if (mode == 1) v = nan;
                if (mode == 2) v = inf;
                if (mode == 3) v = -inf;
                if (mode == 4 && (i & 1)) v = nan;
                if (mode == 5 && i >= 3) v = nan;
                logits[(size_t) t * 1024 + (size_t) i] = v;
            }
        CK(cudaMemcpy(d_logits, logits.data(), logits.size() * 4, cudaMemcpyHostToDevice));
        for (int which = 0; which < 2; ++which) {
            CK(cudaMemset(d_idx, 0xff, (size_t) T * kPicks * 4));
            if (which == 0)
                route(d_logits, d_bias, n, k, T, 1.5f, d_idx, d_w, 0);
            else
                route_publish(d_logits, d_bias, n, k, T, 1.5f, d_vram, nullptr, 0, d_idx, d_w, d_plan, nullptr, 0, 0);
            const cudaError_t er = cudaDeviceSynchronize();
            char what[160];
            std::snprintf(what, sizeof what, "%s: %s runs without a device error", names[mode], which ? "route_publish" : "route");
            check(er == cudaSuccess, what);
            if (er != cudaSuccess) {
                std::printf("  %s\n", cudaGetErrorString(er));
                return 1;   // (the context is gone)
            }
            std::vector<int> idx((size_t) T * kPicks);
            CK(cudaMemcpy(idx.data(), d_idx, idx.size() * 4, cudaMemcpyDeviceToHost));
            bool valid = true, distinct = true, same = true;
            for (int t = 0; t < T; ++t) {
                for (int r = 0; r < k; ++r) {
                    const int e = idx[(size_t) t * kPicks + r];
                    valid = valid && e >= 0 && e < n;
                    for (int q = 0; q < r; ++q) distinct = distinct && idx[(size_t) t * kPicks + q] != e;
                }
                if (mode == 0) {   // the host's top-k of the same scores
                    std::vector<std::pair<float, int>> sc;
                    for (int i = 0; i < n; ++i) {
                        const float l = logits[(size_t) t * 1024 + (size_t) i];
                        const float sp = l > 20.0f ? l : std::log1p(std::exp(l));
                        sc.push_back({std::sqrt(sp) + bias[(size_t) i], i});
                    }
                    std::stable_sort(sc.begin(), sc.end(), [](const auto& a, const auto& b) { return a.first > b.first; });
                    for (int r = 0; r < k; ++r) same = same && sc[(size_t) r].second == idx[(size_t) t * kPicks + r];
                }
            }
            if (!valid || !distinct) {
                std::printf("  token 0 picks:");
                for (int r = 0; r < kPicks; ++r) std::printf(" %d", idx[(size_t) r]);
                std::printf("\n");
            }
            std::snprintf(what, sizeof what, "%s: %s picks are all in [0, %d)", names[mode], which ? "route_publish" : "route", n);
            check(valid, what);
            // (with fewer than k finite scores the picked entries' -FLT_MAX ties with the start value, so the router re-takes the
            //  lowest of them: a repeat, harmless to the tables - and the weights are NaN there anyway - so only checked when
            //  at least k scores are usable)
            if (mode != 1 && mode != 5) {
                std::snprintf(what, sizeof what, "%s: %s picks are distinct", names[mode], which ? "route_publish" : "route");
                check(distinct, what);
            }
            if (mode == 0) {
                std::snprintf(what, sizeof what, "finite: %s picks equal the host's top-%d", which ? "route_publish" : "route", k);
                check(same, what);
            }
            {   // the flag the engine polls: raised when a pick had to be made up, clear (and cleared by the call) otherwise
                const bool flagged = router_nonfinite_take(0);
                const bool again = router_nonfinite_take(0);
                std::snprintf(what, sizeof what, "%s: %s %s the router's non-finite flag, and the call clears it", names[mode],
                              which ? "route_publish" : "route", mode == 1 ? "raises" : "leaves clear");   // (fewer than k finite: the picked entries' -FLT_MAX re-wins, no stand-in is needed)
                check(flagged == (mode == 1) && !again, what);
            }
            if (which == 1) {
                std::vector<unsigned long long> pl((size_t) T * kPicks);
                CK(cudaMemcpy(pl.data(), d_plan, pl.size() * 8, cudaMemcpyDeviceToHost));
                bool plan_ok = true;
                for (int t = 0; t < T; ++t)
                    for (int r = 0; r < k; ++r) plan_ok = plan_ok && pl[(size_t) t * kPicks + r] == vram[(size_t) idx[(size_t) t * kPicks + r]];
                std::snprintf(what, sizeof what, "%s: route_publish's plan pointers are the table's entries of the picks", names[mode]);
                check(plan_ok, what);
            }
        }
    }

    if (std::getenv("DS4_ROUTE_TIMING") != nullptr) {
        // finite logits again; T = 1 (a decode token) and T = 16 (a verify window), per launch with a kernel launch each
        unsigned s = 777u;
        for (size_t i = 0; i < logits.size(); ++i) {
            s = s * 1664525u + 1013904223u;
            logits[i] = ((float) (s >> 8) / (float) (1 << 24) - 0.5f) * 8.0f;
        }
        CK(cudaMemcpy(d_logits, logits.data(), logits.size() * 4, cudaMemcpyHostToDevice));
        cudaEvent_t e0, e1;
        CK(cudaEventCreate(&e0));
        CK(cudaEventCreate(&e1));
        for (int which = 0; which < 2; ++which)
            for (int tt : {1, 16}) {
                const int reps = 20000;
                double best = 1e30;
                for (int round = 0; round < 7; ++round) {
                    for (int i = 0; i < 500; ++i)
                        which ? route_publish(d_logits, d_bias, n, k, tt, 1.5f, d_vram, nullptr, 0, d_idx, d_w, d_plan, nullptr, 0, 0)
                              : route(d_logits, d_bias, n, k, tt, 1.5f, d_idx, d_w, 0);
                    CK(cudaDeviceSynchronize());
                    CK(cudaEventRecord(e0));
                    for (int i = 0; i < reps; ++i)
                        which ? route_publish(d_logits, d_bias, n, k, tt, 1.5f, d_vram, nullptr, 0, d_idx, d_w, d_plan, nullptr, 0, 0)
                              : route(d_logits, d_bias, n, k, tt, 1.5f, d_idx, d_w, 0);
                    CK(cudaEventRecord(e1));
                    CK(cudaEventSynchronize(e1));
                    float ms = 0;
                    CK(cudaEventElapsedTime(&ms, e0, e1));
                    best = std::min(best, (double) ms * 1000.0 / reps);
                }
                std::printf("TIMING %s T=%d best of 7 rounds: %.3f us a launch\n", which ? "route_publish" : "route", tt, best);
            }
    }
    return 0;
}

int main() {
    for (int n : {384, 192})
        if (run(n) != 0) return 1;      // (a device error: the context is gone)
    std::printf("%s\n", fails ? "ds4_route_guard_test: FAILED" : "ds4_route_guard_test: all ok");
    return fails ? 1 : 0;
}
