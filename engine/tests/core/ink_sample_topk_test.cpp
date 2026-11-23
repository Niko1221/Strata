// ink_top_k_ids (ink_serve.h) must leave exactly what std::partial_sort over an iota leaves, for rows with many equal logits too
// (#6779: the host sampler runs once per token per slot; the partial_sort over 200058 ids was the cost).  No GPU.
#include <algorithm>
#include <chrono>
#include <cstdio>
#include <numeric>
#include <random>
#include <vector>

#include "ink_serve.h"

static std::vector<int> ref_top_k(const float* lg, int V, int K) {
    std::vector<int> idx(V);
    std::iota(idx.begin(), idx.end(), 0);
    std::partial_sort(idx.begin(), idx.begin() + K, idx.end(),
                      [&](int a, int b) { return lg[a] > lg[b] || (lg[a] == lg[b] && a < b); });
    idx.resize(K);
    return idx;
}

int main() {
    std::mt19937_64 rng(12345);
    int bad = 0, cases = 0;
    const int Vs[] = {1, 2, 7, 64, 129, 1000, 200058};
    const int Ks[] = {1, 2, 5, 64, 128};
    for (int V : Vs) {
        for (int K : Ks) {
            const int Ke = std::min(K, V);
            for (int mode = 0; mode < 3; ++mode) {
                std::vector<float> lg(V);
                for (int i = 0; i < V; ++i) {
                    if (mode == 0) lg[i] = std::uniform_real_distribution<float>(-20.f, 20.f)(rng);
                    else if (mode == 1) lg[i] = (float)(int)std::uniform_int_distribution<int>(-8, 8)(rng);   // heavy ties
                    else lg[i] = (float)(i % 3);                                                              // ties across the whole row
                }
                std::vector<int> got;
                ink2::ink_top_k_ids(lg.data(), V, K, got);
                const std::vector<int> want = ref_top_k(lg.data(), V, Ke);
                ++cases;
                if (got != want) {
                    ++bad;
                    std::printf("MISMATCH V=%d K=%d mode=%d\n", V, K, mode);
                }
            }
        }
    }
    // informative only: the cost per call on a vocabulary-sized row
    {
        const int V = 200058;
        std::vector<float> lg(V);
        for (int i = 0; i < V; ++i) lg[i] = std::uniform_real_distribution<float>(-20.f, 20.f)(rng);
        std::vector<int> out;
        auto t0 = std::chrono::steady_clock::now();
        for (int r = 0; r < 50; ++r) ink2::ink_top_k_ids(lg.data(), V, 64, out);
        auto t1 = std::chrono::steady_clock::now();
        for (int r = 0; r < 50; ++r) { auto w = ref_top_k(lg.data(), V, 64); if (w.empty()) return 2; }
        auto t2 = std::chrono::steady_clock::now();
        std::printf("top-64 of %d: new %.3f ms, partial_sort %.3f ms per call\n", V,
                    std::chrono::duration<double, std::milli>(t1 - t0).count() / 50,
                    std::chrono::duration<double, std::milli>(t2 - t1).count() / 50);
    }
    std::printf("ink_sample_topk_test: %d cases, %d mismatches\n", cases, bad);
    return bad ? 1 : 0;
}
