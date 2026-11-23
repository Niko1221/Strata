// ink_top_k_ids (ink_serve.h) must leave exactly what std::partial_sort over an iota leaves, and the sampled token streams
// (sample_logits / verify_draft_logits, same seed) must be the same as with the sampler of origin/main (ink_sample_old.inc).
// #6779: the host sampler runs once per token per slot; the partial_sort over 200058 ids was the cost.  No GPU.
//
// Tie order: the old code sorted with "larger logit first, then the smaller id first" - a strict total order over distinct ids,
// so the partial_sort result is unique whatever its stability, and the new one-pass window uses the same order (a later id with
// an equal logit goes after the kept ones).  The candidate list, hence every p[i], keep and idx[i] the sampler uses, is identical.
// (A NaN logit has no place in any order; the engine checks logits for non-finite values before sampling.)
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <numeric>
#include <random>
#include <vector>

#include "ink_serve.h"
#include "ink_sample_old.inc"

static std::vector<int> ref_top_k(const float* lg, int V, int K) {
    std::vector<int> idx(V);
    std::iota(idx.begin(), idx.end(), 0);
    std::partial_sort(idx.begin(), idx.begin() + K, idx.end(),
                      [&](int a, int b) { return lg[a] > lg[b] || (lg[a] == lg[b] && a < b); });
    idx.resize(K);
    return idx;
}

// row kinds: 0 random, 1 heavy ties (17 distinct values), 2 ties across the whole row, 3 softcapped (30 tanh(x/30)) of a coarse grid,
// 4 constant, 5 two-valued with a long tie at the top
static void make_row(std::mt19937_64& rng, int kind, int V, std::vector<float>& lg) {
    lg.assign(V, 0.f);
    for (int i = 0; i < V; ++i) {
        switch (kind) {
            case 0: lg[i] = std::uniform_real_distribution<float>(-20.f, 20.f)(rng); break;
            case 1: lg[i] = (float)std::uniform_int_distribution<int>(-8, 8)(rng); break;
            case 2: lg[i] = (float)(i % 3); break;
            case 3: { float x = (float)std::uniform_int_distribution<int>(-400, 400)(rng) * 0.25f; lg[i] = 30.f * std::tanh(x / 30.f); break; }
            case 4: lg[i] = 1.25f; break;
            default: lg[i] = (i % 7 == 0) ? 5.f : 0.f; break;
        }
    }
}

int main() {
    std::mt19937_64 rng(12345);
    int bad = 0, cases = 0;
    const int Vs[] = {1, 2, 7, 64, 129, 1000, 200058};
    const int Ks[] = {1, 2, 5, 64, 128};
    for (int V : Vs) {
        for (int K : Ks) {
            const int Ke = std::min(K, V);
            for (int kind = 0; kind < 6; ++kind) {
                std::vector<float> lg;
                make_row(rng, kind, V, lg);
                std::vector<int> got;
                ink2::ink_top_k_ids(lg.data(), V, K, got);
                const std::vector<int> want = ref_top_k(lg.data(), V, Ke);
                ++cases;
                if (got != want) {
                    ++bad;
                    std::printf("TOPK MISMATCH V=%d K=%d kind=%d\n", V, K, kind);
                }
            }
        }
    }
    std::printf("candidate lists: %d cases, %d mismatches\n", cases, bad);

    // sampled token streams: same seed, same rows, old sampler vs new sampler
    long streams = 0, steps = 0, sbad = 0, vbad = 0, vcalls = 0;
    const float temps[] = {0.0f, 0.3f, 0.7f, 1.0f, 1.5f};
    const int top_ks[] = {1, 5, 40, 64, 128, 500, 0};
    const float top_ps[] = {1.0f, 0.9f, 0.5f};
    const float min_ps[] = {0.0f, 0.05f};
    const int Vq[] = {10, 300, 5000};
    for (float t : temps) for (int tk : top_ks) for (float tp : top_ps) for (float mp : min_ps) for (int V : Vq) {
        for (int pen = 0; pen < 2; ++pen) {
            ink2::SampleOpts so;
            so.temperature = t; so.top_k = tk; so.top_p = tp; so.min_p = mp;
            if (pen) { so.rep = 1.1f; so.freq = 0.05f; so.pres = 0.02f; so.last_n = 32; }
            const int kind = (int)((streams) % 6);
            std::mt19937_64 rng_new(777 + streams), rng_old(777 + streams), rows(99 + streams), drafts(5 + streams);
            std::vector<int32_t> hist_new, hist_old;
            ++streams;
            for (int step = 0; step < 300; ++step) {
                std::vector<float> row;
                make_row(rows, kind, V, row);
                std::vector<float> a = row, b = row;
                const int32_t tn = ink2::sample_logits(a.data(), V, so, hist_new, rng_new);
                const int32_t to = ink2old::sample_logits(b.data(), V, so, hist_old, rng_old);
                ++steps;
                if (tn != to) ++sbad;
                hist_new.push_back(tn); hist_old.push_back(to);
                if (step % 3 == 0) {   // speculative verification with a draft token (about half the time one of the best ids)
                    std::vector<float> c = row, d = row;
                    const int32_t draft = (drafts() & 1) ? (int32_t)(drafts() % (uint64_t)V) : (int32_t)(std::max_element(row.begin(), row.end()) - row.begin());
                    bool acc_n = false, acc_o = false;
                    const int32_t vn = ink2::verify_draft_logits(c.data(), V, so, hist_new, rng_new, draft, acc_n);
                    const int32_t vo = ink2old::verify_draft_logits(d.data(), V, so, hist_old, rng_old, draft, acc_o);
                    ++vcalls;
                    if (vn != vo || acc_n != acc_o) ++vbad;
                }
            }
        }
    }
    std::printf("sampled streams: %ld streams, %ld sample_logits steps (%ld differ), %ld verify_draft_logits calls (%ld differ); greedy (temperature 0) included\n",
                streams, steps, sbad, vcalls, vbad);
    bad += (int)(sbad + vbad);

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
    std::printf("ink_sample_topk_test: %s\n", bad ? "FAILED" : "OK");
    return bad ? 1 : 0;
}
