// ink_topk_gpu_test: the GPU candidate rows (ink_topk.cuh) must be what the host sampler would have found itself, and the tokens
// sampled from them (sample_candidates) must be the tokens of the sampler of origin/main (ink_sample_old.inc), same seed, same rows.
// Also checks the finite flag against finite_logits().  Build by hand:
//   nvcc -O2 -std=c++17 -I engine/include -I engine/src/inkling2 -I engine/src/inkling2/cuda \
//        engine/tests/cuda/ink_topk_gpu_test.cu -o ink_topk_gpu_test     (run: ./ink_topk_gpu_test; compute-sanitizer works on it)
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <numeric>
#include <random>
#include <vector>

#include "ink_serve.h"
#include "../core/ink_sample_old.inc"
#include "ink_topk.cuh"

#define CK(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) { std::printf("CUDA error %s at line %d\n", cudaGetErrorString(e_), __LINE__); return 2; } } while (0)

static std::vector<int> ref_top_k(const float* lg, int V, int K) {
    std::vector<int> idx(V);
    std::iota(idx.begin(), idx.end(), 0);
    std::partial_sort(idx.begin(), idx.begin() + K, idx.end(),
                      [&](int a, int b) { return lg[a] > lg[b] || (lg[a] == lg[b] && a < b); });
    idx.resize(K);
    return idx;
}

// row kinds: 0 random, 1 heavy ties, 2 ties across the whole row, 3 softcapped coarse grid, 4 constant, 5 two-valued, 6 with +-0 and +-inf-free extremes
static void make_row(std::mt19937_64& rng, int kind, int V, std::vector<float>& lg) {
    lg.assign(V, 0.f);
    for (int i = 0; i < V; ++i) {
        switch (kind) {
            case 0: lg[i] = std::uniform_real_distribution<float>(-20.f, 20.f)(rng); break;
            case 1: lg[i] = (float)std::uniform_int_distribution<int>(-8, 8)(rng); break;
            case 2: lg[i] = (float)(i % 3); break;
            case 3: { float x = (float)std::uniform_int_distribution<int>(-400, 400)(rng) * 0.25f; lg[i] = 30.f * std::tanh(x / 30.f); break; }
            case 4: lg[i] = 1.25f; break;
            case 5: lg[i] = (i % 7 == 0) ? 5.f : 0.f; break;
            default: { int r = std::uniform_int_distribution<int>(0, 5)(rng); lg[i] = r == 0 ? 0.0f : r == 1 ? -0.0f : r == 2 ? -29.99f : r == 3 ? 29.99f : (float)r; break; }
        }
    }
}

int main() {
    strata::ink::InkTopkScratch sc;
    if (!sc.ensure(512)) { std::printf("no scratch\n"); return 2; }
    std::mt19937_64 rng(4242);
    long bad = 0, lists = 0;

    auto run_rows = [&](const std::vector<std::vector<float>>& rows, int V, std::vector<uint64_t>& keys, std::vector<int>& fin) -> int {
        const int n = (int)rows.size();
        float* d = nullptr;
        CK(cudaMalloc(&d, (size_t)n * V * sizeof(float)));
        for (int r = 0; r < n; ++r) CK(cudaMemcpy(d + (size_t)r * V, rows[r].data(), (size_t)V * sizeof(float), cudaMemcpyHostToDevice));
        CK(sc.run(d, n, V, 0));
        CK(cudaStreamSynchronize(0));
        keys.assign(sc.h_out, sc.h_out + (size_t)n * strata::ink::kTopkCand);
        fin.assign(sc.h_fin, sc.h_fin + n);
        CK(cudaFree(d));
        return 0;
    };

    // 1. candidate lists and finite flags against the host
    const int Vs[] = {10, 129, 300, 5000, 200058};
    for (int V : Vs) {
        const int kc = std::min(128, V);
        for (int kind = 0; kind < 7; ++kind) {
            std::vector<std::vector<float>> rows(8);
            for (auto& r : rows) make_row(rng, kind, V, r);
            // rows 6, 7: a non-finite logit at a multiple of 67 (flag 0) / elsewhere (flag 1, like finite_logits)
            rows[6][(size_t)(V > 67 ? 67 : 0)] = INFINITY;
            if (V > 70) rows[7][(size_t)70] = NAN; else rows[7][0] = 1.0f;
            std::vector<uint64_t> keys; std::vector<int> fin;
            if (run_rows(rows, V, keys, fin)) return 2;
            for (int r = 0; r < 8; ++r) {
                ++lists;
                if (r >= 6) {
                    if ((fin[r] != 0) != ink2::finite_logits(rows[r].data(), V)) { ++bad; std::printf("FINITE MISMATCH V=%d kind=%d row=%d\n", V, kind, r); }
                    continue;   // the top lists of non-finite rows are not compared
                }
                const std::vector<int> want = ref_top_k(rows[r].data(), V, kc);
                bool ok = (fin[r] != 0);
                for (int j = 0; j < kc && ok; ++j) {
                    const uint64_t k = keys[(size_t)r * strata::ink::kTopkCand + j];
                    if (strata::ink::ink_topk_key_index(k) != want[j]) ok = false;
                    else if (strata::ink::ink_topk_key_value(k) != rows[r][(size_t)want[j]]) ok = false;   // (-0 == +0 compares equal)
                }
                if (!ok) { ++bad; std::printf("LIST MISMATCH V=%d kind=%d row=%d\n", V, kind, r); }
            }
        }
    }
    std::printf("candidate lists: %ld rows (incl. finite flag), %ld mismatches\n", lists, bad);

    // 2. sampled token streams: old sampler on the full rows vs sample_candidates on the GPU candidates, same seed
    const bool fast = std::getenv("INK_TOPK_TEST_FAST") != nullptr;   // sanitizer runs: every 9th stream, 12 steps
    long seen = 0;
    long streams = 0, steps = 0, sbad = 0;
    const float temps[] = {0.0f, 0.3f, 0.7f, 1.0f, 1.5f};
    const int top_ks[] = {1, 5, 40, 64, 128, 500, 0};
    const float top_ps[] = {1.0f, 0.9f, 0.5f};
    const float min_ps[] = {0.0f, 0.05f};
    const int Vq[] = {10, 300, 5000};
    for (float t : temps) for (int tk : top_ks) for (float tp : top_ps) for (float mp : min_ps) for (int V : Vq) {
        if (fast && (seen++ % 9) != 0) continue;
        ink2::SampleOpts so;
        so.temperature = t; so.top_k = tk; so.top_p = tp; so.min_p = mp;
        const int kind = (int)(streams % 7);
        std::mt19937_64 rng_new(777 + streams), rng_old(777 + streams), rows_rng(99 + streams);
        ++streams;
        const int n = fast ? 12 : 300;
        std::vector<std::vector<float>> rows(n);
        for (auto& r : rows) make_row(rows_rng, kind, V, r);
        std::vector<uint64_t> keys; std::vector<int> fin;
        if (run_rows(rows, V, keys, fin)) return 2;
        std::vector<int32_t> hist;
        for (int step = 0; step < n; ++step) {
            std::vector<float> cv(128, 0.f); std::vector<int32_t> ci(128, 0);
            const int kc = std::min(128, V);
            for (int j = 0; j < kc; ++j) {
                const uint64_t k = keys[(size_t)step * strata::ink::kTopkCand + j];
                cv[j] = strata::ink::ink_topk_key_value(k);
                ci[j] = strata::ink::ink_topk_key_index(k);
            }
            std::vector<float> b = rows[step];
            const int32_t tn = ink2::sample_candidates(cv.data(), ci.data(), V, so, rng_new);
            const int32_t to = ink2old::sample_logits(b.data(), V, so, hist, rng_old);
            ++steps;
            if (tn != to) ++sbad;
            hist.push_back(to);
        }
    }
    std::printf("sampled streams from GPU candidates: %ld streams, %ld steps, %ld differ from the old sampler (greedy included)\n", streams, steps, sbad);
    // 3. the vocabulary-sized rows, a few steps each (all kinds, greedy and sampled)
    long vbad = 0, vsteps = 0;
    for (int kind = 0; kind < 7; ++kind) {
        for (float t : {0.0f, 0.7f, 1.0f}) {
            ink2::SampleOpts so; so.temperature = t; so.top_k = 64; so.top_p = 0.95f;
            std::mt19937_64 rng_new(31 + kind), rng_old(31 + kind), rr(7 + kind);
            std::vector<std::vector<float>> rows(12);
            for (auto& r : rows) make_row(rr, kind, 200058, r);
            std::vector<uint64_t> keys; std::vector<int> fin;
            if (run_rows(rows, 200058, keys, fin)) return 2;
            std::vector<int32_t> hist;
            for (int step = 0; step < 12; ++step) {
                std::vector<float> cv(128); std::vector<int32_t> ci(128);
                for (int j = 0; j < 128; ++j) {
                    const uint64_t k = keys[(size_t)step * strata::ink::kTopkCand + j];
                    cv[j] = strata::ink::ink_topk_key_value(k); ci[j] = strata::ink::ink_topk_key_index(k);
                }
                std::vector<float> b = rows[step];
                const int32_t tn = ink2::sample_candidates(cv.data(), ci.data(), 200058, so, rng_new);
                const int32_t to = ink2old::sample_logits(b.data(), 200058, so, hist, rng_old);
                ++vsteps; if (tn != to) ++vbad;
                hist.push_back(to);
            }
        }
    }
    std::printf("vocabulary-sized rows: %ld steps, %ld differ\n", vsteps, vbad);
    bad += sbad + vbad;
    sc.free_all();
    std::printf("ink_topk_gpu_test: %s\n", bad ? "FAILED" : "OK");
    return bad ? 1 : 0;
}
