// src/kernels/sampler_parity.cpp - P2.S2's test for the sampler chain.
//
// THE CHECK THAT MATTERS IS THAT THE ORDER IS OBSERVABLE.  A sampler in the wrong order still returns a valid
// token, so "it produced a token" proves nothing; and because temperature is MONOTONIC it does not change which
// tokens `top_k` keeps, so a top-k-only fixture cannot see the order either.  What it changes is `top_p`'s CUT:
// at T < 1 the distribution sharpens, the cumulative mass reaches p sooner, and fewer tokens survive.
//
// So this builds a fixture where that happens, computes the greedy pick under BOTH orders, and requires them to
// DIFFER - then requires the kernel to agree with the specified one.  Without the first half, the test would
// pass against either order.
#include "strata/kernels/sampler.hpp"

#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <string>
#include <vector>

namespace {

void check(cudaError_t e, const char* what) {
    if (e != cudaSuccess) {
        std::fprintf(stderr, "%s: %s\n", what, cudaGetErrorString(e));
        std::exit(1);
    }
}

// The host reference for the specified order: top_k -> top_p -> temperature -> argmax.
// `temp_first` swaps the first and last stages, which is the intuitive-but-wrong order.
int reference_pick(const std::vector<float>& l, const strata::kernels::SamplerParams& p, bool temp_first) {
    const int nv = (int) l.size();
    const float inv_t = p.temperature > 0.0f ? 1.0f / p.temperature : 0.0f;
    auto val = [&](int v) { return temp_first ? l[(size_t) v] * inv_t : l[(size_t) v]; };

    std::vector<int> ids;
    const int k = p.top_k > 0 ? p.top_k : nv;
    std::vector<char> taken((size_t) nv, 0);
    for (int i = 0; i < k; ++i) {
        int best = -1;
        float bv = 0;
        for (int v = 0; v < nv; ++v) {
            if (taken[(size_t) v]) continue;
            if (best < 0 || val(v) > bv) { best = v; bv = val(v); }
        }
        taken[(size_t) best] = 1;
        ids.push_back(best);
    }
    if (p.top_p < 1.0f) {
        float mx = val(ids[0]);
        for (int v : ids) mx = std::fmax(mx, val(v));
        double sum = 0;
        for (int v : ids) sum += std::exp((double) val(v) - (double) mx);
        double cum = 0;
        int cut = (int) ids.size();
        for (size_t i = 0; i < ids.size(); ++i) {
            cum += std::exp((double) val(ids[i]) - (double) mx) / sum;
            if (cum >= (double) p.top_p) { cut = (int) i + 1; break; }
        }
        if (cut < p.min_keep) cut = p.min_keep < (int) ids.size() ? p.min_keep : (int) ids.size();
        ids.resize((size_t) cut);
    }
    return ids[0];      // greedy: the largest SURVIVING logit, and `ids` is in descending order
}

// The number of survivors AFTER top_p, in the given order - the quantity the order actually changes.
int reference_cut(const std::vector<float>& l, const strata::kernels::SamplerParams& p, bool temp_first) {
    const int nv = (int) l.size();
    const float inv_t = p.temperature > 0.0f ? 1.0f / p.temperature : 0.0f;
    auto val = [&](int v) { return temp_first ? l[(size_t) v] * inv_t : l[(size_t) v]; };
    const int k = p.top_k > 0 ? p.top_k : nv;
    std::vector<int> ids;
    std::vector<char> taken((size_t) nv, 0);
    for (int i = 0; i < k; ++i) {
        int best = -1; float bv = 0;
        for (int v = 0; v < nv; ++v) {
            if (taken[(size_t) v]) continue;
            if (best < 0 || val(v) > bv) { best = v; bv = val(v); }
        }
        taken[(size_t) best] = 1; ids.push_back(best);
    }
    if (p.top_p < 1.0f) {
        float mx = val(ids[0]);
        for (int v : ids) mx = std::fmax(mx, val(v));
        double sum = 0;
        for (int v : ids) sum += std::exp((double) val(v) - (double) mx);
        double cum = 0;
        int cut = (int) ids.size();
        for (size_t i = 0; i < ids.size(); ++i) {
            cum += std::exp((double) val(ids[i]) - (double) mx) / sum;
            if (cum >= (double) p.top_p) { cut = (int) i + 1; break; }
        }
        if (cut < p.min_keep) cut = p.min_keep < (int) ids.size() ? p.min_keep : (int) ids.size();
        return cut;
    }
    return (int) ids.size();
}

int run(const char* name, const std::vector<float>& logits, int n_tokens, const strata::kernels::SamplerParams& p,
        const std::vector<int>& want, const std::vector<int>& hist = {}, int hist_len = 0) {
    float* d_l = nullptr;
    int* d_o = nullptr;
    check(cudaMalloc(&d_l, logits.size() * sizeof(float)), "malloc logits");
    check(cudaMalloc(&d_o, (size_t) n_tokens * sizeof(int)), "malloc out");
    check(cudaMemcpy(d_l, logits.data(), logits.size() * sizeof(float), cudaMemcpyHostToDevice), "copy");
    int* d_h = nullptr;
    if (hist_len > 0) {
        check(cudaMalloc(&d_h, hist.size() * sizeof(int)), "malloc hist");
        check(cudaMemcpy(d_h, hist.data(), hist.size() * sizeof(int), cudaMemcpyHostToDevice), "copy hist");
    }
    strata::kernels::sample_tokens(d_l, n_tokens, (int) (logits.size() / n_tokens), d_h, hist_len, p, d_o,
                                   nullptr);
    std::vector<int> got((size_t) n_tokens);
    check(cudaMemcpy(got.data(), d_o, got.size() * sizeof(int), cudaMemcpyDeviceToHost), "back");
    int bad = 0;
    for (int t = 0; t < n_tokens; ++t) if (got[(size_t) t] != want[(size_t) t]) ++bad;
    std::printf("  %-34s %s (%d of %d differ)", name, bad ? "*** WRONG ***" : "matches", bad, n_tokens);
    if (bad) std::printf("   first: want %d got %d", want[0], got[0]);
    std::printf("\n");
    cudaFree(d_l);
    cudaFree(d_o);
    if (d_h) cudaFree(d_h);
    return bad;
}

// The Philox draw, host side - a transcription of the kernel's `philox_uniform` so the SAMPLED pick (not
// just the greedy argmax) can be pinned against a reference.  `__umulhi(a, b)` is the high half of a 32x32
// multiply, spelled `(uint32_t)(((uint64_t) a * b) >> 32)` here.
struct PhiloxRound {
    uint32_t& c0; uint32_t& c1; uint32_t& c2; uint32_t& c3;
    void step(uint32_t k0, uint32_t k1) const {
        const uint32_t hi0 = (uint32_t) (((uint64_t) 0x9E3779B9u * c0) >> 32);
        const uint32_t hi1 = (uint32_t) (((uint64_t) 0xBB67AE85u * c2) >> 32);
        const uint32_t lo0 = 0x9E3779B9u * c0;
        const uint32_t lo1 = 0xBB67AE85u * c2;
        const uint32_t n0 = hi1 ^ c1 ^ k0;
        const uint32_t n1 = lo1;
        const uint32_t n2 = hi0 ^ c3 ^ k1;
        const uint32_t n3 = lo0;
        c0 = n0; c1 = n1; c2 = n2; c3 = n3;
    }
};

float host_philox_uniform(uint64_t seed, uint64_t counter) {
    uint32_t c0 = (uint32_t) counter, c1 = (uint32_t) (counter >> 32);
    uint32_t c2 = (uint32_t) seed, c3 = (uint32_t) (seed >> 32);
    PhiloxRound r{c0, c1, c2, c3};
    for (int i = 0; i < 10; ++i) r.step((uint32_t) i, 0u);
    return (float) (c0 >> 8) * (1.0f / 16777216.0f);
}

// The full SAMPLED chain, host side - the kernel's `sampler_kernel` in serial form: penalties ONCE, on the raw
// logits during the top_k selection (ties to the lowest index - llama.cpp's single PENALTIES stage, issue #53),
// the min_p prefix cut, top_p's cut in double, then the temperature scaling of the survivors and one Philox
// draw at (seed, counter + row).
int sampled_reference(const std::vector<float>& l, const std::vector<int>& hist,
                      const strata::kernels::SamplerParams& p, int row) {
    auto penal = [&](float logit, int count) {
        if (count <= 0) return logit;
        if (logit <= 0.0f) logit *= p.penalty_repeat; else logit /= p.penalty_repeat;
        logit -= (float) count * p.penalty_freq + (count > 0 ? 1.0f : 0.0f) * p.penalty_present;
        return logit;
    };
    auto count = [&](int v) { int c = 0; for (int h : hist) if (h == v) ++c; return c; };
    const int nv = (int) l.size();
    const int KMAX = 64;
    const int k = p.top_k > 0 ? (p.top_k < KMAX ? p.top_k : KMAX) : 0;
    std::vector<int> sel_ids;
    std::vector<float> sel_logit;
    std::vector<char> taken((size_t) nv, 0);
    for (int i = 0; i < k; ++i) {
        int best = -1; float bv = 0;
        for (int v = 0; v < nv; ++v) {
            if (taken[(size_t) v]) continue;
            const float s = penal(l[(size_t) v], count(v));
            if (best < 0 || s > bv) { best = v; bv = s; }
        }
        taken[(size_t) best] = 1;
        sel_ids.push_back(best); sel_logit.push_back(bv);
    }
    int n_minp = (int) sel_ids.size();
    if (p.min_p > 0.0f) {
        const float thresh = sel_logit[0] + std::log(p.min_p);
        for (int i = 0; i < (int) sel_ids.size(); ++i)
            if (sel_logit[(size_t) i] < thresh) { n_minp = i; break; }
    }
    int n_keep = n_minp;
    if (p.top_p < 1.0f) {
        double sum = 0.0;
        for (int i = 0; i < n_minp; ++i) sum += std::exp((double) sel_logit[(size_t) i] - (double) sel_logit[0]);
        double cum = 0.0;
        int cut = n_minp;
        for (int i = 0; i < n_minp; ++i) {
            cum += std::exp((double) sel_logit[(size_t) i] - (double) sel_logit[0]) / sum;
            if (cum >= (double) p.top_p) { cut = i + 1; break; }
        }
        if (cut < p.min_keep) cut = p.min_keep < n_minp ? p.min_keep : n_minp;
        n_keep = cut;
    }
    const float inv_t = p.temperature > 0.0f ? 1.0f / p.temperature : 0.0f;
    // TEMPERATURE ONLY - the penalties rode once, in the selection above (issue #53: the old reference ran a
    // second pass here, so it agreed with the kernel's old two-pass bug instead of with llama.cpp).
    auto scaled = [&](int i) { return sel_logit[(size_t) i] * inv_t; };
    float smx = scaled(0);
    for (int i = 1; i < n_keep; ++i) smx = std::fmax(smx, scaled(i));
    double sum = 0.0;
    for (int i = 0; i < n_keep; ++i) sum += std::exp((double) scaled(i) - (double) smx);
    const float u = host_philox_uniform(p.seed, p.counter + (uint64_t) row);
    double cum = 0.0;
    int pick = sel_ids[(size_t) (n_keep - 1)];
    for (int i = 0; i < n_keep; ++i) {
        cum += std::exp((double) scaled(i) - (double) smx) / sum;
        if ((double) u < cum) { pick = sel_ids[(size_t) i]; break; }
    }
    return pick;
}

// Survivors after the min_p + top_p cuts, in the sampled chain - the quantity an order or a threshold
// actually changes, used to assert a fixture can SEE the feature before asserting the kernel matches.
int sampled_cut(const std::vector<float>& l, const std::vector<int>& hist, const strata::kernels::SamplerParams& p) {
    auto penal = [&](float logit, int count) {
        if (count <= 0) return logit;
        if (logit <= 0.0f) logit *= p.penalty_repeat; else logit /= p.penalty_repeat;
        logit -= (float) count * p.penalty_freq + (count > 0 ? 1.0f : 0.0f) * p.penalty_present;
        return logit;
    };
    auto count = [&](int v) { int c = 0; for (int h : hist) if (h == v) ++c; return c; };
    const int nv = (int) l.size();
    const int KMAX = 64;
    const int k = p.top_k > 0 ? (p.top_k < KMAX ? p.top_k : KMAX) : 0;
    std::vector<float> sel;
    std::vector<char> taken((size_t) nv, 0);
    for (int i = 0; i < k; ++i) {
        int best = -1; float bv = 0;
        for (int v = 0; v < nv; ++v) {
            if (taken[(size_t) v]) continue;
            const float s = penal(l[(size_t) v], count(v));
            if (best < 0 || s > bv) { best = v; bv = s; }
        }
        taken[(size_t) best] = 1; sel.push_back(bv);
    }
    int n_minp = (int) sel.size();
    if (p.min_p > 0.0f) {
        const float thresh = sel[0] + std::log(p.min_p);
        for (int i = 0; i < (int) sel.size(); ++i)
            if (sel[(size_t) i] < thresh) { n_minp = i; break; }
    }
    if (p.top_p >= 1.0f) return n_minp;
    double sum = 0.0;
    for (int i = 0; i < n_minp; ++i) sum += std::exp((double) sel[(size_t) i] - (double) sel[0]);
    double cum = 0.0;
    int cut = n_minp;
    for (int i = 0; i < n_minp; ++i) {
        cum += std::exp((double) sel[(size_t) i] - (double) sel[0]) / sum;
        if (cum >= (double) p.top_p) { cut = i + 1; break; }
    }
    if (cut < p.min_keep) cut = p.min_keep < n_minp ? p.min_keep : n_minp;
    return cut;
}

}  // namespace

int main(int argc, char** argv) {
    bool selftest = false;
    for (int i = 1; i < argc; ++i) {
        if (std::string(argv[i]) == "--selftest") selftest = true;
        else { std::fprintf(stderr, "usage: sampler_parity [--selftest]\n"); return 2; }
    }
    int bad = 0;
    const int NV = 512, NT = 4;

    // ---- fixture 1: plain greedy.  top_k = 0 (disabled), top_p = 1 (disabled), T = 1 -> argmax.
    {
        strata::kernels::SamplerParams p; p.top_k = 0; p.top_p = 1.0f; p.temperature = 1.0f; p.greedy = true;
        std::mt19937 rng(3); std::normal_distribution<float> g(0.0f, 1.0f);
        std::vector<float> l((size_t) NV * NT);
        for (auto& v : l) v = g(rng);
        std::vector<int> want((size_t) NT);
        for (int t = 0; t < NT; ++t) want[(size_t) t] = reference_pick({l.begin() + (size_t) t * NV, l.begin() + (size_t) (t + 1) * NV}, p, false);
        bad += run("greedy argmax", l, NT, p, want);
    }

    // ---- fixture 2: THE ORDER FIXTURE.  T = 0.5 sharpens the distribution enough that top_p = 0.5 cuts
    // differently before and after the scaling, and the two orders then pick DIFFERENT tokens.
    {
        strata::kernels::SamplerParams p; p.top_k = 0; p.top_p = 0.5f; p.temperature = 0.5f;
        p.min_keep = 1; p.greedy = true;
        std::vector<float> l((size_t) NV * NT, -1000.0f);
        for (int t = 0; t < NT; ++t) {
            // a flat-ish head so the cumulative mass crosses 0.5 inside it, and one clear leader
            l[(size_t) t * NV + 0] = 3.0f;
            for (int v = 1; v < 8; ++v) l[(size_t) t * NV + v] = 2.6f - 0.05f * (float) v;
        }
        std::vector<int> want((size_t) NT), other((size_t) NT);
        for (int t = 0; t < NT; ++t) {
            const std::vector<float> row(l.begin() + (size_t) t * NV, l.begin() + (size_t) (t + 1) * NV);
            want[(size_t) t] = reference_pick(row, p, false);      // the SPECIFIED order
            other[(size_t) t] = reference_pick(row, p, true);      // temperature first
        }
        // If the two orders agree on this fixture the test cannot see the order, and saying "the kernel
        // matches the spec" would be vacuous.
        // GREEDY CANNOT SEE THE ORDER, and saying otherwise would be a vacuous check: no filter removes the
        // global argmax, and temperature is monotonic, so the greedy pick is order-independent by
        // construction.  What the order changes is the top_p CUT, so the fixture is asserted to be
        // order-SENSITIVE at the cut, which is a property of the fixture rather than of the kernel.
        int cut_spec = 0, cut_alt = 0;
        for (int t = 0; t < NT; ++t) {
            const std::vector<float> row(l.begin() + (size_t) t * NV, l.begin() + (size_t) (t + 1) * NV);
            cut_spec += reference_cut(row, p, false);
            cut_alt += reference_cut(row, p, true);
        }
        const bool distinguishable = (cut_spec != cut_alt);
        std::printf("  %-34s %s (survivors: spec %d, temp-first %d)\n", "order is observable on this fixture",
                    distinguishable ? "yes" : "*** NO - THE FIXTURE CANNOT SEE THE ORDER ***", cut_spec,
                    cut_alt);
        if (!distinguishable) ++bad;
        // greedy is still checked here, but as an ARGMAX check, not an order check
        bad += run("greedy over this fixture", l, NT, p, want);
    }

    // ---- fixture 3: greedy consumes NO random number.  Two runs with different seeds must agree, or the
    // seeded streams diverge between greedy and sampled runs - which docs/sampling.md §3 calls out.
    {
        strata::kernels::SamplerParams a; a.top_k = 20; a.top_p = 0.95f; a.temperature = 1.0f; a.greedy = true; a.seed = 1;
        strata::kernels::SamplerParams b = a; b.seed = 999999;
        std::mt19937 rng(5); std::normal_distribution<float> g(0.0f, 1.0f);
        std::vector<float> l((size_t) NV * NT);
        for (auto& v : l) v = g(rng);
        std::vector<int> wa((size_t) NT);
        for (int t = 0; t < NT; ++t) wa[(size_t) t] = reference_pick({l.begin() + (size_t) t * NV, l.begin() + (size_t) (t + 1) * NV}, a, false);
        float* d_l = nullptr; int *d_a = nullptr, *d_b = nullptr;
        check(cudaMalloc(&d_l, l.size() * sizeof(float)), "m1");
        check(cudaMalloc(&d_a, (size_t) NT * sizeof(int)), "m2");
        check(cudaMalloc(&d_b, (size_t) NT * sizeof(int)), "m3");
        check(cudaMemcpy(d_l, l.data(), l.size() * sizeof(float), cudaMemcpyHostToDevice), "c1");
        strata::kernels::sample_tokens(d_l, NT, NV, nullptr, 0, a, d_a, nullptr);
        strata::kernels::sample_tokens(d_l, NT, NV, nullptr, 0, b, d_b, nullptr);
        std::vector<int> ga((size_t) NT), gb((size_t) NT);
        check(cudaMemcpy(ga.data(), d_a, ga.size() * sizeof(int), cudaMemcpyDeviceToHost), "g1");
        check(cudaMemcpy(gb.data(), d_b, gb.size() * sizeof(int), cudaMemcpyDeviceToHost), "g2");
        int mismatch = 0, wrong = 0;
        for (int t = 0; t < NT; ++t) {
            if (ga[(size_t) t] != gb[(size_t) t]) ++mismatch;
            if (ga[(size_t) t] != wa[(size_t) t]) ++wrong;
        }
        std::printf("  %-34s %s (seed-independent: %d differ; vs reference: %d wrong)\n",
                    "greedy ignores the seed", (!mismatch && !wrong) ? "matches" : "*** WRONG ***", mismatch,
                    wrong);
        bad += mismatch + wrong;
        cudaFree(d_l); cudaFree(d_a); cudaFree(d_b);
    }


    // ---- fixture 4: PENALTIES.  Two sub-cases, each built so the rule it tests decides the answer.
    {
        // host reference for the penalty stage, transcribed from llama_sampler_penalties_apply
        auto penal = [](float logit, int count, const strata::kernels::SamplerParams& p) {
            if (count <= 0) return logit;
            if (logit <= 0.0f) logit *= p.penalty_repeat; else logit /= p.penalty_repeat;
            logit -= (float) count * p.penalty_freq + (count > 0 ? 1.0f : 0.0f) * p.penalty_present;
            return logit;
        };
        auto pick = [&](const std::vector<float>& l, const std::vector<int>& hist,
                        const strata::kernels::SamplerParams& p, bool divide_unconditionally) {
            int best = 0; float bv = 0; bool first = true;
            for (int v = 0; v < (int) l.size(); ++v) {
                int c = 0; for (int h : hist) if (h == v) ++c;
                float s;
                if (divide_unconditionally && c > 0) {
                    s = l[(size_t) v] / p.penalty_repeat
                        - (float) c * p.penalty_freq - (c > 0 ? 1.0f : 0.0f) * p.penalty_present;
                } else {
                    s = penal(l[(size_t) v], c, p);
                }
                if (first || s > bv) { bv = s; best = v; first = false; }
            }
            return best;
        };

        const int NV2 = 8, NT2 = 2;
        strata::kernels::SamplerParams p; p.top_k = 0; p.top_p = 1.0f; p.temperature = 1.0f;
        p.greedy = true; p.penalty_last_n = 4; p.penalty_repeat = 2.0f;

        // A: ALL logits negative, so the multiply-or-divide rule decides the argmax
        std::vector<float> la((size_t) NV2 * NT2, -8.0f);
        for (int t = 0; t < NT2; ++t) {
            la[(size_t) t * NV2 + 0] = -1.0f;      // in the history -> penalised
            la[(size_t) t * NV2 + 1] = -1.2f;      // not penalised -> should win
        }
        std::vector<int> hist_a((size_t) NT2 * 4, -1);
        for (int t = 0; t < NT2; ++t) hist_a[(size_t) t * 4 + 0] = 0;
        std::vector<int> want_a((size_t) NT2), alt_a((size_t) NT2);
        for (int t = 0; t < NT2; ++t) {
            const std::vector<float> row(la.begin() + (size_t) t * NV2, la.begin() + (size_t) (t + 1) * NV2);
            const std::vector<int> h(hist_a.begin() + (size_t) t * 4, hist_a.begin() + (size_t) (t + 1) * 4);
            want_a[(size_t) t] = pick(row, h, p, false);
            alt_a[(size_t) t] = pick(row, h, p, true);      // divide unconditionally
        }
        const bool A_visible = want_a[0] != alt_a[0];
        std::printf("  %-34s %s (multiply-rule %d, divide-always %d)\n",
                    "multiply-or-divide is observable", A_visible ? "yes" : "*** NO ***", want_a[0],
                    alt_a[0]);
        if (!A_visible) ++bad;
        else bad += run("penalties: repeat on negatives", la, NT2, p, want_a, hist_a, 4);

        // B: the PRESENCE penalty is a boolean, so two occurrences cost the same as one.  The runner's margin
        // is inside the difference between one and two applications.
        std::vector<float> lb((size_t) NV2 * NT2, -8.0f);
        for (int t = 0; t < NT2; ++t) {
            lb[(size_t) t * NV2 + 0] = 5.0f;       // seen twice -> penalised ONCE (presence) + freq*2
            lb[(size_t) t * NV2 + 1] = 3.4f;       // unseen
        }
        std::vector<int> hist_b((size_t) NT2 * 4, -1);
        for (int t = 0; t < NT2; ++t) {
            hist_b[(size_t) t * 4 + 0] = 0;
            hist_b[(size_t) t * 4 + 1] = 0;        // twice
        }
        strata::kernels::SamplerParams q = p; q.penalty_present = 1.5f; q.penalty_freq = 0.0f;
        std::vector<int> want_b((size_t) NT2);
        for (int t = 0; t < NT2; ++t) {
            const std::vector<float> row(lb.begin() + (size_t) t * NV2, lb.begin() + (size_t) (t + 1) * NV2);
            const std::vector<int> h(hist_b.begin() + (size_t) t * 4, hist_b.begin() + (size_t) (t + 1) * 4);
            want_b[(size_t) t] = pick(row, h, q, false);
        }
        std::printf("  %-34s want token %d (with present=1.5, token 0 goes 5.0/2 - 1.5 = 1.0 vs token 1 at "
                    "3.4)\n", "presence penalty is a boolean", want_b[0]);
        bad += run("penalties: presence is boolean", lb, NT2, q, want_b, hist_b, 4);
    }

    // ---- fixture 5: TEMPERATURE 0 MUST STILL RETURN THE ARGMAX.  Regression test for a real bug.
    //
    // The greedy branch used to read `apply_penalties(l[v] * inv_t, ...)`.  `inv_t` is 0.0f whenever
    // temperature <= 0, so at temperature 0 EVERY logit became 0.0f and the argmax returned index 0 - the
    // sampler emitted token 0 forever, whatever the model predicted.  OpenAI clients send `temperature: 0`
    // for greedy decoding, so this was reachable from any ordinary client.
    //
    // It survived because EVERY other greedy fixture in this file sets temperature = 1.0f, where inv_t = 1.0
    // and the extra multiply is harmless.  The bug needs temperature <= 0 to appear, and no fixture used it.
    // The fixture below makes token 0 the WORST token in every row, so returning 0 is unambiguously wrong.
    {
        const int NV3 = 512, NT3 = 4;
        std::vector<float> l((size_t) NV3 * NT3, -5.0f);
        std::vector<int> want((size_t) NT3);
        for (int t = 0; t < NT3; ++t) {
            const int best = 100 + t;                    // the argmax is never token 0
            l[(size_t) t * NV3 + best] = 3.0f;
            l[(size_t) t * NV3 + 0] = -9.0f;             // token 0 is the worst in the row
            want[(size_t) t] = best;
        }
        strata::kernels::SamplerParams p0;
        p0.top_k = 0; p0.top_p = 1.0f; p0.temperature = 0.0f; p0.greedy = false;
        bad += run("T=0 greedy=false is the argmax", l, NT3, p0, want);

        strata::kernels::SamplerParams p1 = p0; p1.greedy = true;
        bad += run("T=0 greedy=true  is the argmax", l, NT3, p1, want);

        strata::kernels::SamplerParams p2 = p0; p2.greedy = true; p2.temperature = 1.0f;
        bad += run("T=1 greedy=true  is the argmax", l, NT3, p2, want);
    }

    // ---- fixture 6: PENALTIES IN THE SAMPLED CHAIN.  Fixture 4 pins the greedy (argmax) path; the sampled
    // chain carries the penalties through the selection, the cuts, the temperature scaling and the draw, so it
    // gets its own reference (the full single-pass chain with the host Philox) and its own observability
    // check: with the penalties on, the history row's favourite must LOSE a pick it would win penalty-free.
    {
        const int NV2 = 8, NT2 = 2;
        strata::kernels::SamplerParams p;
        p.top_k = 5; p.top_p = 0.9f; p.temperature = 0.8f; p.seed = 9; p.counter = 0;
        p.penalty_last_n = 4; p.penalty_repeat = 3.0f; p.penalty_freq = 0.2f; p.penalty_present = 0.6f;

        std::vector<float> l((size_t) NV2 * NT2, -8.0f);
        std::vector<int> hist((size_t) NT2 * 4, -1);
        for (int t = 0; t < NT2; ++t) {
            float* row = l.data() + (size_t) t * NV2;
            row[0] = 9.0f; row[1] = 4.5f; row[2] = 4.4f; row[3] = 4.3f;   // token 0 leads clean (9 vs 4.5)
            hist[(size_t) t * 4 + 0] = 0;                                  // and falls to 2.2/2.0 penalised
            hist[(size_t) t * 4 + 1] = t == 1 ? 0 : -1;                    // (repeat 3, freq, presence)
        }
        std::vector<int> want((size_t) NT2), clean((size_t) NT2);
        strata::kernels::SamplerParams clean_p = p;
        clean_p.penalty_last_n = 0; clean_p.penalty_repeat = 1.0f;
        clean_p.penalty_freq = 0.0f; clean_p.penalty_present = 0.0f;
        for (int t = 0; t < NT2; ++t) {
            const std::vector<float> row(l.begin() + (size_t) t * NV2, l.begin() + (size_t) (t + 1) * NV2);
            const std::vector<int> h(hist.begin() + (size_t) t * 4, hist.begin() + (size_t) (t + 1) * 4);
            want[(size_t) t] = sampled_reference(row, h, p, t);
            clean[(size_t) t] = sampled_reference(row, h, clean_p, t);
        }
        const bool visible = want[0] != clean[0] || want[1] != clean[1];
        std::printf("  %-34s %s (penalised picks %d/%d, clean %d/%d)\n",
                    "sampled penalties are observable", visible ? "yes" : "*** NO ***", want[0], want[1],
                    clean[0], clean[1]);
        if (!visible) ++bad;
        else bad += run("sampled chain: penalties + top_k/p", l, NT2, p, want, hist, 4);
    }

    // ---- fixture 7: MIN_P.  The cut is a PREFIX of the descending top_k list (logit >= max + log(min_p)),
    // so the fixture asserts the survivor count moves with the threshold (the observability half) and that
    // the kernel's pick equals the reference's through the full sampled chain (the correctness half).
    {
        const int NV3 = 8, NT3 = 2;
        std::vector<float> l((size_t) NV3 * NT3, -8.0f);
        for (int t = 0; t < NT3; ++t) {
            float* row = l.data() + (size_t) t * NV3;
            row[0] = 4.0f; row[1] = 3.5f; row[2] = 3.2f; row[3] = 3.1f;   // gaps keep the cut off the
            row[4] = 2.0f;                                                // logf/rounding knife edge
        }
        strata::kernels::SamplerParams base;
        base.top_k = 6; base.top_p = 1.0f; base.temperature = 0.9f; base.seed = 77;

        int c0 = 0, c05 = 0, c09 = 0;
        for (int t = 0; t < NT3; ++t) {
            const std::vector<float> row(l.begin() + (size_t) t * NV3, l.begin() + (size_t) (t + 1) * NV3);
            strata::kernels::SamplerParams q = base; q.min_p = 0.0f;
            c0 += sampled_cut(row, {}, q);
            q.min_p = 0.5f; c05 += sampled_cut(row, {}, q);
            q.min_p = 0.9f; c09 += sampled_cut(row, {}, q);
        }
        const bool visible = c0 > c05 && c05 > c09 && c09 >= NT3;
        std::printf("  %-34s %s (survivors: min_p 0 -> %d, 0.5 -> %d, 0.9 -> %d)\n",
                    "min_p cut is observable", visible ? "yes" : "*** NO ***", c0, c05, c09);
        if (!visible) ++bad;

        for (float mp : {0.0f, 0.5f, 0.9f}) {
            strata::kernels::SamplerParams q = base; q.min_p = mp;
            std::vector<int> want((size_t) NT3);
            for (int t = 0; t < NT3; ++t) {
                const std::vector<float> row(l.begin() + (size_t) t * NV3, l.begin() + (size_t) (t + 1) * NV3);
                want[(size_t) t] = sampled_reference(row, {}, q, t);
            }
            char name[64];
            std::snprintf(name, sizeof name, "sampled chain: min_p=%.1f", (double) mp);
            bad += run(name, l, NT3, q, want);
        }
    }

    // ---- fixture 8: THE PENALTY WINDOW IS THE TAIL.  With an 8-entry history and penalty_last_n = 4, only
    // the LAST four entries count: a token punished in the old half must come back to full strength, and one
    // punished in the tail half stays down.  The reference counts the same tail; the observability check runs
    // the reference once more WITHOUT the clamp (counting all 8) and requires the picks to differ.
    {
        const int NV4 = 8;
        strata::kernels::SamplerParams p;
        p.top_k = 0; p.top_p = 1.0f; p.temperature = 1.0f; p.greedy = true;
        p.penalty_last_n = 4; p.penalty_repeat = 3.0f; p.penalty_freq = 0.3f; p.penalty_present = 0.5f;

        std::vector<float> l((size_t) NV4, -8.0f);
        l[0] = 6.0f; l[3] = 6.5f;                       // token 0 leads clean; token 3 is the tail offender
        std::vector<int> hist = {0, 0, 0, 0, 3, 3, 3, 3};   // token 0 old (out), token 3 in the tail

        auto pick_clamped = [&](bool clamp) {
            int best = 0; float bv = 0; bool first = true;
            for (int v = 0; v < NV4; ++v) {
                int c = 0;
                for (int i = 0; i < (clamp ? 4 : 8); ++i) if (hist[(size_t) (8 - (clamp ? 4 : 8) + i)] == v) ++c;
                float logit = l[(size_t) v];
                if (c > 0) { logit = logit <= 0.0f ? logit * p.penalty_repeat : logit / p.penalty_repeat;
                             logit -= (float) c * p.penalty_freq + p.penalty_present; }
                if (first || logit > bv) { bv = logit; best = v; first = false; }
            }
            return best;
        };
        const int want = pick_clamped(true), unclamped = pick_clamped(false);
        const bool visible = want != unclamped;
        std::printf("  %-34s %s (clamped pick %d, full-history pick %d)\n",
                    "penalty window clamp is observable", visible ? "yes" : "*** NO ***", want, unclamped);
        if (!visible) ++bad;
        else bad += run("penalty window: tail only", {l.begin(), l.end()}, 1, p, {want}, hist, 8);
    }

    // ---- fixture 9: THE ONE-PASS PENALTY ARITHMETIC, PINNED AGAINST ISSUE #53'S OWN NUMBERS.  The fixture is
    // the issue's two-token example, computed independently of the `penal`/`sampled_reference` helpers above:
    // logits [0, 0], history [0], last_n 1, repeat 1, freq 0, present 1.5, T 0.7, top_k 2 - both candidates
    // survive every filter, so only the penalty pass and the scaling decide.  One pass puts token 0 at
    // (0 - 1.5)/T = -2.1428... against token 1's 0, probability 0.10500059; the old two-pass chain penalised
    // it once more after the scaling (-3.6428...) and got 0.02550967.  Token 1 draws while u is below its
    // cumulative share, so token 0 wins iff u >= 1 - p(token 0), and the two chains' thresholds leave a ~8%
    // band of draws where they disagree - a counter whose uniform lands inside it must pick token 0.
    {
        const int NV5 = 2, NT5 = 1;
        strata::kernels::SamplerParams p;
        p.top_k = 2; p.top_p = 1.0f; p.temperature = 0.7f; p.seed = 4; p.counter = 0;
        p.penalty_last_n = 1; p.penalty_repeat = 1.0f; p.penalty_freq = 0.0f; p.penalty_present = 1.5f;

        // each chain's token-0 logit from its own closed form, kernel-faithful down to the float reciprocal
        // (the kernel scales by `sel_logit * (1/T)`, not `sel_logit / T`), then the two-token softmax
        const float inv_t = 1.0f / p.temperature;
        const double a_single = (double) ((0.0f - p.penalty_present) * inv_t);
        const double a_double = a_single - (double) p.penalty_present;
        auto share = [](double a) { return std::exp(a) / (std::exp(a) + 1.0); };
        const double p_single = share(a_single), p_double = share(a_double);
        const bool anchors = std::abs(p_single - 0.1050005850) < 1e-6 &&
                             std::abs(p_double - 0.0255096664) < 1e-6;
        std::printf("  %-34s %s (single-pass p = %.10f, two-pass p = %.10f)\n",
                    "#53's probabilities", anchors ? "match" : "*** WRONG ***", p_single, p_double);
        if (!anchors) ++bad;

        const double lo = 1.0 - p_single, hi = 1.0 - p_double;
        uint64_t found = 0;
        bool have = false;
        for (uint64_t c = 0; c < 0x100000ull && !have; ++c) {
            const double u = host_philox_uniform(p.seed, c);
            if (u > lo + 1e-9 && u < hi - 1e-9) { found = c; have = true; }
        }
        std::printf("  %-34s %s (counter %llu: u = %.7f in [%.7f, %.7f) -> single-pass picks 0, "
                    "two-pass picks 1)\n", "#53's band is observable", have ? "yes" : "*** NO ***",
                    (unsigned long long) found, have ? host_philox_uniform(p.seed, found) : 0.0, lo, hi);
        if (!have) ++bad;
        else {
            const std::vector<float> l((size_t) NV5 * NT5, 0.0f);   // logits [0, 0]
            const std::vector<int> hist = {0};                      // history [A]
            strata::kernels::SamplerParams q = p; q.counter = found;
            bad += run("#53: one penalty pass, then T", l, NT5, q, {0}, hist, 1);
        }
    }

    // ---- fixture 10: A STALE HISTORY WITH last_n = 0 IS INERT.  A caller can hand over a history buffer
    // from a previous penalised request while this request disables the penalties - the run must equal the
    // no-history run in both kernels, and the bitmap the launch did not size must stay untouched.
    {
        std::mt19937 rng(11); std::normal_distribution<float> g(0.0f, 1.0f);
        std::vector<float> l((size_t) NV * NT);
        for (auto& v : l) v = g(rng);
        std::vector<int> hist((size_t) NT * 8, -1);
        for (int t = 0; t < NT; ++t) hist[(size_t) t * 8] = 3;

        strata::kernels::SamplerParams p;
        p.top_k = 20; p.top_p = 0.95f; p.temperature = 0.8f; p.seed = 5;
        std::vector<int> want((size_t) NT);
        for (int t = 0; t < NT; ++t)
            want[(size_t) t] = sampled_reference({l.begin() + (size_t) t * NV,
                                                  l.begin() + (size_t) (t + 1) * NV}, {}, p, t);
        bad += run("stale history, last_n=0 (sampled)", l, NT, p, want, hist, 8);

        strata::kernels::SamplerParams gp = p; gp.greedy = true; gp.top_k = 0; gp.top_p = 1.0f;
        std::vector<int> gwant((size_t) NT);
        for (int t = 0; t < NT; ++t)
            gwant[(size_t) t] = reference_pick({l.begin() + (size_t) t * NV,
                                                l.begin() + (size_t) (t + 1) * NV}, gp, false);
        bad += run("stale history, last_n=0 (greedy)", l, NT, gp, gwant, hist, 8);
    }

    // A continuous stream and individual decode calls consume the same draw counters.
    {
        constexpr int count = 32, vocab = 16;
        std::vector<float> uniform(count * vocab, 0.0f);
        float* input = nullptr;
        int* output = nullptr;
        check(cudaMalloc(&input, uniform.size() * sizeof(float)), "counter logits");
        check(cudaMalloc(&output, count * sizeof(int)), "counter output");
        check(cudaMemcpy(input, uniform.data(), uniform.size() * sizeof(float), cudaMemcpyHostToDevice), "counter upload");
        strata::kernels::SamplerParams p;
        p.top_k = vocab; p.top_p = 1.0f; p.seed = 123; p.counter = (uint64_t(1) << 32) + 7;
        strata::kernels::sample_tokens(input, count, vocab, nullptr, 0, p, output, nullptr);
        std::vector<int> batch(count), singles(count), repeated(count);
        check(cudaMemcpy(batch.data(), output, count * sizeof(int), cudaMemcpyDeviceToHost), "counter batch");
        for (int i = 0; i < count; ++i) {
            auto one = p; one.counter += i;
            strata::kernels::sample_tokens(input, 1, vocab, nullptr, 0, one, output + i, nullptr);
        }
        check(cudaMemcpy(singles.data(), output, count * sizeof(int), cudaMemcpyDeviceToHost), "counter singles");
        strata::kernels::sample_tokens(input, count, vocab, nullptr, 0, p, output, nullptr);
        check(cudaMemcpy(repeated.data(), output, count * sizeof(int), cudaMemcpyDeviceToHost), "counter repeated");
        bool varies = false;
        for (int i = 1; i < count; ++i) varies |= batch[i] != batch[0];
        const bool valid = batch == singles && batch == repeated && varies;
        std::printf("  sampler draw counter segmentation/repeat: %s\n", valid ? "PASS" : "FAIL");
        bad += !valid;
        cudaFree(input); cudaFree(output);
    }

    std::printf("\nsampler: %d failures\n", bad);
    if (bad) return 1;
    if (selftest) std::printf("sampler_parity OK\n");
    return 0;
}
