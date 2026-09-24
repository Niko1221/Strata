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
