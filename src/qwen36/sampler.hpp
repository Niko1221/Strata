// sampler.hpp - host-side sampler with Strata's semantics: an absent temperature means greedy, an absent filter means off,
// and a sampled draw looks at no more than 64 candidates (top_k 0 or > 64 -> 64).  CUDA-free, unit-tested on a CPU.
#pragma once

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <random>
#include <vector>

namespace q36 {

struct SamplingParams {
    float temperature = 0.f, top_p = 1.f, min_p = 0.f;
    int top_k = 0;
    float repeat_penalty = 1.f, freq_penalty = 0.f, presence_penalty = 0.f;
    int penalty_last_n = 64;
    uint64_t seed = 0;  // 0 = random
};

class Sampler {
  public:
    explicit Sampler(const SamplingParams& p) : p_(p), rng_(p.seed ? p.seed : std::random_device{}()) {}

    // `logits` is modified (penalties are applied in place).  `history` = every token so far (prompt + generated).
    int sample(std::vector<float>& logits, const std::vector<int>& history) {
        apply_penalties(logits, history);
        const int n = (int)logits.size();
        if (p_.temperature <= 0.f) return (int)(std::max_element(logits.begin(), logits.end()) - logits.begin());

        const int k = std::min(n, (p_.top_k > 0 && p_.top_k <= 64) ? p_.top_k : 64);
        std::vector<int> idx(n);
        for (int i = 0; i < n; ++i) idx[i] = i;
        std::partial_sort(idx.begin(), idx.begin() + k, idx.end(), [&](int a, int b) {
            return logits[a] > logits[b] || (logits[a] == logits[b] && a < b);
        });
        idx.resize(k);
        std::vector<double> pr(k);
        const double top = logits[idx[0]];
        double sum = 0;
        for (int i = 0; i < k; ++i) { pr[i] = std::exp(((double)logits[idx[i]] - top) / p_.temperature); sum += pr[i]; }
        for (double& x : pr) x /= sum;
        int keep = k;
        if (p_.min_p > 0.f) {
            int j = 0;
            while (j < keep && pr[j] >= (double)p_.min_p * pr[0]) ++j;
            keep = std::max(1, j);
        }
        if (p_.top_p > 0.f && p_.top_p < 1.f) {
            double acc = 0;
            int j = 0;
            while (j < keep) { acc += pr[j]; ++j; if (acc >= p_.top_p) break; }
            keep = std::max(1, j);
        }
        double tot = 0;
        for (int i = 0; i < keep; ++i) tot += pr[i];
        std::uniform_real_distribution<double> u(0.0, tot);
        double r = u(rng_), acc = 0;
        for (int i = 0; i < keep; ++i) { acc += pr[i]; if (r <= acc) return idx[i]; }
        return idx[keep - 1];
    }

  private:
    void apply_penalties(std::vector<float>& logits, const std::vector<int>& history) const {
        if (p_.repeat_penalty == 1.f && p_.freq_penalty == 0.f && p_.presence_penalty == 0.f) return;
        const int n = (int)history.size();
        const int from = p_.penalty_last_n > 0 ? std::max(0, n - p_.penalty_last_n) : 0;
        std::vector<std::pair<int, int>> counts;  // (token, count), small
        for (int i = from; i < n; ++i) {
            const int t = history[i];
            if (t < 0 || t >= (int)logits.size()) continue;
            auto it = std::find_if(counts.begin(), counts.end(), [&](const auto& c) { return c.first == t; });
            if (it == counts.end()) counts.emplace_back(t, 1); else ++it->second;
        }
        for (const auto& [t, cnt] : counts) {
            float& l = logits[t];
            if (p_.repeat_penalty != 1.f) l = l > 0 ? l / p_.repeat_penalty : l * p_.repeat_penalty;
            l -= p_.freq_penalty * (float)cnt + p_.presence_penalty;
        }
    }
    SamplingParams p_;
    std::mt19937_64 rng_;
};

}  // namespace q36
