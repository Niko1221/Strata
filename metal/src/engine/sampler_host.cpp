// metal/src/engine/sampler_host.cpp - the sampler chain on the CPU: strata::kernels::sample_tokens for the Metal engine.
//
// The same semantics as src/kernels/cuda/sampler.cu (include/strata/kernels/sampler.hpp): penalties -> top_k (<= 64,
// 0 = 64) -> top_p -> min_p -> temperature -> one Philox draw keyed by (seed, counter + row).  The logits live in
// unified memory, so the head's rows are read where they lie; the pass is one scan of the vocabulary per row (a
// k-entry insertion list), not a sort.  Unlike the CUDA entry point, `logits`, `history` and `out` are HOST pointers
// here (strata::gpu::host_view of the device-looking ones).
//
// Verify windows: row t's draw is Philox(seed, counter + t), tied to the POSITION it samples.  The engine accepts a
// draft when the sampled token equals it, so with a point-mass draft q the acceptance probability is p_target(d) and a
// rejected row's token is a sample of p conditioned on != d - exactly the normalised residual max(p - q, 0).  That is
// the standard rejection step, with no second code path.
#include "strata/kernels/sampler.hpp"

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <utility>
#include <limits>
#include <vector>

namespace strata::kernels {
namespace {

// sampler.cu's philox4x32_round / philox_uniform, bit for bit
float philox_uniform(uint64_t seed, uint64_t counter) {
    uint32_t c0 = (uint32_t) counter, c1 = (uint32_t) (counter >> 32);
    uint32_t c2 = (uint32_t) seed, c3 = (uint32_t) (seed >> 32);
    for (uint32_t i = 0; i < 10; ++i) {
        const uint32_t hi0 = (uint32_t) (((uint64_t) 0x9E3779B9u * c0) >> 32);
        const uint32_t hi1 = (uint32_t) (((uint64_t) 0xBB67AE85u * c2) >> 32);
        const uint32_t lo0 = 0x9E3779B9u * c0;
        const uint32_t lo1 = 0xBB67AE85u * c2;
        const uint32_t n0 = hi1 ^ c1 ^ i;
        const uint32_t n1 = lo1;
        const uint32_t n2 = hi0 ^ c3 ^ 0u;
        const uint32_t n3 = lo0;
        c0 = n0; c1 = n1; c2 = n2; c3 = n3;
    }
    return (float) (c0 >> 8) * (1.0f / 16777216.0f);
}

float apply_penalties(float logit, int count, const SamplerParams& p) {
    if (count <= 0) return logit;
    if (logit <= 0.0f) logit *= p.penalty_repeat;
    else               logit /= p.penalty_repeat;
    logit -= (float) count * p.penalty_freq + 1.0f * p.penalty_present;
    return logit;
}

constexpr int kSelMax = 64;

void sample_row(const float* l, int n_vocab, const int* hrow, int hlen, const SamplerParams& p, uint64_t draw, int* out) {
    // the penalty window is the TAIL of the row's history; ids outside the vocabulary are ignored
    std::vector<std::pair<int, int>> counts;   // (id, occurrences), few entries: the window is short
    if (hrow != nullptr && hlen > 0 && p.penalty_last_n > 0) {
        const int w = hlen < p.penalty_last_n ? hlen : p.penalty_last_n;
        for (int i = hlen - w; i < hlen; ++i) {
            const int v = hrow[i];
            if (v < 0 || v >= n_vocab) continue;
            bool found = false;
            for (auto& c : counts) if (c.first == v) { ++c.second; found = true; break; }
            if (!found) counts.push_back({v, 1});
        }
    }
    const bool greedy = p.greedy || p.temperature <= 0.0f;
    int k = greedy ? 1 : ((p.top_k > 0 && p.top_k < kSelMax) ? p.top_k : kSelMax);
    if (k > n_vocab) k = n_vocab;

    // the top-k list in selection order: descending by value, ties to the lower id; NaN and -inf are never picked
    int ids[kSelMax];
    float lg[kSelMax];
    int n = 0;
    const float neg_inf = -std::numeric_limits<float>::infinity();
    float floor_v = neg_inf;   // the k-th value once the list is full
    auto offer = [&](int v, float s) {
        if (!(s > neg_inf)) return;
        if (n == k && !(s > lg[k - 1])) return;   // equal to the last: the lower id (already in the list) wins
        int i = n < k ? n++ : k - 1;
        while (i > 0 && lg[i - 1] < s) { lg[i] = lg[i - 1]; ids[i] = ids[i - 1]; --i; }
        lg[i] = s;
        ids[i] = v;
    };
    if (counts.empty()) {
        for (int v = 0; v < n_vocab; ++v) {
            const float s = l[v];
            if (n == k && !(s > floor_v)) continue;
            offer(v, s);
            if (n == k) floor_v = lg[k - 1];
        }
    } else {
        // penalised ids are offered at their penalised value, in id order with the rest (the tie rule is by id)
        std::vector<float> pen(counts.size());
        for (size_t j = 0; j < counts.size(); ++j) pen[j] = apply_penalties(l[counts[j].first], counts[j].second, p);
        for (int v = 0; v < n_vocab; ++v) {
            float s = l[v];
            for (size_t j = 0; j < counts.size(); ++j) if (counts[j].first == v) { s = pen[j]; break; }
            if (n == k && !(s > floor_v)) continue;
            offer(v, s);
            if (n == k) floor_v = lg[k - 1];
        }
    }
    for (int i = n; i < k; ++i) { ids[i] = 0; lg[i] = neg_inf; }   // the sampler's "no candidate" sentinel
    if (greedy) { *out = ids[0]; return; }

    // ---- top_p over the list (double arithmetic as the kernel), min_p, temperature, one draw
    int n_keep = k;
    float mx = lg[0];
    for (int i = 1; i < k; ++i) mx = std::fmax(mx, lg[i]);
    if (p.top_p < 1.0f) {
        double ex[kSelMax], sum = 0.0;
        for (int i = 0; i < k; ++i) { ex[i] = std::exp((double) lg[i] - (double) mx); sum += ex[i]; }
        double cum = 0.0;
        int cut = k;
        for (int i = 0; i < k; ++i) {
            cum += ex[i] / sum;
            if (cum >= (double) p.top_p) { cut = i + 1; break; }
        }
        if (cut < p.min_keep) cut = p.min_keep < k ? p.min_keep : k;
        n_keep = cut;
    }
    if (p.min_p > 0.0f) {
        const float thresh = lg[0] + std::log(p.min_p);
        for (int i = 0; i < n_keep; ++i)
            if (lg[i] < thresh) { n_keep = i; break; }
    }
    if (n_keep < 1) n_keep = 1;
    const float inv_t = 1.0f / p.temperature;
    float smx = lg[0] * inv_t;
    for (int i = 1; i < n_keep; ++i) smx = std::fmax(smx, lg[i] * inv_t);
    double ex[kSelMax], sum = 0.0;
    for (int i = 0; i < n_keep; ++i) { ex[i] = std::exp((double) (lg[i] * inv_t) - (double) smx); sum += ex[i]; }
    int pick = ids[n_keep - 1];
    const float u = philox_uniform(p.seed, draw);
    double cum = 0.0;
    for (int i = 0; i < n_keep; ++i) {
        cum += ex[i] / sum;
        if ((double) u < cum) { pick = ids[i]; break; }
    }
    *out = pick;
}

}  // namespace

// Host-pointer version of the CUDA entry point (the Metal engine has no stream: `stream` is ignored).
void sample_tokens(const float* logits, int n_tokens, int n_vocab, const int* history, int history_len,
                   const SamplerParams& p, int* out, void* stream) {
    (void) stream;
    if (n_tokens <= 0 || n_vocab <= 0) return;
    if (p.penalty_last_n > 0 && (history == nullptr || history_len <= 0)) {
        std::fprintf(stderr, "sample_tokens: penalty_last_n %d needs a history (got %p, len %d)\n", p.penalty_last_n,
                     (const void*) history, history_len);
        std::exit(1);
    }
    for (int t = 0; t < n_tokens; ++t)
        sample_row(logits + (size_t) t * (size_t) n_vocab, n_vocab,
                   history != nullptr ? history + (size_t) t * (size_t) history_len : nullptr, history_len, p,
                   p.counter + (uint64_t) t, out + t);
}

}  // namespace strata::kernels
