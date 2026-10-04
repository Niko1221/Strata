// dsv4/sampler.cpp - see dsv4/sampler.hpp.  Reference semantics follow llama.cpp's sample.cpp:
// penalties first (on the raw logits), then temperature, then top-k, then the nucleus (top-p) and min_p,
// then a roulette over what is left.  With temperature == 0 it is a plain argmax.
#include "dsv4/sampler.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstring>
#include <unordered_map>

namespace dsv4 {

namespace {

// xorshift64*: 12 nanoseconds a draw, and reproducible from the seed (a request's seed must replay).
inline uint64_t xs_next(uint64_t& s) {
    s ^= s >> 12;
    s ^= s << 25;
    s ^= s >> 27;
    return s * 2685821657736338717ULL;
}

}  // namespace

void Sampler::init_rng() {
    st_ = o_.seed ? o_.seed
                  : (uint64_t) std::chrono::high_resolution_clock::now().time_since_epoch().count() ^
                        (uint64_t) (uintptr_t) &st_;
    if (!st_) st_ = 0x9E3779B97F4A7C15ULL;
    for (int i = 0; i < 8; ++i) xs_next(st_);  // warm up: a seed of 1 must not start near 0
}

float Sampler::uniform() {
    // 53 bits of mantissa, like every real RNG: (a >> 11) * 2^-53 in [0,1).
    return (float) ((double) (xs_next(st_) >> 11) * (1.0 / 9007199254740992.0));
}

int Sampler::sample(const float* logits, int n, const std::vector<int>& history) {
    if (n <= 0) return 0;
    std::vector<float> buf;
    const float* lg = logits;
    const bool penal = (o_.penalty_repeat != 1.0f || o_.penalty_freq != 0.0f || o_.penalty_present != 0.0f) &&
                       o_.penalty_last_n > 0;
    if (penal) {
        buf.assign(logits, logits + n);
        const int win = std::min((int) history.size(), o_.penalty_last_n);
        const int first = (int) history.size() - win;
        std::unordered_map<int, int> count;
        for (int i = first; i < (int) history.size(); ++i) {
            const int t = history[(size_t) i];
            if (t >= 0 && t < n) ++count[t];
        }
        for (const auto& kv : count) {
            float& x = buf[(size_t) kv.first];
            if (o_.penalty_repeat != 1.0f) x = x >= 0.0f ? x / o_.penalty_repeat : x * o_.penalty_repeat;
            x -= o_.penalty_freq * (float) kv.second;
            x -= o_.penalty_present;
        }
        lg = buf.data();
    }

    // Greedy: the highest logit, no sort, no RNG.
    if (!(o_.temperature > 0.0f)) {
        int best = 0;
        float bv = lg[0];
        for (int i = 1; i < n; ++i) if (lg[i] > bv) { bv = lg[i]; best = i; }
        return best;
    }

    // Keep the top_k candidates (top_k = 0: everything).  nth_element, then sort only that slice.
    std::vector<int> idx((size_t) n);
    for (int i = 0; i < n; ++i) idx[(size_t) i] = i;
    int keep = n;
    if (o_.top_k > 0 && o_.top_k < n) {
        keep = o_.top_k;
        std::nth_element(idx.begin(), idx.begin() + keep, idx.end(),
                         [&](int a, int b) { return lg[a] > lg[b]; });
        idx.resize((size_t) keep);
    }
    std::sort(idx.begin(), idx.end(), [&](int a, int b) { return lg[a] > lg[b]; });

    // Softmax over the kept candidates, in the temperature-scaled order.
    const float inv_t = 1.0f / o_.temperature;
    const float m = lg[idx[0]];
    std::vector<float> p((size_t) idx.size());
    double sum = 0.0;
    for (size_t i = 0; i < idx.size(); ++i) {
        const double e = std::exp(((double) lg[idx[i]] - (double) m) * inv_t);
        p[i] = (float) e;
        sum += e;
    }
    if (!(sum > 0.0) || !std::isfinite(sum)) {  // every logit was -inf/NaN: fall back to the argmax
        return idx[0];
    }
    const double inv_sum = 1.0 / sum;
    for (float& x : p) x = (float) ((double) x * inv_sum);

    // top-p (nucleus) and min_p cut the tail; both are relative to the SORTED probabilities.
    size_t cut = p.size();
    if (o_.top_p > 0.0f && o_.top_p < 1.0f) {
        double acc = 0.0;
        for (size_t i = 0; i < p.size(); ++i) {
            acc += p[i];
            cut = i + 1;
            if (acc >= (double) o_.top_p) break;
        }
    }
    if (o_.min_p > 0.0f) {
        const double lim = (double) p[0] * (double) o_.min_p;
        while (cut > 1 && (double) p[cut - 1] < lim) --cut;
    }
    if (cut < 1) cut = 1;

    // Roulette over the survivors.
    double total = 0.0;
    for (size_t i = 0; i < cut; ++i) total += (double) p[i];
    double r = (double) uniform() * total;
    size_t pick = cut - 1;
    for (size_t i = 0; i < cut; ++i) {
        r -= (double) p[i];
        if (r <= 0.0) { pick = i; break; }
    }
    return idx[pick];
}

}  // namespace dsv4
