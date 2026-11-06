// metal/src/engine/host_sampler.cpp - see host_sampler.h.
//
// The selection is the CUDA kernels' list: candidates ranked by (score descending, id ascending), a strict total
// order, so the list does not depend on the order candidates are offered in.  NaN and -inf are never candidates; a
// list with fewer than k candidates is filled with (id 0, -inf), as a kernel round that finds nothing stores.  The
// scan offers a block of 16 logits only when one of them reaches the list's last score (NEON compare), so a row costs
// about one pass over its 248,320 floats.  The penalised ids (at most the window's length) are taken out of the scan
// and offered with their penalised scores.  The tail (top_p, min_p, temperature, the draw) is the kernels' arithmetic
// in the same order: float scores, double sums.
#include "host_sampler.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <utility>
#include <vector>

#if defined(__ARM_NEON)
#include <arm_neon.h>
#endif

namespace strata::metal_engine {
namespace {

constexpr int kSelMax = 64;   // the widest top_k list (sampler.cu's kSelMax)
constexpr float kNegInf = -std::numeric_limits<float>::infinity();

struct TopList {
    int k = 1, n = 0;
    float val[kSelMax];
    int id[kSelMax];
    static bool before(float v, int i, float w, int j) { return v > w || (v == w && i < j); }
    // the list's last score once full (a candidate below it cannot enter), -inf before
    float floor() const { return n == k ? val[k - 1] : kNegInf; }
    void offer(float v, int i) {
        if (!(v > kNegInf)) return;   // NaN and -inf are never picked
        if (n == k) {
            if (!before(v, i, val[k - 1], id[k - 1])) return;
            --n;
        }
        int at = n;
        for (; at > 0 && before(v, i, val[at - 1], id[at - 1]); --at) {
            val[at] = val[at - 1];
            id[at] = id[at - 1];
        }
        val[at] = v;
        id[at] = i;
        ++n;
    }
};

// sampler.cu's apply_penalties (llama_sampler_penalties_apply): the repeat penalty multiplies non-positive scores and
// divides positive ones; the presence penalty is boolean
float penalised(float s, int count, const strata::kernels::SamplerParams& p) {
    if (count <= 0) return s;
    if (s <= 0.0f) s *= p.penalty_repeat;
    else           s /= p.penalty_repeat;
    s -= (float) count * p.penalty_freq + (count > 0 ? 1.0f : 0.0f) * p.penalty_present;
    return s;
}

// offers every v in [0, nv) of `s` that is not marked in `skip` (null: none)
void scan(const float* s, int nv, const std::vector<uint64_t>* skip, TopList& L) {
    auto one = [&](int v) {
        if (skip != nullptr && (((*skip)[(size_t) v >> 6] >> (v & 63)) & 1u)) return;
        L.offer(s[v], v);
    };
    int v = 0;
    for (; v < nv && L.n < L.k; ++v) one(v);
#if defined(__ARM_NEON)
    for (; v + 16 <= nv; v += 16) {
        const float32x4_t f = vdupq_n_f32(L.floor());
        // a score equal to the floor can still enter (a lower id than the last entry's, from the penalised set)
        const uint32x4_t m = vorrq_u32(vorrq_u32(vcgeq_f32(vld1q_f32(s + v), f), vcgeq_f32(vld1q_f32(s + v + 4), f)),
                                       vorrq_u32(vcgeq_f32(vld1q_f32(s + v + 8), f), vcgeq_f32(vld1q_f32(s + v + 12), f)));
        if (vmaxvq_u32(m) == 0) continue;
        for (int j = 0; j < 16; ++j) one(v + j);
    }
#endif
    for (; v < nv; ++v)
        if (s[v] >= L.floor()) one(v);
}

}  // namespace

float philox_uniform(uint64_t seed, uint64_t counter) {
    // sampler.cu's philox_uniform: Philox 4x32-10, key (round, 0), the counter in (c0, c1), the seed in (c2, c3)
    uint32_t c0 = (uint32_t) counter, c1 = (uint32_t) (counter >> 32);
    uint32_t c2 = (uint32_t) seed, c3 = (uint32_t) (seed >> 32);
    for (uint32_t i = 0; i < 10; ++i) {
        const uint64_t m0 = (uint64_t) 0x9E3779B9u * c0, m1 = (uint64_t) 0xBB67AE85u * c2;
        const uint32_t n0 = (uint32_t) (m1 >> 32) ^ c1 ^ i;
        const uint32_t n2 = (uint32_t) (m0 >> 32) ^ c3;
        c1 = (uint32_t) m1;
        c3 = (uint32_t) m0;
        c0 = n0;
        c2 = n2;
    }
    return (float) (c0 >> 8) * (1.0f / 16777216.0f);   // 24 bits: uniform in [0, 1), never 1.0
}

int sample_row(const float* logits, int nv, const int32_t* history, int history_len,
               const strata::kernels::SamplerParams& p, const float* logit_bias, uint64_t counter) {
    if (nv <= 0) return 0;
    const bool greedy = p.greedy || p.temperature <= 0.0f;
    // the scores before the penalties: the logits, or logits + bias (one pass into a scratch row)
    const float* s = logits;
    thread_local std::vector<float> biased;
    if (logit_bias != nullptr) {
        biased.resize((size_t) nv);
        for (int v = 0; v < nv; ++v) biased[(size_t) v] = logits[v] + logit_bias[v];
        s = biased.data();
    }
    // the counted window: the last min(penalty_last_n, history_len) entries; ids outside the vocabulary are ignored
    thread_local std::vector<std::pair<int, int>> pen;   // (id, count)
    thread_local std::vector<uint64_t> mark;
    pen.clear();
    const int hlen = (history != nullptr && p.penalty_last_n > 0) ? std::min(p.penalty_last_n, history_len) : 0;
    if (hlen > 0) {
        for (int i = history_len - hlen; i < history_len; ++i)
            if (history[i] >= 0 && history[i] < nv) pen.push_back({history[i], 0});
        std::sort(pen.begin(), pen.end());
        size_t w = 0;
        for (size_t r = 0; r < pen.size();) {
            size_t e = r;
            while (e < pen.size() && pen[e].first == pen[r].first) ++e;
            pen[w++] = {pen[r].first, (int) (e - r)};
            r = e;
        }
        pen.resize(w);
        mark.assign(((size_t) nv + 63) / 64, 0);
        for (const auto& [id, c] : pen) mark[(size_t) id >> 6] |= 1ull << (id & 63);
    }
    TopList L;
    L.k = greedy ? 1 : std::min(nv, (p.top_k > 0 && p.top_k < kSelMax) ? p.top_k : kSelMax);
    scan(s, nv, pen.empty() ? nullptr : &mark, L);
    for (const auto& [id, c] : pen) L.offer(penalised(s[id], c, p), id);
    if (greedy) return L.n > 0 ? L.id[0] : 0;   // sampler_greedy_kernel: 0 when nothing is above -inf
    const int k = L.k;
    for (int i = L.n; i < k; ++i) { L.id[i] = 0; L.val[i] = kNegInf; }   // the rounds that found nothing
    const float* lg = L.val;
    // top_p over the list, then min_p on its survivors, then the temperature and one draw (sampler.cu's tail)
    int n_keep = k;
    float mx = lg[0];
    for (int i = 1; i < k; ++i) mx = std::fmax(mx, lg[i]);
    if (p.top_p < 1.0f) {
        double sum = 0.0;
        for (int i = 0; i < k; ++i) sum += std::exp((double) lg[i] - (double) mx);
        double cum = 0.0;
        int cut = k;
        for (int i = 0; i < k; ++i) {
            cum += std::exp((double) lg[i] - (double) mx) / sum;
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
    const float inv_t = 1.0f / p.temperature;
    float smx = lg[0] * inv_t;
    for (int i = 1; i < n_keep; ++i) smx = std::fmax(smx, lg[i] * inv_t);
    double sum = 0.0;
    for (int i = 0; i < n_keep; ++i) sum += std::exp((double) (lg[i] * inv_t) - (double) smx);
    int pick = L.id[n_keep > 0 ? n_keep - 1 : 0];
    const float u = philox_uniform(p.seed, counter);
    double cum = 0.0;
    for (int i = 0; i < n_keep; ++i) {
        cum += std::exp((double) (lg[i] * inv_t) - (double) smx) / sum;
        if ((double) u < cum) { pick = L.id[i]; break; }
    }
    return pick;
}

}  // namespace strata::metal_engine
