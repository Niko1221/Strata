// metal/tests/engine/host_sampler_parity.cpp - the Mac's host sampler (src/engine/host_sampler.cpp) against a serial
// transcription of the CUDA sampler, draw for draw, and its time per row.  CPU only: no GPU, no model.
//
//   host_sampler_parity [--bench]        exit 0 = every pick equals the reference's
//
// The reference below is src/kernels/sampler_parity.cpp's `mirror_select` + `mirror_pick` (the CUDA kernels'
// semantics, which that test pins on the GPU), with the logit_bias added before the penalties as the kernels do, and
// the greedy pick of `sampler_greedy_kernel` (argmax after bias and penalties, ties to the lowest id, 0 when every
// value is -inf or NaN).  Rows: 248,320 logits (the model's vocabulary) with planted ties at the top, -inf and NaN
// entries, a logit_bias with bans, and a penalty history; configurations from greedy to top_k 64 with top_p and min_p.
#include "../../src/engine/host_sampler.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <limits>
#include <random>
#include <vector>

using strata::kernels::SamplerParams;

namespace {

float ref_philox(uint64_t seed, uint64_t counter) {   // sampler_parity.cpp's host_philox_uniform
    uint32_t c0 = (uint32_t) counter, c1 = (uint32_t) (counter >> 32);
    uint32_t c2 = (uint32_t) seed, c3 = (uint32_t) (seed >> 32);
    for (uint32_t i = 0; i < 10; ++i) {
        const uint32_t hi0 = (uint32_t) (((uint64_t) 0x9E3779B9u * c0) >> 32);
        const uint32_t hi1 = (uint32_t) (((uint64_t) 0xBB67AE85u * c2) >> 32);
        const uint32_t lo0 = 0x9E3779B9u * c0;
        const uint32_t lo1 = 0xBB67AE85u * c2;
        const uint32_t n0 = hi1 ^ c1 ^ i, n1 = lo1, n2 = hi0 ^ c3 ^ 0u, n3 = lo0;
        c0 = n0; c1 = n1; c2 = n2; c3 = n3;
    }
    return (float) (c0 >> 8) * (1.0f / 16777216.0f);
}

// the scores the kernels rank: logit (+ bias), then the penalties over the counted window
std::vector<float> ref_scores(const float* l, int nv, const float* bias, const std::vector<int>& window,
                              const SamplerParams& p) {
    std::vector<float> s((size_t) nv);
    for (int v = 0; v < nv; ++v) s[(size_t) v] = bias ? l[v] + bias[v] : l[v];
    std::vector<int> cnt((size_t) nv, 0);
    for (int h : window) if (h >= 0 && h < nv) ++cnt[(size_t) h];
    for (int v = 0; v < nv; ++v) {
        const int c = cnt[(size_t) v];
        if (c <= 0) continue;
        float x = s[(size_t) v];
        if (x <= 0.0f) x *= p.penalty_repeat; else x /= p.penalty_repeat;
        x -= (float) c * p.penalty_freq + (c > 0 ? 1.0f : 0.0f) * p.penalty_present;
        s[(size_t) v] = x;
    }
    return s;
}

int ref_pick(const float* l, int nv, const float* bias, const std::vector<int>& window, const SamplerParams& p,
             uint64_t counter) {
    const std::vector<float> s = ref_scores(l, nv, bias, window, p);
    if (p.greedy || p.temperature <= 0.0f) {   // sampler_greedy_kernel
        float bv = -std::numeric_limits<float>::infinity();
        int best = nv;
        for (int v = 0; v < nv; ++v) if (s[(size_t) v] > bv) { bv = s[(size_t) v]; best = v; }
        return best < nv ? best : 0;
    }
    const int k = std::min(nv, (p.top_k > 0 && p.top_k < 64) ? p.top_k : 64);
    std::vector<int> ids;   // mirror_select: k argmax rounds over the not-yet-taken
    std::vector<float> lg;
    std::vector<char> taken((size_t) nv, 0);
    for (int i = 0; i < k; ++i) {
        int best = nv;
        float bv = -std::numeric_limits<float>::infinity();
        for (int v = 0; v < nv; ++v)
            if (!taken[(size_t) v] && s[(size_t) v] > bv) { bv = s[(size_t) v]; best = v; }
        const int id = best < nv ? best : 0;
        taken[(size_t) id] = 1;
        ids.push_back(id);
        lg.push_back(bv);
    }
    // mirror_pick
    int n_keep = k;
    float mx = lg[0];
    for (int i = 1; i < k; ++i) mx = std::fmax(mx, lg[(size_t) i]);
    if (p.top_p < 1.0f) {
        double sum = 0.0;
        for (int i = 0; i < k; ++i) sum += std::exp((double) lg[(size_t) i] - (double) mx);
        double cum = 0.0;
        int cut = k;
        for (int i = 0; i < k; ++i) {
            cum += std::exp((double) lg[(size_t) i] - (double) mx) / sum;
            if (cum >= (double) p.top_p) { cut = i + 1; break; }
        }
        if (cut < p.min_keep) cut = p.min_keep < k ? p.min_keep : k;
        n_keep = cut;
    }
    if (p.min_p > 0.0f) {
        const float thresh = lg[0] + std::log(p.min_p);
        for (int i = 0; i < n_keep; ++i)
            if (lg[(size_t) i] < thresh) { n_keep = i; break; }
    }
    const float inv_t = p.temperature > 0.0f ? 1.0f / p.temperature : 0.0f;
    auto scaled = [&](int i) { return lg[(size_t) i] * inv_t; };
    float smx = scaled(0);
    for (int i = 1; i < n_keep; ++i) smx = std::fmax(smx, scaled(i));
    double sum = 0.0;
    for (int i = 0; i < n_keep; ++i) sum += std::exp((double) scaled(i) - (double) smx);
    int pick = ids[(size_t) (n_keep > 0 ? n_keep - 1 : 0)];
    const float u = ref_philox(p.seed, counter);
    double cum = 0.0;
    for (int i = 0; i < n_keep; ++i) {
        cum += std::exp((double) scaled(i) - (double) smx) / sum;
        if ((double) u < cum) { pick = ids[(size_t) i]; break; }
    }
    return pick;
}

// the last min(penalty_last_n, len) entries of a history row (the kernels' window)
std::vector<int> window_of(const std::vector<int32_t>& h, const SamplerParams& p) {
    const int n = std::min<int>(p.penalty_last_n, (int) h.size());
    return n > 0 ? std::vector<int>(h.end() - n, h.end()) : std::vector<int>{};
}

}  // namespace

int main(int argc, char** argv) {
    const bool bench = argc > 1 && std::strcmp(argv[1], "--bench") == 0;
    constexpr int NV = 248320;
    std::mt19937_64 rng(11);
    std::normal_distribution<float> nd(0.0f, 2.5f);
    int bad = 0, checks = 0;

    // Philox: the engine's draw is the reference's
    for (uint64_t c = 0; c < 4096; ++c) {
        const uint64_t seed = rng();
        if (strata::metal_engine::philox_uniform(seed, c * 977) != ref_philox(seed, c * 977)) ++bad;
    }
    std::printf("  philox: %s\n", bad ? "*** WRONG ***" : "matches 4096 draws");

    struct Cfg { const char* name; SamplerParams p; bool bias, pen; };
    auto mk = [](float t, int k, float tp, float mp, int pen_n, float rep, float fr, float pr) {
        SamplerParams p;
        p.greedy = t <= 0.0f; p.temperature = t; p.top_k = k; p.top_p = tp; p.min_p = mp;
        p.penalty_last_n = pen_n; p.penalty_repeat = rep; p.penalty_freq = fr; p.penalty_present = pr;
        return p;
    };
    const Cfg cfgs[] = {
        {"greedy + logit_bias", mk(0.0f, 20, 1.0f, 0.0f, 0, 1.0f, 0.0f, 0.0f), true, false},
        {"greedy + penalties", mk(0.0f, 20, 1.0f, 0.0f, 64, 1.3f, 0.0f, 0.0f), false, true},
        {"t0.7 k20 p0.95", mk(0.7f, 20, 0.95f, 0.0f, 0, 1.0f, 0.0f, 0.0f), false, false},
        {"t1.0 k40 p1.0", mk(1.0f, 40, 1.0f, 0.0f, 0, 1.0f, 0.0f, 0.0f), false, false},
        {"t1.0 k0 (64) p0.9 min_p0.05", mk(1.0f, 0, 0.9f, 0.05f, 0, 1.0f, 0.0f, 0.0f), false, false},
        {"t0.6 k1", mk(0.6f, 1, 0.95f, 0.0f, 0, 1.0f, 0.0f, 0.0f), false, false},
        {"t1.2 k64 + bias + penalties", mk(1.2f, 64, 0.95f, 0.02f, 256, 1.5f, 0.1f, 0.2f), true, true},
        {"t3.0 k50 p1.0 (flat)", mk(3.0f, 50, 1.0f, 0.0f, 0, 1.0f, 0.0f, 0.0f), false, false},
    };
    std::vector<float> l((size_t) NV), bias((size_t) NV);
    std::vector<int32_t> hist(512);
    for (int trial = 0; trial < 12; ++trial) {
        for (float& x : l) x = nd(rng);
        // a head the chain cuts into: a few strong logits, two ties at the top, some -inf and a NaN
        for (int j = 0; j < 30; ++j) l[(size_t) (rng() % NV)] = 9.0f + (float) (rng() % 40) * 0.1f;
        const int a = (int) (rng() % NV), b = (int) (rng() % NV);
        l[(size_t) a] = l[(size_t) b] = 14.0f;
        for (int j = 0; j < 50; ++j) l[(size_t) (rng() % NV)] = -std::numeric_limits<float>::infinity();
        l[(size_t) (rng() % NV)] = std::numeric_limits<float>::quiet_NaN();
        std::fill(bias.begin(), bias.end(), 0.0f);
        bias[(size_t) std::min(a, b)] = -std::numeric_limits<float>::infinity();   // ban one of the tie
        for (int j = 0; j < 20; ++j) bias[(size_t) (rng() % NV)] = (float) ((int) (rng() % 40) - 20) * 0.5f;
        for (size_t j = 0; j < hist.size(); ++j) hist[j] = j < 20 ? -1 : (int32_t) (rng() % 64 == 0 ? a : rng() % NV);
        hist[hist.size() - 1] = b;   // the other of the tie is penalised
        for (const Cfg& c : cfgs) {
            int wrong = 0;
            for (int d = 0; d < 6; ++d) {
                SamplerParams p = c.p;
                p.seed = rng();
                const uint64_t counter = rng() % 300000;
                const float* bp = c.bias ? bias.data() : nullptr;
                const int32_t* hp = c.pen ? hist.data() : nullptr;
                const int got = strata::metal_engine::sample_row(l.data(), NV, hp, c.pen ? (int) hist.size() : 0, p, bp,
                                                                 counter);
                const int want = ref_pick(l.data(), NV, bp, c.pen ? window_of(hist, p) : std::vector<int>{}, p, counter);
                wrong += got != want;
                ++checks;
            }
            if (wrong) std::printf("  trial %d %-30s *** %d WRONG ***\n", trial, c.name, wrong);
            bad += wrong;
        }
    }
    // edge rows: all -inf (greedy and sampled answer 0), fewer finite logits than top_k
    {
        std::vector<float> e((size_t) NV, -std::numeric_limits<float>::infinity());
        for (const Cfg& c : cfgs) {
            SamplerParams p = c.p;
            p.seed = 5;
            bad += strata::metal_engine::sample_row(e.data(), NV, nullptr, 0, p, nullptr, 9) != ref_pick(e.data(), NV, nullptr, {}, p, 9);
            e[100] = 1.0f; e[7] = 1.0f; e[200000] = 3.0f;
            bad += strata::metal_engine::sample_row(e.data(), NV, nullptr, 0, p, nullptr, 9) != ref_pick(e.data(), NV, nullptr, {}, p, 9);
            checks += 2;
        }
    }
    std::printf("host_sampler_parity: %d picks, %s\n", checks, bad ? "*** WRONG ***" : "all equal the reference");

    if (bench) {   // time per row on a model-sized row (the window's head: 248,320 logits)
        for (float& x : l) x = nd(rng);
        for (const Cfg& c : cfgs) {
            SamplerParams p = c.p;
            const int R = 400;
            volatile int sink = 0;
            const auto t0 = std::chrono::steady_clock::now();
            for (int r = 0; r < R; ++r) {
                p.seed = (uint64_t) r;
                sink = sink + strata::metal_engine::sample_row(l.data(), NV, c.pen ? hist.data() : nullptr,
                                                               c.pen ? (int) hist.size() : 0, p, c.bias ? bias.data() : nullptr,
                                                               (uint64_t) r);
            }
            const double us = std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() - t0).count() / R;
            std::printf("  bench %-30s %7.1f us/row\n", c.name, us);
        }
    }
    return bad ? 1 : 0;
}
