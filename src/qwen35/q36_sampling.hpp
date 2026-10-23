// src/qwen35/q36_sampling.hpp - the q36 engine's sampler (host): the chain, the draw, and --mtp-sampled's judge of a
// sampled request's speculative step.  Header-only so that q36_sampling_test.cpp checks it without a GPU.
#pragma once

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <map>
#include <random>
#include <vector>

#include "strata/core/spec_prob.hpp"

namespace strata::qwen35::sampling {

struct Sampling {
    float temperature = 0.f, top_p = 1.f, min_p = 0.f, repeat = 1.f, freq = 0.f, present = 0.f;
    int top_k = 20, last_n = 0;
    unsigned long long seed = 0;
    bool needs_logits() const { return temperature > 0.f || repeat != 1.f || freq != 0.f || present != 0.f; }
};

/// The sampling chain up to the draw: penalties (in place on lg, over the last last_n tokens of hist; an id past lg's
/// end, a history token outside a --draft-vocab row, is not in the row and takes none), top_k, the softmax at the
/// temperature, min_p, top_p.  The kept ids are c.idx[0 .. c.keep), their unnormalised weights c.p,
/// c.tot their sum.  Greedy (temperature <= 0): returns the (penalised) argmax and fills nothing, else -1.
struct Chain {
    std::vector<int> idx;
    std::vector<double> p;
    size_t keep = 0;
    double tot = 0;
};
inline int chain(std::vector<float>& lg, const Sampling& s, const std::vector<int32_t>& hist, Chain& c) {
    if (s.last_n != 0 && (s.repeat != 1.f || s.freq != 0.f || s.present != 0.f)) {
        std::map<int, int> cnt;
        const size_t from = s.last_n < 0 || (size_t) s.last_n >= hist.size() ? 0 : hist.size() - (size_t) s.last_n;
        for (size_t i = from; i < hist.size(); ++i) ++cnt[hist[i]];
        for (const auto& [id, c] : cnt) {
            if (id < 0 || (size_t) id >= lg.size()) continue;
            float& v = lg[(size_t) id];
            if (s.repeat != 1.f) v = v > 0 ? v / s.repeat : v * s.repeat;
            v -= (float) c * s.freq + s.present;
        }
    }
    if (s.temperature <= 0.f) return (int) (std::max_element(lg.begin(), lg.end()) - lg.begin());
    const int kk = std::max(1, std::min(s.top_k > 0 ? s.top_k : 64, (int) lg.size()));
    std::vector<int>& idx = c.idx;
    idx.resize(lg.size());
    for (size_t i = 0; i < idx.size(); ++i) idx[i] = (int) i;
    std::partial_sort(idx.begin(), idx.begin() + kk, idx.end(), [&](int a, int b) { return lg[a] > lg[b]; });
    idx.resize((size_t) kk);
    std::vector<double>& p = c.p;
    p.resize(idx.size());
    const double mx = lg[(size_t) idx[0]];
    double sum = 0;
    for (size_t i = 0; i < idx.size(); ++i) sum += (p[i] = std::exp((lg[(size_t) idx[i]] - mx) / s.temperature));
    size_t keep = idx.size();
    double cum = 0;
    for (size_t i = 0; i < idx.size(); ++i) {
        if (p[i] / sum < s.min_p * (p[0] / sum)) { keep = i; break; }
        cum += p[i] / sum;
        if (cum >= s.top_p) { keep = i + 1; break; }
    }
    keep = std::max<size_t>(1, keep);
    double tot = 0;
    for (size_t i = 0; i < keep; ++i) tot += p[i];
    c.keep = keep;
    c.tot = tot;
    return -1;
}

/// The draw from a chain's kept list: one uniform from rng.
inline int draw(const Chain& c, std::mt19937_64& rng) {
    std::uniform_real_distribution<double> u(0.0, c.tot);
    double r = u(rng);
    for (size_t i = 0; i < c.keep; ++i) {
        if ((r -= c.p[i]) <= 0) return c.idx[i];
    }
    return c.idx[c.keep - 1];
}

inline int sample(std::vector<float>& lg, const Sampling& s, const std::vector<int32_t>& hist, std::mt19937_64& rng) {
    Chain c;
    const int g = chain(lg, s, hist, c);
    return g >= 0 ? g : draw(c, rng);
}

/// --mtp-sampled: the MTP drafts of a SAMPLED request (agent clients run the model at the model card's temperature
/// 1.0, so the greedy-only speculative step never ran for them).  Two rules, both giving every token the target chain's
/// distribution (penalties over the history WITH the drafts kept before it, top_k, top_p, min_p, temperature):
///   match  the draft is the block's greedy pick (a point mass); each window row samples its token from the target
///          chain exactly as a plain decode step would (sample(), one draw of the request's rng) and the window keeps
///          going while that token equals the draft.  That is rejection sampling for a point-mass q: the draft is kept
///          with probability p(d), and a reject's token is p without d, renormalised.  Rows are sampled lazily (none
///          after the first reject), so the rng gives one draw per token in order, as without drafts: a fixed seed
///          gives the same text as --mtp 0 until a near-tie of a draw falls differently because a window row's logits
///          differ in the last bits from a decode step's.
///   q      the draft is SAMPLED from q, the same chain over the block's logits (spec_prob.hpp's rule): kept with
///          probability min(1, p(d) / q(d)), a reject's token from norm(max(0, p - q)), the window's last row a plain
///          sample.  The rng gives the drafts' draws, then per row an accept and a residual draw: a fixed seed repeats
///          itself (same build, same --mtp N), but its text is not --mtp 0's.
enum class SpecRule { off, match, q };

/// a uniform in [0, 1) from 24 bits of rng (spec_prob.hpp's rule takes floats)
inline float u01(std::mt19937_64& rng) { return (float) (rng() >> 40) * 0x1p-24f; }

/// The kept list of a chain as spec_prob.hpp's normalised (ids, probabilities).
inline void chain_list(const Chain& c, std::vector<int32_t>& ids, std::vector<double>& pr) {
    ids.resize(c.keep);
    pr.resize(c.keep);
    for (size_t i = 0; i < c.keep; ++i) {
        ids[i] = c.idx[i];
        pr[i] = c.p[i] / c.tot;
    }
}

/// One request's sampled speculative steps: the hooks for Engine::spec_step over the request's sampling, history
/// (the tokens so far, the last one x) and rng.
struct SpecJudge {
    SpecJudge(SpecRule r, const Sampling& smp, std::vector<int32_t>& h, std::mt19937_64& g)
        : rule(r), s(smp), hist(h), rng(g) {}
    SpecRule rule;
    const Sampling& s;
    std::vector<int32_t>& hist;
    std::mt19937_64& rng;
    std::vector<float> lg;
    Chain c;
    std::vector<int32_t> pid;
    std::vector<double> pp;
    std::vector<int32_t> qid[8];   // the drafts' q lists (q: kVerify - 1 drafts at most)
    std::vector<float> qp[8];

    /// q: draft d sampled from the chain over the block's logits row (history: hist and the drafts before d).  *prob: what
    /// --spec-min-p gates on, q's TOP probability.  Not the drawn token's q (the main path's default gate, spec_prob.hpp's
    /// spec_gate_pick): a draft kept only when q(d) >= p_min is drawn from q cut to those tokens, not from q, and
    /// min(1, p/q) with the uncut q no longer gives p (q36_sampling_test's control shows it).  q's top depends on the
    /// tokens before d only, so the gate is a choice of window length and the rule stays exact.
    bool gate_pick = false;   // (the biased gate: q36_sampling_test's control only)
    int32_t draft(int d, const int32_t* prev, const float* row, int n, float* prob) {
        lg.assign(row, row + n);
        const size_t h0 = hist.size();
        hist.insert(hist.end(), prev, prev + d);
        const int g = chain(lg, s, hist, c);
        hist.resize(h0);
        if (g >= 0) {   // (not reached: q runs for temperature > 0 only)
            qid[d].assign(1, g);
            qp[d].assign(1, 1.f);
            *prob = 1.f;
            return g;
        }
        qid[d].resize(c.keep);
        qp[d].resize(c.keep);
        for (size_t i = 0; i < c.keep; ++i) {
            qid[d][i] = c.idx[i];
            qp[d][i] = (float) (c.p[i] / c.tot);
        }
        // the draw over q as the list says (the float probabilities, so the q(d) the judge divides by is the one drawn with)
        const float u = u01(rng);
        float cum = 0.f;
        size_t at = c.keep - 1;
        for (size_t i = 0; i < c.keep; ++i) {
            cum += qp[d][i];
            if (u < cum) { at = i; break; }
        }
        *prob = gate_pick ? qp[d][at] : qp[d][0];
        return qid[d][at];
    }

    /// the window's rows in order, each judged under the history with the drafts kept before it; returns a.
    /// row(r, lg): the target's logits of window row r (the engine's host_logits_row).
    template <class Row> int judge(Row&& row, int D, const std::vector<int32_t>& drafts, int32_t* picks) {
        const size_t h0 = hist.size();
        int a = 0;
        for (;; ++a) {
            row(a, lg);
            if (rule == SpecRule::match || a == D) {
                picks[a] = sample(lg, s, hist, rng);
                if (a == D || picks[a] != drafts[(size_t) a]) break;
            } else {
                const int g = chain(lg, s, hist, c);
                if (g >= 0) {   // (greedy with penalties: the rule is the compare)
                    picks[a] = g;
                    if (g != drafts[(size_t) a]) break;
                } else {
                    chain_list(c, pid, pp);
                    const strata::core::SpecP P{pid.data(), pp.data(), (int) pid.size()};
                    const strata::core::SpecQ Q{qid[a].data(), qp[a].data(), (int) qid[a].size()};
                    const float ua = u01(rng), ur = u01(rng);
                    const strata::core::SpecResult r = strata::core::spec_verify_row(P, drafts[(size_t) a], Q, ua, ur);
                    picks[a] = r.token;
                    if (!r.accepted) break;
                }
            }
            hist.push_back(picks[a]);
        }
        hist.resize(h0);
        return a;
    }
};

}  // namespace strata::qwen35::sampling
