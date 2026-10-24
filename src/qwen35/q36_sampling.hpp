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

/// The tie rule of the top-k on the GPU shortlist (k_topk_cand) and, with --mtp-sampled or --sample-gpu (tie_by_id()), on the host
/// too: the higher logit first, then the lower id.  Without those options the host's top-k keeps the comparator it always had
/// (lg[a] > lg[b]; std::partial_sort leaves equal logits in an unspecified order), so a sampled request's text is unchanged.
inline bool before(float a, int ia, float b, int ib) { return a > b || (a == b && ia < ib); }
/// A process-wide switch, written once in run() before any request (the solo, batch and serve paths all read it); tests set it themselves.
inline bool& tie_by_id() { static bool on = false; return on; }

inline bool penalised(const Sampling& s) {
    return s.last_n != 0 && (s.repeat != 1.f || s.freq != 0.f || s.present != 0.f);
}
/// the counts of the ids among the last last_n tokens of hist
inline void penalty_counts(const Sampling& s, const std::vector<int32_t>& hist, std::map<int, int>& cnt) {
    const size_t from = s.last_n < 0 || (size_t) s.last_n >= hist.size() ? 0 : hist.size() - (size_t) s.last_n;
    for (size_t i = from; i < hist.size(); ++i) ++cnt[hist[i]];
}
/// one id's penalty (the same float operations wherever it runs)
inline void penalise(float& v, int count, const Sampling& s) {
    if (s.repeat != 1.f) v = v > 0 ? v / s.repeat : v * s.repeat;
    v -= (float) count * s.freq + s.present;
}

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
/// the chain after the top_k: c.idx the kept ids (best first, the tie rule), vals[i] the penalised logit of c.idx[i]
inline void chain_tail(const std::vector<float>& vals, const Sampling& s, Chain& c) {
    std::vector<int>& idx = c.idx;
    std::vector<double>& p = c.p;
    p.resize(idx.size());
    const double mx = vals[0];
    double sum = 0;
    for (size_t i = 0; i < idx.size(); ++i) sum += (p[i] = std::exp((vals[i] - mx) / s.temperature));
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
}
inline int chain(std::vector<float>& lg, const Sampling& s, const std::vector<int32_t>& hist, Chain& c) {
    if (penalised(s)) {
        std::map<int, int> cnt;
        penalty_counts(s, hist, cnt);
        for (const auto& [id, n] : cnt) {
            if (id < 0 || (size_t) id >= lg.size()) continue;
            penalise(lg[(size_t) id], n, s);
        }
    }
    if (s.temperature <= 0.f) return (int) (std::max_element(lg.begin(), lg.end()) - lg.begin());
    const int kk = std::max(1, std::min(s.top_k > 0 ? s.top_k : 64, (int) lg.size()));
    std::vector<int>& idx = c.idx;
    idx.resize(lg.size());
    for (size_t i = 0; i < idx.size(); ++i) idx[i] = (int) i;
    if (tie_by_id())
        std::partial_sort(idx.begin(), idx.begin() + kk, idx.end(),
                          [&](int a, int b) { return before(lg[(size_t) a], a, lg[(size_t) b], b); });
    else
        std::partial_sort(idx.begin(), idx.begin() + kk, idx.end(), [&](int a, int b) { return lg[a] > lg[b]; });
    idx.resize((size_t) kk);
    std::vector<float> vals(idx.size());
    for (size_t i = 0; i < idx.size(); ++i) vals[i] = lg[(size_t) idx[i]];
    chain_tail(vals, s, c);
    return -1;
}

/// SHORTLIST (--sample-gpu): the GPU hands back the K highest RAW logits of a row (k_cand_part / k_cand_merge, the tie rule, any order)
/// instead of the 1 MB row.  Penalties here only lower a logit (repeat >= 1, freq >= 0, present >= 0), so with
/// K = top_k + the number of penalised ids no outside id can reach the penalised top_k: the shortlist holds the same
/// top_k as the whole row, and chain_c gives chain()'s bits.
struct Cand {
    std::vector<int32_t> id;
    std::vector<float> v;
};
/// K for a row whose history is hist followed by `extra` (the drafts before it, or a window's drafts): 0 when the
/// shortlist cannot be used (a penalty that raises a logit, an unbounded window, K past kMax) and the row is copied whole.
constexpr int kCandMax = 160;
/// the GPU kernels split a row in kCandBlocks slices of at most kCandSlice logits, each holding at least K
constexpr int kCandBlocks = 32, kCandSlice = 8192;
/// a row of n logits is cut in slices of ceil(n / 32); every slice, the last (shorter) one too, must hold K of them
inline bool cand_rows_ok(size_t n, int K) {
    const size_t S = (n + kCandBlocks - 1) / kCandBlocks;
    return n <= (size_t) kCandBlocks * kCandSlice && n > (size_t) (kCandBlocks - 1) * S && n - (size_t) (kCandBlocks - 1) * S >= (size_t) K;
}
inline int cand_need(const Sampling& s, const std::vector<int32_t>& hist, const int32_t* extra, int n_extra, size_t n_row) {
    int kk = s.temperature <= 0.f ? 1 : std::max(1, std::min(s.top_k > 0 ? s.top_k : 64, (int) n_row));
    int P = 0;
    if (penalised(s)) {
        if (!(s.repeat >= 1.f) || !(s.freq >= 0.f) || !(s.present >= 0.f) || s.last_n < 0) return 0;   // (NaN too)
        std::map<int, int> seen;
        const size_t want = (size_t) s.last_n + (size_t) n_extra;
        const size_t tot = hist.size() + (size_t) n_extra;
        for (size_t i = tot > want ? tot - want : 0; i < tot; ++i)
            seen[i < hist.size() ? hist[i] : extra[i - hist.size()]] = 1;
        P = (int) seen.size();
    }
    const int K = kk + P;
    return K > kCandMax || !cand_rows_ok(n_row, K) ? 0 : K;
}
inline int chain_c(Cand& cd, size_t n_row, const Sampling& s, const std::vector<int32_t>& hist, Chain& c) {
    if (penalised(s)) {
        std::map<int, int> cnt;
        penalty_counts(s, hist, cnt);
        for (const auto& [id, n] : cnt)
            for (size_t j = 0; j < cd.id.size(); ++j)
                if (cd.id[j] == id) { penalise(cd.v[j], n, s); break; }
    }
    std::vector<int>& idx = c.idx;
    idx.resize(cd.id.size());
    for (size_t j = 0; j < idx.size(); ++j) idx[j] = (int) j;
    const auto better = [&](int a, int b) { return before(cd.v[(size_t) a], cd.id[(size_t) a], cd.v[(size_t) b], cd.id[(size_t) b]); };
    if (s.temperature <= 0.f) return cd.id[(size_t) *std::min_element(idx.begin(), idx.end(), better)];
    const int kk = std::max(1, std::min(s.top_k > 0 ? s.top_k : 64, (int) n_row));
    std::partial_sort(idx.begin(), idx.begin() + kk, idx.end(), better);
    idx.resize((size_t) kk);
    std::vector<float> vals(idx.size());
    for (size_t i = 0; i < idx.size(); ++i) {
        vals[i] = cd.v[(size_t) idx[i]];
        idx[i] = cd.id[(size_t) idx[i]];
    }
    chain_tail(vals, s, c);
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
inline int sample_c(Cand& cd, size_t n_row, const Sampling& s, const std::vector<int32_t>& hist, std::mt19937_64& rng) {
    Chain c;
    const int g = chain_c(cd, n_row, s, hist, c);
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
        return draft_pick(d, g, prob);
    }
    /// draft() from a shortlist of the block's row (n_row ids wide)
    int32_t draft_c(int d, const int32_t* prev, Cand& cd, size_t n_row, float* prob) {
        const size_t h0 = hist.size();
        hist.insert(hist.end(), prev, prev + d);
        const int g = chain_c(cd, n_row, s, hist, c);
        hist.resize(h0);
        return draft_pick(d, g, prob);
    }
    /// K of draft d's shortlist (0: copy the whole row), and of a window whose drafts are `drafts`
    int draft_k(int d, const int32_t* prev, size_t n_row) const { return cand_need(s, hist, prev, d, n_row); }
    int window_k(const std::vector<int32_t>& drafts, size_t n_row) const {
        return cand_need(s, hist, drafts.data(), (int) drafts.size(), n_row);
    }
    int32_t draft_pick(int d, int g, float* prob) {
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
        return judge_rows([&](int a) { row(a, lg); return chain(lg, s, hist, c); },
                          [&](int a) { row(a, lg); return sample(lg, s, hist, rng); }, D, drafts, picks);
    }
    /// judge() over shortlists: row(r, cd) fills row r's (--sample-gpu); n_row the width of the target's row
    template <class Row> int judge_c(Row&& row, size_t n_row, int D, const std::vector<int32_t>& drafts, int32_t* picks) {
        return judge_rows([&](int a) { row(a, cd); return chain_c(cd, n_row, s, hist, c); },
                          [&](int a) { row(a, cd); return sample_c(cd, n_row, s, hist, rng); }, D, drafts, picks);
    }
    Cand cd;
    /// chn(a): row a's chain (fills c; returns the greedy pick or -1); smp(a): row a's plain sample
    template <class Chn, class Smp>
    int judge_rows(Chn&& chn, Smp&& smp, int D, const std::vector<int32_t>& drafts, int32_t* picks) {
        const size_t h0 = hist.size();
        int a = 0;
        for (;; ++a) {
            if (rule == SpecRule::match || a == D) {
                picks[a] = smp(a);
                if (a == D || picks[a] != drafts[(size_t) a]) break;
            } else {
                const int g = chn(a);
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
