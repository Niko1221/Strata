// src/qwen35/q36_sampling_test.cpp - --mtp-sampled's judge (src/qwen35/q36_sampling.hpp), no GPU.
//
// A toy model stands in for the engine: the target's logits are a fixed random table over the last two tokens of
// the history (an 8-token vocabulary), the MTP block's are the target's plus noise (a drafter that is often but not
// always right).  The engine's decode loop is replayed on it: a token x, D drafts (the block's greedy pick for
// match, SpecJudge::draft for q), the judge over the window's rows (row r = the target on the history with x and
// drafts 0 .. r - 1), the kept drafts and the judge's last pick emitted.  The sampling chain has every part on
// (presence and repetition penalties over the history, top_k, top_p, min_p, temperature), so a judge that forgot a
// kept draft in the penalties' history would show.
//
//   1. match, the same seed: the speculative text equals the plain sampler's text token for token (the rng gives one
//      draw per token in order).
//   2. q and match: the joint of the first four tokens over 400,000 seeds against the exact distribution (the chain's
//      probabilities multiplied out over every 4-token continuation), chi-square at a 99.9% bound.
//   3. two controls must FAIL test 2: the judge handed a wrong q (uniform over q's list), and a target whose window
//      rows ignore the drafts before them.
//   4. --draft-vocab: the block's row covers the first ids only, the history (and the penalties) every id: a history id
//      past the row takes no penalty there (libstdc++'s assertions, on below, abort an access past the row), and q over
//      the short row still gives the target's joint.
#define _GLIBCXX_ASSERTIONS 1
#include "q36_sampling.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <functional>
#include <map>
#include <random>
#include <vector>

using namespace strata::qwen35::sampling;

namespace {

constexpr int V = 8, L = 4;
int g_fail = 0;

std::vector<float> g_tab, g_noise;   // [V][V][V]: logits by (second last, last) token

std::vector<float> target(const std::vector<int32_t>& h) {
    const size_t o = ((size_t) h[h.size() - 2] * V + (size_t) h.back()) * V;
    return {g_tab.begin() + (long) o, g_tab.begin() + (long) (o + V)};
}
std::vector<float> drafter(const std::vector<int32_t>& h) {
    std::vector<float> t = target(h);
    const size_t o = ((size_t) h[h.size() - 2] * V + (size_t) h.back()) * V;
    for (int i = 0; i < V; ++i) t[(size_t) i] += g_noise[o + (size_t) i];
    return t;
}

const std::vector<int32_t> kPrompt = {3, 5, 3};

/// the plain sampler: L tokens
std::vector<int32_t> plain(const Sampling& s, unsigned long long seed) {
    std::mt19937_64 rng(seed);
    std::vector<int32_t> h = kPrompt, out;
    while ((int) out.size() < L) {
        std::vector<float> lg = target(h);
        const int t = sample(lg, s, h, rng);
        out.push_back(t);
        h.push_back(t);
    }
    return out;
}

enum class Control { none, wrong_q, blind_rows, gate_pick };
float g_min_p = 0.f;   // --spec-min-p for spec(): a draft below it ends the step's drafts
int g_draft_v = V;     // --draft-vocab for spec(): the block's row holds ids 0 .. g_draft_v - 1

/// the engine's loop (Serve::request with --mtp-sampled): L tokens, D drafts per step
std::vector<int32_t> spec(const Sampling& s, unsigned long long seed, SpecRule rule, int D, Control ctl,
                          long long& offered, long long& accepted) {
    std::mt19937_64 rng(seed);
    std::vector<int32_t> hist = kPrompt, out;
    SpecJudge sj{rule, s, hist, rng};
    sj.gate_pick = ctl == Control::gate_pick;
    std::vector<float> lg = target(hist);
    int nx = sample(lg, s, hist, rng);
    for (;;) {
        out.push_back(nx);
        hist.push_back(nx);
        if ((int) out.size() >= L) break;
        std::vector<int32_t> drafts((size_t) D, 0);
        int Dn = D;
        for (int d = 0; d < Dn; ++d) {
            std::vector<int32_t> h = hist;
            h.insert(h.end(), drafts.begin(), drafts.begin() + d);
            std::vector<float> row = drafter(h);
            row.resize((size_t) g_draft_v);
            float pr;
            if (rule == SpecRule::q) {
                drafts[(size_t) d] = sj.draft(d, drafts.data(), row.data(), (int) row.size(), &pr);
                if (ctl == Control::wrong_q)
                    for (float& q : sj.qp[d]) q = 1.f / (float) sj.qp[d].size();
            } else {   // the engine's k_pick_prob: the pick's softmax probability over the raw logits
                const int g = (int) (std::max_element(row.begin(), row.end()) - row.begin());
                double z = 0;
                for (float v : row) z += std::exp((double) v - row[(size_t) g]);
                drafts[(size_t) d] = g;
                pr = (float) (1.0 / z);
            }
            if (pr < g_min_p) Dn = d;   // the engine's gate (spec_step)
        }
        drafts.resize((size_t) Dn);
        int32_t picks[16];
        const size_t h0 = hist.size();
        auto row = [&](int r, std::vector<float>& o) {
            std::vector<int32_t> h(hist.begin(), hist.begin() + (long) h0);
            if (ctl != Control::blind_rows) h.insert(h.end(), drafts.begin(), drafts.begin() + r);
            o = target(h);
        };
        const int a = sj.judge(row, Dn, drafts, picks);
        offered += Dn;
        accepted += a;
        bool ended = false;
        for (int i = 0; i < a && !ended; ++i) {
            out.push_back(drafts[(size_t) i]);
            hist.push_back(drafts[(size_t) i]);
            ended = (int) out.size() >= L;
        }
        if (ended) break;
        nx = picks[a];
    }
    out.resize((size_t) L);
    return out;
}

/// the exact probability of every L-token continuation (index: base-V digits)
void exact(const Sampling& s, std::vector<int32_t>& h, int depth, double pr, size_t code, std::vector<double>& P) {
    if (depth == L) { P[code] += pr; return; }
    std::vector<float> lg = target(h);
    Chain c;
    const int g = chain(lg, s, h, c);
    if (g >= 0) {
        h.push_back(g);
        exact(s, h, depth + 1, pr, code * V + (size_t) g, P);
        h.pop_back();
        return;
    }
    for (size_t i = 0; i < c.keep; ++i) {
        h.push_back(c.idx[i]);
        exact(s, h, depth + 1, pr * c.p[i] / c.tot, code * V + (size_t) c.idx[i], P);
        h.pop_back();
    }
}

/// chi-square of counts against probabilities P (cells under 5 expected pooled); returns z (Wilson-Hilferty)
double chi_z(const std::vector<long long>& cnt, const std::vector<double>& P, long long n, double& stat, int& dof) {
    stat = 0;
    dof = -1;
    double pool_e = 0, pool_o = 0;
    for (size_t i = 0; i < P.size(); ++i) {
        const double e = P[i] * (double) n;
        if (e == 0) {
            if (cnt[i] > 0) { stat = 1e30; dof = 1; return 1e30; }   // a continuation the chain never emits
            continue;
        }
        if (e < 5) { pool_e += e; pool_o += (double) cnt[i]; continue; }
        stat += ((double) cnt[i] - e) * ((double) cnt[i] - e) / e;
        ++dof;
    }
    if (pool_e > 0) { stat += (pool_o - pool_e) * (pool_o - pool_e) / pool_e; ++dof; }
    const double k = std::max(1, dof);
    return (std::cbrt(stat / k) - (1 - 2 / (9 * k))) / std::sqrt(2 / (9 * k));
}

constexpr double kZ = 3.09;   // one-sided 99.9%

bool dist_case(const char* name, const Sampling& s, SpecRule rule, int D, Control ctl, int n, bool want_pass) {
    std::vector<double> P((size_t) std::pow(V, L), 0.0);
    std::vector<int32_t> h = kPrompt;
    exact(s, h, 0, 1.0, 0, P);
    std::vector<long long> cnt(P.size(), 0);
    long long off = 0, acc = 0;
    for (int i = 0; i < n; ++i) {
        const std::vector<int32_t> o = spec(s, 1000003ull * (unsigned long long) i + 7, rule, D, ctl, off, acc);
        size_t code = 0;
        for (int t : o) code = code * V + (size_t) t;
        ++cnt[code];
    }
    double stat;
    int dof;
    const double z = chi_z(cnt, P, n, stat, dof);
    const bool pass = z < kZ;
    const bool ok = pass == want_pass;
    if (!ok) ++g_fail;
    std::printf("  %-44s chi2 %10.1f dof %4d z %7.2f  accepted %5.1f%%  %s%s\n", name, stat, dof, std::min(z, 1e6),
                off ? 100.0 * (double) acc / (double) off : 0.0, pass ? "matches p" : "differs from p",
                ok ? "" : "   <-- FAIL");
    return ok;
}

}  // namespace

int main() {
    std::mt19937_64 g(20261022);
    std::normal_distribution<float> nd(0.f, 1.5f);
    g_tab.resize((size_t) V * V * V);
    g_noise.resize(g_tab.size());
    for (float& x : g_tab) x = nd(g);
    for (float& x : g_noise) x = 0.6f * nd(g);

    Sampling card;   // the model card's thinking set as the server sends it, plus a repetition penalty and min_p
    card.temperature = 1.0f;
    card.top_p = 0.95f;
    card.top_k = 20;
    card.present = 1.5f;
    card.last_n = 64;
    Sampling cold = card;
    cold.temperature = 0.6f;
    cold.top_k = 5;
    cold.top_p = 0.8f;
    cold.min_p = 0.05f;
    cold.repeat = 1.1f;
    cold.present = 0.5f;
    Sampling greedy_pen;   // temperature 0 with a penalty: sample() is the penalised argmax, no draw
    greedy_pen.present = 1.5f;
    greedy_pen.last_n = 64;

    std::printf("q36_sampling_test: %d-token vocabulary, %d-token continuations\n", V, L);
    // 1. match: the same text as the plain sampler, seed for seed
    int same = 0, n1 = 20000;
    for (const Sampling* s : {&card, &cold, &greedy_pen})
        for (int D = 1; D <= 4; ++D)
            for (int i = 0; i < n1; ++i) {
                long long o = 0, a = 0;
                same += spec(*s, (unsigned long long) i, SpecRule::match, D, Control::none, o, a) == plain(*s, (unsigned long long) i);
            }
    const int n1_all = 3 * 4 * n1;
    std::printf("  match = plain sampler, seed for seed: %d of %d%s\n", same, n1_all, same == n1_all ? "" : "   <-- FAIL");
    if (same != n1_all) ++g_fail;

    // 2. the joint of the first L tokens against the exact distribution
    const int n = 400000;
    dist_case("match, card sampling, D=2", card, SpecRule::match, 2, Control::none, n, true);
    dist_case("q, card sampling, D=1", card, SpecRule::q, 1, Control::none, n, true);
    dist_case("q, card sampling, D=2", card, SpecRule::q, 2, Control::none, n, true);
    dist_case("q, card sampling, D=3", card, SpecRule::q, 3, Control::none, n, true);
    dist_case("q, cold sampling (min_p, repeat), D=2", cold, SpecRule::q, 2, Control::none, n, true);
    dist_case("match, cold sampling, D=3", cold, SpecRule::match, 3, Control::none, n, true);
    // gated drafting (--spec-min-p): a variable number of drafts per step
    g_min_p = 0.5f;
    dist_case("match, card sampling, D=3, min_p 0.5", card, SpecRule::match, 3, Control::none, n, true);
    dist_case("q, card sampling, D=3, min_p 0.5 (q's top)", card, SpecRule::q, 3, Control::none, n, true);
    dist_case("q, cold sampling, D=3, min_p 0.5 (q's top)", cold, SpecRule::q, 3, Control::none, n, true);
    dist_case("control: q gated on the drawn token's q", card, SpecRule::q, 3, Control::gate_pick, n, false);
    g_min_p = 0.f;
    // 3. controls: must differ
    dist_case("control: q with a wrong q list, D=2", card, SpecRule::q, 2, Control::wrong_q, n, false);
    dist_case("control: rows blind to the drafts, D=2", card, SpecRule::match, 2, Control::blind_rows, n, false);

    // 4. --draft-vocab: a row of 6 of the 8 ids, a history that holds ids 6 and 7
    {
        std::vector<float> row(6, 0.5f);
        const std::vector<int32_t> h = {1, 7, 6, 1, 2};
        Chain c;
        chain(row, card, h, c);
        const bool ok = row.size() == 6 && row[1] == 0.5f - 1.5f && row[2] == 0.5f - 1.5f && row[0] == 0.5f && row[5] == 0.5f;
        std::printf("  draft-vocab row: penalties on its own ids only: %s\n", ok ? "yes" : "no   <-- FAIL");
        if (!ok) ++g_fail;
    }
    g_draft_v = 6;
    dist_case("q, card sampling, D=2, draft vocab 6 of 8", card, SpecRule::q, 2, Control::none, n, true);
    dist_case("match, card sampling, D=2, draft vocab 6 of 8", card, SpecRule::match, 2, Control::none, n, true);
    g_draft_v = V;

    std::printf("q36_sampling_test: %s\n", g_fail ? "FAILED" : "passed");
    return g_fail ? 1 : 0;
}
