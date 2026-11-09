// include/strata/spec/window_rule.hpp - which rule picks a round's verify window, by the round's ACTIVE draft source.
//
// `use_mtp` means "the MTP layer is loaded"; with DFlash2 it stays true because the MTP is the fallback.  The decisions below
// read what the MTP chain just produced (the cost-aware rule's `chain_rows`, the draft probabilities), so they apply only when
// the MTP chain is the source of this round's candidates.  With DFlash2 as the source, `chain_rows` is never written (the chain
// callback does not run) and would pin the window at one row.  Header only, no device types: the CLI loop and the serve loop of
// every backend call it.
#pragma once

#include <cmath>
#include <cstdlib>
#include <algorithm>

namespace strata::spec {

struct WindowRuleIn {
    bool use_mtp = false;        ///< the MTP layer is loaded (a fallback beside DFlash2 counts)
    bool draft_on = false;       ///< some drafter proposes (the MTP, or DFlash2 alone with no MTP loaded)
    bool df2_active = false;     ///< DFlash2 drew this round's candidates (CLI: use_df2; serve: this request's drafter)
    bool spec_cost = false;      ///< STRATA_SPEC_COST=1: the cost-aware MTP window
    bool first_window = false;   ///< the prompt's last token alone: the caller sets its own width afterwards
    int chain_rows = 1;          ///< the window the cost rule kept in the MTP chain just drafted (MTP only)
    int cap = 1;                 ///< S_mtp: the widest window of this source
    int start = 1;               ///< the width before any rule (CLI: cap; serve: cap with the MTP, 1 without)
    double spec_min_p = 0.0;     ///< --spec-min-p: stop at the first draft whose probability is below it (0: off)
    double draft_conf = 0.0;     ///< --draft-conf: DFlash2's window ends where the product of the probabilities falls below it (0: off)
    const float* dprob = nullptr;   ///< the drafts' probabilities, dprob[j] for draft j (the row j + 1 of the window)
};

/// The cost-aware MTP window applies only to the MTP chain's own candidates.
inline bool cost_window_applies(bool use_mtp, bool spec_cost, bool df2_active, bool first_window) {
    return use_mtp && spec_cost && !df2_active && !first_window;
}

/// The MTP's q lists (STRATA_SPEC_PROB) are the proposal law only for candidates the MTP drew.
inline bool mtp_q_applies(bool use_mtp, bool df2_active, bool mtp_prob, bool from_lookup, bool first_window, int mtp_rows) {
    return use_mtp && !df2_active && mtp_prob && !from_lookup && !first_window && mtp_rows > 1;
}

/// DFlash2 with a LONG suffix match: the lookup window is taken without a DFlash2 draft round.  The choice "lookup or DFlash2" needs the draft only to
/// test that its first token agrees; when the match is long the lookup is the likelier winner and the draft (2.4 ms on Metal, 1.2 ms on the RTX PRO
/// 5000) would be thrown away.  The round's draft is then DEFERRED to the start of the next iteration, where the policy decides (lookup: the committed
/// rows only update the drafter's ring; otherwise the draft is made then, from the same taps).  min_match 0 = off.
inline bool skip_draft_for_lookup(bool df2_active, int min_match, int lookup_k, int match) {
    return df2_active && min_match > 0 && lookup_k > 0 && match >= min_match;
}

/// STRATA_DF2_LOOKUP_SKIP=N (match length in tokens; 0 = off).  `build_default`: what a build without the variable uses (Metal 8, others 0 until measured).
inline int df2_lookup_skip_min(int build_default) {
    static const int v = [build_default] {
        const char* e = std::getenv("STRATA_DF2_LOOKUP_SKIP");
        return e != nullptr && e[0] != '\0' ? std::atoi(e) : build_default;
    }();
    return v;
}

inline int choose_window(const WindowRuleIn& in) {
    int T = in.start;
    if (cost_window_applies(in.use_mtp, in.spec_cost, in.df2_active, in.first_window)) {
        T = std::clamp(in.chain_rows, 1, in.cap);
    } else if (in.draft_on && in.spec_min_p > 0.0 && in.dprob != nullptr) {
        T = 1;
        while (T < in.cap && in.dprob[T - 1] >= (float) in.spec_min_p) ++T;
    }
    if (in.df2_active && in.draft_conf > 0.0 && in.dprob != nullptr) {   // DFlash2's own confidence: every draft so far holds
        T = 1;
        double chain = 1.0;
        while (T < in.cap && (chain *= (double) in.dprob[T - 1]) >= in.draft_conf) ++T;
    }
    return T;
}

/// STRATA_DF2_GATE=1 (opt-in, measured on the Mac): DFlash2's window from a calibrated survival instead of a fixed confidence.  The selector's cumulative confidence C_j of
/// draft j over-states the real chance that drafts 1..j all hold (M5 Max UD-Q4_K_XL: C = 0.84, survival 0.70 at j = 5), and the right confidence differs by request
/// (a story wants a narrower window than JSON).  Per request this keeps, for every position j and confidence bucket, how often the first j drafts held, shrunk toward a
/// prior of C^1.3, and extends the window to row j+1 while  survival * (ms per committed token) >= row_ms  (what one more verify row costs).
struct AdaptiveGate {
    static constexpr int NB = 6;
    double n[8][NB] = {}, h[8][NB] = {};
    double ms_per_tok = 0.0;       // EMA of the round's wall ms per committed token
    double row_ms = 3.0;           // one more verify row (M5 Max: 2.65-3.3 ms)
    double prior_n = 3.0;
    static int bucket(double c) { return c < 0.15 ? 0 : c < 0.3 ? 1 : c < 0.5 ? 2 : c < 0.7 ? 3 : c < 0.85 ? 4 : 5; }
    double survival(int j, double c) const {
        const int b = bucket(c);
        const double prior = std::pow(c, 1.3);
        return (h[j][b] + prior_n * prior) / (n[j][b] + prior_n);
    }
    /// rows of the next window: 1 + the drafts kept (cap = widest window)
    int choose(const float* dprob, int cap) const {
        int T = 1;
        double chain = 1.0;
        const double v = ms_per_tok > 0.0 ? ms_per_tok : 9.0;
        while (T < cap) {
            chain *= (double) dprob[T - 1];
            if (survival(T, chain) * v < row_ms) break;
            ++T;
        }
        return T;
    }
    /// after a DFlash2 round of T rows, a drafts accepted: drafts 1..T-1 were offered, draft j survived when a >= j
    void observe(const float* dprob, int T, int a) {
        double chain = 1.0;
        for (int j = 1; j < T && j < 8; ++j) {
            chain *= (double) dprob[j - 1];
            const int b = bucket(chain);
            n[j][b] += 1.0;
            h[j][b] += a >= j ? 1.0 : 0.0;
        }
    }
    void observe_round(double round_ms, int committed) {
        if (committed < 1) return;
        const double m = round_ms / committed;
        ms_per_tok = ms_per_tok > 0.0 ? 0.9 * ms_per_tok + 0.1 * m : m;
    }
};

inline bool df2_gate_on() {
    static const bool v = [] { const char* e = std::getenv("STRATA_DF2_GATE"); return e != nullptr && e[0] == '1'; }();
    return v;
}
inline double df2_gate_row_ms() {
    static const double v = [] { const char* e = std::getenv("STRATA_DF2_ROW_MS"); return e != nullptr && e[0] != '\0' ? std::atof(e) : 3.0; }();
    return v;
}

}  // namespace strata::spec
