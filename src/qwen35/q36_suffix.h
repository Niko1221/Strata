// src/qwen35/q36_suffix.h - opt-in suffix-lookup drafts for q36's single-session decode.
//
// Flash-Next decides per round whether a suffix (prompt-lookup) window beats the MTP window, comparing expected
// committed tokens per millisecond (include/strata/spec/draft_policy.hpp).  This header is the q36 side of that
// decision: it holds the drafter and the policy, and answers one question per speculative step - "what window do
// we verify this round?" - without touching the GPU path.  With --suffix-draft unset nothing here is constructed
// and spec_step never asks.
#pragma once

#include "strata/spec/draft_policy.hpp"
#include "strata/spec/suffix_drafter.hpp"

#include <algorithm>
#include <cstdint>
#include <vector>

namespace strata::qwen35 {

/// One round's window choice.  `lookup` means the verify window is filled with suffix drafts instead of MTP drafts;
/// `deff` is the number of drafts the window verifies (0 = no drafts, a plain step).
struct SuffixChoice {
    bool lookup = false;
    int deff = 0;     // drafts in the window
    int match = 0;    // length of the match behind the proposal (0 when none)
};

class Q36Suffix {
public:
    // min_match 6, not the drafter's default 3: a 3-token match gives E ~ 1.33 against plain decode's 1.0, which is
    // not worth the risk on prose (measured offline: prose yields 0.14 accepted tokens per step).  max_match 32 is
    // the drafter's own ceiling.
    explicit Q36Suffix(int max_k) : drafter_(6, 32, 1u << 19), policy_(8, 0.03), max_k_(std::max(1, max_k)) {}

    void reset() { drafter_.reset(); }

    void append(int32_t token) { drafter_.append(token); }
    void append(const int32_t* tokens, size_t n) { drafter_.append(tokens, n); }
    size_t size() const { return drafter_.size(); }

    /// Decide this round's window.  `x_tok` is the token just emitted (not yet in the history); `d_mtp` the MTP
    /// depth that would otherwise run; `room` the drafts the rebalance clamp and the verify window still allow.
    /// The proposal is kept in `buf_` until `fill()` writes it.  A lookup window replaces the MTP drafts for this
    /// round (Flash-Next's choose(), not chain(): q36's draft head returns an argmax and no probability).
    SuffixChoice choose(int32_t x_tok, int d_mtp, int room) {
        SuffixChoice c;
        c.deff = std::clamp(d_mtp, 0, room);
        if (room <= 0) return c;
        const int cap = std::min({max_k_, room, 63});
        const int k = drafter_.propose_after(&x_tok, 1, cap, buf_);
        if (k <= 0) return c;
        const int match = drafter_.last_match();
        // choose() compares a lookup window against an MTP window of t_mtp tokens (1 + drafts).  A plain step is an
        // MTP window of 1, so it is a fair comparison either way.  The policy can also ask to re-measure a stale
        // size; clamp that back to the proposal and the room, or the window would verify tokens nobody proposed.
        const strata::spec::DraftPolicy::Pick pk = policy_.choose(std::max(1, d_mtp + 1), k, match);
        if (!pk.lookup) return c;
        c.lookup = true;
        c.deff = std::clamp(pk.t - 1, 1, k);
        c.match = match;
        return c;
    }

    /// Write the chosen suffix drafts into `drafts[0 .. deff)`.  Valid only after choose() returned lookup.
    void fill(std::vector<int32_t>& drafts, int deff) const {
        for (int i = 0; i < deff && i < (int) drafts.size(); ++i) drafts[(size_t) i] = buf_[(size_t) i];
    }

    void observe(bool lookup, int t, int accepted, int match, double round_ms) {
        policy_.observe(lookup, t, accepted, match, round_ms);
    }

private:
    strata::spec::SuffixDrafter drafter_;
    strata::spec::DraftPolicy policy_;
    int max_k_;
    int32_t buf_[64] = {};
};

}  // namespace strata::qwen35
