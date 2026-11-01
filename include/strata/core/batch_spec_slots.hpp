// include/strata/core/batch_spec_slots.hpp - the two decisions a batch window makes per slot when the slots carry
// their own DFlash2 proposals, on top of batch_spec.hpp (the row split and the accept rule), as pure logic so a CPU
// test can pin them down while the GPU is busy.
//
// batch_spec_rows gives every slot the same proposal budget; a real window cannot.  A slot admitted this round has
// no draft ready (its ring was primed, its first round has not run), and a drafter round that failed leaves its slot
// proposing nothing.  Those slots keep their anchor row and take no proposal row, so the window's spare rows go to
// the ready slots alone, round-robin among them, each bounded by max_drafts - the same fair split batch_spec_rows
// makes, over the slots that can propose.  The rows stay one contiguous group per slot, so the window's own law
// (batch_rows_error, include/strata/core/batch_rows.hpp) accepts the layout.
//
// df2_stage_rows is the geometry of the next draft round: the drafter reads the committed rows' taps, and a slot's
// committed prefix is the window's rows [first, first + keep) - contiguous, inside its own group, so a row the
// target rejected is never staged (the ring only ever sees committed rows, include/strata/core/dflash2.hpp).
#pragma once

#include "strata/core/batch_dflash2_depth.hpp"
#include "strata/core/batch_spec.hpp"
#include "strata/kernels/verify_kernels.hpp"

#include <cstdint>

namespace strata::core {

/// The window's rows when only some of its slots have a proposal ready.  The window holds `cap` rows and the loop
/// selected `n_sel` slots (at most `cap`, one anchor row each); `ready[a]` says whether selected slot `a` has a
/// draft to propose.  `rows_out[t]` is the selected slot of row t (an index 0..n_sel-1, exactly as batch_spec_rows
/// numbers them: `rows_out[first[a]] == a`), `ntok_out[a]` is that slot's group length.
///
/// Every selected slot gets its anchor row; the `cap - n_sel` spare rows go round-robin to the READY slots, a group
/// never longer than 1 + max_drafts, so the window carries n_sel + min(cap - n_sel, ready * max_drafts) rows.  A
/// slot that is not ready keeps its anchor row alone.  With every slot ready this is batch_spec_rows' split, and a
/// `ready` of nullptr means all ready.  Returns the rows used, leaving both arrays untouched when the window cannot
/// hold one anchor row per slot, there is no slot, an array is missing, or more than kVerifyMaxT slots are selected.
/// The contract at the caller matches batch_spec_rows: `rows_out`, `ntok_out` and `ready` are kVerifyMaxT entries
/// each, `ready` read for the selected slots alone, and a `cap` larger than kVerifyMaxT is CLAMPED to it, never
/// honoured - no call reads or writes past those arrays.
inline int batch_spec_slot_rows(int cap, int n_sel, const bool* ready, int max_drafts, int* rows_out, int* ntok_out) {
    if (cap <= 0 || n_sel <= 0 || rows_out == nullptr || ntok_out == nullptr) return 0;
    if (cap > strata::kernels::kVerifyMaxT)
        cap = strata::kernels::kVerifyMaxT;  // the caller's arrays hold kVerifyMaxT rows: a wider cap is clamped, never honoured
    if (cap < n_sel) return 0;  // every slot keeps an anchor row: the loop narrows the window's slots instead
    if (n_sel > strata::kernels::kVerifyMaxT) return 0;  // the window's own cap: the caller selects no more
    const int md = max_drafts > 0 ? max_drafts : 0;
    int n_ready = 0;
    for (int a = 0; a < n_sel; ++a)
        if (ready == nullptr || ready[a]) ++n_ready;
    const long long most = (long long) n_ready * md;
    const long long left = cap - n_sel;
    const int extra = (int) (left < most ? left : most);  // rows spent on proposals, round-robin over the ready slots
    // the laps go in slot order over the ready slots, so the longer groups come first and no ready slot is short by
    // more than one proposal of another (the same fairness batch_spec_rows' division has)
    int k[strata::kernels::kVerifyMaxT];
    for (int a = 0; a < n_sel; ++a) k[a] = 0;
    for (int given = 0; given < extra;) {
        for (int a = 0; a < n_sel && given < extra; ++a) {
            if (ready != nullptr && !ready[a]) continue;
            if (k[a] >= md) continue;
            ++k[a];
            ++given;
        }
    }
    int t = 0;
    for (int a = 0; a < n_sel; ++a) {
        ntok_out[a] = 1 + k[a];
        for (int j = 0; j <= k[a]; ++j) rows_out[t++] = a;
    }
    return t;
}

/// Per-slot proposal caps `max_drafts[0..n_sel)` (STRATA_BATCH_DF2_DEPTH): 0 keeps a slot's anchor row alone, >0 asks
/// for up to that many proposals.  When `ema` is provided and the requested proposals exceed `cap` while at least one
/// proposal per ready slot fits, `batch_draft_depth_fit` drops from the slot with the lowest `ema[a]^k[a]` first and never
/// drops a ready slot's first proposal; when `ema` is nullptr (or not even one proposal per ready slot fits), spare
/// rows are dealt round-robin from the left.
inline int batch_spec_slot_rows(int cap, int n_sel, const int* max_drafts, int* rows_out, int* ntok_out,
                                const float* ema = nullptr) {
    if (cap <= 0 || n_sel <= 0 || rows_out == nullptr || ntok_out == nullptr) return 0;
    if (cap > strata::kernels::kVerifyMaxT) cap = strata::kernels::kVerifyMaxT;
    if (cap < n_sel || n_sel > strata::kernels::kVerifyMaxT) return 0;
    const int max_extra = cap - n_sel;
    int k[strata::kernels::kVerifyMaxT] = {};
    int cap_k[strata::kernels::kVerifyMaxT] = {};
    int n_ready = 0;
    int total_want = n_sel;
    for (int a = 0; a < n_sel; ++a) {
        const int md = (max_drafts != nullptr && max_drafts[a] > 0)
                           ? (max_drafts[a] < max_extra ? max_drafts[a] : max_extra)
                           : 0;
        cap_k[a] = md;
        if (md > 0) ++n_ready;
        total_want += md;
    }
    if (n_ready > 0) {
        if (total_want <= cap) {
            for (int a = 0; a < n_sel; ++a) k[a] = cap_k[a];
        } else if (ema != nullptr && n_sel + n_ready <= cap) {
            for (int a = 0; a < n_sel; ++a) k[a] = cap_k[a];
            (void) batch_draft_depth_fit(k, ema, n_sel, cap);
        } else {
            int given = 0;
            while (n_sel + given < cap) {
                bool grew = false;
                for (int a = 0; a < n_sel && n_sel + given < cap; ++a) {
                    if (k[a] >= cap_k[a]) continue;
                    ++k[a];
                    ++given;
                    grew = true;
                }
                if (!grew) break;
            }
        }
    }
    int t = 0;
    for (int a = 0; a < n_sel; ++a) {
        ntok_out[a] = 1 + k[a];
        for (int j = 0; j <= k[a]; ++j) rows_out[t++] = a;
    }
    return t;
}

/// What one window of a DFlash2 batch run does: how many of the active slots it runs, and how deep each of those
/// slots' chains may go.  A window holds `cap` rows, so breadth (slots per window) and depth (proposals per slot)
/// trade against each other: at cap 8, four slots take 1 proposal each, two slots take 3, one slot takes 7.
struct Df2WindowPlan {
    int n_sel = 0;   ///< the slots the window runs, in the caller's rotation order
    int depth = 0;   ///< the proposals each ready slot may take: batch_spec_slot_rows' `max_drafts`
};

/// The plan for `n_active` active slots in a `cap`-row window whose drafter proposes at most `max_drafts`.
///
/// `want_depth` 0 keeps the widest window that still speculates: every active slot runs (up to `cap` of them) and
/// each takes `cap / n_sel - 1` proposals - four slots at cap 8 take one each.  `want_depth >= 1` puts that depth
/// first: the window runs at most `cap / (1 + want_depth)` slots so each of them can hold `want_depth` proposals,
/// and the active slots that do not fit wait for a later window (the caller rotates, so the wait is fair).  Once the
/// slots are chosen the depth is whatever fits them, so a lone slot with the window to itself still takes the
/// drafter's full chain rather than stopping at the asked depth: depth = min(max_drafts, cap / n_sel - 1).
///
/// Clamps: `want_depth` never exceeds `max_drafts`; `cap` at or under `n_active` gives an anchor-only window
/// (depth 0, the plain --batch layout); no slots, no cap or a non-positive drafter depth give an empty plan.
inline Df2WindowPlan df2_window_plan(int cap, int n_active, int max_drafts, int want_depth) {
    Df2WindowPlan p;
    if (cap <= 0 || n_active <= 0 || max_drafts <= 0) return p;
    const int want = want_depth < 0 ? 0 : (want_depth > max_drafts ? max_drafts : want_depth);
    p.n_sel = n_active < cap ? n_active : cap;   // an anchor row per slot: never more slots than rows
    if (want > 0) {
        const int fits = cap / (1 + want);       // slots that still leave `want` proposal rows each
        if (fits < p.n_sel) p.n_sel = fits < 1 ? 1 : fits;   // never empty: one slot keeps the window running
    }
    const int room = cap / p.n_sel - 1;
    p.depth = room < 0 ? 0 : (room > max_drafts ? max_drafts : room);
    return p;
}

/// The pure row contract for one pipelined DFlash2 group. `active` and `ready` are group-local slot masks (`nullptr`
/// means all active/ready). The window chooses active slots in cyclic order starting at `cursor`, then gives each an
/// anchor and splits proposal rows only across ready slots. `selected_slots[0..n_sel)` records the local slots in
/// execution order; `row_slots[0..rows)` maps each contiguous row group back to its local slot id; `ntok_out[a]` is
/// the selected slot's row count. All output arrays need kVerifyMaxT entries. The active/ready masks may be longer
/// than the row capacity: a group can rotate more slots through successive windows. Returns the next cursor just
/// after the last selected slot. This is a host-testable planning primitive; it does not imply that the live
/// layer-split path can yet collect DFlash2 tap rows across stages.
struct Df2PipedGroupRows {
    int n_sel = 0;
    int depth = 0;
    int rows = 0;
    int next = 0;
};

inline Df2PipedGroupRows df2_piped_group_rows(int cap, int n_slots, const bool* active, const bool* ready,
                                              int max_drafts, int want_depth, int cursor, int* selected_slots,
                                              int* row_slots, int* ntok_out) {
    Df2PipedGroupRows out;
    if (cap <= 0 || n_slots <= 0 || max_drafts <= 0 || selected_slots == nullptr || row_slots == nullptr ||
        ntok_out == nullptr)
        return out;
    if (cap > strata::kernels::kVerifyMaxT) cap = strata::kernels::kVerifyMaxT;
    const int start = ((cursor % n_slots) + n_slots) % n_slots;
    int n_active = 0;
    for (int i = 0; i < n_slots; ++i)
        if (active == nullptr || active[i]) ++n_active;
    if (n_active == 0) { out.next = start; return out; }
    const Df2WindowPlan plan = df2_window_plan(cap, n_active, max_drafts, want_depth);
    if (plan.n_sel <= 0) { out.next = start; return out; }

    int selected = 0;
    for (int step = 0; step < n_slots && selected < plan.n_sel; ++step) {
        const int slot = (start + step) % n_slots;
        if (active == nullptr || active[slot]) selected_slots[selected++] = slot;
    }
    if (selected != plan.n_sel) return Df2PipedGroupRows{};  // defensive; n_active was counted from the same mask

    bool selected_ready[strata::kernels::kVerifyMaxT] = {};
    for (int i = 0; i < plan.n_sel; ++i)
        selected_ready[i] = ready == nullptr || ready[selected_slots[i]];
    const int rows = batch_spec_slot_rows(cap, plan.n_sel, selected_ready, plan.depth, row_slots, ntok_out);
    if (rows <= 0) return Df2PipedGroupRows{};
    for (int r = 0; r < rows; ++r) row_slots[r] = selected_slots[row_slots[r]];

    out.n_sel = plan.n_sel;
    out.depth = plan.depth;
    out.rows = rows;
    out.next = (selected_slots[plan.n_sel - 1] + 1) % n_slots;
    return out;
}

/// What the batch loop watched a run of DFlash2 windows do.  Every counter is a plain sum, so the test can feed the
/// rounds and check the numbers the log line prints.  `depth[d]` counts the slot-rounds that had `d` proposal rows
/// verified and had `d` confirmed - the shape of the acceptance, not just its rate: a run that accepts one-token
/// chains is a different drafter problem from one that throws its proposals away at the first row.
struct Df2DraftStats {
    int64_t rounds = 0;      ///< slot-rounds (the window's anchor rows)
    int64_t drafted = 0;     ///< rounds that started with at least one proposal stored
    int64_t stored = 0;      ///< proposals the drafter left in the slots for these rounds
    int64_t offered = 0;     ///< proposal rows the window actually verified
    int64_t accepted = 0;    ///< proposal rows the target confirmed
    int64_t depth[strata::kernels::kVerifyMaxT + 1] = {};   ///< rounds whose accepted-prefix length is d
    int64_t planned = 0;       ///< windows counted by note_plan
    int64_t plan_slots = 0;    ///< their slot totals
    int64_t plan_depth = 0;    ///< their proposal-depth totals

    /// One window's plan: it ran `n_sel` slots, each allowed `depth` proposal rows.  Counted once per window, not
    /// per slot-round, so a log line can say what shape the windows actually had.
    void note_plan(int n_sel, int depth_each) {
        ++planned;
        plan_slots += n_sel < 0 ? 0 : n_sel;
        plan_depth += depth_each < 0 ? 0 : depth_each;
    }
    double avg_slots() const { return planned == 0 ? 0.0 : (double) plan_slots / (double) planned; }
    double avg_depth() const { return planned == 0 ? 0.0 : (double) plan_depth / (double) planned; }

    /// One slot-round: `n_stored` proposals sat in the slot, `n_offered` of them got a row in the window, and
    /// `n_accepted` of those were confirmed (`n_offered - n_accepted` wasted rows).  Negative inputs count as 0.
    void note_round(int n_stored, int n_offered, int n_accepted) {
        const int stored_n = n_stored < 0 ? 0 : n_stored;
        const int offered_n = n_offered < 0 ? 0 : n_offered;
        const int accepted_n = n_accepted < 0 ? 0 : (n_accepted > offered_n ? offered_n : n_accepted);
        ++rounds;
        stored += stored_n;
        if (stored_n > 0) ++drafted;
        offered += offered_n;
        accepted += accepted_n;
        const int d = accepted_n > strata::kernels::kVerifyMaxT ? strata::kernels::kVerifyMaxT : accepted_n;
        ++depth[d];
    }

    /// Tokens a round yields on average: the anchor row plus the proposals it confirmed.
    double tokens_per_round() const { return rounds == 0 ? 0.0 : (double) (rounds + accepted) / (double) rounds; }
    /// Proposals stored that never reached a row: what the window's row budget threw away.
    int64_t unverified() const { return stored > offered ? stored - offered : 0; }
    /// The accepted depth, highest first, so a log line can show the shape in one glance.
    int deepest_accepted() const {
        for (int d = strata::kernels::kVerifyMaxT; d >= 1; --d)
            if (depth[d] != 0) return d;
        return 0;
    }
};

/// The tap rows slot `b` stages for its next draft: `[first, first + keep)` - the window's rows of its own committed
/// prefix, contiguous, never past its group (`keep <= ntok`) and never past the window's rows (`first + keep <= S`),
/// so a row the target rejected is outside the range.  `keep` 0 is a legal empty range (lo == hi): the slot stages
/// nothing.  nullptr when the range is not legal, leaving both outputs alone.
inline bool df2_stage_rows(int first, int keep, int ntok, int S, int* lo, int* hi) {
    if (lo == nullptr || hi == nullptr) return false;
    if (first < 0 || keep < 0 || ntok < 0 || S < 0) return false;
    if (keep > ntok) return false;
    if (first + keep > S) return false;
    *lo = first;
    *hi = first + keep;
    return true;
}

}  // namespace strata::core
