// include/strata/core/batch_spec.hpp - how a batch window shares its rows between the slots that speculate, and what a
// slot keeps from its group, as pure logic so a CPU test can pin the numbers down while the GPU is busy.
//
// A window holds strata::kernels::kVerifyMaxT rows (include/strata/kernels/verify_kernels.hpp:21) and every slot keeps
// one anchor row, so `n_active` slots leave `cap - n_active` rows for proposals: `cap / n_active - 1` each - 7 drafts
// with one slot (what --draft dflash2 does solo today), 3 with two (what --batch-mtp does today), 1 with four and none
// from 8 slots up, where --batch stays plain.  The division leaves a remainder when the slots do not divide the window;
// batch_spec_rows hands those rows round-robin to the first slots, so the split stays fair and the layout stays one the
// window's own law accepts (batch_rows_error, include/strata/core/batch_rows.hpp): a slot's rows one contiguous group at
// consecutive positions.
//
// batch_spec_accept is the rule src/program/generate.cpp applies today to --batch-mtp's two-row group (13244-13252),
// written for a group of any length: keep the proposals the target confirmed, stop at the first mismatch, and let the
// slot's next anchor and position follow the last emitted token, as the commit loop does (13283-13302).
#pragma once

#include <cstdint>

#include "strata/kernels/verify_kernels.hpp"

namespace strata::core {

/// The proposals EVERY one of `n_active` slots gets from a `cap`-row window: the anchor rows first, so
/// `cap / n_active - 1` proposals each, never more than `max_drafts` and never below 0.  A window with no room for one
/// anchor row per slot, and no active slot, give none.  This is the floor: batch_spec_rows hands the rows the division
/// leaves over to the first slots, so no slot gets fewer proposals than this and some get one more.
inline int batch_spec_draft_budget(int cap, int n_active, int max_drafts) {
    if (cap <= 0 || n_active <= 0 || max_drafts <= 0) return 0;
    const int k = cap / n_active - 1;
    if (k <= 0) return 0;
    return k < max_drafts ? k : max_drafts;
}

/// The count of proposals a slot may retain for the next window. The slot-aware row splitter spends the shared
/// proposal rows among READY slots later, including a fair remainder that the minimum-per-slot budget above may be
/// zero for (for example, cap 5 with 4 active slots still assigns one proposal row). Keep the drafter's available
/// proposals here; batch_spec_slot_rows clamps each selected slot to the rows actually allocated.
inline int batch_spec_draft_store_cap(int max_drafts) {
    if (max_drafts <= 0) return 0;
    const int window_max = strata::kernels::kVerifyMaxT - 1;
    return max_drafts < window_max ? max_drafts : window_max;
}

/// The window's rows.  `rows_out[t]` is the slot of row t - an index 0..n_active-1 into the slots the batch loop walks
/// (its `active[]`, so `rows_out[first[t]] == active[t]`) - and `ntok_out[a]` is slot a's group length: its anchor row
/// and its proposals.  The `cap - n_active` rows past the anchor rows go round-robin to the first slots, a group never
/// longer than 1 + max_drafts, so the window carries min(cap, n_active * (1 + max_drafts)) rows.  Returns that count S,
/// leaving both arrays untouched at 0 when the window cannot hold one anchor row per slot or there is no slot.  The
/// contract at the caller: `rows_out` and `ntok_out` are kVerifyMaxT entries each (what Verifier::init_slots holds),
/// n_active never exceeds kVerifyMaxT, and a `cap` larger than kVerifyMaxT is CLAMPED to it, never honoured - a caller
/// passing the verifier's own capacity (ver.max_t(), above 8 with --spec) gets the 8-row window, and no read or write
/// passes those arrays.  max_drafts 0 is plain --batch: one row per slot, S == n_active.
inline int batch_spec_rows(int cap, int n_active, int max_drafts, int* rows_out, int* ntok_out) {
    if (cap <= 0 || n_active <= 0 || rows_out == nullptr || ntok_out == nullptr) return 0;
    if (cap > strata::kernels::kVerifyMaxT)
        cap = strata::kernels::kVerifyMaxT;  // the caller's arrays hold kVerifyMaxT rows: a wider cap is clamped, never honoured
    if (n_active > strata::kernels::kVerifyMaxT) return 0;  // the window never holds more anchor rows than this
    if (cap < n_active) return 0;  // every slot keeps an anchor row: the loop narrows the window's slots instead
    const int md = max_drafts > 0 ? max_drafts : 0;
    const long long most = (long long) n_active * md;
    const long long left = cap - n_active;
    const int extra = (int) (left < most ? left : most);  // rows spent on proposals, round-robin: a lap, then the first
    const int each = extra / n_active, first = extra % n_active;
    int t = 0;
    for (int a = 0; a < n_active; ++a) {
        const int k = each + (a < first ? 1 : 0);
        ntok_out[a] = 1 + k;
        for (int j = 0; j <= k; ++j) rows_out[t++] = a;
    }
    return t;
}

/// What one slot keeps from its group.
struct BatchSpecAccept {
    int keep = 0;                                     // the rows kept (0 when the group had none)
    int32_t emitted[strata::kernels::kVerifyMaxT];    // the tokens to emit, [0, keep); -1 past the kept prefix
    int32_t anchor = 0;                               // the slot's next current token: the last emitted one (`sl.x`)
    int64_t pos_after = 0;                            // the next window's anchor position for this slot (`sl.p`)
    bool eos = false;                                 // an emitted token ended the stream: the slot is done
    bool capped = false;                              // --max-new or max_context dropped a confirmed proposal
};

/// The accepted prefix of one slot's group, the same rule src/program/generate.cpp:13250 applies to a two-row
/// --batch-mtp group, run down `ntok` rows.
///
/// `target_pick[j]` is row j's greedy pick from the target model - `outb[first[t] + j]`, what the loop emits - and
/// `drafts[j]` is the slot's proposal for row j+1's position, the token that was put ON that row (`tok[i + 1]`).
/// Proposal j is kept when the target picked it, exactly as `outb[i] == tok[i + 1]` does; the run stops at the first
/// mismatch, so `keep` is 1 plus the matching prefix and never more than `ntok`.  What is emitted is the target's own
/// picks, `target_pick[0 .. keep-1]`; `anchor` is the last of them and `pos_after` is `p0 + keep`.
///
/// `anchor_in` is the token the slot held at `p0`, the one row 0 was built from: the target re-picks row 0, so the rule
/// never reads it, and it survives as `anchor` only when nothing is emitted.  `p0` is the anchor row's position and
/// `pos_limit` o.max_context; `produced` / `max_new` are the slot's token budget and `eos_ids` / `n_eos` its
/// end-of-stream tokens; `stop` is the slot's cancel flag - a cancelled slot takes no proposal, as `!sl.stop` says
/// today, but still emits its own pick.
///
/// A proposal is dropped, and `capped` set, when it would leave no room: for `keep` emitted tokens the slot needs
/// `produced + keep <= max_new` and `p0 + keep + 1 <= pos_limit` (the token after the last emitted one, which the next
/// window's anchor row reads), which for keep == 2 is today's `sl.produced + 2 <= sl.max_new && sl.p + 3 <=
/// o.max_context`.  The longest prefix that fits is kept.  An end-of-stream token truncates the group at its own row; a
/// prefix cut by a bound can therefore end before an eos token that the proposals would have reached.  `eos` and
/// `capped` are independent statements about the group - what the emitted tokens say, and what the bounds did - and a
/// group can carry both; `stop` is what clears `capped`, because a cancelled slot was never going to take the proposal.
inline BatchSpecAccept batch_spec_accept(const int32_t* target_pick, const int32_t* drafts, int ntok,
                                         int32_t anchor_in, int64_t p0, int64_t pos_limit, int64_t produced,
                                         int64_t max_new, const int32_t* eos_ids, int n_eos, bool stop) {
    BatchSpecAccept r;
    for (int j = 0; j < strata::kernels::kVerifyMaxT; ++j) r.emitted[j] = -1;
    r.anchor = anchor_in;
    r.pos_after = p0;
    if (target_pick == nullptr || ntok < 1) return r;  // no row for this slot: emit nothing, keep its token as the anchor
    const int n = ntok > strata::kernels::kVerifyMaxT ? strata::kernels::kVerifyMaxT : ntok;
    auto eos_here = [&](int32_t y) {
        if (eos_ids == nullptr) return false;
        for (int i = 0; i < n_eos; ++i)
            if (eos_ids[i] == y) return true;
        return false;
    };
    int keep = 1;  // the anchor row is the target's own pick, confirmed or not
    if (!stop && drafts != nullptr)
        while (keep < n && target_pick[keep - 1] == drafts[keep - 1]) ++keep;
    while (keep > 1 && (produced + keep > max_new || p0 + keep + 1 > pos_limit)) {
        --keep;
        r.capped = true;
    }
    for (int j = 0; j < keep; ++j) {
        if (!eos_here(target_pick[j])) continue;
        keep = j + 1;
        r.eos = true;
        break;
    }
    for (int j = 0; j < keep; ++j) r.emitted[j] = target_pick[j];
    r.keep = keep;
    r.anchor = target_pick[keep - 1];
    r.pos_after = p0 + keep;
    return r;
}

}  // namespace strata::core
