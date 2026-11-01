// tests/core/batch_spec_test.cpp - how a batch window shares its rows between slots that speculate, and what a slot
// keeps from its group (include/strata/core/batch_spec.hpp).  Two things are pinned here:
//
//   1. the row split at the legacy 8-row cap and the opt-in 16-row storage/runtime cap. Every slot keeps one anchor
//      row; the split is checked against the window's own layout law (batch_rows_error), including the wider shape;
//   2. the storage clamp. Caller arrays hold kVerifyMaxT entries: caps above 16 clamp at storage capacity, while
//      cap 16 is fully honored (audit finding 3);
//   3. the accept rule for one slot's group.  For a two-row group it must give exactly what src/program/generate.cpp
//      computes today for --batch-mtp (13244-13252), token for token, including the eos, --max-new and max_context
//      guards and the slot's next anchor; for longer groups it is the same rule run down the row of proposals.
//
// Header-only, CPU: no verifier, no model, no GPU.
#include "strata/core/batch_spec.hpp"

#include "strata/core/batch_rows.hpp"
#include "strata/kernels/verify_kernels.hpp"

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <vector>

using strata::core::BatchSpecAccept;

namespace {
int g_fail = 0;
long g_checks = 0;
void check(bool ok, const char* what) {
    ++g_checks;
    if (ok) return;
    std::fprintf(stderr, "FAIL: %s\n", what);
    ++g_fail;
}

constexpr int kCap = strata::kernels::kVerifyDefaultMaxT;  // preserved default runtime window: 8 rows
constexpr int kStorageCap = strata::kernels::kVerifyMaxT;  // output arrays and opt-in wide window: 16 rows
// The wide window is a build option (-DSTRATA_WINDOW_MAX_T=16; the HIP engine builds it by default).  In a narrow
// build storage is 8 rows, so a cap-16 plan clamps to 8 and the wide checks below do not apply.
constexpr bool kWideWindow = strata::kernels::kVerifyMaxT >= 16;

bool is_eos(int32_t y, const std::vector<int32_t>& eos_ids) {
    for (int32_t e : eos_ids)
        if (e == y) return true;
    return false;
}

// What src/program/generate.cpp:13250 does today for a two-row --batch-mtp group, copied as the parity target.
int mtp_keep(int32_t pick0, int32_t draft0, const std::vector<int32_t>& eos_ids, int64_t p0, int64_t limit,
             int64_t produced, int64_t max_new, bool stop) {
    return draft0 == pick0 && !is_eos(pick0, eos_ids) && !stop && produced + 2 <= max_new && p0 + 3 <= limit ? 2 : 1;
}

// The rows a split returns as a window: each slot's group at consecutive positions of its own, every slot starting
// somewhere different so a mix-up of the slots shows up.
void positions(const int* rows, int S, int64_t* pos) {
    int64_t next[kStorageCap];
    for (int a = 0; a < kStorageCap; ++a) next[a] = (int64_t) a * 100;
    for (int t = 0; t < S; ++t) pos[t] = next[rows[t]]++;
}

BatchSpecAccept accept(const std::vector<int32_t>& pick, const std::vector<int32_t>& drafts, int32_t anchor_in,
                       int64_t p0, int64_t limit, int64_t produced, int64_t max_new,
                       const std::vector<int32_t>& eos_ids, bool stop) {
    return strata::core::batch_spec_accept(pick.data(), drafts.data(), (int) pick.size(), anchor_in, p0, limit,
                                           produced, max_new, eos_ids.data(), (int) eos_ids.size(), stop);
}

// The invariants every result must hold: the kept prefix is never longer than the group, it is the target's own picks,
// the anchor and the position follow the last emitted token, and nothing past the prefix is written.
void check_accept(const BatchSpecAccept& r, const std::vector<int32_t>& pick, int64_t p0, const char* what) {
    check(r.keep >= 0 && r.keep <= (int) pick.size(), what);
    for (int j = 0; j < r.keep; ++j)
        if (r.emitted[j] != pick[(size_t) j]) {
            check(false, what);
            return;
        }
    if (r.keep > 0) {
        check(r.anchor == r.emitted[r.keep - 1], what);
        check(r.pos_after == p0 + r.keep, what);
    } else {
        check(r.pos_after == p0, what);
    }
    for (int j = r.keep; j < kStorageCap; ++j)
        if (r.emitted[j] != -1) {
            check(false, what);
            return;
        }
}
}  // namespace

int main() {
    // ---- storing draft proposals is independent of this window's floor budget ----
    // The row splitter will allocate a leftover proposal row to a ready slot (e.g. cap 5, 4 slots -> 1 proposal).
    // Do not store the per-slot floor here: that is 0 in this case and would suppress drafting entirely.
    {
        check(strata::core::batch_spec_draft_store_cap(0) == 0, "zero draft cap stores none");
        check(strata::core::batch_spec_draft_store_cap(-3) == 0, "negative draft cap stores none");
        check(strata::core::batch_spec_draft_store_cap(4) == 4, "retain the configured proposals for later row allocation");
        check(strata::core::batch_spec_draft_store_cap(100) == kStorageCap - 1,
              "store up to the 16-row storage ceiling's proposal capacity");
    }

    // ---- the proposal budget: the minimum number of proposals every slot gets from a cap-row window ----
    {
        const int k7[kCap] = {7, 3, 1, 1, 0, 0, 0, 0};
        for (int a = 1; a <= kCap; ++a) {
            char why[96];
            std::snprintf(why, sizeof why, "cap 8, %d active slot(s), max_drafts 7: k == %d", a, k7[a - 1]);
            check(strata::core::batch_spec_draft_budget(kCap, a, 7) == k7[a - 1], why);
        }
        const int k3[kCap] = {3, 3, 1, 1, 0, 0, 0, 0};
        for (int a = 1; a <= kCap; ++a) {
            char why[96];
            std::snprintf(why, sizeof why, "cap 8, %d active slot(s), max_drafts 3: k == %d", a, k3[a - 1]);
            check(strata::core::batch_spec_draft_budget(kCap, a, 3) == k3[a - 1], why);
        }
        check(strata::core::batch_spec_draft_budget(kCap, 1, 100) == 7, "one slot: the window bounds it, not max_drafts");
        check(strata::core::batch_spec_draft_budget(kCap, 2, 1) == 1, "max_drafts below the window: max_drafts bounds it");
        check(strata::core::batch_spec_draft_budget(kCap, 2, 0) == 0, "max_drafts 0: plain --batch, no proposals");
        check(strata::core::batch_spec_draft_budget(kCap, 2, -1) == 0, "a negative max_drafts: no proposals");
        check(strata::core::batch_spec_draft_budget(6, 3, 7) == 1, "cap 6, 3 slots: one proposal each");
        check(strata::core::batch_spec_draft_budget(8, 0, 7) == 0, "no active slot: no proposals");
        check(strata::core::batch_spec_draft_budget(8, -3, 7) == 0, "a negative slot count: no proposals");
        check(strata::core::batch_spec_draft_budget(2, 3, 7) == 0, "cap below the slots: no room, no proposals");
        check(strata::core::batch_spec_draft_budget(0, 1, 7) == 0, "cap 0: no proposals");
        check(strata::core::batch_spec_draft_budget(-4, 1, 7) == 0, "a negative cap: no proposals");
    }

    // ---- the row split: who owns each row of the window, and how long each slot's group is ----
    {
        int rows[kStorageCap], ntok[kStorageCap];
        const int S = strata::core::batch_spec_rows(kCap, 3, 7, rows, ntok);
        check(S == 8, "3 slots in an 8-row window: the window is full");
        check(ntok[0] == 3 && ntok[1] == 3 && ntok[2] == 2, "the group lengths are 3, 3, 2 - the extra rows go round-robin");
        const int want[kCap] = {0, 0, 0, 1, 1, 1, 2, 2};
        bool shape = true;
        for (int t = 0; t < S; ++t)
            if (rows[t] != want[t]) shape = false;
        check(shape, "each slot's rows are one contiguous group, the slots in order");
        int64_t pos[kStorageCap];
        positions(rows, S, pos);
        check(strata::core::batch_rows_error(rows, S, pos, 3, kCap, 0, false, kCap) == nullptr,
              "the split is a layout the verifier accepts");
        check(strata::core::batch_rows_error(rows, S, pos, 3, kCap, 0, true, kCap) == nullptr,
              "... and across a layer split's hand-off");
        check(strata::core::batch_rows_error(rows, S, pos, 3, kCap, 1, true, kCap) != nullptr,
              "the same rows past the hand-off's end are refused");
    }
    {   // max_drafts 0 is plain --batch: one row per slot, no proposals
        int rows[kStorageCap], ntok[kStorageCap];
        const int S = strata::core::batch_spec_rows(kCap, 3, 0, rows, ntok);
        check(S == 3, "max_drafts 0: one row per slot");
        check(ntok[0] == 1 && ntok[1] == 1 && ntok[2] == 1, "max_drafts 0: no slot has a proposal");
        check(rows[0] == 0 && rows[1] == 1 && rows[2] == 2, "max_drafts 0: the rows are the slots in order");
        const int Sneg = strata::core::batch_spec_rows(kCap, 3, -1, rows, ntok);
        check(Sneg == 3, "a negative max_drafts is 0 proposals");
    }
    {   // one slot owns the whole window: solo drafting, kVerifyMaxT rows in one group
        int rows[kStorageCap], ntok[kStorageCap];
        const int S = strata::core::batch_spec_rows(kCap, 1, 7, rows, ntok);
        check(S == 8 && ntok[0] == 8, "one slot: its group is the whole window");
        bool all0 = true;
        for (int t = 0; t < S; ++t)
            if (rows[t] != 0) all0 = false;
        check(all0, "one slot: every row is that slot's");
        int64_t pos[kStorageCap];
        positions(rows, S, pos);
        check(strata::core::batch_rows_error(rows, S, pos, 1, kCap, 0, false, kCap) == nullptr,
              "8 rows of one slot are one group the verifier accepts");
    }
    {   // the leftover rows go to the first slots, and never past 1 + max_drafts
        int rows[kStorageCap], ntok[kStorageCap];
        check(strata::core::batch_spec_rows(kCap, 5, 7, rows, ntok) == 8, "5 slots, 8 rows: the window is full");
        check(ntok[0] == 2 && ntok[1] == 2 && ntok[2] == 2 && ntok[3] == 1 && ntok[4] == 1,
              "5 slots: the 3 extra rows go to the first 3 slots");
        const int S1 = strata::core::batch_spec_rows(kCap, 3, 1, rows, ntok);
        check(S1 == 6, "3 slots, max_drafts 1: 6 rows, the window is not full");
        check(ntok[0] == 2 && ntok[1] == 2 && ntok[2] == 2, "3 slots, max_drafts 1: every group is 2 rows");
        const int S2 = strata::core::batch_spec_rows(kCap, 3, 2, rows, ntok);
        check(S2 == 8 && ntok[0] == 3 && ntok[1] == 3 && ntok[2] == 2, "3 slots, max_drafts 2: the cap binds, 3/3/2");
        const int S3 = strata::core::batch_spec_rows(kCap, 3, 100, rows, ntok);
        check(S3 == 8 && ntok[0] == 3 && ntok[1] == 3 && ntok[2] == 2, "3 slots, max_drafts 100: the window still bounds it");
    }
    {   // a window that cannot hold every slot's anchor row is refused, and writes nothing
        int rows[kStorageCap], ntok[kStorageCap];
        for (int i = 0; i < kCap; ++i) {
            rows[i] = -7;
            ntok[i] = -7;
        }
        check(strata::core::batch_spec_rows(2, 3, 7, rows, ntok) == 0, "cap below the slots: no rows");
        check(strata::core::batch_spec_rows(0, 1, 7, rows, ntok) == 0, "cap 0: no rows");
        check(strata::core::batch_spec_rows(kCap, 0, 7, rows, ntok) == 0, "no active slot: no rows");
        check(strata::core::batch_spec_rows(kCap, 3, 7, nullptr, ntok) == 0, "no row array: no rows");
        check(strata::core::batch_spec_rows(kCap, 3, 7, rows, nullptr) == 0, "no length array: no rows");
        bool untouched = true;
        for (int i = 0; i < kCap; ++i)
            if (rows[i] != -7 || ntok[i] != -7) untouched = false;
        check(untouched, "a refused split leaves both arrays alone");
    }
    {   // every shape the window can hold, checked against the layout law
        for (int cap = 1; cap <= kCap; ++cap) {
            for (int na = 1; na <= kCap; ++na) {
                for (int md = 0; md <= kCap; ++md) {
                    int rows[kStorageCap], ntok[kStorageCap];
                    const int S = strata::core::batch_spec_rows(cap, na, md, rows, ntok);
                    if (cap < na) {
                        check(S == 0, "cap below the slots: no window at all");
                        continue;
                    }
                    const int most = na * (1 + md) < cap ? na * (1 + md) : cap;
                    check(S == most, "the rows used are min(cap, slots * (1 + max_drafts))");
                    int sum = 0;
                    bool ok_len = true, mono = true, shape = true, min_k = true, spread = true;
                    for (int a = 0; a < na; ++a) {
                        sum += ntok[a];
                        if (ntok[a] < 1 || ntok[a] > 1 + md) ok_len = false;
                        if (a > 0 && ntok[a] > ntok[a - 1]) mono = false;
                        if (ntok[a] - 1 < strata::core::batch_spec_draft_budget(cap, na, md)) min_k = false;
                    }
                    if (ntok[0] - ntok[na - 1] > 1) spread = false;
                    int t = 0;
                    for (int a = 0; a < na; ++a)
                        for (int j = 0; j < ntok[a]; ++j)
                            if (rows[t++] != a) shape = false;
                    check(sum == S, "the group lengths add up to the window's rows");
                    check(ok_len, "a group is between one row and 1 + max_drafts");
                    check(mono, "the longer groups come first (round-robin, from the left)");
                    check(spread, "no slot is short by more than one row of another slot");
                    check(min_k, "every slot gets at least batch_spec_draft_budget proposals");
                    check(shape && t == S, "the rows are one contiguous group per slot");
                    int64_t pos[kStorageCap];
                    positions(rows, S, pos);
                    check(strata::core::batch_rows_error(rows, S, pos, na, kCap, 0, false, kCap) == nullptr,
                          "the split passes the layout law on one stage");
                    check(strata::core::batch_rows_error(rows, S, pos, na, kCap, 0, true, kCap) == nullptr,
                          "the split passes it across a layer split's hand-off too");
                }
            }
        }
    }

    // ---- the 16-row window is fully usable; only capacities above storage clamp to 16 ----
    if (kWideWindow) {
        int rows[kStorageCap], ntok[kStorageCap];
        for (int i = 0; i < kStorageCap; ++i) rows[i] = ntok[i] = -7;
        const int S = strata::core::batch_spec_rows(kStorageCap, 5, 7, rows, ntok);
        check(S == kStorageCap, "cap 16: the full opt-in window is honored");
        check(ntok[0] == 4 && ntok[1] == 3 && ntok[2] == 3 && ntok[3] == 3 && ntok[4] == 3,
              "cap 16, 5 slots: 11 proposal rows split 3/2/2/2/2");
        bool valid_ids = true, shape = true;
        int t = 0;
        for (int a = 0; a < 5; ++a) {
            if (ntok[a] < 1 || ntok[a] > 1 + 7) valid_ids = false;
            for (int j = 0; j < ntok[a]; ++j)
                if (rows[t++] != a) shape = false;
        }
        check(valid_ids && t == S, "cap 16: all entries map to one of the 5 selected slots");
        check(shape, "cap 16: each slot's rows remain one contiguous group in slot order");
        int64_t pos[kStorageCap];
        positions(rows, S, pos);
        check(strata::core::batch_rows_error(rows, S, pos, 5, kStorageCap, 0, false, kStorageCap) == nullptr,
              "cap 16 split passes the verifier's row-layout law");
        bool past_untouched = true;
        for (int i = 5; i < kStorageCap; ++i)
            if (ntok[i] != -7) past_untouched = false;
        check(past_untouched, "cap 16: unused slot-length entries remain untouched");

        int wide[kStorageCap], wide_n[kStorageCap];
        for (int i = 0; i < kStorageCap; ++i) wide[i] = wide_n[i] = -7;
        const int Sw = strata::core::batch_spec_rows(32, 5, 7, wide, wide_n);
        bool same = Sw == S;
        for (int r = 0; same && r < S; ++r)
            if (wide[r] != rows[r]) same = false;
        for (int a = 0; same && a < 5; ++a)
            if (wide_n[a] != ntok[a]) same = false;
        check(same, "cap above storage clamps to the supported 16-row window");
    }
    else { std::fputs("skip: the cap-16 window plan checks needs a wide build (kVerifyMaxT == 16)\n", stderr); }
    {   // More selected slots than the supplied cap is refused without touching either output.
        int rows[kStorageCap], ntok[kStorageCap];
        for (int i = 0; i < kStorageCap; ++i) rows[i] = ntok[i] = -7;
        check(strata::core::batch_spec_rows(kCap, kCap + 1, 7, rows, ntok) == 0,
              "cap 8, 9 slots: refused");
        check(strata::core::batch_spec_rows(kStorageCap, kStorageCap + 1, 7, rows, ntok) == 0,
              "cap 16, 17 slots: refused");
        bool untouched = true;
        for (int i = 0; i < kStorageCap; ++i)
            if (rows[i] != -7 || ntok[i] != -7) untouched = false;
        check(untouched, "too many slots: both output arrays stay untouched");
    }
    {   // Every cap above storage capacity behaves like cap 16, never writes beyond its output arrays.
        for (int cap = kStorageCap + 1; cap <= 2 * kStorageCap; ++cap) {
            for (int na = 1; na <= kStorageCap; ++na) {
                for (int md = 0; md <= kStorageCap; ++md) {
                    int wide[kStorageCap], wide_n[kStorageCap], at16[kStorageCap], at16_n[kStorageCap];
                    for (int i = 0; i < kStorageCap; ++i)
                        wide[i] = wide_n[i] = at16[i] = at16_n[i] = -7;
                    const int Sw = strata::core::batch_spec_rows(cap, na, md, wide, wide_n);
                    const int S16 = strata::core::batch_spec_rows(kStorageCap, na, md, at16, at16_n);
                    bool same = Sw == S16 && Sw <= kStorageCap;
                    for (int t = 0; same && t < S16; ++t)
                        if (wide[t] != at16[t]) same = false;
                    for (int a = 0; same && a < na; ++a)
                        if (wide_n[a] != at16_n[a]) same = false;
                    check(same, "cap above storage splits exactly like cap 16");
                    if (Sw >= 1 && Sw <= kStorageCap) {
                        int64_t pos[kStorageCap];
                        positions(wide, Sw, pos);
                        check(strata::core::batch_rows_error(wide, Sw, pos, na, kStorageCap, 0, false,
                                                             kStorageCap) == nullptr,
                              "the clamped 16-row window passes the layout law");
                    }
                }
            }
        }
    }

    // ---- the accept rule for one slot's group ----
    const std::vector<int32_t> no_eos;
    const std::vector<int32_t> eos_ids = {2, 99};
    {   // every proposal accepted: the whole group is kept
        const std::vector<int32_t> pick = {5, 6, 7, 8}, drafts = {5, 6, 7};
        const auto r = accept(pick, drafts, 4, 10, 1024, 0, 100, no_eos, false);
        check_accept(r, pick, 10, "all drafts accepted: invariants");
        check(r.keep == 4, "all drafts accepted: keep is the whole group");
        check(r.emitted[0] == 5 && r.emitted[1] == 6 && r.emitted[2] == 7 && r.emitted[3] == 8,
              "all drafts accepted: the emitted tokens are the target's picks");
        check(r.anchor == 8 && r.pos_after == 14, "all drafts accepted: the anchor and position follow the last token");
        check(!r.eos && !r.capped, "all drafts accepted: neither eos nor a bound cut the group");
    }
    {   // a slot with no proposals at all: generate.cpp passes a null drafts array for a cold or failed drafter,
        // and the slot still emits the target's own pick as its next anchor
        const std::vector<int32_t> pick = {5, 6, 7, 8};
        const auto r = strata::core::batch_spec_accept(pick.data(), nullptr, 4, 4, 10, 1024, 0, 100,
                                                       no_eos.data(), 0, false);
        check_accept(r, pick, 10, "no proposals: invariants");
        check(r.keep == 1, "no proposals: keep is the anchor row alone");
        check(r.emitted[0] == 5 && r.anchor == 5, "no proposals: the target's pick is the next anchor");
        check(r.pos_after == 11, "no proposals: one row's position");
        check(!r.eos && !r.capped, "no proposals: neither a bound nor eos fires");
    }
    {   // the first proposal is wrong: only the target's own pick is kept, and it is the next anchor
        const std::vector<int32_t> pick = {5, 6, 7, 8}, drafts = {9, 6, 7};
        const auto r = accept(pick, drafts, 4, 10, 1024, 0, 100, no_eos, false);
        check_accept(r, pick, 10, "first draft wrong: invariants");
        check(r.keep == 1, "first draft wrong: keep 1");
        check(r.emitted[0] == 5 && r.anchor == 5, "first draft wrong: the anchor is the target's pick, not the draft");
        check(r.pos_after == 11, "first draft wrong: one row's position");
        check(!r.eos && !r.capped, "first draft wrong: a mismatch is not a bound");
    }
    {   // one match, then a wrong proposal: the run stops at the first mismatch, whatever matches later
        const std::vector<int32_t> pick = {5, 6, 7, 8}, drafts = {5, 0, 7};
        const auto r = accept(pick, drafts, 4, 10, 1024, 0, 100, no_eos, false);
        check_accept(r, pick, 10, "second draft wrong: invariants");
        check(r.keep == 2 && r.anchor == 6 && r.pos_after == 12, "second draft wrong: keep 2, the match is kept");
        const std::vector<int32_t> d3 = {5, 6, 1};
        const auto r3 = accept(pick, d3, 4, 10, 1024, 0, 100, no_eos, false);
        check_accept(r3, pick, 10, "third draft wrong: invariants");
        check(r3.keep == 3 && r3.anchor == 7, "third draft wrong: keep 3");
    }
    {   // a plain batch row: one row, nothing proposed
        const std::vector<int32_t> pick = {7}, drafts;
        const auto r = accept(pick, drafts, 3, 20, 1024, 0, 100, no_eos, false);
        check_accept(r, pick, 20, "ntok 1: invariants");
        check(r.keep == 1 && r.emitted[0] == 7, "ntok 1: keep 1 and the target's pick");
        check(r.anchor == 7 && r.pos_after == 21, "ntok 1: the anchor is the pick, not the token that was on the row");
        check(!r.eos && !r.capped, "ntok 1: nothing cut it");
        const std::vector<int32_t> empty_pick;
        const auto e = accept(empty_pick, drafts, 3, 20, 1024, 0, 100, no_eos, false);
        check(e.keep == 0 && e.pos_after == 20 && e.anchor == 3, "ntok 0: nothing is kept, the slot's token stays the anchor");
        check(!e.eos && !e.capped, "ntok 0: no flags");
    }
    {   // eos inside the proposal run
        const std::vector<int32_t> pick = {5, 6, 99, 8}, d_stop = {5, 6, 8}, d_eos = {5, 6, 99};
        const auto r = accept(pick, d_stop, 4, 10, 1024, 0, 100, eos_ids, false);
        check_accept(r, pick, 10, "eos at the row the run stops on: invariants");
        check(r.keep == 3 && r.anchor == 99 && r.pos_after == 13, "eos at the row the run stops on: keep up to it");
        check(r.eos && !r.capped, "eos at the row the run stops on: eos, not a bound");
        const auto r2 = accept(pick, d_eos, 4, 10, 1024, 0, 100, eos_ids, false);
        check_accept(r2, pick, 10, "eos proposed and confirmed: invariants");
        check(r2.keep == 3, "eos proposed and confirmed: the group is truncated at the eos row");
        check(r2.eos && r2.emitted[2] == 99, "eos proposed and confirmed: the eos token is emitted");
    }
    {   // eos on the anchor row: the group ends before any proposal
        const std::vector<int32_t> pick = {99, 6, 7}, drafts = {99, 6};
        const auto r = accept(pick, drafts, 4, 10, 1024, 0, 100, eos_ids, false);
        check_accept(r, pick, 10, "eos on the anchor row: invariants");
        check(r.keep == 1 && r.emitted[0] == 99 && r.anchor == 99, "eos on the anchor row: only that token");
        check(r.eos && r.pos_after == 11, "eos on the anchor row: eos, one row of position");
    }
    {   // a bound that drops the proposal also drops the eos that was behind it
        const std::vector<int32_t> pick = {5, 6, 99}, drafts = {5, 6};
        const auto r = accept(pick, drafts, 4, 10, 1024, 0, 1, eos_ids, false);
        check_accept(r, pick, 10, "eos behind a --max-new bound: invariants");
        check(r.keep == 1 && r.capped, "eos behind a --max-new bound: the bound truncates to the anchor row");
        check(!r.eos, "eos behind a --max-new bound: no eos token was emitted");
    }
    {   // --max-new: no proposal is taken past the limit
        const std::vector<int32_t> pick = {5, 6, 7, 8}, drafts = {5, 6, 7};
        const auto r = accept(pick, drafts, 4, 10, 1024, 8, 10, no_eos, false);
        check_accept(r, pick, 10, "max_new room for 2: invariants");
        check(r.keep == 2 && r.anchor == 6 && r.capped, "max_new room for 2: the proposals past it are dropped");
        const auto r1 = accept(pick, drafts, 4, 10, 1024, 9, 10, no_eos, false);
        check_accept(r1, pick, 10, "max_new room for 1: invariants");
        check(r1.keep == 1 && r1.capped, "max_new room for 1: the anchor row is still emitted");
        const auto r0 = accept(pick, drafts, 4, 10, 1024, 5, 5, no_eos, false);
        check(r0.keep == 1 && r0.capped && !r0.eos, "max_new already reached: the anchor row, capped, not eos");
        const auto rf = accept(pick, drafts, 4, 10, 1024, 0, 100, no_eos, false);
        check(rf.keep == 4 && !rf.capped, "room for all 4: no bound");
    }
    {   // max_context: a proposal needs the row it emits and the next anchor row to fit
        const std::vector<int32_t> pick = {5, 6, 7, 8}, drafts = {5, 6, 7};
        const int64_t lim[4] = {105, 104, 103, 102};
        const int want[4] = {4, 3, 2, 1};
        for (int i = 0; i < 4; ++i) {
            const auto r = accept(pick, drafts, 4, 100, lim[i], 0, 100, no_eos, false);
            char why[96];
            std::snprintf(why, sizeof why, "position 100, max_context %lld: keep %d", (long long) lim[i], want[i]);
            check_accept(r, pick, 100, why);
            check(r.keep == want[i], why);
            check(r.capped == (want[i] < 4), why);
        }
    }
    {   // a stopped slot does not take its proposals, but still emits its own pick
        const std::vector<int32_t> pick = {5, 6}, drafts = {5};
        const auto r = accept(pick, drafts, 4, 10, 1024, 0, 100, no_eos, true);
        check_accept(r, pick, 10, "stop: invariants");
        check(r.keep == 1 && r.emitted[0] == 5 && r.anchor == 5, "stop: the target's own pick, no proposal");
        check(!r.capped && !r.eos, "stop: not capped, not eos");
        const std::vector<int32_t> eos_pick = {99, 6};
        const auto re = accept(eos_pick, drafts, 4, 10, 1024, 0, 100, eos_ids, true);
        check_accept(re, eos_pick, 10, "stop on an eos token: invariants");
        check(re.keep == 1 && re.eos && !re.capped, "stop on an eos token: the token itself is eos");
        const auto rl = accept(pick, drafts, 4, 10, 1024, 5, 5, no_eos, true);
        check(rl.keep == 1 && !rl.capped, "stop with max_new reached: stop is the reason, not the bound");
    }
    {   // parity with the --batch-mtp rule of generate.cpp, case by case
        const int32_t pick0s[] = {7, 99};      // the anchor row's pick: plain, or an eos token
        const int32_t pick1s[] = {8, 99};      // the second row's pick: plain, or an eos token
        const int32_t drafts[] = {7, 8};       // the proposal: confirmed, or not
        const int64_t bounds[][4] = {
            {0, 100, 5, 1024},   // room everywhere
            {0, 1, 5, 1024},     // --max-new room for one token
            {99, 100, 5, 1024},  // --max-new room for one more token
            {0, 100, 100, 102},  // no room for the proposal's next anchor row
            {0, 100, 100, 103},  // exactly enough room
            {0, 100, 100, 104},  // room to spare
        };
        long cases = 0;
        for (int32_t p0t : pick0s) {
            for (int32_t p1t : pick1s) {
                for (int32_t d : drafts) {
                    for (int64_t b = 0; b < (int64_t) (sizeof bounds / sizeof bounds[0]); ++b) {
                        for (int s = 0; s < 2; ++s) {
                            const std::vector<int32_t> pick = {p0t, p1t}, draft = {d};
                            const int64_t produced = bounds[b][0], max_new = bounds[b][1], p0 = bounds[b][2],
                                            limit = bounds[b][3];
                            const bool stop = s != 0;
                            const auto r = accept(pick, draft, 3, p0, limit, produced, max_new, eos_ids, stop);
                            ++cases;
                            check_accept(r, pick, p0, "batch-mtp parity: invariants");
                            check(r.keep == mtp_keep(p0t, d, eos_ids, p0, limit, produced, max_new, stop),
                                  "batch-mtp parity: the same keep as generate.cpp:13250");
                            check(r.emitted[0] == p0t, "batch-mtp parity: outb[first[t]] is emitted first");
                            check(r.keep < 2 || r.emitted[1] == p1t, "batch-mtp parity: the second token is outb[first[t]+1]");
                            check(r.anchor == (r.keep == 2 ? p1t : p0t), "batch-mtp parity: sl.x is the last emitted token");
                            check(r.pos_after == p0 + r.keep, "batch-mtp parity: sl.p advances by keep");
                            check(r.eos == (is_eos(p0t, eos_ids) || (r.keep == 2 && is_eos(p1t, eos_ids))),
                                  "batch-mtp parity: eos is what the emitted tokens say");
                            const bool cap_hit = p0t == d && !stop && (produced + 2 > max_new || p0 + 3 > limit);
                            check(r.capped == cap_hit, "batch-mtp parity: capped only when a bound dropped the proposal");
                        }
                    }
                }
            }
        }
        check(cases == 96, "the parity sweep covered all 96 two-row cases (2 picks x 2 picks x 2 drafts x 6 bounds x stop/not)");
    }

    if (g_fail != 0) {
        std::fprintf(stderr, "batch_spec_test: %d of %ld checks FAILED\n", g_fail, g_checks);
        return EXIT_FAILURE;
    }
    std::printf("batch_spec_test: ok (%ld checks)\n", g_checks);
    return EXIT_SUCCESS;
}
