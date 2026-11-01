// tests/core/batch_spec_slots_test.cpp - the decisions the batch loop makes per slot with DFlash2 drafts, on the
// row split of batch_spec.hpp (include/strata/core/batch_spec_slots.hpp):
//
//   1. how many proposals each slot gets when only some of the active slots have a draft ready: every selected slot
//      keeps its anchor row, the rows the window has left go round-robin to the READY slots alone, bounded by
//      max_drafts, and the split stays a layout the window's own law accepts (batch_rows_error);
//   2. the runtime default remains 8 rows while storage permits a 16-row DFlash2 batch window; wider inputs are
//      clamped only at the hard storage ceiling, and the host planner keeps every write inside its arrays;
//   3. the tap rows a slot's committed prefix writes for its next draft: [first, first + keep), contiguous, inside
//      the slot's own group, so a row the target rejected is never staged.
//
// Header-only, CPU: no verifier, no drafter, no GPU.
#include "strata/core/batch_spec_slots.hpp"

#include "strata/core/batch_rows.hpp"
#include "strata/core/batch_spec.hpp"
#include "strata/kernels/verify_kernels.hpp"

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <vector>

namespace {
int g_fail = 0;
long g_checks = 0;
void check(bool ok, const char* what) {
    ++g_checks;
    if (ok) return;
    std::fprintf(stderr, "FAIL: %s\n", what);
    ++g_fail;
}

constexpr int kCap = strata::kernels::kVerifyDefaultMaxT;  // the preserved runtime window default
constexpr int kStorageCap = strata::kernels::kVerifyMaxT;  // all host arrays cover the 16-row hard ceiling
// The 16-row window only exists in a wide build (-DSTRATA_WINDOW_MAX_T=16; the HIP engine builds it by default).
// A narrow build has 8-row storage, so the 16-row plans below clamp to 8 and their checks are skipped, not passed.
constexpr bool kWideWindow = strata::kernels::kVerifyMaxT >= 16;

// The positions a window gives the rows: each slot's group at consecutive positions of its own, every slot starting
// somewhere different so a mix-up of the slots shows up (as in tests/core/batch_spec_test.cpp).
void positions(const int* rows, int S, int64_t* pos) {
    const int64_t start[kStorageCap] = {0, 4, 9, 20, 31, 44, 55, 66};
    int64_t next[kStorageCap];
    for (int a = 0; a < kStorageCap; ++a) next[a] = start[a];
    for (int t = 0; t < S; ++t) pos[t] = next[rows[t]]++;
}
}  // namespace

int main() {
    // ---- the proposal split with a per-slot ready mask ----
    {   // every slot ready: the same split batch_spec.hpp gives (the new overload must not move the numbers)
        int rows[kCap], ntok[kCap];
        const bool ready[kCap] = {true, true, true};
        const int S = strata::core::batch_spec_slot_rows(kCap, 3, ready, 7, rows, ntok);
        check(S == 8, "all ready, 3 slots, 8 rows: the window is full");
        check(ntok[0] == 3 && ntok[1] == 3 && ntok[2] == 2, "all ready: the split is batch_spec_rows' 3 / 3 / 2");
        const int want[kCap] = {0, 0, 0, 1, 1, 1, 2, 2};
        bool shape = true;
        for (int t = 0; t < S; ++t)
            if (rows[t] != want[t]) shape = false;
        check(shape, "all ready: each slot's rows are one contiguous group, the slots in order");
        int64_t pos[kCap];
        positions(rows, S, pos);
        check(strata::core::batch_rows_error(rows, S, pos, 3, kCap, 0, false, kCap) == nullptr,
              "all ready: the split is a layout the verifier accepts");
    }
    {   // the middle slot has no draft ready (newly admitted, cold drafter, a failed round): it keeps its anchor row
        // alone and the other two share the window's five spare rows round-robin
        int rows[kCap], ntok[kCap];
        const bool ready[kCap] = {true, false, true};
        const int S = strata::core::batch_spec_slot_rows(kCap, 3, ready, 7, rows, ntok);
        check(S == 8, "one slot unready, 3 slots, 8 rows: the window is full");
        check(ntok[1] == 1, "an unready slot keeps its anchor row alone");
        check(ntok[0] == 4 && ntok[2] == 3, "the spare rows go round-robin to the ready slots: 3 and 2 proposals");
        check(ntok[0] - 1 == (8 - 3 + 1) / 2 && ntok[2] - 1 == (8 - 3) / 2,
              "the ready slots split the spare rows as evenly as whole rows allow");
        int64_t pos[kCap];
        positions(rows, S, pos);
        check(strata::core::batch_rows_error(rows, S, pos, 3, kCap, 0, false, kCap) == nullptr,
              "one slot unready: the split passes the layout law");
        check(strata::core::batch_rows_error(rows, S, pos, 3, kCap, 0, true, kCap) == nullptr,
              "... and across a layer split's hand-off");
    }
    {   // no slot ready: plain --batch, one row per slot
        int rows[kCap], ntok[kCap];
        const bool ready[kCap] = {false, false, false};
        const int S = strata::core::batch_spec_slot_rows(kCap, 3, ready, 7, rows, ntok);
        check(S == 3, "none ready: one row per slot");
        check(ntok[0] == 1 && ntok[1] == 1 && ntok[2] == 1, "none ready: no slot has a proposal");
        check(rows[0] == 0 && rows[1] == 1 && rows[2] == 2, "none ready: the rows are the slots in order");
    }
    {   // max_drafts bounds the ready slots, so a window can stay half empty
        int rows[kCap], ntok[kCap];
        const bool ready2[kCap] = {true, true, true};
        const int S1 = strata::core::batch_spec_slot_rows(kCap, 2, ready2, 1, rows, ntok);
        check(S1 == 4, "2 slots ready, max_drafts 1: 4 rows, the window is not full");
        check(ntok[0] == 2 && ntok[1] == 2, "max_drafts 1: every ready slot gets exactly one proposal");
        const bool ready1[kCap] = {true, false, true};
        const int S2 = strata::core::batch_spec_slot_rows(kCap, 3, ready1, 1, rows, ntok);
        check(S2 == 5, "3 slots, 2 ready, max_drafts 1: 5 rows (3 anchors + 2 proposals)");
        check(ntok[0] == 2 && ntok[1] == 1 && ntok[2] == 2, "max_drafts 1 with a cold slot: 2 / 1 / 2");
        const int S3 = strata::core::batch_spec_slot_rows(kCap, 3, ready1, 0, rows, ntok);
        check(S3 == 3 && ntok[0] == 1 && ntok[1] == 1 && ntok[2] == 1, "max_drafts 0 is plain --batch, ready mask or not");
        const int S4 = strata::core::batch_spec_slot_rows(kCap, 3, ready1, -2, rows, ntok);
        check(S4 == 3, "a negative max_drafts is 0 proposals");
    }
    {   // a window with no spare row gives every slot its anchor row alone
        int rows[kCap], ntok[kCap];
        const bool ready[kCap] = {true, true, true, true};
        const int S = strata::core::batch_spec_slot_rows(4, 4, ready, 7, rows, ntok);
        check(S == 4, "cap == slots: the window has no proposal row");
        check(ntok[0] == 1 && ntok[1] == 1 && ntok[2] == 1 && ntok[3] == 1, "cap == slots: anchor rows only");
    }
    {   // a null mask is every slot ready (the caller's plain --batch spec path)
        int rows[kCap], ntok[kCap];
        const int S = strata::core::batch_spec_slot_rows(kCap, 3, nullptr, 7, rows, ntok);
        const int S_ref = strata::core::batch_spec_rows(kCap, 3, 7, rows, ntok);
        check(S == S_ref && S == 8, "no mask given: every slot is ready, batch_spec_rows' numbers");
    }
    {   // refusals: no window, and nothing written
        int rows[kCap], ntok[kCap];
        for (int i = 0; i < kCap; ++i) {
            rows[i] = -7;
            ntok[i] = -7;
        }
        const bool ready[kCap] = {true, true, true};
        check(strata::core::batch_spec_slot_rows(2, 3, ready, 7, rows, ntok) == 0, "cap below the slots: no rows");
        check(strata::core::batch_spec_slot_rows(0, 1, ready, 7, rows, ntok) == 0, "cap 0: no rows");
        check(strata::core::batch_spec_slot_rows(kCap, 0, ready, 7, rows, ntok) == 0, "no selected slot: no rows");
        check(strata::core::batch_spec_slot_rows(kCap, 3, ready, 7, nullptr, ntok) == 0, "no row array: no rows");
        check(strata::core::batch_spec_slot_rows(kCap, 3, ready, 7, rows, nullptr) == 0, "no length array: no rows");
        bool untouched = true;
        for (int i = 0; i < kCap; ++i)
            if (rows[i] != -7 || ntok[i] != -7) untouched = false;
        check(untouched, "a refused split leaves both arrays alone");
    }
    {   // every shape the window can hold, with every ready mask, checked against the layout law and the rules
        for (int cap = 1; cap <= kCap; ++cap) {
            for (int ns = 1; ns <= kCap; ++ns) {
                for (int md = 0; md <= kCap; ++md) {
                    const int masks = 1 << ns;
                    for (int m = 0; m < masks; ++m) {
                        bool ready[kCap];
                        int n_ready = 0;
                        for (int a = 0; a < ns; ++a) {
                            ready[a] = (m & (1 << a)) != 0;
                            if (ready[a]) ++n_ready;
                        }
                        int rows[kCap], ntok[kCap];
                        const int S = strata::core::batch_spec_slot_rows(cap, ns, ready, md, rows, ntok);
                        if (cap < ns) {
                            check(S == 0, "cap below the slots: no window at all");
                            continue;
                        }
                        const int spare = cap - ns;
                        const long long most = (long long) n_ready * md;
                        const int extra = (int) (spare < most ? spare : most);
                        check(S == ns + extra, "the rows are the anchors plus the ready slots' proposals the window holds");
                        check(S <= cap, "never past the window's rows");
                        bool ok_len = true, unready_one = true, shape = true, fair_spread = true, fair_order = true;
                        int kmin = kCap + 1, kmax = -1;
                        int t = 0;
                        int last_ready = -1;
                        for (int a = 0; a < ns; ++a) {
                            const int k = ntok[a] - 1;
                            if (k < 0 || k > md) ok_len = false;
                            if (!ready[a] && ntok[a] != 1) unready_one = false;
                            if (ready[a]) {
                                if (k < kmin) kmin = k;
                                if (k > kmax) kmax = k;
                                if (last_ready >= 0 && k > ntok[last_ready] - 1) fair_order = false;
                                last_ready = a;
                            }
                            for (int j = 0; j < ntok[a]; ++j)
                                if (rows[t++] != a) shape = false;
                        }
                        if (kmax - kmin > 1) fair_spread = false;
                        check(shape && t == S, "the rows are one contiguous group per slot, the slots in order");
                        check(ok_len, "a group is between one row and 1 + max_drafts");
                        check(unready_one, "a slot with no draft ready keeps its anchor row alone");
                        check(fair_spread, "no ready slot is short by more than one proposal of another");
                        check(fair_order, "the longer ready groups come first (round-robin, from the left)");
                        if (n_ready > 0 && md > 0) {
                            const int floor_k = (spare / n_ready) < md ? spare / n_ready : md;
                            check(kmin == floor_k, "the ready slots all get at least the even share of the spare rows");
                        }
                        int64_t pos[kCap];
                        positions(rows, S, pos);
                        check(strata::core::batch_rows_error(rows, S, pos, ns, kCap, 0, false, kCap) == nullptr,
                              "the split passes the layout law on one stage");
                        check(strata::core::batch_rows_error(rows, S, pos, ns, kCap, 0, true, kCap) == nullptr,
                              "the split passes it across a layer split's hand-off too");
                    }
                }
            }
        }
    }

    // ---- the 16-row hard ceiling: wider caps are clamped to storage, not to the 8-row runtime default ----
    {   // cap 32, five selected slots with one cold: the 16-row plan stays inside every caller array
        int rows[kStorageCap], ntok[kStorageCap];
        for (int i = 0; i < kStorageCap; ++i) {
            rows[i] = -7;
            ntok[i] = -7;
        }
        const bool ready[kStorageCap] = {true, true, false, true, true};
        const int S = strata::core::batch_spec_slot_rows(32, 5, ready, 7, rows, ntok);
        check(S == kStorageCap, "cap 32 clamps to the 16-row storage ceiling");
        bool valid_ids = true;
        for (int t = 0; t < S && t < kStorageCap; ++t)
            if (rows[t] < 0 || rows[t] >= 5) valid_ids = false;
        check(valid_ids, "cap 32: every returned row id is one of the 5 selected slots");
        int sum = 0;
        for (int a = 0; a < 5; ++a) sum += ntok[a];
        check(sum == S, "cap 32: the group lengths add up to the returned row count");
        bool past_untouched = true;
        for (int t = S; t < kStorageCap; ++t)
            if (rows[t] != -7) past_untouched = false;
        for (int a = 5; a < kStorageCap; ++a)
            if (ntok[a] != -7) past_untouched = false;
        check(past_untouched, "cap 32: neither array is written past the returned rows");
        if (S != kStorageCap) {
            check(false, "cap 32: the count stays inside the caller's arrays before the cross-checks run");
        } else {
            int64_t pos[kStorageCap];
            positions(rows, S, pos);
            check(strata::core::batch_rows_error(rows, S, pos, 5, kStorageCap, 0, false, kStorageCap) == nullptr,
                  "cap 32: the split passes the layout law at the 16-row ceiling");
            bool len_ok = true, shape = true, cold_one = true, fair = true;
            int t = 0, kmin = kStorageCap + 1, kmax = -1, last_ready = -1;
            for (int a = 0; a < 5; ++a) {
                const int k = ntok[a] - 1;
                if (ntok[a] < 1 || k > 7) len_ok = false;
                if (!ready[a] && ntok[a] != 1) cold_one = false;
                if (ready[a]) {
                    if (k < kmin) kmin = k;
                    if (k > kmax) kmax = k;
                    if (last_ready >= 0 && k > ntok[last_ready] - 1) fair = false;
                    last_ready = a;
                }
                for (int j = 0; j < ntok[a]; ++j)
                    if (rows[t++] != a) shape = false;
            }
            check(len_ok, "cap 32: every slot's group still holds 1..1+max_drafts rows");
            check(shape && t == S, "cap 32: the groups stay contiguous and ordered");
            check(cold_one, "cap 32: the slot with no draft ready keeps its anchor row alone");
            check(fair && kmax - kmin <= 1, "cap 32: the ready slots still share the spare rows fairly");
        }
        // A cap above storage has the same plan as the hard ceiling.
        int rows16[kStorageCap], ntok16[kStorageCap];
        const int S16 = strata::core::batch_spec_slot_rows(kStorageCap, 5, ready, 7, rows16, ntok16);
        bool same = S == S16;
        for (int t = 0; same && t < S16; ++t)
            if (rows[t] != rows16[t]) same = false;
        for (int a = 0; same && a < 5; ++a)
            if (ntok[a] != ntok16[a]) same = false;
        check(same, "cap 32 is cap kVerifyMaxT: the hard storage ceiling is honoured");
    }
    {   // More selected slots than the 16-row hard ceiling: refuse without touching either output array.
        bool ready17[17];
        for (bool& value : ready17) value = true;
        int rows[kStorageCap], ntok[kStorageCap];
        for (int i = 0; i < kStorageCap; ++i) rows[i] = ntok[i] = -7;
        check(strata::core::batch_spec_slot_rows(kStorageCap, 17, ready17, 7, rows, ntok) == 0,
              "cap 16, 17 selected: too many anchor rows are refused");
        check(strata::core::batch_spec_slot_rows(32, 17, ready17, 7, rows, ntok) == 0,
              "cap 32 still refuses 17 selected after hard-ceiling clamp");
        bool untouched = true;
        for (int i = 0; i < kStorageCap; ++i)
            if (rows[i] != -7 || ntok[i] != -7) untouched = false;
        check(untouched, "more slots than the storage ceiling holds: both arrays stay alone");
        const int S = strata::core::batch_spec_slot_rows(kStorageCap, kStorageCap, ready17, 7, rows, ntok);
        bool full = S == kStorageCap;
        for (int a = 0; full && a < kStorageCap; ++a)
            if (ntok[a] != 1) full = false;
        check(full, "16 selected under a 16-row window use all rows for anchors");
    }
    {   // Every small ready-mask shape above storage splits exactly as the same shape at the 16-row ceiling.
        for (int cap = kStorageCap + 1; cap <= 2 * kStorageCap; ++cap) {
            for (int ns = 1; ns <= kCap; ++ns) {
                const int masks = 1 << ns;
                for (int md = 0; md <= kCap; ++md) {
                    for (int m = 0; m < masks; ++m) {
                        bool ready[kStorageCap] = {};
                        for (int a = 0; a < ns; ++a) ready[a] = (m & (1 << a)) != 0;
                        int wide[kStorageCap], wide_n[kStorageCap], at16[kStorageCap], at16_n[kStorageCap];
                        for (int i = 0; i < kStorageCap; ++i)
                            wide[i] = wide_n[i] = at16[i] = at16_n[i] = -7;
                        const int Sw = strata::core::batch_spec_slot_rows(cap, ns, ready, md, wide, wide_n);
                        const int S16 = strata::core::batch_spec_slot_rows(kStorageCap, ns, ready, md, at16, at16_n);
                        bool same = Sw == S16 && Sw <= kStorageCap;
                        for (int t = 0; same && t < S16; ++t)
                            if (wide[t] != at16[t]) same = false;
                        for (int a = 0; same && a < ns; ++a)
                            if (wide_n[a] != at16_n[a]) same = false;
                        check(same, "cap above kVerifyMaxT splits exactly as cap kVerifyMaxT does, every mask");
                        if (Sw >= 1 && Sw <= kStorageCap) {
                            int64_t pos[kStorageCap];
                            positions(wide, Sw, pos);
                            check(strata::core::batch_rows_error(wide, Sw, pos, ns, kStorageCap, 0, false,
                                                                 kStorageCap) == nullptr,
                                  "a clamped window passes the layout law");
                        }
                    }
                }
            }
        }
    }

    // ---- the tap rows a committed prefix stages: [first, first + keep) ----
    {   // a full window of 3 slots (ready mask all true, so groups 3 / 3 / 2) with keeps 2 / 3 / 1
        const int rows[kCap] = {0, 0, 0, 1, 1, 1, 2, 2};
        const int ntok[kCap] = {3, 3, 2};
        const int first[kCap] = {0, 3, 6};
        const int keep[kCap] = {2, 3, 1};
        const int S = 8;
        int lo[kCap], hi[kCap];
        bool all_ok = true, exact = true, inside = true, disjoint = true;
        bool covers[kCap] = {};
        for (int b = 0; b < 3; ++b) {
            if (!strata::core::df2_stage_rows(first[b], keep[b], ntok[b], S, &lo[b], &hi[b])) all_ok = false;
            if (lo[b] != first[b] || hi[b] != first[b] + keep[b]) exact = false;
            for (int r = lo[b]; r < hi[b]; ++r) {
                if (rows[r] != b) inside = false;   // the staged rows are this slot's rows, contiguous
                if (covers[r]) disjoint = false;    // no row is staged for two slots
                covers[r] = true;
            }
            for (int r = first[b] + keep[b]; r < first[b] + ntok[b]; ++r)
                if (r >= lo[b] && r < hi[b]) inside = false;   // a rejected row of this slot is never staged
        }
        check(all_ok, "the staged ranges of a 3-slot window are legal");
        check(exact, "slot b's range is [first[b], first[b] + keep[b])");
        check(inside, "the range is the slot's own rows, contiguous, and stops at the first rejected row");
        check(disjoint, "no window row is staged for two slots");
        int n_covered = 0;
        for (int r = 0; r < S; ++r)
            if (covers[r]) ++n_covered;
        check(n_covered == 2 + 3 + 1, "the ranges cover exactly the committed rows");
    }
    {   // keep 0 (a group the accept rule emitted nothing for): an empty range, legal, staging nothing
        int lo = -1, hi = -1;
        check(strata::core::df2_stage_rows(4, 0, 3, kCap, &lo, &hi), "keep 0: a legal, empty range");
        check(lo == 4 && hi == 4, "keep 0: lo == hi, so it stages nothing");
        check(strata::core::df2_stage_rows(kCap, 0, 1, kCap, &lo, &hi), "keep 0 at the window's end is still inside it");
    }
    {   // refusals: past the group (a rejected row would be staged), past the window, and nonsense input
        int lo = -1, hi = -1;
        check(!strata::core::df2_stage_rows(0, 4, 3, kCap, &lo, &hi), "keep past the group: refused (a rejected row)");
        check(!strata::core::df2_stage_rows(6, 3, 3, kCap, &lo, &hi), "the range past the window's rows: refused");
        check(!strata::core::df2_stage_rows(-1, 1, 3, kCap, &lo, &hi), "a negative first row: refused");
        check(!strata::core::df2_stage_rows(0, -1, 3, kCap, &lo, &hi), "a negative keep: refused");
        check(!strata::core::df2_stage_rows(0, 1, 0, kCap, &lo, &hi), "a slot with no row in the window stages nothing legal?");
        check(!strata::core::df2_stage_rows(0, 1, 3, kCap, nullptr, &hi), "no lo out: refused");
        check(!strata::core::df2_stage_rows(0, 1, 3, kCap, &lo, nullptr), "no hi out: refused");
        check(lo == -1 && hi == -1, "a refused call leaves both outputs alone");
    }
    {   // the staging geometry over the split's own numbers: for every window shape, keep everything, keep the
        // anchor only, keep the middle - and the ranges always tile the committed rows inside each slot's group
        for (int cap = 1; cap <= kCap; ++cap) {
            for (int ns = 1; ns <= kCap; ++ns) {
                if (cap < ns) continue;
                const bool ready[kCap] = {true};
                for (int md = 0; md <= kCap; ++md) {
                    int rows[kCap], ntok[kCap];
                    const int S = strata::core::batch_spec_slot_rows(cap, ns, ready, md, rows, ntok);
                    int64_t pos[kCap];
                    positions(rows, S, pos);
                    if (strata::core::batch_rows_error(rows, S, pos, ns, kCap, 0, false, kCap) != nullptr) {
                        check(false, "the split the geometry test walks is a layout the verifier accepts");
                        continue;
                    }
                    int first[kCap];
                    for (int a = 0, t = 0; a < ns; ++a) {
                        first[a] = t;
                        t += ntok[a];
                    }
                    for (int mode = 0; mode < 3; ++mode) {
                        int keep[kCap];
                        for (int a = 0; a < ns; ++a)
                            keep[a] = mode == 0 ? ntok[a] : (mode == 1 ? 1 : (ntok[a] > 1 ? ntok[a] - 1 : 1));
                        bool covers[kCap] = {};
                        bool ok = true, inside = true, tiled = true;
                        for (int b = 0; b < ns; ++b) {
                            int lo = -1, hi = -1;
                            if (!strata::core::df2_stage_rows(first[b], keep[b], ntok[b], S, &lo, &hi)) ok = false;
                            for (int r = lo; r < hi; ++r) {
                                if (rows[r] != b) inside = false;
                                if (covers[r]) tiled = false;
                                covers[r] = true;
                            }
                        }
                        int n_covered = 0, sum_keep = 0;
                        for (int r = 0; r < S; ++r)
                            if (covers[r]) ++n_covered;
                        for (int b = 0; b < ns; ++b) sum_keep += keep[b];
                        check(ok && inside && tiled && n_covered == sum_keep,
                              "every window shape: the committed ranges are contiguous, per slot, and tile the committed rows");
                    }
                }
            }
        }
    }

    // ---- the opt-in 16-row DFlash window (RED until the verifier capacity is raised) ----
    if (kWideWindow) {
        using strata::core::df2_window_plan;
        constexpr int wide_cap = 16;
        int rows[wide_cap] = {}, ntok[wide_cap] = {};
        bool ready[wide_cap] = {};
        for (int i = 0; i < 4; ++i) ready[i] = true;
        const auto p = df2_window_plan(wide_cap, 4, 7, 0);
        const int S = strata::core::batch_spec_slot_rows(wide_cap, p.n_sel, ready, p.depth, rows, ntok);
        check(p.n_sel == 4 && p.depth == 3 && S == 16,
              "16 rows let all four slots carry three proposal rows each");
        bool full = true;
        for (int i = 0; i < 4; ++i) full = full && ntok[i] == 4;
        check(full, "the opt-in 16-row window stores four rows per selected slot");
    }
    else { std::fputs("skip: the opt-in 16-row DFlash window plan needs a wide build (kVerifyMaxT == 16)\n", stderr); }

    // ---- default DFlash windows remain 8 rows after the verifier storage ceiling grows to 16 ----
    {
        const int cap = strata::kernels::kVerifyDefaultMaxT;
        int rows[16] = {}, ntok[16] = {};
        bool ready[16] = {true, true, true, true};
        const auto p = strata::core::df2_window_plan(cap, 4, 7, 0);
        const int S = strata::core::batch_spec_slot_rows(cap, p.n_sel, ready, p.depth, rows, ntok);
        check(cap == 8 && p.n_sel == 4 && p.depth == 1 && S == 8,
              "default DFlash batch windows stay 8 rows (four slots get one proposal each)");
    }

    // ---- a pure row layout contract for future pipelined DFlash2 groups ----
    {
        using strata::core::df2_piped_group_rows;
        bool active[6] = {true, true, true, true, true, true};
        bool ready[6] = {true, true, true, true, true, true};
        int selected[16] = {}, rows[16] = {}, ntok[16] = {};
        auto p = df2_piped_group_rows(16, 6, active, ready, 7, 3, 4, selected, rows, ntok);
        if (kWideWindow) {
            check(p.n_sel == 4 && p.depth == 3 && p.rows == 16 && p.next == 2,
                  "piped row planner fits four 3-proposal groups and advances the rotation");
            const int want_sel[4] = {4, 5, 0, 1};
            bool shape_ok = true;
            for (int i = 0; i < 4; ++i) shape_ok = shape_ok && selected[i] == want_sel[i] && ntok[i] == 4;
            for (int r = 0; r < 16; ++r) shape_ok = shape_ok && rows[r] == want_sel[r / 4];
            check(shape_ok, "piped rows stay contiguous and map back to the rotating group-local slots");
            p = df2_piped_group_rows(16, 6, active, ready, 7, 3, p.next, selected, rows, ntok);
            check(p.n_sel == 4 && selected[0] == 2 && selected[1] == 3 && selected[2] == 4 && selected[3] == 5,
                  "piped row rotation gives waiting slots their turn");
        } else {
            std::fputs("skip: the 16-row pipelined DFlash2 group contract needs a wide build (kVerifyMaxT == 16)\n", stderr);
        }

        bool cold[4] = {true, true, true, true};
        bool has_draft[4] = {true, false, true, false};
        int selected4[16] = {}, rows4[16] = {}, ntok4[16] = {};
        p = df2_piped_group_rows(8, 4, cold, has_draft, 7, 1, 0, selected4, rows4, ntok4);
        check(p.n_sel == 4 && p.depth == 1 && p.rows == 6 && ntok4[0] == 2 && ntok4[1] == 1 &&
                  ntok4[2] == 2 && ntok4[3] == 1,
              "piped cold slots keep anchor rows and lend proposal rows only to ready slots");
        bool untouched_active[1] = {true};
        int untouched_selected[16];
        for (int& v : untouched_selected) v = 77;
        p = df2_piped_group_rows(0, 1, untouched_active, nullptr, 7, 0, 0, untouched_selected, rows4, ntok4);
        check(p.rows == 0 && untouched_selected[0] == 77,
              "piped row planner leaves outputs untouched when the cap is empty");
    }

    // ---- the window plan: how many slots run, and how deep each of them goes ----
    {   // breadth first (want_depth 0): every active slot runs, the rows left over split evenly
        using strata::core::df2_window_plan;
        const auto p4 = df2_window_plan(kCap, 4, 7, 0);
        check(p4.n_sel == 4 && p4.depth == 1, "8 rows, 4 slots, breadth first: 4 slots, 1 proposal each");
        const auto p2 = df2_window_plan(kCap, 2, 7, 0);
        check(p2.n_sel == 2 && p2.depth == 3, "8 rows, 2 slots, breadth first: 2 slots, 3 proposals each");
        const auto p1 = df2_window_plan(kCap, 1, 7, 0);
        check(p1.n_sel == 1 && p1.depth == 7, "8 rows, 1 slot, breadth first: the drafter's whole chain");
        const auto p8 = df2_window_plan(kCap, 8, 7, 0);
        check(p8.n_sel == 8 && p8.depth == 0, "8 rows, 8 slots: no row left for a proposal (plain --batch)");
        const auto p12 = df2_window_plan(kCap, 12, 7, 0);
        check(p12.n_sel == kCap && p12.depth == 0, "more slots than rows: the window runs kVerifyMaxT, anchor rows only");
    }
    {   // depth first (want_depth >= 1): the chains keep their depth and the slots that do not fit wait
        using strata::core::df2_window_plan;
        const auto p4 = df2_window_plan(kCap, 4, 7, 3);
        check(p4.n_sel == 2 && p4.depth == 3, "8 rows, 4 slots, depth 3 asked: 2 slots at 3 proposals each");
        const auto p6 = df2_window_plan(kCap, 6, 7, 3);
        check(p6.n_sel == 2 && p6.depth == 3, "8 rows, 6 slots, depth 3 asked: still 2 slots, the rest wait their turn");
        const auto p2 = df2_window_plan(kCap, 2, 7, 3);
        check(p2.n_sel == 2 && p2.depth == 3, "8 rows, 2 slots, depth 3 asked: both run, exactly the asked depth");
        const auto p1 = df2_window_plan(kCap, 1, 7, 3);
        check(p1.n_sel == 1 && p1.depth == 7, "one slot with the window to itself takes the full chain, not just 3");
        const auto p7 = df2_window_plan(kCap, 4, 7, 7);
        check(p7.n_sel == 1 && p7.depth == 7, "depth 7 asked: one slot holds the whole window");
        const auto cl = df2_window_plan(kCap, 4, 1, 5);
        check(cl.n_sel == 4 && cl.depth == 1, "asking deeper than the drafter goes clamps to the drafter's 1");
        const auto none = df2_window_plan(kCap, 4, 0, 3);
        check(none.n_sel == 0 && none.depth == 0, "a drafter that cannot draft has no plan");
        check(strata::core::df2_window_plan(0, 4, 7, 3).n_sel == 0, "no rows: no plan");
        check(strata::core::df2_window_plan(kCap, 0, 7, 3).n_sel == 0, "no active slots: no plan");
    }
    {   // every plan feeds the row splitter a layout the window's own law accepts, at the depth it promised
        for (int cap = 1; cap <= kCap; ++cap)
            for (int ns = 1; ns <= kCap; ++ns)
                for (int md = 0; md <= 7; ++md)
                    for (int want = 0; want <= 7; ++want) {
                        const auto p = strata::core::df2_window_plan(cap, ns, md, want);
                        if (p.n_sel <= 0) continue;
                        int rows[kCap], ntok[kCap];
                        const int S = strata::core::batch_spec_slot_rows(cap, p.n_sel, nullptr, p.depth, rows, ntok);
                        bool fits = S <= cap && S == p.n_sel * (1 + p.depth);
                        for (int a = 0; a < p.n_sel && fits; ++a)
                            if (ntok[a] != 1 + p.depth) fits = false;
                        check(fits, "every plan: the splitter gives each slot exactly its planned depth, inside the rows");
                        int64_t pos[kCap];
                        positions(rows, S, pos);
                        check(strata::core::batch_rows_error(rows, S, pos, p.n_sel, cap, 0, false, kCap) == nullptr,
                              "every plan: the window's law accepts the layout the plan produces");
                        check(p.depth <= md || md == 0, "the plan never promises deeper than the drafter drafts");
                    }
    }
    // ---- what a run watched ----
    {
        strata::core::Df2DraftStats st;
        check(st.rounds == 0 && st.tokens_per_round() == 0.0 && st.deepest_accepted() == 0,
              "an empty run: no rounds, no tokens per round");
        st.note_round(1, 1, 1);   // the 4-slot window today: one stored, one verified, confirmed
        st.note_round(1, 1, 0);   // ...and rejected
        st.note_round(0, 0, 0);   // a cold round: nothing drafted, nothing offered
        st.note_round(7, 3, 2);   // a deeper window: 7 stored, 3 rows verified, 2 confirmed
        check(st.rounds == 4, "four rounds");
        check(st.drafted == 3, "the cold round is not a drafted round");
        check(st.stored == 9 && st.offered == 5 && st.accepted == 3, "the counters add the rows and the confirmations");
        check(st.unverified() == 4, "the proposals the row budget never got a row for");
        check(st.deepest_accepted() == 2, "the deepest accepted chain in the run is 2");
        check(st.depth[0] == 2 && st.depth[1] == 1 && st.depth[2] == 1 && st.depth[3] == 0,
              "two rounds took no proposal (cold, and one rejected), one accepted 1 and one accepted 2");
        check(st.tokens_per_round() == (4.0 + 3.0) / 4.0, "tokens per round counts the anchor row plus the confirmations");
        strata::core::Df2DraftStats cl;
        cl.note_round(2, 1, 5);   // nonsense input: more confirmed than rows
        cl.note_round(-3, -1, -1);
        check(cl.rounds == 2 && cl.offered == 1 && cl.accepted == 1 && cl.stored == 2,
              "a round clamps to what it can mean: never past the rows, never negative");
        strata::core::Df2DraftStats wide;
        for (int d = 0; d <= strata::kernels::kVerifyMaxT + 2; ++d)
            wide.note_round(strata::kernels::kVerifyMaxT + 2, strata::kernels::kVerifyMaxT + 2, d);
        check(wide.deepest_accepted() == strata::kernels::kVerifyMaxT, "the histogram stops at the window's rows");
        strata::core::Df2DraftStats pl;
        check(pl.avg_slots() == 0.0 && pl.avg_depth() == 0.0, "no windows planned: no averages");
        pl.note_plan(4, 1);
        pl.note_plan(2, 3);
        pl.note_plan(-1, -2);
        check(pl.planned == 3 && pl.avg_slots() == 6.0 / 3.0 && pl.avg_depth() == 4.0 / 3.0,
              "the planned shape averages the windows, and a negative one counts as nothing");
    }

    // ---- per-slot max_drafts (STRATA_BATCH_DF2_DEPTH) ----
    {
        int rows[kStorageCap] = {}, ntok[kStorageCap] = {};
        int64_t pos[kStorageCap];
        if (kWideWindow) {
            // Under cap in a 16-row window: each slot gets its own adaptive depth (e.g. 1, 3, 0, 2 -> 4 + 6 = 10 rows).
            const int max_d[4] = {1, 3, 0, 2};
            const float ema[4] = {0.5f, 0.85f, 0.5f, 0.75f};
            const int S = strata::core::batch_spec_slot_rows(kStorageCap, 4, max_d, rows, ntok, ema);
            check(S == 10 && ntok[0] == 2 && ntok[1] == 4 && ntok[2] == 1 && ntok[3] == 3,
                  "per-slot max_drafts under cap: each slot keeps its exact requested depth (10 rows in a 16-row window)");
            positions(rows, S, pos);
            check(strata::core::batch_rows_error(rows, S, pos, 4, kStorageCap, 0, false, kStorageCap) == nullptr,
                  "per-slot max_drafts under cap: passes the layout law");

            // Over cap with EMA: 4 slots requesting {3, 3, 3, 3} in a 12-row window -> lowest ema^k dropped first.
            const int max_d12[4] = {3, 3, 3, 3};
            const float ema12[4] = {0.9f, 0.6f, 0.8f, 0.7f};
            const int S12 = strata::core::batch_spec_slot_rows(12, 4, max_d12, rows, ntok, ema12);
            check(S12 == 12 && ntok[0] == 4 && ntok[1] == 2 && ntok[2] == 4 && ntok[3] == 2,
                  "per-slot max_drafts over cap with EMA: best slots keep depth 3, lower EMA slots trim to 1");
            positions(rows, S12, pos);
            check(strata::core::batch_rows_error(rows, S12, pos, 4, kStorageCap, 0, false, 12) == nullptr,
                  "per-slot max_drafts over cap with EMA: passes the layout law");
        } else {
            std::fputs("skip: the 10-row and 12-row per-slot max_drafts checks need a wide build (kVerifyMaxT == 16)\n", stderr);
        }

        // Under/over cap in an 8-row window (runs in both narrow and wide builds).
        const int max_d8[4] = {1, 2, 0, 2};
        const float ema8[4] = {0.6f, 0.9f, 0.5f, 0.7f};
        const int S8 = strata::core::batch_spec_slot_rows(kCap, 4, max_d8, rows, ntok, ema8);
        check(S8 == 8 && ntok[0] == 2 && ntok[1] == 3 && ntok[2] == 1 && ntok[3] == 2,
              "per-slot max_drafts in 8-row window with EMA: highest-EMA slot keeps depth 2, slot 2 stays anchor-only");
        positions(rows, S8, pos);
        check(strata::core::batch_rows_error(rows, S8, pos, 4, kCap, 0, false, kCap) == nullptr,
              "per-slot max_drafts in 8-row window with EMA: passes the layout law");

        // Over cap without EMA: round-robin from the left up to each slot's cap.
        const int max_d_rr[3] = {1, 4, 4};
        const int Srr = strata::core::batch_spec_slot_rows(kCap, 3, max_d_rr, rows, ntok, nullptr);
        check(Srr == 8 && ntok[0] == 2 && ntok[1] == 3 && ntok[2] == 3,
              "per-slot max_drafts without EMA: slot 0 stops at its cap 1, slots 1 and 2 share the remaining rows");
    }

    if (g_fail != 0) {
        std::fprintf(stderr, "batch_spec_slots_test: %d of %ld checks FAILED\n", g_fail, g_checks);
        return EXIT_FAILURE;
    }
    std::printf("batch_spec_slots_test: ok (%ld checks)\n", g_checks);
    return EXIT_SUCCESS;
}
