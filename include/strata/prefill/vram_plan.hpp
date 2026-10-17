// include/strata/prefill/vram_plan.hpp - #796 part C: the ADVISORY VRAM planner.
//
// Startup used to size the expert cache from `cudaMemGetInfo` and a one-line prefill estimate, and only the first
// prompt told whether the chunk it had committed to could still be served.  Part A (0.1.40) priced the owned
// buffers exactly (`Prefill::bytes_needed_owned`) and put the streamed ring in one allocation; part C here says
// what that arithmetic predicts BEFORE the cache is committed - free VRAM, the reserve, the later allocations,
// the prompt path's cost, the loan, the cache budget - and warns when a configuration looks short or when the
// runtime later does something else than the prediction.
//
// THIS PLANNER IS ADVISORY ONLY.  It never refuses: a predicted shortfall is a warning with the knobs that make
// room, a mismatch between the prediction and the runtime is a warning, and the engine's own sizing, its loan
// scan and its existing error paths stay the only authorities.  `STRATA_ADVISORY_PLAN=0` silences it.  A helper
// like `check_runtime_prefill_use` returns a mismatch for DIAGNOSTICS; no caller may abort on it.
//
// Everything in this header is pure arithmetic over injected byte costs, so the policy is unit-testable without
// a GPU (tests/core/vram_plan_test.cpp); generate.cpp injects Prefill's real costs.
#pragma once

#include <cstdint>
#include <functional>
#include <string>
#include <vector>

namespace strata::prefill {

/// One environment-flag parser for the planner's knobs, with the usual on/off semantics: unset (or empty) means
/// `default_on`, a value starting with '0' is OFF, anything else is ON.  STRATA_ADVISORY_PLAN defaults on
/// (`=0` silences the planner), STRATA_POSTTOUCH defaults off (the deep touch only when asked for).
inline bool env_enabled(const char* value, bool default_on) {
    if (value == nullptr || value[0] == '\0') return default_on;
    return value[0] != '0';
}

/// Read-only lend surface of one expert cache - the real one (`ExpertCache`, whose slot offsets exist once open)
/// or a planned layout (offsets built from the profile's per-pair slot sizes).  One definition of "what the
/// prompt path may borrow" for both.
struct CacheLendView {
    int64_t slots = 0;
    int64_t bytes = 0;
    const uint64_t* slot_offsets = nullptr;   ///< open_sized's n+1 ascending slot starts; null = uniform slots
    int64_t max_blob = 0;                     ///< the uniform slot size (the pack's largest blob)

    /// The bytes the cache's last `k` slots hold - the region a prompt borrows (generate.cpp's bytes_from_slots).
    uint64_t tail_bytes(int64_t k) const {
        if (k <= 0 || slots <= 0) return 0;
        if (k > slots) k = slots;
        if (slot_offsets != nullptr) return (uint64_t) (bytes - (int64_t) slot_offsets[slots - k]);
        return (uint64_t) k * (uint64_t) max_blob;
    }
    /// The smallest slot count whose tail holds `need` (exact over sized offsets, ceil over the uniform blob;
    /// past the cache's own bytes the sized count stops at `slots`).
    int64_t slots_for_bytes(uint64_t need) const {
        if (need == 0) return 0;
        if (slot_offsets == nullptr) {
            const int64_t blob = max_blob > 0 ? max_blob : 1;
            return (int64_t) ((need + (uint64_t) blob - 1) / (uint64_t) blob);
        }
        int64_t k = 0;
        while (k < slots && tail_bytes(k) < need) ++k;
        return k;
    }
};

/// The chunk/lend policy knobs the advisory predictor describes - the same numbers the runtime's loan scan
/// obeys.  The predictor never sets them; it answers "what would a loan of this cache look like".
struct LendOpts {
    int64_t min_keep_slots = 128;   ///< slots a loan must leave outside itself (#496's non-lendable minimum)
    int64_t lend_pct = 90;          ///< an auto loan takes at most this share of the cache (STRATA_PREFILL_LEND_PCT)
    int64_t auto_ceiling = 8192;    ///< --prefill auto's largest chunk: max(8192, min(--prefill auto:N, context))
    int64_t prefill_auto_max = 8192;
    int64_t max_context = 0;
    bool ring_bytes = true;         ///< STRATA_RING_BYTES != 0 (0.1.39b's byte-budget ring and scan)
};

/// The byte costs the predictor prices its candidates with.  At runtime these wrap Prefill's statics (v0.1.40's
/// exact owned pricing included); the unit tests inject fakes.
/// `bytes_with_ring(chunk, ring_budget)`: `ring_budget` >= 0 prices a what-if ring of exactly that many slots
/// (the auto scan's rooms, clamped to at least 16 like the runtime's ring resolution); < 0 prices at the ring
/// the CURRENT globals resolve to.  `bytes_owned` is the exact owned-prefill price (`bytes_needed_owned` plus a
/// margin); `bytes_owned_reserved` is what the runtime's cache sizing withholds for owned buffers (the 0.1.39
/// rule, or the exact price under STRATA_OWNED_PRICE=exact) - the planner reports both.
struct LendCosts {
    std::function<uint64_t(int64_t chunk, int64_t ring_budget)> bytes_with_ring;
    std::function<int64_t(int64_t chunk, int64_t ring_budget)> ring_slots_under;
    std::function<int64_t(int64_t small_chunk)> ring_cap_for;
    std::function<int64_t()> ring_default_slots;
    std::function<uint64_t(int64_t chunk)> bytes_owned;
    std::function<uint64_t(int64_t chunk)> bytes_owned_reserved;
};

/// The chunk/loan prediction for one cache: `chunk` 0 = nothing lends.  `ring_budget` >= 0 is what the runtime
/// would set with `Prefill::set_ring_budget(ring_budget, ring_small_max)` to run the prediction; < 0 leaves the
/// globals.  Pure: no Prefill global is touched.
struct LendOutcome {
    int64_t chunk = 0;
    int64_t slots = 0;          ///< the loan's slot count
    int64_t ring_budget = -1;
    int64_t ring_small_max = 0;
};

/// The prompt path's chunk and loan against one cache - the policy behind generate.cpp's `plan_lend` (--prefill
/// auto's bisection over the 256-token grid, 0.1.39's list under STRATA_RING_BYTES=0, a fixed chunk's halving),
/// mirrored here so the advisory can PREDICT it.  Pure.  `chunk` in/out: auto ignores the incoming value, a
/// fixed chunk starts halving from it.  This is a DIAGNOSTIC mirror, not the policy: the runtime's scan stays
/// the only decision maker, and the mirror's job is precisely to be compared against it - a future drift
/// between the two is the caller's mismatch warning, never a decision of its own.
LendOutcome plan_lend_chunks(const CacheLendView& cache, const LendOpts& opts, const LendCosts& costs,
                             bool auto_chunk, int64_t& chunk);

/// The largest chunk on the 256-token grid whose `ok` holds (`ok` monotone in the chunk) - the auto scan's
/// bisection, shared with the runtime's scans.  log2(ceiling/256) probes at one cost evaluation each.
int64_t biggest_lend_chunk(int64_t ceiling, const std::function<bool(int64_t)>& ok);

/// The inputs the advisory prices.  `pair_slot_bytes` is the profile's ranked pairs' raw blob sizes (a native
/// pack's per-layer slots); null or empty keeps uniform `max_blob` slots.
struct AdvisoryInput {
    uint64_t free_bytes = 0;            ///< free VRAM at planning time (after weights, session, drafter, head)
    uint64_t user_reserve_bytes = 0;    ///< --vram-reserve-mib: left free for the driver/desktop, not a Strata budget
    bool reserve_given = false;         ///< on the command line (#496: the auto adaptation may not shrink it)
    int64_t small_reserve_mib = 300;    ///< #496's floor for the adapted reserve
    uint64_t mtp_bytes = 0;             ///< the draft head/logits bound after the cache (mtp.bind_bytes)
    uint64_t runtime_reserve_bytes = 0; ///< later allocations booked beside the reserve (--pipeline-windows)

    int64_t max_blob = 0;
    bool sized_slots_wanted = false;    ///< native pack + profile + not --expert-cache-per-layer
    const std::vector<int64_t>* pair_slot_bytes = nullptr;  ///< raw blob bytes per ranked pair
    int64_t profile_pairs = -1;         ///< cap on the auto slot count (-1: none)
    int64_t min_loan_slots = 1;         ///< the smallest working cache the runtime's sizing aims for
                                        ///  (a 256-token loan + the 128-slot floor when borrowing, else 1)

    bool prefill_borrow = false;        ///< borrowing is possible (an expert profile, not --no-prefill-borrow)
    bool prefill_auto = false;
    int64_t prefill_chunk = 0;          ///< the requested explicit chunk (0 = the token path)
    bool fixed_chunk_halves = false;    ///< serve semantics: the loan scan may run a HALVED fixed chunk (generate
                                        ///  rejects a halve and keeps the requested chunk on buffers of its own)
    bool spec_needs_cache = false;      ///< --spec T: the verify windows read the residency graph, so a
                                        ///  zero-slot cache cannot run the engine the user asked for
    bool kv_stream_possible = false;    ///< the model can stream KV from RAM (--kv-resident is a suggestion)

    bool explicit_cache = false;        ///< --expert-cache N: the plan describes it instead of sizing it
    int64_t explicit_cache_slots = -1;

    LendOpts lend;
    LendCosts costs;
};

/// One advisory startup picture: the mandatory items, the prompt path's predicted mode and cost, and the expert
/// residency that is left - what the engine is about to commit, said before it commits.  `warnings` non-empty
/// means the configuration looks dangerous; NOTHING here stops startup.
struct VramAdvisory {
    uint64_t free_at_plan = 0;
    uint64_t user_reserve_bytes = 0;
    uint64_t mtp_bytes = 0;
    uint64_t runtime_reserve_bytes = 0;
    uint64_t mandatory_bytes = 0;       ///< user reserve + mtp + runtime reserve

    bool prefill_borrow = false;
    bool prefill_auto = false;
    bool fixed_chunk_halves = false;    ///< the input's serve/generate semantics, carried for the re-derivation
    int64_t requested_prefill = 0;
    int64_t selected_prefill = 0;       ///< the chunk the plan expects to run (0 = the token path)
    uint64_t prefill_bytes = 0;         ///< owned: the exact allocation; borrowed: the loan's bytes
    uint64_t prefill_reserved_bytes = 0;///< owned: what the runtime's sizing rule withholds for it
    bool prefill_owned = false;         ///< the prompt path is expected on its own buffers
    int64_t lend_slots = 0;             ///< the predicted loan
    LendOutcome lend;                   ///< the predicted chunk/loan outcome (informational)

    int64_t max_blob = 0;               ///< the uniform slot size the plan was priced with

    uint64_t expert_budget_bytes = 0;   ///< what the cache may take
    int64_t expert_slots = 0;           ///< the predicted cache (the runtime sizes it on its own)
    std::vector<int64_t> sized_slots;   ///< the native pack's per-slot byte sizes, hottest pair first
    std::vector<uint64_t> sized_offsets;///< the predicted open_sized layout (the lend view over it)

    int reserve_adapted_from_mib = 0;   ///< the #496 adaptation the runtime is expected to make (predicted, not made)
    int64_t short_by_bytes = 0;         ///< > 0: how far the predicted configuration is over the free VRAM

    std::vector<std::string> notes;     ///< every clamp and fallback, printed verbatim by the caller
    std::vector<std::string> warnings;  ///< predicted shortfalls, too-large caches, expected owned fallbacks
    std::vector<std::string> suggestions;///< what to lower when a warning stands (smaller --max-context, ...)

    bool has_warning() const { return !warnings.empty(); }
};

/// The VRAM the predicted cache layout takes: the sized layout's exact bytes when one is predicted (native pack
/// + profile), the uniform slot count times the blob otherwise.
uint64_t actual_cache_bytes(const VramAdvisory& p);

/// What must still fit after the cache is committed: the mandatory items (the user reserve, the draft head, the
/// runtime reserve) plus an owned prompt path at its EXACT price, booked exactly once.  A borrowed prompt path
/// lives inside the cache and is not part of this.  The advisory's post-open check compares the free read with it.
uint64_t post_cache_required_bytes(const VramAdvisory& p);

/// The prompt path re-derived against a cache that actually exists (the opened one, after a WDDM shrink): what
/// the runtime is about to do, predicted from the same policy.  Pure; callers WARN on divergence, never enforce.
struct EffectivePrefillPlan {
    bool borrowed = false;
    bool owned = false;
    int64_t chunk = 0;          ///< the chunk the runtime is expected to run (an owned auto fallback: at most 1024)
    int64_t lend_slots = 0;     ///< borrowed: the loan against this cache
    uint64_t lend_bytes = 0;    ///< borrowed: the loan's bytes
    uint64_t owned_bytes = 0;   ///< owned: the buffers' exact price
    LendOutcome lend;           ///< borrowed auto: the outcome the runtime is expected to re-derive
};

EffectivePrefillPlan revalidate_prefill_after_cache(const VramAdvisory& startup, const CacheLendView& cache,
                                                    const LendOpts& opts, const LendCosts& costs);

/// The post-open requirement after the re-derivation: the mandatory items plus the re-derived owned buffers,
/// counted exactly once (a borrowed prompt path lives inside the cache and adds nothing here).
uint64_t effective_post_cache_required(const VramAdvisory& startup, const EffectivePrefillPlan& effective);

/// The prompt path as the runtime is about to run it, for the diagnostic comparison.
struct RuntimePrefillUse {
    bool borrowed = false;
    int64_t chunk = 0;
    uint64_t borrow_bytes = 0;  ///< borrowed: the loan's bytes (0 when the caller does not track them)
};

/// A mismatch between the advisory prediction and the runtime, for the caller's warning line.  `mismatch` is
/// information, never permission to stop: the runtime may run whatever the engine's own policy chose.
struct RuntimePlanCheck {
    bool mismatch = false;
    const char* why = nullptr;   ///< mismatch: what differs, for the warning
};
RuntimePlanCheck check_runtime_prefill_use(const EffectivePrefillPlan& predicted, const RuntimePrefillUse& actual);

/// The post-touch arithmetic for one opened cache: does the free read meet the requirement within the
/// tolerance, and - informational - how many bytes the cache would have to give up if it does not.  The
/// tolerance enters twice BY DESIGN (acceptance margin and shrink target, so a borderline cache is not
/// re-flagged on the next read).  The advisory never drives a shrink; the engine's open-retry loop stays.
struct PostTouchVerdict {
    bool accept = false;
    int64_t give_back_bytes = 0;    ///< !accept: the shortfall the next open would have to make room for
};
inline PostTouchVerdict post_touch_verdict(uint64_t free_after_touch, uint64_t required,
                                           int64_t tolerance_bytes, bool can_shrink_more) {
    PostTouchVerdict v;
    const uint64_t tolerance = (uint64_t) (tolerance_bytes > 0 ? tolerance_bytes : 0);
    v.accept = free_after_touch + tolerance >= required;
    if (!v.accept)
        v.give_back_bytes =
            can_shrink_more ? (int64_t) (required - free_after_touch + tolerance) : 0;
    return v;
}

/// The advisory startup plan: the prompt path's exact requirement said before the expert cache is committed,
/// borrowing never booked twice, and a predicted shortfall WARNED, never refused.
VramAdvisory advise_startup_vram(const AdvisoryInput& in);

}  // namespace strata::prefill
