// include/strata/prefill/vram_plan.hpp - #765: the startup VRAM plan.
//
// Startup used to size the expert cache from `cudaMemGetInfo` and a one-line prefill estimate, and only found out at
// the first prompt whether the chunk it had committed to could still be served - the 524K failure of #760: an int8 KV
// of ~3.7 GiB, a cache that shrank around it, and a `--prefill 24576` whose ~2.4 GiB of buffers no longer fit
// anywhere, so the engine died between the model load and READY (or crawled under WDDM). The planner here prices the
// prompt path with the exact `Prefill::bytes_needed` BEFORE the cache is committed, treats expert residency as the
// elastic consumer, and fails before allocating when no combination fits.
//
// Everything in this header is pure arithmetic over injected byte costs, so the policy is unit-testable without a
// GPU (tests/core/vram_plan_test.cpp); generate.cpp injects Prefill's real costs.
#pragma once

#include <cstdint>
#include <functional>
#include <string>
#include <vector>

namespace strata::prefill {

/// Read-only lend surface of one expert cache - the real one (`ExpertCache`, whose slot offsets exist once open) or
/// the startup plan's planned layout (offsets built from the profile's per-pair slot sizes). One definition of "what
/// the prompt path may borrow" for both.
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
    /// The smallest slot count whose tail holds `need` (generate.cpp's slots_from_bytes: exact over the sized
    /// offsets, ceil over the uniform blob; past the cache's own bytes the sized count stops at `slots`).
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

/// The chunk/lend policy knobs the startup planner and the runtime `plan_lend` share.
struct LendOpts {
    int64_t min_keep_slots = 128;   ///< slots a loan must leave outside itself (#496's non-lendable minimum)
    int64_t lend_pct = 90;          ///< an auto loan takes at most this share of the cache (STRATA_PREFILL_LEND_PCT)
    int64_t auto_ceiling = 8192;    ///< --prefill auto's largest chunk: max(8192, min(--prefill auto:N, context))
    int64_t prefill_auto_max = 8192;
    int64_t max_context = 0;
    bool ring_bytes = true;         ///< STRATA_RING_BYTES != 0 (0.1.39b's byte-budget ring and scan)
};

/// The byte costs the policy prices its candidates with. At runtime these wrap Prefill's statics; the unit tests
/// inject fakes. `ring_budget` < 0 prices at the current globals (the fixed-chunk path's contract); a budget >= 0
/// prices as if `set_ring_budget(budget, 0)` had just run (the scan's what-if, without touching the globals).
struct LendCosts {
    std::function<uint64_t(int64_t chunk, int64_t ring_budget)> bytes_with_ring;
    std::function<int64_t(int64_t chunk, int64_t ring_budget)> ring_slots_under;
    std::function<int64_t(int64_t small_chunk)> ring_cap_for;
    std::function<int64_t()> ring_default_slots;
};

/// The chunk/loan decision for one cache: `chunk` 0 = nothing fits. `ring_budget` >= 0 is what the caller sets with
/// `Prefill::set_ring_budget(ring_budget, ring_small_max)` when it runs the plan for real; < 0 leaves the globals.
struct LendOutcome {
    int64_t chunk = 0;
    int64_t slots = 0;          ///< the loan's slot count
    int64_t ring_budget = -1;
    int64_t ring_small_max = 0;
};

/// The prompt path's chunk and loan against one cache - the one policy behind generate.cpp's `plan_lend` (--prefill
/// auto's bisection over the 256-token grid, 0.1.39's list under STRATA_RING_BYTES=0, a fixed chunk's halving).
/// Pure: no Prefill global is touched. `chunk` in/out: auto ignores the incoming value, a fixed chunk starts halving
/// from it.
LendOutcome plan_lend_chunks(const CacheLendView& cache, const LendOpts& opts, const LendCosts& costs,
                             bool auto_chunk, int64_t& chunk);

/// The largest chunk on the 256-token grid whose `ok` holds (`ok` monotone in the chunk) - the auto scan's
/// bisection, shared with the serve path's per-stage scan. log2(ceiling/256) probes at one cost evaluation each.
int64_t biggest_lend_chunk(int64_t ceiling, const std::function<bool(int64_t)>& ok);

/// The inputs the startup plan prices. `pair_slot_bytes` is the profile's ranked pairs' raw blob sizes (a native
/// pack's per-layer slots); null or empty keeps uniform `max_blob` slots.
struct StartupVramInput {
    uint64_t free_bytes = 0;            ///< cudaMemGetInfo at planning time (after weights, session, drafter, head)
    uint64_t user_reserve_bytes = 0;    ///< --vram-reserve-mib: left free for the driver/desktop, not a Strata budget
    bool reserve_given = false;         ///< on the command line (#496: the auto adaptation may not shrink it)
    int64_t small_reserve_mib = 300;    ///< #496's floor for the adapted reserve
    uint64_t mtp_bytes = 0;             ///< the draft head/logits bound after the cache (mtp.bind_bytes)
    /// Internal later allocations booked beside the reserve - a layer split's per-stage verify windows, today.
    /// On one GPU the hit-path scratch, the verify windows and the decode graphs are funded by the reserve BY
    /// DESIGN (that is --vram-reserve-mib's documented job, #199): the planner does not claim to price them
    /// individually, and the WDDM post-touch correction keeps an owned prefill from consuming their room.
    uint64_t runtime_reserve_bytes = 0;

    int64_t max_blob = 0;
    bool sized_slots_wanted = false;    ///< native pack + profile + not --expert-cache-per-layer
    const std::vector<int64_t>* pair_slot_bytes = nullptr;  ///< raw blob bytes per ranked pair
    int64_t profile_pairs = -1;         ///< cap on the auto slot count (-1: none)

    bool prefill_borrow = false;
    bool prefill_auto = false;
    int64_t prefill_chunk = 0;          ///< the requested explicit chunk (0 = the token path)
    bool spec_needs_cache = false;      ///< --spec T: the verify windows read the residency graph, so a
                                        ///  zero-slot cache cannot run the engine the user asked for

    LendOpts lend;
    LendCosts costs;

    /// an explicitly sized cache (--expert-cache N > 0): the plan validates it instead of sizing it
    int64_t explicit_cache_slots = -1;
};

/// One internally consistent startup VRAM plan (#765): the mandatory items, the prompt path's requirement, and the
/// expert residency that is left - decided BEFORE any elastic allocation is committed.
struct VramPlan {
    bool ok = false;
    int64_t short_by_bytes = 0;         ///< !ok: how far the minimum configuration is over the free VRAM
    std::string fail_why;               ///< !ok: the limiting component, for the budget table

    uint64_t free_at_plan = 0;
    uint64_t user_reserve_bytes = 0;
    uint64_t mtp_bytes = 0;
    uint64_t runtime_reserve_bytes = 0;
    uint64_t mandatory_bytes = 0;       ///< user reserve + mtp + runtime reserve

    bool prefill_borrow = false;
    bool prefill_auto = false;
    int64_t requested_prefill = 0;
    int64_t selected_prefill = 0;       ///< the chunk the plan will run (0 = the token path)
    uint64_t prefill_bytes = 0;         ///< owned: booked beside the cache; borrowed: the loan the cache must hold
    bool prefill_owned = false;         ///< the plan runs the prompt on its own buffers (borrowing not possible)
    int64_t lend_slots = 0;             ///< the planned loan
    LendOutcome lend;                   ///< the runtime ring budget it implies (informational at startup)

    int64_t max_blob = 0;               ///< the uniform slot size the plan was priced with

    uint64_t expert_budget_bytes = 0;   ///< what the cache may take
    int64_t expert_slots = 0;
    std::vector<int64_t> sized_slots;   ///< the native pack's per-slot byte sizes, hottest pair first
    std::vector<uint64_t> sized_offsets;///< the planned open_sized layout (the lend view over it)

    int reserve_adapted_from_mib = 0;   ///< #496: the auto reserve was lowered to this, from the default
    std::vector<std::string> notes;     ///< every clamp and fallback, printed verbatim by the caller
};

/// The VRAM the plan's final cache layout actually takes: the sized layout's exact bytes when one is planned
/// (native pack + profile), the uniform slot count times the blob otherwise. The final budget invariant and the
/// tests price the cache with this - never expert_slots * max_blob for a sized plan.
uint64_t actual_cache_bytes(const VramPlan& p);

/// What must still fit after the cache is committed: the mandatory items (the user reserve, the draft head, the
/// runtime reserve) plus an owned prompt path, booked exactly once. A borrowed prompt path lives inside the cache
/// and is not part of this. The WDDM post-touch correction holds the cache to this figure.
uint64_t post_cache_required_bytes(const VramPlan& p);

/// Decide the startup plan: the prompt path's exact requirement participates before the expert cache is committed,
/// borrowing is never booked twice, and an impossible configuration returns ok=false with the shortfall.
VramPlan plan_startup_vram(const StartupVramInput& in);

}  // namespace strata::prefill
