// src/prefill/vram_plan.cpp - #796 part C: the ADVISORY startup VRAM plan and the shared chunk/loan predictor.
//
// `plan_lend_chunks` mirrors generate.cpp's `plan_lend` (the same 0.1.39 list, the same #583 bisection and ring
// what-ifs, the same fixed-chunk halving) so the advisory can say what the runtime's loan scan will pick; the
// runtime's scan stays the only decision maker, and a drift between the two shows up as the caller's mismatch
// warning.  `advise_startup_vram` prices the prompt path with v0.1.40's exact owned pricing BEFORE the cache is
// committed and warns about shortfalls - it never refuses and never sets an allocation.

#include "strata/prefill/vram_plan.hpp"

#include <algorithm>
#include <utility>

namespace strata::prefill {
namespace {

// 0.1.39's auto chunk list (also the size up to which a prompt keeps that ring under #583)
constexpr int64_t kAutoChunks[] = {32768, 16384, 8192, 6144, 4096, 3072, 2048, 1024, 512, 256};
constexpr int64_t kRingMin = 16;   // the second scan's floor, when no chunk affords a full ring

// the bytes for `chunk` with no ring counted: bytes_needed's ring term is exactly additive, so pricing at any
// budget and subtracting that budget's slots is budget-invariant
uint64_t no_ring_bytes(const CacheLendView& cache, const LendCosts& costs, int64_t chunk) {
    const int64_t blob = cache.max_blob > 0 ? cache.max_blob : 1;
    return costs.bytes_with_ring(chunk, 0) - (uint64_t) costs.ring_slots_under(chunk, 0) * (uint64_t) blob;
}

}  // namespace

// the largest chunk on the 256-token grid whose `ok` holds; `ok` is monotone in the chunk
int64_t biggest_lend_chunk(int64_t ceiling, const std::function<bool(int64_t)>& ok) {
    if (ceiling < 256) return 0;
    int64_t lo = 1, hi = ceiling / 256, best = 0;   // n = T / 256
    while (lo <= hi) {
        const int64_t mid = lo + (hi - lo) / 2;
        if (ok(mid * 256)) { best = mid * 256; lo = mid + 1; } else hi = mid - 1;
    }
    return best;
}

LendOutcome plan_lend_chunks(const CacheLendView& cache, const LendOpts& opts, const LendCosts& costs,
                             bool auto_chunk, int64_t& chunk) {
    auto fits = [&](int64_t k, bool pct) {
        return k + opts.min_keep_slots <= cache.slots && (!pct || k * 100 <= opts.lend_pct * cache.slots);
    };
    // 0.1.39's list, priced at the pinned-default ring (the runtime sets set_ring_budget(0, 0) first)
    auto old_rule = [&]() -> LendOutcome {
        for (const int64_t c : kAutoChunks) {
            // above 8192: only when asked for, and only when a prompt of the context can use it
            if (c > 8192 && (c > opts.prefill_auto_max || c > opts.max_context)) continue;
            const int64_t k = cache.slots_for_bytes(costs.bytes_with_ring(c, -1));
            if (fits(k, true)) return {c, k, 0, 0};
        }
        return {0, 0, 0, 0};
    };
    if (auto_chunk && !opts.ring_bytes) return old_rule();   // STRATA_RING_BYTES=0: 0.1.39's list
    if (auto_chunk) {   // the default since 0.1.39b (#583): the ring and the chunk are one byte budget
        const LendOutcome small = old_rule();
        const int64_t ring_max = costs.ring_cap_for(small.chunk);
        const int64_t budget_slots = std::min(cache.slots - opts.min_keep_slots, opts.lend_pct * cache.slots / 100);
        const uint64_t avail = cache.tail_bytes(budget_slots);
        auto room_of = [&](int64_t t) -> int64_t {
            const uint64_t nonring = no_ring_bytes(cache, costs, t);
            const int64_t blob = cache.max_blob > 0 ? cache.max_blob : 1;
            return std::min((int64_t) ((avail - std::min(avail, nonring)) / (uint64_t) blob), ring_max);
        };
        // `room_of` is capped at ring_max, so "the ring is full" is exactly room == ring_max, and both that test
        // and the ones below it only get harder as t grows - the bisection stays valid
        auto scan = [&](int64_t floor) -> int64_t {
            return biggest_lend_chunk(opts.auto_ceiling, [&](int64_t t) -> bool {
                const int64_t room = room_of(t);
                if (room < floor) return false;
                const int64_t k = cache.slots_for_bytes(costs.bytes_with_ring(t, std::max<int64_t>(room, kRingMin)));
                return fits(k, true);
            });
        };
        int64_t c = scan(ring_max);
        if (c == 0 && ring_max < costs.ring_default_slots()) c = scan(kRingMin);
        if (small.chunk >= c) {   // the scan bought nothing: 0.1.39's choice
            if (small.chunk > 0)
                return {small.chunk, cache.slots_for_bytes(costs.bytes_with_ring(small.chunk, -1)), 0, 0};
            return {0, 0, 0, 0};
        }
        return {c, cache.slots_for_bytes(costs.bytes_with_ring(c, room_of(c))), room_of(c), small.chunk};
    }
    for (int64_t c = chunk; c >= 256; c /= 2) {   // a fixed chunk halves until its loan fits
        const int64_t k = cache.slots_for_bytes(costs.bytes_with_ring(c, -1));
        if (fits(k, false)) { chunk = c; return {c, k, -1, 0}; }
    }
    chunk = 0;
    return {0, 0, -1, 0};
}

namespace {

/// the lend view over a predicted layout (offsets like ExpertCache::open_sized's), written into `offs`
CacheLendView planned_view(int64_t slots, const std::vector<int64_t>& sized, int64_t max_blob,
                           std::vector<uint64_t>& offs) {
    CacheLendView v;
    v.slots = slots;
    v.max_blob = max_blob;
    if (!sized.empty()) {
        offs.assign(sized.size() + 1, 0);
        for (size_t i = 0; i < sized.size(); ++i)
            offs[(size_t) i + 1] = offs[i] + (uint64_t) ((sized[i] + 255) / 256 * 256);
        v.bytes = (int64_t) offs.back();
        v.slot_offsets = offs.data();
    } else {
        v.bytes = slots * max_blob;
        v.slot_offsets = nullptr;
    }
    return v;
}

}  // namespace

uint64_t actual_cache_bytes(const VramAdvisory& p) {
    if (!p.sized_offsets.empty()) return p.sized_offsets.back();
    return (uint64_t) (p.expert_slots > 0 ? p.expert_slots : 0) * (uint64_t) (p.max_blob > 0 ? p.max_blob : 0);
}

uint64_t post_cache_required_bytes(const VramAdvisory& p) {
    return p.mandatory_bytes + (p.prefill_owned ? p.prefill_bytes : 0);
}

VramAdvisory advise_startup_vram(const AdvisoryInput& in) {
    VramAdvisory p;
    p.free_at_plan = in.free_bytes;
    p.user_reserve_bytes = in.user_reserve_bytes;
    p.mtp_bytes = in.mtp_bytes;
    p.runtime_reserve_bytes = in.runtime_reserve_bytes;
    p.mandatory_bytes = in.user_reserve_bytes + in.mtp_bytes + in.runtime_reserve_bytes;
    p.prefill_borrow = in.prefill_borrow;
    p.prefill_auto = in.prefill_auto;
    p.fixed_chunk_halves = in.fixed_chunk_halves;
    p.requested_prefill = in.prefill_chunk;
    p.max_blob = in.max_blob;

    const int64_t blob = in.max_blob > 0 ? in.max_blob : 1;
    const bool prefill_on = in.prefill_auto || in.prefill_chunk > 0;
    const int64_t keep = in.lend.min_keep_slots;

    // The prompt path, said before the cache is committed.  Owned buffers are priced EXACTLY (v0.1.40's
    // bytes_needed_owned plus the allocator margin the caller folds into bytes_owned); what the runtime's sizing
    // rule withholds for them is reported beside it, because the cache is sized from the rule, not the truth.
    const auto owned_exact = [&](int64_t c) -> uint64_t { return in.costs.bytes_owned(c); };
    const auto owned_reserved = [&](int64_t c) -> uint64_t { return in.costs.bytes_owned_reserved(c); };
    const auto warn_short = [&](uint64_t need, const char* what) {
        if (in.free_bytes >= need) return;
        p.short_by_bytes = (int64_t) (need - in.free_bytes);
        p.warnings.push_back(std::string("the advisory VRAM plan predicts this configuration is short by ") +
                             std::to_string((p.short_by_bytes + (1 << 20) - 1) >> 20) + " MiB (" + what + ")");
        if (in.kv_stream_possible)
            p.suggestions.push_back("stream more KV from RAM (--kv-resident, e.g. 32768)");
        p.suggestions.push_back("a smaller --max-context");
        p.suggestions.push_back("a smaller --kv");
        if (prefill_on && !p.prefill_owned) p.suggestions.push_back("a smaller --prefill");
        p.suggestions.push_back("a smaller --vram-reserve-mib");
    };

    // the elastic consumer: everything after the mandatory items (and, when the prompt path owns its buffers,
    // what the sizing rule withholds for them) is expert residency
    const auto cache_budget_of = [&]() -> uint64_t {
        const uint64_t reserved = p.prefill_owned && p.selected_prefill > 0 ? owned_reserved(p.selected_prefill) : 0;
        const uint64_t taken = p.mandatory_bytes + reserved;
        return in.free_bytes > taken ? in.free_bytes - taken : 0;
    };

    uint64_t cache_budget = 0;
    int64_t slots = 0;
    std::vector<int64_t> sized;

    // the predicted cache layout: at most `n` uniform slots under `cap` bytes; the native pack's per-pair sizes
    // (the profile's hottest pairs first, each slot one whole aligned blob - the layout open_sized() will get)
    // where they are wanted.  A PREDICTION of the runtime's walk, not a decision.
    auto build_layout = [&](int64_t n, uint64_t cap) {
        std::vector<int64_t> out;
        if (in.sized_slots_wanted && in.pair_slot_bytes != nullptr) {
            cap = std::min<uint64_t>(cap, (uint64_t) n * (uint64_t) blob);
            uint64_t used = 0;
            for (const int64_t b : *in.pair_slot_bytes) {
                const uint64_t rb = (uint64_t) ((b + 255) / 256 * 256);
                if (used + rb > cap) break;
                used += rb;
                out.push_back(b);
            }
        }
        return std::make_pair(out.empty() ? n : (int64_t) out.size(), std::move(out));
    };

    // The owned reservation exists ONLY where the runtime's sizing makes it: with borrowing possible the sizing
    // withholds nothing (it expects a loan), so a lend that fails lands its buffers ON TOP of the cache - the
    // advisory says so instead of pretending the reservation away.
    if (prefill_on && !in.prefill_auto && !in.prefill_borrow) {
        p.selected_prefill = in.prefill_chunk;
        p.prefill_owned = true;
        p.prefill_bytes = owned_exact(in.prefill_chunk);
        p.prefill_reserved_bytes = owned_reserved(in.prefill_chunk);
        if (p.prefill_reserved_bytes < p.prefill_bytes)
            p.notes.push_back("the prompt path's own buffers for a " + std::to_string(in.prefill_chunk) +
                              "-token chunk price at " + std::to_string(p.prefill_bytes >> 20) +
                              " MiB exact, where the sizing rule withholds " +
                              std::to_string(p.prefill_reserved_bytes >> 20) + " MiB");
    }
    cache_budget = cache_budget_of();

    // the #496 adaptation the runtime is expected to make (it makes it itself; predicted here so the table is
    // the truth): a default reserve that leaves less than the minimum working cache shrinks to what leaves
    // exactly that, down to small_reserve_mib.  A reserve given on the command line is kept, and so is an
    // explicit cache (the operator's decision).
    if (!in.explicit_cache && !in.reserve_given &&
        in.user_reserve_bytes > (uint64_t) in.small_reserve_mib << 20) {
        slots = (int64_t) (cache_budget / (uint64_t) blob);
        if (in.profile_pairs >= 0) slots = std::min<int64_t>(slots, in.profile_pairs);
        if (slots < in.min_loan_slots) {
            const uint64_t fixed = p.mandatory_bytes - in.user_reserve_bytes;
            const int64_t fit_mib =
                ((int64_t) (in.free_bytes - fixed) - in.min_loan_slots * blob) / (1 << 20) -
                (int64_t) (p.prefill_owned && p.selected_prefill > 0 ? owned_reserved(p.selected_prefill) >> 20 : 0);
            if (fit_mib >= in.small_reserve_mib) {
                const int64_t r = std::min<int64_t>(fit_mib, (int64_t) (in.user_reserve_bytes >> 20));
                p.reserve_adapted_from_mib = (int64_t) (in.user_reserve_bytes >> 20);
                p.notes.push_back("the runtime is expected to lower the " + std::to_string(p.reserve_adapted_from_mib) +
                                  " MiB reserve to " + std::to_string(r) + " MiB (a working cache needs " +
                                  std::to_string(in.min_loan_slots) + " slots)");
                const uint64_t reserve2 = (uint64_t) r << 20;
                p.user_reserve_bytes = reserve2;
                p.mandatory_bytes = fixed + reserve2;
                cache_budget = cache_budget_of();
            }
        }
    }

    if (in.explicit_cache) {
        slots = in.explicit_cache_slots;
        if (prefill_on && in.prefill_borrow) {
            // the lend decision over the cache the operator asked for (its layout, capped by the room past the
            // reserve - the walk the sizing makes for a native pack)
            std::vector<uint64_t> offs;
            const uint64_t room = in.free_bytes > in.user_reserve_bytes + in.runtime_reserve_bytes
                                      ? in.free_bytes - in.user_reserve_bytes - in.runtime_reserve_bytes : 0;
            auto [se, sizede] = build_layout(slots, room);
            slots = se;
            sized = std::move(sizede);
            const CacheLendView v = planned_view(slots, sized, blob, offs);
            uint64_t cache_bytes_local = (uint64_t) v.slots * (uint64_t) blob;
            if (!sized.empty()) {
                cache_bytes_local = 0;
                for (const int64_t s : sized) cache_bytes_local += (uint64_t) ((s + 255) / 256 * 256);
            }
            if (in.prefill_auto || in.fixed_chunk_halves) {
                int64_t chunk_in = in.prefill_chunk;
                p.lend = plan_lend_chunks(v, in.lend, in.costs, in.prefill_auto, chunk_in);
            } else {
                // generate semantics: the requested chunk is lent EXACTLY or not at all
                const int64_t k = v.slots_for_bytes(in.costs.bytes_with_ring(in.prefill_chunk, -1));
                if (k + keep <= v.slots) p.lend = {in.prefill_chunk, k, -1, 0};
            }
            if (p.lend.chunk > 0) {
                p.prefill_bytes = v.tail_bytes(p.lend.slots);
                p.lend_slots = p.lend.slots;
                p.selected_prefill = p.lend.chunk;
                if (in.prefill_auto)
                    p.notes.push_back("prefill auto: the loan scan is expected to pick " +
                                      std::to_string(p.lend.chunk) + " tokens (" + std::to_string(p.lend_slots) +
                                      " cache slots) from the --expert-cache " +
                                      std::to_string(in.explicit_cache_slots) + "-slot cache");
            } else {
                // the sizing withholds nothing when borrowing is possible: the owned buffers land beside the
                // cache the operator asked for - said now, not at init
                p.prefill_owned = true;
                p.selected_prefill = in.prefill_auto ? 1024 : in.prefill_chunk;
                p.prefill_bytes = owned_exact(p.selected_prefill);
                p.notes.push_back("the expert cache (" + std::to_string(v.slots) + " slots) cannot lend the prompt "
                                  "path's buffers; the prompt path is expected on buffers of its own (" +
                                  std::to_string(p.selected_prefill) + "-token chunk)");
                p.warnings.push_back("the advisory VRAM plan expects the prompt path on its own buffers: the "
                                     "--expert-cache " + std::to_string(in.explicit_cache_slots) + "-slot cache "
                                     "cannot lend " + std::string(in.prefill_auto ? "a chunk's loan" : "a " +
                                                                         std::to_string(in.prefill_chunk) +
                                                                         "-token chunk"));
                warn_short(p.mandatory_bytes + p.prefill_bytes + cache_bytes_local,
                           "the prompt path's own buffers beside the explicit cache");
            }
        }
        // too large beside the mandatory items and the owned buffers?
        const uint64_t reserved = p.prefill_owned && p.selected_prefill > 0 ? owned_reserved(p.selected_prefill) : 0;
        const uint64_t cache_bytes = (uint64_t) slots * (uint64_t) blob;   // the sizing rule's uniform price
        if ((uint64_t) slots > 0 && in.free_bytes < p.mandatory_bytes + reserved + cache_bytes) {
            const uint64_t room = in.free_bytes > p.mandatory_bytes + reserved ? in.free_bytes - p.mandatory_bytes - reserved : 0;
            const int64_t fit = (int64_t) (room / (uint64_t) blob);
            p.warnings.push_back("the advisory VRAM plan reads --expert-cache " + std::to_string(in.explicit_cache_slots) +
                                 " as too large: " + std::to_string(std::max<int64_t>(fit, 0)) +
                                 " slots fit beside the reserve" +
                                 std::string(p.prefill_owned ? " and the prompt path's own buffers" : ""));
        }
    } else if (prefill_on && in.prefill_borrow) {
        // the auto sizing's cache, predicted: everything the mandatory items leave, capped by the profile
        slots = (int64_t) (cache_budget / (uint64_t) blob);
        if (in.profile_pairs >= 0) slots = std::min<int64_t>(slots, in.profile_pairs);
        std::vector<uint64_t> offs;
        auto [s0, sized0] = build_layout(slots, cache_budget);
        slots = s0;
        sized = std::move(sized0);
        const CacheLendView v0 = planned_view(slots, sized, blob, offs);
        int64_t chunk_in = in.prefill_chunk;   // auto ignores it; a fixed chunk starts halving from it
        if (in.prefill_auto || in.fixed_chunk_halves) {
            p.lend = plan_lend_chunks(v0, in.lend, in.costs, in.prefill_auto, chunk_in);
        } else {
            // generate semantics: the requested chunk is lent EXACTLY or not at all - a halve is discarded and
            // the requested chunk runs on buffers of its own
            const int64_t k = v0.slots_for_bytes(in.costs.bytes_with_ring(in.prefill_chunk, -1));
            if (k + keep <= v0.slots) p.lend = {in.prefill_chunk, k, -1, 0};
        }
        if (p.lend.chunk > 0) {
            p.prefill_owned = false;
            p.lend_slots = p.lend.slots;
            p.prefill_bytes = v0.tail_bytes(p.lend.slots);
            p.selected_prefill = p.lend.chunk;
            if (in.prefill_auto)
                p.notes.push_back("prefill auto: the loan scan is expected to pick " + std::to_string(p.lend.chunk) +
                                  " tokens (" + std::to_string(p.lend_slots) + " cache slots)");
        } else {
            // no loan affords: the runtime's fallback - auto at its 1024 bound, a fixed chunk at its own size -
            // lands its buffers on top of a cache the sizing booked no reservation for (it expected a loan)
            p.prefill_owned = true;
            p.selected_prefill = in.prefill_auto ? 1024 : in.prefill_chunk;
            p.prefill_bytes = owned_exact(p.selected_prefill);
            p.notes.push_back(std::string("no cache can lend even a 256-token chunk's buffers; the prompt path is "
                                          "expected on its own buffers for a ") +
                              std::to_string(p.selected_prefill) + "-token chunk");
            p.warnings.push_back("the advisory VRAM plan expects the prompt path on its own buffers (no cache can "
                                 "lend a loan): " + std::to_string(p.prefill_bytes >> 20) + " MiB on top of a " +
                                 std::to_string(slots) + "-slot cache");
            warn_short(p.mandatory_bytes + p.prefill_bytes + (uint64_t) slots * (uint64_t) blob,
                       "the prompt path's own buffers on top of the cache the sizing booked no room for");
        }
    } else if (prefill_on) {
        // no borrowing (no profile, or --no-prefill-borrow): an owned chunk is a real allocation, and
        // --prefill auto without a profile stays the runtime's no-op (it needs the profile to lend from)
        if (in.prefill_auto)
            p.notes.push_back("prefill auto needs an expert profile to lend from (none is loaded); the token path "
                              "reads the prompt");
    }

    // the final predicted layout
    p.expert_budget_bytes = cache_budget;
    p.expert_slots = slots < 0 ? 0 : slots;
    if (!sized.empty()) {
        p.sized_slots = std::move(sized);
        p.sized_offsets.assign(p.sized_slots.size() + 1, 0);
        for (size_t i = 0; i < p.sized_slots.size(); ++i)
            p.sized_offsets[(size_t) i + 1] =
                p.sized_offsets[i] + (uint64_t) ((p.sized_slots[i] + 255) / 256 * 256);
    }

    // THE advisory check, in one place, over the predicted layout: the mandatory items, an owned prompt path at
    // its exact price, and the cache inside the VRAM the plan saw.  Over the line: a warning with the knobs.
    const uint64_t need_final = post_cache_required_bytes(p) + actual_cache_bytes(p);
    if (need_final > in.free_bytes && p.short_by_bytes == 0) {
        p.short_by_bytes = (int64_t) (need_final - in.free_bytes);
        p.warnings.push_back("the advisory VRAM plan predicts this configuration is short by " +
                             std::to_string((p.short_by_bytes + (1 << 20) - 1) >> 20) + " MiB (the " +
                             std::to_string(p.expert_slots) + "-slot expert cache" +
                             (p.prefill_owned ? ", the prompt path's own buffers" : "") + " and the reserve together)");
        if (in.kv_stream_possible)
            p.suggestions.push_back("stream more KV from RAM (--kv-resident, e.g. 32768)");
        p.suggestions.push_back("a smaller --max-context");
        p.suggestions.push_back("a smaller --kv");
        p.suggestions.push_back("a smaller --prefill");
        p.suggestions.push_back("a smaller --vram-reserve-mib");
    }

    // what the prediction must still answer for: a zero-slot cache with the prompt path on it, or with --spec
    if (p.expert_slots <= 0 && prefill_on && in.prefill_borrow && p.lend_slots <= 0 && !p.prefill_owned)
        p.warnings.push_back("the advisory VRAM plan predicts no expert-cache slot at all for a prompt path that "
                             "expects to borrow one");
    if (p.expert_slots <= 0 && in.spec_needs_cache)
        p.warnings.push_back("the advisory VRAM plan predicts no expert-cache slot, and --spec needs one (the "
                             "verify window cannot start)");

    return p;
}

EffectivePrefillPlan revalidate_prefill_after_cache(const VramAdvisory& startup, const CacheLendView& cache,
                                                    const LendOpts& opts, const LendCosts& costs) {
    EffectivePrefillPlan e;
    if (startup.prefill_owned) {   // an owned prediction stays owned; the cache cannot take its room back
        e.owned = true;
        e.chunk = startup.selected_prefill;
        e.owned_bytes = startup.selected_prefill > 0 ? costs.bytes_owned(startup.selected_prefill) : 0;
        return e;
    }
    if (startup.prefill_auto) {
        // the same policy the prediction used, against the cache that actually exists now
        int64_t ignored = 0;
        e.lend = plan_lend_chunks(cache, opts, costs, true, ignored);
        if (e.lend.chunk > 0) {
            e.borrowed = true;
            e.chunk = e.lend.chunk;
            e.lend_slots = e.lend.slots;
            e.lend_bytes = cache.tail_bytes(e.lend.slots);
        } else {
            e.owned = true;          // the runtime's fallback: an owned path at request_chunk's 1024 bound
            e.chunk = 1024;
            e.owned_bytes = costs.bytes_owned(1024);
        }
        return e;
    }
    if (startup.selected_prefill <= 0) return e;   // the token path: nothing to re-derive
    if (startup.fixed_chunk_halves) {
        // serve semantics: the loan scan may run a halved fixed chunk
        int64_t chunk = startup.selected_prefill;
        e.lend = plan_lend_chunks(cache, opts, costs, false, chunk);
        if (e.lend.chunk > 0) {
            e.borrowed = true;
            e.chunk = e.lend.chunk;
            e.lend_slots = e.lend.slots;
            e.lend_bytes = cache.tail_bytes(e.lend.slots);
        } else {
            e.owned = true;
            e.chunk = startup.selected_prefill;
            e.owned_bytes = costs.bytes_owned(e.chunk);
        }
        return e;
    }
    // generate semantics: a fixed chunk must lend EXACTLY - the runtime rejects a halved chunk (k = 0) and
    // allocates its own buffers at the original size, so a smaller chunk the halving would find is not a loan here
    const int64_t k = cache.slots_for_bytes(costs.bytes_with_ring(startup.selected_prefill, -1));
    if (k + opts.min_keep_slots <= cache.slots) {
        e.borrowed = true;
        e.chunk = startup.selected_prefill;
        e.lend_slots = k;
        e.lend_bytes = cache.tail_bytes(k);
    } else {
        e.owned = true;
        e.chunk = startup.selected_prefill;
        e.owned_bytes = costs.bytes_owned(startup.selected_prefill);
    }
    return e;
}

uint64_t effective_post_cache_required(const VramAdvisory& startup, const EffectivePrefillPlan& effective) {
    return startup.mandatory_bytes + (effective.owned ? effective.owned_bytes : 0);
}

RuntimePlanCheck check_runtime_prefill_use(const EffectivePrefillPlan& predicted, const RuntimePrefillUse& actual) {    RuntimePlanCheck c;
    if (predicted.borrowed) {
        if (!actual.borrowed)
            c.why = "the advisory VRAM plan expected the prompt path to borrow from the expert cache, but the "
                    "runtime is on buffers of its own";
        else if (actual.chunk > predicted.chunk)
            c.why = "the runtime's loan is larger than the predicted chunk";
        else if (predicted.lend_bytes > 0 && actual.borrow_bytes > predicted.lend_bytes)
            c.why = "the runtime's loan is larger than the predicted loan bytes";
    } else if (predicted.owned) {
        if (actual.borrowed)
            c.why = "the advisory VRAM plan expected the prompt path on its own buffers, but the runtime set up "
                    "a loan";
        else if (actual.chunk > predicted.chunk)
            c.why = "the runtime's owned prompt buffers are larger than the predicted chunk";
    } else if (actual.chunk > 0) {
        c.why = "the advisory VRAM plan expected no prompt path; the runtime is running one";
    }
    c.mismatch = c.why != nullptr;
    return c;
}

}  // namespace strata::prefill
