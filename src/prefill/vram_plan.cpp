// src/prefill/vram_plan.cpp - #765: the startup VRAM plan and the shared chunk/loan policy.
//
// `plan_lend_chunks` is generate.cpp's `plan_lend` with the cache behind a CacheLendView and the byte costs
// injected: the same 0.1.39 list, the same #583 bisection and ring what-ifs, the same fixed-chunk halving - one
// source of truth for the startup plan and the runtime layout. `plan_startup_vram` prices the prompt path with the
// exact bytes BEFORE the expert cache is committed and treats expert residency as the elastic consumer.

#include "strata/prefill/vram_plan.hpp"

#include <algorithm>
#include <utility>

namespace strata::prefill {
namespace {

// 0.1.39's auto chunk list (also the size up to which a prompt keeps that ring under #583)
constexpr int64_t kAutoChunks[] = {32768, 16384, 8192, 6144, 4096, 3072, 2048, 1024, 512, 256};
constexpr int64_t kRingMin = 16;   // the second scan's floor, when no chunk affords a full ring

// the bytes for `chunk` with no ring counted: bytes_needed's ring term is exactly additive, so pricing at any
// budget and subtracting that budget's slots is budget-invariant (the runtime helper reads the leftover globals,
// which after 0.1.39's rule is the same (0, small_max=0) state this prices at)
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
            const int64_t k = cache.slots_for_bytes(costs.bytes_with_ring(c, 0));
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
                const int64_t k = cache.slots_for_bytes(costs.bytes_with_ring(t, room));
                return fits(k, true);
            });
        };
        int64_t c = scan(ring_max);
        if (c == 0 && ring_max < costs.ring_default_slots()) c = scan(kRingMin);
        if (small.chunk >= c) {   // the scan bought nothing: 0.1.39's choice
            if (small.chunk > 0)
                return {small.chunk, cache.slots_for_bytes(costs.bytes_with_ring(small.chunk, 0)), 0, 0};
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

/// the lend view over a planned layout (offsets like ExpertCache::open_sized's), written into `offs`
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

VramPlan plan_startup_vram(const StartupVramInput& in) {
    VramPlan p;
    p.free_at_plan = in.free_bytes;
    p.user_reserve_bytes = in.user_reserve_bytes;
    p.mtp_bytes = in.mtp_bytes;
    p.runtime_reserve_bytes = in.runtime_reserve_bytes;
    p.mandatory_bytes = in.user_reserve_bytes + in.mtp_bytes + in.runtime_reserve_bytes;
    p.prefill_borrow = in.prefill_borrow;
    p.prefill_auto = in.prefill_auto;
    p.requested_prefill = in.prefill_chunk;
    p.max_blob = in.max_blob;

    const int64_t blob = in.max_blob > 0 ? in.max_blob : 1;
    const bool prefill_on = in.prefill_auto || in.prefill_chunk > 0;
    const int64_t keep = in.lend.min_keep_slots;
    // the owned prompt path's price: `bytes_needed`'s sum plus the 2 MiB page rounding of its ~35 separate
    // cudaMallocs (the streamed ring, one allocation now, was the bulk of it) - the same figure the post-cache
    // revalidation prices the buffers with
    const auto owned_bytes = [&](int64_t c) -> uint64_t {
        return in.costs.bytes_with_ring(c, -1) + kOwnedPageMarginBytes;
    };
    const auto chunk_fits_alone = [&](int64_t c, uint64_t beside) {
        return in.free_bytes >= p.mandatory_bytes + owned_bytes(c) + beside;
    };

    // the elastic consumer: everything after the mandatory items is expert residency
    uint64_t cache_budget = in.free_bytes > p.mandatory_bytes ? in.free_bytes - p.mandatory_bytes : 0;
    const bool auto_cache = in.explicit_cache_slots < 0;
    int64_t slots = auto_cache ? 0 : in.explicit_cache_slots;
    std::vector<int64_t> sized;

    // the cache layout: at most `n` uniform slots under `cap` bytes; the native pack's per-pair sizes (the
    // profile's hottest pairs first, each slot one whole aligned blob - the layout open_sized() will get) where
    // they are wanted. One walk for the auto sizing and the explicit caches alike.
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
    // the planned cache under a byte budget: a uniform slot count, then the per-pair sizes inside it
    auto build_cache = [&](uint64_t budget) {
        int64_t n = (int64_t) (budget / (uint64_t) blob);
        if (in.profile_pairs >= 0) n = std::min<int64_t>(n, in.profile_pairs);
        return build_layout(n, budget);
    };

    // #496, at sizing time as before: a default reserve that leaves less than the minimum working cache - a
    // 256-token loan's buffers plus the non-lendable minimum when an explicit prefill is to be lent for, one
    // slot otherwise - shrinks to what leaves exactly that, down to kSmallReserveMib.  A reserve given on the
    // command line is kept, and so is an explicit cache (the operator's decision).
    if (auto_cache && !in.reserve_given &&
        in.user_reserve_bytes > (uint64_t) in.small_reserve_mib << 20) {
        auto [s0, sized0] = build_cache(cache_budget);
        slots = s0;
        sized = std::move(sized0);
        const int64_t min_slots = in.prefill_borrow && in.prefill_chunk > 0
            ? ((int64_t) in.costs.bytes_with_ring(256, -1) + blob - 1) / blob + keep : 1;
        if (slots < min_slots) {
            const int64_t fit_mib = ((int64_t) (in.free_bytes - in.mtp_bytes) - min_slots * blob) / (1 << 20);
            if (fit_mib >= in.small_reserve_mib) {
                const int64_t r = std::min<int64_t>(fit_mib, (int64_t) (in.user_reserve_bytes >> 20));
                p.reserve_adapted_from_mib = (int64_t) (in.user_reserve_bytes >> 20);
                p.notes.push_back("the " + std::to_string(p.reserve_adapted_from_mib) +
                                  " MiB reserve leaves too few slots on this card (a working cache needs " +
                                  std::to_string(min_slots) + "): a " + std::to_string(r) + " MiB reserve instead");
                const uint64_t reserve2 = (uint64_t) r << 20;
                p.user_reserve_bytes = reserve2;
                p.mandatory_bytes = reserve2 + in.mtp_bytes + in.runtime_reserve_bytes;
                cache_budget = in.free_bytes > p.mandatory_bytes ? in.free_bytes - p.mandatory_bytes : 0;
                auto [s2, sized2] = build_cache(cache_budget);
                slots = s2;
                sized = std::move(sized2);
            }
        }
    }

    // the owned prompt path is a real allocation; the borrowed one only needs a lendable tail (never both - the
    // double booking this replaces is what starved #760's card)
    auto book_owned = [&](int64_t c) {
        const uint64_t bytes = owned_bytes(c);
        p.prefill_bytes = bytes;
        p.prefill_owned = true;
        p.selected_prefill = c;
        if (cache_budget >= bytes) cache_budget -= bytes;
        return bytes;
    };
    auto plan_lends = [&](const CacheLendView& v, int64_t c) -> bool {
        const int64_t k = v.slots_for_bytes(in.costs.bytes_with_ring(c, -1));
        if (k + keep > v.slots) return false;
        p.lend_slots = k;
        p.prefill_bytes = v.tail_bytes(k);
        p.selected_prefill = c;
        p.lend.chunk = c;
        p.lend.slots = k;
        p.lend.ring_budget = -1;   // the runtime re-derives the ring from the real cache
        return true;
    };
    // the largest chunk on the 256-token grid, walking down from `from`, whose own buffers fit `beside` bytes
    // of cache (0: none) - an explicit chunk walks from its request, the auto fallback from its 1024 bound
    auto largest_owned_fit = [&](int64_t from, uint64_t beside) {
        for (int64_t c = from / 256 * 256; c >= 256; c -= 256)
            if (chunk_fits_alone(c, beside)) return c;
        return (int64_t) 0;
    };

    if (prefill_on && in.prefill_borrow && auto_cache) {
        auto [s0, sized0] = build_cache(cache_budget);
        const CacheLendView v0 = planned_view(s0, sized0, blob, p.sized_offsets);
        if (in.prefill_auto) {
            int64_t ignored = 0;
            p.lend = plan_lend_chunks(v0, in.lend, in.costs, true, ignored);
            if (p.lend.chunk > 0) {
                slots = s0;
                sized = std::move(sized0);
                p.lend_slots = p.lend.slots;
                p.prefill_bytes = v0.tail_bytes(p.lend.slots);
                p.selected_prefill = p.lend.chunk;
            } else {
                // no chunk affords a loan: the prompt path gets its own buffers, as the runtime's fallback does
                const uint64_t own = book_owned(1024);
                p.notes.push_back("prefill auto: no cache can lend even a 256-token chunk's buffers; the prompt "
                                  "path plans its own buffers for a 1024-token chunk");
                if (in.free_bytes < p.mandatory_bytes + own) {
                    // the fallback itself is reduced on the 256-token grid before the cache is committed:
                    // 1024 is only the initial candidate, and whatever survives becomes the runtime's ceiling
                    // (its owned fallback runs request_chunk(n, this chunk), never a fresh 1024)
                    const int64_t fit = largest_owned_fit(1024, 0);
                    if (fit == 0) {
                        p.ok = false;
                        p.short_by_bytes = (int64_t) (p.mandatory_bytes + own - in.free_bytes);
                        p.fail_why = "the prompt path's own buffers (1024-token chunk)";
                        return p;
                    }
                    p.prefill_bytes = owned_bytes(fit);
                    p.selected_prefill = fit;
                    cache_budget = in.free_bytes - p.mandatory_bytes - p.prefill_bytes;
                    p.notes.push_back("the 1024-token fallback does not fit either; " + std::to_string(fit) +
                                      " is the largest chunk whose own buffers do");
                }
                auto [s1, sized1] = build_cache(cache_budget);
                slots = s1;
                sized = std::move(sized1);
            }
        } else if (plan_lends(v0, in.prefill_chunk)) {
            slots = s0;
            sized = std::move(sized0);
        } else {
            // the explicit chunk cannot be lent even by the full cache: its own buffers, now priced exactly and
            // before the cache is committed (this is the late #760 failure, moved into the plan)
            const uint64_t own = book_owned(in.prefill_chunk);
            p.notes.push_back("prefill " + std::to_string(in.prefill_chunk) +
                              ": the planned cache cannot lend its buffers; the prompt path plans its own (the "
                              "cache gives up the same bytes)");
            if (in.free_bytes < p.mandatory_bytes + own) {
                // --spec reads the residency graph: the cache cannot go to zero, so the chunk yields a slot
                const int64_t fit =
                    largest_owned_fit(in.prefill_chunk, in.spec_needs_cache ? (uint64_t) blob : 0);
                if (fit == 0) {
                    p.ok = false;
                    p.prefill_owned = false;
                    p.selected_prefill = 0;
                    p.prefill_bytes = in.costs.bytes_with_ring(256, -1) +
                                      (in.spec_needs_cache ? (uint64_t) blob : 0);
                    p.short_by_bytes = (int64_t) (p.mandatory_bytes + p.prefill_bytes - in.free_bytes);
                    p.fail_why = "the prompt path's own buffers (" + std::to_string(in.prefill_chunk) +
                                 "-token chunk)" +
                                 (in.spec_needs_cache ? " and one expert-cache slot for --spec" : "");
                    return p;
                }
                p.prefill_bytes = owned_bytes(fit);
                p.selected_prefill = fit;
                cache_budget = in.free_bytes - p.mandatory_bytes - p.prefill_bytes;
                p.notes.push_back("prefill " + std::to_string(in.prefill_chunk) + " does not fit; " +
                                  std::to_string(fit) + " is the largest chunk whose own buffers do");
            }
            auto [s1, sized1] = build_cache(cache_budget);
            slots = s1;
            sized = std::move(sized1);
        }
    } else if (prefill_on && in.prefill_borrow) {
        // an explicitly sized cache: its EXACT layout - the per-pair slots open_sized() will get, under the cap
        // the old sizing used (the uniform count and the free VRAM past the reserve) - is built FIRST, and the
        // lend decision runs over that layout. The uniform view prices the tail at max_blob a slot, which
        // OVERSTATES what the real suffix holds and approves loans the opened cache cannot serve.
        {
            const uint64_t room =
                in.free_bytes > in.user_reserve_bytes ? in.free_bytes - in.user_reserve_bytes : 0;
            auto [se, sizede] = build_layout(slots, room);
            slots = se;
            sized = std::move(sizede);
        }
        std::vector<uint64_t> offs;
        const CacheLendView v = planned_view(slots, sized, blob, offs);
        const uint64_t cache_bytes = (uint64_t) (v.bytes > 0 ? v.bytes : 0);
        if (in.prefill_auto) {
            // --prefill auto resolves with the same policy as the auto cache's: the bisection over the
            // 256-token grid against this cache's real tail (one chunk-selection policy, startup and runtime)
            int64_t ignored = 0;
            p.lend = plan_lend_chunks(v, in.lend, in.costs, true, ignored);
            if (p.lend.chunk > 0) {
                p.lend_slots = p.lend.slots;
                p.prefill_bytes = v.tail_bytes(p.lend.slots);
                p.selected_prefill = p.lend.chunk;
            } else {
                // no chunk affords a loan from this cache: the prompt path gets its own buffers, as the auto
                // cache's fallback does; the cache stays as the operator asked and the final invariant judges
                const uint64_t own = book_owned(1024);
                p.notes.push_back("prefill auto: the explicit cache cannot lend a chunk's buffers; the prompt "
                                  "path plans its own for a 1024-token chunk (the cache stays as asked)");
                if (in.free_bytes < p.mandatory_bytes + own + cache_bytes) {
                    // the fallback is reduced like any owned chunk - 1024 was only the initial candidate
                    const int64_t fit = largest_owned_fit(1024, cache_bytes);
                    if (fit == 0) {
                        p.ok = false;
                        p.prefill_bytes = in.costs.bytes_with_ring(256, -1);
                        p.short_by_bytes = (int64_t) (p.mandatory_bytes + p.prefill_bytes + cache_bytes -
                                                      in.free_bytes);
                        p.fail_why = "the prompt path's own buffers beside the explicit " + std::to_string(slots) +
                                     "-slot cache";
                        return p;
                    }
                    p.prefill_bytes = owned_bytes(fit);
                    p.selected_prefill = fit;
                    p.notes.push_back("the 1024-token fallback does not fit beside the cache either; " +
                                      std::to_string(fit) + " is the largest chunk that does");
                }
            }
        } else if (plan_lends(v, in.prefill_chunk)) {
            // lent from the explicit cache's real tail
        } else {
            const uint64_t own = book_owned(in.prefill_chunk);
            p.notes.push_back("the expert cache (" + std::to_string(slots) +
                              " slots) cannot lend the prompt path's buffers; they are planned beside it");
            if (in.free_bytes < p.mandatory_bytes + own + cache_bytes) {
                const int64_t fit = largest_owned_fit(in.prefill_chunk, cache_bytes);
                if (fit == 0) {
                    p.ok = false;
                    p.prefill_owned = false;
                    p.selected_prefill = 0;
                    p.prefill_bytes = in.costs.bytes_with_ring(256, -1);
                    p.short_by_bytes =
                        (int64_t) (p.mandatory_bytes + p.prefill_bytes + cache_bytes - in.free_bytes);
                    p.fail_why = "the prompt path's own buffers beside the explicit " + std::to_string(slots) +
                                 "-slot cache";
                    return p;
                }
                p.prefill_bytes = owned_bytes(fit);
                p.selected_prefill = fit;
                p.notes.push_back("prefill " + std::to_string(in.prefill_chunk) + " does not fit beside the "
                                  "cache; " + std::to_string(fit) + " is the largest chunk that does");
            }
        }
    } else {
        // no borrowing (no profile, --no-prefill-borrow, or the prompt path off): an owned chunk is a real
        // allocation, priced exactly instead of 160 + chunk * 680 / 1024
        int64_t own_chunk = prefill_on && !in.prefill_auto ? in.prefill_chunk : 0;
        uint64_t own_bytes = own_chunk > 0 ? owned_bytes(own_chunk) : 0;
        // --prefill auto without borrowing keeps today's no-op: the runtime resolves auto only against a cache it
        // can lend from (it needs the profile), so the plan says so instead of inventing a chunk
        if (prefill_on && in.prefill_auto)
            p.notes.push_back("prefill auto needs an expert profile to lend from (none is loaded); the token path "
                              "reads the prompt");
        if (auto_cache) {
            if (own_chunk > 0 && cache_budget < own_bytes) {
                const int64_t fit = largest_owned_fit(in.prefill_chunk, blob);
                if (fit == 0) {
                    p.ok = false;
                    p.prefill_bytes = own_bytes;
                    p.short_by_bytes =
                        (int64_t) (p.mandatory_bytes + own_bytes + (uint64_t) blob - in.free_bytes);
                    p.fail_why = "the prompt path's own buffers (" + std::to_string(in.prefill_chunk) +
                                 "-token chunk, no borrowing) and one expert-cache slot";
                    return p;
                }
                own_chunk = fit;
                own_bytes = owned_bytes(fit);
                cache_budget = in.free_bytes - p.mandatory_bytes - own_bytes;
                p.notes.push_back("prefill " + std::to_string(in.prefill_chunk) + " does not fit without "
                                  "borrowing; " + std::to_string(fit) + " is the largest chunk that does");
            } else if (own_chunk > 0) {
                cache_budget -= own_bytes;
            }
            p.prefill_bytes = own_bytes;
            p.prefill_owned = own_chunk > 0;
            p.selected_prefill = own_chunk;
            auto [s0, sized0] = build_cache(cache_budget);
            slots = s0;
            sized = std::move(sized0);
        } else {
            p.prefill_bytes = own_bytes;
            p.prefill_owned = own_chunk > 0;
            p.selected_prefill = own_chunk;
            const uint64_t cache_bytes = (uint64_t) slots * (uint64_t) blob;
            if (in.free_bytes < p.mandatory_bytes + own_bytes + cache_bytes) {
                // the explicit cache is reduced before the fact (as the layer split's branch always did), never
                // discovered at the first prompt
                const uint64_t room = in.free_bytes > p.mandatory_bytes + own_bytes
                                          ? in.free_bytes - p.mandatory_bytes - own_bytes
                                          : 0;
                const int64_t fit = (int64_t) (room / (uint64_t) blob);
                if (fit >= 1) {
                    p.notes.push_back("--expert-cache " + std::to_string(slots) + " leaves no room for the prompt "
                                      "path's buffers and the reserve: " + std::to_string(fit) +
                                      " slots instead");
                    slots = fit;
                } else {
                    p.ok = false;
                    p.short_by_bytes =
                        (int64_t) (p.mandatory_bytes + own_bytes + cache_bytes - in.free_bytes);
                    p.fail_why = "the " + std::to_string(slots) + "-slot cache, the prompt path's buffers and the "
                                 "reserve together";
                    return p;
                }
            }
            // the explicit cache's sized layout, under the same cap the old sizing used (the uniform count and
            // the free VRAM past the reserve - the operator chose the slot count)
            if (slots > 0 && in.sized_slots_wanted) {
                const uint64_t room =
                    in.free_bytes > in.user_reserve_bytes ? in.free_bytes - in.user_reserve_bytes : 0;
                auto [se, sizede] = build_layout(slots, room);
                slots = se;
                sized = std::move(sizede);
            }
        }
    }

    // the planned layouts (the planning views above borrowed scratch vectors; rebuilt from the final list)
    p.expert_budget_bytes = cache_budget;
    p.expert_slots = slots < 0 ? 0 : slots;
    if (!sized.empty()) {
        p.sized_slots = std::move(sized);
        p.sized_offsets.assign(p.sized_slots.size() + 1, 0);
        for (size_t i = 0; i < p.sized_slots.size(); ++i)
            p.sized_offsets[(size_t) i + 1] =
                p.sized_offsets[i] + (uint64_t) ((p.sized_slots[i] + 255) / 256 * 256);
    }

    // THE invariant, in one place, over the layout that will actually be opened - whatever branch produced the
    // configuration: the mandatory Strata allocations, an owned prompt path booked exactly once (a borrowed one
    // lives inside the cache) and the final cache fit inside the VRAM the plan saw. A lend decision made against
    // a layout that then grew, or an explicit cache that never fit beside the mandatory items, is caught here
    // instead of at ExpertCache::open().
    const uint64_t cache_bytes_final = actual_cache_bytes(p);
    const uint64_t need_final = post_cache_required_bytes(p) + cache_bytes_final;
    if (need_final > in.free_bytes) {
        p.ok = false;
        p.short_by_bytes = (int64_t) (need_final - in.free_bytes);
        p.fail_why = "the " + std::to_string(p.expert_slots) + "-slot expert cache" +
                     (p.prefill_owned ? ", the prompt path's own buffers" : "") + " and the reserve together";
        return p;
    }

    // what the plan must still answer for: a prompt path with nowhere to go fails here instead of at the first
    // request; a zero-slot cache with the token path keeps today's behavior (startup continues with advice)
    if (p.expert_slots <= 0 && prefill_on && in.prefill_borrow && p.lend_slots <= 0 && !p.prefill_owned) {
        p.ok = false;
        p.short_by_bytes = (int64_t) (p.mandatory_bytes + (uint64_t) blob - in.free_bytes);
        p.fail_why = "one expert-cache slot";
        return p;
    }
    if (p.expert_slots <= 0 && in.spec_needs_cache) {
        // #174: the verify window cannot start without a cache - better said here, with the budget, than at the
        // first window after the prompt has been read
        p.ok = false;
        p.short_by_bytes =
            (int64_t) (p.mandatory_bytes + p.prefill_bytes + (uint64_t) blob - in.free_bytes);
        p.fail_why = "one expert-cache slot for --spec (the prompt path's own buffers take the rest)";
        return p;
    }
    p.ok = true;
    return p;
}

uint64_t actual_cache_bytes(const VramPlan& p) {
    if (!p.sized_offsets.empty()) return p.sized_offsets.back();
    return (uint64_t) (p.expert_slots > 0 ? p.expert_slots : 0) * (uint64_t) (p.max_blob > 0 ? p.max_blob : 0);
}

uint64_t post_cache_required_bytes(const VramPlan& p) {
    return p.mandatory_bytes + (p.prefill_owned ? p.prefill_bytes : 0);
}

EffectivePrefillPlan revalidate_prefill_after_cache(const VramPlan& startup, const CacheLendView& cache,
                                                    const LendOpts& opts, const LendCosts& costs) {
    EffectivePrefillPlan e;
    if (startup.prefill_owned) {   // an owned plan stays owned; the cache cannot take its room back
        e.owned = true;
        e.chunk = startup.selected_prefill;
        e.owned_bytes = startup.selected_prefill > 0
            ? costs.bytes_with_ring(startup.selected_prefill, -1) + kOwnedPageMarginBytes : 0;
        return e;
    }
    if (startup.prefill_auto) {
        // the same policy the startup plan used, against the cache that actually exists now
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
            e.owned_bytes = costs.bytes_with_ring(1024, -1) + kOwnedPageMarginBytes;
        }
        return e;
    }
    if (startup.selected_prefill <= 0) return e;   // the token path: nothing to revalidate
    // a fixed chunk must lend EXACTLY: the runtime rejects a halved chunk (k = 0) and allocates its own
    // buffers at the original size, so a smaller chunk the halving would find is not a loan here
    const int64_t k = cache.slots_for_bytes(costs.bytes_with_ring(startup.selected_prefill, -1));
    if (k + opts.min_keep_slots <= cache.slots) {
        e.borrowed = true;
        e.chunk = startup.selected_prefill;
        e.lend_slots = k;
        e.lend_bytes = cache.tail_bytes(k);
    } else {
        e.owned = true;
        e.chunk = startup.selected_prefill;
        e.owned_bytes = costs.bytes_with_ring(startup.selected_prefill, -1) + kOwnedPageMarginBytes;
    }
    return e;
}

uint64_t effective_post_cache_required(const VramPlan& startup, const EffectivePrefillPlan& effective) {
    return startup.mandatory_bytes + (effective.owned ? effective.owned_bytes : 0);
}

RuntimePlanCheck check_runtime_prefill_use(const EffectivePrefillPlan& accepted, const RuntimePrefillUse& actual) {
    RuntimePlanCheck c;
    if (accepted.borrowed) {
        if (!actual.borrowed)
            c.why = "the accepted VRAM plan lends the prompt path from the expert cache, but the runtime is "
                    "about to allocate its own buffers";
        else if (actual.chunk > accepted.chunk)
            c.why = "the runtime's loan is larger than the accepted plan's chunk";
        else if (accepted.lend_bytes > 0 && actual.borrow_bytes > accepted.lend_bytes)
            // the loan's BYTES are the ceiling: a shorter chunk under a different ring rule could price more
            c.why = "the runtime's loan is larger than the accepted plan's loan bytes";
    } else if (accepted.owned) {
        if (actual.borrowed)
            c.why = "the accepted VRAM plan runs the prompt path on its own buffers, but the runtime set up a "
                    "loan";
        else if (actual.chunk > accepted.chunk)
            c.why = "the runtime's owned prompt buffers are larger than the accepted plan's chunk";
    } else if (actual.chunk > 0) {
        c.why = "the accepted VRAM plan has no prompt path; the runtime is about to allocate one";
    }
    c.ok = c.why == nullptr;
    return c;
}

}  // namespace strata::prefill
