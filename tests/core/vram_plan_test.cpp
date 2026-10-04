// tests/core/vram_plan_test.cpp - #765: the startup VRAM plan and the shared chunk/loan policy.
//
// The planner is pure arithmetic over injected byte costs, so every budget case runs without a GPU: expert
// residency giving way to a requested prefill (A), --prefill auto's fallback onto the 256-token grid (B, with a
// non-round boundary), the deterministic no-fit failure (C), borrowed prefill never booked twice (D), sized
// (native-pack) slots priced over their exact offsets (E), and the reserve staying the user's knob (F).
#include "strata/prefill/vram_plan.hpp"

#include <cstdio>
#include <algorithm>
#include <string>
#include <vector>

namespace {
int fails = 0;
void check(bool ok, const char* what) {
    if (!ok) {
        std::fprintf(stderr, "FAIL: %s\n", what);
        ++fails;
    }
}

constexpr int64_t BLOB = 2 << 20;          // one uniform expert slot: 2 MiB
constexpr int64_t KIB = 1024, MIB = 1024 * 1024, GIB = 1024 * 1024 * 1024;

// the fake pack: a chunk costs 64 KiB a token plus its streamed ring at 2 MiB a slot, 199 slots full (the
// ring_cap every fake answers), 8 slots below stream_all_min - monotone in the chunk, like the real one
int64_t base_bytes(int64_t chunk) { return chunk * 64 * KIB; }
int64_t fake_ring(int64_t chunk, int64_t budget) {
    if (chunk < 1024) return 8;
    return budget > 0 ? std::min<int64_t>(budget, 199) : 199;
}
uint64_t fake_bytes(int64_t chunk, int64_t budget) {
    return (uint64_t) (base_bytes(chunk) + fake_ring(chunk, budget) * BLOB);
}

strata::prefill::LendCosts fake_costs() {
    strata::prefill::LendCosts c;
    c.bytes_with_ring = fake_bytes;
    c.ring_slots_under = fake_ring;
    c.ring_cap_for = [](int64_t) { return (int64_t) 199; };
    c.ring_default_slots = [] { return (int64_t) 199; };
    return c;
}

strata::prefill::LendOpts fake_opts(int64_t ceiling = 8192) {
    strata::prefill::LendOpts o;
    o.min_keep_slots = 128;
    o.lend_pct = 90;
    o.auto_ceiling = ceiling;
    o.prefill_auto_max = 32768;
    o.max_context = 131072;
    o.ring_bytes = true;
    return o;
}

strata::prefill::StartupVramInput plan_input(uint64_t free_bytes) {
    strata::prefill::StartupVramInput in;
    in.free_bytes = free_bytes;
    in.user_reserve_bytes = 700 * MIB;
    in.small_reserve_mib = 300;
    in.max_blob = BLOB;
    in.profile_pairs = -1;
    in.lend = fake_opts();
    in.costs = fake_costs();
    return in;
}

// the invariant every successful plan must keep - priced with the planner's own helpers, over the FINAL layout
// (a sized cache's exact bytes, never expert_slots * max_blob)
bool plan_fits(const strata::prefill::VramPlan& p, uint64_t free_bytes) {
    return strata::prefill::post_cache_required_bytes(p) + strata::prefill::actual_cache_bytes(p) <= free_bytes;
}
void check_ok(const strata::prefill::VramPlan& p, uint64_t free_bytes, const char* what) {
    check(p.ok, what);
    if (p.ok) check(plan_fits(p, free_bytes), (std::string(what) + ": the final invariant").c_str());
}
}  // namespace

int main() {
    using namespace strata::prefill;

    // ---- Case A: a requested prefill is retained and the expert residency shrinks instead
    {
        StartupVramInput in = plan_input(8 * GIB);
        in.prefill_chunk = 24576;              // own buffers: ~1.9 GiB of the fake pack's bytes
        const VramPlan p = plan_startup_vram(in);
        check_ok(p, in.free_bytes, "A: the plan is valid");
        check(p.prefill_owned && p.selected_prefill == 24576, "A: the requested chunk is kept, on its own buffers");
        check(p.expert_budget_bytes == (uint64_t) 8 * GIB - p.mandatory_bytes - p.prefill_bytes,
              "A: the cache gets exactly what the mandatory items and the owned prefill leave");
        check(plan_fits(p, in.free_bytes), "A: the plan fits inside the free VRAM");
        // a tighter card: the chunk gives way only as far as it must, on the 256-token grid
        StartupVramInput tight = plan_input(1200 * MIB);
        tight.prefill_chunk = 24576;
        const VramPlan t = plan_startup_vram(tight);
        check_ok(t, tight.free_bytes, "A: the tight plan is valid");
        check(t.selected_prefill > 0 && t.selected_prefill < 24576 && t.selected_prefill % 256 == 0,
              "A: the chunk is reduced on the 256-token grid, not dropped");
        int64_t expect = 0;   // the largest chunk whose own buffers (with the page-rounding margin) fit
        for (int64_t c = 24576 / 256 * 256; c >= 256; c -= 256)
            if ((uint64_t) 1200 * MIB >= t.mandatory_bytes + fake_bytes(c, -1) + ((uint64_t) 64 << 20)) {
                expect = c; break;
            }
        check(t.selected_prefill == expect, "A: the reduced chunk is the largest one whose own buffers fit");
    }

    // ---- Case B: --prefill auto falls back to the largest 256-grid chunk the loan affords
    {
        StartupVramInput in = plan_input(0);   // free_bytes set below; the budget line is what sizes the cache
        in.prefill_borrow = true;
        in.prefill_auto = true;
        in.lend.auto_ceiling = 24576;
        // 760 slots: 16384's loan does not fit, and neither does anything above it - the bisection lands on
        // 13824, a size no hardcoded list holds (its 631-slot loan leaves exactly the 128-slot minimum + 1)
        in.free_bytes = (uint64_t) 760 * BLOB + 700 * MIB;
        const VramPlan p = plan_startup_vram(in);
        check_ok(p, in.free_bytes, "B: the plan is valid");
        check(p.selected_prefill == 13824, "B: the largest fitting chunk on the grid is selected");
        check(!p.prefill_owned && p.lend_slots == 631, "B: the chunk is lent, 631 slots");
        check(p.expert_slots >= p.lend_slots + 128, "B: the loan leaves the non-lendable minimum");
        // a smaller budget whose true maximum is 8704 - also off every list - to pin the bisection, not a list
        StartupVramInput odd = plan_input((uint64_t) 600 * BLOB + 700 * MIB);
        odd.prefill_borrow = true;
        odd.prefill_auto = true;
        odd.lend.auto_ceiling = 24576;
        const VramPlan o = plan_startup_vram(odd);
        check_ok(o, odd.free_bytes, "B: the non-round boundary lands on 8704");
        check(o.selected_prefill == 8704, "B: 8704 selected");
    }

    // ---- Case C: no configuration fits - a deterministic failure with a positive shortfall
    {
        StartupVramInput in = plan_input(718 * MIB);   // the reserve alone eats nearly all of it
        in.prefill_chunk = 4096;
        const VramPlan p = plan_startup_vram(in);
        check(!p.ok, "C: the impossible plan is refused");
        check(p.short_by_bytes > 0, "C: the shortfall is positive");
        check(p.expert_slots >= 0, "C: no negative slot count");
        check(!plan_fits(p, in.free_bytes) || p.prefill_bytes > 0,
              "C: the reported minimum is genuinely over the free VRAM");
        // the borrowed arm fails its own way: nothing can be lent, nothing can be owned
        StartupVramInput b = plan_input(718 * MIB);
        b.prefill_borrow = true;
        b.prefill_auto = true;
        const VramPlan pb = plan_startup_vram(b);
        check(!pb.ok && pb.short_by_bytes > 0, "C: the borrowed arm fails deterministically too");
        // a chunk that fits only by taking the cache's last slot fails when --spec needs that slot
        StartupVramInput sp = plan_input((uint64_t) 700 * MIB + fake_bytes(4096, -1) + ((uint64_t) 64 << 20));
        sp.prefill_borrow = true;
        sp.prefill_chunk = 4096;
        sp.spec_needs_cache = true;
        const VramPlan ps = plan_startup_vram(sp);
        check(!ps.ok && ps.short_by_bytes >= 0, "C: --spec refuses the plan that leaves the cache zero slots");
        sp.spec_needs_cache = false;
        const VramPlan pn = plan_startup_vram(sp);
        check(pn.ok && pn.expert_slots == 0 && pn.prefill_owned,
              "C: without --spec the same budget plans the token-path cache of zero slots");
    }

    // ---- Case D: borrowing is never booked twice; owning is booked exactly once
    {
        StartupVramInput owned = plan_input(8 * GIB);
        owned.prefill_chunk = 4096;            // no profile: the prompt path owns its buffers
        const VramPlan po = plan_startup_vram(owned);
        StartupVramInput borrowed = plan_input(8 * GIB);
        borrowed.prefill_borrow = true;
        borrowed.prefill_chunk = 4096;         // the same chunk, lent from the cache instead
        const VramPlan pb = plan_startup_vram(borrowed);
        check_ok(po, owned.free_bytes, "D: the owned plan is valid");
        check_ok(pb, borrowed.free_bytes, "D: the borrowed plan is valid");
        check(po.prefill_owned && po.prefill_bytes == fake_bytes(4096, -1) + ((uint64_t) 64 << 20),
              "D: the owned bytes are booked once, with the page-rounding margin");
        check(!pb.prefill_owned && pb.expert_budget_bytes == (uint64_t) 8 * GIB - pb.mandatory_bytes,
              "D: the borrowed plan deducts no prefill bytes from the cache's budget");
        check(pb.expert_budget_bytes - po.expert_budget_bytes >= po.prefill_bytes,
              "D: lending keeps the cache exactly the owned prefill's bytes larger");
        check(strata::prefill::post_cache_required_bytes(pb) == pb.mandatory_bytes,
              "R5: a borrowed plan's post-cache need is the mandatory items alone - no prefill deduction");
        check(strata::prefill::post_cache_required_bytes(po) == po.mandatory_bytes + po.prefill_bytes,
              "R6: an owned plan's post-cache need carries the prefill exactly once");
    }

    // ---- Case E: variable-size slots price the lend over their exact offsets, not max_blob * slots
    {
        CacheLendView v;
        const std::vector<uint64_t> sizes = {3 * MIB, 1 * MIB, 2 * MIB};   // one blob each, unequal
        std::vector<uint64_t> offs(sizes.size() + 1, 0);
        for (size_t i = 0; i < sizes.size(); ++i) offs[(size_t) i + 1] = offs[i] + sizes[i];
        v.slots = (int64_t) sizes.size();
        v.bytes = (int64_t) offs.back();       // 6 MiB
        v.slot_offsets = offs.data();
        v.max_blob = 3 * MIB;
        check(v.tail_bytes(2) == 3 * MIB && v.tail_bytes(3) == 6 * MIB, "E: the tail is the suffix sum");
        check(v.slots_for_bytes((uint64_t) (3 * MIB + 512 * KIB)) == 3,
              "E: 3.5 MiB needs the three exact-offset slots (the last slot alone is 2 MiB, two are 3)");
        CacheLendView u;
        u.slots = 3;
        u.bytes = 6 * MIB;
        u.max_blob = 3 * MIB;
        check(u.slots_for_bytes((uint64_t) (3 * MIB + 512 * KIB)) == 2,
              "E: the uniform view's ceil(need / max_blob) answers 2 - the two views differ, as they must");
        // through the planner: the sized cache holds more than its uniform count would, and keeps the bigger chunk
        StartupVramInput in = plan_input((uint64_t) 610 * BLOB + 700 * MIB);
        in.prefill_borrow = true;
        in.prefill_auto = true;
        in.sized_slots_wanted = true;
        in.profile_pairs = 3;
        const std::vector<int64_t> pairs = {1 * MIB, 1 * MIB, 1 * MIB};   // each aligned slot: 1 MiB, not 2
        in.pair_slot_bytes = &pairs;
        const VramPlan p = plan_startup_vram(in);
        check_ok(p, in.free_bytes, "E: the sized layout is planned");
        uint64_t sized_bytes = 0;
        for (const int64_t s : p.sized_slots) sized_bytes += (uint64_t) ((s + 255) / 256 * 256);
        check(!p.sized_slots.empty() && actual_cache_bytes(p) == sized_bytes,
              "E: actual_cache_bytes is the sized layout's exact total");
        check(sized_bytes < (uint64_t) p.expert_slots * BLOB,
              "E: the sized cache holds fewer bytes than its uniform count prices");
        check(p.selected_prefill > 0, "E: the sized plan still runs the prompt path");
    }

    // ---- Case F: the reserve stays the user's knob - bigger reserve, smaller cache, nothing else moves
    {
        StartupVramInput in = plan_input(8 * GIB);
        in.prefill_chunk = 4096;
        const VramPlan a = plan_startup_vram(in);
        in.user_reserve_bytes = 1400 * MIB;
        in.reserve_given = true;
        const VramPlan b = plan_startup_vram(in);
        check_ok(a, in.free_bytes, "F: the smaller reserve plans fine");
        check_ok(b, in.free_bytes, "F: the bigger reserve plans fine");
        check(b.expert_budget_bytes == a.expert_budget_bytes - 700 * MIB,
              "F: the extra 700 MiB of reserve comes out of the cache alone");
        check(b.mandatory_bytes == a.mandatory_bytes + 700 * MIB, "F: the mandatory side carries the reserve");
        check(plan_fits(a, in.free_bytes) && plan_fits(b, in.free_bytes),
              "F: used_by_strata + the reserve stays inside the free VRAM");
        // a given reserve is never adapted; the default one is, before the prompt path is planned (#496's order)
        StartupVramInput small_card = plan_input(700 * MIB);   // the reserve alone would eat all of it
        small_card.prefill_borrow = true;
        small_card.prefill_chunk = 4096;
        small_card.reserve_given = true;
        const VramPlan given = plan_startup_vram(small_card);
        check(!given.ok && given.short_by_bytes > 0 && given.reserve_adapted_from_mib == 0 &&
                  given.user_reserve_bytes == (uint64_t) 700 * MIB,
              "F: a reserve given on the command line is kept - and the impossible plan fails here, not at the "
              "first prompt");
        small_card.reserve_given = false;
        const VramPlan adapted = plan_startup_vram(small_card);
        check_ok(adapted, small_card.free_bytes, "F: the default reserve is adapted");
        check(adapted.reserve_adapted_from_mib == 700, "F: adapted from 700");
        check(adapted.user_reserve_bytes == (uint64_t) 412 * MIB, "F: the adapted reserve leaves 144 slots");
        check(adapted.expert_slots >= 1 && adapted.user_reserve_bytes >= (uint64_t) 300 * MIB,
              "F: the adaptation leaves a working cache and stays above its floor");
        check(plan_fits(adapted, small_card.free_bytes), "F: the adapted plan fits");
    }

    // ---- the shared policy: 0.1.39's list, the fixed chunk's halving, the bisection
    {
        CacheLendView big;
        big.slots = 4000;
        big.bytes = 4000 * BLOB;
        big.max_blob = BLOB;
        // STRATA_RING_BYTES=0: the old list, ceiling and opt-in respected
        int64_t chunk = 0;
        LendOpts old = fake_opts();
        old.ring_bytes = false;
        LendOutcome out = plan_lend_chunks(big, old, fake_costs(), true, chunk);
        check(out.chunk == 32768 && out.ring_budget == 0, "policy: the old list takes the opt-in ceiling");
        old.prefill_auto_max = 8192;
        out = plan_lend_chunks(big, old, fake_costs(), true, chunk);
        check(out.chunk == 8192, "policy: without the opt-in the old list stops at 8192");
        // a fixed chunk halves until the loan fits (the small chunks' STAGE ring included)
        CacheLendView mid;
        mid.slots = 300;
        mid.bytes = 300 * BLOB;
        mid.max_blob = BLOB;
        chunk = 8192;
        out = plan_lend_chunks(mid, fake_opts(), fake_costs(), false, chunk);
        check(out.chunk == 512 && chunk == 512 && out.ring_budget == -1,
              "policy: the fixed chunk halves to 512 (the STAGE ring) and leaves the ring globals alone");
        chunk = 256;
        out = plan_lend_chunks(CacheLendView{.slots = 8, .bytes = 8 * BLOB, .max_blob = BLOB}, fake_opts(),
                               fake_costs(), false, chunk);
        check(out.chunk == 0 && chunk == 0, "policy: nothing lends from a tiny cache");
        // the bisection lands on the largest grid size under a monotone cap
        check(biggest_lend_chunk(8704, [](int64_t t) { return t <= 8704; }) == 8704,
              "policy: the bisection holds the non-round boundary");
        check(biggest_lend_chunk(8192, [](int64_t t) { return t < 512; }) == 256,
              "policy: the bisection walks the 256-token grid");
    }

    // ---- review 1: an explicit cache that does not fit beside the mandatory items is refused before
    // ExpertCache::open() - even though its prefill would lend fine
    {
        StartupVramInput in = plan_input(1600 * MIB);
        in.prefill_borrow = true;
        in.prefill_chunk = 512;                 // lendable from 500 slots: 24 + 128 <= 500
        in.explicit_cache_slots = 500;          // 1000 MiB the 1600 MiB of free VRAM cannot cover
        const VramPlan p = plan_startup_vram(in);
        check(!p.ok && p.short_by_bytes > 0, "R1: the overflowing explicit cache is refused at planning");
        check(p.lend_slots > 0, "R1: the lend itself was fine - the budget is what fails");
    }

    // ---- review 2: an explicit cache resolves --prefill auto with the ONE policy, plan_lend_chunks
    {
        StartupVramInput in = plan_input((uint64_t) 460 * BLOB + 700 * MIB);
        in.prefill_borrow = true;
        in.prefill_auto = true;
        in.explicit_cache_slots = 460;          // 8192's loan does not fit; 4096's does
        const VramPlan p = plan_startup_vram(in);
        // the reference: the same view the planner builds, through the shared policy
        CacheLendView v;
        v.slots = 460;
        v.bytes = 460 * BLOB;
        v.max_blob = BLOB;
        int64_t ignored = 0;
        const LendOutcome ref = plan_lend_chunks(v, in.lend, in.costs, true, ignored);
        check(ref.chunk == 4096, "R2: the policy picks 4096 for this budget");
        check_ok(p, in.free_bytes, "R2: the explicit-cache auto plan is valid");
        check(!p.prefill_owned && p.selected_prefill == ref.chunk && p.lend_slots == ref.slots,
              "R2: startup equals plan_lend_chunks' decision, borrowed, not owned");
        // a non-list boundary through the same path
        StartupVramInput odd = in;
        odd.free_bytes = (uint64_t) 600 * BLOB + 700 * MIB;
        odd.explicit_cache_slots = 600;
        odd.lend.auto_ceiling = 24576;
        const VramPlan o2 = plan_startup_vram(odd);
        check_ok(o2, odd.free_bytes, "R2: the boundary plan is valid");
        check(!o2.prefill_owned && o2.selected_prefill == 8704,
              "R2: the explicit cache's bisection lands on 8704 too");
    }

    // ---- review 3: a sized native cache lends over its exact suffix - the uniform view would approve a loan
    // the real layout cannot hold
    {
        std::vector<int64_t> pairs;             // 300 hot pairs at 2 MiB, then 100 tail pairs at 0.25 MiB
        for (int i = 0; i < 300; ++i) pairs.push_back(2 * MIB);
        for (int i = 0; i < 100; ++i) pairs.push_back(256 * 1024);
        StartupVramInput in = plan_input((uint64_t) 700 * MIB + 625 * MIB + fake_bytes(1024, -1) +
                                         ((uint64_t) 64 << 20) + (uint64_t) 50 * MIB);
        in.prefill_borrow = true;
        in.prefill_chunk = 1024;
        in.explicit_cache_slots = 400;
        in.sized_slots_wanted = true;
        in.pair_slot_bytes = &pairs;
        in.profile_pairs = (int64_t) pairs.size();
        const VramPlan p = plan_startup_vram(in);
        check(p.ok && p.prefill_owned && p.selected_prefill == 1024,
              "R3: the false loan is refused; the prompt plans its own buffers");
        check(actual_cache_bytes(p) == (uint64_t) 625 * MIB, "R3: the cache is priced at its exact 625 MiB");
        check(plan_fits(p, in.free_bytes), "R3: the final invariant holds");
        // the uniform view WOULD have lent this chunk - the planner did not use it
        CacheLendView u;
        u.slots = 400;
        u.bytes = 400 * BLOB;
        u.max_blob = BLOB;
        check(u.slots_for_bytes(fake_bytes(1024, -1)) + 128 <= 400,
              "R3: the uniform view approves what the exact layout refuses");
    }

    // ---- review 7: the WDDM post-touch correction's figure - the reserve, the head and an owned prefill, a
    // borrowed one nothing
    {
        VramPlan w;
        w.user_reserve_bytes = (uint64_t) 700 * MIB;
        w.mtp_bytes = (uint64_t) 200 * MIB;
        w.mandatory_bytes = w.user_reserve_bytes + w.mtp_bytes;
        w.prefill_owned = true;
        w.prefill_bytes = (uint64_t) 1000 * MIB;
        check(post_cache_required_bytes(w) == (uint64_t) 1900 * MIB,
              "R7: the post-touch requirement is reserve + head + owned prefill (1900 MiB)");
        w.prefill_owned = false;
        check(post_cache_required_bytes(w) == (uint64_t) 900 * MIB,
              "R7: a borrowed prompt adds nothing to the post-touch requirement");
    }

    // ---- review 2, R8-R11 + R15: the prompt plan re-derived against the FINAL physical cache
    {
        VramPlan startup;   // a borrowed fixed-chunk plan, as the planner emits it
        startup.mandatory_bytes = (uint64_t) 700 * MIB;
        startup.prefill_borrow = true;
        startup.selected_prefill = 8192;
        startup.max_blob = BLOB;
        const LendOpts opts = fake_opts(8192);
        const LendCosts costs = fake_costs();
        const auto view_of = [](int64_t slots) {
            CacheLendView v;
            v.slots = slots;
            v.bytes = slots * BLOB;
            v.max_blob = BLOB;
            return v;
        };
        const uint64_t margin = (uint64_t) 64 << 20;
        // R8: the cache still lends 8192: borrowed, nothing owned, requirement = the mandatory items
        const EffectivePrefillPlan r8 = revalidate_prefill_after_cache(startup, view_of(760), opts, costs);
        check(r8.borrowed && !r8.owned && r8.chunk == 8192 && r8.lend_slots == 455 && r8.owned_bytes == 0,
              "R8: the borrowed fixed chunk stays borrowed");
        check(effective_post_cache_required(startup, r8) == (uint64_t) 700 * MIB,
              "R8: the effective requirement is the mandatory items alone");
        // R9: after a shrink 8192 no longer lends (500 slots still lend 4096): owned AT 8192 - the runtime
        // rejects a halved chunk as a loan - and the requirement carries the buffers
        const EffectivePrefillPlan r9 = revalidate_prefill_after_cache(startup, view_of(500), opts, costs);
        check(r9.owned && !r9.borrowed && r9.chunk == 8192,
              "R9: the non-lendable fixed chunk becomes owned at its original size");
        check(r9.owned_bytes == fake_bytes(8192, -1) + margin, "R9: the buffers are priced");
        check(effective_post_cache_required(startup, r9) == (uint64_t) 700 * MIB + r9.owned_bytes,
              "R9: the effective requirement carries the owned buffers");
        // R10: auto re-runs the policy against the shrunken cache: a smaller BORROWED chunk, nothing owned
        VramPlan auto_startup = startup;
        auto_startup.prefill_auto = true;
        const EffectivePrefillPlan r10 = revalidate_prefill_after_cache(auto_startup, view_of(460), opts, costs);
        check(r10.borrowed && !r10.owned && r10.chunk == 4096 && r10.owned_bytes == 0,
              "R10: auto falls to a smaller borrowed chunk, not owned buffers");
        check(effective_post_cache_required(auto_startup, r10) == (uint64_t) 700 * MIB,
              "R10: no owned buffers booked for the smaller loan");
        // R11: auto with nothing left to lend: the runtime's owned 1024 fallback, priced
        const EffectivePrefillPlan r11 = revalidate_prefill_after_cache(auto_startup, view_of(100), opts, costs);
        check(r11.owned && !r11.borrowed && r11.chunk == 1024 && r11.owned_bytes == fake_bytes(1024, -1) + margin,
              "R11: the auto fallback is owned at 1024 and priced");
        check(effective_post_cache_required(auto_startup, r11) == (uint64_t) 700 * MIB + r11.owned_bytes,
              "R11: the fallback is in the effective requirement");

        // R15: a sized cache revalidates over its exact offsets: the uniform view lends this chunk, the real
        // suffix does not (512 MiB needed; the top 272 real slots hold 369 MiB)
        std::vector<int64_t> pairs;
        for (int i = 0; i < 300; ++i) pairs.push_back(2 * MIB);
        for (int i = 0; i < 100; ++i) pairs.push_back(256 * 1024);
        std::vector<uint64_t> offs(pairs.size() + 1, 0);
        for (size_t i = 0; i < pairs.size(); ++i)
            offs[(size_t) i + 1] = offs[(size_t) i] + (uint64_t) ((pairs[(size_t) i] + 255) / 256 * 256);
        CacheLendView sized;
        sized.slots = (int64_t) pairs.size();
        sized.bytes = (int64_t) offs.back();
        sized.slot_offsets = offs.data();
        sized.max_blob = BLOB;
        VramPlan fixed1792 = startup;
        fixed1792.selected_prefill = 1792;
        const EffectivePrefillPlan r15 = revalidate_prefill_after_cache(fixed1792, sized, opts, costs);
        check(r15.owned && r15.chunk == 1792, "R15: the sized cache revalidates over exact offsets - owned");
        const EffectivePrefillPlan r15u = revalidate_prefill_after_cache(fixed1792, view_of(400), opts, costs);
        check(r15u.borrowed && r15u.chunk == 1792,
              "R15: the uniform view of the same slot count lends - the revalidation did not use it");
    }

    // ---- review 2, R12-R14: the correction's decision arithmetic
    {
        // R12: 1200 MiB free against a borrowed-looking 900 MiB requirement - but the cache no longer lends and
        // the prompt is owned 1000 MiB: 1900 MiB required, SHRINK ~764 MiB, never an accept
        const PostTouchVerdict r12 =
            post_touch_verdict((uint64_t) 1200 * MIB, (uint64_t) 1900 * MIB, 64ll << 20, true);
        check(!r12.accept && r12.give_back_bytes >= (int64_t) 700 * MIB,
              "R12: borrowed->owned forces a shrink of ~700+ MiB");
        // R13: the last allowed cache, still short: FAIL - the retry limit is not an accept
        const PostTouchVerdict r13 =
            post_touch_verdict((uint64_t) 800 * MIB, (uint64_t) 1900 * MIB, 64ll << 20, false);
        check(!r13.accept && r13.give_back_bytes == 0, "R13: the last allowed cache fails instead of being accepted");
        // R14: the last allowed cache that meets the budget: ACCEPT
        const PostTouchVerdict r14 =
            post_touch_verdict((uint64_t) 1900 * MIB, (uint64_t) 1900 * MIB, 64ll << 20, false);
        check(r14.accept, "R14: the last allowed cache that meets the budget is accepted");
    }

    // ---- review 3, R16-R18: the borrowing capability is ONE resolved flag; the planner plans owned whenever
    // it is off, whatever the profile's presence would suggest
    {
        StartupVramInput cap = plan_input((uint64_t) 760 * BLOB + 700 * MIB);
        cap.prefill_borrow = true;              // R16: available (profile + the residency map + not disabled)
        cap.prefill_chunk = 4096;
        const VramPlan on = plan_startup_vram(cap);
        check(on.ok && !on.prefill_owned && on.selected_prefill == 4096 && on.lend_slots > 0,
              "R16: with borrowing available the plan lends the chunk");
        // R17: the profile exists but the runtime's residency map will not - generate.cpp resolves the flag
        // off, and the planner must then book owned buffers, never a borrowed plan
        StartupVramInput off = cap;
        off.prefill_borrow = false;
        const VramPlan r17 = plan_startup_vram(off);
        check(r17.ok && r17.prefill_owned && r17.selected_prefill == 4096 && r17.lend_slots == 0,
              "R17: a resolved-off capability plans owned buffers at the same chunk");
        // R18: --no-prefill-borrow resolves the same flag off; runtime reads the same flag and stays owned
        check(!off.prefill_borrow && r17.prefill_owned && !r17.prefill_borrow,
              "R18: the disabled flag keeps both sides owned");
    }

    // ---- Enky's #765 findings, E1-E5: the fallbacks a WDDM driver used to paper over
    {
        const uint64_t margin = kOwnedPageMarginBytes;
        // E1: auto with nothing to lend - the owned 1024 fallback is an explicit, priced plan state
        StartupVramInput e1 = plan_input((uint64_t) 100 * BLOB + 700 * MIB);
        e1.prefill_borrow = true;
        e1.prefill_auto = true;
        const VramPlan p1 = plan_startup_vram(e1);
        check(p1.ok && p1.prefill_owned && p1.lend_slots == 0 && p1.selected_prefill > 0 &&
                  p1.selected_prefill <= 1024,
              "E1: the auto->owned fallback is an explicit, priced plan state (the chunk itself may be reduced)");
        // E2: the requested owned chunk does not fit, a smaller one does - reduced at planning, with a note
        StartupVramInput e2 = plan_input((uint64_t) 700 * MIB + fake_bytes(6144, -1) + margin +
                                         (uint64_t) 100 * MIB);
        e2.prefill_borrow = true;               // lending is tried first and fails; the chunk is owned either way
        e2.prefill_chunk = 24576;
        const VramPlan p2 = plan_startup_vram(e2);
        int64_t e2expect = 0;
        for (int64_t c = 24576 / 256 * 256; c >= 256; c -= 256)
            if (e2.free_bytes >= e2.user_reserve_bytes + e2.mtp_bytes + fake_bytes(c, -1) + margin) {
                e2expect = c; break;
            }
        check(p2.ok && p2.prefill_owned && p2.selected_prefill == e2expect && e2expect >= 4096,
              "E2: the requested chunk is reduced to the largest arithmetic fit");
        bool noted = false;
        for (const std::string& n : p2.notes) noted = noted || n.find("does not fit") != std::string::npos;
        check(noted, "E2: the reduction is said at planning, not discovered at Prefill::init()");
        // E3: even 512 does not fit - the plan refuses; 24576 never reaches an allocation
        StartupVramInput e3 = plan_input((uint64_t) 700 * MIB + fake_bytes(256, -1));
        e3.prefill_chunk = 24576;
        const VramPlan p3 = plan_startup_vram(e3);
        check(!p3.ok && p3.selected_prefill == 0 && p3.short_by_bytes > 0,
              "E3: nothing fits - the plan refuses instead of keeping the requested chunk");
        // E4: the decision is the arithmetic alone - the planner never asks an allocator, so a driver that
        // would 'succeed' the allocation (WDDM sysmem fallback) cannot change the outcome
        check(p2.ok && p2.selected_prefill == e2expect,
              "E4: the budget alone decides the chunk; hypothetical allocation success is not consulted");
        // E5: the pricing is the injected bytes_needed plus the shared page margin, never a fixed estimate -
        // priced for the chunk the plan actually selected
        check(p1.prefill_bytes == fake_bytes(p1.selected_prefill, -1) + margin,
              "E5: the fallback's price is bytes_needed + kOwnedPageMarginBytes");
    }

    // ---- review 4, R19-R28: the accepted plan is the runtime's contract - mode and size, less is allowed
    {
        const auto use = [](bool borrowed, int64_t chunk) {
            return RuntimePrefillUse{borrowed, chunk};
        };
        const EffectivePrefillPlan borrowed8192 = [] {
            EffectivePrefillPlan e;
            e.borrowed = true;
            e.chunk = 8192;
            e.lend_slots = 455;
            return e;
        }();
        const EffectivePrefillPlan owned768 = [] {
            EffectivePrefillPlan e;
            e.owned = true;
            e.chunk = 768;
            e.owned_bytes = fake_bytes(768, -1) + kOwnedPageMarginBytes;
            return e;
        }();
        const EffectivePrefillPlan owned2048 = [] {
            EffectivePrefillPlan e;
            e.owned = true;
            e.chunk = 2048;
            return e;
        }();
        // R19: accepted borrowed, runtime owned - the late failure #765 closes
        check(!check_runtime_prefill_use(borrowed8192, use(false, 8192)).ok, "R19: borrowed accepted, owned runtime refuses");
        // R20: accepted borrowed, matching loan
        check(check_runtime_prefill_use(borrowed8192, use(true, 8192)).ok, "R20: borrowed accepted, borrowed runtime passes");
        // R21: accepted owned, runtime borrowed
        check(!check_runtime_prefill_use(owned2048, use(true, 2048)).ok, "R21: owned accepted, borrowed runtime refuses");
        // R22 + R24: the auto owned fallback reduced below 1024 cannot grow back - this is the 5b6cb2f bug
        StartupVramInput r22 = plan_input((uint64_t) 900 * MIB);   // 1024 owned does not fit; 768 does
        r22.prefill_borrow = true;
        r22.prefill_auto = true;
        const VramPlan p22 = plan_startup_vram(r22);
        check(p22.ok && p22.prefill_owned && p22.selected_prefill == 768,
              "R22: the auto owned fallback is planned at 768, below the 1024 candidate");
        const EffectivePrefillPlan accepted768 = [] {
            EffectivePrefillPlan e;
            e.owned = true;
            e.chunk = 768;
            return e;
        }();
        check(check_runtime_prefill_use(accepted768, use(false, 768)).ok,
              "R22: the runtime runs the accepted 768 for a long prompt");
        check(!check_runtime_prefill_use(accepted768, use(false, 1024)).ok,
              "R22: the runtime fallback restored to 1024 refuses - the accepted 768 is the ceiling");
        // R23: a shorter prompt may run less
        check(check_runtime_prefill_use(accepted768, use(false, 512)).ok, "R23: owned 512 for a short prompt passes");
        // R24: runtime owned larger than accepted
        check(!check_runtime_prefill_use(accepted768, use(false, 1024)).ok, "R24: owned 1024 over accepted 768 refuses");
        // R25: a borrowed auto chunk larger than accepted refuses (the runtime ceiling is the contract, not
        // o.prefill_chunk, which an auto scan does not treat as its ceiling)
        check(!check_runtime_prefill_use(
                  [] { EffectivePrefillPlan e; e.borrowed = true; e.chunk = 4096; return e; }(),
                  use(true, 8192)).ok,
              "R25: borrowed 8192 over accepted 4096 refuses");
        // R26: a shorter borrowed request is allowed
        check(check_runtime_prefill_use(borrowed8192, use(true, 2048)).ok, "R26: borrowed 2048 under accepted 8192 passes");
        // the loan's BYTES are part of the contract: a shorter chunk under a different ring rule could price a
        // bigger loan than the accepted one, so the chunk alone does not prove safety
        EffectivePrefillPlan loan_cap = borrowed8192;
        loan_cap.lend_bytes = (uint64_t) 4 << 30;   // the accepted loan's bytes
        check(check_runtime_prefill_use(loan_cap, RuntimePrefillUse{true, 4096, (uint64_t) 2 << 30}).ok,
              "R34: a shorter request's smaller loan passes the byte ceiling");
        check(!check_runtime_prefill_use(loan_cap, RuntimePrefillUse{true, 512, (uint64_t) 5 << 30}).ok,
              "R34: a short chunk whose ring prices a bigger loan than accepted refuses");
        // R27/R28's rules are the same two checks the serve guard runs (its GPU hook, STRATA_TEST_SERVE_DROP_LOAN,
        // exercises the refused direction on hardware)
        check(!check_runtime_prefill_use(borrowed8192, use(false, 4096)).ok,
              "R27: the serve guard's refused direction (borrowed accepted, no loan)");
        check(check_runtime_prefill_use(owned2048, use(false, 2048)).ok,
              "R28: the serve guard's passing direction (owned accepted, owned runtime)");
    }

    if (fails == 0) std::fprintf(stderr, "vram_plan_test: all checks passed\n");
    return fails == 0 ? 0 : 1;
}
