// tests/core/vram_plan_test.cpp - #796 part C: the ADVISORY VRAM planner and the shared chunk/loan predictor.
//
// The planner is pure arithmetic over injected byte costs, so every case runs without a GPU.  The one semantic
// under test throughout: a predicted shortfall, an over-large explicit cache or an owned fallback is a WARNING
// the caller is expected to print - the advisory has no refusal to make and startup goes on.  Where the old
// #796 draft asserted `!p.ok` (a fatal plan), these cases assert `has_warning()` plus usable numbers, and the
// runtime-mismatch checker is asserted to REPORT, never to stop.
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
constexpr uint64_t MARGIN = 64ull << 20;   // the allocator margin the engine's exact owned price carries

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
    c.bytes_owned = [](int64_t chunk) { return fake_bytes(chunk, -1) + MARGIN; };
    c.bytes_owned_reserved = [](int64_t chunk) { return fake_bytes(chunk, -1) + MARGIN; };
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

strata::prefill::AdvisoryInput plan_input(uint64_t free_bytes) {
    strata::prefill::AdvisoryInput in;
    in.free_bytes = free_bytes;
    in.user_reserve_bytes = 700 * MIB;
    in.small_reserve_mib = 300;
    in.max_blob = BLOB;
    in.profile_pairs = -1;
    in.min_loan_slots = 1;
    in.lend = fake_opts();
    in.costs = fake_costs();
    return in;
}

// the invariant every healthy advisory keeps - the predicted final layout inside the free VRAM
bool plan_fits(const strata::prefill::VramAdvisory& p, uint64_t free_bytes) {
    return strata::prefill::post_cache_required_bytes(p) + strata::prefill::actual_cache_bytes(p) <= free_bytes;
}
void check_healthy(const strata::prefill::VramAdvisory& p, uint64_t free_bytes, const char* what) {
    check(!p.has_warning(), (std::string(what) + ": no warning").c_str());
    check(plan_fits(p, free_bytes), (std::string(what) + ": the predicted layout fits").c_str());
}
}  // namespace

int main() {
    using namespace strata::prefill;

    // ---- Case A: plenty of VRAM - a healthy plan, no warning, the requested chunk kept on its own buffers
    {
        AdvisoryInput in = plan_input(8 * GIB);
        in.prefill_chunk = 24576;              // own buffers: ~1.9 GiB of the fake pack's bytes
        const VramAdvisory p = advise_startup_vram(in);
        check_healthy(p, in.free_bytes, "A: the plan is healthy");
        check(p.prefill_owned && p.selected_prefill == 24576, "A: the requested chunk is kept, on its own buffers");
        check(p.prefill_bytes == fake_bytes(24576, -1) + MARGIN, "A: the exact owned price is reported");
        check(p.prefill_reserved_bytes == p.prefill_bytes, "A: the sizing rule reserves the same (no divergence note)");
        check(p.expert_slots >= 1, "A: the cache keeps what the owned buffers leave");
        // the sizing rule's reservation is reported beside the exact price: over-reserving (0.1.39's rule at big
        // chunks) is quiet; under-reserving honestly over-subscribes the card, and the warning says so
        AdvisoryInput over = in;
        over.costs.bytes_owned_reserved = [](int64_t chunk) {
            return (uint64_t) (160 + (chunk * 680) / 1024) * MIB;   // 0.1.39's rule: ~16 GiB at 24576
        };
        const VramAdvisory ov = advise_startup_vram(over);
        check(ov.prefill_reserved_bytes > ov.prefill_bytes, "A: the 0.1.39 rule over-reserves at this chunk");
        check(!ov.has_warning(), "A: an over-reserving rule is not a warning");
        AdvisoryInput under = in;
        under.costs.bytes_owned_reserved = [](int64_t c) { return fake_bytes(c, -1) + MARGIN - 10 * MIB; };
        const VramAdvisory un = advise_startup_vram(under);
        check(un.prefill_bytes > un.prefill_reserved_bytes, "A: the exact price exceeds the rule's reservation");
        check(un.has_warning() && un.short_by_bytes > 0,
              "A: under-reserving over-subscribes the card by the gap - WARNED");
        bool noted = false;
        for (const std::string& n : un.notes) noted = noted || n.find("MiB exact") != std::string::npos;
        check(noted, "A: the divergence is a note naming both prices");
    }

    // ---- Case B: --prefill auto predicted through the shared policy, including a non-list boundary
    {
        AdvisoryInput in = plan_input((uint64_t) 760 * BLOB + 700 * MIB);
        in.prefill_borrow = true;
        in.prefill_auto = true;
        in.lend.auto_ceiling = 24576;
        // 760 slots: 16384's loan does not fit - the bisection lands on 13824, a size no hardcoded list holds
        const VramAdvisory p = advise_startup_vram(in);
        check_healthy(p, in.free_bytes, "B: the plan is healthy");
        check(p.selected_prefill == 13824, "B: the largest fitting chunk on the grid is predicted");
        check(!p.prefill_owned && p.lend_slots == 631, "B: the chunk is lent, 631 slots");
        check(p.expert_slots >= p.lend_slots + 128, "B: the loan leaves the non-lendable minimum");
        // a smaller budget whose true maximum is 8704 - also off every list - to pin the bisection
        AdvisoryInput odd = plan_input((uint64_t) 600 * BLOB + 700 * MIB);
        odd.prefill_borrow = true;
        odd.prefill_auto = true;
        odd.lend.auto_ceiling = 24576;
        const VramAdvisory o = advise_startup_vram(odd);
        check_healthy(o, odd.free_bytes, "B: the non-round boundary is healthy");
        check(o.selected_prefill == 8704, "B: 8704 predicted");
    }

    // ---- Case C: nothing fits - a WARNING with a positive shortfall and usable numbers, never a refusal
    {
        AdvisoryInput in = plan_input(718 * MIB);   // the reserve alone eats nearly all of it
        in.prefill_chunk = 4096;
        const VramAdvisory p = advise_startup_vram(in);
        check(p.has_warning(), "C: the impossible configuration is WARNED");
        check(p.short_by_bytes > 0, "C: the shortfall is positive");
        check(p.expert_slots >= 0, "C: the advisory still reports a slot count (the caller prints and continues)");
        bool says_short = false;
        for (const std::string& w : p.warnings) says_short = says_short || w.find("short by") != std::string::npos;
        check(says_short, "C: the warning says how far the configuration is short");
        check(!p.suggestions.empty(), "C: the warning carries the knobs that make room");
        // the borrowed arm warns its own way: nothing can be lent, nothing can be owned
        AdvisoryInput b = plan_input(718 * MIB);
        b.prefill_borrow = true;
        b.prefill_auto = true;
        const VramAdvisory pb = advise_startup_vram(b);
        check(pb.has_warning() && pb.short_by_bytes > 0, "C: the borrowed arm warns deterministically too");
        // a configuration with no cache slot at all warns about --spec exactly when --spec is set (300 MiB free:
        // the #496 adaptation's floor, so the reserve keeps its 700 MiB and the cache is predicted zero slots)
        AdvisoryInput sp = plan_input(300 * MIB);
        sp.prefill_borrow = true;
        sp.prefill_chunk = 4096;
        sp.spec_needs_cache = true;
        const VramAdvisory ps = advise_startup_vram(sp);
        check(ps.has_warning() && ps.expert_slots == 0, "C: the zero-slot configuration is WARNED about");
        bool says_spec = false;
        for (const std::string& w : ps.warnings) says_spec = says_spec || w.find("--spec") != std::string::npos;
        sp.spec_needs_cache = false;
        const VramAdvisory pn = advise_startup_vram(sp);
        bool says_spec_without = false;
        for (const std::string& w : pn.warnings) says_spec_without = says_spec_without || w.find("--spec") != std::string::npos;
        check(says_spec && !says_spec_without, "C: the --spec warning appears only when --spec is set");
    }

    // ---- Case D: borrowing is never booked twice; owning is booked exactly once
    {
        AdvisoryInput owned = plan_input(8 * GIB);
        owned.prefill_chunk = 4096;            // no profile: the prompt path owns its buffers
        const VramAdvisory po = advise_startup_vram(owned);
        AdvisoryInput borrowed = plan_input(8 * GIB);
        borrowed.prefill_borrow = true;
        borrowed.prefill_chunk = 4096;         // the same chunk, lent from the cache instead
        const VramAdvisory pb = advise_startup_vram(borrowed);
        check_healthy(po, owned.free_bytes, "D: the owned plan is healthy");
        check_healthy(pb, borrowed.free_bytes, "D: the borrowed plan is healthy");
        check(po.prefill_owned && po.prefill_bytes == fake_bytes(4096, -1) + MARGIN,
              "D: the owned bytes are booked once, at the exact price");
        check(!pb.prefill_owned && pb.expert_budget_bytes == (uint64_t) 8 * GIB - pb.mandatory_bytes,
              "D: the borrowed plan deducts no prefill bytes from the cache's budget");
        check(pb.expert_budget_bytes - po.expert_budget_bytes >= po.prefill_bytes,
              "D: lending keeps the cache exactly the owned prefill's bytes larger");
        check(strata::prefill::post_cache_required_bytes(pb) == pb.mandatory_bytes,
              "D: a borrowed plan's post-cache need is the mandatory items alone - no prefill deduction");
        check(strata::prefill::post_cache_required_bytes(po) == po.mandatory_bytes + po.prefill_bytes,
              "D: an owned plan's post-cache need carries the prefill exactly once");
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
        // through the planner: the predicted sized cache prices over its exact offsets (the profile cap keeps
        // this cache tiny on purpose, so the lend falls to the owned fallback - warned, numbers usable)
        AdvisoryInput in = plan_input((uint64_t) 610 * BLOB + 700 * MIB);
        in.prefill_borrow = true;
        in.prefill_auto = true;
        in.sized_slots_wanted = true;
        in.profile_pairs = 3;
        const std::vector<int64_t> pairs = {1 * MIB, 1 * MIB, 1 * MIB};   // each aligned slot: 1 MiB, not 2
        in.pair_slot_bytes = &pairs;
        const VramAdvisory p = advise_startup_vram(in);
        check(p.has_warning() && p.prefill_owned,
              "E: the 3-slot cache cannot lend; the owned fallback is WARNED, not refused");
        check(!p.sized_slots.empty() && actual_cache_bytes(p) == 3 * MIB,
              "E: actual_cache_bytes is the sized layout's exact total");
        check(actual_cache_bytes(p) < (uint64_t) 3 * BLOB,
              "E: the sized cache holds fewer bytes than its uniform count prices");
        check(p.selected_prefill == 1024, "E: the owned fallback is the runtime's 1024 bound");
    }

    // ---- Case F: the reserve stays the user's knob; the #496 adaptation is PREDICTED, not made
    {
        AdvisoryInput in = plan_input(8 * GIB);
        in.prefill_chunk = 4096;
        const VramAdvisory a = advise_startup_vram(in);
        in.user_reserve_bytes = 1400 * MIB;
        in.reserve_given = true;
        const VramAdvisory b = advise_startup_vram(in);
        check_healthy(a, in.free_bytes, "F: the smaller reserve plans fine");
        check_healthy(b, in.free_bytes, "F: the bigger reserve plans fine");
        check(b.expert_budget_bytes == a.expert_budget_bytes - 700 * MIB,
              "F: the extra 700 MiB of reserve comes out of the cache alone");
        check(b.mandatory_bytes == a.mandatory_bytes + 700 * MIB, "F: the mandatory side carries the reserve");
        // a given reserve is never adapted; the default one is predicted as adapted (the runtime makes it).
        // A 700 MiB card with a 4096-token chunk cannot afford the loan either way - the advisory warns and
        // keeps every number usable, it does not refuse.
        AdvisoryInput small_card = plan_input(700 * MIB);   // the reserve alone would eat all of it
        small_card.prefill_borrow = true;
        small_card.prefill_chunk = 4096;
        small_card.min_loan_slots = 144;        // a 256-token loan plus the 128-slot floor, as generate aims for
        small_card.reserve_given = true;
        const VramAdvisory given = advise_startup_vram(small_card);
        check(given.has_warning() && given.short_by_bytes > 0 && given.reserve_adapted_from_mib == 0 &&
                  given.user_reserve_bytes == (uint64_t) 700 * MIB,
              "F: a reserve given on the command line is kept - and the shortfall is a WARNING");
        small_card.reserve_given = false;
        const VramAdvisory adapted = advise_startup_vram(small_card);
        check(adapted.reserve_adapted_from_mib == 700, "F: predicted adaptation from 700");
        check(adapted.user_reserve_bytes == (uint64_t) 412 * MIB, "F: the predicted adaptation leaves 412 MiB");
        check(adapted.expert_slots == 144 && adapted.user_reserve_bytes >= (uint64_t) 300 * MIB,
              "F: the predicted adaptation leaves exactly the working cache, above its floor");
        bool noted = false;
        for (const std::string& n : adapted.notes) noted = noted || n.find("expected to lower") != std::string::npos;
        check(noted, "F: the adaptation is a note about what the RUNTIME is expected to do");
        check(adapted.has_warning() && adapted.prefill_owned && adapted.selected_prefill == 4096,
              "F: the chunk the 144-slot cache cannot lend is predicted owned, and warned");
        check(plan_fits(adapted, 700 * MIB) == false,
              "F: the prediction honestly does not fit - that is what the warning says");
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

    // ---- R1: an explicit cache that does not fit beside the mandatory items is WARNED, and the lend that would
    // have worked is said too - the operator decides; nothing is refused or shrunk here
    {
        AdvisoryInput in = plan_input(1600 * MIB);
        in.prefill_borrow = true;
        in.prefill_chunk = 512;                 // lendable from 500 slots: 24 + 128 <= 500
        in.explicit_cache = true;
        in.explicit_cache_slots = 500;          // 1000 MiB the 1600 MiB of free VRAM cannot cover
        const VramAdvisory p = advise_startup_vram(in);
        check(p.has_warning() && p.short_by_bytes > 0, "R1: the overflowing explicit cache is WARNED at planning");
        check(p.lend_slots > 0, "R1: the lend itself was fine - the budget is what the warning names");
    }

    // ---- R2: an explicit cache resolves --prefill auto with the ONE policy, plan_lend_chunks
    {
        AdvisoryInput in = plan_input((uint64_t) 460 * BLOB + 700 * MIB);
        in.prefill_borrow = true;
        in.prefill_auto = true;
        in.explicit_cache = true;
        in.explicit_cache_slots = 460;          // 8192's loan does not fit; 4096's does
        const VramAdvisory p = advise_startup_vram(in);
        // the reference: the same view the planner builds, through the shared policy
        CacheLendView v;
        v.slots = 460;
        v.bytes = 460 * BLOB;
        v.max_blob = BLOB;
        int64_t ignored = 0;
        const LendOutcome ref = plan_lend_chunks(v, in.lend, in.costs, true, ignored);
        check(ref.chunk == 4096, "R2: the policy picks 4096 for this budget");
        check_healthy(p, in.free_bytes, "R2: the explicit-cache auto plan is healthy");
        check(!p.prefill_owned && p.selected_prefill == ref.chunk && p.lend_slots == ref.slots,
              "R2: the prediction equals plan_lend_chunks' decision, borrowed, not owned");
        // a non-list boundary through the same path
        AdvisoryInput odd = in;
        odd.free_bytes = (uint64_t) 600 * BLOB + 700 * MIB;
        odd.explicit_cache_slots = 600;
        odd.lend.auto_ceiling = 24576;
        const VramAdvisory o2 = advise_startup_vram(odd);
        check_healthy(o2, odd.free_bytes, "R2: the boundary plan is healthy");
        check(!o2.prefill_owned && o2.selected_prefill == 8704,
              "R2: the explicit cache's bisection lands on 8704 too");
    }

    // ---- R3: a sized native cache lends over its exact suffix - the uniform view would approve a loan
    // the real layout cannot hold
    {
        std::vector<int64_t> pairs;             // 300 hot pairs at 2 MiB, then 100 tail pairs at 0.25 MiB
        for (int i = 0; i < 300; ++i) pairs.push_back(2 * MIB);
        for (int i = 0; i < 100; ++i) pairs.push_back(256 * 1024);
        AdvisoryInput in = plan_input((uint64_t) 700 * MIB + 625 * MIB + fake_bytes(1024, -1) + MARGIN +
                                      (uint64_t) 50 * MIB);
        in.prefill_borrow = true;
        in.prefill_chunk = 1024;
        in.explicit_cache = true;
        in.explicit_cache_slots = 400;
        in.sized_slots_wanted = true;
        in.pair_slot_bytes = &pairs;
        in.profile_pairs = (int64_t) pairs.size();
        const VramAdvisory p = advise_startup_vram(in);
        check(p.prefill_owned && p.selected_prefill == 1024,
              "R3: the false loan is not taken; the prompt is predicted on its own buffers");
        check(actual_cache_bytes(p) == (uint64_t) 625 * MIB, "R3: the cache is priced at its exact 625 MiB");
        check(p.has_warning(), "R3: the owned fallback beside the explicit cache is WARNED, not refused");
        check(plan_fits(p, in.free_bytes), "R3: the predicted layout still fits");
        // the uniform view WOULD have lent this chunk - the planner did not use it
        CacheLendView u;
        u.slots = 400;
        u.bytes = 400 * BLOB;
        u.max_blob = BLOB;
        check(u.slots_for_bytes(fake_bytes(1024, -1)) + 128 <= 400,
              "R3: the uniform view approves what the exact layout refuses");
    }

    // ---- R7: the post-open check's figure - the reserve, the head and an owned prefill, a borrowed one nothing
    {
        VramAdvisory w;
        w.user_reserve_bytes = (uint64_t) 700 * MIB;
        w.mtp_bytes = (uint64_t) 200 * MIB;
        w.mandatory_bytes = w.user_reserve_bytes + w.mtp_bytes;
        w.prefill_owned = true;
        w.prefill_bytes = (uint64_t) 1000 * MIB;
        check(post_cache_required_bytes(w) == (uint64_t) 1900 * MIB,
              "R7: the post-open requirement is reserve + head + owned prefill (1900 MiB)");
        w.prefill_owned = false;
        check(post_cache_required_bytes(w) == (uint64_t) 900 * MIB,
              "R7: a borrowed prompt adds nothing to the post-open requirement");
    }

    // ---- R8-R15: the prompt plan re-derived against a FINAL physical cache (diagnostics only)
    {
        VramAdvisory startup;   // a borrowed fixed-chunk prediction, as the planner emits it
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
        // R8: the cache still lends 8192: borrowed, nothing owned
        const EffectivePrefillPlan r8 = revalidate_prefill_after_cache(startup, view_of(760), opts, costs);
        check(r8.borrowed && !r8.owned && r8.chunk == 8192 && r8.lend_slots == 455 && r8.owned_bytes == 0,
              "R8: the borrowed fixed chunk stays borrowed");
        // R9: after a shrink 8192 no longer lends (500 slots still lend 4096): owned AT 8192 - the runtime
        // rejects a halved chunk as a loan - and the mismatch is reportable
        const EffectivePrefillPlan r9 = revalidate_prefill_after_cache(startup, view_of(500), opts, costs);
        check(r9.owned && !r9.borrowed && r9.chunk == 8192,
              "R9: the non-lendable fixed chunk becomes owned at its original size");
        check(r9.owned_bytes == fake_bytes(8192, -1) + MARGIN, "R9: the buffers are priced");
        check(effective_post_cache_required(startup, r9) == (uint64_t) 700 * MIB + r9.owned_bytes,
              "R9: the effective requirement carries the owned buffers");
        // R10: auto re-runs the policy against the shrunken cache: a smaller BORROWED chunk, nothing owned
        VramAdvisory auto_startup = startup;
        auto_startup.prefill_auto = true;
        const EffectivePrefillPlan r10 = revalidate_prefill_after_cache(auto_startup, view_of(460), opts, costs);
        check(r10.borrowed && !r10.owned && r10.chunk == 4096 && r10.owned_bytes == 0,
              "R10: auto falls to a smaller borrowed chunk, not owned buffers");
        // R11: auto with nothing left to lend: the runtime's owned 1024 fallback, priced
        const EffectivePrefillPlan r11 = revalidate_prefill_after_cache(auto_startup, view_of(100), opts, costs);
        check(r11.owned && !r11.borrowed && r11.chunk == 1024 && r11.owned_bytes == fake_bytes(1024, -1) + MARGIN,
              "R11: the auto fallback is owned at 1024 and priced");
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
        VramAdvisory fixed1792 = startup;
        fixed1792.selected_prefill = 1792;
        const EffectivePrefillPlan r15 = revalidate_prefill_after_cache(fixed1792, sized, opts, costs);
        check(r15.owned && r15.chunk == 1792, "R15: the sized cache revalidates over exact offsets - owned");
        const EffectivePrefillPlan r15u = revalidate_prefill_after_cache(fixed1792, view_of(400), opts, costs);
        check(r15u.borrowed && r15u.chunk == 1792,
              "R15: the uniform view of the same slot count lends - the revalidation did not use it");
    }

    // ---- R12-R14: the post-touch verdict arithmetic (informational: the engine's own loop shrinks)
    {
        const PostTouchVerdict r12 =
            post_touch_verdict((uint64_t) 1200 * MIB, (uint64_t) 1900 * MIB, 64ll << 20, true);
        check(!r12.accept && r12.give_back_bytes >= (int64_t) 700 * MIB,
              "R12: borrowed->owned reports a shortfall of ~700+ MiB");
        const PostTouchVerdict r13 =
            post_touch_verdict((uint64_t) 800 * MIB, (uint64_t) 1900 * MIB, 64ll << 20, false);
        check(!r13.accept && r13.give_back_bytes == 0, "R13: the last cache reports the shortfall without a target");
        const PostTouchVerdict r14 =
            post_touch_verdict((uint64_t) 1900 * MIB, (uint64_t) 1900 * MIB, 64ll << 20, false);
        check(r14.accept, "R14: the last cache that meets the budget is accepted");
    }

    // ---- R16-R18: the borrowing capability is ONE resolved flag; the advisory predicts owned whenever it is
    // off, whatever the profile's presence would suggest
    {
        AdvisoryInput cap = plan_input((uint64_t) 760 * BLOB + 700 * MIB);
        cap.prefill_borrow = true;              // R16: available (profile + not disabled)
        cap.prefill_chunk = 4096;
        const VramAdvisory on = advise_startup_vram(cap);
        check_healthy(on, cap.free_bytes, "R16: the borrowing prediction is healthy");
        check(!on.prefill_owned && on.selected_prefill == 4096 && on.lend_slots > 0,
              "R16: with borrowing available the prediction lends the chunk");
        // R17: the capability resolved off (no profile or --no-prefill-borrow) plans owned at the same chunk
        AdvisoryInput off = cap;
        off.prefill_borrow = false;
        const VramAdvisory r17 = advise_startup_vram(off);
        check(!r17.has_warning() && r17.prefill_owned && r17.selected_prefill == 4096 && r17.lend_slots == 0,
              "R17: a resolved-off capability predicts owned buffers at the same chunk");
        // R18: the disabled flag keeps both sides owned - the runtime reads the same flag
        check(!off.prefill_borrow && r17.prefill_owned && !r17.prefill_borrow,
              "R18: the disabled flag keeps both sides owned");
    }

    // ---- E1-E5: the fallbacks a WDDM driver used to paper over, predicted and said
    {
        // E1: auto with nothing to lend - the owned 1024 fallback is an explicit, priced, WARNED state
        AdvisoryInput e1 = plan_input((uint64_t) 100 * BLOB + 700 * MIB);
        e1.prefill_borrow = true;
        e1.prefill_auto = true;
        const VramAdvisory p1 = advise_startup_vram(e1);
        check(p1.prefill_owned && p1.lend_slots == 0 && p1.selected_prefill > 0 && p1.selected_prefill <= 1024,
              "E1: the auto->owned fallback is an explicit, priced prediction (the chunk itself may be reduced)");
        // E2: the requested owned chunk does not fit, a smaller one does - the shortfall is WARNED with the knobs
        AdvisoryInput e2 = plan_input((uint64_t) 700 * MIB + fake_bytes(6144, -1) + MARGIN + (uint64_t) 100 * MIB);
        e2.prefill_borrow = true;               // lending is tried first and fails; the chunk is owned either way
        e2.prefill_chunk = 24576;
        const VramAdvisory p2 = advise_startup_vram(e2);
        check(p2.has_warning() && p2.short_by_bytes > 0,
              "E2: the over-budget chunk is a warning with a shortfall, not a stop");
        check(p2.selected_prefill == 24576, "E2: the advisory does not invent a chunk - the engine's init may");
        bool noted = false;
        for (const std::string& w : p2.warnings) noted = noted || w.find("short by") != std::string::npos;
        check(noted, "E2: the shortfall is said at planning, before Prefill::init() could hit it");
        // E3: even 512 does not fit - warned; the caller prints it and the engine's own error path stays theirs
        AdvisoryInput e3 = plan_input((uint64_t) 700 * MIB + fake_bytes(256, -1));
        e3.prefill_chunk = 24576;
        const VramAdvisory p3 = advise_startup_vram(e3);
        check(p3.has_warning() && p3.short_by_bytes > 0,
              "E3: nothing fits - the advisory WARNS instead of refusing startup");
        check(p3.selected_prefill == 24576 && p3.expert_slots >= 0,
              "E3: the numbers stay usable for the warning line");
        // E5: the price is the injected exact bytes plus the margin, for the chunk the plan actually reports
        check(p1.prefill_bytes == fake_bytes(p1.selected_prefill, -1) + MARGIN,
              "E5: the fallback's price is bytes_owned(selected)");
    }

    // ---- R19-R28: the runtime-vs-prediction checker - a mismatch is REPORTABLE, and every rule is
    // "less is allowed"; no result here is permission to stop
    {
        const auto use = [](bool borrowed, int64_t chunk) {
            return RuntimePrefillUse{borrowed, chunk};
        };
        const auto mismatch_of = [](const EffectivePrefillPlan& e, const RuntimePrefillUse& u) {
            return check_runtime_prefill_use(e, u).mismatch;
        };
        const EffectivePrefillPlan borrowed8192 = [] {
            EffectivePrefillPlan e;
            e.borrowed = true;
            e.chunk = 8192;
            e.lend_slots = 455;
            return e;
        }();
        const EffectivePrefillPlan owned2048 = [] {
            EffectivePrefillPlan e;
            e.owned = true;
            e.chunk = 2048;
            return e;
        }();
        // R19: predicted borrowed, runtime owned - the late failure part B closes; the checker NAMES it
        check(mismatch_of(borrowed8192, use(false, 8192)), "R19: borrowed predicted, owned runtime is a mismatch");
        // R20: predicted borrowed, matching loan
        check(!mismatch_of(borrowed8192, use(true, 8192)), "R20: borrowed predicted, borrowed runtime matches");
        // R21: predicted owned, runtime borrowed
        check(mismatch_of(owned2048, use(true, 2048)), "R21: owned predicted, borrowed runtime is a mismatch");
        // R22 + R24: the auto owned fallback's chunk is the ceiling; a runtime above it is a mismatch
        const EffectivePrefillPlan accepted768 = [] {
            EffectivePrefillPlan e;
            e.owned = true;
            e.chunk = 768;
            return e;
        }();
        check(!mismatch_of(accepted768, use(false, 768)), "R22: the runtime runs the predicted 768 for a long prompt");
        check(mismatch_of(accepted768, use(false, 1024)), "R22: the runtime fallback restored to 1024 is reported");
        // R23: a shorter prompt may run less
        check(!mismatch_of(accepted768, use(false, 512)), "R23: owned 512 for a short prompt matches");
        // R25: a borrowed auto chunk larger than predicted is reported
        check(mismatch_of(
                  [] { EffectivePrefillPlan e; e.borrowed = true; e.chunk = 4096; return e; }(),
                  use(true, 8192)),
              "R25: borrowed 8192 over predicted 4096 is a mismatch");
        // R26: a shorter borrowed request is allowed
        check(!mismatch_of(borrowed8192, use(true, 2048)), "R26: borrowed 2048 under predicted 8192 matches");
        // the loan's BYTES are part of the comparison: a shorter chunk under a different ring rule could price
        // a bigger loan than the predicted one, so the chunk alone does not prove safety
        EffectivePrefillPlan loan_cap = borrowed8192;
        loan_cap.lend_bytes = (uint64_t) 4 << 30;   // the predicted loan's bytes
        check(!mismatch_of(loan_cap, RuntimePrefillUse{true, 4096, (uint64_t) 2 << 30}),
              "R34: a shorter request's smaller loan matches the byte ceiling");
        check(mismatch_of(loan_cap, RuntimePrefillUse{true, 512, (uint64_t) 5 << 30}),
              "R34: a short chunk whose ring prices a bigger loan than predicted is reported");
        // the `why` text is what the caller's warning line prints
        const RuntimePlanCheck c = check_runtime_prefill_use(borrowed8192, use(false, 8192));
        check(c.mismatch && c.why != nullptr && std::string(c.why).find("borrow") != std::string::npos,
              "R19: the mismatch names the difference, for the caller's warning");
    }

    // ---- env flags: one parser, unset means the caller's default, =0 is off, anything else is on
    {
        check(env_enabled(nullptr, true) && env_enabled("", true), "env: unset/empty with default on is on");
        check(!env_enabled(nullptr, false) && !env_enabled("", false), "env: unset/empty with default off is off");
        check(!env_enabled("0", true), "env: =0 is off even with default on (STRATA_ADVISORY_PLAN=0)");
        check(!env_enabled("0", false), "env: =0 is off (STRATA_POSTTOUCH=0)");
        check(env_enabled("1", false) && env_enabled("true", false) && env_enabled("yes", false),
              "env: any non-zero value is on");
        check(!env_enabled("00", true), "env: a value starting with 0 is off");
    }

    if (fails == 0) std::fprintf(stderr, "vram_plan_test: all checks passed\n");
    return fails == 0 ? 0 : 1;
}
