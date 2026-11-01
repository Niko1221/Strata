// tests/core/dflash2_slots_test.cpp - the per-conversation slots of one DFlash2 drafter (include/strata/core/
// dflash2_slots.hpp): the byte size of a slot's K/V rings and of a park blob built from them, how many of a
// requested number of slots fit the VRAM still free after load(), the host ring_pos_ mirror sizing, the per-
// plane bytes the vram_ accounting records, the slot count a failed set_n_slots growth leaves behind, and the
// per-slot taps guard.  The numbers are the default pack geometry (window 2048 -> ring 2064 slots, L 5, NKV 8,
// HD 128) computed by hand: one ring plane (K or V, one layer) holds 2064 * 8 * 128 = 2113536 elements,
// 4227072 bytes in fp16; K and V, five layers: 42270720 bytes.
#include "strata/core/dflash2_slots.hpp"

#include <cstdio>
#include <string>

using strata::core::df2_batch_slot_max_k;
using strata::core::df2_bind_bytes;
using strata::core::df2_graphs_valid_for;
using strata::core::df2_ring_bytes;
using strata::core::df2_ring_mirror_entries;
using strata::core::df2_ring_mirror_index;
using strata::core::df2_ring_plane_bytes;
using strata::core::df2_slot_ok;
using strata::core::df2_slots_after_failed_grow;
using strata::core::df2_slots_fit;
using strata::core::df2_state_bytes;
using strata::core::kDf2MaxSlots;

// Compile-time proofs of the audited decisions (report-audit-1-3 findings 1, 5, 6, 7, 8).  The compiler checks
// them before main() can run, so a regression fails the build even where the test binary cannot be started;
// the runtime checks in main() repeat the same facts with printed results.
namespace ct {
constexpr uint64_t ring = df2_ring_bytes(2064, 5, 8, 128, 2);
constexpr uint64_t graph = 1ull << 20;

// finding 8: the budget is VRAM still free after load(), so slot 0's already-allocated ring is NOT charged to
// it: one free set buys one extra slot (2 total), three free sets buy three extra slots (4 total).
static_assert(df2_slots_fit(ring + graph, ring, graph, 8) == 2, "df2_slots_fit: one free set = slot 0 + 1 extra");
static_assert(df2_slots_fit(3 * (ring + graph), ring, graph, 8) == 4, "df2_slots_fit: three free sets = slot 0 + 3 extra");
static_assert(df2_slots_fit(ring + graph - 1, ring, graph, 8) == 1, "df2_slots_fit: one byte short of a free set = slot 0 only");

// finding 1: the host ring_pos_ mirror has ring_slots entries wherever a slot comes to exist - never a fixed
// default like the default pack's 2064, which an 8k-window pack (ring 8208) would write past.
static_assert(df2_ring_mirror_entries(8208) == 8208, "the mirror is sized by ring_slots, not by kRing (8208, not 2064)");
static_assert(df2_ring_mirror_index(1000000007, 8208) < df2_ring_mirror_entries(8208), "no indexed write passes the mirror");

// finding 6: alloc_slot_rings records the bytes per ring plane, free_slot subtracts exactly them again;
// 2 * L of these planes are one slot's rings, so vram_ cannot drift between allocate and free.
static_assert(2 * 5 * df2_ring_plane_bytes(2064, 8, 128, 2) == ring, "recorded plane bytes * 2L = one slot's rings");

// finding 5: a failed set_n_slots growth leaves the slot count it started with, whatever it had pushed.
static_assert(df2_slots_after_failed_grow(1, 3) == 1, "failed growth: back to the old count, not old + pushed");

// finding 7: the taps guard is per-slot pointer identity - the slot's own captured pointer, not a
// drafter-wide one that another slot's prepare() could move.
constexpr const float t1 = 0, t2 = 0;
static_assert(df2_graphs_valid_for(&t1, &t1) && !df2_graphs_valid_for(&t1, &t2), "the taps guard reads the slot's own captured pointer");
static_assert(df2_batch_slot_max_k(8, 4, 4, 6) == 1, "8-row window with 4 selected+ready slots fits at most 1 proposal per slot");
static_assert(df2_batch_slot_max_k(8, 3, 3, 6) == 3, "8-row window with 3 selected+ready slots fits up to 3 proposals on one slot");
static_assert(df2_batch_slot_max_k(16, 4, 4, 3) == 3, "16-row window with 4 slots capped at depth 3 allows all 3 proposals");
}  // namespace ct

namespace {
int g_fail = 0;
long g_checks = 0;
void check(bool ok, const std::string& what) {
    ++g_checks;
    if (ok) return;
    std::fprintf(stderr, "FAIL: %s\n", what.c_str());
    ++g_fail;
}
void check_eq(uint64_t got, uint64_t want, const std::string& what) {
    ++g_checks;
    if (got == want) return;
    std::fprintf(stderr, "FAIL: %s (got %llu, want %llu)\n", what.c_str(), (unsigned long long) got, (unsigned long long) want);
    ++g_fail;
}
}  // namespace

int main() {
    // the cap: one drafter serves at most kDf2MaxSlots conversations, and the batch window holds 8 rows
    check(kDf2MaxSlots == 8, "kDf2MaxSlots is 8");

    // the default geometry, fp16 ring: one plane 2064*8*128*2 = 4227072, K+V over 5 layers = 42270720
    check_eq(df2_ring_bytes(2064, 5, 8, 128, 2), 2064ull * 8 * 128 * 2 * 2 * 5, "ring bytes: the arithmetic spelled out");
    check_eq(df2_ring_bytes(2064, 5, 8, 128, 2), 42270720ull, "default geometry, fp16: 40.3125 MiB of rings per slot");
    check_eq(df2_ring_bytes(2064, 5, 8, 128, 4), 84541440ull, "the fp32 ring (elem 4) doubles it");
    check_eq(df2_ring_bytes(0, 5, 8, 128, 2), 0, "no ring slots: no ring bytes");
    check_eq(df2_ring_bytes(2064, 1, 8, 128, 2), 8454144ull, "one layer is a fifth of the five");

    // the park blob: 128-byte header + the ring_slots int32 position table + the planar K/V
    check_eq(df2_state_bytes(2064, 5, 8, 128, 2), 128 + 2064ull * 4 + 42270720ull, "state bytes = header + slot table + rings");
    check_eq(df2_state_bytes(2064, 5, 8, 128, 2), 42279104ull, "default geometry, fp16 state: 42279104");
    check_eq(df2_state_bytes(2064, 5, 8, 128, 4), 128 + 2064ull * 4 + 84541440ull, "the fp32 state doubles only its ring part");
    check_eq(df2_state_bytes(2064, 5, 8, 128, 4), 84549824ull, "default geometry, fp32 state: 84549824");

    // the equality Dflash2Drafter::state_bytes() rests on: the same formula the drafter states by hand
    const auto drafter_formula = [](int ring, int L, int NKV, int HD, int elem) {
        return 128 + (uint64_t) ring * 4 + (uint64_t) L * 2 * ((uint64_t) ring * NKV * HD * elem);
    };
    check_eq(df2_state_bytes(2064, 5, 8, 128, 2), drafter_formula(2064, 5, 8, 128, 2), "state_bytes equals the drafter's formula (default fp16)");
    check_eq(df2_state_bytes(2064, 5, 8, 128, 4), drafter_formula(2064, 5, 8, 128, 4), "state_bytes equals the drafter's formula (default fp32)");
    check_eq(df2_state_bytes(8208, 5, 8, 128, 2), drafter_formula(8208, 5, 8, 128, 2), "state_bytes equals the drafter's formula (the 8k-window ring)");
    check_eq(df2_state_bytes(64, 1, 1, 64, 3), drafter_formula(64, 1, 1, 64, 3), "state_bytes equals the drafter's formula (a small odd geometry)");

    // how many requested slots fit the VRAM still free after load() (audit finding 8): load() has already
    // allocated slot 0's rings, so the budget buys the EXTRA slots - each one a ring set plus its own graph
    // set - and the answer counts slot 0.  per_slot_graph_overhead_bytes is a measured number: one slot's 8
    // captured round graphs, read off the VRAM delta (rocm-smi / nvidia-smi) around prepare() on the card;
    // 0 means "not measured yet" and the fit counts rings only.
    const uint64_t ring = df2_ring_bytes(2064, 5, 8, 128, 2), graph = 1ull << 20;
    check_eq(df2_slots_fit(ring + graph, ring, graph, 8), 2, "a free budget of one set fits one extra slot plus the loaded slot 0");
    check_eq(df2_slots_fit(ring + graph - 1, ring, graph, 8), 1, "one byte short of one extra set fits only the already-loaded slot 0");
    check_eq(df2_slots_fit(3 * (ring + graph), ring, graph, 8), 4, "a budget of 3 extra sets fits 3 extra slots plus slot 0 (the ring is not charged twice)");
    check_eq(df2_slots_fit(3 * (ring + graph) - 1, ring, graph, 8), 3, "one byte short of 3 extra sets fits 2 extra plus slot 0");
    check_eq(df2_slots_fit(ring - 1, ring, graph, 8), 1, "a budget below one extra ring still gives 1 (single-slot mode is always allowed)");
    check_eq(df2_slots_fit(ring, ring, 0, 8), 2, "with the graph overhead unmeasured (0), one free ring buys one extra slot");
    check_eq(df2_slots_fit(ring - 1, ring, 0, 8), 1, "unmeasured overhead, budget below one ring: still slot 0 only");
    check_eq(df2_slots_fit(0, ring, graph, 8), 8, "budget 0 means no cap");
    check_eq(df2_slots_fit(0, ring, graph, 3), 3, "budget 0 keeps the request below the cap");
    check_eq(df2_slots_fit(1ull << 40, ring, graph, 40), (uint64_t) kDf2MaxSlots, "a huge budget stops at kDf2MaxSlots");
    check_eq(df2_slots_fit(2 * (ring + graph), ring, graph, 40), 3, "2 extra sets with a request above the cap: 3 slots");
    check_eq(df2_slots_fit(0, ring, graph, 0), 1, "requested 0 clamps to 1");
    check_eq(df2_slots_fit(0, ring, graph, -5), 1, "requested below 1 clamps to 1");
    check_eq(df2_slots_fit(ring - 1, ring, graph, 0), 1, "requested 0 under a tiny budget is still 1");
    check_eq(df2_slots_fit(5 * (ring + graph), 0, 0, 8), 8, "free slots (zero cost) reach the cap under any budget");

    // the host ring_pos_ mirror sizing (audit finding 1): every writer indexes it with pos % ring_slots -
    // context()/draft() writes, rewind()'s scan, save_state()'s memcpy of ring_slots entries - so the mirror
    // is ring_slots entries.  The old fixed default was the default pack's 2064; an 8k-window pack (window
    // 8192 + 16 slack = 8208 ring slots) overran it into the heap.
    check_eq(df2_ring_mirror_entries(2064), 2064, "the default pack's ring: 2064 mirror entries");
    check_eq(df2_ring_mirror_entries(8208), 8208, "the 8k-window pack's ring: 8208 mirror entries, not the default 2064");
    check_eq(df2_ring_mirror_entries(0), 0, "no geometry yet: no mirror entries (a slot that predates load() is sized by it)");
    check_eq(df2_ring_mirror_entries(-4), 0, "a non-positive ring sizes no mirror");
    check(df2_ring_mirror_entries(8208) != df2_ring_mirror_entries(2064), "mirror entries follow ring_slots, never a fixed default");
    check_eq(df2_ring_mirror_index(8207, 8208), 8207, "the last position before the wrap indexes the last cell");
    check_eq(df2_ring_mirror_index(8208, 8208), 0, "the mirror wraps at ring_slots");
    check_eq(df2_ring_mirror_index(8209, 8208), 1, "one past the wrap is cell 1");
    for (const int ring_slots : {17, 2064, 8208})
        for (int64_t pos = 0; pos < 262144; pos += 4093)
            check(df2_ring_mirror_index(pos, ring_slots) < df2_ring_mirror_entries(ring_slots),
                  "indexed write stays inside the mirror (ring " + std::to_string(ring_slots) + ", pos " + std::to_string(pos) + ")");
    check(df2_ring_mirror_index(2147483647ll, 2064) < df2_ring_mirror_entries(2064), "the largest int32 position stays inside the default mirror");
    check(df2_ring_mirror_index(999999999999ll, 8208) < df2_ring_mirror_entries(8208), "a position past 2^40 stays inside the 8k-window mirror");

    // the per-plane bytes recorded at allocation (audit finding 6): alloc_slot_rings adds these to vram_ and
    // free_slot subtracts the stored number - not a recompute from the geometry at free time, which a
    // set_fp16_ring or a second load() between allocate and free would make drift (vram_ is unsigned).
    check_eq(df2_ring_plane_bytes(2064, 8, 128, 2), 4227072ull, "one fp16 ring plane (2064 cells of 8 heads x 128) matches the header's hand number");
    check_eq(df2_ring_plane_bytes(2064, 8, 128, 4), 8454144ull, "the fp32 plane doubles it");
    check_eq(df2_ring_plane_bytes(8208, 8, 128, 2), 8208ull * 8 * 128 * 2, "the 8k-window fp16 plane");
    check_eq(df2_ring_plane_bytes(0, 8, 128, 2), 0, "no ring slots: no plane bytes");
    struct Geom { int ring, L, NKV, HD, elem; };
    for (const Geom g : {Geom{2064, 5, 8, 128, 2}, Geom{8208, 5, 8, 128, 2}, Geom{2064, 5, 8, 128, 4}, Geom{64, 1, 1, 64, 3}, Geom{17, 8, 4, 64, 2}}) {
        check_eq(df2_ring_bytes(g.ring, g.L, g.NKV, g.HD, g.elem),
                 2ull * (uint64_t) g.L * df2_ring_plane_bytes(g.ring, g.NKV, g.HD, g.elem),
                 "a slot's rings are 2*L planes of exactly the recorded plane bytes (ring " + std::to_string(g.ring) +
                 ", L " + std::to_string(g.L) + ", elem " + std::to_string(g.elem) + ")");
    }
    check(df2_ring_plane_bytes(2064, 8, 128, 2) != df2_ring_plane_bytes(2064, 8, 128, 4),
          "a geometry change after allocate makes a recompute disagree with the stored plane bytes - the drift finding 6 is about");

    // the slot count a failed growth leaves (audit finding 5): set_n_slots growth is all-or-nothing - the
    // slots pushed before the failure are freed and dropped, so n_slots() and vram_bytes() again describe
    // the drafter that could serve its existing slots before the call.
    for (int pushed = 0; pushed <= 7; ++pushed)
        check_eq(df2_slots_after_failed_grow(1, pushed), 1, "a grow from 1 that fails after pushing " + std::to_string(pushed) + " leaves 1");
    for (int pushed = 0; pushed <= 4; ++pushed)
        check_eq(df2_slots_after_failed_grow(4, pushed), 4, "a grow from 4 that fails after pushing " + std::to_string(pushed) + " leaves 4");
    check_eq(df2_slots_after_failed_grow(0, 0), 1, "a drafter always holds at least slot 0");

    // the taps guard is per-slot (audit finding 7): a slot's captured graphs read the buffer that slot was
    // captured on (Slot::taps_).  A drafter-wide pointer let prepare(taps2) for one slot clear another
    // slot's guard and launch its stale graphs.  All slots draft into ONE shared staging buffer by design
    // (the batch loop stages each slot's rows into it), so the pointer only moves when the engine rebuilds
    // the buffer - and then every slot with graphs on the old one must fail its guard, not silently draft.
    static const float taps1_cell = 0, taps2_cell = 0;   // stand-in buffers; the guard compares pointer identity only
    const float* taps1 = &taps1_cell;
    const float* taps2 = &taps2_cell;
    check(df2_graphs_valid_for(nullptr, taps1), "a slot with no captured graphs yet: its first draft() captures on this taps");
    check(df2_graphs_valid_for(taps1, taps1), "the buffer the slot captured on stays valid for that slot");
    check(!df2_graphs_valid_for(taps1, taps2), "a draft on another buffer is refused: no stale-taps graph launch");
    check(!df2_graphs_valid_for(taps2, taps1), "and the other way round");
    check(df2_graphs_valid_for(taps1, taps1), "a prepare(taps2) on another slot changes nothing for this slot: its own pointer still decides");

    // the active-slot range test
    check(df2_slot_ok(0, 1), "slot 0 of a single-slot drafter is the only one");
    check(!df2_slot_ok(1, 1), "slot 1 is out of range for a single-slot drafter");
    check(!df2_slot_ok(-1, 1), "slot -1 is out of range");
    check(df2_slot_ok(7, 8), "slot 7 is the last of 8");
    check(!df2_slot_ok(8, 8), "slot 8 is out of range for 8 slots");
    check(!df2_slot_ok(0, 0), "no slot is valid in 0 slots");
    check(!df2_slot_ok(-1, 0), "slot -1 in 0 slots is out of range");

    // the bind sizing law the engine budgets with, before it allocates the draft head
    check(df2_bind_bytes(0, 0, 0, 7) == 0ull, "no head and no vocabulary: no device bytes claimed");
    check(df2_bind_bytes(1024, 151936, 0, 7) == 151936ull * 4ull * 7ull,
          "an unpruned head stays where it is: only the draft's logit rows are new");
    {
        const uint64_t row = 1024, n_dv = 32768;
        check(df2_bind_bytes(row, 151936, (int64_t) n_dv, 7) == n_dv * row + n_dv * 4ull + 7ull * n_dv * 4ull,
              "a pruned vocabulary: the gathered head rows, the index map and the logit rows, all over the pruned size");
        check(df2_bind_bytes(0, 151936, (int64_t) n_dv, 7) < df2_bind_bytes(0, 151936, 0, 7),
              "a pruned vocabulary shrinks the draft's logit rows, which is what leaves room for the batch slots");
    }

    // df2_batch_slot_max_k: exact upper bound on proposal rows a slot can be offered in the next window
    check_eq((uint64_t) df2_batch_slot_max_k(8, 4, 4, 6), 1, "W=8, 4 selected, 4 ready -> at most 1 proposal/slot (4 total head rows)");
    check_eq((uint64_t) df2_batch_slot_max_k(8, 3, 3, 6), 3, "W=8, 3 selected, 3 ready -> up to 3 proposals on a slot");
    check_eq((uint64_t) df2_batch_slot_max_k(8, 2, 2, 3), 3, "W=8, 2 selected, 2 ready, k_cap=3 -> all 3 proposals fit");
    check_eq((uint64_t) df2_batch_slot_max_k(16, 4, 4, 3), 3, "W=16, 4 selected, 4 ready, k_cap=3 -> all 3 proposals fit");
    check_eq((uint64_t) df2_batch_slot_max_k(16, 4, 4, 6), 6, "W=16, 4 selected, 4 ready, k_cap=6 -> up to 6 proposals fit");
    check_eq((uint64_t) df2_batch_slot_max_k(8, 8, 8, 6), 1, "tight window never drops below 1 proposal when k_cap >= 1");
    check_eq((uint64_t) df2_batch_slot_max_k(8, 4, 4, 0), 0, "k_cap 0 -> 0 proposals");

    std::printf("dflash2_slots_test: %ld checks, %s\n", g_checks, g_fail ? "FAIL" : "PASS");
    return g_fail ? 1 : 0;
}
