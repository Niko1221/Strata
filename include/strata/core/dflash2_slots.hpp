// include/strata/core/dflash2_slots.hpp - the per-conversation slots of one DFlash2 drafter, as pure sizing and
// capacity logic so a CPU test can pin it down (the drafter itself is include/strata/core/dflash2.hpp).
//
// One drafter instance carries the weights, head and scratch for the card (about 0.8 GB); each conversation it
// serves adds a K/V ring set (L layers of planar K and V) and a set of captured round graphs (a graph bakes the
// ring pointers it recorded, so a ring needs its own). A park blob is the 128-byte self-describing header, the
// ring slot position table (one int32 per ring slot) and the ring bytes themselves.
#pragma once

#include <cstdint>

namespace strata::core {

/// The hard cap on conversations served by one drafter: the batch window holds kVerifyMaxT = 8 rows, so no batch
/// mode ever asks for more, and the rings/graphs of more would only waste VRAM.
constexpr int kDf2MaxSlots = 8;

/// One slot's K+V ring bytes: L layers, K and V, `ring_slots` cells of NKV * HD elements at `elem_bytes` each
/// (the head-major [NKV][ring_slots][HD] layout the kv_ring kernels write; fp16 rings use elem_bytes 2).
inline constexpr uint64_t df2_ring_bytes(int ring_slots, int L, int NKV, int HD, int elem_bytes) {
    return (uint64_t) L * 2 * (uint64_t) ring_slots * (uint64_t) NKV * (uint64_t) HD * (uint64_t) elem_bytes;
}

/// One ring plane's bytes: K or V for a single layer - `ring_slots` cells of NKV * HD elements at `elem_bytes`
/// each - so one slot's rings are 2 * L planes of exactly this size (df2_ring_bytes = 2 * L * this).  A slot
/// records the number it was allocated with and free_slot subtracts that stored number from vram_, never a
/// recompute from the geometry at free time: a set_fp16_ring or a second load() between allocate and free would
/// otherwise make vram_bytes() drift, or underflow it (vram_ is unsigned) when the recompute comes out bigger.
inline constexpr uint64_t df2_ring_plane_bytes(int ring_slots, int NKV, int HD, int elem_bytes) {
    return (uint64_t) ring_slots * (uint64_t) NKV * (uint64_t) HD * (uint64_t) elem_bytes;
}

/// The bytes one Dflash2Drafter::save_state() blob holds: the 128-byte header, the slot table (one int32 per ring
/// slot) and the rings. This restates the drafter's state_bytes() formula; the drafter calls this so the two
/// cannot drift.
inline constexpr uint64_t df2_state_bytes(int ring_slots, int L, int NKV, int HD, int elem_bytes) {
    return 128 + (uint64_t) ring_slots * 4 + df2_ring_bytes(ring_slots, L, NKV, HD, elem_bytes);
}

/// Host entries the ring_pos_ mirror needs for a ring of `ring_slots` cells.  Every writer indexes it with
/// pos % ring_slots - context()/draft()'s position writes, rewind()'s scan, save_state()'s memcpy of ring_slots
/// entries - so a slot's mirror is exactly ring_slots entries long, sized wherever a slot comes to exist
/// (load() sizes every slot; set_n_slots() sizes each one it adds).  Never a fixed default: the default pack's
/// 2064 sat under an 8k-window pack's 8208 and its writes landed in the heap.  A slot that predates load()
/// keeps 0 entries (geometry unknown) rather than a default that could hide a missed sizing.
inline constexpr uint64_t df2_ring_mirror_entries(int ring_slots) {
    return ring_slots < 1 ? 0 : (uint64_t) ring_slots;
}

/// The mirror cell holding the write position of absolute position `pos` (>= 0) in a ring of `ring_slots`
/// cells.  Always < df2_ring_mirror_entries(ring_slots) - the invariant the drafter asserts at every indexed
/// mirror write.
inline constexpr uint64_t df2_ring_mirror_index(int64_t pos, int ring_slots) {
    return ring_slots < 1 ? 0 : (uint64_t) (pos % (int64_t) ring_slots);
}

/// How many of `requested` slots fit the VRAM that is still free: load() has already allocated the pack and
/// slot 0's rings, so slot 0 is not charged to `budget_bytes` - the budget buys the EXTRA slots, each costing
/// one ring set (`ring_bytes`) plus its own captured graph set (`per_slot_graph_overhead_bytes`; measured on
/// the card as the VRAM delta around prepare()'s 8 cudaGraphInstantiate calls for one slot (rocm-smi /
/// nvidia-smi), 0 = not measured, which counts rings only).  The returned count includes slot 0.  Clamped to
/// [1, kDf2MaxSlots]; a request below 1 becomes 1; a `budget_bytes` of 0 means "no cap".  Single-slot mode is
/// always allowed: those bytes are already spent before any choice exists.
inline constexpr int df2_slots_fit(uint64_t budget_bytes, uint64_t ring_bytes, uint64_t per_slot_graph_overhead_bytes, int requested) {
    const int want = requested < 1 ? 1 : (requested > kDf2MaxSlots ? kDf2MaxSlots : requested);
    if (budget_bytes == 0) return want;
    const uint64_t per_slot = ring_bytes + per_slot_graph_overhead_bytes;
    int fit = 1;
    for (int s = 2; s <= want; ++s) {
        if ((uint64_t) (s - 1) * per_slot > budget_bytes) break;   // only the extra slots pay out of the budget
        fit = s;
    }
    return fit;
}

/// The slot count a failed set_n_slots growth leaves behind: growth from `have` toward more that fails after
/// `pushed` extra slots were allocated is all-or-nothing - the pushed slots are freed and dropped, and the
/// drafter is back to the `have` slots it had before the call, so n_slots() and vram_bytes() again describe a
/// drafter that can serve its existing conversations.
inline constexpr int df2_slots_after_failed_grow(int have, int pushed) {
    (void) pushed;   // however many were pushed before the failure, every one of them goes back
    return have < 1 ? 1 : have;
}

/// Whether the active slot's captured round graphs may launch against `taps`: they were captured on the buffer
/// this slot's own captured pointer names (Slot::taps_ - per slot, because the graphs belong to the slot), or
/// the slot has captured nothing yet (null) and its first draft() captures on `taps`.  A drafter-wide pointer
/// would let prepare(taps2) on one slot clear another slot's guard while its graphs still read taps1 - a
/// silent stale-taps launch.  By design all slots draft into ONE shared taps staging buffer (the batch loop
/// stages each slot's committed rows into it before the round), so the buffer only moves when the engine
/// rebuilds it, and then every slot that captured on the old one fails its guard until recaptured.
inline constexpr bool df2_graphs_valid_for(const float* captured, const float* taps) {
    return captured == nullptr || captured == taps;
}

/// Whether `i` is a valid active slot of a drafter holding `n` slots.
inline constexpr bool df2_slot_ok(int i, int n) { return i >= 0 && i < n; }

/// Exact device bytes Dflash2Drafter::bind() allocates: with a pruned draft vocabulary (`n_dvocab > 0`), the gathered
/// head rows (`n_dvocab * head_row_bytes`), the int32 index map (`n_dvocab * 4`), and `k_drafts` float logit rows over
/// `n_dvocab`; without one (`n_dvocab <= 0`), the full head stays in place and only `k_drafts * n_vocab * 4` is allocated.
inline constexpr uint64_t df2_bind_bytes(uint64_t head_row_bytes, int64_t n_vocab, int64_t n_dvocab, int k_drafts) {
    if (n_dvocab > 0)
        return (uint64_t) n_dvocab * head_row_bytes + (uint64_t) n_dvocab * 4ull +
               (uint64_t) k_drafts * (uint64_t) n_dvocab * 4ull;
    return n_vocab > 0 ? (uint64_t) k_drafts * (uint64_t) n_vocab * 4ull : 0ull;
}

/// Upper bound on how many proposal rows a single slot can be offered in the next verify window of `window_cap` rows
/// when `n_sel` slots are selected (`df2_window_plan().n_sel`), `n_ready` slots are co-drafting for that window, and
/// each slot is capped at `k_cap` proposals.  Every selected slot takes 1 anchor row (`n_sel` rows), and the other
/// `min(n_ready, n_sel) - 1` ready slots each take at least 1 proposal row, leaving at most
/// `window_cap - n_sel - (min(n_ready, n_sel) - 1)` proposal rows for any one slot (clamped to `[1, k_cap]`).
inline constexpr int df2_batch_slot_max_k(int window_cap, int n_sel, int n_ready, int k_cap) {
    if (k_cap < 1) return 0;
    const int a = n_sel < 1 ? 1 : n_sel;
    const int r = n_ready < a ? (n_ready < 1 ? 1 : n_ready) : a;
    const int fit = window_cap - a - (r - 1);
    const int hi = fit < 1 ? 1 : fit;
    return hi < k_cap ? hi : k_cap;
}

}  // namespace strata::core
