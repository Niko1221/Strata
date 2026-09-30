// include/strata/program/layer_split.hpp - preparing a layer split across GPUs (docs/MULTI_GPU.md).
//
// Pure arithmetic, so src/program/layer_split_test.cpp can check it by hand; generate.cpp owns the devices.
//
//   * how much of the host expert arena to pin when several CUDA contexts map it;
//   * the prompt path's buffers, lent by every card's expert cache.  One card has always borrowed the top slots of
//     its cache for them (plan v0.3 P5: a lent slot's expert is streamed during the prompt and refilled after it).
//     A split used to give each card its own 2048-token buffers instead (~1.5 GiB a card, reserved for the whole
//     session, and 3x the chunks of one card's auto 6144): every card now lends, and all of them read the same
//     chunk, because the rows handed from one card to the next are that size.
#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>

namespace strata::program::layer_split {

/// The most of the host expert arena to pin when it is registered (0 = all of it).  Pinned, a missed expert can
/// cross PCIe by DMA (the decode's PCIe share, the prompt's streamed ring); unpinned, the CPU pool computes it or a
/// host copy stages it.  Under WDDM, pinning all of it into two contexts left the driver refusing every later
/// allocation (the 5080 + 3090 rig), so there it stays at 8 GiB; elsewhere a split pins all of it, as one GPU does
/// (a failed whole registration still falls back to slices).
inline uint64_t arena_pin_cap(bool several_contexts, bool wddm) {
    return several_contexts && wddm ? (8ull << 30) : 0;
}

/// An expert cache's slot layout: `slots` of `max_blob` bytes, or sized slots at `off` (slots + 1 offsets, the last
/// one the end), as ExpertCache::slot_offsets.
struct CacheSlots {
    int64_t slots = 0;
    int64_t bytes = 0;
    const uint64_t* off = nullptr;
    int64_t max_blob = 0;
};

/// The slots, from the end of the cache, whose bytes hold `need` (all of them when the cache is smaller).
inline int64_t slots_to_lend(const CacheSlots& c, uint64_t need) {
    if (c.off == nullptr) {
        const int64_t k = c.max_blob > 0 ? (int64_t) ((need + (uint64_t) c.max_blob - 1) / (uint64_t) c.max_blob) : c.slots;
        return k < c.slots ? k : c.slots;
    }
    int64_t k = 0;
    while (k < c.slots && (uint64_t) c.bytes - c.off[c.slots - k] < need) ++k;
    return k;
}

/// The bytes from slot `first` to the end of the cache.
inline uint64_t lent_bytes(const CacheSlots& c, int64_t first) {
    return c.off ? (uint64_t) c.bytes - c.off[first] : (uint64_t) (c.slots - first) * (uint64_t) c.max_blob;
}

/// A prompt chunk every card can lend the buffers for, and the slots each lends (one per cache, in order).
/// chunk 0 (and no slots): some card cannot lend even the smallest chunk.
struct LendPlan {
    int64_t chunk = 0;
    std::vector<int64_t> slots;
};

/// `need(i, chunk)`: the device bytes card i's prompt path needs for a chunk (Prefill::bytes_needed).  Every card
/// keeps at least 128 slots.  `auto_chunk` (--prefill auto): the largest of 8192 ... 256 whose buffers take at most
/// `lend_pct` % of every card's slots (at 8192-token chunks nearly every expert streams anyway, so a lent slot costs
/// little); otherwise `chunk`, halved until it fits.  The same rule one card has always used, over every card.
template <class NeedFn>
LendPlan plan_lend(const std::vector<CacheSlots>& caches, NeedFn need, int64_t chunk, bool auto_chunk, int64_t lend_pct) {
    LendPlan p;
    if (caches.empty()) return p;
    auto fits = [&](int64_t c, bool cap_share) -> bool {
        p.slots.assign(caches.size(), 0);
        for (size_t i = 0; i < caches.size(); ++i) {
            const int64_t k = slots_to_lend(caches[i], need(i, c));
            if (k + 128 > caches[i].slots || (cap_share && k * 100 > lend_pct * caches[i].slots)) return false;
            p.slots[i] = k;
        }
        p.chunk = c;
        return true;
    };
    if (auto_chunk) {
        static constexpr int64_t kAutoChunks[] = {8192, 6144, 4096, 3072, 2048, 1024, 512, 256};
        for (const int64_t c : kAutoChunks)
            if (fits(c, true)) return p;
    } else {
        for (int64_t c = chunk; c >= 256; c /= 2)
            if (fits(c, false)) return p;
    }
    return {};
}

}  // namespace strata::program::layer_split
