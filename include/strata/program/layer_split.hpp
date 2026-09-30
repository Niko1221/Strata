// include/strata/program/layer_split.hpp - preparing a layer split across GPUs (docs/MULTI_GPU.md).
//
// Pure arithmetic, so src/program/layer_split_test.cpp can check it by hand; generate.cpp owns the devices.
//
//   * how much of the host expert arena to pin when several CUDA contexts map it;
//   * the prompt path's buffers, lent by every card's expert cache: as on one card, the top slots of each cache hold
//     the prompt buffers (a lent slot's expert is streamed during the prompt and refilled after it).  All cards read
//     the same chunk, because the rows handed from one card to the next are that size;
//   * where the split points go (--layer-split auto), and whether a split is predicted to pay at all.
#pragma once

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <utility>
#include <vector>

namespace strata::program::layer_split {

/// The most of the host expert arena to pin when it is registered (0 = all of it).  Pinned, a missed expert can
/// cross PCIe by DMA (the decode's PCIe share, the prompt's streamed ring); unpinned, the CPU pool computes it or a
/// host copy stages it.  Under WDDM, pinning all of it into two contexts left the driver refusing every later
/// allocation (RTX 5080 + RTX 3090), so there it stays at 8 GiB; elsewhere a split pins all of it, as one GPU does
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

// ---- where the split points go (--layer-split auto): a cost model of one decode window, fitted on an RTX 5080 + RTX 3090
// (bench/results/2026-09-29-layer-split):
//   - every layer costs its card a time inversely proportional to SMs x clock (layer_ms_estimate);
//   - a unit of routed mass no cache holds costs `miss_ms` (the CPU pool computes it);
//   - every later card adds one hand-off per window (`handoff_ms`: the rows cross pinned host memory);
//   - which experts a cache holds: its layers' profiled pairs, hottest first, until its room is used (the engine's
//     fill, which stops at the first pair that does not fit); the routed mass of profile rank r is taken as
//     (r+1)^-mass_exp.
// The model's constants are per model and per machine; Costs carries them so the engine can pass what it knows.

/// A card's time per layer and decode window, from SMs x clock: 0.33 ms on an RTX 5080 (84 SMs, 2.62 GHz) and 0.50
/// on a 3090 (the per-layer round trip and kernels, not the bytes - both cards have ~950 GB/s).
inline double layer_ms_estimate(int sms, double ghz) {
    const double speed = (double) sms * ghz;
    return 0.33 * (84.0 * 2.617) / (speed > 1.0 ? speed : 1.0);
}

/// One card of a placement: the VRAM its expert cache may take, and what a layer costs it per window.
struct Card {
    int64_t cap_bytes = 0;
    double layer_ms = 0;
};

struct Costs {
    double miss_ms = 190.0;    ///< a unit of routed mass no cache holds (STRATA_SPLIT_MISS_MS)
    double handoff_ms = 0.0;   ///< one card handing a window to the next
    double mass_exp = 1.2;     ///< the routed mass of profile rank r: (r+1)^-mass_exp
};

struct Placement {
    std::vector<int64_t> at;   ///< the first layer of every later card (empty: one card)
    double ms = 0;             ///< the predicted decode window
    double held_mass = 0;      ///< the share of the routed mass the caches hold
    int64_t held = 0;          ///< the profiled pairs they hold
};

class Planner {
public:
    /// `profile`: the ranked (layer, expert) pairs, hottest first; `slot_bytes[l]`: the VRAM one expert of layer l
    /// takes in a cache.
    Planner(int64_t n_layers, std::vector<std::pair<int32_t, int32_t>> profile, std::vector<int64_t> slot_bytes, Costs c)
        : n_layers_(n_layers), profile_(std::move(profile)), slot_bytes_(std::move(slot_bytes)), costs_(c),
          mass_(profile_.size()) {
        for (size_t r = 0; r < profile_.size(); ++r) total_mass_ += (mass_[r] = std::pow((double) r + 1.0, -c.mass_exp));
    }

    /// The window time of `cards` running from the layers `at` on (at.size() == cards.size() - 1).
    Placement predict(const std::vector<Card>& cards, const std::vector<int64_t>& at) const {
        Placement p;
        p.at = at;
        const size_t ns = cards.size();
        std::vector<int64_t> used(ns, 0);
        std::vector<bool> full(ns, false);
        double mass = 0;
        for (size_t r = 0; r < profile_.size(); ++r) {
            const int64_t l = profile_[r].first;
            size_t st = 0;
            while (st + 1 < ns && l >= at[st]) ++st;
            if (full[st]) continue;
            const int64_t b = slot_bytes_[(size_t) l];
            if (used[st] + b > cards[st].cap_bytes) { full[st] = true; continue; }
            used[st] += b;
            mass += mass_[r];
            ++p.held;
        }
        p.held_mass = total_mass_ > 0 ? mass / total_mass_ : 1.0;
        p.ms = costs_.miss_ms * (1.0 - p.held_mass) + costs_.handoff_ms * (double) (ns > 0 ? ns - 1 : 0);
        for (size_t i = 0; i < ns; ++i) {
            const int64_t lb = i == 0 ? 0 : at[i - 1], le = i + 1 < ns ? at[i] : n_layers_;
            p.ms += (double) (le - lb) * cards[i].layer_ms;
        }
        return p;
    }

    /// The fastest placement: every one for two or three cards (a later card starts at layer 2 or later), the layers
    /// in proportion to speed beyond.  Ties keep the earlier split point.
    Placement best(const std::vector<Card>& cards) const {
        const size_t ns = cards.size();
        const int64_t L = n_layers_;
        Placement best;
        best.ms = 1e300;
        std::vector<int64_t> at(ns > 0 ? ns - 1 : 0);
        auto consider = [&]() {
            Placement p = predict(cards, at);
            if (p.ms < best.ms) best = std::move(p);
        };
        if (ns <= 1) {
            consider();
        } else if (ns == 2) {
            for (int64_t k = 2; k < L; ++k) { at[0] = k; consider(); }
        } else if (ns == 3) {
            for (int64_t k1 = 2; k1 + 1 < L; ++k1)
                for (int64_t k2 = k1 + 1; k2 < L; ++k2) { at[0] = k1; at[1] = k2; consider(); }
        } else {
            double total = 0, acc = 0;
            for (const Card& c : cards) total += 1.0 / c.layer_ms;
            for (size_t i = 0; i + 1 < ns; ++i) {
                acc += 1.0 / cards[i].layer_ms;
                const int64_t lo = i == 0 ? 2 : at[i - 1] + 1, hi = L - (int64_t) (ns - 1 - i);
                at[i] = std::clamp<int64_t>((int64_t) std::llround(acc / total * (double) L), lo, hi);
            }
            consider();
        }
        return best;
    }

private:
    int64_t n_layers_;
    std::vector<std::pair<int32_t, int32_t>> profile_;
    std::vector<int64_t> slot_bytes_;
    Costs costs_;
    std::vector<double> mass_;
    double total_mass_ = 0;
};

/// Keep a split only when it is predicted faster than one card by more than `margin` (a share: 0.05 = 5%): a split
/// costs what the model leaves out (each card's own buffers, prompts crossing between cards).
inline bool split_pays(double split_ms, double one_ms, double margin) { return split_ms < one_ms * (1.0 - margin); }

}  // namespace strata::program::layer_split
