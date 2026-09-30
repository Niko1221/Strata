// include/strata/program/layer_split.hpp - the layer split's arithmetic (docs/MULTI_GPU.md), apart from the devices,
// so src/program/layer_split_test.cpp can check it.
#pragma once

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <utility>
#include <vector>

namespace strata::program::layer_split {

/// The most of the host expert arena to pin (0 = all of it).  Under WDDM, pinning all of it into several contexts
/// left the driver refusing later allocations, so there it stays at 8 GiB.
inline uint64_t arena_pin_cap(bool several_contexts, bool wddm) {
    return several_contexts && wddm ? (8ull << 30) : 0;
}

// --layer-split auto: the predicted time of one decode window, fitted on an RTX 5080 + RTX 3090
// (bench/results/2026-09-29-layer-split).  Each card's layers cost layer_ms each, every later card a hand-off, and
// the routed mass no cache holds miss_ms per unit, profile rank r weighing (r+1)^-mass_exp.  A cache takes its layers'
// pairs hottest first, up to the first that does not fit (as the engine fills it), in the room its session leaves.

/// A card's time per layer and window, from SMs x clock: 0.33 ms on an RTX 5080 (84 SMs, 2.62 GHz).
inline double layer_ms_estimate(int sms, double ghz) {
    const double speed = (double) sms * ghz;
    return 0.33 * (84.0 * 2.617) / (speed > 1.0 ? speed : 1.0);
}

/// One card of a placement: the VRAM its expert cache and its session may take, and what a layer costs it per window.
struct Card {
    int64_t cap_bytes = 0;
    double layer_ms = 0;
};

struct Costs {
    double miss_ms = 190.0;    ///< a unit of routed mass no cache holds (STRATA_SPLIT_MISS_MS)
    double handoff_ms = 1.0;   ///< one card handing a window to the next (~1.1 ms on 2 x RTX 2080 Ti, PCIe gen3)
    double mass_exp = 1.2;     ///< the routed mass of profile rank r: (r+1)^-mass_exp
};

struct Placement {
    std::vector<int64_t> at;   ///< the first layer of every later card (empty: one card)
    double ms = 0;             ///< the predicted decode window
    double held_mass = 0;      ///< the share of the routed mass the caches hold
    int64_t held = 0;          ///< the profiled pairs they hold
};

/// The VRAM a card running layers [lb, le) needs for their session (the carve), which its expert cache cannot use.
using RangeBytes = std::function<int64_t(int64_t lb, int64_t le)>;

class Planner {
public:
    /// `profile`: the ranked (layer, expert) pairs, hottest first; `slot_bytes[l]`: the VRAM one expert of layer l
    /// takes in a cache; `range_bytes`: what a card's layer range takes out of its `cap_bytes` (empty: nothing).
    Planner(int64_t n_layers, std::vector<std::pair<int32_t, int32_t>> profile, std::vector<int64_t> slot_bytes, Costs c,
            RangeBytes range_bytes = {})
        : n_layers_(n_layers), profile_(std::move(profile)), slot_bytes_(std::move(slot_bytes)), costs_(c),
          range_bytes_(std::move(range_bytes)), mass_(profile_.size()) {
        for (size_t r = 0; r < profile_.size(); ++r) total_mass_ += (mass_[r] = std::pow((double) r + 1.0, -c.mass_exp));
    }

    /// The window time of `cards` running from the layers `at` on (at.size() == cards.size() - 1).
    Placement predict(const std::vector<Card>& cards, const std::vector<int64_t>& at) const {
        Placement p;
        p.at = at;
        const size_t ns = cards.size();
        auto lb = [&](size_t i) { return i == 0 ? 0 : at[i - 1]; };
        auto le = [&](size_t i) { return i + 1 < ns ? at[i] : n_layers_; };
        std::vector<int64_t> room(ns), used(ns, 0);
        for (size_t i = 0; i < ns; ++i) room[i] = cards[i].cap_bytes - (range_bytes_ ? range_bytes_(lb(i), le(i)) : 0);
        std::vector<bool> full(ns, false);
        double mass = 0;
        for (size_t r = 0; r < profile_.size(); ++r) {
            const int64_t l = profile_[r].first;
            size_t st = 0;
            while (st + 1 < ns && l >= at[st]) ++st;
            if (full[st]) continue;
            const int64_t b = slot_bytes_[(size_t) l];
            if (used[st] + b > room[st]) { full[st] = true; continue; }
            used[st] += b;
            mass += mass_[r];
            ++p.held;
        }
        p.held_mass = total_mass_ > 0 ? mass / total_mass_ : 1.0;
        p.ms = costs_.miss_ms * (1.0 - p.held_mass) + costs_.handoff_ms * (double) (ns > 0 ? ns - 1 : 0);
        for (size_t i = 0; i < ns; ++i) p.ms += (double) (le(i) - lb(i)) * cards[i].layer_ms;
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
    RangeBytes range_bytes_;
    std::vector<double> mass_;
    double total_mass_ = 0;
};

}  // namespace strata::program::layer_split
