// Inkling-2-Mini hot-expert tier: which experts hold a slot in the card, decided from how often the router picks them.
//
// The rules are the Qwen engine's adaptive tier (generate.cpp adapt(), core/low_vram_rules.hpp), applied to one pool of equal slots:
//   * every routed expert adds 1.0 to its usage (the middle layers run three times per token, so each of their experts counts up to 3x);
//   * a round (the caller runs one every few tokens) looks at the experts that are not in the card and have usage >= usage_floor,
//     the hottest first, and swaps each with the coldest unpinned expert in the card as long as usage(in) > usage(out) + min_gain;
//   * at most swaps_cap swaps per round; a small card gets fewer (16 below 12.5% of the experts resident, 32 below 25%);
//   * the pinned experts (the top share of the starting order, by default 25%) never move;
//   * after a round all usage is multiplied by `decay`.
// A round only PLANS: it returns the swaps and updates its own picture of what is resident. The caller copies the bytes and flips the
// expert table later (between tokens, never inside one: the three passes of a token must see the same experts).
// Pure host code, no GPU, shared by the CUDA / HIP / SYCL backends and tested without a card.
#pragma once

#include <algorithm>
#include <cstdint>
#include <vector>

namespace strata::core {

struct InkExpertPolicyCfg {
    float decay = 0.7f;
    float min_gain = 1.5f;
    float usage_floor = 2.0f;
    int swaps_cap = 96;
    float pin_frac = 0.25f;     // share of the starting order that is pinned (used by InkExpertPolicy::pin_count)
};

// swap cap for a card that holds `resident` of `total` experts (low_vram_rules.hpp: tiny caches move little per round)
inline int ink_policy_swaps_cap(int resident, int total, int base_cap) {
    if (total <= 0) return 0;
    const double frac = (double)resident / (double)total;
    if (frac < 0.125) return std::min(base_cap, 16);
    if (frac < 0.25) return std::min(base_cap, 32);
    return base_cap;
}

class InkExpertPolicy {
public:
    struct Swap {
        int in_key;    // expert that gets the slot
        int out_key;   // expert that loses it
        int slot;      // the slot index (0 .. n_slots-1) that changes owner
    };

    // n_keys experts (layer * experts_per_layer + id); `pinned` are resident and never move; `slot_owner` = who holds each movable slot
    // at the start (-1 = empty slot).
    void init(int n_keys, const std::vector<int>& pinned, const std::vector<int>& slot_owner, const InkExpertPolicyCfg& cfg = {}) {
        cfg_ = cfg;
        n_keys_ = n_keys;
        use_.assign((size_t)n_keys, 0.0f);
        state_.assign((size_t)n_keys, kOut);
        slot_of_.assign((size_t)n_keys, -1);
        owner_ = slot_owner;
        n_pinned_ = 0;
        for (int k : pinned) {
            if (k >= 0 && k < n_keys && state_[(size_t)k] == kOut) { state_[(size_t)k] = kPinned; ++n_pinned_; }
        }
        for (int s = 0; s < (int)owner_.size(); ++s) {
            const int k = owner_[(size_t)s];
            if (k >= 0 && k < n_keys && state_[(size_t)k] == kOut) { state_[(size_t)k] = kSlot; slot_of_[(size_t)k] = s; }
            else owner_[(size_t)s] = -1;
        }
    }

    // How many of the first experts of a starting order are pinned: pin_frac of the card, at least one when there is a card.
    static int pin_count(int resident, float pin_frac) {
        if (resident <= 0) return 0;
        const int n = (int)((double)resident * (double)std::max(0.0f, std::min(0.75f, pin_frac)));
        return std::max(1, n);
    }

    bool resident(int key) const { return key >= 0 && key < n_keys_ && state_[(size_t)key] != kOut; }
    int slot_of(int key) const { return (key >= 0 && key < n_keys_) ? slot_of_[(size_t)key] : -1; }
    int n_slots() const { return (int)owner_.size(); }
    int n_resident() const {
        int n = n_pinned_;
        for (int k : owner_) n += (k >= 0) ? 1 : 0;
        return n;
    }
    float usage(int key) const { return (key >= 0 && key < n_keys_) ? use_[(size_t)key] : 0.0f; }

    void add_use(int key, float w = 1.0f) {
        if (key >= 0 && key < n_keys_) use_[(size_t)key] += w;
    }
    // Add a whole count array (e.g. the counters copied back from the card).
    void add_counts(const float* c, int n) {
        for (int i = 0; i < n && i < n_keys_; ++i) use_[(size_t)i] += c[i];
    }

    // One round: plan the swaps and decay the usage. The picture of what is resident is updated at once.
    std::vector<Swap> round() {
        std::vector<Swap> out;
        const int cap = ink_policy_swaps_cap(n_resident(), n_keys_, cfg_.swaps_cap);
        if (cap > 0 && !owner_.empty()) {
            std::vector<int> cand;
            for (int k = 0; k < n_keys_; ++k)
                if (state_[(size_t)k] == kOut && use_[(size_t)k] >= cfg_.usage_floor) cand.push_back(k);
            std::stable_sort(cand.begin(), cand.end(), [&](int a, int b) { return use_[(size_t)a] > use_[(size_t)b]; });
            // coldest movable slots first (empty slots count as usage -1: they take anything)
            std::vector<int> slots((size_t)owner_.size());
            for (int s = 0; s < (int)slots.size(); ++s) slots[(size_t)s] = s;
            auto slot_use = [&](int s) { const int k = owner_[(size_t)s]; return k < 0 ? -1.0f : use_[(size_t)k]; };
            std::stable_sort(slots.begin(), slots.end(), [&](int a, int b) { return slot_use(a) < slot_use(b); });
            size_t si = 0;
            for (int c : cand) {
                if ((int)out.size() >= cap || si >= slots.size()) break;
                const int s = slots[si];
                const int v = owner_[(size_t)s];
                if (v >= 0 && !(use_[(size_t)c] > use_[(size_t)v] + cfg_.min_gain)) break;   // sorted: nothing further can win either
                out.push_back({c, v, s});
                if (v >= 0) { state_[(size_t)v] = kOut; slot_of_[(size_t)v] = -1; }
                state_[(size_t)c] = kSlot;
                slot_of_[(size_t)c] = s;
                owner_[(size_t)s] = c;
                ++si;
            }
        }
        for (auto& u : use_) u *= cfg_.decay;
        return out;
    }

private:
    enum : uint8_t { kOut = 0, kPinned = 1, kSlot = 2 };
    InkExpertPolicyCfg cfg_;
    int n_keys_ = 0;
    int n_pinned_ = 0;
    std::vector<float> use_;
    std::vector<uint8_t> state_;
    std::vector<int> slot_of_;
    std::vector<int> owner_;
};

}  // namespace strata::core
