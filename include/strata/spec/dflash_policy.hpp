#pragma once

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>

namespace strata::spec {

// Attention kernels mask cells using the device step record. A bounded launch
// bucket can therefore serve later positions without rebuilding the draft graph.
inline int64_t dflash_attention_bucket(int64_t visible, int64_t capacity) {
    if (visible < 1 || visible > capacity) return 0;
    int64_t cells = std::min<int64_t>(64, capacity);
    while (cells < visible) {
        if (cells > capacity / 4) return capacity;
        cells *= 4;
    }
    return cells;
}

// A target verifies anchor + K proposals; DFlash2 also evaluates an anchor
// in its draft forward. Subtraction keeps untrusted large positions overflow-safe.
inline bool dflash_proposal_fits(int64_t pos, int k, bool dflash2,
                                 int64_t target_capacity, int64_t draft_capacity) {
    return pos >= 0 && k >= 1 && k <= 7 && pos <= target_capacity &&
           pos <= draft_capacity && k + 1 <= target_capacity - pos &&
           k + (dflash2 ? 1 : 0) <= draft_capacity - pos;
}

// Measure real forwards at each block length: non-causal DFlash logits at K=2
// are not a truncated K=7 forward. Score the committed tokens per millisecond,
// including the draft that produced this window. K=0 maintains context but
// skips proposing. Periodic probes allow drafting to resume as the text changes.
class DFlashPolicy {
public:
    explicit DFlashPolicy(int max_k) : max_k_(std::clamp(max_k, 0, 7)) { reset(); }

    // Costs and acceptance belong to this request; graph capture happens once
    // per width in the verifier, so retain only its warm state across requests.
    void reset() {
        n_.fill(0); tokens_.fill(0); ms_.fill(0);
        best_ = rounds_ = 0;
        probe_ = max_k_ >= 4 ? 4 : 0;
    }

    int choose() {
        // Start with small windows; wide non-causal forwards are still measured
        // by periodic probes, without charging every short reply for a full sweep.
        static constexpr int order[] = {2, 3, 1, 0};
        for (int k : order)
            if (k <= max_k_ && n_[k] < 2) return k;
        if (rounds_ % 16 == 0) {
            const int probe = probe_++ % (max_k_ + 1);
            return probe;
        }
        int best = best_;
        for (int k = 0; k <= max_k_; ++k)
            if (n_[k] > 0 && tokens_[k] / ms_[k] > 1.04 * tokens_[best] / ms_[best]) best = k;
        best_ = best;
        return best;
    }

    void observe(int k, int emitted, double ms) {
        if (k < 0 || k > max_k_ || emitted < 1 || emitted > k + 1 || !std::isfinite(ms) || ms <= 0) return;
        // The first window of each width can capture a CUDA graph. Its one-time
        // cost must not make an otherwise useful block look permanently slow.
        if (seen_[k]++ == 0) return;
        if (n_[k] == 0) { tokens_[k] = emitted; ms_[k] = ms; }
        else {
            tokens_[k] = 0.8 * tokens_[k] + 0.2 * emitted;
            ms_[k] = 0.8 * ms_[k] + 0.2 * ms;
        }
        ++n_[k];
        ++rounds_;
    }

private:
    int max_k_, best_ = 0, rounds_ = 0, probe_ = 0;
    std::array<int, 8> n_{};
    std::array<int, 8> seen_{};
    std::array<double, 8> tokens_{}, ms_{};
};

}  // namespace strata::spec
