#pragma once

#include <algorithm>
#include <array>
#include <cmath>

namespace strata::spec {

// Measure real forwards at each block length: non-causal DFlash logits at K=2
// are not a truncated K=7 forward. Score the committed tokens per millisecond,
// including the draft that produced this window. K=0 maintains context but
// skips proposing. Periodic probes allow drafting to resume as the text changes.
class DFlashPolicy {
public:
    explicit DFlashPolicy(int max_k) : max_k_(std::clamp(max_k, 0, 7)) {}

    int choose() {
        static constexpr int order[] = {2, 3, 1, 0, 4, 5, 6, 7};
        for (int k : order)
            if (k <= max_k_ && n_[k] < 2) return k;
        if (rounds_ % 32 == 0) {
            const int probe = probe_++ % (max_k_ + 1);
            return probe;
        }
        int best = best_;
        for (int k = 0; k <= max_k_; ++k)
            if (tokens_[k] / ms_[k] > 1.04 * tokens_[best] / ms_[best]) best = k;
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
