#pragma once
// Adaptive proposal depth for batched DFlash2 drafting (opt-in through STRATA_BATCH_DF2_DEPTH).
// The slot's acceptance EMA gates its next maximum depth; selector probabilities can trim a low-confidence tail.
#include <cmath>
#include <cstdlib>
#include <cstring>

namespace strata::core {

struct BatchDraftDepthConfig {
    int max_depth = 1;        ///< proposals per slot at most; 1 = off
    float floor = 0.5f;       ///< minimum expected acceptance / probability product for deeper proposals
    float alpha = 0.25f;      ///< EMA weight of the newest window
    float ema_init = 0.5f;    ///< initial acceptance estimate for a new slot
    bool on() const { return max_depth > 1; }
};

/// Parse STRATA_BATCH_DF2_DEPTH (clamped to [1, max_cap]) and its probability floor. "auto" takes the ceiling (max_cap). Unset/empty/invalid depth is off.
inline BatchDraftDepthConfig batch_draft_depth_config(const char* depth_env, const char* floor_env, int max_cap) {
    BatchDraftDepthConfig c;
    if (depth_env != nullptr && depth_env[0] != '\0') {
        const int n = std::strcmp(depth_env, "auto") == 0 ? max_cap : std::atoi(depth_env);
        c.max_depth = n < 1 ? 1 : (n > max_cap ? (max_cap < 1 ? 1 : max_cap) : n);
    }
    if (floor_env != nullptr && floor_env[0] != '\0') {
        const float f = static_cast<float>(std::atof(floor_env));
        if (f > 0.0f && f <= 1.0f) c.floor = f;
    }
    return c;
}

/// Update the acceptance estimate after a window that offered the slot's first proposal.
inline float batch_draft_depth_ema(float ema, bool accepted, float alpha) {
    return (1.0f - alpha) * ema + alpha * (accepted ? 1.0f : 0.0f);
}

/// Deepest d in [1, max_depth] with ema^d >= floor; always returns at least one proposal.
inline int batch_draft_depth_for(float ema, const BatchDraftDepthConfig& c) {
    int d = 1;
    float e = ema;
    while (d < c.max_depth) {
        e *= ema;
        if (!(e >= c.floor)) break;
        ++d;
    }
    return d;
}

/// Keep proposal 1, then extend while the cumulative draft-probability product remains above floor.
/// A null probability array means all n proposals are retained.
inline int batch_draft_depth_trim(const float* probs, int n, float floor) {
    if (n <= 0) return 0;
    if (probs == nullptr) return n;
    int k = 1;
    float cumulative = probs[0];
    while (k < n) {
        cumulative *= probs[k];
        if (!(cumulative >= floor)) break;
        ++k;
    }
    return k;
}

/// Fit per-slot proposal counts into a row cap by dropping the lowest-expected-value tail first.
/// Each selected slot keeps its anchor and, when possible, at least its first proposal.
inline int batch_draft_depth_fit(int* k, const float* ema, int n_sel, int cap) {
    int rows = 0;
    for (int a = 0; a < n_sel; ++a) rows += 1 + k[a];
    while (rows > cap) {
        int worst = -1;
        float worst_value = 0.0f;
        for (int a = 0; a < n_sel; ++a) {
            if (k[a] <= 1) continue;
            const float value = std::pow(ema[a], static_cast<float>(k[a]));
            if (worst < 0 || value <= worst_value) {
                worst = a;
                worst_value = value;
            }
        }
        if (worst < 0) break;
        --k[worst];
        --rows;
    }
    return rows;
}

}  // namespace strata::core
