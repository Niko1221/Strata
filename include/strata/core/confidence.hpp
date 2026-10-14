// include/strata/core/confidence.hpp - the confidence monitor's math (ActKV, arXiv:2609.31395, section 6).
//
// The signal: when a KV budget (or a lossy KV format) corrupts action-critical information, the model compensates
// with longer, lower-confidence reasoning.  ActKV turns that into an online number: per generated token the
// confidence T is the negative mean log-probability of the top-k predicted tokens, the sequence is min-pooled over
// a sliding window, the lowest b% of the pooled values are kept, and a first-order fit over them gives the trend
// slope - negative means the confidence is falling.
//
// All host-side, all on the logits rows the engine already has after a window: see Verifier::window_confidence.
// The constants are ActKV's evaluation values (k=20, w=64, s=8, b=10%); nothing here is a claim about what the
// thresholds should be for Strata - the monitor is measurement only (STRATA_CONFIDENCE=1).
#pragma once

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <vector>

namespace strata::core {

inline constexpr int kConfTopK = 20;          // ActKV's k
inline constexpr int kConfWindow = 64;        // the min-pool window w
inline constexpr int kConfStride = 8;         // the min-pool stride s
inline constexpr double kConfBottom = 0.10;   // the bottom b% of the pooled values

/// T for one logits row: -(1/k) * the sum of log P over the k largest probabilities (ActKV's token confidence).
/// `k` is clamped to the row's length.  A row of all equal logits gives T = log(vocab).
inline double token_confidence(const float* row, int64_t vocab, int k = kConfTopK) {
    if (row == nullptr || vocab <= 0) return 0.0;
    k = (int) std::max<int64_t>(1, std::min<int64_t>(k, vocab));
    // the largest k logits by a partial selection, then the softmax over the row (max-shifted, one pass)
    std::vector<int64_t> order((size_t) vocab);
    for (int64_t v = 0; v < vocab; ++v) order[(size_t) v] = v;
    std::partial_sort(order.begin(), order.begin() + k, order.end(), [&](int64_t a, int64_t b) {
        return row[a] > row[b];
    });
    double maxv = (double) row[order[0]];
    double sum = 0.0;
    for (int64_t v = 0; v < vocab; ++v) sum += std::exp((double) row[v] - maxv);
    const double lse = maxv + std::log(sum);
    double s = 0.0;
    for (int j = 0; j < k; ++j) s += (double) row[order[(size_t) j]] - lse;   // log P of each top-k token
    return -s / k;
}

/// ActKV's trace-level confidence: sliding min-pool (window w, stride s), then the bottom b% of the pooled
/// values.  Returns them in order; empty for an empty sequence.
inline std::vector<double> pooled_bottom(const std::vector<double>& t, int w = kConfWindow, int s = kConfStride,
                                         double bottom = kConfBottom) {
    std::vector<double> r;
    if (t.empty()) return r;
    w = std::max(1, w);
    s = std::max(1, s);
    for (int start = 0; start + w <= (int) t.size(); start += s) {
        double m = t[(size_t) start];
        for (int i = start + 1; i < start + w; ++i) m = std::min(m, t[(size_t) i]);
        r.push_back(m);
    }
    if (r.empty()) r.push_back(*std::min_element(t.begin(), t.end()));   // shorter than one window: its minimum
    const size_t keep = std::max<size_t>(1, (size_t) std::ceil(r.size() * bottom));
    std::vector<size_t> idx(r.size());
    for (size_t i = 0; i < r.size(); ++i) idx[i] = i;
    std::partial_sort(idx.begin(), idx.begin() + keep, idx.end(), [&](size_t a, size_t b) {
        return r[a] < r[b];
    });
    idx.resize(keep);
    std::sort(idx.begin(), idx.end());            // the bottom set, back in time order: the trend fit needs it
    std::vector<double> out;
    out.reserve(keep);
    for (size_t i : idx) out.push_back(r[i]);
    return out;
}

/// The first-order least-squares slope of the values against their position (ActKV's alpha).  0 for fewer than
/// two points.
inline double fit_slope(const std::vector<double>& y) {
    if (y.size() < 2) return 0.0;
    const double n = (double) y.size();
    double sx = 0, sy = 0, sxx = 0, sxy = 0;
    for (size_t i = 0; i < y.size(); ++i) {
        const double x = (double) i;
        sx += x;
        sy += y[i];
        sxx += x * x;
        sxy += x * y[i];
    }
    const double den = n * sxx - sx * sx;
    return den != 0.0 ? (n * sxy - sx * sy) / den : 0.0;
}

/// What one request reports: the mean token confidence and the trend slope of the pooled bottom sequence.
struct ConfidenceSummary {
    int64_t tokens = 0;
    double mean = 0.0;
    double slope = 0.0;
};

inline ConfidenceSummary confidence_summary(const std::vector<double>& t) {
    ConfidenceSummary out;
    out.tokens = (int64_t) t.size();
    if (t.empty()) return out;
    double s = 0;
    for (double x : t) s += x;
    out.mean = s / (double) t.size();
    out.slope = fit_slope(pooled_bottom(t));
    return out;
}

}  // namespace strata::core
