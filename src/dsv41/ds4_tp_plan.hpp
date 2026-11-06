#pragma once

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>
#include <utility>
#include <vector>

namespace strata::ds4 {

struct TpBlockLayer {
    std::vector<int> blocks;
    // Resident bytes for a member with b blocks, including routed and shared matrices.
    // The final index is the largest supported block count.
    std::vector<double> bytes;
};

// capacity is the per-member budget after the caller's fixed-cost estimate;
// a common offset does not affect balancing. Return true only when a changed
// plan is committed; invalid input and unsuccessful searches leave layers intact.
inline bool rebalance_tp_blocks(std::vector<TpBlockLayer>& layers,
                                const std::vector<double>& capacity) {
    const std::size_t members = capacity.size();
    if (members == 0 || layers.empty()) return false;

    // Normalize before summing: finite double inputs can otherwise overflow even
    // when only their residual differences matter. All objectives use this one scale.
    long double scale = 0;
    bool equal_capacity = true;
    for (double value : capacity) {
        if (!std::isfinite(value)) return false;
        scale = std::max(scale, std::abs(static_cast<long double>(value)));
        equal_capacity = equal_capacity && value == capacity.front();
    }
    std::vector<int> max_blocks;
    max_blocks.reserve(layers.size());
    for (const auto& layer : layers) {
        if (layer.blocks.size() != members || layer.bytes.empty()) return false;
        int maximum = 0;
        for (int count : layer.blocks) {
            if (count < 1 || static_cast<std::size_t>(count) >= layer.bytes.size()) return false;
            maximum = std::max(maximum, count);
        }
        for (double value : layer.bytes) {
            if (!std::isfinite(value) || value < 0) return false;
            scale = std::max(scale, static_cast<long double>(value));
        }
        // Never exceed the existing maximum width, even if bytes has extra entries.
        max_blocks.push_back(maximum);
    }
    if (members == 1 || equal_capacity || scale == 0) return false;

    std::vector<std::vector<int>> plan;
    plan.reserve(layers.size());
    for (const auto& layer : layers) plan.push_back(layer.blocks);

    std::vector<long double> residual(members);
    const auto compute_residual = [&]() {
        for (std::size_t member = 0; member < members; ++member) {
            long double value = static_cast<long double>(capacity[member]) / scale;
            for (std::size_t layer = 0; layer < layers.size(); ++layer) {
                value -= static_cast<long double>(layers[layer].bytes[plan[layer][member]]) / scale;
            }
            residual[member] = value;
        }
    };
    const auto objective = [&]() {
        long double mean = 0;
        for (long double value : residual) mean += value;
        mean /= static_cast<long double>(members);
        long double variance = 0;
        for (long double value : residual) {
            const long double difference = value - mean;
            variance += difference * difference;
        }
        return variance;
    };

    compute_residual();
    const long double initial_min = *std::min_element(residual.begin(), residual.end());
    long double current = objective();
    for (;;) {
        long double mean = 0;
        for (long double value : residual) mean += value;
        mean /= static_cast<long double>(members);
        const long double tolerance = 64 * std::numeric_limits<long double>::epsilon()
                                      * std::max(1.0L, current);
        long double best_gain = tolerance;
        std::size_t best_layer = layers.size();
        std::size_t best_donor = 0;
        std::size_t best_receiver = 0;

        // Stable iteration order resolves exact ties without depending on hashing
        // or device timing. Nonlinear bytes curves can change the residual mean.
        for (std::size_t layer = 0; layer < layers.size(); ++layer) {
            const auto& bytes = layers[layer].bytes;
            const auto& blocks = plan[layer];
            for (std::size_t donor = 0; donor < members; ++donor) {
                const int from = blocks[donor];
                if (from <= 1) continue;
                const long double released = (static_cast<long double>(bytes[from])
                                              - static_cast<long double>(bytes[from - 1])) / scale;
                for (std::size_t receiver = 0; receiver < members; ++receiver) {
                    const int to = blocks[receiver];
                    if (receiver == donor || to >= max_blocks[layer]) continue;
                    const long double consumed = (static_cast<long double>(bytes[to + 1])
                                                  - static_cast<long double>(bytes[to])) / scale;
                    const long double sum_change = released - consumed;
                    // Negative change in sum((residual - mean)^2), including
                    // the mean shift when released != consumed.
                    const long double gain = -(
                        2 * (residual[donor] - mean) * released
                        - 2 * (residual[receiver] - mean) * consumed
                        + released * released + consumed * consumed
                        - sum_change * sum_change / static_cast<long double>(members));
                    if (gain > best_gain) {
                        best_gain = gain;
                        best_layer = layer;
                        best_donor = donor;
                        best_receiver = receiver;
                    }
                }
            }
        }
        if (best_layer == layers.size()) break;
        --plan[best_layer][best_donor];
        ++plan[best_layer][best_receiver];
        // Recompute rather than accumulating rounding error across moves. Strict
        // descent by at least tolerance over finitely many plans prevents cycles.
        compute_residual();
        const long double next = objective();
        if (!(current - next > tolerance)) {
            ++plan[best_layer][best_donor];
            --plan[best_layer][best_receiver];
            break;
        }
        current = next;
    }

    compute_residual();
    // Variance alone can sacrifice the tightest member on nonlinear curves.
    // Conservatively discard the entire search rather than commit such a plan.
    if (*std::min_element(residual.begin(), residual.end()) < initial_min) return false;
    bool changed = false;
    for (std::size_t layer = 0; layer < layers.size(); ++layer) {
        changed = changed || plan[layer] != layers[layer].blocks;
    }
    if (!changed) return false;
    for (std::size_t layer = 0; layer < layers.size(); ++layer) {
        layers[layer].blocks = std::move(plan[layer]);
    }
    return true;
}

}  // namespace strata::ds4
