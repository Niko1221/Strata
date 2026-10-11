// include/strata/program/split_rules.hpp - rules for the automatic layer split that do not need a GPU to check.
//
// A stage that owns one or two layers holds a handful of expert slots (a 12 GB 3060 with one layer held 512), and
// the prompt chunk of the whole pipeline is clamped to what the smallest cache can lend (#448): 576 -> 347 tok/s in
// #1616.  So the search first looks for a placement in which every stage owns at least kMinStageLayers layers, and
// only falls back to the unrestricted best when no such placement can start the prompt path.
#pragma once

#include <cstdint>
#include <vector>

namespace strata::program::split_rules {

constexpr int64_t kMinStageLayers = 3;

/// The restriction only makes sense when the model has the layers for it.
inline bool restriction_possible(int64_t n_layers, int stages) {
    return stages >= 2 && n_layers >= kMinStageLayers * stages;
}

/// cuts: the first layer of each later stage (stages - 1 of them); n_layers: the model's layers.
/// Returns the size of the smallest stage.
inline int64_t smallest_stage(const std::vector<int64_t>& cuts, int64_t n_layers) {
    int64_t lo = n_layers, prev = 0;
    for (const int64_t c : cuts) {
        if (c - prev < lo) lo = c - prev;
        prev = c;
    }
    if (n_layers - prev < lo) lo = n_layers - prev;
    return lo;
}

/// True when some stage owns fewer than kMinStageLayers layers.
inline bool has_tiny_stage(const std::vector<int64_t>& cuts, int64_t n_layers) {
    return smallest_stage(cuts, n_layers) < kMinStageLayers;
}

}  // namespace strata::program::split_rules
