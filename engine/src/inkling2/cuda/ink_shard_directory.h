#pragma once

#include "strata/artifact/strata_reader.hpp"
#include <array>

namespace strata::ink {

// Shard offsets are indexed by expert ID, while entries on disk are in routing order.
template <size_t Experts>
bool valid_shard_directory(const std::vector<artifact::StrataExpertEntry>& entries,
                           size_t layers, uint64_t section_bytes) {
    if (entries.size() != layers * Experts) return false;
    for (size_t layer = 0; layer < layers; ++layer) {
        std::array<bool, Experts> seen{};
        for (size_t slot = 0; slot < Experts; ++slot) {
            const auto& entry = entries[layer * Experts + slot];
            if (entry.expert_id >= Experts || seen[entry.expert_id]) return false;
            if (entry.dn_offset >= entry.body_bytes || entry.body_offset > section_bytes ||
                entry.body_bytes > section_bytes - entry.body_offset) return false;
            seen[entry.expert_id] = true;
        }
    }
    return true;
}
} // namespace strata::ink
