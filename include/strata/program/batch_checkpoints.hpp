#pragma once

#include "strata/core/conversation_cache.hpp"
#include "strata/program/conv_cache.hpp"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <vector>

namespace strata::program {

// A text checkpoint is usable only over cells that were copied into the destination.
// Keep the complete snapshot: GDN, PLE, QSA tails/indexer state and every later stage.
// Stage identity is positional; dev_tag identifies cached state, not a device/layer range.
inline bool batch_checkpoint_prefix(const core::ConversationCheckpoint& c, const std::vector<int32_t>& held,
                                    size_t restored, size_t stages) {
    if (c.ids.empty() || !c.imgs.empty() || c.ids.size() > restored || c.ids.size() > held.size() ||
        c.stage_parts.size() != stages || !std::equal(c.ids.begin(), c.ids.end(), held.begin())) return false;
    for (const auto& part : c.stage_parts)
        if (part.ids != c.ids || !part.imgs.empty()) return false;
    return true;
}

// Selection contains only references: reject invalid paths and apply the existing
// cache budget before copying state. The caller must finish copies before moving
// or erasing any selected source entry. An optional borrowed point stays in its slot.
inline std::vector<const core::ConversationCheckpoint*> select_batch_checkpoints(
    const std::vector<core::ConversationCheckpoint>& source, const std::vector<int32_t>& held,
    size_t restored, size_t stages, size_t capacity, bool spread = false, int64_t tail = -1,
    const core::ConversationCheckpoint* borrowed = nullptr) {
    std::vector<const core::ConversationCheckpoint*> valid;
    if (capacity == 0) return valid;
    for (const auto& c : source)
        if (batch_checkpoint_prefix(c, held, restored, stages)) valid.push_back(&c);
    if (borrowed != nullptr && batch_checkpoint_prefix(*borrowed, held, restored, stages)) valid.push_back(borrowed);
    std::stable_sort(valid.begin(), valid.end(), [](const auto* a, const auto* b) {
        return a->ids.size() < b->ids.size();
    });
    valid.erase(std::unique(valid.begin(), valid.end(), [](const auto* a, const auto* b) {
        return a->ids.size() == b->ids.size();
    }), valid.end());
    // Preserve the existing one-checkpoint slot behavior. With more room, use the
    // main cache's root/pin/LRU (or spread) policy only when the chain exceeds it.
    if (capacity == 1 && valid.size() > 1) return {valid.back()};
    while (valid.size() > capacity) {
        std::vector<int64_t> lengths;
        std::vector<uint64_t> stamps;
        std::unique_ptr<bool[]> tails(new bool[valid.size()]), pins(new bool[valid.size()]);
        for (size_t i = 0; i < valid.size(); ++i) {
            lengths.push_back((int64_t) valid[i]->ids.size());
            stamps.push_back(valid[i]->used);
            tails[i] = tail >= 0 && lengths.back() == tail;
            pins[i] = valid[i]->pinned;
        }
        const size_t victim = spread
            ? conv_cache::spread_victim(lengths.data(), stamps.data(), valid.size(), (int64_t) capacity,
                                        tails.get(), pins.get())
            : conv_cache::eviction_victim(stamps.data(), valid.size(), (int64_t) capacity, tails.get(), pins.get());
        valid.erase(valid.begin() + (std::ptrdiff_t) victim);
    }
    return valid;
}

}  // namespace strata::program
