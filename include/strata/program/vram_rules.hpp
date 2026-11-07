// include/strata/program/vram_rules.hpp - the residency bookkeeping of the `VRAM <reserve_mib>|- [device]` command
// (#533, #2522): which rows a cache's shrink takes out of the residency table, and how a grow puts them back.  Apart
// from the engine so a test can drive it with a real segmented cache and failures it chooses.
#pragma once

#include <cstdint>
#include <string>
#include <utility>
#include <vector>

#include "strata/core/expert_cache.hpp"

namespace strata::program::vram_rules {

/// (residency row, slot): what a cache's shrinks took, so a grow puts them back
using Evicted = std::vector<std::pair<int32_t, int32_t>>;

/// After a shrink: the rows of this cache (`owns(row)`) whose slot is at or past `live` (the cache's slots now) are
/// not resident any more.  Slot numbers are per cache: only this cache's rows can lose one.
template <class Owns>
inline void evict_past(std::vector<int32_t>& host_res, int64_t live, Owns&& owns, Evicted& evicted) {
    for (size_t i = 0; i < host_res.size(); ++i) {
        if (!owns((int64_t) i)) continue;
        if (host_res[i] >= live) {
            evicted.emplace_back((int32_t) i, host_res[i]);
            host_res[i] = strata::core::kNotResident;
        }
    }
}

/// After a grow: each evicted row whose slot is live again gets its expert back - or, when the adaptive tier has
/// brought that row back already, `pick_miss(row)` names the expert that takes the slot (-1: none).
/// `fill(row, slot, err)` queues the copy, `sync(err)` waits for the queue.
/// A failure keeps the table true and the debt whole: after a failed copy the queue is still waited for, the row
/// that failed and those after it stay owed and the rows refilled before it are not owed twice (the next grow gave
/// their slots to other experts while the table still named them), and after a failed wait no row whose copy was
/// in the queue is resident - each is owed again with its slot.
template <class PickMiss, class Fill, class Sync>
inline bool refill(std::vector<int32_t>& host_res, int64_t live, Evicted& evicted, PickMiss&& pick_miss, Fill&& fill,
                   Sync&& sync, std::string& err) {
    Evicted keep_out, queued;
    bool ok = true;
    for (const auto& [i, slot] : evicted) {
        if (!ok || slot >= live) { keep_out.emplace_back(i, slot); continue; }
        const int64_t pick = host_res[(size_t) i] >= 0 ? pick_miss((int64_t) i) : (int64_t) i;
        if (pick < 0) continue;
        if (!fill(pick, slot, err)) { ok = false; keep_out.emplace_back(i, slot); continue; }
        host_res[(size_t) pick] = slot;
        queued.emplace_back((int32_t) pick, slot);
    }
    std::string serr;
    if (!sync(ok ? err : serr)) {
        for (const auto& [r, slot] : queued) {
            host_res[(size_t) r] = strata::core::kNotResident;
            keep_out.emplace_back(r, slot);
        }
        ok = false;
    }
    evicted.swap(keep_out);
    return ok;
}

}  // namespace strata::program::vram_rules
