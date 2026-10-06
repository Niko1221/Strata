// src/core/kv_chunks.cpp - see include/strata/core/kv_chunks.hpp.
#include "strata/core/kv_chunks.hpp"

#include <algorithm>

namespace strata::core {

void KvChunkMap::init(int64_t n_chunks) {
    n_chunks_ = std::max<int64_t>(n_chunks, 0);
    version_ = 0;
    lanes_.clear();
    free_.clear();
    for (int64_t c = n_chunks_ - 1; c >= 0; --c) free_.push_back((int32_t) c);   // the lowest chunk is handed out first
}

KvChunkMap::Lane KvChunkMap::add_lane(int64_t logical_chunks) {
    auto l = std::make_unique<LaneData>();
    l->table.assign((size_t) std::max<int64_t>(logical_chunks, 0), trash());
    lanes_.push_back(std::move(l));
    return (Lane) lanes_.size() - 1;
}

bool KvChunkMap::reserve(Lane lane, int64_t need, bool* changed) {
    if (changed != nullptr) *changed = false;
    LaneData& l = *lanes_[(size_t) lane];
    need = std::min<int64_t>(std::max<int64_t>(need, 0), (int64_t) l.table.size());
    if (need <= l.held) return true;
    if (need - l.held > (int64_t) free_.size()) return false;
    for (; l.held < need; ++l.held) {
        l.table[(size_t) l.held] = free_.back();
        free_.pop_back();
    }
    ++version_;
    if (changed != nullptr) *changed = true;
    return true;
}

bool KvChunkMap::shrink(Lane lane, int64_t keep) {
    LaneData& l = *lanes_[(size_t) lane];
    keep = std::max<int64_t>(keep, 0);
    if (l.held <= keep) return false;
    for (int64_t c = l.held - 1; c >= keep; --c) {   // the highest first: the lowest ends up on top of the stack
        free_.push_back(l.table[(size_t) c]);
        l.table[(size_t) c] = trash();
    }
    l.held = keep;
    ++version_;
    return true;
}

bool KvChunkMap::swap(Lane a, Lane b) {
    if (a == b) return false;
    LaneData& la = *lanes_[(size_t) a];
    LaneData& lb = *lanes_[(size_t) b];
    if (la.table.size() != lb.table.size()) return false;
    std::swap_ranges(la.table.begin(), la.table.end(), lb.table.begin());
    std::swap(la.held, lb.held);
    ++version_;
    return true;
}

}  // namespace strata::core
