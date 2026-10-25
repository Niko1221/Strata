// include/strata/core/kv_pool_policy.hpp - who gives way when the shared KV pool is full (--kv-pool-tokens).
//
// Pure arithmetic on what each lane holds, so the order is tested without a GPU (tests/core/kv_chunks_test.cpp).
// The server loop (generate.cpp, pool_reserve) fills in one KvPoolLane per batch slot and acts on the answer.
#pragma once

#include <algorithm>
#include <cstdint>
#include <vector>

namespace strata::core {

/// What the policy needs to know about one batch slot.
struct KvPoolLane {
    bool can_give = false;     ///< idle, not the slot this request reads from, and holding something
    int rank = 0;              ///< 0: holds nothing worth keeping, 1: a cached conversation, 2: a prompt read that gave way
    int64_t used = 0;          ///< when it last ended a request (smaller = longer ago)
    int64_t held_cells = 0;    ///< what giving it back frees
};

/// The slots to give back, in this order, so that `want_cells` can be reserved for a lane that already holds
/// `own_cells` while `free_cells` are free: lowest rank first, the one used longest ago first among equals, the lower
/// slot number among those. False, with `victims` empty, when all the slots that can give way together would not
/// cover it - then nothing is given back at all. True with no victims when the free cells already do.
inline bool kv_pool_eviction_plan(int64_t free_cells, int64_t own_cells, int64_t want_cells,
                                  const std::vector<KvPoolLane>& lanes, std::vector<int>& victims) {
    victims.clear();
    int64_t have = free_cells + own_cells;
    if (have >= want_cells) return true;
    std::vector<int> order;
    int64_t could = have;
    for (int b = 0; b < (int) lanes.size(); ++b)
        if (lanes[(size_t) b].can_give) {
            order.push_back(b);
            could += lanes[(size_t) b].held_cells;
        }
    if (could < want_cells) return false;
    std::stable_sort(order.begin(), order.end(), [&](int a, int b) {
        const KvPoolLane& x = lanes[(size_t) a];
        const KvPoolLane& y = lanes[(size_t) b];
        return x.rank != y.rank ? x.rank < y.rank : x.used < y.used;
    });
    for (const int b : order) {
        if (have >= want_cells) break;
        victims.push_back(b);
        have += lanes[(size_t) b].held_cells;
    }
    return true;
}

}  // namespace strata::core
