#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>

namespace strata::program {

inline int64_t message_checkpoint_boundary(const std::vector<int64_t>& ids, int64_t resume,
        int64_t turn_at, int64_t turn_token, int64_t min_prefix = 8192, int64_t max_tail = 1024) {
    if (resume < 0 || turn_token < 0 || turn_at <= resume || turn_at >= int64_t(ids.size()) ||
        ids[std::size_t(turn_at)] != turn_token || min_prefix < 0 || max_tail <= 0) return -1;
    for (int64_t i = turn_at - 1; i > resume; --i) {
        if (turn_at - i > max_tail) return -1;
        if (ids[std::size_t(i)] == turn_token) return i >= min_prefix ? i : -1;
    }
    return -1;
}

// Selects an extra snapshot, never authorizes token or image cache reuse.
inline bool message_branch_observed(const std::vector<int64_t>& ids,
        const std::vector<int32_t>& live, int64_t message_at, int64_t turn_at,
        bool final_image_edit = false) {
    if (message_at < 0 || turn_at <= message_at || turn_at >= int64_t(ids.size()) ||
        int64_t(live.size()) <= message_at) return false;
    int64_t common = 0;
    while (common < turn_at && common < int64_t(live.size()) &&
           ids[std::size_t(common)] == live[std::size_t(common)]) ++common;
    if (common <= message_at) return false;
    return final_image_edit || (common < turn_at && common < int64_t(live.size()));
}

} // namespace strata::program
