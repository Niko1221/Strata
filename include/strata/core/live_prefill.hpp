#pragma once

#include <algorithm>
#include <cstdint>

namespace strata::core {

// Sized-slot loans must fit wholly within the active prefix. The offsets themselves stay stable under VMM.
inline int64_t live_prefill_first(const uint64_t* offsets, int64_t slots, uint64_t bytes, int64_t keep = 128) {
    if (!offsets || slots <= keep || bytes == 0 || offsets[slots] - offsets[keep] < bytes) return -1;
    return std::upper_bound(offsets + keep, offsets + slots, offsets[slots] - bytes) - offsets - 1;
}

inline int64_t live_prefill_floor(const uint64_t* offsets, int64_t capacity, uint64_t bytes, int64_t keep = 128) {
    if (!offsets || capacity <= keep || bytes == 0 || offsets[capacity] - offsets[keep] < bytes) return -1;
    return std::lower_bound(offsets + keep + 1, offsets + capacity + 1, offsets[keep] + bytes) - offsets;
}

} // namespace strata::core
