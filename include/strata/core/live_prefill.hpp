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

// A donor can never belong to this loan, including a borrower retained earlier in the same pass. Prefer a
// duplicate still backed by the GPU; otherwise use the coldest measured RAM expert of this same-sized layer.
template <class Resident>
inline int32_t live_prefill_donor(const int32_t* slots, const float* heat, int64_t experts, int32_t first,
                                  Resident resident) {
    int32_t donor = -1;
    for (int32_t e = 0; e < experts; ++e) {
        if (slots[e] >= first || !resident(e) || (slots[e] < 0 && !heat)) continue;
        const bool gpu = slots[e] >= 0;
        const bool old_gpu = donor >= 0 && slots[donor] >= 0;
        if (donor < 0 || (gpu && !old_gpu) || (gpu == old_gpu && heat && heat[e] < heat[donor])) donor = e;
    }
    return donor;
}

} // namespace strata::core
