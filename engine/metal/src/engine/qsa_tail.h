#pragma once

#include "geo.h"

#include <algorithm>
#include <cstdint>
#include <cstring>

namespace strata_metal {

// Earlier accepted keys already occupy their slots. Only the last four
// accepted rows of this window can survive in the ring.
inline void commit_qsa_tail(float* tail, const float* keys, uint32_t pos, int kept) {
    for (int row = std::max(0, kept - 4); row < kept; ++row) {
        const uint32_t slot = (pos + (uint32_t) row) & 3;
        std::memcpy(tail + slot * geo::ID, keys + row * geo::ID, sizeof(float) * geo::ID);
    }
}

}  // namespace strata_metal
