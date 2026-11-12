#pragma once

#include <cstdint>

namespace strata::core {

inline constexpr uint64_t kRemoteExpertReserveBytes = 512ull << 20;

/// Whether another expert blob fits while preserving the helper's startup/runtime reserve.
inline constexpr bool remote_expert_capacity_fits(uint64_t used_bytes, uint64_t next_bytes,
                                                   uint64_t free_bytes) noexcept {
    return used_bytes <= free_bytes &&
           kRemoteExpertReserveBytes <= free_bytes - used_bytes &&
           next_bytes <= free_bytes - used_bytes - kRemoteExpertReserveBytes;
}

}  // namespace strata::core
