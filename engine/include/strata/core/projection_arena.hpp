#pragma once

#include <cstdint>
#include <limits>

namespace strata::core {

// Padding each size makes the capacity independent of tensor upload order.
struct ProjectionArenaLayout {
    std::uint64_t bytes = 0;

    bool append(std::uint64_t size, std::uint64_t& offset) {
        constexpr std::uint64_t alignment = 512;
        constexpr auto limit = (std::numeric_limits<std::uint64_t>::max)();
        if (size == 0 || size > limit - (alignment - 1)) return false;
        const std::uint64_t padded = (size + alignment - 1) & ~(alignment - 1);
        if (padded > limit - bytes) return false;
        offset = bytes;
        bytes += padded;
        return true;
    }
};

} // namespace strata::core
