#pragma once

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstdlib>

namespace strata::core::glm_prefill_memory {

constexpr int kMinLand = 12;

inline int landing_slots(int64_t available, size_t stride, const char* override_slots) {
    if (override_slots) return std::max(kMinLand, std::atoi(override_slots));
    return stride ? (int) std::clamp(0.02 * (double) available / (double) stride, (double) kMinLand, 64.0)
                  : kMinLand;
}

// The allocator reports failure after clearing its error and resetting the destination pointer.
template<class Allocate>
int allocate_landing(int want, Allocate allocate) {
    for (int n = std::max(kMinLand, want);; n = std::max(kMinLand, n / 2)) {
        if (allocate(n)) return n;
        if (n == kMinLand) return 0;
    }
}

}  // namespace strata::core::glm_prefill_memory
