#pragma once

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstdlib>

namespace strata::core::glm_prefill_memory {

constexpr int kMinLand = 12;

// The state arena is addressed in F32 units, including its packed FP16 latent caches.
inline int64_t latent_floats(int kv_lora, int64_t context, bool fast) {
    const int64_t elements = (int64_t) kv_lora * context;
    return fast ? (elements + 1) / 2 : elements;
}

inline int64_t latent_start(int64_t offset, bool fast) {
    return fast ? (offset + 3) & ~int64_t(3) : offset;  // 16-byte vector-load alignment
}

inline int landing_slots(int64_t available, size_t stride, const char* override_slots) {
    if (override_slots) return std::max(kMinLand, std::atoi(override_slots));
    return stride ? (int) std::clamp(0.03 * (double) available / (double) stride, (double) kMinLand, 96.0)
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
