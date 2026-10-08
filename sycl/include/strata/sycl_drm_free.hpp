// include/strata/sycl_drm_free.hpp - the SYCL port: when the DRM fdinfo may replace the driver's free-memory figure.
// The fdinfo total adds up every card the process uses, so it stands in only for a driver that reports the whole card
// as free (an Arc A750 on i915). The figure xe reports per card already counts this process.
#pragma once
#include <cstdint>

namespace strata {
inline bool drm_total_replaces_free(uint64_t free_b, uint64_t total_b, uint64_t own) {
    return own > 0 && total_b > own && total_b - own < free_b && free_b + (16ull << 20) >= total_b;
}
}  // namespace strata
