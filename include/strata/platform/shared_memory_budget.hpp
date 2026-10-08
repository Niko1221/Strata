#pragma once
#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>

namespace strata::platform {
inline uint64_t uma_headroom_bytes() {
    static const uint64_t head = []() -> uint64_t {
        const char* h = std::getenv("STRATA_UMA_HEADROOM_GIB");
        if (h == nullptr) return 6ull << 30;
        char* end = nullptr;
        const long v = std::strtol(h, &end, 10);
        if (end != h && *end == '\0' && v >= 0 && v <= 1024) return uint64_t(v) << 30;
        std::fprintf(stderr, "strata: STRATA_UMA_HEADROOM_GIB=%s is not a whole number of GiB (0-1024): using 6\n", h);
        return 6ull << 30;
    }();
    return head;
}
// available is a fresh OS snapshot, already bounded by cgroup limits. CUDA's
// total is an allocation ceiling, not a second memory pool to add to it.
inline uint64_t shared_allocation_budget(uint64_t available, uint64_t cuda_total, uint64_t headroom) {
    const uint64_t usable = available > headroom ? available - headroom : 0;
    return std::min(usable, cuda_total);
}
}  // namespace strata::platform
