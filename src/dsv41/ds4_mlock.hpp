// DS4 environment policy; the shared engine's residency utility does not read environment variables.
#pragma once
#include "strata/platform/memory_lock.hpp"
#include <cstdlib>
#include <vector>

namespace strata::ds4 {
using EngineMemoryLock = platform::OwnedMemoryLock<platform::MemoryOwner::ds4>;
inline bool mlock_flag(const char* name) {
    const char* value = std::getenv(name);
    if (!value || !*value || std::strcmp(value, "0") == 0) return false;
    if (std::strcmp(value, "1") == 0) return true;
    std::fprintf(stderr, "ds4 mlock: %s=%s invalid (expected 0 or 1); using 0\n", name, value);
    return false;
}
struct WeightLockPolicy {
    static constexpr auto owner = platform::MemoryOwner::ds4;
    static constexpr auto category = platform::MemoryCategory::weights;
    static bool enabled() { static const bool on = mlock_flag("DS4_MLOCK_WEIGHTS"); return on; }
};
struct ContextLockPolicy {
    static constexpr auto owner = platform::MemoryOwner::ds4;
    static constexpr auto category = platform::MemoryCategory::context;
    static bool enabled() { static const bool on = mlock_flag("DS4_MLOCK_CONTEXT"); return on; }
};
template<class T> using ContextVector = std::vector<T, platform::LockingAllocator<T, ContextLockPolicy>>;
template<class T> using WeightVector = std::vector<T, platform::LockingAllocator<T, WeightLockPolicy>>;
// Copied-layer swaps transfer the evicted GPU expert through the spare copied slot. An evictable
// RAM tier is different: any CPU expert may fall back to its mapping later. Startup-only selection.
inline bool mapped_experts_need_lock(bool copy_mode, const std::vector<int>& gpu_slots,
                                     const std::vector<int>& ram_slots) {
    for (size_t e = 0; e < gpu_slots.size(); ++e)
        if (gpu_slots[e] < 0 && (!copy_mode || e >= ram_slots.size() || ram_slots[e] < 0)) return true;
    return false;
}
inline void mlock_startup() {
    static const bool printed = [] {
        platform::memory_lock_startup(WeightLockPolicy::enabled(), ContextLockPolicy::enabled(), "ds4",
                                      "DS4_MLOCK_WEIGHTS", "DS4_MLOCK_CONTEXT"); return true;
    }();
    (void) printed;
}
} // namespace strata::ds4
