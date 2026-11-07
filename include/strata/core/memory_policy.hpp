// Main-engine policy. DS4 has its own configuration; neither family reads the other's switches.
#pragma once
#include "strata/platform/memory_lock.hpp"
#include <cstdlib>
#include <initializer_list>
#include <vector>

namespace strata::core {
using EngineMemoryLock = platform::OwnedMemoryLock<platform::MemoryOwner::core>;
struct MemoryLockSetting {
    bool enabled = false, specified = false;
    explicit MemoryLockSetting(const char* name) {
        const char* value = std::getenv(name);
        if (!value || !*value) return;
        if (std::strcmp(value, "0") == 0 || std::strcmp(value, "1") == 0) {
            specified = true; enabled = *value == '1'; return;
        }
        std::fprintf(stderr, "strata mlock: %s=%s invalid (expected 0 or 1); using default 0\n", name, value);
    }
};
struct EngineMemoryLockPolicy {
    MemoryLockSetting weights{"STRATA_MLOCK_WEIGHTS"}, context{"STRATA_MLOCK_CONTEXT"};
};
inline const EngineMemoryLockPolicy& engine_memory_lock_policy() {
    static const EngineMemoryLockPolicy policy;
    return policy;
}
struct WeightLockPolicy {
    static constexpr auto owner = platform::MemoryOwner::core;
    static constexpr auto category = platform::MemoryCategory::weights;
    static bool enabled() { return engine_memory_lock_policy().weights.enabled; }
};
struct ContextLockPolicy {
    static constexpr auto owner = platform::MemoryOwner::core;
    static constexpr auto category = platform::MemoryCategory::context;
    static bool enabled() { return engine_memory_lock_policy().context.enabled; }
};
// Zero and unset preserve legacy arena/PLE choices. Only enabled tracked ownership replaces them.
inline bool legacy_weight_lock_allowed() {
#if defined(__linux__)
    return !engine_memory_lock_policy().weights.enabled;
#else
    return true;
#endif
}
// Still free()able, but no context/temporary allocation can share the selected weight pages.
inline void* malloc_weight_storage(size_t bytes) {
#if defined(__linux__)
    if (WeightLockPolicy::enabled()) {
        const size_t page = platform::memory_page_size();
        if (bytes > SIZE_MAX - (page - 1)) { errno = ENOMEM; return nullptr; }
        void* p = nullptr;
        const int error = ::posix_memalign(&p, page, (bytes + page - 1) / page * page);
        if (error) { errno = error; return nullptr; }
        return p;
    }
#endif
    return std::malloc(bytes);
}
inline size_t context_allocation_bytes(size_t bytes) {
#if defined(__linux__)
    if (bytes && ContextLockPolicy::enabled()) {
        const size_t page = platform::memory_page_size();
        if (bytes > SIZE_MAX - 2 * page) return SIZE_MAX;
        return (bytes + page - 1) / page * page + page;
    }
#endif
    return bytes;
}
inline size_t memory_bytes_sum(std::initializer_list<size_t> parts) {
    size_t total = 0;
    for (size_t n : parts) {
        if (n > SIZE_MAX - total) return SIZE_MAX;
        total += n;
    }
    return total;
}
inline void memory_lock_startup() {
    static const bool once = [] {
        platform::memory_lock_startup(WeightLockPolicy::enabled(), ContextLockPolicy::enabled(), "strata",
                                      "STRATA_MLOCK_WEIGHTS", "STRATA_MLOCK_CONTEXT");
        return true;
    }();
    (void) once;
}
template<class T> using ContextAllocator = platform::LockingAllocator<T, ContextLockPolicy>;
template<class T> using WeightVector = std::vector<T, platform::LockingAllocator<T, WeightLockPolicy>>;
// Payload vectors accept ordinary vectors at the C++ API boundary without exposing their allocator.
template<class T> class ContextVector : public std::vector<T, ContextAllocator<T>> {
    using Base = std::vector<T, ContextAllocator<T>>;
public:
    using Base::Base;
    using Base::operator=;
    ContextVector() = default;
    template<class A> ContextVector(const std::vector<T, A>& v) : Base(v.begin(), v.end()) {}
    template<class A> ContextVector& operator=(const std::vector<T, A>& v) {
        this->assign(v.begin(), v.end()); return *this;
    }
};
// Sidecars for pre-existing raw CPU allocations; reserve before calling into this where possible.
// Failure to allocate optional bookkeeping must not fail model/context creation.
inline void track_memory_lock(std::vector<EngineMemoryLock>& owners, platform::MemoryCategory category,
                              const void* p, size_t bytes, bool enabled) {
    if (!enabled || !p || !bytes) return;
    try { owners.emplace_back(category, p, bytes); }
    catch (const std::bad_alloc&) { EngineMemoryLock::allocation_failed(category, bytes); }
}
} // namespace strata::core
