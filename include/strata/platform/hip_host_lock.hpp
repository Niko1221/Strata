// strata/platform/hip_host_lock.hpp - HIP host registration and pinned allocation with an mlock on Linux, for both
// HIP compat layers (include/strata/hip_compat/cuda_runtime.h and include/strata/platform/hip_compat/strata_hip.h).
#pragma once

#include <hip/hip_runtime.h>
#include <hip/hip_version.h>

#include <cerrno>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <mutex>
#if defined(__linux__)
#include <sys/mman.h>
#endif

// HIP on Linux maps host memory for the GPU through a KFD userptr (HMM): hipHostRegister and, with ROCr's default
// HSA_USERPTR_FOR_PAGED_MEM, hipHostMalloc leave the pages pageable - unlike cudaHostRegister / cudaHostAlloc on NVIDIA,
// which pin them and which the engine's callers assume ("the driver page-locks it already").  Under memory pressure the
// kernel can swap out or reclaim such pages, and every invalidation stops the GPU's queues until amdgpu restores the
// range: verify timeouts and "no progress" stalls (#649, #1705, docs/AMD_HIP.md).  So every registration and pinned
// allocation is also mlock'd here, once for all callers, and munlock'd before it is unregistered or freed.  Best effort:
// a refusal (RLIMIT_MEMLOCK) is logged once and changes nothing else.  mlock stops swap and reclaim, not page migration:
// compaction still moves locked pages unless vm.compact_unevictable_allowed=0.  On a GTT mapping (with
// HSA_USERPTR_FOR_PAGED_MEM=0) mlock is accepted and does nothing; VmLck in /proc/self/status shows what the kernel
// locked.  STRATA_HIP_MLOCK=0: as before (nothing locked here).  Found by the Ben1078/Strata fork
// (rdna3-moe-offload-perf), which locked the expert complement and the KV copies one by one.
namespace strata::hip_compat {
namespace host_lock_detail {
#if defined(__linux__)
struct State {
    std::mutex mu;
    std::map<const void*, size_t> ranges;   // what was locked here, by the address it was registered / allocated at
    uint64_t asked = 0;
    bool noted = false, warned = false;
};
inline State& st() {
    static State s;
    return s;
}
inline bool on() {
    static const bool v = [] { const char* e = std::getenv("STRATA_HIP_MLOCK"); return e == nullptr || std::atoi(e) != 0; }();
    return v;
}
// ponytail: whole pages are locked and unlocked; two registrations that share a page would unlock each other's shared
// page.  The engine's registrations are page-aligned allocations, so there is no such pair today.
inline void lock(const void* p, size_t n) {
    if (!on() || p == nullptr || n == 0) return;
    State& s = st();
    std::lock_guard<std::mutex> g(s.mu);
    if (::mlock(p, n) == 0) {
        s.ranges[p] = n;
        s.asked += n;
        if (!s.noted && s.asked >= (64ull << 20)) {
            s.noted = true;
            std::fprintf(stderr, "strata: HIP: host memory mapped for the GPU is mlock'd as well (ROCm does not pin it; "
                                 "STRATA_HIP_MLOCK=0 turns this off; VmLck in /proc/self/status shows what is locked)\n");
        }
    } else if (!s.warned) {
        s.warned = true;
        const int e = errno;
        std::fprintf(stderr, "strata: HIP: mlock of %.1f MiB of host memory mapped for the GPU was refused (%s): it stays "
                             "pageable and memory pressure can stall the GPU - raise the memlock limit (ulimit -l "
                             "unlimited, LimitMEMLOCK=infinity, docker --ulimit memlock=-1)\n",
                     (double) n / 1048576.0, std::strerror(e));
    }
}
inline void unlock(const void* p) {
    if (!on() || p == nullptr) return;
    State& s = st();
    std::lock_guard<std::mutex> g(s.mu);
    const auto it = s.ranges.find(p);
    if (it == s.ranges.end()) return;
    (void) ::munlock(p, it->second);
    s.ranges.erase(it);
}
#else
inline void lock(const void*, size_t) {}
inline void unlock(const void*) {}
#endif
}  // namespace host_lock_detail

inline hipError_t host_register(void* p, size_t bytes, unsigned flags) {
    const hipError_t e = hipHostRegister(p, bytes, flags);
    if (e == hipSuccess) host_lock_detail::lock(p, bytes);
    return e;
}
inline hipError_t host_unregister(void* p) {
    host_lock_detail::unlock(p);
    return hipHostUnregister(p);
}
template <typename T>
inline hipError_t host_alloc(T** pointer, size_t bytes, unsigned flags) {
    void** p = reinterpret_cast<void**>(pointer);
    const hipError_t e = hipHostMalloc(p, bytes, flags);
    if (e == hipSuccess) host_lock_detail::lock(*p, bytes);
    return e;
}
inline hipError_t host_free(void* p) {
    host_lock_detail::unlock(p);
#if HIP_VERSION_MAJOR < 7
    return hipHostFree(p);
#else
    return hipFreeHost(p);
#endif
}
}  // namespace strata::hip_compat
