// strata/platform/hip_host_lock.hpp - HIP host registration and pinned allocation with an opt-in mlock on Linux, for both
// HIP compat layers (include/strata/hip_compat/cuda_runtime.h and include/strata/platform/hip_compat/strata_hip.h).
#pragma once
#include <hip/hip_runtime.h>
#include <hip/hip_version.h>
#include <cstddef>
#if defined(__linux__)
#include <sys/mman.h>
#include <sys/resource.h>
#include <unistd.h>
#include <cerrno>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <mutex>
#endif
// ROCm on Linux maps host memory for the GPU through a KFD userptr (HMM): hipHostRegister and, with ROCr's default
// HSA_USERPTR_FOR_PAGED_MEM, hipHostMalloc leave the pages pageable, unlike cudaHostRegister / cudaHostAlloc on NVIDIA,
// which pin them and which the callers assume.  Under memory pressure the kernel reclaims or swaps such pages, and every
// invalidation stops the GPU's queues until amdgpu restores the range: verify timeouts and "no progress" stalls (#649,
// #1705, docs/AMD_HIP.md).  With STRATA_HIP_MLOCK=1 every registration and pinned allocation is mlock'd here, once for
// all callers, and munlock'd before it is unregistered or freed; unset or 0, nothing changes.  Best effort: a refusal
// (RLIMIT_MEMLOCK) is logged once.  The ranges FileExpertSource locks itself (the unregistered part of the expert arena)
// never overlap these.  Found by the Ben1078/Strata fork (rdna3-moe-offload-perf), which locked the expert arena and
// the KV copies one by one.
namespace strata::hip_compat {
namespace host_lock_detail {
#if defined(__linux__)
inline bool on() {
    static const bool v = [] { const char* e = std::getenv("STRATA_HIP_MLOCK"); return e != nullptr && std::atoi(e) != 0; }();
    return v;
}
struct State {
    std::mutex mu;
    std::map<const void*, size_t> ranges;   // what was locked here, by the address it was registered / allocated at
    bool noted = false, warned = false;
};
inline State& st() {
    static State s;
    return s;
}
inline void lock(const void* p, size_t n) {
    if (!on() || p == nullptr || n == 0) return;
    State& s = st();
    std::lock_guard<std::mutex> g(s.mu);
    if (::mlock(p, n) == 0) {
        s.ranges[p] = n;
        if (!s.noted) {
            s.noted = true;
            rlimit rl{};
            char lim[32] = "unlimited";
            if (getrlimit(RLIMIT_MEMLOCK, &rl) == 0 && rl.rlim_cur != RLIM_INFINITY)
                std::snprintf(lim, sizeof lim, "%.0f MiB", (double) rl.rlim_cur / 1048576.0);
            std::fprintf(stderr, "strata: HIP: host memory mapped for the GPU is mlock'd as well (ROCm does not pin it; "
                                 "STRATA_HIP_MLOCK=1, memlock limit %s; /health locked_mib shows what is locked)\n", lim);
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
    // only the whole pages inside the range: a page it shares with a neighbouring range stays locked until it is freed
    const uintptr_t pg = (uintptr_t) sysconf(_SC_PAGESIZE);
    const uintptr_t a = ((uintptr_t) p + pg - 1) & ~(pg - 1), b = ((uintptr_t) p + it->second) & ~(pg - 1);
    if (b > a) (void) ::munlock((const void*) a, b - a);
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
