#pragma once
#include <atomic>
#include "strata/platform/cpu_arch.hpp"
#if STRATA_CPU_X86
#include <immintrin.h>
#endif

namespace strata::platform {
// CPU side of CUDA's mapped-host doorbell protocol. The GPU publishes with
// __threadfence_system(). ARM needs an explicit system barrier before consuming
// payloads and before notifying the GPU; volatile polling alone is insufficient.
inline void mapped_memory_acquire() noexcept {
#if defined(__aarch64__) || defined(__arm__)
    __asm__ __volatile__("dmb sy" ::: "memory");
#else
    std::atomic_thread_fence(std::memory_order_acquire);
#endif
}
inline void mapped_memory_release() noexcept {
#if defined(__aarch64__) || defined(__arm__)
    __asm__ __volatile__("dmb sy" ::: "memory");
#elif STRATA_CPU_X86
    _mm_sfence();  // Includes write-combined mapped stores.
#else
    std::atomic_thread_fence(std::memory_order_seq_cst);
#endif
}
}  // namespace strata::platform
