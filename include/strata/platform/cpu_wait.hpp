#pragma once

#include <atomic>
#if defined(_WIN32)
#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#include <windows.h>
#elif defined(__x86_64__) || defined(__i386__)
#include <immintrin.h>
#endif

namespace strata::platform {

inline void cpu_pause() {
#if defined(_M_X64) || defined(_M_IX86) || defined(__x86_64__) || defined(__i386__)
    _mm_pause();
#elif defined(_WIN32)
    YieldProcessor();
#elif defined(__aarch64__)
    __asm__ volatile("yield");
#else
    std::atomic_signal_fence(std::memory_order_seq_cst);
#endif
}

inline void store_fence() {
#if defined(_M_X64) || defined(_M_IX86) || defined(__x86_64__) || defined(__i386__)
    _mm_sfence();
#else
    std::atomic_thread_fence(std::memory_order_seq_cst);
#endif
}

}  // namespace strata::platform
