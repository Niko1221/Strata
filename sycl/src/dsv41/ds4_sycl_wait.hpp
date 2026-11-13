#pragma once
#include "strata/sycl_doorbell.hpp"
#include <chrono>
#include <thread>

// Finish the queued router and resident expert work before waiting for the CPU
// result. CUDA's volatile GPU spin is not a coherent host-memory load on Xe.
// The CPU expert worker still overlaps the queued GPU work; no orphaned kernel
// can spin on the device if the worker fails. The caller receives a real error.
inline void ds4_sycl_wait_host(const unsigned* flag, unsigned sequence, sycl::queue& queue) {
    queue.wait_and_throw();
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(60);
    while (*reinterpret_cast<volatile const unsigned*>(flag) != sequence) {
        if (std::chrono::steady_clock::now() >= deadline)
            throw sycl::exception(sycl::make_error_code(sycl::errc::runtime), "DeepSeek CPU expert mailbox timed out");
        std::this_thread::yield();
    }
    std::atomic_thread_fence(std::memory_order_acquire);
}
