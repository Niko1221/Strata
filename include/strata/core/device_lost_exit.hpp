// include/strata/core/device_lost_exit.hpp - #5990: end the engine when the GPU has stopped answering.
//
// A card that leaves the PCIe bus does not fail the next call, it stops completing them, and the driver then
// serialises: a `cudaFreeHost` (or a stream synchronisation) for memory a kernel still reads waits for that kernel,
// and while it waits it holds the device context, so every other thread's CUDA call blocks behind it.  Measured on
// a V100-SXM2 with driver 580.178.04 (issue #5990): the free never returned, a 4-byte copy on an independent,
// non-blocking stream blocked, and every cudaMalloc probe blocked with it, while the device itself stayed healthy.
//
// Unwinding, and any watchdog that polls the runtime, is therefore not available once this has started: the release
// the stall watchdog (#29) runs before it ends the engine is exactly such a call, and it is where the process can
// stay alive for ever.
//
// So the exit has to come from a thread that touches NO CUDA call.  It watches a deadline, prints the line the server
// reads as an engine death, and ends the process with std::_Exit(3): no destructors, no driver call.  The window's
// own error still reaches the caller and the request; this only guarantees that the process ends.
//
// STRATA_LOST_EXIT_S: seconds between arming and the exit (default 10; 0 disables it).  This is not the stall
// watchdog (#29, STRATA_WATCHDOG_S): that one reacts to a request that stopped beating, this one to a device that
// answered an error, and it never asks the runtime anything.
#pragma once

#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <mutex>
#include <string>
#include <thread>

namespace strata::core {

struct DeviceLostExit {
    std::atomic<int64_t> at_ms{0};   ///< steady-clock ms the process must be gone by; 0 = not armed
    std::atomic<bool> started{false};
    std::mutex m;
    std::string why;                 ///< what the runtime answered, for the two lines below (first one wins)
};

inline DeviceLostExit& device_lost_exit() {
    static DeviceLostExit s;
    return s;
}

inline int64_t device_lost_now_ms() {
    using namespace std::chrono;
    return duration_cast<milliseconds>(steady_clock::now().time_since_epoch()).count();
}

/// The grace period in seconds; `seconds` >= 0 overrides the environment.
inline int device_lost_grace_s(int seconds = -1) {
    if (seconds >= 0) return seconds;
    static const int from_env = [] {
        const char* v = std::getenv("STRATA_LOST_EXIT_S");
        return v != nullptr ? std::atoi(v) : 10;
    }();
    return from_env;
}

inline void device_lost_watchdog() {
    DeviceLostExit& s = device_lost_exit();
    if (s.started.exchange(true)) return;
    std::thread([] {
        DeviceLostExit& st = device_lost_exit();
        for (;;) {
            std::this_thread::sleep_for(std::chrono::milliseconds(200));
            const int64_t at = st.at_ms.load(std::memory_order_acquire);
            if (at == 0 || device_lost_now_ms() < at) continue;
            std::string why;
            { std::lock_guard<std::mutex> g(st.m); why = st.why; }
            // The two lines the engine already uses for a fatal GPU error (#4350): the server reads the stdout one as
            // an engine death, fails the request in flight and starts the engine again.  Neither line asks the runtime.
            std::fprintf(stderr, "strata: fatal GPU error (%s): the engine ends without unwinding (#5990)\n", why.c_str());
            std::fflush(stderr);
            std::printf("ERR fatal GPU error: %s (#5990)\n", why.c_str());
            std::fflush(stdout);
            std::_Exit(3);
        }
    }).detach();
}

/// Arm the exit `seconds` from now (default: STRATA_LOST_EXIT_S).  Safe from any thread, and idempotent for the
/// message: the first reason set is kept, so the line names what the runtime actually answered.
inline void arm_device_lost_exit(const std::string& why, int seconds = -1) {
    const int s = device_lost_grace_s(seconds);
    if (s <= 0) return;
    DeviceLostExit& st = device_lost_exit();
    {
        std::lock_guard<std::mutex> g(st.m);
        if (st.why.empty()) st.why = why;
    }
    const int64_t at = device_lost_now_ms() + (int64_t) s * 1000;
    int64_t previous = st.at_ms.load(std::memory_order_relaxed);
    // #5990: a second error may shorten the grace period, but must never postpone it.  `release_gpu_waits` can run
    // again from the stall watchdog and from teardown, and a re-arming storm would otherwise keep the process alive
    // for ever - the one thing this exists to prevent (found by mercant1982, PR #6137).
    while ((previous == 0 || at < previous) &&
           !st.at_ms.compare_exchange_weak(previous, at, std::memory_order_release, std::memory_order_relaxed)) {}
    device_lost_watchdog();
}

}  // namespace strata::core
