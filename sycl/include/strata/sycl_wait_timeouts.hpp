// include/strata/sycl_wait_timeouts.hpp - the SYCL port: how many bounded device waits gave up.
//
// A device wait for a flag the host raises is bounded (kSpinMax, sycl_doorbell.hpp) because an unbounded spin that never
// sees its flag wedges the card. The price is that a wait whose flag does not arrive in time simply lets the window go on,
// with whatever the host was supposed to write still missing, and nothing says so. Each wait that gives up counts itself
// here, in host-visible memory, and the verifier reads and clears the count when the window is done.
#pragma once
#include <sycl/sycl.hpp>
#include <cstdint>
#include <cstdlib>
#include "strata/sycl_doorbell.hpp"
#include "strata/wait_timeout_registry.hpp"

namespace strata {
/// The waits that gave up since the last call, per kind; clears the counts. Call after the queue has finished the window.
inline uint32_t wait_timeouts_take(uint32_t* counters, uint32_t (&per_kind)[kWaitKinds]) {
    uint32_t total = 0;
    for (int k = 0; k < kWaitKinds; ++k) {
        per_kind[k] = sys_atomic_u32(counters[k]).exchange(0u);
        total += per_kind[k];
    }
    return total;
}
}  // namespace strata
