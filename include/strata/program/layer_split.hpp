// include/strata/program/layer_split.hpp - the layer split's arithmetic (docs/MULTI_GPU.md), apart from the devices
// so src/program/layer_split_test.cpp can check it.
#pragma once

#include <cstdint>

namespace strata::program::layer_split {

/// The most of the host expert arena to pin (0 = all of it).  Under WDDM, pinning all of it into several contexts
/// left the driver refusing later allocations, so there it stays at 8 GiB.
inline uint64_t arena_pin_cap(bool several_contexts, bool wddm) {
    return several_contexts && wddm ? (8ull << 30) : 0;
}

}  // namespace strata::program::layer_split
