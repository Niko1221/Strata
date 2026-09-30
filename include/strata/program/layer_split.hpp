// include/strata/program/layer_split.hpp - preparing a layer split across GPUs (docs/MULTI_GPU.md).
//
// Pure arithmetic, so src/program/layer_split_test.cpp can check it by hand; generate.cpp owns the devices.
//
//   * how much of the host expert arena to pin when several CUDA contexts map it.
#pragma once

#include <cstdint>

namespace strata::program::layer_split {

/// The most of the host expert arena to pin when it is registered (0 = all of it).  Pinned, a missed expert can
/// cross PCIe by DMA (the decode's PCIe share, the prompt's streamed ring); unpinned, the CPU pool computes it or a
/// host copy stages it.  Under WDDM, pinning all of it into two contexts left the driver refusing every later
/// allocation (the 5080 + 3090 rig), so there it stays at 8 GiB; elsewhere a split pins all of it, as one GPU does
/// (a failed whole registration still falls back to slices).
inline uint64_t arena_pin_cap(bool several_contexts, bool wddm) {
    return several_contexts && wddm ? (8ull << 30) : 0;
}

}  // namespace strata::program::layer_split
