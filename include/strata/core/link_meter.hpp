// include/strata/core/link_meter.hpp - which AMD cards count their own PCIe link traffic in the engine.
//
// An AMD card on Linux is read through sysfs / ROCm: load, VRAM, temperature, power and the link's GENERATION and
// WIDTH are there, but the driver has no byte counter for the link (`pcie_bandwidth_acc` is N/A in rocm-smi 7.2.0 on
// an RX 7900 GRE, measured 2026-10-31), so nothing outside the engine can say how many bytes crossed it.  The engine
// can: it is the thing doing the copying, so it adds up the expert blobs it staged host -> device for a prompt and
// the ones its decode read back over the link, and the Monitor graphs that instead of a driver counter (docs/AMD_HIP.md).
//
// It is on for gfx1100 only - the RX 7900 XTX / XT / GRE and the W7900 / W7800 - because that is the card it was
// measured on, and because the figure is only worth having on a discrete card: on an integrated Radeon (gfx1151
// Strix Halo, gfx1150, gfx1103) those blobs never cross a link at all, they stay in the host's memory.  Every other
// part - CUDA, SYCL, and a HIP build on another gfx - gets nothing: the engine's output is the same as the last
// release.  STRATA_LINK_CHIPS=all turns the count on for any AMD card, for an owner who wants the number knowing the
// caveat; STRATA_LINK_STATS=0 turns the per-second line off again.
//
// The rule is a pure string test (no device, no environment), like the other arch gates in this tree
// (include/strata/kernels/gfx_arch.hpp), so tests/core/link_meter_test.cpp can check it without a GPU.
#pragma once

#include "strata/kernels/gfx_arch.hpp"

#include <cstring>

namespace strata::core {

/// The architecture whose run counts its link bytes.  `gcn` is hipDeviceProp_t::gcnArchName ("gfx1100", or
/// "gfx1100:sramecc-:xnack-"); "" / null (a CUDA build has none) is never counted.  `chips` is STRATA_LINK_CHIPS:
/// "all" counts any AMD card, anything else (or null) keeps the gfx1100 rule.
inline bool link_meter_arch(const char* gcn, const char* chips) {
    if (gcn == nullptr || gcn[0] == '\0')
        return false;
    if (chips != nullptr && std::strcmp(chips, "all") == 0)
        return std::strncmp(gcn, "gfx", 3) == 0;
    return strata::kernels::gfx_arch_is(gcn, "gfx1100");
}

}  // namespace strata::core
