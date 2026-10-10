// The joint_matrix tile the grouped XMX GEMMs use (STRATA_PF_XMX: sycl/src/prefill/xmx_moe.dp.cpp and
// iq_xmx_grouped in sycl/src/kernels/cuda/iq_kernels.dp.cpp), per device.
//
//   8:  8x8x16, sub-group 8 - Xe-HPG (DG2, Arc A-series). DG2 has no 16-column B/accumulator tiles.
//   16: 16x16x16, sub-group 16 - Xe2 (Battlemage, Arc B-series), which has no sub-group 8.
//
// A JIT build carries both and picks per device: 8 where the device offers sub-group 8, else 16. An AOT build
// (STRATA_SYCL_AOT, an Xe2 target) carries only the 16, since ocloc refuses the 8 there at link time;
// sycl/CMakeLists.txt sets STRATA_XMX_TILE16 for that. STRATA_XMX_TILE=8|16 forces one (debug).
#pragma once
#include <sycl/sycl.hpp>
#include <algorithm>
#include <cstdlib>
#include <vector>

#if !defined(STRATA_XMX_TILE8) && !defined(STRATA_XMX_TILE16)
#define STRATA_XMX_TILE8 1
#define STRATA_XMX_TILE16 1
#endif

namespace strata {
inline int xmx_tile_for(const sycl::device& d) {
    if (const char* e = std::getenv("STRATA_XMX_TILE")) {
        const int v = std::atoi(e);
        if (v == 8 || v == 16) return v;
    }
    const std::vector<size_t> sgs = d.get_info<sycl::info::device::sub_group_sizes>();
    const bool sg8 = std::find(sgs.begin(), sgs.end(), (size_t) 8) != sgs.end();
#if defined(STRATA_XMX_TILE8) && defined(STRATA_XMX_TILE16)
    return sg8 ? 8 : 16;
#elif defined(STRATA_XMX_TILE8)
    (void) sg8;
    return 8;
#else
    (void) sg8;
    return 16;
#endif
}
}  // namespace strata
