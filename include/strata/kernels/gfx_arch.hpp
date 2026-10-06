// include/strata/kernels/gfx_arch.hpp - which AMD architecture names the gfx11 matrix-core code was built for.
//
// The WMMA kernels select their intrinsic with `#if defined(__gfx1100__) || ... || defined(__gfx1151__)`: a gfx11 part
// outside that list (gfx1103, gfx1152, ...) gets an empty or trapping body.  The runtime gates therefore match the SAME
// list, not the "gfx11" prefix, so such a part falls back to hipBLASLt / the portable kernels instead of launching code
// that is not there.  Pure string matching (no HIP headers), usable from host code of any build.
#pragma once

#include <cstddef>
#include <cstring>

namespace strata::kernels {

/// `gcn` is hipDeviceProp_t::gcnArchName ("gfx1151", or "gfx1151:sramecc-:xnack-"): true when it is exactly `name`.
inline bool gfx_arch_is(const char* gcn, const char* name) {
    if (gcn == nullptr) return false;
    const std::size_t n = std::strlen(name);
    return std::strncmp(gcn, name, n) == 0 && (gcn[n] == '\0' || gcn[n] == ':');
}

/// The gfx11 targets the matrix-core kernels (prompt GEMM / attention / scorer / fused experts) are compiled for.
inline bool gfx_arch_is_gfx11_wmma(const char* gcn) {
    return gfx_arch_is(gcn, "gfx1100") || gfx_arch_is(gcn, "gfx1101") || gfx_arch_is(gcn, "gfx1102") ||
           gfx_arch_is(gcn, "gfx1150") || gfx_arch_is(gcn, "gfx1151");
}

/// Strix Halo (Ryzen AI Max, RDNA3.5): the part the gfx1151 defaults (docs/STRIX_HALO.md) are measured on.
inline bool gfx_arch_is_gfx1151(const char* gcn) { return gfx_arch_is(gcn, "gfx1151"); }

}  // namespace strata::kernels
