// include/strata/core/arch_defaults.hpp - the per-architecture defaults of switches that are exact everywhere.
//
// Every gfx1151 (Strix Halo, RDNA3.5, unified memory) switch in the table was measured on Aurora (experiments/s23 .. s26,
// the c-best3-v4.2 config) with BYTE-IDENTICAL output against the same engine with the switch off: the same token ids at
// 4K / 32K / 64K and on four models, and a bitwise harness per kernel.  They are therefore on by default on that
// architecture and nowhere else: a CUDA build and every other HIP architecture run the code they always ran.
// Switches that change bits (STRATA_PF_GEMM, STRATA_PF_FUSED, STRATA_HC_UPMIX, STRATA_PA_FAST, STRATA_HIP_WMMA,
// STRATA_SELECT_WMMA, STRATA_HC_Q8: rounding-level, KL-gated) stay opt-in; docs/STRIX_HALO.md lists them.
//
// A switch the user set (to anything) is never overridden; STRATA_GFX1151_DEFAULTS=0 turns the whole table off.
#pragma once

#include <string>
#include <utility>
#include <vector>

namespace strata::core {

/// The environment settings that are the defaults of `gcn_arch` (hipDeviceProp_t::gcnArchName; "gfx1151" or
/// "gfx1151:sramecc-:xnack-"): empty for every other architecture.
std::vector<std::pair<std::string, std::string>> arch_default_env(const char* gcn_arch);

/// Sets `arch_default_env(gcn_arch)` in this process's environment where the user has not set the variable, prints one
/// line to stderr when it set anything, and returns the names it set.  Call once, before any engine code reads its
/// switches (they are read lazily, at first use).
std::vector<std::string> apply_arch_defaults(const char* gcn_arch);

}  // namespace strata::core
