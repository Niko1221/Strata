// strata/core/bw_dense_defaults.hpp - defaults of the bitwise load-ordering variants from the Blackwell Qwen speed work (BW-DENSE).
//
// Each is bitwise equal to the kernel it replaces (the screens and the engine A/Bs are in phaseA-tests/blackwell/decisions/E15.md and
// E21-small-kernel-loads.md), measured on an RTX PRO 5000 Blackwell (compute capability 12.0), so they are on there and nowhere else
// (unmeasured on other cards).  STRATA_SM120_PERF=0 and STRATA_DEFAULTS=026 turn them off like the rest of the 12.0 set; an explicit variable
// (STRATA_GR_V3_PF, STRATA_SMALL_UNR) wins on any card in both directions.
#pragma once

#include "strata/core/defaults_026.hpp"
#include "strata/core/sm120_series.hpp"

namespace strata {

/// STRATA_GR_V3_PF when unset (gr_down_v3 / gr_up_v3 load-ordering variants): bits 0 (gr_down batched staging loads) and 2 (gr_up batched loads).
inline int gr_v3_pf_default(int major, int minor) { return sm120_default(major, minor) && !qwen_defaults_026() ? 5 : 0; }

/// STRATA_SMALL_UNR when unset: bit 0 the bf16 router read, bit 1 the GDN a/b projection (constant-trip loops).
inline int small_unr_default(int major, int minor) { return sm120_default(major, minor) && !qwen_defaults_026() ? 3 : 0; }

}  // namespace strata
