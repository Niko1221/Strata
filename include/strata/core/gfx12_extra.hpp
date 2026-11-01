// include/strata/core/gfx12_extra.hpp - defaults for gfx12 (RDNA4: R9700, RX 9070, RX 9060) that came out of the PR-AMD pass of 2026-11-01 and are
// kept apart from the main table (arch_defaults.cpp) so each can be dropped alone.  Same off switches as that table: STRATA_GFX12_DEFAULTS=0
// and STRATA_DEFAULTS=026 give the old behaviour, and a variable the user sets is never overridden.
//
// Exact (bit for bit the same answers):
//   STRATA_HEAD_MIX_MULTI=1  the final hyper-connection mixer of the verify window and of the draft layer as one fused_gr_read_multi
//                            instead of one gr_read per token (#3264, bsorensen110).  R9700 (p620, IQ3_S, serve, static residency,
//                            8 interleaved pairs, A/A beside it): json_extract +0.82% (6/8 pairs), agent +0.63% (6/8), code_py -0.96%
//                            (3/8; that prompt's A/A moves +-3.5% a pair), greedy ids identical in 16/16 and 15/16 request sets.
#pragma once

#include <cstdlib>
#include <string>
#include <utility>
#include <vector>

#include "strata/core/defaults_026.hpp"
#include "strata/kernels/gfx_arch.hpp"

namespace strata::core {

/// `defaults026`: STRATA_DEFAULTS=026 (defaults_026.hpp reads it once, so a test passes it).
inline std::vector<std::pair<std::string, std::string>> gfx12_exact_extra_env(const char* gcn_arch, bool defaults026) {
    std::vector<std::pair<std::string, std::string>> t;
    if (!strata::kernels::gfx_arch_is_gfx12(gcn_arch)) return t;
    const char* off = std::getenv("STRATA_GFX12_DEFAULTS");
    if (off != nullptr && off[0] == '0') return t;
    if (defaults026) return t;
    t.push_back({"STRATA_HEAD_MIX_MULTI", "1"});
    return t;
}
inline std::vector<std::pair<std::string, std::string>> gfx12_exact_extra_env(const char* gcn_arch) {
    return gfx12_exact_extra_env(gcn_arch, strata::qwen_defaults_026());
}

}  // namespace strata::core
