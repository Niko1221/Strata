// include/strata/core/tc_sm75.hpp - sm_75 (RTX 20 / Turing) tensor-core paths.
//
// Turing has no tf32 MMA and no cp.async, so three of this engine's prompt-path kernels are compiled out below
// sm_80 and fall back to slower paths (see `qsa_prompt_attn.cu`, `qsa_select.cu`).  It does have fp16 MMA
// (m16n8k8; m16n8k16 is sm_80+) and cuBLAS has no bf16 GEMM kernel below sm_80 - it runs an FFMA one, which
// measured 19.5% of the prompt's GPU time on a 2x2080 Ti.  These switches select the Turing paths.
//
// **ONLY sm_75 IS EVER AFFECTED.**  The predicate below answers true for compute capability 7.5 and for nothing
// else: cc 7.0/7.2 (Volta) keeps its own behaviour, and every cc >= 8 keeps the paths the engine was written
// with.  `STRATA_TC_SM75` can only turn the Turing paths OFF (for an A/B or a bisect); it cannot turn them on
// somewhere else, so a machine that is not a 2080 Ti cannot be moved onto them by an environment variable.
// The kernels themselves are gated a second time, at compile time, by `__CUDA_ARCH__ == 750`.
#pragma once

#include <cstdlib>

namespace strata::core {

/// The Turing paths are for compute capability 7.5 exactly.  `auto` (the default): on.  "0": off (the
/// pre-patch behaviour).  Any other value, and any other device, is off.
inline bool tc_sm75_enabled(int cc_major, int cc_minor) {
    if (cc_major != 7 || cc_minor != 5) return false;   // 7.5 and only 7.5 - never another SM model
    const char* e = std::getenv("STRATA_TC_SM75");
    if (e != nullptr && e[0] == '0') return false;
    return true;
}

}  // namespace strata::core
