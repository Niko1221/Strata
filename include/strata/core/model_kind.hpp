// include/strata/core/model_kind.hpp - WHICH architecture a loaded artifact is.
//
// The engine grew around one model (Qwen4Exp) and its geometry is spelled out in `layout.hpp`.  Qwen35MoE
// (Ornith-1.5) is a different architecture with different tensors, a different residual, a different
// attention split and a different MTP block; the two must not be allowed to drift into one another through
// an unstructured bag of fields.  This enum is the single identity every loader and graph builder switches
// on, so a Qwen35 checkpoint that reaches a Qwen4Exp-only code path is a refusal, not a wrong answer.
#pragma once

namespace strata::core {

enum class ModelKind {
    Unknown = 0,   ///< not a supported architecture; refuse
    Qwen4Exp,      ///< the pack-based Qwen3.8-Flash-Next family (docs/DETAILS.md)
    Qwen35Moe,     ///< Qwen3.5/3.6 MoE, e.g. Ornith-1.5-35B-A3B (docs/ORNITH_QWEN35MOE.md)
};

inline const char* model_kind_name(ModelKind k) {
    switch (k) {
    case ModelKind::Qwen4Exp:  return "qwen4exp";
    case ModelKind::Qwen35Moe: return "qwen35moe";
    default:                   return "unknown";
    }
}

}  // namespace strata::core
