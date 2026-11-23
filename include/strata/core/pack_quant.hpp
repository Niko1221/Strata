// strata/core/pack_quant.hpp - one fact the Turing expert-mode default asks: is the loaded native pack an
// i-quant family (IQ2_XS / IQ3_XXS / IQ3_S)?  A header-only accessor (one program-wide instance) so the CUDA
// translation unit needs no CPU kernel library; the value is set once at pack load in program/generate.cpp.
#pragma once

namespace strata::core {

/// set once at pack load (program/generate.cpp); read by the Turing expert-mode default (exp_mode).
/// Header-only accessor with one program-wide instance, so the CUDA translation unit needs no CPU library.
inline bool& native_pack_iquant() {
    static bool v = false;
    return v;
}

}  // namespace strata::core