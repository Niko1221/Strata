#pragma once

#include <string_view>

namespace strata::kernels::detail {

constexpr int kGrVariantPlain = 1;
constexpr int kGrVariantSplit = 2;
constexpr int kGrVariantStaged = 3;
constexpr int kGrVariantP4 = 7;

// STRATA_HC_SPLIT is the sole fused-GR selector. Unset selects P4 on gfx12 and staged elsewhere.
// Explicit 0/1/2 select legacy paths; 7 selects P4 only on gfx12. Unknown values fail safe to staged.
constexpr int fused_gr_env_variant(std::string_view value, bool gfx12) noexcept {
    if (value == "0") return kGrVariantPlain;
    if (value == "1") return kGrVariantSplit;
    if (value == "2") return kGrVariantStaged;
    if (value == "7") return gfx12 ? kGrVariantP4 : kGrVariantStaged;
    if (value.empty() && gfx12) return kGrVariantP4;
    return kGrVariantStaged;
}

}  // namespace strata::kernels::detail
