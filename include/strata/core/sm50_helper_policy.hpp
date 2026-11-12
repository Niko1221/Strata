#pragma once

namespace strata::core {

/// sm_50 is permitted only on dedicated expert-cache helpers, never on the primary or a layer stage.
inline constexpr bool sm50_primary_or_stage_supported(int compute_capability_tenths) noexcept {
    return compute_capability_tenths >= 52;
}

}  // namespace strata::core
