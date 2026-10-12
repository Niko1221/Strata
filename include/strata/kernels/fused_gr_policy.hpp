#pragma once

namespace strata::kernels::detail {

// P4 may only be promoted when the architecture and both sides of its staged-parity oracle pass.
constexpr bool fused_gr_p4_parity_gate(bool gfx12, bool staged_ok, bool p4_ok) noexcept {
    return gfx12 && staged_ok && p4_ok;
}

// Emit the q32 parity diagnostic only when the complete self-test succeeded and this path was checked.
constexpr bool fused_gr_p4_q32_parity_marker_ok(bool selftest_ok, bool qfuse_hc, bool staged_ok, bool p4_ok) noexcept {
    return selftest_ok && qfuse_hc && staged_ok && p4_ok;
}

}  // namespace strata::kernels::detail
