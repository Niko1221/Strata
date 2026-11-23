// include/strata/core/adapt_coverage_rules.hpp - the #4870 mid-coverage adaptive-tier defaults as a pure rule (no GPU needed to test).
//
// 0.1.66 (v0.2.20) made every start whose VRAM expert cache holds 15% to 65% of the routed experts adapt twice as often and swap twice
// as many experts at a time (--adapt-every 4 -> 2, --adapt-swaps 96 -> 192, --adapt-decay 0.70 -> 0.92), on the grounds of "+8% hit
// rate, +10% decode tok/s".  Measured with one engine start per arm and round, arms alternating, greedy, against the same binary with the
// old three values given explicitly (phaseA-tests/rel0221/RESULTS-0221-PERF.md):
//   CUDA, RTX A4000 (sm_86), IQ2_XS, 30% of the experts in VRAM, 3 rounds: code +5.7%, chat +6.7%, 8K document +9.0%  -> stays on.
//   HIP, RX 7900 XT (gfx1100), same pack, 18.8%: chat 512 tokens -7.8%, agent -7.8%, code -2.8% / -4.2%, 8K document +2.8%; 32.5%: 8K document -4.8%,
//   code -3.6%, chat and agent +0.4% / +0.3%; 46.8%: chat -0.2%, code -1.5%, 8K document -2.0%.  Slower in 10 of 13 cells  -> off.
// #6791 (RX 7900 XTX, 52 -> 47 tok/s, 0.2.19 -> 0.2.20, a 19% cache with 3 MB experts) is that loss.  A default that costs speed on a measured
// card does not stay on (rules-142), so: on by default for the CUDA build, off for the HIP build.  STRATA_ADAPT_MID=1 turns it on and
// STRATA_ADAPT_MID=0 off in either build; --adapt-every / --adapt-swaps / --adapt-decay on the command line always win over all of it.
#pragma once

namespace strata::core {

constexpr double kAdaptMidCoverageLow = 0.15;    ///< share of the routed experts the VRAM cache holds: lower bound
constexpr double kAdaptMidCoverageHigh = 0.65;   ///< ...and upper bound of the mid-coverage band

/// The build's default for the mid-coverage curve: on for CUDA, off for HIP (measured, see above).
constexpr bool adapt_mid_default_on() noexcept {
#if defined(STRATA_USE_HIP)
    return false;
#else
    return true;
#endif
}

/// STRATA_ADAPT_MID: "1" -> on, "0" -> off, unset or anything else -> `default_on`.
inline bool adapt_mid_coverage_enabled(const char* env, bool default_on) noexcept {
    if (env != nullptr && env[0] == '1' && env[1] == '\0') return true;
    if (env != nullptr && env[0] == '0' && env[1] == '\0') return false;
    return default_on;
}

/// Whether the 2 / 192 / 0.92 defaults replace the 4 / 96 / 0.70 ones for a cache holding `coverage` (0..1) of the routed experts.
inline bool adapt_mid_coverage_applies(double coverage, const char* env, bool default_on) noexcept {
    return adapt_mid_coverage_enabled(env, default_on) && coverage >= kAdaptMidCoverageLow && coverage <= kAdaptMidCoverageHigh;
}

}  // namespace strata::core
