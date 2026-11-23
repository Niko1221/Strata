// include/strata/core/adapt_coverage_rules.hpp - the #4870 mid-coverage adaptive-tier defaults as a pure rule (no GPU needed to test).
//
// 0.1.66 (v0.2.20) made every start whose VRAM expert cache holds 15% to 65% of the routed experts adapt twice as often and swap twice
// as many experts at a time (--adapt-every 4 -> 2, --adapt-swaps 96 -> 192, --adapt-decay 0.70 -> 0.92), on the grounds of "+8% hit
// rate, +10% decode tok/s".  Measured against 0.1.65 on an RX 7900 XT (gfx1100, Qwen3.8-Flash-Next IQ2_XS, same card, alternating
// starts, greedy, 4-5 rounds each; phaseA-tests/rel0221/RESULTS-0221-PERF.md) the hit rate does rise, but the decode speed does not:
// against the same binary with the old three values given explicitly (which is as fast as 0.1.65 in every cell): 18.8% of the experts in VRAM
// chat 512 tokens -7.8%, agent -7.8%, code -2.8% / -4.2% (5 of 5 and 4 of 4 rounds slower), 8K document +2.8%; 32.5%: 8K document -4.8%, code -3.6%,
// chat and agent +0.4% / +0.3%; 46.8%: chat -0.2%, code -1.5%, 8K document -2.0%.  Slower in 10 of 13 cells, faster in 3 (+2.8, +0.4, +0.3).
// #6791 (RX 7900 XTX, 52 -> 47 tok/s, 0.2.19 -> 0.2.20, a 19% cache with 3 MB experts) is the
// same kind of loss.  A default that costs speed on a measured card does not stay on (rules-142), so the faster curve is opt-in:
// STRATA_ADAPT_MID=1 brings it back; --adapt-every / --adapt-swaps / --adapt-decay on the command line always win over both.
#pragma once

namespace strata::core {

constexpr double kAdaptMidCoverageLow = 0.15;    ///< share of the routed experts the VRAM cache holds: lower bound
constexpr double kAdaptMidCoverageHigh = 0.65;   ///< ...and upper bound of the mid-coverage band

/// True when STRATA_ADAPT_MID=1 (the only value that turns the mid-coverage curve on).
inline bool adapt_mid_coverage_enabled(const char* env) noexcept { return env != nullptr && env[0] == '1' && env[1] == '\0'; }

/// Whether the 2 / 192 / 0.92 defaults replace the 4 / 96 / 0.70 ones for a cache holding `coverage` (0..1) of the routed experts.
inline bool adapt_mid_coverage_applies(double coverage, const char* env) noexcept {
    return adapt_mid_coverage_enabled(env) && coverage >= kAdaptMidCoverageLow && coverage <= kAdaptMidCoverageHigh;
}

}  // namespace strata::core
