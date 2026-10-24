// strata/core/nv_defaults.hpp - per-architecture performance defaults for NVIDIA sm_86, sm_70 and sm_60.
//
// For the sm_86/sm_60 tables, 148-nv measured every switch of the 0.1.45 bit-exact ON set, one at a time and in the stacked set, on an RTX 3060, an RTX A4000
// (sm_86) and a Tesla P100 (sm_60): a switch is in a table only when its token ids equal the default's and it was faster on that
// architecture (5+ interleaved pairs of fresh engines, noise arm, equal expert-cache slots; notes-148-nv.md).  The identity gate
// (10 prompts + exact_pp 4K/20K, fixed flags) is IDENTICAL with the whole table on.  Switches that change bits, hang or lost speed
// on a card (STRATA_EXCHANGE_ROTATE, STRATA_PCIE_BALANCE, STRATA_QSA_WARP, ...) are not in any table.
//
// The sm_70 table was measured as a combined stack on dual V100 PCIe 32 GB with IQ3_S/MTP4: ten output texts matched in the extended trial,
// and repeated 32K/two-stream checks passed. This is not the broader token-id/exact_pp gate above; single V100, 16 GB,
// SXM2 and other models remain unmeasured. See bench/results/2026-10-20-v100-sm70-defaults/README.md.
//
//   STRATA_SM70_PERF=0   turns the sm_70 table off (Volta GV100: Tesla V100)
//   STRATA_SM86_PERF=0   turns the sm_86 table off (cards of compute capability 8.6: RTX 3050-3090, RTX A2000-A6000)
//   STRATA_SM60_PERF=0   turns the sm_60 table off (Pascal GP100: Tesla P100)
//
// A switch the user set (to anything, including 0) is never overridden.  Applied once at start, before any engine code reads its
// switches; the table is used only when every visible CUDA device has the same compute capability.
#pragma once

#include <string>
#include <utility>
#include <vector>

namespace strata::core {

/// The environment settings that are the defaults of a card with compute capability major.minor (raw, as the driver reports it);
/// empty for every other card and when its off switch is 0.
std::vector<std::pair<std::string, std::string>> nv_default_env(int major, int minor);

/// Sets `nv_default_env(major, minor)` in this process's environment where the user has not set the variable, prints one line to
/// stderr when it set anything, and returns the names it set.
std::vector<std::string> apply_nv_defaults(int major, int minor);

}  // namespace strata::core

namespace strata::core {

/// dflash2-kern: the speculative runs' faster start-up adaptation of the expert tier (opt-in on sm_86 with STRATA_SPEC_ADAPT_SM86=1 since owner round 9; STRATA_SPEC_ADAPT_EVERY / _WARM, read by
/// generate.cpp): sm_86 only (z440 A4000 Q2_0; the P100 is opt-in through the two variables, notes-dflash2-kern.md).  Not part of
/// the exact table above: the residency it changes decides which experts the GPU rounds, as the adaptive tier always does.  The user's
/// own settings and the cards' off switches (STRATA_SM86_PERF=0) is kept.
void apply_spec_adapt_default(int major, int minor);

}  // namespace strata::core
