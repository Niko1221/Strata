// include/strata/core/arch_defaults.hpp - the per-architecture defaults of switches that are exact everywhere.
//
// Every gfx1151 (Strix Halo, RDNA3.5, unified memory) switch in the table was measured on Aurora (experiments/s23 .. s26,
// the c-best3-v4.2 config) with BYTE-IDENTICAL output against the same engine with the switch off: the same token ids at
// 4K / 32K / 64K and on four models, and a bitwise harness per kernel.  They are therefore on by default on that
// architecture and nowhere else: a CUDA build and every other HIP architecture run the code they always ran.
// Switches that change bits (STRATA_PF_GEMM, STRATA_PF_FUSED, STRATA_HC_UPMIX, STRATA_PA_FAST, STRATA_HIP_WMMA,
// STRATA_SELECT_WMMA, STRATA_HC_Q8: rounding-level, KL-gated) stay opt-in; docs/STRIX_HALO.md lists them.
//
// A switch the user set (to anything) is never overridden; STRATA_GFX1151_DEFAULTS=0 turns the whole table off.
#pragma once

#include <string>
#include <utility>
#include <vector>

namespace strata::core {

/// The environment settings that are the defaults of `gcn_arch` (hipDeviceProp_t::gcnArchName; "gfx1151" or
/// "gfx1151:sramecc-:xnack-"): empty for every other architecture.
std::vector<std::pair<std::string, std::string>> arch_default_env(const char* gcn_arch);

/// Sets `arch_default_env(gcn_arch)` in this process's environment where the user has not set the variable, prints one
/// line to stderr when it set anything, and returns the names it set.  Call once, before any engine code reads its
/// switches (they are read lazily, at first use).
std::vector<std::string> apply_arch_defaults(const char* gcn_arch);

// gfx1200 / gfx1201 (RDNA4) layer split, OPT-IN: STRATA_GFX12_SPLIT_DECODE=1 adds nine decode switches of the gfx1151
// table (all of its decode half but STRATA_Q8_PACKED and STRATA_SH_STREAM) to the gfx12 table above, when the resolved
// split runs on two or more GPUs, every one gfx12, on Linux.  Without the flag nothing changes: a gfx12 run gets the five
// prompt switches only.  These nine are bitwise on gfx1151 (above); on gfx12 only the generated output was checked, on
// 2x R9700 with one IQ3_S pack, and 11 of 12 seeded sampled runs matched, so they are speed switches there, not shown to
// be bitwise.  Numbers and the output check: docs/AMD_HIP.md.  On one R9700 they measured slower (#1478).

/// What a run does with the gfx12 split decode switches.
enum class Gfx12SplitDecode {
    off,    ///< not set; `why` says why when the flag asked for them
    hint,   ///< the split qualifies but STRATA_GFX12_SPLIT_DECODE is unset: the caller prints one hint line
    on,     ///< STRATA_GFX12_SPLIT_DECODE=1 and the split qualifies: the caller calls apply_gfx12_split_decode
};

/// Decided from the RESOLVED split, before any switch is read.  `gpu_archs` is the gcnArchName of every GPU the run
/// computes on (CUDA0, each later stage's device, any --expert-cache-remote helper; empty or one entry for a run on one
/// GPU).  `measured_os` is false where the gain was not measured (the caller passes false on Windows).  Qualifies: two or
/// more GPUs, every one gfx12, a measured OS, and STRATA_GFX12_DEFAULTS not 0.  Then STRATA_GFX12_SPLIT_DECODE=1 gives
/// `on` (exactly "1"), unset gives `hint`, any other value `off`.  An `auto` split with --split-skip-if-fits still qualifies: the
/// user asked, so the switches stay on even if the split later falls back to one GPU.  `why` (optional) gets the reason
/// for `off`.  Pure: no device calls.
Gfx12SplitDecode gfx12_split_decode(const std::vector<std::string>& gpu_archs, bool measured_os, std::string* why = nullptr);

/// The nine decode switches STRATA_GFX12_SPLIT_DECODE=1 adds, in order.
std::vector<std::pair<std::string, std::string>> gfx12_split_decode_env();

/// Sets `gfx12_split_decode_env()` where the user has not set the variable, prints one line to stderr when it set
/// anything, and returns the names it set.  Call only for `Gfx12SplitDecode::on`, before any engine code reads its
/// switches.
std::vector<std::string> apply_gfx12_split_decode();

}  // namespace strata::core
