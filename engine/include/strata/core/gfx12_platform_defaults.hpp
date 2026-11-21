// include/strata/core/gfx12_platform_defaults.hpp - which platforms take the two approved gfx12 prompt defaults.
//
// 148-amd measured and the owner approved STRATA_HIP_WMMA (matrix-core prompt attention) and the native Q2_0 pack's fused
// int8 prompt experts (STRATA_PF_FUSED) as defaults on gfx12.  Both were measured on Linux first, and the Windows HIP
// build kept them off "until tested there".  That test is done for the prompt attention: the engine series A.17-A.22
// (Windows, gfx1201, 17.11.2026) ran the matrix-core prompt path green - identical parity checksums, no hang, and it is
// the path the KV default q4u32 rides (A.21: 2.5x the decode kernel it replaces).  So WMMA is on by default on Windows
// gfx12 too; STRATA_HIP_WMMA=0 still turns it off.  STRATA_PF_FUSED and STRATA_MMQ_KQ were not measured on Windows and
// keep the Linux-only default; they have their own constant so the two switches cannot drift together.
#pragma once

namespace strata::core {

constexpr bool gfx12_prompt_defaults_on(bool windows) noexcept { return !windows; }

//: the matrix-core prompt attention, measured on both platforms (A.17-A.22 on Windows gfx1201)
constexpr bool gfx12_wmma_default_on(bool /*windows*/) noexcept { return true; }

#if defined(_WIN32)
inline constexpr bool kGfx12PromptDefaultsOn = gfx12_prompt_defaults_on(true);
inline constexpr bool kGfx12WmmaDefaultOn = gfx12_wmma_default_on(true);
#else
inline constexpr bool kGfx12PromptDefaultsOn = gfx12_prompt_defaults_on(false);
inline constexpr bool kGfx12WmmaDefaultOn = gfx12_wmma_default_on(false);
#endif

}  // namespace strata::core
