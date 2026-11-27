// engine/src/inkling2/ink_host_avx512.h - Runtime-dispatched AVX-512 host dot kernels for InkHybridEngine (the CPU lane of the fast mode)
// Compiled in a separate TU (ink_host_avx512.cpp) with -mavx512f -mavx512bw -mavx512vl -mavx512dq -mavx2 -mfma -mf16c (MSVC /arch:AVX512)
// and ONLY called after strata::quant::caps().avx512 (F, DQ, BW, VL and the OS's ZMM state) is verified at runtime, so a CPU without
// AVX-512 (Haswell .. Zen 3, Alder Lake) never executes one of these instructions. No AVX-512 VNNI is used: Skylake-X runs it too.
// Same signatures and the same format codes as host_avx2 (ink_host_avx2.h); AVX2 and then plain C++ stay the fallbacks.
// MIT License - Copyright (c) 2026 Strata contributors
#pragma once

#include <cstdint>

namespace strata::ink::host_avx512 {

// True if this binary includes the AVX-512 host TU AND the running CPU supports AVX-512 F/DQ/BW/VL (STRATA_FORCE_ISA=avx2 turns it off).
bool available() noexcept;

float dot_sf4_row(const uint8_t* p, const uint8_t* sc, float w_outer, const float* x, int cols,
                  const float* e4_lut) noexcept;
float dot_mx8_row(const uint8_t* w, const uint8_t* sc, const float* x, int cols,
                  const float* e4_lut, const float* u8_lut) noexcept;
float dot_sk_sub64(const uint8_t* pl, int fmt, int g, const int8_t* qs, float x_d, float x_s,
                   float eff, const int8_t (*s15_lut)[5]) noexcept;

}  // namespace strata::ink::host_avx512
