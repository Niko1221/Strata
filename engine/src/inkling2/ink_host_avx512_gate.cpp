// engine/src/inkling2/ink_host_avx512_gate.cpp - the runtime gate of the AVX-512 host kernels (ink_host_avx512.cpp).
// Built with the baseline flags of the core (SSE4.2): this is the code that runs on EVERY CPU, so it must hold no AVX-512 (or AVX2) instruction.
// The kernels' own TU is compiled with -mavx512f ... and a compiler may use those instructions anywhere in it, so the gate lives here.
// INK_HOST_AVX512_BUILT is set by CMakeLists.txt on the core target exactly when ink_host_avx512.cpp is built with AVX-512 flags
// (not for STRATA_ISA_FLOOR = avx / none).
// MIT License - Copyright (c) 2026 Strata contributors
#include "ink_host_avx512.h"
#include "strata/quant/isa.hpp"

namespace strata::ink::host_avx512 {

bool available() noexcept {
#if defined(INK_HOST_AVX512_BUILT)
    static const bool ok = strata::quant::caps().avx512;   // F, DQ, BW, VL and the OS's ZMM state; STRATA_FORCE_ISA=avx2 / avx / sse closes it
    return ok;
#else
    return false;
#endif
}

}  // namespace strata::ink::host_avx512
