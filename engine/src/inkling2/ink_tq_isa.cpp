// engine/src/inkling2/ink_tq_isa.cpp - the CPU probes the trellis rows (tq_cpu.cpp) ask, for the standalone strata-inkling
// build, which has no strata_kernels_cpu.  AVX2 from strata::quant's probe; the AVX-512 trellis rows also need VBMI,
// which that probe does not report, so they stay off here and the AVX2 rows run.
// MIT License - Copyright (c) 2026 Strata contributors
#include "strata/kernels/cpu/expert_layout.hpp"
#include "strata/quant/isa.hpp"

namespace strata::kernels::cpu {

bool cpu_avx2_ok() { return strata::quant::caps().avx2; }
bool cpu_avx512_ok() { return false; }

}  // namespace strata::kernels::cpu
