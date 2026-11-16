// include/strata/kernels/fuse_gu.hpp - STRATA_FUSE_GU (p58/fuse-gu, opt-in): the grouped decode experts' Q6_K gate/up,
// SwiGLU and the q8_1 blocks of h in one launch (src/kernels/cuda/fuse_gu.cuh), bitwise the default kernels.
#pragma once

#include "strata/kernels/iq_kernels.hpp"

namespace strata::kernels {

/// STRATA_FUSE_GU's variant: 0 off (the default kernels), 1 the default fused variant (8 warps a block, one row's weight
/// words in registers, activations read through L1), 2..5 the bench's alternatives (fuse_gu.cuh); -1 back to the
/// environment's value.  Set before graph capture (the bench).
void native_expert_set_fuse_gu(int v);
/// The variant in effect (reads STRATA_FUSE_GU on first use).
int native_expert_fuse_gu();
/// Whether the fused kernel takes this layout (Q6_K gate/up at n_embd 2560, n_ff a multiple of 32, no NVFP4 tail, no
/// Hadamard-folded down, not a 4-bit slot).
bool native_expert_fuse_gu_takes(const NativeExpertLayout& L);

}  // namespace strata::kernels
