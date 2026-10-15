// metal/src/engine/rope_window.h - the decode window's RoPE table (qsa.metal's `cs`, buffer 9 of qsa_prep): cos and
// sin of every position's 32 NEOX pairs (the first 64 dims), as float2 rows [pos][32].
//
// Built with the PC's builder (build_rope_table, rope.hpp: rope.cu's, kept in setup_host.cpp) under the process's
// rope scaling (rope_scaling.hpp), which generate.cpp resolves from --rope-scaling / --rope-scale / --rope-freq-base /
// --rope-freq-scale / --yarn-* and the model file's keys, and sets before the window exists.  Unscaled it is the
// window's earlier loop byte for byte (the same pow / cos / sin in the same order).  Linear and YaRN put the scaled
// angles and the magnitude factor (YaRN's mscale, or --yarn-attn-factor) into the values, so every qsa_prep (the
// decode window, the MTP draft layer, both prompt paths) rotates by them without a kernel change.
// metal/tests/engine/rope_table_parity.cpp checks both.
#pragma once

#include "strata/kernels/rope.hpp"

#include <cstddef>
#include <vector>

namespace strata::metal_engine {

inline void window_rope_table(const strata::kernels::RopeScaling& scaling, int rows, float* cs) {
    std::vector<float> c((size_t) rows * 32), s((size_t) rows * 32);
    strata::kernels::build_rope_table(64, scaling, rows, c.data(), s.data());
    for (size_t k = 0; k < (size_t) rows * 32; ++k) {
        cs[2 * k] = c[k];
        cs[2 * k + 1] = s[k];
    }
}

}  // namespace strata::metal_engine
