// strata-inkling: a BF16 / F16 mtp.safetensors (the original drafter checkpoint) keeps the drafter's norm weights, biases and fuse_weight
// narrow. The CUDA drafter reads those as F32 (the Q8_0 GGUF path widens them while loading), so a narrow copy made it read garbage:
// 0 percent of the drafted tokens were accepted (RTX 3090, UD-Q4_K_XL + the BF16 mtp.safetensors, 2026-11-23).
#pragma once

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <string>

namespace strata::core {

// Weight format codes as the inkling loaders use them: 101 = F32, 102 = BF16, 103 = F16.
inline bool ink_dspark_wants_f32(const std::string& name, size_t rank, int wfmt) {
    if (wfmt != 102 && wfmt != 103) return false;
    return rank <= 1 || name.find("norm") != std::string::npos || name.find("bias") != std::string::npos ||
           name.find("fuse_weight") != std::string::npos;
}

// Raw HF safetensors checkpoints (e.g. Inkling-2-Mini-HF, Inkblot-2-mini-HF) store non-matmul trunk tensors
// (norms, ShortConv kernels/static taps, L2 clip scales, rel-pos projections, router gates, loop controllers)
// in BF16, whereas canonical .strata/GGUF and the Metal/CUDA trunk kernels read them as F32.
inline bool ink_trunk_wants_f32(const std::string& name, size_t rank, int wfmt) {
    if (wfmt != 102 && wfmt != 103) return false;
    if (rank <= 1) return true;
    if (name.find("norm") != std::string::npos || name.find("bias") != std::string::npos) return true;
    if (name.find("attn_res.") != std::string::npos || name.find("middle_cycle_cross_pass") != std::string::npos) return true;
    if (name.find("l2_clip") != std::string::npos || name.find("rel_logits_proj") != std::string::npos) return true;
    if (name.find("mlp.gate.") != std::string::npos || name.find(".controllers.") != std::string::npos) return true;
    if (name.find("sconv.") != std::string::npos && name.find("weight_proj") == std::string::npos) return true;
    return false;
}

// n little-endian 16-bit values (BF16 when wfmt == 102, IEEE half when 103) widened to n floats at dst.
inline void ink_widen16_to_f32(const uint8_t* src, int wfmt, size_t n, uint8_t* dst) {
    for (size_t i = 0; i < n; ++i) {
        const uint32_t h = static_cast<uint32_t>(src[2 * i]) | (static_cast<uint32_t>(src[2 * i + 1]) << 8);
        uint32_t u;
        if (wfmt == 102) {
            u = h << 16;
        } else {
            const uint32_t sign = (h & 0x8000u) << 16;
            uint32_t exp = (h >> 10) & 0x1Fu, man = h & 0x3FFu;
            if (exp == 0) {
                if (man == 0) {
                    u = sign;
                } else {
                    exp = 127 - 15 + 1;
                    while (!(man & 0x400u)) { man <<= 1; --exp; }
                    man &= 0x3FFu;
                    u = sign | (exp << 23) | (man << 13);
                }
            } else if (exp == 31) {
                u = sign | 0x7F800000u | (man << 13);
            } else {
                u = sign | ((exp + 127 - 15) << 23) | (man << 13);
            }
        }
        std::memcpy(dst + 4 * i, &u, 4);
    }
}

}  // namespace strata::core
