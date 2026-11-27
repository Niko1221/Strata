#pragma once

#include <cstddef>
#include <cstdint>

namespace strata_vit {

inline bool vit_q8_can_reuse_workspace(std::size_t q_elems, std::size_t score_elems,
                                      std::size_t layer_elems, std::size_t projector_elems) {
    if ((layer_elems == 0 && projector_elems == 0) || layer_elems > score_elems) return false;
    if (projector_elems == 0) return true;
    // Keep all three bases as well aligned as independent CUDA allocations.
    constexpr std::size_t alignment = 256 / sizeof(uint16_t);
    return q_elems % alignment == 0 && projector_elems <= 3 * q_elems;
}

}  // namespace strata_vit
