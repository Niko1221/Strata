#pragma once

#include "ref_quant.h"

#include <cstddef>
#include <cstdint>
#include <cstring>

namespace strata_metal {

// Same 34 bytes per block as Q8_0. Store all aligned code lanes first,
// then the exact binary16 scales. No full interleaved temporary is needed.
inline void quantize_hc_q8_planes(const float* source, size_t elements, uint8_t* dest) {
    uint8_t* scales = dest + elements;
    for (size_t i = 0; i < elements / 32; ++i) {
        uint8_t block[34];
        ref::quantize_q8_0(source + i * 32, 32, block);
        std::memcpy(dest + i * 32, block + 2, 32);
        std::memcpy(scales + i * 2, block, 2);
    }
}

}  // namespace strata_metal
