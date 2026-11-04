#pragma once

#include <sycl/sycl.hpp>
#include <cstdint>

namespace strata::kernels {
inline uint32_t iq4_lut4_swar(uint32_t q4) {
    constexpr uint32_t table[] = {
        0x81818181u, 0x98989898u, 0xADADADADu, 0xBFBFBFBFu,
        0xCFCFCFCFu, 0xDDDDDDDDu, 0xEAEAEAEAu, 0xF6F6F6F6u,
        0x01010101u, 0x0D0D0D0Du, 0x19191919u, 0x26262626u,
        0x35353535u, 0x45454545u, 0x59595959u, 0x71717171u,
    };
    uint32_t mask[4], a[8], b[4], c[2];
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        const uint32_t bit = (q4 >> i) & 0x01010101u;
        mask[i] = (bit << 8) - bit;
    }
#pragma unroll
    for (int i = 0; i < 8; ++i) a[i] = sycl::bitselect(table[2 * i], table[2 * i + 1], mask[0]);
#pragma unroll
    for (int i = 0; i < 4; ++i) b[i] = sycl::bitselect(a[2 * i], a[2 * i + 1], mask[1]);
#pragma unroll
    for (int i = 0; i < 2; ++i) c[i] = sycl::bitselect(b[2 * i], b[2 * i + 1], mask[2]);
    return sycl::bitselect(c[0], c[1], mask[3]);
}
}
