#pragma once
#include <sycl/sycl.hpp>
#include <cstdint>

namespace strata::kernels::iq4_decode {
// Four byte lookups in parallel through an exact bit-select tree. Whether these
// operations become native BFN instructions must be checked in the emitted ISA.
inline uint32_t bit_select(uint32_t q) {
    constexpr uint32_t table[] = {
        0x81818181u, 0x98989898u, 0xadadadadu, 0xbfbfbfbfu,
        0xcfcfcfcfu, 0xddddddddu, 0xeaeaeaeau, 0xf6f6f6f6u,
        0x01010101u, 0x0d0d0d0du, 0x19191919u, 0x26262626u,
        0x35353535u, 0x45454545u, 0x59595959u, 0x71717171u};
    uint32_t mask[4];
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        const uint32_t b = (q >> i) & 0x01010101u;
        // A byte containing 0/1 maps to 0/255 without a carry to its neighbor.
        // The first form becomes an integer multiply by 255 in current IGC.
        mask[i] = (b << 8) - b;
    }
    uint32_t a[8], b[4], c[2];
#pragma unroll
    for (int i = 0; i < 8; ++i) a[i] = sycl::bitselect(table[2 * i], table[2 * i + 1], mask[0]);
#pragma unroll
    for (int i = 0; i < 4; ++i) b[i] = sycl::bitselect(a[2 * i], a[2 * i + 1], mask[1]);
#pragma unroll
    for (int i = 0; i < 2; ++i) c[i] = sycl::bitselect(b[2 * i], b[2 * i + 1], mask[2]);
    return sycl::bitselect(c[0], c[1], mask[3]);
}
} // namespace strata::kernels::iq4_decode
