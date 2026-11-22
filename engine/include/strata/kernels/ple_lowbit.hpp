// include/strata/kernels/ple_lowbit.hpp - the low-bit PLE table rows C1, C2, C2s, C3, C4 and C6 (tools/ple_lowbit_emulate.py --native).
//
// A row is 160 values: 20 * bits bytes of codes, then the row's scales.
//   codes  a little-endian bit stream: 8 codes in `bits` bytes read as one little-endian word, value j's code is bits
//          bits*(j%8) .. bits*(j%8)+bits-1 of the word at byte bits*(j/8) (3-bit: 8 codes in 3 bytes; 4-bit: the low
//          nibble first; 2-bit: the low 2 bits first)
//   C1     4-bit, one fp16 scale for the row (2 B): 82 B a row
//   C2     3-bit, 5 fp16 scales, one per 32 values (10 B): 70 B a row
//   C2s    3-bit, 10 E4M3 scales, one per 16 values (10 B): 70 B a row
//   C3     3-bit, 5 E4M3 scales, one per 32 values (5 B): 65 B a row
//   C4     3-bit, one fp16 scale for the row (2 B): 62 B a row
//   C6     2-bit, 5 fp16 scales, one per 32 values (10 B): 50 B a row
// value j = s * codebook[code j], with s the block's fp16 scale (C1, C2, C4, C6) or e4m3(block's byte) * table scale (C2s,
// C3), each product rounded to float32 in that order: exactly the floats the emulator's quantizer produces. (Its Q8_0
// emulation stores those floats after Q8_0's own rounding, about 2.5e-5 NMSE more, so a native table is close to the
// emulation but not bit identical to it.) The codebook of 2^bits levels (`strata.ple.codebook`; C1's is IQ4_NL's grid / 127)
// and the table scale (`strata.ple.scale`) are the GGUF's metadata. Header-only so the CPU test
// (tests/ngram/ple_lowbit_test.cpp) decodes with this very code.
#pragma once

#include "strata/kernels/f16_bits.hpp"

#include <cstdint>
#include <cstring>

namespace strata::kernels {

/// Bytes of one row: 20 * bits of codes, then 160 / block scales of 2 (fp16) or 1 (E4M3) bytes.
constexpr int ple_lowbit_row_bytes(int bits, int block, bool f16_scales) {
    return 20 * bits + (160 / block) * (f16_scales ? 2 : 1);
}

/// An E4M3 ("fn": no infinities) byte as a float, exactly; 0x7F / 0xFF (NaN) read as 0, as the FP8 table does.
inline float ple_e4m3(uint8_t x) {
    const uint32_t e = (x >> 3) & 15u, m = x & 7u;
    float f;
    if (e == 0) {
        f = (float) m * 0.001953125f;                          // subnormal: m * 2^-9
    } else if (e == 15 && m == 7) {
        f = 0.0f;
    } else {
        const uint32_t bits = ((e + 120u) << 23) | (m << 20);  // (1 + m/8) * 2^(e-7)
        std::memcpy(&f, &bits, 4);
    }
    return (x & 0x80) ? -f : f;
}

/// One row -> 160 floats. `bits` 2, 3 or 4; `block` 160, 32 or 16 values a scale; fp16 scales, or E4M3 times
/// `table_scale` (unused with fp16). `codebook` has 2^bits levels.
inline void ple_lowbit_dequant_row(const uint8_t* row, int bits, int block, bool f16_scales, const float* codebook,
                                   float table_scale, float* out160) {
    const uint8_t* sc = row + 20 * bits;
    float s[10];
    for (int b = 0; b < 160 / block; ++b) {
        if (f16_scales) {
            uint16_t h;
            std::memcpy(&h, sc + 2 * b, 2);
            s[b] = f32_from_f16(h);
        } else {
            s[b] = ple_e4m3(sc[b]) * table_scale;
        }
    }
    const uint32_t mask = (1u << bits) - 1u;
    for (int g = 0; g < 20; ++g) {
        uint32_t w = 0;
        for (int i = 0; i < bits; ++i) w |= (uint32_t) row[bits * g + i] << (8 * i);
        for (int k = 0; k < 8; ++k) {
            const int j = 8 * g + k;
            out160[j] = s[j / block] * codebook[(w >> (bits * k)) & mask];
        }
    }
}

}  // namespace strata::kernels
