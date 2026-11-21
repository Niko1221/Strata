// include/strata/kernels/ple_lowbit.hpp - the 3-bit PLE table rows: C4, C3 and C2s (tools/ple_lowbit_emulate.py --native).
//
// A row is 160 values: 60 bytes of 3-bit codes, then the row's scales.
//   codes  8 codes in 3 bytes, a little-endian 24-bit word: value j's code is bits 3*(j%8) .. 3*(j%8)+2 of the word at
//          byte 3*(j/8)
//   C4     one fp16 scale for the row (2 B): 62 B a row
//   C3     5 E4M3 scales, one per 32 values (5 B): 65 B a row
//   C2s    10 E4M3 scales, one per 16 values (10 B): 70 B a row
// value j = s * codebook[code j], with s the fp16 scale (C4) or e4m3(block's byte) * table scale (C3, C2s), each product
// rounded to float32 in that order: exactly the floats the emulator's quantizer produces. (Its Q8_0 emulation stores those
// floats after Q8_0's own rounding, about 2.5e-5 NMSE more, so a native table is close to the emulation but not bit
// identical to it.) The 8-level codebook (`strata.ple.codebook`) and the table scale (`strata.ple.scale`) are the GGUF's
// metadata. Header-only so the CPU test (tests/ngram/ple_lowbit_test.cpp) decodes with this very code.
#pragma once

#include "strata/kernels/f16_bits.hpp"

#include <cstdint>
#include <cstring>

namespace strata::kernels {

inline constexpr int PLE_LOWBIT_CODE_BYTES = 60;   // 160 values x 3 bits

/// Bytes of one row with blocks of `block` values: 62 (160, one fp16), 65 (32) or 70 (16).
constexpr int ple_lowbit_row_bytes(int block) { return PLE_LOWBIT_CODE_BYTES + (block == 160 ? 2 : 160 / block); }

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

/// One row -> 160 floats. `block` = 160 (C4: the fp16 scale; `table_scale` unused), 32 (C3) or 16 (C2s).
inline void ple_lowbit_dequant_row(const uint8_t* row, int block, const float* codebook, float table_scale, float* out160) {
    const uint8_t* sc = row + PLE_LOWBIT_CODE_BYTES;
    float s[10];
    if (block == 160) {
        uint16_t h;
        std::memcpy(&h, sc, 2);
        s[0] = f32_from_f16(h);
    } else {
        for (int b = 0; b < 160 / block; ++b) s[b] = ple_e4m3(sc[b]) * table_scale;
    }
    for (int g = 0; g < 20; ++g) {
        const uint32_t w = (uint32_t) row[3 * g] | (uint32_t) row[3 * g + 1] << 8 | (uint32_t) row[3 * g + 2] << 16;
        for (int k = 0; k < 8; ++k) {
            const int j = 8 * g + k;
            out160[j] = s[j / block] * codebook[(w >> (3 * k)) & 7u];
        }
    }
}

}  // namespace strata::kernels
