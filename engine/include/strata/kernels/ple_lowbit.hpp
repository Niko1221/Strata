// include/strata/kernels/ple_lowbit.hpp - the low-bit PLE table rows C1, C2, C2s, C3, C4, C5, C6, C5e, T58 and T58e
// (tools/ple_lowbit_emulate.py --native). The trellis rows (C5, C5e, T58, T58e) are at the end.
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

/// C5: the 3-bit codes of a C4 row (60 B, then one fp16 scale: 62 B) name trellis states, not levels. A register of
/// PLE_C5_STATE_BITS bits takes each code in at the bottom, state_j = ((state_{j-1} << 3) | code_j) mod 2^bits, and value
/// j = s * lut[state_j], rounded to float32 once, with `lut` the table's PLE_C5_LEVELS floats (`strata.ple.codebook`). The
/// trellis is tail-biting: the register starts with the row's own last codes (157, 158, 159; their low
/// PLE_C5_STATE_BITS - 3 bits matter), as tools/ple_c5_viterbi.cpp encodes it.
inline constexpr int PLE_C5_STATE_BITS = 12;
inline constexpr int PLE_C5_LEVELS = 1 << PLE_C5_STATE_BITS;

/// C5's 60 B of codes (bytes 0..59) through the register, times the row scale `s`.
inline void ple_c5_codes_dequant(const uint8_t* row, const float* lut, float s, float* out160) {
    uint8_t c[160];
    for (int g = 0; g < 20; ++g) {
        const uint32_t w = (uint32_t) row[3 * g] | (uint32_t) row[3 * g + 1] << 8 | (uint32_t) row[3 * g + 2] << 16;
        for (int k = 0; k < 8; ++k) c[8 * g + k] = (uint8_t) ((w >> (3 * k)) & 7u);
    }
    uint32_t st = 0;
    for (int j = 160 - (PLE_C5_STATE_BITS - 1) / 3; j < 160; ++j) st = (st << 3) | c[j];
    for (int j = 0; j < 160; ++j) {
        st = ((st << 3) | c[j]) & (uint32_t) (PLE_C5_LEVELS - 1);
        out160[j] = s * lut[st];
    }
}

inline void ple_c5_dequant_row(const uint8_t* row, const float* lut, float* out160) {
    uint16_t h;
    std::memcpy(&h, row + 60, 2);
    ple_c5_codes_dequant(row, lut, f32_from_f16(h), out160);
}

/// C5e: C5's codes, then one E4M3 byte: s = e4m3(byte 60) * table_scale (as C3's block scales), 61 B a row.
inline constexpr int PLE_C5E_ROW_BYTES = 61;
inline void ple_c5e_dequant_row(const uint8_t* row, const float* lut, float table_scale, float* out160) {
    ple_c5_codes_dequant(row, lut, ple_e4m3(row[60]) * table_scale, out160);
}

/// T58 (2.8 bits a value): the row's first PLE_T58_CODE_BITS / 8 = 56 bytes are one circular bit stream, most significant
/// bit first (stream bit t is bit 7 - t % 8 of byte t / 8). Value j's state is the PLE_C5_STATE_BITS stream bits ending at
/// e_j = (j + 1) * 448 / 160 (so value j shifts 2 3 3 3 3 2 3 ... new bits in, five values a 14-bit step), wrapping to
/// the stream's end for the first values (tail-biting: value 0's state is bits 438..447, 0, 1), and value j =
/// s * lut[state] with C5's 4096-value lut, rounded to float32 once. T58: s is the fp16 at bytes 56..57 (58 B a row);
/// T58e: s = e4m3(byte 56) * table_scale (57 B). tools/ple_t_viterbi.cpp searches the stream.
inline constexpr int PLE_T58_CODE_BITS = 448;
inline constexpr int PLE_T58_ROW_BYTES = PLE_T58_CODE_BITS / 8 + 2;
inline constexpr int PLE_T58E_ROW_BYTES = PLE_T58_CODE_BITS / 8 + 1;

inline void ple_t58_codes_dequant(const uint8_t* row, const float* lut, float s, float* out160) {
    constexpr int nb = PLE_T58_CODE_BITS / 8;
    // the stream read in order: acc holds it up to bit 8 * pos (newest bit lowest), after its last 16 bits (the wrap), so
    // state j = (acc >> (8 * pos - e_j)) & 0xFFF; a value shifts in at most 3 bits, so at most one byte is read per value
    uint64_t acc = (uint64_t) row[nb - 2] << 8 | row[nb - 1];
    int pos = 0;
    for (int j = 0; j < 160; ++j) {
        const int e = (j + 1) * PLE_T58_CODE_BITS / 160;
        if (8 * pos < e) acc = acc << 8 | row[pos++];
        out160[j] = s * lut[(acc >> (8 * pos - e)) & (uint64_t) (PLE_C5_LEVELS - 1)];
    }
}

inline void ple_t58_dequant_row(const uint8_t* row, const float* lut, float* out160) {
    uint16_t h;
    std::memcpy(&h, row + PLE_T58_CODE_BITS / 8, 2);
    ple_t58_codes_dequant(row, lut, f32_from_f16(h), out160);
}

inline void ple_t58e_dequant_row(const uint8_t* row, const float* lut, float table_scale, float* out160) {
    ple_t58_codes_dequant(row, lut, ple_e4m3(row[PLE_T58_CODE_BITS / 8]) * table_scale, out160);
}

}  // namespace strata::kernels
