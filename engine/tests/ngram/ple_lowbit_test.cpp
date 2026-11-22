// tests/ngram/ple_lowbit_test.cpp - the low-bit PLE rows (C4, C3, C2s, C1, C2, C6) as the engine decodes them
// (strata/kernels/ple_lowbit.hpp),
// against the floats the emulator's quantizer produced for the same rows, bit for bit. No model, no GPU.
//
// The fixture (ple_lowbit_fixture.inc) is written by the emulator's own --native code path:
//     python terminal/tools/test_ple_lowbit_emulate.py --write-fixture
#include "strata/kernels/ple_lowbit.hpp"

#include "ple_lowbit_fixture.inc"

#include <cstdio>
#include <cstring>

namespace k = strata::kernels;

int main() {
    int bad = 0;
    for (const ple_lowbit_fixture::Case& c : ple_lowbit_fixture::cases) {
        const int rb = k::ple_lowbit_row_bytes(c.bits, c.block, c.f16_scales);
        if (c.row_bytes != rb) {
            std::printf("%s: %d B rows in the fixture, %d in the engine\n", c.name, c.row_bytes, rb);
            ++bad;
            continue;
        }
        float codebook[16], table_scale;
        std::memcpy(codebook, c.codebook, sizeof codebook);
        std::memcpy(&table_scale, &c.table_scale, 4);
        int differ = 0;
        for (int r = 0; r < c.rows; ++r) {
            float out[160];
            k::ple_lowbit_dequant_row(c.packed + (size_t) r * (size_t) c.row_bytes, c.bits, c.block, c.f16_scales, codebook,
                                      table_scale, out);
            for (int j = 0; j < 160; ++j) {
                uint32_t got;
                std::memcpy(&got, &out[j], 4);
                if (got != c.want[r * 160 + j] && differ++ < 4)
                    std::printf("%s: row %d value %d is 0x%08x, the emulator's is 0x%08x\n", c.name, r, j, got, c.want[r * 160 + j]);
            }
        }
        std::printf("%s: %d rows of %d B, %d of %d values differ\n", c.name, c.rows, c.row_bytes, differ, c.rows * 160);
        bad += differ;
    }
    // every E4M3 byte: 0, the subnormals m * 2^-9, the normals up to 448, the sign (also of 0), NaN read as 0
    for (int x = 0; x < 256; ++x) {
        const int e = (x >> 3) & 15, m = x & 7;
        float want = e == 0 ? (float) m / 512.0f : (e == 15 && m == 7) ? 0.0f : (1.0f + (float) m / 8.0f) * (float) (1 << e) / 128.0f;
        if (x & 0x80) want = -want;
        const float got = k::ple_e4m3((uint8_t) x);
        if (std::memcmp(&got, &want, 4) != 0) {
            std::printf("e4m3 0x%02x reads as %g, not %g\n", x, (double) got, (double) want);
            ++bad;
        }
    }
    std::printf("ple_lowbit_test: %s\n", bad ? "FAILED" : "OK");
    return bad ? 1 : 0;
}
