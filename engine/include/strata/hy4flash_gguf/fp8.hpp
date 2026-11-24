#pragma once
#include <array>
#include <cmath>
#include <cstdint>
#include <limits>

namespace strata::hy4_flash {

// OCP E4M3FN, matching llama.cpp's ggml_f8_e4m3_to_fp32. A lookup avoids
// exponent arithmetic in the random-row Engram hot path; conversion is exact.
inline void dequant_f8_e4m3_row(const uint8_t* src, int n, float* dst) {
    static const std::array<float, 256> values = [] {
        std::array<float, 256> table{};
        for (int i = 0; i < 256; ++i) {
            const int magnitude = i & 127, exponent = magnitude >> 3, mantissa = magnitude & 7;
            const float value = magnitude == 127 ? std::numeric_limits<float>::quiet_NaN() :
                exponent == 0 ? std::ldexp(float(mantissa), -9) :
                                std::ldexp(1.f + float(mantissa) / 8.f, exponent - 7);
            table[i] = i & 128 ? -value : value;
        }
        return table;
    }();
    for (int i = 0; i < n; ++i) dst[i] = values[src[i]];
}

} // namespace strata::hy4_flash
