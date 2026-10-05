// Portable BF16 router dot products for ARM64 hosts.
#include "strata/kernels/cpu/kq_avx1.hpp"
#include "strata/kernels/cpu/kq_avx2.hpp"

#include <cstring>

namespace strata::kernels::cpu {

static inline float bf16_to_float(uint16_t v) {
    uint32_t bits = (uint32_t) v << 16;
    float out;
    std::memcpy(&out, &bits, sizeof out);
    return out;
}

static void dot_impl(const uint16_t* w, int rows, int cols, const float* x, int nt, float* out) {
    for (int r = 0; r < rows; ++r) {
        const uint16_t* wr = w + (size_t) r * (size_t) cols;
        for (int t = 0; t < nt; ++t) {
            float sum = 0.0f;
            const float* xt = x + (size_t) t * (size_t) cols;
            for (int c = 0; c < cols; ++c) sum += bf16_to_float(wr[c]) * xt[c];
            out[(size_t) t * (size_t) rows + r] = sum;
        }
    }
}

void bf16_rows_dot_multi(const uint16_t* w, int rows, int cols, const float* x, int nt, float* out) {
    dot_impl(w, rows, cols, x, nt, out);
}

void bf16_rows_dot_multi_avx1(const uint16_t* w, int rows, int cols, const float* x, int nt, float* out) {
    dot_impl(w, rows, cols, x, nt, out);
}

}  // namespace strata::kernels::cpu
