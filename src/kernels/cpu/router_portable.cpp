// Routing lookahead on ARM: BF16 weights, FP32 inputs, no x86 intrinsics.
#include "strata/kernels/cpu/kq_avx2.hpp"
#include "strata/kernels/cpu/kq_avx1.hpp"
#include <cstring>

namespace strata::kernels::cpu {
void bf16_rows_dot_multi(const uint16_t* w, int rows, int cols, const float* x, int nt, float* out) {
    for (int t = 0; t < nt; ++t) {
        for (int r = 0; r < rows; ++r) {
            float sum = 0.0f;
            for (int c = 0; c < cols; ++c) {
                const uint32_t bits = uint32_t(w[size_t(r) * cols + c]) << 16;
                float weight;
                std::memcpy(&weight, &bits, sizeof weight);
                sum += weight * x[size_t(t) * cols + c];
            }
            out[size_t(t) * rows + r] = sum;
        }
    }
}
void bf16_rows_dot(const uint16_t* w, int rows, int cols, const float* x, float* out) {
    bf16_rows_dot_multi(w, rows, cols, x, 1, out);
}
// Kept for callers compiled with runtime dispatch; ARM never selects AVX1.
void bf16_rows_dot_multi_avx1(const uint16_t* w, int rows, int cols, const float* x, int nt, float* out) {
    bf16_rows_dot_multi(w, rows, cols, x, nt, out);
}
}  // namespace strata::kernels::cpu
