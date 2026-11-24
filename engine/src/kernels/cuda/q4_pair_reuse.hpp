#pragma once

#include <cstddef>
#include <cstdint>

namespace strata::kernels {

bool q4_pair_reuse_enabled(int ni, int no, int groups, int rows);

// The caller validates shapes. Positive groups_per_token selects token-major activation columns. Zero uses
// per-group stride.
void q4_pair_reuse_launch(const void* gate, std::size_t gate_stride,
                          const void* up, std::size_t up_stride,
                          const void* x, std::size_t x_stride, float* yg, float* yu,
                          int ni, int no, int groups, int rows, void* stream,
                          const int32_t* ids, int groups_per_token = 0);

} // namespace strata::kernels
