// include/strata/kernels/qns_gather.hpp - a row subset of a qns trellis matrix (u32 words in 32-row groups, row
// scales beside them): the MTP drafter's draft head, gathered from the trellis head.
#pragma once

#include <cstdint>

#include <cuda_runtime.h>

namespace strata::kernels {

/// Row i of dst (words and scales, the same 32-row-group layout, W words a row) = row rows[i] of src, for i < n;
/// rows n .. n_pad - 1 (n_pad a multiple of 32) are zero words with scale 0.
void qns_gather_rows(const uint32_t* words, const float* scales, int W, const int32_t* rows, int n, int n_pad,
                     uint32_t* dst_words, float* dst_scales, cudaStream_t s);

}  // namespace strata::kernels
