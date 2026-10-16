// sycl/include/strata/kernels/fused_gr_q8.hpp - STRATA_HC_Q8=1 on SYCL: the layout the Q8_0 hyper-connection read
// takes (fused_gr.dp.cpp), shared by the loader (core/native_dense.cpp) and gr_q8_parity.
#pragma once

#include <cstdint>

namespace strata::kernels {

/// The Q8_0 read keeps a projection's bytes split: every row's int8 quants row-major ([rows][K] bytes, 32 a block),
/// then every row's fp16 block scales ([rows][K / 32] halves). The same bytes as ggml's Q8_0 blocks (34 a block: the
/// scale, then 32 quants), reordered so a lane reads its row's quants as 16-byte chunks and its scales as halves.
/// `blocks` holds the rows' ggml Q8_0 blocks in order (rows * K / 32 blocks); `out` takes rows * K / 32 * 34 bytes.
/// K must be a multiple of 32.
void fused_gr_q8_0_split_host(const uint8_t* blocks, uint8_t* out, int64_t rows, int64_t K);

}  // namespace strata::kernels
