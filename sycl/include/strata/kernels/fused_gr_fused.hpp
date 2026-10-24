// sycl/include/strata/kernels/fused_gr_fused.hpp - STRATA_GR_FUSED on SYCL: the two-kernel BF16 hyper-connection read
// (fused_gr.dp.cpp, gr_fused), switched for a test or bench.
#pragma once

namespace strata::kernels {

/// The two-kernel BF16 read on (1) or off (0) for every fused_gr_read_multi of this process from now on; -1 = back to
/// STRATA_GR_FUSED. The parity test runs both reads in one process with it.
void fused_gr_set_fused(int on);

}  // namespace strata::kernels
