#pragma once

// Dense short-window INT8 tensor-core projections (BW-DENSE, P2.1). A separate translation unit
// (src/kernels/cuda/dense_tc.cu), CUDA sm_120 only; every other build and every ineligible call keeps native_mmvq /
// native_mmvq_il. All pointers are device pointers.
//
// Inputs are the plain q8_1 blocks native_quantize_q8_1 writes (36 bytes per 32 values, column j at
// x + j * n_in / 32 blocks) and the unmodified GGUF weight blocks, so no extra copy of either is made.
// Output column j is y + j * n_out (native_mmvq's layout).
//
// Arithmetic contract (docs: phaseA-tests/blackwell/decisions/E03.md): the integer dot products of each
// (row, 16- or 32-value sub-block, column) are exact int32 (mma.sync s8 x s8 -> s32); the scale products and the
// float accumulation have a fixed order that depends on the row, the block and (n_in, n_out) only, never on the
// column count, so a column's result does not depend on how many columns share the call. That order is not
// native_mmvq's (it multiplies and rounds per 4-value dp4a group), so the output is NOT bitwise native_mmvq's: this
// is a rounding-changing path, off unless dense_tc_enable(true) / STRATA_DENSE_TC=1.
#include <cstddef>

namespace strata::kernels {

// 0 = eligible. Otherwise a stable reason, checked before graph capture; the caller then keeps its existing path.
enum DenseTcStatus {
    kDenseTcOk = 0,
    kDenseTcOff = 1,      // switch off (default)
    kDenseTcArch = 2,     // not an sm_120 card / build without sm_120 code
    kDenseTcType = 3,     // weight type without a tensor-core kernel
    kDenseTcCols = 4,     // ncols outside 1..8
    kDenseTcShape = 5,    // n_in not a multiple of 256, n_out < 1, or an activation tile that does not fit
    kDenseTcAlign = 6,    // a pointer is not 4-byte aligned (weights: 2-byte aligned blocks are fine)
    kDenseTcSmem = 7,     // the card refuses the shared memory the tile needs
    kDenseTcBuild = 8,    // not built (HIP / SYCL / CPU / no sm_120 code)
};
const char* dense_tc_reason(int status) noexcept;

// Process switch. Default: STRATA_DENSE_TC (unset or 0 = off).
void dense_tc_enable(int on);   // 1 on, 0 off, -1 back to the environment
bool dense_tc_enabled();

// Eligibility of one projection (no launch, no allocation). `weights`, `x_q8_1` and `y` may be null to skip the
// pointer checks (a plan made before the buffers exist).
int dense_tc_check(int ggml_type, int n_in, int n_out, int ncols, const void* weights, const void* x_q8_1,
                   const void* y) noexcept;

// Launches the kernel on `stream` when dense_tc_check says kDenseTcOk and returns true; returns false (nothing
// launched) otherwise, and the caller calls native_mmvq / native_mmvq_il.
bool dense_tc(int ggml_type, const void* weights, const void* x_q8_1, float* y, int n_in, int n_out, int ncols,
              void* stream);

// Tests and benchmarks.
void dense_tc_tile(int row_tiles, int k_split);                 // 0, 0: the shape table (default)
int dense_tc_split_for(int ggml_type, int n_in, int n_out);     // the k_split a launch of this shape uses (the rounding order's part)
int dense_tc_variant(int v);                                     // -1 read, else set the kernel variant (experiments)
// Registers/spills/shared memory of the kernels in this build (cudaFuncGetAttributes), one line per instance, to stdout.
void dense_tc_print_resources();

}  // namespace strata::kernels
