#ifndef SPIN8_QUERY_CUDA_H
#define SPIN8_QUERY_CUDA_H

#include "spin8/cuda.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct spin8_query_cuda_plan spin8_query_cuda_plan;

/* Build immutable signed-permutation constants from the actual
 * chirality-ordered Clifford generators. The caller owns the plan and keeps it
 * alive until all launches using it have completed. Plans may be shared by
 * concurrent callers.
 */
spin8_status spin8_query_cuda_plan_create(spin8_query_cuda_plan **out);
void spin8_query_cuda_plan_free(spin8_query_cuda_plan *plan);

enum { SPIN8_QUERY_CUDA_INVALID_SPINOR = 8 };

/* c is device F32 [rows,8], with c_stride in elements. src/dst are device
 * F32 [rows,width]. All three capacities are bytes, including the final
 * accessed scalar. Full groups of eight use A(c)=C(c)C(e0)^T; a tail of 1..7
 * is copied exactly. src==dst with equal strides is supported. Other overlap,
 * including overlap with c or error_flag, is rejected before launch.
 *
 * Each c must be finite and have norm within 1e-5 of one. The caller clears
 * the device error_flag before the first launch; this API only ORs flags.
 * SPIN8_CUDA_NONFINITE reports nonfinite c/input/result, and
 * SPIN8_QUERY_CUDA_INVALID_SPINOR reports an invalid c. A nonzero flag means
 * destination contents must be discarded; kernel failure is not atomic.
 * No allocation, host-device copy, or stream synchronization occurs here.
 * CUDA launch errors are returned; asynchronous errors require the caller to
 * synchronize the supplied stream. Capacities cannot establish actual device
 * allocation ownership, which remains the caller's responsibility.
 */
cudaError_t spin8_query_cuda_transform_f32(
    const spin8_query_cuda_plan *plan, const float *c, size_t c_capacity_bytes,
    size_t c_stride, const float *src, size_t src_capacity_bytes,
    size_t src_stride, float *dst, size_t dst_capacity_bytes, size_t dst_stride,
    size_t rows, size_t width, int *error_flag, cudaStream_t stream);

#ifdef __cplusplus
}
#endif
#endif
