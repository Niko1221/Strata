#ifndef SPIN8_QUERY_V3_CUDA_H
#define SPIN8_QUERY_V3_CUDA_H

#include "spin8/cuda.h"
#include "spin8/query_v3.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct spin8_query_v3_cuda_plan spin8_query_v3_cuda_plan;

/* Host descriptor, device arrays. All parameters are contiguous row-major F32:
 * Win[8,width], bin[8], Wout[7,8], bout[7]. Capacities count bytes. Arrays must
 * be disjoint from each other, source, destination and error_flag. The caller
 * keeps arrays immutable until every launch using them finishes. ALL rows in
 * one call share this parameter set; callers choose the per-head grouping. */
typedef struct {
  const float *win;
  size_t win_bytes;
  const float *bin;
  size_t bin_bytes;
  const float *wout;
  size_t wout_bytes;
  const float *bout;
  size_t bout_bytes;
} spin8_query_v3_cuda_parameters_f32;

#define SPIN8_QUERY_V3_CUDA_BANK_VERSION 1u

/* Version 1 host descriptor of immutable device F32 arrays. Each head owns
 * contiguous Win[8,width], bin[8], Wout[7,8], bout[7], shared across tokens.
 * Each array has its own byte capacity and byte head stride (including
 * padding). Strides must be float-aligned and at least one head's payload, even
 * for one head. Array occupied spans, including inter-head padding, must be
 * disjoint from one another, src/dst and error_flag. No implicit broadcast is
 * allowed. */
typedef struct {
  uint32_t version;
  size_t heads;
  size_t width;
  const float *win;
  size_t win_bytes;
  size_t win_head_stride_bytes;
  const float *bin;
  size_t bin_bytes;
  size_t bin_head_stride_bytes;
  const float *wout;
  size_t wout_bytes;
  size_t wout_head_stride_bytes;
  const float *bout;
  size_t bout_bytes;
  size_t bout_head_stride_bytes;
} spin8_query_v3_cuda_bank_f32;

/* Host-only creation/destruction. Version must be SPIN8_QUERY_V3_VERSION.
 * Derives and audits signed permutations from spin8_query_v3_generators;
 * no GPU allocation, upload or device state. *out is null on failure.
 * Plans are immutable, shareable and copied into each launch. The caller
 * serializes destruction against calls that are still enqueueing work. */
spin8_status spin8_query_v3_cuda_plan_create(uint32_t version,
                                             spin8_query_v3_cuda_plan **out);
void spin8_query_v3_cuda_plan_free(spin8_query_v3_cuda_plan *plan);

/* Explicit opt-in fused v3 controller and Triality Q transform. src/dst are
 * device F32 [rows,width]; BOTH strides and capacities count BYTES. q is
 * already model-normalized: no extra RMS. Per row, on GPU:
 * a=Wout*tanh(Win*q+bin)+bout; tau=tan(pi/8);
 * u=tau*a/sqrt(tau^2+||a||^2); c=(1,u)/sqrt(1+||u||^2).
 * Full block8 groups share c and apply c0*I+sum(c[a+1]*J[a]). A 1..7 tail
 * feeds the controller but copies exactly. Wout=bout=0 gives bitwise identity,
 * including signed zeros. Double accumulation/chart favors reference accuracy;
 * product/sum rounding and dot reduction order follow the CPU reference.
 * Host/device tanh may differ. Tolerances are measured for tested cases only;
 * no universal CPU bitwise/tolerance guarantee holds for arbitrary finite
 * weights. Callers must calibrate/audit their parameter profiles.
 * No speed claim is made. No per-row dense matrices or global spinor buffer.
 *
 * ALL source/parameter scalars and computed results must be finite, even at
 * identity. Unlike CPU apply's identity shortcut, GPU checks Win/bin too.
 * Caller clears device error_flag; this function only ORs SPIN8_CUDA_NONFINITE.
 * Discard ALL output on any nonzero flag. Device failure is NOT atomic.
 *
 * Nonzero rows/width, float alignment, checked spans, capacity and alias rules
 * are validated before launch. Only exact src==dst with equal strides may
 * overlap. Padding is untouched. Device pointer validity, actual allocation
 * bounds, device/stream ownership and buffer lifetime remain caller contracts.
 * No allocation, host-device copy, synchronization or hidden default-stream
 * work. One kernel is enqueued on stream; launch errors are returned. Caller
 * must synchronize stream and check its asynchronous status and error_flag
 * before consuming output. The host descriptor is borrowed only for this call.
 */
cudaError_t spin8_query_v3_cuda_apply_f32(
    const spin8_query_v3_cuda_plan *plan,
    const spin8_query_v3_cuda_parameters_f32 *parameters, const float *src,
    size_t src_capacity_bytes, size_t src_stride_bytes, float *dst,
    size_t dst_capacity_bytes, size_t dst_stride_bytes, size_t rows,
    size_t width, int *error_flag, cudaStream_t stream);

/* Batched opt-in v3 execution with the SAME mathematical/numerical and caller
 * ownership contracts as apply_f32 above. Logical tensor shape is
 * [channels=width, query_heads=heads, tokens], channels contiguous. Address:
 * base + token*token_stride_bytes + head*head_stride_bytes + channel*4.
 * A head stride is >= width*4; a token stride is >= the occupied head span
 * (heads-1)*head_stride + width*4. All strides are float-aligned, positive,
 * and required even for singleton dimensions. This supports padded disjoint
 * rows, prefill and decode. heads/width must match the bank; all dimensions
 * are nonzero. Exact in-place requires equal src/dst head AND token strides.
 * Other overlaps, including overlaps through padding, are rejected on host.
 * Capacities, pointer alignment, spans, version and arithmetic overflow are
 * checked before enqueueing one kernel on stream, without allocation or sync.
 * Padding remains untouched; identity tails feed Win and copy bitwise.
 * Caller clears error_flag before use. ANY flagged nonfinite input, parameter
 * or result invalidates the ENTIRE batch; discard all output on nonzero flag
 * after checking asynchronous stream completion. This is NOT atomic output
 * rollback: finite rows may already have been written when another row fails.
 * Parameter padding is not read or checked for finiteness. Host bank lifetime
 * ends on return; device arrays must remain immutable/alive through completion.
 */
cudaError_t spin8_query_v3_cuda_apply_batched_f32(
    const spin8_query_v3_cuda_plan *plan,
    const spin8_query_v3_cuda_bank_f32 *bank, const float *src,
    size_t src_capacity_bytes, size_t src_head_stride_bytes,
    size_t src_token_stride_bytes, float *dst, size_t dst_capacity_bytes,
    size_t dst_head_stride_bytes, size_t dst_token_stride_bytes, size_t heads,
    size_t tokens, size_t width, int *error_flag, cudaStream_t stream);

#ifdef __cplusplus
}
#endif
#endif
