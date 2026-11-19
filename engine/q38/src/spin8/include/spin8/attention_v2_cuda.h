#ifndef SPIN8_ATTENTION_V2_CUDA_H
#define SPIN8_ATTENTION_V2_CUDA_H
#include "spin8/attention_v2.h"
#include <cuda_runtime_api.h>
#ifdef __cplusplus
extern "C" {
#endif

#define SPIN8_ATTENTION_V2_CUDA_ABI_VERSION 2u
#define SPIN8_ATTENTION_V2_CUDA_REFERENCE 0u
#define SPIN8_ATTENTION_V2_CUDA_NONFINITE 1u
#define SPIN8_ATTENTION_V2_CUDA_ALL_MASKED 2u

typedef struct spin8_attention_v2_cuda_plan spin8_attention_v2_cuda_plan;
/* Runtime failures are SPIN8_INVALIDATED with the actual CUDA error. Semantic
 * and descriptor errors have cudaSuccess and a non-OK Spin8 status. */
typedef struct {
  spin8_status status;
  cudaError_t cuda_error;
} spin8_attention_v2_cuda_result;

typedef struct {
  uint32_t abi_version, struct_bytes;
  uint32_t operation, k_dtype, v_dtype;
  uint32_t reserved0;
  /* Same geometry, scale and ordering as B1. Here Q/bias/gate/output are F32;
   * K/V storage types are independent. Mask remains bytes. All data is device
   * memory on the plan device, naturally aligned with item-aligned strides. */
  spin8_attention_v2_request attention;
  /* Explicit created stream, not null/legacy/per-thread. */
  cudaStream_t stream;
  void *workspace; /* 8-byte aligned; query exact size below. */
  size_t workspace_bytes;
  uint32_t *device_status;
  size_t device_status_bytes;
  uint64_t reserved[4];
} spin8_attention_v2_cuda_request;

/* Reuses the B1 config version/size/geometry, except dtype MUST be F32 here.
 * Creates/copies audited coefficients and mapping once. Device must be current;
 * this bounded implementation supports sm_120. Creation is outside capture.
 * Success means the initialization upload has completed and been checked;
 * immediate use on another nonblocking stream needs no caller-side wait.
 * No caller data is retained. *out is unchanged on failure. */
spin8_attention_v2_cuda_result
spin8_attention_v2_cuda_plan_create(const spin8_attention_v2_config *,
                                    size_t config_bytes, int device,
                                    spin8_attention_v2_cuda_plan **out);
/* Caller completes all uses (including graph replays) first, on the current
 * plan device. cudaFree may synchronize; this is not a hot-path operation.
 * The library never switches the current device. */
spin8_attention_v2_cuda_result
spin8_attention_v2_cuda_plan_free(spin8_attention_v2_cuda_plan *);

/* Includes packed FP64 Q/K/V, full scores and staged FP64 output; no hidden
 * per-call allocations. On failure *bytes is unchanged. */
spin8_attention_v2_cuda_result
spin8_attention_v2_cuda_workspace_bytes(const spin8_attention_v2_cuda_plan *,
                                        size_t n_query, size_t n_key,
                                        size_t *bytes);

/* Success means ENQUEUED, not completed or numerically valid. After checking
 * CUDA stream/event completion, read device_status in stream order: 0 success,
 * otherwise the bitmask above. A separate final kernel publishes output ONLY
 * if every row passed. Host validation precedes any enqueue or output/status
 * mutation. CUDA execution failures require discarding the invocation.
 * Output/workspace/status are mutually disjoint and disjoint from all inputs,
 * descriptors and plan. Inputs may alias each other. Keep device allocations
 * live and inputs stable until completion; keep plan live through graph replay.
 * Distinct workspace/output/status allow concurrent streams on one plan.
 * No allocation, host copy or synchronization in enqueue. See the CUDA guide.
 */
spin8_attention_v2_cuda_result
spin8_attention_v2_cuda_enqueue(const spin8_attention_v2_cuda_plan *,
                                const spin8_attention_v2_cuda_request *,
                                size_t request_bytes);

#ifdef __cplusplus
}
#endif
#endif
