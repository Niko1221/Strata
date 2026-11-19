#ifndef TQCODEC_DIALECTS_CUDA_H
#define TQCODEC_DIALECTS_CUDA_H
#include "dialects.h"
#ifdef __cplusplus
extern "C" {
#endif

/* Additive, asynchronous CUDA ABI. Does NOT change dialects.h CPU semantics.
 * Only the five pinned TheTom Turbo KV dialects, F32, columns 128 or 256,
 * and sm120 are qualified. Each 128-value block is independent.
 * Operation IDs and coordinate/packing conventions are those in dialects.h.
 * PREPARE -> optional B -> PACK applies F exactly once. DECODE returns stored
 * coordinates; INVERSE applies F^-1, not an implicit B^-1.
 *
 * Host OK means ENQUEUED, not complete or valid. No per-call allocation,
 * synchronization, device switching, or default-stream work. Caller selects
 * device_ordinal first and supplies a live explicitly created nondefault CUDA
 * stream on that device (opaque cudaStream_t cast to void*). CUDA must already
 * be initialized. Request is copied before return; device buffers must remain
 * alive until completion. Inputs must remain immutable until their operation
 * completes; later writes ordered on the same stream are permitted. Caller owns
 * actual CUDA allocation bounds, device ownership and inter-stream
 * dependencies. Declared byte capacities/strides, integer overflow, alignment
 * and aliases are checked on host, not actual allocation provenance. No
 * overlapping spans, including row padding, status and host descriptor. Unused
 * norms must be all zero.
 *
 * status is a device uint32_t, aligned to 4, with status_capacity_bytes >= 4.
 * Caller initializes it to zero on the supplied stream before first use.
 * Kernel atomically ORs error bits; it never clears status. A preexisting
 * nonzero status skips work. A newly flagged error invalidates ALL output
 * of this call; other blocks may already have written. Discard it. Only consume
 * results after successful stream completion AND status == 0. CUDA runtime
 * failures may prevent status writes: checking status alone is insufficient.
 * A host structural rejection leaves output/status unchanged. RUNTIME means
 * CUDA validation/launch failed (possibly a preceding asynchronous error);
 * treat output as invalid and check stream completion before releasing buffers.
 * Input numerical limits and FP16 overflow rejection match CPU dialects.h.
 */
enum { TQC_DC_RUNTIME = 100 };
enum { TQC_DC_NONFINITE = 1u, TQC_DC_RANGE = 2u };
typedef struct tqc_dialect_cuda_v1_request {
  uint32_t version, size;
  uint32_t dialect, operation, precision;
  int32_t device_ordinal;
  void *stream;
  size_t rows, columns;
  tqc_dialect_v1_span src, dst, norms;
  uint32_t *status;
  size_t status_capacity_bytes;
} tqc_dialect_cuda_v1_request;
typedef struct tqc_dialect_cuda_v1_info {
  uint32_t version, size, dialect, block_elements, block_bytes;
  uint32_t operation_mask; /* bit (1u << TQC_D_* operation) */
  uint32_t precision, compute_capability, columns_128, columns_256;
  const char *identity;
  const char *source_revision;
} tqc_dialect_cuda_v1_info;
/* Host-only static capability query; does not initialize CUDA. */
int tqc_dialect_cuda_v1_query(uint32_t dialect, tqc_dialect_cuda_v1_info *info);
int tqc_dialect_cuda_v1_enqueue(const tqc_dialect_cuda_v1_request *request);
#ifdef __cplusplus
}
#endif
#endif
