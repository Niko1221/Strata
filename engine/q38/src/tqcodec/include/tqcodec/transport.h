#ifndef TQCODEC_TRANSPORT_H
#define TQCODEC_TRANSPORT_H

#include "tqcodec/cache.h"
#ifdef __cplusplus
extern "C" {
#endif

/* CPU FP64 reference only. One fixed proper orthogonal S applies to EVERY
 * selected logical K: logical reconstruction = S * stored reconstruction.
 * The cache, its three U matrices, shared V and QJL signs/G/norms never change.
 * This is not a per-row position manager or a model/RoPE integration. */
typedef struct tq_transport_plan tq_transport_plan;
typedef struct {
  size_t width;
  double gram_frobenius_error; /* ||S^T S-I||F / sqrt(width), normalized E_F */
  double gram_max_error;
  double log_abs_determinant;
  int determinant_sign;
} tq_transport_audit;
typedef struct {
  size_t fixed_bytes;
  size_t matrix_bytes;
  size_t total_bytes;
} tq_transport_ledger;

/* S is row-major width x width, width >= 8. matrix_count is an accessible
 * element count; matrix_byte_cap bounds the allocated matrix bytes (not the
 * plan object or temporary audit scratch). The plan snapshots S and audits
 * S^T*S and determinant +1 once, using the cache's FP64 1e-12 tolerances.
 * Source ownership stays with the caller. Failure leaves *out unchanged. */
tq_status tq_transport_plan_create(size_t width, const double *matrix,
    size_t matrix_count, size_t matrix_byte_cap, tq_transport_plan **out);
void tq_transport_plan_destroy(tq_transport_plan *plan);
tq_status tq_transport_plan_audit(const tq_transport_plan *plan,
    tq_transport_audit *out);
/* External plan storage, separate from both cache ledger ABIs. Counts sizeof
 * the plan and allocated matrix capacity. Excludes allocator metadata,
 * alignment overhead and transient scratch. */
tq_status tq_transport_plan_bytes(const tq_transport_plan *plan,
    tq_transport_ledger *out);

/* Transform one logical query or NEW logical key to canonical storage
 * coordinates: dst = S^T * src. Counts are elements, width or larger.
 * Canonicalize new logical keys before tq_cache_append; V is untransformed.
 * Exact in-place is supported through staging; partial overlap is invalid.
 * All typed pointers require their natural alignment. Errors leave outputs
 * unchanged. Input/result nonfinites return TQ_NONFINITE. */
tq_status tq_transport_to_storage_f64(const tq_transport_plan *plan,
    const double *src, size_t src_count, double *dst, size_t dst_count);

/* Computes S^T*q once, then calls the existing single-softmax cache attend.
 * The same S applies uniformly to all allowed rows; masked rows are excluded.
 * QJL uses the existing sign(G*e) with G*(S^T*q), never sign(G*S*e).
 * Exact query/output in-place is supported, partial overlap is invalid.
 * Output may not overlap mask or plan/cache storage. No persistent position
 * metadata, per-row different S, CUDA, or full-network rerun equivalence. */
tq_status tq_cache_attend_transported(const tq_cache *cache,
    const tq_transport_plan *plan, const double *query, size_t query_count,
    const uint8_t *allowed, size_t allowed_count, double scale, double *out,
    size_t out_count);

#ifdef __cplusplus
}
#endif
#endif
