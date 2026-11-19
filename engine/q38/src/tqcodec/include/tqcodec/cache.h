#ifndef TQCODEC_CACHE_H
#define TQCODEC_CACHE_H

#include "tqcodec/tqcodec.h"
#ifdef __cplusplus
extern "C" {
#endif

/* CPU reference dialect. Matrices are row-major d x d and orthogonal.
 * All pointers are borrowed only during creation; the cache snapshots them.
 * The legacy create API has no QJL correction. No model-specific packing or
 * persistence is implied by either API. */
typedef struct tq_cache tq_cache;
typedef struct {
  size_t width;
  size_t capacity_rows;
  unsigned key_bits;
  unsigned value_bits;
  const double *transforms[3];
  double alpha[3];
  const double *key_centroids;
  const double *value_centroids;
} tq_cache_config;

#define TQ_CACHE_VERSION_2 2u
/* Explicit opt-in, one cache per KV head. version must be TQ_CACHE_VERSION_2.
 * beta is in [0,1]. beta=0 ignores all sketch fields and allocates no sketch
 * arrays or residual norms. Otherwise gaussian is a caller-supplied row-major
 * qjl_rows x base.width matrix, gaussian_count is its accessible element count,
 * and gaussian_byte_cap bounds its allocated bytes. All values are copied.
 * Caller owns matrix provenance/versioning; no randomness is generated here.
 * Gaussian independence required by error-moment theory is NOT validated. */
typedef struct {
  uint32_t version;
  tq_cache_config base;
  double qjl_beta;
  size_t qjl_rows;
  const double *gaussian;
  size_t gaussian_count;
  size_t gaussian_byte_cap;
} tq_cache_config_v2;

/* The ledger counts allocated vector capacities, including unused row slots.
 * fixed_bytes plus alpha_bytes is sizeof the cache object. Allocator metadata, alignment,
 * temporary stack/heap scratch, and any external controller are excluded. */
typedef struct {
  size_t fixed_bytes;
  size_t transform_bytes;
  size_t codebook_bytes;
  size_t alpha_bytes;
  size_t key_payload_bytes;
  size_t key_norm_bytes;
  size_t value_payload_bytes;
  size_t value_norm_bytes;
  size_t total_bytes;
} tq_cache_ledger;

/* base.total_bytes excludes the three added allocation categories; total_bytes
 * includes them. Both include unused row capacity. Same exclusions as above. */
typedef struct {
  uint32_t version;
  tq_cache_ledger base;
  size_t gaussian_bytes;
  size_t residual_sign_bytes;
  size_t residual_norm_bytes;
  size_t total_bytes;
} tq_cache_ledger_v2;

tq_status tq_cache_create(const tq_cache_config *config, tq_cache **out);
tq_status tq_cache_create_v2(const tq_cache_config_v2 *config, tq_cache **out);
void tq_cache_destroy(tq_cache *cache);
tq_status tq_cache_clone(const tq_cache *source, tq_cache **out);
size_t tq_cache_rows(const tq_cache *cache);
tq_status tq_cache_clear(tq_cache *cache);
tq_status tq_cache_bytes(const tq_cache *cache, tq_cache_ledger *out);
/* Legacy bytes rejects active QJL with TQ_INVALID, leaving out unchanged. */
tq_status tq_cache_bytes_v2(const tq_cache *cache, tq_cache_ledger_v2 *out);

/* key_count/value_count/query_count/out_count are element counts, not bytes.
 * Exact width or larger input buffers are accepted. No output/state changes
 * on a failed operation. */
tq_status tq_cache_append(tq_cache *cache, const double *key,
                          size_t key_count, const double *value,
                          size_t value_count);
tq_status tq_cache_attend(const tq_cache *cache, const double *query,
                          size_t query_count, const uint8_t *allowed,
                          size_t allowed_count, double scale, double *out,
                          size_t out_count);

/* Read-only copy of a physical row: view 0..2 = independently packed K;
 * view 3 = the single shared packed V. norm is copied separately. */
tq_status tq_cache_copy_row(const tq_cache *cache, size_t row, unsigned view,
                            uint8_t *payload, size_t payload_capacity,
                            double *norm);
/* Shared residual e=k-sum(alpha[r]*U[r]^T*decode(K[r])). Its single sign
 * vector is LSB-first, sign(0)=+1, unused tail bits=0, norm is FP64.
 * Zero residual has norm=0 and all active sign bits=1. Disabled QJL rejects
 * readback. Outputs must not overlap; errors leave them unchanged. */
tq_status tq_cache_copy_residual_row(const tq_cache *cache, size_t row,
                                     uint8_t *signs, size_t signs_capacity,
                                     double *norm);

#ifdef __cplusplus
}
#endif
#endif
