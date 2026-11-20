#ifndef TQCODEC_H
#define TQCODEC_H
#include <stddef.h>
#include <stdint.h>
#ifdef __cplusplus
extern "C" {
#endif
typedef enum {
  TQ_OK = 0,
  TQ_INVALID = 1,
  TQ_NONFINITE = 2,
  TQ_CAPACITY = 3
} tq_status;
/* Explicit scalar-codebook building block, NOT a legacy GGUF block layout.
 * Caller owns rotation and codebook provenance. No implicit Hadamard is
 * applied. Indices use little-endian, least-significant-bit-first packing,
 * bits=2..6. Norm is a separate FP64 reference value; physical sizes include
 * it. */
typedef struct {
  size_t payload_bytes;
  size_t norm_bytes;
  size_t total_bytes;
} tq_size;
tq_status tq_block_size(size_t width, unsigned bits, tq_size *out);
tq_status tq_encode_f64(const double *input, size_t width,
                        const double *centroids, unsigned bits, uint8_t *packed,
                        size_t capacity, double *norm);
tq_status tq_decode_f64(const uint8_t *packed, size_t capacity, size_t width,
                        const double *centroids, unsigned bits, double norm,
                        double *out);
tq_status tq_score_f64(const double *query, const uint8_t *packed,
                       size_t capacity, size_t width, const double *centroids,
                       unsigned bits, double norm, double *unscaled_score);
/* G[m,d] must be a caller-supplied versioned Gaussian fixture for QJL claims.
 * Signs are bitpacked, zero projections use +1. Zero residual has norm=0.
 * No random generator or stochastic state is hidden in these routines. */
tq_status tq_qjl_encode_f64(const double *residual, size_t d, const double *g,
                            size_t m, uint8_t *signs, size_t capacity,
                            double *norm);
tq_status tq_qjl_score_f64(const double *query, size_t d, const double *g,
                           size_t m, const uint8_t *signs, size_t capacity,
                           double norm, double beta, double *correction);
/* Error of (approximate base score + beta * QJL residual correction) relative
 * to the exact score, with fixed query/residual independent of iid N(0,1) G.
 * Includes attention_scale, but excludes softmax and V error. These are
 * distributional moments across independent G, NOT a guarantee for a fixed
 * sketch reused during generation or fitted against the same sketch.
 * beta=0 requires no sketch storage; m still specifies the comparison sketch.
 * Finite intermediate overflow returns TQ_NONFINITE without changing out. */
typedef struct {
  double bias;
  double variance;
  double mean_squared_error;
} tq_qjl_moments;
tq_status tq_qjl_error_moments_f64(const double *query,
                                   const double *residual, size_t d, size_t m,
                                   double beta, double attention_scale,
                                   tq_qjl_moments *out);
/* All three reconstructions are in COMMON coordinates; simplex weights only.
 * out may equal key, but must not overlap the reconstruction array. */
tq_status tq_shared_residual_f64(const double *key, const double *views[3],
                                 size_t d, const double alpha[3], double *out);
#ifdef __cplusplus
}
#endif
#endif
