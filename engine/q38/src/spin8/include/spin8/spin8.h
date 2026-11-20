#ifndef SPIN8_SPIN8_H
#define SPIN8_SPIN8_H
#include <stddef.h>
#include <stdint.h>
#ifdef __cplusplus
extern "C" {
#endif
typedef enum {
  SPIN8_OK = 0,
  SPIN8_INVALID_ARGUMENT = 1,
  SPIN8_NONFINITE = 2,
  SPIN8_UNSUPPORTED = 3,
  SPIN8_ALLOCATION_FAILED = 4,
  SPIN8_AUDIT_FAILED = 5,
  SPIN8_INVALIDATED = 6
} spin8_status;
typedef enum {
  SPIN8_VECTOR = 0,
  SPIN8_SPIN_PLUS = 1,
  SPIN8_SPIN_MINUS = 2
} spin8_view;
typedef enum {
  SPIN8_F32 = 0,
  SPIN8_F64 = 1,
  SPIN8_F16 = 2,
  SPIN8_BF16 = 3
} spin8_dtype;
typedef struct spin8_plan spin8_plan;
typedef struct {
 uint32_t abi_version;
 uint32_t cpu_dtype_mask;
 uint32_t representation_dimension;
 uint32_t hadamard_max_block;
} spin8_capabilities;
/* Reports compiled CPU operations, not device presence or model compatibility. */
spin8_status spin8_get_capabilities(spin8_capabilities *out);
typedef struct {
  uint32_t a, b;
  double angle;
} spin8_plane;
typedef struct {
  double ef_normalized, max_error, log_abs_det;
  int det_sign;
} spin8_matrix_audit;
typedef struct {
  spin8_matrix_audit views[3];
  double triality_max_error;
} spin8_audit_report;
typedef struct {
  uint64_t last_tokens;
  double last_seconds;
  spin8_status failure;
  int valid;
} spin8_monitor;
/* Input plane indices are 0..7. Factors act in array order: final U=Ulast ...
 * Ufirst. A null planes pointer is legal only for count=0 (identity). Plans own
 * no caller data. Immutable plans may be shared across threads. No exception
 * crosses the C boundary. */
spin8_status spin8_plan_create(const spin8_plane *planes, size_t count,
                               spin8_plan **out);
void spin8_plan_free(spin8_plan *plan);
spin8_status spin8_plan_matrix(const spin8_plan *plan, spin8_view view,
                               double out[64]);
spin8_status spin8_plan_audit(const spin8_plan *plan, spin8_audit_report *out);
/* Diagnostic mathematical construction: 8 real row-major 16x16 Cl(8,0)
 * generators. Chirality is Gamma0 ... Gamma7; its +1 and -1 eigenspaces are the
 * two 8 blocks. */
spin8_status spin8_clifford_generators(double out[2048]);
/* Strides are in scalar elements; capacity is checked by caller using
 * (rows-1)*stride+cols. Overlapping storage is rejected except exact in-place
 * with equal strides. Full blocks of 8 transform; a final 1..7 scalars stay
 * unchanged. All arguments and finite input are checked before writing any
 * output. */
spin8_status spin8_transform(const spin8_plan *, spin8_view, spin8_dtype,
                             const void *src, size_t src_stride, void *dst,
                             size_t dst_stride, size_t rows, size_t cols,
                             int inverse);
spin8_status spin8_hadamard(spin8_dtype, const void *src, size_t src_stride,
                            void *dst, size_t dst_stride, size_t rows,
                            size_t cols, uint8_t sign_mask, int inverse);
spin8_status spin8_hadamard_matrix(uint8_t sign_mask, int inverse,
                                   double out[64]);
/* General Sylvester HD/DH, including Bonsai block=128. Block is a power of two
 * in [1,1024]. signs[block] contains +1/-1; null = all +1. Incomplete tails are
 * identity. FP64 butterfly accumulation, with one final cast for F32 storage.
 */
spin8_status spin8_hadamard_block(spin8_dtype, const void *src,
                                  size_t src_stride, void *dst,
                                  size_t dst_stride, size_t rows, size_t cols,
                                  size_t block, const int8_t *signs,
                                  int inverse);
/* Factor audit for actual rounded coefficient c (F32 or F64): Gram error is
 * abs(block*c*c-1); logabsdet=block*log(abs(c)*sqrt(block)); sign is Sylvester
 * parity times product(signs). This analytical coefficient-model audit is
 * distinct from numerical LU and measured runtime forward/inverse errors. */
spin8_status spin8_hadamard_block_audit(spin8_dtype, size_t block,
                                        const int8_t *signs,
                                        spin8_matrix_audit *out);
spin8_status spin8_audit_matrix(const double matrix[64],
                                spin8_matrix_audit *out);
double spin8_precision_tolerance(spin8_dtype dtype);
/* Weighted scalar reduction of three equally shaped views. This is a convex
 * reduction, not a Spin(8) representation or norm-preserving rotation. */
spin8_status spin8_reduce(spin8_dtype, const void *const views[3],
                          size_t src_stride, void *dst, size_t dst_stride,
                          size_t rows, size_t cols, const double weights[3]);
/* C = weighted sum_v (Mv Arow) dot (Mv Brow); weights null = {1/3,1/3,1/3}.
 * Custom weights must be finite, nonnegative and sum to 1 within 1e-12.
 * Matrices are contiguous; A[m,k], B[n,k], out[m,n]. Outputs must not alias
 * inputs. */
spin8_status spin8_matmul_f64(const spin8_plan *, const double *a, size_t m,
                              const double *b, size_t n, size_t k,
                              const double weights[3], double *out);
spin8_status spin8_matmul_f32(const spin8_plan *, const float *a, size_t m,
                              const float *b, size_t n, size_t k,
                              const double weights[3], float *out);
/* Q[nq,d],K[nk,d],V[nk,dv], out[nq,dv]. Mask[nq,nk]: nonzero means allowed;
 * null means all allowed. Weighted three-view logits / sqrt(d), then ONE
 * softmax and reduction of original V. Empty/all-masked rows are errors. */
spin8_status spin8_attention_f64(const spin8_plan *, const double *q, size_t nq,
                                 const double *k, size_t nk, const double *v,
                                 size_t dv, size_t d, const uint8_t *mask,
                                 const double weights[3], double *out);
spin8_status spin8_attention_f32(const spin8_plan *, const float *q, size_t nq,
                                 const float *k, size_t nk, const float *v,
                                 size_t dv, size_t d, const uint8_t *mask,
                                 const double weights[3], float *out);
/* Explicit monotonic clock/token inputs make scheduling deterministic. Initial
 * audit is due. Reaudit at 1024 tokens OR 60 seconds. Failure is sticky until
 * reset. Caller serializes monitor access; immutable plan operations need no
 * monitor. */
void spin8_monitor_init(spin8_monitor *);
spin8_status spin8_monitor_due(const spin8_monitor *, uint64_t tokens,
                               double seconds, int *due);
spin8_status spin8_monitor_record(spin8_monitor *, uint64_t tokens,
                                  double seconds, spin8_status result);
const char *spin8_status_string(spin8_status);
#ifdef __cplusplus
}
#endif
#endif
