#ifndef SPIN8_QUERY_H
#define SPIN8_QUERY_H

#include "spin8/spin8.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct spin8_query_controller spin8_query_controller;

/* T(v,p,m) = sum_i,a,b v[i] p[a] Gamma_i[a,8+b] m[b], with the
 * chirality-ordered generators returned by spin8_clifford_generators.
 * All three vectors have eight finite doubles. */
spin8_status spin8_query_contract(const double vector[8],
                                  const double spin_plus[8],
                                  const double spin_minus[8], double *out);

/* C(c)[i,a] = sum_b Gamma_i[a,8+b] c[b]; A(c)=C(c) C(e0)^T.
 * c must be finite and unit length within 1e-13. No output is written on
 * failure. The resulting proper orthogonal matrix is audited by the shared
 * spin8_audit_matrix routine. */
spin8_status spin8_query_matrix(const double spinor[8], double out[64]);
spin8_status spin8_query_matrix_audit(const double spinor[8],
                                      spin8_matrix_audit *out);

/* h[7] maps to c=cos(theta)e0+sinc(theta)u, with
 * u=(pi/8)h/sqrt(1+||h||^2), theta=||u|| and e1..e7 as tangent basis.
 * This bounded map accepts every finite h, including large magnitudes. */
spin8_status spin8_query_spinor_from_tangent(const double h[7], double out[8]);

/* Apply A(c) to each full group of eight columns in each row. The remaining
 * 1..7 columns are copied exactly. Strides count elements; capacities count
 * bytes and include the final accessed element. Partial overlap is rejected;
 * exact in-place storage with equal strides is allowed. All readable values
 * and computed results are checked before any output is written. F32 and F64
 * storage are supported. The caller owns the buffers for the call. */
spin8_status spin8_query_transform(const double spinor[8], spin8_dtype dtype,
                                   const void *src, size_t src_capacity_bytes,
                                   size_t src_stride, void *dst,
                                   size_t dst_capacity_bytes, size_t dst_stride,
                                   size_t rows, size_t head_width);

/* Create an immutable rank-eight controller. Weights are copied from row-major
 * Wdown[8,head_width] and Wup[7,8], with no bias. Capacities are byte counts.
 * A zero Wup yields exact identity for finite inputs. No model state is owned.
 */
spin8_status spin8_query_controller_create(size_t head_width,
                                           const double *w_down,
                                           size_t w_down_capacity_bytes,
                                           const double *w_up,
                                           size_t w_up_capacity_bytes,
                                           spin8_query_controller **out);
void spin8_query_controller_free(spin8_query_controller *controller);

/* Per row: xhat=x/sqrt(mean(x^2)+1e-6), h=Wup*tanh(Wdown*xhat), then
 * transform the same query row with A(c(h)). One spinor is shared by all
 * complete eight-column groups of that row; the tail stays unchanged.
 * Capacity, aliasing, dtype, and failure-atomic rules match transform. */
spin8_status spin8_query_controller_apply(
    const spin8_query_controller *controller, spin8_dtype dtype,
    const void *src, size_t src_capacity_bytes, size_t src_stride, void *dst,
    size_t dst_capacity_bytes, size_t dst_stride, size_t rows);

#ifdef __cplusplus
}
#endif
#endif
