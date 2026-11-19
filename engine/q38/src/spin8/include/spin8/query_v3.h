#ifndef SPIN8_QUERY_V3_H
#define SPIN8_QUERY_V3_H

#include "spin8/spin8.h"

#ifdef __cplusplus
extern "C" {
#endif

#define SPIN8_QUERY_V3_VERSION 3u

typedef struct spin8_query_v3_controller spin8_query_v3_controller;
typedef struct {
  double normed_max_error;
  double polarization_max_error;
  double skew_max_error;
  double clifford_max_error;
} spin8_query_v3_basis_report;

/* Fixed positive-chirality convention from spin8_clifford_generators:
 * C(c)[i,a]=sum_b Gamma_i[a,8+b]c[b], J_a=C(e_{a+1})C(e0)^T.
 * out contains seven row-major 8x8 matrices. This is a seven-dimensional
 * isoclinic rotation family, not all SO(8), and is not closed under products.
 * Audits cover normed L and all eight C basis polarization identities as well
 * as skew symmetry and Cl(0,7). Outputs are unchanged on failure. */
spin8_status spin8_query_v3_generators(double out[448]);
spin8_status spin8_query_v3_basis_audit(spin8_query_v3_basis_report *out);
spin8_status spin8_query_v3_matrix(const double unit_spinor[8], double out[64]);

/* Rational chart: tau=tan(pi/8), u=tau*a/sqrt(tau^2+||a||^2),
 * c=(1,u)/sqrt(1+||u||^2). Accepts every finite a, even DBL_MAX in all seven
 * coordinates. c(0)=e0 exactly; Dc(0)=[e1,...,e7]. No silent normalization of
 * supplied spinors. The angle is bounded by pi/8 up to rounding. */
spin8_status spin8_query_v3_spinor_from_tangent(const double a[7],
                                                double out[8]);

/* Explicit v3 only. Copies finite row-major Win[8,width], bin[8], Wout[7,8],
 * bout[7], with capacities in bytes. Win is not forced to zero. The immutable
 * object may be shared across threads; caller serializes destruction.
 * a=Wout*tanh(Win*q+bin)+bout, where q is ALREADY normalized by the model.
 * There is no extra RMS normalization. Wout=bout=0 gives exact identity.
 * The caller chooses initialization; no model state or random state is owned.
 * *out is null on failure. No C++ exceptions cross this interface. */
spin8_status spin8_query_v3_controller_create(
    uint32_t version, size_t head_width, const double *win, size_t win_bytes,
    const double *bin, size_t bin_bytes, const double *wout, size_t wout_bytes,
    const double *bout, size_t bout_bytes, spin8_query_v3_controller **out);
void spin8_query_v3_controller_free(spin8_query_v3_controller *controller);
uint32_t
spin8_query_v3_controller_version(const spin8_query_v3_controller *controller);

/* F32/F64 rows; strides count elements, capacities count bytes including the
 * final accessed element. Full blocks of eight share one spinor per head row;
 * the final 1..7 scalars are copied exactly. Partial overlap is rejected;
 * exact in-place with equal strides is allowed. All input and computed output
 * is validated before writing. Row padding is untouched. Zero rows/width are
 * invalid. Arithmetic overflow returns NONFINITE without modifying output. */
spin8_status spin8_query_v3_controller_apply(
    const spin8_query_v3_controller *controller, spin8_dtype dtype,
    const void *src, size_t src_capacity_bytes, size_t src_stride, void *dst,
    size_t dst_capacity_bytes, size_t dst_stride, size_t rows);

/* Same controller, returns row-major F64 spinors for an external consumer.
 * dst_stride >= 8. Any source/destination overlap is rejected. Same validation
 * and failure-atomic rules; inputs and output are owned by the caller. This
 * CPU call does not enqueue device work or claim device/stream compatibility.
 */
spin8_status spin8_query_v3_controller_spinors(
    const spin8_query_v3_controller *controller, spin8_dtype dtype,
    const void *src, size_t src_capacity_bytes, size_t src_stride, double *dst,
    size_t dst_capacity_bytes, size_t dst_stride, size_t rows);

typedef struct {
  double *dq;
  size_t dq_bytes;
  double *dwin;
  size_t dwin_bytes;
  double *dbin;
  size_t dbin_bytes;
  double *dwout;
  size_t dwout_bytes;
  double *dbout;
  size_t dbout_bytes;
} spin8_query_v3_gradients_f64;

/* Analytic VJP of one F64 head row: dy is the adjoint of apply's output.
 * q is already model-normalized; dq includes the controller's dependence on
 * every q component, including the copied tail. No additional RMS is applied.
 * Required gradient sizes are width, 8*width, 8, 56 and 7 doubles,
 * respectively; weights use the same row-major layout as create. All buffers
 * are caller-owned. On success gradients are overwritten, never accumulated; on
 * any failure all remain unchanged. Accessed output spans must be disjoint from
 * each other, q, dy, this descriptor and controller storage. Capacities count
 * bytes. Every input and intermediate is validated even at identity
 * initialization; unlike apply's identity shortcut this can reject an
 * overflowing Win*q. Zero Wout/bout still yields nonzero dWout when the loss
 * has such a direction. ALLOCATION_FAILED reports staging allocation failure;
 * arithmetic overflow reports NONFINITE. No optimizer state or C++ exceptions
 * cross this CPU API. A(c(q)) is orthogonal, but its query-dependent Jacobian
 * need not be. */
spin8_status spin8_query_v3_controller_vjp_f64(
    const spin8_query_v3_controller *controller, const double *q,
    size_t q_bytes, const double *dy, size_t dy_bytes,
    const spin8_query_v3_gradients_f64 *gradients);

#ifdef __cplusplus
}
#endif
#endif
