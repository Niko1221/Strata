#ifndef TQCODEC_CALIBRATE_H
#define TQCODEC_CALIBRATE_H
#include "tqcodec/tqcodec.h"
#ifdef __cplusplus
extern "C" {
#endif
typedef struct {
  double alpha[3];
  double normalized_objective;
  double error_ratio_to_uniform;
} tq_consensus_fit;
/* Calibration-only FP64 active-set fit. errors[sample][view][component] has
 * three candidate Attention-output error vectors per sample (e.g. first-order
 * K score effects PLUS the same shared-V error in every view). Minimizes the
 * mean squared norm of sum_r alpha_r errors_r subject to alpha>=0,sum=1.
 * A dimensionless ridge penalizes ||alpha-(1/3,1/3,1/3)||^2 after normalizing
 * the error Gram matrix by its trace. No heap allocation or query-dependent
 * coefficient selection. Replacing fitted coefficients invalidates residual
 * caches. Use calibration data, never final evaluation data, as input.
 * QJL's sign/norm depends on alpha: fit this linear objective with QJL off,
 * then rebuild residuals and evaluate any QJL candidate separately.
 * Finite arrays with huge values are scaled before products. The result's
 * objective is dimensionless; ratio excludes the ridge. Zero-error data uses
 * uniform weights and ratio=1. Outputs unchanged on failure; overlap rejected.
 * This first-order calibration objective does not guarantee task quality. */
tq_status tq_consensus_fit_f64(const double *errors, size_t capacity_bytes,
                               size_t samples, size_t components, double ridge,
                               tq_consensus_fit *out);
#ifdef __cplusplus
}
#endif
#endif
