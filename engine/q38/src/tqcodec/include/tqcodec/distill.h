#ifndef TQCODEC_DISTILL_H
#define TQCODEC_DISTILL_H
#include "tqcodec/tqcodec.h"
#ifdef __cplusplus
extern "C" {
#endif

/* CPU FP64, one query/head at a time. Keys/values are row major, with explicit
 * byte strides/capacities. The teacher is detached. No token-by-token square
 * attention matrix is stored. This is a reference loss, not model training. */
typedef struct {
  const double *data;
  size_t rows;
  size_t columns;
  size_t row_stride_bytes;
  size_t capacity_bytes;
} tq_f64_rows;
typedef struct {
  double normalized_output_mse;
  double attention_kl;
  double total;
} tq_distill_loss;

/* Workspace contains 2*value_width + query_width doubles. The caller can cap
 * captures and reuse this scratch across queries. Outputs and scratch must
 * not overlap any readable input, or each other. Outputs unchanged on error;
 * workspace may change. Padding is untouched. No hidden allocations. */
tq_status tq_distill_workspace_size(size_t query_width, size_t value_width,
                                    size_t *bytes);
/* logit_bias has N entries; NULL/zero capacity means no bias. -Inf masks a key;
 * NaN/+Inf and an entirely masked row are rejected. Scale and mask are shared
 * by teacher and student. MSE = ||o_s-o_t||^2/(epsilon+||o_t||^2),
 * KL = KL(p_t || p_s); dq is the derivative of their weighted sum with respect
 * to student_query. The derivative includes attention_scale. */
tq_status
tq_distill_head_f64(const double *teacher_query, size_t teacher_query_bytes,
                    const double *student_query, size_t student_query_bytes,
                    tq_f64_rows teacher_keys, tq_f64_rows student_keys,
                    tq_f64_rows teacher_values, tq_f64_rows student_values,
                    const double *logit_bias, size_t logit_bias_bytes,
                    double attention_scale, double mse_weight, double kl_weight,
                    double epsilon, void *workspace, size_t workspace_bytes,
                    tq_distill_loss *loss, double *student_query_gradient,
                    size_t gradient_bytes);
#ifdef __cplusplus
}
#endif
#endif
