#ifndef SPIN8_ATTENTION_V2_H
#define SPIN8_ATTENTION_V2_H
#include "spin8/spin8.h"
#ifdef __cplusplus
extern "C" {
#endif

#define SPIN8_ATTENTION_V2_ABI_VERSION 2u
#define SPIN8_ATTENTION_V2_SELF 1u
#define SPIN8_ATTENTION_V2_POST_POSITION 1u
#define SPIN8_ATTENTION_V2_CODEC_NONE 0u
#define SPIN8_ATTENTION_V2_IDENTITY 0u
#define SPIN8_ATTENTION_V2_FIXED 1u
#define SPIN8_ATTENTION_V2_SCALE_BIAS_SOFTCAP 0u
#define SPIN8_ATTENTION_V2_SCALE_SOFTCAP_BIAS 1u

typedef struct spin8_attention_v2_plan spin8_attention_v2_plan;

/* All strides/capacities are bytes. Tensor axes are [token,head,channel].
 * Mask/bias axes are [query,query-head,key]. Optional buffers are all-zero.
 * Descriptors need native alignment; tensor data may be unaligned. */
typedef struct {
  const void *data;
  size_t capacity_bytes;
  size_t token_stride_bytes, head_stride_bytes, channel_stride_bytes;
} spin8_attention_v2_input;
typedef struct {
  void *data;
  size_t capacity_bytes;
  size_t token_stride_bytes, head_stride_bytes, channel_stride_bytes;
} spin8_attention_v2_output;

typedef struct {
  uint32_t abi_version, struct_bytes;
  uint32_t semantics, dtype, codec, basis;
  uint32_t view, hadamard_block; /* fixed view 0..2; block 0 or 128 */
  /* Nonzero caller assertion of one self-attention source. */
  uint64_t source_id;
  size_t n_q_heads, n_kv_heads, d_qk, d_v;
  const uint32_t *q_to_kv;
  size_t q_to_kv_bytes;
  const spin8_plane *planes;
  size_t plane_count, planes_bytes;
  uint64_t reserved[4];
} spin8_attention_v2_config;

typedef struct {
  uint32_t abi_version, struct_bytes;
  uint32_t position_stage, logit_order;
  uint64_t q_source_id, kv_source_id;
  size_t n_query, n_key;
  /* Explicit finite scale; softcap 0 disables, else >0. */
  double scale, softcap;
  spin8_attention_v2_input q, k, v, mask, bias, gate;
  spin8_attention_v2_output output;
  uint64_t reserved[4];
} spin8_attention_v2_request;

/* Exact version/struct size; reserved fields zero. FP64, self-attention and
 * unencoded buffers only. Mapping and plane-derived audited plan are owned.
 * Identity requires view/block/plane fields zero. Fixed uses one common basis
 * for Q and K, including every query sharing a KV head. No sessions/multi-view.
 * On failure *out is unchanged. Caller supplies accessible, stable spans. */
spin8_status spin8_attention_v2_plan_create(const spin8_attention_v2_config *,
                                            size_t config_bytes,
                                            spin8_attention_v2_plan **out);
void spin8_attention_v2_plan_free(spin8_attention_v2_plan *);

/* Q/K are caller-positioned. Source IDs must equal the plan's nonzero ID.
 * Mask is required: zero excludes, any nonzero allows. Bias/gate optional and
 * finite, including masked entries. All-masked rows fail. One stable softmax.
 * Forward: optional normalized full H128 blocks, then complete Spin8 blocks;
 * incomplete tails are identity at each stage. Inverse V in reverse order,
 * then elementwise gate. Output is caller-owned and disjoint from all inputs
 * and descriptors. Failure leaves all output bytes unchanged. No exceptions
 * cross this ABI. Immutable plans allow concurrent calls with distinct outputs;
 * caller serializes destruction. See spin8/docs/attention_v2.md for limits. */
spin8_status spin8_attention_v2_execute(const spin8_attention_v2_plan *,
                                        const spin8_attention_v2_request *,
                                        size_t request_bytes);
#ifdef __cplusplus
}
#endif
#endif
