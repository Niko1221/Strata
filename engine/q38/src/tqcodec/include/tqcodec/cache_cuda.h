#ifndef TQCODEC_CACHE_CUDA_H
#define TQCODEC_CACHE_CUDA_H
#include "dialects_cuda.h"
#ifdef __cplusplus
extern "C" {
#endif
/* v1 independent cache: F32, sm120, 128/256 columns. K=PACK(B PREPARE(k)),
 * Q=B F(q), shared V=PACK(PREPARE(v)), output F^-1. Same strict Spin element
 * supplies all three repeated 8x8 B matrices. Diagnostic F32 omits
 * quantisation. Caller owns a 256-aligned GPU arena/session and external
 * tensors. Explicit nondefault stream, selected device, no per-call
 * allocation/sync/device switch. Host OK=enqueued; consume only after
 * successful completion AND status=0 AND valid=1. Any device error invalidates
 * ALL views/output. Host rejection changes nothing. Runtime errors invalidate
 * results even if status cannot be written. One ordered stream owns each
 * session; caller supplies event dependencies on transfer. GPU buffers survive
 * graph replay. Descriptors are copied at launch; persistent command
 * counts/position/indices may change between ordered replays.
 */
enum {
  TQC_CC_NUMERICAL = 1,
  TQC_CC_CAPACITY = 2,
  TQC_CC_INVALID_STATE = 4,
  TQC_CC_EMPTY = 8,
  TQC_CC_ALL_MASKED = 16,
  TQC_CC_TRANSFORM_INVARIANT_VIOLATION = 32,
  TQC_CC_INDEX = 64
};
enum { TQC_CC_COMPRESSED = 0, TQC_CC_DIAGNOSTIC_F32 = 1 };
enum {
  TQC_CC_COMMIT = 1,
  TQC_CC_CANCEL = 2,
  TQC_CC_TRUNCATE = 3,
  TQC_CC_INVALIDATE = 4
};
typedef struct tqc_cache_cuda_plan tqc_cache_cuda_plan;
typedef struct tqc_cache_cuda_plane {
  uint32_t a, b;
  double angle;
} tqc_cache_cuda_plane;
typedef struct tqc_cache_cuda_config {
  uint32_t version, size, precision, representation, k_dialect, v_dialect,
      view_mask;
  uint32_t columns, kv_heads, q_heads;
  uint64_t capacity, max_queries, max_append_rows;
  double alpha[3],
      beta; /* explicit simplex; uniform default: 1/3 each (single: 1) */
  uint8_t model[32], adapter[32],
      position[32]; /* required caller SHA256 identities */
  const tqc_cache_cuda_plane *planes;
  size_t plane_count; /* <=64 */
} tqc_cache_cuda_config;
typedef struct tqc_cache_cuda_layout {
  uint32_t version, size;
  size_t total_bytes, state, command, identity_offset, matrices, audit,
      positions, k[3], v;
  size_t k_bytes[3], v_bytes, k_row_bytes, v_row_bytes;
  size_t metadata_bytes, alignment_bytes, host_plan_bytes, constant_bytes;
  uint8_t identity[32], matrix_identity[3][32];
} tqc_cache_cuda_layout;
typedef struct tqc_cache_cuda_state {
  uint64_t committed, tentative;
  uint32_t status, valid;
} tqc_cache_cuda_state;
typedef struct tqc_cache_cuda_command {
  uint64_t append_rows, query_rows, target_rows;
} tqc_cache_cuda_command;
typedef struct tqc_cache_cuda_audit {
  /* B metrics are per repeated 8x8 block; F metrics are full 128x128. */
  double ef[4], gram[4], logabsdet[4], drift[4];
  int32_t det_sign[4];
  uint32_t finite[4];
} tqc_cache_cuda_audit;
typedef struct tqc_cache_cuda_context {
  uint32_t version, size;
  const tqc_cache_cuda_plan *plan;
  void *arena;
  size_t arena_bytes;
  int32_t device_ordinal;
  void *stream;
} tqc_cache_cuda_context;
typedef struct tqc_cache_cuda_append {
  uint32_t version, size;
  /* k/v flatten [token,kv_head], byte strides. Buffers cover max_append_rows
   * tokens. positions is max_append_rows int64 rows. All spans must be
   * disjoint. */
  tqc_dialect_v1_span k, v, positions;
} tqc_cache_cuda_append;
typedef struct tqc_cache_cuda_attention {
  uint32_t version, size, causal, reserved;
  float scale;
  /* q/out flatten [query,q_head], positions max_queries int64 rows.
   * optional mask u8/bias f32: [query*q_head,capacity].
   * optional indices int64: [query,selection_width], -1 skips. Duplicates fail.
   * Empty/all-masked are typed errors invalidating cache/output, not zero
   * success. All spans cover maximum counts and are disjoint, including the
   * arena. */
  tqc_dialect_v1_span q, out, positions, mask, bias, indices;
  size_t selection_width;
} tqc_cache_cuda_attention;
/* Only host construction allocates. All identities include all three matrices.
 * Unsupported precision, position transport and QJL beta>0 fail UNSUPPORTED. */
/* Capability-false fields are intentional: no approximation or silent fallback.
 */
typedef struct tqc_cache_cuda_capabilities {
  uint32_t version, size, position_transport, shared_qjl, coefficient_f16,
      coefficient_bf16;
} tqc_cache_cuda_capabilities;
int tqc_cache_cuda_query(tqc_cache_cuda_capabilities *);
/* Per-kernel compiler/runtime resources in order: audit,append,finish,control,
 * attend,copy. Local bytes are per CUDA thread, shared bytes include static
 * plus required dynamic shared memory (including fixed audit F matrix), and
 * constant bytes are static; registers are per thread. Driver
 * module, context/allocator reservations and external tensors are additional.
 * This is a quiescent metadata query, not a total GPU-peak measurement. */
typedef struct tqc_cache_cuda_resources {
  uint32_t version, size;
  size_t local_bytes[6], shared_bytes[6], constant_bytes[6];
  int32_t registers[6], max_threads[6];
} tqc_cache_cuda_resources;
int tqc_cache_cuda_resources_query(const tqc_cache_cuda_context *,
                                   tqc_cache_cuda_resources *);
int tqc_cache_cuda_create(const tqc_cache_cuda_config *,
                          tqc_cache_cuda_plan **);
void tqc_cache_cuda_free(tqc_cache_cuda_plan *);
int tqc_cache_cuda_get_layout(const tqc_cache_cuda_plan *,
                              tqc_cache_cuda_layout *);
/* Quiescent initialise/reset synchronises the supplied stream and audits actual
 * consumed coefficients. No capture. Clears all history and errors. */
int tqc_cache_cuda_initialize(const tqc_cache_cuda_context *);
int tqc_cache_cuda_append_rows(const tqc_cache_cuda_context *,
                               const tqc_cache_cuda_append *);
int tqc_cache_cuda_attend(const tqc_cache_cuda_context *,
                          const tqc_cache_cuda_attention *);
/* Additive bounded calls use the existing v1 descriptor. call_row_capacity is
 * positive and <= the corresponding plan maximum. Every external span covers
 * this capacity (tokens, not flattened heads). GPU command counts above it
 * invalidate BEFORE any tensor read/publication. Layout/identity and legacy
 * calls are unchanged; device buffers survive completion and graph replay. */
int tqc_cache_cuda_append_rows_bounded(const tqc_cache_cuda_context *,
                                       const tqc_cache_cuda_append *,
                                       size_t call_row_capacity);
int tqc_cache_cuda_attend_bounded(const tqc_cache_cuda_context *,
                                  const tqc_cache_cuda_attention *,
                                  size_t call_row_capacity);
/* Attention reads tentative rows for speculative verification. Commit exposes
 * all successful rows. Cancel drops tentative; truncate uses target<=committed.
 * These operations never recover invalid state; initialise is required. */
int tqc_cache_cuda_control(const tqc_cache_cuda_context *, uint32_t operation);
int tqc_cache_cuda_audit_device(const tqc_cache_cuda_context *);
/* Physical copy appends selected committed rows to DISTINCT identical cache.
 * Both context streams must be the same, after caller dependencies.
 * indices: capacity int64 rows, command.append_rows selects length. Duplicates,
 * range errors and any arena overlap reject. clone copies all committed rows to
 * empty destination. No positional-coordinate change is implemented. */
int tqc_cache_cuda_copy(const tqc_cache_cuda_context *,
                        const tqc_cache_cuda_context *,
                        tqc_dialect_v1_span indices, int clone);
int tqc_cache_cuda_position_transform(const tqc_cache_cuda_context *);
/* Explicit synchronous bounded host snapshots, only committed quiescent state.
 * Storage exact returned size; no file IO. Complete integrity/identity checked
 * before device mutation. Host staging may allocate <=snapshot bytes. Caller
 * protects this private model-derived data and keeps it immutable during calls.
 */
int tqc_cache_cuda_snapshot_size(const tqc_cache_cuda_plan *, size_t *);
int tqc_cache_cuda_snapshot(const tqc_cache_cuda_context *, void *, size_t);
int tqc_cache_cuda_restore(const tqc_cache_cuda_context *, const void *,
                           size_t);
#ifdef __cplusplus
}
#endif
#endif
