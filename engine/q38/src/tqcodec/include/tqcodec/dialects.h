#ifndef TQCODEC_DIALECTS_H
#define TQCODEC_DIALECTS_H
#include <stddef.h>
#include <stdint.h>
#ifdef __cplusplus
extern "C" {
#endif

/* Additive ABI, unrelated to tqcodec's generic FP64 reference packing.
 * All structures require version=1 and size=sizeof(struct). No ownership
 * transfer. CPU IEEE binary32, round-to-nearest, gradual underflow, masked FP
 * exceptions, little-endian payloads. Input magnitudes <=65504. Original
 * encoders can emit infinite FP16 scales; this API returns RANGE instead. Tiny
 * linear-format blocks with an overflowing reciprocal are rejected (RANGE). No
 * overlap/in-place, including descriptors and receipt versus writable spans.
 * Caller keeps descriptors and buffers alive and unchanged for the whole call.
 * Failures leave ALL outputs, including the receipt, unchanged. No allocation.
 */
enum tqc_dialect_v1_id {
  TQC_THETOM_64395689_TURBO2 = 1,
  TQC_THETOM_64395689_TURBO3,
  TQC_THETOM_64395689_TURBO4,
  TQC_THETOM_64395689_TURBO5,
  TQC_THETOM_64395689_TURBO6,
  TQC_THETOM_64395689_TQ3_1S,
  TQC_THETOM_64395689_TQ4_1S,
  TQC_PRISM_73BFE6CC_PQ2_0,
  TQC_PRISM_73BFE6CC_PTQ1_0,
  TQC_PRISM_73BFE6CC_Q2_0,
  TQC_STRATA_3CF03257_Q2_0
};
enum tqc_dialect_v1_status {
  TQC_D_OK = 0,
  TQC_D_INVALID,
  TQC_D_VERSION,
  TQC_D_UNSUPPORTED,
  TQC_D_OVERFLOW,
  TQC_D_CAPACITY,
  TQC_D_ALIGNMENT,
  TQC_D_OVERLAP,
  TQC_D_NONFINITE,
  TQC_D_RANGE,
  TQC_D_AMBIGUOUS
};
enum tqc_dialect_v1_operation {
  TQC_D_ENCODE = 1,
  TQC_D_DECODE,
  /* Turbo only: output F(x/||x||), and original F32 L2 norms in norms.
   * Threshold/zero handling matches the original format wrapper. */
  TQC_D_PREPARE,
  /* Turbo only: input normalized STORED coordinates, norms contains the
   * original nonnegative finite F32 L2 norm, one per 128 values. No F or
   * renormalization here. Unit length is caller's mathematical responsibility;
   * arbitrary finite coordinates are permitted for scalar centroid probes.
   * Caller may insert B between PREPARE and PACK: U=B F, never F B.
   * Stored norm = original_norm / ||selected_centroids||, rounded to FP16. */
  TQC_D_PACK,
  /* Turbo F and F^-1 only; no normalization or quantization. */
  TQC_D_FORWARD,
  TQC_D_INVERSE
};
enum { TQC_D_CPU = 0, TQC_D_CUDA = 1, TQC_D_F32 = 1 };
enum { TQC_D_KV = 1, TQC_D_WEIGHT = 2, TQC_D_LINEAR = 3 };
typedef struct tqc_dialect_v1_span {
  void *data;
  size_t capacity_bytes;
  size_t row_stride_bytes;
} tqc_dialect_v1_span;
typedef struct tqc_dialect_v1_request {
  uint32_t version, size;
  uint32_t dialect, operation, device, precision;
  void *stream;         /* must be NULL on CPU */
  size_t rows, columns; /* positive; columns multiple of format block width */
  tqc_dialect_v1_span src, dst, norms;
} tqc_dialect_v1_request;
typedef struct tqc_dialect_v1_receipt {
  uint32_t version, size;
  size_t rows, blocks, payload_bytes;
} tqc_dialect_v1_receipt;
typedef struct tqc_dialect_v1_info {
  uint32_t version, size, dialect, kind;
  uint32_t block_elements, block_bytes, metadata_bytes, scale_offset;
  uint32_t cpu_f32, cuda, scratch_bytes,
      reserved;         /* caller scratch=0; stack only */
  const char *identity; /* canonical format identity, not a GGML type number */
  const char *source_revision;
  const char *coordinates; /* factor ledger and reconstruction coordinates */
  const char *packing;
} tqc_dialect_v1_info;
/* Full identity + declared block layout required for later GGUF importers.
 * Type number, payload divisibility, or model name alone are insufficient.
 * identity_capacity bounds a NUL-terminated string. Unknown/old group128 Q2_0
 * and legacy 68-byte Turbo4 fail. No GGUF modification is performed. */
typedef struct tqc_dialect_v1_layout {
  uint32_t version, size, dialect, block_elements, block_bytes;
  const char *identity;
  size_t identity_capacity;
} tqc_dialect_v1_layout;
int tqc_dialect_v1_query(uint32_t dialect, tqc_dialect_v1_info *info);
int tqc_dialect_v1_validate_layout(const tqc_dialect_v1_layout *layout);
int tqc_dialect_v1_execute(const tqc_dialect_v1_request *request,
                           tqc_dialect_v1_receipt *receipt);
#ifdef __cplusplus
}
#endif
#endif
