#ifndef SPIN8_QUERY_ADAPTER_H
#define SPIN8_QUERY_ADAPTER_H
#include "spin8/spin8.h"
#ifdef __cplusplus
extern "C" {
#endif

#define SPIN8_QUERY_ADAPTER_ABI_VERSION 1u
#define SPIN8_QUERY_ADAPTER_MAX_BYTES 67108864u
#define SPIN8_QUERY_ADAPTER_MAX_LAYERS 4096u

typedef struct spin8_query_adapter spin8_query_adapter;
typedef struct {
  uint32_t layer_index, q_heads, kv_heads, head_width;
} spin8_query_adapter_layer_spec;

/* Expectations MUST come from the selected model files, codec/profile and
 * position settings, independently of the sidecar. Digests are SHA-256;
 * model_digests is an ORDERED contiguous array of 1..256 digests. Layer specs
 * are sorted, unique and exact (no missing or extra selected layer).
 * All byte capacities include the full accessed span. reserved must be zero.
 */
typedef struct {
  uint32_t abi_version, struct_bytes, model_count, layer_count;
  const uint8_t *model_digests;
  size_t model_digests_bytes;
  uint8_t profile_digest[32], position_digest[32];
  const spin8_query_adapter_layer_spec *layers;
  size_t layers_bytes;
  uint64_t reserved;
} spin8_query_adapter_expectations;

typedef struct {
  uint32_t abi_version, struct_bytes, model_count, layer_count;
  const uint8_t *model_digests;
  size_t model_digests_bytes;
  uint8_t profile_digest[32], position_digest[32];
  uint8_t convention_digest[32], integrity_digest[32], content_digest[32];
} spin8_query_adapter_info;

typedef struct {
  uint32_t abi_version, struct_bytes;
  spin8_query_adapter_layer_spec spec;
  const float *win, *bin, *wout, *bout;
  size_t win_bytes, bin_bytes, wout_bytes, bout_bytes;
} spin8_query_adapter_layer_view;

/* Memory-only, CPU-only v1 container loader for query-v3 (not old controllers).
 * Input bytes may be unaligned. Descriptors/outputs require native alignment.
 * Copies finite IEEE F32 parameters into owned float-aligned immutable storage.
 * Caller provides valid accessible spans, stable for the duration of the call.
 * Before writing out, all declared nonnull input spans must have representable
 * address ranges and be disjoint from out. Otherwise return INVALID_ARGUMENT
 * without writing out or inputs. A nonnull zero-capacity input protects its
 * base byte for this check. After these checks, validation failures set out
 * null. Invalid/aliased out remains untouched. No exceptions cross the ABI.
 * INVALID_ARGUMENT: bad spans, descriptor, geometry or canonical layout;
 * UNSUPPORTED: unknown wire/ABI version or mathematical conventions/flags;
 * AUDIT_FAILED: checksum or independently expected binding/geometry mismatch;
 * NONFINITE: NaN/Inf parameter; ALLOCATION_FAILED: owned storage unavailable.
 */
spin8_status
spin8_query_adapter_load(const void *bytes, size_t byte_count,
                         const spin8_query_adapter_expectations *expected,
                         size_t expected_bytes, spin8_query_adapter **out);
void spin8_query_adapter_free(spin8_query_adapter *adapter);

/* Getters leave output unchanged on failure. ordinal indexes the canonical
 * directory, not the model layer number. Borrowed pointers are read-only and
 * live until free. Readers may share an adapter; caller serializes destruction.
 * Output spans must be disjoint from the adapter and all borrowed storage.
 */
spin8_status spin8_query_adapter_get_info(const spin8_query_adapter *adapter,
                                          spin8_query_adapter_info *out,
                                          size_t out_bytes);
spin8_status spin8_query_adapter_get_layer(const spin8_query_adapter *adapter,
                                           uint32_t ordinal,
                                           spin8_query_adapter_layer_view *out,
                                           size_t out_bytes);
#ifdef __cplusplus
}
#endif
#endif
