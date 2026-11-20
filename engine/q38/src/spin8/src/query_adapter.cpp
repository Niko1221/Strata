#include "spin8/query_adapter.h"

#include <array>
#include <cstring>
#include <limits>
#include <memory>
#include <new>
#include <vector>

extern "C" {
#include "../third_party/sha256/sha256.h"
}

struct spin8_query_adapter {
  struct Layer {
    spin8_query_adapter_layer_spec spec;
    size_t offset;
  };
  spin8_query_adapter_info info{};
  std::vector<uint8_t> models;
  std::vector<Layer> layers;
  std::vector<float> values;
};

namespace {
constexpr size_t header_bytes = 160, digest_bytes = 32, record_bytes = 32;
constexpr uint32_t max_models = 256;
constexpr unsigned char magic[8] = {'S', 'P', '8', 'Q', 'V', '3', '\r', '\n'};
constexpr char contract[] =
    "spin8-query-v3;rank=8;chart=rational-tau-tan-pi/8;"
    "basis=positive-chirality-Gamma-C-J-v1;"
    "map=C(c)C(e0)^T;placement=post-Qnorm-pre-RoPE;"
    "mlp=Wout*tanh(Win*q+bin)+bout;no-extra-RMS;tail=identity";

bool pointer(const void *p, size_t n, size_t alignment) {
  const auto address = reinterpret_cast<uintptr_t>(p);
  return p && address % alignment == 0 && n <= UINTPTR_MAX - address;
}
bool overlap(const void *a, size_t an, const void *b, size_t bn) {
  const auto x = reinterpret_cast<uintptr_t>(a);
  const auto y = reinterpret_cast<uintptr_t>(b);
  return x < y + bn && y < x + an;
}
bool disjoint_input(const void *result, size_t result_bytes, const void *input,
                    size_t input_bytes) {
  if (!input)
    return true;
  // A malformed span must not disable alias protection. If its endpoint cannot
  // be represented, disjointness is unknown: reject before any output write.
  // Even a zero-capacity nonnull input protects its base byte from output.
  const size_t checked_bytes = input_bytes ? input_bytes : 1;
  return pointer(input, checked_bytes, 1) &&
         !overlap(result, result_bytes, input, checked_bytes);
}
uint32_t u32(const uint8_t *p) {
  return uint32_t(p[0]) | (uint32_t(p[1]) << 8) | (uint32_t(p[2]) << 16) |
         (uint32_t(p[3]) << 24);
}
uint64_t u64(const uint8_t *p) {
  return uint64_t(u32(p)) | (uint64_t(u32(p + 4)) << 32);
}
bool nonzero(const uint8_t *p) {
  uint8_t combined = 0;
  for (size_t i = 0; i < digest_bytes; ++i)
    combined |= p[i];
  return combined != 0;
}
bool geometry(const spin8_query_adapter_layer_spec &s) {
  return s.layer_index <= 1048575 && s.q_heads && s.q_heads <= 4096 &&
         s.kv_heads && s.kv_heads <= 4096 && s.q_heads % s.kv_heads == 0 &&
         s.head_width && s.head_width <= 16384;
}
bool equal(const spin8_query_adapter_layer_spec &a,
           const spin8_query_adapter_layer_spec &b) {
  return a.layer_index == b.layer_index && a.q_heads == b.q_heads &&
         a.kv_heads == b.kv_heads && a.head_width == b.head_width;
}
// geometry() establishes strict bounds before these multiplications. uint64_t
// retains the largest permitted layer even when size_t is only 32 bits.
uint64_t payload_bytes(const spin8_query_adapter_layer_spec &s) {
  return uint64_t(s.q_heads) * (uint64_t(8) * s.head_width + 71) * 4;
}
bool output_ok(const spin8_query_adapter *a, const void *p, size_t n,
               size_t alignment) {
  return pointer(p, n, alignment) && !overlap(p, n, a, sizeof(*a)) &&
         !overlap(p, n, a->models.data(), a->models.size()) &&
         !overlap(p, n, a->layers.data(),
                  a->layers.size() * sizeof(a->layers[0])) &&
         !overlap(p, n, a->values.data(), a->values.size() * sizeof(float));
}
spin8_status load_impl(const uint8_t *data, size_t n,
                       const spin8_query_adapter_expectations &e,
                       spin8_query_adapter **out) {
  if (n < header_bytes + digest_bytes || n > SPIN8_QUERY_ADAPTER_MAX_BYTES)
    return SPIN8_INVALID_ARGUMENT;
  if (std::memcmp(data, magic, sizeof(magic)) != 0)
    return SPIN8_INVALID_ARGUMENT;
  if (u32(data + 8) != 1 || u32(data + 16) != 0 || u32(data + 40) != 3 ||
      u32(data + 44) != 8 || u32(data + 48) != 1 || u32(data + 52) != 1 ||
      u32(data + 56) != 1 || u32(data + 60) != 0x3ec90fdb)
    return SPIN8_UNSUPPORTED;
  const uint32_t models = u32(data + 20), count = u32(data + 24);
  if (u32(data + 12) != header_bytes || u32(data + 28) != record_bytes ||
      u64(data + 32) != n || !models || models > max_models || !count ||
      count > SPIN8_QUERY_ADAPTER_MAX_LAYERS)
    return SPIN8_INVALID_ARGUMENT;
  const size_t directory = header_bytes + size_t(models) * digest_bytes;
  size_t cursor = directory + size_t(count) * record_bytes;
  if (cursor > n - digest_bytes)
    return SPIN8_INVALID_ARGUMENT;
  std::array<uint8_t, digest_bytes> checksum{}, math{};
  spin8_sidecar_sha256_hash(checksum.data(), data, n - digest_bytes);
  if (std::memcmp(checksum.data(), data + n - digest_bytes, digest_bytes) != 0)
    return SPIN8_AUDIT_FAILED;
  spin8_sidecar_sha256_hash(math.data(),
                            reinterpret_cast<const unsigned char *>(contract),
                            sizeof(contract) - 1);
  if (models != e.model_count || count != e.layer_count ||
      std::memcmp(data + 64, e.profile_digest, digest_bytes) != 0 ||
      std::memcmp(data + 96, e.position_digest, digest_bytes) != 0 ||
      std::memcmp(data + 128, math.data(), digest_bytes) != 0 ||
      std::memcmp(data + header_bytes, e.model_digests,
                  models * digest_bytes) != 0)
    return SPIN8_AUDIT_FAILED;
  // Validate every record and scalar before allocating owned payload storage.
  const size_t first_payload = cursor;
  for (uint32_t i = 0; i < count; ++i) {
    const auto *p = data + directory + size_t(i) * record_bytes;
    const spin8_query_adapter_layer_spec spec{u32(p), u32(p + 4), u32(p + 8),
                                              u32(p + 12)};
    if (!geometry(spec))
      return SPIN8_INVALID_ARGUMENT;
    if (!equal(spec, e.layers[i]))
      return SPIN8_AUDIT_FAILED;
    const uint64_t bytes = payload_bytes(spec);
    if (u64(p + 16) != cursor || u64(p + 24) != bytes ||
        bytes > n - digest_bytes - cursor)
      return SPIN8_INVALID_ARGUMENT;
    cursor += static_cast<size_t>(bytes);
  }
  if (cursor != n - digest_bytes)
    return SPIN8_INVALID_ARGUMENT;
  for (size_t offset = first_payload; offset < cursor; offset += 4)
    if ((u32(data + offset) & 0x7f800000) == 0x7f800000)
      return SPIN8_NONFINITE;
  auto adapter = std::make_unique<spin8_query_adapter>();
  adapter->models.assign(data + header_bytes, data + directory);
  adapter->layers.reserve(count);
  adapter->values.resize((cursor - first_payload) / 4);
  for (size_t i = 0; i < adapter->values.size(); ++i) {
    const auto bits = u32(data + first_payload + i * 4);
    std::memcpy(&adapter->values[i], &bits, sizeof(bits));
  }
  for (uint32_t i = 0; i < count; ++i) {
    const auto *p = data + directory + size_t(i) * record_bytes;
    adapter->layers.push_back(
        {e.layers[i], (static_cast<size_t>(u64(p + 16)) - first_payload) / 4});
  }
  auto &info = adapter->info;
  info.abi_version = SPIN8_QUERY_ADAPTER_ABI_VERSION;
  info.struct_bytes = sizeof(info);
  info.model_count = models;
  info.layer_count = count;
  info.model_digests = adapter->models.data();
  info.model_digests_bytes = adapter->models.size();
  std::memcpy(info.profile_digest, e.profile_digest, digest_bytes);
  std::memcpy(info.position_digest, e.position_digest, digest_bytes);
  std::memcpy(info.convention_digest, math.data(), digest_bytes);
  std::memcpy(info.integrity_digest, checksum.data(), digest_bytes);
  spin8_sidecar_sha256_hash(info.content_digest, data, n);
  *out = adapter.release();
  return SPIN8_OK;
}
} // namespace

extern "C" spin8_status
spin8_query_adapter_load(const void *bytes, size_t n,
                         const spin8_query_adapter_expectations *expected,
                         size_t expected_bytes, spin8_query_adapter **out) {
  static_assert(sizeof(float) == 4 && std::numeric_limits<float>::is_iec559,
                "sidecar requires IEEE binary32 float");
  const void *result_address = static_cast<const void *>(out);
  if (!pointer(result_address, sizeof(*out), alignof(spin8_query_adapter *)))
    return SPIN8_INVALID_ARGUMENT;
  // No output write until every declared input range is representable and
  // disjoint. In particular SIZE_MAX/zero capacities cannot bypass this guard.
  if (!disjoint_input(result_address, sizeof(*out), bytes, n) ||
      !disjoint_input(result_address, sizeof(*out), expected, expected_bytes))
    return SPIN8_INVALID_ARGUMENT;
  if (!pointer(expected, sizeof(*expected),
               alignof(spin8_query_adapter_expectations)) ||
      expected_bytes != sizeof(*expected)) {
    *out = nullptr;
    return SPIN8_INVALID_ARGUMENT;
  }
  const auto &e = *expected;
  if (!disjoint_input(result_address, sizeof(*out), e.model_digests,
                      e.model_digests_bytes) ||
      !disjoint_input(result_address, sizeof(*out), e.layers, e.layers_bytes))
    return SPIN8_INVALID_ARGUMENT;
  const bool model_span =
      e.model_count && e.model_count <= max_models &&
      e.model_digests_bytes >= size_t(e.model_count) * digest_bytes &&
      pointer(e.model_digests, size_t(e.model_count) * digest_bytes, 1);
  const bool layer_span =
      e.layer_count && e.layer_count <= SPIN8_QUERY_ADAPTER_MAX_LAYERS &&
      e.layers_bytes >= size_t(e.layer_count) * sizeof(*e.layers) &&
      pointer(e.layers, size_t(e.layer_count) * sizeof(*e.layers),
              alignof(spin8_query_adapter_layer_spec));
  *out = nullptr;
  if (e.abi_version != SPIN8_QUERY_ADAPTER_ABI_VERSION)
    return SPIN8_UNSUPPORTED;
  if (e.struct_bytes != sizeof(e) || e.reserved || !model_span || !layer_span ||
      !pointer(bytes, n, 1) || !nonzero(e.profile_digest) ||
      !nonzero(e.position_digest))
    return SPIN8_INVALID_ARGUMENT;
  for (uint32_t i = 0; i < e.model_count; ++i)
    if (!nonzero(e.model_digests + size_t(i) * digest_bytes))
      return SPIN8_INVALID_ARGUMENT;
  for (uint32_t i = 0; i < e.layer_count; ++i)
    if (!geometry(e.layers[i]) ||
        (i && e.layers[i - 1].layer_index >= e.layers[i].layer_index))
      return SPIN8_INVALID_ARGUMENT;
  try {
    return load_impl(static_cast<const uint8_t *>(bytes), n, e, out);
  } catch (const std::bad_alloc &) {
    return SPIN8_ALLOCATION_FAILED;
  } catch (...) {
    return SPIN8_INVALID_ARGUMENT;
  }
}

extern "C" void spin8_query_adapter_free(spin8_query_adapter *a) { delete a; }

extern "C" spin8_status
spin8_query_adapter_get_info(const spin8_query_adapter *a,
                             spin8_query_adapter_info *out, size_t n) {
  if (!a || n < sizeof(*out) ||
      !output_ok(a, out, sizeof(*out), alignof(spin8_query_adapter_info)))
    return SPIN8_INVALID_ARGUMENT;
  *out = a->info;
  return SPIN8_OK;
}

extern "C" spin8_status
spin8_query_adapter_get_layer(const spin8_query_adapter *a, uint32_t ordinal,
                              spin8_query_adapter_layer_view *out, size_t n) {
  if (!a || ordinal >= a->layers.size() || n < sizeof(*out) ||
      !output_ok(a, out, sizeof(*out), alignof(spin8_query_adapter_layer_view)))
    return SPIN8_INVALID_ARGUMENT;
  const auto &layer = a->layers[ordinal];
  const size_t heads = layer.spec.q_heads;
  spin8_query_adapter_layer_view value{};
  value.abi_version = SPIN8_QUERY_ADAPTER_ABI_VERSION;
  value.struct_bytes = sizeof(value);
  value.spec = layer.spec;
  value.win = a->values.data() + layer.offset;
  const size_t win_scalars = heads * 8 * layer.spec.head_width;
  value.win_bytes = win_scalars * sizeof(float);
  value.bin = value.win + win_scalars;
  value.bin_bytes = heads * 8 * sizeof(float);
  value.wout = value.bin + heads * 8;
  value.wout_bytes = heads * 56 * sizeof(float);
  value.bout = value.wout + heads * 56;
  value.bout_bytes = heads * 7 * sizeof(float);
  *out = value;
  return SPIN8_OK;
}
