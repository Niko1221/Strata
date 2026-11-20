#pragma once
#include "spin8/query_adapter.h"
#include "tqcodec/cache_cuda.h"
#include <array>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>
struct spin8_query_v3_cuda_plan;
namespace strata::core {
using NativeCacheDigest = std::array<uint8_t, 32>;
// These are KV dialects, never GGML weight type numbers.
enum class NativeKvFormat : uint32_t {
  Turbo2 = 1,
  Turbo3,
  Turbo4,
  Turbo5,
  Turbo6
};
enum class NativeCacheRepresentation : uint32_t {
  Compressed = 0,
  DiagnosticF32 = 1
};
enum class NativeCachePrecision : uint32_t { F32 = 1, F16 = 2, BF16 = 3 };
enum class NativeCacheMode : uint32_t {
  Resident = 1,
  Ring,
  Streaming,
  Elastic,
  Split,
  Parallel,
  ExternalPersistence
};
struct NativeCacheRequested {
  uint32_t version = 1;
  NativeKvFormat k = NativeKvFormat::Turbo4, v = NativeKvFormat::Turbo2;
  NativeCacheRepresentation representation =
      NativeCacheRepresentation::Compressed;
  NativeCachePrecision precision = NativeCachePrecision::F32;
  NativeCacheMode mode = NativeCacheMode::Resident;
  uint32_t view_mask = 7;
  std::array<double, 3> alpha{}; // Caller must explicitly supply a simplex.
  double beta = 0;
  bool position_transport = false;
  uint64_t capacity = 0, max_queries = 0, max_append_rows = 0;
  std::vector<tqc_cache_cuda_plane> planes;
};
struct NativeCacheBinding {
  // INDEPENDENTLY VERIFIED caller inputs. This API does not verify files,
  // receipts, source selection or loader geometry. Sidecars supply no expected
  // identity. Synthetic unit identities are not evidence of a bound model.
  std::vector<NativeCacheDigest> raw_models;
  std::vector<spin8_query_adapter_layer_spec>
      normal_layers;           // sorted/unique/exact
  uint32_t selected_layer = 0; // logical model index, never a directory ordinal
  NativeCacheDigest position{};
};
struct NativeCacheOwnerLayout {
  tqc_cache_cuda_layout cache{};
  size_t total_bytes = 0;
  std::array<size_t, 4> parameter_offsets{}, parameter_bytes{};
  size_t query_status = 0, bank_alignment_bytes = 0, host_parameter_bytes = 0;
};
class NativeCacheOwner;
class NativeCachePlan final {
public:
  // CPU only: no current device, stream, CUDA allocation or model IO.
  // Base profile excludes adapter content, geometry and execution limits.
  static bool base_profile(const NativeCacheRequested &,
                           const NativeCacheDigest &position,
                           NativeCacheDigest &, std::string &error);
  static std::shared_ptr<const NativeCachePlan>
  bind(const NativeCacheRequested &, const NativeCacheBinding &,
       const void *sidecar, size_t sidecar_bytes, std::string &error);
  ~NativeCachePlan();
  NativeCachePlan(const NativeCachePlan &) = delete;
  NativeCachePlan &operator=(const NativeCachePlan &) = delete;
  const NativeCacheDigest &profile_digest() const;
  const NativeCacheDigest &layer_digest() const;
  const NativeCacheDigest &adapter_digest() const;
  const NativeCacheOwnerLayout &layout() const;
  const spin8_query_adapter_layer_spec &geometry() const;
  const tqc_cache_cuda_plan *cache_plan() const;
  uint64_t capacity() const;
  uint64_t max_queries() const;
  uint64_t max_append_rows() const;
  bool has_adapter() const;

private:
  struct Impl;
  explicit NativeCachePlan(std::unique_ptr<Impl>);
  std::unique_ptr<Impl> impl_;
  friend class NativeCacheOwner;
  spin8_query_v3_cuda_plan *query_plan() const;
  spin8_query_adapter_layer_view parameters() const;
};
} // namespace strata::core