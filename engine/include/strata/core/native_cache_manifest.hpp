#pragma once
#include "strata/core/native_cache_runtime.hpp"
#include "strata/core/geometry.hpp"
#include "strata/kernels/rope_scaling.hpp"
namespace strata::core {
struct NativeManifestInputs {
  std::vector<std::string> sources;
  std::string pack, k, v;
  ModelGeometry geometry;
  strata::kernels::RopeScaling rope;
  uint64_t capacity = 0, rows = 0;
  bool explicit_off = false;
};
// Expected digest comes from the trusted producer/caller, never the capsule.
bool native_manifest_load(const std::string &path, const std::string &expected_sha256,
                          const NativeManifestInputs &, NativeSessionConfig &,
                          std::string &error);
} // namespace strata::core
