#pragma once
// Test-only independent canonical sidecar encoder. Identities and geometry are
// synthetic; no claim about model files or the native loader is made here.
#include "strata/core/native_cache_config.hpp"
#include <cstring>
#include <stdexcept>
extern "C" {
#include "../../spin8/third_party/sha256/sha256.h"
}
namespace native_fixture {
using namespace strata::core;
inline NativeCacheRequested request() {
  NativeCacheRequested r;
  r.alpha = {1. / 3, 1. / 3, 1. / 3};
  r.capacity = 8;
  r.max_queries = 3;
  r.max_append_rows = 3;
  r.planes = {{0, 1, .37}, {2, 5, -.61}};
  return r;
}
inline NativeCacheBinding binding() {
  NativeCacheBinding b;
  b.raw_models = {NativeCacheDigest{1}, NativeCacheDigest{2}};
  b.position[0] = 3;
  b.normal_layers = {{2, 4, 2, 128}, {7, 4, 2, 128}};
  b.selected_layer = 2;
  return b;
}
inline void word(std::vector<uint8_t> &v, size_t at, uint64_t x, size_t n = 4) {
  for (size_t i = 0; i < n; ++i)
    v.at(at + i) = uint8_t(x >> (8 * i));
}
inline NativeCacheDigest hash(const void *p, size_t n) {
  NativeCacheDigest d{};
  spin8_sidecar_sha256_hash(d.data(), static_cast<const unsigned char *>(p), n);
  return d;
}
inline std::string hex(const NativeCacheDigest &d) {
  const char *alphabet = "0123456789abcdef";
  std::string s;
  for (auto x : d) {
    s += alphabet[x >> 4];
    s += alphabet[x & 15];
  }
  return s;
}
inline std::vector<uint8_t> sidecar(const NativeCacheRequested &r,
                                    const NativeCacheBinding &b, bool nonzero) {
  NativeCacheDigest profile{};
  std::string error;
  if (!NativeCachePlan::base_profile(r, b.position, profile, error))
    throw std::runtime_error(error);
  const size_t prefix =
      160 + 32 * b.raw_models.size() + 32 * b.normal_layers.size();
  size_t total = prefix + 32;
  for (const auto &s : b.normal_layers)
    total += size_t(s.q_heads) * (8 * s.head_width + 71) * 4;
  std::vector<uint8_t> v(total, 0);
  std::memcpy(v.data(), "SP8QV3\r\n", 8);
  word(v, 8, 1);
  word(v, 12, 160);
  word(v, 20, b.raw_models.size());
  word(v, 24, b.normal_layers.size());
  word(v, 28, 32);
  word(v, 32, total, 8);
  word(v, 40, 3);
  word(v, 44, 8);
  word(v, 48, 1);
  word(v, 52, 1);
  word(v, 56, 1);
  word(v, 60, 0x3ec90fdb);
  std::memcpy(v.data() + 64, profile.data(), 32);
  std::memcpy(v.data() + 96, b.position.data(), 32);
  const char contract[] =
      "spin8-query-v3;rank=8;chart=rational-tau-tan-pi/8;"
      "basis=positive-chirality-Gamma-C-J-v1;map=C(c)C(e0)^T;"
      "placement=post-Qnorm-pre-RoPE;mlp=Wout*tanh(Win*q+bin)+bout;no-extra-"
      "RMS;tail=identity";
  auto convention = hash(contract, sizeof(contract) - 1);
  std::memcpy(v.data() + 128, convention.data(), 32);
  for (size_t i = 0; i < b.raw_models.size(); ++i)
    std::memcpy(v.data() + 160 + i * 32, b.raw_models[i].data(), 32);
  size_t cursor = prefix;
  for (size_t i = 0; i < b.normal_layers.size(); ++i) {
    const auto &s = b.normal_layers[i];
    const size_t d = 160 + b.raw_models.size() * 32 + i * 32;
    word(v, d, s.layer_index);
    word(v, d + 4, s.q_heads);
    word(v, d + 8, s.kv_heads);
    word(v, d + 12, s.head_width);
    word(v, d + 16, cursor, 8);
    const size_t counts[4] = {size_t(s.q_heads) * 8 * s.head_width,
                              size_t(s.q_heads) * 8, size_t(s.q_heads) * 56,
                              size_t(s.q_heads) * 7};
    const size_t bytes = size_t(s.q_heads) * (8 * s.head_width + 71) * 4;
    word(v, d + 24, bytes, 8);
    for (size_t section = 0; section < 4; ++section)
      for (size_t j = 0; j < counts[section]; ++j) {
        // Nonzero Win/bin even at identity; Wout/bout zero means exact
        // identity.
        const float value =
            section < 2 || nonzero
                ? float(int((j + i * 11 + section * 17) % 29) - 14) / 512.f
                : 0.f;
        uint32_t bits;
        std::memcpy(&bits, &value, 4);
        word(v, cursor, bits);
        cursor += 4;
      }
  }
  auto checksum = hash(v.data(), total - 32);
  std::memcpy(v.data() + total - 32, checksum.data(), 32);
  return v;
}
} // namespace native_fixture