#include "native_cache_fixture.hpp"
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <limits>
using namespace strata::core;
static size_t checks = 0;
#define CHECK(x)                                                               \
  do {                                                                         \
    ++checks;                                                                  \
    if (!(x)) {                                                                \
      std::fprintf(stderr, "line %d: %s\n", __LINE__, #x);                     \
      throw std::runtime_error(#x);                                            \
    }                                                                          \
  } while (0)
int main() {
  try {
    auto r = native_fixture::request();
    auto b = native_fixture::binding();
    std::string error;
    NativeCacheDigest profile{};
    CHECK(NativeCachePlan::base_profile(r, b.position, profile, error));
    auto base = NativeCachePlan::bind(r, b, nullptr, 0, error);
    CHECK(base && error.empty());
    CHECK(profile == base->profile_digest() && !base->has_adapter());
    const auto &l = base->layout();
    size_t parameter_bytes = 0;
    for (size_t i = 0; i < 4; ++i) {
      parameter_bytes += l.parameter_bytes[i];
      CHECK(l.parameter_offsets[i] % 256 == 0);
      CHECK(l.parameter_offsets[i] >= l.cache.total_bytes);
    }
    CHECK(l.total_bytes % 256 == 0 && l.query_status % 256 == 0);
    CHECK(l.total_bytes == l.cache.total_bytes + parameter_bytes + sizeof(int) +
                               l.bank_alignment_bytes);
    CHECK(l.cache.k_bytes[0] == l.cache.k_bytes[1] &&
          l.cache.k_bytes[1] == l.cache.k_bytes[2]);
    CHECK(l.cache.v_bytes == r.capacity * 2 * l.cache.v_row_bytes);
    auto zero = native_fixture::sidecar(r, b, false),
         nz = native_fixture::sidecar(r, b, true);
    auto a = NativeCachePlan::bind(r, b, zero.data(), zero.size(), error);
    CHECK(a && a->has_adapter());
    auto c = NativeCachePlan::bind(r, b, nz.data(), nz.size(), error);
    CHECK(c);
    CHECK(a->profile_digest() == profile && c->profile_digest() == profile);
    CHECK(a->adapter_digest() != c->adapter_digest());
    CHECK(std::memcmp(a->layout().cache.identity, c->layout().cache.identity,
                      32) != 0);
    b.selected_layer = 7;
    auto other = NativeCachePlan::bind(r, b, nz.data(), nz.size(), error);
    CHECK(other);
    CHECK(other->layer_digest() != c->layer_digest());
    CHECK(std::memcmp(other->layout().cache.identity,
                      c->layout().cache.identity, 32) != 0);
    b = native_fixture::binding();
    auto bad_binding = [&](NativeCacheBinding bad) {
      CHECK(!NativeCachePlan::bind(r, bad, nz.data(), nz.size(), error));
      CHECK(!error.empty());
    };
    auto bad = b;
    std::swap(bad.raw_models[0], bad.raw_models[1]);
    bad_binding(bad);
    bad = b;
    bad.raw_models[0].back() = 1;
    bad_binding(bad);
    bad = b;
    bad.position.back() = 1;
    bad_binding(bad);
    bad = b;
    bad.selected_layer = 0;
    bad_binding(bad);
    bad = b;
    bad.normal_layers.pop_back();
    bad_binding(bad);
    bad = b;
    bad.normal_layers[1].layer_index = 2;
    bad_binding(bad);
    bad = b;
    std::reverse(bad.normal_layers.begin(), bad.normal_layers.end());
    bad_binding(bad);
    bad = b;
    bad.normal_layers[0].head_width = 256;
    bad_binding(bad);
    bad = b;
    bad.normal_layers[0].q_heads = 3;
    bad_binding(bad);
    bad = b;
    bad.raw_models.clear();
    bad_binding(bad);
    bad = b;
    bad.raw_models[0] = {};
    bad_binding(bad);
    CHECK(!NativeCachePlan::bind(r, b, nz.data(), 0, error));
    CHECK(!NativeCachePlan::bind(r, b, nullptr, nz.size(), error));
    auto broken = nz;
    broken.back() ^= 1;
    CHECK(!NativeCachePlan::bind(r, b, broken.data(), broken.size(), error));
    auto bad_request = [&](NativeCacheRequested v) {
      CHECK(!NativeCachePlan::bind(v, b, nullptr, 0, error));
      CHECK(!error.empty());
    };
    auto v = r;
    v.version = 2;
    bad_request(v);
    v = r;
    v.alpha = {};
    bad_request(v);
    v = r;
    v.alpha = {-1, 1, 1};
    bad_request(v);
    v = r;
    v.alpha[0] = std::numeric_limits<double>::quiet_NaN();
    bad_request(v);
    v = r;
    v.view_mask = 3;
    bad_request(v);
    v = r;
    v.view_mask = 1;
    bad_request(v);
    v = r;
    v.position_transport = true;
    bad_request(v);
    v = r;
    v.beta = .01;
    bad_request(v);
    v = r;
    v.precision = NativeCachePrecision::F16;
    bad_request(v);
    v = r;
    v.precision = NativeCachePrecision::BF16;
    bad_request(v);
    v = r;
    v.k = static_cast<NativeKvFormat>(6);
    bad_request(v); // weight-format path
    v = r;
    v.v = static_cast<NativeKvFormat>(99);
    bad_request(v);
    v = r;
    v.representation = static_cast<NativeCacheRepresentation>(2);
    bad_request(v);
    for (uint32_t mode = 2; mode <= 8; ++mode) {
      v = r;
      v.mode = static_cast<NativeCacheMode>(mode);
      bad_request(v);
    }
    v = r;
    v.max_queries = 0;
    bad_request(v);
    v = r;
    v.max_queries = UINT64_MAX;
    bad_request(v);
    v = r;
    v.max_append_rows = 0;
    bad_request(v);
    v = r;
    v.max_append_rows = v.capacity + 1;
    bad_request(v);
    v = r;
    v.capacity = UINT64_MAX;
    bad_request(v);
    v = r;
    v.planes[0].a = 8;
    bad_request(v);
    v = r;
    v.planes.resize(65);
    bad_request(v);
    for (uint32_t k = 1; k <= 5; ++k)
      for (uint32_t value = 1; value <= 5; ++value) {
        v = r;
        v.k = static_cast<NativeKvFormat>(k);
        v.v = static_cast<NativeKvFormat>(value);
        auto plan = NativeCachePlan::bind(v, b, nullptr, 0, error);
        CHECK(plan);
        CHECK(plan->layout().cache.k_row_bytes == 18 + 16 * k);
        CHECK(plan->layout().cache.v_row_bytes == 18 + 16 * value);
      }
    for (uint32_t mask : {1u, 2u, 4u}) {
      v = r;
      v.view_mask = mask;
      v.alpha = {};
      for (size_t i = 0; i < 3; ++i)
        if (mask & (1u << i))
          v.alpha[i] = 1;
      auto plan = NativeCachePlan::bind(v, b, nullptr, 0, error);
      CHECK(plan);
      for (size_t i = 0; i < 3; ++i)
        CHECK(bool(plan->layout().cache.k_bytes[i]) == bool(mask & (1u << i)));
    }
    v = r;
    v.representation = NativeCacheRepresentation::DiagnosticF32;
    auto diagnostic = NativeCachePlan::bind(v, b, nullptr, 0, error);
    CHECK(diagnostic);
    CHECK(diagnostic->layout().cache.k_row_bytes == 512 &&
          diagnostic->layout().cache.v_row_bytes == 512);
    CHECK(diagnostic->profile_digest() != profile);
    v = r;
    v.capacity = 16;
    v.max_queries = 4;
    auto limits = NativeCachePlan::bind(v, b, nullptr, 0, error);
    CHECK(limits);
    CHECK(limits->profile_digest() == profile &&
          std::memcmp(limits->layout().cache.identity,
                      base->layout().cache.identity, 32) != 0);
    std::printf(
        "%zu native config checks; synthetic zero fixture %s; nonzero %s\n",
        checks,
        native_fixture::hex(native_fixture::hash(zero.data(), zero.size()))
            .c_str(),
        native_fixture::hex(native_fixture::hash(nz.data(), nz.size()))
            .c_str());
    return 0;
  } catch (const std::exception &e) {
    std::fprintf(stderr, "%s\n", e.what());
    return 1;
  }
}