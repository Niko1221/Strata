#include "strata/core/native_cache_config.hpp"
#include "spin8/query_v3_cuda.h"
#include <algorithm>
#include <cmath>
#include <cstring>
#include <limits>
extern "C" {
#include "../../spin8/third_party/sha256/sha256.h"
}
namespace strata::core {
namespace {
struct Hash {
  spin8_sidecar_sha256_t state;
  explicit Hash(const char *domain) {
    spin8_sidecar_sha256_init(&state);
    bytes(domain, std::strlen(domain) + 1);
  }
  void bytes(const void *p, size_t n) {
    spin8_sidecar_sha256_update(&state, static_cast<const unsigned char *>(p),
                                n);
  }
  void integer(uint64_t v) {
    uint8_t b[8];
    for (size_t i = 0; i < 8; ++i)
      b[i] = uint8_t(v >> (8 * i));
    bytes(b, sizeof(b));
  }
  void real(double v) {
    static_assert(sizeof(double) == 8 &&
                  std::numeric_limits<double>::is_iec559);
    uint64_t bits;
    std::memcpy(&bits, &v, sizeof(bits));
    integer(bits);
  }
  NativeCacheDigest finish() {
    NativeCacheDigest out{};
    spin8_sidecar_sha256_final(&state, out.data());
    return out;
  }
};
bool nonzero(const NativeCacheDigest &d) {
  return std::any_of(d.begin(), d.end(), [](uint8_t v) { return v != 0; });
}
bool reject(std::string &error, const char *text) {
  error = text;
  return false;
}
bool request_valid(const NativeCacheRequested &r, std::string &error) {
  if (r.version != 1)
    return reject(error, "unknown native configuration version");
  if (r.mode != NativeCacheMode::Resident || r.position_transport ||
      r.beta != 0 || r.precision != NativeCachePrecision::F32 ||
      uint32_t(r.representation) > 1 || uint32_t(r.k) < 1 ||
      uint32_t(r.k) > 5 || uint32_t(r.v) < 1 || uint32_t(r.v) > 5)
    return reject(
        error,
        "unsupported native mode, precision, KV dialect, transport or QJL");
  if (!r.capacity || !r.max_queries || !r.max_append_rows ||
      r.max_append_rows > r.capacity || r.planes.size() > 64)
    return reject(error, "invalid native execution limits");
  if (r.view_mask != 1 && r.view_mask != 2 && r.view_mask != 4 &&
      r.view_mask != 7)
    return reject(error, "unknown native view mask");
  double sum = 0;
  for (size_t i = 0; i < 3; ++i) {
    if (!std::isfinite(r.alpha[i]) || r.alpha[i] < 0 ||
        (!(r.view_mask & (1u << i)) && r.alpha[i] != 0))
      return reject(error,
                    "alpha must be an explicit simplex on enabled views");
    sum += r.alpha[i];
  }
  if (std::abs(sum - 1) > 1e-12)
    return reject(error, "alpha must sum to one; all-zero alpha is invalid");
  return true;
}
bool layer_valid(const spin8_query_adapter_layer_spec &s) {
  return s.layer_index <= 1048575 && s.q_heads && s.q_heads <= 4096 &&
         s.kv_heads && s.kv_heads <= 4096 && s.q_heads % s.kv_heads == 0 &&
         (s.head_width == 128 || s.head_width == 256);
}
void geometry_hash(Hash &h, const spin8_query_adapter_layer_spec &s) {
  h.integer(s.layer_index);
  h.integer(s.q_heads);
  h.integer(s.kv_heads);
  h.integer(s.head_width);
}
NativeCacheDigest no_adapter() {
  return Hash("strata-native-cache/query-v3/no-adapter-identity/v1").finish();
}
} // namespace
struct NativeCachePlan::Impl {
  NativeCacheDigest profile{}, layer{}, adapter_digest{};
  NativeCacheBinding binding;
  NativeCacheOwnerLayout layout;
  spin8_query_adapter_layer_spec geometry{};
  spin8_query_adapter_layer_view view{};
  std::vector<float> identity_parameters;
  uint64_t capacity = 0, max_queries = 0, max_append_rows = 0;
  std::unique_ptr<tqc_cache_cuda_plan, decltype(&tqc_cache_cuda_free)> cache{
      nullptr, tqc_cache_cuda_free};
  std::unique_ptr<spin8_query_adapter, decltype(&spin8_query_adapter_free)>
      adapter{nullptr, spin8_query_adapter_free};
  std::unique_ptr<spin8_query_v3_cuda_plan,
                  decltype(&spin8_query_v3_cuda_plan_free)>
      query{nullptr, spin8_query_v3_cuda_plan_free};
};
NativeCachePlan::NativeCachePlan(std::unique_ptr<Impl> p)
    : impl_(std::move(p)) {}
NativeCachePlan::~NativeCachePlan() = default;
bool NativeCachePlan::base_profile(const NativeCacheRequested &r,
                                   const NativeCacheDigest &position,
                                   NativeCacheDigest &out, std::string &error) {
  error.clear();
  if (!request_valid(r, error) || !nonzero(position)) {
    if (error.empty())
      error = "missing resolved position digest";
    return false;
  }
  // All three B representations come from ONE strict Spin element. Never
  // construct three unrelated SO(8) rotations or use the old adapter profile.
  spin8_plane planes[64]{};
  for (size_t i = 0; i < r.planes.size(); ++i)
    planes[i] = {r.planes[i].a, r.planes[i].b, r.planes[i].angle};
  spin8_plan *raw = nullptr;
  if (spin8_plan_create(planes, r.planes.size(), &raw) != SPIN8_OK)
    return reject(error, "invalid strict Spin plane sequence");
  std::unique_ptr<spin8_plan, decltype(&spin8_plan_free)> spin(raw,
                                                               spin8_plan_free);
  spin8_audit_report audit{};
  if (spin8_plan_audit(raw, &audit) != SPIN8_OK)
    return reject(error, "strict Spin audit failed");
  Hash h("strata-native-cache/base/v1;F=D2*H128/"
         "sqrt128*D1;TheTom-643956891dfdf51aecdbd05ca508dd21e21e8195;Qnorm-"
         "query-v3-rank8-tau-pi8-no-extra-RMS");
  h.integer(uint32_t(r.k));
  h.integer(uint32_t(r.v));
  h.integer(uint32_t(r.representation));
  h.integer(uint32_t(r.precision));
  h.integer(uint32_t(r.mode));
  h.integer(r.view_mask);
  for (auto a : r.alpha)
    h.real(a);
  h.real(r.beta);
  h.integer(r.position_transport);
  h.bytes(position.data(), position.size());
  for (int v = 0; v < 3; ++v) {
    const auto &a = audit.views[v];
    if (a.max_error > 1e-12 || a.ef_normalized > 1e-12 ||
        std::abs(a.log_abs_det) > 8e-12 || a.det_sign != 1)
      return reject(error, "strict Spin matrix invariant failed");
    double matrix[64];
    if (spin8_plan_matrix(raw, static_cast<spin8_view>(v), matrix) != SPIN8_OK)
      return reject(error, "strict Spin matrix unavailable");
    for (double x : matrix)
      h.real(x);
  }
  out = h.finish();
  return true;
}
std::shared_ptr<const NativeCachePlan>
NativeCachePlan::bind(const NativeCacheRequested &r,
                      const NativeCacheBinding &b, const void *sidecar,
                      size_t sidecar_bytes, std::string &error) {
  error.clear();
  try {
    auto p = std::make_unique<Impl>();
    if (!base_profile(r, b.position, p->profile, error))
      return {};
    if (b.raw_models.empty() || b.raw_models.size() > 256 ||
        b.normal_layers.empty() ||
        b.normal_layers.size() > SPIN8_QUERY_ADAPTER_MAX_LAYERS ||
        bool(sidecar) != bool(sidecar_bytes) ||
        sidecar_bytes > SPIN8_QUERY_ADAPTER_MAX_BYTES) {
      error = "missing or excessive independently supplied binding";
      return {};
    }
    for (const auto &d : b.raw_models)
      if (!nonzero(d)) {
        error = "missing raw model SHA-256";
        return {};
      }
    size_t selected = SIZE_MAX;
    for (size_t i = 0; i < b.normal_layers.size(); ++i) {
      const auto &s = b.normal_layers[i];
      if (!layer_valid(s) ||
          (i && b.normal_layers[i - 1].layer_index >= s.layer_index)) {
        error = "normal-layer directory must be exact, supported, sorted and "
                "unique";
        return {};
      }
      if (s.layer_index == b.selected_layer)
        selected = i;
    }
    if (selected == SIZE_MAX) {
      error = "logical selected layer missing from directory";
      return {};
    }
    p->binding = b;
    p->geometry = b.normal_layers[selected];
    p->capacity = r.capacity;
    p->max_queries = r.max_queries;
    p->max_append_rows = r.max_append_rows;
    Hash layer("strata-native-cache/logical-layer/v1");
    layer.integer(b.raw_models.size());
    for (const auto &d : b.raw_models)
      layer.bytes(d.data(), d.size());
    geometry_hash(layer, p->geometry);
    p->layer = layer.finish();
    p->adapter_digest = no_adapter();
    if (sidecar) {
      spin8_query_adapter_expectations e{};
      e.abi_version = SPIN8_QUERY_ADAPTER_ABI_VERSION;
      e.struct_bytes = sizeof(e);
      e.model_count = uint32_t(b.raw_models.size());
      e.layer_count = uint32_t(b.normal_layers.size());
      e.model_digests = b.raw_models.front().data();
      e.model_digests_bytes = b.raw_models.size() * 32;
      std::copy(p->profile.begin(), p->profile.end(), e.profile_digest);
      std::copy(b.position.begin(), b.position.end(), e.position_digest);
      e.layers = b.normal_layers.data();
      e.layers_bytes = b.normal_layers.size() * sizeof(e.layers[0]);
      spin8_query_adapter *adapter = nullptr;
      auto status = spin8_query_adapter_load(sidecar, sidecar_bytes, &e,
                                             sizeof(e), &adapter);
      if (status != SPIN8_OK) {
        error = "independent query-v3 sidecar binding/validation failed: " +
                std::to_string(status);
        return {};
      }
      p->adapter.reset(adapter);
      spin8_query_adapter_info info{};
      info.abi_version = 1;
      info.struct_bytes = sizeof(info);
      p->view.abi_version = 1;
      p->view.struct_bytes = sizeof(p->view);
      if (spin8_query_adapter_get_info(adapter, &info, sizeof(info)) !=
              SPIN8_OK ||
          spin8_query_adapter_get_layer(adapter, uint32_t(selected), &p->view,
                                        sizeof(p->view)) != SPIN8_OK) {
        error = "validated adapter view unavailable";
        return {};
      }
      std::copy_n(info.content_digest, 32, p->adapter_digest.begin());
      for (const auto &s : b.normal_layers)
        p->layout.host_parameter_bytes +=
            size_t(s.q_heads) * (size_t(8) * s.head_width + 71) * 4;
    }
    tqc_cache_cuda_config c{};
    c.version = 1;
    c.size = sizeof(c);
    c.precision = TQC_D_F32;
    c.representation = uint32_t(r.representation);
    c.k_dialect = uint32_t(r.k);
    c.v_dialect = uint32_t(r.v);
    c.view_mask = r.view_mask;
    c.columns = p->geometry.head_width;
    c.q_heads = p->geometry.q_heads;
    c.kv_heads = p->geometry.kv_heads;
    c.capacity = r.capacity;
    c.max_queries = r.max_queries;
    c.max_append_rows = r.max_append_rows;
    std::copy(r.alpha.begin(), r.alpha.end(), c.alpha);
    std::copy(p->layer.begin(), p->layer.end(), c.model);
    std::copy(p->adapter_digest.begin(), p->adapter_digest.end(), c.adapter);
    std::copy(b.position.begin(), b.position.end(), c.position);
    c.planes = r.planes.data();
    c.plane_count = r.planes.size();
    tqc_cache_cuda_plan *cache = nullptr;
    int rc = tqc_cache_cuda_create(&c, &cache);
    if (rc) {
      error = "native cache layout rejected: " + std::to_string(rc);
      return {};
    }
    p->cache.reset(cache);
    auto &l = p->layout;
    l.cache.version = 1;
    l.cache.size = sizeof(l.cache);
    if (tqc_cache_cuda_get_layout(cache, &l.cache)) {
      error = "native cache layout unavailable";
      return {};
    }
    const size_t heads = c.q_heads, width = c.columns;
    l.parameter_bytes = {heads * 8 * width * 4, heads * 8 * 4, heads * 56 * 4,
                         heads * 7 * 4};
    size_t bank_bytes = 0;
    for (auto bytes : l.parameter_bytes)
      bank_bytes += bytes;
    if (bank_bytes > SPIN8_QUERY_ADAPTER_MAX_BYTES) {
      error = "native parameter bank exceeds bounded host metadata";
      return {};
    }
    size_t cursor = l.cache.total_bytes, padding = 0;
    auto region = [&](size_t bytes, size_t &offset) {
      if (cursor > SIZE_MAX - 255)
        return false;
      offset = (cursor + 255) & ~size_t(255);
      padding += offset - cursor;
      if (bytes > SIZE_MAX - offset)
        return false;
      cursor = offset + bytes;
      return true;
    };
    for (size_t i = 0; i < 4; ++i)
      if (!region(l.parameter_bytes[i], l.parameter_offsets[i])) {
        error = "native bank byte overflow";
        return {};
      }
    if (!region(sizeof(int), l.query_status) || cursor > SIZE_MAX - 255) {
      error = "native owner byte overflow";
      return {};
    }
    l.total_bytes = (cursor + 255) & ~size_t(255);
    l.bank_alignment_bytes = padding + l.total_bytes - cursor;
    if (!p->adapter) {
      p->identity_parameters.resize(bank_bytes / 4, 0.f);
      l.host_parameter_bytes = bank_bytes;
      auto &v = p->view;
      v.abi_version = 1;
      v.struct_bytes = sizeof(v);
      v.spec = p->geometry;
      v.win = p->identity_parameters.data();
      v.bin = v.win + l.parameter_bytes[0] / 4;
      v.wout = v.bin + l.parameter_bytes[1] / 4;
      v.bout = v.wout + l.parameter_bytes[2] / 4;
      v.win_bytes = l.parameter_bytes[0];
      v.bin_bytes = l.parameter_bytes[1];
      v.wout_bytes = l.parameter_bytes[2];
      v.bout_bytes = l.parameter_bytes[3];
    }
    spin8_query_v3_cuda_plan *query = nullptr;
    if (spin8_query_v3_cuda_plan_create(SPIN8_QUERY_V3_VERSION, &query) !=
        SPIN8_OK) {
      error = "query-v3 host plan audit failed";
      return {};
    }
    p->query.reset(query);
    return std::shared_ptr<const NativeCachePlan>(
        new NativeCachePlan(std::move(p)));
  } catch (const std::exception &e) {
    error = std::string("native host plan failed: ") + e.what();
    return {};
  }
}
const NativeCacheDigest &NativeCachePlan::profile_digest() const {
  return impl_->profile;
}
const NativeCacheDigest &NativeCachePlan::layer_digest() const {
  return impl_->layer;
}
const NativeCacheDigest &NativeCachePlan::adapter_digest() const {
  return impl_->adapter_digest;
}
const NativeCacheOwnerLayout &NativeCachePlan::layout() const {
  return impl_->layout;
}
const spin8_query_adapter_layer_spec &NativeCachePlan::geometry() const {
  return impl_->geometry;
}
const tqc_cache_cuda_plan *NativeCachePlan::cache_plan() const {
  return impl_->cache.get();
}
uint64_t NativeCachePlan::capacity() const { return impl_->capacity; }
uint64_t NativeCachePlan::max_queries() const { return impl_->max_queries; }
uint64_t NativeCachePlan::max_append_rows() const {
  return impl_->max_append_rows;
}
bool NativeCachePlan::has_adapter() const { return bool(impl_->adapter); }
spin8_query_v3_cuda_plan *NativeCachePlan::query_plan() const {
  return impl_->query.get();
}
spin8_query_adapter_layer_view NativeCachePlan::parameters() const {
  return impl_->view;
}
} // namespace strata::core