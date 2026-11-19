#include "spin8/attention_v2.h"
#include <algorithm>
#include <array>
#include <cmath>
#include <cstring>
#include <limits>
#include <memory>
#include <new>
#include <stdexcept>
#include <vector>

struct spin8_attention_v2_plan {
  uint64_t source_id;
  size_t n_q_heads, n_kv_heads, d_qk, d_v;
  std::vector<uint32_t> map;
  std::unique_ptr<spin8_plan, decltype(&spin8_plan_free)> rotation{
      nullptr, spin8_plan_free};
  spin8_view view = SPIN8_VECTOR;
  bool h128 = false;
};

namespace {
struct Failure {
  spin8_status status;
};
void require(bool condition, spin8_status status = SPIN8_INVALID_ARGUMENT) {
  if (!condition)
    throw Failure{status};
}
size_t multiply(size_t a, size_t b) {
  require(!b || a <= SIZE_MAX / b);
  return a * b;
}
size_t add(size_t a, size_t b) {
  require(a <= SIZE_MAX - b);
  return a + b;
}
size_t count(size_t a, size_t b, size_t c) {
  const size_t n = multiply(multiply(a, b), c);
  require(n <= static_cast<size_t>(PTRDIFF_MAX) / sizeof(double));
  return n;
}
void span(const void *p, size_t bytes) {
  require(p && bytes && bytes <= static_cast<size_t>(PTRDIFF_MAX));
  require(reinterpret_cast<uintptr_t>(p) <= UINTPTR_MAX - bytes);
}
template <class T> void object(const T *p, size_t bytes = sizeof(T)) {
  require(bytes == sizeof(T));
  span(static_cast<const void *>(p), bytes);
  require(reinterpret_cast<uintptr_t>(p) % alignof(T) == 0);
}
bool overlaps(const void *a, size_t an, const void *b, size_t bn) {
  const auto x = reinterpret_cast<uintptr_t>(a);
  const auto y = reinterpret_cast<uintptr_t>(b);
  // Both ranges have already passed span().
  return x < y + bn && y < x + an;
}
template <class T> void header(const T &s) {
  require(s.struct_bytes == sizeof(T));
  require(s.abi_version == SPIN8_ATTENTION_V2_ABI_VERSION, SPIN8_UNSUPPORTED);
  for (uint64_t value : s.reserved)
    require(value == 0);
}
bool absent(const spin8_attention_v2_input &b) {
  if (b.data)
    return false;
  require(!b.capacity_bytes && !b.token_stride_bytes && !b.head_stride_bytes &&
          !b.channel_stride_bytes);
  return true;
}
size_t offset(const spin8_attention_v2_input &b, size_t t, size_t h, size_t c) {
  // Geometry validation proves each product and sum fits before any load.
  return t * b.token_stride_bytes + h * b.head_stride_bytes +
         c * b.channel_stride_bytes;
}
void tensor(const spin8_attention_v2_input &b, size_t t, size_t h, size_t c,
            size_t item) {
  require(t && h && c && b.token_stride_bytes && b.head_stride_bytes &&
          b.channel_stride_bytes);
  span(b.data, b.capacity_bytes);
  size_t end = add(multiply(t - 1, b.token_stride_bytes),
                   multiply(h - 1, b.head_stride_bytes));
  end = add(add(end, multiply(c - 1, b.channel_stride_bytes)), item);
  require(end <= b.capacity_bytes);
}
spin8_attention_v2_input output_layout(const spin8_attention_v2_output &b,
                                       size_t t, size_t h, size_t c) {
  spin8_attention_v2_input r{b.data, b.capacity_bytes, b.token_stride_bytes,
                             b.head_stride_bytes, b.channel_stride_bytes};
  tensor(r, t, h, c, sizeof(double));
  // Accept padded/permuted axis layouts, proving no output elements overlap.
  std::array<std::pair<size_t, size_t>, 3> axes{{{b.token_stride_bytes, t},
                                                 {b.head_stride_bytes, h},
                                                 {b.channel_stride_bytes, c}}};
  std::sort(axes.begin(), axes.end());
  size_t occupied = sizeof(double);
  for (const auto &axis : axes)
    if (axis.second > 1) {
      require(axis.first >= occupied);
      occupied = add(occupied, multiply(axis.first, axis.second - 1));
    }
  return r;
}
double load(const spin8_attention_v2_input &b, size_t t, size_t h, size_t c) {
  double value;
  std::memcpy(&value,
              static_cast<const unsigned char *>(b.data) + offset(b, t, h, c),
              sizeof(value));
  return value;
}
bool allowed(const spin8_attention_v2_input &b, size_t t, size_t h, size_t k) {
  return static_cast<const unsigned char *>(b.data)[offset(b, t, h, k)] != 0;
}
void finite(double value) { require(std::isfinite(value), SPIN8_NONFINITE); }
void success(spin8_status status) { require(status == SPIN8_OK, status); }
std::vector<double> pack(const spin8_attention_v2_input &b, size_t t, size_t h,
                         size_t c) {
  std::vector<double> result(count(t, h, c));
  for (size_t i = 0; i < t; ++i)
    for (size_t j = 0; j < h; ++j)
      for (size_t k = 0; k < c; ++k) {
        const double x = load(b, i, j, k);
        finite(x);
        result[(i * h + j) * c + k] = x;
      }
  return result;
}

void transform(const spin8_attention_v2_plan &p, std::vector<double> &data,
               size_t width, bool inverse) {
  if (!p.rotation)
    return;
  const size_t rows = data.size() / width;
  auto hadamard = [&] {
    success(spin8_hadamard_block(SPIN8_F64, data.data(), width, data.data(),
                                 width, rows, width, 128, nullptr,
                                 inverse ? 1 : 0));
  };
  if (p.h128 && !inverse)
    hadamard();
  success(spin8_transform(p.rotation.get(), p.view, SPIN8_F64, data.data(),
                          width, data.data(), width, rows, width,
                          inverse ? 1 : 0));
  if (p.h128 && inverse)
    hadamard();
}

spin8_status create(const spin8_attention_v2_config *c, size_t bytes,
                    spin8_attention_v2_plan **out) {
  object(c, bytes);
  object(out);
  const void *out_storage = static_cast<const void *>(out);
  require(!overlaps(c, bytes, out_storage, sizeof(*out)));
  header(*c);
  require(c->semantics == SPIN8_ATTENTION_V2_SELF && c->dtype == SPIN8_F64 &&
              c->codec == SPIN8_ATTENTION_V2_CODEC_NONE &&
              c->basis <= SPIN8_ATTENTION_V2_FIXED,
          SPIN8_UNSUPPORTED);
  require(c->source_id && c->n_q_heads && c->n_kv_heads && c->d_qk && c->d_v);
  count(1, c->n_q_heads, c->d_qk);
  count(1, c->n_kv_heads, c->d_qk);
  count(1, c->n_q_heads, c->d_v);
  count(1, c->n_kv_heads, c->d_v);
  require(c->q_to_kv_bytes >= multiply(c->n_q_heads, sizeof(uint32_t)));
  span(c->q_to_kv, c->q_to_kv_bytes);
  require(!overlaps(c->q_to_kv, c->q_to_kv_bytes, out_storage, sizeof(*out)));
  if (c->basis == SPIN8_ATTENTION_V2_IDENTITY) {
    require(!c->view && !c->hadamard_block && !c->planes && !c->plane_count &&
            !c->planes_bytes);
  } else {
    require(c->view <= SPIN8_SPIN_MINUS &&
                (c->hadamard_block == 0 || c->hadamard_block == 128),
            SPIN8_UNSUPPORTED);
    if (c->plane_count) {
      require(c->planes_bytes >= multiply(c->plane_count, sizeof(spin8_plane)));
      span(c->planes, c->planes_bytes);
      require(!overlaps(c->planes, c->planes_bytes, out_storage, sizeof(*out)));
    } else {
      require(!c->planes && !c->planes_bytes);
    }
  }
  auto p = std::make_unique<spin8_attention_v2_plan>();
  p->source_id = c->source_id;
  p->n_q_heads = c->n_q_heads;
  p->n_kv_heads = c->n_kv_heads;
  p->d_qk = c->d_qk;
  p->d_v = c->d_v;
  p->map.resize(c->n_q_heads);
  // Copy bytes instead of dereferencing a possibly unaligned mapping.
  std::memcpy(p->map.data(), c->q_to_kv, c->n_q_heads * sizeof(uint32_t));
  for (uint32_t index : p->map)
    require(index < c->n_kv_heads);
  if (c->basis == SPIN8_ATTENTION_V2_FIXED) {
    std::vector<spin8_plane> planes(c->plane_count);
    if (c->plane_count)
      std::memcpy(planes.data(), c->planes,
                  c->plane_count * sizeof(spin8_plane));
    spin8_plan *rotation = nullptr;
    // The v1 constructor audits the actual three representations before
    // success.
    success(spin8_plan_create(planes.data(), planes.size(), &rotation));
    p->rotation.reset(rotation);
    p->view = static_cast<spin8_view>(c->view);
    p->h128 = c->hadamard_block == 128;
    if (p->h128) {
      spin8_matrix_audit audit{};
      success(spin8_hadamard_block_audit(SPIN8_F64, 128, nullptr, &audit));
      require(audit.ef_normalized <= spin8_precision_tolerance(SPIN8_F64) &&
                  audit.max_error <= spin8_precision_tolerance(SPIN8_F64) &&
                  audit.det_sign == 1 && std::abs(audit.log_abs_det) <= 1e-12,
              SPIN8_AUDIT_FAILED);
    }
  }
  *out = p.release();
  return SPIN8_OK;
}

spin8_status execute(const spin8_attention_v2_plan *p,
                     const spin8_attention_v2_request *r, size_t bytes) {
  object(p);
  object(r, bytes);
  header(*r);
  require(r->position_stage == SPIN8_ATTENTION_V2_POST_POSITION &&
              r->q_source_id == p->source_id &&
              r->kv_source_id == p->source_id &&
              r->logit_order <= SPIN8_ATTENTION_V2_SCALE_SOFTCAP_BIAS,
          SPIN8_UNSUPPORTED);
  require(r->n_query && r->n_key);
  finite(r->scale);
  finite(r->softcap);
  require(r->softcap >= 0);
  const size_t nq = r->n_query, nk = r->n_key, nh = p->n_q_heads;
  const size_t kvh = p->n_kv_heads, dq = p->d_qk, dv = p->d_v;
  count(nq, nh, dq);
  count(nk, kvh, dq);
  count(nk, kvh, dv);
  const size_t outputs = count(nq, nh, dv);
  count(nq, nh, nk);
  const auto output = output_layout(r->output, nq, nh, dv);
  require(!overlaps(output.data, output.capacity_bytes, r, bytes));
  require(!overlaps(output.data, output.capacity_bytes, p, sizeof(*p)));
  require(!overlaps(output.data, output.capacity_bytes, p->map.data(),
                    p->map.size() * sizeof(uint32_t)));
  tensor(r->q, nq, nh, dq, sizeof(double));
  tensor(r->k, nk, kvh, dq, sizeof(double));
  tensor(r->v, nk, kvh, dv, sizeof(double));
  tensor(r->mask, nq, nh, nk, 1);
  const bool has_bias = !absent(r->bias), has_gate = !absent(r->gate);
  if (has_bias)
    tensor(r->bias, nq, nh, nk, sizeof(double));
  if (has_gate)
    tensor(r->gate, nq, nh, dv, sizeof(double));
  for (const auto *b : {&r->q, &r->k, &r->v, &r->mask, &r->bias, &r->gate})
    if (b->data)
      require(!overlaps(output.data, output.capacity_bytes, b->data,
                        b->capacity_bytes));

  auto q = pack(r->q, nq, nh, dq), k = pack(r->k, nk, kvh, dq);
  auto v = pack(r->v, nk, kvh, dv);
  const auto bias =
      has_bias ? pack(r->bias, nq, nh, nk) : std::vector<double>{};
  const auto gate =
      has_gate ? pack(r->gate, nq, nh, dv) : std::vector<double>{};
  transform(*p, q, dq, false);
  transform(*p, k, dq, false);
  transform(*p, v, dv, false);
  std::vector<double> result(outputs, 0), scores(nk);
  for (size_t t = 0; t < nq; ++t)
    for (size_t h = 0; h < nh; ++h) {
      double maximum = -std::numeric_limits<double>::infinity();
      bool any = false;
      const size_t head = p->map[h], row = t * nh + h;
      for (size_t j = 0; j < nk; ++j) {
        scores[j] = 0;
        if (!allowed(r->mask, t, h, j))
          continue; // Exclusion is structural, never a softcapped sentinel.
        any = true;
        double dot = 0;
        for (size_t c = 0; c < dq; ++c)
          dot += q[row * dq + c] * k[(j * kvh + head) * dq + c];
        finite(dot);
        double s = dot * r->scale;
        finite(s);
        const double b = has_bias ? bias[row * nk + j] : 0;
        if (r->logit_order == SPIN8_ATTENTION_V2_SCALE_BIAS_SOFTCAP) {
          s += b;
          finite(s);
        }
        if (r->softcap > 0)
          s = r->softcap * std::tanh(s / r->softcap);
        if (r->logit_order == SPIN8_ATTENTION_V2_SCALE_SOFTCAP_BIAS)
          s += b;
        finite(s);
        scores[j] = s;
        maximum = std::max(maximum, s);
      }
      require(any);
      double sum = 0;
      for (size_t j = 0; j < nk; ++j) {
        scores[j] =
            allowed(r->mask, t, h, j) ? std::exp(scores[j] - maximum) : 0;
        sum += scores[j];
      }
      finite(sum);
      require(sum > 0, SPIN8_NONFINITE);
      for (size_t c = 0; c < dv; ++c) {
        double value = 0;
        for (size_t j = 0; j < nk; ++j)
          value += (scores[j] / sum) * v[(j * kvh + head) * dv + c];
        finite(value);
        result[row * dv + c] = value;
      }
    }
  transform(*p, result, dv, true);
  for (size_t i = 0; i < outputs; ++i) {
    if (has_gate)
      result[i] *= gate[i];
    finite(result[i]);
  }
  // This is the only write to caller output; every possible failure is earlier.
  for (size_t t = 0; t < nq; ++t)
    for (size_t h = 0; h < nh; ++h)
      for (size_t c = 0; c < dv; ++c)
        std::memcpy(static_cast<unsigned char *>(r->output.data) +
                        offset(output, t, h, c),
                    &result[(t * nh + h) * dv + c], sizeof(double));
  return SPIN8_OK;
}

template <class F> spin8_status boundary(F &&f) noexcept {
  try {
    return f();
  } catch (const Failure &failure) {
    return failure.status;
  } catch (const std::bad_alloc &) {
    return SPIN8_ALLOCATION_FAILED;
  } catch (const std::length_error &) {
    return SPIN8_ALLOCATION_FAILED;
  } catch (...) {
    return SPIN8_INVALID_ARGUMENT;
  }
}
} // namespace

extern "C" spin8_status
spin8_attention_v2_plan_create(const spin8_attention_v2_config *c, size_t bytes,
                               spin8_attention_v2_plan **out) {
  return boundary([&] { return create(c, bytes, out); });
}
extern "C" void spin8_attention_v2_plan_free(spin8_attention_v2_plan *p) {
  delete p; // Only noexcept standard-library destruction; null is allowed.
}
extern "C" spin8_status
spin8_attention_v2_execute(const spin8_attention_v2_plan *p,
                           const spin8_attention_v2_request *r, size_t bytes) {
  return boundary([&] { return execute(p, r, bytes); });
}
