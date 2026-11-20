#include "spin8/query_v3.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <limits>
#include <memory>
#include <new>
#include <vector>

struct spin8_query_v3_controller {
  size_t width;
  std::vector<double> win;
  std::array<double, 8> bin;
  std::array<double, 56> wout;
  std::array<double, 7> bout;
  bool identity;
};

namespace {
using M8 = std::array<double, 64>;
using V8 = std::array<double, 8>;
bool product(size_t a, size_t b, size_t &out) {
  if (b && a > SIZE_MAX / b)
    return false;
  out = a * b;
  return true;
}
bool pointer(const void *p, size_t n, size_t alignment) {
  const auto address = reinterpret_cast<uintptr_t>(p);
  return p && address % alignment == 0 && address <= UINTPTR_MAX - n;
}
bool span(size_t rows, size_t width, size_t stride, size_t item,
          size_t capacity, size_t &bytes) {
  size_t last = 0;
  return rows && width && stride >= width && product(rows - 1, stride, last) &&
         last <= SIZE_MAX - width && product(last + width, item, bytes) &&
         bytes <= capacity;
}
bool overlap(const void *a, size_t an, const void *b, size_t bn) {
  const auto x = reinterpret_cast<uintptr_t>(a);
  const auto y = reinterpret_cast<uintptr_t>(b);
  return x < y + bn && y < x + an;
}
bool finite_product(double a, double b, double &out) {
  out = a * b;
  return std::isfinite(out);
}
bool finite_add(double a, double b, double &out) {
  out = a + b;
  return std::isfinite(out);
}
bool finite_madd(double a, double b, double &sum) {
  double term = 0.;
  return finite_product(a, b, term) && finite_add(sum, term, sum);
}
spin8_status vjp_spans(const spin8_query_v3_controller *ctl, const double *q,
                       size_t qc, const double *dy, size_t yc,
                       const spin8_query_v3_gradients_f64 *g) {
  if (!ctl || !pointer(g, sizeof(*g), alignof(spin8_query_v3_gradients_f64)))
    return SPIN8_INVALID_ARGUMENT;
  size_t n = 0, wn = 0;
  if (!product(ctl->width, sizeof(double), n) || !product(8, n, wn) || n > qc ||
      n > yc || !pointer(q, n, alignof(double)) ||
      !pointer(dy, n, alignof(double)))
    return SPIN8_INVALID_ARGUMENT;
  const void *outputs[] = {g->dq, g->dwin, g->dbin, g->dwout, g->dbout};
  const size_t sizes[] = {n, wn, 8 * sizeof(double), 56 * sizeof(double),
                          7 * sizeof(double)};
  const size_t capacities[] = {g->dq_bytes, g->dwin_bytes, g->dbin_bytes,
                               g->dwout_bytes, g->dbout_bytes};
  // The descriptor and immutable controller are also inputs to this call.
  const void *inputs[] = {q, dy, g, ctl, ctl->win.data()};
  const size_t input_sizes[] = {n, n, sizeof(*g), sizeof(*ctl), wn};
  for (size_t i = 0; i < 5; ++i) {
    if (sizes[i] > capacities[i] ||
        !pointer(outputs[i], sizes[i], alignof(double)))
      return SPIN8_INVALID_ARGUMENT;
    for (size_t j = 0; j < 5; ++j)
      if (overlap(outputs[i], sizes[i], inputs[j], input_sizes[j]))
        return SPIN8_INVALID_ARGUMENT;
    for (size_t j = 0; j < i; ++j)
      if (overlap(outputs[i], sizes[i], outputs[j], sizes[j]))
        return SPIN8_INVALID_ARGUMENT;
  }
  for (size_t i = 0; i < ctl->width; ++i)
    if (!std::isfinite(q[i]) || !std::isfinite(dy[i]))
      return SPIN8_NONFINITE;
  return SPIN8_OK;
}
M8 times_transpose(const M8 &a, const M8 &b) {
  M8 out{};
  for (size_t i = 0; i < 8; ++i)
    for (size_t j = 0; j < 8; ++j)
      for (size_t k = 0; k < 8; ++k)
        out[i * 8 + j] += a[i * 8 + k] * b[j * 8 + k];
  return out;
}
struct Basis {
  std::array<M8, 7> j{};
  spin8_query_v3_basis_report report{};
  spin8_status status = SPIN8_AUDIT_FAILED;
  Basis() {
    std::array<double, 2048> gamma{};
    status = spin8_clifford_generators(gamma.data());
    if (status != SPIN8_OK)
      return;
    std::array<M8, 8> l{}, c{};
    for (size_t i = 0; i < 8; ++i)
      for (size_t a = 0; a < 8; ++a)
        for (size_t b = 0; b < 8; ++b) {
          const double value = gamma[i * 256 + a * 16 + 8 + b];
          l[i][a * 8 + b] = value;
          c[b][i * 8 + a] = value;
        }
    // Polarization of ||L(v)c||^2=||v||^2||c||^2 proves the normed
    // tensor identity. Independently polarize C(c)C(c)^T over all 8 axes.
    for (size_t a = 0; a < 8; ++a)
      for (size_t b = 0; b < 8; ++b)
        for (size_t i = 0; i < 8; ++i)
          for (size_t k = 0; k < 8; ++k) {
            double lp = 0, cp = 0;
            for (size_t t = 0; t < 8; ++t) {
              lp += l[a][t * 8 + i] * l[b][t * 8 + k] +
                    l[b][t * 8 + i] * l[a][t * 8 + k];
              cp += c[a][i * 8 + t] * c[b][k * 8 + t] +
                    c[b][i * 8 + t] * c[a][k * 8 + t];
            }
            const double expected = a == b && i == k ? 2. : 0.;
            report.normed_max_error =
                std::max(report.normed_max_error, std::abs(lp - expected));
            report.polarization_max_error = std::max(
                report.polarization_max_error, std::abs(cp - expected));
          }
    for (size_t a = 0; a < 7; ++a)
      j[a] = times_transpose(c[a + 1], c[0]);
    for (size_t a = 0; a < 7; ++a)
      for (size_t i = 0; i < 8; ++i)
        for (size_t k = 0; k < 8; ++k) {
          report.skew_max_error =
              std::max(report.skew_max_error,
                       std::abs(j[a][i * 8 + k] + j[a][k * 8 + i]));
          for (size_t b = 0; b < 7; ++b) {
            double sum = 0;
            for (size_t t = 0; t < 8; ++t)
              sum += j[a][i * 8 + t] * j[b][t * 8 + k] +
                     j[b][i * 8 + t] * j[a][t * 8 + k];
            const double expected = a == b && i == k ? -2. : 0.;
            report.clifford_max_error =
                std::max(report.clifford_max_error, std::abs(sum - expected));
          }
        }
    status = report.normed_max_error <= 1e-12 &&
                     report.polarization_max_error <= 1e-12 &&
                     report.skew_max_error <= 1e-12 &&
                     report.clifford_max_error <= 1e-12
                 ? SPIN8_OK
                 : SPIN8_AUDIT_FAILED;
  }
};
const Basis &basis() {
  static const Basis result;
  return result;
}
spin8_status matrix(const double *c, M8 &out) {
  if (!pointer(c, 8 * sizeof(double), alignof(double)))
    return SPIN8_INVALID_ARGUMENT;
  double norm = 0;
  for (size_t i = 0; i < 8; ++i) {
    if (!std::isfinite(c[i]))
      return SPIN8_NONFINITE;
    norm = std::hypot(norm, c[i]);
  }
  if (std::abs(norm - 1.) > 1e-13)
    return SPIN8_INVALID_ARGUMENT;
  const auto &b = basis();
  if (b.status != SPIN8_OK)
    return b.status;
  M8 result{};
  for (size_t i = 0; i < 8; ++i)
    result[i * 8 + i] = c[0];
  for (size_t a = 0; a < 7; ++a)
    for (size_t i = 0; i < 64; ++i)
      result[i] += c[a + 1] * b.j[a][i];
  spin8_matrix_audit audit{};
  const auto status = spin8_audit_matrix(result.data(), &audit);
  if (status != SPIN8_OK)
    return status;
  if (audit.det_sign != 1 || audit.max_error > 1e-12 ||
      audit.ef_normalized > 1e-12 || std::abs(audit.log_abs_det) > 1e-12)
    return SPIN8_AUDIT_FAILED;
  out = result;
  return SPIN8_OK;
}
template <class T>
spin8_status input_span(const spin8_query_v3_controller *ctl, const void *src,
                        size_t capacity, size_t stride, size_t rows,
                        size_t &bytes) {
  if (!span(rows, ctl->width, stride, sizeof(T), capacity, bytes) ||
      !pointer(src, bytes, alignof(T)))
    return SPIN8_INVALID_ARGUMENT;
  const auto *q = static_cast<const T *>(src);
  for (size_t r = 0; r < rows; ++r)
    for (size_t i = 0; i < ctl->width; ++i)
      if (!std::isfinite(q[r * stride + i]))
        return SPIN8_NONFINITE;
  return SPIN8_OK;
}
template <class T>
spin8_status spinor(const spin8_query_v3_controller *ctl, const T *q, V8 &c) {
  if (ctl->identity) {
    c = {1., 0., 0., 0., 0., 0., 0., 0.};
    return SPIN8_OK;
  }
  V8 hidden{};
  for (size_t k = 0; k < 8; ++k) {
    double z = ctl->bin[k];
    for (size_t i = 0; i < ctl->width; ++i)
      z += ctl->win[k * ctl->width + i] * static_cast<double>(q[i]);
    if (!std::isfinite(z))
      return SPIN8_NONFINITE;
    hidden[k] = std::tanh(z);
  }
  std::array<double, 7> a = ctl->bout;
  for (size_t i = 0; i < 7; ++i) {
    for (size_t k = 0; k < 8; ++k)
      a[i] += ctl->wout[i * 8 + k] * hidden[k];
    if (!std::isfinite(a[i]))
      return SPIN8_NONFINITE;
  }
  return spin8_query_v3_spinor_from_tangent(a.data(), c.data());
}
template <class T>
spin8_status apply(const spin8_query_v3_controller *ctl, const void *src,
                   size_t sc, size_t ss, void *dst, size_t dc, size_t ds,
                   size_t rows) {
  size_t sn = 0, dn = 0, count = 0;
  if (!span(rows, ctl->width, ss, sizeof(T), sc, sn) ||
      !span(rows, ctl->width, ds, sizeof(T), dc, dn) ||
      !pointer(src, sn, alignof(T)) || !pointer(dst, dn, alignof(T)) ||
      !product(rows, ctl->width, count) ||
      (overlap(src, sn, dst, dn) && (src != dst || ss != ds)))
    return SPIN8_INVALID_ARGUMENT;
  auto status = input_span<T>(ctl, src, sc, ss, rows, sn);
  if (status != SPIN8_OK)
    return status;
  try {
    const auto *q = static_cast<const T *>(src);
    std::vector<T> result(count);
    for (size_t r = 0; r < rows; ++r) {
      V8 c{};
      status = spinor(ctl, q + r * ss, c);
      if (status != SPIN8_OK)
        return status;
      M8 m{};
      if (!ctl->identity) {
        status = matrix(c.data(), m);
        if (status != SPIN8_OK)
          return status;
      }
      size_t base = 0;
      if (!ctl->identity)
        for (; ctl->width - base >= 8; base += 8)
          for (size_t i = 0; i < 8; ++i) {
            double sum = 0;
            for (size_t j = 0; j < 8; ++j)
              sum += m[i * 8 + j] * static_cast<double>(q[r * ss + base + j]);
            if (!std::isfinite(sum) ||
                std::abs(sum) >
                    static_cast<double>(std::numeric_limits<T>::max()))
              return SPIN8_NONFINITE;
            result[r * ctl->width + base + i] = static_cast<T>(sum);
          }
      for (; base < ctl->width; ++base)
        result[r * ctl->width + base] = q[r * ss + base];
    }
    for (size_t r = 0; r < rows; ++r)
      std::copy_n(result.data() + r * ctl->width, ctl->width,
                  static_cast<T *>(dst) + r * ds);
    return SPIN8_OK;
  } catch (const std::bad_alloc &) {
    return SPIN8_ALLOCATION_FAILED;
  } catch (...) {
    return SPIN8_INVALID_ARGUMENT;
  }
}
template <class T>
spin8_status spinors(const spin8_query_v3_controller *ctl, const void *src,
                     size_t sc, size_t ss, double *dst, size_t dc, size_t ds,
                     size_t rows) {
  size_t sn = 0, dn = 0;
  if (!span(rows, ctl->width, ss, sizeof(T), sc, sn) ||
      !span(rows, 8, ds, sizeof(double), dc, dn) ||
      !pointer(src, sn, alignof(T)) || !pointer(dst, dn, alignof(double)) ||
      overlap(src, sn, dst, dn))
    return SPIN8_INVALID_ARGUMENT;
  auto status = input_span<T>(ctl, src, sc, ss, rows, sn);
  if (status != SPIN8_OK)
    return status;
  try {
    std::vector<V8> result(rows);
    const auto *q = static_cast<const T *>(src);
    for (size_t r = 0; r < rows; ++r) {
      status = spinor(ctl, q + r * ss, result[r]);
      if (status != SPIN8_OK)
        return status;
    }
    for (size_t r = 0; r < rows; ++r)
      std::copy(result[r].begin(), result[r].end(), dst + r * ds);
    return SPIN8_OK;
  } catch (const std::bad_alloc &) {
    return SPIN8_ALLOCATION_FAILED;
  } catch (...) {
    return SPIN8_INVALID_ARGUMENT;
  }
}
} // namespace

extern "C" {
spin8_status spin8_query_v3_generators(double *out) {
  if (!pointer(out, 448 * sizeof(double), alignof(double)))
    return SPIN8_INVALID_ARGUMENT;
  const auto &b = basis();
  if (b.status != SPIN8_OK)
    return b.status;
  for (size_t i = 0; i < 7; ++i)
    std::copy(b.j[i].begin(), b.j[i].end(), out + 64 * i);
  return SPIN8_OK;
}
spin8_status spin8_query_v3_basis_audit(spin8_query_v3_basis_report *out) {
  if (!pointer(out, sizeof(*out), alignof(spin8_query_v3_basis_report)))
    return SPIN8_INVALID_ARGUMENT;
  const auto &b = basis();
  if (b.status != SPIN8_OK)
    return b.status;
  *out = b.report;
  return SPIN8_OK;
}
spin8_status spin8_query_v3_matrix(const double *c, double *out) {
  if (!pointer(out, 64 * sizeof(double), alignof(double)))
    return SPIN8_INVALID_ARGUMENT;
  M8 result{};
  const auto status = matrix(c, result);
  if (status == SPIN8_OK)
    std::copy(result.begin(), result.end(), out);
  return status;
}
spin8_status spin8_query_v3_spinor_from_tangent(const double *a, double *out) {
  if (!pointer(a, 7 * sizeof(double), alignof(double)) ||
      !pointer(out, 8 * sizeof(double), alignof(double)))
    return SPIN8_INVALID_ARGUMENT;
  constexpr double tau = 0.41421356237309504880168872420970;
  double scale = tau;
  for (size_t i = 0; i < 7; ++i) {
    if (!std::isfinite(a[i]))
      return SPIN8_NONFINITE;
    scale = std::max(scale, std::abs(a[i]));
  }
  // Scale before accumulating: even hypot(DBL_MAX,DBL_MAX) overflows.
  double squares = (tau / scale) * (tau / scale);
  for (size_t i = 0; i < 7; ++i)
    squares += (a[i] / scale) * (a[i] / scale);
  const double denominator = std::sqrt(squares);
  V8 c{};
  double norm_squared = 1.;
  for (size_t i = 0; i < 7; ++i) {
    c[i + 1] = tau * (a[i] / scale) / denominator;
    norm_squared += c[i + 1] * c[i + 1];
  }
  c[0] = 1.;
  const double inverse_norm = 1. / std::sqrt(norm_squared);
  for (auto &value : c)
    value *= inverse_norm;
  std::copy(c.begin(), c.end(), out);
  return SPIN8_OK;
}
spin8_status spin8_query_v3_controller_create(
    uint32_t version, size_t width, const double *win, size_t win_bytes,
    const double *bin, size_t bin_bytes, const double *wout, size_t wout_bytes,
    const double *bout, size_t bout_bytes, spin8_query_v3_controller **out) {
  if (!out)
    return SPIN8_INVALID_ARGUMENT;
  *out = nullptr;
  size_t count = 0, bytes = 0;
  if (version != SPIN8_QUERY_V3_VERSION || !width ||
      !product(8, width, count) || !product(count, sizeof(double), bytes) ||
      bytes > win_bytes || 8 * sizeof(double) > bin_bytes ||
      56 * sizeof(double) > wout_bytes || 7 * sizeof(double) > bout_bytes ||
      !pointer(win, bytes, alignof(double)) ||
      !pointer(bin, 8 * sizeof(double), alignof(double)) ||
      !pointer(wout, 56 * sizeof(double), alignof(double)) ||
      !pointer(bout, 7 * sizeof(double), alignof(double)))
    return SPIN8_INVALID_ARGUMENT;
  for (size_t i = 0; i < count; ++i)
    if (!std::isfinite(win[i]))
      return SPIN8_NONFINITE;
  for (size_t i = 0; i < 8; ++i)
    if (!std::isfinite(bin[i]))
      return SPIN8_NONFINITE;
  for (size_t i = 0; i < 56; ++i)
    if (!std::isfinite(wout[i]))
      return SPIN8_NONFINITE;
  for (size_t i = 0; i < 7; ++i)
    if (!std::isfinite(bout[i]))
      return SPIN8_NONFINITE;
  try {
    auto ctl = std::make_unique<spin8_query_v3_controller>();
    ctl->width = width;
    ctl->win.assign(win, win + count);
    std::copy_n(bin, 8, ctl->bin.begin());
    std::copy_n(wout, 56, ctl->wout.begin());
    std::copy_n(bout, 7, ctl->bout.begin());
    ctl->identity = std::all_of(ctl->wout.begin(), ctl->wout.end(),
                                [](double v) { return v == 0.; }) &&
                    std::all_of(ctl->bout.begin(), ctl->bout.end(),
                                [](double v) { return v == 0.; });
    *out = ctl.release();
    return SPIN8_OK;
  } catch (const std::bad_alloc &) {
    return SPIN8_ALLOCATION_FAILED;
  } catch (...) {
    return SPIN8_INVALID_ARGUMENT;
  }
}
void spin8_query_v3_controller_free(spin8_query_v3_controller *p) { delete p; }
uint32_t spin8_query_v3_controller_version(const spin8_query_v3_controller *p) {
  return p ? SPIN8_QUERY_V3_VERSION : 0u;
}
spin8_status spin8_query_v3_controller_apply(const spin8_query_v3_controller *p,
                                             spin8_dtype dtype, const void *src,
                                             size_t sc, size_t ss, void *dst,
                                             size_t dc, size_t ds,
                                             size_t rows) {
  if (!p)
    return SPIN8_INVALID_ARGUMENT;
  if (dtype == SPIN8_F64)
    return apply<double>(p, src, sc, ss, dst, dc, ds, rows);
  if (dtype == SPIN8_F32)
    return apply<float>(p, src, sc, ss, dst, dc, ds, rows);
  return dtype == SPIN8_F16 || dtype == SPIN8_BF16 ? SPIN8_UNSUPPORTED
                                                   : SPIN8_INVALID_ARGUMENT;
}
spin8_status spin8_query_v3_controller_spinors(
    const spin8_query_v3_controller *p, spin8_dtype dtype, const void *src,
    size_t sc, size_t ss, double *dst, size_t dc, size_t ds, size_t rows) {
  if (!p)
    return SPIN8_INVALID_ARGUMENT;
  if (dtype == SPIN8_F64)
    return spinors<double>(p, src, sc, ss, dst, dc, ds, rows);
  if (dtype == SPIN8_F32)
    return spinors<float>(p, src, sc, ss, dst, dc, ds, rows);
  return dtype == SPIN8_F16 || dtype == SPIN8_BF16 ? SPIN8_UNSUPPORTED
                                                   : SPIN8_INVALID_ARGUMENT;
}
spin8_status spin8_query_v3_controller_vjp_f64(
    const spin8_query_v3_controller *ctl, const double *q, size_t qc,
    const double *dy, size_t yc, const spin8_query_v3_gradients_f64 *g) {
  auto status = vjp_spans(ctl, q, qc, dy, yc, g);
  if (status != SPIN8_OK)
    return status;
  try {
    V8 hidden{};
    for (size_t k = 0; k < 8; ++k) {
      double z = ctl->bin[k];
      for (size_t i = 0; i < ctl->width; ++i)
        if (!finite_madd(ctl->win[k * ctl->width + i], q[i], z))
          return SPIN8_NONFINITE;
      hidden[k] = std::tanh(z);
    }
    auto a = ctl->bout;
    for (size_t i = 0; i < 7; ++i)
      for (size_t k = 0; k < 8; ++k)
        if (!finite_madd(ctl->wout[i * 8 + k], hidden[k], a[i]))
          return SPIN8_NONFINITE;
    V8 c{};
    status = spin8_query_v3_spinor_from_tangent(a.data(), c.data());
    if (status != SPIN8_OK)
      return status;
    M8 m{};
    status = matrix(c.data(), m);
    if (status != SPIN8_OK)
      return status;
    const auto &b = basis();
    std::vector<double> dq(ctl->width), dwin(ctl->win.size());
    V8 gc{}, gx{};
    size_t base = 0;
    for (; ctl->width - base >= 8; base += 8) {
      for (size_t i = 0; i < 8; ++i) {
        if (!finite_madd(dy[base + i], q[base + i], gc[0]))
          return SPIN8_NONFINITE;
        for (size_t j = 0; j < 8; ++j)
          if (!finite_madd(m[j * 8 + i], dy[base + j], dq[base + i]))
            return SPIN8_NONFINITE;
      }
      for (size_t t = 0; t < 7; ++t)
        for (size_t i = 0; i < 8; ++i) {
          double jq = 0.;
          for (size_t j = 0; j < 8; ++j)
            if (!finite_madd(b.j[t][i * 8 + j], q[base + j], jq))
              return SPIN8_NONFINITE;
          if (!finite_madd(dy[base + i], jq, gc[t + 1]))
            return SPIN8_NONFINITE;
        }
    }
    for (; base < ctl->width; ++base)
      dq[base] = dy[base];

    // Differentiate both chart normalizations using the same scaled radius
    // as the forward chart. Never form tau^2+||a||^2 at the original scale.
    constexpr double tau = 0.41421356237309504880168872420970;
    double scale = tau, gc_dot_c = 0.;
    for (double value : a)
      scale = std::max(scale, std::abs(value));
    double squares = (tau / scale) * (tau / scale);
    for (double value : a)
      squares += (value / scale) * (value / scale);
    const double radius = std::sqrt(squares);
    const double r = (tau / scale) / radius;
    std::array<double, 7> v{}, gu{}, ga{};
    for (size_t i = 0; i < 8; ++i)
      if (!finite_madd(gc[i], c[i], gc_dot_c))
        return SPIN8_NONFINITE;
    double v_dot_gu = 0.;
    for (size_t i = 0; i < 7; ++i) {
      double term = 0.;
      v[i] = (a[i] / scale) / radius;
      if (!finite_product(c[i + 1], gc_dot_c, term) ||
          !finite_add(gc[i + 1], -term, gu[i]) ||
          !finite_product(gu[i], c[0], gu[i]) ||
          !finite_madd(v[i], gu[i], v_dot_gu))
        return SPIN8_NONFINITE;
    }
    for (size_t i = 0; i < 7; ++i) {
      double term = 0.;
      if (!finite_product(v[i], v_dot_gu, term) ||
          !finite_add(gu[i], -term, ga[i]) || !finite_product(r, ga[i], ga[i]))
        return SPIN8_NONFINITE;
    }
    std::array<double, 56> dwout{};
    for (size_t i = 0; i < 7; ++i)
      for (size_t k = 0; k < 8; ++k)
        if (!finite_product(ga[i], hidden[k], dwout[i * 8 + k]) ||
            !finite_madd(ctl->wout[i * 8 + k], ga[i], gx[k]))
          return SPIN8_NONFINITE;
    for (size_t k = 0; k < 8; ++k) {
      if (!finite_product(gx[k], 1. - hidden[k] * hidden[k], gx[k]))
        return SPIN8_NONFINITE;
      for (size_t i = 0; i < ctl->width; ++i)
        if (!finite_product(gx[k], q[i], dwin[k * ctl->width + i]) ||
            !finite_madd(ctl->win[k * ctl->width + i], gx[k], dq[i]))
          return SPIN8_NONFINITE;
    }
    std::copy(dq.begin(), dq.end(), g->dq);
    std::copy(dwin.begin(), dwin.end(), g->dwin);
    std::copy(gx.begin(), gx.end(), g->dbin);
    std::copy(dwout.begin(), dwout.end(), g->dwout);
    std::copy(ga.begin(), ga.end(), g->dbout);
    return SPIN8_OK;
  } catch (const std::bad_alloc &) {
    return SPIN8_ALLOCATION_FAILED;
  } catch (...) {
    return SPIN8_INVALID_ARGUMENT;
  }
}
} // extern C
