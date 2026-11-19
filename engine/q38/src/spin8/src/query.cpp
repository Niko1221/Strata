#include "spin8/query.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <limits>
#include <new>
#include <vector>

struct spin8_query_controller {
  size_t width;
  std::vector<double> down;
  std::array<double, 56> up;
  bool zero_up;
};

namespace {
using M8 = std::array<double, 64>;
using V8 = std::array<double, 8>;

const std::array<double, 2048> &gamma() {
  static const auto value = [] {
    std::array<double, 2048> result{};
    spin8_clifford_generators(result.data());
    return result;
  }();
  return value;
}

bool multiply(size_t a, size_t b, size_t &out) {
  if (b && a > SIZE_MAX / b)
    return false;
  out = a * b;
  return true;
}

bool span(size_t rows, size_t width, size_t stride, size_t item,
          size_t capacity, size_t &bytes) {
  size_t last = 0;
  if (!rows || !width || stride < width || !multiply(rows - 1, stride, last) ||
      last > SIZE_MAX - width || !multiply(last + width, item, bytes))
    return false;
  return bytes <= capacity;
}

bool overlaps(const void *a, size_t an, const void *b, size_t bn) {
  const auto x = reinterpret_cast<uintptr_t>(a);
  const auto y = reinterpret_cast<uintptr_t>(b);
  if (x > UINTPTR_MAX - an || y > UINTPTR_MAX - bn)
    return true;
  return x < y + bn && y < x + an;
}

template <class T>
spin8_status buffers(const void *src, size_t sc, size_t ss, void *dst,
                     size_t dc, size_t ds, size_t rows, size_t width,
                     size_t &count) {
  size_t sn = 0, dn = 0;
  if (!src || !dst || reinterpret_cast<uintptr_t>(src) % alignof(T) ||
      reinterpret_cast<uintptr_t>(dst) % alignof(T) ||
      !span(rows, width, ss, sizeof(T), sc, sn) ||
      !span(rows, width, ds, sizeof(T), dc, dn) ||
      !multiply(rows, width, count) ||
      reinterpret_cast<uintptr_t>(src) > UINTPTR_MAX - sn ||
      reinterpret_cast<uintptr_t>(dst) > UINTPTR_MAX - dn)
    return SPIN8_INVALID_ARGUMENT;
  if (overlaps(src, sn, dst, dn) && (src != dst || ss != ds))
    return SPIN8_INVALID_ARGUMENT;
  return SPIN8_OK;
}

spin8_status validate_spinor(const double *c) {
  if (!c)
    return SPIN8_INVALID_ARGUMENT;
  double norm = 0;
  for (size_t i = 0; i < 8; ++i) {
    if (!std::isfinite(c[i]))
      return SPIN8_NONFINITE;
    norm = std::hypot(norm, c[i]);
  }
  return std::abs(norm - 1.) <= 1e-13 ? SPIN8_OK : SPIN8_INVALID_ARGUMENT;
}

M8 contraction_matrix(const double *c) {
  M8 result{};
  const auto &g = gamma();
  for (size_t i = 0; i < 8; ++i)
    for (size_t a = 0; a < 8; ++a)
      for (size_t b = 0; b < 8; ++b)
        result[i * 8 + a] += g[i * 256 + a * 16 + 8 + b] * c[b];
  return result;
}

spin8_status matrix(const double *c, M8 &out) {
  auto status = validate_spinor(c);
  if (status != SPIN8_OK)
    return status;
  const auto v = contraction_matrix(c);
  static const V8 c0 = {1., 0., 0., 0., 0., 0., 0., 0.};
  static const M8 base = contraction_matrix(c0.data());
  M8 result{};
  for (size_t i = 0; i < 8; ++i)
    for (size_t j = 0; j < 8; ++j)
      for (size_t a = 0; a < 8; ++a)
        result[i * 8 + j] += v[i * 8 + a] * base[j * 8 + a];
  spin8_matrix_audit audit{};
  status = spin8_audit_matrix(result.data(), &audit);
  if (status != SPIN8_OK)
    return status;
  if (audit.det_sign != 1 || audit.max_error > 1e-12 ||
      audit.ef_normalized > 1e-12 || std::abs(audit.log_abs_det) > 1e-12)
    return SPIN8_AUDIT_FAILED;
  out = result;
  return SPIN8_OK;
}

template <class T>
spin8_status apply_matrix_rows(const T *src, size_t ss, T *dst, size_t ds,
                               size_t rows, size_t width, const M8 *matrices,
                               bool zero_controller) {
  size_t count = 0;
  if (!multiply(rows, width, count))
    return SPIN8_INVALID_ARGUMENT;
  try {
    std::vector<T> result(count);
    for (size_t r = 0; r < rows; ++r) {
      const auto &m = matrices[zero_controller ? 0 : r];
      for (size_t base = 0; base < width;) {
        if (width - base < 8) {
          for (size_t j = base; j < width; ++j)
            result[r * width + j] = src[r * ss + j];
          break;
        }
        for (size_t i = 0; i < 8; ++i) {
          double sum = 0;
          for (size_t j = 0; j < 8; ++j)
            sum += m[i * 8 + j] * static_cast<double>(src[r * ss + base + j]);
          const auto converted = static_cast<T>(sum);
          if (!std::isfinite(sum) || !std::isfinite(converted))
            return SPIN8_NONFINITE;
          result[r * width + base + i] = converted;
        }
        base += 8;
      }
    }
    for (size_t r = 0; r < rows; ++r)
      std::copy_n(result.data() + r * width, width, dst + r * ds);
    return SPIN8_OK;
  } catch (const std::bad_alloc &) {
    return SPIN8_ALLOCATION_FAILED;
  } catch (...) {
    return SPIN8_INVALID_ARGUMENT;
  }
}

template <class T>
spin8_status transform(const double *c, const void *src, size_t sc, size_t ss,
                       void *dst, size_t dc, size_t ds, size_t rows,
                       size_t width) {
  size_t count = 0;
  auto status = buffers<T>(src, sc, ss, dst, dc, ds, rows, width, count);
  if (status != SPIN8_OK)
    return status;
  M8 m{};
  status = matrix(c, m);
  if (status != SPIN8_OK)
    return status;
  const auto *input = static_cast<const T *>(src);
  for (size_t r = 0; r < rows; ++r)
    for (size_t j = 0; j < width; ++j)
      if (!std::isfinite(input[r * ss + j]))
        return SPIN8_NONFINITE;
  return apply_matrix_rows(input, ss, static_cast<T *>(dst), ds, rows, width,
                           &m, true);
}

template <class T>
spin8_status controller_apply(const spin8_query_controller *ctl,
                              const void *src, size_t sc, size_t ss, void *dst,
                              size_t dc, size_t ds, size_t rows) {
  size_t count = 0;
  const size_t width = ctl->width;
  auto status = buffers<T>(src, sc, ss, dst, dc, ds, rows, width, count);
  if (status != SPIN8_OK)
    return status;
  const auto *input = static_cast<const T *>(src);
  for (size_t r = 0; r < rows; ++r)
    for (size_t j = 0; j < width; ++j)
      if (!std::isfinite(input[r * ss + j]))
        return SPIN8_NONFINITE;
  if (ctl->zero_up) {
    try {
      std::vector<T> result(count);
      for (size_t r = 0; r < rows; ++r)
        std::copy_n(input + r * ss, width, result.data() + r * width);
      for (size_t r = 0; r < rows; ++r)
        std::copy_n(result.data() + r * width, width,
                    static_cast<T *>(dst) + r * ds);
      return SPIN8_OK;
    } catch (const std::bad_alloc &) {
      return SPIN8_ALLOCATION_FAILED;
    } catch (...) {
      return SPIN8_INVALID_ARGUMENT;
    }
  }
  try {
    std::vector<M8> matrices(rows);
    std::vector<double> normalized(width);
    for (size_t r = 0; r < rows; ++r) {
      double scale = 0;
      for (size_t j = 0; j < width; ++j)
        scale =
            std::max(scale, std::abs(static_cast<double>(input[r * ss + j])));
      if (scale <= 1e-3) {
        double mean = 0;
        for (size_t j = 0; j < width; ++j) {
          double v = input[r * ss + j];
          mean += v * v / static_cast<double>(width);
        }
        const double denom = std::sqrt(mean + 1e-6);
        for (size_t j = 0; j < width; ++j)
          normalized[j] = static_cast<double>(input[r * ss + j]) / denom;
      } else {
        double mean = 0;
        for (size_t j = 0; j < width; ++j) {
          const double v = static_cast<double>(input[r * ss + j]) / scale;
          mean += v * v / static_cast<double>(width);
        }
        const double denom = std::sqrt(mean + (1e-6 / scale) / scale);
        for (size_t j = 0; j < width; ++j)
          normalized[j] =
              (static_cast<double>(input[r * ss + j]) / scale) / denom;
      }
      V8 hidden{};
      for (size_t k = 0; k < 8; ++k) {
        double z = 0;
        for (size_t j = 0; j < width; ++j)
          z += ctl->down[k * width + j] * normalized[j];
        if (!std::isfinite(z))
          return SPIN8_NONFINITE;
        hidden[k] = std::tanh(z);
      }
      std::array<double, 7> tangent{};
      for (size_t i = 0; i < 7; ++i) {
        for (size_t k = 0; k < 8; ++k)
          tangent[i] += ctl->up[i * 8 + k] * hidden[k];
        if (!std::isfinite(tangent[i]))
          return SPIN8_NONFINITE;
      }
      V8 spinor{};
      status = spin8_query_spinor_from_tangent(tangent.data(), spinor.data());
      if (status != SPIN8_OK)
        return status;
      status = matrix(spinor.data(), matrices[r]);
      if (status != SPIN8_OK)
        return status;
    }
    return apply_matrix_rows(input, ss, static_cast<T *>(dst), ds, rows, width,
                             matrices.data(), false);
  } catch (const std::bad_alloc &) {
    return SPIN8_ALLOCATION_FAILED;
  } catch (...) {
    return SPIN8_INVALID_ARGUMENT;
  }
}
} // namespace

extern "C" {
spin8_status spin8_query_contract(const double *v, const double *p,
                                  const double *m, double *out) {
  if (!v || !p || !m || !out)
    return SPIN8_INVALID_ARGUMENT;
  for (size_t i = 0; i < 8; ++i)
    if (!std::isfinite(v[i]) || !std::isfinite(p[i]) || !std::isfinite(m[i]))
      return SPIN8_NONFINITE;
  const auto &g = gamma();
  double value = 0;
  for (size_t i = 0; i < 8; ++i)
    for (size_t a = 0; a < 8; ++a)
      for (size_t b = 0; b < 8; ++b)
        value += v[i] * p[a] * g[i * 256 + a * 16 + 8 + b] * m[b];
  if (!std::isfinite(value))
    return SPIN8_NONFINITE;
  *out = value;
  return SPIN8_OK;
}

spin8_status spin8_query_matrix(const double *c, double *out) {
  if (!out)
    return SPIN8_INVALID_ARGUMENT;
  M8 result{};
  const auto status = matrix(c, result);
  if (status == SPIN8_OK)
    std::copy(result.begin(), result.end(), out);
  return status;
}

spin8_status spin8_query_matrix_audit(const double *c,
                                      spin8_matrix_audit *out) {
  if (!out)
    return SPIN8_INVALID_ARGUMENT;
  M8 result{};
  const auto status = matrix(c, result);
  if (status != SPIN8_OK)
    return status;
  return spin8_audit_matrix(result.data(), out);
}

spin8_status spin8_query_spinor_from_tangent(const double *h, double *out) {
  if (!h || !out)
    return SPIN8_INVALID_ARGUMENT;
  double scale = 0;
  for (size_t i = 0; i < 7; ++i) {
    if (!std::isfinite(h[i]))
      return SPIN8_NONFINITE;
    scale = std::max(scale, std::abs(h[i]));
  }
  constexpr double pi_over_eight = 0.39269908169872415480783042290994;
  std::array<double, 7> u{};
  double theta = 0;
  if (scale < 1.) {
    double norm = 0;
    for (size_t i = 0; i < 7; ++i)
      norm = std::hypot(norm, h[i]);
    const double factor = pi_over_eight / std::hypot(1., norm);
    for (size_t i = 0; i < 7; ++i)
      u[i] = h[i] * factor;
  } else {
    double squares = 0;
    for (size_t i = 0; i < 7; ++i) {
      const double v = h[i] / scale;
      squares += v * v;
    }
    const double denominator = std::sqrt(squares + (1. / scale) / scale);
    for (size_t i = 0; i < 7; ++i)
      u[i] = pi_over_eight * (h[i] / scale) / denominator;
  }
  for (double v : u)
    theta = std::hypot(theta, v);
  const double theta2 = theta * theta;
  const double sinc = theta < 1e-4 ? 1. - theta2 / 6. + theta2 * theta2 / 120.
                                   : std::sin(theta) / theta;
  V8 result{};
  result[0] = std::cos(theta);
  for (size_t i = 0; i < 7; ++i)
    result[i + 1] = sinc * u[i];
  std::copy(result.begin(), result.end(), out);
  return SPIN8_OK;
}

spin8_status spin8_query_transform(const double *c, spin8_dtype dtype,
                                   const void *src, size_t sc, size_t ss,
                                   void *dst, size_t dc, size_t ds, size_t rows,
                                   size_t width) {
  if (dtype == SPIN8_F64)
    return transform<double>(c, src, sc, ss, dst, dc, ds, rows, width);
  if (dtype == SPIN8_F32)
    return transform<float>(c, src, sc, ss, dst, dc, ds, rows, width);
  return dtype == SPIN8_F16 || dtype == SPIN8_BF16 ? SPIN8_UNSUPPORTED
                                                   : SPIN8_INVALID_ARGUMENT;
}

spin8_status spin8_query_controller_create(size_t width, const double *down,
                                           size_t down_bytes, const double *up,
                                           size_t up_bytes,
                                           spin8_query_controller **out) {
  if (!out)
    return SPIN8_INVALID_ARGUMENT;
  *out = nullptr;
  size_t count = 0, required = 0;
  if (!width || !down || !up ||
      reinterpret_cast<uintptr_t>(down) % alignof(double) ||
      reinterpret_cast<uintptr_t>(up) % alignof(double) ||
      !multiply(8, width, count) ||
      !multiply(count, sizeof(double), required) || required > down_bytes ||
      sizeof(double) * 56 > up_bytes ||
      reinterpret_cast<uintptr_t>(down) > UINTPTR_MAX - required ||
      reinterpret_cast<uintptr_t>(up) > UINTPTR_MAX - sizeof(double) * 56)
    return SPIN8_INVALID_ARGUMENT;
  for (size_t i = 0; i < count; ++i)
    if (!std::isfinite(down[i]))
      return SPIN8_NONFINITE;
  for (size_t i = 0; i < 56; ++i)
    if (!std::isfinite(up[i]))
      return SPIN8_NONFINITE;
  try {
    auto *p = new spin8_query_controller;
    try {
      p->width = width;
      p->down.assign(down, down + count);
      std::copy_n(up, 56, p->up.begin());
      p->zero_up = std::all_of(p->up.begin(), p->up.end(),
                               [](double v) { return v == 0.; });
    } catch (...) {
      delete p;
      throw;
    }
    *out = p;
    return SPIN8_OK;
  } catch (const std::bad_alloc &) {
    return SPIN8_ALLOCATION_FAILED;
  } catch (...) {
    return SPIN8_INVALID_ARGUMENT;
  }
}

void spin8_query_controller_free(spin8_query_controller *p) { delete p; }

spin8_status spin8_query_controller_apply(const spin8_query_controller *p,
                                          spin8_dtype dtype, const void *src,
                                          size_t sc, size_t ss, void *dst,
                                          size_t dc, size_t ds, size_t rows) {
  if (!p)
    return SPIN8_INVALID_ARGUMENT;
  if (dtype == SPIN8_F64)
    return controller_apply<double>(p, src, sc, ss, dst, dc, ds, rows);
  if (dtype == SPIN8_F32)
    return controller_apply<float>(p, src, sc, ss, dst, dc, ds, rows);
  return dtype == SPIN8_F16 || dtype == SPIN8_BF16 ? SPIN8_UNSUPPORTED
                                                   : SPIN8_INVALID_ARGUMENT;
}
}
