#include "spin8/spin8.h"
#include <algorithm>
#include <array>
#include <cmath>
#include <cstring>
#include <limits>
#include <new>
#include <vector>

namespace {
template <size_t N> using Mat = std::array<double, N * N>;
template <size_t N> Mat<N> identity() noexcept {
  Mat<N> r{};
  for (size_t i = 0; i < N; ++i)
    r[i * N + i] = 1;
  return r;
}
template <size_t N> Mat<N> mul(const Mat<N> &a, const Mat<N> &b) noexcept {
  Mat<N> r{};
  for (size_t i = 0; i < N; ++i)
    for (size_t j = 0; j < N; ++j)
      for (size_t k = 0; k < N; ++k)
        r[i * N + j] += a[i * N + k] * b[k * N + j];
  return r;
}
template <size_t N> Mat<N> transpose(const Mat<N> &a) noexcept {
  Mat<N> r{};
  for (size_t i = 0; i < N; ++i)
    for (size_t j = 0; j < N; ++j)
      r[i * N + j] = a[j * N + i];
  return r;
}
using M8 = Mat<8>;
using M16 = Mat<16>;
std::array<M16, 8> generators() noexcept {
  std::array<M16, 8> g{};
  for (int i = 0; i < 8; ++i) {
    g[0][i * 16 + 8 + i] = 1;
    g[0][(8 + i) * 16 + i] = 1;
  }
  const int triples[7][3] = {{1, 2, 3}, {1, 4, 5}, {1, 7, 6}, {2, 4, 6},
                             {2, 5, 7}, {3, 4, 7}, {3, 6, 5}};
  for (int a = 1; a < 8; ++a) {
    M8 l{};
    l[static_cast<size_t>(a) * 8] = 1;
    l[a] = -1;
    for (const auto &t : triples)
      for (int n = 0; n < 3; ++n) {
        int x = t[n], y = t[(n + 1) % 3], z = t[(n + 2) % 3];
        if (x == a) {
          l[z * 8 + y] = 1;
          l[y * 8 + z] = -1;
        }
      }
    for (int i = 0; i < 8; ++i)
      for (int j = 0; j < 8; ++j) {
        g[a][i * 16 + 8 + j] = l[i * 8 + j];
        g[a][(8 + i) * 16 + j] = -l[i * 8 + j];
      }
  }
  M16 chir = identity<16>();
  for (const auto &a : g)
    chir = mul<16>(chir, a);
  // Put positive chirality first, without an eigensolver or floating-point
  // basis.
  if (chir[0] < 0)
    for (auto &a : g) {
      M16 b{};
      for (int i = 0; i < 16; ++i)
        for (int j = 0; j < 16; ++j)
          b[i * 16 + j] = a[((i + 8) % 16) * 16 + (j + 8) % 16];
      a = b;
    }
  return g;
}
const auto gamma = generators();
bool view_ok(spin8_view v) {
  return v >= SPIN8_VECTOR && v <= SPIN8_SPIN_MINUS;
}
bool hadamard_config(size_t block, const int8_t *signs) {
  if (!block || block > 1024 || (block & (block - 1)))
    return false;
  if (signs)
    for (size_t i = 0; i < block; ++i)
      if (signs[i] != 1 && signs[i] != -1)
        return false;
  return true;
}
bool product(size_t a, size_t b, size_t &out) {
  if (b && a > SIZE_MAX / b)
    return false;
  out = a * b;
  return true;
}
bool extent(size_t rows, size_t cols, size_t stride, size_t item,
            size_t &bytes) {
  size_t n = 0;
  if (stride < cols || !product(rows ? rows - 1 : 0, stride, n) ||
      n > SIZE_MAX - cols)
    return false;
  return product(rows ? n + cols : 0, item, bytes);
}
bool overlap(const void *a, size_t an, const void *b, size_t bn) {
  const auto x = reinterpret_cast<uintptr_t>(a),
             y = reinterpret_cast<uintptr_t>(b);
  if (x > UINTPTR_MAX - an || y > UINTPTR_MAX - bn)
    return true;
  return an && bn && x < y + bn && y < x + an;
}
spin8_status weights_check(const double *w, std::array<double, 3> &r) {
  r = {1. / 3, 1. / 3, 1. / 3};
  if (!w)
    return SPIN8_OK;
  double sum = 0;
  for (int i = 0; i < 3; ++i) {
    if (!std::isfinite(w[i]))
      return SPIN8_NONFINITE;
    if (w[i] < 0)
      return SPIN8_INVALID_ARGUMENT;
    r[i] = w[i];
    sum += w[i];
  }
  return std::isfinite(sum) && std::abs(sum - 1) <= 1e-12
             ? SPIN8_OK
             : SPIN8_INVALID_ARGUMENT;
}
} // namespace
struct spin8_plan {
  std::array<M8, 3> m;
};

extern "C" {
const char *spin8_status_string(spin8_status s) {
  switch (s) {
  case SPIN8_OK:
    return "ok";
  case SPIN8_INVALID_ARGUMENT:
    return "invalid argument";
  case SPIN8_NONFINITE:
    return "nonfinite";
  case SPIN8_UNSUPPORTED:
    return "unsupported";
  case SPIN8_ALLOCATION_FAILED:
    return "allocation failed";
  case SPIN8_AUDIT_FAILED:
    return "audit failed";
  case SPIN8_INVALIDATED:
    return "invalidated";
  default:
    return "unknown status";
  }
}
double spin8_precision_tolerance(spin8_dtype t) {
  switch (t) {
  case SPIN8_F64:
    return 1e-12;
  case SPIN8_F32:
    return 1e-5;
  case SPIN8_F16:
    return 2e-3;
  case SPIN8_BF16:
    return 2e-2;
  default:
    return 0;
  }
}
spin8_status spin8_clifford_generators(double *out) {
  if (!out)
    return SPIN8_INVALID_ARGUMENT;
  for (size_t i = 0; i < 8; ++i)
    std::copy(gamma[i].begin(), gamma[i].end(), out + i * 256);
  return SPIN8_OK;
}
spin8_status spin8_audit_matrix(const double *m, spin8_matrix_audit *out) {
  if (!m || !out)
    return SPIN8_INVALID_ARGUMENT;
  M8 a{};
  std::copy(m, m + 64, a.begin());
  for (double x : a)
    if (!std::isfinite(x))
      return SPIN8_NONFINITE;
  auto gram = mul<8>(transpose<8>(a), a);
  for (double x : gram)
    if (!std::isfinite(x))
      return SPIN8_NONFINITE;
  spin8_matrix_audit r{};
  double sq = 0;
  for (size_t i = 0; i < 64; ++i) {
    double e = gram[i] - ((i / 8 == i % 8) ? 1. : 0.);
    sq += e * e;
    r.max_error = std::max(r.max_error, std::abs(e));
  }
  r.ef_normalized = std::sqrt(sq / 8.);
  r.det_sign = 1;
  // Partial-pivot LU yields an independent determinant check; never abs(det)=1.
  for (int k = 0; k < 8; ++k) {
    int p = k;
    for (int i = k + 1; i < 8; ++i)
      if (std::abs(a[i * 8 + k]) > std::abs(a[p * 8 + k]))
        p = i;
    if (a[p * 8 + k] == 0) {
      r.det_sign = 0;
      r.log_abs_det = -std::numeric_limits<double>::infinity();
      break;
    }
    if (p != k) {
      for (int j = 0; j < 8; ++j)
        std::swap(a[p * 8 + j], a[k * 8 + j]);
      r.det_sign = -r.det_sign;
    }
    double pivot = a[k * 8 + k];
    if (pivot < 0)
      r.det_sign = -r.det_sign;
    r.log_abs_det += std::log(std::abs(pivot));
    for (int i = k + 1; i < 8; ++i) {
      double f = a[i * 8 + k] / pivot;
      for (int j = k + 1; j < 8; ++j)
        a[i * 8 + j] -= f * a[k * 8 + j];
    }
  }
  if (!std::isfinite(r.ef_normalized) || !std::isfinite(r.max_error) ||
      (r.det_sign && !std::isfinite(r.log_abs_det)))
    return SPIN8_NONFINITE;
  *out = r;
  return SPIN8_OK;
}
spin8_status spin8_plan_audit(const spin8_plan *p, spin8_audit_report *out) {
  if (!p || !out)
    return SPIN8_INVALID_ARGUMENT;
  spin8_audit_report r{};
  M16 u{};
  for (int v = 0; v < 3; ++v) {
    auto s = spin8_audit_matrix(p->m[v].data(), &r.views[v]);
    if (s != SPIN8_OK)
      return s;
  }
  for (int b = 0; b < 2; ++b)
    for (int i = 0; i < 8; ++i)
      for (int j = 0; j < 8; ++j)
        u[(i + 8 * b) * 16 + j + 8 * b] = p->m[b + 1][i * 8 + j];
  for (int a = 0; a < 8; ++a) {
    auto lhs = mul<16>(mul<16>(u, gamma[a]), transpose<16>(u));
    for (int z = 0; z < 256; ++z) {
      double rhs = 0;
      for (int b = 0; b < 8; ++b)
        rhs += p->m[0][b * 8 + a] * gamma[b][z];
      r.triality_max_error =
          std::max(r.triality_max_error, std::abs(lhs[z] - rhs));
    }
  }
  *out = r;
  for (const auto &a : r.views)
    if (a.ef_normalized > 1e-12 || a.max_error > 1e-12 || a.det_sign != 1 ||
        std::abs(a.log_abs_det) > 1e-12)
      return SPIN8_AUDIT_FAILED;
  return r.triality_max_error <= 1e-12 ? SPIN8_OK : SPIN8_AUDIT_FAILED;
}
spin8_status spin8_plan_create(const spin8_plane *planes, size_t count,
                               spin8_plan **out) {
  if (!out)
    return SPIN8_INVALID_ARGUMENT;
  *out = nullptr;
  if ((count && !planes) || count > SIZE_MAX / sizeof(spin8_plane))
    return SPIN8_INVALID_ARGUMENT;
  for (size_t i = 0; i < count; ++i) {
    if (planes[i].a >= 8 || planes[i].b >= 8 || planes[i].a == planes[i].b)
      return SPIN8_INVALID_ARGUMENT;
    if (!std::isfinite(planes[i].angle))
      return SPIN8_NONFINITE;
  }
  try {
    auto p = new spin8_plan;
    p->m[0] = identity<8>();
    M16 u = identity<16>();
    for (size_t i = 0; i < count; ++i) {
      const auto &f = planes[i];
      double c = std::cos(f.angle / 2), s = std::sin(f.angle / 2);
      auto biv = mul<16>(gamma[f.a], gamma[f.b]);
      auto rotor = identity<16>();
      for (int j = 0; j < 256; ++j)
        rotor[j] = c * rotor[j] - s * biv[j];
      u = mul<16>(rotor, u);
      M8 r = identity<8>();
      c = std::cos(f.angle);
      s = std::sin(f.angle);
      r[f.a * 8 + f.a] = c;
      r[f.b * 8 + f.b] = c;
      r[f.b * 8 + f.a] = s;
      r[f.a * 8 + f.b] = -s;
      p->m[0] = mul<8>(r, p->m[0]);
    }
    for (int b = 0; b < 2; ++b)
      for (int i = 0; i < 8; ++i)
        for (int j = 0; j < 8; ++j)
          p->m[b + 1][i * 8 + j] = u[(i + b * 8) * 16 + j + b * 8];
    spin8_audit_report audit{};
    auto status = spin8_plan_audit(p, &audit);
    if (status != SPIN8_OK) {
      delete p;
      return status;
    }
    *out = p;
    return SPIN8_OK;
  } catch (const std::bad_alloc &) {
    return SPIN8_ALLOCATION_FAILED;
  } catch (...) {
    return SPIN8_INVALID_ARGUMENT;
  }
}
void spin8_plan_free(spin8_plan *p) { delete p; }
spin8_status spin8_plan_matrix(const spin8_plan *p, spin8_view v, double *out) {
  if (!p || !out || !view_ok(v))
    return SPIN8_INVALID_ARGUMENT;
  std::copy(p->m[v].begin(), p->m[v].end(), out);
  return SPIN8_OK;
}
spin8_status spin8_hadamard_matrix(uint8_t mask, int inv, double *out) {
  if (!out || (inv != 0 && inv != 1))
    return SPIN8_INVALID_ARGUMENT;
  const double scale = 1 / std::sqrt(8.);
  for (int i = 0; i < 8; ++i)
    for (int j = 0; j < 8; ++j) {
      int bits = i & j, parity = 0;
      while (bits) {
        parity ^= 1;
        bits &= bits - 1;
      }
      double h = parity ? -scale : scale;
      int sign = ((mask >> (inv ? i : j)) & 1) ? -1 : 1;
      out[i * 8 + j] = h * sign;
    }
  return SPIN8_OK;
}
}

namespace {
template <class T>
spin8_status transform_impl(const M8 &m, const T *src, size_t ss, T *dst,
                            size_t ds, size_t rows, size_t cols) {
  size_t sn = 0, dn = 0;
  if (!src || !dst || !extent(rows, cols, ss, sizeof(T), sn) ||
      !extent(rows, cols, ds, sizeof(T), dn))
    return SPIN8_INVALID_ARGUMENT;
  if (overlap(src, sn, dst, dn) && (src != dst || ss != ds))
    return SPIN8_INVALID_ARGUMENT;
  for (size_t r = 0; r < rows; ++r)
    for (size_t c = 0; c < cols; ++c)
      if (!std::isfinite(src[r * ss + c]))
        return SPIN8_NONFINITE;
  // Stage results so overflow and allocation failures leave output untouched.
  size_t n = 0;
  if (!product(rows, cols, n))
    return SPIN8_INVALID_ARGUMENT;
  try {
    std::vector<T> tmp(n);
    for (size_t r = 0; r < rows; ++r) {
      size_t c = 0;
      for (; cols - c >= 8; c += 8)
        for (size_t i = 0; i < 8; ++i) {
          double v = 0;
          for (size_t j = 0; j < 8; ++j)
            v += m[i * 8 + j] * src[r * ss + c + j];
          tmp[r * cols + c + i] = static_cast<T>(v);
          if (!std::isfinite(tmp[r * cols + c + i]))
            return SPIN8_NONFINITE;
        }
      for (; c < cols; ++c)
        tmp[r * cols + c] = src[r * ss + c];
    }
    for (size_t r = 0; r < rows; ++r)
      std::copy_n(tmp.data() + r * cols, cols, dst + r * ds);
    return SPIN8_OK;
  } catch (const std::bad_alloc &) {
    return SPIN8_ALLOCATION_FAILED;
  } catch (...) {
    return SPIN8_INVALID_ARGUMENT;
  }
}
template <class T>
spin8_status hadamard_block_impl(const T *src, size_t ss, T *dst, size_t ds,
                                 size_t rows, size_t cols, size_t block,
                                 const int8_t *signs, int inverse) {
  size_t sn = 0, dn = 0, count = 0;
  if (!src || !dst || !extent(rows, cols, ss, sizeof(T), sn) ||
      !extent(rows, cols, ds, sizeof(T), dn) || !product(rows, cols, count))
    return SPIN8_INVALID_ARGUMENT;
  if (overlap(src, sn, dst, dn) && (src != dst || ss != ds))
    return SPIN8_INVALID_ARGUMENT;
  for (size_t r = 0; r < rows; ++r)
    for (size_t c = 0; c < cols; ++c)
      if (!std::isfinite(src[r * ss + c]))
        return SPIN8_NONFINITE;
  try {
    std::vector<T> result(count);
    std::vector<double> work(block);
    const double scale = 1 / std::sqrt(static_cast<double>(block));
    for (size_t r = 0; r < rows; ++r) {
      size_t base = 0;
      for (; cols - base >= block; base += block) {
        for (size_t i = 0; i < block; ++i)
          work[i] =
              src[r * ss + base + i] * ((!inverse && signs) ? signs[i] : 1);
        for (size_t step = 1; step < block; step *= 2)
          for (size_t start = 0; start < block; start += 2 * step)
            for (size_t i = 0; i < step; ++i) {
              double a = work[start + i], b = work[start + i + step];
              work[start + i] = a + b;
              work[start + i + step] = a - b;
            }
        for (size_t i = 0; i < block; ++i) {
          result[r * cols + base + i] = static_cast<T>(
              work[i] * scale * ((inverse && signs) ? signs[i] : 1));
          if (!std::isfinite(result[r * cols + base + i]))
            return SPIN8_NONFINITE;
        }
      }
      for (; base < cols; ++base)
        result[r * cols + base] = src[r * ss + base];
    }
    for (size_t r = 0; r < rows; ++r)
      std::copy_n(result.data() + r * cols, cols, dst + r * ds);
    return SPIN8_OK;
  } catch (const std::bad_alloc &) {
    return SPIN8_ALLOCATION_FAILED;
  } catch (...) {
    return SPIN8_INVALID_ARGUMENT;
  }
}
spin8_status dispatch(const M8 &m, spin8_dtype t, const void *s, size_t ss,
                      void *d, size_t ds, size_t r, size_t c) {
  if (t == SPIN8_F64)
    return transform_impl(m, static_cast<const double *>(s), ss,
                          static_cast<double *>(d), ds, r, c);
  if (t == SPIN8_F32)
    return transform_impl(m, static_cast<const float *>(s), ss,
                          static_cast<float *>(d), ds, r, c);
  return t == SPIN8_F16 || t == SPIN8_BF16 ? SPIN8_UNSUPPORTED
                                           : SPIN8_INVALID_ARGUMENT;
}
template <class T>
spin8_status reduce_impl(const void *const *views, size_t ss, T *dst, size_t ds,
                         size_t rows, size_t cols,
                         const std::array<double, 3> &w) {
  size_t sn = 0, dn = 0, n = 0;
  if (!views || !dst || !extent(rows, cols, ss, sizeof(T), sn) ||
      !extent(rows, cols, ds, sizeof(T), dn) || !product(rows, cols, n))
    return SPIN8_INVALID_ARGUMENT;
  for (int v = 0; v < 3; ++v)
    if (!views[v] ||
        (overlap(views[v], sn, dst, dn) && (views[v] != dst || ss != ds)))
      return SPIN8_INVALID_ARGUMENT;
  try {
    std::vector<T> result(n);
    for (size_t r = 0; r < rows; ++r)
      for (size_t c = 0; c < cols; ++c) {
        double sum = 0;
        for (int v = 0; v < 3; ++v) {
          double x = static_cast<const T *>(views[v])[r * ss + c];
          if (!std::isfinite(x))
            return SPIN8_NONFINITE;
          sum += w[v] * x;
        }
        result[r * cols + c] = static_cast<T>(sum);
        if (!std::isfinite(result[r * cols + c]))
          return SPIN8_NONFINITE;
      }
    for (size_t r = 0; r < rows; ++r)
      std::copy_n(result.data() + r * cols, cols, dst + r * ds);
    return SPIN8_OK;
  } catch (const std::bad_alloc &) {
    return SPIN8_ALLOCATION_FAILED;
  } catch (...) {
    return SPIN8_INVALID_ARGUMENT;
  }
}
template <class T>
spin8_status reference(const spin8_plan *p, const T *a, size_t m, const T *b,
                       size_t n, size_t k, const T *v, size_t dv,
                       const uint8_t *mask, const double *w, T *out,
                       bool attention) {
  std::array<double, 3> weight{};
  auto ws = weights_check(w, weight);
  if (ws != SPIN8_OK)
    return ws;
  size_t ac = 0, bc = 0, oc = 0, vc = 0, mn = 0;
  if (!p || !a || !b || !out || !k || !m || !n || !product(m, k, ac) ||
      !product(n, k, bc) || !product(m, n, mn) ||
      !product(m, attention ? dv : n, oc) || !product(n, dv, vc) ||
      (attention && (!v || !dv)))
    return SPIN8_INVALID_ARGUMENT;
  size_t ab = 0, bb = 0, ob = 0, vb = 0;
  if (!product(ac, sizeof(T), ab) || !product(bc, sizeof(T), bb) ||
      !product(oc, sizeof(T), ob) || !product(vc, sizeof(T), vb))
    return SPIN8_INVALID_ARGUMENT;
  if (overlap(out, ob, a, ab) || overlap(out, ob, b, bb) ||
      (attention && overlap(out, ob, v, vb)))
    return SPIN8_INVALID_ARGUMENT;
  for (size_t i = 0; i < ac; ++i)
    if (!std::isfinite(a[i]))
      return SPIN8_NONFINITE;
  for (size_t i = 0; i < bc; ++i)
    if (!std::isfinite(b[i]))
      return SPIN8_NONFINITE;
  if (attention) {
    for (size_t i = 0; i < vc; ++i)
      if (!std::isfinite(v[i]))
        return SPIN8_NONFINITE;
    if (mask)
      for (size_t i = 0; i < m; ++i) {
        bool any = false;
        for (size_t j = 0; j < n; ++j)
          any = any || mask[i * n + j] != 0;
        if (!any)
          return SPIN8_INVALID_ARGUMENT;
      }
  }
  try {
    std::vector<double> logits(mn, 0), ta(ac), tb(bc);
    std::vector<T> result(oc);
    // Use FP64 accumulation for both exposed storage types.
    for (int view = 0; view < 3; ++view) {
      for (size_t r = 0; r < m; ++r)
        for (size_t c = 0; c < k; ++c) {
          size_t base = (c / 8) * 8;
          double value = a[r * k + c];
          if (k - base >= 8) {
            value = 0;
            for (size_t j = 0; j < 8; ++j)
              value += p->m[view][(c % 8) * 8 + j] * a[r * k + base + j];
          }
          ta[r * k + c] = value;
        }
      for (size_t r = 0; r < n; ++r)
        for (size_t c = 0; c < k; ++c) {
          size_t base = (c / 8) * 8;
          double value = b[r * k + c];
          if (k - base >= 8) {
            value = 0;
            for (size_t j = 0; j < 8; ++j)
              value += p->m[view][(c % 8) * 8 + j] * b[r * k + base + j];
          }
          tb[r * k + c] = value;
        }
      for (size_t i = 0; i < m; ++i)
        for (size_t j = 0; j < n; ++j) {
          double dot = 0;
          for (size_t c = 0; c < k; ++c)
            dot += ta[i * k + c] * tb[j * k + c];
          logits[i * n + j] += weight[view] * dot;
        }
    }
    if (attention) {
      const double scale = 1 / std::sqrt(static_cast<double>(k));
      for (size_t i = 0; i < m; ++i) {
        double mx = -std::numeric_limits<double>::infinity();
        for (size_t j = 0; j < n; ++j)
          if (!mask || mask[i * n + j]) {
            logits[i * n + j] *= scale;
            if (!std::isfinite(logits[i * n + j]))
              return SPIN8_NONFINITE;
            mx = std::max(mx, logits[i * n + j]);
          }
        double sum = 0;
        for (size_t j = 0; j < n; ++j) {
          logits[i * n + j] =
              (!mask || mask[i * n + j]) ? std::exp(logits[i * n + j] - mx) : 0;
          sum += logits[i * n + j];
        }
        for (size_t c = 0; c < dv; ++c) {
          double value = 0;
          for (size_t j = 0; j < n; ++j)
            value += logits[i * n + j] / sum * v[j * dv + c];
          result[i * dv + c] = static_cast<T>(value);
        }
      }
    } else
      for (size_t i = 0; i < oc; ++i)
        result[i] = static_cast<T>(logits[i]);
    for (T x : result)
      if (!std::isfinite(x))
        return SPIN8_NONFINITE;
    std::copy(result.begin(), result.end(), out);
    return SPIN8_OK;
  } catch (const std::bad_alloc &) {
    return SPIN8_ALLOCATION_FAILED;
  } catch (...) {
    return SPIN8_INVALID_ARGUMENT;
  }
}
} // namespace
extern "C" {
spin8_status spin8_hadamard_block(spin8_dtype t, const void *s, size_t ss,
                                  void *d, size_t ds, size_t rows, size_t cols,
                                  size_t block, const int8_t *signs,
                                  int inverse) {
  if (!hadamard_config(block, signs) || (inverse != 0 && inverse != 1))
    return SPIN8_INVALID_ARGUMENT;
  if (t == SPIN8_F64)
    return hadamard_block_impl(static_cast<const double *>(s), ss,
                               static_cast<double *>(d), ds, rows, cols, block,
                               signs, inverse);
  if (t == SPIN8_F32)
    return hadamard_block_impl(static_cast<const float *>(s), ss,
                               static_cast<float *>(d), ds, rows, cols, block,
                               signs, inverse);
  return t == SPIN8_F16 || t == SPIN8_BF16 ? SPIN8_UNSUPPORTED
                                           : SPIN8_INVALID_ARGUMENT;
}
spin8_status spin8_hadamard_block_audit(spin8_dtype t, size_t block,
                                        const int8_t *signs,
                                        spin8_matrix_audit *out) {
  if (!out || !hadamard_config(block, signs))
    return SPIN8_INVALID_ARGUMENT;
  if (t != SPIN8_F64 && t != SPIN8_F32)
    return t == SPIN8_F16 || t == SPIN8_BF16 ? SPIN8_UNSUPPORTED
                                             : SPIN8_INVALID_ARGUMENT;
  const double n = static_cast<double>(block);
  double c = 1 / std::sqrt(n);
  if (t == SPIN8_F32)
    c = static_cast<float>(c);
  spin8_matrix_audit r{};
  r.max_error = r.ef_normalized = std::abs(n * c * c - 1);
  r.log_abs_det = n * std::log(std::abs(c) * std::sqrt(n));
  r.det_sign = block == 2 ? -1 : 1;
  if (signs)
    for (size_t i = 0; i < block; ++i)
      r.det_sign *= signs[i];
  *out = r;
  return SPIN8_OK;
}
spin8_status spin8_transform(const spin8_plan *p, spin8_view v, spin8_dtype t,
                             const void *s, size_t ss, void *d, size_t ds,
                             size_t r, size_t c, int inverse) {
  if (!p || !view_ok(v) || (inverse != 0 && inverse != 1))
    return SPIN8_INVALID_ARGUMENT;
  return dispatch(inverse ? transpose<8>(p->m[v]) : p->m[v], t, s, ss, d, ds, r,
                  c);
}
spin8_status spin8_hadamard(spin8_dtype t, const void *s, size_t ss, void *d,
                            size_t ds, size_t r, size_t c, uint8_t mask,
                            int inv) {
  M8 mat{};
  auto status = spin8_hadamard_matrix(mask, inv, mat.data());
  return status == SPIN8_OK ? dispatch(mat, t, s, ss, d, ds, r, c) : status;
}
spin8_status spin8_reduce(spin8_dtype t, const void *const views[3], size_t ss,
                          void *dst, size_t ds, size_t rows, size_t cols,
                          const double *w) {
  std::array<double, 3> weight{};
  auto status = weights_check(w, weight);
  if (status != SPIN8_OK)
    return status;
  if (t == SPIN8_F64)
    return reduce_impl(views, ss, static_cast<double *>(dst), ds, rows, cols,
                       weight);
  if (t == SPIN8_F32)
    return reduce_impl(views, ss, static_cast<float *>(dst), ds, rows, cols,
                       weight);
  return t == SPIN8_F16 || t == SPIN8_BF16 ? SPIN8_UNSUPPORTED
                                           : SPIN8_INVALID_ARGUMENT;
}
spin8_status spin8_matmul_f64(const spin8_plan *p, const double *a, size_t m,
                              const double *b, size_t n, size_t k,
                              const double *w, double *out) {
  return reference(p, a, m, b, n, k, static_cast<const double *>(nullptr), 0,
                   nullptr, w, out, false);
}
spin8_status spin8_matmul_f32(const spin8_plan *p, const float *a, size_t m,
                              const float *b, size_t n, size_t k,
                              const double *w, float *out) {
  return reference(p, a, m, b, n, k, static_cast<const float *>(nullptr), 0,
                   nullptr, w, out, false);
}
spin8_status spin8_attention_f64(const spin8_plan *p, const double *q,
                                 size_t nq, const double *k, size_t nk,
                                 const double *v, size_t dv, size_t d,
                                 const uint8_t *mask, const double *w,
                                 double *out) {
  return reference(p, q, nq, k, nk, d, v, dv, mask, w, out, true);
}
spin8_status spin8_attention_f32(const spin8_plan *p, const float *q, size_t nq,
                                 const float *k, size_t nk, const float *v,
                                 size_t dv, size_t d, const uint8_t *mask,
                                 const double *w, float *out) {
  return reference(p, q, nq, k, nk, d, v, dv, mask, w, out, true);
}
void spin8_monitor_init(spin8_monitor *m) {
  if (m)
    *m = {0, 0, SPIN8_OK, 0};
}
spin8_status spin8_monitor_due(const spin8_monitor *m, uint64_t tokens,
                               double seconds, int *due) {
  if (!m || !due || !std::isfinite(seconds) || seconds < 0 ||
      tokens < m->last_tokens || seconds < m->last_seconds)
    return SPIN8_INVALID_ARGUMENT;
  if (m->failure != SPIN8_OK)
    return SPIN8_INVALIDATED;
  *due = !m->valid || tokens - m->last_tokens >= 1024 ||
         seconds - m->last_seconds >= 60;
  return SPIN8_OK;
}
spin8_status spin8_monitor_record(spin8_monitor *m, uint64_t tokens,
                                  double seconds, spin8_status result) {
  int due = 0;
  auto s = spin8_monitor_due(m, tokens, seconds, &due);
  if (s != SPIN8_OK)
    return s;
  if (result < SPIN8_OK || result > SPIN8_INVALIDATED)
    return SPIN8_INVALID_ARGUMENT;
  m->last_tokens = tokens;
  m->last_seconds = seconds;
  m->valid = result == SPIN8_OK;
  m->failure = result;
  return result;
}
}
