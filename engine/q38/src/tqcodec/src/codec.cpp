#include "tqcodec/tqcodec.h"
#include <algorithm>
#include <cmath>
#include <cstring>
#include <limits>

namespace {
bool valid(const double *p, size_t n) {
  if (!p || n > SIZE_MAX / sizeof(double))
    return false;
  for (size_t i = 0; i < n; ++i)
    if (!std::isfinite(p[i]))
      return false;
  return true;
}
bool book(const double *p, unsigned bits) {
  if (bits < 2 || bits > 6 || !valid(p, size_t{1} << bits))
    return false;
  for (size_t i = 1; i < (size_t{1} << bits); ++i)
    if (p[i] <= p[i - 1])
      return false;
  return true;
}
double length(const double *p, size_t n) {
  double v = 0;
  for (size_t i = 0; i < n; ++i)
    v = std::hypot(v, p[i]);
  return v;
}
unsigned index_at(const uint8_t *p, size_t i, unsigned b) {
  unsigned v = 0;
  for (unsigned j = 0; j < b; ++j) {
    size_t z = i * b + j;
    v |= unsigned((p[z / 8] >> (z % 8)) & 1) << j;
  }
  return v;
}
bool overlaps(const void *p, size_t n, const void *q, size_t m) {
  auto a = reinterpret_cast<uintptr_t>(p), b = reinterpret_cast<uintptr_t>(q);
  if (a > UINTPTR_MAX - n || b > UINTPTR_MAX - m)
    return true;
  return n && m && a < b + m && b < a + n;
}
bool matrix_ok(size_t d, size_t m) {
  return d && m && d <= SIZE_MAX / m && d * m <= SIZE_MAX / sizeof(double) &&
         m <= SIZE_MAX - 7;
}
double product3(double a, double b, double c) {
  int ea = 0, eb = 0, ec = 0;
  const double ma = std::frexp(a, &ea), mb = std::frexp(b, &eb), mc = std::frexp(c, &ec);
  return std::ldexp(ma * mb * mc, ea + eb + ec);
}
} // namespace
extern "C" {
tq_status tq_block_size(size_t d, unsigned b, tq_size *out) {
  if (!out || !d || b < 2 || b > 6 || d > (SIZE_MAX - 7) / b)
    return TQ_INVALID;
  const size_t bytes = (d * b + 7) / 8;
  if (bytes > SIZE_MAX - sizeof(double))
    return TQ_INVALID;
  *out = {bytes, sizeof(double), bytes + sizeof(double)};
  return TQ_OK;
}
tq_status tq_encode_f64(const double *x, size_t d, const double *cb, unsigned b,
                        uint8_t *p, size_t cap, double *n) {
  tq_size s{};
  if (tq_block_size(d, b, &s) != TQ_OK || !p || !n || !book(cb, b))
    return TQ_INVALID;
  if (cap < s.payload_bytes)
    return TQ_CAPACITY;
  if (!valid(x, d))
    return TQ_NONFINITE;
  if (overlaps(p, s.payload_bytes, x, d * sizeof(double)) ||
      overlaps(p, s.payload_bytes, cb, (size_t{1} << b) * sizeof(double)) ||
      overlaps(n, sizeof(double), p, s.payload_bytes) ||
      overlaps(n, sizeof(double), x, d * sizeof(double)) ||
      overlaps(n, sizeof(double), cb, (size_t{1} << b) * sizeof(double)))
    return TQ_INVALID;
  const double scale = length(x, d);
  if (!std::isfinite(scale))
    return TQ_NONFINITE;
  std::memset(p, 0, s.payload_bytes);
  for (size_t i = 0; i < d; ++i) {
    double v = scale == 0 ? 0 : x[i] / scale;
    unsigned best = 0;
    for (unsigned j = 1; j < (1u << b); ++j)
      if (std::abs(v - cb[j]) < std::abs(v - cb[best]))
        best = j;
    for (unsigned j = 0; j < b; ++j) {
      size_t z = i * b + j;
      p[z / 8] |= uint8_t(((best >> j) & 1) << (z % 8));
    }
  }
  *n = scale;
  return TQ_OK;
}
tq_status tq_decode_f64(const uint8_t *p, size_t cap, size_t d,
                        const double *cb, unsigned b, double n, double *out) {
  tq_size s{};
  if (tq_block_size(d, b, &s) != TQ_OK || !p || !out || !book(cb, b) ||
      !std::isfinite(n) || n < 0 || d > SIZE_MAX / sizeof(double))
    return TQ_INVALID;
  if (cap < s.payload_bytes)
    return TQ_CAPACITY;
  if (overlaps(out, d * sizeof(double), p, s.payload_bytes) ||
      overlaps(out, d * sizeof(double), cb, (size_t{1} << b) * sizeof(double)))
    return TQ_INVALID;
  for (size_t i = 0; i < d; ++i)
    if (!std::isfinite(n * cb[index_at(p, i, b)]))
      return TQ_NONFINITE;
  for (size_t i = 0; i < d; ++i)
    out[i] = n * cb[index_at(p, i, b)];
  return TQ_OK;
}
tq_status tq_score_f64(const double *q, const uint8_t *p, size_t cap, size_t d,
                       const double *cb, unsigned b, double n, double *out) {
  tq_size s{};
  if (!out || !p || tq_block_size(d, b, &s) != TQ_OK || !book(cb, b) ||
      !std::isfinite(n) || n < 0)
    return TQ_INVALID;
  if (cap < s.payload_bytes)
    return TQ_CAPACITY;
  if (!valid(q, d))
    return TQ_NONFINITE;
  double sum = 0;
  for (size_t i = 0; i < d; ++i)
    sum += q[i] * (n * cb[index_at(p, i, b)]);
  if (!std::isfinite(sum))
    return TQ_NONFINITE;
  *out = sum;
  return TQ_OK;
}
tq_status tq_qjl_encode_f64(const double *e, size_t d, const double *g,
                            size_t m, uint8_t *p, size_t cap, double *n) {
  if (!matrix_ok(d, m) || !p || !n)
    return TQ_INVALID;
  const size_t bytes = (m + 7) / 8;
  if (cap < bytes)
    return TQ_CAPACITY;
  if (!valid(e, d) || !valid(g, d * m))
    return TQ_NONFINITE;
  if (overlaps(p, bytes, e, d * sizeof(double)) ||
      overlaps(p, bytes, g, d * m * sizeof(double)) ||
      overlaps(n, sizeof(double), p, bytes) ||
      overlaps(n, sizeof(double), e, d * sizeof(double)) ||
      overlaps(n, sizeof(double), g, d * m * sizeof(double)))
    return TQ_INVALID;
  double scale = length(e, d);
  if (!std::isfinite(scale))
    return TQ_NONFINITE;
  // Two passes preserve the output if a projection overflows.
  for (int pass = 0; pass < 2; ++pass) {
    if (pass)
      std::memset(p, 0, bytes);
    for (size_t j = 0; j < m; ++j) {
      double sum = 0;
      for (size_t i = 0; i < d; ++i)
        sum += g[j * d + i] * e[i];
      if (!std::isfinite(sum))
        return TQ_NONFINITE;
      if (pass && sum >= 0)
        p[j / 8] |= uint8_t(1u << (j % 8));
    }
  }
  *n = scale;
  return TQ_OK;
}
tq_status tq_qjl_score_f64(const double *q, size_t d, const double *g, size_t m,
                           const uint8_t *p, size_t cap, double n, double beta,
                           double *out) {
  if (!matrix_ok(d, m) || !p || !out || !std::isfinite(n) || n < 0 ||
      !std::isfinite(beta) || beta < 0 || beta > 1)
    return TQ_INVALID;
  if (cap < (m + 7) / 8)
    return TQ_CAPACITY;
  if (!valid(q, d) || !valid(g, d * m))
    return TQ_NONFINITE;
  if (beta == 0 || n == 0) {
    *out = 0;
    return TQ_OK;
  }
  double sum = 0;
  for (size_t j = 0; j < m; ++j) {
    double v = 0;
    for (size_t i = 0; i < d; ++i)
      v += g[j * d + i] * q[i];
    sum += ((p[j / 8] >> (j % 8)) & 1) ? v : -v;
  }
  const double result =
      beta * std::sqrt(std::acos(-1.) / 2) * n * (sum / static_cast<double>(m));
  if (!std::isfinite(result))
    return TQ_NONFINITE;
  *out = result;
  return TQ_OK;
}
tq_status tq_qjl_error_moments_f64(const double *q, const double *e, size_t d,
                                   size_t m, double beta, double scale,
                                   tq_qjl_moments *out) {
  if (!out || !d || !m || d > SIZE_MAX / sizeof(double) ||
      !std::isfinite(beta) || beta < 0 || beta > 1 || !std::isfinite(scale))
    return TQ_INVALID;
  if (!valid(q, d) || !valid(e, d))
    return TQ_NONFINITE;
  const double qn = length(q, d), en = length(e, d);
  if (!std::isfinite(qn) || !std::isfinite(en))
    return TQ_NONFINITE;
  if (qn == 0 || en == 0 || scale == 0) {
    *out = {0, 0, 0};
    return TQ_OK;
  }
  // Normalize first so the variance never subtracts two very large squares.
  // pi/2 - cosine^2 is bounded below by pi/2 - 1, even for parallel inputs.
  double cosine = 0, scaled_dot = 0;
  for (size_t i = 0; i < d; ++i) {
    cosine += (q[i] / qn) * (e[i] / en);
    // Computing bias via cosine loses tiny cross-components when the norm
    // product is huge. Scale each dot contribution with exponent balancing.
    scaled_dot += product3(scale, q[i], e[i]);
  }
  cosine = std::max(-1., std::min(1., cosine));
  const double scaled_product = product3(scale, qn, en);
  const double bias = (beta - 1) * scaled_dot;
  const double amplitude = beta * scaled_product / std::sqrt(static_cast<double>(m));
  const double variance = amplitude * amplitude * (std::acos(-1.) / 2 - cosine * cosine);
  const double mse = bias * bias + variance;
  if (!std::isfinite(scaled_product) || !std::isfinite(scaled_dot) || !std::isfinite(bias) ||
      !std::isfinite(variance) || !std::isfinite(mse))
    return TQ_NONFINITE;
  *out = {bias, variance, mse};
  return TQ_OK;
}

tq_status tq_shared_residual_f64(const double *k, const double *v[3], size_t d,
                                 const double a[3], double *out) {
  if (!out || !v || !d || !valid(a, 3) || a[0] < 0 || a[1] < 0 || a[2] < 0 ||
      std::abs(a[0] + a[1] + a[2] - 1) > 1e-12)
    return TQ_INVALID;
  if (!valid(k, d))
    return TQ_NONFINITE;
  if (overlaps(out, d * sizeof(double), a, 3 * sizeof(double)) ||
      overlaps(out, d * sizeof(double), v, 3 * sizeof(*v)))
    return TQ_INVALID;
  for (int r = 0; r < 3; ++r) {
    if (!valid(v[r], d))
      return TQ_NONFINITE;
    if (overlaps(out, d * sizeof(double), v[r], d * sizeof(double)))
      return TQ_INVALID;
  }
  if (k != out && overlaps(out, d * sizeof(double), k, d * sizeof(double)))
    return TQ_INVALID;
  for (size_t i = 0; i < d; ++i)
    if (!std::isfinite(k[i] -
                       (a[0] * v[0][i] + a[1] * v[1][i] + a[2] * v[2][i])))
      return TQ_NONFINITE;
  for (size_t i = 0; i < d; ++i)
    out[i] = k[i] - (a[0] * v[0][i] + a[1] * v[1][i] + a[2] * v[2][i]);
  return TQ_OK;
}
}
