#include "tqcodec/distill.h"
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>

namespace {
bool overlap(const void *p, size_t n, const void *q, size_t m) {
  const auto a = reinterpret_cast<uintptr_t>(p),
             b = reinterpret_cast<uintptr_t>(q);
  return a > UINTPTR_MAX - n || b > UINTPTR_MAX - m ||
         (n && m && a < b + m && b < a + n);
}
bool shape(tq_f64_rows t, size_t &span) {
  if (!t.data || !t.rows || !t.columns ||
      t.columns > SIZE_MAX / sizeof(double) ||
      t.row_stride_bytes < t.columns * sizeof(double) ||
      t.row_stride_bytes % alignof(double) ||
      reinterpret_cast<uintptr_t>(t.data) % alignof(double) ||
      t.rows - 1 > (SIZE_MAX - t.columns * sizeof(double)) / t.row_stride_bytes)
    return false;
  span = (t.rows - 1) * t.row_stride_bytes + t.columns * sizeof(double);
  return true;
}
const double *row(tq_f64_rows t, size_t i) {
  return reinterpret_cast<const double *>(
      reinterpret_cast<const uint8_t *>(t.data) + i * t.row_stride_bytes);
}
bool finite(const double *p, size_t n) {
  for (size_t i = 0; i < n; ++i)
    if (!std::isfinite(p[i]))
      return false;
  return true;
}
double score(const double *q, const double *k, size_t d, double scale,
             double bias) {
  double dot = 0;
  for (size_t i = 0; i < d; ++i)
    dot += q[i] * k[i];
  return scale * dot + bias;
}
} // namespace
extern "C" {
tq_status tq_distill_workspace_size(size_t d, size_t v, size_t *bytes) {
  if (!bytes || !d || !v || v > (SIZE_MAX - d) / 2 ||
      d + 2 * v > SIZE_MAX / sizeof(double))
    return TQ_INVALID;
  *bytes = (d + 2 * v) * sizeof(double);
  return TQ_OK;
}
tq_status tq_distill_head_f64(const double *qt, size_t qt_bytes,
                              const double *qs, size_t qs_bytes, tq_f64_rows kt,
                              tq_f64_rows ks, tq_f64_rows vt, tq_f64_rows vs,
                              const double *bias, size_t bias_bytes,
                              double scale, double wm, double wk, double eps,
                              void *workspace, size_t workspace_bytes,
                              tq_distill_loss *loss, double *gradient,
                              size_t gradient_bytes) {
  const size_t d = kt.columns, v = vt.columns, n = kt.rows;
  if (!qt || !qs || !workspace || !loss || !gradient || !std::isfinite(scale) ||
      !std::isfinite(wm) || wm < 0 || !std::isfinite(wk) || wk < 0 ||
      !std::isfinite(eps) || eps <= 0 || ks.columns != d || vs.columns != v ||
      ks.rows != n || vt.rows != n || vs.rows != n ||
      n > SIZE_MAX / sizeof(double) || (!bias && bias_bytes) ||
      reinterpret_cast<uintptr_t>(loss) % alignof(tq_distill_loss) ||
      reinterpret_cast<uintptr_t>(workspace) % alignof(double) ||
      reinterpret_cast<uintptr_t>(qt) % alignof(double) ||
      reinterpret_cast<uintptr_t>(qs) % alignof(double) ||
      reinterpret_cast<uintptr_t>(gradient) % alignof(double) ||
      (bias && reinterpret_cast<uintptr_t>(bias) % alignof(double)))
    return TQ_INVALID;
  tq_f64_rows matrices[] = {kt, ks, vt, vs};
  size_t spans[4], scratch_bytes = 0;
  if (tq_distill_workspace_size(d, v, &scratch_bytes) != TQ_OK)
    return TQ_INVALID;
  for (int j = 0; j < 4; ++j) {
    if (!shape(matrices[j], spans[j]))
      return TQ_INVALID;
    if (matrices[j].capacity_bytes < spans[j])
      return TQ_CAPACITY;
  }
  const size_t qbytes = d * sizeof(double),
               bbytes = bias ? n * sizeof(double) : 0;
  if (qt_bytes < qbytes || qs_bytes < qbytes ||
      workspace_bytes < scratch_bytes || gradient_bytes < qbytes ||
      bias_bytes < bbytes)
    return TQ_CAPACITY;
  const void *inputs[] = {qt, qs, kt.data, ks.data, vt.data, vs.data, bias};
  const size_t input_sizes[] = {qbytes,   qbytes,   spans[0], spans[1],
                                spans[2], spans[3], bbytes};
  const void *outputs[] = {workspace, loss, gradient};
  const size_t output_sizes[] = {scratch_bytes, sizeof(*loss), qbytes};
  for (int j = 0; j < 3; ++j) {
    for (int i = 0; i < 7; ++i)
      if (overlap(outputs[j], output_sizes[j], inputs[i], input_sizes[i]))
        return TQ_INVALID;
    for (int i = 0; i < j; ++i)
      if (overlap(outputs[j], output_sizes[j], outputs[i], output_sizes[i]))
        return TQ_INVALID;
  }
  if (!finite(qt, d) || !finite(qs, d))
    return TQ_NONFINITE;
  for (const auto &t : matrices)
    for (size_t j = 0; j < n; ++j)
      if (!finite(row(t, j), t.columns))
        return TQ_NONFINITE;
  double mt = -std::numeric_limits<double>::infinity(), ms = mt;
  for (size_t j = 0; j < n; ++j) {
    const double b = bias ? bias[j] : 0;
    if (std::isnan(b) || b == std::numeric_limits<double>::infinity())
      return TQ_NONFINITE;
    if (b == -std::numeric_limits<double>::infinity())
      continue;
    const double st = score(qt, row(kt, j), d, scale, b),
                 ss = score(qs, row(ks, j), d, scale, b);
    if (!std::isfinite(st) || !std::isfinite(ss))
      return TQ_NONFINITE;
    mt = std::max(mt, st);
    ms = std::max(ms, ss);
  }
  if (!std::isfinite(mt))
    return TQ_INVALID;
  auto *ot = static_cast<double *>(workspace), *os = ot + v, *dq = os + v;
  std::fill(ot, dq + d, 0.);
  double zt = 0, zs = 0;
  for (size_t j = 0; j < n; ++j) {
    const double b = bias ? bias[j] : 0;
    if (!std::isfinite(b))
      continue;
    const double pt = std::exp(score(qt, row(kt, j), d, scale, b) - mt),
                 ps = std::exp(score(qs, row(ks, j), d, scale, b) - ms);
    zt += pt;
    zs += ps;
    for (size_t i = 0; i < v; ++i) {
      ot[i] += pt * row(vt, j)[i];
      os[i] += ps * row(vs, j)[i];
    }
  }
  double denom = eps, squared_error = 0;
  for (size_t i = 0; i < v; ++i) {
    ot[i] /= zt;
    os[i] /= zs;
    denom += ot[i] * ot[i];
    squared_error += (os[i] - ot[i]) * (os[i] - ot[i]);
  }
  double kl_sum = 0;
  const double log_ratio = std::log(zs) - std::log(zt);
  for (size_t j = 0; j < n; ++j) {
    const double b = bias ? bias[j] : 0;
    if (!std::isfinite(b))
      continue;
    const double st = score(qt, row(kt, j), d, scale, b) - mt,
                 ss = score(qs, row(ks, j), d, scale, b) - ms;
    const double pt = std::exp(st) / zt, ps = std::exp(ss) / zs;
    if (pt != 0)
      kl_sum += pt * (st - ss + log_ratio);
    double value_sensitivity = 0;
    for (size_t i = 0; i < v; ++i)
      value_sensitivity +=
          2 * wm * (os[i] - ot[i]) / denom * (row(vs, j)[i] - os[i]);
    const double ds = ps * value_sensitivity + wk * (ps - pt);
    for (size_t i = 0; i < d; ++i)
      dq[i] += scale * ds * row(ks, j)[i];
  }
  const tq_distill_loss result{squared_error / denom, kl_sum,
                               wm * squared_error / denom + wk * kl_sum};
  if (!std::isfinite(zt) || !std::isfinite(zs) || !std::isfinite(denom) ||
      !std::isfinite(result.normalized_output_mse) ||
      !std::isfinite(result.attention_kl) || !std::isfinite(result.total) ||
      !finite(dq, d))
    return TQ_NONFINITE;
  std::copy(dq, dq + d, gradient);
  *loss = result;
  return TQ_OK;
}
}
