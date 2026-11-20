#include "tqcodec/calibrate.h"
#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <limits>
namespace {
using V = std::array<double, 3>;
bool overlaps(const void *p, size_t n, const void *q, size_t m) {
  const auto a = reinterpret_cast<uintptr_t>(p),
             b = reinterpret_cast<uintptr_t>(q);
  return a > UINTPTR_MAX - n || b > UINTPTR_MAX - m || (a < b + m && b < a + n);
}
} // namespace
extern "C" tq_status tq_consensus_fit_f64(const double *errors, size_t capacity,
                                          size_t samples, size_t components,
                                          double ridge, tq_consensus_fit *out) {
  if (!errors || !out || !samples || !components || !std::isfinite(ridge) ||
      ridge < 0 || samples > SIZE_MAX / 3 ||
      components > SIZE_MAX / (samples * 3) ||
      samples * 3 * components > SIZE_MAX / sizeof(double) ||
      reinterpret_cast<uintptr_t>(errors) % alignof(double) ||
      reinterpret_cast<uintptr_t>(out) % alignof(tq_consensus_fit))
    return TQ_INVALID;
  const size_t elements = samples * 3 * components,
               bytes = elements * sizeof(double);
  if (capacity < bytes)
    return TQ_CAPACITY;
  if (overlaps(errors, bytes, out, sizeof(*out)))
    return TQ_INVALID;
  double maximum = 0;
  for (size_t i = 0; i < elements; ++i) {
    if (!std::isfinite(errors[i]))
      return TQ_NONFINITE;
    maximum = std::max(maximum, std::abs(errors[i]));
  }
  const V uniform{1. / 3, 1. / 3, 1. / 3};
  if (maximum == 0) {
    *out = {{uniform[0], uniform[1], uniform[2]}, 0, 1};
    return TQ_OK;
  }
  double trace = 0;
  for (size_t i = 0; i < elements; ++i) {
    const double value = errors[i] / maximum;
    trace += value * value;
  }
  if (!std::isfinite(trace) || trace <= 0)
    return TQ_NONFINITE;
  // Divide the entire regularized system by (1+ridge) without overflowing.
  // This changes neither its minimizer nor the relative tie handling.
  const double hscale =
      ridge > 1 ? (1 / ridge) / (1 + 1 / ridge) : 1 / (1 + ridge);
  const double rscale = ridge > 1 ? 1 / (1 + 1 / ridge) : ridge / (1 + ridge);
  const double data_scale = std::sqrt(hscale) / std::sqrt(trace);
  // Evaluate candidate residuals directly: the Gram quadratic can cancel to
  // negative/zero when two large error vectors almost exactly cancel. There
  // are only eight candidates, and calibration is outside the decoding path.
  auto data_loss = [&](const V &a) {
    double sum = 0;
    for (size_t s = 0; s < samples; ++s)
      for (size_t k = 0; k < components; ++k) {
        double residual = 0;
        for (size_t r = 0; r < 3; ++r)
          residual += a[r] * (errors[(s * 3 + r) * components + k] / maximum);
        sum += residual * residual;
      }
    return sum / trace;
  };
  auto penalty = [&](const V &a) {
    double sum = 0;
    for (double x : a)
      sum += (x - 1. / 3) * (x - 1. / 3);
    return sum;
  };
  auto objective_scaled = [&](const V &a) {
    return hscale * data_loss(a) + rscale * penalty(a);
  };
  V best = uniform;
  double best_value = objective_scaled(best);
  auto consider = [&](V a) {
    for (double &x : a) {
      if (!std::isfinite(x) || x < -1e-13)
        return;
      x = std::max(0., x);
    }
    const double sum = a[0] + a[1] + a[2];
    if (!(sum > 0))
      return;
    for (double &x : a)
      x /= sum;
    const double value = objective_scaled(a);
    if (value <
        best_value * (1 - 32 * std::numeric_limits<double>::epsilon())) {
      best = a;
      best_value = value;
    }
  };
  // Every optimum lies at a vertex, on an edge, or in the interior. Singular
  // interior minima have a flat direction reaching an edge with the same loss.
  for (size_t i = 0; i < 3; ++i) {
    V a{};
    a[i] = 1;
    consider(a);
  }
  for (size_t i = 0; i < 3; ++i)
    for (size_t j = i + 1; j < 3; ++j) {
      double curvature = 2 * rscale, numerator = rscale;
      for (size_t s = 0; s < samples; ++s)
        for (size_t k = 0; k < components; ++k) {
          const double ei = errors[(s * 3 + i) * components + k] / maximum,
                       ej = errors[(s * 3 + j) * components + k] / maximum,
                       direction = (ei - ej) * data_scale;
          curvature += direction * direction;
          numerator -= direction * (ej * data_scale);
        }
      if (curvature > 0) {
        const double t = std::clamp(numerator / curvature, 0., 1.);
        V a{};
        a[i] = t;
        a[j] = 1 - t;
        consider(a);
      }
    }
  // Streaming Givens QR of the two affine error directions. Forming normal
  // equations would square the condition number and discard small but
  // independent directions (including exactly cancellable interior solutions).
  double r00 = 0, r01 = 0, r11 = 0, y0 = 0, y1 = 0;
  auto qr_row = [&](double a, double b, double rhs) {
    const double first = std::hypot(r00, a);
    if (first > 0) {
      const double c = r00 / first, s = a / first, next_b = -s * r01 + c * b,
                   next_rhs = -s * y0 + c * rhs;
      r01 = c * r01 + s * b;
      y0 = c * y0 + s * rhs;
      r00 = first;
      b = next_b;
      rhs = next_rhs;
    }
    const double second = std::hypot(r11, b);
    if (second > 0) {
      y1 = (r11 / second) * y1 + (b / second) * rhs;
      r11 = second;
    }
  };
  for (size_t s = 0; s < samples; ++s)
    for (size_t k = 0; k < components; ++k) {
      V e{};
      for (size_t r = 0; r < 3; ++r)
        e[r] = errors[(s * 3 + r) * components + k] / maximum;
      qr_row((e[0] - e[2]) * data_scale, (e[1] - e[2]) * data_scale,
             -e[2] * data_scale);
    }
  const double shrink = std::sqrt(rscale);
  qr_row(shrink, 0, shrink / 3);
  qr_row(0, shrink, shrink / 3);
  qr_row(-shrink, -shrink, -2 * shrink / 3);
  if (r00 > 0 && r11 > 0) {
    const double y = y1 / r11, x = (y0 - r01 * y) / r00;
    consider({x, y, 1 - x - y});
  }
  const double uniform_error = data_loss(uniform);
  const double candidate_error = data_loss(best);
  const double objective = candidate_error + ridge * penalty(best);
  const double ratio = uniform_error > 0 ? candidate_error / uniform_error : 1.;
  if (!std::isfinite(objective) || !std::isfinite(ratio))
    return TQ_NONFINITE;
  *out = {{best[0], best[1], best[2]}, objective, ratio};
  return TQ_OK;
}
