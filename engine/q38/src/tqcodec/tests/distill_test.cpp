#include "tqcodec/distill.h"
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <limits>
#include <vector>
static int failures;
#define CHECK(c)                                                               \
  do {                                                                         \
    if (!(c)) {                                                                \
      std::fprintf(stderr, "%d: %s\n", __LINE__, #c);                          \
      ++failures;                                                              \
    }                                                                          \
  } while (0)
static tq_f64_rows view(const std::vector<double> &x, size_t n, size_t d,
                        size_t stride) {
  return {x.data(), n, d, stride * sizeof(double), x.size() * sizeof(double)};
}
// Independent dense-softmax path stores N scores/probabilities. The library
// recomputes rows and uses only O(d+dv) scratch instead.
static double reference(const double *qt, const double *qs,
                        const std::vector<double> &kt,
                        const std::vector<double> &ks,
                        const std::vector<double> &vt,
                        const std::vector<double> &vs, const double *bias,
                        double scale, double wm, double wk) {
  std::vector<double> pt(5), ps(5), ot(2), os(2);
  for (size_t j = 0; j < 5; ++j) {
    pt[j] = ps[j] = bias[j];
    for (size_t i = 0; i < 3; ++i) {
      pt[j] += scale * qt[i] * kt[j * 4 + i];
      ps[j] += scale * qs[i] * ks[j * 4 + i];
    }
  }
  for (auto *p : {&pt, &ps}) {
    const double m = *std::max_element(p->begin(), p->end());
    double z = 0;
    for (double &x : *p) {
      x = std::exp(x - m);
      z += x;
    }
    for (double &x : *p)
      x /= z;
  }
  double kl = 0, err = 0, den = 1e-6;
  for (size_t j = 0; j < 5; ++j) {
    if (pt[j])
      kl += pt[j] * std::log(pt[j] / ps[j]);
    for (size_t i = 0; i < 2; ++i) {
      ot[i] += pt[j] * vt[j * 3 + i];
      os[i] += ps[j] * vs[j * 3 + i];
    }
  }
  for (size_t i = 0; i < 2; ++i) {
    den += ot[i] * ot[i];
    err += (os[i] - ot[i]) * (os[i] - ot[i]);
  }
  return wm * err / den + wk * kl;
}
int main() {
  std::vector<double> kt(20), ks(20), vt(15), vs(15);
  for (size_t i = 0; i < 20; ++i) {
    kt[i] = std::sin(double(i));
    ks[i] = .8 * kt[i] + .1;
  }
  for (size_t i = 0; i < 15; ++i) {
    vt[i] = std::cos(double(i));
    vs[i] = .9 * vt[i] - .1;
  }
  // Padding is deliberately NaN: only addressed elements may be read.
  for (size_t j = 0; j < 5; ++j) {
    kt[j * 4 + 3] = ks[j * 4 + 3] = std::numeric_limits<double>::quiet_NaN();
    vt[j * 3 + 2] = vs[j * 3 + 2] = std::numeric_limits<double>::quiet_NaN();
  }
  double qt[] = {.3, -.4, .8}, qs[] = {-.2, .7, .4},
         bias[] = {.2, -.1, -INFINITY, .3, .5};
  double gradient[3] = {}, scratch[8] = {};
  tq_distill_loss loss{};
  auto call = [&](double wm, double wk) {
    return tq_distill_head_f64(
        qt, sizeof(qt), qs, sizeof(qs), view(kt, 5, 3, 4), view(ks, 5, 3, 4),
        view(vt, 5, 2, 3), view(vs, 5, 2, 3), bias, sizeof(bias), .7, wm, wk,
        1e-6, scratch, sizeof(scratch), &loss, gradient, sizeof(gradient));
  };
  size_t bytes = 0;
  CHECK(tq_distill_workspace_size(3, 2, &bytes) == TQ_OK &&
        bytes == 7 * sizeof(double));
  CHECK(tq_distill_workspace_size(SIZE_MAX, 2, &bytes) == TQ_INVALID);
  for (const auto &weights :
       {std::vector<double>{1., 0.}, {0., 1.}, {.7, .2}}) {
    const double wm = weights[0], wk = weights[1];
    CHECK(call(wm, wk) == TQ_OK);
    const double original = loss.total;
    double analytic[3];
    std::copy(gradient, gradient + 3, analytic);
    const double expected_loss =
        reference(qt, qs, kt, ks, vt, vs, bias, .7, wm, wk);
    // The independent path scales each summand, the library scales a dot.
    // Bound their rounding discrepancy relative to the normalized loss.
    CHECK(std::abs(loss.total - expected_loss) <
          32 * std::numeric_limits<double>::epsilon() *
              (1 + std::abs(expected_loss)));
    for (size_t i = 0; i < 3; ++i) {
      const double old = qs[i], step = 1e-5;
      qs[i] = old + step;
      CHECK(call(wm, wk) == TQ_OK);
      const double plus = loss.total;
      qs[i] = old - step;
      CHECK(call(wm, wk) == TQ_OK);
      const double minus = loss.total;
      qs[i] = old;
      CHECK(std::abs(analytic[i] - (plus - minus) / (2 * step)) < 2e-8);
    }
    double old[3];
    std::copy(qs, qs + 3, old);
    for (size_t i = 0; i < 3; ++i)
      qs[i] -= .01 * analytic[i];
    CHECK(call(wm, wk) == TQ_OK && loss.total < original);
    std::copy(old, old + 3, qs);
  }
  // Detached teacher: no mutation, even when teacher and student are identical.
  CHECK(tq_distill_head_f64(qt, sizeof(qt), qt, sizeof(qt), view(kt, 5, 3, 4),
                            view(kt, 5, 3, 4), view(vt, 5, 2, 3),
                            view(vt, 5, 2, 3), bias, sizeof(bias), .7, 1, 1,
                            1e-6, scratch, sizeof(scratch), &loss, gradient,
                            sizeof(gradient)) == TQ_OK);
  CHECK(loss.total == 0 && gradient[0] == 0 && gradient[1] == 0 &&
        gradient[2] == 0);
  loss = {11, 22, 33};
  gradient[0] = 44;
  auto short_ks = view(ks, 5, 3, 4);
  short_ks.capacity_bytes = 8;
  CHECK(tq_distill_head_f64(qt, sizeof(qt), qs, sizeof(qs), view(kt, 5, 3, 4),
                            short_ks, view(vt, 5, 2, 3), view(vs, 5, 2, 3),
                            bias, sizeof(bias), .7, 1, 1, 1e-6, scratch,
                            sizeof(scratch), &loss, gradient,
                            sizeof(gradient)) == TQ_CAPACITY);
  CHECK(loss.total == 33 && gradient[0] == 44);
  CHECK(tq_distill_head_f64(qt, sizeof(qt), qs, sizeof(qs), view(kt, 5, 3, 4),
                            view(ks, 5, 3, 4), view(vt, 5, 2, 3),
                            view(vs, 5, 2, 3), bias, sizeof(bias), .7, 1, 1,
                            1e-6, scratch, sizeof(scratch), &loss, qs,
                            sizeof(qs)) == TQ_INVALID);
  for (double &b : bias)
    b = -INFINITY;
  CHECK(call(1, 1) == TQ_INVALID && loss.total == 33 && gradient[0] == 44);
  bias[0] = INFINITY;
  CHECK(call(1, 1) == TQ_NONFINITE && loss.total == 33 && gradient[0] == 44);
  bias[0] = 0;
  // KL stays finite when a student's probability underflows to zero. Taking
  // log(probability) would fail; the implementation uses logit differences.
  const double extreme_kt[] = {1000, -1000}, extreme_ks[] = {-1000, 1000};
  const double extreme_v[] = {1, -1}, one_query[] = {1};
  const tq_f64_rows ekt{extreme_kt, 2, 1, sizeof(double), sizeof(extreme_kt)},
      eks{extreme_ks, 2, 1, sizeof(double), sizeof(extreme_ks)},
      ev{extreme_v, 2, 1, sizeof(double), sizeof(extreme_v)};
  CHECK(tq_distill_head_f64(one_query, sizeof(one_query), one_query,
                            sizeof(one_query), ekt, eks, ev, ev, nullptr, 0, 1,
                            0, 1, 1e-6, scratch, sizeof(scratch), &loss,
                            gradient, sizeof(gradient)) == TQ_OK);
  CHECK(loss.attention_kl == 2000 && loss.total == 2000 && gradient[0] == 2000);
  loss = {11, 22, 33};
  gradient[0] = 44;
  qs[1] = std::numeric_limits<double>::quiet_NaN();
  CHECK(call(1, 1) == TQ_NONFINITE && loss.total == 33 && gradient[0] == 44);
  std::printf("bounded distillation reference failures: %d\n", failures);
  return failures ? 1 : 0;
}
