#include "tqcodec/calibrate.h"
#include <cmath>
#include <cstdio>
#include <initializer_list>
#include <limits>
int main() {
  int failures = 0;
  auto check = [&](bool condition) {
    if (!condition)
      ++failures;
  };
  const double diagonal[] = {1, 0, 0, 0, 2, 0, 0, 0, 3};
  tq_consensus_fit fit{};
  check(tq_consensus_fit_f64(diagonal, sizeof(diagonal), 1, 3, 0, &fit) ==
        TQ_OK);
  check(std::abs(fit.alpha[0] - 36. / 49) < 1e-13);
  check(std::abs(fit.alpha[1] - 9. / 49) < 1e-13);
  check(std::abs(fit.alpha[2] - 4. / 49) < 1e-13);
  const double optimum = fit.normalized_objective;
  // Independent exhaustive grid cannot improve the analytic constrained fit.
  for (int i = 0; i <= 100; ++i)
    for (int j = 0; j <= 100 - i; ++j) {
      const double a = i / 100., b = j / 100., c = 1 - a - b;
      check((a * a + 4 * b * b + 9 * c * c) / 14 >= optimum - 1e-14);
    }
  for (double ridge : {.1, 10.}) {
    check(tq_consensus_fit_f64(diagonal, sizeof(diagonal), 1, 3, ridge, &fit) ==
          TQ_OK);
    check(std::abs(fit.alpha[0] + fit.alpha[1] + fit.alpha[2] - 1) < 1e-14);
    for (int i = 0; i <= 100; ++i)
      for (int j = 0; j <= 100 - i; ++j) {
        const double a = i / 100., b = j / 100., c = 1 - a - b;
        const double penalty = (a - 1. / 3) * (a - 1. / 3) +
                               (b - 1. / 3) * (b - 1. / 3) +
                               (c - 1. / 3) * (c - 1. / 3);
        check((a * a + 4 * b * b + 9 * c * c) / 14 + ridge * penalty >=
              fit.normalized_objective - 1e-13);
      }
  }
  check(tq_consensus_fit_f64(diagonal, sizeof(diagonal), 1, 3, 0, &fit) ==
        TQ_OK);
  double repeated[18];
  for (int i = 0; i < 18; ++i)
    repeated[i] = diagonal[i % 9];
  tq_consensus_fit multi{};
  check(tq_consensus_fit_f64(repeated, sizeof(repeated), 2, 3, 0, &multi) ==
        TQ_OK);
  for (int i = 0; i < 3; ++i)
    check(std::abs(multi.alpha[i] - fit.alpha[i]) < 1e-14);
  double huge[9];
  for (int i = 0; i < 9; ++i)
    huge[i] = diagonal[i] * 1e200;
  tq_consensus_fit large{};
  check(tq_consensus_fit_f64(huge, sizeof(huge), 1, 3, 0, &large) == TQ_OK);
  for (int i = 0; i < 3; ++i)
    check(std::abs(large.alpha[i] - fit.alpha[i]) < 1e-14);
  const double clean[] = {0, 1, 2};
  check(tq_consensus_fit_f64(clean, sizeof(clean), 1, 1, 0, &fit) == TQ_OK);
  check(fit.alpha[0] == 1 && fit.alpha[1] == 0 && fit.alpha[2] == 0);
  const double cancel[] = {1, -2, 3};
  check(tq_consensus_fit_f64(cancel, sizeof(cancel), 1, 1, 0, &fit) == TQ_OK);
  check(fit.normalized_objective < 1e-14 && fit.error_ratio_to_uniform < 1e-14);
  const double same[] = {2, 2, 2}, zero[] = {0, 0, 0};
  for (const double *data : {same, zero}) {
    check(tq_consensus_fit_f64(data, sizeof(same), 1, 1, 0, &fit) == TQ_OK);
    for (double a : fit.alpha)
      check(a == 1. / 3);
    check(fit.error_ratio_to_uniform == 1);
  }
  const double nearly_cancelled[] = {1, -1, 1e-7};
  check(tq_consensus_fit_f64(nearly_cancelled, sizeof(nearly_cancelled), 1, 1,
                             0, &fit) == TQ_OK);
  check(fit.alpha[0] == .5 && fit.alpha[1] == .5 && fit.alpha[2] == 0);
  check(fit.normalized_objective == 0 && fit.error_ratio_to_uniform == 0);
  const double ill_conditioned[] = {1, 1e-8, -1, 1e-8, 0, -1e-8};
  check(tq_consensus_fit_f64(ill_conditioned, sizeof(ill_conditioned), 1, 2, 0,
                             &fit) == TQ_OK);
  check(std::abs(fit.alpha[0] - .25) < 1e-14 &&
        std::abs(fit.alpha[1] - .25) < 1e-14);
  check(std::abs(fit.alpha[2] - .5) < 1e-14 &&
        fit.error_ratio_to_uniform < 1e-12);
  check(tq_consensus_fit_f64(diagonal, sizeof(diagonal), 1, 3, 1e300, &fit) ==
        TQ_OK);
  for (double a : fit.alpha)
    check(a == 1. / 3);
  fit = {{11, 22, 33}, 44, 55};
  check(tq_consensus_fit_f64(diagonal, 1, 1, 3, 0, &fit) == TQ_CAPACITY);
  check(tq_consensus_fit_f64(diagonal, sizeof(diagonal), SIZE_MAX, 3, 0,
                             &fit) == TQ_INVALID);
  check(tq_consensus_fit_f64(diagonal, sizeof(diagonal), 1, 3, -1, &fit) ==
        TQ_INVALID);
  double bad[] = {0, 1, std::numeric_limits<double>::quiet_NaN()};
  check(tq_consensus_fit_f64(bad, sizeof(bad), 1, 1, 0, &fit) == TQ_NONFINITE);
  check(fit.alpha[0] == 11 && fit.normalized_objective == 44 &&
        fit.error_ratio_to_uniform == 55);
  std::printf("calibration active-set reference failures: %d\n", failures);
  return failures ? 1 : 0;
}
