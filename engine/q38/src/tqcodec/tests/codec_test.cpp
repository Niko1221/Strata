#include "tqcodec/tqcodec.h"
#include <cmath>
#include <cstdio>
#include <limits>
#include <random>
#include <vector>
static int failures = 0;
#define CHECK(x)                                                               \
  do {                                                                         \
    if (!(x)) {                                                                \
      std::fprintf(stderr, "line %d: %s\n", __LINE__, #x);                     \
      ++failures;                                                              \
    }                                                                          \
  } while (0)
int main() {
  const double cb[] = {-1., -1. / 3, 1. / 3, 1.};
  const double x[] = {-.5, -.5, .5, .5};
  uint8_t packed[16] = {};
  double norm = -1, out[20] = {}, score = -1;
  tq_size sizes{};
  CHECK(tq_block_size(4, 2, &sizes) == TQ_OK);
  CHECK(sizes.payload_bytes == 1 && sizes.total_bytes == 1 + sizeof(double));
  CHECK(tq_block_size(SIZE_MAX, 6, &sizes) == TQ_INVALID);
  CHECK(tq_encode_f64(x, 4, cb, 2, packed, sizeof(packed), &norm) == TQ_OK);
  CHECK(norm == 1 && packed[0] == 0xa5);
  CHECK(tq_decode_f64(packed, 1, 4, cb, 2, norm, out) == TQ_OK);
  CHECK(out[0] == -1. / 3 && out[3] == 1. / 3);
  CHECK(tq_score_f64(x, packed, 1, 4, cb, 2, norm, &score) == TQ_OK);
  CHECK(std::abs(score - 2. / 3) < 1e-15);
  packed[0] = 0x7b;
  norm = 99;
  CHECK(tq_encode_f64(x, 4, cb, 2, packed, 0, &norm) == TQ_CAPACITY);
  CHECK(packed[0] == 0x7b && norm == 99);
  double bad[] = {0, 1, std::numeric_limits<double>::quiet_NaN(), 2};
  CHECK(tq_encode_f64(bad, 4, cb, 2, packed, 16, &norm) == TQ_NONFINITE);
  CHECK(packed[0] == 0x7b && norm == 99);
  for (unsigned b = 2; b <= 6; ++b) {
    std::vector<double> book(size_t{1} << b), input(17), decoded(17);
    for (size_t i = 0; i < book.size(); ++i)
      book[i] = -1 + 2. * i / (book.size() - 1);
    for (size_t i = 0; i < input.size(); ++i)
      input[i] = std::sin(double(i));
    std::vector<uint8_t> bytes((17 * b + 7) / 8);
    CHECK(tq_encode_f64(input.data(), 17, book.data(), b, bytes.data(),
                        bytes.size(), &norm) == TQ_OK);
    CHECK(tq_decode_f64(bytes.data(), bytes.size(), 17, book.data(), b, norm,
                        decoded.data()) == TQ_OK);
    double reference = 0;
    for (size_t i = 0; i < 17; ++i) {
      size_t j = 0;
      for (size_t k = 1; k < book.size(); ++k)
        if (std::abs(input[i] / norm - book[k]) <
            std::abs(input[i] / norm - book[j]))
          j = k;
      CHECK(decoded[i] == norm * book[j]);
      reference += input[i] * decoded[i];
    }
    CHECK(tq_score_f64(input.data(), bytes.data(), bytes.size(), 17,
                       book.data(), b, norm, &score) == TQ_OK);
    CHECK(std::abs(reference - score) < 1e-13);
  }
  const double g[] = {1, 0, 0, 0, 1, 0, 0, 0, 1, 1, 1, 1}, e[] = {-3, 4, 0},
               q[] = {2, -1, 3};
  CHECK(tq_qjl_encode_f64(e, 3, g, 4, packed, 16, &norm) == TQ_OK);
  CHECK(packed[0] == 14 && norm == 5);
  CHECK(tq_qjl_score_f64(q, 3, g, 4, packed, 1, norm, .5, &score) == TQ_OK);
  CHECK(std::abs(score - 2.5 * std::sqrt(std::acos(-1.) / 2)) < 1e-14);
  CHECK(tq_qjl_score_f64(q, 3, g, 4, packed, 1, norm, 0, &score) == TQ_OK &&
        score == 0);
  const double zero[] = {0, 0, 0};
  CHECK(tq_qjl_encode_f64(zero, 3, g, 4, packed, 16, &norm) == TQ_OK &&
        norm == 0 && packed[0] == 15);
  CHECK(tq_qjl_score_f64(q, 3, g, 4, packed, 1, norm, 1, &score) == TQ_OK &&
        score == 0);
  CHECK(tq_qjl_score_f64(q, 3, g, 4, packed, 1, norm, 1.1, &score) ==
        TQ_INVALID);
  const double *views[] = {q, zero, e}, alpha[] = {.25, .5, .25};
  CHECK(tq_shared_residual_f64(q, views, 3, alpha, out) == TQ_OK);
  for (int i = 0; i < 3; ++i)
    CHECK(out[i] == q[i] - .25 * q[i] - .25 * e[i]);
  CHECK(tq_shared_residual_f64(q, views, 3, alpha, const_cast<double *>(q)) ==
        TQ_INVALID);
  double weights_out[] = {.25, .5, .25};
  CHECK(tq_shared_residual_f64(q, views, 3, weights_out, weights_out) ==
        TQ_INVALID);
  CHECK(weights_out[0] == .25 && weights_out[1] == .5 && weights_out[2] == .25);
  double mutable_book[] = {-1., -1. / 3, 1. / 3, 1.};
  CHECK(tq_encode_f64(x, 4, mutable_book, 2, packed, 16, mutable_book) ==
        TQ_INVALID);
  CHECK(mutable_book[0] == -1.);
  double mutable_g[] = {1, 0, 0, 0, 1, 0, 0, 0, 1, 1, 1, 1};
  CHECK(tq_qjl_encode_f64(e, 3, mutable_g, 4, packed, 16, mutable_g) ==
        TQ_INVALID);
  CHECK(mutable_g[0] == 1.);
  // Independent Gaussian sketches: test mean and derived variance, not one
  // seed's output.
  std::mt19937 rng(20261023);
  std::normal_distribution<double> gaussian;
  constexpr size_t m = 16, trials = 6000;
  std::vector<double> matrix(m * 3);
  double sum = 0, sum2 = 0;
  uint8_t signs[2];
  for (size_t t = 0; t < trials; ++t) {
    for (double &v : matrix)
      v = gaussian(rng);
    CHECK(tq_qjl_encode_f64(e, 3, matrix.data(), m, signs, 2, &norm) == TQ_OK);
    CHECK(tq_qjl_score_f64(q, 3, matrix.data(), m, signs, 2, norm, 1, &score) ==
          TQ_OK);
    sum += score;
    sum2 += score * score;
  }
  const double mean = sum / trials, expected = -10.,
               variance = (std::acos(-1.) / 2 * 25 * 14 - 100) / m;
  CHECK(std::abs(mean - expected) < 5 * std::sqrt(variance / trials));
  CHECK(std::abs((sum2 / trials - mean * mean) / variance - 1) < .1);
  for (double beta : {0., .25, .5, 1.}) {
    tq_qjl_moments moments{};
    constexpr double scale = .25;
    CHECK(tq_qjl_error_moments_f64(q, e, 3, m, beta, scale, &moments) == TQ_OK);
    CHECK(std::abs(moments.bias - scale * (beta - 1) * expected) < 1e-14);
    CHECK(std::abs(moments.variance - scale * scale * beta * beta * variance) < 1e-14);
    const double empirical_bias = scale * (beta * mean - expected);
    const double empirical_var = scale * scale * beta * beta * (sum2 / trials - mean * mean);
    const double empirical_mse = scale * scale *
      (beta * beta * sum2 / trials - 2 * beta * expected * mean + expected * expected);
    CHECK(std::abs(empirical_bias - moments.bias) <
          5 * std::sqrt(moments.variance / trials) + 1e-14);
    CHECK(std::abs(empirical_var - moments.variance) < .1 * moments.variance + 1e-14);
    CHECK(std::abs(empirical_mse - moments.mean_squared_error) <
          .1 * moments.mean_squared_error + 1e-14);
    tq_qjl_moments negative{};
    CHECK(tq_qjl_error_moments_f64(q, e, 3, m, beta, -scale, &negative) == TQ_OK);
    CHECK(negative.bias == -moments.bias && negative.variance == moments.variance);
  }
  tq_qjl_moments moments{11, 22, 33};
  CHECK(tq_qjl_error_moments_f64(q, e, 3, 0, 1, 1, &moments) == TQ_INVALID);
  CHECK(tq_qjl_error_moments_f64(q, e, 3, m, 1.1, 1, &moments) == TQ_INVALID);
  CHECK(moments.bias == 11 && moments.variance == 22 && moments.mean_squared_error == 33);
  const double tiny_q[] = {1e-200}, huge_e[] = {1e300};
  CHECK(tq_qjl_error_moments_f64(tiny_q, huge_e, 1, 1, .5, 1e-200, &moments) == TQ_OK);
  CHECK(std::abs(moments.bias / -5e-101 - 1) < 1e-14);
  const double expected_tiny_var = .25e-200 * (std::acos(-1.) / 2 - 1);
  CHECK(std::abs(moments.variance / expected_tiny_var - 1) < 1e-14);
  CHECK(std::abs(moments.mean_squared_error / (expected_tiny_var + .25e-200) - 1) < 1e-14);
  const double crossed_q[] = {1e308, 1e-100}, crossed_e[] = {1e-100, 1e308};
  CHECK(tq_qjl_error_moments_f64(crossed_q, crossed_e, 2, 1, 0, 1e-308, &moments) == TQ_OK);
  CHECK(std::abs(moments.bias / -2e-100 - 1) < 1e-14 && moments.variance == 0);
  CHECK(std::abs(moments.mean_squared_error / 4e-200 - 1) < 1e-14);
  CHECK(tq_qjl_error_moments_f64(q, zero, 3, m, 1, 1, &moments) == TQ_OK);
  CHECK(moments.bias == 0 && moments.variance == 0 && moments.mean_squared_error == 0);
  CHECK(tq_qjl_error_moments_f64(q, e, 3, m, .5, 0, &moments) == TQ_OK);
  CHECK(moments.bias == 0 && moments.variance == 0 && moments.mean_squared_error == 0);
  moments = {11, 22, 33};
  CHECK(tq_qjl_error_moments_f64(bad, e, 3, m, 1, 1, &moments) == TQ_NONFINITE);
  CHECK(moments.bias == 11 && moments.variance == 22 && moments.mean_squared_error == 33);
  std::printf("tqcodec reference failures: %d\n", failures);
  return failures ? 1 : 0;
}
