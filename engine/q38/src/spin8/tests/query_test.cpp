#include "spin8/query.h"

#include <array>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <limits>
#include <random>
#include <vector>

namespace {
void check(bool ok, const char *message) {
  if (!ok) {
    std::cerr << "query test: " << message << '\n';
    std::exit(1);
  }
}
bool near(double a, double b, double tolerance = 1e-12) {
  return std::abs(a - b) <= tolerance;
}
using M8 = std::array<double, 64>;
using V8 = std::array<double, 8>;

// Independent left-octonion multiplication reference in the documented Fano
// orientation. Chirality ordering may swap the two blocks, changing Li's sign.
M8 octonion_left(size_t basis, double orientation) {
  M8 out{};
  if (basis == 0) {
    for (size_t i = 0; i < 8; ++i)
      out[i * 8 + i] = 1.;
    return out;
  }
  out[basis * 8] = orientation;
  out[basis] = -orientation;
  constexpr int triples[7][3] = {{1, 2, 3}, {1, 4, 5}, {1, 7, 6}, {2, 4, 6},
                                 {2, 5, 7}, {3, 4, 7}, {3, 6, 5}};
  for (const auto &t : triples)
    for (int n = 0; n < 3; ++n)
      if (static_cast<size_t>(t[n]) == basis) {
        const int y = t[(n + 1) % 3], z = t[(n + 2) % 3];
        out[z * 8 + y] = orientation;
        out[y * 8 + z] = -orientation;
      }
  return out;
}

M8 reference_matrix(const V8 &c, double orientation) {
  M8 contraction{}, base{}, result{};
  for (size_t i = 0; i < 8; ++i) {
    const auto l = octonion_left(i, orientation);
    for (size_t a = 0; a < 8; ++a) {
      base[i * 8 + a] = l[a * 8];
      for (size_t b = 0; b < 8; ++b)
        contraction[i * 8 + a] += l[a * 8 + b] * c[b];
    }
  }
  for (size_t i = 0; i < 8; ++i)
    for (size_t j = 0; j < 8; ++j)
      for (size_t a = 0; a < 8; ++a)
        result[i * 8 + j] += contraction[i * 8 + a] * base[j * 8 + a];
  return result;
}

void matrix_tests() {
  std::array<double, 2048> g{};
  check(spin8_clifford_generators(g.data()) == SPIN8_OK, "generators");
  const double orientation = g[256 + 1 * 16 + 8] > 0 ? 1. : -1.;
  check(std::abs(g[256 + 1 * 16 + 8]) == 1., "basis convention");
  V8 c0{1., 0., 0., 0., 0., 0., 0., 0.};
  M8 a{};
  check(spin8_query_matrix(c0.data(), a.data()) == SPIN8_OK, "identity matrix");
  for (size_t i = 0; i < 8; ++i)
    for (size_t j = 0; j < 8; ++j)
      check(a[i * 8 + j] == (i == j ? 1. : 0.), "identity exact");

  std::mt19937_64 rng(0x715A1ULL);
  std::normal_distribution<double> normal;
  for (size_t sample = 0; sample < 100; ++sample) {
    V8 c{};
    double norm = 0;
    for (auto &v : c) {
      v = normal(rng);
      norm += v * v;
    }
    norm = std::sqrt(norm);
    for (auto &v : c)
      v /= norm;
    check(spin8_query_matrix(c.data(), a.data()) == SPIN8_OK, "random matrix");
    const auto expected = reference_matrix(c, orientation);
    for (size_t i = 0; i < 64; ++i)
      check(near(a[i], expected[i]), "independent octonion reference");
    spin8_matrix_audit audit{};
    check(spin8_query_matrix_audit(c.data(), &audit) == SPIN8_OK,
          "matrix audit");
    check(audit.det_sign == 1 && audit.max_error < 1e-12 &&
              std::abs(audit.log_abs_det) < 1e-12,
          "proper orthogonal audit");
    double contraction = 0, expected_contraction = 0;
    V8 p{}, v{};
    for (auto &z : p)
      z = normal(rng);
    for (auto &z : v)
      z = normal(rng);
    check(spin8_query_contract(v.data(), p.data(), c.data(), &contraction) ==
              SPIN8_OK,
          "contraction");
    for (size_t i = 0; i < 8; ++i) {
      const auto l = octonion_left(i, orientation);
      for (size_t row = 0; row < 8; ++row)
        for (size_t col = 0; col < 8; ++col)
          expected_contraction += v[i] * p[row] * l[row * 8 + col] * c[col];
    }
    check(near(contraction, expected_contraction, 1e-10),
          "independent contraction");
  }
  c0[1] = .1;
  check(spin8_query_matrix(c0.data(), a.data()) == SPIN8_INVALID_ARGUMENT,
        "nonunit spinor");
  c0[1] = std::numeric_limits<double>::quiet_NaN();
  check(spin8_query_matrix(c0.data(), a.data()) == SPIN8_NONFINITE,
        "nonfinite spinor");
}

void tangent_tests() {
  std::array<double, 7> h{};
  V8 c{};
  std::array<double, 2048> g{};
  check(spin8_clifford_generators(g.data()) == SPIN8_OK, "tangent generators");
  const double orientation = g[256 + 1 * 16 + 8] > 0 ? 1. : -1.;
  check(spin8_query_spinor_from_tangent(h.data(), c.data()) == SPIN8_OK &&
            c[0] == 1.,
        "zero tangent");
  for (size_t i = 1; i < 8; ++i)
    check(c[i] == 0., "zero tangent tail");
  constexpr double pi_over_eight = 0.39269908169872415480783042290994;
  const double delta = 1e-6;
  for (size_t i = 0; i < 7; ++i) {
    h[i] = delta;
    check(spin8_query_spinor_from_tangent(h.data(), c.data()) == SPIN8_OK,
          "tangent finite difference");
    check(near((c[i + 1] / delta), pi_over_eight, 1e-10), "tangent derivative");
    M8 actual{};
    check(spin8_query_matrix(c.data(), actual.data()) == SPIN8_OK,
          "tangent matrix");
    V8 basis{};
    basis[i + 1] = 1.;
    const auto derivative = reference_matrix(basis, orientation);
    for (size_t row = 0; row < 8; ++row)
      for (size_t col = 0; col < 8; ++col) {
        const double difference =
            (actual[row * 8 + col] - (row == col ? 1. : 0.)) / delta;
        check(near(difference, pi_over_eight * derivative[row * 8 + col], 2e-6),
              "matrix tangent finite difference");
      }
    h[i] = 0.;
  }
  h[0] = std::numeric_limits<double>::max();
  h[1] = std::numeric_limits<double>::max();
  check(spin8_query_spinor_from_tangent(h.data(), c.data()) == SPIN8_OK,
        "overflow-safe tangent");
  double norm = 0;
  for (double x : c)
    norm += x * x;
  check(near(norm, 1.) && c[1] > 0 && c[2] > 0,
        "large tangent remains unit and directional");
  h[0] = std::numeric_limits<double>::infinity();
  check(spin8_query_spinor_from_tangent(h.data(), c.data()) == SPIN8_NONFINITE,
        "nonfinite tangent");
}

void transform_tests() {
  std::array<double, 7> h{.4, -.2, .1, 0., 0., 0., 0.};
  V8 c{};
  M8 a{};
  check(spin8_query_spinor_from_tangent(h.data(), c.data()) == SPIN8_OK &&
            spin8_query_matrix(c.data(), a.data()) == SPIN8_OK,
        "transform setup");
  std::array<double, 42> src{}, dst{}, inplace{};
  for (size_t r = 0; r < 2; ++r)
    for (size_t j = 0; j < 19; ++j)
      src[r * 21 + j] = static_cast<double>(r * 19 + j + 1) / 11.;
  inplace = src;
  check(spin8_query_transform(c.data(), SPIN8_F64, src.data(), sizeof(src), 21,
                              dst.data(), sizeof(dst), 21, 2, 19) == SPIN8_OK,
        "strided transform");
  for (size_t r = 0; r < 2; ++r) {
    for (size_t base : {0U, 8U})
      for (size_t i = 0; i < 8; ++i) {
        double expected = 0;
        for (size_t j = 0; j < 8; ++j)
          expected += a[i * 8 + j] * src[r * 21 + base + j];
        check(near(dst[r * 21 + base + i], expected), "block values");
      }
    for (size_t j = 16; j < 19; ++j)
      check(dst[r * 21 + j] == src[r * 21 + j], "tail identity");
  }
  check(spin8_query_transform(c.data(), SPIN8_F64, inplace.data(),
                              sizeof(inplace), 21, inplace.data(),
                              sizeof(inplace), 21, 2, 19) == SPIN8_OK,
        "exact in-place");
  for (size_t r = 0; r < 2; ++r)
    for (size_t j = 0; j < 19; ++j)
      check(near(inplace[r * 21 + j], dst[r * 21 + j]), "in-place values");
  std::array<float, 19> fsrc{}, fdst{};
  for (size_t j = 0; j < 19; ++j)
    fsrc[j] = static_cast<float>(src[j]);
  check(spin8_query_transform(c.data(), SPIN8_F32, fsrc.data(), sizeof(fsrc),
                              19, fdst.data(), sizeof(fdst), 19, 1,
                              19) == SPIN8_OK,
        "F32 transform");
  for (size_t j = 0; j < 19; ++j)
    check(near(fdst[j], dst[j], 2e-6), "F32 values");

  const auto before = dst;
  check(spin8_query_transform(c.data(), SPIN8_F64, src.data(), sizeof(src) - 1,
                              21, dst.data(), sizeof(dst), 21, 2,
                              19) == SPIN8_OK,
        "unused final padding is not required");
  check(spin8_query_transform(c.data(), SPIN8_F64, src.data(), 39 * 8 - 1, 21,
                              dst.data(), sizeof(dst), 21, 2,
                              19) == SPIN8_INVALID_ARGUMENT,
        "capacity limit");
  check(spin8_query_transform(c.data(), SPIN8_F64, src.data(), sizeof(src), 21,
                              dst.data(), 39 * 8 - 1, 21, 2,
                              19) == SPIN8_INVALID_ARGUMENT,
        "destination capacity limit");
  check(spin8_query_transform(c.data(), SPIN8_F64, src.data(), sizeof(src), 18,
                              dst.data(), sizeof(dst), 21, 2,
                              19) == SPIN8_INVALID_ARGUMENT,
        "short stride");
  check(spin8_query_transform(c.data(), SPIN8_F64, src.data(), sizeof(src), 21,
                              const_cast<double *>(src.data()) + 1,
                              sizeof(src) - 8, 21, 2,
                              19) == SPIN8_INVALID_ARGUMENT,
        "partial overlap");
  check(spin8_query_transform(c.data(), SPIN8_F64, src.data(), SIZE_MAX,
                              SIZE_MAX, dst.data(), sizeof(dst), 21, 2,
                              19) == SPIN8_INVALID_ARGUMENT,
        "arithmetic overflow");
  check(spin8_query_transform(c.data(), SPIN8_F16, src.data(), sizeof(src), 21,
                              dst.data(), sizeof(dst), 21, 2,
                              19) == SPIN8_UNSUPPORTED,
        "unsupported dtype");
  auto bad = src;
  bad[0] = std::numeric_limits<double>::quiet_NaN();
  check(spin8_query_transform(c.data(), SPIN8_F64, bad.data(), sizeof(bad), 21,
                              dst.data(), sizeof(dst), 21, 2,
                              19) == SPIN8_NONFINITE,
        "nonfinite input");
  check(dst == before, "failure leaves output unchanged");
}

void controller_tests() {
  constexpr size_t width = 19;
  std::array<double, 8 * width> down{};
  std::array<double, 56> up{};
  std::array<double, width> src{}, dst{};
  for (size_t i = 0; i < width; ++i)
    src[i] = static_cast<double>(i + 1) / 20.;
  spin8_query_controller *ctl = nullptr;
  check(spin8_query_controller_create(width, down.data(), sizeof(down),
                                      up.data(), sizeof(up),
                                      &ctl) == SPIN8_OK &&
            ctl,
        "zero controller create");
  check(spin8_query_controller_apply(ctl, SPIN8_F64, src.data(), sizeof(src),
                                     width, dst.data(), sizeof(dst), width,
                                     1) == SPIN8_OK &&
            src == dst,
        "zero Wup exact identity");
  spin8_query_controller_free(ctl);
  down[0] = 1.;
  up[0] = 1.;
  check(spin8_query_controller_create(width, down.data(), sizeof(down),
                                      up.data(), sizeof(up), &ctl) == SPIN8_OK,
        "nonzero controller create");
  down[0] = 0.;
  up[0] = 0.;
  check(spin8_query_controller_apply(ctl, SPIN8_F64, src.data(), sizeof(src),
                                     width, dst.data(), sizeof(dst), width,
                                     1) == SPIN8_OK,
        "nonzero controller apply");
  bool changed = false;
  for (size_t i = 0; i < 16; ++i)
    changed |= std::abs(src[i] - dst[i]) > 1e-8;
  check(changed, "nonzero controller changes blocks");
  for (size_t i = 16; i < width; ++i)
    check(src[i] == dst[i], "controller tail identity");
  std::array<float, width> float_src{}, float_dst{};
  for (size_t i = 0; i < width; ++i)
    float_src[i] = static_cast<float>(src[i]);
  check(spin8_query_controller_apply(ctl, SPIN8_F32, float_src.data(),
                                     sizeof(float_src), width, float_dst.data(),
                                     sizeof(float_dst), width, 1) == SPIN8_OK,
        "controller F32");
  for (size_t i = 0; i < width; ++i)
    check(near(float_dst[i], dst[i], 2e-6), "controller F32 values");
  const auto preserved = dst;
  auto bad = src;
  bad[0] = std::numeric_limits<double>::quiet_NaN();
  check(spin8_query_controller_apply(ctl, SPIN8_F64, bad.data(), sizeof(bad),
                                     width, dst.data(), sizeof(dst), width,
                                     1) == SPIN8_NONFINITE &&
            dst == preserved,
        "controller nonfinite atomicity");
  check(spin8_query_controller_apply(ctl, SPIN8_F64, src.data(), sizeof(src),
                                     width, dst.data(), sizeof(dst) - 8, width,
                                     1) == SPIN8_INVALID_ARGUMENT,
        "controller destination capacity");
  check(spin8_query_controller_apply(
            ctl, SPIN8_F64, src.data(), sizeof(src), width,
            const_cast<double *>(src.data()) + 1, sizeof(src) - 8, width,
            1) == SPIN8_INVALID_ARGUMENT,
        "controller overlap");
  spin8_query_controller_free(ctl);
  check(spin8_query_controller_create(width, down.data(), sizeof(down) - 1,
                                      up.data(), sizeof(up),
                                      &ctl) == SPIN8_INVALID_ARGUMENT &&
            !ctl,
        "controller capacity");
  up[0] = std::numeric_limits<double>::infinity();
  check(spin8_query_controller_create(width, down.data(), sizeof(down),
                                      up.data(), sizeof(up),
                                      &ctl) == SPIN8_NONFINITE &&
            !ctl,
        "controller nonfinite weight");
}
} // namespace

int main() {
  matrix_tests();
  tangent_tests();
  transform_tests();
  controller_tests();
  return 0;
}
