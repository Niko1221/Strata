#include "spin8/query.h"
#include "spin8/query_v3.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <limits>
#include <memory>
#include <random>
#include <vector>

namespace {
using V8 = std::array<double, 8>;
using A7 = std::array<double, 7>;
using M8 = std::array<double, 64>;
void check(bool ok, const char *message) {
  if (!ok) {
    std::cerr << "query v3 test: " << message << '\n';
    std::exit(1);
  }
}
bool near(double a, double b, double eps = 1e-12) {
  return std::abs(a - b) <= eps;
}
// Separate octonion multiplication table, independent of the production
// gamma-to-C-to-J construction. Chirality can swap the two gamma blocks.
M8 octonion(size_t basis, double orientation) {
  M8 l{};
  if (!basis) {
    for (size_t i = 0; i < 8; ++i)
      l[i * 8 + i] = 1.;
    return l;
  }
  l[basis * 8] = orientation;
  l[basis] = -orientation;
  constexpr int triples[7][3] = {{1, 2, 3}, {1, 4, 5}, {1, 7, 6}, {2, 4, 6},
                                 {2, 5, 7}, {3, 4, 7}, {3, 6, 5}};
  for (const auto &t : triples)
    for (int k = 0; k < 3; ++k)
      if (static_cast<size_t>(t[k]) == basis) {
        const int y = t[(k + 1) % 3], z = t[(k + 2) % 3];
        l[z * 8 + y] = orientation;
        l[y * 8 + z] = -orientation;
      }
  return l;
}
M8 reference_c(const V8 &c, double orientation) {
  M8 result{};
  for (size_t i = 0; i < 8; ++i) {
    const auto l = octonion(i, orientation);
    for (size_t a = 0; a < 8; ++a)
      for (size_t b = 0; b < 8; ++b)
        result[i * 8 + a] += l[a * 8 + b] * c[b];
  }
  return result;
}
M8 reference_a(const V8 &c, double orientation) {
  const auto value = reference_c(c, orientation);
  const auto base =
      reference_c(V8{1., 0., 0., 0., 0., 0., 0., 0.}, orientation);
  M8 out{};
  for (size_t i = 0; i < 8; ++i)
    for (size_t j = 0; j < 8; ++j)
      for (size_t k = 0; k < 8; ++k)
        out[i * 8 + j] += value[i * 8 + k] * base[j * 8 + k];
  return out;
}
void basis_tests() {
  std::array<double, 2048> gamma{};
  check(spin8_clifford_generators(gamma.data()) == SPIN8_OK, "gamma");
  const double orientation = gamma[256 + 16 + 8] > 0. ? 1. : -1.;
  std::array<double, 448> j{};
  spin8_query_v3_basis_report audit{};
  check(spin8_query_v3_generators(j.data()) == SPIN8_OK, "generators");
  check(spin8_query_v3_basis_audit(&audit) == SPIN8_OK, "basis audit");
  check(audit.normed_max_error == 0. && audit.polarization_max_error == 0. &&
            audit.skew_max_error == 0. && audit.clifford_max_error == 0.,
        "exact normed tensor, C polarization, skew and Clifford audit");
  std::array<M8, 8> c_basis{};
  for (size_t a = 0; a < 8; ++a) {
    V8 e{};
    e[a] = 1.;
    c_basis[a] = reference_c(e, orientation);
    if (a) {
      const auto expected = reference_a(e, orientation);
      for (size_t i = 0; i < 64; ++i)
        check(j[(a - 1) * 64 + i] == expected[i], "independent J basis");
    }
  }
  for (size_t a = 0; a < 8; ++a)
    for (size_t b = 0; b < 8; ++b)
      for (size_t i = 0; i < 8; ++i)
        for (size_t k = 0; k < 8; ++k) {
          double actual = 0.;
          for (size_t t = 0; t < 8; ++t)
            actual += c_basis[a][i * 8 + t] * c_basis[b][k * 8 + t] +
                      c_basis[b][i * 8 + t] * c_basis[a][k * 8 + t];
          check(actual == (a == b && i == k ? 2. : 0.), "all C polarizations");
        }
  for (size_t a = 0; a < 7; ++a)
    for (size_t b = 0; b < 7; ++b)
      for (size_t i = 0; i < 8; ++i)
        for (size_t k = 0; k < 8; ++k) {
          check(j[a * 64 + i * 8 + k] == -j[a * 64 + k * 8 + i], "J skew");
          double sum = 0.;
          for (size_t t = 0; t < 8; ++t)
            sum += j[a * 64 + i * 8 + t] * j[b * 64 + t * 8 + k] +
                   j[b * 64 + i * 8 + t] * j[a * 64 + t * 8 + k];
          check(sum == (a == b && i == k ? -2. : 0.), "Cl(0,7)");
        }
  std::mt19937_64 rng(20261023);
  std::normal_distribution<double> normal;
  for (size_t sample = 0; sample < 512; ++sample) {
    V8 c{};
    double norm = 0.;
    for (auto &v : c) {
      v = normal(rng);
      norm = std::hypot(norm, v);
    }
    for (auto &v : c)
      v /= norm;
    M8 actual{}, old{};
    check(spin8_query_v3_matrix(c.data(), actual.data()) == SPIN8_OK, "matrix");
    check(spin8_query_matrix(c.data(), old.data()) == SPIN8_OK,
          "v2 comparison");
    const auto expected = reference_a(c, orientation);
    for (size_t i = 0; i < 64; ++i) {
      check(near(actual[i], expected[i]), "independent matrix");
      check(near(actual[i], old[i]), "v2 matrix unchanged");
    }
    spin8_matrix_audit report{};
    check(spin8_audit_matrix(actual.data(), &report) == SPIN8_OK &&
              report.det_sign == 1 && report.max_error < 1e-12 &&
              std::abs(report.log_abs_det) < 1e-12,
          "SO(8) audit");
  }
  // A product of two coordinate rotations contains J1 J2, outside span(I,J).
  A7 a1{.3, 0., 0., 0., 0., 0., 0.}, a2{0., .3, 0., 0., 0., 0., 0.};
  V8 c1{}, c2{};
  spin8_query_v3_spinor_from_tangent(a1.data(), c1.data());
  spin8_query_v3_spinor_from_tangent(a2.data(), c2.data());
  const auto m1 = reference_a(c1, orientation),
             m2 = reference_a(c2, orientation);
  M8 product{}, projection{};
  for (size_t i = 0; i < 8; ++i)
    for (size_t k = 0; k < 8; ++k)
      for (size_t t = 0; t < 8; ++t)
        product[i * 8 + k] += m1[i * 8 + t] * m2[t * 8 + k];
  double trace = 0.;
  for (size_t i = 0; i < 8; ++i)
    trace += product[i * 8 + i] / 8.;
  for (size_t i = 0; i < 8; ++i)
    projection[i * 8 + i] = trace;
  for (size_t a = 0; a < 7; ++a) {
    double coefficient = 0.;
    for (size_t i = 0; i < 64; ++i)
      coefficient += product[i] * j[a * 64 + i] / 8.;
    for (size_t i = 0; i < 64; ++i)
      projection[i] += coefficient * j[a * 64 + i];
  }
  double residual = 0.;
  for (size_t i = 0; i < 64; ++i)
    residual = std::hypot(residual, product[i] - projection[i]);
  check(residual > .01, "family not closed under composition");
  M8 sentinel{};
  sentinel.fill(91.);
  V8 bad{};
  check(spin8_query_v3_matrix(bad.data(), sentinel.data()) ==
                SPIN8_INVALID_ARGUMENT &&
            sentinel[0] == 91.,
        "nonunit failure atomic");
  bad[0] = std::numeric_limits<double>::quiet_NaN();
  check(spin8_query_v3_matrix(bad.data(), sentinel.data()) == SPIN8_NONFINITE &&
            sentinel[0] == 91.,
        "nonfinite failure atomic");
}
void chart_tests() {
  A7 a{};
  V8 c{};
  check(spin8_query_v3_spinor_from_tangent(a.data(), c.data()) == SPIN8_OK &&
            c == V8{1., 0., 0., 0., 0., 0., 0., 0.},
        "exact origin");
  constexpr double h = 1e-6;
  std::array<double, 448> generators{};
  check(spin8_query_v3_generators(generators.data()) == SPIN8_OK,
        "chart J basis");
  for (size_t j = 0; j < 7; ++j) {
    a[j] = h;
    V8 plus{}, minus{};
    spin8_query_v3_spinor_from_tangent(a.data(), plus.data());
    a[j] = -h;
    spin8_query_v3_spinor_from_tangent(a.data(), minus.data());
    for (size_t i = 0; i < 8; ++i)
      check(near((plus[i] - minus[i]) / (2. * h), i == j + 1 ? 1. : 0., 1e-10),
            "unit derivative at origin");
    M8 mp{}, mm{};
    check(spin8_query_v3_matrix(plus.data(), mp.data()) == SPIN8_OK &&
              spin8_query_v3_matrix(minus.data(), mm.data()) == SPIN8_OK,
          "matrix derivative samples");
    for (size_t i = 0; i < 64; ++i)
      check(near((mp[i] - mm[i]) / (2. * h), generators[j * 64 + i], 1e-10),
            "matrix derivative is J");
    a[j] = 0.;
  }
  a = {.12, -.07, .21, .13, -.09, .04, -.17};
  constexpr double tau = .4142135623730950488;
  double den2 = tau * tau;
  for (double v : a)
    den2 += v * v;
  const double den = std::sqrt(den2);
  A7 u{};
  double s2 = 1.;
  for (size_t i = 0; i < 7; ++i) {
    u[i] = tau * a[i] / den;
    s2 += u[i] * u[i];
  }
  const double s = std::sqrt(s2);
  for (size_t j = 0; j < 7; ++j) {
    A7 du{};
    double dot = 0.;
    for (size_t i = 0; i < 7; ++i) {
      du[i] = tau / den * ((i == j ? 1. : 0.) - a[i] * a[j] / den2);
      dot += u[i] * du[i];
    }
    // Five-point differences avoid cancellation in c0 near one while keeping
    // the truncation error O(step^4). Do not widen the acceptance threshold.
    constexpr double step = 1e-4;
    auto p = a, m = a, pp = a, mm = a;
    p[j] += step;
    m[j] -= step;
    pp[j] += 2. * step;
    mm[j] -= 2. * step;
    V8 plus{}, minus{}, plus2{}, minus2{}, difference{};
    spin8_query_v3_spinor_from_tangent(p.data(), plus.data());
    spin8_query_v3_spinor_from_tangent(m.data(), minus.data());
    spin8_query_v3_spinor_from_tangent(pp.data(), plus2.data());
    spin8_query_v3_spinor_from_tangent(mm.data(), minus2.data());
    for (size_t i = 0; i < 8; ++i)
      difference[i] =
          (minus2[i] - 8. * minus[i] + 8. * plus[i] - plus2[i]) / (12. * step);
    check(near(difference[0], -dot / (s * s2), 2e-10), "chart c0 Jacobian");
    for (size_t i = 0; i < 7; ++i)
      check(near(difference[i + 1], du[i] / s - u[i] * dot / (s * s2), 2e-10),
            "chart Jacobian");
  }
  for (double scale : {std::numeric_limits<double>::denorm_min(), 1e-200, 1e-12,
                       1., 1e100, std::numeric_limits<double>::max()}) {
    for (size_t i = 0; i < 7; ++i)
      a[i] = i % 2 ? -scale : scale;
    check(spin8_query_v3_spinor_from_tangent(a.data(), c.data()) == SPIN8_OK,
          "finite chart all scales");
    double norm = 0., tangent = 0.;
    for (size_t i = 0; i < 8; ++i) {
      check(std::isfinite(c[i]), "finite spinor");
      norm += c[i] * c[i];
      if (i)
        tangent = std::hypot(tangent, c[i]);
    }
    check(near(norm, 1.) &&
              std::atan2(tangent, c[0]) <= .39269908169872416 + 1e-15,
          "bounded angle and unit norm");
  }
  a.fill(std::numeric_limits<double>::max());
  check(spin8_query_v3_spinor_from_tangent(a.data(), c.data()) == SPIN8_OK,
        "seven DBL_MAX");
  const V8 before = c;
  a[4] = std::numeric_limits<double>::infinity();
  check(spin8_query_v3_spinor_from_tangent(a.data(), c.data()) ==
                SPIN8_NONFINITE &&
            c == before,
        "chart failure atomic");
}
struct Weights {
  size_t width;
  std::vector<double> win;
  V8 bin{};
  std::array<double, 56> wout{};
  A7 bout{};
  explicit Weights(size_t n) : width(n), win(8 * n) {
    for (size_t i = 0; i < win.size(); ++i)
      win[i] = .01 * static_cast<double>((i % 11) + 1);
  }
};
using Controller = std::unique_ptr<spin8_query_v3_controller,
                                   decltype(&spin8_query_v3_controller_free)>;
Controller create(const Weights &w) {
  spin8_query_v3_controller *p = nullptr;
  check(spin8_query_v3_controller_create(
            3, w.width, w.win.data(), w.win.size() * sizeof(double),
            w.bin.data(), sizeof(w.bin), w.wout.data(), sizeof(w.wout),
            w.bout.data(), sizeof(w.bout), &p) == SPIN8_OK,
        "create v3");
  return Controller(p, spin8_query_v3_controller_free);
}
V8 get_spinor(const Weights &w, const std::vector<double> &q) {
  const auto ctl = create(w);
  V8 c{};
  check(spin8_query_v3_controller_spinors(
            ctl.get(), SPIN8_F64, q.data(), q.size() * sizeof(double), w.width,
            c.data(), sizeof(c), 8, 1) == SPIN8_OK,
        "get spinor");
  return c;
}
void controller_tests() {
  Weights w(19);
  const size_t rows = 3, stride = 22;
  std::vector<double> q(rows * stride, -917.), dst(rows * stride, 81.);
  for (size_t r = 0; r < rows; ++r)
    for (size_t i = 0; i < w.width; ++i)
      q[r * stride + i] = .03 * static_cast<double>(i + 2 * r + 1);
  auto identity = create(w);
  check(spin8_query_v3_controller_version(identity.get()) == 3 &&
            spin8_query_v3_controller_version(nullptr) == 0,
        "version");
  check(spin8_query_v3_controller_apply(
            identity.get(), SPIN8_F64, q.data(), q.size() * 8, stride,
            dst.data(), dst.size() * 8, stride, rows) == SPIN8_OK,
        "identity apply");
  for (size_t r = 0; r < rows; ++r)
    for (size_t i = 0; i < stride; ++i)
      check(dst[r * stride + i] == (i < w.width ? q[r * stride + i] : 81.),
            "exact identity and padding");
  w.bin = {.01, -.02, .03, -.04, .05, -.06, .07, -.08};
  for (size_t i = 0; i < 56; ++i)
    w.wout[i] = .02 * static_cast<double>((i % 5) + 1);
  w.bout = {.01, -.02, .03, -.04, .05, -.06, .07};
  auto ctl = create(w);
  std::vector<double> cs(30, 43.);
  check(spin8_query_v3_controller_spinors(ctl.get(), SPIN8_F64, q.data(),
                                          q.size() * 8, stride, cs.data(),
                                          cs.size() * 8, 10, rows) == SPIN8_OK,
        "spinor rows");
  for (size_t r = 0; r < rows; ++r) {
    V8 hidden{};
    for (size_t k = 0; k < 8; ++k) {
      double z = w.bin[k];
      for (size_t i = 0; i < w.width; ++i)
        z += w.win[k * w.width + i] * q[r * stride + i];
      hidden[k] = std::tanh(z);
    }
    A7 a = w.bout;
    for (size_t i = 0; i < 7; ++i)
      for (size_t k = 0; k < 8; ++k)
        a[i] += w.wout[i * 8 + k] * hidden[k];
    V8 expected{};
    spin8_query_v3_spinor_from_tangent(a.data(), expected.data());
    for (size_t i = 0; i < 8; ++i)
      check(near(cs[r * 10 + i], expected[i]), "biases and NO second RMS");
    check(cs[r * 10 + 8] == 43. && cs[r * 10 + 9] == 43., "spinor padding");
  }
  check(spin8_query_v3_controller_apply(
            ctl.get(), SPIN8_F64, q.data(), q.size() * 8, stride, dst.data(),
            dst.size() * 8, stride, rows) == SPIN8_OK,
        "apply");
  std::vector<double> expected(dst.size(), 81.);
  for (size_t r = 0; r < rows; ++r) {
    M8 m{};
    spin8_query_matrix(cs.data() + r * 10, m.data());
    for (size_t base = 0; base < 16; base += 8)
      for (size_t i = 0; i < 8; ++i) {
        double sum = 0.;
        for (size_t j = 0; j < 8; ++j)
          sum += m[i * 8 + j] * q[r * stride + base + j];
        expected[r * stride + base + i] = sum;
      }
    for (size_t i = 16; i < 19; ++i)
      expected[r * stride + i] = q[r * stride + i];
  }
  for (size_t i = 0; i < dst.size(); ++i)
    check(near(dst[i], expected[i]), "all head blocks, tail, padding");
  auto inplace = q;
  check(spin8_query_v3_controller_apply(
            ctl.get(), SPIN8_F64, inplace.data(), inplace.size() * 8, stride,
            inplace.data(), inplace.size() * 8, stride, rows) == SPIN8_OK,
        "inplace");
  for (size_t r = 0; r < rows; ++r)
    for (size_t i = 0; i < w.width; ++i)
      check(near(inplace[r * stride + i], expected[r * stride + i]),
            "inplace result");
  // Mutate every caller-owned weight; copied state must be immutable.
  std::fill(w.win.begin(), w.win.end(), 0.);
  w.bin.fill(0.);
  w.wout.fill(0.);
  w.bout.fill(0.);
  std::vector<double> copied(dst.size(), 81.);
  check(spin8_query_v3_controller_apply(
            ctl.get(), SPIN8_F64, q.data(), q.size() * 8, stride, copied.data(),
            copied.size() * 8, stride, rows) == SPIN8_OK &&
            copied == dst,
        "all weights copied");
  const auto sentinel = dst;
  const size_t extent = ((rows - 1) * stride + w.width) * 8;
  check(spin8_query_v3_controller_apply(
            ctl.get(), SPIN8_F64, q.data(), extent - 1, stride, dst.data(),
            dst.size() * 8, stride, rows) == SPIN8_INVALID_ARGUMENT &&
            dst == sentinel,
        "source capacity");
  check(spin8_query_v3_controller_apply(
            ctl.get(), SPIN8_F64, q.data(), q.size() * 8, stride, dst.data(),
            extent - 1, stride, rows) == SPIN8_INVALID_ARGUMENT &&
            dst == sentinel,
        "destination capacity");
  check(spin8_query_v3_controller_apply(
            ctl.get(), SPIN8_F64, q.data(), q.size() * 8, stride, q.data() + 1,
            (q.size() - 1) * 8, stride, rows) == SPIN8_INVALID_ARGUMENT,
        "partial alias");
  check(spin8_query_v3_controller_spinors(
            ctl.get(), SPIN8_F64, q.data(), q.size() * 8, stride, q.data(),
            q.size() * 8, 8, rows) == SPIN8_INVALID_ARGUMENT,
        "spinor alias");
  check(spin8_query_v3_controller_apply(
            ctl.get(), SPIN8_F16, q.data(), q.size() * 8, stride, dst.data(),
            dst.size() * 8, stride, rows) == SPIN8_UNSUPPORTED,
        "unsupported dtype");
  check(spin8_query_v3_controller_apply(
            ctl.get(), SPIN8_F64, q.data(), q.size() * 8, stride, dst.data(),
            dst.size() * 8, stride, 0) == SPIN8_INVALID_ARGUMENT,
        "zero rows");
  q[2 * stride + 17] = std::numeric_limits<double>::quiet_NaN();
  check(spin8_query_v3_controller_apply(
            ctl.get(), SPIN8_F64, q.data(), q.size() * 8, stride, dst.data(),
            dst.size() * 8, stride, rows) == SPIN8_NONFINITE &&
            dst == sentinel,
        "last row nonfinite failure atomic");
  const auto cs_before = cs;
  check(spin8_query_v3_controller_spinors(
            ctl.get(), SPIN8_F64, q.data(), q.size() * 8, stride, cs.data(),
            cs.size() * 8, 10, rows) == SPIN8_NONFINITE &&
            cs == cs_before,
        "spinor failure atomic");
  check(spin8_query_v3_controller_spinors(ctl.get(), SPIN8_F64, q.data(),
                                          q.size() * 8, stride, cs.data(),
                                          ((rows - 1) * 10 + 8) * 8 - 1, 10,
                                          rows) == SPIN8_INVALID_ARGUMENT &&
            cs == cs_before,
        "spinor capacity");
  check(spin8_query_v3_controller_apply(
            ctl.get(), SPIN8_F64, q.data(), SIZE_MAX, SIZE_MAX, dst.data(),
            dst.size() * 8, stride, rows) == SPIN8_INVALID_ARGUMENT,
        "row extent overflow");
  check(spin8_query_v3_controller_apply(
            ctl.get(), SPIN8_F64, q.data(), q.size() * 8, 18, dst.data(),
            dst.size() * 8, stride, rows) == SPIN8_INVALID_ARGUMENT,
        "stride smaller than width");
  // Finite huge input still has an exact identity path despite nonzero Win.
  q.assign(q.size(), std::numeric_limits<double>::max());
  check(spin8_query_v3_controller_apply(
            identity.get(), SPIN8_F64, q.data(), q.size() * 8, stride,
            dst.data(), dst.size() * 8, stride, rows) == SPIN8_OK,
        "huge exact identity");
  Weights overflow(19);
  overflow.win[0] = std::numeric_limits<double>::max();
  overflow.wout[0] = 1.;
  const auto overflow_ctl = create(overflow);
  dst = sentinel;
  check(spin8_query_v3_controller_apply(
            overflow_ctl.get(), SPIN8_F64, q.data(), q.size() * 8, stride,
            dst.data(), dst.size() * 8, stride, rows) == SPIN8_NONFINITE &&
            dst == sentinel,
        "arithmetic overflow atomic");
  Weights wf(9);
  wf.bout[2] = .2;
  wf.wout[0] = .1;
  auto cf = create(wf);
  std::array<float, 24> fq{}, fd{};
  for (size_t i = 0; i < fq.size(); ++i)
    fq[i] = static_cast<float>(i) * .1f;
  fd.fill(91.f);
  check(spin8_query_v3_controller_apply(cf.get(), SPIN8_F32, fq.data(),
                                        sizeof(fq), 12, fd.data(), sizeof(fd),
                                        12, 2) == SPIN8_OK,
        "F32 apply");
  for (size_t r = 0; r < 2; ++r)
    check(fd[r * 12 + 8] == fq[r * 12 + 8] && fd[r * 12 + 9] == 91.f,
          "F32 tail padding");
  for (size_t r = 0; r < 2; ++r) {
    std::vector<double> row(9);
    for (size_t i = 0; i < 9; ++i)
      row[i] = fq[r * 12 + i];
    const auto c = get_spinor(wf, row);
    M8 m{};
    check(spin8_query_matrix(c.data(), m.data()) == SPIN8_OK,
          "F32 oracle matrix");
    for (size_t i = 0; i < 8; ++i) {
      double sum = 0.;
      for (size_t j = 0; j < 8; ++j)
        sum += m[i * 8 + j] * row[j];
      check(near(fd[r * 12 + i], sum, 3e-7), "F32 versus F64 oracle");
    }
  }
  auto fi = fq;
  check(spin8_query_v3_controller_apply(cf.get(), SPIN8_F32, fi.data(),
                                        sizeof(fi), 12, fi.data(), sizeof(fi),
                                        12, 2) == SPIN8_OK,
        "F32 inplace");
  for (size_t r = 0; r < 2; ++r)
    for (size_t i = 0; i < 9; ++i)
      check(fi[r * 12 + i] == fd[r * 12 + i], "F32 result");
}
void gradient_and_validation_tests() {
  Weights w(8);
  const std::vector<double> q{.3, -.1, .4, .2, -.5, .7, .8, -.6};
  constexpr double h = 1e-6;
  auto plus = w, minus = w;
  plus.wout[0] = h;
  minus.wout[0] = -h;
  const auto cp = get_spinor(plus, q), cm = get_spinor(minus, q);
  double hidden = 0.;
  for (size_t i = 0; i < 8; ++i)
    hidden += w.win[i] * q[i];
  hidden = std::tanh(hidden);
  const double gradient = (cp[1] - cm[1]) / (2. * h);
  check(std::abs(gradient) > 1e-4 && near(gradient, hidden, 1e-10),
        "nonzero Wout first-step gradient");
  const auto output_plus_ctl = create(plus), output_minus_ctl = create(minus);
  V8 output_plus{}, output_minus{};
  check(spin8_query_v3_controller_apply(
            output_plus_ctl.get(), SPIN8_F64, q.data(), q.size() * 8, 8,
            output_plus.data(), sizeof(output_plus), 8, 1) == SPIN8_OK &&
            spin8_query_v3_controller_apply(
                output_minus_ctl.get(), SPIN8_F64, q.data(), q.size() * 8, 8,
                output_minus.data(), sizeof(output_minus), 8, 1) == SPIN8_OK,
        "Wout output gradient samples");
  V8 axis{};
  axis[1] = 1.;
  M8 derivative{};
  check(spin8_query_matrix(axis.data(), derivative.data()) == SPIN8_OK,
        "independent output derivative");
  double derivative_norm = 0.;
  for (size_t i = 0; i < 8; ++i) {
    double expected = 0.;
    for (size_t j = 0; j < 8; ++j)
      expected += derivative[i * 8 + j] * q[j] * hidden;
    const double actual = (output_plus[i] - output_minus[i]) / (2. * h);
    check(near(actual, expected, 2e-10),
          "Q output gradient matches J q tanh(Win q)");
    derivative_norm = std::hypot(derivative_norm, actual);
  }
  check(derivative_norm > 1e-4, "nonzero Q output Wout gradient");
  plus = w;
  minus = w;
  plus.win[0] += h;
  minus.win[0] -= h;
  check(get_spinor(plus, q) == get_spinor(minus, q),
        "Win first-step gradient zero");
  plus = w;
  minus = w;
  plus.bout[1] += h;
  minus.bout[1] -= h;
  check(near((get_spinor(plus, q)[2] - get_spinor(minus, q)[2]) / (2. * h), 1.,
             1e-10),
        "bout first-step derivative");
  spin8_query_v3_controller *raw = nullptr;
  check(spin8_query_v3_controller_create(
            2, 8, w.win.data(), w.win.size() * 8, w.bin.data(), sizeof(w.bin),
            w.wout.data(), sizeof(w.wout), w.bout.data(), sizeof(w.bout),
            &raw) == SPIN8_INVALID_ARGUMENT &&
            !raw,
        "reject v2 version");
  for (size_t field = 0; field < 4; ++field) {
    check(spin8_query_v3_controller_create(
              3, 8, w.win.data(), field == 0 ? 0 : w.win.size() * 8,
              w.bin.data(), field == 1 ? 0 : sizeof(w.bin), w.wout.data(),
              field == 2 ? 0 : sizeof(w.wout), w.bout.data(),
              field == 3 ? 0 : sizeof(w.bout),
              &raw) == SPIN8_INVALID_ARGUMENT &&
              !raw,
          "weight capacities");
    auto bad = w;
    double *fields[] = {bad.win.data(), bad.bin.data(), bad.wout.data(),
                        bad.bout.data()};
    fields[field][0] = std::numeric_limits<double>::infinity();
    check(spin8_query_v3_controller_create(
              3, 8, bad.win.data(), bad.win.size() * 8, bad.bin.data(),
              sizeof(bad.bin), bad.wout.data(), sizeof(bad.wout),
              bad.bout.data(), sizeof(bad.bout), &raw) == SPIN8_NONFINITE &&
              !raw,
          "all weight groups finite");
  }
  check(spin8_query_v3_controller_create(
            3, SIZE_MAX, w.win.data(), SIZE_MAX, w.bin.data(), sizeof(w.bin),
            w.wout.data(), sizeof(w.wout), w.bout.data(), sizeof(w.bout),
            &raw) == SPIN8_INVALID_ARGUMENT,
        "extent overflow");
}
struct Gradients {
  std::vector<double> dq, dwin;
  V8 dbin{};
  std::array<double, 56> dwout{};
  A7 dbout{};
  explicit Gradients(size_t width) : dq(width), dwin(8 * width) {}
  spin8_query_v3_gradients_f64 buffers() {
    return {dq.data(),    dq.size() * sizeof(double),
            dwin.data(),  dwin.size() * sizeof(double),
            dbin.data(),  sizeof(dbin),
            dwout.data(), sizeof(dwout),
            dbout.data(), sizeof(dbout)};
  }
  void fill(double value) {
    std::fill(dq.begin(), dq.end(), value);
    std::fill(dwin.begin(), dwin.end(), value);
    dbin.fill(value);
    dwout.fill(value);
    dbout.fill(value);
  }
  bool same(const Gradients &other) const {
    return dq == other.dq && dwin == other.dwin && dbin == other.dbin &&
           dwout == other.dwout && dbout == other.dbout;
  }
};
double output_objective(const Weights &w, const std::vector<double> &q,
                        const std::vector<double> &dy) {
  const auto ctl = create(w);
  std::vector<double> z(q.size());
  check(spin8_query_v3_controller_apply(
            ctl.get(), SPIN8_F64, q.data(), q.size() * sizeof(double), q.size(),
            z.data(), z.size() * sizeof(double), z.size(), 1) == SPIN8_OK,
        "VJP finite difference forward");
  double loss = 0.;
  for (size_t i = 0; i < z.size(); ++i)
    loss += z[i] * dy[i];
  return loss;
}
Gradients analytic(const Weights &w, const std::vector<double> &q,
                   const std::vector<double> &dy) {
  const auto ctl = create(w);
  Gradients g(w.width);
  g.fill(std::numeric_limits<double>::quiet_NaN());
  const auto buffers = g.buffers();
  check(spin8_query_v3_controller_vjp_f64(
            ctl.get(), q.data(), q.size() * sizeof(double), dy.data(),
            dy.size() * sizeof(double), &buffers) == SPIN8_OK,
        "analytic VJP");
  return g;
}
template <class Objective> double five_point(Objective objective) {
  constexpr double step = 1e-4;
  return (objective(-2. * step) - 8. * objective(-step) + 8. * objective(step) -
          objective(2. * step)) /
         (12. * step);
}
void vjp_derivative_tests() {
  Weights w(9);
  w.bin = {.01, -.02, .03, -.04, .05, -.06, .07, -.08};
  w.bout = {.03, -.05, .08, -.02, .04, .01, -.07};
  for (size_t i = 0; i < w.wout.size(); ++i)
    w.wout[i] = .015 * (static_cast<double>(i % 9) - 4.);
  const std::vector<double> q{.3, -.1, .4, .2, -.5, .7, .8, -.6, .9};
  const std::vector<double> dy{-.2, .4, .1, -.6, .3, .5, -.7, .8, -.4};
  const auto g = analytic(w, q, dy);
  // Every scalar in all four parameter groups and Q is independently varied.
  for (size_t group = 0; group < 4; ++group) {
    const double *actual[] = {g.dwin.data(), g.dbin.data(), g.dwout.data(),
                              g.dbout.data()};
    const size_t counts[] = {w.win.size(), w.bin.size(), w.wout.size(),
                             w.bout.size()};
    for (size_t i = 0; i < counts[group]; ++i) {
      const double fd = five_point([&](double delta) {
        auto perturbed = w;
        double *parameters[] = {perturbed.win.data(), perturbed.bin.data(),
                                perturbed.wout.data(), perturbed.bout.data()};
        parameters[group][i] += delta;
        return output_objective(perturbed, q, dy);
      });
      check(near(actual[group][i], fd, 3e-10), "every parameter analytic VJP");
    }
  }
  for (size_t i = 0; i < q.size(); ++i) {
    const double fd = five_point([&](double delta) {
      auto perturbed = q;
      perturbed[i] += delta;
      return output_objective(w, perturbed, dy);
    });
    check(near(g.dq[i], fd, 3e-10), "every Q analytic VJP");
  }
  check(std::abs(g.dq.back() - dy.back()) > 1e-5,
        "copied tail still feeds controller gradient");
  w.wout.fill(0.);
  w.bout.fill(0.);
  const auto identity = analytic(w, q, dy);
  check(identity.dq == dy &&
            std::all_of(identity.dwin.begin(), identity.dwin.end(),
                        [](double value) { return value == 0.; }) &&
            identity.dbin == V8{},
        "identity dq and zero first-layer gradients");
  double output_gradient = 0.;
  for (double value : identity.dwout)
    output_gradient = std::hypot(output_gradient, value);
  check(output_gradient > .01, "identity retains nonzero Wout gradient");
  const double fd = five_point([&](double delta) {
    auto perturbed = w;
    perturbed.wout[0] += delta;
    return output_objective(perturbed, q, dy);
  });
  check(near(identity.dwout[0], fd, 3e-10), "identity Wout VJP derivative");

  Weights blocks(17);
  blocks.bin = w.bin;
  for (size_t i = 0; i < blocks.wout.size(); ++i)
    blocks.wout[i] = .02 * (static_cast<double>(i % 7) - 3.);
  std::vector<double> qb(17), db(17);
  for (size_t i = 0; i < qb.size(); ++i) {
    qb[i] = .04 * (static_cast<double>(i) - 6.);
    db[i] = .07 * (static_cast<double>(i % 5) - 2.);
  }
  const auto gb = analytic(blocks, qb, db);
  for (size_t i = 0; i < qb.size(); ++i) {
    const double derivative = five_point([&](double delta) {
      auto perturbed = qb;
      perturbed[i] += delta;
      return output_objective(blocks, perturbed, db);
    });
    check(near(gb.dq[i], derivative, 3e-10), "two blocks and tail VJP");
  }
  check(std::abs(gb.dq.back() - db.back()) > 1e-6,
        "multiple blocks contribute to tail controller derivative");
  Weights tail_only(3);
  tail_only.wout.fill(.03);
  const std::vector<double> qt{.1, .2, .3}, dt{.4, .5, .6};
  const auto gt = analytic(tail_only, qt, dt);
  check(gt.dq == dt && gt.dbin == V8{} && gt.dbout == A7{} &&
            std::all_of(gt.dwout.begin(), gt.dwout.end(),
                        [](double value) { return value == 0.; }),
        "tail-only row has no rotation parameter gradient");
}
void vjp_validation_tests() {
  Weights w(9);
  const auto ctl = create(w);
  std::vector<double> q{.3, -.1, .4, .2, -.5, .7, .8, -.6, .9};
  std::vector<double> dy{-.2, .4, .1, -.6, .3, .5, -.7, .8, -.4};
  Gradients g(w.width);
  g.fill(41.);
  const auto original = g;
  const auto good = g.buffers();
  const size_t bytes = q.size() * sizeof(double);
  auto rejected = [&](const spin8_query_v3_gradients_f64 *buffers,
                      const double *src, size_t sc, const double *adj,
                      size_t ac, spin8_status expected) {
    check(spin8_query_v3_controller_vjp_f64(ctl.get(), src, sc, adj, ac,
                                            buffers) == expected &&
              g.same(original),
          "VJP invalid input leaves every gradient unchanged");
  };
  for (size_t group = 0; group < 5; ++group) {
    for (size_t shortfall : {size_t{1}, SIZE_MAX}) {
      auto bad = good;
      size_t *capacities[] = {&bad.dq_bytes, &bad.dwin_bytes, &bad.dbin_bytes,
                              &bad.dwout_bytes, &bad.dbout_bytes};
      *capacities[group] = shortfall == SIZE_MAX ? 0 : *capacities[group] - 1;
      rejected(&bad, q.data(), bytes, dy.data(), bytes, SPIN8_INVALID_ARGUMENT);
    }
    for (size_t kind = 0; kind < 5; ++kind) {
      auto bad = good;
      double **outputs[] = {&bad.dq, &bad.dwin, &bad.dbin, &bad.dwout,
                            &bad.dbout};
      double *bad_addresses[] = {
          nullptr,
          reinterpret_cast<double *>(reinterpret_cast<char *>(g.dwin.data()) +
                                     1),
          q.data(), dy.data(), reinterpret_cast<double *>(&bad)};
      *outputs[group] = bad_addresses[kind];
      rejected(&bad, q.data(), bytes, dy.data(), bytes, SPIN8_INVALID_ARGUMENT);
    }
    for (size_t other = 0; other < group; ++other) {
      auto bad = good;
      double **outputs[] = {&bad.dq, &bad.dwin, &bad.dbin, &bad.dwout,
                            &bad.dbout};
      *outputs[group] = g.dwin.data();
      *outputs[other] = g.dwin.data();
      rejected(&bad, q.data(), bytes, dy.data(), bytes, SPIN8_INVALID_ARGUMENT);
    }
  }
  rejected(nullptr, q.data(), bytes, dy.data(), bytes, SPIN8_INVALID_ARGUMENT);
  rejected(reinterpret_cast<const spin8_query_v3_gradients_f64 *>(
               reinterpret_cast<const char *>(&good) + 1),
           q.data(), bytes, dy.data(), bytes, SPIN8_INVALID_ARGUMENT);
  for (size_t capacity : {size_t{0}, bytes - 1}) {
    rejected(&good, q.data(), capacity, dy.data(), bytes,
             SPIN8_INVALID_ARGUMENT);
    rejected(&good, q.data(), bytes, dy.data(), capacity,
             SPIN8_INVALID_ARGUMENT);
  }
  rejected(&good, nullptr, bytes, dy.data(), bytes, SPIN8_INVALID_ARGUMENT);
  rejected(&good, q.data(), bytes, nullptr, bytes, SPIN8_INVALID_ARGUMENT);
  const auto *misaligned = reinterpret_cast<const double *>(
      reinterpret_cast<const char *>(q.data()) + 1);
  rejected(&good, misaligned, bytes, dy.data(), bytes, SPIN8_INVALID_ARGUMENT);
  rejected(&good, q.data(), bytes, misaligned, bytes, SPIN8_INVALID_ARGUMENT);
  const auto *overflowing_address =
      reinterpret_cast<const double *>(UINTPTR_MAX - (alignof(double) - 1));
  rejected(&good, overflowing_address, SIZE_MAX, dy.data(), bytes,
           SPIN8_INVALID_ARGUMENT);
  auto bad = good;
  bad.dbout = const_cast<double *>(overflowing_address);
  rejected(&bad, q.data(), bytes, dy.data(), bytes, SPIN8_INVALID_ARGUMENT);
  bad = good;
  bad.dq = q.data() + 1;
  rejected(&bad, q.data(), bytes, dy.data(), bytes, SPIN8_INVALID_ARGUMENT);
  bad = good;
  bad.dwout = g.dwin.data() + 1;
  rejected(&bad, q.data(), bytes, dy.data(), bytes, SPIN8_INVALID_ARGUMENT);
  bad = good;
  bad.dq = reinterpret_cast<double *>(ctl.get());
  rejected(&bad, q.data(), bytes, dy.data(), bytes, SPIN8_INVALID_ARGUMENT);
  check(spin8_query_v3_controller_vjp_f64(nullptr, q.data(), bytes, dy.data(),
                                          bytes,
                                          &good) == SPIN8_INVALID_ARGUMENT &&
            g.same(original),
        "null controller failure atomic");
  for (double nonfinite : {std::numeric_limits<double>::quiet_NaN(),
                           std::numeric_limits<double>::infinity()}) {
    q.back() = nonfinite;
    rejected(&good, q.data(), bytes, dy.data(), bytes, SPIN8_NONFINITE);
    q.back() = .9;
    dy.back() = nonfinite;
    rejected(&good, q.data(), bytes, dy.data(), bytes, SPIN8_NONFINITE);
    dy.back() = -.4;
  }
  std::fill(q.begin(), q.end(), 2.);
  std::fill(dy.begin(), dy.end(), std::numeric_limits<double>::max());
  rejected(&good, q.data(), bytes, dy.data(), bytes, SPIN8_NONFINITE);

  // Apply can skip the hidden computation at identity, but the VJP cannot:
  // first-step Wout gradients depend on its actual hidden activation.
  Weights overflow(9);
  overflow.win[0] = std::numeric_limits<double>::max();
  const auto overflow_ctl = create(overflow);
  std::fill(dy.begin(), dy.end(), .1);
  check(spin8_query_v3_controller_vjp_f64(overflow_ctl.get(), q.data(), bytes,
                                          dy.data(), bytes,
                                          &good) == SPIN8_NONFINITE &&
            g.same(original),
        "identity hidden overflow failure atomic");
  // A late outer-product overflow must not publish earlier parameter groups.
  Weights outer(9);
  outer.wout[0] = 1.;
  outer.win.assign(outer.win.size(), 0.);
  const auto outer_ctl = create(outer);
  std::fill(q.begin(), q.end(), .2);
  q.back() = std::numeric_limits<double>::max();
  dy = {0., 20., -10., 30., -20., 10., -30., 40., 0.};
  check(spin8_query_v3_controller_vjp_f64(outer_ctl.get(), q.data(), bytes,
                                          dy.data(), bytes,
                                          &good) == SPIN8_NONFINITE &&
            g.same(original),
        "late parameter gradient overflow failure atomic");

  Weights huge(9);
  huge.bout.fill(std::numeric_limits<double>::max());
  q = {.3, -.1, .4, .2, -.5, .7, .8, -.6, .9};
  dy = {-.2, .4, .1, -.6, .3, .5, -.7, .8, -.4};
  const auto gh = analytic(huge, q, dy);
  check(std::all_of(gh.dbout.begin(), gh.dbout.end(),
                    [](double value) { return std::isfinite(value); }) &&
            std::any_of(gh.dbout.begin(), gh.dbout.end(),
                        [](double value) { return value != 0.; }),
        "scaled VJP for seven DBL_MAX tangents");
  std::fill(dy.begin(), dy.end(), 0.);
  const auto zero = analytic(w, q, dy);
  Gradients expected(w.width);
  check(zero.same(expected), "zero adjoint overwrites all gradients with zero");
}
} // namespace
int main() {
  basis_tests();
  chart_tests();
  controller_tests();
  gradient_and_validation_tests();
  vjp_derivative_tests();
  vjp_validation_tests();
  std::cout
      << "query v3: independent basis/polarization, 512 matrices, chart "
         "Jacobian, "
         "analytic VJP, controller gradients and buffer contracts passed\n";
}
