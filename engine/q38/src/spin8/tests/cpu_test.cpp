#include "spin8/spin8.h"
#include <algorithm>
#include <array>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <limits>
#include <random>
#include <vector>

static void check(bool value, const char *what) {
  if (!value) {
    std::fprintf(stderr, "FAIL: %s\n", what);
    std::exit(1);
  }
}
template <size_t N> using Matrix = std::array<double, N * N>;
template <size_t N>
static Matrix<N> multiply(const Matrix<N> &a, const Matrix<N> &b) {
  Matrix<N> c{};
  for (size_t i = 0; i < N; ++i)
    for (size_t j = 0; j < N; ++j) {
      long double s = 0;
      for (size_t k = 0; k < N; ++k)
        s += static_cast<long double>(a[i * N + k]) * b[k * N + j];
      c[i * N + j] = static_cast<double>(s);
    }
  return c;
}
template <size_t N> static Matrix<N> trans(const Matrix<N> &a) {
  Matrix<N> b{};
  for (size_t i = 0; i < N; ++i)
    for (size_t j = 0; j < N; ++j)
      b[i * N + j] = a[j * N + i];
  return b;
}
template <size_t N>
static double difference(const Matrix<N> &a, const Matrix<N> &b) {
  double e = 0;
  for (size_t i = 0; i < N * N; ++i)
    e = std::max(e, std::abs(a[i] - b[i]));
  return e;
}
template <size_t N> static Matrix<N> identity() {
  Matrix<N> a{};
  for (size_t i = 0; i < N; ++i)
    a[i * N + i] = 1;
  return a;
}
static Matrix<8> matrix(spin8_plan *p, int view) {
  Matrix<8> m{};
  check(spin8_plan_matrix(p, static_cast<spin8_view>(view), m.data()) ==
            SPIN8_OK,
        "extract matrix");
  return m;
}
static spin8_plan *make(const std::vector<spin8_plane> &f) {
  spin8_plan *p = nullptr;
  check(spin8_plan_create(f.data(), f.size(), &p) == SPIN8_OK,
        "create audited plan");
  return p;
}

int main() {
  std::array<Matrix<16>, 8> g{};
  std::array<double, 2048> flat_gamma{};
  check(spin8_clifford_generators(flat_gamma.data()) == SPIN8_OK, "generators");
  for (size_t a = 0; a < 8; ++a)
    std::copy_n(flat_gamma.data() + a * 256, 256, g[a].begin());
  for (size_t a = 0; a < 8; ++a)
    for (size_t b = 0; b < 8; ++b) {
      auto ab = multiply<16>(g[a], g[b]), ba = multiply<16>(g[b], g[a]);
      for (size_t i = 0; i < 256; ++i)
        check(std::abs(ab[i] + ba[i] -
                       ((a == b && i / 16 == i % 16) ? 2. : 0.)) < 1e-14,
              "Clifford anticommutator");
      check(difference<16>(trans<16>(g[a]), g[a]) == 0, "symmetric real gamma");
    }
  auto chir = identity<16>();
  for (const auto &a : g)
    chir = multiply<16>(chir, a);
  for (size_t i = 0; i < 256; ++i)
    check(chir[i] == ((i / 16 == i % 16) ? (i / 16 < 8 ? 1. : -1.) : 0.),
          "8+8 chirality");
  std::mt19937_64 rng(0x5350494e38);
  std::uniform_real_distribution<double> angle(-2, 2);
  double worst = 0;
  for (int trial = 0; trial < 80; ++trial) {
    std::vector<spin8_plane> f;
    for (int j = 0; j < 12; ++j) {
      unsigned a = static_cast<unsigned>(rng() % 8),
               b = static_cast<unsigned>(rng() % 7);
      if (b >= a)
        ++b;
      f.push_back({a, b, angle(rng)});
    }
    auto *p = make(f);
    spin8_audit_report report{};
    check(spin8_plan_audit(p, &report) == SPIN8_OK, "plan audit");
    worst = std::max(worst, report.triality_max_error);
    Matrix<16> u{};
    auto plus = matrix(p, 1), minus = matrix(p, 2), r = matrix(p, 0);
    for (size_t i = 0; i < 8; ++i)
      for (size_t j = 0; j < 8; ++j) {
        u[i * 16 + j] = plus[i * 8 + j];
        u[(i + 8) * 16 + j + 8] = minus[i * 8 + j];
      }
    // Independent arbitrary-vector invariant, not merely a call to the library
    // audit.
    Matrix<16> gv{}, grv{};
    double v[8], rv[8]{};
    for (double &x : v)
      x = angle(rng);
    for (size_t i = 0; i < 8; ++i)
      for (size_t j = 0; j < 8; ++j)
        rv[i] += r[i * 8 + j] * v[j];
    for (size_t i = 0; i < 256; ++i)
      for (size_t a = 0; a < 8; ++a) {
        gv[i] += v[a] * g[a][i];
        grv[i] += rv[a] * g[a][i];
      }
    check(difference<16>(multiply<16>(multiply<16>(u, gv), trans<16>(u)), grv) <
              1e-12,
          "triality intertwiner");
    auto first = make({f.begin(), f.begin() + 6}),
         second = make({f.begin() + 6, f.end()});
    for (int view = 0; view < 3; ++view)
      check(
          difference<8>(multiply<8>(matrix(second, view), matrix(first, view)),
                        matrix(p, view)) < 1e-12,
          "group composition");
    spin8_plan_free(first);
    spin8_plan_free(second);
    spin8_plan_free(p);
  }
  constexpr double pi = 3.1415926535897932384626433832795;
  auto *p2 = make({{0, 1, 2 * pi}}), *p4 = make({{0, 1, 4 * pi}});
  for (int v = 0; v < 3; ++v) {
    auto expected = identity<8>();
    if (v)
      for (double &x : expected)
        x = -x;
    check(difference<8>(matrix(p2, v), expected) < 1e-12, "2pi double cover");
    check(difference<8>(matrix(p4, v), identity<8>()) < 1e-12, "4pi identity");
  }
  spin8_plan_free(p2);
  spin8_plan_free(p4);
  auto *p = make({{0, 2, 0.73}, {4, 7, -0.38}, {1, 6, 1.23}});
  std::vector<double> x(3 * 24, -99), y(3 * 25, -88), z(3 * 26, -77);
  for (size_t i = 0; i < 3; ++i)
    for (size_t j = 0; j < 19; ++j)
      x[i * 24 + j] = std::sin(static_cast<double>(i * 19 + j));
  for (int view = 0; view < 3; ++view) {
    check(spin8_transform(p, static_cast<spin8_view>(view), SPIN8_F64, x.data(),
                          24, y.data(), 25, 3, 19, 0) == SPIN8_OK,
          "strided forward");
    check(spin8_transform(p, static_cast<spin8_view>(view), SPIN8_F64, y.data(),
                          25, z.data(), 26, 3, 19, 1) == SPIN8_OK,
          "strided inverse");
    for (size_t i = 0; i < 3; ++i)
      for (size_t j = 0; j < 19; ++j)
        check(std::abs(z[i * 26 + j] - x[i * 24 + j]) < 1e-12,
              "inverse reference");
    check(y[19] == -88, "padding untouched");
    check(y[18] == x[18], "tail identity");
  }
  for (unsigned mask = 0; mask < 256; ++mask) {
    Matrix<8> h{}, hi{};
    check(spin8_hadamard_matrix(static_cast<uint8_t>(mask), 0, h.data()) ==
              SPIN8_OK,
          "HD matrix");
    spin8_hadamard_matrix(static_cast<uint8_t>(mask), 1, hi.data());
    check(difference<8>(multiply<8>(hi, h), identity<8>()) < 1e-12,
          "DH HD inverse");
    spin8_matrix_audit audit{};
    spin8_audit_matrix(h.data(), &audit);
    unsigned bits = mask, parity = 0;
    while (bits) {
      parity ^= 1;
      bits &= bits - 1;
    }
    check(audit.det_sign == (parity ? -1 : 1), "Hadamard determinant sign");
    check(audit.ef_normalized < 1e-12, "Hadamard orthogonality");
  }
  check(spin8_hadamard(SPIN8_F64, x.data(), 24, y.data(), 25, 3, 19, 23, 0) ==
            SPIN8_OK,
        "HD strided");
  check(spin8_hadamard(SPIN8_F64, y.data(), 25, y.data(), 25, 3, 19, 23, 1) ==
            SPIN8_OK,
        "HD in-place inverse");
  for (size_t i = 0; i < 3; ++i)
    for (size_t j = 0; j < 19; ++j)
      check(std::abs(y[i * 25 + j] - x[i * 24 + j]) < 1e-12, "HD recovery");
  // General Sylvester block: independent dense parity oracle and signed DH
  // inverse.
  for (size_t block :
       {size_t(1), size_t(2), size_t(8), size_t(128), size_t(1024)}) {
    std::vector<int8_t> signs(block, 1);
    signs[0] = -1;
    std::vector<double> input(block + 3), output(block + 3), inverse(block + 3);
    for (double &t : input)
      t = angle(rng);
    check(spin8_hadamard_block(SPIN8_F64, input.data(), input.size(),
                               output.data(), output.size(), 1, input.size(),
                               block, signs.data(), 0) == SPIN8_OK,
          "general HD");
    for (size_t i = 0; i < block; ++i) {
      long double sum = 0;
      for (size_t j = 0; j < block; ++j) {
        size_t bits = i & j, parity = 0;
        while (bits) {
          parity ^= 1;
          bits &= bits - 1;
        }
        sum +=
            static_cast<long double>(input[j]) * signs[j] * (parity ? -1 : 1);
      }
      check(std::abs(output[i] -
                     static_cast<double>(
                         sum / std::sqrt(static_cast<long double>(block)))) <
                1e-12,
            "general HD dense oracle");
    }
    check(spin8_hadamard_block(SPIN8_F64, output.data(), output.size(),
                               inverse.data(), inverse.size(), 1, input.size(),
                               block, signs.data(), 1) == SPIN8_OK,
          "general DH");
    for (size_t i = 0; i < input.size(); ++i)
      check(std::abs(inverse[i] - input[i]) < 1e-12, "general HD DH recovery");
    spin8_matrix_audit ha{};
    check(spin8_hadamard_block_audit(SPIN8_F64, block, signs.data(), &ha) ==
              SPIN8_OK,
          "F64 factor audit");
    check(ha.det_sign == (block == 2 ? 1 : -1) && ha.ef_normalized < 1e-12,
          "general factor sign");
    check(spin8_hadamard_block_audit(SPIN8_F32, block, signs.data(), &ha) ==
                  SPIN8_OK &&
              ha.ef_normalized < 1e-5,
          "F32 coefficient factor audit");
    if (block == 128)
      check(ha.log_abs_det != 0,
            "F32 rounded coefficient determinant is not exact");
    // block=1 divides every scalar; there is no incomplete tail in that case.
    if (block > 3)
      check(output[block + 2] == input[block + 2], "general tail identity");
  }
  check(spin8_hadamard_block(SPIN8_F64, x.data(), 24, y.data(), 25, 1, 19, 3,
                             nullptr, 0) == SPIN8_INVALID_ARGUMENT,
        "reject non-power2 block");
  int8_t bad_sign[8] = {1, 1, 1, 1, 0, 1, 1, 1};
  check(spin8_hadamard_block(SPIN8_F64, x.data(), 24, y.data(), 25, 1, 19, 8,
                             bad_sign, 0) == SPIN8_INVALID_ARGUMENT,
        "reject invalid sign");
  // Arbitrary head widths including tails; direct original-space oracle.
  for (size_t d : {size_t(1), size_t(7), size_t(8), size_t(13), size_t(40),
                   size_t(64), size_t(96), size_t(128)}) {
    const size_t nq = 3, nk = 5, dv = 9;
    std::vector<double> q(nq * d), k(nk * d), v(nk * dv), c(nq * nk),
        out(nq * dv);
    for (double &t : q)
      t = angle(rng);
    for (double &t : k)
      t = angle(rng);
    for (double &t : v)
      t = angle(rng);
    std::vector<uint8_t> mask(nq * nk, 1);
    mask[1] = 0;
    double weights[3] = {0.1, 0.2, 0.7};
    check(spin8_matmul_f64(p, q.data(), nq, k.data(), nk, d, weights,
                           c.data()) == SPIN8_OK,
          "matmul");
    for (size_t i = 0; i < nq; ++i)
      for (size_t j = 0; j < nk; ++j) {
        long double sum = 0;
        for (size_t t = 0; t < d; ++t)
          sum += static_cast<long double>(q[i * d + t]) * k[j * d + t];
        check(std::abs(c[i * nk + j] - static_cast<double>(sum)) < 1e-12,
              "matmul oracle");
      }
    check(spin8_attention_f64(p, q.data(), nq, k.data(), nk, v.data(), dv, d,
                              mask.data(), weights, out.data()) == SPIN8_OK,
          "attention");
    for (size_t i = 0; i < nq; ++i) {
      std::vector<long double> prob(nk);
      long double mx = -1e30L;
      for (size_t j = 0; j < nk; ++j) {
        for (size_t t = 0; t < d; ++t)
          prob[j] += static_cast<long double>(q[i * d + t]) * k[j * d + t];
        prob[j] /= std::sqrt(static_cast<long double>(d));
        if (mask[i * nk + j])
          mx = std::max(mx, prob[j]);
      }
      long double sum = 0;
      for (size_t j = 0; j < nk; ++j) {
        prob[j] = mask[i * nk + j] ? std::exp(prob[j] - mx) : 0;
        sum += prob[j];
      }
      for (size_t t = 0; t < dv; ++t) {
        long double value = 0;
        for (size_t j = 0; j < nk; ++j)
          value += prob[j] / sum * v[j * dv + t];
        check(std::abs(out[i * dv + t] - static_cast<double>(value)) < 1e-12,
              "one-softmax attention oracle");
      }
    }
    std::vector<float> qf(q.size()), kf(k.size()), vf(v.size()), of(nq * dv);
    std::transform(q.begin(), q.end(), qf.begin(),
                   [](double t) { return static_cast<float>(t); });
    std::transform(k.begin(), k.end(), kf.begin(),
                   [](double t) { return static_cast<float>(t); });
    std::transform(v.begin(), v.end(), vf.begin(),
                   [](double t) { return static_cast<float>(t); });
    check(spin8_attention_f32(p, qf.data(), nq, kf.data(), nk, vf.data(), dv, d,
                              mask.data(), nullptr, of.data()) == SPIN8_OK,
          "F32 attention");
    for (size_t i = 0; i < of.size(); ++i)
      check(std::abs(of[i] - out[i]) < 1e-5, "F32 tolerance");
  }
  spin8_plan *bad = reinterpret_cast<spin8_plan *>(1);
  spin8_plane invalid{0, 0, 1};
  check(spin8_plan_create(&invalid, 1, &bad) == SPIN8_INVALID_ARGUMENT && !bad,
        "bad plane resets output");
  invalid = {0, 1, std::numeric_limits<double>::infinity()};
  check(spin8_plan_create(&invalid, 1, &bad) == SPIN8_NONFINITE,
        "nonfinite config");
  y.assign(75, -88);
  x[18] = std::numeric_limits<double>::quiet_NaN();
  check(spin8_transform(p, SPIN8_VECTOR, SPIN8_F64, x.data(), 24, y.data(), 25,
                        3, 19, 0) == SPIN8_NONFINITE &&
            y[0] == -88,
        "transactional nonfinite rejection");
  x[18] = 1;
  check(spin8_transform(p, SPIN8_VECTOR, SPIN8_F64, x.data(), 18, y.data(), 25,
                        3, 19, 0) == SPIN8_INVALID_ARGUMENT,
        "short stride");
  check(spin8_transform(p, SPIN8_VECTOR, SPIN8_F64, x.data(), 24, x.data() + 1,
                        24, 2, 19, 0) == SPIN8_INVALID_ARGUMENT,
        "partial overlap");
  check(spin8_transform(p, SPIN8_VECTOR, SPIN8_F64, x.data(), SIZE_MAX,
                        y.data(), 25, 3, 19, 0) == SPIN8_INVALID_ARGUMENT,
        "overflow stride");
  check(spin8_transform(p, SPIN8_VECTOR, SPIN8_F16, x.data(), 24, y.data(), 25,
                        3, 19, 0) == SPIN8_UNSUPPORTED,
        "F16 truthful unsupported");
  double a[8] = {}, b[8] = {}, o[1] = {77}, weights[3] = {-1, 1, 1};
  double red_a[3] = {1, 2, 3}, red_b[3] = {4, 5, 6}, red_c[3] = {7, 8, 9},
         reduced[3] = {};
  const void *views[3] = {red_a, red_b, red_c};
  check(spin8_reduce(SPIN8_F64, views, 3, reduced, 3, 1, 3, nullptr) ==
            SPIN8_OK,
        "default reduction");
  check(std::abs(reduced[0] - 4) < 1e-12 && std::abs(reduced[2] - 6) < 1e-12,
        "reduction oracle");
  double custom_weights[3] = {0.25, 0.5, 0.25};
  check(spin8_reduce(SPIN8_F64, views, 3, red_a, 3, 1, 3, custom_weights) ==
                SPIN8_OK &&
            red_a[1] == 5,
        "in-place weighted reduction");
  custom_weights[1] = std::numeric_limits<double>::quiet_NaN();
  check(spin8_reduce(SPIN8_F64, views, 3, reduced, 3, 1, 3, custom_weights) ==
            SPIN8_NONFINITE,
        "nonfinite reduction weights");
  float fwd[19], recovered[19];
  for (size_t i = 0; i < 19; ++i)
    fwd[i] = static_cast<float>(std::sin(static_cast<double>(i)));
  std::array<float, 19> original{};
  std::copy_n(fwd, 19, original.begin());
  check(spin8_transform(p, SPIN8_SPIN_PLUS, SPIN8_F32, fwd, 19, fwd, 19, 1, 19,
                        0) == SPIN8_OK,
        "F32 in-place forward");
  check(spin8_transform(p, SPIN8_SPIN_PLUS, SPIN8_F32, fwd, 19, recovered, 19,
                        1, 19, 1) == SPIN8_OK,
        "F32 inverse");
  for (size_t i = 0; i < 19; ++i)
    check(std::abs(recovered[i] - original[i]) < 1e-5, "F32 inverse tolerance");
  auto huge = identity<8>();
  huge[0] = std::numeric_limits<double>::max();
  spin8_matrix_audit huge_audit{};
  check(spin8_audit_matrix(huge.data(), &huge_audit) == SPIN8_NONFINITE,
        "finite audit overflow");
  check(spin8_matmul_f64(p, a, 1, b, 1, 8, weights, o) ==
                SPIN8_INVALID_ARGUMENT &&
            o[0] == 77,
        "negative weights");
  weights[0] = 0;
  check(spin8_matmul_f64(p, a, 1, b, 1, 8, weights, o) ==
            SPIN8_INVALID_ARGUMENT,
        "weight sum");
  uint8_t empty = 0;
  check(spin8_attention_f64(p, a, 1, b, 1, a, 8, 8, &empty, nullptr, a) ==
            SPIN8_INVALID_ARGUMENT,
        "aliased attention");
  double ao[8];
  check(spin8_attention_f64(p, a, 1, b, 1, a, 8, 8, &empty, nullptr, ao) ==
            SPIN8_INVALID_ARGUMENT,
        "all masked row");
  spin8_monitor monitor{};
  spin8_monitor_init(&monitor);
  int due = 0;
  check(spin8_monitor_due(&monitor, 0, 0, &due) == SPIN8_OK && due,
        "initial audit due");
  check(spin8_monitor_record(&monitor, 0, 0, SPIN8_OK) == SPIN8_OK,
        "record audit");
  spin8_monitor_due(&monitor, 1023, 59, &due);
  check(!due, "schedule before threshold");
  spin8_monitor_due(&monitor, 1024, 59, &due);
  check(due, "token schedule");
  spin8_monitor_due(&monitor, 2, 60, &due);
  check(due, "time schedule");
  check(spin8_monitor_record(&monitor, 2, 60, SPIN8_AUDIT_FAILED) ==
            SPIN8_AUDIT_FAILED,
        "typed audit failure");
  check(spin8_monitor_due(&monitor, 3, 61, &due) == SPIN8_INVALIDATED,
        "sticky invalidation");
  check(spin8_monitor_record(&monitor, 3, 61, SPIN8_OK) == SPIN8_INVALIDATED,
        "no silent revalidation");
  spin8_plan_free(p);
  std::printf(
      "PASS Spin(8) CPU: Clifford, chirality, 80 group trials, double cover, "
      "HD/DH, attention, boundaries, monitoring; worst triality %.17g\n",
      worst);
  return 0;
}
