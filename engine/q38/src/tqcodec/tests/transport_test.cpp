#include "tqcodec/transport.h"
#include <algorithm>
#include <array>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <limits>
#include <vector>

static int errors = 0;
#define CHECK(p) do { if (!(p)) { std::fprintf(stderr, "line %d: %s\n", __LINE__, #p); ++errors; } } while (0)
constexpr size_t d = 9, m = 10;
using Vector = std::array<double, d>;
using Matrix = std::array<double, d*d>;

static Matrix identity() {
  Matrix a{};
  for (size_t i = 0; i < d; ++i) a[i*d+i] = 1;
  return a;
}
// Left-compose explicit proper plane rotations. No Spin8 dependency, and no
// claim that these deterministic fixtures are random Gaussian draws.
static void plane(Matrix &a, size_t i, size_t j, double c, double s) {
  for (size_t k = 0; k < d; ++k) {
    const double x = a[i*d+k], y = a[j*d+k];
    a[i*d+k] = c*x-s*y; a[j*d+k] = s*x+c*y;
  }
}
static Vector multiply(const Matrix &a, const Vector &v, bool transpose = false) {
  Vector out{};
  for (size_t i = 0; i < d; ++i) for (size_t j = 0; j < d; ++j)
    out[i] += a[transpose ? j*d+i : i*d+j]*v[j];
  return out;
}
static double dot(const Vector &a, const Vector &b) {
  double out = 0;
  for (size_t i = 0; i < d; ++i) out += a[i]*b[i];
  return out;
}
static void close(const Vector &a, const Vector &b, double tolerance = 2e-13) {
  for (size_t i = 0; i < d; ++i) CHECK(std::abs(a[i]-b[i]) < tolerance);
}
struct Physical {
  std::array<std::array<uint8_t, 3>, 4> packed{};
  std::array<double, 4> norm{};
  std::array<uint8_t, 2> signs{};
  double residual_norm = 0;
};
static Physical snapshot(const tq_cache *cache, size_t row, bool active) {
  Physical p{};
  for (unsigned r = 0; r < 4; ++r)
    CHECK(tq_cache_copy_row(cache, row, r, p.packed[r].data(), 3,
                            &p.norm[r]) == TQ_OK);
  if (active)
    CHECK(tq_cache_copy_residual_row(cache, row, p.signs.data(), 2,
                                    &p.residual_norm) == TQ_OK);
  return p;
}
static void same(const Physical &a, const Physical &b) {
  CHECK(a.packed == b.packed && a.signs == b.signs);
  CHECK(std::memcmp(a.norm.data(), b.norm.data(), sizeof(a.norm)) == 0);
  CHECK(std::memcmp(&a.residual_norm, &b.residual_norm, sizeof(double)) == 0);
}

static Vector oracle(const tq_cache *cache, const tq_cache_config_v2 &cfg,
    const Matrix (&u)[3], const Matrix &s, const Vector &query,
    const uint8_t *mask, double scale, bool active) {
  const size_t rows = tq_cache_rows(cache);
  std::vector<double> logits(rows);
  std::vector<Vector> values(rows);
  const Vector canonical = multiply(s, query, true);
  double maximum = -std::numeric_limits<double>::infinity();
  for (size_t row = 0; row < rows; ++row) {
    if (mask && !mask[row]) continue;
    const Physical p = snapshot(cache, row, active);
    double logical_score = 0, physical_score = 0;
    for (unsigned r = 0; r < 3; ++r) {
      Vector decoded{};
      CHECK(tq_decode_f64(p.packed[r].data(), 3, d, cfg.base.key_centroids,
                          2, p.norm[r], decoded.data()) == TQ_OK);
      const Vector common = multiply(u[r], decoded, true);
      logical_score += cfg.base.alpha[r]*dot(query, multiply(s, common));
      const Vector rotated_query = multiply(u[r], canonical);
      double part = 0;
      CHECK(tq_score_f64(rotated_query.data(), p.packed[r].data(), 3, d,
                         cfg.base.key_centroids, 2, p.norm[r], &part) == TQ_OK);
      physical_score += cfg.base.alpha[r]*part;
    }
    if (active && p.residual_norm != 0) {
      double signed_projection = 0;
      for (size_t j = 0; j < m; ++j) {
        // Independently form the transported projection row G[j]*S^T,
        // then dot it with logical q. Stored signs remain sign(G*e).
        double g_st_q = 0;
        for (size_t a = 0; a < d; ++a) {
          double coefficient = 0;
          for (size_t b = 0; b < d; ++b)
            coefficient += cfg.gaussian[j*d+b]*s[a*d+b];
          g_st_q += coefficient*query[a];
        }
        signed_projection += ((p.signs[j/8] >> (j%8)) & 1) ? g_st_q : -g_st_q;
      }
      logical_score += cfg.qjl_beta*std::sqrt(std::acos(-1.)/2)*p.residual_norm*
                       signed_projection/static_cast<double>(m);
      double correction = 0;
      CHECK(tq_qjl_score_f64(canonical.data(), d, cfg.gaussian, m,
          p.signs.data(), 2, p.residual_norm, cfg.qjl_beta, &correction) == TQ_OK);
      physical_score += correction;
    }
    // Equality is checked BEFORE scale and softmax, so normalization cannot
    // conceal a wrong query/transform order or a missing residual correction.
    CHECK(std::abs(logical_score-physical_score) < 2e-13);
    logits[row] = scale*logical_score;
    maximum = std::max(maximum, logits[row]);
    CHECK(tq_decode_f64(p.packed[3].data(), 3, d, cfg.base.value_centroids,
                        2, p.norm[3], values[row].data()) == TQ_OK);
  }
  Vector out{};
  double denominator = 0;
  for (size_t row = 0; row < rows; ++row) if (!mask || mask[row]) {
    const double weight = std::exp(logits[row]-maximum);
    denominator += weight;
    for (size_t i = 0; i < d; ++i) out[i] += weight*values[row][i];
  }
  for (double &v : out) v /= denominator;
  return out;
}

static void integration() {
  Matrix u[3] = {identity(), identity(), identity()};
  // Signed normalized H8 in the first eight dimensions, plus dimension nine.
  for (size_t i = 0; i < 8; ++i) for (size_t j = 0; j < 8; ++j) {
    size_t bits = i & j, parity = 0;
    while (bits) { parity ^= bits & 1; bits >>= 1; }
    u[0][i*d+j] = (parity ? -1. : 1.)/std::sqrt(8.);
  }
  plane(u[1], 0, 8, .6, .8); plane(u[1], 1, 3, .8, .6);
  plane(u[2], 2, 5, 5./13, 12./13); plane(u[2], 0, 7, .6, -.8);
  Matrix s = identity();
  plane(s, 0, 1, .8, .6); plane(s, 1, 4, 12./13, 5./13);
  plane(s, 2, 8, .6, .8);
  Vector query = {.8, -.35, .1, .2, -.4, .17, .3, -.2, .11};
  const Vector su = multiply(s, multiply(u[0], query));
  const Vector us = multiply(u[0], multiply(s, query));
  double noncommuting = 0;
  for (size_t i = 0; i < d; ++i) noncommuting += std::abs(su[i]-us[i]);
  CHECK(noncommuting > .1);
  const Matrix original_s = s;
  tq_transport_plan *plan = nullptr, *identity_plan = nullptr;
  CHECK(tq_transport_plan_create(d, s.data(), s.size(), sizeof(s), &plan) == TQ_OK);
  const Matrix id = identity();
  CHECK(tq_transport_plan_create(d, id.data(), id.size(), sizeof(id), &identity_plan) == TQ_OK);
  tq_transport_audit audit{};
  CHECK(tq_transport_plan_audit(plan, &audit) == TQ_OK);
  CHECK(audit.width == d && audit.determinant_sign == 1);
  CHECK(audit.gram_max_error < 1e-12 && audit.gram_frobenius_error < 1e-12);
  CHECK(std::abs(audit.log_abs_determinant) < 1e-12);
  // An accepted near-orthogonal matrix distinguishes normalized E_F from
  // the raw Frobenius norm by sqrt(d), beyond floating-point tolerances.
  Matrix near_orthogonal = id;
  for (size_t i = 0; i < d; ++i) near_orthogonal[i*d+i] += 2e-13;
  tq_transport_plan *near_plan = nullptr;
  CHECK(tq_transport_plan_create(d, near_orthogonal.data(), near_orthogonal.size(),
                                 sizeof(near_orthogonal), &near_plan) == TQ_OK);
  tq_transport_audit near_audit{};
  CHECK(tq_transport_plan_audit(near_plan, &near_audit) == TQ_OK);
  const double expected_ef = std::abs(near_orthogonal[0]*near_orthogonal[0]-1.);
  CHECK(std::abs(near_audit.gram_frobenius_error-expected_ef) < 1e-27);
  CHECK(near_audit.gram_frobenius_error < 1e-12);
  tq_transport_plan_destroy(near_plan);
  tq_transport_ledger plan_bytes{};
  CHECK(tq_transport_plan_bytes(plan, &plan_bytes) == TQ_OK);
  CHECK(plan_bytes.matrix_bytes >= sizeof(s) && plan_bytes.fixed_bytes > 0);
  CHECK(plan_bytes.total_bytes == plan_bytes.fixed_bytes+plan_bytes.matrix_bytes);
  std::fill(s.begin(), s.end(), 0); // Plan owns its copy.
  Vector canonical{}, inplace = query;
  CHECK(tq_transport_to_storage_f64(plan, query.data(), d, canonical.data(), d) == TQ_OK);
  close(canonical, multiply(original_s, query, true));
  CHECK(tq_transport_to_storage_f64(plan, inplace.data(), d, inplace.data(), d) == TQ_OK);
  CHECK(inplace == canonical);

  double book[] = {-1, 0, 1, 2};
  std::array<double, m*d> gaussian{};
  for (size_t j = 0; j < m; ++j) for (size_t i = 0; i < d; ++i)
    gaussian[j*d+i] = (static_cast<double>((j*11+i*7)%19)-9)/5;
  tq_cache_config_v2 cfg{};
  cfg.version = TQ_CACHE_VERSION_2;
  cfg.base.width = d; cfg.base.capacity_rows = 3;
  cfg.base.key_bits = cfg.base.value_bits = 2;
  cfg.base.key_centroids = cfg.base.value_centroids = book;
  cfg.base.alpha[0] = .2; cfg.base.alpha[1] = .3; cfg.base.alpha[2] = .5;
  for (unsigned r = 0; r < 3; ++r) cfg.base.transforms[r] = u[r].data();
  cfg.qjl_beta = .65; cfg.qjl_rows = m; cfg.gaussian = gaussian.data();
  cfg.gaussian_count = gaussian.size(); cfg.gaussian_byte_cap = sizeof(gaussian);
  const Vector keys[] = {{.7, .4, -.2, .15, .3, -.1, .11, .05, .3},
                         {-.3, .9, .17, -.11, .1, -.2, .4, .09, -.2}};
  Vector values[3]{};
  values[0][2] = 1; values[1][3] = 1; values[2][6] = 1;
  Vector disabled_output{};
  for (bool active : {false, true}) {
    cfg.qjl_beta = active ? .65 : 0;
    tq_cache *cache = nullptr;
    CHECK(tq_cache_create_v2(&cfg, &cache) == TQ_OK);
    for (size_t row = 0; row < 2; ++row)
      CHECK(tq_cache_append(cache, keys[row].data(), d, values[row].data(), d) == TQ_OK);
    const Physical before[] = {snapshot(cache, 0, active), snapshot(cache, 1, active)};
    if (active) CHECK(before[0].residual_norm > .01 && before[1].residual_norm > .01);
    tq_cache_ledger_v2 ledger_before{}, ledger_after{};
    CHECK(tq_cache_bytes_v2(cache, &ledger_before) == TQ_OK);
    Vector output{}, base{}, direct{};
    const double scale = .37;
    CHECK(tq_cache_attend_transported(cache, plan, query.data(), d, nullptr, 0,
                                      scale, output.data(), d) == TQ_OK);
    close(output, oracle(cache, cfg, u, original_s, query, nullptr, scale, active));
    if (active) CHECK(std::abs(output[2]-disabled_output[2]) > 1e-5);
    else disabled_output = output;
    CHECK(tq_cache_attend(cache, canonical.data(), d, nullptr, 0,
                          scale, direct.data(), d) == TQ_OK);
    CHECK(std::memcmp(output.data(), direct.data(), sizeof(output)) == 0);
    CHECK(tq_cache_attend(cache, query.data(), d, nullptr, 0,
                          scale, base.data(), d) == TQ_OK);
    CHECK(std::abs(base[2]-output[2]) > 1e-5);
    CHECK(tq_cache_attend_transported(cache, identity_plan, query.data(), d,
        nullptr, 0, scale, direct.data(), d) == TQ_OK);
    CHECK(std::memcmp(base.data(), direct.data(), sizeof(base)) == 0);
    inplace = query;
    CHECK(tq_cache_attend_transported(cache, plan, inplace.data(), d, nullptr,
        0, scale, inplace.data(), d) == TQ_OK);
    CHECK(inplace == output);
    uint8_t mask[] = {0, 1, 0};
    CHECK(tq_cache_attend_transported(cache, plan, query.data(), d, mask, 3,
                                      scale, output.data(), d) == TQ_OK);
    close(output, values[1]);
    for (size_t row = 0; row < 2; ++row) same(before[row], snapshot(cache, row, active));
    CHECK(tq_cache_bytes_v2(cache, &ledger_after) == TQ_OK);
    CHECK(ledger_before.total_bytes == ledger_after.total_bytes);
    CHECK(ledger_before.gaussian_bytes == ledger_after.gaussian_bytes);
    CHECK(ledger_before.residual_sign_bytes == ledger_after.residual_sign_bytes);
    CHECK(ledger_before.residual_norm_bytes == ledger_after.residual_norm_bytes);

    // A new key is supplied in logical coordinates, and canonicalized before
    // append. Check all physical bytes against independent transpose math.
    Vector logical_key = {.21, -.51, .31, .15, -.1, .12, .3, -.17, .08};
    Vector new_key{};
    CHECK(tq_transport_to_storage_f64(plan, logical_key.data(), d, new_key.data(), d) == TQ_OK);
    const Vector reference_key = multiply(original_s, logical_key, true);
    tq_cache *reference = nullptr;
    CHECK(tq_cache_clone(cache, &reference) == TQ_OK);
    CHECK(tq_cache_append(cache, new_key.data(), d, values[2].data(), d) == TQ_OK);
    CHECK(tq_cache_append(reference, reference_key.data(), d, values[2].data(), d) == TQ_OK);
    same(snapshot(cache, 2, active), snapshot(reference, 2, active));
    mask[0] = mask[2] = 1; mask[1] = 0;
    CHECK(tq_cache_attend_transported(cache, plan, query.data(), d, mask, 3,
                                      scale, output.data(), d) == TQ_OK);
    close(output, oracle(cache, cfg, u, original_s, query, mask, scale, active));
    for (size_t row = 0; row < 2; ++row) same(before[row], snapshot(cache, row, active));
    output.fill(17);
    mask[0] = mask[2] = 0;
    CHECK(tq_cache_attend_transported(cache, plan, query.data(), d, mask, 3,
                                      scale, output.data(), d) == TQ_INVALID);
    for (double v : output) CHECK(v == 17);
    CHECK(tq_cache_attend_transported(cache, plan, query.data(), d, nullptr, 0,
                                      scale, output.data(), d-1) == TQ_CAPACITY);
    CHECK(tq_cache_attend_transported(cache, plan, query.data(), d-1, nullptr, 0,
                                      scale, output.data(), d) == TQ_INVALID);
    CHECK(tq_cache_attend_transported(cache, plan, query.data(), d, mask, 2,
                                      scale, output.data(), d) == TQ_INVALID);
    CHECK(tq_cache_attend_transported(cache, plan, query.data(), d, nullptr, 0,
                                      0, output.data(), d) == TQ_INVALID);
    std::array<double, 64> small_identity{};
    for (size_t i = 0; i < 8; ++i) small_identity[i*8+i] = 1;
    tq_transport_plan *small_plan = nullptr;
    CHECK(tq_transport_plan_create(8, small_identity.data(), small_identity.size(),
        sizeof(small_identity), &small_plan) == TQ_OK);
    CHECK(tq_cache_attend_transported(cache, small_plan, query.data(), d,
        nullptr, 0, scale, output.data(), d) == TQ_INVALID);
    tq_transport_plan_destroy(small_plan);
    Vector bad_query = query;
    bad_query[0] = std::numeric_limits<double>::quiet_NaN();
    CHECK(tq_cache_attend_transported(cache, plan, bad_query.data(), d, nullptr, 0,
                                      scale, output.data(), d) == TQ_NONFINITE);
    std::array<double, d+1> overlap{};
    CHECK(tq_cache_attend_transported(cache, plan, overlap.data(), d, nullptr,
        0, scale, overlap.data()+1, d) == TQ_INVALID);
    CHECK(tq_cache_attend_transported(cache, plan, query.data(), d,
        reinterpret_cast<const uint8_t *>(output.data()), sizeof(output),
        scale, output.data(), d) == TQ_INVALID);
    for (double v : output) CHECK(v == 17);
    tq_cache_destroy(reference); tq_cache_destroy(cache);
  }
  tq_transport_plan_destroy(identity_plan); tq_transport_plan_destroy(plan);
}

static void invalid_inputs() {
  Matrix a = identity();
  tq_transport_plan *sentinel = reinterpret_cast<tq_transport_plan *>(uintptr_t{1});
  CHECK(tq_transport_plan_create(d, a.data(), a.size()-1, sizeof(a), &sentinel) == TQ_INVALID);
  CHECK(tq_transport_plan_create(d, a.data(), a.size(), sizeof(a)-1, &sentinel) == TQ_CAPACITY);
  CHECK(tq_transport_plan_create(7, a.data(), a.size(), sizeof(a), &sentinel) == TQ_INVALID);
  CHECK(tq_transport_plan_create(SIZE_MAX, a.data(), SIZE_MAX, SIZE_MAX, &sentinel) == TQ_INVALID);
  CHECK(tq_transport_plan_create(d, nullptr, a.size(), sizeof(a), &sentinel) == TQ_INVALID);
  CHECK(tq_transport_plan_create(d, a.data(), a.size(), sizeof(a), nullptr) == TQ_INVALID);
  CHECK(tq_transport_plan_create(d, a.data(), a.size(), sizeof(a),
      reinterpret_cast<tq_transport_plan **>(a.data())) == TQ_INVALID);
  CHECK(a == identity());
  a[0] = -1; // Reflection is valid for old cache U, but forbidden for S.
  CHECK(tq_transport_plan_create(d, a.data(), a.size(), sizeof(a), &sentinel) == TQ_INVALID);
  a[0] = 1.001;
  CHECK(tq_transport_plan_create(d, a.data(), a.size(), sizeof(a), &sentinel) == TQ_INVALID);
  a[0] = std::numeric_limits<double>::quiet_NaN();
  CHECK(tq_transport_plan_create(d, a.data(), a.size(), sizeof(a), &sentinel) == TQ_NONFINITE);
  a[0] = std::numeric_limits<double>::infinity();
  CHECK(tq_transport_plan_create(d, a.data(), a.size(), sizeof(a), &sentinel) == TQ_NONFINITE);
  CHECK(sentinel == reinterpret_cast<tq_transport_plan *>(uintptr_t{1}));
  alignas(double) unsigned char misaligned[sizeof(Matrix)+1]{};
  CHECK(tq_transport_plan_create(d, reinterpret_cast<double *>(misaligned+1),
      a.size(), sizeof(a), &sentinel) == TQ_INVALID);
  a = identity(); plane(a, 0, 1, .8, .6);
  tq_transport_plan *plan = nullptr;
  CHECK(tq_transport_plan_create(d, a.data(), a.size(), sizeof(a), &plan) == TQ_OK);
  Vector src{}, dst{}; dst.fill(17);
  CHECK(tq_transport_to_storage_f64(plan, src.data(), d-1, dst.data(), d) == TQ_INVALID);
  CHECK(tq_transport_to_storage_f64(plan, src.data(), d, dst.data(), d-1) == TQ_CAPACITY);
  src[0] = std::numeric_limits<double>::infinity();
  CHECK(tq_transport_to_storage_f64(plan, src.data(), d, dst.data(), d) == TQ_NONFINITE);
  src.fill(std::numeric_limits<double>::max());
  CHECK(tq_transport_to_storage_f64(plan, src.data(), d, dst.data(), d) == TQ_NONFINITE);
  for (double v : dst) CHECK(v == 17);
  std::array<double, d+1> overlap{};
  CHECK(tq_transport_to_storage_f64(plan, overlap.data(), d,
      overlap.data()+1, d) == TQ_INVALID);
  CHECK(tq_transport_to_storage_f64(plan, overlap.data()+1, d,
      overlap.data(), d) == TQ_INVALID);
  CHECK(tq_transport_to_storage_f64(plan, reinterpret_cast<double *>(misaligned+1),
      d, dst.data(), d) == TQ_INVALID);
  CHECK(tq_transport_to_storage_f64(plan, dst.data(), d,
      reinterpret_cast<double *>(misaligned+1), d) == TQ_INVALID);
  CHECK(tq_transport_to_storage_f64(nullptr, dst.data(), d, dst.data(), d) == TQ_INVALID);
  CHECK(tq_transport_to_storage_f64(plan, dst.data(), d,
      reinterpret_cast<double *>(plan), d) == TQ_INVALID);
  tq_transport_audit audit{}; audit.width = 123;
  tq_transport_ledger ledger{}; ledger.total_bytes = 456;
  CHECK(tq_transport_plan_audit(nullptr, &audit) == TQ_INVALID && audit.width == 123);
  CHECK(tq_transport_plan_bytes(nullptr, &ledger) == TQ_INVALID && ledger.total_bytes == 456);
  CHECK(tq_transport_plan_audit(plan, reinterpret_cast<tq_transport_audit *>(plan)) == TQ_INVALID);
  CHECK(tq_transport_plan_bytes(plan, reinterpret_cast<tq_transport_ledger *>(plan)) == TQ_INVALID);
  tq_transport_plan_destroy(plan);
  tq_transport_plan_destroy(nullptr);

  // sign(G*e) cannot determine sign(G*S*e). The old signs and norms agree
  // for these residuals but a proper quarter-turn produces opposite signs.
  const double z = 1/std::sqrt(2.);
  const double e_plus[] = {z, z}, e_minus[] = {z, -z};
  CHECK(e_plus[0] >= 0 && e_minus[0] >= 0);
  CHECK(-e_plus[1] < 0 && -e_minus[1] > 0);
  CHECK(e_plus[0]*e_plus[0]+e_plus[1]*e_plus[1] ==
        e_minus[0]*e_minus[0]+e_minus[1]*e_minus[1]);
}

int main() {
  integration(); invalid_inputs();
  std::printf("tqcodec uniform transport reference failures: %d\n", errors);
  return errors ? 1 : 0;
}
