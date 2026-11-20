#include "tqcodec/cache.h"
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <limits>
#include <vector>

static int errors = 0;
#define CHECK(p) do { if (!(p)) { std::fprintf(stderr, "line %d: %s\n", __LINE__, #p); ++errors; } } while (0)

static void qjl_tests() {
  constexpr size_t d = 9, m = 10;
  std::vector<double> rotations[3], gaussian(m*d);
  for (auto &r : rotations) r.assign(d*d, 0);
  for (size_t j = 0; j < d; ++j) {
    rotations[0][j*d+j] = 1;
    rotations[1][j*d+(j+1)%d] = 1;
    rotations[2][j*d+j] = j == 0 ? -1 : 1;
  }
  // Deliberately explicit finite fixture; not a claimed independent Gaussian draw.
  for (size_t j = 0; j < m; ++j) for (size_t i = 0; i < d; ++i)
    gaussian[j*d+i] = (static_cast<double>((j*11+i*7)%19)-9)/5;
  double book[] = {-1, 0, 1, 2};
  tq_cache_config_v2 cfg{};
  cfg.version = TQ_CACHE_VERSION_2;
  cfg.base.width = d; cfg.base.capacity_rows = 3;
  cfg.base.key_bits = cfg.base.value_bits = 2;
  cfg.base.key_centroids = cfg.base.value_centroids = book;
  cfg.base.alpha[0] = .2; cfg.base.alpha[1] = .3; cfg.base.alpha[2] = .5;
  for (unsigned r = 0; r < 3; ++r) cfg.base.transforms[r] = rotations[r].data();
  cfg.qjl_beta = .65; cfg.qjl_rows = m; cfg.gaussian = gaussian.data();
  cfg.gaussian_count = gaussian.size(); cfg.gaussian_byte_cap = gaussian.size()*sizeof(double);
  tq_cache *active = nullptr, *base = nullptr, *disabled = nullptr;
  CHECK(tq_cache_create_v2(&cfg, &active) == TQ_OK);
  CHECK(tq_cache_create(&cfg.base, &base) == TQ_OK);
  auto off = cfg; off.qjl_beta = 0; off.gaussian = nullptr;
  off.qjl_rows = SIZE_MAX; off.gaussian_count = off.gaussian_byte_cap = 0;
  CHECK(tq_cache_create_v2(&off, &disabled) == TQ_OK);
  tq_cache_ledger_v2 lb{}, la{}, ld{};
  CHECK(tq_cache_bytes_v2(base, &lb) == TQ_OK);
  CHECK(tq_cache_bytes_v2(disabled, &ld) == TQ_OK);
  CHECK(lb.total_bytes == ld.total_bytes);
  CHECK(ld.gaussian_bytes == 0 && ld.residual_sign_bytes == 0 && ld.residual_norm_bytes == 0);
  CHECK(tq_cache_bytes_v2(active, &la) == TQ_OK && la.version == TQ_CACHE_VERSION_2);
  CHECK(la.gaussian_bytes == m*d*sizeof(double));
  CHECK(la.residual_sign_bytes >= 3*((m+7)/8) && la.residual_norm_bytes >= 3*sizeof(double));
  CHECK(la.total_bytes == la.base.total_bytes+la.gaussian_bytes+la.residual_sign_bytes+la.residual_norm_bytes);
  tq_cache_ledger old{}; old.total_bytes = 42;
  CHECK(tq_cache_bytes(active, &old) == TQ_INVALID && old.total_bytes == 42);
  double keys[2][d] = {{.7, .4, -.2, .15}, {-.3, .9, .17, -.11}};
  double values[2][d] = {}; values[0][2] = 1; values[1][3] = 1;
  double query[d] = {.8, -.35, .1, .2}, corrected[2] = {}, raw[2] = {};
  const double scale = .37;
  for (size_t row = 0; row < 2; ++row) {
    for (auto *cache : {active, base, disabled})
      CHECK(tq_cache_append(cache, keys[row], d, values[row], d) == TQ_OK);
    double residual[d]; std::copy(keys[row], keys[row]+d, residual);
    for (unsigned r = 0; r < 3; ++r) {
      uint8_t packed[3] = {}, baseline[3] = {}, zero_beta[3] = {};
      double norm = 0, bn = 0, zn = 0, decoded[d] = {};
      CHECK(tq_cache_copy_row(active, row, r, packed, 3, &norm) == TQ_OK);
      CHECK(tq_cache_copy_row(base, row, r, baseline, 3, &bn) == TQ_OK);
      CHECK(tq_cache_copy_row(disabled, row, r, zero_beta, 3, &zn) == TQ_OK);
      CHECK(std::memcmp(packed, baseline, 3) == 0 && std::memcmp(packed, zero_beta, 3) == 0);
      CHECK(norm == bn && norm == zn);
      CHECK(tq_decode_f64(packed, 3, d, book, 2, norm, decoded) == TQ_OK);
      for (size_t j = 0; j < d; ++j) {
        double common = 0;
        for (size_t i = 0; i < d; ++i) common += rotations[r][i*d+j]*decoded[i];
        residual[j] -= cfg.base.alpha[r]*common;
        raw[row] += cfg.base.alpha[r]*query[j]*common;
      }
    }
    uint8_t signs[2] = {}; double norm = 0, squared = 0, signed_query = 0;
    CHECK(tq_cache_copy_residual_row(active, row, signs, 2, &norm) == TQ_OK);
    CHECK((signs[1] & 0xfc) == 0);
    for (double x : residual) squared += x*x;
    CHECK(std::abs(norm-std::sqrt(squared)) < 1e-14);
    for (size_t j = 0; j < m; ++j) {
      double ge = 0, gq = 0;
      for (size_t i = 0; i < d; ++i) {
        ge += gaussian[j*d+i]*residual[i]; gq += gaussian[j*d+i]*query[i];
      }
      CHECK(((signs[j/8] >> (j%8)) & 1) == (ge >= 0 ? 1 : 0));
      signed_query += ge >= 0 ? gq : -gq;
    }
    corrected[row] = scale*(raw[row]+cfg.qjl_beta*std::sqrt(std::acos(-1.)/2)*
                           std::sqrt(squared)*signed_query/static_cast<double>(m));
  }
  double output[d] = {}, baseline_output[d] = {}, disabled_output[d] = {};
  CHECK(tq_cache_attend(active, query, d, nullptr, 0, scale, output, d) == TQ_OK);
  const double expected = 1/(1+std::exp(corrected[1]-corrected[0]));
  CHECK(std::abs(output[2]-expected) < 1e-13 && std::abs(output[3]-(1-expected)) < 1e-13);
  CHECK(tq_cache_attend(base, query, d, nullptr, 0, scale, baseline_output, d) == TQ_OK);
  CHECK(tq_cache_attend(disabled, query, d, nullptr, 0, scale, disabled_output, d) == TQ_OK);
  CHECK(std::memcmp(baseline_output, disabled_output, sizeof(output)) == 0);
  CHECK(std::abs(output[2]-baseline_output[2]) > 1e-5);
  uint8_t mask[] = {0, 1};
  CHECK(tq_cache_attend(active, query, d, mask, 2, scale, disabled_output, d) == TQ_OK);
  CHECK(disabled_output[2] == 0 && disabled_output[3] == 1);
  uint8_t signs[2] = {0xaa, 0xbb}; double norm = 17;
  CHECK(tq_cache_copy_residual_row(active, 0, signs, 1, &norm) == TQ_CAPACITY);
  CHECK(signs[0] == 0xaa && signs[1] == 0xbb && norm == 17);
  alignas(double) uint8_t alias[16] = {0xaa};
  CHECK(tq_cache_copy_residual_row(active, 0, alias, 2, reinterpret_cast<double *>(alias)) == TQ_INVALID);
  CHECK(alias[0] == 0xaa);
  CHECK(tq_cache_copy_residual_row(disabled, 0, signs, 2, &norm) == TQ_INVALID);

  // Snapshot, clone in both directions, and rollback of all physical views.
  std::fill(gaussian.begin(), gaussian.end(), 0);
  cfg.qjl_beta = 0;
  CHECK(tq_cache_attend(active, query, d, nullptr, 0, scale, disabled_output, d) == TQ_OK);
  CHECK(std::memcmp(output, disabled_output, sizeof(output)) == 0);
  tq_cache *copy = nullptr;
  CHECK(tq_cache_clone(active, &copy) == TQ_OK);
  tq_cache *disabled_copy = nullptr;
  CHECK(tq_cache_clone(disabled, &disabled_copy) == TQ_OK);
  CHECK(tq_cache_bytes_v2(disabled_copy, &ld) == TQ_OK);
  CHECK(ld.gaussian_bytes == 0 && ld.residual_sign_bytes == 0 && ld.residual_norm_bytes == 0);
  tq_cache_destroy(disabled_copy);
  CHECK(tq_cache_clear(active) == TQ_OK);
  CHECK(tq_cache_bytes_v2(active, &ld) == TQ_OK && ld.total_bytes == la.total_bytes);
  CHECK(tq_cache_attend(copy, query, d, nullptr, 0, scale, disabled_output, d) == TQ_OK);
  CHECK(std::memcmp(output, disabled_output, sizeof(output)) == 0);
  double zero[d] = {};
  CHECK(tq_cache_append(active, zero, d, values[0], d) == TQ_OK);
  CHECK(tq_cache_copy_residual_row(active, 0, signs, 2, &norm) == TQ_OK);
  CHECK(norm == 0 && signs[0] == 0xff && signs[1] == 3);
  double representable[d] = {}; representable[0] = 1;
  CHECK(tq_cache_append(active, representable, d, values[0], d) == TQ_OK);
  CHECK(tq_cache_copy_residual_row(active, 1, signs, 2, &norm) == TQ_OK);
  CHECK(norm == 0 && signs[0] == 0xff && signs[1] == 3);
  CHECK(tq_cache_clear(copy) == TQ_OK && tq_cache_rows(active) == 2);
  CHECK(tq_cache_append(copy, keys[0], d, values[0], d) == TQ_OK);
  uint8_t physical_before[4][3] = {}, saved_signs[2] = {}; double physical_norms[4] = {}, saved_norm = 0;
  for (unsigned r = 0; r < 4; ++r)
    CHECK(tq_cache_copy_row(copy, 0, r, physical_before[r], 3, physical_norms+r) == TQ_OK);
  CHECK(tq_cache_copy_residual_row(copy, 0, saved_signs, 2, &saved_norm) == TQ_OK);
  double bad[d] = {}; bad[0] = std::numeric_limits<double>::infinity();
  CHECK(tq_cache_append(copy, keys[1], d, bad, d) == TQ_NONFINITE);
  CHECK(tq_cache_rows(copy) == 1);
  for (unsigned r = 0; r < 4; ++r) {
    uint8_t after[3] = {}; double after_norm = 0;
    CHECK(tq_cache_copy_row(copy, 0, r, after, 3, &after_norm) == TQ_OK);
    CHECK(std::memcmp(after, physical_before[r], 3) == 0 && after_norm == physical_norms[r]);
  }
  CHECK(tq_cache_copy_residual_row(copy, 0, signs, 2, &norm) == TQ_OK);
  CHECK(std::memcmp(signs, saved_signs, 2) == 0 && norm == saved_norm);
  CHECK(tq_cache_copy_residual_row(copy, 1, signs, 2, &norm) == TQ_INVALID);
  tq_cache_destroy(copy); tq_cache_destroy(active); tq_cache_destroy(base); tq_cache_destroy(disabled);

  // Explicit invalid/capacity cases never replace the output cache pointer.
  cfg.qjl_beta = .5;
  tq_cache *sentinel = reinterpret_cast<tq_cache *>(uintptr_t{1});
  auto invalid = cfg; invalid.qjl_beta = std::numeric_limits<double>::quiet_NaN();
  CHECK(tq_cache_create_v2(&invalid, &sentinel) == TQ_INVALID);
  invalid.qjl_beta = -.1; CHECK(tq_cache_create_v2(&invalid, &sentinel) == TQ_INVALID);
  invalid.qjl_beta = 1.1; CHECK(tq_cache_create_v2(&invalid, &sentinel) == TQ_INVALID);
  invalid = cfg; invalid.version = 99; CHECK(tq_cache_create_v2(&invalid, &sentinel) == TQ_INVALID);
  invalid = cfg; --invalid.gaussian_count; CHECK(tq_cache_create_v2(&invalid, &sentinel) == TQ_INVALID);
  invalid = cfg; --invalid.gaussian_byte_cap; CHECK(tq_cache_create_v2(&invalid, &sentinel) == TQ_CAPACITY);
  invalid = cfg; invalid.qjl_rows = SIZE_MAX; CHECK(tq_cache_create_v2(&invalid, &sentinel) == TQ_INVALID);
  invalid = cfg; invalid.qjl_rows = SIZE_MAX/sizeof(double); CHECK(tq_cache_create_v2(&invalid, &sentinel) == TQ_INVALID);
  invalid = cfg; invalid.gaussian = nullptr; CHECK(tq_cache_create_v2(&invalid, &sentinel) == TQ_INVALID);
  invalid = cfg;
  CHECK(tq_cache_create_v2(&invalid, reinterpret_cast<tq_cache **>(&invalid)) == TQ_INVALID);
  CHECK(invalid.version == TQ_CACHE_VERSION_2);
  CHECK(tq_cache_create_v2(&cfg, reinterpret_cast<tq_cache **>(gaussian.data())) == TQ_INVALID);
  gaussian[0] = std::numeric_limits<double>::infinity();
  CHECK(tq_cache_create_v2(&cfg, &sentinel) == TQ_NONFINITE);
  CHECK(sentinel == reinterpret_cast<tq_cache *>(uintptr_t{1}));
  std::fill(gaussian.begin(), gaussian.end(), 1e308);
  CHECK(tq_cache_create_v2(&cfg, &active) == TQ_OK);
  CHECK(tq_cache_append(active, zero, d, values[0], d) == TQ_OK);
  double overflowing[d]; std::fill(overflowing, overflowing+d, .3);
  CHECK(tq_cache_append(active, overflowing, d, values[1], d) == TQ_NONFINITE);
  CHECK(tq_cache_rows(active) == 1);
  CHECK(tq_cache_copy_residual_row(active, 0, signs, 2, &norm) == TQ_OK);
  CHECK(norm == 0 && signs[0] == 0xff && signs[1] == 3);
  // Zero residual does not project the query, even for a huge finite G.
  std::fill(overflowing, overflowing+d, 1.);
  CHECK(tq_cache_attend(active, overflowing, d, nullptr, 0, 1, output, d) == TQ_OK);
  CHECK(output[2] == 1);
  CHECK(tq_cache_append(active, keys[0], d, values[1], d) == TQ_OK);
  std::fill(output, output+d, 17);
  CHECK(tq_cache_attend(active, overflowing, d, nullptr, 0, 1, output, d) == TQ_NONFINITE);
  for (double x : output) CHECK(x == 17);
  CHECK(tq_cache_rows(active) == 2);
  tq_cache_destroy(active);
}

int main() {
  qjl_tests();
  constexpr size_t d = 9;
  std::vector<double> r[3];
  for (auto &m : r) m.assign(d*d, 0);
  for (size_t i = 0; i < d; ++i) {
    r[0][i*d+i] = 1;
    r[1][i*d+(i+1)%d] = 1;
    r[2][i*d+i] = i == 0 ? -1 : 1;
  }
  double kb[] = {-1, 0, 1, 2};
  double vb[] = {-1, -.5, 0, .25, .5, .75, 1, 2};
  tq_cache_config cfg{};
  cfg.width = d; cfg.capacity_rows = 2; cfg.key_bits = 2; cfg.value_bits = 3;
  for (unsigned i = 0; i < 3; ++i) cfg.transforms[i] = r[i].data();
  cfg.alpha[0] = .2; cfg.alpha[1] = .3; cfg.alpha[2] = .5;
  cfg.key_centroids = kb; cfg.value_centroids = vb;
  tq_cache *c = nullptr;
  CHECK(tq_cache_create(&cfg, &c) == TQ_OK && c);
  auto one = [d](size_t where) { std::vector<double> v(d, 0); v[where] = 1; return v; };
  auto k0 = one(0), k1 = one(1), v0 = one(2), v1 = one(3), q = one(0);
  CHECK(tq_cache_append(c, nullptr, d, v0.data(), d) == TQ_INVALID);
  CHECK(tq_cache_append(c, k0.data(), d-1, v0.data(), d) == TQ_INVALID);
  CHECK(tq_cache_append(c, k0.data(), d, v0.data(), d-1) == TQ_INVALID);
  CHECK(tq_cache_rows(c) == 0);
  CHECK(tq_cache_append(c, k0.data(), d, v0.data(), d) == TQ_OK);
  CHECK(tq_cache_append(c, k1.data(), d, v1.data(), d) == TQ_OK);
  CHECK(tq_cache_rows(c) == 2);
  uint8_t p0[3] = {}, p1[3] = {}, p2[3] = {}, pv[4] = {};
  double n0 = 0, n1 = 0, n2 = 0, nv = 0;
  CHECK(tq_cache_copy_row(c, 0, 0, p0, 3, &n0) == TQ_OK);
  CHECK(tq_cache_copy_row(c, 0, 1, p1, 3, &n1) == TQ_OK);
  CHECK(tq_cache_copy_row(c, 0, 2, p2, 3, &n2) == TQ_OK);
  CHECK(tq_cache_copy_row(c, 0, 3, pv, 4, &nv) == TQ_OK);
  uint8_t too_small[3] = {0xaa, 0xbb, 0xcc};
  double untouched_norm = 17;
  CHECK(tq_cache_copy_row(c, 0, 3, too_small, 3, &untouched_norm) == TQ_CAPACITY);
  CHECK(too_small[0] == 0xaa && too_small[1] == 0xbb && untouched_norm == 17);
  alignas(double) uint8_t overlap[16] = {0xaa};
  CHECK(tq_cache_copy_row(c, 0, 0, overlap, 3,
                          reinterpret_cast<double *>(overlap)) == TQ_INVALID);
  CHECK(overlap[0] == 0xaa);
  CHECK(p0[0] != p1[0] || p0[1] != p1[1] || p0[2] != p1[2]);
  CHECK(p0[0] != p2[0] || p0[1] != p2[1] || p0[2] != p2[2]);
  CHECK(n0 == 1 && n1 == 1 && n2 == 1 && nv == 1);
  tq_cache_ledger ledger{};
  CHECK(tq_cache_bytes(c, &ledger) == TQ_OK);
  CHECK(ledger.key_payload_bytes >= 3*2*3);
  CHECK(ledger.value_payload_bytes >= 2*4);
  CHECK(ledger.key_norm_bytes >= 3*2*sizeof(double));
  CHECK(ledger.value_norm_bytes >= 2*sizeof(double));
  CHECK(ledger.transform_bytes >= 3*d*d*sizeof(double));
  CHECK(ledger.total_bytes == ledger.fixed_bytes+ledger.alpha_bytes+
        ledger.transform_bytes+ledger.codebook_bytes+ledger.key_payload_bytes+
        ledger.key_norm_bytes+ledger.value_payload_bytes+ledger.value_norm_bytes);
  // Mutating the supplied matrices, books, and weights cannot affect the cache.
  r[0][0] = 7; kb[0] = -99; vb[0] = -99; cfg.alpha[0] = 1;
  double out[d] = {};
  CHECK(tq_cache_attend(c, q.data(), d, nullptr, 0, 1, out, d) == TQ_OK);
  const double w0 = std::exp(1.)/(std::exp(1.)+1.);
  CHECK(std::abs(out[2]-w0) < 1e-13);
  CHECK(std::abs(out[3]-(1-w0)) < 1e-13);
  for (size_t i = 0; i < d; ++i) if (i != 2 && i != 3) CHECK(out[i] == 0);
  CHECK(tq_cache_attend(c, q.data(), d, nullptr, 0, 1000, out, d) == TQ_OK);
  CHECK(out[2] == 1 && out[3] >= 0 && out[3] < 1e-300);
  auto negative_q = q; negative_q[0] = -1;
  CHECK(tq_cache_attend(c, negative_q.data(), d, nullptr, 0, 1000, out, d) == TQ_OK);
  CHECK(out[3] == 1 && out[2] >= 0 && out[2] < 1e-300);
  out[0] = 123;
  CHECK(tq_cache_attend(c, nullptr, d, nullptr, 0, 1, out, d) == TQ_INVALID);
  CHECK(tq_cache_attend(c, q.data(), d-1, nullptr, 0, 1, out, d) == TQ_INVALID);
  CHECK(out[0] == 123);
  uint8_t mask[] = {0, 1};
  CHECK(tq_cache_attend(c, q.data(), d, mask, 2, 1, out, d) == TQ_OK);
  CHECK(out[2] == 0 && out[3] == 1);
  mask[1] = 0; out[0] = 123;
  CHECK(tq_cache_attend(c, q.data(), d, mask, 2, 1, out, d) == TQ_INVALID);
  CHECK(out[0] == 123);
  CHECK(tq_cache_attend(c, q.data(), d, nullptr, 0, 1, out, d-1) == TQ_CAPACITY);
  CHECK(out[0] == 123);
  CHECK(tq_cache_append(c, k0.data(), d, v0.data(), d) == TQ_CAPACITY);
  CHECK(tq_cache_rows(c) == 2);
  tq_cache *copy = nullptr;
  CHECK(tq_cache_clone(c, &copy) == TQ_OK && copy);
  CHECK(tq_cache_clear(c) == TQ_OK && tq_cache_rows(c) == 0);
  CHECK(tq_cache_rows(copy) == 2);
  CHECK(tq_cache_attend(copy, q.data(), d, nullptr, 0, 1, out, d) == TQ_OK);
  CHECK(std::abs(out[2]-w0) < 1e-13);
  CHECK(tq_cache_append(c, k0.data(), d, v0.data(), d) == TQ_OK);
  CHECK(tq_cache_clear(copy) == TQ_OK);
  CHECK(tq_cache_rows(copy) == 0 && tq_cache_rows(c) == 1);
  CHECK(tq_cache_attend(c, q.data(), d, nullptr, 0, 1, out, d) == TQ_OK);
  CHECK(out[2] == 1 && out[3] == 0);
  CHECK(tq_cache_clear(c) == TQ_OK);
  double bad[d] = {};
  bad[0] = std::numeric_limits<double>::quiet_NaN();
  CHECK(tq_cache_append(c, bad, d, v0.data(), d) == TQ_NONFINITE);
  CHECK(tq_cache_rows(c) == 0);
  CHECK(tq_cache_append(c, k0.data(), d, v0.data(), d) == TQ_OK);
  CHECK(tq_cache_rows(c) == 1);
  CHECK(tq_cache_attend(c, bad, d, nullptr, 0, 1, out, d) == TQ_NONFINITE);
  tq_cache_destroy(copy); tq_cache_destroy(c);

  // Nonrepresentable keys: independently expand all physical rows and score
  // them in each transformed coordinate system, then apply one softmax.
  for (size_t i = 0; i < d; ++i) r[0][i*d+i] = 1;
  kb[0] = -1; vb[0] = -1; cfg.alpha[0] = .2;
  CHECK(tq_cache_create(&cfg, &c) == TQ_OK);
  k0[0] = .7; k0[1] = .4; k1[0] = -.3; k1[1] = .9;
  CHECK(tq_cache_append(c, k0.data(), d, v0.data(), d) == TQ_OK);
  CHECK(tq_cache_append(c, k1.data(), d, v1.data(), d) == TQ_OK);
  double scores[2] = {}, reconstructed[d] = {};
  for (size_t row = 0; row < 2; ++row) for (unsigned view = 0; view < 3; ++view) {
    uint8_t packed[3] = {};
    double norm = 0;
    CHECK(tq_cache_copy_row(c, row, view, packed, 3, &norm) == TQ_OK);
    CHECK(tq_decode_f64(packed, 3, d, kb, 2, norm, reconstructed) == TQ_OK);
    for (size_t i = 0; i < d; ++i) {
      double rotated_q = 0;
      for (size_t j = 0; j < d; ++j) rotated_q += r[view][i*d+j]*q[j];
      scores[row] += cfg.alpha[view]*rotated_q*reconstructed[i];
    }
  }
  const double e0 = std::exp(scores[0]-std::max(scores[0], scores[1]));
  const double e1 = std::exp(scores[1]-std::max(scores[0], scores[1]));
  CHECK(tq_cache_attend(c, q.data(), d, nullptr, 0, 1, out, d) == TQ_OK);
  CHECK(std::abs(out[2]-e0/(e0+e1)) < 1e-13);
  CHECK(std::abs(out[3]-e1/(e0+e1)) < 1e-13);
  tq_cache_destroy(c);

  r[0][0] = 2;
  c = nullptr;
  CHECK(tq_cache_create(&cfg, &c) == TQ_INVALID && !c);
  std::printf("tqcodec cache reference failures: %d\n", errors);
  return errors ? 1 : 0;
}
