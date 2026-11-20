#include "tqcodec/cache.h"
#include "tqcodec/transport.h"
#include <algorithm>
#include <cmath>
#include <cstring>
#include <limits>
#include <new>
#include <stdexcept>
#include <vector>

struct tq_cache {
  size_t d = 0, limit = 0, rows = 0, key_stride = 0, value_stride = 0;
  unsigned key_bits = 0, value_bits = 0;
  double alpha[3] = {};
  std::vector<double> rotation[3], key_book, value_book;
  std::vector<uint8_t> key_data[3], value_data;
  std::vector<double> key_norm[3], value_norm;
  double qjl_beta = 0;
  size_t qjl_rows = 0, sign_stride = 0;
  std::vector<double> gaussian, residual_norm;
  std::vector<uint8_t> residual_signs;
};

struct tq_transport_plan {
  tq_transport_audit audit{};
  std::vector<double> matrix;
};

namespace {
bool product(size_t a, size_t b, size_t &out) {
  if (b && a > SIZE_MAX / b) return false;
  out = a * b;
  return true;
}
bool add_product(size_t count, size_t stride, size_t &total) {
  size_t bytes;
  if (!product(count, stride, bytes) || bytes > SIZE_MAX-total) return false;
  total += bytes;
  return true;
}
bool finite_vector(const double *p, size_t n) {
  if (!p || n > SIZE_MAX / sizeof(double)) return false;
  for (size_t i = 0; i < n; ++i) if (!std::isfinite(p[i])) return false;
  return true;
}
bool book_ok(const double *p, unsigned bits) {
  if (bits < 2 || bits > 6 || !finite_vector(p, size_t{1} << bits)) return false;
  for (size_t i = 1; i < (size_t{1} << bits); ++i)
    if (p[i] <= p[i - 1]) return false;
  return true;
}
bool orthogonal(const double *a, size_t d, tq_transport_audit *audit = nullptr,
                bool proper = false) {
  size_t n;
  if (!product(d, d, n) || !finite_vector(a, n)) return false;
  // FP64 Gram test, then a pivoted determinant test. Legacy cache U admits
  // reflections; the external transport plan explicitly requires det +1.
  double gram_error_squared = 0, gram_max_error = 0;
  for (size_t i = 0; i < d; ++i) for (size_t j = i; j < d; ++j) {
    double dot = 0;
    for (size_t k = 0; k < d; ++k) dot += a[k*d+i] * a[k*d+j];
    const double error = dot - (i == j ? 1. : 0.);
    if (!std::isfinite(error) || std::abs(error) > 1e-12)
      return false;
    gram_error_squared += (i == j ? 1. : 2.)*error*error;
    gram_max_error = std::max(gram_max_error, std::abs(error));
  }
  if (!std::isfinite(gram_error_squared) ||
      std::sqrt(gram_error_squared/static_cast<double>(d)) > 1e-12)
    return false;
  std::vector<double> m(a, a+n);
  double log_abs_determinant = 0;
  int determinant_sign = 1;
  for (size_t col = 0; col < d; ++col) {
    size_t pivot = col;
    for (size_t row = col+1; row < d; ++row)
      if (std::abs(m[row*d+col]) > std::abs(m[pivot*d+col])) pivot = row;
    if (std::abs(m[pivot*d+col]) < 1e-14) return false;
    if (pivot != col) {
      for (size_t j = 0; j < d; ++j) std::swap(m[pivot*d+j], m[col*d+j]);
      determinant_sign = -determinant_sign;
    }
    const double diagonal = m[col*d+col];
    if (diagonal < 0) determinant_sign = -determinant_sign;
    log_abs_determinant += std::log(std::abs(diagonal));
    for (size_t row = col+1; row < d; ++row) {
      const double factor = m[row*d+col] / diagonal;
      for (size_t j = col+1; j < d; ++j)
        m[row*d+j] -= factor * m[col*d+j];
    }
  }
  const double determinant = determinant_sign*std::exp(log_abs_determinant);
  if (!std::isfinite(determinant) ||
      std::abs(log_abs_determinant) > static_cast<double>(d)*1e-12 ||
      (proper && determinant_sign != 1)) return false;
  if (audit) {
    audit->width = d;
    audit->gram_frobenius_error =
        std::sqrt(gram_error_squared/static_cast<double>(d));
    audit->gram_max_error = gram_max_error;
    audit->log_abs_determinant = log_abs_determinant;
    audit->determinant_sign = determinant_sign;
  }
  return true;
}
bool rotated(const std::vector<double> &r, const double *x, size_t d,
             std::vector<double> &y) {
  for (size_t i = 0; i < d; ++i) {
    double sum = 0;
    for (size_t j = 0; j < d; ++j) sum += r[i*d+j] * x[j];
    if (!std::isfinite(sum)) return false;
    y[i] = sum;
  }
  return true;
}
bool overlaps(const void *p, size_t n, const void *q, size_t m) {
  auto a = reinterpret_cast<uintptr_t>(p), b = reinterpret_cast<uintptr_t>(q);
  if (a > UINTPTR_MAX-n || b > UINTPTR_MAX-m) return true;
  return n && m && a < b+m && b < a+n;
}
template<class T> bool aligned(const T *p) {
  return p && reinterpret_cast<uintptr_t>(p) % alignof(T) == 0;
}
bool address_range(const void *p, size_t bytes) {
  return p && reinterpret_cast<uintptr_t>(p) <= UINTPTR_MAX-bytes;
}
bool plan_overlap(const tq_transport_plan *p, const void *out, size_t bytes) {
  return overlaps(p, sizeof(*p), out, bytes) ||
      overlaps(p->matrix.data(), p->matrix.capacity()*sizeof(double), out, bytes);
}
template<class T> bool vector_overlap(const std::vector<T> &v,
                                    const void *out, size_t bytes) {
  return overlaps(v.data(), v.capacity()*sizeof(T), out, bytes);
}
bool cache_overlap(const tq_cache *c, const void *out, size_t bytes) {
  if (overlaps(c, sizeof(*c), out, bytes) ||
      vector_overlap(c->key_book, out, bytes) ||
      vector_overlap(c->value_book, out, bytes) ||
      vector_overlap(c->value_data, out, bytes) ||
      vector_overlap(c->value_norm, out, bytes) ||
      vector_overlap(c->gaussian, out, bytes) ||
      vector_overlap(c->residual_norm, out, bytes) ||
      vector_overlap(c->residual_signs, out, bytes)) return true;
  for (unsigned r = 0; r < 3; ++r)
    if (vector_overlap(c->rotation[r], out, bytes) ||
        vector_overlap(c->key_data[r], out, bytes) ||
        vector_overlap(c->key_norm[r], out, bytes)) return true;
  return false;
}
} // namespace

extern "C" {
tq_status tq_transport_plan_create(size_t d, const double *matrix,
    size_t matrix_count, size_t matrix_byte_cap, tq_transport_plan **out) {
  size_t n, bytes;
  if (d < 8 || !aligned(matrix) || !aligned(out) ||
      !product(d, d, n) || !product(n, sizeof(double), bytes) ||
      matrix_count < n || !address_range(matrix, bytes) ||
      !address_range(out, sizeof(*out)) ||
      overlaps(matrix, bytes, out, sizeof(*out))) return TQ_INVALID;
  if (matrix_byte_cap < bytes) return TQ_CAPACITY;
  if (!finite_vector(matrix, n)) return TQ_NONFINITE;
  try {
    tq_transport_audit audit{};
    if (!orthogonal(matrix, d, &audit, true)) return TQ_INVALID;
    tq_transport_plan *p = new tq_transport_plan;
    try {
      p->audit = audit;
      p->matrix.assign(matrix, matrix+n);
      size_t allocated;
      if (!product(p->matrix.capacity(), sizeof(double), allocated) ||
          allocated > matrix_byte_cap) { delete p; return TQ_CAPACITY; }
    } catch (...) { delete p; throw; }
    *out = p;
    return TQ_OK;
  } catch (const std::bad_alloc &) { return TQ_CAPACITY;
  } catch (const std::length_error &) { return TQ_CAPACITY;
  } catch (...) { return TQ_INVALID; }
}
void tq_transport_plan_destroy(tq_transport_plan *p) { delete p; }
tq_status tq_transport_plan_audit(const tq_transport_plan *p,
                                tq_transport_audit *out) {
  if (!aligned(p) || !aligned(out) || !address_range(out, sizeof(*out)) ||
      plan_overlap(p, out, sizeof(*out))) return TQ_INVALID;
  *out = p->audit;
  return TQ_OK;
}
tq_status tq_transport_plan_bytes(const tq_transport_plan *p,
                                tq_transport_ledger *out) {
  if (!aligned(p) || !aligned(out) || !address_range(out, sizeof(*out)) ||
      plan_overlap(p, out, sizeof(*out))) return TQ_INVALID;
  tq_transport_ledger ledger{};
  ledger.fixed_bytes = sizeof(*p);
  ledger.total_bytes = sizeof(*p);
  if (!product(p->matrix.capacity(), sizeof(double), ledger.matrix_bytes) ||
      !add_product(ledger.matrix_bytes, 1, ledger.total_bytes)) return TQ_INVALID;
  *out = ledger;
  return TQ_OK;
}
tq_status tq_transport_to_storage_f64(const tq_transport_plan *p,
    const double *src, size_t src_count, double *dst, size_t dst_count) {
  if (!aligned(p) || !aligned(src) || !aligned(dst)) return TQ_INVALID;
  const size_t d = p->audit.width, bytes = d*sizeof(double);
  if (src_count < d || !address_range(src, bytes) ||
      !address_range(dst, bytes) || plan_overlap(p, dst, bytes) ||
      (src != dst && overlaps(src, bytes, dst, bytes))) return TQ_INVALID;
  if (dst_count < d) return TQ_CAPACITY;
  if (!finite_vector(src, d)) return TQ_NONFINITE;
  try {
    std::vector<double> staged(d, 0);
    for (size_t j = 0; j < d; ++j) {
      for (size_t i = 0; i < d; ++i)
        staged[j] += p->matrix[i*d+j]*src[i];
      if (!std::isfinite(staged[j])) return TQ_NONFINITE;
    }
    std::memcpy(dst, staged.data(), bytes);
    return TQ_OK;
  } catch (const std::bad_alloc &) { return TQ_CAPACITY;
  } catch (const std::length_error &) { return TQ_CAPACITY;
  } catch (...) { return TQ_INVALID; }
}
tq_status tq_cache_attend_transported(const tq_cache *c,
    const tq_transport_plan *p, const double *query, size_t query_count,
    const uint8_t *allowed, size_t allowed_count, double scale, double *out,
    size_t out_count) {
  if (!aligned(c) || !aligned(p) || !aligned(query) || !aligned(out) ||
      c->d != p->audit.width || query_count < c->d || !c->rows ||
      !std::isfinite(scale) || scale <= 0 ||
      (allowed && (allowed_count < c->rows ||
                   !address_range(allowed, c->rows)))) return TQ_INVALID;
  const size_t bytes = c->d*sizeof(double);
  if (!address_range(query, bytes) || !address_range(out, bytes) ||
      plan_overlap(p, out, bytes) || cache_overlap(c, out, bytes) ||
      (query != out && overlaps(query, bytes, out, bytes)) ||
      (allowed && overlaps(allowed, c->rows, out, bytes))) return TQ_INVALID;
  if (out_count < c->d) return TQ_CAPACITY;
  try {
    std::vector<double> canonical(c->d);
    const tq_status st = tq_transport_to_storage_f64(p, query, query_count,
                                                   canonical.data(), c->d);
    if (st != TQ_OK) return st;
    return tq_cache_attend(c, canonical.data(), c->d, allowed, allowed_count,
                           scale, out, out_count);
  } catch (const std::bad_alloc &) { return TQ_CAPACITY;
  } catch (const std::length_error &) { return TQ_CAPACITY;
  } catch (...) { return TQ_INVALID; }
}
tq_status tq_cache_create(const tq_cache_config *cfg, tq_cache **out) {
  if (!cfg || !out || cfg->width < 8 || !cfg->capacity_rows ||
      !book_ok(cfg->key_centroids, cfg->key_bits) ||
      !book_ok(cfg->value_centroids, cfg->value_bits)) return TQ_INVALID;
  size_t matrix_elems, capacity_elems;
  if (!product(cfg->width, cfg->width, matrix_elems) ||
      !product(matrix_elems, sizeof(double), capacity_elems) ||
      !product(cfg->capacity_rows, sizeof(double), capacity_elems)) return TQ_INVALID;
  double sum = 0;
  for (unsigned i = 0; i < 3; ++i) {
    if (!std::isfinite(cfg->alpha[i]) || cfg->alpha[i] < 0 ||
        !cfg->transforms[i]) return TQ_INVALID;
    sum += cfg->alpha[i];
  }
  if (!std::isfinite(sum) || std::abs(sum-1.) > 1e-12) return TQ_INVALID;
  tq_size ks{}, vs{};
  if (tq_block_size(cfg->width, cfg->key_bits, &ks) != TQ_OK ||
      tq_block_size(cfg->width, cfg->value_bits, &vs) != TQ_OK ||
      !product(cfg->capacity_rows, ks.payload_bytes, capacity_elems) ||
      !product(cfg->capacity_rows, vs.payload_bytes, capacity_elems))
    return TQ_INVALID;
  try {
    for (unsigned i = 0; i < 3; ++i)
      if (!orthogonal(cfg->transforms[i], cfg->width)) return TQ_INVALID;
    tq_cache *c = new tq_cache;
    try {
      c->d = cfg->width; c->limit = cfg->capacity_rows;
      c->key_bits = cfg->key_bits; c->value_bits = cfg->value_bits;
      c->key_stride = ks.payload_bytes; c->value_stride = vs.payload_bytes;
      for (unsigned i = 0; i < 3; ++i) {
        c->alpha[i] = cfg->alpha[i];
        c->rotation[i].assign(cfg->transforms[i], cfg->transforms[i]+matrix_elems);
        c->key_data[i].resize(c->limit * c->key_stride);
        c->key_norm[i].resize(c->limit);
      }
      c->key_book.assign(cfg->key_centroids,
                         cfg->key_centroids+(size_t{1} << c->key_bits));
      c->value_book.assign(cfg->value_centroids,
                           cfg->value_centroids+(size_t{1} << c->value_bits));
      c->value_data.resize(c->limit * c->value_stride);
      c->value_norm.resize(c->limit);
    } catch (...) { delete c; throw; }
    *out = c;
    return TQ_OK;
  } catch (const std::bad_alloc &) { return TQ_CAPACITY;
  } catch (const std::length_error &) { return TQ_CAPACITY; }
}
tq_status tq_cache_create_v2(const tq_cache_config_v2 *cfg, tq_cache **out) {
  if (!cfg || !out || cfg->version != TQ_CACHE_VERSION_2 ||
      !std::isfinite(cfg->qjl_beta) || cfg->qjl_beta < 0 || cfg->qjl_beta > 1 ||
      overlaps(cfg, sizeof(*cfg), out, sizeof(*out))) return TQ_INVALID;
  if (cfg->qjl_beta == 0) return tq_cache_create(&cfg->base, out);
  size_t elements, bytes, sign_bytes, norm_bytes;
  if (!cfg->gaussian || !cfg->qjl_rows || cfg->qjl_rows > SIZE_MAX-7 ||
      !product(cfg->qjl_rows, cfg->base.width, elements) ||
      !product(elements, sizeof(double), bytes) ||
      cfg->gaussian_count < elements ||
      !product(cfg->base.capacity_rows, (cfg->qjl_rows+7)/8, sign_bytes) ||
      !product(cfg->base.capacity_rows, sizeof(double), norm_bytes) ||
      overlaps(cfg->gaussian, bytes, out, sizeof(*out))) return TQ_INVALID;
  if (bytes > cfg->gaussian_byte_cap) return TQ_CAPACITY;
  if (!finite_vector(cfg->gaussian, elements)) return TQ_NONFINITE;
  tq_cache *c = nullptr;
  const tq_status st = tq_cache_create(&cfg->base, &c);
  if (st != TQ_OK) return st;
  try {
    c->qjl_beta = cfg->qjl_beta; c->qjl_rows = cfg->qjl_rows;
    c->sign_stride = (cfg->qjl_rows+7)/8;
    c->gaussian.assign(cfg->gaussian, cfg->gaussian+elements);
    size_t allocated;
    if (!product(c->gaussian.capacity(), sizeof(double), allocated) ||
        allocated > cfg->gaussian_byte_cap) {
      delete c; return TQ_CAPACITY;
    }
    c->residual_signs.resize(sign_bytes);
    c->residual_norm.resize(cfg->base.capacity_rows);
    *out = c;
    return TQ_OK;
  } catch (const std::bad_alloc &) { delete c; return TQ_CAPACITY;
  } catch (const std::length_error &) { delete c; return TQ_CAPACITY; }
}
void tq_cache_destroy(tq_cache *c) { delete c; }
tq_status tq_cache_clone(const tq_cache *source, tq_cache **out) {
  if (!source || !out) return TQ_INVALID;
  try { tq_cache *copy = new tq_cache(*source); *out = copy; return TQ_OK;
  } catch (const std::bad_alloc &) { return TQ_CAPACITY;
  } catch (const std::length_error &) { return TQ_CAPACITY; }
}
size_t tq_cache_rows(const tq_cache *c) { return c ? c->rows : 0; }
tq_status tq_cache_clear(tq_cache *c) {
  if (!c) return TQ_INVALID;
  for (unsigned i = 0; i < 3; ++i) {
    std::fill(c->key_data[i].begin(), c->key_data[i].end(), uint8_t{0});
    std::fill(c->key_norm[i].begin(), c->key_norm[i].end(), 0);
  }
  std::fill(c->value_data.begin(), c->value_data.end(), uint8_t{0});
  std::fill(c->value_norm.begin(), c->value_norm.end(), 0);
  std::fill(c->residual_signs.begin(), c->residual_signs.end(), uint8_t{0});
  std::fill(c->residual_norm.begin(), c->residual_norm.end(), 0);
  c->rows = 0;
  return TQ_OK;
}
static tq_status cache_base_bytes(const tq_cache *c, tq_cache_ledger *out) {
  if (!c || !out) return TQ_INVALID;
  tq_cache_ledger l{};
  l.fixed_bytes = sizeof(*c)-sizeof(c->alpha);
  l.alpha_bytes = sizeof(c->alpha);
  for (unsigned i = 0; i < 3; ++i) {
    if (!add_product(c->rotation[i].capacity(), sizeof(double), l.transform_bytes) ||
        !add_product(c->key_data[i].capacity(), 1, l.key_payload_bytes) ||
        !add_product(c->key_norm[i].capacity(), sizeof(double), l.key_norm_bytes))
      return TQ_INVALID;
  }
  if (!add_product(c->key_book.capacity(), sizeof(double), l.codebook_bytes) ||
      !add_product(c->value_book.capacity(), sizeof(double), l.codebook_bytes))
    return TQ_INVALID;
  l.value_payload_bytes = c->value_data.capacity();
  if (!product(c->value_norm.capacity(), sizeof(double), l.value_norm_bytes) ||
      !add_product(l.fixed_bytes, 1, l.total_bytes) ||
      !add_product(l.alpha_bytes, 1, l.total_bytes) ||
      !add_product(l.transform_bytes, 1, l.total_bytes) ||
      !add_product(l.codebook_bytes, 1, l.total_bytes) ||
      !add_product(l.key_payload_bytes, 1, l.total_bytes) ||
      !add_product(l.key_norm_bytes, 1, l.total_bytes) ||
      !add_product(l.value_payload_bytes, 1, l.total_bytes) ||
      !add_product(l.value_norm_bytes, 1, l.total_bytes)) return TQ_INVALID;
  *out = l;
  return TQ_OK;
}
tq_status tq_cache_bytes(const tq_cache *c, tq_cache_ledger *out) {
  if (!c || c->qjl_beta != 0) return TQ_INVALID;
  return cache_base_bytes(c, out);
}
tq_status tq_cache_bytes_v2(const tq_cache *c, tq_cache_ledger_v2 *out) {
  if (!c || !out) return TQ_INVALID;
  tq_cache_ledger_v2 l{};
  l.version = TQ_CACHE_VERSION_2;
  if (cache_base_bytes(c, &l.base) != TQ_OK ||
      !product(c->gaussian.capacity(), sizeof(double), l.gaussian_bytes) ||
      !product(c->residual_norm.capacity(), sizeof(double), l.residual_norm_bytes))
    return TQ_INVALID;
  l.residual_sign_bytes = c->residual_signs.capacity();
  l.total_bytes = l.base.total_bytes;
  if (!add_product(l.gaussian_bytes, 1, l.total_bytes) ||
      !add_product(l.residual_sign_bytes, 1, l.total_bytes) ||
      !add_product(l.residual_norm_bytes, 1, l.total_bytes)) return TQ_INVALID;
  *out = l;
  return TQ_OK;
}
tq_status tq_cache_append(tq_cache *c, const double *key, size_t key_count,
                          const double *value, size_t value_count) {
  if (!c || !key || !value || key_count < c->d || value_count < c->d)
    return TQ_INVALID;
  if (c->rows == c->limit) return TQ_CAPACITY;
  if (!finite_vector(key, c->d) || !finite_vector(value, c->d))
    return TQ_NONFINITE;
  try {
    std::vector<double> rotated_key(c->d);
    std::vector<uint8_t> kp[3];
    double kn[3] = {}, vn = 0;
    for (unsigned i = 0; i < 3; ++i) {
      if (!rotated(c->rotation[i], key, c->d, rotated_key)) return TQ_NONFINITE;
      kp[i].resize(c->key_stride);
      tq_status st = tq_encode_f64(rotated_key.data(), c->d,
          c->key_book.data(), c->key_bits, kp[i].data(), kp[i].size(), &kn[i]);
      if (st != TQ_OK) return st;
    }
    std::vector<uint8_t> vp(c->value_stride);
    tq_status st = tq_encode_f64(value, c->d, c->value_book.data(),
        c->value_bits, vp.data(), vp.size(), &vn);
    if (st != TQ_OK) return st;
    std::vector<uint8_t> signs;
    double residual_norm = 0;
    if (c->qjl_beta != 0) {
      std::vector<double> common[3], decoded(c->d), residual(c->d);
      const double *views[3];
      for (unsigned r = 0; r < 3; ++r) {
        common[r].resize(c->d);
        st = tq_decode_f64(kp[r].data(), kp[r].size(), c->d,
                          c->key_book.data(), c->key_bits, kn[r], decoded.data());
        if (st != TQ_OK) return st;
        for (size_t j = 0; j < c->d; ++j) {
          double sum = 0;
          for (size_t i = 0; i < c->d; ++i)
            sum += c->rotation[r][i*c->d+j]*decoded[i];
          if (!std::isfinite(sum)) return TQ_NONFINITE;
          common[r][j] = sum;
        }
        views[r] = common[r].data();
      }
      st = tq_shared_residual_f64(key, views, c->d, c->alpha, residual.data());
      if (st != TQ_OK) return st;
      signs.resize(c->sign_stride);
      st = tq_qjl_encode_f64(residual.data(), c->d, c->gaussian.data(),
                            c->qjl_rows, signs.data(), signs.size(), &residual_norm);
      if (st != TQ_OK) return st;
    }
    const size_t row = c->rows;
    for (unsigned i = 0; i < 3; ++i) {
      std::memcpy(c->key_data[i].data()+row*c->key_stride,
                  kp[i].data(), c->key_stride);
      c->key_norm[i][row] = kn[i];
    }
    std::memcpy(c->value_data.data()+row*c->value_stride,
                vp.data(), c->value_stride);
    c->value_norm[row] = vn;
    if (c->qjl_beta != 0) {
      std::memcpy(c->residual_signs.data()+row*c->sign_stride,
                  signs.data(), c->sign_stride);
      c->residual_norm[row] = residual_norm;
    }
    ++c->rows;
    return TQ_OK;
  } catch (const std::bad_alloc &) { return TQ_CAPACITY;
  } catch (const std::length_error &) { return TQ_CAPACITY; }
}
tq_status tq_cache_attend(const tq_cache *c, const double *query,
                          size_t query_count, const uint8_t *allowed,
                          size_t allowed_count, double scale, double *out,
                          size_t out_count) {
  if (!c || !query || !out || query_count < c->d || !c->rows ||
      !std::isfinite(scale) || scale <= 0 ||
      (allowed && allowed_count < c->rows)) return TQ_INVALID;
  if (out_count < c->d) return TQ_CAPACITY;
  if (!finite_vector(query, c->d)) return TQ_NONFINITE;
  try {
    std::vector<double> rq[3], value(c->d), acc(c->d, 0);
    for (unsigned i = 0; i < 3; ++i) {
      rq[i].resize(c->d);
      if (!rotated(c->rotation[i], query, c->d, rq[i])) return TQ_NONFINITE;
    }
    std::vector<double> projected_query;
    if (c->qjl_beta != 0) {
      bool needed = false;
      for (size_t row = 0; row < c->rows; ++row)
        if ((!allowed || allowed[row]) && c->residual_norm[row] != 0) needed = true;
      if (needed) {
        projected_query.resize(c->qjl_rows);
        for (size_t j = 0; j < c->qjl_rows; ++j) {
          double sum = 0;
          for (size_t i = 0; i < c->d; ++i)
            sum += c->gaussian[j*c->d+i]*query[i];
          if (!std::isfinite(sum)) return TQ_NONFINITE;
          projected_query[j] = sum;
        }
      }
    }
    bool any = false;
    double maximum = 0, denominator = 0;
    for (size_t row = 0; row < c->rows; ++row) {
      if (allowed && !allowed[row]) continue;
      double weighted = 0;
      for (unsigned i = 0; i < 3; ++i) {
        double part = 0;
        tq_status st = tq_score_f64(rq[i].data(),
            c->key_data[i].data()+row*c->key_stride, c->key_stride,
            c->d, c->key_book.data(), c->key_bits, c->key_norm[i][row], &part);
        if (st != TQ_OK) return st;
        weighted += c->alpha[i]*part;
      }
      if (c->qjl_beta != 0 && c->residual_norm[row] != 0) {
        const uint8_t *signs = c->residual_signs.data()+row*c->sign_stride;
        double sum = 0;
        for (size_t j = 0; j < c->qjl_rows; ++j)
          sum += ((signs[j/8] >> (j%8)) & 1) ? projected_query[j] : -projected_query[j];
        const double correction = c->qjl_beta*std::sqrt(std::acos(-1.)/2)*
            c->residual_norm[row]*(sum/static_cast<double>(c->qjl_rows));
        if (!std::isfinite(correction)) return TQ_NONFINITE;
        weighted += correction;
      }
      const double score = scale*weighted;
      if (!std::isfinite(score)) return TQ_NONFINITE;
      tq_status st = tq_decode_f64(c->value_data.data()+row*c->value_stride,
          c->value_stride, c->d, c->value_book.data(), c->value_bits,
          c->value_norm[row], value.data());
      if (st != TQ_OK) return st;
      const double next_max = any ? std::max(maximum, score) : score;
      const double old_weight = any ? std::exp(maximum-next_max) : 0.;
      const double new_weight = std::exp(score-next_max);
      denominator = denominator*old_weight+new_weight;
      if (!std::isfinite(denominator) || denominator <= 0) return TQ_NONFINITE;
      for (size_t j = 0; j < c->d; ++j) {
        acc[j] = acc[j]*old_weight+new_weight*value[j];
        if (!std::isfinite(acc[j])) return TQ_NONFINITE;
      }
      maximum = next_max; any = true;
    }
    if (!any) return TQ_INVALID;
    for (double &v : acc) {
      v /= denominator;
      if (!std::isfinite(v)) return TQ_NONFINITE;
    }
    std::memcpy(out, acc.data(), c->d*sizeof(double));
    return TQ_OK;
  } catch (const std::bad_alloc &) { return TQ_CAPACITY;
  } catch (const std::length_error &) { return TQ_CAPACITY; }
}
tq_status tq_cache_copy_row(const tq_cache *c, size_t row, unsigned view,
                            uint8_t *payload, size_t payload_capacity,
                            double *norm) {
  if (!c || row >= c->rows || view > 3 || !payload || !norm)
    return TQ_INVALID;
  const size_t stride = view == 3 ? c->value_stride : c->key_stride;
  if (payload_capacity < stride) return TQ_CAPACITY;
  if (overlaps(payload, stride, norm, sizeof(double))) return TQ_INVALID;
  const uint8_t *source = view == 3 ? c->value_data.data()+row*stride :
                                     c->key_data[view].data()+row*stride;
  const double n = view == 3 ? c->value_norm[row] : c->key_norm[view][row];
  std::memcpy(payload, source, stride);
  *norm = n;
  return TQ_OK;
}
tq_status tq_cache_copy_residual_row(const tq_cache *c, size_t row,
                                     uint8_t *signs, size_t capacity, double *norm) {
  if (!c || c->qjl_beta == 0 || row >= c->rows || !signs || !norm)
    return TQ_INVALID;
  if (capacity < c->sign_stride) return TQ_CAPACITY;
  if (overlaps(signs, c->sign_stride, norm, sizeof(double))) return TQ_INVALID;
  std::memcpy(signs, c->residual_signs.data()+row*c->sign_stride, c->sign_stride);
  *norm = c->residual_norm[row];
  return TQ_OK;
}
}
