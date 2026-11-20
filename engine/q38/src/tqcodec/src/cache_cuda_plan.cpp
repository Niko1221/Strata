#include "cache_cuda_internal.h"
#include "spin8/spin8.h"
#include <algorithm>
#include <cmath>
#include <memory>
#include <new>
#include <vector>
extern "C" {
#include "../../spin8/third_party/sha256/sha256.h"
}
namespace {

bool identity(const uint8_t *p) {
  for (int i = 0; i < 32; ++i)
    if (p[i])
      return true;
  return false;
}
} // namespace
namespace tqc_cache_detail {
void hash(const void *p, size_t n, uint8_t out[32]) {
  spin8_sidecar_sha256_hash(out, static_cast<const unsigned char *>(p), n);
}
} // namespace tqc_cache_detail
extern "C" int tqc_cache_cuda_create(const tqc_cache_cuda_config *c,
                                     tqc_cache_cuda_plan **out) {
  using namespace tqc_cache_detail;
  if (!range(c, sizeof(*c), alignof(tqc_cache_cuda_config)) ||
      !range(static_cast<const void *>(out), sizeof(*out), alignof(void *)))
    return TQC_D_INVALID;
  if (overlap(c, sizeof(*c), static_cast<const void *>(out), sizeof(*out)))
    return TQC_D_OVERLAP;
  if (c->version != 1 || c->size != sizeof(*c))
    return TQC_D_VERSION;
  if (c->precision != TQC_D_F32 || c->representation > 1 ||
      (c->columns != 128 && c->columns != 256) ||
      (c->view_mask != 1 && c->view_mask != 2 && c->view_mask != 4 &&
       c->view_mask != 7) ||
      c->k_dialect < 1 || c->k_dialect > 5 || c->v_dialect < 1 ||
      c->v_dialect > 5 || c->beta != 0)
    return TQC_D_UNSUPPORTED;
  if (!c->kv_heads || !c->q_heads || c->q_heads % c->kv_heads || !c->capacity ||
      !c->max_queries || !c->max_append_rows ||
      c->max_append_rows > c->capacity || !identity(c->model) ||
      !identity(c->adapter) || !identity(c->position) || c->plane_count > 64)
    return TQC_D_INVALID;
  size_t plane_bytes = 0;
  if (!mul(c->plane_count, sizeof(tqc_cache_cuda_plane), plane_bytes))
    return TQC_D_OVERFLOW;
  if (plane_bytes &&
      !range(c->planes, plane_bytes, alignof(tqc_cache_cuda_plane)))
    return TQC_D_INVALID;
  if (overlap(c->planes, plane_bytes, static_cast<const void *>(out),
              sizeof(*out)))
    return TQC_D_OVERLAP;
  try {
    auto p = std::make_unique<tqc_cache_cuda_plan>();
    p->config = *c;
    p->config.planes = nullptr;
    p->config.plane_count = 0;
    double sum = 0;
    for (int i = 0; i < 3; ++i) {
      if (!std::isfinite(c->alpha[i]) || c->alpha[i] < 0 ||
          (!(c->view_mask & (1u << i)) && c->alpha[i] != 0))
        return TQC_D_INVALID;
      sum += c->alpha[i];
    }
    if (std::abs(sum - 1.0) > 1e-12)
      return TQC_D_INVALID;
    spin8_plane planes[64]{};
    for (size_t i = 0; i < c->plane_count; ++i)
      planes[i] = {c->planes[i].a, c->planes[i].b, c->planes[i].angle};
    spin8_plan *raw = nullptr;
    if (spin8_plan_create(planes, c->plane_count, &raw) != SPIN8_OK)
      return TQC_D_INVALID;
    std::unique_ptr<spin8_plan, decltype(&spin8_plan_free)> spin(
        raw, spin8_plan_free);
    spin8_audit_report audit{};
    if (spin8_plan_audit(raw, &audit) != SPIN8_OK)
      return TQC_D_INVALID;
    for (int r = 0; r < 3; ++r) {
      if (audit.views[r].max_error > 1e-12 ||
          audit.views[r].ef_normalized > 1e-12 ||
          std::abs(audit.views[r].log_abs_det) > 8e-12 ||
          audit.views[r].det_sign != 1)
        return TQC_D_INVALID;
      double b[64];
      if (spin8_plan_matrix(raw, static_cast<spin8_view>(r), b) != SPIN8_OK)
        return TQC_D_INVALID;
      hash(b, sizeof(b), p->layout.matrix_identity[r]);
      for (int i = 0; i < 64; ++i)
        p->b[r * 64 + i] = static_cast<float>(b[i]);
    }
    // Recover the pinned signed factors through the independent CPU F
    // operation.
    // This convention fixes D2[0]=+1; no quantisation participates.
    float input[128]{}, output[128]{};
    for (int j = 0; j < 128; ++j) {
      std::fill_n(input, 128, 0.0f);
      input[j] = 1;
      tqc_dialect_v1_request r{1,
                               sizeof(r),
                               1,
                               TQC_D_FORWARD,
                               TQC_D_CPU,
                               TQC_D_F32,
                               nullptr,
                               1,
                               128,
                               {input, sizeof(input), sizeof(input)},
                               {output, sizeof(output), sizeof(output)},
                               {}};
      tqc_dialect_v1_receipt receipt{1, sizeof(receipt), 0, 0, 0};
      if (tqc_dialect_v1_execute(&r, &receipt))
        return TQC_D_INVALID;
      p->s1[j] = output[0] < 0 ? -1.0f : 1.0f;
      if (j == 0)
        for (int i = 0; i < 128; ++i)
          p->s2[i] = (output[i] < 0 ? -1.0f : 1.0f) * p->s1[0];
    }
    auto &l = p->layout;
    l.version = 1;
    l.size = sizeof(l);
    l.host_plan_bytes = sizeof(*p);
    l.constant_bytes =
        sizeof(float) * (4 + 8 + 16 + 32 + 31 + 64 + 63 + 128 + 128);
    size_t n = 0, used = 0, kv = 0, q = 0;
    if (!mul(size_t(c->capacity), c->kv_heads, kv) ||
        !mul(size_t(c->max_queries), c->q_heads, q) || q > UINT32_MAX)
      return TQC_D_OVERFLOW;
    auto region = [&](size_t bytes, size_t &offset) {
      size_t next = 0;
      if (!add(n, 255, next))
        return false;
      offset = next & ~size_t(255);
      if (!add(offset, bytes, n) || !add(used, bytes, used))
        return false;
      return true;
    };
    if (!region(sizeof(tqc_cache_cuda_state), l.state) ||
        !region(sizeof(tqc_cache_cuda_command), l.command) ||
        !region(32, l.identity_offset) || !region(sizeof(p->b), l.matrices) ||
        !region(sizeof(tqc_cache_cuda_audit), l.audit))
      return TQC_D_OVERFLOW;
    size_t pos = 0;
    if (!mul(size_t(c->capacity), 8, pos) || !region(pos, l.positions))
      return TQC_D_OVERFLOW;
    l.metadata_bytes = used;
    l.k_row_bytes = size_t(c->columns / 128) *
                    (c->representation ? 512 : 18 + 16 * c->k_dialect);
    l.v_row_bytes = size_t(c->columns / 128) *
                    (c->representation ? 512 : 18 + 16 * c->v_dialect);
    for (int r = 0; r < 3; ++r)
      if (c->view_mask & (1u << r)) {
        if (!mul(kv, l.k_row_bytes, l.k_bytes[r]) ||
            !region(l.k_bytes[r], l.k[r]))
          return TQC_D_OVERFLOW;
      }
    if (!mul(kv, l.v_row_bytes, l.v_bytes) || !region(l.v_bytes, l.v) ||
        !add(n, 255, n))
      return TQC_D_OVERFLOW;
    l.total_bytes = n & ~size_t(255);
    l.alignment_bytes = l.total_bytes - used;
    /* Hash explicit fields, never pointers or unspecified padding. All-three
     * FP64 identities preserve Spin double-cover provenance even in single
     * mode. */
    spin8_sidecar_sha256_t h;
    spin8_sidecar_sha256_init(&h);
    auto update = [&](const void *v, size_t bytes) {
      spin8_sidecar_sha256_update(&h, static_cast<const unsigned char *>(v),
                                  bytes);
    };
    const char convention[] = "cache-v1/U=B*D2*H128/sqrt128*D1/V=F/"
                              "643956891dfdf51aecdbd05ca508dd21e21e8195";
    update(convention, sizeof(convention));
    uint64_t fields[] = {c->precision,   c->representation, c->k_dialect,
                         c->v_dialect,   c->view_mask,      c->columns,
                         c->kv_heads,    c->q_heads,        c->capacity,
                         c->max_queries, c->max_append_rows};
    update(fields, sizeof(fields));
    update(p->config.alpha, sizeof(c->alpha));
    update(c->model, 32);
    update(c->adapter, 32);
    update(c->position, 32);
    update(l.matrix_identity, sizeof(l.matrix_identity));
    update(p->b, sizeof(p->b));
    update(p->s1, sizeof(p->s1));
    update(p->s2, sizeof(p->s2));
    spin8_sidecar_sha256_final(&h, l.identity);
    *out = p.release();
    return TQC_D_OK;
  } catch (const std::bad_alloc &) {
    return TQC_D_CAPACITY;
  } catch (...) {
    return TQC_D_INVALID;
  }
}
extern "C" void tqc_cache_cuda_free(tqc_cache_cuda_plan *p) { delete p; }
extern "C" int tqc_cache_cuda_get_layout(const tqc_cache_cuda_plan *p,
                                         tqc_cache_cuda_layout *out) {
  using namespace tqc_cache_detail;
  if (!p || !range(out, sizeof(*out), alignof(tqc_cache_cuda_layout)))
    return TQC_D_INVALID;
  if (out->version != 1 || out->size != sizeof(*out))
    return TQC_D_VERSION;
  if (overlap(p, sizeof(*p), out, sizeof(*out)))
    return TQC_D_OVERLAP;
  *out = p->layout;
  return 0;
}
extern "C" int tqc_cache_cuda_snapshot_size(const tqc_cache_cuda_plan *p,
                                            size_t *out) {
  if (!p || !tqc_cache_detail::range(out, sizeof(*out), alignof(size_t)))
    return TQC_D_INVALID;
  if (tqc_cache_detail::overlap(p, sizeof(*p), out, sizeof(*out)))
    return TQC_D_OVERLAP;
  size_t n = 0;
  if (!tqc_cache_detail::add(p->layout.total_bytes, 80, n))
    return TQC_D_OVERFLOW;
  *out = n;
  return 0;
}
extern "C" int tqc_cache_cuda_query(tqc_cache_cuda_capabilities *out) {
  using namespace tqc_cache_detail;
  if (!range(out, sizeof(*out), alignof(tqc_cache_cuda_capabilities)))
    return TQC_D_INVALID;
  if (out->version != 1 || out->size != sizeof(*out))
    return TQC_D_VERSION;
  *out = {1, sizeof(*out), 0, 0, 0, 0};
  return 0;
}
