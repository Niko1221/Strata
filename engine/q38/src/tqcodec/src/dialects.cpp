#include "tqcodec/dialects.h"
#include <cassert>
#include <cfenv>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <initializer_list>
#include <limits>
#if defined(_M_X64) || defined(__SSE__)
#include <xmmintrin.h>
#endif

namespace {
static_assert(sizeof(float) == 4 && std::numeric_limits<float>::is_iec559,
              "dialect v1 requires IEEE binary32");
float from_bits(uint32_t u) {
  float f;
  std::memcpy(&f, &u, 4);
  return f;
}
uint32_t bits(float f) {
  uint32_t u;
  std::memcpy(&u, &f, 4);
  return u;
}
// ggml-impl.h software conversion, MIT attribution in DIALECT_LICENSE.txt.
uint16_t half_encode(float f) {
  // Avoid an intermediate FP32 overflow for finite values outside FP16 range.
  // The pinned RNE converter returns this same infinity representation.
  if (std::fabs(f) >= 65520.0f)
    return uint16_t((bits(f) >> 16 & 0x8000u) | 0x7c00u);
  float base = (std::fabs(f) * from_bits(0x77800000)) * from_bits(0x08800000);
  const uint32_t w = bits(f), shl = w + w, sign = w & 0x80000000u;
  uint32_t bias = shl & 0xff000000u;
  if (bias < 0x71000000u)
    bias = 0x71000000u;
  base = from_bits((bias >> 1) + 0x07800000u) + base;
  const uint32_t v = bits(base);
  return static_cast<uint16_t>(
      (sign >> 16) |
      (shl > 0xff000000u ? 0x7e00u : ((v >> 13) & 0x7c00u) + (v & 0xfffu)));
}
float half_decode(uint16_t h) {
  const uint32_t w = uint32_t(h) << 16, twice = w + w;
  const float normal =
      from_bits((twice >> 4) + (0xe0u << 23)) * from_bits(0x07800000);
  const float sub = from_bits((twice >> 17) | (126u << 23)) - 0.5f;
  return from_bits((w & 0x80000000u) | bits(twice < (1u << 27) ? sub : normal));
}
#include "dialect_tables.inc"

constexpr const char *tom = "643956891dfdf51aecdbd05ca508dd21e21e8195";
constexpr const char *prism = "73bfe6cc485554822f80c473a3c0c4f25e80f41b";
constexpr const char *native = "3cf03257f219afbe7334045ff7c6a06ac68c627d";
constexpr const char *ledger =
    "normalize; F=D2*(H128/sqrt128)*D1; optional B after F; decode=stored "
    "domain; inverse U: B^T then D2,H128/sqrt128,D1";
struct Format {
  uint32_t n, bytes, metadata, offset, kind;
  const char *id, *sha, *coords, *packing;
};
const Format formats[] = {
    {128, 34, 2, 0, TQC_D_KV, "thetom-64395689/turbo2-128-f16norm/v1", tom,
     ledger, "LSB2; Lloyd-Max2; midpoint ties upper; corrected L2 FP16 RNE"},
    {128, 50, 2, 0, TQC_D_KV, "thetom-64395689/turbo3-128-f16norm/v1", tom,
     ledger,
     "low2[32],high1[16]; Lloyd-Max3; ties upper; corrected L2 FP16 RNE"},
    {128, 66, 2, 0, TQC_D_KV, "thetom-64395689/turbo4-128-4bit/v1", tom, ledger,
     "nibbles[64]; Lloyd-Max4; ties upper; corrected L2 FP16 RNE; no QJL"},
    {128, 82, 2, 0, TQC_D_KV, "thetom-64395689/turbo5-128-signmag/v1", tom,
     ledger,
     "magnitude nibbles[64],negative bits[16]; Lloyd-Max5; corrected L2 FP16 "
     "RNE"},
    {128, 98, 2, 0, TQC_D_KV, "thetom-64395689/turbo6-128-planes/v1", tom,
     ledger,
     "low4[64],high2[32]; Lloyd-Max6; ties upper; corrected L2 FP16 RNE"},
    {32, 16, 4, 0, TQC_D_WEIGHT, "thetom-64395689/tq3-1s-32/v1", tom,
     "encode H32/sqrt32*D; decode D*H32/sqrt32 (original coordinates); "
     "unrelated to KV F",
     "two FP16 RNE scales; 9-point search/6 refinements; dense3 groups8"},
    {32, 20, 4, 0, TQC_D_WEIGHT, "thetom-64395689/tq4-1s-32/v1", tom,
     "encode H32/sqrt32*D; decode D*H32/sqrt32 (original coordinates); "
     "unrelated to KV F",
     "two FP16 RNE scales; 9-point search/6 refinements; nibbles"},
    {128, 34, 2, 0, TQC_D_WEIGHT, "prism-73bfe6cc/pq2-0-128/v1", prism,
     "no transform; decode input coordinates (may be externally weight-folded)",
     "FP16 RNE maxabs; LSB2; round away from zero; code minus1"},
    {128, 28, 2, 26, TQC_D_WEIGHT, "prism-73bfe6cc/ptq1-0-128/v1", prism,
     "no transform; decode input coordinates (may be externally weight-folded)",
     "qs24 staged16,8; qh2; FP16 RNE maxabs at26; ternary round away"},
    {64, 18, 2, 0, TQC_D_LINEAR, "prism-73bfe6cc/q2-0-64/v1", prism,
     "no transform; decode input coordinates",
     "FP16 RNE maxabs; LSB2; round away from zero; code minus1"},
    {64, 18, 2, 0, TQC_D_LINEAR, "strata-3cf03257/q2-0-64/v1", native,
     "no transform; decode input coordinates",
     "FP16 RNE maxabs; LSB2; round away from zero; code minus1"}};
const Format *format(uint32_t id) {
  return id >= 1 && id <= 11 ? &formats[id - 1] : nullptr;
}
bool mul(size_t a, size_t b, size_t &v) {
  if (b && a > SIZE_MAX / b)
    return false;
  v = a * b;
  return true;
}
struct Range {
  uintptr_t begin, end;
};
int range(const void *p, size_t n, Range &r) {
  r.begin = reinterpret_cast<uintptr_t>(p);
  if (n > UINTPTR_MAX - r.begin)
    return TQC_D_OVERFLOW;
  if (!p && n)
    return TQC_D_INVALID;
  r.end = r.begin + n;
  return TQC_D_OK;
}
bool overlaps(Range a, Range b) { return a.begin < b.end && b.begin < a.end; }
int span(tqc_dialect_v1_span s, size_t rows, size_t width, size_t align,
         Range &r) {
  size_t last = 0;
  if (!mul(rows - 1, s.row_stride_bytes, last) || last > SIZE_MAX - width)
    return TQC_D_OVERFLOW;
  if (s.row_stride_bytes < width || s.capacity_bytes < last + width)
    return TQC_D_CAPACITY;
  int rc = range(s.data, last + width, r);
  if (rc)
    return rc;
  // Also reject a declared capacity which wraps the address space.
  Range full{};
  rc = range(s.data, s.capacity_bytes, full);
  if (rc)
    return rc;
  if (r.begin % align || s.row_stride_bytes % align)
    return TQC_D_ALIGNMENT;
  return TQC_D_OK;
}
uint16_t load16(const uint8_t *p) {
  return uint16_t(p[0]) | uint16_t(uint16_t(p[1]) << 8);
}
void store16(uint8_t *p, uint16_t v) {
  p[0] = uint8_t(v);
  p[1] = uint8_t(v >> 8);
}
void fwht(float *x, bool inverse) {
  const float *a = inverse ? turbo_cpu_s2 : turbo_cpu_s1;
  const float *b = inverse ? turbo_cpu_s1 : turbo_cpu_s2;
  for (int i = 0; i < 128; ++i)
    x[i] *= a[i];
  for (int h = 1; h < 128; h *= 2)
    for (int i = 0; i < 128; i += h * 2)
      for (int j = i; j < i + h; ++j) {
        const float u = x[j], v = x[j + h];
        x[j] = u + v;
        x[j + h] = u - v;
      }
  for (int i = 0; i < 128; ++i)
    x[i] *= 0.08838834764831845f * b[i];
}
float prepare(uint32_t id, const float *x, float *out) {
  float sum = 0;
  for (int i = 0; i < 128; ++i)
    sum += x[i] * x[i];
  const float norm = std::sqrt(sum), inv = norm > 1e-10f ? 1.0f / norm : 0.0f;
  for (int i = 0; i < 128; ++i)
    out[i] = (id <= 2 || norm > 1e-10f) ? x[i] * inv : 0.0f;
  fwht(out, false);
  return norm;
}
const float *centroids(uint32_t id) {
  switch (id) {
  case 1:
    return CENTROIDS_2BIT;
  case 2:
    return CENTROIDS_3BIT;
  case 3:
    return CENTROIDS_4BIT;
  case 4:
    return TURBO5_CENTROIDS;
  default:
    return TURBO6_CENTROIDS;
  }
}
int nearest(uint32_t id, float v) {
  switch (id) {
  case 1:
    return nearest_centroid_2bit(v);
  case 2:
    return nearest_centroid_3bit(v);
  case 3:
    return nearest_centroid_4bit(v);
  case 4:
    return nearest_centroid_5bit(v);
  default:
    return nearest_centroid_6bit(v);
  }
}
void turbo_pack(uint32_t id, const float *x, float norm, uint8_t *p) {
  std::memset(p, 0, formats[id - 1].bytes);
  const float *c = centroids(id);
  float sum = 0;
  for (int i = 0; i < 128; ++i) {
    int idx = nearest(id, x[i]);
    sum += c[idx] * c[idx];
    if (id <= 2) {
      p[2 + i / 4] |= uint8_t((idx & 3) << ((i % 4) * 2));
      if (id == 2)
        p[34 + i / 8] |= uint8_t((idx >> 2) << (i % 8));
    } else {
      int lo = idx & 15;
      if (id == 4) {
        const int neg = idx < 16;
        lo = neg ? 15 - idx : idx - 16;
        p[66 + i / 8] |= uint8_t(neg << (i % 8));
      }
      if (id == 5)
        p[66 + i / 4] |= uint8_t((idx >> 4) << ((i % 4) * 2));
      p[2 + i / 2] |= uint8_t(lo << ((i % 2) * 4));
    }
  }
  const float recon = std::sqrt(sum);
  store16(p, half_encode(recon > 1e-10f ? norm / recon : norm));
}
void turbo_decode(uint32_t id, const uint8_t *p, float *x) {
  const float norm = half_decode(load16(p)), *c = centroids(id);
  for (int i = 0; i < 128; ++i) {
    int idx;
    if (id <= 2) {
      idx = (p[2 + i / 4] >> ((i % 4) * 2)) & 3;
      if (id == 2)
        idx |= ((p[34 + i / 8] >> (i % 8)) & 1) << 2;
    } else {
      idx = (p[2 + i / 2] >> ((i % 2) * 4)) & 15;
      if (id == 4)
        idx = ((p[66 + i / 8] >> (i % 8)) & 1) ? 15 - idx : 16 + idx;
      if (id == 5)
        idx |= ((p[66 + i / 4] >> ((i % 4) * 2)) & 3) << 4;
    }
    x[i] = c[idx] * norm;
  }
}
void weight_encode(uint32_t id, const float *x, uint8_t *p) {
  if (id == 6) {
    block_tq3_1s b{};
    quantize_row_tq3_1s_ref(x, &b, 32);
    store16(p, b.d0);
    store16(p + 2, b.d1);
    std::memcpy(p + 4, b.qs, 12);
  } else {
    block_tq4_1s b{};
    quantize_row_tq4_1s_ref(x, &b, 32);
    store16(p, b.d0);
    store16(p + 2, b.d1);
    std::memcpy(p + 4, b.qs, 16);
  }
}
void weight_decode(uint32_t id, const uint8_t *p, float *x) {
  if (id == 6) {
    block_tq3_1s b{};
    b.d0 = load16(p);
    b.d1 = load16(p + 2);
    std::memcpy(b.qs, p + 4, 12);
    dequantize_row_tq3_1s(&b, x, 32);
  } else {
    block_tq4_1s b{};
    b.d0 = load16(p);
    b.d1 = load16(p + 2);
    std::memcpy(b.qs, p + 4, 16);
    dequantize_row_tq4_1s(&b, x, 32);
  }
}
int linear_encode(uint32_t id, const float *x, uint8_t *p) {
  const auto &f = formats[id - 1];
  float scale = 0;
  for (uint32_t i = 0; i < f.n; ++i)
    scale = std::fmax(scale, std::fabs(x[i]));
  const float inv = scale ? 1.0f / scale : 0;
  if (!std::isfinite(inv))
    return TQC_D_RANGE; // upstream float->int would be undefined
  std::memset(p, 0, f.bytes);
  store16(p + f.offset, half_encode(scale));
  if (id != 9) {
    for (uint32_t i = 0; i < f.n; ++i) {
      const int q = int(std::round(x[i] * inv)) + 1;
      p[2 + i / 4] |= uint8_t(q << ((i % 4) * 2));
    }
  } else {
    size_t pos = 0, off = 0;
    for (size_t count : {size_t(16), size_t(8)}) {
      for (size_t m = 0; m < count; ++m) {
        unsigned q = 0;
        for (size_t n = 0; n < 5; ++n)
          q = q * 3 + unsigned(std::lround(x[off + m + n * count] * inv) + 1);
        p[pos + m] = uint8_t((q * 256 + 242) / 243);
      }
      pos += count;
      off += 5 * count;
    }
    for (size_t h = 0; h < 2; ++h) {
      unsigned q = 0;
      for (size_t m = 0; m < 4; ++m)
        q = q * 3 + unsigned(std::lround(x[120 + h + m * 2] * inv) + 1);
      p[24 + h] = uint8_t((q * 3 * 256 + 242) / 243);
    }
  }
  return TQC_D_OK;
}
void linear_decode(uint32_t id, const uint8_t *p, float *x) {
  const auto &f = formats[id - 1];
  const float scale = half_decode(load16(p + f.offset));
  if (id != 9)
    for (uint32_t i = 0; i < f.n; ++i)
      x[i] = float(int((p[2 + i / 4] >> ((i % 4) * 2)) & 3) - 1) * scale;
  else {
    unsigned power = 1;
    size_t pos = 0, off = 0;
    for (size_t count : {size_t(16), size_t(8)}) {
      power = 1;
      for (size_t n = 0; n < 5; ++n, power *= 3)
        for (size_t m = 0; m < count; ++m)
          x[off + n * count + m] =
              float(int((unsigned(uint8_t(p[pos + m] * power)) * 3) >> 8) - 1) *
              scale;
      pos += count;
      off += count * 5;
    }
    power = 1;
    for (size_t n = 0; n < 4; ++n, power *= 3)
      for (size_t h = 0; h < 2; ++h)
        x[120 + n * 2 + h] =
            float(int((unsigned(uint8_t(p[24 + h] * power)) * 3) >> 8) - 1) *
            scale;
  }
}
bool fp_contract() {
  if (std::fegetround() != FE_TONEAREST)
    return false;
#if defined(_M_X64) || defined(__SSE__)
  if ((_mm_getcsr() & ((1u << 15) | (1u << 6))) != 0 ||
      (_mm_getcsr() & 0x1f80u) != 0x1f80u)
    return false;
#endif
  return true;
}
} // namespace

extern "C" int tqc_dialect_v1_query(uint32_t id, tqc_dialect_v1_info *out) {
  Range r{};
  int rc = range(out, sizeof(*out), r);
  if (rc)
    return rc;
  if (r.begin % alignof(tqc_dialect_v1_info))
    return TQC_D_ALIGNMENT;
  if (out->version != 1 || out->size != sizeof(*out))
    return TQC_D_VERSION;
  const auto *f = format(id);
  if (!f)
    return TQC_D_UNSUPPORTED;
  *out = {1,           sizeof(*out), id,        f->kind,   f->n, f->bytes,
          f->metadata, f->offset,    1,         0,         0,    0,
          f->id,       f->sha,       f->coords, f->packing};
  return TQC_D_OK;
}
extern "C" int tqc_dialect_v1_validate_layout(const tqc_dialect_v1_layout *p) {
  Range r{};
  int rc = range(p, sizeof(*p), r);
  if (rc)
    return rc;
  if (r.begin % alignof(tqc_dialect_v1_layout))
    return TQC_D_ALIGNMENT;
  if (p->version != 1 || p->size != sizeof(*p))
    return TQC_D_VERSION;
  const auto *f = format(p->dialect);
  if (!f)
    return TQC_D_UNSUPPORTED;
  if (!p->identity || !p->identity_capacity)
    return TQC_D_AMBIGUOUS;
  rc = range(p->identity, p->identity_capacity, r);
  if (rc)
    return rc;
  const size_t n = std::strlen(f->id) + 1;
  if (p->identity_capacity < n || std::memcmp(p->identity, f->id, n) != 0)
    return TQC_D_AMBIGUOUS;
  if (p->block_elements != f->n || p->block_bytes != f->bytes)
    return TQC_D_AMBIGUOUS;
  return TQC_D_OK;
}
extern "C" int tqc_dialect_v1_execute(const tqc_dialect_v1_request *p,
                                      tqc_dialect_v1_receipt *receipt) {
  Range req{}, rec{};
  int rc = range(p, sizeof(*p), req);
  if (rc)
    return rc;
  if (req.begin % alignof(tqc_dialect_v1_request))
    return TQC_D_ALIGNMENT;
  if (p->version != 1 || p->size != sizeof(*p))
    return TQC_D_VERSION;
  rc = range(receipt, sizeof(*receipt), rec);
  if (rc)
    return rc;
  if (rec.begin % alignof(tqc_dialect_v1_receipt))
    return TQC_D_ALIGNMENT;
  if (receipt->version != 1 || receipt->size != sizeof(*receipt))
    return TQC_D_VERSION;
  const auto *f = format(p->dialect);
  if (!f)
    return TQC_D_UNSUPPORTED;
  const uint32_t op = p->operation;
  if (p->device != TQC_D_CPU || p->precision != TQC_D_F32 || p->stream ||
      !fp_contract())
    return TQC_D_UNSUPPORTED;
  if (op < 1 || op > 6 || (op > 2 && p->dialect > 5))
    return TQC_D_UNSUPPORTED;
  if (!p->rows || !p->columns || p->columns % f->n)
    return TQC_D_INVALID;
  const size_t blocks = p->columns / f->n;
  size_t rowbytes = 0, fbytes = 0, nbytes = 0, totalblocks = 0, payload = 0;
  if (!mul(blocks, f->bytes, rowbytes) || !mul(p->columns, 4, fbytes) ||
      !mul(blocks, 4, nbytes) || !mul(p->rows, blocks, totalblocks) ||
      !mul(totalblocks, f->bytes, payload))
    return TQC_D_OVERFLOW;
  const bool enc = op == TQC_D_ENCODE || op == TQC_D_PACK,
             dec = op == TQC_D_DECODE;
  Range src{}, dst{}, norms{};
  rc = span(p->src, p->rows, dec ? rowbytes : fbytes, dec ? 1 : 4, src);
  if (rc)
    return rc;
  rc = span(p->dst, p->rows, enc ? rowbytes : fbytes, enc ? 1 : 4, dst);
  if (rc)
    return rc;
  const bool hasnorm = op == TQC_D_PREPARE || op == TQC_D_PACK;
  if (hasnorm) {
    rc = span(p->norms, p->rows, nbytes, 4, norms);
    if (rc)
      return rc;
  } else if (p->norms.data || p->norms.capacity_bytes ||
             p->norms.row_stride_bytes)
    return TQC_D_INVALID;
  if (overlaps(src, dst) || overlaps(src, rec) || overlaps(dst, rec) ||
      overlaps(req, rec) || overlaps(dst, req) ||
      (hasnorm && (overlaps(norms, src) || overlaps(norms, dst) ||
                   overlaps(norms, rec) || overlaps(norms, req))))
    return TQC_D_OVERLAP;
  // Two passes: validate every block (including encoded scales) before the
  // first write. Fixed scratch, no heap and no exception-producing operations.
  // O(rows*columns).
  for (int pass = 0; pass < 2; ++pass)
    for (size_t row = 0; row < p->rows; ++row)
      for (size_t block = 0; block < blocks; ++block) {
        float x[128]{}, y[128]{};
        uint8_t packed[98]{};
        float norm = 0;
        const auto *s = static_cast<const uint8_t *>(p->src.data) +
                        row * p->src.row_stride_bytes +
                        block * (dec ? f->bytes : f->n * 4);
        auto *d = static_cast<uint8_t *>(p->dst.data) +
                  row * p->dst.row_stride_bytes +
                  block * (enc ? f->bytes : f->n * 4);
        uint8_t *np = hasnorm ? static_cast<uint8_t *>(p->norms.data) +
                                    row * p->norms.row_stride_bytes + block * 4
                              : nullptr;
        if (!dec) {
          std::memcpy(x, s, f->n * 4);
          for (uint32_t i = 0; i < f->n; ++i) {
            if (!std::isfinite(x[i]))
              return TQC_D_NONFINITE;
            if (std::fabs(x[i]) > 65504.0f)
              return TQC_D_RANGE;
          }
        } else {
          std::memcpy(packed, s, f->bytes);
          for (uint32_t i = 0; i < f->metadata; i += 2)
            if (!std::isfinite(half_decode(load16(packed + f->offset + i))))
              return TQC_D_NONFINITE;
        }
        if (op == TQC_D_PACK) {
          std::memcpy(&norm, np, 4);
          if (!std::isfinite(norm))
            return TQC_D_NONFINITE;
          if (norm < 0)
            return TQC_D_RANGE;
          turbo_pack(p->dialect, x, norm, packed);
        } else if (op == TQC_D_PREPARE)
          norm = prepare(p->dialect, x, y);
        else if (op == TQC_D_FORWARD || op == TQC_D_INVERSE) {
          std::memcpy(y, x, 512);
          fwht(y, op == TQC_D_INVERSE);
        } else if (enc) {
          if (p->dialect <= 5) {
            norm = prepare(p->dialect, x, y);
            turbo_pack(p->dialect, y, norm, packed);
          } else if (p->dialect <= 7)
            weight_encode(p->dialect, x, packed);
          else {
            rc = linear_encode(p->dialect, x, packed);
            if (rc)
              return rc;
          }
        } else {
          if (p->dialect <= 5)
            turbo_decode(p->dialect, packed, y);
          else if (p->dialect <= 7)
            weight_decode(p->dialect, packed, y);
          else
            linear_decode(p->dialect, packed, y);
        }
        if (enc)
          for (uint32_t i = 0; i < f->metadata; i += 2)
            if (!std::isfinite(half_decode(load16(packed + f->offset + i))))
              return TQC_D_RANGE;
        if (!enc)
          for (uint32_t i = 0; i < f->n; ++i)
            if (!std::isfinite(y[i]))
              return TQC_D_RANGE;
        if (pass) {
          std::memcpy(d,
                      enc ? static_cast<const void *>(packed)
                          : static_cast<const void *>(y),
                      enc ? f->bytes : f->n * 4);
          if (op == TQC_D_PREPARE)
            std::memcpy(np, &norm, 4);
        }
      }
  *receipt = {1, sizeof(*receipt), p->rows, totalblocks, payload};
  return TQC_D_OK;
}
