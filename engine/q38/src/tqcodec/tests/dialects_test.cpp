#include "tqcodec/dialects.h"
#include <array>
#include <cfenv>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <limits>
#include <vector>

static int failures = 0;
#define CHECK(x)                                                               \
  do {                                                                         \
    if (!(x)) {                                                                \
      std::fprintf(stderr, "line %d: %s\n", __LINE__, #x);                     \
      ++failures;                                                              \
    }                                                                          \
  } while (0)
static tqc_dialect_v1_receipt receipt() {
  return {1, sizeof(tqc_dialect_v1_receipt), 0, 0, 0};
}
static tqc_dialect_v1_request request(unsigned id, unsigned op, void *src,
                                      size_t sc, void *dst, size_t dc,
                                      size_t cols = 128, size_t rows = 1) {
  return {1,
          sizeof(tqc_dialect_v1_request),
          id,
          op,
          TQC_D_CPU,
          TQC_D_F32,
          nullptr,
          rows,
          cols,
          {src, sc, sc / rows},
          {dst, dc, dc / rows},
          {nullptr, 0, 0}};
}
static void boundaries() {
  alignas(16) std::array<float, 256> x{}, out{};
  std::array<uint8_t, 196> p{};
  auto r =
      request(5, TQC_D_ENCODE, x.data(), sizeof(x), p.data(), p.size(), 256);
  auto z = receipt();
  CHECK(tqc_dialect_v1_execute(&r, &z) == 0);
  CHECK(z.payload_bytes == 196 && z.blocks == 2);
  auto bad = [&](tqc_dialect_v1_request q, int expected) {
    p.fill(0xa5);
    const auto before = p;
    auto rr = receipt();
    rr.rows = 99;
    const auto rb = rr;
    CHECK(tqc_dialect_v1_execute(&q, &rr) == expected);
    CHECK(p == before);
    CHECK(std::memcmp(&rr, &rb, sizeof(rr)) == 0);
  };
  auto q = r;
  q.version = 2;
  bad(q, TQC_D_VERSION);
  q = r;
  q.size--;
  bad(q, TQC_D_VERSION);
  q = r;
  q.dialect = 0;
  bad(q, TQC_D_UNSUPPORTED);
  q = r;
  q.dialect = 42;
  bad(q, TQC_D_UNSUPPORTED);
  q = r;
  q.device = TQC_D_CUDA;
  q.src.data = reinterpret_cast<void *>(4096);
  bad(q, TQC_D_UNSUPPORTED);
  q = r;
  q.precision = 2;
  bad(q, TQC_D_UNSUPPORTED);
  q = r;
  q.stream = x.data();
  bad(q, TQC_D_UNSUPPORTED);
  q = r;
  q.rows = 0;
  bad(q, TQC_D_INVALID);
  q = r;
  q.columns = 127;
  bad(q, TQC_D_INVALID);
  q = r;
  q.src.capacity_bytes--;
  bad(q, TQC_D_CAPACITY);
  q = r;
  q.dst.capacity_bytes--;
  bad(q, TQC_D_CAPACITY);
  q = r;
  q.src.data = reinterpret_cast<uint8_t *>(x.data()) + 1;
  bad(q, TQC_D_ALIGNMENT);
  q = r;
  q.dst.data = x.data();
  bad(q, TQC_D_OVERLAP);
  q = r;
  q.columns = SIZE_MAX - 127;
  bad(q, TQC_D_OVERFLOW);
  q = r;
  q.rows = SIZE_MAX;
  q.src.row_stride_bytes = SIZE_MAX;
  bad(q, TQC_D_OVERFLOW);
  q = r;
  q.src.capacity_bytes = SIZE_MAX;
  bad(q, TQC_D_OVERFLOW);
  // Combined alias plus arithmetic overflow: fail before reading or writing
  // either.
  q = r;
  q.dst.data = x.data();
  q.src.data = reinterpret_cast<void *>(UINTPTR_MAX - 7);
  bad(q, TQC_D_OVERFLOW);
  q = r;
  q.dst.data = &q;
  CHECK(tqc_dialect_v1_execute(&q, &z) == TQC_D_OVERLAP);
  x[255] = std::numeric_limits<float>::quiet_NaN();
  bad(r, TQC_D_NONFINITE);
  x[255] = std::numeric_limits<float>::infinity();
  bad(r, TQC_D_NONFINITE);
  x[255] = std::numeric_limits<float>::max();
  bad(r, TQC_D_RANGE);
  x[255] = 0;
  x.fill(65504);
  bad(r, TQC_D_RANGE);
  x.fill(0);
  const int rounding = std::fegetround();
  CHECK(std::fesetround(FE_DOWNWARD) == 0);
  bad(r, TQC_D_UNSUPPORTED);
  CHECK(std::fesetround(rounding) == 0);
  q = r;
  q.dst.data = &z;
  q.dst.capacity_bytes = 196;
  auto oldz = z;
  CHECK(tqc_dialect_v1_execute(&q, &z) == TQC_D_OVERLAP);
  CHECK(std::memcmp(&z, &oldz, sizeof(z)) == 0);
  std::array<float, 2> norms{1, 1};
  std::array<float, 256> rotated{};
  q = request(5, TQC_D_PREPARE, x.data(), sizeof(x), rotated.data(),
              sizeof(rotated), 256);
  q.norms = {norms.data(), sizeof(norms), sizeof(norms)};
  CHECK(tqc_dialect_v1_execute(&q, &z) == 0);
  q.norms.capacity_bytes = 4;
  CHECK(tqc_dialect_v1_execute(&q, &z) == TQC_D_CAPACITY);
  q.norms = {x.data(), 8, 8};
  CHECK(tqc_dialect_v1_execute(&q, &z) == TQC_D_OVERLAP);
  q.norms = {reinterpret_cast<uint8_t *>(norms.data()) + 1, 8, 8};
  CHECK(tqc_dialect_v1_execute(&q, &z) == TQC_D_ALIGNMENT);
  q = request(5, TQC_D_PACK, x.data(), sizeof(x), p.data(), p.size(), 256);
  q.norms = {norms.data(), 8, 8};
  norms[1] = std::numeric_limits<float>::quiet_NaN();
  bad(q, TQC_D_NONFINITE);
  norms[1] = -1;
  bad(q, TQC_D_RANGE);
  norms[1] = 1;
  q.norms.data = reinterpret_cast<void *>(UINTPTR_MAX - 3);
  bad(q, TQC_D_OVERFLOW);
  r = request(5, TQC_D_DECODE, p.data(), p.size(), out.data(), sizeof(out),
              256);
  p.fill(0);
  p[99] = 0x7c;
  out.fill(123);
  auto before = out;
  CHECK(tqc_dialect_v1_execute(&r, &z) == TQC_D_NONFINITE);
  CHECK(out == before);
  for (unsigned id = 1; id <= 11; ++id) {
    tqc_dialect_v1_info info{};
    info.version = 1;
    info.size = sizeof(info);
    CHECK(tqc_dialect_v1_query(id, &info) == 0);
    CHECK(info.cpu_f32 == 1 && info.cuda == 0 && info.scratch_bytes <= 4096);
    tqc_dialect_v1_layout layout{1,
                                 sizeof(layout),
                                 id,
                                 info.block_elements,
                                 info.block_bytes,
                                 info.identity,
                                 std::strlen(info.identity) + 1};
    CHECK(tqc_dialect_v1_validate_layout(&layout) == 0);
    layout.identity = nullptr;
    CHECK(tqc_dialect_v1_validate_layout(&layout) == TQC_D_AMBIGUOUS);
    layout.identity = info.identity;
    layout.block_bytes++;
    CHECK(tqc_dialect_v1_validate_layout(&layout) == TQC_D_AMBIGUOUS);
    layout.block_bytes = info.block_bytes;
    layout.identity_capacity--;
    CHECK(tqc_dialect_v1_validate_layout(&layout) == TQC_D_AMBIGUOUS);
  }
  r = request(10, TQC_D_ENCODE, x.data(), 256, p.data(), 18, 64);
  x[0] = std::numeric_limits<float>::denorm_min();
  CHECK(tqc_dialect_v1_execute(&r, &z) == TQC_D_RANGE);
}

// Independent dense FP64 Walsh matrix, not the implementation's butterflies.
static const int s1[128] = {
    -1, 1,  1,  -1, -1, 1,  -1, 1,  -1, -1, 1,  1,  1,  1,  1,  1,  1,  -1, 1,
    -1, 1,  -1, -1, 1,  1,  1,  -1, 1,  1,  -1, -1, -1, -1, 1,  1,  -1, 1,  1,
    -1, 1,  -1, 1,  1,  -1, -1, 1,  -1, 1,  1,  1,  1,  -1, -1, -1, -1, -1, 1,
    -1, 1,  1,  1,  1,  -1, 1,  -1, -1, 1,  -1, -1, -1, 1,  -1, -1, -1, 1,  -1,
    -1, -1, 1,  1,  1,  -1, -1, 1,  1,  1,  -1, -1, 1,  1,  -1, 1,  1,  -1, 1,
    -1, -1, 1,  1,  -1, 1,  -1, 1,  -1, 1,  1,  1,  1,  -1, 1,  -1, 1,  1,  -1,
    1,  1,  -1, -1, -1, -1, -1, 1,  1,  -1, 1,  1,  -1, 1};
static const int s2[128] = {
    1,  1,  1,  1,  -1, 1,  1,  -1, 1,  -1, -1, -1, 1,  -1, -1, -1, 1,  1,  -1,
    -1, 1,  -1, 1,  -1, 1,  -1, -1, 1,  -1, 1,  1,  1,  1,  1,  -1, -1, -1, 1,
    -1, -1, -1, -1, -1, -1, 1,  1,  1,  -1, 1,  -1, 1,  1,  1,  -1, -1, 1,  -1,
    -1, -1, -1, -1, -1, 1,  1,  1,  -1, 1,  -1, -1, -1, -1, 1,  -1, 1,  -1, 1,
    -1, -1, 1,  1,  -1, 1,  -1, 1,  1,  -1, 1,  -1, -1, -1, -1, 1,  -1, -1, 1,
    -1, 1,  -1, 1,  1,  1,  -1, -1, 1,  -1, 1,  -1, 1,  1,  -1, -1, 1,  -1, 1,
    -1, 1,  1,  -1, 1,  -1, 1,  -1, -1, -1, -1, -1, 1,  -1};
static double matrix(int i, int j) {
  unsigned v = unsigned(i & j), parity = 0;
  while (v) {
    parity ^= v & 1;
    v >>= 1;
  }
  return s2[i] * s1[j] * (parity ? -1 : 1) / std::sqrt(128.0);
}
static void spin(double *x, bool inverse) {
  // Proper orthogonal 8-lane plane rotation, repeated across the 128-lane
  // group.
  const double c = std::cos(.37), s = (inverse ? -1 : 1) * std::sin(.37);
  for (int i = 0; i < 128; i += 8) {
    double a = x[i], b = x[i + 1];
    x[i] = c * a - s * b;
    x[i + 1] = s * a + c * b;
  }
}
static void factors() {
  std::array<float, 128> x{}, y{}, back{};
  auto z = receipt();
  double max_basis = 0, max_roundtrip = 0;
  for (int k = 0; k < 128; ++k) {
    x.fill(0);
    x[k] = 1;
    auto r = request(1, TQC_D_FORWARD, x.data(), 512, y.data(), 512);
    CHECK(tqc_dialect_v1_execute(&r, &z) == 0);
    for (int i = 0; i < 128; ++i)
      max_basis = std::fmax(max_basis, std::fabs(y[i] - matrix(i, k)));
    r = request(1, TQC_D_INVERSE, y.data(), 512, back.data(), 512);
    CHECK(tqc_dialect_v1_execute(&r, &z) == 0);
    for (int i = 0; i < 128; ++i)
      max_roundtrip =
          std::fmax(max_roundtrip, std::fabs(double(back[i] - x[i])));
  }
  CHECK(max_basis < 1e-8);
  CHECK(max_roundtrip < 2e-7);
  double a[128], fa[128]{}, ba[128], fb[128]{}, u[128], restored[128]{};
  for (int i = 0; i < 128; ++i) {
    a[i] = std::sin(i * .17) + i * .01;
    ba[i] = a[i];
  }
  for (int i = 0; i < 128; ++i)
    for (int j = 0; j < 128; ++j)
      fa[i] += matrix(i, j) * a[j];
  std::memcpy(u, fa, sizeof(u));
  spin(u, false);
  spin(ba, false);
  for (int i = 0; i < 128; ++i)
    for (int j = 0; j < 128; ++j)
      fb[i] += matrix(i, j) * ba[j];
  double order_error = 0, norm_a = 0, norm_u = 0;
  for (int i = 0; i < 128; ++i) {
    order_error = std::fmax(order_error, std::fabs(u[i] - fb[i]));
    norm_a += a[i] * a[i];
    norm_u += u[i] * u[i];
  }
  CHECK(order_error > .1);
  CHECK(std::fabs(norm_a - norm_u) < 1e-10);
  double wrong[128]{};
  for (int i = 0; i < 128; ++i)
    for (int j = 0; j < 128; ++j)
      wrong[i] += matrix(j, i) * u[j];
  spin(wrong, true);
  spin(u, true);
  for (int i = 0; i < 128; ++i)
    for (int j = 0; j < 128; ++j)
      restored[i] += matrix(j, i) * u[j];
  double inverse_error = 0, wrong_inverse = 0;
  for (int i = 0; i < 128; ++i) {
    inverse_error = std::fmax(inverse_error, std::fabs(restored[i] - a[i]));
    wrong_inverse = std::fmax(wrong_inverse, std::fabs(wrong[i] - a[i]));
  }
  CHECK(inverse_error < 1e-12);
  CHECK(wrong_inverse > .1);
  // Missing normalization and sign mutation are distinguishable from codec
  // loss.
  CHECK(std::fabs(matrix(0, 0) * std::sqrt(128.0) - matrix(0, 0)) > .5);
  CHECK(std::fabs(-matrix(0, 0) - matrix(0, 0)) > .1);
  for (unsigned id = 1; id <= 5; ++id) {
    std::array<uint8_t, 98> wrapper{}, factored{}, spun{};
    float norm = 0;
    for (int i = 0; i < 128; ++i)
      x[i] = float(a[i]);
    const size_t bytes = 18 + id * 16;
    auto r = request(id, TQC_D_ENCODE, x.data(), 512, wrapper.data(), bytes);
    CHECK(tqc_dialect_v1_execute(&r, &z) == 0);
    r = request(id, TQC_D_PREPARE, x.data(), 512, y.data(), 512);
    r.norms = {&norm, 4, 4};
    CHECK(tqc_dialect_v1_execute(&r, &z) == 0);
    r = request(id, TQC_D_PACK, y.data(), 512, factored.data(), bytes);
    r.norms = {&norm, 4, 4};
    CHECK(tqc_dialect_v1_execute(&r, &z) == 0);
    CHECK(std::memcmp(wrapper.data(), factored.data(), bytes) == 0);
    for (int i = 0; i < 128; ++i)
      u[i] = y[i];
    spin(u, false);
    for (int i = 0; i < 128; ++i)
      back[i] = float(u[i]);
    r = request(id, TQC_D_PACK, back.data(), 512, spun.data(), bytes);
    r.norms = {&norm, 4, 4};
    CHECK(tqc_dialect_v1_execute(&r, &z) == 0);
    CHECK(std::memcmp(spun.data(), wrapper.data(), bytes) != 0);
  }
  std::printf("FP64 factor maxima: basis %.17g, F32 roundtrip %.17g, inverse "
              "%.17g, wrong order %.17g, wrong inverse %.17g\n",
              max_basis, max_roundtrip, inverse_error, order_error,
              wrong_inverse);
}

// Fixture records: id, mode(0 wrapper,1 scalar,2 arbitrary decode), columns,
// norm(F32), input[columns], byte_count, packed[byte_count], decoded[columns].
static void fixtures(const char *path) {
  FILE *f = nullptr;
#ifdef _MSC_VER
  fopen_s(&f, path, "rb");
#else
  f = std::fopen(path, "rb");
#endif
  CHECK(f != nullptr);
  if (!f)
    return;
  unsigned count = 0;
  while (true) {
    uint32_t id = 0, mode = 0, n = 0, nb = 0;
    float norm = 0;
    if (std::fread(&id, 4, 1, f) != 1)
      break;
    CHECK(std::fread(&mode, 4, 1, f) == 1);
    CHECK(std::fread(&n, 4, 1, f) == 1);
    CHECK(std::fread(&norm, 4, 1, f) == 1);
    CHECK(n > 0 && n <= 256);
    if (n == 0 || n > 256)
      break;
    std::vector<float> x(n), decoded(n), out(n);
    CHECK(std::fread(x.data(), 4, n, f) == n);
    CHECK(std::fread(&nb, 4, 1, f) == 1);
    CHECK(nb > 0 && nb <= 196);
    if (!nb || nb > 196)
      break;
    std::vector<uint8_t> expected(nb), packed(nb);
    CHECK(std::fread(expected.data(), 1, nb, f) == nb);
    CHECK(std::fread(decoded.data(), 4, n, f) == n);
    tqc_dialect_v1_info fi{};
    fi.version = 1;
    fi.size = sizeof(fi);
    CHECK(tqc_dialect_v1_query(id, &fi) == 0);
    bool finite_scales = true;
    for (size_t b = 0; b < nb; b += fi.block_bytes)
      for (unsigned s = 0; s < fi.metadata_bytes; s += 2)
        finite_scales =
            finite_scales &&
            ((expected[b + fi.scale_offset + s + 1] & 0x7c) != 0x7c);
    auto z = receipt();
    auto r = request(id, mode == 1 ? TQC_D_PACK : TQC_D_ENCODE, x.data(), n * 4,
                     packed.data(), nb, n);
    if (mode == 1)
      r.norms = {&norm, 4, 4};
    // Predeclared exception: source may overflow FP16 corrected norms. The
    // checked API rejects this before writes rather than returning Inf.
    if (!finite_scales) {
      CHECK(mode < 2);
      CHECK(tqc_dialect_v1_execute(&r, &z) == TQC_D_RANGE);
      for (auto byte : packed)
        CHECK(byte == 0);
      r = request(id, TQC_D_DECODE, expected.data(), nb, out.data(), n * 4, n);
      CHECK(tqc_dialect_v1_execute(&r, &z) == TQC_D_NONFINITE);
      ++count;
      continue;
    }
    if (mode < 2) {
      CHECK(tqc_dialect_v1_execute(&r, &z) == 0);
      if (packed != expected) {
        std::fprintf(stderr, "oracle byte mismatch record %u id %u mode %u\n",
                     count, id, mode);
        ++failures;
      }
    }
    r = request(id, TQC_D_DECODE, expected.data(), nb, out.data(), n * 4, n);
    CHECK(tqc_dialect_v1_execute(&r, &z) == 0);
    if (std::memcmp(out.data(), decoded.data(), n * 4) != 0) {
      std::fprintf(stderr, "oracle F32 mismatch record %u id %u mode %u\n",
                   count, id, mode);
      ++failures;
    }
    if (id == 10 || id == 11) {
      const unsigned other = id == 10 ? 11 : 10;
      std::vector<uint8_t> paired(nb);
      if (mode == 0) {
        auto pair =
            request(other, TQC_D_ENCODE, x.data(), n * 4, paired.data(), nb, n);
        CHECK(tqc_dialect_v1_execute(&pair, &z) == 0);
        CHECK(paired == expected);
      }
      auto pair = request(other, TQC_D_DECODE, expected.data(), nb, out.data(),
                          n * 4, n);
      CHECK(tqc_dialect_v1_execute(&pair, &z) == 0);
      CHECK(std::memcmp(out.data(), decoded.data(), n * 4) == 0);
    }
    // Repeat the same record in two padded, non-contiguous rows.
    if (mode == 0) {
      std::vector<float> strided(2 * (n + 4), 999);
      std::vector<uint8_t> rows(2 * (nb + 7), 0xa5);
      std::memcpy(strided.data(), x.data(), n * 4);
      std::memcpy(strided.data() + n + 4, x.data(), n * 4);
      r = request(id, TQC_D_ENCODE, strided.data(), strided.size() * 4,
                  rows.data(), rows.size(), n, 2);
      CHECK(tqc_dialect_v1_execute(&r, &z) == 0);
      for (int j = 0; j < 2; ++j) {
        CHECK(std::memcmp(rows.data() + j * (nb + 7), expected.data(), nb) ==
              0);
        for (size_t k = nb; k < nb + 7; ++k)
          CHECK(rows[j * (nb + 7) + k] == 0xa5);
      }
    }
    ++count;
  }
  CHECK(std::feof(f) != 0);
  std::fclose(f);
  CHECK(count > 0);
  std::printf("oracle fixtures: %u, exact bytes and F32 bits\n", count);
}
int main(int argc, char **argv) {
  boundaries();
  factors();
  for (int i = 1; i < argc; ++i)
    fixtures(argv[i]);
  return failures ? 1 : 0;
}
