#include "spin8/spin8.h"
#include "tqcodec/cache_cuda.h"
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cuda_runtime_api.h>
#include <limits>
#include <stdexcept>
#include <vector>
static size_t checks = 0, bytes_checked = 0;
static double max_error = 0;
#define CHECK(x)                                                               \
  do {                                                                         \
    ++checks;                                                                  \
    if (!(x)) {                                                                \
      std::fprintf(stderr, "line %d: %s\n", __LINE__, #x);                     \
      throw std::runtime_error(#x);                                            \
    }                                                                          \
  } while (0)
#define CUDA(x)                                                                \
  do {                                                                         \
    auto e = (x);                                                              \
    if (e != cudaSuccess)                                                      \
      throw std::runtime_error(cudaGetErrorString(e));                         \
  } while (0)
struct Buffer {
  void *p{};
  size_t n;
  cudaStream_t stream;
  Buffer(size_t bytes, cudaStream_t s) : n(bytes), stream(s) {
    CUDA(cudaMalloc(&p, n));
  }
  ~Buffer() { cudaFree(p); }
  Buffer(const Buffer &) = delete;
  void put(const void *src) {
    CUDA(cudaMemcpyAsync(p, src, n, cudaMemcpyHostToDevice, stream));
  }
  void get(void *dst) {
    CUDA(cudaMemcpyAsync(dst, p, n, cudaMemcpyDeviceToHost, stream));
    CUDA(cudaStreamSynchronize(stream));
  }
  tqc_dialect_v1_span span(size_t stride) { return {p, n, stride}; }
};
static tqc_cache_cuda_plane planes[] = {{0, 1, 0.37}, {2, 5, -0.61}};
static tqc_cache_cuda_config config(unsigned columns = 128, unsigned k = 3,
                                    unsigned v = 1, unsigned mask = 7,
                                    unsigned diag = 0) {
  tqc_cache_cuda_config c{};
  c.version = 1;
  c.size = sizeof(c);
  c.precision = TQC_D_F32;
  c.representation = diag;
  c.columns = columns;
  c.kv_heads = 2;
  c.q_heads = 4;
  c.capacity = 5;
  c.max_queries = 2;
  c.max_append_rows = 3;
  for (int i = 0; i < 3; ++i)
    c.alpha[i] = (mask & (1u << i)) ? (mask == 7 ? 1.0 / 3.0 : 1.0) : 0;
  c.view_mask = mask;
  c.k_dialect = k;
  c.v_dialect = v;
  c.model[0] = 1;
  c.adapter[0] = 2;
  c.position[0] = 3;
  c.planes = planes;
  c.plane_count = 2;
  return c;
}
struct Cache {
  tqc_cache_cuda_config c;
  tqc_cache_cuda_plan *p{};
  tqc_cache_cuda_layout l{1, sizeof(l)};
  cudaStream_t stream{};
  void *arena{};
  tqc_cache_cuda_context ctx{};
  Cache(tqc_cache_cuda_config cfg) : c(cfg) {
    CHECK(tqc_cache_cuda_create(&c, &p) == 0);
    CHECK(tqc_cache_cuda_get_layout(p, &l) == 0);
    CUDA(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
    CUDA(cudaMalloc(&arena, l.total_bytes));
    ctx = {1, sizeof(ctx), p, arena, l.total_bytes, 0, stream};
    CHECK(tqc_cache_cuda_initialize(&ctx) == 0);
  }
  ~Cache() {
    cudaStreamSynchronize(stream);
    cudaFree(arena);
    cudaStreamDestroy(stream);
    tqc_cache_cuda_free(p);
  }
  void cmd(uint64_t n, uint64_t q = 2, uint64_t target = 0) {
    tqc_cache_cuda_command command{n, q, target};
    CUDA(cudaMemcpyAsync(static_cast<char *>(arena) + l.command, &command,
                         sizeof(command), cudaMemcpyHostToDevice, stream));
    CUDA(cudaStreamSynchronize(stream));
  }
  tqc_cache_cuda_state state() {
    tqc_cache_cuda_state s{};
    CUDA(cudaMemcpyAsync(&s, static_cast<char *>(arena) + l.state, sizeof(s),
                         cudaMemcpyDeviceToHost, stream));
    CUDA(cudaStreamSynchronize(stream));
    return s;
  }
  void good(uint64_t committed, uint64_t tentative) {
    auto s = state();
    CHECK(s.valid == 1 && s.status == 0 && s.committed == committed &&
          s.tentative == tentative);
  }
  void reset() {
    CHECK(tqc_cache_cuda_initialize(&ctx) == 0);
    good(0, 0);
  }
  std::vector<unsigned char> image() {
    std::vector<unsigned char> data(l.total_bytes);
    CUDA(cudaMemcpyAsync(data.data(), arena, data.size(),
                         cudaMemcpyDeviceToHost, stream));
    CUDA(cudaStreamSynchronize(stream));
    return data;
  }
};
static std::vector<float> data(size_t rows, size_t stride, int seed) {
  std::vector<float> a(rows * stride, 777);
  for (size_t i = 0; i < a.size(); ++i)
    a[i] = float(int((i * 37 + seed) % 211) - 105) * .003f;
  return a;
}
static void dialect(unsigned id, unsigned op, float *x, void *y,
                    float *norm = nullptr) {
  tqc_dialect_v1_request r{1,
                           sizeof(r),
                           id,
                           op,
                           TQC_D_CPU,
                           TQC_D_F32,
                           nullptr,
                           1,
                           128,
                           {x, 512, 512},
                           {y, op == TQC_D_PACK ? size_t(18 + 16 * id) : 512,
                            op == TQC_D_PACK ? size_t(18 + 16 * id) : 512},
                           norm ? tqc_dialect_v1_span{norm, 4, 4}
                                : tqc_dialect_v1_span{}};
  if (op == TQC_D_DECODE)
    r.src = {x, size_t(18 + 16 * id), size_t(18 + 16 * id)};
  tqc_dialect_v1_receipt receipt{1, sizeof(receipt), 0, 0, 0};
  CHECK(tqc_dialect_v1_execute(&r, &receipt) == 0);
}
static void rotate(float *x, const double *b) {
  for (int g = 0; g < 128; g += 8) {
    float y[8]{};
    for (int i = 0; i < 8; ++i)
      for (int j = 0; j < 8; ++j)
        y[i] += static_cast<float>(b[i * 8 + j]) * x[g + j];
    std::copy_n(y, 8, x + g);
  }
}
static void matrices(double *b) {
  spin8_plane ps[] = {{0, 1, .37}, {2, 5, -.61}};
  spin8_plan *p = nullptr;
  CHECK(spin8_plan_create(ps, 2, &p) == SPIN8_OK);
  for (int r = 0; r < 3; ++r)
    CHECK(spin8_plan_matrix(p, static_cast<spin8_view>(r), b + r * 64) ==
          SPIN8_OK);
  spin8_plan_free(p);
}
static void physical_and_attention(unsigned columns, unsigned kid, unsigned vid,
                                   unsigned mask, unsigned diag) {
  auto cfg = config(columns, kid, vid, mask, diag);
  if (kid == 3 && vid == 1 && mask == 7 && !diag) {
    cfg.alpha[0] = .2;
    cfg.alpha[1] = .3;
    cfg.alpha[2] = .5;
  }
  Cache cache(cfg);
  const auto c = cache.c;
  const auto l = cache.l;
  size_t payload = l.v_bytes;
  for (int i = 0; i < 3; ++i)
    payload += l.k_bytes[i];
  CHECK(l.total_bytes == payload + l.metadata_bytes + l.alignment_bytes);
  std::printf("layout cols=%u K=%u V=%u views=%u diagnostic=%u arena=%zu "
              "payload=%zu metadata=%zu padding=%zu\n",
              columns, kid + 1, vid + 1, mask, diag, l.total_bytes, payload,
              l.metadata_bytes, l.alignment_bytes);
  CHECK(l.v_bytes == c.capacity * c.kv_heads * (columns / 128) *
                         (diag ? 512 : 18 + 16 * vid));
  for (int r = 0; r < 3; ++r)
    CHECK(l.k_bytes[r] == ((mask & (1u << r))
                               ? c.capacity * c.kv_heads * (columns / 128) *
                                     (diag ? 512 : 18 + 16 * kid)
                               : 0));
  CHECK(std::memcmp(l.matrix_identity[0], l.matrix_identity[1], 32) != 0);
  size_t fs = columns + 4, rows = size_t(c.max_append_rows) * c.kv_heads,
         qr = size_t(c.max_queries) * c.q_heads;
  auto k = data(rows, fs, 7), v = data(rows, fs, 11), q = data(qr, fs, 19);
  std::vector<float> out(qr * fs, -99);
  int64_t pos[3] = {0, 1, 2}, qp[2] = {1, 2};
  Buffer dk(k.size() * 4, cache.stream), dv(v.size() * 4, cache.stream),
      dq(q.size() * 4, cache.stream), dout(out.size() * 4, cache.stream),
      dp(sizeof(pos), cache.stream), dqp(sizeof(qp), cache.stream);
  dk.put(k.data());
  dv.put(v.data());
  dq.put(q.data());
  dp.put(pos);
  dqp.put(qp);
  tqc_cache_cuda_append append{1, sizeof(append), dk.span(fs * 4),
                               dv.span(fs * 4), dp.span(8)};
  cache.cmd(3);
  CHECK(tqc_cache_cuda_append_rows(&cache.ctx, &append) == 0);
  cache.good(0, 3);
  auto image = cache.image();
  double b[192];
  matrices(b);
  std::vector<float> decoded_k[3], decoded_v(3 * c.kv_heads * columns);
  for (auto &a : decoded_k)
    a.resize(3 * c.kv_heads * columns);
  for (size_t row = 0; row < 3 * c.kv_heads; ++row)
    for (unsigned block = 0; block < columns / 128; ++block) {
      float x[128], z[128], norm = 0;
      std::copy_n(k.data() + row * fs + block * 128, 128, x);
      dialect(kid, diag ? TQC_D_FORWARD : TQC_D_PREPARE, x, z,
              diag ? nullptr : &norm);
      for (int r = 0; r < 3; ++r)
        if (mask & (1u << r)) {
          float transformed[128];
          std::copy_n(z, 128, transformed);
          rotate(transformed, b + 64 * r);
          auto *actual = image.data() + l.k[r] + row * l.k_row_bytes +
                         block * (diag ? 512 : 18 + 16 * kid);
          if (diag) {
            CHECK(std::memcmp(actual, transformed, 512) == 0);
            std::copy_n(transformed, 128,
                        decoded_k[r].data() + row * columns + block * 128);
            bytes_checked += 512;
          } else {
            unsigned char expected[98]{};
            dialect(kid, TQC_D_PACK, transformed, expected, &norm);
            CHECK(std::memcmp(actual, expected, 18 + 16 * kid) == 0);
            bytes_checked += 18 + 16 * kid;
            dialect(kid, TQC_D_DECODE, reinterpret_cast<float *>(expected),
                    decoded_k[r].data() + row * columns + block * 128);
          }
        }
      auto *actual = image.data() + l.v + row * l.v_row_bytes +
                     block * (diag ? 512 : 18 + 16 * vid);
      std::copy_n(v.data() + row * fs + block * 128, 128, x);
      dialect(vid, diag ? TQC_D_FORWARD : TQC_D_PREPARE, x, z,
              diag ? nullptr : &norm);
      if (diag) {
        CHECK(std::memcmp(actual, z, 512) == 0);
        bytes_checked += 512;
      } else {
        unsigned char expected[98]{};
        dialect(vid, TQC_D_PACK, z, expected, &norm);
        CHECK(std::memcmp(actual, expected, 18 + 16 * vid) == 0);
        bytes_checked += 18 + 16 * vid;
        dialect(vid, TQC_D_DECODE, reinterpret_cast<float *>(expected), z);
      }
      dialect(vid, TQC_D_INVERSE, z,
              decoded_v.data() + row * columns + block * 128);
    }
  tqc_cache_cuda_attention att{1,
                               sizeof(att),
                               1,
                               0,
                               .13f,
                               dq.span(fs * 4),
                               dout.span(fs * 4),
                               dqp.span(8),
                               {},
                               {},
                               {},
                               0};
  CHECK(tqc_cache_cuda_attend(&cache.ctx, &att) == 0);
  cache.good(0, 3);
  dout.get(out.data());
  for (size_t qi = 0; qi < c.max_queries; ++qi)
    for (unsigned h = 0; h < c.q_heads; ++h) {
      size_t row = qi * c.q_heads + h, kv = h / (c.q_heads / c.kv_heads);
      double scores[3]{}, den = 0;
      std::vector<double> expected(columns, 0);
      for (int t = 0; t < 3; ++t) {
        if (pos[t] > qp[qi])
          continue;
        double score = 0;
        if (diag) {
          for (unsigned j = 0; j < columns; ++j)
            score +=
                double(q[row * fs + j]) * k[(t * c.kv_heads + kv) * fs + j];
        } else
          for (int r = 0; r < 3; ++r)
            if (mask & (1u << r)) {
              double dot = 0;
              for (unsigned block = 0; block < columns / 128; ++block) {
                float x[128], z[128];
                std::copy_n(q.data() + row * fs + block * 128, 128, x);
                dialect(kid, TQC_D_FORWARD, x, z);
                rotate(z, b + 64 * r);
                for (int j = 0; j < 128; ++j)
                  dot += double(z[j]) *
                         decoded_k[r][(t * c.kv_heads + kv) * columns +
                                      block * 128 + j];
              }
              score += dot * c.alpha[r];
            }
        scores[t] = std::exp(score * .13f);
        den += scores[t];
        for (unsigned j = 0; j < columns; ++j)
          expected[j] +=
              scores[t] *
              (diag ? v[(t * c.kv_heads + kv) * fs + j]
                    : decoded_v[(t * c.kv_heads + kv) * columns + j]);
      }
      for (unsigned j = 0; j < columns; ++j) {
        double err = std::abs(out[row * fs + j] - expected[j] / den);
        max_error = std::max(max_error, err);
        CHECK(err < 2e-5);
      }
    }
  CHECK(tqc_cache_cuda_control(&cache.ctx, TQC_CC_COMMIT) == 0);
  cache.good(3, 3);
  if (kid == 3 && vid == 1 && mask == 7 && !diag) {
    Cache restored(cache.c);
    size_t n = 0;
    CHECK(tqc_cache_cuda_snapshot_size(cache.p, &n) == 0);
    std::vector<unsigned char> snapshot(n);
    CHECK(tqc_cache_cuda_snapshot(&cache.ctx, snapshot.data(), n) == 0);
    CHECK(tqc_cache_cuda_restore(&restored.ctx, snapshot.data(), n) == 0);
    restored.good(3, 3);
    CHECK(restored.image() == cache.image());
  }
  cache.cmd(0, 2, 2);
  CHECK(tqc_cache_cuda_control(&cache.ctx, TQC_CC_TRUNCATE) == 0);
  cache.good(2, 2);
  cache.cmd(1);
  CHECK(tqc_cache_cuda_append_rows(&cache.ctx, &append) == 0);
  cache.good(2, 3);
  CHECK(tqc_cache_cuda_control(&cache.ctx, TQC_CC_CANCEL) == 0);
  cache.good(2, 2);
  auto rejected = cache.ctx;
  rejected.stream = nullptr;
  auto before = cache.image();
  CHECK(tqc_cache_cuda_append_rows(&rejected, &append) == TQC_D_UNSUPPORTED);
  CHECK(cache.image() == before);
  auto alias = append;
  alias.v = alias.k;
  CHECK(tqc_cache_cuda_append_rows(&cache.ctx, &alias) == TQC_D_OVERLAP);
  CHECK(cache.image() == before);
  CHECK(tqc_cache_cuda_position_transform(&cache.ctx) == TQC_D_UNSUPPORTED);
  // Actual device corruption must invalidate all views and forbid Attention.
  float corrupt = 2;
  CUDA(cudaMemcpyAsync(static_cast<char *>(cache.arena) + l.matrices, &corrupt,
                       4, cudaMemcpyHostToDevice, cache.stream));
  CHECK(tqc_cache_cuda_audit_device(&cache.ctx) == 0);
  auto failed = cache.state();
  CHECK(!failed.valid &&
        (failed.status & TQC_CC_TRANSFORM_INVARIANT_VIOLATION));
  CHECK(tqc_cache_cuda_attend(&cache.ctx, &att) == 0);
  CHECK(!cache.state().valid);
  cache.reset();
  cache.cmd(1);
  v[0] = std::numeric_limits<float>::quiet_NaN();
  dv.put(v.data());
  CHECK(tqc_cache_cuda_append_rows(&cache.ctx, &append) == 0);
  CHECK(tqc_cache_cuda_control(&cache.ctx, TQC_CC_COMMIT) == 0);
  failed = cache.state();
  CHECK(!failed.valid && failed.committed == 0 && failed.tentative == 0 &&
        (failed.status & TQC_CC_NUMERICAL));
}
static void lifecycle() {
  Cache source(config(128, 3, 1, 7, 1));
  Cache dest(source.c);
  tqc_cache_cuda_capabilities caps{1, sizeof(caps)};
  CHECK(tqc_cache_cuda_query(&caps) == 0);
  CHECK(!caps.position_transport && !caps.shared_qjl && !caps.coefficient_f16 &&
        !caps.coefficient_bf16);
  tqc_cache_cuda_resources resources{1, sizeof(resources)};
  CHECK(tqc_cache_cuda_resources_query(&source.ctx, &resources) == 0);
  CHECK(resources.local_bytes[0] < 4096 &&
        resources.shared_bytes[0] >= 128 * 128 * sizeof(float));
  for (int i = 0; i < 6; ++i)
    std::printf("kernel %d local=%zu shared=%zu constant=%zu registers=%d "
                "maxthreads=%d\n",
                i, resources.local_bytes[i], resources.shared_bytes[i],
                resources.constant_bytes[i], resources.registers[i],
                resources.max_threads[i]);
  std::printf("diagnostic arena=%zu metadata=%zu padding=%zu hostplan=%zu "
              "module-table-bytes=%zu\n",
              source.l.total_bytes, source.l.metadata_bytes,
              source.l.alignment_bytes, source.l.host_plan_bytes,
              source.l.constant_bytes);
  tqc_cache_cuda_audit audit{};
  CUDA(cudaMemcpyAsync(&audit,
                       static_cast<char *>(source.arena) + source.l.audit,
                       sizeof(audit), cudaMemcpyDeviceToHost, source.stream));
  CUDA(cudaStreamSynchronize(source.stream));
  for (int i = 0; i < 4; ++i) {
    CHECK(audit.finite[i] && audit.ef[i] <= 1e-5 && audit.gram[i] <= 1e-5 &&
          audit.drift[i] <= 1e-5 &&
          std::abs(audit.logabsdet[i]) <= (i == 3 ? 128 : 8) * 1e-5);
    CHECK(i == 3 ? std::abs(audit.det_sign[i]) == 1 : audit.det_sign[i] == 1);
    std::printf("audit %d ef=%.17g gram=%.17g logabsdet=%.17g sign=%d "
                "finite=%u drift=%.17g\n",
                i, audit.ef[i], audit.gram[i], audit.logabsdet[i],
                audit.det_sign[i], audit.finite[i], audit.drift[i]);
  }
  const size_t fs = 128, rows = source.c.max_append_rows * source.c.kv_heads;
  auto k = data(rows, fs, 17);
  std::vector<float> v(rows * fs), q(2 * 4 * fs, 0), out(q.size());
  for (size_t t = 0; t < source.c.max_append_rows; ++t)
    for (size_t h = 0; h < 2; ++h)
      for (size_t j = 0; j < fs; ++j)
        v[(t * 2 + h) * fs + j] = float(t + 1) * .1f;
  int64_t pos[3] = {0, 1, 2}, qp[2] = {9, 9};
  Buffer dk(k.size() * 4, source.stream), dv(v.size() * 4, source.stream),
      dp(sizeof(pos), source.stream), dq(q.size() * 4, source.stream),
      dout(out.size() * 4, source.stream), dqp(sizeof(qp), source.stream);
  dk.put(k.data());
  dv.put(v.data());
  dp.put(pos);
  dq.put(q.data());
  dqp.put(qp);
  tqc_cache_cuda_append app{1, sizeof(app), dk.span(fs * 4), dv.span(fs * 4),
                            dp.span(8)};
  source.cmd(3);
  CHECK(tqc_cache_cuda_append_rows(&source.ctx, &app) == 0);
  CHECK(tqc_cache_cuda_control(&source.ctx, TQC_CC_COMMIT) == 0);
  source.good(3, 3);
  dest.good(0, 0);
  size_t size = 0;
  CHECK(tqc_cache_cuda_snapshot_size(source.p, &size) == 0);
  std::vector<unsigned char> snap(size);
  CHECK(tqc_cache_cuda_snapshot(&source.ctx, snap.data(), snap.size()) == 0);
  CHECK(tqc_cache_cuda_restore(&dest.ctx, snap.data(), snap.size()) == 0);
  dest.good(3, 3);
  auto srcimage = source.image(), dstimage = dest.image();
  CHECK(srcimage == dstimage);
  auto corrupt = snap;
  corrupt.back() ^= 1;
  CHECK(tqc_cache_cuda_restore(&dest.ctx, corrupt.data(), corrupt.size()) ==
        TQC_D_INVALID);
  CHECK(dest.image() == dstimage);
  CHECK(tqc_cache_cuda_restore(&dest.ctx, snap.data(), snap.size() - 1) ==
        TQC_D_CAPACITY);
  CHECK(dest.image() == dstimage);
  auto incompatible = source.c;
  incompatible.adapter[0] = 8;
  Cache other(incompatible);
  auto original = other.image();
  CHECK(tqc_cache_cuda_restore(&other.ctx, snap.data(), snap.size()) ==
        TQC_D_INVALID);
  CHECK(other.image() == original);
  dest.reset();
  auto sc = source.ctx;
  sc.stream = dest.stream;
  CHECK(tqc_cache_cuda_copy(&dest.ctx, &sc, {}, 1) == 0);
  dest.good(3, 3);
  dstimage = dest.image();
  for (int r = 0; r < 3; ++r)
    CHECK(std::memcmp(dstimage.data() + dest.l.k[r],
                      srcimage.data() + source.l.k[r],
                      3 * 2 * source.l.k_row_bytes) == 0);
  dest.reset();
  int64_t indices[5] = {2, 0, -1, -1, -1};
  Buffer di(sizeof(indices), dest.stream);
  di.put(indices);
  dest.cmd(2);
  CHECK(tqc_cache_cuda_copy(&dest.ctx, &sc, di.span(8), 0) == 0);
  dest.good(0, 2);
  CHECK(tqc_cache_cuda_control(&dest.ctx, TQC_CC_COMMIT) == 0);
  dest.good(2, 2);
  dstimage = dest.image();
  for (int r = 0; r < 3; ++r)
    for (size_t t = 0; t < 2; ++t)
      CHECK(std::memcmp(dstimage.data() + dest.l.k[r] +
                            t * 2 * dest.l.k_row_bytes,
                        srcimage.data() + source.l.k[r] +
                            indices[t] * 2 * source.l.k_row_bytes,
                        2 * source.l.k_row_bytes) == 0);
  CHECK(tqc_cache_cuda_copy(&dest.ctx, &dest.ctx, di.span(8), 0) ==
        TQC_D_OVERLAP);
  dest.reset();
  indices[1] = 2;
  di.put(indices);
  dest.cmd(2);
  CHECK(tqc_cache_cuda_copy(&dest.ctx, &sc, di.span(8), 0) == 0);
  auto failed = dest.state();
  CHECK(!failed.valid && failed.tentative == 0 &&
        (failed.status & TQC_CC_INDEX));
  dest.reset();
  indices[0] = 5;
  di.put(indices);
  dest.cmd(1);
  CHECK(tqc_cache_cuda_copy(&dest.ctx, &sc, di.span(8), 0) == 0);
  CHECK(dest.state().status & TQC_CC_INDEX);
  // Sparse key indices, explicit bias and mask use original physical key IDs.
  int64_t sparse[4] = {2, 0, 2, 0};
  std::vector<float> bias(8 * 5, 0);
  std::vector<unsigned char> mask(8 * 5, 1);
  for (int row = 0; row < 8; ++row)
    bias[row * 5 + 2] = std::log(3.0f);
  Buffer si(sizeof(sparse), source.stream), db(bias.size() * 4, source.stream),
      dm(mask.size(), source.stream);
  si.put(sparse);
  db.put(bias.data());
  dm.put(mask.data());
  tqc_cache_cuda_attention att{1,
                               sizeof(att),
                               0,
                               0,
                               .2f,
                               dq.span(fs * 4),
                               dout.span(fs * 4),
                               dqp.span(8),
                               dm.span(5),
                               db.span(20),
                               si.span(16),
                               2};
  CHECK(tqc_cache_cuda_attend(&source.ctx, &att) == 0);
  source.good(3, 3);
  dout.get(out.data());
  for (auto x : out)
    CHECK(std::abs(x - .25f) < 2e-6);
  std::fill(mask.begin(), mask.end(), 0);
  dm.put(mask.data());
  CHECK(tqc_cache_cuda_attend(&source.ctx, &att) == 0);
  CHECK(source.state().status & TQC_CC_ALL_MASKED);
  source.reset();
  source.cmd(0);
  att.mask = {};
  att.bias = {};
  att.indices = {};
  att.selection_width = 0;
  CHECK(tqc_cache_cuda_attend(&source.ctx, &att) == 0);
  CHECK(source.state().status & TQC_CC_EMPTY);
  source.reset();
  source.cmd(4); // above max_append_rows, below cache capacity
  CHECK(tqc_cache_cuda_append_rows(&source.ctx, &app) == 0);
  failed = source.state();
  CHECK(!failed.valid && failed.committed == 0 && failed.tentative == 0 &&
        (failed.status & TQC_CC_CAPACITY));
  // Replay reads current device counts and positions, not captured host
  // scalars.
  source.reset();
  source.cmd(1);
  cudaGraph_t graph{};
  cudaGraphExec_t executable{};
  CUDA(cudaStreamBeginCapture(source.stream, cudaStreamCaptureModeThreadLocal));
  CHECK(tqc_cache_cuda_append_rows(&source.ctx, &app) == 0);
  CHECK(tqc_cache_cuda_control(&source.ctx, TQC_CC_COMMIT) == 0);
  CHECK(tqc_cache_cuda_attend(&source.ctx, &att) == 0);
  CUDA(cudaStreamEndCapture(source.stream, &graph));
  CUDA(cudaGraphInstantiate(&executable, graph, 0));
  CUDA(cudaGraphLaunch(executable, source.stream));
  source.good(1, 1);
  pos[0] = 42;
  dp.put(pos);
  std::fill(out.begin(), out.end(), -99.0f);
  dout.put(out.data());
  source.cmd(1, 1);
  CUDA(cudaGraphLaunch(executable, source.stream));
  source.good(2, 2);
  dout.get(out.data());
  for (size_t i = 0; i < out.size(); ++i)
    CHECK(i < 4 * fs ? std::abs(out[i] - .1f) < 2e-6 : out[i] == -99.0f);
  auto replay = source.image();
  int64_t actual = 0;
  std::memcpy(&actual, replay.data() + source.l.positions + 8, 8);
  CHECK(actual == 42);
  CUDA(cudaGraphExecDestroy(executable));
  CUDA(cudaGraphDestroy(graph));
  // A different model/adapter/profile must never reuse an already bound arena.
  auto cross = source.ctx;
  cross.plan = other.p;
  CHECK(tqc_cache_cuda_attend(&cross, &att) == 0);
  CHECK(source.state().status & TQC_CC_INVALID_STATE);
  // Host rejections leave the entire arena unchanged, including sticky status.
  auto bound = source.image();
  auto bad = app;
  bad.k.capacity_bytes = 4;
  CHECK(tqc_cache_cuda_append_rows(&source.ctx, &bad) == TQC_D_CAPACITY);
  CHECK(source.image() == bound);
  bad = app;
  bad.k.data = reinterpret_cast<void *>(UINTPTR_MAX - 3);
  CHECK(tqc_cache_cuda_append_rows(&source.ctx, &bad) == TQC_D_OVERFLOW);
  CHECK(source.image() == bound);
  bad = app;
  bad.k.data = static_cast<char *>(dk.p) + 1;
  CHECK(tqc_cache_cuda_append_rows(&source.ctx, &bad) == TQC_D_ALIGNMENT);
  CHECK(source.image() == bound);
  auto badctx = source.ctx;
  badctx.device_ordinal = 1;
  CHECK(tqc_cache_cuda_append_rows(&badctx, &app) == TQC_D_UNSUPPORTED);
  CHECK(source.image() == bound);
  source.reset();
  source.cmd(0, 0);
  CHECK(tqc_cache_cuda_append_rows(&source.ctx, &app) == 0);
  CHECK(tqc_cache_cuda_attend(&source.ctx, &att) == 0);
  source.good(0, 0);
  source.cmd(1, 2);
  CHECK(tqc_cache_cuda_append_rows(&source.ctx, &app) == 0);
  CHECK(tqc_cache_cuda_control(&source.ctx, TQC_CC_COMMIT) == 0);
  source.good(1, 1);
  q[0] = std::numeric_limits<float>::infinity();
  dq.put(q.data());
  CHECK(tqc_cache_cuda_attend(&source.ctx, &att) == 0);
  CHECK(source.state().status & TQC_CC_NUMERICAL);
  auto invalid = source.c;
  invalid.beta = .1;
  tqc_cache_cuda_plan *unchanged = nullptr;
  CHECK(tqc_cache_cuda_create(&invalid, &unchanged) == TQC_D_UNSUPPORTED &&
        unchanged == nullptr);
  invalid = source.c;
  invalid.alpha[0] = -.1;
  CHECK(tqc_cache_cuda_create(&invalid, &unchanged) == TQC_D_INVALID);
  invalid = source.c;
  invalid.max_append_rows = 0;
  CHECK(tqc_cache_cuda_create(&invalid, &unchanged) == TQC_D_INVALID);
  invalid = source.c;
  invalid.max_append_rows = invalid.capacity + 1;
  CHECK(tqc_cache_cuda_create(&invalid, &unchanged) == TQC_D_INVALID);
  invalid = source.c;
  invalid.capacity = UINT64_MAX;
  CHECK(tqc_cache_cuda_create(&invalid, &unchanged) == TQC_D_OVERFLOW);
  invalid = source.c;
  invalid.precision = 2;
  CHECK(tqc_cache_cuda_create(&invalid, &unchanged) == TQC_D_UNSUPPORTED);
  invalid = source.c;
  invalid.columns = 192;
  CHECK(tqc_cache_cuda_create(&invalid, &unchanged) == TQC_D_UNSUPPORTED);
}
static void bounded_calls() {
  auto cfg = config();
  cfg.max_queries = 3; // both plan maxima exceed the true two-token buffers
  Cache c(cfg);
  const size_t row = cfg.columns * 4, heads = cfg.q_heads, kv = cfg.kv_heads;
  std::vector<float> k(2 * kv * cfg.columns, .125f),
      v(2 * kv * cfg.columns, .25f), q(2 * heads * cfg.columns, .03125f),
      out(q.size(), -99);
  int64_t positions[2] = {0, 1};
  Buffer dk(k.size() * 4, c.stream), dv(v.size() * 4, c.stream),
      dq(q.size() * 4, c.stream), dout(out.size() * 4, c.stream),
      dp(sizeof(positions), c.stream), dqp(sizeof(positions), c.stream);
  dk.put(k.data());
  dv.put(v.data());
  dq.put(q.data());
  dout.put(out.data());
  dp.put(positions);
  dqp.put(positions);
  tqc_cache_cuda_append app{1, sizeof(app), dk.span(row), dv.span(row),
                            dp.span(8)};
  tqc_cache_cuda_attention att{
      1, sizeof(att), 0, 0, .125f, dq.span(row), dout.span(row), dqp.span(8)};
  // Truly smaller external allocations: the old entry points still require max.
  CHECK(tqc_cache_cuda_append_rows(&c.ctx, &app) == TQC_D_CAPACITY);
  CHECK(tqc_cache_cuda_attend(&c.ctx, &att) == TQC_D_CAPACITY);
  c.cmd(1, 1);
  CHECK(tqc_cache_cuda_append_rows_bounded(&c.ctx, &app, 2) == 0);
  CHECK(tqc_cache_cuda_control(&c.ctx, TQC_CC_COMMIT) == 0);
  CHECK(tqc_cache_cuda_attend_bounded(&c.ctx, &att, 2) == 0);
  c.good(1, 1);
  dout.get(out.data());
  for (size_t i = 0; i < out.size(); ++i)
    CHECK(i < heads * cfg.columns ? std::isfinite(out[i]) && out[i] != -99
                                  : out[i] == -99);
  c.reset();
  c.cmd(2, 2);
  CHECK(tqc_cache_cuda_append_rows_bounded(&c.ctx, &app, 2) == 0);
  CHECK(tqc_cache_cuda_attend_bounded(&c.ctx, &att, 2) == 0);
  c.good(0, 2);
  c.reset();
  c.cmd(0, 0);
  CHECK(tqc_cache_cuda_append_rows_bounded(&c.ctx, &app, 2) == 0);
  CHECK(tqc_cache_cuda_attend_bounded(&c.ctx, &att, 2) == 0);
  c.good(0, 0);
  auto before = c.image();
  for (size_t bound : {size_t(0), size_t(4), SIZE_MAX}) {
    CHECK(tqc_cache_cuda_append_rows_bounded(&c.ctx, &app, bound) ==
          TQC_D_INVALID);
    CHECK(tqc_cache_cuda_attend_bounded(&c.ctx, &att, bound) == TQC_D_INVALID);
    CHECK(c.image() == before);
  }
  // Above call bound, within BOTH plan maximum and cache capacity. No payload
  // or positions may be read/written and no tentative prefix may be published.
  c.cmd(3, 0);
  before = c.image();
  CHECK(tqc_cache_cuda_append_rows_bounded(&c.ctx, &app, 2) == 0);
  auto state = c.state();
  CHECK(!state.valid && state.status == TQC_CC_CAPACITY &&
        state.tentative == 0);
  auto after = c.image();
  for (size_t i = 0; i < before.size(); ++i)
    if (i < c.l.state || i >= c.l.state + sizeof(state))
      CHECK(before[i] == after[i]);
  c.reset();
  c.cmd(1, 3);
  CHECK(tqc_cache_cuda_append_rows_bounded(&c.ctx, &app, 2) == 0);
  std::fill(out.begin(), out.end(), -99.f);
  dout.put(out.data());
  CHECK(tqc_cache_cuda_attend_bounded(&c.ctx, &att, 2) == 0);
  state = c.state();
  CHECK(!state.valid && state.status == TQC_CC_CAPACITY);
  dout.get(out.data());
  for (float x : out)
    CHECK(x == -99);
  // The same graph observes current persistent counts on each replay.
  c.reset();
  c.cmd(1, 1);
  cudaGraph_t graph{};
  cudaGraphExec_t executable{};
  CUDA(cudaStreamBeginCapture(c.stream, cudaStreamCaptureModeThreadLocal));
  CHECK(tqc_cache_cuda_append_rows_bounded(&c.ctx, &app, 2) == 0);
  CHECK(tqc_cache_cuda_control(&c.ctx, TQC_CC_COMMIT) == 0);
  CHECK(tqc_cache_cuda_attend_bounded(&c.ctx, &att, 2) == 0);
  CUDA(cudaStreamEndCapture(c.stream, &graph));
  CUDA(cudaGraphInstantiate(&executable, graph, 0));
  CUDA(cudaGraphLaunch(executable, c.stream));
  c.good(1, 1);
  c.cmd(2, 2);
  CUDA(cudaGraphLaunch(executable, c.stream));
  c.good(3, 3);
  c.cmd(3, 1);
  CUDA(cudaGraphLaunch(executable, c.stream));
  state = c.state();
  CHECK(!state.valid && state.tentative == 3);
  CUDA(cudaGraphExecDestroy(executable));
  CUDA(cudaGraphDestroy(graph));
}
static void bounded_query_replay() {
  auto cfg = config();
  cfg.max_queries = 3;
  Cache c(cfg);
  const size_t row = cfg.columns * 4;
  std::vector<float> kv(cfg.kv_heads * cfg.columns, .125f),
      q(2 * cfg.q_heads * cfg.columns, .25f), out(q.size(), -99.f);
  int64_t pos[2] = {0, 0};
  Buffer dk(kv.size() * 4, c.stream), dv(kv.size() * 4, c.stream),
      dq(q.size() * 4, c.stream), dout(out.size() * 4, c.stream),
      dp(8, c.stream), dqp(16, c.stream);
  dk.put(kv.data());
  dv.put(kv.data());
  dq.put(q.data());
  dout.put(out.data());
  dp.put(pos);
  dqp.put(pos);
  tqc_cache_cuda_append app{1, sizeof(app), dk.span(row), dv.span(row),
                            dp.span(8)};
  tqc_cache_cuda_attention att{
      1, sizeof(att), 0, 0, .125f, dq.span(row), dout.span(row), dqp.span(8)};
  c.cmd(1, 1);
  CHECK(tqc_cache_cuda_append_rows_bounded(&c.ctx, &app, 1) == 0);
  cudaGraph_t graph{};
  cudaGraphExec_t executable{};
  CUDA(cudaStreamBeginCapture(c.stream, cudaStreamCaptureModeThreadLocal));
  CHECK(tqc_cache_cuda_attend_bounded(&c.ctx, &att, 2) == 0);
  CUDA(cudaStreamEndCapture(c.stream, &graph));
  CUDA(cudaGraphInstantiate(&executable, graph, 0));
  CUDA(cudaGraphLaunch(executable, c.stream));
  c.good(0, 1);
  c.cmd(0, 2);
  CUDA(cudaGraphLaunch(executable, c.stream));
  c.good(0, 1);
  dout.put(out.data());
  c.cmd(0, 3);
  CUDA(cudaGraphLaunch(executable, c.stream));
  auto state = c.state();
  CHECK(!state.valid && state.status == TQC_CC_CAPACITY);
  dout.get(out.data());
  for (auto x : out)
    CHECK(x == -99.f);
  CUDA(cudaGraphExecDestroy(executable));
  CUDA(cudaGraphDestroy(graph));
}
static void alpha_contract() {
  auto invalid = config();
  std::fill_n(invalid.alpha, 3, 0.0);
  tqc_cache_cuda_plan *p = nullptr;
  CHECK(tqc_cache_cuda_create(&invalid, &p) == TQC_D_INVALID);
  CHECK(p == nullptr);
}
int main(int argc, char **argv) {
  try {
    if (argc > 1 && std::strcmp(argv[1], "--bounded") == 0) {
      bounded_calls();
      bounded_query_replay();
      std::printf("%zu bounded-call checks\n", checks);
      return 0;
    }
    bounded_calls();
    bounded_query_replay();
    alpha_contract();
    lifecycle();
    if (argc > 1 && std::strcmp(argv[1], "--lifecycle") == 0) {
      std::printf("%zu lifecycle checks\n", checks);
      return 0;
    }
    for (unsigned id = 1; id <= 5; ++id)
      for (unsigned columns : {128u, 256u})
        physical_and_attention(columns, id, id, 7, 0);
    physical_and_attention(128, 3, 1, 7, 0);
    for (unsigned mask : {1u, 2u, 4u})
      physical_and_attention(128, 3, 1, mask, 0);
    physical_and_attention(128, 3, 1, 7, 1);
    physical_and_attention(256, 3, 1, 7, 1);
    std::printf("%zu checks; %zu independently compared payload bytes; max "
                "Attention error %.9g\n",
                checks, bytes_checked, max_error);
    return 0;
  } catch (const std::exception &e) {
    std::fprintf(stderr, "%s\n", e.what());
    return 1;
  }
}
