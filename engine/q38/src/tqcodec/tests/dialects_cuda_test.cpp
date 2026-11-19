#include "tqcodec/dialects_cuda.h"
#include <array>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cuda_runtime_api.h>
#include <limits>
#include <vector>

static unsigned failures = 0, checks = 0;
static size_t compared_bytes = 0, compared_f32_bytes = 0;
#define CHECK(x)                                                               \
  do {                                                                         \
    ++checks;                                                                  \
    if (!(x)) {                                                                \
      if (failures < 30)                                                       \
        std::fprintf(stderr, "line %d: %s\n", __LINE__, #x);                   \
      ++failures;                                                              \
    }                                                                          \
  } while (0)
#define CUDA(x)                                                                \
  do {                                                                         \
    auto e = (x);                                                              \
    if (e != cudaSuccess) {                                                    \
      std::fprintf(stderr, "CUDA line %d: %s\n", __LINE__,                     \
                   cudaGetErrorString(e));                                     \
      std::exit(2);                                                            \
    }                                                                          \
  } while (0)

// Test-owned allocations and synchronization only; the library owns neither.
struct Device {
  void *src{}, *dst{}, *norms{};
  uint32_t *status{};
  cudaStream_t stream{};
  Device() {
    CUDA(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
    CUDA(cudaMalloc(&src, 4096));
    CUDA(cudaMalloc(&dst, 4096));
    CUDA(cudaMalloc(&norms, 4096));
    CUDA(cudaMalloc(reinterpret_cast<void **>(&status), 4));
  }
  ~Device() {
    CUDA(cudaFree(src));
    CUDA(cudaFree(dst));
    CUDA(cudaFree(norms));
    CUDA(cudaFree(status));
    CUDA(cudaStreamDestroy(stream));
  }
  Device(const Device &) = delete;
  Device &operator=(const Device &) = delete;
  tqc_dialect_cuda_v1_request request(unsigned id, unsigned op, size_t cols,
                                      size_t rows, size_t ss, size_t ds,
                                      size_t ns = 0) {
    return {1,
            sizeof(tqc_dialect_cuda_v1_request),
            id,
            op,
            TQC_D_F32,
            0,
            stream,
            rows,
            cols,
            {src, rows * ss, ss},
            {dst, rows * ds, ds},
            ns ? tqc_dialect_v1_span{norms, rows * ns, ns}
               : tqc_dialect_v1_span{},
            status,
            4};
  }
  uint32_t run(const tqc_dialect_cuda_v1_request &q, const void *input,
               void *output, void *scales = nullptr, uint32_t sticky = 0) {
    CUDA(cudaMemcpyAsync(src, input, q.src.capacity_bytes,
                         cudaMemcpyHostToDevice, stream));
    CUDA(cudaMemsetAsync(dst, 0xa5, q.dst.capacity_bytes, stream));
    CUDA(cudaMemcpyAsync(status, &sticky, 4, cudaMemcpyHostToDevice, stream));
    if (q.norms.data) {
      if (q.operation == TQC_D_PACK)
        CUDA(cudaMemcpyAsync(norms, scales, q.norms.capacity_bytes,
                             cudaMemcpyHostToDevice, stream));
      else
        CUDA(cudaMemsetAsync(norms, 0xa5, q.norms.capacity_bytes, stream));
    }
    CHECK(tqc_dialect_cuda_v1_enqueue(&q) == 0);
    CUDA(cudaMemcpyAsync(output, dst, q.dst.capacity_bytes,
                         cudaMemcpyDeviceToHost, stream));
    if (q.operation == TQC_D_PREPARE)
      CUDA(cudaMemcpyAsync(scales, norms, q.norms.capacity_bytes,
                           cudaMemcpyDeviceToHost, stream));
    uint32_t result = 0;
    CUDA(cudaMemcpyAsync(&result, status, 4, cudaMemcpyDeviceToHost, stream));
    CUDA(cudaStreamSynchronize(stream));
    return result;
  }
};

static void equal_bytes(const void *a, const void *b, size_t n,
                        bool f32 = false) {
  CHECK(std::memcmp(a, b, n) == 0);
  (f32 ? compared_f32_bytes : compared_bytes) += n;
}
static void fixture_file(Device &gpu, const char *path) {
  FILE *f = nullptr;
#ifdef _MSC_VER
  fopen_s(&f, path, "rb");
#else
  f = std::fopen(path, "rb");
#endif
  CHECK(f != nullptr);
  if (!f)
    return;
  unsigned count[5]{}, modes[3]{};
  for (;;) {
    uint32_t id = 0, mode = 0, n = 0, nb = 0;
    float norm = 0;
    if (std::fread(&id, 4, 1, f) != 1)
      break;
    CHECK(std::fread(&mode, 4, 1, f) == 1);
    CHECK(std::fread(&n, 4, 1, f) == 1);
    CHECK(std::fread(&norm, 4, 1, f) == 1);
    if (!n || n > 256 || mode > 2) {
      CHECK(false);
      break;
    }
    std::vector<float> x(n), decoded(n);
    CHECK(std::fread(x.data(), 4, n, f) == n);
    CHECK(std::fread(&nb, 4, 1, f) == 1);
    if (!nb || nb > 196) {
      CHECK(false);
      break;
    }
    std::vector<uint8_t> expected(nb);
    CHECK(std::fread(expected.data(), 1, nb, f) == nb);
    CHECK(std::fread(decoded.data(), 4, n, f) == n);
    if (id > 5)
      continue;
    CHECK(id > 0);
    if (!id)
      break;
    ++count[id - 1];
    ++modes[mode];
    const size_t blockbytes = 18 + 16 * id, xs = n + 4, ps = nb + 7,
                 ns = n / 128 + 3;
    bool finite = true;
    for (size_t b = 0; b < nb; b += blockbytes)
      finite &= (expected[b + 1] & 0x7c) != 0x7c;
    // Repeat every oracle record in two padded rows, including scalar probes
    // and arbitrary encoded values. Padding must survive every operation.
    std::vector<float> input(2 * xs, 999), norms(2 * ns, norm), out(2 * xs);
    std::vector<uint8_t> packed(2 * ps, 0xa5), actual(2 * ps);
    for (size_t row = 0; row < 2; ++row) {
      std::memcpy(input.data() + row * xs, x.data(), n * 4);
      std::memcpy(packed.data() + row * ps, expected.data(), nb);
    }
    if (mode < 2) {
      auto q = gpu.request(id, mode == 1 ? TQC_D_PACK : TQC_D_ENCODE, n, 2,
                           xs * 4, ps, mode == 1 ? ns * 4 : 0);
      const auto flag = gpu.run(q, input.data(), actual.data(), norms.data());
      CHECK(flag == (finite ? 0u : TQC_DC_RANGE));
      if (finite)
        equal_bytes(actual.data(), packed.data(), actual.size());
    }
    auto q = gpu.request(id, TQC_D_DECODE, n, 2, ps, xs * 4);
    const auto flag = gpu.run(q, packed.data(), out.data());
    CHECK(flag == (finite ? 0u : TQC_DC_NONFINITE));
    if (finite)
      for (size_t row = 0; row < 2; ++row) {
        equal_bytes(out.data() + row * xs, decoded.data(), n * 4, true);
        const auto *bytes =
            reinterpret_cast<const uint8_t *>(out.data() + row * xs + n);
        for (size_t i = 0; i < 16; ++i)
          CHECK(bytes[i] == 0xa5);
      }
  }
  CHECK(std::feof(f) != 0);
  std::fclose(f);
  for (auto count_id : count)
    CHECK(count_id > 0);
  std::printf("oracle %s: Turbo2..6 %u/%u/%u/%u/%u; modes %u/%u/%u\n", path,
              count[0], count[1], count[2], count[3], count[4], modes[0],
              modes[1], modes[2]);
}

static void cpu_compare(Device &gpu) {
  for (unsigned id = 1; id <= 5; ++id)
    for (size_t n : {size_t(128), size_t(256)}) {
      const size_t fs = n + 4, ns = n / 128 + 3,
                   ps = (n / 128) * (18 + 16 * id) + 7;
      std::vector<float> input(3 * fs), out(3 * fs), cpu(3 * fs),
          scales(3 * ns), cn(3 * ns);
      for (size_t i = 0; i < input.size(); ++i)
        input[i] = float(int(i * 37 % 211) - 105) * 0.003f;
      input[0] = -0.0f;
      input[1] = std::numeric_limits<float>::denorm_min();
      std::vector<uint8_t> encoded(3 * ps), split(3 * ps);
      for (unsigned op : {TQC_D_FORWARD, TQC_D_INVERSE, TQC_D_PREPARE}) {
        std::memset(cpu.data(), 0xa5, cpu.size() * 4);
        std::memset(cn.data(), 0xa5, cn.size() * 4);
        auto q = gpu.request(id, op, n, 3, fs * 4, fs * 4,
                             op == TQC_D_PREPARE ? ns * 4 : 0);
        tqc_dialect_v1_request r{
            1,
            sizeof(r),
            id,
            op,
            TQC_D_CPU,
            TQC_D_F32,
            nullptr,
            3,
            n,
            {input.data(), input.size() * 4, fs * 4},
            {cpu.data(), cpu.size() * 4, fs * 4},
            op == TQC_D_PREPARE
                ? tqc_dialect_v1_span{cn.data(), cn.size() * 4, ns * 4}
                : tqc_dialect_v1_span{}};
        tqc_dialect_v1_receipt receipt{1, sizeof(receipt), 0, 0, 0};
        CHECK(tqc_dialect_v1_execute(&r, &receipt) == 0);
        CHECK(gpu.run(q, input.data(), out.data(), scales.data()) == 0);
        equal_bytes(cpu.data(), out.data(), cpu.size() * 4, true);
        if (op == TQC_D_PREPARE)
          equal_bytes(cn.data(), scales.data(), cn.size() * 4, true);
      }
      auto q = gpu.request(id, TQC_D_ENCODE, n, 3, fs * 4, ps);
      CHECK(gpu.run(q, input.data(), encoded.data()) == 0);
      q = gpu.request(id, TQC_D_PACK, n, 3, fs * 4, ps, ns * 4);
      CHECK(gpu.run(q, out.data(), split.data(), scales.data()) == 0);
      equal_bytes(encoded.data(), split.data(), split.size());
      // Actual device-to-device pipeline on the explicit stream, with neither
      // host synchronization nor host staging between PREPARE and PACK.
      CUDA(cudaMemcpyAsync(gpu.src, input.data(), input.size() * 4,
                           cudaMemcpyHostToDevice, gpu.stream));
      CUDA(cudaMemsetAsync(gpu.status, 0, 4, gpu.stream));
      auto prep = gpu.request(id, TQC_D_PREPARE, n, 3, fs * 4, fs * 4, ns * 4);
      CHECK(tqc_dialect_cuda_v1_enqueue(&prep) == 0);
      CUDA(cudaMemsetAsync(gpu.src, 0xa5, split.size(), gpu.stream));
      auto pack = q;
      pack.src.data = gpu.dst;
      pack.dst.data = gpu.src;
      CHECK(tqc_dialect_cuda_v1_enqueue(&pack) == 0);
      CUDA(cudaMemcpyAsync(split.data(), gpu.src, split.size(),
                           cudaMemcpyDeviceToHost, gpu.stream));
      uint32_t pipeline_status = 99;
      CUDA(cudaMemcpyAsync(&pipeline_status, gpu.status, 4,
                           cudaMemcpyDeviceToHost, gpu.stream));
      CUDA(cudaStreamSynchronize(gpu.stream));
      CHECK(pipeline_status == 0);
      equal_bytes(encoded.data(), split.data(), split.size());
      // B=-I is an orthogonal inserted factor. PACK must consume B*F(x), never
      // reapply F or normalize. Compare the independently implemented CPU PACK.
      for (auto &v : out)
        v = -v;
      std::vector<uint8_t> cpu_packed(3 * ps, 0xa5);
      tqc_dialect_v1_request r{1,
                               sizeof(r),
                               id,
                               TQC_D_PACK,
                               TQC_D_CPU,
                               TQC_D_F32,
                               nullptr,
                               3,
                               n,
                               {out.data(), out.size() * 4, fs * 4},
                               {cpu_packed.data(), cpu_packed.size(), ps},
                               {scales.data(), scales.size() * 4, ns * 4}};
      tqc_dialect_v1_receipt receipt{1, sizeof(receipt), 0, 0, 0};
      CHECK(tqc_dialect_v1_execute(&r, &receipt) == 0);
      CHECK(gpu.run(q, out.data(), split.data(), scales.data()) == 0);
      equal_bytes(cpu_packed.data(), split.data(), split.size());
    }
}

static void rejection_tests(Device &gpu) {
  std::array<float, 256> x{};
  std::array<uint8_t, 196> p{};
  auto q = gpu.request(5, TQC_D_ENCODE, 256, 1, 1024, 196);
  auto bad = [&](const tqc_dialect_cuda_v1_request &r, int code) {
    CUDA(cudaMemsetAsync(gpu.dst, 0xa5, 196, gpu.stream));
    CUDA(cudaMemsetAsync(gpu.status, 0x5a, 4, gpu.stream));
    const int actual = tqc_dialect_cuda_v1_enqueue(&r);
    if (actual != code)
      std::fprintf(stderr, "host rejection expected %d got %d (op %u)\n", code,
                   actual, r.operation);
    CHECK(actual == code);
    uint32_t status = 0;
    CUDA(cudaMemcpyAsync(p.data(), gpu.dst, 196, cudaMemcpyDeviceToHost,
                         gpu.stream));
    CUDA(cudaMemcpyAsync(&status, gpu.status, 4, cudaMemcpyDeviceToHost,
                         gpu.stream));
    CUDA(cudaStreamSynchronize(gpu.stream));
    for (auto b : p)
      CHECK(b == 0xa5);
    CHECK(status == 0x5a5a5a5a);
  };
  auto r = q;
  r.version = 2;
  bad(r, TQC_D_VERSION);
  r = q;
  --r.size;
  bad(r, TQC_D_VERSION);
  r = q;
  r.dialect = 6;
  bad(r, TQC_D_UNSUPPORTED);
  r = q;
  r.operation = 7;
  bad(r, TQC_D_UNSUPPORTED);
  r = q;
  r.precision = 0;
  bad(r, TQC_D_UNSUPPORTED);
  r = q;
  r.device_ordinal = -1;
  bad(r, TQC_D_UNSUPPORTED);
  r = q;
  r.device_ordinal = 1;
  bad(r, TQC_D_UNSUPPORTED);
  r = q;
  r.rows = 0;
  bad(r, TQC_D_INVALID);
  r = q;
  r.columns = 384;
  bad(r, TQC_D_UNSUPPORTED);
  r = q;
  r.columns = 127;
  bad(r, TQC_D_INVALID);
  r = q;
  r.rows = SIZE_MAX;
  bad(r, TQC_D_OVERFLOW);
  r = q;
  r.columns = SIZE_MAX - 127;
  bad(r, TQC_D_OVERFLOW);
  r = q;
  r.src.data = reinterpret_cast<void *>(UINTPTR_MAX - 7);
  bad(r, TQC_D_OVERFLOW);
  r = q;
  --r.src.capacity_bytes;
  bad(r, TQC_D_CAPACITY);
  r = q;
  --r.dst.capacity_bytes;
  bad(r, TQC_D_CAPACITY);
  r = q;
  r.src.capacity_bytes = SIZE_MAX;
  bad(r, TQC_D_OVERFLOW);
  r = q;
  r.src.data = static_cast<uint8_t *>(gpu.src) + 1;
  bad(r, TQC_D_ALIGNMENT);
  r = q;
  r.dst.data = gpu.src;
  bad(r, TQC_D_OVERLAP);
  r = q;
  r.status = static_cast<uint32_t *>(gpu.dst);
  bad(r, TQC_D_OVERLAP);
  r = q;
  r.status_capacity_bytes = 3;
  bad(r, TQC_D_CAPACITY);
  r = q;
  r.status_capacity_bytes = SIZE_MAX;
  bad(r, TQC_D_OVERFLOW);
  r = q;
  r.status = reinterpret_cast<uint32_t *>(static_cast<uint8_t *>(gpu.src) + 1);
  bad(r, TQC_D_ALIGNMENT);
  r = q;
  r.stream = nullptr;
  bad(r, TQC_D_UNSUPPORTED);
  r = q;
  r.stream = reinterpret_cast<void *>(1);
  bad(r, TQC_D_UNSUPPORTED);
  r = q;
  r.stream = reinterpret_cast<void *>(2);
  bad(r, TQC_D_UNSUPPORTED);
  r = q;
  r.norms = {gpu.norms, 4, 4};
  bad(r, TQC_D_INVALID);
  r = q;
  r.dst.data = &r;
  bad(r, TQC_D_OVERLAP);
  r = q;
  r.operation = TQC_D_PACK;
  r.norms = {gpu.norms, 8, 8};
  auto s = r;
  s.norms.capacity_bytes = 4;
  bad(s, TQC_D_CAPACITY);
  s = r;
  s.norms.data = gpu.src;
  bad(s, TQC_D_OVERLAP);
  s = r;
  s.norms.data = static_cast<uint8_t *>(gpu.norms) + 1;
  bad(s, TQC_D_ALIGNMENT);
  s = r;
  s.norms.row_stride_bytes = 7;
  bad(s, TQC_D_CAPACITY);
  s = r;
  s.rows = 2;
  s.src.capacity_bytes = 2048;
  s.dst.capacity_bytes = 392;
  s.norms.capacity_bytes = 20;
  s.norms.row_stride_bytes = 9;
  bad(s, TQC_D_ALIGNMENT);
  for (unsigned id = 1; id <= 5; ++id) {
    const size_t bytes = 2 * (18 + 16 * id);
    q = gpu.request(id, TQC_D_ENCODE, 256, 1, 1024, bytes);
    x.fill(0);
    CHECK(gpu.run(q, x.data(), p.data()) == 0);
    x.fill(-0.0f);
    CHECK(gpu.run(q, x.data(), p.data()) == 0);
    x.fill(std::numeric_limits<float>::denorm_min());
    CHECK(gpu.run(q, x.data(), p.data()) == 0);
    x.fill(0);
    x[255] = std::numeric_limits<float>::quiet_NaN();
    CHECK(gpu.run(q, x.data(), p.data()) == TQC_DC_NONFINITE);
    // Status from that actual failed call remains sticky across the next
    // successful-data call; the library must not silently reset it.
    x.fill(0);
    CUDA(cudaMemcpyAsync(gpu.src, x.data(), 1024, cudaMemcpyHostToDevice,
                         gpu.stream));
    CUDA(cudaMemsetAsync(gpu.dst, 0xa5, bytes, gpu.stream));
    CHECK(tqc_dialect_cuda_v1_enqueue(&q) == 0);
    uint32_t sticky = 0;
    CUDA(cudaMemcpyAsync(&sticky, gpu.status, 4, cudaMemcpyDeviceToHost,
                         gpu.stream));
    CUDA(cudaMemcpyAsync(p.data(), gpu.dst, bytes, cudaMemcpyDeviceToHost,
                         gpu.stream));
    CUDA(cudaStreamSynchronize(gpu.stream));
    CHECK(sticky == TQC_DC_NONFINITE);
    for (size_t i = 0; i < bytes; ++i)
      CHECK(p[i] == 0xa5);
    x[255] = std::numeric_limits<float>::infinity();
    CHECK(gpu.run(q, x.data(), p.data()) == TQC_DC_NONFINITE);
    x[255] = 65505;
    CHECK(gpu.run(q, x.data(), p.data()) == TQC_DC_RANGE);
    x.fill(65504);
    CHECK(gpu.run(q, x.data(), p.data()) == TQC_DC_RANGE);
    x.fill(0);
    CHECK(gpu.run(q, x.data(), p.data(), nullptr, 0x80) == 0x80);
    for (size_t i = 0; i < bytes; ++i)
      CHECK(p[i] == 0xa5);
    std::array<float, 2> norms{1, 1};
    q = gpu.request(id, TQC_D_PACK, 256, 1, 1024, bytes, 8);
    for (float v : {-1.0f, std::numeric_limits<float>::max()}) {
      norms[1] = v;
      CHECK(gpu.run(q, x.data(), p.data(), norms.data()) == TQC_DC_RANGE);
    }
    norms[1] = std::numeric_limits<float>::quiet_NaN();
    CHECK(gpu.run(q, x.data(), p.data(), norms.data()) == TQC_DC_NONFINITE);
    q = gpu.request(id, TQC_D_DECODE, 256, 1, bytes, 1024);
    for (uint8_t high : {uint8_t(0x7c), uint8_t(0x7e), uint8_t(0xfc)}) {
      p.fill(0);
      p[bytes / 2 + 1] = high;
      CHECK(gpu.run(q, p.data(), x.data()) == TQC_DC_NONFINITE);
    }
  }
  CHECK(tqc_dialect_cuda_v1_enqueue(nullptr) == TQC_D_INVALID);
  alignas(16) uint8_t unaligned[sizeof(q) + 16]{};
  CHECK(tqc_dialect_cuda_v1_enqueue(
            reinterpret_cast<const tqc_dialect_cuda_v1_request *>(
                unaligned + 1)) == TQC_D_ALIGNMENT);
  tqc_dialect_cuda_v1_info info{1, sizeof(info), 0,      0, 0, 0, 0, 0, 0,
                                0, nullptr,      nullptr};
  for (unsigned id = 1; id <= 5; ++id) {
    tqc_dialect_v1_info cpu{};
    cpu.version = 1;
    cpu.size = sizeof(cpu);
    CHECK(tqc_dialect_v1_query(id, &cpu) == 0);
    CHECK(tqc_dialect_cuda_v1_query(id, &info) == 0);
    CHECK(std::strcmp(cpu.identity, info.identity) == 0);
  }
  auto before = info;
  info.version = 2;
  auto invalid = info;
  CHECK(tqc_dialect_cuda_v1_query(1, &info) == TQC_D_VERSION);
  equal_bytes(&invalid, &info, sizeof(info));
  info = before;
  CHECK(tqc_dialect_cuda_v1_query(0, &info) == TQC_D_UNSUPPORTED);
  equal_bytes(&before, &info, sizeof(info));
  CHECK(tqc_dialect_cuda_v1_query(1, nullptr) == TQC_D_INVALID);
}

int main(int argc, char **argv) {
  int count = 0;
  CUDA(cudaGetDeviceCount(&count));
  CHECK(count > 0);
  CUDA(cudaSetDevice(0));
  cudaDeviceProp prop{};
  CUDA(cudaGetDeviceProperties(&prop, 0));
  CHECK(prop.major == 12 && prop.minor == 0);
  std::printf("device %s compute %d.%d\n", prop.name, prop.major, prop.minor);
  Device gpu;
  rejection_tests(gpu);
  cpu_compare(gpu);
  for (int i = 1; i < argc; ++i)
    fixture_file(gpu, argv[i]);
  // A second nondefault stream has separate status and allocations.
  {
    Device second;
    cpu_compare(second);
  }
  std::printf("checks %u failures %u; compared payload/padding bytes %zu; "
              "F32/padding bytes %zu; required bit error 0\n",
              checks, failures, compared_bytes, compared_f32_bytes);
  return failures ? 1 : 0;
}
