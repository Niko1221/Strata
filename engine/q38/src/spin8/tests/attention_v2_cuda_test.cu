#include "spin8/attention_v2_cuda.h"
#include <algorithm>
#include <array>
#include <cfloat>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <limits>
#include <vector>
#ifdef _WIN32
#define NOMINMAX
#include <windows.h>
#endif

// Predeclared before any GPU run: absolute error <= 2e-5 for BOTH storage
// formats, against scalar FP64 arithmetic on the actual rounded inputs.
constexpr double tolerance = 2e-5;
static void check(bool ok, const char *message) {
  if (!ok) {
    std::fprintf(stderr, "FAIL: %s\n", message);
    std::exit(1);
  }
}
static void cuda_ok(cudaError_t e) {
  if (e != cudaSuccess) {
    std::fprintf(stderr, "CUDA: %s\n", cudaGetErrorString(e));
    std::exit(1);
  }
}
static void ok(spin8_attention_v2_cuda_result r) {
  if (r.status != SPIN8_OK)
    std::fprintf(stderr, "Spin8: %s; CUDA: %s\n", spin8_status_string(r.status),
                 cudaGetErrorString(r.cuda_error));
  check(r.status == SPIN8_OK, "Spin8 host status");
  cuda_ok(r.cuda_error);
}
static void resources() {
#ifdef _WIN32
  MEMORYSTATUSEX memory{};
  memory.dwLength = sizeof(memory);
  check(GlobalMemoryStatusEx(&memory) != 0, "physical RAM query");
  if (memory.ullAvailPhys < (uint64_t(3) << 30)) {
    std::puts("BLOCKED: available physical RAM below 3 GiB");
    std::exit(77);
  }
#endif
  size_t free_bytes = 0, total = 0;
  cuda_ok(cudaMemGetInfo(&free_bytes, &total));
  if (free_bytes < size_t(1) << 30) {
    std::puts("BLOCKED: GPU free below 1 GiB");
    std::exit(77);
  }
}
static void identity_attention() {
  resources();
  cudaStream_t stream;
  cuda_ok(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
  uint32_t map = 0;
  spin8_attention_v2_config c{};
  c.abi_version = SPIN8_ATTENTION_V2_ABI_VERSION;
  c.struct_bytes = sizeof(c);
  c.semantics = SPIN8_ATTENTION_V2_SELF;
  c.dtype = SPIN8_F32;
  c.source_id = 42;
  c.n_q_heads = c.n_kv_heads = 1;
  c.d_qk = 2;
  c.d_v = 3;
  c.q_to_kv = &map;
  c.q_to_kv_bytes = sizeof(map);
  spin8_attention_v2_cuda_plan *p = nullptr;
  resources();
  ok(spin8_attention_v2_cuda_plan_create(&c, sizeof(c), 0, &p));
  size_t workspace_bytes = 0;
  ok(spin8_attention_v2_cuda_workspace_bytes(p, 1, 2, &workspace_bytes));
  check(workspace_bytes < 256 * 1024 * 1024, "bounded workspace");
  void *workspace = nullptr;
  float *data = nullptr;
  uint32_t *status = nullptr;
  unsigned char *mask = nullptr;
  resources();
  cuda_ok(cudaMalloc(&workspace, workspace_bytes));
  resources();
  cuda_ok(cudaMalloc(&data, 15 * sizeof(float)));
  resources();
  cuda_ok(cudaMalloc(&status, sizeof(*status)));
  resources();
  cuda_ok(cudaMalloc(&mask, 2));
  float host[] = {1, 0, 1, 0, 0, 1, 1, 2, 3, 5, 6, 7, -99, -99, -99};
  cuda_ok(cudaMemcpyAsync(data, host, sizeof(host), cudaMemcpyHostToDevice,
                          stream));
  cuda_ok(cudaMemsetAsync(mask, 1, 2, stream));
  spin8_attention_v2_cuda_request r{};
  r.abi_version = SPIN8_ATTENTION_V2_CUDA_ABI_VERSION;
  r.struct_bytes = sizeof(r);
  r.operation = SPIN8_ATTENTION_V2_CUDA_REFERENCE;
  r.k_dtype = r.v_dtype = SPIN8_F32;
  r.stream = stream;
  r.workspace = workspace;
  r.workspace_bytes = workspace_bytes;
  r.device_status = status;
  r.device_status_bytes = sizeof(*status);
  auto &a = r.attention;
  a.abi_version = SPIN8_ATTENTION_V2_ABI_VERSION;
  a.struct_bytes = sizeof(a);
  a.position_stage = SPIN8_ATTENTION_V2_POST_POSITION;
  a.q_source_id = a.kv_source_id = 42;
  a.n_query = 1;
  a.n_key = 2;
  a.scale = .7;
  a.q = {data, 8, 8, 8, 4};
  a.k = {data + 2, 16, 8, 8, 4};
  a.v = {data + 6, 24, 12, 12, 4};
  a.mask = {mask, 2, 2, 2, 1};
  a.output = {data + 12, 12, 12, 12, 4};
  ok(spin8_attention_v2_cuda_enqueue(p, &r, sizeof(r)));
  uint32_t flag = 99;
  cuda_ok(cudaMemcpyAsync(&flag, status, sizeof(flag), cudaMemcpyDeviceToHost,
                          stream));
  cuda_ok(cudaMemcpyAsync(host, data, sizeof(host), cudaMemcpyDeviceToHost,
                          stream));
  cuda_ok(cudaStreamSynchronize(stream));
  check(flag == 0, "device completion status");
  const double prob = std::exp(.7) / (std::exp(.7) + 1);
  for (size_t i = 0; i < 3; ++i)
    check(std::abs(host[12 + i] - (prob * host[6 + i] +
                                   (1 - prob) * host[9 + i])) <= tolerance,
          "actual identity kernel vs FP64 oracle");
  ok(spin8_attention_v2_cuda_plan_free(p));
  cuda_ok(cudaFree(workspace));
  cuda_ok(cudaFree(data));
  cuda_ok(cudaFree(status));
  cuda_ok(cudaFree(mask));
  cuda_ok(cudaStreamDestroy(stream));
  std::puts("PASS attention_v2 CUDA identity");
}

static size_t live_bytes = 0, peak_bytes = 0, cases = 0;
static double max_f32 = 0, max_f16 = 0, max_transform = 0;
struct Device {
  void *data = nullptr;
  size_t bytes = 0;
  explicit Device(size_t n = 0) {
    if (n)
      allocate(n);
  }
  Device(const Device &) = delete;
  Device &operator=(const Device &) = delete;
  void allocate(size_t n) {
    check(!data, "single device allocation ownership");
    resources();
    check(n < 256 * 1024 * 1024 && live_bytes + n < 256 * 1024 * 1024,
          "aggregate test device allocations below 256 MiB");
    cuda_ok(cudaMalloc(&data, n));
    bytes = n;
    live_bytes += n;
    peak_bytes = std::max(peak_bytes, live_bytes);
  }
  ~Device() {
    if (data) {
      cuda_ok(cudaFree(data));
      live_bytes -= bytes;
    }
  }
};
struct Buffer {
  size_t nt, nh, nc, item, ts, hs, cs, capacity;
  uint32_t dtype;
  std::vector<unsigned char> host;
  Device device;
  Buffer(size_t t, size_t h, size_t c, uint32_t type = SPIN8_F32,
         bool interleaved = false, bool mask = false)
      : nt(t), nh(h), nc(c), item(mask                ? 1
                                  : type == SPIN8_F16 ? 2
                                                      : 4),
        dtype(type) {
    cs = item * 2;
    hs = c * cs + item * 3;
    if (interleaved) {
      hs = item * 2;
      cs = h * hs + item * 3;
    }
    ts = (h - 1) * hs + (c - 1) * cs + item * 5;
    capacity = (t - 1) * ts + (h - 1) * hs + (c - 1) * cs + item;
    host.assign(capacity + 32, 0xa5);
    device.allocate(host.size());
  }
  size_t at(size_t t, size_t h, size_t c) const {
    return 16 + t * ts + h * hs + c * cs;
  }
  void put(size_t t, size_t h, size_t c, double x) {
    const size_t pos = at(t, h, c);
    if (item == 1) {
      host[pos] = x != 0 ? 7 : 0;
    } else if (dtype == SPIN8_F16) {
      const __half value = __float2half_rn(float(x));
      std::memcpy(host.data() + pos, &value, 2);
    } else {
      const float value = float(x);
      std::memcpy(host.data() + pos, &value, 4);
    }
  }
  double get(size_t t, size_t h, size_t c) const {
    const size_t pos = at(t, h, c);
    if (item == 1)
      return host[pos];
    if (dtype == SPIN8_F16) {
      __half value;
      std::memcpy(&value, host.data() + pos, 2);
      return __half2float(value);
    }
    float value;
    std::memcpy(&value, host.data() + pos, 4);
    return value;
  }
  spin8_attention_v2_input input() const {
    return {static_cast<char *>(device.data) + 16, capacity, ts, hs, cs};
  }
  spin8_attention_v2_output output() const {
    return {static_cast<char *>(device.data) + 16, capacity, ts, hs, cs};
  }
  void upload(cudaStream_t stream) const {
    cuda_ok(cudaMemcpyAsync(device.data, host.data(), host.size(),
                            cudaMemcpyHostToDevice, stream));
  }
  std::vector<unsigned char> download(cudaStream_t stream) const {
    std::vector<unsigned char> bytes(host.size());
    cuda_ok(cudaMemcpyAsync(bytes.data(), device.data, bytes.size(),
                            cudaMemcpyDeviceToHost, stream));
    cuda_ok(cudaStreamSynchronize(stream));
    return bytes;
  }
  void guards() const {
    std::vector<bool> payload(host.size());
    for (size_t t = 0; t < nt; ++t)
      for (size_t h = 0; h < nh; ++h)
        for (size_t c = 0; c < nc; ++c)
          for (size_t j = 0; j < item; ++j)
            payload[at(t, h, c) + j] = true;
    for (size_t i = 0; i < host.size(); ++i)
      check(payload[i] || host[i] == 0xa5, "output padding/guards unchanged");
  }
};
struct Fixture {
  Buffer q, k, v, mask, bias, gate, out;
  std::vector<uint32_t> map;
  spin8_plane planes[3]{{0, 2, .73}, {4, 7, -.38}, {1, 6, 1.23}};
  spin8_attention_v2_config config{};
  spin8_attention_v2_cuda_request request{};
  spin8_attention_v2_cuda_plan *plan = nullptr;
  size_t plan_bytes = 0;
  cudaStream_t stream = nullptr;
  Device workspace, status{4};
  Fixture(size_t dq, size_t dv, size_t heads, size_t kvheads,
          uint32_t kt = SPIN8_F32, uint32_t vt = SPIN8_F32)
      : q(3, heads, dq), k(5, kvheads, dq, kt, true), v(5, kvheads, dv, vt),
        mask(3, heads, 5, SPIN8_F32, true, true), bias(3, heads, 5),
        gate(3, heads, dv, SPIN8_F32, true), out(3, heads, dv, SPIN8_F32, true),
        map(heads) {
    cuda_ok(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
    for (size_t h = 0; h < heads; ++h)
      map[h] = uint32_t((2 * h + kvheads - 1) % kvheads);
    config.abi_version = SPIN8_ATTENTION_V2_ABI_VERSION;
    config.struct_bytes = sizeof(config);
    config.semantics = SPIN8_ATTENTION_V2_SELF;
    config.dtype = SPIN8_F32;
    config.source_id = 471;
    config.n_q_heads = heads;
    config.n_kv_heads = kvheads;
    config.d_qk = dq;
    config.d_v = dv;
    config.q_to_kv = map.data();
    config.q_to_kv_bytes = map.size() * 4;
    request.abi_version = SPIN8_ATTENTION_V2_CUDA_ABI_VERSION;
    request.struct_bytes = sizeof(request);
    request.operation = SPIN8_ATTENTION_V2_CUDA_REFERENCE;
    request.k_dtype = kt;
    request.v_dtype = vt;
    request.stream = stream;
    request.device_status = static_cast<uint32_t *>(status.data);
    request.device_status_bytes = 4;
    auto &a = request.attention;
    a.abi_version = SPIN8_ATTENTION_V2_ABI_VERSION;
    a.struct_bytes = sizeof(a);
    a.position_stage = SPIN8_ATTENTION_V2_POST_POSITION;
    a.q_source_id = a.kv_source_id = config.source_id;
    a.n_query = 3;
    a.n_key = 5;
    a.scale = .37;
    a.softcap = 1.3;
    a.q = q.input();
    a.k = k.input();
    a.v = v.input();
    a.mask = mask.input();
    a.bias = bias.input();
    a.gate = gate.input();
    a.output = out.output();
    for (Buffer *b : {&q, &k, &v, &gate})
      for (size_t t = 0; t < b->nt; ++t)
        for (size_t h = 0; h < b->nh; ++h)
          for (size_t c = 0; c < b->nc; ++c)
            b->put(t, h, c,
                   std::sin(.13 * double(1 + t * 71 + h * 31 + c * 7)) +
                       .17 * double(h + t));
    for (size_t t = 0; t < q.nt; ++t)
      for (size_t h = 0; h < heads; ++h)
        for (size_t j = 0; j < k.nt; ++j) {
          mask.put(t, h, j, j == 0 || (t + h + j) % 3 != 0);
          bias.put(t, h, j, .6 * double(j) - 1.1);
        }
  }
  ~Fixture() {
    cuda_ok(cudaStreamSynchronize(stream));
    ok(spin8_attention_v2_cuda_plan_free(plan));
    live_bytes -= plan_bytes;
    cuda_ok(cudaStreamDestroy(stream));
  }
  void fixed(uint32_t view, bool h128) {
    config.basis = SPIN8_ATTENTION_V2_FIXED;
    config.view = view;
    config.hadamard_block = h128 ? 128 : 0;
    config.planes = planes;
    config.plane_count = 3;
    config.planes_bytes = sizeof(planes);
  }
  void make() {
    resources();
    ok(spin8_attention_v2_cuda_plan_create(&config, sizeof(config), 0, &plan));
    plan_bytes = 512 + 4 * config.n_q_heads;
    live_bytes += plan_bytes;
    peak_bytes = std::max(peak_bytes, live_bytes);
    check(live_bytes < 256 * 1024 * 1024,
          "aggregate includes plan coefficients and mapping");
    size_t bytes = 0;
    ok(spin8_attention_v2_cuda_workspace_bytes(plan, q.nt, k.nt, &bytes));
    const size_t expected =
        8 * (q.nt * q.nh * q.nc + k.nt * k.nh * k.nc + v.nt * v.nh * v.nc +
             out.nt * out.nh * out.nc + q.nt * q.nh * k.nt);
    check(
        bytes == expected,
        "workspace includes all packed tensors, full scores and staged output");
    workspace.allocate(bytes);
    request.workspace = workspace.data;
    request.workspace_bytes = bytes;
  }
  void upload() {
    for (const Buffer *b : {&q, &k, &v, &mask, &bias, &gate, &out})
      b->upload(stream);
  }
  // Scalar original-space oracle. No production attention/transform is called.
  std::vector<double> oracle() const {
    const auto &a = request.attention;
    std::vector<double> result(q.nt * q.nh * v.nc);
    for (size_t t = 0; t < q.nt; ++t)
      for (size_t h = 0; h < q.nh; ++h) {
        std::vector<double> scores(k.nt);
        double maxval = -std::numeric_limits<double>::infinity();
        for (size_t j = 0; j < k.nt; ++j) {
          double dot = 0;
          for (size_t c = 0; c < q.nc; ++c)
            dot += q.get(t, h, c) * k.get(j, map[h], c);
          double s = a.scale * dot, b = a.bias.data ? bias.get(t, h, j) : 0;
          if (a.logit_order == 0)
            s += b;
          if (a.softcap > 0)
            s = a.softcap * std::tanh(s / a.softcap);
          if (a.logit_order == 1)
            s += b;
          scores[j] = s;
          if (mask.get(t, h, j))
            maxval = std::max(maxval, s);
        }
        double denom = 0;
        for (size_t j = 0; j < k.nt; ++j) {
          scores[j] = mask.get(t, h, j) ? std::exp(scores[j] - maxval) : 0;
          denom += scores[j];
        }
        for (size_t c = 0; c < v.nc; ++c) {
          double value = 0;
          for (size_t j = 0; j < k.nt; ++j)
            value += scores[j] / denom * v.get(j, map[h], c);
          if (a.gate.data)
            value *= gate.get(t, h, c);
          result[(t * q.nh + h) * v.nc + c] = value;
        }
      }
    return result;
  }
  uint32_t finish() {
    uint32_t flag = 99;
    cuda_ok(
        cudaMemcpyAsync(&flag, status.data, 4, cudaMemcpyDeviceToHost, stream));
    cuda_ok(cudaStreamSynchronize(stream));
    return flag;
  }
  void verify() {
    check(finish() == 0, "successful completed device status");
    const auto expected = oracle();
    out.host = out.download(stream);
    double &worst = request.k_dtype == SPIN8_F16 || request.v_dtype == SPIN8_F16
                        ? max_f16
                        : max_f32;
    for (size_t t = 0; t < q.nt; ++t)
      for (size_t h = 0; h < q.nh; ++h)
        for (size_t c = 0; c < v.nc; ++c) {
          const double error =
              std::abs(out.get(t, h, c) - expected[(t * q.nh + h) * v.nc + c]);
          check(std::isfinite(error) && error <= tolerance,
                "FP64 rounded-input oracle <=2e-5");
          worst = std::max(worst, error);
        }
    out.guards();
    for (const Buffer *b : {&q, &k, &v, &mask, &bias, &gate})
      check(b->host == b->download(stream), "input bytes unchanged");
    ++cases;
  }
  void compare() {
    upload();
    ok(spin8_attention_v2_cuda_enqueue(plan, &request, sizeof(request)));
    verify();
  }
};

static void geometry_cases() {
  const size_t widths[] = {64, 80, 96, 128, 130, 192, 256};
  const size_t values[] = {130, 96, 80, 256, 192, 64, 128};
  for (size_t i = 0; i < 7; ++i)
    for (size_t topology = 0; topology < 3; ++topology)
      for (uint32_t kt : {uint32_t(SPIN8_F32), uint32_t(SPIN8_F16)})
        for (uint32_t vt : {uint32_t(SPIN8_F32), uint32_t(SPIN8_F16)}) {
          Fixture f(widths[i], values[i], topology == 0 ? 3 : 5,
                    topology == 2 ? 1 : 3, kt, vt);
          f.make();
          for (uint32_t order = 0; order < 2; ++order) {
            f.request.attention.logit_order = order;
            f.compare();
            f.request.attention.softcap = 0;
            f.compare();
            f.request.attention.softcap = 1.3;
          }
        }
  std::puts("PASS identity geometry: widths, MHA/GQA/MQA, independent K/V "
            "F32/F16 and both softcap orders");
}

static void fixed_cases() {
  std::puts("BEGIN actual fixed-basis cases");
  const size_t widths[] = {64, 80, 96, 128, 130, 192, 256};
  for (size_t i = 0; i < 7; ++i)
    for (uint32_t view = 0; view < 3; ++view)
      for (bool h128 : {false, true}) {
        Fixture f(widths[i], widths[(i + 3) % 7], 5, 3,
                  view == 1 ? SPIN8_F16 : SPIN8_F32,
                  view == 2 ? SPIN8_F16 : SPIN8_F32);
        f.fixed(view, h128);
        f.make();
        const auto saved_map = f.map;
        std::fill(f.map.begin(), f.map.end(), UINT32_MAX);
        f.planes[0].angle = std::numeric_limits<double>::quiet_NaN();
        f.upload();
        ok(spin8_attention_v2_cuda_enqueue(f.plan, &f.request,
                                           sizeof(f.request)));
        f.map = saved_map;
        f.verify();
        f.request.attention.logit_order = 1;
        f.compare();
      }
}

static std::vector<double> dense_basis(std::vector<double> row,
                                       const std::array<double, 64> &matrix,
                                       bool inverse) {
  auto h128 = [&] {
    const auto input = row;
    for (size_t base = 0; base + 128 <= row.size(); base += 128)
      for (size_t i = 0; i < 128; ++i) {
        double value = 0;
        for (size_t j = 0; j < 128; ++j) {
          size_t bits = i & j;
          int parity = 0;
          while (bits) {
            parity ^= 1;
            bits &= bits - 1;
          }
          value += input[base + j] * (parity ? -1. : 1.);
        }
        row[base + i] = value / std::sqrt(128.);
      }
  };
  if (!inverse)
    h128();
  const auto input = row;
  for (size_t base = 0; base + 8 <= row.size(); base += 8)
    for (size_t i = 0; i < 8; ++i) {
      double value = 0;
      for (size_t j = 0; j < 8; ++j)
        value += matrix[inverse ? j * 8 + i : i * 8 + j] * input[base + j];
      row[base + i] = value;
    }
  if (inverse)
    h128();
  return row;
}
static double difference(const std::vector<double> &a,
                         const std::vector<double> &b) {
  check(a.size() == b.size(), "oracle size");
  double error = 0;
  for (size_t i = 0; i < a.size(); ++i)
    error = std::max(error, std::abs(a[i] - b[i]));
  return error;
}
static void preallocated_first_use() {
  double worst_staging = 0;
  for (size_t repeat = 0; repeat < 8; ++repeat)
    for (uint32_t mode = 0; mode < 4; ++mode) {
      Fixture f(130, 80, 5, 3, repeat & 1 ? SPIN8_F16 : SPIN8_F32,
                repeat & 2 ? SPIN8_F16 : SPIN8_F32);
      for (size_t h = 0; h < f.map.size(); ++h)
        f.map[h] = uint32_t((2 * h + repeat + mode) % f.k.nh);
      if (mode) {
        f.fixed(mode - 1, true);
        f.planes[0].angle += .03 * double(repeat);
      }
      // Preallocate ALL invocation storage and prepare the nonblocking stream
      // before creation. Fixture::make()/compare() would introduce allocations
      // or uploads after creation and could hide a pending plan-upload race.
      const size_t bytes =
          8 * (f.q.nt * f.q.nh * f.q.nc + f.k.nt * f.k.nh * f.k.nc +
               f.v.nt * f.v.nh * f.v.nc + f.out.nt * f.out.nh * f.out.nc +
               f.q.nt * f.q.nh * f.k.nt);
      f.workspace.allocate(bytes);
      f.request.workspace = f.workspace.data;
      f.request.workspace_bytes = bytes;
      std::vector<double> expected_q(f.q.nc), actual_q(f.q.nc);
      for (size_t c = 0; c < f.q.nc; ++c)
        expected_q[c] = f.q.get(0, 0, c);
      if (mode) {
        spin8_plan *cpu = nullptr;
        check(spin8_plan_create(f.planes, 3, &cpu) == SPIN8_OK,
              "first-use audited matrix source");
        std::array<double, 64> matrix{};
        check(spin8_plan_matrix(cpu, static_cast<spin8_view>(mode - 1),
                                matrix.data()) == SPIN8_OK,
              "first-use matrix export");
        spin8_plan_free(cpu);
        expected_q = dense_basis(expected_q, matrix, false);
      }
      unsigned flags = 0;
      cuda_ok(cudaStreamGetFlags(f.stream, &flags));
      check(flags == cudaStreamNonBlocking, "first-use nonblocking stream");
      f.upload();
      cuda_ok(cudaStreamSynchronize(f.stream));
      resources();

      ok(spin8_attention_v2_cuda_plan_create(&f.config, sizeof(f.config), 0,
                                             &f.plan));
      // Deliberately adjacent: no caller allocation, copy, resource query,
      // event, wait or synchronization between publication and first use.
      ok(spin8_attention_v2_cuda_enqueue(f.plan, &f.request,
                                         sizeof(f.request)));

      f.plan_bytes = 512 + 4 * f.config.n_q_heads;
      live_bytes += f.plan_bytes;
      peak_bytes = std::max(peak_bytes, live_bytes);
      check(live_bytes < 256 * 1024 * 1024, "first-use bounded allocations");
      f.verify();
      cuda_ok(cudaMemcpy(actual_q.data(), f.workspace.data,
                         actual_q.size() * sizeof(double),
                         cudaMemcpyDeviceToHost));
      for (size_t c = 0; c < actual_q.size(); ++c) {
        const double error = std::abs(actual_q[c] - expected_q[c]);
        check(std::isfinite(error) && error <= 1e-11,
              "first-use published matrix vs dense Q staging oracle");
        worst_staging = std::max(worst_staging, error);
      }
    }
  std::printf("PASS preallocated immediate nonblocking first use: 32 plans; "
              "max Q staging error %.17g\n",
              worst_staging);
}
static void transforms_controls() {
  Fixture f(130, 130, 5, 3, SPIN8_F16, SPIN8_F32);
  f.fixed(1, true);
  f.make();
  f.compare();
  spin8_plan *spin = nullptr;
  check(spin8_plan_create(f.planes, 3, &spin) == SPIN8_OK,
        "test audited coefficient source");
  std::array<double, 64> matrix{};
  check(spin8_plan_matrix(spin, SPIN8_SPIN_PLUS, matrix.data()) == SPIN8_OK,
        "test audited matrix");
  spin8_plan_free(spin);
  // Original-space attention alone cannot prove that an orthogonal transform
  // actually ran. Check REAL device Q/K/V staging against independent dense
  // matrix and Sylvester-parity transforms, including identity tails.
  std::vector<double> work(f.workspace.bytes / 8);
  cuda_ok(cudaMemcpy(work.data(), f.workspace.data, f.workspace.bytes,
                     cudaMemcpyDeviceToHost));
  size_t start = 0;
  double transformed_change = 0;
  for (const Buffer *b : {&f.q, &f.k, &f.v}) {
    for (size_t t = 0; t < b->nt; ++t)
      for (size_t h = 0; h < b->nh; ++h) {
        std::vector<double> row(b->nc);
        for (size_t c = 0; c < b->nc; ++c)
          row[c] = b->get(t, h, c);
        const auto expected = dense_basis(row, matrix, false);
        transformed_change =
            std::max(transformed_change, difference(row, expected));
        for (size_t c = 0; c < b->nc; ++c) {
          const double error =
              std::abs(work[start + (t * b->nh + h) * b->nc + c] - expected[c]);
          check(std::isfinite(error) && error <= 1e-11,
                "actual GPU transform vs independent dense oracle");
          max_transform = std::max(max_transform, error);
        }
      }
    start += b->nt * b->nh * b->nc;
  }
  check(transformed_change > .1, "transform witness is not identity");
  const auto expected = f.oracle();
  const double original_scale = f.request.attention.scale;
  f.request.attention.scale = 1 / std::sqrt(double(f.q.nc));
  const double scale_error = difference(expected, f.oracle());
  check(scale_error > 1e-4, "wrong original scale negative control diverges");
  f.request.attention.scale = original_scale;
  f.request.attention.gate = {};
  const auto ungated = f.oracle();
  f.request.attention.gate = f.gate.input();
  std::vector<double> wrong(ungated.size());
  for (size_t t = 0; t < f.q.nt; ++t)
    for (size_t h = 0; h < f.q.nh; ++h) {
      const size_t at = (t * f.q.nh + h) * f.v.nc;
      std::vector<double> row(ungated.begin() + at,
                              ungated.begin() + at + f.v.nc);
      row = dense_basis(row, matrix, false);
      for (size_t c = 0; c < f.v.nc; ++c)
        row[c] *= f.gate.get(t, h, c);
      row = dense_basis(row, matrix, true);
      std::copy(row.begin(), row.end(), wrong.begin() + at);
    }
  const double gate_error = difference(expected, wrong);
  check(gate_error > 1e-4, "gate-before-inverse negative control diverges");
  f.compare();
  std::printf("Negative controls: wrong original scale %.9g; "
              "gate-before-inverse %.9g; actual transform change %.9g\n",
              scale_error, gate_error, transformed_change);
  f.request.attention.bias = {};
  f.request.attention.gate = {};
  f.request.attention.scale = -.2;
  f.compare();
  f.request.attention.scale = 0;
  f.compare();
}

static void host_validation() {
  Fixture f(8, 9, 5, 3);
  f.make();
  f.upload();
  cuda_ok(cudaMemsetAsync(f.workspace.data, 0x5a, f.workspace.bytes, f.stream));
  cuda_ok(cudaMemsetAsync(f.status.data, 0x5a, 4, f.stream));
  cuda_ok(cudaStreamSynchronize(f.stream));
  auto bad = [&](spin8_attention_v2_cuda_request r,
                 spin8_status expected = SPIN8_INVALID_ARGUMENT,
                 size_t bytes = sizeof(spin8_attention_v2_cuda_request)) {
    auto result = spin8_attention_v2_cuda_enqueue(f.plan, &r, bytes);
    check(result.status == expected && result.cuda_error == cudaSuccess,
          "host rejection status");
    check(f.finish() == 0x5a5a5a5a,
          "host rejection leaves device status unchanged");
    check(f.out.host == f.out.download(f.stream),
          "host rejection leaves output unchanged");
    std::vector<unsigned char> work(f.workspace.bytes);
    cuda_ok(cudaMemcpy(work.data(), f.workspace.data, work.size(),
                       cudaMemcpyDeviceToHost));
    check(std::all_of(work.begin(), work.end(),
                      [](unsigned char x) { return x == 0x5a; }),
          "host rejection leaves workspace unchanged");
    ++cases;
  };
  auto r = f.request;
  r.abi_version++;
  bad(r, SPIN8_UNSUPPORTED);
  r = f.request;
  r.struct_bytes--;
  bad(r);
  bad(f.request, SPIN8_INVALID_ARGUMENT, sizeof(r) - 1);
  r = f.request;
  r.reserved0 = 1;
  bad(r);
  for (size_t i = 0; i < 4; ++i) {
    r = f.request;
    r.reserved[i] = 1;
    bad(r);
    r = f.request;
    r.attention.reserved[i] = 1;
    bad(r);
  }
  r = f.request;
  r.operation = 99;
  bad(r, SPIN8_UNSUPPORTED);
  r = f.request;
  r.k_dtype = SPIN8_BF16;
  bad(r, SPIN8_UNSUPPORTED);
  r = f.request;
  r.v_dtype = SPIN8_F64;
  bad(r, SPIN8_UNSUPPORTED);
  r = f.request;
  r.attention.abi_version++;
  bad(r, SPIN8_UNSUPPORTED);
  r = f.request;
  r.attention.struct_bytes--;
  bad(r);
  r = f.request;
  r.attention.q_source_id++;
  bad(r, SPIN8_UNSUPPORTED);
  r = f.request;
  r.attention.kv_source_id++;
  bad(r, SPIN8_UNSUPPORTED);
  r = f.request;
  r.attention.position_stage = 0;
  bad(r, SPIN8_UNSUPPORTED);
  r = f.request;
  r.attention.logit_order = 99;
  bad(r, SPIN8_UNSUPPORTED);
  r = f.request;
  r.attention.scale = std::numeric_limits<double>::infinity();
  bad(r, SPIN8_NONFINITE);
  r = f.request;
  r.attention.softcap = -1;
  bad(r);
  r = f.request;
  r.attention.n_key = SIZE_MAX;
  bad(r);
  r = f.request;
  r.attention.n_query = 0;
  bad(r);
  r = f.request;
  r.workspace_bytes--;
  bad(r);
  r = f.request;
  r.device_status_bytes = 3;
  bad(r);
  r = f.request;
  r.workspace = static_cast<char *>(r.workspace) + 1;
  bad(r);
  r = f.request;
  r.device_status =
      reinterpret_cast<uint32_t *>(static_cast<char *>(f.status.data) + 1);
  bad(r);
  r = f.request;
  r.workspace_bytes = SIZE_MAX;
  bad(r);
  r = f.request;
  r.stream = nullptr;
  bad(r, SPIN8_UNSUPPORTED);
  r = f.request;
  r.stream = cudaStreamLegacy;
  bad(r, SPIN8_UNSUPPORTED);
  r = f.request;
  r.attention.mask = {};
  bad(r);
  r = f.request;
  r.attention.output.capacity_bytes--;
  bad(r);
  r = f.request;
  r.attention.output.channel_stride_bytes = 4;
  bad(r);
  r = f.request;
  r.attention.output.data = nullptr;
  bad(r);
  r = f.request;
  r.workspace = f.out.device.data;
  r.workspace_bytes = std::max(r.workspace_bytes, f.out.device.bytes);
  bad(r);
  r = f.request;
  r.device_status = static_cast<uint32_t *>(r.workspace);
  bad(r);
  r = f.request;
  r.device_status = static_cast<uint32_t *>(r.attention.output.data);
  bad(r);
  for (auto field :
       {&spin8_attention_v2_request::q, &spin8_attention_v2_request::k,
        &spin8_attention_v2_request::v, &spin8_attention_v2_request::mask,
        &spin8_attention_v2_request::bias, &spin8_attention_v2_request::gate}) {
    r = f.request;
    (r.attention.*field).capacity_bytes--;
    bad(r);
    r = f.request;
    (r.attention.*field).token_stride_bytes = 0;
    bad(r);
    r = f.request;
    (r.attention.*field).head_stride_bytes = 0;
    bad(r);
    r = f.request;
    (r.attention.*field).channel_stride_bytes = 0;
    bad(r);
    r = f.request;
    (r.attention.*field).token_stride_bytes = SIZE_MAX;
    bad(r);
    r = f.request;
    (r.attention.*field).capacity_bytes = SIZE_MAX;
    bad(r);
    r = f.request;
    (r.attention.*field).data = nullptr;
    bad(r);
    if (field != &spin8_attention_v2_request::mask) {
      r = f.request;
      (r.attention.*field).data =
          static_cast<const char *>((r.attention.*field).data) + 1;
      bad(r);
      r = f.request;
      (r.attention.*field).channel_stride_bytes++;
      bad(r);
    }
    for (size_t shift : {size_t(0), size_t(4)}) {
      r = f.request;
      (r.attention.*field).data =
          static_cast<char *>(r.attention.output.data) + shift;
      // Capacities/layouts remain valid; alias must reject before device
      // access.
      bad(r);
    }
  }
  size_t bytes = 123;
  check(spin8_attention_v2_cuda_workspace_bytes(f.plan, SIZE_MAX, 5, &bytes)
                    .status == SPIN8_INVALID_ARGUMENT &&
            bytes == 123,
        "workspace overflow unchanged");
  auto bad_config = [&](spin8_attention_v2_config c,
                        spin8_status expected = SPIN8_INVALID_ARGUMENT,
                        int device = 0) {
    auto *p = f.plan;
    auto result =
        spin8_attention_v2_cuda_plan_create(&c, sizeof(c), device, &p);
    check(result.status == expected && p == f.plan,
          "invalid config preserves plan output");
    ++cases;
  };
  auto c = f.config;
  c.abi_version++;
  bad_config(c, SPIN8_UNSUPPORTED);
  c = f.config;
  c.struct_bytes--;
  bad_config(c);
  c = f.config;
  c.reserved[0] = 1;
  bad_config(c);
  c = f.config;
  c.dtype = SPIN8_F64;
  bad_config(c, SPIN8_UNSUPPORTED);
  c = f.config;
  c.codec = 1;
  bad_config(c, SPIN8_UNSUPPORTED);
  c = f.config;
  c.semantics = 2;
  bad_config(c, SPIN8_UNSUPPORTED);
  c = f.config;
  c.basis = 77;
  bad_config(c, SPIN8_UNSUPPORTED);
  c = f.config;
  c.source_id = 0;
  bad_config(c);
  c = f.config;
  c.d_qk = SIZE_MAX;
  bad_config(c);
  c = f.config;
  c.q_to_kv_bytes--;
  bad_config(c);
  c = f.config;
  c.q_to_kv = nullptr;
  bad_config(c);
  bad_config(f.config, SPIN8_UNSUPPORTED, -1);
  f.map.back() = 99;
  bad_config(f.config);
  f.map.back() = 1;
}

static void device_validation() {
  Fixture f(130, 96, 5, 3, SPIN8_F16, SPIN8_F32);
  f.fixed(1, true);
  f.make();
  auto bad = [&](uint32_t bit) {
    f.upload();
    ok(spin8_attention_v2_cuda_enqueue(f.plan, &f.request, sizeof(f.request)));
    check((f.finish() & bit) != 0, "device error observable after completion");
    check(f.out.host == f.out.download(f.stream),
          "late device error leaves ALL output bytes unchanged");
    ++cases;
  };
  for (double x : {std::numeric_limits<double>::quiet_NaN(),
                   std::numeric_limits<double>::infinity(),
                   -std::numeric_limits<double>::infinity()})
    for (Buffer *b : {&f.q, &f.k, &f.v, &f.bias, &f.gate}) {
      const double saved = b->get(b->nt - 1, b->nh - 1, b->nc - 1);
      b->put(b->nt - 1, b->nh - 1, b->nc - 1, x);
      bad(SPIN8_ATTENTION_V2_CUDA_NONFINITE);
      b->put(b->nt - 1, b->nh - 1, b->nc - 1, saved);
    }
  f.mask.put(2, 4, 4, 0);
  f.bias.put(2, 4, 4, std::numeric_limits<double>::infinity());
  bad(SPIN8_ATTENTION_V2_CUDA_NONFINITE);
  f.bias.put(2, 4, 4, 0);
  for (size_t j = 0; j < f.k.nt; ++j)
    f.mask.put(2, 4, j, 0);
  bad(SPIN8_ATTENTION_V2_CUDA_ALL_MASKED);
  for (size_t j = 0; j < f.k.nt; ++j)
    f.mask.put(2, 4, j, 1);
  f.request.attention.scale = std::numeric_limits<double>::max();
  bad(SPIN8_ATTENTION_V2_CUDA_NONFINITE);
  f.request.attention.scale = .37;
  for (size_t t = 0; t < f.v.nt; ++t)
    for (size_t h = 0; h < f.v.nh; ++h)
      for (size_t c = 0; c < f.v.nc; ++c)
        f.v.put(t, h, c, 2);
  f.gate.put(2, 4, 95, FLT_MAX);
  bad(SPIN8_ATTENTION_V2_CUDA_NONFINITE);
  f.gate.put(2, 4, 95, .5);
  f.compare(); // replay resets its own prior device error
}

static void streams_graphs() {
  Fixture a(80, 130, 3, 3), b(130, 80, 5, 1, SPIN8_F16, SPIN8_F16);
  a.fixed(0, false);
  b.fixed(2, true);
  a.make();
  b.make();
  a.upload();
  b.upload();
  ok(spin8_attention_v2_cuda_enqueue(a.plan, &a.request, sizeof(a.request)));
  ok(spin8_attention_v2_cuda_enqueue(b.plan, &b.request, sizeof(b.request)));
  a.verify();
  b.verify();
  Fixture shared(80, 130, 3, 3);
  shared.make();
  shared.upload();
  ok(spin8_attention_v2_cuda_enqueue(a.plan, &a.request, sizeof(a.request)));
  ok(spin8_attention_v2_cuda_enqueue(a.plan, &shared.request,
                                     sizeof(shared.request)));
  shared.verify();
  a.verify();
  cudaGraph_t graph = nullptr;
  cudaGraphExec_t executable = nullptr;
  cuda_ok(cudaStreamBeginCapture(b.stream, cudaStreamCaptureModeThreadLocal));
  ok(spin8_attention_v2_cuda_enqueue(b.plan, &b.request, sizeof(b.request)));
  cuda_ok(cudaStreamEndCapture(b.stream, &graph));
  cuda_ok(cudaGraphInstantiate(&executable, graph, 0));
  for (int i = 0; i < 2; ++i) {
    cuda_ok(cudaGraphLaunch(executable, b.stream));
    b.verify();
  }
  const double saved = b.gate.get(2, 4, 79);
  b.gate.put(2, 4, 79, std::numeric_limits<double>::quiet_NaN());
  b.gate.upload(b.stream);
  const auto before = b.out.download(b.stream);
  cuda_ok(cudaGraphLaunch(executable, b.stream));
  check(b.finish() & SPIN8_ATTENTION_V2_CUDA_NONFINITE,
        "graph replay device error");
  check(before == b.out.download(b.stream),
        "graph late error preserves output");
  b.gate.put(2, 4, 79, saved);
  b.gate.upload(b.stream);
  cuda_ok(cudaGraphLaunch(executable, b.stream));
  b.verify();
  cuda_ok(cudaGraphExecDestroy(executable));
  cuda_ok(cudaGraphDestroy(graph));
  std::puts("PASS independent streams/plans, shared immutable plan, graph "
            "capture/replay/error recovery");
}

static void boundary_supplement() {
  Fixture f(8, 9, 129, 3);
  f.make();
  f.compare();
  // More than one CTA of rows: an error in the LAST row must prevent every
  // earlier CTA's output from publishing, not just the failing row.
  for (size_t j = 0; j < f.k.nt; ++j)
    f.mask.put(2, 128, j, 0);
  const auto saved = f.out.host;
  f.upload();
  ok(spin8_attention_v2_cuda_enqueue(f.plan, &f.request, sizeof(f.request)));
  check(f.finish() == SPIN8_ATTENTION_V2_CUDA_ALL_MASKED,
        "multi-CTA late all-masked status");
  check(f.out.download(f.stream) == saved,
        "multi-CTA whole-output transaction");
  ++cases;
  auto r = f.request;
  check(spin8_attention_v2_cuda_enqueue(nullptr, &r, sizeof(r)).status ==
            SPIN8_INVALID_ARGUMENT,
        "null plan");
  check(spin8_attention_v2_cuda_enqueue(f.plan, nullptr, sizeof(r)).status ==
            SPIN8_INVALID_ARGUMENT,
        "null request");
  alignas(
      spin8_attention_v2_cuda_request) unsigned char unaligned[sizeof(r) + 8]{};
  std::memcpy(unaligned + 1, &r, sizeof(r));
  check(spin8_attention_v2_cuda_enqueue(
            f.plan,
            reinterpret_cast<const spin8_attention_v2_cuda_request *>(
                unaligned + 1),
            sizeof(r))
                .status == SPIN8_INVALID_ARGUMENT,
        "unaligned request");
  auto c = f.config;
  check(spin8_attention_v2_cuda_plan_create(nullptr, sizeof(c), 0, &f.plan)
                .status == SPIN8_INVALID_ARGUMENT,
        "null config");
  check(spin8_attention_v2_cuda_plan_create(&c, sizeof(c), 0, nullptr).status ==
            SPIN8_INVALID_ARGUMENT,
        "null plan output");
  check(spin8_attention_v2_cuda_plan_create(&c, sizeof(c) - 1, 0, &f.plan)
                .status == SPIN8_INVALID_ARGUMENT,
        "short config span");
  check(spin8_attention_v2_cuda_plan_create(
            &c, sizeof(c), 0,
            reinterpret_cast<spin8_attention_v2_cuda_plan **>(&c.reserved[0]))
                .status == SPIN8_INVALID_ARGUMENT,
        "plan output aliases config");
  // Source metadata is host memory; passing it as tensor storage fails before
  // enqueue. Valid layouts, and an address range that does not alias writes.
  r = f.request;
  r.attention.q.data = f.q.host.data() + 16;
  check(spin8_attention_v2_cuda_enqueue(f.plan, &r, sizeof(r)).status ==
            SPIN8_UNSUPPORTED,
        "host tensor rejected");
  cases += 8;

  Fixture fixed(130, 130, 3, 3);
  fixed.fixed(2, true);
  auto reject = [&](spin8_attention_v2_config cfg, spin8_status expected) {
    spin8_attention_v2_cuda_plan *p = nullptr;
    check(
        spin8_attention_v2_cuda_plan_create(&cfg, sizeof(cfg), 0, &p).status ==
                expected &&
            !p,
        "fixed invalid config");
    ++cases;
  };
  c = fixed.config;
  c.view = 3;
  reject(c, SPIN8_UNSUPPORTED);
  c = fixed.config;
  c.hadamard_block = 8;
  reject(c, SPIN8_UNSUPPORTED);
  c = fixed.config;
  c.planes_bytes--;
  reject(c, SPIN8_INVALID_ARGUMENT);
  c = fixed.config;
  c.planes = nullptr;
  reject(c, SPIN8_INVALID_ARGUMENT);
  c = fixed.config;
  c.plane_count = SIZE_MAX;
  reject(c, SPIN8_INVALID_ARGUMENT);
  fixed.planes[0].angle = std::numeric_limits<double>::infinity();
  reject(fixed.config, SPIN8_NONFINITE);
  fixed.planes[0].angle = .73;
  fixed.planes[0].a = 8;
  reject(fixed.config, SPIN8_INVALID_ARGUMENT);
  fixed.config.planes = nullptr;
  fixed.config.plane_count = fixed.config.planes_bytes = 0;
  fixed.make();
  fixed.compare(); // H128 without a Spin8 plane, including tails.
  ok(spin8_attention_v2_cuda_plan_free(nullptr));
}

int main(int argc, char **argv) {
  resources();
  preallocated_first_use();
  if (argc == 2 && std::strcmp(argv[1], "--first-use-only") == 0)
    return 0;
  identity_attention();
  host_validation();
  geometry_cases();
  fixed_cases();
  transforms_controls();
  device_validation();
  streams_graphs();
  boundary_supplement();
  std::printf("PASS attention_v2 CUDA: %zu cases; max F32 %.17g; max F16 "
              "%.17g; transform %.17g; tracked peak %zu bytes\n",
              cases, max_f32, max_f16, max_transform, peak_bytes);
}
