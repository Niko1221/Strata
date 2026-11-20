#include "native_cache_fixture.hpp"
#include "spin8/query_v3.h"
#include "strata/core/native_cache_owner.hpp"
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cuda_runtime_api.h>
#include <limits>
using namespace strata::core;
static size_t checks = 0, payload_bytes = 0;
static double query_error = 0, attention_error = 0;
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
    auto code = (x);                                                           \
    if (code != cudaSuccess)                                                   \
      throw std::runtime_error(cudaGetErrorString(code));                      \
  } while (0)
struct Buffer {
  void *p = nullptr;
  size_t bytes;
  cudaStream_t stream;
  Buffer(size_t n, cudaStream_t s) : bytes(n), stream(s) {
    CUDA(cudaMalloc(&p, n));
  }
  ~Buffer() { cudaFree(p); }
  Buffer(const Buffer &) = delete;
  void put(const void *src) {
    CUDA(cudaMemcpyAsync(p, src, bytes, cudaMemcpyHostToDevice, stream));
  }
  void get(void *dst) {
    CUDA(cudaMemcpyAsync(dst, p, bytes, cudaMemcpyDeviceToHost, stream));
    CUDA(cudaStreamSynchronize(stream));
  }
  tqc_dialect_v1_span span(size_t stride) { return {p, bytes, stride}; }
};
struct Session {
  std::shared_ptr<const NativeCachePlan> plan;
  cudaStream_t stream{};
  void *arena = nullptr;
  std::unique_ptr<NativeCacheOwner> owner;
  explicit Session(std::shared_ptr<const NativeCachePlan> p)
      : plan(std::move(p)) {
    CUDA(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
    CUDA(cudaMalloc(&arena, plan->layout().total_bytes));
    int result = -1;
    owner = NativeCacheOwner::attach(plan, arena, plan->layout().total_bytes, 0,
                                     stream, result);
    CHECK(owner && result == 0 && owner->host_usable());
  }
  ~Session() {
    cudaStreamSynchronize(stream);
    owner.reset();
    cudaFree(arena);
    cudaStreamDestroy(stream);
  }
  void cmd(uint64_t append, uint64_t query, uint64_t target = 0) {
    tqc_cache_cuda_command command{append, query, target};
    CUDA(cudaMemcpyAsync(owner->command_device(), &command, sizeof(command),
                         cudaMemcpyHostToDevice, stream));
    CUDA(cudaStreamSynchronize(stream));
  }
  tqc_cache_cuda_state status(int &query) {
    tqc_cache_cuda_state state{};
    CUDA(cudaMemcpyAsync(&state, owner->cache_status_device(), sizeof(state),
                         cudaMemcpyDeviceToHost, stream));
    CUDA(cudaMemcpyAsync(&query, owner->query_status_device(), sizeof(query),
                         cudaMemcpyDeviceToHost, stream));
    CUDA(cudaStreamSynchronize(stream));
    return state;
  }
  void good(uint64_t committed, uint64_t tentative) {
    int query = -1;
    auto state = status(query);
    CHECK(owner->observe_completion(true, state, query));
    CHECK(state.committed == committed && state.tentative == tentative);
  }
  std::vector<uint8_t> image() {
    std::vector<uint8_t> v(plan->layout().cache.total_bytes);
    CUDA(cudaMemcpyAsync(v.data(), arena, v.size(), cudaMemcpyDeviceToHost,
                         stream));
    CUDA(cudaStreamSynchronize(stream));
    return v;
  }
};
static void dialect(unsigned id, unsigned op, const void *src, void *dst,
                    float *norm = nullptr) {
  const size_t packed = 18 + 16 * id;
  tqc_dialect_v1_request r{
      1,
      sizeof(r),
      id,
      op,
      TQC_D_CPU,
      TQC_D_F32,
      nullptr,
      1,
      128,
      {const_cast<void *>(src), op == TQC_D_DECODE ? packed : 512,
       op == TQC_D_DECODE ? packed : 512},
      {dst, op == TQC_D_PACK ? packed : 512, op == TQC_D_PACK ? packed : 512},
      norm ? tqc_dialect_v1_span{norm, 4, 4} : tqc_dialect_v1_span{}};
  tqc_dialect_v1_receipt receipt{1, sizeof(receipt), 0, 0, 0};
  CHECK(tqc_dialect_v1_execute(&r, &receipt) == 0);
}
static void rotate(const float *x, const double *b, float *y) {
  for (size_t g = 0; g < 128; g += 8)
    for (size_t i = 0; i < 8; ++i) {
      float sum = 0;
      for (size_t j = 0; j < 8; ++j)
        sum += float(b[i * 8 + j]) * x[g + j];
      y[g + i] = sum;
    }
}
static std::vector<float>
query_reference(const NativeCacheRequested &r, const NativeCacheBinding &b,
                const std::vector<uint8_t> &sidecar,
                const std::vector<float> &q, size_t head_stride,
                size_t token_stride, size_t output_stride) {
  const auto g = b.normal_layers[0];
  std::vector<float> output(2 * g.q_heads * output_stride, -99.f);
  spin8_query_adapter *adapter = nullptr;
  spin8_query_adapter_layer_view view{};
  view.abi_version = 1;
  view.struct_bytes = sizeof(view);
  if (!sidecar.empty()) {
    NativeCacheDigest profile{};
    std::string error;
    CHECK(NativeCachePlan::base_profile(r, b.position, profile, error));
    spin8_query_adapter_expectations expected{};
    expected.abi_version = 1;
    expected.struct_bytes = sizeof(expected);
    expected.model_count = uint32_t(b.raw_models.size());
    expected.layer_count = uint32_t(b.normal_layers.size());
    expected.model_digests = b.raw_models[0].data();
    expected.model_digests_bytes = b.raw_models.size() * 32;
    std::copy(profile.begin(), profile.end(), expected.profile_digest);
    std::copy(b.position.begin(), b.position.end(), expected.position_digest);
    expected.layers = b.normal_layers.data();
    expected.layers_bytes = b.normal_layers.size() * sizeof(g);
    CHECK(spin8_query_adapter_load(sidecar.data(), sidecar.size(), &expected,
                                   sizeof(expected), &adapter) == SPIN8_OK);
    CHECK(spin8_query_adapter_get_layer(adapter, 0, &view, sizeof(view)) ==
          SPIN8_OK);
  }
  for (size_t head = 0; head < g.q_heads; ++head) {
    spin8_query_v3_controller *controller = nullptr;
    if (adapter) {
      std::vector<double> win(view.win + head * 8 * g.head_width,
                              view.win + (head + 1) * 8 * g.head_width),
          bin(view.bin + head * 8, view.bin + (head + 1) * 8),
          wout(view.wout + head * 56, view.wout + (head + 1) * 56),
          bout(view.bout + head * 7, view.bout + (head + 1) * 7);
      CHECK(spin8_query_v3_controller_create(
                3, g.head_width, win.data(), win.size() * 8, bin.data(),
                bin.size() * 8, wout.data(), wout.size() * 8, bout.data(),
                bout.size() * 8, &controller) == SPIN8_OK);
    }
    for (size_t token = 0; token < 2; ++token) {
      const auto *src = q.data() + token * token_stride + head * head_stride;
      auto *dst = output.data() + (token * g.q_heads + head) * output_stride;
      if (controller)
        CHECK(spin8_query_v3_controller_apply(
                  controller, SPIN8_F32, src, g.head_width * 4, g.head_width,
                  dst, g.head_width * 4, g.head_width, 1) == SPIN8_OK);
      else
        std::copy_n(src, g.head_width, dst);
    }
    spin8_query_v3_controller_free(controller);
  }
  spin8_query_adapter_free(adapter);
  return output;
}
static void check_payload_and_attention(
    const NativeCacheRequested &r, const NativeCacheBinding &b,
    const std::vector<float> &k, const std::vector<float> &v, size_t stride,
    const std::vector<float> &controlled, const std::vector<float> &out,
    const NativeCacheOwnerLayout &layout, const std::vector<uint8_t> &image) {
  const auto &g = b.normal_layers[0];
  const auto &l = layout.cache;
  spin8_plane planes[] = {{0, 1, .37}, {2, 5, -.61}};
  spin8_plan *spin = nullptr;
  CHECK(spin8_plan_create(planes, 2, &spin) == SPIN8_OK);
  double matrices[192];
  for (int view = 0; view < 3; ++view)
    CHECK(spin8_plan_matrix(spin, static_cast<spin8_view>(view),
                            matrices + view * 64) == SPIN8_OK);
  spin8_plan_free(spin);
  std::vector<float> keys[3], values(2 * g.kv_heads * g.head_width);
  for (auto &x : keys)
    x.resize(values.size());
  const bool diagnostic =
      r.representation == NativeCacheRepresentation::DiagnosticF32;
  for (size_t row = 0; row < 2 * g.kv_heads; ++row)
    for (size_t block = 0; block < g.head_width / 128; ++block) {
      float prepared[128], rotated[128], norm = 0;
      dialect(uint32_t(r.k), diagnostic ? TQC_D_FORWARD : TQC_D_PREPARE,
              k.data() + row * stride + block * 128, prepared,
              diagnostic ? nullptr : &norm);
      for (int view = 0; view < 3; ++view) {
        rotate(prepared, matrices + 64 * view, rotated);
        const auto *actual =
            image.data() + l.k[view] + row * l.k_row_bytes +
            block * (diagnostic ? 512 : 18 + 16 * uint32_t(r.k));
        uint8_t expected[512]{};
        size_t n = diagnostic ? 512 : 18 + 16 * uint32_t(r.k);
        if (diagnostic)
          std::memcpy(expected, rotated, n);
        else
          dialect(uint32_t(r.k), TQC_D_PACK, rotated, expected, &norm);
        CHECK(std::memcmp(actual, expected, n) == 0);
        payload_bytes += n;
        auto *decoded = keys[view].data() + row * g.head_width + block * 128;
        if (diagnostic)
          std::memcpy(decoded, actual, n);
        else
          dialect(uint32_t(r.k), TQC_D_DECODE, actual, decoded);
      }
      dialect(uint32_t(r.v), diagnostic ? TQC_D_FORWARD : TQC_D_PREPARE,
              v.data() + row * stride + block * 128, prepared,
              diagnostic ? nullptr : &norm);
      const auto *actual = image.data() + l.v + row * l.v_row_bytes +
                           block * (diagnostic ? 512 : 18 + 16 * uint32_t(r.v));
      uint8_t expected[512]{};
      const size_t n = diagnostic ? 512 : 18 + 16 * uint32_t(r.v);
      if (diagnostic)
        std::memcpy(expected, prepared, n);
      else
        dialect(uint32_t(r.v), TQC_D_PACK, prepared, expected, &norm);
      CHECK(std::memcmp(actual, expected, n) == 0);
      payload_bytes += n;
      auto *decoded = values.data() + row * g.head_width + block * 128;
      if (diagnostic)
        std::memcpy(decoded, actual, n);
      else
        dialect(uint32_t(r.v), TQC_D_DECODE, actual, decoded);
    }
  for (size_t qr = 0; qr < 2 * g.q_heads; ++qr) {
    std::vector<float> query[3];
    for (auto &x : query)
      x.resize(g.head_width);
    for (size_t block = 0; block < g.head_width / 128; ++block) {
      float prepared[128];
      dialect(1, TQC_D_FORWARD, controlled.data() + qr * stride + block * 128,
              prepared);
      for (int view = 0; view < 3; ++view)
        rotate(prepared, matrices + view * 64,
               query[view].data() + block * 128);
    }
    const size_t kv = (qr % g.q_heads) / (g.q_heads / g.kv_heads);
    double score[2]{};
    for (size_t token = 0; token < 2; ++token)
      for (int view = 0; view < 3; ++view) {
        double dot = 0;
        for (size_t col = 0; col < g.head_width; ++col)
          dot += double(query[view][col]) *
                 keys[view][(token * g.kv_heads + kv) * g.head_width + col];
        score[token] += r.alpha[view] * dot * .125;
      }
    double weight = 1 / (1 + std::exp(score[1] - score[0]));
    for (size_t block = 0; block < g.head_width / 128; ++block) {
      float mixed[128], expected[128];
      for (size_t col = 0; col < 128; ++col)
        mixed[col] = float(
            weight * values[kv * g.head_width + block * 128 + col] +
            (1 - weight) *
                values[(g.kv_heads + kv) * g.head_width + block * 128 + col]);
      dialect(1, TQC_D_INVERSE, mixed, expected);
      for (size_t col = 0; col < 128; ++col) {
        const double error = std::abs(
            double(out[qr * stride + block * 128 + col]) - expected[col]);
        attention_error = std::max(attention_error, error);
        CHECK(error < 2e-5);
      }
    }
    for (size_t col = g.head_width; col < stride; ++col)
      CHECK(out[qr * stride + col] == -99.f);
  }
}
static void exercise(unsigned width, int adapter_kind, bool diagnostic) {
  auto r = native_fixture::request();
  auto b = native_fixture::binding();
  for (auto &g : b.normal_layers)
    g.head_width = width;
  if (diagnostic)
    r.representation = NativeCacheRepresentation::DiagnosticF32;
  auto fixture = adapter_kind ? native_fixture::sidecar(r, b, adapter_kind == 2)
                              : std::vector<uint8_t>{};
  std::string error;
  auto plan = NativeCachePlan::bind(
      r, b, fixture.empty() ? nullptr : fixture.data(), fixture.size(), error);
  CHECK(plan && plan->geometry().layer_index == 2);
  const auto g = plan->geometry();
  const size_t stride = width + 3, token_stride = g.q_heads * stride + 5;
  std::vector<float> k(2 * g.kv_heads * stride, -77.f), v(k.size(), -77.f),
      q(2 * token_stride, -77.f), controlled(2 * g.q_heads * stride, -99.f),
      out(controlled.size(), -99.f);
  for (size_t row = 0; row < 2 * g.kv_heads; ++row)
    for (size_t col = 0; col < width; ++col) {
      k[row * stride + col] =
          float(int((row * 19 + col * 7) % 101) - 50) / 512.f;
      v[row * stride + col] =
          float(int((row * 31 + col * 11) % 97) - 48) / 256.f;
    }
  for (size_t token = 0; token < 2; ++token)
    for (size_t head = 0; head < g.q_heads; ++head)
      for (size_t col = 0; col < width; ++col)
        q[token * token_stride + head * stride + col] =
            float(int((token * 17 + head * 31 + col * 3) % 107) - 53) / 128.f;
  q[0] = -0.f;
  auto reference =
      query_reference(r, b, fixture, q, stride, token_stride, stride);
  // The plan owns validated F32 storage. Original untrusted container bytes
  // need not remain alive during later quiescent upload or graph replay.
  std::fill(fixture.begin(), fixture.end(), uint8_t(0));
  fixture.clear();
  Session s(plan);
  plan.reset();
  Buffer dk(k.size() * 4, s.stream), dv(v.size() * 4, s.stream),
      dq(q.size() * 4, s.stream), dc(controlled.size() * 4, s.stream),
      dout(out.size() * 4, s.stream), dp(16, s.stream), dqp(16, s.stream);
  int64_t positions[2] = {0, 1}, query_positions[2] = {2, 2};
  dk.put(k.data());
  dv.put(v.data());
  dq.put(q.data());
  dc.put(controlled.data());
  dout.put(out.data());
  dp.put(positions);
  dqp.put(query_positions);
  NativeCacheQueryBatch batch{static_cast<const float *>(dq.p),
                              dq.bytes,
                              stride * 4,
                              token_stride * 4,
                              static_cast<float *>(dc.p),
                              dc.bytes,
                              stride * 4,
                              g.q_heads * stride * 4,
                              2};
  tqc_cache_cuda_append app{1, sizeof(app), dk.span(stride * 4),
                            dv.span(stride * 4), dp.span(8)};
  tqc_cache_cuda_attention att{1,
                               sizeof(att),
                               1,
                               0,
                               .125f,
                               dc.span(stride * 4),
                               dout.span(stride * 4),
                               dqp.span(8)};
  s.cmd(2, 2);
  CHECK(s.owner->query(batch) == 0);
  CHECK(s.owner->append(app, 2) == 0);
  CHECK(s.owner->control(TQC_CC_COMMIT) == 0);
  CHECK(s.owner->attend(att, 2) == 0);
  s.good(2, 2);
  dc.get(controlled.data());
  dout.get(out.data());
  for (size_t i = 0; i < controlled.size(); ++i) {
    query_error =
        std::max(query_error, std::abs(double(controlled[i]) - reference[i]));
    if (adapter_kind != 2)
      CHECK(std::memcmp(&controlled[i], &reference[i], 4) == 0);
    else
      CHECK(std::abs(double(controlled[i]) - reference[i]) < 2e-6);
  }
  check_payload_and_attention(r, b, k, v, stride, reference, out,
                              s.plan->layout(), s.image());
  // One immutable plan can back two independent session/layer owners; commands,
  // statuses, banks and payloads live only in each borrowed GPU region.
  {
    Session other(s.plan);
    other.good(0, 0);
    CHECK(other.owner->context().arena != s.owner->context().arena);
  }
  if (width == 128 && adapter_kind == 2 && !diagnostic) {
    CHECK(s.owner->initialize_quiescent() == 0);
    s.cmd(1, 1);
    cudaGraph_t graph{};
    cudaGraphExec_t executable{};
    CUDA(cudaStreamBeginCapture(s.stream, cudaStreamCaptureModeThreadLocal));
    CHECK(s.owner->query(batch) == 0);
    CHECK(s.owner->append(app, 2) == 0);
    CHECK(s.owner->control(TQC_CC_COMMIT) == 0);
    CHECK(s.owner->attend(att, 2) == 0);
    CUDA(cudaStreamEndCapture(s.stream, &graph));
    CUDA(cudaGraphInstantiate(&executable, graph, 0));
    CUDA(cudaGraphLaunch(executable, s.stream));
    s.good(1, 1);
    s.cmd(2, 2);
    CUDA(cudaGraphLaunch(executable, s.stream));
    s.good(3, 3);
    s.cmd(3, 1);
    CUDA(cudaGraphLaunch(executable, s.stream));
    int flag = 0;
    auto state = s.status(flag);
    CHECK(state.tentative == 3 && state.status == TQC_CC_CAPACITY);
    CHECK(!s.owner->observe_completion(true, state, flag));
    CHECK(s.owner->append(app, 2) == TQC_D_INVALID);
    CUDA(cudaGraphExecDestroy(executable));
    CUDA(cudaGraphDestroy(graph));
    CHECK(s.owner->initialize_quiescent() == 0);
    q[0] = std::numeric_limits<float>::quiet_NaN();
    dq.put(q.data());
    CHECK(s.owner->query(batch) == 0);
    state = s.status(flag);
    CHECK(flag != 0 && state.valid == 1);
    CHECK(!s.owner->observe_completion(true, state, flag));
    CHECK(s.owner->attend(att, 2) == TQC_D_INVALID);
    CHECK(s.owner->query(batch) == TQC_D_INVALID &&
          s.owner->control(TQC_CC_COMMIT) == TQC_D_INVALID);
    CHECK(s.owner->initialize_quiescent() == 0);
    state = s.status(flag);
    CHECK(!s.owner->observe_completion(
        false, state, flag)); // runtime may fail before flag write
    CHECK(!s.owner->observe_completion(true, state,
                                       flag)); // sticky host invalidity
    CHECK(s.owner->audit() == TQC_D_INVALID);
    CHECK(s.owner->initialize_quiescent() == 0);
    float corrupt = 2.f;
    CUDA(cudaMemcpyAsync(static_cast<char *>(s.arena) +
                             s.plan->layout().cache.matrices,
                         &corrupt, 4, cudaMemcpyHostToDevice, s.stream));
    CHECK(s.owner->audit() == 0);
    state = s.status(flag);
    CHECK(state.status & TQC_CC_TRANSFORM_INVARIANT_VIOLATION);
    CHECK(!s.owner->observe_completion(true, state, flag));
    CHECK(s.owner->initialize_quiescent() == 0);
    s.good(0, 0);
    int result = -1;
    CHECK(!NativeCacheOwner::attach(s.plan, s.arena,
                                    s.plan->layout().total_bytes - 1, 0,
                                    s.stream, result) &&
          result == TQC_D_CAPACITY);
    CHECK(!NativeCacheOwner::attach(s.plan, static_cast<char *>(s.arena) + 1,
                                    s.plan->layout().total_bytes - 1, 0,
                                    s.stream, result));
    CHECK(!NativeCacheOwner::attach(
        s.plan, s.arena, s.plan->layout().total_bytes, 0, nullptr, result));
    CHECK(!NativeCacheOwner::attach(s.plan, s.arena,
                                    s.plan->layout().total_bytes, 1, s.stream,
                                    result) &&
          result == TQC_D_UNSUPPORTED);
    auto wrong = batch;
    wrong.tokens = 4;
    CHECK(s.owner->query(wrong) == TQC_D_INVALID);
    wrong = batch;
    wrong.src = static_cast<const float *>(s.arena);
    CHECK(s.owner->query(wrong) == TQC_D_OVERLAP);
    s.good(0, 0);
    tqc_cache_cuda_resources resources{1, sizeof(resources)};
    CHECK(tqc_cache_cuda_resources_query(&s.owner->context(), &resources) == 0);
    for (size_t i = 0; i < 6; ++i)
      std::printf("cache kernel %zu local=%zu shared=%zu constant=%zu "
                  "registers=%d max_threads=%d\n",
                  i, resources.local_bytes[i], resources.shared_bytes[i],
                  resources.constant_bytes[i], resources.registers[i],
                  resources.max_threads[i]);
  }
  const auto &l = s.plan->layout();
  std::printf("owner width=%u adapter=%d diagnostic=%d total=%zu cache=%zu "
              "bank=%zu status=%zu padding=%zu\n",
              width, adapter_kind, int(diagnostic), l.total_bytes,
              l.cache.total_bytes,
              l.parameter_bytes[0] + l.parameter_bytes[1] +
                  l.parameter_bytes[2] + l.parameter_bytes[3],
              sizeof(int), l.bank_alignment_bytes);
}
int main() {
  try {
    for (int adapter : {0, 1, 2})
      exercise(128, adapter, false);
    exercise(256, 2, false);
    exercise(128, 2, true);
    exercise(256, 2, true);
    std::printf("%zu owner checks; %zu exact CPU-reference payload bytes; max "
                "query error %.9g; max Attention error %.9g\n",
                checks, payload_bytes, query_error, attention_error);
    return 0;
  } catch (const std::exception &e) {
    std::fprintf(stderr, "%s\n", e.what());
    return 1;
  }
}