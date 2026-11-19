// Exact prefill recurrence parity and same-process CUDA-event comparisons.
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <cuda_runtime.h>
#include <limits>
#include <stdexcept>
#include <vector>
namespace {
#include "../../src/qwen35/q36_gdn_prefill.cuh"
void check(cudaError_t e) {
  if (e != cudaSuccess)
    throw std::runtime_error(cudaGetErrorString(e));
}
struct Device {
  float *p = nullptr;
  explicit Device(size_t n) { check(cudaMalloc(&p, n * sizeof(float))); }
  ~Device() { cudaFree(p); }
  Device(const Device &) = delete;
  Device &operator=(const Device &) = delete;
  void put(const std::vector<float> &v) {
    check(cudaMemcpy(p, v.data(), v.size() * 4, cudaMemcpyHostToDevice));
  }
  std::vector<float> get(size_t n) {
    std::vector<float> v(n);
    check(cudaMemcpy(v.data(), p, n * 4, cudaMemcpyDeviceToHost));
    return v;
  }
};
struct Event {
  cudaEvent_t e{};
  Event() { check(cudaEventCreate(&e)); }
  ~Event() { cudaEventDestroy(e); }
  Event(const Event &) = delete;
  Event &operator=(const Event &) = delete;
};
int candidateGroups = 8;
void launch(bool candidate, float *state, const float *h, int C,
            const float *gate, const float *beta, float *output, int hk, int hv,
            int tokens) {
  if (candidate && candidateGroups == 8)
    k_gdn_seq_cols<8><<<dim3(hv, 8), dim3(16, 16)>>>(state, h, C, gate, beta,
                                                     output, hk, hv, tokens);
  else if (candidate)
    k_gdn_seq_cols<16><<<dim3(hv, 16), dim3(8, 16)>>>(state, h, C, gate, beta,
                                                      output, hk, hv, tokens);
  else
    k_gdn_seq_cols<4><<<dim3(hv, kGdnColGroups), dim3(32, 16)>>>(
        state, h, C, gate, beta, output, hk, hv, tokens);
  check(cudaGetLastError());
}
void identical(const std::vector<float> &a, const std::vector<float> &b,
               const char *label) {
  for (size_t i = 0; i < a.size(); ++i) {
    if (std::isnan(a[i]) && std::isnan(b[i]))
      continue;
    if (std::memcmp(&a[i], &b[i], sizeof(float))) {
      std::fprintf(stderr, "%s mismatch at %zu: %.9g / %.9g\n", label, i, a[i],
                   b[i]);
      throw std::runtime_error("FP32 parity failure");
    }
  }
}
void run(int tokens, int hk, int hv, int mode, bool timing = false) {
  const int C = (2 * hk + hv) * 128;
  const size_t ns = (size_t)128 * hv * 128, no = (size_t)tokens * hv * 128;
  std::vector<float> initial(ns + 16, 12345.0f),
      inputs(std::max<size_t>(1, (size_t)tokens * C)),
      gates(std::max(1, tokens * hv)), betas(gates.size()),
      empty(no + 16, 12345.0f);
  uint32_t seed = 832791;
  auto value = [&]() {
    seed = seed * 1664525u + 1013904223u;
    return (float)(int(seed >> 16) - 32768) / 32768.0f;
  };
  for (size_t i = 0; i < ns; ++i)
    initial[i] = mode == 1 ? 0 : value() * 0.05f;
  for (auto &v : inputs)
    v = mode == 1 ? 0 : value() * 0.02f;
  for (auto &v : gates)
    v = -0.03f + value() * 0.01f;
  for (auto &v : betas)
    v = 0.3f + value() * 0.1f;
  if (mode == 2 && tokens)
    inputs[C / 2] = std::numeric_limits<float>::quiet_NaN();
  Device a(initial.size()), b(initial.size()), h(inputs.size()),
      g(gates.size()), bt(betas.size()), oa(empty.size()), ob(empty.size());
  h.put(inputs);
  g.put(gates);
  bt.put(betas);
  a.put(initial);
  b.put(initial);
  oa.put(empty);
  ob.put(empty);
  launch(false, a.p, h.p, C, g.p, bt.p, oa.p, hk, hv, tokens);
  launch(true, b.p, h.p, C, g.p, bt.p, ob.p, hk, hv, tokens);
  check(cudaDeviceSynchronize());
  const auto ref = a.get(initial.size()), candidate = b.get(initial.size()),
             out = oa.get(empty.size()), got = ob.get(empty.size());
  identical(ref, candidate, "state");
  identical(out, got, "output");
  for (size_t i = ns; i < initial.size(); ++i)
    if (candidate[i] != 12345.0f)
      throw std::runtime_error("state canary overwritten");
  for (size_t i = no; i < empty.size(); ++i)
    if (got[i] != 12345.0f)
      throw std::runtime_error("output canary overwritten");
  if (tokens > 1 && mode == 0) {
    b.put(initial);
    ob.put(empty);
    const int cut = std::min(127, tokens / 2);
    launch(true, b.p, h.p, C, g.p, bt.p, ob.p, hk, hv, cut);
    launch(true, b.p, h.p + (size_t)cut * C, C, g.p + cut * hv, bt.p + cut * hv,
           ob.p + (size_t)cut * hv * 128, hk, hv, tokens - cut);
    check(cudaDeviceSynchronize());
    identical(ref, b.get(initial.size()), "chunk state");
    identical(out, ob.get(empty.size()), "chunk output");
  }
  std::printf("{\"kind\":\"parity\",\"tokens\":%d,\"hk\":%d,\"hv\":%d,\"mode\":"
              "%d,\"bitwise\":true}\n",
              tokens, hk, hv, mode);
  if (timing) {
    constexpr int iterations = 5;
    auto measure = [&](bool warp) {
      b.put(initial);
      Event begin, end;
      check(cudaEventRecord(begin.e));
      for (int i = 0; i < iterations; ++i)
        launch(warp, b.p, h.p, C, g.p, bt.p, ob.p, hk, hv, tokens);
      check(cudaEventRecord(end.e));
      check(cudaEventSynchronize(end.e));
      float ms = 0;
      check(cudaEventElapsedTime(&ms, begin.e, end.e));
      return ms / iterations;
    };
    for (int pair = 0; pair < 5; ++pair) {
      float baseline, optimized;
      if (pair % 2) {
        optimized = measure(true);
        baseline = measure(false);
      } else {
        baseline = measure(false);
        optimized = measure(true);
      }
      std::printf("{\"kind\":\"timing\",\"tokens\":%d,\"pair\":%d,\"reference_"
                  "ms\":%.6f,\"candidate_ms\":%.6f}\n",
                  tokens, pair, baseline, optimized);
    }
  }
}
} // namespace
int main(int argc, char **argv) {
  try {
    const bool quick = argc == 2 && std::strcmp(argv[1], "--quick") == 0;
    if (argc > 1 && !quick)
      throw std::runtime_error("Only --quick is supported");
    cudaDeviceProp prop{};
    check(cudaGetDeviceProperties(&prop, 0));
    std::printf("{\"device\":\"%s\",\"compute\":\"%d.%d\",\"sm_count\":%d}\n",
                prop.name, prop.major, prop.minor, prop.multiProcessorCount);
    for (int groups : {8, 16}) {
      candidateGroups = groups;
      cudaFuncAttributes attr{};
      check(cudaFuncGetAttributes(&attr, groups == 8 ? k_gdn_seq_cols<8>
                                                     : k_gdn_seq_cols<16>));
      std::printf("{\"kernel\":\"gdn_columns\",\"groups\":%d,\"registers\":%d,"
                  "\"local_bytes\":%zu,"
                  "\"shared_bytes\":%zu}\n",
                  groups, attr.numRegs, attr.localSizeBytes,
                  attr.sharedSizeBytes);
      if (quick)
        for (int t : {0, 9, 129})
          run(t, 16, 32, 0);
      else
        for (int t : {0, 1, 2, 7, 8, 9, 127, 128, 129, 511, 512, 513, 2048})
          run(t, 16, 32, 0, t == 128 || t == 512 || t == 2048);
      run(129, 3, 5, 0);
      run(129, 16, 32, 1);
      run(129, 16, 32, 2);
    }
    check(cudaDeviceSynchronize());
    std::puts("{\"passed\":true}");
    return 0;
  } catch (const std::exception &e) {
    std::fprintf(stderr, "FAIL: %s\n", e.what());
    return 1;
  }
}
