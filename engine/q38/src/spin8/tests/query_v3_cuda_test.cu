#include "spin8/query_v3_cuda.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <cuda_runtime.h>
#include <limits>
#include <memory>
#include <stdexcept>
#include <vector>

namespace {
void require(bool value, const char *message) {
  if (!value)
    throw std::runtime_error(message);
}
void cu(cudaError_t status) {
  if (status != cudaSuccess)
    throw std::runtime_error(cudaGetErrorString(status));
}
struct Stream {
  cudaStream_t value{};
  Stream() { cu(cudaStreamCreateWithFlags(&value, cudaStreamNonBlocking)); }
  ~Stream() { cudaStreamDestroy(value); }
  Stream(const Stream &) = delete;
  Stream &operator=(const Stream &) = delete;
};
struct Plan {
  spin8_query_v3_cuda_plan *value{};
  Plan() {
    require(spin8_query_v3_cuda_plan_create(SPIN8_QUERY_V3_VERSION, &value) ==
                SPIN8_OK,
            "v3 CUDA plan");
  }
  ~Plan() { spin8_query_v3_cuda_plan_free(value); }
  Plan(const Plan &) = delete;
  Plan &operator=(const Plan &) = delete;
};
template <class T> struct Device {
  T *value{};
  size_t count;
  explicit Device(size_t n) : count(n) {
    cu(cudaMalloc(reinterpret_cast<void **>(&value), bytes()));
  }
  ~Device() { cudaFree(value); }
  Device(const Device &) = delete;
  Device &operator=(const Device &) = delete;
  size_t bytes() const { return count * sizeof(T); }
  void put(const std::vector<T> &data, cudaStream_t stream) {
    require(data.size() == count, "upload size");
    cu(cudaMemcpyAsync(value, data.data(), bytes(), cudaMemcpyHostToDevice,
                       stream));
    // Test host vectors may be temporaries; finish their use before return.
    cu(cudaStreamSynchronize(stream));
  }
  std::vector<T> get(cudaStream_t stream) const {
    std::vector<T> data(count);
    cu(cudaMemcpyAsync(data.data(), value, bytes(), cudaMemcpyDeviceToHost,
                       stream));
    cu(cudaStreamSynchronize(stream));
    return data;
  }
};
struct Parameters {
  std::vector<float> win, bin, wout, bout;
  Device<float> dwin, dbin, dwout, dbout;
  explicit Parameters(size_t width)
      : win(8 * width), bin(8), wout(56), bout(7), dwin(win.size()), dbin(8),
        dwout(56), dbout(7) {}
  void upload(cudaStream_t stream) {
    dwin.put(win, stream);
    dbin.put(bin, stream);
    dwout.put(wout, stream);
    dbout.put(bout, stream);
  }
  spin8_query_v3_cuda_parameters_f32 descriptor() const {
    return {dwin.value,  dwin.bytes(),  dbin.value,  dbin.bytes(),
            dwout.value, dwout.bytes(), dbout.value, dbout.bytes()};
  }
  void nontrivial() {
    for (size_t i = 0; i < win.size(); ++i)
      win[i] = float(.037 * std::sin(double(i) * .51));
    for (size_t i = 0; i < bin.size(); ++i)
      bin[i] = float(.11 * std::cos(double(i)));
    for (size_t i = 0; i < wout.size(); ++i)
      wout[i] = float(.15 * std::cos(double(i) * .67));
    for (size_t i = 0; i < bout.size(); ++i)
      bout[i] = float(.03 * std::sin(double(i) + .5));
  }
};
using Controller = std::unique_ptr<spin8_query_v3_controller,
                                   decltype(&spin8_query_v3_controller_free)>;
Controller cpu_controller(const Parameters &p, size_t width) {
  // The CPU owns doubles representing EXACTLY the F32 device weights.
  const std::vector<double> win(p.win.begin(), p.win.end());
  const std::vector<double> bin(p.bin.begin(), p.bin.end());
  const std::vector<double> wout(p.wout.begin(), p.wout.end());
  const std::vector<double> bout(p.bout.begin(), p.bout.end());
  spin8_query_v3_controller *raw = nullptr;
  require(spin8_query_v3_controller_create(
              SPIN8_QUERY_V3_VERSION, width, win.data(), win.size() * 8,
              bin.data(), bin.size() * 8, wout.data(), wout.size() * 8,
              bout.data(), bout.size() * 8, &raw) == SPIN8_OK,
          "CPU v3 rounded-F32 controller");
  return Controller(raw, spin8_query_v3_controller_free);
}
bool same(float a, float b) { return std::memcmp(&a, &b, sizeof(float)) == 0; }
void close(double actual, double expected, const char *message) {
  require(std::abs(actual - expected) <= 2e-6 * (1. + std::abs(expected)),
          message);
}

// Independent direct dense reference: reconstruct C from Gamma, form J by
// dense multiplication, compute the controller and rational chart directly.
std::vector<double> direct_reference(const Parameters &p, const float *q,
                                     size_t width) {
  double gamma[2048]{};
  require(spin8_clifford_generators(gamma) == SPIN8_OK, "direct Gamma");
  double hidden[8]{}, a[7]{}, c[8]{1.};
  for (size_t k = 0; k < 8; ++k) {
    double z = p.bin[k];
    for (size_t i = 0; i < width; ++i)
      z += double(p.win[k * width + i]) * q[i];
    hidden[k] = std::tanh(z);
  }
  constexpr double tau = .41421356237309504880168872420970;
  double radius2 = tau * tau;
  for (size_t i = 0; i < 7; ++i) {
    a[i] = p.bout[i];
    for (size_t k = 0; k < 8; ++k)
      a[i] += double(p.wout[i * 8 + k]) * hidden[k];
    radius2 += a[i] * a[i];
  }
  double norm2 = 1.;
  for (size_t i = 0; i < 7; ++i) {
    c[i + 1] = tau * a[i] / std::sqrt(radius2);
    norm2 += c[i + 1] * c[i + 1];
  }
  for (double &value : c)
    value /= std::sqrt(norm2);
  double matrix[64]{};
  for (size_t i = 0; i < 8; ++i)
    for (size_t j = 0; j < 8; ++j) {
      matrix[i * 8 + j] = i == j ? c[0] : 0.;
      for (size_t axis = 0; axis < 7; ++axis)
        for (size_t t = 0; t < 8; ++t)
          matrix[i * 8 + j] += c[axis + 1] *
                               gamma[i * 256 + t * 16 + 8 + axis + 1] *
                               gamma[j * 256 + t * 16 + 8];
    }
  std::vector<double> out(q, q + width);
  for (size_t base = 0; width - base >= 8; base += 8)
    for (size_t i = 0; i < 8; ++i) {
      out[base + i] = 0.;
      for (size_t j = 0; j < 8; ++j)
        out[base + i] += matrix[i * 8 + j] * q[base + j];
    }
  return out;
}

void reference_cases(const Plan &plan, cudaStream_t stream) {
  constexpr size_t rows = 3;
  for (size_t width : {size_t{1}, size_t{7}, size_t{8}, size_t{23}, size_t{256},
                       size_t{1032}}) {
    const size_t ss = width + 5, ds = width + 9;
    std::vector<float> input(rows * ss, -71.f), sentinel(rows * ds, -93.f);
    for (size_t r = 0; r < rows; ++r)
      for (size_t i = 0; i < width; ++i)
        input[r * ss + i] = float(std::cos(double(i) * .6 + double(r) * .8));
    input[width > 2 ? 2 : 0] = -0.f;
    Parameters p(width);
    for (bool identity : {true, false}) {
      p.nontrivial();
      if (identity) {
        std::fill(p.wout.begin(), p.wout.end(), 0.f);
        std::fill(p.bout.begin(), p.bout.end(), -0.f);
      }
      p.upload(stream);
      const auto descriptor = p.descriptor();
      auto cpu = cpu_controller(p, width);
      auto expected = sentinel;
      require(spin8_query_v3_controller_apply(
                  cpu.get(), SPIN8_F32, input.data(), input.size() * 4, ss,
                  expected.data(), expected.size() * 4, ds, rows) == SPIN8_OK,
              "CPU v3 apply");
      if (!identity) {
        std::array<double, rows * 8> spinors{};
        require(spin8_query_v3_controller_spinors(
                    cpu.get(), SPIN8_F32, input.data(), input.size() * 4, ss,
                    spinors.data(), sizeof(spinors), 8, rows) == SPIN8_OK,
                "CPU spinors");
        require(std::abs(spinors[1] - spinors[9]) > 1e-5,
                "different rows produce different controller spinors");
      }
      Device<float> src(input.size()), dst(sentinel.size());
      Device<int> flag(1);
      src.put(input, stream);
      dst.put(sentinel, stream);
      flag.put({0}, stream);
      cu(spin8_query_v3_cuda_apply_f32(
          plan.value, &descriptor, src.value, (rows - 1) * ss * 4 + width * 4,
          ss * 4, dst.value, (rows - 1) * ds * 4 + width * 4, ds * 4, rows,
          width, flag.value, stream));
      require(flag.get(stream)[0] == 0, "finite reference flag");
      const auto actual = dst.get(stream);
      for (size_t r = 0; r < rows; ++r) {
        const auto direct = direct_reference(p, input.data() + r * ss, width);
        double before = 0., after = 0.;
        for (size_t i = 0; i < width; ++i) {
          const float y = actual[r * ds + i];
          close(y, expected[r * ds + i], "GPU versus CPU v3");
          close(y, direct[i], "GPU versus independent dense reference");
          if (identity || (i / 8) * 8 + 8 > width)
            require(same(y, input[r * ss + i]), "exact identity/tail");
          before += double(input[r * ss + i]) * input[r * ss + i];
          after += double(y) * y;
        }
        require(std::abs(after - before) <= 1e-6 * (1 + before), "row norm");
        for (size_t i = width; i < ds; ++i)
          require(actual[r * ds + i] == -93.f, "destination padding");
      }
      cu(spin8_query_v3_cuda_apply_f32(
          plan.value, &descriptor, src.value, src.bytes(), ss * 4, src.value,
          src.bytes(), ss * 4, rows, width, flag.value, stream));
      require(flag.get(stream)[0] == 0, "inplace flag");
      const auto inplace = src.get(stream);
      for (size_t r = 0; r < rows; ++r) {
        for (size_t i = 0; i < width; ++i)
          require(same(inplace[r * ss + i], actual[r * ds + i]),
                  "exact match inplace versus out-of-place");
        for (size_t i = width; i < ss; ++i)
          require(inplace[r * ss + i] == -71.f, "inplace padding");
      }
    }
  }
  std::puts("v3 fused CPU/dense reference, rows, identity, padding, tails, "
            "inplace and norm PASS");
}

void invalid_layouts(const Plan &plan, cudaStream_t stream) {
  Parameters p(8);
  p.upload(stream);
  auto descriptor = p.descriptor();
  Device<float> src(24), dst(24);
  Device<int> flag(1);
  src.put(std::vector<float>(24, .3f), stream);
  dst.put(std::vector<float>(24, -31.f), stream);
  flag.put({0}, stream);
  auto call = [&](const spin8_query_v3_cuda_parameters_f32 *params,
                  const float *s, size_t sc, size_t ss, float *d, size_t dc,
                  size_t ds, size_t rows, size_t width, int *f) {
    return spin8_query_v3_cuda_apply_f32(plan.value, params, s, sc, ss, d, dc,
                                         ds, rows, width, f, stream);
  };
  auto invalid = [](cudaError_t value) {
    require(value == cudaErrorInvalidValue, "reject invalid layout");
  };
  invalid(
      call(nullptr, src.value, 96, 32, dst.value, 96, 32, 2, 8, flag.value));
  invalid(call(&descriptor, src.value, 63, 32, dst.value, 96, 32, 2, 8,
               flag.value));
  invalid(call(&descriptor, src.value, 96, 31, dst.value, 96, 32, 2, 8,
               flag.value));
  invalid(call(&descriptor, src.value, 96, 32, dst.value, 63, 32, 2, 8,
               flag.value));
  invalid(call(&descriptor, src.value, 96, 32, dst.value, 96, 31, 2, 8,
               flag.value));
  invalid(call(&descriptor, src.value, 96, 32, dst.value, 96, 32, 0, 8,
               flag.value));
  invalid(call(&descriptor, src.value, 96, 32, dst.value, 96, 32, 2, 0,
               flag.value));
  invalid(call(&descriptor, src.value, SIZE_MAX, SIZE_MAX - 3, dst.value,
               SIZE_MAX, 32, SIZE_MAX, 8, flag.value));
  invalid(call(&descriptor, src.value, SIZE_MAX, 32, dst.value, SIZE_MAX, 32, 1,
               SIZE_MAX / 4, flag.value));
  invalid(call(&descriptor, src.value, 96, 32, src.value + 1, 92, 32, 2, 8,
               flag.value));
  invalid(call(&descriptor, src.value, 96, 32, src.value, 96, 36, 2, 8,
               flag.value));
  invalid(call(&descriptor, src.value, 96, 32, dst.value, 96, 32, 2, 8,
               reinterpret_cast<int *>(dst.value)));
  invalid(call(&descriptor,
               reinterpret_cast<const float *>(
                   reinterpret_cast<const char *>(src.value) + 1),
               95, 32, dst.value, 96, 32, 2, 8, flag.value));
  invalid(call(&descriptor, reinterpret_cast<const float *>(UINTPTR_MAX - 3),
               96, 32, dst.value, 96, 32, 2, 8, flag.value));
  for (int which = 0; which < 4; ++which) {
    auto bad = descriptor;
    if (which == 0)
      bad.win_bytes--;
    if (which == 1)
      bad.bin_bytes--;
    if (which == 2)
      bad.wout_bytes--;
    if (which == 3)
      bad.bout_bytes--;
    invalid(call(&bad, src.value, 96, 32, dst.value, 96, 32, 2, 8, flag.value));
  }
  auto alias = descriptor;
  alias.bin = alias.win;
  invalid(call(&alias, src.value, 96, 32, dst.value, 96, 32, 2, 8, flag.value));
  alias = descriptor;
  alias.bout = dst.value;
  invalid(call(&alias, src.value, 96, 32, dst.value, 96, 32, 2, 8, flag.value));
  for (float value : dst.get(stream))
    require(value == -31.f, "invalid host call leaves output unchanged");
  require(flag.get(stream)[0] == 0, "invalid host call leaves flag unchanged");
  spin8_query_v3_cuda_plan *bad_plan = plan.value;
  require(spin8_query_v3_cuda_plan_create(2, &bad_plan) ==
                  SPIN8_INVALID_ARGUMENT &&
              !bad_plan,
          "reject version and clear output");
  require(spin8_query_v3_cuda_plan_create(3, nullptr) == SPIN8_INVALID_ARGUMENT,
          "reject null plan output");
  spin8_query_v3_cuda_plan_free(nullptr);
  std::puts("v3 fused layout/version/capacity/overflow/alias rejection PASS");
}

void nonfinite_and_extremes(const Plan &plan, cudaStream_t stream) {
  constexpr size_t width = 23;
  Parameters p(width);
  Device<float> src(width), dst(width);
  Device<int> flag(1);
  std::vector<float> q(width, .3f);
  auto run = [&]() {
    p.upload(stream);
    src.put(q, stream);
    const auto descriptor = p.descriptor();
    cu(spin8_query_v3_cuda_apply_f32(
        plan.value, &descriptor, src.value, src.bytes(), src.bytes(), dst.value,
        dst.bytes(), dst.bytes(), 1, width, flag.value, stream));
    return flag.get(stream)[0];
  };
  for (int which = 0; which < 5; ++which) {
    p.nontrivial();
    q.assign(width, .3f);
    const float nan = std::numeric_limits<float>::quiet_NaN();
    const float inf = std::numeric_limits<float>::infinity();
    if (which == 0)
      p.win.back() = nan;
    if (which == 1)
      p.bin.back() = inf;
    if (which == 2)
      p.wout.back() = nan;
    if (which == 3)
      p.bout.back() = inf;
    if (which == 4)
      q.back() = nan;
    flag.put({16}, stream);
    require(run() == (16 | SPIN8_CUDA_NONFINITE), "nonfinite sticky OR flag");
  }
  // Identity still validates all immutable arrays on GPU; CPU create rejects
  // these arrays already, while CPU apply can skip finite Win*q overflow.
  p.nontrivial();
  std::fill(p.wout.begin(), p.wout.end(), 0.f);
  std::fill(p.bout.begin(), p.bout.end(), 0.f);
  p.win.back() = std::numeric_limits<float>::quiet_NaN();
  q.assign(width, .3f);
  flag.put({0}, stream);
  require(run() == SPIN8_CUDA_NONFINITE, "identity does not bypass NaN Win");
  p.win.back() = 0.f;
  p.bin.back() = std::numeric_limits<float>::infinity();
  flag.put({0}, stream);
  require(run() == SPIN8_CUDA_NONFINITE, "identity does not bypass Inf bin");

  // Finite F32 products/sums exceed F32 range but must survive FP64 controller
  // and the scaled chart. Rotated q stays small, so no output overflow occurs.
  std::fill(p.win.begin(), p.win.end(), std::numeric_limits<float>::max());
  std::fill(p.bin.begin(), p.bin.end(), std::numeric_limits<float>::max());
  std::fill(p.wout.begin(), p.wout.end(), std::numeric_limits<float>::max());
  std::fill(p.bout.begin(), p.bout.end(), std::numeric_limits<float>::max());
  flag.put({0}, stream);
  require(run() == 0, "large finite weights/chart accepted");
  const auto direct = direct_reference(p, q.data(), width);
  const auto actual = dst.get(stream);
  for (size_t i = 0; i < width; ++i)
    close(actual[i], direct[i], "large weight direct reference");
  std::fill(p.wout.begin(), p.wout.end(), 0.f);
  std::fill(p.bout.begin(), p.bout.end(), 0.f);
  q.assign(width, std::numeric_limits<float>::max());
  flag.put({0}, stream);
  require(run() == 0, "large finite q and weights identity accepted");
  const auto identity = dst.get(stream);
  require(std::memcmp(identity.data(), q.data(), q.size() * sizeof(float)) == 0,
          "large finite identity exact");

  // A valid orthogonal transform may overflow F32 storage and must flag it.
  std::fill(p.win.begin(), p.win.end(), 0.f);
  std::fill(p.bin.begin(), p.bin.end(), 0.f);
  p.bout[0] = .15f;
  double generators[448]{};
  require(spin8_query_v3_generators(generators) == SPIN8_OK, "overflow basis");
  q.assign(width, 0.f);
  q[0] = std::numeric_limits<float>::max();
  for (size_t j = 1; j < 8; ++j)
    if (generators[j] != 0.)
      q[j] = float(generators[j]) * std::numeric_limits<float>::max();
  flag.put({0}, stream);
  require(run() == SPIN8_CUDA_NONFINITE, "F32 output overflow flagged");
  std::puts("v3 fused strict nonfinite/identity, sticky flags, large finite "
            "controller/chart and output overflow PASS");
}

void tail_controls_rotation(const Plan &plan, cudaStream_t stream) {
  constexpr size_t width = 9;
  Parameters p(width);
  p.win[8] = 1.f;
  p.wout[0] = .2f;
  p.upload(stream);
  const auto descriptor = p.descriptor();
  std::vector<float> q(18, .3f);
  q[8] = -1.f;
  q[17] = 1.f;
  Device<float> src(q.size()), dst(q.size());
  Device<int> flag(1);
  src.put(q, stream);
  flag.put({0}, stream);
  cu(spin8_query_v3_cuda_apply_f32(
      plan.value, &descriptor, src.value, src.bytes(), width * 4, dst.value,
      dst.bytes(), width * 4, 2, width, flag.value, stream));
  require(flag.get(stream)[0] == 0, "tail-controlled flag");
  const auto out = dst.get(stream);
  double difference = 0.;
  for (size_t i = 0; i < 8; ++i)
    difference += std::abs(double(out[i]) - out[width + i]);
  require(difference > .01, "copied tail changes complete-block rotation");
  require(same(out[8], q[8]) && same(out[17], q[17]), "tail copied exactly");
  std::puts("v3 copied tail participates in controller PASS");
}

void large_weight_cancellation(const Plan &plan, cudaStream_t stream) {
  constexpr size_t width = 8;
  Parameters p(width);
  p.bin[0] = p.bin[1] = 1.f;
  p.wout[0] = std::numeric_limits<float>::max();
  p.wout[1] = -std::numeric_limits<float>::max();
  p.upload(stream);
  const auto descriptor = p.descriptor();
  const std::vector<float> input{.1f, .2f, .3f, .4f, .5f, .6f, .7f, .8f};
  auto expected = input;
  const auto cpu = cpu_controller(p, width);
  require(spin8_query_v3_controller_apply(
              cpu.get(), SPIN8_F32, input.data(), input.size() * sizeof(float),
              width, expected.data(), expected.size() * sizeof(float), width,
              1) == SPIN8_OK,
          "CPU cancellation reference");
  require(std::memcmp(input.data(), expected.data(),
                      input.size() * sizeof(float)) == 0,
          "CPU separately rounded products cancel exactly");
  Device<float> src(width), dst(width);
  Device<int> flag(1);
  src.put(input, stream);
  flag.put({0}, stream);
  cu(spin8_query_v3_cuda_apply_f32(
      plan.value, &descriptor, src.value, src.bytes(), src.bytes(), dst.value,
      dst.bytes(), dst.bytes(), 1, width, flag.value, stream));
  require(flag.get(stream)[0] == 0, "cancellation finite flag");
  const auto actual = dst.get(stream);
  for (size_t i = 0; i < width; ++i)
    close(actual[i], expected[i], "large Wout cancellation matches CPU");
  std::puts("v3 large opposite Wout products cancel as in CPU PASS");
}

void mixed_nonfinite_rows(const Plan &plan, cudaStream_t stream) {
  for (size_t width : {size_t{7}, size_t{23}}) {
    constexpr size_t rows = 3;
    const size_t ss = width + 3, ds = width + 5;
    Parameters p(width);
    p.nontrivial();
    p.upload(stream);
    const auto descriptor = p.descriptor();
    std::vector<float> input(rows * ss, -41.f);
    for (size_t row = 0; row < rows; ++row)
      for (size_t i = 0; i < width; ++i)
        input[row * ss + i] = float(.1 * double(i) + .2 * double(row));
    input[ss + width - 1] = std::numeric_limits<float>::infinity();
    Device<float> src(input.size()), dst(rows * ds);
    Device<int> flag(1);
    src.put(input, stream);
    dst.put(std::vector<float>(rows * ds, -53.f), stream);
    flag.put({0}, stream);
    cu(spin8_query_v3_cuda_apply_f32(plan.value, &descriptor, src.value,
                                     src.bytes(), ss * sizeof(float), dst.value,
                                     dst.bytes(), ds * sizeof(float), rows,
                                     width, flag.value, stream));
    require(flag.get(stream)[0] == SPIN8_CUDA_NONFINITE,
            "mixed-row finite flag");
    const auto out = dst.get(stream);
    // Diagnostic only: callers still discard the whole output on this flag.
    for (size_t row : {size_t{0}, size_t{2}}) {
      const auto direct = direct_reference(p, input.data() + row * ss, width);
      for (size_t i = 0; i < width; ++i)
        close(out[row * ds + i], direct[i], "valid rows around nonfinite row");
    }
  }
  std::puts("v3 tail-only and mixed finite/nonfinite row detection PASS");
}

void capped_grid_rows(const Plan &plan, cudaStream_t stream) {
  constexpr size_t rows = 65536, width = 8;
  Parameters p(width);
  p.nontrivial();
  p.upload(stream);
  const auto descriptor = p.descriptor();
  std::vector<float> input(rows * width);
  for (size_t row = 0; row < rows; ++row)
    for (size_t i = 0; i < width; ++i)
      input[row * width + i] = float(double(row) * .00001 + double(i) * .1);
  Device<float> data(input.size());
  Device<int> flag(1);
  data.put(input, stream);
  flag.put({0}, stream);
  cu(spin8_query_v3_cuda_apply_f32(
      plan.value, &descriptor, data.value, data.bytes(), width * sizeof(float),
      data.value, data.bytes(), width * sizeof(float), rows, width, flag.value,
      stream));
  require(flag.get(stream)[0] == 0, "capped-grid inplace flag");
  const auto out = data.get(stream);
  // CTA zero processes rows zero and 65535; the penultimate row belongs to
  // another CTA. Distinct row content detects stale shared controller state.
  for (size_t row : {size_t{0}, rows - 2, rows - 1}) {
    const auto direct = direct_reference(p, input.data() + row * width, width);
    for (size_t i = 0; i < width; ++i)
      close(out[row * width + i], direct[i], "capped-grid row reference");
  }
  std::puts("v3 65536-row capped-grid shared reset and inplace PASS");
}

void graph_single_launch(const Plan &plan, cudaStream_t stream) {
  Parameters p(8);
  p.nontrivial();
  p.upload(stream);
  const auto descriptor = p.descriptor();
  Device<float> src(8), dst(8);
  Device<int> flag(1);
  src.put(std::vector<float>(8, .2f), stream);
  flag.put({0}, stream);
  cudaGraph_t graph{};
  cudaGraphExec_t executable{};
  cu(cudaStreamBeginCapture(stream, cudaStreamCaptureModeThreadLocal));
  cu(spin8_query_v3_cuda_apply_f32(
      plan.value, &descriptor, src.value, src.bytes(), src.bytes(), dst.value,
      dst.bytes(), dst.bytes(), 1, 8, flag.value, stream));
  cu(cudaStreamEndCapture(stream, &graph));
  size_t nodes = 0;
  cu(cudaGraphGetNodes(graph, nullptr, &nodes));
  require(nodes == 1,
          "one kernel, no allocation/copy/synchronization in apply");
  cudaGraphNode_t node{};
  cu(cudaGraphGetNodes(graph, &node, &nodes));
  cudaGraphNodeType type{};
  cu(cudaGraphNodeGetType(node, &type));
  require(type == cudaGraphNodeTypeKernel, "single kernel graph node");
  cu(cudaGraphInstantiate(&executable, graph, nullptr, nullptr, 0));
  cu(cudaGraphLaunch(executable, stream));
  require(flag.get(stream)[0] == 0, "captured nondefault-stream flag");
  const auto direct = direct_reference(p, std::vector<float>(8, .2f).data(), 8);
  const auto actual = dst.get(stream);
  for (size_t i = 0; i < 8; ++i)
    close(actual[i], direct[i], "captured output");
  cu(cudaGraphExecDestroy(executable));
  cu(cudaGraphDestroy(graph));
  std::puts("v3 nondefault stream and captured single-kernel replay PASS");
}
// Independent storage strides deliberately differ for all four parameter
// arrays. NaN padding makes an accidental dense head stride observable.
struct Bank {
  size_t heads, width, ws, bs, os, ts;
  std::vector<float> win, bin, wout, bout;
  Device<float> dwin, dbin, dwout, dbout;
  Bank(size_t h, size_t w)
      : heads(h), width(w), ws(8 * w + 3), bs(13), os(61), ts(11),
        win(h * ws, std::numeric_limits<float>::quiet_NaN()),
        bin(h * bs, std::numeric_limits<float>::quiet_NaN()),
        wout(h * os, std::numeric_limits<float>::quiet_NaN()),
        bout(h * ts, std::numeric_limits<float>::quiet_NaN()), dwin(win.size()),
        dbin(bin.size()), dwout(wout.size()), dbout(bout.size()) {
    for (size_t head = 0; head < heads; ++head) {
      Parameters p(width);
      p.nontrivial();
      for (float &value : p.win)
        value *= float(head + 1) / float(heads);
      for (float &value : p.bout)
        value += float(head) * .013f;
      if (head % 3 == 0) {
        std::fill(p.wout.begin(), p.wout.end(), 0.f);
        std::fill(p.bout.begin(), p.bout.end(), -0.f);
      }
      std::copy(p.win.begin(), p.win.end(), win.begin() + head * ws);
      std::copy(p.bin.begin(), p.bin.end(), bin.begin() + head * bs);
      std::copy(p.wout.begin(), p.wout.end(), wout.begin() + head * os);
      std::copy(p.bout.begin(), p.bout.end(), bout.begin() + head * ts);
    }
  }
  void upload(cudaStream_t stream) {
    dwin.put(win, stream);
    dbin.put(bin, stream);
    dwout.put(wout, stream);
    dbout.put(bout, stream);
  }
  spin8_query_v3_cuda_bank_f32 descriptor() const {
    return {SPIN8_QUERY_V3_CUDA_BANK_VERSION,
            heads,
            width,
            dwin.value,
            ((heads - 1) * ws + 8 * width) * 4,
            ws * 4,
            dbin.value,
            ((heads - 1) * bs + 8) * 4,
            bs * 4,
            dwout.value,
            ((heads - 1) * os + 56) * 4,
            os * 4,
            dbout.value,
            ((heads - 1) * ts + 7) * 4,
            ts * 4};
  }
  Controller controller(size_t head) const {
    // Extract one head into the existing independent CPU controller API.
    Parameters p(width);
    std::copy_n(win.begin() + head * ws, p.win.size(), p.win.begin());
    std::copy_n(bin.begin() + head * bs, p.bin.size(), p.bin.begin());
    std::copy_n(wout.begin() + head * os, p.wout.size(), p.wout.begin());
    std::copy_n(bout.begin() + head * ts, p.bout.size(), p.bout.begin());
    return cpu_controller(p, width);
  }
};

void batched_reference(const Plan &plan, cudaStream_t stream) {
  constexpr size_t heads = 24;
  double max_error = 0.;
  for (size_t width : {size_t{256}, size_t{263}, size_t{7}})
    for (size_t tokens : {size_t{1}, size_t{4}}) {
      Bank bank(heads, width);
      bank.upload(stream);
      const auto descriptor = bank.descriptor();
      const size_t sh = width + 3, st = heads * sh + 11;
      const size_t dh = width + 7, dt = heads * dh + 17;
      std::vector<float> input(tokens * st, -71.f),
          expected(tokens * dt, -93.f);
      for (size_t token = 0; token < tokens; ++token)
        for (size_t head = 0; head < heads; ++head) {
          const size_t offset = token * st + head * sh;
          // Same q across heads forces output differences to come from bank.
          for (size_t i = 0; i < width; ++i)
            input[offset + i] =
                float(std::cos(.31 * double(i) + double(token)));
          input[offset] = -0.f;
          input[offset + 1] = 0.f;
          if (head % 3 == 0 && token == tokens - 1)
            for (size_t i = 0; i < width; ++i)
              input[offset + i] = i % 2 ? 0.f : -0.f;
        }
      for (size_t head = 0; head < heads; ++head) {
        auto cpu = bank.controller(head);
        require(spin8_query_v3_controller_apply(
                    cpu.get(), SPIN8_F32, input.data() + head * sh,
                    (input.size() - head * sh) * 4, st,
                    expected.data() + head * dh,
                    (expected.size() - head * dh) * 4, dt, tokens) == SPIN8_OK,
                "independent CPU per-head controller");
      }
      Device<float> src(input.size()), dst(expected.size());
      Device<int> flag(1);
      src.put(input, stream);
      dst.put(std::vector<float>(expected.size(), -93.f), stream);
      flag.put({0}, stream);
      // Capture proves the batch is one kernel on a non-default stream.
      cudaGraph_t graph{};
      cudaGraphExec_t executable{};
      cu(cudaStreamBeginCapture(stream, cudaStreamCaptureModeThreadLocal));
      cu(spin8_query_v3_cuda_apply_batched_f32(
          plan.value, &descriptor, src.value,
          ((tokens - 1) * st + (heads - 1) * sh + width) * 4, sh * 4, st * 4,
          dst.value, ((tokens - 1) * dt + (heads - 1) * dh + width) * 4, dh * 4,
          dt * 4, heads, tokens, width, flag.value, stream));
      cu(cudaStreamEndCapture(stream, &graph));
      size_t nodes = 0;
      cu(cudaGraphGetNodes(graph, nullptr, &nodes));
      require(nodes == 1, "batched single kernel without allocation/copy/sync");
      cudaGraphNode_t node{};
      cu(cudaGraphGetNodes(graph, &node, &nodes));
      cudaGraphNodeType type{};
      cu(cudaGraphNodeGetType(node, &type));
      require(type == cudaGraphNodeTypeKernel, "batched graph kernel node");
      cu(cudaGraphInstantiate(&executable, graph, nullptr, nullptr, 0));
      cu(cudaGraphLaunch(executable, stream));
      require(flag.get(stream)[0] == 0, "batched finite flag");
      const auto actual = dst.get(stream);
      for (size_t i = 0; i < actual.size(); ++i) {
        max_error =
            std::max(max_error, std::abs(double(actual[i]) - expected[i]));
        close(actual[i], expected[i], "batched per-head CPU reference/padding");
      }
      for (size_t token = 0; token < tokens; ++token)
        for (size_t head = 0; head < heads; ++head)
          for (size_t i = 0; i < width; ++i)
            if (head % 3 == 0 || i >= width / 8 * 8)
              require(same(actual[token * dt + head * dh + i],
                           input[token * st + head * sh + i]),
                      "batched exact signed-zero identity and tail");
      if (width >= 8)
        require(std::abs(actual[dh + 2] - actual[2 * dh + 2]) > 1e-5,
                "distinct heads use distinct parameters for identical query");
      cu(spin8_query_v3_cuda_apply_batched_f32(
          plan.value, &descriptor, src.value, src.bytes(), sh * 4, st * 4,
          src.value, src.bytes(), sh * 4, st * 4, heads, tokens, width,
          flag.value, stream));
      require(flag.get(stream)[0] == 0, "batched inplace flag");
      const auto inplace = src.get(stream);
      auto inplace_expected = input;
      for (size_t token = 0; token < tokens; ++token)
        for (size_t head = 0; head < heads; ++head)
          std::copy_n(actual.begin() + token * dt + head * dh, width,
                      inplace_expected.begin() + token * st + head * sh);
      require(std::memcmp(inplace.data(), inplace_expected.data(),
                          src.bytes()) == 0,
              "batched inplace equals out-of-place bitwise, padding unchanged");
      cu(cudaGraphExecDestroy(executable));
      cu(cudaGraphDestroy(graph));
    }
  std::printf(
      "v3 batched 24-head decode/prefill, width256/tails, CPU per-head "
      "reference, padded bank/tensor, inplace and single-kernel stream PASS "
      "max_abs_error=%.9g\n",
      max_error);
}

void batched_invalid_layouts(const Plan &plan, cudaStream_t stream) {
  Bank bank(2, 8);
  bank.upload(stream);
  const auto descriptor = bank.descriptor();
  Device<float> src(64), dst(64);
  Device<int> flag(1);
  src.put(std::vector<float>(64, .3f), stream);
  dst.put(std::vector<float>(64, -31.f), stream);
  flag.put({16}, stream);
  auto call = [&](const spin8_query_v3_cuda_bank_f32 *p, const float *s,
                  size_t sc, size_t sh, size_t st, float *d, size_t dc,
                  size_t dh, size_t dt, size_t heads, size_t tokens,
                  size_t width, int *f) {
    require(spin8_query_v3_cuda_apply_batched_f32(
                plan.value, p, s, sc, sh, st, d, dc, dh, dt, heads, tokens,
                width, f, stream) == cudaErrorInvalidValue,
            "reject batched invalid host contract");
  };
  auto bad_bank = [&](const spin8_query_v3_cuda_bank_f32 *p) {
    call(p, src.value, 128, 32, 64, dst.value, 128, 32, 64, 2, 2, 8,
         flag.value);
  };
  bad_bank(nullptr);
  bad_bank(reinterpret_cast<const spin8_query_v3_cuda_bank_f32 *>(
      reinterpret_cast<const char *>(&descriptor) + 1));
  for (int which = 0; which < 16; ++which) {
    auto bad = descriptor;
    if (which == 0)
      bad.version = 0;
    if (which == 1)
      bad.version = SPIN8_QUERY_V3_CUDA_BANK_VERSION + 1;
    if (which == 2)
      bad.heads = 1;
    if (which == 3)
      bad.width = 7;
    if (which == 4)
      --bad.win_bytes;
    if (which == 5)
      --bad.bin_bytes;
    if (which == 6)
      --bad.wout_bytes;
    if (which == 7)
      --bad.bout_bytes;
    if (which == 8)
      bad.win_head_stride_bytes = 8 * 8 * 4 - 4;
    if (which == 9)
      bad.bin_head_stride_bytes = 31;
    if (which == 10)
      bad.wout_head_stride_bytes = 0;
    if (which == 11)
      bad.bout_head_stride_bytes = SIZE_MAX - 3;
    if (which == 12)
      bad.bin = bad.win;
    if (which == 13)
      bad.bout = dst.value;
    if (which == 14)
      bad.win = reinterpret_cast<const float *>(UINTPTR_MAX - 3);
    if (which == 15)
      bad.win = nullptr;
    bad_bank(&bad);
  }
  auto alias = descriptor;
  alias.bin = alias.win + 8 * 8; // Inside Win's head-padding span.
  bad_bank(&alias);
  alias = descriptor;
  alias.bout = reinterpret_cast<const float *>(flag.value);
  bad_bank(&alias);
  alias = descriptor;
  alias.win = reinterpret_cast<const float *>(
      reinterpret_cast<const char *>(alias.win) + 1);
  bad_bank(&alias);
  call(&descriptor, src.value, 127, 32, 64, dst.value, 128, 32, 64, 2, 2, 8,
       flag.value);
  call(&descriptor, src.value, 128, 31, 64, dst.value, 128, 32, 64, 2, 2, 8,
       flag.value);
  call(&descriptor, src.value, 128, 32, 60, dst.value, 128, 32, 64, 2, 2, 8,
       flag.value);
  call(&descriptor, src.value, 128, 32, 65, dst.value, 128, 32, 64, 2, 2, 8,
       flag.value);
  call(&descriptor, src.value, 128, 32, 64, dst.value, 127, 32, 64, 2, 2, 8,
       flag.value);
  call(&descriptor, src.value, 128, 32, 64, dst.value, 128, 28, 64, 2, 2, 8,
       flag.value);
  call(&descriptor, src.value, 128, 32, 64, dst.value, 128, 32, 60, 2, 2, 8,
       flag.value);
  call(&descriptor, src.value, 128, 32, 64, src.value + 1, 128, 32, 64, 2, 2, 8,
       flag.value);
  call(&descriptor, src.value, 256, 32, 64, src.value, 256, 36, 72, 2, 2, 8,
       flag.value);
  call(&descriptor, src.value, 256, 32, 64, src.value, 256, 32, 68, 2, 2, 8,
       flag.value);
  call(&descriptor, src.value, 128, 32, 64, dst.value, 128, 32, 64, 2, 2, 8,
       reinterpret_cast<int *>(dst.value));
  call(&descriptor, src.value, SIZE_MAX, 32, SIZE_MAX - 3, dst.value, 128, 32,
       64, 2, 2, 8, flag.value);
  call(&descriptor, src.value, SIZE_MAX, 32, 64, dst.value, SIZE_MAX, 32, 64, 2,
       SIZE_MAX, 8, flag.value);
  call(&descriptor, src.value, 128, 32, 64, dst.value, 128, 32, 64, 2, 0, 8,
       flag.value);
  auto empty = descriptor;
  empty.heads = 0;
  call(&empty, src.value, 128, 32, 64, dst.value, 128, 32, 64, 0, 2, 8,
       flag.value);
  empty = descriptor;
  empty.width = 0;
  call(&empty, src.value, 128, 32, 64, dst.value, 128, 32, 64, 2, 2, 0,
       flag.value);
  auto bad = descriptor;
  bad.width = SIZE_MAX;
  call(&bad, src.value, SIZE_MAX, 32, 64, dst.value, SIZE_MAX, 32, 64, 2, 1,
       SIZE_MAX, flag.value);
  bad = descriptor;
  bad.width = SIZE_MAX / 4;
  call(&bad, src.value, SIZE_MAX, SIZE_MAX - 3, SIZE_MAX - 3, dst.value,
       SIZE_MAX, SIZE_MAX - 3, SIZE_MAX - 3, 2, 1, SIZE_MAX / 4, flag.value);
  bad = descriptor;
  bad.heads = SIZE_MAX;
  bad.width = SIZE_MAX;
  call(&bad, src.value, SIZE_MAX, 32, 64, dst.value, SIZE_MAX, 32, 64, SIZE_MAX,
       2, SIZE_MAX, flag.value);
  for (float value : dst.get(stream))
    require(value == -31.f, "batched host rejection leaves output unchanged");
  require(flag.get(stream)[0] == 16,
          "batched host rejection leaves flag unchanged");
  std::puts("v3 batched version/shape/capacity/stride/overlap/overflow "
            "rejection PASS");
}

void batched_nonfinite(const Plan &plan, cudaStream_t stream) {
  constexpr size_t heads = 3, width = 9, tokens = 2;
  Bank bank(heads, width);
  Device<float> src(heads * tokens * width), dst(heads * tokens * width);
  Device<int> flag(1);
  const auto original_win = bank.win, original_bin = bank.bin;
  const auto original_wout = bank.wout, original_bout = bank.bout;
  for (int which = 0; which < 7; ++which) {
    bank.win = original_win;
    bank.bin = original_bin;
    bank.wout = original_wout;
    bank.bout = original_bout;
    std::vector<float> input(src.count, .3f);
    const float nan = std::numeric_limits<float>::quiet_NaN();
    const float inf = std::numeric_limits<float>::infinity();
    // Corrupt a non-first head, including the all-zero output controller.
    if (which == 0)
      bank.win[bank.ws + 8 * width - 1] = nan;
    if (which == 1)
      bank.bin[2 * bank.bs + 7] = inf;
    if (which == 2)
      bank.wout[2 * bank.os + 55] = nan;
    if (which == 3)
      bank.bout[2 * bank.ts + 6] = inf;
    if (which == 4)
      input.back() = inf; // last token/head's identity tail
    if (which == 5)
      bank.win[8 * width - 1] = nan; // identity head
    if (which == 6) {
      // First generator's row-zero sign makes F32 output overflow in head 2.
      std::fill_n(bank.win.begin() + 2 * bank.ws, 8 * width, 0.f);
      std::fill_n(bank.bin.begin() + 2 * bank.bs, 8, 0.f);
      std::fill_n(bank.wout.begin() + 2 * bank.os, 56, 0.f);
      std::fill_n(bank.bout.begin() + 2 * bank.ts, 7, 0.f);
      bank.bout[2 * bank.ts] = .15f;
      double generators[448]{};
      require(spin8_query_v3_generators(generators) == SPIN8_OK,
              "batch overflow basis");
      input[2 * width] = std::numeric_limits<float>::max();
      for (size_t j = 1; j < 8; ++j)
        if (generators[j] != 0.)
          input[2 * width + j] =
              float(generators[j]) * std::numeric_limits<float>::max();
    }
    bank.upload(stream);
    src.put(input, stream);
    flag.put({16}, stream);
    const auto descriptor = bank.descriptor();
    cu(spin8_query_v3_cuda_apply_batched_f32(
        plan.value, &descriptor, src.value, src.bytes(), width * 4,
        heads * width * 4, dst.value, dst.bytes(), width * 4, heads * width * 4,
        heads, tokens, width, flag.value, stream));
    require(flag.get(stream)[0] == (16 | SPIN8_CUDA_NONFINITE),
            "nonfinite anywhere invalidates complete batch with sticky flag");
  }
  std::puts("v3 batched non-first-head parameter/input/tail/identity/result "
            "nonfinite complete-batch invalidation PASS");
}

void batched_tail_controls_rotation(const Plan &plan, cudaStream_t stream) {
  constexpr size_t heads = 2, width = 9, tokens = 2;
  Bank bank(heads, width);
  // Only the tail controls hidden unit zero, with distinct rotations per head.
  for (size_t head = 0; head < heads; ++head) {
    std::fill_n(bank.win.begin() + head * bank.ws, 8 * width, 0.f);
    std::fill_n(bank.bin.begin() + head * bank.bs, 8, 0.f);
    std::fill_n(bank.wout.begin() + head * bank.os, 56, 0.f);
    std::fill_n(bank.bout.begin() + head * bank.ts, 7, 0.f);
    bank.win[head * bank.ws + width - 1] = 1.f;
    bank.wout[head * bank.os] = .2f * float(head + 1);
  }
  bank.upload(stream);
  const auto descriptor = bank.descriptor();
  // Token stride is exactly the occupied padded head span, with no extra
  // final-head padding. This is valid and differs from heads*head_stride.
  constexpr size_t hs = width + 3, ts = hs + width;
  std::vector<float> input((tokens - 1) * ts + hs + width, -51.f);
  for (size_t token = 0; token < tokens; ++token)
    for (size_t head = 0; head < heads; ++head) {
      std::fill_n(input.begin() + token * ts + head * hs, width, .3f);
      input[token * ts + head * hs + 8] = token ? 1.f : -1.f;
    }
  Device<float> src(input.size()), dst(input.size());
  Device<int> flag(1);
  src.put(input, stream);
  dst.put(input, stream);
  flag.put({0}, stream);
  cu(spin8_query_v3_cuda_apply_batched_f32(
      plan.value, &descriptor, src.value, src.bytes(), hs * 4, ts * 4,
      dst.value, dst.bytes(), hs * 4, ts * 4, heads, tokens, width, flag.value,
      stream));
  require(flag.get(stream)[0] == 0, "batched tail control flag");
  const auto actual = dst.get(stream);
  for (size_t head = 0; head < heads; ++head) {
    double difference = 0.;
    for (size_t i = 0; i < 8; ++i)
      difference +=
          std::abs(double(actual[head * hs + i]) - actual[ts + head * hs + i]);
    require(difference > .01,
            "each head's tail changes its rotation across tokens");
    for (size_t token = 0; token < tokens; ++token)
      require(same(actual[token * ts + head * hs + 8],
                   input[token * ts + head * hs + 8]),
              "batched identity tail copied");
  }
  std::puts(
      "v3 batched tail controls each head, minimal padded token span PASS");
}
} // namespace

int main() {
  try {
    Stream stream;
    Plan plan;
    reference_cases(plan, stream.value);
    invalid_layouts(plan, stream.value);
    nonfinite_and_extremes(plan, stream.value);
    tail_controls_rotation(plan, stream.value);
    large_weight_cancellation(plan, stream.value);
    mixed_nonfinite_rows(plan, stream.value);
    capped_grid_rows(plan, stream.value);
    graph_single_launch(plan, stream.value);
    batched_reference(plan, stream.value);
    batched_invalid_layouts(plan, stream.value);
    batched_nonfinite(plan, stream.value);
    batched_tail_controls_rotation(plan, stream.value);
    std::puts("PASS spin8_query_v3_cuda_test");
    return 0;
  } catch (const std::exception &error) {
    std::fprintf(stderr, "FAIL spin8_query_v3_cuda_test: %s\n", error.what());
    return 1;
  }
}
