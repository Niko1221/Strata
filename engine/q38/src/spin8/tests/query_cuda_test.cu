#include "spin8/query.h"
#include "spin8/query_cuda.h"
#include "spin8/query_v3.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <cuda_runtime.h>
#include <limits>
#include <memory>
#include <random>
#include <stdexcept>
#include <vector>

namespace {
void require(bool condition, const char *message) {
  if (!condition)
    throw std::runtime_error(message);
}
void cu(cudaError_t result) {
  if (result != cudaSuccess)
    throw std::runtime_error(cudaGetErrorString(result));
}
struct Stream {
  cudaStream_t value{};
  Stream() { cu(cudaStreamCreateWithFlags(&value, cudaStreamNonBlocking)); }
  ~Stream() { cudaStreamDestroy(value); }
};
struct Plan {
  spin8_query_cuda_plan *value{};
  Plan() {
    require(spin8_query_cuda_plan_create(&value) == SPIN8_OK, "create plan");
  }
  ~Plan() { spin8_query_cuda_plan_free(value); }
};
template <typename T> struct Device {
  T *value{};
  size_t count{};
  explicit Device(size_t count_) : count(count_) {
    cu(cudaMalloc(reinterpret_cast<void **>(&value), count * sizeof(T)));
  }
  ~Device() { cudaFree(value); }
  Device(const Device &) = delete;
  Device &operator=(const Device &) = delete;
  void put(const std::vector<T> &data, cudaStream_t stream) {
    require(data.size() == count, "upload count");
    cu(cudaMemcpyAsync(value, data.data(), count * sizeof(T),
                       cudaMemcpyHostToDevice, stream));
  }
  std::vector<T> get(cudaStream_t stream) const {
    std::vector<T> data(count);
    cu(cudaMemcpyAsync(data.data(), value, count * sizeof(T),
                       cudaMemcpyDeviceToHost, stream));
    cu(cudaStreamSynchronize(stream));
    return data;
  }
};

cudaError_t launch(const Plan &plan, const Device<float> &spinors,
                   size_t c_stride, const float *src, size_t src_capacity,
                   size_t src_stride, float *dst, size_t dst_capacity,
                   size_t dst_stride, size_t rows, size_t width,
                   Device<int> &flag, cudaStream_t stream) {
  return spin8_query_cuda_transform_f32(
      plan.value, spinors.value, spinors.count * sizeof(float), c_stride, src,
      src_capacity, src_stride, dst, dst_capacity, dst_stride, rows, width,
      flag.value, stream);
}

void random_rows(const Plan &plan, cudaStream_t stream) {
  constexpr size_t rows = 100, width = 23, ss = 29, ds = 31, cs = 11;
  std::mt19937_64 rng(0x715A1ULL);
  std::normal_distribution<double> normal;
  std::vector<double> reference_spinors(rows * 8);
  std::vector<float> spinors(rows * cs, -99.0f);
  std::vector<float> src(rows * ss, -77.0f), sentinel(rows * ds, -333.0f);
  for (size_t row = 0; row < rows; ++row) {
    double norm2 = 0;
    for (size_t b = 0; b < 8; ++b) {
      const double value = normal(rng);
      reference_spinors[row * 8 + b] = value;
      norm2 += value * value;
    }
    for (size_t b = 0; b < 8; ++b) {
      const double value = reference_spinors[row * 8 + b] / std::sqrt(norm2);
      reference_spinors[row * 8 + b] = value;
      spinors[row * cs + b] = float(value);
    }
    for (size_t col = 0; col < width; ++col)
      src[row * ss + col] = float(normal(rng));
  }
  Device<float> dc(spinors.size()), dx(src.size()), dy(sentinel.size());
  Device<int> flag(1);
  dc.put(spinors, stream);
  dx.put(src, stream);
  dy.put(sentinel, stream);
  flag.put({0}, stream);
  cu(launch(plan, dc, cs, dx.value, dx.count * sizeof(float), ss, dy.value,
            dy.count * sizeof(float), ds, rows, width, flag, stream));
  require(flag.get(stream)[0] == 0, "random finite flag");
  const auto output = dy.get(stream);
  for (size_t row = 0; row < rows; ++row) {
    double matrix[64];
    require(spin8_query_matrix(reference_spinors.data() + row * 8, matrix) ==
                SPIN8_OK,
            "CPU query matrix");
    for (size_t col = 0; col < width; ++col) {
      const size_t block = (col / 8) * 8;
      if (block + 8 > width) {
        require(std::memcmp(&output[row * ds + col], &src[row * ss + col],
                            sizeof(float)) == 0,
                "tail exact");
        continue;
      }
      double expected = 0;
      for (size_t j = 0; j < 8; ++j)
        expected += matrix[(col % 8) * 8 + j] * src[row * ss + block + j];
      require(std::abs(double(output[row * ds + col]) - expected) <=
                  3e-6 * (1.0 + std::abs(expected)),
              "GPU vs CPU query reference");
    }
    for (size_t col = width; col < ds; ++col)
      require(output[row * ds + col] == -333.0f, "destination padding");
  }
  std::puts("random query rows: 100 CPU matrices, 2 complete groups, exact "
            "tail and padding PASS");
}

void identity_and_inplace(const Plan &plan, cudaStream_t stream) {
  constexpr size_t rows = 3, width = 17, stride = 20;
  std::vector<float> c(rows * 8, 0.0f), values(rows * stride, -123.0f);
  for (size_t r = 0; r < rows; ++r) {
    c[r * 8] = 1.0f;
    for (size_t j = 0; j < width; ++j)
      values[r * stride + j] = float(int(r * width + j) - 23) / 11.0f;
  }
  values[2] = -0.0f;
  Device<float> dc(c.size()), data(values.size());
  Device<int> flag(1);
  dc.put(c, stream);
  data.put(values, stream);
  flag.put({0}, stream);
  cu(launch(plan, dc, 8, data.value, data.count * sizeof(float), stride,
            data.value, data.count * sizeof(float), stride, rows, width, flag,
            stream));
  require(flag.get(stream)[0] == 0, "identity finite flag");
  const auto actual = data.get(stream);
  require(std::memcmp(actual.data(), values.data(),
                      values.size() * sizeof(float)) == 0,
          "c=e0 exact in-place identity including signed zero");
  std::puts("c=e0 exact in-place, signed zero, and tail PASS");
}

void nontrivial_inplace_tails(const Plan &plan, cudaStream_t stream) {
  const double angle = 0.23;
  double exact[8]{};
  exact[0] = std::cos(angle);
  exact[2] = std::sin(angle);
  double matrix[64]{};
  require(spin8_query_matrix(exact, matrix) == SPIN8_OK,
          "nontrivial CPU matrix");
  std::vector<float> c(8, 0.0f);
  c[0] = float(exact[0]);
  c[2] = float(exact[2]);
  Device<float> dc(8), data(16);
  Device<int> flag(1);
  dc.put(c, stream);
  for (size_t tail = 1; tail <= 7; ++tail) {
    const size_t width = 8 + tail;
    std::vector<float> input(16, -93.0f);
    for (size_t j = 0; j < width; ++j)
      input[j] = float(int(j) - 5) / 7.0f;
    data.put(input, stream);
    flag.put({0}, stream);
    cu(launch(plan, dc, 8, data.value, 16 * sizeof(float), 16, data.value,
              16 * sizeof(float), 16, 1, width, flag, stream));
    require(flag.get(stream)[0] == 0, "nontrivial inplace finite");
    const auto output = data.get(stream);
    for (size_t i = 0; i < 8; ++i) {
      double expected = 0;
      for (size_t j = 0; j < 8; ++j)
        expected += matrix[i * 8 + j] * input[j];
      require(std::abs(double(output[i]) - expected) <=
                  2e-6 * (1.0 + std::abs(expected)),
              "nontrivial inplace GPU vs CPU");
    }
    for (size_t i = 8; i < 16; ++i)
      require(std::memcmp(&output[i], &input[i], sizeof(float)) == 0,
              "tail and padding bitwise");
  }
  std::puts("nontrivial exact in-place and every tail length 1..7 PASS");
}

void invalids(const Plan &plan, cudaStream_t stream) {
  std::vector<float> c(8, 0.0f), x(17, 1.0f), y(17, -3.0f);
  c[0] = 1.0f;
  Device<float> dc(8), dx(17), dy(17);
  Device<int> flag(1);
  dc.put(c, stream);
  dx.put(x, stream);
  dy.put(y, stream);
  flag.put({0}, stream);
  const size_t cap = 17 * sizeof(float);
  require(launch(plan, dc, 8, dx.value, cap - 1, 17, dy.value, cap, 17, 1, 17,
                 flag, stream) == cudaErrorInvalidValue,
          "source capacity reject");
  require(launch(plan, dc, 8, dx.value, cap, 17, dy.value, cap - 1, 17, 1, 17,
                 flag, stream) == cudaErrorInvalidValue,
          "destination capacity reject");
  require(spin8_query_cuda_transform_f32(
              plan.value, dc.value, 8 * sizeof(float) - 1, 8, dx.value, cap, 17,
              dy.value, cap, 17, 1, 17, flag.value,
              stream) == cudaErrorInvalidValue,
          "spinor capacity reject");
  require(launch(plan, dc, 8, dx.value, cap, 16, dy.value, cap, 17, 1, 17, flag,
                 stream) == cudaErrorInvalidValue,
          "stride reject");
  require(launch(plan, dc, 8, dx.value, cap, 17, dx.value + 1, cap - 4, 17, 1,
                 16, flag, stream) == cudaErrorInvalidValue,
          "partial overlap reject");
  require(spin8_query_cuda_transform_f32(
              plan.value, dx.value, cap, 8, dx.value, cap, 17, dy.value, cap,
              17, 1, 17, flag.value, stream) == cudaErrorInvalidValue,
          "spinor source overlap reject");
  require(flag.get(stream)[0] == 0, "rejection preserved flag");
  require(dy.get(stream) == y, "rejection preserved destination");

  c[0] = 0.5f;
  dc.put(c, stream);
  cu(launch(plan, dc, 8, dx.value, cap, 17, dy.value, cap, 17, 1, 17, flag,
            stream));
  require(flag.get(stream)[0] & SPIN8_QUERY_CUDA_INVALID_SPINOR,
          "nonunit spinor flag");
  c[0] = std::numeric_limits<float>::quiet_NaN();
  dc.put(c, stream);
  cu(launch(plan, dc, 8, dx.value, cap, 17, dy.value, cap, 17, 1, 17, flag,
            stream));
  const int sticky = flag.get(stream)[0];
  require((sticky & SPIN8_QUERY_CUDA_INVALID_SPINOR) &&
              (sticky & SPIN8_CUDA_NONFINITE),
          "NaN spinor sticky flag");
  c[0] = 1.0f;
  dc.put(c, stream);
  x[16] = std::numeric_limits<float>::quiet_NaN();
  dx.put(x, stream);
  cu(launch(plan, dc, 8, dx.value, cap, 17, dy.value, cap, 17, 1, 17, flag,
            stream));
  require(flag.get(stream)[0] & SPIN8_CUDA_NONFINITE, "NaN tail flagged");
  std::puts("capacities, strides, overlaps, nonunit/NaN spinors, NaN tail, "
            "sticky flag PASS");
}

void v3_controller_handoff(const Plan &plan, cudaStream_t stream) {
  constexpr size_t rows = 8;
  for (const size_t width : {size_t{23}, size_t{256}}) {
    const size_t stride = width + 5;
    std::vector<double> win(8 * width), bin(8), wout(56), bout(7), c(rows * 8);
    std::vector<float> input(rows * stride, -91.f), expected(rows * stride, -93.f);
    for (size_t i = 0; i < win.size(); ++i) win[i] = .01 * std::sin(double(i));
    for (size_t r = 0; r < rows; ++r)
      for (size_t i = 0; i < width; ++i) input[r * stride + i] = float(std::cos(double(r * width + i)));
    for (int active = 0; active < 2; ++active) {
      for (size_t i = 0; i < wout.size(); ++i) wout[i] = active ? .04 * std::cos(double(i)) : 0.;
      for (size_t i = 0; i < bout.size(); ++i) bout[i] = active ? .01 * std::sin(double(i)) : 0.;
      spin8_query_v3_controller *raw = nullptr;
      require(spin8_query_v3_controller_create(SPIN8_QUERY_V3_VERSION, width,
        win.data(), win.size() * sizeof(double), bin.data(), bin.size() * sizeof(double),
        wout.data(), wout.size() * sizeof(double), bout.data(), bout.size() * sizeof(double),
        &raw) == SPIN8_OK, "create v3 handoff controller");
      const std::unique_ptr<spin8_query_v3_controller, decltype(&spin8_query_v3_controller_free)>
        controller(raw, spin8_query_v3_controller_free);
      require(spin8_query_v3_controller_apply(raw, SPIN8_F32, input.data(), input.size() * sizeof(float),
        stride, expected.data(), expected.size() * sizeof(float), stride, rows) == SPIN8_OK, "v3 CPU output");
      require(spin8_query_v3_controller_spinors(raw, SPIN8_F32, input.data(), input.size() * sizeof(float),
        stride, c.data(), c.size() * sizeof(double), 8, rows) == SPIN8_OK, "v3 CPU spinors");
      std::vector<float> cf(c.size()), sentinel(input.size(), -93.f);
      for (size_t i = 0; i < c.size(); ++i) cf[i] = float(c[i]);
      Device<float> dc(cf.size()), dx(input.size()), dy(sentinel.size()); Device<int> flag(1);
      dc.put(cf, stream); dx.put(input, stream); dy.put(sentinel, stream); flag.put({0}, stream);
      cu(launch(plan, dc, 8, dx.value, dx.count * sizeof(float), stride,
        dy.value, dy.count * sizeof(float), stride, rows, width, flag, stream));
      require(flag.get(stream)[0] == 0, "v3 handoff device flag");
      const auto actual = dy.get(stream);
      for (size_t r = 0; r < rows; ++r) {
        double before = 0, after = 0;
        for (size_t i = 0; i < width; ++i) {
          const float a = actual[r * stride + i], e = expected[r * stride + i];
          require(std::abs(double(a) - e) <= 3e-6 * (1 + std::abs(double(e))), "v3 handoff GPU vs CPU");
          if (!active || (i / 8) * 8 + 8 > width)
            require(std::memcmp(&a, &input[r * stride + i], sizeof(float)) == 0, "v3 exact zero/tail");
          before += double(input[r * stride + i]) * input[r * stride + i]; after += double(a) * a;
        }
        require(std::abs(after - before) <= 1e-5 * (1 + before), "v3 GPU norm invariant");
        for (size_t i = width; i < stride; ++i)
          require(actual[r * stride + i] == -93.f, "v3 handoff untouched padding");
      }
    }
  }
  std::puts("v3 CPU controller -> FP32 GPU contractions: widths23/256, identity/nonzero, norm/tails PASS");
}
} // namespace

int main() {
  try {
    Stream stream;
    Plan plan;
    random_rows(plan, stream.value);
    identity_and_inplace(plan, stream.value);
    nontrivial_inplace_tails(plan, stream.value);
    invalids(plan, stream.value);
    v3_controller_handoff(plan, stream.value);
    std::puts("PASS spin8_query_cuda_test");
    return 0;
  } catch (const std::exception &error) {
    std::fprintf(stderr, "FAIL spin8_query_cuda_test: %s\n", error.what());
    return 1;
  }
}
