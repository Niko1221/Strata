#include "spin8/cuda.h"
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <limits>
#include <stdexcept>
#include <vector>

namespace {
void require(bool ok, const char *text) {
  if (!ok)
    throw std::runtime_error(text);
}
void cu(cudaError_t e) {
  if (e != cudaSuccess)
    throw std::runtime_error(cudaGetErrorString(e));
}
struct Stream {
  cudaStream_t value{};
  Stream() { cu(cudaStreamCreateWithFlags(&value, cudaStreamNonBlocking)); }
  ~Stream() { cudaStreamDestroy(value); }
};
template <class T> struct Device {
  T *ptr{};
  size_t count;
  explicit Device(size_t n) : count(n) {
    cu(cudaMalloc(reinterpret_cast<void **>(&ptr), n * sizeof(T)));
  }
  ~Device() { cudaFree(ptr); }
  Device(const Device &) = delete;
  Device &operator=(const Device &) = delete;
  void put(const std::vector<T> &x, cudaStream_t s) {
    require(x.size() == count, "upload size");
    cu(cudaMemcpyAsync(ptr, x.data(), count * sizeof(T), cudaMemcpyHostToDevice,
                       s));
  }
  std::vector<T> get(cudaStream_t s) {
    std::vector<T> x(count);
    cu(cudaMemcpyAsync(x.data(), ptr, count * sizeof(T), cudaMemcpyDeviceToHost,
                       s));
    cu(cudaStreamSynchronize(s));
    return x;
  }
};
struct Plan {
  spin8_plan *ptr{};
  double m[192]{};
  Plan() {
    const spin8_plane planes[] = {{0, 1, .37}, {2, 6, -.81}, {1, 7, 1.13},
                                  {3, 4, -.2}, {0, 5, .51},  {4, 7, -.63}};
    require(spin8_plan_create(planes, 6, &ptr) == SPIN8_OK, "mixed plane plan");
    for (int v = 0; v < 3; ++v)
      require(spin8_plan_matrix(ptr, static_cast<spin8_view>(v), m + v * 64) ==
                  SPIN8_OK,
              "matrix");
  }
  ~Plan() { spin8_plan_free(ptr); }
};
unsigned seed = 0x514d1337;
size_t checks = 0;
double max_transform_error = 0, max_attention_error = 0, max_norm_error = 0,
       max_inner_error = 0;
float random_value() {
  seed ^= seed << 13;
  seed ^= seed >> 17;
  seed ^= seed << 5;
  return static_cast<float>(static_cast<int>(seed % 20001) - 10000) / 7000.0f;
}
void near(double actual, double expected, double tol, const char *name) {
  ++checks;
  if (!(std::isfinite(actual) &&
        std::abs(actual - expected) <= tol * (1 + std::abs(expected)))) {
    std::fprintf(stderr, "%s actual=%.12g expected=%.12g tolerance=%.3g\n",
                 name, actual, expected, tol);
    throw std::runtime_error(name);
  }
}
int flag_value(Device<int> &flag, cudaStream_t s) { return flag.get(s)[0]; }
std::vector<float> reference(const std::vector<float> &x, size_t rows,
                             size_t cols, size_t ss, size_t ds, const double *m,
                             bool inverse) {
  std::vector<float> y(rows * ds, -777);
  for (size_t r = 0; r < rows; ++r)
    for (size_t c = 0; c < cols; c += 8) {
      const size_t w = std::min(size_t(8), cols - c);
      for (size_t i = 0; i < w; ++i) {
        double sum = 0;
        if (w < 8)
          sum = x[r * ss + c + i];
        else
          for (size_t j = 0; j < 8; ++j)
            sum += m[inverse ? j * 8 + i : i * 8 + j] * x[r * ss + c + j];
        y[r * ds + c + i] = static_cast<float>(sum);
      }
    }
  return y;
}
void transforms(Plan &p, Stream &stream) {
  const size_t widths[] = {1, 7, 8, 9, 15, 16, 17, 31, 127, 256, 513};
  for (size_t cols : widths)
    for (size_t rows : {size_t(1), size_t(3), size_t(129)}) {
      const size_t ss = cols + 5, ds = cols + 11;
      std::vector<float> x(rows * ss, -555), z(rows * ss, -555);
      for (size_t r = 0; r < rows; ++r)
        for (size_t c = 0; c < cols; ++c) {
          x[r * ss + c] = random_value();
          z[r * ss + c] = random_value();
        }
      Device<float> dx(x.size()), dz(z.size()), a(rows * ds), b(rows * ds),
          c(rows * ds), back(rows * ss);
      Device<int> flag(1);
      dx.put(x, stream.value);
      dz.put(z, stream.value);
      for (int view = 0; view < 3; ++view) {
        std::vector<float> sentinel(rows * ds, -777);
        a.put(sentinel, stream.value);
        b.put(sentinel, stream.value);
        cu(spin8_cuda_transform_f32(dx.ptr, ss, a.ptr, ds, rows, cols,
                                    p.m + view * 64, 0, flag.ptr,
                                    stream.value));
        require(flag_value(flag, stream.value) == 0, "transform finite");
        auto actual = a.get(stream.value);
        auto expected =
            reference(x, rows, cols, ss, ds, p.m + view * 64, false);
        cu(spin8_cuda_transform_f32(dz.ptr, ss, b.ptr, ds, rows, cols,
                                    p.m + view * 64, 0, flag.ptr,
                                    stream.value));
        auto za = b.get(stream.value);
        double n0 = 0, n1 = 0, i0 = 0, i1 = 0;
        for (size_t r = 0; r < rows; ++r) {
          for (size_t j = 0; j < cols; ++j) {
            const auto at = r * ds + j, ix = r * ss + j;
            max_transform_error =
                std::max(max_transform_error,
                         std::abs(double(actual[at]) - expected[at]));
            near(actual[at], expected[at], 2e-6, "transform reference");
            n0 += double(x[ix]) * x[ix];
            n1 += double(actual[at]) * actual[at];
            i0 += double(x[ix]) * z[ix];
            i1 += double(actual[at]) * za[at];
          }
          for (size_t j = cols; j < ds; ++j)
            near(actual[r * ds + j], -777, 0, "padding unchanged");
        }
        max_norm_error = std::max(max_norm_error, std::abs(n0 - n1) / (1 + n0));
        max_inner_error =
            std::max(max_inner_error, std::abs(i0 - i1) / (1 + std::sqrt(n0)));
        near(n1, n0, 2e-6, "norm preserved");
        near(i1, i0, 2e-5, "inner product preserved");
        cu(spin8_cuda_transform_f32(a.ptr, ds, back.ptr, ss, rows, cols,
                                    p.m + view * 64, 1, flag.ptr,
                                    stream.value));
        auto round = back.get(stream.value);
        for (size_t r = 0; r < rows; ++r)
          for (size_t j = 0; j < cols; ++j)
            near(round[r * ss + j], x[r * ss + j], 3e-6, "roundtrip");
      }
      cu(spin8_cuda_transform3_f32(dx.ptr, ss, a.ptr, b.ptr, c.ptr, ds, rows,
                                   cols, p.m, flag.ptr, stream.value));
      require(flag_value(flag, stream.value) == 0, "fused3 flag");
      auto av = a.get(stream.value), bv = b.get(stream.value),
           cv = c.get(stream.value);
      const std::vector<float> *views[] = {&av, &bv, &cv};
      for (int view = 0; view < 3; ++view) {
        auto expected =
            reference(x, rows, cols, ss, ds, p.m + view * 64, false);
        for (size_t r = 0; r < rows; ++r)
          for (size_t j = 0; j < cols; ++j)
            near((*views[view])[r * ds + j], expected[r * ds + j], 2e-6,
                 "fused3 reference");
      }
      const double weights[] = {.2, .3, .5};
      cu(spin8_cuda_reduce3_f32(a.ptr, b.ptr, c.ptr, ds, back.ptr, ss, rows,
                                cols, p.m, weights, flag.ptr, stream.value));
      auto reduced = back.get(stream.value);
      for (size_t r = 0; r < rows; ++r)
        for (size_t j = 0; j < cols; ++j)
          near(reduced[r * ss + j], x[r * ss + j], 3e-6,
               "common space reduction");
      cu(spin8_cuda_transform_f32(dx.ptr, ss, dx.ptr, ss, rows, cols, p.m, 0,
                                  flag.ptr, stream.value));
      cu(spin8_cuda_transform_f32(dx.ptr, ss, dx.ptr, ss, rows, cols, p.m, 1,
                                  flag.ptr, stream.value));
      auto inplace = dx.get(stream.value);
      for (size_t i = 0; i < x.size(); ++i)
        near(inplace[i], x[i], 3e-6, "inplace");
    }
  std::printf("transforms: widths=1,7,8,9,15,16,17,31,127,256,513 rows=1,3,129 "
              "mixed_planes=6 max_abs=%.9g norm_rel=%.9g inner_scaled=%.9g\n",
              max_transform_error, max_norm_error, max_inner_error);
}
void hadamards(Stream &stream) {
  const size_t rows = 3, cols = 25, ss = 31;
  std::vector<float> x(rows * ss, -777);
  for (size_t r = 0; r < rows; ++r)
    for (size_t j = 0; j < cols; ++j)
      x[r * ss + j] = random_value();
  Device<float> dx(x.size()), dy(x.size());
  Device<int> flag(1);
  for (unsigned signs = 0; signs < 256; ++signs) {
    double m[64];
    require(spin8_hadamard_matrix(static_cast<uint8_t>(signs), 0, m) ==
                SPIN8_OK,
            "HD matrix");
    dx.put(x, stream.value);
    dy.put(std::vector<float>(x.size(), -777), stream.value);
    cu(spin8_cuda_hadamard_f32(dx.ptr, ss, dy.ptr, ss, rows, cols,
                               static_cast<uint8_t>(signs), 0, flag.ptr,
                               stream.value));
    auto y = dy.get(stream.value);
    auto ref = reference(x, rows, cols, ss, ss, m, false);
    for (size_t i = 0; i < x.size(); ++i)
      near(y[i], ref[i], 2e-6, "HD independent matrix");
    cu(spin8_cuda_hadamard_f32(dy.ptr, ss, dy.ptr, ss, rows, cols,
                               static_cast<uint8_t>(signs), 1, flag.ptr,
                               stream.value));
    y = dy.get(stream.value);
    for (size_t i = 0; i < x.size(); ++i)
      near(y[i], x[i], 3e-6, "DH inverse");
    require(flag_value(flag, stream.value) == 0, "HD flag");
  }
  std::puts("signed Hadamard: all 256 sign masks, HD forward / DH inverse, "
            "strided rows and identity tail passed");
}
template <class T> T cast_value(float f) { return static_cast<T>(f); }
template <> __half cast_value(float f) { return __float2half(f); }
template <class T> double as_double(T x) { return static_cast<double>(x); }
template <> double as_double(__half x) { return __half2float(x); }
template <class T>
void attention_case(Plan &p, Stream &stream, size_t d, size_t nk,
                    bool dynamic) {
  const size_t nq = 2, qs = d + 3, ks = d * 3 + 5, os = d + 7,
               active = dynamic ? nk - 1 : nk;
  std::vector<float> q(nq * qs, -777);
  std::vector<T> k(nk * ks), v(nk * ks);
  std::vector<uint8_t> mask(nq * nk, 1);
  for (size_t r = 0; r < nq; ++r)
    for (size_t j = 0; j < d; ++j)
      q[r * qs + j] = random_value();
  for (size_t r = 0; r < nk; ++r)
    for (size_t j = 0; j < d; ++j) {
      k[r * ks + j] =
          cast_value<T>(r < active ? random_value()
                                   : std::numeric_limits<float>::quiet_NaN());
      v[r * ks + j] =
          cast_value<T>(r < active ? random_value()
                                   : std::numeric_limits<float>::quiet_NaN());
    }
  for (size_t r = 0; r < nq; ++r)
    for (size_t j = 1; j < active; ++j)
      mask[r * nk + j] = static_cast<uint8_t>((j + r) % 3 != 0);
  std::vector<double> cq(nq * d), ck(active * d), cv(active * d),
      expected(nq * d);
  std::vector<uint8_t> cm(nq * active);
  for (size_t r = 0; r < nq; ++r) {
    for (size_t j = 0; j < d; ++j)
      cq[r * d + j] = q[r * qs + j];
    for (size_t j = 0; j < active; ++j)
      cm[r * active + j] = mask[r * nk + j];
  }
  for (size_t r = 0; r < active; ++r)
    for (size_t j = 0; j < d; ++j) {
      ck[r * d + j] = as_double(k[r * ks + j]);
      cv[r * d + j] = as_double(v[r * ks + j]);
    }
  const double weights[] = {.15, .25, .6};
  require(spin8_attention_f64(p.ptr, cq.data(), nq, ck.data(), active,
                              cv.data(), d, d, cm.data(), weights,
                              expected.data()) == SPIN8_OK,
          "CPU attention reference");
  Device<float> dq(q.size()), out(nq * os), scores(nq * nk);
  Device<T> dk(k.size()), dv(v.size());
  Device<uint8_t> dm(mask.size());
  Device<int> flag(1), count(1);
  dq.put(q, stream.value);
  dk.put(k, stream.value);
  dv.put(v, stream.value);
  dm.put(mask, stream.value);
  count.put({static_cast<int>(active)}, stream.value);
  for (int mode = 0; mode < 3; ++mode) {
    out.put(std::vector<float>(nq * os, -777), stream.value);
    cu(spin8_cuda_attention_f32(
        dq.ptr, qs, dk.ptr, dv.ptr, ks, sizeof(T) == 2 ? SPIN8_F16 : SPIN8_F32,
        out.ptr, os, nq, nk, d, dm.ptr, p.m, weights, mode, 0xa5, scores.ptr,
        dynamic ? count.ptr : nullptr, flag.ptr, 1, stream.value));
    require(flag_value(flag, stream.value) == 0, "attention flag");
    auto actual = out.get(stream.value), prob = scores.get(stream.value);
    for (size_t r = 0; r < nq; ++r) {
      double sum = 0;
      for (size_t j = 0; j < nk; ++j) {
        require(prob[r * nk + j] >= 0, "nonnegative probability");
        sum += prob[r * nk + j];
        if (j >= active || !mask[r * nk + j])
          near(prob[r * nk + j], 0, 0, "masked probability");
      }
      near(sum, 1, 2e-6, "one softmax sum");
      for (size_t j = 0; j < d; ++j) {
        max_attention_error =
            std::max(max_attention_error, std::abs(double(actual[r * os + j]) -
                                                   expected[r * d + j]));
        near(actual[r * os + j], expected[r * d + j], 8e-6,
             "attention vs CPU F64");
      }
      for (size_t j = d; j < os; ++j)
        near(actual[r * os + j], -777, 0, "attention padding");
    }
  }
  if (dynamic) {
    // Capture exact launch arguments once; device count changes without a host
    // read.
    cudaGraph_t graph{};
    cudaGraphExec_t executable{};
    cu(cudaStreamBeginCapture(stream.value, cudaStreamCaptureModeThreadLocal));
    cu(spin8_cuda_attention_f32(
        dq.ptr, qs, dk.ptr, dv.ptr, ks, sizeof(T) == 2 ? SPIN8_F16 : SPIN8_F32,
        out.ptr, os, nq, nk, d, dm.ptr, p.m, weights, 1, 0, scores.ptr,
        count.ptr, flag.ptr, 1, stream.value));
    cu(cudaStreamEndCapture(stream.value, &graph));
    cu(cudaGraphInstantiate(&executable, graph, 0));
    for (int n : {1, static_cast<int>(active), 0, static_cast<int>(nk + 1)}) {
      count.put({n}, stream.value);
      cu(cudaGraphLaunch(executable, stream.value));
      int f = flag_value(flag, stream.value);
      auto actual = out.get(stream.value);
      if (n == 1) {
        require(f == 0, "graph count one");
        for (size_t r = 0; r < nq; ++r)
          for (size_t j = 0; j < d; ++j)
            near(actual[r * os + j], as_double(v[j]), 2e-6,
                 "graph count changed");
      } else if (n == static_cast<int>(active)) {
        require(f == 0, "graph restored count");
        for (size_t r = 0; r < nq; ++r)
          for (size_t j = 0; j < d; ++j)
            near(actual[r * os + j], expected[r * d + j], 8e-6,
                 "graph full count reference");
      } else {
        require((f & SPIN8_CUDA_INVALID_COUNT) != 0,
                "graph invalid count flag");
        for (size_t r = 0; r < nq; ++r)
          for (size_t j = 0; j < d; ++j)
            near(actual[r * os + j], 0, 0, "invalid count zero");
      }
    }
    cu(cudaGraphExecDestroy(executable));
    cu(cudaGraphDestroy(graph));
  }
  dm.put(std::vector<uint8_t>(mask.size(), 0), stream.value);
  cu(spin8_cuda_attention_f32(dq.ptr, qs, dk.ptr, dv.ptr, ks,
                              sizeof(T) == 2 ? SPIN8_F16 : SPIN8_F32, out.ptr,
                              os, nq, nk, d, dm.ptr, p.m, weights, 1, 0,
                              scores.ptr, nullptr, flag.ptr, 1, stream.value));
  require((flag_value(flag, stream.value) & SPIN8_CUDA_ALL_MASKED) != 0,
          "all masked flag");
  auto zero = out.get(stream.value);
  for (size_t r = 0; r < nq; ++r)
    for (size_t j = 0; j < d; ++j)
      near(zero[r * os + j], 0, 0, "all masked zero");
}
void invalids(Plan &p, Stream &stream) {
  Device<float> x(64), y(64), z(64);
  Device<int> flag(1);
  std::vector<float> values(64, 1);
  x.put(values, stream.value);
  y.put(values, stream.value);
  z.put(values, stream.value);
  require(spin8_cuda_transform_f32(x.ptr, 7, y.ptr, 8, 1, 8, p.m, 0, flag.ptr,
                                   stream.value) == cudaErrorInvalidValue,
          "bad stride");
  require(spin8_cuda_transform_f32(x.ptr, 8, x.ptr + 1, 8, 1, 8, p.m, 0,
                                   flag.ptr,
                                   stream.value) == cudaErrorInvalidValue,
          "partial alias");
  require(spin8_cuda_transform_f32(x.ptr, 8, y.ptr, 8, SIZE_MAX, 8, p.m, 0,
                                   flag.ptr,
                                   stream.value) == cudaErrorInvalidValue,
          "overflow");
  require(spin8_cuda_transform_f32(x.ptr, 8, y.ptr, 8, 1, 8, p.m, 2, flag.ptr,
                                   stream.value) == cudaErrorInvalidValue,
          "bad inverse");
  double bad[192];
  std::copy(p.m, p.m + 192, bad);
  bad[0] = std::numeric_limits<double>::infinity();
  require(spin8_cuda_transform_f32(x.ptr, 8, y.ptr, 8, 1, 8, bad, 0, flag.ptr,
                                   stream.value) == cudaErrorInvalidValue,
          "nonfinite matrix");
  std::copy(p.m, p.m + 192, bad);
  bad[0] = 0;
  require(spin8_cuda_transform_f32(x.ptr, 8, y.ptr, 8, 1, 8, bad, 0, flag.ptr,
                                   stream.value) == cudaErrorInvalidValue,
          "nonorthogonal matrix");
  const double bad_weights[] = {1, 1, 1};
  require(spin8_cuda_reduce3_f32(x.ptr, x.ptr, x.ptr, 8, y.ptr, 8, 1, 8, p.m,
                                 bad_weights, flag.ptr,
                                 stream.value) == cudaErrorInvalidValue,
          "bad weights");
  require(spin8_cuda_transform3_f32(x.ptr, 8, y.ptr, y.ptr, z.ptr, 8, 1, 8, p.m,
                                    flag.ptr,
                                    stream.value) == cudaErrorInvalidValue,
          "fused aliases");
  require(spin8_cuda_transform_f32(x.ptr, 8, y.ptr, 8, 1, 8, p.m, 0,
                                   reinterpret_cast<int *>(y.ptr),
                                   stream.value) == cudaErrorInvalidValue,
          "flag aliases output");
  for (size_t at : {size_t(0), size_t(8)}) {
    values.assign(64, 1);
    values[at] = std::numeric_limits<float>::quiet_NaN();
    x.put(values, stream.value);
    cu(spin8_cuda_transform_f32(x.ptr, 9, y.ptr, 9, 1, 9, p.m, 0, flag.ptr,
                                stream.value));
    require(flag_value(flag, stream.value) & SPIN8_CUDA_NONFINITE,
            "NaN full block or tail");
  }
  values.assign(64, 1);
  values[1] = std::numeric_limits<float>::infinity();
  x.put(values, stream.value);
  cu(spin8_cuda_attention_f32(x.ptr, 8, y.ptr, z.ptr, 8, SPIN8_F32, y.ptr + 32,
                              8, 1, 1, 8, nullptr, p.m, nullptr, 1, 0,
                              z.ptr + 32, nullptr, flag.ptr, 1, stream.value));
  require(flag_value(flag, stream.value) & SPIN8_CUDA_NONFINITE,
          "attention nonfinite Q");
  values.assign(64, 1);
  x.put(values, stream.value);
  cu(spin8_cuda_attention_f32(x.ptr, 8, y.ptr, z.ptr, 8, SPIN8_F32, y.ptr + 32,
                              8, 1, 1, 8, nullptr, p.m, nullptr, 1, 0,
                              z.ptr + 32, nullptr, flag.ptr, 0, stream.value));
  require(flag_value(flag, stream.value) & SPIN8_CUDA_NONFINITE,
          "multihead sticky error");
  cu(spin8_cuda_attention_f32(x.ptr, 8, y.ptr, z.ptr, 8, SPIN8_F32, y.ptr + 32,
                              8, 1, 1, 8, nullptr, p.m, nullptr, 1, 0,
                              z.ptr + 32, nullptr, flag.ptr, 1, stream.value));
  require(flag_value(flag, stream.value) == 0, "explicit flag reset");
  Device<__half> half_k(8), half_v(8);
  std::vector<__half> half_values(8, __float2half(1.0f));
  half_values[3] = __float2half(std::numeric_limits<float>::quiet_NaN());
  half_k.put(half_values, stream.value);
  half_v.put(half_values, stream.value);
  cu(spin8_cuda_attention_f32(x.ptr, 8, half_k.ptr, half_v.ptr, 8, SPIN8_F16,
                              y.ptr + 32, 8, 1, 1, 8, nullptr, p.m, nullptr, 1,
                              0, z.ptr + 32, nullptr, flag.ptr, 1,
                              stream.value));
  require(flag_value(flag, stream.value) & SPIN8_CUDA_NONFINITE,
          "half KV NaN detected");
  values.assign(64, std::numeric_limits<float>::max());
  x.put(values, stream.value);
  cu(spin8_cuda_hadamard_f32(x.ptr, 8, y.ptr, 8, 1, 8, 0, 0, flag.ptr,
                             stream.value));
  require(flag_value(flag, stream.value) & SPIN8_CUDA_NONFINITE,
          "arithmetic overflow detected");
  require(spin8_cuda_attention_f32(x.ptr, 8, y.ptr, z.ptr, 8, SPIN8_BF16,
                                   y.ptr + 32, 8, 1, 1, 8, nullptr, p.m,
                                   nullptr, 1, 0, z.ptr + 32, nullptr, flag.ptr,
                                   1, stream.value) == cudaErrorInvalidValue,
          "unsupported KV type");
  std::puts("invalid inputs: strides, overflow, overlap, flags, inverse, "
            "matrices, weights, NaN, infinity, unsupported KV dtype and "
            "invalid dynamic counts rejected/detected");
}
} // namespace
int main() {
  try {
    int dev = 0;
    cu(cudaGetDevice(&dev));
    cudaDeviceProp prop{};
    cu(cudaGetDeviceProperties(&prop, dev));
    std::printf("GPU=%s compute=%d.%d CUDA_runtime=%d\n", prop.name, prop.major,
                prop.minor, CUDART_VERSION);
    Plan p;
    Stream stream;
    transforms(p, stream);
    hadamards(stream);
    for (size_t d : {size_t(1), size_t(7), size_t(8), size_t(9), size_t(17),
                     size_t(64), size_t(128)})
      for (size_t n : {size_t(1), size_t(3), size_t(137)}) {
        attention_case<float>(p, stream, d, n, false);
        attention_case<__half>(p, stream, d, n, false);
      }
    attention_case<float>(p, stream, 17, 7, true);
    attention_case<__half>(p, stream, 64, 7, true);
    invalids(p, stream);
    cu(cudaStreamSynchronize(stream.value));
    std::printf("attention: F32/F16, modes 0/1/2, strided GQA-like layouts, "
                "masks, tails, graph replay and dynamic key count; "
                "max_abs_vs_CPU_F64=%.9g\nPASS checks=%zu\n",
                max_attention_error, checks);
    return 0;
  } catch (const std::exception &e) {
    std::fprintf(stderr, "FAIL: %s\n", e.what());
    return 1;
  }
}
