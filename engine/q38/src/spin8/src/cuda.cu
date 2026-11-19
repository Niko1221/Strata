#include "spin8/cuda.h"
#include <cmath>
#include <cstdint>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <limits>
#include <math_constants.h>

namespace {
constexpr unsigned threads = 128;
struct Matrices {
  float m[192];
  float weights[3];
};
bool span(size_t rows, size_t cols, size_t stride, size_t bytes,
          size_t &result) {
  if (!rows || !cols || stride < cols || rows - 1 > (SIZE_MAX - cols) / stride)
    return false;
  const size_t count = (rows - 1) * stride + cols;
  if (count > SIZE_MAX / bytes)
    return false;
  result = count * bytes;
  return true;
}
bool overlap(const void *a, size_t an, const void *b, size_t bn) {
  const auto x = reinterpret_cast<uintptr_t>(a),
             y = reinterpret_cast<uintptr_t>(b);
  if (x > UINTPTR_MAX - an || y > UINTPTR_MAX - bn)
    return true;
  return x < y + bn && y < x + an;
}
bool matrices_load(const double *src, unsigned count, Matrices &p) {
  if (!src)
    return false;
  for (unsigned n = 0; n < count; ++n) {
    for (unsigned i = 0; i < 64; ++i) {
      const double value = src[n * 64 + i];
      if (!std::isfinite(value) || std::abs(value) > 1.00001)
        return false;
      p.m[n * 64 + i] = static_cast<float>(value);
    }
    // A transpose is an inverse only for orthogonal input matrices.
    for (unsigned i = 0; i < 8; ++i)
      for (unsigned j = 0; j < 8; ++j) {
        double dot = 0;
        for (unsigned l = 0; l < 8; ++l)
          dot += src[n * 64 + i * 8 + l] * src[n * 64 + j * 8 + l];
        if (std::abs(dot - (i == j ? 1.0 : 0.0)) > 1e-8)
          return false;
      }
  }
  return true;
}
bool weights_load(const double *weights, Matrices &p) {
  double sum = 0;
  for (unsigned i = 0; i < 3; ++i) {
    const double w = weights ? weights[i] : 1.0 / 3.0;
    if (!std::isfinite(w) || w < 0)
      return false;
    sum += w;
    p.weights[i] = static_cast<float>(w);
  }
  return std::abs(sum - 1.0) <= 1e-12;
}
bool layout(const float *src, size_t ss, float *dst, size_t ds, size_t rows,
            size_t cols, int *flag, size_t &sn, size_t &dn) {
  if (!src || !dst || !flag || !span(rows, cols, ss, 4, sn) ||
      !span(rows, cols, ds, 4, dn))
    return false;
  if (overlap(flag, sizeof(int), src, sn) ||
      overlap(flag, sizeof(int), dst, dn))
    return false;
  return !overlap(src, sn, dst, dn) || (src == dst && ss == ds);
}
unsigned blocks(size_t n) {
  const size_t b = n / threads + (n % threads != 0);
  return static_cast<unsigned>(b > 65535 ? 65535 : b);
}
__device__ void check(float x, int *flag) {
  if (!isfinite(x))
    atomicOr(flag, SPIN8_CUDA_NONFINITE);
}
__device__ size_t active_count(size_t cap, const int *count, int *flag) {
  if (!count)
    return cap;
  const int n = *count;
  if (n <= 0 || static_cast<size_t>(n) > cap) {
    atomicOr(flag, SPIN8_CUDA_INVALID_COUNT);
    return 0;
  }
  return static_cast<size_t>(n);
}
__device__ void rotate(const float x[8], float y[8], const float *m,
                       bool inverse) {
  for (int i = 0; i < 8; ++i) {
    float sum = 0;
    for (int j = 0; j < 8; ++j)
      sum += m[inverse ? j * 8 + i : i * 8 + j] * x[j];
    y[i] = sum;
  }
}
__device__ void hadamard(float x[8], uint8_t signs, bool inverse) {
  if (!inverse)
    for (int i = 0; i < 8; ++i)
      if (signs & (1u << i))
        x[i] = -x[i];
  for (int step = 1; step < 8; step *= 2)
    for (int b = 0; b < 8; b += step * 2)
      for (int i = 0; i < step; ++i) {
        float a = x[b + i], c = x[b + i + step];
        x[b + i] = a + c;
        x[b + i + step] = a - c;
      }
  for (int i = 0; i < 8; ++i) {
    x[i] *= 0.3535533905932737622f;
    if (inverse && (signs & (1u << i)))
      x[i] = -x[i];
  }
}
__global__ void transform_kernel(const float *src, size_t ss, float *a,
                                 float *b, float *c, size_t ds, size_t rows,
                                 size_t cols, Matrices p, int kind,
                                 bool inverse, uint8_t signs, int *flag) {
  const size_t tiles = cols / 8 + (cols % 8 != 0), count = rows * tiles;
  for (size_t t = blockIdx.x * blockDim.x + threadIdx.x; t < count;
       t += size_t(gridDim.x) * blockDim.x) {
    const size_t row = t / tiles, col = (t % tiles) * 8;
    float x[8] = {}, y[8];
    const int width = static_cast<int>(cols - col < 8 ? cols - col : 8);
    for (int j = 0; j < width; ++j) {
      x[j] = src[row * ss + col + j];
      check(x[j], flag);
    }
    const int views = kind == 2 ? 3 : 1;
    float *dst[3] = {a, b, c};
    for (int v = 0; v < views; ++v) {
      if (width != 8)
        for (int j = 0; j < width; ++j)
          y[j] = x[j];
      else if (kind == 1) {
        for (int j = 0; j < 8; ++j)
          y[j] = x[j];
        hadamard(y, signs, inverse);
      } else
        rotate(x, y, p.m + v * 64, inverse);
      for (int j = 0; j < width; ++j) {
        check(y[j], flag);
        dst[v][row * ds + col + j] = y[j];
      }
    }
  }
}
__global__ void reduce_kernel(const float *a, const float *b, const float *c,
                              size_t ss, float *dst, size_t ds, size_t rows,
                              size_t cols, Matrices p, int *flag) {
  const size_t tiles = cols / 8 + (cols % 8 != 0), count = rows * tiles;
  const float *src[3] = {a, b, c};
  for (size_t t = blockIdx.x * blockDim.x + threadIdx.x; t < count;
       t += size_t(gridDim.x) * blockDim.x) {
    const size_t row = t / tiles, col = (t % tiles) * 8;
    const int width = static_cast<int>(cols - col < 8 ? cols - col : 8);
    float sum[8] = {};
    for (int v = 0; v < 3; ++v) {
      float x[8] = {}, y[8];
      for (int j = 0; j < width; ++j) {
        x[j] = src[v][row * ss + col + j];
        check(x[j], flag);
      }
      if (width == 8)
        rotate(x, y, p.m + v * 64, true);
      else
        for (int j = 0; j < width; ++j)
          y[j] = x[j];
      for (int j = 0; j < width; ++j)
        sum[j] += p.weights[v] * y[j];
    }
    for (int j = 0; j < width; ++j) {
      check(sum[j], flag);
      dst[row * ds + col + j] = sum[j];
    }
  }
}
template <class T> __device__ float read(const T *ptr, size_t i) {
  return static_cast<float>(ptr[i]);
}
template <> __device__ float read(const __half *ptr, size_t i) {
  return __half2float(ptr[i]);
}
template <class T>
__global__ void validate_kernel(const float *q, size_t qs, const T *k,
                                const T *v, size_t ks, size_t nq, size_t cap,
                                size_t d, const int *count, int *flag) {
  const size_t nk = active_count(cap, count, flag);
  for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < (nq + 2 * nk) * d;
       i += size_t(gridDim.x) * blockDim.x) {
    if (i < nq * d)
      check(q[(i / d) * qs + i % d], flag);
    else {
      const size_t j = i - nq * d;
      check(j < nk * d ? read(k, (j / d) * ks + j % d)
                       : read(v, ((j - nk * d) / d) * ks + j % d),
            flag);
    }
  }
}
template <class T>
__global__ void
score_kernel(const float *q, size_t qs, const T *k, size_t ks, size_t nq,
             size_t nk, size_t d, const uint8_t *mask, Matrices p, int mode,
             uint8_t signs, float *scores, const int *count, int *flag) {
  const size_t active = active_count(nk, count, flag);
  for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < nq * nk;
       i += size_t(gridDim.x) * blockDim.x) {
    if (i % nk >= active || (mask && !mask[i])) {
      scores[i] = -CUDART_INF_F;
      continue;
    }
    const size_t qr = (i / nk) * qs, kr = (i % nk) * ks;
    float sum = 0;
    for (size_t col = 0; col < d; col += 8) {
      const int width = static_cast<int>(d - col < 8 ? d - col : 8);
      float x[8] = {}, y[8] = {};
      for (int j = 0; j < width; ++j) {
        x[j] = q[qr + col + j];
        y[j] = read(k, kr + col + j);
      }
      if (width == 8 && mode == 1) {
        for (int view = 0; view < 3; ++view) {
          float a[8], b[8];
          rotate(x, a, p.m + view * 64, false);
          rotate(y, b, p.m + view * 64, false);
          float dot = 0;
          for (int j = 0; j < 8; ++j)
            dot += a[j] * b[j];
          sum += p.weights[view] * dot;
        }
      } else {
        if (width == 8 && mode == 2) {
          hadamard(x, signs, false);
          hadamard(y, signs, false);
        }
        for (int j = 0; j < width; ++j)
          sum += x[j] * y[j];
      }
    }
    scores[i] = sum * rsqrtf(static_cast<float>(d));
    check(scores[i], flag);
  }
}
template <class T>
__global__ void softmax_value_kernel(const T *v, size_t vs, float *out,
                                     size_t os, size_t nq, size_t nk, size_t d,
                                     float *scores, const int *count,
                                     int *flag) {
  const size_t active = active_count(nk, count, flag);
  __shared__ float scratch[threads];
  __shared__ float maxval, denom;
  for (size_t row = blockIdx.x; row < nq; row += gridDim.x) {
    float m = -CUDART_INF_F;
    for (size_t i = threadIdx.x; i < nk; i += blockDim.x)
      m = fmaxf(m, scores[row * nk + i]);
    scratch[threadIdx.x] = m;
    __syncthreads();
    for (unsigned s = blockDim.x / 2; s; s /= 2) {
      if (threadIdx.x < s)
        scratch[threadIdx.x] =
            fmaxf(scratch[threadIdx.x], scratch[threadIdx.x + s]);
      __syncthreads();
    }
    if (threadIdx.x == 0) {
      maxval = scratch[0];
      if (active && maxval == -CUDART_INF_F)
        atomicOr(flag, SPIN8_CUDA_ALL_MASKED);
    }
    __syncthreads();
    float sum = 0;
    for (size_t i = threadIdx.x; i < nk; i += blockDim.x) {
      float prob =
          maxval == -CUDART_INF_F ? 0 : expf(scores[row * nk + i] - maxval);
      scores[row * nk + i] = prob;
      sum += prob;
    }
    scratch[threadIdx.x] = sum;
    __syncthreads();
    for (unsigned s = blockDim.x / 2; s; s /= 2) {
      if (threadIdx.x < s)
        scratch[threadIdx.x] += scratch[threadIdx.x + s];
      __syncthreads();
    }
    if (threadIdx.x == 0)
      denom = scratch[0];
    __syncthreads();
    for (size_t i = threadIdx.x; i < nk; i += blockDim.x)
      scores[row * nk + i] = denom > 0 ? scores[row * nk + i] / denom : 0;
    __syncthreads();
    for (size_t col = threadIdx.x; col < d; col += blockDim.x) {
      float value = 0;
      for (size_t i = 0; i < nk; ++i) {
        const float prob = scores[row * nk + i];
        if (prob != 0)
          value += prob * read(v, i * vs + col);
      }
      check(value, flag);
      out[row * os + col] = value;
    }
    __syncthreads();
  }
}
template <class T>
cudaError_t attention_launch(const float *q, size_t qs, const void *k,
                             const void *v, size_t ks, float *out, size_t os,
                             size_t nq, size_t nk, size_t d,
                             const uint8_t *mask, Matrices p, int mode,
                             uint8_t signs, float *scores, const int *count,
                             int *flag, cudaStream_t stream) {
  validate_kernel<<<blocks((nq + 2 * nk) * d), threads, 0, stream>>>(
      q, qs, static_cast<const T *>(k), static_cast<const T *>(v), ks, nq, nk,
      d, count, flag);
  auto e = cudaGetLastError();
  if (e != cudaSuccess)
    return e;
  score_kernel<<<blocks(nq * nk), threads, 0, stream>>>(
      q, qs, static_cast<const T *>(k), ks, nq, nk, d, mask, p, mode, signs,
      scores, count, flag);
  e = cudaGetLastError();
  if (e != cudaSuccess)
    return e;
  softmax_value_kernel<<<static_cast<unsigned>(nq < 65535 ? nq : 65535),
                         threads, 0, stream>>>(
      static_cast<const T *>(v), ks, out, os, nq, nk, d, scores, count, flag);
  return cudaGetLastError();
}
} // namespace
extern "C" cudaError_t
spin8_cuda_transform_f32(const float *src, size_t ss, float *dst, size_t ds,
                         size_t rows, size_t cols, const double *m, int inverse,
                         int *flag, cudaStream_t stream) {
  size_t sn, dn;
  Matrices p{};
  if (!layout(src, ss, dst, ds, rows, cols, flag, sn, dn) ||
      !matrices_load(m, 1, p) || (inverse != 0 && inverse != 1))
    return cudaErrorInvalidValue;
  auto e = cudaMemsetAsync(flag, 0, sizeof(int), stream);
  if (e != cudaSuccess)
    return e;
  transform_kernel<<<blocks(rows * (cols / 8 + (cols % 8 != 0))), threads, 0,
                     stream>>>(src, ss, dst, nullptr, nullptr, ds, rows, cols,
                               p, 0, inverse != 0, 0, flag);
  return cudaGetLastError();
}
extern "C" cudaError_t spin8_cuda_hadamard_f32(const float *src, size_t ss,
                                               float *dst, size_t ds,
                                               size_t rows, size_t cols,
                                               uint8_t signs, int inverse,
                                               int *flag, cudaStream_t stream) {
  size_t sn, dn;
  Matrices p{};
  if (!layout(src, ss, dst, ds, rows, cols, flag, sn, dn) ||
      (inverse != 0 && inverse != 1))
    return cudaErrorInvalidValue;
  auto e = cudaMemsetAsync(flag, 0, sizeof(int), stream);
  if (e != cudaSuccess)
    return e;
  transform_kernel<<<blocks(rows * (cols / 8 + (cols % 8 != 0))), threads, 0,
                     stream>>>(src, ss, dst, nullptr, nullptr, ds, rows, cols,
                               p, 1, inverse != 0, signs, flag);
  return cudaGetLastError();
}
extern "C" cudaError_t
spin8_cuda_transform3_f32(const float *src, size_t ss, float *a, float *b,
                          float *c, size_t ds, size_t rows, size_t cols,
                          const double *m, int *flag, cudaStream_t stream) {
  size_t sn, dn;
  Matrices p{};
  if (!layout(src, ss, a, ds, rows, cols, flag, sn, dn) ||
      !layout(src, ss, b, ds, rows, cols, flag, sn, dn) ||
      !layout(src, ss, c, ds, rows, cols, flag, sn, dn) ||
      overlap(a, dn, b, dn) || overlap(a, dn, c, dn) || overlap(b, dn, c, dn) ||
      !matrices_load(m, 3, p))
    return cudaErrorInvalidValue;
  auto e = cudaMemsetAsync(flag, 0, sizeof(int), stream);
  if (e != cudaSuccess)
    return e;
  transform_kernel<<<blocks(rows * (cols / 8 + (cols % 8 != 0))), threads, 0,
                     stream>>>(src, ss, a, b, c, ds, rows, cols, p, 2, false, 0,
                               flag);
  return cudaGetLastError();
}
extern "C" cudaError_t
spin8_cuda_reduce3_f32(const float *a, const float *b, const float *c,
                       size_t ss, float *dst, size_t ds, size_t rows,
                       size_t cols, const double *m, const double *weights,
                       int *flag, cudaStream_t stream) {
  size_t sn, dn;
  Matrices p{};
  if (!layout(a, ss, dst, ds, rows, cols, flag, sn, dn) ||
      !layout(b, ss, dst, ds, rows, cols, flag, sn, dn) ||
      !layout(c, ss, dst, ds, rows, cols, flag, sn, dn) ||
      overlap(dst, dn, a, sn) || overlap(dst, dn, b, sn) ||
      overlap(dst, dn, c, sn) || !matrices_load(m, 3, p) ||
      !weights_load(weights, p))
    return cudaErrorInvalidValue;
  auto e = cudaMemsetAsync(flag, 0, sizeof(int), stream);
  if (e != cudaSuccess)
    return e;
  reduce_kernel<<<blocks(rows * (cols / 8 + (cols % 8 != 0))), threads, 0,
                  stream>>>(a, b, c, ss, dst, ds, rows, cols, p, flag);
  return cudaGetLastError();
}
extern "C" cudaError_t spin8_cuda_attention_f32(
    const float *q, size_t qs, const void *k, const void *v, size_t ks,
    spin8_dtype dtype, float *out, size_t os, size_t nq, size_t nk, size_t d,
    const uint8_t *mask, const double *m, const double *weights, int mode,
    uint8_t signs, float *scores, const int *count, int *flag, int reset,
    cudaStream_t stream) {
  size_t qn, kn, on, sn;
  Matrices p{};
  const size_t elem = dtype == SPIN8_F16 ? 2 : 4;
  if ((dtype != SPIN8_F32 && dtype != SPIN8_F16) || !q || !k || !v || !out ||
      !scores || !flag || mode < 0 || mode > 2 || (reset != 0 && reset != 1) ||
      !span(nq, d, qs, 4, qn) || !span(nk, d, ks, elem, kn) ||
      !span(nq, d, os, 4, on) || !span(nq, nk, nk, 4, sn) ||
      nk > (SIZE_MAX - nq) / 2 || (nq + 2 * nk) > SIZE_MAX / d ||
      (mode == 1 && (!matrices_load(m, 3, p) || !weights_load(weights, p))))
    return cudaErrorInvalidValue;
  const void *inputs[4] = {q, k, v, mask};
  const size_t sizes[4] = {qn, kn, kn, mask ? nq * nk : 0};
  for (int i = 0; i < 4; ++i)
    if (sizes[i] && (overlap(out, on, inputs[i], sizes[i]) ||
                     overlap(scores, sn, inputs[i], sizes[i]) ||
                     overlap(flag, 4, inputs[i], sizes[i])))
      return cudaErrorInvalidValue;
  if (overlap(out, on, scores, sn) || overlap(out, on, flag, 4) ||
      overlap(scores, sn, flag, 4))
    return cudaErrorInvalidValue;
  if (count && (overlap(out, on, count, 4) || overlap(scores, sn, count, 4) ||
                overlap(flag, 4, count, 4)))
    return cudaErrorInvalidValue;
  if (reset) {
    auto e = cudaMemsetAsync(flag, 0, sizeof(int), stream);
    if (e != cudaSuccess)
      return e;
  }
  if (dtype == SPIN8_F16)
    return attention_launch<__half>(q, qs, k, v, ks, out, os, nq, nk, d, mask,
                                    p, mode, signs, scores, count, flag,
                                    stream);
  return attention_launch<float>(q, qs, k, v, ks, out, os, nq, nk, d, mask, p,
                                 mode, signs, scores, count, flag, stream);
}
