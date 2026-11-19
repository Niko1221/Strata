#include "spin8/attention_v2_cuda.h"
#include <algorithm>
#include <array>
#include <cfloat>
#include <cmath>
#include <cstring>
#include <cuda_fp16.h>
#include <memory>
#include <new>
#include <stdexcept>
#include <vector>

namespace {
struct Geometry {
  uint64_t source;
  size_t nh, kvh, dq, dv;
  const double *matrix;
  const uint32_t *map;
  bool fixed, h128;
  double hscale;
};
struct Layout {
  size_t q, k, v, result, scores, elements;
};
struct Failure {
  spin8_status status;
  cudaError_t cuda_error = cudaSuccess;
};
void require(bool ok, spin8_status status = SPIN8_INVALID_ARGUMENT) {
  if (!ok)
    throw Failure{status};
}
void runtime(cudaError_t e) {
  if (e != cudaSuccess)
    throw Failure{SPIN8_INVALIDATED, e};
}
void success(spin8_status s) { require(s == SPIN8_OK, s); }
size_t add(size_t a, size_t b) {
  require(a <= SIZE_MAX - b);
  return a + b;
}
size_t mul(size_t a, size_t b) {
  require(!b || a <= SIZE_MAX / b);
  return a * b;
}
void span(const void *p, size_t n, size_t alignment = 1) {
  const auto address = reinterpret_cast<uintptr_t>(p);
  require(p && n && n <= size_t(PTRDIFF_MAX) && address % alignment == 0 &&
          address <= UINTPTR_MAX - n);
}
template <class T> void object(const T *p, size_t n = sizeof(T)) {
  require(n == sizeof(T));
  span(static_cast<const void *>(p), n, alignof(T));
}
bool overlap(const void *a, size_t an, const void *b, size_t bn) {
  const auto x = reinterpret_cast<uintptr_t>(a),
             y = reinterpret_cast<uintptr_t>(b);
  return x < y + bn && y < x + an;
}
template <class T> void header(const T &s, uint32_t version) {
  require(s.struct_bytes == sizeof(T));
  require(s.abi_version == version, SPIN8_UNSUPPORTED);
  for (auto value : s.reserved)
    require(value == 0);
}
Layout layout(const Geometry &g, size_t nq, size_t nk) {
  require(nq && nk);
  Layout l{};
  l.k = mul(mul(nq, g.nh), g.dq);
  l.v = add(l.k, mul(mul(nk, g.kvh), g.dq));
  l.result = add(l.v, mul(mul(nk, g.kvh), g.dv));
  l.scores = add(l.result, mul(mul(nq, g.nh), g.dv));
  l.elements = add(l.scores, mul(mul(nq, g.nh), nk));
  require(l.elements <= size_t(PTRDIFF_MAX) / sizeof(double));
  return l;
}
template <class F> spin8_attention_v2_cuda_result boundary(F &&f) noexcept {
  try {
    f();
    return {SPIN8_OK, cudaSuccess};
  } catch (const Failure &e) {
    return {e.status, e.cuda_error};
  } catch (const std::bad_alloc &) {
    return {SPIN8_ALLOCATION_FAILED, cudaSuccess};
  } catch (const std::length_error &) {
    return {SPIN8_ALLOCATION_FAILED, cudaSuccess};
  } catch (...) {
    return {SPIN8_INVALID_ARGUMENT, cudaSuccess};
  }
}
} // namespace

struct spin8_attention_v2_cuda_plan {
  Geometry g{};
  int device = -1;
  void *storage = nullptr;
  size_t storage_bytes = 0;
};

namespace {
void current_device(int device) {
  int current = -1;
  runtime(cudaGetDevice(&current));
  require(device == current, SPIN8_UNSUPPORTED);
}
bool absent(const spin8_attention_v2_input &b) {
  if (b.data)
    return false;
  require(!b.capacity_bytes && !b.token_stride_bytes && !b.head_stride_bytes &&
          !b.channel_stride_bytes);
  return true;
}
void tensor(const spin8_attention_v2_input &b, size_t t, size_t h, size_t c,
            size_t item) {
  span(b.data, b.capacity_bytes, item);
  require(t && h && c && b.token_stride_bytes && b.head_stride_bytes &&
          b.channel_stride_bytes && b.token_stride_bytes % item == 0 &&
          b.head_stride_bytes % item == 0 &&
          b.channel_stride_bytes % item == 0);
  const size_t end = add(
      add(mul(t - 1, b.token_stride_bytes), mul(h - 1, b.head_stride_bytes)),
      add(mul(c - 1, b.channel_stride_bytes), item));
  require(end <= b.capacity_bytes);
}
spin8_attention_v2_input output_layout(const spin8_attention_v2_output &b,
                                       size_t t, size_t h, size_t c) {
  spin8_attention_v2_input out{b.data, b.capacity_bytes, b.token_stride_bytes,
                               b.head_stride_bytes, b.channel_stride_bytes};
  tensor(out, t, h, c, 4);
  std::array<std::pair<size_t, size_t>, 3> axes{{{b.token_stride_bytes, t},
                                                 {b.head_stride_bytes, h},
                                                 {b.channel_stride_bytes, c}}};
  std::sort(axes.begin(), axes.end());
  size_t occupied = 4;
  for (const auto &axis : axes)
    if (axis.second > 1) {
      require(axis.first >= occupied);
      occupied = add(occupied, mul(axis.first, axis.second - 1));
    }
  return out;
}
void device_pointer(const void *p, int device) {
  cudaPointerAttributes a{};
  runtime(cudaPointerGetAttributes(&a, p));
  require(a.type == cudaMemoryTypeDevice && a.device == device,
          SPIN8_UNSUPPORTED);
}
Layout validate(const spin8_attention_v2_cuda_plan *p,
                const spin8_attention_v2_cuda_request *r, size_t bytes) {
  object(p);
  object(r, bytes);
  header(*r, SPIN8_ATTENTION_V2_CUDA_ABI_VERSION);
  require(!r->reserved0);
  require(r->operation == SPIN8_ATTENTION_V2_CUDA_REFERENCE &&
              (r->k_dtype == SPIN8_F32 || r->k_dtype == SPIN8_F16) &&
              (r->v_dtype == SPIN8_F32 || r->v_dtype == SPIN8_F16),
          SPIN8_UNSUPPORTED);
  const auto &a = r->attention;
  header(a, SPIN8_ATTENTION_V2_ABI_VERSION);
  require(a.position_stage == SPIN8_ATTENTION_V2_POST_POSITION &&
              a.q_source_id == p->g.source && a.kv_source_id == p->g.source &&
              a.logit_order <= SPIN8_ATTENTION_V2_SCALE_SOFTCAP_BIAS,
          SPIN8_UNSUPPORTED);
  require(std::isfinite(a.scale) && std::isfinite(a.softcap), SPIN8_NONFINITE);
  require(a.softcap >= 0);
  const auto l = layout(p->g, a.n_query, a.n_key);
  const auto out = output_layout(a.output, a.n_query, p->g.nh, p->g.dv);
  tensor(a.q, a.n_query, p->g.nh, p->g.dq, 4);
  tensor(a.k, a.n_key, p->g.kvh, p->g.dq, r->k_dtype == SPIN8_F16 ? 2 : 4);
  tensor(a.v, a.n_key, p->g.kvh, p->g.dv, r->v_dtype == SPIN8_F16 ? 2 : 4);
  tensor(a.mask, a.n_query, p->g.nh, a.n_key, 1);
  if (!absent(a.bias))
    tensor(a.bias, a.n_query, p->g.nh, a.n_key, 4);
  if (!absent(a.gate))
    tensor(a.gate, a.n_query, p->g.nh, p->g.dv, 4);
  span(r->workspace, r->workspace_bytes, alignof(double));
  span(r->device_status, r->device_status_bytes, alignof(uint32_t));
  require(r->workspace_bytes >= l.elements * sizeof(double) &&
          r->device_status_bytes >= sizeof(uint32_t));
  const void *writes[] = {out.data, r->workspace, r->device_status};
  const size_t sizes[] = {out.capacity_bytes, r->workspace_bytes,
                          r->device_status_bytes};
  for (size_t i = 0; i < 3; ++i) {
    require(!overlap(writes[i], sizes[i], r, bytes) &&
            !overlap(writes[i], sizes[i], p, sizeof(*p)) &&
            !overlap(writes[i], sizes[i], p->storage, p->storage_bytes));
    for (size_t j = 0; j < i; ++j)
      require(!overlap(writes[i], sizes[i], writes[j], sizes[j]));
    for (const auto *input : {&a.q, &a.k, &a.v, &a.mask, &a.bias, &a.gate})
      if (input->data)
        require(
            !overlap(writes[i], sizes[i], input->data, input->capacity_bytes));
  }
  require(r->stream && r->stream != cudaStreamLegacy &&
              r->stream != cudaStreamPerThread,
          SPIN8_UNSUPPORTED);
  current_device(p->device);
  int stream_device = -1;
  runtime(cudaStreamGetDevice(r->stream, &stream_device));
  require(stream_device == p->device, SPIN8_UNSUPPORTED);
  for (const auto *input : {&a.q, &a.k, &a.v, &a.mask, &a.bias, &a.gate})
    if (input->data)
      device_pointer(input->data, p->device);
  for (const auto *write : writes)
    device_pointer(write, p->device);
  return l;
}
constexpr unsigned threads = 128;
unsigned blocks(size_t n) {
  return static_cast<unsigned>(std::min(size_t(65535), (n - 1) / threads + 1));
}
__device__ size_t offset(spin8_attention_v2_input b, size_t t, size_t h,
                         size_t c) {
  return t * b.token_stride_bytes + h * b.head_stride_bytes +
         c * b.channel_stride_bytes;
}
__device__ double load(spin8_attention_v2_input b, size_t t, size_t h,
                       size_t c) {
  return *reinterpret_cast<const float *>(static_cast<const char *>(b.data) +
                                          offset(b, t, h, c));
}
__device__ bool allowed(spin8_attention_v2_input b, size_t t, size_t h,
                        size_t c) {
  return static_cast<const unsigned char *>(b.data)[offset(b, t, h, c)] != 0;
}
__device__ void finite(double value, uint32_t *flag) {
  if (!isfinite(value))
    atomicOr(flag, SPIN8_ATTENTION_V2_CUDA_NONFINITE);
}
__global__ void pack_kernel(spin8_attention_v2_input b, size_t n, size_t heads,
                            size_t width, double *dst, uint32_t *flag,
                            uint32_t dtype = SPIN8_F32) {
  for (size_t i = size_t(blockIdx.x) * blockDim.x + threadIdx.x; i < n;
       i += size_t(gridDim.x) * blockDim.x) {
    const size_t t = i / (heads * width), h = (i / width) % heads,
                 c = i % width;
    const double x =
        dtype == SPIN8_F16
            ? double(__half2float(*reinterpret_cast<const __half *>(
                  static_cast<const char *>(b.data) + offset(b, t, h, c))))
            : load(b, t, h, c);
    finite(x, flag);
    if (dst)
      dst[i] = x;
  }
}
__device__ void hadamard_row(double *row, size_t width, double scale,
                             uint32_t *flag) {
  // The caller workspace holds the butterfly state. No width-sized local
  // array, dynamic allocation or cross-thread row dependency is needed.
  for (size_t base = 0; base + 128 <= width; base += 128) {
    for (size_t step = 1; step < 128; step *= 2)
      for (size_t block = 0; block < 128; block += 2 * step)
        for (size_t j = 0; j < step; ++j) {
          const size_t at = base + block + j;
          const double a = row[at], b = row[at + step];
          row[at] = __dadd_rn(a, b);
          row[at + step] = __dadd_rn(a, -b);
          finite(row[at], flag);
          finite(row[at + step], flag);
        }
    for (size_t j = 0; j < 128; ++j) {
      row[base + j] *= scale;
      finite(row[base + j], flag);
    }
  }
}
__global__ void transform_kernel(Geometry g, double *data, size_t rows,
                                 size_t width, bool inverse, uint32_t *flag) {
  for (size_t row = size_t(blockIdx.x) * blockDim.x + threadIdx.x; row < rows;
       row += size_t(gridDim.x) * blockDim.x) {
    double *dst = data + row * width;
    if (g.h128 && !inverse)
      hadamard_row(dst, width, g.hscale, flag);
    for (size_t base = 0; base + 8 <= width; base += 8) {
      double x[8];
      for (size_t j = 0; j < 8; ++j)
        x[j] = dst[base + j];
      for (size_t i = 0; i < 8; ++i) {
        double value = 0;
        for (size_t j = 0; j < 8; ++j)
          value = __dadd_rn(
              value,
              __dmul_rn(g.matrix[inverse ? j * 8 + i : i * 8 + j], x[j]));
        finite(value, flag);
        dst[base + i] = value;
      }
    }
    if (g.h128 && inverse)
      hadamard_row(dst, width, g.hscale, flag);
  }
}
__global__ void attention_kernel(Geometry g, spin8_attention_v2_request a,
                                 double *work, Layout l, uint32_t *flag) {
  const double *q = work, *k = work + l.k, *v = work + l.v;
  for (size_t row = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
       row < a.n_query * g.nh; row += size_t(gridDim.x) * blockDim.x) {
    const size_t t = row / g.nh, h = row % g.nh, head = g.map[h];
    double *scores = work + l.scores + row * a.n_key;
    double maximum = -DBL_MAX;
    bool any = false;
    for (size_t j = 0; j < a.n_key; ++j) {
      scores[j] = 0;
      if (!allowed(a.mask, t, h, j))
        continue;
      any = true;
      double dot = 0;
      for (size_t c = 0; c < g.dq; ++c)
        dot = __dadd_rn(dot, __dmul_rn(q[row * g.dq + c],
                                       k[(j * g.kvh + head) * g.dq + c]));
      finite(dot, flag);
      double s = dot * a.scale;
      finite(s, flag);
      const double b = a.bias.data ? load(a.bias, t, h, j) : 0;
      if (a.logit_order == SPIN8_ATTENTION_V2_SCALE_BIAS_SOFTCAP) {
        s += b;
        finite(s, flag);
      }
      if (a.softcap > 0)
        s = a.softcap * tanh(s / a.softcap);
      if (a.logit_order == SPIN8_ATTENTION_V2_SCALE_SOFTCAP_BIAS)
        s += b;
      finite(s, flag);
      scores[j] = s;
      maximum = fmax(maximum, s);
    }
    if (!any)
      atomicOr(flag, SPIN8_ATTENTION_V2_CUDA_ALL_MASKED);
    double sum = 0;
    for (size_t j = 0; j < a.n_key; ++j) {
      scores[j] = allowed(a.mask, t, h, j) ? exp(scores[j] - maximum) : 0;
      sum += scores[j];
    }
    finite(sum, flag);
    if (any && !(sum > 0))
      atomicOr(flag, SPIN8_ATTENTION_V2_CUDA_NONFINITE);
    for (size_t c = 0; c < g.dv; ++c) {
      double value = 0;
      if (any)
        for (size_t j = 0; j < a.n_key; ++j)
          value = __dadd_rn(value, __dmul_rn(scores[j] / sum,
                                             v[(j * g.kvh + head) * g.dv + c]));
      finite(value, flag);
      work[l.result + row * g.dv + c] = value;
    }
  }
}
__global__ void gate_kernel(Geometry g, spin8_attention_v2_request a,
                            double *result, uint32_t *flag) {
  for (size_t i = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
       i < a.n_query * g.nh * g.dv; i += size_t(gridDim.x) * blockDim.x) {
    double x = result[i];
    if (a.gate.data)
      x *= load(a.gate, i / (g.nh * g.dv), (i / g.dv) % g.nh, i % g.dv);
    finite(x, flag);
    if (fabs(x) > double(FLT_MAX))
      atomicOr(flag, SPIN8_ATTENTION_V2_CUDA_NONFINITE);
    result[i] = x;
  }
}
__global__ void commit_kernel(Geometry g, spin8_attention_v2_request a,
                              const double *result, const uint32_t *flag) {
  // Separate kernel boundary is the global validation barrier. No flag writer
  // runs concurrently with this read; no row can publish ahead of a late error.
  if (*flag)
    return;
  for (size_t i = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
       i < a.n_query * g.nh * g.dv; i += size_t(gridDim.x) * blockDim.x) {
    const size_t at = (i / (g.nh * g.dv)) * a.output.token_stride_bytes +
                      ((i / g.dv) % g.nh) * a.output.head_stride_bytes +
                      (i % g.dv) * a.output.channel_stride_bytes;
    *reinterpret_cast<float *>(static_cast<char *>(a.output.data) + at) =
        float(result[i]);
  }
}
} // namespace

extern "C" spin8_attention_v2_cuda_result
spin8_attention_v2_cuda_plan_create(const spin8_attention_v2_config *c,
                                    size_t bytes, int device,
                                    spin8_attention_v2_cuda_plan **out) {
  return boundary([&] {
    object(c, bytes);
    object(out);
    header(*c, SPIN8_ATTENTION_V2_ABI_VERSION);
    require(c->dtype == SPIN8_F32 && c->basis <= SPIN8_ATTENTION_V2_FIXED,
            SPIN8_UNSUPPORTED);
    const void *out_storage = static_cast<const void *>(out);
    require(!overlap(out_storage, sizeof(*out), c, bytes));
    // Reuse actual B1 geometry, mapping and semantic validation without
    // modifying its FP64 public contract or numerical implementation.
    auto cpu_config = *c;
    cpu_config.dtype = SPIN8_F64;
    spin8_attention_v2_plan *cpu = nullptr;
    success(
        spin8_attention_v2_plan_create(&cpu_config, sizeof(cpu_config), &cpu));
    std::unique_ptr<spin8_attention_v2_plan,
                    decltype(&spin8_attention_v2_plan_free)>
        checked(cpu, spin8_attention_v2_plan_free);
    require(!overlap(out_storage, sizeof(*out), c->q_to_kv, c->q_to_kv_bytes));
    if (c->planes)
      require(!overlap(out_storage, sizeof(*out), c->planes, c->planes_bytes));
    current_device(device);
    cudaDeviceProp properties{};
    runtime(cudaGetDeviceProperties(&properties, device));
    require(properties.major == 12 && properties.minor == 0 &&
                properties.unifiedAddressing,
            SPIN8_UNSUPPORTED);
    auto p = std::make_unique<spin8_attention_v2_cuda_plan>();
    p->device = device;
    p->g = {c->source_id,
            c->n_q_heads,
            c->n_kv_heads,
            c->d_qk,
            c->d_v,
            nullptr,
            nullptr,
            c->basis == SPIN8_ATTENTION_V2_FIXED,
            c->hadamard_block == 128,
            1.0 / std::sqrt(128.0)};
    p->storage_bytes =
        add(64 * sizeof(double), mul(c->n_q_heads, sizeof(uint32_t)));
    require(p->storage_bytes <= size_t(PTRDIFF_MAX));
    std::vector<unsigned char> coefficients(p->storage_bytes, 0);
    if (p->g.fixed) {
      std::vector<spin8_plane> planes(c->plane_count);
      if (c->plane_count)
        std::memcpy(planes.data(), c->planes,
                    c->plane_count * sizeof(spin8_plane));
      spin8_plan *rotation = nullptr;
      success(spin8_plan_create(planes.data(), planes.size(), &rotation));
      std::unique_ptr<spin8_plan, decltype(&spin8_plan_free)> owned(
          rotation, spin8_plan_free);
      double matrix[64];
      // Export the actual audited v1 coefficients, without rounding to F32 or
      // substituting a second octonion/rotation convention.
      success(spin8_plan_matrix(rotation, static_cast<spin8_view>(c->view),
                                matrix));
      std::memcpy(coefficients.data(), matrix, sizeof(matrix));
    }
    std::memcpy(coefficients.data() + 64 * sizeof(double), c->q_to_kv,
                c->n_q_heads * sizeof(uint32_t));
    runtime(cudaMalloc(&p->storage, p->storage_bytes));
    cudaStream_t upload = nullptr;
    auto e = cudaStreamCreateWithFlags(&upload, cudaStreamNonBlocking);
    if (e == cudaSuccess) {
      const auto copy =
          cudaMemcpyAsync(p->storage, coefficients.data(), p->storage_bytes,
                          cudaMemcpyHostToDevice, upload);
      // A pageable H2D copy can return before DMA completes. Finish this
      // private stream before publishing to arbitrary caller streams or
      // releasing the host vector. Also drain/check it when submission fails.
      const auto completed = cudaStreamSynchronize(upload);
      const auto destroyed = cudaStreamDestroy(upload);
      e = copy != cudaSuccess        ? copy
          : completed != cudaSuccess ? completed
                                     : destroyed;
    }
    if (e != cudaSuccess) {
      const auto cleanup = cudaFree(p->storage);
      runtime(cleanup != cudaSuccess ? cleanup : e);
    }
    p->g.matrix = static_cast<const double *>(p->storage);
    p->g.map = reinterpret_cast<const uint32_t *>(
        static_cast<const char *>(p->storage) + 64 * sizeof(double));
    *out = p.release();
  });
}
extern "C" spin8_attention_v2_cuda_result
spin8_attention_v2_cuda_plan_free(spin8_attention_v2_cuda_plan *p) {
  return boundary([&] {
    if (!p)
      return;
    object(p);
    current_device(p->device);
    runtime(cudaFree(p->storage));
    delete p;
  });
}
extern "C" spin8_attention_v2_cuda_result
spin8_attention_v2_cuda_workspace_bytes(const spin8_attention_v2_cuda_plan *p,
                                        size_t nq, size_t nk, size_t *bytes) {
  return boundary([&] {
    object(p);
    object(bytes);
    require(!overlap(bytes, sizeof(*bytes), p, sizeof(*p)));
    const auto l = layout(p->g, nq, nk);
    *bytes = l.elements * sizeof(double);
  });
}
extern "C" spin8_attention_v2_cuda_result
spin8_attention_v2_cuda_enqueue(const spin8_attention_v2_cuda_plan *p,
                                const spin8_attention_v2_cuda_request *r,
                                size_t bytes) {
  return boundary([&] {
    const auto l = validate(p, r, bytes);
    const auto a = r->attention;
    const auto g = p->g;
    auto *work = static_cast<double *>(r->workspace);
    auto *flag = r->device_status;
    const auto stream = r->stream;
    runtime(cudaMemsetAsync(flag, 0, sizeof(*flag), stream));
    pack_kernel<<<blocks(l.k), threads, 0, stream>>>(a.q, l.k, g.nh, g.dq, work,
                                                     flag);
    runtime(cudaGetLastError());
    pack_kernel<<<blocks(l.v - l.k), threads, 0, stream>>>(
        a.k, l.v - l.k, g.kvh, g.dq, work + l.k, flag, r->k_dtype);
    runtime(cudaGetLastError());
    pack_kernel<<<blocks(l.result - l.v), threads, 0, stream>>>(
        a.v, l.result - l.v, g.kvh, g.dv, work + l.v, flag, r->v_dtype);
    runtime(cudaGetLastError());
    if (a.bias.data) {
      const auto n = a.n_query * g.nh * a.n_key;
      pack_kernel<<<blocks(n), threads, 0, stream>>>(a.bias, n, g.nh, a.n_key,
                                                     nullptr, flag);
      runtime(cudaGetLastError());
    }
    if (a.gate.data) {
      const auto n = a.n_query * g.nh * g.dv;
      pack_kernel<<<blocks(n), threads, 0, stream>>>(a.gate, n, g.nh, g.dv,
                                                     nullptr, flag);
      runtime(cudaGetLastError());
    }
    if (g.fixed) {
      transform_kernel<<<blocks(a.n_query * g.nh), threads, 0, stream>>>(
          g, work, a.n_query * g.nh, g.dq, false, flag);
      runtime(cudaGetLastError());
      transform_kernel<<<blocks(a.n_key * g.kvh), threads, 0, stream>>>(
          g, work + l.k, a.n_key * g.kvh, g.dq, false, flag);
      runtime(cudaGetLastError());
      transform_kernel<<<blocks(a.n_key * g.kvh), threads, 0, stream>>>(
          g, work + l.v, a.n_key * g.kvh, g.dv, false, flag);
      runtime(cudaGetLastError());
    }
    attention_kernel<<<blocks(a.n_query * g.nh), threads, 0, stream>>>(
        g, a, work, l, flag);
    runtime(cudaGetLastError());
    if (g.fixed) {
      transform_kernel<<<blocks(a.n_query * g.nh), threads, 0, stream>>>(
          g, work + l.result, a.n_query * g.nh, g.dv, true, flag);
      runtime(cudaGetLastError());
    }
    gate_kernel<<<blocks(l.scores - l.result), threads, 0, stream>>>(
        g, a, work + l.result, flag);
    runtime(cudaGetLastError());
    commit_kernel<<<blocks(l.scores - l.result), threads, 0, stream>>>(
        g, a, work + l.result, flag);
    runtime(cudaGetLastError());
  });
}
