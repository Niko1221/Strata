#include "cache_cuda_internal.h"
#include "dialect_cuda_device.cuh"
#include <algorithm>
#include <cmath>
namespace tqc_cache_detail {
int context(const tqc_cache_cuda_context *c) {
  if (!range(c, sizeof(*c), alignof(tqc_cache_cuda_context)))
    return TQC_D_INVALID;
  if (c->version != 1 || c->size != sizeof(*c))
    return TQC_D_VERSION;
  if (!c->plan || !range(c->arena, c->arena_bytes, 256))
    return TQC_D_INVALID;
  if (c->arena_bytes < c->plan->layout.total_bytes)
    return TQC_D_CAPACITY;
  if (overlap(c, sizeof(*c), c->arena, c->arena_bytes) ||
      overlap(c->plan, sizeof(*c->plan), c->arena, c->arena_bytes))
    return TQC_D_OVERLAP;
  if (!c->stream || c->stream == reinterpret_cast<void *>(1) ||
      c->stream == reinterpret_cast<void *>(2))
    return TQC_D_UNSUPPORTED;
  int d = -1, sd = -1;
  auto stream = static_cast<cudaStream_t>(c->stream);
  if (cudaGetDevice(&d) != cudaSuccess ||
      cudaStreamGetDevice(stream, &sd) != cudaSuccess)
    return TQC_DC_RUNTIME;
  if (d != c->device_ordinal || sd != d)
    return TQC_D_UNSUPPORTED;
  cudaDeviceProp prop{};
  if (cudaGetDeviceProperties(&prop, d) != cudaSuccess)
    return TQC_DC_RUNTIME;
  if (prop.major != 12 || prop.minor != 0)
    return TQC_D_UNSUPPORTED;
  return cudaPeekAtLastError() == cudaSuccess ? 0 : TQC_DC_RUNTIME;
}
} // namespace tqc_cache_detail
namespace {
using namespace tqc_cache_detail;
using namespace tqc_cuda_detail;
tqc_cache_cuda_device device(const tqc_cache_cuda_context *c) {
  return {c->plan->config, c->plan->layout,
          static_cast<unsigned char *>(c->arena)};
}
__device__ tqc_cache_cuda_state *state(tqc_cache_cuda_device p) {
  return reinterpret_cast<tqc_cache_cuda_state *>(p.arena + p.layout.state);
}
__device__ tqc_cache_cuda_command *command(tqc_cache_cuda_device p) {
  return reinterpret_cast<tqc_cache_cuda_command *>(p.arena + p.layout.command);
}
__device__ void fail(tqc_cache_cuda_device p, unsigned code) {
  atomicOr(&state(p)->status, code);
  atomicExch(&state(p)->valid, 0u);
}
__device__ bool ready(tqc_cache_cuda_device p) {
  auto *s = state(p);
  for (int i = 0; i < 32; ++i)
    if (p.arena[p.layout.identity_offset + i] != p.layout.identity[i]) {
      fail(p, TQC_CC_INVALID_STATE);
      return false;
    }
  if (atomicOr(&s->status, 0u) || !atomicOr(&s->valid, 0u))
    return false;
  if (s->committed > s->tentative || s->tentative > p.config.capacity) {
    fail(p, TQC_CC_INVALID_STATE);
    return false;
  }
  return true;
}
struct Baseline {
  float b[192], s1[128], s2[128];
};
Baseline baseline(const tqc_cache_cuda_plan *p) {
  Baseline b{};
  std::copy_n(p->b, 192, b.b);
  std::copy_n(p->s1, 128, b.s1);
  std::copy_n(p->s2, 128, b.s2);
  return b;
}
__device__ void matrix_audit(const float *m, int n, double &ef, double &gram,
                             double &logdet, int32_t &sign, uint32_t &finite) {
  // B is only 8x8. The F determinant is evaluated from its actual factors
  // below.
  double work[64];
  ef = 0;
  gram = 0;
  logdet = 0;
  sign = 1;
  finite = 1;
  for (int i = 0; i < n * n; ++i) {
    work[i] = m[i];
    if (!isfinite(m[i]))
      finite = 0;
  }
  for (int i = 0; i < n; ++i)
    for (int j = 0; j < n; ++j) {
      double x = 0;
      for (int k = 0; k < n; ++k)
        x += double(m[k * n + i]) * m[k * n + j];
      x -= i == j ? 1 : 0;
      ef += x * x;
      gram = fmax(gram, fabs(x));
    }
  ef = sqrt(ef) / sqrt(double(n));
  for (int j = 0; j < n; ++j) {
    int pivot = j;
    for (int i = j + 1; i < n; ++i)
      if (fabs(work[i * n + j]) > fabs(work[pivot * n + j]))
        pivot = i;
    if (pivot != j) {
      for (int k = 0; k < n; ++k) {
        double x = work[j * n + k];
        work[j * n + k] = work[pivot * n + k];
        work[pivot * n + k] = x;
      }
      sign = -sign;
    }
    const double d = work[j * n + j];
    if (d == 0 || !isfinite(d)) {
      sign = 0;
      logdet = -INFINITY;
      return;
    }
    if (d < 0)
      sign = -sign;
    logdet += log(fabs(d));
    for (int i = j + 1; i < n; ++i) {
      const double ratio = work[i * n + j] / d;
      for (int k = j + 1; k < n; ++k)
        work[i * n + k] -= ratio * work[j * n + k];
    }
  }
}
__global__ void audit_kernel(tqc_cache_cuda_device p, Baseline base) {
  if (threadIdx.x || blockIdx.x)
    return;
  auto *a = reinterpret_cast<tqc_cache_cuda_audit *>(p.arena + p.layout.audit);
  const float *b = reinterpret_cast<const float *>(p.arena + p.layout.matrices);
  bool bad = false;
  for (int r = 0; r < 3; ++r) {
    a->drift[r] = 0;
    matrix_audit(b + 64 * r, 8, a->ef[r], a->gram[r], a->logabsdet[r],
                 a->det_sign[r], a->finite[r]);
    for (int i = 0; i < 64; ++i)
      a->drift[r] =
          fmax(a->drift[r], fabs(double(b[64 * r + i]) - base.b[64 * r + i]));
    bad |= !a->finite[r] || a->ef[r] > 1e-5 || a->gram[r] > 1e-5 ||
           a->det_sign[r] != 1 || fabs(a->logabsdet[r]) > 8e-5 ||
           a->drift[r] > 1e-5;
  }
  // Audit the actual fwht primitive/constant storage, including normalization.
  // Store only this fixed 128x128 operator, never K/V history or QxK scores.
  extern __shared__ float f[];
  float x[128];
  a->finite[3] = 1;
  a->drift[3] = 0;
  a->ef[3] = 0;
  a->gram[3] = 0;
  for (int j = 0; j < 128; ++j) {
    for (int i = 0; i < 128; ++i)
      x[i] = i == j ? 1.0f : 0.0f;
    fwht(x, false);
    for (int i = 0; i < 128; ++i) {
      f[i * 128 + j] = x[i];
      if (!isfinite(x[i]))
        a->finite[3] = 0;
      unsigned bits = unsigned(i & j), parity = 0;
      while (bits) {
        parity ^= bits & 1u;
        bits >>= 1;
      }
      double expected =
          double(base.s2[i]) * base.s1[j] * (parity ? -1.0 : 1.0) / sqrt(128.0);
      a->drift[3] = fmax(a->drift[3], fabs(double(x[i]) - expected));
    }
  }
  for (int i = 0; i < 128; ++i)
    for (int j = 0; j < 128; ++j) {
      double sum = 0;
      for (int k = 0; k < 128; ++k)
        sum += double(f[k * 128 + i]) * f[k * 128 + j];
      sum -= i == j ? 1.0 : 0.0;
      a->ef[3] += sum * sum;
      a->gram[3] = fmax(a->gram[3], fabs(sum));
    }
  a->ef[3] = sqrt(a->ef[3] / 128.0);
  a->det_sign[3] = 1;
  a->logabsdet[3] = 0;
  // det(H128/sqrt128)=+1. Both actual diagonal factors participate; their
  // magnitudes and actual primitive normalization determine logabsdet.
  for (int i = 0; i < 128; ++i) {
    double z = double(turbo_cpu_s1[i]) * turbo_cpu_s2[i];
    if (!isfinite(z) || z == 0) {
      a->finite[3] = 0;
      a->det_sign[3] = 0;
    } else {
      if (z < 0)
        a->det_sign[3] = -a->det_sign[3];
      a->logabsdet[3] += log(fabs(z));
    }
  }
  const double norm =
      fabs(double(f[0])) / fabs(double(turbo_cpu_s1[0]) * turbo_cpu_s2[0]);
  a->logabsdet[3] += 128 * log(norm * sqrt(128.0));
  int expected_sign = 1;
  for (int i = 0; i < 128; ++i)
    if (base.s1[i] * base.s2[i] < 0)
      expected_sign = -expected_sign;
  bad |= !a->finite[3] || a->ef[3] > 1e-5 || a->gram[3] > 1e-5 ||
         a->det_sign[3] != expected_sign || !isfinite(a->logabsdet[3]) ||
         fabs(a->logabsdet[3]) > 128e-5 || a->drift[3] > 1e-5;
  if (bad)
    fail(p, TQC_CC_TRANSFORM_INVARIANT_VIOLATION);
}
__global__ void init_state(tqc_cache_cuda_device p) {
  if (!threadIdx.x && !blockIdx.x)
    state(p)->valid = 1;
}
int launch_status() {
  return cudaPeekAtLastError() == cudaSuccess ? 0 : TQC_DC_RUNTIME;
}
constexpr size_t kAuditDynamicSharedBytes = 128 * 128 * sizeof(float);
int configure_audit_kernel(int device) {
  int optin = 0;
  if (cudaDeviceGetAttribute(&optin, cudaDevAttrMaxSharedMemoryPerBlockOptin,
                             device) != cudaSuccess)
    return TQC_DC_RUNTIME;
  if (optin < 0 || size_t(optin) < kAuditDynamicSharedBytes)
    return TQC_D_UNSUPPORTED;
  if (cudaFuncSetAttribute(
          audit_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize,
          static_cast<int>(kAuditDynamicSharedBytes)) != cudaSuccess)
    return TQC_DC_RUNTIME;
  return 0;
}
} // namespace
extern "C" int tqc_cache_cuda_audit_device(const tqc_cache_cuda_context *c) {
  int rc = tqc_cache_detail::context(c);
  if (rc)
    return rc;
  auto stream = static_cast<cudaStream_t>(c->stream);
  cudaStreamCaptureStatus capture = cudaStreamCaptureStatusNone;
  if (cudaStreamIsCapturing(stream, &capture) != cudaSuccess)
    return TQC_DC_RUNTIME;
  if (capture == cudaStreamCaptureStatusNone) {
    rc = configure_audit_kernel(c->device_ordinal);
    if (rc)
      return rc;
  }
  audit_kernel<<<1, 1, kAuditDynamicSharedBytes, stream>>>(
      device(c), baseline(c->plan));
  return launch_status();
}
extern "C" int tqc_cache_cuda_initialize(const tqc_cache_cuda_context *c) {
  int rc = tqc_cache_detail::context(c);
  if (rc)
    return rc;
  auto stream = static_cast<cudaStream_t>(c->stream);
  cudaStreamCaptureStatus capture;
  if (cudaStreamIsCapturing(stream, &capture) != cudaSuccess)
    return TQC_DC_RUNTIME;
  if (capture != cudaStreamCaptureStatusNone)
    return TQC_D_UNSUPPORTED;
  if (cudaMemsetAsync(c->arena, 0, c->plan->layout.total_bytes, stream) !=
          cudaSuccess ||
      cudaMemcpyAsync(static_cast<char *>(c->arena) + c->plan->layout.matrices,
                      c->plan->b, sizeof(c->plan->b), cudaMemcpyHostToDevice,
                      stream) != cudaSuccess)
    return TQC_DC_RUNTIME;
  if (cudaMemcpyAsync(static_cast<char *>(c->arena) +
                          c->plan->layout.identity_offset,
                      c->plan->layout.identity, 32, cudaMemcpyHostToDevice,
                      stream) != cudaSuccess)
    return TQC_DC_RUNTIME;
  init_state<<<1, 1, 0, stream>>>(device(c));
  rc = launch_status();
  if (rc)
    return rc;
  rc = tqc_cache_cuda_audit_device(c);
  if (rc)
    return rc;
  return cudaStreamSynchronize(stream) == cudaSuccess ? 0 : TQC_DC_RUNTIME;
}
namespace {
int span_check(tqc_dialect_v1_span s, size_t rows, size_t width, size_t align,
               bool optional) {
  if (optional && !s.data && !s.capacity_bytes && !s.row_stride_bytes)
    return 0;
  size_t last = 0;
  if (!mul(rows - 1, s.row_stride_bytes, last) || !add(last, width, last))
    return TQC_D_OVERFLOW;
  if (s.row_stride_bytes < width || s.capacity_bytes < last)
    return TQC_D_CAPACITY;
  if (!s.data ||
      s.capacity_bytes > UINTPTR_MAX - reinterpret_cast<uintptr_t>(s.data))
    return TQC_D_OVERFLOW;
  if (reinterpret_cast<uintptr_t>(s.data) % align || s.row_stride_bytes % align)
    return TQC_D_ALIGNMENT;
  return 0;
}
int aliases(const tqc_cache_cuda_context *c, const void *request, size_t size,
            const tqc_dialect_v1_span *spans, size_t count) {
  if (overlap(request, size, c->arena, c->arena_bytes) ||
      overlap(request, size, c->plan, sizeof(*c->plan)))
    return TQC_D_OVERLAP;
  for (size_t i = 0; i < count; ++i) {
    const auto s = spans[i];
    if (overlap(s.data, s.capacity_bytes, c->arena, c->arena_bytes) ||
        overlap(s.data, s.capacity_bytes, request, size) ||
        overlap(s.data, s.capacity_bytes, c, sizeof(*c)) ||
        overlap(s.data, s.capacity_bytes, c->plan, sizeof(*c->plan)))
      return TQC_D_OVERLAP;
    for (size_t j = 0; j < i; ++j)
      if (overlap(s.data, s.capacity_bytes, spans[j].data,
                  spans[j].capacity_bytes))
        return TQC_D_OVERLAP;
  }
  return 0;
}
__device__ const float *float_row(tqc_dialect_v1_span s, size_t row) {
  return reinterpret_cast<const float *>(
      static_cast<const unsigned char *>(s.data) + row * s.row_stride_bytes);
}
__device__ int64_t position(tqc_dialect_v1_span s, size_t row) {
  return *reinterpret_cast<const int64_t *>(
      static_cast<const unsigned char *>(s.data) + row * s.row_stride_bytes);
}
__device__ void rotate_block(const float *b, const float *x, float *y) {
  for (int g = 0; g < 128; g += 8)
    for (int i = 0; i < 8; ++i) {
      float sum = 0;
      for (int j = 0; j < 8; ++j)
        sum = __fadd_rn(sum, __fmul_rn(b[i * 8 + j], x[g + j]));
      y[g + i] = sum;
    }
}
__device__ bool finite_input(const float *x) {
  for (int i = 0; i < 128; ++i)
    if (!isfinite(x[i]) || fabsf(x[i]) > 65504)
      return false;
  return true;
}
__global__ void append_kernel(tqc_cache_cuda_device p, tqc_cache_cuda_append r,
                              size_t call_capacity) {
  if (!ready(p))
    return;
  const size_t count = command(p)->append_rows, first = state(p)->tentative;
  if (count > call_capacity || count > p.config.capacity - first) {
    fail(p, TQC_CC_CAPACITY);
    return;
  }
  const size_t blocks = p.config.columns / 128,
               total = count * p.config.kv_heads * blocks;
  const float *b = reinterpret_cast<const float *>(p.arena + p.layout.matrices);
  for (size_t i = size_t(blockIdx.x) * blockDim.x + threadIdx.x; i < total;
       i += size_t(gridDim.x) * blockDim.x) {
    if (!ready(p))
      return;
    size_t row = i / blocks, block = i % blocks,
           drow = first * p.config.kv_heads + row;
    const float *k = float_row(r.k, row) + block * 128,
                *v = float_row(r.v, row) + block * 128;
    if (!finite_input(k) || !finite_input(v)) {
      fail(p, TQC_CC_NUMERICAL);
      return;
    }
    float x[128], y[128];
    unsigned char packed[98];
    float norm = 0;
    if (p.config.representation) {
      for (int j = 0; j < 128; ++j)
        x[j] = k[j];
      fwht(x, false);
    } else
      norm = prepare(p.config.k_dialect, k, x);
    for (int view = 0; view < 3; ++view)
      if (p.config.view_mask & (1u << view)) {
        rotate_block(b + 64 * view, x, y);
        unsigned char *dst =
            p.arena + p.layout.k[view] + drow * p.layout.k_row_bytes +
            block *
                (p.config.representation ? 512 : 18 + 16 * p.config.k_dialect);
        if (p.config.representation) {
          for (int j = 0; j < 128; ++j) {
            if (!isfinite(y[j])) {
              fail(p, TQC_CC_NUMERICAL);
              return;
            }
            reinterpret_cast<float *>(dst)[j] = y[j];
          }
        } else {
          turbo_pack(p.config.k_dialect, y, norm, packed);
          if (!isfinite(half_decode(load16(packed)))) {
            fail(p, TQC_CC_NUMERICAL);
            return;
          }
          for (unsigned j = 0; j < 18 + 16 * p.config.k_dialect; ++j)
            dst[j] = packed[j];
        }
      }
    unsigned char *dst =
        p.arena + p.layout.v + drow * p.layout.v_row_bytes +
        block * (p.config.representation ? 512 : 18 + 16 * p.config.v_dialect);
    if (p.config.representation) {
      for (int j = 0; j < 128; ++j)
        x[j] = v[j];
      fwht(x, false);
      for (int j = 0; j < 128; ++j)
        reinterpret_cast<float *>(dst)[j] = x[j];
    } else {
      norm = prepare(p.config.v_dialect, v, x);
      turbo_pack(p.config.v_dialect, x, norm, packed);
      if (!isfinite(half_decode(load16(packed)))) {
        fail(p, TQC_CC_NUMERICAL);
        return;
      }
      for (unsigned j = 0; j < 18 + 16 * p.config.v_dialect; ++j)
        dst[j] = packed[j];
    }
    if (row % p.config.kv_heads == 0 && block == 0)
      reinterpret_cast<int64_t *>(
          p.arena + p.layout.positions)[first + row / p.config.kv_heads] =
          position(r.positions, row / p.config.kv_heads);
  }
}
__global__ void finish_append(tqc_cache_cuda_device p, size_t call_capacity) {
  if (!threadIdx.x && !blockIdx.x && ready(p)) {
    size_t n = command(p)->append_rows;
    if (n > call_capacity || n > p.config.capacity - state(p)->tentative) {
      fail(p, TQC_CC_CAPACITY);
      return;
    }
    state(p)->tentative += n;
  }
}
__global__ void control_kernel(tqc_cache_cuda_device p, unsigned op) {
  if (threadIdx.x || blockIdx.x)
    return;
  if (op == TQC_CC_INVALIDATE) {
    fail(p, TQC_CC_INVALID_STATE);
    return;
  }
  if (!ready(p))
    return;
  auto *s = state(p);

  if (op == TQC_CC_COMMIT)
    s->committed = s->tentative;
  else if (op == TQC_CC_CANCEL)
    s->tentative = s->committed;
  else {
    auto target = command(p)->target_rows;
    if (target > s->committed) {
      fail(p, TQC_CC_INDEX);
      return;
    }
    s->tentative = target;
    s->committed = target;
  }
}
__device__ void decoded(tqc_cache_cuda_device p, bool value, int view,
                        size_t row, unsigned block, float *out) {
  unsigned dialect = value ? p.config.v_dialect : p.config.k_dialect;
  const unsigned char *src =
      p.arena + (value ? p.layout.v : p.layout.k[view]) +
      row * (value ? p.layout.v_row_bytes : p.layout.k_row_bytes) +
      block * (p.config.representation ? 512 : 18 + 16 * dialect);
  if (p.config.representation)
    for (int i = 0; i < 128; ++i)
      out[i] = reinterpret_cast<const float *>(src)[i];
  else
    turbo_decode(dialect, src, out);
}
__global__ void attend_kernel(tqc_cache_cuda_device p,
                              tqc_cache_cuda_attention r,
                              size_t call_capacity) {
  const size_t row = size_t(blockIdx.x) * blockDim.x + threadIdx.x,
               nq = command(p)->query_rows;
  if (!ready(p))
    return;
  if (nq > call_capacity) {
    fail(p, TQC_CC_CAPACITY);
    return;
  }
  if (row >= nq * p.config.q_heads)
    return;
  size_t count = state(p)->tentative;
  if (!count) {
    fail(p, TQC_CC_EMPTY);
    return;
  }
  const size_t qi = row / p.config.q_heads, head = row % p.config.q_heads,
               kv = head / (p.config.q_heads / p.config.kv_heads);
  float query[3][256], x[128], y[128];
  const float *b = reinterpret_cast<const float *>(p.arena + p.layout.matrices);
  for (unsigned block = 0; block < p.config.columns / 128; ++block) {
    const float *in = float_row(r.q, row) + block * 128;
    if (!finite_input(in)) {
      fail(p, TQC_CC_NUMERICAL);
      return;
    }
    for (int j = 0; j < 128; ++j)
      x[j] = in[j];
    fwht(x, false);
    for (int view = 0; view < 3; ++view)
      if (p.config.view_mask & (1u << view)) {
        rotate_block(b + 64 * view, x, y);
        for (int j = 0; j < 128; ++j)
          query[view][block * 128 + j] = y[j];
      }
  }
  double accum[256];
  for (unsigned j = 0; j < p.config.columns; ++j)
    accum[j] = 0;
  double maximum = -INFINITY, den = 0;
  const size_t selected = r.indices.data ? r.selection_width : count;
  for (size_t selection = 0; selection < selected; ++selection) {
    if (!ready(p))
      return;
    int64_t t = r.indices.data
                    ? reinterpret_cast<const int64_t *>(
                          static_cast<const unsigned char *>(r.indices.data) +
                          qi * r.indices.row_stride_bytes)[selection]
                    : int64_t(selection);
    if (t == -1)
      continue;
    if (t < 0 || uint64_t(t) >= count) {
      fail(p, TQC_CC_INDEX);
      return;
    }
    if (r.indices.data)
      for (size_t j = 0; j < selection; ++j)
        if (reinterpret_cast<const int64_t *>(
                static_cast<const unsigned char *>(r.indices.data) +
                qi * r.indices.row_stride_bytes)[j] == t) {
          fail(p, TQC_CC_INDEX);
          return;
        }
    if (r.causal &&
        reinterpret_cast<const int64_t *>(p.arena + p.layout.positions)[t] >
            position(r.positions, qi))
      continue;
    if (r.mask.data && !*(static_cast<const unsigned char *>(r.mask.data) +
                          row * r.mask.row_stride_bytes + size_t(t)))
      continue;
    double score = 0;
    for (int view = 0; view < 3; ++view)
      if (p.config.view_mask & (1u << view)) {
        double dot = 0;
        for (unsigned block = 0; block < p.config.columns / 128; ++block) {
          decoded(p, false, view, size_t(t) * p.config.kv_heads + kv, block, x);
          for (int j = 0; j < 128; ++j)
            dot += double(query[view][block * 128 + j]) * x[j];
        }
        score += p.config.alpha[view] * dot;
      }
    score *= r.scale;
    if (r.bias.data)
      score += float_row(r.bias, row)[t];
    if (!isfinite(score)) {
      fail(p, TQC_CC_NUMERICAL);
      return;
    }
    double next = fmax(maximum, score), old = exp(maximum - next),
           weight = exp(score - next);
    den = den * old + weight;
    maximum = next;
    for (unsigned block = 0; block < p.config.columns / 128; ++block) {
      decoded(p, true, 0, size_t(t) * p.config.kv_heads + kv, block, x);
      for (int j = 0; j < 128; ++j) {
        if (!isfinite(x[j])) {
          fail(p, TQC_CC_NUMERICAL);
          return;
        }
        unsigned column = block * 128 + j;
        accum[column] = accum[column] * old + weight * x[j];
      }
    }
  }
  if (den == 0) {
    fail(p, TQC_CC_ALL_MASKED);
    return;
  }
  auto *out = reinterpret_cast<float *>(
      static_cast<unsigned char *>(r.out.data) + row * r.out.row_stride_bytes);
  for (unsigned block = 0; block < p.config.columns / 128; ++block) {
    for (int j = 0; j < 128; ++j)
      x[j] = float(accum[block * 128 + j] / den);
    fwht(x, true);
    for (int j = 0; j < 128; ++j) {
      if (!isfinite(x[j])) {
        fail(p, TQC_CC_NUMERICAL);
        return;
      }
      out[block * 128 + j] = x[j];
    }
  }
}
} // namespace
static int append_rows_impl(const tqc_cache_cuda_context *c,
                            const tqc_cache_cuda_append *r,
                            size_t call_capacity, bool legacy) {
  int rc = context(c);
  if (rc)
    return rc;
  if (!range(r, sizeof(*r), alignof(tqc_cache_cuda_append)))
    return TQC_D_INVALID;
  if (r->version != 1 || r->size != sizeof(*r))
    return TQC_D_VERSION;
  auto cfg = c->plan->config;
  if (legacy)
    call_capacity = size_t(cfg.max_append_rows);
  if (!call_capacity || call_capacity > cfg.max_append_rows)
    return TQC_D_INVALID;
  size_t rows = call_capacity * cfg.kv_heads;
  tqc_dialect_v1_span spans[] = {r->k, r->v, r->positions};
  for (int i = 0; i < 3; ++i) {
    rc = span_check(spans[i], i == 2 ? call_capacity : rows,
                    i == 2 ? 8 : cfg.columns * 4, i == 2 ? 8 : 4, false);
    if (rc)
      return rc;
  }
  rc = aliases(c, r, sizeof(*r), spans, 3);
  if (rc)
    return rc;
  auto stream = static_cast<cudaStream_t>(c->stream);
  auto p = device(c);
  append_kernel<<<unsigned(std::min<size_t>(
                      65535, (rows * (cfg.columns / 128) + 31) / 32)),
                  32, 0, stream>>>(p, *r, call_capacity);
  rc = launch_status();
  if (rc)
    return rc;
  finish_append<<<1, 1, 0, stream>>>(p, call_capacity);
  return launch_status();
}
extern "C" int tqc_cache_cuda_control(const tqc_cache_cuda_context *c,
                                      uint32_t op) {
  int rc = context(c);
  if (rc)
    return rc;
  if (op < 1 || op > 4)
    return TQC_D_UNSUPPORTED;
  control_kernel<<<1, 1, 0, static_cast<cudaStream_t>(c->stream)>>>(device(c),
                                                                    op);
  return launch_status();
}
static int attend_impl(const tqc_cache_cuda_context *c,
                       const tqc_cache_cuda_attention *r, size_t call_capacity,
                       bool legacy) {
  int rc = context(c);
  if (rc)
    return rc;
  if (!range(r, sizeof(*r), alignof(tqc_cache_cuda_attention)))
    return TQC_D_INVALID;
  if (r->version != 1 || r->size != sizeof(*r))
    return TQC_D_VERSION;
  if (r->causal > 1 || r->reserved || !std::isfinite(r->scale))
    return TQC_D_INVALID;
  const auto cfg = c->plan->config;
  if (legacy)
    call_capacity = size_t(cfg.max_queries);
  if (!call_capacity || call_capacity > cfg.max_queries)
    return TQC_D_INVALID;
  const size_t nq = call_capacity, rows = nq * cfg.q_heads;
  if ((r->indices.data &&
       (!r->selection_width || r->selection_width > cfg.capacity)) ||
      (!r->indices.data && r->selection_width))
    return TQC_D_INVALID;
  tqc_dialect_v1_span spans[] = {r->q,    r->out,  r->positions,
                                 r->mask, r->bias, r->indices};
  size_t widths[] = {
      cfg.columns * 4,      cfg.columns * 4,          8,
      size_t(cfg.capacity), size_t(cfg.capacity) * 4, r->selection_width * 8};
  size_t aligns[] = {4, 4, 8, 1, 4, 8};
  for (int i = 0; i < 6; ++i) {
    rc = span_check(spans[i], i == 2 || i == 5 ? nq : rows, widths[i],
                    aligns[i], i >= 3);
    if (rc)
      return rc;
  }
  rc = aliases(c, r, sizeof(*r), spans, 6);
  if (rc)
    return rc;
  attend_kernel<<<unsigned((rows + 31) / 32), 32, 0,
                  static_cast<cudaStream_t>(c->stream)>>>(device(c), *r,
                                                          call_capacity);
  return launch_status();
}

extern "C" int
tqc_cache_cuda_position_transform(const tqc_cache_cuda_context *) {
  return TQC_D_UNSUPPORTED;
}
namespace {
__global__ void copy_kernel(tqc_cache_cuda_device dst,
                            tqc_cache_cuda_device src,
                            tqc_dialect_v1_span indices, bool clone) {
  if (threadIdx.x || blockIdx.x)
    return;
  if (!ready(dst))
    return;
  if (!ready(src)) {
    fail(dst, TQC_CC_INVALID_STATE);
    return;
  }
  auto *d = state(dst);
  const auto *s = state(src);
  const uint64_t n = clone ? s->committed : command(dst)->append_rows;
  if (clone && d->tentative) {
    fail(dst, TQC_CC_INVALID_STATE);
    return;
  }
  if (n > dst.config.capacity - d->tentative) {
    fail(dst, TQC_CC_CAPACITY);
    return;
  }
  // Prevalidate the entire selection. No partial physical copy on a bad map.
  for (uint64_t i = 0; i < n; ++i) {
    int64_t key = clone ? int64_t(i) : position(indices, size_t(i));
    if (key < 0 || uint64_t(key) >= s->committed) {
      fail(dst, TQC_CC_INDEX);
      return;
    }
    if (!clone)
      for (uint64_t j = 0; j < i; ++j)
        if (position(indices, size_t(j)) == key) {
          fail(dst, TQC_CC_INDEX);
          return;
        }
  }
  for (uint64_t i = 0; i < n; ++i) {
    size_t key = clone ? size_t(i) : size_t(position(indices, size_t(i))),
           row = size_t(d->tentative + i);
    for (int view = 0; view < 4; ++view) {
      if (view < 3 && !(dst.config.view_mask & (1u << view)))
        continue;
      size_t doffset = view < 3 ? dst.layout.k[view] : dst.layout.v,
             soffset = view < 3 ? src.layout.k[view] : src.layout.v;
      size_t stride =
          (view < 3 ? dst.layout.k_row_bytes : dst.layout.v_row_bytes) *
          dst.config.kv_heads;
      for (size_t j = 0; j < stride; ++j)
        dst.arena[doffset + row * stride + j] =
            src.arena[soffset + key * stride + j];
    }
    reinterpret_cast<int64_t *>(dst.arena + dst.layout.positions)[row] =
        reinterpret_cast<const int64_t *>(src.arena +
                                          src.layout.positions)[key];
  }
  d->tentative += n;
  if (clone)
    d->committed = d->tentative;
}
} // namespace
extern "C" int tqc_cache_cuda_copy(const tqc_cache_cuda_context *dst,
                                   const tqc_cache_cuda_context *src,
                                   tqc_dialect_v1_span indices, int clone) {
  int rc = context(dst);
  if (rc)
    return rc;
  rc = context(src);
  if (rc)
    return rc;
  if (clone != 0 && clone != 1)
    return TQC_D_INVALID;
  if (!identical(dst->plan, src->plan))
    return TQC_D_INVALID;
  if (overlap(dst->arena, dst->arena_bytes, src->arena, src->arena_bytes))
    return TQC_D_OVERLAP;
  if (dst->stream != src->stream || dst->device_ordinal != src->device_ordinal)
    return TQC_D_INVALID;
  if (clone) {
    if (indices.data || indices.capacity_bytes || indices.row_stride_bytes)
      return TQC_D_INVALID;
  } else {
    rc = span_check(indices, size_t(dst->plan->config.capacity), 8, 8, false);
    if (rc)
      return rc;
  }
  rc = aliases(dst, src, sizeof(*src), &indices, 1);
  if (rc)
    return rc;
  rc = aliases(src, dst, sizeof(*dst), &indices, 1);
  if (rc)
    return rc;
  copy_kernel<<<1, 1, 0, static_cast<cudaStream_t>(dst->stream)>>>(
      device(dst), device(src), indices, clone != 0);
  return launch_status();
}
extern "C" int tqc_cache_cuda_resources_query(const tqc_cache_cuda_context *c,
                                              tqc_cache_cuda_resources *out) {
  int rc = context(c);
  if (rc)
    return rc;
  if (!range(out, sizeof(*out), alignof(tqc_cache_cuda_resources)))
    return TQC_D_INVALID;
  if (out->version != 1 || out->size != sizeof(*out))
    return TQC_D_VERSION;
  if (overlap(out, sizeof(*out), c, sizeof(*c)) ||
      overlap(out, sizeof(*out), c->plan, sizeof(*c->plan)) ||
      overlap(out, sizeof(*out), c->arena, c->arena_bytes))
    return TQC_D_OVERLAP;
  rc = configure_audit_kernel(c->device_ordinal);
  if (rc)
    return rc;
  const void *functions[] = {reinterpret_cast<const void *>(audit_kernel),
                             reinterpret_cast<const void *>(append_kernel),
                             reinterpret_cast<const void *>(finish_append),
                             reinterpret_cast<const void *>(control_kernel),
                             reinterpret_cast<const void *>(attend_kernel),
                             reinterpret_cast<const void *>(copy_kernel)};
  tqc_cache_cuda_resources result{1, sizeof(result)};
  for (int i = 0; i < 6; ++i) {
    cudaFuncAttributes a{};
    if (cudaFuncGetAttributes(&a, functions[i]) != cudaSuccess)
      return TQC_DC_RUNTIME;
    result.local_bytes[i] = a.localSizeBytes;
    result.shared_bytes[i] =
        a.sharedSizeBytes + (i == 0 ? kAuditDynamicSharedBytes : 0);
    result.constant_bytes[i] = a.constSizeBytes;
    result.registers[i] = a.numRegs;
    result.max_threads[i] = a.maxThreadsPerBlock;
  }
  *out = result;
  return 0;
}

extern "C" int tqc_cache_cuda_append_rows(const tqc_cache_cuda_context *c,
                                          const tqc_cache_cuda_append *r) {
  return append_rows_impl(c, r, 0, true);
}
extern "C" int
tqc_cache_cuda_append_rows_bounded(const tqc_cache_cuda_context *c,
                                   const tqc_cache_cuda_append *r,
                                   size_t bound) {
  return append_rows_impl(c, r, bound, false);
}
extern "C" int tqc_cache_cuda_attend(const tqc_cache_cuda_context *c,
                                     const tqc_cache_cuda_attention *r) {
  return attend_impl(c, r, 0, true);
}
extern "C" int tqc_cache_cuda_attend_bounded(const tqc_cache_cuda_context *c,
                                             const tqc_cache_cuda_attention *r,
                                             size_t bound) {
  return attend_impl(c, r, bound, false);
}