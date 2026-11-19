#include "spin8/query_v3_cuda.h"

#include <cfloat>
#include <cmath>
#include <cstdint>
#include <cuda_runtime.h>
#include <new>

struct spin8_query_v3_cuda_plan {
  uint8_t coefficient[64]{};
  int8_t sign[64]{};
};

namespace {
bool product(size_t a, size_t b, size_t &out) {
  if (b && a > SIZE_MAX / b)
    return false;
  out = a * b;
  return true;
}
bool pointer(const void *p, size_t bytes, size_t alignment) {
  const auto address = reinterpret_cast<uintptr_t>(p);
  return p && address % alignment == 0 && address <= UINTPTR_MAX - bytes;
}
bool span(size_t rows, size_t width_bytes, size_t stride_bytes, size_t capacity,
          size_t &bytes) {
  size_t last = 0;
  if (!rows || !width_bytes || stride_bytes < width_bytes ||
      stride_bytes % sizeof(float) || !product(rows - 1, stride_bytes, last) ||
      last > SIZE_MAX - width_bytes)
    return false;
  bytes = last + width_bytes;
  return bytes <= capacity;
}
bool overlaps(const void *a, size_t an, const void *b, size_t bn) {
  // Every range is checked for address wrap before reaching this helper.
  const auto x = reinterpret_cast<uintptr_t>(a);
  const auto y = reinterpret_cast<uintptr_t>(b);
  return x < y + bn && y < x + an;
}
bool valid_layout(const spin8_query_v3_cuda_plan *plan,
                  const spin8_query_v3_cuda_parameters_f32 *p, const float *src,
                  size_t sc, size_t ss, float *dst, size_t dc, size_t ds,
                  size_t rows, size_t width, int *flag) {
  size_t row_bytes = 0, win_bytes = 0, sn = 0, dn = 0;
  if (!pointer(plan, sizeof(*plan), alignof(spin8_query_v3_cuda_plan)) ||
      !pointer(p, sizeof(*p), alignof(spin8_query_v3_cuda_parameters_f32)) ||
      !product(width, sizeof(float), row_bytes) ||
      !product(8, row_bytes, win_bytes) || !span(rows, row_bytes, ss, sc, sn) ||
      !span(rows, row_bytes, ds, dc, dn) || !pointer(src, sn, alignof(float)) ||
      !pointer(dst, dn, alignof(float)) ||
      !pointer(flag, sizeof(int), alignof(int)))
    return false;
  const void *parameters[] = {p->win, p->bin, p->wout, p->bout};
  const size_t sizes[] = {win_bytes, 8 * sizeof(float), 56 * sizeof(float),
                          7 * sizeof(float)};
  const size_t capacities[] = {p->win_bytes, p->bin_bytes, p->wout_bytes,
                               p->bout_bytes};
  for (size_t i = 0; i < 4; ++i) {
    if (sizes[i] > capacities[i] ||
        !pointer(parameters[i], sizes[i], alignof(float)))
      return false;
    if (overlaps(parameters[i], sizes[i], src, sn) ||
        overlaps(parameters[i], sizes[i], dst, dn) ||
        overlaps(parameters[i], sizes[i], flag, sizeof(int)))
      return false;
    for (size_t j = 0; j < i; ++j)
      if (overlaps(parameters[i], sizes[i], parameters[j], sizes[j]))
        return false;
  }
  return !overlaps(src, sn, flag, sizeof(int)) &&
         !overlaps(dst, dn, flag, sizeof(int)) &&
         (!overlaps(src, sn, dst, dn) || (src == dst && ss == ds));
}

bool tensor_span(size_t heads, size_t tokens, size_t row_bytes, size_t hs,
                 size_t ts, size_t capacity, size_t &bytes) {
  size_t head_span = 0;
  return span(heads, row_bytes, hs, SIZE_MAX, head_span) &&
         span(tokens, head_span, ts, capacity, bytes);
}

bool valid_batch(const spin8_query_v3_cuda_plan *plan,
                 const spin8_query_v3_cuda_bank_f32 *p, const float *src,
                 size_t sc, size_t sh, size_t st, float *dst, size_t dc,
                 size_t dh, size_t dt, size_t heads, size_t tokens,
                 size_t width, int *flag, size_t &rows) {
  size_t row_bytes = 0, win_bytes = 0, sn = 0, dn = 0;
  if (!pointer(plan, sizeof(*plan), alignof(spin8_query_v3_cuda_plan)) ||
      !pointer(p, sizeof(*p), alignof(spin8_query_v3_cuda_bank_f32)) ||
      p->version != SPIN8_QUERY_V3_CUDA_BANK_VERSION || p->heads != heads ||
      p->width != width || !product(heads, tokens, rows) ||
      !product(width, sizeof(float), row_bytes) ||
      !product(8, row_bytes, win_bytes) ||
      !tensor_span(heads, tokens, row_bytes, sh, st, sc, sn) ||
      !tensor_span(heads, tokens, row_bytes, dh, dt, dc, dn) ||
      !pointer(src, sn, alignof(float)) || !pointer(dst, dn, alignof(float)) ||
      !pointer(flag, sizeof(int), alignof(int)))
    return false;
  const void *parameters[] = {p->win, p->bin, p->wout, p->bout};
  const size_t payloads[] = {win_bytes, 8 * sizeof(float), 56 * sizeof(float),
                             7 * sizeof(float)};
  const size_t strides[] = {p->win_head_stride_bytes, p->bin_head_stride_bytes,
                            p->wout_head_stride_bytes,
                            p->bout_head_stride_bytes};
  const size_t capacities[] = {p->win_bytes, p->bin_bytes, p->wout_bytes,
                               p->bout_bytes};
  size_t sizes[4]{};
  for (size_t i = 0; i < 4; ++i) {
    if (!span(heads, payloads[i], strides[i], capacities[i], sizes[i]) ||
        !pointer(parameters[i], sizes[i], alignof(float)))
      return false;
    if (overlaps(parameters[i], sizes[i], src, sn) ||
        overlaps(parameters[i], sizes[i], dst, dn) ||
        overlaps(parameters[i], sizes[i], flag, sizeof(int)))
      return false;
    for (size_t j = 0; j < i; ++j)
      if (overlaps(parameters[i], sizes[i], parameters[j], sizes[j]))
        return false;
  }
  return !overlaps(src, sn, flag, sizeof(int)) &&
         !overlaps(dst, dn, flag, sizeof(int)) &&
         (!overlaps(src, sn, dst, dn) || (src == dst && sh == dh && st == dt));
}

__global__ void apply_kernel(spin8_query_v3_cuda_plan plan,
                             spin8_query_v3_cuda_bank_f32 bank,
                             const float *src, size_t sh, size_t st, float *dst,
                             size_t dh, size_t dt, size_t rows, size_t width,
                             int *flag) {
  __shared__ double hidden[8];
  __shared__ double c[8];
  __shared__ int invalid;
  __shared__ int identity;
  // Capped grid, with whole rows owned by one CTA. This also avoids a host
  // device-property query and permits arbitrarily many checked rows.
  for (size_t row = blockIdx.x; row < rows; row += gridDim.x) {
    if (threadIdx.x == 0) {
      invalid = 0;
      identity = 1;
    }
    __syncthreads();
    const size_t head = row % bank.heads, token = row / bank.heads;
    const float *q = src + token * st + head * sh;
    float *out = dst + token * dt + head * dh;
    // Only the selected head's payload is read; parameter padding is ignored.
    const spin8_query_v3_cuda_parameters_f32 p{
        bank.win + head * (bank.win_head_stride_bytes / sizeof(float)),   0,
        bank.bin + head * (bank.bin_head_stride_bytes / sizeof(float)),   0,
        bank.wout + head * (bank.wout_head_stride_bytes / sizeof(float)), 0,
        bank.bout + head * (bank.bout_head_stride_bytes / sizeof(float)), 0};
    for (size_t i = threadIdx.x; i < width; i += blockDim.x)
      if (!isfinite(q[i]))
        atomicOr(&invalid, 1);
    // Each of the first eight threads owns one complete dot product. Its
    // sequential FP64 order matches the CPU reference's F32-rounded inputs.
    if (threadIdx.x < 8) {
      const size_t k = threadIdx.x;
      double z = double(p.bin[k]);
      if (!isfinite(z))
        atomicOr(&invalid, 1);
      for (size_t i = 0; i < width; ++i) {
        const float weight = p.win[k * width + i];
        if (!isfinite(weight))
          atomicOr(&invalid, 1);
        z = __dadd_rn(z, __dmul_rn(double(weight), double(q[i])));
      }
      if (!isfinite(z))
        atomicOr(&invalid, 1);
      hidden[k] = tanh(z);
    }
    // This barrier completes ALL query reads made by the controller and
    // finite scan before any thread can overwrite an in-place block.
    __syncthreads();
    if (threadIdx.x == 0) {
      double a[7]{};
      constexpr double tau = .41421356237309504880168872420970;
      double scale = tau;
      for (int axis = 0; axis < 7; ++axis) {
        const float bias = p.bout[axis];
        if (!isfinite(bias))
          invalid = 1;
        identity &= bias == 0.f;
        a[axis] = double(bias);
        for (int k = 0; k < 8; ++k) {
          const float weight = p.wout[axis * 8 + k];
          if (!isfinite(weight))
            invalid = 1;
          identity &= weight == 0.f;
          // CPU /fp:strict rounds product and sum separately. FMA can retain
          // a large residual when opposite large Wout products should cancel.
          a[axis] = __dadd_rn(a[axis], __dmul_rn(double(weight), hidden[k]));
        }
        if (!isfinite(a[axis]))
          invalid = 1;
        scale = fmax(scale, fabs(a[axis]));
      }
      if (!invalid) {
        // Scale before squaring, as in the CPU v3 chart. Large finite F32
        // weights and inputs do not overflow the controller or chart in FP64.
        double radius2 = __dmul_rn(tau / scale, tau / scale);
        for (int axis = 0; axis < 7; ++axis) {
          const double value = a[axis] / scale;
          radius2 = __dadd_rn(radius2, __dmul_rn(value, value));
        }
        const double radius = sqrt(radius2);
        double norm2 = 1.;
        c[0] = 1.;
        for (int axis = 0; axis < 7; ++axis) {
          c[axis + 1] = tau * (a[axis] / scale) / radius;
          norm2 = __dadd_rn(norm2, __dmul_rn(c[axis + 1], c[axis + 1]));
        }
        const double inverse_norm = 1. / sqrt(norm2);
        for (int i = 0; i < 8; ++i) {
          c[i] *= inverse_norm;
          if (!isfinite(c[i]))
            invalid = 1;
        }
      }
      if (invalid)
        atomicOr(flag, SPIN8_CUDA_NONFINITE);
    }
    __syncthreads();
    if (!invalid) {
      const size_t blocks = width / 8;
      for (size_t block = threadIdx.x; block < blocks; block += blockDim.x) {
        const size_t base = block * 8;
        float x[8];
        // Only this thread reads/writes its block after controller completion.
        // Load every component before its first write for exact in-place use.
        for (int j = 0; j < 8; ++j)
          x[j] = q[base + j];
        for (int i = 0; i < 8; ++i) {
          if (identity) {
            out[base + i] = x[i];
            continue;
          }
          double value = 0.;
          // Preserve the CPU dense dot's source-column reduction order without
          // constructing a matrix. Every entry is exactly one signed c index.
          for (int j = 0; j < 8; ++j) {
            const int at = i * 8 + j;
            const double coefficient =
                __dmul_rn(c[plan.coefficient[at]], double(plan.sign[at]));
            value = __dadd_rn(value, __dmul_rn(coefficient, double(x[j])));
          }
          // Casting an out-of-range double is never used as a validity test.
          if (!isfinite(value) || fabs(value) > double(FLT_MAX)) {
            atomicOr(flag, SPIN8_CUDA_NONFINITE);
          } else {
            out[base + i] = float(value);
          }
        }
      }
      for (size_t i = blocks * 8 + threadIdx.x; i < width; i += blockDim.x)
        out[i] = q[i];
    }
    // No thread may start the next row and replace shared c while another
    // thread is still transforming the current row.
    __syncthreads();
  }
}
} // namespace

extern "C" spin8_status
spin8_query_v3_cuda_plan_create(uint32_t version,
                                spin8_query_v3_cuda_plan **out) {
  if (!pointer(out, sizeof(*out), alignof(spin8_query_v3_cuda_plan *)))
    return SPIN8_INVALID_ARGUMENT;
  *out = nullptr;
  if (version != SPIN8_QUERY_V3_VERSION)
    return SPIN8_INVALID_ARGUMENT;
  spin8_query_v3_basis_report report{};
  auto status = spin8_query_v3_basis_audit(&report);
  if (status != SPIN8_OK)
    return status;
  double generators[448]{};
  status = spin8_query_v3_generators(generators);
  if (status != SPIN8_OK)
    return status;
  auto *plan = new (std::nothrow) spin8_query_v3_cuda_plan;
  if (!plan)
    return SPIN8_ALLOCATION_FAILED;
  // Preserve the ACTUAL audited v3 convention, rather than substituting an
  // independent octonion/Fano convention with potentially different signs.
  for (size_t i = 0; i < 8; ++i)
    plan->sign[i * 8 + i] = 1;
  for (size_t axis = 0; axis < 7; ++axis) {
    bool used[8]{};
    for (size_t i = 0; i < 8; ++i) {
      int found = -1;
      for (size_t j = 0; j < 8; ++j) {
        const double value = generators[axis * 64 + i * 8 + j];
        if (value != 0.) {
          if (found >= 0 || used[j] || (value != 1. && value != -1.) ||
              plan->sign[i * 8 + j] != 0) {
            delete plan;
            return SPIN8_AUDIT_FAILED;
          }
          found = int(j);
          used[j] = true;
          plan->coefficient[i * 8 + j] = uint8_t(axis + 1);
          plan->sign[i * 8 + j] = int8_t(value);
        }
      }
      if (found < 0) {
        delete plan;
        return SPIN8_AUDIT_FAILED;
      }
    }
  }
  for (size_t i = 0; i < 64; ++i)
    if (plan->sign[i] == 0) {
      delete plan;
      return SPIN8_AUDIT_FAILED;
    }
  *out = plan;
  return SPIN8_OK;
}

extern "C" void spin8_query_v3_cuda_plan_free(spin8_query_v3_cuda_plan *plan) {
  delete plan;
}

extern "C" cudaError_t spin8_query_v3_cuda_apply_f32(
    const spin8_query_v3_cuda_plan *plan,
    const spin8_query_v3_cuda_parameters_f32 *parameters, const float *src,
    size_t sc, size_t ss, float *dst, size_t dc, size_t ds, size_t rows,
    size_t width, int *flag, cudaStream_t stream) {
  if (!valid_layout(plan, parameters, src, sc, ss, dst, dc, ds, rows, width,
                    flag))
    return cudaErrorInvalidValue;
  const unsigned blocks = unsigned(rows > 65535 ? 65535 : rows);
  const spin8_query_v3_cuda_bank_f32 bank{SPIN8_QUERY_V3_CUDA_BANK_VERSION,
                                          1,
                                          width,
                                          parameters->win,
                                          parameters->win_bytes,
                                          0,
                                          parameters->bin,
                                          parameters->bin_bytes,
                                          0,
                                          parameters->wout,
                                          parameters->wout_bytes,
                                          0,
                                          parameters->bout,
                                          parameters->bout_bytes,
                                          0};
  apply_kernel<<<blocks, 128, 0, stream>>>(
      *plan, bank, src, 0, ss / sizeof(float), dst, 0, ds / sizeof(float), rows,
      width, flag);
  return cudaGetLastError();
}

extern "C" cudaError_t spin8_query_v3_cuda_apply_batched_f32(
    const spin8_query_v3_cuda_plan *plan,
    const spin8_query_v3_cuda_bank_f32 *bank, const float *src, size_t sc,
    size_t sh, size_t st, float *dst, size_t dc, size_t dh, size_t dt,
    size_t heads, size_t tokens, size_t width, int *flag, cudaStream_t stream) {
  size_t rows = 0;
  if (!valid_batch(plan, bank, src, sc, sh, st, dst, dc, dh, dt, heads, tokens,
                   width, flag, rows))
    return cudaErrorInvalidValue;
  const unsigned blocks = unsigned(rows > 65535 ? 65535 : rows);
  apply_kernel<<<blocks, 128, 0, stream>>>(
      *plan, *bank, src, sh / sizeof(float), st / sizeof(float), dst,
      dh / sizeof(float), dt / sizeof(float), rows, width, flag);
  return cudaGetLastError();
}
