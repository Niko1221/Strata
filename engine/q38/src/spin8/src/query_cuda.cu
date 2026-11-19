#include "spin8/query_cuda.h"

#include <cmath>
#include <cstdint>
#include <cuda_runtime.h>
#include <new>

struct spin8_query_cuda_plan {
  uint8_t spinor_index[64]{};
  int8_t gamma_sign[64]{};
  uint8_t base_index[8]{};
  int8_t base_sign[8]{};
};

namespace {
bool span(size_t rows, size_t width, size_t stride, size_t capacity,
          size_t &bytes) {
  if (!rows || !width || stride < width ||
      rows - 1 > (SIZE_MAX - width) / stride)
    return false;
  const size_t count = (rows - 1) * stride + width;
  if (count > SIZE_MAX / sizeof(float))
    return false;
  bytes = count * sizeof(float);
  return bytes <= capacity;
}

bool overlaps(const void *a, size_t an, const void *b, size_t bn) {
  const auto x = reinterpret_cast<uintptr_t>(a);
  const auto y = reinterpret_cast<uintptr_t>(b);
  if (x > UINTPTR_MAX - an || y > UINTPTR_MAX - bn)
    return true;
  return x < y + bn && y < x + an;
}

bool valid_layout(const spin8_query_cuda_plan *plan, const float *c, size_t cc,
                  size_t cs, const float *src, size_t sc, size_t ss, float *dst,
                  size_t dc, size_t ds, size_t rows, size_t width, int *flag,
                  size_t &tiles) {
  size_t cn = 0, sn = 0, dn = 0;
  if (!plan || !c || !src || !dst || !flag ||
      reinterpret_cast<uintptr_t>(c) % alignof(float) ||
      reinterpret_cast<uintptr_t>(src) % alignof(float) ||
      reinterpret_cast<uintptr_t>(dst) % alignof(float) ||
      reinterpret_cast<uintptr_t>(flag) % alignof(int) ||
      !span(rows, 8, cs, cc, cn) || !span(rows, width, ss, sc, sn) ||
      !span(rows, width, ds, dc, dn))
    return false;
  tiles = width / 8 + (width % 8 != 0);
  if (rows > SIZE_MAX / tiles || overlaps(c, cn, src, sn) ||
      overlaps(c, cn, dst, dn) || overlaps(c, cn, flag, sizeof(int)) ||
      overlaps(src, sn, flag, sizeof(int)) ||
      overlaps(dst, dn, flag, sizeof(int)) ||
      (overlaps(src, sn, dst, dn) && (src != dst || ss != ds)))
    return false;
  return true;
}

__global__ void transform_kernel(const float *c, size_t cs, const float *src,
                                 size_t ss, float *dst, size_t ds, size_t rows,
                                 size_t width, size_t tiles,
                                 spin8_query_cuda_plan plan, int *flag) {
  const size_t count = rows * tiles;
  for (size_t tile = blockIdx.x * blockDim.x + threadIdx.x; tile < count;
       tile += size_t(gridDim.x) * blockDim.x) {
    const size_t row = tile / tiles;
    const size_t col = (tile % tiles) * 8;
    float spinor[8];
    double norm2 = 0.0;
    bool finite = true, identity = true;
    for (int b = 0; b < 8; ++b) {
      const float value = c[row * cs + b];
      spinor[b] = value;
      finite &= isfinite(value);
      identity &= value == (b == 0 ? 1.0f : 0.0f);
      if (isfinite(value))
        norm2 += double(value) * double(value);
    }
    if (!finite) {
      atomicOr(flag, SPIN8_CUDA_NONFINITE | SPIN8_QUERY_CUDA_INVALID_SPINOR);
      continue;
    }
    if (fabs(sqrt(norm2) - 1.0) > 1e-5) {
      atomicOr(flag, SPIN8_QUERY_CUDA_INVALID_SPINOR);
      continue;
    }
    const int active = int(width - col < 8 ? width - col : 8);
    float x[8]{};
    bool input_finite = true;
    for (int j = 0; j < active; ++j) {
      x[j] = src[row * ss + col + j];
      input_finite &= isfinite(x[j]);
    }
    if (!input_finite)
      atomicOr(flag, SPIN8_CUDA_NONFINITE);
    if (active < 8 || identity) {
      for (int j = 0; j < active; ++j)
        dst[row * ds + col + j] = x[j];
      continue;
    }

    // First contract C(e0)^T with x. Its signed permutation gives one source
    // coordinate per spin-plus index; the second contraction applies C(c).
    float z[8];
    for (int a = 0; a < 8; ++a)
      z[a] = plan.base_sign[a] * x[plan.base_index[a]];
    for (int i = 0; i < 8; ++i) {
      float result = 0.0f;
      for (int a = 0; a < 8; ++a) {
        const int at = i * 8 + a;
        result += plan.gamma_sign[at] * spinor[plan.spinor_index[at]] * z[a];
      }
      if (!isfinite(result))
        atomicOr(flag, SPIN8_CUDA_NONFINITE);
      dst[row * ds + col + i] = result;
    }
  }
}
} // namespace

extern "C" spin8_status
spin8_query_cuda_plan_create(spin8_query_cuda_plan **out) {
  if (!out)
    return SPIN8_INVALID_ARGUMENT;
  *out = nullptr;
  double gamma[2048]{};
  const auto status = spin8_clifford_generators(gamma);
  if (status != SPIN8_OK)
    return status;
  auto *plan = new (std::nothrow) spin8_query_cuda_plan;
  if (!plan)
    return SPIN8_ALLOCATION_FAILED;
  for (int i = 0; i < 8; ++i) {
    bool used_column[8]{};
    for (int a = 0; a < 8; ++a) {
      int found = -1;
      for (int b = 0; b < 8; ++b) {
        const double entry = gamma[i * 256 + a * 16 + 8 + b];
        if (entry != 0.0) {
          if (found >= 0 || (entry != 1.0 && entry != -1.0) || used_column[b]) {
            delete plan;
            return SPIN8_AUDIT_FAILED;
          }
          found = b;
          used_column[b] = true;
          plan->spinor_index[i * 8 + a] = uint8_t(b);
          plan->gamma_sign[i * 8 + a] = int8_t(entry);
        }
      }
      if (found < 0) {
        delete plan;
        return SPIN8_AUDIT_FAILED;
      }
    }
  }
  bool used_output[8]{};
  for (int a = 0; a < 8; ++a) {
    int found = -1;
    for (int j = 0; j < 8; ++j)
      if (plan->spinor_index[j * 8 + a] == 0) {
        if (found >= 0) {
          delete plan;
          return SPIN8_AUDIT_FAILED;
        }
        found = j;
        plan->base_index[a] = uint8_t(j);
        plan->base_sign[a] = plan->gamma_sign[j * 8 + a];
      }
    if (found < 0 || used_output[found]) {
      delete plan;
      return SPIN8_AUDIT_FAILED;
    }
    used_output[found] = true;
  }
  *out = plan;
  return SPIN8_OK;
}

extern "C" void spin8_query_cuda_plan_free(spin8_query_cuda_plan *plan) {
  delete plan;
}

extern "C" cudaError_t spin8_query_cuda_transform_f32(
    const spin8_query_cuda_plan *plan, const float *c, size_t cc, size_t cs,
    const float *src, size_t sc, size_t ss, float *dst, size_t dc, size_t ds,
    size_t rows, size_t width, int *flag, cudaStream_t stream) {
  size_t tiles = 0;
  if (!valid_layout(plan, c, cc, cs, src, sc, ss, dst, dc, ds, rows, width,
                    flag, tiles))
    return cudaErrorInvalidValue;
  const size_t count = rows * tiles;
  constexpr unsigned threads = 128;
  const size_t needed = count / threads + (count % threads != 0);
  const unsigned blocks = unsigned(needed > 65535 ? 65535 : needed);
  transform_kernel<<<blocks, threads, 0, stream>>>(
      c, cs, src, ss, dst, ds, rows, width, tiles, *plan, flag);
  return cudaGetLastError();
}
