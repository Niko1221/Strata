#include "dialect_cuda_device.cuh"
#include "tqcodec/dialects_cuda.h"
#include <algorithm>
#include <cstdint>

namespace {
struct Range {
  uintptr_t begin, end;
};
int range(const void *p, size_t n, Range &r) {
  r.begin = reinterpret_cast<uintptr_t>(p);
  if (n > UINTPTR_MAX - r.begin)
    return TQC_D_OVERFLOW;
  if (!p && n)
    return TQC_D_INVALID;
  r.end = r.begin + n;
  return TQC_D_OK;
}
bool overlaps(Range a, Range b) { return a.begin < b.end && b.begin < a.end; }
bool mul(size_t a, size_t b, size_t &n) {
  if (b && a > SIZE_MAX / b)
    return false;
  n = a * b;
  return true;
}
int span(tqc_dialect_v1_span s, size_t rows, size_t width, size_t align,
         Range &r) {
  size_t last = 0;
  if (!mul(rows - 1, s.row_stride_bytes, last) || last > SIZE_MAX - width)
    return TQC_D_OVERFLOW;
  if (s.row_stride_bytes < width || s.capacity_bytes < last + width)
    return TQC_D_CAPACITY;
  int rc = range(s.data, last + width, r);
  if (rc)
    return rc;
  Range full{};
  rc = range(s.data, s.capacity_bytes, full);
  if (rc)
    return rc;
  return r.begin % align || s.row_stride_bytes % align ? TQC_D_ALIGNMENT
                                                       : TQC_D_OK;
}

// A scalar worker preserves the pinned ordered reductions. Independent blocks
// run concurrently. This contract kernel makes no throughput claim.
__global__ void execute_blocks(tqc_dialect_cuda_v1_request q) {
  using namespace tqc_cuda_detail;
  const size_t blocks = q.columns / 128, total = q.rows * blocks;
  const unsigned bytes = 18 + q.dialect * 16;
  const bool encode = q.operation == TQC_D_ENCODE || q.operation == TQC_D_PACK;
  const bool decode = q.operation == TQC_D_DECODE;
  for (size_t index = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
       index < total; index += size_t(gridDim.x) * blockDim.x) {
    // Atomic read avoids a race with another worker discovering an error.
    if (atomicOr(q.status, 0u))
      return;
    const size_t row = index / blocks, block = index % blocks;
    const auto *src = static_cast<const uint8_t *>(q.src.data) +
                      row * q.src.row_stride_bytes +
                      block * (decode ? bytes : 512);
    auto *dst = static_cast<uint8_t *>(q.dst.data) +
                row * q.dst.row_stride_bytes + block * (encode ? bytes : 512);
    float x[128], y[128], norm = 0;
    uint8_t packed[98];
    uint32_t error = 0;
    if (decode) {
      for (unsigned i = 0; i < bytes; ++i)
        packed[i] = src[i];
      if (!isfinite(half_decode(load16(packed))))
        error |= TQC_DC_NONFINITE;
    } else {
      for (unsigned i = 0; i < 128; ++i) {
        x[i] = reinterpret_cast<const float *>(src)[i];
        if (!isfinite(x[i]))
          error |= TQC_DC_NONFINITE;
        else if (fabsf(x[i]) > 65504.0f)
          error |= TQC_DC_RANGE;
      }
    }
    float *np = nullptr;
    if (q.operation == TQC_D_PREPARE || q.operation == TQC_D_PACK)
      np =
          reinterpret_cast<float *>(static_cast<uint8_t *>(q.norms.data) +
                                    row * q.norms.row_stride_bytes + block * 4);
    if (q.operation == TQC_D_PACK) {
      norm = *np;
      if (!isfinite(norm))
        error |= TQC_DC_NONFINITE;
      else if (norm < 0)
        error |= TQC_DC_RANGE;
    }
    if (error) {
      atomicOr(q.status, error);
      continue;
    }
    if (decode)
      turbo_decode(q.dialect, packed, y);
    else if (q.operation == TQC_D_PACK)
      turbo_pack(q.dialect, x, norm, packed);
    else if (q.operation == TQC_D_FORWARD || q.operation == TQC_D_INVERSE) {
      for (unsigned i = 0; i < 128; ++i)
        y[i] = x[i];
      fwht(y, q.operation == TQC_D_INVERSE);
    } else {
      norm = prepare(q.dialect, x, y);
      if (encode)
        turbo_pack(q.dialect, y, norm, packed);
    }
    if (encode) {
      if (!isfinite(half_decode(load16(packed))))
        error |= TQC_DC_RANGE;
    } else {
      for (unsigned i = 0; i < 128; ++i)
        if (!isfinite(y[i]))
          error |= TQC_DC_RANGE;
    }
    if (error) {
      atomicOr(q.status, error);
      continue;
    }
    if (encode)
      for (unsigned i = 0; i < bytes; ++i)
        dst[i] = packed[i];
    else
      for (unsigned i = 0; i < 128; ++i)
        reinterpret_cast<float *>(dst)[i] = y[i];
    if (q.operation == TQC_D_PREPARE)
      *np = norm;
  }
}
} // namespace

extern "C" int tqc_dialect_cuda_v1_query(uint32_t id,
                                         tqc_dialect_cuda_v1_info *out) {
  Range r{};
  int rc = range(out, sizeof(*out), r);
  if (rc)
    return rc;
  if (r.begin % alignof(tqc_dialect_cuda_v1_info))
    return TQC_D_ALIGNMENT;
  if (out->version != 1 || out->size != sizeof(*out))
    return TQC_D_VERSION;
  if (id < 1 || id > 5)
    return TQC_D_UNSUPPORTED;
  static const char *identities[] = {"thetom-64395689/turbo2-128-f16norm/v1",
                                     "thetom-64395689/turbo3-128-f16norm/v1",
                                     "thetom-64395689/turbo4-128-4bit/v1",
                                     "thetom-64395689/turbo5-128-signmag/v1",
                                     "thetom-64395689/turbo6-128-planes/v1"};
  *out = {1,
          sizeof(*out),
          id,
          128,
          18 + 16 * id,
          126,
          TQC_D_F32,
          120,
          1,
          1,
          identities[id - 1],
          "643956891dfdf51aecdbd05ca508dd21e21e8195"};
  return TQC_D_OK;
}
extern "C" int
tqc_dialect_cuda_v1_enqueue(const tqc_dialect_cuda_v1_request *p) {
  Range descriptor{};
  int rc = range(p, sizeof(*p), descriptor);
  if (rc)
    return rc;
  if (descriptor.begin % alignof(tqc_dialect_cuda_v1_request))
    return TQC_D_ALIGNMENT;
  if (p->version != 1 || p->size != sizeof(*p))
    return TQC_D_VERSION;
  const auto q = *p;
  if (q.dialect < 1 || q.dialect > 5 || q.operation < 1 || q.operation > 6 ||
      q.precision != TQC_D_F32 || q.device_ordinal < 0)
    return TQC_D_UNSUPPORTED;
  if (!q.rows || !q.columns || q.columns % 128)
    return TQC_D_INVALID;
  size_t floats = 0, blocks = q.columns / 128, packed = 0, norms = 0, total = 0,
         payload = 0;
  if (!mul(q.columns, 4, floats) || !mul(blocks, 18 + 16 * q.dialect, packed) ||
      !mul(blocks, 4, norms) || !mul(q.rows, blocks, total) ||
      !mul(total, 18 + 16 * q.dialect, payload))
    return TQC_D_OVERFLOW;
  if (q.columns != 128 && q.columns != 256)
    return TQC_D_UNSUPPORTED;
  const bool encode = q.operation == TQC_D_ENCODE || q.operation == TQC_D_PACK;
  const bool decode = q.operation == TQC_D_DECODE;
  const bool hasnorm =
      q.operation == TQC_D_PREPARE || q.operation == TQC_D_PACK;
  Range spans[4]{};
  rc = span(q.src, q.rows, decode ? packed : floats, decode ? 1 : 4, spans[0]);
  if (rc)
    return rc;
  rc = span(q.dst, q.rows, encode ? packed : floats, encode ? 1 : 4, spans[1]);
  if (rc)
    return rc;
  if (hasnorm) {
    rc = span(q.norms, q.rows, norms, 4, spans[2]);
    if (rc)
      return rc;
  } else if (q.norms.data || q.norms.capacity_bytes || q.norms.row_stride_bytes)
    return TQC_D_INVALID;
  if (q.status_capacity_bytes < 4)
    return TQC_D_CAPACITY;
  rc = range(q.status, q.status_capacity_bytes, spans[3]);
  if (rc)
    return rc;
  if (spans[3].begin % 4)
    return TQC_D_ALIGNMENT;
  for (unsigned i = 0; i < 4; ++i) {
    if (overlaps(spans[i], descriptor))
      return TQC_D_OVERLAP;
    for (unsigned j = 0; j < i; ++j)
      if (overlaps(spans[i], spans[j]))
        return TQC_D_OVERLAP;
  }
  if (!q.stream || q.stream == reinterpret_cast<void *>(1) ||
      q.stream == reinterpret_cast<void *>(2))
    return TQC_D_UNSUPPORTED;
  // No device switch, allocations, pointer-ownership probes or synchronization.
  int current = -1, stream_device = -1;
  if (cudaGetDevice(&current) != cudaSuccess)
    return TQC_DC_RUNTIME;
  if (current != q.device_ordinal)
    return TQC_D_UNSUPPORTED;
  auto stream = static_cast<cudaStream_t>(q.stream);
  if (cudaStreamGetDevice(stream, &stream_device) != cudaSuccess)
    return TQC_DC_RUNTIME;
  if (stream_device != current)
    return TQC_D_UNSUPPORTED;
  cudaDeviceProp prop{};
  if (cudaGetDeviceProperties(&prop, current) != cudaSuccess)
    return TQC_DC_RUNTIME;
  if (prop.major != 12 || prop.minor != 0)
    return TQC_D_UNSUPPORTED;
  if (cudaPeekAtLastError() != cudaSuccess)
    return TQC_DC_RUNTIME;
  const unsigned grid =
      static_cast<unsigned>(std::min(size_t(65535), (total - 1) / 32 + 1));
  execute_blocks<<<grid, 32, 0, stream>>>(q);
  return cudaPeekAtLastError() == cudaSuccess ? TQC_D_OK : TQC_DC_RUNTIME;
}
