#include "cache_cuda_internal.h"
#include <cmath>
#include <cuda_runtime_api.h>
#include <vector>
namespace {
using namespace tqc_cache_detail;
int quiescent(const tqc_cache_cuda_context *c) {
  int rc = context(c);
  if (rc)
    return rc;
  cudaStreamCaptureStatus capture;
  auto s = static_cast<cudaStream_t>(c->stream);
  if (cudaStreamIsCapturing(s, &capture) != cudaSuccess)
    return TQC_DC_RUNTIME;
  return capture == cudaStreamCaptureStatusNone ? 0 : TQC_D_UNSUPPORTED;
}
int storage(const tqc_cache_cuda_context *c, const void *data, size_t bytes) {
  size_t required = 0;
  int rc = tqc_cache_cuda_snapshot_size(c->plan, &required);
  if (rc)
    return rc;
  if (bytes != required)
    return TQC_D_CAPACITY;
  if (!range(data, bytes))
    return TQC_D_INVALID;
  if (overlap(data, bytes, c, sizeof(*c)) ||
      overlap(data, bytes, c->plan, sizeof(*c->plan)) ||
      overlap(data, bytes, c->arena, c->arena_bytes))
    return TQC_D_OVERLAP;
  return 0;
}
bool valid_payload(const tqc_cache_cuda_plan *p, const unsigned char *data) {
  const auto &l = p->layout;
  const auto &c = p->config;
  tqc_cache_cuda_state s{};
  std::memcpy(&s, data + l.state, sizeof(s));
  if (s.status || s.valid != 1 || s.committed != s.tentative ||
      s.committed > c.capacity)
    return false;
  if (std::memcmp(data + l.matrices,
                  reinterpret_cast<const unsigned char *>(p->b),
                  sizeof(p->b)) != 0 ||
      std::memcmp(data + l.identity_offset, l.identity, 32) != 0)
    return false;
  for (int view = 0; view < 4; ++view) {
    if (view < 3 && !(c.view_mask & (1u << view)))
      continue;
    size_t offset = view < 3 ? l.k[view] : l.v,
           stride = view < 3 ? l.k_row_bytes : l.v_row_bytes;
    unsigned dialect = view < 3 ? c.k_dialect : c.v_dialect;
    for (size_t row = 0; row < s.committed * c.kv_heads; ++row)
      for (unsigned block = 0; block < c.columns / 128; ++block) {
        const auto *ptr =
            data + offset + row * stride +
            size_t(block) * (c.representation ? 512 : 18 + 16 * dialect);
        if (c.representation) {
          for (int i = 0; i < 128; ++i) {
            float value = 0;
            std::memcpy(&value, ptr + 4 * size_t(i), 4);
            if (!std::isfinite(value))
              return false;
          }
        } else {
          unsigned norm = ptr[0] | unsigned(ptr[1]) << 8;
          if ((norm & 0x7c00) == 0x7c00 || (norm & 0x8000))
            return false;
        }
      }
  }
  return true;
}
const unsigned char magic[8] = {'T', 'Q', 'C', 'C', 1, 0, 0, 0};
} // namespace
extern "C" int tqc_cache_cuda_snapshot(const tqc_cache_cuda_context *c,
                                       void *output, size_t bytes) {
  int rc = quiescent(c);
  if (rc)
    return rc;
  rc = storage(c, output, bytes);
  if (rc)
    return rc;
  try {
    std::vector<unsigned char> result(bytes, 0);
    auto stream = static_cast<cudaStream_t>(c->stream);
    rc = tqc_cache_cuda_audit_device(c);
    if (rc)
      return rc;
    if (cudaMemcpyAsync(result.data() + 80, c->arena,
                        c->plan->layout.total_bytes, cudaMemcpyDeviceToHost,
                        stream) != cudaSuccess ||
        cudaStreamSynchronize(stream) != cudaSuccess)
      return TQC_DC_RUNTIME;
    if (!valid_payload(c->plan, result.data() + 80))
      return TQC_D_INVALID;
    std::memcpy(result.data(), magic, 8);
    std::memcpy(result.data() + 8, c->plan->layout.identity, 32);
    uint64_t length = bytes;
    std::memcpy(result.data() + 72, &length, 8);
    hash(result.data() + 80, bytes - 80, result.data() + 40);
    std::memcpy(output, result.data(), bytes);
    return 0;
  } catch (const std::bad_alloc &) {
    return TQC_D_CAPACITY;
  } catch (...) {
    return TQC_D_INVALID;
  }
}
extern "C" int tqc_cache_cuda_restore(const tqc_cache_cuda_context *c,
                                      const void *input, size_t bytes) {
  int rc = quiescent(c);
  if (rc)
    return rc;
  rc = storage(c, input, bytes);
  if (rc)
    return rc;
  try {
    // Private immutable copy prevents later caller buffer changes after
    // validation.
    const auto *in = static_cast<const unsigned char *>(input);
    std::vector<unsigned char> copy(in, in + bytes);
    uint64_t length = 0;
    std::memcpy(&length, copy.data() + 72, 8);
    unsigned char digest[32];
    hash(copy.data() + 80, bytes - 80, digest);
    if (std::memcmp(copy.data(), magic, 8) != 0 ||
        std::memcmp(copy.data() + 8, c->plan->layout.identity, 32) != 0 ||
        length != bytes || std::memcmp(copy.data() + 40, digest, 32) != 0 ||
        !valid_payload(c->plan, copy.data() + 80))
      return TQC_D_INVALID;
    auto stream = static_cast<cudaStream_t>(c->stream);
    if (cudaMemcpyAsync(c->arena, copy.data() + 80, bytes - 80,
                        cudaMemcpyHostToDevice, stream) != cudaSuccess) {
      tqc_cache_cuda_control(c, TQC_CC_INVALIDATE);
      return TQC_DC_RUNTIME;
    }
    rc = tqc_cache_cuda_audit_device(c);
    if (rc) {
      tqc_cache_cuda_control(c, TQC_CC_INVALIDATE);
      cudaStreamSynchronize(stream);
      return rc;
    }
    if (cudaStreamSynchronize(stream) != cudaSuccess) {
      tqc_cache_cuda_control(c, TQC_CC_INVALIDATE);
      return TQC_DC_RUNTIME;
    }
    return 0;
  } catch (const std::bad_alloc &) {
    return TQC_D_CAPACITY;
  } catch (...) {
    return TQC_D_INVALID;
  }
}
