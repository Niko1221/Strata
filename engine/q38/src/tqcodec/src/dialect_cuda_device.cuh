#ifndef TQCODEC_DIALECT_CUDA_DEVICE_CUH
#define TQCODEC_DIALECT_CUDA_DEVICE_CUH
#include <cstdint>
#include <cuda_runtime.h>
// Private scalar-per-block primitives, derived from the pinned CPU dialect.
// Preserve ordered binary32 expressions: compile with fmad=false, ftz=false,
// prec-div=true, prec-sqrt=true. Parallelize blocks, not these reductions.
// PREPARE and PACK are deliberately separate for later fused U=B F insertion.
// No allocations, launches, synchronization or status ownership here.
namespace tqc_cuda_detail {
__device__ inline float from_bits(uint32_t u) { return __uint_as_float(u); }
__device__ inline uint32_t bits(float f) { return __float_as_uint(f); }
// Private arithmetic/table material derived from TheTom
// 643956891dfdf51aecdbd05ca508dd21e21e8195. See DIALECT_LICENSE.txt. No GGML
// ABI or dependency is exported.
static __device__ __constant__ const float CENTROIDS_2BIT[4] = {
    -0.133462f, -0.039994f, 0.039994f, 0.133462f};
static __device__ __constant__ const float CENTROIDS_3BIT[8] = {
    -0.190207f, -0.118786f, -0.066822f, -0.021663f,
    0.021663f,  0.066822f,  0.118786f,  0.190207f};
static __device__ __constant__ const float TURBO6_CENTROIDS[64] = {
    -0.330935f, -0.286417f, -0.257865f, -0.236198f, -0.218435f, -0.203203f,
    -0.189753f, -0.177626f, -0.166522f, -0.156230f, -0.146600f, -0.137517f,
    -0.128895f, -0.120663f, -0.112765f, -0.105157f, -0.097801f, -0.090663f,
    -0.083717f, -0.076940f, -0.070310f, -0.063809f, -0.057422f, -0.051133f,
    -0.044929f, -0.038798f, -0.032729f, -0.026710f, -0.020733f, -0.014787f,
    -0.008863f, -0.002953f, 0.002953f,  0.008863f,  0.014787f,  0.020733f,
    0.026710f,  0.032729f,  0.038798f,  0.044929f,  0.051133f,  0.057422f,
    0.063809f,  0.070310f,  0.076940f,  0.083717f,  0.090663f,  0.097801f,
    0.105157f,  0.112765f,  0.120663f,  0.128895f,  0.137517f,  0.146600f,
    0.156230f,  0.166522f,  0.177626f,  0.189753f,  0.203203f,  0.218435f,
    0.236198f,  0.257865f,  0.286417f,  0.330935f};
static __device__ __constant__ const float TURBO6_MID[63] = {
    -0.308676f, -0.272141f, -0.247031f, -0.227316f, -0.210819f, -0.196478f,
    -0.183690f, -0.172074f, -0.161376f, -0.151415f, -0.142059f, -0.133206f,
    -0.124779f, -0.116714f, -0.108961f, -0.101479f, -0.094232f, -0.087190f,
    -0.080329f, -0.073625f, -0.067060f, -0.060616f, -0.054277f, -0.048031f,
    -0.041864f, -0.035763f, -0.029719f, -0.023722f, -0.017760f, -0.011825f,
    -0.005908f, 0.000000f,  0.005908f,  0.011825f,  0.017760f,  0.023722f,
    0.029719f,  0.035763f,  0.041864f,  0.048031f,  0.054277f,  0.060616f,
    0.067060f,  0.073625f,  0.080329f,  0.087190f,  0.094232f,  0.101479f,
    0.108961f,  0.116714f,  0.124779f,  0.133206f,  0.142059f,  0.151415f,
    0.161376f,  0.172074f,  0.183690f,  0.196478f,  0.210819f,  0.227316f,
    0.247031f,  0.272141f,  0.308676f};
static __device__ __constant__ const float TURBO5_CENTROIDS[32] = {
    -0.288236f, -0.237892f, -0.204892f, -0.179348f, -0.158003f, -0.139353f,
    -0.122569f, -0.107141f, -0.092730f, -0.079097f, -0.066064f, -0.053491f,
    -0.041269f, -0.029304f, -0.017515f, -0.005827f, 0.005827f,  0.017515f,
    0.029304f,  0.041269f,  0.053491f,  0.066064f,  0.079097f,  0.092730f,
    0.107141f,  0.122569f,  0.139353f,  0.158003f,  0.179348f,  0.204892f,
    0.237892f,  0.288236f};
static __device__ __constant__ const float TURBO5_MID[31] = {
    -0.263064f, -0.221392f, -0.192120f, -0.168675f, -0.148678f, -0.130961f,
    -0.114855f, -0.099935f, -0.085914f, -0.072580f, -0.059778f, -0.047380f,
    -0.035287f, -0.023410f, -0.011671f, 0.000000f,  0.011671f,  0.023410f,
    0.035287f,  0.047380f,  0.059778f,  0.072580f,  0.085914f,  0.099935f,
    0.114855f,  0.130961f,  0.148678f,  0.168675f,  0.192120f,  0.221392f,
    0.263064f};
static __device__ __constant__ const float turbo_cpu_s1[128] = {
    -1, 1,  1,  -1, -1, 1,  -1, 1,  -1, -1, 1,  1,  1,  1,  1,  1,  1,  -1, 1,
    -1, 1,  -1, -1, 1,  1,  1,  -1, 1,  1,  -1, -1, -1, -1, 1,  1,  -1, 1,  1,
    -1, 1,  -1, 1,  1,  -1, -1, 1,  -1, 1,  1,  1,  1,  -1, -1, -1, -1, -1, 1,
    -1, 1,  1,  1,  1,  -1, 1,  -1, -1, 1,  -1, -1, -1, 1,  -1, -1, -1, 1,  -1,
    -1, -1, 1,  1,  1,  -1, -1, 1,  1,  1,  -1, -1, 1,  1,  -1, 1,  1,  -1, 1,
    -1, -1, 1,  1,  -1, 1,  -1, 1,  -1, 1,  1,  1,  1,  -1, 1,  -1, 1,  1,  -1,
    1,  1,  -1, -1, -1, -1, -1, 1,  1,  -1, 1,  1,  -1, 1};
static __device__ __constant__ const float turbo_cpu_s2[128] = {
    1,  1,  1,  1,  -1, 1,  1,  -1, 1,  -1, -1, -1, 1,  -1, -1, -1, 1,  1,  -1,
    -1, 1,  -1, 1,  -1, 1,  -1, -1, 1,  -1, 1,  1,  1,  1,  1,  -1, -1, -1, 1,
    -1, -1, -1, -1, -1, -1, 1,  1,  1,  -1, 1,  -1, 1,  1,  1,  -1, -1, 1,  -1,
    -1, -1, -1, -1, -1, 1,  1,  1,  -1, 1,  -1, -1, -1, -1, 1,  -1, 1,  -1, 1,
    -1, -1, 1,  1,  -1, 1,  -1, 1,  1,  -1, 1,  -1, -1, -1, -1, 1,  -1, -1, 1,
    -1, 1,  -1, 1,  1,  1,  -1, -1, 1,  -1, 1,  -1, 1,  1,  -1, -1, 1,  -1, 1,
    -1, 1,  1,  -1, 1,  -1, 1,  -1, -1, -1, -1, -1, 1,  -1};
static __device__ __constant__ const float CENTROIDS_4BIT[16] = {
    -0.241529f, -0.182877f, -0.143016f, -0.111036f, -0.083292f, -0.058050f,
    -0.034299f, -0.011349f, 0.011349f,  0.034299f,  0.058050f,  0.083292f,
    0.111036f,  0.143016f,  0.182877f,  0.241529f};

static __device__ int nearest_centroid_2bit(float val) {
  /* Binary search on midpoints: {-0.133, -0.040, 0.040, 0.133} */
  if (val < -0.086728f)
    return 0; /* midpoint(-0.133, -0.040) */
  if (val < 0.000000f)
    return 1; /* midpoint(-0.040, 0.040) */
  if (val < 0.086728f)
    return 2; /* midpoint(0.040, 0.133) */
  return 3;
}
static __device__ int nearest_centroid_3bit(float val) {
  /* 8 centroids, find nearest via midpoints */
  if (val < -0.154496f)
    return 0;
  if (val < -0.092804f)
    return 1;
  if (val < -0.044243f)
    return 2;
  if (val < 0.000000f)
    return 3;
  if (val < 0.044243f)
    return 4;
  if (val < 0.092804f)
    return 5;
  if (val < 0.154496f)
    return 6;
  return 7;
}
static __device__ int nearest_centroid_4bit(float val) {
  /* 16 centroids, optimal for N(0, 1/sqrt(128)), find nearest via midpoints */
  if (val < -0.212203f)
    return 0;
  if (val < -0.162947f)
    return 1;
  if (val < -0.127026f)
    return 2;
  if (val < -0.097164f)
    return 3;
  if (val < -0.070671f)
    return 4;
  if (val < -0.046174f)
    return 5;
  if (val < -0.022824f)
    return 6;
  if (val < 0.000000f)
    return 7;
  if (val < 0.022824f)
    return 8;
  if (val < 0.046174f)
    return 9;
  if (val < 0.070671f)
    return 10;
  if (val < 0.097164f)
    return 11;
  if (val < 0.127026f)
    return 12;
  if (val < 0.162947f)
    return 13;
  if (val < 0.212203f)
    return 14;
  return 15;
}
static __device__ int nearest_centroid_5bit(float val) {
  int lo = 0;
  int hi = 31;
  while (lo < hi) {
    const int mid = (lo + hi) / 2;
    if (val < TURBO5_MID[mid]) {
      hi = mid;
    } else {
      lo = mid + 1;
    }
  }
  return lo;
}
static __device__ int nearest_centroid_6bit(float val) {
  int lo = 0; /* candidate code range [lo, hi] */
  int hi = 63;
  while (lo < hi) {
    const int mid =
        (lo + hi) / 2; /* compare against the lo/hi split at TURBO6_MID[mid] */
    if (val < TURBO6_MID[mid]) {
      hi = mid;
    } else {
      lo = mid + 1;
    }
  }
  return lo;
}

// ggml-impl.h software conversion, MIT attribution in DIALECT_LICENSE.txt.
__device__ inline uint16_t half_encode(float f) {
  // Avoid an intermediate FP32 overflow for finite values outside FP16 range.
  // The pinned RNE converter returns this same infinity representation.
  if (fabsf(f) >= 65520.0f)
    return uint16_t((bits(f) >> 16 & 0x8000u) | 0x7c00u);
  float base = (fabsf(f) * from_bits(0x77800000)) * from_bits(0x08800000);
  const uint32_t w = bits(f), shl = w + w, sign = w & 0x80000000u;
  uint32_t bias = shl & 0xff000000u;
  if (bias < 0x71000000u)
    bias = 0x71000000u;
  base = from_bits((bias >> 1) + 0x07800000u) + base;
  const uint32_t v = bits(base);
  return static_cast<uint16_t>(
      (sign >> 16) |
      (shl > 0xff000000u ? 0x7e00u : ((v >> 13) & 0x7c00u) + (v & 0xfffu)));
}
__device__ inline float half_decode(uint16_t h) {
  const uint32_t w = uint32_t(h) << 16, twice = w + w;
  const float normal =
      from_bits((twice >> 4) + (0xe0u << 23)) * from_bits(0x07800000);
  const float sub = from_bits((twice >> 17) | (126u << 23)) - 0.5f;
  return from_bits((w & 0x80000000u) | bits(twice < (1u << 27) ? sub : normal));
}

__device__ inline uint16_t load16(const uint8_t *p) {
  return uint16_t(p[0]) | uint16_t(uint16_t(p[1]) << 8);
}
__device__ inline void store16(uint8_t *p, uint16_t v) {
  p[0] = uint8_t(v);
  p[1] = uint8_t(v >> 8);
}
__device__ inline void fwht(float *x, bool inverse) {
  const float *a = inverse ? turbo_cpu_s2 : turbo_cpu_s1;
  const float *b = inverse ? turbo_cpu_s1 : turbo_cpu_s2;
#pragma unroll 1
  for (int i = 0; i < 128; ++i)
    x[i] = __fmul_rn(x[i], a[i]);
#pragma unroll 1
  for (int h = 1; h < 128; h *= 2)
#pragma unroll 1
    for (int i = 0; i < 128; i += h * 2)
#pragma unroll 1
      for (int j = i; j < i + h; ++j) {
        const float u = x[j], v = x[j + h];
        x[j] = __fadd_rn(u, v);
        x[j + h] = __fsub_rn(u, v);
      }
#pragma unroll 1
  for (int i = 0; i < 128; ++i)
    x[i] = __fmul_rn(x[i], __fmul_rn(0.08838834764831845f, b[i]));
}
__device__ inline float prepare(uint32_t id, const float *x, float *out) {
  float sum = 0;
#pragma unroll 1
  for (int i = 0; i < 128; ++i)
    sum = __fadd_rn(sum, __fmul_rn(x[i], x[i]));
  const float norm = __fsqrt_rn(sum),
              inv = norm > 1e-10f ? __fdiv_rn(1.0f, norm) : 0.0f;
#pragma unroll 1
  for (int i = 0; i < 128; ++i)
    out[i] = (id <= 2 || norm > 1e-10f) ? __fmul_rn(x[i], inv) : 0.0f;
  fwht(out, false);
  return norm;
}
__device__ inline const float *centroids(uint32_t id) {
  switch (id) {
  case 1:
    return CENTROIDS_2BIT;
  case 2:
    return CENTROIDS_3BIT;
  case 3:
    return CENTROIDS_4BIT;
  case 4:
    return TURBO5_CENTROIDS;
  default:
    return TURBO6_CENTROIDS;
  }
}
__device__ inline int nearest(uint32_t id, float v) {
  switch (id) {
  case 1:
    return nearest_centroid_2bit(v);
  case 2:
    return nearest_centroid_3bit(v);
  case 3:
    return nearest_centroid_4bit(v);
  case 4:
    return nearest_centroid_5bit(v);
  default:
    return nearest_centroid_6bit(v);
  }
}
__device__ inline void turbo_pack(uint32_t id, const float *x, float norm,
                                  uint8_t *p) {
#pragma unroll 1
  for (unsigned i = 0; i < 18 + 16 * id; ++i)
    p[i] = 0;
  const float *c = centroids(id);
  float sum = 0;
#pragma unroll 1
  for (int i = 0; i < 128; ++i) {
    int idx = nearest(id, x[i]);
    sum = __fadd_rn(sum, __fmul_rn(c[idx], c[idx]));
    if (id <= 2) {
      p[2 + i / 4] |= uint8_t((idx & 3) << ((i % 4) * 2));
      if (id == 2)
        p[34 + i / 8] |= uint8_t((idx >> 2) << (i % 8));
    } else {
      int lo = idx & 15;
      if (id == 4) {
        const int neg = idx < 16;
        lo = neg ? 15 - idx : idx - 16;
        p[66 + i / 8] |= uint8_t(neg << (i % 8));
      }
      if (id == 5)
        p[66 + i / 4] |= uint8_t((idx >> 4) << ((i % 4) * 2));
      p[2 + i / 2] |= uint8_t(lo << ((i % 2) * 4));
    }
  }
  const float recon = __fsqrt_rn(sum);
  store16(p, half_encode(recon > 1e-10f ? __fdiv_rn(norm, recon) : norm));
}
// Random-access stored-basis decode also serves native packed attention.
__device__ inline float turbo_value(uint32_t id, const uint8_t *p, int i) {
    int idx;
    if (id <= 2) {
      idx = (p[2 + i / 4] >> ((i % 4) * 2)) & 3;
      if (id == 2)
        idx |= ((p[34 + i / 8] >> (i % 8)) & 1) << 2;
    } else {
      idx = (p[2 + i / 2] >> ((i % 2) * 4)) & 15;
      if (id == 4)
        idx = ((p[66 + i / 8] >> (i % 8)) & 1) ? 15 - idx : 16 + idx;
      if (id == 5)
        idx |= ((p[66 + i / 4] >> ((i % 4) * 2)) & 3) << 4;
    }
    return __fmul_rn(centroids(id)[idx], half_decode(load16(p)));
}
__device__ inline void turbo_decode(uint32_t id, const uint8_t *p, float *x) {
#pragma unroll 1
  for (int i = 0; i < 128; ++i) x[i] = turbo_value(id, p, i);
}

} // namespace tqc_cuda_detail
#endif
