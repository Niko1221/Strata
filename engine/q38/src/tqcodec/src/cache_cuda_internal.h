#ifndef TQCODEC_CACHE_CUDA_INTERNAL_H
#define TQCODEC_CACHE_CUDA_INTERNAL_H
#include "tqcodec/cache_cuda.h"
#include <cstring>
struct tqc_cache_cuda_plan {
  tqc_cache_cuda_config config;
  tqc_cache_cuda_layout layout;
  float b[192], s1[128], s2[128];
};
struct tqc_cache_cuda_device {
  tqc_cache_cuda_config config;
  tqc_cache_cuda_layout layout;
  unsigned char *arena;
};
namespace tqc_cache_detail {
inline bool mul(size_t a, size_t b, size_t &n) {
  if (b && a > SIZE_MAX / b)
    return false;
  n = a * b;
  return true;
}
inline bool add(size_t a, size_t b, size_t &n) {
  if (a > SIZE_MAX - b)
    return false;
  n = a + b;
  return true;
}
inline bool overlap(const void *a, size_t an, const void *b, size_t bn) {
  return an && bn &&
         reinterpret_cast<uintptr_t>(a) < reinterpret_cast<uintptr_t>(b) + bn &&
         reinterpret_cast<uintptr_t>(b) < reinterpret_cast<uintptr_t>(a) + an;
}
inline bool range(const void *p, size_t n, size_t align = 1) {
  return p && reinterpret_cast<uintptr_t>(p) % align == 0 &&
         n <= UINTPTR_MAX - reinterpret_cast<uintptr_t>(p);
}
inline bool identical(const tqc_cache_cuda_plan *a,
                      const tqc_cache_cuda_plan *b) {
  return std::memcmp(a->layout.identity, b->layout.identity, 32) == 0;
}
void hash(const void *, size_t, uint8_t[32]);
int context(const tqc_cache_cuda_context *);
} // namespace tqc_cache_detail
#endif
