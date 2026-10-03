#pragma once
// Intel XPU stand-in for <cuda_fp16.h>. Layout is 2 bytes so kernels can
// reinterpret_cast packed fp16 words. Rounding is SYCL's half conversion.
#include <cstdint>
#include <cstring>
#include <sycl/sycl.hpp>

struct __attribute__((packed, aligned(2))) __half {
    uint16_t bits;
    operator float() const {
        sycl::half s;
        std::memcpy(&s, &bits, 2);
        return static_cast<float>(s);
    }
};
inline __half strata_half_bits(unsigned short u) {
    __half h;
    h.bits = u;
    return h;
}

struct __attribute__((packed, aligned(4))) __half2 {
    __half x, y;
};
using half = __half;
using half2 = __half2;

inline __half __float2half(float f) {
    sycl::half h(f);
    __half out;
    std::memcpy(&out.bits, &h, 2);
    return out;
}
inline __half __float2half_rn(float f) { return __float2half(f); }
inline float __half2float(__half h) {
    sycl::half s;
    std::memcpy(&s, &h.bits, 2);
    return static_cast<float>(s);
}
inline unsigned short __half_as_ushort(__half h) { return h.bits; }
inline __half __ushort_as_half(unsigned short u) { return strata_half_bits(u); }
inline __half __ushort_as_half(unsigned int u) { return strata_half_bits(static_cast<unsigned short>(u)); }

inline __half2 __floats2half2_rn(float a, float b) {
    __half2 h;
    h.x = __float2half_rn(a);
    h.y = __float2half_rn(b);
    return h;
}
inline __half2 __halves2half2(__half a, __half b) {
    __half2 h;
    h.x = a;
    h.y = b;
    return h;
}
inline float __low2float(__half2 h) { return __half2float(h.x); }
inline float __high2float(__half2 h) { return __half2float(h.y); }
inline __half __low2half(__half2 h) { return h.x; }
inline __half __high2half(__half2 h) { return h.y; }
inline float __low2float(sycl::half2 h) { return static_cast<float>(h[0]); }
inline float __high2float(sycl::half2 h) { return static_cast<float>(h[1]); }
inline half2 make_half2(half a, half b) { return __halves2half2(a, b); }
inline half2 make_half2(float a, float b) { return __floats2half2_rn(a, b); }

inline __half2 __hadd2(__half2 a, __half2 b) {
    return __floats2half2_rn(__half2float(a.x) + __half2float(b.x), __half2float(a.y) + __half2float(b.y));
}
inline __half2 __hsub2(__half2 a, __half2 b) {
    return __floats2half2_rn(__half2float(a.x) - __half2float(b.x), __half2float(a.y) - __half2float(b.y));
}
