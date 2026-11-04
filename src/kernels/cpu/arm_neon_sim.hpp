// src/kernels/cpu/arm_neon_sim.hpp - deterministic C++ emulation of Arm64 NEON / FEAT_DotProd / FEAT_FP16 intrinsics
// for testing src/kernels/cpu/portable.cpp on x86-64 hosts (arm_neon_parity).
// On real aarch64 / _M_ARM64 builds, <arm_neon.h> is included instead.
#pragma once

#include <cmath>
#include <cstdint>
#include <cstring>

struct int8x8_t    { int8_t   v[8]; };
struct uint8x8_t   { uint8_t  v[8]; };
struct int16x4_t   { int16_t  v[4]; };
struct uint16x4_t  { uint16_t v[4]; };
struct int32x2_t   { int32_t  v[2]; };
struct uint32x2_t  { uint32_t v[2]; };
struct float32x2_t { float    v[2]; };

struct int8x16_t   { int8_t   v[16]; };
struct uint8x16_t  { uint8_t  v[16]; };
struct int16x8_t   { int16_t  v[8]; };
struct uint16x8_t  { uint16_t v[8]; };
struct int32x4_t   { int32_t  v[4]; };
struct uint32x4_t  { uint32_t v[4]; };
struct float32x4_t { float    v[4]; };

struct int8x16x2_t  { int8x16_t  val[2]; };
struct uint8x16x2_t { uint8x16_t val[2]; };
struct int8x16x4_t  { int8x16_t  val[4]; };
struct uint8x16x4_t { uint8x16_t val[4]; };
struct int16x8x2_t  { int16x8_t  val[2]; };

inline int8x8_t   vld1_s8(const int8_t* p)   { int8x8_t r;   std::memcpy(r.v, p, 8);  return r; }
inline uint8x8_t  vld1_u8(const uint8_t* p)  { uint8x8_t r;  std::memcpy(r.v, p, 8);  return r; }
inline int8x16_t  vld1q_s8(const int8_t* p)  { int8x16_t r;  std::memcpy(r.v, p, 16); return r; }
inline uint8x16_t vld1q_u8(const uint8_t* p) { uint8x16_t r; std::memcpy(r.v, p, 16); return r; }
inline int16x8_t  vld1q_s16(const int16_t* p){ int16x8_t r;  std::memcpy(r.v, p, 16); return r; }
inline uint16x8_t vld1q_u16(const uint16_t* p){uint16x8_t r; std::memcpy(r.v, p, 16); return r; }
inline float32x4_t vld1q_f32(const float* p) { float32x4_t r;std::memcpy(r.v, p, 16); return r; }

inline void vst1_s8(int8_t* p, int8x8_t a)    { std::memcpy(p, a.v, 8); }
inline void vst1q_s8(int8_t* p, int8x16_t a)  { std::memcpy(p, a.v, 16); }
inline void vst1q_u8(uint8_t* p, uint8x16_t a){ std::memcpy(p, a.v, 16); }

inline int8x8_t   vdup_n_s8(int8_t x)    { int8x8_t r;   for (int i = 0; i < 8; ++i)  r.v[i] = x; return r; }
inline uint8x8_t  vdup_n_u8(uint8_t x)   { uint8x8_t r;  for (int i = 0; i < 8; ++i)  r.v[i] = x; return r; }
inline int8x16_t  vdupq_n_s8(int8_t x)   { int8x16_t r;  for (int i = 0; i < 16; ++i) r.v[i] = x; return r; }
inline uint8x16_t vdupq_n_u8(uint8_t x)  { uint8x16_t r; for (int i = 0; i < 16; ++i) r.v[i] = x; return r; }
inline int16x8_t  vdupq_n_s16(int16_t x) { int16x8_t r;  for (int i = 0; i < 8; ++i)  r.v[i] = x; return r; }
inline uint16x8_t vdupq_n_u16(uint16_t x){ uint16x8_t r; for (int i = 0; i < 8; ++i)  r.v[i] = x; return r; }
inline int32x4_t  vdupq_n_s32(int32_t x) { int32x4_t r;  for (int i = 0; i < 4; ++i)  r.v[i] = x; return r; }
inline uint32x4_t vdupq_n_u32(uint32_t x){ uint32x4_t r; for (int i = 0; i < 4; ++i)  r.v[i] = x; return r; }
inline float32x4_t vdupq_n_f32(float x)  { float32x4_t r;for (int i = 0; i < 4; ++i)  r.v[i] = x; return r; }

inline int8x16_t  vcombine_s8(int8x8_t a, int8x8_t b)   { int8x16_t r;  std::memcpy(r.v, a.v, 8); std::memcpy(r.v + 8, b.v, 8); return r; }
inline uint8x16_t vcombine_u8(uint8x8_t a, uint8x8_t b) { uint8x16_t r; std::memcpy(r.v, a.v, 8); std::memcpy(r.v + 8, b.v, 8); return r; }
inline int16x8_t  vcombine_s16(int16x4_t a, int16x4_t b){ int16x8_t r;  std::memcpy(r.v, a.v, 8); std::memcpy(r.v + 4, b.v, 8); return r; }

inline int8x8_t   vget_low_s8(int8x16_t a)    { int8x8_t r;   std::memcpy(r.v, a.v, 8);     return r; }
inline int8x8_t   vget_high_s8(int8x16_t a)   { int8x8_t r;   std::memcpy(r.v, a.v + 8, 8); return r; }
inline uint8x8_t  vget_low_u8(uint8x16_t a)   { uint8x8_t r;  std::memcpy(r.v, a.v, 8);     return r; }
inline uint8x8_t  vget_high_u8(uint8x16_t a)  { uint8x8_t r;  std::memcpy(r.v, a.v + 8, 8); return r; }
inline int16x4_t  vget_low_s16(int16x8_t a)   { int16x4_t r;  std::memcpy(r.v, a.v, 8);     return r; }
inline int16x4_t  vget_high_s16(int16x8_t a)  { int16x4_t r;  std::memcpy(r.v, a.v + 4, 8); return r; }
inline uint16x4_t vget_low_u16(uint16x8_t a)  { uint16x4_t r; std::memcpy(r.v, a.v, 8);     return r; }
inline uint16x4_t vget_high_u16(uint16x8_t a) { uint16x4_t r; std::memcpy(r.v, a.v + 4, 8); return r; }

inline int8x16_t   vreinterpretq_s8_u8(uint8x16_t a)   { int8x16_t r;   std::memcpy(&r, &a, 16); return r; }
inline uint8x16_t  vreinterpretq_u8_s8(int8x16_t a)    { uint8x16_t r;  std::memcpy(&r, &a, 16); return r; }
inline int8x16_t   vreinterpretq_s8_u32(uint32x4_t a)  { int8x16_t r;   std::memcpy(&r, &a, 16); return r; }
inline uint8x16_t  vreinterpretq_u8_u32(uint32x4_t a)  { uint8x16_t r;  std::memcpy(&r, &a, 16); return r; }
inline int16x8_t   vreinterpretq_s16_u16(uint16x8_t a) { int16x8_t r;   std::memcpy(&r, &a, 16); return r; }
inline int32x4_t   vreinterpretq_s32_u32(uint32x4_t a) { int32x4_t r;   std::memcpy(&r, &a, 16); return r; }
inline uint32x4_t  vreinterpretq_u32_f32(float32x4_t a){ uint32x4_t r;  std::memcpy(&r, &a, 16); return r; }
inline int32x4_t   vreinterpretq_s32_f32(float32x4_t a){ int32x4_t r;   std::memcpy(&r, &a, 16); return r; }
inline float32x4_t vreinterpretq_f32_u32(uint32x4_t a) { float32x4_t r; std::memcpy(&r, &a, 16); return r; }

inline uint8x8_t  vand_u8(uint8x8_t a, uint8x8_t b)    { uint8x8_t r;  for (int i = 0; i < 8; ++i)  r.v[i] = a.v[i] & b.v[i]; return r; }
inline uint8x8_t  vorr_u8(uint8x8_t a, uint8x8_t b)    { uint8x8_t r;  for (int i = 0; i < 8; ++i)  r.v[i] = a.v[i] | b.v[i]; return r; }
inline uint8x16_t vandq_u8(uint8x16_t a, uint8x16_t b) { uint8x16_t r; for (int i = 0; i < 16; ++i) r.v[i] = a.v[i] & b.v[i]; return r; }
inline uint8x16_t vorrq_u8(uint8x16_t a, uint8x16_t b) { uint8x16_t r; for (int i = 0; i < 16; ++i) r.v[i] = a.v[i] | b.v[i]; return r; }
inline int8x16_t  vorrq_s8(int8x16_t a, int8x16_t b)   { int8x16_t r;  for (int i = 0; i < 16; ++i) r.v[i] = a.v[i] | b.v[i]; return r; }
inline int32x4_t  vandq_s32(int32x4_t a, int32x4_t b)  { int32x4_t r;  for (int i = 0; i < 4; ++i)  r.v[i] = a.v[i] & b.v[i]; return r; }

inline uint8x8_t  vshl_n_u8(uint8x8_t a, int n)   { uint8x8_t r;  for (int i = 0; i < 8; ++i)  r.v[i] = (uint8_t) (a.v[i] << n); return r; }
inline uint8x8_t  vshr_n_u8(uint8x8_t a, int n)   { uint8x8_t r;  for (int i = 0; i < 8; ++i)  r.v[i] = (uint8_t) (a.v[i] >> n); return r; }
inline uint8x16_t vshlq_n_u8(uint8x16_t a, int n) { uint8x16_t r; for (int i = 0; i < 16; ++i) r.v[i] = (uint8_t) (a.v[i] << n); return r; }
inline uint8x16_t vshrq_n_u8(uint8x16_t a, int n) { uint8x16_t r; for (int i = 0; i < 16; ++i) r.v[i] = (uint8_t) (a.v[i] >> n); return r; }
inline int32x4_t  vshlq_n_s32(int32x4_t a, int n) { int32x4_t r;  for (int i = 0; i < 4; ++i)  r.v[i] = (int32_t) ((uint32_t) a.v[i] << n); return r; }
inline int32x4_t  vshrq_n_s32(int32x4_t a, int n) { int32x4_t r;  for (int i = 0; i < 4; ++i)  r.v[i] = a.v[i] >> n; return r; }
inline uint8x16_t vshlq_u8(uint8x16_t a, int8x16_t s) {
    uint8x16_t r;
    for (int i = 0; i < 16; ++i) {
        int sh = s.v[i];
        r.v[i] = sh >= 0 ? (sh < 8 ? (uint8_t) (a.v[i] << sh) : 0) : (-sh < 8 ? (uint8_t) (a.v[i] >> (-sh)) : 0);
    }
    return r;
}
inline uint32x4_t vshll_n_u16(uint16x4_t a, int n) { uint32x4_t r; for (int i = 0; i < 4; ++i) r.v[i] = (uint32_t) a.v[i] << n; return r; }

inline uint8x8_t vcnt_u8(uint8x8_t a) {
    uint8x8_t r;
    for (int i = 0; i < 8; ++i) {
        uint8_t x = a.v[i], c = 0;
        for (; x; x &= (uint8_t) (x - 1)) ++c;
        r.v[i] = c;
    }
    return r;
}

inline uint8x16_t vceqq_u8(uint8x16_t a, uint8x16_t b) { uint8x16_t r; for (int i = 0; i < 16; ++i) r.v[i] = a.v[i] == b.v[i] ? 0xFFu : 0u; return r; }
inline uint8x16_t vtstq_u8(uint8x16_t a, uint8x16_t b) { uint8x16_t r; for (int i = 0; i < 16; ++i) r.v[i] = (a.v[i] & b.v[i]) != 0 ? 0xFFu : 0u; return r; }

inline int8x16_t  vsubq_s8(int8x16_t a, int8x16_t b)   { int8x16_t r;  for (int i = 0; i < 16; ++i) r.v[i] = (int8_t) (a.v[i] - b.v[i]); return r; }
inline int8x16_t  vmulq_s8(int8x16_t a, int8x16_t b)   { int8x16_t r;  for (int i = 0; i < 16; ++i) r.v[i] = (int8_t) (a.v[i] * b.v[i]); return r; }
inline uint8x16_t vaddq_u8(uint8x16_t a, uint8x16_t b) { uint8x16_t r; for (int i = 0; i < 16; ++i) r.v[i] = (uint8_t) (a.v[i] + b.v[i]); return r; }
inline int32x4_t  vaddq_s32(int32x4_t a, int32x4_t b)  { int32x4_t r;  for (int i = 0; i < 4; ++i)  r.v[i] = a.v[i] + b.v[i]; return r; }
inline int32x4_t  vsubq_s32(int32x4_t a, int32x4_t b)  { int32x4_t r;  for (int i = 0; i < 4; ++i)  r.v[i] = a.v[i] - b.v[i]; return r; }
inline int32x4_t  vminq_s32(int32x4_t a, int32x4_t b)  { int32x4_t r;  for (int i = 0; i < 4; ++i)  r.v[i] = a.v[i] < b.v[i] ? a.v[i] : b.v[i]; return r; }
inline int32x4_t  vmaxq_s32(int32x4_t a, int32x4_t b)  { int32x4_t r;  for (int i = 0; i < 4; ++i)  r.v[i] = a.v[i] > b.v[i] ? a.v[i] : b.v[i]; return r; }
inline int32x4_t  vmlaq_s32(int32x4_t a, int32x4_t b, int32x4_t c) { int32x4_t r; for (int i = 0; i < 4; ++i) r.v[i] = a.v[i] + b.v[i] * c.v[i]; return r; }
inline int32x4_t  vpaddq_s32(int32x4_t a, int32x4_t b) { return {{a.v[0] + a.v[1], a.v[2] + a.v[3], b.v[0] + b.v[1], b.v[2] + b.v[3]}}; }
inline int16x8_t  vpaddq_s16(int16x8_t a, int16x8_t b) {
    return {{(int16_t)(a.v[0]+a.v[1]), (int16_t)(a.v[2]+a.v[3]), (int16_t)(a.v[4]+a.v[5]), (int16_t)(a.v[6]+a.v[7]),
             (int16_t)(b.v[0]+b.v[1]), (int16_t)(b.v[2]+b.v[3]), (int16_t)(b.v[4]+b.v[5]), (int16_t)(b.v[6]+b.v[7])}};
}
inline uint8x8_t vzip1_u8(uint8x8_t a, uint8x8_t b) { return {{a.v[0], b.v[0], a.v[1], b.v[1], a.v[2], b.v[2], a.v[3], b.v[3]}}; }
inline uint8x8_t vzip2_u8(uint8x8_t a, uint8x8_t b) { return {{a.v[4], b.v[4], a.v[5], b.v[5], a.v[6], b.v[6], a.v[7], b.v[7]}}; }

inline int16x8_t  vmovl_s8(int8x8_t a)    { int16x8_t r;  for (int i = 0; i < 8; ++i) r.v[i] = a.v[i]; return r; }
inline uint16x8_t vmovl_u8(uint8x8_t a)   { uint16x8_t r; for (int i = 0; i < 8; ++i) r.v[i] = a.v[i]; return r; }
inline uint32x4_t vmovl_u16(uint16x4_t a) { uint32x4_t r; for (int i = 0; i < 4; ++i) r.v[i] = a.v[i]; return r; }
inline int32x4_t  vmull_s16(int16x4_t a, int16x4_t b) { int32x4_t r; for (int i = 0; i < 4; ++i) r.v[i] = (int32_t) a.v[i] * (int32_t) b.v[i]; return r; }

inline int16x4_t vmovn_s32(int32x4_t a)  { int16x4_t r; for (int i = 0; i < 4; ++i) r.v[i] = (int16_t) a.v[i]; return r; }
inline int8x8_t  vmovn_s16(int16x8_t a)  { int8x8_t r;  for (int i = 0; i < 8; ++i) r.v[i] = (int8_t) a.v[i]; return r; }
inline int16x4_t vqmovn_s32(int32x4_t a) { int16x4_t r; for (int i = 0; i < 4; ++i) r.v[i] = a.v[i] > 32767 ? 32767 : (a.v[i] < -32768 ? -32768 : (int16_t) a.v[i]); return r; }
inline int8x8_t  vqmovn_s16(int16x8_t a) { int8x8_t r;  for (int i = 0; i < 8; ++i) r.v[i] = a.v[i] > 127 ? 127 : (a.v[i] < -128 ? -128 : (int8_t) a.v[i]); return r; }

inline int8x16_t  vqtbl1q_s8(int8x16_t t, uint8x16_t idx)  { int8x16_t r;  for (int i = 0; i < 16; ++i) r.v[i] = idx.v[i] < 16 ? t.v[idx.v[i]] : 0; return r; }
inline uint8x16_t vqtbl1q_u8(uint8x16_t t, uint8x16_t idx) { uint8x16_t r; for (int i = 0; i < 16; ++i) r.v[i] = idx.v[i] < 16 ? t.v[idx.v[i]] : 0; return r; }

inline int32x4_t vdotq_s32(int32x4_t acc, int8x16_t a, int8x16_t b) {
    int32x4_t r = acc;
    for (int k = 0; k < 4; ++k)
        for (int j = 0; j < 4; ++j)
            r.v[k] += (int32_t) a.v[4 * k + j] * (int32_t) b.v[4 * k + j];
    return r;
}

inline float32x4_t vabsq_f32(float32x4_t a) {
    float32x4_t r;
    for (int i = 0; i < 4; ++i) {
        uint32_t u;
        std::memcpy(&u, &a.v[i], 4);
        u &= 0x7fffffffU;
        std::memcpy(&r.v[i], &u, 4);
    }
    return r;
}
inline float32x4_t vaddq_f32(float32x4_t a, float32x4_t b) { float32x4_t r; for (int i = 0; i < 4; ++i) r.v[i] = a.v[i] + b.v[i]; return r; }
inline float32x4_t vmulq_f32(float32x4_t a, float32x4_t b) { float32x4_t r; for (int i = 0; i < 4; ++i) r.v[i] = a.v[i] * b.v[i]; return r; }
inline float32x4_t vmulq_n_f32(float32x4_t a, float b)     { float32x4_t r; for (int i = 0; i < 4; ++i) r.v[i] = a.v[i] * b; return r; }
inline float32x4_t vfmaq_f32(float32x4_t a, float32x4_t b, float32x4_t c) { float32x4_t r; for (int i = 0; i < 4; ++i) r.v[i] = std::fma(b.v[i], c.v[i], a.v[i]); return r; }
inline float32x4_t vmlaq_n_f32(float32x4_t a, float32x4_t b, float c)     { float32x4_t r; for (int i = 0; i < 4; ++i) r.v[i] = std::fma(b.v[i], c, a.v[i]); return r; }
inline float32x4_t vcvtq_f32_s32(int32x4_t a) { float32x4_t r; for (int i = 0; i < 4; ++i) r.v[i] = (float) a.v[i]; return r; }
inline int32x4_t vcvtaq_s32_f32(float32x4_t a) {
    int32x4_t r;
    for (int i = 0; i < 4; ++i) {
        float t = a.v[i];
        r.v[i] = (int32_t) (t + (t >= 0.f ? 0.5f : -0.5f));
    }
    return r;
}

inline int32_t vaddvq_s32(int32x4_t a)  { return a.v[0] + a.v[1] + a.v[2] + a.v[3]; }
inline float   vaddvq_f32(float32x4_t a){ return (a.v[0] + a.v[2]) + (a.v[1] + a.v[3]); }
inline float   vmaxvq_f32(float32x4_t a){
    float m = 0.f;
    for (int i = 0; i < 4; ++i) if (a.v[i] > m) m = a.v[i];
    return m;
}
inline float32x4_t vmaxnmq_f32(float32x4_t a, float32x4_t b) {
    float32x4_t r;
    for (int i = 0; i < 4; ++i) r.v[i] = (a.v[i] > b.v[i]) ? a.v[i] : b.v[i];
    return r;
}
inline float vmaxnmvq_f32(float32x4_t a) {
    float m = 0.f;
    for (int i = 0; i < 4; ++i) if (a.v[i] > m) m = a.v[i];
    return m;
}
inline uint32x4_t vceqq_f32(float32x4_t a, float32x4_t b) {
    uint32x4_t r;
    for (int i = 0; i < 4; ++i) r.v[i] = (a.v[i] == b.v[i]) ? 0xFFFFFFFFu : 0u;
    return r;
}
inline uint32_t vmaxvq_u32(uint32x4_t a) {
    uint32_t m = a.v[0];
    for (int i = 1; i < 4; ++i) if (a.v[i] > m) m = a.v[i];
    return m;
}
