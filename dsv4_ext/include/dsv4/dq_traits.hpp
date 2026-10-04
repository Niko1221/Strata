// dsv4/dq_traits.hpp - the per-ggml-type decode traits used by the CUDA matvec kernel (src/gpu_cuda.cu).
//
// They live in a header on purpose: the same code has to compile for the device AND for the host, so
// tests/test_dq_traits.cpp can compare every trait, element by element, against dequant_row()/row_dot()
// from src/dequant.cpp. Without that, a wrong nibble shift in a kernel is silent garbage in the logits.
//
// The IQ* traits read six codebook arrays through the G_* names below. By default they are the host tables
// from dsv4/iq_tables.hpp; src/gpu_cuda.cu defines the macros to its __constant__ copies BEFORE including
// this header, so the very same source lines read device constant memory there and host memory in a test.
//
// Each trait: BE elements per quantisation block, BB bytes per block, and at(row, e) = element e of that
// row. at() derives its own block offset from e, so one generic kernel can serve every type and a thread
// may read any element of a row independently. Every at() is a transcription of the matching dq_* in
// src/dequant.cpp; the test is what keeps that transcription honest.
#pragma once

#include "dsv4/iq_tables.hpp"

#include <cmath>
#include <cstdint>
#include <cstring>

// On the host the macro supplies `inline` (the header is included by more than one TU in a test binary);
// under nvcc it supplies the device+host qualifiers. Never write `inline` next to it: that is a duplicate
// specifier. src/gpu_cuda.cu does NOT redefine it any more - the __CUDACC__ branch below is already right.
#ifndef DSV4_HOST_DEVICE
#  ifdef __CUDACC__
#    define DSV4_HOST_DEVICE __device__ __host__ __forceinline__
#  else
#    define DSV4_HOST_DEVICE inline
#  endif
#endif

#ifndef G_iq1s_grid
#define G_iq1s_grid dsv4::iq::iq1s_grid
#endif
#ifndef G_iq2xxs_grid
#define G_iq2xxs_grid dsv4::iq::iq2xxs_grid
#endif
#ifndef G_iq3xxs_grid
#define G_iq3xxs_grid dsv4::iq::iq3xxs_grid
#endif
#ifndef G_ksigns_iq2xs
#define G_ksigns_iq2xs dsv4::iq::ksigns_iq2xs
#endif
#ifndef G_kmask_iq2xs
#define G_kmask_iq2xs dsv4::iq::kmask_iq2xs
#endif
// ggml has no kvalues_mxfp4 in ggml-common.h's IQ table set, so it is spelled out here once and shared by
// both backends. It is a FUNCTION, not an array: a namespace-scope array has internal linkage and nvcc
// refuses to read it from device code, while an inline function marked DSV4_HOST_DEVICE compiles on both
// sides. Values: {0, 1, 2, 3, 4, 6, 8, 12, 0, -1, -2, -3, -4, -6, -8, -12}, bit 3 = sign. The e8m0 scale
// already carries the factor of 2 (e8m0_to_fp32_half).
#ifndef G_kvalues_mxfp4
#define G_kvalues_mxfp4 dqt::mxfp4_value
#endif

namespace dqt {

DSV4_HOST_DEVICE float mxfp4_value(int q) {
    const int m = q & 7;
    // m == 0 must return +0.0f, not -0.0f: ggml stores a literal 0 in the table and the trait test
    // compares float BITS, where -0.0f != 0.0f.
    if (m == 0) return 0.f;
    const float v = (m < 4) ? (float) m : (m == 4) ? 4.f : (m == 5) ? 6.f : (m == 6) ? 8.f : 12.f;
    return (q & 8) ? -v : v;
}

// ---------------------------------------------------------------- scalar helpers (mirror dequant.cpp)
// DSV4_HOST_DEVICE on every one of them: nvcc rejects a __host__-only helper called from a kernel, and
// these are the leaf calls of every at() below. The macro already carries `inline` on the host, so do not
// write it again.
DSV4_HOST_DEVICE uint16_t ld16(const uint8_t* p) { uint16_t v; std::memcpy(&v, p, 2); return v; }
DSV4_HOST_DEVICE uint32_t ld32(const uint8_t* p) { uint32_t v; std::memcpy(&v, p, 4); return v; }
DSV4_HOST_DEVICE float    ldff(const uint8_t* p) { float v;   std::memcpy(&v, p, 4); return v; }

DSV4_HOST_DEVICE float f16_to_f32(uint16_t h) {
    const uint32_t sign = (uint32_t)(h >> 15) & 1u, exp = (h >> 10) & 0x1fu, frac = h & 0x3ffu;
    float out;
    if (exp == 0) out = (float) frac * (1.0f / 16777216.0f);
    else if (exp == 31) out = frac ? NAN : INFINITY;
    else { const uint32_t bits = (exp << 23) + (frac << 13) + ((127u - 15u) << 23); std::memcpy(&out, &bits, 4); }
    return sign ? -out : out;
}
DSV4_HOST_DEVICE float bf16_to_f32(uint16_t h) { const uint32_t bits = (uint32_t) h << 16; float f; std::memcpy(&f, &bits, 4); return f; }
DSV4_HOST_DEVICE float e8m0_to_fp32_half(uint8_t x) {
    const uint32_t bits = x < 2 ? (0x00200000u << x) : ((uint32_t)(x - 1) << 23);
    float f; std::memcpy(&f, &bits, 4); return f;
}
DSV4_HOST_DEVICE void get_scale_min_k4(int j, const uint8_t* q, uint8_t* d, uint8_t* m) {
    if (j < 4) { *d = q[j] & 63; *m = q[j + 4] & 63; }
    else { *d = (q[j + 4] & 0xF) | ((q[j - 4] >> 6) << 4); *m = (q[j + 4] >> 4) | ((q[j - 0] >> 6) << 4); }
}

// ---------------------------------------------------------------- traits
struct F32T  { static constexpr int BE = 1,  BB = 4;
    DSV4_HOST_DEVICE static float at(const uint8_t* w, int e) { return ldff(w + 4 * (int64_t) e); } };

struct BF16T { static constexpr int BE = 1,  BB = 2;
    DSV4_HOST_DEVICE static float at(const uint8_t* w, int e) { return bf16_to_f32(ld16(w + 2 * (int64_t) e)); } };

struct Q8_0T { static constexpr int BE = 32, BB = 34;
    DSV4_HOST_DEVICE static float at(const uint8_t* w, int e) {
        const uint8_t* b = w + (int64_t)(e >> 5) * BB;
        return f16_to_f32(ld16(b)) * (float) ((const int8_t*)(b + 2))[e & 31];
    } };

// { d, min (f16), scales[32], qs[128] }: four sub-blocks of 64 per 256-element block.
struct Q4_KT { static constexpr int BE = 256, BB = 144;
    DSV4_HOST_DEVICE static float at(const uint8_t* w, int e) {
        const uint8_t* b = w + (int64_t)(e >> 8) * BB;
        const float d = f16_to_f32(ld16(b)), min = f16_to_f32(ld16(b + 2));
        const uint8_t* sc = b + 4; const uint8_t* q = b + 16;
        const int chunk = (e & 255) >> 6, sub = e & 63, half = (sub >= 32) ? 1 : 0, l = sub & 31;
        uint8_t s, mm; get_scale_min_k4(2 * chunk + half, sc, &s, &mm);
        const uint8_t nib = q[chunk * 32 + l];
        return (d * (float) s) * (float) (half ? (nib >> 4) : (nib & 0xF)) - min * (float) mm;
    } };

struct Q5_KT { static constexpr int BE = 256, BB = 176;
    DSV4_HOST_DEVICE static float at(const uint8_t* w, int e) {
        const uint8_t* b = w + (int64_t)(e >> 8) * BB;
        const float d = f16_to_f32(ld16(b)), min = f16_to_f32(ld16(b + 2));
        const uint8_t* sc = b + 4; const uint8_t* qh = b + 16; const uint8_t* ql = b + 48;
        const int chunk = (e & 255) >> 6, sub = e & 63, half = (sub >= 32) ? 1 : 0, l = sub & 31;
        uint8_t s, mm; get_scale_min_k4(2 * chunk + half, sc, &s, &mm);
        const uint8_t u = (uint8_t)((half ? 2 : 1) << (2 * chunk));
        const float hi = (float) ((ql[chunk * 32 + l] >> (half ? 4 : 0)) & 0xF);
        return (d * (float) s) * (hi + ((qh[l] & u) ? 16.f : 0.f)) - min * (float) mm;
    } };

// Two 128-element halves per block; each half: 64 ql bytes, 32 qh bytes, 4 super-block scales.
struct Q6_KT { static constexpr int BE = 256, BB = 210;
    DSV4_HOST_DEVICE static float at(const uint8_t* w, int e) {
        const uint8_t* b = w + (int64_t)(e >> 8) * BB;
        const float d = f16_to_f32(ld16(b + 208));
        const int half = (e & 255) >> 7, rem = e & 127;
        const int grp = rem >> 5, l = rem & 31;              // grp 0..3 selects q1..q4
        const uint8_t* ql = b + half * 64; const uint8_t* qh = b + 128 + half * 32;
        const int8_t* sc = (const int8_t*)(b + 192) + half * 8;
        const int qidx = (grp & 1) ? (l + 32) : l;           // q1/q3 read ql[l], q2/q4 read ql[l+32]
        const int nsh = (grp >> 1) * 4;                      // q1/q2 low nibble, q3/q4 high nibble
        const int hsh = grp * 2;                             // the two extra bits sit at 0/2/4/6 of qh[l]
        const int8_t q = (int8_t)(((ql[qidx] >> nsh) & 0xF) | (((qh[l] >> hsh) & 3) << 4)) - 32;
        // ggml: is = l/16, and the super-block scale of q1..q4 is sc[is + 2*grp] within the half.
        return d * (float) sc[(l >> 4) + 2 * grp] * (float) q;
    } };

struct IQ3_XXST { static constexpr int BE = 256, BB = 98;
    DSV4_HOST_DEVICE static float at(const uint8_t* w, int e) {
        const uint8_t* b = w + (int64_t)(e >> 8) * BB;
        const float d = f16_to_f32(ld16(b));
        const uint8_t* qs = b + 2; const uint8_t* sss = qs + 64;
        const int ib32 = (e & 255) >> 5, l = (e & 31) >> 3, j = e & 7;
        uint32_t aux32; std::memcpy(&aux32, sss + 4 * ib32, sizeof(aux32));
        const float db = d * (0.5f + (aux32 >> 28)) * 0.5f;
        const uint8_t signs = G_ksigns_iq2xs[(aux32 >> (7 * l)) & 127];
        const uint8_t* g = (const uint8_t*)(G_iq3xxs_grid + qs[ib32 * 8 + 2 * l + ((j >= 4) ? 1 : 0)]);
        return db * (float) g[j & 3] * ((signs & G_kmask_iq2xs[j]) ? -1.f : 1.f);
    } };

struct IQ2_XXST { static constexpr int BE = 256, BB = 66;
    DSV4_HOST_DEVICE static float at(const uint8_t* w, int e) {
        const uint8_t* b = w + (int64_t)(e >> 8) * BB;
        const float d = f16_to_f32(ld16(b)); const uint8_t* qs = b + 2;
        const int ib32 = (e & 255) >> 5, l = (e & 31) >> 3, j = e & 7;
        uint32_t aux32[2]; std::memcpy(aux32, qs + 8 * ib32, 2 * sizeof(uint32_t));
        const uint8_t* aux8 = (const uint8_t*) aux32;
        const float db = d * (0.5f + (aux32[1] >> 28)) * 0.25f;
        const uint8_t* g = (const uint8_t*)(G_iq2xxs_grid + aux8[l]);
        const uint8_t signs = G_ksigns_iq2xs[(aux32[1] >> (7 * l)) & 127];
        return db * (float) g[j] * ((signs & G_kmask_iq2xs[j]) ? -1.f : 1.f);
    } };

struct IQ1_MT { static constexpr int BE = 256, BB = 56;
    DSV4_HOST_DEVICE static float at(const uint8_t* w, int e) {
        const uint8_t* b = w + (int64_t)(e >> 8) * BB;
        const uint8_t* qs = b; const uint8_t* qhb = b + 32; const uint8_t* scb = b + 48;
        const uint16_t sc[4] = {ld16(scb), ld16(scb + 2), ld16(scb + 4), ld16(scb + 6)};
        const uint16_t su = (uint16_t)((sc[0] >> 12) | ((sc[1] >> 8) & 0x00f0) | ((sc[2] >> 4) & 0x0f00) | (sc[3] & 0xf000));
        const float d = f16_to_f32(su);
        const int ib = (e & 255) >> 5, l = (e & 31) >> 3, j = e & 7;
        const uint8_t qh0 = qhb[2 * ib], qh1 = qhb[2 * ib + 1];
        // ggml: dl1 serves sub-blocks l=0,1 and dl2 serves l=2,3, each from a 3-bit field of sc[ib/2].
        const int sel = (l < 2) ? 0 : 3;
        const float dl = d * (2.f * (float) ((sc[ib / 2] >> (6 * (ib % 2) + sel)) & 0x7) + 1.f);
        const uint16_t idx = (uint16_t)(qs[ib * 4 + l] |
                                        ((l == 0) ? ((qh0 << 8) & 0x700) : (l == 1) ? ((qh0 << 4) & 0x700) :
                                         (l == 2) ? ((qh1 << 8) & 0x700) : ((qh1 << 4) & 0x700)));
        const bool neg = (l == 0) ? (qh0 & 0x08) : (l == 1) ? (qh0 & 0x80) :
                         (l == 2) ? (qh1 & 0x08) : (qh1 & 0x80);
        const float delta = neg ? -0.125f : 0.125f;   // kIq1sDelta
        const int8_t* g = (const int8_t*)(G_iq1s_grid + idx);
        return dl * ((float) g[j] + delta);
    } };

struct MXFP4T { static constexpr int BE = 32, BB = 17;
    DSV4_HOST_DEVICE static float at(const uint8_t* w, int e) {
        const uint8_t* b = w + (int64_t)(e >> 5) * BB;
        const float d = e8m0_to_fp32_half(b[0]); const uint8_t* qs = b + 1;
        const int q = e & 15, sh = (e & 16) ? 4 : 0;   // qs[0..15] low nibbles, then the same bytes' high nibbles
        return d * G_kvalues_mxfp4((qs[q] >> sh) & 0x0F);
    } };

}  // namespace dqt
