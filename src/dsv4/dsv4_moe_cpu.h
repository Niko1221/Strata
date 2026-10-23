// ---------------------------------------------------------------------------
// dsv4_moe_cpu.h - host (CPU) AVX-512 implementation of the NVFP4 expert GEMV.
//
// BIT-EXACT BY CONSTRUCTION vs the GPU kernels in dsv4_moe.cu:
//   * dsv4_nvfp4_gemv_kernel          (R=1 path,   src/dsv4/dsv4_moe.cu:291-328)
//   * ..._grouped_kernel / ..._reuse_kernel (same per-lane expression, :340-421 / :437-514)
//
// The GPU assigns one 64-value weight block (one "lane") per thread and accumulates that
// lane's 64 products into ONE float register in a fixed order; the block's four UE4M3
// sub-block scales are applied per product, the final tail scale s2 once at the very end.
// Bit-exactness therefore requires (a) the same per-lane serial order and (b) the same
// warp-reduction shape:
//
//     acc = 0
//     for s in 0..3:                      // 4 sub-blocks of 16 values
//         dv = ue4m3(blk[s])
//         for j in 0..7:                  // low nibble of byte j, then high nibble of byte j
//             acc += LUT[lo(byte_sj)] * dv * x[s*16 + j]
//             acc += LUT[hi(byte_sj)] * dv * x[s*16 + 8 + j]
//     // warp tree: off=16,8,4,2,1  (off=16 => lane i += lane i+16)
//     total = tpr==64 ? warp0 + warp1 : warp0
//     y[row] = total * s2
//
// This file reproduces that chain with 16 independent lanes per AVX-512 register: vector
// lane i holds the accumulator of weight block (row, lane_i), so every vector operation is
// elementwise over 16 *independent* serial chains - no cross-lane reassociation happens
// anywhere, and the only cross-lane step is the explicit final tree, which is executed in
// the same order and grouping as __shfl_down_sync.
//
// Usage (Stage 1):  one call per (expert, role), rows split across threads freely, each
// thread passing its own [r0,r1) range and its own xT scratch (cols floats, 64 B aligned).
// ---------------------------------------------------------------------------
#ifndef STRATA_DSV4_MOE_CPU_H
#define STRATA_DSV4_MOE_CPU_H

#include <stdint.h>
#include <string.h>
#include <math.h>
#include <cmath>
#include <algorithm>
#include <atomic>
#include <condition_variable>
#include <memory>
#include <mutex>
#include <pthread.h>
#include <sched.h>
#include <thread>
#include <vector>

#if defined(__x86_64__) || defined(_M_X64) || defined(__i386__)
#include <immintrin.h>
#define STRATA_DSV4_CPU_X86 1
#endif

namespace strata {
namespace dsv4 {

// nvcc's default for `acc += a * dv * x` (see dsv4_moe.cu:314-315) is a fused multiply-add
// of the already-rounded (LUT*dv) product.  Variant 0 reproduces that, variant 1 keeps the
// separate round-to-nearest multiply + add.  Both are implemented so the choice can be
// verified against the GPU instead of assumed; the bit-exact test reports which one matches.
enum : int { kCpuVariantFma = 0, kCpuVariantMulAdd = 1 };

// The 16 FP4 magnitudes, identical (including the negative zero at index 8) to kFP4Lut in
// dsv4_moe.cu:77-78 and to dsv4_fp4_dec() there.
static const float kCpuFP4Lut[16] = {0.0f,  0.5f,  1.0f,  1.5f,  2.0f,  3.0f,  4.0f,  6.0f,
                                     -0.0f, -0.5f, -1.0f, -1.5f, -2.0f, -3.0f, -4.0f, -6.0f};

// ---------------------------------------------------------------------------
// One NVFP4 block (36 B: 4 UE4M3 sub-block scales + 32 B of FP4) x 64 activations.
// Scalars are used only by the non-AVX512 reference path and by the unit tests.
// ---------------------------------------------------------------------------
static inline float dsv4_cpu_ue4m3(uint8_t x) {
    if (x == 0 || x == 0x7f) return 0.0f;
    const int e = (x >> 3) & 0xf, m = x & 7;
    if (e == 0) return std::ldexp((float) m, -9);
    return std::ldexp(1.0f + (float) m / 8.0f, e - 7);
}

// p = LUT[nib] * dv, rounded once; then acc = fma(p, xv, acc) or acc + p*xv (round each step).
// fmaf() carries a guaranteed single-rounding contract, and the volatile store pins the mul+add
// variant so that no compiler (with or without -ffp-contract) can fuse it into an fma.
static inline float dsv4_cpu_step(int variant, float acc, float p, float xv) {
    if (variant == kCpuVariantFma) return fmaf(p, xv, acc);
    volatile float t = p * xv;
    return acc + t;
}

static inline float dsv4_cpu_block_ref(const uint8_t* blk, const float* xb, int variant) {
    const float d[4] = {dsv4_cpu_ue4m3(blk[0]), dsv4_cpu_ue4m3(blk[1]), dsv4_cpu_ue4m3(blk[2]),
                        dsv4_cpu_ue4m3(blk[3])};
    const uint8_t* qs = blk + 4;
    float acc = 0.0f;
    for (int s = 0; s < 4; ++s) {
        const float dv = d[s];
        for (int j = 0; j < 8; ++j) {
            const uint8_t byte = qs[s * 8 + j];
            acc = dsv4_cpu_step(variant, acc, kCpuFP4Lut[byte & 0x0f] * dv, xb[s * 16 + j]);
            acc = dsv4_cpu_step(variant, acc, kCpuFP4Lut[byte >> 4] * dv, xb[s * 16 + 8 + j]);
        }
    }
    return acc;
}

// ---------------------------------------------------------------------------
// Job description.  Same fields (and same meaning) as the GPU kernel arguments, plus the
// role/tail offsets the grouped launch applies to its blob pointer.
//   w       = blob + role_off                         (this role's weight plane)
//   s2      = tail_off ? ((float*)(blob + tail_off))[s2i] : 1.0f
//   x row m = x + (xidx ? xidx[m] : m) * xstride       (natural layout, cols floats)
//   y row m = y + m * ystride
// ---------------------------------------------------------------------------
struct CpuGemvJob {
    const uint8_t* blob = nullptr;
    int64_t role_off = 0;
    int64_t tail_off = 0;
    int32_t s2i = 0;
    const float* x = nullptr;
    int64_t xstride = 0;
    const int32_t* xidx = nullptr;
    int nent = 1;
    int cols = 0;
    int rows = 0;
    float* y = nullptr;
    int64_t ystride = 0;
    int variant = kCpuVariantFma;
    int tile = 0;   // 0 = auto (row-tiled kernel), 1 = force the one-row-at-a-time kernel
};

// ---------------------------------------------------------------------------
// Capability check: AVX-512 F+DQ+BW+VL (available on Skylake-X / i9-7980XE) and FMA3.
// The CPU arm must stay disabled when this returns false (the scalar path is far too slow
// to be useful and is only a correctness reference).
// ---------------------------------------------------------------------------
static inline bool dsv4_cpu_gemv_supported() {
#if defined(STRATA_DSV4_CPU_X86)
    return __builtin_cpu_supports("avx512f") && __builtin_cpu_supports("avx512dq") &&
           __builtin_cpu_supports("avx512bw") && __builtin_cpu_supports("avx512vl") &&
           __builtin_cpu_supports("fma") && __builtin_cpu_supports("avx2");
#else
    return false;
#endif
}

// ---------------------------------------------------------------------------
// Scalar reference: same math, any cols, no vector requirements.  Bit-exact with the GPU
// kernel (verified by the Stage-0 test) but ~20x slower; used for cols < 1024 and for tests.
// ---------------------------------------------------------------------------
inline void dsv4_cpu_gemv_rows_ref(const CpuGemvJob& job, int r0, int r1, int variant) {
    const int tpr = job.cols >> 6;
    const uint8_t* w = job.blob + job.role_off;
    const float s2 = job.tail_off ? ((const float*) (job.blob + job.tail_off))[job.s2i] : 1.0f;
    for (int m = 0; m < job.nent; ++m) {
        const float* xr = job.x + (int64_t) (job.xidx ? job.xidx[m] : m) * job.xstride;
        float* yr = job.y + (int64_t) m * job.ystride;
        for (int row = r0; row < r1; ++row) {
            const uint8_t* rb = w + (int64_t) row * tpr * 36;
            // Per-lane partial sums, then the exact __shfl_down_sync tree over the tpr lanes.
            float part[2] = {0.f, 0.f};
            // A warp's tree: off=16 folds lane i (0..15) with lane i+16, i.e. it sums the two
            // halves of the warp in lane order; the remaining offsets fold halves of 16 lanes.
            float warp[2] = {0.f, 0.f};
            for (int wp = 0; wp < (tpr == 64 ? 2 : 1); ++wp) {
                float lanes[32];
                for (int l = 0; l < 32; ++l) {
                    const int lane = wp * 32 + l;
                    lanes[l] = lane < tpr ? dsv4_cpu_block_ref(rb + (int64_t) lane * 36, xr + lane * 64, variant)
                                          : 0.0f;
                }
                for (int off = 16; off > 0; off >>= 1) {
                    for (int i = 0; i < off; ++i) lanes[i] += lanes[i + off];
                }
                warp[wp] = lanes[0];
            }
            part[0] = warp[0];
            if (tpr == 64) part[1] = warp[1];
            float total = part[0];
            if (tpr == 64) total += part[1];
            yr[row] = total * s2;
        }
    }
}

#if defined(STRATA_DSV4_CPU_X86)
#define STRATA_DSV4_TGT512 __attribute__((target("avx512f,avx512dq,avx512bw,avx512vl,avx2,fma")))

// Transpose 8 lanes x 8 dwords: on entry base + i*stride points at lane i's first dword;
// on exit T[v] holds dword v of lanes 0..7.  Classic 3-stage 8x8 dword transpose, 24 ops.
STRATA_DSV4_TGT512 static inline void dsv4_cpu_tr8x8(const uint8_t* base, int stride, __m256i T[8]) {
    __m256i r[8];
    for (int i = 0; i < 8; ++i) r[i] = _mm256_loadu_si256((const __m256i*) (base + (int64_t) i * stride));
    const __m256i t0 = _mm256_unpacklo_epi32(r[0], r[1]);
    const __m256i t1 = _mm256_unpackhi_epi32(r[0], r[1]);
    const __m256i t2 = _mm256_unpacklo_epi32(r[2], r[3]);
    const __m256i t3 = _mm256_unpackhi_epi32(r[2], r[3]);
    const __m256i t4 = _mm256_unpacklo_epi32(r[4], r[5]);
    const __m256i t5 = _mm256_unpackhi_epi32(r[4], r[5]);
    const __m256i t6 = _mm256_unpacklo_epi32(r[6], r[7]);
    const __m256i t7 = _mm256_unpackhi_epi32(r[6], r[7]);
    const __m256i s0 = _mm256_unpacklo_epi64(t0, t2), s1 = _mm256_unpackhi_epi64(t0, t2);
    const __m256i s2 = _mm256_unpacklo_epi64(t1, t3), s3 = _mm256_unpackhi_epi64(t1, t3);
    const __m256i s4 = _mm256_unpacklo_epi64(t4, t6), s5 = _mm256_unpackhi_epi64(t4, t6);
    const __m256i s6 = _mm256_unpacklo_epi64(t5, t7), s7 = _mm256_unpackhi_epi64(t5, t7);
    T[0] = _mm256_permute2x128_si256(s0, s4, 0x20);
    T[1] = _mm256_permute2x128_si256(s1, s5, 0x20);
    T[2] = _mm256_permute2x128_si256(s2, s6, 0x20);
    T[3] = _mm256_permute2x128_si256(s3, s7, 0x20);
    T[4] = _mm256_permute2x128_si256(s0, s4, 0x31);
    T[5] = _mm256_permute2x128_si256(s1, s5, 0x31);
    T[6] = _mm256_permute2x128_si256(s2, s6, 0x31);
    T[7] = _mm256_permute2x128_si256(s3, s7, 0x31);
}

// 16 lanes x 8 dwords -> V[v][lane] = dword v of lane  (`stride` bytes per lane, dwords at +4).
STRATA_DSV4_TGT512 static inline void dsv4_cpu_tr16x8(const uint8_t* base, int stride, __m512i V[8]) {
    __m256i lo[8], hi[8];
    dsv4_cpu_tr8x8(base, stride, lo);
    dsv4_cpu_tr8x8(base + (int64_t) 8 * stride, stride, hi);
    for (int v = 0; v < 8; ++v) V[v] = _mm512_inserti64x4(_mm512_castsi256_si512(lo[v]), hi[v], 1);
}

// Vector UE4M3 decode for the 16 lanes' sub-block scale s.
//   bits = (b << 20) + 0x3C000000 == ((e+120)<<23)|(m<<20)  for e>0   (b = (e<<3)|m)
//   e==0 -> m * 2^-9 ;  b==0x7f -> 0 ;  b==0 is already 0 through the e==0 path.
STRATA_DSV4_TGT512 static inline __m512 dsv4_cpu_ue4m3_vec(__m512i w, int s) {
    const __m512i b = _mm512_and_si512(_mm512_srli_epi32(w, s * 8), _mm512_set1_epi32(0xff));
    const __m512i bits = _mm512_add_epi32(_mm512_slli_epi32(b, 20), _mm512_set1_epi32(0x3C000000));
    __m512 v = _mm512_castsi512_ps(bits);
    const __mmask16 e0 = _mm512_cmplt_epu32_mask(b, _mm512_set1_epi32(8));
    const __m512 v0 = _mm512_mul_ps(_mm512_cvtepi32_ps(b), _mm512_set1_ps(0x1p-9f));
    v = _mm512_mask_mov_ps(v, e0, v0);
    const __mmask16 nan = _mm512_cmpeq_epi32_mask(b, _mm512_set1_epi32(0x7f));
    return _mm512_maskz_mov_ps((__mmask16) ~nan, v);
}

// 16-lane fold used for off=8,4,2,1: exactly __shfl_down_sync on a 16-lane vector
// (lane i += lane i+8, then i+=i+4, i+=i+2, i+=i+1), returning lane 0.
STRATA_DSV4_TGT512 static inline float dsv4_cpu_fold16(__m512 t) {
    float out;
    const __m256 s8 = _mm256_add_ps(_mm512_castps512_ps256(t), _mm512_extractf32x8_ps(t, 1));
    const __m128 s4 = _mm_add_ps(_mm256_castps256_ps128(s8), _mm256_extractf128_ps(s8, 1));
    const __m128 s2 = _mm_add_ps(s4, _mm_movehl_ps(s4, s4));
    const __m128 s1 = _mm_add_ss(s2, _mm_shuffle_ps(s2, s2, 0x01));
    _mm_store_ss(&out, s1);
    return out;
}


// Per-entry activation transpose: xT[(g*64 + k)*16 + i] = x[(xidx?xidx[m]:m)][(g*16+i)*64 + k].
// 16 lanes hold 16 different 64-value blocks, so a K step is one 64 B load.  Done once per entry
// and reused by every row of the projection.
STRATA_DSV4_TGT512 static inline void dsv4_cpu_xpose(const CpuGemvJob& job, int m, int ng, float* xT) {
    const float* xrow = job.x + (int64_t) (job.xidx ? job.xidx[m] : m) * job.xstride;
    for (int g = 0; g < ng; ++g) {
        float* dst = xT + (int64_t) g * 64 * 16;
        for (int k = 0; k < 64; ++k) {
            const float* src = xrow + k;
            for (int i = 0; i < 16; ++i) dst[k * 16 + i] = src[(int64_t) (g * 16 + i) * 64];
        }
    }
}

// ---------------------------------------------------------------------------
// The AVX-512 GEMV: rows [r0,r1) of one role, all job.nent entries.
//   xT    scratch of cols floats, 64 B aligned: xT[(g*64 + k)*16 + i] = x[(g*16+i)*64 + k]
//         (the per-entry transpose that turns "lane L owns 64 consecutive x values" into
//         "16 lanes' value k of a sub-block are one contiguous vector", amortised over rows)
// Requires (cols>>6) >= 16 and a multiple of 16; otherwise the caller must use the scalar path.
// ---------------------------------------------------------------------------
STRATA_DSV4_TGT512 static inline void dsv4_cpu_gemv_rows_avx512_v1(const CpuGemvJob& job, int r0, int r1,
                                                                  float* xT) {
    const int variant = job.variant;
    const int tpr = job.cols >> 6;         // 64 or 32 for the real model
    const int ng = tpr >> 4;               // 4 or 2 groups of 16 lanes
    const uint8_t* w = job.blob + job.role_off;
    const float s2 = job.tail_off ? ((const float*) (job.blob + job.tail_off))[job.s2i] : 1.0f;
    const __m512 LUT = _mm512_setr_ps(0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f, -0.0f, -0.5f,
                                      -1.0f, -1.5f, -2.0f, -3.0f, -4.0f, -6.0f);
    const float* xrow = nullptr;
    for (int m = 0; m < job.nent; ++m) {
        xrow = job.x + (int64_t) (job.xidx ? job.xidx[m] : m) * job.xstride;
        // Transpose this entry's activation row into the lane-major layout.
        for (int g = 0; g < ng; ++g) {
            float* dst = xT + (int64_t) g * 64 * 16;
            for (int k = 0; k < 64; ++k) {
                const float* src = xrow + k;
                for (int i = 0; i < 16; ++i) dst[k * 16 + i] = src[(int64_t) (g * 16 + i) * 64];
            }
        }
        float* yr = job.y + (int64_t) m * job.ystride;
        for (int row = r0; row < r1; ++row) {
            const uint8_t* rb = w + (int64_t) row * tpr * 36;
            __m512 av[4];
            for (int g = 0; g < ng; ++g) {
                const uint8_t* gb = rb + (int64_t) g * 16 * 36;
                __m512i V[8];
                dsv4_cpu_tr16x8(gb + 4, 36, V);
                uint32_t sc[16];
                for (int i = 0; i < 16; ++i) memcpy(&sc[i], gb + (int64_t) i * 36, 4);
                const __m512i sw = _mm512_loadu_si512((const void*) sc);
                __m512 dv[4];
                for (int s = 0; s < 4; ++s) dv[s] = dsv4_cpu_ue4m3_vec(sw, s);
                __m512 a = _mm512_setzero_ps();
                const float* xg = xT + (int64_t) g * 64 * 16;
                for (int v = 0; v < 8; ++v) {
                    const int s = v >> 1;
                    const __m512 d = dv[s];
                    for (int c = 0; c < 4; ++c) {
                        const int j = (v & 1) * 4 + c;
                        // low nibble -> x[s*16 + j], high nibble -> x[s*16 + 8 + j]
                        const __m512 xlo = _mm512_load_ps(xg + (int64_t) (s * 16 + j) * 16);
                        const __m512 xhi = _mm512_load_ps(xg + (int64_t) (s * 16 + 8 + j) * 16);
                        const __m512 plo = _mm512_mul_ps(_mm512_permutexvar_ps(_mm512_srli_epi32(V[v], c * 8), LUT), d);
                        a = (variant == kCpuVariantFma) ? _mm512_fmadd_ps(plo, xlo, a)
                                                        : _mm512_add_ps(a, _mm512_mul_ps(plo, xlo));
                        const __m512 phi = _mm512_mul_ps(
                            _mm512_permutexvar_ps(_mm512_srli_epi32(V[v], c * 8 + 4), LUT), d);
                        a = (variant == kCpuVariantFma) ? _mm512_fmadd_ps(phi, xhi, a)
                                                        : _mm512_add_ps(a, _mm512_mul_ps(phi, xhi));
                    }
                }
                av[g] = a;
            }
            // Warp tree: tpr==64 has two warps of 16 blocks each; tpr==32 has one.
            const float warp0 = dsv4_cpu_fold16(_mm512_add_ps(av[0], av[1]));
            float total = warp0;
            if (tpr == 64) {
                const float warp1 = dsv4_cpu_fold16(_mm512_add_ps(av[2], av[3]));
                total = warp0 + warp1;   // part[0] += part[1]
            }
            yr[row] = total * s2;
        }
    }
}
#endif  // STRATA_DSV4_CPU_X86


// ---------------------------------------------------------------------------
// Row-tiled form: R rows share one pass over the K dimension.  Nothing about the per-(row,block)
// accumulation changes - each of the R rows keeps its own accumulator vector and its own
// v,c iteration order - but R independent fmadd chains now overlap, which is what the serial
// 64-step chain per lane cannot do on its own (fma latency 4, one chain = 4 cycles per step).
// The transposed weight dwords go through a small L1 scratch buffer instead of 32 live zmm
// registers; that costs 8 stores + 8 loads per (row, group) and frees the whole register file.
// ---------------------------------------------------------------------------
template <int R>
STRATA_DSV4_TGT512 static inline void dsv4_cpu_gemv_tile(const uint8_t* w, int tpr, int ng, const float* xT,
                                                         float* yr, int row0, float s2) {
    alignas(64) float VT[R][8][16];
    alignas(64) float DV[R][4][16];
    alignas(64) float PART[R][4][16];
    const __m512 LUT = _mm512_setr_ps(0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f, -0.0f, -0.5f,
                                      -1.0f, -1.5f, -2.0f, -3.0f, -4.0f, -6.0f);
    for (int g = 0; g < ng; ++g) {
        // stage + transpose this tile's weight dwords and decode its 4 sub-block scales
        for (int r = 0; r < R; ++r) {
            const uint8_t* gb = w + (int64_t) (row0 + r) * tpr * 36 + (int64_t) g * 16 * 36;
            __m512i V[8];
            dsv4_cpu_tr16x8(gb + 4, 36, V);
            for (int v = 0; v < 8; ++v) _mm512_store_ps(VT[r][v], _mm512_castsi512_ps(V[v]));
            uint32_t sc[16];
            for (int i = 0; i < 16; ++i) memcpy(&sc[i], gb + (int64_t) i * 36, 4);
            const __m512i sw = _mm512_loadu_si512((const void*) sc);
            for (int s = 0; s < 4; ++s) _mm512_store_ps(DV[r][s], dsv4_cpu_ue4m3_vec(sw, s));
        }
        __m512 a[R];
        for (int r = 0; r < R; ++r) a[r] = _mm512_setzero_ps();
        const float* xg = xT + (int64_t) g * 64 * 16;
        for (int v = 0; v < 8; ++v) {
            const int s = v >> 1;
            __m512i Vr[R];
            __m512 Dr[R];
            for (int r = 0; r < R; ++r) {
                Vr[r] = _mm512_load_si512((const void*) VT[r][v]);
                Dr[r] = _mm512_load_ps(DV[r][s]);
            }
            for (int c = 0; c < 4; ++c) {
                const int j = (v & 1) * 4 + c;
                // low nibble -> x[s*16 + j]  (same value for every row of the tile)
                const __m512 xlo = _mm512_load_ps(xg + (int64_t) (s * 16 + j) * 16);
                const __m512 xhi = _mm512_load_ps(xg + (int64_t) (s * 16 + 8 + j) * 16);
                for (int r = 0; r < R; ++r) {
                    const __m512 plo =
                        _mm512_mul_ps(_mm512_permutexvar_ps(_mm512_srli_epi32(Vr[r], c * 8), LUT), Dr[r]);
                    a[r] = _mm512_fmadd_ps(plo, xlo, a[r]);
                    const __m512 phi =
                        _mm512_mul_ps(_mm512_permutexvar_ps(_mm512_srli_epi32(Vr[r], c * 8 + 4), LUT), Dr[r]);
                    a[r] = _mm512_fmadd_ps(phi, xhi, a[r]);
                }
            }
        }
        for (int r = 0; r < R; ++r) _mm512_store_ps(PART[r][g], a[r]);
    }
    // exact warp tree per row (off=16 folds the two 16-block groups, then off=8,4,2,1)
    for (int r = 0; r < R; ++r) {
        const float warp0 = dsv4_cpu_fold16(_mm512_add_ps(_mm512_load_ps(PART[r][0]), _mm512_load_ps(PART[r][1])));
        float total = warp0;
        if (tpr == 64)
            total = warp0 + dsv4_cpu_fold16(_mm512_add_ps(_mm512_load_ps(PART[r][2]), _mm512_load_ps(PART[r][3])));
        yr[row0 + r] = total * s2;
    }
}

// Row-tiled dispatch: full R=4 tiles take the ILP kernel, the <4 row tail reuses it with R=1
// (identical arithmetic, just one chain at a time).
STRATA_DSV4_TGT512 static inline void dsv4_cpu_gemv_rows_avx512_v2(const CpuGemvJob& job, int r0, int r1,
                                                                  float* xT) {
    const int tpr = job.cols >> 6;
    const int ng = tpr >> 4;
    const uint8_t* w = job.blob + job.role_off;
    const float s2 = job.tail_off ? ((const float*) (job.blob + job.tail_off))[job.s2i] : 1.0f;
    for (int m = 0; m < job.nent; ++m) {
        dsv4_cpu_xpose(job, m, ng, xT);
        float* yr = job.y + (int64_t) m * job.ystride;
        int row = r0;
        for (; row + 4 <= r1; row += 4) dsv4_cpu_gemv_tile<4>(w, tpr, ng, xT, yr, row, s2);
        for (; row < r1; ++row) dsv4_cpu_gemv_tile<1>(w, tpr, ng, xT, yr, row, s2);
    }
}

// True when the AVX-512 path can run this job (else the scalar reference must be used).
// cols 4096 (tpr=64, the gate/up projection) and cols 2048 (tpr=32, the down projection) are
// exactly the two shapes the model uses; anything else takes the scalar path so that this
// kernel never has to guess the word/warp layout of a shape it was not written for.
static inline bool dsv4_cpu_gemv_avx512_ok(const CpuGemvJob& job) {
#if defined(STRATA_DSV4_CPU_X86)
    return job.cols == 4096 || job.cols == 2048;
#else
    (void) job;
    return false;
#endif
}

// Scratch size (floats) a thread needs for one entry.
static inline int dsv4_cpu_gemv_scratch_floats(const CpuGemvJob& job) { return job.cols; }

// Single-threaded dispatch over rows [r0,r1).
inline void dsv4_cpu_gemv_rows(const CpuGemvJob& job, int r0, int r1, float* xT) {
#if defined(STRATA_DSV4_CPU_X86)
    if (dsv4_cpu_gemv_avx512_ok(job)) {
        // The tiled kernel is FMA-only (the mul+add form is kept purely as the negative control that
        // proves nvcc contracted the GPU expression); that arm therefore runs the v1 kernel.
        if (job.tile == 1 || job.variant != kCpuVariantFma) dsv4_cpu_gemv_rows_avx512_v1(job, r0, r1, xT);
        else                                                dsv4_cpu_gemv_rows_avx512_v2(job, r0, r1, xT);
        return;
    }
#endif
    (void) xT;
    dsv4_cpu_gemv_rows_ref(job, r0, r1, job.variant);
}

// ---------------------------------------------------------------------------
// Small persistent worker pool (Stage 1 needs one pool for the whole prefill; spawning
// std::threads per call would cost more than the work).  Static fan-out over thread ids.
// ---------------------------------------------------------------------------
struct CpuSink {
    virtual ~CpuSink() {}
    virtual void run(int tid) = 0;
};

class CpuPool {
  public:
    // pin: bind worker t to logical CPU `pin_base + t` so that N workers occupy N *physical* cores.
    // On this host logical CPUs i and i+18 are SMT siblings; an unpinned 18-thread run lands on a
    // sibling pair often enough to be bimodal (63-71 vs 40-42 GMAC/s), so Stage 1 must pin.
    explicit CpuPool(int n, bool pin = false, int pin_base = 1) : n_(n < 1 ? 1 : n), pin_(pin), pin_base_(pin_base) {
        workers_.reserve((size_t) (n_ - 1));
        for (int t = 1; t < n_; ++t) workers_.emplace_back([this, t] { worker(t); });
    }
    ~CpuPool() {
        {
            std::lock_guard<std::mutex> lk(m_);
            stop_ = true;
            ++gen_;
        }
        cv_.notify_all();
        for (auto& w : workers_) w.join();
    }
    int size() const { return n_; }

    // Runs sink->run(tid) for tid in [0,n) and returns when every id finished.  tid 0 runs on
    // the calling thread.  A nullptr sink is a no-op.
    void run(CpuSink* sink) {
        if (!sink) return;
        if (n_ == 1) {
            sink->run(0);
            return;
        }
        // Wake the workers FIRST, then do tid 0's share on the calling thread: running tid 0 to
        // completion before releasing the pool would serialise one full chunk ahead of the rest
        // (measured: that alone kept a 4-thread run at single-thread speed).
        {
            std::lock_guard<std::mutex> lk(m_);
            sink_ = sink;
            done_ = 0;
            ++gen_;
        }
        cv_.notify_all();
        sink->run(0);
        std::unique_lock<std::mutex> lk(m_);
        cv_done_.wait(lk, [this] { return done_ == n_ - 1; });
        sink_ = nullptr;
    }

  private:
    void worker(int tid) {
        if (pin_) {
            cpu_set_t cs;
            CPU_ZERO(&cs);
            CPU_SET(pin_base_ + tid, &cs);
            pthread_setaffinity_np(pthread_self(), sizeof(cs), &cs);
        }
        int seen = 0;
        for (;;) {
            CpuSink* s = nullptr;
            {
                std::unique_lock<std::mutex> lk(m_);
                cv_.wait(lk, [&] { return gen_ != seen; });
                seen = gen_;
                if (stop_) return;
                s = sink_;
            }
            s->run(tid);
            {
                std::lock_guard<std::mutex> lk(m_);
                if (++done_ == n_ - 1) cv_done_.notify_one();
            }
        }
    }
    int n_;
    bool pin_ = false;
    int pin_base_ = 1;
    std::vector<std::thread> workers_;
    std::mutex m_;
    std::condition_variable cv_, cv_done_;
    CpuSink* sink_ = nullptr;
    int gen_ = 0, done_ = 0;
    bool stop_ = false;
};

// Fan (m, row-range) pairs out over the pool: per entry the rows are split evenly across the
// threads, so each thread owns one contiguous [r0,r1) range of every entry.
struct CpuGemvSink : CpuSink {
    const CpuGemvJob* job;
    int nthr;
    float* xT;  // nthr * cols floats, each thread's scratch base is xT + tid*cols
    void run(int tid) override {
        const int rows = job->rows;
        const int64_t lo = (int64_t) rows * tid / nthr;
        const int64_t hi = (int64_t) rows * (tid + 1) / nthr;
        if (hi <= lo) return;
        dsv4_cpu_gemv_rows(*job, (int) lo, (int) hi, xT + (int64_t) tid * job->cols);
    }
};

// Whole-role multi-threaded entry point.  `scratch` must hold nthr*cols floats, 64 B aligned.
// Returns false (and does nothing) when out of range / unsupported, so callers keep control.
inline bool dsv4_cpu_gemv(const CpuGemvJob& job, int nthr, float* scratch, CpuPool* pool) {
    if (job.rows <= 0 || job.cols <= 0 || job.nent <= 0) return false;
    if (nthr < 1) nthr = 1;
    if (job.rows < nthr) nthr = job.rows;
    CpuGemvSink sink;
    sink.job = &job;
    sink.nthr = nthr;
    sink.xT = scratch;
    if (nthr == 1 || pool == nullptr) {
        if (nthr != 1) {
            // No pool supplied: run the per-thread ranges back to back on this thread.
            for (int t = 0; t < nthr; ++t) sink.run(t);
        } else {
            sink.run(0);
        }
        return true;
    }
    pool->run(&sink);
    return true;
}

// ---------------------------------------------------------------------------
// [S11] Batch sink: one pool barrier for a whole ROLE, i.e. for every expert the CPU arm owns in
// this layer.  A layer's CPU slice is ~32 experts x 3 roles; a barrier per expert (67 us at 18
// threads) would cost ~6 ms/layer, which is the same order as the work it schedules.  The jobs'
// row spaces are concatenated and thread `tid` owns one contiguous slice of the concatenation, so
// the split is still static (no work stealing, no atomics) and each thread's slice covers at most
// two jobs.  Every job's arithmetic is unchanged: the same row range of the same job.
struct CpuGemvBatchSink : CpuSink {
    const CpuGemvJob* jobs = nullptr;
    int njobs = 0;
    int nthr = 1;
    int total = 0;      // sum of jobs[].rows
    int stride = 0;     // scratch floats per thread (>= every job's cols)
    float* xT = nullptr;
    void run(int tid) override {
        const int64_t lo = (int64_t) total * tid / nthr;
        const int64_t hi = (int64_t) total * (tid + 1) / nthr;
        if (hi <= lo) return;
        float* sc = xT + (int64_t) tid * stride;
        int base = 0;
        for (int j = 0; j < njobs; ++j) {
            const int r = jobs[j].rows;
            const int a = (int) (std::max<int64_t>(lo, base) - base);
            const int b = (int) (std::min<int64_t>(hi, base + r) - base);
            if (b > a) dsv4_cpu_gemv_rows(jobs[j], a, b, sc);
            base += r;
            if (base >= hi) break;
        }
    }
};

// Multi-threaded GEMV over a batch of jobs (one barrier).  `scratch` holds nthr*stride floats,
// 64 B aligned, stride >= max(jobs[].cols).  Returns the number of rows scheduled.
inline int64_t dsv4_cpu_gemv_batch(const CpuGemvJob* jobs, int njobs, int nthr, int stride, float* scratch,
                                   CpuPool* pool) {
    if (jobs == nullptr || njobs <= 0 || nthr < 1 || scratch == nullptr) return 0;
    int64_t total = 0;
    for (int j = 0; j < njobs; ++j) {
        if (jobs[j].rows > 0 && jobs[j].cols > 0 && jobs[j].nent > 0) total += jobs[j].rows;
    }
    if (total == 0) return 0;
    if ((int64_t) nthr > total) nthr = (int) total;
    CpuGemvBatchSink sink;
    sink.jobs = jobs;
    sink.njobs = njobs;
    sink.nthr = nthr;
    sink.total = (int) total;
    sink.stride = stride;
    sink.xT = scratch;
    if (nthr == 1 || pool == nullptr) {
        for (int t = 0; t < nthr; ++t) sink.run(t);
    } else {
        pool->run(&sink);
    }
    return total;
}

}  // namespace dsv4
}  // namespace strata

#endif  // STRATA_DSV4_MOE_CPU_H
