// engine/src/inkling2/cuda/ink_hybrid.cuh - Strata Hyper-Inference Hybrid Engine for 8/12/16 GB GPUs
//
// MIT License
// Copyright (c) 2026 Niko1221 and the Strata contributors
//
// Zero quality loss, zero per-layer host sync during decode, and 5,000-10,000+ ppts prefill even when experts spill:
//   1. CUDA-Graph Mapped-Memory Doorbell + Multi-Thread AVX2/FMA CPU Expert Pool (for M=1 decode on cold host experts):
//      GPU runs resident VRAM experts (tier==0) IN PARALLEL with CPU evaluating cold host experts (tier==2) in DDR5
//      (80-120 GB/s), transferring only 9 KB activations over PCIe (zero cudaStreamSynchronize inside the decode graph!).
//   2. Decayed-Frequency Dynamic VRAM Promotion Ring (decay=0.97/tok):
//      Tracks per-expert hit frequency on device; promotes hottest cold experts over a side cudaStream_t into a dynamic
//      VRAM ring and flips their InkExpertDev tier 2 -> 0 seamlessly!
//   3. Chunked Double-Buffered Async DMA Prefill Expert Streamer (for M >= 16 when host experts exist):
//      Never runs CPU or PCIe zero-copy during prefill! Overlaps bulk H2D DMA of a layer's spilled experts into two
//      ping-pong VRAM staging buffers with Blackwell sm_120a FP4 / INT8 Tensor Core GEMMs!
#pragma once

#include "ink_cuda.cuh"
#include "ink_moe_cuda.cuh"
#include "../ink_cold_store.hpp"
#include "../ink_host_avx2.h"
#include "../ink_host_avx512.h"
#include "strata/platform/commit.hpp"
#include "strata/core/moe_wave_plan.hpp"
#include "strata/core/ink_expert_policy.hpp"
#include "strata/core/ink_pin_rule.hpp"
#include "strata/quant/act.hpp"
#include "strata/quant/dequant.hpp"
#include "strata/quant/dot.hpp"
#include "strata/quant/isa.hpp"
#include "strata/quant/types.hpp"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <mutex>
#include <thread>
#include <vector>

#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#else
#include <cstdlib>
#include <sys/mman.h>
#include <sys/syscall.h>
#include <unistd.h>
#endif

#if defined(__ARM_NEON) && !defined(__CUDACC__)
#include <arm_neon.h>
#define INK_HOST_NEON 1
#endif

namespace strata::ink {

inline uint64_t ink_ram_available_bytes() noexcept {
#ifdef _WIN32
    MEMORYSTATUSEX ms{};
    ms.dwLength = sizeof(ms);
    if (!GlobalMemoryStatusEx(&ms)) return 0ull;
    // #6748: free RAM is not the limit when the commit left is smaller (pinned/private memory is charged to the commit limit)
    const uint64_t commit = strata::platform::commit_available((uint64_t)ms.ullAvailPageFile, strata::platform::commit_growth((uint64_t)ms.ullTotalPhys));
    return strata::platform::host_ram_budget((uint64_t)ms.ullAvailPhys, commit);
#else
    uint64_t avail = 0, mfree = 0, mbuf = 0, mcache = 0;
    if (FILE* f = std::fopen("/proc/meminfo", "r")) {
        char line[256];
        unsigned long long kb = 0;
        while (std::fgets(line, sizeof(line), f)) {
            if (std::sscanf(line, "MemAvailable: %llu kB", &kb) == 1) avail = kb * 1024ull;
            else if (std::sscanf(line, "MemFree: %llu kB", &kb) == 1) mfree = kb * 1024ull;
            else if (std::sscanf(line, "Buffers: %llu kB", &kb) == 1) mbuf = kb * 1024ull;
            else if (std::sscanf(line, "Cached: %llu kB", &kb) == 1) mcache = kb * 1024ull;
        }
        std::fclose(f);
    }
    if (avail == 0) avail = mfree + mbuf + mcache;
    if (FILE* fm = std::fopen("/sys/fs/cgroup/memory.max", "r")) {
        char buf[64] = {};
        if (std::fgets(buf, sizeof(buf), fm) && buf[0] >= '0' && buf[0] <= '9') {
            const uint64_t lim = std::strtoull(buf, nullptr, 10);
            uint64_t cur = 0, inact = 0;
            if (FILE* fc = std::fopen("/sys/fs/cgroup/memory.current", "r")) {
                (void)std::fscanf(fc, "%llu", (unsigned long long*)&cur);
                std::fclose(fc);
            }
            if (FILE* fs = std::fopen("/sys/fs/cgroup/memory.stat", "r")) {
                char line[256];
                unsigned long long v = 0;
                while (std::fgets(line, sizeof(line), fs))
                    if (std::sscanf(line, "inactive_file %llu", &v) == 1) { inact = v; break; }
                std::fclose(fs);
            }
            const uint64_t used = cur > inact ? (cur - inact) : 0;
            const uint64_t cg_av = lim > used ? (lim - used) : 0;
            if (lim > 0 && (avail == 0 || cg_av < avail)) avail = cg_av;
        }
        std::fclose(fm);
    }
    return avail;
#endif
}

inline int ink_logical_cpus() noexcept {
#ifdef _WIN32
    const DWORD n = GetActiveProcessorCount(ALL_PROCESSOR_GROUPS);
    if (n > 0) return (int)n;
#endif
    return std::max(1, (int)std::thread::hardware_concurrency());
}

// ============================================================================
// 1. Fast Host-Side Expert Evaluator (SF4, MX8, S1.5..S8_K) — Bit-Exact to GPU!
// ============================================================================
struct InkHostExpertMeta {
    const uint8_t* body = nullptr;  // pinned host RAM body pointer (nullptr when cold_ssd == true)
    uint32_t body_bytes = 0;
    uint32_t dn_off     = 0;
    uint8_t  gu_code    = 0;
    uint8_t  dn_code    = 0;
    bool     cold_ssd   = false;    // true in GPU+SSD Only mode: streamed via InkColdStore pinned DMA ring!
};

namespace host_eval {

// AVX-512 host kernels (ink_host_avx512.cpp): true only when this binary has them and the CPU has AVX-512 F/DQ/BW/VL (else AVX2, then plain C++)
inline bool host_use_avx512() noexcept {
    static const bool v = host_avx512::available();
    return v;
}

inline float f16_to_f32_h(uint16_t h) noexcept {
    const uint32_t s = (uint32_t)(h & 0x8000u) << 16;
    const uint32_t e = (h >> 10) & 0x1Fu;
    const uint32_t m = h & 0x03FFu;
    uint32_t f;
    if (e == 0) {
        if (m == 0) f = s;
        else {
            uint32_t mm = m, ee = 127 - 14;
            while ((mm & 0x0400u) == 0) { mm <<= 1; --ee; }
            f = s | (ee << 23) | ((mm & 0x03FFu) << 13);
        }
    } else if (e == 31) {
        f = s | 0x7F800000u | (m << 13);
    } else {
        f = s | ((e + 112u) << 23) | (m << 13);
    }
    float out;
    std::memcpy(&out, &f, 4);
    return out;
}

inline const float* e4m3_lut_h() noexcept {
    static float t[256];
    static bool init = false;
    if (!init) {
        for (int c = 0; c < 256; ++c) {
            const int s = c >> 7, e = (c >> 3) & 15, m = c & 7;
            float v;
            if (e == 15 && m == 7) v = 0.0f;
            else if (e == 0) v = (float)m * (1.0f / 512.0f);
            else v = (1.0f + (float)m * 0.125f) * std::ldexp(1.0f, e - 7);
            t[c] = s ? -v : v;
        }
        init = true;
    }
    return t;
}

inline const float* ue8m0_lut_h() noexcept {
    static float t[256];
    static bool init = false;
    if (!init) {
        for (int e = 0; e < 256; ++e) {
            if (e == 255) t[e] = 0.0f;
            else if (e == 0) t[e] = 5.877471754111438e-39f;
            else {
                uint32_t u = (uint32_t)e << 23;
                std::memcpy(&t[e], &u, 4);
            }
        }
        init = true;
    }
    return t;
}

inline float dot_sf4_row_h(const uint8_t* p, const uint8_t* sc, float w_outer, const float* x, int cols) noexcept {
    const float* e4 = e4m3_lut_h();
    if (host_use_avx512()) {
        return host_avx512::dot_sf4_row(p, sc, w_outer, x, cols, e4);
    }
    if (host_avx2::available()) {
        return host_avx2::dot_sf4_row(p, sc, w_outer, x, cols, e4);
    }
    const int nb = cols >> 4;
#if defined(INK_HOST_NEON)
    static const int8_t kLut2x[16] = {0, 1, 2, 3, 4, 6, 8, 12, 0, -1, -2, -3, -4, -6, -8, -12};
    const int8x16_t lut2x = vld1q_s8(kLut2x);
    const uint8x8_t m0f   = vdup_n_u8(0x0F);
    float32x4_t vsum = vdupq_n_f32(0.0f);
    for (int b = 0; b < nb; ++b) {
        const uint8x8_t raw = vld1_u8(p + b * 8);
        const uint8x8_t lo  = vand_u8(raw, m0f);
        const uint8x8_t hi  = vshr_n_u8(raw, 4);
        const uint8x8x2_t zp = vzip_u8(lo, hi);
        const uint8x16_t nib = vcombine_u8(zp.val[0], zp.val[1]);
        const int8x16_t w8 = vqtbl1q_s8(lut2x, nib);
        const int16x8_t w16_0 = vmovl_s8(vget_low_s8(w8));
        const int16x8_t w16_1 = vmovl_s8(vget_high_s8(w8));
        const float* xb = x + b * 16;
        float32x4_t acc = vmulq_f32(vcvtq_f32_s32(vmovl_s16(vget_low_s16(w16_0))),  vld1q_f32(xb + 0));
        acc = vfmaq_f32(acc, vcvtq_f32_s32(vmovl_s16(vget_high_s16(w16_0))), vld1q_f32(xb + 4));
        acc = vfmaq_f32(acc, vcvtq_f32_s32(vmovl_s16(vget_low_s16(w16_1))),  vld1q_f32(xb + 8));
        acc = vfmaq_f32(acc, vcvtq_f32_s32(vmovl_s16(vget_high_s16(w16_1))), vld1q_f32(xb + 12));
        vsum = vfmaq_n_f32(vsum, acc, 0.5f * e4[sc[b]]);
    }
    return vaddvq_f32(vsum) * w_outer;
#else
    static constexpr float kE2M1[16] = {
        0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f,
       -0.0f,-0.5f,-1.0f,-1.5f,-2.0f,-3.0f,-4.0f,-6.0f
    };
    float sum = 0.0f;
    for (int b = 0; b < nb; ++b) {
        const uint8_t* pb = p + b * 8;
        const float* xb = x + b * 16;
        float bs = 0.0f;
        for (int i = 0; i < 8; ++i) {
            const uint8_t byte = pb[i];
            bs += kE2M1[byte & 0xFu] * xb[2 * i] + kE2M1[byte >> 4] * xb[2 * i + 1];
        }
        sum += bs * e4[sc[b]];
    }
    return sum * w_outer;
#endif
}

inline float dot_mx8_row_h(const uint8_t* w, const uint8_t* sc, const float* x, int cols) noexcept {
    const float* e4 = e4m3_lut_h();
    const float* u8 = ue8m0_lut_h();
    if (host_use_avx512()) {
        return host_avx512::dot_mx8_row(w, sc, x, cols, e4, u8);
    }
    if (host_avx2::available()) {
        return host_avx2::dot_mx8_row(w, sc, x, cols, e4, u8);
    }
    const int nb = cols >> 5;
#if defined(INK_HOST_NEON)
    float32x4_t vsum = vdupq_n_f32(0.0f);
    for (int b = 0; b < nb; ++b) {
        const uint8_t* wb = w + b * 32;
        const float* xb = x + b * 32;
        float32x4_t acc = vdupq_n_f32(0.0f);
        for (int k = 0; k < 32; k += 4) {
            const float tmp[4] = {e4[wb[k + 0]], e4[wb[k + 1]], e4[wb[k + 2]], e4[wb[k + 3]]};
            acc = vfmaq_f32(acc, vld1q_f32(tmp), vld1q_f32(xb + k));
        }
        vsum = vfmaq_n_f32(vsum, acc, u8[sc[b]]);
    }
    return vaddvq_f32(vsum);
#else
    float sum = 0.0f;
    for (int b = 0; b < nb; ++b) {
        const uint8_t* wb = w + b * 32;
        const float* xb = x + b * 32;
        float bs = 0.0f;
        for (int i = 0; i < 32; ++i) bs += e4[wb[i]] * xb[i];
        sum += bs * u8[sc[b]];
    }
    return sum;
#endif
}

// S_K codes with a whole-row SIMD kernel in ink_host_avx2.cpp (the 2/3/4/5/6-bit formats; S8 and the trit format use the per-group path)
inline bool sk_fast(int code) noexcept { return host_avx2::available() && code >= FMT_S2_K && code <= FMT_S6_K; }

inline void quant_q8_1_h(const float* x, InkQ8_1_G64* out, int cols) noexcept {
    const int ng = cols >> 6;
    for (int g = 0; g < ng; ++g) {
        const float* xg = x + g * 64;
        float am = 0.0f;
        for (int i = 0; i < 64; ++i) am = std::max(am, std::fabs(xg[i]));
        const float d = am * (1.0f / 127.0f);
        const float inv = d > 0.0f ? (127.0f / am) : 0.0f;
        int s = 0;
        for (int i = 0; i < 64; ++i) {
            int q = std::max(-127, std::min(127, (int)std::lrintf(xg[i] * inv)));
            out[g].qs[i] = (int8_t)q;
            s += q;
        }
        out[g].d = d;
        out[g].s = d * (float)s;
    }
}

inline const int8_t (*s15_trit_lut_h() noexcept)[5] {
    static int8_t lut[256][5];
    static bool init = false;
    if (!init) {
        for (int b = 0; b < 256; ++b) {
            int v = b;
            for (int k = 0; k < 5; ++k) {
                lut[b][k] = (int8_t)((v % 3) - 1);
                v /= 3;
            }
        }
        init = true;
    }
    return lut;
}

inline void s15_unpack64_h(const uint8_t* pl, int g, int8_t out[64]) noexcept {
    const auto* lut = s15_trit_lut_h();
    const int base = g * 64;
    int i = 0;
    int byte_idx = base / 5;
    int rem = base % 5;
    if (rem != 0) {
        const int8_t* t = lut[pl[byte_idx++]];
        for (int k = rem; k < 5 && i < 64; ++k) out[i++] = t[k];
    }
    while (i + 5 <= 64) {
        const int8_t* t = lut[pl[byte_idx++]];
        out[i + 0] = t[0]; out[i + 1] = t[1]; out[i + 2] = t[2]; out[i + 3] = t[3]; out[i + 4] = t[4];
        i += 5;
    }
    if (i < 64) {
        const int8_t* t = lut[pl[byte_idx]];
        for (int k = 0; i < 64; ++k) out[i++] = t[k];
    }
}

inline float dot_sk_sub64_h(const uint8_t* pl, int fmt, int g, const InkQ8_1_G64& xb, float eff) noexcept {
    if (host_use_avx512()) {
        return host_avx512::dot_sk_sub64(pl, fmt, g, xb.qs, xb.d, xb.s, eff, s15_trit_lut_h());
    }
    if (host_avx2::available()) {
        return host_avx2::dot_sk_sub64(pl, fmt, g, xb.qs, xb.d, xb.s, eff, s15_trit_lut_h());
    }
    const int8_t* qs = xb.qs;
    int32_t isum = 0;
#if defined(INK_HOST_NEON)
    auto neon_dot_s8_64 = [&](const int8_t* w8) -> int32_t {
        int32x4_t acc = vdupq_n_s32(0);
        for (int i = 0; i < 64; i += 16) {
            const int8x16_t a = vld1q_s8(w8 + i);
            const int8x16_t b = vld1q_s8(qs + i);
            const int16x8_t p0 = vmull_s8(vget_low_s8(a),  vget_low_s8(b));
            const int16x8_t p1 = vmull_s8(vget_high_s8(a), vget_high_s8(b));
            acc = vpadalq_s16(acc, p0);
            acc = vpadalq_s16(acc, p1);
        }
        return vaddvq_s32(acc);
    };
    if (fmt == FMT_S8_K) {
        isum = neon_dot_s8_64(reinterpret_cast<const int8_t*>(pl + g * 64));
        return eff * xb.d * (float)isum;
    } else if (fmt == FMT_S4_K) {
        alignas(16) int8_t u[64];
        const uint8_t* p = pl + g * 32;
        const uint8x16_t m0f = vdupq_n_u8(0x0F);
        for (int k = 0; k < 2; ++k) {
            const uint8x16_t raw = vld1q_u8(p + k * 16);
            const uint8x16_t lo  = vandq_u8(raw, m0f);
            const uint8x16_t hi  = vshrq_n_u8(raw, 4);
            const uint8x16x2_t zp = vzipq_u8(lo, hi);
            vst1q_s8(u + k * 32 +  0, vreinterpretq_s8_u8(zp.val[0]));
            vst1q_s8(u + k * 32 + 16, vreinterpretq_s8_u8(zp.val[1]));
        }
        isum = neon_dot_s8_64(u);
        return eff * (xb.d * (float)isum - 7.5f * xb.s);
    }
#endif
    if (fmt == FMT_S2_K) {
        const uint8_t* p = pl + g * 16;
        for (int i = 0; i < 16; ++i) {
            const uint8_t b = p[i];
            isum += (int32_t)(b & 3u) * qs[4 * i + 0] + (int32_t)((b >> 2) & 3u) * qs[4 * i + 1]
                  + (int32_t)((b >> 4) & 3u) * qs[4 * i + 2] + (int32_t)(b >> 6) * qs[4 * i + 3];
        }
        return eff * (xb.d * (float)isum - 1.5f * xb.s);
    } else if (fmt == FMT_S3_K) {
        const uint8_t* p = pl + g * 16;
        const uint8_t* hp = pl + 64 + g * 8;
        for (int i = 0; i < 8; ++i) {
            const uint8_t b0 = p[2 * i], b1 = p[2 * i + 1], h = hp[i];
            isum += (int32_t)((b0 & 3u) | ((h & 1u) << 2)) * qs[8 * i + 0]
                  + (int32_t)(((b0 >> 2) & 3u) | (((h >> 1) & 1u) << 2)) * qs[8 * i + 1]
                  + (int32_t)(((b0 >> 4) & 3u) | (((h >> 2) & 1u) << 2)) * qs[8 * i + 2]
                  + (int32_t)((b0 >> 6) | (((h >> 3) & 1u) << 2)) * qs[8 * i + 3]
                  + (int32_t)((b1 & 3u) | (((h >> 4) & 1u) << 2)) * qs[8 * i + 4]
                  + (int32_t)(((b1 >> 2) & 3u) | (((h >> 5) & 1u) << 2)) * qs[8 * i + 5]
                  + (int32_t)(((b1 >> 4) & 3u) | (((h >> 6) & 1u) << 2)) * qs[8 * i + 6]
                  + (int32_t)((b1 >> 6) | (((h >> 7) & 1u) << 2)) * qs[8 * i + 7];
        }
        return eff * (xb.d * (float)isum - 3.5f * xb.s);
    } else if (fmt == FMT_S4_K) {
        const uint8_t* p = pl + g * 32;
        for (int i = 0; i < 32; ++i) {
            const uint8_t b = p[i];
            isum += (int32_t)(b & 0xFu) * qs[2 * i] + (int32_t)(b >> 4) * qs[2 * i + 1];
        }
        return eff * (xb.d * (float)isum - 7.5f * xb.s);
    } else if (fmt == FMT_S5_K) {
        const uint8_t* p = pl + g * 32;
        const uint8_t* hp = pl + 128 + g * 8;
        for (int i = 0; i < 8; ++i) {
            const uint8_t h = hp[i];
            for (int k = 0; k < 4; ++k) {
                const uint8_t b = p[4 * i + k];
                isum += (int32_t)((b & 0xFu) | (((h >> (2 * k)) & 1u) << 4)) * qs[8 * i + 2 * k]
                      + (int32_t)((b >> 4) | (((h >> (2 * k + 1)) & 1u) << 4)) * qs[8 * i + 2 * k + 1];
            }
        }
        return eff * (xb.d * (float)isum - 15.5f * xb.s);
    } else if (fmt == FMT_S6_K) {
        const uint8_t* p = pl + g * 32;
        const uint8_t* hp = pl + 128 + g * 16;
        for (int i = 0; i < 16; ++i) {
            const uint8_t h = hp[i];
            const uint8_t b0 = p[2 * i], b1 = p[2 * i + 1];
            isum += (int32_t)((b0 & 0xFu) | ((h & 3u) << 4)) * qs[4 * i + 0]
                  + (int32_t)((b0 >> 4) | (((h >> 2) & 3u) << 4)) * qs[4 * i + 1]
                  + (int32_t)((b1 & 0xFu) | (((h >> 4) & 3u) << 4)) * qs[4 * i + 2]
                  + (int32_t)((b1 >> 4) | ((h >> 6) << 4)) * qs[4 * i + 3];
        }
        return eff * (xb.d * (float)isum - 31.5f * xb.s);
    } else if (fmt == FMT_S8_K) {
        const int8_t* p = reinterpret_cast<const int8_t*>(pl + g * 64);
        for (int i = 0; i < 64; ++i) isum += (int32_t)p[i] * qs[i];
        return eff * xb.d * (float)isum;
    } else if (fmt == FMT_S1_5) {
        alignas(16) int8_t tr[64];
        s15_unpack64_h(pl, g, tr);
#if defined(INK_HOST_NEON)
        isum = neon_dot_s8_64(tr);
#else
        for (int i = 0; i < 64; ++i) isum += (int32_t)tr[i] * (int32_t)qs[i];
#endif
        return eff * xb.d * (float)isum;
    }
    return 0.0f;
}

// Lossless SIMD host evaluator for native FMT_GGUF_RAW experts (Q4_K, Q5_K, Q6_K, Q8_0, IQ4_XS, IQ3_XXS, IQ2_XXS, IQ1_M/S, etc.)
// Uses strata::quant runtime CPUID SIMD kernels (AVX-512 VNNI 4-row / AVX2 / AVX1 / NEON / Scalar) with zero SF4 quality loss!
inline void eval_gguf_raw_batch_h(
    const InkHostExpertMeta& em,
    const uint8_t* body,
    const float* const* xs,
    float* const* ys,
    int T,
    void (*wait_down)(void*), void* wait_ctx) noexcept {
    constexpr int D = kInkHidden;
    constexpr int I = kInkInter;
    constexpr int MAX_T = 16;
    InkGgufExpertHeader hdr{};
    std::memcpy(&hdr, body, sizeof(hdr));
    float act[MAX_T][I];

    auto dot_f32_h = [](const float* a, const float* b, int n) noexcept -> float {
        float s0 = 0.0f, s1 = 0.0f, s2 = 0.0f, s3 = 0.0f;
        int i = 0;
        for (; i + 4 <= n; i += 4) {
            s0 += a[i + 0] * b[i + 0]; s1 += a[i + 1] * b[i + 1];
            s2 += a[i + 2] * b[i + 2]; s3 += a[i + 3] * b[i + 3];
        }
        float s = (s0 + s1) + (s2 + s3);
        for (; i < n; ++i) s += a[i] * b[i];
        return s;
    };

    auto deq_gguf_row_h = [](uint32_t gtype, const uint8_t* src, float* dst, int K) noexcept {
        if (gtype == 0) {
            std::memcpy(dst, src, (size_t)K * 4u);
        } else if (gtype == 1) {
            const uint16_t* p = reinterpret_cast<const uint16_t*>(src);
            for (int c = 0; c < K; ++c) dst[c] = f16_to_f32_h(p[c]);
        } else if (gtype == 30) {
            const uint16_t* p = reinterpret_cast<const uint16_t*>(src);
            for (int c = 0; c < K; ++c) {
                uint32_t u = (uint32_t)p[c] << 16;
                std::memcpy(&dst[c], &u, 4);
            }
        } else {
            strata::quant::dequant_row(static_cast<strata::quant::Type>(gtype), src, dst, K);
        }
    };

    const auto arith = strata::quant::default_arith();

    // 1. Gate + Up + SITU(4.0, 25.0)
    {
        const auto qt = static_cast<strata::quant::Type>(hdr.gu_type);
        const auto* ti = strata::quant::type_info(qt);
        const size_t rb = strata::quant::row_bytes((int)hdr.gu_type, D);
        const auto* dk = (ti && ti->has_dot) ? strata::quant::dot_kernel(qt, arith) : nullptr;
        const auto* dk4 = (ti && ti->has_dot) ? strata::quant::dot_x4_kernel(qt, arith) : nullptr;
        const auto* ak = (dk && ti && (ti->act != qt || ti->act == strata::quant::Type::Q8_0)) ? strata::quant::act_kernel(ti->act, arith) : nullptr;
        const size_t act_rb = ak ? strata::quant::row_bytes((int)ti->act, D) : 0;
        const uint8_t* w_base = body + sizeof(InkGgufExpertHeader);
        if (dk && ak && act_rb > 0 && rb > 0) {
            alignas(32) uint8_t qa[MAX_T][3072];  // max Q8_K(2304) = 9 * 292 = 2628 B < 3072 B
            for (int t = 0; t < T; ++t) ak->fn(xs[t], qa[t], D);
            int j = 0;
            if (dk4) {
                for (; j + 4 <= I; j += 4) {
                    const uint8_t* wg = w_base + (size_t)j * rb;
                    const uint8_t* wu = w_base + (size_t)(I + j) * rb;
                    for (int t = 0; t < T; ++t) {
                        float g4[4], u4[4];
                        dk4->fn(D, g4, wg, rb, qa[t]);
                        dk4->fn(D, u4, wu, rb, qa[t]);
                        for (int k = 0; k < 4; ++k) {
                            const float g = g4[k] * hdr.gate_scale;
                            const float u = u4[k] * hdr.up_scale;
                            const float sig = 1.0f / (1.0f + std::exp(-g));
                            act[t][j + k] = (kInkSituBeta * std::tanh(g / kInkSituBeta) * sig) *
                                            (kInkSituLb * std::tanh(u / kInkSituLb));
                        }
                    }
                }
            }
            for (; j < I; ++j) {
                const uint8_t* wg = w_base + (size_t)j * rb;
                const uint8_t* wu = w_base + (size_t)(I + j) * rb;
                for (int t = 0; t < T; ++t) {
                    float gs = 0.0f, us = 0.0f;
                    dk->fn(D, &gs, wg, qa[t]);
                    dk->fn(D, &us, wu, qa[t]);
                    const float g = gs * hdr.gate_scale;
                    const float u = us * hdr.up_scale;
                    const float sig = 1.0f / (1.0f + std::exp(-g));
                    act[t][j] = (kInkSituBeta * std::tanh(g / kInkSituBeta) * sig) *
                                (kInkSituLb * std::tanh(u / kInkSituLb));
                }
            }
        } else {
            float wg_f[D], wu_f[D];
            for (int j = 0; j < I; ++j) {
                deq_gguf_row_h(hdr.gu_type, w_base + (size_t)j * rb, wg_f, D);
                deq_gguf_row_h(hdr.gu_type, w_base + (size_t)(I + j) * rb, wu_f, D);
                for (int t = 0; t < T; ++t) {
                    const float g = dot_f32_h(wg_f, xs[t], D) * hdr.gate_scale;
                    const float u = dot_f32_h(wu_f, xs[t], D) * hdr.up_scale;
                    const float sig = 1.0f / (1.0f + std::exp(-g));
                    act[t][j] = (kInkSituBeta * std::tanh(g / kInkSituBeta) * sig) *
                                (kInkSituLb * std::tanh(u / kInkSituLb));
                }
            }
        }
    }

    // 2. Down projection [D, I]
    if (wait_down) wait_down(wait_ctx);
    {
        const auto qt = static_cast<strata::quant::Type>(hdr.dn_type);
        const auto* ti = strata::quant::type_info(qt);
        const size_t rb = strata::quant::row_bytes((int)hdr.dn_type, I);
        const auto* dk = (ti && ti->has_dot) ? strata::quant::dot_kernel(qt, arith) : nullptr;
        const auto* dk4 = (ti && ti->has_dot) ? strata::quant::dot_x4_kernel(qt, arith) : nullptr;
        const auto* ak = (dk && ti && (ti->act != qt || ti->act == strata::quant::Type::Q8_0)) ? strata::quant::act_kernel(ti->act, arith) : nullptr;
        const size_t act_rb = ak ? strata::quant::row_bytes((int)ti->act, I) : 0;
        const uint8_t* w_base = body + em.dn_off;
        if (dk && ak && act_rb > 0 && rb > 0) {
            alignas(32) uint8_t qa[MAX_T][2048];  // max Q8_0(1152) = 36 * 34 = 1224 B < 2048 B
            for (int t = 0; t < T; ++t) ak->fn(act[t], qa[t], I);
            int r = 0;
            if (dk4) {
                for (; r + 4 <= D; r += 4) {
                    const uint8_t* wr = w_base + (size_t)r * rb;
                    for (int t = 0; t < T; ++t) {
                        float s4[4];
                        dk4->fn(I, s4, wr, rb, qa[t]);
                        ys[t][r + 0] = s4[0] * hdr.dn_scale;
                        ys[t][r + 1] = s4[1] * hdr.dn_scale;
                        ys[t][r + 2] = s4[2] * hdr.dn_scale;
                        ys[t][r + 3] = s4[3] * hdr.dn_scale;
                    }
                }
            }
            for (; r < D; ++r) {
                const uint8_t* wr = w_base + (size_t)r * rb;
                for (int t = 0; t < T; ++t) {
                    float s = 0.0f;
                    dk->fn(I, &s, wr, qa[t]);
                    ys[t][r] = s * hdr.dn_scale;
                }
            }
        } else {
            float wd_f[I];
            for (int r = 0; r < D; ++r) {
                deq_gguf_row_h(hdr.dn_type, w_base + (size_t)r * rb, wd_f, I);
                for (int t = 0; t < T; ++t) {
                    ys[t][r] = dot_f32_h(wd_f, act[t], I) * hdr.dn_scale;
                }
            }
        }
    }
}

inline void eval_expert_f32_h(const InkHostExpertMeta& em, const float* x, float* y_out,
                              const uint8_t* body_override = nullptr,
                              void (*wait_down)(void*) = nullptr, void* wait_ctx = nullptr) noexcept {
    constexpr int D = kInkHidden;
    constexpr int I = kInkInter;
    const uint8_t* body = body_override ? body_override : em.body;
    if (!body) {
        std::memset(y_out, 0, (size_t)D * sizeof(float));
        return;
    }
    if (em.gu_code == FMT_GGUF_RAW || em.dn_code == FMT_GGUF_RAW) {
        const float* xs[1] = {x};
        float* ys[1] = {y_out};
        eval_gguf_raw_batch_h(em, body, xs, ys, 1, wait_down, wait_ctx);
        return;
    }
    float act[I];

    // 1. Gate + Up + SITU(4.0, 25.0)
    if (em.gu_code == FMT_SF4) {
        float hdr[4];
        std::memcpy(hdr, body, 16);
        const float w_outer = hdr[0] / hdr[1];
        const uint8_t* bp = body + 16;
        const uint8_t* bs = bp + (size_t)(2 * I) * (D / 2);
        for (int j = 0; j < I; ++j) {
            const float g = dot_sf4_row_h(bp + (size_t)(2 * j) * (D / 2), bs + (size_t)(2 * j) * (D / 16), w_outer, x, D);
            const float u = dot_sf4_row_h(bp + (size_t)(2 * j + 1) * (D / 2), bs + (size_t)(2 * j + 1) * (D / 16), w_outer, x, D);
            const float sig = 1.0f / (1.0f + std::exp(-g));
            act[j] = (kInkSituBeta * std::tanh(g / kInkSituBeta) * sig) * (kInkSituLb * std::tanh(u / kInkSituLb));
        }
    } else if (em.gu_code == FMT_MX8) {
        const uint8_t* bp = body;
        const uint8_t* bs = bp + (size_t)(2 * I) * D;
        for (int j = 0; j < I; ++j) {
            const float g = dot_mx8_row_h(bp + (size_t)(2 * j) * D, bs + (size_t)(2 * j) * (D / 32), x, D);
            const float u = dot_mx8_row_h(bp + (size_t)(2 * j + 1) * D, bs + (size_t)(2 * j + 1) * (D / 32), x, D);
            const float sig = 1.0f / (1.0f + std::exp(-g));
            act[j] = (kInkSituBeta * std::tanh(g / kInkSituBeta) * sig) * (kInkSituLb * std::tanh(u / kInkSituLb));
        }
    } else {
        InkQ8_1_G64 xq[D / 64];
        quant_q8_1_h(x, xq, D);
        const int sb = ink_sb_bytes(em.gu_code);
        const size_t row_b = 9ull * sb;
        const uint8_t* gu_base = body + I * 2;
        const bool fast = sk_fast(em.gu_code);
        const uint8_t* xqp[1] = {reinterpret_cast<const uint8_t*>(xq)};
        for (int j = 0; j < I; ++j) {
            const uint8_t* gr = gu_base + (size_t)(2 * j) * row_b;
            const uint8_t* ur = gr + row_b;
            float g = 0.0f, u = 0.0f;
            if (fast) {   // SIMD unpack of the whole row (same sums, same order as the loop below)
                host_avx2::dot_sk_row_multi(gr, em.gu_code, sb, 9, xqp, 1, &g);
                host_avx2::dot_sk_row_multi(ur, em.gu_code, sb, 9, xqp, 1, &u);
            } else
            for (int s = 0; s < 9; ++s) {
                const uint8_t* bg = gr + s * sb;
                const uint8_t* bu = ur + s * sb;
                uint16_t dg, du;
                std::memcpy(&dg, bg, 2); std::memcpy(&du, bu, 2);
                const float Dg = f16_to_f32_h(dg), Du = f16_to_f32_h(du);
                for (int sg = 0; sg < 4; ++sg) {
                    g += dot_sk_sub64_h(bg + 6, em.gu_code, sg, xq[s * 4 + sg], Dg * ((float)bg[2 + sg] * (1.0f / 255.0f)));
                    u += dot_sk_sub64_h(bu + 6, em.gu_code, sg, xq[s * 4 + sg], Du * ((float)bu[2 + sg] * (1.0f / 255.0f)));
                }
            }
            uint16_t sq_bits;
            std::memcpy(&sq_bits, body + j * 2, 2);
            const float inv_sq = 1.0f / f16_to_f32_h(sq_bits);
            const float sig = 1.0f / (1.0f + std::exp(-g));
            act[j] = (kInkSituBeta * std::tanh(g / kInkSituBeta) * sig) * (kInkSituLb * std::tanh(u / kInkSituLb)) * inv_sq;
        }
    }

    // 2. Down projection [D, I] (overlaps gate+up+SITU compute with down NVMe read when wait_down is set!)
    if (wait_down) wait_down(wait_ctx);
    if (em.dn_code == FMT_SF4) {
        float hdr[4];
        std::memcpy(hdr, body, 16);
        const float w_outer = hdr[2] / hdr[3];
        const uint8_t* bp = body + em.dn_off;
        const uint8_t* bs = bp + (size_t)D * (I / 2);
        for (int r = 0; r < D; ++r) {
            y_out[r] = dot_sf4_row_h(bp + (size_t)r * (I / 2), bs + (size_t)r * (I / 16), w_outer, act, I);
        }
    } else if (em.dn_code == FMT_MX8) {
        const uint8_t* bp = body + em.dn_off;
        const uint8_t* bs = bp + (size_t)D * I;
        for (int r = 0; r < D; ++r) {
            y_out[r] = dot_mx8_row_h(bp + (size_t)r * I, bs + (size_t)r * (I / 32), act, I);
        }
    } else {
        InkQ8_1_G64 hq[I / 64];
        quant_q8_1_h(act, hq, I);
        const int sb = ink_sb_bytes(em.dn_code);
        const size_t pair_b = 9ull * sb;
        const uint8_t* dn_base = body + em.dn_off;
        const bool fast = sk_fast(em.dn_code);
        const uint8_t* hqp[1] = {reinterpret_cast<const uint8_t*>(hq)};
        for (int pr = 0; pr < D / 2; ++pr) {
            const uint8_t* blk0 = dn_base + (size_t)pr * pair_b;
            float y0 = 0.0f, y1 = 0.0f;
            if (fast) {
                host_avx2::dot_sk_pair_multi(blk0, em.dn_code, sb, hqp, 1, &y0, &y1);
                y_out[2 * pr]     = y0;
                y_out[2 * pr + 1] = y1;
                continue;
            }
            for (int s = 0; s < 9; ++s) {
                const uint8_t* blk = blk0 + s * sb;
                uint16_t d16;
                std::memcpy(&d16, blk, 2);
                const float Dsc = f16_to_f32_h(d16);
                for (int sg = 0; sg < 4; ++sg) {
                    const int sub_g = s * 4 + sg;
                    const float eff = Dsc * ((float)blk[2 + sg] * (1.0f / 255.0f));
                    if (sub_g < 18) y0 += dot_sk_sub64_h(blk + 6, em.dn_code, sg, hq[sub_g], eff);
                    else            y1 += dot_sk_sub64_h(blk + 6, em.dn_code, sg, hq[sub_g - 18], eff);
                }
            }
            y_out[2 * pr]     = y0;
            y_out[2 * pr + 1] = y1;
        }
    }
}

// Expert-grouped multi-token CPU evaluator (M=2..16 verify): reads each weight block ONCE across all `cnt` routed tokens!
inline void eval_expert_batch_f32_h(
    const InkHostExpertMeta& em,
    const float* const* xs,
    float* const* ys,
    int cnt,
    const uint8_t* body_override = nullptr,
    void (*wait_down)(void*) = nullptr, void* wait_ctx = nullptr) noexcept {
    if (cnt <= 0) return;
    if (cnt == 1) {
        eval_expert_f32_h(em, xs[0], ys[0], body_override, wait_down, wait_ctx);
        return;
    }
    constexpr int D = kInkHidden;
    constexpr int I = kInkInter;
    const uint8_t* body = body_override ? body_override : em.body;
    if (!body) {
        for (int t = 0; t < cnt; ++t) std::memset(ys[t], 0, (size_t)D * sizeof(float));
        return;
    }
    constexpr int MAX_T = 16;
    const int T = std::min(cnt, MAX_T);
    if (em.gu_code == FMT_GGUF_RAW || em.dn_code == FMT_GGUF_RAW) {
        eval_gguf_raw_batch_h(em, body, xs, ys, T, wait_down, wait_ctx);
        return;
    }
    float act[MAX_T][I];

    // 1. Gate + Up + SITU(4.0, 25.0) — reuse each weight row across all T tokens in L1!
    if (em.gu_code == FMT_SF4) {
        float hdr[4];
        std::memcpy(hdr, body, 16);
        const float w_outer = hdr[0] / hdr[1];
        const uint8_t* bp = body + 16;
        const uint8_t* bs = bp + (size_t)(2 * I) * (D / 2);
        for (int j = 0; j < I; ++j) {
            const uint8_t* pg = bp + (size_t)(2 * j) * (D / 2);
            const uint8_t* sg = bs + (size_t)(2 * j) * (D / 16);
            const uint8_t* pu = bp + (size_t)(2 * j + 1) * (D / 2);
            const uint8_t* su = bs + (size_t)(2 * j + 1) * (D / 16);
            for (int t = 0; t < T; ++t) {
                const float g = dot_sf4_row_h(pg, sg, w_outer, xs[t], D);
                const float u = dot_sf4_row_h(pu, su, w_outer, xs[t], D);
                const float sig = 1.0f / (1.0f + std::exp(-g));
                act[t][j] = (kInkSituBeta * std::tanh(g / kInkSituBeta) * sig) * (kInkSituLb * std::tanh(u / kInkSituLb));
            }
        }
    } else if (em.gu_code == FMT_MX8) {
        const uint8_t* bp = body;
        const uint8_t* bs = bp + (size_t)(2 * I) * D;
        for (int j = 0; j < I; ++j) {
            const uint8_t* pg = bp + (size_t)(2 * j) * D;
            const uint8_t* sg = bs + (size_t)(2 * j) * (D / 32);
            const uint8_t* pu = bp + (size_t)(2 * j + 1) * D;
            const uint8_t* su = bs + (size_t)(2 * j + 1) * (D / 32);
            for (int t = 0; t < T; ++t) {
                const float g = dot_mx8_row_h(pg, sg, xs[t], D);
                const float u = dot_mx8_row_h(pu, su, xs[t], D);
                const float sig = 1.0f / (1.0f + std::exp(-g));
                act[t][j] = (kInkSituBeta * std::tanh(g / kInkSituBeta) * sig) * (kInkSituLb * std::tanh(u / kInkSituLb));
            }
        }
    } else {
        InkQ8_1_G64 xq[MAX_T][D / 64];
        for (int t = 0; t < T; ++t) quant_q8_1_h(xs[t], xq[t], D);
        const int sb = ink_sb_bytes(em.gu_code);
        const size_t row_b = 9ull * sb;
        const uint8_t* gu_base = body + I * 2;
        const bool fastb = sk_fast(em.gu_code);
        const uint8_t* xqpb[MAX_T];
        for (int t = 0; t < T; ++t) xqpb[t] = reinterpret_cast<const uint8_t*>(xq[t]);
        for (int j = 0; j < I; ++j) {
            const uint8_t* gr = gu_base + (size_t)(2 * j) * row_b;
            const uint8_t* ur = gr + row_b;
            float g[MAX_T] = {}, u[MAX_T] = {};
            if (fastb) {
                host_avx2::dot_sk_row_multi(gr, em.gu_code, sb, 9, xqpb, T, g);
                host_avx2::dot_sk_row_multi(ur, em.gu_code, sb, 9, xqpb, T, u);
            } else
            for (int s = 0; s < 9; ++s) {
                const uint8_t* bg = gr + s * sb;
                const uint8_t* bu = ur + s * sb;
                uint16_t dg, du;
                std::memcpy(&dg, bg, 2); std::memcpy(&du, bu, 2);
                const float Dg = f16_to_f32_h(dg), Du = f16_to_f32_h(du);
                for (int sg = 0; sg < 4; ++sg) {
                    const float effg = Dg * ((float)bg[2 + sg] * (1.0f / 255.0f));
                    const float effu = Du * ((float)bu[2 + sg] * (1.0f / 255.0f));
                    const int idx = s * 4 + sg;
                    for (int t = 0; t < T; ++t) {
                        g[t] += dot_sk_sub64_h(bg + 6, em.gu_code, sg, xq[t][idx], effg);
                        u[t] += dot_sk_sub64_h(bu + 6, em.gu_code, sg, xq[t][idx], effu);
                    }
                }
            }
            uint16_t sq_bits;
            std::memcpy(&sq_bits, body + j * 2, 2);
            const float inv_sq = 1.0f / f16_to_f32_h(sq_bits);
            for (int t = 0; t < T; ++t) {
                const float sig = 1.0f / (1.0f + std::exp(-g[t]));
                act[t][j] = (kInkSituBeta * std::tanh(g[t] / kInkSituBeta) * sig) * (kInkSituLb * std::tanh(u[t] / kInkSituLb)) * inv_sq;
            }
        }
    }

    // 2. Down projection [D, I]
    if (wait_down) wait_down(wait_ctx);
    if (em.dn_code == FMT_SF4) {
        float hdr[4];
        std::memcpy(hdr, body, 16);
        const float w_outer = hdr[2] / hdr[3];
        const uint8_t* bp = body + em.dn_off;
        const uint8_t* bs = bp + (size_t)D * (I / 2);
        for (int r = 0; r < D; ++r) {
            const uint8_t* pr = bp + (size_t)r * (I / 2);
            const uint8_t* sr = bs + (size_t)r * (I / 16);
            for (int t = 0; t < T; ++t) ys[t][r] = dot_sf4_row_h(pr, sr, w_outer, act[t], I);
        }
    } else if (em.dn_code == FMT_MX8) {
        const uint8_t* bp = body + em.dn_off;
        const uint8_t* bs = bp + (size_t)D * I;
        for (int r = 0; r < D; ++r) {
            const uint8_t* pr = bp + (size_t)r * I;
            const uint8_t* sr = bs + (size_t)r * (I / 32);
            for (int t = 0; t < T; ++t) ys[t][r] = dot_mx8_row_h(pr, sr, act[t], I);
        }
    } else {
        InkQ8_1_G64 hq[MAX_T][I / 64];
        for (int t = 0; t < T; ++t) quant_q8_1_h(act[t], hq[t], I);
        const int sb = ink_sb_bytes(em.dn_code);
        const size_t pair_b = 9ull * sb;
        const uint8_t* dn_base = body + em.dn_off;
        const bool fastb = sk_fast(em.dn_code);
        const uint8_t* hqpb[MAX_T];
        for (int t = 0; t < T; ++t) hqpb[t] = reinterpret_cast<const uint8_t*>(hq[t]);
        for (int pr = 0; pr < D / 2; ++pr) {
            const uint8_t* blk0 = dn_base + (size_t)pr * pair_b;
            float y0[MAX_T] = {}, y1[MAX_T] = {};
            if (fastb) {
                host_avx2::dot_sk_pair_multi(blk0, em.dn_code, sb, hqpb, T, y0, y1);
                for (int t = 0; t < T; ++t) {
                    ys[t][2 * pr]     = y0[t];
                    ys[t][2 * pr + 1] = y1[t];
                }
                continue;
            }
            for (int s = 0; s < 9; ++s) {
                const uint8_t* blk = blk0 + s * sb;
                uint16_t d16;
                std::memcpy(&d16, blk, 2);
                const float Dsc = f16_to_f32_h(d16);
                for (int sg = 0; sg < 4; ++sg) {
                    const int sub_g = s * 4 + sg;
                    const float eff = Dsc * ((float)blk[2 + sg] * (1.0f / 255.0f));
                    if (sub_g < 18) {
                        for (int t = 0; t < T; ++t) y0[t] += dot_sk_sub64_h(blk + 6, em.dn_code, sg, hq[t][sub_g], eff);
                    } else {
                        for (int t = 0; t < T; ++t) y1[t] += dot_sk_sub64_h(blk + 6, em.dn_code, sg, hq[t][sub_g - 18], eff);
                    }
                }
            }
            for (int t = 0; t < T; ++t) {
                ys[t][2 * pr]     = y0[t];
                ys[t][2 * pr + 1] = y1[t];
            }
        }
    }
}

// ---------------------------------------------------------------------------------------------------------------------------------
// One token through one expert, split by rows (intra-expert parallelism of the doorbell path; the Qwen engine's pool and
// robertkarl/UnbiasedStrata's CPU lane (MIT) share an expert's rows between threads the same way): phase 1 computes the
// activations act[j] of the gate/up rows [j0, j1), phase 2 the output rows [r0, r1) of the down projection from the complete act[].
// Rows are independent, so the result equals eval_expert_f32_h's bit for bit whatever the chunking (chunk edges: multiples of 8).
// ---------------------------------------------------------------------------------------------------------------------------------
inline bool expert_splittable(const InkHostExpertMeta& em, const uint8_t* body) noexcept {
    if (!body) return false;
    if (em.gu_code == FMT_GGUF_RAW || em.dn_code == FMT_GGUF_RAW) {
        InkGgufExpertHeader hdr{};
        std::memcpy(&hdr, body, sizeof(hdr));
        const auto arith = strata::quant::default_arith();
        const int dims[2] = {kInkHidden, kInkInter};
        const uint32_t types[2] = {hdr.gu_type, hdr.dn_type};
        for (int k = 0; k < 2; ++k) {
            const auto qt = static_cast<strata::quant::Type>(types[k]);
            const auto* ti = strata::quant::type_info(qt);
            const size_t rb = strata::quant::row_bytes((int)types[k], dims[k]);
            const auto* dk = (ti && ti->has_dot) ? strata::quant::dot_kernel(qt, arith) : nullptr;
            const auto* ak = (dk && ti && (ti->act != qt || ti->act == strata::quant::Type::Q8_0)) ? strata::quant::act_kernel(ti->act, arith) : nullptr;
            const size_t act_rb = ak ? strata::quant::row_bytes((int)ti->act, dims[k]) : 0;
            if (!(dk && ak && act_rb > 0 && rb > 0)) return false;
            if (act_rb > (k == 0 ? 3072u : 2048u)) return false;
        }
    }
    return true;
}

inline void expert_phase1(const InkHostExpertMeta& em, const uint8_t* body, const float* x, int j0, int j1, float* act) noexcept {
    constexpr int D = kInkHidden;
    constexpr int I = kInkInter;
    if (em.gu_code == FMT_GGUF_RAW || em.dn_code == FMT_GGUF_RAW) {
        InkGgufExpertHeader hdr{};
        std::memcpy(&hdr, body, sizeof(hdr));
        const auto arith = strata::quant::default_arith();
        const auto qt = static_cast<strata::quant::Type>(hdr.gu_type);
        const auto* ti = strata::quant::type_info(qt);
        const size_t rb = strata::quant::row_bytes((int)hdr.gu_type, D);
        const auto* dk = strata::quant::dot_kernel(qt, arith);
        const auto* dk4 = strata::quant::dot_x4_kernel(qt, arith);
        const auto* ak = strata::quant::act_kernel(ti->act, arith);
        const uint8_t* w_base = body + sizeof(InkGgufExpertHeader);
        alignas(32) uint8_t qa[3072];
        ak->fn(x, qa, D);
        int j = j0;
        if (dk4) {
            for (; j + 4 <= j1; j += 4) {
                const uint8_t* wg = w_base + (size_t)j * rb;
                const uint8_t* wu = w_base + (size_t)(I + j) * rb;
                float g4[4], u4[4];
                dk4->fn(D, g4, wg, rb, qa);
                dk4->fn(D, u4, wu, rb, qa);
                for (int k = 0; k < 4; ++k) {
                    const float g = g4[k] * hdr.gate_scale;
                    const float u = u4[k] * hdr.up_scale;
                    const float sig = 1.0f / (1.0f + std::exp(-g));
                    act[j + k] = (kInkSituBeta * std::tanh(g / kInkSituBeta) * sig) * (kInkSituLb * std::tanh(u / kInkSituLb));
                }
            }
        }
        for (; j < j1; ++j) {
            const uint8_t* wg = w_base + (size_t)j * rb;
            const uint8_t* wu = w_base + (size_t)(I + j) * rb;
            float gs = 0.0f, us = 0.0f;
            dk->fn(D, &gs, wg, qa);
            dk->fn(D, &us, wu, qa);
            const float g = gs * hdr.gate_scale;
            const float u = us * hdr.up_scale;
            const float sig = 1.0f / (1.0f + std::exp(-g));
            act[j] = (kInkSituBeta * std::tanh(g / kInkSituBeta) * sig) * (kInkSituLb * std::tanh(u / kInkSituLb));
        }
        return;
    }
    if (em.gu_code == FMT_SF4) {
        float hdr[4];
        std::memcpy(hdr, body, 16);
        const float w_outer = hdr[0] / hdr[1];
        const uint8_t* bp = body + 16;
        const uint8_t* bs = bp + (size_t)(2 * I) * (D / 2);
        for (int j = j0; j < j1; ++j) {
            const float g = dot_sf4_row_h(bp + (size_t)(2 * j) * (D / 2), bs + (size_t)(2 * j) * (D / 16), w_outer, x, D);
            const float u = dot_sf4_row_h(bp + (size_t)(2 * j + 1) * (D / 2), bs + (size_t)(2 * j + 1) * (D / 16), w_outer, x, D);
            const float sig = 1.0f / (1.0f + std::exp(-g));
            act[j] = (kInkSituBeta * std::tanh(g / kInkSituBeta) * sig) * (kInkSituLb * std::tanh(u / kInkSituLb));
        }
    } else if (em.gu_code == FMT_MX8) {
        const uint8_t* bp = body;
        const uint8_t* bs = bp + (size_t)(2 * I) * D;
        for (int j = j0; j < j1; ++j) {
            const float g = dot_mx8_row_h(bp + (size_t)(2 * j) * D, bs + (size_t)(2 * j) * (D / 32), x, D);
            const float u = dot_mx8_row_h(bp + (size_t)(2 * j + 1) * D, bs + (size_t)(2 * j + 1) * (D / 32), x, D);
            const float sig = 1.0f / (1.0f + std::exp(-g));
            act[j] = (kInkSituBeta * std::tanh(g / kInkSituBeta) * sig) * (kInkSituLb * std::tanh(u / kInkSituLb));
        }
    } else {
        InkQ8_1_G64 xq[D / 64];
        quant_q8_1_h(x, xq, D);
        const int sb = ink_sb_bytes(em.gu_code);
        const size_t row_b = 9ull * sb;
        const uint8_t* gu_base = body + I * 2;
        const bool fast = sk_fast(em.gu_code);
        const uint8_t* xqp[1] = {reinterpret_cast<const uint8_t*>(xq)};
        for (int j = j0; j < j1; ++j) {
            const uint8_t* gr = gu_base + (size_t)(2 * j) * row_b;
            const uint8_t* ur = gr + row_b;
            float g = 0.0f, u = 0.0f;
            if (fast) {
                host_avx2::dot_sk_row_multi(gr, em.gu_code, sb, 9, xqp, 1, &g);
                host_avx2::dot_sk_row_multi(ur, em.gu_code, sb, 9, xqp, 1, &u);
            } else {
                for (int s = 0; s < 9; ++s) {
                    const uint8_t* bg = gr + s * sb;
                    const uint8_t* bu = ur + s * sb;
                    uint16_t dg, du;
                    std::memcpy(&dg, bg, 2); std::memcpy(&du, bu, 2);
                    const float Dg = f16_to_f32_h(dg), Du = f16_to_f32_h(du);
                    for (int sg = 0; sg < 4; ++sg) {
                        g += dot_sk_sub64_h(bg + 6, em.gu_code, sg, xq[s * 4 + sg], Dg * ((float)bg[2 + sg] * (1.0f / 255.0f)));
                        u += dot_sk_sub64_h(bu + 6, em.gu_code, sg, xq[s * 4 + sg], Du * ((float)bu[2 + sg] * (1.0f / 255.0f)));
                    }
                }
            }
            uint16_t sq_bits;
            std::memcpy(&sq_bits, body + j * 2, 2);
            const float inv_sq = 1.0f / f16_to_f32_h(sq_bits);
            const float sig = 1.0f / (1.0f + std::exp(-g));
            act[j] = (kInkSituBeta * std::tanh(g / kInkSituBeta) * sig) * (kInkSituLb * std::tanh(u / kInkSituLb)) * inv_sq;
        }
    }
}

inline void expert_phase2(const InkHostExpertMeta& em, const uint8_t* body, const float* act, int r0, int r1, float* y_out) noexcept {
    constexpr int D = kInkHidden;
    constexpr int I = kInkInter;
    if (em.gu_code == FMT_GGUF_RAW || em.dn_code == FMT_GGUF_RAW) {
        InkGgufExpertHeader hdr{};
        std::memcpy(&hdr, body, sizeof(hdr));
        const auto arith = strata::quant::default_arith();
        const auto qt = static_cast<strata::quant::Type>(hdr.dn_type);
        const auto* ti = strata::quant::type_info(qt);
        const size_t rb = strata::quant::row_bytes((int)hdr.dn_type, I);
        const auto* dk = strata::quant::dot_kernel(qt, arith);
        const auto* dk4 = strata::quant::dot_x4_kernel(qt, arith);
        const auto* ak = strata::quant::act_kernel(ti->act, arith);
        const uint8_t* w_base = body + em.dn_off;
        alignas(32) uint8_t qa[2048];
        ak->fn(act, qa, I);
        int r = r0;
        if (dk4) {
            for (; r + 4 <= r1; r += 4) {
                float s4[4];
                dk4->fn(I, s4, w_base + (size_t)r * rb, rb, qa);
                y_out[r + 0] = s4[0] * hdr.dn_scale;
                y_out[r + 1] = s4[1] * hdr.dn_scale;
                y_out[r + 2] = s4[2] * hdr.dn_scale;
                y_out[r + 3] = s4[3] * hdr.dn_scale;
            }
        }
        for (; r < r1; ++r) {
            float s = 0.0f;
            dk->fn(I, &s, w_base + (size_t)r * rb, qa);
            y_out[r] = s * hdr.dn_scale;
        }
        return;
    }
    if (em.dn_code == FMT_SF4) {
        float hdr[4];
        std::memcpy(hdr, body, 16);
        const float w_outer = hdr[2] / hdr[3];
        const uint8_t* bp = body + em.dn_off;
        const uint8_t* bs = bp + (size_t)D * (I / 2);
        for (int r = r0; r < r1; ++r) y_out[r] = dot_sf4_row_h(bp + (size_t)r * (I / 2), bs + (size_t)r * (I / 16), w_outer, act, I);
    } else if (em.dn_code == FMT_MX8) {
        const uint8_t* bp = body + em.dn_off;
        const uint8_t* bs = bp + (size_t)D * I;
        for (int r = r0; r < r1; ++r) y_out[r] = dot_mx8_row_h(bp + (size_t)r * I, bs + (size_t)r * (I / 32), act, I);
    } else {
        InkQ8_1_G64 hq[I / 64];
        quant_q8_1_h(act, hq, I);
        const int sb = ink_sb_bytes(em.dn_code);
        const size_t pair_b = 9ull * sb;
        const uint8_t* dn_base = body + em.dn_off;
        const bool fast = sk_fast(em.dn_code);
        const uint8_t* hqp[1] = {reinterpret_cast<const uint8_t*>(hq)};
        for (int pr = r0 / 2; pr < r1 / 2; ++pr) {
            const uint8_t* blk0 = dn_base + (size_t)pr * pair_b;
            float y0 = 0.0f, y1 = 0.0f;
            if (fast) {
                host_avx2::dot_sk_pair_multi(blk0, em.dn_code, sb, hqp, 1, &y0, &y1);
            } else {
                for (int s = 0; s < 9; ++s) {
                    const uint8_t* blk = blk0 + s * sb;
                    uint16_t d16;
                    std::memcpy(&d16, blk, 2);
                    const float Dsc = f16_to_f32_h(d16);
                    for (int sg = 0; sg < 4; ++sg) {
                        const int sub_g = s * 4 + sg;
                        const float eff = Dsc * ((float)blk[2 + sg] * (1.0f / 255.0f));
                        if (sub_g < 18) y0 += dot_sk_sub64_h(blk + 6, em.dn_code, sg, hq[sub_g], eff);
                        else            y1 += dot_sk_sub64_h(blk + 6, em.dn_code, sg, hq[sub_g - 18], eff);
                    }
                }
            }
            y_out[2 * pr]     = y0;
            y_out[2 * pr + 1] = y1;
        }
    }
}

// Opt-in multi-token row-sharded SF4/MX8 evaluators (STRATA_INK_RTX5060TI=1):
// Computes Gate+Up+SITU rows [j0, j1) or Down rows [r0, r1) across all T tokens at once using
// dot_sf4_row_multi / dot_mx8_row_multi (unpacks each weight block ONCE, pure AVX2 E4M3FN SIMD decode).
inline void eval_verify_phase1_multi(
    const InkHostExpertMeta& em, const uint8_t* body,
    const float* const* xs, int T, int j0, int j1, float* act_base) noexcept {
    constexpr int D = kInkHidden;
    constexpr int I = kInkInter;
    const bool use_avx2 = host_avx2::available();
    const float* e4 = e4m3_lut_h();
    const float* u8 = ue8m0_lut_h();
    float g[16], u[16];
    if (em.gu_code == FMT_SF4) {
        float hdr[4];
        std::memcpy(hdr, body, 16);
        const float w_outer = hdr[0] / hdr[1];
        const uint8_t* bp = body + 16;
        const uint8_t* bs = bp + (size_t)(2 * I) * (D / 2);
        for (int j = j0; j < j1; ++j) {
            const uint8_t* pg = bp + (size_t)(2 * j) * (D / 2);
            const uint8_t* sg = bs + (size_t)(2 * j) * (D / 16);
            const uint8_t* pu = bp + (size_t)(2 * j + 1) * (D / 2);
            const uint8_t* su = bs + (size_t)(2 * j + 1) * (D / 16);
            if (use_avx2) {
                host_avx2::dot_sf4_row_multi(pg, sg, w_outer, xs, T, D, e4, g);
                host_avx2::dot_sf4_row_multi(pu, su, w_outer, xs, T, D, e4, u);
            } else {
                for (int t = 0; t < T; ++t) {
                    g[t] = dot_sf4_row_h(pg, sg, w_outer, xs[t], D);
                    u[t] = dot_sf4_row_h(pu, su, w_outer, xs[t], D);
                }
            }
            for (int t = 0; t < T; ++t) {
                const float sig = 1.0f / (1.0f + std::exp(-g[t]));
                act_base[(size_t)t * I + j] = (kInkSituBeta * std::tanh(g[t] / kInkSituBeta) * sig) *
                                              (kInkSituLb * std::tanh(u[t] / kInkSituLb));
            }
        }
    } else if (em.gu_code == FMT_MX8) {
        const uint8_t* bp = body;
        const uint8_t* bs = bp + (size_t)(2 * I) * D;
        for (int j = j0; j < j1; ++j) {
            const uint8_t* pg = bp + (size_t)(2 * j) * D;
            const uint8_t* sg = bs + (size_t)(2 * j) * (D / 32);
            const uint8_t* pu = bp + (size_t)(2 * j + 1) * D;
            const uint8_t* su = bs + (size_t)(2 * j + 1) * (D / 32);
            if (use_avx2) {
                host_avx2::dot_mx8_row_multi(pg, sg, xs, T, D, u8, g);
                host_avx2::dot_mx8_row_multi(pu, su, xs, T, D, u8, u);
            } else {
                for (int t = 0; t < T; ++t) {
                    g[t] = dot_mx8_row_h(pg, sg, xs[t], D);
                    u[t] = dot_mx8_row_h(pu, su, xs[t], D);
                }
            }
            for (int t = 0; t < T; ++t) {
                const float sig = 1.0f / (1.0f + std::exp(-g[t]));
                act_base[(size_t)t * I + j] = (kInkSituBeta * std::tanh(g[t] / kInkSituBeta) * sig) *
                                              (kInkSituLb * std::tanh(u[t] / kInkSituLb));
            }
        }
    }
}

inline void eval_verify_phase2_multi(
    const InkHostExpertMeta& em, const uint8_t* body,
    const float* act_base, int T, int r0, int r1, float* const* ys) noexcept {
    constexpr int D = kInkHidden;
    constexpr int I = kInkInter;
    const bool use_avx2 = host_avx2::available();
    const float* e4 = e4m3_lut_h();
    const float* u8 = ue8m0_lut_h();
    const float* act_ptrs[16];
    for (int t = 0; t < T; ++t) act_ptrs[t] = act_base + (size_t)t * I;
    float yv[16];
    if (em.dn_code == FMT_SF4) {
        float hdr[4];
        std::memcpy(hdr, body, 16);
        const float w_outer = hdr[2] / hdr[3];
        const uint8_t* bp = body + em.dn_off;
        const uint8_t* bs = bp + (size_t)D * (I / 2);
        for (int r = r0; r < r1; ++r) {
            const uint8_t* pr = bp + (size_t)r * (I / 2);
            const uint8_t* sr = bs + (size_t)r * (I / 16);
            if (use_avx2) {
                host_avx2::dot_sf4_row_multi(pr, sr, w_outer, act_ptrs, T, I, e4, yv);
                for (int t = 0; t < T; ++t) ys[t][r] = yv[t];
            } else {
                for (int t = 0; t < T; ++t) ys[t][r] = dot_sf4_row_h(pr, sr, w_outer, act_ptrs[t], I);
            }
        }
    } else if (em.dn_code == FMT_MX8) {
        const uint8_t* bp = body + em.dn_off;
        const uint8_t* bs = bp + (size_t)D * I;
        for (int r = r0; r < r1; ++r) {
            const uint8_t* pr = bp + (size_t)r * I;
            const uint8_t* sr = bs + (size_t)r * (I / 32);
            if (use_avx2) {
                host_avx2::dot_mx8_row_multi(pr, sr, act_ptrs, T, I, u8, yv);
                for (int t = 0; t < T; ++t) ys[t][r] = yv[t];
            } else {
                for (int t = 0; t < T; ++t) ys[t][r] = dot_mx8_row_h(pr, sr, act_ptrs[t], I);
            }
        }
    }
}

}  // namespace host_eval

// ============================================================================
// 2. CUDA-Graph Mapped-Memory Doorbell + Spin-Wait CPU Pool (Zero Per-Layer Sync!)
// ============================================================================
struct alignas(64) InkMappedDoorbell {
    volatile uint32_t ring;
    uint32_t pad0[15];
    volatile uint32_t ack;
    uint32_t pad1[15];
    int32_t layer;
    int32_t n_jobs;
    int32_t slot_k[kInkTopK];
    int32_t expert_id[kInkTopK];
    uint32_t pad2[14];
    float x[kInkHidden];
    float y[kInkTopK][kInkHidden];
};

// One routing record of the expert trace (STRATA_INK_TRACE): 16 bytes, written by the doorbell kernel (kind 0: one decode token,
// M = 1) or by the host in the batched path (kind 1: one row of a verify / short-prefill step, `seq` = how many kind-0 records existed).
struct InkTraceRec {
    uint8_t layer;
    uint8_t kind;
    uint8_t row;
    uint8_t M;
    uint8_t ids[kInkTopK];
    uint32_t seq;
};
static_assert(sizeof(InkTraceRec) == 16, "InkTraceRec is 16 bytes");

// Runs INSIDE CUDA Graph right after topk8 router! Inspects tier==2 experts, records misses, rings CPU doorbell!
__global__ void __launch_bounds__(256) k_ink_doorbell_ring(
    int layer,
    const InkExpertDev* __restrict__ experts,
    const int32_t* __restrict__ topk_idx,     // [8]
    const float* __restrict__ x_f32,          // [2304]
    float* __restrict__ use_cnt,              // [24 * 128] optional frequency tracker
    int32_t* __restrict__ want_keys,          // [64] promotion candidates
    int32_t* __restrict__ n_want,             // [1]
    InkMappedDoorbell* bell,
    uint32_t* __restrict__ seq_dev,           // [1]
    uint32_t* __restrict__ cur_seq,           // [1]
    unsigned long long* __restrict__ stat,    // [2] routed experts / of those not in the card (hit-rate counters of the M = 1 path)
    InkTraceRec* __restrict__ trace,          // optional ring of routing records (nullptr = off)
    uint32_t* __restrict__ trace_n,           // [1] records written so far (ring index = value % trace_cap)
    uint32_t trace_cap) {
    __shared__ int s_nj;
    const int tid = threadIdx.x;
    if (tid == 0) {
        int nj = 0;
        int nw = n_want ? *n_want : 0;
        if (trace) {
            const uint32_t ti = *trace_n;
            InkTraceRec r;
            r.layer = (uint8_t)layer; r.kind = 0; r.row = 0; r.M = 1; r.seq = ti;
            for (int k = 0; k < kInkTopK; ++k) r.ids[k] = (uint8_t)topk_idx[k];
            trace[ti % trace_cap] = r;
            *trace_n = ti + 1u;
        }
        for (int k = 0; k < kInkTopK; ++k) {
            const int e = topk_idx[k];
            if (e < 0) continue;   // a slot dropped by the opt-in tail skip: nothing to fetch or compute
            const int key = layer * kInkExperts + e;
            if (use_cnt) use_cnt[key] += 1.0f;
            if (experts[e].tier == 2) {
                bell->slot_k[nj] = k;
                bell->expert_id[nj] = e;
                ++nj;
                if (want_keys && nw < 64) {
                    bool dup = false;
                    for (int i = 0; i < nw; ++i) dup |= (want_keys[i] == key);
                    if (!dup) want_keys[nw++] = key;
                }
            }
        }
        if (n_want) *n_want = nw;
        if (stat) { stat[0] += (unsigned long long)kInkTopK; stat[1] += (unsigned long long)nj; }
        s_nj = nj;
    }
    __syncthreads();
    if (s_nj == 0) {
        if (tid == 0) *cur_seq = 0u;
        return;
    }
    for (int i = tid; i < kInkHidden; i += 256) bell->x[i] = x_f32[i];
    __threadfence_system();
    __syncthreads();
    if (tid == 0) {
        bell->layer = layer;
        bell->n_jobs = s_nj;
        __threadfence_system();
        const uint32_t s = ++(*seq_dev);
        bell->ring = s;
        __threadfence_system();
        *cur_seq = s;
    }
}

// Runs INSIDE CUDA Graph right after GPU VRAM experts finish! Collects CPU y[k] * topk_w[k] into y_pair!
__global__ void __launch_bounds__(256) k_ink_doorbell_wait_collect(
    const InkMappedDoorbell* bell,
    const uint32_t* __restrict__ cur_seq,
    const float* __restrict__ topk_w,         // [8]
    float* __restrict__ y_pair) {             // [8, 2304]
    const uint32_t s = *cur_seq;
    if (s == 0u) return;
    __shared__ int s_nj;
    const int tid = threadIdx.x;
    if (tid == 0) {
        while (bell->ack != s) {
            // Brief spin until CPU pool completes DDR5 expert evaluation (~0.1-0.4 ms)
        }
        s_nj = bell->n_jobs;
    }
    __syncthreads();
    const int j = blockIdx.x;
    if (j >= s_nj) return;
    const int k = bell->slot_k[j];
    const float rw = topk_w[k];
    const float4* src = reinterpret_cast<const float4*>(bell->y[k]);
    float4* dst = reinterpret_cast<float4*>(y_pair + (size_t)k * kInkHidden);
    for (int i = tid; i < (kInkHidden / 4); i += 256) {
        const float4 v = __ldcv(src + i);
        dst[i] = make_float4(v.x * rw, v.y * rw, v.z * rw, v.w * rw);
    }
}

// ---- Exact mode, M = 1, inside the CUDA graph: the cold experts of a visit are copied by the card itself from the pinned arena into per-slot staging
// areas and then computed by the card's own expert kernels, so the bits of a token do not depend on where an expert lives. Three small steps:
//   plan: for each of the 8 routed experts that is cold (tier 2): remember its table entry, point the entry at the staging slot (tier 0);
//   copy: all blocks copy the cold experts' bytes (host -> card, 16-byte loads, many in flight);
//   [the ordinary dispatch of tier <= 1 experts runs]
//   restore: the entries get their old contents back (a cold expert is cold again for the next visit and for the host's own bookkeeping).
struct InkStageState {
    InkExpertDev saved[kInkTopK];
    uint32_t bytes[kInkTopK];
    int32_t eid[kInkTopK];
    int32_t miss[kInkTopK];
};

__global__ void __launch_bounds__(32) k_ink_stage_plan(
    int layer,
    InkExpertDev* __restrict__ experts,
    const int32_t* __restrict__ topk_idx,
    const uint32_t* __restrict__ key_bytes,
    uint8_t* __restrict__ stage, size_t slot_bytes,
    const uint8_t* __restrict__ sec4_base,
    InkStageState* __restrict__ ss,
    float* __restrict__ use_cnt,
    unsigned long long* __restrict__ stat) {
    const int k = threadIdx.x;
    int miss = 0;
    if (k < kInkTopK && topk_idx[k] < 0) {   // a slot dropped by the opt-in tail skip: not a miss, nothing to restore
        ss->eid[k] = -1;
        ss->miss[k] = 0;
        ss->bytes[k] = 0;
    } else if (k < kInkTopK) {
        const int e = topk_idx[k];
        const int key = layer * kInkExperts + e;
        if (use_cnt) atomicAdd(&use_cnt[key], 1.0f);
        InkExpertDev ed = experts[e];
        miss = (ed.tier == 2) ? 1 : 0;
        ss->eid[k] = e;
        ss->miss[k] = miss;
        ss->saved[k] = ed;
        ss->bytes[k] = key_bytes[key];
        if (miss) {
            InkExpertDev p = ed;
            p.tier = 0;
            p.body_off = (uint64_t)((intptr_t)(stage + (size_t)k * slot_bytes) - (intptr_t)sec4_base);
            experts[e] = p;
        }
    }
    const unsigned ballot = __ballot_sync(0xFFu, miss);
    if (k == 0 && stat) { stat[0] += (unsigned long long)kInkTopK; stat[1] += (unsigned long long)__popc(ballot); }
}

// grid (chunks, 8), block 256: chunk c of cold expert k, 64 KiB per block
__global__ void __launch_bounds__(256) k_ink_stage_copy(
    const InkStageState* __restrict__ ss,
    const uint8_t* __restrict__ arena,
    uint8_t* __restrict__ stage, size_t slot_bytes) {
    const int k = blockIdx.y;
    if (!ss->miss[k]) return;
    constexpr uint32_t CH = 64u * 1024u;
    const uint32_t begin = blockIdx.x * CH;
    const uint32_t total = ss->bytes[k];
    if (begin >= total) return;
    const uint32_t len = min(CH, total - begin);
    const uint8_t* src = arena + ss->saved[k].body_off + begin;
    uint8_t* dst = stage + (size_t)k * slot_bytes + begin;
    const uint32_t n16 = len / 16u;
    const uint4* s4 = reinterpret_cast<const uint4*>(src);
    uint4* d4 = reinterpret_cast<uint4*>(dst);
    uint32_t i = threadIdx.x;
    for (; i + 3u * 256u < n16; i += 4u * 256u) {
        const uint4 a = s4[i], b = s4[i + 256u], c = s4[i + 512u], d = s4[i + 768u];
        d4[i] = a; d4[i + 256u] = b; d4[i + 512u] = c; d4[i + 768u] = d;
    }
    for (; i < n16; i += 256u) d4[i] = s4[i];
    if (threadIdx.x < (len & 15u)) dst[n16 * 16u + threadIdx.x] = src[n16 * 16u + threadIdx.x];
}

__global__ void __launch_bounds__(32) k_ink_stage_restore(InkExpertDev* __restrict__ experts, const InkStageState* __restrict__ ss) {
    const int k = threadIdx.x;
    if (k < kInkTopK && ss->miss[k]) experts[ss->eid[k]] = ss->saved[k];
}

__global__ void __launch_bounds__(256) k_ink_decay_use_cnt(float* __restrict__ cnt, int n, float decay) {
    const int i = blockIdx.x * 256 + threadIdx.x;
    if (i < n) cnt[i] *= decay;
}

// Multi-threaded spin-wait CPU worker pool attached to InkMappedDoorbell
struct InkHybridEngine {
    bool active = false;
    int n_layers = kInkPhysLayers;
    int n_experts = kInkExperts;

    // Pinned or pageable host expert table (all 3072 experts; body != nullptr for RAM-resident, cold_ssd == true for GPU+SSD Only)
    // On Windows WDDM / tight RAM hosts (#6520), falls back cleanly to pageable VirtualAlloc/mmap + pinned bounce ring!
    std::vector<InkHostExpertMeta> host_meta;  // [24 * 128]
    uint8_t* h_sec4_pinned   = nullptr;
    uint8_t* h_sec4_pageable = nullptr;
    size_t   h_sec4_bytes    = 0;
    bool     sec4_is_pinned  = false;
    uint8_t* h_pf_bounce[2]  = {};
    uint8_t* h_promo_stg     = nullptr;
    std::unique_ptr<ink2::InkColdStore> cold_store;
    bool ssd_only = false;
    std::atomic<uint64_t> step_gen{1};
    uint8_t* h_copyback_buf = nullptr;
    size_t   h_copyback_bytes = 0;

    // Mapped doorbell + device state
    InkMappedDoorbell* h_bell = nullptr;
    InkMappedDoorbell* d_bell = nullptr;
    uint32_t* d_seq = nullptr;
    uint32_t* d_cur_seq = nullptr;

    // Simultaneous PCIe + CPU miss-sharing buffers for M=2..16 DSpark Block-8 verify
    float* h_v_x    = nullptr;   // pinned [16 * kInkHidden]
    float* h_v_y    = nullptr;   // pinned [16 * kInkTopK * kInkHidden]
    int32_t* h_v_p  = nullptr;   // pinned [16 * kInkTopK] pair indices written by CPU
    int32_t* h_v_idx = nullptr;  // pinned [16 * kInkTopK] topk_idx D2H staging (<= 512 B)
    float* d_v_y    = nullptr;   // device mirror for fast scatter
    int32_t* d_v_p  = nullptr;   // device pair indices
    double dec_pcie_share = -1.0; // -1 = auto, 0 = 100% CPU, 1 = 100% PCIe DMA (STRATA_INK_DEC_PCIE_SHARE)
    // Share of the batched (verify) cold experts that go over PCIe, taken from the launch probe of THIS box: the CPU lane and the card's copy lane work in parallel, so
    // the share that finishes both together is t_cpu / (t_cpu + t_copy) (a lane's rate is 1 / its time per token). Set only when the card's row asks for it
    // (COMPACT_HYBRID: 8 / 12 GB cards); -1 everywhere else, and then the fixed 0.55 / 0.65 of the shipped path applies. Never read when no cold expert exists.
    double auto_pcie_share = -1.0;
    // Cold-expert lane at decode. STRATA_INK_EXACT unset = auto: a short launch probe times both lanes on this machine and keeps the faster one
    // (fast, the CPU lane, unless the card's copy is clearly faster); =0 forces the CPU lane (fast, not bit-identical to the card), =1 forces
    // exact (the card copies every cold expert in and computes it itself: the same bits wherever an expert lives, PCIe bound).
    int lane_pref = 0;           // 0 auto, 1 exact, 2 fast
    bool exact_mode = false;     // the lane in use now (set by set_lane)
    bool stage_ready = false;    // the in-graph staging of exact mode is prepared (pinned arena, staging buffers)
    bool probing = false;        // the launch probe is running: the tier does not move and does not learn
    double dec_share_user = -1.0;
    bool exact_graph = false;    // exact lane with the cold experts of a decode visit staged by the card inside the graph (needs the pinned arena)
    bool needs_eager() const noexcept { return exact_mode && !exact_graph; }   // exact lane without the in-graph staging: decode runs eagerly (host sync per visit)
    void set_lane(bool exact) noexcept {
        exact_mode = exact;
        exact_graph = exact && stage_ready;
        dec_pcie_share = exact ? 1.0 : dec_share_user;   // exact: every cold expert is copied into the card
    }
    uint32_t* d_key_bytes = nullptr;
    uint8_t* d_stage = nullptr;
    size_t stage_slot_bytes = 0;
    InkStageState* d_stage_state = nullptr;

    // After the loaders: prepare the in-graph staging of exact mode (the arena must be pinned: the card reads it).
    void exact_prepare() {
        if (lane_pref == 2 || !active || ssd_only || !sec4_is_pinned || !h_sec4_pinned || host_meta.empty() || stage_ready) return;
        if (const char* e = std::getenv("STRATA_INK_EXACT_GRAPH")) if (e[0] == '0') return;   // keep the eager copy path (for A/B)
        size_t mx = 0;
        std::vector<uint32_t> kb(host_meta.size(), 0u);
        for (size_t k = 0; k < host_meta.size(); ++k) {
            if (host_meta[k].body) { kb[k] = host_meta[k].body_bytes; mx = std::max<size_t>(mx, host_meta[k].body_bytes); }
        }
        if (mx == 0) return;
        stage_slot_bytes = (mx + 255ull) & ~255ull;
        if (cudaMalloc(&d_stage, stage_slot_bytes * kInkTopK) != cudaSuccess) { (void)cudaGetLastError(); d_stage = nullptr; return; }
        d_key_bytes = dalloc<uint32_t>(kb.size());
        INK_CK(cudaMemcpy(d_key_bytes, kb.data(), kb.size() * 4, cudaMemcpyHostToDevice));
        d_stage_state = dalloc<InkStageState>(1);
        INK_CK(cudaMemset(d_stage_state, 0, sizeof(InkStageState)));
        stage_ready = true;
        exact_graph = exact_mode;
    }
    uint32_t promo_step = 0;      // cadence counter so step_promote does NOT sync every single token!
    // Expert-cache statistics of the batched path (M <= 16, i.e. decode): every routed expert counted, and those that were not in the card
    // (host RAM or SSD) counted as misses; STATS reports the hit rate (the share of routed experts served from the card).
    uint64_t stat_selections = 0;
    uint64_t stat_misses = 0;
    // The same two counters for the M = 1 (doorbell) path live on the device (written inside the CUDA graph); stats_total() adds them.
    unsigned long long* d_stat = nullptr;   // [2]
    // CPU-pool profile of the M = 1 doorbell path (printed by STATS when STRATA_INK_CPU_PROF is set)
    std::atomic<uint64_t> prof_ns{0}, prof_visits{0}, prof_jobs{0}, prof_bytes{0};
    std::atomic<uint64_t> prof_plan_ns{0}, prof_own_ns{0}, prof_wait_ns{0}, prof_wbusy_ns{0}, prof_wtask_n{0};   // where a visit's wall time goes (row-shared path)
    bool prof_on = false;

    // Routing trace (STRATA_INK_TRACE=<file>): device ring of kind-0 records (decode, M = 1), host records of kind 1 (batched rows).
    InkTraceRec* d_trace = nullptr;
    uint32_t* d_trace_n = nullptr;
    uint32_t trace_cap = 0;
    uint32_t trace_flushed = 0;
    uint32_t trace_calls = 0;
    mutable std::vector<InkTraceRec> trace_host;   // mutable: STATS (a const call) appends a request marker (kind 2)
    std::FILE* trace_fp = nullptr;

    void trace_open() {
        const char* p = std::getenv("STRATA_INK_TRACE");
        if (!p || !*p || trace_fp) return;
        trace_fp = std::fopen(p, "wb");
        if (!trace_fp) { std::fprintf(stderr, "[InkHybrid] STRATA_INK_TRACE: cannot open %s\n", p); return; }
        trace_cap = 1u << 20;
        INK_CK(cudaMalloc(&d_trace, (size_t)trace_cap * sizeof(InkTraceRec)));
        d_trace_n = dalloc<uint32_t>(1);
        INK_CK(cudaMemset(d_trace_n, 0, 4));
        std::fprintf(stderr, "[InkHybrid] expert trace on: %s (16-byte records)\n", p);
    }
    // Append what the device and the host have recorded since the last call (a sync; trace mode only).
    void trace_flush(bool force) {
        if (!trace_fp) return;
        if (!force && ((++trace_calls) & 31u) != 0u) return;   // every 32nd call
        uint32_t n = 0;
        INK_CK(cudaDeviceSynchronize());
        INK_CK(cudaMemcpy(&n, d_trace_n, 4, cudaMemcpyDeviceToHost));
        if (n - trace_flushed > trace_cap) trace_flushed = n - trace_cap;  // overrun: oldest records are lost
        std::vector<InkTraceRec> dev(n - trace_flushed);
        for (uint32_t i = trace_flushed; i < n; ) {
            const uint32_t at = i % trace_cap;
            const uint32_t run = std::min<uint32_t>(n - i, trace_cap - at);
            INK_CK(cudaMemcpy(dev.data() + (i - trace_flushed), d_trace + at, (size_t)run * sizeof(InkTraceRec), cudaMemcpyDeviceToHost));
            i += run;
        }
        // merge: a host record with seq = s goes after device record s-1
        size_t hi = 0;
        for (uint32_t i = trace_flushed; i < n; ++i) {
            while (hi < trace_host.size() && trace_host[hi].seq <= i) { std::fwrite(&trace_host[hi], sizeof(InkTraceRec), 1, trace_fp); ++hi; }
            std::fwrite(&dev[i - trace_flushed], sizeof(InkTraceRec), 1, trace_fp);
        }
        while (hi < trace_host.size()) { std::fwrite(&trace_host[hi], sizeof(InkTraceRec), 1, trace_fp); ++hi; }
        trace_host.clear();
        trace_flushed = n;
        std::fflush(trace_fp);
    }
    void stats_total(uint64_t* sel, uint64_t* miss) const {
        unsigned long long s[2] = {0, 0};
        if (d_stat) {
            INK_CK(cudaDeviceSynchronize());
            INK_CK(cudaMemcpy(s, d_stat, sizeof(s), cudaMemcpyDeviceToHost));
        }
        *sel = stat_selections + s[0];
        *miss = stat_misses + s[1];
        if (std::getenv("STRATA_INK_CPU_PROF") && prof_visits.load() > 0) {
            const double ns = (double)prof_ns.load(), by = (double)prof_bytes.load(), vis = (double)prof_visits.load(), jobs = (double)prof_jobs.load();
            std::fprintf(stderr, "[InkCpuProf] threads=%d visits=%.0f jobs=%.0f (%.2f experts/visit) cpu wall %.3f s = %.1f us/visit; %.2f GB read = %.2f GB/s aggregate, %.2f GB/s per thread; %.2f MB/expert\n",
                         n_threads, vis, jobs, jobs / vis, ns * 1e-9, ns * 1e-3 / vis, by * 1e-9, by / ns, by / ns / std::max(1, n_threads), by / std::max(1.0, jobs) * 1e-6);
            std::fprintf(stderr, "[InkCpuProf] per visit: read bell + plan %.1f us, master's own share %.1f us, then waiting for the others %.1f us; the other threads busy %.1f us each (of %d)\n",
                         (double)prof_plan_ns.load() * 1e-3 / vis, (double)prof_own_ns.load() * 1e-3 / vis, (double)prof_wait_ns.load() * 1e-3 / vis,
                         (double)prof_wbusy_ns.load() * 1e-3 / vis / std::max(1, n_threads - 1), n_threads - 1);
        }
        if (trace_fp) {   // a STATS request marks the end of a request in the trace (kind 2)
            uint32_t dn = 0;
            INK_CK(cudaMemcpy(&dn, d_trace_n, 4, cudaMemcpyDeviceToHost));
            InkTraceRec r{};
            r.kind = 2; r.seq = dn;
            trace_host.push_back(r);
            const_cast<InkHybridEngine*>(this)->trace_flush(true);   // the file is complete up to this request even if the engine is killed
        }
    }
    bool rtx5060ti_opt = false;        // opt-in STRATA_INK_RTX5060TI=1: 24-core pool + single-sync balanced verify + row-sharded P/E balancing
    float rtx5060ti_pcie_share = 0.28f;
    bool l3_pf_enabled = true;    // STRATA_L3_PF: idle-worker next-layer L3 prefetch
    std::atomic<uint32_t> pf_gen{0};
    int pf_layer = -1, pf_n = 0;
    int32_t pf_eids[32] = {};
    int32_t prev_cpu_eids[24][32] = {};
    int prev_cpu_n[24] = {};

    // Allocate host expert arena safely (pinned if small & within 45% MemAvailable, else pageable + pinned bounce ring!)
    // NUMA: on a machine with several memory nodes, the experts the CPU pool reads would all sit on the node of the thread that loaded them. The
    // pages of the host arena are spread over all nodes (interleaved) so that the pool threads of every socket find about half their data local.
    // STRATA_INK_NUMA=1 turns it on (off by default). Linux only (set_mempolicy / mbind by system call, no libnuma). Returns the node mask (0 = one node / off).
    static unsigned long numa_all_nodes_mask(int* max_node) noexcept {
        *max_node = 0;
#if defined(__linux__)
        const char* env = std::getenv("STRATA_INK_NUMA");
        if (!env || env[0] != '1') return 0;   // opt in: no difference was measured on a 2-socket EPYC (21.7 vs 21.4 tok/s), so it stays off
        unsigned long mask = 0;
        if (FILE* f = std::fopen("/sys/devices/system/node/online", "r")) {
            char buf[128] = {};
            if (std::fgets(buf, sizeof(buf), f)) {
                const char* p = buf;
                while (*p >= '0' && *p <= '9') {
                    char* end = nullptr;
                    const long a = std::strtol(p, &end, 10);
                    long b = a;
                    if (*end == '-') b = std::strtol(end + 1, &end, 10);
                    for (long n = a; n <= b && n < 63; ++n) {
                        mask |= (1ul << n);
                        *max_node = std::max<int>(*max_node, (int)n + 1);
                    }
                    if (*end != ',') break;
                    p = end + 1;
                }
            }
            std::fclose(f);
        }
        return (mask & (mask - 1)) ? mask : 0;   // two or more nodes
#else
        return 0;
#endif
    }

    uint8_t* alloc_host_arena(size_t bytes) {
        h_sec4_bytes = bytes;
        const uint64_t avail = ink_ram_available_bytes();
#if defined(__linux__)
        int numa_max = 0;
        const unsigned long numa_mask = numa_all_nodes_mask(&numa_max);
        if (numa_mask) syscall(SYS_set_mempolicy, 3 /*MPOL_INTERLEAVE*/, &numa_mask, (unsigned long)(numa_max + 1));
        struct NumaRestore { bool on; ~NumaRestore() { if (on) syscall(SYS_set_mempolicy, 0 /*MPOL_DEFAULT*/, nullptr, 0ul); } } numa_restore{numa_mask != 0};
        if (numa_mask) std::fprintf(stderr, "[InkHybrid] host expert arena: pages interleaved over the NUMA nodes (mask 0x%lx)\n", numa_mask);
#endif
#ifdef _WIN32
        constexpr bool kWindows = true;
#else
        constexpr bool kWindows = false;
#endif
        // may the arena be page-locked? (core/ink_pin_rule.hpp: Windows min(16 GiB, 35% of the available RAM) and 8 GiB must stay available, Linux 45%;
        // STRATA_INK_PIN_GB=<n> sets the cap in GiB, =0 turns pinning off.) Whatever happens, the start continues on the pageable path.
        const strata::InkPinRule pin_rule = strata::ink_pin_rule(kWindows, (uint64_t)bytes, avail, std::getenv("STRATA_INK_PIN_GB"));
        if (pin_rule.pin) {
            if (cudaHostAlloc(&h_sec4_pinned, bytes, cudaHostAllocDefault) == cudaSuccess) {
                sec4_is_pinned = true;
                std::fprintf(stderr, "[InkHybrid] host expert arena %.2f GB pinned (%s; available RAM %.1f GB): the card reads the cold experts itself\n",
                             (double)bytes / 1e9, pin_rule.why, (double)avail / 1e9);
                return h_sec4_pinned;
            }
            (void)cudaGetLastError();
            h_sec4_pinned = nullptr;
            std::fprintf(stderr, "[InkHybrid] host expert arena %.2f GB: pinning failed (cudaHostAlloc), using pageable memory\n", (double)bytes / 1e9);
        } else {
            std::fprintf(stderr, "[InkHybrid] host expert arena %.2f GB not pinned (%s; available RAM %.1f GB): pageable memory with a pinned bounce ring\n",
                         (double)bytes / 1e9, pin_rule.why, (double)avail / 1e9);
        }
        sec4_is_pinned = false;
#ifdef _WIN32
        h_sec4_pageable = (uint8_t*)VirtualAlloc(nullptr, bytes, MEM_RESERVE | MEM_COMMIT, PAGE_READWRITE);
#else
        void* p = mmap(nullptr, bytes, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
        if (p != MAP_FAILED) {
#ifdef MADV_HUGEPAGE
            madvise(p, bytes, MADV_HUGEPAGE);
#endif
#if defined(__linux__)
            if (numa_mask) syscall(SYS_mbind, p, bytes, 3 /*MPOL_INTERLEAVE*/, &numa_mask, (unsigned long)(numa_max + 1), 0u);   // before any page is touched
#endif
            h_sec4_pageable = (uint8_t*)p;
        }
#endif
        std::fprintf(stderr, "[InkHybrid] host expert arena %.2f GB allocated as pageable RAM + pinned bounce ring (avail=%.1f GB)\n",
                     (double)bytes / 1e9, (double)avail / 1e9);
        return h_sec4_pageable;
    }

    void kick_l3_prefetch(int layer, const int32_t* eids, int n) noexcept {
        if (!l3_pf_enabled || !active || !eids || n <= 0 || layer < 0 || layer >= n_layers) return;
        const int take = std::min(n, 32);
        pf_layer = layer;
        pf_n = take;
        for (int i = 0; i < take; ++i) pf_eids[i] = eids[i];
        pf_gen.fetch_add(1, std::memory_order_release);
    }

    // Dynamic Decayed-Frequency Promotion Ring in VRAM (optional when host experts exist!)
    uint8_t*  d_promo_pool = nullptr;
    size_t    promo_slot_bytes = 0;
    int       n_promo_slots = 0;
    float*    d_use_cnt = nullptr;       // [3072]
    int32_t*  d_want_keys = nullptr;     // [64]
    int32_t*  d_n_want = nullptr;        // [1]
    std::vector<int> promo_owner;        // [n_promo_slots] -> expert key or -1
    std::vector<int> key_to_slot;        // [3072] -> slot or -1
    std::vector<uint64_t> orig_host_off; // [3072] original tier-1/2 offset
    cudaStream_t promo_st = nullptr;

    // Double-Buffered Prefill Streaming Arena (2 x 16..64 MiB ping-pong VRAM buffers + pinned descriptors!)
    uint8_t* d_pf_stage[2] = {};
    size_t   pf_stage_bytes = 0;
    cudaStream_t pf_dma_st = nullptr;
    cudaEvent_t  pf_ev_dma[2] = {};
    cudaEvent_t  pf_ev_done[2] = {};
    InkExpertDev* d_pf_exp_tmp[2] = {};
    InkExpertDev* h_pf_exp_pin[2] = {};
    bool pf_buf_busy[2] = {false, false};

    // Spin-wait worker threads (shared by M=1 doorbell AND M=2..16 expert-grouped verify — ZERO std::thread spawns!)
    struct CpuVerifyGroup {
        int eid = 0;
        int count = 0;
        int pids[16] = {};
        int out_slots[16] = {};
    };
    std::atomic<bool> stop_flag{false};
    std::atomic<uint32_t> work_gen{0};
    std::atomic<int> work_done{0};
    std::atomic<int> verify_next{0};
    std::atomic<int> workers_sleeping{0};
    std::mutex wake_mu;
    std::condition_variable wake_cv;
    int work_mode = 0;  // 0 = M=1 doorbell jobs (one expert per thread), 1 = M=2..16 expert-grouped verify jobs, 2 = M=1 doorbell with the experts' rows shared
    // Row tasks of mode 2: phase 1 = gate/up rows [begin, end) of expert `job`, phase 2 = down rows; all phase-1 tasks come first in the list,
    // so a thread that takes a phase-2 task only waits for chunks that are already running.
    struct RowTask { int job; int phase; int begin; int end; };
    std::vector<RowTask> row_tasks = std::vector<RowTask>(4096);
    int n_row_tasks = 0;
    std::atomic<int> row_next{0};
    std::atomic<int> p1_left[kInkTopK];
    float act_sh[kInkTopK][kInkInter];
    bool split_experts = true;   // STRATA_INK_SPLIT=0 goes back to one expert per thread
    int cur_layer = 0, cur_nj = 0, cur_ng = 0;

    // Opt-in row-sharded multi-token verify state (work_mode == 3, STRATA_INK_RTX5060TI=1 only)
    std::vector<RowTask> v_tasks;
    int n_v_tasks = 0;
    std::atomic<int> v_task_next{0};
    std::atomic<int> vp1_left[kInkExperts];
    const float* v_xs[kInkExperts][16] = {};
    float* v_ys[kInkExperts][16] = {};
    float* v_act_ptr[kInkExperts] = {};
    std::vector<float> verify_act_sh;  // [128 * kInkInter] = 576 KiB

    void run_verify_row_tasks() noexcept {
        while (true) {
            const int ti = v_task_next.fetch_add(1, std::memory_order_relaxed);
            if (ti >= n_v_tasks) break;
            const RowTask& tk = v_tasks[(size_t)ti];
            const int gi = tk.job;
            const auto& grp = verify_groups[gi];
            const int key = cur_layer * n_experts + grp.eid;
            const auto& hm = host_meta[(size_t)key];
            if (tk.phase == 1) {
                host_eval::eval_verify_phase1_multi(hm, hm.body, v_xs[gi], grp.count, tk.begin, tk.end, v_act_ptr[gi]);
                vp1_left[gi].fetch_sub(1, std::memory_order_release);
            } else {
                while (vp1_left[gi].load(std::memory_order_acquire) > 0) {
#if (defined(__GNUC__) || defined(__clang__)) && defined(__x86_64__)
                    __builtin_ia32_pause();
#else
                    std::this_thread::yield();
#endif
                }
                host_eval::eval_verify_phase2_multi(hm, hm.body, v_act_ptr[gi], grp.count, tk.begin, tk.end, v_ys[gi]);
            }
        }
    }

    bool plan_verify_row_tasks(int ng) noexcept {
        if (!rtx5060ti_opt || !split_experts || n_threads < 2 || ng < 1 || ng > kInkExperts) return false;
        if (verify_act_sh.size() < (size_t)(16 * kInkTopK) * kInkInter) return false;
        int total_cnt = 0;
        for (int gi = 0; gi < ng; ++gi) {
            const auto& grp = verify_groups[gi];
            if (grp.count <= 0 || grp.count > 16) return false;
            const auto& hm = host_meta[(size_t)(cur_layer * n_experts + grp.eid)];
            if (!hm.body) return false;
            if ((hm.gu_code != FMT_SF4 && hm.gu_code != FMT_MX8) ||
                (hm.dn_code != FMT_SF4 && hm.dn_code != FMT_MX8)) return false;
            total_cnt += grp.count;
        }
        if (total_cnt <= 0 || total_cnt > 16 * kInkTopK) return false;
        int act_off = 0;
        int order[kInkExperts];
        for (int gi = 0; gi < ng; ++gi) {
            order[gi] = gi;
            const auto& grp = verify_groups[gi];
            v_act_ptr[gi] = verify_act_sh.data() + (size_t)act_off * kInkInter;
            act_off += grp.count;
            for (int k = 0; k < grp.count; ++k) {
                const int pid = grp.pids[k];
                const int m   = pid / kInkTopK;
                const int os  = grp.out_slots[k];
                h_v_p[os] = pid;
                v_xs[gi][k] = h_v_x + (size_t)m * kInkHidden;
                v_ys[gi][k] = h_v_y + (size_t)os * kInkHidden;
            }
            vp1_left[gi].store(0, std::memory_order_relaxed);
        }
        // Heaviest multi-token experts first so fast P-cores start them immediately and minimize tail latency!
        std::sort(order, order + ng, [&](int a, int b) { return verify_groups[a].count > verify_groups[b].count; });
        const int target_tasks = std::max(n_threads * 4, ng * 2);
        auto chunk16 = [](int rows, int n, int min_c) {
            int c = (rows + n - 1) / std::max(1, n);
            c = (c + 15) & ~15;
            return std::max(min_c, c);
        };
        if (v_tasks.size() < 4096) v_tasks.resize(4096);
        n_v_tasks = 0;
        int c2_arr[kInkExperts];
        for (int idx = 0; idx < ng; ++idx) {
            const int gi = order[idx];
            const int cnt = verify_groups[gi].count;
            const int nc = std::max(1, std::min(16, (cnt * target_tasks + total_cnt - 1) / total_cnt));
            const int c1 = chunk16(kInkInter, nc, 64);
            const int c2 = chunk16(kInkHidden, nc, 128);
            c2_arr[gi] = c2;
            for (int b = 0; b < kInkInter; b += c1) {
                if (n_v_tasks >= (int)v_tasks.size()) return false;
                v_tasks[(size_t)n_v_tasks++] = {gi, 1, b, std::min(kInkInter, b + c1)};
                vp1_left[gi].fetch_add(1, std::memory_order_relaxed);
            }
        }
        for (int idx = 0; idx < ng; ++idx) {
            const int gi = order[idx];
            const int c2 = c2_arr[gi];
            for (int b = 0; b < kInkHidden; b += c2) {
                if (n_v_tasks >= (int)v_tasks.size()) return false;
                v_tasks[(size_t)n_v_tasks++] = {gi, 2, b, std::min(kInkHidden, b + c2)};
            }
        }
        v_task_next.store(0, std::memory_order_relaxed);
        return (n_v_tasks > 0);
    }

    void run_row_tasks() noexcept {
        while (true) {
            const int ti = row_next.fetch_add(1, std::memory_order_relaxed);
            if (ti >= n_row_tasks) break;
            const RowTask& tk = row_tasks[(size_t)ti];
            const int key = cur_layer * n_experts + cur_eid[tk.job];
            const auto& hm = host_meta[(size_t)key];
            float* ydst = const_cast<float*>(h_bell->y[cur_slot_k[tk.job]]);
            if (tk.phase == 1) {
                if (rtx5060ti_opt && (hm.gu_code == FMT_SF4 || hm.gu_code == FMT_MX8)) {
                    const float* x1[1] = {cur_x};
                    host_eval::eval_verify_phase1_multi(hm, hm.body, x1, 1, tk.begin, tk.end, act_sh[tk.job]);
                } else {
                    host_eval::expert_phase1(hm, hm.body, cur_x, tk.begin, tk.end, act_sh[tk.job]);
                }
                p1_left[tk.job].fetch_sub(1, std::memory_order_release);
            } else {
                while (p1_left[tk.job].load(std::memory_order_acquire) > 0) {
#if (defined(__GNUC__) || defined(__clang__)) && defined(__x86_64__)
                    __builtin_ia32_pause();
#else
                    std::this_thread::yield();
#endif
                }
                if (rtx5060ti_opt && (hm.dn_code == FMT_SF4 || hm.dn_code == FMT_MX8)) {
                    float* y1[1] = {ydst};
                    host_eval::eval_verify_phase2_multi(hm, hm.body, act_sh[tk.job], 1, tk.begin, tk.end, y1);
                } else {
                    host_eval::expert_phase2(hm, hm.body, act_sh[tk.job], tk.begin, tk.end, ydst);
                }
            }
        }
    }

    // Can this visit's experts be shared by rows? Then lay out the tasks (returns false: the caller uses one expert per thread).
    bool plan_row_tasks() noexcept {
        if (!split_experts || n_threads < 2 || cur_nj < 1) return false;
        for (int j = 0; j < cur_nj; ++j) {
            const auto& hm = host_meta[(size_t)(cur_layer * n_experts + cur_eid[j])];
            if (!hm.body || cur_cold_slot[j] >= 0 || !host_eval::expert_splittable(hm, hm.body)) return false;
        }
        const int per_phase = std::max(n_threads * 3, cur_nj);
        const int nc = std::max(1, per_phase / cur_nj);
        auto chunk = [](int rows, int n) { int c = (rows + n - 1) / n; c = (c + 7) & ~7; return std::max(8, c); };
        const int c1 = chunk(kInkInter, nc), c2 = chunk(kInkHidden, nc);
        n_row_tasks = 0;
        for (int j = 0; j < cur_nj; ++j) p1_left[j].store(0, std::memory_order_relaxed);
        for (int b = 0; b < kInkInter; b += c1)
            for (int j = 0; j < cur_nj; ++j) {
                row_tasks[(size_t)n_row_tasks++] = {j, 1, b, std::min(kInkInter, b + c1)};
                p1_left[j].fetch_add(1, std::memory_order_relaxed);
            }
        for (int b = 0; b < kInkHidden; b += c2)
            for (int j = 0; j < cur_nj; ++j)
                row_tasks[(size_t)n_row_tasks++] = {j, 2, b, std::min(kInkHidden, b + c2)};
        row_next.store(0, std::memory_order_relaxed);
        return true;
    }
    int cur_slot_k[kInkTopK] = {}, cur_eid[kInkTopK] = {};
    int cur_cold_slot[kInkTopK] = {-1,-1,-1,-1,-1,-1,-1,-1};
    const uint8_t* cur_cold_ptr[kInkTopK] = {};
    float cur_x[kInkHidden] = {};
    CpuVerifyGroup verify_groups[kInkExperts];
    std::vector<std::thread> workers;
    std::thread master_thr;
    int n_threads = 0;

    void eval_verify_group(int gi) noexcept {
        const auto& grp = verify_groups[gi];
        const int key = cur_layer * n_experts + grp.eid;
        const auto& hm = host_meta[key];
        const float* xs[16];
        float* ys[16];
        for (int k = 0; k < grp.count; ++k) {
            const int pid = grp.pids[k];
            const int m   = pid / kInkTopK;
            const int os  = grp.out_slots[k];
            h_v_p[os] = pid;
            xs[k] = h_v_x + (size_t)m * kInkHidden;
            ys[k] = h_v_y + (size_t)os * kInkHidden;
        }
        if (hm.body != nullptr) {
            host_eval::eval_expert_batch_f32_h(hm, xs, ys, grp.count);
        } else if (hm.cold_ssd && cold_store) {
            int cslot = -1;
            const uint64_t g_now = step_gen.load(std::memory_order_relaxed);
            const uint8_t* cptr = cold_store->demand_begin(key, cur_layer, g_now, cslot);
            if (cptr && cslot >= 0) {
                struct DwCtx { ink2::InkColdStore* cs; int slot; };
                DwCtx ctx{cold_store.get(), cslot};
                auto dw_cb = +[](void* p) {
                    auto* c = static_cast<DwCtx*>(p);
                    c->cs->wait_down(c->slot);
                };
                if (cold_store->wait_gate_up(cslot)) {
                    host_eval::eval_expert_batch_f32_h(hm, xs, ys, grp.count, cptr, dw_cb, &ctx);
                } else {
                    for (int k = 0; k < grp.count; ++k) std::memset(ys[k], 0, (size_t)kInkHidden * sizeof(float));
                }
                cold_store->unpin_slot(cslot);
            } else {
                for (int k = 0; k < grp.count; ++k) std::memset(ys[k], 0, (size_t)kInkHidden * sizeof(float));
            }
        } else {
            for (int k = 0; k < grp.count; ++k) std::memset(ys[k], 0, (size_t)kInkHidden * sizeof(float));
        }
    }

    void run_verify_groups(int layer, int ng) noexcept {
        if (ng <= 0) return;
        cur_layer = layer;
        cur_ng = ng;
        if (rtx5060ti_opt && plan_verify_row_tasks(ng)) {
            work_mode = 3;
            work_done.store(0, std::memory_order_relaxed);
            if (n_threads > 1 && n_v_tasks > 1) {
                work_gen.fetch_add(1, std::memory_order_release);
                if (workers_sleeping.load(std::memory_order_acquire) > 0) {
                    std::lock_guard<std::mutex> lk(wake_mu);
                    wake_cv.notify_all();
                }
            }
            run_verify_row_tasks();
            if (n_threads > 1 && n_v_tasks > 1) {
                while (work_done.load(std::memory_order_acquire) < n_threads - 1) std::this_thread::yield();
            }
            work_mode = 0;
            return;
        }
        work_mode = 1;
        verify_next.store(0, std::memory_order_relaxed);
        work_done.store(0, std::memory_order_relaxed);
        if (n_threads > 1 && ng > 1) {
            work_gen.fetch_add(1, std::memory_order_release);
            if (workers_sleeping.load(std::memory_order_acquire) > 0) {
                std::lock_guard<std::mutex> lk(wake_mu);
                wake_cv.notify_all();
            }
        }
        while (true) {
            const int gi = verify_next.fetch_add(1, std::memory_order_relaxed);
            if (gi >= ng) break;
            eval_verify_group(gi);
        }
        if (n_threads > 1 && ng > 1) {
            while (work_done.load(std::memory_order_acquire) < n_threads - 1) std::this_thread::yield();
        }
        work_mode = 0;
    }

    void eval_doorbell_job(int j) noexcept {
        const int key = cur_layer * n_experts + cur_eid[j];
        const auto& hm = host_meta[key];
        float* ydst = const_cast<float*>(h_bell->y[cur_slot_k[j]]);
        if (hm.body != nullptr) {
            host_eval::eval_expert_f32_h(hm, cur_x, ydst);
        } else if (hm.cold_ssd && cold_store && cur_cold_slot[j] >= 0 && cur_cold_ptr[j] != nullptr) {
            struct DwCtx { ink2::InkColdStore* cs; int slot; };
            DwCtx ctx{cold_store.get(), cur_cold_slot[j]};
            auto dw_cb = +[](void* p) {
                auto* c = static_cast<DwCtx*>(p);
                c->cs->wait_down(c->slot);
            };
            if (cold_store->wait_gate_up(cur_cold_slot[j])) {
                host_eval::eval_expert_f32_h(hm, cur_x, ydst, cur_cold_ptr[j], dw_cb, &ctx);
            } else {
                std::memset(ydst, 0, (size_t)kInkHidden * sizeof(float));
            }
        } else {
            std::memset(ydst, 0, (size_t)kInkHidden * sizeof(float));
        }
    }

    void start_workers(int num_threads = 0) {
        if (active) return;
        const int hw = ink_logical_cpus();
        int eff_hw = hw;
#ifndef _WIN32
        if (rtx5060ti_opt) {
            // Respect container cgroup v1/v2 CPU quota so spin-wait workers NEVER trigger CFS throttling stalls!
            long long q_us = -1, p_us = 100000;
            if (FILE* f = std::fopen("/sys/fs/cgroup/cpu.max", "r")) {
                char buf[64] = {};
                if (std::fgets(buf, sizeof(buf), f) && buf[0] >= '0' && buf[0] <= '9') {
                    (void)std::sscanf(buf, "%lld %lld", &q_us, &p_us);
                }
                std::fclose(f);
            } else if (FILE* fq = std::fopen("/sys/fs/cgroup/cpu/cpu.cfs_quota_us", "r")) {
                if (std::fscanf(fq, "%lld", &q_us) == 1 && q_us > 0) {
                    if (FILE* fp = std::fopen("/sys/fs/cgroup/cpu/cpu.cfs_period_us", "r")) {
                        if (std::fscanf(fp, "%lld", &p_us) != 1) p_us = 100000;
                        std::fclose(fp);
                    }
                }
                std::fclose(fq);
            }
            if (q_us > 0 && p_us > 0) {
                const int quota_cpus = std::max(2, (int)(q_us / p_us));
                eff_hw = std::min(eff_hw, std::max(2, quota_cpus - 1));
            }
        }
#endif
        n_threads = (num_threads > 0) ? num_threads : (rtx5060ti_opt ? std::min(24, std::max(2, eff_hw)) : std::min(16, std::max(2, hw / 2)));
        if (const char* e = std::getenv("STRATA_INK_THREADS")) n_threads = std::max(1, std::min(128, std::atoi(e)));   // the pool size, for sweeps
        if (const char* e = std::getenv("STRATA_INK_SPLIT")) split_experts = (e[0] != '0');
        if (const char* e = std::getenv("STRATA_L3_PF")) l3_pf_enabled = (e[0] != '0');
        if (const char* e = std::getenv("STRATA_INK_DEC_PCIE_SHARE")) dec_pcie_share = std::max(0.0, std::min(1.0, std::atof(e)));
        dec_share_user = dec_pcie_share;
        prof_on = std::getenv("STRATA_INK_CPU_PROF") != nullptr;
        if (const char* e = std::getenv("STRATA_INK_EXACT")) lane_pref = (e[0] != '0') ? 1 : 2;
        set_lane(lane_pref == 1);
        if (lane_pref != 0)
            std::fprintf(stderr, "[InkHybrid] cold experts: %s\n", lane_pref == 1 ? "copied into the card and computed by its kernels (STRATA_INK_EXACT=1: the same bits wherever an expert lives)"
                                                                                : "computed on the CPU where they are (STRATA_INK_EXACT=0)");
        INK_CK(cudaHostAlloc(&h_bell, sizeof(InkMappedDoorbell), cudaHostAllocMapped));
        std::memset(h_bell, 0, sizeof(InkMappedDoorbell));
        INK_CK(cudaHostGetDevicePointer(&d_bell, h_bell, 0));
        d_seq     = dalloc<uint32_t>(1);
        d_cur_seq = dalloc<uint32_t>(1);
        INK_CK(cudaMemset(d_seq, 0, 4));
        INK_CK(cudaMemset(d_cur_seq, 0, 4));
        h_v_x   = halloc_pinned<float>(16 * kInkHidden);
        h_v_y   = halloc_pinned<float>(16 * kInkTopK * kInkHidden);
        h_v_p   = halloc_pinned<int32_t>(16 * kInkTopK);
        h_v_idx = halloc_pinned<int32_t>(16 * kInkTopK);
        d_v_y   = dalloc<float>(16 * kInkTopK * kInkHidden);
        d_v_p   = dalloc<int32_t>(16 * kInkTopK);

        d_use_cnt   = dalloc<float>(n_layers * n_experts);
        d_want_keys = dalloc<int32_t>(64);
        d_n_want    = dalloc<int32_t>(1);
        INK_CK(cudaMemset(d_use_cnt, 0, (size_t)n_layers * n_experts * 4));
        INK_CK(cudaMemset(d_n_want, 0, 4));
        d_stat = dalloc<unsigned long long>(2);
        INK_CK(cudaMemset(d_stat, 0, 2 * sizeof(unsigned long long)));
        trace_open();
        INK_CK(cudaStreamCreateWithFlags(&promo_st, cudaStreamNonBlocking));
        INK_CK(cudaStreamCreateWithFlags(&pf_dma_st, cudaStreamNonBlocking));
        for (int b = 0; b < 2; ++b) {
            INK_CK(cudaEventCreateWithFlags(&pf_ev_dma[b], cudaEventDisableTiming));
            INK_CK(cudaEventCreateWithFlags(&pf_ev_done[b], cudaEventDisableTiming));
            d_pf_exp_tmp[b] = dalloc<InkExpertDev>(kInkExperts);
            h_pf_exp_pin[b] = halloc_pinned<InkExpertDev>(kInkExperts);
            pf_buf_busy[b]  = false;
        }

        if (rtx5060ti_opt) {
            verify_act_sh.resize((size_t)(16 * kInkTopK) * kInkInter);
            v_tasks.resize(4096);
        }
        stop_flag.store(false, std::memory_order_relaxed);
        for (int t = 1; t < n_threads; ++t) {
            workers.emplace_back([this, t]() {
                uint32_t seen = 0, pf_seen = 0;
                auto last_t = std::chrono::steady_clock::now();
                while (!stop_flag.load(std::memory_order_relaxed)) {
                    uint32_t g = work_gen.load(std::memory_order_acquire);
                    if (g == seen) {
                        if (l3_pf_enabled) {
                            const uint32_t pg = pf_gen.load(std::memory_order_acquire);
                            if (pg != pf_seen) {
                                pf_seen = pg;
                                const int pl = pf_layer, pn = pf_n;
                                for (int i = t - 1; i < pn; i += std::max(1, n_threads - 1)) {
                                    if (work_gen.load(std::memory_order_acquire) != seen || stop_flag.load(std::memory_order_relaxed)) break;
                                    const int key = pl * n_experts + pf_eids[i];
                                    if (key >= 0 && key < (int)host_meta.size() && host_meta[key].body) {
                                        const char* ptr = reinterpret_cast<const char*>(host_meta[key].body);
                                        const size_t len = host_meta[key].body_bytes;
                                        for (size_t o = 0; o < len; o += 64) {
                                            if ((o & 16383u) == 0 && work_gen.load(std::memory_order_acquire) != seen) break;
#if defined(__GNUC__) || defined(__clang__)
                                            __builtin_prefetch(ptr + o, 0, 2);
#endif
                                        }
                                    }
                                }
                                continue;
                            }
                        }
                        auto dt = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - last_t).count();
                        if (dt > 80) {
                            std::unique_lock<std::mutex> lk(wake_mu);
                            workers_sleeping.fetch_add(1, std::memory_order_release);
                            wake_cv.wait(lk, [&]() {
                                return stop_flag.load(std::memory_order_acquire) ||
                                       work_gen.load(std::memory_order_acquire) != seen;
                            });
                            workers_sleeping.fetch_sub(1, std::memory_order_release);
                            last_t = std::chrono::steady_clock::now();
                        } else {
                            std::this_thread::yield();
                        }
                        continue;
                    }
                    seen = g;
                    if (work_mode == 3) {
                        run_verify_row_tasks();
                    } else if (work_mode == 2) {
                        if (prof_on) {
                            const auto tb0 = std::chrono::steady_clock::now();
                            run_row_tasks();
                            prof_wbusy_ns.fetch_add((uint64_t)std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now() - tb0).count(), std::memory_order_relaxed);
                        } else {
                            run_row_tasks();
                        }
                    } else if (work_mode == 1) {
                        while (true) {
                            const int gi = verify_next.fetch_add(1, std::memory_order_relaxed);
                            if (gi >= cur_ng) break;
                            eval_verify_group(gi);
                        }
                    } else {
                        for (int j = t; j < cur_nj; j += n_threads) {
                            eval_doorbell_job(j);
                        }
                    }
                    work_done.fetch_add(1, std::memory_order_release);
                    last_t = std::chrono::steady_clock::now();
                }
            });
        }
        master_thr = std::thread([this]() {
            uint32_t seen = 0;
            auto last_t = std::chrono::steady_clock::now();
            while (!stop_flag.load(std::memory_order_relaxed)) {
                const uint32_t seq = h_bell->ring;
                if (seq == seen) {
                    auto dt = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - last_t).count();
                    if (dt > 15) std::this_thread::sleep_for(std::chrono::microseconds(100));
                    else std::this_thread::yield();
                    continue;
                }
                seen = seq;
                const auto prof_t0 = std::chrono::steady_clock::now();
                std::atomic_thread_fence(std::memory_order_acquire);
                cur_layer = h_bell->layer;
                cur_nj    = std::min((int)h_bell->n_jobs, kInkTopK);
                const uint64_t g_now = step_gen.load(std::memory_order_relaxed);
                for (int j = 0; j < cur_nj; ++j) {
                    cur_slot_k[j]    = h_bell->slot_k[j];
                    cur_eid[j]       = h_bell->expert_id[j];
                    cur_cold_slot[j] = -1;
                    cur_cold_ptr[j]  = nullptr;
                    const int key = cur_layer * n_experts + cur_eid[j];
                    if (cold_store && host_meta[key].cold_ssd && !host_meta[key].body) {
                        cur_cold_ptr[j] = cold_store->demand_begin(key, cur_layer, g_now, cur_cold_slot[j]);
                    }
                }
                std::memcpy(cur_x, const_cast<const float*>(h_bell->x), kInkHidden * sizeof(float));
                work_done.store(0, std::memory_order_relaxed);
                if (plan_row_tasks()) {   // the experts' rows are shared by all the threads (the latency of a visit is its slowest chunk, not its slowest expert)
                    const auto tq1 = std::chrono::steady_clock::now();
                    work_mode = 2;
                    work_gen.fetch_add(1, std::memory_order_release);
                    if (workers_sleeping.load(std::memory_order_acquire) > 0) {
                        std::lock_guard<std::mutex> lk(wake_mu);
                        wake_cv.notify_all();
                    }
                    run_row_tasks();
                    const auto tq2 = std::chrono::steady_clock::now();
                    while (work_done.load(std::memory_order_acquire) < n_threads - 1) std::this_thread::yield();
                    work_mode = 0;
                    if (prof_on) {
                        const auto tq3 = std::chrono::steady_clock::now();
                        prof_plan_ns.fetch_add((uint64_t)std::chrono::duration_cast<std::chrono::nanoseconds>(tq1 - prof_t0).count(), std::memory_order_relaxed);
                        prof_own_ns.fetch_add((uint64_t)std::chrono::duration_cast<std::chrono::nanoseconds>(tq2 - tq1).count(), std::memory_order_relaxed);
                        prof_wait_ns.fetch_add((uint64_t)std::chrono::duration_cast<std::chrono::nanoseconds>(tq3 - tq2).count(), std::memory_order_relaxed);
                    }
                } else {
                if (n_threads > 1 && cur_nj > 1) {
                    work_gen.fetch_add(1, std::memory_order_release);
                    if (workers_sleeping.load(std::memory_order_acquire) > 0) {
                        std::lock_guard<std::mutex> lk(wake_mu);
                        wake_cv.notify_all();
                    }
                }
                for (int j = 0; j < cur_nj; j += (cur_nj > 1 ? n_threads : 1)) {
                    eval_doorbell_job(j);
                }
                if (n_threads > 1 && cur_nj > 1) {
                    while (work_done.load(std::memory_order_acquire) < n_threads - 1) std::this_thread::yield();
                }
                }
                if (cold_store) {
                    for (int j = 0; j < cur_nj; ++j) {
                        if (cur_cold_slot[j] >= 0) {
                            cold_store->unpin_slot(cur_cold_slot[j]);
                            cur_cold_slot[j] = -1;
                        }
                    }
                }
                std::atomic_thread_fence(std::memory_order_seq_cst);
                h_bell->ack = seq;
                last_t = std::chrono::steady_clock::now();
                {   // CPU-pool profile (STRATA_INK_CPU_PROF): wall time of the evaluation of one visit's cold experts, and the bytes it read
                    prof_ns.fetch_add((uint64_t)std::chrono::duration_cast<std::chrono::nanoseconds>(last_t - prof_t0).count(), std::memory_order_relaxed);
                    prof_visits.fetch_add(1, std::memory_order_relaxed);
                    prof_jobs.fetch_add((uint64_t)cur_nj, std::memory_order_relaxed);
                    uint64_t by = 0;
                    for (int j = 0; j < cur_nj; ++j) by += host_meta[(size_t)(cur_layer * n_experts + cur_eid[j])].body_bytes;
                    prof_bytes.fetch_add(by, std::memory_order_relaxed);
                }
            }
        });
        active = true;
    }

    // Size of each ping-pong prefill staging buffer for experts up to max_exp_bytes (32..64 MiB, at least 1 expert).
    static size_t pf_stage_size(size_t max_exp_bytes) noexcept {
        const size_t align_b = (max_exp_bytes + 255ull) & ~255ull;
        return std::max(align_b, std::min<size_t>(64ull << 20, std::max<size_t>(16 * align_b, 32ull << 20)));
    }

    // #6617: Allocate the 2 ping-pong prefill staging buffers BEFORE d_sec4 and the host pinned arena!
    // Without them, multi-token MoE cannot stream host experts over PCIe; try full size then 16 MiB minimum.
    bool alloc_pf_stage(size_t max_exp_bytes) {
        if (d_pf_stage[0]) return true;
        const size_t align_b = (max_exp_bytes + 255ull) & ~255ull;
        size_t tries[2] = {pf_stage_size(max_exp_bytes), std::max(align_b, (size_t)(16ull << 20))};
        for (size_t b : tries) {
            if (b == 0) continue;
            if (cudaMalloc(&d_pf_stage[0], b) == cudaSuccess && cudaMalloc(&d_pf_stage[1], b) == cudaSuccess) {
                pf_stage_bytes = b;
                return true;
            }
            (void)cudaGetLastError();
            if (d_pf_stage[0]) { cudaFree(d_pf_stage[0]); d_pf_stage[0] = nullptr; }
            if (d_pf_stage[1]) { cudaFree(d_pf_stage[1]); d_pf_stage[1] = nullptr; }
        }
        pf_stage_bytes = 0;
        return false;
    }

    // Allocate ping-pong prefill streaming buffers (unless alloc_pf_stage already did) + optional dynamic decode promotion ring!
    void alloc_staging(size_t max_exp_bytes, size_t vram_avail) {
        if (!active) return;
        const size_t align_b = (max_exp_bytes + 255ull) & ~255ull;
        promo_slot_bytes = align_b;
        // Reserve 2 x 16..64 MiB ping-pong staging buffers for ultra-fast prefill DMA streaming!
        if (!d_pf_stage[0] && vram_avail > (48ull << 20)) {
            const size_t max_each = (vram_avail - (16ull << 20)) / 2;
            pf_stage_bytes = std::min<size_t>(64ull << 20, std::min(max_each, std::max<size_t>(16 * align_b, 32ull << 20)));
            pf_stage_bytes = std::max(pf_stage_bytes, align_b);
            INK_CK(cudaMalloc(&d_pf_stage[0], pf_stage_bytes));
            INK_CK(cudaMalloc(&d_pf_stage[1], pf_stage_bytes));
            vram_avail -= 2 * pf_stage_bytes;
        }
        if (d_pf_stage[0] && !sec4_is_pinned && !h_pf_bounce[0]) {
            for (int b = 0; b < 2; ++b) {
                if (cudaHostAlloc(&h_pf_bounce[b], pf_stage_bytes, cudaHostAllocDefault) != cudaSuccess) {
                    (void)cudaGetLastError();
                    h_pf_bounce[b] = nullptr;
                }
            }
        }
        // Remaining spare VRAM (> 256 MiB headroom) becomes dynamic decayed-frequency promotion ring!
        if (!tier_on && vram_avail > (320ull << 20) && align_b > 0) {
            const size_t ring_bytes = std::min<size_t>(256ull << 20, vram_avail - (256ull << 20));
            n_promo_slots = (int)(ring_bytes / align_b);
            if (n_promo_slots >= 4) {
                INK_CK(cudaMalloc(&d_promo_pool, (size_t)n_promo_slots * align_b));
                if (!sec4_is_pinned) {
                    if (cudaHostAlloc(&h_promo_stg, align_b, cudaHostAllocDefault) != cudaSuccess) {
                        (void)cudaGetLastError();
                        h_promo_stg = nullptr;
                    }
                }
                promo_owner.assign(n_promo_slots, -1);
                key_to_slot.assign(n_layers * n_experts, -1);
            } else {
                n_promo_slots = 0;
            }
        }
    }

    // Periodic decayed-frequency promotion: avoids per-token cudaStreamSynchronize stalls!
    // Fast warmup during first 32 steps (every 4 tokens, up to 16 swaps), then steady-state every 16 tokens (up to 4 swaps).
    void step_promote(uint8_t* d_sec4, InkExpertDev* const d_experts[24], cudaStream_t main_st) {
        trace_flush(false);
        if (!active || n_promo_slots <= 0) return;
        const uint32_t stp = ++promo_step;
        const bool in_warmup = (stp <= 32u);
        const uint32_t period = in_warmup ? 4u : 16u;
        if ((stp % period) != 0u) return;
        int32_t nw = 0;
        INK_CK(cudaMemcpyAsync(&nw, d_n_want, 4, cudaMemcpyDeviceToHost, main_st));
        INK_CK(cudaStreamSynchronize(main_st));
        if (nw <= 0) {
            k_ink_decay_use_cnt<<<(n_layers * n_experts + 255) / 256, 256, 0, main_st>>>(d_use_cnt, n_layers * n_experts, 0.94f);
            return;
        }
        int32_t wants[64];
        const int n_take = std::min(nw, 64);
        INK_CK(cudaMemcpy(wants, d_want_keys, (size_t)n_take * 4, cudaMemcpyDeviceToHost));
        INK_CK(cudaMemsetAsync(d_n_want, 0, 4, main_st));

        std::vector<float> h_cnt(n_layers * n_experts);
        INK_CK(cudaMemcpy(h_cnt.data(), d_use_cnt, h_cnt.size() * 4, cudaMemcpyDeviceToHost));
        std::sort(wants, wants + n_take, [&](int a, int b) { return h_cnt[a] > h_cnt[b]; });

        const int max_swaps = in_warmup ? 16 : 4;
        int promoted = 0;
        std::vector<int> pinned_cold_slots;
        const uint64_t g_now = step_gen.fetch_add(1, std::memory_order_relaxed);
        for (int i = 0; i < n_take && promoted < max_swaps; ++i) {
            const int key = wants[i];
            if (key < 0 || key >= n_layers * n_experts) continue;
            const auto& nm = host_meta[key];
            if (!nm.body && !(nm.cold_ssd && cold_store)) continue;
            if (key_to_slot[key] >= 0) continue;
            // Pick free slot or lowest-use victim
            int victim = -1;
            float vmin = h_cnt[key];
            for (int sl = 0; sl < n_promo_slots; ++sl) {
                if (promo_owner[sl] < 0) { victim = sl; break; }
                if (h_cnt[promo_owner[sl]] < vmin) { vmin = h_cnt[promo_owner[sl]]; victim = sl; }
            }
            if (victim < 0) continue;
            const uint8_t* src_body = nm.body;
            int cold_slot = -1;
            if (!src_body && nm.cold_ssd && cold_store) {
                src_body = cold_store->demand_wait(key, key / n_experts, g_now, cold_slot);
                if (!src_body) continue;
                pinned_cold_slots.push_back(cold_slot);
            }
            const int old_key = promo_owner[victim];
            uint8_t* dst_slot = d_promo_pool + (size_t)victim * promo_slot_bytes;
            if (old_key >= 0) {
                // Full-duplex VRAM -> RAM copyback for SSD-backed evicted experts so future misses hit RAM ring!
                const auto& om = host_meta[old_key];
                if (om.cold_ssd && cold_store && h_copyback_buf && om.body_bytes <= h_copyback_bytes) {
                    INK_CK(cudaMemcpyAsync(h_copyback_buf, dst_slot, om.body_bytes, cudaMemcpyDeviceToHost, promo_st));
                    INK_CK(cudaStreamSynchronize(promo_st));
                    cold_store->insert_from_copyback(old_key, h_copyback_buf, om.body_bytes);
                }
                key_to_slot[old_key] = -1;
                InkExpertDev dem{};
                dem.body_off = orig_host_off[old_key];
                dem.dn_off   = om.dn_off;
                dem.gu_code  = om.gu_code;
                dem.dn_code  = om.dn_code;
                dem.tier     = 2;
                INK_CK(cudaMemcpyAsync(&d_experts[old_key / n_experts][old_key % n_experts], &dem, sizeof(dem), cudaMemcpyHostToDevice, promo_st));
            }
            if (cold_slot < 0 && !sec4_is_pinned && h_promo_stg) {
                INK_CK(cudaStreamSynchronize(promo_st));
                std::memcpy(h_promo_stg, src_body, nm.body_bytes);
                INK_CK(cudaMemcpyAsync(dst_slot, h_promo_stg, nm.body_bytes, cudaMemcpyHostToDevice, promo_st));
            } else {
                INK_CK(cudaMemcpyAsync(dst_slot, src_body, nm.body_bytes, cudaMemcpyHostToDevice, promo_st));
            }
            InkExpertDev pro{};
            pro.body_off = (uint64_t)((intptr_t)dst_slot - (intptr_t)d_sec4);
            pro.dn_off   = nm.dn_off;
            pro.gu_code  = nm.gu_code;
            pro.dn_code  = nm.dn_code;
            pro.tier     = 0;
            INK_CK(cudaMemcpyAsync(&d_experts[key / n_experts][key % n_experts], &pro, sizeof(pro), cudaMemcpyHostToDevice, promo_st));
            promo_owner[victim] = key;
            key_to_slot[key] = victim;
            ++promoted;
        }
        if (promoted > 0) INK_CK(cudaStreamSynchronize(promo_st));
        if (cold_store) {
            for (int cs : pinned_cold_slots) cold_store->unpin_slot(cs);
        }
        k_ink_decay_use_cnt<<<(n_layers * n_experts + 255) / 256, 256, 0, main_st>>>(d_use_cnt, n_layers * n_experts, 0.97f);
    }

    // =====================================================================================================================
    // Adaptive hot-expert tier (STRATA_INK_TIER=0 turns it off and gives the older static fill + 256 MiB ring).
    //
    // The card's expert bytes become a pool of slots, one class of slots per expert size (the Unsloth quants use a few body sizes).
    // Every expert has a copy in host RAM, so a slot can change owner at any time; the experts that start in the card are the first of
    // the load order (looped middle layers first, as before). While the model runs, every routed expert adds to its usage (the decode
    // graph counts on the card, the batched path counts on the host); every few tokens InkExpertPolicy (the Qwen engine's rules, see
    // strata/core/ink_expert_policy.hpp) says which experts take the slots of the coldest ones. A swap is three stream-ordered steps on the
    // decode stream (table entry of the loser -> host, copy of the winner's bytes into the slot, table entry of the winner -> card), run
    // after a token's work and before the next token's: an expert is never changed between the three passes of one token.
    // Measured on routing traces of real chats (SQ-20, 9 requests, 2.4K tokens): 4 GB of experts in the card: static fill 0.39 hit rate,
    // main's 256 MiB ring 0.50, this tier 0.81 (phaseA-tests/rel0221/inkplace/simulate_v1.py, sweep2_v1.py).
    // =====================================================================================================================
    struct TierClass {
        size_t slot_bytes = 0;
        size_t pool_off = 0;
        int n_slots = 0;
        std::vector<int> keys;       // local index -> global key
        strata::core::InkExpertPolicy pol;
    };
    bool tier_on = false;
    std::vector<TierClass> tclass;
    std::vector<int16_t> key_class;  // global key -> class (-1 none)
    std::vector<int> key_local;      // global key -> local index inside its class
    size_t tier_pool_bytes = 0;
    int tier_rows_since = 0;
    int tier_round_rows = 4;         // a round every this many decoded tokens (rows)
    int tier_swaps_per_round = 16;   // total over the classes
    float* h_cnt_snap = nullptr;     // pinned copy of the card's cumulative usage counters
    cudaEvent_t ev_snap = nullptr;
    bool snap_pending = false;
    int snap_age = 0;
    std::vector<float> last_snap;
    std::vector<float> h_use_batch;  // routed counts of the batched path since the last round
    InkExpertDev* h_flip = nullptr;  // pinned staging of table entries (read by the copies when they run)
    int flip_idx = 0;
    static constexpr int kFlipCap = 4096;
    uint64_t tier_swaps_total = 0, tier_rounds = 0;
    bool tier_log = false;


    static bool tier_enabled_env() noexcept {
        const char* e = std::getenv("STRATA_INK_TIER");
        return !(e && e[0] == '0');
    }

    // Plan the pool. key_bytes[key] = body bytes of the expert (0 = not eligible), order_keys = load order (most wanted first),
    // budget = bytes of the pool. Returns how many experts start in the card; sets tier_pool_bytes. Allocates nothing on the card.
    int tier_plan(const std::vector<size_t>& key_bytes, const std::vector<int>& order_keys, size_t budget, const std::vector<double>& key_w) {
        tclass.clear();
        key_class.assign(key_bytes.size(), -1);
        key_local.assign(key_bytes.size(), -1);
        std::vector<size_t> sizes;
        for (size_t k = 0; k < key_bytes.size(); ++k) {
            if (key_bytes[k] == 0) continue;
            const size_t sz = (key_bytes[k] + 255ull) & ~255ull;
            int c = -1;
            for (size_t i = 0; i < sizes.size(); ++i) if (sizes[i] == sz) { c = (int)i; break; }
            if (c < 0) { c = (int)sizes.size(); sizes.push_back(sz); tclass.emplace_back(); tclass.back().slot_bytes = sz; }
            key_class[k] = (int16_t)c;
            key_local[k] = (int)tclass[(size_t)c].keys.size();
            tclass[(size_t)c].keys.push_back((int)k);
        }
        // How many slots each size class starts with: the card's bytes are shared by the classes in proportion to the traffic they carry
        // (every expert is read once per visit of its layer: 3 for the looped layers, 1 for the others), so that every class has the same share of
        // its routed bytes in the card. Inside a class the experts then follow the router.
        std::vector<double> cls_traffic(tclass.size(), 0.0);
        double traffic_total = 0.0;
        for (size_t k = 0; k < key_bytes.size(); ++k) {
            if (key_class[k] < 0) continue;
            const double w = (k < key_w.size() ? key_w[k] : 1.0) * (double)tclass[(size_t)key_class[k]].slot_bytes;
            cls_traffic[(size_t)key_class[k]] += w;
            traffic_total += w;
        }
        std::vector<int> slots_c(tclass.size(), 0);
        size_t used = 0;
        for (size_t c = 0; c < tclass.size(); ++c) {
            const double target = (double)budget * (traffic_total > 0 ? cls_traffic[c] / traffic_total : 0.0);
            slots_c[c] = std::min((int)(target / (double)tclass[c].slot_bytes), (int)tclass[c].keys.size());
            used += (size_t)slots_c[c] * tclass[c].slot_bytes;
        }
        for (bool grew = true; grew;) {   // what is left of the budget: one more slot at a time, to the class that is furthest below its share
            grew = false;
            size_t best = tclass.size();
            double best_gap = 0.0;
            for (size_t c = 0; c < tclass.size(); ++c) {
                if (slots_c[c] >= (int)tclass[c].keys.size() || used + tclass[c].slot_bytes > budget) continue;
                const double target = (double)budget * (traffic_total > 0 ? cls_traffic[c] / traffic_total : 0.0);
                const double gap = target - (double)slots_c[c] * (double)tclass[c].slot_bytes;
                if (best == tclass.size() || gap > best_gap) { best = c; best_gap = gap; }
            }
            if (best < tclass.size()) { ++slots_c[best]; used += tclass[best].slot_bytes; grew = true; }
        }
        std::vector<std::vector<int>> owners(tclass.size());
        int n_start = 0;
        for (int k : order_keys) {
            if (k < 0 || (size_t)k >= key_bytes.size() || key_class[(size_t)k] < 0) continue;
            const size_t c = (size_t)key_class[(size_t)k];
            if ((int)owners[c].size() < slots_c[c]) {
                owners[c].push_back(key_local[(size_t)k]);
                ++n_start;
            }
        }
        size_t total_slots = 0;
        for (auto& o : owners) total_slots += o.size();
        size_t off = 0;
        for (size_t c = 0; c < tclass.size(); ++c) {
            auto& tc = tclass[c];
            tc.n_slots = (int)owners[c].size();
            tc.pool_off = off;
            off += (size_t)tc.n_slots * tc.slot_bytes;
            strata::core::InkExpertPolicyCfg cfg;
            cfg.decay = 0.95f;
            cfg.min_gain = 1.0f;
            cfg.usage_floor = 1.0f;
            cfg.swaps_cap = std::max(1, (int)((double)tier_swaps_per_round * (double)tc.n_slots / (double)std::max<size_t>(1, total_slots) + 0.5));
            std::vector<int> own = owners[c];
            own.resize((size_t)tc.n_slots);
            tc.pol.init((int)tc.keys.size(), {}, own, cfg);
        }
        tier_pool_bytes = off;
        return n_start;
    }

    // Allocate the pool and the bookkeeping; call after start_workers() (needs d_use_cnt) and after the prefill staging buffers.
    bool tier_alloc() {
        if (tclass.empty() || tier_pool_bytes == 0 || !active || !d_use_cnt) return false;
        if (cudaMalloc(&d_promo_pool, tier_pool_bytes) != cudaSuccess) { (void)cudaGetLastError(); d_promo_pool = nullptr; return false; }
        key_to_slot.assign((size_t)n_layers * n_experts, -1);
        // initial owners: the policy knows slot_of for every resident key
        for (auto& tc : tclass)
            for (size_t l = 0; l < tc.keys.size(); ++l) {
                const int sl = tc.pol.slot_of((int)l);
                if (sl >= 0) key_to_slot[(size_t)tc.keys[l]] = sl;
            }
        INK_CK(cudaHostAlloc(&h_cnt_snap, (size_t)n_layers * n_experts * sizeof(float), cudaHostAllocDefault));
        std::memset(h_cnt_snap, 0, (size_t)n_layers * n_experts * sizeof(float));
        INK_CK(cudaEventCreateWithFlags(&ev_snap, cudaEventDisableTiming));
        INK_CK(cudaHostAlloc(&h_flip, (size_t)kFlipCap * sizeof(InkExpertDev), cudaHostAllocDefault));
        last_snap.assign((size_t)n_layers * n_experts, 0.0f);
        h_use_batch.assign((size_t)n_layers * n_experts, 0.0f);
        // the decode graph keeps no candidate list in this mode
        if (d_want_keys) { cudaFree(d_want_keys); d_want_keys = nullptr; }
        if (d_n_want) { cudaFree(d_n_want); d_n_want = nullptr; }
        if (const char* e = std::getenv("STRATA_INK_TIER_ROWS")) tier_round_rows = std::max(1, std::atoi(e));
        if (const char* e = std::getenv("STRATA_INK_TIER_SWAPS")) tier_swaps_per_round = std::max(1, std::atoi(e));
        tier_log = std::getenv("STRATA_INK_TIER_LOG") != nullptr;
        tier_on = true;
        n_promo_slots = 0;
        std::fprintf(stderr, "[InkTier] adaptive expert tier on: %zu classes, pool %.2f GiB, slots", tclass.size(), (double)tier_pool_bytes / 1073741824.0);
        for (auto& tc : tclass) std::fprintf(stderr, " %d x %.2f MB (of %zu experts)", tc.n_slots, (double)tc.slot_bytes / 1e6, tc.keys.size());
        std::fprintf(stderr, "; a round every %d tokens, %d swaps\n", tier_round_rows, tier_swaps_per_round);
        return true;
    }

    // Device address of the slot the expert holds at the start, nullptr when it has none.
    uint8_t* tier_slot_ptr(int key) const {
        if (key < 0 || (size_t)key >= key_class.size() || key_class[(size_t)key] < 0 || !d_promo_pool) return nullptr;
        const auto& tc = tclass[(size_t)key_class[(size_t)key]];
        const int sl = tc.pol.slot_of(key_local[(size_t)key]);
        return sl < 0 ? nullptr : d_promo_pool + tc.pool_off + (size_t)sl * tc.slot_bytes;
    }

    void tier_flip(InkExpertDev* d_entry, const InkExpertDev& v, cudaStream_t st) {
        InkExpertDev* h = &h_flip[flip_idx++ & (kFlipCap - 1)];
        *h = v;
        INK_CK(cudaMemcpyAsync(d_entry, h, sizeof(InkExpertDev), cudaMemcpyHostToDevice, st));
    }

    // One round: usage -> policy -> swaps (all enqueued on `st`, after whatever the decode stream still has to do).
    void tier_round(uint8_t* d_sec4, InkExpertDev* const d_experts[24], cudaStream_t st) {
        const size_t nk = (size_t)n_layers * n_experts;
        for (size_t k = 0; k < nk; ++k) {
            const float d = (h_cnt_snap[k] - last_snap[k]) + h_use_batch[k];
            last_snap[k] = h_cnt_snap[k];
            h_use_batch[k] = 0.0f;
            if (d > 0.0f && key_class[k] >= 0) tclass[(size_t)key_class[k]].pol.add_use(key_local[k], d);
        }
        int swaps = 0;
        for (auto& tc : tclass) {
            if (tc.n_slots <= 0) continue;
            const auto sw = tc.pol.round();
            for (const auto& s : sw) {
                const int in_g = tc.keys[(size_t)s.in_key];
                const int out_g = s.out_key >= 0 ? tc.keys[(size_t)s.out_key] : -1;
                const auto& im = host_meta[(size_t)in_g];
                if (!im.body) continue;   // no host copy: cannot happen in this mode; leave the slot as it is
                uint8_t* dst = d_promo_pool + tc.pool_off + (size_t)s.slot * tc.slot_bytes;
                if (out_g >= 0) {
                    InkExpertDev dem{};
                    const auto& om = host_meta[(size_t)out_g];
                    dem.body_off = orig_host_off[(size_t)out_g];
                    dem.dn_off = om.dn_off;
                    dem.gu_code = om.gu_code;
                    dem.dn_code = om.dn_code;
                    dem.tier = 2;
                    tier_flip(&d_experts[out_g / n_experts][out_g % n_experts], dem, st);
                    key_to_slot[(size_t)out_g] = -1;
                }
                INK_CK(cudaMemcpyAsync(dst, im.body, im.body_bytes, cudaMemcpyHostToDevice, st));
                InkExpertDev pro{};
                pro.body_off = (uint64_t)((intptr_t)dst - (intptr_t)d_sec4);
                pro.dn_off = im.dn_off;
                pro.gu_code = im.gu_code;
                pro.dn_code = im.dn_code;
                pro.tier = 0;
                tier_flip(&d_experts[in_g / n_experts][in_g % n_experts], pro, st);
                key_to_slot[(size_t)in_g] = s.slot;
                ++swaps;
            }
        }
        tier_swaps_total += (uint64_t)swaps;
        ++tier_rounds;
        if (tier_log && (tier_rounds % 64) == 1)
            std::fprintf(stderr, "[InkTier] round %llu: %d swaps (total %llu)\n", (unsigned long long)tier_rounds, swaps, (unsigned long long)tier_swaps_total);
    }

    // Called after every decode step that went through the MoE (rows = tokens in it). With the tier it counts rows and runs a round about
    // every tier_round_rows tokens; the usage counters come back by an asynchronous copy (the round runs when the copy has arrived), so
    // nothing here waits for the card. Without the tier it is the older periodic promotion (M = 1 only).
    void step_after(int rows, uint8_t* d_sec4, InkExpertDev* const d_experts[24], cudaStream_t st) {
        if (probing) return;
        trace_flush(false);
        if (!active) return;
        if (!tier_on) {
            if (rows == 1) step_promote(d_sec4, d_experts, st);
            return;
        }
        tier_rows_since += rows;
        // The placement is a function of the token sequence only (never of how fast a copy happened to be): a snapshot of the usage counters is
        // taken at one step and used at the step after the next one, where the event is waited for (it is normally long done).
        if (snap_pending && ++snap_age >= 2) {
            INK_CK(cudaEventSynchronize(ev_snap));
            tier_round(d_sec4, d_experts, st);
            snap_pending = false;
        }
        if (!snap_pending && tier_rows_since >= tier_round_rows) {
            INK_CK(cudaMemcpyAsync(h_cnt_snap, d_use_cnt, (size_t)n_layers * n_experts * sizeof(float), cudaMemcpyDeviceToHost, st));
            INK_CK(cudaEventRecord(ev_snap, st));
            snap_pending = true;
            snap_age = 0;
            tier_rows_since = 0;
        }
    }

    void free_all() {
        if (tier_on) {
            std::fprintf(stderr, "[InkTier] %llu rounds, %llu swaps\n", (unsigned long long)tier_rounds, (unsigned long long)tier_swaps_total);
            if (h_cnt_snap) { cudaFreeHost(h_cnt_snap); h_cnt_snap = nullptr; }
            if (h_flip) { cudaFreeHost(h_flip); h_flip = nullptr; }
            if (ev_snap) { cudaEventDestroy(ev_snap); ev_snap = nullptr; }
            tier_on = false;
        }
        if (trace_fp) {
            trace_flush(true);
            std::fclose(trace_fp);
            trace_fp = nullptr;
        }
        if (active) {
            stop_flag.store(true, std::memory_order_release);
            work_gen.fetch_add(1, std::memory_order_release);
            {
                std::lock_guard<std::mutex> lk(wake_mu);
            }
            wake_cv.notify_all();
            if (master_thr.joinable()) master_thr.join();
            for (auto& w : workers) if (w.joinable()) w.join();
            workers.clear();
            active = false;
        }
        auto fr = [](auto*& p) { if (p) { cudaFree(p); p = nullptr; } };
        fr(d_seq); fr(d_cur_seq); fr(d_use_cnt); fr(d_want_keys); fr(d_n_want); fr(d_stat); fr(d_trace); fr(d_trace_n);
        fr(d_promo_pool); fr(d_pf_stage[0]); fr(d_pf_stage[1]);
        fr(d_pf_exp_tmp[0]); fr(d_pf_exp_tmp[1]);
        fr(d_v_y); fr(d_v_p);
        for (int b = 0; b < 2; ++b) {
            if (h_pf_exp_pin[b]) { cudaFreeHost(h_pf_exp_pin[b]); h_pf_exp_pin[b] = nullptr; }
            if (h_pf_bounce[b])  { cudaFreeHost(h_pf_bounce[b]);  h_pf_bounce[b]  = nullptr; }
        }
        if (h_promo_stg) { cudaFreeHost(h_promo_stg); h_promo_stg = nullptr; }
        if (h_v_x) { cudaFreeHost(h_v_x); h_v_x = nullptr; }
        if (h_v_y) { cudaFreeHost(h_v_y); h_v_y = nullptr; }
        if (h_v_p) { cudaFreeHost(h_v_p); h_v_p = nullptr; }
        if (h_v_idx) { cudaFreeHost(h_v_idx); h_v_idx = nullptr; }
        if (h_copyback_buf) { cudaFreeHost(h_copyback_buf); h_copyback_buf = nullptr; h_copyback_bytes = 0; }
        cold_store.reset();
        if (h_bell) { cudaFreeHost(h_bell); h_bell = nullptr; d_bell = nullptr; }
        if (h_sec4_pinned) { cudaFreeHost(h_sec4_pinned); h_sec4_pinned = nullptr; h_sec4_bytes = 0; }
        if (h_sec4_pageable) {
#ifdef _WIN32
            VirtualFree(h_sec4_pageable, 0, MEM_RELEASE);
#else
            munmap(h_sec4_pageable, h_sec4_bytes);
#endif
            h_sec4_pageable = nullptr;
            h_sec4_bytes = 0;
        }
        sec4_is_pinned = false;
        if (promo_st) { cudaStreamDestroy(promo_st); promo_st = nullptr; }
        if (pf_dma_st) { cudaStreamDestroy(pf_dma_st); pf_dma_st = nullptr; }
        for (int b = 0; b < 2; ++b) {
            if (pf_ev_dma[b])  { cudaEventDestroy(pf_ev_dma[b]);  pf_ev_dma[b] = nullptr; }
            if (pf_ev_done[b]) { cudaEventDestroy(pf_ev_done[b]); pf_ev_done[b] = nullptr; }
        }
    }
};

__global__ void __launch_bounds__(256) k_ink_scatter_cpu_pairs(
    const float* __restrict__ src_y,       // [n_pairs, kInkHidden]
    const int32_t* __restrict__ pair_idx,  // [n_pairs]
    const float* __restrict__ topk_w,      // [M * 8]
    float* __restrict__ y_pair,            // [M * 8, kInkHidden]
    int n_pairs) {
    const int j = blockIdx.x;
    if (j >= n_pairs) return;
    const int pid = pair_idx[j];
    const float rw = topk_w[pid];
    const float4* s4 = reinterpret_cast<const float4*>(src_y + (size_t)j * kInkHidden);
    float4* d4 = reinterpret_cast<float4*>(y_pair + (size_t)pid * kInkHidden);
    for (int i = threadIdx.x; i < (kInkHidden / 4); i += 256) {
        const float4 v = s4[i];
        d4[i] = make_float4(v.x * rw, v.y * rw, v.z * rw, v.w * rw);
    }
}

// ============================================================================
// 3. Hybrid MoE Launcher: Doorbell Parallel CPU+GPU (M=1) + Simultaneous PCIe+CPU Miss-Sharing (M=2..16) + Ping-Pong DMA Streaming (M>16)!
// ============================================================================
inline void launch_ink_moe_hybrid(
    cudaStream_t st,
    cublasHandle_t cublas,
    const uint8_t* sec4_base,
    const InkExpertDev* experts,
    const InkLayerMoeInfo& info,
    int ip,
    InkHybridEngine& hyb,
    const float* gate_w,
    const float* gate_b,
    const float* x_f32,
    float* y_out,
    int M,
    InkMoeWorkspace& ws,
    float route_scale = kInkRouteScale,
    bool force_exact_dec = false) {
    // 1. Exact Top-8 / 128 sigmoid router
    if (M <= 16) {
        if (M == 1)      k_ink_router_gemv_smallM<1><<<kInkExperts, 256, 0, st>>>(x_f32, gate_w, ws.r_logits, M);
        else if (M <= 4) k_ink_router_gemv_smallM<4><<<kInkExperts, 256, 0, st>>>(x_f32, gate_w, ws.r_logits, M);
        else if (M <= 8) k_ink_router_gemv_smallM<8><<<kInkExperts, 256, 0, st>>>(x_f32, gate_w, ws.r_logits, M);
        else             k_ink_router_gemv_smallM<16><<<kInkExperts, 256, 0, st>>>(x_f32, gate_w, ws.r_logits, M);
    } else {
        const float alpha = 1.0f, beta = 0.0f;
        INK_CUBLAS_CK(cublasGemmEx(
            cublas, CUBLAS_OP_T, CUBLAS_OP_N, kInkExperts, M, kInkHidden, &alpha,
            gate_w, CUDA_R_32F, kInkHidden, x_f32, CUDA_R_32F, kInkHidden, &beta,
            ws.r_logits, CUDA_R_32F, kInkExperts,
            ink_has_int8_tc() ? CUBLAS_COMPUTE_32F_FAST_TF32 : CUBLAS_COMPUTE_32F,
            CUBLAS_GEMM_DEFAULT_TENSOR_OP));
    }
    k_ink_router_select_topk8<<<M, 32, 0, st>>>(ws.r_logits, gate_b, ws.topk_idx, ws.topk_w, route_scale, M, ink_tail_args_now(ws.y_pair, experts));   // opt-in tail skip: a dropped slot has id < 0

    if (M == 1 && hyb.exact_mode && hyb.exact_graph) {
        // exact mode: the card copies the cold experts in and computes them itself (see k_ink_stage_plan)
        k_ink_stage_plan<<<1, 32, 0, st>>>(ip, const_cast<InkExpertDev*>(experts), ws.topk_idx, hyb.d_key_bytes, hyb.d_stage, hyb.stage_slot_bytes,
                                            sec4_base, hyb.d_stage_state, hyb.d_use_cnt, hyb.d_stat);
        k_ink_stage_copy<<<dim3((unsigned)((hyb.stage_slot_bytes + 65535u) / 65536u), kInkTopK), 256, 0, st>>>(hyb.d_stage_state, hyb.h_sec4_pinned, hyb.d_stage, hyb.stage_slot_bytes);
        launch_ink_moe_dispatch_only(st, sec4_base, experts, info, x_f32, 1, ws, force_exact_dec, false);
        k_ink_stage_restore<<<1, 32, 0, st>>>(const_cast<InkExpertDev*>(experts), hyb.d_stage_state);
    } else if (M == 1 && !hyb.exact_mode) {
        // Zero-sync CUDA-Graph-capturable Doorbell: CPU DDR5 computes tier==2 experts IN PARALLEL with GPU tier==0!
        k_ink_doorbell_ring<<<1, 256, 0, st>>>(
            ip, experts, ws.topk_idx, x_f32,
            hyb.d_use_cnt, hyb.d_want_keys, hyb.d_n_want,
            hyb.d_bell, hyb.d_seq, hyb.d_cur_seq,
            hyb.d_stat, hyb.d_trace, hyb.d_trace_n, hyb.trace_cap);
        launch_ink_moe_dispatch_only(st, sec4_base, experts, info, x_f32, 1, ws, force_exact_dec, false);
        k_ink_doorbell_wait_collect<<<kInkTopK, 256, 0, st>>>(hyb.d_bell, hyb.d_cur_seq, ws.topk_w, ws.y_pair);
    } else {
        // #6617: Clear y_pair so no unwritten slot can ever carry an earlier layer's stale output!
        INK_CK(cudaMemsetAsync(ws.y_pair, 0, (size_t)M * kInkTopK * kInkHidden * sizeof(float), st));
        // Double-Buffered Ping-Pong Async DMA Streaming + Simultaneous PCIe/CPU Miss-Sharing (M=2..16) + GPU+SSD Wave Planning!
        std::vector<int> host_list;
        host_list.reserve(kInkExperts);
        std::vector<int> cpu_eids;
        std::vector<std::pair<int, int>> cpu_pairs;  // (pair_id, expert_id)
        auto is_off_gpu = [&](int key) -> bool {
            return (hyb.host_meta[key].body != nullptr || hyb.host_meta[key].cold_ssd) &&
                   (hyb.key_to_slot.empty() || hyb.key_to_slot[key] < 0);
        };
        if (M <= 16 && !hyb.ssd_only) {
            int32_t* h_idx = hyb.h_v_idx;
            int32_t h_idx_fallback[16 * kInkTopK] = {};
            if (!h_idx) h_idx = h_idx_fallback;
            int ecnt[kInkExperts] = {};
            // Opt-in RTX 5060 Ti single-sync path: copy topk_idx AND x_f32 together before 1 cudaStreamSynchronize!
            const bool opt_one_sync = hyb.rtx5060ti_opt && hyb.active && hyb.h_v_x && hyb.h_v_y && (hyb.dec_pcie_share < 0.0 || hyb.dec_pcie_share < 0.999);
            INK_CK(cudaMemcpyAsync(h_idx, ws.topk_idx, (size_t)M * kInkTopK * 4, cudaMemcpyDeviceToHost, st));
            if (opt_one_sync) {
                INK_CK(cudaMemcpyAsync(hyb.h_v_x, x_f32, (size_t)M * kInkHidden * 4, cudaMemcpyDeviceToHost, st));
            }
            INK_CK(cudaStreamSynchronize(st));
            if (hyb.trace_fp) {
                uint32_t dn = 0;
                INK_CK(cudaMemcpy(&dn, hyb.d_trace_n, 4, cudaMemcpyDeviceToHost));
                for (int m = 0; m < M; ++m) {
                    InkTraceRec r;
                    r.layer = (uint8_t)ip; r.kind = 1; r.row = (uint8_t)m; r.M = (uint8_t)M; r.seq = dn;
                    for (int k = 0; k < kInkTopK; ++k) r.ids[k] = (uint8_t)h_idx[m * kInkTopK + k];
                    hyb.trace_host.push_back(r);
                }
                hyb.trace_flush(false);
            }
            for (int i = 0; i < M * kInkTopK; ++i) {
                const int e = h_idx[i];
                const int key = ip * kInkExperts + e;
                if (e >= 0 && e < kInkExperts) {
                    ++hyb.stat_selections;
                    if (hyb.tier_on) hyb.h_use_batch[(size_t)key] += 1.0f;   // the tier learns from the batched steps too (verify, short chunks)
                }
                if (e >= 0 && e < kInkExperts && hyb.host_meta[key].body != nullptr &&
                    (hyb.key_to_slot.empty() || hyb.key_to_slot[key] < 0)) {
                    if (ecnt[e] == 0) host_list.push_back(e);
                    ++ecnt[e];
                    ++hyb.stat_misses;
                }
            }
            // Zero-cliff small-spill fast path: when <= 6 cold experts miss (~2-6 MB) and PCIe staging is ready,
            // stream ALL of them asynchronously over PCIe H2D to GPU Tensor Cores (< 0.15 ms, zero x_f32 D2H, zero CPU wait!)
            const bool stream_all_pcie = !hyb.rtx5060ti_opt && (hyb.dec_pcie_share < 0.0) && (host_list.size() <= 6) &&
                                         (hyb.d_pf_stage[0] != nullptr) && (hyb.pf_stage_bytes > 0);
            const double share = (hyb.dec_pcie_share >= 0.0) ? hyb.dec_pcie_share
                               : (stream_all_pcie ? 1.0
                                  : (hyb.auto_pcie_share >= 0.0 ? hyb.auto_pcie_share
                                     : (hyb.rtx5060ti_opt ? (double)hyb.rtx5060ti_pcie_share
                                                          : (hyb.sec4_is_pinned ? 0.65 : 0.55))));
            if (hyb.active && hyb.h_v_x && hyb.h_v_y && host_list.size() >= 2 && share < 0.999) {
                std::sort(host_list.begin(), host_list.end(), [&](int a, int b) { return ecnt[a] > ecnt[b]; });
                int w_tot = 0;
                for (int e : host_list) w_tot += ecnt[e];
                const double w_goal = share * (double)w_tot;
                int w_acc = 0;
                size_t split = 0;
                while (split < host_list.size() && (double)w_acc < w_goal) {
                    w_acc += ecnt[host_list[split]];
                    ++split;
                }
                uint8_t is_cpu[kInkExperts] = {};
                for (size_t i = split; i < host_list.size(); ++i) {
                    is_cpu[host_list[i]] = 1;
                    cpu_eids.push_back(host_list[i]);
                }
                host_list.resize(split);
                for (int p = 0; p < M * kInkTopK; ++p) {
                    const int e = h_idx[p];
                    if (e >= 0 && e < kInkExperts && is_cpu[e]) cpu_pairs.push_back({p, e});
                }
                // Copy x_f32 D2H ONLY when CPU pool actually has cold misses to evaluate (skipped when already copied in opt_one_sync)!
                if (!cpu_pairs.empty() && !opt_one_sync) {
                    INK_CK(cudaMemcpyAsync(hyb.h_v_x, x_f32, (size_t)M * kInkHidden * 4, cudaMemcpyDeviceToHost, st));
                    INK_CK(cudaStreamSynchronize(st));
                }
            }
        } else if (M <= 16 || hyb.ssd_only) {
            std::vector<int32_t> h_idx((size_t)M * kInkTopK);
            INK_CK(cudaMemcpyAsync(h_idx.data(), ws.topk_idx, (size_t)M * kInkTopK * 4, cudaMemcpyDeviceToHost, st));
            INK_CK(cudaStreamSynchronize(st));
            uint8_t seen[kInkExperts] = {};
            std::vector<int32_t> active_all;
            for (int i = 0; i < M * kInkTopK; ++i) {
                const int e = h_idx[i];
                if (e >= 0 && e < kInkExperts && !seen[e]) {
                    seen[e] = 1;
                    active_all.push_back(e);
                }
            }
            // Apply pe128 prompt-end recency wave ordering so hottest prompt-end experts stay resident in cold_store!
            const auto w_pe = strata::core::compute_pe128_scores(h_idx.data(), M, kInkTopK, kInkExperts);
            std::vector<uint8_t> is_res(kInkExperts, 0);
            for (int32_t e : active_all) {
                if (!is_off_gpu(ip * kInkExperts + e)) is_res[e] = 1;
            }
            const int cap_s = hyb.cold_store ? std::max(16, hyb.cold_store->num_slots()) : 64;
            const auto wplan = strata::core::plan_moe_waves(active_all, is_res, w_pe, cap_s);
            for (const auto& wv : wplan.cold_waves) {
                for (int32_t e : wv) host_list.push_back(e);
            }
        } else {
            for (int e = 0; e < kInkExperts; ++e) {
                if (is_off_gpu(ip * kInkExperts + e)) host_list.push_back(e);
            }
        }

        // Partition GPU-assigned host_list into ping-pong VRAM chunks fitting inside pf_stage_bytes (and cold_store slots/2 when SSD-only)!
        struct PfChunk { std::vector<int> eids; std::vector<size_t> offs; size_t total_bytes = 0; };
        std::vector<PfChunk> chunks;
        const size_t max_e_per_chunk = hyb.cold_store ? (size_t)std::max(1, hyb.cold_store->num_slots() / 2 - 2) : (size_t)kInkExperts;
        if (!host_list.empty() && hyb.d_pf_stage[0] && hyb.pf_stage_bytes > 0) {
            PfChunk cur;
            size_t used = 0;
            for (int e : host_list) {
                const size_t b = (hyb.host_meta[ip * kInkExperts + e].body_bytes + 255ull) & ~255ull;
                if ((used + b > hyb.pf_stage_bytes || cur.eids.size() >= max_e_per_chunk) && !cur.eids.empty()) {
                    cur.total_bytes = used;
                    chunks.push_back(std::move(cur));
                    cur = {};
                    used = 0;
                }
                cur.eids.push_back(e);
                cur.offs.push_back(used);
                used += b;
            }
            if (!cur.eids.empty()) {
                cur.total_bytes = used;
                chunks.push_back(std::move(cur));
            }
        }

        // Track cold_store pinned slots per ping-pong buffer so we unpin only after DMA finishes!
        if (hyb.cold_store) hyb.cold_store->notify_active();
        std::vector<int> buf_cold_slots[2];
        const uint64_t pf_gen = hyb.step_gen.fetch_add(1, std::memory_order_relaxed);
        auto unpin_buf_slots = [&](int buf) {
            if (hyb.cold_store) {
                for (int s : buf_cold_slots[buf]) hyb.cold_store->unpin_slot(s);
            }
            buf_cold_slots[buf].clear();
        };

        // Kick off Chunk `ci` DMA into ping-pong buffer `buf` (uses bounce ring cleanly when host arena is pageable!)
        auto issue_dma = [&](size_t ci, int buf) {
            if (hyb.pf_buf_busy[buf]) {
                INK_CK(cudaEventSynchronize(hyb.pf_ev_done[buf]));
                hyb.pf_buf_busy[buf] = false;
            }
            unpin_buf_slots(buf);
            const auto& ch = chunks[ci];
            InkExpertDev* h_exp = hyb.h_pf_exp_pin[buf];
            for (int e = 0; e < kInkExperts; ++e) { h_exp[e] = {}; h_exp[e].tier = 2; }
            const bool use_bounce = (!hyb.sec4_is_pinned && !hyb.ssd_only && hyb.h_pf_bounce[buf] != nullptr);
            // Pre-issue NVMe demand reads for all cold_ssd experts in this chunk in parallel!
            std::vector<const uint8_t*> ptrs(ch.eids.size(), nullptr);
            std::vector<int> cslots(ch.eids.size(), -1);
            for (size_t i = 0; i < ch.eids.size(); ++i) {
                const int key = ip * kInkExperts + ch.eids[i];
                const auto& hm = hyb.host_meta[key];
                if (hm.body) {
                    ptrs[i] = hm.body;
                } else if (hm.cold_ssd && hyb.cold_store) {
                    ptrs[i] = hyb.cold_store->demand_begin(key, ip, pf_gen, cslots[i]);
                    if (cslots[i] >= 0) buf_cold_slots[buf].push_back(cslots[i]);
                }
            }
            for (size_t i = 0; i < ch.eids.size(); ++i) {
                const int e = ch.eids[i];
                const auto& hm = hyb.host_meta[ip * kInkExperts + e];
                bool ok_io = true;
                if (cslots[i] >= 0 && hyb.cold_store) {
                    ok_io = hyb.cold_store->wait_gate_up(cslots[i]) && hyb.cold_store->wait_down(cslots[i]);
                }
                if (!ptrs[i] || !ok_io) {
                    std::fprintf(stderr, "[inkling-cuda] WARN cold_store expert key=%d ptr=%p ok_io=%d slot=%d\n",
                                 ip * kInkExperts + e, (const void*)ptrs[i], (int)ok_io, cslots[i]);
                    continue;
                }
                h_exp[e].body_off = ch.offs[i];
                h_exp[e].dn_off   = hm.dn_off;
                h_exp[e].gu_code  = hm.gu_code;
                h_exp[e].dn_code  = hm.dn_code;
                h_exp[e].tier     = 0;
                if (use_bounce) {
                    std::memcpy(hyb.h_pf_bounce[buf] + ch.offs[i], ptrs[i], hm.body_bytes);
                } else {
                    INK_CK(cudaMemcpyAsync(hyb.d_pf_stage[buf] + ch.offs[i], ptrs[i], hm.body_bytes, cudaMemcpyHostToDevice, hyb.pf_dma_st));
                }
            }
            if (use_bounce && ch.total_bytes > 0) {
                INK_CK(cudaMemcpyAsync(hyb.d_pf_stage[buf], hyb.h_pf_bounce[buf], ch.total_bytes, cudaMemcpyHostToDevice, hyb.pf_dma_st));
            }
            INK_CK(cudaMemcpyAsync(hyb.d_pf_exp_tmp[buf], h_exp, (size_t)kInkExperts * sizeof(InkExpertDev), cudaMemcpyHostToDevice, hyb.pf_dma_st));
            INK_CK(cudaEventRecord(hyb.pf_ev_dma[buf], hyb.pf_dma_st));
        };
        if (!chunks.empty()) issue_dma(0, 0);

        // Run resident tier==0 VRAM experts on main stream `st` (quantizes x once!)
        launch_ink_moe_dispatch_only(st, sec4_base, experts, info, x_f32, M, ws, force_exact_dec, false);

        // Execute each streamed host chunk on GPU Tensor Cores while pf_dma_st overlaps the NEXT chunk!
        for (size_t ci = 0; ci < chunks.size(); ++ci) {
            const int buf = (int)(ci & 1);
            if (ci + 1 < chunks.size()) {
                issue_dma(ci + 1, 1 - buf);
            }
            INK_CK(cudaStreamWaitEvent(st, hyb.pf_ev_dma[buf], 0));
            launch_ink_moe_dispatch_only(st, hyb.d_pf_stage[buf], hyb.d_pf_exp_tmp[buf], info, x_f32, M, ws, force_exact_dec, true);
            INK_CK(cudaEventRecord(hyb.pf_ev_done[buf], st));
            hyb.pf_buf_busy[buf] = true;
        }

        // Concurrently evaluate CPU-assigned pairs for M=2..16 on the PERSISTENT worker pool (ZERO std::thread spawns!)
        // Deduplicates unique cold experts across M verify tokens -> ONE single weight read per expert!
        if (!cpu_pairs.empty()) {
            const int np = (int)cpu_pairs.size();
            int eid_to_grp[kInkExperts];
            std::memset(eid_to_grp, -1, sizeof(eid_to_grp));
            int ng = 0;
            for (int j = 0; j < np; ++j) {
                const int pid = cpu_pairs[j].first;
                const int e   = cpu_pairs[j].second;
                int gi = eid_to_grp[e];
                if (gi < 0) {
                    gi = ng++;
                    eid_to_grp[e] = gi;
                    hyb.verify_groups[gi].eid = e;
                    hyb.verify_groups[gi].count = 0;
                }
                auto& grp = hyb.verify_groups[gi];
                if (grp.count < 16) {
                    const int k = grp.count++;
                    grp.pids[k] = pid;
                    grp.out_slots[k] = j;
                }
            }
            hyb.run_verify_groups(ip, ng);
            INK_CK(cudaMemcpyAsync(hyb.d_v_y, hyb.h_v_y, (size_t)np * kInkHidden * 4, cudaMemcpyHostToDevice, st));
            INK_CK(cudaMemcpyAsync(hyb.d_v_p, hyb.h_v_p, (size_t)np * 4, cudaMemcpyHostToDevice, st));
            k_ink_scatter_cpu_pairs<<<np, 256, 0, st>>>(hyb.d_v_y, hyb.d_v_p, ws.topk_w, ws.y_pair, np);
            // Record CPU misses for layer ip and kick L3 prefetch for next layer (STRATA_L3_PF)
            const int cn = std::min((int)cpu_eids.size(), 32);
            hyb.prev_cpu_n[ip] = cn;
            for (int i = 0; i < cn; ++i) hyb.prev_cpu_eids[ip][i] = cpu_eids[i];
            if (ip + 1 < kInkPhysLayers && hyb.prev_cpu_n[ip + 1] > 0) {
                hyb.kick_l3_prefetch(ip + 1, hyb.prev_cpu_eids[ip + 1], hyb.prev_cpu_n[ip + 1]);
            }
        }
        // Unpin cold_store slots after the last chunk's DMA has completed (pf_ev_dma already waited by st).
        if (hyb.cold_store && (!buf_cold_slots[0].empty() || !buf_cold_slots[1].empty())) {
            INK_CK(cudaStreamSynchronize(hyb.pf_dma_st));
            unpin_buf_slots(0);
            unpin_buf_slots(1);
        }
    }

    // 3. Deterministic 8-slot weighted sum -> y_out [M, 2304]
    dim3 rgrid((kInkHidden + 255) / 256, M);
    k_ink_moe_reduce8<<<rgrid, 256, 0, st>>>(ws.y_pair, y_out, M);
}

}  // namespace strata::ink
