// src/kernels/cpu/portable_stubs.cpp - the Arm64 stand-in for Strata's x86-only CPU kernels.
//
// Strata's expert kernels are AVX-512 / AVX-2 / AVX1 translation units (expert.cpp, iq_avx*.cpp, kq_avx*.cpp,
// q2_avx*.cpp, nvfp4_avx*.cpp).  On Arm64 those files are not compiled: NEON is the baseline of every Arm64 CPU, so
// expert_layout.cpp answers cpu_avx2_ok() = true (which is what activates the NVFP4 256-bit selection in
// native_expert.cpp) and the CPU port runs the i-quant rows on ggml-cpu's NEON dot products and the NVFP4 rows on
// nvfp4_neon.cpp.  This file supplies the symbols those ARM-compiled files still name, so the static linker
// resolves every kernel the engine can call:
//
//   * the legacy s2_expert_* and the iq / kq / q8k / nvfp4_512 kernels are REFUSED (abort): they are x86-only and
//     are unreachable here behind cpu_avx512_ok() / cpu_avx1_ok() probes that are false, so the abort is never
//     taken in a normal run - it is the honest "this path does not exist on Arm64" fallback;
//   * the Q2_0 activation quantizer and row dot (act_quant_q8_1_avx2, q2_0_gguf_rows_multi_avx2) ARE reachable
//     (cpu_avx2_ok() is true), so they get portable scalar implementations with the same integer arithmetic;
//   * bf16_rows_dot / bf16_rows_dot_multi (the routing-aware prefetch's estimate) are plain portable loops.
#include "strata/kernels/cpu/expert.hpp"
#include "strata/kernels/cpu/iq_avx512.hpp"
#include "strata/kernels/cpu/iq_avx2.hpp"
#include "strata/kernels/cpu/iq_avx1.hpp"
#include "strata/kernels/cpu/kq_avx2.hpp"
#include "strata/kernels/cpu/kq_avx1.hpp"
#include "strata/kernels/cpu/nvfp4_avx512.hpp"

#if defined(__aarch64__) || defined(STRATA_CPU_ARM64)
#define STRATA_CPU_ARM64 1
#endif

#if defined(STRATA_CPU_ARM64)

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>

namespace strata::kernels::cpu {

namespace {

// fp16 -> fp32, the software form of the x86 files' _mm_cvtph_ps.
inline float fp16_to_fp32(const void* p) {
    uint16_t h;
    std::memcpy(&h, p, 2);
    const uint32_t hs = h;
    const uint32_t mantissa = hs & 0x3ffu;
    const uint32_t exponent = (hs >> 10) & 0x1fu;
    const uint32_t sign = (hs >> 15) & 1u;
    uint32_t r;
    if (exponent == 0) {
        if (mantissa == 0) {
            r = sign << 31;
        } else {
            uint32_t m = mantissa, e = 113;
            while (!(m & (1u << 10))) { m <<= 1; e--; }
            m &= (1u << 10) - 1u;
            r = (sign << 31) | (e << 23) | (m << 13);
        }
    } else if (exponent == 31) {
        r = (sign << 31) | 0x7f800000u | (mantissa << 13);
    } else {
        r = (sign << 31) | (((uint32_t) (exponent - 15 + 127)) << 23) | (mantissa << 13);
    }
    float f;
    std::memcpy(&f, &r, 4);
    return f;
}

// bf16 (the high 16 bits of an fp32) -> fp32.
inline float bf16_to_fp32(uint16_t b) {
    const uint32_t bits = (uint32_t) b << 16;
    float f;
    std::memcpy(&f, &bits, 4);
    return f;
}

// The 2-bit code (0..3) of Q2_0 value i: value i lives in code byte i/4, at bits 2*(i%4).
inline int q2_code(const uint8_t* codes, int i) {
    return (codes[i >> 2] >> (2 * (i & 3))) & 3;
}

// The Q2_0 row dot, the arithmetic of q2_avx2_rows.inl's row_multi in scalar: per 64-value weight block (one fp16
// scale, 16 code bytes) and per token, the integer dot of the codes (0..3) with the int8 activation per 32-value
// activation chunk, times (d * scale[chunk]); the -1 code offset is subtracted through ActQ::hx (scale * sum).
// The float accumulation order differs from the AVX2 kernel (a scalar row is deterministic across threads), and
// no ARM parity target compares it to x86.
void q2_rows_any(const uint8_t* w, size_t row_bytes, int nblocks, const ActQ* const* a, int nt,
                 float* const* out, int r0, int r1) {
    for (int r = r0; r < r1; ++r) {
        const uint8_t* row = w + (size_t) r * row_bytes;
        float acc[8], corr[8];
        for (int t = 0; t < nt; ++t) { acc[t] = 0.f; corr[t] = 0.f; }
        for (int b = 0; b < nblocks; ++b) {
            const uint8_t* blk = row + (size_t) b * 18;
            const float d = fp16_to_fp32(blk);
            const uint8_t* codes = blk + 2;
            for (int t = 0; t < nt; ++t) {
                const int8_t* q = a[t]->q + b * 64;
                int s0 = 0, s1 = 0;
                for (int i = 0; i < 32; ++i) {
                    s0 += q2_code(codes, i) * (int) q[i];
                    s1 += q2_code(codes, 32 + i) * (int) q[32 + i];
                }
                acc[t] += d * a[t]->scale[2 * b] * (float) s0;
                acc[t] += d * a[t]->scale[2 * b + 1] * (float) s1;
                corr[t] += d * (a[t]->hx[2 * b] + a[t]->hx[2 * b + 1]);
            }
        }
        for (int t = 0; t < nt; ++t) out[t][r] = acc[t] - corr[t];
    }
}

// The activation quantizer, the scalar rule of q2_avx2.cpp's act_quant_q8_1_avx2, bit for bit (max abs per 32-value
// chunk, fp32 scale = amax/127, round-half-away codes clamped to [-127,127]).
void act_quant_any(const float* x, int n, ActQ& a) {
    a.nchunks = n / QKA;
    for (int k = 0; k < a.nchunks; ++k) {
        const float* xb = x + k * QKA;
        float amax = 0.f;
        for (int i = 0; i < QKA; ++i) {
            const float av = xb[i] < 0.f ? -xb[i] : xb[i];
            if (av > amax) amax = av;
        }
        const float s = amax > 0.f ? amax / 127.f : 0.f;
        const float inv = s > 0.f ? 1.f / s : 0.f;
        int32_t total = 0;
        for (int i = 0; i < QKA; ++i) {
            const float t = xb[i] * inv;
            const float r = t >= 0.f ? t + 0.5f : t - 0.5f;
            int q = (int) std::trunc(r);
            if (q > 127) q = 127;
            if (q < -127) q = -127;
            a.q[k * QKA + i] = (int8_t) q;
            total += q;
        }
        a.scale[k] = s;
        a.sum[k] = total;
        a.hx[k] = s * (float) total;
    }
    a.bp_pairs = 0;
}

[[noreturn]] void refuse(const char* name) {
    std::fprintf(stderr,
                 "strata: %s is an x86 (AVX) kernel and is not built on this Arm64 CPU; refusing to run it.\n",
                 name);
    std::abort();
}

}  // namespace

// ---- reachable on Arm64 (cpu_avx2_ok() = true): real portable implementations --------------------------------

void q2_0_gguf_rows_multi_avx2(const uint8_t* w, size_t row_bytes, int nblocks, const ActQ* const* a, int nt,
                               float* const* out, int r0, int r1) {
    q2_rows_any(w, row_bytes, nblocks, a, nt, out, r0, r1);
}

void q2_0_gguf_rows_multi_avx2_legacy(const uint8_t* w, size_t row_bytes, int nblocks, const ActQ* const* a, int nt,
                                      float* const* out, int r0, int r1) {
    q2_rows_any(w, row_bytes, nblocks, a, nt, out, r0, r1);
}

void q2_0_gguf_rows_multi_avx2_v(bool /*vnni*/, const uint8_t* w, size_t row_bytes, int nblocks, const ActQ* const* a,
                                 int nt, float* const* out, int r0, int r1) {
    q2_rows_any(w, row_bytes, nblocks, a, nt, out, r0, r1);
}

void act_quant_q8_1_avx2(const float* x, int n, ActQ& a) { act_quant_any(x, n, a); }

void bf16_rows_dot(const uint16_t* w, int rows, int cols, const float* x, float* out) {
    for (int r = 0; r < rows; ++r) {
        const uint16_t* wr = w + (size_t) r * (size_t) cols;
        float acc = 0.f;
        for (int c = 0; c < cols; ++c) acc += bf16_to_fp32(wr[c]) * x[c];
        out[r] = acc;
    }
}

void bf16_rows_dot_multi(const uint16_t* w, int rows, int cols, const float* x, int nt, float* out) {
    for (int t = 0; t < nt; ++t) {
        const float* xt = x + (size_t) t * (size_t) cols;
        for (int r = 0; r < rows; ++r) {
            const uint16_t* wr = w + (size_t) r * (size_t) cols;
            float acc = 0.f;
            for (int c = 0; c < cols; ++c) acc += bf16_to_fp32(wr[c]) * xt[c];
            out[(size_t) t * (size_t) rows + r] = acc;
        }
    }
}

// ---- probes that must say "no" on Arm64 ------------------------------------------------------------------------

bool q2_bitplane_enabled() { return false; }
bool nvfp4_512_fits(int) { return false; }

// ---- REFUSED x86-only kernels (unreachable behind cpu_avx512_ok() / cpu_avx1_ok() = false) ---------------------

void q2_0_gguf_rows_multi(const uint8_t*, size_t, int, const ActQ* const*, int, float* const*, int, int) {
    refuse("q2_0_gguf_rows_multi");
}
void q2_0_gguf_rows_multi_avx1(const uint8_t*, size_t, int, const ActQ* const*, int, float* const*, int, int) {
    refuse("q2_0_gguf_rows_multi_avx1");
}
void q2_0_gguf_rows_multi_avx1_legacy(const uint8_t*, size_t, int, const ActQ* const*, int, float* const*, int, int) {
    refuse("q2_0_gguf_rows_multi_avx1_legacy");
}

void act_quant_q8_1(const float*, int, ActQ&) { refuse("act_quant_q8_1"); }
void act_quant_q8_1_avx1(const float*, int, ActQ&) { refuse("act_quant_q8_1_avx1"); }
void bf16_rows_dot_multi_avx1(const uint16_t*, int, int, const float*, int, float*) {
    refuse("bf16_rows_dot_multi_avx1");
}

void iq256_gu_rows_exact_one(int, const uint8_t*, size_t, size_t, int, const void*, float*, int, int) {
    refuse("iq256_gu_rows_exact_one");
}
void iq512_gu_rows(int, const uint8_t*, size_t, size_t, int, const void* const*, int, float* const*, int, int) {
    refuse("iq512_gu_rows");
}
void iq256_gu_rows(int, const uint8_t*, size_t, size_t, int, const void* const*, int, float* const*, int, int) {
    refuse("iq256_gu_rows");
}
// the probes of the AVX1 copies (iq_avx1.cpp): no such kernel here, so they say no; native_expert.cpp asks them
// behind cpu_avx1_ok(), which is false on Arm64, so this answer is what keeps the rows on ggml-cpu's NEON dots.
bool iq128_supported(int) noexcept { return false; }
bool iq128_down_supported(int) noexcept { return false; }
void iq128_gu_rows(int, const uint8_t*, size_t, size_t, int, const void* const*, int, float* const*, int, int) {
    refuse("iq128_gu_rows");
}
void iq128_down_rows(int, const uint8_t*, size_t, int, const void* const*, int, float* const*, int, int) {
    refuse("iq128_down_rows");
}
void iq4nl256_down_rows(const uint8_t*, size_t, int, const void* const*, int, float* const*, int, int) {
    refuse("iq4nl256_down_rows");
}
void kq256_gu_rows(int, const uint8_t*, size_t, size_t, int, const void* const*, int, float* const*, int, int) {
    refuse("kq256_gu_rows");
}
void kq256_rows(int, const uint8_t*, size_t, int, const void* const*, int, float* const*, int, int) {
    refuse("kq256_rows");
}

void nvfp4_512_gu_rows(const uint8_t*, size_t, size_t, int, const void* const*, int, float* const*, int, int, float, float) {
    refuse("nvfp4_512_gu_rows");
}
void nvfp4_512_rows(const uint8_t*, size_t, int, const void* const*, int, float* const*, int, int, float) {
    refuse("nvfp4_512_rows");
}

void s2_expert_vnni(const uint8_t*, const float*, float*, ExpertScratch&) { refuse("s2_expert_vnni"); }
void s2_expert_vnni_q(const uint8_t*, const ActQ&, float*, ExpertScratch&) { refuse("s2_expert_vnni_q"); }
void s2_expert_gu_rows(const uint8_t*, const ActQ&, float*, int, int) { refuse("s2_expert_gu_rows"); }
void s2_expert_down_rows(const uint8_t*, const ActQ&, float*, int, int) { refuse("s2_expert_down_rows"); }
void s2_expert_gu_rows_multi(const uint8_t*, const ActQ* const*, int, float* const*, int, int) {
    refuse("s2_expert_gu_rows_multi");
}
void s2_expert_down_rows_multi(const uint8_t*, const ActQ* const*, int, float* const*, int, int) {
    refuse("s2_expert_down_rows_multi");
}
void s2_expert_vnni_multi(const uint8_t*, const ActQ* const*, int, float* const*, ExpertScratchMulti&) {
    refuse("s2_expert_vnni_multi");
}
void s2_expert_scalar(const uint8_t*, const float*, float*, bool) { refuse("s2_expert_scalar"); }

}  // namespace strata::kernels::cpu

#endif   // STRATA_CPU_ARM64
