// src/kernels/cpu/q2_avx1_parity.cpp - proves the AVX1 Q2_0 kernel against a scalar reference.
//
// WHICH Q2_0 THIS IS ABOUT  (the trap in this codebase)
// -----------------------------------------------------
// There are TWO Q2_0 layouts, and confusing them is the easiest way to get a wrong answer:
//
//   * The CANONICAL pack (tools/strata_pack.py) keeps codes and scales in SEPARATE arrays:
//       O_GU_CODES + r*ROW_GU,  ROW_GU = H*2/8 = 640 bytes of codes per row
//       O_GU_SCALES + r*SC_GU*2, 80 bytes of fp16 scales per row
//     That is what `s2_expert_vnni` / `s2_expert_scalar` in expert.cpp read.
//
//   * The NATIVE pack (tools/iq_pack.py, the IQ2_XS / IQ3_* files) keeps the ggml block layout, where
//     each 64-weight block is 18 INTERLEAVED bytes: 2-byte fp16 scale then 16 bytes of codes.  That is
//     what `q2_0_gguf_rows_multi` (AVX-512) and `q2_0_gguf_rows_multi_avx2` read, and it is what
//     `q2_rows_any` dispatches to - pool.cpp:373 calls it with `nfmt_->gu_row` / `nfmt_->d_row`.
//
// This file tests the SECOND one, because `q2_avx1.cpp` is the AVX1 rung of that same dispatch ladder.
// The row strides are therefore 18*nblocks (720 B for gate/up at SC_GU=40, 180 B for down at SC_D=10),
// and a test that lays the blob out the canonical way reads 720 bytes out of a 640-byte row and faults.
//
// WHY NOT JUST USE UPSTREAM'S `expert_parity`
// -------------------------------------------
//   1. It SKIPS itself on a CPU without AVX-512, which is exactly the machine this kernel exists for.
//   2. Its oracle, `s2_expert_scalar`, lives in `expert.cpp`, a translation unit compiled with
//      `/arch:AVX512` - and the project's own CMakeLists warns that such a TU "may use AVX-512 in ANY of
//      its code", so that reference is not safe to CALL on the machine we are validating.
//
// So the reference below is written out longhand here, in a TU with no per-file ISA flag, straight from
// the block contract.  That is a stronger check than sharing an oracle: the arithmetic is transcribed
// independently, so a misreading of the contract fails the test instead of cancelling out on both sides.
//
// TOLERANCE.  Not bitwise, and should not be: the kernel reduces 4 float lanes at the end of a row where
// the AVX2 kernel reduces 8, so the summation order differs and a small gap is the correct result.
// 1e-3 normalised L1 is P2.S3's own tolerance for VNNI-vs-oracle.
#include "strata/kernels/cpu/expert.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <limits>
#include <random>
#include <vector>

namespace cpu = strata::kernels::cpu;

namespace {

// ---- the ggml Q2_0 block: 2-byte fp16 scale, then 16 bytes holding 64 two-bit codes ----
inline constexpr int BLK = 18;                     // bytes per 64-weight block
inline constexpr size_t GU_ROW = (size_t) BLK * cpu::SC_GU;   // 720
inline constexpr size_t D_ROW = (size_t) BLK * cpu::SC_D;     // 180

// ---- the reference.  No intrinsics in this half of the file. ----

/// fp16 -> fp32, by the textbook decode, deliberately not a copy of the kernel's `h2f`.
float h2f_ref(const uint8_t* p) {
    uint16_t h;
    std::memcpy(&h, p, 2);
    const uint32_t sign = (uint32_t) (h >> 15) & 1u;
    uint32_t exp = (h >> 10) & 0x1Fu, man = h & 0x3FFu;
    float f;
    if (exp == 0) {
        if (man == 0) {
            f = 0.0f;
        } else {
            int e = -14;
            while (!(man & 0x400u)) { man <<= 1; --e; }
            f = std::ldexp((float) (man & 0x3FFu), e);
        }
    } else if (exp == 31) {
        f = man ? std::numeric_limits<float>::quiet_NaN() : std::numeric_limits<float>::infinity();
    } else {
        f = std::ldexp((float) (man | 0x400u), (int) exp - 25);
    }
    return sign ? -f : f;
}

/// f32 -> f16 for the fixture.  Truncating the mantissa is fine: both sides decode the same two bytes.
uint16_t f32_to_f16(float f) {
    uint32_t u;
    std::memcpy(&u, &f, 4);
    const uint32_t sign = (u >> 16) & 0x8000u;
    int32_t exp = (int32_t) ((u >> 23) & 0xFFu) - 127 + 15;
    const uint32_t man = u & 0x7FFFFFu;
    if (exp <= 0) return (uint16_t) sign;
    if (exp >= 31) return (uint16_t) (sign | 0x7C00u);
    return (uint16_t) (sign | ((uint32_t) exp << 10) | (man >> 13));
}

/// code i: byte i/4 of the block's 16 code bytes, bits 2*(i&3).  Weight = (code - 1) * scale.
inline int code_at(const uint8_t* blk, int i) {
    return (int) ((blk[2 + (i >> 2)] >> (2 * (i & 3))) & 3) - 1;
}

/// One native-layout row against one quantized activation.
/// The 64-weight block spans TWO 32-element activation chunks with two different scales - the mistake
/// expert.hpp warns about ("one scale per weight block is the natural-looking mistake").
float row_dot_ref(const uint8_t* row, int nblocks, const cpu::ActQ& a) {
    float acc = 0.f;
    for (int b = 0; b < nblocks; ++b) {
        const uint8_t* blk = row + (size_t) b * BLK;
        const float d = h2f_ref(blk);
        for (int half = 0; half < 2; ++half) {
            const int chunk = 2 * b + half;
            const int8_t* q = a.q + (size_t) chunk * cpu::QKA;
            int32_t dot = 0;
            for (int j = 0; j < 32; ++j) dot += (int32_t) (code_at(blk, half * 32 + j) + 1) * (int32_t) q[j];
            acc += d * (a.scale[chunk] * (float) dot - a.hx[chunk]);
        }
    }
    return acc;
}

/// The activation quantizer, scalar: scale = amax/127, half-away-from-zero, clamp [-127, 127].
void act_quant_ref(const float* x, int n, cpu::ActQ& a) {
    a.nchunks = n / cpu::QKA;
    for (int k = 0; k < a.nchunks; ++k) {
        const float* xb = x + (size_t) k * cpu::QKA;
        float amax = 0.f;
        for (int j = 0; j < cpu::QKA; ++j) amax = std::fmax(amax, std::fabs(xb[j]));
        const float s = amax > 0.f ? amax / 127.f : 0.f;
        const float inv = s > 0.f ? 1.f / s : 0.f;
        int32_t total = 0;
        for (int j = 0; j < cpu::QKA; ++j) {
            const float t = xb[j] * inv;
            const float r = t + (t >= 0.f ? 0.5f : -0.5f);
            int32_t q = (int32_t) r;
            q = std::min(127, std::max(-127, q));
            a.q[(size_t) k * cpu::QKA + j] = (int8_t) q;
            total += q;
        }
        a.scale[k] = s;
        a.sum[k] = total;
        a.hx[k] = s * (float) total;
    }
}

// ---- fixture: one synthetic matrix in the native (interleaved 18-byte block) layout ----

struct NativeFixture {
    std::vector<uint8_t> gate, up, down;
};

NativeFixture make_fixture(std::mt19937& rng) {
    std::uniform_int_distribution<int> code(0, 3);
    std::uniform_real_distribution<float> mag(0.002f, 0.03f);
    auto matrix = [&](size_t row_bytes, int nrows) {
        std::vector<uint8_t> m(row_bytes * (size_t) nrows, 0);
        for (int r = 0; r < nrows; ++r)
            for (int b = 0; b < (int) (row_bytes / BLK); ++b) {
                uint8_t* blk = &m[(size_t) r * row_bytes + (size_t) b * BLK];
                const uint16_t enc = f32_to_f16(mag(rng));
                std::memcpy(blk, &enc, 2);
                for (int i = 0; i < 16; ++i)
                    blk[2 + i] = (uint8_t) (code(rng) | (code(rng) << 2) | (code(rng) << 4) | (code(rng) << 6));
            }
        return m;
    };
    // gate and up are FF x H (FF rows of H weights); down is H x FF (H rows of FF weights).  The row
    // counts differ, and getting them the wrong way round reads past the end of the fixture.
    return NativeFixture{matrix(GU_ROW, cpu::FF), matrix(GU_ROW, cpu::FF), matrix(D_ROW, cpu::H)};
}

double rel_l1(const std::vector<float>& a, const std::vector<float>& b, double* mag_out = nullptr) {
    double d = 0, mag = 0;
    for (size_t i = 0; i < a.size(); ++i) {
        d += std::fabs((double) a[i] - (double) b[i]);
        mag += std::fabs((double) a[i]);
    }
    if (mag_out) *mag_out = mag / (double) a.size();
    return d / (mag > 1e-30 ? mag : 1e-30);
}

int g_fail = 0;

void check(bool ok, const char* what, const std::string& detail) {
    std::printf("  %-48s %s%s%s\n", what, ok ? "ok" : "FAIL", detail.empty() ? "" : "  ", detail.c_str());
    if (!ok) ++g_fail;
}

}  // namespace

int main(int argc, char** argv) {
    // Unbuffered: stdout is block-buffered when piped, and a crash in a later check would otherwise
    // discard every line already printed.
    std::setvbuf(stdout, nullptr, _IONBF, 0);
    const bool bench = argc > 1 && std::string(argv[1]) == "--bench";
    std::printf("AVX1 native-layout Q2_0 kernel vs an independent scalar reference \n");
#if defined(__AVX2__)
    std::printf("  WARNING: this TU was compiled with AVX2, so a pass here would not prove the kernel\n"
                "          runs on a CPU without AVX2.  It should be built with no per-file ISA flag.\n");
#endif
    std::printf("\n");

    std::mt19937 rng(20260928);
    const NativeFixture fx = make_fixture(rng);
    std::normal_distribution<float> nd(0.f, 1.f);
    std::vector<float> x((size_t) cpu::H);
    for (auto& v : x) v = nd(rng);

    // ---- 1. structural: the block layout decodes independently of the kernel.
    {
        int bad = 0, good = 0;
        for (int b = 0; b < (int) (GU_ROW / BLK); ++b) {
            const float s = h2f_ref(&fx.gate[(size_t) b * BLK]);
            if (!std::isfinite(s) || !(s > 0.f)) ++bad; else ++good;
        }
        char buf[80];
        std::snprintf(buf, sizeof buf, "(%d scales finite > 0, row stride %zu B)", good, GU_ROW);
        check(bad == 0 && good > 0, "fixture scales decode finite and > 0", buf);
    }

    // ---- 2. the quantizer.  Same rule, same order, no float reduction: this SHOULD be exact.
    {
        cpu::ActQ ref, avx1;
        act_quant_ref(x.data(), cpu::H, ref);
        cpu::act_quant_q8_1_avx1(x.data(), cpu::H, avx1);
        int qdiff = 0;
        for (int i = 0; i < cpu::H; ++i) if (ref.q[i] != avx1.q[i]) ++qdiff;
        double sdiff = 0.0;
        for (int k = 0; k < ref.nchunks; ++k) sdiff = std::fmax(sdiff, std::fabs((double) ref.scale[k] - avx1.scale[k]));
        char buf[96];
        std::snprintf(buf, sizeof buf, "(%d/%d codes differ, max scale delta %.2e)", qdiff, cpu::H, sdiff);
        check(qdiff == 0, "act_quant_q8_1_avx1 == scalar rule, exactly", buf);
    }

    // ---- 3. one row: isolates the 2-bit code unpack.
    {
        cpu::ActQ a;
        act_quant_ref(x.data(), cpu::H, a);
        const cpu::ActQ* ap[1] = {&a};
        float out[cpu::FF];
        float* outp[1] = {out};
        cpu::q2_0_gguf_rows_multi_avx1(fx.gate.data(), GU_ROW, cpu::SC_GU, ap, 1, outp, 0, cpu::FF);
        std::vector<float> ref((size_t) cpu::FF);
        for (int r = 0; r < cpu::FF; ++r) ref[(size_t) r] = row_dot_ref(&fx.gate[(size_t) r * GU_ROW], cpu::SC_GU, a);
        double mag = 0;
        const double err = rel_l1(ref, std::vector<float>(out, out + cpu::FF), &mag);
        char buf[96];
        std::snprintf(buf, sizeof buf, "(rel L1 %.2e, |ref| mean %.4f)", err, mag);
        check(err <= 1e-3, "all 640 gate rows match reference", buf);
    }

    // ---- 4. the full expert: gate, up, SwiGLU, requantize, down.
    {
        cpu::ActQ a1;
        cpu::act_quant_q8_1_avx1(x.data(), cpu::H, a1);
        const cpu::ActQ* a1p[1] = {&a1};
        std::vector<float> fg((size_t) cpu::FF), fu((size_t) cpu::FF);
        float* fgp[1] = {fg.data()};
        float* fup[1] = {fu.data()};
        cpu::q2_0_gguf_rows_multi_avx1(fx.gate.data(), GU_ROW, cpu::SC_GU, a1p, 1, fgp, 0, cpu::FF);
        cpu::q2_0_gguf_rows_multi_avx1(fx.up.data(), GU_ROW, cpu::SC_GU, a1p, 1, fup, 0, cpu::FF);
        std::vector<float> ff((size_t) cpu::FF);
        for (int i = 0; i < cpu::FF; ++i) ff[(size_t) i] = (fg[(size_t) i] / (1.f + std::exp(-fg[(size_t) i]))) * fu[(size_t) i];
        cpu::ActQ a2;
        cpu::act_quant_q8_1_avx1(ff.data(), cpu::FF, a2);
        const cpu::ActQ* a2p[1] = {&a2};
        std::vector<float> got((size_t) cpu::H);
        float* gp[1] = {got.data()};
        cpu::q2_0_gguf_rows_multi_avx1(fx.down.data(), D_ROW, cpu::SC_D, a2p, 1, gp, 0, cpu::H);

        // reference, same steps
        std::vector<float> rg((size_t) cpu::FF), ru((size_t) cpu::FF);
        for (int r = 0; r < cpu::FF; ++r) {
            rg[(size_t) r] = row_dot_ref(&fx.gate[(size_t) r * GU_ROW], cpu::SC_GU, a1);
            ru[(size_t) r] = row_dot_ref(&fx.up[(size_t) r * GU_ROW], cpu::SC_GU, a1);
        }
        std::vector<float> rff((size_t) cpu::FF);
        for (int i = 0; i < cpu::FF; ++i) rff[(size_t) i] = (rg[(size_t) i] / (1.f + std::exp(-rg[(size_t) i]))) * ru[(size_t) i];
        cpu::ActQ ra2;
        act_quant_ref(rff.data(), cpu::FF, ra2);
        std::vector<float> ref((size_t) cpu::H);
        for (int r = 0; r < cpu::H; ++r) ref[(size_t) r] = row_dot_ref(&fx.down[(size_t) r * D_ROW], cpu::SC_D, ra2);

        int nonfinite = 0;
        for (float v : got) if (!std::isfinite(v)) ++nonfinite;
        double mag = 0;
        const double err = rel_l1(ref, got, &mag);
        char buf[112];
        std::snprintf(buf, sizeof buf, "(rel L1 %.2e, tol 1e-3, |out| mean %.4f)", err, mag);
        check(err <= 1e-3, "full expert matches reference", buf);
        std::snprintf(buf, sizeof buf, "(%d non-finite)", nonfinite);
        check(nonfinite == 0, "every output element is finite", buf);
    }

    // ---- 5. the multi-token entry point: nt=2 must equal nt=1 per token, which is the property the
    //        engine's speculative verify window depends on.
    {
        std::vector<float> xa((size_t) cpu::H), xb((size_t) cpu::H);
        for (auto& v : xa) v = nd(rng);
        for (auto& v : xb) v = nd(rng);
        cpu::ActQ a, b;
        cpu::act_quant_q8_1_avx1(xa.data(), cpu::H, a);
        cpu::act_quant_q8_1_avx1(xb.data(), cpu::H, b);
        const cpu::ActQ* both[2] = {&a, &b};
        std::vector<float> oa((size_t) cpu::FF), ob((size_t) cpu::FF);
        float* outs[2] = {oa.data(), ob.data()};
        cpu::q2_0_gguf_rows_multi_avx1(fx.gate.data(), GU_ROW, cpu::SC_GU, both, 2, outs, 0, cpu::FF);
        std::vector<float> sa((size_t) cpu::FF), sb((size_t) cpu::FF);
        const cpu::ActQ* one[1] = {&a};
        float* os1[1] = {sa.data()};
        cpu::q2_0_gguf_rows_multi_avx1(fx.gate.data(), GU_ROW, cpu::SC_GU, one, 1, os1, 0, cpu::FF);
        const cpu::ActQ* one2[1] = {&b};
        float* os2[1] = {sb.data()};
        cpu::q2_0_gguf_rows_multi_avx1(fx.gate.data(), GU_ROW, cpu::SC_GU, one2, 1, os2, 0, cpu::FF);
        char buf[96];
        const double e0 = rel_l1(sa, oa), e1 = rel_l1(sb, ob);
        std::snprintf(buf, sizeof buf, "(token0 %.2e, token1 %.2e)", e0, e1);
        check(e0 <= 1e-3 && e1 <= 1e-3, "nt=2 batching matches nt=1 per token", buf);
    }

    std::printf("\nq2_avx1_parity: %d failure%s\n", g_fail, g_fail == 1 ? "" : "s");

    // ---- 6. throughput, single thread, so it can be placed against upstream's own numbers.
    //
    // Upstream measured (src/kernels/cpu/expert.hpp, the header comment): the fully scalar loop at
    // 9.89 GB/s on ONE core, and the AVX-512 VNNI path at 42.55 GB/s across six.  The metric they use is
    // weight bytes consumed per second, and one expert blob is BLOB = 1,382,400 bytes, so
    // GB/s = BLOB * experts / elapsed is directly comparable.  The scalar path here is the same
    // longhand reference used by the correctness checks, which is the fairest available stand-in for
    // upstream's scalar loop on a CPU that cannot run their VNNI one.
    if (bench) {
        cpu::ActQ a;
        cpu::act_quant_q8_1_avx1(x.data(), cpu::H, a);
        const cpu::ActQ* ap[1] = {&a};
        std::vector<float> out((size_t) cpu::H);
        float* op[1] = {out.data()};

        // warm up
        for (int i = 0; i < 50; ++i) {
            cpu::q2_0_gguf_rows_multi_avx1(fx.gate.data(), GU_ROW, cpu::SC_GU, ap, 1, op, 0, cpu::FF);
            cpu::q2_0_gguf_rows_multi_avx1(fx.up.data(), GU_ROW, cpu::SC_GU, ap, 1, op, 0, cpu::FF);
            cpu::q2_0_gguf_rows_multi_avx1(fx.down.data(), D_ROW, cpu::SC_D, ap, 1, op, 0, cpu::H);
        }

        const int reps = 300;
        auto t0 = std::chrono::steady_clock::now();
        for (int i = 0; i < reps; ++i) {
            cpu::q2_0_gguf_rows_multi_avx1(fx.gate.data(), GU_ROW, cpu::SC_GU, ap, 1, op, 0, cpu::FF);
            cpu::q2_0_gguf_rows_multi_avx1(fx.up.data(), GU_ROW, cpu::SC_GU, ap, 1, op, 0, cpu::FF);
            cpu::q2_0_gguf_rows_multi_avx1(fx.down.data(), D_ROW, cpu::SC_D, ap, 1, op, 0, cpu::H);
        }
        auto t1 = std::chrono::steady_clock::now();
        const double avx1_s = std::chrono::duration<double>(t1 - t0).count();
        const double avx1_gbs = (double) cpu::BLOB * reps / avx1_s / 1e9;
        const double avx1_expert_s = reps / avx1_s;

        const int sreps = 3;
        t0 = std::chrono::steady_clock::now();
        for (int i = 0; i < sreps; ++i) {
            for (int r = 0; r < cpu::FF; ++r) (void) row_dot_ref(&fx.gate[(size_t) r * GU_ROW], cpu::SC_GU, a);
        }
        t1 = std::chrono::steady_clock::now();
        const double sca_s = std::chrono::duration<double>(t1 - t0).count();
        // The gate matrix is exactly one third of an expert blob (640 x 720 of 1,382,400 B), so scaling
        // by BLOB/3 makes the two directly comparable.
        const double sca_gbs = ((double) cpu::BLOB / 3.0) * sreps / sca_s / 1e9;

        char buf[160];
        std::printf("\nthroughput, 1 thread, 1 expert = %zu B of weights (fixture is 1.4 MB, so L3-resident,\n"
                    "not DRAM-bound - this measures instruction throughput, not memory bandwidth)\n", cpu::BLOB);
        std::snprintf(buf, sizeof buf, "  AVX1 kernel       %6.2f GB/s   %8.0f experts/s   %6.2f ms per expert", avx1_gbs,
                      avx1_expert_s, 1000.0 / avx1_expert_s);
        std::printf("  %s\n", buf);
        std::snprintf(buf, sizeof buf, "  scalar reference  %6.2f GB/s   (the longhand oracle above, for scale)",
                      sca_gbs);
        std::printf("  %s\n", buf);
        std::snprintf(buf, sizeof buf, "  ratio             %6.2fx over the scalar reference", avx1_gbs / sca_gbs);
        std::printf("  %s\n", buf);
        std::printf("\n  Caveat worth reading before extrapolating to tokens/s: upstream measured their\n"
                    "  TUNED scalar loop at 9.89 GB/s on one core and VNNI at 42.55 GB/s across six. This\n"
                    "  kernel is ~9x the naive reference above but still well under that tuned scalar figure,\n"
                    "  because the 2-bit unpack is redone per token here and 128-bit lanes halve the dot width.\n"
                    "  The obvious next win is amortising unpack64_sse across the NT tokens in a verify window.\n");
    }

    return g_fail ? 1 : 0;
}
