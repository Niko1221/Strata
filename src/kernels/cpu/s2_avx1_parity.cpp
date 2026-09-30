// src/kernels/cpu/s2_avx1_parity.cpp - proves the CANONICAL Q2_0 expert path against a
// scalar reference written independently in a plain translation unit.
//
// WHY A SECOND TEST, AND WHY IT CANNOT REUSE UPSTREAM'S
// ------------------------------------------------------
// The first test (q2_avx1_parity.cpp) covers the NATIVE pack layout.  This one covers the CANONICAL
// layout, which is the one a recommended Q2_0 / IQ2_XS install actually produces and the one pool.cpp runs:
//     CANONICAL: codes and scales in SEPARATE arrays - ROW_GU = 640 B of codes per row (16 B per 64-weight
//                block) and SC_GU*2 = 80 B of fp16 scales per row.
//     NATIVE:    ggml-style INTERLEAVED 18-byte blocks.
// They are not interchangeable, and reading one as the other reads past the end of the row - a fault, not
// a wrong answer.
//
// Upstream cannot validate this either way: `expert_parity` skips itself on a CPU without AVX-512 (the only
// machine this port targets), and its oracle `s2_expert_scalar` is defined in `expert.cpp`, an /arch:AVX512
// translation unit, so it is not safe to CALL here - the project says so itself.  So the reference below is
// transcribed longhand in this TU, which carries no ISA flag.
//
// WHAT IS CHECKED
//   * s2_expert_vnni_q_avx1        - the full expert, which is what the pool calls for a single token
//   * s2_expert_gu_rows_avx1      - a ROW RANGE of gate/up rows (how several threads share one expert)
//   * s2_expert_down_rows_avx1    - a row range of down rows
//   * s2_expert_vnni_multi_avx1   - the multi-token form, against one single-token expert per token.  This
//     is the strongest check available: the multi path is supposed to be the SAME computation with the
//     unpack shared, so agreement here means the amortisation changed nothing observable.
//   * the dispatchers (s2_expert_*_any) choose the AVX1 port on this CPU, which is asserted, not assumed.
//
// TOLERANCE: 1e-3 normalised L1, the same number P2.S3 uses for its VNNI-vs-oracle check.  Not bitwise and
// it should not be - the AVX1 kernel reduces four float lanes where the AVX-512 kernel reduces eight, so
// the summation order differs.
#include "strata/kernels/cpu/expert.hpp"
#include "strata/kernels/cpu/expert_layout.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <limits>
#include <random>
#include <string>
#include <vector>

namespace cpu = strata::kernels::cpu;

namespace {

// ---- the reference.  No intrinsics in this half.  Transcribed from the contract in expert.hpp. ----

/// fp16 -> fp32, textbook decode, deliberately not a copy of the kernel's.
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

/// code i: byte i/4 of the row's 16-code-byte block, bits 2*(i%4).  Weight = (code - 1) * scale.
inline int code_at(const uint8_t* blk, int i) {
    return (int) ((blk[i >> 2] >> (2 * (i & 3))) & 3) - 1;
}

/// One canonical row: SEPARATE codes and scales arrays, `n` weights, `nblocks` blocks of 64.
float row_dot_ref(const uint8_t* row_codes, const uint8_t* row_scales, int n, const cpu::ActQ& a) {
    const int nblocks = n / cpu::QK;
    float acc = 0.f;
    for (int b = 0; b < nblocks; ++b) {
        const float d = h2f_ref(row_scales + 2 * b);
        const uint8_t* blk = row_codes + (size_t) b * 16;
        for (int half = 0; half < 2; ++half) {                 // a weight block spans TWO activation chunks
            const int chunk = 2 * b + half;
            const int8_t* q = a.q + (size_t) chunk * cpu::QKA;
            int32_t dot = 0;
            for (int j = 0; j < 32; ++j) dot += (int32_t) (code_at(blk, half * 32 + j) + 1) * (int32_t) q[j];
            acc += d * (a.scale[chunk] * (float) dot - a.hx[chunk]);
        }
    }
    return acc;
}

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

/// The whole expert, longhand: gate/up rows, SwiGLU on the gate, requantize, down rows.
void expert_ref(const uint8_t* blob, const float* x, float* out) {
    cpu::ActQ a1;
    act_quant_ref(x, cpu::H, a1);
    std::vector<float> ff((size_t) cpu::FF);
    for (int r = 0; r < cpu::FF; ++r) {
        const float g = row_dot_ref(blob + cpu::O_GU_CODES + (size_t) (2 * r) * cpu::ROW_GU,
                                    blob + cpu::O_GU_SCALES + (size_t) (2 * r) * cpu::SC_GU * 2, cpu::H, a1);
        const float u = row_dot_ref(blob + cpu::O_GU_CODES + (size_t) (2 * r + 1) * cpu::ROW_GU,
                                    blob + cpu::O_GU_SCALES + (size_t) (2 * r + 1) * cpu::SC_GU * 2, cpu::H, a1);
        ff[(size_t) r] = (g / (1.f + std::exp(-g))) * u;
    }
    cpu::ActQ a2;
    act_quant_ref(ff.data(), cpu::FF, a2);
    for (int r = 0; r < cpu::H; ++r)
        out[r] = row_dot_ref(blob + cpu::O_D_CODES + (size_t) r * cpu::ROW_D,
                             blob + cpu::O_D_SCALES + (size_t) r * cpu::SC_D * 2, cpu::FF, a2);
}

// ---- fixture: one synthetic expert blob in the CANONICAL layout ----

std::vector<uint8_t> make_blob(std::mt19937& rng) {
    std::vector<uint8_t> blob((size_t) cpu::BLOB, 0);
    std::uniform_int_distribution<int> code(0, 3);
    std::uniform_real_distribution<float> mag(0.002f, 0.03f);
    // 2*FF gate/up rows (gate even, up odd) in the codes array, and their own scales array.
    // NOTE the index is `r`, not `2*r`: this loop already walks the COMBINED gate+up row list, so `2*r`
    // walks off the end of the 102,400-byte scale region and past the blob.  Every CONSUMER of these
    // offsets (row_dot_ref, expert_ref, the kernel) indexes with `2*r` over r in [0, FF), which is a
    // different loop and is correct there.  Getting the two confused has bitten this file twice.
    for (int r = 0; r < 2 * cpu::FF; ++r) {
        for (int b = 0; b < cpu::SC_GU; ++b) {
            const uint16_t enc = f32_to_f16(mag(rng));
            std::memcpy(&blob[cpu::O_GU_SCALES + (size_t) r * cpu::SC_GU * 2 + 2 * b], &enc, 2);
            uint8_t* c = &blob[cpu::O_GU_CODES + (size_t) r * cpu::ROW_GU + (size_t) b * 16];
            for (int i = 0; i < 16; ++i)
                c[i] = (uint8_t) (code(rng) | (code(rng) << 2) | (code(rng) << 4) | (code(rng) << 6));
        }
    }
    for (int r = 0; r < cpu::H; ++r) {
        for (int b = 0; b < cpu::SC_D; ++b) {
            const uint16_t enc = f32_to_f16(mag(rng));
            std::memcpy(&blob[cpu::O_D_SCALES + (size_t) r * cpu::SC_D * 2 + 2 * b], &enc, 2);
            uint8_t* c = &blob[cpu::O_D_CODES + (size_t) r * cpu::ROW_D + (size_t) b * 16];
            for (int i = 0; i < 16; ++i)
                c[i] = (uint8_t) (code(rng) | (code(rng) << 2) | (code(rng) << 4) | (code(rng) << 6));
        }
    }
    return blob;
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
    std::printf("  %-50s %s%s%s\n", what, ok ? "ok" : "FAIL", detail.empty() ? "" : "  ", detail.c_str());
    if (!ok) ++g_fail;
}

std::string fmt(const char* label, double err, double tol) {
    char b[112];
    std::snprintf(b, sizeof b, "(rel L1 %.2e, tol %.0e)", err, tol);
    return std::string(label) + " " + b;
}

}  // namespace

int main(int argc, char** argv) {
    std::setvbuf(stdout, nullptr, _IONBF, 0);
    const bool bench = argc > 1 && std::string(argv[1]) == "--bench";
    std::printf("CANONICAL Q2_0 expert path, AVX1 vs an independent scalar reference \n");
#if defined(__AVX512F__) || defined(__AVX2__)
    std::printf("  WARNING: this TU has a per-file ISA flag, so a pass would not prove the kernel runs on a\n"
                "          CPU without it.  It must be built with no per-file ISA flag.\n");
#endif
    std::printf("\n");

    std::mt19937 rng(20260928);
    const std::vector<uint8_t> blob = make_blob(rng);
    std::normal_distribution<float> nd(0.f, 1.f);
    std::vector<float> x((size_t) cpu::H);
    for (auto& v : x) v = nd(rng);

    // ---- 0. the dispatch must actually pick the AVX1 port on this CPU.  Asserted, not assumed: if the
    //         probe were wrong the rest of the run would be testing the AVX-512 code and trapping anyway,
    //         but saying so explicitly makes the reason for the failure obvious.
    {
        const bool a512 = cpu::cpu_avx512_ok();
        const bool a2 = cpu::cpu_avx2_ok();
        const bool a1 = cpu::cpu_avx1_ok();
        char b[160];
        std::snprintf(b, sizeof b, "(avx512=%d avx2=%d avx1=%d)", (int) a512, (int) a2, (int) a1);
        check(!a512 && !a2 && a1, "this CPU selects the AVX1 rung (no AVX-512, no AVX2)", b);
    }

    // ---- 1. structural: scales decode independently of the kernel.
    {
        int bad = 0, good = 0;
        for (int r = 0; r < 2 * cpu::FF; ++r)
            for (int b = 0; b < cpu::SC_GU; ++b) {
                // `r`, not `2*r` - same combined-row list as make_blob.  See the note there.
                const float s = h2f_ref(&blob[cpu::O_GU_SCALES + (size_t) r * cpu::SC_GU * 2 + 2 * b]);
                if (!std::isfinite(s) || !(s > 0.f)) ++bad; else ++good;
            }
        char b[80];
        std::snprintf(b, sizeof b, "(%d scales finite > 0, row %zu B codes / %zu B scales)", good,
                      (size_t) cpu::ROW_GU, (size_t) (cpu::SC_GU * 2));
        check(bad == 0 && good > 0, "fixture scales decode finite and > 0", b);
    }

    // ---- 2. the full single-token expert, against the reference.  This is the path the pool uses per
    //         token, so it is the one that has to be right.
    {
        std::vector<float> ref((size_t) cpu::H), got((size_t) cpu::H);
        expert_ref(blob.data(), x.data(), ref.data());
        cpu::ActQ a1;
        act_quant_ref(x.data(), cpu::H, a1);
        cpu::ExpertScratch ws;
        cpu::s2_expert_vnni_q_avx1(blob.data(), a1, got.data(), ws);
        int nonfinite = 0;
        for (float v : got) if (!std::isfinite(v)) ++nonfinite;
        double mag = 0;
        const double err = rel_l1(ref, got, &mag);
        check(err <= 1e-3, "s2_expert_vnni_q_avx1 vs reference",
              fmt("", err, 1e-3));
        check(nonfinite == 0, "every output element is finite",
              std::to_string(nonfinite) + " non-finite");
    }

    // ---- 3. the row-range forms, which is how several threads share one expert.
    {
        cpu::ActQ a1;
        act_quant_ref(x.data(), cpu::H, a1);
        const int r0 = 100, r1 = 220;
        std::vector<float> ff((size_t) cpu::FF, -12345.f);
        cpu::s2_expert_gu_rows_avx1(blob.data(), a1, ff.data(), r0, r1);
        // reference for the same range
        std::vector<float> ref((size_t) cpu::FF);
        for (int r = r0; r < r1; ++r) {
            const float g = row_dot_ref(blob.data() + cpu::O_GU_CODES + (size_t) (2 * r) * cpu::ROW_GU,
                                        blob.data() + cpu::O_GU_SCALES + (size_t) (2 * r) * cpu::SC_GU * 2,
                                        cpu::H, a1);
            const float u = row_dot_ref(blob.data() + cpu::O_GU_CODES + (size_t) (2 * r + 1) * cpu::ROW_GU,
                                        blob.data() + cpu::O_GU_SCALES + (size_t) (2 * r + 1) * cpu::SC_GU * 2,
                                        cpu::H, a1);
            ref[(size_t) r] = (g / (1.f + std::exp(-g))) * u;
        }
        std::vector<float> got(ff.begin() + r0, ff.begin() + r1), want(ref.begin() + r0, ref.begin() + r1);
        double err = rel_l1(want, got);
        // and the rows OUTSIDE the range must be untouched - a kernel that wrote the whole row range
        // would still pass the comparison above and would race with the other threads in the real pool.
        int clobbered = 0;
        for (int r = 0; r < cpu::FF; ++r)
            if ((r < r0 || r >= r1) && ff[(size_t) r] != -12345.f) ++clobbered;
        check(err <= 1e-3, "s2_expert_gu_rows_avx1 over [r0,r1)", fmt("", err, 1e-3));
        check(clobbered == 0, "rows outside [r0,r1) untouched", std::to_string(clobbered) + " clobbered");

        std::vector<float> dd((size_t) cpu::H, -999.f);
        cpu::s2_expert_down_rows_avx1(blob.data(), a1, dd.data(), 0, 64);
        std::vector<float> dref((size_t) 64);
        for (int r = 0; r < 64; ++r)
            dref[(size_t) r] = row_dot_ref(blob.data() + cpu::O_D_CODES + (size_t) r * cpu::ROW_D,
                                           blob.data() + cpu::O_D_SCALES + (size_t) r * cpu::SC_D * 2, cpu::FF, a1);
        std::vector<float> dgot(dd.begin(), dd.begin() + 64);
        err = rel_l1(dref, dgot);
        check(err <= 1e-3, "s2_expert_down_rows_avx1 over [0,64)", fmt("", err, 1e-3));
    }

    // ---- 4. the multi-token form against one single-token expert per token.  The multi path is supposed
    //         to be the SAME computation with the 2-bit unpack shared across tokens, so agreement here is
    //         the strongest statement available about the amortisation.
    {
        std::vector<float> xa((size_t) cpu::H), xb((size_t) cpu::H);
        for (auto& v : xa) v = nd(rng);
        for (auto& v : xb) v = nd(rng);
        cpu::ActQ a, b;
        act_quant_ref(xa.data(), cpu::H, a);
        act_quant_ref(xb.data(), cpu::H, b);
        const cpu::ActQ* as[2] = {&a, &b};
        std::vector<float> oa((size_t) cpu::H), ob((size_t) cpu::H);
        float* outs[2] = {oa.data(), ob.data()};
        cpu::ExpertScratchMulti ws;
        cpu::s2_expert_vnni_multi_avx1(blob.data(), as, 2, outs, ws);
        std::vector<float> sa((size_t) cpu::H), sb((size_t) cpu::H);
        cpu::ExpertScratch ws1;
        cpu::s2_expert_vnni_q_avx1(blob.data(), a, sa.data(), ws1);
        cpu::s2_expert_vnni_q_avx1(blob.data(), b, sb.data(), ws1);
        const double e0 = rel_l1(sa, oa), e1 = rel_l1(sb, ob);
        char bb[112];
        std::snprintf(bb, sizeof bb, "(token0 %.2e, token1 %.2e)", e0, e1);
        check(e0 <= 1e-3 && e1 <= 1e-3, "s2_expert_vnni_multi_avx1 (nt=2) == nt=1 per token", bb);
    }

    // ---- 5. the DISPATCHERS must be the ones the pool calls, and must take the AVX1 branch here.  Driven
    //         through them so the wiring pool.cpp relies on is covered, not just the kernels.
    {
        std::vector<float> ref((size_t) cpu::H), got((size_t) cpu::H);
        expert_ref(blob.data(), x.data(), ref.data());
        cpu::ActQ a1;
        act_quant_ref(x.data(), cpu::H, a1);
        cpu::ExpertScratch ws;
        cpu::s2_expert_vnni_q_any(blob.data(), a1, got.data(), ws);
        const double err = rel_l1(ref, got);
        check(err <= 1e-3, "s2_expert_vnni_q_any (what pool.cpp calls)", fmt("", err, 1e-3));
    }

    std::printf("\ns2_avx1_parity: %d failure%s\n", g_fail, g_fail == 1 ? "" : "s");

    if (bench) {
        // Single-threaded throughput of the canonical path, in the same units the AVX-512 header quotes
        // (weight bytes per second; one expert blob is BLOB bytes).
        cpu::ActQ a1;
        act_quant_ref(x.data(), cpu::H, a1);
        std::vector<float> out((size_t) cpu::H);
        cpu::ExpertScratch ws;
        for (int i = 0; i < 20; ++i) cpu::s2_expert_vnni_q_avx1(blob.data(), a1, out.data(), ws);
        const int reps = 200;
        const auto t0 = std::chrono::steady_clock::now();
        for (int i = 0; i < reps; ++i) cpu::s2_expert_vnni_q_avx1(blob.data(), a1, out.data(), ws);
        const double s = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
        std::printf("\ncanonical path, 1 thread, 1 expert = %zu B of weights (fixture 1.4 MB, so L3-resident)\n"
                    "  s2_expert_vnni_q_avx1  %6.2f GB/s  %8.0f experts/s  %6.2f ms per expert\n",
                    (size_t) cpu::BLOB, (double) cpu::BLOB * reps / s / 1e9, (double) reps / s, 1000.0 * s / reps);
        std::printf("  upstream's AVX-512 figures for the same work: 42.55 GB/s across 6 cores with VNNI.\n"
                    "  This CPU has no VNNI, so that number is not reachable here.\n");
    }

    return g_fail ? 1 : 0;
}
