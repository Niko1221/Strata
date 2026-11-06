// src/kernels/cpu/arm_neon_parity.cpp - Bit-parity & accuracy verification for Strata's Arm64 NEON / FEAT_DotProd
// CPU kernels (src/kernels/cpu/portable.cpp).
//
// On x86-64 hosts, compiles portable.cpp with STRATA_ARM_NEON_SIM=1 (arm_neon_sim.hpp) inside namespace arm_sim so
// every Arm64 kernel (q8k_quant_avx2, act_quant_q8_1, bf16_rows_dot_multi, iq256_gu_rows, iq256_rows,
// iq4nl256_down_rows, q2_0_gguf_rows_multi_avx2, kq256_gu_rows, kq256_rows) is exercised and verified against
// ggml-cpu's reference implementations on every CI run without requiring physical Arm hardware.
// On aarch64 / _M_ARM64 hosts, calls the native hardware NEON / SDOT functions in portable.cpp directly.

#if defined(__x86_64__) || defined(_M_X64) || defined(__i386__) || defined(_M_IX86)
#define STRATA_ARM_NEON_SIM 1
#endif
#include "portable.cpp"
namespace arm_kern = strata::kernels::cpu;

#include "ggml.h"
#include "ggml-cpu.h"

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <limits>
#include <random>
#include <vector>

extern "C" void quantize_row_q8_K_ref(const float* x, void* y, int64_t k);

namespace {

int g_fails = 0;

#define CHECK(c)                                                                                      \
    do {                                                                                              \
        if (!(c)) {                                                                                   \
            std::fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #c);                         \
            ++g_fails;                                                                                \
        }                                                                                             \
    } while (0)

void test_q8k_quant() {
    constexpr int n = 2560;
    const size_t bytes = ggml_row_size(GGML_TYPE_Q8_K, n);
    std::mt19937 rng(42);
    std::normal_distribution<float> nd(0.f, 1.f);
    for (int ci = 0; ci < 8; ++ci) {
        std::vector<float> x(n);
        for (int j = 0; j < n; ++j) {
            if (ci == 0) x[j] = nd(rng);
            else if (ci == 1) x[j] = 0.f;
            else if (ci == 2) x[j] = (j % 256 == 13) ? -7.5f : (j % 256 == 99) ? 7.5f : nd(rng);
            else if (ci == 3) x[j] = (j % 256 == 5) ? std::numeric_limits<float>::quiet_NaN() : nd(rng);
            else if (ci == 4) x[j] = nd(rng) * 1e-3f;
            else if (ci == 5) x[j] = nd(rng) * 1e3f;
            else if (ci == 6) x[j] = (float) ((j % 255) - 127) / 127.f;
            else x[j] = (j / 256) % 2 ? nd(rng) : 0.f;
        }
        std::vector<uint8_t> ref(bytes, 0x5A), got(bytes, 0x5A);
        quantize_row_q8_K_ref(x.data(), ref.data(), n);
        arm_kern::q8k_quant_avx2(x.data(), got.data(), n);
        CHECK(std::memcmp(ref.data(), got.data(), bytes) == 0);
    }
    std::printf("  q8k_quant_arm (8 cases, incl. NaN/ties/zeros): bit-exact OK\n");
}

void test_act_quant_q8_1() {
    constexpr int n = 2048;
    std::mt19937 rng(99);
    std::normal_distribution<float> nd(0.f, 2.5f);
    std::vector<float> x(n);
    for (auto& v : x) v = nd(rng);
    strata::kernels::cpu::ActQ a{};
    arm_kern::act_quant_q8_1(x.data(), n, a);
    CHECK(a.nchunks == n / strata::kernels::cpu::QKA);
    for (int k = 0; k < a.nchunks; ++k) {
        const float* xb = x.data() + k * strata::kernels::cpu::QKA;
        float amax = 0.f;
        for (int j = 0; j < strata::kernels::cpu::QKA; ++j) amax = std::fmax(amax, std::fabs(xb[j]));
        const float s = amax > 0.f ? amax / 127.f : 0.f;
        const float inv = s > 0.f ? 1.f / s : 0.f;
        int32_t sum = 0;
        for (int j = 0; j < strata::kernels::cpu::QKA; ++j) {
            const float t = xb[j] * inv;
            int v = (int) (t + (t >= 0.f ? 0.5f : -0.5f));
            v = v < -127 ? -127 : (v > 127 ? 127 : v);
            CHECK(a.q[k * strata::kernels::cpu::QKA + j] == (int8_t) v);
            sum += v;
        }
        CHECK(a.sum[k] == sum);
    }
    std::printf("  act_quant_q8_1_arm (FCVTAS): bit-exact OK\n");
}

void test_bf16_router() {
    constexpr int rows = 64, cols = 512, nt = 5;
    uint32_t seed = 13579u;
    auto rnd = [&seed] { seed = seed * 1664525u + 1013904223u; return seed >> 8; };
    std::vector<uint16_t> w((size_t) rows * cols);
    for (auto& v : w) {
        float f = ((float) (rnd() % 2001) - 1000.f) * 0.001f;
        uint32_t bits;
        std::memcpy(&bits, &f, 4);
        v = (uint16_t) (bits >> 16);
    }
    std::vector<float> x((size_t) nt * cols), got((size_t) nt * rows);
    for (auto& v : x) v = ((float) (rnd() % 2001) - 1000.f) * 0.001f;
    arm_kern::bf16_rows_dot_multi(w.data(), rows, cols, x.data(), nt, got.data());
    for (int t = 0; t < nt; ++t) {
        for (int r = 0; r < rows; ++r) {
            double ref = 0.0;
            for (int c = 0; c < cols; ++c) {
                uint32_t b = (uint32_t) w[(size_t) r * cols + c] << 16;
                float wf;
                std::memcpy(&wf, &b, 4);
                ref += (double) wf * (double) x[(size_t) t * cols + c];
            }
            CHECK(std::fabs((double) got[(size_t) t * rows + r] - ref) <= 1e-4 * std::fabs(ref) + 1e-5);
        }
    }
    std::printf("  bf16_rows_dot_multi_arm (4-token tiled): OK\n");
}

void test_expert_formats() {
    constexpr int n = 512, rows = 16, nt = 3;
    uint32_t seed = 24680u;
    auto rnd = [&seed] { seed = seed * 1664525u + 1013904223u; return seed >> 8; };

    // Test I-quant types (16, 17, 18, 21, 22, 23) and K-quant types (12, 13, 14, 6, 7, 8) + IQ4_NL (20) + Q2_0 (42)
    for (int type : {16, 17, 18, 21, 22, 23, 12, 13, 14, 6, 7, 8, 20, 42}) {
        const ggml_type_traits_cpu* tr = ggml_get_type_traits_cpu((ggml_type) type);
        if (!tr || !tr->vec_dot) continue;
        const ggml_type_traits_cpu* t8 = ggml_get_type_traits_cpu(tr->vec_dot_type);
        CHECK(t8 != nullptr && t8->from_float != nullptr);
        const size_t rb = ggml_row_size((ggml_type) type, n);
        const size_t bb = ggml_type_size((ggml_type) type);
        const size_t nb = rb / bb;
        const size_t ab = ggml_row_size(tr->vec_dot_type, n);

        std::vector<uint8_t> blob((size_t) 2 * rows * rb);
        for (auto& b : blob) b = (uint8_t) rnd();
        for (int r = 0; r < 2 * rows; ++r) {
            for (size_t k = 0; k < nb; ++k) {
                uint8_t* d = &blob[(size_t) r * rb + k * bb];
                if (type == 14) { d[bb - 2] = 0x66; d[bb - 1] = 0x2e; }
                else {
                    d[0] = 0x66; d[1] = 0x2e;
                    if (type == 12 || type == 13 || type == 7) { d[2] = 0x66; d[3] = 0x2e; }
                }
            }
        }
        std::vector<std::vector<uint8_t>> act(nt, std::vector<uint8_t>(ab));
        const void* ap[nt];
        for (int t = 0; t < nt; ++t) {
            std::vector<float> x(n);
            for (auto& v : x) v = ((float) (rnd() % 2001) - 1000.f) * 0.001f * (1.f + 0.2f * (float) t);
            t8->from_float(x.data(), act[t].data(), n);
            ap[t] = act[t].data();
        }
        std::vector<std::vector<float>> got(nt, std::vector<float>(rows)), got_gu(nt, std::vector<float>(rows));
        std::vector<float> got1(rows), got1_gu(rows);
        float* op[nt];
        float* ffp[nt];
        for (int t = 0; t < nt; ++t) { op[t] = got[t].data(); ffp[t] = got_gu[t].data(); }
        float* op1[1] = {got1.data()};
        float* ffp1[1] = {got1_gu.data()};

        if (type == 16 || type == 17 || type == 18 || type == 21 || type == 22 || type == 23) {
            arm_kern::iq256_rows(type, blob.data(), rb, n, ap, nt, op, 0, rows);
            arm_kern::iq256_gu_rows(type, blob.data(), rb, (size_t) rows * rb, n, ap, nt, ffp, 0, rows);
            arm_kern::iq256_rows(type, blob.data(), rb, n, ap, 1, op1, 0, rows);
            arm_kern::iq256_gu_rows(type, blob.data(), rb, (size_t) rows * rb, n, ap, 1, ffp1, 0, rows);
            CHECK(std::memcmp(got[0].data(), got1.data(), rows * sizeof(float)) == 0);
            CHECK(std::memcmp(got_gu[0].data(), got1_gu.data(), rows * sizeof(float)) == 0);
        } else if (type == 20) {
            arm_kern::iq4nl256_down_rows(blob.data(), rb, n, ap, nt, op, 0, rows);
            arm_kern::iq4nl256_down_rows(blob.data(), rb, n, ap, 1, op1, 0, rows);
            CHECK(std::memcmp(got[0].data(), got1.data(), rows * sizeof(float)) == 0);
        } else if (type == 42) {
            arm_kern::q2_0_gguf_rows_multi_avx2(blob.data(), rb, n / 64, (const strata::kernels::cpu::ActQ* const*) ap, nt, op, 0, rows);
            arm_kern::q2_0_gguf_rows_multi_avx2(blob.data(), rb, n / 64, (const strata::kernels::cpu::ActQ* const*) ap, 1, op1, 0, rows);
            CHECK(std::memcmp(got[0].data(), got1.data(), rows * sizeof(float)) == 0);
        } else {
            arm_kern::kq256_rows(type, blob.data(), rb, n, ap, nt, op, 0, rows);
            arm_kern::kq256_rows(type, blob.data(), rb, n, ap, 1, op1, 0, rows);
            CHECK(std::memcmp(got[0].data(), got1.data(), rows * sizeof(float)) == 0);
            if (type != 7) {
                arm_kern::kq256_gu_rows(type, blob.data(), rb, (size_t) rows * rb, n, ap, nt, ffp, 0, rows);
                arm_kern::kq256_gu_rows(type, blob.data(), rb, (size_t) rows * rb, n, ap, 1, ffp1, 0, rows);
                CHECK(std::memcmp(got_gu[0].data(), got1_gu.data(), rows * sizeof(float)) == 0);
            }
        }

        double max_rel = 0.0;
        for (int t = 0; t < nt; ++t) {
            for (int r = 0; r < rows; ++r) {
                float want = 0.f;
                tr->vec_dot(n, &want, 0, blob.data() + (size_t) r * rb, 0, act[t].data(), 0, 1);
                const double diff = std::fabs((double) got[t][r] - (double) want);
                const double rel = diff / (std::fabs((double) want) + 1e-4);
                if (rel > max_rel) max_rel = rel;
                CHECK(rel <= 5e-5);
                if (type == 16 || type == 17 || type == 18 || type == 21 || type == 22 || type == 23 ||
                    type == 12 || type == 13 || type == 14 || type == 6 || type == 8) {
                    float wu = 0.f;
                    tr->vec_dot(n, &wu, 0, blob.data() + (size_t) (rows + r) * rb, 0, act[t].data(), 0, 1);
                    const float want_ff = (want / (1.f + std::exp(-want))) * wu;
                    const double gu_rel = std::fabs((double) got_gu[t][r] - (double) want_ff) / (std::fabs((double) want_ff) + 1e-5);
                    CHECK(gu_rel <= 1e-4);
                }
            }
        }
        std::printf("  type %2d (%-8s) nt=%d rows=%d: max_rel %.2e OK\n",
                    type, ggml_type_name((ggml_type) type), nt, rows, max_rel);
    }
}

}  // namespace

int main() {
    ggml_cpu_init();
    std::printf("arm_neon_parity: testing Arm64 NEON / FEAT_DotProd kernels...\n");
    test_q8k_quant();
    test_act_quant_q8_1();
    test_bf16_router();
    test_expert_formats();
    if (g_fails == 0) std::printf("arm_neon_parity: ALL 17 ARM64 KERNEL CHECKS PASSED\n");
    return g_fails == 0 ? 0 : 1;
}
