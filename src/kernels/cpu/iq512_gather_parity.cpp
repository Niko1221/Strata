// iq512_gather_parity: the 512-bit i-quant rows (iq_avx512.cpp) against ggml-cpu's own dot.
//  (a) iq512x_rows, for 1..4 tokens, is ggml-cpu's single-token vec_dot bit for bit (IQ3_XXS, IQ3_S, IQ2_S), whatever the
//      grid lookup is (scalar loads or gathers: STRATA_IQ_GATHER=0 / 1, ctest runs both);
//  (b) iq512_rows (the engine's multi-token rows) agrees with it to float rounding (another summation order);
//  (c) iq512_gu_rows, for 1..8 tokens, is silu(gate) * up of iq512_rows' sums bit for bit.  iq512_rows never takes the
//      STRATA_IQ_FAST rows, so with STRATA_IQ_FAST=1 (a ctest of its own) this compares the block-at-once gate/up rows with
//      the default kernels, and (a) compares the block-at-once one-token rows with ggml-cpu.
// Synthetic blocks (random bytes, a fixed f16 scale), so no quantizer and no imatrix are needed.  SKIP without AVX-512.
#include "strata/kernels/cpu/expert_layout.hpp"
#include "strata/kernels/cpu/iq_avx512.hpp"

#include "ggml.h"
#include "ggml-cpu.h"

#include <cmath>
#include <cstdio>
#include <cstring>
#include <vector>

namespace cpu = strata::kernels::cpu;

static int g_fail = 0;
#define CHECK(c)                                                              \
    do {                                                                      \
        if (!(c)) { std::fprintf(stderr, "FAIL %s:%d  %s\n", __FILE__, __LINE__, #c); ++g_fail; } \
    } while (0)

int main() {
    ggml_cpu_init();
    if (!cpu::cpu_avx512_ok()) {
        std::printf("iq512_gather_parity: SKIP (no AVX-512)\n");
        return 0;
    }
    std::printf("iq512_gather_parity: gather %d, fast %d\n", (int) cpu::iq512_gather_on(), (int) cpu::iq512_fast_on());
    constexpr int n = 2560, rows = 96, ntmax = 4, ntgu = 8;
    const ggml_type_traits_cpu* t8 = ggml_get_type_traits_cpu(GGML_TYPE_Q8_K);
    uint32_t seed = 987654321u;
    auto rnd = [&seed] { seed = seed * 1664525u + 1013904223u; return seed >> 8; };
    const size_t ab = ggml_row_size(GGML_TYPE_Q8_K, n);
    std::vector<std::vector<uint8_t>> act(ntgu, std::vector<uint8_t>(ab));
    const void* ap[ntgu];
    for (int t = 0; t < ntgu; ++t) {
        std::vector<float> x(n);
        for (auto& v : x) v = ((float) (rnd() % 2001) - 1000.f) * 0.001f * (1.f + (float) t);
        t8->from_float(x.data(), act[t].data(), n);
        ap[t] = act[t].data();
    }
    for (int type : {18, 21, 22}) {
        const ggml_type_traits_cpu* tr = ggml_get_type_traits_cpu((ggml_type) type);
        CHECK(tr != nullptr && tr->vec_dot != nullptr);
        CHECK(cpu::iq512x_supported(type) && cpu::iq512_supported(type));
        const size_t rb = ggml_row_size((ggml_type) type, n), nb = (size_t) n / 256, bb = rb / nb;
        std::vector<uint8_t> w((size_t) rows * rb);
        for (auto& b : w) b = (uint8_t) rnd();
        for (int r = 0; r < rows; ++r)
            for (size_t k = 0; k < nb; ++k) { uint8_t* d = &w[(size_t) r * rb + k * bb]; d[0] = 0x66; d[1] = 0x2e; }   // f16 ~0.1
        std::vector<float> want((size_t) ntmax * rows), gx((size_t) ntmax * rows), ge((size_t) ntmax * rows);
        for (int t = 0; t < ntmax; ++t)
            for (int r = 0; r < rows; ++r)
                tr->vec_dot(n, &want[(size_t) t * rows + r], 0, w.data() + (size_t) r * rb, 0, act[t].data(), 0, 1);
        for (int nt = 1; nt <= ntmax; ++nt) {
            float* ox[ntmax];
            float* oe[ntmax];
            for (int t = 0; t < ntmax; ++t) { ox[t] = &gx[(size_t) t * rows]; oe[t] = &ge[(size_t) t * rows]; }
            cpu::iq512x_rows(type, w.data(), rb, n, ap, nt, ox, 0, rows);
            cpu::iq512_rows(type, w.data(), rb, n, ap, nt, oe, 0, rows);
            for (int t = 0; t < nt; ++t)
                for (int r = 0; r < rows; ++r) {
                    const float want_v = want[(size_t) t * rows + r];
                    CHECK(std::memcmp(&gx[(size_t) t * rows + r], &want_v, 4) == 0);   // (a) bit for bit
                    const float e = ge[(size_t) t * rows + r];
                    CHECK(std::fabs(e - want_v) <= 1e-3f * std::fabs(want_v) + 1e-3f);   // (b) float-rounding agreement
                }
        }
        // (c) gate rows = the first half of w, up rows = the second half; whole rows and a range off any vector boundary
        const int half = rows / 2;
        std::vector<float> dg((size_t) ntgu * rows), gu((size_t) ntgu * half);
        for (int nt = 1; nt <= ntgu; ++nt) {
            float* od[ntgu];
            float* og[ntgu];
            for (int t = 0; t < ntgu; ++t) { od[t] = &dg[(size_t) t * rows]; og[t] = &gu[(size_t) t * half]; }
            cpu::iq512_rows(type, w.data(), rb, n, ap, nt, od, 0, rows);
            for (int range = 0; range < 2; ++range) {
                const int r0 = range == 0 ? 0 : 3, r1 = range == 0 ? half : half - 7;
                for (auto& v : gu) v = -1.f;
                cpu::iq512_gu_rows(type, w.data(), rb, (size_t) half * rb, n, ap, nt, og, r0, r1);
                for (int t = 0; t < nt; ++t)
                    for (int r = r0; r < r1; ++r) {
                        const float g = dg[(size_t) t * rows + r], u = dg[(size_t) t * rows + half + r];
                        const float want_v = (g / (1.f + std::exp(-g))) * u;
                        CHECK(std::memcmp(&gu[(size_t) t * half + r], &want_v, 4) == 0);
                    }
            }
        }
        std::printf("  type %d: ok\n", type);
    }
    if (g_fail == 0) std::printf("iq512_gather_parity: OK\n");
    return g_fail == 0 ? 0 : 1;
}
