// iq512_gather_parity: the 512-bit i-quant rows (iq_avx512.cpp) against ggml-cpu's own dot.
//  (a) iq512x_rows, for 1..4 tokens, is ggml-cpu's single-token vec_dot bit for bit (IQ3_XXS, IQ3_S, IQ2_S), whatever the
//      grid lookup is (scalar loads or gathers: STRATA_IQ_GATHER=0 / 1, ctest runs both);
//  (b) iq512_rows (the engine's multi-token rows) agrees with it to float rounding (another summation order);
//  (c) iq2s_pair_rows_1t / iq2s_gu_rows_1t (STRATA_IQ2S_1T, one token) are ggml-cpu's dot and SwiGLU bit for bit (needs VBMI),
//      at an even and an odd superblock count.
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
    std::printf("iq512_gather_parity: gather %d\n", (int) cpu::iq512_gather_on());
    constexpr int n = 2560, rows = 96, ntmax = 4;
    const ggml_type_traits_cpu* t8 = ggml_get_type_traits_cpu(GGML_TYPE_Q8_K);
    uint32_t seed = 987654321u;
    auto rnd = [&seed] { seed = seed * 1664525u + 1013904223u; return seed >> 8; };
    const size_t ab = ggml_row_size(GGML_TYPE_Q8_K, n);
    std::vector<std::vector<uint8_t>> act(ntmax, std::vector<uint8_t>(ab));
    const void* ap[ntmax];
    for (int t = 0; t < ntmax; ++t) {
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
        std::printf("  type %d: ok\n", type);
    }
    if (cpu::cpu_vbmi_ok()) {
        const ggml_type_traits_cpu* tr = ggml_get_type_traits_cpu(GGML_TYPE_IQ2_S);
        for (const int n2 : {2560, 2304}) {
            const size_t rb = ggml_row_size(GGML_TYPE_IQ2_S, n2), nb = (size_t) n2 / 256, bb = rb / nb;
            const size_t ab2 = ggml_row_size(GGML_TYPE_Q8_K, n2);
            std::vector<uint8_t> w((size_t) 2 * rows * rb), a2(ab2);
            for (auto& b : w) b = (uint8_t) rnd();
            for (int r = 0; r < 2 * rows; ++r)
                for (size_t k = 0; k < nb; ++k) { uint8_t* d = &w[(size_t) r * rb + k * bb]; d[0] = 0x66; d[1] = 0x2e; }
            for (int t = 0; t < ntmax; ++t) {
                std::vector<float> x(n2);
                for (auto& v : x) v = ((float) (rnd() % 2001) - 1000.f) * 0.001f * (1.f + (float) t);
                t8->from_float(x.data(), a2.data(), n2);
                std::vector<float> g((size_t) rows), u((size_t) rows), ff((size_t) rows);
                cpu::iq2s_pair_rows_1t(w.data(), rb, (size_t) rows * rb, n2, a2.data(), g.data(), u.data(), 0, rows);
                cpu::iq2s_gu_rows_1t(w.data(), rb, (size_t) rows * rb, n2, a2.data(), ff.data(), 0, rows);
                for (int r = 0; r < rows; ++r) {
                    float wg = 0.f, wu = 0.f;
                    tr->vec_dot(n2, &wg, 0, w.data() + (size_t) r * rb, 0, a2.data(), 0, 1);
                    tr->vec_dot(n2, &wu, 0, w.data() + (size_t) (rows + r) * rb, 0, a2.data(), 0, 1);
                    const float wf = (wg / (1.f + std::exp(-wg))) * wu;
                    CHECK(std::memcmp(&g[(size_t) r], &wg, 4) == 0);   // (c) bit for bit
                    CHECK(std::memcmp(&u[(size_t) r], &wu, 4) == 0);
                    CHECK(std::memcmp(&ff[(size_t) r], &wf, 4) == 0);
                }
            }
        }
        std::printf("  IQ2_S one token: ok\n");
    } else {
        std::printf("  IQ2_S one token: SKIP (no VBMI)\n");
    }
    if (g_fail == 0) std::printf("iq512_gather_parity: OK\n");
    return g_fail == 0 ? 0 : 1;
}
