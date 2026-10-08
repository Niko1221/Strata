// Independent FP64 dequantized reference for the Q2_0 GGUF row path.
#include "strata/kernels/cpu/expert.hpp"
#include "strata/platform/cpu_arch.hpp"
#include <cmath>
#include <cstdio>
#include <cstring>
#include <random>
#include <vector>
namespace c = strata::kernels::cpu;
int main() {
    if (STRATA_CPU_X86 && !c::cpu_features().usable()) {
        std::puts("Q2_0 AVX512 row implementation unavailable on this x86 CPU");
        return 77;
    }
    c::cpu_require_expert_support();
    std::mt19937 rng(87126);
    constexpr int rows = 7;
    int failures = 0;
    // Odd block counts exercise the short SIMD tail as well as the ARM loop.
    for (int blocks : {1, 3, 10, 40}) {
        const int n = blocks * 64;
        std::vector<unsigned char> weights(rows * blocks * 18);
        for (int r = 0; r < rows; ++r) for (int b = 0; b < blocks; ++b) {
            auto* p = weights.data() + (r * blocks + b) * 18;
            const uint16_t half = 0x2400; // exactly 1/64
            std::memcpy(p, &half, 2);
            for (int j = 0; j < 16; ++j) p[2 + j] = static_cast<unsigned char>(rng());
        }
        c::ActQ acts[c::MAXT];
        const c::ActQ* input[c::MAXT];
        float output[c::MAXT][rows];
        float* dst[c::MAXT];
        std::vector<float> x(n);
        for (int t = 0; t < c::MAXT; ++t) {
            for (auto& v : x) v = float(int(rng() % 2001) - 1000) / 317;
            c::act_quant_q8_1(x.data(), n, acts[t]);
            input[t] = &acts[t]; dst[t] = output[t];
        }
        for (int nt = 1; nt <= c::MAXT; ++nt) {
            c::q2_0_gguf_rows_multi(weights.data(), blocks * 18, blocks, input, nt, dst, 0, rows);
            for (int t = 0; t < nt; ++t) for (int r = 0; r < rows; ++r) {
                double reference = 0, absolute_sum = 0;
                for (int i = 0; i < n; ++i) {
                    const auto* p = weights.data() + (r * blocks + i / 64) * 18 + 2;
                    const int code = (p[(i % 64) / 4] >> (2 * (i % 4))) & 3;
                    const double term = (code - 1) / 64.0 * acts[t].q[i] * double(acts[t].scale[i / 32]);
                    reference += term; absolute_sum += std::abs(term);
                }
                // Bound FP32 accumulation error by its absolute terms, avoiding
                // relative-error spikes when signed contributions cancel.
                if (!std::isfinite(output[t][r]) || std::abs(output[t][r] - reference) > 1e-5 * (1 + absolute_sum)) ++failures;
            }
        }
    }
    std::printf("Q2_0 rows vs FP64 dequantized reference: %d failures\n", failures);
    return failures ? 1 : 0;
}
