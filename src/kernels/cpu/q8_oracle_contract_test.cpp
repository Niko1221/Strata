// Compare the optional packed-Q2 activation contract with pinned ggml's
// actual CPU quantizer, including zeros, small scales and rounding ties.
#include "strata/kernels/cpu/expert.hpp"
#include "strata/platform/cpu_arch.hpp"
#include "ggml.h"
#include "ggml-cpu.h"
#include <cmath>
#include <cstdio>
#include <cstring>
#include <random>
#include <vector>
namespace c = strata::kernels::cpu;
int main() {
    if (STRATA_CPU_X86 && !c::cpu_features().usable()) return 77;
    ggml_cpu_init();
    c::expert_set_oracle_q8_0(true);
    const auto* trait = ggml_get_type_traits_cpu(GGML_TYPE_Q8_0);
    std::mt19937 rng(87126);
    c::ActQ actual;
    std::vector<float> x(c::H);
    std::vector<unsigned char> expected(c::H / 32 * 34);
    int failures = 0;
    for (int trial = 0; trial < 64; ++trial) {
        for (int i = 0; i < c::H; ++i) {
            const int block = i / 32;
            if (trial == 0) x[i] = 0;
            else if (trial == 1) x[i] = (i % 32 == 31) ? 127.f : float(i % 31 - 15) + 0.5f;
            else x[i] = float(int(rng() % 20001) - 10000) * std::ldexp(1.f / 317.f, block % 16 - 12);
        }
        trait->from_float(x.data(), expected.data(), c::H);
        c::act_quant_q8_1(x.data(), c::H, actual);
        for (int b = 0; b < c::H / 32; ++b) {
            uint16_t half;
            std::memcpy(&half, expected.data() + b * 34, 2);
            const float scale = ggml_fp16_to_fp32(half);
            const auto* codes = reinterpret_cast<const int8_t*>(expected.data() + b * 34 + 2);
            int sum = 0;
            for (int j = 0; j < 32; ++j) sum += codes[j];
            if (std::memcmp(codes, actual.q + b * 32, 32) != 0 || scale != actual.scale[b] ||
                sum != actual.sum[b] || scale * float(sum) != actual.hx[b]) ++failures;
        }
    }
    c::expert_set_oracle_q8_0(false);
    std::printf("Q8_0 activation contract vs pinned ggml CPU: %d mismatched blocks\n", failures);
    return failures ? 1 : 0;
}
