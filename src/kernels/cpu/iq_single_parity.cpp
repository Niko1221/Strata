// src/kernels/cpu/iq_single_parity.cpp - one token's gate/up rows on an AVX-2 CPU without AVX-512, bit for bit
// ggml-cpu's AVX2 single-token vec_dot: iq256_gu_single_exact (IQ3_XXS, IQ3_S, IQ2_S, SwiGLU included) and the
// engine's dispatch (native_gu_rows at nt = 1, which takes it unless STRATA_CPU_SINGLE_EXACT=0).  No GPU, no model:
// random blocks with finite scales.  Exit 77 (skipped) where ggml-cpu is not the AVX2 build or the CPU has AVX-512.
#include "strata/kernels/cpu/expert_layout.hpp"
#include "strata/kernels/cpu/iq_avx2.hpp"
#include "strata/kernels/cpu/native_expert.hpp"

#include "ggml.h"
#include "ggml-cpu.h"

#include <cmath>
#include <cstdio>
#include <cstring>
#include <random>
#include <string>
#include <vector>

namespace c = strata::kernels::cpu;

namespace {
int failures = 0;
void check(const char* what, int ty, int seed, int row, float ref, float got) {
    if (std::memcmp(&ref, &got, sizeof ref) == 0) return;
    if (failures < 10) std::printf("FAIL %s type %d seed %d row %d ggml %.9g got %.9g\n", what, ty, seed, row, ref, got);
    ++failures;
}
// a finite f16 scale (2^-8 .. 2^-1) every `step` bytes from p: the block's d
void scales(uint8_t* p, size_t bytes, size_t step, std::mt19937& rng) {
    for (size_t i = 0; i + 2 <= bytes; i += step) {
        const uint16_t h = uint16_t(0x1c00 + rng() % 0x800);
        std::memcpy(p + i, &h, 2);
    }
}
}  // namespace

int main() {
    if (!c::cpu_avx2_ok() || c::cpu_avx512_ok() || !ggml_cpu_has_avx2() || ggml_cpu_has_avx512()) {
        std::puts("iq_single_parity: skipped (needs an AVX2 ggml-cpu on a CPU without AVX-512)");
        return 77;
    }
    std::mt19937 rng(42);
    constexpr int H = 2560, FF = 640, R0 = 3, R1 = FF - 1;
    for (int ty : {18, 21, 22}) {
        c::NativeFmt f;
        std::string err;
        if (!c::native_fmt(ty, 20, H, FF, f, err)) { std::printf("native_fmt: %s\n", err.c_str()); return 2; }
        const auto gdot = ggml_get_type_traits_cpu(static_cast<ggml_type>(ty))->vec_dot;
        const size_t gblk = ggml_type_size(static_cast<ggml_type>(ty));
        std::vector<uint8_t> blob(f.down_off);   // gate and up rows
        alignas(64) uint8_t act[c::kNativeActBytes];
        std::vector<float> x(H), ff(FF), ff2(FF);
        for (int seed = 0; seed < 8; ++seed) {
            for (auto& b : blob) b = uint8_t(rng());
            scales(blob.data(), f.down_off, gblk, rng);
            for (auto& v : x) v = float(int(rng() % 20001) - 10000) * 0.001f;
            c::native_quant_act(f, x.data(), act);
            c::iq256_gu_single_exact(ty, blob.data(), f.gu_row, f.up_off, H, act, ff.data(), R0, R1);
            const void* acts[] = {act};
            float* ffs[] = {ff2.data()};
            c::native_gu_rows(f, blob.data(), acts, 1, ffs, R0, R1);
            for (int r = R0; r < R1; ++r) {
                float g, u;
                gdot(H, &g, 0, blob.data() + size_t(r) * f.gu_row, 0, act, 0, 1);
                gdot(H, &u, 0, blob.data() + f.up_off + size_t(r) * f.gu_row, 0, act, 0, 1);
                const float ref = (g / (1.f + std::exp(-g))) * u;
                check("gu_single_exact", ty, seed, r, ref, ff[r]);
                check("native_gu_rows nt=1", ty, seed, r, ref, ff2[r]);
            }
        }
    }
    std::printf("iq_single_parity: %d bitwise mismatches\n", failures);
    return failures ? 1 : 0;
}
