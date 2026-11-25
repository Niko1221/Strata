// strata/quant/dot_x4_policy.hpp - whether the native expert path uses the 4-row shared-activation dot kernels (header-only, no CPU needed to test).
//
// 4414c2988 (0.2.20) put dot_x4 kernels in front of the 1-row dots of the cold-expert path (Q4_K, Q5_K, Q6_K, Q8_0, IQ4_XS, IQ3_XXS ...).  They give the
// same bits (dot_x4_test) and share the activation loads, which pays when the weights sit in cache.  A cold expert is read from DRAM, and there the
// 4-row form is slower: dot_x4_bench, one thread, 512 MiB of rows, n = 2560 (x4 speed against four 1-row calls):
//   Ryzen 7 255 (Zen 4, AVX-512 VNNI):  Q4_0 0.77  Q8_0 0.42  Q4_K 0.26  Q5_K 0.33  Q6_K 0.24  IQ4_XS 0.27  IQ3_XXS 0.52   (cache-resident: 0.8 - 1.3)
//   i9-13900K (AVX-VNNI):               Q8_0 1.20  Q4_K 0.86  IQ4_XS 0.80
// and a Ryzen 7 7700 user (#6552) got the 0.1.65 speed back with STRATA_NO_DOT_X4=1.  So since 0.2.21 the 4-row kernels are opt-in:
// STRATA_DOT_X4=1 turns them on, STRATA_NO_DOT_X4 (any value) still turns them off and wins.
#pragma once

namespace strata::quant {

inline bool dot_x4_enabled(const char* env_on, const char* env_off) noexcept {
    if (env_off != nullptr) return false;
    return env_on != nullptr && env_on[0] == '1' && env_on[1] == '\0';
}

}  // namespace strata::quant
