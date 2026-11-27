// strata/quant/dot_x4_policy.hpp - whether the native expert path uses the 4-row shared-activation dot kernels (header-only, no CPU needed to test).
//
// 4414c2988 (0.2.20) put dot_x4 kernels in front of the 1-row dots of the cold-expert path (Q4_K, Q5_K, Q6_K, Q8_0, IQ4_XS, IQ3_XXS ...).  They give the
// same bits (dot_x4_test) and share the activation loads, which pays when the weights sit in cache.  A cold expert is read from DRAM, and there the
// 4-row form is slower: dot_x4_bench, one thread, 512 MiB of rows, n = 2560 (x4 speed against four 1-row calls):
//   Ryzen 7 255 (Zen 4, AVX-512 VNNI):  Q4_0 0.77  Q8_0 0.42  Q4_K 0.26  Q5_K 0.33  Q6_K 0.24  IQ4_XS 0.27  IQ3_XXS 0.52   (cache-resident: 0.8 - 1.3)
//   i9-13900K (AVX-VNNI):               Q8_0 1.20  Q4_K 0.86  IQ4_XS 0.80
// and a Ryzen 7 7700 user (#6552) got the 0.1.65 speed back with STRATA_NO_DOT_X4=1.  0.2.21 made the 4-row kernels opt-in (STRATA_DOT_X4=1).
//
// #7075 measured the other side on 2x RTX 5090 with PCIe-streamed NVFP4 and four slots: with the kernels off, warm batched decode lost ~15% and
// short prompt reads ~8% against 0.1.66; STRATA_DOT_X4=1 recovered both.  Solo decode (one token per expert) was better with them off.  The work
// differs by the tokens that share the expert in one call (nt): with one token every row is read once, cold, and the 1-row dot is the faster form
// (kq_cpu_bench, the engine's own expert entry points, one thread: Ryzen 7 255, 2 GiB of blobs read cold, Q4_K / Q8_0 pairs: 25 and 33 GB/s with
// the 1-row dots, 8 and 13 with the 4-row form; i9-13900K, Q8_0 cache-resident: 33 against 14; 8 threads, DRAM-bound: 0.87 to 1.0);
// from two tokens the same row is used again from cache and the two forms are level on the Ryzen (0.98 to 1.02 at 2 to 6 tokens, cold and hot)
// and the 4-row form is ahead where the 1-row path slows down with the token count (i9-13900K, Q8_0, 3 tokens: 13 GB/s against 7 to 10;
// 6 tokens: 5 against 2).  So since 0.2.23 the default follows the call:
//   STRATA_NO_DOT_X4 set (any value)   off, whatever nt (wins over everything, as before)
//   STRATA_DOT_X4=1                    on for every nt (as in 0.2.21 / 0.2.22 with the switch)
//   STRATA_DOT_X4=0                    off for every nt (exactly the 0.2.21 / 0.2.22 default)
//   unset / anything else              on from kDotX4MinTokens tokens in the call, off for one token
// The two forms give the same bits (dot_x4_test, cpu_dispatch_dump), so the choice moves speed only.
#pragma once

namespace strata::quant {

/// tokens sharing one expert in one call, from which the 4-row kernels are the default
constexpr int kDotX4MinTokens = 2;

/// 0 = off, 1 = on, -1 = follow the number of tokens (read once by the caller)
inline int dot_x4_mode(const char* env_on, const char* env_off) noexcept {
    if (env_off != nullptr) return 0;
    if (env_on != nullptr && env_on[0] == '1' && env_on[1] == '\0') return 1;
    if (env_on != nullptr && env_on[0] == '0' && env_on[1] == '\0') return 0;
    return -1;
}

/// the 4-row kernels for a call of `nt` tokens in the given mode (dot_x4_mode)
inline bool dot_x4_enabled(int mode, int nt) noexcept {
    if (mode >= 0) return mode == 1;
    return nt >= kDotX4MinTokens;
}

inline bool dot_x4_enabled(const char* env_on, const char* env_off, int nt) noexcept {
    return dot_x4_enabled(dot_x4_mode(env_on, env_off), nt);
}

}  // namespace strata::quant
