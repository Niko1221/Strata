// src/quant/x86/sk_codecs_avx2.cpp - sk_codecs.cpp as the AVX2 tier (strata::quant::sk::avx2), compiled with -mavx2 -mfma -mf16c (MSVC /arch:AVX2).
// Called only after strata::quant::caps().avx2 (sk_dispatch.cpp).
#define STRATA_SK_TIER_AVX2 1
#define STRATA_SK_TIER_NS avx2
#include "../sk_codecs.cpp"
