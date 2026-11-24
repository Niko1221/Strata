// src/quant/x86/sk_codecs_avx512.cpp - sk_codecs.cpp as the AVX-512 tier (strata::quant::sk::avx512), compiled with -mavx512f -mavx512bw -mavx512vl
// -mavx512dq -mavx512vnni (+ avx2, fma, f16c; MSVC /arch:AVX512, clang-cl also /clang:-mavx512vnni).  Called only after caps().avx512 && caps().avx512_vnni.
#define STRATA_SK_TIER_AVX512 1
#define STRATA_SK_TIER_NS avx512
#include "../sk_codecs.cpp"
