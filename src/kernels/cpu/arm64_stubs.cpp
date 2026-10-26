// src/kernels/cpu/arm64_stubs.cpp - Linux arm64 (NVIDIA GB10 / DGX Spark, unofficial; CMakeLists.txt STRATA_ARM64): the
// x86 CPU kernels' entry points, which that build leaves out (expert.cpp, q2_avx2.cpp, iq_avx*.cpp, kq_avx*.cpp,
// nvfp4_avx*.cpp, q2_avx1.cpp).  The same split as the Mac's metal/src/engine/stubs*.cpp: a question ("does this kernel
// take the type?") answers false, so native_expert.cpp takes ggml-cpu's rows; every kernel stops the engine with its
// name, because its callers reach it only behind cpu_avx2_ok() / cpu_avx1_ok() / cpu_avx512_ok(), which read false here
// (include/arm64_compat/cpuid.h).  A stop names what is missing instead of computing wrong numbers.  The Q2_0 packs
// (act_quant_q8_1, q2_0_gguf_rows_multi, s2_expert_*) have no ggml-cpu path: their CPU experts are among the stops.
#include "strata/kernels/cpu/expert.hpp"
#include "strata/kernels/cpu/iq_avx1.hpp"
#include "strata/kernels/cpu/iq_avx2.hpp"
#include "strata/kernels/cpu/iq_avx512.hpp"
#include "strata/kernels/cpu/kq_avx1.hpp"
#include "strata/kernels/cpu/kq_avx2.hpp"
#include "strata/kernels/cpu/nvfp4_avx2.hpp"
#include "strata/kernels/cpu/nvfp4_avx512.hpp"
#include "strata/kernels/cpu/tq_cpu.hpp"

#include <cstdio>
#include <cstdlib>

namespace strata::kernels::cpu {
namespace {
[[noreturn]] void missing(const char* name) {
    std::fprintf(stderr, "strata: %s has no arm64 kernel (an x86 CPU kernel, left out of the Linux arm64 build). "
                         "Use an IQ / Unsloth model, whose CPU experts run on ggml-cpu\n", name);
    std::fflush(stderr);
    std::abort();
}
}  // namespace

// the questions
bool iq128_supported(int) noexcept { return false; }
bool iq128_down_supported(int) noexcept { return false; }
bool iq512_gather_on() noexcept { return false; }
bool iq512x_supported(int) noexcept { return false; }
bool iq256x_supported(int) noexcept { return false; }
bool nvfp4_256_fits(int) { return false; }
bool nvfp4_512_fits(int) { return false; }

// Q2_0 (ISTA's own format: no ggml-cpu path)
void act_quant_q8_1(const float*, int, ActQ&) { missing("act_quant_q8_1"); }
void act_quant_q8_1_avx2(const float*, int, ActQ&) { missing("act_quant_q8_1_avx2"); }
void act_quant_q8_1_avx1(const float*, int, ActQ&) { missing("act_quant_q8_1_avx1"); }
void s2_expert_vnni_q(const uint8_t*, const ActQ&, float*, ExpertScratch&) { missing("s2_expert_vnni_q"); }
void s2_expert_gu_rows(const uint8_t*, const ActQ&, float*, int, int) { missing("s2_expert_gu_rows"); }
void s2_expert_down_rows(const uint8_t*, const ActQ&, float*, int, int) { missing("s2_expert_down_rows"); }
void s2_expert_gu_rows_multi(const uint8_t*, const ActQ* const*, int, float* const*, int, int) {
    missing("s2_expert_gu_rows_multi");
}
void s2_expert_down_rows_multi(const uint8_t*, const ActQ* const*, int, float* const*, int, int) {
    missing("s2_expert_down_rows_multi");
}
void q2_0_gguf_rows_multi(const uint8_t*, size_t, int, const ActQ* const*, int, float* const*, int, int) {
    missing("q2_0_gguf_rows_multi");
}
void q2_0_gguf_rows_multi_avx2(const uint8_t*, size_t, int, const ActQ* const*, int, float* const*, int, int) {
    missing("q2_0_gguf_rows_multi_avx2");
}
void q2_0_gguf_rows_multi_avx2_legacy(const uint8_t*, size_t, int, const ActQ* const*, int, float* const*, int, int) {
    missing("q2_0_gguf_rows_multi_avx2_legacy");
}
void q2_0_gguf_rows_multi_avx1(const uint8_t*, size_t, int, const ActQ* const*, int, float* const*, int, int) {
    missing("q2_0_gguf_rows_multi_avx1");
}
void q2_0_gguf_rows_multi_avx1_legacy(const uint8_t*, size_t, int, const ActQ* const*, int, float* const*, int, int) {
    missing("q2_0_gguf_rows_multi_avx1_legacy");
}

// the i-quant, K-quant, BF16 and NVFP4 rows (ggml-cpu's rows, or plain C++, run instead)
void iq128_gu_rows(int, const uint8_t*, size_t, size_t, int, const void* const*, int, float* const*, int, int) {
    missing("iq128_gu_rows");
}
void iq128_down_rows(int, const uint8_t*, size_t, int, const void* const*, int, float* const*, int, int) {
    missing("iq128_down_rows");
}
void iq256_gu_rows_exact_one(int, const uint8_t*, size_t, size_t, int, const void*, float*, int, int) {
    missing("iq256_gu_rows_exact_one");
}
void iq256_gu_rows(int, const uint8_t*, size_t, size_t, int, const void* const*, int, float* const*, int, int) {
    missing("iq256_gu_rows");
}
void iq4nl256_down_rows(const uint8_t*, size_t, int, const void* const*, int, float* const*, int, int) {
    missing("iq4nl256_down_rows");
}
void iq512_gu_rows(int, const uint8_t*, size_t, size_t, int, const void* const*, int, float* const*, int, int) {
    missing("iq512_gu_rows");
}
void iq512x_rows(int, const uint8_t*, size_t, int, const void* const*, int, float* const*, int, int) {
    missing("iq512x_rows");
}
void iq256x_rows(int, const uint8_t*, size_t, int, const void* const*, int, float* const*, int, int) {
    missing("iq256x_rows");
}
void q8k_quant_avx1(const float*, void*, int64_t) { missing("q8k_quant_avx1"); }
void q8k_quant_avx2(const float*, void*, int64_t) { missing("q8k_quant_avx2"); }
void kq256_gu_rows(int, const uint8_t*, size_t, size_t, int, const void* const*, int, float* const*, int, int) {
    missing("kq256_gu_rows");
}
void kq256_rows(int, const uint8_t*, size_t, int, const void* const*, int, float* const*, int, int) {
    missing("kq256_rows");
}
void bf16_rows_dot_multi(const uint16_t*, int, int, const float*, int, float*) { missing("bf16_rows_dot_multi"); }
void bf16_rows_dot_multi_avx1(const uint16_t*, int, int, const float*, int, float*) {
    missing("bf16_rows_dot_multi_avx1");
}
void nvfp4_256_gu_rows(const uint8_t*, size_t, size_t, int, const void* const*, int, float* const*, int, int, float,
                       float) {
    missing("nvfp4_256_gu_rows");
}
void nvfp4_256_rows(const uint8_t*, size_t, int, const void* const*, int, float* const*, int, int, float) {
    missing("nvfp4_256_rows");
}
void nvfp4_512_gu_rows(const uint8_t*, size_t, size_t, int, const void* const*, int, float* const*, int, int, float,
                       float) {
    missing("nvfp4_512_gu_rows");
}
void nvfp4_512_rows(const uint8_t*, size_t, int, const void* const*, int, float* const*, int, int, float) {
    missing("nvfp4_512_rows");
}

// the trellis rows (the DeepSeek engine's CPU lane, src/dsv41/ds4_tq_cpu.cpp; tq_isa() answers scalar here)
namespace tq_detail {
void tq_rows_avx2(int, const int16_t*, const uint8_t*, size_t, int, const TqAct* const*, int, float* const*, int, int) {
    missing("tq_rows_avx2");
}
void tq_rows_avx512(int, const int16_t*, const uint8_t*, size_t, int, const TqAct* const*, int, float* const*, int, int) {
    missing("tq_rows_avx512");
}
}  // namespace tq_detail

}  // namespace strata::kernels::cpu
