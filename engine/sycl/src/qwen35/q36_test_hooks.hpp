// Test-only launchers for the real kernels in q36_engine.dp.cpp.
// Define STRATA_Q36_TEST_HOOKS while compiling the engine and include the
// matching q36_test_hooks.inc at its end. These do not form the runtime API.
#pragma once
#include <sycl/sycl.hpp>
#include <cstdint>
namespace strata::qwen35::test {
void conv_sequence(sycl::queue*, float*, const float*, const float*, float*, int, int, float, int, bool);
void gdn_sequence(sycl::queue*, float*, const float*, int, const float*, const float*, const float*, const float*, float, float*, int, int, int, bool);
void gdn_sequence_columns(sycl::queue*, float*, const float*, int, const float*, const float*, const float*, const float*, float, float*, int, int, int);
void attention_prepare(sycl::queue*, const float*, const float*, const float*, const float*, const float*, float*, float*, float*, float*, int, int, int, const int32_t*, int, float, float, const int32_t*, const int32_t*);
// Vh: [HD,P], Sh: [G,nt,ldp], out: [nt,H,HD] beginning at this KV head group.
void attention_values_gemm(sycl::queue*, const sycl::half*, const sycl::half*, float*, int HD, int nt, int P, int ldp, int H, int G);
void attention_split(sycl::queue*, const float*, const float*, const float*, const float*, const int32_t*, int, int, int, int, float*, float*, float*, float*);
}

// Executes Engine::add_rms_post itself, including its real old/new verification
// launches. q is borrowed, in order, and in the current DPCT device context.
// The caller owns device/shared-USM x/y/w/h/xq. reference_x/h are optional host
// destinations for the actual old kernel's results (T*n floats each).
namespace strata::qwen35::test {
struct AddRmsHostResult {
    std::int64_t comparisons=0, mismatches=0;
    bool quantized=false;
};
AddRmsHostResult add_rms_host(sycl::queue* q, float* x, const float* y,
                             const float* w, float* h, void* xq, int n, int T,
                             float eps, float* reference_x, float* reference_h);
}
