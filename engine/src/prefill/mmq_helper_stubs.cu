// src/prefill/mmq_helper_stubs.cu - strata-mi50 (PR 3), 2026-11-27.
// strata_mmq_helper (the plugin's MMQ library, cmake/strata_gpu_engine.cmake) compiles moe_mmq.cu and ggml_cuda_host.cu from the tree of the 0.2.15
// engine; moe_mmq.cu names three optional kernels that this library does not carry: the IQ4 int8-tensor-core kernel (sm_80+), and the NVFP4
// W4A4 unit (Blackwell).  gfx906 runs none of them; these definitions say "not covered" so the generic MMQ launch is always used.
#include "strata/prefill/moe_iq4_w4a8.hpp"
#include "common.cuh"
#include "mmq.cuh"

namespace strata::prefill::mmq {
bool run_iq4_w4a8(const Product&, void*) { return false; }
int iq4_w4a8_mode() { return 0; }
bool w4a4_available() { return false; }
void run_nvfp4_w4a8(ggml_backend_cuda_context&, const mmq_args&, cudaStream_t) {}
void run_nvfp4_w4a4(ggml_backend_cuda_context&, const mmq_args&, cudaStream_t) {}
void quantize_nvfp4_w4a4(const float*, const int32_t*, void*, float*, bool, int64_t, int64_t, int64_t, int64_t, cudaStream_t) {}
}  // namespace strata::prefill::mmq
