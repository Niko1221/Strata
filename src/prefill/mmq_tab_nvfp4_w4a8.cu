// src/prefill/mmq_tab_nvfp4_w4a8.cu - the pointer-table MMQ (STRATA_MMQ_DIRECT) for NVFP4 with 8-bit activations
// (W4A8): the same split as the gathered product in mmq_nvfp4_w4a8.cu.  mmq.cuh compiles NVFP4 as FP4 x FP4 when it
// sees Blackwell's FP4 MMA; this TU hides it on both sides - the device code (BLACKWELL_MMA_AVAILABLE) and the host's
// tile config (blackwell_mma_available) - and is built with the plain architectures, so every Blackwell card
// (sm_121 too) gets the int8 tensor-core variant here.  Only mul_mat_q_tab_case gets a renamed name: the mirrored
// kernel, launcher and J switch are static, so each TU instantiates its own copies; the explicit instantiation of
// mul_mat_q_tab_case<NVFP4> is the one symbol the two NVFP4 tab TUs would otherwise fold.
#include "common.cuh"

#if !defined(GGML_USE_HIP) && !defined(__HIPCC__)   // no NVFP4 pointer table on HIP (moe_mmq.cu's tab dispatch is CUDA-only)
#undef BLACKWELL_MMA_AVAILABLE
#define blackwell_mma_available(cc) false
#define mul_mat_q_tab_case strata_w4a8_tab_mul_mat_q_tab_case
#include "moe_mmq_tab.cuh"

namespace strata::prefill::mmq {
template bool strata_w4a8_tab_mul_mat_q_tab_case<GGML_TYPE_NVFP4>(ggml_backend_cuda_context&, const mmq_args&,
                                                                  const mmq_x_tab&, cudaStream_t);
bool mul_mat_q_tab_nvfp4_w4a8(ggml_backend_cuda_context& ctx, const mmq_args& a, const mmq_x_tab& tab, cudaStream_t s) {
    return strata_w4a8_tab_mul_mat_q_tab_case<GGML_TYPE_NVFP4>(ctx, a, tab, s);
}
}  // namespace strata::prefill::mmq
#endif
