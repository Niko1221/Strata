// src/prefill/mmq_tab_nvfp4_w4a4.cu - the pointer-table MMQ (STRATA_MMQ_DIRECT) for NVFP4 with FP4 activations
// (W4A4): the tab twin of mmq_nvfp4_w4a4.cu, compiled by CMake in the same one target with 12x -> 12xa.  mmq.cuh's
// NVFP4 tile loader sees BLACKWELL_MMA_AVAILABLE and reads FP4 activations; the per-row scales travel through
// mmq_args::y_scale, which the mirrored scheduler already indexes for NVFP4 (moe_mmq_tab.cuh's y_scale_tile).
// Renamed like the gathered W4A4 instance so the linker cannot fold the two NVFP4 tab copies; run_tab picks this one
// exactly when w4a4_available() said yes (the same probe as fp4_activations), so an sm_121 card or a build without
// the 12xa image keeps the W4A8 pointer table.
#include "common.cuh"

#if !defined(GGML_USE_HIP) && !defined(__HIPCC__)   // HIP builds glob this TU; it stays empty (moe_mmq.cu's tab dispatch is CUDA-only)
#define mul_mat_q_tab_case strata_w4a4_tab_mul_mat_q_tab_case
#include "moe_mmq_tab.cuh"

namespace strata::prefill::mmq {
template bool strata_w4a4_tab_mul_mat_q_tab_case<GGML_TYPE_NVFP4>(ggml_backend_cuda_context&, const mmq_args&,
                                                                  const mmq_x_tab&, cudaStream_t);
bool mul_mat_q_tab_nvfp4_w4a4(ggml_backend_cuda_context& ctx, const mmq_args& a, const mmq_x_tab& tab, cudaStream_t s) {
    return strata_w4a4_tab_mul_mat_q_tab_case<GGML_TYPE_NVFP4>(ctx, a, tab, s);
}
}  // namespace strata::prefill::mmq
#endif
