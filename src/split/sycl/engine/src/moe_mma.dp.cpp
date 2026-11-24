// The prompt path's routed experts on the tensor cores (moe_mma.cuh), in an object of their own so a change to them
// recompiles in seconds.
#define DPCT_PROFILING_ENABLED
#include <sycl/sycl.hpp>
#include <dpct/dpct.hpp>
#include "kernels.hpp"
#include "strata/kprof.hpp"

#include <algorithm>
#include <stdexcept>
#include <string>

#include "ggml.h"
#if FNX_HIP
// the tree's hip_compat/intrinsics.hpp names these as macros; ggml's HIP vendor header defines its own: it goes first, then ours return
#undef __vsubss4
#undef __vsub4
#undef __vcmpne4
#undef __shfl_sync
#undef __shfl_up_sync
#undef __shfl_xor_sync
#endif
#include "common.dp.hpp"
#if FNX_HIP
#undef __shfl_sync
#undef __shfl_up_sync
#undef __shfl_xor_sync
#define __shfl_xor_sync(...) (::strata::hip_compat::shfl_xor_sync(__VA_ARGS__))
#define __shfl_up_sync(...) (::strata::hip_compat::shfl_up_sync(__VA_ARGS__))
#define __shfl_sync(...) (::strata::hip_compat::shfl_sync(__VA_ARGS__))
#undef cudaFuncSetAttribute
#endif

namespace fnx {

namespace {

void ck_launch(const char* what) {
    
    const dpct::err0 e = 0;
    
    
    
    if (e != 0) throw std::runtime_error(std::string(what) + ": " +
                                         dpct::get_error_string_dummy(e));
}
 __dpct_inline__ float silu_(float x) {
    return x / (1.0f + sycl::native::exp(-x));
}

#include "moe_w.dp.hpp"
#include "moe_mma.dp.hpp"
#include "moe_mma_iq.dp.hpp"
#include <cmath>

template <int TYPE>
void mma_launch(const MoeW &w, const BigJob *jobs, const int *njobs,
                int njobs_cap, const sycl::half *x, float *out,
                dpct::queue_ptr s) {
    constexpr size_t sm = mm_smem<MM_BM>();
    static_assert(sm <= 48 * 1024, "moe_mma: shared memory past the default limit");
    const MoeTables tables{kvalues_iq4nl.get_ptr(*s), kmask_iq2xs.get_ptr(*s),
                           ksigns_iq2xs.get_ptr(*s), iq3s_grid.get_ptr(*s),
                           iq3xxs_grid.get_ptr(*s), iq2s_grid.get_ptr(*s)};
    {
        dpct::has_capability_or_fail(s->get_device(), {sycl::aspect::fp16});

        strata::kprof::kp(s, __FILE__, __func__, __LINE__)->submit([&](sycl::handler &cgh) {
            sycl::local_accessor<uint8_t, 1> dpct_local_acc_ct1(
                sycl::range<1>(sm), cgh);
            sycl::local_accessor<BigJob, 0> jb_acc_ct1(cgh);

            cgh.parallel_for(
                sycl::nd_range<3>(
                    sycl::range<3>(1, njobs_cap,
                                   (w.nrows + MM_BM - 1) / MM_BM) *
                        sycl::range<3>(1, 1, MM_THREADS),
                    sycl::range<3>(1, 1, MM_THREADS)),
                [=](sycl::nd_item<3> item_ct1)
                    [[sycl::reqd_sub_group_size(32)]] {
                        k_moe_mma<TYPE, MM_BM>(
                            w, jobs, njobs, x, w.ncols, out,
                            dpct_local_acc_ct1
                                .get_multi_ptr<sycl::access::decorated::no>()
                                .get(),
                            jb_acc_ct1, tables);
                    });
        });
    }
}

void moe_mma(const DW &A, size_t sa, const BigJob *jobs, const int *njobs,
             int njobs_cap, const sycl::half *x, float *out, void *st) {
    const MoeW w{(const char*) A.p, sa, A.row_bytes, A.n_out, A.n_in, A.ep};
    if (w.ncols % 32) throw std::runtime_error("moe_mma: columns not in units of 32");
    dpct::queue_ptr s = (dpct::queue_ptr)st;
    switch (A.type) {
        case T_Q4_K: mma_launch<GGML_TYPE_Q4_K>(w, jobs, njobs, njobs_cap, x, out, s); break;
        case T_Q5_K: mma_launch<GGML_TYPE_Q5_K>(w, jobs, njobs, njobs_cap, x, out, s); break;
        case T_Q5_1: mma_launch<GGML_TYPE_Q5_1>(w, jobs, njobs, njobs_cap, x, out, s); break;
        case T_Q8_0: mma_launch<GGML_TYPE_Q8_0>(w, jobs, njobs, njobs_cap, x, out, s); break;
        case T_Q4_0: mma_launch<GGML_TYPE_Q4_0>(w, jobs, njobs, njobs_cap, x, out, s); break;
        case T_IQ4_NL: mma_launch<GGML_TYPE_IQ4_NL>(w, jobs, njobs, njobs_cap, x, out, s); break;
        case T_IQ4_XS: mma_launch<GGML_TYPE_IQ4_XS>(w, jobs, njobs, njobs_cap, x, out, s); break;
        case T_IQ3_S: mma_launch<GGML_TYPE_IQ3_S>(w, jobs, njobs, njobs_cap, x, out, s); break;
        case T_IQ3_XXS: mma_launch<GGML_TYPE_IQ3_XXS>(w, jobs, njobs, njobs_cap, x, out, s); break;
        case T_IQ2_S: mma_launch<GGML_TYPE_IQ2_S>(w, jobs, njobs, njobs_cap, x, out, s); break;
        default: throw std::runtime_error("moe_mma: expert type " + std::to_string(A.type) + " not handled");
    }
    ck_launch("moe_mma");
}
}  // namespace
void moe_gate_up_mma(const DW& A, size_t sa, const BigJob* jobs, const int* njobs, int njobs_cap, const uint16_t* xh, float* gu, void* st) {
    moe_mma(A, sa, jobs, njobs, njobs_cap,
            reinterpret_cast<const sycl::half *>(xh), gu, st);
}
void swiglu_h16(const float* gu, int ffs, int P, uint16_t* h16, void* st) {
    {
        dpct::has_capability_or_fail(
            ((sycl::queue *)((dpct::queue_ptr)st))->get_device(),
            {sycl::aspect::fp16});

        strata::kprof::kp((dpct::queue_ptr)st, __FILE__, __func__, __LINE__)->parallel_for(
                sycl::nd_range<3>(
                    sycl::range<3>(1, 1,
                                   (unsigned)std::min<size_t>(
                                       4096, ((size_t)P * ffs + 255) / 256)) *
                        sycl::range<3>(1, 1, 256),
                    sycl::range<3>(1, 1, 256)),
                [=](sycl::nd_item<3> item_ct1) {
                    k_swiglu_h16(gu, ffs, P,
                                 reinterpret_cast<sycl::half *>(h16));
                });
    }
    ck_launch("swiglu_h16");
}
void moe_down_mma(const DW& A, size_t sa, const BigJob* jobs, const int* njobs, int njobs_cap, const uint16_t* h16, float* part, void* st) {
    moe_mma(A, sa, jobs, njobs, njobs_cap,
            reinterpret_cast<const sycl::half *>(h16), part, st);
}

}  // namespace fnx
