// The prompt path's routed experts on the tensor cores (moe_mma.cuh), in an object of their own so a change to them
// recompiles in seconds.
#include "kernels.hpp"

#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <mma.h>

#include <algorithm>
#include <stdexcept>
#include <string>

#include "qdot.cuh"   // block layouts and codebook tables (strata-quant-common.h)
#include "hadamard_meta.hpp"   // HAD_BLOCK
#include "hadamard.cuh"        // had128

namespace fnx {

namespace {

void ck_launch(const char* what) {
    const cudaError_t e = cudaGetLastError();
    if (e != cudaSuccess) throw std::runtime_error(std::string(what) + ": " + cudaGetErrorString(e));
}
__device__ __forceinline__ float silu_(float x) { return x / (1.0f + __expf(-x)); }

#include "moe_w.cuh"
#include "moe_mma.cuh"
#include "moe_mma_iq.cuh"

template <int TYPE>
void mma_launch(const MoeW& w, const BigJob* jobs, const int* njobs, int njobs_cap, const __half* x, float* out, cudaStream_t s) {
    constexpr size_t sm = mm_smem<MM_BM>();
    static_assert(sm <= 48 * 1024, "moe_mma: shared memory past the default limit");
    k_moe_mma<TYPE, MM_BM><<<dim3((w.nrows + MM_BM - 1) / MM_BM, njobs_cap), MM_THREADS, sm, s>>>(w, jobs, njobs, x, w.ncols, out);
}

void moe_mma(const DW& A, size_t sa, const BigJob* jobs, const int* njobs, int njobs_cap, const __half* x, float* out, void* st) {
    const MoeW w{(const char*) A.p, sa, A.row_bytes, A.n_out, A.n_in, A.ep};
    if (w.ncols % 32) throw std::runtime_error("moe_mma: columns not in units of 32");
    if (is_trellis(A.type) && w.ncols % 128) throw std::runtime_error("moe_mma: trellis rows not in whole 128-value blocks");
    cudaStream_t s = (cudaStream_t) st;
    switch (A.type) {
        case T_Q4_K: mma_launch<T_Q4_K>(w, jobs, njobs, njobs_cap, x, out, s); break;
        case T_Q5_K: mma_launch<T_Q5_K>(w, jobs, njobs, njobs_cap, x, out, s); break;
        case T_Q5_1: mma_launch<T_Q5_1>(w, jobs, njobs, njobs_cap, x, out, s); break;
        case T_Q8_0: mma_launch<T_Q8_0>(w, jobs, njobs, njobs_cap, x, out, s); break;
        case T_Q4_0: mma_launch<T_Q4_0>(w, jobs, njobs, njobs_cap, x, out, s); break;
        case T_IQ4_NL: mma_launch<T_IQ4_NL>(w, jobs, njobs, njobs_cap, x, out, s); break;
        case T_IQ4_XS: mma_launch<T_IQ4_XS>(w, jobs, njobs, njobs_cap, x, out, s); break;
        case T_IQ3_S: mma_launch<T_IQ3_S>(w, jobs, njobs, njobs_cap, x, out, s); break;
        case T_IQ3_XXS: mma_launch<T_IQ3_XXS>(w, jobs, njobs, njobs_cap, x, out, s); break;
        case T_IQ2_S: mma_launch<T_IQ2_S>(w, jobs, njobs, njobs_cap, x, out, s); break;
        case T_TQ2_T: mma_launch<T_TQ2_T>(w, jobs, njobs, njobs_cap, x, out, s); break;
        case T_TQK6: mma_launch<T_TQK6>(w, jobs, njobs, njobs_cap, x, out, s); break;
        case T_TQK7: mma_launch<T_TQK7>(w, jobs, njobs, njobs_cap, x, out, s); break;
        default: throw std::runtime_error("moe_mma: expert type " + std::to_string(A.type) + " not handled");
    }
    ck_launch("moe_mma");
}

// swiglu_h16 for a rotated down matrix (swiglu_h16_had): a block of 128 threads per 128 columns of the cover; y strides over the P
// pairs (a prompt chunk of 8192 rows has 81920, more than a grid's y extent of 65535)
__global__ void __launch_bounds__(128) k_swiglu_h16_had(const float* __restrict__ gu, int ffs, int P, int ncols, int off, const float* __restrict__ signs,
                                                       float scale, __half* __restrict__ h) {
    __shared__ float hb[128];
    const int col = blockIdx.x * 128 + threadIdx.x, k = col - off;
    for (int e = blockIdx.y; e < P; e += gridDim.y) {   // the same trip count for the whole block (had128 syncs)
        float v = 0.f;
        if (k >= 0 && k < ffs) {
            const float* g = gu + (size_t) e * 2 * ffs;
            v = silu_(g[k]) * g[ffs + k];
        }
        if (signs) v = had128(v, signs[col], scale, hb);
        h[(size_t) e * ncols + col] = __float2half(v);
    }
}
}  // namespace
void moe_gate_up_mma(const DW& A, size_t sa, const BigJob* jobs, const int* njobs, int njobs_cap, const uint16_t* xh, float* gu, void* st) {
    moe_mma(A, sa, jobs, njobs, njobs_cap, reinterpret_cast<const __half*>(xh), gu, st);
}
void swiglu_h16(const float* gu, int ffs, int P, uint16_t* h16, void* st) {
    k_swiglu_h16<<<(unsigned) std::min<size_t>(4096, ((size_t) P * ffs + 255) / 256), 256, 0, (cudaStream_t) st>>>(gu, ffs, P, reinterpret_cast<__half*>(h16));
    ck_launch("swiglu_h16");
}
void swiglu_h16_had(const float* gu, int ffs, int P, uint16_t* h16, const DW& down, void* st) {
    if (down.n_in % HAD_BLOCK || down.had_off < 0 || down.had_ffs != ffs || down.had_off + ffs > down.n_in)
        throw std::runtime_error("swiglu_h16_had: the card's FF slice does not lie in the down matrix's cover");
    if (P <= 0) return;
    k_swiglu_h16_had<<<dim3(down.n_in / 128, std::min(P, 65535)), 128, 0, (cudaStream_t) st>>>(gu, ffs, P, down.n_in, down.had_off, down.had,
                                                                            1.0f / std::sqrt((float) HAD_BLOCK), reinterpret_cast<__half*>(h16));
    ck_launch("swiglu_h16_had");
}
void moe_down_mma(const DW& A, size_t sa, const BigJob* jobs, const int* njobs, int njobs_cap, const uint16_t* h16, float* part, void* st) {
    moe_mma(A, sa, jobs, njobs, njobs_cap, reinterpret_cast<const __half*>(h16), part, st);
}

}  // namespace fnx
