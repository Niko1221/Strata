// src/kernels/cuda/glm_elt.cu - the two elementwise steps a glm5-next FFN has and the existing elementwise
// family does not.
//
// `glm_swiglu` is `silu(gate) * up` with the clamp the reference DOES NOT APPLY.  The GGUF carries
// `swiglu_clamp_exp`/`swiglu_clamp_shexp` and ik_llama.cpp reads them into its key table and then never uses
// them: `llm_build_ffn`'s `LLM_FFN_SILU` arm is `ggml_silu(gate)` then `mul(gate, up)`, and the only arch that
// consults a limit there is STEP35/BAILINGMOE3.  So the clamp is inert on this model and applying it would be
// the port's own invention - a difference that shows up only on the tokens where a gate exceeds 10.
#include "strata/kernels/glm.hpp"

#include <cuda_runtime.h>

#include <cmath>
#include <cstddef>
#include <cstdio>
#include <cstdlib>

namespace strata::kernels {
namespace {

void sync_if_needed(void* stream, const char* what) {
    if (stream != nullptr) return;
    const cudaError_t e = cudaDeviceSynchronize();
    if (e != cudaSuccess) {
        std::fprintf(stderr, "%s: %s\n", what, cudaGetErrorString(e));
        std::exit(1);
    }
}

void check_launch(const char* what) {
    const cudaError_t e = cudaGetLastError();
    if (e != cudaSuccess) std::fprintf(stderr, "%s launch: %s\n", what, cudaGetErrorString(e));
}

__global__ void swiglu_kernel(float* __restrict__ gate, const float* __restrict__ up, int64_t n) {
    const int64_t i = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const float g = gate[i];
    // The reference's `ggml_silu` (f32): `x / (1 + exp(-x))`, the same form `glm_kda_conv_silu` uses.
    gate[i] = (g / (1.0f + expf(-g))) * up[i];
}

__global__ void add_inplace_kernel(float* __restrict__ dst, const float* __restrict__ src, int64_t n) {
    const int64_t i = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) dst[i] += src[i];
}

/// `dst[i] *= sigmoid(z[i])` - KDA's output gate.  SIGMOID, not SiLU: the reference is `ggml_sigmoid`.
__global__ void sigmoid_mul_kernel(float* __restrict__ dst, const float* __restrict__ z, int64_t n) {
    const int64_t i = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) dst[i] *= 1.0f / (1.0f + expf(-z[i]));
}

/// `y[o] = sum_i W[o][i] * x[i]`, W row-major `[n_out, n_in]`, F32.
///
/// ONE BLOCK PER OUTPUT ROW with a shared tree reduction.  This is the SLOW shape on purpose: it exists for the
/// handful of weights the file stores as plain f32 and which are therefore neither quantized nor served natively
/// - glm5-next's router (`ffn_gate_inp.weight` is F32, not the BF16 qwen4exp uses) is the one that matters, and
/// it is 288 rows.  A row-split kernel would be faster and is not needed: the router is 1.2 M MACs against the
/// ~4.3 GB of expert bytes a token reads.
__global__ void f32_gemv_kernel(const float* __restrict__ x, const float* __restrict__ w, float* __restrict__ y,
                                int64_t n_in, int64_t n_out) {
    const int64_t o = blockIdx.x;
    if (o >= n_out) return;
    const float* row = w + o * n_in;
    extern __shared__ float red[];
    float part = 0.0f;
    for (int64_t i = threadIdx.x; i < n_in; i += blockDim.x) part += row[i] * x[i];
    red[threadIdx.x] = part;
    __syncthreads();
    for (int step = blockDim.x / 2; step > 0; step >>= 1) {
        if ((int) threadIdx.x < step) red[threadIdx.x] += red[threadIdx.x + step];
        __syncthreads();
    }
    if (threadIdx.x == 0) y[o] = red[0];
}

/// The sigmoid router: `probs = sigmoid(logits)`, the SELECTION is `probs + bias`, the WEIGHTS are the
/// UN-BIASED `probs` of the chosen experts, sum-normalised, then scaled.
///
/// **ONE BLOCK, NOT ONE THREAD, AND THE ONE THREAD COST 12% OF A TOKEN.**  This was `<<<1, 1>>>`, and the
/// reasoning was that 288 candidates and 8 slots is 2.3 K operations.  It is not: the slot loop re-scanned the
/// `j` slots already filled for EVERY candidate, so the real count is `8 * 288 * 8/2` ~ 9.2 K iterations - each
/// with an `expf` - and `expf` was also evaluated a second time for every chosen expert, 2 * 8 * 288 of them.
/// nsys measured 650 us a call, 42 calls a token, **27 ms of a 219 ms token**, on one SM while 27 others idled.
///
/// The parallel form computes the same k answers in a DIFFERENT ORDER, so the tie rule no longer falls out of
/// the loop and has to be carried explicitly.  The serial scan took the LOWEST index among equal selections
/// because it walked `e` ascending and compared with `>`; the reduction below compares the same pairs with the
/// same `>`, and the extra `v != -INFINITY` guard reproduces the one case where the two would differ - the
/// serial scan's `best_sel` starts at `-INFINITY` and `-INFINITY > -INFINITY` is false, so a candidate whose
/// selection IS `-INFINITY` can never be chosen, not even the first one.  With that guard the comparison is a
/// total order and the tree below is order-independent.
///
/// Ties break by ASCENDING INDEX, matching `router_top10`; the combine is a sum over slots so the ORDER is not
/// otherwise observable, and a stable rule is what keeps two runs comparable.
constexpr int ROUTER_THREADS = 256;

__global__ void router_sigmoid_topk_kernel(const float* __restrict__ logits, const float* __restrict__ bias,
                                           int32_t* __restrict__ ids, float* __restrict__ weights,
                                           int n_expert, int k, float scale) {
    // `sel` is NOT stored: `p` is, and the selection value is `p + bias` recomputed on read.  Both forms are
    // the same expression on the same bits, and the second array would have been another 4 bytes a candidate of
    // a shared budget that has to hold whatever `n_expert` a future pack carries.
    extern __shared__ unsigned char smem[];
    float* p = reinterpret_cast<float*>(smem);
    unsigned char* taken = smem + (std::size_t) n_expert * sizeof(float);
    __shared__ float rv[ROUTER_THREADS];
    __shared__ int re[ROUTER_THREADS];
    const int tid = (int) threadIdx.x;

    for (int e = tid; e < n_expert; e += ROUTER_THREADS) {
        p[e] = 1.0f / (1.0f + expf(-logits[e]));
        taken[e] = 0;
    }
    __syncthreads();

    for (int j = 0; j < k; ++j) {
        float bv = -INFINITY;
        int be = -1;
        for (int e = tid; e < n_expert; e += ROUTER_THREADS) {
            if (taken[e] != 0) continue;
            const float v = p[e] + (bias != nullptr ? bias[e] : 0.0f);
            if (v > bv || (v == bv && v != -INFINITY && e < be)) { bv = v; be = e; }
        }
        rv[tid] = bv;
        re[tid] = be;
        __syncthreads();
        for (int step = ROUTER_THREADS / 2; step > 0; step >>= 1) {
            if (tid < step) {
                const float cv = rv[tid + step];
                const int ce = re[tid + step];
                if (cv > rv[tid] || (cv == rv[tid] && cv != -INFINITY && ce < re[tid])) {
                    rv[tid] = cv;
                    re[tid] = ce;
                }
            }
            __syncthreads();
        }
        if (tid == 0) {
            const int best = re[0];
            ids[j] = best;
            // The chosen expert's weight is the `p` ALREADY COMPUTED for it, which is bit-for-bit what the
            // serial version's second `1/(1+expf(-logits[best]))` produced - same expression, same input.
            weights[j] = best < 0 ? 0.0f : p[best];
            if (best >= 0) taken[best] = 1;
        }
        __syncthreads();
    }

    if (tid == 0) {
        float sum = 0.0f;
        for (int j = 0; j < k; ++j) sum += weights[j];
        const float inv = sum > 0.0f ? scale / sum : 0.0f;
        for (int j = 0; j < k; ++j) weights[j] *= inv;
    }
}

}  // namespace

void glm_swiglu(float* gate, const float* up, int64_t n, void* stream) {
    if (n <= 0) return;
    const int blocks = (int) ((n + 255) / 256);
    swiglu_kernel<<<blocks, 256, 0, (cudaStream_t) stream>>>(gate, up, n);
    check_launch("glm_swiglu");
    sync_if_needed(stream, "glm_swiglu");
}

void glm_add_inplace(float* dst, const float* src, int64_t n, void* stream) {
    if (n <= 0) return;
    const int blocks = (int) ((n + 255) / 256);
    add_inplace_kernel<<<blocks, 256, 0, (cudaStream_t) stream>>>(dst, src, n);
    check_launch("glm_add_inplace");
    sync_if_needed(stream, "glm_add_inplace");
}

void glm_sigmoid_mul(float* dst, const float* z, int64_t n, void* stream) {
    if (n <= 0) return;
    const int blocks = (int) ((n + 255) / 256);
    sigmoid_mul_kernel<<<blocks, 256, 0, (cudaStream_t) stream>>>(dst, z, n);
    check_launch("glm_sigmoid_mul");
    sync_if_needed(stream, "glm_sigmoid_mul");
}

void glm_f32_gemv(const float* x, const float* w, float* y, int64_t n_in, int64_t n_out, void* stream) {
    if (n_in <= 0 || n_out <= 0) return;
    if (n_out > 65535) {
        std::fprintf(stderr, "glm_f32_gemv: %lld output rows exceed the grid\n", (long long) n_out);
        return;
    }
    f32_gemv_kernel<<<(unsigned) n_out, 256, 256 * sizeof(float), (cudaStream_t) stream>>>(x, w, y, n_in, n_out);
    check_launch("glm_f32_gemv");
    sync_if_needed(stream, "glm_f32_gemv");
}

void glm_router_sigmoid_topk(const float* logits, const float* bias, int32_t* ids, float* weights, int64_t n_expert,
                             int64_t k, float scale, void* stream) {
    if (n_expert <= 0 || k <= 0 || k > n_expert) {
        std::fprintf(stderr, "glm_router_sigmoid_topk: k %lld of %lld experts\n", (long long) k,
                     (long long) n_expert);
        return;
    }
    // The carve is `p` (4 bytes a candidate) plus `taken` (one), so 288 experts is 1440 B and even a 4096-expert
    // pack stays under the 48 KiB a launch may ask for without `cudaFuncSetAttribute`.
    const std::size_t smem = (std::size_t) n_expert * (sizeof(float) + 1);
    if (smem > 48u * 1024u) {
        std::fprintf(stderr, "glm_router_sigmoid_topk: %lld experts need %zu B of the router kernel's shared "
                             "budget, over the 48 KiB a launch can ask for\n",
                     (long long) n_expert, smem);
        return;
    }
    router_sigmoid_topk_kernel<<<1, ROUTER_THREADS, smem, (cudaStream_t) stream>>>(logits, bias, ids, weights,
                                                                                 (int) n_expert, (int) k, scale);
    check_launch("glm_router_sigmoid_topk");
    sync_if_needed(stream, "glm_router_sigmoid_topk");
}

}  // namespace strata::kernels
