// src/kernels/cuda/glm_elt.cu - the two elementwise steps a glm5-next FFN has and the existing elementwise
// family does not.
//
// `glm_swiglu` is the reference's SwiGLU with a limit, and the CLAMP IS APPLIED on this model:
//
//     out = min(silu(gate), limit) * clamp(up, -limit, +limit)
//
// **THIS FILE USED TO SAY THE CLAMP WAS NEVER APPLIED, AND THAT CLAIM WAS WRONG.**  It read: "`swiglu_clamp_exp`
// is 10.0 on every layer, and the reference never applies it: the value is read into the hparams and then not
// passed to the graph ... the only arch that consults a limit there is STEP35/BAILINGMOE3."  It is applied:
//
//   * ik_llama.cpp's `swiglu_limit()` (llama-model.h:686) has `LLM_ARCH_GLM5NEXT` in its allow-list and returns
//     `swiglu_limits[il]` / `swiglu_limits_shared[il]`.  It is called at llama-build-context.cpp:1274 for the
//     routed experts and :1335 for the dense/shared path.
//   * Mainline llama.cpp applies it to this arch too (llama-graph.cpp:1840-1848 for the shared expert,
//     :2235-2243 for the experts).
//   * It was added deliberately: commit 42a9a2fd "model: Add GLM-5.3-Flash (glm5next) runtime support (#2376)"
//     is what put GLM5NEXT into that allow-list.
//
// The mistake was easy to make and easy to keep: on a confident prompt the gate rarely reaches 10, so the port
// matched the oracle for 30 of 32 cases WITH THE CLAMP MISSING and nothing pointed at it.
//
// **THE CLAMP GOES ON THE SILU'S OUTPUT, NOT ON THE RAW GATE, AND BOTH ORACLES SAY SO.**  This is the trap in
// this file, because mainline llama.cpp contains BOTH arithmetics and reads the right one off the arch:
//
//   * `ggml_swiglu_clamp` itself (ggml-cpu/ops.cpp:3448-3451) clamps the RAW GATE - `gate = min(gate, limit)`,
//     `up = clamp(up, +-limit)`, `out = silu(gate) * up` - and that op is reached ONLY by `LLM_ARCH_DEEPSEEK4`
//     and DFLASH-with-hc_mult (llama-graph.cpp:2237 and :1842).  Reading the op and not the dispatch around it
//     gives clamp-before-silu, which is what this file's first version of this fix did.
//   * GLM5NEXT is every OTHER arch, so it takes the decomposed branch (llama-graph.cpp:1840-1848 / :2235-2243):
//     `up = clamp(up, -limit, limit)`, `gate_act = silu(gate)`, `gate_act = clamp(gate_act, -INF, limit)`,
//     `out = gate_act * up`.  The gate's clamp is ABOVE ONLY, on the silu's output.
//   * ik_llama.cpp, the oracle our ladder runs against, computes the same thing twice: the CUDA kernel
//     (`fused_mul_silu_f32` with a limit, ggml-cuda/unary.cu:74-83) is `g = x/(1+expf(-x)); g = min(g, limit);
//     dst = g * max(-limit, min(limit, y))`, and its CPU iqk path (iqk/iqk_mul_mat.cpp:156-171) is the same
//     order.
//
// The two readings are the same arithmetic until `silu(gate) > limit`, which at limit 10 means `gate > 10.00045`;
// there this one gives exactly `10.0` and clamp-before-silu gives `silu(10) = 9.9995460`, 4.5e-5 apart.  The
// limit guard is the reference's: mainline's `eps = 1e-6f` (llama-graph.cpp:2233) and ik's `limit > 1e-6f`
// (iqk_mul_mat.cpp:156), so an absent limit and a zero one are both "no clamp".
//
// The rival readings - no clamp at all (what this port used to compute), the gate clamped on BOTH sides, and the
// raw-gate clamp that DEEPSEEK4 uses - are finite, fluent and different, which is what `glm_parity`'s rival cases
// exist to catch.
#include "strata/kernels/glm.hpp"

#include <cuda_runtime.h>

#include <cmath>
#include <cstddef>
#include <cstdint>
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

__global__ void swiglu_kernel(float* __restrict__ gate, const float* __restrict__ up, float limit, int64_t n) {
    const int64_t i = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    // The reference's `ggml_silu` (f32): `x / (1 + exp(-x))`, the same form `glm_kda_conv_silu` uses.
    const float g = gate[i] / (1.0f + expf(-gate[i]));
    // **THE SILU'S OUTPUT IS WHAT GETS CLAMPED, AND ABOVE ONLY** - see the two readings in this file's header.
    // Clamping the raw gate, or clamping the gate on both sides, is the natural misreading and it is not this.
    // The up is clamped on BOTH sides, and that one IS on the raw value.
    if (limit > 1e-6f) {
        gate[i] = fminf(g, limit) * fminf(fmaxf(up[i], -limit), limit);
    } else {
        gate[i] = g * up[i];
    }
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

/// THE ONE NORM IN THIS ARCH WITH A WEIGHT **AND A BIAS**: the DSA indexer's key norm
/// (`indexer.k_norm.weight` / `.bias`, 128 elements each).  Every other norm here is an RMSNorm with a weight
/// only, and reaching for one of those would run and produce a plausible key.
///
/// It is a MEAN-CENTRED norm, so it is not `rms_norm_weighted` with a zero bias bolted on, and the variance is
/// the BIASED one from the second moment:
///
///     mu = mean(x);  inv = rsqrt(mean(x*x) - mu*mu + eps);  y = (x - mu) * inv * w + b
///
/// which is `layer_norm_kernel` in Project Maya's `glm_model.cu:112-140` — the only transcription of this step
/// outside the two llama.cpp trees, and the one whose arithmetic the ladder oracle's CUDA path also uses.
/// `eps` is `attention.layer_norm_rms_epsilon` (1e-5 on this model), the same number the RMSNorms take; it is
/// added INSIDE the root, so it bounds the divide and not the variance.
///
/// ONE BLOCK PER ROW, with the row strided across the block: `dim` is 128, so most of a 256-thread block is idle
/// in the second pass and the reduction is 8 steps deep.  That is deliberate and copied: the alternative (a
/// warp-per-row shuffle form) changes the summation ORDER, and this norm feeds a selection whose ties are
/// resolved by the value of a dot product.
constexpr int LN_THREADS = 256;

__global__ void layer_norm_kernel(const float* __restrict__ x, const float* __restrict__ w,
                                  const float* __restrict__ b, float* __restrict__ y, int dim, float eps) {
    __shared__ float s_sum[LN_THREADS];
    __shared__ float s_sq[LN_THREADS];
    const int tid = (int) threadIdx.x;
    const float* xr = x + (size_t) dim * blockIdx.x;
    float* yr = y + (size_t) dim * blockIdx.x;

    float sum = 0.0f, sq = 0.0f;
    for (int e = tid; e < dim; e += LN_THREADS) {
        const float v = xr[e];
        sum += v;
        sq += v * v;
    }
    s_sum[tid] = sum;
    s_sq[tid] = sq;
    __syncthreads();
    for (int span = LN_THREADS / 2; span > 0; span >>= 1) {
        if (tid < span) {
            s_sum[tid] += s_sum[tid + span];
            s_sq[tid] += s_sq[tid + span];
        }
        __syncthreads();
    }
    const float mu = s_sum[0] / (float) dim;
    const float inv = rsqrtf(s_sq[0] / (float) dim - mu * mu + eps);
    for (int e = tid; e < dim; e += LN_THREADS) yr[e] = (xr[e] - mu) * inv * w[e] + b[e];
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

void glm_swiglu(float* gate, const float* up, float limit, int64_t n, void* stream) {
    if (n <= 0) return;
    const int blocks = (int) ((n + 255) / 256);
    swiglu_kernel<<<blocks, 256, 0, (cudaStream_t) stream>>>(gate, up, limit, n);
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

void glm_layer_norm(const float* x, const float* w, const float* b, float* y, int64_t rows, int64_t cols,
                    float eps, void* stream) {
    if (rows <= 0 || cols <= 0) return;
    if (cols > INT32_MAX || rows > 65535) {
        std::fprintf(stderr, "glm_layer_norm: %lld rows of %lld exceed the launch's int/grid\n", (long long) rows,
                     (long long) cols);
        return;
    }
    // IN PLACE IS ALLOWED and is how the indexer calls it: a row is read into registers before anything is
    // written (the two passes are separated by `__syncthreads`), so `x == y` needs no extra buffer.
    layer_norm_kernel<<<(unsigned) rows, LN_THREADS, 0, (cudaStream_t) stream>>>(x, w, b, y, (int) cols, eps);
    check_launch("glm_layer_norm");
    sync_if_needed(stream, "glm_layer_norm");
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
