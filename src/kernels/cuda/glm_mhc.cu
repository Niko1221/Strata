// src/kernels/cuda/glm_mhc.cu - GLM-5.3-Flash's hyper-connections (mHC).
//
// **mHC OWNS EVERY RESIDUAL IN THE MODEL.**  There is no `x = x + sublayer(x)` anywhere in a glm5-next block: the
// four residual streams are read through `pre`, written back through `post` and `comb`, and collapsed by a MEAN
// before the head.  A port that keeps a plain residual as well still runs and produces fluent garbage, which is
// why every line here follows ggml's `ggml_compute_forward_hc_pre_f32` /
// `ggml_compute_forward_hc_post_f32` (ik_llama.cpp ggml/src/ggml.c) rather than a reading of the paper.
//
// Three details that are easy to get wrong and impossible to see in the output:
//
//   * `pre` gets `+ eps` and `post` does NOT; `post` gets the factor `2`, and `pre` does not.
//   * The Sinkhorn loop is `softmax rows` then a COLUMN pass, then `iters - 1` rounds of (row, column) - so with
//     `iters = 20` the last normalisation is a column one, 39 passes in all, and the initial softmax is not one
//     of them.
//   * `comb[j*S + i]` is source j, destination i.  The transpose produces numbers of the same magnitude and a
//     model that still talks.
#include "strata/kernels/glm.hpp"

#include "strata/kernels/bf16_bits.hpp"

#include <cuda_runtime.h>

#include <cmath>
#include <cstdio>
#include <cstdlib>

namespace strata::kernels {
namespace {

constexpr int THREADS = 256;

/// Host-side and synchronous when the caller passed no stream, matching the rest of the kernel layer: a null
/// stream is how the tests call in, and they need the error where it happened.
void sync_if_needed(void* stream, const char* what) {
    if (stream != nullptr) return;
    const cudaError_t e = cudaDeviceSynchronize();
    if (e != cudaSuccess) {
        std::fprintf(stderr, "%s: %s\n", what, cudaGetErrorString(e));
        std::exit(1);
    }
}

bool check_launch(const char* what) {
    const cudaError_t e = cudaGetLastError();
    if (e != cudaSuccess) {
        std::fprintf(stderr, "%s launch: %s\n", what, cudaGetErrorString(e));
        return false;
    }
    return true;
}

inline unsigned grid_for(int64_t n) { return (unsigned) ((n + THREADS - 1) / THREADS); }

inline bool shapes_ok(const GlmHcShapes& s, const char* what) {
    if (s.hc < 1 || s.hc > 8 || s.mix != s.hc * (2 + s.hc) || s.sinkhorn_iters < 2 || s.n_embd < 1) {
        std::fprintf(stderr, "%s: bad mHC geometry (hc %lld, mix %lld, iters %lld, n_embd %lld)\n", what,
                     (long long) s.hc, (long long) s.mix, (long long) s.sinkhorn_iters, (long long) s.n_embd);
        return false;
    }
    return true;
}

/// One thread per token: the whole per-token working set is `hc*hc` floats, and the two normalisations inside are
/// over 4 and 16 elements.  Spreading a token over a warp would trade 4-way serial work for a shuffle tree.
__global__ void hc_pre_kernel(const float* __restrict__ mixes, const float* __restrict__ scale,
                              const float* __restrict__ base, float* __restrict__ pre, float* __restrict__ post,
                              float* __restrict__ comb, int S, int iters, float eps, int64_t T) {
    const int64_t t = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= T) return;

    const float* x = mixes + t * (S * S + 2 * S);
    const float* x_pre = x;
    const float* x_post = x + S;
    const float* x_comb = x + 2 * S;
    float* y_pre = pre + t * S;
    float* y_post = post + t * S;
    float* y_comb = comb + t * S * S;

    for (int i = 0; i < S; ++i) {
        // `pre` is sigmoid + eps; `post` is 2 * sigmoid with no eps.  Both halves of that sentence are load-bearing.
        float v = x_pre[i] * scale[0] + base[i];
        y_pre[i] = 1.0f / (1.0f + expf(-v)) + eps;
        v = x_post[i] * scale[1] + base[S + i];
        y_post[i] = 2.0f * (1.0f / (1.0f + expf(-v)));
    }

    float m[64];
    for (int i = 0; i < S * S; ++i) m[i] = x_comb[i] * scale[2] + base[2 * S + i];

    // Softmax, rows.  NOTE: the `+ eps` rides on the RESULT of the divide, so a row never sums to exactly 1.
    for (int r = 0; r < S; ++r) {
        float mx = m[r * S];
        for (int c = 1; c < S; ++c) mx = fmaxf(mx, m[r * S + c]);
        float sum = 0.0f;
        for (int c = 0; c < S; ++c) {
            m[r * S + c] = expf(m[r * S + c] - mx);
            sum += m[r * S + c];
        }
        for (int c = 0; c < S; ++c) m[r * S + c] = m[r * S + c] / sum + eps;
    }

    // The column pass, whose `sum` starts AT eps rather than at 0 - the asymmetry with the row pass above is the
    // reference's, and the pair is what the `iters - 1` loop then repeats.
    for (int c = 0; c < S; ++c) {
        float sum = eps;
        for (int r = 0; r < S; ++r) sum += m[r * S + c];
        for (int r = 0; r < S; ++r) m[r * S + c] /= sum;
    }

    for (int it = 0; it < iters - 1; ++it) {
        for (int r = 0; r < S; ++r) {
            float sum = eps;
            for (int c = 0; c < S; ++c) sum += m[r * S + c];
            for (int c = 0; c < S; ++c) m[r * S + c] /= sum;
        }
        for (int c = 0; c < S; ++c) {
            float sum = eps;
            for (int r = 0; r < S; ++r) sum += m[r * S + c];
            for (int r = 0; r < S; ++r) m[r * S + c] /= sum;
        }
    }

    for (int i = 0; i < S * S; ++i) y_comb[i] = m[i];
}

/// `out[i0 + i*n] = x[i0] * post[i] + sum_j comb[j*S + i] * res[i0 + j*n]`, one thread per (token, i0).
///
/// The residual is read S times per output element, which at S=4 is 16 loads and 16 FMAs for 4 stores - the
/// arithmetic is trivial and the kernel is here for the memory pattern: `res[j*n + i0]` is contiguous in `i0`, so
/// a warp's S reads are S coalesced lines and not a strided gather.
__global__ void hc_post_kernel(const float* __restrict__ x, const float* __restrict__ post,
                               const float* __restrict__ res, const float* __restrict__ comb,
                               float* __restrict__ out, int64_t n, int S, int64_t total) {
    const int64_t i = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= total) return;
    const int64_t t = i / n;
    const int64_t i0 = i - t * n;

    const float* x_r = x + t * n;
    const float* post_r = post + t * S;
    const float* comb_r = comb + t * S * S;
    const float* res_r = res + t * n * S;
    float* out_r = out + t * n * S;

    const float xv = x_r[i0];
    for (int i2 = 0; i2 < S; ++i2) {
        float sum = xv * post_r[i2];
        for (int j = 0; j < S; ++j) sum += comb_r[j * S + i2] * res_r[j * n + i0];
        out_r[i2 * n + i0] = sum;
    }
}

/// The stream a sublayer is fed: `out[i0] = sum_j pre[j] * x[i0 + j*n]`.  The transpose of `hc_post` without the
/// S writes, and the one place mHC is cheap enough that a fused kernel would be the same code.
__global__ void hc_mix_kernel(const float* __restrict__ x, const float* __restrict__ pre, float* __restrict__ out,
                              int64_t n, int S, int64_t total) {
    const int64_t i = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= total) return;
    const int64_t t = i / n;
    const int64_t i0 = i - t * n;

    const float* pre_r = pre + t * S;
    const float* x_r = x + t * n * S;
    float sum = 0.0f;
    for (int j = 0; j < S; ++j) sum += pre_r[j] * x_r[j * n + i0];
    out[t * n + i0] = sum;
}

/// The collapse before the head.  **THE MEAN, NOT THE SUM** - a sum is off by exactly `S`, which after
/// `output_norm` is a constant factor on every logit and survives greedy decoding on a confident prompt.
__global__ void hc_sum_kernel(const float* __restrict__ x, float* __restrict__ out, int64_t n, int S,
                              int64_t total) {
    const int64_t i = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= total) return;
    const int64_t t = i / n;
    const int64_t i0 = i - t * n;
    const float* x_r = x + t * n * S;
    float sum = 0.0f;
    for (int j = 0; j < S; ++j) sum += x_r[j * n + i0];
    out[t * n + i0] = sum / (float) S;
}

/// The weightless RMSNorm over the flattened streams, written straight out as bf16 because the only consumer is
/// the `hc_*_fn` GEMV and that weight is bf16 in every pack (the packer leaves it native).
///
/// One BLOCK per token, not one warp: the row is `n_embd * hc` = 16384 floats, and a warp would walk it in 512
/// dependent steps where 256 threads walk it in 64.
__global__ void hc_norm_bf16_kernel(const float* __restrict__ x, uint16_t* __restrict__ out, float* __restrict__ out_f32,
                                    int64_t cols, float eps, int64_t T) {
    __shared__ float part[THREADS / 32];
    const int64_t t = blockIdx.x;
    if (t >= T) return;
    const float* r = x + t * cols;

    float acc = 0.0f;
    for (int64_t c = threadIdx.x; c < cols; c += THREADS) acc += r[c] * r[c];

    // A warp reduce then one shared slot per warp: 8 slots, one barrier.  `__shfl_down_sync` on a full mask is
    // safe here because every thread of the block reaches it - the loop above is a grid-stride with no early exit.
    for (int off = 16; off > 0; off >>= 1) acc += __shfl_down_sync(0xFFFFFFFFu, acc, off);
    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;
    if (lane == 0) part[warp] = acc;
    __syncthreads();

    __shared__ float inv_sh;
    if (threadIdx.x == 0) {
        float total = 0.0f;
        for (int w = 0; w < THREADS / 32; ++w) total += part[w];
        inv_sh = rsqrtf(total / (float) cols + eps);
    }
    __syncthreads();
    const float inv = inv_sh;

    uint16_t* o = out + t * cols;
    // The f32 form is written from the SAME product, not re-derived from the bf16: the quantized activation
    // images (`hc_*_fn` is Q8_0 in the file) are built from it, and a second rounding between the two would be
    // a systematic offset that shows up as a slightly wrong routing decision rather than as an error.
    float* of = out_f32 ? out_f32 + t * cols : nullptr;
    for (int64_t c = threadIdx.x; c < cols; c += THREADS) {
        const float v = r[c] * inv;
        o[c] = bf16_from_f32(v);
        if (of) of[c] = v;
    }
}

}  // namespace

void glm_hc_pre(const float* mixes, const float* scale, const float* base, float* pre, float* post, float* comb,
                const GlmHcShapes& s, int64_t T, void* stream) {
    if (T <= 0) return;
    if (!shapes_ok(s, "glm_hc_pre")) return;
    hc_pre_kernel<<<grid_for(T), THREADS, 0, (cudaStream_t) stream>>>(mixes, scale, base, pre, post, comb,
                                                                     (int) s.hc, (int) s.sinkhorn_iters, s.eps, T);
    check_launch("glm_hc_pre");
    sync_if_needed(stream, "glm_hc_pre");
}

void glm_hc_post(const float* x, const float* post, const float* residual, const float* comb, float* out,
                 const GlmHcShapes& s, int64_t T, void* stream) {
    if (T <= 0) return;
    if (!shapes_ok(s, "glm_hc_post")) return;
    hc_post_kernel<<<grid_for(T * s.n_embd), THREADS, 0, (cudaStream_t) stream>>>(
        x, post, residual, comb, out, s.n_embd, (int) s.hc, T * s.n_embd);
    check_launch("glm_hc_post");
    sync_if_needed(stream, "glm_hc_post");
}

void glm_hc_mix(const float* x, const float* pre, float* out, const GlmHcShapes& s, int64_t T, void* stream) {
    if (T <= 0) return;
    if (!shapes_ok(s, "glm_hc_mix")) return;
    hc_mix_kernel<<<grid_for(T * s.n_embd), THREADS, 0, (cudaStream_t) stream>>>(x, pre, out, s.n_embd, (int) s.hc,
                                                                               T * s.n_embd);
    check_launch("glm_hc_mix");
    sync_if_needed(stream, "glm_hc_mix");
}

void glm_hc_sum(const float* x, float* out, const GlmHcShapes& s, int64_t T, void* stream) {
    if (T <= 0) return;
    if (!shapes_ok(s, "glm_hc_sum")) return;
    hc_sum_kernel<<<grid_for(T * s.n_embd), THREADS, 0, (cudaStream_t) stream>>>(x, out, s.n_embd, (int) s.hc,
                                                                               T * s.n_embd);
    check_launch("glm_hc_sum");
    sync_if_needed(stream, "glm_hc_sum");
}

void glm_hc_norm_bf16(const float* x, uint16_t* out_bf16, float* out_f32, const GlmHcShapes& s, int64_t T,
                      void* stream) {
    if (T <= 0) return;
    if (!shapes_ok(s, "glm_hc_norm_bf16")) return;
    const int64_t cols = s.n_embd * s.hc;
    hc_norm_bf16_kernel<<<(unsigned) T, THREADS, 0, (cudaStream_t) stream>>>(x, out_bf16, out_f32, cols,
                                                                            s.norm_eps, T);
    check_launch("glm_hc_norm_bf16");
    sync_if_needed(stream, "glm_hc_norm_bf16");
}

}  // namespace strata::kernels
