// src/kernels/cuda/glm_kda.cu - GLM-5.3-Flash's KDA layers: the decay gate, the depthwise conv and the L2 norm.
//
// **THE DELTA RULE ITSELF IS NOT HERE.**  This file is the three things that feed it and the one that reads it;
// the recurrence over the 128x128 per-head state is in `glm_delta.cu`, because that kernel is the only part of
// the model that is sequential over tokens and it wants its own headroom to be written carefully.
//
// Ported from ik_llama.cpp: the gate is `build_kda_beta_gate` (src/llama-kda.cpp), the conv is
// `ggml_compute_forward_ssm_conv_f32` (ggml/src/ggml.c), and the norm is `ggml_compute_forward_l2_norm_f32`.
#include "strata/kernels/glm.hpp"

#include <cuda_runtime.h>

#include <cmath>
#include <cstdio>
#include <cstdlib>

namespace strata::kernels {
namespace {

constexpr int THREADS = 256;

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

inline unsigned grid_for(int64_t n) { return (unsigned) ((n + THREADS - 1) / THREADS); }

/// `g = floor * sigmoid(-(a_head * (raw + dt)))`.
///
/// Three sign decisions, all of which the file settles and none of which the output reveals: the sum is
/// `raw + dt` (not a product), the sigmoid's input is NEGATED on top of `ssm_a`'s own minus (the file stores
/// `ssm_a = -exp(A_log)`, and the reference flips it back with `ggml_scale(..., -1.0f)` before the sigmoid),
/// and `floor` - negative - multiplies the sigmoid rather than being added or clamped to.  The gate lands in
/// (floor, 0), and `exp` of it is the decay: at `floor` a channel forgets, at 0 it remembers exactly.
__global__ void kda_gate_kernel(const float* __restrict__ raw, const float* __restrict__ dt,
                                const float* __restrict__ a, float* __restrict__ gate, int64_t n_v,
                                int64_t head_dim, float floor_mag, int64_t total) {
    const int64_t i = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= total) return;
    const int64_t c = i % n_v;              // position within the token: head_dim fastest
    const int64_t head = c / head_dim;
    const float v = -(a[head] * (raw[i] + dt[c]));
    gate[i] = floor_mag * (1.0f / (1.0f + expf(-v)));
}

/// The depthwise causal convolution over [q; k; v], then SiLU, in place.
///
/// **ONE THREAD PER CHANNEL, THE WHOLE TOKEN AXIS INSIDE IT.**  A channel's window is its own past, so the only
/// sequential axis in the conv is `t` - and it is sequential per channel, not across channels.  Holding the
/// `kernel - 1` taps in registers and walking `T` is therefore both the simplest and the fastest form: no
/// barriers, no state traffic between tokens, and the read-back of the state happens once.
///
/// The three streams share one state because the reference concatenates their filters into a single
/// [kernel, 3*n_v] weight before convolving - so this is not three convs that happen to be next to each other,
/// and splitting it into three would be wrong at the boundary only if the boundary moved, which it does not.
/// It is written as three weights rather than one packed buffer to avoid a 393 KiB per-layer repack at load.
/// `ld` is the TOKEN STRIDE of the three stream arrays: `3 * n_v` for the interleaved `[q;k;v]` the single-token
/// caller keeps, `n_v` for three separate `nt x n_v` blocks.  Everything else - the state's `[kernel-1, 3*n_v]`
/// layout indexed by the CONCATENATED channel `ch`, the shared history, the SiLU - is the same in both.
__global__ void kda_conv_silu3_kernel(float* __restrict__ q, float* __restrict__ k, float* __restrict__ v,
                                      const float* __restrict__ wq, const float* __restrict__ wk,
                                      const float* __restrict__ wv, float* __restrict__ state, int64_t n_v,
                                      int64_t ld, int64_t T, int K) {
    const int64_t n_all = 3 * n_v;
    const int64_t ch = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (ch >= n_all) return;

    float* __restrict__ xs = ch < n_v ? q : (ch < 2 * n_v ? k : v);
    const float* w = ch < n_v ? wq : (ch < 2 * n_v ? wk : wv);
    const int64_t c = ch < n_v ? ch : (ch < 2 * n_v ? ch - n_v : ch - 2 * n_v);
    // `w` is the file's `[K, 1, n_v]`, folded to `[K, n_v]` - so **ne0 = K IS THE FAST AXIS** and channel `c`'s
    // filter is the K floats at `c*K`.  ggml says the same thing in its addressing
    // (`c = src2->data + ir0*src2->nb[1]`, and `src2->nb[1] == src2->ne[0]*sizeof(float)`), which is the
    // authority here: the tap is the contiguous axis, NOT the channel.
    //
    // This was `w + c` indexed `wt[k*n_v]` - tap and channel swapped.  Every shape still matched, the state
    // still carried, and the parity test still passed, because the test's own reference used the same swapped
    // reading.  A self-consistent convention that disagrees with the file is exactly the bug a parity test
    // against itself cannot see.
    const float* wt = w + c * K;

    float hist[8];
    for (int k = 0; k < K - 1; ++k) hist[k] = state[k * n_all + ch];

    for (int64_t t = 0; t < T; ++t) {
        const int64_t at = t * ld + c;
        const float x = xs[at];

        // **THE SHIFT GOES AFTER THE DOT, NOT BEFORE.**  `hist` on entry to iteration `t` is
        // [x_{t-3}, x_{t-2}, x_{t-1}] - the taps of the CURRENT output.  Shifting first overwrites the oldest
        // tap with `x` itself, so the dot then read [x_{t-2}, x_{t-1}, x_t]: tap K-2 double-counted the current
        // input and the oldest sample was dropped, giving `x*(w[K-1] + w[K-2])` at t=0 instead of `x*w[K-1]`.
        // Everything still ran and the parity test still passed, because its reference shifted in the same
        // place.  Measured against the oracle at layer 0 position 0: the conv output was +5.746646 where
        // `ggml_compute_forward_ssm_conv_f32` gives +1.982379, and the implied tap was exactly w[2]+w[3].
        float sum = x * wt[K - 1];
        for (int k = 0; k < K - 1; ++k) sum += hist[k] * wt[k];

        // Guarded so a kernel-1 conv (the identity, which nothing here uses) does not write `hist[-1]`.
        if (K >= 2) {
            for (int k = 0; k + 1 < K - 1; ++k) hist[k] = hist[k + 1];
            hist[K - 2] = x;
        }
        // ggml's f32 SiLU, written the way ggml writes it - `x / (1 + expf(-x))`, not through a double.  The
        // engine's own `silu_inplace` goes through a double to match ITS reference (a numpy one); this model's
        // reference is a C kernel, and matching the kernel is the point.
        xs[at] = sum / (1.0f + expf(-sum));
    }

    for (int k = 0; k < K - 1; ++k) state[k * n_all + ch] = hist[k];
}

/// `x / max(sqrt(sum(x^2)), eps)` per head over q and k.  One warp per row, both tensors in one launch: the row
/// count is `n_head * T` and at T=1 that is 64 rows, which is a third of a wave on the smallest card - doubling
/// it with k costs nothing and saves a launch.
__global__ void kda_l2norm_kernel(float* __restrict__ q, float* __restrict__ k, int64_t head_dim, float eps,
                                  int64_t rows) {
    const int64_t r = ((int64_t) blockIdx.x * (blockDim.x >> 5)) + (threadIdx.x >> 5);
    if (r >= 2 * rows) return;
    float* x = (r < rows) ? q + r * head_dim : k + (r - rows) * head_dim;

    const int lane = threadIdx.x & 31;
    float acc = 0.0f;
    for (int64_t c = lane; c < head_dim; c += 32) acc += x[c] * x[c];
    for (int off = 16; off > 0; off >>= 1) acc += __shfl_down_sync(0xFFFFFFFFu, acc, off);

    float inv = 0.0f;
    if (lane == 0) inv = 1.0f / fmaxf(sqrtf(acc), eps);
    inv = __shfl_sync(0xFFFFFFFFu, inv, 0);
    for (int64_t c = lane; c < head_dim; c += 32) x[c] *= inv;
}

}  // namespace

void glm_kda_gate(const float* raw, const float* dt, const float* a, float* gate, int64_t n_v, int64_t n_head,
                  int64_t head_dim, float floor_mag, int64_t T, void* stream) {
    if (T <= 0 || n_v <= 0) return;
    if (n_head * head_dim != n_v) {
        std::fprintf(stderr, "glm_kda_gate: n_head %lld * head_dim %lld != n_v %lld\n", (long long) n_head,
                     (long long) head_dim, (long long) n_v);
        return;
    }
    kda_gate_kernel<<<grid_for(n_v * T), THREADS, 0, (cudaStream_t) stream>>>(raw, dt, a, gate, n_v, head_dim,
                                                                             floor_mag, n_v * T);
    check_launch("glm_kda_gate");
    sync_if_needed(stream, "glm_kda_gate");
}

void glm_kda_conv_silu3(float* q, float* k, float* v, const float* wq, const float* wk, const float* wv,
                        float* state, int64_t n_v, int64_t ld, int64_t T, int64_t kernel, void* stream) {
    if (T <= 0 || n_v <= 0) return;
    if (kernel < 1 || kernel > 8) {
        std::fprintf(stderr, "glm_kda_conv_silu3: kernel %lld is outside 1..8\n", (long long) kernel);
        return;
    }
    if (ld < n_v) {
        std::fprintf(stderr, "glm_kda_conv_silu3: token stride %lld is narrower than n_v %lld\n", (long long) ld,
                     (long long) n_v);
        return;
    }
    kda_conv_silu3_kernel<<<grid_for(3 * n_v), THREADS, 0, (cudaStream_t) stream>>>(q, k, v, wq, wk, wv, state, n_v,
                                                                                   ld, T, (int) kernel);
    check_launch("glm_kda_conv_silu3");
    sync_if_needed(stream, "glm_kda_conv_silu3");
}

void glm_kda_conv_silu(float* qkv, const float* wq, const float* wk, const float* wv, float* state, int64_t n_v,
                       int64_t T, int64_t kernel, void* stream) {
    // The interleaved form IS the split form at `ld = 3*n_v`, so there is one implementation of the convolution
    // and not two.  With `ld = 3*n_v`, `xs[at] = xs[t*3*n_v + c]` is `qkv[t*3*n_v + c]` for the q stream,
    // `qkv[t*3*n_v + n_v + c]` for k and `qkv[t*3*n_v + 2*n_v + c]` for v - which is what the old body wrote.
    glm_kda_conv_silu3(qkv, qkv + n_v, qkv + 2 * n_v, wq, wk, wv, state, n_v, 3 * n_v, T, kernel, stream);
}

void glm_kda_l2norm(float* q, float* k, int64_t n_head, int64_t head_dim, float eps, int64_t T, void* stream) {
    if (T <= 0 || n_head <= 0 || head_dim <= 0) return;
    const int64_t rows = n_head * T;
    const unsigned warps_per_block = 8;
    const unsigned grid = (unsigned) ((2 * rows + warps_per_block - 1) / warps_per_block);
    kda_l2norm_kernel<<<grid, warps_per_block * 32, 0, (cudaStream_t) stream>>>(q, k, head_dim, eps, rows);
    check_launch("glm_kda_l2norm");
    sync_if_needed(stream, "glm_kda_l2norm");
}

}  // namespace strata::kernels
