// src/kernels/cuda/glm_attn.cu - GLM-5.3-Flash's absorbed NoPE MLA over the shared latent.
//
// WHAT MAKES THIS NOT THE ENGINE'S EXISTING ATTENTION, in one list:
//
//   * **K AND V ARE THE SAME TENSOR.**  The cache holds the 512-wide `kv_lora` latent per token and one token's
//     VALUE is that latent verbatim - there is no separate V projection into the cache.  `wv_b` de-absorbs
//     after the attention, not before it.
//   * **ONE KV HEAD, NOT `n_head_kv`.**  Every one of the 64 query heads reads the same latent.
//   * **NO ROPE.**  `rope.dimension_count` is 0, so there is nothing to rotate and nothing to rotate by - the
//     only positional signal on this layer is the causal mask.
//   * **THE QUERY IS ABSORBED.**  `Qcur = wk_b @ q` maps each 256-wide query head back into the 512-wide
//     latent space, so the score is `sum_l Qcur[l] * K[l]` over the LATENT - a 512-wide dot, not 256.
//   * scale is `1/sqrt(256)` - `n_embd_head_k_full`, the per-head width BEFORE absorption - not
//     `1/sqrt(512)`.
//
// The reference is `build_glm5next_mla_attention`'s `ggml_flash_attn_ext(Qcur, K_cache, K_cache, mask, kq_scale)`.
// Online softmax, because the alternative is materialising `[n_kv, T, n_head]` scores, which is exactly what
// the reference's own non-flash branch does and warns about.
//
// LAYOUT.  `q` and `out` are `[kv_lora, T, n_head]` with kv_lora fastest - `ggml_permute(reshape(q, 256, 64, T),
// 0, 2, 1, 3)` puts the head on the outermost axis and leaves the 512 latent contiguous, which is the same
// `[width, tokens]` convention the rest of this engine uses once the head is the batch.  The cache is
// `[n_kv, kv_lora]` with kv_lora fastest, fp16, written one row per token.
#include "strata/kernels/glm.hpp"

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cstdio>
#include <cstdlib>

namespace strata::kernels {
namespace {

constexpr int ATTN_THREADS = 128;

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

/// ONE BLOCK PER (head, query token); one thread per latent slot, four slots per thread at kv_lora 512.
///
/// The online-softmax accumulator lives in REGISTERS, one per slot the thread owns, so the only shared state is
/// the per-step reduction.  A thread's slots are `lane + i*ATTN_THREADS`, so the four loads of a step coalesce.
__global__ void mla_attn_kernel(const float* __restrict__ q, const __half* __restrict__ kc,
                                float* __restrict__ out, int kv, int n_kv, int pos_base, float scale) {
    const int h = blockIdx.x;
    const int t = blockIdx.y;
    const int lane = threadIdx.x;
    const int NPER = (kv + ATTN_THREADS - 1) / ATTN_THREADS;   // 4 at kv 512
    const int64_t T = gridDim.y;
    // **BOTH THE HEAD AND THE TOKEN STRIDE, AND THE FIRST CUT HAD ONLY THE HEAD.**  `q`/`out` are
    // `[kv_lora, T, n_head]` with the head outermost, so head `h`, token `t` starts at `(h*T + t)*kv`.  With
    // `t` missing, every token of a window READ the same query row and WROTE the same output row - and the
    // per-token `limit` still varied, so the result was finite, plausible, and only wrong for T > 1, which is
    // why the T = 1 decode path this engine actually drives would never have shown it.  `glm_parity`'s
    // multi-token MLA case is what caught it.
    const float* Q = q + ((int64_t) h * T + t) * kv;
    float* O = out + ((int64_t) h * T + t) * kv;

    float acc[8];
#pragma unroll
    for (int i = 0; i < 8; ++i) acc[i] = 0.0f;
    float m = -INFINITY;
    float l = 0.0f;

    // A query at absolute position `pos_base + t` sees every cached token up to and including itself.
    const int limit = pos_base + t + 1 < n_kv ? pos_base + t + 1 : n_kv;

    __shared__ float red[ATTN_THREADS];
    for (int s = 0; s < limit; ++s) {
        const __half* K = kc + (int64_t) s * kv;
        float part = 0.0f;
#pragma unroll
        for (int i = 0; i < 8; ++i) {
            const int c = lane + i * ATTN_THREADS;
            if (i < NPER && c < kv) part += Q[c] * __half2float(K[c]);
        }
        red[lane] = part;
        __syncthreads();
        for (int step = ATTN_THREADS / 2; step > 0; step >>= 1) {
            if (lane < step) red[lane] += red[lane + step];
            __syncthreads();
        }
        const float score = red[0] * scale;
        __syncthreads();

        const float m_new = fmaxf(m, score);
        // `expf(-inf - m_new)` is 0, which is what the first step wants: the running sum starts empty.
        const float corr = (m == -INFINITY) ? 0.0f : expf(m - m_new);
        const float p = expf(score - m_new);
        l = l * corr + p;
#pragma unroll
        for (int i = 0; i < 8; ++i) {
            const int c = lane + i * ATTN_THREADS;
            if (i < NPER && c < kv) acc[i] = acc[i] * corr + p * __half2float(K[c]);
        }
        m = m_new;
        __syncthreads();
    }

    const float inv = l > 0.0f ? 1.0f / l : 0.0f;
#pragma unroll
    for (int i = 0; i < 8; ++i) {
        const int c = lane + i * ATTN_THREADS;
        if (i < NPER && c < kv) O[c] = acc[i] * inv;
    }
}

/// `[t][h][w] -> [h][t][w]`: swaps the two outer axes of a `n_tok x n_head` stack of `w`-wide slices, and is
/// its own inverse (call it again with `n_head` and `n_tok` exchanged and the operands swapped).
///
/// This exists so the absorbed-MLA bands can be projected one head at a time over a WHOLE GROUP of tokens.
/// `project_rows` takes `ncols` activation columns that are CONTIGUOUS and `w` apart, which token-major
/// `qfull`/`kqv` are not - the head is the fastest axis and the stride between one token's copy of head `h`
/// and the next is `n_head * w`.  Head-major makes the group's columns adjacent, which is the only thing
/// standing between the band loop and one call per head per group instead of one call per head per token.
///
/// One block per (head, token) pair, threads walking the width: both the read and the write are coalesced on
/// the inner axis.  At this model's shapes the move is `n_head * ntok * w` floats - for a group of eight, 512
/// KiB for the 256-wide absorption and 1 MiB for the 512-wide de-absorption - which is a launch and not a
/// kernel.
__global__ void heads_major_kernel(const float* __restrict__ src, float* __restrict__ dst, int nh, int ntok,
                                   int w) {
    const int ht = blockIdx.y;
    const int h = ht / ntok, t = ht - h * ntok;
    const int64_t i = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= w) return;
    dst[((int64_t) h * ntok + t) * w + i] = src[((int64_t) t * nh + h) * w + i];
}

/// `x` is `[kv, T]` (the width fastest, the engine's convention) and row `t` lands at cache row `pos + t`.
/// `blockIdx.y` is the token: with `T == 1` the grid collapses to the same one-block-per-256-channels shape the
/// single-token path always used, and the index arithmetic is `pos * kv + i` either way.
__global__ void mla_cache_store_kernel(const float* __restrict__ x, __half* __restrict__ cache, int kv, int64_t pos) {
    const int64_t i = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= kv) return;
    const int64_t t = blockIdx.y;
    cache[(pos + t) * kv + i] = __float2half(x[t * kv + i]);
}

}  // namespace

void glm_mla_cache_store(const float* x, uint16_t* cache, int64_t pos, int64_t kv_lora, void* stream) {
    glm_mla_cache_store_t(x, cache, pos, kv_lora, 1, stream);
}

void glm_mla_cache_store_t(const float* x, uint16_t* cache, int64_t pos, int64_t kv_lora, int64_t T, void* stream) {
    if (kv_lora <= 0 || T <= 0) return;
    __half* c = (__half*) cache;
    const int blocks = (int) ((kv_lora + 255) / 256);
    dim3 grid((unsigned) blocks, (unsigned) T);
    mla_cache_store_kernel<<<grid, 256, 0, (cudaStream_t) stream>>>(x, c, (int) kv_lora, pos);
    check_launch("glm_mla_cache_store");
    sync_if_needed(stream, "glm_mla_cache_store");
}

void glm_heads_major(const float* src, float* dst, int64_t n_head, int64_t n_tok, int64_t width, void* stream) {
    if (n_head <= 0 || n_tok <= 0 || width <= 0 || src == dst) return;
    if (n_head * n_tok > 65535) {
        std::fprintf(stderr, "glm_heads_major: %lld heads x %lld tokens exceeds the grid's %d\n",
                     (long long) n_head, (long long) n_tok, 65535);
        return;
    }
    const int w = (int) width;
    dim3 grid((unsigned) ((w + 255) / 256), (unsigned) (n_head * n_tok));
    heads_major_kernel<<<grid, 256, 0, (cudaStream_t) stream>>>(src, dst, (int) n_head, (int) n_tok, w);
    check_launch("glm_heads_major");
    sync_if_needed(stream, "glm_heads_major");
}

void glm_mla_attn(const float* q, const uint16_t* k_cache, float* out, int64_t n_head, int64_t kv_lora,
                  int64_t n_kv, int64_t T, int64_t pos_base, float scale, void* stream) {
    if (T <= 0 || n_head <= 0 || kv_lora <= 0 || n_kv <= 0) return;
    if (kv_lora > (int64_t) ATTN_THREADS * 8) {
        std::fprintf(stderr, "glm_mla_attn: kv_lora %lld exceeds the %d slots a thread carries\n",
                     (long long) kv_lora, ATTN_THREADS * 8);
        return;
    }
    dim3 grid((unsigned) n_head, (unsigned) T);
    mla_attn_kernel<<<grid, ATTN_THREADS, 0, (cudaStream_t) stream>>>(
        q, (const __half*) k_cache, out, (int) kv_lora, (int) n_kv, (int) pos_base, scale);
    check_launch("glm_mla_attn");
    sync_if_needed(stream, "glm_mla_attn");
}

}  // namespace strata::kernels
