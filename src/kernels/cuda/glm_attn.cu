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

/// How many latent slots one lane of the batched kernel carries, and so the widest `kv_lora` it takes: 16
/// covers 512, which is this model's rank and the parity test's larger case.  A wider latent falls back to
/// `mla_attn_kernel`, which carries eight and so reaches 1024 - stale code is worth less than a wrong answer,
/// but silently dropping half a row is worse than either.
constexpr int SPLIT_MAX_SLOTS = 16;

/// Heads a warp carries in the batched kernel.  2 halves the blocks and the cache rows they walk between
/// them, but doubles the query registers a lane holds and the accumulator it rescales when a row raises
/// the running max.  Which way that lands at this model's shapes is UNMEASURED - it is 1, and the `<2>`
/// instantiation in the dispatch below is there so that measuring it is a one-line change.
constexpr int MLA_HG = 1;

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

/// ONE WARP A (head, query) PAIR, AND NO BARRIER ANYWHERE IN THE CACHE WALK.
///
/// **WHAT THE KERNEL ABOVE COSTS, MEASURED.** `mla_attn_kernel` is one 128-thread block per (head, query) with
/// one thread per latent slot, and it reduces over the block for EVERY cache row: `red[lane] = part`, then
/// seven `__syncthreads()` to sum 512 products.  Timed standalone on an RTX 5060 at this model's shapes
/// (`n_head` 64, `kv_lora_rank` 512), on a 5060's 30 SMs:
///
///     n_kv    512: 0.3501 ms a call    (0.684 us a cache row)
///     n_kv   1024: 0.6941 ms a call    (0.678 us a cache row)
///     n_kv   2048: 1.3865 ms a call    (0.677 us a cache row)
///
/// Dead linear in `n_kv`, and 0.68 us for 1 KB of cache row is **~1.5 GB/s** against the card's 391 GB/s
/// measured D2D roofline.  The kernel is not moving bytes; it is synchronising.  A 4096-token chunk then
/// costs a whole cache walk per token per head - 64 heads x 8.4 M rows - and the engine's own section timer
/// charges the attention mixer 30.6 s of a 4096-token chunk's 12 layers, 82% of the card's `pre`, of which
/// this kernel is the larger part.
///
/// Here a warp owns the whole 512-wide (head, query) dot - 16 slots a lane, summed by five `__shfl_xor` with
/// no barrier at all - and a block's eight warps all work on the SAME query, so the eight reads of one cache
/// row are one L1 miss and seven hits.  Blocks in the same `blockIdx.y` re-read the same rows, but the whole
/// cache at this model's ceiling is 8192 x 512 x 2 B = 8 MiB and stays resident in L2, which is what the
/// standalone timing above shows the second sweep at n_kv 4096 paying for.
///
/// THE ONLINE SOFTMAX IS THE OTHER HALF OF THE COST, and it is a comparison, not an `expf`.  In the usual
/// form every row needs two exponentials - `p = exp(s - m_new)` and the rescale `corr = exp(m - m_new)` - and
/// a full 512-wide multiply-add over the accumulator *even when `corr` is 1*.  Writing the update as two
/// branches makes `corr = 1` and `p = 1` fall out of the branch that is taken when a row does NOT raise the
/// maximum, which after the first few rows is almost every row: `m` is the running max of a softmax, so it
/// stops moving early and the rescale runs O(log n_kv) times instead of n_kv.  The arithmetic is the same
/// online softmax, reassociated - `l = l*corr + p` with `p == 1` is `l = l*corr + 1` - and it is within
/// `close_enough` of the reference the parity harness already runs for T > 1.
template <int HG, int NSLOT>
__global__ void mla_attn_nt_kernel(const float* __restrict__ q, const __half* __restrict__ kc,
                                   float* __restrict__ out, int nh, int kv, int n_kv, int pos_base, float scale) {
    const int nwarp = (int) (blockDim.x >> 5);
    const int warp = (int) (threadIdx.x >> 5);
    const int lane = (int) (threadIdx.x & 31);
    const int t = (int) blockIdx.y;
    const int h0 = ((int) blockIdx.x * nwarp + warp) * HG;
    if (h0 >= nh) return;
    const int NPER = (kv + 31) / 32;   // 16 at kv_lora 512; 1 for the parity test's kv 8
    // **THE HEAD IS THE OUTERMOST AXIS AND THE TOKEN IS INSIDE IT** - `(h*T + t)*kv`, the contract the header
    // states and the parity test's "head and token swapped" rival pins down.  The engine's per-token calls
    // could not tell the two apart (at T == 1 both indexings collapse to `h*kv`), so this is the first kernel
    // that has to get it right, and getting it wrong is a silent permutation of the same elements.
    const int64_t T = gridDim.y;
    const float* Q = q + ((size_t) h0 * T + t) * kv;
    float* O = out + ((size_t) h0 * T + t) * kv;

    float qv[HG][NSLOT];
    float acc[HG][NSLOT];
#pragma unroll
    for (int j = 0; j < HG; ++j) {
#pragma unroll
        for (int i = 0; i < NSLOT; ++i) {
            const int c = lane + i * 32;
            qv[j][i] = (i < NPER && c < kv) ? Q[j * kv + c] : 0.0f;   // 0 keeps a short row out of the shuffle
            acc[j][i] = 0.0f;
        }
    }
    float m[HG], l[HG];
#pragma unroll
    for (int j = 0; j < HG; ++j) {
        m[j] = -INFINITY;
        l[j] = 0.0f;
    }

    const int limit = (pos_base + t + 1 < n_kv) ? pos_base + t + 1 : n_kv;
    const unsigned full = 0xffffffffu;
    for (int s = 0; s < limit; ++s) {
        const __half* K = kc + (size_t) s * kv;
        float k[NSLOT];
#pragma unroll
        for (int i = 0; i < NSLOT; ++i) {
            const int c = lane + i * 32;
            // The load is PREDICATED, not merely multiplied by zero: a lane past the end of a short row would
            // otherwise read into the next row, and a zero times a NaN is a NaN, not a zero.
            k[i] = (i < NPER && c < kv) ? __half2float(K[c]) : 0.0f;
        }
#pragma unroll
        for (int j = 0; j < HG; ++j) {
            float part = 0.0f;
#pragma unroll
            for (int i = 0; i < NSLOT; ++i) part += qv[j][i] * k[i];
#pragma unroll
            for (int off = 16; off > 0; off >>= 1) part += __shfl_xor_sync(full, part, off);
            const float score = part * scale;
            // Uniform across the warp - every lane reduced to the same `part` - so neither branch diverges.
            if (score > m[j]) {
                const float corr = (m[j] == -INFINITY) ? 0.0f : __expf(m[j] - score);
                l[j] = l[j] * corr + 1.0f;   // p == exp(score - score) == 1
#pragma unroll
                for (int i = 0; i < NSLOT; ++i) acc[j][i] = acc[j][i] * corr + k[i];
                m[j] = score;
            } else {
                const float p = __expf(score - m[j]);
                l[j] += p;
#pragma unroll
                for (int i = 0; i < NSLOT; ++i) acc[j][i] += p * k[i];
            }
        }
    }

#pragma unroll
    for (int j = 0; j < HG; ++j) {
        const float inv = l[j] > 0.0f ? 1.0f / l[j] : 0.0f;
#pragma unroll
        for (int i = 0; i < NSLOT; ++i) {
            const int c = lane + i * 32;
            if (i < NPER && c < kv) O[j * kv + c] = acc[j][i] * inv;
        }
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
    // **T > 1 TAKES THE BATCHED KERNEL, AND T == 1 KEEPS THIS ONE.**  Not for compatibility - for occupancy.
    // The batched kernel's grid is (heads a block, T), so at T == 1 it is eight blocks of 256 threads, 2048
    // threads against a 30-SM card's 61,440 slots, and a block-reduction kernel's 64 blocks of 128 is the
    // better of two bad shapes when there is only one query to spread.  A chunk never has T == 1 (the engine
    // groups by `GLM_MAX_NTOK`), and decode never has T > 1, so the split costs nothing at either end.  What a
    // one-query decode actually wants is the cache split across blocks - see the note in the PR.
    if (T > 1 && kv_lora <= (int64_t) 32 * SPLIT_MAX_SLOTS) {
        // 8 warps a block, one head a warp: the block's eight reads of a cache row are one miss and seven
        // L1 hits, and `n_head` 64 / 8 = 8 blocks of them walk the same rows.  The grid's y is the token.
        const int nwarp = (int) (ATTN_THREADS / 32);
        const int per_block = nwarp * MLA_HG;
        const int hblocks = (int) ((n_head + per_block - 1) / per_block);
        dim3 grid((unsigned) hblocks, (unsigned) T);
        if (MLA_HG == 1) {
            mla_attn_nt_kernel<1, SPLIT_MAX_SLOTS><<<grid, ATTN_THREADS, 0, (cudaStream_t) stream>>>(
                q, (const __half*) k_cache, out, (int) n_head, (int) kv_lora, (int) n_kv, (int) pos_base, scale);
        } else {
            mla_attn_nt_kernel<2, SPLIT_MAX_SLOTS><<<grid, ATTN_THREADS, 0, (cudaStream_t) stream>>>(
                q, (const __half*) k_cache, out, (int) n_head, (int) kv_lora, (int) n_kv, (int) pos_base, scale);
        }
        check_launch("glm_mla_attn");
        sync_if_needed(stream, "glm_mla_attn");
        return;
    }
    dim3 grid((unsigned) n_head, (unsigned) T);
    mla_attn_kernel<<<grid, ATTN_THREADS, 0, (cudaStream_t) stream>>>(
        q, (const __half*) k_cache, out, (int) kv_lora, (int) n_kv, (int) pos_base, scale);
    check_launch("glm_mla_attn");
    sync_if_needed(stream, "glm_mla_attn");
}

}  // namespace strata::kernels
