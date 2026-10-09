// src/kernels/cuda/glm_dsa.cu - glm5-next's DSA indexer and the sparse attention it selects for.
// The math, the layouts and the `select_tail` disagreement are in `glm_dsa.hpp`; this file is the arithmetic.
//
// Four kernels:
//
//   pool      one thread per (key-dim element, pool): the kpool logits, then the weighted sum.  **The softmax
//             is PER KEY-DIM ELEMENT** - the reference permutes (kpool, key_dim) and normalises over the member
//             axis - so the loop below is over `m` with `e` fixed, and nothing here touches a second `e`.
//   score     one thread per (query, pool): a relu'd dot per indexer head, weighted and summed.
//   select    one thread per query: an MSD radix pass over the VISIBLE pools finds the `top_pools`-th largest
//             score, and a stable radix sort of the picks it lets through puts them in the descending-score order
//             the header promises.  O(n_vis) a pass, four passes, then four counting passes over the picks.  While
//             fewer than `top_pools` pools are visible there is no boundary to find and the walk is skipped - but
//             the SORT is not, because "all of them are selected" is a statement about the set and not the order.
//             **THIS WAS O(n_vis^2) - the router rank formula, recomputed pool by pool - and the note here said
//             a radix top-k "would only pay at prefill lengths".  This model's prefill IS long lengths, and the
//             cost was measured rather than reasoned about:** at 16k context the caller launches this once per
//             token with `n_pools` = the pools so far, so ONE THREAD made 4,096^2 = 16.8M comparisons, 512
//             times a chunk, per MLA layer.  The chunk's `attn` section was 210.0 s of a 210.8 s card time
//             against 21.3 s for the same chunk on the dense path - a 10x regression whose whole cause was this
//             loop.  See `glm_dsa_key` for the tie order, which is preserved.
//   attend    one block per (query token, head): scores over the row's cells, softmax on one thread (n_sel is
//             ~2051 values), ctx in shared, and the -1 cells EXCLUDED - which is numerically the graph's -inf
//             additive mask rather than a branch that could drift from it.
//
// All arithmetic in f32 with precise `expf`.  `__expf`'s error is scaled by the argument, and the pool logits
// here are not small - they are a learned gate plus a learned per-member embedding - so the fast intrinsic's few
// ulp at |x| ~ 10 is visible in the selection this whole path exists to make.  `rsqrtf` is used for the
// attention scale only, where the value is a constant.
#include "strata/kernels/glm_dsa.hpp"

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <atomic>
#include <cstdint>
#include <cstdio>
#include <cstdlib>

namespace strata::kernels {

/// The opt-in switch.  One atomic read per layer per token, and an atomic because the CLI sets it on the main
/// thread while a session may already have been created - the alternative (a plain bool) is a data race the
/// standard does not forgive just because it is set once before any read in practice.
namespace {
std::atomic<bool> dsa_enabled{false};
}  // namespace

void glm_dsa_set_enabled(bool enabled) { dsa_enabled.store(enabled, std::memory_order_relaxed); }
bool glm_dsa_enabled() { return dsa_enabled.load(std::memory_order_relaxed); }

namespace {

constexpr int DS_THREADS = 256;
/// The slots one thread owns in `glm_dsa_attn_kernel`, exactly as many as `glm_mla_attn`'s: kv_lora 512 over
/// 256 threads is two of them, and the launcher refuses a wider latent rather than truncating the copy.
constexpr int DS_SLOTS = 8;
/// How many selected pools `glm_dsa_select_kernel` will rank in one thread's local array.  Device code has no
/// variable-length array, and this file's rule is that a bound is refused rather than truncated (`DS_SLOTS`
/// above, `glm_dsa_pool`'s pool-width check) - so `top_pools` past this falls back to the rank formula run
/// over every visible pool, which is what this kernel did before and is correct at any width.  This model's
/// `idx_top_k / idx_kpool` is 2048/4 = 512, exactly the bound.
constexpr int DS_MAX_TOP = 512;

void check_launch(const char* what) {
    const cudaError_t e = cudaGetLastError();
    if (e != cudaSuccess) std::fprintf(stderr, "%s launch: %s\n", what, cudaGetErrorString(e));
}

void sync_if_needed(void* stream, const char* what) {
    if (stream != nullptr) return;
    const cudaError_t e = cudaDeviceSynchronize();
    if (e != cudaSuccess) {
        std::fprintf(stderr, "%s: %s\n", what, cudaGetErrorString(e));
        std::exit(1);
    }
}

/// `logits[m] = ig[e, p*kpool+m] + ape[e, m]`, and the softmax over `m` for this ONE `e`.
/// The three passes recompute `expf` rather than caching it: identical arguments give identical results, so the
/// arithmetic is the reference's, and the kernel needs no fixed-size array and therefore no upper bound on
/// `kpool` - an array would have to be sized for the largest pool the model might carry and a pool larger than
/// that would be a silent truncation instead of a wrong answer.
__global__ void glm_dsa_pool_kernel(const float* __restrict__ ik, const float* __restrict__ ig,
                                    const float* __restrict__ ape, int key_dim, int kpool, int n_pools,
                                    float* __restrict__ pooled) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= key_dim * n_pools) return;
    const int p = idx / key_dim;
    const int e = idx - p * key_dim;
    const float* ge = ig + (size_t) e + (size_t) key_dim * p * kpool;
    const float* ke = ik + (size_t) e + (size_t) key_dim * p * kpool;
    const float* ae = ape + e;

    float mx = -INFINITY;
    for (int m = 0; m < kpool; ++m) mx = fmaxf(mx, ge[(size_t) key_dim * m] + ae[(size_t) key_dim * m]);
    float denom = 0.0f;
    for (int m = 0; m < kpool; ++m) denom += expf(ge[(size_t) key_dim * m] + ae[(size_t) key_dim * m] - mx);
    float acc = 0.0f;
    for (int m = 0; m < kpool; ++m)
        acc += (expf(ge[(size_t) key_dim * m] + ae[(size_t) key_dim * m] - mx) / denom) * ke[(size_t) key_dim * m];
    pooled[(size_t) e + (size_t) key_dim * p] = acc;
}

__global__ void glm_dsa_score_kernel(const float* __restrict__ iq, const float* __restrict__ pooled,
                                     const float* __restrict__ weights, int key_dim, int idx_heads,
                                     int n_tokens, int n_pools, float* __restrict__ score) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n_tokens * n_pools) return;
    const int p = idx % n_pools;
    const int t = idx / n_pools;
    const float* pt = pooled + (size_t) key_dim * p;
    const float* qt = iq + (size_t) key_dim * idx_heads * t;

    float acc = 0.0f;
    for (int h = 0; h < idx_heads; ++h) {
        const float* qh = qt + (size_t) key_dim * h;
        float dot = 0.0f;
        for (int e = 0; e < key_dim; ++e) dot += qh[e] * pt[e];
        acc += fmaxf(dot, 0.0f) * weights[(size_t) h + (size_t) idx_heads * t];
    }
    score[(size_t) p + (size_t) n_pools * t] = acc;
}

/// The order-preserving unsigned image of a float: comparing two of these as integers is comparing the two
/// floats, so a selection over `unsigned` keys is a selection over scores.  The sign is flipped and the rest
/// inverted for negatives, which is what makes the negatives sort below the positives.
__device__ __forceinline__ unsigned glm_dsa_key(float f) {
    unsigned b = __float_as_uint(f);
    // -0.0 and +0.0 are `==` as floats but are two different bit patterns, and the picks are ranked by a float
    // comparison afterwards.  Without this they would be one key here and one tied pair there, so a -0.0 could
    // be taken over a +0.0 at the boundary that the rank formula would have resolved by index.
    if ((b & 0x7FFFFFFFu) == 0u) b = 0u;
    return (b & 0x80000000u) ? ~b : (b | 0x80000000u);
}

__global__ void glm_dsa_select_kernel(const float* __restrict__ score, int n_pools, int kpool, int top_pools,
                                      int select_tail, int n_tokens, int n_sel, const int* __restrict__ pos,
                                      int* __restrict__ cells) {
    const int t = blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= n_tokens) return;
    // A pool is visible to this query only when its LAST member is at or before the query's own cell:
    // pool p covers p*kpool .. p*kpool+kpool-1, so p is visible iff (p+1)*kpool - 1 <= pos[t].
    const int n_vis = (pos[t] + 1) / kpool;
    int* row = cells + (size_t) n_sel * t;
    for (int s = 0; s < n_sel; ++s) row[s] = -1;

    if (n_vis > 0) {
        const float* __restrict__ sc = score + (size_t) n_pools * (size_t) t;
        if (top_pools <= DS_MAX_TOP) {
            // ---- the boundary: the `top_pools`-th largest key, by eight bits a pass from the top.
            //
            // INVARIANT: every key is either STRICTLY ABOVE the range `prefix .. prefix | undecided` or inside
            // it, and `above` counts the former.  A pass splits the range into 256 sub-ranges by its next eight
            // bits and walks them from the highest down, because "how many keys are above this one" is a
            // cumulative count and that is exactly what the quadratic loop was recomputing 4,096 times.
            //
            // After four passes `undecided` is zero, so the range IS one key: `prefix` is the `top_pools`-th
            // largest, `above` how many beat it, and `top_pools - above` of the keys that equal it are still
            // needed.
            //
            // When every visible pool is selected - the whole of a prompt's first `top_pools` pools, and every
            // query until the context reaches `top_pools * kpool` cells - there is nothing to cut, so the walk is
            // skipped entirely.  **THE PICKS STILL HAVE TO BE SORTED, AND THE FIRST VERSION OF THIS FAST PATH
            // WROTE THEM IN POOL ORDER INSTEAD.**  "All of them are selected" says which cells are in `row` and
            // nothing about where; the header's DESCENDING SCORE order is a separate promise, the rank formula
            // kept it here too (every rank is below `top_pools`, so all `n_vis` wrote out, ranked), and the
            // attention permutes its summation over the row.  Measured: 344 tokens, greedy, `--dsa` against
            // dense - `A mixture-of-experts (MoE` against `# Mixture-of-Experts Routing in`, the engine's own
            // below-the-budget invariant broken by a rewrite that was only supposed to make the walk cheaper.
            unsigned kth = 0u;
            int need = 0;
            const bool all_visible = n_vis <= top_pools;
            if (!all_visible) {
                unsigned prefix = 0u, undecided = 0xFFFFFFFFu;
                int above = 0;
                int hist[256];
                for (int pass = 0; pass < 4; ++pass) {
                    const int shift = 24 - 8 * pass;
                    for (int b = 0; b < 256; ++b) hist[b] = 0;
                    const unsigned decided = ~undecided;
                    for (int p = 0; p < n_vis; ++p) {
                        const unsigned k = glm_dsa_key(__ldg(sc + p));
                        if ((k & decided) != prefix) continue;
                        ++hist[(int) ((k >> shift) & 0xFFu)];
                    }
                    int acc = 0, chosen = 0;
                    for (int b = 255; b >= 0; --b) {
                        // Fewer than `top_pools` keys are above this sub-range's bottom, so the boundary is at
                        // or below it - take the whole sub-range as "above" and keep looking downward.
                        if (above + acc + hist[b] < top_pools) { acc += hist[b]; continue; }
                        chosen = b;
                        break;
                    }
                    above += acc;
                    prefix |= ((unsigned) chosen) << shift;
                    undecided &= ~(0xFFu << shift);
                }
                kth = prefix;
                need = top_pools - above;   // >= 1: `above` < top_pools is what stopped the walk
            }

            // ---- the picks, ordered the way the header promises.  `cells` is in DESCENDING SCORE order, which
            // `glm_parity`'s `dsa_select` case asserts element by element against the reference and which its
            // "cells in cell order" rival must fail - above the budget and below it alike.  The walk above hands
            // the picks back in POOL INDEX order, so they are sorted here - by the same rule the rank formula
            // encoded, without the rank formula.
            //
            // **THE RANK FORMULA WAS O(top_pools^2) A CALL, AND IT SHOWED.**  Counting for each pick how many
            // picks beat it is 262,144 comparisons at this model's `top_pools` = 512, and the caller makes one
            // call per token per MLA layer - 5,632 a chunk of 512.  Measured on the 16K run, the card's `pre`
            // jumped 5.8 s -> 14.7 s per stage-chunk exactly at the 2,048-cell context where the selection
            // switches from "every visible pool" to this path, and then held ~15.0 s while the walk's own
            // O(n_vis) passes grew by 0.2 s over the next seven chunks: the comparisons were ~9.3 s of it.
            //
            // A stable LSD radix sort is the same ordering for a counting cost.  Sorting `~key` ascending is
            // `key` descending, and stability is what carries the tie rule: the picks were collected in index
            // order, so equal keys come out lower index first - exactly the reference's tie-break - with no
            // comparison between them.  Four passes of eight bits over `top_pools` entries is 4 * 1,024 counting
            // operations where the formula spent 262,144 comparisons.
            //
            // The key and its pool are one 64-bit entry, `~key` high and the index low, and the passes read the
            // key's four bytes only: leaving the index out of the sort is what makes it stable and not a
            // secondary sort, and the swap count is even so `src` ends on `ent`.
            uint64_t ent[DS_MAX_TOP], tmp[DS_MAX_TOP];
            int npick = 0, taken = 0;
            for (int p = 0; p < n_vis; ++p) {
                const unsigned k = glm_dsa_key(__ldg(sc + p));
                // Ties go to the LOWER index, as the router's top-k does: the equals are taken in index order
                // until `need` of them are used up, which is the same rule the rank formula encodes.
                const bool hit = all_visible || (k > kth) || (k == kth && taken < need);
                if (!hit) continue;
                ent[npick++] = ((uint64_t) (~k) << 32) | (uint32_t) p;
                if (!all_visible && k == kth) ++taken;
            }
            uint64_t* src = ent;
            uint64_t* dst = tmp;
            int cnt[256];
            for (int pass = 0; pass < 4; ++pass) {
                const int shift = 32 + 8 * pass;
                for (int b = 0; b < 256; ++b) cnt[b] = 0;
                for (int i = 0; i < npick; ++i) ++cnt[(int) ((src[i] >> shift) & 0xFFu)];
                int acc = 0;
                for (int b = 0; b < 256; ++b) { const int c = cnt[b]; cnt[b] = acc; acc += c; }
                for (int i = 0; i < npick; ++i) {
                    const int b = (int) ((src[i] >> shift) & 0xFFu);
                    dst[cnt[b]++] = src[i];
                }
                uint64_t* sw = src; src = dst; dst = sw;
            }
            for (int i = 0; i < npick; ++i) {
                const int p = (int) (src[i] & 0xFFFFFFFFu);
                for (int m = 0; m < kpool; ++m) row[i * kpool + m] = p * kpool + m;
            }
        } else {
            // `top_pools` is wider than one thread's array.  The rank formula over every visible pool, which is
            // this kernel's first form: O(n_vis^2), correct at any width, and it is what a model with a larger
            // `idx_top_k` keeps getting rather than a truncated selection.
            for (int p = 0; p < n_vis; ++p) {
                const float v = __ldg(sc + p);
                int rank = 0;
                for (int q = 0; q < n_vis; ++q) {
                    const float u = __ldg(sc + q);
                    if (u > v) ++rank;
                    else if (u == v && q < p) ++rank;
                }
                if (rank < top_pools)
                    for (int m = 0; m < kpool; ++m) row[rank * kpool + m] = p * kpool + m;
            }
        }
    }
    if (select_tail) {
        // The cells after the last complete pool, in cell order.  At `pos` 0..kpool-1 this is the ONLY content
        // of the row, so a run with `select_tail` off attends to nothing until the first pool completes.
        for (int m = 0; m < kpool - 1; ++m) {
            const int cell = n_vis * kpool + m;
            if (cell <= pos[t]) row[top_pools * kpool + m] = cell;
        }
    }
}

__global__ void glm_dsa_attn_kernel(const float* __restrict__ q_abs, const __half* __restrict__ latents,
                                    const int* __restrict__ cells, int kv_lora, int n_head, int qk_nope,
                                    int n_sel, float* __restrict__ out) {
    const int h = blockIdx.x;
    const int t = blockIdx.y;
    const int lane = threadIdx.x;
    const int nt = blockDim.x;
    const int* row = cells + (size_t) n_sel * t;
    const float scale = rsqrtf((float) qk_nope);
    const float* qa = q_abs + (size_t) kv_lora * h + (size_t) kv_lora * n_head * t;
    float* o = out + (size_t) kv_lora * h + (size_t) kv_lora * n_head * t;

    extern __shared__ float smem[];   // the row's scores, then the context
    float* s_prob = smem;
    float* s_ctx = smem + n_sel;

    for (int s = lane; s < n_sel; s += nt) {
        const int cell = row[s];
        if (cell < 0) {
            s_prob[s] = -INFINITY;
            continue;
        }
        const __half* lat = latents + (size_t) kv_lora * cell;
        float dot = 0.0f;
        for (int e = 0; e < kv_lora; ++e) dot += qa[e] * __half2float(lat[e]);
        s_prob[s] = dot * scale;
    }
    __syncthreads();

    // max, then exp, then scale - the reference's `ggml_soft_max` order.  A cell that is not there stays at
    // exp(0) = 0 in the denominator and contributes nothing to the numerator, which is the -inf mask exactly.
    if (lane == 0) {
        float mx = -INFINITY;
        for (int s = 0; s < n_sel; ++s) mx = fmaxf(mx, s_prob[s]);
        float denom = 0.0f;
        for (int s = 0; s < n_sel; ++s) {
            const float v = (s_prob[s] == -INFINITY) ? 0.0f : expf(s_prob[s] - mx);
            s_prob[s] = v;
            denom += v;
        }
        const float inv = denom > 0.0f ? 1.0f / denom : 0.0f;
        for (int s = 0; s < n_sel; ++s) s_prob[s] *= inv;
    }
    __syncthreads();

    for (int e = lane; e < kv_lora; e += nt) {
        float acc = 0.0f;
        for (int s = 0; s < n_sel; ++s) {
            const int cell = row[s];
            if (cell >= 0) acc += s_prob[s] * __half2float(latents[(size_t) e + (size_t) kv_lora * cell]);
        }
        s_ctx[e] = acc;
    }
    __syncthreads();

    // Out in the q_abs layout, and WITHOUT the de-absorption: `attn_v_b` is quantized in the pack, so it stays
    // where the dense path already takes it - `project_rows` over per-head bands of `kqv`.
    for (int e = lane; e < kv_lora; e += nt) o[e] = s_ctx[e];
}

}  // namespace

void glm_dsa_pool(const float* ik, const float* ig, const float* ape, int key_dim, int kpool, int n_pools,
                  float* pooled, void* stream) {
    if (key_dim <= 0 || kpool <= 0 || n_pools <= 0) return;
    const int total = key_dim * n_pools;
    glm_dsa_pool_kernel<<<(total + DS_THREADS - 1) / DS_THREADS, DS_THREADS, 0, (cudaStream_t) stream>>>(
        ik, ig, ape, key_dim, kpool, n_pools, pooled);
    check_launch("glm_dsa_pool");
    sync_if_needed(stream, "glm_dsa_pool");
}

void glm_dsa_score(const float* iq, const float* pooled, const float* weights, int key_dim, int idx_heads,
                   int n_tokens, int n_pools, float* score, void* stream) {
    if (key_dim <= 0 || idx_heads <= 0 || n_tokens <= 0 || n_pools <= 0) return;
    const int total = n_tokens * n_pools;
    glm_dsa_score_kernel<<<(total + DS_THREADS - 1) / DS_THREADS, DS_THREADS, 0, (cudaStream_t) stream>>>(
        iq, pooled, weights, key_dim, idx_heads, n_tokens, n_pools, score);
    check_launch("glm_dsa_score");
    sync_if_needed(stream, "glm_dsa_score");
}

void glm_dsa_select(const float* score, int n_pools, int kpool, int top_pools, int select_tail, int n_tokens,
                    int n_sel, const int* pos, int* cells, void* stream) {
    if (n_pools < 0 || kpool <= 0 || top_pools < 0 || n_tokens <= 0 || n_sel <= 0) return;
    glm_dsa_select_kernel<<<(n_tokens + DS_THREADS - 1) / DS_THREADS, DS_THREADS, 0, (cudaStream_t) stream>>>(
        score, n_pools, kpool, top_pools, select_tail, n_tokens, n_sel, pos, cells);
    check_launch("glm_dsa_select");
    sync_if_needed(stream, "glm_dsa_select");
}

void glm_dsa_attn(const float* q_abs, const uint16_t* latents, const int* cells, int kv_lora, int n_head,
                  int qk_nope, int n_tokens, int n_sel, float* out, void* stream) {
    if (kv_lora <= 0 || n_head <= 0 || qk_nope <= 0 || n_tokens <= 0 || n_sel <= 0) return;
    if (kv_lora > DS_THREADS * DS_SLOTS) {
        std::fprintf(stderr, "glm_dsa_attn: kv_lora %d exceeds the %d slots a thread carries\n", kv_lora,
                     DS_THREADS * DS_SLOTS);
        return;
    }
    const size_t smem = ((size_t) n_sel + (size_t) kv_lora) * sizeof(float);
    dim3 grid((unsigned) n_head, (unsigned) n_tokens);
    glm_dsa_attn_kernel<<<grid, DS_THREADS, smem, (cudaStream_t) stream>>>(
        q_abs, (const __half*) latents, cells, kv_lora, n_head, qk_nope, n_sel, out);
    check_launch("glm_dsa_attn");
    sync_if_needed(stream, "glm_dsa_attn");
}

}  // namespace strata::kernels
