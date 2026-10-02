// src/kernels/cuda/sampler.cu - P2.S2: the sampler chain, in llama.cpp's order.
//
//     penalties -> top_k -> top_p -> min_p -> temperature -> pick
//
// THE ORDER IS THE WHOLE CONTENT OF THIS FILE.  llama.cpp builds its chain by walking `params.samplers`, whose
// default is { PENALTIES, DRY, TOP_N_SIGMA, TOP_K, TYPICAL_P, TOP_P, MIN_P, XTC, TEMPERATURE } (`common/common.h`
// at 3cf03257) - ONE penalties stage, first, and TEMPERATURE AFTER THE TRUNCATION FILTERS.  (Issue #53: this file
// used to apply the penalties a second time after the temperature, and min_p before top_p - both taken from the
// order of the `case` labels in `common/sampling.cpp`, which is not the order the chain runs.)  Every order
// produces a valid token, so only a comparison at the distribution level can tell them apart; the parity test
// does that against an independently computed distribution.
//
// `sampler_greedy_kernel` is the plain argmax, one block per token over the vocabulary.  The sampled chain has
// three implementations that pick the same token, bit for bit:
//   - the SPLIT top_k (default): `sampler_split_part_kernel` cuts each row into 4,096-logit blocks over the whole
//     GPU, each keeps its own top_k, and `sampler_split_merge_kernel` merges those lists and runs the tail;
//   - `sampler_one_block_kernel` (`STRATA_SAMPLER_ONE_BLOCK=1`, and the fallback when the split cannot run): one
//     block per token, `top_k` block-argmax rounds, each over the logits after the previous pick;
//   - `sampler_kernel` (`STRATA_OLD_SAMPLER=1`), the kernel of engine 0.1.20, kept as the reference.
// The two new ones share `sampled_tail_warp` (top_p / min_p / temperature / draw on one warp).
#include "strata/kernels/sampler.hpp"
#include "strata/core/coupled_draft.hpp"

#include <cuda_runtime.h>

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <vector>

namespace strata::kernels {
namespace {

// Philox 4x32-10, the counter-based generator the phase asks for.  Counter-based matters because it makes the
// stream a function of (seed, position) rather than of how many draws came before - so a batch can be sampled
// in any order and a run is reproducible.
__device__ __forceinline__ uint32_t philox4x32_round(uint32_t& c0, uint32_t& c1, uint32_t& c2, uint32_t& c3,
                                                     uint32_t k0, uint32_t k1) {
    const uint32_t hi0 = __umulhi(0x9E3779B9u, c0);
    const uint32_t hi1 = __umulhi(0xBB67AE85u, c2);
    const uint32_t lo0 = 0x9E3779B9u * c0;
    const uint32_t lo1 = 0xBB67AE85u * c2;
    const uint32_t n0 = hi1 ^ c1 ^ k0;
    const uint32_t n1 = lo1;
    const uint32_t n2 = hi0 ^ c3 ^ k1;
    const uint32_t n3 = lo0;
    c0 = n0; c1 = n1; c2 = n2; c3 = n3;
    return 0;
}

__device__ __forceinline__ float philox_uniform(uint64_t seed, uint64_t counter) {
    uint32_t c0 = (uint32_t) counter, c1 = (uint32_t) (counter >> 32);
    uint32_t c2 = (uint32_t) seed, c3 = (uint32_t) (seed >> 32);
    for (int i = 0; i < 10; ++i) {
        philox4x32_round(c0, c1, c2, c3, (uint32_t) i, 0u);
    }
    // 24 bits of mantissa, so the value is uniform in [0,1) with no rounding to 1.0
    return (float) (c0 >> 8) * (1.0f / 16777216.0f);
}

// `count_in_history` and the penalty application, transcribed from `llama_sampler_penalties_apply`.
// The repeat penalty MULTIPLIES for non-positive logits and DIVIDES for positive ones - dividing
// unconditionally is the natural reading of the source paper and it INVERTS the penalty on half the
// vocabulary.  The presence penalty is `float(count > 0)`, a boolean, not the count.
__device__ __forceinline__ int history_count(const int* __restrict__ h, int n, int v) {
    int c = 0;
    for (int i = 0; i < n; ++i) if (h[i] == v) ++c;
    return c;
}

__device__ __forceinline__ float apply_penalties(float logit, int count, const SamplerParams& p) {
    if (count <= 0) return logit;
    if (logit <= 0.0f) logit *= p.penalty_repeat;
    else               logit /= p.penalty_repeat;
    logit -= (float) count * p.penalty_freq + (count > 0 ? 1.0f : 0.0f) * p.penalty_present;
    return logit;
}

/// **THE GREEDY ARGMAX, ONE BLOCK PER TOKEN, COVERING THE VOCABULARY.**
///
/// **WHY THIS IS A SEPARATE KERNEL AND NOT A BRANCH.**  `sampler_kernel` is launched as a grid over TOKENS
/// with 64 threads and a `if (t >= n_tokens) return;` at the top.  The decode path has `n_tokens == 1`, so
/// that launch was `<<<1, 64>>>`, 63 threads exited on the first line, and ONE THREAD walked all 248,320
/// logits in a dependent loop on one SM of 48.  Measured in isolation (`bench/micro/sampler_cost.cu`):
/// **3.11 ms per token**, 5.7% of an ~54 ms token, and the whole of round 309's `sample` phase - the two
/// synchronisations around it are 0.03 ms each.
///
/// The obvious repair is to parallelise the scan inside `sampler_kernel`, and it is WRONG: with one thread
/// per token, a block reduction over the vocabulary has nothing to reduce, and the threads that returned
/// early are not there for `__syncthreads` or `__shfl_down_sync`.  The first attempt did exactly that and
/// produced the token `5120` thirty-two times.  The grid has to be over tokens with the BLOCK over the
/// vocabulary, which is a different launch configuration and therefore a different kernel.
///
/// **THE TIE RULE IS UNCHANGED AND THAT IS THE WHOLE CORRECTNESS ARGUMENT.**  The serial scan walked `v`
/// ascending with `if (s > bv)`, so the LOWEST index wins a tie.  Each thread keeps that rule over its own
/// strided subset and the reduction resolves two candidates by taking the larger value and, on equality, the
/// SMALLER index - the same total order, so `sampler_parity` and C1 see no change.
__global__ void sampler_greedy_kernel(const float* __restrict__ logits, int n_vocab,
                                      const int* __restrict__ history, int history_len, const SamplerParams p,
                                      int pmin, int plen, int* __restrict__ out) {
    const int t = blockIdx.x;
    const float* l = logits + (size_t) t * n_vocab;
    (void) pmin;
    const int* hrow = history ? history + (size_t) t * history_len : nullptr;
    int hlen = 0;
    if (hrow) {
        hlen = plen < history_len ? plen : history_len;
        if (hlen < 0) hlen = 0;
        hrow += history_len - hlen;          // the window is the TAIL
    }

    // PENALTY MEMBERSHIP AS A BITMAP.  The history touches at most `hlen` tokens of a quarter-million
    // vocabulary, but the naive `history_count` per candidate per argmax round costs O(k x n_vocab x hlen)
    // integer compares (~318 M per token at k=20, hlen=64 - measured 45 -> 31 tok/s on a real workload).
    // A shared bitmap gives an O(1) membership test, and only the (at most hlen) hits pay the count scan;
    // the counts - and therefore every sampled value - are exactly what the per-candidate scan produced.
    extern __shared__ unsigned int penal_bits[];
    const int bits_words = (int) ((n_vocab + 31) / 32);
    // The gate needs a NON-EMPTY WINDOW (`hlen > 0`): the launch sizes the shared bitmap only when penalties
    // are on, so a caller handing over a history buffer with `penalty_last_n == 0` must not touch it.
    const bool use_bits = hrow != nullptr && hlen > 0 && bits_words > 0;
    if (use_bits) {
        for (int w = threadIdx.x; w < bits_words; w += blockDim.x) penal_bits[w] = 0u;
        __syncthreads();
        for (int i = threadIdx.x; i < hlen; i += blockDim.x)
            if (hrow[i] >= 0 && hrow[i] < n_vocab)   // an id outside the vocabulary is never a candidate
                atomicOr(&penal_bits[hrow[i] >> 5], 1u << (hrow[i] & 31));
        __syncthreads();
    }
    auto hit_count = [&](int v) -> int {
        if (!use_bits || !(penal_bits[v >> 5] & (1u << (v & 31)))) return 0;
        return history_count(hrow, hlen, v);
    };

    // `n_vocab` is the "no candidate" index: it loses every comparison to a real one, so a thread with no
    // elements contributes nothing rather than contributing a bogus zero.
    float bv = __int_as_float(0xff800000);   // -inf
    int best = n_vocab;
    for (int v = threadIdx.x; v < n_vocab; v += blockDim.x) {
        const float s = apply_penalties(l[v], hit_count(v), p);
        if (s > bv) { bv = s; best = v; }
    }
    for (int off = 16; off > 0; off >>= 1) {
        const float ov = __shfl_down_sync(0xFFFFFFFFu, bv, off);
        const int oi = __shfl_down_sync(0xFFFFFFFFu, best, off);
        if (ov > bv || (ov == bv && oi < best)) { bv = ov; best = oi; }
    }
    __shared__ float sv[32];
    __shared__ int si[32];
    const int warp = (int) (threadIdx.x >> 5), lane = (int) (threadIdx.x & 31);
    if (lane == 0) { sv[warp] = bv; si[warp] = best; }
    __syncthreads();
    if (warp == 0) {
        const int nw = (int) ((blockDim.x + 31) >> 5);
        float wv = lane < nw ? sv[lane] : __int_as_float(0xff800000);
        int wi = lane < nw ? si[lane] : n_vocab;
        for (int off = 16; off > 0; off >>= 1) {
            const float ov = __shfl_down_sync(0xFFFFFFFFu, wv, off);
            const int oi = __shfl_down_sync(0xFFFFFFFFu, wi, off);
            if (ov > wv || (ov == wv && oi < wi)) { wv = ov; wi = oi; }
        }
        // A tie between two `-inf` candidates leaves `wi == n_vocab`, and the serial version answered 0.
        if (lane == 0) out[t] = (wi < n_vocab) ? wi : 0;
    }
}

/// **THE SAMPLED PATH, ONE BLOCK PER TOKEN.**  The kernel below replaced a version that ran the whole chain
/// in ONE THREAD per token (`<<<ceil(T/64), 64>>>`, so a 4-token window fielded four threads): `top_k` alone
/// was `k` sequential scans of the vocabulary with an inner sweep over the already-taken list - 20 x 248,320
/// iterations of dependent work on one SM - and a verify window measured **1.6 s in the sampler**, which made
/// every temperature-bearing request ~30x slower than a greedy one.  The selection is `k` argmax rounds, and
/// an argmax over the vocabulary parallelises exactly like `sampler_greedy_kernel` (block over the vocab), so
/// the rounds run back to back inside a block-per-token launch: the per-token cost falls to
/// `k x n_vocab / 1024` plus `k` block reductions.
///
/// THE SEMANTICS ARE THE SERIAL ONES, EXACTLY.  Each round's argmax resolves ties to the LOWEST index (the
/// serial scan's strict `>` keeps the first maximum it meets), so the kept sequence - both its set and its
/// order - is unchanged; `top_p`'s cut reads that order in double arithmetic as before; temperature and the
/// Philox draw apply after the cut.  `sampler_parity` pins all of it against the host reference.
///
/// **KEPT AS THE REFERENCE, BEHIND `STRATA_OLD_SAMPLER=1`.**  Two costs remain in it: the `taken`
/// sweep is O(k) per logit per round, O(k^2 x n_vocab) per row (47 M shared-memory compares at k = 20, 500 M at
/// 64), and the double-precision tail runs on all 1,024 threads where one warp suffices - GeForce issues FP64 at
/// 1/64 of FP32.  The kernels after this one remove both and select the same list in the same order.
__global__ void sampler_kernel(const float* __restrict__ logits, int n_vocab, int n_tokens,
                               const int* __restrict__ history, int history_len, const SamplerParams p,
                               int* __restrict__ out) {
    const int t = blockIdx.x;
    if (t >= n_tokens) return;
    const float* l = logits + (size_t) t * n_vocab;

    // Temperature is needed by BOTH stages below, so it is computed here; the chain still APPLIES it after
    // the truncation filters - the survivors are chosen on the raw logits and only then scaled.
    const float inv_t = p.temperature > 0.0f ? 1.0f / p.temperature : 0.0f;

    // The penalty window is the last `penalty_last_n` entries of this row's history (disabled at this
    // launch: `sample_tokens` refuses a non-zero `penalty_last_n` without a history buffer).
    const int* hrow = history ? history + (size_t) t * history_len : nullptr;
    int hlen = 0;
    if (hrow) {
        hlen = p.penalty_last_n < history_len ? p.penalty_last_n : history_len;
        if (hlen < 0) hlen = 0;
        hrow += history_len - hlen;          // the window is the TAIL
    }

    // the membership bitmap, as in `sampler_greedy_kernel` - see the cost note there.  The gate needs an
    // NON-EMPTY WINDOW too: the launch sizes the bitmap only when penalties are on, so a caller that hands over
    // a stale history buffer with `penalty_last_n == 0` must not touch it.
    extern __shared__ unsigned int penal_bits[];
    const int bits_words = (int) ((n_vocab + 31) / 32);
    const bool use_bits = hrow != nullptr && hlen > 0 && bits_words > 0;
    if (use_bits) {
        for (int w = threadIdx.x; w < bits_words; w += blockDim.x) penal_bits[w] = 0u;
        __syncthreads();
        for (int i = threadIdx.x; i < hlen; i += blockDim.x)
            if (hrow[i] >= 0 && hrow[i] < n_vocab)   // an id outside the vocabulary is never a candidate
                atomicOr(&penal_bits[hrow[i] >> 5], 1u << (hrow[i] & 31));
        __syncthreads();
    }
    auto hit_count = [&](int v) -> int {
        if (!use_bits || !(penal_bits[v >> 5] & (1u << (v & 31)))) return 0;
        return history_count(hrow, hlen, v);
    };

    // top_k in 1..64 is taken as given; 0 ("off") and anything wider mean the widest shortlist the kernel
    // keeps, 64.  Every row writes out[t]: a verify window reads all of them.
    const int KMAX = 64;
    int k = (p.top_k > 0 && p.top_k < KMAX) ? p.top_k : KMAX;
    if (k > n_vocab) k = n_vocab;

    // ---- top_k: k rounds of a block argmax over the not-yet-taken.  `sel_*` holds the kept ids and their
    // raw logits in selection order: descending by value, ties to the lower index, which is the order the
    // top_p cut below is defined over.
    __shared__ int sel_ids[KMAX];
    __shared__ float sel_logit[KMAX];
    __shared__ float sv[32];
    __shared__ int si[32];
    for (int i = 0; i < k; ++i) {
        // `n_vocab` is the "no candidate" index: it loses every comparison to a real one (same convention as
        // the greedy kernel, whose tie rule this reduction shares).
        float bv = __int_as_float(0xff800000);   // -inf
        int best = n_vocab;
        for (int v = threadIdx.x; v < n_vocab; v += blockDim.x) {
            bool taken = false;
            for (int j = 0; j < i; ++j) if (sel_ids[j] == v) { taken = true; break; }
            if (taken) continue;
            const float s = apply_penalties(l[v], hit_count(v), p);
            if (s > bv) { bv = s; best = v; }
        }
        for (int off = 16; off > 0; off >>= 1) {
            const float ov = __shfl_down_sync(0xFFFFFFFFu, bv, off);
            const int oi = __shfl_down_sync(0xFFFFFFFFu, best, off);
            if (ov > bv || (ov == bv && oi < best)) { bv = ov; best = oi; }
        }
        const int warp = (int) (threadIdx.x >> 5), lane = (int) (threadIdx.x & 31);
        if (lane == 0) { sv[warp] = bv; si[warp] = best; }
        __syncthreads();
        if (warp == 0) {
            const int nw = (int) ((blockDim.x + 31) >> 5);
            float wv = lane < nw ? sv[lane] : __int_as_float(0xff800000);
            int wi = lane < nw ? si[lane] : n_vocab;
            for (int off = 16; off > 0; off >>= 1) {
                const float ov = __shfl_down_sync(0xFFFFFFFFu, wv, off);
                const int oi = __shfl_down_sync(0xFFFFFFFFu, wi, off);
                if (ov > wv || (ov == wv && oi < wi)) { wv = ov; wi = oi; }
            }
            if (lane == 0) { sel_ids[i] = (wi < n_vocab) ? wi : 0; sel_logit[i] = wv; }
        }
        __syncthreads();
    }

    // ---- top_p over the top_k list (penalised logits, descending as the selection produced them), then min_p,
    // then temperature and one Philox draw - llama.cpp's order (issue #53).  Every thread computes the same chain
    // redundantly over `sel_*` - the arithmetic is the serial kernel's, instruction for instruction - so they
    // agree on `pick` and thread 0 writes it.
    int n_keep = k;
    float mx = sel_logit[0];
    for (int i = 1; i < k; ++i) mx = fmaxf(mx, sel_logit[i]);
    if (p.top_p < 1.0f) {
        double sum = 0.0;
        for (int i = 0; i < k; ++i) sum += exp((double) sel_logit[i] - (double) mx);
        double cum = 0.0;
        int cut = k;
        for (int i = 0; i < k; ++i) {
            cum += exp((double) sel_logit[i] - (double) mx) / sum;
            if (cum >= (double) p.top_p) { cut = i + 1; break; }
        }
        if (cut < p.min_keep) cut = p.min_keep < k ? p.min_keep : k;
        n_keep = cut;
    }
    // ---- min_p on top_p's survivors: the descending prefix whose probability is at least `min_p` of the top
    // token's.  In logit space the threshold is `sel_logit[0] + logf(min_p)` - equivalent to `p >= min_p * p_max`
    // without the overflow an exp of raw logits risks.  0 disables, and the head itself always survives
    // (`expf(0) == 1 >= min_p` for min_p in 0..1), so the count never reaches zero.
    if (p.min_p > 0.0f) {
        const float thresh = sel_logit[0] + logf(p.min_p);
        for (int i = 0; i < n_keep; ++i)
            if (sel_logit[i] < thresh) { n_keep = i; break; }
    }
    // temperature only: the penalties were applied once, before the selection (issue #53: they were applied a
    // second time here, after the temperature scaling - llama.cpp's chain has one penalties stage)
    auto scaled = [&](int i) { return sel_logit[i] * inv_t; };
    float smx = scaled(0);
    for (int i = 1; i < n_keep; ++i) smx = fmaxf(smx, scaled(i));
    double sum = 0.0;
    for (int i = 0; i < n_keep; ++i) sum += exp((double) scaled(i) - (double) smx);
    const float u = philox_uniform(p.seed, p.counter + (uint64_t) t);
    double cum = 0.0;
    int pick = sel_ids[n_keep - 1];
    for (int i = 0; i < n_keep; ++i) {
        cum += exp((double) scaled(i) - (double) smx) / sum;
        if ((double) u < cum) { pick = sel_ids[i]; break; }
    }
    if (threadIdx.x == 0) out[t] = pick;
}

// ---- the same list, in the same order, without the `taken` sweep ----
//
// THE SELECTION ORDER.  `sampler_kernel`'s rounds rank a candidate (value, id) before (value', id') when value >
// value', or value == value' and id < id' - each thread's ascending scan keeps the first maximum it meets (strict
// `>`), and the reductions resolve a tie to the lower id.  That is a strict total order (-0 and +0 compare equal and
// fall to the id, as there), so round i picks the i-th logit of the order.  NaN and -inf are never picked (`s > bv`
// from -inf fails); a round that finds nothing yields the SENTINEL (-inf, n_vocab), stored as id 0.

constexpr int kSelMax = 64;               // the widest top_k list, `sampler_kernel`'s KMAX
constexpr unsigned kFullMask = 0xFFFFFFFFu;

// top_k 1..64 as given; 0 ("off") and anything wider keep 64; never more than the vocabulary
__host__ __device__ __forceinline__ int sampled_k(int top_k, int n_vocab) {
    int k = (top_k > 0 && top_k < kSelMax) ? top_k : kSelMax;
    return k > n_vocab ? n_vocab : k;
}

// (bv, bi) <- the first of (bv, bi) and (ov, oi) in the selection order
__device__ __forceinline__ void take_first(float& bv, int& bi, float ov, int oi) {
    if (ov > bv || (ov == bv && oi < bi)) { bv = ov; bi = oi; }
}

// The first of the warp's 32 candidates, left in EVERY lane.  An XOR butterfly is exact here: "the first of two" is
// associative and commutative in a strict total order (two lanes never hold different candidates that compare
// equal - an id is in one lane at most, and two sentinels are the same pair), so every lane ends with the same pair.
__device__ __forceinline__ void warp_first(float& bv, int& bi) {
#pragma unroll
    for (int off = 16; off > 0; off >>= 1) {
        const float ov = __shfl_xor_sync(kFullMask, bv, off);
        const int oi = __shfl_xor_sync(kFullMask, bi, off);
        take_first(bv, bi, ov, oi);
    }
}

/// **THE TAIL ON ONE WARP, WITH `sampler_kernel`'S ARITHMETIC.**  `sel_ids` / `sel_logit` (shared, `k` entries) are
/// the top_k list in selection order; the 32 lanes of one warp call this and lane 0 writes `out[t]`.  The old tail
/// ran on all 1,024 threads, each computing the same ~4k double `exp`s - FP64 issues at 1/64 of FP32 on GeForce.
/// Here the lanes share the `exp`s (one per entry, into `ex`) and then the quotients, and lane 0 alone runs the two
/// ORDERED sums and the two cumulative scans.  Every double is the one the old chain computed: the same `exp` of
/// the same argument, the sums in the same order, `cum += e / sum` with the same correctly rounded quotient - so
/// the cut, the survivors and the pick are the same.  (`n_keep == 0`, reachable only with min_p > 1, which the
/// callers clamp, read `sel_ids[-1]` in the old tail; it reads `sel_ids[0]` here.)
/// kProb (the coupled draft only): lane 0 also writes the pick's probability under the final distribution to
/// `*prob_out`.  The sampler's own calls take kProb = false, whose code is the tail above, unchanged.
template <bool kProb = false>
__device__ void sampled_tail_warp(const int* sel_ids, const float* sel_logit, int k, const SamplerParams& p, int t,
                                  int* __restrict__ out, double* ex, float* prob_out = nullptr) {
    const int lane = (int) (threadIdx.x & 31);
    const float inv_t = p.temperature > 0.0f ? 1.0f / p.temperature : 0.0f;
    int n_keep = k;
    float mx = sel_logit[0];
    for (int i = 1; i < k; ++i) mx = fmaxf(mx, sel_logit[i]);
    if (p.top_p < 1.0f) {
        for (int i = lane; i < k; i += 32) ex[i] = exp((double) sel_logit[i] - (double) mx);
        __syncwarp();
        double sum = 0.0;
        if (lane == 0)
            for (int i = 0; i < k; ++i) sum += ex[i];
        sum = __shfl_sync(kFullMask, sum, 0);
        __syncwarp();                            // lane 0 has read every `ex` before it is overwritten
        for (int i = lane; i < k; i += 32) ex[i] = ex[i] / sum;
        __syncwarp();
        int cut = k;
        if (lane == 0) {
            double cum = 0.0;
            for (int i = 0; i < k; ++i) {
                cum += ex[i];
                if (cum >= (double) p.top_p) { cut = i + 1; break; }
            }
        }
        cut = __shfl_sync(kFullMask, cut, 0);
        if (cut < p.min_keep) cut = p.min_keep < k ? p.min_keep : k;
        n_keep = cut;
        __syncwarp();                            // `ex` is written again below
    }
    // min_p on top_p's survivors, as in `sampler_kernel` (every lane, the same float arithmetic)
    if (p.min_p > 0.0f) {
        const float thresh = sel_logit[0] + logf(p.min_p);
        for (int i = 0; i < n_keep; ++i)
            if (sel_logit[i] < thresh) { n_keep = i; break; }
    }
    // temperature, then one Philox draw
    float smx = sel_logit[0] * inv_t;
    for (int i = 1; i < n_keep; ++i) smx = fmaxf(smx, sel_logit[i] * inv_t);
    for (int i = lane; i < n_keep; i += 32) ex[i] = exp((double) (sel_logit[i] * inv_t) - (double) smx);
    __syncwarp();
    double sum = 0.0;
    if (lane == 0)
        for (int i = 0; i < n_keep; ++i) sum += ex[i];
    sum = __shfl_sync(kFullMask, sum, 0);
    __syncwarp();
    for (int i = lane; i < n_keep; i += 32) ex[i] = ex[i] / sum;
    __syncwarp();
    if (lane == 0) {
        const float u = philox_uniform(p.seed, p.counter + (uint64_t) t);
        double cum = 0.0;
        int pi = n_keep > 0 ? n_keep - 1 : 0;
        int pick = sel_ids[pi];
        for (int i = 0; i < n_keep; ++i) {
            cum += ex[i];
            if ((double) u < cum) { pick = sel_ids[i]; pi = i; break; }
        }
        out[t] = pick;
        if constexpr (kProb) *prob_out = n_keep > 0 ? (float) ex[pi] : 1.0f;
    }
}

/// **THE ONE-BLOCK SAMPLED PATH: `sampler_kernel` WITHOUT THE `taken` SWEEP.**  Round i's candidates are the logits
/// strictly AFTER round i-1's pick in the selection order, `s < prev_v || (s == prev_v && v > prev_i)`, which is
/// exactly the set `taken` left: the picks so far are the first i of the order, so what comes after the last one is
/// what has not been picked.  The round is then the same block argmax with the same tie rule, so the list is the
/// same list in the same order.  An empty round leaves (-inf, id 0) as before, and every round after it is empty
/// in both versions (nothing after -inf beats -inf).  O(k x n_vocab) per row instead of O(k^2 x n_vocab), then warp
/// 0 runs the tail.  `STRATA_SAMPLER_ONE_BLOCK=1`, and the fallback when the split path cannot run.
__global__ void __launch_bounds__(1024)
sampler_one_block_kernel(const float* __restrict__ logits, int n_vocab, const int* __restrict__ history,
                         int history_len, const SamplerParams p, int* __restrict__ out) {
    const int t = blockIdx.x;
    const float* l = logits + (size_t) t * n_vocab;

    // the penalty window and its membership bitmap, exactly as in `sampler_kernel`
    const int* hrow = history ? history + (size_t) t * history_len : nullptr;
    int hlen = 0;
    if (hrow) {
        hlen = p.penalty_last_n < history_len ? p.penalty_last_n : history_len;
        if (hlen < 0) hlen = 0;
        hrow += history_len - hlen;          // the window is the TAIL
    }
    extern __shared__ unsigned int penal_bits[];
    const int bits_words = (int) ((n_vocab + 31) / 32);
    const bool use_bits = hrow != nullptr && hlen > 0 && bits_words > 0;
    if (use_bits) {
        for (int w = threadIdx.x; w < bits_words; w += blockDim.x) penal_bits[w] = 0u;
        __syncthreads();
        for (int i = threadIdx.x; i < hlen; i += blockDim.x)
            if (hrow[i] >= 0 && hrow[i] < n_vocab)   // an id outside the vocabulary is never a candidate
                atomicOr(&penal_bits[hrow[i] >> 5], 1u << (hrow[i] & 31));
        __syncthreads();
    }
    auto hit_count = [&](int v) -> int {
        if (!use_bits || !(penal_bits[v >> 5] & (1u << (v & 31)))) return 0;
        return history_count(hrow, hlen, v);
    };

    const int k = sampled_k(p.top_k, n_vocab);
    __shared__ int sel_ids[kSelMax];
    __shared__ float sel_logit[kSelMax];
    __shared__ double ex[kSelMax];
    __shared__ float sv[32];
    __shared__ int si[32];
    const int warp = (int) (threadIdx.x >> 5), lane = (int) (threadIdx.x & 31);
    float prev_v = __int_as_float(0x7f800000);   // +inf and id -1: round 0 takes every logit
    int prev_i = -1;
    for (int i = 0; i < k; ++i) {
        float bv = __int_as_float(0xff800000);   // -inf
        int best = n_vocab;
        for (int v = threadIdx.x; v < n_vocab; v += blockDim.x) {
            const float s = apply_penalties(l[v], hit_count(v), p);
            if ((s < prev_v || (s == prev_v && v > prev_i)) && s > bv) { bv = s; best = v; }
        }
        for (int off = 16; off > 0; off >>= 1) {
            const float ov = __shfl_down_sync(0xFFFFFFFFu, bv, off);
            const int oi = __shfl_down_sync(0xFFFFFFFFu, best, off);
            if (ov > bv || (ov == bv && oi < best)) { bv = ov; best = oi; }
        }
        if (lane == 0) { sv[warp] = bv; si[warp] = best; }
        __syncthreads();
        if (warp == 0) {
            const int nw = (int) ((blockDim.x + 31) >> 5);
            float wv = lane < nw ? sv[lane] : __int_as_float(0xff800000);
            int wi = lane < nw ? si[lane] : n_vocab;
            for (int off = 16; off > 0; off >>= 1) {
                const float ov = __shfl_down_sync(0xFFFFFFFFu, wv, off);
                const int oi = __shfl_down_sync(0xFFFFFFFFu, wi, off);
                if (ov > wv || (ov == wv && oi < wi)) { wv = ov; wi = oi; }
            }
            if (lane == 0) { sel_ids[i] = (wi < n_vocab) ? wi : 0; sel_logit[i] = wv; }
        }
        __syncthreads();
        // An empty round leaves (-inf, 0): nothing comes after it, as nothing was left untaken.
        prev_v = sel_logit[i];
        prev_i = sel_ids[i];
    }
    if (warp != 0) return;
    sampled_tail_warp(sel_ids, sel_logit, k, p, t, out, ex);
}

// ---- THE SPLIT top_k.  `sampler_kernel` and the one-block kernel keep a row on ONE SM of 170 (5090) and walk its
// 248,320 logits k times.  Here a warp keeps the top_k of 1,024 logits held in registers, four warps make a block of
// 4,096 logits that merges their lists, and a row is 61 such blocks over the whole GPU; one warp per row then
// merges the 61 lists and runs the tail.  Exact, because the first k of a union are within the first k of each part
// and a merge of ordered lists is ordered - the selection order being a strict total order (above).
constexpr int kSplitPerLane = 32;                               // logits per lane, in registers
constexpr int kSplitWarpSpan = 32 * kSplitPerLane;              // 1,024 logits per warp
constexpr int kSplitWarps = 4;
constexpr int kSplitBlockSpan = kSplitWarps * kSplitWarpSpan;   // 4,096 logits per block
constexpr int kSplitMaxBlocks = 64;                             // lists the merge holds: n_vocab <= 262,144
constexpr int kSplitMaxRows = 64;                               // rows per split launch (the scratch's bound)

/// Merge `nl` (<= 64) lists of `k` candidates - list L at `lists[L * stride]`, each in the selection order and
/// padded with sentinels - into their first `k`: `sink(i, value, id)` runs in every lane for i = 0..k-1 with the
/// same pair.  Lane owns lists `lane` and `lane + 32`; a round takes the first of all heads and advances the list
/// it came from.  An id is in one list at most (the lists cover disjoint logits), so exactly one head matches.
template <typename Sink>
__device__ __forceinline__ void warp_merge_lists(const int2* lists, int nl, int stride, int k, int n_vocab,
                                                 Sink&& sink) {
    const int lane = (int) (threadIdx.x & 31);
    float hv[2];
    int hi[2], pos[2];
#pragma unroll
    for (int m = 0; m < 2; ++m) {
        const int L = lane + 32 * m;
        pos[m] = 0;
        hv[m] = __int_as_float(0xff800000);
        hi[m] = n_vocab;
        if (L < nl) {
            const int2 c = lists[(size_t) L * stride];
            hi[m] = c.x;
            hv[m] = __int_as_float(c.y);
        }
    }
    for (int i = 0; i < k; ++i) {
        float bv = hv[0];
        int bi = hi[0];
        take_first(bv, bi, hv[1], hi[1]);
        warp_first(bv, bi);
        sink(i, bv, bi);
        if (bi < n_vocab) {
#pragma unroll
            for (int m = 0; m < 2; ++m) {
                if (hi[m] != bi) continue;
                if (++pos[m] < k) {
                    const int2 c = lists[(size_t) (lane + 32 * m) * stride + pos[m]];
                    hi[m] = c.x;
                    hv[m] = __int_as_float(c.y);
                } else {
                    hi[m] = n_vocab;
                    hv[m] = __int_as_float(0xff800000);
                }
            }
        }
    }
}

/// **SPLIT STAGE 1: THE top_k OF EACH 4,096-LOGIT BLOCK.**  Grid (blocks per row, rows), 128 threads.  Each warp
/// loads its 1,024 penalised logits once into registers (lane + 32 j: every load is one coalesced 128-byte line)
/// and runs `k` warp-argmax rounds over them with the threshold of `sampler_one_block_kernel` - no shared memory
/// and no block barrier per round.  Warp 0 then merges the four warp lists into the block's list in `cand`
/// (row-major: row t, block b, entry i at `(t * n_blocks + b) * k + i`, as (id, value bits)).
__global__ void __launch_bounds__(kSplitWarps * 32)
sampler_split_part_kernel(const float* __restrict__ logits, int n_vocab, const int* __restrict__ history,
                          int history_len, const SamplerParams p, int k, int n_blocks, int2* __restrict__ cand) {
    const int t = blockIdx.y;
    const float* l = logits + (size_t) t * n_vocab;
    const int warp = (int) (threadIdx.x >> 5), lane = (int) (threadIdx.x & 31);
    const int blo = (int) blockIdx.x * kSplitBlockSpan;

    const int* hrow = history ? history + (size_t) t * history_len : nullptr;
    int hlen = 0;
    if (hrow) {
        hlen = p.penalty_last_n < history_len ? p.penalty_last_n : history_len;
        if (hlen < 0) hlen = 0;
        hrow += history_len - hlen;          // the window is the TAIL
    }
    // The membership bitmap of THIS BLOCK'S 4,096 logits (512 bytes, not the vocabulary's 31 KB): the same test,
    // and a hit pays the same exact count over the whole window.
    __shared__ unsigned int bits[kSplitBlockSpan / 32];
    const bool use_bits = hrow != nullptr && hlen > 0;
    if (use_bits) {
        for (int w = threadIdx.x; w < kSplitBlockSpan / 32; w += blockDim.x) bits[w] = 0u;
        __syncthreads();
        for (int i = threadIdx.x; i < hlen; i += blockDim.x) {
            const int h = hrow[i];
            if (h >= 0 && h < n_vocab && h >= blo && h - blo < kSplitBlockSpan)
                atomicOr(&bits[(h - blo) >> 5], 1u << ((h - blo) & 31));
        }
        __syncthreads();
    }

    // This warp's logits, penalised: `apply_penalties` with a zero count returns the logit unchanged, so only the
    // bitmap's hits go through it.  Past the vocabulary: -inf, which no round picks.
    const int lo = blo + warp * kSplitWarpSpan;
    float s[kSplitPerLane];
#pragma unroll
    for (int j = 0; j < kSplitPerLane; ++j) {
        const int v = lo + 32 * j + lane;
        s[j] = v < n_vocab ? l[v] : __int_as_float(0xff800000);
    }
    if (use_bits) {
#pragma unroll
        for (int j = 0; j < kSplitPerLane; ++j) {
            const int v = lo + 32 * j + lane, b = v - blo;
            if (v < n_vocab && (bits[b >> 5] & (1u << (b & 31))))
                s[j] = apply_penalties(s[j], history_count(hrow, hlen, v), p);
        }
    }

    __shared__ int2 wl[kSplitWarps][kSelMax];
    float prev_v = __int_as_float(0x7f800000);   // +inf and id -1: round 0 takes every logit
    int prev_i = -1;
    int i = 0;
    for (; i < k; ++i) {
        // two chains (even and odd j), each walked in ascending id with a strict `>`, so each keeps its first in
        // the order; `take_first` then orders the two
        float b0 = __int_as_float(0xff800000), b1 = __int_as_float(0xff800000);
        int i0 = n_vocab, i1 = n_vocab;
#pragma unroll
        for (int j = 0; j < kSplitPerLane; j += 2) {
            const int v0 = lo + 32 * j + lane, v1 = v0 + 32;
            const float x0 = s[j], x1 = s[j + 1];
            if ((x0 < prev_v || (x0 == prev_v && v0 > prev_i)) && x0 > b0) { b0 = x0; i0 = v0; }
            if ((x1 < prev_v || (x1 == prev_v && v1 > prev_i)) && x1 > b1) { b1 = x1; i1 = v1; }
        }
        take_first(b0, i0, b1, i1);
        warp_first(b0, i0);
        if (i0 >= n_vocab) break;                // the same in every lane: nothing left in these 1,024 logits
        if (lane == 0) wl[warp][i] = make_int2(i0, __float_as_int(b0));
        prev_v = b0;
        prev_i = i0;
    }
    for (int r = i + lane; r < k; r += 32) wl[warp][r] = make_int2(n_vocab, (int) 0xff800000u);   // sentinels
    __syncthreads();
    if (warp == 0) {
        int2* dst = cand + ((size_t) t * n_blocks + blockIdx.x) * k;
        warp_merge_lists(&wl[0][0], kSplitWarps, kSelMax, k, n_vocab, [&](int r, float v, int id) {
            if (lane == 0) dst[r] = make_int2(id, __float_as_int(v));
        });
    }
}

/// **SPLIT STAGE 2: ONE WARP PER ROW MERGES THE BLOCK LISTS, THEN RUNS THE TAIL.**  The row's lists (at most 64 x
/// 64 entries, 32 KB) are copied to shared memory, merged into the row's top_k list, and `sampled_tail_warp` picks.
__global__ void __launch_bounds__(32)
sampler_split_merge_kernel(const int2* __restrict__ cand, int n_blocks, int n_vocab, const SamplerParams p, int k,
                           int* __restrict__ out) {
    const int t = blockIdx.x;
    const int lane = (int) threadIdx.x;
    __shared__ int2 lists[kSplitMaxBlocks * kSelMax];
    __shared__ int sel_ids[kSelMax];
    __shared__ float sel_logit[kSelMax];
    __shared__ double ex[kSelMax];
    const int2* src = cand + (size_t) t * n_blocks * k;
    for (int e = lane; e < n_blocks * k; e += 32) lists[e] = src[e];
    __syncwarp();
    warp_merge_lists(lists, n_blocks, k, k, n_vocab, [&](int i, float v, int id) {
        if (lane == 0) { sel_ids[i] = id < n_vocab ? id : 0; sel_logit[i] = v; }
    });
    __syncwarp();
    sampled_tail_warp(sel_ids, sel_logit, k, p, t, out, ex);
}

// ---- COUPLED DRAFT SAMPLING (include/strata/core/coupled_draft.hpp): the MTP draft layer samples its draft with the
// target's chain and the target's Philox draw.  Everything that varies per request or per round - the chain's
// parameters, the seed, the counter (from the cell's step record), the penalty history - is read from DEVICE memory:
// these kernels are captured into the drafter's round/step graphs.  One row, `nv` logits: the draft head's
// vocabulary subset (rt/draft_vocab.bin) or the whole vocabulary; `sub_to_id` maps a subset index to its token id.

/// The round's inputs: the request's SamplerParams and the history base (the last h slots before `cap`), from
/// mapped host memory into the device copies the chain's kernels read.
__global__ void coupled_stage_kernel(const SamplerParams* __restrict__ mp, const int* __restrict__ mh,
                                     SamplerParams* __restrict__ dp, int* __restrict__ ring, int cap) {
    const volatile int* s = (const volatile int*) mp;
    int* d = (int*) dp;
    for (int i = threadIdx.x; i < (int) (sizeof(SamplerParams) / sizeof(int)); i += blockDim.x) d[i] = s[i];
    const int h = strata::core::coupled_hist_len(((const volatile SamplerParams*) mp)->penalty_last_n, cap);
    const volatile int* vh = (const volatile int*) mh;
    for (int i = cap - h + (int) threadIdx.x; i < cap; i += blockDim.x) ring[i] = vh[i];
}

/// The penalties, applied in place to the draft logits before the selection - the target applies the same
/// `apply_penalties` to the same token with the same count over the same window before its own.  Draft j's window
/// is the ring's [cap + j - h, cap + j): the base plus drafts 0 .. j-1.  A history token outside the draft head's
/// subset has no logit here.  Each distinct token is penalised once: the entry whose `atomicOr` sets its bit does it.
__global__ void coupled_penalize_kernel(float* __restrict__ logits, int nv, const int* __restrict__ id_to_sub,
                                        int id_vocab, const SamplerParams* __restrict__ dp, const int* __restrict__ ring,
                                        int cap, int j) {
    const SamplerParams p = *dp;
    const int h = strata::core::coupled_hist_len(p.penalty_last_n, cap);
    if (h <= 0) return;
    const int* hrow = ring + strata::core::coupled_hist_start(cap, j, h);
    extern __shared__ unsigned int seen[];
    const int words = (nv + 31) / 32;
    for (int w = threadIdx.x; w < words; w += blockDim.x) seen[w] = 0u;
    __syncthreads();
    for (int i = threadIdx.x; i < h; i += blockDim.x) {
        const int v = hrow[i];
        if (v < 0 || v >= id_vocab) continue;
        const int s = id_to_sub != nullptr ? id_to_sub[v] : v;
        if (s < 0 || s >= nv) continue;
        const unsigned bit = 1u << (s & 31);
        if (atomicOr(&seen[s >> 5], bit) & bit) continue;
        logits[s] = apply_penalties(logits[s], history_count(hrow, h, v), p);
    }
}

/// The merge of `sampler_split_merge_kernel` (lists of `kpart` entries, the request's top_k taken from them), then
/// `sampled_tail_warp` with the counter of the row that will verify this draft.  Lane 0 maps the pick to its token
/// id, writes it and its probability, and appends it to the ring for the next draft's penalty window.
__global__ void __launch_bounds__(32)
coupled_merge_kernel(const int2* __restrict__ cand, int n_blocks, int nv, int kpart,
                     const SamplerParams* __restrict__ dp, const int* __restrict__ step_rec,
                     const int* __restrict__ sub_to_id, int* __restrict__ ring, int cap, int j, int* __restrict__ out_id,
                     float* __restrict__ out_prob) {
    const int lane = (int) threadIdx.x;
    SamplerParams p = *dp;
    p.counter = strata::core::coupled_draft_counter((int64_t) step_rec[0]);
    const int k = sampled_k(p.top_k, nv);    // <= kpart: the first k of a union lie in the first k of each list
    __shared__ int2 lists[kSplitMaxBlocks * kSelMax];
    __shared__ int sel_ids[kSelMax];
    __shared__ float sel_logit[kSelMax];
    __shared__ double ex[kSelMax];
    __shared__ int pick[1];
    __shared__ float prob[1];
    for (int e = lane; e < n_blocks * kpart; e += 32) lists[e] = cand[e];
    __syncwarp();
    warp_merge_lists(lists, n_blocks, kpart, k, nv, [&](int i, float v, int id) {
        if (lane == 0) { sel_ids[i] = id < nv ? id : 0; sel_logit[i] = v; }
    });
    __syncwarp();
    if (p.greedy || p.temperature <= 0.0f) {   // never launched for greedy requests; the argmax, defensively
        if (lane == 0) { pick[0] = sel_ids[0]; prob[0] = 1.0f; }
    } else {
        sampled_tail_warp<true>(sel_ids, sel_logit, k, p, 0, pick, ex, prob);
    }
    __syncwarp();
    if (lane == 0) {
        const int s = pick[0];
        const int id = sub_to_id != nullptr ? sub_to_id[s] : s;
        *out_id = id;
        *out_prob = prob[0];
        ring[cap + j] = id;
    }
}

// Which sampled path runs, read once: `STRATA_OLD_SAMPLER=1` is `sampler_kernel` (engine 0.1.20),
// `STRATA_SAMPLER_ONE_BLOCK=1` the one-block kernel; by default the split top_k wherever it applies.
enum class SampledPath { Split, OneBlock, Old };

bool env_flag(const char* name) {
    const char* e = std::getenv(name);
    return e != nullptr && *e != '\0' && std::strcmp(e, "0") != 0;
}

SampledPath sampled_path() {
    static const SampledPath path = env_flag("STRATA_OLD_SAMPLER")         ? SampledPath::Old
                                    : env_flag("STRATA_SAMPLER_ONE_BLOCK") ? SampledPath::OneBlock
                                                                           : SampledPath::Split;
    return path;
}

// A stream being captured into a graph must not reach `split_scratch` (cudaMalloc): it
// gets the one-block kernel, which needs no memory of its own.  The legacy stream cannot be captured.
bool stream_capturing(void* stream) {
    if (stream == nullptr) return false;
    cudaStreamCaptureStatus st = cudaStreamCaptureStatusNone;
    if (cudaStreamIsCapturing((cudaStream_t) stream, &st) != cudaSuccess) {
        (void) cudaGetLastError();
        return true;
    }
    return st != cudaStreamCaptureStatusNone;
}

// The split's block lists, one buffer per (device, stream): launches on one stream run in order, so a stream reuses
// its buffer with no sync, and two streams never share one.  Grown on demand (at least doubling, up to the size of
// `kSplitMaxRows` rows at the widest vocabulary: 2 MB), never shrunk.  A grown-out buffer is retired, not freed: a
// pointer handed out earlier (to another host thread on the same stream, say) may still be waiting for its launch,
// and freeing it would need a sync that proves nothing about that thread.  Doubling keeps a slot's retired buffers
// smaller than its live one, so a slot holds about 4 MB at most (0.5 MB for the engine's <= 16 rows).  nullptr
// when the memory cannot be had, and the caller falls back to the one-block kernel; a failed grow is remembered, so
// later calls of that size do not retry cudaMalloc each time.
int2* split_scratch(void* stream, size_t entries) {
    struct Slot {
        int device;
        void* stream;
        int2* ptr;
        size_t entries;
        size_t failed;                       // the smallest size cudaMalloc refused (0: none)
    };
    static std::mutex mu;
    static std::vector<Slot> slots;
    static std::vector<int2*> retired;
    int device = 0;
    if (cudaGetDevice(&device) != cudaSuccess) {
        (void) cudaGetLastError();
        return nullptr;
    }
    std::lock_guard<std::mutex> lock(mu);
    Slot* slot = nullptr;
    for (Slot& s : slots)
        if (s.device == device && s.stream == stream) slot = &s;
    if (slot == nullptr) {
        slots.push_back({device, stream, nullptr, 0, 0});
        slot = &slots.back();
    }
    if (slot->entries >= entries) return slot->ptr;
    if (slot->failed != 0 && entries >= slot->failed) return nullptr;
    constexpr size_t kCap = (size_t) kSplitMaxRows * kSplitMaxBlocks * kSelMax;
    size_t want = 2 * slot->entries < kCap ? 2 * slot->entries : kCap;
    if (want < entries) want = entries;
    int2* ptr = nullptr;
    if (cudaMalloc(&ptr, want * sizeof(int2)) != cudaSuccess) {
        (void) cudaGetLastError();
        want = entries;
        if (cudaMalloc(&ptr, want * sizeof(int2)) != cudaSuccess) {
            (void) cudaGetLastError();
            slot->failed = entries;
            return nullptr;
        }
    }
    if (slot->ptr != nullptr) retired.push_back(slot->ptr);
    slot->ptr = ptr;
    slot->entries = want;
    return ptr;
}

}  // namespace

void sample_tokens(const float* logits, int n_tokens, int n_vocab, const int* history, int history_len,
                   const SamplerParams& p, int* out, void* stream) {
    if (n_tokens <= 0 || n_vocab <= 0) return;
    if (p.penalty_last_n > 0 && (history == nullptr || history_len <= 0)) {
        std::fprintf(stderr, "sample_tokens: penalty_last_n %d needs a history (got %p, len %d)\n",
                     p.penalty_last_n, (const void*) history, history_len);
        std::exit(1);
    }
    const unsigned shmem = (history != nullptr && history_len > 0 && p.penalty_last_n > 0)
                               ? (unsigned) ((n_vocab + 31) / 32) * sizeof(unsigned)   // the penalty bitmap
                               : 0;
    if (p.greedy || p.temperature <= 0.0f) {
        // One block per token, 1,024 threads over the vocabulary.  See `sampler_greedy_kernel`.
        const int gthreads = 1024;
        sampler_greedy_kernel<<<(unsigned) n_tokens, gthreads, shmem, (cudaStream_t) stream>>>(
            logits, n_vocab, history, history_len, p, p.penalty_last_n, p.penalty_last_n, out);
    } else if (sampled_path() == SampledPath::Old) {
        // The same block-per-token shape: the selection's k argmax rounds reduce inside the block.  See
        // `sampler_kernel`'s header for what the old one-thread-per-token launch cost.
        sampler_kernel<<<(unsigned) n_tokens, 1024, shmem, (cudaStream_t) stream>>>(
            logits, n_vocab, n_tokens, history, history_len, p, out);
    } else {
        // The split top_k by default: stage 1 over (61 blocks x rows) for 248,320 logits, stage 2 one warp per
        // row.  The one-block kernel when asked for, or when the split cannot run: a wider vocabulary than the merge
        // holds, more than `kSplitMaxRows` rows (the engine samples at most a verify window), a stream under capture,
        // no scratch.
        const int k = sampled_k(p.top_k, n_vocab);
        const int n_blocks = (n_vocab + kSplitBlockSpan - 1) / kSplitBlockSpan;
        int2* scratch = nullptr;
        if (sampled_path() == SampledPath::Split && n_blocks <= kSplitMaxBlocks && n_tokens <= kSplitMaxRows &&
            !stream_capturing(stream))
            // sized for 16 rows and 64 entries at least, so a verify window or a wider top_k does not regrow it
            scratch = split_scratch(stream, (size_t) (n_tokens > 16 ? n_tokens : 16) * n_blocks * kSelMax);
        if (scratch != nullptr) {
            sampler_split_part_kernel<<<dim3((unsigned) n_blocks, (unsigned) n_tokens), kSplitWarps * 32, 0,
                                        (cudaStream_t) stream>>>(logits, n_vocab, history, history_len, p, k,
                                                                 n_blocks, scratch);
            sampler_split_merge_kernel<<<(unsigned) n_tokens, 32, 0, (cudaStream_t) stream>>>(
                scratch, n_blocks, n_vocab, p, k, out);
        } else {
            sampler_one_block_kernel<<<(unsigned) n_tokens, 1024, shmem, (cudaStream_t) stream>>>(
                logits, n_vocab, history, history_len, p, out);
        }
    }
    const cudaError_t e = cudaGetLastError();
    if (e != cudaSuccess) {
        std::fprintf(stderr, "sample_tokens launch: %s\n", cudaGetErrorString(e));
        std::exit(1);
    }
    if (stream == nullptr) cudaDeviceSynchronize();
}

namespace {
int coupled_blocks(int nv) { return (nv + kSplitBlockSpan - 1) / kSplitBlockSpan; }
int coupled_kpart(int nv) { return nv < kSelMax ? nv : kSelMax; }
void coupled_check(const char* what) {
    const cudaError_t e = cudaGetLastError();
    if (e != cudaSuccess) {
        std::fprintf(stderr, "%s launch: %s\n", what, cudaGetErrorString(e));
        std::exit(1);
    }
}
}  // namespace

size_t coupled_draft_scratch_bytes(int nv) {
    if (nv <= 0 || coupled_blocks(nv) > kSplitMaxBlocks) return 0;
    return (size_t) coupled_blocks(nv) * (size_t) kSelMax * sizeof(int2);
}

void coupled_draft_stage(const SamplerParams* mapped_params, const int32_t* mapped_hist, SamplerParams* params,
                         int32_t* ring, int cap, void* stream) {
    coupled_stage_kernel<<<1, 256, 0, (cudaStream_t) stream>>>(mapped_params, mapped_hist, params, ring, cap);
    coupled_check("coupled_draft_stage");
}

void coupled_draft_sample(float* logits, int nv, const int32_t* sub_to_id, const int32_t* id_to_sub, int id_vocab,
                          const SamplerParams* params, int32_t* ring, int cap, int j, const int32_t* step_rec,
                          void* scratch, int32_t* out_id, float* out_prob, void* stream) {
    const cudaStream_t s = (cudaStream_t) stream;
    const int n_blocks = coupled_blocks(nv), kpart = coupled_kpart(nv);
    if (nv <= 0 || n_blocks > kSplitMaxBlocks || scratch == nullptr) {
        std::fprintf(stderr, "coupled_draft_sample: %d logits need scratch and at most %d blocks\n", nv, kSplitMaxBlocks);
        std::exit(1);
    }
    coupled_penalize_kernel<<<1, 1024, (unsigned) ((nv + 31) / 32) * sizeof(unsigned), s>>>(logits, nv, id_to_sub,
                                                                                           id_vocab, params, ring, cap, j);
    coupled_check("coupled_penalize");
    // the selection of the split sampler, unchanged: every 4,096-logit block's first `kpart` (the widest list,
    // since the request's top_k is only known on the device), penalties already applied above
    sampler_split_part_kernel<<<dim3((unsigned) n_blocks, 1u), kSplitWarps * 32, 0, s>>>(
        logits, nv, nullptr, 0, SamplerParams{}, kpart, n_blocks, (int2*) scratch);
    coupled_check("coupled_draft split part");
    coupled_merge_kernel<<<1, 32, 0, s>>>((const int2*) scratch, n_blocks, nv, kpart, params, step_rec, sub_to_id, ring,
                                          cap, j, out_id, out_prob);
    coupled_check("coupled_draft merge");
}

}  // namespace strata::kernels
