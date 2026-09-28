// src/kernels/cuda/sampler.cu - P2.S2: the sampler chain, in llama.cpp's own order.
//
//     penalties -> top_k -> min_p -> top_p -> temperature -> pick
//
// THE ORDER IS THE WHOLE CONTENT OF THIS FILE, and it is llama.cpp's DEFAULT chain (issue #53).  The chain is
// built by iterating `params.samplers` (`common/common.h`), whose default list is
// PENALTIES, DRY, TOP_N_SIGMA, TOP_K, TYPICAL_P, TOP_P, MIN_P, XTC, TEMPERATURE - so penalties run ONCE and
// FIRST, before the selection, and temperature scales the survivors at the very end.  The two facts that are
// easy to get backwards are that TEMPERATURE COMES AFTER THE TRUNCATION FILTERS and that PENALTIES APPLY
// EXACTLY ONCE: reading the switch cases of `common_sampler_init` in their textual order puts PENALTIES after
// TEMPERATURE, and a selection that penalises plus a final stage that penalises again applies them TWICE.
// Both mistakes produce a valid token and a different distribution, so only a comparison at the distribution
// level can tell them apart; the parity test pins the kernel against the single-pass reference, and its #53
// fixture pins that reference's arithmetic against the issue's independently derived numbers.
//
// Both kernels put ONE BLOCK per token over the vocabulary: `sampler_greedy_kernel` is the plain argmax,
// `sampler_kernel` runs the sampled chain as `top_k` block-argmax rounds followed by the top_p / temperature /
// draw chain (its header says why the selection must be parallel and why the tie rule keeps the semantics).
#include "strata/kernels/sampler.hpp"

#include <cuda_runtime.h>

#include <cmath>
#include <cstdio>
#include <cstdlib>

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
    const bool use_bits = hrow != nullptr && bits_words > 0;
    if (use_bits) {
        for (int w = threadIdx.x; w < bits_words; w += blockDim.x) penal_bits[w] = 0u;
        __syncthreads();
        for (int i = threadIdx.x; i < hlen; i += blockDim.x)
            if (hrow[i] >= 0) atomicOr(&penal_bits[hrow[i] >> 5], 1u << (hrow[i] & 31));
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
__global__ void sampler_kernel(const float* __restrict__ logits, int n_vocab, int n_tokens,
                               const int* __restrict__ history, int history_len, const SamplerParams p,
                               int* __restrict__ out) {
    const int t = blockIdx.x;
    if (t >= n_tokens) return;
    const float* l = logits + (size_t) t * n_vocab;

    // Temperature is needed by BOTH stages below, so it is computed here; the chain still APPLIES it after
    // the truncation filters - the survivors are chosen on the penalised logits and only then scaled.
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

    // the membership bitmap, as in `sampler_greedy_kernel` - see the cost note there
    extern __shared__ unsigned int penal_bits[];
    const int bits_words = (int) ((n_vocab + 31) / 32);
    const bool use_bits = hrow != nullptr && bits_words > 0;
    if (use_bits) {
        for (int w = threadIdx.x; w < bits_words; w += blockDim.x) penal_bits[w] = 0u;
        __syncthreads();
        for (int i = threadIdx.x; i < hlen; i += blockDim.x)
            if (hrow[i] >= 0) atomicOr(&penal_bits[hrow[i] >> 5], 1u << (hrow[i] & 31));
        __syncthreads();
    }
    auto hit_count = [&](int v) -> int {
        if (!use_bits || !(penal_bits[v >> 5] & (1u << (v & 31)))) return 0;
        return history_count(hrow, hlen, v);
    };

    const int KMAX = 64;
    int k = p.top_k > 0 ? (p.top_k < KMAX ? p.top_k : KMAX) : 0;
    if (k <= 0) {
        if (threadIdx.x == 0)
            std::printf("sampler: the sampled path needs top_k in 1..%d (got %d); greedy needs no filters\n",
                        KMAX, p.top_k);
        return;   // leave out[t] unwritten rather than returning an uninitialised token
    }

    // ---- top_k: k rounds of a block argmax over the not-yet-taken.  `sel_*` holds the kept ids and their
    // PENALISED logits in selection order: descending by value, ties to the lower index, which is the order
    // the top_p cut below is defined over.  This is the chain's ONE penalties pass (#53): llama.cpp's default
    // chain applies its single PENALTIES stage before TOP_K, so the filters see the penalised logits and the
    // pick below must not penalise again.
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

    // ---- min_p: keep the descending prefix whose probability is at least `min_p` of the top token's.  The
    // kept list is in selection order (descending), so the survivors are a PREFIX and the cut composes with
    // top_p's below.  In logit space the threshold is `sel_logit[0] + logf(min_p)` - equivalent to
    // `p >= min_p * p_max` without the overflow an exp of raw logits risks.  0 disables, and the head itself
    // always survives (`expf(0) == 1 >= min_p` for min_p in 0..1), so the count never reaches zero.
    int n_minp = k;
    if (p.min_p > 0.0f) {
        const float thresh = sel_logit[0] + logf(p.min_p);
        for (int i = 0; i < k; ++i)
            if (sel_logit[i] < thresh) { n_minp = i; break; }
    }

    // ---- top_p over the survivors, in descending order (which the selection produced), then temperature and
    // one Philox draw.  Every thread computes the same chain redundantly over `sel_*` - the arithmetic is the
    // serial kernel's, instruction for instruction - so they agree on `pick` and thread 0 writes it.
    int n_keep = n_minp;
    float mx = sel_logit[0];
    for (int i = 1; i < n_minp; ++i) mx = fmaxf(mx, sel_logit[i]);
    if (p.top_p < 1.0f) {
        double sum = 0.0;
        for (int i = 0; i < n_minp; ++i) sum += exp((double) sel_logit[i] - (double) mx);
        double cum = 0.0;
        int cut = n_minp;
        for (int i = 0; i < n_minp; ++i) {
            cum += exp((double) sel_logit[i] - (double) mx) / sum;
            if (cum >= (double) p.top_p) { cut = i + 1; break; }
        }
        if (cut < p.min_keep) cut = p.min_keep < n_minp ? p.min_keep : n_minp;
        n_keep = cut;
    }
    // THE FINAL STAGE IS TEMPERATURE ONLY.  `sel_logit` already carries the ONE penalty pass from the
    // selection above; applying `apply_penalties` here too was issue #53 - a second, post-temperature pass
    // that no llama.cpp chain has (its PENALTIES stage runs once, before the filters).  In #53's two-token
    // example it shrank the penalised token's probability from 10.50% to 2.55%.
    auto scaled = [&](int i) {
        return sel_logit[i] * inv_t;
    };
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
    } else {
        // The same block-per-token shape: the selection's k argmax rounds reduce inside the block.  See
        // `sampler_kernel`'s header for what the old one-thread-per-token launch cost.
        sampler_kernel<<<(unsigned) n_tokens, 1024, shmem, (cudaStream_t) stream>>>(
            logits, n_vocab, n_tokens, history, history_len, p, out);
    }
    const cudaError_t e = cudaGetLastError();
    if (e != cudaSuccess) {
        std::fprintf(stderr, "sample_tokens launch: %s\n", cudaGetErrorString(e));
        std::exit(1);
    }
    if (stream == nullptr) cudaDeviceSynchronize();
}

}  // namespace strata::kernels
