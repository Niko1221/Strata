// src/kernels/cuda/sampler.cu - P2.S2: the sampler chain, in llama.cpp's order.
//
//     penalties -> DRY -> top_k -> top_p -> min_p -> temperature -> pick
//
// THE ORDER IS THE WHOLE CONTENT OF THIS FILE.  llama.cpp builds its chain by walking `params.samplers`, whose
// default is { PENALTIES, DRY, TOP_N_SIGMA, TOP_K, TYPICAL_P, TOP_P, MIN_P, XTC, TEMPERATURE } (`common/common.h`)
// - ONE penalties stage, then DRY, and TEMPERATURE AFTER THE TRUNCATION FILTERS. (Issue #53: this file
// used to apply the penalties a second time after the temperature, and min_p before top_p - both taken from the
// order of the `case` labels in `common/sampling.cpp`, which is not the order the chain runs.)  Every order
// produces a valid token, so only a comparison at the distribution level can tell them apart; the parity test
// does that against an independently computed distribution.
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

// `count_in_history` and the token-penalty application, transcribed from `llama_sampler_penalties_apply`.
// The repeat penalty MULTIPLIES for non-positive logits and DIVIDES for positive ones - dividing
// unconditionally is the natural reading of the source paper and it INVERTS the penalty on half the
// vocabulary.  The presence penalty is `float(count > 0)`, a boolean, not the count.
__device__ __forceinline__ int history_count(const int* __restrict__ h, int n, int v) {
    int c = 0;
    for (int i = 0; i < n; ++i) if (h[i] == v) ++c;
    return c;
}

__device__ __forceinline__ bool dry_enabled(const SamplerParams& p) {
    return p.dry_multiplier > 0.0f && p.dry_base >= 1.0f && p.dry_allowed_length >= 0 &&
           p.dry_penalty_last_n > 0;
}

__device__ __forceinline__ int dry_breaker_lower_bound(const DrySequenceBreaker* breakers, int count, int head) {
    int lo = 0, hi = count;
    while (lo < hi) {
        const int mid = lo + (hi - lo) / 2;
        if (breakers[mid].head < head) lo = mid + 1;
        else hi = mid;
    }
    return lo;
}

__device__ __forceinline__ int dry_breaker_match_length(const int* newest_history, int hlen, int i,
                                                        int token, const SamplerParams& p) {
    int longest = -1;
    if (p.dry_breaker_count <= 0 || p.dry_breakers == nullptr) return longest;
    const int first = dry_breaker_lower_bound(p.dry_breakers, p.dry_breaker_count, token);
    for (int b = first; b < p.dry_breaker_count && p.dry_breakers[b].head == token; ++b) {
        const DrySequenceBreaker entry = p.dry_breakers[b];
        if (entry.tail_length > i || entry.tail_length <= longest ||
            (entry.tail_length > 0 && p.dry_breaker_tails == nullptr)) continue;
        bool match = true;
        for (int j = 0; j < entry.tail_length; ++j) {
            // The current head is `i` tokens back from the newest token; its tail continues toward the newest.
            if (p.dry_breaker_tails[entry.tail_offset + j] != newest_history[hlen - i + j]) {
                match = false;
                break;
            }
        }
        if (match) longest = entry.tail_length;
    }
    return longest;
}

__device__ __forceinline__ bool dry_single_token_breaker(int token, const SamplerParams& p) {
    if (p.dry_breaker_count <= 0 || p.dry_breakers == nullptr) return false;
    const int first = dry_breaker_lower_bound(p.dry_breakers, p.dry_breaker_count, token);
    for (int b = first; b < p.dry_breaker_count && p.dry_breakers[b].head == token; ++b)
        if (p.dry_breakers[b].tail_length == 0) return true;
    return false;
}

__device__ __forceinline__ void dry_sift_down(unsigned long long* values, int root, int count) {
    while (root < count / 2) {
        int child = root * 2 + 1;
        if (child + 1 < count && values[child] < values[child + 1]) ++child;
        if (values[root] >= values[child]) break;
        const unsigned long long tmp = values[root];
        values[root] = values[child];
        values[child] = tmp;
        root = child;
    }
}

__device__ int build_dry_map(const int* newest_history, int hlen, const SamplerParams& p,
                             int* repeat_count, unsigned long long* token_repeats) {
    if (!dry_enabled(p) || hlen <= p.dry_allowed_length) return 0;

    int rep_limit = hlen;
    for (int i = 0; i < hlen; ++i) {
        const int token = newest_history[hlen - 1 - i];
        const int longest = dry_breaker_match_length(newest_history, hlen, i, token, p);
        if (longest >= 0) {
            rep_limit = i - longest;
            break;
        }
    }
    if (rep_limit < p.dry_allowed_length) return 0;

    for (int i = 0; i < hlen; ++i) repeat_count[i] = 0;
    // Reverse Z algorithm from llama_sampler_dry_apply. `newest_history` is oldest-to-newest in memory,
    // so `rat(i)` in llama.cpp is `newest_history[hlen - 1 - i]` here.
    const int last = hlen - 1;
    int rt = 0, lt = 0;
    for (int k = 1; k < hlen; ++k) {
        if (k > rt) {
            int n = 0;
            while (n + k < hlen && newest_history[hlen - 1 - n] == newest_history[hlen - 1 - (n + k)]) ++n;
            repeat_count[last - k] = n < rep_limit ? n : rep_limit;
            if (n > 0) { lt = k; rt = k + n - 1; }
        } else {
            const int pair = k - lt;
            const int right_part_len = rt - k + 1;
            if (repeat_count[last - pair] < right_part_len) {
                const int n = repeat_count[last - pair] < rep_limit ? repeat_count[last - pair] : rep_limit;
                repeat_count[last - k] = n;
            } else {
                int i = rt + 1;
                while (i < hlen && newest_history[hlen - 1 - i] == newest_history[hlen - 1 - (i - k)]) ++i;
                const int n = i - k < rep_limit ? i - k : rep_limit;
                repeat_count[last - k] = n;
                lt = k;
                rt = i - 1;
            }
        }
    }

    int count = 0;
    for (int i = 0; i < hlen - 1; ++i) {
        const int repeat_len = repeat_count[i];
        const int token = newest_history[i + 1];  // llama.cpp: rat(hlen - 2 - i)
        if (repeat_len >= p.dry_allowed_length && token >= 0) {
            token_repeats[count++] = ((unsigned long long) (unsigned int) token << 32) |
                                    (unsigned int) repeat_len;
        }
    }
    // Sort by (token id, repeat length), then retain the maximum repeat length per token. The input is bounded
    // by the 4096-token DRY window, so an in-place heapsort keeps the shared-memory lookup table compact.
    for (int root = count / 2; root > 0; --root) dry_sift_down(token_repeats, root - 1, count);
    for (int end = count - 1; end > 0; --end) {
        const unsigned long long tmp = token_repeats[0];
        token_repeats[0] = token_repeats[end];
        token_repeats[end] = tmp;
        dry_sift_down(token_repeats, 0, end);
    }
    int unique = 0;
    for (int i = 0; i < count;) {
        const unsigned int token = (unsigned int) (token_repeats[i] >> 32);
        int j = i + 1;
        while (j < count && (unsigned int) (token_repeats[j] >> 32) == token) ++j;
        token_repeats[unique++] = token_repeats[j - 1];  // largest repeat length in this token's run
        i = j;
    }
    return unique;
}

__device__ __forceinline__ float dry_penalty_for_token(int token, const unsigned long long* token_repeats,
                                                       int count, const SamplerParams& p) {
    if (count <= 0 || dry_single_token_breaker(token, p)) return 0.0f;
    int lo = 0, hi = count;
    while (lo < hi) {
        const int mid = lo + (hi - lo) / 2;
        if ((unsigned int) (token_repeats[mid] >> 32) < (unsigned int) token) lo = mid + 1;
        else hi = mid;
    }
    if (lo >= count || (unsigned int) (token_repeats[lo] >> 32) != (unsigned int) token) return 0.0f;
    int exponent = (int) (unsigned int) token_repeats[lo] - p.dry_allowed_length;
    if (p.dry_base > 1.000001f) {
        const int max_exponent = (int) (88.7228391f / logf(p.dry_base));
        if (exponent > max_exponent) exponent = max_exponent;
    }
    return p.dry_multiplier * powf(p.dry_base, (float) exponent);
}

__device__ __forceinline__ float apply_penalties(float logit, int count, float dry_penalty,
                                                  const SamplerParams& p) {
    if (count > 0) {
        if (logit <= 0.0f) logit *= p.penalty_repeat;
        else               logit /= p.penalty_repeat;
        logit -= (float) count * p.penalty_freq + p.penalty_present;
    }
    if (dry_penalty != 0.0f) logit -= dry_penalty;
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
    const int* row = history ? history + (size_t) t * history_len : nullptr;
    int penalty_len = row ? (plen < history_len ? plen : history_len) : 0;
    if (penalty_len < 0) penalty_len = 0;
    const int dry_last_n = dry_enabled(p) ? (p.dry_penalty_last_n < 4096 ? p.dry_penalty_last_n : 4096) : 0;
    int dry_len = row ? (dry_last_n < history_len ? dry_last_n : history_len) : 0;
    if (dry_len < 0) dry_len = 0;
    const int union_len = penalty_len > dry_len ? penalty_len : dry_len;
    const int* penalty_history = row && penalty_len > 0 ? row + history_len - penalty_len : nullptr;
    const int* dry_history = row && dry_len > 0 ? row + history_len - dry_len : nullptr;
    const int* union_history = row && union_len > 0 ? row + history_len - union_len : nullptr;

    // PENALTY MEMBERSHIP AS A BITMAP.  The history touches at most `hlen` tokens of a quarter-million
    // vocabulary, but the naive `history_count` per candidate per argmax round costs O(k x n_vocab x hlen)
    // integer compares (~318 M per token at k=20, hlen=64 - measured 45 -> 31 tok/s on a real workload).
    // A shared bitmap gives an O(1) membership test, and only the (at most hlen) hits pay the count scan;
    // the counts - and therefore every sampled value - are exactly what the per-candidate scan produced.
    extern __shared__ unsigned char sampler_shared[];
    __shared__ int dry_entries_shared;
    const int bits_words = (int) ((n_vocab + 31) / 32);
    const size_t bits_bytes = (size_t) bits_words * sizeof(unsigned int);
    unsigned int* penal_bits = (unsigned int*) sampler_shared;
    const bool use_bits = union_history != nullptr && union_len > 0 && bits_words > 0;
    if (use_bits) {
        for (int w = threadIdx.x; w < bits_words; w += blockDim.x) penal_bits[w] = 0u;
        __syncthreads();
        for (int i = threadIdx.x; i < union_len; i += blockDim.x)
            if (union_history[i] >= 0 && union_history[i] < n_vocab)   // an id outside the vocabulary is never a candidate
                atomicOr(&penal_bits[union_history[i] >> 5], 1u << (union_history[i] & 31));
        __syncthreads();
    }
    auto hit_count = [&](int v) -> int {
        if (!use_bits || !(penal_bits[v >> 5] & (1u << (v & 31)))) return 0;
        return history_count(penalty_history, penalty_len, v);
    };
    const int dry_workspace_len = dry_len > p.dry_allowed_length ? dry_len : 0;
    int dry_entries_count = 0;
    int* dry_repeat_count = nullptr;
    unsigned long long* dry_token_repeats = nullptr;
    if (dry_workspace_len > 0) {
        const size_t repeat_offset = (bits_bytes + 7u) & ~size_t(7u);
        const size_t pairs_offset = (repeat_offset + (size_t) dry_workspace_len * sizeof(int) + 7u) & ~size_t(7u);
        dry_repeat_count = (int*) (sampler_shared + repeat_offset);
        dry_token_repeats = (unsigned long long*) (sampler_shared + pairs_offset);
        if (threadIdx.x == 0)
            dry_entries_count = build_dry_map(dry_history, dry_len, p, dry_repeat_count, dry_token_repeats);
        if (threadIdx.x == 0) dry_entries_shared = dry_entries_count;
        __syncthreads();
        dry_entries_count = dry_entries_shared;
    }
    auto dry_penalty = [&](int v) -> float {
        if (!use_bits || !(penal_bits[v >> 5] & (1u << (v & 31)))) return 0.0f;
        return dry_penalty_for_token(v, dry_token_repeats, dry_entries_count, p);
    };

    // `n_vocab` is the "no candidate" index: it loses every comparison to a real one, so a thread with no
    // elements contributes nothing rather than contributing a bogus zero.
    float bv = __int_as_float(0xff800000);   // -inf
    int best = n_vocab;
    for (int v = threadIdx.x; v < n_vocab; v += blockDim.x) {
        const float s = apply_penalties(l[v], hit_count(v), dry_penalty(v), p);
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
    // the truncation filters - the survivors are chosen on the raw logits and only then scaled.
    const float inv_t = p.temperature > 0.0f ? 1.0f / p.temperature : 0.0f;

    // Ordinary token penalties and DRY can use different tails of the same per-row history.
    const int* row = history ? history + (size_t) t * history_len : nullptr;
    int penalty_len = row ? (p.penalty_last_n < history_len ? p.penalty_last_n : history_len) : 0;
    if (penalty_len < 0) penalty_len = 0;
    const int dry_last_n = dry_enabled(p) ? (p.dry_penalty_last_n < 4096 ? p.dry_penalty_last_n : 4096) : 0;
    int dry_len = row ? (dry_last_n < history_len ? dry_last_n : history_len) : 0;
    if (dry_len < 0) dry_len = 0;
    const int union_len = penalty_len > dry_len ? penalty_len : dry_len;
    const int* penalty_history = row && penalty_len > 0 ? row + history_len - penalty_len : nullptr;
    const int* dry_history = row && dry_len > 0 ? row + history_len - dry_len : nullptr;
    const int* union_history = row && union_len > 0 ? row + history_len - union_len : nullptr;

    // The bitmap covers the union of the token-penalty and DRY windows; see the greedy kernel for its cost note.
    extern __shared__ unsigned char sampler_shared[];
    __shared__ int dry_entries_shared;
    const int bits_words = (int) ((n_vocab + 31) / 32);
    const size_t bits_bytes = (size_t) bits_words * sizeof(unsigned int);
    unsigned int* penal_bits = (unsigned int*) sampler_shared;
    const bool use_bits = union_history != nullptr && union_len > 0 && bits_words > 0;
    if (use_bits) {
        for (int w = threadIdx.x; w < bits_words; w += blockDim.x) penal_bits[w] = 0u;
        __syncthreads();
        for (int i = threadIdx.x; i < union_len; i += blockDim.x)
            if (union_history[i] >= 0 && union_history[i] < n_vocab)
                atomicOr(&penal_bits[union_history[i] >> 5], 1u << (union_history[i] & 31));
        __syncthreads();
    }
    auto hit_count = [&](int v) -> int {
        if (!use_bits || !(penal_bits[v >> 5] & (1u << (v & 31)))) return 0;
        return history_count(penalty_history, penalty_len, v);
    };
    const int dry_workspace_len = dry_len > p.dry_allowed_length ? dry_len : 0;
    int dry_entries_count = 0;
    int* dry_repeat_count = nullptr;
    unsigned long long* dry_token_repeats = nullptr;
    if (dry_workspace_len > 0) {
        const size_t repeat_offset = (bits_bytes + 7u) & ~size_t(7u);
        const size_t pairs_offset = (repeat_offset + (size_t) dry_workspace_len * sizeof(int) + 7u) & ~size_t(7u);
        dry_repeat_count = (int*) (sampler_shared + repeat_offset);
        dry_token_repeats = (unsigned long long*) (sampler_shared + pairs_offset);
        if (threadIdx.x == 0)
            dry_entries_count = build_dry_map(dry_history, dry_len, p, dry_repeat_count, dry_token_repeats);
        if (threadIdx.x == 0) dry_entries_shared = dry_entries_count;
        __syncthreads();
        dry_entries_count = dry_entries_shared;
    }
    auto dry_penalty = [&](int v) -> float {
        if (!use_bits || !(penal_bits[v >> 5] & (1u << (v & 31)))) return 0.0f;
        return dry_penalty_for_token(v, dry_token_repeats, dry_entries_count, p);
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
            const float s = apply_penalties(l[v], hit_count(v), dry_penalty(v), p);
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

}  // namespace

void sample_tokens(const float* logits, int n_tokens, int n_vocab, const int* history, int history_len,
                   const SamplerParams& p, int* out, void* stream) {
    if (n_tokens <= 0 || n_vocab <= 0) return;
    const bool dry_on = p.dry_multiplier > 0.0f && p.dry_base >= 1.0f && p.dry_allowed_length >= 0 &&
                        p.dry_penalty_last_n > 0;
    if ((p.penalty_last_n > 0 || dry_on) && (history == nullptr || history_len <= 0)) {
        std::fprintf(stderr, "sample_tokens: token penalties or DRY need a history (got %p, len %d)\n",
                     (const void*) history, history_len);
        std::exit(1);
    }
    const bool have_history_window = history != nullptr && history_len > 0 &&
                                     (p.penalty_last_n > 0 || (dry_on && p.dry_penalty_last_n > 0));
    const size_t bits_bytes = have_history_window ? (size_t) ((n_vocab + 31) / 32) * sizeof(unsigned) : 0;
    const int dry_len = have_history_window && dry_on
                            ? std::min(4096, std::min(p.dry_penalty_last_n, history_len)) : 0;
    const int dry_workspace_len = dry_len > p.dry_allowed_length ? dry_len : 0;
    const size_t repeat_offset = (bits_bytes + 7u) & ~size_t(7u);
    const size_t pairs_offset = (repeat_offset + (size_t) dry_workspace_len * sizeof(int) + 7u) & ~size_t(7u);
    const size_t shmem_size = dry_workspace_len > 0 ? pairs_offset + (size_t) dry_workspace_len * sizeof(unsigned long long)
                                                     : bits_bytes;
    const unsigned shmem = (unsigned) shmem_size;
    if (p.greedy || p.temperature <= 0.0f) {
        if (shmem_size > 48u * 1024u) {
            const cudaError_t attr = cudaFuncSetAttribute(sampler_greedy_kernel,
                                                          cudaFuncAttributeMaxDynamicSharedMemorySize,
                                                          (int) shmem_size);
            if (attr != cudaSuccess) {
                std::fprintf(stderr, "sample_tokens: DRY needs %zu bytes of shared memory: %s\n",
                             shmem_size, cudaGetErrorString(attr));
                std::exit(1);
            }
        }
        // One block per token, 1,024 threads over the vocabulary.  See `sampler_greedy_kernel`.
        const int gthreads = 1024;
        sampler_greedy_kernel<<<(unsigned) n_tokens, gthreads, shmem, (cudaStream_t) stream>>>(
            logits, n_vocab, history, history_len, p, p.penalty_last_n, p.penalty_last_n, out);
    } else {
        if (shmem_size > 48u * 1024u) {
            const cudaError_t attr = cudaFuncSetAttribute(sampler_kernel,
                                                          cudaFuncAttributeMaxDynamicSharedMemorySize,
                                                          (int) shmem_size);
            if (attr != cudaSuccess) {
                std::fprintf(stderr, "sample_tokens: DRY needs %zu bytes of shared memory: %s\n",
                             shmem_size, cudaGetErrorString(attr));
                std::exit(1);
            }
        }
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
