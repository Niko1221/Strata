// src/kernels/cuda/sampler.cu - P2.S2: the sampler chain, in the order docs/sampling.md settles.
//
//     penalties  ->  top_k  ->  top_p  ->  temperature  ->  pick
//
// THE ORDER IS THE WHOLE CONTENT OF THIS FILE.  `docs/sampling.md` transcribes it from llama.cpp's own chain
// (`common/sampling.cpp` L357/360/375/381/399) and the two facts that are easy to get backwards are that
// TEMPERATURE COMES AFTER THE TRUNCATION FILTERS and PENALTIES COME AFTER TEMPERATURE.  The intuitive order -
// scale first, then truncate, with penalties as pre-processing - is a different distribution.  Both produce a
// valid token, so only a comparison at the distribution level can tell them apart; the parity test does that
// explicitly by running the wrong order and requiring it to differ.
//
// Everything happens on the logits of ONE token in one thread.  The vocab is 248,320, which is far too large
// to sort per token on a naive path, so `top_k` uses `k` passes of a maximum scan - 20 x 248,320 = 5.0M
// comparisons per token.  Phase 3's note ("sort-free sampler: top-k = 20 makes this easy") is the optimisation;
// this is the correctness version.
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

    // `n_vocab` is the "no candidate" index: it loses every comparison to a real one, so a thread with no
    // elements contributes nothing rather than contributing a bogus zero.
    float bv = __int_as_float(0xff800000);   // -inf
    int best = n_vocab;
    for (int v = threadIdx.x; v < n_vocab; v += blockDim.x) {
        const float s = apply_penalties(l[v], hrow ? history_count(hrow, hlen, v) : 0, p);
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

__global__ void sampler_kernel(const float* __restrict__ logits, int n_vocab, int n_tokens,
                               const int* __restrict__ history, int history_len, const SamplerParams p,
                               int* __restrict__ out) {
    const int t = blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= n_tokens) return;

    // Work on a private copy: the chain mutates the distribution and the caller's logits are shared with the
    // next stage.  A 248,320-float allocation per thread is far too much for shared memory, so this uses a
    // two-pass approach instead - first find the keep set, then normalise over it - and never materialises a
    // second full-size buffer.
    const float* l = logits + (size_t) t * n_vocab;

    // Temperature is needed by BOTH paths below, so it is computed here; the chain still APPLIES it after
    // the truncation filters - the survivors are chosen on the raw logits and only then scaled.
    const float inv_t = p.temperature > 0.0f ? 1.0f / p.temperature : 0.0f;

    // ---- penalties: NOT IMPLEMENTED, and the ORDER they must take is transcribed here so the next person
    // does not have to guess it.  `llama_sampler_penalties_apply` (src/llama-sampler.cpp) runs AFTER
    // temperature in the chain (L381 against L375), and for every token seen in the last `penalty_last_n`:
    //
    //     if (logit <= 0) logit *= penalty_repeat;   else logit /= penalty_repeat;
    //     logit -= float(count) * penalty_freq + float(count > 0) * penalty_present;
    //
    // TWO details there are counter-intuitive.  The repeat penalty MULTIPLIES for non-positive logits and
    // DIVIDES for positive ones - the source notes the paper only ever divided, "but that would cause tokens
    // with negative logits to become more likely, which is obviously wrong".  And the presence penalty rides
    // on `float(count > 0)`, a BOOLEAN cast to float, not the count - so it applies once however many times
    // the token appeared.  Both are the kind of thing a plausible implementation gets wrong while still
    // producing valid tokens.
    //
    // It needs the token history, which is not in this signature, so a caller passing a non-zero
    // `penalty_last_n` gets a loud failure rather than a silently unpenalised sample.
    // ---- top_k: k passes of a maximum scan, skipping the already-taken
    // ---- top_p: cumulative mass over the survivors in descending order
    // The two are done together in one selection pass so the survivors and their mass are consistent.
    // GREEDY IS THE GLOBAL ARGMAX, WHATEVER THE FILTERS SAY - so it is computed directly and the filter chain
    // is skipped entirely.  No filter can remove the maximum: top_k keeps the k largest, top_p keeps a prefix
    // of the descending order that always contains the largest (min_keep >= 1), and temperature is monotonic so
    // it cannot reorder anything.  That also makes greedy INDEPENDENT OF THE ORDER, which is why the ordering
    // test cannot use it - see sampler_parity.cpp.
    // The penalty window is the last `penalty_last_n` entries of this row's history.
    const int* hrow = history ? history + (size_t) t * history_len : nullptr;
    int hlen = 0;
    if (hrow) {
        hlen = p.penalty_last_n < history_len ? p.penalty_last_n : history_len;
        if (hlen < 0) hlen = 0;
        hrow += history_len - hlen;          // the window is the TAIL
    }

    if (p.greedy || p.temperature <= 0.0f) {
        // Greedy is the global argmax ONLY when no penalty is active: a penalty can lower a token enough to
        // change the maximum, so the argmax has to be taken over the penalised logits, not the raw ones.
        //
        // *** TEMPERATURE MUST NOT BE APPLIED HERE.  THIS WAS A REAL BUG. ***
        // This branch used to read `apply_penalties(l[v] * inv_t, ...)`.  `inv_t` is 0.0f whenever
        // temperature <= 0 (see above), so at temperature 0 EVERY logit became 0.0f and the argmax returned
        // index 0 - the sampler emitted token 0 forever, whatever the model predicted.  OpenAI clients send
        // `temperature: 0` for greedy decoding, so this was reachable from any ordinary client.
        // It is also pointless even when inv_t is non-zero: scaling by a positive constant is monotonic and
        // cannot reorder the argmax, which is exactly what the note above this block already says.
        int best = 0;
        float bv = apply_penalties(l[0], hrow ? history_count(hrow, hlen, 0) : 0, p);
        for (int v = 1; v < n_vocab; ++v) {
            const float s = apply_penalties(l[v], hrow ? history_count(hrow, hlen, v) : 0, p);
            if (s > bv) { bv = s; best = v; }
        }
        out[t] = best;
        return;
    }

    const int KMAX = 64;
    int keep_ids[KMAX];
    float keep_logit[KMAX];
    int k = p.top_k > 0 ? (p.top_k < KMAX ? p.top_k : KMAX) : KMAX;
    if (p.top_k <= 0) k = 0;                      // 0 means "no top-k filter" -> handled by the mask below

    if (k <= 0) {
        std::printf("sampler: the sampled path needs top_k in 1..%d (got %d); greedy needs no filters\n", KMAX, p.top_k);
        return;   // leave out[t] unwritten rather than returning an uninitialised token
    }
    int n_keep = 0;
    {
        for (int i = 0; i < k; ++i) {
            int best = -1;
            float bv = 0.0f;
            for (int v = 0; v < n_vocab; ++v) {
                bool taken = false;
                for (int j = 0; j < i; ++j) if (keep_ids[j] == v) { taken = true; break; }
                if (taken) continue;
                if (best < 0 || l[v] > bv) { best = v; bv = l[v]; }
            }
            keep_ids[i] = best;
            keep_logit[i] = bv;
        }
        n_keep = k;

        // ---- top_p over the survivors, in descending order (which the selection above already produced)
        if (p.top_p < 1.0f) {
            // softmax over the keep set for the cumulative mass
            float mx = keep_logit[0];
            for (int i = 1; i < n_keep; ++i) mx = fmaxf(mx, keep_logit[i]);
            double sum = 0.0;
            for (int i = 0; i < n_keep; ++i) sum += exp((double) keep_logit[i] - (double) mx);
            double cum = 0.0;
            int cut = n_keep;
            for (int i = 0; i < n_keep; ++i) {
                cum += exp((double) keep_logit[i] - (double) mx) / sum;
                if (cum >= (double) p.top_p) { cut = i + 1; break; }
            }
            if (cut < p.min_keep) cut = p.min_keep < n_keep ? p.min_keep : n_keep;
            n_keep = cut;
        }
    }


    if (p.greedy || p.temperature <= 0.0f) {
        // DEAD CODE AS WRITTEN: the identical condition at the top of this function already returned, so this
        // block cannot be reached.  It is kept and corrected rather than deleted because if that early return
        // is ever narrowed, this is the path that would run - and it carried the same `* inv_t` bug (every
        // logit 0.0f at temperature 0, so `best` would always be `keep_ids[0]`).  Temperature is monotonic and
        // cannot reorder a maximum, so it must not appear here at all.
        // argmax over the SURVIVORS, ties by smallest id (the same convention as the router and top_k)
        int best = keep_ids[0];
        float bv = keep_logit[0];
        for (int i = 1; i < n_keep; ++i) {
            const float v = keep_logit[i];
            if (v > bv) { bv = v; best = keep_ids[i]; }
        }
        out[t] = best;
        return;
    }

    // ---- pick: temperature, then penalties, then softmax over the survivors, then one uniform draw.
    // The ORDER is the chain's: the survivors were chosen on the RAW logits, temperature scales them, and the
    // penalty is applied to the scaled value - which is why the `logit <= 0` branch inside the penalty sees a
    // temperature-scaled logit and not the model's own.
    for (int i = 0; i < n_keep; ++i) {
        keep_logit[i] = apply_penalties(keep_logit[i] * inv_t,
                                        hrow ? history_count(hrow, hlen, keep_ids[i]) : 0, p);
    }
    float mx = keep_logit[0];
    for (int i = 1; i < n_keep; ++i) mx = fmaxf(mx, keep_logit[i]);
    double sum = 0.0;
    for (int i = 0; i < n_keep; ++i) sum += exp((double) keep_logit[i] - (double) mx);
    const float u = philox_uniform(p.seed, p.counter + (uint64_t) t);
    double cum = 0.0;
    int pick = keep_ids[n_keep - 1];
    for (int i = 0; i < n_keep; ++i) {
        cum += exp((double) keep_logit[i] - (double) mx) / sum;
        if ((double) u < cum) { pick = keep_ids[i]; break; }
    }
    out[t] = pick;
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
    const int threads = 64;
    const unsigned grid = (unsigned) ((n_tokens + threads - 1) / threads);
    if (p.greedy || p.temperature <= 0.0f) {
        // One block per token, 1,024 threads over the vocabulary.  See `sampler_greedy_kernel`.
        const int gthreads = 1024;
        sampler_greedy_kernel<<<(unsigned) n_tokens, gthreads, 0, (cudaStream_t) stream>>>(
            logits, n_vocab, history, history_len, p, p.penalty_last_n, p.penalty_last_n, out);
    } else {
        sampler_kernel<<<grid, threads, 0, (cudaStream_t) stream>>>(logits, n_vocab, n_tokens, history,
                                                                   history_len, p, out);
    }
    const cudaError_t e = cudaGetLastError();
    if (e != cudaSuccess) {
        std::fprintf(stderr, "sample_tokens launch: %s\n", cudaGetErrorString(e));
        std::exit(1);
    }
    if (stream == nullptr) cudaDeviceSynchronize();
}

}  // namespace strata::kernels
