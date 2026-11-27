// Exact SM75 Top64: the existing radix selection on 32 tiles, followed by an ordered merge (MIT).
#include "env.hpp"
#include "kernels.hpp"
#include <cstdio>
#include <cuda_runtime.h>
#include <stdexcept>
#include <string>
namespace fnx {
namespace sm75_top64 {

// The sampler's candidates (included by kernels.cu): per verify row, this card's slice of the head's logits with the
// row's penalties applied (llama.cpp's penalties stage, as Strata applies it: repeat multiplies a non-positive logit
// and divides a positive one, then frequency x count and presence once), then its TOPK_N highest, ordered by logit
// (descending) and token id (ascending) on ties. A block per row: a radix select of the TOPK_N-th highest key, then
// everything at or above it sorted. More than TOPK_CAP at or above it (a tie at the threshold that wide: a row of equal
// logits) takes a second pass that picks the same TOPK_N, so the result never depends on how many tie. A NaN or inf
// among the logits sets err bit 16.
namespace {

constexpr int TOPK_THREADS = 256, TOPK_CAP = 256;

__device__ __forceinline__ unsigned tk_key(float v) { // order-preserving: a larger float, a larger key
    const unsigned u = __float_as_uint(v);
    return (u & 0x80000000u) ? ~u : (u | 0x80000000u);
}

__global__ void __launch_bounds__(TOPK_THREADS)
    k_topk_tiles(float *__restrict__ logits, int n, int v0, const int *__restrict__ pen_tok,
                 const int *__restrict__ pen_cnt, const int *__restrict__ pen_n, int pen_stride,
                 const float *__restrict__ pen_par, float *__restrict__ out_v, int *__restrict__ out_i, unsigned *err) {
    const int row = blockIdx.y, tid = threadIdx.x, lane = tid & 31;
    const int full_n = n, start = (int)blockIdx.x * full_n / (int)gridDim.x;
    n = ((int)blockIdx.x + 1) * full_n / (int)gridDim.x - start;
    float *L = logits + (size_t)row * full_n + start;
    v0 += start;
    // a NaN or inf among the head's logits (before the penalties, which a request sets): err bit 16
    bool bad = false;
    for (int j = tid; j < n; j += TOPK_THREADS)
        bad = bad || !isfinite(L[j]);
    if (bad)
        atomicOr(err, 16u);
    __syncthreads(); // every logit checked before any penalty changes one
    // the row's penalties on this card's tokens (each token once in the list)
    const float repeat = pen_par[row * 3], freq = pen_par[row * 3 + 1],
                present = pen_par[row * 3 + 2]; // the row's penalty parameters
    for (int i = tid; i < pen_n[row]; i += TOPK_THREADS) {
        const int tok = pen_tok[row * pen_stride + i] - v0, cnt = pen_cnt[row * pen_stride + i];
        if (tok < 0 || tok >= n || cnt <= 0)
            continue;
        float l = L[tok];
        l = l <= 0.f ? l * repeat : l / repeat;
        L[tok] = l - (float)cnt * freq - present;
    }
    __syncthreads();
    // the TOPK_N-th highest key: four passes of 8 bits
    __shared__ unsigned hist[256];
    __shared__ unsigned prefix_s, want_s;
    unsigned prefix = 0, mask = 0, want = TOPK_N;
    for (int shift = 24; shift >= 0; shift -= 8) {
        for (int b = tid; b < 256; b += TOPK_THREADS)
            hist[b] = 0;
        __syncthreads();
        for (int jb = 0; jb < n; jb += TOPK_THREADS) {
            const int j = jb + tid;
            const unsigned k = j < n ? tk_key(L[j]) : 0u;
            const bool in = j < n && (k & mask) == prefix;
            const unsigned bin = in ? (k >> shift) & 255u : 256u;
            const unsigned peers = __match_any_sync(0xffffffffu, bin);
            if (in && lane == __ffs(peers) - 1)
                atomicAdd(&hist[bin], (unsigned)__popc(peers));
        }
        __syncthreads();
        if (tid == 0) {
            unsigned acc = 0;
            int b = 255;
            for (; b > 0; --b) {
                if (acc + hist[b] >= want)
                    break;
                acc += hist[b];
            }
            prefix_s = prefix | ((unsigned)b << shift);
            want_s = want - acc;
        }
        __syncthreads();
        prefix = prefix_s;
        want = want_s;
        mask |= 255u << shift;
        __syncthreads();
    }
    // everything at or above the threshold, then sorted: (key desc, id asc) as one 64-bit key, descending
    __shared__ unsigned long long cand[TOPK_CAP];
    __shared__ unsigned n_cand;
    for (int i = tid; i < TOPK_CAP; i += TOPK_THREADS)
        cand[i] = 0ull;
    if (tid == 0)
        n_cand = 0;
    __syncthreads();
    for (int j = tid; j < n; j += TOPK_THREADS) {
        const unsigned k = tk_key(L[j]);
        if (k >= prefix) {
            const unsigned s = atomicAdd(&n_cand, 1u);
            if (s < TOPK_CAP)
                cand[s] = ((unsigned long long)k << 32) | (0xffffffffu - (unsigned)j);
        }
    }
    __syncthreads();
    const bool wide = n_cand > TOPK_CAP;
    __syncthreads(); // every thread has read n_cand before the wide pass resets it
    if (wide) {      // block-uniform. The first TOPK_N of the sorted set are every key above the threshold
        // (TOPK_N - want of them) and the `want` lowest ids at it: those exactly, the ties taken in id order a chunk at
        // a time (a block-wide rank from each warp's ballot), stopping once `want` are in
        __shared__ unsigned warp_n[TOPK_THREADS / 32];
        const int warp = tid >> 5;
        for (int i = tid; i < TOPK_CAP; i += TOPK_THREADS)
            cand[i] = 0ull;
        if (tid == 0)
            n_cand = 0;
        __syncthreads();
        for (int j = tid; j < n; j += TOPK_THREADS) {
            const unsigned k = tk_key(L[j]);
            if (k > prefix)
                cand[atomicAdd(&n_cand, 1u)] = ((unsigned long long)k << 32) | (0xffffffffu - (unsigned)j);
        }
        const unsigned base = TOPK_N - want;
        unsigned taken = 0;
        for (int jb = 0; jb < n && taken < want; jb += TOPK_THREADS) {
            const int j = jb + tid;
            const bool eq = j < n && tk_key(L[j]) == prefix;
            const unsigned bal = __ballot_sync(0xffffffffu, eq);
            if (lane == 0)
                warp_n[warp] = (unsigned)__popc(bal);
            __syncthreads();
            unsigned before = (unsigned)__popc(bal & ((1u << lane) - 1u)), total = 0;
            for (int w = 0; w < TOPK_THREADS / 32; ++w) {
                before += w < warp ? warp_n[w] : 0u;
                total += warp_n[w];
            }
            if (eq && taken + before < want)
                cand[base + taken + before] = ((unsigned long long)prefix << 32) | (0xffffffffu - (unsigned)j);
            taken += total;
            __syncthreads(); // warp_n read by every thread before the next chunk writes it
        }
        __syncthreads();
    }
    for (int size = 2; size <= TOPK_CAP; size <<= 1)
        for (int stride = size >> 1; stride > 0; stride >>= 1) {
            if (tid < TOPK_CAP) {
                const int o = tid ^ stride;
                if (o > tid) {
                    const bool desc = (tid & size) == 0;
                    const unsigned long long a = cand[tid], b = cand[o];
                    if (desc ? a < b : a > b) {
                        cand[tid] = b;
                        cand[o] = a;
                    }
                }
            }
            __syncthreads();
        }
    if (tid < TOPK_N) {
        const unsigned long long c = cand[tid];
        const int j = (int)(0xffffffffu - (unsigned)(c & 0xffffffffu));
        out_i[(row * gridDim.x + blockIdx.x) * TOPK_N + tid] = v0 + j;
        out_v[(row * gridDim.x + blockIdx.x) * TOPK_N + tid] = L[j];
    }
}

} // namespace

__device__ __forceinline__ unsigned ordered_key(float value) {
    const unsigned bits = __float_as_uint(value);
    return bits & 0x80000000u ? ~bits : bits | 0x80000000u;
}
__global__ void merge_sorted_tiles(const float *values, const int *indices, float *out_values, int *out_indices,
                                   int parts) {
    const int row = blockIdx.x, lane = threadIdx.x;
    const int base = (row * parts + lane) * TOPK_N;
    int position = 0;
    unsigned long long head =
        lane < parts ? ((unsigned long long)ordered_key(values[base]) << 32) | (0xffffffffu - (unsigned)indices[base])
                     : 0ull;
    // Each tile is already sorted. Only the winning list advances at each exact global pick.
#pragma unroll 1
    for (int pick = 0; pick < TOPK_N; ++pick) {
        unsigned long long best = head;
#pragma unroll
        for (int offset = 16; offset; offset >>= 1)
            best = max(best, __shfl_xor_sync(0xffffffffu, best, offset));
        if (lane == (pick & 31)) {
            const unsigned key = best >> 32, bits = key & 0x80000000u ? key & 0x7fffffffu : ~key;
            out_values[row * TOPK_N + pick] = __uint_as_float(bits);
            out_indices[row * TOPK_N + pick] = (int)(0xffffffffu - (unsigned)best);
        }
        if (head == best) {
            ++position;
            head = position < TOPK_N ? ((unsigned long long)ordered_key(values[base + position]) << 32) |
                                           (0xffffffffu - (unsigned)indices[base + position])
                                     : 0ull;
        }
    }
}

} // namespace sm75_top64

// CMake renames only the existing CUDA kernel translation unit's entry point.
// Shared engine sources and all non-SM75 builds keep their original entry point.
void top_tokens_reference_sm75(float *, int, int, int, const int *, const int *, const int *, int, const float *,
                               float *, int *, unsigned *, void *);

namespace {
void check_top64_cuda(cudaError_t status, const char *operation) {
    if (status != cudaSuccess)
        throw std::runtime_error(std::string(operation) + ": " + cudaGetErrorString(status));
}
} // namespace

void top_tokens_parallel_sm75(float *logits, int n, int rows, int first_token, const int *penalty_tokens,
                              const int *penalty_counts, const int *penalty_sizes, int penalty_stride,
                              const float *penalties, float *values, int *indices, unsigned *error, void *stream) {
    // The 32-tile path wins on large head slices. Smaller slices retain the
    // reference launch: allocation and merge overhead outweighed their scan.
    if (n < 65536 || rows < 1) {
        top_tokens_reference_sm75(logits, n, rows, first_token, penalty_tokens, penalty_counts, penalty_sizes,
                                  penalty_stride, penalties, values, indices, error, stream);
        return;
    }
    const cudaStream_t st = static_cast<cudaStream_t>(stream);
    const size_t partials = static_cast<size_t>(rows) * 32 * TOPK_N;
    void *workspace = nullptr;
    // A matched allocate/free pair is captured as graph memory nodes. Each call
    // owns its scratch lifetime, including distinct lanes and concurrent streams.
    check_top64_cuda(cudaMallocAsync(&workspace, partials * (sizeof(float) + sizeof(int)), st), "Top64 scratch");
    float *partial_values = static_cast<float *>(workspace);
    int *partial_indices = reinterpret_cast<int *>(partial_values + partials);
    sm75_top64::k_topk_tiles<<<dim3(32, rows), 256, 0, st>>>(logits, n, first_token, penalty_tokens, penalty_counts,
                                                             penalty_sizes, penalty_stride, penalties, partial_values,
                                                             partial_indices, error);
    sm75_top64::merge_sorted_tiles<<<rows, 32, 0, st>>>(partial_values, partial_indices, values, indices, 32);
    const cudaError_t launch_status = cudaGetLastError();
    const cudaError_t free_status = cudaFreeAsync(workspace, st);
    check_top64_cuda(launch_status, "Top64 kernels");
    check_top64_cuda(free_status, "Top64 scratch release");
}

void top_tokens(float *logits, int n, int rows, int first_token, const int *penalty_tokens, const int *penalty_counts,
                const int *penalty_sizes, int penalty_stride, const float *penalties, float *values, int *indices,
                unsigned *error, void *stream) {
    static const bool enabled = env_int("FNX_SM75_TOP64", 0) != 0;
    if (enabled) {
        top_tokens_parallel_sm75(logits, n, rows, first_token, penalty_tokens, penalty_counts, penalty_sizes,
                                 penalty_stride, penalties, values, indices, error, stream);
    } else {
        top_tokens_reference_sm75(logits, n, rows, first_token, penalty_tokens, penalty_counts, penalty_sizes,
                                  penalty_stride, penalties, values, indices, error, stream);
    }
}
} // namespace fnx
