// src/dsv4/dsv4_comp.cu - DeepSeek V4-Flash compressor block + compress-layer attention (dsv4-mvp S4),
// plus the S5 indexer + sparse attention of the r=4 layers (dsv4ref indexer_forward :895, the
// attention_forward :967 ratio==4 arm): indexer q from the SHARED qr (indexer.attn_q_b Q8_0, rope at the
// token position, no renorm), the indexer's own compressor (indexer_compressor_*, ratio 4 / d 128 /
// coff 2) writing fp32 keys into the layer's indexer cache slice, weights indexer.proj (RAW BF16,
// errata #9) scaled by 128^-0.5 * 64^-0.5, score = sum_h relu(q.k) * wgt, and the EXACT top-512
// selection on the host (fp32, stable sort, ties ascending - integration plan section 6.2), then the
// same [window | indexer topk] MQA-512 tail as dsv4_comp_attn_forward.
//
// Semantics follow src/dsv4/dsv4ref.cpp line by line:
//   - compressor_forward (:623): BF16 kv/gate GEMMs (dequantized in-kernel, double-accumulated),
//     per-feature softmax pooling in double over the pooling rows of (score + ape[(tok%ratio)]),
//     overlap (coff=2, r=4) rows [i<r: tok=j*r+i-r, dims 0:d] / [i>=r: same tok, dims d:2d] with
//     tok<0 skipped, then rmsnorm(norm) + rope at phase j*ratio (YaRN compress table) and the cache
//     slot write (slot j prefill / start_pos/ratio decode).
//   - attention_forward (:967) ratio>0 arm: S3's q/kv front half, compressor, kv_all =
//     [kv_new | kv_comp] (prefill), combined [window | compress] topk (offset s prefill / win decode,
//     causal t >= (j+1)/ratio masked), MQA-512, inverse rope, grouped O projection.
//   - dsv4_attn_half: the run_pass (:1229) attention arm per layer class - hc_pre/hc_post are the
//     same numerics as dsv4_moe.cu's ffn-side kernels (kept as independent copies here so the S2
//     translation unit stays untouched).
//
// The shared S3 kernels (Q8_0 GEMM / rmsnorm / rope / MQA-512) are reused through attn.hpp's S4
// launcher section, not copied.
//
// Numerical contract: L03 (r=128) post_attn_main vs the golden oracle cos >= 0.9999 / rel_amax <= 3e-2,
// hc streams vs dsv4ref with the same replay input cos >= 0.99999 (integration plan errata #6).

#include "strata/dsv4/comp.hpp"

#include "dsv4_attn_time.h"

#include <chrono>
#include <cuda_bf16.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <vector>

#include "strata/dsv4/geometry.hpp"

namespace strata::dsv4 {
namespace {

#define DSV4_CK(x)                                                            \
    do {                                                                      \
        cudaError_t err_ = (x);                                               \
        if (err_ != cudaSuccess) {                                            \
            std::fprintf(stderr, "dsv4_comp: CUDA error %s at %s:%d\n",       \
                         cudaGetErrorString(err_), __FILE__, __LINE__);       \
            std::abort();                                                     \
        }                                                                     \
    } while (0)

constexpr int kNEmbd = 4096;
constexpr int kHeadDim = 512;
constexpr int kNHead = 64;
constexpr double kRmsEps = 1e-6;

__device__ __forceinline__ float comp_sigmoid(float v) { return 1.0f / (1.0f + expf(-v)); }

// [M2] write a pass's s KV rows into the ring (start_pos > 0), the batched twin of the per-row copies
// of the default path: s <= win, so at most one wrap and two copies cover it.
void ring_write_rows(float* kv_cache, const float* kv_new, int win, int64_t start_pos, int s,
                     cudaStream_t stream) {
    const int sp0 = (int) (start_pos % win);
    const int first = (int) std::min<int64_t>(s, win - sp0);
    DSV4_CK(cudaMemcpyAsync(kv_cache + (size_t) sp0 * kHeadDim, kv_new,
                            (size_t) first * kHeadDim * 4, cudaMemcpyDeviceToDevice, stream));
    if (s > first)
        DSV4_CK(cudaMemcpyAsync(kv_cache, kv_new + (size_t) first * kHeadDim,
                                (size_t) (s - first) * kHeadDim * 4, cudaMemcpyDeviceToDevice,
                                stream));
}

// ---------------------------------------------------------------------------
// Compressor state init: kv_state = 0, score_state = -inf (dsv4ref CompState::init).
// ---------------------------------------------------------------------------
__global__ void dsv4_comp_init_kernel(float* kv_state, float* score_state, int64_t n) {
    const int64_t i = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    kv_state[i] = 0.0f;
    score_state[i] = -INFINITY;
}

// ---------------------------------------------------------------------------
// BF16 GEMM: out[t][r] = dot(x[t], bf16(W_row r)), double accumulation (the compressor is FP32
// semantics in the oracle; the double reduction keeps the GPU at least as accurate as the host
// reference's float dot).  One block per output row, 128 threads, token loop.
// ---------------------------------------------------------------------------
__global__ void dsv4_gemm_bf16_kernel(const uint16_t* __restrict__ w, int rows, int cols,
                                      const float* __restrict__ x, int m, float* __restrict__ out) {
    const int r = blockIdx.x;
    if (r >= rows) return;
    const uint16_t* wrow = w + (size_t) r * (size_t) cols;
    __shared__ double red[128];
    for (int t = 0; t < m; ++t) {
        const float* xt = x + (size_t) t * (size_t) cols;
        double acc = 0.0;
        // [S7b] 8-deep load batching: the fp64 accumulate chain serializes the FMAs but the
        // loads are independent; batching keeps 16 loads in flight per thread (MLP) while the
        // accumulation order (and hence every intermediate double value) is unchanged.
        int i = threadIdx.x;
        for (; i + 7 * 128 < cols; i += 8 * 128) {
            float xv[8];
            uint16_t wv[8];
#pragma unroll
            for (int k = 0; k < 8; ++k) {
                xv[k] = xt[i + k * 128];
                wv[k] = wrow[i + k * 128];
            }
#pragma unroll
            for (int k = 0; k < 8; ++k)
                acc += (double) xv[k] * (double) __bfloat162float(__ushort_as_bfloat16(wv[k]));
        }
        for (; i < cols; i += 128)
            acc += (double) xt[i] * (double) __bfloat162float(__ushort_as_bfloat16(wrow[i]));
        red[threadIdx.x] = acc;
        __syncthreads();
        for (int off = 64; off > 0; off >>= 1) {
            if (threadIdx.x < off) red[threadIdx.x] += red[threadIdx.x + off];
            __syncthreads();
        }
        if (threadIdx.x == 0) out[(size_t) t * rows + r] = (float) red[0];
        __syncthreads();
    }
}

// ---------------------------------------------------------------------------
// [M3] Row-tiled variant of the BF16 GEMM above.  blockIdx.y picks a tile of output rows, so the same
// (row t, output r) dot product is computed by the same 128-thread k-loop and the same fp64 tree
// reduction -- only the number of resident blocks changes, never an accumulation order.  Used as the
// fallback when the output count is not a multiple of the coarsening factor of dsv4_gemm_bf16_coars_kernel;
// measured on a 1024-row projection it is 4.19 ms against 4.08 ms for the coarsened kernel.
// ---------------------------------------------------------------------------
template <int kRowsPerBlock>
__global__ void dsv4_gemm_bf16_split_kernel(const uint16_t* __restrict__ w, int rows, int cols,
                                           const float* __restrict__ x, int m,
                                           float* __restrict__ out) {
    const int r = blockIdx.x;
    if (r >= rows) return;
    const int t0 = (int) blockIdx.y * kRowsPerBlock;
    const int t1 = t0 + kRowsPerBlock < m ? t0 + kRowsPerBlock : m;
    const uint16_t* wrow = w + (size_t) r * (size_t) cols;
    __shared__ double red[128];
    for (int t = t0; t < t1; ++t) {
        const float* xt = x + (size_t) t * (size_t) cols;
        double acc = 0.0;
        // same 8-deep load batching / same k order as dsv4_gemm_bf16_kernel
        int i = threadIdx.x;
        for (; i + 7 * 128 < cols; i += 8 * 128) {
            float xv[8];
            uint16_t wv[8];
#pragma unroll
            for (int k = 0; k < 8; ++k) {
                xv[k] = xt[i + k * 128];
                wv[k] = wrow[i + k * 128];
            }
#pragma unroll
            for (int k = 0; k < 8; ++k)
                acc += (double) xv[k] * (double) __bfloat162float(__ushort_as_bfloat16(wv[k]));
        }
        for (; i < cols; i += 128)
            acc += (double) xt[i] * (double) __bfloat162float(__ushort_as_bfloat16(wrow[i]));
        red[threadIdx.x] = acc;
        __syncthreads();
        for (int off = 64; off > 0; off >>= 1) {
            if (threadIdx.x < off) red[threadIdx.x] += red[threadIdx.x + off];
            __syncthreads();
        }
        if (threadIdx.x == 0) out[(size_t) t * rows + r] = (float) red[0];
        __syncthreads();
    }
}


// ---------------------------------------------------------------------------
// [M3] Output-coarsened BF16 GEMM: one thread accumulates kOuts independent output rows while it walks
// k, so the dependent fp64 chain of a single dot product no longer serialises the fp64 pipe.  Every
// (row t, output r) dot product still runs the same ascending-k loop and the same fp64 tree reduction,
// so every intermediate double is identical; only the interleaving of *independent* chains changes.
// ---------------------------------------------------------------------------
template <int kOuts, int kRowsPerBlock>
__global__ void dsv4_gemm_bf16_coars_kernel(const uint16_t* __restrict__ w, int rows, int cols,
                                            const float* __restrict__ x, int m,
                                            float* __restrict__ out) {
    const int r0 = (int) blockIdx.x * kOuts;
    const int t0 = (int) blockIdx.y * kRowsPerBlock;
    const int t1 = t0 + kRowsPerBlock < m ? t0 + kRowsPerBlock : m;
    const uint16_t* wr[kOuts];
#pragma unroll
    for (int o = 0; o < kOuts; ++o) {
        const int r = r0 + o;
        wr[o] = r < rows ? w + (size_t) r * (size_t) cols : nullptr;
    }
    __shared__ double red[kOuts][128];
    for (int t = t0; t < t1; ++t) {
        const float* xt = x + (size_t) t * (size_t) cols;
        double acc[kOuts];
#pragma unroll
        for (int o = 0; o < kOuts; ++o) acc[o] = 0.0;
        int i = threadIdx.x;
        for (; i + 7 * 128 < cols; i += 8 * 128) {
            float xv[8];
#pragma unroll
            for (int k = 0; k < 8; ++k) xv[k] = xt[i + k * 128];
#pragma unroll
            for (int o = 0; o < kOuts; ++o) {
                if (wr[o] == nullptr) continue;
#pragma unroll
                for (int k = 0; k < 8; ++k)
                    acc[o] += (double) xv[k] *
                              (double) __bfloat162float(__ushort_as_bfloat16(wr[o][i + k * 128]));
            }
        }
        for (; i < cols; i += 128) {
            const float xv = xt[i];
#pragma unroll
            for (int o = 0; o < kOuts; ++o) {
                if (wr[o] == nullptr) continue;
                acc[o] += (double) xv *
                          (double) __bfloat162float(__ushort_as_bfloat16(wr[o][i]));
            }
        }
#pragma unroll
        for (int o = 0; o < kOuts; ++o) red[o][threadIdx.x] = acc[o];
        __syncthreads();
        for (int off = 64; off > 0; off >>= 1) {
            if (threadIdx.x < off) {
#pragma unroll
                for (int o = 0; o < kOuts; ++o) red[o][threadIdx.x] += red[o][threadIdx.x + off];
            }
            __syncthreads();
        }
        if (threadIdx.x == 0) {
#pragma unroll
            for (int o = 0; o < kOuts; ++o) {
                const int r = r0 + o;
                if (r < rows) out[(size_t) t * rows + r] = (float) red[o][0];
            }
        }
        __syncthreads();
    }
}

// [M3] compressor projections: tiles are fixed at 8 rows x 4 outputs per block.  Measured on a 4k
// chunked prefill: rows-per-block 4/8/16 differ by <0.3 % (the fp64 accumulate chain, not occupancy,
// is the limit), and 4 outputs per block is ~2 % faster than the one-output kernel (4.17 -> 4.08 ms
// per 1024-row projection, i.e. 237 GMAC/s, essentially the fp64 FMA rate of this GPU).  Going faster
// would need a different accumulation order, which the bit-exactness contract forbids.
static void dsv4_bf16_gemm(const uint16_t* w, int rows, int cols, const float* x, int64_t m,
                           float* out, cudaStream_t stream) {
    if (!dsv4_attn_batch_enabled()) {
        dsv4_gemm_bf16_kernel<<<(unsigned) rows, 128, 0, stream>>>(w, rows, cols, x, (int) m, out);
        return;
    }
    const int rpb = 8;
    const int outs = 4;
    const unsigned gy = (unsigned) (((int) m + rpb - 1) / rpb);
    if ((rows % outs) == 0) {
        const dim3 grid((unsigned) (rows / outs), gy, 1);
        dsv4_gemm_bf16_coars_kernel<4, 8><<<grid, 128, 0, stream>>>(w, rows, cols, x, (int) m, out);
    } else {
        const dim3 grid((unsigned) rows, gy, 1);
        dsv4_gemm_bf16_split_kernel<8><<<grid, 128, 0, stream>>>(w, rows, cols, x, (int) m, out);
    }
}

// ---------------------------------------------------------------------------
// Prefill state store: dst[idx] = src[idx] + ape[idx] over a [rows][W] span.  Both prefill segments
// (the overlap tail block and the remainder block) index ape by (tok % ratio), which is the local row
// inside each segment (cutoff is ratio-aligned), so one elementwise kernel covers both.
// ---------------------------------------------------------------------------
__global__ void dsv4_comp_ape_store_kernel(float* __restrict__ dst, const float* __restrict__ src,
                                           const float* __restrict__ ape, int64_t n) {
    const int64_t i = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) dst[i] = src[i] + ape[i];
}

// ---------------------------------------------------------------------------
// Prefill pooling (dsv4ref :661-729): one thread per (compressed token j, feature dd).  rows = ratio
// (plain) or 2*ratio (overlap); overlap maps i<ratio -> tok=j*ratio+i-ratio, off=dd and i>=ratio ->
// tok=j*ratio+(i-ratio), off=d+dd, skipping tok<0.  Three double loops in the reference's order:
// max, denom, then acc += (exp(v-mx)/denom) * kv.
// ---------------------------------------------------------------------------
__global__ void dsv4_comp_pool_kernel(const float* __restrict__ kv, const float* __restrict__ score,
                                      const float* __restrict__ ape, int64_t ncomp, int64_t d,
                                      int64_t ratio, int coff, float* __restrict__ kv_c) {
    const int64_t idx = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= ncomp * d) return;
    const int64_t j = idx / d;
    const int64_t dd = idx % d;
    const int64_t W = coff * d;
    const int64_t rows = coff * ratio;
    const bool overlap = coff == 2;
    auto map = [&](int64_t i, int64_t& tok, int64_t& off) {
        if (overlap) {
            if (i < ratio) {
                tok = j * ratio + i - ratio;
                off = dd;
            } else {
                tok = j * ratio + (i - ratio);
                off = d + dd;
            }
        } else {
            tok = j * ratio + i;
            off = dd;
        }
    };
    double mx = -1.0 / 0.0;
    for (int64_t i = 0; i < rows; ++i) {
        int64_t tok, off;
        map(i, tok, off);
        if (overlap && tok < 0) continue;
        const double v = (double) score[(size_t) (tok * W + off)] +
                         (double) ape[(size_t) ((tok % ratio) * W + off)];
        if (v > mx) mx = v;
    }
    double denom = 0.0;
    for (int64_t i = 0; i < rows; ++i) {
        int64_t tok, off;
        map(i, tok, off);
        if (overlap && tok < 0) continue;
        denom += exp((double) score[(size_t) (tok * W + off)] +
                     (double) ape[(size_t) ((tok % ratio) * W + off)] - mx);
    }
    double acc = 0.0;
    for (int64_t i = 0; i < rows; ++i) {
        int64_t tok, off;
        map(i, tok, off);
        if (overlap && tok < 0) continue;
        const double p = exp((double) score[(size_t) (tok * W + off)] +
                             (double) ape[(size_t) ((tok % ratio) * W + off)] - mx) /
                         denom;
        acc += p * (double) kv[(size_t) (tok * W + off)];
    }
    kv_c[(size_t) idx] = (float) acc;
}

// ---------------------------------------------------------------------------
// Decode pooling over the carry state (dsv4ref :742-788): one thread per feature dd.  overlap pools
// [i<ratio: state row i, dim dd] and [i>=ratio: state row i, dim d+dd]; plain pools rows 0..ratio-1.
// Unnormalized accumulation, acc/denom at the end (the reference's decode order).
// ---------------------------------------------------------------------------
__global__ void dsv4_comp_pool_decode_kernel(const float* __restrict__ kv_state,
                                             const float* __restrict__ score_state, int64_t d,
                                             int64_t ratio, int coff, float* __restrict__ kv_c) {
    const int64_t dd = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (dd >= d) return;
    const int64_t W = coff * d;
    const int64_t rows = coff * ratio;
    const bool overlap = coff == 2;
    double mx = -1.0 / 0.0;
    for (int64_t i = 0; i < rows; ++i) {
        const double v = (double) score_state[(size_t) (i * W + ((!overlap || i < ratio) ? dd : d + dd))];
        if (v > mx) mx = v;
    }
    double denom = 0.0, acc = 0.0;
    for (int64_t i = 0; i < rows; ++i) {
        const int64_t off = (!overlap || i < ratio) ? dd : d + dd;
        const double p = exp((double) score_state[(size_t) (i * W + off)] - mx);
        denom += p;
        acc += p * (double) kv_state[(size_t) (i * W + off)];
    }
    kv_c[(size_t) dd] = (float) (acc / denom);
}

// ---------------------------------------------------------------------------
// Chunked-prefill pooling over the carry state: a pass boundary split the ratio block, so this is the
// prefill arithmetic (dsv4_comp_pool_kernel: v = double(score) + double(ape), p = exp(v-mx)/denom,
// acc += p*kv) applied to the state rows instead of the decode order (fp32 ape add, trailing divide).
// Row mapping is the carry state's, not the prefill buffer's: for overlap, rows [0, ratio) are the
// previous block (dims 0..d) and rows [ratio, 2*ratio) the current block (dims d..2d); row i belongs
// to ape row i % ratio either way, because block starts are multiples of ratio.
// ---------------------------------------------------------------------------
__global__ void dsv4_comp_pool_chunk_kernel(const float* __restrict__ kv_state,
                                            const float* __restrict__ raw_state,
                                            const float* __restrict__ ape, int64_t d, int64_t ratio,
                                            int coff, float* __restrict__ kv_c) {
    const int64_t dd = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (dd >= d) return;
    const int64_t W = coff * d;
    const int64_t rows = coff * ratio;
    const bool overlap = coff == 2;
    auto v_of = [&](int64_t i) {
        const int64_t off = (!overlap || i < ratio) ? dd : d + dd;
        return (double) raw_state[(size_t) (i * W + off)] +
               (double) ape[(size_t) ((i % ratio) * W + off)];
    };
    double mx = -1.0 / 0.0;
    for (int64_t i = 0; i < rows; ++i) {
        const double v = v_of(i);
        if (v > mx) mx = v;
    }
    double denom = 0.0;
    for (int64_t i = 0; i < rows; ++i) denom += exp(v_of(i) - mx);
    double acc = 0.0;
    for (int64_t i = 0; i < rows; ++i) {   // the prefill order: divide first, then multiply
        const int64_t off = (!overlap || i < ratio) ? dd : d + dd;
        const double p = exp(v_of(i) - mx) / denom;
        acc += p * (double) kv_state[(size_t) (i * W + off)];
    }
    kv_c[(size_t) dd] = (float) acc;
}

// ---------------------------------------------------------------------------
// Decode: score[0][c] += ape[rem*W + c] before the state store (dsv4ref :735-736).
// ---------------------------------------------------------------------------
__global__ void dsv4_comp_ape_add_row_kernel(float* __restrict__ score,
                                             const float* __restrict__ ape, int64_t W) {
    const int64_t c = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (c < W) score[c] += ape[c];
}

// ---------------------------------------------------------------------------
// hc_pre / hc_post (the attention half's fn/base/scale) - the same numerics as dsv4_moe.cu's ffn-side
// kernels (dsv4ref.cpp:447-470 / :488-498), kept as an independent copy so the S2 unit stays untouched.
// One block per token, 256 threads; sq and the 24 mix dots reduce in double; sinkhorn(20) on thread 0
// in the reference's exact iteration order.
// ---------------------------------------------------------------------------
__global__ void dsv4_comp_hc_mix_kernel(const float* __restrict__ x, const float* __restrict__ fn,
                                        int64_t n, double rms_eps, float* __restrict__ mixes) {
    // [S7b] one block per (token, mix dot); per-dot accumulation order identical to the old
    // single-block kernel (thread-strided float partial -> double shared tree 128..1).
    constexpr int kThreads = 256;
    const int64_t n4 = 4 * n;
    const float* xt = x + (int64_t) blockIdx.x * n4;
    const int r = blockIdx.y;
    __shared__ double red[kThreads];
    const int tid = threadIdx.x;
    float sq = 0.f;
    for (int64_t i = tid; i < n4; i += kThreads) sq += xt[i] * xt[i];
    red[tid] = (double) sq;
    __syncthreads();
    for (int off = kThreads / 2; off > 0; off >>= 1) {
        if (tid < off) red[tid] += red[tid + off];
        __syncthreads();
    }
    const float rsqrt = (float) (1.0 / sqrt(red[0] / (double) n4 + rms_eps));
    const float* f = fn + (int64_t) r * n4;
    float acc = 0.f;
    for (int64_t i = tid; i < n4; i += kThreads) acc += f[i] * xt[i];
    red[tid] = (double) acc;
    __syncthreads();
    for (int off = kThreads / 2; off > 0; off >>= 1) {
        if (tid < off) red[tid] += red[tid + off];
        __syncthreads();
    }
    if (tid == 0) mixes[(int64_t) blockIdx.x * 24 + r] = (float) red[0] * rsqrt;
}

__global__ void dsv4_comp_hc_fin_kernel(const float* __restrict__ x, const float* __restrict__ base,
                                        const float* __restrict__ scale,
                                        const float* __restrict__ mixes, int64_t n, double hc_eps,
                                        float* __restrict__ y, float* __restrict__ post,
                                        float* __restrict__ comb) {
    // [S7b] serial sigmoid/sinkhorn on thread 0 + per-element float combine; bit-identical.
    constexpr int kThreads = 256;
    const int64_t n4 = 4 * n;
    const float* xt = x + (int64_t) blockIdx.x * n4;
    const float* mx = mixes + (int64_t) blockIdx.x * 24;
    __shared__ float sh_pre[4];
    const int tid = threadIdx.x;
    __shared__ float sh_cb[16];
    __shared__ double sh_sum[4];
    const float eps = (float) hc_eps;
    if (tid < 4) {
        // sigmoid gates: each channel's math unchanged, one channel per thread
        sh_pre[tid] = comp_sigmoid(mx[tid] * scale[0] + base[tid]) + eps;
        post[(int64_t) blockIdx.x * 4 + tid] = 2.0f * comp_sigmoid(mx[4 + tid] * scale[1] + base[4 + tid]);
    }
    if (tid < 16) sh_cb[tid] = mx[8 + tid] * scale[2] + base[8 + tid];
    __syncthreads();
    // hc_sinkhorn20 (dsv4ref.cpp:415): row softmax + eps, col-norm, 19 x (row-norm, col-norm)
    if (tid < 4) {
        // row softmax: one row per thread; max/sum keep the serial j order
        float mxv = sh_cb[tid * 4];
        for (int j = 1; j < 4; ++j) mxv = fmaxf(mxv, sh_cb[tid * 4 + j]);
        for (int j = 0; j < 4; ++j) sh_cb[tid * 4 + j] = expf(sh_cb[tid * 4 + j] - mxv);
    }
    __syncthreads();
    if (tid < 4) {
        double sum = 0;
        for (int j = 0; j < 4; ++j) sum += sh_cb[tid * 4 + j];
        sh_sum[tid] = sum;
    }
    __syncthreads();
    if (tid < 16) sh_cb[tid] = (float) (sh_cb[tid] / sh_sum[tid >> 2] + (double) eps);
    __syncthreads();
    for (int iter = 0; iter < 20; ++iter) {
        if (tid < 4) {  // col-norm: column tid, serial c order
            double s = 0;
            for (int c = 0; c < 4; ++c) s += sh_cb[c * 4 + tid];
            sh_sum[tid] = s;
        }
        __syncthreads();
        if (tid < 16) sh_cb[tid] = (float) (sh_cb[tid] / (sh_sum[tid & 3] + (double) eps));
        __syncthreads();
        if (iter == 19) break;
        if (tid < 4) {  // row-norm: row tid, serial j order
            double s = 0;
            for (int j = 0; j < 4; ++j) s += sh_cb[tid * 4 + j];
            sh_sum[tid] = s;
        }
        __syncthreads();
        if (tid < 16) sh_cb[tid] = (float) (sh_cb[tid] / (sh_sum[tid >> 2] + (double) eps));
        __syncthreads();
    }
    if (tid < 16) comb[(int64_t) blockIdx.x * 16 + tid] = sh_cb[tid];
    __syncthreads();
    float* yt = y + (int64_t) blockIdx.x * n;
    for (int64_t i = tid; i < n; i += kThreads) {
        float acc = 0.f;
        for (int c = 0; c < 4; ++c) acc += sh_pre[c] * xt[(int64_t) c * n + i];
        yt[i] = acc;
    }
}

__global__ void dsv4_comp_hc_post_kernel(const float* __restrict__ h, const float* __restrict__ bout,
                                         const float* __restrict__ post,
                                         const float* __restrict__ comb, int64_t n,
                                         float* __restrict__ out) {
    const int64_t b = blockIdx.x;
    const float* hb = h + b * 4 * n;
    const float* xb = bout + b * n;
    const float* pb = post + b * 4;
    const float* cb = comb + b * 16;
    float* ob = out + b * 4 * n;
    for (int c = 0; c < 4; ++c)
        for (int64_t i = threadIdx.x; i < n; i += blockDim.x) {
            float acc = pb[c] * xb[i];
            for (int j = 0; j < 4; ++j) acc += cb[c * 4 + j] * hb[(int64_t) j * n + i];
            ob[(int64_t) c * n + i] = acc;
        }
}

}  // namespace

// ============================================================================================= state

bool dsv4_comp_state_init(Dsv4CompState& st, int64_t ratio, int64_t d, int64_t coff,
                          cudaStream_t stream) {
    st.ratio = ratio;
    st.d = d;
    st.coff = coff;
    const int64_t n = coff * ratio * coff * d;   // [coff*ratio][coff*d]
    if (cudaMalloc(&st.kv_state, (size_t) n * 4) != cudaSuccess ||
        cudaMalloc(&st.score_state, (size_t) n * 4) != cudaSuccess ||
        cudaMalloc(&st.raw_state, (size_t) n * 4) != cudaSuccess) {
        std::fprintf(stderr, "dsv4_comp: state allocation failed\n");
        dsv4_comp_state_free(st);
        return false;
    }
    const int64_t threads = 256, blocks = (n + threads - 1) / threads;
    dsv4_comp_init_kernel<<<(unsigned) blocks, (unsigned) threads, 0, stream>>>(st.kv_state,
                                                                                st.score_state, n);
    // raw_state (the chunked-prefill pre-ape mirror) is zeroed, not -inf: it is written before it is
    // ever read (a chunk pass stores every row it pools), so this only keeps a stray read finite.
    DSV4_CK(cudaMemsetAsync(st.raw_state, 0, (size_t) n * 4, stream));
    DSV4_CK(cudaGetLastError());
    return true;
}

void dsv4_comp_state_free(Dsv4CompState& st) {
    cudaFree(st.kv_state);
    cudaFree(st.score_state);
    cudaFree(st.raw_state);
    st = Dsv4CompState{};
}

// ========================================================================================= workspace

bool dsv4_comp_workspace_init(Dsv4CompWorkspace& ws, int max_s, int64_t W, int max_comp, int win,
                              int64_t n_embd, int64_t state_rows) {
    ws.max_s = max_s;
    ws.max_comp = max_comp;
    ws.W = W;
    ws.state_rows = state_rows;
    const size_t kv = (size_t) max_s * (size_t) W, kvc = (size_t) max_comp * (size_t) kHeadDim,
                 shift = (size_t) state_rows * (size_t) W,
                 kv_all = ((size_t) max_s + (size_t) max_comp) * (size_t) kHeadDim,
                 topk = (size_t) max_s * (size_t) (win + max_comp), y = (size_t) max_s * (size_t) n_embd,
                 h2 = (size_t) max_s * 4 * (size_t) n_embd;
    bool ok = cudaMalloc(&ws.kv, kv * 4) == cudaSuccess &&
              cudaMalloc(&ws.score, kv * 4) == cudaSuccess &&
              cudaMalloc(&ws.kv_c, kvc * 4) == cudaSuccess &&
              cudaMalloc(&ws.shift, shift * 4) == cudaSuccess &&
              cudaMalloc(&ws.kv_all, kv_all * 4) == cudaSuccess &&
              cudaMalloc(&ws.topk, topk * 4) == cudaSuccess &&
              cudaMalloc(&ws.y, y * 4) == cudaSuccess &&
              cudaMalloc(&ws.post, (size_t) max_s * 4 * 4) == cudaSuccess &&
              cudaMalloc(&ws.mixes, (size_t) max_s * 24 * 4) == cudaSuccess &&
              cudaMalloc(&ws.comb, (size_t) max_s * 16 * 4) == cudaSuccess &&
              cudaMalloc(&ws.bout, y * 4) == cudaSuccess &&
              cudaMalloc(&ws.h2, h2 * 4) == cudaSuccess;
    if (!ok) {
        std::fprintf(stderr, "dsv4_comp: workspace allocation failed\n");
        dsv4_comp_workspace_free(ws);
        return false;
    }
    return true;
}

void dsv4_comp_workspace_free(Dsv4CompWorkspace& ws) {
    cudaFree(ws.kv);
    cudaFree(ws.score);
    cudaFree(ws.kv_c);
    cudaFree(ws.shift);
    cudaFree(ws.kv_all);
    cudaFree(ws.topk);
    cudaFree(ws.y);
    cudaFree(ws.post);
    cudaFree(ws.mixes);
    cudaFree(ws.comb);
    cudaFree(ws.bout);
    cudaFree(ws.h2);
    ws = Dsv4CompWorkspace{};
}

// ======================================================================================== compressor

int64_t dsv4_comp_forward(const Dsv4CompWeights& w, Dsv4CompState& st, Dsv4CompWorkspace& ws,
                          const float* x, int64_t s, int64_t start_pos, const Dsv4RopeTables& rope,
                          double eps, float* out_cache, float* kv_out, cudaStream_t stream,
                          bool chunk_prefill) {
    const int64_t ratio = st.ratio, d = st.d, coff = st.coff, W = coff * d;
    const bool overlap = coff == 2;
    if (s > ws.max_s || W > ws.W) {
        std::fprintf(stderr, "dsv4_comp: workspace undersized (s %lld, W %lld)\n", (long long) s,
                     (long long) W);
        std::abort();
    }
    // kv/score projections: [s][W] each, BF16 weights, double accumulation
    {
        AttnTimeScope t(kAtCompGemm, stream);
        // [M3] tiled launch behind STRATA_DSV4_ATTN_BATCH (bit-identical: same k order, same fp64 tree)
        dsv4_bf16_gemm(w.wkv, (int) W, kNEmbd, x, s, ws.kv, stream);
        DSV4_CK(cudaGetLastError());
        dsv4_bf16_gemm(w.wgate, (int) W, kNEmbd, x, s, ws.score, stream);
        DSV4_CK(cudaGetLastError());
    }

    int64_t ncomp = 0;
    if (start_pos == 0) {
        const bool should = s >= ratio;
        const int64_t rem = s % ratio, cutoff = s - rem;
        const int64_t offset = overlap ? ratio : 0;
        // overlap: the last full block becomes the carry's first half (scores carry the ape add)
        if (overlap && cutoff >= ratio) {
            DSV4_CK(cudaMemcpyAsync(st.kv_state, ws.kv + (size_t) (cutoff - ratio) * W,
                                    (size_t) (ratio * W) * 4, cudaMemcpyDeviceToDevice, stream));
            const int64_t n = ratio * W;
            AttnTimeScope t(kAtCompPool, stream);
            dsv4_comp_ape_store_kernel<<<(unsigned) ((n + 255) / 256), 256, 0, stream>>>(
                st.score_state, ws.score + (size_t) (cutoff - ratio) * W, w.ape, n);
            DSV4_CK(cudaGetLastError());
            // [chunk] a continuation pass pools this half through raw_state (prefill order, ape in
            // double), so keep the un-added score next to the added one
            if (chunk_prefill)
                DSV4_CK(cudaMemcpyAsync(st.raw_state, ws.score + (size_t) (cutoff - ratio) * W,
                                        (size_t) (ratio * W) * 4, cudaMemcpyDeviceToDevice, stream));
        }
        if (rem > 0) {
            DSV4_CK(cudaMemcpyAsync(st.kv_state + (size_t) offset * W, ws.kv + (size_t) cutoff * W,
                                    (size_t) (rem * W) * 4, cudaMemcpyDeviceToDevice, stream));
            const int64_t n = rem * W;
            AttnTimeScope t(kAtCompPool, stream);
            dsv4_comp_ape_store_kernel<<<(unsigned) ((n + 255) / 256), 256, 0, stream>>>(
                st.score_state + (size_t) offset * W, ws.score + (size_t) cutoff * W, w.ape, n);
            DSV4_CK(cudaGetLastError());
            if (chunk_prefill)
                DSV4_CK(cudaMemcpyAsync(st.raw_state + (size_t) offset * W,
                                        ws.score + (size_t) cutoff * W, (size_t) (rem * W) * 4,
                                        cudaMemcpyDeviceToDevice, stream));
        }
        if (!should) return 0;
        ncomp = cutoff / ratio;
        const int64_t n = ncomp * d;
        AttnTimeScope t(kAtCompPool, stream);
        dsv4_comp_pool_kernel<<<(unsigned) ((n + 255) / 256), 256, 0, stream>>>(
            ws.kv, ws.score, w.ape, ncomp, d, ratio, (int) coff, ws.kv_c);
        DSV4_CK(cudaGetLastError());
    } else {
        // [G1] decode, s consecutive rows (positions start_pos .. start_pos+s-1).  Row j runs the exact
        // single-token body on its own kv/score row and then advances the state, so an s-row pass is
        // bit-identical to s sequential single-token calls.  A batch may cross a block boundary (and one
        // row completes at most one block), so each emitted block records its own rope position and
        // out_cache slot instead of assuming a single one.
        std::vector<int64_t> emit_pos, emit_slot;
        for (int64_t j = 0; j < s; ++j) {
            const int64_t pos = start_pos + j;
            const bool should = (pos + 1) % ratio == 0;
            const int64_t rem = pos % ratio;
            const int64_t dst_row = overlap ? ratio + rem : rem;
            if (chunk_prefill) {
                // [chunk] keep the score un-added; the spanning block's pool re-applies ape in double
                DSV4_CK(cudaMemcpyAsync(st.raw_state + (size_t) dst_row * W, ws.score + (size_t) j * W,
                                        (size_t) W * 4, cudaMemcpyDeviceToDevice, stream));
            } else {
                AttnTimeScope t(kAtCompPool, stream);
                dsv4_comp_ape_add_row_kernel<<<(unsigned) ((W + 255) / 256), 256, 0, stream>>>(
                    ws.score + (size_t) j * W, w.ape + (size_t) rem * W, W);
                DSV4_CK(cudaGetLastError());
            }
            DSV4_CK(cudaMemcpyAsync(st.kv_state + (size_t) dst_row * W, ws.kv + (size_t) j * W,
                                    (size_t) W * 4, cudaMemcpyDeviceToDevice, stream));
            if (!chunk_prefill)
                DSV4_CK(cudaMemcpyAsync(st.score_state + (size_t) dst_row * W,
                                        ws.score + (size_t) j * W, (size_t) W * 4,
                                        cudaMemcpyDeviceToDevice, stream));
            if (!should) continue;
            {
                AttnTimeScope t(kAtCompPool, stream);
                if (chunk_prefill) {
                    dsv4_comp_pool_chunk_kernel<<<(unsigned) ((d + 255) / 256), 256, 0, stream>>>(
                        st.kv_state, st.raw_state, w.ape, d, ratio, (int) coff,
                        ws.kv_c + (size_t) emit_pos.size() * d);
                } else {
                    dsv4_comp_pool_decode_kernel<<<(unsigned) ((d + 255) / 256), 256, 0, stream>>>(
                        st.kv_state, st.score_state, d, ratio, (int) coff,
                        ws.kv_c + (size_t) emit_pos.size() * d);
                }
                DSV4_CK(cudaGetLastError());
            }
            if (overlap) {
                // memmove(state, state + ratio*W, ratio*W) via staging (same-buffer D2D overlap is UB)
                const size_t bytes = (size_t) (ratio * W) * 4;
                DSV4_CK(cudaMemcpyAsync(ws.shift, st.kv_state + (size_t) ratio * W, bytes,
                                        cudaMemcpyDeviceToDevice, stream));
                DSV4_CK(cudaMemcpyAsync(st.kv_state, ws.shift, bytes, cudaMemcpyDeviceToDevice,
                                        stream));
                // [chunk] the continuation reads raw_state, so that is the half that has to move
                float* src_half = chunk_prefill ? st.raw_state : st.score_state;
                DSV4_CK(cudaMemcpyAsync(ws.shift, src_half + (size_t) ratio * W, bytes,
                                        cudaMemcpyDeviceToDevice, stream));
                DSV4_CK(cudaMemcpyAsync(src_half, ws.shift, bytes, cudaMemcpyDeviceToDevice,
                                        stream));
            }
            emit_pos.push_back(pos + 1 - ratio);
            emit_slot.push_back(pos / ratio);
        }
        ncomp = (int64_t) emit_pos.size();
        if (ncomp == 0) return 0;
        // rmsnorm(norm) -> rope at each block's own phase -> its own out_cache slot
        for (int64_t b = 0; b < ncomp; ++b) {
            float* row = ws.kv_c + (size_t) b * d;
            {
                AttnTimeScope t(kAtNorm, stream);
                dsv4_rmsnorm_rope_rows(row, 1, (int) d, rope.half, 1, (int) ratio, rope,
                                       emit_pos[(size_t) b], 0, w.norm, stream);
            }
            if (out_cache || kv_out) {
                AttnTimeScope t(kAtMem, stream);
                if (out_cache)
                    DSV4_CK(cudaMemcpyAsync(out_cache + (size_t) emit_slot[(size_t) b] * d, row,
                                            (size_t) d * 4, cudaMemcpyDeviceToDevice, stream));
                if (kv_out)
                    DSV4_CK(cudaMemcpyAsync(kv_out + (size_t) b * d, row, (size_t) d * 4,
                                            cudaMemcpyDeviceToDevice, stream));
            }
        }
        return ncomp;
    }

    // rmsnorm(norm) -> rope at phase j*ratio (prefill) -> cache + kv_out
    // [S7e] norm + rope in one launch (bit-identical, see dsv4_rmsnorm_rope_kernel)
    {
        AttnTimeScope t(kAtNorm, stream);
        dsv4_rmsnorm_rope_rows(ws.kv_c, (int) ncomp, (int) d, rope.half, 1, (int) ratio, rope, 0,
                               0, w.norm, stream);
    }
    {
        AttnTimeScope t(kAtMem, stream);
        if (out_cache) {
            DSV4_CK(cudaMemcpyAsync(out_cache, ws.kv_c, (size_t) (ncomp * d) * 4,
                                    cudaMemcpyDeviceToDevice, stream));
        }
        if (kv_out)
            DSV4_CK(cudaMemcpyAsync(kv_out, ws.kv_c, (size_t) (ncomp * d) * 4,
                                    cudaMemcpyDeviceToDevice, stream));
    }
    return ncomp;
}

// ========================================================================================== topk

int64_t dsv4_compress_topk(int64_t ratio, int64_t s, int64_t start_pos, int64_t offset,
                           int32_t* topk) {
    if (start_pos > 0) {
        // [G1] multi-row decode: row j has n_j = (start_pos+j+1)/ratio blocks available, so the dense
        // row stride is the largest of them and the not-yet-written slots are -1.  s == 1 is unchanged.
        const int64_t kc = (start_pos + s) / ratio;
        for (int64_t j = 0; j < s; ++j) {
            const int64_t n = (start_pos + j + 1) / ratio;
            for (int64_t t = 0; t < kc; ++t)
                topk[(size_t) (j * kc + t)] = (int32_t) (t < n ? t + offset : -1);
        }
        return kc;
    }
    const int64_t nc = s / ratio;
    for (int64_t j = 0; j < s; ++j)
        for (int64_t t = 0; t < nc; ++t)
            topk[(size_t) (j * nc + t)] = (int32_t) (t >= (j + 1) / ratio ? -1 : t + offset);
    return nc;
}

// ============================================================================================ [S5] indexer

bool dsv4_indexer_workspace_init(Dsv4IndexerWorkspace& ws, int64_t max_s) {
    const cudaError_t e1 = cudaMalloc(&ws.q, (size_t) (max_s * 64 * 128) * 4);
    const cudaError_t e2 = cudaMalloc(&ws.wgt, (size_t) (max_s * 64) * 4);
    if (e1 != cudaSuccess || e2 != cudaSuccess) {
        cudaFree(ws.q);
        cudaFree(ws.wgt);
        ws = Dsv4IndexerWorkspace{};
        return false;
    }
    ws.dp = nullptr;
    ws.sc = nullptr;
    ws.tk = nullptr;
    ws.cap = 0;
    ws.cap_s = 0;
    ws.max_s = max_s;
    return true;
}

bool dsv4_indexer_workspace_reserve(Dsv4IndexerWorkspace& ws, int64_t s, int64_t nc) {
    const int64_t need = s * nc;
    if (need > ws.cap || s > ws.cap_s) {
        const int64_t cap = std::max<int64_t>(need * 2, 1024);
        const int64_t cap_s = std::max<int64_t>(s, ws.cap_s);
        cudaFree(ws.dp);
        cudaFree(ws.sc);
        cudaFree(ws.tk);
        ws.dp = nullptr;
        ws.sc = nullptr;
        ws.tk = nullptr;
        ws.cap = 0;
        ws.cap_s = 0;
        if (cudaMalloc(&ws.dp, (size_t) (cap * 64) * 4) != cudaSuccess ||
            cudaMalloc(&ws.sc, (size_t) cap * 4) != cudaSuccess ||
            cudaMalloc(&ws.tk, (size_t) (cap_s * 512) * 4) != cudaSuccess) {
            cudaFree(ws.dp);
            cudaFree(ws.sc);
            cudaFree(ws.tk);
            ws.dp = nullptr;
            ws.sc = nullptr;
            ws.tk = nullptr;
            return false;
        }
        ws.cap = cap;
        ws.cap_s = cap_s;
    }
    return true;
}

void dsv4_indexer_workspace_free(Dsv4IndexerWorkspace& ws) {
    cudaFree(ws.q);
    cudaFree(ws.wgt);
    cudaFree(ws.dp);
    cudaFree(ws.sc);
    cudaFree(ws.tk);
    ws = Dsv4IndexerWorkspace{};
}

namespace {

// dsv4ref.cpp:140 dot_fp32 verbatim (8-way float accumulation, pairwise 4+4 combine) - the indexer
// score's dot.  Host-side so the score bits track the reference's accumulation order as closely as the
// upstream tensors allow.
float dsv4_idx_dot(const float* a, const float* b, int64_t n) {
    float acc0 = 0, acc1 = 0, acc2 = 0, acc3 = 0, acc4 = 0, acc5 = 0, acc6 = 0, acc7 = 0;
    int64_t i = 0;
    for (; i + 8 <= n; i += 8) {
        acc0 += a[i] * b[i];
        acc1 += a[i + 1] * b[i + 1];
        acc2 += a[i + 2] * b[i + 2];
        acc3 += a[i + 3] * b[i + 3];
        acc4 += a[i + 4] * b[i + 4];
        acc5 += a[i + 5] * b[i + 5];
        acc6 += a[i + 6] * b[i + 6];
        acc7 += a[i + 7] * b[i + 7];
    }
    float acc = ((acc0 + acc1) + (acc2 + acc3)) + ((acc4 + acc5) + (acc6 + acc7));
    for (; i < n; ++i) acc += a[i] * b[i];
    return acc;
}


// ============================================================================== [S7c] indexer top-k on device
//
// The host contract (dsv4_indexer_topk_host above) is: score[j][t] = sum_h relu(q[j][h].keys[t]) *
// wgt[j][h] in fp32 with dsv4_idx_dot's 8-accumulator order and an ascending-h sum, then a
// STABLE descending sort whose ties keep ascending t.  All three steps are bit-exact ports: the dot
// and the h-sum use __fmul_rn/__fadd_rn so nvcc cannot contract them into FMA (the g++ host build
// does not either), and the selection sorts one 64-bit key per t: the order-preserving float flip of
// the score, complemented (descending) in the high half, t in the low half.  Comparing that key as an
// unsigned integer is therefore exactly "score desc, then t asc", i.e. the stable tie order.  -inf
// masked entries map to the largest key and sort last, as on the host.

__device__ __forceinline__ unsigned int idx_mono(float f) {
    const unsigned int u = __float_as_uint(f);
    return u ^ (unsigned int) (((int) u >> 31) | 0x80000000);
}

__device__ __forceinline__ float idx_dot_exact(const float* __restrict__ a, const float* __restrict__ b,
                                               int n) {
    float acc0 = 0, acc1 = 0, acc2 = 0, acc3 = 0, acc4 = 0, acc5 = 0, acc6 = 0, acc7 = 0;
    int i = 0;
    for (; i + 8 <= n; i += 8) {
        acc0 = __fadd_rn(acc0, __fmul_rn(a[i + 0], b[i + 0]));
        acc1 = __fadd_rn(acc1, __fmul_rn(a[i + 1], b[i + 1]));
        acc2 = __fadd_rn(acc2, __fmul_rn(a[i + 2], b[i + 2]));
        acc3 = __fadd_rn(acc3, __fmul_rn(a[i + 3], b[i + 3]));
        acc4 = __fadd_rn(acc4, __fmul_rn(a[i + 4], b[i + 4]));
        acc5 = __fadd_rn(acc5, __fmul_rn(a[i + 5], b[i + 5]));
        acc6 = __fadd_rn(acc6, __fmul_rn(a[i + 6], b[i + 6]));
        acc7 = __fadd_rn(acc7, __fmul_rn(a[i + 7], b[i + 7]));
    }
    float acc = __fadd_rn(__fadd_rn(__fadd_rn(acc0, acc1), __fadd_rn(acc2, acc3)),
                          __fadd_rn(__fadd_rn(acc4, acc5), __fadd_rn(acc6, acc7)));
    for (; i < n; ++i) acc = __fadd_rn(acc, __fmul_rn(a[i], b[i]));
    return acc;
}

/// dproj[j][t][h] = q[j][h] . keys[t]; one block per (j,t), one lane per head.  dsv4_idx_dot order.
__global__ void dsv4_idx_dp_kernel(const float* __restrict__ q, const float* __restrict__ keys,
                                   float* __restrict__ dp, int s, int nc, int nh, int hd) {
    const int jt = blockIdx.x;
    const int j = jt / nc, t = jt - j * nc;
    const int h = threadIdx.x;
    if (h >= nh) return;
    dp[(size_t) jt * nh + h] = idx_dot_exact(q + ((size_t) j * nh + h) * hd, keys + (size_t) t * hd, hd);
}

/// score[j][t] = sum_h relu(dproj) * (wgt_raw[h] * wscale), ascending h; row j masks t >= n_j with
/// n_j = (start_pos+j+1)/ratio blocks (the uniform causal rule: prefill start_pos == 0, decode rows past).
__global__ void dsv4_idx_score_kernel(const float* __restrict__ dp, const float* __restrict__ wgt,
                                      float* __restrict__ sc, int s, int nc, int nh, int ratio,
                                      int64_t start_pos, float wscale) {
    const int64_t n = (int64_t) s * nc;
    for (int64_t i = blockIdx.x * (int64_t) blockDim.x + threadIdx.x; i < n;
         i += (int64_t) gridDim.x * blockDim.x) {
        const int j = (int) (i / nc), t = (int) (i - (int64_t) j * nc);
        const float* d = dp + i * nh;
        float acc = 0.0f;
        for (int h = 0; h < nh; ++h) {
            const float raw = d[h];
            const float r = (raw < 0.0f) ? 0.0f : raw;
            acc = __fadd_rn(acc, __fmul_rn(r, __fmul_rn(wgt[j * nh + h], wscale)));
        }
        if (t >= (start_pos + j + 1) / ratio) acc = -__int_as_float(0x7f800000);  // -inf, host mask
        sc[i] = acc;
    }
}

/// One block per j: bitonic sort of the 64-bit (desc score, asc t) keys, then emit the first K rows of
/// the dense list with the host's offset / prefill cutoff mapping.
__global__ void dsv4_idx_select_kernel(const float* __restrict__ sc, int32_t* __restrict__ tk,
                                       int s, int nc, int n2, int64_t start_pos, int64_t offset,
                                       int top_k, int ratio) {
    extern __shared__ unsigned long long key[];  // n2 entries, n2 = next_pow2(nc)
    const int j = blockIdx.x;
    const int K = min(top_k, nc);
    const int64_t cutoff = (start_pos + j + 1) / ratio;  // [G1] row j's own block count
    for (int i = threadIdx.x; i < n2; i += blockDim.x) {
        unsigned long long k = ~0ull;
        if (i < nc) {
            const float v = sc[(size_t) j * nc + i];
            k = ((unsigned long long) (unsigned int) ~idx_mono(v) << 32) | (unsigned int) i;
        }
        key[i] = k;
    }
    __syncthreads();
    for (int k = 2; k <= n2; k <<= 1) {
        for (int m = k >> 1; m > 0; m >>= 1) {
            for (int i = threadIdx.x; i < n2; i += blockDim.x) {
                const int l = i ^ m;
                if (l > i) {
                    const bool up = ((i & k) == 0);
                    const unsigned long long a = key[i], b = key[l];
                    if ((a > b) == up) {
                        key[i] = b;
                        key[l] = a;
                    }
                }
            }
            __syncthreads();
        }
    }
    for (int k = threadIdx.x; k < K; k += blockDim.x) {
        int64_t idx = (int64_t) (key[k] & 0xffffffffull);
        if (idx >= cutoff)
            idx = -1;
        else
            idx += offset;
        tk[(size_t) j * top_k + k] = (int32_t) idx;
    }
}

// dsv4ref.cpp:895 indexer_forward's score + selection, host-side for the EXACT top-512 contract
// (integration plan section 6.2: fp32 keys, stable sort, ties ascending; never TF32/bf16 in the select).
// q [s][64][128] (roped), keys the indexer cache [nc][128] fp32, wgt [s][64] ALREADY scaled by
// 128^-0.5 * 64^-0.5.  score[j][t] = sum_h relu(q[j][h] . keys[t]) * wgt[j][h]; prefill masks
// t >= (j+1)/4; the selected indices get +offset (s prefill / win decode), masked entries stay -1.
// Writes topk row-major [s][K]; returns K = min(top_k, nc).
int64_t dsv4_indexer_topk_host(const float* q, const float* keys, const float* wgt, int64_t s,
                               int64_t nc, int64_t start_pos, int64_t offset, int64_t top_k,
                               int32_t* topk) {
    const int64_t nh = 64, hd = 128, ratio = 4;
    const float NEG = -std::numeric_limits<float>::infinity();
    const int64_t K = std::min(top_k, nc);
    for (int64_t j = 0; j < s; ++j) {
        std::vector<float> score((size_t) nc);
        for (int64_t t = 0; t < nc; ++t) {
            float acc = 0;
            const float* kv = keys + (size_t) t * hd;
            for (int64_t h = 0; h < nh; ++h) {
                const float dprod = dsv4_idx_dot(q + (size_t) (j * nh + h) * hd, kv, hd);
                acc += std::max(dprod, 0.0f) * wgt[(size_t) (j * nh + h)];
            }
            score[(size_t) t] = acc;
        }
        const int64_t cutoff = (start_pos + j + 1) / ratio;
        for (int64_t t = 0; t < nc; ++t)
            if (t >= cutoff) score[(size_t) t] = NEG;
        std::vector<int64_t> order((size_t) nc);
        for (int64_t i = 0; i < nc; ++i) order[(size_t) i] = i;
        std::stable_sort(order.begin(), order.end(),
                         [&](int64_t a, int64_t b) { return score[(size_t) a] > score[(size_t) b]; });
        for (int64_t k = 0; k < K; ++k) {
            int64_t idx = order[(size_t) k];
            if (idx >= cutoff)
                idx = -1;
            else
                idx += offset;
            topk[(size_t) (j * K + k)] = (int32_t) idx;
        }
    }
    return K;
}

}  // namespace

// The r=4 layer attention block (dsv4ref attention_forward :967 ratio==4 arm).  Same skeleton as
// dsv4_comp_attn_forward, with the compress topk replaced by the indexer top-512.
bool dsv4_idx_attn_forward(const Dsv4AttnWeights& w, const Dsv4RopeTables& rope,
                           Dsv4AttnWorkspace& aws, const Dsv4CompWeights& cw, Dsv4CompState& cst,
                           const Dsv4IndexerWeights& iw, Dsv4CompState& ist, float* idx_cache,
                           Dsv4CompWorkspace& cws, Dsv4IndexerWorkspace& iws, const float* x, float* out,
                           float* kv_cache, int64_t s, int64_t start_pos, int64_t sliding_window,
                           cudaStream_t stream, bool chunk_prefill) {
    constexpr int kQRank = 1024, kORows = 8192, kOGroup = 1024;
    constexpr int kIdxHeads = 64, kIdxDim = 128;
    const int win = (int) sliding_window;
    // [R-02/R-04] input bounds, checked here as well as at run_pass (see dsv4_attn_forward).
    if (s <= 0 || s > (int64_t) aws.max_s || s > (int64_t) cws.max_s) {
        std::fprintf(stderr, "dsv4_idx_attn: %lld rows exceed the workspace width (attn %d / comp %d)\n",
                     (long long) s, aws.max_s, cws.max_s);
        return false;
    }
    if (start_pos < 0) {
        std::fprintf(stderr, "dsv4_idx_attn: negative start_pos %lld\n", (long long) start_pos);
        return false;
    }
    if (start_pos == 0 && s > (int64_t) win) {
        std::fprintf(stderr,
                     "dsv4_idx_attn: prefill of %lld rows exceeds the sliding window %d (ring shift is undefined "
                     "for s > win)\n", (long long) s, win);
        return false;
    }
    // q path: gemm -> rmsnorm(q_norm) -> gemm -> per-head renorm -> rope (qr is shared with the indexer)
    // [S7c] gemms consume the warp-slice-transposed copies (bit-identical values)
    {
        AttnTimeScope t(kAtXpose, stream);
        dsv4_xpose1024_rows(x, aws.xt, s * kNEmbd, stream);
    }
    {
        AttnTimeScope t(kAtProj, stream);
        dsv4_gemm_q8_rows(w.wq_a, kQRank, kNEmbd, 0, aws.xt, (int) s, kNEmbd, aws.qr, stream);
    }
    {
        AttnTimeScope t(kAtNorm, stream);
        dsv4_rmsnorm_rows(aws.qr, (int) s, kQRank, w.q_norm, stream);
    }
    {
        AttnTimeScope t(kAtXpose, stream);
        dsv4_xpose1024_rows(aws.qr, aws.qr_t, s * kQRank, stream);
    }
    {
        AttnTimeScope t(kAtProj, stream);
        dsv4_gemm_q8_rows(w.wq_b, kNHead * kHeadDim, kQRank, 0, aws.qr_t, (int) s, kQRank, aws.q,
                          stream);
    }
    {
        AttnTimeScope t(kAtNorm, stream);
        dsv4_rmsnorm_rope_rows(aws.q, (int) s * kNHead, kHeadDim, rope.half, kNHead, 1, rope,
                               start_pos, 0, nullptr, stream);  // [S7e] norm + rope in one launch
    }
    // kv path: gemm -> rmsnorm(kv_norm) -> rope
    {
        AttnTimeScope t(kAtProj, stream);
        dsv4_gemm_q8_rows(w.wkv, kHeadDim, kNEmbd, 0, aws.xt, (int) s, kNEmbd, aws.kv_new, stream);
    }
    {
        AttnTimeScope t(kAtNorm, stream);
        dsv4_rmsnorm_rope_rows(aws.kv_new, (int) s, kHeadDim, rope.half, 1, 1, rope, start_pos, 0,
                               w.kv_norm, stream);  // [S7e] norm + rope in one launch
    }
    // window part of the cache (S3/S4 semantics: prefill writes/ring-shifts, decode writes one slot)
    {
        AttnTimeScope t(kAtMem, stream);
        if (start_pos == 0) {
            if (s <= win) {
                DSV4_CK(cudaMemcpyAsync(kv_cache, aws.kv_new, (size_t) (s * kHeadDim) * 4,
                                        cudaMemcpyDeviceToDevice, stream));
            } else {
                DSV4_CK(cudaMemcpyAsync(aws.cache_tmp, kv_cache + (size_t) s * kHeadDim,
                                        (size_t) ((win - s) * kHeadDim) * 4,
                                        cudaMemcpyDeviceToDevice, stream));
                DSV4_CK(cudaMemcpyAsync(kv_cache, aws.cache_tmp, (size_t) ((win - s) * kHeadDim) * 4,
                                        cudaMemcpyDeviceToDevice, stream));
                DSV4_CK(cudaMemcpyAsync(kv_cache + (size_t) (win - s) * kHeadDim, aws.kv_new,
                                        (size_t) (s * kHeadDim) * 4, cudaMemcpyDeviceToDevice,
                                        stream));
            }
        } else if (s == 1) {
            // decode: the single new row lands at slot start_pos % win
            DSV4_CK(cudaMemcpyAsync(kv_cache + (start_pos % win) * kHeadDim, aws.kv_new,
                                    (size_t) kHeadDim * 4, cudaMemcpyDeviceToDevice, stream));
        }
    }
    // [R-01] decode with s > 1: the ring slots are written PER ROW inside the attention loop below.
    // Writing all R rows up front is wrong once the batch reaches the ring: row j writes slot
    // (start_pos+j) % win, which is exactly the oldest slot of row j-1's window (position
    // start_pos+j-win, same slot), so rows 0..R-2 would attend to a FUTURE row's KV - silently.
    // attention compressor: compressed slots live behind the window region (rope phase j*ratio fixed at
    // write time; decode slot start_pos/ratio)
    const int64_t ncomp =
        dsv4_comp_forward(cw, cst, cws, x, s, start_pos, rope, kRmsEps,
                          kv_cache + (size_t) win * kHeadDim, nullptr, stream, chunk_prefill);
    // kv source + compressed-section offset (s prefill / win decode, dsv4ref :997)
    const float* kv_ptr = nullptr;
    int64_t offset = 0;
    if (start_pos == 0) {
        offset = s;
        if (ncomp > 0) {
            AttnTimeScope t(kAtMem, stream);
            DSV4_CK(cudaMemcpyAsync(cws.kv_all, aws.kv_new, (size_t) (s * kHeadDim) * 4,
                                    cudaMemcpyDeviceToDevice, stream));
            DSV4_CK(cudaMemcpyAsync(cws.kv_all + (size_t) s * kHeadDim,
                                    kv_cache + (size_t) win * kHeadDim, (size_t) (ncomp * kHeadDim) * 4,
                                    cudaMemcpyDeviceToDevice, stream));
            kv_ptr = cws.kv_all;
        } else {
            kv_ptr = aws.kv_new;
        }
    } else {
        kv_ptr = kv_cache;
        offset = win;
    }
    // [R8] readable KV rows for this pass: prefill sees kv_new (+ the compressed section when kv_all
    // stages it), decode sees the window plus the compressed slots behind it
    const int64_t kv_slots = start_pos == 0 ? (ncomp > 0 ? s + ncomp : s) : (int64_t) win;
    // indexer q: gemm(indexer.attn_q_b) over the SHARED qr, rope at the token position, NO renorm
    // (dsv4ref :903-907)
    {
        AttnTimeScope t(kAtProj, stream);
        dsv4_gemm_q8_rows(iw.wq_b, kIdxHeads * kIdxDim, kQRank, 0, aws.qr_t, (int) s, kQRank, iws.q,
                          stream);
    }
    {
        AttnTimeScope t(kAtRope, stream);
        dsv4_rope_rows_mul(iws.q, (int) s * kIdxHeads, kIdxDim, rope.half, kIdxHeads, 1, rope,
                           start_pos, 0, stream);
    }
    // indexer compressor: fp32 keys into the layer's indexer cache slice (own ratio-4/d-128/coff-2 state)
    dsv4_comp_forward(iw.comp, ist, cws, x, s, start_pos, rope, kRmsEps, idx_cache, nullptr, stream,
                      chunk_prefill);
    // weights: gemm(indexer.proj, raw BF16, double accumulation); the 128^-0.5 * 64^-0.5 scale is applied
    // on the host below, in dsv4ref's :919-921 order
    {
        AttnTimeScope t(kAtIdx, stream);
        dsv4_gemm_bf16_kernel<<<kIdxHeads, 128, 0, stream>>>(iw.proj, kIdxHeads, kNEmbd, x, (int) s,
                                                             iws.wgt);
        DSV4_CK(cudaGetLastError());
    }
    // selection (EXACT top-512 contract, dsv4ref :895): device score + bitonic selection, no host round
    // trip and no stream sync.  The key order reproduces the host's stable descending sort (ties asc t)
    // and the score kernels are __fmul_rn/__fadd_rn ports of dsv4_idx_dot / the ascending-h sum.
    const int64_t nc = (start_pos + s) / 4;
    // [chunk] a narrow prefill pass pads its window list to `win` entries so the compressed section
    // starts at the same column a full-width pass uses (the softmax denominator then groups the same
    // terms and the pass is bit-identical to the wide pass's first s rows).
    const bool pad_kw = chunk_prefill && start_pos == 0 && s < win;
    const int64_t kw_cap = pad_kw ? win : (start_pos > 0 ? win : std::min<int64_t>(s, win));
    // [M2] batched: the window list is built on the device and the ring is written after the launch.
    const bool batch = dsv4_attn_batch_enabled() && start_pos > 0 && s > 1;
    std::vector<int32_t> h_topk;
    int64_t kw = 0;
    if (batch) {
        kw = win;  // start_pos > 0 always carries the full ring list
    } else {
        h_topk.assign((size_t) (s * kw_cap), -1);
        const auto t_tab0 = std::chrono::steady_clock::now();
        kw = pad_kw ? dsv4_window_topk_pad(win, (int) s, h_topk.data())
                    : dsv4_window_topk(win, (int) s, start_pos, h_topk.data());
        attn_time_host_table(std::chrono::duration<double, std::milli>(
                                 std::chrono::steady_clock::now() - t_tab0).count());
    }
    int64_t kc = 0;
    if (nc > 0) {
        if (!dsv4_indexer_workspace_reserve(iws, s, nc)) {
            // [A5] a failed growth is an error, not a reason to abort the process
            std::fprintf(stderr, "dsv4_idx_attn: indexer workspace growth failed (s=%lld nc=%lld)\n",
                         (long long) s, (long long) nc);
            return false;
        }
        const float wscale = static_cast<float>(1.0 / std::sqrt((double) kIdxDim) *
                                                (1.0 / std::sqrt((double) kIdxHeads)));
        AttnTimeScope t(kAtIdx, stream);
        dsv4_idx_dp_kernel<<<(unsigned) (s * nc), kIdxHeads, 0, stream>>>(
            iws.q, idx_cache, iws.dp, (int) s, (int) nc, kIdxHeads, kIdxDim);
        const int th = 256;
        const int nb = (int) std::min<int64_t>((s * nc + th - 1) / th, 1024);
        dsv4_idx_score_kernel<<<nb, th, 0, stream>>>(iws.dp, iws.wgt, iws.sc, (int) s, (int) nc,
                                                     kIdxHeads, 4, start_pos, wscale);
        int n2 = 1;
        while (n2 < nc) n2 <<= 1;
        dsv4_idx_select_kernel<<<(unsigned) s, 256, (size_t) n2 * 8, stream>>>(
            iws.sc, iws.tk, (int) s, (int) nc, n2, start_pos, offset, 512, 4);
        DSV4_CK(cudaGetLastError());
        kc = std::min<int64_t>(512, nc);
    }
    // STRATA_DSV4_IDX_CHECK=1: diff the device selection against the host contract, once per layer
    if (kc > 0 && getenv("STRATA_DSV4_IDX_CHECK") != nullptr) {
        std::vector<float> h_q((size_t) (s * kIdxHeads * kIdxDim)), h_keys((size_t) (nc * kIdxDim)),
            h_wgt((size_t) (s * kIdxHeads));
        std::vector<int32_t> h_dev((size_t) (s * 512)), h_ref((size_t) (s * 512), -1);
        DSV4_CK(cudaStreamSynchronize(stream));
        DSV4_CK(cudaMemcpy(h_q.data(), iws.q, h_q.size() * 4, cudaMemcpyDeviceToHost));
        DSV4_CK(cudaMemcpy(h_keys.data(), idx_cache, h_keys.size() * 4, cudaMemcpyDeviceToHost));
        DSV4_CK(cudaMemcpy(h_wgt.data(), iws.wgt, h_wgt.size() * 4, cudaMemcpyDeviceToHost));
        DSV4_CK(cudaMemcpy(h_dev.data(), iws.tk, (size_t) (s * 512) * 4, cudaMemcpyDeviceToHost));
        const float wscale = static_cast<float>(1.0 / std::sqrt((double) kIdxDim) *
                                                (1.0 / std::sqrt((double) kIdxHeads)));
        for (auto& v : h_wgt) v *= wscale;
        dsv4_indexer_topk_host(h_q.data(), h_keys.data(), h_wgt.data(), s, nc, start_pos, offset, 512,
                               h_ref.data());
        const int64_t Kc = std::min<int64_t>(512, nc);
        int64_t bad = 0;
        for (int64_t j = 0; j < s; ++j)
            for (int64_t k = 0; k < Kc; ++k)
                bad += (h_dev[(size_t) (j * 512 + k)] != h_ref[(size_t) (j * Kc + k)]);
        fprintf(stderr, "dsv4 idx-check: nc=%lld K=%lld mismatched=%lld\n", (long long) nc,
                (long long) std::min<int64_t>(512, nc), (long long) bad);
    }
    // window rows come from the host list (or the device build, [M2]), indexer rows from the device
    // buffer; one 2-D copy each
    const int64_t K = kw + kc;
    {
        AttnTimeScope t(kAtTableH2D, stream);
        if (batch) {
            dsv4_window_topk_dev(win, (int) s, start_pos, (int) K, (int) kw, cws.topk, stream);
        } else {
            DSV4_CK(cudaMemcpy2DAsync(cws.topk, (size_t) K * 4, h_topk.data(), (size_t) kw_cap * 4,
                                      (size_t) kw * 4, (size_t) s, cudaMemcpyHostToDevice, stream));
        }
        if (kc > 0)
            DSV4_CK(cudaMemcpy2DAsync((int32_t*) cws.topk + kw, (size_t) K * 4, iws.tk,
                                      (size_t) 512 * 4, (size_t) kc * 4, (size_t) s,
                                      cudaMemcpyDeviceToDevice, stream));
    }
    // dense MQA-512 over the combined list, softmax scale 512^-0.5, sinks in the denominator only
    const float scale = 1.0f / std::sqrt(static_cast<float>(kHeadDim));
    {
        AttnTimeScope t(kAtMqa, stream);
        if (batch) {
            // [M2] one launch for all s rows; the ring is written afterwards (see dsv4_attn.cu's
            // dsv4_mqa512_rows_ring: a window entry never resolves to a row after the attending row).
            dsv4_mqa512_rows_ring(aws.q, aws.kv_new, kv_ptr, cws.topk, w.sinks, (int) s, (int) K,
                                  (int) (kv_slots + kc), win, start_pos, scale, aws.attn, stream);
        } else if (start_pos > 0 && s > 1) {
            // [G1] one launch per row: the kernel's row index is 0, so only the row's own table
            // changes.  [R-01] row j's ring slot is written immediately before row j's attention, so a
            // later row's KV can never land in an earlier row's window.
            for (int64_t j = 0; j < s; ++j) {
                DSV4_CK(cudaMemcpyAsync(kv_cache + ((start_pos + j) % win) * kHeadDim,
                                        aws.kv_new + j * kHeadDim, (size_t) kHeadDim * 4,
                                        cudaMemcpyDeviceToDevice, stream));
                dsv4_mqa512_rows(aws.q + j * kNHead * kHeadDim, kv_ptr, cws.topk + j * K, w.sinks,
                                 1, (int) K, (int) (kv_slots + kc), scale,
                                 aws.attn + j * kNHead * kHeadDim, stream);
            }
        } else {
            dsv4_mqa512_rows(aws.q, kv_ptr, cws.topk, w.sinks, (int) s, (int) K,
                             (int) (kv_slots + kc), scale, aws.attn, stream);
        }
    }
    if (batch) {
        AttnTimeScope t(kAtMem, stream);
        ring_write_rows(kv_cache, aws.kv_new, win, start_pos, s, stream);
    }
    // inverse rope on the attention output, then the grouped O projection
    {
        AttnTimeScope t(kAtRope, stream);
        dsv4_rope_rows_mul(aws.attn, (int) s * kNHead, kHeadDim, rope.half, kNHead, 1, rope,
                           start_pos, 1, stream);
    }
    // [S7c] wo_a reads the transposed attention output and writes o8k directly transposed for wo_b
    {
        AttnTimeScope t(kAtXpose, stream);
        dsv4_xpose1024_rows(aws.attn, aws.attn_t, s * kNHead * kHeadDim, stream);
    }
    {
        AttnTimeScope t(kAtProj, stream);
        dsv4_gemm_q8_rows(w.wo_a, kORows, kNEmbd, kOGroup, aws.attn_t, (int) s, kNHead * kHeadDim,
                          aws.o8k, stream, 1);
        dsv4_gemm_q8_rows(w.wo_b, kNEmbd, kORows, 0, aws.o8k, (int) s, kORows, out, stream);
    }
    return true;
}

// ============================================================================== compress attention

bool dsv4_comp_attn_forward(const Dsv4AttnWeights& w, const Dsv4RopeTables& rope,
                            Dsv4AttnWorkspace& aws, const Dsv4CompWeights& cw, Dsv4CompState& cst,
                            Dsv4CompWorkspace& cws, const float* x, float* out, float* kv_cache,
                            int64_t ratio, int64_t s, int64_t start_pos, int64_t sliding_window,
                            cudaStream_t stream, bool chunk_prefill) {
    constexpr int kQRank = 1024, kORows = 8192, kOGroup = 1024;
    const int win = (int) sliding_window;
    // [R-02/R-04] input bounds, checked here as well as at run_pass (see dsv4_attn_forward).
    if (s <= 0 || s > (int64_t) aws.max_s || s > (int64_t) cws.max_s) {
        std::fprintf(stderr, "dsv4_comp_attn: %lld rows exceed the workspace width (attn %d / comp %d)\n",
                     (long long) s, aws.max_s, cws.max_s);
        return false;
    }
    if (start_pos < 0) {
        std::fprintf(stderr, "dsv4_comp_attn: negative start_pos %lld\n", (long long) start_pos);
        return false;
    }
    if (start_pos == 0 && s > (int64_t) win) {
        std::fprintf(stderr,
                     "dsv4_comp_attn: prefill of %lld rows exceeds the sliding window %d (ring shift is undefined "
                     "for s > win)\n", (long long) s, win);
        return false;
    }
    // q path: gemm -> rmsnorm(q_norm) -> gemm -> per-head renorm -> rope (YaRN compress table)
    // [S7c] gemms consume the warp-slice-transposed copies (bit-identical values)
    {
        AttnTimeScope t(kAtXpose, stream);
        dsv4_xpose1024_rows(x, aws.xt, s * kNEmbd, stream);
    }
    {
        AttnTimeScope t(kAtProj, stream);
        dsv4_gemm_q8_rows(w.wq_a, kQRank, kNEmbd, 0, aws.xt, (int) s, kNEmbd, aws.qr, stream);
    }
    {
        AttnTimeScope t(kAtNorm, stream);
        dsv4_rmsnorm_rows(aws.qr, (int) s, kQRank, w.q_norm, stream);
    }
    {
        AttnTimeScope t(kAtXpose, stream);
        dsv4_xpose1024_rows(aws.qr, aws.qr_t, s * kQRank, stream);
    }
    {
        AttnTimeScope t(kAtProj, stream);
        dsv4_gemm_q8_rows(w.wq_b, kNHead * kHeadDim, kQRank, 0, aws.qr_t, (int) s, kQRank, aws.q,
                          stream);
    }
    {
        AttnTimeScope t(kAtNorm, stream);
        dsv4_rmsnorm_rope_rows(aws.q, (int) s * kNHead, kHeadDim, rope.half, kNHead, 1, rope,
                               start_pos, 0, nullptr, stream);  // [S7e] norm + rope in one launch
    }
    // kv path: gemm -> rmsnorm(kv_norm) -> rope
    {
        AttnTimeScope t(kAtProj, stream);
        dsv4_gemm_q8_rows(w.wkv, kHeadDim, kNEmbd, 0, aws.xt, (int) s, kNEmbd, aws.kv_new, stream);
    }
    {
        AttnTimeScope t(kAtNorm, stream);
        dsv4_rmsnorm_rope_rows(aws.kv_new, (int) s, kHeadDim, rope.half, 1, 1, rope, start_pos, 0,
                               w.kv_norm, stream);  // [S7e] norm + rope in one launch
    }
    // window part of the cache (S3 semantics: prefill writes/ring-shifts, decode writes one slot)
    {
        AttnTimeScope t(kAtMem, stream);
        if (start_pos == 0) {
            if (s <= win) {
                DSV4_CK(cudaMemcpyAsync(kv_cache, aws.kv_new, (size_t) (s * kHeadDim) * 4,
                                        cudaMemcpyDeviceToDevice, stream));
            } else {
                DSV4_CK(cudaMemcpyAsync(aws.cache_tmp, kv_cache + (size_t) s * kHeadDim,
                                        (size_t) ((win - s) * kHeadDim) * 4,
                                        cudaMemcpyDeviceToDevice, stream));
                DSV4_CK(cudaMemcpyAsync(kv_cache, aws.cache_tmp, (size_t) ((win - s) * kHeadDim) * 4,
                                        cudaMemcpyDeviceToDevice, stream));
                DSV4_CK(cudaMemcpyAsync(kv_cache + (size_t) (win - s) * kHeadDim, aws.kv_new,
                                        (size_t) (s * kHeadDim) * 4, cudaMemcpyDeviceToDevice,
                                        stream));
            }
        } else if (s == 1) {
            // decode: the single new row lands at slot start_pos % win
            DSV4_CK(cudaMemcpyAsync(kv_cache + (start_pos % win) * kHeadDim, aws.kv_new,
                                    (size_t) kHeadDim * 4, cudaMemcpyDeviceToDevice, stream));
        }
    }
    // [R-01] decode with s > 1: the ring slots are written PER ROW inside the attention loop below.
    // Writing all R rows up front is wrong once the batch reaches the ring: row j writes slot
    // (start_pos+j) % win, which is exactly the oldest slot of row j-1's window (position
    // start_pos+j-win, same slot), so rows 0..R-2 would attend to a FUTURE row's KV - silently.
    // compressor: compressed slots live behind the window region
    const int64_t ncomp =
        dsv4_comp_forward(cw, cst, cws, x, s, start_pos, rope, kRmsEps,
                          kv_cache + (size_t) win * kHeadDim, cws.kv_c, stream, chunk_prefill);
    // kv source + topk (host-built, then uploaded)
    const float* kv_ptr = nullptr;
    int64_t offset = 0;
    if (start_pos == 0) {
        offset = s;
        if (ncomp > 0) {
            AttnTimeScope t(kAtMem, stream);
            DSV4_CK(cudaMemcpyAsync(cws.kv_all, aws.kv_new, (size_t) (s * kHeadDim) * 4,
                                    cudaMemcpyDeviceToDevice, stream));
            DSV4_CK(cudaMemcpyAsync(cws.kv_all + (size_t) s * kHeadDim, cws.kv_c,
                                    (size_t) (ncomp * kHeadDim) * 4, cudaMemcpyDeviceToDevice,
                                    stream));
            kv_ptr = cws.kv_all;
        } else {
            kv_ptr = aws.kv_new;
        }
    } else {
        kv_ptr = kv_cache;
        offset = win;
    }
    // [R8] readable KV rows: prefill sees s rows (kv_new or the staged kv_all) plus the compressed
    // section, decode sees the window plus the compressed slots behind it
    const int64_t kv_slots = start_pos == 0 ? (ncomp > 0 ? s + ncomp : s) : (int64_t) win;
    // [chunk] see the r=4 path: pad the window list so a narrow prefill pass keeps the full-width
    // column layout of the compressed section.
    const bool pad_kw = chunk_prefill && start_pos == 0 && s < win;
    const int64_t kw_cap = pad_kw ? win : (start_pos > 0 ? win : std::min<int64_t>(s, win));
    // [G1] the compress block's row stride is the LARGEST row count in the batch ((start_pos+s)/ratio);
    // for s == 1 this is the old (start_pos+1)/ratio.
    const int64_t kc_cap = (start_pos + s) / ratio;
    // [M2] batched: both table blocks are built on the device (the compress block is the pure index
    // function t + offset, so it needs no host pass) and the ring is written after the launch.
    const bool batch = dsv4_attn_batch_enabled() && start_pos > 0 && s > 1;
    std::vector<int32_t> h_topk;
    std::vector<int32_t> h_dense;
    int64_t kw = 0;
    int64_t kc = 0;
    if (batch) {
        kw = win;         // start_pos > 0 always carries the full ring list
        kc = kc_cap;      // = dsv4_compress_topk's return for start_pos > 0
    } else {
        h_topk.resize((size_t) (s * (kw_cap + kc_cap)));
        const auto t_tab0 = std::chrono::steady_clock::now();
        kw = pad_kw ? dsv4_window_topk_pad(win, (int) s, h_topk.data())
                    : dsv4_window_topk(win, (int) s, start_pos, h_topk.data());
        kc = dsv4_compress_topk(ratio, s, start_pos, offset,
                                h_topk.data() + (size_t) s * kw_cap);
        // fold the two row blocks into dense [s][kw+kc] rows
        h_dense.resize((size_t) (s * (kw + kc)));
        for (int64_t j = 0; j < s; ++j) {
            std::memcpy(h_dense.data() + (size_t) j * (kw + kc), h_topk.data() + (size_t) j * kw_cap,
                        (size_t) kw * sizeof(int32_t));
            if (kc > 0)
                std::memcpy(h_dense.data() + (size_t) j * (kw + kc) + kw,
                            h_topk.data() + (size_t) s * kw_cap + (size_t) j * kc_cap,
                            (size_t) kc * sizeof(int32_t));
        }
        attn_time_host_table(std::chrono::duration<double, std::milli>(
                                 std::chrono::steady_clock::now() - t_tab0).count());
    }
    const int64_t K = kw + kc;
    {
        AttnTimeScope t(kAtTableH2D, stream);
        if (batch) {
            dsv4_window_topk_dev(win, (int) s, start_pos, (int) K, (int) kw, cws.topk, stream);
            if (kc > 0)
                dsv4_compress_topk_dev(ratio, (int) s, start_pos, offset, (int) kw, (int) kc, (int) K,
                                       cws.topk, stream);
        } else {
            DSV4_CK(cudaMemcpyAsync(cws.topk, h_dense.data(), (size_t) (s * K) * sizeof(int32_t),
                                    cudaMemcpyHostToDevice, stream));
        }
    }
    // dense MQA-512 over the combined list, softmax scale 512^-0.5, sinks in the denominator only
    const float scale = 1.0f / std::sqrt(static_cast<float>(kHeadDim));
    {
        AttnTimeScope t(kAtMqa, stream);
        if (batch) {
            // [M2] one launch for all s rows; the ring is written afterwards (see dsv4_attn.cu's
            // dsv4_mqa512_rows_ring: a window entry never resolves to a row after the attending row).
            dsv4_mqa512_rows_ring(aws.q, aws.kv_new, kv_ptr, cws.topk, w.sinks, (int) s, (int) K,
                                  (int) (kv_slots + kc), win, start_pos, scale, aws.attn, stream);
        } else if (start_pos > 0 && s > 1) {
            // [G1] one launch per row: the kernel's row index is 0, so only the row's own table
            // changes.  [R-01] row j's ring slot is written immediately before row j's attention, so a
            // later row's KV can never land in an earlier row's window.
            for (int64_t j = 0; j < s; ++j) {
                DSV4_CK(cudaMemcpyAsync(kv_cache + ((start_pos + j) % win) * kHeadDim,
                                        aws.kv_new + j * kHeadDim, (size_t) kHeadDim * 4,
                                        cudaMemcpyDeviceToDevice, stream));
                dsv4_mqa512_rows(aws.q + j * kNHead * kHeadDim, kv_ptr, cws.topk + j * K, w.sinks,
                                 1, (int) K, (int) (kv_slots + kc), scale,
                                 aws.attn + j * kNHead * kHeadDim, stream);
            }
        } else {
            dsv4_mqa512_rows(aws.q, kv_ptr, cws.topk, w.sinks, (int) s, (int) K,
                             (int) (kv_slots + kc), scale, aws.attn, stream);
        }
    }
    if (batch) {
        AttnTimeScope t(kAtMem, stream);
        ring_write_rows(kv_cache, aws.kv_new, win, start_pos, s, stream);
    }
    // inverse rope on the attention output, then the grouped O projection
    {
        AttnTimeScope t(kAtRope, stream);
        dsv4_rope_rows_mul(aws.attn, (int) s * kNHead, kHeadDim, rope.half, kNHead, 1, rope,
                           start_pos, 1, stream);
    }
    // [S7c] wo_a reads the transposed attention output and writes o8k directly transposed for wo_b
    {
        AttnTimeScope t(kAtXpose, stream);
        dsv4_xpose1024_rows(aws.attn, aws.attn_t, s * kNHead * kHeadDim, stream);
    }
    {
        AttnTimeScope t(kAtProj, stream);
        dsv4_gemm_q8_rows(w.wo_a, kORows, kNEmbd, kOGroup, aws.attn_t, (int) s, kNHead * kHeadDim,
                          aws.o8k, stream, 1);
        dsv4_gemm_q8_rows(w.wo_b, kNEmbd, kORows, 0, aws.o8k, (int) s, kORows, out, stream);
    }
    // compared token by token (filename carries the pass ordinal: p0 = start_pos 0, p1 = later).
    return true;
}

// ================================================================================== attention half

bool dsv4_attn_half(const Dsv4Geometry& g, int64_t layer, const Dsv4AttnHalfWeights& w,
                    const Dsv4RopeTables& rope_plain, const Dsv4RopeTables& rope_yarn,
                    float* d_h, float* kv_cache, Dsv4CompState* comp_state,
                    Dsv4CompState* idx_state, float* idx_cache, Dsv4AttnWorkspace& aws,
                    Dsv4CompWorkspace& cws, Dsv4IndexerWorkspace& iws, int64_t s, int64_t start_pos,
                    cudaStream_t stream, std::string& err, bool chunk_prefill) {
    const int64_t N = g.n_embd;
    const int64_t ratio = dsv4_compress_ratio(layer);
    attn_time_set_class(ratio == 0 ? kAtHash : (ratio == 128 ? kAtR128 : kAtIdxLayer));
    if (s > cws.max_s || s > iws.max_s) {
        err = "dsv4_attn_half: workspace undersized for s=" + std::to_string(s);
        return false;
    }
    if (ratio == 4 && (comp_state == nullptr || idx_state == nullptr || idx_cache == nullptr)) {
        err = "dsv4_attn_half: r=4 layer without compressor/indexer state";
        return false;
    }
    // hc_pre (attn side) -> rmsnorm(attn_norm): y is the attention block input
    {
        AttnTimeScope t(kAtHc, stream);
        dsv4_comp_hc_mix_kernel<<<dim3((unsigned) s, 24), 256, 0, stream>>>(d_h, w.hc_fn, N,
                                                                             g.rms_eps, cws.mixes);
        dsv4_comp_hc_fin_kernel<<<(unsigned) s, 256, 0, stream>>>(d_h, w.hc_base, w.hc_scale,
                                                                  cws.mixes, N, g.hc_eps, cws.y,
                                                                  cws.post, cws.comb);
        DSV4_CK(cudaGetLastError());
    }
    {
        AttnTimeScope t(kAtNorm, stream);
        dsv4_rmsnorm_rows(cws.y, (int) s, (int) N, w.attn_norm, stream);
    }
    if (ratio == 0) {
        // hash layers 0/1: dense MLA over the window cache, plain rope (theta 1e4)
        if (!dsv4_attn_forward(w.attn, rope_plain, aws, cws.y, cws.bout, kv_cache, (int) s, start_pos,
                               (int) g.sliding_window, stream)) {
            err = "dsv4_attn_half: hash-layer attention rejected the pass (see the message above)";
            return false;
        }
    } else if (ratio == 128) {
        // r=128 layers: compressor + MLA over the window + compressed KV, YaRN rope (theta 160000)
        if (!dsv4_comp_attn_forward(w.attn, rope_yarn, aws, w.comp, *comp_state, cws, cws.y, cws.bout,
                                    kv_cache, ratio, s, start_pos, g.sliding_window, stream,
                                    chunk_prefill)) {
            err = "dsv4_attn_half: compressed attention rejected the pass (see the message above)";
            return false;
        }
    } else {
        // r=4 layers (even 2..42; layer 2 is the hash+DSA hybrid - the hash side lives in the MoE
        // half's tid2eid routing, the attention here is the full DSA path): compressor + indexer
        // top-512 + MQA-512 (S5).  dsv4ref run_pass (:1287) picks the YaRN compress table for every
        // ratio>0 layer, L02 included, so this path uses rope_yarn (theta 160000).
        if (!dsv4_idx_attn_forward(w.attn, rope_yarn, aws, w.comp, *comp_state, w.idx, *idx_state,
                                   idx_cache, cws, iws, cws.y, cws.bout, kv_cache, s, start_pos,
                                   g.sliding_window, stream, chunk_prefill)) {
            err = "dsv4_attn_half: indexer attention rejected the pass (see the message above)";
            return false;
        }
    }
    // hc_post into staging, then back over h
    {
        AttnTimeScope t(kAtHc, stream);
        dsv4_comp_hc_post_kernel<<<(unsigned) s, 256, 0, stream>>>(d_h, cws.bout, cws.post, cws.comb,
                                                                   N, cws.h2);
        DSV4_CK(cudaGetLastError());
    }
    {
        AttnTimeScope t(kAtMem, stream);
        DSV4_CK(cudaMemcpyAsync(d_h, cws.h2, (size_t) (s * 4 * N) * 4, cudaMemcpyDeviceToDevice,
                                stream));
    }
    return true;
}

}  // namespace strata::dsv4
