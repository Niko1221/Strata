// src/kernels/cuda/qsa_select.cu - see include/strata/kernels/qsa_select.hpp.
#include "strata/kernels/qsa_select.hpp"

#include <cuda_runtime.h>

#include <cfloat>
#include <cstdio>
#include <cstdlib>

namespace strata::kernels {
namespace {

constexpr int IDX_DIM = 128, IDX_HEADS = 4, R = 4;
constexpr unsigned FULL = 0xffffffffu;
constexpr int SCORE_WARPS = 8;
constexpr int SCORE_KB = 2;                                // key blocks per 8-lane group
constexpr int SCORE_TILE = SCORE_WARPS * 4 * SCORE_KB;     // key blocks per thread block and round
constexpr int SCORE_QC = 16;                               // queries staged in shared memory at a time (32 KB)
constexpr int TOPK_T = 1024, TOPK_W = TOPK_T / 32;
constexpr int TOPK_ROWS = 32;                              // rows of TOPK_T blocks held in registers
constexpr int TOPK_MAX_ROWS = 64;                          // 65536 blocks: 262144 cells

__device__ __forceinline__ uint32_t order_key(float s) {
    const float v = s + 0.0f;
    if (!(v == v)) return 0u;
    const uint32_t b = __float_as_uint(v);
    return (b & 0x80000000u) ? ~b : (b | 0x80000000u);
}

// One query's scores for the group's SCORE_KB key blocks.  Lane j of the 8-lane group holds each key's dims 4l..4l+3
// for l = j, j+8, j+16, j+24; per head its four 4-term dot products d_l are summed in the lane as
// (d_j + d_j+16) + (d_j+8 + d_j+24), then across the group by xor 4, 2 and 1: the tree in which a warp-wide xor
// butterfly (16, 8, 4, 2, 1) over lanes l sums the same d_l, so each score is bit for bit that of one warp per
// (query, block).  TAIL: key `tail_kb` (-1: none) is the query's tail block and takes the `dead` key instead.
template <bool TAIL>
__device__ __forceinline__ void group_scores(const float4 (&k)[SCORE_KB][4], const float* __restrict__ dead,
                                             int tail_kb, const float* q, int j, float (&score)[SCORE_KB]) {
#pragma unroll
    for (int kb = 0; kb < SCORE_KB; ++kb) score[kb] = 0.0f;
#pragma unroll
    for (int h = 0; h < IDX_HEADS; ++h) {
        float4 q4[4];
#pragma unroll
        for (int i = 0; i < 4; ++i) q4[i] = *reinterpret_cast<const float4*>(q + h * IDX_DIM + 4 * (j + 8 * i));
#pragma unroll
        for (int kb = 0; kb < SCORE_KB; ++kb) {
            float d[4];
#pragma unroll
            for (int i = 0; i < 4; ++i) {
                float4 k4 = k[kb][i];
                if (TAIL && kb == tail_kb) k4 = *reinterpret_cast<const float4*>(dead + 4 * (j + 8 * i));
                d[i] = k4.x * q4[i].x + k4.y * q4[i].y + k4.z * q4[i].z + k4.w * q4[i].w;
            }
            float c = (d[0] + d[2]) + (d[1] + d[3]);
            c += __shfl_xor_sync(FULL, c, 4);
            c += __shfl_xor_sync(FULL, c, 2);
            c += __shfl_xor_sync(FULL, c, 1);
            score[kb] += c > 0.0f ? c : 0.0f;
        }
    }
}

// 8-lane groups of SCORE_KB consecutive key blocks each, read once for all the queries, which go through shared
// memory SCORE_QC at a time.  Block n_bid of a query is its incomplete tail: the `dead` key, +1e9 when it has cells.
__global__ void __launch_bounds__(SCORE_WARPS * 32) block_scores_kernel(const float* __restrict__ pooled,
                                                                        const float* __restrict__ dead,
                                                                        const float* __restrict__ q_idx,
                                                                        const int32_t* __restrict__ steps, int nq,
                                                                        int64_t max_blocks, float* __restrict__ out) {
    __shared__ __align__(16) float qs[SCORE_QC * IDX_HEADS * IDX_DIM];
    __shared__ int s_nkv[SCORE_QC], s_nbid[SCORE_QC];
    const int t = threadIdx.x, lane = t & 31, j = lane & 7;
    const int slot = ((t >> 5) * 4 + (lane >> 3)) * SCORE_KB;   // the group's first block within a tile
    int64_t top = -1;   // the queries' last block: the rounds end there
    for (int qi = 0; qi < nq; ++qi) {
        const int64_t n_bid = steps[(int64_t) qi * kStepCount + kStepNBid];
        const int64_t last = n_bid < max_blocks - 1 ? n_bid : max_blocks - 1;
        top = last > top ? last : top;
    }
    for (int64_t b0 = (int64_t) blockIdx.x * SCORE_TILE; b0 <= top; b0 += (int64_t) gridDim.x * SCORE_TILE) {
        const int64_t bg = b0 + slot;
        float4 k[SCORE_KB][4];
#pragma unroll
        for (int kb = 0; kb < SCORE_KB; ++kb) {
            const float* key = pooled + (bg + kb <= top ? bg + kb : 0) * IDX_DIM;
#pragma unroll
            for (int i = 0; i < 4; ++i) k[kb][i] = __ldg(reinterpret_cast<const float4*>(key + 4 * (j + 8 * i)));
        }
        for (int q0 = 0; q0 < nq; q0 += SCORE_QC) {
            const int qn = nq - q0 < SCORE_QC ? nq - q0 : SCORE_QC;
            __syncthreads();
            const float4* src = reinterpret_cast<const float4*>(q_idx + (int64_t) q0 * IDX_HEADS * IDX_DIM);
            for (int i = t; i < qn * IDX_HEADS * IDX_DIM / 4; i += SCORE_WARPS * 32)
                reinterpret_cast<float4*>(qs)[i] = src[i];
            if (t < qn) {
                s_nkv[t] = steps[(int64_t) (q0 + t) * kStepCount + kStepNKv];
                s_nbid[t] = steps[(int64_t) (q0 + t) * kStepCount + kStepNBid];
            }
            __syncthreads();
            for (int qq = 0; qq < qn; ++qq) {
                const int64_t n_kv = s_nkv[qq], n_bid = s_nbid[qq];
                const int64_t last = n_bid < max_blocks - 1 ? n_bid : max_blocks - 1;
                const int tail_kb = n_bid >= bg && n_bid < bg + SCORE_KB ? (int) (n_bid - bg) : -1;
                const float* q = qs + qq * IDX_HEADS * IDX_DIM;
                float score[SCORE_KB];
                // warp-uniform branches: every lane of the warp runs the group shuffles
                if (__any_sync(FULL, tail_kb >= 0)) group_scores<true>(k, dead, tail_kb, q, j, score);
                else group_scores<false>(k, dead, -1, q, j, score);
                if (j == 0) {
                    float* o = out + (int64_t) (q0 + qq) * max_blocks;
#pragma unroll
                    for (int kb = 0; kb < SCORE_KB; ++kb) {
                        if (bg + kb > last) break;
                        o[bg + kb] = kb == tail_kb && n_kv % R != 0 ? score[kb] + 1e9f : score[kb];
                    }
                }
            }
        }
    }
}

__device__ __forceinline__ int warp_sum(int v) {
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(FULL, v, o);
    return v;
}

// The exclusive prefixes of a and b over the block's threads in thread order, and their totals.
__device__ __forceinline__ void block_exclusive2(int a, int b, int& ea, int& eb, int& ta, int& tb, int* s_w) {
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    int ia = a, ib = b;
#pragma unroll
    for (int o = 1; o < 32; o <<= 1) {
        const int ya = __shfl_up_sync(FULL, ia, o), yb = __shfl_up_sync(FULL, ib, o);
        if (lane >= o) { ia += ya; ib += yb; }
    }
    if (lane == 31) { s_w[warp] = ia; s_w[TOPK_W + warp] = ib; }
    __syncthreads();
    if (warp == 0) {
        const int wa = s_w[lane], wb = s_w[TOPK_W + lane];
        int xa = wa, xb = wb;
#pragma unroll
        for (int o = 1; o < 32; o <<= 1) {
            const int ya = __shfl_up_sync(FULL, xa, o), yb = __shfl_up_sync(FULL, xb, o);
            if (lane >= o) { xa += ya; xb += yb; }
        }
        s_w[lane] = xa - wa;
        s_w[TOPK_W + lane] = xb - wb;
        if (lane == 31) { s_w[2 * TOPK_W] = xa; s_w[2 * TOPK_W + 1] = xb; }
    }
    __syncthreads();
    ea = s_w[warp] + ia - a;
    eb = s_w[TOPK_W + warp] + ib - b;
    ta = s_w[2 * TOPK_W];
    tb = s_w[2 * TOPK_W + 1];
    __syncthreads();
}

// The same selection as the per-cell topk_kernel: the `width` cells of the largest block scores (a block's score
// counts for each of its cells), ties to the lowest cell, emitted ascending.  A radix select, 8 bits at a time, over
// the order keys, with a histogram per warp.  The complete blocks go in rows of TOPK_T: thread t has block
// row * TOPK_T + t, the first TOPK_ROWS rows in registers (read once, 131K cells), the others from memory in each
// pass.  A block's cells then land at (cells above the threshold before it) + min(cells at it before it, budget):
// ballots within a (row, warp), one scan over the (row, warp) totals.  The tail block (n_bid) is thread 0's.
__global__ void __launch_bounds__(TOPK_T) block_topk_kernel(const float* __restrict__ scores,
                                                            const int32_t* __restrict__ steps, int64_t max_blocks,
                                                            int64_t cap, int32_t* __restrict__ ids) {
    __shared__ int sm[TOPK_W * 256];                   // the per-warp histograms; then the scan table and the output
    __shared__ int hist[256];
    __shared__ int s_w[2 * TOPK_W + 2];
    __shared__ int s_digit, s_above;
    const int64_t qi = blockIdx.x;
    const int32_t* st = steps + qi * kStepCount;
    const int64_t n_kv = st[kStepNKv], n_bid = st[kStepNBid], width = st[kStepWidth];
    int32_t* out = ids + qi * cap;
    const int t = threadIdx.x, lane = t & 31, warp = t >> 5;
    if (n_kv <= width) {                               // everything is selected: the identity, ascending
        for (int64_t j = t; j < n_kv; j += TOPK_T) out[j] = (int32_t) j;
        return;
    }
    const float* sc = scores + qi * max_blocks;
    const int tail_w = (int) (n_kv - n_bid * R);       // the cells of block n_bid, 0..3
    const int nrows = (int) ((n_bid + TOPK_T - 1) / TOPK_T);
    uint32_t key[TOPK_ROWS];
#pragma unroll
    for (int i = 0; i < TOPK_ROWS; ++i) {
        const int64_t b = (int64_t) i * TOPK_T + t;
        key[i] = b < n_bid ? order_key(sc[b]) : 0u;
    }
    // ---- radix select: the largest key thr with (cells with key >= thr) >= width
    uint32_t prefix = 0;
    int above = 0;                                     // cells strictly above the digits fixed so far
    int* whist = sm + warp * 256;
    for (int shift = 24; shift >= 0; shift -= 8) {
        for (int i = lane; i < 256; i += 32) whist[i] = 0;
        __syncwarp();
        const uint32_t hi_mask = shift == 24 ? 0u : (0xffffffffu << (shift + 8));
        int cd = -1, cc = 0;                           // a run of one digit is added at once
        auto count = [&](uint32_t k) {
            if ((k & hi_mask) != (prefix & hi_mask)) return;
            const int d = (int) ((k >> shift) & 255);
            if (d != cd) {
                if (cc != 0) atomicAdd(&whist[cd], cc);
                cd = d;
                cc = 0;
            }
            cc += R;
        };
#pragma unroll
        for (int i = 0; i < TOPK_ROWS; ++i)
            if ((int64_t) i * TOPK_T + t < n_bid) count(key[i]);
        for (int i = TOPK_ROWS; i < nrows; ++i) {
            const int64_t b = (int64_t) i * TOPK_T + t;
            if (b < n_bid) count(order_key(sc[b]));
        }
        if (cc != 0) atomicAdd(&whist[cd], cc);
        if (t == 0 && tail_w > 0) {
            const uint32_t k = order_key(sc[n_bid]);
            if ((k & hi_mask) == (prefix & hi_mask)) atomicAdd(&whist[(k >> shift) & 255], tail_w);
        }
        __syncthreads();
        if (t < 256) {
            int s = 0;
#pragma unroll 8
            for (int w = 0; w < TOPK_W; ++w) s += sm[w * 256 + t];
            hist[t] = s;
        }
        __syncthreads();
        if (warp == 0) {
            // the digit: the largest d >= 1 with above + (cells with a digit >= d) >= width, else 0; lane l holds the
            // bins 255 - 8l down to 248 - 8l
            int v[8], s = 0;
#pragma unroll
            for (int i = 0; i < 8; ++i) { v[i] = hist[255 - (lane * 8 + i)]; s += v[i]; }
            int incl = s;
#pragma unroll
            for (int o = 1; o < 32; o <<= 1) {
                const int y = __shfl_up_sync(FULL, incl, o);
                if (lane >= o) incl += y;
            }
            int run = incl - s, below = 0;
#pragma unroll
            for (int i = 0; i < 8; ++i) {
                run += v[i];
                if (255 - (lane * 8 + i) >= 1 && above + run < width) ++below;
            }
            const int digit = 255 - warp_sum(below);
            int gt = 0;
#pragma unroll
            for (int i = 0; i < 8; ++i)
                if (255 - (lane * 8 + i) > digit) gt += v[i];
            gt = warp_sum(gt);
            if (lane == 0) {
                s_digit = digit;
                s_above = above + gt;
            }
        }
        __syncthreads();
        prefix |= (uint32_t) s_digit << shift;
        above = s_above;
    }
    const uint32_t thr = prefix;
    const int budget = (int) (width - above);          // cells equal to thr that fit, lowest index first
    // ---- the cells above and at thr per (row, warp), their exclusive prefixes in block order
    int* tab_gt = sm;
    int* tab_eq = sm + TOPK_MAX_ROWS * 32;
    int* obuf = sm + 2 * TOPK_MAX_ROWS * 32;           // the selection, copied out coalesced
    __syncthreads();                                   // the histograms are no longer read
    auto row_key = [&](int i) { return i < TOPK_ROWS ? 0u : order_key(sc[(int64_t) i * TOPK_T + t]); };
#pragma unroll
    for (int i = 0; i < TOPK_ROWS; ++i) {
        if (i >= nrows) break;
        const bool in = (int64_t) i * TOPK_T + t < n_bid;
        const unsigned mg = __ballot_sync(FULL, in && key[i] > thr), me = __ballot_sync(FULL, in && key[i] == thr);
        if (lane == 0) { tab_gt[i * 32 + warp] = R * __popc(mg); tab_eq[i * 32 + warp] = R * __popc(me); }
    }
    for (int i = TOPK_ROWS; i < nrows; ++i) {
        const bool in = (int64_t) i * TOPK_T + t < n_bid;
        const uint32_t k = in ? row_key(i) : 0u;
        const unsigned mg = __ballot_sync(FULL, in && k > thr), me = __ballot_sync(FULL, in && k == thr);
        if (lane == 0) { tab_gt[i * 32 + warp] = R * __popc(mg); tab_eq[i * 32 + warp] = R * __popc(me); }
    }
    __syncthreads();
    int gt_all, eq_all;
    {
        const int e = nrows * 32, k0 = 2 * t;          // entries (row, warp) in block order, two per thread
        const int g0 = k0 < e ? tab_gt[k0] : 0, e0 = k0 < e ? tab_eq[k0] : 0;
        const int g1 = k0 + 1 < e ? tab_gt[k0 + 1] : 0, e1 = k0 + 1 < e ? tab_eq[k0 + 1] : 0;
        int gx, ex;
        block_exclusive2(g0 + g1, e0 + e1, gx, ex, gt_all, eq_all, s_w);
        if (k0 < e) { tab_gt[k0] = gx; tab_eq[k0] = ex; }
        if (k0 + 1 < e) { tab_gt[k0 + 1] = gx + g0; tab_eq[k0 + 1] = ex + e0; }
    }
    __syncthreads();
    // ---- the cells, into shared memory
    const unsigned lt = (1u << lane) - 1u;
    auto emit = [&](int i, uint32_t k, bool in) {
        const bool g = in && k > thr, q = in && k == thr;
        const unsigned mg = __ballot_sync(FULL, g), me = __ballot_sync(FULL, q);
        if (!g && !q) return;
        const int gb = tab_gt[i * 32 + warp] + R * __popc(mg & lt);
        const int eb = tab_eq[i * 32 + warp] + R * __popc(me & lt);
        const int pos = gb + (eb < budget ? eb : budget);
        const int take = g ? R : (budget - eb < 0 ? 0 : (budget - eb > R ? R : budget - eb));
        const int64_t b = (int64_t) i * TOPK_T + t;
        for (int c = 0; c < take; ++c) obuf[pos + c] = (int32_t) (b * R + c);
    };
#pragma unroll
    for (int i = 0; i < TOPK_ROWS; ++i) {
        if (i >= nrows) break;
        emit(i, key[i], (int64_t) i * TOPK_T + t < n_bid);
    }
    for (int i = TOPK_ROWS; i < nrows; ++i) {
        const bool in = (int64_t) i * TOPK_T + t < n_bid;
        emit(i, in ? row_key(i) : 0u, in);
    }
    if (t == 0 && tail_w > 0) {                        // the tail block comes last
        const uint32_t k = order_key(sc[n_bid]);
        const int pos = gt_all + (eq_all < budget ? eq_all : budget);
        const int take = k > thr ? tail_w : (k == thr ? (budget - eq_all < 0 ? 0 : (budget - eq_all > tail_w ? tail_w
                                                                                        : budget - eq_all)) : 0);
        for (int c = 0; c < take; ++c) obuf[pos + c] = (int32_t) (n_bid * R + c);
    }
    __syncthreads();
    for (int64_t j = t; j < width; j += TOPK_T) out[j] = obuf[j];
}

}  // namespace

void qsa_block_scores(const float* pooled, const float* dead, const float* q_idx, const int32_t* steps, int64_t nq,
                      int64_t max_blocks, const QsaShapes& s, float* scores, void* stream, int64_t grid_blocks) {
    if (nq <= 0) return;
    if (s.idx_dim != IDX_DIM || s.idx_n_head != IDX_HEADS || s.idx_block != R || nq > 65535 || grid_blocks < 1) {
        std::fprintf(stderr, "qsa_block_scores: unsupported indexer geometry\n");
        std::exit(1);
    }
    const int64_t covered = grid_blocks < max_blocks ? grid_blocks : max_blocks;
    const unsigned grid = (unsigned) ((covered + SCORE_TILE - 1) / SCORE_TILE);
    block_scores_kernel<<<grid, SCORE_WARPS * 32, 0, (cudaStream_t) stream>>>(pooled, dead, q_idx, steps, (int) nq,
                                                                              max_blocks, scores);
    const cudaError_t e = cudaGetLastError();
    if (e != cudaSuccess) { std::fprintf(stderr, "qsa_block_scores: %s\n", cudaGetErrorString(e)); std::exit(1); }
}

void qsa_block_topk(const float* scores, const int32_t* steps, int64_t nq, int64_t max_blocks, int64_t cap,
                    const QsaShapes& s, int32_t* ids, void* stream) {
    if (nq <= 0) return;
    if (s.idx_block != R || cap < qsa_selection_width(kTopkMaxCells, s) || max_blocks > (int64_t) TOPK_MAX_ROWS * TOPK_T) {
        std::fprintf(stderr, "qsa_block_topk: unsupported geometry or cap\n");
        std::exit(1);
    }
    block_topk_kernel<<<(unsigned) nq, TOPK_T, 0, (cudaStream_t) stream>>>(scores, steps, max_blocks, cap, ids);
    const cudaError_t e = cudaGetLastError();
    if (e != cudaSuccess) { std::fprintf(stderr, "qsa_block_topk: %s\n", cudaGetErrorString(e)); std::exit(1); }
}

}  // namespace strata::kernels
