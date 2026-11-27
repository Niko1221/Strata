#include "strata/hy4flash_gguf/indexer.hpp"
#include "strata/hy4flash_gguf/round_state.hpp"

#include "strata/hy4flash_gguf/attention.hpp"
#include "strata/hy4flash_gguf/mat.hpp"
#include "strata/kernels/native_mmvq.hpp"
#include "strata/kernels/elementwise.hpp"
#include "strata/kernels/elementwise.hpp"
#include "strata/kernels/pdl.hpp"

#include "reference_ops.hpp"

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace strata::hy4_flash {
namespace {

constexpr int kBlock = 256;
__global__ void commit_carry_k(float* vg,const float* saved,int kd,int kp,int base,int old_end,const RoundVerdict* verdict,const DeviceLoopState* state=nullptr) {
    if(state){base=state->base;old_end+=base;}
    const int pos=verdict->next_position,carry=pos%kp;
    if(!verdict->count||pos==old_end||carry==0||pos/kp==old_end/kp)return;
    const int offset=(pos/kp-base/kp)*kp*2*kd;
    for(int i=blockIdx.x*blockDim.x+threadIdx.x;i<carry*2*kd;i+=gridDim.x*blockDim.x)vg[i]=saved[offset+i];
}
__global__ void commit_counters_k(int* counters,int kp,const RoundVerdict* verdict) {
    if(!threadIdx.x && verdict->count){counters[0]=verdict->next_position%kp;counters[1]=verdict->next_position/kp;}
}

inline int grid_for(long long n) { return (int) ((n + kBlock - 1) / kBlock); }

/// This device's SM count, once. `score_k`'s grid is a multiple of it rather than of the pool count, so the
/// launch does not move with the position - see the note in that kernel.
inline int idx_sm_count() {
    static const int n = [] {
        int dev = 0, sms = 0;
        if (cudaGetDevice(&dev) != cudaSuccess) return 1;
        if (cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, dev) != cudaSuccess) return 1;
        return sms > 0 ? sms : 1;
    }();
    return n;
}

/// Use fixed grids and device counters so the same graph can replay pooling at a new position.
__device__ inline int idx_carry(const int32_t* cnt) { return cnt[0]; }
__device__ inline int idx_npool(const int32_t* cnt) { return cnt[1]; }
__device__ inline int idx_nnew(const int32_t* cnt, int n_tok, int kpool) {
    return (idx_carry(cnt) + n_tok) / kpool;
}

/// The unfinished group's rows, from the layer's own store into the front of `vgs`.
__global__ void carry_load_k(const float* __restrict__ vg, float* __restrict__ vgs, int key_dim, int kpool,
                             const int32_t* __restrict__ cnt) {
    const long long n = (long long) 2 * key_dim * idx_carry(cnt);
    for (long long i = (long long) blockIdx.x * kBlock + threadIdx.x; i < n;
         i += (long long) gridDim.x * kBlock)
        vgs[i] = vg[i];
}

// A fixed grid copies only the live carry + window rows. Both the copy extent and its
// position change between graph replays, so neither can be a captured host memcpy size.
__global__ void save_window_k(const float* src, float* dst, int kd, int nt, const int32_t* cnt) {
    const long long n = (long long) 2 * kd * (idx_carry(cnt) + nt);
    for (long long i = (long long) blockIdx.x * kBlock + threadIdx.x; i < n;
         i += (long long) gridDim.x * kBlock) dst[i] = src[i];
}

/// Pack `value` and `gate` into the `[2 * key_dim, n]` column layout `pool_k` reads, after the carry.
__global__ void pack_vg(const float* __restrict__ v, const float* __restrict__ g, float* __restrict__ out,
                        int key_dim, int n, const int32_t* __restrict__ cnt, bool reference) {
    const long long base = (long long) 2 * key_dim * idx_carry(cnt);
    const long long total = (long long) key_dim * n;
    for (long long i = (long long) blockIdx.x * kBlock + threadIdx.x; i < total;
         i += (long long) gridDim.x * kBlock) {
        const int d = (int) (i % key_dim), tk = (int) (i / key_dim);
        out[base + (std::size_t)tk * 2 * key_dim + d] = reference ? __half2float(__float2half_rn(v[i])) : v[i];
        out[base + (std::size_t)tk * 2 * key_dim + key_dim + d] =
            reference ? __half2float(__float2half_rn(g[i])) : g[i];
    }
}

/// Launch to reserved capacity and check the live group count on device.
__global__ void pool_k(const float* __restrict__ vgs, const float* __restrict__ ape, float* __restrict__ out,
                       int key_dim, int kpool, int n_tok, const int32_t* __restrict__ cnt) {
    const int g = (int) blockIdx.x;
    const int d = (int) threadIdx.x;
    if (d >= key_dim) return;
    if (g >= idx_nnew(cnt, n_tok, kpool)) {
        out[(std::size_t) g * key_dim + d] = 0.f;
        return;
    }
    const int row = 2 * key_dim;

    // softmax over the kpool members, independently per dimension
    float mx = -INFINITY;
    for (int k = 0; k < kpool; ++k) {
        const float l = vgs[(std::size_t) (kpool * g + k) * row + key_dim + d] + ape[(std::size_t) k * key_dim + d];
        mx = l > mx ? l : mx;
    }
    float sum = 0.f, acc = 0.f;
    for (int k = 0; k < kpool; ++k) {
        const float l = vgs[(std::size_t) (kpool * g + k) * row + key_dim + d] + ape[(std::size_t) k * key_dim + d];
        const float e = __expf(l - mx);
        sum += e;
        acc = __fmaf_rn(e, vgs[(std::size_t) (kpool * g + k) * row + d], acc);
    }
    out[(std::size_t) g * key_dim + d] = acc / sum;
}

// Match the four-row softmax and the pairwise sum used by GGML.
__global__ void reference_pool_four(const float* input, const float* ape, float* out, int kd, const int32_t* cnt,
                                    int nt) {
    const int group = blockIdx.x, channel = threadIdx.x;
    if (channel >= kd) return;
    if (group >= idx_nnew(cnt, nt, 4)) {
        out[size_t(group)*kd+channel]=0.f;
        return;
    }
    float logits[4], values[4];
#pragma unroll
    for (int j = 0; j < 4; j++) {
        const float* row = input + size_t(4 * group + j) * 2 * kd;
        logits[j] = row[kd + channel] + ape[j * kd + channel];
        values[j] = row[channel];
    }
    float mx = fmaxf(fmaxf(logits[0], logits[2]), fmaxf(logits[1], logits[3]));
    for (int j = 0; j < 4; j++)
        logits[j] = __expf(logits[j] - mx);
    const float sum = (logits[0] + logits[2]) + (logits[1] + logits[3]);
    const float inv = __fdividef(1.f, sum);
    for (int j = 0; j < 4; j++)
        values[j] = __fmul_rn(__fmul_rn(logits[j], inv), values[j]);
    out[size_t(group) * kd + channel] =
        __fadd_rn(__fadd_rn(values[0], values[2]), __fadd_rn(values[1], values[3]));
}

/// The first member's position for each group formed in this call: `kpool * (n_pool + g)`. A dead slot gets
/// 0, which is a position the rope accepts and nothing reads.
__global__ void pool_pos(int32_t* __restrict__ pos, int kpool, int max_new, int n_tok,
                         const int32_t* __restrict__ cnt) {
    const int g = (int) blockIdx.x * kBlock + (int) threadIdx.x;
    if (g >= max_new) return;
    pos[g] = g < idx_nnew(cnt, n_tok, kpool) ? (int32_t) (kpool * (idx_npool(cnt) + g)) : 0;
}

/// Append completed keys after the previously pooled keys.
__global__ void commit_pooled_k(const float* __restrict__ stage, float* __restrict__ pooled, int key_dim,
                                int max_new, int n_tok, int kpool, const int32_t* __restrict__ cnt) {
    const int n_new = idx_nnew(cnt, n_tok, kpool);
    const long long total = (long long) key_dim * max_new;
    const long long base = (long long) idx_npool(cnt) * key_dim;
    for (long long i = (long long) blockIdx.x * kBlock + threadIdx.x; i < total;
         i += (long long) gridDim.x * kBlock) {
        if (i >= (long long) key_dim * n_new) return;
        pooled[base + i] = stage[i];
    }
}

/// What is left of the group still being filled, back into the layer's store.
__global__ void carry_store_k(const float* __restrict__ vgs, float* __restrict__ vg, int key_dim, int kpool,
                              int n_tok, const int32_t* __restrict__ cnt) {
    const int have = idx_carry(cnt) + n_tok;
    const int new_carry = have % kpool;
    const long long n = (long long) 2 * key_dim * new_carry;
    const long long from = (long long) 2 * key_dim * (have - new_carry);
    for (long long i = (long long) blockIdx.x * kBlock + threadIdx.x; i < n;
         i += (long long) gridDim.x * kBlock)
        vg[i] = vgs[from + i];
}

/// Advance the device pair, ONCE for all layers - the same rule the host mirror applies to itself.
__global__ void idx_advance_k(int32_t* __restrict__ cnt, int n_tok, int kpool) {
    if (blockIdx.x != 0 || threadIdx.x != 0) return;
    const int have = cnt[0] + n_tok;
    cnt[1] += have / kpool;
    cnt[0] = have % kpool;
}

// Each warp scores one pool, with four consecutive channels per lane.
template <int kHeads, int kDim>
__global__
__launch_bounds__(kDim) void reference_score_k(const float* __restrict__ iq, const float* __restrict__ wgt,
                                               const float* __restrict__ pooled, float* __restrict__ score,
                                               int pool_stride, int kpool, const int32_t* __restrict__ q_pos,
                                               int n_keep, bool select_tail) {
    const int t = blockIdx.y;
    const int pos = q_pos[t];
    const int visible = (pos + (select_tail ? 0 : 1)) / kpool;
    if (visible <= n_keep)
        return;
    static_assert(kDim == 32 * 4);
    const int lane = threadIdx.x % 32;
    const int warp = threadIdx.x / 32;
    const int warps_per_block = blockDim.x / 32;
    for (int j = blockIdx.x * warps_per_block + warp; j < visible; j += gridDim.x * warps_per_block) {
        const float* key = pooled + size_t(j) * kDim + 4 * lane;
        float total = 0.f;
        for (int h = 0; h < kHeads; h++) {
            const float* q = iq + (size_t(t) * kHeads + h) * kDim + 4 * lane;
            float dot = 0.f;
            for (int k = 0; k < 4; k++)
                dot = __fmaf_rn(q[k], key[k], dot);
            for (int off = 16; off; off /= 2)
                dot = __fadd_rn(dot, __shfl_xor_sync(0xffffffff, dot, off));
            if (!lane)
                total = __fmaf_rn(fmaxf(dot, 0.f), wgt[size_t(t) * kHeads + h], total);
        }
        if (!lane)
            score[size_t(t) * pool_stride + j] = total;
    }
}

/// Score each (token, pool) by the weighted sum of its rectified head dot products.
template <int kHeads, int kDim>
__global__ __launch_bounds__(kDim) void score_k(const float* __restrict__ iq, const float* __restrict__ wgt,
                                                const float* __restrict__ pooled, float* __restrict__ score,
                                                int pool_stride, int n_tok, int kpool,
                                                const int32_t* __restrict__ cnt) {
    const int t = (int) blockIdx.y;
    if (t >= n_tok) return;
    const int d = (int) threadIdx.x;
    // Grid-stride over pools to decouple launch dimensions from the active sequence length.
    const int n_live = idx_npool(cnt) + idx_nnew(cnt, n_tok, kpool);

    __shared__ float pk[kDim];
    __shared__ float red[kDim / 32];
    for (int j = (int) blockIdx.x; j < n_live; j += (int) gridDim.x) {
        __syncthreads();                       // the previous iteration's readers are done with pk/red
        pk[d] = pooled[(std::size_t) j * kDim + d];
        __syncthreads();
        float total = 0.f;
        for (int h = 0; h < kHeads; ++h) {
            float dot = iq[(std::size_t) t * kHeads * kDim + (std::size_t) h * kDim + d] * pk[d];
            for (int o = 16; o > 0; o >>= 1) dot += __shfl_down_sync(0xffffffff, dot, o);
            if ((d & 31) == 0) red[d / 32] = dot;
            __syncthreads();
            if (d == 0) {
                float s = 0.f;
                for (int w = 0; w < kDim / 32; ++w) s += red[w];
                total = __fmaf_rn(s > 0.f ? s : 0.f, wgt[(std::size_t) t * kHeads + h], total);
            }
            __syncthreads();
        }
        if (d == 0) score[(std::size_t) t * pool_stride + j] = total;
    }
}

/// Scale by `1 / sqrt(key_dim * heads)`, which is where the reference puts the indexer's score scale.
__global__ void scale_k(float* __restrict__ x, float s, long long n) {
    const long long i = (long long) blockIdx.x * kBlock + threadIdx.x;
    if (i < n) x[i] *= s;
}

/// Retain the highest-scoring visible pools and append the mandatory causal tail.
__global__ __launch_bounds__(kBlock) void select_k(const float* __restrict__ score, int32_t* __restrict__ sel,
                                                   int pool_stride, int n_sel, int kpool, int n_keep,
                                                   const int32_t* __restrict__ q_pos, bool select_tail,
                                                   bool reference) {
    const int t = (int) blockIdx.x;
    const float* sc = score + (std::size_t) t * pool_stride;
    int32_t* out = sel + (std::size_t) t * n_sel;
    // FROM `q_pos`, which is already a device array, rather than from a `base_pos` argument that moves with
    // every token and would be baked by a capture.
    const int pos = (int) q_pos[t];
    // `pos / kpool` when the tail is forced: the token's own group is reserved for it and does not compete.
    // Without a forced tail a complete group has to be reachable through coarse recall, so it becomes
    // visible as soon as it completes - `(pos + 1) / kpool`. This model sets kpool_select_tail.
    const int nv = select_tail ? pos / kpool : (pos + 1) / kpool;
    const int keep = nv < n_keep ? nv : n_keep;

    if (reference && nv <= n_keep) {
        for (int j = threadIdx.x; j < n_keep * kpool; j += kBlock)
            out[j] = j < nv * kpool ? j : -1;
        if (select_tail)
            for (int j = threadIdx.x; j < kpool; j += kBlock)
                out[n_keep * kpool + j] = j <= pos % kpool ? nv * kpool + j : -1;
        return;
    }
    // Map float to an unsigned key that compares the same way, so a bisection on the integer is a
    // bisection on the value. Scores can be negative (wgt is unconstrained), so the sign flip matters.
    auto key = [](float f) -> unsigned {
        unsigned u = __float_as_uint(f);
        return (u & 0x80000000u) ? ~u : (u | 0x80000000u);
    };

    // `lo`/`hi` stay identical across the block because every thread reads the same `cnt_s` and applies
    // the same update, so the bisection needs no broadcast of its own.
    __shared__ int cnt_s;
    unsigned lo = 0u, hi = 0xffffffffu;
    for (int it = 0; it < 32; ++it) {
        // Largest key with at least `keep` scores >= it: the kth-largest score itself.
        // Use the upper midpoint so a two-key interval always makes progress, without overflow.
        const unsigned delta = hi - lo;
        const unsigned mid = lo + (delta >> 1) + (delta & 1u);
        int c = 0;
        for (int j = (int) threadIdx.x; j < nv; j += kBlock)
            if (key(sc[j]) >= mid) ++c;
        __syncthreads();
        if (threadIdx.x == 0) cnt_s = 0;
        __syncthreads();
        atomicAdd(&cnt_s, c);
        __syncthreads();
        if (cnt_s >= keep) lo = mid; else hi = mid - 1;
        __syncthreads();
    }

    // Include all keys above the cutoff, then use index order to resolve cutoff ties.
    const unsigned cut = hi;
    __shared__ int ab[kBlock], eq[kBlock];
    const int span = (nv + kBlock - 1) / kBlock;           // CONTIGUOUS per thread, so a scan orders by j
    const int jlo = (int) threadIdx.x * span;
    const int jhi = min(nv, jlo + span);
    {
        int na = 0, ne = 0;
        for (int j = jlo; j < jhi; ++j) {
            const unsigned k = key(sc[j]);
            if (k > cut) ++na;
            else if (k == cut) ++ne;
        }
        ab[threadIdx.x] = na;
        eq[threadIdx.x] = ne;
    }
    __syncthreads();
    if (threadIdx.x == 0) {                                 // exclusive prefix, both at once
        int sa = 0, se = 0;
        for (int i = 0; i < kBlock; ++i) {
            const int na = ab[i], ne = eq[i];
            ab[i] = sa;
            eq[i] = se;
            sa += na;
            se += ne;
        }
        cnt_s = sa;                                         // total strictly above the cut
    }
    __syncthreads();
    {
        const int need_eq = keep - cnt_s;
        int at = ab[threadIdx.x] + min(eq[threadIdx.x], need_eq > 0 ? need_eq : 0);
        int erank = eq[threadIdx.x];
        for (int j = jlo; j < jhi; ++j) {
            const unsigned k = key(sc[j]);
            bool take = false;
            if (k > cut) take = true;
            else if (k == cut) {
                take = erank < need_eq;
                ++erank;
            }
            if (take && at < keep) {
                for (int kk = 0; kk < kpool; ++kk) out[at * kpool + kk] = (int32_t) (kpool * j + kk);
                ++at;
            }
        }
    }
    __syncthreads();

    // Pad absent pools with invalid indices before appending the token's own group.
    for (int s = keep * kpool + (int) threadIdx.x; s < n_keep * kpool; s += kBlock) out[s] = -1;
    if (select_tail) {
        const int n_tail = pos % kpool + 1;
        const int tail_start = kpool * (pos / kpool);
        for (int k = (int) threadIdx.x; k < kpool; k += kBlock)
            out[n_keep * kpool + k] = k < n_tail ? (int32_t) (tail_start + k) : -1;
    }
}

}  // namespace

IndexerCache::~IndexerCache() {
    for (void* p : pooled) cudaFree(p);
    for (void* p : vg) cudaFree(p);
    for (void* p : verify_vg) cudaFree(p);
    if (counters != nullptr) cudaFree(counters);
}

std::size_t indexer_cache_bytes(int n_layers, int key_dim, int kpool, int max_tokens) {
    const std::size_t max_pool = (std::size_t) ((max_tokens + kpool - 1) / kpool) + 1;
    return (std::size_t) n_layers *
           (max_pool * (std::size_t) key_dim * sizeof(float) +
            (std::size_t) 2 * key_dim * kpool * sizeof(float));
}

bool indexer_cache_init(IndexerCache& c, const std::vector<int32_t>& block_has_indexer, int key_dim,
                        int kpool, int max_tokens, std::string& err) {
    if (key_dim <= 0 || kpool <= 0 || max_tokens <= 0) {
        err = "indexer_cache_init: positive extents";
        return false;
    }
    c.ids.assign(block_has_indexer.size(), -1);
    int n = 0;
    for (std::size_t i = 0; i < block_has_indexer.size(); ++i)
        if (block_has_indexer[i] != 0) c.ids[i] = n++;
    c.n_layers = n;
    c.key_dim = key_dim;
    c.kpool = kpool;
    c.max_pool = (max_tokens + kpool - 1) / kpool + 1;
    c.n_pool = 0;
    c.carry = 0;
    c.pooled.assign((std::size_t) n, nullptr);
    c.vg.assign((std::size_t) n, nullptr);
    // Allocated and ZEROED before anything captures, like the epoch arena in patch 03: a counter a graph
    // reads has to exist and be sane at capture time, not just at replay time.
    if (cudaMalloc(&c.counters, 2 * sizeof(int32_t)) != cudaSuccess ||
        cudaMemset(c.counters, 0, 2 * sizeof(int32_t)) != cudaSuccess) {
        cudaGetLastError();
        err = "indexer_cache_init: no room for the device pool counters";
        return false;
    }
    for (int i = 0; i < n; ++i) {
        if (cudaMalloc(&c.pooled[(std::size_t) i],
                       (std::size_t) c.max_pool * key_dim * sizeof(float)) != cudaSuccess ||
            cudaMalloc(&c.vg[(std::size_t) i],
                       (std::size_t) 2 * key_dim * kpool * sizeof(float)) != cudaSuccess) {
            cudaGetLastError();
            err = "indexer_cache_init: no room for the pooled keys (" +
                  std::to_string(indexer_cache_bytes(n, key_dim, kpool, max_tokens) / (1024 * 1024)) +
                  " MiB for " + std::to_string(n) + " sparse blocks)";
            return false;
        }
    }
    return true;
}

void indexer_commit_device(IndexerCache& c,const RoundVerdict* verdict,void* stream) {
    if(c.verify_base<0||c.verify_tokens<1||c.verify_vg.size()!=c.vg.size()||!verdict||!stream)
        throw std::runtime_error("indexer device commit requires a live verification checkpoint");
    const auto s=static_cast<cudaStream_t>(stream);
    for(size_t i=0;i<c.vg.size();++i)commit_carry_k<<<(2*c.key_dim*c.kpool+255)/256,256,0,s>>>(
        static_cast<float*>(c.vg[i]),static_cast<const float*>(c.verify_vg[i]),c.key_dim,c.kpool,c.verify_base,c.valid_pos,verdict);
    commit_counters_k<<<1,1,0,s>>>(static_cast<int*>(c.counters),c.kpool,verdict);
    if(cudaGetLastError()!=cudaSuccess)throw std::runtime_error("indexer device commit launch failed");
}
void indexer_commit_loop(IndexerCache& c,const RoundVerdict* verdict,const DeviceLoopState* state,int rows,void* stream) {
    const auto s=static_cast<cudaStream_t>(stream);
    for(size_t i=0;i<c.vg.size();++i)commit_carry_k<<<(2*c.key_dim*c.kpool+255)/256,256,0,s>>>(
        static_cast<float*>(c.vg[i]),static_cast<const float*>(c.verify_vg[i]),c.key_dim,c.kpool,0,rows,verdict,state);
    commit_counters_k<<<1,1,0,s>>>(static_cast<int*>(c.counters),c.kpool,verdict);
    if(cudaGetLastError()!=cudaSuccess)throw std::runtime_error("loop indexer commit launch");
}

bool indexer_cache_rewind(IndexerCache& c, int pos) {
    if (pos == c.valid_pos) return true;
    if (pos > c.valid_pos) return false;         // forward is not a rewind; a gap cannot be filled later
    const int kp = c.kpool > 0 ? c.kpool : 1;
    const int carry = pos % kp;
    if (carry != 0 && pos / kp != c.valid_pos / kp) {
        if (c.verify_base < 0 || pos < c.verify_base || pos > c.verify_base + c.verify_tokens ||
            c.verify_vg.size() != c.vg.size()) return false;
        const std::size_t row = (std::size_t) 2 * c.key_dim * sizeof(float);
        const int from = (pos / kp - c.verify_base / kp) * kp;
        for (std::size_t i = 0; i < c.vg.size(); ++i)
            if (cudaMemcpy(c.vg[i], static_cast<char*>(c.verify_vg[i]) + from * row,
                           carry * row, cudaMemcpyDeviceToDevice) != cudaSuccess) return false;
    }
    c.n_pool = pos / kp;
    c.carry = carry;
    c.valid_pos = pos;
    // Rewind device counters with the host mirror.
    if (c.counters != nullptr) {
        const int32_t pair[2] = {(int32_t) c.carry, (int32_t) c.n_pool};
        if (cudaMemcpy(c.counters, pair, sizeof pair, cudaMemcpyHostToDevice) != cudaSuccess) {
            cudaGetLastError();
            return false;
        }
    }
    return true;
}

IndexerPlan indexer_plan(int top_k, int kpool, bool select_tail, int base_pos, int n_tok) {
    IndexerPlan p;
    if (top_k <= 0 || kpool <= 0 || n_tok <= 0) return p;
    // Reserve one group's worth of selection capacity for the tail.
    const int cap = select_tail ? top_k / kpool - 1 : top_k / kpool;
    const int last = base_pos + n_tok - 1;
    // Exclude the token's own pool from coarse recall. Its rows are added by the forced tail.
    const int nv = select_tail ? last / kpool : (last + 1) / kpool;
    p.n_top = nv < cap ? nv : cap;
    p.select_tail = select_tail;
    p.n_sel = kpool * p.n_top + (select_tail ? kpool : 0);
    // The identity below the cap: every visible pool is kept (cells 0 .. kpool*nv-1) and the forced tail is
    // cells kpool*nv .. pos, so the union is 0 .. pos - the whole causal set. Staying dense there is not an
    // approximation, it is the same function.
    p.engaged = nv > cap;
    return p;
}

std::size_t indexer_scratch_bytes(int heads, int key_dim, int kpool, int n_tok, int max_pool, int n_sel) {
    const std::size_t f = sizeof(float);
    std::size_t b = 0;
    auto add = [&b, f](std::size_t n) { b += (n * f + 15) & ~(std::size_t) 15; };
    add((std::size_t) key_dim * n_tok);                       // ik
    add((std::size_t) key_dim * n_tok);                       // iv
    add((std::size_t) key_dim * n_tok);                       // ig
    add((std::size_t) 2 * key_dim * (n_tok + kpool));         // vgs
    add((std::size_t) key_dim * ((n_tok / kpool) + 2));       // pooled_raw
    add((std::size_t) key_dim * ((n_tok / kpool) + 2));       // the rope's fixed staging, see commit_pooled_k
    add((std::size_t) heads * key_dim * n_tok);               // iq
    add((std::size_t) heads * key_dim * n_tok);               // iq roped
    add((std::size_t) heads * n_tok);                         // wgt
    add((std::size_t) max_pool * n_tok);                      // score
    b += (((std::size_t) ((n_tok / kpool) + 2) * sizeof(int32_t)) + 15) & ~(std::size_t) 15;  // pool pos
    b += 4096;  // the Q8_1 images below, rounded up with room for the block padding
    b += (std::size_t) kernels::native_q8_1_bytes_wide(key_dim, n_tok);
    (void) n_sel;
    return b;
}

void indexer_advance(IndexerCache& c, int n_tok, void* stream) {
    if (c.counters == nullptr || c.kpool <= 0 || n_tok <= 0) return;
    // Queue the counter update on the stream so it participates in graph capture.
    idx_advance_k<<<1, 1, 0, static_cast<cudaStream_t>(stream)>>>((int32_t*) c.counters, n_tok, c.kpool);
}

void indexer_select(IndexerCache& c, int layer, const IndexerWeights& w, const IndexerPlan& plan,
                    const void* cur_q8, const float* cur_f16, const void* q_resid_q8, const float* q_resid_f16,
                    MatCtx* mc, int n_embd, int q_lora, int heads, int n_rot, int n_tok, int base_pos,
                    const int32_t* q_pos, float freq_base, float rms_eps, int32_t* sel_cells, void* scratch,
                    void* stream, ImageRope image, float* window_save) {
    if (layer < 0 || layer >= c.n_layers) throw std::invalid_argument("indexer_select: layer out of range");
    if (cur_q8 == nullptr || q_resid_q8 == nullptr || scratch == nullptr || stream == nullptr)
        throw std::invalid_argument("indexer_select: a stream and non-null inputs");
    const int kd = c.key_dim, kp = c.kpool;
    cudaStream_t s = static_cast<cudaStream_t>(stream);

    char* p = static_cast<char*>(scratch);
    auto take = [&p](std::size_t bytes) {
        void* at = p;
        p += (bytes + 15) & ~(std::size_t) 15;
        return at;
    };
    float* ik = (float*) take((std::size_t) kd * n_tok * sizeof(float));
    float* iv = (float*) take((std::size_t) kd * n_tok * sizeof(float));
    float* ig = (float*) take((std::size_t) kd * n_tok * sizeof(float));
    float* vgs = (float*) take((std::size_t) 2 * kd * (n_tok + kp) * sizeof(float));
    float* praw = (float*) take((std::size_t) kd * ((n_tok / kp) + 2) * sizeof(float));
    float* pstage = (float*) take((std::size_t) kd * ((n_tok / kp) + 2) * sizeof(float));
    float* iq = (float*) take((std::size_t) heads * kd * n_tok * sizeof(float));
    float* iqr = (float*) take((std::size_t) heads * kd * n_tok * sizeof(float));
    float* wgt = (float*) take((std::size_t) heads * n_tok * sizeof(float));
    float* score = (float*) take((std::size_t) c.max_pool * n_tok * sizeof(float));
    int32_t* ppos = (int32_t*) take((std::size_t) ((n_tok / kp) + 2) * sizeof(int32_t));
    void* ikq8 = take((std::size_t) kernels::native_q8_1_bytes_wide(kd, n_tok) + 2048);

    // Derive pooled values and gates from the normalized key projection.
    const MatIn cur_in = mat_staged(mc, const_cast<float*>(cur_f16), cur_q8, n_embd, n_tok, stream, kSiteQA);
    mat_apply(cur_in, w.attn_k_type, w.attn_k, ik, n_embd, kd, n_tok, stream, kSiteQA);
    kernels::rms_norm_weighted_wide(ik, w.k_norm, n_tok, kd, rms_eps, stream);
    const bool reference = reference_ops_path() != nullptr;
    using MatFn = void (*)(const void*, int, const float*, float*, int, int, int, void*);
    static auto mm = reference_ops_path() ? reference_op<MatFn>("hy4_flash_reference_mm") : nullptr;
    if (reference && n_tok > 1) {
        mm(w.value, w.value_type, ik, iv, kd, kd, n_tok, stream);
        mm(w.gate, w.gate_type, ik, ig, kd, kd, n_tok, stream);
    } else {
        kernels::native_quantize_q8_1_wide(ik, ikq8, kd, n_tok, stream);
        const MatIn ik_in = mat_staged(mc, w.value_type == 30 || w.gate_type == 30 ? ik : nullptr, ikq8, kd,
                                       n_tok, stream, kSiteQA);
        mat_apply(ik_in, w.value_type, w.value, iv, kd, kd, n_tok, stream, kSiteQA);
        mat_apply(ik_in, w.gate_type, w.gate, ig, kd, kd, n_tok, stream, kSiteQA);
    }

    // Complete pooled keys even when the current sequence is too short to use sparse selection.
    const int32_t* cnt = (const int32_t*) c.counters;
    const int max_new = n_tok / kp + 1;        // the most groups a chunk of n_tok can complete
    carry_load_k<<<grid_for((long long) 2 * kd * kp), kBlock, 0, s>>>((const float*) c.vg[(std::size_t) layer],
                                                                     vgs, kd, kp, cnt);
    pack_vg<<<grid_for((long long)kd * n_tok), kBlock, 0, s>>>(iv, ig, vgs, kd, n_tok, cnt, reference);
    if (c.verify_capacity && n_tok <= c.verify_capacity && !c.verify_vg.empty())
        cudaMemcpyAsync(c.verify_vg[(std::size_t) layer], vgs,
            (std::size_t) 2 * kd * (n_tok + base_pos % kp) * sizeof(float), cudaMemcpyDeviceToDevice, s);
    if (window_save != nullptr)
        save_window_k<<<grid_for((long long) 2 * kd * (kp - 1 + n_tok)), kBlock, 0, s>>>(
            vgs, window_save, kd, n_tok, cnt);
    if (reference && kp != 4)
        throw std::invalid_argument("HY4 reference pooling requires groups of four");
    if (reference)
        reference_pool_four<<<max_new, kd, 0, s>>>(vgs, w.ape, praw, kd, cnt, n_tok);
    else
        pool_k<<<max_new, kd, 0, s>>>(vgs, w.ape, praw, kd, kp, n_tok, cnt);
    kernels::rms_norm_weighted_wide(praw, w.compressor_norm, max_new, kd, rms_eps, stream);
    pool_pos<<<grid_for(max_new), kBlock, 0, s>>>(ppos, kp, max_new, n_tok, cnt);
    // Rotate only the tail of the 128-channel head.
    rope_norm_head_tail(praw, pstage, kd, n_rot, 1, max_new, ppos, freq_base, stream, image);
    commit_pooled_k<<<grid_for((long long)kd * max_new), kBlock, 0, s>>>(
        pstage, (float*)c.pooled[(std::size_t)layer], kd, max_new, n_tok, kp, cnt);
    carry_store_k<<<grid_for((long long) 2 * kd * kp), kBlock, 0, s>>>(vgs, (float*) c.vg[(std::size_t) layer],
                                                                      kd, kp, n_tok, cnt);
    // The caller advances both counters once for all layers; this one only writes its own rows.

    if (!plan.engaged) return;

    // ---- the query half, and the score.
    const MatIn qr_in = mat_staged(mc, const_cast<float*>(q_resid_f16), q_resid_q8, q_lora, n_tok, stream,
                                   kSiteQB);
    mat_apply(qr_in, w.attn_q_b_type, w.attn_q_b, iq, q_lora, heads * kd, n_tok, stream, kSiteQB);
    // The same layout rule as the pooled key, and it has to be: the score is their dot product, so a query
    // roped into different channels from the key is a different function, not a different rounding.
    rope_norm_head_tail(iq, iqr, kd, n_rot, heads, n_tok, q_pos, freq_base, stream, image);
    if (reference && n_tok > 1)
        mm(w.proj, w.proj_type, cur_f16, wgt, n_embd, heads, n_tok, stream);
    else
        mat_apply(cur_in, w.proj_type, w.proj, wgt, n_embd, heads, n_tok, stream, kSiteQA);
    scale_k<<<grid_for((long long) heads * n_tok), kBlock, 0, s>>>(
        wgt, 1.f / std::sqrt((float) (kd * heads)), (long long) heads * n_tok);

    if (heads != 16 || kd != 128)
        throw std::invalid_argument("indexer_select: the score kernel is specialised for 16 heads of 128");
    // Grid-stride over the live pool count read on device.
    const int score_blocks = std::min(2 * idx_sm_count(), c.max_pool);
    dim3 g((unsigned) (score_blocks > 0 ? score_blocks : 1), (unsigned) n_tok);
    if (reference)
        reference_score_k<16, 128><<<g, 128, 0, s>>>(iqr, wgt, (const float*)c.pooled[layer], score, c.max_pool,
                                                     kp, q_pos, plan.n_top, plan.select_tail);
    else
        score_k<16, 128><<<g, 128, 0, s>>>(iqr, wgt, (const float*)c.pooled[(std::size_t)layer], score,
                                           c.max_pool, n_tok, kp, cnt);
    select_k<<<n_tok, kBlock, 0, s>>>(score, sel_cells, c.max_pool, plan.n_sel, kp, plan.n_top, q_pos,
                                      plan.select_tail, reference);
}

bool indexer_restore_window(IndexerCache& c, const std::vector<void*>& windows, int win_start, int end_pos,
                            int new_len, std::string& err) {
    const int kp = c.kpool > 0 ? c.kpool : 1;
    if (c.counters == nullptr || (int) windows.size() != c.n_layers) {
        err = "indexer_restore_window: windows do not match the cache's layers";
        return false;
    }
    if (win_start % kp != 0 || new_len > end_pos || new_len < win_start || c.valid_pos != end_pos) {
        err = "indexer_restore_window: new_len " + std::to_string(new_len) + " outside window [" +
              std::to_string(win_start) + "," + std::to_string(end_pos) + "] or cache at " +
              std::to_string(c.valid_pos);
        return false;
    }
    const int g = new_len / kp, carry = new_len % kp;
    const int first = kp * g;                                 // position of the in-flight group's row 0
    if (first < win_start) {                                  // cannot happen with new_len >= win_start
        err = "indexer_restore_window: the in-flight group starts before the window";
        return false;
    }
    const std::size_t row = (std::size_t) 2 * c.key_dim;
    if (carry > 0) {
        for (int l = 0; l < c.n_layers; ++l) {
            if (windows[(std::size_t) l] == nullptr || c.vg[(std::size_t) l] == nullptr) {
                err = "indexer_restore_window: a layer has no window";
                return false;
            }
            const float* src = (const float*) windows[(std::size_t) l] + (std::size_t) (first - win_start) * row;
            if (cudaMemcpy(c.vg[(std::size_t) l], src, (std::size_t) carry * row * sizeof(float),
                           cudaMemcpyDeviceToDevice) != cudaSuccess) {
                cudaGetLastError();
                err = "indexer_restore_window: a layer's rows could not be restored";
                return false;
            }
        }
    }
    const int32_t pair[2] = {(int32_t) carry, (int32_t) g};
    if (cudaMemcpy(c.counters, pair, sizeof pair, cudaMemcpyHostToDevice) != cudaSuccess ||
        cudaDeviceSynchronize() != cudaSuccess) {
        cudaGetLastError();
        err = "indexer_restore_window: the device counters could not be moved";
        return false;
    }
    c.n_pool = g;
    c.carry = carry;
    c.valid_pos = new_len;
    return true;
}

bool indexer_force_reset(IndexerCache& c, std::string& err) {
    if (c.counters == nullptr) {
        err = "indexer_force_reset: no device counters";
        return false;
    }
    const int32_t pair[2] = {0, 0};
    if (cudaDeviceSynchronize() != cudaSuccess ||
        cudaMemcpy(c.counters, pair, sizeof pair, cudaMemcpyHostToDevice) != cudaSuccess ||
        cudaDeviceSynchronize() != cudaSuccess) {
        cudaGetLastError();
        err = "indexer_force_reset: the device counters could not be written";
        return false;
    }
    c.n_pool = 0;
    c.carry = 0;
    c.valid_pos = 0;
    return true;
}

}  // namespace strata::hy4_flash
