// src/kernels/cuda/qsa_decode_attn.cu - see include/strata/kernels/qsa_decode_attn.hpp.
#include "strata/kernels/qsa_decode_attn.hpp"
#include "strata/kernels/kv_q8.hpp"

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cfloat>
#include <cstdio>
#include <cstdlib>

namespace strata::kernels {
namespace {

constexpr int HD = 256;          // head_dim
constexpr int G = 12;            // query heads per KV head (24 / 2)
constexpr int THREADS = 256;
constexpr int WARPS = THREADS / 32;
constexpr int PT = 64;           // cells per tile
constexpr int SPLIT_MAX = 20;    // a decode query's splits at most

__device__ __forceinline__ float warp_sum(float v) {
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
    return v;
}
__device__ __forceinline__ float warp_max(float v) {
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) v = fmaxf(v, __shfl_xor_sync(0xffffffffu, v, o));
    return v;
}

// The decode's partial results, per query `stride` floats: [split][head][HD] sums of p * v (p relative to the split's
// maximum), then at `m` and `l` (offsets from `acc`) [split][head] maxima and sums of p.
struct Partials {
    float* acc = nullptr;
    long long m = 0, l = 0, stride = 0;
    int splits = 0;
};
// the cells of each of a query's splits: split k takes [k c, min((k + 1) c, n)), whole tiles while the cells last
__host__ __device__ __forceinline__ int split_cells(int n, int splits) {
    const int c = (n + splits - 1) / splits;
    return c > PT ? c : PT;
}

// ---- FP16 pools: a block per (KV head, query) walks all of the query's cells in tiles, online softmax; SPLIT: a
// block per (KV head, query, split) its split's cells, writing partials.  Scores: thread (cell, quarter) dots its
// cell's 64 dimensions of that quarter with the 12 query heads, the quarters summed through shared memory.  Values:
// thread (dimension pair, cell parity) accumulates its two dimensions of the 12 heads over every other cell; the
// parities are added at the end.
template <bool SPLIT>
__global__ void __launch_bounds__(THREADS) attn_f16_kernel(const float* __restrict__ q, QsaAttnPools p,
                                                           const int32_t* __restrict__ ids,
                                                           const int32_t* __restrict__ steps, int n_kv_heads,
                                                           int page_size, float scale, float* __restrict__ attn,
                                                           int cap, Partials part) {
    const int kvh = blockIdx.x;
    const size_t qi = blockIdx.y;
    const size_t head0 = qi * (size_t) (n_kv_heads * G) + (size_t) kvh * G;
    q += head0 * HD;
    if constexpr (!SPLIT) attn += head0 * HD;
    ids += qi * (size_t) cap;
    int n_ids = __ldg(steps + qi * kStepCount + kStepWidth);
    if constexpr (SPLIT) {
        const int c = split_cells(n_ids, part.splits), c0 = (int) blockIdx.z * c;
        if (c0 >= n_ids) return;
        ids += c0;
        n_ids = min(c, n_ids - c0);
    }
    __shared__ __align__(16) float sq[G][HD];      // 12 KB: this KV head's query heads
    __shared__ __align__(16) float spart[4][G][PT];  // 12 KB: the quarters' partial scores; the parity sums at the end
    __shared__ __align__(16) float sp[PT][G];      // probabilities, cell-major
    __shared__ long long srow[PT];
    __shared__ float s_alpha[G], s_m[G], s_l[G];
    const int t = threadIdx.x, lane = t & 31, warp = t >> 5;
    for (int i = t; i < G * HD; i += THREADS) sq[i / HD][i % HD] = q[i];
    if (t < G) { s_m[t] = -FLT_MAX; s_l[t] = 0.0f; }
    const int quarter = warp >> 1, cell = ((warp & 1) << 5) | lane;   // scores
    const int parity = t >> 7, d0 = 2 * (t & 127);                     // values
    float acc0[G], acc1[G];
#pragma unroll
    for (int h = 0; h < G; ++h) acc0[h] = acc1[h] = 0.0f;
    for (int c0 = 0; c0 < n_ids; c0 += PT) {
        const int n_here = min(PT, n_ids - c0);
        if (t < PT) {
            long long r = -1;
            if (t < n_here) {
                const int id = ids[c0 + t];
                r = ((long long) p.page_table[id / page_size] * n_kv_heads + kvh) * page_size + id % page_size;
            }
            srow[t] = r;
        }
        __syncthreads();
        float s[G];
#pragma unroll
        for (int h = 0; h < G; ++h) s[h] = 0.0f;
        if (cell < n_here) {
            const long long row = srow[cell];
#pragma unroll
            for (int j = 0; j < 64; j += 16) {
                const int d = quarter * 64 + j;
                float kf[16];
                const uint4* src = reinterpret_cast<const uint4*>(p.k_pool + row * HD + d);
#pragma unroll
                for (int u = 0; u < 2; ++u) {
                    const uint4 raw = src[u];
                    const __half2* h2 = reinterpret_cast<const __half2*>(&raw);
#pragma unroll
                    for (int v = 0; v < 4; ++v) {
                        const float2 f = __half22float2(h2[v]);
                        kf[8 * u + 2 * v] = f.x;
                        kf[8 * u + 2 * v + 1] = f.y;
                    }
                }
#pragma unroll
                for (int h = 0; h < G; ++h) {
                    const float4* qh = reinterpret_cast<const float4*>(&sq[h][d]);
                    float a = s[h];
#pragma unroll
                    for (int v = 0; v < 4; ++v) {
                        const float4 q4 = qh[v];
                        a = fmaf(kf[4 * v], q4.x, a);
                        a = fmaf(kf[4 * v + 1], q4.y, a);
                        a = fmaf(kf[4 * v + 2], q4.z, a);
                        a = fmaf(kf[4 * v + 3], q4.w, a);
                    }
                    s[h] = a;
                }
            }
        }
#pragma unroll
        for (int h = 0; h < G; ++h) spart[quarter][h][cell] = s[h];
        __syncthreads();
        // online softmax: warp w takes heads w and w + 8, a lane cells lane and lane + 32
        for (int h = warp; h < G; h += WARPS) {
            float a = -FLT_MAX, b = -FLT_MAX;
            if (lane < n_here) a = (spart[0][h][lane] + spart[1][h][lane] + spart[2][h][lane] + spart[3][h][lane]) * scale;
            if (lane + 32 < n_here)
                b = (spart[0][h][lane + 32] + spart[1][h][lane + 32] + spart[2][h][lane + 32] + spart[3][h][lane + 32]) * scale;
            const float m_old = s_m[h];
            const float m_new = fmaxf(m_old, warp_max(fmaxf(a, b)));
            const float ea = lane < n_here ? __expf(a - m_new) : 0.0f;
            const float eb = lane + 32 < n_here ? __expf(b - m_new) : 0.0f;
            sp[lane][h] = ea;
            sp[lane + 32][h] = eb;
            const float l = warp_sum(ea + eb);
            if (lane == 0) {
                const float alpha = __expf(m_old - m_new);
                s_alpha[h] = alpha;
                s_l[h] = fmaf(s_l[h], alpha, l);
                s_m[h] = m_new;
            }
        }
        __syncthreads();
#pragma unroll
        for (int h = 0; h < G; ++h) {
            acc0[h] *= s_alpha[h];
            acc1[h] *= s_alpha[h];
        }
        for (int c = parity; c < n_here; c += 2) {
            const long long r = srow[c];
            const float2 f = __half22float2(*reinterpret_cast<const __half2*>(p.v_pool + r * HD + d0));
            const float v0 = f.x, v1 = f.y;
            const float4* pc = reinterpret_cast<const float4*>(sp[c]);
            const float4 pa = pc[0], pb = pc[1], pd = pc[2];
            const float pr[G] = {pa.x, pa.y, pa.z, pa.w, pb.x, pb.y, pb.z, pb.w, pd.x, pd.y, pd.z, pd.w};
#pragma unroll
            for (int h = 0; h < G; ++h) {
                acc0[h] = fmaf(pr[h], v0, acc0[h]);
                acc1[h] = fmaf(pr[h], v1, acc1[h]);
            }
        }
        __syncthreads();   // srow, sp and s_alpha are rewritten by the next tile
    }
    // the odd cells' sums into shared memory, added to the even ones'
    float* odd = &spart[0][0][0];   // G x HD floats
    if (parity == 1) {
#pragma unroll
        for (int h = 0; h < G; ++h) {
            odd[h * HD + d0] = acc0[h];
            odd[h * HD + d0 + 1] = acc1[h];
        }
    }
    __syncthreads();
    if constexpr (SPLIT) {
        float* base = part.acc + qi * part.stride;
        const size_t hq = (size_t) blockIdx.z * n_kv_heads * G + (size_t) kvh * G;   // the block's first head
        if (parity == 0) {
#pragma unroll
            for (int h = 0; h < G; ++h) {
                base[(hq + h) * HD + d0] = acc0[h] + odd[h * HD + d0];
                base[(hq + h) * HD + d0 + 1] = acc1[h] + odd[h * HD + d0 + 1];
            }
        }
        if (t < G) {
            base[part.m + hq + t] = s_m[t];
            base[part.l + hq + t] = s_l[t];
        }
    } else if (parity == 0) {
#pragma unroll
        for (int h = 0; h < G; ++h) {
            const float l = s_l[h];
            attn[(size_t) h * HD + d0] = l > 0.0f ? (acc0[h] + odd[h * HD + d0]) / l : 0.0f;
            attn[(size_t) h * HD + d0 + 1] = l > 0.0f ? (acc1[h] + odd[h * HD + d0 + 1]) / l : 0.0f;
        }
    }
}

// ---- INT8 pools: the codes on the tensor cores (mma m16n8k32, int8).  A block per (KV head, query), or per (KV head,
// query, split) with SPLIT, its 12 query heads the rows of the 16-row fragments, 64 cells a tile.  q and the
// probabilities enter as 24-bit fixed point, three 8-bit limbs each; the integer sums are exact, so the error is the
// fixed point's: q to 2^-23 of its row's power-of-two bound, a probability times its V scale to 2^-23 of its tile's.
// Warp-specialized, one block an SM:
//   warps 0-3   score 16 cells of each tile over all 256 dims (q's limbs in registers) and run the online softmax;
//   warps 4-7   multiply a 64-dim group of V each: the probabilities times the V scales as limbs against V through
//               ldmatrix.trans (cells and dims permuted inside the fragments, the same way on both sides);
//   warps 8-11  fetch by cp.async, 16 cells of each tile a warp: their K and V codes and scales together (with
//               K and V in warps of their own, V's copies queued behind K's, which ran ahead), the pool rows two
//               tiles ahead.
// Rings in shared memory between them (K 2 slots, V 3, probabilities 2), synchronized by named barriers.
namespace pf {
constexpr int NT = 384;
constexpr int KS = 2, VS = 3;   // slots of the K and V rings
enum : int {
    END = 0, FULL_K = 1, EMPTY_K = FULL_K + KS, FULL_V = EMPTY_K + KS, EMPTY_V = FULL_V + VS, FULL_P = EMPTY_V + VS,
    EMPTY_P = FULL_P + 2, SRED = EMPTY_P + 2
};
static_assert(SRED < 16, "sixteen named barriers");
constexpr int NKB = 128 + 128, NVB = 128 + 128, NPB = 128 + 128;   // threads at the K, V, P barriers (END as P)
static_assert(KV_Q8_GROUP == 64 && HD == 256, "four 64-dim scale groups a row");
constexpr int MAGIC = 0x4B400000;   // 1.5 * 2^23 as float bits: an int x (|x| < 2^22) added gives 1.5 * 2^23 + x
constexpr float MAGIC_F = 12582912.0f;

struct __align__(16) Smem {
    uint8_t k[KS][PT * HD];   // K codes, 16-byte chunk j of cell c at j ^ ksw(c)
    uint8_t v[VS][PT * HD];   // V codes, chunk j of cell c at j ^ (c & 7)
    uint2 ks[KS][PT];         // FP16 scales of a cell's four dim groups
    uint2 vs[VS][PT];
    float p[2][16][PT];       // probabilities, float4 i of row r at i ^ ((r & 1) << 2)
    float alpha[2][16];       // a row's rescale of the running sums, and the tile's largest probability
    float pmax[2][16];
    float red[2][4][16];      // the score warps' tile maxima
    float lsum[4][16];
    float qmax[16], qsc[16];
    float mfin[16];           // SPLIT: the rows' final maxima
};

// Score warp w's B fragments: column j of its n-tile n is cell 16w + 4 (j >> 1) + 2n + (j & 1), so a lane's C
// fragments hold four consecutive cells; this swizzle puts the eight cells an ldmatrix reads on distinct banks.
__device__ __forceinline__ int ksw(int c) { return ((c >> 1) & 6) | (c & 1); }
__device__ __forceinline__ void bar_sync(int id, int n) {
    asm volatile("bar.sync %0, %1;\n" ::"r"(id), "r"(n) : "memory");
}
__device__ __forceinline__ void bar_arrive(int id, int n) {
    asm volatile("bar.arrive %0, %1;\n" ::"r"(id), "r"(n) : "memory");
}
// where cell c of probability row r lives in Smem::p
__device__ __forceinline__ int p_at(int r, int c) { return (((c >> 2) ^ ((r & 1) << 2)) << 2) | (c & 3); }
__device__ __forceinline__ void mma_u8s8(int (&d)[4], const unsigned (&a)[4], unsigned b0, unsigned b1) {
    asm volatile("mma.sync.aligned.m16n8k32.row.col.s32.u8.s8.s32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
                 : "+r"(d[0]), "+r"(d[1]), "+r"(d[2]), "+r"(d[3])
                 : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
}
__device__ __forceinline__ void mma_s8s8(int (&d)[4], const unsigned (&a)[4], unsigned b0, unsigned b1) {
    asm volatile("mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
                 : "+r"(d[0]), "+r"(d[1]), "+r"(d[2]), "+r"(d[3])
                 : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
}
__device__ __forceinline__ void ldsm4(unsigned (&r)[4], const void* s) {
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n"
                 : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3])
                 : "r"((unsigned) __cvta_generic_to_shared(s)));
}
__device__ __forceinline__ void ldsm4t(unsigned (&r)[4], const void* s) {
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0,%1,%2,%3}, [%4];\n"
                 : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3])
                 : "r"((unsigned) __cvta_generic_to_shared(s)));
}
__device__ __forceinline__ void cp_async16(void* s, const void* g) {
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" ::"r"((unsigned) __cvta_generic_to_shared(s)), "l"(g));
}
__device__ __forceinline__ void cp_async8(void* s, const void* g) {
    asm volatile("cp.async.ca.shared.global [%0], [%1], 8;\n" ::"r"((unsigned) __cvta_generic_to_shared(s)), "l"(g));
}
__device__ __forceinline__ int exp_of(float x) { return (__float_as_int(x) >> 23) & 255; }   // x < 2^(E - 126)
__device__ __forceinline__ float pow2f(int e) { return __int_as_float((e + 127) << 23); }     // -126 <= e <= 127
// w (65536 r2 + 256 r1 + r0) from three limbs' sums (|r| < 2^21), w a power of two and wk = -65793 C w: the first
// multiply-add is exact, the other two round
__device__ __forceinline__ float limbs3(int r0, int r1, int r2, float w65536, float w256, float w, float wk) {
    const float u0 = __int_as_float(r0 + MAGIC), u1 = __int_as_float(r1 + MAGIC), u2 = __int_as_float(r2 + MAGIC);
    return fmaf(u0, w, fmaf(u1, w256, fmaf(u2, w65536, wk)));
}
// bytes 0-2 of four values as three words: byte j of word i is byte i of value j
__device__ __forceinline__ void pack3(unsigned f0, unsigned f1, unsigned f2, unsigned f3, unsigned& w0, unsigned& w1,
                                      unsigned& w2) {
    const unsigned t01 = __byte_perm(f0, f1, 0x5140), t23 = __byte_perm(f2, f3, 0x5140);
    w0 = __byte_perm(t01, t23, 0x5410);
    w1 = __byte_perm(t01, t23, 0x7632);
    w2 = __byte_perm(__byte_perm(f0, f1, 0x0062), __byte_perm(f2, f3, 0x0062), 0x5410);
}
// x >= 0 rounded to an integer below 2^23, in the low 23 bits
__device__ __forceinline__ unsigned fx23(float x) { return __float_as_uint(fminf(x, 8388607.0f) + 8388608.0f); }
// dim group g's FP16 scale of a cell's four
__device__ __forceinline__ float scale_of(uint2 s, int g) {
    return __half2float(__ushort_as_half((unsigned short) ((g < 2 ? s.x : s.y) >> (16 * (g & 1)))));
}
}  // namespace pf

template <bool SPLIT>
__global__ void __launch_bounds__(pf::NT, 1) attn_mma_kernel(const float* __restrict__ q, QsaAttnPools p,
                                                             const int32_t* __restrict__ ids,
                                                             const int32_t* __restrict__ steps, int n_kv_heads,
                                                             int page_size, float scale, float* __restrict__ attn,
                                                             int cap, Partials part) {
    using namespace pf;
    extern __shared__ __align__(16) uint8_t smem_raw[];
    Smem& sm = *reinterpret_cast<Smem*>(smem_raw);
    const int kvh = blockIdx.x;
    const size_t qi = blockIdx.y;
    const size_t head0 = qi * (size_t) (n_kv_heads * G) + (size_t) kvh * G;
    q += head0 * HD;
    if constexpr (!SPLIT) attn += head0 * HD;
    ids += qi * (size_t) cap;
    int n_ids = __ldg(steps + qi * kStepCount + kStepWidth);
    if constexpr (SPLIT) {
        const int c = split_cells(n_ids, part.splits), c0 = (int) blockIdx.z * c;
        if (c0 >= n_ids) return;
        ids += c0;
        n_ids = min(c, n_ids - c0);
    }
    const int n_tiles = (n_ids + PT - 1) / PT;
    const int t = threadIdx.x, lane = t & 31, warp = t >> 5, gid = lane >> 2, tig = lane & 3;
    const float kS = -65793.0f * MAGIC_F;

    // each query row's largest magnitude, and its scale 2^(e - 23) (|q| < 2^e) times the softmax scale
    if (t < 256) {
        const int r = t >> 4, ch = t & 15;
        float mx = 0.0f;
        if (r < G)
#pragma unroll
            for (int i = 0; i < 16; i += 4) {
                const float4 f = *reinterpret_cast<const float4*>(q + (size_t) r * HD + ch * 16 + i);
                mx = fmaxf(mx, fmaxf(fmaxf(fabsf(f.x), fabsf(f.y)), fmaxf(fabsf(f.z), fabsf(f.w))));
            }
#pragma unroll
        for (int o = 8; o > 0; o >>= 1) mx = fmaxf(mx, __shfl_xor_sync(0xffffffffu, mx, o));
        if (ch == 0) {
            sm.qmax[r] = mx;
            sm.qsc[r] = mx > 0.0f ? pow2f(exp_of(mx) - 126 - 23) * scale : 0.0f;
        }
    }
    __syncthreads();
    if (!SPLIT && n_tiles == 0) {
        for (int i = t; i < G * HD; i += NT) attn[i] = 0.0f;
        return;
    }

    if (warp >= 8) {
        // ======== fetch, K and V together: warp 8 + f takes cells 16f .. 16f+15 of each tile (lane l & 15 looks up
        // a cell's row); tile i is issued, then tile i-1 signalled once its copies have landed
        const int f = warp - 8, cl = 16 * f + (lane & 15);
        auto row_of = [&](int id, int pg) { return (pg * n_kv_heads + kvh) * page_size + id % page_size; };
        int row = -1, id1 = 0, pg1 = 0, id2 = 0;
        if (cl < n_ids) {
            const int id = ids[cl];
            row = row_of(id, p.page_table[id / page_size]);
        }
        if (PT + cl < n_ids) {
            id1 = ids[PT + cl];
            pg1 = p.page_table[id1 / page_size];
        }
        if (2 * PT + cl < n_ids) id2 = ids[2 * PT + cl];
        for (int i = 0; i < n_tiles; ++i) {
            const int sk = i % KS, sv = i % VS, c0 = i * PT, n_here = min(PT, n_ids - c0);
            if (i >= KS) bar_sync(EMPTY_K + sk, NKB);
            if (i >= VS) bar_sync(EMPTY_V + sv, NVB);
#pragma unroll
            for (int u = 0; u < 8; ++u) {
                const int e = lane + 32 * u, cw = e >> 4, j = e & 15, c = 16 * f + cw;
                const int r = __shfl_sync(0xffffffffu, row, cw);
                if (c < n_here) {
                    cp_async16(&sm.k[sk][c * HD + ((j ^ ksw(c)) << 4)], p.k_q + (size_t) r * HD + j * 16);
                    cp_async16(&sm.v[sv][c * HD + ((j ^ (c & 7)) << 4)], p.v_q + (size_t) r * HD + j * 16);
                }
            }
            const bool vl = lane >= 16;                        // lanes 0-15 a cell's K scales, 16-31 its V scales
            uint2* sd = vl ? &sm.vs[sv][cl] : &sm.ks[sk][cl];
            if (cl < n_here)
                cp_async8(sd, (vl ? p.v_scale : p.k_scale) + (size_t) row * (HD / KV_Q8_GROUP));
            else *sd = make_uint2(0u, 0u);   // no scale for the multiply of a missing cell's stale codes
            asm volatile("cp.async.commit_group;\n" ::);
            if (i >= 1) {
                asm volatile("cp.async.wait_group 1;\n" ::);
                bar_arrive(FULL_K + (i - 1) % KS, NKB);
                bar_arrive(FULL_V + (i - 1) % VS, NVB);
            }
            row = cl < n_ids - c0 - PT ? row_of(id1, pg1) : -1;
            id1 = id2;
            pg1 = c0 + 2 * PT + cl < n_ids ? p.page_table[id2 / page_size] : 0;
            id2 = c0 + 3 * PT + cl < n_ids ? ids[c0 + 3 * PT + cl] : 0;
        }
        asm volatile("cp.async.wait_group 0;\n" ::);
        bar_arrive(FULL_K + (n_tiles - 1) % KS, NKB);
        bar_arrive(FULL_V + (n_tiles - 1) % VS, NVB);
    } else if (warp < 4) {
        // ======== scores: cells 16w + 4tig .. +3 of each tile in this lane's C fragments (rows gid, gid + 8)
        const int w = warp;
        unsigned qa[3][8][4];   // q's limbs as A fragments: [limb][k-step][register]
        {
            float inv[2];
#pragma unroll
            for (int r = 0; r < 2; ++r) {
                const float mx = sm.qmax[gid + 8 * r];
                inv[r] = mx > 0.0f ? pow2f(23 - (exp_of(mx) - 126)) : 0.0f;
            }
#pragma unroll
            for (int ks = 0; ks < 8; ++ks)
#pragma unroll
                for (int r = 0; r < 2; ++r)
#pragma unroll
                    for (int hf = 0; hf < 2; ++hf) {
                        const int row = gid + 8 * r;
                        float4 f = make_float4(0.f, 0.f, 0.f, 0.f);
                        if (row < G)
                            f = *reinterpret_cast<const float4*>(q + (size_t) row * HD + 32 * ks + 16 * hf + 4 * tig);
                        const float fv[4] = {f.x, f.y, f.z, f.w};
                        unsigned u[4];
#pragma unroll
                        for (int e = 0; e < 4; ++e)
                            u[e] = (unsigned) max(-8388607, min(8388607, __float2int_rn(fv[e] * inv[r])));
                        pack3(u[0], u[1], u[2], u[3], qa[0][ks][r + 2 * hf], qa[1][ks][r + 2 * hf], qa[2][ks][r + 2 * hf]);
                    }
        }
        const float qs[2] = {sm.qsc[gid], sm.qsc[gid + 8]};
        float m_run[2] = {-FLT_MAX, -FLT_MAX}, lp[2] = {0.0f, 0.0f};
        for (int i = 0; i < n_tiles; ++i) {
            const int s = i % KS, sp = i & 1, n_here = min(PT, n_ids - i * PT);
            bar_sync(FULL_K + s, NKB);
            uint2 kr[4];
#pragma unroll
            for (int e = 0; e < 4; ++e) kr[e] = sm.ks[s][16 * w + 4 * tig + e];
            float sv[2][4] = {{0.f, 0.f, 0.f, 0.f}, {0.f, 0.f, 0.f, 0.f}};
#pragma unroll
            for (int g = 0; g < 4; ++g) {
                unsigned b[2][4];
#pragma unroll
                for (int n = 0; n < 2; ++n) {
                    const int cb = 16 * w + 4 * ((lane & 7) >> 1) + 2 * n + (lane & 1);
                    ldsm4(b[n], &sm.k[s][cb * HD + (((4 * g + (lane >> 3)) ^ ksw(cb)) << 4)]);
                }
                int acc[3][2][4];
#pragma unroll
                for (int j = 0; j < 3; ++j)
#pragma unroll
                    for (int n = 0; n < 2; ++n)
#pragma unroll
                        for (int ii = 0; ii < 4; ++ii) acc[j][n][ii] = 0;
#pragma unroll
                for (int kk = 0; kk < 2; ++kk)
#pragma unroll
                    for (int j = 0; j < 3; ++j)
#pragma unroll
                        for (int n = 0; n < 2; ++n) {
                            if (j < 2) mma_u8s8(acc[j][n], qa[j][2 * g + kk], b[n][2 * kk], b[n][2 * kk + 1]);
                            else mma_s8s8(acc[j][n], qa[j][2 * g + kk], b[n][2 * kk], b[n][2 * kk + 1]);
                        }
#pragma unroll
                for (int n = 0; n < 2; ++n)
#pragma unroll
                    for (int ii = 0; ii < 4; ++ii)
                        sv[n][ii] = fmaf(limbs3(acc[0][n][ii], acc[1][n][ii], acc[2][n][ii], 65536.0f, 256.0f, 1.0f, kS),
                                         scale_of(kr[2 * n + (ii & 1)], g), sv[n][ii]);
            }
            if (i + KS < n_tiles) bar_arrive(EMPTY_K + s, NKB);
            // online softmax: the tile's maximum over the four warps
            float mx[2] = {-FLT_MAX, -FLT_MAX};
#pragma unroll
            for (int n = 0; n < 2; ++n)
#pragma unroll
                for (int ii = 0; ii < 4; ++ii) {
                    const int r = ii >> 1, cell = 16 * w + 4 * tig + 2 * n + (ii & 1);
                    const float v = cell < n_here ? sv[n][ii] * qs[r] : -FLT_MAX;
                    sv[n][ii] = v;
                    mx[r] = fmaxf(mx[r], v);
                }
#pragma unroll
            for (int r = 0; r < 2; ++r) {
                mx[r] = fmaxf(mx[r], __shfl_xor_sync(0xffffffffu, mx[r], 1));
                mx[r] = fmaxf(mx[r], __shfl_xor_sync(0xffffffffu, mx[r], 2));
            }
            if (tig == 0) { sm.red[sp][w][gid] = mx[0]; sm.red[sp][w][gid + 8] = mx[1]; }
            bar_sync(SRED, 128);
            float al[2], pm[2], pv[2][4];
#pragma unroll
            for (int r = 0; r < 2; ++r) {
                const int row = gid + 8 * r;
                const float tmax = fmaxf(fmaxf(sm.red[sp][0][row], sm.red[sp][1][row]),
                                         fmaxf(sm.red[sp][2][row], sm.red[sp][3][row]));
                const float m_new = fmaxf(m_run[r], tmax);
                al[r] = __expf(m_run[r] - m_new);
                pm[r] = __expf(tmax - m_new);
                m_run[r] = m_new;
            }
#pragma unroll
            for (int n = 0; n < 2; ++n)
#pragma unroll
                for (int ii = 0; ii < 4; ++ii) {
                    const int r = ii >> 1, cell = 16 * w + 4 * tig + 2 * n + (ii & 1);
                    pv[r][2 * n + (ii & 1)] = cell < n_here ? __expf(sv[n][ii] - m_run[r]) : 0.0f;
                }
#pragma unroll
            for (int r = 0; r < 2; ++r) lp[r] = fmaf(lp[r], al[r], (pv[r][0] + pv[r][1]) + (pv[r][2] + pv[r][3]));
            if (i >= 2) bar_sync(EMPTY_P + sp, NPB);
#pragma unroll
            for (int r = 0; r < 2; ++r) {
                const int row = gid + 8 * r;
                *reinterpret_cast<float4*>(&sm.p[sp][row][p_at(row, 16 * w + 4 * tig)]) =
                    make_float4(pv[r][0], pv[r][1], pv[r][2], pv[r][3]);
            }
            if (w == 0 && tig == 0) {
                sm.alpha[sp][gid] = al[0]; sm.alpha[sp][gid + 8] = al[1];
                sm.pmax[sp][gid] = pm[0]; sm.pmax[sp][gid + 8] = pm[1];
            }
            bar_arrive(FULL_P + sp, NPB);
        }
#pragma unroll
        for (int r = 0; r < 2; ++r) {
            lp[r] += __shfl_xor_sync(0xffffffffu, lp[r], 1);
            lp[r] += __shfl_xor_sync(0xffffffffu, lp[r], 2);
        }
        if (tig == 0) { sm.lsum[w][gid] = lp[0]; sm.lsum[w][gid + 8] = lp[1]; }
        if (SPLIT && w == 0 && tig == 0) { sm.mfin[gid] = m_run[0]; sm.mfin[gid + 8] = m_run[1]; }
        bar_sync(END, NPB);
    } else {
        // ======== values: dims 64g .. 64g+63 as 16-dim blocks db; n-tile (db, par) column j is dim
        // 64g + 16db + 2j + par, and k index 4tig + e of a k-step's half hf is cell 16hf + {2tig, 2tig+1, 2tig+8,
        // 2tig+9}[e] (as ldmatrix.trans hands out V and two byte permutes rearrange it)
        const int g = warp - 4;
        float o[4][2][4];
#pragma unroll
        for (int a = 0; a < 4; ++a)
#pragma unroll
            for (int b = 0; b < 2; ++b)
#pragma unroll
                for (int ii = 0; ii < 4; ++ii) o[a][b][ii] = 0.0f;
        for (int i = 0; i < n_tiles; ++i) {
            const int sp = i & 1, s = i % VS;
            bar_sync(FULL_P + sp, NPB);
            bar_sync(FULL_V + s, NVB);
            // the fixed point of p * vs: 2^-23 of the tile's largest p times its largest V scale in this group
            float vmax = fmaxf(scale_of(sm.vs[s][lane], g), scale_of(sm.vs[s][lane + 32], g));
#pragma unroll
            for (int o_ = 16; o_ > 0; o_ >>= 1) vmax = fmaxf(vmax, __shfl_xor_sync(0xffffffffu, vmax, o_));
            float inv[2], ps[2], al[2];
#pragma unroll
            for (int r = 0; r < 2; ++r) {
                const int row = gid + 8 * r;
                const int eb = exp_of(sm.pmax[sp][row] * vmax) - 126;
                const bool ok = eb >= -103;
                inv[r] = ok ? pow2f(23 - eb) : 0.0f;
                ps[r] = ok ? pow2f(eb - 23) : 0.0f;
                al[r] = sm.alpha[sp][row];
            }
            unsigned pa[3][2][4];   // [limb][k-step][register]
#pragma unroll
            for (int kk = 0; kk < 2; ++kk)
#pragma unroll
                for (int hf = 0; hf < 2; ++hf) {
                    const int c = 32 * kk + 16 * hf + 2 * tig;   // cells c, c+1, c+8, c+9
                    const float v0 = scale_of(sm.vs[s][c], g), v1 = scale_of(sm.vs[s][c + 1], g);
                    const float v8 = scale_of(sm.vs[s][c + 8], g), v9 = scale_of(sm.vs[s][c + 9], g);
#pragma unroll
                    for (int r = 0; r < 2; ++r) {
                        const int row = gid + 8 * r;
                        const float2 pa2 = *reinterpret_cast<const float2*>(&sm.p[sp][row][p_at(row, c)]);
                        const float2 pb2 = *reinterpret_cast<const float2*>(&sm.p[sp][row][p_at(row, c + 8)]);
                        const int x = r + 2 * hf;
                        pack3(fx23(pa2.x * v0 * inv[r]), fx23(pa2.y * v1 * inv[r]), fx23(pb2.x * v8 * inv[r]),
                              fx23(pb2.y * v9 * inv[r]), pa[0][kk][x], pa[1][kk][x], pa[2][kk][x]);
                    }
                }
            if (i + 2 < n_tiles) bar_arrive(EMPTY_P + sp, NPB);
#pragma unroll
            for (int db = 0; db < 4; ++db) {
                unsigned bv[2][2][2];   // [k-step][parity][b0, b1]
#pragma unroll
                for (int kk = 0; kk < 2; ++kk) {
                    const int cell = 32 * kk + 8 * (lane >> 3) + (lane & 7);
                    unsigned r4[4];
                    ldsm4t(r4, &sm.v[s][cell * HD + (((4 * g + db) ^ (cell & 7)) << 4)]);
                    bv[kk][0][0] = __byte_perm(r4[0], r4[1], 0x6420);
                    bv[kk][1][0] = __byte_perm(r4[0], r4[1], 0x7531);
                    bv[kk][0][1] = __byte_perm(r4[2], r4[3], 0x6420);
                    bv[kk][1][1] = __byte_perm(r4[2], r4[3], 0x7531);
                }
                int acc[3][2][4];
#pragma unroll
                for (int j = 0; j < 3; ++j)
#pragma unroll
                    for (int par = 0; par < 2; ++par)
#pragma unroll
                        for (int ii = 0; ii < 4; ++ii) acc[j][par][ii] = 0;
#pragma unroll
                for (int kk = 0; kk < 2; ++kk)
#pragma unroll
                    for (int j = 0; j < 3; ++j)
#pragma unroll
                        for (int par = 0; par < 2; ++par) mma_u8s8(acc[j][par], pa[j][kk], bv[kk][par][0], bv[kk][par][1]);
#pragma unroll
                for (int par = 0; par < 2; ++par)
#pragma unroll
                    for (int ii = 0; ii < 4; ++ii) {
                        const int r = ii >> 1;
                        o[db][par][ii] = fmaf(o[db][par][ii], al[r],
                                              limbs3(acc[0][par][ii], acc[1][par][ii], acc[2][par][ii], 65536.0f * ps[r],
                                                     256.0f * ps[r], ps[r], kS * ps[r]));
                    }
            }
            if (i + VS < n_tiles) bar_arrive(EMPTY_V + s, NVB);
        }
        bar_sync(END, NPB);
        if constexpr (SPLIT) {   // o as it is, with each row's maximum and sum
            float* base = part.acc + qi * part.stride;
            const size_t hq = (size_t) blockIdx.z * n_kv_heads * G + (size_t) kvh * G;   // the block's first head
#pragma unroll
            for (int r = 0; r < 2; ++r) {
                const int row = gid + 8 * r;
                if (row >= G) continue;
#pragma unroll
                for (int db = 0; db < 4; ++db)
                    *reinterpret_cast<float4*>(&base[(hq + row) * HD + 64 * g + 16 * db + 4 * tig]) =
                        make_float4(o[db][0][2 * r], o[db][1][2 * r], o[db][0][2 * r + 1], o[db][1][2 * r + 1]);
            }
            if (g == 0 && lane < G) {
                base[part.m + hq + lane] = sm.mfin[lane];
                base[part.l + hq + lane] =
                    (sm.lsum[0][lane] + sm.lsum[1][lane]) + (sm.lsum[2][lane] + sm.lsum[3][lane]);
            }
        } else {
#pragma unroll
            for (int r = 0; r < 2; ++r) {
                const int row = gid + 8 * r;
                if (row >= G) continue;
                const float l = (sm.lsum[0][row] + sm.lsum[1][row]) + (sm.lsum[2][row] + sm.lsum[3][row]);
#pragma unroll
                for (int db = 0; db < 4; ++db) {   // dims 64g + 16db + 4tig + {0, 1, 2, 3}
                    float4 out = make_float4(0.f, 0.f, 0.f, 0.f);
                    if (l > 0.0f) out = make_float4(o[db][0][2 * r] / l, o[db][1][2 * r] / l, o[db][0][2 * r + 1] / l,
                                                    o[db][1][2 * r + 1] / l);
                    *reinterpret_cast<float4*>(&attn[(size_t) row * HD + 64 * g + 16 * db + 4 * tig]) = out;
                }
            }
        }
    }
}

// x * sigmoid(raw) as native_qsa_gate_apply computes it: its TU's fast-math instructions
__device__ __forceinline__ float gated(float x, float raw) {
    float r;
    asm("{\n\t.reg .f32 e;\n\tmul.ftz.f32 e, %2, 0fBFB8AA3B;\n\tex2.approx.ftz.f32 e, e;\n\t"
        "add.ftz.f32 e, e, 0f3F800000;\n\trcp.approx.ftz.f32 e, e;\n\tmul.ftz.f32 %0, %1, e;\n\t}"
        : "=f"(r)
        : "f"(x), "f"(raw));
    return r;
}

// The decode's merge: a block per (head, query), 4 groups of 64 threads adding every 4th split's sums, a thread 4
// dims.  Every thread reads all the splits' maxima and sums, so all of its loads go out at once.
__global__ void __launch_bounds__(THREADS) attn_merge_kernel(Partials part, const int32_t* __restrict__ steps,
                                                             const float* __restrict__ gate, float* __restrict__ attn) {
    const int h = blockIdx.x, n_head = gridDim.x;
    const size_t qi = blockIdx.y;
    const int n_ids = __ldg(steps + qi * kStepCount + kStepWidth), c = split_cells(n_ids, part.splits);
    const int n_sp = (n_ids + c - 1) / c;
    const float* base = part.acc + qi * part.stride;
    const int t = threadIdx.x, grp = t >> 6, d4 = 4 * (t & 63);
    float4 o[SPLIT_MAX / 4];
    float m[SPLIT_MAX], l[SPLIT_MAX];
#pragma unroll
    for (int k = 0; k < SPLIT_MAX / 4; ++k)
        if (grp + 4 * k < n_sp)
            o[k] = *reinterpret_cast<const float4*>(base + ((size_t) (grp + 4 * k) * n_head + h) * HD + d4);
#pragma unroll
    for (int k = 0; k < SPLIT_MAX; ++k)
        if (k < n_sp) {
            m[k] = base[part.m + k * n_head + h];
            l[k] = base[part.l + k * n_head + h];
        }
    float4 raw = make_float4(0.f, 0.f, 0.f, 0.f);
    if (gate && grp == 0) raw = *reinterpret_cast<const float4*>(gate + (qi * n_head + h) * 2 * HD + HD + d4);
    float M = -FLT_MAX, L = 0.0f;
#pragma unroll
    for (int k = 0; k < SPLIT_MAX; ++k)
        if (k < n_sp) M = fmaxf(M, m[k]);
    float4 a = make_float4(0.f, 0.f, 0.f, 0.f);
#pragma unroll
    for (int k = 0; k < SPLIT_MAX; ++k)
        if (k < n_sp) {
            const float w = __expf(m[k] - M);
            L = fmaf(l[k], w, L);
            if ((k & 3) == grp) {
                const float4 x = o[k >> 2];
                a = make_float4(fmaf(x.x, w, a.x), fmaf(x.y, w, a.y), fmaf(x.z, w, a.z), fmaf(x.w, w, a.w));
            }
        }
    __shared__ float4 red[3][64];
    if (grp > 0) red[grp - 1][t & 63] = a;
    __syncthreads();
    if (grp > 0) return;
#pragma unroll
    for (int k = 0; k < 3; ++k) {
        const float4 b = red[k][t];
        a = make_float4(a.x + b.x, a.y + b.y, a.z + b.z, a.w + b.w);
    }
    float4 r = make_float4(0.f, 0.f, 0.f, 0.f);
    if (L > 0.0f) r = make_float4(a.x / L, a.y / L, a.z / L, a.w / L);
    if (gate) r = make_float4(gated(r.x, raw.x), gated(r.y, raw.y), gated(r.z, raw.z), gated(r.w, raw.w));
    *reinterpret_cast<float4*>(attn + (qi * n_head + h) * HD + d4) = r;
}

// A decode query's splits at most: SPLIT_MAX (more cost the merge and each block's setup more than they gain), whole
// tiles.
int64_t max_splits(int64_t cap) { return std::max<int64_t>(1, std::min<int64_t>(SPLIT_MAX, (cap + PT - 1) / PT)); }
// n_q queries' splits: a block of the INT8 kernel takes an SM, so all queries' blocks in one wave
int decode_splits(int64_t cap, int64_t n_q, int64_t n_kv) {
    int dev = 0, sms = 0;
    cudaGetDevice(&dev);
    cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, dev);
    return (int) std::max<int64_t>(1, std::min<int64_t>(max_splits(cap), sms / (n_kv * n_q)));
}

// the per-query scratch's layout for `splits`
Partials partials(float* scratch, int splits, int64_t n_head, int64_t stride) {
    Partials part;
    part.acc = scratch;
    part.m = (long long) splits * n_head * HD;
    part.l = part.m + (long long) splits * n_head;
    part.stride = stride;
    part.splits = splits;
    return part;
}

// the INT8 kernel's shared memory, above the default limit (once for each form)
template <bool SPLIT>
void allow_mma_smem() {
    static const cudaError_t e = cudaFuncSetAttribute(
        attn_mma_kernel<SPLIT>, cudaFuncAttributeMaxDynamicSharedMemorySize, (int) sizeof(pf::Smem));
    (void) e;
}

}  // namespace

void qsa_prefill_attn(const float* q, const QsaAttnPools& pools, const int32_t* ids, const int32_t* steps, int64_t cap,
                      const QsaShapes& s, float* attn, int64_t n_q, void* stream) {
    if (n_q <= 0) return;
    const bool int8 = pools.k_q != nullptr;
    if (s.head_dim != HD || s.n_head != (int64_t) G * s.n_head_kv || cap <= 0 || !ids || !steps || !pools.page_table ||
        (int8 ? (!pools.v_q || !pools.k_scale || !pools.v_scale) : (!pools.k_pool || !pools.v_pool))) {
        std::fprintf(stderr, "qsa_prefill_attn: unsupported geometry or missing buffers\n");
        std::exit(1);
    }
    const float scale = 1.0f / sqrtf((float) HD);
    cudaStream_t st = (cudaStream_t) stream;
    if (int8) allow_mma_smem<false>();
    for (int64_t q0 = 0; q0 < n_q; q0 += 65535) {
        const int64_t nb = n_q - q0 < 65535 ? n_q - q0 : 65535;
        const dim3 grid((unsigned) s.n_head_kv, (unsigned) nb);
        const size_t qo = (size_t) q0 * (size_t) s.n_head * HD;
        const int32_t* ids_q = ids + (size_t) q0 * (size_t) cap;
        const int32_t* steps_q = steps + (size_t) q0 * kStepCount;
        if (int8)
            attn_mma_kernel<false><<<grid, pf::NT, sizeof(pf::Smem), st>>>(
                q + qo, pools, ids_q, steps_q, (int) s.n_head_kv, (int) s.page_size, scale, attn + qo, (int) cap, {});
        else
            attn_f16_kernel<false><<<grid, THREADS, 0, st>>>(
                q + qo, pools, ids_q, steps_q, (int) s.n_head_kv, (int) s.page_size, scale, attn + qo, (int) cap, {});
    }
    const cudaError_t e = cudaGetLastError();
    if (e != cudaSuccess) {
        std::fprintf(stderr, "qsa_prefill_attn: %s\n", cudaGetErrorString(e));
        std::exit(1);
    }
}

void qsa_decode_attn_batch(const float* q, const QsaAttnPools& pools, const int32_t* ids, const int32_t* steps,
                           int64_t cap, const QsaShapes& s, float* scratch, float* attn, int64_t n_q, void* stream,
                           const float* gate) {
    if (n_q <= 0) return;
    const bool int8 = pools.k_q != nullptr;
    if (s.head_dim != HD || s.n_head != (int64_t) G * s.n_head_kv || cap <= 0 || !scratch || !ids || !steps ||
        !pools.page_table || n_q > 65535 ||
        (int8 ? (!pools.v_q || !pools.k_scale || !pools.v_scale) : (!pools.k_pool || !pools.v_pool))) {
        std::fprintf(stderr, "qsa_decode_attn: unsupported geometry or missing buffers\n");
        std::exit(1);
    }
    const int splits = decode_splits(cap, n_q, s.n_head_kv);
    const Partials part = partials(scratch, splits, s.n_head, (long long) qsa_decode_attn_scratch_floats(cap, s));
    const float scale = 1.0f / sqrtf((float) HD);
    const dim3 grid((unsigned) s.n_head_kv, (unsigned) n_q, (unsigned) splits);
    cudaStream_t st = (cudaStream_t) stream;
    if (int8) {
        allow_mma_smem<true>();
        attn_mma_kernel<true><<<grid, pf::NT, sizeof(pf::Smem), st>>>(
            q, pools, ids, steps, (int) s.n_head_kv, (int) s.page_size, scale, nullptr, (int) cap, part);
    } else {
        attn_f16_kernel<true><<<grid, THREADS, 0, st>>>(
            q, pools, ids, steps, (int) s.n_head_kv, (int) s.page_size, scale, nullptr, (int) cap, part);
    }
    attn_merge_kernel<<<dim3((unsigned) s.n_head, (unsigned) n_q), THREADS, 0, st>>>(part, steps, gate, attn);
    const cudaError_t e = cudaGetLastError();
    if (e != cudaSuccess) {
        std::fprintf(stderr, "qsa_decode_attn: %s\n", cudaGetErrorString(e));
        std::exit(1);
    }
}

uint64_t qsa_decode_attn_scratch_floats(int64_t cap, const QsaShapes& s) {
    return (uint64_t) max_splits(cap) * (uint64_t) s.n_head * (HD + 2) + 64;
}

void qsa_decode_attn_step(const float* q, const QsaAttnPools& pools, const int32_t* ids, const int32_t* step,
                          int64_t cap, const QsaShapes& s, float* scratch, float* attn, void* stream) {
    qsa_decode_attn_batch(q, pools, ids, step, cap, s, scratch, attn, 1, stream);
}

}  // namespace strata::kernels
