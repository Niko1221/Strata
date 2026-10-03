// src/kernels/cuda/qsa_prompt_attn_sm70.cu - see src/kernels/cuda/qsa_prompt_attn_sm70.hpp and
// include/strata/kernels/qsa_prompt_attn.hpp.  The same prompt attention as qsa_prompt_attn.cu, on Volta (sm_70).
//
// Why this file: sm_70 has no mma.sync m16n8k8/m16n8k16 (the shapes of qsa_prompt_attn.cu, which traps below sm_75)
// and no cp.async (sm_80).  The only tensor-core MMA on Volta is m8n8k4, and the WMMA API wraps four of them as
// m16n16k16 - the shape this kernel uses.  The algorithm is the one in qsa_prompt_attn.hpp: one block per (query,
// KV head), the same per-query selection read through the page table, 32-cell chunks with an online softmax,
// q and p split into FP16 hi + lo halves (about 22 bits kept), the stored values entering exactly (int8 codes are
// exact in FP16, their per-64 scales multiply in FP32 on the q.k partials and fold into p for p.v).  The result is
// FP32-level like the rest of that kernel family: not bitwise equal to qsa_decode_attn_batch, bounded by
// qsa_prompt_attn_parity.  One documented difference from the sm_75/80 kernels: a cell whose page the KV streaming
// left non-resident (page_table = -1) is masked here (score -inf, weight 0), as qsa_decode_attn_batch masks it.
//
// Volta fragment layouts (m16n16k16, checked element by element on a V100-SXM2, CUDA 12.8):
//   matrix_a row_major / matrix_b col_major: element i of the thread's fragment is row (l&3) + 8*((l>>2)&1) +
//     4*((l>>4)&1) at k/n = i (the lane's row is held twice per warp - the four parallel 8x8x4 MMAs);
//   matrix_b row_major: b[i] = B[k=(l&3) + 4*(i>>2)][n = 8*((l>>3)&1) + 4*((l>>4)&1) + (i&3)];
//   accumulator float: row = (l&1) + 8*((l>>2)&1) + 4*((l>>4)&1) + 2*((i>>1)&1),
//     col = 2*((l>>1)&1) + 8*((l>>3)&1) + (i&1) + 4*((i>>2)&1), i = 0..7.
// A fragments of p are built in registers from that layout (p is per-warp: its V scale fold differs per scale
// group); everything else loads from the shared staging through load_matrix_sync.
#include "qsa_prompt_attn_sm70.hpp"

#include "strata/kernels/kv_q4.hpp"
#include "strata/kernels/kv_q8.hpp"

#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <mma.h>

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <type_traits>

namespace strata::kernels {
namespace {

constexpr int HD = 256;           // head_dim
constexpr int G = 12;             // query heads per KV head
constexpr int CH = 32;            // cells per chunk
constexpr int THREADS = 256;      // 8 warps: a dependent WMMA chain costs 111 cycles on a V100 (58 with 4
                                  // independent chains), so warps interleave over score tiles, scale groups and
                                  // dim tiles; this keeps 8 such chains in flight per SM
constexpr int QS = HD + 8;        // q row stride in halves (padding, as the sm_75/80 kernel)
constexpr int KROW = HD + 8;      // staged K row stride in halves
constexpr int VROW = HD + 8;

// The WMMA API exists in device code only from sm_70 (crt/mma.h); pre-Volta builds compile the kernels to a trap
// and the host keeps the old kernel.  HIP has no nvcuda::wmma - the host entry below refuses there.
#if defined(__HIPCC__)
#define STRATA_PA70 0
#elif !defined(__CUDA_ARCH__) || __CUDA_ARCH__ >= 700
#define STRATA_PA70 1
#else
#define STRATA_PA70 0
#endif

#if STRATA_PA70
using FragA = nvcuda::wmma::fragment<nvcuda::wmma::matrix_a, 16, 16, 16, __half, nvcuda::wmma::row_major>;
using FragBc = nvcuda::wmma::fragment<nvcuda::wmma::matrix_b, 16, 16, 16, __half, nvcuda::wmma::col_major>;
using FragBr = nvcuda::wmma::fragment<nvcuda::wmma::matrix_b, 16, 16, 16, __half, nvcuda::wmma::row_major>;
using FragAcc = nvcuda::wmma::fragment<nvcuda::wmma::accumulator, 16, 16, 16, float>;

__device__ __forceinline__ int acc_row(int i) {
    const int l = threadIdx.x & 31;
    return (l & 1) + (((l >> 2) & 1) << 3) + (((l >> 4) & 1) << 2) + (((i >> 1) & 1) << 1);
}
__device__ __forceinline__ int acc_col(int i) {
    const int l = threadIdx.x & 31;
    return (((l >> 1) & 1) << 1) + (((l >> 3) & 1) << 3) + (i & 1) + (((i >> 2) & 1) << 2);
}
__device__ __forceinline__ int frag_row() {   // the row of matrix_a this lane's fragment holds
    const int l = threadIdx.x & 31;
    return (l & 3) + (((l >> 2) & 1) << 3) + (((l >> 4) & 1) << 2);
}

// Two int8 codes (low byte first) as an exact half2: 1024 + (c + 128) built in the mantissa, minus 1152 (as the
// sm_75/80 kernel).  Staging converts 16 codes per 16-byte piece; this is exact for every int8.
__device__ __forceinline__ uint32_t i8x2_to_h2(uint32_t x) {
    uint32_t y = ((x & 0xffu) | ((x & 0xff00u) << 8)) ^ 0x00800080u;
    y |= 0x64006400u;
    __half2 h = *reinterpret_cast<__half2*>(&y);
    h = __hsub2(h, __halves2half2(__float2half(1152.f), __float2half(1152.f)));
    return *reinterpret_cast<uint32_t*>(&h);
}
__device__ __forceinline__ void store_i8x16_as_h16(__half* dst, uint4 x) {
    const uint8_t* b = reinterpret_cast<const uint8_t*>(&x);
    uint32_t* d = reinterpret_cast<uint32_t*>(dst);
#pragma unroll
    for (int j = 0; j < 8; ++j) d[j] = i8x2_to_h2((uint32_t) b[2 * j] | ((uint32_t) b[2 * j + 1] << 8));
}
#endif

// KV_MODE 1: int8 codes + fp16 scale per 64 values.  KV_MODE 0: fp16 values (scales 1).  KV_MODE 3 (hybrid K8V4):
// K as mode 1, V as q4_0 blocks dequantized to fp16 at staging (the caller un-rotates the output).  Both sides
// stage as FP16: WMMA loads its B fragments from shared memory, and int8 codes are exact in FP16.
template <int KV_MODE>
struct Smem7 {
    __half qh[16][QS];
    __half ql[16][QS];
    __half k[CH][KROW];
    __half v[CH][VROW];
    float ks[CH][4];
    float vs[CH][4];
    float part[4][16][CH + 1];   // q.k raw partial per scale group (two warp tiles per group), the scales
                                 // multiply in the softmax's fixed fma chain
    float p[16][CH + 1];         // softmax weights
    float qmax[THREADS / 32];
    float alpha[16];
    float lsum[16];
    float mrow[16];
    long long row[2][CH];        // double-buffered pool rows: the next chunk's lookups hide under p.v
};

template <int KV_MODE>
__global__ void __launch_bounds__(THREADS) prompt_attn_sm70_kernel(const float* __restrict__ q, QsaAttnPools p,
                                                                   const int32_t* __restrict__ ids,
                                                                   const int32_t* __restrict__ steps, int n_kv_heads,
                                                                   int page_size, float scale_log2,
                                                                   float* __restrict__ attn, int cap) {
#if STRATA_PA70
    extern __shared__ __align__(16) unsigned char smem_raw[];
    Smem7<KV_MODE>& S = *reinterpret_cast<Smem7<KV_MODE>*>(smem_raw);
    const int qi = blockIdx.x, kvh = blockIdx.y;
    const int n_head = n_kv_heads * G;
    q += (size_t) qi * n_head * HD + (size_t) kvh * G * HD;
    attn += (size_t) qi * n_head * HD + (size_t) kvh * G * HD;
    ids += (size_t) qi * cap;
    const int n = __ldg(steps + (size_t) qi * kStepCount + kStepWidth);
    const int t = threadIdx.x, lane = t & 31, warp = t >> 5;
    const int dim0 = warp * 32;

    // q: 12 heads + 4 zero rows, scaled by a power of two that puts its largest value near 2^14 (exact, and the
    // lo halves stay out of FP16's subnormal range), then split into hi + lo halves (as the sm_75/80 kernel)
    float qm = 0.0f;
    for (int i = t; i < G * HD; i += THREADS) qm = fmaxf(qm, fabsf(q[i]));
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) qm = fmaxf(qm, __shfl_xor_sync(0xffffffffu, qm, o));
    if (lane == 0) S.qmax[warp] = qm;
    __syncthreads();
    qm = 0.0f;
#pragma unroll
    for (int w = 0; w < THREADS / 32; ++w) qm = fmaxf(qm, S.qmax[w]);
    int qe = 0;
    if (qm > 0.0f) frexpf(qm, &qe);                 // qm < 2^qe
    const float qup = ldexpf(1.0f, 14 - qe), qdown = ldexpf(scale_log2, qe - 14);
    for (int i = t; i < 16 * HD; i += THREADS) {
        const int h = i / HD, d = i % HD;
        const float x = h < G ? q[(size_t) h * HD + d] * qup : 0.0f;
        const __half hi = __float2half_rn(x);
        S.qh[h][d] = hi;
        S.ql[h][d] = __float2half_rn(x - __half2float(hi));
    }
    if (t < 16) { S.mrow[t] = -INFINITY; S.lsum[t] = 0.0f; }

    // acc of p.v: 2 dim tiles of 16 per warp (dims [32w, 32w+32)), 8 FP32 elements per accumulator fragment;
    // element i keeps the fragment's (row, col) for every chunk
    float acc[2][8];
#pragma unroll
    for (int jt = 0; jt < 2; ++jt)
#pragma unroll
        for (int i = 0; i < 8; ++i) acc[jt][i] = 0.0f;

    // the pool rows of a chunk: -1 past the selection and for a page the KV streaming left non-resident (the
    // decode kernel masks those, so this kernel does too).  Chunk 0's rows fill one buffer here; every later
    // chunk's lookups fill the other just before the previous p.v, where their global latency hides.
    int rbuf = 0;
    if (t < CH) {
        long long r = -1;
        if (t < min(CH, n)) {
            const int cell = ids[t];
            const long long page = (long long) p.page_table[cell / page_size];
            if (page >= 0) r = (page * n_kv_heads + kvh) * page_size + (cell % page_size);
        }
        S.row[0][t] = r;
    }
    for (int c0 = 0; c0 < n; c0 += CH, rbuf ^= 1) {
        const int nh = min(CH, n - c0);
        __syncthreads();   // rows ready; the previous chunk's p.v is done with k, v
        // gather the chunk's K and V rows as FP16 (int8 codes converted exactly; K8V4's V dequantized from its
        // q4_0 blocks) and their per-64-dim scales.  Each thread issues ALL of its global loads before any of
        // its stores: they then overlap each other instead of paying the HBM latency one at a time.
        {
            constexpr int KPC = CH * (HD / 16);            // 16-dim pieces of a chunk
            constexpr int KPT = (KPC + THREADS - 1) / THREADS;
            static_assert(KPC % THREADS == 0, "the staging pieces divide evenly");
            uint4 xk[KPT][2], xv[KPT][2];   // [piece][16B]: mode 0 fills both 16B, the int8 modes one
            int kc[KPT], kp[KPT];
            long long kr[KPT];
#pragma unroll
            for (int s = 0; s < KPT; ++s) {
                const int i = t + s * THREADS;
                kc[s] = i / (HD / 16); kp[s] = i % (HD / 16);
                kr[s] = S.row[rbuf][kc[s]];
                xk[s][0] = make_uint4(0, 0, 0, 0); xk[s][1] = make_uint4(0, 0, 0, 0);
                xv[s][0] = make_uint4(0, 0, 0, 0); xv[s][1] = make_uint4(0, 0, 0, 0);
                if constexpr (KV_MODE == 0) {
                    if (kr[s] >= 0) {
                        const uint4* sk = reinterpret_cast<const uint4*>(p.k_pool + kr[s] * HD);
                        const uint4* sv = reinterpret_cast<const uint4*>(p.v_pool + kr[s] * HD);
                        xk[s][0] = __ldg(sk + kp[s] * 2); xk[s][1] = __ldg(sk + kp[s] * 2 + 1);
                        xv[s][0] = __ldg(sv + kp[s] * 2); xv[s][1] = __ldg(sv + kp[s] * 2 + 1);
                    }
                } else {
                    if (kr[s] >= 0) {
                        xk[s][0] = __ldg(reinterpret_cast<const uint4*>(p.k_q + kr[s] * HD) + kp[s]);
                        if constexpr (KV_MODE == 1)
                            xv[s][0] = __ldg(reinterpret_cast<const uint4*>(p.v_q + kr[s] * HD) + kp[s]);
                    }
                }
            }
#pragma unroll
            for (int s = 0; s < KPT; ++s) {
                __half* dk = &S.k[kc[s]][kp[s] * 16];
                if constexpr (KV_MODE == 0) {
                    *reinterpret_cast<uint4*>(dk) = xk[s][0];
                    *reinterpret_cast<uint4*>(dk + 8) = xk[s][1];
                } else {
                    store_i8x16_as_h16(dk, xk[s][0]);
                }
                if constexpr (KV_MODE != 3) {
                    __half* dv = &S.v[kc[s]][kp[s] * 16];
                    if constexpr (KV_MODE == 0) {
                        *reinterpret_cast<uint4*>(dv) = xv[s][0];
                        *reinterpret_cast<uint4*>(dv + 8) = xv[s][1];
                    } else {
                        store_i8x16_as_h16(dv, xv[s][0]);
                    }
                }
            }
            if constexpr (KV_MODE == 3) {   // V: dequantize the row's q4_0 blocks straight into the fp16 V row
                constexpr int BLKS = HD / QK4_0;
                constexpr int BYTES = BLKS * (int) sizeof(block_q4_0);
                for (int i = t; i < CH * BLKS; i += THREADS) {
                    const int c = i / BLKS, b = i % BLKS;
                    const long long r = S.row[rbuf][c];
                    __half* dv = &S.v[c][b * QK4_0];
#pragma unroll
                    for (int j = 0; j < QK4_0; ++j) dv[j] = __half(0);
                    if (r >= 0) {
                        const block_q4_0* blk = reinterpret_cast<const block_q4_0*>(p.v_q4 + r * BYTES) + b;
                        const float d = __half2float(__ushort_as_half(blk->d));
#pragma unroll
                        for (int j = 0; j < QK4_0 / 2; ++j) {
                            dv[j] = __float2half_rn((float) ((int) (blk->qs[j] & 0x0F) - 8) * d);
                            dv[j + QK4_0 / 2] = __float2half_rn((float) ((int) (blk->qs[j] >> 4) - 8) * d);
                        }
                    }
                }
            }
            for (int i = t; i < CH * 4; i += THREADS) {
                const int c = i / 4, g = i % 4;
                const long long r = S.row[rbuf][c];
                float a = 0.0f, b = 0.0f;
                if (r >= 0) {
                    if constexpr (KV_MODE == 1) {
                        a = __half2float(__ushort_as_half(p.k_scale[r * (HD / KV_Q8_GROUP) + g]));
                        b = __half2float(__ushort_as_half(p.v_scale[r * (HD / KV_Q8_GROUP) + g]));
                    } else if constexpr (KV_MODE == 3) {   // K as int8, V dequantized to fp16 (scale 1)
                        a = __half2float(__ushort_as_half(p.k_scale[r * (HD / KV_Q8_GROUP) + g]));
                        b = 1.0f;
                    } else {
                        a = b = 1.0f;
                    }
                }
                S.ks[c][g] = a;
                S.vs[c][g] = b;
            }
        }
        __syncthreads();
        // scores: warp w takes the 16-cell tile (w & 1) of the scale group (w >> 1) and writes its raw partial to
        // part[g]; the per-64 K scales multiply in the fixed fma chain below, as the sm_75/80 kernel accumulates
        // them
        {
            const int cbase = (warp & 1) * 16, g = warp >> 1;
            // two independent 4-step chains (a dependent WMMA chain costs 111 cycles on a V100), summed
            FragAcc f0, f1;
            nvcuda::wmma::fill_fragment(f0, 0.0f);
            nvcuda::wmma::fill_fragment(f1, 0.0f);
#pragma unroll
            for (int kk = 0; kk < 4; ++kk) {   // the group's 64 dims in 16-dim steps
                const int d0 = g * 64 + kk * 16;
                FragA ah, al;
                FragBc b;
                nvcuda::wmma::load_matrix_sync(ah, &S.qh[0][d0], QS);
                nvcuda::wmma::load_matrix_sync(al, &S.ql[0][d0], QS);
                nvcuda::wmma::load_matrix_sync(b, &S.k[cbase][d0], KROW);
                FragAcc& f = kk < 2 ? f0 : f1;
                nvcuda::wmma::mma_sync(f, ah, b, f);
                nvcuda::wmma::mma_sync(f, al, b, f);
            }
#pragma unroll
            for (int i = 0; i < 8; ++i) {
                const int row = acc_row(i), col = acc_col(i);
                S.part[g][row][cbase + col] = f0.x[i] + f1.x[i];
            }
        }
        __syncthreads();
        // online softmax: row t/16, 2 cells per thread, 16 threads per row (lanes 16r..16r+15 of a warp).  The
        // four groups' partials enter with their K scales in one fma chain, the order the sm_75/80 kernel uses.
        {
            constexpr int PER = CH / 16;
            const int r = t >> 4, sub = t & 15;
            float x[PER], mx = -INFINITY;
#pragma unroll
            for (int j = 0; j < PER; ++j) {
                const int c = sub * PER + j;
                float sc = 0.0f;
                sc = fmaf(S.part[0][r][c], S.ks[c][0], sc);
                sc = fmaf(S.part[1][r][c], S.ks[c][1], sc);
                sc = fmaf(S.part[2][r][c], S.ks[c][2], sc);
                sc = fmaf(S.part[3][r][c], S.ks[c][3], sc);
                x[j] = (c < nh && S.row[rbuf][c] >= 0) ? sc * qdown : -INFINITY;
                mx = fmaxf(mx, x[j]);
            }
#pragma unroll
            for (int o = 8; o > 0; o >>= 1) mx = fmaxf(mx, __shfl_xor_sync(0xffffffffu, mx, o));
            const float m_old = S.mrow[r];
            const float m_new = fmaxf(m_old, mx);
            float sum = 0.0f;
#pragma unroll
            for (int j = 0; j < PER; ++j) {
                const float e = x[j] == -INFINITY ? 0.0f : exp2f(x[j] - m_new);
                S.p[r][sub * PER + j] = e;
                sum += e;
            }
#pragma unroll
            for (int o = 8; o > 0; o >>= 1) sum += __shfl_xor_sync(0xffffffffu, sum, o);
            __syncwarp();
            if (sub == 0) {
                const float a = m_old == -INFINITY ? 0.0f : exp2f(m_old - m_new);
                S.alpha[r] = a;
                S.lsum[r] = fmaf(S.lsum[r], a, sum);
                S.mrow[r] = m_new;
            }
        }
        __syncthreads();
        // the next chunk's pool rows, into the other buffer: the two global lookups per cell hide under p.v
        if (c0 + CH < n && t < CH) {
            long long r = -1;
            const int idx = c0 + CH + t;
            if (idx < n) {
                const int cell = ids[idx];
                const long long page = (long long) p.page_table[cell / page_size];
                if (page >= 0) r = (page * n_kv_heads + kvh) * page_size + (cell % page_size);
            }
            S.row[rbuf ^ 1][t] = r;
        }
        // p.v: warp w owns dims [32w, 32w+32), two tiles of the scale group (w >> 1).  The scale is folded into p
        // relative to the chunk's largest, times 2^14 (p' <= 2^14: its lo half stays out of FP16's subnormal range);
        // the chunk's sum is then added to the running one in FP32 with the factor taken back out (as the sm_75/80
        // kernel).  p' is per-warp (its vup differs per scale group): the A fragments are built in registers from
        // the matrix_a layout above, once per 16-cell tile and shared by both dim tiles.
        {
            const int grp = warp >> 1;
            float vmax = 0.0f;
#pragma unroll
            for (int c = lane; c < CH; c += 32) vmax = fmaxf(vmax, S.vs[c][grp]);
#pragma unroll
            for (int o = 16; o > 0; o >>= 1) vmax = fmaxf(vmax, __shfl_xor_sync(0xffffffffu, vmax, o));
            const float vup = vmax > 0.0f ? 16384.0f / vmax : 0.0f, vdown = vmax * (1.0f / 16384.0f);
            const int pr = frag_row();   // the head this lane's A fragments hold
            FragA ah[CH / 16], al[CH / 16];
#pragma unroll
            for (int kt = 0; kt < CH / 16; ++kt)
#pragma unroll
                for (int i = 0; i < 16; ++i) {
                    const int c = kt * 16 + i;
                    const float pv = S.p[pr][c] * (S.vs[c][grp] * vup);
                    const __half hi = __float2half_rn(pv);
                    ah[kt].x[i] = hi;
                    al[kt].x[i] = __float2half_rn(pv - __half2float(hi));
                }
#pragma unroll
            for (int jt = 0; jt < 2; ++jt) {
                // the hi and lo halves go into two 2-step chains (as the scores above) and add at the end
                FragAcc fh, fl;
                nvcuda::wmma::fill_fragment(fh, 0.0f);
                nvcuda::wmma::fill_fragment(fl, 0.0f);
#pragma unroll
                for (int kt = 0; kt < CH / 16; ++kt) {
                    FragBr b;
                    nvcuda::wmma::load_matrix_sync(b, &S.v[kt * 16][dim0 + jt * 16], VROW);
                    nvcuda::wmma::mma_sync(fh, ah[kt], b, fh);
                    nvcuda::wmma::mma_sync(fl, al[kt], b, fl);
                }
#pragma unroll
                for (int i = 0; i < 8; ++i) {
                    const int row = acc_row(i);
                    acc[jt][i] = fmaf(acc[jt][i], S.alpha[row], (fh.x[i] + fl.x[i]) * vdown);
                }
            }
        }
    }
    __syncthreads();
    // normalize by the row's exp sum and write the 12 heads (the 4 pad rows stay in the fragment layout)
#pragma unroll
    for (int jt = 0; jt < 2; ++jt)
#pragma unroll
        for (int i = 0; i < 8; ++i) {
            const int row = acc_row(i), col = acc_col(i);
            if (row < G) {
                const float l = S.lsum[row];
                const float inv = l > 0.0f ? 1.0f / l : 0.0f;
                attn[(size_t) row * HD + dim0 + jt * 16 + col] = acc[jt][i] * inv;
            }
        }
#else
    (void) q; (void) p; (void) ids; (void) steps; (void) n_kv_heads; (void) page_size; (void) scale_log2; (void) attn;
    (void) cap;
    __trap();   // pre-Volta builds: no WMMA (the host keeps the old kernel there)
#endif
}

#if !defined(__HIPCC__)
template <int KV_MODE>
bool launch70(const float* q, const QsaAttnPools& pools, const int32_t* ids, const int32_t* steps, int64_t cap,
              const QsaShapes& s, float* attn, int64_t n_q, cudaStream_t st) {
    static bool attr[64] = {};   // the shared-memory opt-in is per device (a layer split runs this on several)
    int dev = 0;
    cudaGetDevice(&dev);
    const int bytes = (int) sizeof(Smem7<KV_MODE>);
    if (dev < 0 || dev >= 64) return false;
    if (!attr[dev]) {
        if (cudaFuncSetAttribute(prompt_attn_sm70_kernel<KV_MODE>, cudaFuncAttributeMaxDynamicSharedMemorySize,
                                 bytes) != cudaSuccess) {
            cudaGetLastError();
            return false;
        }
        attr[dev] = true;
    }
    const float scale_log2 = 1.4426950408889634f / sqrtf((float) HD);
    for (int64_t q0 = 0; q0 < n_q; q0 += 65535) {
        const int64_t nb = n_q - q0 < 65535 ? n_q - q0 : 65535;
        prompt_attn_sm70_kernel<KV_MODE><<<dim3((unsigned) nb, (unsigned) s.n_head_kv), THREADS, bytes, st>>>(
            q + q0 * s.n_head * HD, pools, ids + q0 * cap, steps + q0 * kStepCount, (int) s.n_head_kv,
            (int) s.page_size, scale_log2, attn + q0 * s.n_head * HD, (int) cap);
    }
    const cudaError_t e = cudaGetLastError();
    if (e != cudaSuccess) {
        std::fprintf(stderr, "qsa_prompt_attn_sm70_batch: %s\n", cudaGetErrorString(e));
        std::exit(1);
    }
    return true;
}
#endif

}  // namespace

bool qsa_prompt_attn_sm70_batch(const float* q, const QsaAttnPools& pools, const int32_t* ids, const int32_t* steps,
                                int64_t cap, const QsaShapes& s, float* attn, int64_t n_q, cudaStream_t stream) {
#if defined(__HIPCC__)
    (void) q; (void) pools; (void) ids; (void) steps; (void) cap; (void) s; (void) attn; (void) n_q; (void) stream;
    return false;
#else
    if (n_q <= 0) return true;
    if (pools.k_q4 != nullptr || s.head_dim != HD || s.n_head != (int64_t) G * s.n_head_kv || cap <= 0 || !ids ||
        !steps || !pools.page_table)
        return false;
    if (pools.k_q != nullptr && pools.v_q4 != nullptr) {   // hybrid K8V4: int8 K + dequantized-q4 V
        if (!pools.k_scale) return false;
        return launch70<3>(q, pools, ids, steps, cap, s, attn, n_q, stream);
    }
    if (pools.k_q != nullptr) {
        if (!pools.v_q || !pools.k_scale || !pools.v_scale) return false;
        return launch70<1>(q, pools, ids, steps, cap, s, attn, n_q, stream);
    }
    if (!pools.k_pool || !pools.v_pool) return false;
    return launch70<0>(q, pools, ids, steps, cap, s, attn, n_q, stream);
#endif
}

}  // namespace strata::kernels
