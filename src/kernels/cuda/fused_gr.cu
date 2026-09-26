// src/kernels/cuda/fused_gr.cu - see include/strata/kernels/fused_gr.hpp.
#include "strata/kernels/fused_gr.hpp"
#include "strata/kernels/bf16_bits.hpp"
#include "xor_scatter.cuh"

#include <cuda_runtime.h>

#include <cstdio>
#include <cstdlib>

namespace strata::kernels {
namespace {

constexpr int N = 2560;         // n_embd
constexpr int HC = 4;           // streams
constexpr int D = N * HC;       // 10240
constexpr int LR = 320;         // hc_lr
constexpr int THREADS = 256;
constexpr int WARPS = THREADS / 32;
constexpr int DOWN_BLOCKS = LR / WARPS;          // 40 blocks of 8 rows; one more for the inject rows
constexpr int UP_COLS = 32;                      // columns d per `up` block (x 4 streams = 128 rows)
constexpr int UP_BLOCKS = N / UP_COLS;           // 80

__device__ __forceinline__ float warp_sum(float v) {
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
    return v;
}
__device__ __forceinline__ float sigmoidf_(float x) { return 1.0f / (1.0f + __expf(-x)); }

// 8 bf16 packed in a uint4 against 8 floats.
__device__ __forceinline__ float dot8(const uint4 w, const float* x) {
    float acc = 0.0f;
    const uint32_t v[4] = {w.x, w.y, w.z, w.w};
#pragma unroll
    for (int j = 0; j < 4; ++j) {
        acc = fmaf(__uint_as_float(v[j] << 16), x[2 * j], acc);
        acc = fmaf(__uint_as_float(v[j] & 0xffff0000u), x[2 * j + 1], acc);
    }
    return acc;
}

__global__ void __launch_bounds__(THREADS) gr_down_kernel(FusedGrArgs a) {
    __shared__ __align__(16) float xn[D];
    __shared__ float part[WARPS][HC];
    __shared__ float s_rs[HC];
    const int t = threadIdx.x, lane = t & 31, warp = t >> 5;
    float gw[HC];
#pragma unroll
    for (int c = 0; c < HC; ++c) gw[c] = a.apply ? 2.0f * sigmoidf_(a.inj_prev[c] / (float) HC) : 0.0f;
    // 1. R' * w_norm into shared memory, and the per-stream sums of squares of R'.
    float ss[HC] = {0.0f, 0.0f, 0.0f, 0.0f};
    for (int i = t * 4; i < D; i += THREADS * 4) {
        const int c = i / N, d = i - c * N;
        float4 r = *reinterpret_cast<const float4*>(a.R + i);
        if (a.apply) {
            const float4 b = *reinterpret_cast<const float4*>(a.bo_prev + d);
            r.x = fmaf(b.x, gw[c], r.x); r.y = fmaf(b.y, gw[c], r.y);
            r.z = fmaf(b.z, gw[c], r.z); r.w = fmaf(b.w, gw[c], r.w);
        }
        const float4 g = *reinterpret_cast<const float4*>(a.w_norm + i);
        float sq = r.x * r.x + r.y * r.y + r.z * r.z + r.w * r.w;
#pragma unroll
        for (int cc = 0; cc < HC; ++cc) if (cc == c) ss[cc] += sq;
        *reinterpret_cast<float4*>(xn + i) = make_float4(r.x * g.x, r.y * g.y, r.z * g.z, r.w * g.w);
    }
#pragma unroll
    for (int c = 0; c < HC; ++c) {
        const float v = warp_sum(ss[c]);
        if (lane == 0) part[warp][c] = v;
    }
    __syncthreads();
    if (t < HC) {
        float s = 0.0f;
        for (int w = 0; w < WARPS; ++w) s += part[w][t];
        s_rs[t] = rsqrtf(s / (float) N + a.eps);
        if (blockIdx.x == 0) a.rs[t] = s_rs[t];
    }
    __syncthreads();
    for (int i = t; i < D; i += THREADS) xn[i] *= s_rs[i / N];
    __syncthreads();
    // 2. one warp per output row: 10240 bf16 = 1280 chunks of 8, 40 per lane.
    const bool inject_block = blockIdx.x == DOWN_BLOCKS;
    const int row = inject_block ? warp : blockIdx.x * WARPS + warp;
    if (inject_block && (a.w_inject == nullptr || warp >= HC)) return;
    const uint16_t* wrow = (inject_block ? a.w_inject : a.w_down) + (size_t) row * D;
    const uint4* w4 = reinterpret_cast<const uint4*>(wrow);
    float acc = 0.0f;
#pragma unroll 4
    for (int j = lane; j < D / 8; j += 32) acc += dot8(__ldg(w4 + j), xn + j * 8);
    acc = warp_sum(acc);
    if (lane != 0) return;
    if (inject_block) {
        a.inject_out[row] = acc;
    } else {
        const float x = acc / (float) HC;
        a.lo[row] = x / (1.0f + __expf(-x));
    }
}

__global__ void __launch_bounds__(THREADS) gr_up_kernel(FusedGrArgs a) {
    __shared__ __align__(16) float lo[LR];
    __shared__ float g[HC][UP_COLS];
    const int t = threadIdx.x, lane = t & 31, warp = t >> 5;
    const int d0 = blockIdx.x * UP_COLS;
    for (int k = t; k < LR; k += THREADS) lo[k] = a.lo[k];
    __syncthreads();
    // 128 rows (4 streams x 32 columns), 16 per warp: 320 bf16 = 40 chunks of 8.
    for (int r = warp; r < HC * UP_COLS; r += WARPS) {
        const int c = r / UP_COLS, dd = r - c * UP_COLS, i = c * N + d0 + dd;
        const uint4* w4 = reinterpret_cast<const uint4*>(a.w_up + (size_t) i * LR);
        float acc = dot8(__ldg(w4 + lane), lo + lane * 8);
        if (lane < LR / 8 - 32) acc += dot8(__ldg(w4 + 32 + lane), lo + (32 + lane) * 8);
        acc = warp_sum(acc);
        if (lane == 0) {
            float rv = a.R[i];
            if (a.apply) {
                rv = fmaf(a.bo_prev[d0 + dd], 2.0f * sigmoidf_(a.inj_prev[c] / (float) HC), rv);
                a.R_out[i] = rv;                       // this block owns column d0+dd of every stream
            }
            const float x = rv * a.w_norm[i] * a.rs[c];
            g[c][dd] = x * sigmoidf_(acc);
        }
    }
    __syncthreads();
    if (t < UP_COLS) {
        float s = 0.0f;
#pragma unroll
        for (int c = 0; c < HC; ++c) s += g[c][t];
        a.mixed[d0 + t] = s / (float) HC;
    }
}

// ================================ plan v0.3 P6: T tokens, one weight read ================================
struct GrMulti {
    FusedGrArgs a[kFusedGrMaxT];
    float* xn;
    int T;
};

// Step 1 of `gr_down_kernel`, one block per (token, stream): each thread visits its elements of that stream in the
// order `gr_down_kernel`'s thread does, and the stream's warp partials are summed in the same order.  rs and xn to
// global.
__global__ void __launch_bounds__(THREADS) gr_norm_multi_kernel(GrMulti m) {
    __shared__ float part[WARPS];
    const int tok = blockIdx.x, c = blockIdx.y;
    const FusedGrArgs& a = m.a[tok];
    float* xn = m.xn + (size_t) tok * D;
    const int t = threadIdx.x, lane = t & 31, warp = t >> 5;
    const float gw = a.apply ? 2.0f * sigmoidf_(a.inj_prev[c] / (float) HC) : 0.0f;
    constexpr int STEPS = D / (THREADS * 4);   // 10
    float4 p[STEPS];
    float ss = 0.0f;
#pragma unroll
    for (int j = 0; j < STEPS; ++j) {
        const int i = t * 4 + j * THREADS * 4;
        if (i / N != c) continue;
        float4 r = *reinterpret_cast<const float4*>(a.R + i);
        if (a.apply) {
            const float4 bo = *reinterpret_cast<const float4*>(a.bo_prev + (i - c * N));
            r.x = fmaf(bo.x, gw, r.x); r.y = fmaf(bo.y, gw, r.y);
            r.z = fmaf(bo.z, gw, r.z); r.w = fmaf(bo.w, gw, r.w);
        }
        const float4 g = *reinterpret_cast<const float4*>(a.w_norm + i);
        const float sq = r.x * r.x + r.y * r.y + r.z * r.z + r.w * r.w;
        ss += sq;
        p[j] = make_float4(r.x * g.x, r.y * g.y, r.z * g.z, r.w * g.w);
    }
    ss = warp_sum(ss);
    if (lane == 0) part[warp] = ss;
    __syncthreads();
    float sum = 0.0f;
    for (int w = 0; w < WARPS; ++w) sum += part[w];
    const float rs = rsqrtf(sum / (float) N + a.eps);
    if (t == 0) a.rs[c] = rs;
#pragma unroll
    for (int j = 0; j < STEPS; ++j) {
        const int i = t * 4 + j * THREADS * 4;
        if (i / N != c) continue;
        *reinterpret_cast<float4*>(xn + i) = make_float4(p[j].x * rs, p[j].y * rs, p[j].z * rs, p[j].w * rs);
    }
}

// 8 bf16 packed in a uint4 against 8 floats held as two float4 (the halves of a staged chunk): `dot8`'s order.
__device__ __forceinline__ float dot8h(const uint4 w, const float4 x0, const float4 x1) {
    float acc = 0.0f;
    acc = fmaf(__uint_as_float(w.x << 16), x0.x, acc);
    acc = fmaf(__uint_as_float(w.x & 0xffff0000u), x0.y, acc);
    acc = fmaf(__uint_as_float(w.y << 16), x0.z, acc);
    acc = fmaf(__uint_as_float(w.y & 0xffff0000u), x0.w, acc);
    acc = fmaf(__uint_as_float(w.z << 16), x1.x, acc);
    acc = fmaf(__uint_as_float(w.z & 0xffff0000u), x1.y, acc);
    acc = fmaf(__uint_as_float(w.w << 16), x1.z, acc);
    acc = fmaf(__uint_as_float(w.w & 0xffff0000u), x1.w, acc);
    return acc;
}

// Step 2 of `gr_down_kernel` for T tokens.  One warp per row, so each lane accumulates the same chunks in the same
// order as the single-token kernel.  The xn tiles of every token are staged split into the first and second halves
// of the 8-float chunks (conflict-free 16-byte reads) and double-buffered; each tile's weights are loaded one tile
// ahead; blocks start their staging at different offsets, which spreads the reads of the same lines over time.
// TC chunks per tile: 320 for up to 4 tokens, 160 above (the two buffers must fit shared memory).
template <int TC>
__global__ void __launch_bounds__(THREADS) gr_down_multi_kernel(GrMulti m) {
    extern __shared__ __align__(16) float4 buf[];   // [2][T][2][TC]
    constexpr int TQ = TC / 32, NT = D / 8 / TC;
    const int t = threadIdx.x, lane = t & 31, warp = t >> 5;
    const int T = m.T;
    const bool inject_block = blockIdx.x == DOWN_BLOCKS;
    const int row = inject_block ? warp : blockIdx.x * WARPS + warp;
    const bool active = !(inject_block && (m.a[0].w_inject == nullptr || warp >= HC));
    const uint4* w4 = reinterpret_cast<const uint4*>((inject_block ? m.a[0].w_inject : m.a[0].w_down) +
                                                     (size_t) (active ? row : 0) * D);
    float acc[kFusedGrMaxT];
#pragma unroll
    for (int k = 0; k < kFusedGrMaxT; ++k) acc[k] = 0.0f;
    const int per_buf = T * 2 * TC, cnt = T * 2 * TC;
    const int rot = (int) ((blockIdx.x * 331u) % (unsigned) cnt);
    const float4* src4 = reinterpret_cast<const float4*>(m.xn);
    auto stage = [&](int tile, float4* dst) {
        for (int ii = t; ii < cnt; ii += THREADS) {
            int i = ii + rot;
            if (i >= cnt) i -= cnt;
            const int k = i / (2 * TC), rem = i - k * 2 * TC, j = rem >> 1, h = rem & 1;
            dst[(k * 2 + h) * TC + j] = src4[((size_t) k * D + (size_t) tile * TC * 8) / 4 + j * 2 + h];
        }
    };
    uint4 wv[TQ], wn[TQ];
    if (active) {
#pragma unroll
        for (int q = 0; q < TQ; ++q) wn[q] = __ldg(w4 + lane + 32 * q);
    }
    stage(0, buf);
    __syncthreads();
    for (int tile = 0; tile < NT; ++tile) {
        const float4* xb = buf + (size_t) (tile & 1) * per_buf;
#pragma unroll
        for (int q = 0; q < TQ; ++q) wv[q] = wn[q];
        if (tile + 1 < NT) {
            if (active) {
#pragma unroll
                for (int q = 0; q < TQ; ++q) wn[q] = __ldg(w4 + (tile + 1) * TC + lane + 32 * q);
            }
            stage(tile + 1, buf + (size_t) ((tile + 1) & 1) * per_buf);
        }
        if (active) {
#pragma unroll
            for (int q = 0; q < TQ; ++q) {
                const int j = lane + 32 * q;
#pragma unroll
                for (int k = 0; k < kFusedGrMaxT; ++k) {
                    if (k >= T) break;
                    acc[k] += dot8h(wv[q], xb[(k * 2) * TC + j], xb[(k * 2 + 1) * TC + j]);
                }
            }
        }
        __syncthreads();
    }
    if (!active) return;
    float s[kFusedGrMaxT];
#pragma unroll
    for (int k = 0; k < kFusedGrMaxT; ++k) s[k] = k < T ? warp_sum(acc[k]) : 0.0f;
    // lane k writes token k (every lane holds every sum after the xor reduction)
#pragma unroll
    for (int k = 0; k < kFusedGrMaxT; ++k) {
        if (k >= T || lane != k) continue;
        if (inject_block) {
            m.a[k].inject_out[row] = s[k];
        } else {
            const float x = s[k] / (float) HC;
            m.a[k].lo[row] = x / (1.0f + __expf(-x));
        }
    }
}

constexpr int UPM_COLS = 16;                      // columns per block (x 4 streams = 64 rows, 8 per warp)
constexpr int UPM_BLOCKS = N / UPM_COLS;          // 160
constexpr int UPM_ROWS = HC * UPM_COLS / WARPS;   // rows a warp

// `gr_up_kernel` for TT tokens: each row of w_up read once, RW rows a warp at a time; the TT x RW dots of those rows
// summed by one xor tree (xor_scatter, each sum bitwise warp_sum's) and every (row, token) epilogue on the lane holding
// its sum, its inputs fetched while the dots run.  The lo vectors are staged split into chunk halves.
template <int TT>
__global__ void __launch_bounds__(THREADS) gr_up_multi_kernel(GrMulti m) {
    constexpr int RW = TT <= 4 ? 4 : 2;
    constexpr int NV = pow2_at_least(TT * RW);    // sums a lane reduces, zero-padded
    constexpr int LPV = 32 / NV;                  // lanes that end up holding each sum
    __shared__ __align__(16) float4 lo4[TT][2][LR / 8];
    __shared__ float g[TT][HC][UPM_COLS];
    const int t = threadIdx.x, lane = t & 31, warp = t >> 5;
    const int d0 = blockIdx.x * UPM_COLS;
    for (int i = t; i < TT * LR / 4; i += THREADS) {
        const int k = i / (LR / 4), f = i - k * (LR / 4);
        lo4[k][f & 1][f >> 1] = reinterpret_cast<const float4*>(m.a[k].lo)[f];
    }
    __syncthreads();
    const int vi = xor_scatter_index<NV>(lane);   // the sum this lane ends with: row vi / TT, token vi % TT
    const int vq = vi / TT, vk = vi - vq * TT;
    const bool epi = (lane & (LPV - 1)) == 0 && vi < TT * RW;
    for (int r0 = warp * UPM_ROWS; r0 < (warp + 1) * UPM_ROWS; r0 += RW) {
        uint4 wa[RW], wb[RW];
#pragma unroll
        for (int q = 0; q < RW; ++q) {
            const int r = r0 + q, c = r / UPM_COLS, dd = r - c * UPM_COLS, i = c * N + d0 + dd;
            const uint4* w4 = reinterpret_cast<const uint4*>(m.a[0].w_up + (size_t) i * LR);
            wa[q] = __ldg(w4 + lane);
            wb[q] = lane < LR / 8 - 32 ? __ldg(w4 + 32 + lane) : make_uint4(0, 0, 0, 0);
        }
        const int er = r0 + vq, ec = er / UPM_COLS, edd = er - ec * UPM_COLS, ei = ec * N + d0 + edd;
        float rv = 0.0f, wn = 0.0f, rsc = 0.0f, bo = 0.0f, ip = 0.0f;
        bool apply = false;
        if (epi) {
            const FusedGrArgs& a = m.a[vk];
            rv = a.R[ei];
            wn = a.w_norm[ei];
            rsc = a.rs[ec];
            apply = a.apply;
            if (apply) { bo = a.bo_prev[d0 + edd]; ip = a.inj_prev[ec]; }
        }
        float v[NV];
#pragma unroll
        for (int q = 0; q < RW; ++q)
#pragma unroll
            for (int k = 0; k < TT; ++k) {
                float acc = dot8h(wa[q], lo4[k][0][lane], lo4[k][1][lane]);
                if (lane < LR / 8 - 32) acc += dot8h(wb[q], lo4[k][0][32 + lane], lo4[k][1][32 + lane]);
                v[q * TT + k] = acc;
            }
#pragma unroll
        for (int e = TT * RW; e < NV; ++e) v[e] = 0.0f;
        xor_scatter<NV>(v, lane);
        if (epi) {
            float x = rv;
            if (apply) {
                x = fmaf(bo, 2.0f * sigmoidf_(ip / (float) HC), x);
                m.a[vk].R_out[ei] = x;
            }
            const float y = x * wn * rsc;
            g[vk][ec][edd] = y * sigmoidf_(v[0]);
        }
    }
    __syncthreads();
    for (int i = t; i < TT * UPM_COLS; i += THREADS) {
        const int k = i / UPM_COLS, col = i - k * UPM_COLS;
        float s = 0.0f;
#pragma unroll
        for (int c = 0; c < HC; ++c) s += g[k][c][col];
        m.a[k].mixed[d0 + col] = s / (float) HC;
    }
}

}  // namespace

void fused_gr_read_multi(const FusedGrArgs* a, int n_tok, float* xn_scratch, void* stream) {
    if (n_tok < 1 || n_tok > kFusedGrMaxT || xn_scratch == nullptr) {
        std::fprintf(stderr, "fused_gr_read_multi: invalid arguments\n");
        std::exit(1);
    }
    GrMulti m;
    for (int t = 0; t < n_tok; ++t) {
        m.a[t] = a[t];
        const FusedGrArgs& x = a[t];
        if (!x.R || !x.w_norm || !x.w_down || !x.w_up || !x.lo || !x.rs || !x.mixed || (x.w_inject && !x.inject_out) ||
            (x.apply && (!x.bo_prev || !x.inj_prev || !x.R_out)) || x.w_down != a[0].w_down || x.w_up != a[0].w_up ||
            x.w_inject != a[0].w_inject || x.w_norm != a[0].w_norm) {
            std::fprintf(stderr, "fused_gr_read_multi: invalid arguments for token %d\n", t);
            std::exit(1);
        }
    }
    m.xn = xn_scratch;
    m.T = n_tok;
    cudaStream_t st = (cudaStream_t) stream;
    gr_norm_multi_kernel<<<dim3((unsigned) n_tok, HC), THREADS, 0, st>>>(m);
    static bool attr = false;
    if (!attr) {   // two buffers of the largest tile: 4 tokens x 320 chunks, or 8 x 160
        const int bytes = 2 * 4 * 2 * 320 * (int) sizeof(float4);
        cudaFuncSetAttribute(gr_down_multi_kernel<320>, cudaFuncAttributeMaxDynamicSharedMemorySize, bytes);
        cudaFuncSetAttribute(gr_down_multi_kernel<160>, cudaFuncAttributeMaxDynamicSharedMemorySize, bytes);
        attr = true;
    }
    if (n_tok <= 4)
        gr_down_multi_kernel<320><<<DOWN_BLOCKS + 1, THREADS, (size_t) 2 * n_tok * 2 * 320 * sizeof(float4), st>>>(m);
    else
        gr_down_multi_kernel<160><<<DOWN_BLOCKS + 1, THREADS, (size_t) 2 * n_tok * 2 * 160 * sizeof(float4), st>>>(m);
    switch (n_tok) {
#define STRATA_UP(T) case T: gr_up_multi_kernel<T><<<UPM_BLOCKS, THREADS, 0, st>>>(m); break;
        STRATA_UP(1) STRATA_UP(2) STRATA_UP(3) STRATA_UP(4) STRATA_UP(5) STRATA_UP(6) STRATA_UP(7) STRATA_UP(8)
#undef STRATA_UP
    }
    const cudaError_t e = cudaGetLastError();
    if (e != cudaSuccess) {
        std::fprintf(stderr, "fused_gr_read_multi: %s\n", cudaGetErrorString(e));
        std::exit(1);
    }
}

bool fused_gr_supported(int64_t n_embd, int64_t hc, int64_t hc_lr) {
    return n_embd == N && hc == HC && hc_lr == LR;
}

void fused_gr_read(const FusedGrArgs& a, void* stream) {
    if (!a.R || !a.w_norm || !a.w_down || !a.w_up || !a.lo || !a.rs || !a.mixed ||
        (a.w_inject && !a.inject_out) || (a.apply && (!a.bo_prev || !a.inj_prev || !a.R_out)) ||
        (a.apply && a.inj_prev == a.inject_out)) {
        std::fprintf(stderr, "fused_gr_read: invalid arguments\n");
        std::exit(1);
    }
    cudaStream_t st = (cudaStream_t) stream;
    gr_down_kernel<<<DOWN_BLOCKS + 1, THREADS, 0, st>>>(a);
    gr_up_kernel<<<UP_BLOCKS, THREADS, 0, st>>>(a);
    const cudaError_t e = cudaGetLastError();
    if (e != cudaSuccess) {
        std::fprintf(stderr, "fused_gr_read: %s\n", cudaGetErrorString(e));
        std::exit(1);
    }
}

}  // namespace strata::kernels
