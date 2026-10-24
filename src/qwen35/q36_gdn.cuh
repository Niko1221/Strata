// src/qwen35/q36_gdn.cuh - the q36 engine's chunked gated delta rule for prompt chunks (--chunked-gdn).
//
// The prompt's recurrence (k_gdn_seq_cols in q36_engine.cu) walks the chunk's tokens one by one: per V head and state
// column  S_t = g_t S_{t-1} + k_t delta_t^T,  delta_t = beta_t (v_t - g_t S_{t-1}^T k_t),  o_t = S_t^T q_t / sqrt(128)
// (g_t = exp(gate_t), gate_t <= 0; S is [128 rows][128 cols], q / k the L2-normed conv outputs). Two barriers per token,
// 2048 tokens in series, 30 layers. Here the tokens are cut into pieces of CH tokens; with c_t the inclusive cumulative
// sum of the gates inside a piece, D_ts = exp(c_t - c_s) (s <= t, so every exponent is <= 0) and W the state at the
// piece's start, unrolling the recurrence gives, per state column,
//     S_t     = e^{c_t} W + sum_{s<=t} D_ts k_s delta_s^T
//     (I + L) delta = b          L[t][s] = beta_t D_ts (k_t . k_s)  (s < t),   b_t = beta_t (v_t - e^{c_t} k_t^T W)
//     o_t     = (e^{c_t} q_t^T W + sum_{s<=t} D_ts (q_t . k_s) delta_s) / sqrt(128)
//     W_out   = e^{c_last} W + sum_s e^{c_last - c_s} k_s delta_s^T
// (k_t^T S_{t-1} g_t = e^{c_t} k_t^T W + sum_{s<t} D_ts (k_t . k_s) delta_s, which is the system's left side). The system
// is lower triangular and unit diagonal, and the columns do not interact, so one thread per column solves it by forward
// substitution with no barrier. `pre` computes L and P[t][s] = D_ts (q_t . k_s) / sqrt(128) for every piece and head at
// once (no dependence on the state); `scan` carries the state through the pieces of one (head, 32 columns). A tail piece
// is zero-padded (gate 0, beta 0, q = k = 0: the identity step). FP32 throughout, CUDA cores. The sums run in another
// order than the token loop: not bit-identical. The state layout is the engine's ([row][head][col]), so decode and verify steps
// read it as before.
#pragma once
#if !defined(__HIPCC__)
#include <cuda_runtime.h>
#include <cstddef>
#include <cstdint>

namespace q36gdn {

constexpr int S = 128;        // head dim (state rows and columns)
constexpr int CH = 16;        // tokens per piece (CH / 8 tokens per warp)
constexpr int TPW = CH / 8;
constexpr int NCOL = 4;       // column groups per head
constexpr int NC = S / NCOL;  // columns per block
constexpr int kThreads = 256;

inline size_t pieces(int T) { return (size_t) (T + CH - 1) / CH; }
/// floats of the two matrix workspaces (L and P) for up to T tokens of h_v heads, and of the cumulative gates
inline size_t mat_floats(int T, int h_v) { return pieces(T) * (size_t) h_v * CH * CH; }
inline size_t cum_floats(int T, int h_v) { return pieces(T) * (size_t) h_v * CH; }

/// Grid (pieces, h_v), 256 threads. h: [T][C] conv outputs (q at qh * S, k at (h_k + qh) * S, v at (2 h_k + head) * S,
/// qh = head % h_k); gate / beta [T][h_v] (gate before exp).
__global__ void __launch_bounds__(kThreads)
pre(const float* __restrict__ h, int C, const float* __restrict__ gate, const float* __restrict__ beta, int h_k, int h_v,
    int T, int nch, float scale, float* __restrict__ Lw, float* __restrict__ Pw, float* __restrict__ cumw) {
    __shared__ float Ks[CH * (S + 1)], Qs[CH * (S + 1)];
    __shared__ float cs[CH], bt[CH];
    const int ci = blockIdx.x, hv = blockIdx.y, tid = threadIdx.x, qh = hv % h_k, t0 = ci * CH;
    for (int idx = tid; idx < CH * S; idx += kThreads) {
        const int t = idx >> 7, i = idx & (S - 1), tg = t0 + t;
        const bool ok = tg < T;
        Ks[t * (S + 1) + i] = ok ? h[(size_t) tg * C + (h_k + qh) * S + i] : 0.f;
        Qs[t * (S + 1) + i] = ok ? h[(size_t) tg * C + qh * S + i] : 0.f;
    }
    if (tid < CH) {
        const int tg = t0 + tid;
        cs[tid] = tg < T ? gate[(size_t) tg * h_v + hv] : 0.f;
        bt[tid] = tg < T ? beta[(size_t) tg * h_v + hv] : 0.f;
    }
    __syncthreads();
    if (tid == 0) {
        float a = 0.f;
        for (int t = 0; t < CH; ++t) {
            a += cs[t];
            cs[t] = a;
        }
    }
    __syncthreads();
    const size_t base = ((size_t) hv * nch + ci);
    if (tid < CH) cumw[base * CH + tid] = cs[tid];
    for (int p = tid; p < CH * CH; p += kThreads) {
        const int t = p / CH, s = p % CH;
        float l = 0.f, pv = 0.f;
        if (s <= t) {
            float kk = 0.f, qk = 0.f;
#pragma unroll 8
            for (int i = 0; i < S; ++i) {
                const float ks = Ks[s * (S + 1) + i];
                kk = fmaf(Ks[t * (S + 1) + i], ks, kk);
                qk = fmaf(Qs[t * (S + 1) + i], ks, qk);
            }
            const float D = __expf(cs[t] - cs[s]);
            l = s < t ? bt[t] * D * kk : 0.f;
            pv = D * qk * scale;
        }
        Lw[base * CH * CH + p] = l;
        Pw[base * CH * CH + p] = pv;
    }
}

constexpr size_t scan_smem_bytes() { return (size_t) (2 * CH * S + S * NC + CH * NC + 2 * CH * CH + 2 * CH) * sizeof(float); }

/// Grid (h_v, NCOL), 256 threads (lane = a column of the block's 32, warp = a group of CH / 8 tokens and 16 state rows),
/// scan_smem_bytes() of dynamic shared memory. state [S rows][h_v][S cols] in place; o_raw [T][h_v][S] the raw outputs
/// (what k_gdn_seq_cols writes; k_gdn_norm follows).
__global__ void __launch_bounds__(kThreads)
scan(float* __restrict__ state, const float* __restrict__ h, int C, const float* __restrict__ beta,
     const float* __restrict__ Lw, const float* __restrict__ Pw, const float* __restrict__ cumw, float* __restrict__ o_raw,
     int h_k, int h_v, int T, int nch, float scale) {
    extern __shared__ __align__(16) float sm[];
    float* Ks = sm;
    float* Qs = Ks + CH * S;
    float* W = Qs + CH * S;       // [S][NC]
    float* Dl = W + S * NC;       // [CH][NC]
    float* Ls = Dl + CH * NC;     // [CH][CH]
    float* Ps = Ls + CH * CH;     // [CH][CH]
    float* cs = Ps + CH * CH;
    float* bt = cs + CH;
    const int tid = threadIdx.x, tx = tid & 31, ty = tid >> 5;
    const int hv = blockIdx.x, j0 = blockIdx.y * NC, qh = hv % h_k;
    for (int i = ty; i < S; i += 8) W[i * NC + tx] = state[((size_t) i * h_v + hv) * S + j0 + tx];
    // the next piece's inputs are fetched into registers while this one is computed (a piece is ~10 us of work, a cold load ~1)
    constexpr int NKQ = CH * S / 4 / kThreads;   // float4 of K and of Q per thread
    static_assert(CH * S / 4 % kThreads == 0 && CH * CH / 4 <= kThreads, "piece tiles");
    struct Fetch { float4 k[NKQ], q[NKQ]; float4 l, p; float cs, bt, v[TPW]; };
    auto fetch = [&](int ci, Fetch& f) {
        const int t0 = ci * CH;
        const size_t base = (size_t) hv * nch + ci;
#pragma unroll
        for (int m = 0; m < NKQ; ++m) {
            const int idx = tid + m * kThreads, t = idx >> 5, i4 = (idx & 31) * 4, tg = t0 + t;
            f.k[m] = f.q[m] = make_float4(0.f, 0.f, 0.f, 0.f);
            if (tg < T) {
                f.k[m] = *reinterpret_cast<const float4*>(h + (size_t) tg * C + (h_k + qh) * S + i4);
                f.q[m] = *reinterpret_cast<const float4*>(h + (size_t) tg * C + qh * S + i4);
            }
        }
        if (tid < CH * CH / 4) {
            f.l = *reinterpret_cast<const float4*>(Lw + base * CH * CH + tid * 4);
            f.p = *reinterpret_cast<const float4*>(Pw + base * CH * CH + tid * 4);
        }
        if (tid < CH) {
            f.cs = cumw[base * CH + tid];
            f.bt = t0 + tid < T ? beta[(size_t) (t0 + tid) * h_v + hv] : 0.f;
        }
#pragma unroll
        for (int t = 0; t < TPW; ++t) {
            const int tg = t0 + ty * TPW + t;
            f.v[t] = tg < T ? h[(size_t) tg * C + (2 * h_k + hv) * S + j0 + tx] : 0.f;
        }
    };
    Fetch cur, nxt;
    fetch(0, nxt);
    for (int ci = 0; ci < nch; ++ci) {
        const int t0 = ci * CH;
        cur = nxt;
#pragma unroll
        for (int m = 0; m < NKQ; ++m) {
            const int idx = tid + m * kThreads, t = idx >> 5, i4 = (idx & 31) * 4;
            *reinterpret_cast<float4*>(Ks + t * S + i4) = cur.k[m];
            *reinterpret_cast<float4*>(Qs + t * S + i4) = cur.q[m];
        }
        if (tid < CH * CH / 4) {
            *reinterpret_cast<float4*>(Ls + tid * 4) = cur.l;
            *reinterpret_cast<float4*>(Ps + tid * 4) = cur.p;
        }
        if (tid < CH) {
            cs[tid] = cur.cs;
            bt[tid] = cur.bt;
        }
        if (ci + 1 < nch) fetch(ci + 1, nxt);
        __syncthreads();
        // U = K W and QW = Q W for the warp's CH / 8 tokens and this lane's column
        float u[TPW], qw[TPW];
#pragma unroll
        for (int t = 0; t < TPW; ++t) u[t] = qw[t] = 0.f;
#pragma unroll 4
        for (int i4 = 0; i4 < S / 4; ++i4) {
            const float w0 = W[(4 * i4) * NC + tx], w1 = W[(4 * i4 + 1) * NC + tx], w2 = W[(4 * i4 + 2) * NC + tx],
                        w3 = W[(4 * i4 + 3) * NC + tx];
#pragma unroll
            for (int t = 0; t < TPW; ++t) {
                const float4 kv = *reinterpret_cast<const float4*>(Ks + (ty * TPW + t) * S + 4 * i4);
                const float4 qv = *reinterpret_cast<const float4*>(Qs + (ty * TPW + t) * S + 4 * i4);
                u[t] = fmaf(kv.x, w0, fmaf(kv.y, w1, fmaf(kv.z, w2, fmaf(kv.w, w3, u[t]))));
                qw[t] = fmaf(qv.x, w0, fmaf(qv.y, w1, fmaf(qv.z, w2, fmaf(qv.w, w3, qw[t]))));
            }
        }
#pragma unroll
        for (int t = 0; t < TPW; ++t) {
            const int tt = ty * TPW + t;
            Dl[tt * NC + tx] = bt[tt] * (cur.v[t] - __expf(cs[tt]) * u[t]);
        }
        __syncthreads();
        if (ty == 0) {   // (I + L) delta = b, one column per lane, no barrier
            float d[CH];
#pragma unroll
            for (int t = 0; t < CH; ++t) {
                float a = Dl[t * NC + tx];
#pragma unroll
                for (int s = 0; s < t; ++s) a = fmaf(-Ls[t * CH + s], d[s], a);
                d[t] = a;
                Dl[t * NC + tx] = a;
            }
        }
        __syncthreads();
#pragma unroll
        for (int t = 0; t < TPW; ++t) {
            const int tt = ty * TPW + t, tg = t0 + tt;
            float a = 0.f;
            for (int s = 0; s <= tt; ++s) a = fmaf(Ps[tt * CH + s], Dl[s * NC + tx], a);
            if (tg < T) o_raw[(size_t) tg * h_v * S + hv * S + j0 + tx] = a + scale * __expf(cs[tt]) * qw[t];
        }
        __syncthreads();
        for (int idx = tid; idx < CH * NC; idx += kThreads) Dl[idx] *= __expf(cs[CH - 1] - cs[idx / NC]);
        __syncthreads();
        // W <- e^{c_last} W + K^T Dl for the warp's 16 state rows
        float wa[16];
        const float es = __expf(cs[CH - 1]);
#pragma unroll
        for (int r = 0; r < 16; ++r) wa[r] = es * W[(ty * 16 + r) * NC + tx];
#pragma unroll 4
        for (int s = 0; s < CH; ++s) {
            const float d = Dl[s * NC + tx];
#pragma unroll
            for (int r4 = 0; r4 < 4; ++r4) {
                const float4 kv = *reinterpret_cast<const float4*>(Ks + s * S + ty * 16 + 4 * r4);
                wa[4 * r4] = fmaf(kv.x, d, wa[4 * r4]);
                wa[4 * r4 + 1] = fmaf(kv.y, d, wa[4 * r4 + 1]);
                wa[4 * r4 + 2] = fmaf(kv.z, d, wa[4 * r4 + 2]);
                wa[4 * r4 + 3] = fmaf(kv.w, d, wa[4 * r4 + 3]);
            }
        }
#pragma unroll
        for (int r = 0; r < 16; ++r) W[(ty * 16 + r) * NC + tx] = wa[r];
        __syncthreads();
    }
    for (int i = ty; i < S; i += 8) state[((size_t) i * h_v + hv) * S + j0 + tx] = W[i * NC + tx];
}

/// The chunked recurrence for T tokens: same arguments and result as k_gdn_seq_cols (raw outputs `o_raw`, state in
/// place). Lw / Pw: mat_floats(T, h_v) floats each, cumw: cum_floats(T, h_v).
inline cudaError_t launch(float* state, const float* h, int C, const float* gate, const float* beta, float* o_raw, int h_k,
                          int h_v, int T, float* Lw, float* Pw, float* cumw, cudaStream_t st) {
    static bool attr = false;
    if (!attr) {
        cudaFuncSetAttribute(scan, cudaFuncAttributeMaxDynamicSharedMemorySize, (int) scan_smem_bytes());
        attr = true;
    }
    const int nch = (int) pieces(T);
    const float scale = 1.0f / sqrtf((float) S);
    pre<<<dim3(nch, h_v), kThreads, 0, st>>>(h, C, gate, beta, h_k, h_v, T, nch, scale, Lw, Pw, cumw);
    scan<<<dim3(h_v, NCOL), kThreads, scan_smem_bytes(), st>>>(state, h, C, beta, Lw, Pw, cumw, o_raw, h_k, h_v, T, nch, scale);
    return cudaGetLastError();
}

}  // namespace q36gdn
#else
// HIP: no kernels; the engine clears --chunked-gdn there (the references in q36_engine.cu still have to compile)
#include <cuda_runtime.h>
namespace q36gdn {
inline size_t mat_floats(int, int) { return 0; }
inline size_t cum_floats(int, int) { return 0; }
inline cudaError_t launch(float*, const float*, int, const float*, const float*, float*, int, int, int, float*, float*, float*,
                          cudaStream_t) {
    return cudaErrorNotSupported;
}
}  // namespace q36gdn
#endif
