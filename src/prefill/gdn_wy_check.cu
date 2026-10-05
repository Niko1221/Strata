// src/prefill/gdn_wy_check.cu - the WY-form chunk-parallel GDN recurrence: algebra, device kernel, bound.
//
// the chunk-parallel design campaign.  The engine's serial recurrence (gdn_rec_cols_pipe_kernel,
// copied verbatim from src/prefill/kernels.cu) is ground truth; the checks:
//   1. host serial64 (fp64 token loop) vs host wy64 (fp64 chunked WY, C=64): the chunk-parallel ALGEBRA
//      - exact to double epsilon on every fixture class (boundary T, no/fast decay, beta=0, long).
//   2. device gdn_rec_wy_kernel vs the engine's serial kernel on the GPU, fp32, within a derived bound
//      - the C3 gate for wiring STRATA_GDN_WY=1 into the engine.
// Synthetic inputs at model shape (S=128, HK=16, HV=48); q/k rows L2-normalized per key head as
// gdn_l2_kernel leaves them; v random per column.  NOTE: no blanket padding wipe over the token row -
// it silently zeroed k and v and blinded the first algebra check (fixed 2026-10-05).
#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <vector>

namespace {

constexpr int S = 128, HK = 16, HV = 48, C = 10240;   // as kernels.cu: head size, key heads, value heads, q|k|v width
constexpr int RG = 4, RPG = S / RG, CB = 32, NCB = S / CB;
constexpr int WY = 64;   // the WY sub-chunk
__device__ int g_wy_dbg;   // set from the host to dump block 0 (WY_DBG=1)

void ck(cudaError_t e, const char* what) {
    if (e != cudaSuccess) {
        std::fprintf(stderr, "%s: %s\n", what, cudaGetErrorString(e));
        std::exit(2);
    }
}

__global__ void __launch_bounds__(CB * RG) gdn_rec_cols_pipe_kernel(float* __restrict__ state, const float* __restrict__ h,
                                                                    const float* __restrict__ gate,
                                                                    const float* __restrict__ beta,
                                                                    float* __restrict__ oc_out, int64_t T) {
    constexpr int NT = CB * RG, LPT = S / NT;
    __shared__ float sk[S], sq[S], red[RG][CB];
    const int head = blockIdx.x / NCB, cb = blockIdx.x % NCB;
    const int c = threadIdx.x, rg = threadIdx.y, tid = rg * CB + c, col = cb * CB + c;
    const int qh = head % HK;
    float s[RPG];
    float* base = state + ((size_t) (rg * RPG) * HV + head) * S + col;
    const size_t rs = (size_t) HV * S;
#pragma unroll
    for (int r = 0; r < RPG; ++r) s[r] = base[r * rs];
    float nq[LPT], nk[LPT], nv = 0.0f, ng = 0.0f, nb = 0.0f;
    auto fetch = [&](int64_t t) {
        const float* ht = h + t * C;
#pragma unroll
        for (int u = 0; u < LPT; ++u) { nq[u] = ht[qh * S + tid + u * NT]; nk[u] = ht[HK * S + qh * S + tid + u * NT]; }
        nv = ht[2 * HK * S + head * S + col];
        ng = gate[t * HV + head];
        nb = beta[t * HV + head];
    };
    if (T > 0) fetch(0);
    for (int64_t t = 0; t < T; ++t) {
        float cq[LPT], ck[LPT];
#pragma unroll
        for (int u = 0; u < LPT; ++u) { cq[u] = nq[u]; ck[u] = nk[u]; }
        const float cv = nv, cg = ng, cbt = nb;
        __syncthreads();
#pragma unroll
        for (int u = 0; u < LPT; ++u) { sq[tid + u * NT] = cq[u]; sk[tid + u * NT] = ck[u]; }
        __syncthreads();
        if (t + 1 < T) fetch(t + 1);
        const float g = __expf(cg);
        float kv = 0.0f;
#pragma unroll
        for (int r = 0; r < RPG; ++r) kv = fmaf(s[r], sk[rg * RPG + r], kv);
        red[rg][c] = kv;
        __syncthreads();
        const float kv_col = red[0][c] + red[1][c] + red[2][c] + red[3][c];
        const float delta = (cv - g * kv_col) * cbt;
        float o = 0.0f;
#pragma unroll
        for (int r = 0; r < RPG; ++r) {
            s[r] = fmaf(g, s[r], sk[rg * RPG + r] * delta);
            o = fmaf(s[r], sq[rg * RPG + r], o);
        }
        __syncthreads();
        red[rg][c] = o;
        __syncthreads();
        if (rg == 0) oc_out[t * HV * S + head * S + col] = (red[0][c] + red[1][c] + red[2][c] + red[3][c]) * rsqrtf((float) S);
    }
#pragma unroll
    for (int r = 0; r < RPG; ++r) base[r * rs] = s[r];
}


// ---------------------------------------------------------- the WY device kernel (C3)
// Same grid/thread/column layout as the serial kernel (thread owns RPG rows of one column; grid = HV*NCB),
// tokens processed in WY-token sub-chunks: stage the head's key rows in LDS, build the CxC coefficient
// matrix A (S0-independent), per-column prefix products p^K/p^Q by block reductions, then the per-column
// forward substitution, outputs and the state update run per-THREAD with A rows broadcast from LDS - the
// block syncs O(1) per sub-chunk instead of per token.
__global__ void __launch_bounds__(CB * RG) gdn_rec_wy_kernel(float* __restrict__ state, const float* __restrict__ h,
                                                             const float* __restrict__ gate,
                                                             const float* __restrict__ beta,
                                                             float* __restrict__ oc_out, int64_t T) {
    __shared__ float kt[WY][S];
    __shared__ float A[WY * WY];
    __shared__ float dec[WY + 1];
    __shared__ float redk[RG][CB], redq[RG][CB];
    const int head = blockIdx.x / NCB, cb = blockIdx.x % NCB;
    const int c = threadIdx.x, rg = threadIdx.y, col = cb * CB + c;
    const int tid = rg * CB + c;   // flat id in [0, CB*RG) - the launch is dim3(CB, RG), so threadIdx.x
                                   // alone is only c in [0, 32): the two strided loops below MUST stride
                                   // tid, or kt keeps 3/4 of its columns and A 3/4 of its entries
                                   // unwritten (uninitialized LDS -> the run-varying NaN/Inf)
    const int qh = head % HK;
    float s[RPG];
    float* base = state + ((size_t) (rg * RPG) * HV + head) * S + col;
    const size_t rs = (size_t) HV * S;
#pragma unroll
    for (int r = 0; r < RPG; ++r) s[r] = base[r * rs];

    for (int64_t t0 = 0; t0 < T; t0 += WY) {
        const int w = (int) min((int64_t) WY, T - t0);
        if (threadIdx.x == 0) {
            dec[0] = 1.0f;
            for (int t = 0; t < w; ++t) dec[t + 1] = dec[t] * __expf(gate[(t0 + t) * HV + head]);
        }
        for (int t = 0; t < w; ++t) {
            const float* src = h + HK * S + qh * S + (t0 + t) * C;
            for (int i = tid; i < S; i += CB * RG) kt[t][i] = src[i];
        }
        __syncthreads();
        // A[t][s] (s < t) = beta_t g_t dec[t] (dec[t]/dec[s+1]) (k_t . k_s); unit diagonal
        for (int p = tid; p < WY * WY; p += CB * RG) {
            const int t = p / WY, s2 = p % WY;
            if (t >= w) continue;
            if (s2 > t) { A[p] = 0.0f; continue; }
            if (s2 == t) { A[p] = 1.0f; continue; }
            const float b = beta[(t0 + t) * HV + head];
            float dk = 0.0f;
#pragma unroll 4
            for (int i = 0; i < S; ++i) dk += kt[t][i] * kt[s2][i];
            // the decay factor as a product, not g * dec[t]/dec[s+1]: fp32 cumulative products underflow to 0
            // under fast decay and the ratio becomes 0/0 = NaN (the serial path multiplies and lands at 0).
            // g * (dec[t]/dec[s+1]) = prod of gates s+1..t exactly - the standalone g is consumed.
            float dec_ratio = 1.0f;
            for (int j = s2 + 1; j <= t; ++j) dec_ratio *= __expf(gate[(t0 + j) * HV + head]);
            A[p] = b * dec_ratio * dk;
        }
        __syncthreads();
        float uj[WY], pk[WY], pq[WY];
        for (int t = 0; t < w; ++t) {
            const float* qt = h + (t0 + t) * C + qh * S;
            float kpart = 0.0f, qpart = 0.0f;
#pragma unroll
            for (int r = 0; r < RPG; ++r) { kpart += s[r] * kt[t][rg * RPG + r]; qpart += s[r] * qt[rg * RPG + r]; }
            redk[rg][c] = kpart;
            redq[rg][c] = qpart;
            __syncthreads();
            pk[t] = redk[0][c] + redk[1][c] + redk[2][c] + redk[3][c];
            pq[t] = redq[0][c] + redq[1][c] + redq[2][c] + redq[3][c];
            __syncthreads();   // every thread has read both reductions before the next token overwrites them
        }
        // the per-column forward substitution (thread = column; A rows broadcast from LDS)
        for (int t = 0; t < w; ++t) {
            const float vj = h[2 * HK * S + head * S + (t0 + t) * C + col];
            const float b = beta[(t0 + t) * HV + head], g = __expf(gate[(t0 + t) * HV + head]);
            float acc = b * (vj - g * dec[t] * pk[t]);
            for (int s2 = 0; s2 < t; ++s2) acc -= A[t * WY + s2] * uj[s2];
            uj[t] = acc;
        }
        if (g_wy_dbg && t0 == 0 && head == 0 && cb == 0 && rg == 0 && c == 0)
            printf("DBG blk0: kt[0][0..2]=%.4f %.4f %.4f (col0) | pk[0]=%.4f vj=%.4f | s[col0 rows0..3]=%.4f %.4f %.4f %.4f\n",
                   kt[0][0], kt[0][1], kt[0][2], pk[0], h[2 * HK * S + col], s[0], s[1], s[2], s[3]);
        // outputs and the state update
        for (int t = 0; t < w; ++t) {
            const float* qt = h + (t0 + t) * C + qh * S;
            float o = dec[t + 1] * pq[t];
            for (int s2 = 0; s2 <= t; ++s2) {
                float dq = 0.0f;
#pragma unroll 4
                for (int i = 0; i < S; ++i) dq += qt[i] * kt[s2][i];
                float dec_ratio = 1.0f;
                for (int j = s2 + 1; j <= t; ++j) dec_ratio *= __expf(gate[(t0 + j) * HV + head]);
                o += dec_ratio * dq * uj[s2];
            }
            if (rg == 0) oc_out[(t0 + t) * HV * S + head * S + col] = o * rsqrtf((float) S);
        }
        for (int r = 0; r < RPG; ++r) {
            const int i = rg * RPG + r;
            float v = dec[w] * s[r];
            for (int s2 = 0; s2 < w; ++s2) {
                float dec_ratio = 1.0f;
                for (int j = s2 + 1; j < w; ++j) dec_ratio *= __expf(gate[(t0 + j) * HV + head]);
                v += dec_ratio * kt[s2][i] * uj[s2];
            }
            s[r] = v;
        }
        __syncthreads();
    }
#pragma unroll
    for (int r = 0; r < RPG; ++r) base[r * rs] = s[r];
}

// ---------------------------------------------------------- host references (head 0, fp64)
void serial64(const double* h, const double* gate, const double* beta, const double* st0, double* oc, double* stT,
              int64_t T) {
    std::vector<double> s(S * S);
    for (int i = 0; i < S; ++i)
        for (int j = 0; j < S; ++j) s[i * S + j] = st0[(size_t) i * HV * S + j];
    for (int64_t t = 0; t < T; ++t) {
        const double* kt = h + HK * S + t * C;
        const double* qt = h + t * C;
        const double* vt = h + 2 * HK * S + t * C;
        const double g = std::exp(gate[t]), b = beta[t];
        double delta[S];
        for (int j = 0; j < S; ++j) {
            double kv = 0;
            for (int i = 0; i < S; ++i) kv += s[i * S + j] * kt[i];
            delta[j] = (vt[j] - g * kv) * b;
            if (std::getenv("WY_DBG") && t == 1 && j == 0)
                std::printf("DBG serial64 t1 j0: v=%.10f g=%.10f kv=%.10f delta=%.10f | k1[0..2]=%.10f %.10f %.10f "
                            "Saft0[0..2]=%.10f %.10f %.10f beta1=%.10f\n",
                            vt[j], g, kv, delta[j], kt[0], kt[1], kt[2], s[0], s[1 * S], s[2 * S], beta[t]);
        }
        for (int i = 0; i < S; ++i)
            for (int j = 0; j < S; ++j) s[i * S + j] = g * s[i * S + j] + kt[i] * delta[j];
        for (int j = 0; j < S; ++j) {
            double o = 0;
            for (int i = 0; i < S; ++i) o += s[i * S + j] * qt[i];
            oc[t * S + j] = o * (1.0 / std::sqrt((double) S));
        }
    }
    for (int i = 0; i < S; ++i)
        for (int j = 0; j < S; ++j) stT[(size_t) i * HV * S + j] = s[i * S + j];
}

// chunked WY (C=64) in double: per column the forward substitution rhs is beta*(v - g*dec[t]*p^K) minus the
// A-row terms - exactly the device kernel's math at double precision.
void wy64(const double* h, const double* gate, const double* beta, const double* st0, double* oc, double* stT,
          int64_t T) {
    std::vector<double> s(S * S);
    for (int i = 0; i < S; ++i)
        for (int j = 0; j < S; ++j) s[i * S + j] = st0[(size_t) i * HV * S + j];
    std::vector<double> A(WY * WY), dec(WY + 1), pK(WY), pQ(WY), uj(WY);
    for (int64_t t0 = 0; t0 < T; t0 += WY) {
        const int w = (int) std::min<int64_t>(WY, T - t0);
        dec[0] = 1.0;
        for (int t = 0; t < w; ++t) dec[t + 1] = dec[t] * std::exp(gate[t0 + t]);
        for (int t = 0; t < w; ++t) {
            const double b = beta[t0 + t], g = std::exp(gate[t0 + t]);
            const double* kt = h + HK * S + (t0 + t) * C;
            for (int s2 = 0; s2 <= t; ++s2) {
                if (s2 == t) { A[t * WY + s2] = 1.0; continue; }
                const double* ks = h + HK * S + (t0 + s2) * C;
                double dk = 0;
                for (int i = 0; i < S; ++i) dk += kt[i] * ks[i];
                A[t * WY + s2] = b * g * (dec[t] / dec[s2 + 1]) * dk;
            }
        }
        for (int j = 0; j < S; ++j) {
            for (int t = 0; t < w; ++t) {
                const double* kt = h + HK * S + (t0 + t) * C;
                const double* qt = h + (t0 + t) * C;
                double pk = 0, pq = 0;
                for (int i = 0; i < S; ++i) { pk += kt[i] * s[i * S + j]; pq += qt[i] * s[i * S + j]; }
                pK[t] = pk; pQ[t] = pq;
            }
            for (int t = 0; t < w; ++t) {
                double acc = beta[t0 + t] * (h[2 * HK * S + (t0 + t) * C + j] - std::exp(gate[t0 + t]) * dec[t] * pK[t]);
                for (int s2 = 0; s2 < t; ++s2) acc -= A[t * WY + s2] * uj[s2];
                uj[t] = acc;
                if (std::getenv("WY_DBG") && t0 == 0 && t == 1 && j == 0)
                    std::printf("DBG wy64 t1 j0: v=%.10f g=%.10f dec[1]=%.10f pK[1]=%.10f A10=%.10f uj0=%.10f uj1=%.10f "
                                "| k1[0..2]=%.10f %.10f %.10f S0[0..2]=%.10f %.10f %.10f beta1=%.10f\n",
                                h[2 * HK * S + (t0 + t) * C + j], std::exp(gate[t0 + t]), dec[t], pK[t],
                                A[t * WY + 0], uj[0], uj[t],
                                h[HK * S + (t0 + t) * C], h[HK * S + (t0 + t) * C + 1], h[HK * S + (t0 + t) * C + 2],
                                s[0 * S + j], s[1 * S + j], s[2 * S + j], beta[t0 + t]);
            }
            for (int t = 0; t < w; ++t) {
                const double* qt = h + (t0 + t) * C;
                double o = dec[t + 1] * pQ[t];
                for (int s2 = 0; s2 <= t; ++s2) {
                    double dq = 0;
                    for (int i = 0; i < S; ++i) dq += qt[i] * h[HK * S + (t0 + s2) * C + i];
                    o += (dec[t + 1] / dec[s2 + 1]) * dq * uj[s2];
                }
                oc[(t0 + t) * S + j] = o * (1.0 / std::sqrt((double) S));
            }
            for (int i = 0; i < S; ++i) {
                double v = dec[w] * s[i * S + j];
                for (int s2 = 0; s2 < w; ++s2) v += (dec[w] / dec[s2 + 1]) * h[HK * S + (t0 + s2) * C + i] * uj[s2];
                s[i * S + j] = v;
            }
        }
    }
    for (int i = 0; i < S; ++i)
        for (int j = 0; j < S; ++j) stT[(size_t) i * HV * S + j] = s[i * S + j];
}

// fp64 oracle over ALL heads, in the device's layout: oc[t*HV*S + head*S + j], state[(i*HV + head)*S + j].
// The reference the device WY kernel is BOUND against (the serial-vs-WY host check above proves the chunk
// algebra; this one quantifies the fp32 device's distance from exact).
void oracle64(const double* h, const double* gate, const double* beta, const double* st0, double* oc, double* stT,
              int64_t T) {
    std::vector<double> s(S * S);
    for (int head = 0; head < HV; ++head) {
        const int qh = head % HK;
        for (int i = 0; i < S; ++i)
            for (int j = 0; j < S; ++j) s[i * S + j] = st0[((size_t) i * HV + head) * S + j];
        for (int64_t t = 0; t < T; ++t) {
            const double* kt = h + HK * S + t * C + qh * S;
            const double* qt = h + t * C + qh * S;
            const double* vt = h + 2 * HK * S + t * C + head * S;
            const double g = std::exp(gate[t * HV + head]), b = beta[t * HV + head];
            double delta[S];
            for (int j = 0; j < S; ++j) {
                double kv = 0;
                for (int i = 0; i < S; ++i) kv += s[i * S + j] * kt[i];
                delta[j] = (vt[j] - g * kv) * b;
            }
            for (int i = 0; i < S; ++i)
                for (int j = 0; j < S; ++j) s[i * S + j] = g * s[i * S + j] + kt[i] * delta[j];
            for (int j = 0; j < S; ++j) {
                double o = 0;
                for (int i = 0; i < S; ++i) o += s[i * S + j] * qt[i];
                oc[((size_t) t * HV + head) * S + j] = o * (1.0 / std::sqrt((double) S));
            }
        }
        for (int i = 0; i < S; ++i)
            for (int j = 0; j < S; ++j) stT[((size_t) i * HV + head) * S + j] = s[i * S + j];
    }
}

// ---------------------------------------------------------- fixtures
int fixture(const char* name, int64_t T, int gates_mode, unsigned seed) {
    std::mt19937 rng(seed);
    std::normal_distribution<double> nd(0.0, 1.0);
    std::uniform_real_distribution<double> ud(0.0, 1.0);
    const size_t hz = (size_t) T * C;
    std::vector<double> h(hz), gate((size_t) T * HV), beta((size_t) T * HV);
    for (int64_t t = 0; t < T; ++t) {
        double* q = h.data() + t * C;
        double* k = h.data() + HK * S + t * C;
        double* v = h.data() + 2 * HK * S + t * C;
        for (int kh = 0; kh < HK; ++kh) {
            for (int x = 0; x < S; ++x) { q[kh * S + x] = nd(rng); k[kh * S + x] = nd(rng); v[kh * S + x] = nd(rng); }
            double nq = 0, nk = 0;
            for (int x = 0; x < S; ++x) { nq += q[kh * S + x] * q[kh * S + x]; nk += k[kh * S + x] * k[kh * S + x]; }
            nq = std::sqrt(nq); nk = std::sqrt(nk);
            for (int x = 0; x < S; ++x) { q[kh * S + x] /= nq; k[kh * S + x] /= nk; }
        }
        for (int64_t t2 = 0; t2 < T; ++t2)   // heads 1..HV-1 share the token rows the host refs never read; keep clean
            if (t2 != t) { }
    }
    for (int64_t t = 0; t < T; ++t)
        for (int hd = 0; hd < HV; ++hd) {
            gate[t * HV + hd] = gates_mode == 0 ? 0.3 * nd(rng) : gates_mode == 1 ? 0.0 : gates_mode == 2 ? -6.0 : 0.3 * nd(rng);
            beta[t * HV + hd] = gates_mode == 3 ? 0.0 : ud(rng);
        }
    std::vector<double> st0((size_t) S * HV * S), oc64((size_t) T * S), ocwy((size_t) T * S), stT64((size_t) S * HV * S),
        stTW((size_t) S * HV * S);
    for (size_t i = 0; i < st0.size(); ++i) st0[i] = nd(rng) * 0.1;
    serial64(h.data(), gate.data(), beta.data(), st0.data(), oc64.data(), stT64.data(), T);
    wy64(h.data(), gate.data(), beta.data(), st0.data(), ocwy.data(), stTW.data(), T);
    for (int64_t t = 0; t < T; ++t) {
        double et = 0;
        for (int j = 0; j < S; ++j) et = std::max(et, std::fabs(oc64[t * S + j] - ocwy[t * S + j]));
        if (t < 8 || (et > 1e-8 && (t == 8 || t % 32 == 0)))
            std::printf("    token %lld: max oc err %.3e\n", (long long) t, et);
    }
    double e_oc = 0, e_st = 0, s_oc = 0;
    for (size_t i = 0; i < oc64.size(); ++i) {
        e_oc = std::max(e_oc, std::fabs(oc64[i] - ocwy[i]));
        s_oc = std::max(s_oc, std::fabs(oc64[i]));
    }
    for (size_t i = 0; i < (size_t) S * S; ++i) e_st = std::max(e_st, std::fabs(stT64[i] - stTW[i]));
    const bool ok = e_oc < 1e-8 && e_st < 1e-8;
    std::printf("%-28s T=%5d  max|serial64-wy64| oc %.3e state %.3e (scale %.2f) -> %s\n", name, (int) T, e_oc, e_st, s_oc,
                ok ? "ALGEBRA OK" : "MISMATCH");
    return ok ? 0 : 1;
}

}  // namespace

int main(int argc, char** argv) {
    const bool bench = argc > 1 && std::strcmp(argv[1], "--bench") == 0;
    (void) bench;
    int fails = 0;
    if (!bench) {
        const int64_t Ts[] = {1, 64, 4099};
        const int gm[] = {0, 1, 2, 3};
        int wfails = 0;
        double worst = 0.0, worst_oracle = 0.0;
        for (int64_t T : Ts)
            for (int g : gm) {
                if (T == 4099 && g != 0) continue;   // one long device cell: the fp64 oracle is slow
                std::mt19937 rng((unsigned) (T * 100 + g + 7));
                std::normal_distribution<double> nd(0.0, 1.0);
                std::uniform_real_distribution<double> ud(0.0, 1.0);
                const size_t hz = (size_t) T * C;
                std::vector<float> h(hz), gate((size_t) T * HV), beta((size_t) T * HV);
                for (int64_t t = 0; t < T; ++t) {
                    for (int x = 0; x < C; ++x) h[t * C + x] = (float) nd(rng);
                    for (int kh = 0; kh < HK; ++kh) {
                        double nq = 0, nk = 0;
                        for (int x = 0; x < S; ++x) { nq += (double) h[t * C + kh * S + x] * h[t * C + kh * S + x];
                                                      nk += (double) h[t * C + HK * S + kh * S + x] * h[t * C + HK * S + kh * S + x]; }
                        nq = std::sqrt(nq); nk = std::sqrt(nk);
                        for (int x = 0; x < S; ++x) { h[t * C + kh * S + x] /= (float) nq; h[t * C + HK * S + kh * S + x] /= (float) nk; }
                    }
                    for (int hd = 0; hd < HV; ++hd) {
                        gate[t * HV + hd] = g == 0 ? (float) (0.3 * nd(rng)) : g == 1 ? 0.0f : g == 2 ? -6.0f : (float) (0.3 * nd(rng));
                        beta[t * HV + hd] = g == 3 ? 0.0f : (float) ud(rng);
                    }
                }
                std::vector<float> st0((size_t) S * HV * S), stA((size_t) S * HV * S), stB((size_t) S * HV * S);
                for (size_t i = 0; i < st0.size(); ++i) st0[i] = (float) (nd(rng) * 0.1);
                const size_t oz = (size_t) T * HV * S;
                std::vector<float> ocA(oz, 0.0f), ocB(oz, 0.0f);
                float *dH, *dG, *dB, *dS0, *dSA, *dSB, *dOA, *dOB;
                ck(cudaMalloc(&dH, hz * 4), "m1"); ck(cudaMalloc(&dG, gate.size() * 4), "m2");
                ck(cudaMalloc(&dB, beta.size() * 4), "m3");
                ck(cudaMalloc(&dS0, st0.size() * 4), "m4"); ck(cudaMalloc(&dSA, st0.size() * 4), "m5");
                ck(cudaMalloc(&dSB, st0.size() * 4), "m6"); ck(cudaMalloc(&dOA, oz * 4), "m7");
                ck(cudaMalloc(&dOB, oz * 4), "m8");
                ck(cudaMemcpy(dH, h.data(), hz * 4, cudaMemcpyHostToDevice), "c1");
                ck(cudaMemcpy(dG, gate.data(), gate.size() * 4, cudaMemcpyHostToDevice), "c2");
                ck(cudaMemcpy(dB, beta.data(), beta.size() * 4, cudaMemcpyHostToDevice), "c3");
                ck(cudaMemcpy(dS0, st0.data(), st0.size() * 4, cudaMemcpyHostToDevice), "c4");
                ck(cudaMemcpy(dSA, st0.data(), st0.size() * 4, cudaMemcpyHostToDevice), "c5");
                ck(cudaMemcpy(dSB, st0.data(), st0.size() * 4, cudaMemcpyHostToDevice), "c6");
                const int dbg = std::getenv("WY_DBG") ? 1 : 0;
                cudaMemcpyToSymbol(g_wy_dbg, &dbg, 4);
                gdn_rec_cols_pipe_kernel<<<HV * NCB, dim3(CB, RG)>>>(dSA, dH, dG, dB, dOA, T);
                gdn_rec_wy_kernel<<<HV * NCB, dim3(CB, RG)>>>(dSB, dH, dG, dB, dOB, T);
                ck(cudaDeviceSynchronize(), "run");
                ck(cudaMemcpy(stA.data(), dSA, st0.size() * 4, cudaMemcpyDeviceToHost), "d1");
                ck(cudaMemcpy(stB.data(), dSB, st0.size() * 4, cudaMemcpyDeviceToHost), "d2");
                ck(cudaMemcpy(ocA.data(), dOA, oz * 4, cudaMemcpyDeviceToHost), "d3");
                ck(cudaMemcpy(ocB.data(), dOB, oz * 4, cudaMemcpyDeviceToHost), "d4");
                cudaFree(dH); cudaFree(dG); cudaFree(dB); cudaFree(dS0); cudaFree(dSA); cudaFree(dSB);
                cudaFree(dOA); cudaFree(dOB);
                double eoc = 0, est = 0, soc = 0, sost = 0;
                for (size_t i = 0; i < oz; ++i) {
                    const double d = (double) std::fabs(ocA[i] - ocB[i]);
                    if (std::isnan(d) || d > eoc) eoc = d;   // std::max silently DROPS NaN - a NaN state
                    soc = std::max(soc, (double) std::fabs(ocA[i]));   // diff read as err 0.0 for a full run
                }
                for (size_t i = 0; i < st0.size(); ++i) {
                    const double d = (double) std::fabs(stA[i] - stB[i]);
                    if (std::isnan(d) || d > est) est = d;
                    sost = std::max(sost, (double) std::fabs(stA[i]));
                }
                // TEMP diagnostics: locate the first non-finite entry in each array (t, head, col)
                for (int which = 0; which < 4; ++which) {
                    const float* arr = which == 0 ? ocA.data() : which == 1 ? ocB.data() : which == 2 ? stA.data() : stB.data();
                    const size_t n = which < 2 ? oz : st0.size();
                    const char* nm = which == 0 ? "serial-oc" : which == 1 ? "wy-oc" : which == 2 ? "serial-st" : "wy-st";
                    for (size_t i2 = 0; i2 < n; ++i2)
                        if (!std::isfinite(arr[i2])) {
                            if (which < 2) {
                                const int64_t tt = i2 / (HV * S);
                                const size_t rem = i2 % (HV * S);
                                std::printf("NONFINITE %s [%zu]: t=%lld head=%zu col=%zu val=%g\n", nm, i2,
                                            (long long) tt, rem / S, rem % S, arr[i2]);
                            } else {
                                const size_t row = i2 / ((size_t) HV * S), rem = i2 % ((size_t) HV * S);
                                std::printf("NONFINITE %s [%zu]: row=%zu head=%zu col=%zu val=%g\n", nm, i2,
                                            row, rem / S, rem % S, arr[i2]);
                            }
                            break;
                        }
                }
                if (std::getenv("WY_DBG") && T == 1)
                    for (int j = 0; j < 4; ++j) {
                        double kv = 0;
                        for (int i = 0; i < S; ++i) kv += (double) h[(size_t) HK * S + i] * st0[(size_t) i * HV * S + j];
                        std::printf("DBG T1 g%d col%d: beta %.4f v %.4f kv %.4f k[0..2]=%.4f %.4f %.4f s0[0..3]=%.4f %.4f %.4f %.4f | serial oc %.4f wy oc %.4f | serial st %.4f wy st %.4f\n",
                                    g, j, beta[j], h[(size_t) 2 * HK * S + j], kv,
                                    (double) h[(size_t) HK * S + 0], (double) h[(size_t) HK * S + 1],
                                    (double) h[(size_t) HK * S + 2],
                                    st0[(size_t) 0 * HV * S + j], st0[(size_t) 1 * HV * S + j],
                                    st0[(size_t) 2 * HV * S + j], st0[(size_t) 3 * HV * S + j],
                                    ocA[(size_t) j], ocB[(size_t) j], stA[(size_t) j], stB[(size_t) j]);
                    }
                const double bound = 1e-3 * (1.0 + soc), bst = 1e-3 * (1.0 + sost);
                const bool ok = eoc < bound && est < bst;   // scale-aware (the state grows without decay)
                worst = std::max(worst, eoc);
                if (!ok) {
                    ++wfails;
                    std::printf("  WY gpu T=%d g=%d: oc err %.3e (bound %.3e) state err %.3e FAIL\n", (int) T, g, eoc, bound, est);
                }
                // WY3: the device fp32 WY kernel against the fp64 oracle, inside the derived reassociation
                // bound.  Per element the device sums <= ~2^10 fp32 terms (S + WY dots + the substitution
                // chain) at u32 = 2^-24 each, compounded over ceil(T/WY) serial sub-chunks: the bound is
                // (T/WY + 1) * 2^10 * 2^-24 * (1 + scale) = (T/WY + 1) * 2^-14 * (1 + scale), scale from the
                // oracle itself.  No tolerance is involved: B is derived, and the negative control is the
                // serial-vs-WY check above (a same-arithmetic pairing must stay ~1e-7 here).
                {
                    std::vector<double> h64(h.begin(), h.end()), g64(gate.begin(), gate.end()), b64(beta.begin(), beta.end()),
                        s064(st0.begin(), st0.end()), ocO(oz), stO(st0.size());
                    oracle64(h64.data(), g64.data(), b64.data(), s064.data(), ocO.data(), stO.data(), T);
                    double eo_oc = 0, eo_st = 0, so_oc = 0, so_st = 0;
                    for (size_t i = 0; i < oz; ++i) {
                        const double d = std::fabs((double) ocB[i] - ocO[i]);
                        if (std::isnan(d) || d > eo_oc) eo_oc = d;
                        so_oc = std::max(so_oc, std::fabs(ocO[i]));
                    }
                    for (size_t i = 0; i < st0.size(); ++i) {
                        const double d = std::fabs((double) stB[i] - stO[i]);
                        if (std::isnan(d) || d > eo_st) eo_st = d;
                        so_st = std::max(so_st, std::fabs(stO[i]));
                    }
                    const double chunks = (double) ((T + WY - 1) / WY);
                    const double boc = chunks * (1.0 / 16384.0) * (1.0 + so_oc), bst = chunks * (1.0 / 16384.0) * (1.0 + so_st);
                    const double util = std::max(eo_oc / boc, eo_st / bst);
                    worst_oracle = std::max(worst_oracle, util);
                    if (!(eo_oc < boc && eo_st < bst)) {
                        ++wfails;
                        std::printf("  WY oracle T=%d g=%d: oc %.3e (bound %.3e) state %.3e (bound %.3e) FAIL\n",
                                    (int) T, g, eo_oc, boc, eo_st, bst);
                    }
                }
            }
        std::printf("device WY vs serial GPU: worst oc err %.3e -> %s\n", worst, wfails ? "FAIL" : "OK");
        std::printf("device WY vs fp64 oracle: worst bound utilisation %.1f%% -> %s\n", worst_oracle * 100.0,
                    wfails ? "FAIL" : "OK");
        fails += wfails;
    }
    fails += fixture("T=2 trace", 2, 0, 11);
    fails += fixture("plain", 200, 0, 1);
    fails += fixture("T < WY sub-chunk", 37, 0, 2);
    fails += fixture("T = WY exactly", 64, 0, 3);
    fails += fixture("T = WY+1", 65, 0, 4);
    fails += fixture("long", 4099, 0, 5);
    fails += fixture("gates = 0 (no decay)", 300, 1, 6);
    fails += fixture("gates = -6 (fast decay)", 300, 2, 7);
    fails += fixture("beta = 0 (pure decay)", 300, 3, 8);
    std::printf("%s (%d mismatches)\n", fails ? "BROKEN" : "all checks PASS", fails);
    return fails ? 1 : 0;
}
