// src/prefill/kernels.cu - see include/strata/prefill/kernels.hpp.
#include "strata/prefill/kernels.hpp"
#include "strata/kernels/mrope.hpp"
#include "strata/kernels/verify_kernels.hpp"

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cfloat>
#include <cmath>
#include <cstdio>
#include <cstdlib>

namespace strata::prefill {
namespace {

constexpr int N = 2560, HC = 4, D = N * HC, LR = 320;
constexpr int S = 128, HK = 16, HV = 48, C = 10240;

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
__device__ __forceinline__ uint16_t bf(float f) {
    uint32_t u = __float_as_uint(f);
    u += 0x7fffu + ((u >> 16) & 1u);
    return (uint16_t) (u >> 16);
}
__device__ __forceinline__ float sigm(float x) { return 1.0f / (1.0f + __expf(-x)); }
__device__ __forceinline__ uint16_t hf(float f) { return __half_as_ushort(__float2half_rn(f)); }
// block-wide sum for blockDim.x <= 1024, result broadcast
__device__ float block_sum(float v, float* sh) {
    const int lane = threadIdx.x & 31, w = threadIdx.x >> 5;
    v = warp_sum(v);
    __syncthreads();
    if (lane == 0) sh[w] = v;
    __syncthreads();
    const int nw = (blockDim.x + 31) >> 5;
    float t = (threadIdx.x < nw) ? sh[threadIdx.x] : 0.0f;
    if (w == 0) t = warp_sum(t);
    if (threadIdx.x == 0) sh[0] = t;
    __syncthreads();
    return sh[0];
}
void check(const char* what) {
    const cudaError_t e = cudaGetLastError();
    if (e != cudaSuccess) { std::fprintf(stderr, "prefill %s: %s\n", what, cudaGetErrorString(e)); std::exit(1); }
}
unsigned blocks_for(int64_t n, int t = 256) { return (unsigned) ((n + t - 1) / t); }

// ---------------------------------------------------------------- hyper-connection
__global__ void gr_norm_kernel(const float* __restrict__ R, const float* __restrict__ w, float eps,
                               float* __restrict__ xn, uint16_t* __restrict__ xn16) {
    __shared__ float sh[32];
    const int64_t row = blockIdx.x;                 // t * 4 + c
    const int c = (int) (row % HC);
    const float* r = R + row * N;
    float ss = 0.0f;
    for (int d = threadIdx.x; d < N; d += blockDim.x) ss += r[d] * r[d];
    const float rs = rsqrtf(block_sum(ss, sh) / (float) N + eps);
    for (int d = threadIdx.x; d < N; d += blockDim.x) {
        const float v = r[d] * rs * w[c * N + d];
        xn[row * N + d] = v;
        xn16[row * N + d] = bf(v);
    }
}
// gr_norm_kernel's row scale and BF16 image; gr_mix_r_kernel recomputes r * rs * w in the same order, so the FP32
// copy of the normalized rows is neither written nor read
__global__ void gr_norm_rs_kernel(const float* __restrict__ R, const float* __restrict__ w, float eps,
                                  float* __restrict__ rs_out, uint16_t* __restrict__ xn16) {
    __shared__ float sh[32];
    const int64_t row = blockIdx.x;                 // t * 4 + c
    const int c = (int) (row % HC);
    const float* r = R + row * N;
    float ss = 0.0f;
    for (int d = threadIdx.x; d < N; d += blockDim.x) ss += r[d] * r[d];
    const float rs = rsqrtf(block_sum(ss, sh) / (float) N + eps);
    if (threadIdx.x == 0) rs_out[row] = rs;
    for (int d = threadIdx.x; d < N; d += blockDim.x) xn16[row * N + d] = bf(r[d] * rs * w[c * N + d]);
}
__global__ void gr_mix_r_kernel(const float* __restrict__ R, const float* __restrict__ rs, const float* __restrict__ w,
                                const float* __restrict__ g, float* __restrict__ mixed, uint16_t* __restrict__ mixed16,
                                int64_t T, uint16_t* __restrict__ mixed_h) {
    const int64_t i = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= T * N) return;
    const int64_t t = i / N, d = i % N;
    float s = 0.0f;
#pragma unroll
    for (int c = 0; c < HC; ++c) {
        const int64_t j = t * D + c * N + d;
        const float x = R[j] * rs[t * HC + c] * w[c * N + d];   // gr_norm_kernel's value, bit for bit
        s = fmaf(x, sigm(g[j]), s);
    }
    s /= (float) HC;
    if (mixed) mixed[i] = s;
    if (mixed16) mixed16[i] = bf(s);
    if (mixed_h) mixed_h[i] = hf(s);
}
// gr_write_kernel for one row (t, c), then gr_norm_rs_kernel's reduction over it with the next read's norm weights:
// the same thread-to-element mapping (256 threads, stride 256) and block_sum, so rs and the BF16 image are the same
// bits, and R is not read back
constexpr int GRW_PER = N / 256;
__global__ void __launch_bounds__(256)
gr_write_norm_rs_kernel(float* __restrict__ R, const float* __restrict__ bo, const float* __restrict__ inj,
                        int64_t inj_ld, const float* __restrict__ w, float eps, float* __restrict__ rs_out,
                        uint16_t* __restrict__ xn16, const uint16_t* __restrict__ partial) {
    __shared__ float sh[32];
    const int64_t row = blockIdx.x;                 // t * 4 + c
    const int64_t t = row / HC;
    const int c = (int) (row % HC);
    float* r = R + row * N;
    const float sc = 2.0f * sigm(inj[t * inj_ld + c] / (float) HC);
    float v[GRW_PER];
    float ss = 0.0f;
#pragma unroll
    for (int k = 0; k < GRW_PER; ++k) {
        const int d = threadIdx.x + 256 * k;
        const float b = partial ? bo[t * N + d] + __half2float(__ushort_as_half(partial[t * N + d])) : bo[t * N + d];
        const float x = fmaf(b, sc, r[d]);
        r[d] = x;
        v[k] = x;
        ss += x * x;
    }
    const float rs = rsqrtf(block_sum(ss, sh) / (float) N + eps);
    if (threadIdx.x == 0) rs_out[row] = rs;
#pragma unroll
    for (int k = 0; k < GRW_PER; ++k) {
        const int d = threadIdx.x + 256 * k;
        xn16[row * N + d] = bf(v[k] * rs * w[c * N + d]);
    }
}
__global__ void gr_silu_kernel(const float* __restrict__ lo, uint16_t* __restrict__ lo16, int64_t n) {
    const int64_t i = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const float x = lo[i] / (float) HC;
    lo16[i] = bf(x / (1.0f + __expf(-x)));
}
__global__ void gr_mix_kernel(const float* __restrict__ xn, const float* __restrict__ g, float* __restrict__ mixed,
                              uint16_t* __restrict__ mixed16, int64_t T, uint16_t* __restrict__ mixed_h) {
    const int64_t i = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= T * N) return;
    const int64_t t = i / N, d = i % N;
    float s = 0.0f;
#pragma unroll
    for (int c = 0; c < HC; ++c) {
        const int64_t j = t * D + c * N + d;
        s = fmaf(xn[j], sigm(g[j]), s);
    }
    s /= (float) HC;
    if (mixed) mixed[i] = s;
    if (mixed16) mixed16[i] = bf(s);
    if (mixed_h) mixed_h[i] = hf(s);
}
__global__ void gr_write_kernel(float* __restrict__ R, const float* __restrict__ bo, const float* __restrict__ inj,
                                int64_t inj_ld, int64_t T, const uint16_t* __restrict__ partial) {
    const int64_t i = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= T * D) return;
    const int64_t t = i / D, c = (i % D) / N, d = i % N;
    const float b = partial ? bo[t * N + d] + __half2float(__ushort_as_half(partial[t * N + d])) : bo[t * N + d];
    R[i] = fmaf(b, 2.0f * sigm(inj[t * inj_ld + c] / (float) HC), R[i]);
}
__global__ void gr_broadcast_kernel(const float* __restrict__ e, float* __restrict__ R, int64_t T) {
    const int64_t i = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= T * D) return;
    const int64_t t = i / D, d = i % N;
    R[i] = e[t * N + d];
}

// ---------------------------------------------------------------- GDN
__global__ void gdn_gates_kernel(const float* __restrict__ ab, const float* __restrict__ dt,
                                 const float* __restrict__ ssm_a, float* __restrict__ gate, float* __restrict__ beta,
                                 int64_t T) {
    const int64_t i = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= T * HV) return;
    const int64_t t = i / HV, h = i % HV;
    const float v = ab[t * 2 * HV + h] + dt[h];
    gate[i] = (v > 20.0f ? v : log1pf(__expf(v))) * ssm_a[h];
    beta[i] = sigm(ab[t * 2 * HV + HV + h]);
}
// The conv history after a chunk: its last three inputs, older ones moved up when the chunk is shorter.
__global__ void gdn_conv_hist_kernel(float* __restrict__ hist, const float* __restrict__ qkv, int64_t T) {
    const int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= C) return;
    const float old[3] = {hist[c * 3], hist[c * 3 + 1], hist[c * 3 + 2]};
#pragma unroll
    for (int r = 0; r < 3; ++r) {
        const int64_t j = T - 3 + r;
        hist[c * 3 + r] = j >= 0 ? qkv[j * C + c] : old[r + T];
    }
}
// cp.async of one 4-byte word into shared memory (sm_80+), and its group bookkeeping.
__device__ __forceinline__ void cp_async4(float* dst, const float* src) {
    const unsigned d = (unsigned) __cvta_generic_to_shared(dst);
    asm volatile("cp.async.ca.shared.global [%0], [%1], 4;\n" ::"r"(d), "l"(src));
}
__device__ __forceinline__ void cp_async_commit() { asm volatile("cp.async.commit_group;\n" ::); }
template <int N> __device__ __forceinline__ void cp_async_wait() { asm volatile("cp.async.wait_group %0;\n" ::"n"(N)); }

// The recurrence over the chunk, the arithmetic of gdn_step_norm_multi: a thread per (CPT columns, group of 32
// state rows), a head's columns split over S / CB blocks.  A column's four row groups sit in adjacent lanes, so
// both sums over the row groups are shuffles added in the decode kernels' order (group 0 + 1 + 2 + 3).  Each
// token's q, k, v, gate and beta are copied (cp.async) into a ring of DEPTH token slots DEPTH - 1 tokens ahead, so
// the reads from memory overlap the steps before them: one barrier per token.  Each thread reads its row group's
// 32 q and 32 k values into registers with 16-byte loads once for its CPT columns; a row group is padded to 36
// words, so the four groups a warp reads at once sit in different banks.  Writes o (before the norm) to `out`;
// gdn_norm_kernel finishes each token.
constexpr int RG = 4, RPG = S / RG, RP = RPG + 4, DEPTH = 8, SLOT = 2 * RG * RP;
template <int CB, int CPT>
__global__ void __launch_bounds__(CB / CPT * RG) gdn_scan_kernel(float* __restrict__ state,
                                                                 const float* __restrict__ h,
                                                                 const float* __restrict__ gate,
                                                                 const float* __restrict__ beta,
                                                                 float* __restrict__ out, int T) {
    constexpr int NT = CB / CPT * RG, QL = (2 * S + NT - 1) / NT;   // threads; q/k values each thread copies
    static_assert(CB % (8 * CPT) == 0 && NT >= CB + 2, "8 column groups a warp; a thread each for v, gate, beta");
    __shared__ __align__(16) float sqk[DEPTH * SLOT];   // per slot: q, then k, [row group][row]
    __shared__ float sv[DEPTH * CB], sgb[DEPTH * 2];
    const int head = blockIdx.x, tid = threadIdx.x, lane = tid & 31;
    const int cl = ((tid >> 5) * 8 + (lane >> 2)) * CPT, rg = lane & 3, g0 = lane & ~3;   // first local column
    const int qh = head % HK;
    // what this thread copies for every token: q/k value i = tid + j * NT < 2S (row i & 127 of q or k), then v of
    // local column tid (tid < CB), the gate (tid == CB) or beta (tid == CB + 1)
    int qk_off[QL], qk_at[QL];
#pragma unroll
    for (int j = 0; j < QL; ++j) {
        const int i = tid + j * NT;
        qk_off[j] = i < 2 * S ? (i < S ? 0 : HK * S) + qh * S + (i & (S - 1)) : -1;
        qk_at[j] = (i >> 7) * RG * RP + ((i & (S - 1)) / RPG) * RP + (i & (RPG - 1));
    }
    const int v_off = 2 * HK * S + head * S + blockIdx.y * CB + tid;
    const float* gb = (tid == CB ? gate : beta) + head;
    float s[CPT][RPG];
    float* base = state + ((size_t) (rg * RPG) * HV + head) * S + blockIdx.y * CB + cl;
    const size_t rs = (size_t) HV * S;
#pragma unroll
    for (int c = 0; c < CPT; ++c)
#pragma unroll
        for (int r = 0; r < RPG; ++r) s[c][r] = base[r * rs + c];
    int next = 0;   // the next token to copy
    auto issue = [&]() {
        if (next < T) {
            const int slot = next & (DEPTH - 1);
            const float* ht = h + (size_t) next * C;
#pragma unroll
            for (int j = 0; j < QL; ++j)
                if (qk_off[j] >= 0) cp_async4(&sqk[slot * SLOT + qk_at[j]], ht + qk_off[j]);
            if (tid < CB) cp_async4(&sv[slot * CB + tid], ht + v_off);
            else if (tid < CB + 2) cp_async4(&sgb[slot * 2 + tid - CB], gb + (size_t) next * HV);
        }
        cp_async_commit();
        ++next;
    };
    for (int t = 0; t < DEPTH - 1; ++t) issue();
    for (int t = 0; t < T; ++t) {
        cp_async_wait<DEPTH - 2>();
        __syncthreads();   // token t's slot is filled; every thread is done with token t - 1's slot
        issue();
        const int slot = t & (DEPTH - 1);
        float qr[RPG], kr[RPG];
        const float4* q4 = reinterpret_cast<const float4*>(&sqk[slot * SLOT + rg * RP]);
        const float4* k4 = reinterpret_cast<const float4*>(&sqk[slot * SLOT + RG * RP + rg * RP]);
#pragma unroll
        for (int j = 0; j < RPG / 4; ++j) {
            const float4 a = k4[j], b = q4[j];
            kr[4 * j] = a.x; kr[4 * j + 1] = a.y; kr[4 * j + 2] = a.z; kr[4 * j + 3] = a.w;
            qr[4 * j] = b.x; qr[4 * j + 1] = b.y; qr[4 * j + 2] = b.z; qr[4 * j + 3] = b.w;
        }
        const float g = __expf(sgb[slot * 2]), bt = sgb[slot * 2 + 1];
        float kv[CPT], o[CPT];
#pragma unroll
        for (int c = 0; c < CPT; ++c) kv[c] = o[c] = 0.0f;
#pragma unroll
        for (int r = 0; r < RPG; ++r)
#pragma unroll
            for (int c = 0; c < CPT; ++c) kv[c] = fmaf(s[c][r], kr[r], kv[c]);
        float delta[CPT];
#pragma unroll
        for (int c = 0; c < CPT; ++c) {
            const float k0 = __shfl_sync(0xffffffffu, kv[c], g0), k1 = __shfl_sync(0xffffffffu, kv[c], g0 + 1),
                        k2 = __shfl_sync(0xffffffffu, kv[c], g0 + 2), k3 = __shfl_sync(0xffffffffu, kv[c], g0 + 3);
            const float kv_col = k0 + k1 + k2 + k3;
            delta[c] = (sv[slot * CB + cl + c] - g * kv_col) * bt;
        }
#pragma unroll
        for (int r = 0; r < RPG; ++r)
#pragma unroll
            for (int c = 0; c < CPT; ++c) {
                s[c][r] = fmaf(g, s[c][r], kr[r] * delta[c]);
                o[c] = fmaf(s[c][r], qr[r], o[c]);
            }
        float* ot = out + (size_t) t * HV * S + head * S + blockIdx.y * CB + cl;
#pragma unroll
        for (int c = 0; c < CPT; ++c) {
            const float o0 = __shfl_sync(0xffffffffu, o[c], g0), o1 = __shfl_sync(0xffffffffu, o[c], g0 + 1),
                        o2 = __shfl_sync(0xffffffffu, o[c], g0 + 2), o3 = __shfl_sync(0xffffffffu, o[c], g0 + 3);
            if (rg == 0) ot[c] = (o0 + o1 + o2 + o3) * rsqrtf((float) S);
        }
    }
    cp_async_wait<0>();
#pragma unroll
    for (int c = 0; c < CPT; ++c)
#pragma unroll
        for (int r = 0; r < RPG; ++r) base[r * rs + c] = s[c][r];
}
// y = rmsnorm(o) * gamma * sigmoid(z) per (token, head), in place over o; also the FP16 image
__global__ void __launch_bounds__(S) gdn_norm_kernel(float* __restrict__ y, const float* __restrict__ z,
                                                     const float* __restrict__ gamma, float eps,
                                                     uint16_t* __restrict__ y16) {
    __shared__ float wsum[S / 32];
    const size_t i = ((size_t) blockIdx.y * HV + blockIdx.x) * S + threadIdx.x;
    const float oc = y[i];
    const float sp = warp_sum(oc * oc);
    if ((threadIdx.x & 31) == 0) wsum[threadIdx.x >> 5] = sp;
    __syncthreads();
    const float ss = wsum[0] + wsum[1] + wsum[2] + wsum[3];
    const float v = oc * rsqrtf(ss / (float) S + eps) * gamma[threadIdx.x] * sigm(z[i]);
    y[i] = v;
    y16[i] = hf(v);
}

// ---------------------------------------------------------------- MoE
__global__ void route_kernel(const float* __restrict__ logits, int32_t* __restrict__ ids, float* __restrict__ wout,
                             int64_t T) {
    const int64_t t = (int64_t) blockIdx.x * (blockDim.x >> 5) + (threadIdx.x >> 5);
    if (t >= T) return;
    const int lane = threadIdx.x & 31;
    const float* lg = logits + t * 512;
    float v[16];
#pragma unroll
    for (int i = 0; i < 16; ++i) v[i] = lg[lane + i * 32];
    float mx = -INFINITY;
#pragma unroll
    for (int i = 0; i < 16; ++i) mx = fmaxf(mx, v[i]);
    mx = warp_max(mx);
    float sum = 0.0f;
#pragma unroll
    for (int i = 0; i < 16; ++i) { v[i] = expf(v[i] - mx); sum += v[i]; }
    const float rcp = 1.0f / warp_sum(sum);
#pragma unroll
    for (int i = 0; i < 16; ++i) { v[i] *= rcp; if (isnan(v[i])) v[i] = -FLT_MAX; }
    float selected = 0.0f, selected_sum = 0.0f;
    for (int rank = 0; rank < 10; ++rank) {
        float best = v[0];
        int ex = lane;
#pragma unroll
        for (int i = 1; i < 16; ++i) if (v[i] > best) { best = v[i]; ex = lane + i * 32; }
#pragma unroll
        for (int m = 16; m; m >>= 1) {
            const float ob = __shfl_xor_sync(0xffffffffu, best, m);
            const int oi = __shfl_xor_sync(0xffffffffu, ex, m);
            if (ob > best || (ob == best && oi < ex)) { best = ob; ex = oi; }
        }
        if ((ex & 31) == lane) { v[ex / 32] = -INFINITY; selected_sum += best; }
        if (lane == 0) ids[t * 10 + rank] = ex;
        if (rank == lane) selected = best;
    }
    selected_sum = fmaxf(warp_sum(selected_sum), 6.103515625e-5f);
    if (lane < 10) wout[t * 10 + lane] = selected / selected_sum;
}
// Strata blob: gate/up codes [1280][640 B], down codes [2560][160 B], gate/up scales [1280][40] f16, down scales [2560][10] f16
template <bool HALF>
__global__ void blob_dequant_kernel(const uint8_t* __restrict__ blob, uint16_t* __restrict__ gu16,
                                    uint16_t* __restrict__ d16) {
    constexpr size_t O_D_CODES = (size_t) 1280 * 640, O_GU_SC = O_D_CODES + (size_t) 2560 * 160,
                     O_D_SC = O_GU_SC + (size_t) 1280 * 40 * 2;
    const int64_t i = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;   // one thread per 4 weights (one code byte)
    const int64_t n_gu = 1280LL * 640, n_d = 2560LL * 160;
    if (i < n_gu) {
        const int64_t row = i / 640, byte = i % 640;
        const uint8_t c = blob[row * 640 + byte];
        const uint8_t* sp = blob + O_GU_SC + (size_t) (row * 40 + (byte * 4) / 64) * 2;
        const float d = __half2float(__ushort_as_half((uint16_t) (sp[0] | (sp[1] << 8))));
        uint16_t* o = gu16 + row * 2560 + byte * 4;
#pragma unroll
        for (int k = 0; k < 4; ++k) { const float v = (float) (((c >> (2 * k)) & 3) - 1) * d; o[k] = HALF ? hf(v) : bf(v); }
    } else if (i < n_gu + n_d) {
        const int64_t j = i - n_gu, row = j / 160, byte = j % 160;
        const uint8_t c = blob[O_D_CODES + row * 160 + byte];
        const uint8_t* sp = blob + O_D_SC + (size_t) (row * 10 + (byte * 4) / 64) * 2;
        const float d = __half2float(__ushort_as_half((uint16_t) (sp[0] | (sp[1] << 8))));
        uint16_t* o = d16 + row * 640 + byte * 4;
#pragma unroll
        for (int k = 0; k < 4; ++k) { const float v = (float) (((c >> (2 * k)) & 3) - 1) * d; o[k] = HALF ? hf(v) : bf(v); }
    }
}
__global__ void swiglu_il_kernel(const float* __restrict__ gu, uint16_t* __restrict__ h16, int64_t n) {
    const int64_t i = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n * 640) return;
    const int64_t r = i / 640, k = i % 640;
    const float g = gu[r * 1280 + 2 * k], u = gu[r * 1280 + 2 * k + 1];
    h16[i] = hf(g / (1.0f + __expf(-g)) * u);
}
__global__ void swiglu_pair_kernel(const float* __restrict__ g, const float* __restrict__ u, uint16_t* __restrict__ h16,
                                   int64_t n) {
    const int64_t i = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n * 640) return;
    const float a = g[i];
    h16[i] = hf(a / (1.0f + __expf(-a)) * u[i]);
}
__global__ void gather_rows16_kernel(const uint16_t* __restrict__ x, const int32_t* __restrict__ src,
                                     uint16_t* __restrict__ dst, int64_t n, int64_t width) {
    const int64_t i = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;   // one uint4 (8 bf16)
    const int64_t per = width / 8;
    if (i >= n * per) return;
    const int64_t r = i / per, j = i % per;
    reinterpret_cast<uint4*>(dst)[r * per + j] = reinterpret_cast<const uint4*>(x)[(int64_t) src[r] * per + j];
}
__global__ void gather_rows16_f32_kernel(const uint16_t* __restrict__ x, const int32_t* __restrict__ src,
                                         float* __restrict__ dst, int64_t n, int64_t width) {
    const int64_t i = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;   // 8 values
    const int64_t per = width / 8;
    if (i >= n * per) return;
    const int64_t r = i / per, j = i % per;
    const uint4 v = reinterpret_cast<const uint4*>(x)[(int64_t) src[r] * per + j];
    const __half2* h = reinterpret_cast<const __half2*>(&v);
    float4* o = reinterpret_cast<float4*>(dst + r * width + j * 8);
    const float2 a = __half22float2(h[0]), b = __half22float2(h[1]), c = __half22float2(h[2]), d = __half22float2(h[3]);
    o[0] = make_float4(a.x, a.y, b.x, b.y);
    o[1] = make_float4(c.x, c.y, d.x, d.y);
}
__global__ void moe_shared_gate_kernel(float* __restrict__ x, const float* __restrict__ sg, int64_t T) {
    const int64_t i = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= T * N) return;
    x[i] = x[i] * sigm(sg[i / N]);
}
__global__ void moe_scatter_add_kernel(float* __restrict__ sum, const float* __restrict__ rows,
                                       const float* __restrict__ w, const int32_t* __restrict__ src, int64_t n) {
    const int64_t i = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n * N) return;
    const int64_t r = i / N, d = i % N;
    float* o = sum + (int64_t) src[r] * N + d;
    *o = fmaf(w[r], rows[i], *o);
}

// ---------------------------------------------------------------- QSA helpers
__global__ void rms_rows_kernel(float* __restrict__ x, const float* __restrict__ w, int64_t cols, int64_t ld, float eps) {
    __shared__ float sh[32];
    float* r = x + (int64_t) blockIdx.x * ld;
    float ss = 0.0f;
    for (int64_t c = threadIdx.x; c < cols; c += blockDim.x) ss += r[c] * r[c];
    const float s = rsqrtf(block_sum(ss, sh) / (float) cols + eps);
    __syncthreads();
    for (int64_t c = threadIdx.x; c < cols; c += blockDim.x) r[c] = s * r[c] * w[c];
}
__global__ void rope_kernel(float* __restrict__ x, int64_t heads, int64_t dim, int64_t ld, int64_t pos0,
                            float theta_scale, const int32_t* __restrict__ mtab) {
    const int64_t row = blockIdx.x;             // t * heads + h
    const int pair = threadIdx.x;               // 0..31
    const int64_t t = row / heads, h = row % heads;
    float* p = x + t * ld + h * dim;
    const float theta = (float) strata::kernels::mrope_pos(mtab, (int) (pos0 + t), pair) * powf(theta_scale, (float) pair);
    const float c = cosf(theta), s = sinf(theta);
    const float a = p[pair], b = p[pair + 32];
    p[pair] = a * c - b * s;
    p[pair + 32] = a * s + b * c;
}
__global__ void split_q_kernel(const float* __restrict__ qf, float* __restrict__ q, int64_t T) {
    const int64_t i = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= T * 24 * 256) return;
    const int64_t t = i / (24 * 256), h = (i / 256) % 24, d = i % 256;
    q[i] = qf[t * 24 * 512 + h * 512 + d];
}
__global__ void gate_attn_kernel(const float* __restrict__ a, const float* __restrict__ qf, uint16_t* __restrict__ o16,
                                 int64_t T) {
    const int64_t i = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= T * 24 * 256) return;
    const int64_t t = i / (24 * 256), h = (i / 256) % 24, d = i % 256;
    o16[i] = hf(a[i] * (1.0f / (1.0f + expf(-qf[t * 24 * 512 + h * 512 + 256 + d]))));
}

// one block per (token, kv head, 64-value group); 64 threads
__global__ void kv_append_kernel(const float* __restrict__ K, const float* __restrict__ V, int64_t pos0,
                                 const int32_t* __restrict__ table, int64_t page_size, uint16_t* k_pool,
                                 uint16_t* v_pool, int8_t* k_q, int8_t* v_q, uint16_t* k_scale, uint16_t* v_scale) {
    const int64_t t = blockIdx.x;
    const int kvh = blockIdx.y, g = blockIdx.z >> 1;
    const bool is_v = (blockIdx.z & 1) != 0;
    const int d = g * 64 + threadIdx.x;
    const float x = (is_v ? V : K)[t * 512 + kvh * 256 + d];
    const int64_t pos = pos0 + t;
    const int64_t page = table[pos / page_size];
    const int64_t row = (page * 2 + kvh) * page_size + pos % page_size;
    if (k_pool != nullptr) {
        (is_v ? v_pool : k_pool)[row * 256 + d] = hf(x);
        return;
    }
    float a = fabsf(x);
    for (int o = 16; o > 0; o >>= 1) a = fmaxf(a, __shfl_xor_sync(0xffffffffu, a, o));
    __shared__ float wm[2];
    if ((threadIdx.x & 31) == 0) wm[threadIdx.x >> 5] = a;
    __syncthreads();
    const float amax = fmaxf(wm[0], wm[1]);
    const uint16_t sb = hf(amax / 127.0f);
    const float sf = __half2float(__ushort_as_half(sb));
    int q = 0;
    if (sf > 0.0f) { q = __float2int_rn(x / sf); q = q < -127 ? -127 : (q > 127 ? 127 : q); }
    (is_v ? v_q : k_q)[row * 256 + d] = (int8_t) q;
    if (threadIdx.x == 0) (is_v ? v_scale : k_scale)[row * 4 + g] = sb;
}
__global__ void to_f16_kernel(const float* __restrict__ x, uint16_t* __restrict__ y, int64_t n) {
    for (int64_t i = (int64_t) blockIdx.x * blockDim.x + threadIdx.x; i < n; i += (int64_t) gridDim.x * blockDim.x)
        y[i] = hf(x[i]);
}
__global__ void to_bf16_kernel(const float* __restrict__ x, uint16_t* __restrict__ y, int64_t n) {
    for (int64_t i = (int64_t) blockIdx.x * blockDim.x + threadIdx.x; i < n; i += (int64_t) gridDim.x * blockDim.x)
        y[i] = bf(x[i]);
}

}  // namespace

void kv_append(const float* K, const float* V, int64_t T, int64_t pos0, const int32_t* page_table, int64_t page_size,
               uint16_t* k_pool, uint16_t* v_pool, int8_t* k_q, int8_t* v_q, uint16_t* k_scale, uint16_t* v_scale,
               void* stream) {
    if (T <= 0) return;
    kv_append_kernel<<<dim3((unsigned) T, 2, 8), 64, 0, (cudaStream_t) stream>>>(K, V, pos0, page_table, page_size, k_pool,
                                                                                  v_pool, k_q, v_q, k_scale, v_scale);
    check("kv_append");
}
void to_f16(const float* x, uint16_t* y, int64_t n, void* stream) {
    if (n <= 0) return;
    to_f16_kernel<<<(unsigned) ((n + 255) / 256 < 4096 ? (n + 255) / 256 : 4096), 256, 0, (cudaStream_t) stream>>>(x, y, n);
    check("to_f16");
}
void to_bf16(const float* x, uint16_t* y, int64_t n, void* stream) {
    if (n <= 0) return;
    to_bf16_kernel<<<(unsigned) ((n + 255) / 256 < 4096 ? (n + 255) / 256 : 4096), 256, 0, (cudaStream_t) stream>>>(x, y, n);
    check("to_bf16");
}

void gr_norm(const float* R, const float* w_norm, float eps, float* xn, uint16_t* xn16, int64_t T, void* stream) {
    gr_norm_kernel<<<(unsigned) (T * HC), 256, 0, (cudaStream_t) stream>>>(R, w_norm, eps, xn, xn16);
    check("gr_norm");
}
void gr_norm_rs(const float* R, const float* w_norm, float eps, float* rs, uint16_t* xn16, int64_t T, void* stream) {
    gr_norm_rs_kernel<<<(unsigned) (T * HC), 256, 0, (cudaStream_t) stream>>>(R, w_norm, eps, rs, xn16);
    check("gr_norm_rs");
}
void gr_mix_r(const float* R, const float* rs, const float* w_norm, const float* gated, float* mixed, uint16_t* mixed16,
              int64_t T, void* stream, uint16_t* mixed_h) {
    gr_mix_r_kernel<<<blocks_for(T * N), 256, 0, (cudaStream_t) stream>>>(R, rs, w_norm, gated, mixed, mixed16, T,
                                                                          mixed_h);
    check("gr_mix_r");
}
void gr_write_norm_rs(float* R, const float* bo, const float* inj, int64_t inj_ld, const float* w_norm, float eps,
                      float* rs, uint16_t* xn16, int64_t T, void* stream, const uint16_t* partial) {
    gr_write_norm_rs_kernel<<<(unsigned) (T * HC), 256, 0, (cudaStream_t) stream>>>(R, bo, inj, inj_ld, w_norm, eps,
                                                                                     rs, xn16, partial);
    check("gr_write_norm_rs");
}
void gr_silu(const float* lo, uint16_t* lo16, int64_t T, void* stream) {
    gr_silu_kernel<<<blocks_for(T * LR), 256, 0, (cudaStream_t) stream>>>(lo, lo16, T * LR);
    check("gr_silu");
}
void gr_mix(const float* xn, const float* gated, float* mixed, uint16_t* mixed16, int64_t T, void* stream,
            uint16_t* mixed_h) {
    gr_mix_kernel<<<blocks_for(T * N), 256, 0, (cudaStream_t) stream>>>(xn, gated, mixed, mixed16, T, mixed_h);
    check("gr_mix");
}
void gr_write(float* R, const float* bo, const float* inj, int64_t inj_ld, int64_t T, void* stream,
              const uint16_t* partial) {
    gr_write_kernel<<<blocks_for(T * D), 256, 0, (cudaStream_t) stream>>>(R, bo, inj, inj_ld, T, partial);
    check("gr_write");
}
void gr_broadcast(const float* e, float* R, int64_t T, void* stream) {
    gr_broadcast_kernel<<<blocks_for(T * D), 256, 0, (cudaStream_t) stream>>>(e, R, T);
    check("gr_broadcast");
}
void gdn_gates(const float* ab, const float* dt, const float* ssm_a, float* gate, float* beta, int64_t T, void* stream) {
    gdn_gates_kernel<<<blocks_for(T * HV), 256, 0, (cudaStream_t) stream>>>(ab, dt, ssm_a, gate, beta, T);
    check("gdn_gates");
}
void gdn_conv(float* history, const float* qkv, const float* conv_w, float* h, int64_t T, float eps, void* stream) {
    for (int64_t t0 = 0; t0 < T; t0 += 65535)
        strata::kernels::gdn_conv_l2_multi(history, qkv, conv_w, h, C, 2 * HK, eps, (int) std::min<int64_t>(65535, T - t0),
                                           stream, (int) t0);
    gdn_conv_hist_kernel<<<C / 256, 256, 0, (cudaStream_t) stream>>>(history, qkv, T);
    check("gdn_conv");
}
void gdn_scan(float* state, const float* h, const float* gate, const float* beta, float* y, int64_t T, void* stream) {
    // 32 columns a block, 2 a thread (fastest of 16-128 columns, 1-2 a thread: 0.67 us a token on the 3090)
    constexpr int CB = 32, CPT = 2;
    gdn_scan_kernel<CB, CPT><<<dim3(HV, S / CB), CB / CPT * RG, 0, (cudaStream_t) stream>>>(state, h, gate, beta, y, (int) T);
    check("gdn_scan");
}
void gdn_out_norm(float* y, const float* z, const float* gamma, float eps, uint16_t* y16, int64_t T, void* stream) {
    for (int64_t t0 = 0; t0 < T; t0 += 65535) {
        const int64_t n = std::min<int64_t>(65535, T - t0);
        gdn_norm_kernel<<<dim3(HV, (unsigned) n), S, 0, (cudaStream_t) stream>>>(y + t0 * HV * S, z + t0 * HV * S, gamma,
                                                                                eps, y16 + t0 * HV * S);
    }
    check("gdn_out_norm");
}
void route(const float* logits, int32_t* ids, float* weights, int64_t T, void* stream) {
    route_kernel<<<(unsigned) ((T + 7) / 8), 256, 0, (cudaStream_t) stream>>>(logits, ids, weights, T);
    check("route");
}
void blob_dequant(const uint8_t* blob, uint16_t* gu16, uint16_t* down16, void* stream) {
    blob_dequant_kernel<false><<<blocks_for(1280LL * 640 + 2560LL * 160), 256, 0, (cudaStream_t) stream>>>(blob, gu16, down16);
    check("blob_dequant");
}
void blob_dequant_f16(const uint8_t* blob, uint16_t* gu16, uint16_t* down16, void* stream) {
    blob_dequant_kernel<true><<<blocks_for(1280LL * 640 + 2560LL * 160), 256, 0, (cudaStream_t) stream>>>(blob, gu16, down16);
    check("blob_dequant_f16");
}
void swiglu_interleaved(const float* gu, uint16_t* h16, int64_t n, void* stream) {
    if (n <= 0) return;
    swiglu_il_kernel<<<blocks_for(n * 640), 256, 0, (cudaStream_t) stream>>>(gu, h16, n);
    check("swiglu_interleaved");
}
void swiglu_pair(const float* g, const float* u, uint16_t* h16, int64_t n, void* stream) {
    swiglu_pair_kernel<<<blocks_for(n * 640), 256, 0, (cudaStream_t) stream>>>(g, u, h16, n);
    check("swiglu_pair");
}
void gather_rows16(const uint16_t* x16, const int32_t* src, uint16_t* dst16, int64_t n, int64_t width, void* stream) {
    if (n <= 0) return;
    gather_rows16_kernel<<<blocks_for(n * (width / 8)), 256, 0, (cudaStream_t) stream>>>(x16, src, dst16, n, width);
    check("gather_rows16");
}
void gather_rows16_f32(const uint16_t* x16, const int32_t* src, float* dst, int64_t n, int64_t width, void* stream) {
    if (n <= 0) return;
    gather_rows16_f32_kernel<<<blocks_for(n * (width / 8)), 256, 0, (cudaStream_t) stream>>>(x16, src, dst, n, width);
    check("gather_rows16_f32");
}
__global__ void sums_to_f16_kernel(const float* __restrict__ x, uint16_t* __restrict__ y, int64_t n) {
    const int64_t i = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) y[i] = hf(fminf(fmaxf(x[i], -65504.0f), 65504.0f));
}
void sums_to_f16(const float* x, uint16_t* y, int64_t n, void* stream) {
    sums_to_f16_kernel<<<blocks_for(n), 256, 0, (cudaStream_t) stream>>>(x, y, n);
    check("sums_to_f16");
}
void moe_shared_gate(float* x, const float* sg, int64_t T, void* stream) {
    moe_shared_gate_kernel<<<blocks_for(T * N), 256, 0, (cudaStream_t) stream>>>(x, sg, T);
    check("moe_shared_gate");
}
__global__ void moe_gather_add_kernel(float* __restrict__ sum, const float* __restrict__ rows, int64_t r0,
                                      const float* __restrict__ w, const int32_t* __restrict__ tok,
                                      const int32_t* __restrict__ start, const int32_t* __restrict__ list) {
    const int64_t d = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (d >= N) return;
    const int b = blockIdx.y;
    float* o = sum + (int64_t) tok[b] * N + d;
    float s = *o;
    for (int i = start[b]; i < start[b + 1]; ++i) {
        const int r = list[i];
        s = fmaf(w[r], rows[(r - r0) * N + d], s);
    }
    *o = s;
}
void moe_gather_add(float* sum, const float* rows, int64_t r0, const float* w, const int32_t* tok, const int32_t* start,
                    const int32_t* list, int64_t n_tok, void* stream) {
    if (n_tok <= 0) return;
    moe_gather_add_kernel<<<dim3((unsigned) ((N + 255) / 256), (unsigned) n_tok), 256, 0, (cudaStream_t) stream>>>(
        sum, rows, r0, w, tok, start, list);
    check("moe_gather_add");
}
void moe_scatter_add(float* sum, const float* rows, const float* w, const int32_t* src, int64_t n, void* stream) {
    if (n <= 0) return;
    moe_scatter_add_kernel<<<blocks_for(n * N), 256, 0, (cudaStream_t) stream>>>(sum, rows, w, src, n);
    check("moe_scatter_add");
}
void rms_rows(float* x, const float* w, int64_t rows, int64_t cols, int64_t ld, float eps, void* stream) {
    if (rows <= 0) return;
    rms_rows_kernel<<<(unsigned) rows, 256, 0, (cudaStream_t) stream>>>(x, w, cols, ld, eps);
    check("rms_rows");
}
void rope(float* x, int64_t T, int64_t heads, int64_t dim, int64_t ld, int64_t pos0, float freq_base, void* stream) {
    const float theta_scale = powf(freq_base, -2.0f / 64.0f);
    rope_kernel<<<(unsigned) (T * heads), 32, 0, (cudaStream_t) stream>>>(x, heads, dim, ld, pos0, theta_scale, strata::kernels::mrope_table());
    check("rope");
}
void split_q(const float* q_full, float* q, int64_t T, void* stream) {
    split_q_kernel<<<blocks_for(T * 24 * 256), 256, 0, (cudaStream_t) stream>>>(q_full, q, T);
    check("split_q");
}
void gate_attn(const float* attn, const float* q_full, uint16_t* out16, int64_t T, void* stream) {
    gate_attn_kernel<<<blocks_for(T * 24 * 256), 256, 0, (cudaStream_t) stream>>>(attn, q_full, out16, T);
    check("gate_attn");
}

}  // namespace strata::prefill
