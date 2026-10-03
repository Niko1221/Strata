// qwen36_kernels.cu - see qwen36_kernels.cuh.  Plain, readable kernels: correctness first (Phase 1).
#include "qwen36_kernels.cuh"

#include <cuda_runtime.h>

#include <cmath>
#include <cstdio>
#include <cstdlib>

namespace q36 {
namespace {

inline cudaStream_t S(void* s) { return reinterpret_cast<cudaStream_t>(s); }

__device__ __forceinline__ float warp_sum(float v) {
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
    return v;
}

// Sum over the whole block (any blockDim.x multiple of 32, <= 1024).  Every thread gets the result.
__device__ __forceinline__ float block_sum(float v) {
    __shared__ float part[32];
    __shared__ float total;
    v = warp_sum(v);
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    if (lane == 0) part[warp] = v;
    __syncthreads();
    if (warp == 0) {
        const int nw = (blockDim.x + 31) >> 5;
        float t = lane < nw ? part[lane] : 0.f;
        t = warp_sum(t);
        if (lane == 0) total = t;
    }
    __syncthreads();
    const float r = total;
    __syncthreads();  // `part`/`total` may be reused by the next call
    return r;
}

__device__ __forceinline__ float sigmoidf_(float x) { return 1.f / (1.f + expf(-x)); }
__device__ __forceinline__ float siluf_(float x) { return x / (1.f + expf(-x)); }

__global__ void k_rmsnorm(float* y, const float* x, const float* w, int n, float eps) {
    const float* xr = x + (size_t)blockIdx.x * n;
    float* yr = y + (size_t)blockIdx.x * n;
    float ss = 0.f;
    for (int i = threadIdx.x; i < n; i += blockDim.x) ss += xr[i] * xr[i];
    ss = block_sum(ss);
    const float inv = rsqrtf(ss / (float)n + eps);
    for (int i = threadIdx.x; i < n; i += blockDim.x) yr[i] = xr[i] * inv * w[i];
}

template <int TYPE>
__device__ __forceinline__ float ld_w(const void* p, size_t i) {
    if (TYPE == 0) return reinterpret_cast<const float*>(p)[i];
    if (TYPE == 1) return __half2float(reinterpret_cast<const __half*>(p)[i]);
    return __uint_as_float(((uint32_t)reinterpret_cast<const uint16_t*>(p)[i]) << 16);  // BF16
}

template <int TYPE>
__global__ void k_gemv(const void* W, const float* x, float* y, int n_in, int n_out) {
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const int row = blockIdx.x * (blockDim.x >> 5) + warp;
    if (row >= n_out) return;
    float acc = 0.f;
    for (int i = lane; i < n_in; i += 32) acc += ld_w<TYPE>(W, (size_t)row * n_in + i) * x[i];
    acc = warp_sum(acc);
    if (lane == 0) y[row] = acc;
}

__global__ void k_add(float* y, const float* x, int n) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) y[i] += x[i];
}

__global__ void k_silu_mul(float* o, const float* g, const float* u, int n) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) o[i] = siluf_(g[i]) * u[i];
}

// One block of 256 threads; dynamic smem = n_expert floats.
__global__ void k_router(const float* logits, int n_expert, int k, int* ids, float* wts) {
    extern __shared__ float p[];
    logits += (size_t)blockIdx.x * n_expert;   // one block per row; row 0 for the decode path
    ids += (size_t)blockIdx.x * k;
    wts += (size_t)blockIdx.x * k;
    __shared__ float sv[256];
    __shared__ int si[256];
    __shared__ float chosen[64];
    const int t = threadIdx.x;
    float m = -INFINITY;
    for (int i = t; i < n_expert; i += 256) m = fmaxf(m, logits[i]);
    sv[t] = m;
    __syncthreads();
    for (int s = 128; s > 0; s >>= 1) {
        if (t < s) sv[t] = fmaxf(sv[t], sv[t + s]);
        __syncthreads();
    }
    m = sv[0];
    __syncthreads();
    float sum = 0.f;
    for (int i = t; i < n_expert; i += 256) {
        p[i] = expf(logits[i] - m);
        sum += p[i];
    }
    sum = block_sum(sum);
    for (int i = t; i < n_expert; i += 256) p[i] /= sum;
    __syncthreads();
    for (int j = 0; j < k; ++j) {
        float bv = -1.f;
        int bi = 0x7fffffff;
        for (int i = t; i < n_expert; i += 256)
            if (p[i] > bv) { bv = p[i]; bi = i; }       // strided scan keeps the lowest index per thread
        sv[t] = bv;
        si[t] = bi;
        __syncthreads();
        for (int s = 128; s > 0; s >>= 1) {
            if (t < s) {
                const float ov = sv[t + s];
                const int oi = si[t + s];
                if (ov > sv[t] || (ov == sv[t] && oi < si[t])) { sv[t] = ov; si[t] = oi; }
            }
            __syncthreads();
        }
        if (t == 0) {
            ids[j] = si[0];
            chosen[j] = sv[0];
            p[si[0]] = -1.f;
        }
        __syncthreads();
    }
    if (t == 0) {
        float tot = 0.f;
        for (int j = 0; j < k; ++j) tot += chosen[j];
        for (int j = 0; j < k; ++j) wts[j] = chosen[j] / tot;
    }
}

__global__ void k_moe_combine(float* x, const float* down, const float* wts, const float* shexp, const float* sg, int k,
                              int n) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    float acc = x[i];
    for (int j = 0; j < k; ++j) acc += wts[j] * down[(size_t)j * n + i];
    acc += sigmoidf_(sg[0]) * shexp[i];
    x[i] = acc;
}

__global__ void k_dot(float* out, const float* a, const float* b, int n) {
    float s = 0.f;
    for (int i = threadIdx.x; i < n; i += blockDim.x) s += a[i] * b[i];
    s = block_sum(s);
    if (threadIdx.x == 0) out[0] = s;
}

__global__ void k_gdn_out_norm(float* dst, const float* o, const float* z, const float* gamma, int cols, float eps) {
    const float* r = o + (size_t)blockIdx.x * cols;
    const float* zr = z + (size_t)blockIdx.x * cols;
    float* d = dst + (size_t)blockIdx.x * cols;
    float ss = 0.f;
    for (int i = threadIdx.x; i < cols; i += blockDim.x) ss += r[i] * r[i];
    ss = block_sum(ss);
    const float inv = rsqrtf(ss / (float)cols + eps);
    for (int i = threadIdx.x; i < cols; i += blockDim.x) d[i] = r[i] * inv * gamma[i] * siluf_(zr[i]);
}

// NeoX partial rotary on x[0..rot): pair (i, i+rot/2), angle = pos * theta^(-2i/rot).
__device__ __forceinline__ void rope_pair(float* x, int i, int rot, float theta, int pos) {
    const int half = rot >> 1;
    const float ang = (float)pos * powf(theta, -2.0f * (float)i / (float)rot);
    float s, c;
    sincosf(ang, &s, &c);
    const float a = x[i], b = x[i + half];
    x[i] = a * c - b * s;
    x[i + half] = b * c + a * s;
}

__global__ void k_prep_q(float* q_out, float* gate_out, const float* q2, const float* qnorm, int hd, int rot,
                         float theta, int pos, float eps) {
    const int h = blockIdx.x;
    const float* q = q2 + (size_t)h * 2 * hd;
    const float* g = q + hd;
    float* qo = q_out + (size_t)h * hd;
    float ss = 0.f;
    for (int i = threadIdx.x; i < hd; i += blockDim.x) ss += q[i] * q[i];
    ss = block_sum(ss);
    const float inv = rsqrtf(ss / (float)hd + eps);
    for (int i = threadIdx.x; i < hd; i += blockDim.x) {
        qo[i] = q[i] * inv * qnorm[i];
        gate_out[(size_t)h * hd + i] = g[i];
    }
    __syncthreads();
    for (int i = threadIdx.x; i < (rot >> 1); i += blockDim.x) rope_pair(qo, i, rot, theta, pos);
}

__global__ void k_prep_kv(__half* kc, __half* vc, const float* k, const float* v, const float* knorm, int hd, int rot,
                          float theta, int pos, int max_ctx, float eps, float* scratch) {
    const int h = blockIdx.x;
    const float* kin = k + (size_t)h * hd;
    float* ks = scratch + (size_t)h * hd;  // per-head F32 staging so rope can pair elements across threads
    float ss = 0.f;
    for (int i = threadIdx.x; i < hd; i += blockDim.x) ss += kin[i] * kin[i];
    ss = block_sum(ss);
    const float inv = rsqrtf(ss / (float)hd + eps);
    for (int i = threadIdx.x; i < hd; i += blockDim.x) ks[i] = kin[i] * inv * knorm[i];
    __syncthreads();
    for (int i = threadIdx.x; i < (rot >> 1); i += blockDim.x) rope_pair(ks, i, rot, theta, pos);
    __syncthreads();
    const size_t base = ((size_t)h * max_ctx + pos) * hd;
    for (int i = threadIdx.x; i < hd; i += blockDim.x) {
        kc[base + i] = __float2half(ks[i]);
        vc[base + i] = __float2half(v[(size_t)h * hd + i]);
    }
}

template <int DPL>
__device__ __forceinline__ void load_half(const __half* p, float* out) {
    static_assert(DPL == 4 || DPL == 8, "dims per lane");
    if (DPL == 8) {
        const uint4 raw = *reinterpret_cast<const uint4*>(p);
        const __half2* h2 = reinterpret_cast<const __half2*>(&raw);
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            const float2 f = __half22float2(h2[j]);
            out[2 * j] = f.x;
            out[2 * j + 1] = f.y;
        }
    } else {
        const uint2 raw = *reinterpret_cast<const uint2*>(p);
        const __half2* h2 = reinterpret_cast<const __half2*>(&raw);
#pragma unroll
        for (int j = 0; j < 2; ++j) {
            const float2 f = __half22float2(h2[j]);
            out[2 * j] = f.x;
            out[2 * j + 1] = f.y;
        }
    }
}

// grid = n_head, block = 256 (8 warps). Each warp walks positions t = warp, warp+8, ... with an online softmax;
// the 8 partial (m, l, o) are merged at the end.
template <int HD>
__global__ void k_attn(float* out, const float* q, const float* gate, const __half* kc, const __half* vc, int n_head,
                       int n_kv, int n_ctx, int max_ctx) {
    constexpr int DPL = HD / 32;
    __shared__ float sm[8], sl[8];
    __shared__ float so[8][HD];
    const int h = blockIdx.x, kvh = h / (n_head / n_kv);
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const float scale = rsqrtf((float)HD);
    float qv[DPL], o[DPL];
#pragma unroll
    for (int j = 0; j < DPL; ++j) {
        qv[j] = q[(size_t)h * HD + lane * DPL + j] * scale;
        o[j] = 0.f;
    }
    float m = -INFINITY, l = 0.f;
    const __half* kb = kc + (size_t)kvh * max_ctx * HD;
    const __half* vb = vc + (size_t)kvh * max_ctx * HD;
    for (int t = warp; t < n_ctx; t += 8) {
        float kf[DPL], vf[DPL];
        load_half<DPL>(kb + (size_t)t * HD + lane * DPL, kf);
        float d = 0.f;
#pragma unroll
        for (int j = 0; j < DPL; ++j) d += qv[j] * kf[j];
        d = warp_sum(d);
        const float mn = fmaxf(m, d);
        const float a = expf(m - mn), p = expf(d - mn);
        l = l * a + p;
        load_half<DPL>(vb + (size_t)t * HD + lane * DPL, vf);
#pragma unroll
        for (int j = 0; j < DPL; ++j) o[j] = o[j] * a + p * vf[j];
        m = mn;
    }
#pragma unroll
    for (int j = 0; j < DPL; ++j) so[warp][lane * DPL + j] = o[j];
    if (lane == 0) { sm[warp] = m; sl[warp] = l; }
    __syncthreads();
    const int d = threadIdx.x;
    if (d < HD) {
        float M = -INFINITY;
        for (int w = 0; w < 8; ++w) M = fmaxf(M, sm[w]);
        float L = 0.f, O = 0.f;
        for (int w = 0; w < 8; ++w) {
            if (sm[w] == -INFINITY) continue;
            const float e = expf(sm[w] - M);
            L += sl[w] * e;
            O += so[w][d] * e;
        }
        out[(size_t)h * HD + d] = (O / L) * sigmoidf_(gate[(size_t)h * HD + d]);
    }
}


// ================================================================================================ batched (prefill)
template <int TYPE>
__global__ void k_gemv_cols(const void* W, const float* X, float* Y, int n_in, int n_out) {
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const int row = blockIdx.x * (blockDim.x >> 5) + warp, col = blockIdx.y;
    if (row >= n_out) return;
    const float* x = X + (size_t)col * n_in;
    float acc = 0.f;
    for (int i = lane; i < n_in; i += 32) acc += ld_w<TYPE>(W, (size_t)row * n_in + i) * x[i];
    acc = warp_sum(acc);
    if (lane == 0) Y[(size_t)col * n_out + row] = acc;
}

// One thread per channel, sequential over the B tokens (the recurrence of a causal depthwise conv).
__global__ void k_conv_b(float* hist, const float* X, const float* w, float* out, int C, int B) {
    const int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= C) return;
    float h0 = hist[c * 3], h1 = hist[c * 3 + 1], h2 = hist[c * 3 + 2];
    const float w0 = w[c * 4], w1 = w[c * 4 + 1], w2 = w[c * 4 + 2], w3 = w[c * 4 + 3];
    for (int t = 0; t < B; ++t) {
        const float x = X[(size_t)t * C + c];
        float sum = 0.f;
        sum += h0 * w0; sum += h1 * w1; sum += h2 * w2; sum += x * w3;       // same order as the native kernel
        out[(size_t)t * C + c] = sum / (1.0f + expf(-sum));
        h0 = h1; h1 = h2; h2 = x;
    }
    hist[c * 3] = h0; hist[c * 3 + 1] = h1; hist[c * 3 + 2] = h2;
}

// grid = B * rows_per_token, block = 128 (S).  Same arithmetic as Strata's l2_norm kernel.
__global__ void k_l2_b(float* qkv, int C, int rows_per_token, float eps) {
    const int b = blockIdx.x / rows_per_token, h = blockIdx.x % rows_per_token;
    float* row = qkv + (size_t)b * C + (size_t)h * 128;
    const int col = threadIdx.x;
    const float v = row[col];
    const float ss = block_sum(v * v);
    const float scale = rsqrtf(ss / 128.0f + eps / 128.0f);
    row[col] = (scale * v) * (1.0f / sqrtf(128.0f));
}

__global__ void k_gate_beta_b(const float* alpha, const float* dt, const float* ssm_a, float* gate, float* beta, int hv,
                              int total) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= total) return;
    const int h = i % hv;
    const float value = alpha[i] + dt[h];
    const float sp = value > 20.0f ? value : log1pf(expf(value));
    gate[i] = sp * ssm_a[h];
    beta[i] = sigmoidf_(beta[i]);
}

// Same structure (and therefore the same summation order) as native_gdn_step, looped over B tokens:
// grid = (hv, 32), block = 128 = 4 warps; a warp owns ONE state column `col`, each lane 4 rows (i = r*32 + lane).
// state[(i*hv + head)*128 + col]: i = key index.
__global__ void __launch_bounds__(128) k_gdn_step_b(float* state, const float* qkv, const float* gate, const float* beta,
                                                     float* out, int hk, int hv, int C, int B) {
    const int head = blockIdx.x, lane = threadIdx.x & 31, col = blockIdx.y * 4 + (threadIdx.x >> 5);
    const int qh = head % hk;
    const float scale = 1.0f / sqrtf(128.0f);
    float s[4];
#pragma unroll
    for (int r = 0; r < 4; ++r) s[r] = state[((size_t)(r * 32 + lane) * hv + head) * 128 + col];
    for (int t = 0; t < B; ++t) {
        const float* base = qkv + (size_t)t * C;
        float kr[4], qr[4];
#pragma unroll
        for (int r = 0; r < 4; ++r) {
            const int i = r * 32 + lane;
            qr[r] = base[(size_t)qh * 128 + i];
            kr[r] = base[(size_t)hk * 128 + (size_t)qh * 128 + i];
        }
        const float g = expf(gate[(size_t)t * hv + head]);
        const float bt = beta[(size_t)t * hv + head];
        const float v = base[(size_t)2 * hk * 128 + (size_t)head * 128 + col];
        float kvp = 0.f;
#pragma unroll
        for (int r = 0; r < 4; ++r) kvp += s[r] * kr[r];
        const float kv_col = warp_sum(kvp);
        const float delta = (v - g * kv_col) * bt;
        float ap = 0.f;
#pragma unroll
        for (int r = 0; r < 4; ++r) {
            s[r] = g * s[r] + kr[r] * delta;
            ap += s[r] * qr[r];
        }
        const float attn_col = warp_sum(ap);
        if (lane == 0) out[(size_t)t * hv * 128 + (size_t)head * 128 + col] = attn_col * scale;
    }
#pragma unroll
    for (int r = 0; r < 4; ++r) state[((size_t)(r * 32 + lane) * hv + head) * 128 + col] = s[r];
}

__global__ void k_gather(float* dst, const float* src, const int* idx, int n) {
    const float* s = src + (size_t)idx[blockIdx.x] * n;
    float* d = dst + (size_t)blockIdx.x * n;
    for (int i = threadIdx.x; i < n; i += blockDim.x) d[i] = s[i];
}

__global__ void k_moe_combine_b(float* x, const float* D, const float* wts, const int* pos, const float* shexp,
                                const float* sg, int k, int n) {
    const int b = blockIdx.y, i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    float acc = x[(size_t)b * n + i];
    for (int j = 0; j < k; ++j) acc += wts[b * k + j] * D[(size_t)pos[b * k + j] * n + i];
    acc += sigmoidf_(sg[b]) * shexp[(size_t)b * n + i];
    x[(size_t)b * n + i] = acc;
}

__global__ void k_dot_rows(float* out, const float* w, const float* X, int n) {
    const float* x = X + (size_t)blockIdx.x * n;
    float s = 0.f;
    for (int i = threadIdx.x; i < n; i += blockDim.x) s += w[i] * x[i];
    s = block_sum(s);
    if (threadIdx.x == 0) out[blockIdx.x] = s;
}

__global__ void k_prep_q_b(float* q_out, float* gate_out, const float* q2, const float* qnorm, int n_head, int hd, int rot,
                           float theta, int pos0, float eps) {
    const int h = blockIdx.x, b = blockIdx.y;
    const float* q = q2 + (size_t)b * n_head * 2 * hd + (size_t)h * 2 * hd;
    const float* g = q + hd;
    float* qo = q_out + (size_t)b * n_head * hd + (size_t)h * hd;
    float ss = 0.f;
    for (int i = threadIdx.x; i < hd; i += blockDim.x) ss += q[i] * q[i];
    ss = block_sum(ss);
    const float inv = rsqrtf(ss / (float)hd + eps);
    for (int i = threadIdx.x; i < hd; i += blockDim.x) {
        qo[i] = q[i] * inv * qnorm[i];
        gate_out[(size_t)b * n_head * hd + (size_t)h * hd + i] = g[i];
    }
    __syncthreads();
    for (int i = threadIdx.x; i < (rot >> 1); i += blockDim.x) rope_pair(qo, i, rot, theta, pos0 + b);
}

__global__ void k_prep_kv_b(__half* kc, __half* vc, const float* k, const float* v, const float* knorm, int n_kv, int hd,
                            int rot, float theta, int pos0, int max_ctx, float eps) {
    extern __shared__ float ks[];                      // hd floats
    const int h = blockIdx.x, b = blockIdx.y, pos = pos0 + b;
    const float* kin = k + (size_t)b * n_kv * hd + (size_t)h * hd;
    float ss = 0.f;
    for (int i = threadIdx.x; i < hd; i += blockDim.x) ss += kin[i] * kin[i];
    ss = block_sum(ss);
    const float inv = rsqrtf(ss / (float)hd + eps);
    for (int i = threadIdx.x; i < hd; i += blockDim.x) ks[i] = kin[i] * inv * knorm[i];
    __syncthreads();
    for (int i = threadIdx.x; i < (rot >> 1); i += blockDim.x) rope_pair(ks, i, rot, theta, pos);
    __syncthreads();
    const size_t base = ((size_t)h * max_ctx + pos) * hd;
    for (int i = threadIdx.x; i < hd; i += blockDim.x) {
        kc[base + i] = __float2half(ks[i]);
        vc[base + i] = __float2half(v[(size_t)b * n_kv * hd + (size_t)h * hd + i]);
    }
}

// Same algorithm as k_attn; grid = (n_head, B), token b sees pos0 + b + 1 positions.
template <int HD>
__global__ void k_attn_b(float* out, const float* q, const float* gate, const __half* kc, const __half* vc, int n_head,
                         int n_kv, int pos0, int max_ctx) {
    constexpr int DPL = HD / 32;
    __shared__ float sm[8], sl[8];
    __shared__ float so[8][HD];
    const int h = blockIdx.x, b = blockIdx.y, kvh = h / (n_head / n_kv), n_ctx = pos0 + b + 1;
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const float scale = rsqrtf((float)HD);
    const size_t qo = (size_t)b * n_head * HD + (size_t)h * HD;
    float qv[DPL], o[DPL];
#pragma unroll
    for (int j = 0; j < DPL; ++j) {
        qv[j] = q[qo + lane * DPL + j] * scale;
        o[j] = 0.f;
    }
    float m = -INFINITY, l = 0.f;
    const __half* kb = kc + (size_t)kvh * max_ctx * HD;
    const __half* vb = vc + (size_t)kvh * max_ctx * HD;
    for (int t = warp; t < n_ctx; t += 8) {
        float kf[DPL], vf[DPL];
        load_half<DPL>(kb + (size_t)t * HD + lane * DPL, kf);
        float d = 0.f;
#pragma unroll
        for (int j = 0; j < DPL; ++j) d += qv[j] * kf[j];
        d = warp_sum(d);
        const float mn = fmaxf(m, d);
        const float a = expf(m - mn), p = expf(d - mn);
        l = l * a + p;
        load_half<DPL>(vb + (size_t)t * HD + lane * DPL, vf);
#pragma unroll
        for (int j = 0; j < DPL; ++j) o[j] = o[j] * a + p * vf[j];
        m = mn;
    }
#pragma unroll
    for (int j = 0; j < DPL; ++j) so[warp][lane * DPL + j] = o[j];
    if (lane == 0) { sm[warp] = m; sl[warp] = l; }
    __syncthreads();
    const int d = threadIdx.x;
    if (d < HD) {
        float M = -INFINITY;
        for (int w = 0; w < 8; ++w) M = fmaxf(M, sm[w]);
        float L = 0.f, O = 0.f;
        for (int w = 0; w < 8; ++w) {
            if (sm[w] == -INFINITY) continue;
            const float e = expf(sm[w] - M);
            L += sl[w] * e;
            O += so[w][d] * e;
        }
        out[qo + d] = (O / L) * sigmoidf_(gate[qo + d]);
    }
}

}  // namespace

void rmsnorm(float* y, const float* x, const float* w, int rows, int n, float eps, void* st) {
    k_rmsnorm<<<rows, 256, 0, S(st)>>>(y, x, w, n, eps);
}

void gemv_float(int type, const void* W, const float* x, float* y, int n_in, int n_out, void* st) {
    const int blocks = (n_out + 7) / 8;
    if (type == 0) k_gemv<0><<<blocks, 256, 0, S(st)>>>(W, x, y, n_in, n_out);
    else if (type == 1) k_gemv<1><<<blocks, 256, 0, S(st)>>>(W, x, y, n_in, n_out);
    else if (type == 30) k_gemv<30><<<blocks, 256, 0, S(st)>>>(W, x, y, n_in, n_out);
    else { std::fprintf(stderr, "gemv_float: unsupported type %d\n", type); std::abort(); }
}

void add_inplace(float* y, const float* x, int n, void* st) { k_add<<<(n + 255) / 256, 256, 0, S(st)>>>(y, x, n); }

void silu_mul(float* out, const float* g, const float* u, int n, void* st) {
    k_silu_mul<<<(n + 255) / 256, 256, 0, S(st)>>>(out, g, u, n);
}

void router_topk(const float* logits, int n_expert, int k, int* ids, float* wts, void* st) {
    if (k > 64) { std::fprintf(stderr, "router_topk: k > 64\n"); std::abort(); }
    k_router<<<1, 256, (size_t)n_expert * sizeof(float), S(st)>>>(logits, n_expert, k, ids, wts);
}

void router_topk_rows(const float* logits, int rows, int n_expert, int k, int* ids, float* wts, void* st) {
    if (k > 64) { std::fprintf(stderr, "router_topk_rows: k > 64\n"); std::abort(); }
    k_router<<<rows, 256, (size_t)n_expert * sizeof(float), S(st)>>>(logits, n_expert, k, ids, wts);
}

void moe_combine(float* x, const float* down, const float* wts, const float* shexp, const float* sg, int k, int n,
                 void* st) {
    k_moe_combine<<<(n + 255) / 256, 256, 0, S(st)>>>(x, down, wts, shexp, sg, k, n);
}

void dot_f32(float* out, const float* a, const float* b, int n, void* st) { k_dot<<<1, 256, 0, S(st)>>>(out, a, b, n); }

void gdn_out_norm_silu(float* dst, const float* o, const float* z, const float* gamma, int heads, int cols, float eps,
                       void* st) {
    k_gdn_out_norm<<<heads, 128, 0, S(st)>>>(dst, o, z, gamma, cols, eps);
}

void attn_prep_q(float* q_out, float* gate_out, const float* q2, const float* qnorm, int n_head, int hd, int rot,
                 float theta, int pos, float eps, void* st) {
    k_prep_q<<<n_head, 256, 0, S(st)>>>(q_out, gate_out, q2, qnorm, hd, rot, theta, pos, eps);
}

void attn_prep_kv(__half* kc, __half* vc, const float* k, const float* v, const float* knorm, int n_kv, int hd, int rot,
                  float theta, int pos, int max_ctx, float eps, void* st) {
    // staging for the normalised k: n_kv*hd floats (tiny); allocated once per process.
    static float* scratch = nullptr;
    static size_t cap = 0;
    const size_t need = (size_t)n_kv * hd;
    if (need > cap) {
        if (scratch) cudaFree(scratch);
        cudaMalloc(&scratch, need * sizeof(float));
        cap = need;
    }
    k_prep_kv<<<n_kv, 256, 0, S(st)>>>(kc, vc, k, v, knorm, hd, rot, theta, pos, max_ctx, eps, scratch);
}

void attn_decode(float* out, const float* q, const float* gate, const __half* kc, const __half* vc, int n_head, int n_kv,
                 int hd, int n_ctx, int max_ctx, void* st) {
    if (hd == 256) k_attn<256><<<n_head, 256, 0, S(st)>>>(out, q, gate, kc, vc, n_head, n_kv, n_ctx, max_ctx);
    else if (hd == 128) k_attn<128><<<n_head, 256, 0, S(st)>>>(out, q, gate, kc, vc, n_head, n_kv, n_ctx, max_ctx);
    else { std::fprintf(stderr, "attn_decode: head_dim %d unsupported (128 or 256)\n", hd); std::abort(); }
}

void gemv_float_cols(int type, const void* W, const float* X, float* Y, int n_in, int n_out, int ncols, void* st) {
    const dim3 grid((n_out + 7) / 8, ncols);
    if (type == 0) k_gemv_cols<0><<<grid, 256, 0, S(st)>>>(W, X, Y, n_in, n_out);
    else if (type == 1) k_gemv_cols<1><<<grid, 256, 0, S(st)>>>(W, X, Y, n_in, n_out);
    else if (type == 30) k_gemv_cols<30><<<grid, 256, 0, S(st)>>>(W, X, Y, n_in, n_out);
    else { std::fprintf(stderr, "gemv_float_cols: unsupported type %d\n", type); std::abort(); }
}

void gdn_conv_silu_b(float* hist, const float* X, const float* w, float* out, int C, int B, void* st) {
    k_conv_b<<<(C + 255) / 256, 256, 0, S(st)>>>(hist, X, w, out, C, B);
}

void gdn_l2norm_qk_b(float* qkv, int C, int rows_per_token, int Sz, int B, float eps, void* st) {
    if (Sz != 128) { std::fprintf(stderr, "gdn_l2norm_qk_b: S must be 128\n"); std::abort(); }
    k_l2_b<<<B * rows_per_token, 128, 0, S(st)>>>(qkv, C, rows_per_token, eps);
}

void gdn_gate_beta_b(const float* alpha, const float* dt, const float* ssm_a, float* gate, float* beta, int hv, int B,
                     void* st) {
    const int total = hv * B;
    k_gate_beta_b<<<(total + 255) / 256, 256, 0, S(st)>>>(alpha, dt, ssm_a, gate, beta, hv, total);
}

void gdn_step_b(float* state, const float* qkv, const float* gate, const float* beta, float* out, int hk, int hv, int C,
                int B, void* st) {
    k_gdn_step_b<<<dim3(hv, 32), 128, 0, S(st)>>>(state, qkv, gate, beta, out, hk, hv, C, B);
}

void gather_rows(float* dst, const float* src, const int* idx, int n_rows, int n, void* st) {
    if (n_rows > 0) k_gather<<<n_rows, 256, 0, S(st)>>>(dst, src, idx, n);
}

void moe_combine_b(float* x, const float* D, const float* wts, const int* pos, const float* shexp, const float* sg, int B,
                   int k, int n, void* st) {
    k_moe_combine_b<<<dim3((n + 255) / 256, B), 256, 0, S(st)>>>(x, D, wts, pos, shexp, sg, k, n);
}

void dot_rows(float* out, const float* w, const float* X, int rows, int n, void* st) {
    k_dot_rows<<<rows, 256, 0, S(st)>>>(out, w, X, n);
}

void attn_prep_q_b(float* q_out, float* gate_out, const float* q2, const float* qnorm, int n_head, int hd, int rot,
                   float theta, int pos0, int B, float eps, void* st) {
    k_prep_q_b<<<dim3(n_head, B), 256, 0, S(st)>>>(q_out, gate_out, q2, qnorm, n_head, hd, rot, theta, pos0, eps);
}

void attn_prep_kv_b(__half* kc, __half* vc, const float* k, const float* v, const float* knorm, int n_kv, int hd, int rot,
                    float theta, int pos0, int B, int max_ctx, float eps, void* st) {
    k_prep_kv_b<<<dim3(n_kv, B), 256, (size_t)hd * sizeof(float), S(st)>>>(kc, vc, k, v, knorm, n_kv, hd, rot, theta, pos0,
                                                                          max_ctx, eps);
}

void attn_decode_b(float* out, const float* q, const float* gate, const __half* kc, const __half* vc, int n_head, int n_kv,
                   int hd, int pos0, int B, int max_ctx, void* st) {
    const dim3 grid(n_head, B);
    if (hd == 256) k_attn_b<256><<<grid, 256, 0, S(st)>>>(out, q, gate, kc, vc, n_head, n_kv, pos0, max_ctx);
    else if (hd == 128) k_attn_b<128><<<grid, 256, 0, S(st)>>>(out, q, gate, kc, vc, n_head, n_kv, pos0, max_ctx);
    else { std::fprintf(stderr, "attn_decode_b: head_dim %d unsupported (128 or 256)\n", hd); std::abort(); }
}

}  // namespace q36
