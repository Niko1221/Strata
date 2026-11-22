// Ported from strata-inkling2 by kuraneko1 (_ryu15_)
// https://github.com/kuraneko1/Strata (MIT License)
// See docs/third-party/inkling2-kuraneko/CREDITS.md
// inkling2_dspark.cuh - Inkling-2-Mini DSpark speculative drafter (3-layer DFlare + recurrent Markov head).
// Included directly by inkling2_engine.cu inside namespace strata::inkling2.
#pragma once

namespace dspark_detail {

__device__ __forceinline__ float bf16_to_f32_d(uint16_t u) {
    return __uint_as_float((uint32_t) u << 16);
}

__device__ __forceinline__ float warp_sum_d(float v) {
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
    return v;
}

__device__ __forceinline__ float warp_max_d(float v) {
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) v = fmaxf(v, __shfl_xor_sync(0xffffffffu, v, o));
    return v;
}

/// Mix 4 target feature tensors via softmax(fuse_row[0..3]) and RMSNorm: grid (T), block 256.
__global__ void k_dspark_mix_features(const float* __restrict__ f0, const float* __restrict__ f1,
                                      const float* __restrict__ f2, const float* __restrict__ f3,
                                      const float* __restrict__ fuse_row, const float* __restrict__ norm_w,
                                      float* __restrict__ out, int N, float eps) {
    __shared__ float sw[4];
    __shared__ float ssum[8];
    const int t = blockIdx.x;
    if (threadIdx.x == 0) {
        float a[4] = {fuse_row[0], fuse_row[1], fuse_row[2], fuse_row[3]};
        float mx = fmaxf(fmaxf(a[0], a[1]), fmaxf(a[2], a[3]));
        float s = 0.0f;
        for (int k = 0; k < 4; ++k) { a[k] = expf(a[k] - mx); s += a[k]; }
        const float inv = 1.0f / s;
        for (int k = 0; k < 4; ++k) sw[k] = a[k] * inv;
    }
    __syncthreads();
    const float w0 = sw[0], w1 = sw[1], w2 = sw[2], w3 = sw[3];
    const size_t base = (size_t) t * N;
    float* dst = out + base;
    float acc = 0.0f;
    for (int c = threadIdx.x; c < N; c += blockDim.x) {
        const size_t i = base + c;
        const float x = w0 * f0[i] + w1 * f1[i] + w2 * f2[i] + w3 * f3[i];
        dst[c] = x;
        acc += x * x;
    }
    acc = warp_sum_d(acc);
    if ((threadIdx.x & 31) == 0) ssum[threadIdx.x >> 5] = acc;
    __syncthreads();
    float tot = 0.0f;
    if (threadIdx.x < 8) tot = ssum[threadIdx.x];
    if (threadIdx.x < 32) {
#pragma unroll
        for (int o = 4; o > 0; o >>= 1) tot += __shfl_xor_sync(0xffffffffu, tot, o);
        if (threadIdx.x == 0) ssum[0] = rsqrtf(tot / (float) N + eps);
    }
    __syncthreads();
    const float sc = ssum[0];
    for (int c = threadIdx.x; c < N; c += blockDim.x) {
        dst[c] = dst[c] * sc * norm_w[c];
    }
}

/// Fused per-head RMSNorm(k) + NeoX RoPE(k) + Q8 store for injected k_ctx, v_ctx.
/// Grid (n_head_kv, T), Block 64 (d = 0..63 pairs (d, d+64) for D=128).
__global__ void k_dspark_norm_rope_kv_store_q8(const float* __restrict__ K, const float* __restrict__ V,
                                               const float* __restrict__ k_norm_w,
                                               int8_t* __restrict__ kq, __half* __restrict__ ks,
                                               int8_t* __restrict__ vq, __half* __restrict__ vs,
                                               int n_head_kv, int pos0, int cap, int is_swa,
                                               float eps, float freq_base, int n_rot) {
    __shared__ float s_red[2];
    const int hkv = blockIdx.x, t = blockIdx.y;
    const int d = threadIdx.x;   // 0..63
    const int warp = d >> 5;
    const int kv_dim = n_head_kv * 128;
    const size_t src_off = (size_t) t * kv_dim + hkv * 128;
    const float k0_raw = K[src_off + d];
    const float k1_raw = K[src_off + d + 64];
    const float v0 = V[src_off + d];
    const float v1 = V[src_off + d + 64];

    float sq = warp_sum_d(k0_raw * k0_raw + k1_raw * k1_raw);
    if ((d & 31) == 0) s_red[warp] = sq;
    __syncthreads();
    const float rms = rsqrtf((s_red[0] + s_red[1]) / 128.0f + eps);
    float k0 = k0_raw * rms * k_norm_w[d];
    float k1 = k1_raw * rms * k_norm_w[d + 64];

    // NeoX RoPE on pairs (d, d + n_rot/2) when d < n_rot/2
    if (d * 2 < n_rot) {
        const float theta = (float) (pos0 + t) * powf(freq_base, -2.0f * (float) d / (float) n_rot);
        float sv, cv;
        __sincosf(theta, &sv, &cv);
        const float rk0 = k0 * cv - k1 * sv;
        const float rk1 = k0 * sv + k1 * cv;
        k0 = rk0;
        k1 = rk1;
    }

    float ak = warp_max_d(fmaxf(fabsf(k0), fabsf(k1)));
    float av = warp_max_d(fmaxf(fabsf(v0), fabsf(v1)));
    if ((d & 31) == 0) { s_red[warp] = ak; }
    __syncthreads();
    const float amax_k = fmaxf(s_red[0], s_red[1]);
    if ((d & 31) == 0) { s_red[warp] = av; }
    __syncthreads();
    const float amax_v = fmaxf(s_red[0], s_red[1]);

    const int pos = pos0 + t;
    const int slot = is_swa ? (pos % cap) : pos;
    const float dk = amax_k / 127.0f;
    const float dv = amax_v / 127.0f;
    const float idk = dk > 0.0f ? (127.0f / amax_k) : 0.0f;
    const float idv = dv > 0.0f ? (127.0f / amax_v) : 0.0f;

    const size_t dst_off = (size_t) slot * kv_dim + hkv * 128;
    kq[dst_off + d]      = (int8_t) __float2int_rn(k0 * idk);
    kq[dst_off + d + 64] = (int8_t) __float2int_rn(k1 * idk);
    vq[dst_off + d]      = (int8_t) __float2int_rn(v0 * idv);
    vq[dst_off + d + 64] = (int8_t) __float2int_rn(v1 * idv);
    if (d == 0) {
        ks[(size_t) slot * n_head_kv + hkv] = __float2half(dk);
        vs[(size_t) slot * n_head_kv + hkv] = __float2half(dv);
    }
}

/// Fused per-head RMSNorm + NeoX RoPE for Q (blocks 0..n_head-1) and K (blocks n_head..n_head+n_head_kv-1).
/// Reads *pos0_d dynamically from device memory so the DFlare block can be captured in a CUDA Graph.
/// Grid (n_head + n_head_kv, T), Block 64 (D = 128).
__global__ void k_dspark_qk_norm_rope(float* __restrict__ Q, float* __restrict__ K,
                                      const float* __restrict__ q_norm_w, const float* __restrict__ k_norm_w,
                                      int n_head, int n_head_kv, const int32_t* __restrict__ pos0_d,
                                      float eps, float freq_base, int n_rot) {
    __shared__ float s_sq[2];
    const int idx_h = blockIdx.x, t = blockIdx.y;
    const int d = threadIdx.x;
    const int warp = d >> 5;
    const bool is_k = (idx_h >= n_head);
    const int h = is_k ? (idx_h - n_head) : idx_h;
    const int nh = is_k ? n_head_kv : n_head;
    float* ptr = (is_k ? K : Q) + ((size_t) t * nh + h) * 128;
    const float* nw = is_k ? k_norm_w : q_norm_w;

    const float x0 = ptr[d];
    const float x1 = ptr[d + 64];
    float sq = warp_sum_d(x0 * x0 + x1 * x1);
    if ((d & 31) == 0) s_sq[warp] = sq;
    __syncthreads();
    const float rms = rsqrtf((s_sq[0] + s_sq[1]) / 128.0f + eps);
    float y0 = x0 * rms * nw[d];
    float y1 = x1 * rms * nw[d + 64];

    if (d * 2 < n_rot) {
        const float theta = (float) (pos0_d[0] + t) * powf(freq_base, -2.0f * (float) d / (float) n_rot);
        float sv, cv;
        __sincosf(theta, &sv, &cv);
        const float ry0 = y0 * cv - y1 * sv;
        const float ry1 = y0 * sv + y1 * cv;
        y0 = ry0;
        y1 = ry1;
    }
    ptr[d] = y0;
    ptr[d + 64] = y1;
}

/// Non-causal block-uniform attention over context Q8 KV [j0..P-1] + FP32 self KV [0..block_n-1].
/// Reads P = *pos_d dynamically from device memory for CUDA Graph capture.
/// Grid (n_head, block_n), Block 128 (4 warps).
__global__ void k_dspark_attn_q8(const float* __restrict__ Q,
                                 const int8_t* __restrict__ kq_ctx, const __half* __restrict__ ks_ctx,
                                 const int8_t* __restrict__ vq_ctx, const __half* __restrict__ vs_ctx,
                                 const float* __restrict__ K_self, const float* __restrict__ V_self,
                                 float* __restrict__ out,
                                 int n_head, int n_head_kv, int cap, int is_swa, int n_swa,
                                 const int32_t* __restrict__ pos_d, int block_n, float kq_scale) {
    __shared__ float sq[128];
    __shared__ float s_m[4];
    __shared__ float s_l[4];
    __shared__ float s_acc[4][128];

    const int h = blockIdx.x, t = blockIdx.y;
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const int hkv = h / (n_head / n_head_kv);
    const int kv_dim = n_head_kv * 128;
    const int P = pos_d[0];
    const float* qv = Q + ((size_t) t * n_head + h) * 128;
    sq[threadIdx.x] = qv[threadIdx.x];
    __syncthreads();

    const int j0 = is_swa ? max(0, P - n_swa + 1) : 0;
    float m = -1e30f;
    float l = 0.0f;
    float acc0 = 0.0f, acc1 = 0.0f, acc2 = 0.0f, acc3 = 0.0f;

    // 1. Context KV entries j in [j0 .. P-1]
    for (int j = j0 + warp; j < P; j += 4) {
        const int slot = is_swa ? (j % cap) : j;
        const size_t k_off = (size_t) slot * kv_dim + hkv * 128;
        const int8_t* kj = kq_ctx + k_off;
        const int8_t* vj = vq_ctx + k_off;
        const float ksc = __half2float(ks_ctx[(size_t) slot * n_head_kv + hkv]);
        const float vsc = __half2float(vs_ctx[(size_t) slot * n_head_kv + hkv]);
        float dot = sq[lane] * (float) kj[lane]
                  + sq[32 + lane] * (float) kj[32 + lane]
                  + sq[64 + lane] * (float) kj[64 + lane]
                  + sq[96 + lane] * (float) kj[96 + lane];
        dot = warp_sum_d(dot) * ksc;
        const float sc = dot * kq_scale;
        const float m_new = fmaxf(m, sc);
        const float alpha = l > 0.0f ? expf(m - m_new) : 0.0f;
        const float p = expf(sc - m_new);
        l = l * alpha + p;
        const float pv = p * vsc;
        acc0 = acc0 * alpha + pv * (float) vj[lane];
        acc1 = acc1 * alpha + pv * (float) vj[32 + lane];
        acc2 = acc2 * alpha + pv * (float) vj[64 + lane];
        acc3 = acc3 * alpha + pv * (float) vj[96 + lane];
        m = m_new;
    }

    // 2. Self block KV entries s in [0 .. block_n - 1] (non-causal)
    for (int s = warp; s < block_n; s += 4) {
        const size_t s_off = ((size_t) s * n_head_kv + hkv) * 128;
        const float* ks = K_self + s_off;
        const float* vs = V_self + s_off;
        float dot = sq[lane] * ks[lane]
                  + sq[32 + lane] * ks[32 + lane]
                  + sq[64 + lane] * ks[64 + lane]
                  + sq[96 + lane] * ks[96 + lane];
        dot = warp_sum_d(dot);
        const float sc = dot * kq_scale;
        const float m_new = fmaxf(m, sc);
        const float alpha = l > 0.0f ? expf(m - m_new) : 0.0f;
        const float p = expf(sc - m_new);
        l = l * alpha + p;
        acc0 = acc0 * alpha + p * vs[lane];
        acc1 = acc1 * alpha + p * vs[32 + lane];
        acc2 = acc2 * alpha + p * vs[64 + lane];
        acc3 = acc3 * alpha + p * vs[96 + lane];
        m = m_new;
    }

    if (lane == 0) { s_m[warp] = m; s_l[warp] = l; }
    s_acc[warp][lane]      = acc0;
    s_acc[warp][32 + lane] = acc1;
    s_acc[warp][64 + lane] = acc2;
    s_acc[warp][96 + lane] = acc3;
    __syncthreads();

    float mg = -1e30f;
    for (int w = 0; w < 4; ++w) if (s_l[w] > 0.0f && s_m[w] > mg) mg = s_m[w];
    float lg = 0.0f, og = 0.0f;
    for (int w = 0; w < 4; ++w) {
        if (s_l[w] > 0.0f) {
            const float c = expf(s_m[w] - mg);
            lg += s_l[w] * c;
            og += s_acc[w][threadIdx.x] * c;
        }
    }
    out[((size_t) t * n_head + h) * 128 + threadIdx.x] = lg > 0.0f ? (og / lg) : 0.0f;
}

/// Fast SIMD 4x int8 -> 2x half2 decode for DSpark attention.
__device__ __forceinline__ void dspark_i8x4_to_h2x2(uint32_t packed_i8x4, __half2& out0, __half2& out1) {
    const uint32_t u = packed_i8x4 ^ 0x80808080u;
    const uint32_t lo = __byte_perm(u, 0x6400u, 0x5150u);
    const uint32_t hi = __byte_perm(u, 0x6400u, 0x5352u);
    const __half2 bias = __half2half2(__float2half(1152.0f));
    out0 = __hsub2(*reinterpret_cast<const __half2*>(&lo), bias);
    out1 = __hsub2(*reinterpret_cast<const __half2*>(&hi), bias);
}

#if !defined(__CUDA_ARCH__) || __CUDA_ARCH__ >= 700
__device__ __forceinline__ void dspark_i8x4_to_h2x2_fma(uint32_t packed_i8x4, __half2 sc2, __half2 neg_bias_sc2,
                                                        __half2& out0, __half2& out1) {
    const uint32_t u = packed_i8x4 ^ 0x80808080u;
    const uint32_t lo = __byte_perm(u, 0x6400u, 0x5150u);
    const uint32_t hi = __byte_perm(u, 0x6400u, 0x5352u);
    out0 = __hfma2(*reinterpret_cast<const __half2*>(&lo), sc2, neg_bias_sc2);
    out1 = __hfma2(*reinterpret_cast<const __half2*>(&hi), sc2, neg_bias_sc2);
}

__device__ __forceinline__ float dspark_fast_exp2f_hw(float x) {
    float y;
    asm("ex2.approx.ftz.f32 %0, %1;" : "=f"(y) : "f"(x));
    return y;
}
#endif

/// Shared-KV Split-KV non-causal attention for DSpark full-attention layer (GQA-3, block_n <= 8, D=128):
/// 128 threads (8 half-warps = 1 per block token) cooperatively stage BK=16 Q8 KV tokens into shared memory ONCE per split!
/// Uses SIMD i8x4_to_h2x2 decode, __half2 V accumulation, and log2 exp2f softmax. Split 0 also adds self-block [0..block_n-1].
__global__ void __launch_bounds__(128, 6) k_dspark_attn_q8_sharedkv_split(
        const float* __restrict__ Q,
        const int8_t* __restrict__ kq_ctx, const __half* __restrict__ ks_ctx,
        const int8_t* __restrict__ vq_ctx, const __half* __restrict__ vs_ctx,
        const float* __restrict__ K_self, const float* __restrict__ V_self,
        float* __restrict__ split_out,
        int n_head, int n_head_kv, int P, int block_n, float kq_scale, int split_n) {
    constexpr int BK = 16;
    __shared__ uint2 sK[BK][16];
    __shared__ uint2 sV[BK][16];
    __shared__ float sDK[BK];
    __shared__ float sDV[BK];

    const int hkv = blockIdx.x, sp = blockIdx.y;
    const int tid = threadIdx.x, hw = tid >> 4, hlane = tid & 15;
    const int h0 = 3 * hkv, kv_dim = n_head_kv * 128;
    const int base8 = hlane * 8;
    constexpr float kLog2e = 1.4426950408889634f;
    const float kq_l2 = kq_scale * kLog2e;

    const int chunk_sz = max(64, (P + split_n - 1) / split_n);
    const int c_beg = sp * chunk_sz;
    if (c_beg >= P) {
        if (hw < block_n && hlane == 0) {
            for (int g = 0; g < 3; ++g) {
                float* dst = split_out + (((size_t) hw * split_n + sp) * n_head + (h0 + g)) * 130;
                dst[128] = -1e30f;
                dst[129] = 0.0f;
            }
        }
        return;
    }
    const int c_end = min(P, c_beg + chunk_sz);

    __half2 q2[3][4];
    if (hw < block_n) {
#pragma unroll
        for (int g = 0; g < 3; ++g) {
            const float* qp = Q + ((size_t) hw * n_head + (h0 + g)) * 128 + base8;
            q2[g][0] = __float22half2_rn(*reinterpret_cast<const float2*>(qp + 0));
            q2[g][1] = __float22half2_rn(*reinterpret_cast<const float2*>(qp + 2));
            q2[g][2] = __float22half2_rn(*reinterpret_cast<const float2*>(qp + 4));
            q2[g][3] = __float22half2_rn(*reinterpret_cast<const float2*>(qp + 6));
        }
    }

    const __half2 z2 = __half2half2(__float2half(0.0f));
    float m[3] = {-1e30f, -1e30f, -1e30f};
    float l[3] = {0.0f, 0.0f, 0.0f};
    __half2 acc2[3][4];
#pragma unroll
    for (int g = 0; g < 3; ++g) {
#pragma unroll
        for (int i = 0; i < 4; ++i) acc2[g][i] = z2;
    }

    for (int jb = c_beg; jb < c_end; jb += BK) {
        const int nk = min(BK, c_end - jb);
        for (int r = hw; r < nk; r += 8) {
            const int j = jb + r;
            sK[r][hlane] = *reinterpret_cast<const uint2*>(kq_ctx + (size_t) j * kv_dim + hkv * 128 + base8);
            sV[r][hlane] = *reinterpret_cast<const uint2*>(vq_ctx + (size_t) j * kv_dim + hkv * 128 + base8);
            if (hlane == 0) {
                const size_t idx = (size_t) j * n_head_kv + hkv;
                sDK[r] = __half2float(__ldg(&ks_ctx[idx])) * kq_l2;
                sDV[r] = __half2float(__ldg(&vs_ctx[idx]));
            }
        }
        __syncthreads();

        if (hw < block_n) {
            for (int k = 0; k < nk; ++k) {
                const uint2 k8 = sK[k][hlane];
                const uint2 v8 = sV[k][hlane];
                const float dkkq2 = sDK[k], d_v = sDV[k];
                __half2 kh[4], vh[4];
                dspark_i8x4_to_h2x2(k8.x, kh[0], kh[1]);
                dspark_i8x4_to_h2x2(k8.y, kh[2], kh[3]);
                dspark_i8x4_to_h2x2(v8.x, vh[0], vh[1]);
                dspark_i8x4_to_h2x2(v8.y, vh[2], vh[3]);
                float dots[3];
#pragma unroll
                for (int g = 0; g < 3; ++g) {
                    __half2 d2 = __hmul2(q2[g][0], kh[0]);
                    d2 = __hfma2(q2[g][1], kh[1], d2);
                    d2 = __hfma2(q2[g][2], kh[2], d2);
                    d2 = __hfma2(q2[g][3], kh[3], d2);
                    float dt = __half2float(__low2half(d2)) + __half2float(__high2half(d2));
#pragma unroll
                    for (int o = 8; o > 0; o >>= 1) dt += __shfl_xor_sync(__activemask(), dt, o, 16);
                    dots[g] = dt;
                }
#pragma unroll
                for (int g = 0; g < 3; ++g) {
                    const float sc = dots[g] * dkkq2;
                    const float m_new = fmaxf(m[g], sc);
                    const float alpha = l[g] > 0.0f ? exp2f(m[g] - m_new) : 0.0f;
                    const float p = exp2f(sc - m_new);
                    const __half2 a2 = __half2half2(__float2half(alpha));
                    const __half2 pv2 = __half2half2(__float2half(p * d_v));
                    l[g] = l[g] * alpha + p;
#pragma unroll
                    for (int i = 0; i < 4; ++i) acc2[g][i] = __hfma2(acc2[g][i], a2, __hmul2(pv2, vh[i]));
                    m[g] = m_new;
                }
            }
        }
        __syncthreads();
    }

    if (sp == 0 && hw < block_n) {
        for (int s = 0; s < block_n; ++s) {
            const size_t s_off = ((size_t) s * n_head_kv + hkv) * 128 + base8;
            const float* ks = K_self + s_off;
            const float* vs = V_self + s_off;
            __half2 kh[4], vh[4];
#pragma unroll
            for (int i = 0; i < 4; ++i) {
                kh[i] = __float22half2_rn(*reinterpret_cast<const float2*>(ks + 2 * i));
                vh[i] = __float22half2_rn(*reinterpret_cast<const float2*>(vs + 2 * i));
            }
            float dots[3];
#pragma unroll
            for (int g = 0; g < 3; ++g) {
                __half2 d2 = __hmul2(q2[g][0], kh[0]);
                d2 = __hfma2(q2[g][1], kh[1], d2);
                d2 = __hfma2(q2[g][2], kh[2], d2);
                d2 = __hfma2(q2[g][3], kh[3], d2);
                float dt = __half2float(__low2half(d2)) + __half2float(__high2half(d2));
#pragma unroll
                for (int o = 8; o > 0; o >>= 1) dt += __shfl_xor_sync(__activemask(), dt, o, 16);
                dots[g] = dt;
            }
#pragma unroll
            for (int g = 0; g < 3; ++g) {
                const float sc = dots[g] * kq_l2;
                const float m_new = fmaxf(m[g], sc);
                const float alpha = l[g] > 0.0f ? exp2f(m[g] - m_new) : 0.0f;
                const float p = exp2f(sc - m_new);
                const __half2 a2 = __half2half2(__float2half(alpha));
                const __half2 p2 = __half2half2(__float2half(p));
                l[g] = l[g] * alpha + p;
#pragma unroll
                for (int i = 0; i < 4; ++i) acc2[g][i] = __hfma2(acc2[g][i], a2, __hmul2(p2, vh[i]));
                m[g] = m_new;
            }
        }
    }

    if (hw < block_n) {
#pragma unroll
        for (int g = 0; g < 3; ++g) {
            const int h = h0 + g;
            float* dst = split_out + (((size_t) hw * split_n + sp) * n_head + h) * 130;
#pragma unroll
            for (int i = 0; i < 4; ++i) {
                const float2 f2 = __half22float2(acc2[g][i]);
                dst[base8 + 2 * i + 0] = f2.x;
                dst[base8 + 2 * i + 1] = f2.y;
            }
            if (hlane == 0) {
                dst[128] = m[g];
                dst[129] = l[g];
            }
        }
    }
}

/// Zero-Sync Tensor-Core (WMMA 16x16x16) Split-KV non-causal attention for DSpark full-attention (GQA-3, block_n <= 8, D=128):
/// 4 warps (128 thr, 4 blk/SM, 18,432 B SMEM, <= 128 regs, 0 spills): 2 row-tiles (rt=0..1) x 2 independent KV streams (kw=0..1),
/// sharing 16x136 sKV[kw] via 64-thr bar.sync with ZERO __syncthreads(), ZERO sS/sP SMEM via wmma_acc_to_matA_16x16,
/// V prefetch overlapping QK^T, direct FP16 self-block in sp == 0, and single 2-pass epilogue combine.
__global__ void __launch_bounds__(128, 4) k_dspark_attn_q8_wmma_split(
        const float* __restrict__ Q,
        const int8_t* __restrict__ kq_ctx, const __half* __restrict__ ks_ctx,
        const int8_t* __restrict__ vq_ctx, const __half* __restrict__ vs_ctx,
        const float* __restrict__ K_self, const float* __restrict__ V_self,
        float* __restrict__ split_out,
        int n_head, int n_head_kv, const int32_t* __restrict__ pos_d, int block_n, float kq_scale, int split_n) {
#if !defined(__CUDA_ARCH__) || __CUDA_ARCH__ >= 700
    using namespace nvcuda;
    extern __shared__ __align__(16) char smem_raw[];
    // 18,432 B (4 blk/SM): sQ (32*136*2 = 8,704 B) | sKV[2] (2*16*136*2 = 8,704 B) | epilogue scalars (1,024 B).
    __half* sQ = reinterpret_cast<__half*>(smem_raw);
    __half* sKV_all = sQ + 32 * 136;

    const int hkv = blockIdx.x, sp = blockIdx.y;
    const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
    const int rt = warp >> 1;      // 0: rows 0..15, 1: rows 16..31
    const int kw = warp & 1;       // 0..1: 2 independent 64-thr KV streams stepping by 32
    const int h0 = 3 * hkv, kv_dim = n_head_kv * 128;
    const int n_rows = block_n * 3;
    const int P = pos_d[0];
    constexpr float kLog2e = 1.4426950408889634f;
    const float qscale_all = kq_scale * kLog2e;

    const int chunk_sz = max(64, (P + split_n - 1) / split_n);
    const int c_beg = sp * chunk_sz;
    if (c_beg >= P && sp > 0) {
        if (tid < n_rows) {
            float* dst = split_out + (((size_t) (tid / 3) * split_n + sp) * n_head + (h0 + (tid % 3))) * 130;
            dst[128] = -1e30f;
            dst[129] = 0.0f;
        }
        return;
    }
    const int c_end = min(P, c_beg + chunk_sz);

    // 128 threads load 32x128 Q -> sQ in 2 iterations (16 rows/iter, 8 thr/row, 16 elems/thr).
#pragma unroll
    for (int br = 0; br < 32; br += 16) {
        const int qr = br + (tid >> 3), qc0 = (tid & 7) * 16;
        const float* qp = (qr < n_rows) ? (Q + ((size_t) (qr / 3) * n_head + (h0 + (qr % 3))) * 128 + qc0) : nullptr;
        __half2* dst2 = reinterpret_cast<__half2*>(sQ + qr * 136 + qc0);
        if (qp) {
#pragma unroll
            for (int i = 0; i < 4; ++i) {
                const float4 v = *reinterpret_cast<const float4*>(qp + i * 4);
                dst2[i * 2 + 0] = __halves2half2(__float2half(v.x * qscale_all), __float2half(v.y * qscale_all));
                dst2[i * 2 + 1] = __halves2half2(__float2half(v.z * qscale_all), __float2half(v.w * qscale_all));
            }
        } else {
            const __half2 z2 = __halves2half2(__float2half(0.0f), __float2half(0.0f));
#pragma unroll
            for (int i = 0; i < 8; ++i) dst2[i] = z2;
        }
    }
    __syncthreads();

    int acc_r0, acc_r1, acc_c0, acc_c1;
    wmma_acc_layout_16x16(lane, acc_r0, acc_r1, acc_c0, acc_c1);
    const bool is_leader = wmma_row_leader_16(lane);

    float rm0 = -1e30f, rm1 = -1e30f, rl0 = 0.0f, rl1 = 0.0f;
    wmma::fragment<wmma::accumulator, 16, 16, 16, float> acc[8];
#pragma unroll
    for (int n = 0; n < 8; ++n) wmma::fill_fragment(acc[n], 0.0f);

    __half* sKV = sKV_all + kw * 16 * 136;
    __half* sQ_rt = sQ + (rt * 16) * 136;
    const int tid_kw = (rt << 5) | lane;  // 0..63 within the 2-warp group sharing sKV[kw]
    const int kr = tid_kw >> 2;           // 0..15 (4 threads/row, 32 bytes/thread = 2 uint4s!)
    const int kc0 = (tid_kw & 3) * 32;
    const __half2 neg1152 = __half2half2(__float2half(-1152.0f));

    // Bulk context loop (all 16 keys valid): ZERO __syncthreads(), V prefetch overlapping QK^T!
    int jb = c_beg + kw * 16;
    for (; jb <= c_end - 16; jb += 32) {
        const int j = jb + kr;
        {
            const __half2 dk2 = __half2half2(__ldg(&ks_ctx[(size_t) j * n_head_kv + hkv]));
            const __half2 kb = __hmul2(neg1152, dk2);
            const int8_t* kp = kq_ctx + (size_t) j * kv_dim + hkv * 128 + kc0;
            __half* dst = sKV + kr * 136 + kc0;
#pragma unroll
            for (int i = 0; i < 2; ++i) {
                const uint4 raw = *reinterpret_cast<const uint4*>(kp + i * 16);
                __half2 h2[8];
                dspark_i8x4_to_h2x2_fma(raw.x, dk2, kb, h2[0], h2[1]); dspark_i8x4_to_h2x2_fma(raw.y, dk2, kb, h2[2], h2[3]);
                dspark_i8x4_to_h2x2_fma(raw.z, dk2, kb, h2[4], h2[5]); dspark_i8x4_to_h2x2_fma(raw.w, dk2, kb, h2[6], h2[7]);
                *reinterpret_cast<uint4*>(dst + i * 16)     = *reinterpret_cast<const uint4*>(h2);
                *reinterpret_cast<uint4*>(dst + i * 16 + 8) = *reinterpret_cast<const uint4*>(h2 + 4);
            }
        }
        asm volatile("bar.sync %0, 64;" :: "r"(kw + 1) : "memory");

        const __half2 dv2 = __half2half2(__ldg(&vs_ctx[(size_t) j * n_head_kv + hkv]));
        const int8_t* vp = vq_ctx + (size_t) j * kv_dim + hkv * 128 + kc0;
        const uint4 vr0 = *reinterpret_cast<const uint4*>(vp);
        const uint4 vr1 = *reinterpret_cast<const uint4*>(vp + 16);

        wmma::fragment<wmma::accumulator, 16, 16, 16, float> fc;
        wmma::fill_fragment(fc, 0.0f);
#pragma unroll
        for (int k = 0; k < 128; k += 16) {
            wmma::fragment<wmma::matrix_a, 16, 16, 16, __half, wmma::row_major> fa;
            wmma::fragment<wmma::matrix_b, 16, 16, 16, __half, wmma::col_major> fb;
            wmma::load_matrix_sync(fa, sQ_rt + k, 136);
            wmma::load_matrix_sync(fb, sKV + k, 136);
            wmma::mma_sync(fc, fa, fb, fc);
        }
        asm volatile("bar.sync %0, 64;" :: "r"(kw + 1) : "memory");

        {
            const __half2 vb = __hmul2(neg1152, dv2);
            __half* dst = sKV + kr * 136 + kc0;
            __half2 h2[8];
            dspark_i8x4_to_h2x2_fma(vr0.x, dv2, vb, h2[0], h2[1]); dspark_i8x4_to_h2x2_fma(vr0.y, dv2, vb, h2[2], h2[3]);
            dspark_i8x4_to_h2x2_fma(vr0.z, dv2, vb, h2[4], h2[5]); dspark_i8x4_to_h2x2_fma(vr0.w, dv2, vb, h2[6], h2[7]);
            *reinterpret_cast<uint4*>(dst) = *reinterpret_cast<const uint4*>(h2); *reinterpret_cast<uint4*>(dst + 8) = *reinterpret_cast<const uint4*>(h2 + 4);
            dspark_i8x4_to_h2x2_fma(vr1.x, dv2, vb, h2[0], h2[1]); dspark_i8x4_to_h2x2_fma(vr1.y, dv2, vb, h2[2], h2[3]);
            dspark_i8x4_to_h2x2_fma(vr1.z, dv2, vb, h2[4], h2[5]); dspark_i8x4_to_h2x2_fma(vr1.w, dv2, vb, h2[6], h2[7]);
            *reinterpret_cast<uint4*>(dst + 16) = *reinterpret_cast<const uint4*>(h2); *reinterpret_cast<uint4*>(dst + 24) = *reinterpret_cast<const uint4*>(h2 + 4);
        }

        const float mx0 = wmma_row_max_16(fmaxf(fmaxf(fc.x[0], fc.x[1]), fmaxf(fc.x[4], fc.x[5])));
        const float mx1 = wmma_row_max_16(fmaxf(fmaxf(fc.x[2], fc.x[3]), fmaxf(fc.x[6], fc.x[7])));
        const float m_new0 = fmaxf(rm0, mx0), m_new1 = fmaxf(rm1, mx1);
        const float a0 = (rm0 > -1e20f) ? ((m_new0 == rm0) ? 1.0f : dspark_fast_exp2f_hw(rm0 - m_new0)) : 0.0f;
        const float a1 = (rm1 > -1e20f) ? ((m_new1 == rm1) ? 1.0f : dspark_fast_exp2f_hw(rm1 - m_new1)) : 0.0f;
        if (__any_sync(0xffffffffu, a0 != 1.0f || a1 != 1.0f)) {
#pragma unroll
            for (int n = 0; n < 8; ++n) wmma_rescale_acc(acc[n], a0, a1);
        }
        wmma::fragment<wmma::matrix_a, 16, 16, 16, __half, wmma::row_major> fa_p;
        float ps0 = 0.0f, ps1 = 0.0f;
        wmma_acc_to_matA_16x16<false>(fc, m_new0, m_new1, ps0, ps1, fa_p, lane);
        rm0 = m_new0; rl0 = rl0 * a0 + wmma_row_sum_16(ps0);
        rm1 = m_new1; rl1 = rl1 * a1 + wmma_row_sum_16(ps1);
        asm volatile("bar.sync %0, 64;" :: "r"(kw + 1) : "memory");

#pragma unroll
        for (int n = 0; n < 8; ++n) {
            wmma::fragment<wmma::matrix_b, 16, 16, 16, __half, wmma::row_major> fb;
            wmma::load_matrix_sync(fb, sKV + n * 16, 136);
            wmma::mma_sync(acc[n], fa_p, fb, acc[n]);
        }
        asm volatile("bar.sync %0, 64;" :: "r"(kw + 1) : "memory");
    }

    // Partial tail context tile (< 16 keys)
    if (jb < c_end) {
        const int j = jb + kr;
        {
            __half* dst = sKV + kr * 136 + kc0;
            if (j < c_end) {
                const __half2 dk2 = __half2half2(__ldg(&ks_ctx[(size_t) j * n_head_kv + hkv]));
                const __half2 kb = __hmul2(neg1152, dk2);
                const int8_t* kp = kq_ctx + (size_t) j * kv_dim + hkv * 128 + kc0;
#pragma unroll 1
                for (int i = 0; i < 2; ++i) {
                    const uint4 raw = *reinterpret_cast<const uint4*>(kp + i * 16);
                    __half2 h2[8];
                    dspark_i8x4_to_h2x2_fma(raw.x, dk2, kb, h2[0], h2[1]); dspark_i8x4_to_h2x2_fma(raw.y, dk2, kb, h2[2], h2[3]);
                    dspark_i8x4_to_h2x2_fma(raw.z, dk2, kb, h2[4], h2[5]); dspark_i8x4_to_h2x2_fma(raw.w, dk2, kb, h2[6], h2[7]);
                    *reinterpret_cast<uint4*>(dst + i * 16)     = *reinterpret_cast<const uint4*>(h2);
                    *reinterpret_cast<uint4*>(dst + i * 16 + 8) = *reinterpret_cast<const uint4*>(h2 + 4);
                }
            } else {
                const uint4 z = make_uint4(0, 0, 0, 0);
#pragma unroll 1
                for (int i = 0; i < 4; ++i) *reinterpret_cast<uint4*>(dst + i * 8) = z;
            }
        }
        asm volatile("bar.sync %0, 64;" :: "r"(kw + 1) : "memory");

        wmma::fragment<wmma::accumulator, 16, 16, 16, float> fc;
        wmma::fill_fragment(fc, 0.0f);
#pragma unroll 1
        for (int k = 0; k < 128; k += 16) {
            wmma::fragment<wmma::matrix_a, 16, 16, 16, __half, wmma::row_major> fa;
            wmma::fragment<wmma::matrix_b, 16, 16, 16, __half, wmma::col_major> fb;
            wmma::load_matrix_sync(fa, sQ_rt + k, 136);
            wmma::load_matrix_sync(fb, sKV + k, 136);
            wmma::mma_sync(fc, fa, fb, fc);
        }
        asm volatile("bar.sync %0, 64;" :: "r"(kw + 1) : "memory");

        {
            __half* dst = sKV + kr * 136 + kc0;
            if (j < c_end) {
                const __half2 dv2 = __half2half2(__ldg(&vs_ctx[(size_t) j * n_head_kv + hkv]));
                const __half2 vb = __hmul2(neg1152, dv2);
                const int8_t* vp = vq_ctx + (size_t) j * kv_dim + hkv * 128 + kc0;
#pragma unroll 1
                for (int i = 0; i < 2; ++i) {
                    const uint4 raw = *reinterpret_cast<const uint4*>(vp + i * 16);
                    __half2 h2[8];
                    dspark_i8x4_to_h2x2_fma(raw.x, dv2, vb, h2[0], h2[1]); dspark_i8x4_to_h2x2_fma(raw.y, dv2, vb, h2[2], h2[3]);
                    dspark_i8x4_to_h2x2_fma(raw.z, dv2, vb, h2[4], h2[5]); dspark_i8x4_to_h2x2_fma(raw.w, dv2, vb, h2[6], h2[7]);
                    *reinterpret_cast<uint4*>(dst + i * 16)     = *reinterpret_cast<const uint4*>(h2);
                    *reinterpret_cast<uint4*>(dst + i * 16 + 8) = *reinterpret_cast<const uint4*>(h2 + 4);
                }
            } else {
                const uint4 z = make_uint4(0, 0, 0, 0);
#pragma unroll 1
                for (int i = 0; i < 4; ++i) *reinterpret_cast<uint4*>(dst + i * 8) = z;
            }
        }
        const int rem = c_end - jb;
        if (acc_c0     >= rem) { fc.x[0] = -1e30f; fc.x[2] = -1e30f; }
        if (acc_c0 + 1 >= rem) { fc.x[1] = -1e30f; fc.x[3] = -1e30f; }
        if (acc_c1     >= rem) { fc.x[4] = -1e30f; fc.x[6] = -1e30f; }
        if (acc_c1 + 1 >= rem) { fc.x[5] = -1e30f; fc.x[7] = -1e30f; }
        const float mx0 = wmma_row_max_16(fmaxf(fmaxf(fc.x[0], fc.x[1]), fmaxf(fc.x[4], fc.x[5])));
        const float mx1 = wmma_row_max_16(fmaxf(fmaxf(fc.x[2], fc.x[3]), fmaxf(fc.x[6], fc.x[7])));
        const float m_new0 = fmaxf(rm0, mx0), m_new1 = fmaxf(rm1, mx1);
        const float a0 = (rm0 > -1e20f) ? ((m_new0 == rm0) ? 1.0f : dspark_fast_exp2f_hw(rm0 - m_new0)) : 0.0f;
        const float a1 = (rm1 > -1e20f) ? ((m_new1 == rm1) ? 1.0f : dspark_fast_exp2f_hw(rm1 - m_new1)) : 0.0f;
        if (__any_sync(0xffffffffu, a0 != 1.0f || a1 != 1.0f)) {
#pragma unroll
            for (int n = 0; n < 8; ++n) wmma_rescale_acc(acc[n], a0, a1);
        }
        wmma::fragment<wmma::matrix_a, 16, 16, 16, __half, wmma::row_major> fa_p;
        float ps0 = 0.0f, ps1 = 0.0f;
        wmma_acc_to_matA_16x16<true>(fc, m_new0, m_new1, ps0, ps1, fa_p, lane);
        rm0 = m_new0; rl0 = rl0 * a0 + wmma_row_sum_16(ps0);
        rm1 = m_new1; rl1 = rl1 * a1 + wmma_row_sum_16(ps1);
        asm volatile("bar.sync %0, 64;" :: "r"(kw + 1) : "memory");

#pragma unroll
        for (int n = 0; n < 8; ++n) {
            wmma::fragment<wmma::matrix_b, 16, 16, 16, __half, wmma::row_major> fb;
            wmma::load_matrix_sync(fb, sKV + n * 16, 136);
            wmma::mma_sync(acc[n], fa_p, fb, acc[n]);
        }
        asm volatile("bar.sync %0, 64;" :: "r"(kw + 1) : "memory");
    }

    // Split 0 appends exact WMMA self-block [0..block_n-1] via kw == 0 (64 threads, sKV[0]).
    if (sp == 0 && kw == 0) {
        {
            const float* kp = (kr < block_n) ? (K_self + ((size_t) kr * n_head_kv + hkv) * 128 + kc0) : nullptr;
            __half2* dst2 = reinterpret_cast<__half2*>(sKV + kr * 136 + kc0);
            if (kp) {
#pragma unroll 1
                for (int i = 0; i < 8; ++i) {
                    const float4 v = *reinterpret_cast<const float4*>(kp + i * 4);
                    dst2[i * 2 + 0] = __halves2half2(__float2half(v.x), __float2half(v.y));
                    dst2[i * 2 + 1] = __halves2half2(__float2half(v.z), __float2half(v.w));
                }
            } else {
                const __half2 z2 = __halves2half2(__float2half(0.0f), __float2half(0.0f));
#pragma unroll 1
                for (int i = 0; i < 16; ++i) dst2[i] = z2;
            }
        }
        asm volatile("bar.sync 1, 64;" ::: "memory");

        wmma::fragment<wmma::accumulator, 16, 16, 16, float> fc;
        wmma::fill_fragment(fc, 0.0f);
#pragma unroll 1
        for (int k = 0; k < 128; k += 16) {
            wmma::fragment<wmma::matrix_a, 16, 16, 16, __half, wmma::row_major> fa;
            wmma::fragment<wmma::matrix_b, 16, 16, 16, __half, wmma::col_major> fb;
            wmma::load_matrix_sync(fa, sQ_rt + k, 136);
            wmma::load_matrix_sync(fb, sKV + k, 136);
            wmma::mma_sync(fc, fa, fb, fc);
        }
        asm volatile("bar.sync 1, 64;" ::: "memory");

        {
            const float* vp = (kr < block_n) ? (V_self + ((size_t) kr * n_head_kv + hkv) * 128 + kc0) : nullptr;
            __half2* dst2 = reinterpret_cast<__half2*>(sKV + kr * 136 + kc0);
            if (vp) {
#pragma unroll 1
                for (int i = 0; i < 8; ++i) {
                    const float4 v = *reinterpret_cast<const float4*>(vp + i * 4);
                    dst2[i * 2 + 0] = __halves2half2(__float2half(v.x), __float2half(v.y));
                    dst2[i * 2 + 1] = __halves2half2(__float2half(v.z), __float2half(v.w));
                }
            } else {
                const __half2 z2 = __halves2half2(__float2half(0.0f), __float2half(0.0f));
#pragma unroll 1
                for (int i = 0; i < 16; ++i) dst2[i] = z2;
            }
        }
        if (acc_c0     >= block_n) { fc.x[0] = -1e30f; fc.x[2] = -1e30f; }
        if (acc_c0 + 1 >= block_n) { fc.x[1] = -1e30f; fc.x[3] = -1e30f; }
        if (acc_c1     >= block_n) { fc.x[4] = -1e30f; fc.x[6] = -1e30f; }
        if (acc_c1 + 1 >= block_n) { fc.x[5] = -1e30f; fc.x[7] = -1e30f; }
        const float mx0 = wmma_row_max_16(fmaxf(fmaxf(fc.x[0], fc.x[1]), fmaxf(fc.x[4], fc.x[5])));
        const float mx1 = wmma_row_max_16(fmaxf(fmaxf(fc.x[2], fc.x[3]), fmaxf(fc.x[6], fc.x[7])));
        const float m_new0 = fmaxf(rm0, mx0), m_new1 = fmaxf(rm1, mx1);
        const float a0 = (rm0 > -1e20f) ? ((m_new0 == rm0) ? 1.0f : dspark_fast_exp2f_hw(rm0 - m_new0)) : 0.0f;
        const float a1 = (rm1 > -1e20f) ? ((m_new1 == rm1) ? 1.0f : dspark_fast_exp2f_hw(rm1 - m_new1)) : 0.0f;
        if (__any_sync(0xffffffffu, a0 != 1.0f || a1 != 1.0f)) {
#pragma unroll
            for (int n = 0; n < 8; ++n) wmma_rescale_acc(acc[n], a0, a1);
        }
        wmma::fragment<wmma::matrix_a, 16, 16, 16, __half, wmma::row_major> fa_p;
        float ps0 = 0.0f, ps1 = 0.0f;
        wmma_acc_to_matA_16x16<true>(fc, m_new0, m_new1, ps0, ps1, fa_p, lane);
        rm0 = m_new0; rl0 = rl0 * a0 + wmma_row_sum_16(ps0);
        rm1 = m_new1; rl1 = rl1 * a1 + wmma_row_sum_16(ps1);
        asm volatile("bar.sync 1, 64;" ::: "memory");

#pragma unroll
        for (int n = 0; n < 8; ++n) {
            wmma::fragment<wmma::matrix_b, 16, 16, 16, __half, wmma::row_major> fb;
            wmma::load_matrix_sync(fb, sKV + n * 16, 136);
            wmma::mma_sync(acc[n], fa_p, fb, acc[n]);
        }
    }

    // Epilogue combine kw=0 & kw=1 for rt=0 and rt=1: s_out + s_tmp = 16,384 B at [0..16383], scalars at [17408..18431].
    __syncthreads();
    float* s_wm = reinterpret_cast<float*>(smem_raw + 17408); // [2][32] = 256 B
    float* s_wl = s_wm + 64;                                  // [2][32] = 256 B
    float* s_mg = s_wl + 64;                                  // [32]    = 128 B
    float* s_lg = s_mg + 32;                                  // [32]    = 128 B
    if (is_leader) {
        s_wm[kw * 32 + rt * 16 + acc_r0] = rm0; s_wl[kw * 32 + rt * 16 + acc_r0] = rl0;
        s_wm[kw * 32 + rt * 16 + acc_r1] = rm1; s_wl[kw * 32 + rt * 16 + acc_r1] = rl1;
    }
    __syncthreads();
    if (tid < n_rows) {
        float mg = -1e30f;
        if (s_wl[tid]      > 0.0f) mg = fmaxf(mg, s_wm[tid]);
        if (s_wl[32 + tid] > 0.0f) mg = fmaxf(mg, s_wm[32 + tid]);
        float lg = 0.0f;
        if (s_wl[tid]      > 0.0f) lg += s_wl[tid]      * dspark_fast_exp2f_hw(s_wm[tid]      - mg);
        if (s_wl[32 + tid] > 0.0f) lg += s_wl[32 + tid] * dspark_fast_exp2f_hw(s_wm[32 + tid] - mg);
        s_mg[tid] = mg;
        s_lg[tid] = lg;
    }
    __syncthreads();
    {
        const int r0 = rt * 16 + acc_r0, r1 = rt * 16 + acc_r1;
        const float cw0 = (r0 < n_rows && rl0 > 0.0f) ? dspark_fast_exp2f_hw(rm0 - s_mg[r0]) : 0.0f;
        const float cw1 = (r1 < n_rows && rl1 > 0.0f) ? dspark_fast_exp2f_hw(rm1 - s_mg[r1]) : 0.0f;
#pragma unroll
        for (int n = 0; n < 8; ++n) wmma_rescale_acc(acc[n], cw0, cw1);
    }
    float* s_out = reinterpret_cast<float*>(smem_raw);
    float* s_tmp = s_out + 16 * 128;
    for (int g_rt = 0; g_rt < 2; ++g_rt) {
        if (g_rt * 16 >= n_rows) break;
        if (rt == g_rt) {
            float* dst_s = (kw == 0) ? s_out : s_tmp;
#pragma unroll
            for (int n = 0; n < 8; ++n) wmma::store_matrix_sync(dst_s + n * 16, acc[n], 128, wmma::mem_row_major);
        }
        __syncthreads();
        const int r_lim = min(16, n_rows - g_rt * 16);
        for (int r = 0; r < r_lim; ++r) {
            const int rr = g_rt * 16 + r;
            float* dst = split_out + (((size_t) (rr / 3) * split_n + sp) * n_head + (h0 + (rr % 3))) * 130;
            dst[tid] = s_out[r * 128 + tid] + s_tmp[r * 128 + tid];
            if (tid == 0) {
                dst[128] = s_mg[rr];
                dst[129] = s_lg[rr];
            }
        }
        __syncthreads();
    }
#endif
}

/// Combine Split-KV partials for DSpark: grid (n_head, block_n), block 128.
/// Reads P = *pos_d dynamically from device memory for CUDA Graph capture.
__global__ void k_dspark_split_combine(const float* __restrict__ split_out, float* __restrict__ out,
                                       int n_head, const int32_t* __restrict__ pos_d, int split_n) {
    const int h = blockIdx.x, t = blockIdx.y, d = threadIdx.x;
    const int P = pos_d[0];
    const int chunk_sz = max(64, (P + split_n - 1) / split_n);
    const int active_splits = max(1, min(split_n, (P + chunk_sz - 1) / chunk_sz));
    float mg = -1e30f;
    for (int s = 0; s < active_splits; ++s) {
        const float* src = split_out + (((size_t) t * split_n + s) * n_head + h) * 130;
        if (src[129] > 0.0f && src[128] > mg) mg = src[128];
    }
    float lg = 0.0f, og = 0.0f;
    for (int s = 0; s < active_splits; ++s) {
        const float* src = split_out + (((size_t) t * split_n + s) * n_head + h) * 130;
        if (src[129] > 0.0f) {
            const float c = exp2f(src[128] - mg);
            lg += src[129] * c;
            og += src[d] * c;
        }
    }
    out[((size_t) t * n_head + h) * 128 + d] = lg > 0.0f ? (og / lg) : 0.0f;
}

__global__ void k_dspark_broadcast_tail(float* __restrict__ dst, const float* __restrict__ row, int total_rows, int N) {
    const int t = blockIdx.y + 1;
    const int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (t < total_rows && c < N) {
        dst[(size_t) t * N + c] = row[c];
    }
}

__global__ void k_dspark_swiglu(const float* __restrict__ gate, const float* __restrict__ up, float* __restrict__ out, int64_t n) {
    const int64_t i = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        const float g = gate[i];
        out[i] = (g / (1.0f + expf(-g))) * up[i];
    }
}

__global__ void k_dspark_add(float* __restrict__ dst, const float* __restrict__ a, const float* __restrict__ b, int64_t n) {
    const int64_t i = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) dst[i] = a[i] + b[i];
}

__global__ void k_dspark_add3(float* __restrict__ dst, const float* __restrict__ a, const float* __restrict__ b, const float* __restrict__ c, int64_t n) {
    const int64_t i = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) dst[i] = a[i] + b[i] + c[i];
}

__global__ void k_dspark_prep_joint(const float* __restrict__ state, const void* __restrict__ markov_w1, int w1_type,
                                    int prev_id, const float* __restrict__ hidden_i, const float* __restrict__ context,
                                    float* __restrict__ joint_inp, int rank, int N) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    const int total = 2 * rank + 2 * N;
    if (i >= total) return;
    if (i < rank) {
        joint_inp[i] = state[i];
    } else if (i < 2 * rank) {
        const int c = i - rank;
        const size_t idx = (size_t) prev_id * rank + c;
        if (w1_type == 30) joint_inp[i] = bf16_to_f32_d(((const uint16_t*) markov_w1)[idx]);
        else if (w1_type == 1) joint_inp[i] = __half2float(((const __half*) markov_w1)[idx]);
        else joint_inp[i] = ((const float*) markov_w1)[idx];
    } else if (i < 2 * rank + N) {
        joint_inp[i] = hidden_i[i - 2 * rank];
    } else {
        joint_inp[i] = context[i - 2 * rank - N];
    }
}

__global__ void k_dspark_rnn_update(const float* __restrict__ joint_out, const float* __restrict__ joint_b,
                                    float* __restrict__ state, float* __restrict__ latent, int rank) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= rank) return;
    const float g_raw = joint_out[i] + joint_b[i];
    const float c_raw = joint_out[rank + i] + joint_b[rank + i];
    const float l_raw = joint_out[2 * rank + i] + joint_b[2 * rank + i];
    const float gate = 1.0f / (1.0f + expf(-g_raw));
    const float cand = tanhf(c_raw);
    const float s = state[i];
    state[i] = cand + gate * (s - cand);
    latent[i] = tanhf(l_raw);
}

__global__ void k_dspark_copy_row(float* __restrict__ dst, const float* __restrict__ src, int row, int N) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) dst[i] = src[(size_t) row * N + i];
}

__global__ void k_dspark_softmax_probs(const float* __restrict__ logits, const float* __restrict__ logits_buf_row,
                                       int V, float inv_temp, float max_val, float* __restrict__ tmp_sum) {
    // Wait: host-side sampling reuses Engine::sample_logits directly cleanly when probabilistic is active.
    (void) logits; (void) logits_buf_row; (void) V; (void) inv_temp; (void) max_val; (void) tmp_sum;
}

}  // namespace dspark_detail

struct DsparkHparams {
    int n_layer = 3;
    int block_size = 8;
    int n_embd = 2304;
    int n_ff = 9216;
    int n_head = 24;
    int n_head_kv = 8;
    int head_dim = 128;
    int n_rot = 128;
    int q_dim = 3072;
    int kv_dim = 1024;
    int n_swa = 1025;       // gguf sliding_window + 1
    int markov_rank = 256;
    int64_t vocab_padded = 201024;
    int64_t vocab_unpadded = 200058;
    float rms_eps = 1e-6f;
    float freq_base = 1e6f;
    float softcap = 30.0f;
    std::vector<uint8_t> swa_pattern;
    std::vector<int> target_layers = {5, 23, 41, 48};

    bool is_swa(int il) const {
        if ((size_t) il < swa_pattern.size()) return swa_pattern[(size_t) il] != 0;
        return il != 1;
    }
};

struct DsparkLayer {
    bool is_swa = true;
    float* dflare_norm = nullptr;
    float* attn_norm = nullptr;
    float* ffn_norm = nullptr;
    float* attn_q_norm = nullptr;
    float* attn_k_norm = nullptr;
    float* attn_k_ctx_norm = nullptr;

    DenseMat attn_k_ctx;
    DenseMat attn_v_ctx;
    DenseMat attn_q;
    DenseMat attn_k;
    DenseMat attn_v;
    DenseMat attn_output;
    DenseMat ffn_gate;
    DenseMat ffn_up;
    DenseMat ffn_down;
};

struct DsparkModel : GpuWeightStore {
    std::unique_ptr<strata::GgufModel> gguf;
    DsparkHparams hp;

    float* dspark_fuse = nullptr;       // [n_layer + 1, ntarget] = [4, 4]
    float* dspark_norm = nullptr;       // [n_embd]
    float* mask_embd = nullptr;         // [n_embd]
    float* output_norm = nullptr;       // [n_embd]
    void* markov_w1 = nullptr;          // [vocab_padded, markov_rank] on GPU
    int markov_w1_type = 30;
    DenseMat joint_proj;                // [2*rank + 2*N, 3*rank]
    float* joint_proj_b = nullptr;      // [3*rank]
    DenseMat hidden_out;                // [rank, N]
    float* hidden_out_b = nullptr;      // [N]
    std::vector<DsparkLayer> layers;

    ~DsparkModel() { free_store(); }

    const uint8_t* find_tensor(const char* name, const strata::TensorInfo** out = nullptr) const override {
        size_t shard = 0;
        const strata::TensorInfo* t = gguf->find(name, &shard);
        if (out) *out = t;
        return t ? gguf->shard(shard).tensor_data(*t) : nullptr;
    }
};

inline bool load_dspark_model(const std::string& path, const Hparams& target_hp, DsparkModel& dm, std::string& err) {
    const auto t0 = Clock::now();
    try {
        dm.gguf = std::make_unique<strata::GgufModel>(strata::gguf_split_paths(path));
    } catch (const std::exception& ex) {
        err = ex.what();
        return false;
    }
    const strata::GgufFile& meta = dm.gguf->meta();
    const auto* arch = meta.get("general.architecture");
    if (!arch || arch->s != "dflash") {
        err = "expected draft GGUF architecture dflash";
        return false;
    }
    auto get_i = [&](const char* k, int64_t def) -> int64_t {
        const auto* v = meta.get(k);
        return v && v->is_num() ? (int64_t) v->u : def;
    };
    auto get_f = [&](const char* k, float def) -> float {
        const auto* v = meta.get(k);
        return v ? (float) v->num() : def;
    };

    auto& hp = dm.hp;
    hp.n_layer = (int) get_i("dflash.block_count", 3);
    hp.block_size = (int) get_i("dflash.block_size", 8);
    hp.n_embd = (int) get_i("dflash.embedding_length", (int) target_hp.n_embd);
    hp.n_ff = (int) get_i("dflash.feed_forward_length", 9216);
    hp.n_head = (int) get_i("dflash.attention.head_count", 24);
    hp.n_head_kv = (int) get_i("dflash.attention.head_count_kv", 8);
    hp.head_dim = (int) get_i("dflash.attention.key_length", 128);
    hp.n_rot = (int) get_i("dflash.rope.dimension_count", hp.head_dim);
    hp.q_dim = hp.n_head * hp.head_dim;
    hp.kv_dim = hp.n_head_kv * hp.head_dim;
    hp.n_swa = (int) get_i("dflash.attention.sliding_window", 1024) + 1;   // anchor + window before it
    hp.markov_rank = (int) get_i("dspark.markov_rank", 256);
    hp.vocab_padded = get_i("dflash.vocab_size", target_hp.vocab_padded);
    hp.vocab_unpadded = get_i("dflash.vocab_size_unpadded", target_hp.vocab_unpadded);
    hp.rms_eps = get_f("dflash.attention.layer_norm_rms_epsilon", target_hp.rms_eps);
    hp.freq_base = get_f("dflash.rope.freq_base", 1e6f);
    hp.softcap = get_f("dflash.final_logit_softcapping", target_hp.softcap);

    if (hp.n_embd != target_hp.n_embd || hp.vocab_unpadded != target_hp.vocab_unpadded ||
        hp.head_dim != 128 || hp.block_size != 8) {
        err = "dspark hyperparameters incompatible with target model";
        return false;
    }

    if (const auto* v = meta.get("dflash.attention.sliding_window_pattern")) {
        hp.swa_pattern.resize(v->items.size());
        for (size_t i = 0; i < v->items.size(); ++i) hp.swa_pattern[i] = v->items[i].u ? 1 : 0;
    }
    if (const auto* v = meta.get("dflash.target_layer_ids")) {
        hp.target_layers.clear();
        for (const auto& item : v->items) hp.target_layers.push_back((int) item.num());
    }
    if (hp.target_layers.size() != 4) {
        err = "dspark expects 4 target_layer_ids";
        return false;
    }

    const int64_t N = hp.n_embd;
    const int64_t rank = hp.markov_rank;
    dm.dspark_fuse = dm.upload_f32("dspark_fuse.weight", (size_t) (hp.n_layer + 1) * 4);
    dm.dspark_norm = dm.upload_f32("dspark_norm.weight", (size_t) N);
    dm.mask_embd = dm.upload_f32("mask_embd.weight", (size_t) N);
    dm.output_norm = dm.upload_f32("output_norm.weight", (size_t) N);

    {
        const strata::TensorInfo* wt = nullptr;
        const uint8_t* raw = dm.find_tensor("markov_w1.weight", &wt);
        if (!raw || !wt || (wt->type != 0 && wt->type != 1 && wt->type != 30)) {
            err = "markov_w1.weight missing or unsupported type";
            return false;
        }
        dm.markov_w1_type = (int) wt->type;
        const size_t elem_sz = (wt->type == 0) ? sizeof(float) : sizeof(uint16_t);
        const size_t bytes = (size_t) wt->elements() * elem_sz;
        void* d = dm.gpu_alloc<uint8_t>(bytes);
        CK(cudaMemcpy(d, raw, bytes, cudaMemcpyHostToDevice));
        dm.markov_w1 = d;
    }

    dm.joint_proj = dm.upload_dense("joint_proj.weight", (size_t) (3 * rank), (size_t) (2 * rank + 2 * N));
    dm.joint_proj_b = dm.upload_f32("joint_proj.bias", (size_t) (3 * rank));
    dm.hidden_out = dm.upload_dense("dspark_hidden_out.weight", (size_t) N, (size_t) rank);
    dm.hidden_out_b = dm.upload_f32("dspark_hidden_out.bias", (size_t) N);

    size_t fr_vram = 0, tot_vram = 0;
    const bool pack_dspark_q8 = (cudaMemGetInfo(&fr_vram, &tot_vram) == cudaSuccess && tot_vram >= (20ull << 30));
    dm.layers.resize((size_t) hp.n_layer);
    char nm[128];
    for (int il = 0; il < hp.n_layer; ++il) {
        auto& L = dm.layers[(size_t) il];
        L.is_swa = hp.is_swa(il);
        auto tn = [&](const char* sfx) {
            std::snprintf(nm, sizeof nm, "blk.%d.%s", il, sfx);
            return nm;
        };
        L.dflare_norm = dm.upload_f32(tn("dflare_norm.weight"), (size_t) N);
        L.attn_norm = dm.upload_f32(tn("attn_norm.weight"), (size_t) N);
        L.ffn_norm = dm.upload_f32(tn("ffn_norm.weight"), (size_t) N);
        L.attn_q_norm = dm.upload_f32(tn("attn_q_norm.weight"), (size_t) hp.head_dim);
        L.attn_k_norm = dm.upload_f32(tn("attn_k_norm.weight"), (size_t) hp.head_dim);
        L.attn_k_ctx_norm = dm.upload_f32(tn("attn_k_ctx_norm.weight"), (size_t) hp.head_dim);

        L.attn_k_ctx = dm.upload_dense(tn("attn_k_ctx.weight"), (size_t) hp.kv_dim, (size_t) N, false, pack_dspark_q8);
        L.attn_v_ctx = dm.upload_dense(tn("attn_v_ctx.weight"), (size_t) hp.kv_dim, (size_t) N, false, pack_dspark_q8);
        L.attn_q = dm.upload_dense(tn("attn_q.weight"), (size_t) hp.q_dim, (size_t) N, false, pack_dspark_q8);
        L.attn_k = dm.upload_dense(tn("attn_k.weight"), (size_t) hp.kv_dim, (size_t) N, false, pack_dspark_q8);
        L.attn_v = dm.upload_dense(tn("attn_v.weight"), (size_t) hp.kv_dim, (size_t) N, false, pack_dspark_q8);
        L.attn_output = dm.upload_dense(tn("attn_output.weight"), (size_t) N, (size_t) hp.q_dim, false, pack_dspark_q8);
        L.ffn_gate = dm.upload_dense(tn("ffn_gate.weight"), (size_t) hp.n_ff, (size_t) N, false, pack_dspark_q8);
        L.ffn_up = dm.upload_dense(tn("ffn_up.weight"), (size_t) hp.n_ff, (size_t) N, false, pack_dspark_q8);
        L.ffn_down = dm.upload_dense(tn("ffn_down.weight"), (size_t) N, (size_t) hp.n_ff, false, pack_dspark_q8);
    }

    std::fprintf(stderr, "strata-inkling2: loaded dspark draft (%d layers, block=%d, rank=%d) in %.2f s\n",
                 hp.n_layer, hp.block_size, hp.markov_rank, ms_since(t0) / 1000.0);
    return true;
}

struct DsparkEngine {
    const DsparkModel* dm = nullptr;
    int ctx = 0;
    int tmax = 0;
    int pos = 0;

    // Context Q8 KV cache per layer
    std::vector<int> kv_cap;
    std::vector<int8_t*> kq, vq;
    std::vector<__half*> ks, vs;

    // Inject & draft scratch
    float* ctx_norm = nullptr;       // [tmax * N]
    float* ctx_k = nullptr;          // [tmax * kv_dim]
    float* ctx_v = nullptr;          // [tmax * kv_dim]
    float* f_rnn_buf = nullptr;      // [tmax * N]
    float* f_rnn_last = nullptr;     // [N]

    float* block_cur = nullptr;      // [block_size * N]
    float* x_norm = nullptr;         // [block_size * N]
    float* Q = nullptr;              // [block_size * q_dim]
    float* K = nullptr;              // [block_size * kv_dim]
    float* V = nullptr;              // [block_size * kv_dim]
    float* att = nullptr;            // [block_size * q_dim]
    float* split_out = nullptr;      // [block_size * split_full * n_head * 130]
    int split_full = 40;
    float* wo = nullptr;             // [block_size * N]
    float* gate = nullptr;           // [block_size * n_ff]
    float* up = nullptr;             // [block_size * n_ff]

    float* rnn_state = nullptr;      // [rank]
    float* joint_inp = nullptr;      // [2*rank + 2*N]
    float* joint_out = nullptr;      // [3*rank]
    float* rnn_latent = nullptr;     // [rank]
    float* rnn_corr = nullptr;       // [N]
    float* h_pred = nullptr;         // [N]
    int32_t* pos_d = nullptr;        // [1] device pos for CUDA Graph DFlare capture
    float* h_anchor_pin = nullptr;   // pinned [N] host buffer for async anchor upload
    cudaGraphExec_t dflare_graph_short = nullptr; // captured 3-layer DFlare block (pos < 256)
    cudaGraphExec_t dflare_graph_split = nullptr; // captured 3-layer DFlare block (pos >= 256, WMMA Split-KV)
    std::vector<float> h_anchor;     // [N]

    // Stats
    int64_t n_draft_calls = 0;
    int64_t n_drafted = 0;
    int64_t n_accepted = 0;
    double draft_ms = 0.0;
    double inject_ms = 0.0;

    ~DsparkEngine() {
        if (dflare_graph_short) cudaGraphExecDestroy(dflare_graph_short);
        if (dflare_graph_split) cudaGraphExecDestroy(dflare_graph_split);
        for (auto* p : kq) cudaFree(p);
        for (auto* p : vq) cudaFree(p);
        for (auto* p : ks) cudaFree(p);
        for (auto* p : vs) cudaFree(p);
        cudaFree(ctx_norm);
        cudaFree(ctx_k);
        cudaFree(ctx_v);
        cudaFree(f_rnn_buf);
        cudaFree(f_rnn_last);
        cudaFree(block_cur);
        cudaFree(x_norm);
        cudaFree(Q);
        cudaFree(K);
        cudaFree(V);
        cudaFree(att);
        cudaFree(split_out);
        cudaFree(wo);
        cudaFree(gate);
        cudaFree(up);
        cudaFree(rnn_state);
        cudaFree(joint_inp);
        cudaFree(joint_out);
        cudaFree(rnn_latent);
        cudaFree(rnn_corr);
        cudaFree(h_pred);
        cudaFree(pos_d);
        if (h_anchor_pin) cudaFreeHost(h_anchor_pin);
    }

    void init(const DsparkModel& model, int max_ctx, int max_t) {
        dm = &model;
        ctx = max_ctx;
        tmax = std::max(max_t, 16);
        const auto& hp = dm->hp;
        kq.resize((size_t) hp.n_layer, nullptr);
        vq.resize((size_t) hp.n_layer, nullptr);
        ks.resize((size_t) hp.n_layer, nullptr);
        vs.resize((size_t) hp.n_layer, nullptr);
        kv_cap.resize((size_t) hp.n_layer, 0);
        for (int il = 0; il < hp.n_layer; ++il) {
            const int cap = hp.is_swa(il) ? hp.n_swa : max_ctx;
            kv_cap[(size_t) il] = cap;
            CK(cudaMalloc(&kq[(size_t) il], (size_t) cap * hp.kv_dim * sizeof(int8_t)));
            CK(cudaMalloc(&vq[(size_t) il], (size_t) cap * hp.kv_dim * sizeof(int8_t)));
            CK(cudaMalloc(&ks[(size_t) il], (size_t) cap * hp.n_head_kv * sizeof(__half)));
            CK(cudaMalloc(&vs[(size_t) il], (size_t) cap * hp.n_head_kv * sizeof(__half)));
        }
        auto d_f32 = [](float** p, size_t n) { CK(cudaMalloc(p, n * sizeof(float))); };
        const int B = hp.block_size;
        d_f32(&ctx_norm, (size_t) tmax * hp.n_embd);
        d_f32(&ctx_k, (size_t) tmax * hp.kv_dim);
        d_f32(&ctx_v, (size_t) tmax * hp.kv_dim);
        d_f32(&f_rnn_buf, (size_t) tmax * hp.n_embd);
        d_f32(&f_rnn_last, (size_t) hp.n_embd);

        d_f32(&block_cur, (size_t) B * hp.n_embd);
        d_f32(&x_norm, (size_t) B * hp.n_embd);
        d_f32(&Q, (size_t) B * hp.q_dim);
        d_f32(&K, (size_t) B * hp.kv_dim);
        d_f32(&V, (size_t) B * hp.kv_dim);
        d_f32(&att, (size_t) B * hp.q_dim);
        int dev = 0, n_sm = 80;
        cudaGetDevice(&dev);
        cudaDeviceGetAttribute(&n_sm, cudaDevAttrMultiProcessorCount, dev);
        split_full = std::clamp((n_sm * 24) / std::max(1, hp.n_head_kv), 40, 240);
        d_f32(&split_out, (size_t) B * split_full * hp.n_head * 130);
        cudaFuncSetAttribute(dspark_detail::k_dspark_attn_q8_wmma_split,
                             cudaFuncAttributeMaxDynamicSharedMemorySize, 24576);
        cudaFuncSetAttribute(dspark_detail::k_dspark_attn_q8_wmma_split,
                             cudaFuncAttributePreferredSharedMemoryCarveout, cudaSharedmemCarveoutMaxShared);
        d_f32(&wo, (size_t) B * hp.n_embd);
        d_f32(&gate, (size_t) B * hp.n_ff);
        d_f32(&up, (size_t) B * hp.n_ff);

        d_f32(&rnn_state, (size_t) hp.markov_rank);
        d_f32(&joint_inp, (size_t) (2 * hp.markov_rank + 2 * hp.n_embd));
        d_f32(&joint_out, (size_t) (3 * hp.markov_rank));
        d_f32(&rnn_latent, (size_t) hp.markov_rank);
        d_f32(&rnn_corr, (size_t) hp.n_embd);
        d_f32(&h_pred, (size_t) hp.n_embd);
        CK(cudaMalloc(&pos_d, sizeof(int32_t)));
        if (cudaMallocHost(&h_anchor_pin, (size_t) hp.n_embd * sizeof(float)) != cudaSuccess) {
            (void) cudaGetLastError();
            h_anchor_pin = nullptr;
        }
        h_anchor.resize((size_t) hp.n_embd);
        pos = 0;
    }

    void reset(cudaStream_t st) {
        pos = 0;
        CK(cudaMemsetAsync(f_rnn_last, 0, (size_t) dm->hp.n_embd * sizeof(float), st));
    }

    size_t checkpoint_bytes() const {
        const auto& hp = dm->hp;
        size_t total = (size_t) hp.n_embd * sizeof(float);   // f_rnn_last
        for (int il = 0; il < hp.n_layer; ++il) {
            if (!hp.is_swa(il)) continue;
            const size_t cap = (size_t) kv_cap[(size_t) il];
            total += cap * (size_t) hp.kv_dim * 2 + cap * (size_t) hp.n_head_kv * sizeof(__half) * 2;
        }
        return total;
    }

    void save_checkpoint(uint8_t* dst, cudaStream_t st) const {
        const auto& hp = dm->hp;
        size_t off = 0;
        const size_t rnn_bytes = (size_t) hp.n_embd * sizeof(float);
        CK(cudaMemcpyAsync(dst + off, f_rnn_last, rnn_bytes, cudaMemcpyDeviceToHost, st));
        off += rnn_bytes;
        for (int il = 0; il < hp.n_layer; ++il) {
            if (!hp.is_swa(il)) continue;
            const size_t cap = (size_t) kv_cap[(size_t) il];
            const size_t qb = cap * (size_t) hp.kv_dim;
            const size_t sb = cap * (size_t) hp.n_head_kv * sizeof(__half);
            CK(cudaMemcpyAsync(dst + off, kq[(size_t) il], qb, cudaMemcpyDeviceToHost, st)); off += qb;
            CK(cudaMemcpyAsync(dst + off, vq[(size_t) il], qb, cudaMemcpyDeviceToHost, st)); off += qb;
            CK(cudaMemcpyAsync(dst + off, ks[(size_t) il], sb, cudaMemcpyDeviceToHost, st)); off += sb;
            CK(cudaMemcpyAsync(dst + off, vs[(size_t) il], sb, cudaMemcpyDeviceToHost, st)); off += sb;
        }
    }

    void restore_checkpoint(const uint8_t* src, int new_pos, cudaStream_t st) {
        const auto& hp = dm->hp;
        size_t off = 0;
        const size_t rnn_bytes = (size_t) hp.n_embd * sizeof(float);
        CK(cudaMemcpyAsync(f_rnn_last, src + off, rnn_bytes, cudaMemcpyHostToDevice, st));
        off += rnn_bytes;
        for (int il = 0; il < hp.n_layer; ++il) {
            if (!hp.is_swa(il)) continue;
            const size_t cap = (size_t) kv_cap[(size_t) il];
            const size_t qb = cap * (size_t) hp.kv_dim;
            const size_t sb = cap * (size_t) hp.n_head_kv * sizeof(__half);
            CK(cudaMemcpyAsync(kq[(size_t) il], src + off, qb, cudaMemcpyHostToDevice, st)); off += qb;
            CK(cudaMemcpyAsync(vq[(size_t) il], src + off, qb, cudaMemcpyHostToDevice, st)); off += qb;
            CK(cudaMemcpyAsync(ks[(size_t) il], src + off, sb, cudaMemcpyHostToDevice, st)); off += sb;
            CK(cudaMemcpyAsync(vs[(size_t) il], src + off, sb, cudaMemcpyHostToDevice, st)); off += sb;
        }
        pos = new_pos;
    }

    /// Inject `T` committed target feature vectors (from pos0 .. pos0+T-1) into DSpark KV + update `f_rnn_last`.
    template <typename EngineT>
    void inject(EngineT& E, int pos0, int T) {
        if (T <= 0) return;
        const auto t0 = Clock::now();
        const auto& hp = dm->hp;
        const float* f0 = E.d_tgt_feat[0];
        const float* f1 = E.d_tgt_feat[1];
        const float* f2 = E.d_tgt_feat[2];
        const float* f3 = E.d_tgt_feat[3];
        for (int il = 0; il < hp.n_layer; ++il) {
            const auto& L = dm->layers[(size_t) il];
            dspark_detail::k_dspark_mix_features<<<(unsigned) T, 256, 0, E.st>>>(
                f0, f1, f2, f3, dm->dspark_fuse + il * 4, L.dflare_norm, ctx_norm, hp.n_embd, hp.rms_eps);
            E.prep_q8(ctx_norm, E.xq, T, hp.n_embd);
            E.project_preq(ctx_norm, T <= 8 ? E.xq : nullptr, L.attn_k_ctx, ctx_k, T, hp.kv_dim, hp.n_embd);
            E.project_preq(ctx_norm, T <= 8 ? E.xq : nullptr, L.attn_v_ctx, ctx_v, T, hp.kv_dim, hp.n_embd);
            dspark_detail::k_dspark_norm_rope_kv_store_q8<<<dim3((unsigned) hp.n_head_kv, (unsigned) T), 64, 0, E.st>>>(
                ctx_k, ctx_v, L.attn_k_ctx_norm,
                kq[(size_t) il], ks[(size_t) il], vq[(size_t) il], vs[(size_t) il],
                hp.n_head_kv, pos0, kv_cap[(size_t) il], hp.is_swa(il) ? 1 : 0,
                hp.rms_eps, hp.freq_base, hp.n_rot);
        }
        dspark_detail::k_dspark_mix_features<<<(unsigned) T, 256, 0, E.st>>>(
            f0, f1, f2, f3, dm->dspark_fuse + hp.n_layer * 4, dm->dspark_norm, f_rnn_buf, hp.n_embd, hp.rms_eps);
        CK(cudaMemcpyAsync(f_rnn_last, f_rnn_buf + (size_t) (T - 1) * hp.n_embd,
                           (size_t) hp.n_embd * sizeof(float), cudaMemcpyDeviceToDevice, E.st));
        pos = pos0 + T;
        if (E.prof) {
            CK(cudaStreamSynchronize(E.st));
            inject_ms += ms_since(t0);
        }
    }

    /// Draft `n_draft` (1..7) speculative tokens from `anchor` (placed at `pos`, seeing context `0..pos-1`).
    /// When `out_logits != nullptr` (`n_draft` rows of `vocab_unpadded`), uses `sample_fn(row_vec)` for each draft step.
    template <typename EngineT, typename SampleFn>
    void draft(EngineT& E, int32_t anchor, int n_draft, std::vector<int32_t>& out_tokens,
               std::vector<std::vector<float>>* out_logits, SampleFn&& sample_fn) {
        const auto t0 = Clock::now();
        const auto& hp = dm->hp;
        const int block_n = hp.block_size;
        const int N = hp.n_embd;
        const int rank = hp.markov_rank;

        out_tokens.resize((size_t) n_draft);
        if (out_logits) out_logits->resize((size_t) n_draft);

        // 1. Anchor embedding (with target tok_embd_norm) + device pos upload
        float* anchor_src = h_anchor_pin ? h_anchor_pin : h_anchor.data();
        E.m->dequant_tok(anchor, anchor_src);
        CK(cudaMemcpyAsync(x_norm, anchor_src, (size_t) N * sizeof(float), cudaMemcpyHostToDevice, E.st));
        CK(cudaMemcpyAsync(pos_d, &pos, sizeof(int32_t), cudaMemcpyHostToDevice, E.st));

        // 2. 3 DFlare transformer layers over block_cur [block_n, N] + output_norm + rnn_state zeroing (CUDA Graph captured)
        const float kq_scale = 1.0f / std::sqrt((float) hp.head_dim);
        const bool use_split = (pos >= 256 && hp.n_head == 3 * hp.n_head_kv && hp.head_dim == 128 && block_n <= 8);
        auto emit_dflare = [&]() {
            k_rms_norm<<<1, kThreads, 0, E.st>>>(x_norm, E.m->tok_embd_norm, block_cur, N, E.m->hp.rms_eps);
            dspark_detail::k_dspark_broadcast_tail<<<dim3((unsigned) ((N + 255) / 256), (unsigned) (block_n - 1)), 256, 0, E.st>>>(
                block_cur, dm->mask_embd, block_n, N);
            for (int il = 0; il < hp.n_layer; ++il) {
                const auto& L = dm->layers[(size_t) il];
                k_rms_norm<<<(unsigned) block_n, kThreads, 0, E.st>>>(block_cur, L.attn_norm, x_norm, N, hp.rms_eps);
                E.prep_q8(x_norm, E.xq, block_n, N);
                E.project_preq(x_norm, E.xq, L.attn_q, Q, block_n, hp.q_dim, N);
                E.project_preq(x_norm, E.xq, L.attn_k, K, block_n, hp.kv_dim, N);
                E.project_preq(x_norm, E.xq, L.attn_v, V, block_n, hp.kv_dim, N);
                dspark_detail::k_dspark_qk_norm_rope<<<dim3((unsigned) (hp.n_head + hp.n_head_kv), (unsigned) block_n), 64, 0, E.st>>>(
                    Q, K, L.attn_q_norm, L.attn_k_norm, hp.n_head, hp.n_head_kv, pos_d, hp.rms_eps, hp.freq_base, hp.n_rot);
                if (!L.is_swa && use_split) {
                    dspark_detail::k_dspark_attn_q8_wmma_split<<<dim3((unsigned) hp.n_head_kv, (unsigned) split_full), 128, 18432, E.st>>>(
                        Q, kq[(size_t) il], ks[(size_t) il], vq[(size_t) il], vs[(size_t) il], K, V, split_out,
                        hp.n_head, hp.n_head_kv, pos_d, block_n, kq_scale, split_full);
                    dspark_detail::k_dspark_split_combine<<<dim3((unsigned) hp.n_head, (unsigned) block_n), 128, 0, E.st>>>(
                        split_out, att, hp.n_head, pos_d, split_full);
                } else {
                    dspark_detail::k_dspark_attn_q8<<<dim3((unsigned) hp.n_head, (unsigned) block_n), 128, 0, E.st>>>(
                        Q, kq[(size_t) il], ks[(size_t) il], vq[(size_t) il], vs[(size_t) il], K, V, att,
                        hp.n_head, hp.n_head_kv, kv_cap[(size_t) il], hp.is_swa(il) ? 1 : 0, hp.n_swa, pos_d, block_n, kq_scale);
                }
                E.project(att, L.attn_output, wo, block_n, N, hp.q_dim);
                dspark_detail::k_dspark_add<<<(unsigned) (((int64_t) block_n * N + 255) / 256), 256, 0, E.st>>>(
                    block_cur, block_cur, wo, (int64_t) block_n * N);

                k_rms_norm<<<(unsigned) block_n, kThreads, 0, E.st>>>(block_cur, L.ffn_norm, x_norm, N, hp.rms_eps);
                E.prep_q8(x_norm, E.xq, block_n, N);
                E.project_preq(x_norm, E.xq, L.ffn_gate, gate, block_n, hp.n_ff, N);
                E.project_preq(x_norm, E.xq, L.ffn_up, up, block_n, hp.n_ff, N);
                dspark_detail::k_dspark_swiglu<<<(unsigned) (((int64_t) block_n * hp.n_ff + 255) / 256), 256, 0, E.st>>>(
                    gate, up, gate, (int64_t) block_n * hp.n_ff);
                E.project(gate, L.ffn_down, wo, block_n, N, hp.n_ff);
                dspark_detail::k_dspark_add<<<(unsigned) (((int64_t) block_n * N + 255) / 256), 256, 0, E.st>>>(
                    block_cur, block_cur, wo, (int64_t) block_n * N);
            }
            k_rms_norm<<<(unsigned) block_n, kThreads, 0, E.st>>>(block_cur, dm->output_norm, block_cur, N, hp.rms_eps);
            CK(cudaMemsetAsync(rnn_state, 0, (size_t) rank * sizeof(float), E.st));
        };
        if (!E.prof) {
            cudaGraphExec_t& dg = use_split ? dflare_graph_split : dflare_graph_short;
            if (!dg) {
                cudaGraph_t g = nullptr;
                CK(cudaStreamBeginCapture(E.st, cudaStreamCaptureModeThreadLocal));
                emit_dflare();
                CK(cudaStreamEndCapture(E.st, &g));
                CK(cudaGraphInstantiate(&dg, g, nullptr, nullptr, 0));
                CK(cudaGraphDestroy(g));
            }
            CK(cudaGraphLaunch(dg, E.st));
        } else {
            emit_dflare();
        }

        // 3. Recurrent Markov chain for i = 1 .. n_draft
        int32_t prev_id = anchor;
        const int joint_tot = 2 * rank + 2 * N;
        for (int i = 0; i < n_draft; ++i) {
            const float* hi = block_cur + (size_t) (i + 1) * N;
            dspark_detail::k_dspark_prep_joint<<<(unsigned) ((joint_tot + 255) / 256), 256, 0, E.st>>>(
                rnn_state, dm->markov_w1, dm->markov_w1_type, prev_id, hi, f_rnn_last, joint_inp, rank, N);
            E.project(joint_inp, dm->joint_proj, joint_out, 1, 3 * rank, joint_tot);
            dspark_detail::k_dspark_rnn_update<<<(unsigned) ((rank + 255) / 256), 256, 0, E.st>>>(
                joint_out, dm->joint_proj_b, rnn_state, rnn_latent, rank);
            E.project(rnn_latent, dm->hidden_out, rnn_corr, 1, N, rank);
            dspark_detail::k_dspark_add3<<<(unsigned) ((N + 255) / 256), 256, 0, E.st>>>(
                h_pred, hi, rnn_corr, dm->hidden_out_b, N);
            E.project(h_pred, E.m->output_weight, E.logits, 1, hp.vocab_unpadded, N);
            if (out_logits) {
                k_softcap<<<(unsigned) ((hp.vocab_unpadded + 255) / 256), 256, 0, E.st>>>(
                    E.logits, hp.vocab_unpadded, hp.softcap);
                auto& row_lg = (*out_logits)[(size_t) i];
                row_lg.resize((size_t) hp.vocab_unpadded);
                CK(cudaMemcpyAsync(row_lg.data(), E.logits, (size_t) hp.vocab_unpadded * sizeof(float),
                                   cudaMemcpyDeviceToHost, E.st));
                CK(cudaStreamSynchronize(E.st));
                prev_id = sample_fn(i, row_lg);
            } else {
                k_argmax<<<1, 256, 0, E.st>>>(E.logits, E.pick_d, (int) hp.vocab_unpadded);
                int32_t next_id = 0;
                CK(cudaMemcpyAsync(&next_id, E.pick_d, sizeof(int32_t), cudaMemcpyDeviceToHost, E.st));
                CK(cudaStreamSynchronize(E.st));
                prev_id = next_id;
            }
            out_tokens[(size_t) i] = prev_id;
        }
        ++n_draft_calls;
        n_drafted += n_draft;
        if (E.prof) draft_ms += ms_since(t0);
    }
};
