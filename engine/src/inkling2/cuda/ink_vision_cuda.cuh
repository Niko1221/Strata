// engine/src/inkling2/cuda/ink_vision_cuda.cuh - GPU picture reader (23-layer ViT) for Inkling-2-Mini on NVIDIA cards
//
// MIT License
// Copyright (c) 2026 Niko1221 and the Strata contributors
//
// The same network as ink2::CpuVisionEncoder (ink_vision.h, the exact reference): patch embedding, bilinear position embedding, ln_pre, 23 pre-LN blocks
// (axial 2D RoPE, full non-causal attention, LayerScale, GELU MLP), the 2x2 merge projector.  Weights are the file's BF16, kept on the card as BF16 (the
// small tensors and the biases as FP32).  Three ways to multiply (STRATA_INK_VIT_PREC):
//   tc2   (default on sm_80 and newer): the FP32 activation is split into two BF16 numbers (hi + lo, 16 bits of mantissa) and both go through one
//         cuBLAS BF16 tensor-core GEMM with FP32 accumulation, so the result is within about 1e-5 of the FP32 reference.
//   bf16  one BF16 GEMM (activation rounded to BF16): about twice as fast, about 3e-3 relative error.
//   fp32  a plain FP32 kernel with the BF16 weights (default below sm_80, which has no BF16 tensor cores): slowest, same numbers as the CPU reference.
// Attention is cuBLAS FP32 (strided batched over the heads, in chunks of queries so the score matrix stays small), softmax in FP32.
//
// The reader is built to come and go: load() puts the weights on the card (about 0.64 GB), encode() takes its working buffers for one picture and gives them
// back, release() frees everything.  Nothing here ends the process: every failure (no memory, no BF16 code for the card) is an error string and a false
// return, and the caller reads the picture on the CPU instead.
#pragma once

#include "ink_cuda.cuh"
#include "ink_manifest.h"   // the tensor formats ink_vision.h uses (F_BF16 ...)
#include "ink_vision.h"
#include "strata/artifact/strata_reader.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

namespace strata::ink {

enum class VitPrec { TC2 = 0, BF16 = 1, FP32 = 2 };

namespace vit_k {

__device__ __forceinline__ float warp_sum(float v) {
    for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
    return v;
}

// Row layer norm over C, one block of 256 threads per row; mean first, then the mean of squared deviations (as the CPU reference does).
__global__ void __launch_bounds__(256) k_layernorm(const float* __restrict__ x, const float* __restrict__ w, const float* __restrict__ b,
                                                   float* __restrict__ out, int N, int C, float eps) {
    const int row = blockIdx.x;
    if (row >= N) return;
    const int tid = threadIdx.x, wave = tid >> 5, lane = tid & 31;
    const float* xr = x + (size_t)row * C;
    float* yr = out + (size_t)row * C;
    __shared__ float s_w[8];
    __shared__ float s_mean, s_inv;
    float sum = 0.0f;
    for (int c = tid; c < C; c += 256) sum += xr[c];
    sum = warp_sum(sum);
    if (lane == 0) s_w[wave] = sum;
    __syncthreads();
    if (tid == 0) {
        float t = 0.0f;
        for (int i = 0; i < 8; ++i) t += s_w[i];
        s_mean = t / (float)C;
    }
    __syncthreads();
    const float mean = s_mean;
    float var = 0.0f;
    for (int c = tid; c < C; c += 256) {
        const float d = xr[c] - mean;
        var = fmaf(d, d, var);
    }
    var = warp_sum(var);
    __syncthreads();
    if (lane == 0) s_w[wave] = var;
    __syncthreads();
    if (tid == 0) {
        float t = 0.0f;
        for (int i = 0; i < 8; ++i) t += s_w[i];
        s_inv = 1.0f / sqrtf(t / (float)C + eps);
    }
    __syncthreads();
    const float inv = s_inv;
    for (int c = tid; c < C; c += 256) yr[c] = (xr[c] - mean) * inv * w[c] + b[c];
}

// FP32 -> BF16 high part (and, when lo != nullptr, the BF16 remainder), n elements.
__global__ void __launch_bounds__(256) k_to_bf16(const float* __restrict__ x, __nv_bfloat16* __restrict__ hi, __nv_bfloat16* __restrict__ lo, size_t n) {
    const size_t i = (size_t)blockIdx.x * 256 + threadIdx.x;
    if (i >= n) return;
    const float v = x[i];
    const __nv_bfloat16 h = __float2bfloat16_rn(v);
    hi[i] = h;
    if (lo) lo[i] = __float2bfloat16_rn(v - __bfloat162float(h));
}

enum EpiMode { kEpiBias = 0, kEpiBiasGelu = 1, kEpiResLs = 2, kEpiAddAux = 3 };

// Epilogue of a GEMM whose raw result C is [M, N] (tc2: rows M..2M-1 hold the lo part, added here).  In place (out == C, ldo == N) is safe: every
// element is read and written by the same thread.
//   kEpiBias:     out[m, n] = v + bias[n]
//   kEpiBiasGelu: out[m, n] = gelu(v + bias[n])
//   kEpiResLs:    out[m, n] += aux[n] * (v + bias[n])              (LayerScale residual, aux = gamma)
//   kEpiAddAux:   out[m + row0, n] = v + aux[(m + row0) * ldo + n] (patch embedding + position embedding)
__global__ void __launch_bounds__(256) k_epilogue(const float* __restrict__ C, int M, int N, int tc2, const float* __restrict__ bias,
                                                  const float* __restrict__ aux, float* out, int ldo, int row0, int mode) {
    const size_t i = (size_t)blockIdx.x * 256 + threadIdx.x;
    if (i >= (size_t)M * N) return;
    const int m = (int)(i / N), n = (int)(i % N);
    float v = C[i];
    if (tc2) v += C[(size_t)(M + m) * N + n];
    if (bias) v += bias[n];
    if (mode == kEpiBiasGelu) {
        v = 0.5f * v * (1.0f + erff(v * 0.7071067811865475f));
        out[(size_t)m * ldo + n] = v;
    } else if (mode == kEpiResLs) {
        float* o = out + (size_t)m * ldo + n;
        *o = fmaf(aux[n], v, *o);
    } else if (mode == kEpiAddAux) {
        out[(size_t)(m + row0) * ldo + n] = v + aux[(size_t)(m + row0) * ldo + n];
    } else {
        out[(size_t)m * ldo + n] = v;
    }
}

// Plain FP32 GEMM with BF16 weights: C[M, N] = X[M, K] * W[N, K]^T.  64 x 64 tile, 16 x 16 threads of 4 x 4, K step 16.
__global__ void __launch_bounds__(256) k_gemm_f32_bf16w(const float* __restrict__ X, const __nv_bfloat16* __restrict__ W, float* __restrict__ C,
                                                        int M, int N, int K) {
    __shared__ float xs[16][65];
    __shared__ float ws[16][65];
    const int tx = threadIdx.x & 15, ty = threadIdx.x >> 4;
    const int m0 = blockIdx.y * 64, n0 = blockIdx.x * 64;
    float acc[4][4];
#pragma unroll
    for (int i = 0; i < 4; ++i)
#pragma unroll
        for (int j = 0; j < 4; ++j) acc[i][j] = 0.0f;
    for (int k0 = 0; k0 < K; k0 += 16) {
        for (int i = threadIdx.x; i < 1024; i += 256) {
            const int r = i >> 4, c = i & 15;
            const int gm = m0 + r, gn = n0 + r, gk = k0 + c;
            xs[c][r] = (gm < M && gk < K) ? X[(size_t)gm * K + gk] : 0.0f;
            ws[c][r] = (gn < N && gk < K) ? __bfloat162float(W[(size_t)gn * K + gk]) : 0.0f;
        }
        __syncthreads();
#pragma unroll
        for (int kk = 0; kk < 16; ++kk) {
            float a[4], b[4];
#pragma unroll
            for (int i = 0; i < 4; ++i) a[i] = xs[kk][ty * 4 + i];
#pragma unroll
            for (int j = 0; j < 4; ++j) b[j] = ws[kk][tx * 4 + j];
#pragma unroll
            for (int i = 0; i < 4; ++i)
#pragma unroll
                for (int j = 0; j < 4; ++j) acc[i][j] = fmaf(a[i], b[j], acc[i][j]);
        }
        __syncthreads();
    }
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        const int gm = m0 + ty * 4 + i;
        if (gm >= M) continue;
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            const int gn = n0 + tx * 4 + j;
            if (gn < N) C[(size_t)gm * N + gn] = acc[i][j];
        }
    }
}

// Pairwise axial 2D RoPE on Q and K inside the fused QKV matrix [S, 3D] (Q at column 0, K at column D): grid (H, S), block hd/2.
__global__ void __launch_bounds__(64) k_rope_2d(float* __restrict__ qkv, int ld, int D, const float* __restrict__ rc, const float* __restrict__ rs, int S,
                                                int H, int hd) {
    const int pair = threadIdx.x, h = blockIdx.x, t = blockIdx.y;
    if (pair * 2 + 1 >= hd || h >= H || t >= S) return;
    const int j = pair * 2;
    const float c0 = rc[(size_t)t * hd + j], c1 = rc[(size_t)t * hd + j + 1];
    const float s0 = rs[(size_t)t * hd + j], s1 = rs[(size_t)t * hd + j + 1];
    float* qh = qkv + (size_t)t * ld + (size_t)h * hd;
    float* kh = qh + D;
    const float q0 = qh[j], q1 = qh[j + 1];
    qh[j] = q0 * c0 - q1 * s0;
    qh[j + 1] = q1 * c1 + q0 * s1;
    const float k0 = kh[j], k1 = kh[j + 1];
    kh[j] = k0 * c0 - k1 * s0;
    kh[j + 1] = k1 * c1 + k0 * s1;
}

// Softmax over the rows of the (already scaled) score matrix, in place: row r = h * q + qi, one block of 256 threads per row, S columns.
__global__ void __launch_bounds__(256) k_softmax_rows(float* __restrict__ sc, int q, int S, long long head_stride) {
    const int h = blockIdx.x / q, qi = blockIdx.x % q;
    float* row = sc + (size_t)h * (size_t)head_stride + (size_t)qi * S;
    __shared__ float s_w[8];
    __shared__ float s_v;
    const int tid = threadIdx.x, wave = tid >> 5, lane = tid & 31;
    float mx = -1e30f;
    for (int c = tid; c < S; c += 256) mx = fmaxf(mx, row[c]);
    for (int o = 16; o > 0; o >>= 1) mx = fmaxf(mx, __shfl_xor_sync(0xffffffffu, mx, o));
    if (lane == 0) s_w[wave] = mx;
    __syncthreads();
    if (tid == 0) {
        float t = s_w[0];
        for (int i = 1; i < 8; ++i) t = fmaxf(t, s_w[i]);
        s_v = t;
    }
    __syncthreads();
    mx = s_v;
    __syncthreads();
    float sum = 0.0f;
    for (int c = tid; c < S; c += 256) {
        const float e = expf(row[c] - mx);
        row[c] = e;
        sum += e;
    }
    sum = warp_sum(sum);
    if (lane == 0) s_w[wave] = sum;
    __syncthreads();
    if (tid == 0) {
        float t = 0.0f;
        for (int i = 0; i < 8; ++i) t += s_w[i];
        s_v = 1.0f / t;
    }
    __syncthreads();
    const float inv = s_v;
    for (int c = tid; c < S; c += 256) row[c] *= inv;
}

}  // namespace vit_k

class InkVisionCuda {
public:
    ~InkVisionCuda() { release(); }

    bool loaded() const { return loaded_; }
    size_t weight_bytes() const { return w_bytes_; }
    // The bytes held now (weights plus the working buffers of a picture in flight; the latter are 0 between pictures).
    size_t held_bytes() const { return w_bytes_ + act_bytes_; }
    size_t last_act_bytes() const { return last_act_bytes_; }
    double last_encode_ms() const { return last_ms_; }
    VitPrec precision() const { return prec_; }
    const ink2::VitCfg& cfg() const { return info_.c; }

    // BF16 tensor cores (cublasGemmEx with CUDA_R_16BF) need compute capability 8.0 or newer.
    static VitPrec default_precision(int sm) {
        if (const char* e = std::getenv("STRATA_INK_VIT_PREC")) {
            const std::string s(e);
            if (s == "fp32") return VitPrec::FP32;
            if (sm >= 80 && s == "bf16") return VitPrec::BF16;
            if (sm >= 80 && s == "tc2") return VitPrec::TC2;
        }
        return sm >= 80 ? VitPrec::TC2 : VitPrec::FP32;
    }

    // Working memory one picture of n_tok image tokens needs on top of the weights (what encode() allocates), for the planner's reserve.
    static size_t activation_bytes(const ink2::VitCfg& c, int n_tok, VitPrec prec) {
        const size_t S = (size_t)n_tok * c.merge * c.merge + 1;
        const size_t D = c.hidden, hd = c.head_dim;
        const size_t patch_in = 3 * (size_t)c.patch * c.patch;
        const size_t rows2 = (prec == VitPrec::TC2) ? 2 : 1;
        const size_t nmax = std::max<size_t>({3 * D, (size_t)c.inter, (size_t)c.proj_hidden, (size_t)c.out_hidden, D});
        const size_t kmax = std::max<size_t>({D, (size_t)c.inter, (size_t)c.merge * c.merge * D, (size_t)c.proj_hidden, patch_in});
        size_t b = 0;
        b += (S - 1) * patch_in * 4;                       // patches
        b += S * D * 4 * 4;                                // pos, cur, norm, ao
        b += S * hd * 4 * 2;                               // rope cos, sin
        b += rows2 * S * nmax * 4;                         // GEMM result / qkv / MLP hidden
        if (prec != VitPrec::FP32) b += rows2 * S * kmax * 2;   // BF16 copy of the activation (hi and lo)
        else b += S * nmax * 4;                            // second result buffer (a GEMM must not write the buffer it reads)
        b += scores_bytes(c, (int)S);                      // attention scores of one chunk of queries
        b += (size_t)n_tok * c.out_hidden * 4;             // output
        return b + (4u << 20);
    }

    bool load(const std::string& path, int gpu, std::string& err) {
        release();
        gpu_ = gpu;
        if (cudaSetDevice(gpu_) != cudaSuccess) { err = "vision: cudaSetDevice failed"; return false; }
        int sm = 0;
        {
            cudaDeviceProp p{};
            if (cudaGetDeviceProperties(&p, gpu_) != cudaSuccess) { err = "vision: cudaGetDeviceProperties failed"; return false; }
            sm = p.major * 10 + p.minor;
        }
        sm_ = sm;
        prec_ = default_precision(sm);
        if (!ink2::vision_probe(path, info_, err)) return false;
        if (info_.is_safetensors) { err = "vision: the GPU reader takes the .strata Vision shard (the CPU reader reads the safetensors folder)"; return false; }
        const ink2::VitCfg& c = info_.c;
        if (c.head_dim != 64 || c.hidden != c.heads * c.head_dim) { err = "vision: unexpected head size"; return false; }
        strata::artifact::StrataFileReader rd;
        if (!rd.open(path, err)) return false;
        std::vector<uint8_t> s1;
        if (!rd.read_section(1, s1, err)) return false;

        auto need = [&](const std::string& n) -> const ink2::VitTensor* {
            const ink2::VitTensor* t = info_.find(n);
            if (!t) err = "vision shard missing tensor " + n;
            return t;
        };
        auto numel = [](const ink2::VitTensor& t) { size_t n = 1; for (int64_t d : t.shape) n *= (size_t)d; return n; };
        auto to_f32 = [&](const ink2::VitTensor& t, std::vector<float>& v) {
            const size_t n = numel(t);
            v.resize(n);
            const uint8_t* src = s1.data() + t.off;
            if (t.fmt == ink2::F_BF16) { const uint16_t* p = (const uint16_t*)src; for (size_t i = 0; i < n; ++i) v[i] = ink2::bf16_to_f32(p[i]); }
            else if (t.fmt == ink2::F_F16) { const uint16_t* p = (const uint16_t*)src; for (size_t i = 0; i < n; ++i) v[i] = ink2::fp16_to_f32(p[i]); }
            else std::memcpy(v.data(), src, n * 4);
        };
        auto to_bf16 = [&](const ink2::VitTensor& t, std::vector<uint16_t>& v) {
            const size_t n = numel(t);
            v.resize(n);
            if (t.fmt == ink2::F_BF16) { std::memcpy(v.data(), s1.data() + t.off, n * 2); return; }
            std::vector<float> f;
            to_f32(t, f);
            for (size_t i = 0; i < n; ++i) {
                uint32_t u; std::memcpy(&u, &f[i], 4);
                u += 0x7FFFu + ((u >> 16) & 1u);
                v[i] = (uint16_t)(u >> 16);
            }
        };

        // 1. sizes of the two arenas (BF16 matrices, FP32 vectors), 256-byte aligned pieces
        const std::string bb = "model.visual.backbone.";
        struct Piece { std::vector<const ink2::VitTensor*> src; bool bf; size_t off; size_t count; };
        std::vector<Piece> pieces;
        size_t bf_bytes = 0, f_bytes = 0;
        auto add = [&](std::vector<const ink2::VitTensor*> src, bool bf) -> int {
            Piece p; p.src = src; p.bf = bf; p.count = 0;
            for (auto* t : src) p.count += numel(*t);
            size_t& tot = bf ? bf_bytes : f_bytes;
            tot = (tot + 255) & ~(size_t)255;
            p.off = tot;
            tot += p.count * (bf ? 2 : 4);
            pieces.push_back(p);
            return (int)pieces.size() - 1;
        };
        struct LayerIdx { int qkv, bqkv, o, bo, fc, bfc, pr, bpr, ln1w, ln1b, ln2w, ln2b, ls1, ls2; };
        std::vector<LayerIdx> lidx(c.layers);
        auto N1 = [&](const std::string& n) { return need(n); };
        const ink2::VitTensor* t_patch = N1(bb + "embeddings.patch_embedding.weight");
        const ink2::VitTensor* t_cls = N1(bb + "embeddings.class_embedding");
        const ink2::VitTensor* t_clsp = N1(bb + "embeddings.cls_position_embedding");
        const ink2::VitTensor* t_pos = N1(bb + "embeddings.position_embedding");
        const ink2::VitTensor* t_lnw = N1(bb + "ln_pre.weight");
        const ink2::VitTensor* t_lnb = N1(bb + "ln_pre.bias");
        const ink2::VitTensor* t_pin = N1("model.visual.proj_in.weight");
        const ink2::VitTensor* t_pout = N1("model.visual.proj_out.weight");
        if (!t_patch || !t_cls || !t_clsp || !t_pos || !t_lnw || !t_lnb || !t_pin || !t_pout) return false;
        const int i_patch = add({t_patch}, true), i_lnw = add({t_lnw}, false), i_lnb = add({t_lnb}, false);
        const int i_pin = add({t_pin}, true), i_pout = add({t_pout}, true);
        for (int il = 0; il < c.layers; ++il) {
            const std::string p = bb + "layers." + std::to_string(il);
            const ink2::VitTensor *wq = N1(p + ".self_attn.q_proj.weight"), *wk = N1(p + ".self_attn.k_proj.weight"), *wv = N1(p + ".self_attn.v_proj.weight");
            const ink2::VitTensor *bq = N1(p + ".self_attn.q_proj.bias"), *bk = N1(p + ".self_attn.k_proj.bias"), *bv = N1(p + ".self_attn.v_proj.bias");
            const ink2::VitTensor *wo = N1(p + ".self_attn.out_proj.weight"), *bo = N1(p + ".self_attn.out_proj.bias");
            const ink2::VitTensor *wf = N1(p + ".mlp.c_fc.weight"), *bf = N1(p + ".mlp.c_fc.bias");
            const ink2::VitTensor *wp = N1(p + ".mlp.c_proj.weight"), *bp = N1(p + ".mlp.c_proj.bias");
            const ink2::VitTensor *l1w = N1(p + ".ln_1.weight"), *l1b = N1(p + ".ln_1.bias"), *l2w = N1(p + ".ln_2.weight"), *l2b = N1(p + ".ln_2.bias");
            const ink2::VitTensor *g1 = N1(p + ".ls_1.gamma"), *g2 = N1(p + ".ls_2.gamma");
            if (!wq || !wk || !wv || !bq || !bk || !bv || !wo || !bo || !wf || !bf || !wp || !bp || !l1w || !l1b || !l2w || !l2b || !g1 || !g2) return false;
            LayerIdx& L = lidx[il];
            L.qkv = add({wq, wk, wv}, true);  L.bqkv = add({bq, bk, bv}, false);
            L.o = add({wo}, true);            L.bo = add({bo}, false);
            L.fc = add({wf}, true);           L.bfc = add({bf}, false);
            L.pr = add({wp}, true);           L.bpr = add({bp}, false);
            L.ln1w = add({l1w}, false); L.ln1b = add({l1b}, false); L.ln2w = add({l2w}, false); L.ln2b = add({l2b}, false);
            L.ls1 = add({g1}, false); L.ls2 = add({g2}, false);
        }
        bf_bytes = (bf_bytes + 255) & ~(size_t)255;
        f_bytes = (f_bytes + 255) & ~(size_t)255;

        // 2. the card: two allocations
        if (!ck(cudaMalloc(&d_bf_, bf_bytes), "weights", err)) { release(); return false; }
        if (!ck(cudaMalloc(&d_f_, f_bytes), "small weights", err)) { release(); return false; }
        w_bytes_ = bf_bytes + f_bytes;
        std::vector<uint16_t> hb;
        std::vector<float> hf;
        for (const Piece& p : pieces) {
            if (p.bf) {
                std::vector<uint16_t> all;
                for (auto* t : p.src) { to_bf16(*t, hb); all.insert(all.end(), hb.begin(), hb.end()); }
                if (!ck(cudaMemcpy((char*)d_bf_ + p.off, all.data(), all.size() * 2, cudaMemcpyHostToDevice), "weight upload", err)) { release(); return false; }
            } else {
                std::vector<float> all;
                for (auto* t : p.src) { to_f32(*t, hf); all.insert(all.end(), hf.begin(), hf.end()); }
                if (!ck(cudaMemcpy((char*)d_f_ + p.off, all.data(), all.size() * 4, cudaMemcpyHostToDevice), "weight upload", err)) { release(); return false; }
            }
        }
        auto B = [&](int i) { return (const __nv_bfloat16*)((const char*)d_bf_ + pieces[i].off); };
        auto F = [&](int i) { return (const float*)((const char*)d_f_ + pieces[i].off); };
        w_patch_ = B(i_patch); ln_pre_w_ = F(i_lnw); ln_pre_b_ = F(i_lnb); w_pin_ = B(i_pin); w_pout_ = B(i_pout);
        layers_.resize(c.layers);
        for (int il = 0; il < c.layers; ++il) {
            const LayerIdx& L = lidx[il];
            Layer& o = layers_[il];
            o.w_qkv = B(L.qkv); o.b_qkv = F(L.bqkv); o.w_o = B(L.o); o.b_o = F(L.bo); o.w_fc = B(L.fc); o.b_fc = F(L.bfc); o.w_pr = B(L.pr); o.b_pr = F(L.bpr);
            o.ln1w = F(L.ln1w); o.ln1b = F(L.ln1b); o.ln2w = F(L.ln2w); o.ln2b = F(L.ln2b); o.ls1 = F(L.ls1); o.ls2 = F(L.ls2);
        }
        to_f32(*t_cls, h_cls_);
        to_f32(*t_clsp, h_cls_pos_);
        to_f32(*t_pos, h_pos_emb_);
        if ((int)h_cls_.size() != c.hidden || (int)h_cls_pos_.size() != c.hidden ||
            h_pos_emb_.size() != (size_t)c.posemb_grid * c.posemb_grid * c.hidden) {
            err = "vision: unexpected embedding shapes";
            release();
            return false;
        }
        if (cudaStreamCreateWithFlags(&stream_, cudaStreamNonBlocking) != cudaSuccess) { err = "vision: stream"; release(); return false; }
        if (cublasCreate(&blas_) != CUBLAS_STATUS_SUCCESS) { blas_ = nullptr; err = "vision: cublasCreate failed"; release(); return false; }
        cublasSetStream(blas_, stream_);
        cublasSetMathMode(blas_, CUBLAS_DEFAULT_MATH);
        loaded_ = true;
        return true;
    }

    void release() {
        free_act();
        if (blas_) { cublasDestroy(blas_); blas_ = nullptr; }
        if (stream_) { cudaStreamDestroy(stream_); stream_ = nullptr; }
        if (d_bf_) { cudaFree(d_bf_); d_bf_ = nullptr; }
        if (d_f_) { cudaFree(d_f_); d_f_ = nullptr; }
        layers_.clear();
        w_bytes_ = 0;
        loaded_ = false;
    }

    bool encode(const ink2::ImageRec& img, std::vector<float>& out_embeds, std::string& err) {
        if (!loaded_) { err = "vision: reader not loaded"; return false; }
        if (cudaSetDevice(gpu_) != cudaSuccess) { err = "vision: cudaSetDevice failed"; return false; }
        const auto t0 = std::chrono::steady_clock::now();
        const ink2::VitCfg& vc = info_.c;
        const int merge = vc.merge, patch = vc.patch, cell = patch * merge;
        const int n_tok = img.num_lm_tokens();
        const int ph = img.grid_h * merge, pw = img.grid_w * merge;
        const int n_patches = ph * pw, S = n_patches + 1;
        const int D = vc.hidden, H = vc.heads, hd = vc.head_dim, patch_in = 3 * patch * patch;
        if (n_tok <= 0) { err = "vision: empty picture"; return false; }

        // 1. host side (< 20 ms at 1,024 tokens): patchify, interpolated position embedding, 2D RoPE tables; the same arithmetic as the CPU reference
        std::vector<float> patch_buf((size_t)n_patches * patch_in);
        std::vector<int> pos_h(S, 0), pos_w(S, 0);
        {
            int pi = 0;
            for (int gr = 0; gr < img.grid_h; ++gr)
                for (int gc = 0; gc < img.grid_w; ++gc) {
                    const size_t tok = (size_t)gr * img.grid_w + gc;
                    const float* cell_px = img.pixel_values.data() + tok * cell * cell * 3;
                    for (int mr = 0; mr < merge; ++mr)
                        for (int mc = 0; mc < merge; ++mc) {
                            float* dst = patch_buf.data() + (size_t)pi * patch_in;
                            int idx = 0;
                            for (int py = 0; py < patch; ++py) {
                                const int ly = mr * patch + py;
                                for (int px = 0; px < patch; ++px) {
                                    const int lx = mc * patch + px;
                                    const float* rgb = cell_px + ((size_t)ly * cell + lx) * 3;
                                    dst[idx++] = rgb[0]; dst[idx++] = rgb[1]; dst[idx++] = rgb[2];
                                }
                            }
                            pos_h[pi + 1] = gr * merge + mr + 1;
                            pos_w[pi + 1] = gc * merge + mc + 1;
                            ++pi;
                        }
                }
        }
        std::vector<float> h_pos((size_t)S * D, 0.0f);
        for (int d = 0; d < D; ++d) h_pos[d] = h_cls_[d] + h_cls_pos_[d];
        const int G = vc.posemb_grid;
        for (int p = 0; p < n_patches; ++p) {
            const float y = ((float)(pos_h[p + 1] - 1) + 0.5f) * ((float)G / ph) - 0.5f;
            const float x = ((float)(pos_w[p + 1] - 1) + 0.5f) * ((float)G / pw) - 0.5f;
            const int y0 = std::max(0, std::min(G - 1, (int)std::floor(y)));
            const int y1 = std::min(G - 1, y0 + 1);
            const int x0 = std::max(0, std::min(G - 1, (int)std::floor(x)));
            const int x1 = std::min(G - 1, x0 + 1);
            const float wy = std::max(0.0f, std::min(1.0f, y - y0));
            const float wx = std::max(0.0f, std::min(1.0f, x - x0));
            const float w00 = (1.0f - wy) * (1.0f - wx), w01 = (1.0f - wy) * wx, w10 = wy * (1.0f - wx), w11 = wy * wx;
            const float* p00 = h_pos_emb_.data() + ((size_t)y0 * G + x0) * D;
            const float* p01 = h_pos_emb_.data() + ((size_t)y0 * G + x1) * D;
            const float* p10 = h_pos_emb_.data() + ((size_t)y1 * G + x0) * D;
            const float* p11 = h_pos_emb_.data() + ((size_t)y1 * G + x1) * D;
            float* dst = h_pos.data() + (size_t)(p + 1) * D;
            for (int d = 0; d < D; ++d) dst[d] = w00 * p00[d] + w01 * p01[d] + w10 * p10[d] + w11 * p11[d];
        }
        const int sp_dim = hd / 2, n_freq = sp_dim / 2;
        std::vector<float> inv_freq(n_freq);
        for (int i = 0; i < n_freq; ++i) inv_freq[i] = 1.0f / std::pow(vc.rope_theta, (float)(2 * i) / sp_dim);
        std::vector<float> h_rc((size_t)S * hd), h_rs((size_t)S * hd);
        for (int t = 0; t < S; ++t)
            for (int i = 0; i < n_freq; ++i) {
                const float aw = (float)pos_w[t] * inv_freq[i], ah = (float)pos_h[t] * inv_freq[i];
                h_rc[(size_t)t * hd + 2 * i] = h_rc[(size_t)t * hd + 2 * i + 1] = std::cos(aw);
                h_rs[(size_t)t * hd + 2 * i] = h_rs[(size_t)t * hd + 2 * i + 1] = std::sin(aw);
                h_rc[(size_t)t * hd + sp_dim + 2 * i] = h_rc[(size_t)t * hd + sp_dim + 2 * i + 1] = std::cos(ah);
                h_rs[(size_t)t * hd + sp_dim + 2 * i] = h_rs[(size_t)t * hd + sp_dim + 2 * i + 1] = std::sin(ah);
            }

        // 2. the picture's working buffers
        const bool tc = prec_ != VitPrec::FP32;
        const size_t rows2 = (prec_ == VitPrec::TC2) ? 2 : 1;
        const size_t nmax = std::max<size_t>({3 * (size_t)D, (size_t)vc.inter, (size_t)vc.proj_hidden, (size_t)vc.out_hidden, (size_t)D});
        const size_t kmax = std::max<size_t>({(size_t)D, (size_t)vc.inter, (size_t)merge * merge * D, (size_t)vc.proj_hidden, (size_t)patch_in});
        const int qc = query_chunk(vc, S);
        bool ok = true;
        auto A = [&](void** p, size_t bytes, const char* what) {
            if (!ok) return;
            if (!ck(cudaMalloc(p, bytes), what, err)) { ok = false; return; }
            act_bytes_ += bytes;
        };
        A((void**)&d_patch_, (size_t)n_patches * patch_in * 4, "vision patches");
        A((void**)&d_pos_, (size_t)S * D * 4, "vision position embedding");
        A((void**)&d_rc_, (size_t)S * hd * 4, "vision rope");
        A((void**)&d_rs_, (size_t)S * hd * 4, "vision rope");
        A((void**)&d_cur_, (size_t)S * D * 4, "vision activations");
        A((void**)&d_norm_, (size_t)S * D * 4, "vision activations");
        A((void**)&d_ao_, (size_t)S * D * 4, "vision attention output");
        A((void**)&d_c_, rows2 * (size_t)S * nmax * 4, "vision gemm result");
        if (tc) A((void**)&d_abf_, rows2 * (size_t)S * kmax * 2, "vision bf16 activation");
        else A((void**)&d_calt_, (size_t)S * nmax * 4, "vision gemm result");
        A((void**)&d_sc_, (size_t)H * qc * S * 4, "vision attention scores");
        A((void**)&d_out_, (size_t)n_tok * vc.out_hidden * 4, "vision output");
        if (!ok) { cudaGetLastError(); free_act(); return false; }
        last_act_bytes_ = act_bytes_;

        auto up = [&](void* d, const void* h, size_t bytes) {
            if (!ok) return;
            if (!ck(cudaMemcpyAsync(d, h, bytes, cudaMemcpyHostToDevice, stream_), "vision upload", err)) ok = false;
        };
        up(d_patch_, patch_buf.data(), patch_buf.size() * 4);
        up(d_pos_, h_pos.data(), h_pos.size() * 4);
        up(d_rc_, h_rc.data(), h_rc.size() * 4);
        up(d_rs_, h_rs.data(), h_rs.size() * 4);

        // 3. the network
        using namespace vit_k;
        auto lastk = [&](const char* what) { if (ok && cudaGetLastError() != cudaSuccess) { err = std::string("vision kernel failed: ") + what; ok = false; } };
        // y = epilogue(X[M, K] * W[N, K]^T); the result goes to `out` (ldo) through the epilogue, the raw GEMM result lives in d_c_ (and is overwritten in place)
        auto gemm = [&](const float* X, const __nv_bfloat16* W, int M, int N, int K, const float* bias, int mode, const float* aux, float* out, int ldo, int row0) {
            if (!ok) return;
            const int tc2 = (prec_ == VitPrec::TC2) ? 1 : 0;
            float* cbuf = d_c_;
            if (tc) {
                const size_t n = (size_t)M * K;
                const unsigned nb = (unsigned)((n + 255) / 256);
                k_to_bf16<<<nb, 256, 0, stream_>>>(X, d_abf_, tc2 ? d_abf_ + n : nullptr, n);
                lastk("bf16 split");
                const float one = 1.0f, zero = 0.0f;
                const int rows = tc2 ? 2 * M : M;
                const cublasStatus_t st = cublasGemmEx(blas_, CUBLAS_OP_T, CUBLAS_OP_N, N, rows, K, &one, W, CUDA_R_16BF, K, d_abf_, CUDA_R_16BF, K, &zero,
                                                       d_c_, CUDA_R_32F, N, CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP);
                if (ok && st != CUBLAS_STATUS_SUCCESS) { err = "vision: cublasGemmEx failed (" + std::to_string((int)st) + ")"; ok = false; return; }
            } else {
                // the plain kernel reads X while it writes the result: when X is the result buffer, the result goes to the second one
                cbuf = (X == d_c_) ? d_calt_ : d_c_;
                dim3 g((N + 63) / 64, (M + 63) / 64);
                k_gemm_f32_bf16w<<<g, 256, 0, stream_>>>(X, W, cbuf, M, N, K);
                lastk("fp32 gemm");
            }
            const size_t n = (size_t)M * N;
            k_epilogue<<<(unsigned)((n + 255) / 256), 256, 0, stream_>>>(cbuf, M, N, tc2, bias, aux, out, ldo, row0, mode);
            lastk("epilogue");
        };

        // cur[0] = cls + cls_pos; cur[1..] = patch_embedding(patches) + interpolated position embedding
        if (ok) {
            if (!ck(cudaMemcpyAsync(d_cur_, d_pos_, (size_t)D * 4, cudaMemcpyDeviceToDevice, stream_), "vision cls row", err)) ok = false;
        }
        gemm(d_patch_, w_patch_, n_patches, D, patch_in, nullptr, kEpiAddAux, d_pos_, d_cur_, D, 1);
        if (ok) { k_layernorm<<<S, 256, 0, stream_>>>(d_cur_, ln_pre_w_, ln_pre_b_, d_cur_, S, D, vc.ln_eps); lastk("ln_pre"); }
        const float attn_scale = 1.0f / std::sqrt((float)hd);
        const int ld3 = 3 * D;
        for (int il = 0; il < vc.layers && ok; ++il) {
            const Layer& L = layers_[il];
            k_layernorm<<<S, 256, 0, stream_>>>(d_cur_, L.ln1w, L.ln1b, d_norm_, S, D, vc.ln_eps);
            lastk("ln_1");
            gemm(d_norm_, L.w_qkv, S, 3 * D, D, L.b_qkv, kEpiBias, nullptr, d_c_, 3 * D, 0);
            if (!ok) break;
            k_rope_2d<<<dim3(H, S), hd / 2, 0, stream_>>>(d_c_, ld3, D, d_rc_, d_rs_, S, H, hd);
            lastk("rope");
            // attention, a chunk of queries at a time: scores = scale * Q K^T per head, softmax, O = P V
            const long long head_stride = (long long)qc * S;
            for (int qs = 0; qs < S && ok; qs += qc) {
                const int q = std::min(qc, S - qs);
                const float zero = 0.0f, one = 1.0f;
                cublasStatus_t st = cublasSgemmStridedBatched(blas_, CUBLAS_OP_T, CUBLAS_OP_N, S, q, hd, &attn_scale, d_c_ + D, ld3, (long long)hd,
                                                              d_c_ + (size_t)qs * ld3, ld3, (long long)hd, &zero, d_sc_, S, head_stride, H);
                if (st != CUBLAS_STATUS_SUCCESS) { err = "vision: attention scores gemm failed"; ok = false; break; }
                k_softmax_rows<<<H * q, 256, 0, stream_>>>(d_sc_, q, S, head_stride);
                lastk("softmax");
                st = cublasSgemmStridedBatched(blas_, CUBLAS_OP_N, CUBLAS_OP_N, hd, q, S, &one, d_c_ + 2 * D, ld3, (long long)hd, d_sc_, S, head_stride, &zero,
                                               d_ao_ + (size_t)qs * D, D, (long long)hd, H);
                if (st != CUBLAS_STATUS_SUCCESS) { err = "vision: attention values gemm failed"; ok = false; break; }
            }
            gemm(d_ao_, L.w_o, S, D, D, L.b_o, kEpiResLs, L.ls1, d_cur_, D, 0);
            k_layernorm<<<S, 256, 0, stream_>>>(d_cur_, L.ln2w, L.ln2b, d_norm_, S, D, vc.ln_eps);
            lastk("ln_2");
            gemm(d_norm_, L.w_fc, S, vc.inter, D, L.b_fc, kEpiBiasGelu, nullptr, d_c_, vc.inter, 0);
            gemm(d_c_, L.w_pr, S, D, vc.inter, L.b_pr, kEpiResLs, L.ls2, d_cur_, D, 0);
        }
        // 2x2 merge projector (CLS dropped): [n_tok, 4 * D] -> proj_in -> GELU -> proj_out
        const int merged_dim = merge * merge * D;
        gemm(d_cur_ + D, w_pin_, n_tok, vc.proj_hidden, merged_dim, nullptr, kEpiBiasGelu, nullptr, d_c_, vc.proj_hidden, 0);
        gemm(d_c_, w_pout_, n_tok, vc.out_hidden, vc.proj_hidden, nullptr, kEpiBias, nullptr, d_out_, vc.out_hidden, 0);
        if (ok) {
            out_embeds.resize((size_t)n_tok * vc.out_hidden);
            if (!ck(cudaMemcpyAsync(out_embeds.data(), d_out_, out_embeds.size() * 4, cudaMemcpyDeviceToHost, stream_), "vision download", err)) ok = false;
        }
        if (ok && !ck(cudaStreamSynchronize(stream_), "vision run", err)) ok = false;
        free_act();
        cudaGetLastError();
        last_ms_ = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
        return ok;
    }

private:
    struct Layer {
        const __nv_bfloat16 *w_qkv = nullptr, *w_o = nullptr, *w_fc = nullptr, *w_pr = nullptr;
        const float *b_qkv = nullptr, *b_o = nullptr, *b_fc = nullptr, *b_pr = nullptr;
        const float *ln1w = nullptr, *ln1b = nullptr, *ln2w = nullptr, *ln2b = nullptr, *ls1 = nullptr, *ls2 = nullptr;
    };

    // Queries per attention chunk: the score matrix [H, q, S] FP32 stays near 64 MiB.
    static int query_chunk(const ink2::VitCfg& c, int S) {
        const size_t per_q = (size_t)c.heads * S * 4;
        const size_t q = (64u << 20) / std::max<size_t>(1, per_q);
        return (int)std::max<size_t>(32, std::min<size_t>(q, (size_t)S));
    }
    static size_t scores_bytes(const ink2::VitCfg& c, int S) { return (size_t)c.heads * query_chunk(c, S) * S * 4; }

    static bool ck(cudaError_t e, const char* what, std::string& err) {
        if (e == cudaSuccess) return true;
        err = std::string("vision: ") + what + ": " + cudaGetErrorString(e);
        cudaGetLastError();
        return false;
    }

    void free_act() {
        auto fr = [](auto*& p) { if (p) { cudaFree(p); p = nullptr; } };
        fr(d_patch_); fr(d_pos_); fr(d_rc_); fr(d_rs_); fr(d_cur_); fr(d_norm_); fr(d_ao_); fr(d_c_); fr(d_calt_); fr(d_abf_); fr(d_sc_); fr(d_out_);
        act_bytes_ = 0;
    }

    ink2::VisionShardInfo info_;
    bool loaded_ = false;
    int gpu_ = 0, sm_ = 0;
    VitPrec prec_ = VitPrec::TC2;
    void* d_bf_ = nullptr;
    void* d_f_ = nullptr;
    size_t w_bytes_ = 0, act_bytes_ = 0, last_act_bytes_ = 0;
    double last_ms_ = 0.0;
    const __nv_bfloat16 *w_patch_ = nullptr, *w_pin_ = nullptr, *w_pout_ = nullptr;
    const float *ln_pre_w_ = nullptr, *ln_pre_b_ = nullptr;
    std::vector<Layer> layers_;
    std::vector<float> h_cls_, h_cls_pos_, h_pos_emb_;
    cudaStream_t stream_ = nullptr;
    cublasHandle_t blas_ = nullptr;
    float *d_patch_ = nullptr, *d_pos_ = nullptr, *d_rc_ = nullptr, *d_rs_ = nullptr, *d_cur_ = nullptr, *d_norm_ = nullptr, *d_ao_ = nullptr, *d_c_ = nullptr;
    float *d_sc_ = nullptr, *d_out_ = nullptr, *d_calt_ = nullptr;
    __nv_bfloat16* d_abf_ = nullptr;
};

}  // namespace strata::ink
