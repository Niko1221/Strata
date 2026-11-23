// engine/src/inkling2/cuda/ink_cuda_vision.cuh - Inkling-2-Mini Vision ViT on CUDA (InkBackend::encode_image)
//
// The same 23-layer ViT + 2x2 merge projector as CpuVisionEncoder (ink_vision.h), on the GPU.  Without it the CUDA
// engine fell back to the CPU reference for every picture (tens of seconds each, single-threaded).
//
// Loaded on the first picture from the weights CpuVisionEncoder reads.  By default every layer runs in FP32 (cuBLAS
// SGEMM, about 1.3 GB of VRAM): the embeddings match the CPU reference (cosine 1.000000, max |diff| 1e-4 on a 1280x800
// screenshot, 640 tokens, about 60-100 ms on an RTX 5090 against about 37 s on the CPU).  STRATA_INKLING_VIT_BF16=1
// keeps the matrices in BF16 as shipped (about 0.65 GB) and rounds the activations to BF16 for GemmEx, as the HIP
// encoder does (mean cosine 0.9997, worst token 0.88 on the same picture).  Attention: FP32 SGEMM per head over blocks
// of queries, so the score buffer stays near 64 MB for large pictures.  If loading fails (VRAM taken by the experts
// and the KV cache, say), the engine keeps reading pictures on the CPU.
#pragma once

#include <cublas_v2.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <string>
#include <unordered_map>
#include <vector>

#include "ink_manifest.h"
#include "ink_vision.h"

namespace ink2 {
namespace cuvit {

__global__ void k_f32_to_bf16(const float* __restrict__ x, __nv_bfloat16* __restrict__ y, size_t n) {
    const size_t i = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i < n) y[i] = __float2bfloat16(x[i]);
}

__global__ void k_add_bias_rows(float* __restrict__ y, const float* __restrict__ b, int rows, int cols) {
    const size_t i = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i < static_cast<size_t>(rows) * cols) y[i] += b[i % cols];
}

__global__ void k_add(float* __restrict__ x, const float* __restrict__ a, size_t n) {
    const size_t i = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i < n) x[i] += a[i];
}

__global__ void k_gelu_exact(float* __restrict__ x, size_t n) {
    const size_t i = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i < n) { const float v = x[i]; x[i] = 0.5f * v * (1.0f + erff(v * 0.7071067811865475f)); }
}

__global__ void k_layerscale_res(float* __restrict__ cur, const float* __restrict__ proj, const float* __restrict__ g,
                                 int rows, int D) {
    const size_t i = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i < static_cast<size_t>(rows) * D) cur[i] += g[i % D] * proj[i];
}

// One block (256 threads) per row: out = (x - mean) / sqrt(var + eps) * w + b   (out may alias x)
__global__ void k_layernorm(const float* x, const float* __restrict__ w, const float* __restrict__ b, float* out,
                            int D, float eps) {
    __shared__ float red[32];
    const float* xr = x + static_cast<size_t>(blockIdx.x) * D;
    float* orow = out + static_cast<size_t>(blockIdx.x) * D;
    auto block_sum = [&](float v) -> float {
        for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
        if ((threadIdx.x & 31) == 0) red[threadIdx.x >> 5] = v;
        __syncthreads();
        float t = (threadIdx.x < (blockDim.x >> 5)) ? red[threadIdx.x] : 0.0f;
        if (threadIdx.x < 32) for (int o = 16; o > 0; o >>= 1) t += __shfl_xor_sync(0xffffffffu, t, o);
        if (threadIdx.x == 0) red[0] = t;
        __syncthreads();
        const float r = red[0];
        __syncthreads();
        return r;
    };
    float s = 0.0f;
    for (int d = threadIdx.x; d < D; d += blockDim.x) s += xr[d];
    const float mean = block_sum(s) / static_cast<float>(D);
    float v = 0.0f;
    for (int d = threadIdx.x; d < D; d += blockDim.x) { const float t = xr[d] - mean; v += t * t; }
    const float inv = rsqrtf(block_sum(v) / static_cast<float>(D) + eps);
    for (int d = threadIdx.x; d < D; d += blockDim.x) orow[d] = (xr[d] - mean) * inv * w[d] + b[d];
}

// Pairwise axial 2D RoPE on Q and K ([S, H*hd]); grid (H, S), hd/2 threads.
__global__ void k_rope_2d(float* __restrict__ q, float* __restrict__ k, const float* __restrict__ rc,
                          const float* __restrict__ rs, int H, int hd) {
    const int h = blockIdx.x, t = blockIdx.y, j = threadIdx.x * 2;
    if (j >= hd) return;
    const size_t base = (static_cast<size_t>(t) * H + h) * hd;
    const float c0 = rc[static_cast<size_t>(t) * hd + j], c1 = rc[static_cast<size_t>(t) * hd + j + 1];
    const float s0 = rs[static_cast<size_t>(t) * hd + j], s1 = rs[static_cast<size_t>(t) * hd + j + 1];
    const float q0 = q[base + j], q1 = q[base + j + 1];
    q[base + j] = q0 * c0 - q1 * s0;
    q[base + j + 1] = q1 * c1 + q0 * s1;
    const float k0 = k[base + j], k1 = k[base + j + 1];
    k[base + j] = k0 * c0 - k1 * s0;
    k[base + j + 1] = k1 * c1 + k0 * s1;
}

// Row softmax of scale * x over `cols` (in place); one block (256 threads) per row.
__global__ void k_softmax_rows(float* __restrict__ x, int cols, float scale) {
    __shared__ float red[32];
    float* xr = x + static_cast<size_t>(blockIdx.x) * cols;
    float m = -3.0e38f;
    for (int i = threadIdx.x; i < cols; i += blockDim.x) m = fmaxf(m, xr[i] * scale);
    for (int o = 16; o > 0; o >>= 1) m = fmaxf(m, __shfl_xor_sync(0xffffffffu, m, o));
    if ((threadIdx.x & 31) == 0) red[threadIdx.x >> 5] = m;
    __syncthreads();
    if (threadIdx.x < 32) {
        float t = (threadIdx.x < (blockDim.x >> 5)) ? red[threadIdx.x] : -3.0e38f;
        for (int o = 16; o > 0; o >>= 1) t = fmaxf(t, __shfl_xor_sync(0xffffffffu, t, o));
        if (threadIdx.x == 0) red[0] = t;
    }
    __syncthreads();
    m = red[0];
    __syncthreads();
    float s = 0.0f;
    for (int i = threadIdx.x; i < cols; i += blockDim.x) { const float e = __expf(xr[i] * scale - m); xr[i] = e; s += e; }
    for (int o = 16; o > 0; o >>= 1) s += __shfl_xor_sync(0xffffffffu, s, o);
    if ((threadIdx.x & 31) == 0) red[threadIdx.x >> 5] = s;
    __syncthreads();
    if (threadIdx.x < 32) {
        float t = (threadIdx.x < (blockDim.x >> 5)) ? red[threadIdx.x] : 0.0f;
        for (int o = 16; o > 0; o >>= 1) t += __shfl_xor_sync(0xffffffffu, t, o);
        if (threadIdx.x == 0) red[0] = t;
    }
    __syncthreads();
    const float inv = 1.0f / red[0];
    for (int i = threadIdx.x; i < cols; i += blockDim.x) xr[i] *= inv;
}

inline uint16_t f32_to_bf16_rne(float f) {
    uint32_t u;
    std::memcpy(&u, &f, 4);
    if ((u & 0x7fffffffu) > 0x7f800000u) return static_cast<uint16_t>((u >> 16) | 0x40);   // NaN stays NaN
    u += 0x7fffu + ((u >> 16) & 1u);
    return static_cast<uint16_t>(u >> 16);
}

class CudaVit {
public:
    ~CudaVit() { release(); }

    bool loaded() const noexcept { return loaded_; }

    void release() {
        for (auto& kv : f32_) cudaFree(kv.second);
        for (auto& kv : bf16_) cudaFree(kv.second);
        f32_.clear();
        bf16_.clear();
        if (cublas_) { cublasDestroy(cublas_); cublas_ = nullptr; }
        if (st_) { cudaStreamDestroy(st_); st_ = nullptr; }
        loaded_ = false;
    }

    bool load(const std::string& path, int device, std::string& err) {
        if (loaded_) return true;
        device_ = device;
        const char* e = std::getenv("STRATA_INKLING_VIT_BF16");
        bf16_weights_ = e && *e && std::strcmp(e, "0") != 0;
        if (!ok(cudaSetDevice(device_), "cudaSetDevice", err)) return false;
        CpuVisionEncoder host;
        if (!host.load(path, err)) return false;
        info_ = host.info;
        if (!ok(cudaStreamCreateWithFlags(&st_, cudaStreamNonBlocking), "vit stream", err)) return false;
        if (cublasCreate(&cublas_) != CUBLAS_STATUS_SUCCESS || cublasSetStream(cublas_, st_) != CUBLAS_STATUS_SUCCESS) {
            err = "vit: cublasCreate failed";
            release();
            return false;
        }
        for (const auto& kv : host.weights()) {
            const auto it = info_.T.find(kv.first);
            const bool matrix = it != info_.T.end() && it->second.shape.size() >= 2 &&
                                kv.first.size() > 7 && kv.first.compare(kv.first.size() - 7, 7, ".weight") == 0;
            const std::vector<float>& w = kv.second;
            if (matrix && bf16_weights_) {
                std::vector<uint16_t> hb(w.size());
                for (size_t i = 0; i < w.size(); ++i) hb[i] = f32_to_bf16_rne(w[i]);
                void* d = nullptr;
                if (!ok(cudaMalloc(&d, hb.size() * 2), "vit weight", err) ||
                    !ok(cudaMemcpy(d, hb.data(), hb.size() * 2, cudaMemcpyHostToDevice), "vit weight H2D", err)) {
                    if (d) cudaFree(d);
                    release();
                    return false;
                }
                bf16_[kv.first] = static_cast<__nv_bfloat16*>(d);
            } else {
                void* d = nullptr;
                if (!ok(cudaMalloc(&d, w.size() * 4), "vit vector", err) ||
                    !ok(cudaMemcpy(d, w.data(), w.size() * 4, cudaMemcpyHostToDevice), "vit vector H2D", err)) {
                    if (d) cudaFree(d);
                    release();
                    return false;
                }
                f32_[kv.first] = static_cast<float*>(d);
            }
        }
        // Host copies the per-picture host work reads (position embedding, CLS)
        auto keep = [&](const char* name, std::vector<float>& dst) {
            auto it = host.weights().find(name);
            if (it != host.weights().end()) dst = it->second;
        };
        keep("model.visual.backbone.embeddings.class_embedding", h_cls_);
        keep("model.visual.backbone.embeddings.cls_position_embedding", h_cls_pos_);
        keep("model.visual.backbone.embeddings.position_embedding", h_pos_emb_);
        const VitCfg& c = info_.c;
        if (h_cls_.size() < static_cast<size_t>(c.hidden) || h_cls_pos_.size() < static_cast<size_t>(c.hidden) ||
            h_pos_emb_.size() < static_cast<size_t>(c.posemb_grid) * c.posemb_grid * c.hidden) {
            err = "vit: embedding tensors missing from the Vision shard";
            release();
            return false;
        }
        loaded_ = true;
        return true;
    }

    // One prepared picture -> [grid_h * grid_w, out_hidden] FP32 embeddings, as CpuVisionEncoder::encode.
    bool encode(const ImageRec& img, std::vector<float>& out, std::string& err) {
        if (!loaded_) { err = "vit: not loaded"; return false; }
        if (!ok(cudaSetDevice(device_), "cudaSetDevice", err)) return false;
        const VitCfg& vc = info_.c;
        const int merge = vc.merge, patch = vc.patch, cell = patch * merge;
        const int n_tok = img.num_lm_tokens();
        const int ph = img.grid_h * merge, pw = img.grid_w * merge;
        const int n_patches = ph * pw;
        const int S = n_patches + 1;   // CLS at 0
        const int D = vc.hidden, H = vc.heads, hd = vc.head_dim;
        const int patch_in = 3 * patch * patch;
        const int merged_dim = merge * merge * D;

        // 1. Patchify, 2D positions, interpolated position embedding and RoPE tables on the host (as ink_vision.h)
        std::vector<float> patch_buf(static_cast<size_t>(n_patches) * patch_in);
        std::vector<int> pos_h(S, 0), pos_w(S, 0);
        int pi = 0;
        for (int gr = 0; gr < img.grid_h; ++gr) {
            for (int gc = 0; gc < img.grid_w; ++gc) {
                const size_t tok = static_cast<size_t>(gr) * img.grid_w + gc;
                const float* cell_px = img.pixel_values.data() + tok * cell * cell * 3;
                for (int mr = 0; mr < merge; ++mr) {
                    for (int mc = 0; mc < merge; ++mc) {
                        float* dst = patch_buf.data() + static_cast<size_t>(pi) * patch_in;
                        int idx = 0;
                        for (int py = 0; py < patch; ++py) {
                            const int ly = mr * patch + py;
                            for (int px = 0; px < patch; ++px) {
                                const int lx = mc * patch + px;
                                const float* rgb = cell_px + (static_cast<size_t>(ly) * cell + lx) * 3;
                                dst[idx++] = rgb[0]; dst[idx++] = rgb[1]; dst[idx++] = rgb[2];
                            }
                        }
                        pos_h[pi + 1] = gr * merge + mr + 1;
                        pos_w[pi + 1] = gc * merge + mc + 1;
                        ++pi;
                    }
                }
            }
        }
        std::vector<float> h_pos_add(static_cast<size_t>(S) * D, 0.0f);
        for (int d = 0; d < D; ++d) h_pos_add[d] = h_cls_[d] + h_cls_pos_[d];
        const int G = vc.posemb_grid;
        for (int p = 0; p < n_patches; ++p) {
            const float y = (static_cast<float>(pos_h[p + 1] - 1) + 0.5f) * (static_cast<float>(G) / ph) - 0.5f;
            const float x = (static_cast<float>(pos_w[p + 1] - 1) + 0.5f) * (static_cast<float>(G) / pw) - 0.5f;
            const int y0 = std::max(0, std::min(G - 1, static_cast<int>(std::floor(y))));
            const int y1 = std::min(G - 1, y0 + 1);
            const int x0 = std::max(0, std::min(G - 1, static_cast<int>(std::floor(x))));
            const int x1 = std::min(G - 1, x0 + 1);
            const float wy = std::max(0.0f, std::min(1.0f, y - y0));
            const float wx = std::max(0.0f, std::min(1.0f, x - x0));
            const float w00 = (1.0f - wy) * (1.0f - wx), w01 = (1.0f - wy) * wx;
            const float w10 = wy * (1.0f - wx), w11 = wy * wx;
            const float* p00 = h_pos_emb_.data() + (static_cast<size_t>(y0) * G + x0) * D;
            const float* p01 = h_pos_emb_.data() + (static_cast<size_t>(y0) * G + x1) * D;
            const float* p10 = h_pos_emb_.data() + (static_cast<size_t>(y1) * G + x0) * D;
            const float* p11 = h_pos_emb_.data() + (static_cast<size_t>(y1) * G + x1) * D;
            float* dst = h_pos_add.data() + static_cast<size_t>(p + 1) * D;
            for (int d = 0; d < D; ++d) dst[d] = w00 * p00[d] + w01 * p01[d] + w10 * p10[d] + w11 * p11[d];
        }
        const int sp_dim = hd / 2, n_freq = sp_dim / 2;
        std::vector<float> inv_freq(n_freq);
        for (int i = 0; i < n_freq; ++i) inv_freq[i] = 1.0f / std::pow(vc.rope_theta, static_cast<float>(2 * i) / sp_dim);
        std::vector<float> h_rc(static_cast<size_t>(S) * hd), h_rs(static_cast<size_t>(S) * hd);
        for (int t = 0; t < S; ++t) {
            for (int i = 0; i < n_freq; ++i) {
                const float aw = static_cast<float>(pos_w[t]) * inv_freq[i];
                const float ah = static_cast<float>(pos_h[t]) * inv_freq[i];
                h_rc[t * hd + 2 * i] = h_rc[t * hd + 2 * i + 1] = std::cos(aw);
                h_rs[t * hd + 2 * i] = h_rs[t * hd + 2 * i + 1] = std::sin(aw);
                h_rc[t * hd + sp_dim + 2 * i] = h_rc[t * hd + sp_dim + 2 * i + 1] = std::cos(ah);
                h_rs[t * hd + sp_dim + 2 * i] = h_rs[t * hd + sp_dim + 2 * i + 1] = std::sin(ah);
            }
        }

        // 2. Per-picture buffers
        const int rows_max = std::max(S, n_tok);
        const int cols_max = std::max({D, vc.inter, merged_dim, vc.proj_hidden, patch_in});
        // queries per attention block: the score buffer [qb, S] stays near 64 MB
        const int qb = std::max(1, std::min(S, static_cast<int>((64ull << 20) / (4ull * S))));
        std::vector<void*> bufs;
        auto dalloc = [&](size_t bytes) -> void* {
            void* p = nullptr;
            if (cudaMalloc(&p, std::max<size_t>(bytes, 4)) != cudaSuccess) return nullptr;
            bufs.push_back(p);
            return p;
        };
        auto free_all = [&]() { for (void* p : bufs) cudaFree(p); bufs.clear(); };
        float* d_patch = static_cast<float*>(dalloc(patch_buf.size() * 4));
        float* d_pos = static_cast<float*>(dalloc(h_pos_add.size() * 4));
        float* d_rc = static_cast<float*>(dalloc(h_rc.size() * 4));
        float* d_rs = static_cast<float*>(dalloc(h_rs.size() * 4));
        float* cur = static_cast<float*>(dalloc(static_cast<size_t>(S) * D * 4));
        float* nrm = static_cast<float*>(dalloc(static_cast<size_t>(S) * D * 4));
        float* q = static_cast<float*>(dalloc(static_cast<size_t>(S) * D * 4));
        float* k = static_cast<float*>(dalloc(static_cast<size_t>(S) * D * 4));
        float* v = static_cast<float*>(dalloc(static_cast<size_t>(S) * D * 4));
        float* ao = static_cast<float*>(dalloc(static_cast<size_t>(S) * D * 4));
        float* proj = static_cast<float*>(dalloc(static_cast<size_t>(S) * D * 4));
        float* mlp = static_cast<float*>(dalloc(static_cast<size_t>(S) * vc.inter * 4));
        float* mid = static_cast<float*>(dalloc(static_cast<size_t>(n_tok) * vc.proj_hidden * 4));
        float* dout = static_cast<float*>(dalloc(static_cast<size_t>(n_tok) * vc.out_hidden * 4));
        float* sc = static_cast<float*>(dalloc(static_cast<size_t>(qb) * S * 4));
        __nv_bfloat16* xbf = static_cast<__nv_bfloat16*>(dalloc(static_cast<size_t>(rows_max) * cols_max * 2));
        if (bufs.size() != 16) {   // dalloc keeps only what it got
            err = "vit: cudaMalloc failed for a picture of " + std::to_string(S) + " patches";
            free_all();
            return false;
        }
        if (!ok(cudaMemcpyAsync(d_patch, patch_buf.data(), patch_buf.size() * 4, cudaMemcpyHostToDevice, st_), "vit H2D", err) ||
            !ok(cudaMemcpyAsync(d_pos, h_pos_add.data(), h_pos_add.size() * 4, cudaMemcpyHostToDevice, st_), "vit H2D", err) ||
            !ok(cudaMemcpyAsync(d_rc, h_rc.data(), h_rc.size() * 4, cudaMemcpyHostToDevice, st_), "vit H2D", err) ||
            !ok(cudaMemcpyAsync(d_rs, h_rs.data(), h_rs.size() * 4, cudaMemcpyHostToDevice, st_), "vit H2D", err)) {
            free_all();
            return false;
        }

        bool good = true;
        // y[M, N] = x[M, K] . W[N, K]^T (+ b)
        auto linear = [&](const std::string& wname, const float* x, const char* bname, float* y, int M, int N, int K) {
            if (!good) return;
            const float one = 1.0f, zero = 0.0f;
            auto wit = bf16_.find(wname);
            if (wit != bf16_.end()) {   // STRATA_INKLING_VIT_BF16: BF16 x BF16 -> FP32
                const size_t n = static_cast<size_t>(M) * K;
                k_f32_to_bf16<<<static_cast<unsigned>((n + 255) / 256), 256, 0, st_>>>(x, xbf, n);
                if (cublasGemmEx(cublas_, CUBLAS_OP_T, CUBLAS_OP_N, N, M, K, &one, wit->second, CUDA_R_16BF, K, xbf,
                                 CUDA_R_16BF, K, &zero, y, CUDA_R_32F, N, CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT) !=
                    CUBLAS_STATUS_SUCCESS) {
                    err = "vit: cublasGemmEx failed for " + wname;
                    good = false;
                    return;
                }
            } else {                    // FP32, as the CPU reference
                const float* w = f(wname);
                if (!w) { err = "vit: missing " + wname; good = false; return; }
                if (cublasSgemm(cublas_, CUBLAS_OP_T, CUBLAS_OP_N, N, M, K, &one, w, K, x, K, &zero, y, N) !=
                    CUBLAS_STATUS_SUCCESS) {
                    err = "vit: cublasSgemm failed for " + wname;
                    good = false;
                    return;
                }
            }
            if (bname) {
                const float* b = f(bname);
                if (!b) { err = std::string("vit: missing ") + bname; good = false; return; }
                const size_t m = static_cast<size_t>(M) * N;
                k_add_bias_rows<<<static_cast<unsigned>((m + 255) / 256), 256, 0, st_>>>(y, b, M, N);
            }
        };
        auto grid = [](size_t n) { return static_cast<unsigned>((n + 255) / 256); };
        const size_t SD = static_cast<size_t>(S) * D;

        cudaMemsetAsync(cur, 0, static_cast<size_t>(D) * 4, st_);
        linear("model.visual.backbone.embeddings.patch_embedding.weight", d_patch, nullptr, cur + D, n_patches, D, patch_in);
        k_add<<<grid(SD), 256, 0, st_>>>(cur, d_pos, SD);
        k_layernorm<<<S, 256, 0, st_>>>(cur, f("model.visual.backbone.ln_pre.weight"), f("model.visual.backbone.ln_pre.bias"),
                                         cur, D, vc.ln_eps);
        const float attn_scale = 1.0f / std::sqrt(static_cast<float>(hd));
        for (int il = 0; il < vc.layers && good; ++il) {
            const std::string p = "model.visual.backbone.layers." + std::to_string(il);
            k_layernorm<<<S, 256, 0, st_>>>(cur, f(p + ".ln_1.weight"), f(p + ".ln_1.bias"), nrm, D, vc.ln_eps);
            linear(p + ".self_attn.q_proj.weight", nrm, (p + ".self_attn.q_proj.bias").c_str(), q, S, D, D);
            linear(p + ".self_attn.k_proj.weight", nrm, (p + ".self_attn.k_proj.bias").c_str(), k, S, D, D);
            linear(p + ".self_attn.v_proj.weight", nrm, (p + ".self_attn.v_proj.bias").c_str(), v, S, D, D);
            k_rope_2d<<<dim3(H, S), hd / 2, 0, st_>>>(q, k, d_rc, d_rs, H, hd);
            // Full (non-causal) attention per head, over blocks of qb queries
            for (int h = 0; h < H && good; ++h) {
                for (int q0 = 0; q0 < S && good; q0 += qb) {
                    const int nq = std::min(qb, S - q0);
                    const float one = 1.0f, zero = 0.0f;
                    // scores[nq, S] (row-major) = Q_h[q0:q0+nq] . K_h^T
                    if (cublasSgemm(cublas_, CUBLAS_OP_T, CUBLAS_OP_N, S, nq, hd, &one, k + static_cast<size_t>(h) * hd, D,
                                    q + static_cast<size_t>(q0) * D + static_cast<size_t>(h) * hd, D, &zero, sc, S) !=
                        CUBLAS_STATUS_SUCCESS) { err = "vit: attention scores failed"; good = false; break; }
                    k_softmax_rows<<<nq, 256, 0, st_>>>(sc, S, attn_scale);
                    // O_h[q0:q0+nq] = P . V_h
                    if (cublasSgemm(cublas_, CUBLAS_OP_N, CUBLAS_OP_N, hd, nq, S, &one, v + static_cast<size_t>(h) * hd, D,
                                    sc, S, &zero, ao + static_cast<size_t>(q0) * D + static_cast<size_t>(h) * hd, D) !=
                        CUBLAS_STATUS_SUCCESS) { err = "vit: attention output failed"; good = false; break; }
                }
            }
            linear(p + ".self_attn.out_proj.weight", ao, (p + ".self_attn.out_proj.bias").c_str(), proj, S, D, D);
            k_layerscale_res<<<grid(SD), 256, 0, st_>>>(cur, proj, f(p + ".ls_1.gamma"), S, D);
            k_layernorm<<<S, 256, 0, st_>>>(cur, f(p + ".ln_2.weight"), f(p + ".ln_2.bias"), nrm, D, vc.ln_eps);
            linear(p + ".mlp.c_fc.weight", nrm, (p + ".mlp.c_fc.bias").c_str(), mlp, S, vc.inter, D);
            k_gelu_exact<<<grid(static_cast<size_t>(S) * vc.inter), 256, 0, st_>>>(mlp, static_cast<size_t>(S) * vc.inter);
            linear(p + ".mlp.c_proj.weight", mlp, (p + ".mlp.c_proj.bias").c_str(), proj, S, D, vc.inter);
            k_layerscale_res<<<grid(SD), 256, 0, st_>>>(cur, proj, f(p + ".ls_2.gamma"), S, D);
        }
        // 3. Drop CLS, 2x2 merge projector: [n_tok, 4D] -> proj_in -> GELU -> proj_out
        linear("model.visual.proj_in.weight", cur + D, nullptr, mid, n_tok, vc.proj_hidden, merged_dim);
        k_gelu_exact<<<grid(static_cast<size_t>(n_tok) * vc.proj_hidden), 256, 0, st_>>>(mid, static_cast<size_t>(n_tok) * vc.proj_hidden);
        linear("model.visual.proj_out.weight", mid, nullptr, dout, n_tok, vc.out_hidden, vc.proj_hidden);

        if (good) {
            out.resize(static_cast<size_t>(n_tok) * vc.out_hidden);
            good = ok(cudaMemcpyAsync(out.data(), dout, out.size() * 4, cudaMemcpyDeviceToHost, st_), "vit D2H", err) &&
                   ok(cudaStreamSynchronize(st_), "vit sync", err) && ok(cudaGetLastError(), "vit kernels", err);
        } else {
            cudaStreamSynchronize(st_);
        }
        free_all();
        return good;
    }

private:
    static bool ok(cudaError_t e, const char* what, std::string& err) {
        if (e == cudaSuccess) return true;
        err = std::string(what) + ": " + cudaGetErrorString(e);
        return false;
    }
    const float* f(const std::string& name) const {
        auto it = f32_.find(name);
        return it == f32_.end() ? nullptr : it->second;
    }

    bool loaded_ = false;
    bool bf16_weights_ = false;
    int device_ = 0;
    VisionShardInfo info_;
    cudaStream_t st_ = nullptr;
    cublasHandle_t cublas_ = nullptr;
    std::unordered_map<std::string, float*> f32_;
    std::unordered_map<std::string, __nv_bfloat16*> bf16_;
    std::vector<float> h_cls_, h_cls_pos_, h_pos_emb_;
};

}  // namespace cuvit
}  // namespace ink2
