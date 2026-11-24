// Compute the Engram injection from rows selected by the host-side hash.
#include "strata/hy4flash_gguf/engram.hpp"
#include "strata/kernels/elementwise.hpp"

#include "strata/hy4flash_gguf/mat.hpp"
#include "strata/kernels/native_mmvq.hpp"
#include "strata/kernels/pdl.hpp"

#include <stdexcept>
#include <string>

#if defined(__HIP_PLATFORM_AMD__) || defined(STRATA_USE_HIP)
#include <hip/hip_runtime.h>
#else
#include <cuda_runtime.h>
#endif

namespace strata::hy4_flash {
namespace {

using strata::kernels::launch_pdl;
using strata::kernels::pdl_trigger;
using strata::kernels::pdl_wait;

constexpr int kBlock = 256;
/// `engram_gate_k` alone: its dot is reduced over the whole block, so the thread count is arithmetic.
constexpr int kEngramGateThreads = 512;
/// FLT_EPSILON, which is what build_engram uses for every norm here - NOT the model's rms_norm_eps (1e-5).
constexpr float kEngramEps = 1.1920929e-07f;

__device__ __forceinline__ float warp_sum(float v) {
#pragma unroll
    for (int m = 16; m; m >>= 1) v += __shfl_xor_sync(0xffffffffu, v, m, 32);
    return v;
}

/// One block per (column, token): rms_norm that row of `head_dim`, scale by its head's weight and by valid.
__global__ __launch_bounds__(kBlock) void rows_normed_k(const float* rows_,
                                                        const float* __restrict__ lookup_norm,
                                                        const float* valid_, int head_dim,
                                                        int columns, float* out_) {
    const float* STRATA_PDL_RESTRICT rows = rows_;   // activations (pdl.hpp); lookup_norm is a weight
    const float* STRATA_PDL_RESTRICT valid = valid_;
    float* STRATA_PDL_RESTRICT out = out_;
    pdl_trigger();
    pdl_wait();
    // `rows` arrives already rms-normalised (see engram_layer): all this does is the per-column weight and
    // the valid mask. The norm moved out because its reduction has to be ggml's, not this block's.
    const int ct = (int) blockIdx.x;
    const int c = ct % columns;
    const long long t = ct / columns;
    const float* r = rows + ((long long) t * columns + c) * head_dim;
    float* o = out + ((long long) t * columns + c) * head_dim;

    // The per-column valid is belt and braces in the reference - valid is uniform across a token's columns -
    // but it is applied because the reference applies it.
    const float vmask = valid[(long long) t * columns + c];
    for (int i = (int) threadIdx.x; i < head_dim; i += kBlock)
        o[i] = r[i] * lookup_norm[(long long) c * head_dim + i] * vmask;
}

/// kv += kv_b, broadcast over tokens: kv is [n_embd, hc+1, n_tok], kv_b is [n_embd, hc+1].
__global__ __launch_bounds__(kBlock) void add_kv_bias_k(float* kv_,
                                                        const float* __restrict__ kv_b, int n_embd, int slices,
                                                        int n_tok) {
    float* STRATA_PDL_RESTRICT kv = kv_;   // kv_b is a weight
    pdl_trigger();
    pdl_wait();
    const long long n = (long long) n_embd * slices * n_tok;
    for (long long i = blockIdx.x * (long long) blockDim.x + threadIdx.x; i < n;
         i += (long long) gridDim.x * blockDim.x)
        kv[i] += kv_b[i % ((long long) n_embd * slices)];
}

/// Reduce the gated dot product per residual stream and token.
__global__ __launch_bounds__(kEngramGateThreads) void engram_gate_k(const float* x_,
                                                        const float* kv_,
                                                        const float* __restrict__ gate, int n_embd, int hc,
                                                        int n_tok, float* out_gate_,
                                                        const float* norm_x_,
                                                        const float* norm_key_) {
    const float* STRATA_PDL_RESTRICT x = x_;   // `gate` is the weight; the rest are activations (pdl.hpp)
    const float* STRATA_PDL_RESTRICT kv = kv_;
    float* STRATA_PDL_RESTRICT out_gate = out_gate_;
    const float* STRATA_PDL_RESTRICT norm_x = norm_x_;
    const float* STRATA_PDL_RESTRICT norm_key = norm_key_;
    pdl_trigger();
    pdl_wait();
    const int ht = (int) blockIdx.x;
    const int h = ht % hc;
    const long long t = ht / hc;
    // Both already rms-normalised by the shared primitive, so `ix` and `ik` below are 1: the norms moved
    // out because their reduction has to be ggml's. `x` and `kv` stay as parameters for the layout.
    const float* xr = (norm_x != nullptr ? norm_x : x) + ((long long) t * hc + h) * n_embd;
    // key is slices 0..hc-1 of kv, which is [n_embd, hc+1, n_tok].
    const float* kr = norm_key != nullptr ? norm_key + ((long long) t * hc + h) * n_embd
                                          : kv + ((long long) t * (hc + 1) + h) * n_embd;
    const float* gr = gate + (long long) h * n_embd;

    float sx = 0.f, sk = 0.f;
    for (int d = (int) threadIdx.x; d < n_embd; d += kEngramGateThreads) {
        sx += xr[d] * xr[d];
        sk += kr[d] * kr[d];
    }
    sx = warp_sum(sx);
    sk = warp_sum(sk);
    __shared__ float px[kEngramGateThreads / 32], pk[kEngramGateThreads / 32];
    if ((threadIdx.x & 31) == 0) {
        px[threadIdx.x / 32] = sx;
        pk[threadIdx.x / 32] = sk;
    }
    __syncthreads();
    __shared__ float ix, ik;
    if (threadIdx.x < 32) {
        float a = threadIdx.x < kEngramGateThreads / 32 ? px[threadIdx.x] : 0.f;
        float b = threadIdx.x < kEngramGateThreads / 32 ? pk[threadIdx.x] : 0.f;
        a = warp_sum(a);
        b = warp_sum(b);
        if (threadIdx.x == 0) {
            ix = norm_x != nullptr ? 1.f : rsqrtf(a / (float) n_embd + kEngramEps);
            ik = norm_key != nullptr ? 1.f : rsqrtf(b / (float) n_embd + kEngramEps);
        }
    }
    __syncthreads();

    float acc = 0.f;
    // Each product rounded on its own and then added, rather than contracted into an FMA.
    for (int d = (int) threadIdx.x; d < n_embd; d += kEngramGateThreads)
        acc = __fadd_rn(acc, __fmul_rn(__fmul_rn(xr[d] * ix, kr[d] * ik), gr[d]));
    acc = warp_sum(acc);
    __shared__ float pd[kEngramGateThreads / 32];
    if ((threadIdx.x & 31) == 0) pd[threadIdx.x / 32] = acc;
    __syncthreads();
    if (threadIdx.x < 32) {
        float v = threadIdx.x < kEngramGateThreads / 32 ? pd[threadIdx.x] : 0.f;
        v = warp_sum(v);
        if (threadIdx.x == 0) {
            // A reciprocal and a multiply, not a divide; and the APPROXIMATE square root, both as c50.
            const float dot = __fmul_rn(v, 1.f / sqrtf((float) n_embd));
            // sigmoid(copysign(sqrt(max(|dot|, 1e-6)), dot)), with sign(0) = +1 as copysign gives.
            float mag;
            const float arg = fmaxf(fabsf(dot), 1e-6f);
            asm("sqrt.approx.ftz.f32 %0, %1;" : "=f"(mag) : "f"(arg));
            const float sgn = dot < 0.f ? -1.f : 1.f;
            out_gate[(long long) t * hc + h] = __fdividef(1.f, 1.f + __expf(-(mag * sgn)));
        }
    }
}

/// Add the normalized value to each residual stream only when its lookup is valid.
__global__ __launch_bounds__(kBlock) void engram_delta_k(const float* x_,
                                                         const float* kv_,
                                                         const float* __restrict__ pre_gamma,
                                                         const float* __restrict__ hc_scale,
                                                         const float* gate_,
                                                         const float* valid_, int n_embd, int hc,
                                                         int columns, int n_tok, float* out_,
                                                         const float* norm_value_) {
    const float* STRATA_PDL_RESTRICT x = x_;   // pre_gamma and hc_scale are weights
    const float* STRATA_PDL_RESTRICT kv = kv_;
    const float* STRATA_PDL_RESTRICT gate = gate_;
    const float* STRATA_PDL_RESTRICT valid = valid_;
    float* STRATA_PDL_RESTRICT out = out_;
    const float* STRATA_PDL_RESTRICT norm_value = norm_value_;
    pdl_trigger();
    pdl_wait();
    const long long t = blockIdx.x;
    if (t >= n_tok) return;
    // value is the LAST of the hc+1 slices.
    const float* vr = norm_value != nullptr ? norm_value + t * (long long) n_embd
                                            : kv + ((long long) t * (hc + 1) + hc) * n_embd;

    float sv = 0.f;
    for (int d = (int) threadIdx.x; d < n_embd; d += kBlock) sv += vr[d] * vr[d];
    sv = warp_sum(sv);
    __shared__ float pv[kBlock / 32];
    if ((threadIdx.x & 31) == 0) pv[threadIdx.x / 32] = sv;
    __syncthreads();
    __shared__ float iv, any;
    if (threadIdx.x < 32) {
        float a = threadIdx.x < kBlock / 32 ? pv[threadIdx.x] : 0.f;
        a = warp_sum(a);
        if (threadIdx.x == 0) iv = norm_value != nullptr ? 1.f : rsqrtf(a / (float) n_embd + kEngramEps);
    }
    if (threadIdx.x == 0) {
        // A token with NO valid column contributes nothing at all. Zeroing the columns is not enough: kv_b
        // would still push a constant through the value path. First token of a sequence, or an image.
        float s = 0.f;
        for (int c = 0; c < columns; ++c) s += valid[t * columns + c];
        any = s > 0.f ? 1.f : 0.f;
    }
    __syncthreads();

    for (int h = 0; h < hc; ++h) {
        const float g = gate[t * hc + h];
        const float* hs = hc_scale + (long long) h * n_embd;
        const long long base = (t * hc + h) * (long long) n_embd;
        for (int d = (int) threadIdx.x; d < n_embd; d += kBlock) {
            const float signal = vr[d] * iv * pre_gamma[d];
            // hc_scale is ADDED to the gate, not multiplied by it, and is per channel and per stream.
            out[base + d] = x[base + d] + signal * (g + hs[d]) * any;
        }
    }
}

unsigned grid_for(long long n) {
    const long long g = (n + kBlock - 1) / kBlock;
    return (unsigned) (g < 1 ? 1 : (g > 65535 ? 65535 : g));
}

}  // namespace

std::size_t engram_layer_scratch_bytes(int n_embd, int hc, int head_dim, int columns, int n_tok) {
    if (n_tok <= 0) return 0;
    const std::size_t f = sizeof(float);
    std::size_t b = 0;
    b += (std::size_t) head_dim * columns * n_tok * f;       // the normed rows
    b += (std::size_t) n_embd * (hc + 1) * n_tok * f;        // the fused key/value projection
    b += (std::size_t) hc * n_tok * f;                        // the gate
    b += (std::size_t) n_embd * hc * n_tok * f;               // the normed x
    b += (std::size_t) n_embd * hc * n_tok * f;               // the normed key
    b += (std::size_t) n_embd * n_tok * f;                    // the normed value
    b += kernels::native_q8_1_bytes_wide(head_dim * columns, n_tok);
    return b + 256;
}

void engram_layer(const float* x, const float* rows, const float* valid, const EngramLayerWeights& w,
                  int n_embd, int hc, int head_dim, int columns, int n_tok, float* out, void* scratch,
                  void* stream, MatCtx* mc) {
    if (x == nullptr || rows == nullptr || valid == nullptr || out == nullptr || scratch == nullptr ||
        stream == nullptr || w.kv == nullptr || w.kv_b == nullptr || w.gate == nullptr ||
        w.lookup_norm == nullptr || w.pre_gamma == nullptr || w.hc_scale == nullptr)
        throw std::invalid_argument("engram_layer: a stream and every weight");
    if (n_embd <= 0 || hc <= 0 || head_dim <= 0 || columns <= 0 || n_tok <= 0)
        throw std::invalid_argument("engram_layer: positive extents");

    char* p = static_cast<char*>(scratch);
    auto take = [&p](std::size_t bytes) {
        void* at = p;
        p += (bytes + 15) & ~(std::size_t) 15;
        return at;
    };
    const std::size_t f = sizeof(float);
    float* r = (float*) take((std::size_t) head_dim * columns * n_tok * f);
    float* kv = (float*) take((std::size_t) n_embd * (hc + 1) * n_tok * f);
    float* gate = (float*) take((std::size_t) hc * n_tok * f);
    void* q8 = take(kernels::native_q8_1_bytes_wide(head_dim * columns, n_tok));
    float* nx = (float*) take((std::size_t) n_embd * hc * n_tok * f);
    float* nk = (float*) take((std::size_t) n_embd * hc * n_tok * f);
    float* nv = (float*) take((std::size_t) n_embd * n_tok * f);

    // Use the same CUDA RMS reduction order for all four streams.
    // The cost is four extra launches an Engram layer, against four layers a token.
    kernels::rms_norm_block(rows, nullptr, r, (long long) columns * n_tok, head_dim, kEngramEps, stream);
    launch_pdl(rows_normed_k, (unsigned) (columns * n_tok), kBlock, 0, static_cast<cudaStream_t>(stream),
               r, w.lookup_norm, valid, head_dim, columns, r);

    // One fused key/value projection of the flattened lookup, then its bias.
    const MatIn look = mat_stage(mc, r, q8, head_dim * columns, n_tok, stream, kSiteEngramKV);
    mat_apply(look, w.kv_type, w.kv, kv, head_dim * columns, n_embd * (hc + 1), n_tok, stream,
              kSiteEngramKV);
    launch_pdl(add_kv_bias_k, grid_for((long long) n_embd * (hc + 1) * n_tok), kBlock, 0,
               static_cast<cudaStream_t>(stream), kv, w.kv_b, n_embd, hc + 1, n_tok);

    // x is [n_embd, hc, n_tok] and its rows are contiguous, so one call covers them. The key and the
    // value are slices of kv, which is [n_embd, hc + 1, n_tok]: a token's hc key rows ARE contiguous but
    // the next token's are a slice away, so that one is a call a token (n_tok is 1 decoding and at most 8).
    kernels::rms_norm_block(x, nullptr, nx, (long long) hc * n_tok, n_embd, kEngramEps, stream);
    for (int t = 0; t < n_tok; ++t) {
        kernels::rms_norm_block(kv + (std::size_t) t * (hc + 1) * n_embd, nullptr,
                                nk + (std::size_t) t * hc * n_embd, hc, n_embd, kEngramEps, stream);
        kernels::rms_norm_block(kv + ((std::size_t) t * (hc + 1) + hc) * n_embd, nullptr,
                                nv + (std::size_t) t * n_embd, 1, n_embd, kEngramEps, stream);
    }
    launch_pdl(engram_gate_k, (unsigned) (hc * n_tok), kEngramGateThreads, 0,
               static_cast<cudaStream_t>(stream), x, kv, w.gate, n_embd, hc, n_tok, gate, nx, nk);
    launch_pdl(engram_delta_k, (unsigned) n_tok, kBlock, 0, static_cast<cudaStream_t>(stream),
               x, kv, w.pre_gamma, w.hc_scale, gate, valid, n_embd, hc, columns, n_tok, out, nv);

    const auto e = cudaGetLastError();
    if (e != cudaSuccess) throw std::runtime_error(std::string("engram_layer: ") + cudaGetErrorString(e));
}

}  // namespace strata::hy4_flash
