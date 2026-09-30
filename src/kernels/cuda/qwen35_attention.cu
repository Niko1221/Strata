#include "strata/kernels/qwen35_attention.hpp"

#include "strata/kernels/f16_bits.hpp"
#include "strata/kernels/rope.hpp"

#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <stdexcept>
#include <string>
#include <vector>

namespace strata::kernels {
namespace {

constexpr int QWEN35_N_ROT = 64;
constexpr double QWEN35_ROPE_THETA = 10000000.0;
constexpr int THREADS = 256;

__device__ __forceinline__ float warp_sum(float x) {
    for (int off = 16; off; off >>= 1)
        x += __shfl_down_sync(0xffffffffu, x, off);
    return x;
}

__global__ void kv_append_kernel(const float * k, const float * v,
                                 uint16_t * cache_k, uint16_t * cache_v,
                                 int64_t pos, int kv_heads, int head_dim) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    const int n = kv_heads * head_dim;
    if (i >= n) return;
    const int base = (int) (pos * (int64_t)n);
    cache_k[base + i] = f16_from_f32(k[i]);
    cache_v[base + i] = f16_from_f32(v[i]);
}

/// One block per query head. Every warp computes a partial dot for every cached key;
/// warp 0 reduces the eight partials. This is deliberately simple: it is a correctness
/// baseline for the dense model, while the existing Strata QSA kernel remains untouched.
__global__ void score_kernel(const float * q, const uint16_t * cache_k,
                             int64_t n_kv, int max_context,
                             int n_head_kv, int head_dim, float scale,
                             float * scores) {
    const int h = blockIdx.x;
    const int tid = threadIdx.x;
    const int d = tid;
    const int kvh = h / (gridDim.x / n_head_kv);

    __shared__ float partial[8];
    for (int64_t t = 0; t < n_kv; ++t) {
        float x = 0.0f;
        if (d < head_dim) {
            const int qoff = h * head_dim + d;
            const int koff = (int) (t * (int64_t)n_head_kv * head_dim + kvh * head_dim + d);
            x = q[qoff] * f32_from_f16(cache_k[koff]);
        }
        x = warp_sum(x);
        if ((tid & 31) == 0) partial[tid >> 5] = x;
        __syncthreads();
        float dot = 0.0f;
        if (tid < 8) dot = partial[tid];
        dot = warp_sum(dot);
        if (tid == 0)
            scores[h * max_context + t] = dot * scale;
        __syncthreads();
    }
}

/// One block per query head, one thread per output dimension. Softmax is evaluated
/// in FP32 over the complete causal row; K/V are stored as FP16.
__global__ void value_kernel(const float * scores, const uint16_t * cache_v,
                             int64_t n_kv, int max_context,
                             int n_head, int n_head_kv, int head_dim,
                             const float * gate, float * out) {
    const int h = blockIdx.x;
    const int d = threadIdx.x;
    const int kvh = h / (n_head / n_head_kv);
    __shared__ float smem_max;
    __shared__ float smem_sum;

    if (d == 0) {
        float mx = -CUDART_INF_F;
        for (int64_t t = 0; t < n_kv; ++t)
            mx = fmaxf(mx, scores[h * max_context + t]);
        float sum = 0.0f;
        for (int64_t t = 0; t < n_kv; ++t)
            sum += expf(scores[h * max_context + t] - mx);
        smem_max = mx;
        smem_sum = sum;
    }
    __syncthreads();

    float acc = 0.0f;
    for (int64_t t = 0; t < n_kv; ++t) {
        const float p = expf(scores[h * max_context + t] - smem_max) / smem_sum;
        const int voff = (int) (t * (int64_t)n_head_kv * head_dim + kvh * head_dim + d);
        acc += p * f32_from_f16(cache_v[voff]);
    }
    const float g = 1.0f / (1.0f + expf(-gate[h * head_dim + d]));
    out[h * head_dim + d] = acc * g;
}




__global__ void qwen35_gdn_norm_kernel(const float * o, const float * z, const float * w,
                                       float * y, int head_dim, float eps) {
    const int h = blockIdx.x;
    const int lane = threadIdx.x;
    const float * po = o + (int64_t)h * head_dim;
    const float * pz = z + (int64_t)h * head_dim;
    float * py = y + (int64_t)h * head_dim;
    float acc = 0.0f;
    for (int i = lane; i < head_dim; i += 32) acc += po[i] * po[i];
    for (int off = 16; off; off >>= 1) acc += __shfl_down_sync(0xffffffffu, acc, off);
    acc = __shfl_sync(0xffffffffu, acc, 0);
    const float inv = rsqrtf(acc / (float)head_dim + eps);
    for (int i = lane; i < head_dim; i += 32) {
        const float zz = pz[i];
        const float silu = zz / (1.0f + expf(-zz));
        py[i] = po[i] * inv * w[i] * silu;
    }
}

__global__ void rms_norm_qwen35_kernel(float * x, const float * w,
                                       int64_t rows, int64_t cols, float eps) {
    const int lane = threadIdx.x & 31;
    const int64_t row = (int64_t)blockIdx.x * (blockDim.x >> 5) + (threadIdx.x >> 5);
    if (row >= rows) return;
    float * r = x + row * cols;
    float acc = 0.0f;
    for (int64_t c = lane; c < cols; c += 32)
        acc += r[c] * r[c];
    for (int off = 16; off; off >>= 1)
        acc += __shfl_down_sync(0xffffffffu, acc, off);
    float inv = 0.0f;
    if (lane == 0)
        inv = rsqrtf(acc / (float)cols + eps);
    inv = __shfl_sync(0xffffffffu, inv, 0);
    for (int64_t c = lane; c < cols; c += 32)
        r[c] *= (1.0f + w[c]) * inv;
}

__global__ void silu_mul_kernel(const float * gate, const float * up, float * out, int64_t n) {
    const int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const float x = gate[i];
    const float silu = x / (1.0f + expf(-x));
    out[i] = silu * up[i];
}

void check(cudaError_t e, const char * what) {
    if (e != cudaSuccess)
        throw std::runtime_error(std::string("qwen35_attention: ") + what + ": " + cudaGetErrorString(e));
}

} // namespace

void qwen35_rope_init(int64_t max_context, float ** cos_dev, float ** sin_dev, int ** pos_dev) {
    if (max_context <= 0)
        throw std::invalid_argument("qwen35_rope_init: max_context must be positive");
    const size_t tab_bytes = (size_t)max_context * (QWEN35_N_ROT / 2) * sizeof(float);
    check(cudaMalloc((void **)cos_dev, tab_bytes), "cudaMalloc(cos)");
    check(cudaMalloc((void **)sin_dev, tab_bytes), "cudaMalloc(sin)");
    check(cudaMalloc((void **)pos_dev, sizeof(int)), "cudaMalloc(pos)");

    std::vector<float> hcos((size_t)max_context * (QWEN35_N_ROT / 2));
    std::vector<float> hsin((size_t)max_context * (QWEN35_N_ROT / 2));
    for (int64_t p = 0; p < max_context; ++p) {
        for (int i = 0; i < QWEN35_N_ROT / 2; ++i) {
            const double inv = std::pow(QWEN35_ROPE_THETA,
                                        -2.0 * (double)i / (double)QWEN35_N_ROT);
            const double a = (double)p * inv;
            hcos[(size_t)p * (QWEN35_N_ROT / 2) + i] = (float)std::cos(a);
            hsin[(size_t)p * (QWEN35_N_ROT / 2) + i] = (float)std::sin(a);
        }
    }
    check(cudaMemcpy(*cos_dev, hcos.data(), tab_bytes, cudaMemcpyHostToDevice), "copy(cos)");
    check(cudaMemcpy(*sin_dev, hsin.data(), tab_bytes, cudaMemcpyHostToDevice), "copy(sin)");
    const int zero = 0;
    check(cudaMemcpy(*pos_dev, &zero, sizeof(zero), cudaMemcpyHostToDevice), "copy(pos)");
}

void qwen35_rope_free(float * cos_dev, float * sin_dev, int * pos_dev) {
    if (cos_dev) cudaFree(cos_dev);
    if (sin_dev) cudaFree(sin_dev);
    if (pos_dev) cudaFree(pos_dev);
}

void qwen35_rope_set_pos(int * pos_dev, int64_t pos, void * stream) {
    const int p = (int)pos;
    check(cudaMemcpyAsync(pos_dev, &p, sizeof(p), cudaMemcpyHostToDevice, (cudaStream_t)stream), "copy(pos)");
}

void qwen35_rope_apply(float * x, int64_t rows, int head_dim, const float * cos_dev,
                       const float * sin_dev, const int * pos_dev, void * stream) {
    if (rows <= 0) return;
    // Reuse Strata's validated NEOX kernel; it already implements the exact 64/256 partial rotation.
    rope_neox_apply(x, x, rows, head_dim, QWEN35_N_ROT, cos_dev, sin_dev, pos_dev, stream);
}

void qwen35_gdn_out_norm(const float * o, const float * z, const float * weight,
                         float * y, int64_t heads, int64_t head_dim, float eps, void * stream) {
    if (heads <= 0 || head_dim <= 0) return;
    qwen35_gdn_norm_kernel<<<(int)heads, 32, 0, (cudaStream_t)stream>>>(
        o, z, weight, y, (int)head_dim, eps);
    check(cudaGetLastError(), "gdn output norm");
}

void qwen35_rms_norm_weighted(float * x, const float * weight, int64_t rows, int64_t cols,
                              float eps, void * stream) {
    if (rows <= 0 || cols <= 0) return;
    const int warps = 8;
    const int blocks = (int)((rows + warps - 1) / warps);
    rms_norm_qwen35_kernel<<<blocks, warps * 32, 0, (cudaStream_t)stream>>>(
        x, weight, rows, cols, eps);
    check(cudaGetLastError(), "rms norm");
}

void qwen35_silu_mul(const float * gate, const float * up, float * out, int64_t n, void * stream) {
    if (n <= 0) return;
    silu_mul_kernel<<<(n + THREADS - 1) / THREADS, THREADS, 0, (cudaStream_t)stream>>>(gate, up, out, n);
    check(cudaGetLastError(), "silu/mul");
}

void qwen35_full_attention_step(const float * q, const float * k, const float * v,
                                uint16_t * cache_k, uint16_t * cache_v,
                                int64_t pos, int64_t max_context,
                                int n_head, int n_head_kv, int head_dim,
                                const float * gate,
                                float * scores, float * out, void * stream) {
    if (pos < 0 || pos >= max_context)
        throw std::invalid_argument("qwen35_full_attention_step: position outside KV cache");
    if (n_head <= 0 || n_head_kv <= 0 || head_dim <= 0 || n_head % n_head_kv != 0)
        throw std::invalid_argument("qwen35_full_attention_step: invalid geometry");

    const cudaStream_t st = (cudaStream_t)stream;
    const int kv_count = n_head_kv * head_dim;
    kv_append_kernel<<<(kv_count + THREADS - 1) / THREADS, THREADS, 0, st>>>(
        k, v, cache_k, cache_v, pos, n_head_kv, head_dim);
    check(cudaGetLastError(), "kv append");

    const float scale = 1.0f / std::sqrt((float)head_dim);
    score_kernel<<<n_head, THREADS, 0, st>>>(
        q, cache_k, pos + 1, (int)max_context, n_head_kv, head_dim, scale, scores);
    check(cudaGetLastError(), "scores");

    value_kernel<<<n_head, THREADS, 0, st>>>(
        scores, cache_v, pos + 1, (int)max_context,
        n_head, n_head_kv, head_dim, gate, out);
    check(cudaGetLastError(), "values");
}

} // namespace strata::kernels
