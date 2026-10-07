// GLM int4-g64 CUDA compute. FP32 cuBLAS preserves the CPU path's precision;
// weights are streamed for prefill, and hot decode experts stay in a bounded VRAM cache.
#include "strata/glm/cuda.hpp"
#include <cuda_runtime.h>
#include <math_constants.h>
#include <cublas_v2.h>
#include <algorithm>
#include <climits>
#include <stdexcept>
#include <unordered_map>
#include <vector>

namespace strata::glm {
namespace {
void ck(cudaError_t e) { if (e != cudaSuccess) throw std::runtime_error(std::string("GLM CUDA: ") + cudaGetErrorString(e)); }
void cb(cublasStatus_t s) { if (s != CUBLAS_STATUS_SUCCESS) throw std::runtime_error("GLM cuBLAS error " + std::to_string(s)); }
struct Buffer {
    uint8_t* p = nullptr; size_t cap = 0;
    ~Buffer() { if (p) cudaFree(p); }
    void reserve(size_t n) {
        if (n <= cap) return;
        uint8_t* next = nullptr; ck(cudaMalloc(&next, n));
        if (p) ck(cudaFree(p)); p = next; cap = n;
    }
    float* f() { return (float*)p; }
    void put(const void* src, size_t n) { reserve(n); ck(cudaMemcpy(p, src, n, cudaMemcpyHostToDevice)); }
};
struct DQ4 { int O, I; const uint8_t* q; const float* s; };
__device__ float weight(DQ4 w, int r, int i) {
    uint8_t b = w.q[(size_t)r * (w.I / 2) + i / 2];
    return (float)((int)((i & 1) ? b >> 4 : b & 15) - 8) * w.s[(size_t)r * (w.I / 64) + i / 64];
}
__global__ void gemv(DQ4 w, const float* x, float* y) {
    int row = blockIdx.x * 8 + threadIdx.x / 32, lane = threadIdx.x & 31;
    if (row >= w.O) return;
    float v = 0;
    for (int i = lane; i < w.I; i += 32) v = fmaf(weight(w, row, i), x[i], v);
    for (int d = 16; d; d /= 2) v += __shfl_down_sync(0xffffffffu, v, d);
    if (!lane) y[row] = v;
}
__global__ void dequant(DQ4 w, float* out) {
    size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x, n = (size_t)w.O * w.I;
    if (i < n) out[i] = weight(w, (int)(i / w.I), (int)(i % w.I));
}
__global__ void gated(float* g, const float* u, size_t n) {
    size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) g[i] = (g[i] / (1.0f + expf(-g[i]))) * u[i];
}

// Online softmax in 128-key tiles: scratch is constant, even for a long context.
// One CTA per query/head. The latent stays absorbed, so there is no H-fold expanded KV cache.
__global__ void mla(DQ4 w, const float* Q, const float* KV, int S, int pos0,
                    int H, int nope, int rope, int vh, float* out) {
    const int hd = blockIdx.x, s = blockIdx.y, tid = threadIdx.x;
    const int latent = w.I, row = latent + rope, qh = nope + rope;
    const int rb = hd * (nope + vh), nt = pos0 + s + 1;
    const float* q = Q + ((size_t)s * H + hd) * qh;
    __shared__ float qa[512], ctx[512], sc[128], red[256], mx, denom, old_scale;
    for (int i = tid; i < latent; i += 256) {
        float v = 0;
        for (int r = 0; r < nope; ++r) v = fmaf(weight(w, rb + r, i), q[r], v);
        qa[i] = v; ctx[i] = 0;
    }
    if (!tid) { mx = -CUDART_INF_F; denom = 0; }
    __syncthreads();
    for (int b = 0; b < nt; b += 128) {
        const int n = min(128, nt - b), lane = tid & 31, warp = tid / 32;
        for (int t = warp; t < n; t += 8) {
            const float* k = KV + (size_t)(b + t) * row;
            float v = 0;
            for (int i = lane; i < latent; i += 32) v = fmaf(qa[i], k[i], v);
            for (int i = lane; i < rope; i += 32) v = fmaf(q[nope + i], k[latent + i], v);
            for (int d = 16; d; d /= 2) v += __shfl_down_sync(0xffffffffu, v, d);
            if (!lane) sc[t] = v * rsqrtf((float)qh);
        }
        __syncthreads();
        red[tid] = tid < n ? sc[tid] : -CUDART_INF_F;
        __syncthreads();
        for (int d = 128; d; d /= 2) { if (tid < d) red[tid] = fmaxf(red[tid], red[tid + d]); __syncthreads(); }
        if (!tid) { float next = fmaxf(mx, red[0]); old_scale = expf(mx - next); mx = next; }
        __syncthreads();
        if (tid < n) sc[tid] = expf(sc[tid] - mx);
        __syncthreads();
        red[tid] = tid < n ? sc[tid] : 0;
        __syncthreads();
        for (int d = 128; d; d /= 2) { if (tid < d) red[tid] += red[tid + d]; __syncthreads(); }
        if (!tid) denom = denom * old_scale + red[0];
        for (int i = tid; i < latent; i += 256) {
            float v = ctx[i] * old_scale;
            for (int t = 0; t < n; ++t) v = fmaf(sc[t], KV[(size_t)(b + t) * row + i], v);
            ctx[i] = v;
        }
        __syncthreads();
    }
    for (int i = tid; i < latent; i += 256) ctx[i] /= denom;
    __syncthreads();
    const int lane = tid & 31, warp = tid / 32;
    for (int r = warp; r < vh; r += 8) {
        float v = 0;
        for (int i = lane; i < latent; i += 32) v = fmaf(weight(w, rb + nope + r, i), ctx[i], v);
        for (int d = 16; d; d /= 2) v += __shfl_down_sync(0xffffffffu, v, d);
        if (!lane) out[((size_t)s * H + hd) * vh + r] = v;
    }
}
uint64_t key(int layer, int e) { return (uint64_t)(uint32_t)layer << 32 | (uint32_t)e; }
size_t matrix_bytes(int O, int I) { return (size_t)O * I / 2 + (size_t)O * I / 64 * 4; }
DQ4 at(uint8_t* p, int O, int I) { return {O, I, p, (float*)(p + (size_t)O * I / 2)}; }
void put_matrix(uint8_t* dst, const Q4& w) {
    ck(cudaMemcpy(dst, w.codes, (size_t)w.O * w.I / 2, cudaMemcpyHostToDevice));
    ck(cudaMemcpy(dst + (size_t)w.O * w.I / 2, w.scales, (size_t)w.O * w.I / 64 * 4, cudaMemcpyHostToDevice));
}
} // namespace

struct CudaBackend::Impl {
    cublasHandle_t blas = nullptr;
    Buffer cache, weights, unpacked, x, y, g, u, q, kv;
    int D = 0, I = 0, threshold = 2, nslots = 0;
    size_t slot_size = 0, plane_size = 0;
    uint64_t clock = 0;
    struct Slot { uint64_t k = UINT64_MAX, age = 0; };
    std::vector<Slot> slots;
    std::unordered_map<uint64_t, int> where;
    std::unordered_map<uint64_t, uint64_t> frequency;
    CudaStats stats;
    ~Impl() { if (blas) cublasDestroy(blas); }
    void mul(DQ4 w, const float* a, int rows, float* b) {
        if (rows == 1) gemv<<<(w.O + 7) / 8, 256>>>(w, a, b);
        else {
            unpacked.reserve((size_t)w.O * w.I * 4);
            dequant<<<(unsigned)(((size_t)w.O * w.I + 255) / 256), 256>>>(w, unpacked.f());
            const float one = 1, zero = 0;
            // Row-major W[O,I] and X[S,I] are column-major [I,O] and [I,S].
            cb(cublasSgemm(blas, CUBLAS_OP_T, CUBLAS_OP_N, w.O, rows, w.I,
                          &one, unpacked.f(), w.I, a, w.I, &zero, b, w.O));
        }
        ck(cudaGetLastError());
    }
    void put_expert(uint8_t* p, const ExpertView& v) {
        put_matrix(p, v.gate); put_matrix(p + plane_size, v.up); put_matrix(p + 2 * plane_size, v.down);
    }
};
CudaBackend::CudaBackend() : p_(std::make_unique<Impl>()) {}
CudaBackend::~CudaBackend() = default;
bool CudaBackend::init(int device, int hidden, int intermediate, uint64_t budget, uint64_t reserve,
                       int promote_after, std::string& err) {
    try {
        ck(cudaSetDevice(device)); cb(cublasCreate(&p_->blas));
        cb(cublasSetMathMode(p_->blas, CUBLAS_PEDANTIC_MATH));
        p_->D = hidden; p_->I = intermediate; p_->threshold = std::max(1, promote_after);
        p_->plane_size = matrix_bytes(hidden, intermediate); p_->slot_size = 3 * p_->plane_size;
        size_t free = 0, total = 0; ck(cudaMemGetInfo(&free, &total));
        // Leave working memory for streamed dense GEMMs and the windowing system, independent of the expert tier.
        const uint64_t scratch = (uint64_t)1536 << 20;
        const uint64_t usable = free > reserve + scratch ? free - reserve - scratch : 0;
        if (budget > usable) throw std::runtime_error("GLM CUDA expert budget leaves insufficient working VRAM");
        const uint64_t chosen = budget ? budget : usable;
        p_->nslots = (int)(chosen / p_->slot_size);
        p_->cache.reserve((size_t)p_->nslots * p_->slot_size);
        p_->slots.resize(p_->nslots);
        return true;
    } catch (const std::exception& e) { err = e.what(); return false; }
}
bool CudaBackend::contains(int l, int e) const { return p_->where.count(key(l, e)) != 0; }
void CudaBackend::promote(int l, int e, const ExpertView& v) {
    auto& p = *p_; const uint64_t k = key(l, e); ++p.stats.requests;
    const uint64_t count = ++p.frequency[k];
    if (!p.nslots || count < (uint64_t)p.threshold || p.where.count(k)) return;
    int victim = -1;
    for (int i = 0; i < p.nslots; ++i) {
        if (p.slots[i].k == UINT64_MAX) { victim = i; break; }
        if (victim < 0 || p.frequency[p.slots[i].k] < p.frequency[p.slots[victim].k] ||
            (p.frequency[p.slots[i].k] == p.frequency[p.slots[victim].k] && p.slots[i].age < p.slots[victim].age)) victim = i;
    }
    if (victim < 0) return;
    auto& s = p.slots[victim];
    if (s.k != UINT64_MAX && p.frequency[s.k] > count) return;
    p.put_expert(p.cache.p + (size_t)victim * p.slot_size, v);
    if (s.k != UINT64_MAX) p.where.erase(s.k);
    s = {k, ++p.clock}; p.where[k] = victim; ++p.stats.promotions;
}
void CudaBackend::expert(int l, int e, const ExpertView* host, const float* x, int rows, float* y) {
    auto& p = *p_; const uint64_t k = key(l, e);
    auto it = p.where.find(k); uint8_t* w;
    if (it != p.where.end()) {
        w = p.cache.p + (size_t)it->second * p.slot_size;
        ++p.stats.hits; ++p.stats.requests; ++p.frequency[k]; p.slots[it->second].age = ++p.clock;
    } else {
        if (!host) throw std::runtime_error("GLM CUDA expert is not resident");
        p.weights.reserve(p.slot_size); p.put_expert(p.weights.p, *host); w = p.weights.p;
    }
    p.x.put(x, (size_t)rows * p.D * 4); p.y.reserve((size_t)rows * p.D * 4);
    p.g.reserve((size_t)rows * p.I * 4); p.u.reserve((size_t)rows * p.I * 4);
    p.mul(at(w, p.I, p.D), p.x.f(), rows, p.g.f());
    p.mul(at(w + p.plane_size, p.I, p.D), p.x.f(), rows, p.u.f());
    gated<<<(unsigned)(((size_t)rows * p.I + 255) / 256), 256>>>(p.g.f(), p.u.f(), (size_t)rows * p.I);
    p.mul(at(w + 2 * p.plane_size, p.D, p.I), p.g.f(), rows, p.y.f());
    ck(cudaMemcpy(y, p.y.p, (size_t)rows * p.D * 4, cudaMemcpyDeviceToHost));
}
void CudaBackend::gemm(const Q4& w, const float* x, int rows, float* y) {
    auto& p = *p_; p.weights.reserve(matrix_bytes(w.O, w.I)); put_matrix(p.weights.p, w);
    p.x.put(x, (size_t)rows * w.I * 4); p.y.reserve((size_t)rows * w.O * 4);
    p.mul(at(p.weights.p, w.O, w.I), p.x.f(), rows, p.y.f());
    ck(cudaMemcpy(y, p.y.p, (size_t)rows * w.O * 4, cudaMemcpyDeviceToHost));
}
void CudaBackend::attention(const Q4& w, const float* q, const float* kv, int rows, int pos0,
                            int heads, int nope, int rope, int value, float* ctx) {
    auto& p = *p_; p.weights.reserve(matrix_bytes(w.O, w.I)); put_matrix(p.weights.p, w);
    p.q.put(q, (size_t)rows * heads * (nope + rope) * 4);
    p.kv.put(kv, (size_t)(pos0 + rows) * (w.I + rope) * 4);
    p.y.reserve((size_t)rows * heads * value * 4);
    mla<<<dim3(heads, rows), 256>>>(at(p.weights.p, w.O, w.I), p.q.f(), p.kv.f(), rows, pos0, heads, nope, rope, value, p.y.f());
    ck(cudaGetLastError());
    ck(cudaMemcpy(ctx, p.y.p, (size_t)rows * heads * value * 4, cudaMemcpyDeviceToHost));
}
int CudaBackend::slots() const { return p_->nslots; }
uint64_t CudaBackend::bytes() const { return (uint64_t)p_->nslots * p_->slot_size; }
const CudaStats& CudaBackend::stats() const { return p_->stats; }
} // namespace strata::glm
