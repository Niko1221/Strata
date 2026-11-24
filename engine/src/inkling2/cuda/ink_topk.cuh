// ink_topk.cuh - exact top-128 candidates (and the finite flag) of full logit rows on the GPU, so a decode step with many slots
// copies 128 ids + logits per row to the host instead of 200058 floats (#6779).
//
// Order: larger logit first, the smaller id first on equal logits (-0 counts as +0) - the order of std::partial_sort with the
// sampler's comparator, so the host sampler sees exactly the candidates it used to compute itself.  A NaN logit gets the smallest
// key (the engine rejects non-finite rows before sampling anyway).
//
// How: key = (order-preserving 32-bit image of the logit) << 32 | (0xFFFFFFFF - id); the largest key is the best candidate.
//   pass 1: grid (128 slices, rows): each block sorts its 1563-element slice (2048 keys, bitonic, shared memory) and keeps 128
//   pass 2: grid (8 groups, rows): merges 16 sorted lists of 128 (2048 keys) -> 128
//   pass 3: grid (1, rows): merges 8 lists (1024 keys, padded) -> 128
// The finite flag: every id that is a multiple of 67 (what finite_logits() in ink_serve.h looks at) must be finite.
#pragma once

#include <cstdint>
#include <cstring>
#include <cuda_runtime.h>

namespace strata {
namespace ink {

constexpr int kTopkCand = 128;
constexpr int kTopkSlices = 128;      // pass-1 blocks per row
constexpr int kTopkSortN = 2048;      // keys one block sorts
constexpr int kTopkThreads = 256;

__device__ __forceinline__ uint64_t ink_topk_key(float v, int idx) {
    if (v != v) return 0ull;                          // NaN: below every real key
    if (v == 0.0f) v = __uint_as_float(0u);           // -0 -> +0 (equal logits order by id)
    uint32_t b = __float_as_uint(v);
    b ^= (b & 0x80000000u) ? 0xFFFFFFFFu : 0x80000000u;
    return ((uint64_t)b << 32) | (uint64_t)(0xFFFFFFFFu - (uint32_t)idx);
}

// Bitonic sort of kTopkSortN keys in shared memory, largest first.
__device__ __forceinline__ void ink_topk_sort2048(uint64_t* a) {
    for (int k = 2; k <= kTopkSortN; k <<= 1) {
        for (int j = k >> 1; j > 0; j >>= 1) {
            for (int i = threadIdx.x; i < kTopkSortN; i += kTopkThreads) {
                const int ixj = i ^ j;
                if (ixj > i) {
                    const uint64_t x = a[i], y = a[ixj];
                    const bool desc = ((i & k) == 0);
                    if (desc ? (x < y) : (x > y)) { a[i] = y; a[ixj] = x; }
                }
            }
            __syncthreads();
        }
    }
}

// pass 1: rows [nrows, V] row-major -> out[(row * kTopkSlices + slice) * 128 ..), fin[row] &= every 67th logit finite
__global__ void __launch_bounds__(kTopkThreads) k_ink_topk_pass1(const float* __restrict__ logits, int V,
                                                                 uint64_t* __restrict__ out, int* __restrict__ fin) {
    __shared__ uint64_t s_key[kTopkSortN];
    const int row = blockIdx.y, slice = blockIdx.x;
    const int S = (V + kTopkSlices - 1) / kTopkSlices;             // 1563 for V = 200058
    const int base = slice * S;
    const float* r = logits + (size_t)row * V;
    int bad = 0;
    for (int i = threadIdx.x; i < kTopkSortN; i += kTopkThreads) {
        const int g = base + i;
        uint64_t key = 0ull;
        if (i < S && g < V) {
            const float v = r[g];
            key = ink_topk_key(v, g);
            if (g % 67 == 0 && !isfinite(v)) bad = 1;
        }
        s_key[i] = key;
    }
    if (bad) atomicAnd(&fin[row], 0);
    __syncthreads();
    ink_topk_sort2048(s_key);
    uint64_t* o = out + ((size_t)row * kTopkSlices + slice) * kTopkCand;
    for (int i = threadIdx.x; i < kTopkCand; i += kTopkThreads) o[i] = s_key[i];
}

// pass 2 / 3: `lists` sorted lists of 128 keys per row in `in`; groups of `m` lists (m * 128 <= 2048) -> one sorted list of 128
__global__ void __launch_bounds__(kTopkThreads) k_ink_topk_merge(const uint64_t* __restrict__ in, int lists, int m,
                                                                 uint64_t* __restrict__ out, int groups) {
    __shared__ uint64_t s_key[kTopkSortN];
    const int row = blockIdx.y, g = blockIdx.x;
    const uint64_t* src = in + ((size_t)row * lists + (size_t)g * m) * kTopkCand;
    const int n = m * kTopkCand;
    for (int i = threadIdx.x; i < kTopkSortN; i += kTopkThreads) s_key[i] = (i < n) ? src[i] : 0ull;
    __syncthreads();
    ink_topk_sort2048(s_key);
    uint64_t* o = out + ((size_t)row * groups + g) * kTopkCand;
    for (int i = threadIdx.x; i < kTopkCand; i += kTopkThreads) o[i] = s_key[i];
}

// Scratch for up to `rows` rows: pass-1 output, pass-2 output, the final lists, the flags.
struct InkTopkScratch {
    uint64_t* d_p1 = nullptr;   // rows * 128 * 128
    uint64_t* d_p2 = nullptr;   // rows * 8 * 128
    uint64_t* d_out = nullptr;  // rows * 128
    int* d_fin = nullptr;       // rows
    uint64_t* h_out = nullptr;  // pinned, rows * 128
    int* h_fin = nullptr;       // pinned, rows
    int cap = 0;

    void free_all() {
        if (d_p1) cudaFree(d_p1);
        if (d_p2) cudaFree(d_p2);
        if (d_out) cudaFree(d_out);
        if (d_fin) cudaFree(d_fin);
        if (h_out) cudaFreeHost(h_out);
        if (h_fin) cudaFreeHost(h_fin);
        d_p1 = d_p2 = d_out = nullptr; d_fin = nullptr; h_out = nullptr; h_fin = nullptr; cap = 0;
    }

    // false when an allocation fails (the caller then keeps the full-row path)
    bool ensure(int rows) {
        if (rows <= cap) return true;
        free_all();
        const int c = rows < 64 ? 64 : rows;
        if (cudaMalloc(&d_p1, (size_t)c * kTopkSlices * kTopkCand * 8) != cudaSuccess ||
            cudaMalloc(&d_p2, (size_t)c * 8 * kTopkCand * 8) != cudaSuccess ||
            cudaMalloc(&d_out, (size_t)c * kTopkCand * 8) != cudaSuccess ||
            cudaMalloc(&d_fin, (size_t)c * sizeof(int)) != cudaSuccess ||
            cudaMallocHost(&h_out, (size_t)c * kTopkCand * 8) != cudaSuccess ||
            cudaMallocHost(&h_fin, (size_t)c * sizeof(int)) != cudaSuccess) {
            cudaGetLastError();
            free_all();
            return false;
        }
        cap = c;
        return true;
    }

    // launches the three passes and the copy of the small results to the pinned buffers on `st` (the caller synchronizes)
    cudaError_t run(const float* d_logits, int rows, int V, cudaStream_t st) {
        // every flag starts non-zero (0x01010101); a block that sees a non-finite logit at a multiple of 67 ANDs it to 0
        cudaError_t e = cudaMemsetAsync(d_fin, 1, (size_t)rows * sizeof(int), st);
        if (e != cudaSuccess) return e;
        k_ink_topk_pass1<<<dim3(kTopkSlices, rows), kTopkThreads, 0, st>>>(d_logits, V, d_p1, d_fin);
        k_ink_topk_merge<<<dim3(8, rows), kTopkThreads, 0, st>>>(d_p1, kTopkSlices, 16, d_p2, 8);
        k_ink_topk_merge<<<dim3(1, rows), kTopkThreads, 0, st>>>(d_p2, 8, 8, d_out, 1);
        e = cudaGetLastError();
        if (e != cudaSuccess) return e;
        e = cudaMemcpyAsync(h_out, d_out, (size_t)rows * kTopkCand * 8, cudaMemcpyDeviceToHost, st);
        if (e != cudaSuccess) return e;
        return cudaMemcpyAsync(h_fin, d_fin, (size_t)rows * sizeof(int), cudaMemcpyDeviceToHost, st);
    }
};

// key -> logit / id (host side)
inline float ink_topk_key_value(uint64_t key) {
    uint32_t b = (uint32_t)(key >> 32);
    b ^= (b & 0x80000000u) ? 0x80000000u : 0xFFFFFFFFu;
    float f;
    static_assert(sizeof(f) == sizeof(b), "float is 32 bit");
    std::memcpy(&f, &b, sizeof(f));
    return f;
}
inline int32_t ink_topk_key_index(uint64_t key) { return (int32_t)(0xFFFFFFFFu - (uint32_t)(key & 0xFFFFFFFFu)); }

}  // namespace ink
}  // namespace strata
