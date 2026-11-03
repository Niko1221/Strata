// src/kernels/cuda/kv_octave.cu - see include/strata/kernels/kv_octave.hpp. 3-bit KV on a Lloyd-Max grid.
//
// One warp per 32-value group, the same shape kv_q4.cu uses, so the append/gather entry points below mirror its
// page table, host copy and staging pool one for one. The two reductions use the same pairwise tree as the host
// reference (strides 16, 8, 4, 2, 1 through __shfl_down_sync), so its bytes match this kernel's bit for bit.
//
// The codes go out as three bit-planes by __ballot_sync, and every store/load is BYTE-wise: block_oct3 is 14
// bytes, so its qs starts at offset 2 and a uint32 view of it is a misaligned access (measured: garbage codes).
#include "strata/kernels/kv_octave.hpp"
#include "strata/kernels/f16_bits.hpp"
#include "strata/kernels/kv_stream.hpp"   // kStepPos, kStepWidth, KvHostPools

#include <cuda_runtime.h>

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>

namespace strata::kernels {
namespace {

void check(const char* what) {
    const cudaError_t e = cudaGetLastError();
    if (e != cudaSuccess) {
        std::fprintf(stderr, "kv_octave: %s: %s\n", what, cudaGetErrorString(e));
        std::exit(1);
    }
}

/// The total lands in lane 0; every lane needs it, so it is broadcast back.
__device__ __forceinline__ float warp_sum_bcast(float v) {
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) v += __shfl_down_sync(0xffffffffu, v, o);
    return __shfl_sync(0xffffffffu, v, 0);
}

__device__ __forceinline__ int oct_code(float x, float s0) {
    if (!(s0 > 0.0f)) return 0;
    const float a = fabsf(x / s0) * 32.0f;
    const int m = a < kOctThreshold[0] ? 0 : (a < kOctThreshold[1] ? 1 : (a < kOctThreshold[2] ? 2 : 3));
    return m + (x < 0.0f ? 4 : 0);
}

/// One 32-value group in one warp: returns this lane's code and the group's projection-preserving scale.
struct OctGroup {
    int code;
    float d;
};

__device__ __forceinline__ OctGroup oct_group(float x) {
    const float sxx = warp_sum_bcast(x * x);
    const float s0 = sxx > 0.0f ? sqrtf(sxx / (float) QK_OCT) : 0.0f;
    const int code = oct_code(x, s0);
    const float level = s0 > 0.0f ? (float) kOctLevel[code] * (1.0f / 32.0f) : 0.0f;
    const float sxL = warp_sum_bcast(x * level);
    return OctGroup{code, sxL > 0.0f ? sxx / sxL : 0.0f};
}

/// All 32 lanes of the warp must call this together (the ballots are what pack the block).
__device__ __forceinline__ void oct_store(block_oct3* blk, int lane, OctGroup g) {
    const uint32_t w0 = __ballot_sync(0xffffffffu, (g.code & 1) != 0);
    const uint32_t w1 = __ballot_sync(0xffffffffu, (g.code & 2) != 0);
    const uint32_t w2 = __ballot_sync(0xffffffffu, (g.code & 4) != 0);
    if (lane == 0) blk->d = f16_from_f32(g.d);
    if (lane < 3) {
        const uint32_t w = lane == 0 ? w0 : (lane == 1 ? w1 : w2);
        uint8_t* q = blk->qs + lane * 4;
        q[0] = (uint8_t) (w & 0xffu);
        q[1] = (uint8_t) ((w >> 8) & 0xffu);
        q[2] = (uint8_t) ((w >> 16) & 0xffu);
        q[3] = (uint8_t) ((w >> 24) & 0xffu);
    }
}

__device__ __forceinline__ void oct_store_row(uint8_t* pool, long long row, int b, int lane, OctGroup g) {
    block_oct3* blk = reinterpret_cast<block_oct3*>(pool + row * (long long) sizeof(block_oct3) * 8) + b;
    oct_store(blk, lane, g);
}

/// This lane's three bits of `blk`, read as bytes (same alignment reason as the writer).
__device__ __forceinline__ int oct_load_code(const block_oct3* blk, int lane) {
    const uint8_t* q = blk->qs;
    const int by = lane >> 3, bi = lane & 7;
    return ((q[by] >> bi) & 1) | (((q[4 + by] >> bi) & 1) << 1) | (((q[8 + by] >> bi) & 1) << 2);
}

__device__ __forceinline__ float oct_level(int code) { return (float) kOctLevel[code] * (1.0f / 32.0f); }

// ---- standalone (contiguous groups) ---------------------------------------------------------------
__global__ void kv_oct_quantize_kernel(const float* __restrict__ src, uint8_t* __restrict__ dst, int64_t n_groups) {
    const int64_t g = blockIdx.x;
    if (g >= n_groups) return;
    const int lane = threadIdx.x;
    const OctGroup r = oct_group(src[g * QK_OCT + lane]);
    oct_store(reinterpret_cast<block_oct3*>(dst) + g, lane, r);
}

__global__ void kv_oct_dequantize_kernel(const uint8_t* __restrict__ src, float* __restrict__ dst, int64_t n_groups) {
    const int64_t g = blockIdx.x;
    if (g >= n_groups) return;
    const int lane = threadIdx.x;
    const block_oct3* blk = reinterpret_cast<const block_oct3*>(src) + g;
    dst[g * QK_OCT + lane] = f32_from_f16(blk->d) * oct_level(oct_load_code(blk, lane));
}

// ---- the engine's entry points, shaped like kv_q4.cu's -------------------------------------------
// grid (kv_heads, blocks_per_head, n_steps * planes), 32 threads: one element of one 32-value group.
__global__ void kv_append_oct_kernel(uint8_t* __restrict__ k_oct, uint8_t* __restrict__ v_oct,
                                     const int32_t* __restrict__ table, const int32_t* __restrict__ step,
                                     int step_stride, int planes, const float* __restrict__ kcur,
                                     const float* __restrict__ vcur, int kv_heads, int head_dim, int page_size,
                                     KvHostPools host) {
    const int step_idx = (int) (blockIdx.z / (unsigned) planes);
    const bool is_v = (blockIdx.z % (unsigned) planes) == 1u;
    const long long pos = (long long) __ldg(step + (long long) step_idx * step_stride + kStepPos);
    const int h = blockIdx.x, b = blockIdx.y, t = threadIdx.x;
    const long long step_off = (long long) step_idx * kv_heads * head_dim;
    const float x = (is_v ? vcur : kcur)[step_off + h * head_dim + b * QK_OCT + t];
    const OctGroup g = oct_group(x);
    const long long page = (long long) table[pos / page_size];
    const long long row = (page * kv_heads + h) * page_size + (pos % page_size);
    if (page >= 0) oct_store_row(is_v ? v_oct : k_oct, row, b, t, g);
    if (host.k_q4 != nullptr) {
        const long long hrow = ((pos / page_size) * kv_heads + h) * page_size + (pos % page_size);
        oct_store_row(is_v ? host.v_q4 : host.k_q4, hrow, b, t, g);
    }
}

// The prompt path: grid (T, kv_heads, blocks_per_head), K then V; also into the staging pool when given.
__global__ void kv_append_oct_batch_kernel(uint8_t* __restrict__ k_oct, uint8_t* __restrict__ v_oct,
                                           const int32_t* __restrict__ table, int64_t pos0,
                                           const float* __restrict__ K, const float* __restrict__ V, int kv_heads,
                                           int head_dim, int page_size, int is_v_grid, KvHostPools host,
                                           KvHostPools stage) {
    const long long t = blockIdx.x;
    const long long pos = pos0 + t;
    const int h = blockIdx.y, b = blockIdx.z, th = threadIdx.x;
    const bool is_v = is_v_grid != 0;
    const float x = (is_v ? V : K)[t * (kv_heads * head_dim) + h * head_dim + b * QK_OCT + th];
    const OctGroup g = oct_group(x);
    const long long page = (long long) table[pos / page_size];
    const long long row_id = ((pos / page_size) * kv_heads + h) * page_size + (pos % page_size);
    if (page >= 0) oct_store_row(is_v ? v_oct : k_oct, (page * kv_heads + h) * page_size + (pos % page_size), b, th, g);
    if (host.k_q4 != nullptr) oct_store_row(is_v ? host.v_q4 : host.k_q4, row_id, b, th, g);
    if (stage.k_q4 != nullptr) oct_store_row(is_v ? stage.v_q4 : stage.k_q4, row_id, b, th, g);
}

// Gather step[kStepWidth] cells into FP16 scratch (the non-fused attention paths).
__global__ void kv_gather_oct_kernel(const uint8_t* __restrict__ k_oct, const uint8_t* __restrict__ v_oct,
                                     const int32_t* __restrict__ table, const int32_t* __restrict__ ids,
                                     const int32_t* __restrict__ step, int kv_heads, int head_dim, int page_size,
                                     uint16_t* __restrict__ k_scratch, uint16_t* __restrict__ v_scratch) {
    const long long n_ids = (long long) __ldg(step + kStepWidth);
    const int bph = head_dim / QK_OCT;
    const int bytes_per_head = bph * (int) sizeof(block_oct3);
    const long long total_blocks = n_ids * kv_heads * bph;
    const long long blk_idx = (long long) blockIdx.x * blockDim.y + threadIdx.y;
    if (blk_idx >= total_blocks) return;

    const int t = threadIdx.x;
    const long long id = blk_idx / (kv_heads * bph);
    const int rem = (int) (blk_idx % (kv_heads * bph));
    const int h = rem / bph, b = rem % bph;
    const int cell = ids[id];
    const long long page = (long long) table[cell / page_size];
    const long long row = (page * kv_heads + h) * page_size + (cell % page_size);
    const block_oct3* kb = reinterpret_cast<const block_oct3*>(k_oct + row * bytes_per_head) + b;
    const block_oct3* vb = reinterpret_cast<const block_oct3*>(v_oct + row * bytes_per_head) + b;
    const long long dst = ((id * kv_heads + h) * head_dim) + (b * QK_OCT + t);
    k_scratch[dst] = f16_from_f32(f32_from_f16(kb->d) * oct_level(oct_load_code(kb, t)));
    v_scratch[dst] = f16_from_f32(f32_from_f16(vb->d) * oct_level(oct_load_code(vb, t)));
}

void need_256(const QsaShapes& s, const char* what) {
    if (s.head_dim != 256) {
        std::fprintf(stderr, "%s: head_dim must be 256 (the Hadamard transform's size)\n", what);
        std::exit(1);
    }
}

}  // namespace

void kv_oct_quantize(const float* src, uint8_t* dst, int64_t n_groups, void* stream) {
    if (n_groups <= 0) return;
    kv_oct_quantize_kernel<<<(unsigned) n_groups, 32, 0, (cudaStream_t) stream>>>(src, dst, n_groups);
    check("quantize launch");
}

void kv_oct_dequantize(const uint8_t* src, float* dst, int64_t n_groups, void* stream) {
    if (n_groups <= 0) return;
    kv_oct_dequantize_kernel<<<(unsigned) n_groups, 32, 0, (cudaStream_t) stream>>>(src, dst, n_groups);
    check("dequantize launch");
}

void kv_append_oct_steps(uint8_t* k_oct, uint8_t* v_oct, const int32_t* page_table, const int32_t* step,
                         int step_stride, int n_steps, const float* kcur, const float* vcur, const QsaShapes& s,
                         void* stream, const KvHostPools* host) {
    if (n_steps <= 0) return;
    need_256(s, "kv_append_oct");
    const int planes = (v_oct == nullptr || (v_oct == k_oct && vcur == kcur)) ? 1 : 2;
    const dim3 grid((unsigned) s.n_head_kv, (unsigned) (s.head_dim / QK_OCT), (unsigned) (n_steps * planes));
    kv_append_oct_kernel<<<grid, 32, 0, (cudaStream_t) stream>>>(
        k_oct, v_oct, page_table, step, step_stride, planes, kcur, vcur, (int) s.n_head_kv, (int) s.head_dim,
        (int) s.page_size, host ? *host : KvHostPools{});
    check("kv_append_oct launch");
}

void kv_append_oct_step(uint8_t* k_oct, uint8_t* v_oct, const int32_t* page_table, const int32_t* step,
                        const float* kcur, const float* vcur, const QsaShapes& s, void* stream,
                        const KvHostPools* host) {
    kv_append_oct_steps(k_oct, v_oct, page_table, step, 0, 1, kcur, vcur, s, stream, host);
}

void kv_append_oct(uint8_t* k_oct, uint8_t* v_oct, const int32_t* page_table, int64_t pos0, int64_t T, const float* K,
                   const float* V, const QsaShapes& s, void* stream, const KvHostPools* host, const KvHostPools* stage) {
    if (T <= 0) return;
    need_256(s, "kv_append_oct");
    const dim3 grid((unsigned) T, (unsigned) s.n_head_kv, (unsigned) (s.head_dim / QK_OCT));
    cudaStream_t cs = (cudaStream_t) stream;
    const KvHostPools h = host ? *host : KvHostPools{}, st = stage ? *stage : KvHostPools{};
    for (int is_v = 0; is_v < 2; ++is_v)
        kv_append_oct_batch_kernel<<<grid, 32, 0, cs>>>(k_oct, v_oct, page_table, pos0, K, V, (int) s.n_head_kv,
                                                        (int) s.head_dim, (int) s.page_size, is_v, h, st);
    check("kv_append_oct batch launch");
}

void kv_gather_oct_step(const uint8_t* k_oct, const uint8_t* v_oct, const int32_t* page_table, const int32_t* ids,
                        const int32_t* step, int64_t max_ids, const QsaShapes& s, uint16_t* k_scratch,
                        uint16_t* v_scratch, void* stream) {
    if (max_ids <= 0) return;
    const int bph = (int) (s.head_dim / QK_OCT);
    const int64_t total_blocks = max_ids * s.n_head_kv * bph;
    const int rows_per_block = 4;
    const unsigned nb = (unsigned) ((total_blocks + rows_per_block - 1) / rows_per_block);
    kv_gather_oct_kernel<<<dim3(nb), dim3(32, rows_per_block), 0, (cudaStream_t) stream>>>(
        k_oct, v_oct, page_table, ids, step, (int) s.n_head_kv, (int) s.head_dim, (int) s.page_size, k_scratch,
        v_scratch);
    check("kv_gather_oct launch");
}

// ---- host reference: the same tree, the same constants ------------------------------------------------
namespace {
float tree_sum32(const float* v) {
    float t[32];
    for (int i = 0; i < 32; ++i) t[i] = v[i];
    for (int o = 16; o > 0; o >>= 1)
        for (int i = 0; i < o; ++i) t[i] += t[i + o];
    return t[0];
}
}  // namespace

void kv_oct_quantize_group_host(const float* x, block_oct3& blk) {
    float sq[32], xl[32];
    int codes[32];
    for (int i = 0; i < 32; ++i) sq[i] = x[i] * x[i];
    const float sxx = tree_sum32(sq);
    const float s0 = sxx > 0.0f ? std::sqrt(sxx / (float) QK_OCT) : 0.0f;
    for (int i = 0; i < 32; ++i) {
        int c = 0;
        float level = 0.0f;
        if (s0 > 0.0f) {
            const float a = std::fabs(x[i] / s0) * 32.0f;
            const int m = a < kOctThreshold[0] ? 0 : (a < kOctThreshold[1] ? 1 : (a < kOctThreshold[2] ? 2 : 3));
            c = m + (x[i] < 0.0f ? 4 : 0);
            level = (float) kOctLevel[c] * (1.0f / 32.0f);
        }
        codes[i] = c;
        xl[i] = x[i] * level;
    }
    const float sxL = tree_sum32(xl);
    blk.d = f16_from_f32(sxL > 0.0f ? sxx / sxL : 0.0f);
    uint32_t w[3] = {0u, 0u, 0u};
    for (int i = 0; i < 32; ++i)
        for (int b = 0; b < 3; ++b)
            if ((codes[i] & (1 << b)) != 0) w[b] |= 1u << i;
    std::memcpy(blk.qs, w, sizeof(w));
}

void kv_oct_dequantize_group_host(const block_oct3& blk, float* x) {
    uint32_t w[3];
    std::memcpy(w, blk.qs, sizeof(w));
    const float d = f32_from_f16(blk.d);
    for (int i = 0; i < 32; ++i) {
        const int code = (int) (((w[0] >> i) & 1u) | (((w[1] >> i) & 1u) << 1) | (((w[2] >> i) & 1u) << 2));
        x[i] = d * ((float) kOctLevel[code] * (1.0f / 32.0f));
    }
}

}  // namespace strata::kernels
