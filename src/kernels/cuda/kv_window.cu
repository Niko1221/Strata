// src/kernels/cuda/kv_window.cu - see include/strata/kernels/kv_window.hpp.
//
// Three small kernels: the FP16 write that mirrors an append into the overlay (per token, batched, and the prompt
// form), and the restore that rebuilds the overlay from the quantized bulk the snapshot brought back.  All of them
// read the position from the step buffer on the DEVICE, so every launch is graph-capturable and the append call
// sites need no host arithmetic.
#include "strata/core/emulate.hpp"

#include "strata/kernels/f16_bits.hpp"
#include "strata/kernels/kv_window.hpp"
#include "strata/kernels/kv_q4.hpp"
#include "strata/kernels/kv_q2.hpp"
#include "strata/kernels/kv_q8.hpp"
#include "strata/kernels/qsa_decode_attn.hpp"

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cstdio>
#include <cstdlib>

namespace strata::kernels {
namespace {

constexpr int THREADS = 256;

// f16_from_f32, not __float2half: this engine keeps its own round-to-nearest-even converter (f16_bits.hpp) precisely
// because the intrinsic produced wrong bits, and the overlay has to agree with the FP16 pools bit for bit.
__device__ __forceinline__ uint16_t win_half(float x) { return f16_from_f32(x); }

/// One appended cell: copy this KV head's row of kcur/vcur into slot (sinks: the cell's own, window: a ring).
__global__ void __launch_bounds__(THREADS) window_write_step_kernel(uint16_t* __restrict__ kw,
                                                                   uint16_t* __restrict__ vw, int32_t window,
                                                                   int32_t sinks, const int32_t* __restrict__ step,
                                                                   const float* __restrict__ kcur,
                                                                   const float* __restrict__ vcur, int n_head_kv,
                                                                   int head_dim) {
    const int kvh = blockIdx.x;
    const int64_t cell = __ldg(step + kStepPos);
    const int64_t n_kv = __ldg(step + kStepNKv);
    const int32_t slot = kv_window_slot(cell, n_kv, window, sinks);
    if (slot < 0) return;
    const int64_t dst = ((int64_t) slot * n_head_kv + kvh) * head_dim;
    const int64_t src = (int64_t) kvh * head_dim;
    for (int d = threadIdx.x; d < head_dim; d += blockDim.x) {
        kw[dst + d] = win_half(kcur[src + d]);
        vw[dst + d] = win_half(vcur[src + d]);
    }
}

/// The batched decode form: one block per step, each cell's own position and cache length out of its step row.
__global__ void __launch_bounds__(THREADS) window_write_steps_kernel(uint16_t* __restrict__ kw,
                                                                    uint16_t* __restrict__ vw, int32_t window,
                                                                    int32_t sinks, const int32_t* __restrict__ step,
                                                                    int step_stride, const float* __restrict__ kcur,
                                                                    const float* __restrict__ vcur, int n_head_kv,
                                                                    int head_dim) {
    const int32_t* st = step + (int64_t) blockIdx.x * step_stride;
    const int64_t cell = __ldg(st + kStepPos);
    const int64_t n_kv = __ldg(st + kStepNKv);
    const int32_t slot = kv_window_slot(cell, n_kv, window, sinks);
    if (slot < 0) return;
    const float* kc = kcur + (int64_t) blockIdx.x * n_head_kv * head_dim;
    const float* vc = vcur + (int64_t) blockIdx.x * n_head_kv * head_dim;
    for (int i = threadIdx.x; i < n_head_kv * head_dim; i += blockDim.x) {
        const int kvh = i / head_dim, d = i % head_dim;
        const int64_t dst = ((int64_t) slot * n_head_kv + kvh) * head_dim + d;
        kw[dst] = win_half(kc[i]);
        vw[dst] = win_half(vc[i]);
    }
}

/// The prompt form: T consecutive cells from pos0, the cache length being pos0 + T (the chunk's own append).
/// A cell whose slot is out of the window then is left alone: a later append of the cell that does own that slot
/// overwrites it, so the ring always holds the last `window` cells once the prompt is in.
__global__ void __launch_bounds__(THREADS) window_write_batch_kernel(uint16_t* __restrict__ kw,
                                                                    uint16_t* __restrict__ vw, int32_t window,
                                                                    int32_t sinks, int64_t pos0, int64_t T,
                                                                    const float* __restrict__ K,
                                                                    const float* __restrict__ V, int n_head_kv,
                                                                    int head_dim) {
    const int64_t cell = pos0 + blockIdx.x;
    const int64_t n_kv = pos0 + T;
    const int32_t slot = kv_window_slot(cell, n_kv, window, sinks);
    if (slot < 0) return;
    const float* kc = K + (int64_t) blockIdx.x * n_head_kv * head_dim;
    const float* vc = V + (int64_t) blockIdx.x * n_head_kv * head_dim;
    for (int i = threadIdx.x; i < n_head_kv * head_dim; i += blockDim.x) {
        const int kvh = i / head_dim, d = i % head_dim;
        const int64_t dst = ((int64_t) slot * n_head_kv + kvh) * head_dim + d;
        kw[dst] = win_half(kc[i]);
        vw[dst] = win_half(vc[i]);
    }
}

// ---- restore: dequantize one element of a cell from the identity (un-resident) layout of the bulk pools ----

// q2 (kv_q2.hpp): the 4-level Lloyd-Max dequantization `kv_gather_q2_kernel` writes with `v3level = false`, the
// shipping setting for K and V, so a restored window holds exactly what the decode attention reads from the bulk.
// Per-channel scales: the cell's code lives in the slot (group, head, dim) of its 32-cell group, the group being
// 32 consecutive cells OF THE POOL - the same decomposition `load8_q2` and `load_v1` do, from the same row.
__device__ __forceinline__ float bulk_q2(const uint8_t* pool, long long row, int d, int kv_heads, int page_size) {
    const long long cell = (row / ((long long) kv_heads * page_size)) * page_size + (row % page_size);
    const int h = (int) ((row / page_size) % kv_heads);
    const block_q2c_0* blk = q2c_slot(pool, cell / Q2_0, h, d, kv_heads, 256);
    const int idx = (int) (cell % Q2_0);
    const int code = (blk->qs[idx >> 2] >> (2 * (idx & 3))) & 3;
    return q2_lm4_level(code) * f32_from_f16(blk->d);
}

template <int KV_MODE>
__device__ __forceinline__ float bulk_k(const QsaAttnPools& p, long long row, int d, int n_head_kv, int page_size) {
    if constexpr (KV_MODE == 0) {
        return f32_from_f16(p.k_pool[row * 256 + d]);
    } else if constexpr (KV_MODE == 2) {
        constexpr int BYTES = (256 / QK4_0) * sizeof(block_q4_0);
        const block_q4_0* blk = reinterpret_cast<const block_q4_0*>(p.k_q4 + row * BYTES) + d / QK4_0;
        const int rem = d % QK4_0;
        const int nib = rem < 16 ? (blk->qs[rem] & 0x0F) : (blk->qs[rem - 16] >> 4);
        return (float) (nib - 8) * __half2float(__ushort_as_half(blk->d));
    } else if constexpr (KV_MODE == 4) {
        return bulk_q2(p.k_q2, row, d, n_head_kv, page_size);
    } else {
        const float sc = __half2float(__ushort_as_half(p.k_scale[row * (256 / KV_Q8_GROUP) + d / KV_Q8_GROUP]));
        return (float) p.k_q[row * 256 + d] * sc;
    }
}

template <int KV_MODE>
__device__ __forceinline__ float bulk_v(const QsaAttnPools& p, long long row, int d, int n_head_kv, int page_size) {
    if constexpr (KV_MODE == 0) {
        return f32_from_f16(p.v_pool[row * 256 + d]);
    } else if constexpr (KV_MODE == 1) {
        const float sc = __half2float(__ushort_as_half(p.v_scale[row * (256 / KV_Q8_GROUP) + d / KV_Q8_GROUP]));
        return (float) p.v_q[row * 256 + d] * sc;
    } else if constexpr (KV_MODE == 4) {
        return bulk_q2(p.v_q2, row, d, n_head_kv, page_size);
    } else {   // modes 2 and 3: V is q4_0
        constexpr int BYTES = (256 / QK4_0) * sizeof(block_q4_0);
        const block_q4_0* blk = reinterpret_cast<const block_q4_0*>(p.v_q4 + row * BYTES) + d / QK4_0;
        const int rem = d % QK4_0;
        const int nib = rem < 16 ? (blk->qs[rem] & 0x0F) : (blk->qs[rem - 16] >> 4);
        return (float) (nib - 8) * __half2float(__ushort_as_half(blk->d));
    }
}

/// Cells [cs, ce) of the sequence: `blockIdx.x` is one (cell, kv head) pair, one cell per block over dims.
template <int KV_MODE>
__global__ void __launch_bounds__(THREADS) window_restore_kernel(uint16_t* __restrict__ kw,
                                                                uint16_t* __restrict__ vw, int32_t window,
                                                                int32_t sinks, int64_t n_kv, QsaAttnPools p,
                                                                int n_head_kv, int head_dim, int page_size, int64_t cs) {
    const int64_t cell = cs + blockIdx.x / n_head_kv;
    const int kvh = (int) (blockIdx.x % n_head_kv);
    if (cell >= n_kv) return;
    const int32_t slot = kv_window_slot(cell, n_kv, window, sinks);
    if (slot < 0) return;
    // the authoritative copy is the identity layout: the cell's page is cell / page_size
    const long long page = (long long) (cell / page_size);
    const long long row = (page * n_head_kv + kvh) * page_size + (long long) (cell % page_size);
    const int64_t dst = ((int64_t) slot * n_head_kv + kvh) * head_dim;
    for (int d = threadIdx.x; d < head_dim; d += blockDim.x) {
        kw[dst + d] = win_half(bulk_k<KV_MODE>(p, row, d, n_head_kv, page_size));
        vw[dst + d] = win_half(bulk_v<KV_MODE>(p, row, d, n_head_kv, page_size));
    }
}

void win_check(const char* what) {
    const cudaError_t e = cudaGetLastError();
    if (e != cudaSuccess) {
        std::fprintf(stderr, "kv_window %s: %s\n", what, cudaGetErrorString(e));
        std::exit(1);
    }
}

bool win_off(const KvWindowPools& w, const KvWindowShape& shape) { return !shape.on() || !w.k || !w.v; }

}  // namespace

void kv_window_write_step(const KvWindowPools& w, const KvWindowShape& shape, const int32_t* step, const float* kcur,
                          const float* vcur, const QsaShapes& s, void* stream) {
    if (win_off(w, shape)) return;
    const dim3 grid((unsigned) s.n_head_kv);
    window_write_step_kernel<<<grid, THREADS, 0, (cudaStream_t) stream>>>(w.k, w.v, shape.window, shape.sinks, step,
                                                                         kcur, vcur, (int) s.n_head_kv,
                                                                         (int) s.head_dim);
    win_check("write_step");
}

void kv_window_write_steps(const KvWindowPools& w, const KvWindowShape& shape, const int32_t* step, int step_stride,
                           int n, const float* kcur, const float* vcur, const QsaShapes& s, void* stream) {
    if (win_off(w, shape) || n <= 0) return;
    window_write_steps_kernel<<<(unsigned) n, THREADS, 0, (cudaStream_t) stream>>>(
        w.k, w.v, shape.window, shape.sinks, step, step_stride, kcur, vcur, (int) s.n_head_kv, (int) s.head_dim);
    win_check("write_steps");
}

void kv_window_write_batch(const KvWindowPools& w, const KvWindowShape& shape, int64_t pos0, int64_t T, const float* K,
                           const float* V, const QsaShapes& s, void* stream) {
    if (win_off(w, shape) || T <= 0) return;
    window_write_batch_kernel<<<(unsigned) T, THREADS, 0, (cudaStream_t) stream>>>(
        w.k, w.v, shape.window, shape.sinks, pos0, T, K, V, (int) s.n_head_kv, (int) s.head_dim);
    win_check("write_batch");
}

void kv_window_restore(const KvWindowPools& w, const KvWindowShape& shape, const QsaAttnPools& bulk, int kv_mode,
                       int64_t n_kv, const QsaShapes& s, void* stream) {
    if (win_off(w, shape) || n_kv <= 0) return;
    const int nh = (int) s.n_head_kv, ps = (int) s.page_size;
    cudaStream_t st = (cudaStream_t) stream;
    const int sink_cells = (int) ((shape.sinks < n_kv) ? shape.sinks : n_kv);
    if (sink_cells > 0) {
        const dim3 grid((unsigned) (sink_cells * nh));
        switch (kv_mode) {
            case 2: window_restore_kernel<2><<<grid, THREADS, 0, st>>>(w.k, w.v, shape.window, shape.sinks, n_kv, bulk, nh, (int) s.head_dim, ps, 0); break;
            case 1: window_restore_kernel<1><<<grid, THREADS, 0, st>>>(w.k, w.v, shape.window, shape.sinks, n_kv, bulk, nh, (int) s.head_dim, ps, 0); break;
            case 3: window_restore_kernel<3><<<grid, THREADS, 0, st>>>(w.k, w.v, shape.window, shape.sinks, n_kv, bulk, nh, (int) s.head_dim, ps, 0); break;
            case 4: window_restore_kernel<4><<<grid, THREADS, 0, st>>>(w.k, w.v, shape.window, shape.sinks, n_kv, bulk, nh, (int) s.head_dim, ps, 0); break;
            default: window_restore_kernel<0><<<grid, THREADS, 0, st>>>(w.k, w.v, shape.window, shape.sinks, n_kv, bulk, nh, (int) s.head_dim, ps, 0); break;
        }
    }
    int64_t ws = n_kv - shape.window;
    if (ws < shape.sinks) ws = shape.sinks;   // the sinks own the head of the sequence
    if (ws < n_kv) {
        const dim3 grid((unsigned) ((n_kv - ws) * nh));
        switch (kv_mode) {
            case 2: window_restore_kernel<2><<<grid, THREADS, 0, st>>>(w.k, w.v, shape.window, shape.sinks, n_kv, bulk, nh, (int) s.head_dim, ps, ws); break;
            case 1: window_restore_kernel<1><<<grid, THREADS, 0, st>>>(w.k, w.v, shape.window, shape.sinks, n_kv, bulk, nh, (int) s.head_dim, ps, ws); break;
            case 3: window_restore_kernel<3><<<grid, THREADS, 0, st>>>(w.k, w.v, shape.window, shape.sinks, n_kv, bulk, nh, (int) s.head_dim, ps, ws); break;
            case 4: window_restore_kernel<4><<<grid, THREADS, 0, st>>>(w.k, w.v, shape.window, shape.sinks, n_kv, bulk, nh, (int) s.head_dim, ps, ws); break;
            default: window_restore_kernel<0><<<grid, THREADS, 0, st>>>(w.k, w.v, shape.window, shape.sinks, n_kv, bulk, nh, (int) s.head_dim, ps, ws); break;
        }
    }
    win_check("restore");
}

}  // namespace strata::kernels
