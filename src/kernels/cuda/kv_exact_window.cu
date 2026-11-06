// src/kernels/cuda/kv_exact_window.cu - see include/strata/kernels/kv_exact_window.hpp.
#include "strata/kernels/kv_exact_window.hpp"
#include "strata/kernels/f16_bits.hpp"
#include "strata/kernels/kv_stream.hpp"   // kStepPos

#include <cuda_runtime.h>

#include <cstdio>
#include <cstdlib>

namespace strata::kernels {
namespace {

void check(const char* what) {
    const cudaError_t e = cudaGetLastError();
    if (e != cudaSuccess) {
        std::fprintf(stderr, "kv_exact_window: %s: %s\n", what, cudaGetErrorString(e));
        std::exit(1);
    }
}

__global__ void kv_exact_window_append_kernel(KvExactWindow w, int layer, const int32_t* __restrict__ step,
                                        const float* __restrict__ kcur, const float* __restrict__ vcur, int n_cells) {
    const int c = blockIdx.x, h = blockIdx.y, d = threadIdx.x;
    const long long pos = __ldg(step + (long long) c * w.pos_stride + kStepPos);
    const int row = kv_window_row(w, pos);
    if (row < 0) return;
    const size_t src = ((size_t) c * w.n_head_kv + h) * w.head_dim + d;
    const size_t dst = ((size_t) layer * w.rows() + row) * (size_t) (w.n_head_kv * w.head_dim) +
                       (size_t) h * w.head_dim + d;
    w.k[dst] = f16_from_f32(kcur[src]);
    w.v[dst] = f16_from_f32(vcur[src]);
    if (d == 0 && h == 0) w.tag[(size_t) layer * w.rows() + row] = (int32_t) pos;
}

/// The prompt path. `end` is the position the prompt ends at, so a cell older than `end - tail` (and not a sink)
/// can never be in the window: it is skipped, which bounds the launch and keeps one writer per ring row.
__global__ void kv_exact_window_append_range_kernel(KvExactWindow w, int layer, long long pos0, long long end,
                                                    const float* __restrict__ kcur, const float* __restrict__ vcur,
                                                    long long n_cells) {
    const long long c = blockIdx.x;
    if (c >= n_cells) return;
    const long long pos = pos0 + c;
    if (pos >= w.sinks && pos < end - w.tail) return;
    const int row = kv_window_row(w, pos);
    if (row < 0) return;
    const int h = blockIdx.y, d = threadIdx.x;
    const size_t src = ((size_t) c * w.n_head_kv + h) * w.head_dim + d;
    const size_t dst = ((size_t) layer * w.rows() + row) * (size_t) (w.n_head_kv * w.head_dim) +
                       (size_t) h * w.head_dim + d;
    w.k[dst] = f16_from_f32(kcur[src]);
    w.v[dst] = f16_from_f32(vcur[src]);
    if (d == 0 && h == 0) w.tag[(size_t) layer * w.rows() + row] = (int32_t) pos;
}

__global__ void kv_exact_window_gather_kernel(const KvExactWindow w, int layer, const int32_t* __restrict__ positions, int64_t n,
                                        uint16_t* __restrict__ k_out, uint16_t* __restrict__ v_out,
                                        int32_t* __restrict__ hit) {
    const int64_t i = blockIdx.x;
    if (i >= n) return;
    const int h = blockIdx.y, d = threadIdx.x;
    const long long pos = __ldg(positions + i);
    const int row = kv_window_row(w, pos);
    const int ok = row >= 0 && __ldg(w.tag + (size_t) layer * w.rows() + row) == (int32_t) pos;
    if (d == 0 && h == 0 && hit != nullptr) hit[i] = ok;
    if (!ok) return;
    const size_t src = ((size_t) layer * w.rows() + row) * (size_t) (w.n_head_kv * w.head_dim) +
                       (size_t) h * w.head_dim + d;
    const size_t dst = ((size_t) i * w.n_head_kv + h) * w.head_dim + d;
    k_out[dst] = w.k[src];
    v_out[dst] = w.v[src];
}

}  // namespace

void kv_exact_window_init(KvExactWindow& w, const QsaShapes& s, int layers, int sinks, int tail, int pos_stride, void* stream) {
    w.layers = layers;
    w.sinks = sinks;
    w.tail = tail;
    w.n_head_kv = (int) s.n_head_kv;
    w.head_dim = (int) s.head_dim;
    w.pos_stride = pos_stride;
    if (!w.present() || layers <= 0 || w.rows() <= 0) return;
    cudaMemsetAsync(w.tag, 0xff, (size_t) layers * (size_t) w.rows() * sizeof(int32_t), (cudaStream_t) stream);
    check("tag memset");
}

void kv_exact_window_append(const KvExactWindow& w, int layer, const int32_t* step, const float* kcur, const float* vcur,
                      int n_cells, void* stream) {
    if (!w.present() || n_cells <= 0) return;
    kv_exact_window_append_kernel<<<dim3((unsigned) n_cells, (unsigned) w.n_head_kv), (unsigned) w.head_dim, 0,
                              (cudaStream_t) stream>>>(w, layer, step, kcur, vcur, n_cells);
    check("append launch");
}

void kv_exact_window_append_range(const KvExactWindow& w, int layer, int64_t pos0, int64_t end, const float* kcur,
                      const float* vcur, int64_t n_cells, void* stream) {
    if (!w.present() || n_cells <= 0) return;
    kv_exact_window_append_range_kernel<<<dim3((unsigned) n_cells, (unsigned) w.n_head_kv), (unsigned) w.head_dim, 0,
                                    (cudaStream_t) stream>>>(w, layer, pos0, end, kcur, vcur, n_cells);
    check("append range launch");
}

void kv_exact_window_gather(const KvExactWindow& w, int layer, const int32_t* positions, int64_t n, uint16_t* k_out,
                      uint16_t* v_out, int32_t* hit, void* stream) {
    if (!w.present() || n <= 0) return;
    kv_exact_window_gather_kernel<<<dim3((unsigned) n, (unsigned) w.n_head_kv), (unsigned) w.head_dim, 0,
                              (cudaStream_t) stream>>>(w, layer, positions, n, k_out, v_out, hit);
    check("gather launch");
}

}  // namespace strata::kernels
