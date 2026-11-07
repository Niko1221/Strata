// Grouped-query decode attention, F32/F16 K/V with FP32 accumulation. A block shares each
// coalesced K/V tile between the query heads of one KV head (up to eight heads).
#pragma once
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cmath>
#include <cstddef>
#include <cstdint>

namespace q36attn {

__device__ __forceinline__ float warp_sum(float x) {
#pragma unroll
    for (int d = 16; d; d >>= 1) x += __shfl_xor_sync(0xffffffffu, x, d, 32);
    return x;
}
__device__ __forceinline__ float warp_max(float x) {
#pragma unroll
    for (int d = 16; d; d >>= 1) x = fmaxf(x, __shfl_xor_sync(0xffffffffu, x, d, 32));
    return x;
}

// Same partial-output layout as k_attn_part, so the existing stable merge applies.
// Grid (ceil(context/256), query_tokens * KV_heads), 256 threads. Rows is RowSel:
// its optional cell/slot/cache tables also cover batched sessions and MTP rows.
template<int Tile = 8, class KV, class Rows>
__global__ void part(const float* q, const KV* kc, const KV* vc, const int32_t* pos0,
                     int t0, int H, int HKV, int ctx, float scale, int chunks,
                     float* pacc, float* pm, float* pl, Rows rs) {
    constexpr int D = 256, C = 256;
    __shared__ float tile[Tile * D];
    __shared__ float scores[8 * C];
    const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
    const int t = t0 + blockIdx.y / HKV, kvh = blockIdx.y % HKV, group = H / HKV;
    const int head = kvh * group + warp;
    const int npos = (rs.cell ? rs.cell[t] : *pos0 + t) + 1;
    const int p0 = blockIdx.x * C;
    if (p0 >= npos) return;
    if (rs.kc) {
        kc = reinterpret_cast<const KV*>(rs.kc[rs.slot[t]]);
        vc = reinterpret_cast<const KV*>(rs.vc[rs.slot[t]]);
    }
    const int n = min(C, npos - p0);
    const size_t base = ((size_t) kvh * ctx + p0) * D;
    float qr[8], acc[8] = {};
    if (warp < group) {
#pragma unroll
        for (int i = 0; i < 8; ++i) qr[i] = q[((size_t) t * H + head) * D + lane + 32 * i];
    }
    // K is cell-major. All lanes load consecutive dimensions, and all query heads
    // consume the shared tile before the next load overwrites it.
    for (int b = 0; b < n; b += Tile) {
        const int count = min(Tile, n - b);
        for (int k = 0; k < count; ++k) tile[k * D + tid] = (float) kc[base + (size_t) (b + k) * D + tid];
        __syncthreads();
        if (warp < group) {
            for (int k = 0; k < count; ++k) {
                float dot = 0.f;
#pragma unroll
                for (int i = 0; i < 8; ++i) dot = fmaf(qr[i], tile[k * D + lane + 32 * i], dot);
                dot = warp_sum(dot);
                if (lane == 0) scores[warp * C + b + k] = dot * scale;
            }
        }
        __syncthreads();
    }
    float maximum = -INFINITY, denominator = 0.f;
    if (warp < group) {
        for (int k = lane; k < n; k += 32) maximum = fmaxf(maximum, scores[warp * C + k]);
        maximum = warp_max(maximum);
        for (int k = lane; k < n; k += 32) {
            const float e = expf(scores[warp * C + k] - maximum);
            scores[warp * C + k] = e;
            denominator += e;
        }
        denominator = warp_sum(denominator);
    }
    __syncthreads();
    // Reuse the same shared allocation for V. Each KV element is fetched once per
    // tile, rather than once per query head.
    for (int b = 0; b < n; b += Tile) {
        const int count = min(Tile, n - b);
        for (int k = 0; k < count; ++k) tile[k * D + tid] = (float) vc[base + (size_t) (b + k) * D + tid];
        __syncthreads();
        if (warp < group) {
            for (int k = 0; k < count; ++k) {
                const float weight = scores[warp * C + b + k];
#pragma unroll
                for (int i = 0; i < 8; ++i) acc[i] = fmaf(weight, tile[k * D + lane + 32 * i], acc[i]);
            }
        }
        __syncthreads();
    }
    if (warp < group) {
        const size_t out = ((size_t) (t - t0) * H + head) * chunks + blockIdx.x;
#pragma unroll
        for (int i = 0; i < 8; ++i) pacc[out * D + lane + 32 * i] = acc[i];
        if (lane == 0) { pm[out] = maximum; pl[out] = denominator; }
    }
}

} // namespace q36attn
