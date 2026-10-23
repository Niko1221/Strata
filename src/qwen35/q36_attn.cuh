// src/qwen35/q36_attn.cuh - the q36 engine's grouped-query decode attention (#3641's kernel, with the int8 and k8v4
// cells added): F32/F16/int8 K/V with FP32 accumulation. A block shares each coalesced K/V tile between the query
// heads of one KV head (up to eight heads).
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

// The int8 cache (--kv int8): a cell is 256 int8 values then 8 fp32 scales, q8_0's grid (per 32 values
// d = amax / 127), cells 288 bytes apart (kv_cell_bytes / kv8_store in q36_engine.cu). KV = q8_cell reads it.
struct __align__(16) q8_cell {
    int8_t q[256];
    float d[8];
};
static_assert(sizeof(q8_cell) == 288, "q8_cell is kv_cell_bytes(256, true)");

// The k8v4 cache's V (--kv k8v4, the main path's hybrid K8V4): a cell is the head's 256 values after the
// orthonormal Walsh-Hadamard rotation (kv_q4.hpp; spreads outlier channels over every dimension), as q4_0:
// per 32 values d = (the value of largest |x|) / -8, q = clamp(trunc(x / d + 8.5), 0, 15), x ~ (q - 8) d.
// 8 blocks of 16 code bytes (value j of block b in the low nibble of qs[16 b + j], j + 16 in the high one),
// then the 8 fp16 scales: 144 bytes, 16-byte aligned (ggml's 18-byte blocks are not).
struct __align__(16) q4_cell {
    uint8_t qs[128];
    __half d[8];
};
static_assert(sizeof(q4_cell) == 144, "q4_cell is 144 bytes");

// --kv k8v4's V (the main path's hybrid K8V4, kv_q4.hpp): value i of a cell (thread i of a whole 256-thread block)
// rotated by the orthonormal Walsh-Hadamard transform (the low 5 index bits across a warp's lanes, the high 3 through
// `sh`), then q4_0 per 32 values as kv_q4.cu's q4_group, into the q4_cell at r.
__device__ __forceinline__ void kv4_store(uint8_t* r, int i, float x, float* sh) {
    const int lane = i & 31, w = i >> 5;
#pragma unroll
    for (int h = 1; h < 32; h *= 2) {
        const float o = __shfl_xor_sync(0xffffffffu, x, h);
        x = (lane & h) == 0 ? x + o : o - x;
    }
    sh[i] = x;
    __syncthreads();
    float y = 0.f;
#pragma unroll
    for (int u = 0; u < 8; ++u) y += (__popc(u & w) & 1) ? -sh[32 * u + lane] : sh[32 * u + lane];
    x = y * (1.f / 16.f);
    float amax = fabsf(x), mval = x;
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) {
        const float a = __shfl_xor_sync(0xffffffffu, amax, o);
        const float v = __shfl_xor_sync(0xffffffffu, mval, o);
        if (a > amax || (a == amax && v > mval)) { amax = a; mval = v; }
    }
    const float d = mval / -8.0f;
    const float id = d != 0.0f ? 1.0f / d : 0.0f;
    const int qi = __float2int_rz(x * id + 8.5f);
    const unsigned qc = (unsigned) (qi < 0 ? 0 : (qi > 15 ? 15 : qi));
    const unsigned qhi = __shfl_down_sync(0xffffffffu, qc, 16);
    q4_cell* c = reinterpret_cast<q4_cell*>(r);
    if (lane < 16) c->qs[16 * w + lane] = (uint8_t) (qc | (qhi << 4));
    if (lane == 0) c->d[w] = __float2half(d);
}

// Value `dim` of a V cell (still rotated for q4_cell); `blk` = dim / 32.
__device__ __forceinline__ float v_value(const q8_cell& c, int dim, int blk) { return (float) c.q[dim] * c.d[blk]; }
__device__ __forceinline__ float v_value(const q4_cell& c, int dim, int blk) {
    const int j = dim & 31;
    const uint8_t b = c.qs[16 * blk + (j & 15)];
    return (float) ((j < 16 ? b & 15 : b >> 4) - 8) * __half2float(c.d[blk]);
}

// Cells [cell, cell + count) of a K or V cache (cell = KV head * context + position) into tile rows of
// 256 floats. 256 threads.
template<class KV>
__device__ __forceinline__ void load_tile(float* tile, const KV* kv, size_t cell, int count, int tid) {
    for (int k = 0; k < count; ++k) tile[k * 256 + tid] = (float) kv[(cell + k) * 256 + tid];
}
// int8: a warp per cell; a lane takes values 4 lane .. +3 and 128 + 4 lane .. +3, so each 4-byte load is a
// coalesced 128-byte read and each float4 store a conflict-free one.
__device__ __forceinline__ void load_tile(float* tile, const q8_cell* kv, size_t cell, int count, int tid) {
    const int lane = tid & 31;
    for (int k = tid >> 5; k < count; k += 8) {
        const q8_cell& c = kv[cell + k];
#pragma unroll
        for (int half = 0; half < 2; ++half) {
            const int i = half * 128 + 4 * lane;
            const char4 w = *reinterpret_cast<const char4*>(c.q + i);
            const float d = c.d[i >> 5];
            *reinterpret_cast<float4*>(tile + k * 256 + i) = make_float4(w.x * d, w.y * d, w.z * d, w.w * d);
        }
    }
}

// Same partial-output layout as k_attn_part, so the existing stable merge applies.
// Grid (ceil(context/256), query_tokens * KV_heads), 256 threads. Rows is RowSel:
// its optional cell/slot/cache tables also cover batched sessions and MTP rows.
template<int Tile = 8, class KV, class Rows>
__global__ void part(const float* q, const KV* kc, const KV* vc, const int32_t* pos0,
                     int t0, int H, int HKV, int ctx, float scale, int chunks,
                     float* pacc, float* pm, float* pl, Rows rs) {
    constexpr int D = 256, C = 256;
    __shared__ __align__(16) float tile[Tile * D];
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
    const size_t cell = (size_t) kvh * ctx + p0;
    float qr[8], acc[8] = {};
    if (warp < group) {
#pragma unroll
        for (int i = 0; i < 8; ++i) qr[i] = q[((size_t) t * H + head) * D + lane + 32 * i];
    }
    // K is cell-major. All lanes load consecutive dimensions, and all query heads
    // consume the shared tile before the next load overwrites it.
    for (int b = 0; b < n; b += Tile) {
        const int count = min(Tile, n - b);
        load_tile(tile, kc, cell + b, count, tid);
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
        load_tile(tile, vc, cell + b, count, tid);
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

// The int8 cache's own kernel (same grid, block and partial layout as part). part<q8_cell> spends
// its time on the shared tile, which every query head reads in FP32 (8 x the cache's bytes in
// shared-memory traffic). Here no tile is used:
//  - scores: a warp per cell (cells warp, warp + 8, ...); lane owns dims 8 lane .. +7 (one 8-byte
//    load, one scale) and keeps every query head's q for them in registers, so a cell's int8
//    values are read once for all heads; the heads' partial dots are reduced by a transposing
//    butterfly (each step halves the heads a lane carries: 4 + 2 + 1 + 1 + 1 shuffles);
//  - values: a warp per 32 dims (one q8_0 block, one scale per cell), a lane per dim, every
//    head's weight of a cell one broadcast read: the lane's 8 heads accumulate in registers.
// Requires H / HKV <= 8. VC: q8_cell (--kv int8), or q4_cell (--kv k8v4: the partials are then rotated, see
// unrotate_gate).
template<class VC, class Rows>
__global__ void __launch_bounds__(256) part_q8(const float* q, const q8_cell* kc, const VC* vc,
                                               const int32_t* pos0, int t0, int H, int HKV, int ctx,
                                               float scale, int chunks, float* pacc, float* pm, float* pl,
                                               Rows rs) {
    constexpr int D = 256, C = 256, G = 8;
    __shared__ __align__(16) float scores[C * G];   // [cell][head]
    __shared__ float stat[2 * G];                   // per head: max, sum
    const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
    const int t = t0 + blockIdx.y / HKV, kvh = blockIdx.y % HKV, group = H / HKV;
    const int npos = (rs.cell ? rs.cell[t] : *pos0 + t) + 1;
    const int p0 = blockIdx.x * C;
    if (p0 >= npos) return;
    if (rs.kc) {
        kc = reinterpret_cast<const q8_cell*>(rs.kc[rs.slot[t]]);
        vc = reinterpret_cast<const VC*>(rs.vc[rs.slot[t]]);
    }
    const int n = min(C, npos - p0);
    const size_t cell = (size_t) kvh * ctx + p0;
    // the head this lane's reduced sum belongs to: bits 4, 3, 2 of the lane
    const int mine = ((lane >> 4) & 1) * 4 + ((lane >> 3) & 1) * 2 + ((lane >> 2) & 1);
    {
        float qr[G][8];
#pragma unroll
        for (int h = 0; h < G; ++h) {
            const float* qh = q + ((size_t) t * H + kvh * group + h) * D + 8 * lane;
            const float4 a = h < group ? *reinterpret_cast<const float4*>(qh) : make_float4(0.f, 0.f, 0.f, 0.f);
            const float4 b = h < group ? *reinterpret_cast<const float4*>(qh + 4) : make_float4(0.f, 0.f, 0.f, 0.f);
            qr[h][0] = a.x; qr[h][1] = a.y; qr[h][2] = a.z; qr[h][3] = a.w;
            qr[h][4] = b.x; qr[h][5] = b.y; qr[h][6] = b.z; qr[h][7] = b.w;
        }
#pragma unroll 2
        for (int k = warp; k < n; k += 8) {
            const q8_cell& c = kc[cell + k];
            const int2 w = *reinterpret_cast<const int2*>(c.q + 8 * lane);
            const float d = c.d[lane >> 2];
            const char4 lo = *reinterpret_cast<const char4*>(&w.x), hi = *reinterpret_cast<const char4*>(&w.y);
            const float x[8] = {lo.x * d, lo.y * d, lo.z * d, lo.w * d, hi.x * d, hi.y * d, hi.z * d, hi.w * d};
            float v[G];
#pragma unroll
            for (int h = 0; h < G; ++h) {
                v[h] = 0.f;
#pragma unroll
                for (int j = 0; j < 8; ++j) v[h] = fmaf(qr[h][j], x[j], v[h]);
            }
            // xor 16: keep heads 4..7 (upper half of the warp) or 0..3, add the partner's
#pragma unroll
            for (int i = 0; i < 4; ++i) {
                const bool up = lane & 16;
                const float send = up ? v[i] : v[i + 4], keep = up ? v[i + 4] : v[i];
                v[i] = keep + __shfl_xor_sync(0xffffffffu, send, 16, 32);
            }
#pragma unroll
            for (int i = 0; i < 2; ++i) {   // xor 8: heads 2 of the 4 left
                const bool up = lane & 8;
                const float send = up ? v[i] : v[i + 2], keep = up ? v[i + 2] : v[i];
                v[i] = keep + __shfl_xor_sync(0xffffffffu, send, 8, 32);
            }
            {   // xor 4: one head left
                const bool up = lane & 4;
                const float send = up ? v[0] : v[1], keep = up ? v[1] : v[0];
                v[0] = keep + __shfl_xor_sync(0xffffffffu, send, 4, 32);
            }
            v[0] += __shfl_xor_sync(0xffffffffu, v[0], 2, 32);
            v[0] += __shfl_xor_sync(0xffffffffu, v[0], 1, 32);
            if ((lane & 3) == 0) scores[k * G + mine] = v[0] * scale;
        }
    }
    __syncthreads();
    // softmax per head (warp h), as in part
    if (warp < group) {
        float maximum = -INFINITY, denominator = 0.f;
        for (int k = lane; k < n; k += 32) maximum = fmaxf(maximum, scores[k * G + warp]);
        maximum = warp_max(maximum);
        for (int k = lane; k < n; k += 32) {
            const float e = expf(scores[k * G + warp] - maximum);
            scores[k * G + warp] = e;
            denominator += e;
        }
        denominator = warp_sum(denominator);
        if (lane == 0) { stat[warp] = maximum; stat[G + warp] = denominator; }
    }
    __syncthreads();
    // values: warp w takes dims 32 w .. 32 w + 31, i.e. q8_0 block w of every cell
    const int dim = 32 * warp + lane;
    float acc[G] = {};
#pragma unroll 4
    for (int k = 0; k < n; ++k) {
        const float x = v_value(vc[cell + k], dim, warp);
        const float4 a = *reinterpret_cast<const float4*>(scores + k * G);
        const float4 b = *reinterpret_cast<const float4*>(scores + k * G + 4);
        acc[0] = fmaf(a.x, x, acc[0]); acc[1] = fmaf(a.y, x, acc[1]);
        acc[2] = fmaf(a.z, x, acc[2]); acc[3] = fmaf(a.w, x, acc[3]);
        acc[4] = fmaf(b.x, x, acc[4]); acc[5] = fmaf(b.y, x, acc[5]);
        acc[6] = fmaf(b.z, x, acc[6]); acc[7] = fmaf(b.w, x, acc[7]);
    }
#pragma unroll
    for (int h = 0; h < G; ++h) {
        if (h >= group) break;
        const size_t out = ((size_t) (t - t0) * H + kvh * group + h) * chunks + blockIdx.x;
        pacc[out * D + dim] = acc[h];
        if (tid == 0) { pm[out] = stat[h]; pl[out] = stat[G + h]; }
    }
}

// The chunks merged, times sigmoid(gate), as k_attn_merge (same partials, same result up to the
// order of the sums), with the work spread wider: k_attn_merge runs a block per query row (16
// blocks for a decode step) walking every chunk, here a block per (row, 32 dims) has its 8 warps
// take every 8th chunk. Grid (query rows, 8), 256 threads.
template<class Rows>
__global__ void __launch_bounds__(256) merge(const float* pacc, const float* pm, const float* pl,
                                             const float* gate, const int32_t* pos0, int t0, int H,
                                             int chunks, float* out, Rows rs) {
    constexpr int D = 256, C = 256;
    __shared__ float smax[8], sden[8], snum[8][32];
    const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
    const int row = blockIdx.x, t = t0 + row / H, h = row % H, dim = blockIdx.y * 32 + lane;
    const int npos = (rs.cell ? rs.cell[t] : *pos0 + t) + 1;
    const int nc = (npos + C - 1) / C;
    const size_t r0 = (size_t) row * chunks;
    float M = -INFINITY;
    for (int c = tid; c < nc; c += 256) M = fmaxf(M, pm[r0 + c]);
    M = warp_max(M);
    if (lane == 0) smax[warp] = M;
    __syncthreads();
#pragma unroll
    for (int w = 0; w < 8; ++w) M = fmaxf(M, smax[w]);
    float num = 0.f, den = 0.f;
    for (int c = warp; c < nc; c += 8) {
        const float w = expf(pm[r0 + c] - M);
        num += w * pacc[(r0 + c) * D + dim];
        den += w * pl[r0 + c];
    }
    snum[warp][lane] = num;
    if (lane == 0) sden[warp] = den;
    __syncthreads();
    if (warp == 0) {
        num = 0.f; den = 0.f;
#pragma unroll
        for (int w = 0; w < 8; ++w) { num += snum[w][lane]; den += sden[w]; }
        const size_t gi = ((size_t) t * H + h) * D + dim;
        // no gate: k8v4's rotated rows, gated by unrotate_gate
        out[gi] = gate ? num / den * (1.0f / (1.0f + expf(-gate[gi]))) : num / den;
    }
}


// k8v4: attention rows of 256 that mix rotated values, rotated back (the orthonormal Walsh-Hadamard transform
// is its own inverse; same butterflies as kv_q4.cu's fwht256_kernel) and times sigmoid(gate). A warp per row,
// value 32 i + lane in register i. Grid (rows), 32 threads.
__global__ void __launch_bounds__(32) unrotate_gate(float* att, const float* gate) {
    const int lane = threadIdx.x;
    float* row = att + (size_t) blockIdx.x * 256;
    const float* g = gate + (size_t) blockIdx.x * 256;
    float r[8];
#pragma unroll
    for (int i = 0; i < 8; ++i) r[i] = row[32 * i + lane] * (1.f / 16.f);
#pragma unroll
    for (int h = 1; h < 32; h *= 2)
#pragma unroll
        for (int i = 0; i < 8; ++i) {
            const float o = __shfl_xor_sync(0xffffffffu, r[i], h, 32);
            r[i] = (lane & h) == 0 ? r[i] + o : o - r[i];
        }
#pragma unroll
    for (int step = 1; step < 8; step *= 2)
#pragma unroll
        for (int i = 0; i < 8; ++i)
            if ((i & step) == 0) {
                const float x = r[i], y = r[i + step];
                r[i] = x + y;
                r[i + step] = x - y;
            }
#pragma unroll
    for (int i = 0; i < 8; ++i) row[32 * i + lane] = r[i] * (1.0f / (1.0f + expf(-g[32 * i + lane])));
}

} // namespace q36attn
