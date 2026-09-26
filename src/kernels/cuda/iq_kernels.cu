// src/kernels/cuda/iq_kernels.cu - see include/strata/kernels/iq_kernels.hpp.
//
// The dequantizers and the q8_1 quantizer are transcribed from llama.cpp (ggml/src/ggml-cuda/dequantize.cuh,
// quantize.cu at the commit in third_party/ggml/VERSION.txt; MIT license, third_party/ggml/LICENSE), the dot products
// in iq_dot.cuh.
#include "strata/kernels/iq_kernels.hpp"
#include "iq_dot.cuh"

#include <cstdio>
#include <cstdlib>

namespace strata::kernels {
namespace {

void check(const char* what) {
    const cudaError_t e = cudaGetLastError();
    if (e != cudaSuccess) { std::fprintf(stderr, "%s: %s\n", what, cudaGetErrorString(e)); std::exit(1); }
}

template<int TY>
__global__ void __launch_bounds__(128) mmvq_kernel(const uint8_t* __restrict__ w, size_t row_bytes,
                                                   const block_q8_1* __restrict__ x, float* __restrict__ y, int n_in,
                                                   int n_out, int ncols) {
    const int row = blockIdx.x * 4 + threadIdx.y;
    if (row >= n_out) return;
    const int lane = threadIdx.x;
    const int nb = n_in / Fmt<TY>::qk;
    const uint8_t* wr = w + (size_t) row * row_bytes;
    for (int c = 0; c < ncols; ++c) {
        const float s = row_dot<TY>(wr, x + (size_t) c * (n_in / 32), nb, lane);
        if (lane == 0) y[(size_t) c * n_out + row] = s;
    }
}

// ---------------------------------------------------------------- grouped native experts
// Two kernels a call.  The first computes a group's gate and up rows for one chunk of 32 rows of n_ff - one q8_1 block
// of h - SwiGLU and that block for each entry; the second the down rows.  A warp takes several rows at once, one
// lane-strided accumulator each, so every value's sums and their order are `row_dot`'s.
constexpr int GUS_ROWS = 2, GUS_WARPS = 16;   // gate and up rows per warp; 8 warps x 4 rows = one q8_1 block
constexpr int GUS_BATCH = 8;                 // entries whose block of h is staged at once
constexpr int DOWN_ROWS = 4, DOWN_WARPS = 8;

// The K-quants' gate/up rows need more registers than the i-quants': with one block per SM they do not spill (Q4_K, Q5_K
// 15-17% faster at 2-4 tokens), while the i-quants run best at the occupancy the compiler picks (0: no bound).
template<int T> constexpr int gus_min_blocks() { return T == 12 || T == 13 || T == 14 ? 1 : 0; }

template<int TG>
__global__ void __launch_bounds__(GUS_WARPS * 32, gus_min_blocks<TG>()) native_gus_kernel(const unsigned long long* __restrict__ grp_ptr,
                                                                    const int32_t* __restrict__ grp_start,
                                                                    const int32_t* __restrict__ n_groups,
                                                                    const int32_t* __restrict__ ent_tok,
                                                                    const block_q8_1* __restrict__ xq,
                                                                    NativeExpertLayout L, block_q8_1* __restrict__ hq) {
    using F = Fmt<TG>;
    const int g = blockIdx.y;
    if (g >= *n_groups) return;
    __shared__ float sh[GUS_BATCH][32];
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31, c = blockIdx.x;
    const uint8_t* wg = (const uint8_t*) grp_ptr[g] + (size_t) (32 * c + GUS_ROWS * warp) * L.gu_row;
    const uint8_t* wu = wg + L.up_off;
    const int nb = (int) (L.n_embd / F::qk), xb = (int) (L.n_embd / 32), hb = (int) (L.n_ff / 32);
    const int e0 = grp_start[g], e1 = grp_start[g + 1];
    for (int b0 = e0; b0 < e1; b0 += GUS_BATCH) {
        const int b1 = min(e1, b0 + GUS_BATCH);
        for (int e = b0; e < b1; ++e) {
            const block_q8_1* x = xq + (size_t) ent_tok[e] * xb;
            float sg[GUS_ROWS], su[GUS_ROWS];
#pragma unroll
            for (int i = 0; i < GUS_ROWS; ++i) sg[i] = su[i] = 0.0f;
            for (int k = lane; k < nb * F::ipb; k += 32) {
                const int kbx = k / F::ipb, iqs = F::step * (k % F::ipb);
                const block_q8_1* xk = x + kbx * (F::qk / 32);
#pragma unroll
                for (int i = 0; i < GUS_ROWS; ++i) {
                    sg[i] += F::dot(wg + (size_t) i * L.gu_row, xk, kbx, iqs);
                    su[i] += F::dot(wu + (size_t) i * L.gu_row, xk, kbx, iqs);
                }
            }
#pragma unroll
            for (int i = 0; i < GUS_ROWS; ++i) {
                const float gv = warp_sum(sg[i]), uv = warp_sum(su[i]);
                if (lane == 0) sh[e - b0][GUS_ROWS * warp + i] = (gv / (1.0f + __expf(-gv))) * uv;
            }
        }
        __syncthreads();
        for (int e = b0 + warp; e < b1; e += GUS_WARPS) {   // an entry's block of h, as quantize_q8_1_kernel
            const float xi = sh[e - b0][lane];
            float amax = fabsf(xi), sum = xi;
#pragma unroll
            for (int o = 16; o > 0; o >>= 1) {
                amax = fmaxf(amax, __shfl_xor_sync(0xffffffffu, amax, o));
                sum += __shfl_xor_sync(0xffffffffu, sum, o);
            }
            const float d = amax / 127.0f;
            const int8_t q = amax == 0.0f ? 0 : roundf(xi / d);
            block_q8_1* y = hq + (size_t) e * hb + c;
            y->qs[lane] = q;
            if (lane == 0) y->ds = make_half2(d, sum);
        }
        __syncthreads();
    }
}

template<int TD>
__global__ void __launch_bounds__(DOWN_WARPS * 32) native_down_kernel(const unsigned long long* __restrict__ grp_ptr,
                                                                      const int32_t* __restrict__ grp_start,
                                                                      const int32_t* __restrict__ n_groups,
                                                                      const int32_t* __restrict__ ent_dst,
                                                                      const block_q8_1* __restrict__ hq,
                                                                      NativeExpertLayout L, float* __restrict__ out) {
    using F = Fmt<TD>;
    const int g = blockIdx.y;
    if (g >= *n_groups) return;
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const int r0 = (blockIdx.x * DOWN_WARPS + warp) * DOWN_ROWS;
    const uint8_t* wr = (const uint8_t*) grp_ptr[g] + L.down_off + (size_t) r0 * L.d_row;
    const int nb = (int) (L.n_ff / F::qk), hb = (int) (L.n_ff / 32);
    const int e0 = grp_start[g], e1 = grp_start[g + 1];
    for (int e = e0; e < e1; ++e) {
        const block_q8_1* x = hq + (size_t) e * hb;
        float sd[DOWN_ROWS];
#pragma unroll
        for (int i = 0; i < DOWN_ROWS; ++i) sd[i] = 0.0f;
        for (int k = lane; k < nb * F::ipb; k += 32) {
            const int kbx = k / F::ipb, iqs = F::step * (k % F::ipb);
            const block_q8_1* xk = x + kbx * (F::qk / 32);
#pragma unroll
            for (int i = 0; i < DOWN_ROWS; ++i) sd[i] += F::dot(wr + (size_t) i * L.d_row, xk, kbx, iqs);
        }
        float v[DOWN_ROWS];
#pragma unroll
        for (int i = 0; i < DOWN_ROWS; ++i) v[i] = warp_sum(sd[i]);
        if (lane == 0) {
            float* o = out + (size_t) ent_dst[e] * L.n_embd + r0;
            if constexpr (DOWN_ROWS == 4) *(float4*) o = make_float4(v[0], v[1], v[2], v[3]);
            else if constexpr (DOWN_ROWS == 2) *(float2*) o = make_float2(v[0], v[1]);
            else
#pragma unroll
                for (int i = 0; i < DOWN_ROWS; ++i) o[i] = v[i];
        }
    }
}

__global__ void rows_out_kernel(const float4* __restrict__ src, const int32_t* __restrict__ n_groups,
                                const int32_t* __restrict__ grp_start, long long n4, float4* __restrict__ dst) {
    const long long total = (long long) grp_start[*n_groups] * n4;
    for (long long i = (long long) blockIdx.x * blockDim.x + threadIdx.x; i < total; i += (long long) gridDim.x * blockDim.x)
        dst[i] = src[i];
}

// ---------------------------------------------------------------- q8_1 (quantize.cu)
__global__ void quantize_q8_1_kernel(const float* __restrict__ x, block_q8_1* __restrict__ y, long long n) {
    const long long i = (long long) blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const float xi = x[i];
    float amax = fabsf(xi), sum = xi;
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) {
        amax = fmaxf(amax, __shfl_xor_sync(0xffffffffu, amax, o));
        sum += __shfl_xor_sync(0xffffffffu, sum, o);
    }
    const float d = amax / 127.0f;
    const int8_t q = amax == 0.0f ? 0 : roundf(xi / d);
    const long long ib = i / 32, iqs = i % 32;
    y[ib].qs[iqs] = q;
    if (iqs == 0) y[ib].ds = make_half2(d, sum);
}

// ---------------------------------------------------------------- dequant (dequantize.cuh)
template<typename dst_t> __device__ __forceinline__ dst_t cvt(float v);
template<> __device__ __forceinline__ float cvt<float>(float v) { return v; }
template<> __device__ __forceinline__ __half cvt<__half>(float v) { return __float2half(v); }

template<typename dst_t>
__device__ void dq_iq2_xxs(const void* vx, int64_t ibs, dst_t* yy, int tid) {
    const block_iq2_xxs* x = (const block_iq2_xxs*) vx;
    const int64_t il = tid / 8, ib = tid % 8;
    dst_t* y = yy + 32 * ib + 8 * il;
    const uint16_t* q2 = x[ibs].qs + 4 * ib;
    const uint8_t* aux8 = (const uint8_t*) q2;
    const uint8_t* grid = (const uint8_t*) (iq2xxs_grid + aux8[il]);
    const uint32_t aux32 = q2[2] | (q2[3] << 16);
    const float d = (float) x[ibs].d * (0.5f + (aux32 >> 28)) * 0.25f;
    const uint8_t signs = ksigns_iq2xs[(aux32 >> 7 * il) & 127];
    for (int j = 0; j < 8; ++j) y[j] = cvt<dst_t>(d * grid[j] * (signs & kmask_iq2xs[j] ? -1.f : 1.f));
}
template<typename dst_t>
__device__ void dq_iq2_xs(const void* vx, int64_t ibs, dst_t* yy, int tid) {
    const block_iq2_xs* x = (const block_iq2_xs*) vx;
    const int64_t il = tid / 8, ib = tid % 8;
    dst_t* y = yy + 32 * ib + 8 * il;
    const uint16_t* q2 = x[ibs].qs + 4 * ib;
    const uint8_t* grid = (const uint8_t*) (iq2xs_grid + (q2[il] & 511));
    const float d = (float) x[ibs].d * (0.5f + ((x[ibs].scales[ib] >> 4 * (il / 2)) & 0xf)) * 0.25f;
    const uint8_t signs = ksigns_iq2xs[q2[il] >> 9];
    for (int j = 0; j < 8; ++j) y[j] = cvt<dst_t>(d * grid[j] * (signs & kmask_iq2xs[j] ? -1.f : 1.f));
}
template<typename dst_t>
__device__ void dq_iq2_s(const void* vx, int64_t ibs, dst_t* yy, int tid) {
    const block_iq2_s* x = (const block_iq2_s*) vx;
    const int64_t il = tid / 8, ib = tid % 8;
    dst_t* y = yy + 32 * ib + 8 * il;
    const uint8_t* grid = (const uint8_t*) (iq2s_grid + (x[ibs].qs[4 * ib + il] | ((x[ibs].qh[ib] << (8 - 2 * il)) & 0x300)));
    const float d = (float) x[ibs].d * (0.5f + ((x[ibs].scales[ib] >> 4 * (il / 2)) & 0xf)) * 0.25f;
    const uint8_t signs = x[ibs].qs[QK_K / 8 + 4 * ib + il];
    for (int j = 0; j < 8; ++j) y[j] = cvt<dst_t>(d * grid[j] * (signs & kmask_iq2xs[j] ? -1.f : 1.f));
}
template<typename dst_t>
__device__ void dq_iq3_xxs(const void* vx, int64_t ibs, dst_t* yy, int tid) {
    const block_iq3_xxs* x = (const block_iq3_xxs*) vx;
    const int64_t il = tid / 8, ib = tid % 8;
    dst_t* y = yy + 32 * ib + 8 * il;
    const uint8_t* q3 = x[ibs].qs + 8 * ib;
    const uint16_t* gas = (const uint16_t*) (x[ibs].qs + QK_K / 4) + 2 * ib;
    const uint8_t* grid1 = (const uint8_t*) (iq3xxs_grid + q3[2 * il + 0]);
    const uint8_t* grid2 = (const uint8_t*) (iq3xxs_grid + q3[2 * il + 1]);
    const uint32_t aux32 = gas[0] | (gas[1] << 16);
    const float d = (float) x[ibs].d * (0.5f + (aux32 >> 28)) * 0.5f;
    const uint8_t signs = ksigns_iq2xs[(aux32 >> 7 * il) & 127];
    for (int j = 0; j < 4; ++j) {
        y[j + 0] = cvt<dst_t>(d * grid1[j] * (signs & kmask_iq2xs[j + 0] ? -1.f : 1.f));
        y[j + 4] = cvt<dst_t>(d * grid2[j] * (signs & kmask_iq2xs[j + 4] ? -1.f : 1.f));
    }
}
template<typename dst_t>
__device__ void dq_iq3_s(const void* vx, int64_t ibs, dst_t* yy, int tid) {
    const block_iq3_s* x = (const block_iq3_s*) vx;
    const int64_t il = tid / 8, ib = tid % 8;
    dst_t* y = yy + 32 * ib + 8 * il;
    const uint8_t* qs = x[ibs].qs + 8 * ib;
    const uint8_t* grid1 = (const uint8_t*) (iq3s_grid + (qs[2 * il + 0] | ((x[ibs].qh[ib] << (8 - 2 * il)) & 256)));
    const uint8_t* grid2 = (const uint8_t*) (iq3s_grid + (qs[2 * il + 1] | ((x[ibs].qh[ib] << (7 - 2 * il)) & 256)));
    const float d = (float) x[ibs].d * (1 + 2 * ((x[ibs].scales[ib / 2] >> 4 * (ib % 2)) & 0xf));
    const uint8_t signs = x[ibs].signs[4 * ib + il];
    for (int j = 0; j < 4; ++j) {
        y[j + 0] = cvt<dst_t>(d * grid1[j] * (signs & kmask_iq2xs[j + 0] ? -1.f : 1.f));
        y[j + 4] = cvt<dst_t>(d * grid2[j] * (signs & kmask_iq2xs[j + 4] ? -1.f : 1.f));
    }
}
template<typename dst_t>
__device__ void dq_iq1_m(const void* vx, int64_t ibs, dst_t* yy, int tid) {
    const block_iq1_m* x = (const block_iq1_m*) vx;
    const int64_t il = tid / 8, ib = tid % 8;
    dst_t* y = yy + 32 * ib + 8 * il;
    const uint16_t* sc = (const uint16_t*) x[ibs].scales;
    iq1m_scale_t scale;
    scale.u16 = (sc[0] >> 12) | ((sc[1] >> 8) & 0x00f0) | ((sc[2] >> 4) & 0x0f00) | (sc[3] & 0xf000);
    const int64_t ib16 = 2 * ib + il / 2;
    const float d = (float) scale.f16 * (2 * ((sc[ib16 / 4] >> 3 * (ib16 % 4)) & 0x7) + 1);
    const float delta = x[ibs].qh[2 * ib + il / 2] & (0x08 << 4 * (il % 2)) ? -1 - IQ1M_DELTA : -1 + IQ1M_DELTA;
    uint32_t grid32[2];
    const int8_t* q = (const int8_t*) grid32;
    grid32[0] = iq1s_grid_gpu[x[ibs].qs[4 * ib + il] | (((x[ibs].qh[2 * ib + il / 2] >> 4 * (il % 2)) & 7) << 8)];
    grid32[1] = (grid32[0] >> 4) & 0x0f0f0f0f;
    grid32[0] &= 0x0f0f0f0f;
    for (int j = 0; j < 8; ++j) y[j] = cvt<dst_t>(d * (q[j] + delta));
}
template<typename dst_t>
__device__ void dq_iq4_nl(const void* vx, int64_t ibs, dst_t* yy, int tid) {
    const block_iq4_nl* x = (const block_iq4_nl*) vx + ibs * (QK_K / QK4_NL);
    const int64_t il = tid / 8, ib = tid % 8;
    dst_t* y = yy + 32 * ib + 4 * il;
    const uint8_t* q4 = x[ib].qs + 4 * il;
    const float d = (float) x[ib].d;
    for (int j = 0; j < 4; ++j) {
        y[j + 0] = cvt<dst_t>(d * kvalues_iq4nl[q4[j] & 0xf]);
        y[j + 16] = cvt<dst_t>(d * kvalues_iq4nl[q4[j] >> 4]);
    }
}
// Q3_K (the Q2_0 file's token_embd): llama.cpp's dequantize_block_q3_K, its 64 threads folded onto 32
template<typename dst_t>
__device__ void dq_q3_k(const void* vx, int64_t ibs, dst_t* yy, int tid) {
    const block_q3_K* x = (const block_q3_K*) vx + ibs;
    for (int tt = tid; tt < 64; tt += 32) {
        const int r = tt / 4, t2 = r / 2, is0 = r % 2;
        const int l0 = 16 * is0 + 4 * (tt % 4);
        const int n = t2 / 4, j = t2 - 4 * n;
        const uint8_t m = (uint8_t) (1 << (4 * n + j));
        const int is = 8 * n + 2 * j + is0;
        const int shift = 2 * j;
        const int8_t us = is < 4  ? (int8_t) ((x->scales[is - 0] & 0xF) | (((x->scales[is + 8] >> 0) & 3) << 4)) :
                          is < 8  ? (int8_t) ((x->scales[is - 0] & 0xF) | (((x->scales[is + 4] >> 2) & 3) << 4)) :
                          is < 12 ? (int8_t) ((x->scales[is - 8] >> 4) | (((x->scales[is + 0] >> 4) & 3) << 4)) :
                                    (int8_t) ((x->scales[is - 8] >> 4) | (((x->scales[is - 4] >> 6) & 3) << 4));
        const float dl = (float) x->d * (us - 32);
        dst_t* y = yy + 128 * n + 32 * j;
        const uint8_t* q = x->qs + 32 * n;
        const uint8_t* hm = x->hmask;
        for (int l = l0; l < l0 + 4; ++l) y[l] = cvt<dst_t>(dl * ((int8_t) ((q[l] >> shift) & 3) - ((hm[l] & m) ? 0 : 4)));
    }
}
template<typename dst_t>
__device__ void dq_iq4_xs(const void* vx, int64_t ibs, dst_t* yy, int tid) {
    const block_iq4_xs* x = (const block_iq4_xs*) vx + ibs;
    const int il = tid / 8, ib = tid % 8;
    dst_t* y = yy + 32 * ib + 4 * il;
    const uint8_t* q4 = x->qs + 16 * ib + 4 * il;
    const float d = (float) x->d * ((((x->scales_l[ib / 2] >> 4 * (ib % 2)) & 0xf) | (((x->scales_h >> 2 * ib) & 3) << 4)) - 32);
    for (int j = 0; j < 4; ++j) {
        y[j + 0] = cvt<dst_t>(d * kvalues_iq4nl[q4[j] & 0xf]);
        y[j + 16] = cvt<dst_t>(d * kvalues_iq4nl[q4[j] >> 4]);
    }
}
template<typename dst_t>
__device__ void dq_q2_0(const void* vx, int64_t ibs, dst_t* yy, int tid) {
    // one "superblock" = 256 values = 4 blocks of 64; thread tid writes 8 values
    const block_q2_0* x = (const block_q2_0*) vx + ibs * 4;
    const int b = tid / 8, part = tid % 8;          // block 0..3, 8 values each
    const float d = (float) x[b].d;
    for (int j = 0; j < 8; ++j) {
        const int i = part * 8 + j;
        const int code = (x[b].qs[i / 4] >> ((i % 4) * 2)) & 3;
        yy[b * 64 + i] = cvt<dst_t>(d * (float) (code - 1));
    }
}
__device__ __forceinline__ void scale_min_k4(int j, const uint8_t* q, uint8_t& d, uint8_t& m) {
    if (j < 4) {
        d = q[j] & 63; m = q[j + 4] & 63;
    } else {
        d = (q[j + 4] & 0xF) | ((q[j - 4] >> 6) << 4);
        m = (q[j + 4] >> 4) | ((q[j - 0] >> 6) << 4);
    }
}
template<typename dst_t>
__device__ void dq_q4_k(const void* vx, int64_t ibs, dst_t* yy, int tid) {
    const block_q4_K* x = (const block_q4_K*) vx;
    const int64_t il = tid / 8, ir = tid % 8, is = 2 * il;
    const int n = 4;
    dst_t* y = yy + 64 * il + n * ir;
    const float dall = __low2half(x[ibs].dm);
    const float dmin = __high2half(x[ibs].dm);
    const uint8_t* q = x[ibs].qs + 32 * il + n * ir;
    uint8_t sc, m;
    scale_min_k4((int) is + 0, x[ibs].scales, sc, m);
    const float d1 = dall * sc, m1 = dmin * m;
    scale_min_k4((int) is + 1, x[ibs].scales, sc, m);
    const float d2 = dall * sc, m2 = dmin * m;
    for (int l = 0; l < n; ++l) {
        y[l + 0] = cvt<dst_t>(d1 * (q[l] & 0xF) - m1);
        y[l + 32] = cvt<dst_t>(d2 * (q[l] >> 4) - m2);
    }
}
// llama.cpp's dequantize_q5_K, its 64 threads folded onto 32
template<typename dst_t>
__device__ void dq_q5_k(const void* vx, int64_t ibs, dst_t* yy, int tid) {
    const block_q5_K* x = (const block_q5_K*) vx;
    for (int tt = tid; tt < 64; tt += 32) {
        const int il = tt / 16, ir = tt % 16, is = 2 * il;
        dst_t* y = yy + 64 * il + 2 * ir;
        const float dall = __low2half(x[ibs].dm);
        const float dmin = __high2half(x[ibs].dm);
        const uint8_t* ql = x[ibs].qs + 32 * il + 2 * ir;
        const uint8_t* qh = x[ibs].qh + 2 * ir;
        uint8_t sc, m;
        scale_min_k4(is + 0, x[ibs].scales, sc, m);
        const float d1 = dall * sc, m1 = dmin * m;
        scale_min_k4(is + 1, x[ibs].scales, sc, m);
        const float d2 = dall * sc, m2 = dmin * m;
        uint8_t hm = (uint8_t) (1 << (2 * il));
        y[0] = cvt<dst_t>(d1 * ((ql[0] & 0xF) + (qh[0] & hm ? 16 : 0)) - m1);
        y[1] = cvt<dst_t>(d1 * ((ql[1] & 0xF) + (qh[1] & hm ? 16 : 0)) - m1);
        hm <<= 1;
        y[32] = cvt<dst_t>(d2 * ((ql[0] >> 4) + (qh[0] & hm ? 16 : 0)) - m2);
        y[33] = cvt<dst_t>(d2 * ((ql[1] >> 4) + (qh[1] & hm ? 16 : 0)) - m2);
    }
}
// 32-value blocks: a "superblock" is 8 of them; thread tid writes 8 values of block tid % 8
template<typename dst_t>
__device__ void dq_q5_1(const void* vx, int64_t ibs, dst_t* yy, int tid) {
    const block_q5_1* x = (const block_q5_1*) vx + ibs * (QK_K / QK5_1);
    const int ib = tid % 8, il = tid / 8;
    const float2 dm = __half22float2(x[ib].dm);
    uint32_t qh;
    memcpy(&qh, x[ib].qh, sizeof(qh));
    dst_t* y = yy + 32 * ib;
    for (int j = 0; j < 4; ++j) {
        const int iqs = 4 * il + j;
        const int xh_0 = ((qh >> (iqs + 0)) << 4) & 0x10;
        const int xh_1 = ((qh >> (iqs + 12))) & 0x10;
        y[iqs] = cvt<dst_t>((float) ((x[ib].qs[iqs] & 0xf) | xh_0) * dm.x + dm.y);
        y[iqs + 16] = cvt<dst_t>((float) ((x[ib].qs[iqs] >> 4) | xh_1) * dm.x + dm.y);
    }
}
template<typename dst_t>
__device__ void dq_q8_0(const void* vx, int64_t ibs, dst_t* yy, int tid) {
    const block_q8_0* x = (const block_q8_0*) vx + ibs * (QK_K / QK8_0);
    const int ib = tid % 8, il = tid / 8;
    const float d = __half2float(x[ib].d);
    dst_t* y = yy + 32 * ib + 8 * il;
    for (int j = 0; j < 8; ++j) y[j] = cvt<dst_t>((float) x[ib].qs[8 * il + j] * d);
}
// llama.cpp's dequantize_q6_K, its 64 threads folded onto 32
template<typename dst_t>
__device__ void dq_q6_k(const void* vx, int64_t ibs, dst_t* yy, int tid) {
    const block_q6_K* x = (const block_q6_K*) vx;
    for (int tt = tid; tt < 64; tt += 32) {
        const int ip = tt / 32, il = tt - 32 * ip, is = 8 * ip + il / 16;
        dst_t* y = yy + 128 * ip + il;
        const float d = __half2float(x[ibs].d);
        const uint8_t* ql = x[ibs].ql + 64 * ip + il;
        const uint8_t qh = x[ibs].qh[32 * ip + il];
        const int8_t* sc = x[ibs].scales + is;
        y[0] = cvt<dst_t>(d * sc[0] * ((int8_t) ((ql[0] & 0xF) | (((qh >> 0) & 3) << 4)) - 32));
        y[32] = cvt<dst_t>(d * sc[2] * ((int8_t) ((ql[32] & 0xF) | (((qh >> 2) & 3) << 4)) - 32));
        y[64] = cvt<dst_t>(d * sc[4] * ((int8_t) ((ql[0] >> 4) | (((qh >> 4) & 3) << 4)) - 32));
        y[96] = cvt<dst_t>(d * sc[6] * ((int8_t) ((ql[32] >> 4) | (((qh >> 6) & 3) << 4)) - 32));
    }
}
template<typename dst_t>
__device__ void dq_iq1_s(const void* vx, int64_t ibs, dst_t* yy, int tid) {
    const block_iq1_s* x = (const block_iq1_s*) vx;
    const int64_t il = tid / 8, ib = tid % 8;
    dst_t* y = yy + 32 * ib + 8 * il;
    const float delta = x[ibs].qh[ib] & 0x8000 ? -1 - IQ1S_DELTA : -1 + IQ1S_DELTA;
    const float d = (float) x[ibs].d * (2 * ((x[ibs].qh[ib] >> 12) & 7) + 1);
    uint32_t grid32[2];
    const int8_t* q = (const int8_t*) grid32;
    grid32[0] = iq1s_grid_gpu[x[ibs].qs[4 * ib + il] | (((x[ibs].qh[ib] >> 3 * il) & 7) << 8)];
    grid32[1] = (grid32[0] >> 4) & 0x0f0f0f0f;
    grid32[0] &= 0x0f0f0f0f;
    for (int j = 0; j < 8; ++j) y[j] = cvt<dst_t>(d * (q[j] + delta));
}

template<typename dst_t>
__device__ __forceinline__ void dq_dispatch(int ty, const void* vx, int64_t ibs, dst_t* y, int tid) {
    switch (ty) {
        case 16: dq_iq2_xxs(vx, ibs, y, tid); break;
        case 17: dq_iq2_xs(vx, ibs, y, tid); break;
        case 18: dq_iq3_xxs(vx, ibs, y, tid); break;
        case 20: dq_iq4_nl(vx, ibs, y, tid); break;
        case 21: dq_iq3_s(vx, ibs, y, tid); break;
        case 22: dq_iq2_s(vx, ibs, y, tid); break;
        case 29: dq_iq1_m(vx, ibs, y, tid); break;
        case 23: dq_iq4_xs(vx, ibs, y, tid); break;
        case 11: dq_q3_k(vx, ibs, y, tid); break;
        case 42: dq_q2_0(vx, ibs, y, tid); break;
        case 12: dq_q4_k(vx, ibs, y, tid); break;
        case 13: dq_q5_k(vx, ibs, y, tid); break;
        case 7: dq_q5_1(vx, ibs, y, tid); break;
        case 8: dq_q8_0(vx, ibs, y, tid); break;
        case 14: dq_q6_k(vx, ibs, y, tid); break;
        case 19: dq_iq1_s(vx, ibs, y, tid); break;
        default: break;
    }
}

// flat: superblock i -> y + 256 i
template<typename dst_t>
__global__ void dequant_flat_kernel(int ty, const void* __restrict__ vx, dst_t* __restrict__ y) {
    const int64_t i = blockIdx.x;
    dq_dispatch<dst_t>(ty, vx, i, y + i * QK_K, threadIdx.x);
}
// gate/up: superblock i of a role matrix (n_embd/256 per row) -> interleaved row 2r + parity
__global__ void dequant_gu_kernel(int ty, const void* __restrict__ gate, const void* __restrict__ up, int64_t per_row,
                                  __half* __restrict__ y) {
    const int64_t i = blockIdx.x;
    const int parity = blockIdx.y;
    const int64_t r = i / per_row, c = i % per_row;
    dq_dispatch<__half>(ty, parity ? up : gate, i, y + ((2 * r + parity) * per_row + c) * QK_K, threadIdx.x);
}

// several experts at once, expert z's blob at blobs[z]: first its gate/up superblocks (as dequant_gu_kernel), then
// its down superblocks (as dequant_flat_kernel)
__global__ void dequant_experts_kernel(int gu_ty, int d_ty, const uint8_t* const* __restrict__ blobs, size_t up_off,
                                       size_t down_off, int64_t per_row, int64_t gu_blocks, __half* __restrict__ gu,
                                       __half* __restrict__ dn, int64_t gu_elems, int64_t dn_elems) {
    const uint8_t* b = blobs[blockIdx.z];
    const int64_t i = blockIdx.x;
    if (i < 2 * gu_blocks) {
        const int parity = (int) (i / gu_blocks);
        const int64_t j = i % gu_blocks, r = j / per_row, c = j % per_row;
        dq_dispatch<__half>(gu_ty, parity ? b + up_off : b, j,
                            gu + blockIdx.z * gu_elems + ((2 * r + parity) * per_row + c) * QK_K, threadIdx.x);
    } else {
        const int64_t j = i - 2 * gu_blocks;
        dq_dispatch<__half>(d_ty, b + down_off, j, dn + blockIdx.z * dn_elems + j * QK_K, threadIdx.x);
    }
}

// the formats dq_dispatch dequantizes
bool is_iq(int t) {
    return t == 16 || t == 17 || t == 18 || t == 20 || t == 21 || t == 22 || t == 23 || t == 29 || t == 42 || t == 11 ||
           t == 12 || t == 13 || t == 14 || t == 19 || t == 7 || t == 8;
}

}  // namespace

bool iq_supported(int t) noexcept { return is_iq(t); }

size_t iq_row_bytes(int t, int64_t n) noexcept {
    switch (t) {
        case 16: return (size_t) (n / 256) * sizeof(block_iq2_xxs);
        case 17: return (size_t) (n / 256) * sizeof(block_iq2_xs);
        case 18: return (size_t) (n / 256) * sizeof(block_iq3_xxs);
        case 20: return (size_t) (n / 32) * sizeof(block_iq4_nl);
        case 21: return (size_t) (n / 256) * sizeof(block_iq3_s);
        case 22: return (size_t) (n / 256) * sizeof(block_iq2_s);
        case 29: return (size_t) (n / 256) * sizeof(block_iq1_m);
        case 23: return (size_t) (n / 256) * sizeof(block_iq4_xs);
        case 11: return (size_t) (n / 256) * sizeof(block_q3_K);
        case 42: return (size_t) (n / 64) * sizeof(block_q2_0);
        case 12: return (size_t) (n / 256) * sizeof(block_q4_K);
        case 13: return (size_t) (n / 256) * sizeof(block_q5_K);
        case 14: return (size_t) (n / 256) * sizeof(block_q6_K);
        case 19: return (size_t) (n / 256) * sizeof(block_iq1_s);
        case 7: return (size_t) (n / 32) * sizeof(block_q5_1);
        case 8: return (size_t) (n / 32) * sizeof(block_q8_0);
        default: return 0;
    }
}

void quantize_q8_1_rows(const float* x, int64_t n_rows, int64_t n_cols, void* y, void* stream) {
    const long long n = (long long) n_rows * n_cols;
    if (n <= 0) return;
    quantize_q8_1_kernel<<<(unsigned) ((n + 255) / 256), 256, 0, (cudaStream_t) stream>>>(x, (block_q8_1*) y, n);
    check("quantize_q8_1_rows");
}

void iq_mmvq(int t, const void* w, const void* x_q8_1, float* y, int n_in, int n_out, int ncols, void* stream) {
    const dim3 grid((unsigned) ((n_out + 3) / 4)), block(32, 4);
    const size_t rb = iq_row_bytes(t, n_in);
    cudaStream_t s = (cudaStream_t) stream;
    const auto* W = (const uint8_t*) w;
    const auto* X = (const block_q8_1*) x_q8_1;
    switch (t) {
#define STRATA_MMVQ(T) case T: mmvq_kernel<T><<<grid, block, 0, s>>>(W, rb, X, y, n_in, n_out, ncols); break;
        STRATA_FMTS(STRATA_MMVQ)
#undef STRATA_MMVQ
        default: std::fprintf(stderr, "iq_mmvq: type %d is not supported\n", t); std::exit(1);
    }
    check("iq_mmvq");
}

void iq_dequant_f16(int t, const void* src, int64_t n, uint16_t* dst, void* stream) {
    if (n % 256 != 0 || !is_iq(t)) { std::fprintf(stderr, "iq_dequant_f16: bad arguments\n"); std::exit(1); }
    dequant_flat_kernel<__half><<<(unsigned) (n / 256), 32, 0, (cudaStream_t) stream>>>(t, src, (__half*) dst);
    check("iq_dequant_f16");
}

namespace {
__global__ void embed_rows_kernel(int ty, const uint8_t* __restrict__ table, size_t row_bytes,
                                  const int32_t* __restrict__ tokens, int64_t n_embd, float* __restrict__ y) {
    const int t = blockIdx.y;
    const int64_t b = blockIdx.x;
    const uint8_t* row = table + (size_t) tokens[t] * row_bytes;
    dq_dispatch<float>(ty, row, b, y + (size_t) t * n_embd + b * QK_K, threadIdx.x);
}
}  // namespace

void iq_embed_rows(int t, const void* table, size_t row_bytes, const int32_t* tokens, int64_t n_tok, int64_t n_embd,
                   float* out, void* stream) {
    if (n_tok <= 0) return;
    if (n_embd % 256 != 0 || !is_iq(t)) { std::fprintf(stderr, "iq_embed_rows: bad arguments\n"); std::exit(1); }
    embed_rows_kernel<<<dim3((unsigned) (n_embd / 256), (unsigned) n_tok), 32, 0, (cudaStream_t) stream>>>(
        t, (const uint8_t*) table, row_bytes, tokens, n_embd, out);
    check("iq_embed_rows");
}

void iq_dequant_f32(int t, const void* src, int64_t n, float* dst, void* stream) {
    if (n % 256 != 0 || !is_iq(t)) { std::fprintf(stderr, "iq_dequant_f32: bad arguments\n"); std::exit(1); }
    dequant_flat_kernel<float><<<(unsigned) (n / 256), 32, 0, (cudaStream_t) stream>>>(t, src, dst);
    check("iq_dequant_f32");
}

void iq_dequant_gu_f16(int t, const void* gate, const void* up, int64_t n_ff, int64_t n_embd, uint16_t* dst, void* stream) {
    const int64_t per_row = n_embd / 256;
    dequant_gu_kernel<<<dim3((unsigned) (n_ff * per_row), 2), 32, 0, (cudaStream_t) stream>>>(t, gate, up, per_row,
                                                                                           (__half*) dst);
    check("iq_dequant_gu_f16");
}

void iq_dequant_experts_f16(int gu_type, int d_type, const uint8_t* const* blobs, int n, size_t up_off, size_t down_off,
                            int64_t n_ff, int64_t n_embd, uint16_t* gu, uint16_t* dn, void* stream) {
    if (n <= 0) return;
    if (n_embd % 256 != 0 || (n_embd * n_ff) % 256 != 0 || !is_iq(gu_type) || !is_iq(d_type) || n > 65535) {
        std::fprintf(stderr, "iq_dequant_experts_f16: bad arguments\n");
        std::exit(1);
    }
    const int64_t per_row = n_embd / 256, gu_blocks = n_ff * per_row, d_blocks = n_embd * n_ff / 256;
    dequant_experts_kernel<<<dim3((unsigned) (2 * gu_blocks + d_blocks), 1, (unsigned) n), 32, 0, (cudaStream_t) stream>>>(
        gu_type, d_type, blobs, up_off, down_off, per_row, gu_blocks, (__half*) gu, (__half*) dn, 2 * n_ff * n_embd,
        n_embd * n_ff);
    check("iq_dequant_experts_f16");
}

bool native_expert_supported(int gu_type, int d_type, int64_t n_embd, int64_t n_ff) noexcept {
    const int qg = fmt_qk(gu_type), qd = fmt_qk(d_type);
    return qg > 0 && qd > 0 && is_iq(gu_type) && is_iq(d_type) && n_embd % qg == 0 && n_ff % qd == 0;
}

NativeExpertLayout native_expert_layout(int gu_type, int d_type, int64_t n_embd, int64_t n_ff) {
    NativeExpertLayout L;
    L.gu_type = gu_type;
    L.d_type = d_type;
    L.n_embd = n_embd;
    L.n_ff = n_ff;
    L.gu_row = iq_row_bytes(gu_type, n_embd);
    L.d_row = iq_row_bytes(d_type, n_ff);
    L.up_off = (size_t) n_ff * L.gu_row;
    L.down_off = 2 * L.up_off;
    L.bytes = L.down_off + (size_t) n_embd * L.d_row;
    return L;
}

size_t native_expert_scratch_bytes(int64_t cap, int64_t n_ff) {
    return ((size_t) cap * (size_t) (n_ff / 32) * sizeof(block_q8_1) + 255) & ~(size_t) 255;
}

void native_expert_grouped(const NativeExpertLayout& L, const unsigned long long* grp_ptr, const int32_t* grp_start,
                           const int32_t* n_groups, const int32_t* ent_dst, const int32_t* ent_tok, int64_t cap_groups,
                           int64_t cap_entries, const void* x_q8_1, void* scratch, float* out, void* stream) {
    if (cap_groups <= 0 || cap_entries <= 0) return;
    if (L.n_ff % 32 != 0 || L.n_embd % (DOWN_WARPS * DOWN_ROWS) != 0) {
        std::fprintf(stderr, "native_expert_grouped: n_ff %lld / n_embd %lld do not divide into the kernels' rows\n",
                     (long long) L.n_ff, (long long) L.n_embd);
        std::exit(1);
    }
    cudaStream_t s = (cudaStream_t) stream;
    auto* hq = (block_q8_1*) scratch;
    const auto* X = (const block_q8_1*) x_q8_1;
    const dim3 ggu((unsigned) (L.n_ff / 32), (unsigned) cap_groups);
    switch (L.gu_type) {
#define STRATA_GU(T) case T: native_gus_kernel<T><<<ggu, GUS_WARPS * 32, 0, s>>>(grp_ptr, grp_start, n_groups, ent_tok, X, L, hq); break;
        STRATA_FMTS(STRATA_GU)
#undef STRATA_GU
        default: std::fprintf(stderr, "native_expert_grouped: gate/up type %d\n", L.gu_type); std::exit(1);
    }
    check("native_expert_grouped/gu");
    const dim3 gd((unsigned) (L.n_embd / (DOWN_WARPS * DOWN_ROWS)), (unsigned) cap_groups);
    switch (L.d_type) {
#define STRATA_DOWN(T) case T: native_down_kernel<T><<<gd, DOWN_WARPS * 32, 0, s>>>(grp_ptr, grp_start, n_groups, ent_dst, hq, L, out); break;
        STRATA_FMTS(STRATA_DOWN)
#undef STRATA_DOWN
        default: std::fprintf(stderr, "native_expert_grouped: down type %d\n", L.d_type); std::exit(1);
    }
    check("native_expert_grouped/down");
}

void native_expert_rows_out(const float* rows, const int32_t* n_groups, const int32_t* grp_start, int64_t n_embd,
                            int64_t max_rows, float* out, void* stream) {
    const long long n4 = n_embd / 4, most = max_rows * n4;
    const unsigned blocks = (unsigned) std::min<long long>((most + 255) / 256, 256);
    rows_out_kernel<<<blocks, 256, 0, (cudaStream_t) stream>>>((const float4*) rows, n_groups, grp_start, n4, (float4*) out);
    check("native_expert_rows_out");
}

}  // namespace strata::kernels
