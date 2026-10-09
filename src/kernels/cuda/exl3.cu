// src/kernels/cuda/exl3.cu - EXL3 GPU reconstruct (docs/EXL3.md).
//
// Mirrors ExLlamaV3's reconstruct_had: decode the trellis tiles, apply the 128-point Hadamard on both
// dimensions, and fold in the suh/svh factors.  The codebook is procedural (decode_3inst), computed
// per window with no lookup table, exactly like the reference kernels.  This first version materializes
// W to fp16; the fused decode-GEMV is the follow-up (see docs/EXL3.md, "Decode optimizations").
#include "strata/kernels/exl3.hpp"

#include <cuda_runtime.h>
#include <cuda_fp16.h>

#include <cstdio>
#include <cstdlib>

namespace strata::kernels {
namespace {

constexpr int HAD = 128;

__device__ __forceinline__ float dh2f(uint16_t h) {
    unsigned sign = (unsigned)(h >> 15) << 31, exp = (h >> 10) & 0x1F, man = h & 0x3FF, f;
    if (exp == 0) {
        if (!man) f = sign;
        else { exp = 127 - 15 + 1; while (!(man & 0x400)) { man <<= 1; --exp; } man &= 0x3FF; f = sign | (exp << 23) | (man << 13); }
    } else if (exp == 0x1F) {
        f = sign | 0x7F800000u | (man << 13);
    } else {
        f = sign | ((exp + 127 - 15) << 23) | (man << 13);
    }
    float o; __builtin_memcpy(&o, &f, 4); return o;
}

__device__ __forceinline__ uint32_t lop3(uint32_t a, uint32_t b, uint32_t c, uint32_t imm) {
    uint32_t r = 0;
    for (int i = 0; i < 8; ++i) {
        uint32_t t = ((imm >> i) & 1) ? 0xFFFFFFFFu : 0u;
        uint32_t ta = (i & 4) ? a : ~a, tb = (i & 2) ? b : ~b, tc = (i & 1) ? c : ~c;
        r |= t & ta & tb & tc;
    }
    return r;
}

// The 16-bit sliding window at bit offset `start`, read word-wise (a funnel shift over uint32).  The
// earliest positions wrap around the tile end (their window starts before bit 0), so the second word
// wraps to 0 - a compare, not the two integer `%` operations the first version paid per weight.
__device__ __forceinline__ uint16_t read_window(const uint16_t* __restrict__ tw, int start, int total_bits) {
    const uint32_t* w32 = (const uint32_t*) tw;
    int n32 = total_bits >> 5;
    int i0 = start >> 5;
    int sh = start & 31;
    uint32_t v = w32[i0] >> sh;
    if (sh + 16 > 32) { int i1 = i0 + 1; if (i1 >= n32) i1 = 0; v |= w32[i1] << (32 - sh); }
    return (uint16_t)(v & 0xFFFF);
}

// mul1 codebook, folded to pure ALU.  decode_window computes h2f(bytesum(w*C)+0x6400) with the software
// fp16->fp32 helper, but that intermediate always has exponent 25 (s in [0x6400,0x67FC]) so it is exactly
// the integer 1024+(s&0x3FF); the two h2f constants fold to plain floats.  This removes the fp16 helper
// and its denormal loop from the GEMV's innermost op.
__device__ __forceinline__ float decode_mul1(uint32_t w) {
    uint32_t x = w * 0x83DCD12Du;
    uint32_t s = (x & 0xFFu) + ((x >> 8) & 0xFFu) + ((x >> 16) & 0xFFu) + (x >> 24) + 0x6400u;
    return (float)(1024 + (int)(s & 0x3FFu)) * 0.00676727294921875f - 10.3828125f;
}

__device__ __forceinline__ uint16_t decode_window(int cb, uint16_t w) {    uint32_t x = w;
    if (cb == 2) {
        x *= 0x83DCD12Du;
        uint32_t s = (x & 0xFF) + ((x >> 8) & 0xFF) + ((x >> 16) & 0xFF) + ((x >> 24) & 0xFF) + 0x6400u;
        float h = dh2f((uint16_t)(s & 0xFFFF));
        return __half_as_ushort(__float2half_rn(__fmaf_rn(h, dh2f(0x1EEEu), dh2f(0xC931u))));
    }
    x = (cb == 1) ? x * 0xCBAC1FEDu : x * 89226354u + 64248484u;
    x = lop3(x, 0x8FFF8FFFu, 0x3B603B60u, 0x6Au);
    return __half_as_ushort(__float2half_rn(dh2f((uint16_t)(x & 0xFFFF)) + dh2f((uint16_t)(x >> 16))));
}

// One block per (nj, ki) tile grid; 256 threads, thread t decodes stored position t and writes it to
// its row-major destination inside the tile (tensor_core_perm: group g = t/8, slot = t%8).
__global__ void decode_kernel(const uint16_t* __restrict__ trellis, int ki, int nj, int words, int bits,
                              int cb, uint16_t* __restrict__ what, int n) {
    int i = blockIdx.y, j = blockIdx.x, t = threadIdx.x; (void)ki;
    const uint16_t* tw = trellis + (long)(i * nj + j) * words;
    int total = 256 * bits;
    int start = t * bits + bits - 16; start = ((start % total) + total) % total;
    unsigned v = 0;
    for (int k2 = 0; k2 < 16; ++k2) { int b = (start + k2) % total; v |= (unsigned)((tw[b >> 4] >> (b & 15)) & 1) << k2; }
    int g = t >> 3, s = t & 7;
    int r0 = (g % 4) * 2, r1 = r0 + 1, r2 = r0 + 8, r3 = r0 + 9, c0 = g / 4, c1 = c0 + 8;
    int r = (s & 3) == 0 ? r0 : (s & 3) == 1 ? r1 : (s & 3) == 2 ? r2 : r3;
    int c = s < 4 ? c0 : c1;
    what[(long)(i * 16 + r) * n + j * 16 + c] = decode_window(cb, (uint16_t)v);
}

__global__ void widen_kernel(const uint16_t* __restrict__ s, float* __restrict__ d, long n) {
    long i = (long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) d[i] = dh2f(s[i]);
}

__global__ void had_rows_kernel(float* __restrict__ w, int k, int n) {
    (void)k;
    int rb = blockIdx.x, c = blockIdx.y;
    float b[HAD];
    for (int i = 0; i < HAD; ++i) b[i] = w[(long)(rb * HAD + i) * n + c];
    for (int step = 1; step < HAD; step <<= 1)
        for (int i = 0; i < HAD; ++i)
            if ((i & step) == 0) { float a = b[i], d = b[i | step]; b[i] = a + d; b[i | step] = a - d; }
    const float inv = 0.08838834764831845f;
    for (int i = 0; i < HAD; ++i) w[(long)(rb * HAD + i) * n + c] = b[i] * inv;
}

__global__ void had_cols_kernel(float* __restrict__ w, int k, int n) {
    (void)k;
    int r = blockIdx.x, cblk = blockIdx.y;
    float b[HAD];
    long base = (long)r * n + cblk * HAD;
    for (int j = 0; j < HAD; ++j) b[j] = w[base + j];
    for (int step = 1; step < HAD; step <<= 1)
        for (int j = 0; j < HAD; ++j)
            if ((j & step) == 0) { float a = b[j], d = b[j | step]; b[j] = a + d; b[j | step] = a - d; }
    const float inv = 0.08838834764831845f;
    for (int j = 0; j < HAD; ++j) w[base + j] = b[j] * inv;
}

__global__ void scale_rows_kernel(float* __restrict__ w, int n, const uint16_t* __restrict__ suh) {
    int i = blockIdx.x;
    float s = dh2f(suh[i]);
    for (int c = threadIdx.x; c < n; c += blockDim.x) w[(long)i * n + c] *= s;
}

__global__ void scale_cols_kernel(float* __restrict__ w, int k, int n, const uint16_t* __restrict__ svh) {
    int c = blockIdx.x;
    float s = dh2f(svh[c]);
    for (int r = threadIdx.x; r < k; r += blockDim.x) w[(long)r * n + c] *= s;
}

__global__ void f32_to_f16_kernel(const float* __restrict__ in, uint16_t* __restrict__ out, long n) {
    long i = (long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = __half_as_ushort(__float2half_rn(in[i]));
}

}  // namespace

void exl3_reconstruct_weight(const uint16_t* trellis, int ki, int nj, int bits, int cb,
                             const uint16_t* suh, const uint16_t* svh, uint16_t* out, void* stream) {
    if (ki <= 0 || nj <= 0 || (ki * 16) % HAD || (nj * 16) % HAD) {
        std::fprintf(stderr, "exl3_reconstruct_weight: dims must be multiples of %d\n", HAD);
        std::abort();
    }
    cudaStream_t st = (cudaStream_t)stream;
    int k = ki * 16, n = nj * 16, words = 256 * bits / 16;
    long nelem = (long)k * n;
    uint16_t* d_what = nullptr;
    float* d_w = nullptr;
    if (cudaMalloc(&d_what, (size_t)nelem * 2) != cudaSuccess ||
        cudaMalloc(&d_w, (size_t)nelem * 4) != cudaSuccess) {
        std::fprintf(stderr, "exl3_reconstruct_weight: cudaMalloc failed\n");
        std::abort();
    }
    decode_kernel<<<dim3(nj, ki), 256, 0, st>>>(trellis, ki, nj, words, bits, cb, d_what, n);
    widen_kernel<<<(int)((nelem + 255) / 256), 256, 0, st>>>(d_what, d_w, nelem);
    had_rows_kernel<<<dim3(k / HAD, n), 1, 0, st>>>(d_w, k, n);
    had_cols_kernel<<<dim3(k, n / HAD), 1, 0, st>>>(d_w, k, n);
    scale_rows_kernel<<<dim3(k), 128, 0, st>>>(d_w, n, suh);
    scale_cols_kernel<<<dim3(n), 128, 0, st>>>(d_w, k, n, svh);
    f32_to_f16_kernel<<<(int)((nelem + 255) / 256), 256, 0, st>>>(d_w, out, nelem);
    (void)cudaFree(d_what);
    (void)cudaFree(d_w);
}

namespace {

__global__ void prescale_had_kernel(const uint16_t* __restrict__ x, const uint16_t* __restrict__ suh,
                                    float* __restrict__ out) {
    __shared__ float b[HAD];
    int t = threadIdx.x, idx = blockIdx.x * HAD + t;
    b[t] = dh2f(x[idx]) * dh2f(suh[idx]);
    __syncthreads();
    for (int step = 1; step < HAD; step <<= 1) {
        if ((t & step) == 0) { float a = b[t], d = b[t | step]; b[t] = a + d; b[t | step] = a - d; }
        __syncthreads();
    }
    out[idx] = b[t] * 0.08838834764831845f;
}

// One block per (output 16-column tile, k-group); 128 threads = 16 columns x 8 k-subchunks.  Thread
// (c, sub) accumulates output column c over its k-slice, the 8 partials are summed in shared, and one
// atomicAdd per column folds the block's k-group into z.  This replaced 16-thread blocks that
// atomicAdd-ed per k-tile: 16-thread blocks cap at ~24 blocks/CU (≈19% occupancy) and every k-tile
// serialized on the same z, which is what pinned the GEMV at ~11 tok/s.
__global__ void gemv_kernel(const float* __restrict__ xh, const uint16_t* __restrict__ trellis,
                            int ki, int nj, int words, int bits, int cb, const int* __restrict__ pinv,
                            float* __restrict__ z) {
    int j = blockIdx.x;
    int c = threadIdx.x & 15, sub = threadIdx.x >> 4;
    int total = 256 * bits;
    int ns = gridDim.y * 8;
    int cid = blockIdx.y * 8 + sub;
    int k0 = (ki * cid) / ns, k1 = (ki * (cid + 1)) / ns;
    int tt[16];
    #pragma unroll
    for (int r = 0; r < 16; ++r) tt[r] = pinv[r * 16 + c];
    float acc = 0;
    for (int i = k0; i < k1; ++i) {
        const uint16_t* tw = trellis + (long)(i * nj + j) * words;
        #pragma unroll
        for (int r = 0; r < 16; ++r) {
            int start = tt[r] * bits + bits - 16; if (start < 0) start += total;
            float wv = cb == 2 ? decode_mul1(read_window(tw, start, total))
                               : __half2float(__ushort_as_half(decode_window(cb, read_window(tw, start, total))));
            acc += xh[i * 16 + r] * wv;
        }
    }
    __shared__ float red[8][16];
    red[sub][c] = acc;
    __syncthreads();
    if (sub == 0) {
        float s = 0;
        #pragma unroll
        for (int k2 = 0; k2 < 8; ++k2) s += red[k2][c];
        atomicAdd(&z[j * 16 + c], s);
    }
}

__global__ void had_postscale_kernel(const float* __restrict__ z, const uint16_t* __restrict__ svh,
                                     uint16_t* __restrict__ y) {
    __shared__ float b[HAD];
    int t = threadIdx.x, idx = blockIdx.x * HAD + t;
    b[t] = z[idx];
    __syncthreads();
    for (int step = 1; step < HAD; step <<= 1) {
        if ((t & step) == 0) { float a = b[t], d = b[t | step]; b[t] = a + d; b[t | step] = a - d; }
        __syncthreads();
    }
    y[idx] = __half_as_ushort(__float2half_rn(b[t] * 0.08838834764831845f * dh2f(svh[idx])));
}

void build_pinv(int* pinv) {
    for (int t = 0; t < 256; ++t) {
        int g = t >> 3, s = t & 7;
        int r0 = (g % 4) * 2, r1 = r0 + 1, r2 = r0 + 8, r3 = r0 + 9, c0 = g / 4, c1 = c0 + 8;
        int r = (s & 3) == 0 ? r0 : (s & 3) == 1 ? r1 : (s & 3) == 2 ? r2 : r3;
        int c = s < 4 ? c0 : c1;
        pinv[r * 16 + c] = t;
    }
}

// The GEMV body, with the three scratch buffers supplied so a caller can reuse them across launches.
void gemv_launch(const uint16_t* x, const uint16_t* suh, const uint16_t* svh, const uint16_t* trellis,
                 int ki, int nj, int bits, int cb, uint16_t* y, float* d_xh, float* d_z, const int* d_pinv,
                 cudaStream_t st) {
    int k = ki * 16, n = nj * 16, words = 256 * bits / 16;
    prescale_had_kernel<<<k / HAD, HAD, 0, st>>>(x, suh, d_xh);
    int gy = (ki + 7) / 8; if (gy > 16) gy = 16; if (gy < 1) gy = 1;
    (void) cudaMemsetAsync(d_z, 0, (size_t)n * 4, st);
    gemv_kernel<<<dim3(nj, gy), 128, 0, st>>>(d_xh, trellis, ki, nj, words, bits, cb, d_pinv, d_z);
    had_postscale_kernel<<<n / HAD, HAD, 0, st>>>(d_z, svh, y);
}

}  // namespace

namespace {
// Persistent GEMV workspace, allocated before any graph capture (cudaMalloc is illegal inside a capture).
struct GemvWs { float* xh = nullptr; float* z = nullptr; int* pinv = nullptr; uint16_t* x16 = nullptr; uint16_t* y16 = nullptr;
                int cxh = 0, cz = 0, cx16 = 0, cy16 = 0; };
GemvWs g_ws;
}  // namespace

void exl3_gemv_reserve(int max_k, int max_n) {
    if (max_k > g_ws.cxh) { (void) cudaFree(g_ws.xh); g_ws.xh = nullptr; if (cudaMalloc(&g_ws.xh, (size_t) max_k * 4) != cudaSuccess) std::abort(); g_ws.cxh = max_k; }
    if (max_n > g_ws.cz) { (void) cudaFree(g_ws.z); g_ws.z = nullptr; if (cudaMalloc(&g_ws.z, (size_t) max_n * 4) != cudaSuccess) std::abort(); g_ws.cz = max_n; }
    if (max_k > g_ws.cx16) { (void) cudaFree(g_ws.x16); g_ws.x16 = nullptr; if (cudaMalloc(&g_ws.x16, (size_t) max_k * 2) != cudaSuccess) std::abort(); g_ws.cx16 = max_k; }
    if (max_n > g_ws.cy16) { (void) cudaFree(g_ws.y16); g_ws.y16 = nullptr; if (cudaMalloc(&g_ws.y16, (size_t) max_n * 2) != cudaSuccess) std::abort(); g_ws.cy16 = max_n; }
    if (g_ws.pinv == nullptr) {
        if (cudaMalloc(&g_ws.pinv, 256 * 4) != cudaSuccess) std::abort();
        int p[256]; build_pinv(p); (void) cudaMemcpy(g_ws.pinv, p, 256 * 4, cudaMemcpyHostToDevice);
    }
}

void exl3_gemv(const uint16_t* x, const uint16_t* suh, const uint16_t* svh, const uint16_t* trellis,
               int ki, int nj, int bits, int cb, uint16_t* y, void* stream) {
    if (ki <= 0 || nj <= 0 || (ki * 16) % HAD || (nj * 16) % HAD) {
        std::fprintf(stderr, "exl3_gemv: dims must be multiples of %d\n", HAD);
        std::abort();
    }
    cudaStream_t st = (cudaStream_t)stream;
    exl3_gemv_reserve(ki * 16, nj * 16);
    gemv_launch(x, suh, svh, trellis, ki, nj, bits, cb, y, g_ws.xh, g_ws.z, g_ws.pinv, st);
}

namespace {

__global__ void f32_to_f16_out_kernel(const float* __restrict__ a, uint16_t* __restrict__ o, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) o[i] = __half_as_ushort(__float2half_rn(a[i]));
}

__global__ void f16_to_f32_kernel(const uint16_t* __restrict__ h, float* __restrict__ y, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) y[i] = dh2f(h[i]);
}

// ---- batched (all-experts) variants: blockIdx.z = expert ----
//
// The engine runs one layer's routed experts per token; doing them one at a time left the GPU idle - each
// expert's prescale->gemv->postscale->silu->down->accum chain is a serial dependency and same-stream
// kernels do not overlap, so throughput scaled exactly linearly with expert count (0.139 ms at k=1 ->
// 1.237 ms at k=10) while the GPU sat ~98% idle.  Making the expert the z dimension lets all experts run
// concurrently inside one kernel and cuts the launch count from 14/expert to ~14/layer.

__global__ void prescale_batched_kernel(const uint16_t* __restrict__ x, long x_stride,
                                        const Exl3Mat* __restrict__ mats, float* __restrict__ xh, int k) {
    __shared__ float b[HAD];
    int e = blockIdx.y, t = threadIdx.x, idx = blockIdx.x * HAD + t;
    b[t] = dh2f(x[e * x_stride + idx]) * dh2f(mats[e].suh[idx]);
    __syncthreads();
    for (int step = 1; step < HAD; step <<= 1) {
        if ((t & step) == 0) { float a = b[t], d = b[t | step]; b[t] = a + d; b[t | step] = a - d; }
        __syncthreads();
    }
    xh[(long)e * k + idx] = b[t] * 0.08838834764831845f;
}

template <int T>
__global__ void __launch_bounds__(256, 6) gemv_batched_kernel(const float* __restrict__ xh, const Exl3Mat* __restrict__ mats,
                                    const int* __restrict__ pinv, float* __restrict__ z) {
    int e = blockIdx.z;
    int c = threadIdx.x & 15, sub = threadIdx.x >> 4;
    int ki = mats[e].ki, nj = mats[e].nj, bits = mats[e].bits, cb = mats[e].cb;
    int total = 256 * bits, words = 256 * bits / 16;
    int ns = gridDim.y * 16;
    int cid = blockIdx.y * 16 + sub;
    int k0 = (ki * cid) / ns, k1 = (ki * (cid + 1)) / ns;
    const uint16_t* trellis = mats[e].trellis;
    const float* xhe = xh + (long)e * (ki * 16);
    int tt[16];
    #pragma unroll
    for (int r = 0; r < 16; ++r) tt[r] = pinv[r * 16 + c];
    float acc[T];
    #pragma unroll
    for (int t = 0; t < T; ++t) acc[t] = 0;
    for (int i = k0; i < k1; ++i) {
        const uint16_t* twp[T];
        #pragma unroll
        for (int t = 0; t < T; ++t) {
            int j = blockIdx.x + t * gridDim.x;
            twp[t] = (j < nj) ? trellis + (long)(i * nj + j) * words : nullptr;
        }
        #pragma unroll 4
        for (int r = 0; r < 16; ++r) {
            int start = tt[r] * bits + bits - 16; if (start < 0) start += total;
            float xr = xhe[i * 16 + r];
            #pragma unroll
            for (int t = 0; t < T; ++t) {
                if (!twp[t]) continue;
                float wv = cb == 2 ? decode_mul1(read_window(twp[t], start, total))
                                   : __half2float(__ushort_as_half(decode_window(cb, read_window(twp[t], start, total))));
                acc[t] += xr * wv;
            }
        }
    }
    __shared__ float red[T][16][16];
    #pragma unroll
    for (int t = 0; t < T; ++t) red[t][sub][c] = acc[t];
    __syncthreads();
    if (sub == 0) {
        #pragma unroll
        for (int t = 0; t < T; ++t) {
            int j = blockIdx.x + t * gridDim.x;
            if (j >= nj) break;
            float s = 0;
            #pragma unroll
            for (int k2 = 0; k2 < 16; ++k2) s += red[t][k2][c];
            atomicAdd(&z[(long)e * (nj * 16) + j * 16 + c], s);
        }
    }
}

__global__ void postscale_batched_kernel(const float* __restrict__ z, const Exl3Mat* __restrict__ mats,
                                         uint16_t* __restrict__ y, int n) {
    __shared__ float b[HAD];
    int e = blockIdx.y, t = threadIdx.x, idx = blockIdx.x * HAD + t;
    b[t] = z[(long)e * n + idx];
    __syncthreads();
    for (int step = 1; step < HAD; step <<= 1) {
        if ((t & step) == 0) { float a = b[t], d = b[t | step]; b[t] = a + d; b[t | step] = a - d; }
        __syncthreads();
    }
    y[(long)e * n + idx] = __half_as_ushort(__float2half_rn(b[t] * 0.08838834764831845f * dh2f(mats[e].svh[idx])));
}

__global__ void silu_batched_kernel(const uint16_t* __restrict__ g, const uint16_t* __restrict__ u,
                                    uint16_t* __restrict__ h, long total) {
    long i = (long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= total) return;
    float gv = __half2float(__ushort_as_half(g[i]));
    float s = gv / (1.0f + expf(-gv));
    h[i] = __half_as_ushort(__float2half_rn(s * __half2float(__ushort_as_half(u[i]))));
}

__global__ void accum_batched_kernel(const uint16_t* __restrict__ d, const float* __restrict__ weights,
                                     int n_experts, int n, float* __restrict__ acc) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    float s = 0;
    for (int e = 0; e < n_experts; ++e) s += weights[e] * __half2float(__ushort_as_half(d[(long)e * n + i]));
    acc[i] = s;
}

}  // namespace

constexpr int GEMV_T = 2;

namespace {

// Run gate/up/silu/down for every expert (batched, grid.z = expert) and return the process-wide device fp16
// buffer holding each expert's down output (n_experts x n).  Shared by the weighted `exl3_moe_ffn` and the
// unweighted `exl3_moe_rows`.
uint16_t* moe_run_experts(const Exl3Mat* gate, const Exl3Mat* up, const Exl3Mat* down, int n_experts,
                          const uint16_t* x, cudaStream_t st) {
    const int k = gate[0].ki * 16;
    const int ff = gate[0].nj * 16;
    const int n = down[0].nj * 16;
    const int maxk = k > ff ? k : ff;      // xh: gate/up transform k, down transforms ff
    const int maxn = ff > n ? ff : n;      // z: gate/up produce ff, down produces n
    static Exl3Mat* d_gate = nullptr; static Exl3Mat* d_up = nullptr; static Exl3Mat* d_down = nullptr;
    static int cne = 0;
    size_t msz = (size_t) n_experts * sizeof(Exl3Mat);
    if (n_experts > cne) {
        (void)cudaFree(d_gate); (void)cudaFree(d_up); (void)cudaFree(d_down);
        d_gate = d_up = d_down = nullptr;
        if (cudaMalloc((void**)&d_gate, msz) || cudaMalloc((void**)&d_up, msz) || cudaMalloc((void**)&d_down, msz)) {
            const cudaError_t me = cudaGetLastError(); size_t fb=0,tb=0; const cudaError_t gi = cudaMemGetInfo(&fb,&tb); std::fprintf(stderr, "exl3_moe: cudaMalloc(msz=%zu) failed: %s; meminfo rc=%s %zu/%zu MiB\n", msz, cudaGetErrorString(me), cudaGetErrorString(gi), fb>>20, tb>>20); std::abort();
        }
        cne = n_experts;
    }
    (void)cudaMemcpyAsync(d_gate, gate, msz, cudaMemcpyHostToDevice, st);
    (void)cudaMemcpyAsync(d_up, up, msz, cudaMemcpyHostToDevice, st);
    (void)cudaMemcpyAsync(d_down, down, msz, cudaMemcpyHostToDevice, st);
    static uint16_t* d_g = nullptr; static uint16_t* d_u = nullptr; static uint16_t* d_h = nullptr;
    static uint16_t* d_d = nullptr; static float* d_xh = nullptr; static float* d_z = nullptr; static int* d_pinv = nullptr;
    static int cff = 0, cn = 0, cmk = 0, cmn = 0;
    if (ff > cff) {
        (void)cudaFree(d_g); (void)cudaFree(d_u); (void)cudaFree(d_h); d_g = d_u = d_h = nullptr;
        if (cudaMalloc((void**)&d_g, (size_t)ff * n_experts * 2) || cudaMalloc((void**)&d_u, (size_t)ff * n_experts * 2) ||
            cudaMalloc((void**)&d_h, (size_t)ff * n_experts * 2)) {
            size_t fb=0,tb=0; cudaMemGetInfo(&fb,&tb);
            std::fprintf(stderr, "exl3_moe: cudaMalloc(ff=%d k=%d -> %.1f MiB) failed: %zu MiB free of %zu MiB\n", ff, n_experts, (double)ff*n_experts*2*3/(1<<20), fb>>20, tb>>20);
            std::abort(); }
        cff = ff;
    }
    if (n > cn) {
        (void)cudaFree(d_d); d_d = nullptr;
        if (cudaMalloc((void**)&d_d, (size_t)n * n_experts * 2)) { std::fprintf(stderr, "exl3_moe: cudaMalloc failed\n"); std::abort(); }
        cn = n;
    }
    if (maxk * n_experts > cmk) { (void)cudaFree(d_xh); d_xh = nullptr; if (cudaMalloc((void**)&d_xh, (size_t)maxk * n_experts * 4)) std::abort(); cmk = maxk * n_experts; }
    if (maxn * n_experts > cmn) { (void)cudaFree(d_z); d_z = nullptr; if (cudaMalloc((void**)&d_z, (size_t)maxn * n_experts * 4)) std::abort(); cmn = maxn * n_experts; }
    if (!d_pinv) { if (cudaMalloc((void**)&d_pinv, 256 * 4)) std::abort(); int pinv[256]; build_pinv(pinv); (void)cudaMemcpy(d_pinv, pinv, 256 * 4, cudaMemcpyHostToDevice); }
    int gy_g = (gate[0].ki + 15) / 16; if (gy_g < 1) gy_g = 1;
    int gy_d = (down[0].ki + 15) / 16; if (gy_d < 1) gy_d = 1;
    prescale_batched_kernel<<<dim3(k / HAD, n_experts), HAD, 0, st>>>(x, 0, d_gate, d_xh, k);
    (void)cudaMemsetAsync(d_z, 0, (size_t)n_experts * ff * 4, st);
    gemv_batched_kernel<GEMV_T><<<dim3((ff / 16 + GEMV_T - 1) / GEMV_T, gy_g, n_experts), 256, 0, st>>>(d_xh, d_gate, d_pinv, d_z);
    postscale_batched_kernel<<<dim3(ff / HAD, n_experts), HAD, 0, st>>>(d_z, d_gate, d_g, ff);
    prescale_batched_kernel<<<dim3(k / HAD, n_experts), HAD, 0, st>>>(x, 0, d_up, d_xh, k);
    (void)cudaMemsetAsync(d_z, 0, (size_t)n_experts * ff * 4, st);
    gemv_batched_kernel<GEMV_T><<<dim3((ff / 16 + GEMV_T - 1) / GEMV_T, gy_g, n_experts), 256, 0, st>>>(d_xh, d_up, d_pinv, d_z);
    postscale_batched_kernel<<<dim3(ff / HAD, n_experts), HAD, 0, st>>>(d_z, d_up, d_u, ff);
    { long tot = (long)n_experts * ff; silu_batched_kernel<<<(unsigned)((tot + 255) / 256), 256, 0, st>>>(d_g, d_u, d_h, tot); }
    prescale_batched_kernel<<<dim3(ff / HAD, n_experts), HAD, 0, st>>>(d_h, ff, d_down, d_xh, ff);
    (void)cudaMemsetAsync(d_z, 0, (size_t)n_experts * n * 4, st);
    gemv_batched_kernel<GEMV_T><<<dim3((n / 16 + GEMV_T - 1) / GEMV_T, gy_d, n_experts), 256, 0, st>>>(d_xh, d_down, d_pinv, d_z);
    postscale_batched_kernel<<<dim3(n / HAD, n_experts), HAD, 0, st>>>(d_z, d_down, d_d, n);
    return d_d;
}

}  // namespace

void exl3_moe_ffn(const Exl3Mat* gate, const Exl3Mat* up, const Exl3Mat* down, const float* weights,
                  int n_experts, const uint16_t* x, uint16_t* out, void* stream) {
    if (n_experts <= 0) return;
    cudaStream_t st = (cudaStream_t) stream;
    const int n = down[0].nj * 16;
    uint16_t* d_d = moe_run_experts(gate, up, down, n_experts, x, st);
    static float* d_acc = nullptr; static float* d_weights = nullptr; static int cw = 0, ca = 0;
    if (n > ca) { (void)cudaFree(d_acc); d_acc = nullptr; if (cudaMalloc((void**)&d_acc, (size_t)n * 4)) std::abort(); ca = n; }
    if (n_experts > cw) { (void)cudaFree(d_weights); d_weights = nullptr; if (cudaMalloc((void**)&d_weights, (size_t)n_experts * 4)) std::abort(); cw = n_experts; }
    (void)cudaMemcpyAsync(d_weights, weights, (size_t)n_experts * 4, cudaMemcpyHostToDevice, st);
    accum_batched_kernel<<<(n + 255) / 256, 256, 0, st>>>(d_d, d_weights, n_experts, n, d_acc);
    f32_to_f16_out_kernel<<<(n + 255) / 256, 256, 0, st>>>(d_acc, out, n);
}

void exl3_moe_rows(const Exl3Mat* gate, const Exl3Mat* up, const Exl3Mat* down, const int* ids,
                   int k, const uint16_t* x, uint16_t* out, void* stream) {
    (void) ids;   // the caller orders gate/up/down to match ids, so row i is expert ids[i]
    if (k <= 0) return;
    cudaStream_t st = (cudaStream_t) stream;
    const int n = down[0].nj * 16;
    uint16_t* d_d = moe_run_experts(gate, up, down, k, x, st);
    (void) cudaMemcpyAsync(out, d_d, (size_t)k * n * 2, cudaMemcpyDeviceToDevice, st);
}

__global__ void embed_gather_kernel(const uint16_t* __restrict__ row, long n, int bf16, float* __restrict__ out) {
    long i = (long) blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const uint16_t h = row[i];
    if (bf16) { uint32_t f = (uint32_t) h << 16; float o; __builtin_memcpy(&o, &f, 4); out[i] = o; }
    else out[i] = __half2float(*(const __half*) &h);
}


__global__ void silu_mul_f32_kernel(const float* __restrict__ g, const float* __restrict__ u, float* __restrict__ h, long n) {
    long i = (long) blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const float gv = g[i];
    h[i] = (gv / (1.0f + expf(-gv))) * u[i];
}

__global__ void shared_scalar_gate_kernel(float* __restrict__ out, const float* __restrict__ x,
                                          const uint16_t* __restrict__ w_ginp, int n_embd) {
    // one block: sigmoid(w_ginp . x) over n_embd, applied to every out element
    __shared__ float red[256];
    const int t = threadIdx.x;
    float s = 0;
    for (int i = t; i < n_embd; i += blockDim.x) {
        const uint16_t h = w_ginp[i];
        uint32_t f = (uint32_t) h << 16; float w; __builtin_memcpy(&w, &f, 4);
        s += w * x[i];
    }
    red[t] = s; __syncthreads();
    for (int step = blockDim.x >> 1; step > 0; step >>= 1) { if (t < step) red[t] += red[t + step]; __syncthreads(); }
    if (t == 0) red[0] = 1.0f / (1.0f + expf(-red[0]));
    __syncthreads();
    const float gate = red[0];
    for (int i = t; i < n_embd; i += blockDim.x) out[i] *= gate;
}

static float* g_sh_g = nullptr; static float* g_sh_u = nullptr; static float* g_sh_h = nullptr;
static int g_sh_ff = 0;

void exl3_shared_reserve(int n_ff) {
    if (n_ff <= g_sh_ff) return;
    (void) cudaFree(g_sh_g); (void) cudaFree(g_sh_u); (void) cudaFree(g_sh_h);
    g_sh_g = g_sh_u = g_sh_h = nullptr;
    if (cudaMalloc((void**) &g_sh_g, (size_t) n_ff * 4) || cudaMalloc((void**) &g_sh_u, (size_t) n_ff * 4) ||
        cudaMalloc((void**) &g_sh_h, (size_t) n_ff * 4)) std::abort();
    g_sh_ff = n_ff;
}

void exl3_shared_expert(const float* x, const Exl3Mat* gate, const Exl3Mat* up, const Exl3Mat* down,
                        const uint16_t* w_ginp, float* out, int n_embd, int n_ff, void* stream) {
    cudaStream_t st = (cudaStream_t) stream;
    exl3_shared_reserve(n_ff);
    float* const d_g = g_sh_g; float* const d_u = g_sh_u; float* const d_h = g_sh_h;
    exl3_gemv_f32(x, gate->suh, gate->svh, gate->trellis, gate->ki, gate->nj, gate->bits, gate->cb, d_g, stream);
    exl3_gemv_f32(x, up->suh, up->svh, up->trellis, up->ki, up->nj, up->bits, up->cb, d_u, stream);
    silu_mul_f32_kernel<<<(unsigned)((n_ff + 255) / 256), 256, 0, st>>>(d_g, d_u, d_h, n_ff);
    exl3_gemv_f32(d_h, down->suh, down->svh, down->trellis, down->ki, down->nj, down->bits, down->cb, out, stream);
    shared_scalar_gate_kernel<<<1, 256, 0, st>>>(out, x, w_ginp, n_embd);
}

void exl3_embed_gather(const uint16_t* row, int64_t n, bool bf16, float* out, void* stream) {
    cudaStream_t st = (cudaStream_t) stream;
    embed_gather_kernel<<<(unsigned)((n + 255) / 256), 256, 0, st>>>(row, n, bf16 ? 1 : 0, out);
}


void exl3_gemv_f32(const float* x, const uint16_t* suh, const uint16_t* svh, const uint16_t* trellis,
                   int ki, int nj, int bits, int cb, float* y, void* stream) {
    cudaStream_t st = (cudaStream_t) stream;
    const int k = ki * 16, n = nj * 16;
    exl3_gemv_reserve(k, n);
    f32_to_f16_out_kernel<<<(k + 255) / 256, 256, 0, st>>>(x, g_ws.x16, k);
    exl3_gemv(g_ws.x16, suh, svh, trellis, ki, nj, bits, cb, g_ws.y16, stream);
    f16_to_f32_kernel<<<(n + 255) / 256, 256, 0, st>>>(g_ws.y16, y, n);
}

}  // namespace strata::kernels
