// bench/micro/s2_unpack_bench.cu - where does the M=2..8 expert GEMV time go?  Isolated benchmark only -
// this file touches nothing in the inference path (the `bench_tc_gemv.cu` rule).
//
// QUESTION.  `s2_expert_grouped.cu`'s M=2..8 path (verify window, `moe_grouped_s2`) spends time unpacking the
// Q2_0 codes inside the kernel: `expand_codes` (2-bit fields -> one byte per element, 16 ALU ops per 32
// elements) and `load_x_chunk` (the activation's 4x4 byte transpose per staged chunk).  How much is that
// unpacking really worth, and can a load-time rearrangement of the weights remove it?
//
// METHOD.  One Q2_0 expert shape (H 2560 / FF 640, the blob geometry of `cpu/expert.hpp`), M entries routed
// to the SAME expert (the "all shared" verify-window shape: weights read once, M activations dotted), gate/up
// and down projections, CUDA events, blobs cycled through a set larger than L2 so codes come from DRAM as the
// engine's do.  The inner loop is `gu_grouped_t_kernel`'s / `down_grouped_t_kernel`'s, with parts switched
// off or replaced:
//
//   0 base          the shipped inner loop (expand_codes + staged X + chunk_s + warp_sum), verbatim
//   1 no-expand     m[] = 0            -> the cost of expand_codes
//   2 no-xload      X[] = 0            -> the cost of the staged activation reads
//   3 no-dp4a       s  = 0             -> the cost of the two dp4a chains
//   4 no-sum        acc += 0           -> the cost of the float expression and warp_sum
//   5 clean         codes PRE-EXPANDED (4x code bytes - a bandwidth-changing reference!), activations
//                   pre-formed and hx precomputed: "direct DP4A/vector load", no unpacking at all
//   6 chains2       base but the dp4a chain split in two independent halves (integer-exact)
//   7 hmma          codes repacked once (QPN8-style, same bytes) + m8n8k4 f16 tensor cores with the 1024
//                   bias trick; f32 accumulation, same per-chunk float expression
//
// V5 is NOT a candidate layout: its codes are 4x the bytes, so at M>=2 it pays more DRAM than the unpacking
// it removes - which is exactly why the repack must keep the byte count.  It is measured to price the
// unpacking, nothing else.  V7 keeps the byte count (the repack is a permutation).
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cstdint>
#include <functional>
#include <random>
#include <string>
#include <vector>

namespace {

constexpr int H = 2560, FF = 640, QK = 64;
constexpr int ROW_GU = H / 4, ROW_D = FF / 4, SC_GU = H / QK, SC_D = FF / QK;
constexpr size_t O_D_CODES = 2ull * FF * ROW_GU;
constexpr size_t O_GU_SCALES = O_D_CODES + (size_t) H * ROW_D;
constexpr size_t O_D_SCALES = O_GU_SCALES + 2ull * FF * SC_GU * 2;
constexpr size_t BLOB = O_D_SCALES + (size_t) H * SC_D * 2;
static_assert(BLOB == 1382400, "blob geometry");
constexpr int GU_CHUNKS = (H / 32 + 31) / 32;   // 3
constexpr int NC_GU = H / 32, NC_D = FF / 32;   // 80, 20
constexpr int GMAX = 8;
constexpr int THREADS = 256, GU_ROWS = 32, D_ROWS = 64;

#define CHK(x) do { const cudaError_t e_ = (x); if (e_ != cudaSuccess) { \
    std::fprintf(stderr, "%s:%d: %s: %s\n", __FILE__, __LINE__, #x, cudaGetErrorString(e_)); std::exit(1); } } while (0)

__device__ __forceinline__ float f16_ld(const uint8_t* p) {
    return __half2float(__ushort_as_half(*(const unsigned short*) p));
}
__device__ __forceinline__ int dp4a(int a, int b, int c) { return __dp4a(a, b, c); }

__device__ __forceinline__ int load_x_chunk(const uint8_t* __restrict__ xb, int X[8]) {
    const uint8_t* q = xb + 2;
    const int off = (int) ((uintptr_t) q & 3);
    const unsigned* p = (const unsigned*) (q - off);
    const unsigned sh = (unsigned) off * 8;
    unsigned v[9];
#pragma unroll
    for (int j = 0; j < 8; ++j) v[j] = __ldg(p + j);
    v[8] = off != 0 ? __ldg(p + 8) : 0u;
    unsigned n[8];
    int hx = 0;
#pragma unroll
    for (int j = 0; j < 8; ++j) {
        n[j] = (unsigned) ((((unsigned long long) v[j + 1] << 32) | v[j]) >> sh);
        hx = dp4a(0x01010101, (int) n[j], hx);
    }
#pragma unroll
    for (int h = 0; h < 2; ++h) {
        const unsigned t0 = __byte_perm(n[4 * h], n[4 * h + 1], 0x5140);
        const unsigned t1 = __byte_perm(n[4 * h], n[4 * h + 1], 0x7362);
        const unsigned t2 = __byte_perm(n[4 * h + 2], n[4 * h + 3], 0x5140);
        const unsigned t3 = __byte_perm(n[4 * h + 2], n[4 * h + 3], 0x7362);
        X[4 * h + 0] = (int) __byte_perm(t0, t2, 0x5410);
        X[4 * h + 1] = (int) __byte_perm(t0, t2, 0x7632);
        X[4 * h + 2] = (int) __byte_perm(t1, t3, 0x5410);
        X[4 * h + 3] = (int) __byte_perm(t1, t3, 0x7632);
    }
    return hx;
}

__device__ __forceinline__ void expand_codes(uint2 cb, int m[8]) {
    const unsigned M = 0x03030303u;
    m[0] = (int) (cb.x & M);
    m[1] = (int) ((cb.x >> 2) & M);
    m[2] = (int) ((cb.x >> 4) & M);
    m[3] = (int) ((cb.x >> 6) & M);
    m[4] = (int) (cb.y & M);
    m[5] = (int) ((cb.y >> 2) & M);
    m[6] = (int) ((cb.y >> 4) & M);
    m[7] = (int) ((cb.y >> 6) & M);
}

// S2T ("skinny transposed"): the load-time bit transpose of one chunk's 8 code bytes.  Byte 4h+f of the
// S2T chunk holds the four codes of elements {16h+4b+f : b=0..3} packed 2-bit - exactly the m-word the
// in-kernel expand_codes would build.  Same bytes, same offsets: a permutation of bits within each chunk.
// The runtime unpack becomes ONE shared-memory vector load per code byte (the 256-entry LUT is the whole
// 2-bit -> byte map), so the GEMV inner loop is "vector load + dp4a" with no unpack ALU and, because the
// m-word element grouping is unchanged, BIT-EXACTLY the previous integer sums and float expression.
__device__ __forceinline__ void expand_s2t(uint2 cb, const int* __restrict__ lut, int m[8]) {
    const unsigned char* b = (const unsigned char*) &cb;
#pragma unroll
    for (int j = 0; j < 8; ++j) m[j] = lut[b[j]];
}

__device__ __forceinline__ int chunk_s(const int m[8], const int X[8]) {
    int s = 0;
#pragma unroll
    for (int j = 0; j < 8; ++j) s = dp4a(m[j], X[j], s);
    return s;
}
// two independent 4-deep chains, integer-exact (the sum of the eight products groups either way)
__device__ __forceinline__ int chunk_s2(const int m[8], const int X[8]) {
    int a = 0, b = 0;
#pragma unroll
    for (int j = 0; j < 4; ++j) {
        a = dp4a(m[j], X[j], a);
        b = dp4a(m[4 + j], X[4 + j], b);
    }
    return a + b;
}
// four independent 2-deep chains, integer-exact
__device__ __forceinline__ int chunk_s4(const int m[8], const int X[8]) {
    int a = 0, b = 0, c = 0, d = 0;
#pragma unroll
    for (int j = 0; j < 2; ++j) {
        a = dp4a(m[j], X[j], a);
        b = dp4a(m[2 + j], X[2 + j], b);
        c = dp4a(m[4 + j], X[4 + j], c);
        d = dp4a(m[6 + j], X[6 + j], d);
    }
    return (a + b) + (c + d);
}

__device__ __forceinline__ float warp_sum(float v) {
#pragma unroll
    for (int off = 16; off > 0; off >>= 1) v += __shfl_down_sync(0xFFFFFFFFu, v, off);
    return v;
}

template <int VAR>
__global__ void __launch_bounds__(256) gu_kernel(const unsigned long long* __restrict__ grp_ptr,
                                                 const int32_t* __restrict__ grp_start,
                                                 const int32_t* __restrict__ n_groups,
                                                 const int32_t* __restrict__ ent_tok,
                                                 const uint8_t* __restrict__ x_q8_0,
                                                 const float* __restrict__ x_scales,
                                                 const int* __restrict__ mwords, const int* __restrict__ xformed,
                                                 const int* __restrict__ xhx, float* __restrict__ gate_up,
                                                 int cap_entries) {
    constexpr int NC = H / 32;
    __shared__ int xs_w[8 * GMAX * NC];
    __shared__ int2 xs_dh[GMAX * NC];
    __shared__ int lut[256];
    if (VAR == 13 && threadIdx.x < 256) {
        const unsigned p = (unsigned) threadIdx.x;
        lut[p] = (int) ((p & 3u) | (((p >> 2) & 3u) << 8) | (((p >> 4) & 3u) << 16) | (((p >> 6) & 3u) << 24));
    }
    if (VAR == 13) __syncthreads();
    const int g = blockIdx.y;
    if (g >= *n_groups) return;
    const int e0 = grp_start[g], ne = min(grp_start[g + 1] - e0, GMAX);
    const int t = threadIdx.x, lane = t & 31, warp = t >> 5;
    for (int i = t; i < ne * NC; i += blockDim.x) {
        const int k = i / NC, c = i - k * NC;
        const int tok = ent_tok[e0 + k];
        if (VAR == 5) {
            xs_dh[i] = make_int2(__float_as_int(x_scales[(size_t) tok * NC + c]), xhx[(size_t) tok * NC + c]);
            for (int j = 0; j < 8; ++j) xs_w[j * (GMAX * NC) + i] = xformed[((size_t) tok * NC + c) * 8 + j];
        } else {
            const uint8_t* xb = x_q8_0 + (size_t) tok * (size_t) NC * 34 + (size_t) c * 34;
            const float dx = x_scales ? x_scales[(size_t) tok * NC + c] : f16_ld(xb);
            int X[8];
            const int hx = (VAR == 2) ? 0 : load_x_chunk(xb, X);
            for (int j = 0; j < 8; ++j) xs_w[j * (GMAX * NC) + i] = (VAR == 2) ? 0 : X[j];
            xs_dh[i] = make_int2(__float_as_int(dx), hx);
        }
    }
    __syncthreads();
    const uint8_t* blob = (const uint8_t*) grp_ptr[g];
    const int row0 = blockIdx.x * GU_ROWS;
    for (int pp = warp; pp < GU_ROWS / 2; pp += 8) {
        const int i = row0 + 2 * pp;
        const uint8_t* codes = blob + (size_t) i * ROW_GU;
        const uint8_t* scales = blob + O_GU_SCALES + (size_t) i * SC_GU * 2;
        int m0[GU_CHUNKS][8], m1[GU_CHUNKS][8];
        float dw0[GU_CHUNKS], dw1[GU_CHUNKS];
#pragma unroll
        for (int q = 0; q < GU_CHUNKS; ++q) {
            const int c = lane + 32 * q;
            if (c < NC) {
                if (VAR == 5) {
                    const int* w0 = mwords + ((size_t) (size_t) i * NC + c) * 8;
                    const int* w1 = mwords + ((size_t) (i + 1) * NC + c) * 8;
                    for (int j = 0; j < 8; ++j) { m0[q][j] = w0[j]; m1[q][j] = w1[j]; }
                } else if (VAR == 1) {
                    for (int j = 0; j < 8; ++j) { m0[q][j] = 0; m1[q][j] = 0; }
                } else if (VAR == 13) {
                    expand_s2t(*(const uint2*) (codes + (size_t) c * 8), lut, m0[q]);
                    expand_s2t(*(const uint2*) (codes + ROW_GU + (size_t) c * 8), lut, m1[q]);
                } else {
                    expand_codes(*(const uint2*) (codes + (size_t) c * 8), m0[q]);
                    expand_codes(*(const uint2*) (codes + ROW_GU + (size_t) c * 8), m1[q]);
                }
                dw0[q] = f16_ld(scales + (size_t) (c >> 1) * 2);
                dw1[q] = f16_ld(scales + SC_GU * 2 + (size_t) (c >> 1) * 2);
            }
        }
        for (int k = 0; k < ne; ++k) {
            float acc0 = 0.0f, acc1 = 0.0f;
#pragma unroll
            for (int q = 0; q < GU_CHUNKS; ++q) {
                const int c = lane + 32 * q;
                if (c >= NC) break;
                const int at = k * NC + c;
                int X[8];
#pragma unroll
                for (int j = 0; j < 8; ++j) X[j] = xs_w[j * (GMAX * NC) + at];
                const int2 dh = xs_dh[at];
                const float dx = __int_as_float(dh.x);
                int s0 = 0, s1 = 0;
                if (VAR != 3) {
                    if (VAR == 6) { s0 = chunk_s2(m0[q], X); s1 = chunk_s2(m1[q], X); }
                    else if (VAR == 7) { s0 = chunk_s4(m0[q], X); s1 = chunk_s4(m1[q], X); }
                    else { s0 = chunk_s(m0[q], X); s1 = chunk_s(m1[q], X); }
                }
                if (VAR != 4) {
                    acc0 += dw0[q] * dx * (float) (s0 - dh.y);
                    acc1 += dw1[q] * dx * (float) (s1 - dh.y);
                }
            }
            const float s0 = (VAR == 4) ? (float) (k + 1) : warp_sum(acc0);
            const float s1 = (VAR == 4) ? (float) (k + 1) : warp_sum(acc1);
            if (lane == 0) {
                const int e = e0 + k, r = i >> 1;
                gate_up[(size_t) e * FF + (size_t) r] = s0;
                gate_up[(size_t) cap_entries * FF + (size_t) e * FF + (size_t) r] = s1;
            }
        }
    }
}

template <int VAR>
__global__ void __launch_bounds__(256) down_kernel(const unsigned long long* __restrict__ grp_ptr,
                                                   const int32_t* __restrict__ grp_start,
                                                   const int32_t* __restrict__ n_groups,
                                                   const int32_t* __restrict__ ent_dst,
                                                   const uint8_t* __restrict__ h_q8_0,
                                                   const float* __restrict__ h_scales,
                                                   const int* __restrict__ mwords, const int* __restrict__ xformed,
                                                   const int* __restrict__ xhx, float* __restrict__ out) {
    constexpr int NC = FF / 32;
    __shared__ int hs_w[8 * GMAX * NC];
    __shared__ int2 hs_dh[GMAX * NC];
    __shared__ int lut[256];
    if (VAR == 13 && threadIdx.x < 256) {
        const unsigned p = (unsigned) threadIdx.x;
        lut[p] = (int) ((p & 3u) | (((p >> 2) & 3u) << 8) | (((p >> 4) & 3u) << 16) | ((p >> 6) & 3u) << 24);
    }
    if (VAR == 13) __syncthreads();
    const int g = blockIdx.y;
    if (g >= *n_groups) return;
    const int e0 = grp_start[g], ne = min(grp_start[g + 1] - e0, GMAX);
    const int t = threadIdx.x, lane = t & 31, warp = t >> 5;
    for (int i = t; i < ne * NC; i += blockDim.x) {
        const int k = i / NC, c = i - k * NC;
        if (VAR == 5) {
            hs_dh[i] = make_int2(__float_as_int(h_scales[(size_t) (e0 + k) * NC + c]), xhx[(size_t) (e0 + k) * NC + c]);
            for (int j = 0; j < 8; ++j) hs_w[j * (GMAX * NC) + i] = xformed[((size_t) (e0 + k) * NC + c) * 8 + j];
        } else {
            const uint8_t* xb = h_q8_0 + (size_t) (e0 + k) * (size_t) NC * 34 + (size_t) c * 34;
            const float dx = h_scales ? h_scales[(size_t) (e0 + k) * NC + c] : f16_ld(xb);
            int X[8];
            const int hx = (VAR == 2) ? 0 : load_x_chunk(xb, X);
            for (int j = 0; j < 8; ++j) hs_w[j * (GMAX * NC) + i] = (VAR == 2) ? 0 : X[j];
            hs_dh[i] = make_int2(__float_as_int(dx), hx);
        }
    }
    __syncthreads();
    const uint8_t* blob = (const uint8_t*) grp_ptr[g];
    const int row0 = blockIdx.x * D_ROWS;
    for (int pp = warp; pp < D_ROWS / 2; pp += 8) {
        const int r = row0 + 2 * pp;
        const uint8_t* codes = blob + O_D_CODES + (size_t) r * ROW_D;
        const uint8_t* scales = blob + O_D_SCALES + (size_t) r * SC_D * 2;
        const int c = lane;
        int m0[8], m1[8];
        float dw0 = 0.0f, dw1 = 0.0f;
        if (c < NC) {
            if (VAR == 5) {
                const int* w0 = mwords + ((size_t) r * NC + c) * 8;
                const int* w1 = mwords + ((size_t) (r + 1) * NC + c) * 8;
                for (int j = 0; j < 8; ++j) { m0[j] = w0[j]; m1[j] = w1[j]; }
            } else if (VAR == 1) {
                for (int j = 0; j < 8; ++j) { m0[j] = 0; m1[j] = 0; }
            } else if (VAR == 13) {
                expand_s2t(*(const uint2*) (codes + (size_t) c * 8), lut, m0);
                expand_s2t(*(const uint2*) (codes + ROW_D + (size_t) c * 8), lut, m1);
            } else {
                expand_codes(*(const uint2*) (codes + (size_t) c * 8), m0);
                expand_codes(*(const uint2*) (codes + ROW_D + (size_t) c * 8), m1);
            }
            dw0 = f16_ld(scales + (size_t) (c >> 1) * 2);
            dw1 = f16_ld(scales + SC_D * 2 + (size_t) (c >> 1) * 2);
        }
        for (int k = 0; k < ne; ++k) {
            float acc0 = 0.0f, acc1 = 0.0f;
            if (c < NC) {
                const int at = k * NC + c;
                int X[8];
#pragma unroll
                for (int j = 0; j < 8; ++j) X[j] = hs_w[j * (GMAX * NC) + at];
                const int2 dh = hs_dh[at];
                const float dx = __int_as_float(dh.x);
                int s0 = 0, s1 = 0;
                if (VAR != 3) {
                    if (VAR == 6) { s0 = chunk_s2(m0, X); s1 = chunk_s2(m1, X); }
                    else if (VAR == 7) { s0 = chunk_s4(m0, X); s1 = chunk_s4(m1, X); }
                    else { s0 = chunk_s(m0, X); s1 = chunk_s(m1, X); }
                }
                if (VAR != 4) {
                    acc0 += dw0 * dx * (float) (s0 - dh.y);
                    acc1 += dw1 * dx * (float) (s1 - dh.y);
                }
            }
            const float s0 = (VAR == 4) ? (float) (k + 1) : warp_sum(acc0);
            const float s1 = (VAR == 4) ? (float) (k + 1) : warp_sum(acc1);
            if (lane == 0) {
                const size_t o = (size_t) ent_dst[e0 + k] * H + (size_t) r;
                out[o] = s0;
                out[o + 1] = s1;
            }
        }
    }
}

// ---- the repack + m8n8k4 path (V8) -------------------------------------------------------------
//
// The QPN8 repack (llama.cpp-v100 q8-skinny.cu / q-skinny-common.cuh, 1Cat fp8_qpn8_sm70.cu): the codes of
// one 32-row tile and one 16-K group land at `codes[(tile * groups_k16 + g) * 32 + lane][slot]` where `lane`
// is `qpn8_lane_from_col(row & 31)` and `slot` is `qpn8_physical_k(j)`, so the runtime's lane reads ONE
// coalesced 4-byte record per (tile, group) and the m8n8k4 B fragment pairs physical slots (s, s + 4)
// exactly as its layout wants (qskinny_codec<Q2_K>::decode_record's pairing).  Pure permutation: the
// repacked codes are the same bytes as the canonical Q2_0 blob's, moved once at load time.

__device__ __forceinline__ int qpn8_col_from_lane(int lane) {
    return ((lane >> 2) & 3) * 8 + (lane & 3) + ((lane & 16) ? 4 : 0);
}
__device__ __forceinline__ int qpn8_lane_from_col(int col) {
    return (col & 3) | (((col >> 3) & 3) << 2) | (((col >> 2) & 1) << 4);
}
__device__ __forceinline__ int qpn8_physical_k(int logical_k) {
    const int local = logical_k & 7;
    return (logical_k & 8) + (local >> 1) + ((local & 1) << 2);
}

// 16 packed 2-bit codes -> 8 half2 in B-fragment order with the 1024 bias trick: values 1024..1027 are
// exact in f16, so subtracting 1024 gives the codes without rounding and without a branch.  half2 i holds
// physical slots (s, s + 4), s = (i & 3) + 8 * (i >> 2) - the pairing s8x8_to_half2x4 gives the Q8_0 case.
__device__ __forceinline__ void q2x16_to_bfrag(unsigned p, half2 out[8]) {
    const unsigned offset_bits = 0x64006400u;
    const half2 offset = *reinterpret_cast<const half2*>(&offset_bits);
#pragma unroll
    for (int i = 0; i < 8; ++i) {
        const int s = (i & 3) + 8 * (i >> 2);          // physical slots (s, s + 4)
        const unsigned q0 = (p >> (2 * s)) & 3u;
        const unsigned q1 = (p >> (2 * (s + 4))) & 3u;
        const unsigned h = (0x6400u | q0) | ((0x6400u | q1) << 16);
        out[i] = __hsub2(*reinterpret_cast<const half2*>(&h), offset);
    }
}

#define Q8_SKINNY_MMA_8N8K4(C, A0, A1, B0, B1)                       \
  asm volatile(                                                      \
      "mma.sync.aligned.m8n8k4.row.col.f32.f16.f16.f32 "             \
      "{%0,%1,%2,%3,%4,%5,%6,%7}, {%8,%9}, {%10,%11}, "              \
      "{%0,%1,%2,%3,%4,%5,%6,%7};\n"                                 \
      : "+f"(C[0]), "+f"(C[1]), "+f"(C[2]), "+f"(C[3]), "+f"(C[4]),  \
        "+f"(C[5]), "+f"(C[6]), "+f"(C[7])                           \
      : "r"(A0), "r"(A1), "r"(B0), "r"(B1))

// The same m8n8k4 flow reading the CANONICAL Q2_0 layout directly (no repack): each lane gathers its
// (row, k16-group) record from the row-major code plane - coalescing is lost, so this measures what the
// repack is worth.
__device__ __forceinline__ int qpn8_col_from_lane2(int lane) {
    return ((lane >> 2) & 3) * 8 + (lane & 3) + ((lane & 16) ? 4 : 0);
}
template <int SplitK, int NAcc>
__global__ void gu_hmma_gather_kernel(const unsigned char* __restrict__ codes, const half* __restrict__ input,
                                      float* __restrict__ out, int n_rows, int m, int groups_k16, int row_stride) {
    __shared__ float partials[SplitK][256];
    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;
    const int tile = blockIdx.x;
    const int row = (lane & 3) + ((lane & 16) ? 4 : 0);
    const int groups_per_warp = groups_k16 / SplitK;
    const int group_begin = warp * groups_per_warp;
    const int crow = tile * 32 + qpn8_col_from_lane2(lane);
    float accum[NAcc][8];
#pragma unroll
    for (int chain = 0; chain < NAcc; ++chain)
#pragma unroll
        for (int i = 0; i < 8; ++i) accum[chain][i] = 0.0f;
    __shared__ int inv[16];
    if (threadIdx.x < 16) {
        const int jmap[16] = {0, 4, 1, 5, 2, 6, 3, 7, 8, 12, 9, 13, 10, 14, 11, 15};   // physical_k
        inv[jmap[threadIdx.x]] = threadIdx.x;
    }
    __syncthreads();
    const unsigned offset_bits = 0x64006400u;
    const half2 offset = *reinterpret_cast<const half2*>(&offset_bits);
    for (int group = group_begin; group < group_begin + groups_per_warp; ++group) {
        // one canonical uint32 load: elements 16g..16g+15 are code bytes 4g..4g+3 of the row
        const unsigned* rec = (const unsigned*) (codes + (size_t) blockIdx.y * (size_t) (n_rows / 32) * 32 *
                                                   (size_t) row_stride + (size_t) crow * (size_t) row_stride) +
                              (size_t) group;
        unsigned packed = __ldg(rec);                        // canonical: code j at bits 2j
        half2 weights[8];
        // same fragment pairing as the repacked decode, codes pulled at the INVERSE physical_k slots
#pragma unroll
        for (int i = 0; i < 8; ++i) {
            const int s = (i & 3) + 8 * (i >> 2);
            const unsigned q0 = (packed >> (2 * inv[s])) & 3u;
            const unsigned q1 = (packed >> (2 * inv[s + 4])) & 3u;
            const unsigned h = (0x6400u | q0) | ((0x6400u | q1) << 16);
            weights[i] = __hsub2(*reinterpret_cast<const half2*>(&h), offset);
        }
        const unsigned* b = reinterpret_cast<const unsigned*>(weights);
        uint4 input01 = make_uint4(0, 0, 0, 0), input23 = make_uint4(0, 0, 0, 0);
        if (row < m) {
            const half* a = input + ((size_t) group * (size_t) m + row) * 16;
            input01 = *reinterpret_cast<const uint4*>(a);
            input23 = *reinterpret_cast<const uint4*>(a + 8);
        }
        const unsigned* a0 = reinterpret_cast<const unsigned*>(&input01);
        const unsigned* a1 = reinterpret_cast<const unsigned*>(&input23);
        Q8_SKINNY_MMA_8N8K4(accum[0], a0[0], a0[1], b[0], b[1]);
        Q8_SKINNY_MMA_8N8K4(accum[1 % NAcc], a0[2], a0[3], b[2], b[3]);
        Q8_SKINNY_MMA_8N8K4(accum[2 % NAcc], a1[0], a1[1], b[4], b[5]);
        Q8_SKINNY_MMA_8N8K4(accum[3 % NAcc], a1[2], a1[3], b[6], b[7]);
    }
#pragma unroll
    for (int chain = 1; chain < NAcc; ++chain)
#pragma unroll
        for (int i = 0; i < 8; ++i) accum[0][i] += accum[chain][i];
    for (int i = 0; i < 8; ++i) {
        const int output_row = (i & 2) + ((lane & 16) ? 4 : 0) + (lane & 1);
        const int output_col = (i & 1) | (((lane >> 1) & 1) << 1) | ((i >> 2) << 2);
        const int col = ((lane >> 2) & 3) * 8 + output_col;
        partials[warp][output_row * 32 + col] = accum[0][i];
    }
    __syncthreads();
    for (int e = threadIdx.x; e < 256; e += blockDim.x) {
        float v = 0.0f;
#pragma unroll
        for (int w = 0; w < SplitK; ++w) v += partials[w][e];
        const int r = e >> 5, c = e & 31;
        if (tile * 32 + c < n_rows && r < 8) out[(size_t) r * (size_t) n_rows + tile * 32 + c] = v;
    }
}

// q8_skinny_kernel's GEMV dataflow with Q2_0 codes: one warp-block per 32-row tile, the M tokens in the m8
// rows, SplitK warps over the k16 groups, NAcc accumulator chains, fixed-order split reduce.  The output
// writeout is q8_skinny's M>1 map (one 8x32 tile); the engine kernel would apply the per-chunk float
// expression at the readout - here only the load/decode/mma mix is timed.
template <int SplitK, int NAcc>
__global__ void gu_hmma_kernel(const unsigned int* __restrict__ codes, const half* __restrict__ input,
                               float* __restrict__ out, int n_rows, int m, int groups_k16) {
    __shared__ float partials[SplitK][256];
    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;
    const int tile = blockIdx.x;
    const int row = (lane & 3) + ((lane & 16) ? 4 : 0);
    const int groups_per_warp = groups_k16 / SplitK;
    const int group_begin = warp * groups_per_warp;
    const unsigned* code_ptr =
        codes + ((size_t) blockIdx.y * (size_t) (n_rows / 32) * groups_k16 + (size_t) tile * groups_k16 +
                 group_begin) * 32 + lane;
    float accum[NAcc][8];
#pragma unroll
    for (int chain = 0; chain < NAcc; ++chain)
#pragma unroll
        for (int i = 0; i < 8; ++i) accum[chain][i] = 0.0f;
    for (int group = group_begin; group < group_begin + groups_per_warp; ++group) {
        const unsigned packed = __ldcs(code_ptr + (size_t) (group - group_begin) * 32);
        half2 weights[8];
        q2x16_to_bfrag(packed, weights);
        const unsigned* b = reinterpret_cast<const unsigned*>(weights);
        uint4 input01 = make_uint4(0, 0, 0, 0), input23 = make_uint4(0, 0, 0, 0);
        if (row < m) {
            const half* a = input + ((size_t) group * (size_t) m + row) * 16;
            input01 = *reinterpret_cast<const uint4*>(a);
            input23 = *reinterpret_cast<const uint4*>(a + 8);
        }
        const unsigned* a0 = reinterpret_cast<const unsigned*>(&input01);
        const unsigned* a1 = reinterpret_cast<const unsigned*>(&input23);
        Q8_SKINNY_MMA_8N8K4(accum[0], a0[0], a0[1], b[0], b[1]);
        Q8_SKINNY_MMA_8N8K4(accum[1 % NAcc], a0[2], a0[3], b[2], b[3]);
        Q8_SKINNY_MMA_8N8K4(accum[2 % NAcc], a1[0], a1[1], b[4], b[5]);
        Q8_SKINNY_MMA_8N8K4(accum[3 % NAcc], a1[2], a1[3], b[6], b[7]);
    }
#pragma unroll
    for (int chain = 1; chain < NAcc; ++chain)
#pragma unroll
        for (int i = 0; i < 8; ++i) accum[0][i] += accum[chain][i];
    for (int i = 0; i < 8; ++i) {
        const int output_row = (i & 2) + ((lane & 16) ? 4 : 0) + (lane & 1);
        const int output_col = (i & 1) | (((lane >> 1) & 1) << 1) | ((i >> 2) << 2);
        const int col = ((lane >> 2) & 3) * 8 + output_col;
        partials[warp][output_row * 32 + col] = accum[0][i];
    }
    __syncthreads();
    for (int e = threadIdx.x; e < 256; e += blockDim.x) {
        float v = 0.0f;
#pragma unroll
        for (int w = 0; w < SplitK; ++w) v += partials[w][e];
        const int r = e >> 5, c = e & 31;
        if (tile * 32 + c < n_rows && r < 8) out[(size_t) r * (size_t) n_rows + tile * 32 + c] = v;
    }
}


// ---- host --------------------------------------------------------------------------------------

struct Blob {
    std::vector<uint8_t> h;
};

void fill_blob(uint8_t* b, std::mt19937& rng) {
    for (size_t i = 0; i < O_GU_SCALES; ++i) b[i] = (uint8_t) rng();
    for (size_t i = O_GU_SCALES; i < BLOB; i += 2) {
        const uint16_t bits = 0x3C00;   // 1.0f-ish placeholder scale bits; values do not matter for timing
        (void) bits;
        b[i] = (uint8_t) rng();
        b[i + 1] = (uint8_t) (rng() & 0x3F);
    }
}

void fill_x(std::vector<uint8_t>& x, std::vector<float>& xs, int n_tok, std::mt19937& rng) {
    x.assign((size_t) n_tok * NC_GU * 34, 0);
    xs.assign((size_t) n_tok * NC_GU, 0.0f);
    for (size_t c = 0; c < (size_t) n_tok * NC_GU; ++c) {
        uint8_t* b = x.data() + c * 34;
        xs[c] = 1e-3f * (float) (1 + rng() % 32);
        for (int j = 2; j < 34; ++j) b[j] = (uint8_t) (int8_t) (rng() % 255 - 127);
    }
}

// pre-expanded codes for V5: 8 int words per (row, chunk) of each plane
std::vector<int> expand_plane(const uint8_t* codes, int rows, int chunks, size_t row_stride) {
    std::vector<int> out((size_t) rows * chunks * 8);
    for (int r = 0; r < rows; ++r) {
        for (int c = 0; c < chunks; ++c) {
            const uint8_t* cb = codes + (size_t) r * row_stride + (size_t) c * 8;
            for (int j = 0; j < 8; ++j) {
                const unsigned v = cb[j];
                out[((size_t) r * chunks + c) * 8 + j] =
                    (int) ((v & 3u) | (((v >> 2) & 3u) << 8) | (((v >> 4) & 3u) << 16) | (((v >> 6) & 3u) << 24));
            }
        }
    }
    return out;
}

// S2T repack of one blob (host): the code planes' chunks bit-transposed; scales untouched.
void s2t_repack_blob(uint8_t* b) {
    auto plane = [b](size_t off, int rows, int chunks, size_t row_stride) {
        for (int r = 0; r < rows; ++r) {
            uint8_t* row = b + off + (size_t) r * row_stride;
            for (int c = 0; c < chunks; ++c) {
                const uint8_t* in = row + (size_t) c * 8;
                uint8_t out[8];
                for (int h = 0; h < 2; ++h)
                    for (int f = 0; f < 4; ++f) {
                        unsigned v = 0;
                        for (int j = 0; j < 4; ++j) {
                            const int e = 16 * h + 4 * j + f;
                            const unsigned code = ((unsigned) in[e / 4] >> (2 * (e % 4))) & 3u;
                            v |= code << (2 * j);
                        }
                        out[4 * h + f] = (uint8_t) v;
                    }
                for (int j = 0; j < 8; ++j) row[(size_t) c * 8 + j] = out[j];
            }
        }
    };
    plane(0, 2 * FF, NC_GU, ROW_GU);
    plane(O_D_CODES, H, NC_D, ROW_D);
}

// host-side qpn8 maps + the repack: canonical Q2_0 code plane -> [(row/32) tiles][k16 groups][32 lanes][4 B]
int host_qpn8_lane_from_col(int col) {
    return (col & 3) | (((col >> 3) & 3) << 2) | (((col >> 2) & 1) << 4);
}
int host_qpn8_physical_k(int logical_k) {
    const int local = logical_k & 7;
    return (logical_k & 8) + (local >> 1) + ((local & 1) << 2);
}
std::vector<unsigned> repack_plane(const uint8_t* codes, int rows, size_t row_stride, int K) {
    const int groups_k16 = K / 16, tiles = rows / 32;
    std::vector<unsigned> out((size_t) tiles * groups_k16 * 32);
    for (int t = 0; t < tiles; ++t)
        for (int g = 0; g < groups_k16; ++g)
            for (int row = 0; row < 32; ++row) {
                unsigned rec = 0;
                for (int j = 0; j < 16; ++j) {
                    const int e = 16 * g + j;
                    const uint8_t* cb = codes + (size_t) (t * 32 + row) * row_stride + (size_t) (e / 4);
                    const unsigned c = ((unsigned) cb[0] >> ((e % 4) * 2)) & 3u;
                    rec |= c << (2 * host_qpn8_physical_k(j));
                }
                out[((size_t) t * groups_k16 + g) * 32 + host_qpn8_lane_from_col(row)] = rec;
            }
    return out;
}

// pre-formed activations for V5: 8 int words + hx per (token, chunk)
unsigned host_byte_perm(unsigned a, unsigned b, unsigned s) {
    unsigned r = 0;
    for (int i = 0; i < 4; ++i) {
        const unsigned sel = (s >> (8 * i)) & 0xFF;
        const unsigned src = (sel & 4) ? b : a;
        const unsigned sh = (sel & 3) * 8;
        r |= ((src >> sh) & 0xFF) << (8 * i);
    }
    return r;
}
void form_x(int n_tok, int chunks, std::vector<int>& formed, std::vector<int>& hxv) {
    formed.assign((size_t) n_tok * chunks * 8, 0);
    hxv.assign((size_t) n_tok * chunks, 0);
    std::mt19937 rng(11);
    for (int t = 0; t < n_tok; ++t) {
        for (int c = 0; c < chunks; ++c) {
            unsigned n[8];
            int hx = 0;
            for (int j = 0; j < 8; ++j) {
                n[j] = rng();
                for (int e = 0; e < 4; ++e) hx += (int) (int8_t) (n[j] >> (8 * e));
            }
            hxv[(size_t) t * chunks + c] = hx;
            // same 4x4 transpose as load_x_chunk (this copy assumes 4-byte-aligned rows)
            unsigned m[8];
            for (int h = 0; h < 2; ++h) {
                const unsigned t0 = host_byte_perm(n[4 * h], n[4 * h + 1], 0x5140);
                const unsigned t1 = host_byte_perm(n[4 * h], n[4 * h + 1], 0x7362);
                const unsigned t2 = host_byte_perm(n[4 * h + 2], n[4 * h + 3], 0x5140);
                const unsigned t3 = host_byte_perm(n[4 * h + 2], n[4 * h + 3], 0x7362);
                m[4 * h + 0] = host_byte_perm(t0, t2, 0x5410);
                m[4 * h + 1] = host_byte_perm(t0, t2, 0x7632);
                m[4 * h + 2] = host_byte_perm(t1, t3, 0x5410);
                m[4 * h + 3] = host_byte_perm(t1, t3, 0x7632);
            }
            for (int j = 0; j < 8; ++j) formed[((size_t) t * chunks + c) * 8 + j] = (int) m[j];
        }
    }
}

double time_ms(const std::function<void(int)>& call, int iters = 200) {
    cudaEvent_t e0, e1;
    CHK(cudaEventCreate(&e0));
    CHK(cudaEventCreate(&e1));
    call(0);
    CHK(cudaDeviceSynchronize());
    CHK(cudaEventRecord(e0));
    for (int i = 0; i < iters; ++i) call(i);
    CHK(cudaEventRecord(e1));
    CHK(cudaEventSynchronize(e1));
    float ms = 0;
    CHK(cudaEventElapsedTime(&ms, e0, e1));
    CHK(cudaEventDestroy(e0));
    CHK(cudaEventDestroy(e1));
    return (double) ms / iters * 1000.0;   // us per call
}

}  // namespace

int main(int argc, char** argv) {
    int M = 8, groups = 10;
    if (argc > 1) M = std::atoi(argv[1]);
    if (argc > 2) groups = std::atoi(argv[2]);
    int dev = 0;
    cudaDeviceProp prop{};
    CHK(cudaGetDevice(&dev));
    CHK(cudaGetDeviceProperties(&prop, dev));
    const int nb = std::max(48, (int) (3ull * (size_t) prop.l2CacheSize / BLOB) + 8);
    std::printf("device %s, L2 %d MB, %d blobs cycled, M=%d entries, %d groups (all shared: ne=%d)\n", prop.name,
                prop.l2CacheSize >> 20, nb, M, groups, M);

    std::mt19937 rng(7);
    std::vector<uint8_t> host_blobs((size_t) nb * BLOB);
    for (int i = 0; i < nb; ++i) fill_blob(host_blobs.data() + (size_t) i * BLOB, rng);
    uint8_t* d_blobs = nullptr;
    CHK(cudaMalloc(&d_blobs, host_blobs.size()));
    CHK(cudaMemcpy(d_blobs, host_blobs.data(), host_blobs.size(), cudaMemcpyHostToDevice));

    std::vector<uint8_t> host_s2t = host_blobs;
    for (int i = 0; i < nb; ++i) s2t_repack_blob(host_s2t.data() + (size_t) i * BLOB);
    uint8_t* d_s2t = nullptr;
    CHK(cudaMalloc(&d_s2t, host_s2t.size()));
    CHK(cudaMemcpy(d_s2t, host_s2t.data(), host_s2t.size(), cudaMemcpyHostToDevice));

    std::vector<uint8_t> x;
    std::vector<float> xs;
    fill_x(x, xs, M, rng);
    uint8_t* d_x = nullptr;
    float* d_xs = nullptr;
    CHK(cudaMalloc(&d_x, x.size()));
    CHK(cudaMemcpy(d_x, x.data(), x.size(), cudaMemcpyHostToDevice));
    CHK(cudaMalloc(&d_xs, xs.size() * sizeof(float)));
    CHK(cudaMemcpy(d_xs, xs.data(), xs.size() * sizeof(float), cudaMemcpyHostToDevice));

    // V5's pre-formed operands (one expert's planes reused for all groups; 4x codes)
    std::vector<int> mwords_gu = expand_plane(host_blobs.data(), 2 * FF, NC_GU, ROW_GU);
    std::vector<int> mwords_dn = expand_plane(host_blobs.data() + O_D_CODES, H, NC_D, ROW_D);
    std::vector<int> xformed, xhx;
    form_x(M, NC_GU, xformed, xhx);
    std::vector<int> hformed, hhx;
    form_x(groups * M, NC_D, hformed, hhx);
    // NOTE: form_x walks 34-byte blocks of the row; the down plane in the engine is the quantized
    // intermediate (also block_q8_0), so the same preparation applies.  For timing it need not be the
    // same numbers as the gu input.
    int *d_mgu = nullptr, *d_mdn = nullptr, *d_xf = nullptr, *d_xh = nullptr, *d_hf = nullptr, *d_hh = nullptr;
    CHK(cudaMalloc(&d_mgu, mwords_gu.size() * 4));
    CHK(cudaMemcpy(d_mgu, mwords_gu.data(), mwords_gu.size() * 4, cudaMemcpyHostToDevice));
    CHK(cudaMalloc(&d_mdn, mwords_dn.size() * 4));
    CHK(cudaMemcpy(d_mdn, mwords_dn.data(), mwords_dn.size() * 4, cudaMemcpyHostToDevice));
    CHK(cudaMalloc(&d_xf, xformed.size() * 4));
    CHK(cudaMemcpy(d_xf, xformed.data(), xformed.size() * 4, cudaMemcpyHostToDevice));
    CHK(cudaMalloc(&d_xh, xhx.size() * 4));
    CHK(cudaMemcpy(d_xh, xhx.data(), xhx.size() * 4, cudaMemcpyHostToDevice));
    CHK(cudaMalloc(&d_hf, hformed.size() * 4));
    CHK(cudaMemcpy(d_hf, hformed.data(), hformed.size() * 4, cudaMemcpyHostToDevice));
    CHK(cudaMalloc(&d_hh, hhx.size() * 4));
    CHK(cudaMemcpy(d_hh, hhx.data(), hhx.size() * 4, cudaMemcpyHostToDevice));

    // groups: all share one expert each; entries = M; blobs cycle
    const int cap = groups * M;
    std::vector<unsigned long long> gptr(groups);
    std::vector<int32_t> gstart(groups + 1), ng(1), edst(cap), etok(cap);
    for (int g = 0; g < groups; ++g) {
        gptr[g] = (unsigned long long) (d_blobs + (size_t) (g % nb) * BLOB);
        gstart[g] = g * M;
        for (int e = 0; e < M; ++e) {
            edst[g * M + e] = g * M + e;
            etok[g * M + e] = e;
        }
    }
    gstart[groups] = cap;
    ng[0] = groups;
    unsigned long long* d_gptr = nullptr;
    int32_t *d_gstart = nullptr, *d_ng = nullptr, *d_edst = nullptr, *d_etok = nullptr;
    CHK(cudaMalloc(&d_gptr, gptr.size() * 8));
    CHK(cudaMemcpy(d_gptr, gptr.data(), gptr.size() * 8, cudaMemcpyHostToDevice));
    CHK(cudaMalloc(&d_gstart, gstart.size() * 4));
    CHK(cudaMemcpy(d_gstart, gstart.data(), gstart.size() * 4, cudaMemcpyHostToDevice));
    CHK(cudaMalloc(&d_ng, 4));
    CHK(cudaMemcpy(d_ng, ng.data(), 4, cudaMemcpyHostToDevice));
    CHK(cudaMalloc(&d_edst, edst.size() * 4));
    CHK(cudaMemcpy(d_edst, edst.data(), edst.size() * 4, cudaMemcpyHostToDevice));
    CHK(cudaMalloc(&d_etok, etok.size() * 4));
    CHK(cudaMemcpy(d_etok, etok.data(), etok.size() * 4, cudaMemcpyHostToDevice));

    float* d_gu = nullptr;
    CHK(cudaMalloc(&d_gu, (size_t) cap * 2 * FF * 4 + 256));
    float* d_out = nullptr;
    CHK(cudaMalloc(&d_out, (size_t) cap * H * 4 + 256));
    uint8_t* d_hq = nullptr;
    CHK(cudaMalloc(&d_hq, (size_t) cap * NC_D * 34 + 256));
    CHK(cudaMemcpy(d_hq, x.data(), std::min(x.size(), (size_t) cap * NC_D * 34), cudaMemcpyHostToDevice));
    float* d_hs = nullptr;
    CHK(cudaMalloc(&d_hs, (size_t) cap * NC_D * 4 + 256));
    CHK(cudaMemcpy(d_hs, xs.data(), std::min(xs.size() * 4, (size_t) cap * NC_D * 4), cudaMemcpyHostToDevice));

    // V8: repacked codes (QPN8) for the two planes, one blob per group as the other variants cycle them,
    // + [group][M][16] half activations
    std::vector<unsigned> rec_gu, rec_dn;
    for (int g = 0; g < groups; ++g) {
        const uint8_t* b = host_blobs.data() + (size_t) (g % nb) * BLOB;
        std::vector<unsigned> a = repack_plane(b, 2 * FF, ROW_GU, H);
        std::vector<unsigned> d = repack_plane(b + O_D_CODES, H, ROW_D, FF);
        rec_gu.insert(rec_gu.end(), a.begin(), a.end());
        rec_dn.insert(rec_dn.end(), d.begin(), d.end());
    }
    unsigned *d_rec_gu = nullptr, *d_rec_dn = nullptr;
    CHK(cudaMalloc(&d_rec_gu, rec_gu.size() * 4));
    CHK(cudaMemcpy(d_rec_gu, rec_gu.data(), rec_gu.size() * 4, cudaMemcpyHostToDevice));
    CHK(cudaMalloc(&d_rec_dn, rec_dn.size() * 4));
    CHK(cudaMemcpy(d_rec_dn, rec_dn.data(), rec_dn.size() * 4, cudaMemcpyHostToDevice));
    std::vector<uint16_t> xin((size_t) (H / 16) * M * 16);
    for (auto& v : xin) v = (uint16_t) (rng() & 0xFFFF);
    uint16_t* d_xin = nullptr;
    CHK(cudaMalloc(&d_xin, xin.size() * 2));
    CHK(cudaMemcpy(d_xin, xin.data(), xin.size() * 2, cudaMemcpyHostToDevice));
    float* d_hout = nullptr;
    CHK(cudaMalloc(&d_hout, (size_t) 8 * H * 4 + 256));

    const dim3 grid_gu((unsigned) (2 * FF / GU_ROWS), (unsigned) groups);
    const dim3 grid_dn((unsigned) (H / D_ROWS), (unsigned) groups);
    // NOTE: the engines' kernels launch cap_groups as the grid's y; this bench launches `groups` (the real
    // count) for every variant so the comparison is the INNER LOOP, not the wasted-block scheduling.
    const int* null_i = nullptr;

    struct V { const char* name; int var; };
    const V vs[] = {{"0 base        ", 0}, {"1 no-expand   ", 1}, {"2 no-xload    ", 2},
                    {"3 no-dp4a     ", 3}, {"4 no-sum      ", 4}, {"5 clean (4x) ", 5},
                    {"6 chains2     ", 6}, {"7 chains4     ", 7}, {"8 hmma s16/8  ", 8}, {"9 hmma gather ", 9}, {"10 hmma s8/4  ", 10}, {"11 hmma s4/4  ", 11}, {"12 hmma s8/4 n2", 12}, {"13 s2t+lut dp4a", 13}};
    for (const V& v : vs) {
        const double us = time_ms([&](int) {
            switch (v.var) {
                case 0:
                    gu_kernel<0><<<grid_gu, THREADS>>>(d_gptr, d_gstart, d_ng, d_etok, d_x, d_xs, null_i, null_i,
                                                       null_i, d_gu, cap);
                    down_kernel<0><<<grid_dn, THREADS>>>(d_gptr, d_gstart, d_ng, d_edst, d_hq, d_hs, null_i, null_i,
                                                         null_i, d_out);
                    break;
                case 1:
                    gu_kernel<1><<<grid_gu, THREADS>>>(d_gptr, d_gstart, d_ng, d_etok, d_x, d_xs, null_i, null_i,
                                                       null_i, d_gu, cap);
                    down_kernel<1><<<grid_dn, THREADS>>>(d_gptr, d_gstart, d_ng, d_edst, d_hq, d_hs, null_i, null_i,
                                                         null_i, d_out);
                    break;
                case 2:
                    gu_kernel<2><<<grid_gu, THREADS>>>(d_gptr, d_gstart, d_ng, d_etok, d_x, d_xs, null_i, null_i,
                                                       null_i, d_gu, cap);
                    down_kernel<2><<<grid_dn, THREADS>>>(d_gptr, d_gstart, d_ng, d_edst, d_hq, d_hs, null_i, null_i,
                                                         null_i, d_out);
                    break;
                case 3:
                    gu_kernel<3><<<grid_gu, THREADS>>>(d_gptr, d_gstart, d_ng, d_etok, d_x, d_xs, null_i, null_i,
                                                       null_i, d_gu, cap);
                    down_kernel<3><<<grid_dn, THREADS>>>(d_gptr, d_gstart, d_ng, d_edst, d_hq, d_hs, null_i, null_i,
                                                         null_i, d_out);
                    break;
                case 4:
                    gu_kernel<4><<<grid_gu, THREADS>>>(d_gptr, d_gstart, d_ng, d_etok, d_x, d_xs, null_i, null_i,
                                                       null_i, d_gu, cap);
                    down_kernel<4><<<grid_dn, THREADS>>>(d_gptr, d_gstart, d_ng, d_edst, d_hq, d_hs, null_i, null_i,
                                                         null_i, d_out);
                    break;
                case 5:
                    gu_kernel<5><<<grid_gu, THREADS>>>(d_gptr, d_gstart, d_ng, d_etok, d_x, d_xs, d_mgu, d_xf, d_xh,
                                                       d_gu, cap);
                    down_kernel<5><<<grid_dn, THREADS>>>(d_gptr, d_gstart, d_ng, d_edst, d_hq, d_hs, d_mdn, d_hf,
                                                         d_hh, d_out);
                    break;
                case 6:
                    gu_kernel<6><<<grid_gu, THREADS>>>(d_gptr, d_gstart, d_ng, d_etok, d_x, d_xs, null_i, null_i,
                                                       null_i, d_gu, cap);
                    down_kernel<6><<<grid_dn, THREADS>>>(d_gptr, d_gstart, d_ng, d_edst, d_hq, d_hs, null_i, null_i,
                                                         null_i, d_out);
                    break;
                case 7:
                    gu_kernel<7><<<grid_gu, THREADS>>>(d_gptr, d_gstart, d_ng, d_etok, d_x, d_xs, null_i, null_i,
                                                       null_i, d_gu, cap);
                    down_kernel<7><<<grid_dn, THREADS>>>(d_gptr, d_gstart, d_ng, d_edst, d_hq, d_hs, null_i, null_i,
                                                         null_i, d_out);
                    break;
                case 9:
                    // gather variant: canonical layout, no repack
                    for (int g = 0; g < groups; ++g) {
                        const uint8_t* b = d_blobs + (size_t) (g % nb) * BLOB;
                        gu_hmma_gather_kernel<16, 4><<<dim3((unsigned) (2 * FF / 32)), 16 * 32>>>(
                            b, (const half*) d_xin, d_hout, 2 * FF, M, H / 16, ROW_GU);
                        gu_hmma_gather_kernel<8, 4><<<dim3((unsigned) (H / 32)), 8 * 32>>>(
                            b + O_D_CODES, (const half*) d_xin, d_hout, H, M, FF / 16, ROW_D);
                    }
                    break;
                case 8: {
                    // QPN8 repacked codes + m8n8k4: gu plane (K=2560, 1280 rows) then down (K=640, 2560 rows)
                    gu_hmma_kernel<16, 4><<<dim3((unsigned) (2 * FF / 32), (unsigned) groups), 16 * 32>>>(
                        d_rec_gu, (const half*) d_xin, d_hout, 2 * FF, M, H / 16);
                    gu_hmma_kernel<8, 4><<<dim3((unsigned) (H / 32), (unsigned) groups), 8 * 32>>>(
                        d_rec_dn, (const half*) d_xin, d_hout, H, M, FF / 16);
                    break;
                }
                case 13: {
                    // S2T-repacked blobs (same bytes, within-chunk bit transpose) + LUT expansion
                    std::vector<unsigned long long> g2(groups);
                    for (int g = 0; g < groups; ++g) g2[g] = (unsigned long long) (d_s2t + (size_t) (g % nb) * BLOB);
                    unsigned long long* d_g2 = nullptr;
                    CHK(cudaMalloc(&d_g2, g2.size() * 8));
                    CHK(cudaMemcpy(d_g2, g2.data(), g2.size() * 8, cudaMemcpyHostToDevice));
                    gu_kernel<13><<<grid_gu, THREADS>>>(d_g2, d_gstart, d_ng, d_etok, d_x, d_xs, null_i, null_i,
                                                        null_i, d_gu, cap);
                    down_kernel<13><<<grid_dn, THREADS>>>(d_g2, d_gstart, d_ng, d_edst, d_hq, d_hs, null_i, null_i,
                                                          null_i, d_out);
                    CHK(cudaFree(d_g2));
                    break;
                }
                case 10:
                    gu_hmma_kernel<8, 4><<<dim3((unsigned) (2 * FF / 32), (unsigned) groups), 8 * 32>>>(
                        d_rec_gu, (const half*) d_xin, d_hout, 2 * FF, M, H / 16);
                    gu_hmma_kernel<4, 4><<<dim3((unsigned) (H / 32), (unsigned) groups), 4 * 32>>>(
                        d_rec_dn, (const half*) d_xin, d_hout, H, M, FF / 16);
                    break;
                case 11:
                    gu_hmma_kernel<4, 4><<<dim3((unsigned) (2 * FF / 32), (unsigned) groups), 4 * 32>>>(
                        d_rec_gu, (const half*) d_xin, d_hout, 2 * FF, M, H / 16);
                    gu_hmma_kernel<4, 4><<<dim3((unsigned) (H / 32), (unsigned) groups), 4 * 32>>>(
                        d_rec_dn, (const half*) d_xin, d_hout, H, M, FF / 16);
                    break;
                case 12:
                    gu_hmma_kernel<8, 2><<<dim3((unsigned) (2 * FF / 32), (unsigned) groups), 8 * 32>>>(
                        d_rec_gu, (const half*) d_xin, d_hout, 2 * FF, M, H / 16);
                    gu_hmma_kernel<4, 2><<<dim3((unsigned) (H / 32), (unsigned) groups), 4 * 32>>>(
                        d_rec_dn, (const half*) d_xin, d_hout, H, M, FF / 16);
                    break;
            }
        });
        const double gb = (double) groups * BLOB / 1e3;   // MB per call -> GB/s when divided by us
        std::printf("  %s  %8.1f us   (%6.0f GB/s of blob bytes)\n", v.name, us, gb / us);
    }
    std::printf("  (blob bytes per call: %.1f MB; DRAM floor at %.0f GB/s = %.1f us)\n",
                groups * (double) BLOB / 1e6, 900.0, groups * (double) BLOB / 1e6 / 900.0 * 1000.0);
    return 0;
}
