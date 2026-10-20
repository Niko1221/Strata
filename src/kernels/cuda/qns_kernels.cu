// qns_kernels.cu - Qwen Next S routed experts on CUDA (qns_cuda.hpp). A port of the vLLM plugin's kernels
// (qwen_next_s/kernels.cu): the same word layout, hash, level map and exact int32 dot products.
//
// A blob row of `cols` rotated weights is a 16-bit trellis path of cols / 2 states (FORMAT.md): state t is the 16-bit
// window at bit t T of the row's MSB-first tape and emits columns 2t, 2t + 1 as scale * (level_j - MEAN_j) / RMS,
// level_j = level(byte j of fmix(state)). The blob holds each row's tape as big-endian u32 words, 32 rows interleaved
// per word ([group][word][lane], W = cols / 64 * T + 1 words a row): a warp with lane = row reads one word of its 32
// rows as one 128-byte load. A 64-column chunk is exactly T words, so chunk k starts at word k T and, with the next
// word, every state of it is a funnel shift at a compile-time offset.
//
// Activations are rotated, x_rot = (H (x) Q)(signs * x), and quantized per token: x_rot ~ step (q0 + q1 / 254). A row's
// output finishes the exact sums C = sum (level + 54) q0, F = sum (level + 54) q1 (finish()).

#include "strata/kernels/qns_cuda.hpp"

#include <cuda_fp16.h>

#include <cstdio>
#include <cstdlib>

namespace strata::kernels {
namespace {

typedef unsigned int u32;

constexpr u32 kMagic = 0x31534E51u;  // 'QNS1'
constexpr float kMean0 = 1.5769348f, kMean1 = 1.4910126f, kRms = 19.183672f;

// The odd factors: Q_5 of the 2560 (0) and 640 (1) rotations and Q_3 of the 6144 one, [o][c]; every lane reads the
// same entry (a constant-cache broadcast). The signs: each lane reads its own columns.
__constant__ float c_q5[2][25], c_q3[9];
__device__ float g_signs2560[2560], g_signs640[640], g_signs6144[6144];

// Rotation widths: 2560 = 512 x 5 (the hidden state), 640 = 128 x 5 (the expert down input), 6144 = 2048 x 3.
template <int COLS>
constexpr int kOdd = COLS == 6144 ? 3 : 5;
template <int COLS>
constexpr float kNorm = COLS == 2560 ? 0.044194173824159216f : COLS == 640 ? 0.088388347648318441f : 0.022097086912079608f;  // 1 / sqrt(COLS / kOdd)
template <int COLS>
__device__ __forceinline__ float q_entry(int i) {
    if constexpr (COLS == 6144) return c_q3[i];
    else return c_q5[COLS == 2560 ? 0 : 1][i];
}
template <int COLS>
__device__ __forceinline__ const float* signs_of() {
    return COLS == 2560 ? g_signs2560 : COLS == 640 ? g_signs640 : g_signs6144;
}

// One token's rotated, quantized activation: q[g] = the plane-0 and the plane-1 bytes of columns 4g .. 4g + 3.
struct Stats {
    float step;
    int sum_q0, sum_q1;
    float even, odd;  // step (sum q0 + sum q1 / 254) over the even / the odd columns
    int pad[3];
};
template <int COLS>
struct Act {
    uint2 q[COLS / 4];
    Stats st;
};
static_assert(sizeof(Act<2560>) % 16 == 0 && sizeof(Act<640>) % 16 == 0 && sizeof(Act<6144>) % 16 == 0, "records stay 16-byte aligned");

void check(const char* what) {
    const cudaError_t e = cudaGetLastError();
    if (e == cudaSuccess) return;
    std::fprintf(stderr, "%s: %s\n", what, cudaGetErrorString(e));
    std::abort();
}

// Bytes 0, 1 of fmix32(s0 * 0xCFCCB83F + 0x584B4AA3) and of the same for s1, as one word. Only the low half of the last
// step x ^ (x >> 16) is used, and it is lo16(x) ^ hi16(x): two byte_perms and one xor for both states.
__device__ __forceinline__ u32 fmix_hash_pair(u32 s0, u32 s1) {
    u32 x0 = s0 * 0xCFCCB83Fu + 0x584B4AA3u, x1 = s1 * 0xCFCCB83Fu + 0x584B4AA3u;
    x0 ^= x0 >> 16;
    x1 ^= x1 >> 16;
    x0 *= 0x85EBCA6Bu;
    x1 *= 0x85EBCA6Bu;
    return __byte_perm(x0, x1, 0x5410) ^ __byte_perm(x0, x1, 0x7632);
}

// 2 (level(b) + 54) for each byte b of h, in [0, 218]: 16 (sum of b's four 2-bit fields) + 2 ((3 (b & 15)) & 15).
// Doubled because then each term takes one mask (2 fewer ALU-pipe ops; the decode is ALU-bound). Per nibble,
// h - 3 ((h >> 2) & 3) is the sum of its two fields; times 17 adds the low nibble's sum into the high one (no carries).
__device__ __forceinline__ u32 levels_x2(u32 h) {
    const u32 pairs = h - 3u * ((h >> 2) & 0x33333333u);
    return ((pairs * 17u) & 0xF0F0F0F0u) + (((h & 0x0F0F0F0Fu) * 6u) & 0x1E1E1E1Eu);
}

// c += a (16 x 32 u8, row-major) * b (32 x 8 s8, column-major) on the int8 tensor cores; exact int32 sums.
__device__ __forceinline__ void mma_u8s8(int (&c)[4], u32 a0, u32 a1, u32 a2, u32 a3, u32 b0, u32 b1) {
    asm volatile("mma.sync.aligned.m16n8k32.row.col.s32.u8.s8.s32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, "
                 "{%0, %1, %2, %3};"
                 : "+r"(c[0]), "+r"(c[1]), "+r"(c[2]), "+r"(c[3])
                 : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
}

// One decoded chunk (lane = row: 16 words of 4 doubled levels, columns 4g .. 4g + 3) times 8 B columns on the tensor
// cores. The warp's tile (row stride 20 words: conflict-free 16-byte writes and fragment reads) turns the rows into
// mma m16n8k32 A fragments (16 rows x 32 columns); xq is this lane's B source, the plane words of B column gid (b0: its
// columns 4 tig .. 4 tig + 3, b1: 16 more). acc: [row tile: rows 0-15, 16-31][C fragment].
__device__ __forceinline__ void mma_chunk(u32 (&tile)[32][20], const u32 (&levels)[16], const u32* __restrict__ xq, int chunk,
                                          int (&acc)[2][4]) {
    const int lane = threadIdx.x & 31, gid = lane >> 2, tig = lane & 3;
#pragma unroll
    for (int k = 0; k < 4; ++k)
        reinterpret_cast<uint4*>(tile[lane])[k] = make_uint4(levels[4 * k], levels[4 * k + 1], levels[4 * k + 2], levels[4 * k + 3]);
    __syncwarp();
#pragma unroll
    for (int kt = 0; kt < 2; ++kt) {  // columns 32 kt .. 32 kt + 31 of the chunk
        const int word = chunk * 16 + 8 * kt + tig;
        const u32 b0 = __ldg(xq + 2 * word), b1 = __ldg(xq + 2 * (word + 4));
#pragma unroll
        for (int rt = 0; rt < 2; ++rt)
            mma_u8s8(acc[rt], tile[16 * rt + gid][8 * kt + tig], tile[16 * rt + gid + 8][8 * kt + tig],
                     tile[16 * rt + gid][8 * kt + 4 + tig], tile[16 * rt + gid + 8][8 * kt + 4 + tig], b0, b1);
    }
    __syncwarp();  // the next chunk rewrites the tile
}


// The 32 states of one chunk from its T words and the next one, as 16 words of four 2 (level + 54) (columns 4g .. 4g + 3).
template <int T>
__device__ __forceinline__ void decode_chunk(const u32 (&w)[T + 1], u32 (&levels)[16]) {
#pragma unroll
    for (int g = 0; g < 16; ++g) {
        const int b0 = 2 * g * T, b1 = (2 * g + 1) * T;
        const u32 s0 = __funnelshift_l(w[b0 / 32 + 1], w[b0 / 32], b0 % 32) >> 16;
        const u32 s1 = __funnelshift_l(w[b1 / 32 + 1], w[b1 / 32], b1 % 32) >> 16;
        levels[g] = levels_x2(fmix_hash_pair(s0, s1));
    }
}

// The row's output from its exact sums. Written with explicit fmaf so that a host mirror can reproduce it bit for bit.
__device__ __forceinline__ float finish(int coarse, int fine, const Stats& st, float scale) {
    constexpr float c = 1.0f / kRms, d0 = -kMean0 / kRms, d1 = -kMean1 / kRms;
    coarse -= 54 * st.sum_q0;
    fine -= 54 * st.sum_q1;
    const float dot = fmaf(static_cast<float>(fine), 1.0f / 254.0f, static_cast<float>(coarse));
    return scale * fmaf(d1, st.odd, fmaf(d0, st.even, c * st.step * dot));
}

// Unnormalized rotation of one vector of COLS = 32 WARPS R Hadamard indices x ODD spread over WARPS warps (the whole CTA
// when WARPS > 1): thread t of them holds indices t R .. t R + R - 1 with their ODD odd-factor components, i.e. the
// contiguous columns ODD R t .. ODD R (t + 1) - 1. v <- (H (x) Q) v, or with TRANSPOSE (H (x) Q^T) v; H the +-1
// Sylvester Hadamard (the caller applies kNorm). Strides below R run inside the thread, R .. 16 R across lanes, the
// larger ones through `exchange` (shared, COLS floats; null with one warp), which the caller may have used before.
template <int COLS, int WARPS, bool TRANSPOSE>
__device__ __forceinline__ void rotate(float (&v)[COLS / 32 / WARPS], float* __restrict__ exchange) {
    constexpr int ODD = kOdd<COLS>, N = COLS / 32 / WARPS, R = N / ODD, THREADS = 32 * WARPS;
    static_assert(R * ODD == N, "whole Hadamard indices a thread");
    const int lane = threadIdx.x & 31;
#pragma unroll
    for (int r = 0; r < R; ++r) {
        float in[ODD];
#pragma unroll
        for (int c = 0; c < ODD; ++c) in[c] = v[ODD * r + c];
#pragma unroll
        for (int o = 0; o < ODD; ++o) {
            float acc = 0.0f;
#pragma unroll
            for (int c = 0; c < ODD; ++c) acc += q_entry<COLS>(TRANSPOSE ? ODD * c + o : ODD * o + c) * in[c];
            v[ODD * r + o] = acc;
        }
    }
#pragma unroll
    for (int stride = 1; stride < R; stride <<= 1)
#pragma unroll
        for (int r = 0; r < R; ++r)
            if ((r & stride) == 0)
#pragma unroll
                for (int o = 0; o < ODD; ++o) {
                    const float a = v[ODD * r + o], b = v[ODD * (r + stride) + o];
                    v[ODD * r + o] = a + b;
                    v[ODD * (r + stride) + o] = a - b;
                }
#pragma unroll
    for (int m = 1; m < 32; m <<= 1) {
        const float sign = (lane & m) ? -1.0f : 1.0f;  // a + b in the lower lane, a - b in the upper one
#pragma unroll
        for (int j = 0; j < N; ++j) v[j] = fmaf(v[j], sign, __shfl_xor_sync(0xffffffffu, v[j], m));
    }
    if constexpr (WARPS > 1) {
        const int warp = threadIdx.x >> 5;
        __syncthreads();  // the caller is done with the buffer
#pragma unroll
        for (int j = 0; j < N; ++j) exchange[j * THREADS + threadIdx.x] = v[j];
        __syncthreads();
#pragma unroll
        for (int j = 0; j < N; ++j) {
            float acc = 0.0f;
#pragma unroll
            for (int w = 0; w < WARPS; ++w) {
                const float other = exchange[j * THREADS + 32 * w + lane];
                acc += (__popc(warp & w) & 1) ? -other : other;
            }
            v[j] = acc;
        }
    }
}

// Quantizes x_rot of one vector spread over the CTA (thread t: columns N t .. N t + N - 1) into act:
// step = max |x_rot| / 127 (1 if 0), q0 = clamp(rn(x / step)), q1 = clamp(rn((x / step - q0) 254)), x / step taken as
// x * (1 / step) like the plugin. The two planes are gathered into whole words of 4 columns in `buffer` (shared, COLS
// floats; written only after every thread is past its first barrier here).
template <int COLS, int N>
__device__ __forceinline__ void quantize(const float (&v)[N], Act<COLS>* __restrict__ act, float* __restrict__ buffer) {
    constexpr int THREADS = COLS / N, WARPS = THREADS / 32;
    __shared__ float maxima[WARPS];
    __shared__ int partial[WARPS][4];
    signed char* planes[2] = {reinterpret_cast<signed char*>(buffer), reinterpret_cast<signed char*>(buffer) + COLS};
    const int t = threadIdx.x, lane = t & 31, warp = t >> 5;
    float maximum = 0.0f;
#pragma unroll
    for (int j = 0; j < N; ++j) maximum = fmaxf(maximum, fabsf(v[j]));
#pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) maximum = fmaxf(maximum, __shfl_xor_sync(0xffffffffu, maximum, offset));
    if (lane == 0) maxima[warp] = maximum;
    __syncthreads();
#pragma unroll
    for (int w = 0; w < WARPS; ++w) maximum = fmaxf(maximum, maxima[w]);
    const float step = maximum > 0.0f ? maximum / 127.0f : 1.0f, inverse = 1.0f / step;
    int sums[4] = {};  // q0, q1 over all columns, q0, q1 over the odd ones
#pragma unroll
    for (int j = 0; j < N; ++j) {
        const int column = N * t + j, odd = column & 1;
        const float scaled = v[j] * inverse;
        const int q0 = min(127, max(-127, __float2int_rn(scaled)));
        const int q1 = min(127, max(-127, __float2int_rn((scaled - q0) * 254.0f)));
        planes[0][column] = static_cast<signed char>(q0);
        planes[1][column] = static_cast<signed char>(q1);
        sums[0] += q0;
        sums[1] += q1;
        sums[2] += odd * q0;
        sums[3] += odd * q1;
    }
#pragma unroll
    for (int i = 0; i < 4; ++i)
#pragma unroll
        for (int offset = 16; offset > 0; offset >>= 1) sums[i] += __shfl_xor_sync(0xffffffffu, sums[i], offset);
    if (lane == 0)
#pragma unroll
        for (int i = 0; i < 4; ++i) partial[warp][i] = sums[i];
    __syncthreads();
    const u32* words0 = reinterpret_cast<const u32*>(planes[0]);
    const u32* words1 = reinterpret_cast<const u32*>(planes[1]);  // COLS is a multiple of 16: both stay word-aligned
    for (int g = t; g < COLS / 4; g += THREADS) act->q[g] = make_uint2(words0[g], words1[g]);
    if (t != 0) return;
#pragma unroll
    for (int w = 1; w < WARPS; ++w)
#pragma unroll
        for (int i = 0; i < 4; ++i) sums[i] += partial[w][i];
    act->st.step = step;
    act->st.sum_q0 = sums[0];
    act->st.sum_q1 = sums[1];
    act->st.even = step * fmaf(static_cast<float>(sums[1] - sums[3]), 1.0f / 254.0f, static_cast<float>(sums[0] - sums[2]));
    act->st.odd = step * fmaf(static_cast<float>(sums[3]), 1.0f / 254.0f, static_cast<float>(sums[2]));
}

// quant_act: grid (n_tok), block 32 WARPS: CTA = token of x [n_tok][COLS] -> act[token]. The checkpoint-ordered,
// signed input signs[c] x[perm[c]] (x[c] without perm) is staged in shared memory with whole-warp loads, then thread t
// takes columns N t .. N t + N - 1.
template <int COLS, int WARPS>
__global__ void __launch_bounds__(32 * WARPS) quant_act(const float* __restrict__ x, const int32_t* __restrict__ perm,
                                                        Act<COLS>* __restrict__ act) {
    constexpr int N = COLS / 32 / WARPS, THREADS = 32 * WARPS;
    __shared__ __align__(16) float buffer[COLS];
    const float* row = x + static_cast<int64_t>(blockIdx.x) * COLS;
    for (int c = threadIdx.x; c < COLS; c += THREADS) buffer[c] = row[perm ? perm[c] : c] * signs_of<COLS>()[c];
    __syncthreads();
    float v[N];
#pragma unroll
    for (int k = 0; k < N / 4; ++k) {
        const float4 a = reinterpret_cast<const float4*>(buffer + N * threadIdx.x)[k];
        v[4 * k] = a.x;
        v[4 * k + 1] = a.y;
        v[4 * k + 2] = a.z;
        v[4 * k + 3] = a.w;
    }
    rotate<COLS, WARPS, false>(v, buffer);
#pragma unroll
    for (int j = 0; j < N; ++j) v[j] *= kNorm<COLS>;
    quantize<COLS, N>(v, act + blockIdx.x, buffer);
}

// act_640: grid (cap_entries), block 128: CTA = entry e of the call ([grp_start[0], grp_start[*n_groups])):
// h = silu(gate) * up of row e of gu [.][1280] (fp32), rotated and quantized into hq[e] (thread t: columns 5 t .. 5 t + 4).
__global__ void __launch_bounds__(128) act_640(const int32_t* __restrict__ grp_start, const int32_t* __restrict__ n_groups,
                                               const float* __restrict__ gu, Act<640>* __restrict__ hq) {
    __shared__ float buffer[640];
    const int e = blockIdx.x;
    if (e < grp_start[0] || e >= grp_start[*n_groups]) return;
    const float* gate = gu + static_cast<size_t>(e) * 1280 + 5 * threadIdx.x;
    float v[5];
#pragma unroll
    for (int j = 0; j < 5; ++j) v[j] = __fdividef(gate[j], 1.0f + __expf(-gate[j])) * gate[640 + j] * g_signs640[5 * threadIdx.x + j];
    rotate<640, 4, false>(v, buffer);
#pragma unroll
    for (int j = 0; j < 5; ++j) v[j] *= kNorm<640>;
    quantize<640, 5>(v, hq + e, buffer);
}

// Block x of a matrix (32 rows, lane = row) against the entries [e0, e1) of its group, 4 at a time: the warps stride
// over the 64-column chunks, each decoded once per pass into the warp's tile and multiplied on the tensor cores with the
// pass's entries (as dense_rows below, one n tile: B column n = entry n / 2, plane n % 2).
// Gate/up (COLS 2560): entry e reads token ent_map[e]'s Act<2560>, writes row e of out [.][1280].
// Down (COLS 640): entry e reads Act<640> e, writes row ent_map[e] of out [.][2560].
template <int T, int COLS, int WARPS>
__device__ __forceinline__ void gemv_block(const u32* __restrict__ words, float rowscale, int e0, int e1,
                                           const int32_t* __restrict__ ent_map, const Act<COLS>* __restrict__ acts,
                                           float* __restrict__ out, u32 (&tile)[WARPS][32][20]) {
    constexpr bool DOWN = COLS == 640;
    constexpr int ROWS = DOWN ? 2560 : 1280;
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5, gid = lane >> 2, tig = lane & 3, row = blockIdx.x * 32 + lane;
    const u32* lane_words = words + blockIdx.x * (COLS / 64 * T + 1) * 32 + lane;
    for (int first = e0; first < e1; first += 4) {
        const int n = min(4, e1 - first), mine = first + min(gid / 2, n - 1);  // an absent entry reads the last one
        const u32* xq = reinterpret_cast<const u32*>(acts[DOWN ? mine : ent_map[mine]].q) + gid % 2;
        int acc[2][4] = {};  // [row tile][C fragment]
        for (int chunk = warp; chunk < COLS / 64; chunk += WARPS) {
            u32 w[T + 1];
#pragma unroll
            for (int i = 0; i <= T; ++i) w[i] = __ldg(lane_words + (chunk * T + i) * 32);
            u32 levels[16];
            decode_chunk<T>(w, levels);
            mma_chunk(tile[warp], levels, xq, chunk, acc);
        }
        __syncthreads();  // every warp is done with its tile
        int(*partial)[4][WARPS][32] = reinterpret_cast<int(*)[4][WARPS][32]>(&tile[0][0][0]);  // [plane][entry][warp][row]
#pragma unroll
        for (int rt = 0; rt < 2; ++rt)
#pragma unroll
            for (int half = 0; half < 2; ++half) {  // C: rows gid, gid + 8; columns 2 tig (plane 0), 2 tig + 1 (plane 1)
                partial[0][tig][warp][16 * rt + gid + 8 * half] = acc[rt][2 * half];
                partial[1][tig][warp][16 * rt + gid + 8 * half] = acc[rt][2 * half + 1];
            }
        __syncthreads();
        if (warp == 0)
            for (int b = 0; b < n; ++b) {
                int c = 0, f = 0;
#pragma unroll
                for (int w = 0; w < WARPS; ++w) {
                    c += partial[0][b][w][lane];
                    f += partial[1][b][w][lane];
                }
                const int e = first + b;
                const size_t dst = DOWN ? static_cast<size_t>(ent_map[e]) * ROWS : static_cast<size_t>(e) * ROWS;
                out[dst + row] = finish(c >> 1, f >> 1, acts[DOWN ? e : ent_map[e]].st, rowscale);  // doubled levels, halved exactly
            }
        __syncthreads();  // the tile memory is reused by the next pass
    }
}

#define QNS_SWITCH_T(t, ...)                                    \
    switch (t) {                                                \
        case 3: { constexpr int T = 3; __VA_ARGS__; break; }   \
        case 4: { constexpr int T = 4; __VA_ARGS__; break; }   \
        case 5: { constexpr int T = 5; __VA_ARGS__; break; }   \
        case 6: { constexpr int T = 6; __VA_ARGS__; break; }   \
        case 7: { constexpr int T = 7; __VA_ARGS__; break; }   \
        case 8: { constexpr int T = 8; __VA_ARGS__; break; }   \
        case 9: { constexpr int T = 9; __VA_ARGS__; break; }   \
        case 10: { constexpr int T = 10; __VA_ARGS__; break; } \
        default: __trap();                                      \
    }

// expert_gemv: grid (rows / 32, grid groups), block 32 WARPS: block (x, y) takes 32-row block x of the gate/up
// (COLS 2560) or the down (COLS 640) matrix of groups y, y + gridDim.y, ... < *n_groups (as native_gu_kernel).
// Header words: 1 T_gu, 2 T_d, 5 / 6 the gate/up / down words' byte offset, 7 / 8 their scales' byte offset.
template <int COLS, int WARPS>
__global__ void __launch_bounds__(32 * WARPS, COLS == 2560 ? 3 : 4)  // the registers to decode without spills
    expert_gemv(const unsigned long long* __restrict__ grp_ptr, const int32_t* __restrict__ grp_start,
                const int32_t* __restrict__ n_groups, const int32_t* __restrict__ ent_map,
                const Act<COLS>* __restrict__ acts, float* __restrict__ out) {
    constexpr int M = COLS == 640;  // 0 gate/up, 1 down
    __shared__ __align__(16) u32 tile[WARPS][32][20];
    const int ng = *n_groups;
    for (int g = blockIdx.y; g < ng; g += gridDim.y) {
        const u32* blob = reinterpret_cast<const u32*>(grp_ptr[g]);
        if (blob[0] != kMagic) __trap();
        const u32* words = blob + blob[5 + M] / 4;
        const float rowscale = reinterpret_cast<const float*>(blob + blob[7 + M] / 4)[blockIdx.x * 32 + (threadIdx.x & 31)];
        QNS_SWITCH_T(blob[1 + M], gemv_block<T, COLS, WARPS>(words, rowscale, grp_start[g], grp_start[g + 1], ent_map,
                                                              acts, out, tile))
    }
}

// dense_gemv: grid (rows / 32), block 32 WARPS: block x takes rows 32 x .. 32 x + 31 of a dense matrix [rows][COLS] for
// n_tok <= 8 tokens; the warps stride over the 64-column chunks, each decoded once (lane = row) and multiplied with 4
// tokens a mma_chunk (B column n of n tile nt: token 4 nt + n / 2, plane n % 2): 1 to 4 tokens cost the same, 5 to 8
// twice that. After the chunks the tile memory holds the warps' sums for the cross-warp reduction. y[t rows + r] = row r . x_t.
template <int T, int COLS, int WARPS>
__device__ __forceinline__ void dense_rows(const u32* __restrict__ words, const float* __restrict__ scales, int rows,
                                           const Act<COLS>* __restrict__ acts, int n_tok, float* __restrict__ y,
                                           u32 (&tile)[WARPS][32][20]) {
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5, gid = lane >> 2, tig = lane & 3;
    const u32* lane_words = words + blockIdx.x * (COLS / 64 * T + 1) * 32 + lane;
    const int ntiles = (n_tok + 3) / 4;
    const u32* xq[2];  // this lane's B source: a token's plane words (an absent token reads the last one)
#pragma unroll
    for (int nt = 0; nt < 2; ++nt) xq[nt] = reinterpret_cast<const u32*>(acts[min(4 * nt + gid / 2, n_tok - 1)].q) + gid % 2;
    int acc[2][2][4] = {};  // [n tile][row tile: rows 0-15, 16-31][C fragment]
    for (int chunk = warp; chunk < COLS / 64; chunk += WARPS) {
        u32 w[T + 1];
#pragma unroll
        for (int i = 0; i <= T; ++i) w[i] = __ldg(lane_words + (chunk * T + i) * 32);
        u32 levels[16];
        decode_chunk<T>(w, levels);
#pragma unroll
        for (int nt = 0; nt < 2; ++nt)
            if (nt < ntiles) mma_chunk(tile[warp], levels, xq[nt], chunk, acc[nt]);  // uniform
    }
    __syncthreads();  // every warp is done with its tile
    int(*partial)[8][WARPS][32] = reinterpret_cast<int(*)[8][WARPS][32]>(&tile[0][0][0]);  // [plane][token][warp][row]
#pragma unroll
    for (int nt = 0; nt < 2; ++nt) {
        if (nt >= ntiles) break;
#pragma unroll
        for (int rt = 0; rt < 2; ++rt)
#pragma unroll
            for (int half = 0; half < 2; ++half) {  // C: rows gid, gid + 8; columns 2 tig (plane 0), 2 tig + 1 (plane 1)
                partial[0][4 * nt + tig][warp][16 * rt + gid + 8 * half] = acc[nt][rt][2 * half];
                partial[1][4 * nt + tig][warp][16 * rt + gid + 8 * half] = acc[nt][rt][2 * half + 1];
            }
    }
    __syncthreads();
    if (warp != 0) return;
    const int row = blockIdx.x * 32 + lane;
    for (int b = 0; b < n_tok; ++b) {
        int c = 0, f = 0;
#pragma unroll
        for (int w = 0; w < WARPS; ++w) {
            c += partial[0][b][w][lane];
            f += partial[1][b][w][lane];
        }
        y[static_cast<size_t>(b) * rows + row] = finish(c >> 1, f >> 1, acts[b].st, scales[row]);  // doubled levels, halved exactly
    }
}

template <int COLS, int WARPS>
__global__ void __launch_bounds__(32 * WARPS) dense_gemv(const u32* __restrict__ words, const float* __restrict__ scales,
                                                         int rows, int T, const Act<COLS>* __restrict__ acts, int n_tok,
                                                         float* __restrict__ y) {
    static_assert(sizeof(int[2][8][WARPS][32]) <= sizeof(u32[WARPS][32][20]), "the sums fit in the tiles");
    __shared__ __align__(16) u32 tile[WARPS][32][20];
    QNS_SWITCH_T(T, dense_rows<T, COLS, WARPS>(words, scales, rows, acts, n_tok, y, tile))
}

// Thread t of a row's CTA decodes states S t .. S t + S - 1 (S = COLS / 64 / WARPS), i.e. the rotated columns rotate()
// expects in it, from its own window of the row's words (clamped to the row), realigned so that state i sits at bit
// i T. The fp16 row is gathered in shared memory (at its engine column, perm[c], when there is a permutation), so that
// every warp store is one contiguous 128 bytes.
template <int T, int COLS, int WARPS>
__device__ __forceinline__ void dequant_row(const u32* __restrict__ words, int row, float scale, const int32_t* __restrict__ perm,
                                            __half* __restrict__ dst, __half (&stage)[COLS], float* __restrict__ exchange) {
    constexpr int THREADS = 32 * WARPS, S = COLS / 64 / WARPS, W = COLS / 64 * T + 1;
    constexpr int A = ((S - 1) * T + 16) / 32 + 2;  // realigned words: state i needs i T / 32 + 1
    const int t = threadIdx.x, bit = t * S * T;
    const u32* row_words = words + (row / 32) * W * 32 + row % 32;
    u32 raw[A + 1], a[A];
#pragma unroll
    for (int i = 0; i <= A; ++i) raw[i] = __ldg(row_words + 32 * min(bit / 32 + i, W - 1));  // past the row: unused bits
#pragma unroll
    for (int i = 0; i < A; ++i) a[i] = __funnelshift_l(raw[i + 1], raw[i], bit % 32);
    const float c = scale / kRms * kNorm<COLS>;  // the rotation's normalization folded in
    float v[2 * S];
#pragma unroll
    for (int i = 0; i < S; i += 2) {  // states i, i + 1 (an odd S decodes its last state twice)
        const int b0 = i * T, b1 = min(i + 1, S - 1) * T;
        const u32 s0 = __funnelshift_l(a[b0 / 32 + 1], a[b0 / 32], b0 % 32) >> 16;
        const u32 s1 = __funnelshift_l(a[b1 / 32 + 1], a[b1 / 32], b1 % 32) >> 16;
        const u32 levels = levels_x2(fmix_hash_pair(s0, s1));
#pragma unroll
        for (int k = 0; k < 4; ++k)
            if (2 * i + k < 2 * S)
                v[2 * i + k] = (static_cast<float>(static_cast<int>((levels >> (8 * k + 1)) & 127u) - 54) - (k & 1 ? kMean1 : kMean0)) * c;
    }
    rotate<COLS, WARPS, true>(v, exchange);
#pragma unroll
    for (int j = 0; j < 2 * S; ++j) {
        const int column = 2 * S * t + j;
        stage[perm ? perm[column] : column] = __float2half_rn(v[j] * signs_of<COLS>()[column]);
    }
    __syncthreads();
    for (int k = t; k < COLS / 2; k += THREADS) reinterpret_cast<u32*>(dst)[k] = reinterpret_cast<const u32*>(stage)[k];
}

// expert_dequant: grid (rows), block 32 WARPS: CTA = one row of a blob's gate/up (COLS 2560, interleaved output rows) or
// down (COLS 640) matrix in the original basis, w = (w_rot @ R) * signs.
template <int COLS, int WARPS>
__global__ void __launch_bounds__(32 * WARPS) expert_dequant(const u32* __restrict__ blob, __half* __restrict__ dst) {
    constexpr int M = COLS == 640;
    __shared__ __align__(16) __half stage[COLS];
    __shared__ float exchange[COLS];
    if (blob[0] != kMagic) __trap();
    const u32* words = blob + blob[5 + M] / 4;
    const float* scales = reinterpret_cast<const float*>(blob + blob[7 + M] / 4);
    const int row = blockIdx.x, out_row = M ? row : (row < 640 ? 2 * row : 2 * (row - 640) + 1);
    QNS_SWITCH_T(blob[1 + M], dequant_row<T, COLS, WARPS>(words, row, scales[row], nullptr, dst + static_cast<size_t>(out_row) * COLS,
                                                          stage, exchange))
}

// dense_dequant: grid (rows), block 32 WARPS: CTA = row r of a dense matrix -> dst[r][perm[c]] (dst[r][c] without perm).
template <int COLS, int WARPS>
__global__ void __launch_bounds__(32 * WARPS) dense_dequant(const u32* __restrict__ words, const float* __restrict__ scales, int T,
                                                            const int32_t* __restrict__ perm, __half* __restrict__ dst) {
    __shared__ __align__(16) __half stage[COLS];
    __shared__ float exchange[COLS];
    const int row = blockIdx.x;
    QNS_SWITCH_T(T, dequant_row<T, COLS, WARPS>(words, row, scales[row], perm, dst + static_cast<size_t>(row) * COLS, stage, exchange))
}

}  // namespace

void qns_cuda_set_rotation(const float* signs2560, const float* q2560, const float* signs640, const float* q640) {
    cudaMemcpyToSymbol(c_q5, q2560, 25 * sizeof(float));
    cudaMemcpyToSymbol(c_q5, q640, 25 * sizeof(float), 25 * sizeof(float));
    cudaMemcpyToSymbol(g_signs2560, signs2560, sizeof(g_signs2560));
    cudaMemcpyToSymbol(g_signs640, signs640, sizeof(g_signs640));
    check("qns_cuda_set_rotation");
}

void qns_cuda_set_rotation_6144(const float* signs6144, const float* q6144) {
    cudaMemcpyToSymbol(c_q3, q6144, 9 * sizeof(float));
    cudaMemcpyToSymbol(g_signs6144, signs6144, sizeof(g_signs6144));
    check("qns_cuda_set_rotation_6144");
}

size_t qns_act_bytes() { return sizeof(Act<2560>); }

void qns_quant_act(const float* x, int64_t n_tok, void* act, cudaStream_t s) {
    if (n_tok <= 0) return;
    quant_act<2560, 4><<<static_cast<unsigned>(n_tok), 128, 0, s>>>(x, nullptr, static_cast<Act<2560>*>(act));
    check("qns_quant_act");
}

size_t qns_grouped_scratch_bytes(int64_t cap_entries) {
    return static_cast<size_t>(cap_entries) * (1280 * sizeof(float) + sizeof(Act<640>));
}

void qns_expert_grouped(const unsigned long long* grp_ptr, const int32_t* grp_start, const int32_t* n_groups,
                        const int32_t* ent_dst, const int32_t* ent_tok, int64_t cap_groups, int64_t cap_entries,
                        const void* act, void* scratch, float* out, cudaStream_t s, int64_t grid_groups) {
    if (cap_groups <= 0 || cap_entries <= 0) return;
    const unsigned gy = static_cast<unsigned>(grid_groups <= 0 || grid_groups > cap_groups ? cap_groups : grid_groups);
    float* gu = static_cast<float*>(scratch);  // [cap_entries][1280], then the down activations
    Act<640>* hq = reinterpret_cast<Act<640>*>(gu + cap_entries * 1280);
    expert_gemv<2560, 8><<<dim3(40, gy), 256, 0, s>>>(grp_ptr, grp_start, n_groups, ent_tok,
                                                      static_cast<const Act<2560>*>(act), gu);
    act_640<<<static_cast<unsigned>(cap_entries), 128, 0, s>>>(grp_start, n_groups, gu, hq);
    expert_gemv<640, 5><<<dim3(80, gy), 160, 0, s>>>(grp_ptr, grp_start, n_groups, ent_dst, hq, out);
    check("qns_expert_grouped");
}

void qns_dequant_gu_f16(const uint8_t* blob, uint16_t* dst, cudaStream_t s) {
    expert_dequant<2560, 4><<<1280, 128, 0, s>>>(reinterpret_cast<const u32*>(blob), reinterpret_cast<__half*>(dst));
    check("qns_dequant_gu_f16");
}

void qns_dequant_d_f16(const uint8_t* blob, uint16_t* dst, cudaStream_t s) {
    expert_dequant<640, 2><<<2560, 64, 0, s>>>(reinterpret_cast<const u32*>(blob), reinterpret_cast<__half*>(dst));
    check("qns_dequant_d_f16");
}

static void dense_shape(const char* what, int rows, int cols, int T) {
    if ((cols == 2560 || cols == 6144) && rows > 0 && rows % 32 == 0 && T >= 3 && T <= 10) return;
    std::fprintf(stderr, "%s: rows %d, cols %d, T %d\n", what, rows, cols, T);
    std::abort();
}

size_t qns_dense_act_bytes(int cols) {
    dense_shape("qns_dense_act_bytes", 32, cols, 10);
    return cols == 2560 ? sizeof(Act<2560>) : sizeof(Act<6144>);
}

void qns_dense_quant(const float* x, int64_t n_tok, int cols, const int32_t* perm, void* act, cudaStream_t s) {
    dense_shape("qns_dense_quant", 32, cols, 10);
    if (n_tok <= 0) return;
    if (cols == 2560) quant_act<2560, 4><<<static_cast<unsigned>(n_tok), 128, 0, s>>>(x, perm, static_cast<Act<2560>*>(act));
    else quant_act<6144, 8><<<static_cast<unsigned>(n_tok), 256, 0, s>>>(x, perm, static_cast<Act<6144>*>(act));
    check("qns_dense_quant");
}

void qns_dense_gemv(const uint32_t* words, const float* scales, int rows, int cols, int T, const void* act, int n_tok, float* y,
                    cudaStream_t s) {
    dense_shape("qns_dense_gemv", rows, cols, T);
    if (n_tok < 1 || n_tok > 8) { std::fprintf(stderr, "qns_dense_gemv: n_tok %d (1..8)\n", n_tok); std::abort(); }
    const bool tall = rows > 2560;  // enough row blocks to fill the GPU with 8 warps a block; else 16 warps split the columns
    if (cols == 2560 && tall) dense_gemv<2560, 8><<<rows / 32, 256, 0, s>>>(words, scales, rows, T, static_cast<const Act<2560>*>(act), n_tok, y);
    else if (cols == 2560) dense_gemv<2560, 16><<<rows / 32, 512, 0, s>>>(words, scales, rows, T, static_cast<const Act<2560>*>(act), n_tok, y);
    else if (tall) dense_gemv<6144, 8><<<rows / 32, 256, 0, s>>>(words, scales, rows, T, static_cast<const Act<6144>*>(act), n_tok, y);
    else dense_gemv<6144, 16><<<rows / 32, 512, 0, s>>>(words, scales, rows, T, static_cast<const Act<6144>*>(act), n_tok, y);
    check("qns_dense_gemv");
}

void qns_dense_dequant_f16(const uint32_t* words, const float* scales, int rows, int cols, int T, const int32_t* perm,
                           uint16_t* dst, cudaStream_t s) {
    dense_shape("qns_dense_dequant_f16", rows, cols, T);
    if (cols == 2560) dense_dequant<2560, 4><<<rows, 128, 0, s>>>(words, scales, T, perm, reinterpret_cast<__half*>(dst));
    else dense_dequant<6144, 8><<<rows, 256, 0, s>>>(words, scales, T, perm, reinterpret_cast<__half*>(dst));
    check("qns_dense_dequant_f16");
}

}  // namespace strata::kernels
