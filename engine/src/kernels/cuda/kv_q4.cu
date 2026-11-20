// src/kernels/cuda/kv_q4.cu - see include/strata/kernels/kv_q4.hpp. Q4_0 KV with Walsh-Hadamard rotation
// (from PR #21 by code-martin; KV-streaming integration and the deterministic group maximum added on merge).
#include "strata/kernels/kv_q4.hpp"
#include "strata/kernels/f16_bits.hpp"
#include "strata/kernels/kv_stream.hpp"
#include "strata/kernels/qsa_decode_attn.hpp"   // QsaAttnPools + qsa_win_slot: one definition of the window's row

#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>

namespace strata::kernels {
namespace {

void check(const char* what) {
    const cudaError_t e = cudaGetLastError();
    if (e != cudaSuccess) {
        std::fprintf(stderr, "kv_q4: %s: %s\n", what, cudaGetErrorString(e));
        std::exit(1);
    }
}

// Fast Walsh-Hadamard Transform for N = 256: one warp per row, 8 values per lane in registers.
// Orthonormal (scale 1/sqrt(256) = 1/16), so it is its own inverse.
__global__ void fwht256_kernel(const float* __restrict__ src, float* __restrict__ dst, int64_t n_rows, float scale) {
    constexpr int warp_size = 32;
    constexpr int N = 256;
    constexpr int el_w = N / warp_size;   // 8

    const int64_t r = (int64_t) blockIdx.x * blockDim.y + threadIdx.y;
    if (r >= n_rows) return;

    const float* row_src = src + r * N;
    float* row_dst = dst + r * N;

    float reg[el_w];
    const int lane = threadIdx.x;

#pragma unroll
    for (int i = 0; i < el_w; ++i) reg[i] = row_src[i * warp_size + lane] * scale;

    // the low 5 index bits live across lanes
#pragma unroll
    for (int h = 1; h < warp_size; h *= 2) {
#pragma unroll
        for (int j = 0; j < el_w; ++j) {
            const float val = reg[j];
            const float val2 = __shfl_xor_sync(0xffffffffu, val, h, warp_size);
            reg[j] = (lane & h) == 0 ? val + val2 : val2 - val;
        }
    }
    // the high 3 bits across each lane's registers
#pragma unroll
    for (int h = warp_size; h < N; h *= 2) {
        const int step = h / warp_size;
#pragma unroll
        for (int j = 0; j < el_w; j += 2 * step) {
#pragma unroll
            for (int k = 0; k < step; ++k) {
                const float x = reg[j + k];
                const float y = reg[j + k + step];
                reg[j + k] = x + y;
                reg[j + k + step] = x - y;
            }
        }
    }
#pragma unroll
    for (int i = 0; i < el_w; ++i) row_dst[i * warp_size + lane] = reg[i];
}

// One 32-value group in one warp (lane = element), ggml's q4_0: d = (the value of largest |x|) / -8,
// q = clamp(trunc(x / d + 8.5), 0, 15). Returns the scale bits; `byte` is lane t's packed byte for t < 16
// (element t in the low nibble, t + 16 in the high one). Ties in |x| resolve to the larger value in EVERY lane,
// so all lanes agree on d (a plain `a > amax` could leave lanes with opposite signs and one block two scales).
__device__ __forceinline__ uint16_t q4_group(float x, int lane, uint8_t& byte) {
    float amax = fabsf(x), mval = x;
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) {
        const float a = __shfl_xor_sync(0xffffffffu, amax, o);
        const float v = __shfl_xor_sync(0xffffffffu, mval, o);
        if (a > amax || (a == amax && v > mval)) { amax = a; mval = v; }
    }
    const float d = mval / -8.0f;
    const float id = d != 0.0f ? 1.0f / d : 0.0f;
    int q = __float2int_rz(x * id + 8.5f);
    const uint8_t qc = (uint8_t) (q < 0 ? 0 : (q > 15 ? 15 : q));
    const uint8_t qhi = __shfl_down_sync(0xffffffffu, qc, 16);
    byte = (uint8_t) (qc | (qhi << 4));
    (void) lane;
    return f16_from_f32(d);
}

// The nearest of the 16 levels for a value already divided by the scale. The 15 boundaries are the midpoints of
// consecutive levels; written as literals so the ladder unrolls into predicated increments (a table read would be
// 15 uniform-index loads for no gain).
__device__ __forceinline__ int q4l_level(float y) {
    int k = 0;
    if (y >= -76.5f) k = 1;
    if (y >= -59.0f) k = 2;
    if (y >= -46.0f) k = 3;
    if (y >= -35.0f) k = 4;
    if (y >= -25.5f) k = 5;
    if (y >= -16.5f) k = 6;
    if (y >= -8.0f) k = 7;
    if (y >= 0.0f) k = 8;
    if (y >= 8.0f) k = 9;
    if (y >= 16.5f) k = 10;
    if (y >= 25.5f) k = 11;
    if (y >= 35.0f) k = 12;
    if (y >= 46.0f) k = 13;
    if (y >= 59.0f) k = 14;
    if (y >= 76.5f) k = 15;
    return k;
}

// The 8-level ladder (kv_q4.hpp's kQ4l8Levels): the midpoints of -69, -43, -24, -8, 8, 24, 43, 69.
__device__ __forceinline__ int q4l_level8(float y) {
    int k = 0;
    if (y >= -56.0f) k = 1;
    if (y >= -33.5f) k = 2;
    if (y >= -16.0f) k = 3;
    if (y >= 0.0f) k = 4;
    if (y >= 16.0f) k = 5;
    if (y >= 33.5f) k = 6;
    if (y >= 56.0f) k = 7;
    return k;
}

// The same group for `--kv q4lm32` / `--kv q4lm32s` (kv_q4.hpp's level table): the nibble is the nearest level and
// the scale is max|x| / 87, or - with FIT - the block's own least-squares scale. max|x| is a plain shuffle maximum
// (unlike q4_0's d there is no sign or tie to agree on, and a maximum is the same in every lane whatever the
// pairing), and the two fit sums are butterflies whose pairing is identical in every lane, so with fp addition
// commuting every lane still reaches the same value. `byte` packs as in q4_group.
// N is 16 (the q4lm32 table) or 8 (the WP-2 probe table); FIT is only ever used with N = 16 today.
template <bool FIT, int N>
__device__ __forceinline__ uint16_t q4l_group(float x, uint8_t& byte) {
    constexpr int8_t lmax = (N == 8) ? kQ4l8Max : kQ4lMax;
    const int8_t* lmap = nullptr;
    if constexpr (N == 8) lmap = kQ4l8Map; else lmap = kQ4lMap;
    float amax = fabsf(x);
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) amax = fmaxf(amax, __shfl_xor_sync(0xffffffffu, amax, o));
    float d = amax * (1.0f / (float) lmax);
    const float y = d != 0.0f ? x / d : 0.0f;
    int k = (N == 8) ? q4l_level8(y) : q4l_level(y);
    if constexpr (FIT) {
        // One least-squares pass over this block: with the level assignment held, d = sum(l * x) / sum(l * l)
        // minimizes the block's error against these very levels, and the assignment is then redone with that d. One
        // pass is most of what the levels can be given (kv_q4.hpp). An all-zero block keeps d = 0 and k = 8, whose
        // level is 0, so the divide guards are never taken.
        const float l = (float) lmap[(uint8_t) k];
        float num = l * x, den = l * l;
#pragma unroll
        for (int o = 16; o > 0; o >>= 1) {
            num += __shfl_xor_sync(0xffffffffu, num, o);
            den += __shfl_xor_sync(0xffffffffu, den, o);
        }
        if (den > 0.0f) {
            d = num / den;
            k = (N == 8) ? q4l_level8(d != 0.0f ? x / d : 0.0f) : q4l_level(d != 0.0f ? x / d : 0.0f);
        }
    }
    const uint8_t kc = (uint8_t) k;
    const uint8_t khi = __shfl_down_sync(0xffffffffu, kc, 16);
    byte = (uint8_t) (kc | (khi << 4));
    return f16_from_f32(d);
}

// The shear: a UNIFORM grid with a per-block offset (the "zero point"), see
// bench/results/2026-11-11-shear-codec/.  The grid is [o, o + 15*s] in the code's own units (code 0
// is its left end) and the reconstruction is v = s*(code - 8) + c with c = o + 8*s, so the reader
// adds ONE constant per block -- the offset corrects the grid's position against the block's own
// distribution, not a vector.
//
// The fit is kShearPasses passes of "assign the codes, then solve the centred least squares for
// (o, s)".  It starts from the grid that COVERS the block (o = min, s = (max-min)/15): a grid of
// +/-max leaves half the range empty, and the least-squares correction cannot find its way back
// from that start -- it costs 1.3 dB of block error (measured, see the README).
//
// The reductions are butterflies over the 32 lanes of the block: sum(code), sum(code^2) and
// sum(code*x); sum(x) does not change over the passes and is taken once, so a pass costs three
// butterflies (plus the assignment) and the first pass four.
template <int PASSES>
__device__ __forceinline__ uint16_t q4u_group(float x, uint8_t& byte, float& c_out) {
    float vmax = x, vmin = x;
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) {
        vmax = fmaxf(vmax, __shfl_xor_sync(0xffffffffu, vmax, o));
        vmin = fminf(vmin, __shfl_xor_sync(0xffffffffu, vmin, o));
    }
    float s = (vmax - vmin) / 15.0f;
    if (!(s > 0.0f)) s = 1.0f;          // an all-zero block keeps s = 1 and the code 0
    float off = vmin;                   // the grid's left end
    float sx = x;                       // sum(x): constant over the passes
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) sx += __shfl_xor_sync(0xffffffffu, sx, o);
    for (int pass = 0; pass < PASSES; ++pass) {
        const float inv = 1.0f / s;
        int code = (int) rintf((x - off) * inv);   // the nearest grid point
        code = code < 0 ? 0 : (code > 15 ? 15 : code);
        const float cf = (float) code;
        float sc = cf, sc2 = cf * cf, scx = cf * x;
#pragma unroll
        for (int o = 16; o > 0; o >>= 1) {
            sc += __shfl_xor_sync(0xffffffffu, sc, o);
            sc2 += __shfl_xor_sync(0xffffffffu, sc2, o);
            scx += __shfl_xor_sync(0xffffffffu, scx, o);
        }
        constexpr float n = 32.0f;
        const float cm = sc / n;                  // mean(code)
        const float xm = sx / n;                  // mean(x)
        // the products are rounded separately (no FMA): the host reference and the harness do
        // the same, so the parity test can compare bitwise and the toolchain cannot move it
        const float den = sc2 - __fmul_rn(n * cm, cm);   // sum((code - cm)^2)
        const float num = scx - __fmul_rn(n * cm, xm);   // sum((code - cm)*(x - xm))
        if (den > 0.0f) {
            const float s_new = num / den;
            if (s_new > 0.0f) s = s_new;          // the step stays positive: the grid's direction
        }
        off = xm - __fmul_rn(s, cm);               // the grid's left end, at the least-squares offset
    }
    const float inv = 1.0f / s;
    int code = (int) rintf((x - off) * inv);
    code = code < 0 ? 0 : (code > 15 ? 15 : code);
    const uint8_t kc = (uint8_t) code;
    const uint8_t khi = __shfl_down_sync(0xffffffffu, kc, 16);
    byte = (uint8_t) (kc | (khi << 4));
    c_out = off + __fmul_rn(8.0f, s);             // the constant the reader adds (no FMA)
    return f16_from_f32(s);
}

// The same uniform grid WITHOUT the offset: the step fitted per block, the grid centred on zero
// (the codes are then exactly `code - 8` times the step, so this mode is affine and rides the
// prompt fast path).  It is the arm's other half: the pair (kUniformLs, kShear) differs in the
// offset and nothing else.  One least-squares pass over the block, the same rule the harness's
// "uniform+LS" mirrors (tools/kv_affine_price.py).
__device__ __forceinline__ uint16_t q4u_ls_group(float x, int lane, uint8_t& byte) {
    (void) lane;   // the lanes agree through the shuffles; the parameter mirrors q4_group's signature
    float amax = fabsf(x), mval = x;
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) {
        const float a = __shfl_xor_sync(0xffffffffu, amax, o);
        const float v = __shfl_xor_sync(0xffffffffu, mval, o);
        if (a > amax || (a == amax && v > mval)) { amax = a; mval = v; }
    }
    float d = mval / -8.0f;
    const float id = d != 0.0f ? 1.0f / d : 0.0f;
    // An all-zero (or all-NaN) block leaves id = 0, so the cast is 8.5 -> 8 and the code is 8: l = 0,
    // den = 0, the pass is skipped and d is returned unchanged -- the same guard q4_group uses, so
    // __float2int_rz never sees an infinity here.  (A NaN in the input already leaves the block
    // meaningless to every reader; this writer does not special-case it, as q4_group does not.)
    int q = __float2int_rz(x * id + 8.5f);
    const uint8_t qc = (uint8_t) (q < 0 ? 0 : (q > 15 ? 15 : q));
    // one least-squares pass with the assignment held: d = sum(l*x) / sum(l*l), l = code - 8
    const float l = (float) qc - 8.0f;
    float num = l * x, den = l * l;
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) {
        num += __shfl_xor_sync(0xffffffffu, num, o);
        den += __shfl_xor_sync(0xffffffffu, den, o);
    }
    if (den > 0.0f) {
        const float d_ls = num / den;
        if (d_ls != 0.0f && (d_ls < 0.0f) == (d < 0.0f)) {
            d = d_ls;
            const float id2 = d != 0.0f ? 1.0f / d : 0.0f;
            const int q2 = __float2int_rz(x * id2 + 8.5f);
            const uint8_t qc2 = (uint8_t) (q2 < 0 ? 0 : (q2 > 15 ? 15 : q2));
            const uint8_t qhi2 = __shfl_down_sync(0xffffffffu, qc2, 16);
            byte = (uint8_t) (qc2 | (qhi2 << 4));
            return f16_from_f32(d);
        }
    }
    const uint8_t qhi = __shfl_down_sync(0xffffffffu, qc, 16);
    byte = (uint8_t) (qc | (qhi << 4));
    return f16_from_f32(d);
}

// `is_v` is the tensor of the pair this block belongs to; only the kLloyd8* probe modes look at it.
template <Q4Mode M>
__device__ __forceinline__ uint16_t q4_group_any(float x, int lane, uint8_t& byte, float& c_out,
                                                bool is_v = false) {
    if constexpr (M == Q4Mode::kUniform) {
        return q4_group(x, lane, byte);
    } else if constexpr (M == Q4Mode::kUniformLs) {
        return q4u_ls_group(x, lane, byte);
    } else if constexpr (M == Q4Mode::kShear) {
        return q4u_group<kShearPasses>(x, byte, c_out);
    } else if constexpr (M == Q4Mode::kLloyd8) {
        return q4l_group<false, 8>(x, byte);
    } else if constexpr (M == Q4Mode::kLloyd8K) {
        if (is_v) return q4l_group<false, 16>(x, byte);
        return q4l_group<false, 8>(x, byte);
    } else if constexpr (M == Q4Mode::kLloyd8V) {
        if (is_v) return q4l_group<false, 8>(x, byte);
        return q4l_group<false, 16>(x, byte);
    } else {
        constexpr bool fit = (M == Q4Mode::kLloydLs);
        return q4l_group<fit, 16>(x, byte);
    }
}

template <Q4Mode MODE>
__device__ __forceinline__ void q4_store(uint8_t* pool, long long row, int b, int lane, uint16_t d, uint16_t c,
                                         uint8_t byte) {
    if constexpr (MODE == Q4Mode::kShear) {
        // the shear's block carries the constant as well (20 bytes)
        block_q4_0z* blk = reinterpret_cast<block_q4_0z*>(pool + row * (long long) sizeof(block_q4_0z) * 8) + b;
        if (lane == 0) { blk->d = d; blk->c = c; }
        if (lane < 16) blk->qs[lane] = byte;
    } else {
        block_q4_0* blk = reinterpret_cast<block_q4_0*>(pool + row * (long long) sizeof(block_q4_0) * 8) + b;
        if (lane == 0) blk->d = d;
        if (lane < 16) blk->qs[lane] = byte;
    }
}

// One block = one 32-value group of one KV head of K (plane 0) or V (plane 1) for token step_idx; 32 threads.
template <Q4Mode MODE>
__global__ void kv_append_q4_kernel(uint8_t* __restrict__ k_q4, uint8_t* __restrict__ v_q4,
                                    const int32_t* __restrict__ table, const int32_t* __restrict__ step,
                                    int step_stride, int planes,
                                    const float* __restrict__ kcur, const float* __restrict__ vcur,
                                    int kv_heads, int head_dim, int page_size, KvHostPools host) {
    const int step_idx = (int) (blockIdx.z / (unsigned) planes);
    const bool is_v = (blockIdx.z % (unsigned) planes) == 1u;
    const long long pos = (long long) __ldg(step + (long long) step_idx * step_stride + kStepPos);
    const int h = blockIdx.x, b = blockIdx.y, t = threadIdx.x;
    const long long step_off = (long long) step_idx * kv_heads * head_dim;
    const float x = (is_v ? vcur : kcur)[step_off + h * head_dim + b * QK4_0 + t];
    uint8_t byte;
    float c = 0.0f;   // the shear's constant (the other modes leave it at 0)
    const uint16_t d = q4_group_any<MODE>(x, t, byte, c, is_v);
    const long long page = (long long) table[pos / page_size];
    const uint16_t cb = f16_from_f32(c);
    if (page >= 0)
        q4_store<MODE>(is_v ? v_q4 : k_q4, (page * kv_heads + h) * page_size + (pos % page_size), b, t, d, cb, byte);
    if (host.k_q4 != nullptr)
        q4_store<MODE>(is_v ? host.v_q4 : host.k_q4, ((pos / page_size) * kv_heads + h) * page_size + (pos % page_size), b, t,
                 d, cb, byte);
}

// The prompt path: grid (T, kv_heads, groups), K then V; also into the staging pool (identity layout) when given.
template <Q4Mode MODE>
__global__ void kv_append_q4_batch_kernel(uint8_t* __restrict__ k_q4, uint8_t* __restrict__ v_q4,
                                          const int32_t* __restrict__ table, int64_t pos0,
                                          const float* __restrict__ K, const float* __restrict__ V,
                                          int kv_heads, int head_dim, int page_size, int is_v_grid, KvHostPools host,
                                          KvHostPools stage) {
    const long long t = blockIdx.x;
    const long long pos = pos0 + t;
    const int h = blockIdx.y, b = blockIdx.z, th = threadIdx.x;
    const bool is_v = is_v_grid != 0;
    const float x = (is_v ? V : K)[t * (kv_heads * head_dim) + h * head_dim + b * QK4_0 + th];
    uint8_t byte;
    float c = 0.0f;   // the shear's constant (the other modes leave it at 0)
    const uint16_t d = q4_group_any<MODE>(x, th, byte, c, is_v);
    const long long page = (long long) table[pos / page_size];
    const long long row_id = ((pos / page_size) * kv_heads + h) * page_size + (pos % page_size);
    const uint16_t cb = f16_from_f32(c);
    if (page >= 0)
        q4_store<MODE>(is_v ? v_q4 : k_q4, (page * kv_heads + h) * page_size + (pos % page_size), b, th, d, cb, byte);
    if (host.k_q4 != nullptr) q4_store<MODE>(is_v ? host.v_q4 : host.k_q4, row_id, b, th, d, cb, byte);
    if (stage.k_q4 != nullptr) q4_store<MODE>(is_v ? stage.v_q4 : stage.k_q4, row_id, b, th, d, cb, byte);
}

// Gather step[kStepWidth] cells into FP16 scratch (the non-fused attention paths). NK / NV are the level count of
// the K side and the V side (0 = q4_0's uniform grid), because the probe can put 8 levels on one and 16 on the other.
template <int NK, int NV, bool SHEAR = false>
__global__ void kv_gather_q4_kernel(const uint8_t* __restrict__ k_q4, const uint8_t* __restrict__ v_q4,
                                    const int32_t* __restrict__ table, const int32_t* __restrict__ ids,
                                    const int32_t* __restrict__ step, int kv_heads, int head_dim, int page_size,
                                    uint16_t* __restrict__ k_scratch, uint16_t* __restrict__ v_scratch,
                                    const uint16_t* __restrict__ k_win, const uint16_t* __restrict__ v_win,
                                    const int32_t* __restrict__ win_tag, int win_s, int win_r, int win_b,
                                    int win_all) {
    const long long n_ids = (long long) __ldg(step + kStepWidth);
    const int blocks_per_head = head_dim / QK4_0;                        // 8
    const int bytes_per_head = blocks_per_head * (SHEAR ? (int) sizeof(block_q4_0z)   // 160
                                                         : (int) sizeof(block_q4_0)); // 144
    const long long total_blocks = n_ids * kv_heads * blocks_per_head;

    const long long blk_idx = (long long) blockIdx.x * blockDim.y + threadIdx.y;
    if (blk_idx >= total_blocks) return;

    const int t = threadIdx.x;
    const long long id = blk_idx / (kv_heads * blocks_per_head);
    const int rem = (int) (blk_idx % (kv_heads * blocks_per_head));
    const int h = rem / blocks_per_head;
    const int b = rem % blocks_per_head;

    const int cell = ids[id];
    const long long page = (long long) table[cell / page_size];
    const long long row = (page * kv_heads + h) * page_size + (cell % page_size);

    const long long dst_offset = ((id * kv_heads + h) * head_dim) + (b * QK4_0 + t);
    // WP-2a window: exact fp16 from the always-resident side pool (row from qsa_win_slot, no paging) - also when
    // the quantized block is not resident. The horizon is step[kStepNKv] (device-fresh, graph-safe). Null pools
    // (every other mode) skip this uniformly; a recent ring row whose tag no longer names this cell (a rewind or
    // a resume left a later cell there) reads quantized instead of wrong.
    if (k_win != nullptr && v_win != nullptr) {
        QsaAttnPools wq;
        wq.k_win = k_win;
        wq.v_win = v_win;
        wq.win_tag = win_tag;
        wq.win_s = win_s;
        wq.win_r = win_r;
        wq.win_b = win_b;
        wq.win_all = win_all;
        const long long ws = qsa_win_slot(wq, cell, (long long) __ldg(step + kStepNKv));
        if (ws >= 0) {
            const long long wrow = (ws * kv_heads + h) * head_dim + (b * QK4_0 + t);
            k_scratch[dst_offset] = k_win[wrow];
            v_scratch[dst_offset] = v_win[wrow];
            return;
        }
    }

    // The shear's block is 20 bytes (the codes, the scale, the constant), so EVERY field has to
    // be read through the 20-byte struct: a block_q4_0* would advance 18 bytes per block and
    // read the wrong block from the second one on.
    const int j = t < 16 ? t : (t - 16);
    float kd, vd, kc = 0.0f, vc = 0.0f;
    uint8_t k_byte, v_byte;
    if constexpr (SHEAR) {
        const block_q4_0z* kz = reinterpret_cast<const block_q4_0z*>(k_q4 + row * bytes_per_head) + b;
        const block_q4_0z* vz = reinterpret_cast<const block_q4_0z*>(v_q4 + row * bytes_per_head) + b;
        kd = f32_from_f16(kz->d);
        vd = f32_from_f16(vz->d);
        kc = f32_from_f16(kz->c);
        vc = f32_from_f16(vz->c);
        k_byte = kz->qs[j];
        v_byte = vz->qs[j];
    } else {
        const block_q4_0* k_blk = reinterpret_cast<const block_q4_0*>(k_q4 + row * bytes_per_head) + b;
        const block_q4_0* v_blk = reinterpret_cast<const block_q4_0*>(v_q4 + row * bytes_per_head) + b;
        kd = f32_from_f16(k_blk->d);
        vd = f32_from_f16(v_blk->d);
        k_byte = k_blk->qs[j];
        v_byte = v_blk->qs[j];
    }
    const int kn = (t < 16) ? (k_byte & 0x0F) : (k_byte >> 4);
    const int vn = (t < 16) ? (v_byte & 0x0F) : (v_byte >> 4);
    const int kq = (NK == 0) ? (kn - 8) : (int) ((NK == 8) ? kQ4l8Map[kn] : kQ4lMap[kn]);
    const int vq = (NV == 0) ? (vn - 8) : (int) ((NV == 8) ? kQ4l8Map[vn] : kQ4lMap[vn]);
    k_scratch[dst_offset] = f16_from_f32((float) kq * kd + kc);
    v_scratch[dst_offset] = f16_from_f32((float) vq * vd + vc);
}

void need_256(const QsaShapes& s, const char* what) {
    if (s.head_dim != 256) {
        std::fprintf(stderr, "%s: head_dim must be 256 (the Hadamard transform's size)\n", what);
        std::exit(1);
    }
}

}  // namespace

void fwht256_cuda(const float* src, float* dst, int64_t n_rows, void* stream) {
    if (n_rows <= 0) return;
    const int rows_per_block = 4;
    const int64_t num_blocks = (n_rows + rows_per_block - 1) / rows_per_block;
    fwht256_kernel<<<dim3((unsigned) num_blocks), dim3(32, rows_per_block), 0, (cudaStream_t) stream>>>(
        src, dst, n_rows, 1.0f / 16.0f);
    check("fwht256 launch");
}

// The rotation of the KV rows with the centering: `offsets` null means the plain rotation, so every
// writer site can call this one function and none of them can forget the centering.
__global__ void fwht256_centered_kernel(float* __restrict__ data, int64_t n_rows, int n_heads,
                                         const float* __restrict__ offsets, float scale);
__global__ void add_center_offsets_kernel(float* __restrict__ data, int64_t n_rows, int n_q_heads, int g,
                                          const float* __restrict__ offsets);
void fwht256_kv_inplace_cuda(float* data, int64_t n_rows, int n_heads, const float* offsets, void* stream) {
    if (n_rows <= 0) return;
    if (offsets == nullptr) {
        fwht256_inplace_cuda(data, n_rows, stream);
        return;
    }
    const int rows_per_block = 4;
    const int64_t num_blocks = (n_rows + rows_per_block - 1) / rows_per_block;
    fwht256_centered_kernel<<<dim3((unsigned) num_blocks), dim3(32, rows_per_block), 0, (cudaStream_t) stream>>>(
        data, n_rows, n_heads, offsets, 1.0f / 16.0f);
    check("fwht256_centered launch");
}

// The V add-back on the attention output (see add_center_offsets_kernel).
void add_center_offsets_cuda(float* data, int64_t n_rows, int n_q_heads, int g, const float* offsets,
                             void* stream) {
    if (data == nullptr || offsets == nullptr || n_rows <= 0) return;
    const int rows_per_block = 4;
    const int64_t num_blocks = (n_rows + rows_per_block - 1) / rows_per_block;
    add_center_offsets_kernel<<<dim3((unsigned) num_blocks), dim3(32, rows_per_block), 0, (cudaStream_t) stream>>>(
        data, n_rows, n_q_heads, g, offsets);
    check("add_center_offsets launch");
}

// The centering (bench/results/2026-11-10-centering/): the per-(KV head, channel) offset of the K
// (or V) rows is subtracted BEFORE the rotation, in the channel space, so the quantizer's scale
// (a maximum statistic) sees the signal without its fixed part.  The rows of the buffer are
// (position, KV head, channel), so the head of a row is (row / N) % n_heads.
__global__ void fwht256_centered_kernel(float* __restrict__ data, int64_t n_rows, int n_heads,
                                         const float* __restrict__ offsets, float scale) {
    constexpr int warp_size = 32;
    constexpr int N = 256;
    constexpr int el_w = N / warp_size;

    const int64_t r = (int64_t) blockIdx.x * blockDim.y + threadIdx.y;
    if (r >= n_rows) return;
    // the rows are (position, KV head, channel) in that order: the head of a row is r % n_heads
    const int head = (int) (r % n_heads);
    const float* __restrict__ off = offsets + (int64_t) head * N;
    float* __restrict__ row = data + r * N;
    const int lane = threadIdx.x;

    float reg[el_w];
#pragma unroll
    for (int i = 0; i < el_w; ++i)
        reg[i] = (row[i * warp_size + lane] - off[i * warp_size + lane]) * scale;

    // the low 5 index bits live across lanes
#pragma unroll
    for (int h = 1; h < warp_size; h *= 2) {
#pragma unroll
        for (int j = 0; j < el_w; ++j) {
            const float val = reg[j];
            const float val2 = __shfl_xor_sync(0xffffffffu, val, h, warp_size);
            reg[j] = (lane & h) == 0 ? val + val2 : val2 - val;
        }
    }
    // the high 3 bits across each lane's registers
#pragma unroll
    for (int h = warp_size; h < N; h *= 2) {
        const int step = h / warp_size;
#pragma unroll
        for (int j = 0; j < el_w; j += 2 * step) {
#pragma unroll
            for (int k = 0; k < step; ++k) {
                const float x = reg[j + k];
                const float y = reg[j + k + step];
                reg[j + k] = x + y;
                reg[j + k + step] = x - y;
            }
        }
    }
#pragma unroll
    for (int i = 0; i < el_w; ++i) row[i * warp_size + lane] = reg[i];
}

// The add-back of the V offsets on the attention output: the stored V rows are (V - m_V) rotated,
// and the attention weights sum to one, so the output is (out - m_V) in the rotated domain.  The
// output rows are (position, QUERY head, channel) and the query heads are grouped per KV head (G
// query heads per KV head), so the row's KV head is (query head) / G.
__global__ void add_center_offsets_kernel(float* __restrict__ data, int64_t n_rows, int n_q_heads, int g,
                                          const float* __restrict__ offsets) {
    constexpr int N = 256;
    const int64_t r = (int64_t) blockIdx.x * blockDim.y + threadIdx.y;
    if (r >= n_rows) return;
    // the rows are (position, QUERY head, channel): the query head of a row is r % n_q_heads
    const int q_head = (int) (r % n_q_heads);
    const int kv_head = g > 0 ? q_head / g : q_head;
    const float* __restrict__ off = offsets + (int64_t) kv_head * N;
    float* __restrict__ row = data + r * N;
    const int lane = threadIdx.x;
    for (int i = lane; i < N; i += 32) row[i] += off[i];
}

// The append kernels are templated on the write side's full Q4Mode; the wrappers pick the variant so that no caller
// has to know which kernel a format needs.
#define STRATA_KV_Q4_APPEND(M)                                                                   \
    kv_append_q4_kernel<M><<<grid, 32, 0, (cudaStream_t) stream>>>(                              \
        k_q4, v_q4, page_table, step, step_stride, planes, kcur, vcur, (int) s.n_head_kv,        \
        (int) s.head_dim, (int) s.page_size, host ? *host : KvHostPools{})

void kv_append_q4_steps(uint8_t* k_q4, uint8_t* v_q4, const int32_t* page_table, const int32_t* step,
                        int step_stride, int n_steps, const float* kcur, const float* vcur, const QsaShapes& s,
                        void* stream, const KvHostPools* host, Q4Mode mode) {
    if (n_steps <= 0) return;
    need_256(s, "kv_append_q4");
    const int planes = (v_q4 == nullptr || (v_q4 == k_q4 && vcur == kcur)) ? 1 : 2;
    const dim3 grid((unsigned) s.n_head_kv, (unsigned) (s.head_dim / QK4_0), (unsigned) (n_steps * planes));
    if (mode == Q4Mode::kLloydLs) STRATA_KV_Q4_APPEND(Q4Mode::kLloydLs);
    else if (mode == Q4Mode::kLloyd) STRATA_KV_Q4_APPEND(Q4Mode::kLloyd);
    else if (mode == Q4Mode::kLloydWin) STRATA_KV_Q4_APPEND(Q4Mode::kLloydWin);   // kLloyd's writer (else branch)
    else if (mode == Q4Mode::kLloyd8) STRATA_KV_Q4_APPEND(Q4Mode::kLloyd8);
    else if (mode == Q4Mode::kLloyd8K) STRATA_KV_Q4_APPEND(Q4Mode::kLloyd8K);
    else if (mode == Q4Mode::kLloyd8V) STRATA_KV_Q4_APPEND(Q4Mode::kLloyd8V);
    else if (mode == Q4Mode::kUniformLs) STRATA_KV_Q4_APPEND(Q4Mode::kUniformLs);
    else if (mode == Q4Mode::kShear) STRATA_KV_Q4_APPEND(Q4Mode::kShear);
    else STRATA_KV_Q4_APPEND(Q4Mode::kUniform);
    check("kv_append_q4 launch");
}
#undef STRATA_KV_Q4_APPEND

void kv_append_q4_step(uint8_t* k_q4, uint8_t* v_q4, const int32_t* page_table, const int32_t* step,
                       const float* kcur, const float* vcur, const QsaShapes& s, void* stream,
                       const KvHostPools* host, Q4Mode mode) {
    kv_append_q4_steps(k_q4, v_q4, page_table, step, 0, 1, kcur, vcur, s, stream, host, mode);
}

#define STRATA_KV_Q4_BATCH(M)                                                                    \
    kv_append_q4_batch_kernel<M><<<grid, 32, 0, cs>>>(k_q4, v_q4, page_table, pos0, K, V,        \
                                                     (int) s.n_head_kv, (int) s.head_dim,        \
                                                     (int) s.page_size, is_v, h, st)

void kv_append_q4(uint8_t* k_q4, uint8_t* v_q4, const int32_t* page_table, int64_t pos0, int64_t T, const float* K,
                  const float* V, const QsaShapes& s, void* stream, const KvHostPools* host, const KvHostPools* stage,
                  Q4Mode mode) {
    if (T <= 0) return;
    need_256(s, "kv_append_q4");
    const dim3 grid((unsigned) T, (unsigned) s.n_head_kv, (unsigned) (s.head_dim / QK4_0));
    cudaStream_t cs = (cudaStream_t) stream;
    const KvHostPools h = host ? *host : KvHostPools{}, st = stage ? *stage : KvHostPools{};
    for (int is_v = 0; is_v < 2; ++is_v) {
        if (mode == Q4Mode::kLloydLs) STRATA_KV_Q4_BATCH(Q4Mode::kLloydLs);
        else if (mode == Q4Mode::kLloyd) STRATA_KV_Q4_BATCH(Q4Mode::kLloyd);
        else if (mode == Q4Mode::kLloydWin) STRATA_KV_Q4_BATCH(Q4Mode::kLloydWin);   // kLloyd's writer (else branch)
        else if (mode == Q4Mode::kLloyd8) STRATA_KV_Q4_BATCH(Q4Mode::kLloyd8);
        else if (mode == Q4Mode::kLloyd8K) STRATA_KV_Q4_BATCH(Q4Mode::kLloyd8K);
        else if (mode == Q4Mode::kLloyd8V) STRATA_KV_Q4_BATCH(Q4Mode::kLloyd8V);
        else if (mode == Q4Mode::kUniformLs) STRATA_KV_Q4_BATCH(Q4Mode::kUniformLs);
        else if (mode == Q4Mode::kShear) STRATA_KV_Q4_BATCH(Q4Mode::kShear);
        else STRATA_KV_Q4_BATCH(Q4Mode::kUniform);
    }
    check("kv_append_q4 batch launch");
}
#undef STRATA_KV_Q4_BATCH

// WP-2a window side pool (kv_q4.hpp, kLloydWin): the exact fp16 twin of the cells a windowed reader takes exact.
// Rows [R sinks][S + B recent ring] per tensor, [(row, kv_head, head_dim)] - the row comes from the same formula
// as `qsa_win_slot`, no paging, always resident. One kernel for every append shape: positions come from the same
// step array the q4 steps-kernel reads (step[step_idx * step_stride + kStepPos]), or from a contiguous (pos0, T)
// batch. Called with the same (already rotated) K/V the q4 append sees, right next to it, on every append path.
//
// The recent ring carries the cell it holds (`win_tag[row] = pos`), written by one thread of the block whose
// values landed on that row - so tag and values belong together, which is what lets a reader trust the tag.
//
// R1: K and V are written in ONE launch (the kernel takes both pools and both sources and skips a side whose pool
// is null), which halves the extra launches per append - a decode step went from 24 to 12, and the tag is written
// once. The alternative - writing the twin inside the templated q4 append kernels, where `cur` is already in a
// register - would also save the re-read of K/V (6345 cells x 49 KiB ~= 300 MiB over a 6.3K prefill), but it
// would put a window branch into every mode's hot path. Not worth it (section 7.19).
__device__ __forceinline__ long long win_row_of(int64_t pos, int win_s, int win_r, int win_b) {
    if (pos < win_r) return pos;   // sink row: cell-addressed, written once, never moved
    const long long ring = (long long) win_s + (long long) win_b;
    return ring > 0 ? (long long) win_r + pos % ring : -1;
}

__global__ void kv_win_append_kernel(uint16_t* __restrict__ k_win, uint16_t* __restrict__ v_win,
                                     int32_t* __restrict__ tag, const float* __restrict__ kcur,
                                     const float* __restrict__ vcur, const int32_t* __restrict__ step,
                                     int step_stride, int n_head_kv, int head_dim, int win_s, int win_r,
                                     int win_b) {
    const int64_t step_idx = (int64_t) blockIdx.z;
    const int64_t pos = (int64_t) step[step_idx * step_stride + kStepPos];
    const int h = blockIdx.x, d0 = (blockIdx.y * (int) blockDim.x + (int) threadIdx.x);
    if (d0 >= head_dim) return;
    // The row is computed, not bounded: sinks are cell-addressed and the recent rows are a ring, so a position
    // can only be dropped when the ring does not exist (sinks-only window) - never reach a neighbouring tensor.
    const int64_t row = win_row_of(pos, win_s, win_r, win_b);
    if (row < 0) return;
    if (tag != nullptr && pos >= win_r && h == 0 && d0 == 0)
        tag[pos % ((int64_t) win_s + win_b)] = (int32_t) pos;
    const int64_t n = (int64_t) n_head_kv * head_dim;
    const int64_t i = row * n + h * head_dim + d0, s = step_idx * n + h * head_dim + d0;
    if (k_win != nullptr) k_win[i] = f16_from_f32(kcur[s]);
    if (v_win != nullptr) v_win[i] = f16_from_f32(vcur[s]);
}

void kv_append_win_steps(uint16_t* k_win, uint16_t* v_win, int32_t* win_tag, const int32_t* step, int step_stride,
                         int n_steps, const float* kcur, const float* vcur, int n_head_kv, int head_dim,
                         void* stream, int win_s, int win_r, int win_b) {
    if (n_steps <= 0) return;
    // K2: this used to return silently, leaving the side pool stale while the caller believed it was written.
    // A geometry the grid cannot cover is a programming error, so it fails like need_256 does.
    if (head_dim <= 0 || head_dim % 32 != 0 || win_s < 0 || win_r < 0 || win_b < 0) {
        std::fprintf(stderr,
                     "strata: kv_append_win_steps: head_dim %d must be a positive multiple of 32 (the window "
                     "writer's grid) and the window grid (%d/%d/%d) non-negative - the exact side pool would not "
                     "be written\n", head_dim, win_s, win_r, win_b);
        std::exit(1);
    }
    // One call's cells must land on distinct recent rows: two cells of the same launch on one row would race the
    // tag against the values (and the values against each other), which is exactly the silent wrongness the ring
    // exists to avoid. The ring holds S + B rows, so a batch is bounded by S + B - and a prompt chunk that is
    // larger fails here, loudly, naming the knob, instead of overwriting rows its own queries still need.
    const int64_t ring = (int64_t) win_s + win_b;
    if (ring > 0 && (int64_t) n_steps > ring) {
        std::fprintf(stderr,
                     "strata: kv_append_win_steps: %d cells in one batch do not fit the %lld-row recent ring "
                     "(S %d + STRATA_KV_WINDOW_BATCH %d) - raise STRATA_KV_WINDOW_BATCH (or lower --prefill)\n",
                     n_steps, (long long) ring, win_s, win_b);
        std::exit(1);
    }
    if (k_win != nullptr || v_win != nullptr)
        kv_win_append_kernel<<<dim3((unsigned) n_head_kv, (unsigned) (head_dim / 32), (unsigned) n_steps), 32, 0,
                                    (cudaStream_t) stream>>>(k_win, v_win, win_tag, kcur, vcur, step, step_stride,
                                                             n_head_kv, head_dim, win_s, win_r, win_b);
    check("kv_append_win launch");
}

// The contiguous batch twin (the prefill path's kv_append_q4 with pos0/T): same rows, positions pos0 + t, and the
// same single launch for K and V.
__global__ void kv_win_append_batch_kernel(uint16_t* __restrict__ k_win, uint16_t* __restrict__ v_win,
                                           int32_t* __restrict__ tag, const float* __restrict__ kcur,
                                           const float* __restrict__ vcur, int64_t pos0, int n_head_kv, int head_dim,
                                           int win_s, int win_r, int win_b) {
    const int64_t t = (int64_t) blockIdx.z;
    const int h = blockIdx.x, d0 = (blockIdx.y * (int) blockDim.x + (int) threadIdx.x);
    if (d0 >= head_dim) return;
    const int64_t pos = pos0 + t;
    const int64_t row = win_row_of(pos, win_s, win_r, win_b);
    if (row < 0) return;
    if (tag != nullptr && pos >= win_r && h == 0 && d0 == 0)
        tag[pos % ((int64_t) win_s + win_b)] = (int32_t) pos;
    const int64_t n = (int64_t) n_head_kv * head_dim;
    const int64_t i = row * n + h * head_dim + d0, s = t * n + h * head_dim + d0;
    if (k_win != nullptr) k_win[i] = f16_from_f32(kcur[s]);
    if (v_win != nullptr) v_win[i] = f16_from_f32(vcur[s]);
}

void kv_append_win(uint16_t* k_win, uint16_t* v_win, int32_t* win_tag, int64_t pos0, int64_t T, const float* K,
                   const float* V, int n_head_kv, int head_dim, void* stream, int win_s, int win_r, int win_b) {
    if (T <= 0) return;
    // K2: loud instead of the old silent return (a stale side pool the caller believes was written).
    if (head_dim <= 0 || head_dim % 32 != 0) {
        std::fprintf(stderr, "strata: kv_append_win: head_dim %d must be a positive multiple of 32 (the window "
                             "writer's grid) - the exact side pool would not be written\n", head_dim);
        std::exit(1);
    }
    // Same invariant as the steps variant: the batch's cells must land on distinct recent rows, so T <= S + B.
    const int64_t ring = (int64_t) win_s + win_b;
    if (ring > 0 && T > ring) {
        std::fprintf(stderr,
                     "strata: kv_append_win: %lld cells in one batch do not fit the %lld-row recent ring "
                     "(S %d + STRATA_KV_WINDOW_BATCH %d) - raise STRATA_KV_WINDOW_BATCH (or lower --prefill)\n",
                     (long long) T, (long long) ring, win_s, win_b);
        std::exit(1);
    }
    if (pos0 < 0) {
        std::fprintf(stderr, "strata: kv_append_win: cells start at %lld, before the window's first cell\n",
                     (long long) pos0);
        std::exit(1);
    }
    if (k_win != nullptr || v_win != nullptr)
        kv_win_append_batch_kernel<<<dim3((unsigned) n_head_kv, (unsigned) (head_dim / 32), (unsigned) T), 32, 0,
                                     (cudaStream_t) stream>>>(k_win, v_win, win_tag, K, V, pos0, n_head_kv,
                                                              head_dim, win_s, win_r, win_b);
    check("kv_append_win batch launch");
}

#define STRATA_KV_Q4_GATHER(NK, NV, SHEAR)                                                                       \
    kv_gather_q4_kernel<NK, NV, SHEAR><<<dim3(num_blocks), dim3(32, rows_per_block), 0, (cudaStream_t) stream>>>(  \
        k_q4, v_q4, page_table, ids, step, (int) s.n_head_kv, (int) s.head_dim, (int) s.page_size, k_scratch,     \
        v_scratch, k_win, v_win, win_tag, win_s, win_r, win_b, win_all)

void kv_gather_q4_step(const uint8_t* k_q4, const uint8_t* v_q4, const int32_t* page_table, const int32_t* ids,
                       const int32_t* step, int64_t max_ids, const QsaShapes& s, uint16_t* k_scratch,
                       uint16_t* v_scratch, void* stream, Q4Mode mode, const uint16_t* k_win,
                       const uint16_t* v_win, const int32_t* win_tag, int win_s, int win_r, int win_b) {
    if (max_ids <= 0) return;
    const int blocks_per_head = (int) (s.head_dim / QK4_0);
    const int64_t total_blocks = max_ids * s.n_head_kv * blocks_per_head;
    const int rows_per_block = 4;
    const unsigned num_blocks = (unsigned) ((total_blocks + rows_per_block - 1) / rows_per_block);
    // Same switch as qsa_attn_pools: STRATA_KV_WINDOW_TAG=1, off unless set. Read here so the scratch path and
    // the fused path agree without a new argument on every backend's stub.
    static const int win_all = [] { const char* e = std::getenv("STRATA_KV_WINDOW_TAG"); return e && e[0] == '1'; }();
    if (q4_lloyd(mode)) {   // the gather reads every level-table mode alike: the scale is in the block
        if (q4_l8_k(mode) && q4_l8_v(mode)) STRATA_KV_Q4_GATHER(8, 8, false);
        else if (q4_l8_k(mode)) STRATA_KV_Q4_GATHER(8, 16, false);
        else if (q4_l8_v(mode)) STRATA_KV_Q4_GATHER(16, 8, false);
        else STRATA_KV_Q4_GATHER(16, 16, false);
    } else if (q4_shear(mode)) {   // the shear: the uniform grid plus the per-block constant
        STRATA_KV_Q4_GATHER(0, 0, true);
    } else {
        STRATA_KV_Q4_GATHER(0, 0, false);
    }
    check("kv_gather_q4 launch");
}
#undef STRATA_KV_Q4_GATHER

}  // namespace strata::kernels
