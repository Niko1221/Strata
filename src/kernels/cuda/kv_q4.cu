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

// `is_v` is the tensor of the pair this block belongs to; only the kLloyd8* probe modes look at it.
template <Q4Mode M>
__device__ __forceinline__ uint16_t q4_group_any(float x, int lane, uint8_t& byte, bool is_v = false) {
    if constexpr (M == Q4Mode::kUniform) {
        return q4_group(x, lane, byte);
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

__device__ __forceinline__ void q4_store(uint8_t* pool, long long row, int b, int lane, uint16_t d, uint8_t byte) {
    block_q4_0* blk = reinterpret_cast<block_q4_0*>(pool + row * (long long) sizeof(block_q4_0) * 8) + b;
    if (lane == 0) blk->d = d;
    if (lane < 16) blk->qs[lane] = byte;
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
    const uint16_t d = q4_group_any<MODE>(x, t, byte, is_v);
    const long long page = (long long) table[pos / page_size];
    if (page >= 0) q4_store(is_v ? v_q4 : k_q4, (page * kv_heads + h) * page_size + (pos % page_size), b, t, d, byte);
    if (host.k_q4 != nullptr)
        q4_store(is_v ? host.v_q4 : host.k_q4, ((pos / page_size) * kv_heads + h) * page_size + (pos % page_size), b, t,
                 d, byte);
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
    const uint16_t d = q4_group_any<MODE>(x, th, byte, is_v);
    const long long page = (long long) table[pos / page_size];
    const long long row_id = ((pos / page_size) * kv_heads + h) * page_size + (pos % page_size);
    if (page >= 0) q4_store(is_v ? v_q4 : k_q4, (page * kv_heads + h) * page_size + (pos % page_size), b, th, d, byte);
    if (host.k_q4 != nullptr) q4_store(is_v ? host.v_q4 : host.k_q4, row_id, b, th, d, byte);
    if (stage.k_q4 != nullptr) q4_store(is_v ? stage.v_q4 : stage.k_q4, row_id, b, th, d, byte);
}

// Gather step[kStepWidth] cells into FP16 scratch (the non-fused attention paths). NK / NV are the level count of
// the K side and the V side (0 = q4_0's uniform grid), because the probe can put 8 levels on one and 16 on the other.
template <int NK, int NV>
__global__ void kv_gather_q4_kernel(const uint8_t* __restrict__ k_q4, const uint8_t* __restrict__ v_q4,
                                    const int32_t* __restrict__ table, const int32_t* __restrict__ ids,
                                    const int32_t* __restrict__ step, int kv_heads, int head_dim, int page_size,
                                    uint16_t* __restrict__ k_scratch, uint16_t* __restrict__ v_scratch,
                                    const uint16_t* __restrict__ k_win, const uint16_t* __restrict__ v_win,
                                    const int32_t* __restrict__ win_tag, int win_s, int win_r, int win_b) {
    const long long n_ids = (long long) __ldg(step + kStepWidth);
    const int blocks_per_head = head_dim / QK4_0;                        // 8
    const int bytes_per_head = blocks_per_head * sizeof(block_q4_0);    // 144
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
        const long long ws = qsa_win_slot(wq, cell, (long long) __ldg(step + kStepNKv));
        if (ws >= 0) {
            const long long wrow = (ws * kv_heads + h) * head_dim + (b * QK4_0 + t);
            k_scratch[dst_offset] = k_win[wrow];
            v_scratch[dst_offset] = v_win[wrow];
            return;
        }
    }

    const block_q4_0* k_blk = reinterpret_cast<const block_q4_0*>(k_q4 + row * bytes_per_head) + b;
    const block_q4_0* v_blk = reinterpret_cast<const block_q4_0*>(v_q4 + row * bytes_per_head) + b;
    const float kd = f32_from_f16(k_blk->d);
    const float vd = f32_from_f16(v_blk->d);
    const int j = t < 16 ? t : (t - 16);
    const uint8_t k_byte = k_blk->qs[j];
    const uint8_t v_byte = v_blk->qs[j];
    const int kn = (t < 16) ? (k_byte & 0x0F) : (k_byte >> 4);
    const int vn = (t < 16) ? (v_byte & 0x0F) : (v_byte >> 4);
    const int kq = (NK == 0) ? (kn - 8) : (int) ((NK == 8) ? kQ4l8Map[kn] : kQ4lMap[kn]);
    const int vq = (NV == 0) ? (vn - 8) : (int) ((NV == 8) ? kQ4l8Map[vn] : kQ4lMap[vn]);
    k_scratch[dst_offset] = f16_from_f32((float) kq * kd);
    v_scratch[dst_offset] = f16_from_f32((float) vq * vd);
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

#define STRATA_KV_Q4_GATHER(NK, NV)                                                                              \
    kv_gather_q4_kernel<NK, NV><<<dim3(num_blocks), dim3(32, rows_per_block), 0, (cudaStream_t) stream>>>(        \
        k_q4, v_q4, page_table, ids, step, (int) s.n_head_kv, (int) s.head_dim, (int) s.page_size, k_scratch,     \
        v_scratch, k_win, v_win, win_tag, win_s, win_r, win_b)

void kv_gather_q4_step(const uint8_t* k_q4, const uint8_t* v_q4, const int32_t* page_table, const int32_t* ids,
                       const int32_t* step, int64_t max_ids, const QsaShapes& s, uint16_t* k_scratch,
                       uint16_t* v_scratch, void* stream, Q4Mode mode, const uint16_t* k_win,
                       const uint16_t* v_win, const int32_t* win_tag, int win_s, int win_r, int win_b) {
    if (max_ids <= 0) return;
    const int blocks_per_head = (int) (s.head_dim / QK4_0);
    const int64_t total_blocks = max_ids * s.n_head_kv * blocks_per_head;
    const int rows_per_block = 4;
    const unsigned num_blocks = (unsigned) ((total_blocks + rows_per_block - 1) / rows_per_block);
    if (q4_lloyd(mode)) {   // the gather reads every level-table mode alike: the scale is in the block
        if (q4_l8_k(mode) && q4_l8_v(mode)) STRATA_KV_Q4_GATHER(8, 8);
        else if (q4_l8_k(mode)) STRATA_KV_Q4_GATHER(8, 16);
        else if (q4_l8_v(mode)) STRATA_KV_Q4_GATHER(16, 8);
        else STRATA_KV_Q4_GATHER(16, 16);
    } else {
        STRATA_KV_Q4_GATHER(0, 0);
    }
    check("kv_gather_q4 launch");
}
#undef STRATA_KV_Q4_GATHER

}  // namespace strata::kernels
