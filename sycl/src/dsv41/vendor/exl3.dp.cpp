// Copyright (c) 2026 Peasant Smith and the Project Maya contributors (MIT; https://github.com/mw00/project-maya, v1.1.17).
// Ported into Strata (third_party/project-maya/NOTICE.md, LICENSE); changes are in the git history of this file.
// src/kernels/cuda/exl3.cu - the EXL3 device kernels (the format: include/strata/kernels/exl3.hpp).
//
// The codebooks and the trellis window arithmetic follow exllamav3 (MIT, turboderp - exllamav3_ext/quant/
// codebook.cuh, exl3_dq.cuh, hadamard_inner.cuh), and so does decoding a tile straight into tensor-core fragments
// (its small-m GEMV and GEMM); the kernels around them are shaped for this engine's decode:
//
//   * a block owns 128 outputs (8 tiles - exactly one output rotation block) over a slice of at most 4096 inputs;
//     warp w takes tile column w through every 16-row slice, decoding 8 weights per lane (one fragment: 2 columns x
//     4 rows) - on sm_80 and later straight into one mma.m16n8k16 a tile with up to 8 activations as its columns,
//     F32 accumulation; before sm_80 (and on HIP) F32 FMAs on the CUDA cores;
//   * the input rotation runs per block, 128 inputs at a time (F32 or q8_1 in, FP16 in shared memory), so the
//     engine's activation buffers are used as they are; the next 128 inputs' tile words load while these decode;
//   * a matrix with more than 4096 inputs, or too few output blocks to fill the device, splits its inputs over
//     blocks: the partial sums go to a per-stream scratch and the LAST block of an output block adds them up in
//     split order (deterministic) and runs the output rotation.
//
// NVIDIA GPUs before sm_80 (V100, T4, RTX 20) decode with v1.1.8's kernels instead (exl3_legacy.cu, use_legacy
// below): this file's CUDA-core fallback was slower there (a V100 with EXL3-3.05bpw: decode 4.9 -> 4.4 tok/s).  The
// prompt path below is this file's on every GPU.
#define DPCT_PROFILING_ENABLED
#include <sycl/sycl.hpp>
#include <dpct/dpct.hpp>
#include "../ds4_launch.h" // strata_ds4_launch: <<<>>> on CUDA and HIP, a dispatch on Metal
#if defined(STRATA_METAL)
#include <ds4_metal_host.h>
#endif
#include "strata/dsv41/exl3.hpp"

#include <algorithm>
#include <atomic>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <mutex>
#include <utility>
#include <cmath>

#include <time.h>

#if 0 // NVIDIA legacy kernels are not part of the SYCL target
namespace strata::dsv41k::exl3::legacy {   // exl3_legacy.cu: v1.1.8's decode kernels
int launch_errors();
bool prepare(dpct::queue_ptr s);
bool several_ok();
bool mv(const MvJob *jobs, int n, dpct::queue_ptr s);
void moe_gate_up(const MoeArgs &a, const float *x, float *h, dpct::queue_ptr s);
void moe_down(const MoeArgs &a, const float *h, const float *sh_out, float *out,
              dpct::queue_ptr s);
bool win_experts(const WinArgs &a, dpct::queue_ptr s);
}  // namespace strata::dsv41k::exl3::legacy
#define EXL3_LEGACY(call) \
    if (use_legacy()) return legacy::call
#else
#define EXL3_LEGACY(call) (void) 0
#endif

namespace strata::dsv41k::exl3 {
namespace {

constexpr int kThreads = 256, kWarps = 8;
constexpr int kCols = 128;                // outputs per block: 8 tiles, one rotation block
constexpr int kMaxKR = 4096;              // inputs per block at most (mv's split)
// a routed expert's widths at most (moe_* / win_*: a block streams its inputs through block_gemv's two 128-value chunk
// buffers, so the shared memory does not grow with them; DeepSeek-V4.1's n_embd is 5120)
constexpr int kMaxMoeK = 8192;
constexpr float kHad = 0.088388347648f;   // 1 / sqrt(128)

struct Q81 {   // the engine's q8_1 block (ggml block_q8_1): d, sum, 32 int8
    sycl::half2 ds;
    int8_t qs[32];
};

std::atomic<int> g_launch_errors{0};
thread_local const int* g_skip = nullptr;   // set_skip: the calling thread's flag (nullptr: none)
void launch_check(const char* what) {
    /*
    DPCT1010:374: SYCL uses exceptions to report errors and does not use the
    error codes. The cudaGetLastError function call was replaced with 0. You
    need to rewrite this code.
    */
    const dpct::err0 e = 0;
    /*
    DPCT1000:373: Error handling if-stmt was detected but could not be
    rewritten.
    */
    if (e != 0) {
        /*
        DPCT1009:375: SYCL reports errors using exceptions and does not use
        error codes. Please replace the "get_error_string_dummy(...)" with a
        real error-handling function.
        */
        std::fprintf(stderr, "exl3 %s: %s\n", what,
                     dpct::get_error_string_dummy(e));
        /*
        DPCT1001:372: The statement could not be removed.
        */
        g_launch_errors.fetch_add(1, std::memory_order_relaxed);
    }
}

// ---------------------------------------------------------------- the codebooks
__dpct_inline__ uint32_t bytesum_plus(uint32_t x, uint32_t acc) {
#if defined(STRATA_USE_HIP)
    return acc + (x & 0xffu) + ((x >> 8) & 0xffu) + ((x >> 16) & 0xffu) + (x >> 24);
#else
    return dpct::dp4a(x, 0x01010101u, acc);
#endif
}

// two states -> two values, FP16 arithmetic as exllamav3's decode_3inst_2 (the tensor-core kernels take the half2)
template <int cb>
__dpct_inline__ sycl::half2 decode2h(uint32_t s0, uint32_t s1) {
    if constexpr (cb == 2) {
        const uint32_t a = bytesum_plus(s0 * 0x83DCD12Du, 0x6400u);
        const uint32_t b = bytesum_plus(s1 * 0x83DCD12Du, 0x6400u);
        const sycl::half2 h = sycl::half2(
            sycl::bit_cast<sycl::half, unsigned short>((unsigned short)a),
            sycl::bit_cast<sycl::half, unsigned short>((unsigned short)b));
        return sycl::fma(
            h, sycl::half2(sycl::bit_cast<sycl::half, unsigned short>(0x1eee)),
            sycl::half2(sycl::bit_cast<sycl::half, unsigned short>(0xc931)));
    } else {
        uint32_t x0 = cb == 1 ? s0 * 0xCBAC1FEDu : s0 * 89226354u + 64248484u;
        uint32_t x1 = cb == 1 ? s1 * 0xCBAC1FEDu : s1 * 89226354u + 64248484u;
        x0 = (x0 & 0x8fff8fffu) ^ 0x3b603b60u;
        x1 = (x1 & 0x8fff8fffu) ^ 0x3b603b60u;
        const sycl::half2 lo =
            sycl::half2(sycl::bit_cast<sycl::half, unsigned short>(
                            (unsigned short)(x0 & 0xffffu)),
                        sycl::bit_cast<sycl::half, unsigned short>(
                            (unsigned short)(x1 & 0xffffu)));
        const sycl::half2 hi =
            sycl::half2(sycl::bit_cast<sycl::half, unsigned short>(
                            (unsigned short)(x0 >> 16)),
                        sycl::bit_cast<sycl::half, unsigned short>(
                            (unsigned short)(x1 >> 16)));
        return lo + hi;
    }
}
template <int cb>
__dpct_inline__ sycl::float2 decode2(uint32_t s0, uint32_t s1) {
    return decode2h<cb>(s0, s1)
        .template convert<float, sycl::rounding_mode::automatic>();
}

__dpct_inline__ uint32_t h2u(sycl::half2 h) {
    return *reinterpret_cast<uint32_t *>(&h);
}
__dpct_inline__ sycl::half2 u2h(uint32_t u) {
    return *reinterpret_cast<sycl::half2 *>(&u);
}

// (hi : lo) >> s, the low 32 bits
__dpct_inline__ uint32_t fsh(uint32_t lo, uint32_t hi, int s) {
    return (uint32_t) ((((uint64_t) hi << 32) | (uint64_t) lo) >> s);
}

// Step p's state is the 16-bit window ending at bit (p + 1) K of the tile's MSB-first bit stream (w: its 8 K words),
// wrapping around the tile.  A lane decodes its 8 consecutive steps lane * 8 .. + 7 together, the windows of a group
// of steps cut from one or two words with funnel shifts - exllamav3's grouping per K (exl3_dq.cuh: dq8_aligned_*bits,
// dq8, dq4, dq2x2), which the steps' fixed offsets keep within two words.  f[i]: steps 2i, 2i + 1 as FP16 values.
// Step lane * 8 + j sits at tile row (lane & 3) * 2 + (j & 1) + 8 (j >> 1 & 1), column (lane >> 2) + 8 (j >> 2).
template <int cb>
__dpct_inline__ void decode_states(const uint32_t (&s)[8], uint32_t (&f)[4]) {
#pragma unroll
    for (int i = 0; i < 4; ++i) f[i] = h2u(decode2h<cb>(s[2 * i] & 0xffffu, s[2 * i + 1] & 0xffffu));
}
// K = 4: the lane's steps end in its own word b (word lane), the first three reach into a (word lane - 1)
__dpct_inline__ void states4(uint32_t a, uint32_t b, uint32_t (&s)[8]) {
    const uint32_t x = fsh(b, a, 20);
    s[0] = x >> 8;
    s[1] = x >> 4;
    s[2] = x;
    s[3] = b >> 16;
    s[4] = b >> 12;
    s[5] = b >> 8;
    s[6] = b >> 4;
    s[7] = b;
}
// step p .. p + G - 1's windows (G = 4 or 2) from the two words that hold them
template <int K, int G>
__dpct_inline__ void states_group(const uint32_t *w, int p, uint32_t *s) {
    constexpr int NW = 8 * K;
    const int b0 = (p + 257) * K - 16, b2 = b0 + (G - 1) * K + 16;
    const int i0 = b0 >> 5, i2 = (b2 - 1) >> 5, s2 = ((i2 + 1) << 5) - b2;
    const uint32_t a = w[i0 % NW], b = w[i2 % NW];
#pragma unroll
    for (int j = 0; j < G; ++j) s[G - 1 - j] = fsh(b, a, s2 + j * K);
}
template <int K, int cb>
__dpct_inline__ void frag8(const uint32_t *w, int lane, uint32_t (&f)[4]) {
    const int p = lane * 8;
    uint32_t s[8];
    if constexpr (K == 4) {
        states4(w[(lane + 31) & 31], w[lane], s);
    } else if constexpr (K == 1 || K == 2) {
        // the 8 windows overlap in one 32-bit window: steps p + j end 8 - 1 - j K bits apart
        constexpr int NW = 8 * K, per = 32 / (8 * K);   // lanes per word
        const int i1 = lane / per, i0 = (i1 + NW - 1) % NW;
        const uint32_t b = fsh(w[i1], w[i0], (per - 1 - lane % per) * 8 * K);
#pragma unroll
        for (int j = 0; j < 8; ++j) s[j] = b >> ((7 - j) * K);
    } else if constexpr (K == 3) {
        // all 8 windows within two words: two funnel shifts, the rest by shifting
        constexpr int NW = 24;
        const int b1 = (p + 257) * 3, b0 = b1 - 16, b2 = b1 + 21;
        const int i0 = b0 >> 5, i2 = (b2 - 1) >> 5, s2 = ((i2 + 1) << 5) - b2;
        const uint32_t a = w[i0 % NW], b = w[i2 % NW];
        const uint32_t hi = fsh(b, a, s2), lo = fsh(b, a, s2 + 12);
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            s[7 - j] = hi >> (3 * j);
            s[3 - j] = lo >> (3 * j);
        }
    } else if constexpr (K == 7) {
#pragma unroll
        for (int g = 0; g < 4; ++g) states_group<K, 2>(w, p + 2 * g, s + 2 * g);
    } else {   // 5, 6, 8
        states_group<K, 4>(w, p, s);
        states_group<K, 4>(w, p + 4, s + 4);
    }
    decode_states<cb>(s, f);
}

// the lane's 8 steps as F32: v[j] = step lane * 8 + j
template <int K, int cb>
__dpct_inline__ void decode8(const uint32_t *w, int lane, float *v) {
    uint32_t f[4];
    frag8<K, cb>(w, lane, f);
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        const sycl::float2 d =
            u2h(f[i]).template convert<float, sycl::rounding_mode::automatic>();
        v[2 * i] = d.x();
        v[2 * i + 1] = d.y();
    }
}

// ---------------------------------------------------------------- the rotation
// the 128-point transform of the 128 values a warp holds 4 per lane (value lane * 4 + i), over sqrt(128)
__dpct_inline__ void had128(float *v, int lane) {
    const float s0 = v[0] + v[1], d0 = v[0] - v[1], s1 = v[2] + v[3], d1 = v[2] - v[3];
    v[0] = s0 + s1;
    v[1] = d0 + d1;
    v[2] = s0 - s1;
    v[3] = d0 - d1;
#pragma unroll
    for (int m = 1; m < 32; m <<= 1) {
#pragma unroll
        for (int i = 0; i < 4; ++i) {
            /*
            DPCT1108:102: '__shfl_xor_sync' was migrated with the experimental
            feature masked sub_group function which may not be supported by all
            compilers or runtimes. You may need to adjust the code.
            */
            const float o = dpct::experimental::permute_sub_group_by_xor(
                0xffffffffu, sycl::ext::oneapi::this_work_item::get_sub_group(),
                v[i], m);
            v[i] = (lane & m) ? o - v[i] : v[i] + o;
        }
    }
#pragma unroll
    for (int i = 0; i < 4; ++i) v[i] *= kHad;
}

__dpct_inline__ float h2f(uint16_t b) {
    return sycl::vec<sycl::half, 1>(
               sycl::bit_cast<sycl::half, unsigned short>(b))
        .convert<float, sycl::rounding_mode::automatic>()[0];
}

// the activation a warp rotates into its block's FP16 chunks: F32 (f) or the engine's q8_1 blocks (q)
struct Xin {
    const float* f = nullptr;
    const Q81* q = nullptr;
};

// the 128-input chunk of x from k0 on: the lane's 4 values and their suh ...
__dpct_inline__ void chunk_load(const Xin &x, const uint16_t *__restrict__ suh,
                                int k0, int lane, float (&v)[4],
                                float (&s)[4]) {
    const int base = k0 + lane * 4;
    if (x.f != nullptr) {
#pragma unroll
        for (int i = 0; i < 4; ++i) v[i] = x.f[base + i];
    } else {
        const Q81* b = x.q + base / 32;
        const float d = b->ds[0];
#pragma unroll
        for (int i = 0; i < 4; ++i) v[i] = d * (float) b->qs[base % 32 + i];
    }
#pragma unroll
    for (int i = 0; i < 4; ++i) s[i] = h2f(suh[base + i]);
}
// ... and H(x * suh) of the chunk, FP16, into dst[0..127]
__dpct_inline__ void chunk_store(float (&v)[4], const float (&s)[4], int lane,
                                 sycl::half *dst) {
#pragma unroll
    for (int i = 0; i < 4; ++i) v[i] *= s[i];
    had128(v, lane);
    sycl::half2 *d = reinterpret_cast<sycl::half2 *>(dst + lane * 4);
    d[0] = sycl::float2(v[0], v[1])
               .convert<sycl::half, sycl::rounding_mode::rte>();
    d[1] = sycl::float2(v[2], v[3])
               .convert<sycl::half, sycl::rounding_mode::rte>();
}

// ---------------------------------------------------------------- the block GEMV
// A block owns 128 outputs - 8 tiles, warp w the tile column tc0 + w through all of the block's inputs [k0, k1) - for
// nc <= kTok activations at once.  Per 128-input chunk, warp c (< nc) scales activation c's chunk by suh and rotates
// it into FP16 (xb, two buffers), then each warp decodes its tile of each of the chunk's 8 slices and multiplies it
// with every activation: on sm_80 and later as one mma.m16n8k16 a tile (the decoded tile is the A operand - a lane's 8
// steps are its A fragment - and the activations are the B operand's columns, as exllamav3's small-m GEMV and its
// tensor-core GEMM decode tiles straight into fragments), F32 accumulation; elsewhere on the CUDA cores, in F32.  An
// activation's sums do not depend on the other columns, so a window's token gets the token path's bits.  The next
// chunk's tile words are loaded while this one is decoded.  out[c * 128 + i]: activation c's sum for output i.  Ends
// with a __syncthreads (out complete).
constexpr int kTok = 8;   // activations a block takes at once (the MMA's n)

#if !defined(STRATA_USE_HIP) && defined(DPCT_COMPATIBILITY_TEMP) &&            \
    DPCT_COMPATIBILITY_TEMP >= 800
#define EXL3_TC 1
#endif

#if !defined(EXL3_TC)
// a lane's 4 weights of one column times the activation's 4 inputs, added to a (the CUDA-core product)
__device__ __forceinline__ float dot4_acc(float a, const float* v, float x0, float x1, float x2, float x3) {
    return __fadd_rn(a, __fmaf_rn(v[3], x3, __fmaf_rn(v[2], x2, __fmaf_rn(v[1], x1, __fmul_rn(v[0], x0)))));
}
#else
__dpct_inline__ void mma_f32(float (&c)[4], uint32_t a0, uint32_t a1,
                             uint32_t a2, uint32_t a3, uint32_t b0,
                             uint32_t b1) {
#if defined(__SYCL_DEVICE_ONLY__) && defined(__NVPTX__)
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
                 "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, "
                 "{%0,%1,%2,%3};\n"
                 : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
                 : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
#else
    {
        volatile void *d_mat_frag_ct1[4] = {&c[0], &c[1], &c[2], &c[3]};
        sycl::vec<uint32_t, 4> a_mat_frag_ct1(a0, a1, a2, a3);
        sycl::vec<uint32_t, 2> b_mat_frag_ct1(b0, b1);
        sycl::vec<float, 4> c_mat_frag_ct1(c[0], c[1], c[2], c[3]);
        dpct::experimental::matrix::mma<16, 8, 16, sycl::half, float>(
            reinterpret_cast<volatile void **>(d_mat_frag_ct1), &a_mat_frag_ct1,
            &b_mat_frag_ct1, &c_mat_frag_ct1);
    }
#endif
}
#endif

// a tile's words, and a lane's share of them (word lane + 32 r)
template <int K>
struct Words {
    static constexpr int W = 8 * K, R = (W + 31) / 32;
};

// the warp's tile of each of a chunk's 8 slices (slice s at tile0 + s sstride)
template <int K>
__dpct_inline__ void load_tiles(const uint32_t *__restrict__ tile0,
                                size_t sstride, int lane,
                                uint32_t (&w)[8][Words<K>::R]) {
    constexpr int W = Words<K>::W, R = Words<K>::R;
#pragma unroll
    for (int s = 0; s < 8; ++s)
#pragma unroll
        for (int r = 0; r < R; ++r) {
            const int i = lane + 32 * r;
            /*
            DPCT1098:376: The '*' expression is used instead of the __ldg call.
            These two expressions do not provide the exact same functionality.
            Check the generated code for potential precision and/or performance
            issues.
            */
            w[s][r] = (W % 32 == 0 || i < W)
                          ? *(tile0 + (size_t)s * sstride + i)
                          : 0u;
        }
}

// one slice's fragment from the lane's words: K = 4 straight from the registers (the lane's steps end in its own word,
// the first three reach into the previous lane's), other K through the warp's stage (8 K words)
template <int K, int cb>
__dpct_inline__ void slice_frags(const uint32_t (&w)[Words<K>::R], uint32_t *st,
                                 int lane, uint32_t (&f)[4]) {
    if constexpr (K == 4) {
        uint32_t s[8];
        /*
        DPCT1108:103: '__shfl_sync' was migrated with the experimental feature
        masked sub_group function which may not be supported by all compilers or
        runtimes. You may need to adjust the code.
        */
        states4(dpct::experimental::select_from_sub_group(
                    0xffffffffu,
                    sycl::ext::oneapi::this_work_item::get_sub_group(), w[0],
                    (lane + 31) & 31),
                w[0], s);
        decode_states<cb>(s, f);
    } else {
        constexpr int W = Words<K>::W, R = Words<K>::R;
        sycl::group_barrier(sycl::ext::oneapi::this_work_item::get_sub_group());
#pragma unroll
        for (int r = 0; r < R; ++r)
            if (W % 32 == 0 || lane + 32 * r < W) st[lane + 32 * r] = w[r];
        sycl::group_barrier(sycl::ext::oneapi::this_work_item::get_sub_group());
        frag8<K, cb>(st, lane, f);
    }
}

// the blocks an SM keeps of the block_gemv kernels (their registers), as measured on an RTX 3090 and an RTX 5060 Ti:
// the routed experts four at K <= 4 (a token's 8 experts, 256 blocks, are one wave on 82 SMs), two at wider codes;
// mv two, three at K > 4 on sm_80 .. sm_90 (sm_120 spills at the 80 registers that takes).  Before sm_80 (the CUDA
// cores' product, more registers) two; HIP reads the second bound differently and keeps its default.
#if defined(DPCT_COMPATIBILITY_TEMP) && DPCT_COMPATIBILITY_TEMP >= 800
template <int K>
constexpr int kMinBlocksX = K <= 4 ? 4 : 2;
#else
template <int K>
constexpr int kMinBlocksX = 2;
#endif
#if defined(DPCT_COMPATIBILITY_TEMP) && DPCT_COMPATIBILITY_TEMP >= 800 &&      \
    DPCT_COMPATIBILITY_TEMP < 1000
template <int K>
constexpr int kMinBlocksMv = K <= 4 ? 2 : 3;
#else
template <int K>
constexpr int kMinBlocksMv = 2;
#endif
#if defined(STRATA_USE_HIP)
#define EXL3_BOUNDS(n) __launch_bounds__(kThreads)
#else
#define EXL3_BOUNDS(n)
#endif

// the shared memory of a block_gemv block
template <int K>
struct BlockSmem {
    sycl::half xb[2 * kTok * kCols]; // two chunk buffers, kTok activations each
    uint32_t stage[kWarps * 8 * K];    // a tile per warp (K != 4)
    float out[kTok * kCols];
};

template <int K, int cb>
/*
DPCT1110:104: The total declared local variable size in device function
block_gemv exceeds 128 bytes and may cause high register pressure. Consult with
your hardware vendor to find the total register size available and adjust the
code, or use smaller sub-group size to avoid high register pressure.
*/
inline void block_gemv(const uint32_t *__restrict__ tr, int ld, int tc0, int k0,
                       int k1, const Xin &x, const uint16_t *__restrict__ suh,
                       int nc, BlockSmem<K> &sm) {
    auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
    constexpr int W = Words<K>::W, R = Words<K>::R;
    const int lane = item_ct1.get_local_id(2) & 31,
              warp = item_ct1.get_local_id(2) >> 5, g = lane >> 2, t = lane & 3;
    const size_t sstride = (size_t) ld * W;
    const uint32_t* tile = tr + ((size_t) (k0 / 16) * ld + tc0 + warp) * W;
    uint32_t* st = sm.stage + warp * W;
    const bool rot = warp < nc;
    const int nq = (k1 - k0) / kCols;
    uint32_t wc[8][R], wn[8][R];
    float xv[4], xs4[4];
    load_tiles<K>(tile, sstride, lane, wc);
    if (rot) chunk_load(x, suh, k0, lane, xv, xs4);
#if defined(EXL3_TC)
    float c[4] = {0.0f, 0.0f, 0.0f, 0.0f};
#else
    float acc[kTok][2];
#pragma unroll
    for (int cc = 0; cc < kTok; ++cc) acc[cc][0] = acc[cc][1] = 0.0f;
#endif
    for (int q = 0; q < nq; ++q) {
        sycl::half *xq = sm.xb + (q & 1) * (kTok * kCols);
        if (rot) chunk_store(xv, xs4, lane, xq + warp * kCols);
        /*
        DPCT1118:105: SYCL group functions and algorithms must be encountered in
        converged control flow. You may need to adjust the code.
        */
        /*
        DPCT1065:378: Consider replacing sycl::nd_item::barrier() with
        sycl::nd_item::barrier(sycl::access::fence_space::local_space) for
        better performance if there is no access to global memory.
        */
        item_ct1.barrier();
        if (q + 1 < nq) {
            load_tiles<K>(tile + (size_t) (q + 1) * 8 * sstride, sstride, lane, wn);
            if (rot) chunk_load(x, suh, k0 + (q + 1) * kCols, lane, xv, xs4);
        }
#pragma unroll
        for (int s = 0; s < 8; ++s) {
            uint32_t f[4];
            slice_frags<K, cb>(wc[s], st, lane, f);
#if defined(EXL3_TC)
            uint32_t b0 = 0u, b1 = 0u;
            if (g < nc) {
                const sycl::half *xr = xq + g * kCols + s * 16 + 2 * t;
                b0 = *reinterpret_cast<const uint32_t*>(xr);
                b1 = *reinterpret_cast<const uint32_t*>(xr + 8);
            }
            // A: rows (outputs) g / g + 8, columns (inputs) 2t.. / 2t + 8..: steps 0,1 / 4,5 / 2,3 / 6,7
            mma_f32(c, f[0], f[2], f[1], f[3], b0, b1);
#else
            float v[8];
#pragma unroll
            for (int i = 0; i < 4; ++i) {
                const float2 d = __half22float2(u2h(f[i]));
                v[2 * i] = d.x;
                v[2 * i + 1] = d.y;
            }
#pragma unroll
            for (int cc = 0; cc < kTok; ++cc)
                if (cc < nc) {
                    const __half* xr = xq + cc * kCols + s * 16 + 2 * t;
                    const float x0 = __half2float(xr[0]), x1 = __half2float(xr[1]);
                    const float x2 = __half2float(xr[8]), x3 = __half2float(xr[9]);
                    acc[cc][0] = dot4_acc(acc[cc][0], v, x0, x1, x2, x3);
                    acc[cc][1] = dot4_acc(acc[cc][1], v + 4, x0, x1, x2, x3);
                }
#endif
        }
#pragma unroll
        for (int s = 0; s < 8; ++s)
#pragma unroll
            for (int r = 0; r < R; ++r) wc[s][r] = wn[s][r];
    }
    const int o = warp * 16 + g;
#if defined(EXL3_TC)
    // c: the tile's outputs g (c[0], c[1]) and g + 8 (c[2], c[3]) for activations 2t (c[0], c[2]) and 2t + 1
    if (2 * t < nc) {
        sm.out[(2 * t) * kCols + o] = c[0];
        sm.out[(2 * t) * kCols + o + 8] = c[2];
    }
    if (2 * t + 1 < nc) {
        sm.out[(2 * t + 1) * kCols + o] = c[1];
        sm.out[(2 * t + 1) * kCols + o + 8] = c[3];
    }
#else
    // the 4 lanes of one g hold parts of the same two outputs
#pragma unroll
    for (int cc = 0; cc < kTok; ++cc)
        if (cc < nc) {
#pragma unroll
            for (int h = 0; h < 2; ++h) {
                acc[cc][h] += __shfl_xor_sync(0xffffffffu, acc[cc][h], 1);
                acc[cc][h] += __shfl_xor_sync(0xffffffffu, acc[cc][h], 2);
            }
            if (t == 0) {
                sm.out[cc * kCols + o] = acc[cc][0];
                sm.out[cc * kCols + o + 8] = acc[cc][1];
            }
        }
#endif
    /*
    DPCT1065:377: Consider replacing sycl::nd_item::barrier() with
    sycl::nd_item::barrier(sycl::access::fence_space::local_space) for better
    performance if there is no access to global memory.
    */
    item_ct1.barrier();
}

// one warp: y[c] = alpha * (H(red) * svh)[c] + bias[c] for the block's 128 outputs
__dpct_inline__ void epilogue(const float *red, const uint16_t *svh,
                              float alpha, const float *bias, float *y) {
    const int lane =
        sycl::ext::oneapi::this_work_item::get_nd_item<3>().get_local_id(2) &
        31;
    float v[4];
#pragma unroll
    for (int i = 0; i < 4; ++i) v[i] = red[lane * 4 + i];
    had128(v, lane);
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        const int c = lane * 4 + i;
        /*
        DPCT1013:379: The rounding mode could not be specified and the generated
        code may have different accuracy than the original code. Verify the
        correctness. SYCL math built-in function rounding mode is aligned with
        OpenCL C 1.2 standard.
        */
        y[c] = sycl::fma(alpha * v[i], h2f(svh[c]),
                         bias != nullptr ? bias[c] : 0.0f);
    }
}

// ---------------------------------------------------------------- mv
struct KJob {
    const uint32_t* tr;
    const uint16_t* suh;
    const uint16_t* svh;
    const float* xf;
    const Q81* xq;
    float* y;
    const float* bias;
    float* part;       // nc x S x n partial sums (S > 1)
    unsigned* cnt;     // n / 128 counters, zero between launches
    float alpha;
    int k, n, ld, S, kr;
    int nc;            // the activations (1..kTok), xf / xq / y advanced by these per activation
    size_t xf_cs, xq_cs, y_cs;   // (floats, bytes, floats)
};
struct KBatch {
    KJob j[kMaxMvJobs];
    int blk_end[kMaxMvJobs];
    int n;
    const int* skip;   // set_skip's flag: non-zero = a pipelined window's token past a wrong draft (nullptr: none)
};

// a launch's blocks: its jobs' (output block, input split) pairs, each for all of the job's nc activations at once
// (block_gemv).  S > 1: the partial sums go to the scratch and the last block of an output block adds them up in split
// order (deterministic) and runs the epilogue
template <int K, int cb>
__dpct_inline__ void EXL3_BOUNDS(kMinBlocksMv<K>) mv_kernel(const KBatch b) {
    auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
auto &sm = *sycl::ext::oneapi::group_local_memory_for_overwrite<BlockSmem<K>>(
    sycl::ext::oneapi::this_work_item::get_work_group<3>());
    auto &last = *sycl::ext::oneapi::group_local_memory_for_overwrite<int>(
        sycl::ext::oneapi::this_work_item::get_work_group<3>());
    int ji = 0;
    while ((int)item_ct1.get_group(2) >= b.blk_end[ji])++ ji;
    const KJob& J = b.j[ji];
    const int local =
        (int)item_ct1.get_group(2) - (ji > 0 ? b.blk_end[ji - 1] : 0);
    const int g = local / J.S, sp = local % J.S;
    const int k0 = sp * J.kr, k1 = sycl::min(J.k, k0 + J.kr);
    const int warp = item_ct1.get_local_id(2) >> 5;
    // a skipped token (set_skip): no work.  The flag changes only between launches (skip_gate, on the same stream), so
    // every block of the launch returns and no split counter moves
    if (b.skip != nullptr && *b.skip) return;
    Xin x;
    if (warp < J.nc) {
        if (J.xf != nullptr) x.f = J.xf + warp * J.xf_cs;
        else x.q = (const Q81*) ((const uint8_t*) J.xq + warp * J.xq_cs);
    }
    block_gemv<K, cb>(J.tr, J.ld, g * 8, k0, k1, x, J.suh, J.nc, sm);
    const float* bias = J.bias != nullptr ? J.bias + g * kCols : nullptr;
    if (J.S == 1) {
        if (warp < J.nc)
            epilogue(sm.out + warp * kCols, J.svh + g * kCols, J.alpha, bias, J.y + warp * J.y_cs + g * kCols);
        return;
    }
    if (item_ct1.get_local_id(2) < kCols)
#pragma unroll
        for (int c = 0; c < J.nc; ++c)
            J.part[((size_t)c * J.S + sp) * J.n + g * kCols +
                   item_ct1.get_local_id(2)] =
                sm.out[c * kCols + item_ct1.get_local_id(2)];
    /*
    DPCT1078:106: Consider replacing memory_order::acq_rel with
    memory_order::seq_cst for correctness if strong memory order restrictions
    are needed.
    */
    sycl::atomic_fence(sycl::memory_order::acq_rel, sycl::memory_scope::device);
    /*
    DPCT1065:380: Consider replacing sycl::nd_item::barrier() with
    sycl::nd_item::barrier(sycl::access::fence_space::local_space) for better
    performance if there is no access to global memory.
    */
    item_ct1.barrier();
    if (item_ct1.get_local_id(2) == 0) last =
        dpct::atomic_fetch_add<sycl::access::address_space::generic_space>(
            &J.cnt[g], 1u) == (unsigned)(J.S - 1);
    /*
    DPCT1065:381: Consider replacing sycl::nd_item::barrier() with
    sycl::nd_item::barrier(sycl::access::fence_space::local_space) for better
    performance if there is no access to global memory.
    */
    item_ct1.barrier();
    if (!last) return;
    /*
    DPCT1078:107: Consider replacing memory_order::acq_rel with
    memory_order::seq_cst for correctness if strong memory order restrictions
    are needed.
    */
    sycl::atomic_fence(sycl::memory_order::acq_rel, sycl::memory_scope::device);
    if (item_ct1.get_local_id(2) < kCols)
        for (int c = 0; c < J.nc; ++c) {
            float s = 0.0f;
#pragma unroll
            for (int i = 0; i < J.S; ++i)
                s += *(volatile const float *)&J
                          .part[((size_t)c * J.S + i) * J.n + g * kCols +
                                item_ct1.get_local_id(2)];
            sm.out[c * kCols + item_ct1.get_local_id(2)] = s;
        }
    /*
    DPCT1065:382: Consider replacing sycl::nd_item::barrier() with
    sycl::nd_item::barrier(sycl::access::fence_space::local_space) for better
    performance if there is no access to global memory.
    */
    item_ct1.barrier();
    if (warp < J.nc)
        epilogue(sm.out + warp * kCols, J.svh + g * kCols, J.alpha, bias, J.y + warp * J.y_cs + g * kCols);
    if (item_ct1.get_local_id(2) == 0) J.cnt[g] = 0u;
}

// ---------------------------------------------------------------- the routed experts
struct KMoe {
    const unsigned long long* plan_ptr;
    const float* plan_w;
    const float* cpu_part;
    const int* cpu_flag;
    const float* x;
    float* h;            // [k][n_ff]: the gate's values, then the swiglu
    float* u;            // [k][n_ff]: the up's values
    const float* sh_out;
    float* out;
    float* part;         // [k][n_embd]
    unsigned* cnt;       // n_embd / 128
    unsigned* cnt_gu;    // k x n_ff / 128: the gate / up pairs
    size_t suh[3], svh[3], tr[3];
    int k, n_embd, n_ff;
    float limit;
    const int* skip;     // set_skip's flag (nullptr: none)
};

// the pieces the token's experts (moe_*_kernel) and a window's grouped experts (win_*_kernel) share, so each window
// token gets the token path's bits: one warp's output block of 128 (H, then svh) ...
__dpct_inline__ void block_out(const float *red, const uint16_t *svh,
                               float *dst) {
    const int lane =
        sycl::ext::oneapi::this_work_item::get_nd_item<3>().get_local_id(2) &
        31;
    float v[4];
#pragma unroll
    for (int i = 0; i < 4; ++i) v[i] = red[lane * 4 + i];
    had128(v, lane);
#pragma unroll
    /*
    DPCT1013:383: The rounding mode could not be specified and the generated
    code may have different accuracy than the original code. Verify the
    correctness. SYCL math built-in function rounding mode is aligned with
    OpenCL C 1.2 standard.
    */
    for (int i = 0; i < 4; ++i)
        dst[lane * 4 + i] = v[i] * h2f(svh[lane * 4 + i]);
}
// ... the clamped swiglu ...
__dpct_inline__ float swiglu(float gv, float uv, float limit) {
    if (limit > 0.0f) {
        gv = sycl::fmin(gv, limit);
        uv = sycl::fmin(sycl::fmax(uv, -limit), limit);
    }
    /*
    DPCT1013:384: The rounding mode could not be specified and the generated
    code may have different accuracy than the original code. Verify the
    correctness. SYCL math built-in function rounding mode is aligned with
    OpenCL C 1.2 standard.
    */
    return gv / (1.0f + sycl::native::exp(-gv)) * uv;
}
// ... the second block of a (gate, up) pair to finish (its counter back at zero for the next launch) ...
__dpct_inline__ bool pair_last(unsigned *cnt, int &last) {
    /*
    DPCT1078:108: Consider replacing memory_order::acq_rel with
    memory_order::seq_cst for correctness if strong memory order restrictions
    are needed.
    */
    auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
    sycl::atomic_fence(sycl::memory_order::acq_rel, sycl::memory_scope::device);
    /*
    DPCT1065:385: Consider replacing sycl::nd_item::barrier() with
    sycl::nd_item::barrier(sycl::access::fence_space::local_space) for better
    performance if there is no access to global memory.
    */
    item_ct1.barrier();
    if (item_ct1.get_local_id(2) == 0) last =
        dpct::atomic_fetch_add<sycl::access::address_space::generic_space>(
            cnt, 1u) == 1u;
    /*
    DPCT1065:386: Consider replacing sycl::nd_item::barrier() with
    sycl::nd_item::barrier(sycl::access::fence_space::local_space) for better
    performance if there is no access to global memory.
    */
    item_ct1.barrier();
    if (!last) return false;
    /*
    DPCT1078:109: Consider replacing memory_order::acq_rel with
    memory_order::seq_cst for correctness if strong memory order restrictions
    are needed.
    */
    sycl::atomic_fence(sycl::memory_order::acq_rel, sycl::memory_scope::device);
    if (item_ct1.get_local_id(2) == 0) *cnt = 0u;
    return true;
}
// ... and one output's sum: the experts' rows in plan order (p[i]: expert i's, 0 for one not on the device), the CPU
// lane's part when flagged (cpu_part: null when not), the shared expert's row (sh: 0 without one)
__dpct_inline__ float combine(const float *plan_w, const float *p, int k,
                              const float *cpu_part, float sh) {
    float m = 0.0f;
    /*
DPCT1013:387: The rounding mode could not be specified and the generated code
may have different accuracy than the original code. Verify the correctness. SYCL
math built-in function rounding mode is aligned with OpenCL C 1.2 standard.
*/
#pragma unroll
    for (int i = 0; i < k; ++i) m = sycl::fma(plan_w[i], p[i], m);
    /*
    DPCT1013:388: The rounding mode could not be specified and the generated
    code may have different accuracy than the original code. Verify the
    correctness. SYCL math built-in function rounding mode is aligned with
    OpenCL C 1.2 standard.
    */
    if (cpu_part != nullptr) m = m + *cpu_part;
    /*
    DPCT1013:389: The rounding mode could not be specified and the generated
    code may have different accuracy than the original code. Verify the
    correctness. SYCL math built-in function rounding mode is aligned with
    OpenCL C 1.2 standard.
    */
    return m + sh;
}

// grid (n_ff / 128, k, 2): expert blockIdx.y's gate (z 0, into h) or up (z 1, into u) for 128 h values; the pair's
// second block to finish turns them into the clamped swiglu, in h
template <int K, int cb>
__dpct_inline__ void EXL3_BOUNDS(kMinBlocksX<K>)
    moe_gate_up_kernel(const KMoe a) {
    auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
auto &sm = *sycl::ext::oneapi::group_local_memory_for_overwrite<BlockSmem<K>>(
    sycl::ext::oneapi::this_work_item::get_work_group<3>());
    auto &last = *sycl::ext::oneapi::group_local_memory_for_overwrite<int>(
        sycl::ext::oneapi::this_work_item::get_work_group<3>());
    const int ei = item_ct1.get_group(1), g = item_ct1.get_group(2),
              r = item_ct1.get_group(0);
    const unsigned long long ptr = a.plan_ptr[ei];
    if (ptr == 0ull) return;
    if (a.skip != nullptr && *a.skip) return;   // a skipped token (set_skip, as mv_kernel)
    const uint8_t* blob = (const uint8_t*) ptr;
    const int warp = item_ct1.get_local_id(2) >> 5;
    Xin x;
    x.f = a.x;
    block_gemv<K, cb>((const uint32_t*) (blob + a.tr[r]), a.n_ff / 16, g * 8, 0, a.n_embd, x,
                      (const uint16_t*) (blob + a.suh[r]), 1, sm);
    const size_t row = (size_t) ei * a.n_ff + g * kCols;
    if (warp == 0) block_out(sm.out, (const uint16_t*) (blob + a.svh[r]) + g * kCols, (r == 0 ? a.h : a.u) + row);
    if (!pair_last(a.cnt_gu + ei * (a.n_ff / kCols) + g, last)) return;
    if (item_ct1.get_local_id(2) < kCols)
        a.h[row + item_ct1.get_local_id(2)] = swiglu(
            *(volatile const float *)&a.h[row + item_ct1.get_local_id(2)],
            *(volatile const float *)&a.u[row + item_ct1.get_local_id(2)],
            a.limit);
}

// grid (n_embd / 128, k): expert blockIdx.y's down rows for 128 outputs; the last block of each output block adds
// the experts up in plan order (as the engine's moe_down), then the CPU lane's part and the shared expert
template <int K, int cb>
__dpct_inline__ void EXL3_BOUNDS(kMinBlocksX<K>) moe_down_kernel(const KMoe a) {
    auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
auto &sm = *sycl::ext::oneapi::group_local_memory_for_overwrite<BlockSmem<K>>(
    sycl::ext::oneapi::this_work_item::get_work_group<3>());
    auto &last = *sycl::ext::oneapi::group_local_memory_for_overwrite<int>(
        sycl::ext::oneapi::this_work_item::get_work_group<3>());
    const int ei = item_ct1.get_group(1), g = item_ct1.get_group(2);
    const unsigned long long ptr = a.plan_ptr[ei];
    const int warp = item_ct1.get_local_id(2) >> 5;
    float* part = a.part + (size_t) ei * a.n_embd + g * kCols;
    if (a.skip != nullptr && *a.skip) return;   // a skipped token (set_skip, as mv_kernel)
    if (ptr != 0ull) {
        const uint8_t* blob = (const uint8_t*) ptr;
        Xin x;
        x.f = a.h + (size_t) ei * a.n_ff;
        block_gemv<K, cb>((const uint32_t*) (blob + a.tr[2]), a.n_embd / 16, g * 8, 0, a.n_ff, x,
                          (const uint16_t*) (blob + a.suh[2]), 1, sm);
        if (warp == 0) block_out(sm.out, (const uint16_t*) (blob + a.svh[2]) + g * kCols, part);
    } else if (item_ct1.get_local_id(2) < kCols) {
        part[item_ct1.get_local_id(2)] = 0.0f;
    }
    /*
    DPCT1078:110: Consider replacing memory_order::acq_rel with
    memory_order::seq_cst for correctness if strong memory order restrictions
    are needed.
    */
    sycl::atomic_fence(sycl::memory_order::acq_rel, sycl::memory_scope::device);
    /*
    DPCT1065:390: Consider replacing sycl::nd_item::barrier() with
    sycl::nd_item::barrier(sycl::access::fence_space::local_space) for better
    performance if there is no access to global memory.
    */
    item_ct1.barrier();
    if (item_ct1.get_local_id(2) == 0) last =
        dpct::atomic_fetch_add<sycl::access::address_space::generic_space>(
            &a.cnt[g], 1u) == (unsigned)(a.k - 1);
    /*
    DPCT1065:391: Consider replacing sycl::nd_item::barrier() with
    sycl::nd_item::barrier(sycl::access::fence_space::local_space) for better
    performance if there is no access to global memory.
    */
    item_ct1.barrier();
    if (!last) return;
    /*
    DPCT1078:111: Consider replacing memory_order::acq_rel with
    memory_order::seq_cst for correctness if strong memory order restrictions
    are needed.
    */
    sycl::atomic_fence(sycl::memory_order::acq_rel, sycl::memory_scope::device);
    if (item_ct1.get_local_id(2) < kCols) {
        const int r = g * kCols + item_ct1.get_local_id(2);
        float p[8];
#pragma unroll
        for (int i = 0; i < a.k; ++i)
            p[i] = *(volatile const float *)&a.part[(size_t)i * a.n_embd + r];
        a.out[r] = combine(a.plan_w, p, a.k, a.cpu_flag != nullptr && *a.cpu_flag ? a.cpu_part + r : nullptr,
                           a.sh_out != nullptr ? a.sh_out[r] : 0.0f);
    }
    if (item_ct1.get_local_id(2) == 0) a.cnt[g] = 0u;
}

// ---------------------------------------------------------------- a verify window's routed experts
// gf::moe_win_experts' grouping (group g: one expert, its blob at grp_ptr[g]; its entries e = t * 8 + i at
// grp_ent[grp_start[g] ..)), each group's weights read and decoded once for up to kTok of its tokens at a time (as
// exllamav3's fused MoE runs a run of slots that picked one expert as the rows of one MMA tile), with the token
// path's arithmetic per entry: every token's output is moe_gate_up + moe_down's for it alone, bit for bit
struct KWin {
    const unsigned long long* grp_ptr;
    const int *grp_start, *grp_ent, *n_groups;
    const unsigned long long* plan_ptr;
    const float* plan_w;
    const float* cpu_part;
    size_t cpu_stride;
    const int* cpu_flag;
    const float* x;
    size_t x_stride;
    float* h;            // [T * 8][n_ff]: the gate's values, then the swiglu
    float* u;            // [T * 8][n_ff]: the up's values
    float* part;         // [T * 8][n_embd]
    const float* sh_out; // [T][n_embd]
    float* out;
    size_t out_stride;
    unsigned* cnt_gu;    // groups x n_ff / 128: the gate / up pairs
    size_t suh[3], svh[3], tr[3];
    int k, n_embd, n_ff;
    float limit;
};

// grid (n_ff / 128, groups at most, 2): group blockIdx.y's gate (z 0, into h) or up (z 1, into u) for 128 h values of
// each of its entries, kTok entries a pass; the pair's second block to finish turns them into the clamped swiglu
template <int K, int cb>
__dpct_inline__ void EXL3_BOUNDS(kMinBlocksX<K>) win_gu_kernel(const KWin a) {
    auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
auto &sm = *sycl::ext::oneapi::group_local_memory_for_overwrite<BlockSmem<K>>(
    sycl::ext::oneapi::this_work_item::get_work_group<3>());
    auto &last = *sycl::ext::oneapi::group_local_memory_for_overwrite<int>(
        sycl::ext::oneapi::this_work_item::get_work_group<3>());
    const int grp = item_ct1.get_group(1), g = item_ct1.get_group(2),
              r = item_ct1.get_group(0);
    if (grp >= *a.n_groups) return;
    const uint8_t* blob = (const uint8_t*) a.grp_ptr[grp];
    const int warp = item_ct1.get_local_id(2) >> 5;
    const int m0 = a.grp_start[grp], m1 = a.grp_start[grp + 1];
    float* dst = r == 0 ? a.h : a.u;
    for (int mm = m0; mm < m1; mm += kTok) {
        const int nc = sycl::min(kTok, m1 - mm);
        Xin x;
        if (warp < nc) x.f = a.x + (size_t) (a.grp_ent[mm + warp] >> 3) * a.x_stride;
        block_gemv<K, cb>((const uint32_t*) (blob + a.tr[r]), a.n_ff / 16, g * 8, 0, a.n_embd, x,
                          (const uint16_t*) (blob + a.suh[r]), nc, sm);
        if (warp < nc)
            block_out(sm.out + warp * kCols, (const uint16_t*) (blob + a.svh[r]) + g * kCols,
                      dst + (size_t) a.grp_ent[mm + warp] * a.n_ff + g * kCols);
        /*
        DPCT1118:112: SYCL group functions and algorithms must be encountered in
        converged control flow. You may need to adjust the code.
        */
        /*
        DPCT1065:392: Consider replacing sycl::nd_item::barrier() with
        sycl::nd_item::barrier(sycl::access::fence_space::local_space) for
        better performance if there is no access to global memory.
        */
        item_ct1.barrier();
    }
    if (!pair_last(a.cnt_gu + grp * (a.n_ff / kCols) + g, last)) return;
    if (item_ct1.get_local_id(2) < kCols)
        for (int m = m0; m < m1; ++m) {
            const size_t row = (size_t)a.grp_ent[m] * a.n_ff + g * kCols +
                               item_ct1.get_local_id(2);
            a.h[row] = swiglu(*(volatile const float*) &a.h[row], *(volatile const float*) &a.u[row], a.limit);
        }
}

// grid (n_embd / 128, groups at most): group blockIdx.y's down rows for 128 outputs of each entry into its part rows
template <int K, int cb>
__dpct_inline__ void EXL3_BOUNDS(kMinBlocksX<K>) win_down_kernel(const KWin a) {
    auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
auto &sm = *sycl::ext::oneapi::group_local_memory_for_overwrite<BlockSmem<K>>(
    sycl::ext::oneapi::this_work_item::get_work_group<3>());
    const int grp = item_ct1.get_group(1), g = item_ct1.get_group(2);
    if (grp >= *a.n_groups) return;
    const uint8_t* blob = (const uint8_t*) a.grp_ptr[grp];
    const int warp = item_ct1.get_local_id(2) >> 5;
    const int m0 = a.grp_start[grp], m1 = a.grp_start[grp + 1];
    for (int mm = m0; mm < m1; mm += kTok) {
        const int nc = sycl::min(kTok, m1 - mm);
        Xin x;
        if (warp < nc) x.f = a.h + (size_t) a.grp_ent[mm + warp] * a.n_ff;
        block_gemv<K, cb>((const uint32_t*) (blob + a.tr[2]), a.n_embd / 16, g * 8, 0, a.n_ff, x,
                          (const uint16_t*) (blob + a.suh[2]), nc, sm);
        if (warp < nc)
            block_out(sm.out + warp * kCols, (const uint16_t*) (blob + a.svh[2]) + g * kCols,
                      a.part + (size_t) a.grp_ent[mm + warp] * a.n_embd + g * kCols);
        /*
        DPCT1118:113: SYCL group functions and algorithms must be encountered in
        converged control flow. You may need to adjust the code.
        */
        /*
        DPCT1065:393: Consider replacing sycl::nd_item::barrier() with
        sycl::nd_item::barrier(sycl::access::fence_space::local_space) for
        better performance if there is no access to global memory.
        */
        item_ct1.barrier();
    }
}

// grid (n_embd / 128, T): each token's outputs, moe_down_kernel's combine (an entry off the device adds 0)
__dpct_inline__ void win_combine_kernel(const KWin a) {
    auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
    const int t = item_ct1.get_group(1),
              r = item_ct1.get_group(2) * kCols + item_ct1.get_local_id(2);
    float p[8];
    for (int i = 0; i < a.k; ++i) {
        const int e = t * 8 + i;
        p[i] = a.plan_ptr[e] != 0ull ? a.part[(size_t) e * a.n_embd + r] : 0.0f;
    }
    a.out[(size_t) t * a.out_stride + r] =
        combine(a.plan_w + t * 8, p, a.k,
                a.cpu_flag != nullptr && a.cpu_flag[t] ? a.cpu_part + (size_t) t * a.cpu_stride + r : nullptr,
                a.sh_out != nullptr ? a.sh_out[(size_t) t * a.n_embd + r] : 0.0f);
}

__dpct_inline__ void win_wait_kernel(const unsigned *p, unsigned v) {
    if (sycl::ext::oneapi::this_work_item::get_nd_item<3>().get_local_id(2) !=
        0) return;
    while (*(volatile const unsigned*) p != v) {
#if defined(DPCT_COMPATIBILITY_TEMP) && DPCT_COMPATIBILITY_TEMP >= 700
        /*
        DPCT1008:394: __nanosleep function is not defined in SYCL. This is a
        hardware-specific feature. Consult with your hardware vendor to find a
        replacement.
        */
        /* CUDA pause hint; polling order is unchanged. */
#endif
    }
    /*
    DPCT1078:114: Consider replacing memory_order::acq_rel with
    memory_order::seq_cst for correctness if strong memory order restrictions
    are needed.
    */
    sycl::atomic_fence(sycl::memory_order::acq_rel, sycl::memory_scope::system);
}

// ---------------------------------------------------------------- reconstruction
// grid (n / 128, k / 128): the 128 x 128 block of W = diag(suh) H W' H diag(svh), written transposed (dst[c][r])
template <int K, int cb>
/*
DPCT1110:115: The total declared local variable size in device function
recon_rows_kernel exceeds 128 bytes and may cause high register pressure.
Consult with your hardware vendor to find the total register size available and
adjust the code, or use smaller sub-group size to avoid high register pressure.
*/
__dpct_inline__ void recon_rows_kernel(const uint32_t *__restrict__ tr, int ld,
                                       const uint16_t *__restrict__ suh,
                                       const uint16_t *__restrict__ svh, int k,
                                       uint16_t *dst, bool bf16) {
    auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
    constexpr int W = 8 * K;
    constexpr int P = kCols + 2;   // padded row (halves)
    auto &tile = *sycl::ext::oneapi::group_local_memory_for_overwrite<
        sycl::half[kCols * P]>(
        sycl::ext::oneapi::this_work_item::get_work_group<3>());
    const int lane = item_ct1.get_local_id(2) & 31,
              warp = item_ct1.get_local_id(2) >> 5;
    const int nb = item_ct1.get_group(2), kb = item_ct1.get_group(1);
    // decode: warp w takes the 16-row slices w (8 tiles each)
    for (int ks = warp; ks < 8; ks += kWarps) {
        const uint32_t* src = tr + ((size_t) (kb * 8 + ks) * ld + nb * 8) * W;
        for (int t = 0; t < 8; ++t) {
            float v[8];
            decode8<K, cb>(src + t * W, lane, v);
#pragma unroll
            for (int j = 0; j < 8; ++j) {
                const int r = ks * 16 + (lane & 3) * 2 + (j & 1) + 8 * ((j >> 1) & 1);
                const int c = t * 16 + (lane >> 2) + 8 * (j >> 2);
                tile[r * P + c] =
                    sycl::vec<float, 1>(v[j])
                        .convert<sycl::half, sycl::rounding_mode::rte>()[0];
            }
        }
    }
    item_ct1.barrier(sycl::access::fence_space::local_space);
    // H along the outputs (each row of 128 is one block)
    for (int r = warp; r < kCols; r += kWarps) {
        float v[4];
#pragma unroll
        for (int i = 0; i < 4; ++i)
            v[i] = sycl::vec<sycl::half, 1>(tile[r * P + lane * 4 + i])
                       .convert<float, sycl::rounding_mode::automatic>()[0];
        had128(v, lane);
#pragma unroll
        for (int i = 0; i < 4; ++i) tile[r * P + lane * 4 + i] =
            sycl::vec<float, 1>(v[i])
                .convert<sycl::half, sycl::rounding_mode::rte>()[0];
    }
    item_ct1.barrier(sycl::access::fence_space::local_space);
    // H along the inputs, the scales, the transposed store
    for (int c = warp; c < kCols; c += kWarps) {
        float v[4];
#pragma unroll
        for (int i = 0; i < 4; ++i)
            v[i] = sycl::vec<sycl::half, 1>(tile[(lane * 4 + i) * P + c])
                       .convert<float, sycl::rounding_mode::automatic>()[0];
        had128(v, lane);
        const float sv = h2f(svh[nb * kCols + c]);
        uint16_t* d = dst + (size_t) (nb * kCols + c) * k + kb * kCols + lane * 4;
#pragma unroll
        for (int i = 0; i < 4; ++i) {
            const float w = v[i] * sv * h2f(suh[kb * kCols + lane * 4 + i]);
            if (bf16) {
                const uint32_t u = sycl::bit_cast<unsigned int>(w);
                d[i] = (uint16_t) ((u + 0x7fffu + ((u >> 16) & 1u)) >> 16);   // round to nearest even
            } else {
                d[i] = sycl::bit_cast<unsigned short, sycl::half>(
                    sycl::vec<float, 1>(w)
                        .convert<sycl::half, sycl::rounding_mode::rte>()[0]);
            }
        }
    }
}

// grid (n / 128, k / 16): W' row-major, one tile per warp
template <int K, int cb>
__dpct_inline__ void recon_inner_kernel(const uint32_t *__restrict__ tr, int ld,
                                        int n, uint16_t *dst) {
    auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
    constexpr int W = 8 * K;
    const int lane = item_ct1.get_local_id(2) & 31,
              t = item_ct1.get_local_id(2) >> 5;
    const int tc = item_ct1.get_group(2) * 8 + t, ks = item_ct1.get_group(1);
    float v[8];
    decode8<K, cb>(tr + ((size_t) ks * ld + tc) * W, lane, v);
#pragma unroll
    for (int j = 0; j < 8; ++j) {
        const int r = ks * 16 + (lane & 3) * 2 + (j & 1) + 8 * ((j >> 1) & 1);
        const int c = tc * 16 + (lane >> 2) + 8 * (j >> 2);
        dst[(size_t)r * n + c] = sycl::bit_cast<unsigned short, sycl::half>(
            sycl::vec<float, 1>(v[j])
                .convert<sycl::half, sycl::rounding_mode::rte>()[0]);
    }
}

// grid (n / 128, k / 128): W' of the 128 x 128 block, transposed (dst[c][r]) - the decoded values alone.  Each warp
// decodes its slice's 8 tiles into the block's tile (half2 pairs of rows, conflict-free), then writes 256-byte rows.
template <int K, int cb>
__dpct_inline__ void inner_rows_kernel(const uint32_t *__restrict__ tr, int ld,
                                       int k, uint16_t *__restrict__ dst) {
    auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
    constexpr int W = 8 * K;
    constexpr int P = kCols + 8;   // a column's 128 rows, padded (halves; 16-byte rows)
    auto &tile = *sycl::ext::oneapi::group_local_memory_for_overwrite<
        sycl::half[kCols * P]>(
        sycl::ext::oneapi::this_work_item::get_work_group<3>()); // tile[c * P +
                                                                 // r]
    const int lane = item_ct1.get_local_id(2) & 31,
              warp = item_ct1.get_local_id(2) >> 5, g = lane >> 2, t = lane & 3;
    const int nb = item_ct1.get_group(2), kb = item_ct1.get_group(1);
    for (int ks = warp; ks < 8; ks += kWarps) {
        const uint32_t* src = tr + ((size_t) (kb * 8 + ks) * ld + nb * 8) * W;
        const int r = ks * 16 + 2 * t;
#pragma unroll
        for (int tc = 0; tc < 8; ++tc) {
            uint32_t f[4];
            frag8<K, cb>(src + tc * W, lane, f);
            const int c = tc * 16 + g;
            *reinterpret_cast<uint32_t*>(&tile[c * P + r]) = f[0];
            *reinterpret_cast<uint32_t*>(&tile[c * P + r + 8]) = f[1];
            *reinterpret_cast<uint32_t*>(&tile[(c + 8) * P + r]) = f[2];
            *reinterpret_cast<uint32_t*>(&tile[(c + 8) * P + r + 8]) = f[3];
        }
    }
    item_ct1.barrier(sycl::access::fence_space::local_space);
#pragma unroll
    for (int c = warp; c < kCols; c += kWarps)
        *reinterpret_cast<sycl::uint2 *>(dst + (size_t)(nb * kCols + c) * k +
                                         kb * kCols + lane * 4) =
            *reinterpret_cast<const sycl::uint2 *>(&tile[c * P + lane * 4]);
}

// one warp per 128 values of a row: the rotated FP16 input of an EXL3 matrix (rows_rot_f16) ...
__dpct_inline__ void rows_rot_f16_kernel(const float *__restrict__ x, int ldx,
                                         const int *__restrict__ idx, int nrows,
                                         int ncols,
                                         const uint16_t *__restrict__ suh,
                                         uint16_t *out) {
    auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
    const int lane = item_ct1.get_local_id(2) & 31, nb = ncols / kCols;
    const int w =
        item_ct1.get_group(2) * kWarps + (item_ct1.get_local_id(2) >> 5);
    if (w >= nrows * nb) return;
    const int r = w / nb, b = w % nb;
    Xin xin;
    xin.f = x + (size_t) (idx != nullptr ? idx[r] : r) * ldx;
    float v[4], sc[4];
    chunk_load(xin, suh, b * kCols, lane, v, sc);
    chunk_store(v, sc, lane,
                reinterpret_cast<sycl::half *>(out) + (size_t)r * ncols +
                    b * kCols);
}
// ... and its output rotation in place (block_out's arithmetic)
__dpct_inline__ void rows_rot_out_kernel(float *y, int ldy, int nrows,
                                         int ncols,
                                         const uint16_t *__restrict__ svh) {
    auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
    const int lane = item_ct1.get_local_id(2) & 31, nb = ncols / kCols;
    const int w =
        item_ct1.get_group(2) * kWarps + (item_ct1.get_local_id(2) >> 5);
    if (w >= nrows * nb) return;
    const int r = w / nb, b = w % nb;
    float* p = y + (size_t) r * ldy + b * kCols;
    float v[4];
#pragma unroll
    for (int i = 0; i < 4; ++i) v[i] = p[lane * 4 + i];
    had128(v, lane);
#pragma unroll
    /*
    DPCT1013:395: The rounding mode could not be specified and the generated
    code may have different accuracy than the original code. Verify the
    correctness. SYCL math built-in function rounding mode is aligned with
    OpenCL C 1.2 standard.
    */
    for (int i = 0; i < 4; ++i)
        p[lane * 4 + i] = v[i] * h2f(svh[b * kCols + lane * 4 + i]);
}

__dpct_inline__ void rows_f16_kernel(const float *__restrict__ x, int ldx,
                                     const int *__restrict__ idx, int ncols,
                                     uint16_t *__restrict__ out) {
    auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
    const int r = item_ct1.get_group(1);
    const float* src = x + (size_t) (idx != nullptr ? idx[r] : r) * ldx;
    uint16_t* dst = out + (size_t) r * ncols;
#pragma unroll
    for (int c = item_ct1.get_group(2) * item_ct1.get_local_range(2) +
                 item_ct1.get_local_id(2);
         c < ncols;
         c += item_ct1.get_group_range(2) * item_ct1.get_local_range(2))
        dst[c] = sycl::bit_cast<unsigned short, sycl::half>(
            sycl::vec<float, 1>(src[c])
                .convert<sycl::half, sycl::rounding_mode::rte>()[0]);
}

// ---------------------------------------------------------------- the prompt path's tensor-core GEMMs (sm_80+)
// A block: BM rows x 128 outputs (one output rotation block).  Per 128 inputs the block's rows are scaled by suh,
// rotated and stored FP16 in shared memory (the A operands); per 16-row slice each warp decodes its 16 x 16 tile
// straight into mma.m16n8k16's B fragments - a lane's 8 steps ARE the fragment order (steps 0,1 / 2,3 rows 2t+{0,1} /
// 2t+8+{0,1} of column g, steps 4..7 the same of column g + 8) - and multiplies every m-tile of the block with them.
// The output rotation and svh run on the F32 tile in shared memory.
#if !defined(STRATA_USE_HIP)
namespace tc {

constexpr int BM = 32, MT = BM / 16, AP = 136;   // rows a block, its m-tiles, the A rows' pitch (halves)

__dpct_inline__ void mma16816(float *c, const uint32_t *a, uint32_t b0,
                              uint32_t b1) {
#if defined(DPCT_COMPATIBILITY_TEMP) && DPCT_COMPATIBILITY_TEMP >= 800
#if defined(__SYCL_DEVICE_ONLY__) && defined(__NVPTX__)
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
                 "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, "
                 "{%0,%1,%2,%3};\n"
                 : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
                 : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0),
                   "r"(b1));
#else
    {
        volatile void *d_mat_frag_ct1[4] = {&c[0], &c[1], &c[2], &c[3]};
        sycl::vec<uint32_t, 4> a_mat_frag_ct1(a[0], a[1], a[2], a[3]);
        sycl::vec<uint32_t, 2> b_mat_frag_ct1(b0, b1);
        sycl::vec<float, 4> c_mat_frag_ct1(c[0], c[1], c[2], c[3]);
        dpct::experimental::matrix::mma<16, 8, 16, sycl::half, float>(
            reinterpret_cast<volatile void **>(d_mat_frag_ct1), &a_mat_frag_ct1,
            &b_mat_frag_ct1, &c_mat_frag_ct1);
    }
#endif
#endif
}
// the lane's 4 B registers of tile t (words staged in st): b[0..1] columns 0..7, b[2..3] columns 8..15
template <int K, int cb>
__dpct_inline__ void tile_frags(const uint32_t *st, int lane, uint32_t *b) {
    uint32_t f[4];
    frag8<K, cb>(st, lane, f);
#pragma unroll
    for (int i = 0; i < 4; ++i) b[i] = f[i];
}

// A[r][j] = FP16(H(x_r * suh)[j]), j < 128, for the block's rows (zero past nrows).  x_r: F32 row idx[r] of xf (ldx)
// when xf is given, else FP16 row r of xh (ldx).  Every warp, BM / 8 rows each.
__dpct_inline__ void load_a(const float *__restrict__ xf,
                            const int *__restrict__ idx,
                            const sycl::half *__restrict__ xh, int ldx, int r0,
                            int nrows, int k0, const uint16_t *__restrict__ suh,
                            sycl::half *A) {
    auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
    const int lane = item_ct1.get_local_id(2) & 31,
              warp = item_ct1.get_local_id(2) >> 5;
    for (int rr = warp; rr < BM; rr += kWarps) {
        float v[4] = {0.0f, 0.0f, 0.0f, 0.0f};
        const int r = r0 + rr;
        if (rr < nrows) {
            const int c = k0 + lane * 4;
            if (xf != nullptr) {
                const float* src = xf + (size_t) idx[r] * ldx + c;
#pragma unroll
                for (int i = 0; i < 4; ++i) v[i] = src[i];
            } else {
                const sycl::half *src = xh + (size_t)r * ldx + c;
#pragma unroll
                for (int i = 0; i < 4; ++i) v[i] =
                    sycl::vec<sycl::half, 1>(src[i])
                        .convert<float, sycl::rounding_mode::automatic>()[0];
            }
#pragma unroll
            for (int i = 0; i < 4; ++i) v[i] *= h2f(suh[c + i]);
            had128(v, lane);
        }
#pragma unroll
        for (int i = 0; i < 4; ++i) A[rr * AP + lane * 4 + i] =
            sycl::vec<float, 1>(v[i])
                .convert<sycl::half, sycl::rounding_mode::rte>()[0];
    }
}

// the m-tile's A registers for the 16-column slice s of A
__dpct_inline__ void a_frags(const sycl::half *A, int mt, int s, int lane,
                             uint32_t *a) {
    const int g = lane >> 2, t = lane & 3;
    const sycl::half *p = A + (mt * 16 + g) * AP + s * 16 + 2 * t;
    a[0] = *reinterpret_cast<const uint32_t*>(p);
    a[1] = *reinterpret_cast<const uint32_t*>(p + 8 * AP);
    a[2] = *reinterpret_cast<const uint32_t*>(p + 8);
    a[3] = *reinterpret_cast<const uint32_t*>(p + 8 * AP + 8);
}

// acc (this warp's 16 columns of the block's BM rows) -> T[BM][128] F32
__dpct_inline__ void store_acc(const float (&acc)[MT][2][4], float *T) {
    auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
    const int lane = item_ct1.get_local_id(2) & 31,
              warp = item_ct1.get_local_id(2) >> 5, g = lane >> 2, t = lane & 3;
#pragma unroll
    for (int mt = 0; mt < MT; ++mt)
#pragma unroll
        for (int ns = 0; ns < 2; ++ns) {
            const int c = warp * 16 + ns * 8 + 2 * t, r = mt * 16 + g;
            T[r * 128 + c] = acc[mt][ns][0];
            T[r * 128 + c + 1] = acc[mt][ns][1];
            T[(r + 8) * 128 + c] = acc[mt][ns][2];
            T[(r + 8) * 128 + c + 1] = acc[mt][ns][3];
        }
}

struct GArgs {
    const uint8_t* wbase;     // expert e's blob at wbase + e * stride (dense: 0, the offsets are pointers)
    size_t stride;
    size_t tr[3], suh[3], svh[3];
    int ld[3];                // tiles per 16-row slice of each matrix
    const int* bounds;        // device: expert e's rows [bounds[e], bounds[e + 1]) (null: one matrix, rows [0, m))
    int row0, m;              // local row of a sorted row: bounds[e] - row0 + r
    int k, n;                 // of the matrix (gate/up: n_embd, n_ff)
    const float* xf;          // F32 input rows by index (idx[local row]) ...
    const int* idx;
    const sycl::half *xh; // ... or FP16 rows by local row
    int ldx;
    float* y;                 // F32 output rows (ldy), y = beta y + out ...
    int ldy;
    float beta;
    sycl::half *h; // ... or (gate/up) the swiglu as FP16 rows (ldh)
    int ldh;
    float limit;
};

// one matrix: y = beta y + H(H(x * suh) W') * svh.  grid (row blocks, n / 128, experts)
template <int K, int cb>
/*
DPCT1110:116: The total declared local variable size in device function
gemm_kernel exceeds 128 bytes and may cause high register pressure. Consult with
your hardware vendor to find the total register size available and adjust the
code, or use smaller sub-group size to avoid high register pressure.
*/
__dpct_inline__ void gemm_kernel(const GArgs a) {
    auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
    constexpr int W = 8 * K;
    auto &sm = *sycl::ext::oneapi::group_local_memory_for_overwrite<
        unsigned char[BM * 128 * 4]>(
        sycl::ext::oneapi::this_work_item::get_work_group<
            3>()); // A (BM x AP halves), then the F32 tile
    auto &stage = *sycl::ext::oneapi::group_local_memory_for_overwrite<
        uint32_t[kWarps][64]>(
        sycl::ext::oneapi::this_work_item::get_work_group<3>());
    sycl::half *A = reinterpret_cast<sycl::half *>(sm);
    float* T = reinterpret_cast<float*>(sm);
    const int e = item_ct1.get_group(0), rb = item_ct1.get_group(2),
              g = item_ct1.get_group(1);
    const int e0 = a.bounds != nullptr ? a.bounds[e] - a.row0 : 0;
    const int ne = a.bounds != nullptr ? a.bounds[e + 1] - a.bounds[e] : a.m;
    const int r0 = e0 + rb * BM, nrows = sycl::min(BM, ne - rb * BM);
    if (nrows <= 0) return;
    const uint8_t* blob = a.wbase + (size_t) e * a.stride;
    const uint32_t* tr = (const uint32_t*) (blob + a.tr[0]);
    const uint16_t* suh = (const uint16_t*) (blob + a.suh[0]);
    const uint16_t* svh = (const uint16_t*) (blob + a.svh[0]);
    const int lane = item_ct1.get_local_id(2) & 31,
              warp = item_ct1.get_local_id(2) >> 5;
    float acc[MT][2][4] = {};
    for (int k0 = 0; k0 < a.k; k0 += 128) {
        /*
        DPCT1118:117: SYCL group functions and algorithms must be encountered in
        converged control flow. You may need to adjust the code.
        */
        /*
        DPCT1065:398: Consider replacing sycl::nd_item::barrier() with
        sycl::nd_item::barrier(sycl::access::fence_space::local_space) for
        better performance if there is no access to global memory.
        */
        item_ct1.barrier();
        load_a(a.xf, a.idx, a.xh, a.ldx, r0, nrows, k0, suh, A);
        /*
        DPCT1118:118: SYCL group functions and algorithms must be encountered in
        converged control flow. You may need to adjust the code.
        */
        /*
        DPCT1065:399: Consider replacing sycl::nd_item::barrier() with
        sycl::nd_item::barrier(sycl::access::fence_space::local_space) for
        better performance if there is no access to global memory.
        */
        item_ct1.barrier();
#pragma unroll 1
        for (int s = 0; s < 8; ++s) {
            const int ks = k0 / 16 + s;
            const uint32_t* src = tr + ((size_t) ks * a.ld[0] + g * 8 + warp) * W;
            sycl::group_barrier(
                sycl::ext::oneapi::this_work_item::get_sub_group());
#pragma unroll
            for (int i = lane; i < W; i += 32) stage[warp][i] = src[i];
            sycl::group_barrier(
                sycl::ext::oneapi::this_work_item::get_sub_group());
            uint32_t b[4];
            tile_frags<K, cb>(stage[warp], lane, b);
#pragma unroll
            for (int mt = 0; mt < MT; ++mt) {
                uint32_t af[4];
                a_frags(A, mt, s, lane, af);
                mma16816(acc[mt][0], af, b[0], b[1]);
                mma16816(acc[mt][1], af, b[2], b[3]);
            }
        }
    }
    /*
    DPCT1065:396: Consider replacing sycl::nd_item::barrier() with
    sycl::nd_item::barrier(sycl::access::fence_space::local_space) for better
    performance if there is no access to global memory.
    */
    item_ct1.barrier();
    store_acc(acc, T);
    /*
    DPCT1065:397: Consider replacing sycl::nd_item::barrier() with
    sycl::nd_item::barrier(sycl::access::fence_space::local_space) for better
    performance if there is no access to global memory.
    */
    item_ct1.barrier();
    for (int rr = warp; rr < nrows; rr += kWarps) {
        float v[4];
#pragma unroll
        for (int i = 0; i < 4; ++i) v[i] = T[rr * 128 + lane * 4 + i];
        had128(v, lane);
        float* yr = a.y + (size_t) (r0 + rr) * a.ldy + g * 128 + lane * 4;
#pragma unroll
        for (int i = 0; i < 4; ++i) {
            const float o = v[i] * h2f(svh[g * 128 + lane * 4 + i]);
            yr[i] = a.beta != 0.0f ? a.beta * yr[i] + o : o;
        }
    }
}

// gate and up (matrices 0 and 1) over the same rows, the clamped swiglu, FP16 h.  grid (row blocks, n / 128, experts)
template <int K, int cb>
/*
DPCT1110:119: The total declared local variable size in device function
gate_up_kernel exceeds 128 bytes and may cause high register pressure. Consult
with your hardware vendor to find the total register size available and adjust
the code, or use smaller sub-group size to avoid high register pressure.
*/
__dpct_inline__ void gate_up_kernel(const GArgs a) {
    auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
    constexpr int W = 8 * K;
    auto &sm = *sycl::ext::oneapi::group_local_memory_for_overwrite<
        unsigned char[2 * BM * 128 * 4]>(
        sycl::ext::oneapi::this_work_item::get_work_group<
            3>()); // A_g, A_u, then the two F32 tiles
    auto &stage = *sycl::ext::oneapi::group_local_memory_for_overwrite<
        uint32_t[kWarps][64]>(
        sycl::ext::oneapi::this_work_item::get_work_group<3>());
    sycl::half *A[2] = {reinterpret_cast<sycl::half *>(sm),
                        reinterpret_cast<sycl::half *>(sm) + BM * AP};
    float* T[2] = {reinterpret_cast<float*>(sm), reinterpret_cast<float*>(sm) + BM * 128};
    const int e = item_ct1.get_group(0), rb = item_ct1.get_group(2),
              g = item_ct1.get_group(1);
    const int e0 = a.bounds[e] - a.row0, ne = a.bounds[e + 1] - a.bounds[e];
    const int r0 = e0 + rb * BM, nrows = sycl::min(BM, ne - rb * BM);
    if (nrows <= 0) return;
    const uint8_t* blob = a.wbase + (size_t) e * a.stride;
    const int lane = item_ct1.get_local_id(2) & 31,
              warp = item_ct1.get_local_id(2) >> 5;
    float acc[2][MT][2][4] = {};
    for (int k0 = 0; k0 < a.k; k0 += 128) {
        /*
        DPCT1118:120: SYCL group functions and algorithms must be encountered in
        converged control flow. You may need to adjust the code.
        */
        /*
        DPCT1065:402: Consider replacing sycl::nd_item::barrier() with
        sycl::nd_item::barrier(sycl::access::fence_space::local_space) for
        better performance if there is no access to global memory.
        */
        item_ct1.barrier();
#pragma unroll
        for (int m = 0; m < 2; ++m)
            load_a(a.xf, a.idx, a.xh, a.ldx, r0, nrows, k0, (const uint16_t*) (blob + a.suh[m]), A[m]);
        /*
        DPCT1118:121: SYCL group functions and algorithms must be encountered in
        converged control flow. You may need to adjust the code.
        */
        /*
        DPCT1065:403: Consider replacing sycl::nd_item::barrier() with
        sycl::nd_item::barrier(sycl::access::fence_space::local_space) for
        better performance if there is no access to global memory.
        */
        item_ct1.barrier();
#pragma unroll 1
        for (int s = 0; s < 8; ++s) {
            const int ks = k0 / 16 + s;
#pragma unroll
            for (int m = 0; m < 2; ++m) {
                const uint32_t* src = (const uint32_t*) (blob + a.tr[m]) + ((size_t) ks * a.ld[m] + g * 8 + warp) * W;
                sycl::group_barrier(
                    sycl::ext::oneapi::this_work_item::get_sub_group());
#pragma unroll
                for (int i = lane; i < W; i += 32) stage[warp][i] = src[i];
                sycl::group_barrier(
                    sycl::ext::oneapi::this_work_item::get_sub_group());
                uint32_t b[4];
                tile_frags<K, cb>(stage[warp], lane, b);
#pragma unroll
                for (int mt = 0; mt < MT; ++mt) {
                    uint32_t af[4];
                    a_frags(A[m], mt, s, lane, af);
                    mma16816(acc[m][mt][0], af, b[0], b[1]);
                    mma16816(acc[m][mt][1], af, b[2], b[3]);
                }
            }
        }
    }
    /*
    DPCT1065:400: Consider replacing sycl::nd_item::barrier() with
    sycl::nd_item::barrier(sycl::access::fence_space::local_space) for better
    performance if there is no access to global memory.
    */
    item_ct1.barrier();
    store_acc(acc[0], T[0]);
    store_acc(acc[1], T[1]);
    /*
    DPCT1065:401: Consider replacing sycl::nd_item::barrier() with
    sycl::nd_item::barrier(sycl::access::fence_space::local_space) for better
    performance if there is no access to global memory.
    */
    item_ct1.barrier();
    const uint16_t* svg = (const uint16_t*) (a.wbase + (size_t) e * a.stride + a.svh[0]) + g * 128;
    const uint16_t* svu = (const uint16_t*) (a.wbase + (size_t) e * a.stride + a.svh[1]) + g * 128;
    for (int rr = warp; rr < nrows; rr += kWarps) {
        float vg[4], vu[4];
#pragma unroll
        for (int i = 0; i < 4; ++i) {
            vg[i] = T[0][rr * 128 + lane * 4 + i];
            vu[i] = T[1][rr * 128 + lane * 4 + i];
        }
        had128(vg, lane);
        had128(vu, lane);
        sycl::half *hr = a.h + (size_t)(r0 + rr) * a.ldh + g * 128 + lane * 4;
#pragma unroll
        for (int i = 0; i < 4; ++i) {
            float gv = vg[i] * h2f(svg[lane * 4 + i]), uv = vu[i] * h2f(svu[lane * 4 + i]);
            if (a.limit > 0.0f) {
                gv = sycl::fmin(gv, a.limit);
                uv = sycl::fmin(sycl::fmax(uv, -a.limit), a.limit);
            }
            hr[i] =
                sycl::vec<float, 1>((gv / (1.0f + sycl::native::exp(-gv))) * uv)
                    .convert<sycl::half, sycl::rounding_mode::rte>()[0];
        }
    }
}

}  // namespace tc
#endif

// ---------------------------------------------------------------- dispatch
// F(K, cb) for K 1..8 and the three codebooks
template <template <int, int> class F, typename... A>
bool dispatch(int K, int cb, A&&... args) {
#define EXL3_CASE(KK)                                                    \
    case KK:                                                             \
        if (cb == 2) F<KK, 2>::run(std::forward<A>(args)...);            \
        else if (cb == 1) F<KK, 1>::run(std::forward<A>(args)...);       \
        else F<KK, 0>::run(std::forward<A>(args)...);                    \
        return true;
    switch (K) {
        EXL3_CASE(1) EXL3_CASE(2) EXL3_CASE(3) EXL3_CASE(4) EXL3_CASE(5) EXL3_CASE(6) EXL3_CASE(7) EXL3_CASE(8)
        default: return false;
    }
#undef EXL3_CASE
}

template <int K, int cb> struct RunMv {
    static void run(const KBatch &b, int blocks, dpct::queue_ptr s) {
        dpct::has_capability_or_fail(s->get_device(), {sycl::aspect::fp16});

      s->parallel_for(
        sycl::nd_range<3>(sycl::range(1, 1, blocks) * sycl::range(1, 1, kThreads), sycl::range(1, 1, kThreads)),
        [=](sycl::nd_item<3> item_ct1) [[sycl::reqd_sub_group_size(32)]] {
            mv_kernel<K, cb>(b);
        }); }
};
template <int K, int cb> struct RunWin {
    static void run(const KWin &a, int groups, dpct::queue_ptr s) {
        dpct::has_capability_or_fail(s->get_device(), {sycl::aspect::fp16});

        s->parallel_for(
            sycl::nd_range<3>(
                sycl::range(2u, (unsigned)groups, (unsigned)(a.n_ff / kCols)) *
                    sycl::range(1, 1, kThreads),
                sycl::range(1, 1, kThreads)),
            [=](sycl::nd_item<3> item_ct1) [[sycl::reqd_sub_group_size(32)]] {
                win_gu_kernel<K, cb>(a);
            });
    }
};
template <int K, int cb> struct RunWinDown {
    static void run(const KWin &a, int groups, dpct::queue_ptr s) {
        dpct::has_capability_or_fail(s->get_device(), {sycl::aspect::fp16});

        s->parallel_for(
            sycl::nd_range<3>(
                sycl::range(1, (unsigned)groups, (unsigned)(a.n_embd / kCols)) *
                    sycl::range(1, 1, kThreads),
                sycl::range(1, 1, kThreads)),
            [=](sycl::nd_item<3> item_ct1) [[sycl::reqd_sub_group_size(32)]] {
                win_down_kernel<K, cb>(a);
            });
    }
};
template <int K, int cb> struct RunGU {
    static void run(const KMoe &a, dpct::queue_ptr s) {
        dpct::has_capability_or_fail(s->get_device(), {sycl::aspect::fp16});

        s->parallel_for(
            sycl::nd_range<3>(
                sycl::range(2u, (unsigned)a.k, (unsigned)(a.n_ff / kCols)) *
                    sycl::range(1, 1, kThreads),
                sycl::range(1, 1, kThreads)),
            [=](sycl::nd_item<3> item_ct1) [[sycl::reqd_sub_group_size(32)]] {
                moe_gate_up_kernel<K, cb>(a);
            });
    }
};
template <int K, int cb> struct RunDown {
    static void run(const KMoe &a, dpct::queue_ptr s) {
        dpct::has_capability_or_fail(s->get_device(), {sycl::aspect::fp16});

        s->parallel_for(
            sycl::nd_range<3>(
                sycl::range(1, (unsigned)a.k, (unsigned)(a.n_embd / kCols)) *
                    sycl::range(1, 1, kThreads),
                sycl::range(1, 1, kThreads)),
            [=](sycl::nd_item<3> item_ct1) [[sycl::reqd_sub_group_size(32)]] {
                moe_down_kernel<K, cb>(a);
            });
    }
};
template <int K, int cb> struct RunRows {
    static void run(const Mat &m, uint16_t *dst, bool bf16, dpct::queue_ptr s) {
        dpct::has_capability_or_fail(s->get_device(), {sycl::aspect::fp16});

        s->submit([&](sycl::handler &cgh) {
            auto m_ld_m_ld_m_n_ct1 = m.ld > 0 ? m.ld : m.n / 16;

            cgh.parallel_for(
                sycl::nd_range<3>(sycl::range(1, (unsigned)(m.k / kCols),
                                              (unsigned)(m.n / kCols)) *
                                      sycl::range(1, 1, kThreads),
                                  sycl::range(1, 1, kThreads)),
                [=](sycl::nd_item<3> item_ct1)
                    [[sycl::reqd_sub_group_size(32)]] {
                        recon_rows_kernel<K, cb>(m.trellis, m_ld_m_ld_m_n_ct1,
                                                 m.suh, m.svh, m.k, dst, bf16);
                    });
        });
    }
};
#if !defined(STRATA_USE_HIP)
template <int K, int cb> struct RunGemm {
    static void run(const tc::GArgs &a, dpct::dim3 grid, dpct::queue_ptr s) {
        dpct::has_capability_or_fail(s->get_device(), {sycl::aspect::fp16});

      s->parallel_for(
        sycl::nd_range<3>(grid * sycl::range(1, 1, kThreads), sycl::range(1, 1, kThreads)),
        [=](sycl::nd_item<3> item_ct1) [[sycl::reqd_sub_group_size(32)]] {
            tc::gemm_kernel<K, cb>(a);
        }); }
};
template <int K, int cb> struct RunGateUp {
    static void run(const tc::GArgs &a, dpct::dim3 grid, dpct::queue_ptr s) {
        dpct::has_capability_or_fail(s->get_device(), {sycl::aspect::fp16});

        s->parallel_for(sycl::nd_range<3>(grid * sycl::range(1, 1, kThreads),
                                          sycl::range(1, 1, kThreads)),
                        [=](sycl::nd_item<3> item_ct1)
                            [[sycl::reqd_sub_group_size(32)]] {
                                tc::gate_up_kernel<K, cb>(a);
                            });
    }
};
#endif
template <int K, int cb> struct RunInnerRows {
    static void run(const Mat &m, uint16_t *dst, dpct::queue_ptr s) {
        dpct::has_capability_or_fail(s->get_device(), {sycl::aspect::fp16});

        s->submit([&](sycl::handler &cgh) {
            auto m_ld_m_ld_m_n_ct1 = m.ld > 0 ? m.ld : m.n / 16;

            cgh.parallel_for(
                sycl::nd_range<3>(sycl::range(1, (unsigned)(m.k / kCols),
                                              (unsigned)(m.n / kCols)) *
                                      sycl::range(1, 1, kThreads),
                                  sycl::range(1, 1, kThreads)),
                [=](sycl::nd_item<3> item_ct1) {
                    inner_rows_kernel<K, cb>(m.trellis, m_ld_m_ld_m_n_ct1, m.k,
                                             dst);
                });
        });
    }
};
template <int K, int cb> struct RunInner {
    static void run(const Mat &m, uint16_t *dst, dpct::queue_ptr s) {
        dpct::has_capability_or_fail(s->get_device(), {sycl::aspect::fp16});

        s->submit([&](sycl::handler &cgh) {
            auto m_ld_m_ld_m_n_ct1 = m.ld > 0 ? m.ld : m.n / 16;

            cgh.parallel_for(
                sycl::nd_range<3>(sycl::range(1, (unsigned)(m.k / 16),
                                              (unsigned)(m.n / kCols)) *
                                      sycl::range(1, 1, kThreads),
                                  sycl::range(1, 1, kThreads)),
                [=](sycl::nd_item<3> item_ct1) {
                    recon_inner_kernel<K, cb>(m.trellis, m_ld_m_ld_m_n_ct1, m.n,
                                              dst);
                });
        });
    }
};

// ---------------------------------------------------------------- per-stream scratch
constexpr size_t kPartFloats = (size_t) 2 << 20;   // split partials: 8 MB
constexpr size_t kCounters = 16384;
constexpr size_t kCntGU = 8192;                    // the counters from here on: the experts' gate / up pairs
constexpr size_t kHFloats = (size_t) 8 * 8192;     // the experts' h
constexpr size_t kUFloats = (size_t) 128 * 4096;   // the experts' up values: a window's 16 tokens x 8, n_ff <= 4096
struct Scratch {
    float* part = nullptr;
    unsigned* cnt = nullptr;
    float* h = nullptr;
    float* u = nullptr;
};
std::mutex g_mu;
std::map<std::pair<int, dpct::queue_ptr>, Scratch> g_scratch;

Scratch *scratch(dpct::queue_ptr s) try {
    int dev = 0;
    dev = dpct::get_current_device_id();
    // (the last stream this thread used: an engine thread per device launches without the lock)
    thread_local int t_dev = -1;
    thread_local dpct::queue_ptr t_s = &dpct::get_in_order_queue();
    thread_local Scratch* t_sc = nullptr;
    if (t_sc != nullptr && t_dev == dev && t_s == s) return t_sc;
    std::lock_guard<std::mutex> lk(g_mu);
    auto it = g_scratch.find({dev, s});
    if (it != g_scratch.end()) {
        t_dev = dev, t_s = s, t_sc = &it->second;
        return &it->second;
    }
    sycl::ext::oneapi::experimental::queue_state cs =
        sycl::ext::oneapi::experimental::queue_state::executing;
    if (DPCT_CHECK_ERROR((cs = s->ext_oneapi_get_state())) == 0 &&
        cs != sycl::ext::oneapi::experimental::queue_state::executing) {
        std::fprintf(stderr, "exl3: scratch for a stream first seen inside a graph capture (call exl3::prepare)\n");
        return nullptr;
    }
    Scratch sc;
    if (DPCT_CHECK_ERROR(sc.part = sycl::malloc_device<float>(
                             kPartFloats, dpct::get_in_order_queue())) != 0 ||
        DPCT_CHECK_ERROR(sc.cnt = sycl::malloc_device<unsigned int>(
                             kCounters, dpct::get_in_order_queue())) != 0 ||
        DPCT_CHECK_ERROR(sc.h = sycl::malloc_device<float>(
                             kHFloats, dpct::get_in_order_queue())) != 0 ||
        DPCT_CHECK_ERROR(sc.u = sycl::malloc_device<float>(
                             kUFloats, dpct::get_in_order_queue())) != 0 ||
        DPCT_CHECK_ERROR(s->memset(sc.cnt, 0, kCounters * sizeof(unsigned))) !=
            0) {
        std::fprintf(stderr, "exl3: scratch did not allocate\n");
        return nullptr;
    }
    s->wait();
    return &(g_scratch[{dev, s}] = sc);
}
catch (sycl::exception const &exc) {
  std::cerr << exc.what() << "Exception caught at file:" << __FILE__
            << ", line:" << __LINE__ << std::endl;
  std::exit(1);
}

int sm_count() {
    int dev = 0, sms = 0;
    dev = dpct::get_current_device_id();
    sms = dpct::get_device(dev).get_max_compute_units();
    return std::max(1, sms);
}

// Whether the current device decodes with v1.1.8's kernels (exl3_legacy.cu): NVIDIA before sm_80, where this file's
// tiles run on the CUDA cores and were slower; STRATA_GLM_EXL3_DECODE=legacy | new picks one on any GPU (A/B).  The
// token path and the verify windows take the same family on a device, so a window stays bit for bit its token path.
// Asked once per device.
bool use_legacy() {
#if 1 // SYCL uses the current portable kernels, never NVIDIA legacy kernels
    // HIP has no v1.1.8 kernels.  Metal has neither them nor a compute capability to ask for: the shim answers
    // cudaDeviceGetAttribute with an error, so `major` stayed 0 and every Metal run without STRATA_GLM_EXL3_DECODE=new
    // took this branch and aborted in exl3::legacy::prepare's stub.  The 2.25 bpw pack never uses EXL3.
    return false;
#else
    static std::atomic<int> cache[64];   // 0 not asked, 1 v1.1.8's kernels, 2 this file's
    int dev = 0;
    dev = dpct::get_current_device_id();
    if (dev < 0 || dev >= 64) return false;
    int c = cache[dev].load(std::memory_order_acquire);
    if (c == 0) {
        int major = 0;
        major = dpct::get_device(dev).get_major_version();
        const char* v = std::getenv("STRATA_GLM_EXL3_DECODE");
        c = v != nullptr && std::strcmp(v, "legacy") == 0 ? 1 : v != nullptr && std::strcmp(v, "new") == 0 ? 2
            : major < 8                                    ? 1
                                                           : 2;
        if (c == 1) std::fprintf(stderr, "exl3: CUDA%d decodes with v1.1.8's kernels\n", dev);
        cache[dev].store(c, std::memory_order_release);
    }
    return c == 1;
#endif
}

// the blocks of mv_kernel<K> an SM of the current device keeps at once (its registers and shared memory), asked once
// per device and K
template <int K, int cb> struct OccMv {
    static void run(int& n) { n = 1; } // a conservative persistent grid; rows retain their grid-stride traversal
};
int mv_occ(int K) {
    static std::atomic<int> cache[64][9];   // 0: not asked
    int dev = 0;
    dev = dpct::get_current_device_id();
    if (dev < 0 || dev >= 64 || K < 1 || K > 8) return 2;
    int n = cache[dev][K].load(std::memory_order_acquire);
    if (n == 0) {
        n = 2;
        dispatch<OccMv>(K, 2, n);
        n = std::max(1, n);
        cache[dev][K].store(n, std::memory_order_release);
    }
    return n;
}

// mv's input split for m: S blocks of kr inputs per output block, at most kMaxKR inputs a block (the same for every
// activation count, so a job of several sums each activation as a job of one).  About two blocks per SM; where an SM
// keeps three or more (`occ`: the RTX 3090's at K > 4) four a SM, down to 128 inputs a block - its decode sets the pace
// there, and the more, shorter blocks balance it over the SMs (RTX 3090, K6 / K5, us: a KDA layer's 8192 x 4096
// output 47.6 -> 45.5, the dense layers' 4096 x 12288 gate 58.9 -> 52.5 and 12288 x 4096 down 59.3 -> 52.2, the
// shared expert's 4096 x 2048 17.4 -> 16.5); with two a SM, a launch that would spill a short second wave (more blocks
// than the SMs keep, less than one and a half times) takes twice the splits (RTX 5060 Ti, K5: that gate 99.4 -> 84.8
// us, that down 100.6 -> 84.9)
// call_groups: the output blocks the call's jobs of this code count together against the target, not the job's own -
// where an SM keeps three or more, all of them (RTX 3090, K6, a KDA layer's q, k and v, three 4096 x 8192 jobs: 1536
// blocks 119.6 us, 384 blocks 105.9); at two a SM only jobs of the same shape (RTX 5060 Ti: those q, k and v 191 ->
// 185 us, a shared expert's gate and up 35.9 -> 35.0; the MLA layer's q_a and kv_a together would lose 8%).  Never the
// activations: a window's token gets the token path's split
void mv_split(const Mat& m, int sms, int occ, int& S, int& kr, int call_groups = 0) {
    const bool many = occ >= 3;
    const int groups = std::max(m.n / kCols, call_groups);
    const int kr_min = many ? 128 : 256, target = (many ? 4 : 2) * sms, P = occ * sms;
    S = (m.k + kMaxKR - 1) / kMaxKR;
    while (groups * S < target && m.k / (S * 2) >= kr_min) S *= 2;
    if (!many && groups * S > P && 2 * groups * S < 3 * P && m.k / (S * 2) >= kr_min) S *= 2;
    kr = ((m.k + S - 1) / S + 127) / 128 * 128;
    S = (m.k + kr - 1) / kr;
}

}  // namespace

Mat view(const Mat& m, int c0, int nc) {
    Mat v = m;
    v.ld = m.ld > 0 ? m.ld : m.n / 16;
    v.trellis = m.trellis + (size_t) (c0 / 16) * 8 * m.K;
    v.svh = m.svh + c0;
    v.n = nc;
    return v;
}

bool supported(const Mat& m) {
    return m.trellis != nullptr && m.suh != nullptr && m.svh != nullptr && m.k > 0 && m.n > 0 && m.k % 128 == 0 &&
           m.n % 128 == 0 && m.K >= 1 && m.K <= 8 && m.cb >= 0 && m.cb <= 2;
}

void set_skip(const int* flag) { g_skip = flag; }

int mv_splits(const Mat& m) {
    if (!supported(m)) return 0;
    int S = 1, kr = 0;
    mv_split(m, sm_count(), mv_occ(m.K), S, kr);
    return S;
}

int mv_cols(const Mat& m) {
    if (!supported(m)) return 0;
#if !defined(STRATA_USE_HIP)
    if (use_legacy()) return 4;   // (v1.1.8's kernels: the runs of four their callers have always used)
#endif
    const int S = mv_splits(m);
    if (S <= 0) return 0;
    if (S == 1) return kTok;
    return (int) std::max<size_t>(1, std::min<size_t>(kTok, kPartFloats / ((size_t) S * m.n)));
}

int launch_errors() {
#if 0 // NVIDIA legacy kernels are not part of the SYCL target
    return g_launch_errors.load(std::memory_order_relaxed) + legacy::launch_errors();
#else
    return g_launch_errors.load(std::memory_order_relaxed);
#endif
}

bool prepare(dpct::queue_ptr s) {
    EXL3_LEGACY(prepare(s));
    return scratch(s) != nullptr;
}

bool several_ok() {
    EXL3_LEGACY(several_ok());
    return true;
}

bool mv(const MvJob *jobs, int n, dpct::queue_ptr s) {
    EXL3_LEGACY(mv(jobs, n, s));
    if (n <= 0 || n > kMaxMvJobs) return false;
    Scratch* sc = scratch(s);
    if (sc == nullptr) return false;
    const int sms = sm_count();
    // the launch units: a job's activations in runs of up to kTok (one pass over its weights each)
    struct Unit {
        int j, c0, nc;
    };
    Unit units[kMaxMvJobs * kMaxMvCols];
    int nu = 0;
    const int run = kTok;
    for (int j = 0; j < n; ++j) {
        const Mat& m = *jobs[j].m;
        if (!supported(m) || jobs[j].y == nullptr || (jobs[j].xf == nullptr && jobs[j].xq == nullptr) ||
            jobs[j].ncol < 1 || jobs[j].ncol > kMaxMvCols) {
            std::fprintf(stderr, "exl3 mv: job %d unsupported (k %d n %d K %d cb %d, %d activations)\n", j, m.k, m.n,
                         m.K, m.cb, jobs[j].ncol);
            return false;
        }
        for (int c0 = 0; c0 < jobs[j].ncol; c0 += run) units[nu++] = {j, c0, std::min(run, jobs[j].ncol - c0)};
    }
    // each job's call_groups (mv_split): the output blocks of the call's jobs of its code - of its shape too where an SM
    // keeps fewer than three
    int cg[kMaxMvJobs] = {};
    for (int j = 0; j < n; ++j) {
        const Mat& a = *jobs[j].m;
        const bool many = mv_occ(a.K) >= 3;
        for (int q = 0; q < n; ++q) {
            const Mat& b = *jobs[q].m;
            if (b.K == a.K && b.cb == a.cb && (many || (b.k == a.k && b.n == a.n))) cg[j] += b.n / kCols;
        }
    }
    bool done[sizeof(units) / sizeof(units[0])] = {};
    for (int i = 0; i < nu; ++i) {
        if (done[i]) continue;
        const Mat& m0 = *jobs[units[i].j].m;
        // a launch: the units of one (K, codebook), as many as KBatch and the split scratch take (the next launch on
        // the stream may use the scratch again: this one's last blocks are done with it by then)
        KBatch b{};
        int blocks = 0, nb = 0;
        size_t part_at = 0, cnt_at = 0;
        for (int u = i; u < nu && nb < kMaxMvJobs; ++u) {
            const MvJob& job = jobs[units[u].j];
            const Mat& m = *job.m;
            if (done[u] || m.K != m0.K || m.cb != m0.cb) continue;
            const int c0 = units[u].c0, nc = units[u].nc, groups = m.n / kCols;
            int S = 1, kr = 0;
            mv_split(m, sms, mv_occ(m.K), S, kr, cg[units[u].j]);
            const size_t need = S > 1 ? (size_t) S * m.n * nc : 0;
            if (S > 1 && (part_at + need > kPartFloats || cnt_at + (size_t) groups > kCounters)) {
                if (nb > 0) continue;   // the next launch
                std::fprintf(stderr, "exl3 mv: scratch too small for k %d n %d (%d activations)\n", m.k, m.n, nc);
                return false;
            }
            done[u] = true;
            KJob& J = b.j[nb];
            J.tr = m.trellis;
            J.suh = m.suh;
            J.svh = m.svh;
            J.xf = job.xf != nullptr ? job.xf + c0 * job.xf_cs : nullptr;
            J.xq = job.xq != nullptr ? (const Q81*) ((const uint8_t*) job.xq + c0 * job.xq_cs) : nullptr;
            J.y = job.y + c0 * job.y_cs;
            J.bias = job.bias;
            J.alpha = job.alpha;
            J.k = m.k;
            J.n = m.n;
            J.ld = m.ld > 0 ? m.ld : m.n / 16;
            J.S = S;
            J.kr = kr;
            J.part = sc->part + part_at;
            J.cnt = sc->cnt + cnt_at;
            J.nc = nc;
            J.xf_cs = job.xf_cs;
            J.xq_cs = job.xq_cs;
            J.y_cs = job.y_cs;
            part_at += need;
            if (S > 1) cnt_at += (size_t) groups;
            blocks += groups * S;
            b.blk_end[nb++] = blocks;
        }
        b.n = nb;
        b.skip = g_skip;
        if (!dispatch<RunMv>(m0.K, m0.cb, b, blocks, s)) return false;
    }
    launch_check("mv");
    return true;
}

static bool moe_args(const MoeArgs &a, KMoe &m, dpct::queue_ptr s) {
    Scratch* sc = scratch(s);
    if (sc == nullptr) return false;
    if (a.k < 1 || a.k > 8 || a.n_embd % kCols != 0 || a.n_ff % kCols != 0 || a.n_embd > kMaxMoeK ||
        a.n_ff > kMaxMoeK || (size_t) a.k * a.n_ff > kHFloats || (size_t) a.k * a.n_embd > kPartFloats ||
        (size_t) (a.n_embd / kCols) > kCntGU || kCntGU + (size_t) a.k * (a.n_ff / kCols) > kCounters) {
        std::fprintf(stderr, "exl3 moe: geometry not covered (k %d n_embd %d n_ff %d)\n", a.k, a.n_embd, a.n_ff);
        return false;
    }
    m = KMoe{};
    m.plan_ptr = a.plan_ptr;
    m.plan_w = a.plan_w;
    m.cpu_part = a.cpu_part;
    m.cpu_flag = a.cpu_flag;
    m.h = sc->h;
    m.u = sc->u;
    m.part = sc->part;
    m.cnt = sc->cnt;
    m.cnt_gu = sc->cnt + kCntGU;
    for (int r = 0; r < 3; ++r) {
        m.suh[r] = a.lay.suh[r];
        m.svh[r] = a.lay.svh[r];
        m.tr[r] = a.lay.trellis[r];
    }
    m.k = a.k;
    m.n_embd = a.n_embd;
    m.n_ff = a.n_ff;
    m.limit = a.limit;
    m.skip = g_skip;
    return true;
}

void moe_gate_up(const MoeArgs &a, const float *x, float *h,
                 dpct::queue_ptr s) {
    EXL3_LEGACY(moe_gate_up(a, x, h, s));
    KMoe m;
    if (!moe_args(a, m, s)) return;
    if (a.lay.K[0] != a.lay.K[1]) {
        std::fprintf(stderr, "exl3 moe_gate_up: gate K %d != up K %d\n", a.lay.K[0], a.lay.K[1]);
        return;
    }
    m.x = x;
    if (h != nullptr) m.h = h;
    if (!dispatch<RunGU>(a.lay.K[0], a.lay.cb, m, s)) std::fprintf(stderr, "exl3 moe_gate_up: K %d\n", a.lay.K[0]);
    launch_check("moe_gate_up");
}

void moe_down(const MoeArgs &a, const float *h, const float *sh_out, float *out,
              dpct::queue_ptr s) {
    EXL3_LEGACY(moe_down(a, h, sh_out, out, s));
    KMoe m;
    if (!moe_args(a, m, s)) return;
    if (h != nullptr) m.h = const_cast<float*>(h);
    m.sh_out = sh_out;
    m.out = out;
    if (!dispatch<RunDown>(a.lay.K[2], a.lay.cb, m, s)) std::fprintf(stderr, "exl3 moe_down: K %d\n", a.lay.K[2]);
    launch_check("moe_down");
}

bool win_experts(const WinArgs &a, dpct::queue_ptr s) {
    EXL3_LEGACY(win_experts(a, s));
    if (a.T < 1 || a.k < 1 || a.k > 8 || a.n_embd % kCols != 0 || a.n_ff % kCols != 0 || a.n_embd > kMaxMoeK ||
        a.n_ff > kMaxMoeK || a.lay.K[0] != a.lay.K[1] || a.h == nullptr || a.part == nullptr || a.out == nullptr) {
        std::fprintf(stderr, "exl3 win_experts: not covered (T %d k %d n_embd %d n_ff %d, gate K %d up K %d)\n", a.T,
                     a.k, a.n_embd, a.n_ff, a.lay.K[0], a.lay.K[1]);
        return false;
    }
    Scratch* sc = scratch(s);
    if (sc == nullptr) return false;
    const int groups = a.T * a.k;   // at most (the blocks past the real count return at once)
    if ((size_t) a.T * 8 * a.n_ff > kUFloats || kCntGU + (size_t) groups * (a.n_ff / kCols) > kCounters) {
        std::fprintf(stderr, "exl3 win_experts: %d tokens are more than the scratch takes\n", a.T);
        return false;
    }
    KWin w{};
    w.grp_ptr = a.grp_ptr;
    w.grp_start = a.grp_start;
    w.grp_ent = a.grp_ent;
    w.n_groups = a.n_groups;
    w.plan_ptr = a.plan_ptr;
    w.plan_w = a.plan_w;
    w.cpu_part = a.cpu_part;
    w.cpu_stride = a.cpu_stride != 0 ? a.cpu_stride : (size_t) a.n_embd;
    w.cpu_flag = a.cpu_flag;
    w.x = a.x;
    w.x_stride = a.x_stride;
    w.h = a.h;
    w.u = sc->u;
    w.cnt_gu = sc->cnt + kCntGU;
    w.part = a.part;
    w.sh_out = a.sh_out;
    w.out = a.out;
    w.out_stride = a.out_stride;
    for (int r = 0; r < 3; ++r) {
        w.suh[r] = a.lay.suh[r];
        w.svh[r] = a.lay.svh[r];
        w.tr[r] = a.lay.trellis[r];
    }
    w.k = a.k;
    w.n_embd = a.n_embd;
    w.n_ff = a.n_ff;
    w.limit = a.limit;
    if (!dispatch<RunWin>(a.lay.K[0], a.lay.cb, w, groups, s) ||
        !dispatch<RunWinDown>(a.lay.K[2], a.lay.cb, w, groups, s))
        return false;
        if (a.wait_ptr != nullptr) s->parallel_for(
            sycl::nd_range<3>(sycl::range(1, 1, 32), sycl::range(1, 1, 32)),
            [=](sycl::nd_item<3> item_ct1) {
                win_wait_kernel(a.wait_ptr, a.wait_val);
            });
    s->parallel_for(
        sycl::nd_range<3>(
            sycl::range(1, (unsigned)a.T, (unsigned)(a.n_embd / kCols)) *
                sycl::range(1, 1, kCols),
            sycl::range(1, 1, kCols)),
        [=](sycl::nd_item<3> item_ct1) {
            win_combine_kernel(w);
        });
    launch_check("win_experts");
    return true;
}

bool mma_supported() {
#if defined(STRATA_USE_HIP)
    return false;
#else
    int dev = 0, major = 0;
    dev = dpct::get_current_device_id();
    major = dpct::get_device(dev).get_major_version();
    return major >= 8;
#endif
}

void set_gate_up(const SetArgs &sa, const float *x, int ldx, const int *row_idx,
                 uint16_t *h16, int ldh, dpct::queue_ptr s) {
#if !defined(STRATA_USE_HIP)
    if (sa.lay.K[0] != sa.lay.K[1]) {
        std::fprintf(stderr, "exl3 set_gate_up: gate K %d != up K %d\n", sa.lay.K[0], sa.lay.K[1]);
        g_launch_errors.fetch_add(1, std::memory_order_relaxed);
        return;
    }
    if (sa.n_exp <= 0 || sa.max_rows <= 0) return;
    tc::GArgs a{};
    a.wbase = sa.wbase;
    a.stride = sa.stride;
    for (int r = 0; r < 2; ++r) {
        a.tr[r] = sa.lay.trellis[r];
        a.suh[r] = sa.lay.suh[r];
        a.svh[r] = sa.lay.svh[r];
        a.ld[r] = sa.n_ff / 16;
    }
    a.bounds = sa.bounds;
    a.row0 = sa.row0;
    a.k = sa.n_embd;
    a.n = sa.n_ff;
    a.xf = x;
    a.idx = row_idx;
    a.ldx = ldx;
    a.h = (sycl::half *)h16;
    a.ldh = ldh;
    a.limit = sa.limit;
    const dpct::dim3 grid((unsigned)((sa.max_rows + tc::BM - 1) / tc::BM),
                          (unsigned)(sa.n_ff / kCols), (unsigned)sa.n_exp);
    if (!dispatch<RunGateUp>(sa.lay.K[0], sa.lay.cb, a, grid, s)) std::fprintf(stderr, "exl3 set_gate_up: K\n");
    launch_check("set_gate_up");
#else
    (void) sa; (void) x; (void) ldx; (void) row_idx; (void) h16; (void) ldh; (void) s;
#endif
}

void set_down(const SetArgs &sa, const uint16_t *h16, int ldh, float *y,
              int ldy, dpct::queue_ptr s) {
#if !defined(STRATA_USE_HIP)
    if (sa.n_exp <= 0 || sa.max_rows <= 0) return;
    tc::GArgs a{};
    a.wbase = sa.wbase;
    a.stride = sa.stride;
    a.tr[0] = sa.lay.trellis[2];
    a.suh[0] = sa.lay.suh[2];
    a.svh[0] = sa.lay.svh[2];
    a.ld[0] = sa.n_embd / 16;
    a.bounds = sa.bounds;
    a.row0 = sa.row0;
    a.k = sa.n_ff;
    a.n = sa.n_embd;
    a.xh = (const sycl::half *)h16;
    a.ldx = ldh;
    a.y = y;
    a.ldy = ldy;
    const dpct::dim3 grid((unsigned)((sa.max_rows + tc::BM - 1) / tc::BM),
                          (unsigned)(sa.n_embd / kCols), (unsigned)sa.n_exp);
    if (!dispatch<RunGemm>(sa.lay.K[2], sa.lay.cb, a, grid, s)) std::fprintf(stderr, "exl3 set_down: K\n");
    launch_check("set_down");
#else
    (void) sa; (void) h16; (void) ldh; (void) y; (void) ldy; (void) s;
#endif
}

void gemm_rows(const Mat &m, const uint16_t *x16, int ldx, int rows, float *y,
               int ldy, float beta, dpct::queue_ptr s) {
#if !defined(STRATA_USE_HIP)
    if (rows <= 0) return;
    tc::GArgs a{};
    a.wbase = nullptr;
    a.tr[0] = (size_t) m.trellis;
    a.suh[0] = (size_t) m.suh;
    a.svh[0] = (size_t) m.svh;
    a.ld[0] = m.ld > 0 ? m.ld : m.n / 16;
    a.m = rows;
    a.k = m.k;
    a.n = m.n;
    a.xh = (const sycl::half *)x16;
    a.ldx = ldx;
    a.y = y;
    a.ldy = ldy;
    a.beta = beta;
    const dpct::dim3 grid((unsigned)((rows + tc::BM - 1) / tc::BM),
                          (unsigned)(m.n / kCols), 1);
    if (!supported(m) || !dispatch<RunGemm>(m.K, m.cb, a, grid, s)) {
        std::fprintf(stderr, "exl3 gemm_rows: unsupported matrix\n");
        g_launch_errors.fetch_add(1, std::memory_order_relaxed);
        return;
    }
    launch_check("gemm_rows");
#else
    (void) m; (void) x16; (void) ldx; (void) rows; (void) y; (void) ldy; (void) beta; (void) s;
#endif
}

void reconstruct_rows(const Mat &m, uint16_t *dst, bool bf16,
                      dpct::queue_ptr s) {
    if (!supported(m) || !dispatch<RunRows>(m.K, m.cb, m, dst, bf16, s)) {
        std::fprintf(stderr, "exl3 reconstruct_rows: unsupported matrix\n");
        g_launch_errors.fetch_add(1, std::memory_order_relaxed);
        return;
    }
    launch_check("reconstruct_rows");
}

void reconstruct_inner_rows(const Mat &m, uint16_t *dst, dpct::queue_ptr s) {
    if (!supported(m) || !dispatch<RunInnerRows>(m.K, m.cb, m, dst, s)) {
        std::fprintf(stderr, "exl3 reconstruct_inner_rows: unsupported matrix\n");
        g_launch_errors.fetch_add(1, std::memory_order_relaxed);
        return;
    }
    launch_check("reconstruct_inner_rows");
}

void rows_rot_f16(const float *x, int ldx, const int *idx, int nrows, int ncols,
                  const uint16_t *suh, uint16_t *out, dpct::queue_ptr s) {
    if (nrows <= 0) return;
    const long long warps = (long long) nrows * (ncols / kCols);
    {
        dpct::has_capability_or_fail(s->get_device(), {sycl::aspect::fp16});

        s->parallel_for(
            sycl::nd_range<3>(
                sycl::range(1, 1, (unsigned)((warps + kWarps - 1) / kWarps)) *
                    sycl::range(1, 1, kThreads),
                sycl::range(1, 1, kThreads)),
            [=](sycl::nd_item<3> item_ct1) [[sycl::reqd_sub_group_size(32)]] {
                rows_rot_f16_kernel(x, ldx, idx, nrows, ncols, suh, out);
            });
    }
    launch_check("rows_rot_f16");
}

void rows_rot_out(float *y, int ldy, int nrows, int ncols, const uint16_t *svh,
                  dpct::queue_ptr s) {
    if (nrows <= 0) return;
    const long long warps = (long long) nrows * (ncols / kCols);
    s->parallel_for(
        sycl::nd_range<3>(
            sycl::range(1, 1, (unsigned)((warps + kWarps - 1) / kWarps)) *
                sycl::range(1, 1, kThreads),
            sycl::range(1, 1, kThreads)),
        [=](sycl::nd_item<3> item_ct1) [[sycl::reqd_sub_group_size(32)]] {
            rows_rot_out_kernel(y, ldy, nrows, ncols, svh);
        });
    launch_check("rows_rot_out");
}

void rows_f16(const float *x, int ldx, const int *idx, int nrows, int ncols,
              uint16_t *out, dpct::queue_ptr s) {
    if (nrows <= 0) return;
#if defined(STRATA_METAL)
    if (::strata::dsv41::metal_host::rows_f16(x, ldx, idx, nrows, ncols, out)) return;
#endif
    s->parallel_for(sycl::nd_range<3>(sycl::range(1, (unsigned)nrows,
                                                  (unsigned)std::min(
                                                      8, (ncols + 255) / 256)) *
                                          sycl::range(1, 1, 256),
                                      sycl::range(1, 1, 256)),
                    [=](sycl::nd_item<3> item_ct1) {
                        rows_f16_kernel(x, ldx, idx, ncols, out);
                    });
    launch_check("rows_f16");
}

void reconstruct_inner(const Mat &m, uint16_t *dst, dpct::queue_ptr s) {
    if (!supported(m) || !dispatch<RunInner>(m.K, m.cb, m, dst, s)) {
        std::fprintf(stderr, "exl3 reconstruct_inner: unsupported matrix\n");
        g_launch_errors.fetch_add(1, std::memory_order_relaxed);
        return;
    }
    launch_check("reconstruct_inner");
}

}  // namespace strata::dsv41k::exl3
