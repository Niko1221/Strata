// src/qwen35/q36_fa.cuh - the prompt attention of q36 on tensor cores (FA pilot, step 1, second iteration).
//
// Replaces the cuBLAS branch of q36_engine.cu:2275-2302 (sliced QK^T + k_attn_softmax16 + sliced PV) with one
// flash-style kernel: no score buffer, so no tq limit and no ~45 000 launches per prompt.
//
// Why this is not a literal port of llama.cpp's fattn-mma-f16.cuh: that kernel is built on the tl:: fragment types
// of ggml-cuda/mma.cuh (1606 lines), which pull in ggml-cuda/common.cuh, and strata_qwen35's include directories
// (CMakeLists.txt:917) do not see them. The math is the same m16n8k16 f16 -> f32 MMA with the fragment mapping
// documented at qsa_prompt_attn.cu:91-95, and that file's mma16816 (qsa_prompt_attn.cu:57-72) is copied verbatim.
//
// The tile is now a template: kQs queries per block (4 or 8, so 32 or 64 MMA rows) and kSlice KV positions per
// slice (32 or 64). The four combinations are the ones Cfg accepts; the launcher dispatches over them, so the
// pilot can pick the tile per layer without touching this file again.
//
// A block owns one KV head and kG * kQs rows = the G = 8 query heads of that KV head for kQs queries; KV is walked
// in slices with an online softmax and the p.v accumulators live in registers across the slices. The warps split
// the rows into 16-row MMA tiles (MT of them) and, within a row block, the columns in half: WarpPerM = 2 in every
// tile, so a warp owns 16 rows, half of the score columns (ScoreTiles of 8) and half of DV (PvTiles of 8 = 128
// columns) - the accumulator is acc[16][4] = 64 registers per thread in all four variants, and the block grows
// with the rows (Threads = kThreadsPerRow * kRows: 128 threads for 32 rows, 256 for 64).
// The layouts are q36's, passed as strides - they are all multiples of 16 bytes, so nothing is repacked
// (prep2/fa-pilot-design.md section 1):
//   Q [T][H][HD] f16   (q16, q36_engine.cu:1657, filled by k_to_f16 :2273)
//   K,V [HKV][P][HD] f16 (k16/v16 :1655-1656, filled by k_kv_f16_kv :2256-2270)
//   O [T][H][HD] f32   (att :1639), before the gate: k_gate_mul :2303 stays outside this kernel.
//
// K and V share one staged buffer (S.kv): the slice's K is consumed by the score MMA and V by p.v, and the barrier
// that already sits between those two phases is what makes the reuse safe. Without the alias the 8x64 tile needs
// 128 512 B of shared and sm_80 allows 101 376 B per block, so that variant could not launch at all; with it the
// four tiles cost 41 216 / 65 536 / 64 256 / 94 720 B (shared_bytes_of below). The price is that the V load no
// longer overlaps the score MMA - it issues one barrier later.
//
// Second change against the first iteration, and it is a bug fix rather than a knob: the running max of a row used
// to live in shared (mrow[kRows]), read by the four threads of the row's quad and written by one of them in the
// same slice. No barrier sits between those two, so a quad thread could read the value its leader had just written
// for the SAME slice and rescale its row by exp2(0) = 1 instead of the real decay - the lsum of that row would then
// grow past the softmax denominator and the row's output would be quietly wrong. The max now lives in the registers
// of the quad (m_cur, identical in all four threads because mx is shuffle-reduced), and shared keeps only lsum and
// alpha: lsum is touched by the leader alone, alpha is written before and read after a barrier.
//
// NOT IN ANY BUILD TARGET ON PURPOSE: no engine file includes it yet, so the engine compiles and runs exactly as
// before until the pilot adds `#include "q36_fa.cuh"` and the STRATA_Q36_FA_LAYER branch.

#pragma once

#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <cstdint>

namespace q36fa {

constexpr int kHD = 256;        // head_dim = DKQ = DV
constexpr int kG = 8;           // query heads per KV head (H / HKV = 16 / 2)
constexpr int kPad = 8;         // row padding in halves: 16-byte rows, banks spread (kLD stays a multiple of 8)
constexpr int kLD = kHD + kPad;
constexpr int kThreadsPerRow = 4;   // a softmax row belongs to a quad: two shfl_xor steps reduce it

// The tile geometry. Every constant the kernel uses is derived here, so a new (kQs, kSlice) pair is one more
// instantiation as long as the asserts below hold; the asserts are the contract the launcher relies on.
template <int Qs_, int Slice_>
struct Cfg {
    static constexpr int Qs = Qs_, Slice = Slice_;
    static constexpr int Rows = kG * Qs;                        // MMA rows of the block: 32 or 64
    static constexpr int MT = Rows / 16;                        // 16-row tiles: 2 or 4
    static constexpr int NW = Slice / 8;                        // 8-column score tiles of a slice: 4 or 8
    static constexpr int Threads = kThreadsPerRow * Rows;       // 128 or 256
    static constexpr int Warps = Threads / 32;                  // 4 or 8
    static constexpr int WarpPerM = Warps / MT;                 // warps sharing a row block: 2 in all four tiles
    static constexpr int ScoreTiles = NW / WarpPerM;            // score tiles a warp computes: 2 or 4
    static constexpr int PvCols = kHD / WarpPerM;               // DV columns a warp accumulates: 128
    static constexpr int PvTiles = PvCols / 8;                  // ... as 8-column tiles: 16
    static constexpr int ColsPerThread = Slice / kThreadsPerRow;   // slice columns one softmax thread owns: 8/16
    static_assert(Qs == 4 || Qs == 8, "the FA pilot instantiates kQs = 4 and 8");
    static_assert(Slice == 32 || Slice == 64, "the FA pilot instantiates kSlice = 32 and 64");
    static_assert(Rows % 16 == 0, "the rows of a block must divide into 16-row MMA tiles");
    static_assert(Warps % MT == 0, "the row blocks must divide the warps");
    static_assert(NW % WarpPerM == 0, "the score tiles of a row block must divide its warps");
    static_assert(PvCols % 8 == 0, "DV must split across the warps of a row block");
    static_assert(Slice % kThreadsPerRow == 0, "the columns of a slice must divide a row's quad");
    static_assert(Threads <= 1024, "the block is bigger than CUDA allows");
};

// m16n8k16 (sm_80+): qsa_prompt_attn.cu:57-72 without the Turing fallback - the pilot requires sm_80+, the caller
// keeps the cuBLAS branch on anything older.
// Fragment mapping (PTX ISA, and the comment at qsa_prompt_attn.cu:91-95):
//   a[0] = A[gid][2*tig, 2*tig+1]        a[1] = A[gid+8][2*tig, 2*tig+1]
//   a[2] = A[gid][2*tig+8, 2*tig+9]      a[3] = A[gid+8][2*tig+8, 2*tig+9]
//   b[0] = B[2*tig, 2*tig+1][gid]        b[1] = B[2*tig+8, 2*tig+9][gid]
//   c[0], c[1] = C[gid][2*tig], C[gid][2*tig+1];  c[2], c[3] = the same columns of row gid+8
__device__ __forceinline__ void mma16816(float* c, const uint32_t* a, const uint32_t* b) {
#if defined(__HIPCC__) || (defined(__CUDA_ARCH__) && __CUDA_ARCH__ < 800)
    __trap();   // unreachable: q36_fa_launch() returns false and the caller keeps the cuBLAS branch
#else
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, "
                 "{%0,%1,%2,%3};\n"
                 : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
                 : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]));
#endif
}

struct Args {
    const __half* Q;            // q16
    const __half* K;            // k16 (head kvh at kvh * stride_kv_h)
    const __half* V;            // v16
    float* O;                   // att
    int T;                      // query rows of this sub-chunk
    int P;                      // p0 + T: the visible prefix, any value (the tail of a slice is masked)
    int p0;                     // absolute position of sub-chunk row 0
    int H;                      // query heads
    int HD;                     // head_dim
    int stride_q_t;             // H * HD
    int stride_kv_h;            // P * HD: k16/v16 are packed by the visible prefix, not by ctx
    float scale;                // 1 / sqrt(HD)
};

template <typename C>
struct SmemT {
    __half q[C::Rows][kLD];
    __half kv[C::Slice][kLD];               // the slice's K, then (after the score MMA) its V
    __half p[C::Rows][C::Slice + kPad];
    float s[C::Rows][C::Slice + 4];
    float lsum[C::Rows];                    // running sum of exp - written only by the row's own quad leader
    float alpha[C::Rows];                   // per-slice rescale factor, read by the p.v warps after a barrier
};

__device__ __forceinline__ uint32_t pack2(const __half lo, const __half hi) {
    __half2 h = __halves2half2(lo, hi);
    return *reinterpret_cast<uint32_t*>(&h);
}

// grid = (ceil(T / Qs), HKV), block = Cfg<Qs, Slice>::Threads.
template <int Qs, int Slice, int Threads = Cfg<Qs, Slice>::Threads>
__global__ void __launch_bounds__(Threads) k_fa_prefill(const Args a) {
    using C = Cfg<Qs, Slice>;
    constexpr int kQs = C::Qs, kSlice = C::Slice, kRows = C::Rows, kThreads = C::Threads;
    constexpr int kScoreTiles = C::ScoreTiles, kPvTiles = C::PvTiles, kPvCols = C::PvCols;
    constexpr int kColsPerThread = C::ColsPerThread, kWarpPerM = C::WarpPerM;

    extern __shared__ __align__(16) unsigned char raw[];
    SmemT<C>& S = *reinterpret_cast<SmemT<C>*>(raw);
    const int t = (int) threadIdx.x;
    const int warp = t >> 5, lane = t & 31;
    const int gid = lane >> 2, tig = lane & 3;
    const int i0 = blockIdx.x * kQs;
    const int kvh = blockIdx.y;
    const int m = warp / kWarpPerM, nhalf = warp % kWarpPerM;   // row block, and its half of the columns
    const float scl2 = a.scale * 1.4426950408889634f;           // 1/sqrt(HD) * log2(e): the softmax runs in exp2

    // Row r is (i, g) = (r / kG, r % kG); its absolute position is a.p0 + i0 + i.
    for (int x = t; x < kRows * (kHD / 8); x += kThreads) {
        const int r = x / (kHD / 8), c8 = (x % (kHD / 8)) * 8;
        const int i = r / kG, g = r % kG, qrow = i0 + i;
        __half* dst = &S.q[r][c8];
        if (qrow < a.T)
            *reinterpret_cast<uint4*>(dst) =
                __ldg(reinterpret_cast<const uint4*>(a.Q + ((size_t) qrow * a.H + kvh * kG + g) * a.HD + c8));
        else
            *reinterpret_cast<uint4*>(dst) = make_uint4(0, 0, 0, 0);
    }
    // The running max of a row lives in the registers of its own quad (all four threads keep the same value): one
    // shared mrow read by four threads and written by one of them inside the same slice is a race no barrier covers.
    // Only the quad leader touches S.lsum, so its per-slice accumulation stays a single-thread read-modify-write.
    for (int r = t; r < kRows; r += kThreads) S.lsum[r] = 0.f;
    float m_cur = -INFINITY;
    __syncthreads();

    float acc[kPvTiles][4];                                 // this warp's PvTiles n8 tiles of its kPvCols of DV
#pragma unroll
    for (int j = 0; j < kPvTiles; ++j)
#pragma unroll
        for (int e = 0; e < 4; ++e) acc[j][e] = 0.f;

    for (int c0 = 0; c0 < a.P; c0 += kSlice) {
        const int ncol = a.P - c0 < kSlice ? a.P - c0 : kSlice;
        // ---- the slice's K into S.kv
        for (int x = t; x < kSlice * (kHD / 8); x += kThreads) {
            const int r = x / (kHD / 8), c8 = (x % (kHD / 8)) * 8;
            uint4 kv0 = make_uint4(0, 0, 0, 0);
            if (r < ncol)
                kv0 = __ldg(reinterpret_cast<const uint4*>(
                    a.K + (size_t) (c0 + r) * a.HD + c8 + (size_t) kvh * a.stride_kv_h));
            *reinterpret_cast<uint4*>(&S.kv[r][c8]) = kv0;
        }
        __syncthreads();

        // ---- q.k: the kRows x kSlice tile is MT x NW tiles of 16x8; the warp takes its m and ScoreTiles of n
        float cq[kScoreTiles][4];
#pragma unroll
        for (int j = 0; j < kScoreTiles; ++j)
#pragma unroll
            for (int e = 0; e < 4; ++e) cq[j][e] = 0.f;
        for (int k0 = 0; k0 < kHD; k0 += 16) {
            uint32_t qa[4];
#pragma unroll
            for (int j = 0; j < 4; ++j) {
                const int row = m * 16 + gid + ((j == 1 || j == 3) ? 8 : 0);
                const int col = k0 + 2 * tig + (j >= 2 ? 8 : 0);   // a0/a1 rows gid/gid+8 at k=2*tig, a2/a3 the same at +8
                qa[j] = pack2(S.q[row][col], S.q[row][col + 1]);
            }
#pragma unroll
            for (int nn = 0; nn < kScoreTiles; ++nn) {
                const int n = nhalf * kScoreTiles + nn;
                uint32_t kb[2];
#pragma unroll
                for (int j = 0; j < 2; ++j) {
                    const int kp = 2 * tig + (j ? 8 : 0);         // B[k][n]: k = the head dim, n = a KV cell
                    const int cell = n * 8 + gid;
                    kb[j] = pack2(S.kv[cell][k0 + kp], S.kv[cell][k0 + kp + 1]);
                }
                mma16816(cq[nn], qa, kb);
            }
        }
        __syncthreads();                                        // S.kv is free: every warp has read its K

        // ---- the slice's V into S.kv, and the scores into S.s (every warp writes only its own columns)
        for (int x = t; x < kSlice * (kHD / 8); x += kThreads) {
            const int r = x / (kHD / 8), c8 = (x % (kHD / 8)) * 8;
            uint4 vv = make_uint4(0, 0, 0, 0);
            if (r < ncol)
                vv = __ldg(reinterpret_cast<const uint4*>(
                    a.V + (size_t) (c0 + r) * a.HD + c8 + (size_t) kvh * a.stride_kv_h));
            *reinterpret_cast<uint4*>(&S.kv[r][c8]) = vv;
        }
#pragma unroll
        for (int nn = 0; nn < kScoreTiles; ++nn) {
            const int n = nhalf * kScoreTiles + nn;
#pragma unroll
            for (int e = 0; e < 4; ++e) {
                const int row = m * 16 + gid + (e >= 2 ? 8 : 0);
                const int col = n * 8 + 2 * tig + (e & 1);
                S.s[row][col] = cq[nn][e] * scl2;
            }
        }
        __syncthreads();

        // ---- online softmax over the slice: kThreadsPerRow threads per row, ColsPerThread columns each, reduced
        // inside the row's quad
        {
            const int row = t / kThreadsPerRow, cg = t % kThreadsPerRow;
            const int i = row / kG, qpos = a.p0 + i0 + i;
            float sv[kColsPerThread], mx = -INFINITY;
#pragma unroll
            for (int j = 0; j < kColsPerThread; ++j) {
                const int col = cg * kColsPerThread + j;
                sv[j] = (col < ncol && c0 + col <= qpos) ? S.s[row][col] : -INFINITY;
                mx = fmaxf(mx, sv[j]);
            }
            mx = fmaxf(mx, __shfl_xor_sync(0xffffffffu, mx, 1));
            mx = fmaxf(mx, __shfl_xor_sync(0xffffffffu, mx, 2));
            const float m_old = m_cur, m_new = fmaxf(m_old, mx);
            const float rescale = m_old == -INFINITY ? 0.f : exp2f(m_old - m_new);
            m_cur = m_new;
            float ps = 0.f;
#pragma unroll
            for (int j = 0; j < kColsPerThread; ++j) {
                const float pv = sv[j] == -INFINITY ? 0.f : exp2f(sv[j] - m_new);
                S.p[row][cg * kColsPerThread + j] = __float2half_rn(pv);
                ps += pv;
            }
            ps += __shfl_xor_sync(0xffffffffu, ps, 1);
            ps += __shfl_xor_sync(0xffffffffu, ps, 2);
            if (cg == 0) {                                      // the quad leader owns the row's sum and factor
                S.lsum[row] = fmaf(S.lsum[row], rescale, ps);
                S.alpha[row] = rescale;
            }
        }
        __syncthreads();

        // ---- p.v: A = p (16 x kSlice, kSlice/16 k steps), B = the slice's V (k = a KV cell, n = the output dim)
#pragma unroll
        for (int j = 0; j < kPvTiles; ++j) {
            const float s0 = S.alpha[m * 16 + gid], s8 = S.alpha[m * 16 + gid + 8];
#pragma unroll
            for (int e = 0; e < 4; ++e) acc[j][e] *= (e >= 2 ? s8 : s0);
        }
        for (int k0 = 0; k0 < kSlice; k0 += 16) {
            uint32_t pa[4];
#pragma unroll
            for (int j = 0; j < 4; ++j) {
                const int row = m * 16 + gid + ((j == 1 || j == 3) ? 8 : 0);
                const int col = k0 + 2 * tig + (j >= 2 ? 8 : 0);
                pa[j] = pack2(S.p[row][col], S.p[row][col + 1]);
            }
#pragma unroll
            for (int nn = 0; nn < kPvTiles; ++nn) {
                const int d = nhalf * kPvCols + nn * 8;
                uint32_t vb[2];
#pragma unroll
                for (int j = 0; j < 2; ++j) {
                    const int kp = 2 * tig + (j ? 8 : 0);       // B[k][n]: k = the KV cell, n = the dim
                    vb[j] = pack2(S.kv[k0 + kp][d + gid], S.kv[k0 + kp + 1][d + gid]);
                }
                mma16816(acc[nn], pa, vb);
            }
        }
        __syncthreads();                                        // the next slice rewrites kv/p/s
    }

    // ---- O = acc / lsum, in the [T][H][HD] layout (the gate is applied by the caller's k_gate_mul)
#pragma unroll
    for (int nn = 0; nn < kPvTiles; ++nn) {
        const int d = nhalf * kPvCols + nn * 8;
#pragma unroll
        for (int e = 0; e < 4; ++e) {
            const int row = m * 16 + gid + (e >= 2 ? 8 : 0);
            const int col = d + 2 * tig + (e & 1);
            const int i = row / kG, g = row % kG, qrow = i0 + i;
            if (qrow < a.T)
                a.O[((size_t) qrow * a.H + kvh * kG + g) * a.HD + col] = acc[nn][e] / S.lsum[row];
        }
    }
}

// The tiles the pilot instantiates and what each one costs in shared memory (q + kv + p + s + lsum + alpha):
//   4 x 32: 16896 + 16896 + 2560 + 4608 + 256 = 41 216 B (40.25 KiB)  -> 2 blocks / SM on GA104
//   8 x 32: 33792 + 16896 + 5120 + 9216 + 512 = 65 536 B (64.00 KiB)  -> 1 block / SM  (256 threads)
//   4 x 64: 16896 + 33792 + 4608 + 8704 + 256 = 64 256 B (62.75 KiB)  -> 1 block / SM
//   8 x 64: 33792 + 33792 + 9216 + 17408 + 512 = 94 720 B (92.5 KiB)  -> 1 block / SM (256 threads)
enum Variant { k4x32 = 0, k8x32 = 1, k4x64 = 2, k8x64 = 3, kVariantCount = 4 };

template <int Qs, int Slice>
constexpr size_t shared_bytes_of() {
    const size_t n = sizeof(SmemT<Cfg<Qs, Slice>>);
    static_assert(n <= 101376, "the tile needs more shared than sm_80 allows per block");
    return n;
}

inline size_t q36_fa_shared(Variant v) {
    switch (v) {
    case k8x32: return shared_bytes_of<8, 32>();
    case k4x64: return shared_bytes_of<4, 64>();
    case k8x64: return shared_bytes_of<8, 64>();
    default: return shared_bytes_of<4, 32>();
    }
}

inline bool q36_fa_available() {
    int dev = 0, major = 0, minor = 0;
    if (cudaGetDevice(&dev) != cudaSuccess) return false;
    if (cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, dev) != cudaSuccess) return false;
    if (cudaDeviceGetAttribute(&minor, cudaDevAttrComputeCapabilityMinor, dev) != cudaSuccess) return false;
    return major * 10 + minor >= 80;
}

// The launch shape of a tile: q36_fa_launch re-derives it and refuses anything else, so ask for it instead of
// spelling out ceil(T / kQs) and the thread count at the call site.
struct Shape {
    dim3 grid, block;
    size_t shared = 0;
};

inline bool q36_fa_shape(Variant v, int T, int HKV, Shape* out) {
    if (T <= 0 || HKV <= 0 || (int) v < 0 || (int) v >= kVariantCount) return false;
    switch (v) {
    case k8x32: out->grid = dim3((unsigned) ((T + 7) / 8), (unsigned) HKV); out->block = dim3(Cfg<8, 32>::Threads); break;
    case k4x64: out->grid = dim3((unsigned) ((T + 3) / 4), (unsigned) HKV); out->block = dim3(Cfg<4, 64>::Threads); break;
    case k8x64: out->grid = dim3((unsigned) ((T + 7) / 8), (unsigned) HKV); out->block = dim3(Cfg<8, 64>::Threads); break;
    default: out->grid = dim3((unsigned) ((T + 3) / 4), (unsigned) HKV); out->block = dim3(Cfg<4, 32>::Threads); break;
    }
    out->shared = q36_fa_shared(v);
    return true;
}

// The geometry this file is compiled for: the KAT head counts. Anything else keeps the cuBLAS branch.
inline bool q36_fa_geometry(int H, int HKV, int HD) {
    return HD == kHD && H == 16 && HKV == 2 && (H / HKV) == kG;
}

template <int Qs, int Slice>
inline bool launch_one(const Shape& sh, cudaStream_t st, const Args& a, int T, int HKV) {
    using C = Cfg<Qs, Slice>;
    static bool raised = false;
    if ((int) sh.block.x != C::Threads || (int) sh.grid.x != (T + C::Qs - 1) / C::Qs || (int) sh.grid.y != HKV)
        return false;
    if (sh.shared != sizeof(SmemT<C>)) return false;
    if (!raised) {
        // Best-effort: the 4x32 tile (40.4 KiB) wants two blocks per SM, which needs the SM's shared carveout at its
        // maximum. The driver picks the carveout itself, so ask for it; if the attribute is refused the kernel still
        // launches, just with one block per SM.
        cudaFuncSetAttribute((const void*) &k_fa_prefill<Qs, Slice>, cudaFuncAttributePreferredSharedMemoryCarveout,
                             100);
        if (cudaFuncSetAttribute((const void*) &k_fa_prefill<Qs, Slice>, cudaFuncAttributeMaxDynamicSharedMemorySize,
                                 (int) sizeof(SmemT<C>)) != cudaSuccess)
            return false;
        raised = true;
    }
    // The error flag is per-thread and sticky: a successful call does not clear it, so a stale error from an
    // unrelated runtime call earlier in this thread would make this launch look failed (and the caller would
    // silently keep the cuBLAS branch). Clear it first; the carveout attribute above is best effort and may set it.
    (void) cudaGetLastError();
    k_fa_prefill<Qs, Slice><<<sh.grid, sh.block, sh.shared, st>>>(a);
    return cudaGetLastError() == cudaSuccess;
}

// One launch per attention layer per sub-chunk. Returns false (and launches nothing) when the geometry, the tile,
// the shape or the device is not one this file can serve, so the caller keeps the cuBLAS branch it replaced.
inline bool q36_fa_launch(dim3 grid, dim3 block, cudaStream_t st, const __half* Q, const __half* K,
                          const __half* V, float* O, int T, int P, int p0, int H, int HKV, int HD, int ctx,
                          float scale, Variant v = k4x32) {
    if (!q36_fa_geometry(H, HKV, HD) || (int) v < 0 || (int) v >= kVariantCount) return false;
    if (P <= 0 || P > ctx) return false;
    if (!q36_fa_available()) return false;
    int dev = 0, optin = 0;
    if (cudaGetDevice(&dev) != cudaSuccess) return false;
    if (cudaDeviceGetAttribute(&optin, cudaDevAttrMaxSharedMemoryPerBlockOptin, dev) != cudaSuccess) return false;
    const size_t need = q36_fa_shared(v);
    if ((size_t) optin < need) return false;
    Shape sh;
    if (!q36_fa_shape(v, T, HKV, &sh)) return false;
    sh.grid = grid;
    sh.block = block;
    sh.shared = need;
    Args a;
    a.Q = Q; a.K = K; a.V = V; a.O = O;
    a.T = T; a.P = P; a.p0 = p0; a.H = H; a.HD = HD;
    a.stride_kv_h = P * HD; a.scale = scale;
    switch (v) {
    case k8x32: return launch_one<8, 32>(sh, st, a, T, HKV);
    case k4x64: return launch_one<4, 64>(sh, st, a, T, HKV);
    case k8x64: return launch_one<8, 64>(sh, st, a, T, HKV);
    default: return launch_one<4, 32>(sh, st, a, T, HKV);
    }
}

}  // namespace q36fa
