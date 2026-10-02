// src/prefill/moe_fused.cu - see include/strata/prefill/moe_fused.hpp (#136: the Q2_0 pack's prompt experts on
// Strata's own int8 tensor-core kernels, with the grouping on the GPU).
//
// The arithmetic.  A Q2_0 block is 64 weights w = d_w * (q - 1), q in 0..3, four codes per byte, weight j of a block
// in byte j / 4 at bits 2 * (j % 4).  An activation block of 32 values is x = d_x * a with a = round(x / d_x) in
// -127..127.  Over one 32-value half-block
//     sum w x = d_w * d_x * (sum q a - sum a),
// so the int8 product takes the codes as they are (0..3 is a valid s8) and the offset is the activations' own sum,
// stored with the block.  The dot product (|sum q a| <= 32 * 3 * 127 < 2^22) becomes a float exactly with one
// integer add: as_float(0x4B400000 + dot) = 1.5 * 2^23 + dot, and the block stores c = -(1.5 * 2^23 + sum a), so
// as_float(0x4B400000 + dot) + c = dot - sum a with no rounding.  Per output and 64-block that is
//     acc += d_w * (d_x0 * t0 + d_x1 * t1)        (t = dot - sum a of each half)
// - two integer adds, two float adds and three multiply-adds.
//
// The K order.  mma.sync m16n8k32 takes four consecutive k of a row (A) or column (B) per register.  Four consecutive
// weights are one code byte; four codes of one register instead come from the same 2-bit field of four consecutive
// bytes - (word >> 2t) & 0x03030303 holds weights t, t+4, t+8, t+12 of a 16-weight word.  So the kernels use that order
// for the k of a half-block: mma k index 16h + 4t + j stands for weight (and activation) 16h + t + 4j.  The
// activations are stored in it (perm32), and a lane's two B registers (h = 0 and 1) are adjacent: one 8-byte load.
#include "strata/prefill/moe_fused.hpp"

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <mutex>

namespace strata::prefill::fused {
namespace {

void ck(cudaError_t e, const char* what) {
    if (e != cudaSuccess) {
        std::fprintf(stderr, "prefill fused experts: %s: %s\n", what, cudaGetErrorString(e));
        std::exit(1);
    }
}

constexpr int AB = 80;                  // activation bytes per 64 values: 64 codes, then {d0, c0, d1, c1}
constexpr int MAGIC = 0x4B400000;       // the bits of 1.5 * 2^23
constexpr float MAGICF = 12582912.0f;
// The Strata Q2_0 blob (moe_mmq.cu strata_q2_kernel, kernels.cu blob_dequant_kernel): gate/up codes [1280][640 B]
// (rows interleaved: gate of feature f at row 2f, its up at 2f + 1), down codes [2560][160 B], gate/up scales
// [1280][40] fp16, down scales [2560][10] fp16.
constexpr size_t O_GU_CODES = 0, O_D_CODES = (size_t) 1280 * 640, O_GU_SC = O_D_CODES + (size_t) 2560 * 160,
                 O_D_SC = O_GU_SC + (size_t) 1280 * 40 * 2;
constexpr int THREADS = 512;            // 16 warps: 4 along the weight rows (64 each) x 4 along the tile rows (16 each)
constexpr int WROWS = 256;              // weight rows per work item: 128 features of gate/up, or 256 outputs of down
constexpr int STAGES = 4;               // cp.async pipeline depth, one 64-weight block of K per stage
constexpr int STAGE_BYTES = WROWS * 16 + kTileRows * AB;
constexpr size_t smem_bytes(bool gu) {
    return (size_t) WROWS * (gu ? 40 : 10) * 2 + STAGES * STAGE_BYTES + kTileRows * 4;
}

// the stored position of value k of a 32-value half-block (see "The K order")
__host__ __device__ __forceinline__ int perm32(int k) { return 8 * (k & 3) + 4 * (k >> 4) + ((k & 15) >> 2); }

struct Tables {
    int32_t *cnt, *off, *fill, *ts;      // per expert: rows, first row (E + 1), placement cursor, first tile (E + 1)
    int2* tiles;                         // {expert, first row}, in expert order
};
size_t tiles_at(int n_expert) { return ((size_t) (4 * n_expert + 2) * 4 + 15) / 16 * 16; }
Tables tables(void* scratch, int n_expert) {
    int32_t* p = (int32_t*) scratch;
    return {p, p + n_expert, p + 2 * n_expert + 1, p + 3 * n_expert + 1,
            (int2*) ((uint8_t*) scratch + tiles_at(n_expert))};
}

// ---- activations: one warp per 64 values
__global__ void quant_act_kernel(const float* __restrict__ x, int64_t nblk, uint8_t* __restrict__ xa) {
    const int64_t w = (int64_t) blockIdx.x * (blockDim.x / 32) + threadIdx.x / 32;
    const int lane = threadIdx.x & 31;
    if (w >= nblk) return;
    const float v0 = x[w * 64 + lane], v1 = x[w * 64 + 32 + lane];
    float a0 = fabsf(v0), a1 = fabsf(v1);
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) {
        a0 = fmaxf(a0, __shfl_xor_sync(0xffffffffu, a0, o));
        a1 = fmaxf(a1, __shfl_xor_sync(0xffffffffu, a1, o));
    }
    const int q0 = a0 > 0.0f ? __float2int_rn(v0 * (127.0f / a0)) : 0;
    const int q1 = a1 > 0.0f ? __float2int_rn(v1 * (127.0f / a1)) : 0;
    int s0 = q0, s1 = q1;
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) {
        s0 += __shfl_xor_sync(0xffffffffu, s0, o);
        s1 += __shfl_xor_sync(0xffffffffu, s1, o);
    }
    uint8_t* out = xa + w * AB;
    out[perm32(lane)] = (uint8_t) (int8_t) q0;
    out[32 + perm32(lane)] = (uint8_t) (int8_t) q1;
    if (lane == 0)
        *(float4*) (out + 64) = make_float4(a0 / 127.0f, -(MAGICF + (float) s0), a1 / 127.0f, -(MAGICF + (float) s1));
}

// ---- grouping: counts, offsets and tiles, placement
__global__ void count_kernel(const int32_t* __restrict__ ids, int64_t n, int E, int32_t* __restrict__ cnt) {
    extern __shared__ int32_t hist[];
    for (int e = threadIdx.x; e < E; e += blockDim.x) hist[e] = 0;
    __syncthreads();
    for (int64_t i = (int64_t) blockIdx.x * blockDim.x + threadIdx.x; i < n; i += (int64_t) gridDim.x * blockDim.x) {
        const int e = ids[i];
        if ((unsigned) e < (unsigned) E) atomicAdd(&hist[e], 1);
    }
    __syncthreads();
    for (int e = threadIdx.x; e < E; e += blockDim.x)
        if (hist[e]) atomicAdd(&cnt[e], hist[e]);
}

// exclusive prefix sum over the block (blockDim a multiple of 32); *total = the block's sum
__device__ int block_excl_sum(int v, int* total, int* sh) {
    const int lane = threadIdx.x & 31, w = threadIdx.x >> 5, nw = blockDim.x >> 5;
    int x = v;
#pragma unroll
    for (int o = 1; o < 32; o <<= 1) {
        const int y = __shfl_up_sync(0xffffffffu, x, o);
        if (lane >= o) x += y;
    }
    if (lane == 31) sh[w] = x;
    __syncthreads();
    if (w == 0) {
        int s = lane < nw ? sh[lane] : 0;
#pragma unroll
        for (int o = 1; o < 32; o <<= 1) {
            const int y = __shfl_up_sync(0xffffffffu, s, o);
            if (lane >= o) s += y;
        }
        sh[lane] = s;
    }
    __syncthreads();
    const int r = x - v + (w > 0 ? sh[w - 1] : 0);
    *total = sh[nw - 1];
    __syncthreads();   // sh is reused by the next call
    return r;
}

// one block, a thread per expert (E <= 1024)
__global__ void scan_kernel(Tables tb, int E) {
    __shared__ int sh[32];
    const int e = threadIdx.x;
    const int c = e < E ? tb.cnt[e] : 0, nt = (c + kTileRows - 1) / kTileRows;
    int rows = 0, tiles = 0;
    const int o = block_excl_sum(c, &rows, sh), to = block_excl_sum(nt, &tiles, sh);
    if (e < E) {
        tb.off[e] = o;
        tb.fill[e] = o;
        tb.ts[e] = to;
        for (int i = 0; i < nt; ++i) tb.tiles[to + i] = make_int2(e, o + i * kTileRows);
    }
    if (e == 0) {
        tb.off[E] = rows;
        tb.ts[E] = tiles;
    }
}

__global__ void place_kernel(const int32_t* __restrict__ ids, int64_t n, int k, int E, Tables tb,
                             int32_t* __restrict__ slot, int32_t* __restrict__ src) {
    const int64_t i = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const int e = ids[i];
    if ((unsigned) e >= (unsigned) E) { slot[i] = 0; return; }   // (the router never writes one)
    const int p = atomicAdd(&tb.fill[e], 1);
    slot[i] = p;
    src[p] = (int32_t) (i / k);
}

// ---- the expert products
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
__device__ __forceinline__ void cp16(void* dst, const void* src) {
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" ::"r"((unsigned) __cvta_generic_to_shared(dst)),
                 "l"(src));
}
__device__ __forceinline__ void cp4(void* dst, const void* src) {
    asm volatile("cp.async.ca.shared.global [%0], [%1], 4;\n" ::"r"((unsigned) __cvta_generic_to_shared(dst)),
                 "l"(src));
}
__device__ __forceinline__ void cp_commit() { asm volatile("cp.async.commit_group;\n" ::); }
template <int N> __device__ __forceinline__ void cp_wait() { asm volatile("cp.async.wait_group %0;\n" ::"n"(N)); }
// d = A (16 x 32 s8, row) * B (32 x 8 s8, col), int32, from zero
__device__ __forceinline__ void mma_s8(int (&d)[4], const uint32_t (&a)[4], uint32_t b0, uint32_t b1) {
    asm("mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%10,%10,%10,%10};\n"
        : "=r"(d[0]), "=r"(d[1]), "=r"(d[2]), "=r"(d[3])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1), "r"(0));
}
#endif

// One work item = (tile of up to 64 routed rows of expert e, a block of 256 weight rows); a persistent grid walks
// the batch's items (their count is on the device).  GU: gate/up rows of 128 features against the rows' tokens'
// activations (`act` per token, 2560 values), SwiGLU, H to int8 per 32 features into `out` (per row, 640 values).
// !GU: down rows (256 outputs) against H (`act` per row), FP32 into `dm` per row.
// Warp (wf, wt): weight rows 64 wf.. as four m16 tiles (GU: a tile = 8 features, its gate rows as mma rows 0-7 and
// its up rows as 8-15, so a lane holds gate and up of one feature: SwiGLU in registers), tile rows 16 wt.. as two n8.
template <bool GU>
__global__ void __launch_bounds__(THREADS, 1)
expert_kernel(const Batch b, const Tables tb, const uint8_t* __restrict__ act, const int32_t* __restrict__ src,
              uint8_t* __restrict__ out, float* __restrict__ dm) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
    constexpr int NKB = GU ? 40 : 10;                 // 64-weight blocks along K
    constexpr int NFB = GU ? 5 : 10;                  // weight-row blocks per tile
    constexpr int CODE_LD = GU ? 640 : 160;           // code bytes per weight row
    constexpr size_t O_CODES = GU ? O_GU_CODES : O_D_CODES, O_SC = GU ? O_GU_SC : O_D_SC;
    constexpr int ACT_LD = NKB * AB;                  // bytes per activation row: 3200 (a token), 800 (a row's H)
    constexpr unsigned M2 = 0x03030303u;
    extern __shared__ __align__(16) uint8_t smem[];
    const __half* wsc = (const __half*) smem;                     // [256][NKB] the work item's weight scales
    uint8_t* stages = smem + WROWS * NKB * 2;                     // [STAGES] x {codes [256][16], act [64][AB]}
    int* srow = (int*) (stages + STAGES * STAGE_BYTES);           // [64] the activation row of each tile row

    const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5, g = lane >> 2, tig = lane & 3;
    const int wf = warp & 3, nb0 = 16 * (warp >> 2);
    int ra[4], rb[4];                                             // the local weight rows of mma rows g and g + 8
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        if (GU) { ra[i] = 2 * (32 * wf + 8 * i + g); rb[i] = ra[i] + 1; }
        else { ra[i] = 64 * wf + 16 * i + g; rb[i] = ra[i] + 8; }
    }
    const int t0 = tb.ts[b.e0], nwork = (tb.ts[b.e1] - t0) * NFB;
    for (int w = blockIdx.x; w < nwork; w += gridDim.x) {
        const int2 tl = tb.tiles[t0 + w / NFB];
        const int fb = w % NFB, e = tl.x, row0 = tl.y, nrows = min(kTileRows, tb.off[e + 1] - row0);
        const uint8_t* blob = b.blob[e - b.e0];
        const int rbase = fb * WROWS;
        __syncthreads();                                          // the previous item is done with the buffers
        if (tid < kTileRows) {
            const int r = row0 + min(tid, nrows - 1);             // rows past the tile's end repeat its last one
            srow[tid] = GU ? src[r] : r;
        }
        for (int c = tid; c < WROWS * 5; c += THREADS) {          // the scales: 5 x 16 B (GU) or 5 x 4 B a row
            const int r = c / 5, q = c % 5;
            if (GU) cp16(smem + r * 80 + q * 16, blob + O_SC + (size_t) (rbase + r) * 80 + q * 16);
            else cp4(smem + r * 20 + q * 4, blob + O_SC + (size_t) (rbase + r) * 20 + q * 4);
        }
        __syncthreads();                                          // srow
        auto load = [&](int kb) {
            uint8_t* st = stages + (kb % STAGES) * STAGE_BYTES;
            if (tid < WROWS) cp16(st + tid * 16, blob + O_CODES + (size_t) (rbase + tid) * CODE_LD + kb * 16);
            const int a = tid - (THREADS - kTileRows * 5);        // the last 320 threads: 64 rows x 5 x 16 B
            if (a >= 0) {
                const int r = a / 5, q = a % 5;
                cp16(st + WROWS * 16 + r * AB + q * 16, act + (size_t) srow[r] * ACT_LD + kb * AB + q * 16);
            }
        };
#pragma unroll
        for (int s = 0; s < STAGES - 1; ++s) {
            if (s < NKB) load(s);
            cp_commit();
        }
        // a warp whose rows are all past the tile's end only takes part in the loads
        const bool on0 = nb0 < nrows, on1 = nb0 + 8 < nrows;
        float acc[4][2][4];
#pragma unroll
        for (int i = 0; i < 4; ++i)
#pragma unroll
            for (int n = 0; n < 2; ++n)
#pragma unroll
                for (int q = 0; q < 4; ++q) acc[i][n][q] = 0.0f;
        for (int kb = 0; kb < NKB; ++kb) {
            cp_wait<STAGES - 2>();
            __syncthreads();
            if (kb + STAGES - 1 < NKB) load(kb + STAGES - 1);
            cp_commit();
            if (!on0) continue;
            const uint8_t* st = stages + (kb % STAGES) * STAGE_BYTES;
            const uint8_t* sa = st + WROWS * 16;
            uint2 bq[2][2];                                       // [n8 tile][half]: the B registers
            float4 sc[2][2];                                      // [n8 tile][column 2 tig + c]: {d0, c0, d1, c1}
#pragma unroll
            for (int n = 0; n < 2; ++n) {
                if (n == 1 && !on1) break;
                const uint8_t* r = sa + (nb0 + 8 * n + g) * AB + 8 * tig;
                bq[n][0] = *(const uint2*) r;
                bq[n][1] = *(const uint2*) (r + 32);
                sc[n][0] = *(const float4*) (sa + (nb0 + 8 * n + 2 * tig) * AB + 64);
                sc[n][1] = *(const float4*) (sa + (nb0 + 8 * n + 2 * tig + 1) * AB + 64);
            }
#pragma unroll
            for (int i = 0; i < 4; ++i) {
                const uint4 ca = *(const uint4*) (st + ra[i] * 16), cb = *(const uint4*) (st + rb[i] * 16);
                const float da = __half2float(wsc[ra[i] * NKB + kb]), db = __half2float(wsc[rb[i] * NKB + kb]);
                const int s = 2 * tig;
                const uint32_t a0[4] = {(ca.x >> s) & M2, (cb.x >> s) & M2, (ca.y >> s) & M2, (cb.y >> s) & M2};
                const uint32_t a1[4] = {(ca.z >> s) & M2, (cb.z >> s) & M2, (ca.w >> s) & M2, (cb.w >> s) & M2};
#pragma unroll
                for (int n = 0; n < 2; ++n) {
                    if (n == 1 && !on1) break;
                    int d0[4], d1[4];
                    mma_s8(d0, a0, bq[n][0].x, bq[n][0].y);
                    mma_s8(d1, a1, bq[n][1].x, bq[n][1].y);
#pragma unroll
                    for (int q = 0; q < 4; ++q) {
                        const float4 z = sc[n][q & 1];
                        const float u0 = __int_as_float(MAGIC + d0[q]) + z.y, u1 = __int_as_float(MAGIC + d1[q]) + z.w;
                        acc[i][n][q] = fmaf(q < 2 ? da : db, fmaf(z.z, u1, z.x * u0), acc[i][n][q]);
                    }
                }
            }
        }
        if (!on0) continue;
        if (GU) {
            // SwiGLU of feature 32 wf + 8 i + g (lane g of m tile i), tile rows nb0 + 8 n + 2 tig + c; H to int8 per
            // row over the warp's 32 features (= one 32-value half-block of the down product's K)
            const int blk = 4 * fb + wf;
#pragma unroll
            for (int n = 0; n < 2; ++n) {
                if (n == 1 && !on1) break;
#pragma unroll
                for (int c = 0; c < 2; ++c) {
                    float h[4], am = 0.0f;
#pragma unroll
                    for (int i = 0; i < 4; ++i) {
                        const float gt = acc[i][n][c], up = acc[i][n][2 + c];
                        h[i] = gt / (1.0f + __expf(-gt)) * up;
                        am = fmaxf(am, fabsf(h[i]));
                    }
#pragma unroll
                    for (int o = 4; o < 32; o <<= 1) am = fmaxf(am, __shfl_xor_sync(0xffffffffu, am, o));
                    const float inv = am > 0.0f ? 127.0f / am : 0.0f;
                    int qv[4], sum = 0;
#pragma unroll
                    for (int i = 0; i < 4; ++i) { qv[i] = __float2int_rn(h[i] * inv); sum += qv[i]; }
#pragma unroll
                    for (int o = 4; o < 32; o <<= 1) sum += __shfl_xor_sync(0xffffffffu, sum, o);
                    const int r = nb0 + 8 * n + 2 * tig + c;
                    if (r < nrows) {
                        uint8_t* o = out + (size_t) (row0 + r) * (10 * AB) + (blk >> 1) * AB;
                        const int hh = blk & 1;
#pragma unroll
                        for (int i = 0; i < 4; ++i) o[32 * hh + perm32(8 * i + g)] = (uint8_t) (int8_t) qv[i];
                        if (g == 0) *(float2*) (o + 64 + 8 * hh) = make_float2(am / 127.0f, -(MAGICF + (float) sum));
                    }
                }
            }
        } else {
#pragma unroll
            for (int i = 0; i < 4; ++i)
#pragma unroll
                for (int n = 0; n < 2; ++n) {
                    if (n == 1 && !on1) break;
#pragma unroll
                    for (int q = 0; q < 4; ++q) {
                        const int r = nb0 + 8 * n + 2 * tig + (q & 1);
                        if (r < nrows) dm[(size_t) (row0 + r) * 2560 + rbase + (q < 2 ? ra[i] : rb[i])] = acc[i][n][q];
                    }
                }
        }
    }
#endif
}

unsigned blocks(int64_t n, int per) { return (unsigned) ((n + per - 1) / per); }

struct DevInfo {
    bool done = false;
    int cc = 0, sms = 0, occ_gu = 0, occ_d = 0;
};
std::mutex g_mu;
DevInfo g_dev[32];

const DevInfo& dev_info() {
    int dev = 0;
    cudaGetDevice(&dev);
    std::lock_guard<std::mutex> lk(g_mu);
    DevInfo& d = g_dev[dev & 31];
    if (d.done) return d;
    d.done = true;
    int major = 0, minor = 0;
    cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, dev);
    cudaDeviceGetAttribute(&minor, cudaDevAttrComputeCapabilityMinor, dev);
    cudaDeviceGetAttribute(&d.sms, cudaDevAttrMultiProcessorCount, dev);
    d.cc = 10 * major + minor;
    if (d.cc < 80) return d;
    // the device code of this card's arch must have the kernels (a build without an sm_80+ target has an empty body)
    cudaFuncAttributes fa{};
    if (cudaFuncGetAttributes(&fa, expert_kernel<true>) != cudaSuccess || fa.ptxVersion < 80) {
        cudaGetLastError();
        d.cc = 0;
        return d;
    }
    ck(cudaFuncSetAttribute(expert_kernel<true>, cudaFuncAttributeMaxDynamicSharedMemorySize, (int) smem_bytes(true)),
       "smem gu");
    ck(cudaFuncSetAttribute(expert_kernel<false>, cudaFuncAttributeMaxDynamicSharedMemorySize, (int) smem_bytes(false)),
       "smem down");
    cudaOccupancyMaxActiveBlocksPerMultiprocessor(&d.occ_gu, expert_kernel<true>, THREADS, smem_bytes(true));
    cudaOccupancyMaxActiveBlocksPerMultiprocessor(&d.occ_d, expert_kernel<false>, THREADS, smem_bytes(false));
    if (d.occ_gu < 1 || d.occ_d < 1) d.cc = 0;                   // does not fit this card: the MMQ path
    cudaGetLastError();
    return d;
}

}  // namespace

bool built() { return true; }
bool available() { return dev_info().cc >= 80; }
// 0.1.36: on by default for the Q2_0 pack (+13-23% prompts on an RTX 5070; teacher-forced against the FP16 path as
// close as MMQ at 8K and closer at 32K: phaseA-tests s18-tf); STRATA_PF_FUSED=0 keeps MMQ.  The native packs' kernels
// (moe_fused_iq) stay opt-in: STRATA_PF_FUSED=1 (requested()).
bool enabled() {
    static const bool env = [] {
        const char* v = std::getenv("STRATA_PF_FUSED");
        return v == nullptr || v[0] != '0';
    }();
    return env && available();
}
bool requested() {
    static const bool env = [] {
        const char* v = std::getenv("STRATA_PF_FUSED");
        return v != nullptr && v[0] == '1';
    }();
    return env && available();
}

size_t act_bytes(int64_t rows, int64_t cols) { return (size_t) rows * (size_t) (cols / 64) * AB; }
size_t group_bytes(int64_t n, int n_expert) {
    return tiles_at(n_expert) + ((size_t) (n + kTileRows - 1) / kTileRows + (size_t) n_expert) * sizeof(int2);
}

void quantize_act(const float* x, int64_t rows, int64_t cols, void* xa, void* stream) {
    if (rows <= 0) return;
    const int64_t nblk = rows * (cols / 64);
    quant_act_kernel<<<blocks(nblk, 8), 256, 0, (cudaStream_t) stream>>>(x, nblk, (uint8_t*) xa);
    ck(cudaGetLastError(), "quantize_act");
}

void group(const int32_t* ids, int64_t n, int k, int n_expert, void* scratch, int32_t* slot, int32_t* src,
           void* stream) {
    if (n_expert < 1 || n_expert > 1024) {
        std::fprintf(stderr, "prefill fused experts: %d experts (the grouping takes 1 to 1024)\n", n_expert);
        std::exit(1);
    }
    const cudaStream_t s = (cudaStream_t) stream;
    const Tables tb = tables(scratch, n_expert);
    ck(cudaMemsetAsync(tb.cnt, 0, (size_t) n_expert * 4, s), "group memset");
    if (n > 0) {
        const unsigned nb = std::min<unsigned>(blocks(n, 1024), 4u * (unsigned) std::max(dev_info().sms, 1));
        count_kernel<<<nb, 1024, (size_t) n_expert * 4, s>>>(ids, n, n_expert, tb.cnt);
    }
    scan_kernel<<<1, (unsigned) ((n_expert + 31) / 32 * 32), 0, s>>>(tb, n_expert);
    if (n > 0) place_kernel<<<blocks(n, 256), 256, 0, s>>>(ids, n, k, n_expert, tb, slot, src);
    ck(cudaGetLastError(), "group");
}

void experts(const Batch& b, int n_expert, int64_t n, const void* scratch, const void* xa, const int32_t* src,
             void* ha, float* dm, void* stream) {
    if (b.e1 <= b.e0 || n <= 0) return;
    const DevInfo& d = dev_info();
    const cudaStream_t s = (cudaStream_t) stream;
    const Tables tb = tables(const_cast<void*>(scratch), n_expert);
    // the most tiles the batch can have: every row in it, plus a partial tile per expert
    const int64_t tiles = (n + kTileRows - 1) / kTileRows + (b.e1 - b.e0);
    const unsigned g_gu = (unsigned) std::min<int64_t>(tiles * 5, (int64_t) d.sms * d.occ_gu);
    const unsigned g_d = (unsigned) std::min<int64_t>(tiles * 10, (int64_t) d.sms * d.occ_d);
    expert_kernel<true><<<g_gu, THREADS, smem_bytes(true), s>>>(b, tb, (const uint8_t*) xa, src, (uint8_t*) ha,
                                                                nullptr);
    expert_kernel<false><<<g_d, THREADS, smem_bytes(false), s>>>(b, tb, (const uint8_t*) ha, src, nullptr, dm);
    ck(cudaGetLastError(), "experts");
}

}  // namespace strata::prefill::fused
