// engine/src/inkling2/cuda/ink_moe_fp4_sm120.cu - Clean-room MIT Blackwell (sm_120a / sm_121a) native mxf4nvf4 & mxf8f6f4 MoE kernels
//
// MIT License
// Copyright (c) 2026 Niko1221 and the Strata contributors
//
// Targets NVIDIA PTX ISA 8.7+ block-scaled Tensor Core MMA instructions on GeForce RTX 50-series / RTX PRO Blackwell (sm_120a)
// and DGX Spark GB10 (sm_121a):
//   - SF4 (NVFP4, L0..19 = 44/48 visits):
//       mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X.m16n8k64.row.col.f32.e2m1.e2m1.f32.ue4m3
//   - MX8 (MXFP8, L20..23 = 4/48 visits):
//       mma.sync.aligned.kind::mxf8f6f4.block_scale.scale_vec::1X.m16n8k32.row.col.f32.e4m3.e4m3.f32.ue8m0
//
// Zero bit-reordering needed! .strata v1.1 SF4 stores sequential E2M1x2 nibbles (lo=even, hi=odd, 16-byte aligned rows:
// 2304/2 = 1152 B and 1152/2 = 576 B, both multiples of 32 B!) with contiguous 4-byte-aligned F8_E4M3 scales per 16 weights.
//
// Activation FP4 slot (48 B per 64 values, 16-byte aligned for cp.async):
//   bytes [ 0..31]: uint8_t qs[32]   (64 sequential E2M1 values, 2 per byte: lo=2j, hi=2j+1)
//   bytes [32..35]: uint8_t d[4]     (4 UE4M3 sub-block scales for sub = 0..3)
//   bytes [36..39]: float   g_blk    (per-64-element FP32 global scale = amax_64 / (6 * 448))
//   bytes [40..47]: padding
#include "ink_moe_fp4_sm120.cuh"
#include "ink_cuda.cuh"
#include "ink_sm120_cfg.cuh"
#include "strata/core/ink_caps.hpp"

#if defined(CUDART_VERSION) && (CUDART_VERSION >= 12080)
#include <cuda_fp4.h>
#endif
#include <cuda_fp8.h>

#include <algorithm>
#include <cstdlib>

namespace strata::ink::sm120 {
namespace {

constexpr int kHidden = kInkHidden;  // 2304 = 36 * 64 = 72 * 32
constexpr int kInter  = kInkInter;   // 1152 = 18 * 64 = 36 * 32
constexpr int kAB     = kActFp4SlotBytes;  // 48 B per 64 values

constexpr int kNT = 256;    // 8 warps: wf in 0..1 (64 weight rows each), wn in 0..3 (16 token rows each)
constexpr int kTR = 64;     // routed token rows per tile
constexpr int kWR = 128;    // weight rows per CTA (GU: 64 interleaved gate+up pairs = 64 SITU outputs; DN: 128 outputs)
[[maybe_unused]] constexpr int kWStride4 = 48; // 32B qs + 16B pad (bank-conflict free ldmatrix.m8n8.x4)
[[maybe_unused]] constexpr int kAStride4 = 48; // 32B qs + 16B pad

#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1200) && (__CUDA_ARCH__ < 1300)
#define INK_SM120_DEV 1
#endif

#if defined(INK_SM120_DEV)

__device__ __forceinline__ uint32_t smem_u32(const void* p) {
    return (uint32_t)__cvta_generic_to_shared(p);
}

__device__ __forceinline__ void cp16_async(void* dst_smem, const void* src_glob) {
    const uint32_t d = smem_u32(dst_smem);
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" :: "r"(d), "l"(src_glob));
}

__device__ __forceinline__ void cp8_async(void* dst_smem, const void* src_glob) {
    const uint32_t d = smem_u32(dst_smem);
    asm volatile("cp.async.ca.shared.global [%0], [%1], 8;\n" :: "r"(d), "l"(src_glob));
}

__device__ __forceinline__ void cp4_async(void* dst_smem, const void* src_glob) {
    const uint32_t d = smem_u32(dst_smem);
    asm volatile("cp.async.ca.shared.global [%0], [%1], 4;\n" :: "r"(d), "l"(src_glob));
}

// L2 prefetch of a contiguous byte range (size a multiple of 16): the weight rows are read 32 B per row and K64 step, 128 rows apart; asking
// DRAM for 256 contiguous bytes of a row at a time keeps the memory pages open for the following 8 steps.
__device__ __forceinline__ void pf_l2(const void* src_glob, uint32_t bytes) {
    asm volatile("cp.async.bulk.prefetch.L2.global [%0], %1;\n" :: "l"(src_glob), "r"(bytes) : "memory");
}

__device__ __forceinline__ void cp_commit() { asm volatile("cp.async.commit_group;\n"); }
__device__ __forceinline__ void cp_wait0()  { asm volatile("cp.async.wait_group 0;\n"); }
__device__ __forceinline__ void cp_wait1()  { asm volatile("cp.async.wait_group 1;\n"); }

__device__ __forceinline__ void ldsm4(uint32_t (&r)[4], const void* src_smem) {
    const uint32_t a = smem_u32(src_smem);
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                 : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3])
                 : "r"(a));
}

__device__ __forceinline__ void mma_nvfp4_m16n8k64(float (&d)[4], const uint32_t (&a)[4], uint32_t b0, uint32_t b1,
                                                   uint32_t sfa, uint32_t sfb) {
    asm volatile(
        "mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X.m16n8k64.row.col.f32.e2m1.e2m1.f32.ue4m3 "
        "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3}, "
        "%10, {0, 0}, %11, {0, 0};\n"
        : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1), "r"(sfa), "r"(sfb));
}

__device__ __forceinline__ void mma_mxfp8_m16n8k32(float (&d)[4], const uint32_t (&a)[4], uint32_t b0, uint32_t b1,
                                                   uint32_t sfa, uint32_t sfb) {
    asm volatile(
        "mma.sync.aligned.kind::mxf8f6f4.block_scale.scale_vec::1X.m16n8k32.row.col.f32.e4m3.e4m3.f32.ue8m0 "
        "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3}, "
        "%10, {0, 0}, %11, {0, 0};\n"
        : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1), "r"(sfa), "r"(sfb));
}

__device__ __forceinline__ uint8_t encode_ue4m3(float x, float& decoded) {
    if (!(x > 0.0f)) {
        decoded = 0.0f;
        return 0;
    }
    const float clamped = fminf(x, 448.0f);
    const __nv_fp8_e4m3 fp8(clamped);
    decoded = (float)fp8;
    return fp8.__x;
}

__device__ __forceinline__ uint8_t pack_e2m1x2(float v_lo, float v_hi) {
    const __nv_fp4x2_e2m1 q2(make_float2(v_lo, v_hi));
    return q2.__x;
}

__device__ __forceinline__ uint8_t encode_e4m3_signed(float x) {
    const float clamped = fminf(fmaxf(x, -448.0f), 448.0f);
    const __nv_fp8_e4m3 fp8(clamped);
    return fp8.__x;
}

// Compute UE8M0 exponent byte e so 2^(e-127) >= amax / 448.0f (smallest power-of-2 scale covering amax in E4M3).
__device__ __forceinline__ uint8_t encode_ue8m0(float amax, float& inv_scale) {
    if (!(amax > 0.0f)) {
        inv_scale = 0.0f;
        return 255u;  // zero scale sentinel
    }
    const float need = amax * (1.0f / 448.0f);
    uint32_t bits = __float_as_uint(need);
    // Round up to next power of 2 if mantissa != 0
    if (bits & 0x007FFFFFu) bits = (bits & 0xFF800000u) + 0x00800000u;
    int exp = (int)((bits >> 23) & 0xFFu);
    if (exp < 1) exp = 1;
    if (exp > 254) exp = 254;
    const float sc = __uint_as_float((uint32_t)exp << 23);
    inv_scale = 1.0f / sc;
    return (uint8_t)exp;
}

#endif  // INK_SM120_DEV

// ---- Activation FP4 quantizer: 1 warp per 64 FP32 values -> 48-byte slot (sequential E2M1x2 matching .strata SF4!)
__global__ void k_quant_act_fp4(const float* __restrict__ x, int64_t nblk, uint8_t* __restrict__ xa) {
#if defined(INK_SM120_DEV)
    const int64_t w = (int64_t)blockIdx.x * (blockDim.x >> 5) + (threadIdx.x >> 5);
    if (w >= nblk) return;
    const int lane = threadIdx.x & 31;
    const int sub = lane >> 3;   // 0..3 (four 16-element sub-blocks)
    const int j   = lane & 7;    // 0..7 -> elements 2*j and 2*j + 1 inside sub-block!

    const float* xb = x + w * 64;
    const float v0 = xb[16 * sub + 2 * j];
    const float v1 = xb[16 * sub + 2 * j + 1];

    float am_sub = fmaxf(fabsf(v0), fabsf(v1));
#pragma unroll
    for (int o = 1; o < 8; o <<= 1) am_sub = fmaxf(am_sub, __shfl_xor_sync(0xFFFFFFFFu, am_sub, o));

    float am_64 = fmaxf(am_sub, __shfl_xor_sync(0xFFFFFFFFu, am_sub, 8));
    am_64 = fmaxf(am_64, __shfl_xor_sync(0xFFFFFFFFu, am_64, 16));

    const float g_blk = am_64 * (1.0f / (6.0f * 448.0f));
    float s_sub = 0.0f;
    const uint8_t d_code = encode_ue4m3(g_blk > 0.0f ? (am_sub / (g_blk * 6.0f)) : 0.0f, s_sub);
    const float denom = g_blk * s_sub;
    const float inv = denom > 0.0f ? (1.0f / denom) : 0.0f;

    uint8_t* out = xa + w * kAB;
    out[lane] = pack_e2m1x2(v0 * inv, v1 * inv);
    if (j == 0) out[32 + sub] = d_code;
    if (lane == 0) *(float*)(out + 36) = g_blk;
#else
    (void)x; (void)nblk; (void)xa;
#endif
}

// ---- Per-row global scale FP4 activation quantizer (1 CTA/row, 256 threads):
// Stores single row scale g_row = amax_row / (6 * 448) in slot 0 bytes [36..39] so NVFP4 TC kernels
// accumulate directly into acc[i][n] with ZERO inner-loop d[4] / fmaf / s_agb overhead!
constexpr float kGHaConst = 100.0f / (6.0f * 448.0f);  // exact mathematical bound of situ_f32(beta=4, lb=25): |h| <= 100.0!

__global__ void __launch_bounds__(256) k_quant_act_fp4_row(
    const float* __restrict__ x, int rows, int cols, uint8_t* __restrict__ xa) {
#if defined(INK_SM120_DEV)
    const int row = blockIdx.x;
    if (row >= rows) return;
    const int tid  = threadIdx.x;
    const int warp = tid >> 5;
    const int lane = tid & 31;
    const float* xr = x + (size_t)row * cols;

    float am = 0.0f;
    const int cols4 = cols >> 2;
    const float4* x4 = reinterpret_cast<const float4*>(xr);
    for (int i = tid; i < cols4; i += 256) {
        const float4 v = x4[i];
        am = fmaxf(am, fmaxf(fmaxf(fabsf(v.x), fabsf(v.y)), fmaxf(fabsf(v.z), fabsf(v.w))));
    }
#pragma unroll
    for (int o = 16; o >= 1; o >>= 1) am = fmaxf(am, __shfl_xor_sync(0xFFFFFFFFu, am, o));
    __shared__ float s_wm[8];
    if (lane == 0) s_wm[warp] = am;
    __syncthreads();
    const float am_row = fmaxf(fmaxf(fmaxf(s_wm[0], s_wm[1]), fmaxf(s_wm[2], s_wm[3])),
                               fmaxf(fmaxf(s_wm[4], s_wm[5]), fmaxf(s_wm[6], s_wm[7])));
    const float g_row = am_row * (1.0f / (6.0f * 448.0f));
    const float inv_6g = (g_row > 0.0f) ? (1.0f / (g_row * 6.0f)) : 0.0f;

    const int nblk = cols >> 6;
    const int sub = lane >> 3;
    const int j   = lane & 7;
    uint8_t* row_out = xa + (size_t)row * nblk * kAB;
    for (int blk = warp; blk < nblk; blk += 8) {
        const float* xb = xr + blk * 64;
        const float v0 = xb[16 * sub + 2 * j];
        const float v1 = xb[16 * sub + 2 * j + 1];
        float am_sub = fmaxf(fabsf(v0), fabsf(v1));
#pragma unroll
        for (int o = 1; o < 8; o <<= 1) am_sub = fmaxf(am_sub, __shfl_xor_sync(0xFFFFFFFFu, am_sub, o));
        float s_sub = 0.0f;
        const uint8_t d_code = encode_ue4m3(am_sub * inv_6g, s_sub);
        const float denom = g_row * s_sub;
        const float inv = (denom > 0.0f) ? (1.0f / denom) : 0.0f;
        uint8_t* out = row_out + (size_t)blk * kAB;
        out[lane] = pack_e2m1x2(v0 * inv, v1 * inv);
        if (j == 0) out[32 + sub] = d_code;
        if (blk == 0 && lane == 0) *(float*)(out + 36) = g_row;
    }
#else
    (void)x; (void)rows; (void)cols; (void)xa;
#endif
}

// ---- Activation MX8 quantizer: 1 warp per 32 FP32 values -> 32B E4M3 + 1B UE8M0
__global__ void k_quant_act_mx8(const float* __restrict__ x, int64_t nblk,
                                uint8_t* __restrict__ xq, uint8_t* __restrict__ xsc) {
#if defined(INK_SM120_DEV)
    const int64_t w = (int64_t)blockIdx.x * (blockDim.x >> 5) + (threadIdx.x >> 5);
    if (w >= nblk) return;
    const int lane = threadIdx.x & 31;
    const float v = x[w * 32 + lane];
    float am = fabsf(v);
#pragma unroll
    for (int o = 16; o >= 1; o >>= 1) am = fmaxf(am, __shfl_xor_sync(0xFFFFFFFFu, am, o));
    float inv_sc = 0.0f;
    const uint8_t sc_byte = encode_ue8m0(am, inv_sc);
    xq[w * 32 + lane] = encode_e4m3_signed(v * inv_sc);
    if (lane == 0) xsc[w] = sc_byte;
#else
    (void)x; (void)nblk; (void)xq; (void)xsc;
#endif
}

// ---- Persistent Blackwell sm_120a mxf4nvf4 MoE kernel (TR in {16, 32, 64}; GU=true: Gate+Up+SITU+FP4-requant; GU=false: Down)
// TR=16 (n_wf=8, n_wn=1): 100% warp utilization for c=16..128 batched decode!
// TR=32 (n_wf=4, n_wn=2): 100% warp utilization for M=129..768 prefill (avg 32 tok/expert at M=512)!
// TR=64 (n_wf=2, n_wn=4): max weight reuse for M>768 prefill!
template <bool GU, int TR, bool ATOMIC_OUT = false>
__global__ void __launch_bounds__(kNT) k_moe_sf4_tc(
    const uint8_t* __restrict__ sec4_base,
    const InkExpertDev* __restrict__ experts,
    InkMoeTables tb,
    const uint8_t* __restrict__ act,
    uint8_t* __restrict__ out_ha,
    const float* __restrict__ topk_w,
    float* __restrict__ y_pair,
    int pf) {
#if defined(INK_SM120_DEV)
    constexpr int ROWS_N   = GU ? (2 * kInter) : kHidden;  // 2304
    constexpr int ROWS_K   = GU ? kHidden : kInter;        // 2304 (GU) or 1152 (DN)
    constexpr int NS       = ROWS_K / 64;                  // 36 (GU) or 18 (DN)
    constexpr int NFB      = ROWS_N / kWR;                 // 18 blocks of 128 rows
    constexpr int N_WN     = TR / 16;                      // 1, 2, or 4
    constexpr int N_WF     = 8 / N_WN;                     // 8, 4, or 2
    constexpr int W_PER_WF = kWR / N_WF;                   // 16, 32, or 64
    constexpr int NI       = W_PER_WF / 16;                // 1, 2, or 4
    constexpr size_t W_P_ROW   = ROWS_K / 2;
    constexpr size_t W_S_ROW   = ROWS_K / 16;
    constexpr size_t ACT_ROW_B = (size_t)NS * kAB;

    __shared__ alignas(16) uint8_t s_wqs[2][kWR * kWStride4];
    __shared__ alignas(16) uint8_t s_aqs[2][TR * kAStride4];
    __shared__ uint32_t s_wsc[2][kWR];
    __shared__ uint32_t s_asc[2][TR];
    __shared__ float    s_agb[2][TR];
    __shared__ int32_t  s_row[TR];
    __shared__ float    s_am[N_WF][TR];
    __shared__ float    s_rw[TR];

    const int tid  = threadIdx.x;
    const int warp = tid >> 5;
    const int lane = tid & 31;
    const int wf   = warp / N_WN;  // 0..N_WF-1
    const int wn   = warp % N_WN;  // 0..N_WN-1
    const int nb0  = 16 * wn;
    const int g    = lane >> 2;
    const int tig  = lane & 3;
    const int lm   = lane >> 3;
    const int l8   = lane & 7;
    const int tidx_a = g + (tig & 1) * 8;

    const int a_off = (W_PER_WF * wf + 8 * (lm & 1) + l8) * kWStride4 + 16 * (lm >> 1);
    const int b_off = (nb0 + 8 * (lm >> 1) + l8) * kAStride4 + 16 * (lm & 1);
    const int ur = tid >> 1;
    const int uj = tid & 1;

    const int total_tiles = *tb.n_tiles;
    const int nwork = total_tiles * NFB;

    for (int w = blockIdx.x; w < nwork; w += gridDim.x) {
        const int2 tl = tb.tiles[w / NFB];
        const int fb = w % NFB;
        const int e = tl.x;
        const int row0 = tl.y;
        const InkExpertDev ed = experts[e];
        if ((GU ? ed.gu_code : ed.dn_code) != FMT_SF4) continue;
        const int nrows = min(TR, tb.off[e + 1] - row0);
        if (nrows <= 0) continue;

        const uint8_t* body = sec4_base + ed.body_off;
        float hdr[4];
        memcpy(hdr, body, 16);
        const float w_outer = GU ? (hdr[0] / hdr[1]) : (hdr[2] / hdr[3]);

        const uint8_t* base_p = GU ? (body + 16) : (body + ed.dn_off);
        const uint8_t* base_s = base_p + (size_t)ROWS_N * W_P_ROW;

        const int phys_row = GU ? (2 * (fb * 64 + 8 * (ur >> 4) + (ur & 7)) + ((ur >> 3) & 1))
                                : (fb * kWR + ur);
        const uint8_t* wrow_p = base_p + (size_t)phys_row * W_P_ROW;
        const uint8_t* wrow_s = base_s + (size_t)phys_row * W_S_ROW;

        __syncthreads();
        if (tid < TR) {
            if (tid < nrows) {
                s_row[tid] = GU ? tb.tok_idx[row0 + tid] : (row0 + tid);
                if constexpr (!GU) s_rw[tid] = topk_w[tb.pairs[row0 + tid]];
            } else {
                s_row[tid] = -1;
                if constexpr (!GU) s_rw[tid] = 0.0f;
                s_asc[0][tid] = 0u; s_asc[1][tid] = 0u;
                s_agb[0][tid] = 0.0f; s_agb[1][tid] = 0.0f;
#pragma unroll
                for (int b = 0; b < 2; ++b) {
                    uint4* z = (uint4*)(s_aqs[b] + tid * kAStride4);
                    z[0] = make_uint4(0u, 0u, 0u, 0u);
                    z[1] = make_uint4(0u, 0u, 0u, 0u);
                }
            }
        }
        __syncthreads();

        auto issue_stage = [&](int st, int buf) {
            cp16_async(s_wqs[buf] + ur * kWStride4 + 16 * uj, wrow_p + (size_t)st * 32 + 16 * uj);
            if (uj == 0) s_wsc[buf][ur] = *(const uint32_t*)(wrow_s + (size_t)st * 4);
            if (tid < 2 * TR) {
                const int r = tid >> 1;
                const int h = tid & 1;
                if (r < nrows) {
                    const uint8_t* ap = act + (size_t)s_row[r] * ACT_ROW_B + (size_t)st * kAB;
                    cp16_async(s_aqs[buf] + r * kAStride4 + 16 * h, ap + 16 * h);
                }
            } else if (tid < 3 * TR) {
                const int r = tid - 2 * TR;
                if (r < nrows) {
                    const uint8_t* ap = act + (size_t)s_row[r] * ACT_ROW_B + (size_t)st * kAB;
                    const uint2 meta = *(const uint2*)(ap + 32);
                    s_asc[buf][r] = meta.x;
                    s_agb[buf][r] = __uint_as_float(meta.y);
                }
            }
            cp_commit();
        };

        if (pf && uj == 0) {
            pf_l2(wrow_p, (uint32_t)min((size_t)pf, W_P_ROW));
            if constexpr (GU) pf_l2(wrow_s, (uint32_t)W_S_ROW);
        }
        issue_stage(0, 0);

        float acc[NI][2][4] = {};
        const bool on0 = nb0 < nrows;
        const bool on1 = nb0 + 8 < nrows;

        for (int st = 0; st < NS; ++st) {
            const int buf = st & 1;
            const int nxt = buf ^ 1;
            if (pf && uj == 0 && (st % (pf / 32)) == 0 && (size_t)(st + pf / 32) * 32 < W_P_ROW) {
                const size_t nb = (size_t)(st + pf / 32) * 32;
                pf_l2(wrow_p + nb, (uint32_t)min((size_t)pf, W_P_ROW - nb));
            }
            if (st + 1 < NS) {
                issue_stage(st + 1, nxt);
                cp_wait1();
            } else {
                cp_wait0();
            }
            __syncthreads();

            if (on0) {
                uint32_t bq[4];
                ldsm4(bq, s_aqs[buf] + b_off);
                const uint32_t sfb0 = s_asc[buf][nb0 + g];
                const uint32_t sfb1 = on1 ? s_asc[buf][nb0 + 8 + g] : 0u;
                const float gb00 = s_agb[buf][nb0 + 2 * tig + 0];
                const float gb01 = s_agb[buf][nb0 + 2 * tig + 1];
                const float gb10 = on1 ? s_agb[buf][nb0 + 8 + 2 * tig + 0] : 0.0f;
                const float gb11 = on1 ? s_agb[buf][nb0 + 8 + 2 * tig + 1] : 0.0f;

#pragma unroll
                for (int i = 0; i < NI; ++i) {
                    uint32_t a[4];
                    ldsm4(a, s_wqs[buf] + a_off + 16 * i * kWStride4);
                    const uint32_t sfa = s_wsc[buf][W_PER_WF * wf + 16 * i + tidx_a];

                    float d0[4] = {};
                    mma_nvfp4_m16n8k64(d0, a, bq[0], bq[1], sfa, sfb0);
                    acc[i][0][0] = fmaf(d0[0], gb00, acc[i][0][0]);
                    acc[i][0][1] = fmaf(d0[1], gb01, acc[i][0][1]);
                    acc[i][0][2] = fmaf(d0[2], gb00, acc[i][0][2]);
                    acc[i][0][3] = fmaf(d0[3], gb01, acc[i][0][3]);

                    if (on1) {
                        float d1[4] = {};
                        mma_nvfp4_m16n8k64(d1, a, bq[2], bq[3], sfa, sfb1);
                        acc[i][1][0] = fmaf(d1[0], gb10, acc[i][1][0]);
                        acc[i][1][1] = fmaf(d1[1], gb11, acc[i][1][1]);
                        acc[i][1][2] = fmaf(d1[2], gb10, acc[i][1][2]);
                        acc[i][1][3] = fmaf(d1[3], gb11, acc[i][1][3]);
                    }
                }
            }
            if (st + 1 < NS) __syncthreads();
        }

        if constexpr (GU) {
            float h_val[2][2][NI] = {};
            float am_sub0[2][2] = {}, am_sub1[2][2] = {};
            if (on0) {
#pragma unroll
                for (int n = 0; n < 2; ++n) {
                    if (n == 1 && !on1) break;
#pragma unroll
                    for (int c = 0; c < 2; ++c) {
#pragma unroll
                        for (int i = 0; i < NI; ++i) {
                            const float gt = acc[i][n][c] * w_outer;
                            const float up = acc[i][n][2 + c] * w_outer;
                            h_val[n][c][i] = situ_f32(gt, up);
                        }
                        float a0 = fabsf(h_val[n][c][0]);
                        if constexpr (NI >= 2) a0 = fmaxf(a0, fabsf(h_val[n][c][1]));
                        float a1 = 0.0f;
                        if constexpr (NI == 4) a1 = fmaxf(fabsf(h_val[n][c][2]), fabsf(h_val[n][c][3]));
#pragma unroll
                        for (int o = 4; o < 32; o <<= 1) {
                            a0 = fmaxf(a0, __shfl_xor_sync(0xFFFFFFFFu, a0, o));
                            if constexpr (NI == 4) a1 = fmaxf(a1, __shfl_xor_sync(0xFFFFFFFFu, a1, o));
                        }
                        am_sub0[n][c] = a0;
                        am_sub1[n][c] = a1;
                        const int r = nb0 + 8 * n + 2 * tig + c;
                        if (g == 0 && r < TR) s_am[wf][r] = (NI == 4) ? fmaxf(a0, a1) : a0;
                    }
                }
            }
            __syncthreads();
            if (on0) {
                constexpr size_t H_ROW_B = (size_t)(kInter / 64) * kAB;
#pragma unroll
                for (int n = 0; n < 2; ++n) {
                    if (n == 1 && !on1) break;
#pragma unroll
                    for (int c = 0; c < 2; ++c) {
                        const int r = nb0 + 8 * n + 2 * tig + c;
                        const bool valid = (r < nrows);
                        float am_64 = 0.0f;
                        if (valid) {
#pragma unroll
                            for (int k = 0; k < N_WF; ++k) am_64 = fmaxf(am_64, s_am[k][r]);
                        }
                        const float g_blk = am_64 * (1.0f / (6.0f * 448.0f));
                        float s0 = 0.0f, s1 = 0.0f;
                        float sub0_am = am_sub0[n][c];
                        if constexpr (NI == 1) {
                            sub0_am = valid ? fmaxf(s_am[wf & ~1][r], s_am[wf | 1][r]) : 0.0f;
                        }
                        const uint8_t d0 = encode_ue4m3(g_blk > 0.0f ? (sub0_am / (g_blk * 6.0f)) : 0.0f, s0);
                        const uint8_t d1 = (NI == 4) ? encode_ue4m3(g_blk > 0.0f ? (am_sub1[n][c] / (g_blk * 6.0f)) : 0.0f, s1) : 0u;
                        const float inv0 = (g_blk * s0) > 0.0f ? (1.0f / (g_blk * s0)) : 0.0f;
                        const float inv1 = (NI == 4 && (g_blk * s1) > 0.0f) ? (1.0f / (g_blk * s1)) : 0.0f;

                        // Unconditional __shfl_xor_sync across all 32 lanes (deadlock-free when nrows % 8 != 0)!
                        float qa[NI], qb[NI];
#pragma unroll
                        for (int i = 0; i < NI; ++i) {
                            qa[i] = h_val[n][c][i] * ((i < 2) ? inv0 : inv1);
                            qb[i] = __shfl_xor_sync(0xFFFFFFFFu, qa[i], 4);
                        }
                        if (valid) {
                            uint8_t* dst_slot = out_ha + (size_t)(row0 + r) * H_ROW_B + (size_t)fb * kAB;
                            if ((g & 1) == 0) {
                                const int bo = (4 * NI) * wf + (g >> 1);
#pragma unroll
                                for (int i = 0; i < NI; ++i) dst_slot[bo + 4 * i] = pack_e2m1x2(qa[i], qb[i]);
                            }
                            if (g == 0) {
                                if constexpr (NI == 4) {
                                    dst_slot[32 + 2 * wf + 0] = d0;
                                    dst_slot[32 + 2 * wf + 1] = d1;
                                } else if constexpr (NI == 2) {
                                    dst_slot[32 + wf] = d0;
                                } else {
                                    if ((wf & 1) == 0) dst_slot[32 + (wf >> 1)] = d0;
                                }
                                if (wf == 0) *(float*)(dst_slot + 36) = g_blk;
                            }
                        }
                    }
                }
            }
        } else if (on0) {
            const int rbase = fb * kWR + W_PER_WF * wf;
#pragma unroll
            for (int n = 0; n < 2; ++n) {
                if (n == 1 && !on1) break;
#pragma unroll
                for (int c = 0; c < 2; ++c) {
                    const int r = nb0 + 8 * n + 2 * tig + c;
                    if (r < nrows) {
                        const float sc = w_outer * s_rw[r];
                        if constexpr (ATOMIC_OUT) {
                            float* dst = y_pair + (size_t)(tb.pairs[row0 + r] >> 3) * kHidden;
#pragma unroll
                            for (int i = 0; i < NI; ++i) {
                                atomicAdd(&dst[rbase + 16 * i + g + 0], acc[i][n][c] * sc);
                                atomicAdd(&dst[rbase + 16 * i + g + 8], acc[i][n][2 + c] * sc);
                            }
                        } else {
                            float* dst = y_pair + (size_t)tb.pairs[row0 + r] * kHidden;
#pragma unroll
                            for (int i = 0; i < NI; ++i) {
                                dst[rbase + 16 * i + g + 0] = acc[i][n][c] * sc;
                                dst[rbase + 16 * i + g + 8] = acc[i][n][2 + c] * sc;
                            }
                        }
                    }
                }
            }
        }
    }
#else
    (void)sec4_base; (void)experts; (void)tb; (void)act; (void)out_ha; (void)topk_w; (void)y_pair; (void)pf;
#endif
}

// ---- High-Reuse TR=128 Blackwell sm_120a mxf4nvf4 MoE kernel (2x fewer weight reads for W >= 1536 prefill!) ----
// 8 warps = 2 wf (64 rows each, NI=4) x 4 wn (32 token rows each = 4x m16n8k64 per weight load!). 26.5 KiB SMEM!
template <bool GU, bool ATOMIC_OUT = false>
__global__ void __launch_bounds__(kNT) k_moe_sf4_tc_tr128_s(
    const uint8_t* __restrict__ sec4_base,
    const InkExpertDev* __restrict__ experts,
    InkMoeTables tb,
    const uint8_t* __restrict__ act,
    uint8_t* __restrict__ out_ha,
    const float* __restrict__ topk_w,
    float* __restrict__ y_pair,
    int pf) {
#if defined(INK_SM120_DEV)
    constexpr int TR       = 128;
    constexpr int ROWS_N   = GU ? (2 * kInter) : kHidden;  // 2304
    constexpr int ROWS_K   = GU ? kHidden : kInter;        // 2304 (GU) or 1152 (DN)
    constexpr int NS       = ROWS_K / 64;                  // 36 (GU) or 18 (DN)
    constexpr int NFB      = ROWS_N / kWR;                 // 18 blocks of 128 rows
    constexpr int N_WN     = 4;
    constexpr int N_WF     = 2;
    constexpr int W_PER_WF = 64;
    constexpr int NI       = 4;
    constexpr size_t W_P_ROW   = ROWS_K / 2;
    constexpr size_t W_S_ROW   = ROWS_K / 16;
    constexpr size_t ACT_ROW_B = (size_t)NS * kAB;

    __shared__ alignas(16) uint8_t s_wqs[2][kWR * kWStride4];
    __shared__ alignas(16) uint8_t s_aqs[2][TR * kAStride4];
    __shared__ uint32_t s_wsc[2][kWR];
    __shared__ uint32_t s_asc[2][TR];
    __shared__ float    s_agb[2][TR];
    __shared__ int32_t  s_row[TR];
    __shared__ float    s_am[N_WF][TR];
    __shared__ float    s_rw[TR];

    const int tid  = threadIdx.x;
    const int warp = tid >> 5;
    const int lane = tid & 31;
    const int wf   = warp / N_WN;  // 0..1
    const int wn   = warp % N_WN;  // 0..3
    const int nb0  = 32 * wn;      // 0, 32, 64, 96
    const int g    = lane >> 2;
    const int tig  = lane & 3;
    const int lm   = lane >> 3;
    const int l8   = lane & 7;
    const int tidx_a = g + (tig & 1) * 8;

    const int a_off  = (W_PER_WF * wf + 8 * (lm & 1) + l8) * kWStride4 + 16 * (lm >> 1);
    const int b_off0 = (nb0 +      8 * (lm >> 1) + l8) * kAStride4 + 16 * (lm & 1);
    const int b_off1 = (nb0 + 16 + 8 * (lm >> 1) + l8) * kAStride4 + 16 * (lm & 1);
    const int ur = tid >> 1;  // 0..127
    const int uj = tid & 1;   // 0..1

    const int total_tiles = *tb.n_tiles;
    const int nwork = total_tiles * NFB;

    for (int w = blockIdx.x; w < nwork; w += gridDim.x) {
        const int2 tl = tb.tiles[w / NFB];
        const int fb = w % NFB;
        const int e = tl.x;
        const int row0 = tl.y;
        const InkExpertDev ed = experts[e];
        if ((GU ? ed.gu_code : ed.dn_code) != FMT_SF4) continue;
        const int nrows = min(TR, tb.off[e + 1] - row0);
        if (nrows <= 0) continue;

        const uint8_t* body = sec4_base + ed.body_off;
        float hdr[4];
        memcpy(hdr, body, 16);
        const float w_outer = GU ? (hdr[0] / hdr[1]) : (hdr[2] / hdr[3]);

        const uint8_t* base_p = GU ? (body + 16) : (body + ed.dn_off);
        const uint8_t* base_s = base_p + (size_t)ROWS_N * W_P_ROW;

        const int phys_row = GU ? (2 * (fb * 64 + 8 * (ur >> 4) + (ur & 7)) + ((ur >> 3) & 1))
                                : (fb * kWR + ur);
        const uint8_t* wrow_p = base_p + (size_t)phys_row * W_P_ROW;
        const uint8_t* wrow_s = base_s + (size_t)phys_row * W_S_ROW;

        __syncthreads();
        if (tid < TR) {
            if (tid < nrows) {
                s_row[tid] = GU ? tb.tok_idx[row0 + tid] : (row0 + tid);
                if constexpr (!GU) s_rw[tid] = topk_w[tb.pairs[row0 + tid]];
            } else {
                s_row[tid] = -1;
                if constexpr (!GU) s_rw[tid] = 0.0f;
                s_asc[0][tid] = 0u; s_asc[1][tid] = 0u;
                s_agb[0][tid] = 0.0f; s_agb[1][tid] = 0.0f;
#pragma unroll
                for (int b = 0; b < 2; ++b) {
                    uint4* z = (uint4*)(s_aqs[b] + tid * kAStride4);
                    z[0] = make_uint4(0u, 0u, 0u, 0u);
                    z[1] = make_uint4(0u, 0u, 0u, 0u);
                }
            }
        }
        __syncthreads();

        auto issue_stage = [&](int st, int buf) {
            cp16_async(s_wqs[buf] + ur * kWStride4 + 16 * uj, wrow_p + (size_t)st * 32 + 16 * uj);
            if (uj == 0) s_wsc[buf][ur] = *(const uint32_t*)(wrow_s + (size_t)st * 4);
            if (ur < nrows) {
                const uint8_t* ap = act + (size_t)s_row[ur] * ACT_ROW_B + (size_t)st * kAB;
                cp16_async(s_aqs[buf] + ur * kAStride4 + 16 * uj, ap + 16 * uj);
                if (uj == 0) {
                    const uint2 meta = *(const uint2*)(ap + 32);
                    s_asc[buf][ur] = meta.x;
                    s_agb[buf][ur] = __uint_as_float(meta.y);
                }
            }
            cp_commit();
        };

        if (pf && uj == 0) {
            pf_l2(wrow_p, (uint32_t)min((size_t)pf, W_P_ROW));
            if constexpr (GU) pf_l2(wrow_s, (uint32_t)W_S_ROW);
        }
        issue_stage(0, 0);

        float acc[NI][4][4] = {};
        const bool on0 = nb0      < nrows;
        const bool on1 = nb0 +  8 < nrows;
        const bool on2 = nb0 + 16 < nrows;
        const bool on3 = nb0 + 24 < nrows;

        for (int st = 0; st < NS; ++st) {
            const int buf = st & 1;
            const int nxt = buf ^ 1;
            if (pf && uj == 0 && (st % (pf / 32)) == 0 && (size_t)(st + pf / 32) * 32 < W_P_ROW) {
                const size_t nb = (size_t)(st + pf / 32) * 32;
                pf_l2(wrow_p + nb, (uint32_t)min((size_t)pf, W_P_ROW - nb));
            }
            if (st + 1 < NS) {
                issue_stage(st + 1, nxt);
                cp_wait1();
            } else {
                cp_wait0();
            }
            __syncthreads();

            if (on0) {
                uint32_t bq0[4], bq1[4];
                ldsm4(bq0, s_aqs[buf] + b_off0);
                if (on2) ldsm4(bq1, s_aqs[buf] + b_off1);
                const uint32_t sfb0 = s_asc[buf][nb0 +      g];
                const uint32_t sfb1 = on1 ? s_asc[buf][nb0 +  8 + g] : 0u;
                const uint32_t sfb2 = on2 ? s_asc[buf][nb0 + 16 + g] : 0u;
                const uint32_t sfb3 = on3 ? s_asc[buf][nb0 + 24 + g] : 0u;
                const float gb00 = s_agb[buf][nb0 +      2 * tig + 0];
                const float gb01 = s_agb[buf][nb0 +      2 * tig + 1];
                const float gb10 = on1 ? s_agb[buf][nb0 +  8 + 2 * tig + 0] : 0.0f;
                const float gb11 = on1 ? s_agb[buf][nb0 +  8 + 2 * tig + 1] : 0.0f;
                const float gb20 = on2 ? s_agb[buf][nb0 + 16 + 2 * tig + 0] : 0.0f;
                const float gb21 = on2 ? s_agb[buf][nb0 + 16 + 2 * tig + 1] : 0.0f;
                const float gb30 = on3 ? s_agb[buf][nb0 + 24 + 2 * tig + 0] : 0.0f;
                const float gb31 = on3 ? s_agb[buf][nb0 + 24 + 2 * tig + 1] : 0.0f;

#pragma unroll
                for (int i = 0; i < NI; ++i) {
                    uint32_t a[4];
                    ldsm4(a, s_wqs[buf] + a_off + 16 * i * kWStride4);
                    const uint32_t sfa = s_wsc[buf][W_PER_WF * wf + 16 * i + tidx_a];

                    float d0[4] = {};
                    mma_nvfp4_m16n8k64(d0, a, bq0[0], bq0[1], sfa, sfb0);
                    acc[i][0][0] = fmaf(d0[0], gb00, acc[i][0][0]);
                    acc[i][0][1] = fmaf(d0[1], gb01, acc[i][0][1]);
                    acc[i][0][2] = fmaf(d0[2], gb00, acc[i][0][2]);
                    acc[i][0][3] = fmaf(d0[3], gb01, acc[i][0][3]);

                    if (on1) {
                        float d1[4] = {};
                        mma_nvfp4_m16n8k64(d1, a, bq0[2], bq0[3], sfa, sfb1);
                        acc[i][1][0] = fmaf(d1[0], gb10, acc[i][1][0]);
                        acc[i][1][1] = fmaf(d1[1], gb11, acc[i][1][1]);
                        acc[i][1][2] = fmaf(d1[2], gb10, acc[i][1][2]);
                        acc[i][1][3] = fmaf(d1[3], gb11, acc[i][1][3]);
                    }
                    if (on2) {
                        float d2[4] = {};
                        mma_nvfp4_m16n8k64(d2, a, bq1[0], bq1[1], sfa, sfb2);
                        acc[i][2][0] = fmaf(d2[0], gb20, acc[i][2][0]);
                        acc[i][2][1] = fmaf(d2[1], gb21, acc[i][2][1]);
                        acc[i][2][2] = fmaf(d2[2], gb20, acc[i][2][2]);
                        acc[i][2][3] = fmaf(d2[3], gb21, acc[i][2][3]);
                    }
                    if (on3) {
                        float d3[4] = {};
                        mma_nvfp4_m16n8k64(d3, a, bq1[2], bq1[3], sfa, sfb3);
                        acc[i][3][0] = fmaf(d3[0], gb30, acc[i][3][0]);
                        acc[i][3][1] = fmaf(d3[1], gb31, acc[i][3][1]);
                        acc[i][3][2] = fmaf(d3[2], gb30, acc[i][3][2]);
                        acc[i][3][3] = fmaf(d3[3], gb31, acc[i][3][3]);
                    }
                }
            }
            if (st + 1 < NS) __syncthreads();
        }

        if constexpr (GU) {
            constexpr size_t H_ROW_B = (size_t)(kInter / 64) * kAB;
            // Process in 2 halves of 16 rows (keeps register count low!)
            for (int hf = 0; hf < 2; ++hf) {
                const int nbase = hf * 2;
                const bool ha_on = (hf == 0) ? on0 : on2;
                const bool hb_on = (hf == 0) ? on1 : on3;
                float h_val[2][2][NI] = {};
                float am_sub0[2][2] = {}, am_sub1[2][2] = {};
                if (ha_on) {
#pragma unroll
                    for (int n = 0; n < 2; ++n) {
                        if (n == 1 && !hb_on) break;
                        const int nn = nbase + n;
#pragma unroll
                        for (int c = 0; c < 2; ++c) {
#pragma unroll
                            for (int i = 0; i < NI; ++i) {
                                h_val[n][c][i] = situ_f32(acc[i][nn][c] * w_outer, acc[i][nn][2 + c] * w_outer);
                            }
                            float a0 = fmaxf(fabsf(h_val[n][c][0]), fabsf(h_val[n][c][1]));
                            float a1 = fmaxf(fabsf(h_val[n][c][2]), fabsf(h_val[n][c][3]));
#pragma unroll
                            for (int o = 4; o < 32; o <<= 1) {
                                a0 = fmaxf(a0, __shfl_xor_sync(0xFFFFFFFFu, a0, o));
                                a1 = fmaxf(a1, __shfl_xor_sync(0xFFFFFFFFu, a1, o));
                            }
                            am_sub0[n][c] = a0;
                            am_sub1[n][c] = a1;
                            const int r = nb0 + 8 * nn + 2 * tig + c;
                            if (g == 0 && r < TR) s_am[wf][r] = fmaxf(a0, a1);
                        }
                    }
                }
                __syncthreads();
                if (ha_on) {
#pragma unroll
                    for (int n = 0; n < 2; ++n) {
                        if (n == 1 && !hb_on) break;
                        const int nn = nbase + n;
#pragma unroll
                        for (int c = 0; c < 2; ++c) {
                            const int r = nb0 + 8 * nn + 2 * tig + c;
                            const bool valid = (r < nrows);
                            const float am_64 = valid ? fmaxf(s_am[0][r], s_am[1][r]) : 0.0f;
                            const float g_blk = am_64 * (1.0f / (6.0f * 448.0f));
                            float s0 = 0.0f, s1 = 0.0f;
                            const uint8_t d0 = encode_ue4m3(g_blk > 0.0f ? (am_sub0[n][c] / (g_blk * 6.0f)) : 0.0f, s0);
                            const uint8_t d1 = encode_ue4m3(g_blk > 0.0f ? (am_sub1[n][c] / (g_blk * 6.0f)) : 0.0f, s1);
                            const float inv0 = (g_blk * s0) > 0.0f ? (1.0f / (g_blk * s0)) : 0.0f;
                            const float inv1 = (g_blk * s1) > 0.0f ? (1.0f / (g_blk * s1)) : 0.0f;
                            float qa[NI], qb[NI];
#pragma unroll
                            for (int i = 0; i < NI; ++i) {
                                qa[i] = h_val[n][c][i] * ((i < 2) ? inv0 : inv1);
                                qb[i] = __shfl_xor_sync(0xFFFFFFFFu, qa[i], 4);
                            }
                            if (valid) {
                                uint8_t* dst_slot = out_ha + (size_t)(row0 + r) * H_ROW_B + (size_t)fb * kAB;
                                if ((g & 1) == 0) {
                                    const int bo = (4 * NI) * wf + (g >> 1);
#pragma unroll
                                    for (int i = 0; i < NI; ++i) dst_slot[bo + 4 * i] = pack_e2m1x2(qa[i], qb[i]);
                                }
                                if (g == 0) {
                                    dst_slot[32 + 2 * wf + 0] = d0;
                                    dst_slot[32 + 2 * wf + 1] = d1;
                                    if (wf == 0) *(float*)(dst_slot + 36) = g_blk;
                                }
                            }
                        }
                    }
                }
                if (hf == 0) __syncthreads();
            }
        } else if (on0) {
            const int rbase = fb * kWR + W_PER_WF * wf;
#pragma unroll
            for (int n = 0; n < 4; ++n) {
                if (nb0 + 8 * n >= nrows) break;
#pragma unroll
                for (int c = 0; c < 2; ++c) {
                    const int r = nb0 + 8 * n + 2 * tig + c;
                    if (r < nrows) {
                        const float sc = w_outer * s_rw[r];
                        if constexpr (ATOMIC_OUT) {
                            float* dst = y_pair + (size_t)(tb.pairs[row0 + r] >> 3) * kHidden;
#pragma unroll
                            for (int i = 0; i < NI; ++i) {
                                atomicAdd(&dst[rbase + 16 * i + g + 0], acc[i][n][c] * sc);
                                atomicAdd(&dst[rbase + 16 * i + g + 8], acc[i][n][2 + c] * sc);
                            }
                        } else {
                            float* dst = y_pair + (size_t)tb.pairs[row0 + r] * kHidden;
#pragma unroll
                            for (int i = 0; i < NI; ++i) {
                                dst[rbase + 16 * i + g + 0] = acc[i][n][c] * sc;
                                dst[rbase + 16 * i + g + 8] = acc[i][n][2 + c] * sc;
                            }
                        }
                    }
                }
            }
        }
    }
#else
    (void)sec4_base; (void)experts; (void)tb; (void)act; (void)out_ha; (void)topk_w; (void)y_pair; (void)pf;
#endif
}

// ---- Opt-In Direct-MMA Per-Row g_row NVFP4 MoE Kernels (TR in {16, 32, 64, 128}; ZERO per-K64 fmaf, +22% faster!) ----
constexpr float kDirectHGlobalScale = 0.25f;  // UE4M3(448) * E2M1(6) * 0.25 = 672.0 max, subnormal precision 0.0005

template <bool GU, int TR, bool ATOMIC_OUT = false>
__global__ void __launch_bounds__(kNT) k_moe_sf4_tc_direct(
    const uint8_t* __restrict__ sec4_base,
    const InkExpertDev* __restrict__ experts,
    InkMoeTables tb,
    const uint8_t* __restrict__ act,
    uint8_t* __restrict__ out_ha,
    const float* __restrict__ topk_w,
    float* __restrict__ y_pair,
    int pf) {
#if defined(INK_SM120_DEV)
    constexpr int ROWS_N   = GU ? (2 * kInter) : kHidden;
    constexpr int ROWS_K   = GU ? kHidden : kInter;
    constexpr int NS       = ROWS_K / 64;
    constexpr int NFB      = ROWS_N / kWR;
    constexpr int N_WN     = TR / 16;
    constexpr int N_WF     = 8 / N_WN;
    constexpr int W_PER_WF = kWR / N_WF;
    constexpr int NI       = W_PER_WF / 16;
    constexpr size_t W_P_ROW   = ROWS_K / 2;
    constexpr size_t W_S_ROW   = ROWS_K / 16;
    constexpr size_t ACT_ROW_B = (size_t)NS * kAB;

    __shared__ alignas(16) uint8_t s_wqs[2][kWR * kWStride4];
    __shared__ alignas(16) uint8_t s_aqs[2][TR * kAStride4];
    __shared__ uint32_t s_wsc[2][kWR];
    __shared__ uint32_t s_asc[2][TR];
    __shared__ float    s_grow[TR];
    __shared__ int32_t  s_row[TR];
    __shared__ float    s_am[N_WF][TR];
    __shared__ float    s_rw[TR];

    const int tid  = threadIdx.x;
    const int warp = tid >> 5;
    const int lane = tid & 31;
    const int wf   = warp / N_WN;
    const int wn   = warp % N_WN;
    const int nb0  = 16 * wn;
    const int g    = lane >> 2;
    const int tig  = lane & 3;
    const int lm   = lane >> 3;
    const int l8   = lane & 7;
    const int tidx_a = g + (tig & 1) * 8;

    const int a_off = (W_PER_WF * wf + 8 * (lm & 1) + l8) * kWStride4 + 16 * (lm >> 1);
    const int b_off = (nb0 + 8 * (lm >> 1) + l8) * kAStride4 + 16 * (lm & 1);
    const int ur = tid >> 1;
    const int uj = tid & 1;

    const int total_tiles = *tb.n_tiles;
    const int nwork = total_tiles * NFB;

    for (int w = blockIdx.x; w < nwork; w += gridDim.x) {
        const int2 tl = tb.tiles[w / NFB];
        const int fb = w % NFB;
        const int e = tl.x;
        const int row0 = tl.y;
        const InkExpertDev ed = experts[e];
        if ((GU ? ed.gu_code : ed.dn_code) != FMT_SF4) continue;
        const int nrows = min(TR, tb.off[e + 1] - row0);
        if (nrows <= 0) continue;

        const uint8_t* body = sec4_base + ed.body_off;
        float hdr[4];
        memcpy(hdr, body, 16);
        const float w_outer = GU ? (hdr[0] / hdr[1]) : (hdr[2] / hdr[3]);

        const uint8_t* base_p = GU ? (body + 16) : (body + ed.dn_off);
        const uint8_t* base_s = base_p + (size_t)ROWS_N * W_P_ROW;

        const int phys_row = GU ? (2 * (fb * 64 + 8 * (ur >> 4) + (ur & 7)) + ((ur >> 3) & 1))
                                : (fb * kWR + ur);
        const uint8_t* wrow_p = base_p + (size_t)phys_row * W_P_ROW;
        const uint8_t* wrow_s = base_s + (size_t)phys_row * W_S_ROW;

        __syncthreads();
        if (tid < TR) {
            if (tid < nrows) {
                const int sr = GU ? tb.tok_idx[row0 + tid] : (row0 + tid);
                s_row[tid] = sr;
                if constexpr (GU) {
                    s_grow[tid] = *(const float*)(act + (size_t)sr * ACT_ROW_B + 36);
                } else {
                    s_grow[tid] = kDirectHGlobalScale;
                    s_rw[tid] = topk_w[tb.pairs[row0 + tid]];
                }
            } else {
                s_row[tid] = -1;
                s_grow[tid] = 0.0f;
                if constexpr (!GU) s_rw[tid] = 0.0f;
                s_asc[0][tid] = 0u; s_asc[1][tid] = 0u;
#pragma unroll
                for (int b = 0; b < 2; ++b) {
                    uint4* z = (uint4*)(s_aqs[b] + tid * kAStride4);
                    z[0] = make_uint4(0u, 0u, 0u, 0u);
                    z[1] = make_uint4(0u, 0u, 0u, 0u);
                }
            }
        }
        __syncthreads();

        auto issue_stage = [&](int st, int buf) {
            cp16_async(s_wqs[buf] + ur * kWStride4 + 16 * uj, wrow_p + (size_t)st * 32 + 16 * uj);
            if (uj == 0) s_wsc[buf][ur] = *(const uint32_t*)(wrow_s + (size_t)st * 4);
            if (tid < 2 * TR) {
                const int r = tid >> 1;
                const int h = tid & 1;
                if (r < nrows) {
                    const uint8_t* ap = act + (size_t)s_row[r] * ACT_ROW_B + (size_t)st * kAB;
                    cp16_async(s_aqs[buf] + r * kAStride4 + 16 * h, ap + 16 * h);
                }
            } else if (tid < 3 * TR) {
                const int r = tid - 2 * TR;
                if (r < nrows) {
                    const uint8_t* ap = act + (size_t)s_row[r] * ACT_ROW_B + (size_t)st * kAB;
                    s_asc[buf][r] = *(const uint32_t*)(ap + 32);
                }
            }
            cp_commit();
        };

        if (pf && uj == 0) {
            pf_l2(wrow_p, (uint32_t)min((size_t)pf, W_P_ROW));
            if constexpr (GU) pf_l2(wrow_s, (uint32_t)W_S_ROW);
        }
        issue_stage(0, 0);

        float acc[NI][2][4] = {};
        const bool on0 = nb0 < nrows;
        const bool on1 = nb0 + 8 < nrows;

        for (int st = 0; st < NS; ++st) {
            const int buf = st & 1;
            const int nxt = buf ^ 1;
            if (pf && uj == 0 && (st % (pf / 32)) == 0 && (size_t)(st + pf / 32) * 32 < W_P_ROW) {
                const size_t nb = (size_t)(st + pf / 32) * 32;
                pf_l2(wrow_p + nb, (uint32_t)min((size_t)pf, W_P_ROW - nb));
            }
            if (st + 1 < NS) {
                issue_stage(st + 1, nxt);
                cp_wait1();
            } else {
                cp_wait0();
            }
            __syncthreads();

            if (on0) {
                uint32_t bq[4];
                ldsm4(bq, s_aqs[buf] + b_off);
                const uint32_t sfb0 = s_asc[buf][nb0 + g];
                const uint32_t sfb1 = on1 ? s_asc[buf][nb0 + 8 + g] : 0u;
#pragma unroll
                for (int i = 0; i < NI; ++i) {
                    uint32_t a[4];
                    ldsm4(a, s_wqs[buf] + a_off + 16 * i * kWStride4);
                    const uint32_t sfa = s_wsc[buf][W_PER_WF * wf + 16 * i + tidx_a];
                    mma_nvfp4_m16n8k64(acc[i][0], a, bq[0], bq[1], sfa, sfb0);
                    if (on1) mma_nvfp4_m16n8k64(acc[i][1], a, bq[2], bq[3], sfa, sfb1);
                }
            }
            if (st + 1 < NS) __syncthreads();
        }

        if constexpr (GU) {
            constexpr size_t H_ROW_B = (size_t)(kInter / 64) * kAB;
            constexpr float inv_g6 = 1.0f / (kDirectHGlobalScale * 6.0f);
            float h_val[2][2][NI] = {};
            float am_sub0[2][2] = {}, am_sub1[2][2] = {};
            if (on0) {
#pragma unroll
                for (int n = 0; n < 2; ++n) {
                    if (n == 1 && !on1) break;
#pragma unroll
                    for (int c = 0; c < 2; ++c) {
                        const int r = nb0 + 8 * n + 2 * tig + c;
                        const float sc_r = (r < TR) ? (w_outer * s_grow[r]) : 0.0f;
#pragma unroll
                        for (int i = 0; i < NI; ++i) {
                            h_val[n][c][i] = situ_f32(acc[i][n][c] * sc_r, acc[i][n][2 + c] * sc_r);
                        }
                        float a0 = fabsf(h_val[n][c][0]);
                        if constexpr (NI >= 2) a0 = fmaxf(a0, fabsf(h_val[n][c][1]));
                        float a1 = 0.0f;
                        if constexpr (NI == 4) a1 = fmaxf(fabsf(h_val[n][c][2]), fabsf(h_val[n][c][3]));
#pragma unroll
                        for (int o = 4; o < 32; o <<= 1) {
                            a0 = fmaxf(a0, __shfl_xor_sync(0xFFFFFFFFu, a0, o));
                            if constexpr (NI == 4) a1 = fmaxf(a1, __shfl_xor_sync(0xFFFFFFFFu, a1, o));
                        }
                        am_sub0[n][c] = a0;
                        am_sub1[n][c] = a1;
                        if constexpr (NI == 1) {
                            if (g == 0 && r < TR) s_am[wf][r] = a0;
                        }
                    }
                }
            }
            if constexpr (NI == 1) __syncthreads();
            if (on0) {
#pragma unroll
                for (int n = 0; n < 2; ++n) {
                    if (n == 1 && !on1) break;
#pragma unroll
                    for (int c = 0; c < 2; ++c) {
                        const int r = nb0 + 8 * n + 2 * tig + c;
                        const bool valid = (r < nrows);
                        float s0 = 0.0f, s1 = 0.0f;
                        float sub0_am = am_sub0[n][c];
                        if constexpr (NI == 1) {
                            sub0_am = valid ? fmaxf(s_am[wf & ~1][r], s_am[wf | 1][r]) : 0.0f;
                        }
                        const uint8_t d0 = encode_ue4m3(sub0_am * inv_g6, s0);
                        const uint8_t d1 = (NI == 4) ? encode_ue4m3(am_sub1[n][c] * inv_g6, s1) : 0u;
                        const float inv0 = (s0 > 0.0f) ? (1.0f / (kDirectHGlobalScale * s0)) : 0.0f;
                        const float inv1 = (NI == 4 && s1 > 0.0f) ? (1.0f / (kDirectHGlobalScale * s1)) : 0.0f;
                        float qa[NI], qb[NI];
#pragma unroll
                        for (int i = 0; i < NI; ++i) {
                            qa[i] = h_val[n][c][i] * ((i < 2) ? inv0 : inv1);
                            qb[i] = __shfl_xor_sync(0xFFFFFFFFu, qa[i], 4);
                        }
                        if (valid) {
                            uint8_t* dst_slot = out_ha + (size_t)(row0 + r) * H_ROW_B + (size_t)fb * kAB;
                            if ((g & 1) == 0) {
                                const int bo = (4 * NI) * wf + (g >> 1);
#pragma unroll
                                for (int i = 0; i < NI; ++i) dst_slot[bo + 4 * i] = pack_e2m1x2(qa[i], qb[i]);
                            }
                            if (g == 0) {
                                if constexpr (NI == 4) {
                                    dst_slot[32 + 2 * wf + 0] = d0;
                                    dst_slot[32 + 2 * wf + 1] = d1;
                                } else if constexpr (NI == 2) {
                                    dst_slot[32 + wf] = d0;
                                } else {
                                    if ((wf & 1) == 0) dst_slot[32 + (wf >> 1)] = d0;
                                }
                            }
                        }
                    }
                }
            }
        } else if (on0) {
            const int rbase = fb * kWR + W_PER_WF * wf;
#pragma unroll
            for (int n = 0; n < 2; ++n) {
                if (n == 1 && !on1) break;
#pragma unroll
                for (int c = 0; c < 2; ++c) {
                    const int r = nb0 + 8 * n + 2 * tig + c;
                    if (r < nrows) {
                        const float sc = w_outer * kDirectHGlobalScale * s_rw[r];
                        if constexpr (ATOMIC_OUT) {
                            float* dst = y_pair + (size_t)(tb.pairs[row0 + r] >> 3) * kHidden;
#pragma unroll
                            for (int i = 0; i < NI; ++i) {
                                atomicAdd(&dst[rbase + 16 * i + g + 0], acc[i][n][c] * sc);
                                atomicAdd(&dst[rbase + 16 * i + g + 8], acc[i][n][2 + c] * sc);
                            }
                        } else {
                            float* dst = y_pair + (size_t)tb.pairs[row0 + r] * kHidden;
#pragma unroll
                            for (int i = 0; i < NI; ++i) {
                                dst[rbase + 16 * i + g + 0] = acc[i][n][c] * sc;
                                dst[rbase + 16 * i + g + 8] = acc[i][n][2 + c] * sc;
                            }
                        }
                    }
                }
            }
        }
    }
#else
    (void)sec4_base; (void)experts; (void)tb; (void)act; (void)out_ha; (void)topk_w; (void)y_pair; (void)pf;
#endif
}

template <bool GU, bool ATOMIC_OUT = false>
__global__ void __launch_bounds__(kNT) k_moe_sf4_tc_tr128_s_direct(
    const uint8_t* __restrict__ sec4_base,
    const InkExpertDev* __restrict__ experts,
    InkMoeTables tb,
    const uint8_t* __restrict__ act,
    uint8_t* __restrict__ out_ha,
    const float* __restrict__ topk_w,
    float* __restrict__ y_pair,
    int pf) {
#if defined(INK_SM120_DEV)
    constexpr int TR       = 128;
    constexpr int ROWS_N   = GU ? (2 * kInter) : kHidden;
    constexpr int ROWS_K   = GU ? kHidden : kInter;
    constexpr int NS       = ROWS_K / 64;
    constexpr int NFB      = ROWS_N / kWR;
    constexpr int N_WN     = 4;
    constexpr int W_PER_WF = 64;
    constexpr int NI       = 4;
    constexpr size_t W_P_ROW   = ROWS_K / 2;
    constexpr size_t W_S_ROW   = ROWS_K / 16;
    constexpr size_t ACT_ROW_B = (size_t)NS * kAB;

    __shared__ alignas(16) uint8_t s_wqs[2][kWR * kWStride4];
    __shared__ alignas(16) uint8_t s_aqs[2][TR * kAStride4];
    __shared__ uint32_t s_wsc[2][kWR];
    __shared__ uint32_t s_asc[2][TR];
    __shared__ float    s_grow[TR];
    __shared__ int32_t  s_row[TR];
    __shared__ float    s_rw[TR];

    const int tid  = threadIdx.x;
    const int warp = tid >> 5;
    const int lane = tid & 31;
    const int wf   = warp / N_WN;
    const int wn   = warp % N_WN;
    const int nb0  = 32 * wn;
    const int g    = lane >> 2;
    const int tig  = lane & 3;
    const int lm   = lane >> 3;
    const int l8   = lane & 7;
    const int tidx_a = g + (tig & 1) * 8;

    const int a_off  = (W_PER_WF * wf + 8 * (lm & 1) + l8) * kWStride4 + 16 * (lm >> 1);
    const int b_off0 = (nb0 +      8 * (lm >> 1) + l8) * kAStride4 + 16 * (lm & 1);
    const int b_off1 = (nb0 + 16 + 8 * (lm >> 1) + l8) * kAStride4 + 16 * (lm & 1);
    const int ur = tid >> 1;
    const int uj = tid & 1;

    const int total_tiles = *tb.n_tiles;
    const int nwork = total_tiles * NFB;

    for (int w = blockIdx.x; w < nwork; w += gridDim.x) {
        const int2 tl = tb.tiles[w / NFB];
        const int fb = w % NFB;
        const int e = tl.x;
        const int row0 = tl.y;
        const InkExpertDev ed = experts[e];
        if ((GU ? ed.gu_code : ed.dn_code) != FMT_SF4) continue;
        const int nrows = min(TR, tb.off[e + 1] - row0);
        if (nrows <= 0) continue;

        const uint8_t* body = sec4_base + ed.body_off;
        float hdr[4];
        memcpy(hdr, body, 16);
        const float w_outer = GU ? (hdr[0] / hdr[1]) : (hdr[2] / hdr[3]);

        const uint8_t* base_p = GU ? (body + 16) : (body + ed.dn_off);
        const uint8_t* base_s = base_p + (size_t)ROWS_N * W_P_ROW;

        const int phys_row = GU ? (2 * (fb * 64 + 8 * (ur >> 4) + (ur & 7)) + ((ur >> 3) & 1))
                                : (fb * kWR + ur);
        const uint8_t* wrow_p = base_p + (size_t)phys_row * W_P_ROW;
        const uint8_t* wrow_s = base_s + (size_t)phys_row * W_S_ROW;

        __syncthreads();
        if (tid < TR) {
            if (tid < nrows) {
                const int sr = GU ? tb.tok_idx[row0 + tid] : (row0 + tid);
                s_row[tid] = sr;
                if constexpr (GU) s_grow[tid] = *(const float*)(act + (size_t)sr * ACT_ROW_B + 36);
                else { s_grow[tid] = kDirectHGlobalScale; s_rw[tid] = topk_w[tb.pairs[row0 + tid]]; }
            } else {
                s_row[tid] = -1;
                s_grow[tid] = 0.0f;
                if constexpr (!GU) s_rw[tid] = 0.0f;
                s_asc[0][tid] = 0u; s_asc[1][tid] = 0u;
#pragma unroll
                for (int b = 0; b < 2; ++b) {
                    uint4* z = (uint4*)(s_aqs[b] + tid * kAStride4);
                    z[0] = make_uint4(0u, 0u, 0u, 0u);
                    z[1] = make_uint4(0u, 0u, 0u, 0u);
                }
            }
        }
        __syncthreads();

        auto issue_stage = [&](int st, int buf) {
            cp16_async(s_wqs[buf] + ur * kWStride4 + 16 * uj, wrow_p + (size_t)st * 32 + 16 * uj);
            if (uj == 0) s_wsc[buf][ur] = *(const uint32_t*)(wrow_s + (size_t)st * 4);
            if (ur < nrows) {
                const uint8_t* ap = act + (size_t)s_row[ur] * ACT_ROW_B + (size_t)st * kAB;
                cp16_async(s_aqs[buf] + ur * kAStride4 + 16 * uj, ap + 16 * uj);
                if (uj == 0) s_asc[buf][ur] = *(const uint32_t*)(ap + 32);
            }
            cp_commit();
        };

        if (pf && uj == 0) {
            pf_l2(wrow_p, (uint32_t)min((size_t)pf, W_P_ROW));
            if constexpr (GU) pf_l2(wrow_s, (uint32_t)W_S_ROW);
        }
        issue_stage(0, 0);

        float acc[NI][4][4] = {};
        const bool on0 = nb0      < nrows;
        const bool on1 = nb0 +  8 < nrows;
        const bool on2 = nb0 + 16 < nrows;
        const bool on3 = nb0 + 24 < nrows;

        for (int st = 0; st < NS; ++st) {
            const int buf = st & 1;
            const int nxt = buf ^ 1;
            if (pf && uj == 0 && (st % (pf / 32)) == 0 && (size_t)(st + pf / 32) * 32 < W_P_ROW) {
                const size_t nb = (size_t)(st + pf / 32) * 32;
                pf_l2(wrow_p + nb, (uint32_t)min((size_t)pf, W_P_ROW - nb));
            }
            if (st + 1 < NS) {
                issue_stage(st + 1, nxt);
                cp_wait1();
            } else {
                cp_wait0();
            }
            __syncthreads();

            if (on0) {
                uint32_t bq0[4], bq1[4];
                ldsm4(bq0, s_aqs[buf] + b_off0);
                if (on2) ldsm4(bq1, s_aqs[buf] + b_off1);
                const uint32_t sfb0 = s_asc[buf][nb0 +      g];
                const uint32_t sfb1 = on1 ? s_asc[buf][nb0 +  8 + g] : 0u;
                const uint32_t sfb2 = on2 ? s_asc[buf][nb0 + 16 + g] : 0u;
                const uint32_t sfb3 = on3 ? s_asc[buf][nb0 + 24 + g] : 0u;
#pragma unroll
                for (int i = 0; i < NI; ++i) {
                    uint32_t a[4];
                    ldsm4(a, s_wqs[buf] + a_off + 16 * i * kWStride4);
                    const uint32_t sfa = s_wsc[buf][W_PER_WF * wf + 16 * i + tidx_a];
                    mma_nvfp4_m16n8k64(acc[i][0], a, bq0[0], bq0[1], sfa, sfb0);
                    if (on1) mma_nvfp4_m16n8k64(acc[i][1], a, bq0[2], bq0[3], sfa, sfb1);
                    if (on2) mma_nvfp4_m16n8k64(acc[i][2], a, bq1[0], bq1[1], sfa, sfb2);
                    if (on3) mma_nvfp4_m16n8k64(acc[i][3], a, bq1[2], bq1[3], sfa, sfb3);
                }
            }
            if (st + 1 < NS) __syncthreads();
        }

        if constexpr (GU) {
            constexpr size_t H_ROW_B = (size_t)(kInter / 64) * kAB;
            constexpr float inv_g6 = 1.0f / (kDirectHGlobalScale * 6.0f);
            if (on0) {
#pragma unroll
                for (int nn = 0; nn < 4; ++nn) {
                    if (nb0 + 8 * nn >= nrows) break;
#pragma unroll
                    for (int c = 0; c < 2; ++c) {
                        const int r = nb0 + 8 * nn + 2 * tig + c;
                        const bool valid = (r < nrows);
                        const float sc_r = valid ? (w_outer * s_grow[r]) : 0.0f;
                        float hv[NI];
#pragma unroll
                        for (int i = 0; i < NI; ++i) {
                            hv[i] = situ_f32(acc[i][nn][c] * sc_r, acc[i][nn][2 + c] * sc_r);
                        }
                        float a0 = fmaxf(fabsf(hv[0]), fabsf(hv[1]));
                        float a1 = fmaxf(fabsf(hv[2]), fabsf(hv[3]));
#pragma unroll
                        for (int o = 4; o < 32; o <<= 1) {
                            a0 = fmaxf(a0, __shfl_xor_sync(0xFFFFFFFFu, a0, o));
                            a1 = fmaxf(a1, __shfl_xor_sync(0xFFFFFFFFu, a1, o));
                        }
                        float s0 = 0.0f, s1 = 0.0f;
                        const uint8_t d0 = encode_ue4m3(a0 * inv_g6, s0);
                        const uint8_t d1 = encode_ue4m3(a1 * inv_g6, s1);
                        const float inv0 = (s0 > 0.0f) ? (1.0f / (kDirectHGlobalScale * s0)) : 0.0f;
                        const float inv1 = (s1 > 0.0f) ? (1.0f / (kDirectHGlobalScale * s1)) : 0.0f;
                        float qa[NI], qb[NI];
#pragma unroll
                        for (int i = 0; i < NI; ++i) {
                            qa[i] = hv[i] * ((i < 2) ? inv0 : inv1);
                            qb[i] = __shfl_xor_sync(0xFFFFFFFFu, qa[i], 4);
                        }
                        if (valid) {
                            uint8_t* dst_slot = out_ha + (size_t)(row0 + r) * H_ROW_B + (size_t)fb * kAB;
                            if ((g & 1) == 0) {
                                const int bo = (4 * NI) * wf + (g >> 1);
#pragma unroll
                                for (int i = 0; i < NI; ++i) dst_slot[bo + 4 * i] = pack_e2m1x2(qa[i], qb[i]);
                            }
                            if (g == 0) {
                                dst_slot[32 + 2 * wf + 0] = d0;
                                dst_slot[32 + 2 * wf + 1] = d1;
                            }
                        }
                    }
                }
            }
        } else if (on0) {
            const int rbase = fb * kWR + W_PER_WF * wf;
#pragma unroll
            for (int n = 0; n < 4; ++n) {
                if (nb0 + 8 * n >= nrows) break;
#pragma unroll
                for (int c = 0; c < 2; ++c) {
                    const int r = nb0 + 8 * n + 2 * tig + c;
                    if (r < nrows) {
                        const float sc = w_outer * kDirectHGlobalScale * s_rw[r];
                        if constexpr (ATOMIC_OUT) {
                            float* dst = y_pair + (size_t)(tb.pairs[row0 + r] >> 3) * kHidden;
#pragma unroll
                            for (int i = 0; i < NI; ++i) {
                                atomicAdd(&dst[rbase + 16 * i + g + 0], acc[i][n][c] * sc);
                                atomicAdd(&dst[rbase + 16 * i + g + 8], acc[i][n][2 + c] * sc);
                            }
                        } else {
                            float* dst = y_pair + (size_t)tb.pairs[row0 + r] * kHidden;
#pragma unroll
                            for (int i = 0; i < NI; ++i) {
                                dst[rbase + 16 * i + g + 0] = acc[i][n][c] * sc;
                                dst[rbase + 16 * i + g + 8] = acc[i][n][2 + c] * sc;
                            }
                        }
                    }
                }
            }
        }
    }
#else
    (void)sec4_base; (void)experts; (void)tb; (void)act; (void)out_ha; (void)topk_w; (void)y_pair; (void)pf;
#endif
}

// ---- Persistent Blackwell sm_120a mxf8f6f4 MoE kernel (TR in {16, 32, 64}; L20..23 MX8 experts, 32-element UE8M0 blocks)
template <bool GU, int TR>
__global__ void __launch_bounds__(kNT) k_moe_mx8_tc(
    const uint8_t* __restrict__ sec4_base,
    const InkExpertDev* __restrict__ experts,
    InkMoeTables tb,
    const uint8_t* __restrict__ xq,
    const uint8_t* __restrict__ xsc,
    uint8_t* __restrict__ hq,
    uint8_t* __restrict__ hsc,
    const float* __restrict__ topk_w,
    float* __restrict__ y_pair,
    int pf) {
#if defined(INK_SM120_DEV)
    constexpr int ROWS_N   = GU ? (2 * kInter) : kHidden;
    constexpr int ROWS_K   = GU ? kHidden : kInter;
    constexpr int NS       = ROWS_K / 32;
    constexpr int NFB      = ROWS_N / kWR;
    constexpr int N_WN     = TR / 16;
    constexpr int N_WF     = 8 / N_WN;
    constexpr int W_PER_WF = kWR / N_WF;
    constexpr int NI       = W_PER_WF / 16;

    __shared__ alignas(16) uint8_t s_wqs[2][kWR * kWStride4];
    __shared__ alignas(16) uint8_t s_aqs[2][TR * kAStride4];
    __shared__ uint8_t  s_wsc[2][kWR];
    __shared__ uint8_t  s_asc[2][TR];
    __shared__ int32_t  s_row[TR];
    __shared__ float    s_am[N_WF][TR];
    __shared__ float    s_rw[TR];

    const int tid  = threadIdx.x;
    const int warp = tid >> 5;
    const int lane = tid & 31;
    const int wf   = warp / N_WN;
    const int wn   = warp % N_WN;
    const int nb0  = 16 * wn;
    const int g    = lane >> 2;
    const int tig  = lane & 3;
    const int lm   = lane >> 3;
    const int l8   = lane & 7;
    const int tidx_a = g + (tig & 1) * 8;

    const int a_off = (W_PER_WF * wf + 8 * (lm & 1) + l8) * kWStride4 + 16 * (lm >> 1);
    const int b_off = (nb0 + 8 * (lm >> 1) + l8) * kAStride4 + 16 * (lm & 1);
    const int ur = tid >> 1;
    const int uj = tid & 1;

    const int total_tiles = *tb.n_tiles;
    const int nwork = total_tiles * NFB;

    for (int w = blockIdx.x; w < nwork; w += gridDim.x) {
        const int2 tl = tb.tiles[w / NFB];
        const int fb = w % NFB;
        const int e = tl.x;
        const int row0 = tl.y;
        const InkExpertDev ed = experts[e];
        if ((GU ? ed.gu_code : ed.dn_code) != FMT_MX8) continue;
        const int nrows = min(TR, tb.off[e + 1] - row0);
        if (nrows <= 0) continue;

        const uint8_t* body = sec4_base + ed.body_off;
        const uint8_t* base_p = GU ? body : (body + ed.dn_off);
        const uint8_t* base_s = base_p + (size_t)ROWS_N * ROWS_K;

        const int phys_row = GU ? (2 * (fb * 64 + 8 * (ur >> 4) + (ur & 7)) + ((ur >> 3) & 1))
                                : (fb * kWR + ur);
        const uint8_t* wrow_p = base_p + (size_t)phys_row * ROWS_K;
        const uint8_t* wrow_s = base_s + (size_t)phys_row * NS;

        __syncthreads();
        if (tid < TR) {
            if (tid < nrows) {
                s_row[tid] = GU ? tb.tok_idx[row0 + tid] : (row0 + tid);
                if constexpr (!GU) s_rw[tid] = topk_w[tb.pairs[row0 + tid]];
            } else {
                s_row[tid] = -1;
                if constexpr (!GU) s_rw[tid] = 0.0f;
                s_asc[0][tid] = 0u; s_asc[1][tid] = 0u;
#pragma unroll
                for (int b = 0; b < 2; ++b) {
                    uint4* z = (uint4*)(s_aqs[b] + tid * kAStride4);
                    z[0] = make_uint4(0u, 0u, 0u, 0u);
                    z[1] = make_uint4(0u, 0u, 0u, 0u);
                }
            }
        }
        __syncthreads();

        auto issue_stage = [&](int st, int buf) {
            cp16_async(s_wqs[buf] + ur * kWStride4 + 16 * uj, wrow_p + (size_t)st * 32 + 16 * uj);
            if (uj == 0) s_wsc[buf][ur] = wrow_s[st];
            if (tid < 2 * TR) {
                const int r = tid >> 1;
                const int h = tid & 1;
                if (r < nrows) {
                    cp16_async(s_aqs[buf] + r * kAStride4 + 16 * h,
                               xq + (size_t)s_row[r] * ROWS_K + (size_t)st * 32 + 16 * h);
                }
            } else if (tid < 3 * TR) {
                const int r = tid - 2 * TR;
                if (r < nrows) s_asc[buf][r] = xsc[(size_t)s_row[r] * NS + st];
            }
            cp_commit();
        };

        if (pf && uj == 0) pf_l2(wrow_p, (uint32_t)min((size_t)pf, (size_t)ROWS_K));
        issue_stage(0, 0);

        float acc[NI][2][4] = {};
        const bool on0 = nb0 < nrows;
        const bool on1 = nb0 + 8 < nrows;

        for (int st = 0; st < NS; ++st) {
            const int buf = st & 1;
            const int nxt = buf ^ 1;
            if (pf && uj == 0 && (st % (pf / 32)) == 0 && (size_t)(st + pf / 32) * 32 < (size_t)ROWS_K) {
                const size_t nb = (size_t)(st + pf / 32) * 32;
                pf_l2(wrow_p + nb, (uint32_t)min((size_t)pf, (size_t)ROWS_K - nb));
            }
            if (st + 1 < NS) {
                issue_stage(st + 1, nxt);
                cp_wait1();
            } else {
                cp_wait0();
            }
            __syncthreads();

            if (on0) {
                uint32_t bq[4];
                ldsm4(bq, s_aqs[buf] + b_off);
                const uint32_t sfb0 = (uint32_t)s_asc[buf][nb0 + g];
                const uint32_t sfb1 = on1 ? (uint32_t)s_asc[buf][nb0 + 8 + g] : 0u;
#pragma unroll
                for (int i = 0; i < NI; ++i) {
                    uint32_t a[4];
                    ldsm4(a, s_wqs[buf] + a_off + 16 * i * kWStride4);
                    const uint32_t sfa = (uint32_t)s_wsc[buf][W_PER_WF * wf + 16 * i + tidx_a];
                    mma_mxfp8_m16n8k32(acc[i][0], a, bq[0], bq[1], sfa, sfb0);
                    if (on1) mma_mxfp8_m16n8k32(acc[i][1], a, bq[2], bq[3], sfa, sfb1);
                }
            }
            if (st + 1 < NS) __syncthreads();
        }

        if constexpr (GU) {
            // Fused SITU(4.0, 25.0) + 32-element MX8 quantization (2 blocks of 32 feats per CTA!)
            float hv[2][2][NI] = {};
            float am_loc[2][2] = {};
            if (on0) {
#pragma unroll
                for (int n = 0; n < 2; ++n) {
                    if (n == 1 && !on1) break;
#pragma unroll
                    for (int c = 0; c < 2; ++c) {
                        float am = 0.0f;
#pragma unroll
                        for (int i = 0; i < NI; ++i) {
                            hv[n][c][i] = situ_f32(acc[i][n][c], acc[i][n][2 + c]);
                            am = fmaxf(am, fabsf(hv[n][c][i]));
                        }
#pragma unroll
                        for (int o = 4; o < 32; o <<= 1) am = fmaxf(am, __shfl_xor_sync(0xFFFFFFFFu, am, o));
                        am_loc[n][c] = am;
                        const int r = nb0 + 8 * n + 2 * tig + c;
                        if (g == 0 && r < TR) s_am[wf][r] = am;
                    }
                }
            }
            if constexpr (NI < 4) __syncthreads();
            if (on0) {
                constexpr int WF_PER_BLK32 = N_WF / 2;  // 1 (TR=64), 2 (TR=32), 4 (TR=16)
                const int half_id = wf / WF_PER_BLK32;  // 0 or 1
                const int sub_in  = wf % WF_PER_BLK32;  // 0..WF_PER_BLK32-1
                const int blk32   = 2 * fb + half_id;
#pragma unroll
                for (int n = 0; n < 2; ++n) {
                    if (n == 1 && !on1) break;
#pragma unroll
                    for (int c = 0; c < 2; ++c) {
                        const int r = nb0 + 8 * n + 2 * tig + c;
                        if (r < nrows) {
                            float am32 = am_loc[n][c];
                            if constexpr (NI < 4) {
                                am32 = 0.0f;
#pragma unroll
                                for (int k = 0; k < WF_PER_BLK32; ++k) am32 = fmaxf(am32, s_am[half_id * WF_PER_BLK32 + k][r]);
                            }
                            float inv_sc = 0.0f;
                            const uint8_t sc_b = encode_ue8m0(am32, inv_sc);
                            uint8_t* dst_q = hq + (size_t)(row0 + r) * kInter + blk32 * 32 + sub_in * (8 * NI);
#pragma unroll
                            for (int i = 0; i < NI; ++i) dst_q[8 * i + g] = encode_e4m3_signed(hv[n][c][i] * inv_sc);
                            if (g == 0 && sub_in == 0) hsc[(size_t)(row0 + r) * (kInter / 32) + blk32] = sc_b;
                        }
                    }
                }
            }
        } else if (on0) {
            const int rbase = fb * kWR + W_PER_WF * wf;
#pragma unroll
            for (int n = 0; n < 2; ++n) {
                if (n == 1 && !on1) break;
#pragma unroll
                for (int c = 0; c < 2; ++c) {
                    const int r = nb0 + 8 * n + 2 * tig + c;
                    if (r < nrows) {
                        const float sc = s_rw[r];
                        float* dst = y_pair + (size_t)tb.pairs[row0 + r] * kHidden;
#pragma unroll
                        for (int i = 0; i < NI; ++i) {
                            dst[rbase + 16 * i + g + 0] = acc[i][n][c] * sc;
                            dst[rbase + 16 * i + g + 8] = acc[i][n][2 + c] * sc;
                        }
                    }
                }
            }
        }
    }
#else
    (void)sec4_base; (void)experts; (void)tb; (void)xq; (void)xsc; (void)hq; (void)hsc; (void)topk_w; (void)y_pair; (void)pf;
#endif
}

// ---- Standalone single-matrix test kernels for FP64 parity & TFLOPS microbenchmarks ----
__global__ void __launch_bounds__(kNT) k_test_sf4_gemm(
    const uint8_t* __restrict__ w_u8, const uint8_t* __restrict__ w_sc, float w_outer,
    const uint8_t* __restrict__ xa, float* __restrict__ y, int M, int N, int K) {
#if defined(INK_SM120_DEV)
    __shared__ alignas(16) uint8_t s_wqs[2][kWR * kWStride4];
    __shared__ alignas(16) uint8_t s_aqs[2][kTR * kAStride4];
    __shared__ uint32_t s_wsc[2][kWR];
    __shared__ uint32_t s_asc[2][kTR];
    __shared__ float    s_agb[2][kTR];

    const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
    const int wf = warp >> 2, wn = warp & 3, nb0 = 16 * wn;
    const int g = lane >> 2, tig = lane & 3, lm = lane >> 3, l8 = lane & 7;
    const int tidx_a = g + (tig & 1) * 8;
    const int a_off = (64 * wf + 8 * (lm & 1) + l8) * kWStride4 + 16 * (lm >> 1);
    const int b_off = (nb0 + 8 * (lm >> 1) + l8) * kAStride4 + 16 * (lm & 1);
    const int ur = tid >> 1, uj = tid & 1;

    const int rbase = blockIdx.x * kWR;
    const int row0 = blockIdx.y * kTR;
    const int nrows = min(kTR, M - row0);
    if (nrows <= 0 || rbase >= N) return;
    const int NS = K / 64;
    const size_t W_P_ROW = (size_t)(K / 2);
    const size_t W_S_ROW = (size_t)(K / 16);
    const size_t ACT_ROW_B = (size_t)NS * kAB;
    const int wrow_i = min(rbase + ur, N - 1);
    const uint8_t* wrow_p = w_u8 + (size_t)wrow_i * W_P_ROW;
    const uint8_t* wrow_s = w_sc + (size_t)wrow_i * W_S_ROW;

    if (tid < kTR && tid >= nrows) {
        s_asc[0][tid] = 0u; s_asc[1][tid] = 0u;
        s_agb[0][tid] = 0.0f; s_agb[1][tid] = 0.0f;
        for (int b = 0; b < 2; ++b) {
            uint4* z = (uint4*)(s_aqs[b] + tid * kAStride4);
            z[0] = make_uint4(0u, 0u, 0u, 0u);
            z[1] = make_uint4(0u, 0u, 0u, 0u);
        }
    }
    __syncthreads();

    auto issue = [&](int st, int buf) {
        cp16_async(s_wqs[buf] + ur * kWStride4 + 16 * uj, wrow_p + (size_t)st * 32 + 16 * uj);
        if (uj == 0) s_wsc[buf][ur] = *(const uint32_t*)(wrow_s + (size_t)st * 4);
        if (tid < 2 * kTR) {
            const int r = tid >> 1, h = tid & 1;
            if (r < nrows) cp16_async(s_aqs[buf] + r * kAStride4 + 16 * h, xa + (size_t)(row0 + r) * ACT_ROW_B + (size_t)st * kAB + 16 * h);
        } else if (tid < 3 * kTR) {
            const int r = tid - 2 * kTR;
            if (r < nrows) {
                const uint2 meta = *(const uint2*)(xa + (size_t)(row0 + r) * ACT_ROW_B + (size_t)st * kAB + 32);
                s_asc[buf][r] = meta.x;
                s_agb[buf][r] = __uint_as_float(meta.y);
            }
        }
        cp_commit();
    };

    issue(0, 0);
    float acc[4][2][4] = {};
    const bool on0 = nb0 < nrows;
    const bool on1 = nb0 + 8 < nrows;

    for (int st = 0; st < NS; ++st) {
        const int buf = st & 1, nxt = buf ^ 1;
        if (st + 1 < NS) { issue(st + 1, nxt); cp_wait1(); } else cp_wait0();
        __syncthreads();
        if (on0) {
            uint32_t bq[4];
            ldsm4(bq, s_aqs[buf] + b_off);
            const uint32_t sfb0 = s_asc[buf][nb0 + g];
            const uint32_t sfb1 = on1 ? s_asc[buf][nb0 + 8 + g] : 0u;
            const float gb00 = s_agb[buf][nb0 + 2 * tig + 0];
            const float gb01 = s_agb[buf][nb0 + 2 * tig + 1];
            const float gb10 = on1 ? s_agb[buf][nb0 + 8 + 2 * tig + 0] : 0.0f;
            const float gb11 = on1 ? s_agb[buf][nb0 + 8 + 2 * tig + 1] : 0.0f;
#pragma unroll
            for (int i = 0; i < 4; ++i) {
                uint32_t a[4];
                ldsm4(a, s_wqs[buf] + a_off + 16 * i * kWStride4);
                const uint32_t sfa = s_wsc[buf][64 * wf + 16 * i + tidx_a];
                float d0[4] = {};
                mma_nvfp4_m16n8k64(d0, a, bq[0], bq[1], sfa, sfb0);
                acc[i][0][0] = fmaf(d0[0], gb00, acc[i][0][0]);
                acc[i][0][1] = fmaf(d0[1], gb01, acc[i][0][1]);
                acc[i][0][2] = fmaf(d0[2], gb00, acc[i][0][2]);
                acc[i][0][3] = fmaf(d0[3], gb01, acc[i][0][3]);
                if (on1) {
                    float d1[4] = {};
                    mma_nvfp4_m16n8k64(d1, a, bq[2], bq[3], sfa, sfb1);
                    acc[i][1][0] = fmaf(d1[0], gb10, acc[i][1][0]);
                    acc[i][1][1] = fmaf(d1[1], gb11, acc[i][1][1]);
                    acc[i][1][2] = fmaf(d1[2], gb10, acc[i][1][2]);
                    acc[i][1][3] = fmaf(d1[3], gb11, acc[i][1][3]);
                }
            }
        }
        if (st + 1 < NS) __syncthreads();
    }

    if (on0) {
        const int rb = rbase + 64 * wf;
#pragma unroll
        for (int n = 0; n < 2; ++n) {
            if (n == 1 && !on1) break;
#pragma unroll
            for (int c = 0; c < 2; ++c) {
                const int r = nb0 + 8 * n + 2 * tig + c;
                if (r < nrows) {
                    float* dst = y + (size_t)(row0 + r) * N;
#pragma unroll
                    for (int i = 0; i < 4; ++i) {
                        const int c0 = rb + 16 * i + g;
                        if (c0 < N) dst[c0] = acc[i][n][c] * w_outer;
                        if (c0 + 8 < N) dst[c0 + 8] = acc[i][n][2 + c] * w_outer;
                    }
                }
            }
        }
    }
#else
    (void)w_u8; (void)w_sc; (void)w_outer; (void)xa; (void)y; (void)M; (void)N; (void)K;
#endif
}

__global__ void __launch_bounds__(kNT) k_test_mx8_gemm(
    const uint8_t* __restrict__ w_u8, const uint8_t* __restrict__ w_sc,
    const uint8_t* __restrict__ xq, const uint8_t* __restrict__ xsc,
    float* __restrict__ y, int M, int N, int K) {
#if defined(INK_SM120_DEV)
    __shared__ alignas(16) uint8_t s_wqs[2][kWR * kWStride4];
    __shared__ alignas(16) uint8_t s_aqs[2][kTR * kAStride4];
    __shared__ uint8_t s_wsc[2][kWR];
    __shared__ uint8_t s_asc[2][kTR];

    const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
    const int wf = warp >> 2, wn = warp & 3, nb0 = 16 * wn;
    const int g = lane >> 2, tig = lane & 3, lm = lane >> 3, l8 = lane & 7;
    const int tidx_a = g + (tig & 1) * 8;
    const int a_off = (64 * wf + 8 * (lm & 1) + l8) * kWStride4 + 16 * (lm >> 1);
    const int b_off = (nb0 + 8 * (lm >> 1) + l8) * kAStride4 + 16 * (lm & 1);
    const int ur = tid >> 1, uj = tid & 1;

    const int rbase = blockIdx.x * kWR;
    const int row0 = blockIdx.y * kTR;
    const int nrows = min(kTR, M - row0);
    if (nrows <= 0 || rbase >= N) return;
    const int NS = K / 32;
    const int wrow_i = min(rbase + ur, N - 1);
    const uint8_t* wrow_p = w_u8 + (size_t)wrow_i * K;
    const uint8_t* wrow_s = w_sc + (size_t)wrow_i * NS;

    if (tid < kTR && tid >= nrows) {
        s_asc[0][tid] = 0u; s_asc[1][tid] = 0u;
        for (int b = 0; b < 2; ++b) {
            uint4* z = (uint4*)(s_aqs[b] + tid * kAStride4);
            z[0] = make_uint4(0u, 0u, 0u, 0u);
            z[1] = make_uint4(0u, 0u, 0u, 0u);
        }
    }
    __syncthreads();

    auto issue = [&](int st, int buf) {
        cp16_async(s_wqs[buf] + ur * kWStride4 + 16 * uj, wrow_p + (size_t)st * 32 + 16 * uj);
        if (uj == 0) s_wsc[buf][ur] = wrow_s[st];
        if (tid < 2 * kTR) {
            const int r = tid >> 1, h = tid & 1;
            if (r < nrows) cp16_async(s_aqs[buf] + r * kAStride4 + 16 * h, xq + (size_t)(row0 + r) * K + (size_t)st * 32 + 16 * h);
        } else if (tid < 3 * kTR) {
            const int r = tid - 2 * kTR;
            if (r < nrows) s_asc[buf][r] = xsc[(size_t)(row0 + r) * NS + st];
        }
        cp_commit();
    };

    issue(0, 0);
    float acc[4][2][4] = {};
    const bool on0 = nb0 < nrows;
    const bool on1 = nb0 + 8 < nrows;

    for (int st = 0; st < NS; ++st) {
        const int buf = st & 1, nxt = buf ^ 1;
        if (st + 1 < NS) { issue(st + 1, nxt); cp_wait1(); } else cp_wait0();
        __syncthreads();
        if (on0) {
            uint32_t bq[4];
            ldsm4(bq, s_aqs[buf] + b_off);
            const uint32_t sfb0 = (uint32_t)s_asc[buf][nb0 + g];
            const uint32_t sfb1 = on1 ? (uint32_t)s_asc[buf][nb0 + 8 + g] : 0u;
#pragma unroll
            for (int i = 0; i < 4; ++i) {
                uint32_t a[4];
                ldsm4(a, s_wqs[buf] + a_off + 16 * i * kWStride4);
                const uint32_t sfa = (uint32_t)s_wsc[buf][64 * wf + 16 * i + tidx_a];
                mma_mxfp8_m16n8k32(acc[i][0], a, bq[0], bq[1], sfa, sfb0);
                if (on1) mma_mxfp8_m16n8k32(acc[i][1], a, bq[2], bq[3], sfa, sfb1);
            }
        }
        if (st + 1 < NS) __syncthreads();
    }

    if (on0) {
        const int rb = rbase + 64 * wf;
#pragma unroll
        for (int n = 0; n < 2; ++n) {
            if (n == 1 && !on1) break;
#pragma unroll
            for (int c = 0; c < 2; ++c) {
                const int r = nb0 + 8 * n + 2 * tig + c;
                if (r < nrows) {
                    float* dst = y + (size_t)(row0 + r) * N;
#pragma unroll
                    for (int i = 0; i < 4; ++i) {
                        const int c0 = rb + 16 * i + g;
                        if (c0 < N) dst[c0] = acc[i][n][c];
                        if (c0 + 8 < N) dst[c0 + 8] = acc[i][n][2 + c];
                    }
                }
            }
        }
    }
#else
    (void)w_u8; (void)w_sc; (void)xq; (void)xsc; (void)y; (void)M; (void)N; (void)K;
#endif
}

// ---- Opt-in RTX 5090 (STRATA_INK_RTX5090=1) BK=128 Pipelined SF4 MoE Kernel ----
// Doubles K per stage (64 B/row contiguous cp.async -> half the __syncthreads() barriers!),
// decouples MMA issue from FMA consumption, and disables per-thread TMA pf_l2 at TR>=32 while
// enabling L2 prefetch on BOTH GU and DN at TR<=16 (C64 bandwidth regime).
template <bool GU, int TR>
__global__ void __launch_bounds__(kNT) k_moe_sf4_tc_bk128(
    const uint8_t* __restrict__ sec4_base,
    const InkExpertDev* __restrict__ experts,
    InkMoeTables tb,
    const uint8_t* __restrict__ act,
    uint8_t* __restrict__ out_ha,
    const float* __restrict__ topk_w,
    float* __restrict__ y_pair,
    int pf) {
#if defined(INK_SM120_DEV)
    constexpr int ROWS_N   = GU ? (2 * kInter) : kHidden;
    constexpr int ROWS_K   = GU ? kHidden : kInter;
    constexpr int NS       = ROWS_K / 64;   // 36 (GU) or 18 (DN)
    constexpr int NS2      = NS / 2;        // 18 (GU) or 9 (DN)
    constexpr int NFB      = ROWS_N / kWR;
    constexpr int N_WN     = TR / 16;
    constexpr int N_WF     = 8 / N_WN;
    constexpr int W_PER_WF = kWR / N_WF;
    constexpr int NI       = W_PER_WF / 16;
    constexpr int kStr80   = 80;            // 64B qs (2x K64) + 16B pad (bank-conflict free ldmatrix!)
    constexpr size_t W_P_ROW   = ROWS_K / 2;
    constexpr size_t W_S_ROW   = ROWS_K / 16;
    constexpr size_t ACT_ROW_B = (size_t)NS * kAB;

    __shared__ alignas(16) uint8_t s_wqs[2][kWR * kStr80];
    __shared__ alignas(16) uint8_t s_aqs[2][TR * kStr80];
    __shared__ alignas(8)  uint32_t s_wsc[2][kWR][2];
    __shared__ alignas(8)  uint32_t s_asc[2][TR][2];
    __shared__ float    s_row_g[TR];
    __shared__ int32_t  s_row[TR];
    __shared__ float    s_am[N_WF][TR];
    __shared__ float    s_rw[TR];

    const int tid  = threadIdx.x;
    const int warp = tid >> 5;
    const int lane = tid & 31;
    const int wf   = warp / N_WN;
    const int wn   = warp % N_WN;
    const int nb0  = 16 * wn;
    const int g    = lane >> 2;
    const int tig  = lane & 3;
    const int lm   = lane >> 3;
    const int l8   = lane & 7;
    const int tidx_a = g + (tig & 1) * 8;

    const int a_off = (W_PER_WF * wf + 8 * (lm & 1) + l8) * kStr80 + 16 * (lm >> 1);
    const int b_off = (nb0 + 8 * (lm >> 1) + l8) * kStr80 + 16 * (lm & 1);
    const int ur = tid >> 1;
    const int uj = tid & 1;

    const int total_tiles = *tb.n_tiles;
    const int nwork = total_tiles * NFB;

    for (int w = blockIdx.x; w < nwork; w += gridDim.x) {
        const int2 tl = tb.tiles[w / NFB];
        const int fb = w % NFB;
        const int e = tl.x;
        const int row0 = tl.y;
        const InkExpertDev ed = experts[e];
        if ((GU ? ed.gu_code : ed.dn_code) != FMT_SF4) continue;
        const int nrows = min(TR, tb.off[e + 1] - row0);
        if (nrows <= 0) continue;

        const uint8_t* body = sec4_base + ed.body_off;
        float hdr[4];
        memcpy(hdr, body, 16);
        const float w_outer = GU ? (hdr[0] / hdr[1]) : (hdr[2] / hdr[3]);

        const uint8_t* base_p = GU ? (body + 16) : (body + ed.dn_off);
        const uint8_t* base_s = base_p + (size_t)ROWS_N * W_P_ROW;

        const int phys_row = GU ? (2 * (fb * 64 + 8 * (ur >> 4) + (ur & 7)) + ((ur >> 3) & 1))
                                : (fb * kWR + ur);
        const uint8_t* wrow_p = base_p + (size_t)phys_row * W_P_ROW;
        const uint8_t* wrow_s = base_s + (size_t)phys_row * W_S_ROW;

        __syncthreads();
        if (tid < TR) {
            if (tid < nrows) {
                const int sr = GU ? tb.tok_idx[row0 + tid] : (row0 + tid);
                s_row[tid] = sr;
                if constexpr (GU) {
                    s_row_g[tid] = *(const float*)(act + (size_t)sr * ACT_ROW_B + 36);
                } else {
                    s_rw[tid] = topk_w[tb.pairs[row0 + tid]];
                }
            } else {
                s_row[tid] = -1;
                if constexpr (GU) s_row_g[tid] = 0.0f;
                else              s_rw[tid] = 0.0f;
                for (int b = 0; b < 2; ++b) {
                    s_asc[b][tid][0] = 0u; s_asc[b][tid][1] = 0u;
                    uint4* z = (uint4*)(s_aqs[b] + tid * kStr80);
                    z[0] = make_uint4(0u, 0u, 0u, 0u);
                    z[1] = make_uint4(0u, 0u, 0u, 0u);
                    z[2] = make_uint4(0u, 0u, 0u, 0u);
                    z[3] = make_uint4(0u, 0u, 0u, 0u);
                }
            }
        }
        __syncthreads();

        auto issue_stage = [&](int s2, int buf) {
            const size_t woff = (size_t)s2 * 64 + 16 * uj;
            cp16_async(s_wqs[buf] + ur * kStr80 + 16 * uj,      wrow_p + woff);
            cp16_async(s_wqs[buf] + ur * kStr80 + 32 + 16 * uj, wrow_p + woff + 32);
            if (uj == 0) {
                cp8_async(&s_wsc[buf][ur][0], wrow_s + (size_t)s2 * 8);
            }
            if (tid < 2 * TR) {
                const int r = tid >> 1;
                const int h = tid & 1;
                if (r < nrows) {
                    const uint8_t* ap0 = act + (size_t)s_row[r] * ACT_ROW_B + (size_t)(2 * s2) * kAB;
                    const uint8_t* ap1 = ap0 + kAB;
                    cp16_async(s_aqs[buf] + r * kStr80 + 16 * h,      ap0 + 16 * h);
                    cp16_async(s_aqs[buf] + r * kStr80 + 32 + 16 * h, ap1 + 16 * h);
                }
            } else if (tid < 3 * TR) {
                const int r = tid - 2 * TR;
                if (r < nrows) {
                    const uint8_t* ap0 = act + (size_t)s_row[r] * ACT_ROW_B + (size_t)(2 * s2) * kAB;
                    cp4_async(&s_asc[buf][r][0], ap0 + 32);
                    cp4_async(&s_asc[buf][r][1], ap0 + kAB + 32);
                }
            }
            cp_commit();
        };

        if (pf && uj == 0) {
            pf_l2(wrow_p, (uint32_t)min((size_t)pf, W_P_ROW));
            if constexpr (GU) pf_l2(wrow_s, (uint32_t)W_S_ROW);
        }
        issue_stage(0, 0);

        float acc[NI][2][4] = {};
        const bool on0 = nb0 < nrows;
        const bool on1 = nb0 + 8 < nrows;

        for (int s2 = 0; s2 < NS2; ++s2) {
            const int buf = s2 & 1;
            const int nxt = buf ^ 1;
            if (pf && uj == 0 && (s2 % (pf / 64)) == 0 && (size_t)(s2 + pf / 64) * 64 < W_P_ROW) {
                const size_t nb = (size_t)(s2 + pf / 64) * 64;
                pf_l2(wrow_p + nb, (uint32_t)min((size_t)pf, W_P_ROW - nb));
            }
            if (s2 + 1 < NS2) {
                issue_stage(s2 + 1, nxt);
                cp_wait1();
            } else {
                cp_wait0();
            }
            __syncthreads();

            if (on0) {
#pragma unroll
                for (int sub = 0; sub < 2; ++sub) {
                    uint32_t bq[4];
                    ldsm4(bq, s_aqs[buf] + b_off + sub * 32);
                    const uint32_t sfb0 = s_asc[buf][nb0 + g][sub];
                    const uint32_t sfb1 = on1 ? s_asc[buf][nb0 + 8 + g][sub] : 0u;
#pragma unroll
                    for (int i = 0; i < NI; ++i) {
                        uint32_t a[4];
                        ldsm4(a, s_wqs[buf] + a_off + 16 * i * kStr80 + sub * 32);
                        const uint32_t sfa = s_wsc[buf][W_PER_WF * wf + 16 * i + tidx_a][sub];
                        mma_nvfp4_m16n8k64(acc[i][0], a, bq[0], bq[1], sfa, sfb0);
                        if (on1) mma_nvfp4_m16n8k64(acc[i][1], a, bq[2], bq[3], sfa, sfb1);
                    }
                }
            }
            if (s2 + 1 < NS2) __syncthreads();
        }

        if constexpr (GU) {
            float h_val[2][2][NI] = {};
            float am_sub0[2][2] = {}, am_sub1[2][2] = {};
            if (on0) {
#pragma unroll
                for (int n = 0; n < 2; ++n) {
                    if (n == 1 && !on1) break;
#pragma unroll
                    for (int c = 0; c < 2; ++c) {
                        const int r = nb0 + 8 * n + 2 * tig + c;
                        const float sc_r = (r < nrows) ? (w_outer * s_row_g[r]) : 0.0f;
#pragma unroll
                        for (int i = 0; i < NI; ++i) {
                            h_val[n][c][i] = situ_f32(acc[i][n][c] * sc_r, acc[i][n][2 + c] * sc_r);
                        }
                        float a0 = fabsf(h_val[n][c][0]);
                        if constexpr (NI >= 2) a0 = fmaxf(a0, fabsf(h_val[n][c][1]));
                        float a1 = 0.0f;
                        if constexpr (NI == 4) a1 = fmaxf(fabsf(h_val[n][c][2]), fabsf(h_val[n][c][3]));
#pragma unroll
                        for (int o = 4; o < 32; o <<= 1) {
                            a0 = fmaxf(a0, __shfl_xor_sync(0xFFFFFFFFu, a0, o));
                            if constexpr (NI == 4) a1 = fmaxf(a1, __shfl_xor_sync(0xFFFFFFFFu, a1, o));
                        }
                        am_sub0[n][c] = a0;
                        am_sub1[n][c] = a1;
                        if constexpr (NI == 1) {
                            if (g == 0 && r < TR) s_am[wf][r] = a0;
                        }
                    }
                }
            }
            if constexpr (NI == 1) __syncthreads();
            if (on0) {
                constexpr size_t H_ROW_B = (size_t)(kInter / 64) * kAB;
                constexpr float inv_6g = 1.0f / (kGHaConst * 6.0f);
#pragma unroll
                for (int n = 0; n < 2; ++n) {
                    if (n == 1 && !on1) break;
#pragma unroll
                    for (int c = 0; c < 2; ++c) {
                        const int r = nb0 + 8 * n + 2 * tig + c;
                        const bool valid = (r < nrows);
                        float s0 = 0.0f, s1 = 0.0f;
                        float sub0_am = am_sub0[n][c];
                        if constexpr (NI == 1) {
                            sub0_am = valid ? fmaxf(s_am[wf & ~1][r], s_am[wf | 1][r]) : 0.0f;
                        }
                        const uint8_t d0 = encode_ue4m3(sub0_am * inv_6g, s0);
                        const uint8_t d1 = (NI == 4) ? encode_ue4m3(am_sub1[n][c] * inv_6g, s1) : 0u;
                        const float inv0 = (kGHaConst * s0) > 0.0f ? (1.0f / (kGHaConst * s0)) : 0.0f;
                        const float inv1 = (NI == 4 && (kGHaConst * s1) > 0.0f) ? (1.0f / (kGHaConst * s1)) : 0.0f;

                        float qa[NI], qb[NI];
#pragma unroll
                        for (int i = 0; i < NI; ++i) {
                            qa[i] = h_val[n][c][i] * ((i < 2) ? inv0 : inv1);
                            qb[i] = __shfl_xor_sync(0xFFFFFFFFu, qa[i], 4);
                        }
                        if (valid) {
                            uint8_t* dst_slot = out_ha + (size_t)(row0 + r) * H_ROW_B + (size_t)fb * kAB;
                            if ((g & 1) == 0) {
                                const int bo = (4 * NI) * wf + (g >> 1);
#pragma unroll
                                for (int i = 0; i < NI; ++i) dst_slot[bo + 4 * i] = pack_e2m1x2(qa[i], qb[i]);
                            }
                            if (g == 0) {
                                if constexpr (NI == 4) {
                                    dst_slot[32 + 2 * wf + 0] = d0;
                                    dst_slot[32 + 2 * wf + 1] = d1;
                                } else if constexpr (NI == 2) {
                                    dst_slot[32 + wf] = d0;
                                } else {
                                    if ((wf & 1) == 0) dst_slot[32 + (wf >> 1)] = d0;
                                }
                            }
                        }
                    }
                }
            }
        } else if (on0) {
            const int rbase = fb * kWR + W_PER_WF * wf;
            const float wg = w_outer * kGHaConst;
#pragma unroll
            for (int n = 0; n < 2; ++n) {
                if (n == 1 && !on1) break;
#pragma unroll
                for (int c = 0; c < 2; ++c) {
                    const int r = nb0 + 8 * n + 2 * tig + c;
                    if (r < nrows) {
                        const float sc = wg * s_rw[r];
                        float* dst = y_pair + (size_t)tb.pairs[row0 + r] * kHidden;
#pragma unroll
                        for (int i = 0; i < NI; ++i) {
                            dst[rbase + 16 * i + g + 0] = acc[i][n][c] * sc;
                            dst[rbase + 16 * i + g + 8] = acc[i][n][2 + c] * sc;
                        }
                    }
                }
            }
        }
    }
#else
    (void)sec4_base; (void)experts; (void)tb; (void)act; (void)out_ha; (void)topk_w; (void)y_pair; (void)pf;
#endif
}

// ---- Opt-in RTX 5090 (STRATA_INK_RTX5090=1) BK=64 Pipelined MX8 MoE Kernel (L20..23) ----
template <bool GU, int TR>
__global__ void __launch_bounds__(kNT) k_moe_mx8_tc_bk64(
    const uint8_t* __restrict__ sec4_base,
    const InkExpertDev* __restrict__ experts,
    InkMoeTables tb,
    const uint8_t* __restrict__ xq,
    const uint8_t* __restrict__ xsc,
    uint8_t* __restrict__ hq,
    uint8_t* __restrict__ hsc,
    const float* __restrict__ topk_w,
    float* __restrict__ y_pair,
    int pf) {
#if defined(INK_SM120_DEV)
    constexpr int ROWS_N   = GU ? (2 * kInter) : kHidden;
    constexpr int ROWS_K   = GU ? kHidden : kInter;
    constexpr int NS       = ROWS_K / 32;  // 72 (GU) or 36 (DN)
    constexpr int NS2      = NS / 2;       // 36 (GU) or 18 (DN)
    constexpr int NFB      = ROWS_N / kWR;
    constexpr int N_WN     = TR / 16;
    constexpr int N_WF     = 8 / N_WN;
    constexpr int W_PER_WF = kWR / N_WF;
    constexpr int NI       = W_PER_WF / 16;
    constexpr int kStr80   = 80;

    __shared__ alignas(16) uint8_t s_wqs[2][kWR * kStr80];
    __shared__ alignas(16) uint8_t s_aqs[2][TR * kStr80];
    __shared__ uint8_t  s_wsc[2][kWR][2];
    __shared__ uint8_t  s_asc[2][TR][2];
    __shared__ int32_t  s_row[TR];
    __shared__ float    s_am[N_WF][TR];
    __shared__ float    s_rw[TR];

    const int tid  = threadIdx.x;
    const int warp = tid >> 5;
    const int lane = tid & 31;
    const int wf   = warp / N_WN;
    const int wn   = warp % N_WN;
    const int nb0  = 16 * wn;
    const int g    = lane >> 2;
    const int tig  = lane & 3;
    const int lm   = lane >> 3;
    const int l8   = lane & 7;
    const int tidx_a = g + (tig & 1) * 8;

    const int a_off = (W_PER_WF * wf + 8 * (lm & 1) + l8) * kStr80 + 16 * (lm >> 1);
    const int b_off = (nb0 + 8 * (lm >> 1) + l8) * kStr80 + 16 * (lm & 1);
    const int ur = tid >> 1;
    const int uj = tid & 1;

    const int total_tiles = *tb.n_tiles;
    const int nwork = total_tiles * NFB;

    for (int w = blockIdx.x; w < nwork; w += gridDim.x) {
        const int2 tl = tb.tiles[w / NFB];
        const int fb = w % NFB;
        const int e = tl.x;
        const int row0 = tl.y;
        const InkExpertDev ed = experts[e];
        if ((GU ? ed.gu_code : ed.dn_code) != FMT_MX8) continue;
        const int nrows = min(TR, tb.off[e + 1] - row0);
        if (nrows <= 0) continue;

        const uint8_t* body = sec4_base + ed.body_off;
        const uint8_t* base_p = GU ? body : (body + ed.dn_off);
        const uint8_t* base_s = base_p + (size_t)ROWS_N * ROWS_K;

        const int phys_row = GU ? (2 * (fb * 64 + 8 * (ur >> 4) + (ur & 7)) + ((ur >> 3) & 1))
                                : (fb * kWR + ur);
        const uint8_t* wrow_p = base_p + (size_t)phys_row * ROWS_K;
        const uint8_t* wrow_s = base_s + (size_t)phys_row * NS;

        __syncthreads();
        if (tid < TR) {
            if (tid < nrows) {
                s_row[tid] = GU ? tb.tok_idx[row0 + tid] : (row0 + tid);
                if constexpr (!GU) s_rw[tid] = topk_w[tb.pairs[row0 + tid]];
            } else {
                s_row[tid] = -1;
                if constexpr (!GU) s_rw[tid] = 0.0f;
                for (int b = 0; b < 2; ++b) {
                    s_asc[b][tid][0] = 0u; s_asc[b][tid][1] = 0u;
                    uint4* z = (uint4*)(s_aqs[b] + tid * kStr80);
                    z[0] = make_uint4(0u, 0u, 0u, 0u);
                    z[1] = make_uint4(0u, 0u, 0u, 0u);
                    z[2] = make_uint4(0u, 0u, 0u, 0u);
                    z[3] = make_uint4(0u, 0u, 0u, 0u);
                }
            }
        }
        __syncthreads();

        auto issue_stage = [&](int s2, int buf) {
            const size_t woff = (size_t)s2 * 64 + 16 * uj;
            cp16_async(s_wqs[buf] + ur * kStr80 + 16 * uj,      wrow_p + woff);
            cp16_async(s_wqs[buf] + ur * kStr80 + 32 + 16 * uj, wrow_p + woff + 32);
            if (uj == 0) {
                s_wsc[buf][ur][0] = wrow_s[2 * s2 + 0];
                s_wsc[buf][ur][1] = wrow_s[2 * s2 + 1];
            }
            if (tid < 2 * TR) {
                const int r = tid >> 1;
                const int h = tid & 1;
                if (r < nrows) {
                    const uint8_t* ap = xq + (size_t)s_row[r] * ROWS_K + (size_t)s2 * 64 + 16 * h;
                    cp16_async(s_aqs[buf] + r * kStr80 + 16 * h,      ap);
                    cp16_async(s_aqs[buf] + r * kStr80 + 32 + 16 * h, ap + 32);
                }
            } else if (tid < 3 * TR) {
                const int r = tid - 2 * TR;
                if (r < nrows) {
                    s_asc[buf][r][0] = xsc[(size_t)s_row[r] * NS + 2 * s2 + 0];
                    s_asc[buf][r][1] = xsc[(size_t)s_row[r] * NS + 2 * s2 + 1];
                }
            }
            cp_commit();
        };

        if (pf && uj == 0) pf_l2(wrow_p, (uint32_t)min((size_t)pf, (size_t)ROWS_K));
        issue_stage(0, 0);

        float acc[NI][2][4] = {};
        const bool on0 = nb0 < nrows;
        const bool on1 = nb0 + 8 < nrows;

        for (int s2 = 0; s2 < NS2; ++s2) {
            const int buf = s2 & 1;
            const int nxt = buf ^ 1;
            if (pf && uj == 0 && (s2 % (pf / 64)) == 0 && (size_t)(s2 + pf / 64) * 64 < (size_t)ROWS_K) {
                const size_t nb = (size_t)(s2 + pf / 64) * 64;
                pf_l2(wrow_p + nb, (uint32_t)min((size_t)pf, (size_t)ROWS_K - nb));
            }
            if (s2 + 1 < NS2) {
                issue_stage(s2 + 1, nxt);
                cp_wait1();
            } else {
                cp_wait0();
            }
            __syncthreads();

            if (on0) {
#pragma unroll
                for (int sub = 0; sub < 2; ++sub) {
                    uint32_t bq[4];
                    ldsm4(bq, s_aqs[buf] + b_off + sub * 32);
                    const uint32_t sfb0 = (uint32_t)s_asc[buf][nb0 + g][sub];
                    const uint32_t sfb1 = on1 ? (uint32_t)s_asc[buf][nb0 + 8 + g][sub] : 0u;
#pragma unroll
                    for (int i = 0; i < NI; ++i) {
                        uint32_t a[4];
                        ldsm4(a, s_wqs[buf] + a_off + 16 * i * kStr80 + sub * 32);
                        const uint32_t sfa = (uint32_t)s_wsc[buf][W_PER_WF * wf + 16 * i + tidx_a][sub];
                        mma_mxfp8_m16n8k32(acc[i][0], a, bq[0], bq[1], sfa, sfb0);
                        if (on1) mma_mxfp8_m16n8k32(acc[i][1], a, bq[2], bq[3], sfa, sfb1);
                    }
                }
            }
            if (s2 + 1 < NS2) __syncthreads();
        }

        if constexpr (GU) {
            float hv[2][2][NI] = {};
            float am_loc[2][2] = {};
            if (on0) {
#pragma unroll
                for (int n = 0; n < 2; ++n) {
                    if (n == 1 && !on1) break;
#pragma unroll
                    for (int c = 0; c < 2; ++c) {
                        float am = 0.0f;
#pragma unroll
                        for (int i = 0; i < NI; ++i) {
                            hv[n][c][i] = situ_f32(acc[i][n][c], acc[i][n][2 + c]);
                            am = fmaxf(am, fabsf(hv[n][c][i]));
                        }
#pragma unroll
                        for (int o = 4; o < 32; o <<= 1) am = fmaxf(am, __shfl_xor_sync(0xFFFFFFFFu, am, o));
                        am_loc[n][c] = am;
                        const int r = nb0 + 8 * n + 2 * tig + c;
                        if (g == 0 && r < TR) s_am[wf][r] = am;
                    }
                }
            }
            if constexpr (NI < 4) __syncthreads();
            if (on0) {
                constexpr int WF_PER_BLK32 = N_WF / 2;
                const int half_id = wf / WF_PER_BLK32;
                const int sub_in  = wf % WF_PER_BLK32;
                const int blk32   = 2 * fb + half_id;
#pragma unroll
                for (int n = 0; n < 2; ++n) {
                    if (n == 1 && !on1) break;
#pragma unroll
                    for (int c = 0; c < 2; ++c) {
                        const int r = nb0 + 8 * n + 2 * tig + c;
                        if (r < nrows) {
                            float am32 = am_loc[n][c];
                            if constexpr (NI < 4) {
                                am32 = 0.0f;
#pragma unroll
                                for (int k = 0; k < WF_PER_BLK32; ++k) am32 = fmaxf(am32, s_am[half_id * WF_PER_BLK32 + k][r]);
                            }
                            float inv_sc = 0.0f;
                            const uint8_t sc_b = encode_ue8m0(am32, inv_sc);
                            uint8_t* dst_q = hq + (size_t)(row0 + r) * kInter + blk32 * 32 + sub_in * (8 * NI);
#pragma unroll
                            for (int i = 0; i < NI; ++i) dst_q[8 * i + g] = encode_e4m3_signed(hv[n][c][i] * inv_sc);
                            if (g == 0 && sub_in == 0) hsc[(size_t)(row0 + r) * (kInter / 32) + blk32] = sc_b;
                        }
                    }
                }
            }
        } else if (on0) {
            const int rbase = fb * kWR + W_PER_WF * wf;
#pragma unroll
            for (int n = 0; n < 2; ++n) {
                if (n == 1 && !on1) break;
#pragma unroll
                for (int c = 0; c < 2; ++c) {
                    const int r = nb0 + 8 * n + 2 * tig + c;
                    if (r < nrows) {
                        const float sc = s_rw[r];
                        float* dst = y_pair + (size_t)tb.pairs[row0 + r] * kHidden;
#pragma unroll
                        for (int i = 0; i < NI; ++i) {
                            dst[rbase + 16 * i + g + 0] = acc[i][n][c] * sc;
                            dst[rbase + 16 * i + g + 8] = acc[i][n][2 + c] * sc;
                        }
                    }
                }
            }
        }
    }
#else
    (void)sec4_base; (void)experts; (void)tb; (void)xq; (void)xsc; (void)hq; (void)hsc; (void)topk_w; (void)y_pair; (void)pf;
#endif
}

// ---- Opt-In BK=128 XOR-Swizzled TR=128 NVFP4 MoE Kernel (512 threads, 41.5 KB smem, kGHaConst!) ----
// Halves __syncthreads() barriers (18 GU / 9 DN) AND uses cp8/cp4_async + constant SITU scale kGHaConst!
constexpr float kGHaConst5060 = 100.0f / (6.0f * 448.0f);

template <bool GU, bool ATOMIC_OUT = false>
__global__ void __launch_bounds__(512) k_moe_sf4_tc_tr128_s_bk128(
    const uint8_t* __restrict__ sec4_base,
    const InkExpertDev* __restrict__ experts,
    InkMoeTables tb,
    const uint8_t* __restrict__ act,
    uint8_t* __restrict__ out_ha,
    const float* __restrict__ topk_w,
    float* __restrict__ y_pair) {
#if defined(INK_SM120_DEV)
    constexpr int TR       = 128;
    constexpr int ROWS_N   = GU ? (2 * kInter) : kHidden;
    constexpr int ROWS_K   = GU ? kHidden : kInter;
    constexpr int NS       = ROWS_K / 64;
    constexpr int NS2      = NS / 2;
    constexpr int NFB      = ROWS_N / kWR;
    constexpr int W_PER_WF = 32;
    constexpr int NI       = 2;
    constexpr int kStr64   = 64;
    constexpr size_t W_P_ROW   = ROWS_K / 2;
    constexpr size_t W_S_ROW   = ROWS_K / 16;
    constexpr size_t ACT_ROW_B = (size_t)NS * kAB;

    __shared__ alignas(16) uint8_t s_wqs[2][kWR * kStr64];
    __shared__ alignas(16) uint8_t s_aqs[2][TR * kStr64];
    __shared__ alignas(8)  uint32_t s_wsc[2][kWR][2];
    __shared__ alignas(8)  uint32_t s_asc[2][TR][2];
    __shared__ float    s_row_g[TR];
    __shared__ int32_t  s_row[TR];
    __shared__ int32_t  s_tok[TR];
    __shared__ float    s_rw[TR];

    auto swiz64 = [](int row, int col16) -> int { return row * kStr64 + ((col16 ^ (row & 3)) << 4); };

    const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
    const int wf = warp >> 2, wn = warp & 3, nb0 = 32 * wn;
    const int g = lane >> 2, tig = lane & 3, lm = lane >> 3, l8 = lane & 7;
    const int tidx_a = g + (tig & 1) * 8;
    const int a_row_base = W_PER_WF * wf + 8 * (lm & 1) + l8, a_col_in = lm >> 1;
    const int b_row0 = nb0 + 8 * (lm >> 1) + l8, b_row1 = b_row0 + 16, b_col_in = lm & 1;

    const bool is_w_ld = (tid < 256);
    const int ur = tid >> 1, uj = tid & 1;
    const int ar = (tid - 256) >> 1, ah = tid & 1;

    const int total_tiles = *tb.n_tiles;
    const int nwork = total_tiles * NFB;

    for (int w = blockIdx.x; w < nwork; w += gridDim.x) {
        const int2 tl = tb.tiles[w / NFB];
        const int fb = w % NFB, e = tl.x, row0 = tl.y;
        const InkExpertDev ed = experts[e];
        if ((GU ? ed.gu_code : ed.dn_code) != FMT_SF4) continue;
        const int nrows = min(TR, tb.off[e + 1] - row0);
        if (nrows <= 0) continue;

        const uint8_t* body = sec4_base + ed.body_off;
        float hdr[4];
        memcpy(hdr, body, 16);
        const float w_outer = GU ? (hdr[0] / hdr[1]) : (hdr[2] / hdr[3]);
        const uint8_t* base_p = GU ? (body + 16) : (body + ed.dn_off);
        const uint8_t* base_s = base_p + (size_t)ROWS_N * W_P_ROW;

        const uint8_t* wrow_p = nullptr;
        const uint8_t* wrow_s = nullptr;
        if (is_w_ld) {
            const int phys_row = GU ? (2 * (fb * 64 + 8 * (ur >> 4) + (ur & 7)) + ((ur >> 3) & 1)) : (fb * kWR + ur);
            wrow_p = base_p + (size_t)phys_row * W_P_ROW;
            wrow_s = base_s + (size_t)phys_row * W_S_ROW;
        }

        __syncthreads();
        if (tid < TR) {
            if (tid < nrows) {
                const int tk = tb.tok_idx[row0 + tid];
                const int sr = GU ? tk : (row0 + tid);
                s_row[tid] = sr;
                if constexpr (GU) s_row_g[tid] = *(const float*)(act + (size_t)sr * ACT_ROW_B + 36);
                else { s_tok[tid] = tk; s_rw[tid] = topk_w[tb.pairs[row0 + tid]]; }
            } else {
                s_row[tid] = -1;
                if constexpr (GU) s_row_g[tid] = 0.0f;
                else { s_tok[tid] = 0; s_rw[tid] = 0.0f; }
                for (int b = 0; b < 2; ++b) {
                    s_asc[b][tid][0] = 0u; s_asc[b][tid][1] = 0u;
                    uint4* z = (uint4*)(s_aqs[b] + tid * kStr64);
                    z[0] = make_uint4(0u,0u,0u,0u); z[1] = make_uint4(0u,0u,0u,0u);
                    z[2] = make_uint4(0u,0u,0u,0u); z[3] = make_uint4(0u,0u,0u,0u);
                }
            }
        }
        __syncthreads();

        auto issue_stage = [&](int s2, int buf) {
            if (is_w_ld) {
                const size_t woff = (size_t)s2 * 64 + 16 * uj;
                cp16_async(s_wqs[buf] + swiz64(ur, uj),     wrow_p + woff);
                cp16_async(s_wqs[buf] + swiz64(ur, 2 + uj), wrow_p + woff + 32);
                if (uj == 0) cp8_async(&s_wsc[buf][ur][0], wrow_s + (size_t)s2 * 8);
            } else if (ar < nrows) {
                const uint8_t* ap0 = act + (size_t)s_row[ar] * ACT_ROW_B + (size_t)(2 * s2) * kAB;
                const uint8_t* ap1 = ap0 + kAB;
                cp16_async(s_aqs[buf] + swiz64(ar, ah),     ap0 + 16 * ah);
                cp16_async(s_aqs[buf] + swiz64(ar, 2 + ah), ap1 + 16 * ah);
                if (ah == 0) { cp4_async(&s_asc[buf][ar][0], ap0 + 32); cp4_async(&s_asc[buf][ar][1], ap1 + 32); }
            }
            cp_commit();
        };

        if (is_w_ld && uj == 0) {
            pf_l2(wrow_p, (uint32_t)min((size_t)256, W_P_ROW));
            if constexpr (GU) pf_l2(wrow_s, (uint32_t)W_S_ROW);
        }
        issue_stage(0, 0);

        float acc[NI][4][4] = {};
        const bool on0 = nb0 < nrows, on1 = nb0 + 8 < nrows, on2 = nb0 + 16 < nrows, on3 = nb0 + 24 < nrows;

        for (int s2 = 0; s2 < NS2; ++s2) {
            const int buf = s2 & 1, nxt = buf ^ 1;
            if (is_w_ld && uj == 0 && ((s2 & 3) == 0) && (size_t)(s2 + 4) * 64 < W_P_ROW) {
                const size_t nb = (size_t)(s2 + 4) * 64;
                pf_l2(wrow_p + nb, (uint32_t)min((size_t)256, W_P_ROW - nb));
            }
            if (s2 + 1 < NS2) { issue_stage(s2 + 1, nxt); cp_wait1(); } else cp_wait0();
            __syncthreads();
            if (on0) {
#pragma unroll
                for (int sub = 0; sub < 2; ++sub) {
                    uint32_t bq0[4], bq1[4];
                    ldsm4(bq0, s_aqs[buf] + swiz64(b_row0, sub * 2 + b_col_in));
                    if (on2) ldsm4(bq1, s_aqs[buf] + swiz64(b_row1, sub * 2 + b_col_in));
                    const uint32_t sfb0 = s_asc[buf][nb0 + g][sub];
                    const uint32_t sfb1 = on1 ? s_asc[buf][nb0 + 8 + g][sub] : 0u;
                    const uint32_t sfb2 = on2 ? s_asc[buf][nb0 + 16 + g][sub] : 0u;
                    const uint32_t sfb3 = on3 ? s_asc[buf][nb0 + 24 + g][sub] : 0u;
#pragma unroll
                    for (int i = 0; i < NI; ++i) {
                        uint32_t a[4];
                        ldsm4(a, s_wqs[buf] + swiz64(a_row_base + 16 * i, sub * 2 + a_col_in));
                        const uint32_t sfa = s_wsc[buf][W_PER_WF * wf + 16 * i + tidx_a][sub];
                        mma_nvfp4_m16n8k64(acc[i][0], a, bq0[0], bq0[1], sfa, sfb0);
                        if (on1) mma_nvfp4_m16n8k64(acc[i][1], a, bq0[2], bq0[3], sfa, sfb1);
                        if (on2) mma_nvfp4_m16n8k64(acc[i][2], a, bq1[0], bq1[1], sfa, sfb2);
                        if (on3) mma_nvfp4_m16n8k64(acc[i][3], a, bq1[2], bq1[3], sfa, sfb3);
                    }
                }
            }
            if (s2 + 1 < NS2) __syncthreads();
        }

        if constexpr (GU) {
            if (on0) {
                constexpr size_t H_ROW_B = (size_t)(kInter / 64) * kAB;
                constexpr float inv_6g = 1.0f / (kGHaConst5060 * 6.0f);
#pragma unroll
                for (int n = 0; n < 4; ++n) {
                    if ((n == 1 && !on1) || (n == 2 && !on2) || (n == 3 && !on3)) break;
#pragma unroll
                    for (int c = 0; c < 2; ++c) {
                        const int r = nb0 + 8 * n + 2 * tig + c;
                        const bool valid = (r < nrows);
                        const float sc_r = valid ? (w_outer * s_row_g[r]) : 0.0f;
                        const float h0 = situ_f32(acc[0][n][c] * sc_r, acc[0][n][2 + c] * sc_r);
                        const float h1 = situ_f32(acc[1][n][c] * sc_r, acc[1][n][2 + c] * sc_r);
                        float a0 = fmaxf(fabsf(h0), fabsf(h1));
#pragma unroll
                        for (int o = 4; o < 32; o <<= 1) a0 = fmaxf(a0, __shfl_xor_sync(0xFFFFFFFFu, a0, o));
                        float s0 = 0.0f;
                        const uint8_t d0 = encode_ue4m3(a0 * inv_6g, s0);
                        const float inv0 = (kGHaConst5060 * s0) > 0.0f ? (1.0f / (kGHaConst5060 * s0)) : 0.0f;
                        const float qa0 = h0 * inv0, qb0 = __shfl_xor_sync(0xFFFFFFFFu, qa0, 4);
                        const float qa1 = h1 * inv0, qb1 = __shfl_xor_sync(0xFFFFFFFFu, qa1, 4);
                        if (valid) {
                            uint8_t* dst_slot = out_ha + (size_t)(row0 + r) * H_ROW_B + (size_t)fb * kAB;
                            if ((g & 1) == 0) {
                                const int bo = 8 * wf + (g >> 1);
                                dst_slot[bo + 0] = pack_e2m1x2(qa0, qb0);
                                dst_slot[bo + 4] = pack_e2m1x2(qa1, qb1);
                            }
                            if (g == 0) dst_slot[32 + wf] = d0;
                        }
                    }
                }
            }
        } else if (on0) {
            const int rbase = fb * kWR + W_PER_WF * wf;
            const float wg = w_outer * kGHaConst5060;
#pragma unroll
            for (int n = 0; n < 4; ++n) {
                if ((n == 1 && !on1) || (n == 2 && !on2) || (n == 3 && !on3)) break;
#pragma unroll
                for (int c = 0; c < 2; ++c) {
                    const int r = nb0 + 8 * n + 2 * tig + c;
                    if (r < nrows) {
                        const float sc = wg * s_rw[r];
                        if constexpr (ATOMIC_OUT) {
                            float* dst = y_pair + (size_t)s_tok[r] * kHidden;
#pragma unroll
                            for (int i = 0; i < NI; ++i) {
                                atomicAdd(dst + rbase + 16 * i + g + 0, acc[i][n][c] * sc);
                                atomicAdd(dst + rbase + 16 * i + g + 8, acc[i][n][2 + c] * sc);
                            }
                        } else {
                            float* dst = y_pair + (size_t)tb.pairs[row0 + r] * kHidden;
#pragma unroll
                            for (int i = 0; i < NI; ++i) {
                                dst[rbase + 16 * i + g + 0] = acc[i][n][c] * sc;
                                dst[rbase + 16 * i + g + 8] = acc[i][n][2 + c] * sc;
                            }
                        }
                    }
                }
            }
        }
    }
#else
    (void)sec4_base; (void)experts; (void)tb; (void)act; (void)out_ha; (void)topk_w; (void)y_pair;
#endif
}

// ---- Opt-in Blackwell sm_120a Dense NVFP4 GEMM (reads w_outer directly on GPU -> 100% CUDA Graph capturable!) ----
template <bool DIRECT>
__global__ void __launch_bounds__(kNT) k_dense_sf4_tc(
    const uint8_t* __restrict__ w_sf4,  // .strata v1.1 FMT_SF4: [float w_outer, float x_sc, qs[N*K/2], sc[N*K/16]]
    const uint8_t* __restrict__ xa,     // [M, K/64, 48]
    float* __restrict__ y,              // [M, N]
    int M, int N, int K, int pf) {
#if defined(INK_SM120_DEV)
    __shared__ alignas(16) uint8_t  s_wqs[2][kWR * kWStride4];
    __shared__ alignas(16) uint32_t s_wsc[2][kWR];
    __shared__ alignas(16) uint8_t  s_aqs[2][kTR * kAStride4];
    __shared__ alignas(16) uint32_t s_asc[2][kTR];
    __shared__ alignas(16) float    s_agb[2][kTR];

    const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
    const int wf = warp >> 2, wn = warp & 3, nb0 = 16 * wn;
    const int g = lane >> 2, tig = lane & 3, lm = lane >> 3, l8 = lane & 7;
    const int tidx_a = g + (tig & 1) * 8;
    const int a_off = (64 * wf + 8 * (lm & 1) + l8) * kWStride4 + 16 * (lm >> 1);
    const int b_off = (nb0 + 8 * (lm >> 1) + l8) * kAStride4 + 16 * (lm & 1);
    const int ur = tid >> 1, uj = tid & 1;

    const int rbase = blockIdx.x * kWR;
    const int row0  = blockIdx.y * kTR;
    const int nrows = min(kTR, M - row0);
    if (nrows <= 0 || rbase >= N) return;

    const float    w_outer = *reinterpret_cast<const float*>(w_sf4);
    const uint8_t* w_u8    = w_sf4 + 16u;
    const size_t   W_P_ROW = (size_t)(K / 2);
    const size_t   W_S_ROW = (size_t)(K / 16);
    const uint8_t* w_sc    = w_u8 + (size_t)N * W_P_ROW;
    const int      NS      = K / 64;
    const size_t   ACT_ROW_B = (size_t)NS * kAB;
    const int      wrow_i  = min(rbase + ur, N - 1);
    const uint8_t* wrow_p  = w_u8 + (size_t)wrow_i * W_P_ROW;
    const uint8_t* wrow_s  = w_sc + (size_t)wrow_i * W_S_ROW;

    if (tid < kTR) {
        if (tid >= nrows) {
            s_asc[0][tid] = 0u; s_asc[1][tid] = 0u;
            s_agb[0][tid] = 0.0f; s_agb[1][tid] = 0.0f;
            for (int b = 0; b < 2; ++b) {
                uint4* z = (uint4*)(s_aqs[b] + tid * kAStride4);
                z[0] = make_uint4(0u, 0u, 0u, 0u);
                z[1] = make_uint4(0u, 0u, 0u, 0u);
            }
        } else if constexpr (DIRECT) {
            s_agb[0][tid] = *reinterpret_cast<const float*>(xa + (size_t)(row0 + tid) * ACT_ROW_B + 36);
        }
    }
    __syncthreads();

    auto issue = [&](int st, int buf) {
        cp16_async(s_wqs[buf] + ur * kWStride4 + 16 * uj, wrow_p + (size_t)st * 32 + 16 * uj);
        if (uj == 0) s_wsc[buf][ur] = *(const uint32_t*)(wrow_s + (size_t)st * 4);
        if (tid < 2 * kTR) {
            const int r = tid >> 1, h = tid & 1;
            if (r < nrows) cp16_async(s_aqs[buf] + r * kAStride4 + 16 * h, xa + (size_t)(row0 + r) * ACT_ROW_B + (size_t)st * kAB + 16 * h);
        } else if (tid < 3 * kTR) {
            const int r = tid - 2 * kTR;
            if (r < nrows) {
                if constexpr (DIRECT) {
                    s_asc[buf][r] = *(const uint32_t*)(xa + (size_t)(row0 + r) * ACT_ROW_B + (size_t)st * kAB + 32);
                } else {
                    const uint2 meta = *(const uint2*)(xa + (size_t)(row0 + r) * ACT_ROW_B + (size_t)st * kAB + 32);
                    s_asc[buf][r] = meta.x;
                    s_agb[buf][r] = __uint_as_float(meta.y);
                }
            }
        }
        cp_commit();
    };

    if (pf && uj == 0) {
        pf_l2(wrow_p, (uint32_t)min((size_t)pf, W_P_ROW));
        pf_l2(wrow_s, (uint32_t)W_S_ROW);
    }
    issue(0, 0);
    float acc[4][2][4] = {};
    const bool on0 = nb0 < nrows;
    const bool on1 = nb0 + 8 < nrows;

    for (int st = 0; st < NS; ++st) {
        const int buf = st & 1, nxt = buf ^ 1;
        if (pf && uj == 0 && (st % (pf / 32)) == 0 && (size_t)(st + pf / 32) * 32 < W_P_ROW) {
            const size_t nb = (size_t)(st + pf / 32) * 32;
            pf_l2(wrow_p + nb, (uint32_t)min((size_t)pf, W_P_ROW - nb));
        }
        if (st + 1 < NS) { issue(st + 1, nxt); cp_wait1(); } else cp_wait0();
        __syncthreads();
        if (on0) {
            uint32_t bq[4];
            ldsm4(bq, s_aqs[buf] + b_off);
            const uint32_t sfb0 = s_asc[buf][nb0 + g];
            const uint32_t sfb1 = on1 ? s_asc[buf][nb0 + 8 + g] : 0u;
            if constexpr (DIRECT) {
#pragma unroll
                for (int i = 0; i < 4; ++i) {
                    uint32_t a[4];
                    ldsm4(a, s_wqs[buf] + a_off + 16 * i * kWStride4);
                    const uint32_t sfa = s_wsc[buf][64 * wf + 16 * i + tidx_a];
                    mma_nvfp4_m16n8k64(acc[i][0], a, bq[0], bq[1], sfa, sfb0);
                    if (on1) mma_nvfp4_m16n8k64(acc[i][1], a, bq[2], bq[3], sfa, sfb1);
                }
            } else {
                const float gb00 = s_agb[buf][nb0 + 2 * tig + 0];
                const float gb01 = s_agb[buf][nb0 + 2 * tig + 1];
                const float gb10 = on1 ? s_agb[buf][nb0 + 8 + 2 * tig + 0] : 0.0f;
                const float gb11 = on1 ? s_agb[buf][nb0 + 8 + 2 * tig + 1] : 0.0f;
#pragma unroll
                for (int i = 0; i < 4; ++i) {
                    uint32_t a[4];
                    ldsm4(a, s_wqs[buf] + a_off + 16 * i * kWStride4);
                    const uint32_t sfa = s_wsc[buf][64 * wf + 16 * i + tidx_a];
                    float d0[4] = {};
                    mma_nvfp4_m16n8k64(d0, a, bq[0], bq[1], sfa, sfb0);
                    acc[i][0][0] = fmaf(d0[0], gb00, acc[i][0][0]);
                    acc[i][0][1] = fmaf(d0[1], gb01, acc[i][0][1]);
                    acc[i][0][2] = fmaf(d0[2], gb00, acc[i][0][2]);
                    acc[i][0][3] = fmaf(d0[3], gb01, acc[i][0][3]);
                    if (on1) {
                        float d1[4] = {};
                        mma_nvfp4_m16n8k64(d1, a, bq[2], bq[3], sfa, sfb1);
                        acc[i][1][0] = fmaf(d1[0], gb10, acc[i][1][0]);
                        acc[i][1][1] = fmaf(d1[1], gb11, acc[i][1][1]);
                        acc[i][1][2] = fmaf(d1[2], gb10, acc[i][1][2]);
                        acc[i][1][3] = fmaf(d1[3], gb11, acc[i][1][3]);
                    }
                }
            }
        }
        if (st + 1 < NS) __syncthreads();
    }

    if (on0) {
        const int rb = rbase + 64 * wf;
#pragma unroll
        for (int n = 0; n < 2; ++n) {
            if (n == 1 && !on1) break;
#pragma unroll
            for (int c = 0; c < 2; ++c) {
                const int r = nb0 + 8 * n + 2 * tig + c;
                if (r < nrows) {
                    const float sc = DIRECT ? (w_outer * s_agb[0][r]) : w_outer;
                    float* dst = y + (size_t)(row0 + r) * N;
#pragma unroll
                    for (int i = 0; i < 4; ++i) {
                        const int c0 = rb + 16 * i + g;
                        if (c0 < N) dst[c0] = acc[i][n][c] * sc;
                        if (c0 + 8 < N) dst[c0 + 8] = acc[i][n][2 + c] * sc;
                    }
                }
            }
        }
    }
#else
    (void)w_sf4; (void)xa; (void)y; (void)M; (void)N; (void)K; (void)pf;
#endif
}


}  // namespace

bool available() {
#if !defined(CUDART_VERSION) || (CUDART_VERSION < 12080)
    return false;
#else
    // per device (#6796, #6808): a cache filled on stage 0 told an sm_86 stage that it could run the sm_120 FP4 kernels
    static strata::core::PerDeviceFlag flag;
    int dev = 0;
    if (cudaGetDevice(&dev) != cudaSuccess) return false;
    return flag.get(dev, [dev] {
        cudaDeviceProp p{};
        if (cudaGetDeviceProperties(&p, dev) != cudaSuccess || !strata::core::ink_cap_sm120(p.major, p.minor)) return false;
        cudaFuncAttributes attr{};
        return cudaFuncGetAttributes(&attr, (const void*)k_moe_sf4_tc<true, 64>) == cudaSuccess && attr.binaryVersion >= 120;
    });
#endif
}



// ---- Opt-in RTX 5090 Prefill TR=128 (kNT=512, 16 warps, BK=128 XOR-swizzled) SF4 MoE Kernel ----
// Halves expert weight reads AND halves __syncthreads() barriers (18 GU / 9 DN) in 41.5 KB static smem!
// Optional y_out_atomic eliminates 302 MB y_pair write+read by accumulating directly into 37.7 MB L2-resident d_moe_raw!
template <bool GU>
__global__ void __launch_bounds__(512) k_moe_sf4_tc_tr128(
    const uint8_t* __restrict__ sec4_base,
    const InkExpertDev* __restrict__ experts,
    InkMoeTables tb,
    const uint8_t* __restrict__ act,
    uint8_t* __restrict__ out_ha,
    const float* __restrict__ topk_w,
    float* __restrict__ y_pair,
    float* __restrict__ y_out_atomic = nullptr) {
#if defined(INK_SM120_DEV)
    constexpr int TR       = 128;
    constexpr int ROWS_N   = GU ? (2 * kInter) : kHidden;
    constexpr int ROWS_K   = GU ? kHidden : kInter;
    constexpr int NS       = ROWS_K / 64;   // 36 (GU) or 18 (DN)
    constexpr int NS2      = NS / 2;        // 18 (GU) or 9 (DN)
    constexpr int NFB      = ROWS_N / kWR;
    constexpr int W_PER_WF = 32;
    constexpr int NI       = 2;
    constexpr int kStr64   = 64;            // 64B (2x K64) with XOR-swizzle -> ZERO bank conflicts, 41.5 KB total smem!
    constexpr size_t W_P_ROW   = ROWS_K / 2;
    constexpr size_t W_S_ROW   = ROWS_K / 16;
    constexpr size_t ACT_ROW_B = (size_t)NS * kAB;

    __shared__ alignas(16) uint8_t s_wqs[2][kWR * kStr64];
    __shared__ alignas(16) uint8_t s_aqs[2][TR * kStr64];
    __shared__ alignas(8)  uint32_t s_wsc[2][kWR][2];
    __shared__ alignas(8)  uint32_t s_asc[2][TR][2];
    __shared__ float    s_row_g[TR];
    __shared__ int32_t  s_row[TR];
    __shared__ int32_t  s_tok[TR];
    __shared__ float    s_rw[TR];

    auto swiz64 = [](int row, int col16) -> int { return row * kStr64 + ((col16 ^ (row & 3)) << 4); };

    const int tid  = threadIdx.x;   // 0..511
    const int warp = tid >> 5;      // 0..15
    const int lane = tid & 31;
    const int wf   = warp >> 2;     // 0..3
    const int wn   = warp & 3;      // 0..3
    const int nb0  = 32 * wn;
    const int g    = lane >> 2;
    const int tig  = lane & 3;
    const int lm   = lane >> 3;
    const int l8   = lane & 7;
    const int tidx_a = g + (tig & 1) * 8;

    const int a_row_base = W_PER_WF * wf + 8 * (lm & 1) + l8;
    const int a_col_in   = lm >> 1;
    const int b_row0     = nb0 + 8 * (lm >> 1) + l8;
    const int b_row1     = b_row0 + 16;
    const int b_col_in   = lm & 1;

    const bool is_w_ld = (tid < 256);
    const int ur = tid >> 1;            // 0..127 for tid < 256
    const int uj = tid & 1;             // 0..1
    const int ar = (tid - 256) >> 1;    // 0..127 for tid >= 256
    const int ah = tid & 1;             // 0..1

    const int total_tiles = *tb.n_tiles;
    const int nwork = total_tiles * NFB;

    for (int w = blockIdx.x; w < nwork; w += gridDim.x) {
        const int2 tl = tb.tiles[w / NFB];
        const int fb = w % NFB;
        const int e = tl.x;
        const int row0 = tl.y;
        const InkExpertDev ed = experts[e];
        if ((GU ? ed.gu_code : ed.dn_code) != FMT_SF4) continue;
        const int nrows = min(TR, tb.off[e + 1] - row0);
        if (nrows <= 0) continue;

        const uint8_t* body = sec4_base + ed.body_off;
        float hdr[4];
        memcpy(hdr, body, 16);
        const float w_outer = GU ? (hdr[0] / hdr[1]) : (hdr[2] / hdr[3]);

        const uint8_t* base_p = GU ? (body + 16) : (body + ed.dn_off);
        const uint8_t* base_s = base_p + (size_t)ROWS_N * W_P_ROW;

        const uint8_t* wrow_p = nullptr;
        const uint8_t* wrow_s = nullptr;
        if (is_w_ld) {
            const int phys_row = GU ? (2 * (fb * 64 + 8 * (ur >> 4) + (ur & 7)) + ((ur >> 3) & 1))
                                    : (fb * kWR + ur);
            wrow_p = base_p + (size_t)phys_row * W_P_ROW;
            wrow_s = base_s + (size_t)phys_row * W_S_ROW;
        }

        __syncthreads();
        if (tid < TR) {
            if (tid < nrows) {
                const int tk = tb.tok_idx[row0 + tid];
                const int sr = GU ? tk : (row0 + tid);
                s_row[tid] = sr;
                if constexpr (GU) {
                    s_row_g[tid] = *(const float*)(act + (size_t)sr * ACT_ROW_B + 36);
                } else {
                    s_tok[tid] = tk;
                    s_rw[tid] = topk_w[tb.pairs[row0 + tid]];
                }
            } else {
                s_row[tid] = -1;
                if constexpr (GU) s_row_g[tid] = 0.0f;
                else { s_tok[tid] = 0; s_rw[tid] = 0.0f; }
                for (int b = 0; b < 2; ++b) {
                    s_asc[b][tid][0] = 0u; s_asc[b][tid][1] = 0u;
                    uint4* z = (uint4*)(s_aqs[b] + tid * kStr64);
                    z[0] = make_uint4(0u, 0u, 0u, 0u);
                    z[1] = make_uint4(0u, 0u, 0u, 0u);
                    z[2] = make_uint4(0u, 0u, 0u, 0u);
                    z[3] = make_uint4(0u, 0u, 0u, 0u);
                }
            }
        }
        __syncthreads();

        auto issue_stage = [&](int s2, int buf) {
            if (is_w_ld) {
                const size_t woff = (size_t)s2 * 64 + 16 * uj;
                cp16_async(s_wqs[buf] + swiz64(ur, uj),     wrow_p + woff);
                cp16_async(s_wqs[buf] + swiz64(ur, 2 + uj), wrow_p + woff + 32);
                if (uj == 0) {
                    cp8_async(&s_wsc[buf][ur][0], wrow_s + (size_t)s2 * 8);
                }
            } else if (ar < nrows) {
                const uint8_t* ap0 = act + (size_t)s_row[ar] * ACT_ROW_B + (size_t)(2 * s2) * kAB;
                const uint8_t* ap1 = ap0 + kAB;
                cp16_async(s_aqs[buf] + swiz64(ar, ah),     ap0 + 16 * ah);
                cp16_async(s_aqs[buf] + swiz64(ar, 2 + ah), ap1 + 16 * ah);
                if (ah == 0) {
                    cp4_async(&s_asc[buf][ar][0], ap0 + 32);
                    cp4_async(&s_asc[buf][ar][1], ap1 + 32);
                }
            }
            cp_commit();
        };

        if (is_w_ld && uj == 0) {
            pf_l2(wrow_p, (uint32_t)min((size_t)256, W_P_ROW));
            if constexpr (GU) pf_l2(wrow_s, (uint32_t)W_S_ROW);
        }
        issue_stage(0, 0);

        float acc[NI][4][4] = {};
        const bool on0 = nb0 < nrows;
        const bool on1 = nb0 + 8 < nrows;
        const bool on2 = nb0 + 16 < nrows;
        const bool on3 = nb0 + 24 < nrows;

        for (int s2 = 0; s2 < NS2; ++s2) {
            const int buf = s2 & 1;
            const int nxt = buf ^ 1;
            if (is_w_ld && uj == 0 && ((s2 & 3) == 0) && (size_t)(s2 + 4) * 64 < W_P_ROW) {
                const size_t nb = (size_t)(s2 + 4) * 64;
                pf_l2(wrow_p + nb, (uint32_t)min((size_t)256, W_P_ROW - nb));
            }
            if (s2 + 1 < NS2) {
                issue_stage(s2 + 1, nxt);
                cp_wait1();
            } else {
                cp_wait0();
            }
            __syncthreads();

            if (on0) {
#pragma unroll
                for (int sub = 0; sub < 2; ++sub) {
                    uint32_t bq0[4], bq1[4];
                    ldsm4(bq0, s_aqs[buf] + swiz64(b_row0, sub * 2 + b_col_in));
                    if (on2) ldsm4(bq1, s_aqs[buf] + swiz64(b_row1, sub * 2 + b_col_in));
                    const uint32_t sfb0 = s_asc[buf][nb0 + g][sub];
                    const uint32_t sfb1 = on1 ? s_asc[buf][nb0 + 8 + g][sub] : 0u;
                    const uint32_t sfb2 = on2 ? s_asc[buf][nb0 + 16 + g][sub] : 0u;
                    const uint32_t sfb3 = on3 ? s_asc[buf][nb0 + 24 + g][sub] : 0u;

#pragma unroll
                    for (int i = 0; i < NI; ++i) {
                        uint32_t a[4];
                        ldsm4(a, s_wqs[buf] + swiz64(a_row_base + 16 * i, sub * 2 + a_col_in));
                        const uint32_t sfa = s_wsc[buf][W_PER_WF * wf + 16 * i + tidx_a][sub];
                        mma_nvfp4_m16n8k64(acc[i][0], a, bq0[0], bq0[1], sfa, sfb0);
                        if (on1) mma_nvfp4_m16n8k64(acc[i][1], a, bq0[2], bq0[3], sfa, sfb1);
                        if (on2) mma_nvfp4_m16n8k64(acc[i][2], a, bq1[0], bq1[1], sfa, sfb2);
                        if (on3) mma_nvfp4_m16n8k64(acc[i][3], a, bq1[2], bq1[3], sfa, sfb3);
                    }
                }
            }
            if (s2 + 1 < NS2) __syncthreads();
        }

        if constexpr (GU) {
            if (on0) {
                constexpr size_t H_ROW_B = (size_t)(kInter / 64) * kAB;
                constexpr float inv_6g = 1.0f / (kGHaConst * 6.0f);
#pragma unroll
                for (int n = 0; n < 4; ++n) {
                    if ((n == 1 && !on1) || (n == 2 && !on2) || (n == 3 && !on3)) break;
#pragma unroll
                    for (int c = 0; c < 2; ++c) {
                        const int r = nb0 + 8 * n + 2 * tig + c;
                        const bool valid = (r < nrows);
                        const float sc_r = valid ? (w_outer * s_row_g[r]) : 0.0f;
                        const float h0 = situ_f32(acc[0][n][c] * sc_r, acc[0][n][2 + c] * sc_r);
                        const float h1 = situ_f32(acc[1][n][c] * sc_r, acc[1][n][2 + c] * sc_r);
                        float a0 = fmaxf(fabsf(h0), fabsf(h1));
#pragma unroll
                        for (int o = 4; o < 32; o <<= 1) a0 = fmaxf(a0, __shfl_xor_sync(0xFFFFFFFFu, a0, o));
                        float s0 = 0.0f;
                        const uint8_t d0 = encode_ue4m3(a0 * inv_6g, s0);
                        const float inv0 = (kGHaConst * s0) > 0.0f ? (1.0f / (kGHaConst * s0)) : 0.0f;
                        const float qa0 = h0 * inv0, qb0 = __shfl_xor_sync(0xFFFFFFFFu, qa0, 4);
                        const float qa1 = h1 * inv0, qb1 = __shfl_xor_sync(0xFFFFFFFFu, qa1, 4);
                        if (valid) {
                            uint8_t* dst_slot = out_ha + (size_t)(row0 + r) * H_ROW_B + (size_t)fb * kAB;
                            if ((g & 1) == 0) {
                                const int bo = 8 * wf + (g >> 1);
                                dst_slot[bo + 0] = pack_e2m1x2(qa0, qb0);
                                dst_slot[bo + 4] = pack_e2m1x2(qa1, qb1);
                            }
                            if (g == 0) dst_slot[32 + wf] = d0;
                        }
                    }
                }
            }
        } else if (on0) {
            const int rbase = fb * kWR + W_PER_WF * wf;
            const float wg = w_outer * kGHaConst;
#pragma unroll
            for (int n = 0; n < 4; ++n) {
                if ((n == 1 && !on1) || (n == 2 && !on2) || (n == 3 && !on3)) break;
#pragma unroll
                for (int c = 0; c < 2; ++c) {
                    const int r = nb0 + 8 * n + 2 * tig + c;
                    if (r < nrows) {
                        const float sc = wg * s_rw[r];
                        if (y_out_atomic) {
                            float* dst = y_out_atomic + (size_t)s_tok[r] * kHidden;
#pragma unroll
                            for (int i = 0; i < NI; ++i) {
                                atomicAdd(dst + rbase + 16 * i + g + 0, acc[i][n][c] * sc);
                                atomicAdd(dst + rbase + 16 * i + g + 8, acc[i][n][2 + c] * sc);
                            }
                        } else {
                            float* dst = y_pair + (size_t)tb.pairs[row0 + r] * kHidden;
#pragma unroll
                            for (int i = 0; i < NI; ++i) {
                                dst[rbase + 16 * i + g + 0] = acc[i][n][c] * sc;
                                dst[rbase + 16 * i + g + 8] = acc[i][n][2 + c] * sc;
                            }
                        }
                    }
                }
            }
        }
    }
#else
    (void)sec4_base; (void)experts; (void)tb; (void)act; (void)out_ha; (void)topk_w; (void)y_pair; (void)y_out_atomic;
#endif
}

// ---- RTX 5090 prefill SF4 expert GEMM, K256 stages (PREFILL lane, f-prefill) ----
// Profile of the shipped k_moe_sf4_tc_tr128: its cp.async streams alone (no MMA at all) take as long as the whole kernel (pf_ldbench: 1.18 ms for the gate+up call
// of one 4096-token chunk), because every warp-wide copy touches 16 different cache lines and moves 32 of the 128 bytes of each (K128 stages = 64 B of a weight row,
// 96 B of an activation row). This kernel keeps the tiles, the MMAs and the K order of k_moe_sf4_tc_tr128 (so every output is bit-identical to it) but
//  * a stage is K256: 128 B of every weight row (one cache line, 8 lanes), 192 B of every activation row (12 lanes, copied raw, scales read from the raw rows),
//    the scales of 16 weight blocks per row in one 16 B copy; the same bytes now need about a third of the line touches;
//  * ONE barrier per stage (two stage buffers: the loads of stage s+1 are issued right after the barrier that ends the reads of stage s-1);
//  * a persistent grid (one CTA per SM) and one continuous ring across work items: the first stage of the next item is loaded while the current one finishes,
//    the metadata of the next item (tile, expert, header, rows) is fetched into registers one item ahead;
//  * the weight scales of a stage are read as one 16 B word per row (the activation scales as 4 B words of the raw row);
//  * the zero-fill of the padding rows is gone (a padding row only feeds its own output column, which is never stored).
struct PfItemQ {
    bool ok;
    int row0, nrows, fb;
    float w_outer;
    const uint8_t* wp0;   // this thread's two weight rows (qs) in the stage copy
    const uint8_t* wp1;
    const uint8_t* wsp;   // this thread's weight scale source (tid < 128 for GU, tid < 256 for DN), else null
    int sr[3];            // this thread's three activation rows (source row index, -1 = none)
    int m_i, m_p;         // tid < 128: DN token index / pair index
    float m_f;            //            GU: the row's global activation scale; DN: the routing weight
};

template <bool GU>
__global__ void __launch_bounds__(512) k_moe_sf4_tc_tr128_q(
    const uint8_t* __restrict__ sec4_base,
    const InkExpertDev* __restrict__ experts,
    InkMoeTables tb,
    const uint8_t* __restrict__ act,
    uint8_t* __restrict__ out_ha,
    const float* __restrict__ topk_w,
    float* __restrict__ y_pair,
    float* __restrict__ y_out_atomic = nullptr) {
#if defined(INK_SM120_DEV)
    constexpr int TR       = 128;
    constexpr int ROWS_N   = GU ? (2 * kInter) : kHidden;
    constexpr int ROWS_K   = GU ? kHidden : kInter;
    constexpr int NS       = ROWS_K / 64;        // K64 blocks: 36 (GU) / 18 (DN)
    constexpr int NSTG     = (NS + 3) / 4;       // K256 stages: 9 / 5 (the last DN stage has two K64 blocks)
    constexpr int NFB      = ROWS_N / kWR;
    constexpr int W_PER_WF = 32;
    constexpr int NI       = 2;
    constexpr size_t W_P_ROW   = ROWS_K / 2;
    constexpr size_t W_S_ROW   = ROWS_K / 16;
    constexpr size_t ACT_ROW_B = (size_t)NS * kAB;
    constexpr int WB = 128 * 128, SB = 128 * 16, AS = 208, AB = TR * AS, STG = WB + SB + AB;   // bytes per stage buffer

    extern __shared__ __align__(128) uint8_t smem_raw[];
    int32_t* const s_m_i = reinterpret_cast<int32_t*>(smem_raw + 2 * STG);
    int32_t* const s_m_p = s_m_i + TR;
    float*   const s_m_f = reinterpret_cast<float*>(s_m_p + TR);

    const int tid  = threadIdx.x;
    const int warp = tid >> 5;
    const int lane = tid & 31;
    const int wf   = warp >> 2;
    const int wn   = warp & 3;
    const int nb0  = 32 * wn;
    const int g    = lane >> 2;
    const int tig  = lane & 3;
    const int lm   = lane >> 3;
    const int l8   = lane & 7;
    const int tidx_a = g + (tig & 1) * 8;

    // ldmatrix lane addresses (bytes inside a stage buffer)
    const int a_row0     = W_PER_WF * wf + 8 * (lm & 1) + l8;      // weight row of this lane for i = 0 (i = 1: +16); its (row & 7) == l8
    const int a_chunk    = lm >> 1;                                  // 0/1: first or second 16 B of the K64 block's 32 B
    const int b_row0     = nb0 + 8 * (lm >> 1) + l8;
    const int b_row1     = b_row0 + 16;
    const int b_col_in   = lm & 1;
    const int sfa_row0   = W_PER_WF * wf + tidx_a;                  // scale row of the A operand for i = 0 (i = 1: +16)

    const int total_tiles = *tb.n_tiles;
    const int nwork = total_tiles * NFB;
    int nw = blockIdx.x;

    auto fetch = [&]() -> PfItemQ {
        PfItemQ it;
        it.ok = false; it.row0 = 0; it.nrows = 0; it.fb = 0; it.w_outer = 0.0f;
        it.wp0 = nullptr; it.wp1 = nullptr; it.wsp = nullptr; it.sr[0] = it.sr[1] = it.sr[2] = -1; it.m_i = 0; it.m_p = 0; it.m_f = 0.0f;
        while (nw < nwork) {
            const int w = nw;
            nw += gridDim.x;
            const int2 tl = tb.tiles[w / NFB];
            const int fb = w % NFB;
            const int e = tl.x;
            const int row0 = tl.y;
            const InkExpertDev ed = experts[e];
            if ((GU ? ed.gu_code : ed.dn_code) != FMT_SF4) continue;
            const int nrows = min(TR, tb.off[e + 1] - row0);
            if (nrows <= 0) continue;
            const uint8_t* body = sec4_base + ed.body_off;
            float hdr[4];
            memcpy(hdr, body, 16);
            it.w_outer = GU ? (hdr[0] / hdr[1]) : (hdr[2] / hdr[3]);
            const uint8_t* base_p = GU ? (body + 16) : (body + ed.dn_off);
            const uint8_t* base_s = base_p + (size_t)ROWS_N * W_P_ROW;
            it.row0 = row0; it.nrows = nrows; it.fb = fb;
#pragma unroll
            for (int k = 0; k < 2; ++k) {
                const int row = (tid + 512 * k) >> 3;
                const int phys_row = GU ? (2 * (fb * 64 + 8 * (row >> 4) + (row & 7)) + ((row >> 3) & 1)) : (fb * kWR + row);
                (k == 0 ? it.wp0 : it.wp1) = base_p + (size_t)phys_row * W_P_ROW;
            }
            if (GU) {
                if (tid < 128) {
                    const int row = tid;
                    const int phys_row = 2 * (fb * 64 + 8 * (row >> 4) + (row & 7)) + ((row >> 3) & 1);
                    it.wsp = base_s + (size_t)phys_row * W_S_ROW;
                }
            } else if (tid < 256) {
                const int row = tid >> 1;
                it.wsp = base_s + (size_t)(fb * kWR + row) * W_S_ROW + (tid & 1) * 8;
            }
#pragma unroll
            for (int k = 0; k < 3; ++k) {
                const int row = (tid + 512 * k) / 12;
                if (row < nrows) it.sr[k] = GU ? tb.tok_idx[row0 + row] : (row0 + row);
            }
            if (tid < TR && tid < nrows) {
                const int tk = tb.tok_idx[row0 + tid];
                if constexpr (GU) {
                    it.m_f = *(const float*)(act + (size_t)tk * ACT_ROW_B + 36);
                } else {
                    it.m_i = tk;
                    const int pr = tb.pairs[row0 + tid];
                    it.m_p = pr;
                    it.m_f = topk_w[pr];
                }
            }
            it.ok = true;
            return it;
        }
        return it;
    };

    // loads of K256 stage s of item `it` into buffer b; one commit group per call (made by the caller)
    auto issue = [&](const PfItemQ& it, int s, int b) {
        uint8_t* base = smem_raw + (size_t)b * STG;
        const int nsub = min(4, NS - 4 * s);          // K64 blocks in this stage (4, or 2 for the last DN stage)
        const int nch = 2 * nsub;                      // 16 B weight chunks per row
#pragma unroll
        for (int k = 0; k < 2; ++k) {
            const int idx = tid + 512 * k;
            const int row = idx >> 3, ch = idx & 7;
            const uint8_t* p = (k == 0 ? it.wp0 : it.wp1);
            if (ch < nch) cp16_async(base + row * 128 + ((ch ^ (row & 7)) << 4), p + (size_t)s * 128 + ch * 16);
            if (ch == 0) {
                if (s == 0) pf_l2(p + 128, (uint32_t)min((size_t)256, W_P_ROW - 128));
                const size_t nb = (size_t)(s + 3) * 128;
                if (nb < W_P_ROW) pf_l2(p + nb, (uint32_t)min((size_t)128, W_P_ROW - nb));
            }
        }
        if (it.wsp != nullptr) {
            if constexpr (GU) cp16_async(base + WB + tid * 16, it.wsp + (size_t)s * 16);
            else if ((tid & 1) == 0 || nsub > 2) cp8_async(base + WB + (tid >> 1) * 16 + (tid & 1) * 8, it.wsp + (size_t)s * 16);
        }
#pragma unroll
        for (int k = 0; k < 3; ++k) {
            const int idx = tid + 512 * k;
            const int row = idx / 12, ch = idx - row * 12;
            if (it.sr[k] >= 0 && ch < 3 * nsub)
                cp16_async(base + WB + SB + row * AS + ch * 16, act + (size_t)it.sr[k] * ACT_ROW_B + (size_t)s * 192 + ch * 16);
        }
    };

    PfItemQ cur = fetch();
    if (!cur.ok) return;
    issue(cur, 0, 0);
    cp_commit();

    int gs = 0;
    while (cur.ok) {
        PfItemQ nxt = fetch();

        float acc[NI][4][4] = {};
        const int nrows = cur.nrows;
        const bool on0 = nb0 < nrows;
        const bool on1 = nb0 + 8 < nrows;
        const bool on2 = nb0 + 16 < nrows;
        const bool on3 = nb0 + 24 < nrows;

        for (int s = 0; s < NSTG; ++s, ++gs) {
            cp_wait0();
            __syncthreads();
            if (s == 0 && tid < TR) {
                if constexpr (GU) { s_m_f[tid] = cur.m_f; }
                else { s_m_i[tid] = cur.m_i; s_m_p[tid] = cur.m_p; s_m_f[tid] = cur.m_f; }
            }
            if (s + 1 < NSTG) issue(cur, s + 1, (gs + 1) & 1);
            else if (nxt.ok) issue(nxt, 0, (gs + 1) & 1);
            cp_commit();
            const uint8_t* sb = smem_raw + (size_t)(gs & 1) * STG;
            const uint8_t* wq = sb;
            const uint8_t* wsc = sb + WB;
            const uint8_t* aq = sb + WB + SB;
            const int nsub = min(4, NS - 4 * s);
            if (on0) {
                const uint4 sa0 = *reinterpret_cast<const uint4*>(wsc + (size_t)sfa_row0 * 16);
                const uint4 sa1 = *reinterpret_cast<const uint4*>(wsc + (size_t)(sfa_row0 + 16) * 16);
#pragma unroll
                for (int sub = 0; sub < 4; ++sub) {
                    if (sub < nsub) {
                        uint32_t bq0[4], bq1[4];
                        ldsm4(bq0, aq + b_row0 * AS + sub * 48 + 16 * b_col_in);
                        if (on2) ldsm4(bq1, aq + b_row1 * AS + sub * 48 + 16 * b_col_in);
                        const uint32_t sfb0 = *reinterpret_cast<const uint32_t*>(aq + (nb0 + g) * AS + sub * 48 + 32);
                        const uint32_t sfb1 = on1 ? *reinterpret_cast<const uint32_t*>(aq + (nb0 + 8 + g) * AS + sub * 48 + 32) : 0u;
                        const uint32_t sfb2 = on2 ? *reinterpret_cast<const uint32_t*>(aq + (nb0 + 16 + g) * AS + sub * 48 + 32) : 0u;
                        const uint32_t sfb3 = on3 ? *reinterpret_cast<const uint32_t*>(aq + (nb0 + 24 + g) * AS + sub * 48 + 32) : 0u;
#pragma unroll
                        for (int i = 0; i < NI; ++i) {
                            uint32_t a[4];
                            const int arow = a_row0 + 16 * i;
                            ldsm4(a, wq + arow * 128 + (((2 * sub + a_chunk) ^ l8) << 4));
                            const uint4 sv = (i == 0) ? sa0 : sa1;
                            const uint32_t sfa = (sub == 0) ? sv.x : (sub == 1) ? sv.y : (sub == 2) ? sv.z : sv.w;
                            mma_nvfp4_m16n8k64(acc[i][0], a, bq0[0], bq0[1], sfa, sfb0);
                            if (on1) mma_nvfp4_m16n8k64(acc[i][1], a, bq0[2], bq0[3], sfa, sfb1);
                            if (on2) mma_nvfp4_m16n8k64(acc[i][2], a, bq1[0], bq1[1], sfa, sfb2);
                            if (on3) mma_nvfp4_m16n8k64(acc[i][3], a, bq1[2], bq1[3], sfa, sfb3);
                        }
                    }
                }
            }
        }

        const int row0 = cur.row0;
        const int fb = cur.fb;
        const float w_outer = cur.w_outer;
        if constexpr (GU) {
            if (on0) {
                constexpr size_t H_ROW_B = (size_t)(kInter / 64) * kAB;
                constexpr float inv_6g = 1.0f / (kGHaConst * 6.0f);
#pragma unroll
                for (int n = 0; n < 4; ++n) {
                    if ((n == 1 && !on1) || (n == 2 && !on2) || (n == 3 && !on3)) break;
#pragma unroll
                    for (int c = 0; c < 2; ++c) {
                        const int r = nb0 + 8 * n + 2 * tig + c;
                        const bool valid = (r < nrows);
                        const float sc_r = valid ? (w_outer * s_m_f[r]) : 0.0f;
                        const float h0 = situ_f32(acc[0][n][c] * sc_r, acc[0][n][2 + c] * sc_r);
                        const float h1 = situ_f32(acc[1][n][c] * sc_r, acc[1][n][2 + c] * sc_r);
                        float a0 = fmaxf(fabsf(h0), fabsf(h1));
#pragma unroll
                        for (int o = 4; o < 32; o <<= 1) a0 = fmaxf(a0, __shfl_xor_sync(0xFFFFFFFFu, a0, o));
                        float s0 = 0.0f;
                        const uint8_t d0 = encode_ue4m3(a0 * inv_6g, s0);
                        const float inv0 = (kGHaConst * s0) > 0.0f ? (1.0f / (kGHaConst * s0)) : 0.0f;
                        const float qa0 = h0 * inv0, qb0 = __shfl_xor_sync(0xFFFFFFFFu, qa0, 4);
                        const float qa1 = h1 * inv0, qb1 = __shfl_xor_sync(0xFFFFFFFFu, qa1, 4);
                        if (valid) {
                            uint8_t* dst_slot = out_ha + (size_t)(row0 + r) * H_ROW_B + (size_t)fb * kAB;
                            if ((g & 1) == 0) {
                                const int bo = 8 * wf + (g >> 1);
                                dst_slot[bo + 0] = pack_e2m1x2(qa0, qb0);
                                dst_slot[bo + 4] = pack_e2m1x2(qa1, qb1);
                            }
                            if (g == 0) dst_slot[32 + wf] = d0;
                        }
                    }
                }
            }
        } else if (on0) {
            const int rbase = fb * kWR + W_PER_WF * wf;
            const float wg = w_outer * kGHaConst;
#pragma unroll
            for (int n = 0; n < 4; ++n) {
                if ((n == 1 && !on1) || (n == 2 && !on2) || (n == 3 && !on3)) break;
#pragma unroll
                for (int c = 0; c < 2; ++c) {
                    const int r = nb0 + 8 * n + 2 * tig + c;
                    if (r < nrows) {
                        const float sc = wg * s_m_f[r];
                        if (y_out_atomic) {
                            float* dst = y_out_atomic + (size_t)s_m_i[r] * kHidden;
#pragma unroll
                            for (int i = 0; i < NI; ++i) {
                                atomicAdd(dst + rbase + 16 * i + g + 0, acc[i][n][c] * sc);
                                atomicAdd(dst + rbase + 16 * i + g + 8, acc[i][n][2 + c] * sc);
                            }
                        } else {
                            float* dst = y_pair + (size_t)s_m_p[r] * kHidden;
#pragma unroll
                            for (int i = 0; i < NI; ++i) {
                                dst[rbase + 16 * i + g + 0] = acc[i][n][c] * sc;
                                dst[rbase + 16 * i + g + 8] = acc[i][n][2 + c] * sc;
                            }
                        }
                    }
                }
            }
        }
        cur = nxt;
    }
    cp_wait0();
#else
    (void)sec4_base; (void)experts; (void)tb; (void)act; (void)out_ha; (void)topk_w; (void)y_pair; (void)y_out_atomic;
#endif
}

// ---- Opt-in RTX 5090 Prefill TR=128 (kNT=512, 16 warps, BK=32) MX8 MoE Kernel ----
template <bool GU>
__global__ void __launch_bounds__(512) k_moe_mx8_tc_tr128(
    const uint8_t* __restrict__ sec4_base,
    const InkExpertDev* __restrict__ experts,
    InkMoeTables tb,
    const uint8_t* __restrict__ xq,
    const uint8_t* __restrict__ xsc,
    uint8_t* __restrict__ hq,
    uint8_t* __restrict__ hsc,
    const float* __restrict__ topk_w,
    float* __restrict__ y_pair,
    float* __restrict__ y_out_atomic = nullptr) {
#if defined(INK_SM120_DEV)
    constexpr int TR       = 128;
    constexpr int ROWS_N   = GU ? (2 * kInter) : kHidden;
    constexpr int ROWS_K   = GU ? kHidden : kInter;
    constexpr int NS       = ROWS_K / 32;  // 72 (GU) or 36 (DN)
    constexpr int NFB      = ROWS_N / kWR;
    constexpr int N_WF     = 4;
    constexpr int W_PER_WF = 32;
    constexpr int NI       = 2;

    __shared__ alignas(16) uint8_t s_wqs[2][kWR * kWStride4];
    __shared__ alignas(16) uint8_t s_aqs[2][TR * kAStride4];
    __shared__ uint8_t  s_wsc[2][kWR];
    __shared__ uint8_t  s_asc[2][TR];
    __shared__ int32_t  s_row[TR];
    __shared__ int32_t  s_tok[TR];
    __shared__ float    s_am[N_WF][TR];
    __shared__ float    s_rw[TR];

    const int tid  = threadIdx.x;
    const int warp = tid >> 5;
    const int lane = tid & 31;
    const int wf   = warp >> 2;
    const int wn   = warp & 3;
    const int nb0  = 32 * wn;
    const int g    = lane >> 2;
    const int tig  = lane & 3;
    const int lm   = lane >> 3;
    const int l8   = lane & 7;
    const int tidx_a = g + (tig & 1) * 8;

    const int a_off  = (W_PER_WF * wf + 8 * (lm & 1) + l8) * kWStride4 + 16 * (lm >> 1);
    const int b_off0 = (nb0 + 8 * (lm >> 1) + l8) * kAStride4 + 16 * (lm & 1);
    const int b_off1 = (nb0 + 16 + 8 * (lm >> 1) + l8) * kAStride4 + 16 * (lm & 1);

    const bool is_w_ld = (tid < 256);
    const int ur = tid >> 1;
    const int uj = tid & 1;
    const int ar = (tid - 256) >> 1;
    const int ah = tid & 1;

    const int total_tiles = *tb.n_tiles;
    const int nwork = total_tiles * NFB;

    for (int w = blockIdx.x; w < nwork; w += gridDim.x) {
        const int2 tl = tb.tiles[w / NFB];
        const int fb = w % NFB;
        const int e = tl.x;
        const int row0 = tl.y;
        const InkExpertDev ed = experts[e];
        if ((GU ? ed.gu_code : ed.dn_code) != FMT_MX8) continue;
        const int nrows = min(TR, tb.off[e + 1] - row0);
        if (nrows <= 0) continue;

        const uint8_t* body   = sec4_base + ed.body_off;
        const uint8_t* base_p = GU ? body : (body + ed.dn_off);
        const uint8_t* base_s = base_p + (size_t)ROWS_N * ROWS_K;

        const uint8_t* wrow_p = nullptr;
        const uint8_t* wrow_s = nullptr;
        if (is_w_ld) {
            const int phys_row = GU ? (2 * (fb * 64 + 8 * (ur >> 4) + (ur & 7)) + ((ur >> 3) & 1))
                                    : (fb * kWR + ur);
            wrow_p = base_p + (size_t)phys_row * ROWS_K;
            wrow_s = base_s + (size_t)phys_row * NS;
        }

        __syncthreads();
        if (tid < TR) {
            if (tid < nrows) {
                const int tk = tb.tok_idx[row0 + tid];
                s_row[tid] = GU ? tk : (row0 + tid);
                if constexpr (!GU) {
                    s_tok[tid] = tk;
                    s_rw[tid] = topk_w[tb.pairs[row0 + tid]];
                }
            } else {
                s_row[tid] = -1;
                if constexpr (!GU) { s_tok[tid] = 0; s_rw[tid] = 0.0f; }
                for (int b = 0; b < 2; ++b) {
                    s_asc[b][tid] = 0u;
                    uint4* z = (uint4*)(s_aqs[b] + tid * kAStride4);
                    z[0] = make_uint4(0u, 0u, 0u, 0u);
                    z[1] = make_uint4(0u, 0u, 0u, 0u);
                }
            }
        }
        __syncthreads();

        auto issue_stage = [&](int st, int buf) {
            if (is_w_ld) {
                cp16_async(s_wqs[buf] + ur * kWStride4 + 16 * uj, wrow_p + (size_t)st * 32 + 16 * uj);
                if (uj == 0) s_wsc[buf][ur] = wrow_s[st];
            } else if (ar < nrows) {
                const int sr = s_row[ar];
                cp16_async(s_aqs[buf] + ar * kAStride4 + 16 * ah, xq + (size_t)sr * ROWS_K + (size_t)st * 32 + 16 * ah);
                if (ah == 0) s_asc[buf][ar] = xsc[(size_t)sr * NS + st];
            }
            cp_commit();
        };

        issue_stage(0, 0);

        float acc[NI][4][4] = {};
        const bool on0 = nb0 < nrows;
        const bool on1 = nb0 + 8 < nrows;
        const bool on2 = nb0 + 16 < nrows;
        const bool on3 = nb0 + 24 < nrows;

        for (int st = 0; st < NS; ++st) {
            const int buf = st & 1;
            const int nxt = buf ^ 1;
            if (st + 1 < NS) {
                issue_stage(st + 1, nxt);
                cp_wait1();
            } else {
                cp_wait0();
            }
            __syncthreads();

            if (on0) {
                uint32_t bq0[4], bq1[4];
                ldsm4(bq0, s_aqs[buf] + b_off0);
                if (on2) ldsm4(bq1, s_aqs[buf] + b_off1);
                const uint32_t sfb0 = (uint32_t)s_asc[buf][nb0 + g];
                const uint32_t sfb1 = on1 ? (uint32_t)s_asc[buf][nb0 + 8 + g] : 0u;
                const uint32_t sfb2 = on2 ? (uint32_t)s_asc[buf][nb0 + 16 + g] : 0u;
                const uint32_t sfb3 = on3 ? (uint32_t)s_asc[buf][nb0 + 24 + g] : 0u;
#pragma unroll
                for (int i = 0; i < NI; ++i) {
                    uint32_t a[4];
                    ldsm4(a, s_wqs[buf] + a_off + 16 * i * kWStride4);
                    const uint32_t sfa = (uint32_t)s_wsc[buf][W_PER_WF * wf + 16 * i + tidx_a];
                    mma_mxfp8_m16n8k32(acc[i][0], a, bq0[0], bq0[1], sfa, sfb0);
                    if (on1) mma_mxfp8_m16n8k32(acc[i][1], a, bq0[2], bq0[3], sfa, sfb1);
                    if (on2) mma_mxfp8_m16n8k32(acc[i][2], a, bq1[0], bq1[1], sfa, sfb2);
                    if (on3) mma_mxfp8_m16n8k32(acc[i][3], a, bq1[2], bq1[3], sfa, sfb3);
                }
            }
            if (st + 1 < NS) __syncthreads();
        }

        if constexpr (GU) {
            float h_val[4][2][NI] = {};
            if (on0) {
#pragma unroll
                for (int n = 0; n < 4; ++n) {
                    if ((n == 1 && !on1) || (n == 2 && !on2) || (n == 3 && !on3)) break;
#pragma unroll
                    for (int c = 0; c < 2; ++c) {
#pragma unroll
                        for (int i = 0; i < NI; ++i) {
                            h_val[n][c][i] = situ_f32(acc[i][n][c], acc[i][n][2 + c]);
                        }
                        float a0 = fmaxf(fabsf(h_val[n][c][0]), fabsf(h_val[n][c][1]));
#pragma unroll
                        for (int o = 4; o < 32; o <<= 1) a0 = fmaxf(a0, __shfl_xor_sync(0xFFFFFFFFu, a0, o));
                        const int r = nb0 + 8 * n + 2 * tig + c;
                        if (g == 0 && r < TR) s_am[wf][r] = a0;
                    }
                }
            }
            __syncthreads();
            if (on0) {
                constexpr int NS_H = kInter / 32;
#pragma unroll
                for (int n = 0; n < 4; ++n) {
                    if ((n == 1 && !on1) || (n == 2 && !on2) || (n == 3 && !on3)) break;
#pragma unroll
                    for (int c = 0; c < 2; ++c) {
                        const int r = nb0 + 8 * n + 2 * tig + c;
                        if (r >= nrows) continue;
                        const float am_32 = fmaxf(s_am[wf & ~1][r], s_am[wf | 1][r]);
                        float inv_sc = 0.0f;
                        const uint8_t eb = encode_ue8m0(am_32, inv_sc);
                        uint8_t* dst_q = hq + (size_t)(row0 + r) * kInter + (size_t)fb * 64 + 16 * wf;
                        dst_q[g]     = encode_e4m3_signed(h_val[n][c][0] * inv_sc);
                        dst_q[8 + g] = encode_e4m3_signed(h_val[n][c][1] * inv_sc);
                        if (g == 0 && (wf & 1) == 0) {
                            hsc[(size_t)(row0 + r) * NS_H + 2 * fb + (wf >> 1)] = eb;
                        }
                    }
                }
            }
        } else if (on0) {
            const int rbase = fb * kWR + W_PER_WF * wf;
#pragma unroll
            for (int n = 0; n < 4; ++n) {
                if ((n == 1 && !on1) || (n == 2 && !on2) || (n == 3 && !on3)) break;
#pragma unroll
                for (int c = 0; c < 2; ++c) {
                    const int r = nb0 + 8 * n + 2 * tig + c;
                    if (r < nrows) {
                        const float sc = s_rw[r];
                        if (y_out_atomic) {
                            float* dst = y_out_atomic + (size_t)s_tok[r] * kHidden;
#pragma unroll
                            for (int i = 0; i < NI; ++i) {
                                atomicAdd(dst + rbase + 16 * i + g + 0, acc[i][n][c] * sc);
                                atomicAdd(dst + rbase + 16 * i + g + 8, acc[i][n][2 + c] * sc);
                            }
                        } else {
                            float* dst = y_pair + (size_t)tb.pairs[row0 + r] * kHidden;
#pragma unroll
                            for (int i = 0; i < NI; ++i) {
                                dst[rbase + 16 * i + g + 0] = acc[i][n][c] * sc;
                                dst[rbase + 16 * i + g + 8] = acc[i][n][2 + c] * sc;
                            }
                        }
                    }
                }
            }
        }
    }
#else
    (void)sec4_base; (void)experts; (void)tb; (void)xq; (void)xsc; (void)hq; (void)hsc; (void)topk_w; (void)y_pair; (void)y_out_atomic;
#endif
}

// ---- Opt-in RTX 5090 Expert-Looped L2-Resident Prefill SF4 MoE Kernel (M >= 512) ----
// Grid: dim3(NFB=18, 128 experts), Block: kNT=256, TR=64, BK=128.
// Loops over all 64-row chunks of expert e inside ONE CTA so the 144 KB weight slice stays 100% HOT IN L2
// (reads every expert from GDDR7 ONCE instead of 2.5-4x, AND all 18 feature blocks share activations in L2!).
template <bool GU>
__global__ void __launch_bounds__(kNT) k_moe_sf4_tc_e_loop(
    const uint8_t* __restrict__ sec4_base,
    const InkExpertDev* __restrict__ experts,
    InkMoeTables tb,
    const uint8_t* __restrict__ act,
    uint8_t* __restrict__ out_ha,
    const float* __restrict__ topk_w,
    float* __restrict__ y_pair) {
#if defined(INK_SM120_DEV)
    constexpr int TR       = 64;
    constexpr int ROWS_N   = GU ? (2 * kInter) : kHidden;
    constexpr int ROWS_K   = GU ? kHidden : kInter;
    constexpr int NS       = ROWS_K / 64;
    constexpr int NS2      = NS / 2;
    constexpr int N_WF     = 2;
    constexpr int W_PER_WF = 64;
    constexpr int NI       = 4;
    constexpr int kStr80   = 80;
    constexpr size_t W_P_ROW   = ROWS_K / 2;
    constexpr size_t W_S_ROW   = ROWS_K / 16;
    constexpr size_t ACT_ROW_B = (size_t)NS * kAB;

    const int fb = blockIdx.x;
    const int e  = blockIdx.y;
    const InkExpertDev ed = experts[e];
    if ((GU ? ed.gu_code : ed.dn_code) != FMT_SF4) return;
    const int r_beg = tb.off[e];
    const int r_end = tb.off[e + 1];
    if (r_beg >= r_end) return;

    __shared__ alignas(16) uint8_t s_wqs[2][kWR * kStr80];
    __shared__ alignas(16) uint8_t s_aqs[2][TR * kStr80];
    __shared__ alignas(8)  uint32_t s_wsc[2][kWR][2];
    __shared__ alignas(8)  uint32_t s_asc[2][TR][2];
    __shared__ float    s_agb[2][TR][2];
    __shared__ int32_t  s_row[TR];
    __shared__ float    s_am[N_WF][TR];
    __shared__ float    s_rw[TR];

    const int tid  = threadIdx.x;
    const int warp = tid >> 5;
    const int lane = tid & 31;
    const int wf   = warp >> 2;
    const int wn   = warp & 3;
    const int nb0  = 16 * wn;
    const int g    = lane >> 2;
    const int tig  = lane & 3;
    const int lm   = lane >> 3;
    const int l8   = lane & 7;
    const int tidx_a = g + (tig & 1) * 8;

    const int a_off = (W_PER_WF * wf + 8 * (lm & 1) + l8) * kStr80 + 16 * (lm >> 1);
    const int b_off = (nb0 + 8 * (lm >> 1) + l8) * kStr80 + 16 * (lm & 1);
    const int ur = tid >> 1;
    const int uj = tid & 1;

    const uint8_t* body = sec4_base + ed.body_off;
    float hdr[4];
    memcpy(hdr, body, 16);
    const float w_outer = GU ? (hdr[0] / hdr[1]) : (hdr[2] / hdr[3]);
    const uint8_t* base_p = GU ? (body + 16) : (body + ed.dn_off);
    const uint8_t* base_s = base_p + (size_t)ROWS_N * W_P_ROW;
    const int phys_row = GU ? (2 * (fb * 64 + 8 * (ur >> 4) + (ur & 7)) + ((ur >> 3) & 1))
                            : (fb * kWR + ur);
    const uint8_t* wrow_p = base_p + (size_t)phys_row * W_P_ROW;
    const uint8_t* wrow_s = base_s + (size_t)phys_row * W_S_ROW;

    for (int row0 = r_beg; row0 < r_end; row0 += TR) {
        const int nrows = min(TR, r_end - row0);
        if (row0 > r_beg) __syncthreads();
        if (tid < TR) {
            if (tid < nrows) {
                s_row[tid] = GU ? tb.tok_idx[row0 + tid] : (row0 + tid);
                if constexpr (!GU) s_rw[tid] = topk_w[tb.pairs[row0 + tid]];
            } else {
                s_row[tid] = -1;
                if constexpr (!GU) s_rw[tid] = 0.0f;
                for (int b = 0; b < 2; ++b) {
                    s_asc[b][tid][0] = 0u; s_asc[b][tid][1] = 0u;
                    s_agb[b][tid][0] = 0.0f; s_agb[b][tid][1] = 0.0f;
                    uint4* z = (uint4*)(s_aqs[b] + tid * kStr80);
                    z[0] = make_uint4(0u, 0u, 0u, 0u);
                    z[1] = make_uint4(0u, 0u, 0u, 0u);
                    z[2] = make_uint4(0u, 0u, 0u, 0u);
                    z[3] = make_uint4(0u, 0u, 0u, 0u);
                }
            }
        }
        __syncthreads();

        auto issue_stage = [&](int s2, int buf) {
            const size_t woff = (size_t)s2 * 64 + 16 * uj;
            cp16_async(s_wqs[buf] + ur * kStr80 + 16 * uj,      wrow_p + woff);
            cp16_async(s_wqs[buf] + ur * kStr80 + 32 + 16 * uj, wrow_p + woff + 32);
            if (uj == 0) {
                const uint2 ws2 = *(const uint2*)(wrow_s + (size_t)s2 * 8);
                s_wsc[buf][ur][0] = ws2.x;
                s_wsc[buf][ur][1] = ws2.y;
            }
            if (tid < 2 * TR) {
                const int r = tid >> 1;
                const int h = tid & 1;
                if (r < nrows) {
                    const uint8_t* ap0 = act + (size_t)s_row[r] * ACT_ROW_B + (size_t)(2 * s2) * kAB;
                    const uint8_t* ap1 = ap0 + kAB;
                    cp16_async(s_aqs[buf] + r * kStr80 + 16 * h,      ap0 + 16 * h);
                    cp16_async(s_aqs[buf] + r * kStr80 + 32 + 16 * h, ap1 + 16 * h);
                }
            } else if (tid < 3 * TR) {
                const int r = tid - 2 * TR;
                if (r < nrows) {
                    const uint8_t* ap0 = act + (size_t)s_row[r] * ACT_ROW_B + (size_t)(2 * s2) * kAB;
                    const uint2 m0 = *(const uint2*)(ap0 + 32);
                    const uint2 m1 = *(const uint2*)(ap0 + kAB + 32);
                    s_asc[buf][r][0] = m0.x; s_agb[buf][r][0] = __uint_as_float(m0.y);
                    s_asc[buf][r][1] = m1.x; s_agb[buf][r][1] = __uint_as_float(m1.y);
                }
            }
            cp_commit();
        };

        issue_stage(0, 0);
        float acc[NI][2][4] = {};
        const bool on0 = nb0 < nrows;
        const bool on1 = nb0 + 8 < nrows;

        for (int s2 = 0; s2 < NS2; ++s2) {
            const int buf = s2 & 1;
            const int nxt = buf ^ 1;
            if (s2 + 1 < NS2) { issue_stage(s2 + 1, nxt); cp_wait1(); } else { cp_wait0(); }
            __syncthreads();
            if (on0) {
#pragma unroll
                for (int sub = 0; sub < 2; ++sub) {
                    uint32_t bq[4];
                    ldsm4(bq, s_aqs[buf] + b_off + sub * 32);
                    const uint32_t sfb0 = s_asc[buf][nb0 + g][sub];
                    const uint32_t sfb1 = on1 ? s_asc[buf][nb0 + 8 + g][sub] : 0u;
                    const float gb00 = s_agb[buf][nb0 + 2 * tig + 0][sub];
                    const float gb01 = s_agb[buf][nb0 + 2 * tig + 1][sub];
                    const float gb10 = on1 ? s_agb[buf][nb0 + 8 + 2 * tig + 0][sub] : 0.0f;
                    const float gb11 = on1 ? s_agb[buf][nb0 + 8 + 2 * tig + 1][sub] : 0.0f;
#pragma unroll
                    for (int i = 0; i < NI; ++i) {
                        uint32_t a[4];
                        ldsm4(a, s_wqs[buf] + a_off + 16 * i * kStr80 + sub * 32);
                        const uint32_t sfa = s_wsc[buf][W_PER_WF * wf + 16 * i + tidx_a][sub];
                        float d0[4] = {}, d1[4] = {};
                        mma_nvfp4_m16n8k64(d0, a, bq[0], bq[1], sfa, sfb0);
                        if (on1) mma_nvfp4_m16n8k64(d1, a, bq[2], bq[3], sfa, sfb1);
                        acc[i][0][0] = fmaf(d0[0], gb00, acc[i][0][0]);
                        acc[i][0][1] = fmaf(d0[1], gb01, acc[i][0][1]);
                        acc[i][0][2] = fmaf(d0[2], gb00, acc[i][0][2]);
                        acc[i][0][3] = fmaf(d0[3], gb01, acc[i][0][3]);
                        if (on1) {
                            acc[i][1][0] = fmaf(d1[0], gb10, acc[i][1][0]);
                            acc[i][1][1] = fmaf(d1[1], gb11, acc[i][1][1]);
                            acc[i][1][2] = fmaf(d1[2], gb10, acc[i][1][2]);
                            acc[i][1][3] = fmaf(d1[3], gb11, acc[i][1][3]);
                        }
                    }
                }
            }
            if (s2 + 1 < NS2) __syncthreads();
        }

        if constexpr (GU) {
            float h_val[2][2][NI] = {};
            float am_sub0[2][2] = {}, am_sub1[2][2] = {};
            if (on0) {
#pragma unroll
                for (int n = 0; n < 2; ++n) {
                    if (n == 1 && !on1) break;
#pragma unroll
                    for (int c = 0; c < 2; ++c) {
#pragma unroll
                        for (int i = 0; i < NI; ++i) {
                            h_val[n][c][i] = situ_f32(acc[i][n][c] * w_outer, acc[i][n][2 + c] * w_outer);
                        }
                        float a0 = fmaxf(fabsf(h_val[n][c][0]), fabsf(h_val[n][c][1]));
                        float a1 = fmaxf(fabsf(h_val[n][c][2]), fabsf(h_val[n][c][3]));
#pragma unroll
                        for (int o = 4; o < 32; o <<= 1) {
                            a0 = fmaxf(a0, __shfl_xor_sync(0xFFFFFFFFu, a0, o));
                            a1 = fmaxf(a1, __shfl_xor_sync(0xFFFFFFFFu, a1, o));
                        }
                        am_sub0[n][c] = a0; am_sub1[n][c] = a1;
                        const int r = nb0 + 8 * n + 2 * tig + c;
                        if (g == 0 && r < TR) s_am[wf][r] = fmaxf(a0, a1);
                    }
                }
            }
            __syncthreads();
            if (on0) {
                constexpr size_t H_ROW_B = (size_t)(kInter / 64) * kAB;
#pragma unroll
                for (int n = 0; n < 2; ++n) {
                    if (n == 1 && !on1) break;
#pragma unroll
                    for (int c = 0; c < 2; ++c) {
                        const int r = nb0 + 8 * n + 2 * tig + c;
                        const bool valid = (r < nrows);
                        const float am_64 = valid ? fmaxf(s_am[0][r], s_am[1][r]) : 0.0f;
                        const float g_blk = am_64 * (1.0f / (6.0f * 448.0f));
                        float s0 = 0.0f, s1 = 0.0f;
                        const uint8_t d0 = encode_ue4m3(g_blk > 0.0f ? (am_sub0[n][c] / (g_blk * 6.0f)) : 0.0f, s0);
                        const uint8_t d1 = encode_ue4m3(g_blk > 0.0f ? (am_sub1[n][c] / (g_blk * 6.0f)) : 0.0f, s1);
                        const float inv0 = (g_blk * s0) > 0.0f ? (1.0f / (g_blk * s0)) : 0.0f;
                        const float inv1 = (g_blk * s1) > 0.0f ? (1.0f / (g_blk * s1)) : 0.0f;
                        float qa[NI], qb[NI];
#pragma unroll
                        for (int i = 0; i < NI; ++i) {
                            qa[i] = h_val[n][c][i] * ((i < 2) ? inv0 : inv1);
                            qb[i] = __shfl_xor_sync(0xFFFFFFFFu, qa[i], 4);
                        }
                        if (valid) {
                            uint8_t* dst_slot = out_ha + (size_t)(row0 + r) * H_ROW_B + (size_t)fb * kAB;
                            if ((g & 1) == 0) {
                                const int bo = 16 * wf + (g >> 1);
#pragma unroll
                                for (int i = 0; i < NI; ++i) dst_slot[bo + 4 * i] = pack_e2m1x2(qa[i], qb[i]);
                            }
                            if (g == 0) {
                                dst_slot[32 + 2 * wf + 0] = d0;
                                dst_slot[32 + 2 * wf + 1] = d1;
                                if (wf == 0) *(float*)(dst_slot + 36) = g_blk;
                            }
                        }
                    }
                }
            }
        } else if (on0) {
            const int rbase = fb * kWR + W_PER_WF * wf;
#pragma unroll
            for (int n = 0; n < 2; ++n) {
                if (n == 1 && !on1) break;
#pragma unroll
                for (int c = 0; c < 2; ++c) {
                    const int r = nb0 + 8 * n + 2 * tig + c;
                    if (r < nrows) {
                        const float sc = w_outer * s_rw[r];
                        float* dst = y_pair + (size_t)tb.pairs[row0 + r] * kHidden;
#pragma unroll
                        for (int i = 0; i < NI; ++i) {
                            dst[rbase + 16 * i + g + 0] = acc[i][n][c] * sc;
                            dst[rbase + 16 * i + g + 8] = acc[i][n][2 + c] * sc;
                        }
                    }
                }
            }
        }
    }
#else
    (void)sec4_base; (void)experts; (void)tb; (void)act; (void)out_ha; (void)topk_w; (void)y_pair;
#endif
}

template <bool GU>
__global__ void __launch_bounds__(kNT) k_moe_mx8_tc_e_loop(
    const uint8_t* __restrict__ sec4_base,
    const InkExpertDev* __restrict__ experts,
    InkMoeTables tb,
    const uint8_t* __restrict__ xq,
    const uint8_t* __restrict__ xsc,
    uint8_t* __restrict__ hq,
    uint8_t* __restrict__ hsc,
    const float* __restrict__ topk_w,
    float* __restrict__ y_pair) {
#if defined(INK_SM120_DEV)
    constexpr int TR       = 64;
    constexpr int ROWS_N   = GU ? (2 * kInter) : kHidden;
    constexpr int ROWS_K   = GU ? kHidden : kInter;
    constexpr int NS       = ROWS_K / 32;
    constexpr int NS2      = NS / 2;
    constexpr int W_PER_WF = 64;
    constexpr int NI       = 4;
    constexpr int kStr80   = 80;

    const int fb = blockIdx.x;
    const int e  = blockIdx.y;
    const InkExpertDev ed = experts[e];
    if ((GU ? ed.gu_code : ed.dn_code) != FMT_MX8) return;
    const int r_beg = tb.off[e];
    const int r_end = tb.off[e + 1];
    if (r_beg >= r_end) return;

    __shared__ alignas(16) uint8_t s_wqs[2][kWR * kStr80];
    __shared__ alignas(16) uint8_t s_aqs[2][TR * kStr80];
    __shared__ uint8_t  s_wsc[2][kWR][2];
    __shared__ uint8_t  s_asc[2][TR][2];
    __shared__ int32_t  s_row[TR];
    __shared__ float    s_rw[TR];

    const int tid  = threadIdx.x;
    const int warp = tid >> 5;
    const int lane = tid & 31;
    const int wf   = warp >> 2;
    const int wn   = warp & 3;
    const int nb0  = 16 * wn;
    const int g    = lane >> 2;
    const int tig  = lane & 3;
    const int lm   = lane >> 3;
    const int l8   = lane & 7;
    const int tidx_a = g + (tig & 1) * 8;

    const int a_off = (W_PER_WF * wf + 8 * (lm & 1) + l8) * kStr80 + 16 * (lm >> 1);
    const int b_off = (nb0 + 8 * (lm >> 1) + l8) * kStr80 + 16 * (lm & 1);
    const int ur = tid >> 1;
    const int uj = tid & 1;

    const uint8_t* body = sec4_base + ed.body_off;
    const uint8_t* base_p = GU ? body : (body + ed.dn_off);
    const uint8_t* base_s = base_p + (size_t)ROWS_N * ROWS_K;
    const int phys_row = GU ? (2 * (fb * 64 + 8 * (ur >> 4) + (ur & 7)) + ((ur >> 3) & 1))
                            : (fb * kWR + ur);
    const uint8_t* wrow_p = base_p + (size_t)phys_row * ROWS_K;
    const uint8_t* wrow_s = base_s + (size_t)phys_row * NS;

    for (int row0 = r_beg; row0 < r_end; row0 += TR) {
        const int nrows = min(TR, r_end - row0);
        if (row0 > r_beg) __syncthreads();
        if (tid < TR) {
            if (tid < nrows) {
                s_row[tid] = GU ? tb.tok_idx[row0 + tid] : (row0 + tid);
                if constexpr (!GU) s_rw[tid] = topk_w[tb.pairs[row0 + tid]];
            } else {
                s_row[tid] = -1;
                if constexpr (!GU) s_rw[tid] = 0.0f;
                for (int b = 0; b < 2; ++b) {
                    s_asc[b][tid][0] = 0u; s_asc[b][tid][1] = 0u;
                    uint4* z = (uint4*)(s_aqs[b] + tid * kStr80);
                    z[0] = make_uint4(0u, 0u, 0u, 0u);
                    z[1] = make_uint4(0u, 0u, 0u, 0u);
                    z[2] = make_uint4(0u, 0u, 0u, 0u);
                    z[3] = make_uint4(0u, 0u, 0u, 0u);
                }
            }
        }
        __syncthreads();

        auto issue_stage = [&](int s2, int buf) {
            const size_t woff = (size_t)s2 * 64 + 16 * uj;
            cp16_async(s_wqs[buf] + ur * kStr80 + 16 * uj,      wrow_p + woff);
            cp16_async(s_wqs[buf] + ur * kStr80 + 32 + 16 * uj, wrow_p + woff + 32);
            if (uj == 0) {
                s_wsc[buf][ur][0] = wrow_s[2 * s2 + 0];
                s_wsc[buf][ur][1] = wrow_s[2 * s2 + 1];
            }
            if (tid < 2 * TR) {
                const int r = tid >> 1;
                const int h = tid & 1;
                if (r < nrows) {
                    const uint8_t* ap = xq + (size_t)s_row[r] * ROWS_K + (size_t)s2 * 64 + 16 * h;
                    cp16_async(s_aqs[buf] + r * kStr80 + 16 * h,      ap);
                    cp16_async(s_aqs[buf] + r * kStr80 + 32 + 16 * h, ap + 32);
                }
            } else if (tid < 3 * TR) {
                const int r = tid - 2 * TR;
                if (r < nrows) {
                    s_asc[buf][r][0] = xsc[(size_t)s_row[r] * NS + 2 * s2 + 0];
                    s_asc[buf][r][1] = xsc[(size_t)s_row[r] * NS + 2 * s2 + 1];
                }
            }
            cp_commit();
        };

        issue_stage(0, 0);
        float acc[NI][2][4] = {};
        const bool on0 = nb0 < nrows;
        const bool on1 = nb0 + 8 < nrows;

        for (int s2 = 0; s2 < NS2; ++s2) {
            const int buf = s2 & 1;
            const int nxt = buf ^ 1;
            if (s2 + 1 < NS2) { issue_stage(s2 + 1, nxt); cp_wait1(); } else { cp_wait0(); }
            __syncthreads();
            if (on0) {
#pragma unroll
                for (int sub = 0; sub < 2; ++sub) {
                    uint32_t bq[4];
                    ldsm4(bq, s_aqs[buf] + b_off + sub * 32);
                    const uint32_t sfb0 = (uint32_t)s_asc[buf][nb0 + g][sub];
                    const uint32_t sfb1 = on1 ? (uint32_t)s_asc[buf][nb0 + 8 + g][sub] : 0u;
#pragma unroll
                    for (int i = 0; i < NI; ++i) {
                        uint32_t a[4];
                        ldsm4(a, s_wqs[buf] + a_off + 16 * i * kStr80 + sub * 32);
                        const uint32_t sfa = (uint32_t)s_wsc[buf][W_PER_WF * wf + 16 * i + tidx_a][sub];
                        mma_mxfp8_m16n8k32(acc[i][0], a, bq[0], bq[1], sfa, sfb0);
                        if (on1) mma_mxfp8_m16n8k32(acc[i][1], a, bq[2], bq[3], sfa, sfb1);
                    }
                }
            }
            if (s2 + 1 < NS2) __syncthreads();
        }

        if constexpr (GU) {
            if (on0) {
                const int blk32 = 2 * fb + wf;
#pragma unroll
                for (int n = 0; n < 2; ++n) {
                    if (n == 1 && !on1) break;
#pragma unroll
                    for (int c = 0; c < 2; ++c) {
                        const int r = nb0 + 8 * n + 2 * tig + c;
                        float hv[NI];
                        float am32 = 0.0f;
#pragma unroll
                        for (int i = 0; i < NI; ++i) {
                            hv[i] = situ_f32(acc[i][n][c], acc[i][n][2 + c]);
                            am32 = fmaxf(am32, fabsf(hv[i]));
                        }
#pragma unroll
                        for (int o = 4; o < 32; o <<= 1) am32 = fmaxf(am32, __shfl_xor_sync(0xFFFFFFFFu, am32, o));
                        if (r < nrows) {
                            float inv_sc = 0.0f;
                            const uint8_t sc_b = encode_ue8m0(am32, inv_sc);
                            uint8_t* dst_q = hq + (size_t)(row0 + r) * kInter + blk32 * 32;
#pragma unroll
                            for (int i = 0; i < NI; ++i) dst_q[8 * i + g] = encode_e4m3_signed(hv[i] * inv_sc);
                            if (g == 0) hsc[(size_t)(row0 + r) * (kInter / 32) + blk32] = sc_b;
                        }
                    }
                }
            }
        } else if (on0) {
            const int rbase = fb * kWR + W_PER_WF * wf;
#pragma unroll
            for (int n = 0; n < 2; ++n) {
                if (n == 1 && !on1) break;
#pragma unroll
                for (int c = 0; c < 2; ++c) {
                    const int r = nb0 + 8 * n + 2 * tig + c;
                    if (r < nrows) {
                        const float sc = s_rw[r];
                        float* dst = y_pair + (size_t)tb.pairs[row0 + r] * kHidden;
#pragma unroll
                        for (int i = 0; i < NI; ++i) {
                            dst[rbase + 16 * i + g + 0] = acc[i][n][c] * sc;
                            dst[rbase + 16 * i + g + 8] = acc[i][n][2 + c] * sc;
                        }
                    }
                }
            }
        }
    }
#else
    (void)sec4_base; (void)experts; (void)tb; (void)xq; (void)xsc; (void)hq; (void)hsc; (void)topk_w; (void)y_pair;
#endif
}

void moe_sf4_gu_tc_eloop(const uint8_t* sec4_base, const InkExpertDev* experts, InkMoeTables tb,
                         const uint8_t* xa, uint8_t* ha, cudaStream_t stream) {
    dim3 grid(2 * kInter / kWR, 128);
    k_moe_sf4_tc_e_loop<true><<<grid, kNT, 0, stream>>>(sec4_base, experts, tb, xa, ha, nullptr, nullptr);
}

void moe_sf4_dn_tc_eloop(const uint8_t* sec4_base, const InkExpertDev* experts, InkMoeTables tb,
                         const uint8_t* ha, const float* topk_w, float* y_pair, cudaStream_t stream) {
    dim3 grid(kHidden / kWR, 128);
    k_moe_sf4_tc_e_loop<false><<<grid, kNT, 0, stream>>>(sec4_base, experts, tb, ha, nullptr, topk_w, y_pair);
}

void moe_mx8_gu_tc_eloop(const uint8_t* sec4_base, const InkExpertDev* experts, InkMoeTables tb,
                         const uint8_t* xq, const uint8_t* xsc, uint8_t* hq, uint8_t* hsc, cudaStream_t stream) {
    dim3 grid(2 * kInter / kWR, 128);
    k_moe_mx8_tc_e_loop<true><<<grid, kNT, 0, stream>>>(sec4_base, experts, tb, xq, xsc, hq, hsc, nullptr, nullptr);
}

void moe_mx8_dn_tc_eloop(const uint8_t* sec4_base, const InkExpertDev* experts, InkMoeTables tb,
                         const uint8_t* hq, const uint8_t* hsc, const float* topk_w, float* y_pair, cudaStream_t stream) {
    dim3 grid(kHidden / kWR, 128);
    k_moe_mx8_tc_e_loop<false><<<grid, kNT, 0, stream>>>(sec4_base, experts, tb, hq, hsc, nullptr, nullptr, topk_w, y_pair);
}

// The K256 TR=128 SF4 expert kernel (k_moe_sf4_tc_tr128_q, bit-identical to k_moe_sf4_tc_tr128): STRATA_INK_SM120_PFQ=0 runs the shipped kernel (default: on).
int& pfq_override() {
    static int v = -1;
    return v;
}

static bool pfq_on() {
    static const bool env_on = [] {
        const char* e = std::getenv("STRATA_INK_SM120_PFQ");
        return !(e && e[0] == '0');
    }();
    const int ov = pfq_override();
    return ov >= 0 ? ov > 0 : env_on;
}

template <bool GU>
static void launch_tr128_q(const uint8_t* sec4_base, const InkExpertDev* experts, InkMoeTables tb, const uint8_t* act, uint8_t* out_ha,
                           const float* topk_w, float* y_pair, float* y_atomic, int max_tiles, cudaStream_t stream) {
    constexpr size_t kSmem = (size_t)2 * (128 * 128 + 128 * 16 + 128 * 208) + (size_t)128 * 12;
    cudaFuncSetAttribute((const void*)k_moe_sf4_tc_tr128_q<GU>, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)kSmem);
    int dev = 0, nsm = 0;
    cudaGetDevice(&dev);
    cudaDeviceGetAttribute(&nsm, cudaDevAttrMultiProcessorCount, dev);
    constexpr int NFB = GU ? (2 * kInter / kWR) : (kHidden / kWR);
    const unsigned grid = (unsigned)std::max(1, std::min(nsm, max_tiles * NFB));   // persistent: one CTA per SM walks the work items
    k_moe_sf4_tc_tr128_q<GU><<<grid, 512, kSmem, stream>>>(sec4_base, experts, tb, act, out_ha, topk_w, y_pair, y_atomic);
}

void moe_sf4_dn_tc_atomic(const uint8_t* sec4_base, const InkExpertDev* experts, InkMoeTables tb,
                          const uint8_t* ha, const float* topk_w, float* y_out_atomic, int max_tiles, cudaStream_t stream) {
    if (pfq_on()) {
        launch_tr128_q<false>(sec4_base, experts, tb, ha, nullptr, topk_w, nullptr, y_out_atomic, max_tiles, stream);
        return;
    }
    const unsigned grid = (unsigned)std::min(std::max(1, max_tiles * (kHidden / kWR)), 4096);
    k_moe_sf4_tc_tr128<false><<<grid, 512, 0, stream>>>(sec4_base, experts, tb, ha, nullptr, topk_w, nullptr, y_out_atomic);
}

void moe_mx8_dn_tc_atomic(const uint8_t* sec4_base, const InkExpertDev* experts, InkMoeTables tb,
                          const uint8_t* hq, const uint8_t* hsc, const float* topk_w, float* y_out_atomic, int max_tiles, cudaStream_t stream) {
    const unsigned grid = (unsigned)std::min(std::max(1, max_tiles * (kHidden / kWR)), 4096);
    k_moe_mx8_tc_tr128<false><<<grid, 512, 0, stream>>>(sec4_base, experts, tb, hq, hsc, nullptr, nullptr, topk_w, nullptr, y_out_atomic);
}

// ---------------------------------------------------------------------------------------------------------------------------------------------
// One MoE GEMM dispatch for every sm_120 card. The variant for a batch range comes from the card's table row (strata/core/ink_sm120_table.hpp) through
// `sel`; with sel == nullptr (or sel->active false) the shipped kernels run with the shipped prefetch rule, exactly as before the RTX 50 work.
// ---------------------------------------------------------------------------------------------------------------------------------------------
using strata::core::Sm120Tc;

static inline bool sel_on(const Sm120MoeSel* s) { return s && s->active; }

// STRATA_INK_SM120_PF (gate/up, default 128) and STRATA_INK_SM120_PFD (down, default 0): L2 prefetch window per weight row in bytes (multiple of 32), 0 = off.
// On a 5090 the 32 B per row and K64 step weight reads left the memory system at about half its bandwidth; asking for 128 contiguous bytes of a row
// ahead of time takes the gate/up phase at 64 slots from 20.6 to 13.1 ms (SQ-20). A prefetch is a hint: it changes no result.
static int sf4_pf_env(bool gu) {
    static const int v_gu = [] { const char* e = std::getenv("STRATA_INK_SM120_PF"); return e ? std::atoi(e) : 128; }();
    static const int v_dn = [] { const char* e = std::getenv("STRATA_INK_SM120_PFD"); return e ? std::atoi(e) : 0; }();
    return gu ? v_gu : v_dn;
}

// The MX8 kernel (layers 20-23 of a native NVFP4 file, SQ-20) takes the same window only on request: STRATA_INK_SM120_PF_MX8 = 1 (128 B) or a window in bytes,
// default 0 = no prefetch, which is the code path before the prefetch existed.
static int mx8_pf_env() {
    static const int v = [] {
        const char* e = std::getenv("STRATA_INK_SM120_PF_MX8");
        const int x = e ? std::atoi(e) : 0;
        return x == 1 ? 128 : (x >= 32 ? (x / 32) * 32 : 0);
    }();
    return v;
}

void quantize_act_fp4(const float* x, int rows, int cols, uint8_t* xa, cudaStream_t stream, bool per_row) {
    if (per_row) {
        if (rows <= 0) return;
        k_quant_act_fp4_row<<<(unsigned)rows, 256, 0, stream>>>(x, rows, cols, xa);
        return;
    }
    const int64_t nblk = (int64_t)rows * (cols / 64);
    const unsigned blocks = (unsigned)std::max<int64_t>(1, (nblk + 7) / 8);
    k_quant_act_fp4<<<blocks, 256, 0, stream>>>(x, nblk, xa);
}

void quantize_act_fp4_row(const float* x, int rows, int cols, uint8_t* xa, cudaStream_t stream) {
    quantize_act_fp4(x, rows, cols, xa, stream, true);
}

void quantize_act_mx8(const float* x, int rows, int cols, uint8_t* xq, uint8_t* xsc, cudaStream_t stream) {
    const int64_t nblk = (int64_t)rows * (cols / 32);
    const unsigned blocks = (unsigned)std::max<int64_t>(1, (nblk + 7) / 8);
    k_quant_act_mx8<<<blocks, 256, 0, stream>>>(x, nblk, xq, xsc);
}

void moe_sf4_gu_tc(const uint8_t* sec4_base, const InkExpertDev* experts, InkMoeTables tb,
                   const uint8_t* xa, uint8_t* ha, int max_tiles, int tile_rows, cudaStream_t stream, const Sm120MoeSel* sel) {
    const Sm120Tc v = sel_on(sel) ? sel->sf4 : Sm120Tc::Base;
    const int pf = (sel_on(sel) && sel->pf_gu >= 0) ? sel->pf_gu : sf4_pf_env(true);
    const bool big = (v == Sm120Tc::Bk128 || v == Sm120Tc::Tr128 || (tile_rows > 64 && v != Sm120Tc::Tr128Small && v != Sm120Tc::Tr128Bk128));
    const unsigned grid = (unsigned)std::min(std::max(1, max_tiles * (2 * kInter / kWR)), big ? 4096 : 2048);
    if (tile_rows > 64) {
        if (v == Sm120Tc::Tr128Small)     k_moe_sf4_tc_tr128_s<true, false><<<grid, kNT, 0, stream>>>(sec4_base, experts, tb, xa, ha, nullptr, nullptr, pf);
        else if (v == Sm120Tc::Tr128Bk128) k_moe_sf4_tc_tr128_s_bk128<true, false><<<grid, 512, 0, stream>>>(sec4_base, experts, tb, xa, ha, nullptr, nullptr);
        else if (pfq_on())         launch_tr128_q<true>(sec4_base, experts, tb, xa, ha, nullptr, nullptr, nullptr, max_tiles, stream);
        else                               k_moe_sf4_tc_tr128<true><<<grid, 512, 0, stream>>>(sec4_base, experts, tb, xa, ha, nullptr, nullptr);
        return;
    }
    if (v == Sm120Tc::Bk128) {
        if (tile_rows <= 16)      k_moe_sf4_tc_bk128<true, 16><<<grid, kNT, 0, stream>>>(sec4_base, experts, tb, xa, ha, nullptr, nullptr, pf);
        else if (tile_rows <= 32) k_moe_sf4_tc_bk128<true, 32><<<grid, kNT, 0, stream>>>(sec4_base, experts, tb, xa, ha, nullptr, nullptr, pf);
        else                      k_moe_sf4_tc_bk128<true, 64><<<grid, kNT, 0, stream>>>(sec4_base, experts, tb, xa, ha, nullptr, nullptr, pf);
    } else if (v == Sm120Tc::Direct) {
        if (tile_rows <= 16)      k_moe_sf4_tc_direct<true, 16><<<grid, kNT, 0, stream>>>(sec4_base, experts, tb, xa, ha, nullptr, nullptr, pf);
        else if (tile_rows <= 32) k_moe_sf4_tc_direct<true, 32><<<grid, kNT, 0, stream>>>(sec4_base, experts, tb, xa, ha, nullptr, nullptr, pf);
        else                      k_moe_sf4_tc_direct<true, 64><<<grid, kNT, 0, stream>>>(sec4_base, experts, tb, xa, ha, nullptr, nullptr, pf);
    } else {
        if (tile_rows <= 16)      k_moe_sf4_tc<true, 16><<<grid, kNT, 0, stream>>>(sec4_base, experts, tb, xa, ha, nullptr, nullptr, pf);
        else if (tile_rows <= 32) k_moe_sf4_tc<true, 32><<<grid, kNT, 0, stream>>>(sec4_base, experts, tb, xa, ha, nullptr, nullptr, pf);
        else                      k_moe_sf4_tc<true, 64><<<grid, kNT, 0, stream>>>(sec4_base, experts, tb, xa, ha, nullptr, nullptr, pf);
    }
}

void moe_sf4_dn_tc(const uint8_t* sec4_base, const InkExpertDev* experts, InkMoeTables tb,
                   const uint8_t* ha, const float* topk_w, float* y_pair, int max_tiles, int tile_rows, cudaStream_t stream,
                   bool atomic_reduce, const Sm120MoeSel* sel) {
    const Sm120Tc v = sel_on(sel) ? sel->sf4 : Sm120Tc::Base;
    const int pf = (sel_on(sel) && sel->pf_dn >= 0) ? sel->pf_dn : sf4_pf_env(false);
    const bool big = (v == Sm120Tc::Bk128 || v == Sm120Tc::Tr128 || (tile_rows > 64 && v != Sm120Tc::Tr128Small && v != Sm120Tc::Tr128Bk128));
    const unsigned grid = (unsigned)std::min(std::max(1, max_tiles * (kHidden / kWR)), big ? 4096 : 2048);
    if (tile_rows > 64) {
        if (v == Sm120Tc::Tr128Small) {
            if (atomic_reduce) k_moe_sf4_tc_tr128_s<false, true><<<grid, kNT, 0, stream>>>(sec4_base, experts, tb, ha, nullptr, topk_w, y_pair, pf);
            else               k_moe_sf4_tc_tr128_s<false, false><<<grid, kNT, 0, stream>>>(sec4_base, experts, tb, ha, nullptr, topk_w, y_pair, pf);
        } else if (v == Sm120Tc::Tr128Bk128) {
            if (atomic_reduce) k_moe_sf4_tc_tr128_s_bk128<false, true><<<grid, 512, 0, stream>>>(sec4_base, experts, tb, ha, nullptr, topk_w, y_pair);
            else               k_moe_sf4_tc_tr128_s_bk128<false, false><<<grid, 512, 0, stream>>>(sec4_base, experts, tb, ha, nullptr, topk_w, y_pair);
        } else {
            // the 5090 TR=128 kernel writes pairs (its atomic form is moe_sf4_dn_tc_atomic)
            if (pfq_on()) launch_tr128_q<false>(sec4_base, experts, tb, ha, nullptr, topk_w, y_pair, nullptr, max_tiles, stream);
            else                  k_moe_sf4_tc_tr128<false><<<grid, 512, 0, stream>>>(sec4_base, experts, tb, ha, nullptr, topk_w, y_pair);
        }
        return;
    }
    if (v == Sm120Tc::Bk128) {
        if (tile_rows <= 16)      k_moe_sf4_tc_bk128<false, 16><<<grid, kNT, 0, stream>>>(sec4_base, experts, tb, ha, nullptr, topk_w, y_pair, pf);
        else if (tile_rows <= 32) k_moe_sf4_tc_bk128<false, 32><<<grid, kNT, 0, stream>>>(sec4_base, experts, tb, ha, nullptr, topk_w, y_pair, pf);
        else                      k_moe_sf4_tc_bk128<false, 64><<<grid, kNT, 0, stream>>>(sec4_base, experts, tb, ha, nullptr, topk_w, y_pair, pf);
    } else if (v == Sm120Tc::Direct) {
        if (atomic_reduce) {
            if (tile_rows <= 16)      k_moe_sf4_tc_direct<false, 16, true><<<grid, kNT, 0, stream>>>(sec4_base, experts, tb, ha, nullptr, topk_w, y_pair, pf);
            else if (tile_rows <= 32) k_moe_sf4_tc_direct<false, 32, true><<<grid, kNT, 0, stream>>>(sec4_base, experts, tb, ha, nullptr, topk_w, y_pair, pf);
            else                      k_moe_sf4_tc_direct<false, 64, true><<<grid, kNT, 0, stream>>>(sec4_base, experts, tb, ha, nullptr, topk_w, y_pair, pf);
        } else {
            if (tile_rows <= 16)      k_moe_sf4_tc_direct<false, 16, false><<<grid, kNT, 0, stream>>>(sec4_base, experts, tb, ha, nullptr, topk_w, y_pair, pf);
            else if (tile_rows <= 32) k_moe_sf4_tc_direct<false, 32, false><<<grid, kNT, 0, stream>>>(sec4_base, experts, tb, ha, nullptr, topk_w, y_pair, pf);
            else                      k_moe_sf4_tc_direct<false, 64, false><<<grid, kNT, 0, stream>>>(sec4_base, experts, tb, ha, nullptr, topk_w, y_pair, pf);
        }
    } else {
        if (atomic_reduce) {
            if (tile_rows <= 16)      k_moe_sf4_tc<false, 16, true><<<grid, kNT, 0, stream>>>(sec4_base, experts, tb, ha, nullptr, topk_w, y_pair, pf);
            else if (tile_rows <= 32) k_moe_sf4_tc<false, 32, true><<<grid, kNT, 0, stream>>>(sec4_base, experts, tb, ha, nullptr, topk_w, y_pair, pf);
            else                      k_moe_sf4_tc<false, 64, true><<<grid, kNT, 0, stream>>>(sec4_base, experts, tb, ha, nullptr, topk_w, y_pair, pf);
        } else {
            if (tile_rows <= 16)      k_moe_sf4_tc<false, 16, false><<<grid, kNT, 0, stream>>>(sec4_base, experts, tb, ha, nullptr, topk_w, y_pair, pf);
            else if (tile_rows <= 32) k_moe_sf4_tc<false, 32, false><<<grid, kNT, 0, stream>>>(sec4_base, experts, tb, ha, nullptr, topk_w, y_pair, pf);
            else                      k_moe_sf4_tc<false, 64, false><<<grid, kNT, 0, stream>>>(sec4_base, experts, tb, ha, nullptr, topk_w, y_pair, pf);
        }
    }
}

void moe_mx8_gu_tc(const uint8_t* sec4_base, const InkExpertDev* experts, InkMoeTables tb,
                   const uint8_t* xq, const uint8_t* xsc, uint8_t* hq, uint8_t* hsc,
                   int max_tiles, int tile_rows, cudaStream_t stream, const Sm120MoeSel* sel) {
    const Sm120Tc v = sel_on(sel) ? sel->mx8 : Sm120Tc::Base;
    const bool on = sel_on(sel);
    const int pf = on ? (sel->pf_dn >= 0 ? sel->pf_dn : 0) : mx8_pf_env();
    const unsigned grid = (unsigned)std::min(std::max(1, max_tiles * (2 * kInter / kWR)), (v == Sm120Tc::Bk128 || v == Sm120Tc::Tr128) ? 4096 : 2048);
    if (tile_rows > 64) {
        // (the 5060 Ti hand-over ran the 64-row kernel on 128-row tiles here; the TR=128 kernel is the one that covers them)
        k_moe_mx8_tc_tr128<true><<<grid, 512, 0, stream>>>(sec4_base, experts, tb, xq, xsc, hq, hsc, nullptr, nullptr);
        return;
    }
    if (v == Sm120Tc::Bk128) {
        if (tile_rows <= 16)      k_moe_mx8_tc_bk64<true, 16><<<grid, kNT, 0, stream>>>(sec4_base, experts, tb, xq, xsc, hq, hsc, nullptr, nullptr, pf);
        else if (tile_rows <= 32) k_moe_mx8_tc_bk64<true, 32><<<grid, kNT, 0, stream>>>(sec4_base, experts, tb, xq, xsc, hq, hsc, nullptr, nullptr, pf);
        else                      k_moe_mx8_tc_bk64<true, 64><<<grid, kNT, 0, stream>>>(sec4_base, experts, tb, xq, xsc, hq, hsc, nullptr, nullptr, pf);
    } else {
        if (tile_rows <= 16)      k_moe_mx8_tc<true, 16><<<grid, kNT, 0, stream>>>(sec4_base, experts, tb, xq, xsc, hq, hsc, nullptr, nullptr, pf);
        else if (tile_rows <= 32) k_moe_mx8_tc<true, 32><<<grid, kNT, 0, stream>>>(sec4_base, experts, tb, xq, xsc, hq, hsc, nullptr, nullptr, pf);
        else                      k_moe_mx8_tc<true, 64><<<grid, kNT, 0, stream>>>(sec4_base, experts, tb, xq, xsc, hq, hsc, nullptr, nullptr, pf);
    }
}

void moe_mx8_dn_tc(const uint8_t* sec4_base, const InkExpertDev* experts, InkMoeTables tb,
                   const uint8_t* hq, const uint8_t* hsc, const float* topk_w, float* y_pair,
                   int max_tiles, int tile_rows, cudaStream_t stream, const Sm120MoeSel* sel) {
    const Sm120Tc v = sel_on(sel) ? sel->mx8 : Sm120Tc::Base;
    const bool on = sel_on(sel);
    // shipped: the MX8 down kernel never prefetches (pf 0); only a row that asks for it does
    const int pf = on ? (sel->pf_dn >= 0 ? sel->pf_dn : 0) : 0;
    const unsigned grid = (unsigned)std::min(std::max(1, max_tiles * (kHidden / kWR)), (v == Sm120Tc::Bk128 || v == Sm120Tc::Tr128) ? 4096 : 2048);
    if (tile_rows > 64) {
        k_moe_mx8_tc_tr128<false><<<grid, 512, 0, stream>>>(sec4_base, experts, tb, hq, hsc, nullptr, nullptr, topk_w, y_pair);
        return;
    }
    if (v == Sm120Tc::Bk128) {
        if (tile_rows <= 16)      k_moe_mx8_tc_bk64<false, 16><<<grid, kNT, 0, stream>>>(sec4_base, experts, tb, hq, hsc, nullptr, nullptr, topk_w, y_pair, pf);
        else if (tile_rows <= 32) k_moe_mx8_tc_bk64<false, 32><<<grid, kNT, 0, stream>>>(sec4_base, experts, tb, hq, hsc, nullptr, nullptr, topk_w, y_pair, pf);
        else                      k_moe_mx8_tc_bk64<false, 64><<<grid, kNT, 0, stream>>>(sec4_base, experts, tb, hq, hsc, nullptr, nullptr, topk_w, y_pair, pf);
    } else {
        if (tile_rows <= 16)      k_moe_mx8_tc<false, 16><<<grid, kNT, 0, stream>>>(sec4_base, experts, tb, hq, hsc, nullptr, nullptr, topk_w, y_pair, pf);
        else if (tile_rows <= 32) k_moe_mx8_tc<false, 32><<<grid, kNT, 0, stream>>>(sec4_base, experts, tb, hq, hsc, nullptr, nullptr, topk_w, y_pair, pf);
        else                      k_moe_mx8_tc<false, 64><<<grid, kNT, 0, stream>>>(sec4_base, experts, tb, hq, hsc, nullptr, nullptr, topk_w, y_pair, pf);
    }
}

void dense_sf4_gemm_tc(const uint8_t* w_sf4, const uint8_t* xa_fp4, float* y_f32,
                       int M, int N, int K, cudaStream_t stream, bool direct) {
    const int pf = sf4_pf_env(true);
    dim3 grid((N + kWR - 1) / kWR, (M + kTR - 1) / kTR);
    if (direct) {
        k_dense_sf4_tc<true><<<grid, kNT, 0, stream>>>(w_sf4, xa_fp4, y_f32, M, N, K, pf);
    } else {
        k_dense_sf4_tc<false><<<grid, kNT, 0, stream>>>(w_sf4, xa_fp4, y_f32, M, N, K, pf);
    }
}


void test_sf4_gemm_tc(const uint8_t* w_u8, const uint8_t* w_sc, float w_outer,
                      const float* x_f32, float* y_f32, int M, int N, int K, cudaStream_t stream) {
    uint8_t* d_xa = dalloc<uint8_t>((size_t)M * (K / 64) * kAB);
    quantize_act_fp4(x_f32, M, K, d_xa, stream);
    dim3 grid((N + kWR - 1) / kWR, (M + kTR - 1) / kTR);
    k_test_sf4_gemm<<<grid, kNT, 0, stream>>>(w_u8, w_sc, w_outer, d_xa, y_f32, M, N, K);
    INK_CK(cudaStreamSynchronize(stream));
    INK_CK(cudaFree(d_xa));
}

void test_mx8_gemm_tc(const uint8_t* w_u8, const uint8_t* w_sc,
                      const float* x_f32, float* y_f32, int M, int N, int K, cudaStream_t stream) {
    uint8_t* d_xq = dalloc<uint8_t>((size_t)M * K);
    uint8_t* d_xsc = dalloc<uint8_t>((size_t)M * (K / 32));
    quantize_act_mx8(x_f32, M, K, d_xq, d_xsc, stream);
    dim3 grid((N + kWR - 1) / kWR, (M + kTR - 1) / kTR);
    k_test_mx8_gemm<<<grid, kNT, 0, stream>>>(w_u8, w_sc, d_xq, d_xsc, y_f32, M, N, K);
    INK_CK(cudaStreamSynchronize(stream));
    INK_CK(cudaFree(d_xq));
    INK_CK(cudaFree(d_xsc));
}

}  // namespace strata::ink::sm120
