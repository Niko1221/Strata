// src/kernels/cuda/s2_qpn8.cu - the V100 expert GEMV: QPN8 load-time repack + m8n8k4.  Read the header first.
//
// THE BLOB GEOMETRY IS `s2_expert_grouped.cu`'s and `cpu/expert.hpp`'s; the outputs are the DP4A kernels'
// (gate-major `gate_up`, `out` rows through `ent_dst`), per chunk bit-exactly their float terms.
//
// The m8n8k4 dataflow is llama.cpp-v100 `q8-skinny.cu`'s GEMV form (itself 1Cat-vLLM fp8_qpn8_sm70.cu,
// dnv2003/v100-skinny): one warp-block per 32-row tile, the group's entries in the m8 rows, SplitK warps
// over the 16-K groups, the B fragments coming from the repacked records through the 1025 bias trick.
#include "strata/kernels/s2_qpn8.hpp"

#include "strata/kernels/s2_expert_grouped.hpp"

#include "strata/kernels/quantize_act.hpp"

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>

namespace strata::kernels {

// PR #600's rule: the m8n8k4 path is compiled only into the experimental Volta build
// (-DSTRATA_EXPERIMENTAL_SM60=ON).  Every other backend gets the fallbacks below, so the ready-made engine loads
// no extra kernels and `s2_qpn8_active()` is a constant false - exactly the DP4A behaviour it had before.
#if !defined(__HIPCC__) && defined(STRATA_EXPERIMENTAL_SM60)
namespace {

constexpr int H = 2560, FF = 640, QK = 64;
constexpr int ROW_GU = H / 4, ROW_D = FF / 4, SC_GU = H / QK, SC_D = FF / QK;
constexpr size_t O_D_CODES = (size_t) 2 * FF * ROW_GU;
constexpr size_t O_GU_SCALES = O_D_CODES + (size_t) H * ROW_D;
constexpr size_t O_D_SCALES = O_GU_SCALES + (size_t) 2 * FF * SC_GU * 2;
constexpr int NC_GU = H / 32, NC_D = FF / 32;
constexpr int GMAX = 8;                       // entries per group (one m8 tile)
constexpr int THREADS = 256;

__device__ __forceinline__ float f16_ld(const uint8_t* p) {
    return __half2float(__ushort_as_half(*(const unsigned short*) p));
}

// 16 consecutive int8 at a 2-byte-aligned `addr` as the 4 aligned 32-bit words the A-fragment converter takes.
// The q8_0 block is 34 bytes, so `addr` is 2-mod-4 half the time and a `memcpy` of 8/16 bytes is scalarised
// into BYTE loads - the A load was 28 of the gu kernel's 74 us at 10 x 8 that way (bench/micro ablation).
// Five `__ldg` of aligned words + funnel shifts give the four words instead; `off` is 0 or 2.
__device__ __forceinline__ void load_a16(const uint8_t* addr, uint2& a01, uint2& a23) {
    const int off = (int) ((uintptr_t) addr & 3);
    const unsigned* p = (const unsigned*) (addr - off);
    unsigned v[5];
#pragma unroll
    for (int j = 0; j < 5; ++j) v[j] = __ldg(p + j);
    const unsigned sh = (unsigned) off * 8;
    unsigned w[4];
#pragma unroll
    for (int j = 0; j < 4; ++j) w[j] = (unsigned) ((((unsigned long long) v[j + 1] << 32) | v[j]) >> sh);
    a01 = make_uint2(w[0], w[1]);
    a23 = make_uint2(w[2], w[3]);
}

// ---- the QPN8 maps (llama.cpp-v100 q-skinny-common.cuh, unchanged) ----
__device__ __forceinline__ int qpn8_lane_from_col(int col) {
    return (col & 3) | (((col >> 3) & 3) << 2) | (((col >> 2) & 1) << 4);
}
__device__ __forceinline__ int qpn8_physical_k(int logical_k) {
    const int local = logical_k & 7;
    return (logical_k & 8) + (local >> 1) + ((local & 1) << 2);
}

// ---- the 1024/1025 bias trick (q8-skinny s8x8_to_half2x4, q4k-skinny decode_record) ----
//
// 1024..1027 are exact in f16, so `0x6400 | code` minus 1024 is the code and minus 1025 is `code - 1` -
// the weight bias the DP4A kernels apply through `s - hx` - with no rounding and no branch.  The products
// with the int8 activations are integers below 255, exact in f16 and in f32, so the chunk sums are exact.

// int8 x8 -> 4 half2, halves (2j, 2j + 1): the A-fragment pairing of a k16 group's logical order.
//
// **`__byte_perm` SELECTS BY NIBBLE, NOT BY BYTE.**  Result byte `i` is source byte `((sel >> 4 * i) & 7)`
// (source bytes 0-3 = `x`, 4-7 = `y`) - `s2_expert_grouped.cu`'s transpose selectors (`0x5140` = "a0 b0 a1
// b1") are the readable example.  The first version of this function placed the second source index at
// `<< 16` (nibble 3) while the mask keeps result bytes 0 and 2 (nibbles 0 and 2), so every half2 came out
// as `(pair, first byte)` and the m8n8k4 products were garbage - `s2_qpn8_parity` failed at 1.0 relative.
// The index belongs at `<< 8`; the filler nibbles are masked off.
__device__ __forceinline__ void s8x8_consec_to_half2x4(uint2 q, half2 out[4]) {
    const unsigned x = q.x ^ 0x80808080u;
    const unsigned y = q.y ^ 0x80808080u;
    const unsigned bias_bits = 0x64806480u;
    const half2 bias = *reinterpret_cast<const half2*>(&bias_bits);
#pragma unroll
    for (int j = 0; j < 4; ++j) {
        // out bytes (0, 2) = source bytes 2j and 2j + 1 of the (x, y) byte stream
        const unsigned b0 = (2 * j) < 4 ? (unsigned) (2 * j) : (((unsigned) (2 * j) - 4u) | 4u);
        const unsigned b1 = (2 * j + 1) < 4 ? (unsigned) (2 * j + 1) : (((unsigned) (2 * j + 1) - 4u) | 4u);
        const unsigned packed = (__byte_perm(x, y, b0 | (b1 << 8)) & 0x00FF00FFu) | 0x64006400u;
        out[j] = __hsub2(*reinterpret_cast<const half2*>(&packed), bias);
    }
}

// 16 packed 2-bit codes (a record: logical j at bits 2 * physical_k(j)) -> 8 half2 of (code - 1).
__device__ __forceinline__ void q2rec_to_bfrag(unsigned p, half2 out[8]) {
    const unsigned offset_bits = 0x64016401u;            // f16 1025, 1025: 1024 + code - 1025 = code - 1
    const half2 offset = *reinterpret_cast<const half2*>(&offset_bits);
#pragma unroll
    for (int i = 0; i < 8; ++i) {
        const int s = (i & 3) + 8 * (i >> 2);            // physical slots (s, s + 4)
        const unsigned q0 = (p >> (2 * s)) & 3u;
        const unsigned q1 = (p >> (2 * (s + 4))) & 3u;
        const unsigned h = (0x6400u | q0) | ((0x6400u | q1) << 16);
        out[i] = __hsub2(*reinterpret_cast<const half2*>(&h), offset);
    }
}

#define S2_QPN8_MMA_8N8K4(C, A0, A1, B0, B1)                           \
  asm volatile(                                                        \
      "mma.sync.aligned.m8n8k4.row.col.f32.f16.f16.f32 "               \
      "{%0,%1,%2,%3,%4,%5,%6,%7}, {%8,%9}, {%10,%11}, "                \
      "{%0,%1,%2,%3,%4,%5,%6,%7};\n"                                   \
      : "+f"(C[0]), "+f"(C[1]), "+f"(C[2]), "+f"(C[3]), "+f"(C[4]),    \
        "+f"(C[5]), "+f"(C[6]), "+f"(C[7])                             \
      : "r"(A0), "r"(A1), "r"(B0), "r"(B1))

// A k16 group's int8 -> the 16 A-fragment halves (the two 8-byte halves, converted and paired).
__device__ __forceinline__ void qpn8_conv_a(const uint8_t* addr, half2 ah[8]) {
    uint2 a01 = make_uint2(0, 0), a23 = make_uint2(0, 0);
    load_a16(addr, a01, a23);
    s8x8_consec_to_half2x4(a01, ah);
    s8x8_consec_to_half2x4(a23, ah + 4);
}

// The precomputed 32-byte A fragment (8 half2) of one (entry, k16 group): two uint4 loads, no conversion.
__device__ __forceinline__ void load_afrag(const uint4* p, half2 ah[8]) {
    const uint4 lo = __ldg(p), hi = __ldg(p + 1);
    memcpy(ah, &lo, 16);
    memcpy(ah + 4, &hi, 16);
}

// One k16 group's four m8n8k4 steps into `s`, from an A fragment already in registers and the group's B record.
// Split out so a chunk's two groups can be issued into two accumulator chains (the loops below); without that
// the eight dependent HMMA of one chunk were the mma pipeline's critical path.
__device__ __forceinline__ void qpn8_mma_ah(float s[8], const half2 ah[8], unsigned rec) {
    half2 bw[8];
    q2rec_to_bfrag(rec, bw);
    const unsigned* a0 = reinterpret_cast<const unsigned*>(ah);
    const unsigned* a1 = reinterpret_cast<const unsigned*>(ah + 4);
    const unsigned* b = reinterpret_cast<const unsigned*>(bw);
    S2_QPN8_MMA_8N8K4(s, a0[0], a0[1], b[0], b[1]);
    S2_QPN8_MMA_8N8K4(s, a0[2], a0[3], b[2], b[3]);
    S2_QPN8_MMA_8N8K4(s, a1[0], a1[1], b[4], b[5]);
    S2_QPN8_MMA_8N8K4(s, a1[2], a1[3], b[6], b[7]);
}

// The A fragments for a whole window: one thread per (entry, k16 group) converts the entry's token row once.
// The gu kernel's per-tile A load/convert was its largest cost (16 us of 48 at 10 x 8, ablation); every one
// of the 40 row-tiles re-read and re-converted the same 16 bytes.  `out[(e * groups + g)]` is the 32-byte
// fragment (8 half2, the 16 halves qpn8_conv_a produces); the gu loop then reads it as two uint4.
__global__ void prep_a_kernel(const uint8_t* __restrict__ src, const int32_t* __restrict__ idx_tok,
                              const int32_t* __restrict__ grp_start, const int32_t* __restrict__ n_groups,
                              int nc, int groups, uint4* __restrict__ out, int ident) {
    const int g = blockIdx.y;
    if (g >= *n_groups) return;
    const int e0 = grp_start[g];
    const int ne = min(grp_start[g + 1] - e0, GMAX);
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    const int k = i / groups, gi = i - k * groups;      // local entry of the group, k16 group
    if (k >= ne) return;
    const int e = e0 + k;
    const int row = ident ? e : idx_tok[e];
    const uint8_t* blk = src + (size_t) row * (size_t) nc * 34 + (size_t) (gi >> 1) * 34 + ((gi & 1) ? 18 : 2);
    half2 ah[8];
    qpn8_conv_a(blk, ah);
    uint4* o = out + ((size_t) e * groups + gi) * 2;
    o[0] = *reinterpret_cast<const uint4*>(ah);
    o[1] = *reinterpret_cast<const uint4*>(ah + 4);
}

// ---- the repack: canonical Q2_0 code plane -> [(row/32) tiles][k16 groups][32 lanes][4 B] ----
//
// One thread per (row, 16-K group): the row's 4 canonical code bytes for elements [16g, 16g + 16) become
// one record, code j at bits 2 * qpn8_physical_k(j), stored at lane `qpn8_lane_from_col(row & 31)` so the
// runtime lane reads one coalesced word per (tile, group).  Pure permutation: same bytes in, same bytes out.
__global__ void qpn8_repack_kernel(const uint8_t* __restrict__ src, unsigned* __restrict__ dst, int rows,
                                   int row_stride, int groups_k16) {
    const size_t i = (size_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= (size_t) rows * (size_t) groups_k16) return;
    const int g = (int) (i % groups_k16);
    const int r = (int) (i / groups_k16);
    const uint8_t* cb = src + (size_t) r * (size_t) row_stride + (size_t) g * 4;
    unsigned rec = 0;
#pragma unroll
    for (int j = 0; j < 16; ++j) {
        const unsigned c = ((unsigned) cb[j >> 2] >> (2 * (j & 3))) & 3u;
        rec |= c << (2 * qpn8_physical_k(j));
    }
    const int tile = r >> 5;
    const int lane = qpn8_lane_from_col(r & 31);
    dst[(((size_t) tile * (size_t) groups_k16 + (size_t) g) << 5) + (size_t) lane] = rec;
}

// ---- the GEMV kernels ----
//
// Per block: one 32-row tile of one group.  SplitK warps walk the k16 groups two at a time (one 32-element
// chunk = the DP4A kernels' reduction unit); after each chunk the C fragment holds the EXACT integer
// `s - hx` of that chunk for the block's 8 x 32 outputs (the m8 rows are the group's entries), and the
// readout applies the DP4A kernels' `dw * dx * (float)(s - hx)` before zeroing C.  The split-K reduce over
// the warps is in a fixed order, so a rerun of the same input reproduces the same bits.

template <int SplitK, int NeGE>
__global__ void __launch_bounds__(256) gu_qpn8_kernel(const unsigned long long* __restrict__ grp_ptr,
                                                      size_t rep_off, const int32_t* __restrict__ grp_start,
                                                      const int32_t* __restrict__ n_groups,
                                                      const int32_t* __restrict__ ent_tok,
                                                      const uint8_t* __restrict__ x_q8_0,
                                                      const float* __restrict__ x_scales,
                                                      const uint4* __restrict__ a_frag,
                                                      float* __restrict__ gate_up, int cap_entries) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ == 700
    __shared__ float partials[SplitK][256];
    // The scales, staged once per block.  The first version loaded `dx` and `dw` per CELL per CHUNK from
    // global memory (16 loads per lane-pair per chunk, 4 of them distinct): the full pipeline then ran
    // 140 us against the DP4A kernels' 110 us at 10 experts x 8 entries even though the GEMV inner loop
    // alone is 46 vs 89 (bench/micro/s2_unpack_bench.cu).  Both planes are tiny (2.5 + 5 KB) and the block
    // owns one 32-row tile and one group's entries, so they live in shared memory and the per-chunk
    // readout is eight fused multiply-adds.  The float expression is unchanged: `dw * dx * s`.
    __shared__ float xds[GMAX][NC_GU];            // the entries' chunk scales
    __shared__ float wds[32][SC_GU];              // the tile's row scales
    const int g = blockIdx.y;
    if (g >= *n_groups) return;
    const int e0 = grp_start[g], ne = min(grp_start[g + 1] - e0, 8);
    if (ne < NeGE) return;                        // the hybrid dispatch's other half owns this group
    const int tile = blockIdx.x;
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    const int mrow = (lane & 3) + ((lane & 16) ? 4 : 0);           // the entry this lane's A holds
    const int groups_k16 = H / 16;
    const int gpw = groups_k16 / SplitK;                           // even: one chunk = two k16 groups
    const int gbegin = warp * gpw;
    const uint8_t* blob = (const uint8_t*) grp_ptr[g] + rep_off;
    for (int i = threadIdx.x; i < ne * NC_GU; i += blockDim.x) {
        const int k = i / NC_GU, c = i - k * NC_GU;
        const int tok = ent_tok[e0 + k];
        xds[k][c] = x_scales ? x_scales[(size_t) tok * NC_GU + c]
                             : f16_ld(x_q8_0 + (size_t) tok * (size_t) NC_GU * 34 + (size_t) c * 34);
    }
    for (int i = threadIdx.x; i < 32 * SC_GU; i += blockDim.x) {
        const int r = i / SC_GU, sc = i - r * SC_GU;
        wds[r][sc] = f16_ld(blob + O_GU_SCALES + (size_t) (tile * 32 + r) * SC_GU * 2 + (size_t) sc * 2);
    }
    __syncthreads();
    const unsigned* code_ptr = (const unsigned*) blob + (((size_t) tile * groups_k16 + gbegin) << 5) + lane;
    // This lane's entry fragment base: `prep_a_kernel` converted every (entry, k16 group) once, so the loop
    // below is two coalesced uint4 loads per group, not a scalar-load+convert per TILE (the 40-tile redo was
    // the gu kernel's largest single cost).
    const bool a_on = mrow < ne;
    const uint4* a_base = a_frag + ((size_t) (e0 + (a_on ? mrow : 0)) * groups_k16) * 2;
    float acc[8];
#pragma unroll
    for (int i = 0; i < 8; ++i) acc[i] = 0.0f;
    float s[8];
#pragma unroll
    for (int i = 0; i < 8; ++i) s[i] = 0.0f;
    float s1[8];
#pragma unroll
    for (int i = 0; i < 8; ++i) s1[i] = 0.0f;
    // The readout's maps are lane constants: this lane's two m rows are `ma, ma + 2` and its four columns
    // are `cbase + {0, 1, 4, 5}` (cell i: m = ma + (i & 2), col = cbase + (i & 1) + 4 * (i >> 2)).  The
    // nested readout below costs two shared `dx` loads and eight `dw` loads per chunk instead of sixteen.
    const int ma = ((lane & 16) ? 4 : 0) + (lane & 1);
    const int cbase = ((lane >> 2) & 3) * 8 + 2 * ((lane >> 1) & 1);
    // A chunk is two k16 groups.  Issue the two groups' four mma each into separate accumulators so the
    // eight dependent HMMA of a chunk are no longer the mma pipeline's critical path; combine, read out and
    // reset per chunk.  gpw is even (gu 20, down 10), so the pair loop covers it exactly.
    for (int j = 0; j < gpw; j += 2) {
        half2 ah0[8], ah1[8];
        load_afrag(a_base + (size_t) (gbegin + j) * 2, ah0);
        load_afrag(a_base + (size_t) (gbegin + j + 1) * 2, ah1);
        qpn8_mma_ah(s, ah0, __ldcs(code_ptr + (size_t) j * 32));
        qpn8_mma_ah(s1, ah1, __ldcs(code_ptr + (size_t) (j + 1) * 32));
#pragma unroll
        for (int i = 0; i < 8; ++i) { s[i] += s1[i]; s1[i] = 0.0f; }
        // chunk complete (two k16 groups): the DP4A kernels' per-chunk term, then C is the next chunk's
        const int chunk = (gbegin + j) >> 1;
#pragma unroll
        for (int cm = 0; cm < 2; ++cm) {
            const int m = ma + 2 * cm;
            if (m >= ne) continue;
            const float dx = xds[m][chunk];
            const int i0 = cm << 1;                       // cells i0, i0|1, i0|4, i0|5 (m bit at i&2)
            acc[i0] += wds[cbase][chunk >> 1] * dx * s[i0];
            acc[i0 | 1] += wds[cbase + 1][chunk >> 1] * dx * s[i0 | 1];
            acc[i0 | 4] += wds[cbase + 4][chunk >> 1] * dx * s[i0 | 4];
            acc[i0 | 5] += wds[cbase + 5][chunk >> 1] * dx * s[i0 | 5];
        }
#pragma unroll
        for (int i = 0; i < 8; ++i) s[i] = 0.0f;
    }
    for (int i = 0; i < 8; ++i) {
        const int m = (i & 2) + ((lane & 16) ? 4 : 0) + (lane & 1);
        const int col = ((lane >> 2) & 3) * 8 + (i & 1) + (((lane >> 1) & 1) << 1) + ((i >> 2) << 2);
        partials[warp][m * 32 + col] = (m < ne) ? acc[i] : 0.0f;
    }
    __syncthreads();
    for (int e = threadIdx.x; e < 256; e += blockDim.x) {
        float v = 0.0f;
#pragma unroll
        for (int w = 0; w < SplitK; ++w) v += partials[w][e];
        const int m = e >> 5, col = e & 31;
        if (m >= ne) continue;
        const int slot = tile * 32 + col;                          // the gate/up row-slot
        const int r = slot >> 1;
        float* o = (slot & 1) ? gate_up + (size_t) cap_entries * FF + (size_t) (e0 + m) * FF
                              : gate_up + (size_t) (e0 + m) * FF;
        o[r] = v;
    }
#else
    (void) grp_ptr; (void) rep_off; (void) grp_start; (void) n_groups; (void) ent_tok; (void) x_q8_0;
    (void) x_scales; (void) a_frag; (void) gate_up; (void) cap_entries;
#endif
}

template <int SplitK, int NeGE>
__global__ void __launch_bounds__(256) down_qpn8_kernel(const unsigned long long* __restrict__ grp_ptr,
                                                        size_t rep_off, const int32_t* __restrict__ grp_start,
                                                        const int32_t* __restrict__ n_groups,
                                                        const int32_t* __restrict__ ent_dst,
                                                        const uint8_t* __restrict__ h_q8_0,
                                                        const float* __restrict__ h_scales,
                                                        const uint4* __restrict__ a_frag,
                                                        float* __restrict__ out) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ == 700
    __shared__ float partials[SplitK][256];
    __shared__ float xds[GMAX][NC_D];              // the entries' chunk scales (see gu_qpn8_kernel)
    __shared__ float wds[32][SC_D];                // the tile's row scales
    const int g = blockIdx.y;
    if (g >= *n_groups) return;
    const int e0 = grp_start[g], ne = min(grp_start[g + 1] - e0, 8);
    if (ne < NeGE) return;
    const int tile = blockIdx.x;
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    const int mrow = (lane & 3) + ((lane & 16) ? 4 : 0);
    const int groups_k16 = FF / 16;
    const int gpw = groups_k16 / SplitK;
    const int gbegin = warp * gpw;
    const uint8_t* blob = (const uint8_t*) grp_ptr[g] + rep_off;
    for (int i = threadIdx.x; i < ne * NC_D; i += blockDim.x) {
        const int k = i / NC_D, c = i - k * NC_D;
        xds[k][c] = h_scales ? h_scales[(size_t) (e0 + k) * NC_D + c]
                             : f16_ld(h_q8_0 + (size_t) (e0 + k) * (size_t) NC_D * 34 + (size_t) c * 34);
    }
    for (int i = threadIdx.x; i < 32 * SC_D; i += blockDim.x) {
        const int r = i / SC_D, sc = i - r * SC_D;
        wds[r][sc] = f16_ld(blob + O_D_SCALES + (size_t) (tile * 32 + r) * SC_D * 2 + (size_t) sc * 2);
    }
    __syncthreads();
    // The down code plane's records start at `O_D_CODES` of the repacked blob (the repack writes them
    // there, matching the canonical offset); without the offset this kernel read the gate/up plane.
    const unsigned* code_ptr =
        (const unsigned*) (blob + O_D_CODES) + (((size_t) tile * groups_k16 + gbegin) << 5) + lane;
    const bool a_on = mrow < ne;
    const uint4* a_base = a_frag + ((size_t) (e0 + (a_on ? mrow : 0)) * groups_k16) * 2;
    float acc[8];
#pragma unroll
    for (int i = 0; i < 8; ++i) acc[i] = 0.0f;
    float s[8];
#pragma unroll
    for (int i = 0; i < 8; ++i) s[i] = 0.0f;
    float s1[8];
#pragma unroll
    for (int i = 0; i < 8; ++i) s1[i] = 0.0f;
    // The readout's maps are lane constants: this lane's two m rows are `ma, ma + 2` and its four columns
    // are `cbase + {0, 1, 4, 5}` (cell i: m = ma + (i & 2), col = cbase + (i & 1) + 4 * (i >> 2)).  The
    // nested readout below costs two shared `dx` loads and eight `dw` loads per chunk instead of sixteen.
    const int ma = ((lane & 16) ? 4 : 0) + (lane & 1);
    const int cbase = ((lane >> 2) & 3) * 8 + 2 * ((lane >> 1) & 1);
    // Two mma accumulator chains per chunk (see gu_qpn8_kernel).  down gpw is 10, even.
    for (int j = 0; j < gpw; j += 2) {
        half2 ah0[8], ah1[8];
        load_afrag(a_base + (size_t) (gbegin + j) * 2, ah0);
        load_afrag(a_base + (size_t) (gbegin + j + 1) * 2, ah1);
        qpn8_mma_ah(s, ah0, __ldcs(code_ptr + (size_t) j * 32));
        qpn8_mma_ah(s1, ah1, __ldcs(code_ptr + (size_t) (j + 1) * 32));
#pragma unroll
        for (int i = 0; i < 8; ++i) { s[i] += s1[i]; s1[i] = 0.0f; }
        const int chunk = (gbegin + j) >> 1;
#pragma unroll
        for (int cm = 0; cm < 2; ++cm) {
            const int m = ma + 2 * cm;
            if (m >= ne) continue;
            const float dx = xds[m][chunk];
            const int i0 = cm << 1;                       // cells i0, i0|1, i0|4, i0|5 (m bit at i&2)
            acc[i0] += wds[cbase][chunk >> 1] * dx * s[i0];
            acc[i0 | 1] += wds[cbase + 1][chunk >> 1] * dx * s[i0 | 1];
            acc[i0 | 4] += wds[cbase + 4][chunk >> 1] * dx * s[i0 | 4];
            acc[i0 | 5] += wds[cbase + 5][chunk >> 1] * dx * s[i0 | 5];
        }
#pragma unroll
        for (int i = 0; i < 8; ++i) s[i] = 0.0f;
    }
    for (int i = 0; i < 8; ++i) {
        const int m = (i & 2) + ((lane & 16) ? 4 : 0) + (lane & 1);
        const int col = ((lane >> 2) & 3) * 8 + (i & 1) + (((lane >> 1) & 1) << 1) + ((i >> 2) << 2);
        partials[warp][m * 32 + col] = (m < ne) ? acc[i] : 0.0f;
    }
    __syncthreads();
    for (int e = threadIdx.x; e < 256; e += blockDim.x) {
        float v = 0.0f;
#pragma unroll
        for (int w = 0; w < SplitK; ++w) v += partials[w][e];
        const int m = e >> 5, col = e & 31;
        if (m >= ne) continue;
        out[(size_t) ent_dst[e0 + m] * H + tile * 32 + col] = v;
    }
#else
    (void) grp_ptr; (void) rep_off; (void) grp_start; (void) n_groups; (void) ent_dst; (void) h_q8_0;
    (void) h_scales; (void) a_frag; (void) out;
#endif
}

__global__ void swiglu_kernel(float* __restrict__ gate_up, long long n_pairs) {
    const long long i = (long long) blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n_pairs) return;
    const float g = gate_up[i];
    const float u = gate_up[n_pairs + i];
    gate_up[i] = (g / (1.0f + __expf(-g))) * u;
}

void check(const char* who) {
    const cudaError_t e = cudaGetLastError();
    if (e != cudaSuccess) {
        std::fprintf(stderr, "%s launch: %s\n", who, cudaGetErrorString(e));
        std::exit(1);
    }
}

// The window's precomputed A fragments (gu then down).  A process-lifetime buffer sized to the largest call
// seen, so the hot path never allocates; the opt-in knob means the DP4A path and the default tree never pay
// for it.  `p` is device memory.
uint8_t* qpn8_window_buf(size_t bytes) {
    static uint8_t* p = nullptr;
    static size_t cap = 0;
    if (bytes > cap) {
        if (p != nullptr) cudaFree(p);
        p = nullptr;
        cap = 0;
        if (cudaMalloc((void**) &p, bytes) != cudaSuccess) {
            std::fprintf(stderr, "s2_qpn8: window buffer alloc (%zu bytes) failed\n", bytes);
            std::exit(1);
        }
        cap = bytes;
    }
    return p;
}

}  // namespace

bool s2_qpn8_active() {
#if defined(STRATA_USE_HIP)
    return false;
#else
    static const bool on = [] {
        const char* e = std::getenv("STRATA_QPN8");
        if (e == nullptr || e[0] != '1') return false;             // opt-in: the trunk and sm_70 default are
        int dev = 0, major = 0, minor = 0;                         // untouched (measured: wins only at
        if (cudaGetDevice(&dev) != cudaSuccess) return false;      // >= 4 entries per expert)
        if (cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, dev) != cudaSuccess) return false;
        if (cudaDeviceGetAttribute(&minor, cudaDevAttrComputeCapabilityMinor, dev) != cudaSuccess) return false;
        return major == 7 && minor == 0;                           // mma.m8n8k4 is Volta-only
    }();
    return on;
#endif
}

int64_t s2_qpn8_blob_bytes() { return (int64_t) (O_D_SCALES + (size_t) H * SC_D * 2); }

int64_t s2_qpn8_slot_bytes(int64_t blob) {
    return (s2_qpn8_active() && blob == s2_qpn8_blob_bytes()) ? 2 * blob : blob;
}

void s2_qpn8_repack_blob(uint8_t* dst, const uint8_t* src, int64_t blob_bytes, void* stream) {
    if (!s2_qpn8_active()) return;
    if (blob_bytes != s2_qpn8_blob_bytes()) {
        std::fprintf(stderr, "s2_qpn8_repack_blob: %lld bytes is not the canonical Q2_0 expert blob\n",
                     (long long) blob_bytes);
        std::exit(1);
    }
    cudaStream_t cs = (cudaStream_t) stream;
    // both code planes; the scales stay where they are (dst + the canonical offsets)
    {
        const int rows = 2 * FF, groups = H / 16;
        const unsigned blocks = (unsigned) ((rows * groups + THREADS - 1) / THREADS);
        qpn8_repack_kernel<<<blocks, THREADS, 0, cs>>>(src, (unsigned*) dst, rows, ROW_GU, groups);
        check("s2_qpn8_repack_blob/gu");
    }
    {
        const int rows = H, groups = FF / 16;
        const unsigned blocks = (unsigned) ((rows * groups + THREADS - 1) / THREADS);
        qpn8_repack_kernel<<<blocks, THREADS, 0, cs>>>(src + O_D_CODES, (unsigned*) (dst + O_D_CODES), rows,
                                                       ROW_D, groups);
        check("s2_qpn8_repack_blob/down");
    }
    // the scale planes are small and read back through the canonical offsets: copy them unchanged
    if (cudaMemcpyAsync(dst + O_GU_SCALES, src + O_GU_SCALES,
                        (size_t) H * SC_D * 2 + (size_t) 2 * FF * SC_GU * 2, cudaMemcpyDeviceToDevice,
                        cs) != cudaSuccess) {
        std::fprintf(stderr, "s2_qpn8_repack_blob: scale copy failed\n");
        std::exit(1);
    }
}

void moe_grouped_s2_qpn8(const unsigned long long* grp_ptr, const int32_t* grp_start,
                         const int32_t* n_groups, const int32_t* ent_dst, const int32_t* ent_tok,
                         int64_t cap_groups, int64_t cap_entries, int64_t blob_bytes, const uint8_t* x_q8_0,
                         const float* x_scales, void* scratch, float* out, void* stream) {
    if (cap_groups <= 0 || cap_entries <= 0) return;
    if (!s2_qpn8_active() || blob_bytes != s2_qpn8_blob_bytes()) {
        // any other geometry (or the knob off): the DP4A kernels, reading the canonical half as always
        moe_grouped_s2(grp_ptr, grp_start, n_groups, ent_dst, ent_tok, cap_groups, cap_entries, x_q8_0,
                       x_scales, scratch, out, stream);
        return;
    }
    cudaStream_t cs = (cudaStream_t) stream;
    const uint64_t gu_bytes = ((uint64_t) cap_entries * (uint64_t) (2 * FF) * 4 + 15) & ~15ull;
    const uint64_t q8_bytes = ((uint64_t) cap_entries * (uint64_t) (FF / 32) * 34 + 15) & ~15ull;
    float* gate_up = (float*) scratch;
    uint8_t* h_q8_0 = (uint8_t*) scratch + gu_bytes;
    float* h_scales = (float*) ((uint8_t*) scratch + gu_bytes + q8_bytes);
    const size_t rep_off = (size_t) blob_bytes;                    // the dual-form slot's second half
    // The A fragments: gu groups_k16 = 160, down 40; 32 bytes per (entry, group).  One prep pass per window,
    // then the gemv reads 2 x uint4 per group instead of re-loading and re-converting per row-tile.
    const uint64_t gu_groups = H / 16, dn_groups = FF / 16;
    const uint64_t gu_ab = (uint64_t) cap_entries * gu_groups * 32;
    uint8_t* abuf = qpn8_window_buf((size_t) (gu_ab + (uint64_t) cap_entries * dn_groups * 32));
    uint4* gu_afrag = (uint4*) abuf;
    uint4* dn_afrag = (uint4*) (abuf + gu_ab);
    {
        const dim3 grid((unsigned) ((GMAX * gu_groups + THREADS - 1) / THREADS), (unsigned) cap_groups);
        prep_a_kernel<<<grid, THREADS, 0, cs>>>(x_q8_0, ent_tok, grp_start, n_groups, NC_GU, (int) gu_groups,
                                                gu_afrag, 0);
        check("moe_grouped_s2_qpn8/prep_gu");
    }
    // SplitK: gu 160 k16-groups -> 8 warps of 20; down 40 -> 4 warps of 10 (both even, chunks align)
    {
        const dim3 grid((unsigned) (2 * FF / 32), (unsigned) cap_groups);
        gu_qpn8_kernel<8, 0><<<grid, 8 * 32, 0, cs>>>(grp_ptr, rep_off, grp_start, n_groups, ent_tok, x_q8_0,
                                                     x_scales, gu_afrag, gate_up, (int) cap_entries);
        check("moe_grouped_s2_qpn8/gu");
    }
    {
        const long long pairs = cap_entries * (long long) FF;
        swiglu_kernel<<<(unsigned) ((pairs + THREADS - 1) / THREADS), THREADS, 0, cs>>>(gate_up, pairs);
        check("moe_grouped_s2_qpn8/swiglu");
    }
    if (x_scales != nullptr) quantize_q8_0_scaled(gate_up, h_q8_0, h_scales, cap_entries * (int64_t) FF, stream);
    else quantize_q8_0(gate_up, h_q8_0, cap_entries * (int64_t) FF, stream);
    {
        const dim3 grid((unsigned) ((GMAX * dn_groups + THREADS - 1) / THREADS), (unsigned) cap_groups);
        prep_a_kernel<<<grid, THREADS, 0, cs>>>(h_q8_0, nullptr, grp_start, n_groups, NC_D, (int) dn_groups,
                                                dn_afrag, 1);
        check("moe_grouped_s2_qpn8/prep_down");
    }
    {
        const dim3 grid((unsigned) (H / 32), (unsigned) cap_groups);
        down_qpn8_kernel<4, 0><<<grid, 4 * 32, 0, cs>>>(grp_ptr, rep_off, grp_start, n_groups, ent_dst, h_q8_0,
                                                       x_scales != nullptr ? h_scales : nullptr, dn_afrag, out);
        check("moe_grouped_s2_qpn8/down");
    }
}
#else   // AMD, or a build without STRATA_EXPERIMENTAL_SM60: no m8n8k4 path (the DP4A kernels keep every default)
bool s2_qpn8_active() { return false; }
int64_t s2_qpn8_blob_bytes() { return (int64_t) 1382400; }   // the canonical Q2_0 blob, for the dual-form decision
int64_t s2_qpn8_slot_bytes(int64_t blob) { return blob; }
void s2_qpn8_repack_blob(uint8_t*, const uint8_t*, int64_t, void*) {}
void moe_grouped_s2_qpn8(const unsigned long long* grp_ptr, const int32_t* grp_start, const int32_t* n_groups,
                         const int32_t* ent_dst, const int32_t* ent_tok, int64_t cap_groups, int64_t cap_entries,
                         int64_t /*blob_bytes*/, const uint8_t* x_q8_0, const float* x_scales, void* scratch,
                         float* out, void* stream) {
    moe_grouped_s2(grp_ptr, grp_start, n_groups, ent_dst, ent_tok, cap_groups, cap_entries, x_q8_0, x_scales,
                   scratch, out, stream);
}
#endif  // !__HIPCC__ && STRATA_EXPERIMENTAL_SM60

}  // namespace strata::kernels
