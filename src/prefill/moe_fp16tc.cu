// src/prefill/moe_fp16tc.cu - see include/strata/prefill/moe_fp16tc.hpp (sm_70 FP16 tensor-core routed experts).
//
// Each block computes 64 routed rows by 128 output columns. Its eight warps each own a 32 by 32 tile.
// The reduction advances by 64 values, which is one native Q2_0 weight block. WMMA m16n16k16 uses Volta's
// FP16 tensor cores with FP32 accumulation; the public WMMA interface avoids dependence on fragment lane maps.
//
// Native Q2_0 weights have a half scale and two-bit codes. Half2 arithmetic reconstructs (q - 1) * scale.
// For q8_1 activations, PRMT reconstructs the signed integer codes exactly, then half2 arithmetic multiplies
// them by the activation scale rounded to FP16. This is not the same rounding as a float multiply followed by
// one FP16 conversion, and it is not bit-identical to MMQ. The numerical regression compares both paths with
// a double-precision reference.
//
// The shared operand strides are multiples of eight halves, as WMMA requires. The same storage holds the
// accumulator tiles after the last reduction step; guarded stores exclude the next expert's rows.
#include "strata/prefill/moe_fp16tc.hpp"

#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <mma.h>

#include <cstdio>
#include <cstdlib>

namespace strata::prefill::fp16tc {
namespace {

void ck(cudaError_t e, const char* what) {
    if (e != cudaSuccess) {
        std::fprintf(stderr, "prefill fp16tc: %s: %s\n", what, cudaGetErrorString(e));
        std::exit(1);
    }
}

constexpr int BM = 64;             // routed rows per block (2 warp row-halves of 32)
constexpr int BN = 128;            // output columns per block (4 warp column-quarters of 32)
constexpr int KS = 64;             // k per step: one Q2_0 weight block
constexpr int TS = 256;            // threads (8 warps: 2 row-halves x 4 column-quarters)
constexpr int AWMMA = 80;          // the wmma path needs a leading dimension that is a multiple of 8 halves
constexpr int WWMMA = 72;
constexpr int QB = 18;             // Q2_0 block bytes: fp16 scale + 16 code bytes
constexpr int ACTB = 144;          // block_q8_1_mmq (D4): 4 float scales + 128 int8
constexpr int Q2 = 42;             // GGML_TYPE_Q2_0

// Bit pattern -> half2 without going through the float pipe (the bits are 1024 + q in each half).
__device__ __forceinline__ __half2 h2_from_u32(uint32_t u) {
    __half2 h;
    __builtin_memcpy(&h, &u, 4);
    return h;
}

// Weight row r of the gathered matrices: GU rows are gate 0..n_ff-1 then up (at `up_off`); down rows are the down
// matrix, which the caller passes separately.
template <bool GU>
__device__ __forceinline__ const uint8_t* wbase(const Batch& b, const Geom& g, int z, int r) {
    if constexpr (GU) {
        return (r < g.n_ff) ? b.blob[z] + (size_t) r * g.gu_row
                            : b.blob[z] + g.up_off + (size_t) (r - g.n_ff) * g.gu_row;
    } else {
        return b.down[z] + (size_t) r * g.d_row;
    }
}

// NCOL weight rows (one Q2_0 block, 64 codes, each) into w_s as FP16.  256 threads cover NCOL rows x 4 quarters;
// a thread takes one row's `64/TPR` code bytes and writes the same count of half2 pairs.
template <bool GU, int WS, int NCOL>
__device__ __forceinline__ void dequant_weights(const Batch& b, const Geom& g, int z, int out_base, int k0, int tid,
                                                __half* w_s) {
    constexpr int TPR = TS / NCOL;                 // threads per row (2 at 128 cols, 4 at 64)
    constexpr int BYTES = 64 / TPR / 4;            // code bytes per thread
    constexpr int NWORD = BYTES / 4;
    constexpr int NPAIR = BYTES * 4 / 2;
    const int col = tid / TPR, sub = tid % TPR;
    const uint8_t* blk = wbase<GU>(b, g, z, out_base + col) + (size_t) (k0 / KS) * QB;
    const __half dh = *reinterpret_cast<const __half*>(blk);
    const uint8_t* qb = blk + 2 + sub * BYTES;
    const __half2 d2 = __half2half2(dh);
    const __half2 k1025 = __float2half2_rn(1025.0f);
    __half2* o = reinterpret_cast<__half2*>(w_s + (size_t) col * WS + sub * BYTES * 4);
    uint32_t words[NWORD];
#pragma unroll
    for (int w = 0; w < NWORD; ++w) {
        // Q2_0's 18-byte blocks align codes to two bytes, not necessarily four.
        const auto* q = reinterpret_cast<const uint16_t*>(qb + 4 * w);
        words[w] = (uint32_t) q[0] | ((uint32_t) q[1] << 16);
    }
#pragma unroll
    for (int k = 0; k < NPAIR; ++k) {   // pair k: codes 2k (bits 4k) and 2k+1 (bits 4k+2)
        const uint32_t bytes = words[k >> 3];
        const uint32_t t = (bytes >> (4 * (k & 7))) & 0xFu;
        const uint32_t p = (t & 3u) | ((t & 0xCu) << 14);
        o[k] = __hmul2(__hsub2(h2_from_u32(0x64006400u | p), k1025), d2);
    }
}

// BM activation rows of the layer's q8_1, 16 values per thread, as FP16.  `act_row_base` shifts the row the q8_1 is
// read from (0 for gate/up, Batch::down_act_row_base for down); the zero fill past the expert's last row keeps the
// mma from reading the next expert's activations.
template <int AS>
__device__ __forceinline__ void dequant_act(int row0, int local, int k0, int tid, const void* act, int64_t act_rows,
                                            int64_t act_row_base, __half* a_s) {
    const int m = tid >> 2, part = tid & 3;
    __half* o = a_s + (size_t) m * AS + part * 16;
    if (m >= local) {   // past the expert's rows: zero, so the mma never sees the next expert's activations
#pragma unroll
        for (int j = 0; j < 16; ++j) o[j] = __float2half(0.0f);
    } else {
        const int64_t row = (int64_t) row0 + m + act_row_base;
        const int64_t kb = k0 / 128;
        const int off = (k0 % 128) + part * 16;
        const uint8_t* base = (const uint8_t*) act + ((size_t) kb * (size_t) act_rows + (size_t) row) * ACTB;
        const float d = ((const float*) base)[off >> 5];
        const uint32_t* qs = reinterpret_cast<const uint32_t*>(base + 16 + off);
        const __half2 d2 = __float2half2_rn(d);
        const __half2 c1152 = h2_from_u32(0x64806480u);   // 1152.0h: the offset baked into the mantissa bias
        __half2* o2 = reinterpret_cast<__half2*>(o);
#pragma unroll
        for (int k = 0; k < 4; ++k) {
            // Sign-flip every code byte, then PRMT each pair into {0x64, byte} halves = 1024 + (byte ^ 128) in
            // [1024, 1279].  Subtracting 1152 leaves the signed code exactly: byte < 128 -> +byte (the XOR added
            // 128), byte >= 128 -> byte - 256.  Both operands and the difference are exact in FP16.
            const uint32_t v = qs[k] ^ 0x80808080u;
            o2[2 * k] = __hmul2(__hsub2(h2_from_u32(__byte_perm(v, 0x64646464u, 0x4140)), c1152), d2);
            o2[2 * k + 1] = __hmul2(__hsub2(h2_from_u32(__byte_perm(v, 0x64646464u, 0x4342)), c1152), d2);
        }
    }
}

// Each warp owns two row tiles and two column tiles. Stage the accumulators through shared memory because
// WMMA does not specify the fragment element map. Partial expert rows are excluded by the final store.
template <bool GU>
__global__ void __launch_bounds__(TS, 1)
expert_kernel_wmma(const Batch b, const Geom g, const int32_t* __restrict__ bounds, const void* __restrict__ act,
                   int64_t act_rows, float* __restrict__ dst, int64_t ld_dst, int64_t dst_row_base, int k_reduce) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 700
    using namespace nvcuda;
    const int z = blockIdx.z;
    const int lo = bounds[z], hi = bounds[z + 1];
    const int row0 = lo + (int) blockIdx.y * BM;
    if (row0 >= hi) return;
    const int local = (hi - row0 < BM) ? (hi - row0) : BM;
    const int out_base = (int) blockIdx.x * BN;
    const int tid = threadIdx.x, warp = tid >> 5;
    const int rh = warp >> 2, cq = warp & 3;
    const int m_base = rh * 32, n_base = cq * 32;
    const int64_t dst_shift = dst_row_base + (GU ? b.gu_dst_row_base : 0);
    const int64_t act_shift = GU ? 0 : b.down_act_row_base;

    __shared__ __align__(32) union {
        struct { __half a[BM * AWMMA]; __half w[BN * WWMMA]; } ab;
        float c[BM * BN];
    } sm;

    wmma::fragment<wmma::accumulator, 16, 16, 16, float> acc[2][2];
#pragma unroll
    for (int r = 0; r < 2; ++r)
#pragma unroll
        for (int c = 0; c < 2; ++c) wmma::fill_fragment(acc[r][c], 0.0f);

    for (int k0 = 0; k0 < k_reduce; k0 += KS) {
        __syncthreads();
        dequant_weights<GU, WWMMA, BN>(b, g, z, out_base, k0, tid, sm.ab.w);
        dequant_act<AWMMA>(row0, local, k0, tid, act, act_rows, act_shift, sm.ab.a);
        __syncthreads();
#pragma unroll
        for (int k16 = 0; k16 < KS / 16; ++k16) {
            wmma::fragment<wmma::matrix_a, 16, 16, 16, __half, wmma::row_major> af[2];
            wmma::fragment<wmma::matrix_b, 16, 16, 16, __half, wmma::col_major> bf[2];
#pragma unroll
            for (int r = 0; r < 2; ++r)
                wmma::load_matrix_sync(af[r], &sm.ab.a[(size_t) (m_base + 16 * r) * AWMMA + k16 * 16], AWMMA);
#pragma unroll
            for (int c = 0; c < 2; ++c)
                wmma::load_matrix_sync(bf[c], &sm.ab.w[(size_t) (n_base + 16 * c) * WWMMA + k16 * 16], WWMMA);
#pragma unroll
            for (int r = 0; r < 2; ++r)
#pragma unroll
                for (int c = 0; c < 2; ++c) wmma::mma_sync(acc[r][c], af[r], bf[c], acc[r][c]);
        }
    }
    __syncthreads();   // the tiles are dead: the same storage becomes the accumulator staging
#pragma unroll
    for (int r = 0; r < 2; ++r)
#pragma unroll
        for (int c = 0; c < 2; ++c)
            wmma::store_matrix_sync(&sm.c[(size_t) (m_base + 16 * r) * BN + n_base + 16 * c], acc[r][c], BN,
                                    wmma::mem_row_major);
    __syncthreads();
    for (int i = tid; i < BM * BN; i += TS) {
        const int grow = row0 + (i >> 7);
        if (grow < hi) dst[(dst_shift + grow) * ld_dst + out_base + (i & 127)] = sm.c[i];
    }
#endif
}

template <bool GU>
void launch(const Batch& b, const Geom& g, const int32_t* bounds, const void* act, int64_t act_rows, float* dst,
            int64_t ld_dst, int64_t dst_row_base, void* stream) {
    if (b.n <= 0 || b.max_rows <= 0) return;
    const int k_reduce = GU ? g.n_embd : g.n_ff;
    const int out_total = GU ? 2 * g.n_ff : g.n_embd;
    const dim3 grid(out_total / BN, (b.max_rows + BM - 1) / BM, b.n);
    const cudaStream_t s = (cudaStream_t) stream;
    expert_kernel_wmma<GU><<<grid, TS, 0, s>>>(b, g, bounds, act, act_rows, dst, ld_dst, dst_row_base, k_reduce);
    ck(cudaGetLastError(), "expert_kernel_wmma");
}

}  // namespace

bool built() { return true; }

bool available() {
    int dev = 0;
    if (cudaGetDevice(&dev) != cudaSuccess) return false;
    int major = 0, minor = 0;
    if (cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, dev) != cudaSuccess) return false;
    if (cudaDeviceGetAttribute(&minor, cudaDevAttrComputeCapabilityMinor, dev) != cudaSuccess) return false;
    return major == 7 && minor == 0;   // retain the existing MMQ dispatch on other architectures
}

bool geom_ok(int gu_type, int d_type, int64_t n_embd, int64_t n_ff, size_t gu_row, size_t d_row, size_t up_off,
             size_t down_off) {
    if (gu_type != Q2 || d_type != Q2) return false;
    if (n_embd <= 0 || n_ff <= 0) return false;
    // the k steps and the down reduction are whole 64-blocks; the block's BN = 128 output columns are whole too
    // (down writes n_embd columns, gate/up write 2 n_ff)
    if (n_embd % 128 != 0 || n_ff % 64 != 0) return false;
    if (gu_row != (size_t) (n_embd / 64) * QB || d_row != (size_t) (n_ff / 64) * QB) return false;
    if (up_off != gu_row * (size_t) n_ff) return false;
    if (down_off != 2 * up_off) return false;
    return true;
}

void gu(const Batch& b, const Geom& g, const int32_t* bounds, const void* xq, int64_t xq_rows, float* dst,
        int64_t ld_dst, void* stream) {
    launch<true>(b, g, bounds, xq, xq_rows, dst, ld_dst, 0, stream);
}

void down(const Batch& b, const Geom& g, const int32_t* bounds, const void* hq, int64_t hq_rows, float* dst,
          int64_t ld_dst, int64_t dst_row_base, void* stream) {
    launch<false>(b, g, bounds, hq, hq_rows, dst, ld_dst, dst_row_base, stream);
}

}  // namespace strata::prefill::fp16tc
