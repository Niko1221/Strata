// Polymorphic dequantization + cuBLAS Tensor Core prefill for strata-q38.
#include "prefill_gemm.cuh"
#include "spin8_triality.cuh"
#include "prefill_wmma.cuh"
#include "ops.cuh"
#include "iq_tables.hpp"
#include "iq_decode.hpp"
#include <cublas_v2.h>
#include <cublasLt.h>
#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <stdexcept>
#include <string>
#include <vector>

namespace strata::qwen38 {
namespace {

inline void ck_cublas(cublasStatus_t st) {
    if (st != CUBLAS_STATUS_SUCCESS) {
        throw std::runtime_error("cuBLAS error: " + std::to_string(static_cast<int>(st)));
    }
}

inline bool fp8_prefill_requested() {
    const char* fp8_env = std::getenv("STRATA_FP8_PREFILL");
    return (fp8_env && fp8_env[0] == '1');
}

inline bool fp32_prefill_requested() {
    const char* env = std::getenv("STRATA_Q38_FP32_PREFILL");
    return (env && env[0] == '1');
}

// Policy-B opt-in: use the tiled FP16 dequant + cuBLAS GEMM path even for
// decode-sized batches (n_tokens < 4). This changes FP32 reduction order vs
// the packed DP4A matvec, so it is off by default and only enabled for the
// performance-first comparison. The dequant tiles and GEMM below are the
// same already-shipped prefill implementation.
inline bool fp16_gemv_requested() {
    const char* env = std::getenv("STRATA_Q38_FP16_GEMV");
    return (env && env[0] == '1');
}

__device__ __forceinline__ float load_fp16_le(const uint8_t* p) {
    return __half2float(__ushort_as_half(static_cast<uint16_t>(p[0]) | (static_cast<uint16_t>(p[1]) << 8)));
}

__device__ __forceinline__ uint32_t load_u32_2b(const uint8_t* p) {
    const uint16_t lo = __ldg(reinterpret_cast<const uint16_t*>(p));
    const uint16_t hi = __ldg(reinterpret_cast<const uint16_t*>(p + 2));
    return static_cast<uint32_t>(lo) | (static_cast<uint32_t>(hi) << 16);
}

__device__ __forceinline__ uint32_t pack_f2_to_u32(float a, float b) {
    const uint16_t lo = __half_as_ushort(__float2half_rn(a));
    const uint16_t hi = __half_as_ushort(__float2half_rn(b));
    return static_cast<uint32_t>(lo) | (static_cast<uint32_t>(hi) << 16);
}

__device__ __forceinline__ uint4 pack_f8_to_u4(float v0, float v1, float v2, float v3,
                                               float v4, float v5, float v6, float v7) {
    return make_uint4(pack_f2_to_u32(v0, v1), pack_f2_to_u32(v2, v3),
                      pack_f2_to_u32(v4, v5), pack_f2_to_u32(v6, v7));
}

// E2M1 value: sign bit + 3-bit magnitude code -> {0, .5, 1, 1.5, 2, 3, 4, 6}.
__device__ __forceinline__ float e2m1_value(uint8_t nibble) {
    constexpr float kMag[8] = {0.f, 0.5f, 1.f, 1.5f, 2.f, 3.f, 4.f, 6.f};
    const float v = kMag[nibble & 7u];
    return (nibble & 8u) ? -v : v;
}
// Unsigned E4M3 scale (exponent bias 7), used by NVFP4 sub-blocks.
__device__ __forceinline__ float ue4m3_value(uint8_t b) {
    const int e = (b >> 3) & 15, m = b & 7;
    if (e == 0) return static_cast<float>(m) / 8.f * (1.f / 64.f);
    return (1.f + static_cast<float>(m) / 8.f) * __int_as_float((e + 120) << 23); // 2^(e-7)
}

// GGML Q3_K (type 11): 256 values / 110 bytes. Layout: fp16 d, 32 B high mask,
// 64 B of 2-bit codes, 12 B of 4-bit sub-block scales. Decoding follows the
// reference dequantize_row_q3_K in src/quant/dequant.cpp exactly.
__device__ __forceinline__ float dequant_q3k_elem(const uint8_t* p, int i) {
    const float d_all = load_fp16_le(p);
    const uint8_t* q = p + 2;
    const uint8_t* hm = p + 34;
    const uint8_t* sr = p + 98;
    const int n = i / 128, local = i % 128;
    const int j = local / 32, half = (local % 32) / 16, l = local % 16;
    const int qidx = n * 32 + l + half * 16;
    const uint8_t qb = q[qidx];
    const int code = static_cast<int>((qb >> (2 * j)) & 3);
    const uint8_t mbit = static_cast<uint8_t>(1u << (4 * n + j));
    const int val = code - ((hm[qidx] & mbit) ? 0 : 4);

    uint32_t aux[4];
    #pragma unroll
    for (int k = 0; k < 4; ++k) {
        aux[k] = static_cast<uint32_t>(sr[4 * k]) | (static_cast<uint32_t>(sr[4 * k + 1]) << 8) |
                 (static_cast<uint32_t>(sr[4 * k + 2]) << 16) | (static_cast<uint32_t>(sr[4 * k + 3]) << 24);
    }
    const uint32_t tmp = aux[2];
    aux[2] = ((aux[0] >> 4) & 0x0f0f0f0fu) | (((tmp >> 4) & 0x03030303u) << 4);
    aux[3] = ((aux[1] >> 4) & 0x0f0f0f0fu) | (((tmp >> 6) & 0x03030303u) << 4);
    aux[0] = (aux[0] & 0x0f0f0f0fu) | (((tmp >> 0) & 0x03030303u) << 4);
    aux[1] = (aux[1] & 0x0f0f0f0fu) | (((tmp >> 2) & 0x03030303u) << 4);
    const int sc = static_cast<int8_t>(reinterpret_cast<const uint8_t*>(aux)[n * 8 + j * 2 + half]);
    return d_all * static_cast<float>(sc - 32) * static_cast<float>(val);
}

__device__ __forceinline__ float dequant_elem(int type, const uint8_t* p, int i) {
    if (type == 0)  return reinterpret_cast<const float*>(p)[i];
    if (type == 1)  return __half2float(reinterpret_cast<const half*>(p)[i]);
    if (type == 30) return __uint_as_float(static_cast<uint32_t>(reinterpret_cast<const uint16_t*>(p)[i]) << 16);
    if (type == 11) {
        p += static_cast<size_t>(i / 256) * 110;
        return dequant_q3k_elem(p, i % 256);
    }
    if (type == 39 || type == 40) {
        // MXFP4: 32 values / 17 B = 1 E8M0 scale + 16 B of E2M1 codes.
        // NVFP4: 64 values / 36 B = 4 UE4M3 sub-scales (16 values each) + 32 B codes.
        const int blk = (type == 39) ? 32 : 64;
        const int bytes = (type == 39) ? 17 : 36;
        const size_t offset = static_cast<size_t>(i / blk) * bytes;
        const int j = i % blk;
        float scale;
        if (type == 39) {
            const int e = p[offset];
            scale = (e > 0 && e < 255) ? __int_as_float(e << 23) : 1.f; // 2^(e-127)
        } else {
            scale = ue4m3_value(p[offset + j / 16]);
        }
        const uint8_t packed = p[offset + ((type == 39) ? 1 : 4) + j / 2];
        const uint8_t nibble = (j & 1) ? (packed >> 4) : (packed & 15);
        return scale * e2m1_value(nibble);
    }
    if (type == 142) {
        p += static_cast<size_t>(i / 128) * 34;
        i %= 128;
        return load_fp16_le(p) * static_cast<float>(static_cast<int>((p[2 + i / 4] >> (2 * (i % 4))) & 3) - 1);
    }
    if (type == 8) {
        p += static_cast<size_t>(i / 32) * 34;
        i %= 32;
        return load_fp16_le(p) * static_cast<float>(reinterpret_cast<const int8_t*>(p + 2)[i]);
    }
    if (type == 21 || type == 22) {
        p += static_cast<size_t>(i / 256) * (type == 21 ? 110 : 82);
        i %= 256;
        return iq_value(type, p, i, load_fp16_le(p), q38_iq2s_grid, q38_iq3s_grid);
    }
    if (type == 20) {
        // IQ4_NL: 32 values / 18 B = fp16 scale + 16 B of packed nibbles.
        p += static_cast<size_t>(i / 32) * 18;
        i %= 32;
        constexpr int kNL[16] = {-127,-104,-83,-65,-49,-35,-22,-10,1,13,25,38,53,69,89,113};
        const int code = (i < 16) ? (p[2 + i] & 15) : (p[2 + i - 16] >> 4);
        return load_fp16_le(p) * static_cast<float>(kNL[code]);
    }
    if (type == 23) {
        p += static_cast<size_t>(i / 256) * 136;
        i %= 256;
        const int b = i / 32, j = i % 32;
        const int scale = ((p[4 + b / 2] >> (4 * (b % 2))) & 15) |
                          ((((static_cast<uint16_t>(p[2]) | (static_cast<uint16_t>(p[3]) << 8)) >> (2 * b)) & 3) << 4);
        constexpr int kTable[16] = {-127,-104,-83,-65,-49,-35,-22,-10,1,13,25,38,53,69,89,113};
        const int code = (p[8 + 16 * b + j % 16] >> (j >= 16 ? 4 : 0)) & 15;
        return load_fp16_le(p) * static_cast<float>(scale - 32) * static_cast<float>(kTable[code]);
    }
    if (type == 12 || type == 13) {
        const bool five = (type == 13);
        p += static_cast<size_t>(i / 256) * (five ? 176 : 144);
        i %= 256;
        const int b = i / 32, j = i % 32;
        const uint8_t* sc = p + 4;
        int d = 0, m = 0;
        if (b < 4) { d = sc[b] & 63; m = sc[b + 4] & 63; }
        else { d = (sc[b + 4] & 15) | ((sc[b - 4] >> 6) << 4); m = (sc[b + 4] >> 4) | ((sc[b] >> 6) << 4); }
        const uint8_t* qs = p + (five ? 48 : 16);
        int code = (qs[(b / 2) * 32 + j] >> (4 * (b % 2))) & 15;
        if (five) code |= ((p[16 + j] >> b) & 1) << 4;
        return load_fp16_le(p) * static_cast<float>(d * code) - load_fp16_le(p + 2) * static_cast<float>(m);
    }
    // Q6_K (type == 14)
    p += static_cast<size_t>(i / 256) * 210;
    i %= 256;
    const int ip = i / 128, j = i % 32, group = (i % 128) / 32;
    const int lo = (p[ip * 64 + (group % 2) * 32 + j] >> (group >= 2 ? 4 : 0)) & 15;
    const int hi = (p[128 + ip * 32 + j] >> (2 * group)) & 3;
    const int scale = reinterpret_cast<const int8_t*>(p + 192)[ip * 8 + group * 2 + j / 16];
    return load_fp16_le(p + 208) * static_cast<float>(scale) * static_cast<float>((lo | (hi << 4)) - 32);
}

// Generic scalar-per-thread fallback for F32/F16/BF16/IQ4_XS/PQ2_0 and non-256-aligned K.
template <int TYPE>
__global__ void __launch_bounds__(256)
dequant_tile_kernel(const uint8_t* __restrict__ src, half* __restrict__ dst, int tile_rows, int K, size_t row_stride) {
    const int row = blockIdx.y;
    const int col = blockIdx.x * 256 + threadIdx.x;
    if (row >= tile_rows) return;
    const uint8_t* row_ptr = src + static_cast<size_t>(row) * row_stride;
    if constexpr (TYPE == 12 || TYPE == 13 || TYPE == 14 || TYPE == 21 || TYPE == 22) {
        constexpr int kBlkBytes = (TYPE == 12) ? 144 : (TYPE == 13) ? 176 : (TYPE == 14) ? 210 : (TYPE == 21) ? 110 : 82;
        __shared__ alignas(2) uint8_t sb[kBlkBytes];
        const uint8_t* bp = row_ptr + static_cast<size_t>(blockIdx.x) * kBlkBytes;
        if (threadIdx.x < kBlkBytes) sb[threadIdx.x] = __ldg(bp + threadIdx.x);
        __syncthreads();
        if (col < K) {
            const float val = (TYPE == 21 || TYPE == 22)
                ? iq_value(TYPE, sb, threadIdx.x, load_fp16_le(sb), q38_iq2s_grid, q38_iq3s_grid)
                : dequant_elem(TYPE, sb, threadIdx.x);
            dst[static_cast<size_t>(row) * K + col] = __float2half_rn(val);
        }
    } else if (col < K) {
        dst[static_cast<size_t>(row) * K + col] = __float2half_rn(dequant_elem(TYPE, row_ptr, col));
    }
}

__device__ __forceinline__ float iq2s_lane_elem(float factor, uint64_t grid, uint8_t signs, int k) {
    return (factor * static_cast<float>((grid >> (8 * k)) & 255u)) * ((signs & (1u << k)) ? -1.0f : 1.0f);
}

// Fast 1-warp-per-256-elem-block IQ2_S (type 22): 8 warps = 8 rows/CTA, 8 KB shared LUT, 128-bit store.
__global__ void __launch_bounds__(256)
dequant_iq2s_256_kernel(const uint8_t* __restrict__ src, half* __restrict__ dst, int tile_rows, int K, size_t row_stride) {
    __shared__ uint64_t sgrid[1024];
    for (int i = threadIdx.x; i < 1024; i += 256) sgrid[i] = __ldg(&q38_iq2s_grid[i]);
    __syncthreads();
    const int row = blockIdx.y * 8 + (threadIdx.x >> 5);
    if (row >= tile_rows) return;
    const int lane = threadIdx.x & 31, sub = lane >> 2, group = lane & 3;
    const uint8_t* p = src + static_cast<size_t>(row) * row_stride + static_cast<size_t>(blockIdx.x) * 82;
    const float d = __half2float(__ushort_as_half(__ldg(reinterpret_cast<const uint16_t*>(p))));
    const int index = static_cast<int>(__ldg(p + 2 + lane)) |
                      ((static_cast<int>(__ldg(p + 66 + sub) >> (2 * group)) & 3) << 8);
    const uint64_t grid = sgrid[index];
    const uint8_t signs = __ldg(p + 34 + lane);
    const int scale = (__ldg(p + 74 + sub) >> (4 * (group >> 1))) & 15;
    const float factor = d * (0.5f + static_cast<float>(scale)) * 0.25f;
    *reinterpret_cast<uint4*>(dst + static_cast<size_t>(row) * K + blockIdx.x * 256 + lane * 8) = pack_f8_to_u4(
        iq2s_lane_elem(factor, grid, signs, 0), iq2s_lane_elem(factor, grid, signs, 1),
        iq2s_lane_elem(factor, grid, signs, 2), iq2s_lane_elem(factor, grid, signs, 3),
        iq2s_lane_elem(factor, grid, signs, 4), iq2s_lane_elem(factor, grid, signs, 5),
        iq2s_lane_elem(factor, grid, signs, 6), iq2s_lane_elem(factor, grid, signs, 7));
}

__device__ __forceinline__ float iq3s_lane_elem(float factor, uint32_t grid, uint8_t signs, int sub_k, int sign_k) {
    return (factor * static_cast<float>((grid >> (8 * sub_k)) & 255u)) * ((signs & (1u << sign_k)) ? -1.0f : 1.0f);
}

// Fast 1-warp-per-256-elem-block IQ3_S (type 21): 8 warps = 8 rows/CTA, 2 KB shared LUT, 128-bit store.
__global__ void __launch_bounds__(256)
dequant_iq3s_256_kernel(const uint8_t* __restrict__ src, half* __restrict__ dst, int tile_rows, int K, size_t row_stride) {
    __shared__ uint32_t sgrid3[512];
    for (int i = threadIdx.x; i < 512; i += 256) sgrid3[i] = __ldg(&q38_iq3s_grid[i]);
    __syncthreads();
    const int row = blockIdx.y * 8 + (threadIdx.x >> 5);
    if (row >= tile_rows) return;
    const int lane = threadIdx.x & 31, sub = lane >> 2, group = lane & 3;
    const int idx0 = group * 2, idx1 = idx0 + 1;
    const uint8_t* p = src + static_cast<size_t>(row) * row_stride + static_cast<size_t>(blockIdx.x) * 110;
    const float d = __half2float(__ushort_as_half(__ldg(reinterpret_cast<const uint16_t*>(p))));
    const uint16_t qs2 = __ldg(reinterpret_cast<const uint16_t*>(p + 2) + lane);
    const uint8_t qh_b = __ldg(p + 66 + sub);
    const uint32_t g0 = sgrid3[static_cast<int>(qs2 & 0xFFu) | ((static_cast<int>(qh_b >> idx0) & 1) << 8)];
    const uint32_t g1 = sgrid3[static_cast<int>(qs2 >> 8) | ((static_cast<int>(qh_b >> idx1) & 1) << 8)];
    const uint8_t signs = __ldg(p + 74 + lane);
    const int scale = (__ldg(p + 106 + (sub >> 1)) >> (4 * (sub & 1))) & 15;
    const float factor = d * static_cast<float>(1 + 2 * scale);
    *reinterpret_cast<uint4*>(dst + static_cast<size_t>(row) * K + blockIdx.x * 256 + lane * 8) = pack_f8_to_u4(
        iq3s_lane_elem(factor, g0, signs, 0, 0), iq3s_lane_elem(factor, g0, signs, 1, 1),
        iq3s_lane_elem(factor, g0, signs, 2, 2), iq3s_lane_elem(factor, g0, signs, 3, 3),
        iq3s_lane_elem(factor, g1, signs, 0, 4), iq3s_lane_elem(factor, g1, signs, 1, 5),
        iq3s_lane_elem(factor, g1, signs, 2, 6), iq3s_lane_elem(factor, g1, signs, 3, 7));
}

template <bool FIVE>
__device__ __forceinline__ float qk45_lane_elem(float d0, int d, float m_val, uint32_t qw, uint32_t qhw,
                                                int shift, int b, int k) {
    int code = static_cast<int>((qw >> (8 * k + shift)) & 15u);
    if constexpr (FIVE) code |= static_cast<int>(((qhw >> (8 * k + b)) & 1u) << 4);
    return d0 * static_cast<float>(d * code) - m_val;
}

// Fast 1-warp-per-256-elem-block Q4_K (type 12) & Q5_K (type 13): 8 warps = 8 rows/CTA, 128-bit store.
template <bool FIVE>
__global__ void __launch_bounds__(256)
dequant_qk45_256_kernel(const uint8_t* __restrict__ src, half* __restrict__ dst, int tile_rows, int K, size_t row_stride) {
    const int row = blockIdx.y * 8 + (threadIdx.x >> 5);
    if (row >= tile_rows) return;
    const int lane = threadIdx.x & 31, b = lane >> 2, j0 = (lane & 3) * 8;
    constexpr int kBlkBytes = FIVE ? 176 : 144;
    const uint8_t* p = src + static_cast<size_t>(row) * row_stride + static_cast<size_t>(blockIdx.x) * kBlkBytes;
    const float d0 = __half2float(__ushort_as_half(__ldg(reinterpret_cast<const uint16_t*>(p))));
    const float dmin0 = __half2float(__ushort_as_half(__ldg(reinterpret_cast<const uint16_t*>(p + 2))));
    const uint8_t* sc = p + 4;
    int d = 0, m = 0;
    if (b < 4) { d = __ldg(sc + b) & 63; m = __ldg(sc + b + 4) & 63; }
    else { d = (__ldg(sc + b + 4) & 15) | ((__ldg(sc + b - 4) >> 6) << 4); m = (__ldg(sc + b + 4) >> 4) | ((__ldg(sc + b) >> 6) << 4); }
    const float m_val = dmin0 * static_cast<float>(m);
    const uint2 q8 = __ldg(reinterpret_cast<const uint2*>(p + (FIVE ? 48 : 16) + (b >> 1) * 32 + j0));
    uint2 qh8 = make_uint2(0u, 0u);
    if constexpr (FIVE) qh8 = __ldg(reinterpret_cast<const uint2*>(p + 16 + j0));
    const int shift = 4 * (b & 1);
    *reinterpret_cast<uint4*>(dst + static_cast<size_t>(row) * K + blockIdx.x * 256 + lane * 8) = pack_f8_to_u4(
        qk45_lane_elem<FIVE>(d0, d, m_val, q8.x, qh8.x, shift, b, 0),
        qk45_lane_elem<FIVE>(d0, d, m_val, q8.x, qh8.x, shift, b, 1),
        qk45_lane_elem<FIVE>(d0, d, m_val, q8.x, qh8.x, shift, b, 2),
        qk45_lane_elem<FIVE>(d0, d, m_val, q8.x, qh8.x, shift, b, 3),
        qk45_lane_elem<FIVE>(d0, d, m_val, q8.y, qh8.y, shift, b, 0),
        qk45_lane_elem<FIVE>(d0, d, m_val, q8.y, qh8.y, shift, b, 1),
        qk45_lane_elem<FIVE>(d0, d, m_val, q8.y, qh8.y, shift, b, 2),
        qk45_lane_elem<FIVE>(d0, d, m_val, q8.y, qh8.y, shift, b, 3));
}

__device__ __forceinline__ float q6k_lane_elem(float d_sc, uint32_t qlw, uint32_t qhw, int ql_shift, int qh_shift, int k) {
    const int lo = static_cast<int>((qlw >> (8 * k + ql_shift)) & 15u);
    const int hi = static_cast<int>((qhw >> (8 * k + qh_shift)) & 3u);
    return d_sc * static_cast<float>((lo | (hi << 4)) - 32);
}

// Fast 1-warp-per-256-elem-block Q6_K (type 14): 8 warps = 8 rows/CTA, 2B-safe loads, 128-bit store.
__global__ void __launch_bounds__(256)
dequant_q6k_256_kernel(const uint8_t* __restrict__ src, half* __restrict__ dst, int tile_rows, int K, size_t row_stride) {
    const int row = blockIdx.y * 8 + (threadIdx.x >> 5);
    if (row >= tile_rows) return;
    const int lane = threadIdx.x & 31, ip = lane >> 4, group = (lane >> 2) & 3, j0 = (lane & 3) * 8;
    const uint8_t* p = src + static_cast<size_t>(row) * row_stride + static_cast<size_t>(blockIdx.x) * 210;
    const float d = __half2float(__ushort_as_half(__ldg(reinterpret_cast<const uint16_t*>(p + 208))));
    const int scale = static_cast<int>(__ldg(reinterpret_cast<const int8_t*>(p + 192) + ip * 8 + group * 2 + ((lane >> 1) & 1)));
    const float d_sc = d * static_cast<float>(scale);
    const uint8_t* ql_ptr = p + ip * 64 + (group & 1) * 32 + j0;
    const uint8_t* qh_ptr = p + 128 + ip * 32 + j0;
    const uint32_t ql0 = load_u32_2b(ql_ptr), ql1 = load_u32_2b(ql_ptr + 4);
    const uint32_t qh0 = load_u32_2b(qh_ptr), qh1 = load_u32_2b(qh_ptr + 4);
    const int ql_shift = (group >= 2) ? 4 : 0, qh_shift = 2 * group;
    *reinterpret_cast<uint4*>(dst + static_cast<size_t>(row) * K + blockIdx.x * 256 + lane * 8) = pack_f8_to_u4(
        q6k_lane_elem(d_sc, ql0, qh0, ql_shift, qh_shift, 0), q6k_lane_elem(d_sc, ql0, qh0, ql_shift, qh_shift, 1),
        q6k_lane_elem(d_sc, ql0, qh0, ql_shift, qh_shift, 2), q6k_lane_elem(d_sc, ql0, qh0, ql_shift, qh_shift, 3),
        q6k_lane_elem(d_sc, ql1, qh1, ql_shift, qh_shift, 0), q6k_lane_elem(d_sc, ql1, qh1, ql_shift, qh_shift, 1),
        q6k_lane_elem(d_sc, ql1, qh1, ql_shift, qh_shift, 2), q6k_lane_elem(d_sc, ql1, qh1, ql_shift, qh_shift, 3));
}

__device__ __forceinline__ float q8_0_lane_elem(float d, uint32_t qw, int k) {
    return d * static_cast<float>(static_cast<int8_t>((qw >> (8 * k)) & 0xFFu));
}

// Fast 8-blocks-per-warp (256 elems/warp) Q8_0 (type 8): 8 warps = 8 rows/CTA, 2B-safe loads, 128-bit store.
__global__ void __launch_bounds__(256)
dequant_q8_0_256_kernel(const uint8_t* __restrict__ src, half* __restrict__ dst, int tile_rows, int K, size_t row_stride) {
    const int row = blockIdx.y * 8 + (threadIdx.x >> 5);
    if (row >= tile_rows) return;
    const int lane = threadIdx.x & 31, sub = lane >> 2, j0 = (lane & 3) * 8;
    const uint8_t* bp = src + static_cast<size_t>(row) * row_stride + (static_cast<size_t>(blockIdx.x) * 8 + sub) * 34;
    const float d = __half2float(__ushort_as_half(__ldg(reinterpret_cast<const uint16_t*>(bp))));
    const uint32_t q0 = load_u32_2b(bp + 2 + j0), q1 = load_u32_2b(bp + 2 + j0 + 4);
    *reinterpret_cast<uint4*>(dst + static_cast<size_t>(row) * K + blockIdx.x * 256 + lane * 8) = pack_f8_to_u4(
        q8_0_lane_elem(d, q0, 0), q8_0_lane_elem(d, q0, 1), q8_0_lane_elem(d, q0, 2), q8_0_lane_elem(d, q0, 3),
        q8_0_lane_elem(d, q1, 0), q8_0_lane_elem(d, q1, 1), q8_0_lane_elem(d, q1, 2), q8_0_lane_elem(d, q1, 3));
}

__global__ void __launch_bounds__(256)
f32_to_fp16_kernel(const float* __restrict__ src, half* __restrict__ dst, size_t n) {
    const size_t i = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i < n) dst[i] = __float2half_rn(src[i]);
}

__device__ __forceinline__ void reduce_token_scale(float local_max, int local_bad, int K,
                                                   float* act_scales, int tok,
                                                   float& out_inv_sc, int& out_bad) {
    #pragma unroll
    for (int mask = 16; mask > 0; mask >>= 1) {
        local_max = fmaxf(local_max, __shfl_xor_sync(0xffffffffu, local_max, mask));
        local_bad |= __shfl_xor_sync(0xffffffffu, local_bad, mask);
    }
    __shared__ float warp_max[8];
    __shared__ int warp_bad[8];
    __shared__ float s_inv_sc;
    __shared__ int s_bad;
    if ((threadIdx.x & 31) == 0) {
        warp_max[threadIdx.x >> 5] = local_max;
        warp_bad[threadIdx.x >> 5] = local_bad;
    }
    __syncthreads();
    if (threadIdx.x == 0) {
        float gm = 0.0f;
        int bad = 0;
        #pragma unroll
        for (int w = 0; w < 8; ++w) { gm = fmaxf(gm, warp_max[w]); bad |= warp_bad[w]; }
        s_bad = bad;
        const float target_max = (K >= 16384) ? 4.0f : 8.0f;
        const float inv_target = (K >= 16384) ? 0.25f : 0.125f;
        if (bad) {
            s_inv_sc = 0.0f;
            act_scales[tok] = __int_as_float(0x7fffffffU);
        } else if (gm > 0.0f) {
            const float eff_gm = fmaxf(gm, 1.0e-35f);
            act_scales[tok] = eff_gm * inv_target;
            s_inv_sc = target_max / eff_gm;
        } else {
            act_scales[tok] = 0.0f;
            s_inv_sc = 0.0f;
        }
    }
    __syncthreads();
    out_inv_sc = s_inv_sc;
    out_bad = s_bad;
}

// Single-pass register-resident activation normalization for trunk widths K=5120 and K=6144 (reads DRAM once).
template <int K_FIXED>
__global__ void __launch_bounds__(256)
prepare_act_scaled_fp16_reg_kernel(const float* __restrict__ x_fp32, half* __restrict__ act_fp16,
                                   float* __restrict__ act_scales, int n_tokens) {
    constexpr int kVec4Count = K_FIXED / 1024;
    const int tok = blockIdx.x;
    if (tok >= n_tokens) return;
    const float4* x4 = reinterpret_cast<const float4*>(x_fp32 + static_cast<size_t>(tok) * K_FIXED);
    uint2* y2 = reinterpret_cast<uint2*>(act_fp16 + static_cast<size_t>(tok) * K_FIXED);

    float4 regs[kVec4Count];
    float local_max = 0.0f;
    int local_bad = 0;
    #pragma unroll
    for (int m = 0; m < kVec4Count; ++m) {
        const float4 v = __ldg(x4 + m * 256 + threadIdx.x);
        regs[m] = v;
        if (!isfinite(v.x) || !isfinite(v.y) || !isfinite(v.z) || !isfinite(v.w)) local_bad = 1;
        else local_max = fmaxf(local_max, fmaxf(fmaxf(fabsf(v.x), fabsf(v.y)), fmaxf(fabsf(v.z), fabsf(v.w))));
    }
    float inv_sc = 0.0f;
    int bad = 0;
    reduce_token_scale(local_max, local_bad, K_FIXED, act_scales, tok, inv_sc, bad);
    if (bad) {
        #pragma unroll
        for (int m = 0; m < kVec4Count; ++m) y2[m * 256 + threadIdx.x] = make_uint2(0u, 0u);
    } else {
        #pragma unroll
        for (int m = 0; m < kVec4Count; ++m) {
            const float4 v = regs[m];
            y2[m * 256 + threadIdx.x] = make_uint2(pack_f2_to_u32(v.x * inv_sc, v.y * inv_sc),
                                                    pack_f2_to_u32(v.z * inv_sc, v.w * inv_sc));
        }
    }
}

// Per-token normalized FP16 activation staging: max |act_fp16| == 4.0f (K>=16384) or 8.0f (K<16384).
template <bool VEC8>
__global__ void __launch_bounds__(256)
prepare_act_scaled_fp16_kernel(const float* __restrict__ x_fp32, half* __restrict__ act_fp16,
                               float* __restrict__ act_scales, int n_tokens, int K) {
    const int tok = blockIdx.x;
    if (tok >= n_tokens) return;
    const float* x_row = x_fp32 + static_cast<size_t>(tok) * K;
    half* y_row = act_fp16 + static_cast<size_t>(tok) * K;

    float local_max = 0.0f;
    int local_bad = 0;
    if constexpr (VEC8) {
        const int n8 = K >> 3;
        const float4* x4 = reinterpret_cast<const float4*>(x_row);
        for (int i = threadIdx.x; i < n8; i += 256) {
            const float4 v0 = __ldg(x4 + i * 2), v1 = __ldg(x4 + i * 2 + 1);
            if (!isfinite(v0.x) || !isfinite(v0.y) || !isfinite(v0.z) || !isfinite(v0.w) ||
                !isfinite(v1.x) || !isfinite(v1.y) || !isfinite(v1.z) || !isfinite(v1.w)) {
                local_bad = 1;
            } else {
                const float m0 = fmaxf(fmaxf(fabsf(v0.x), fabsf(v0.y)), fmaxf(fabsf(v0.z), fabsf(v0.w)));
                const float m1 = fmaxf(fmaxf(fabsf(v1.x), fabsf(v1.y)), fmaxf(fabsf(v1.z), fabsf(v1.w)));
                local_max = fmaxf(local_max, fmaxf(m0, m1));
            }
        }
    } else {
        for (int k = threadIdx.x; k < K; k += 256) {
            const float v = __ldg(x_row + k);
            if (!isfinite(v)) local_bad = 1;
            else local_max = fmaxf(local_max, fabsf(v));
        }
    }
    float inv_sc = 0.0f;
    int bad = 0;
    reduce_token_scale(local_max, local_bad, K, act_scales, tok, inv_sc, bad);
    if constexpr (VEC8) {
        const int n8 = K >> 3;
        const float4* x4 = reinterpret_cast<const float4*>(x_row);
        uint4* y4 = reinterpret_cast<uint4*>(y_row);
        if (bad) {
            for (int i = threadIdx.x; i < n8; i += 256) y4[i] = make_uint4(0u, 0u, 0u, 0u);
        } else {
            for (int i = threadIdx.x; i < n8; i += 256) {
                const float4 v0 = __ldg(x4 + i * 2), v1 = __ldg(x4 + i * 2 + 1);
                y4[i] = pack_f8_to_u4(v0.x * inv_sc, v0.y * inv_sc, v0.z * inv_sc, v0.w * inv_sc,
                                      v1.x * inv_sc, v1.y * inv_sc, v1.z * inv_sc, v1.w * inv_sc);
            }
        }
    } else {
        for (int k = threadIdx.x; k < K; k += 256) {
            y_row[k] = bad ? __ushort_as_half(0u) : __float2half_rn(__ldg(x_row + k) * inv_sc);
        }
    }
}

// 1D flat 128-bit (8x half -> 2x float4) epilogue kernel: 100% active warps even on small N<=64 matrices.
__global__ void __launch_bounds__(256)
unscale_tile_f16_to_f32_vec8_flat_kernel(const half* __restrict__ y_tile_fp16, const float* __restrict__ act_scales,
                                         float* __restrict__ y_fp32, size_t total_octs, int octs_per_tok,
                                         int N, int row0) {
    const size_t idx = static_cast<size_t>(blockIdx.x) * 256 + threadIdx.x;
    if (idx >= total_octs) return;
    const int tok = static_cast<int>(idx / static_cast<size_t>(octs_per_tok));
    const int oct = static_cast<int>(idx - static_cast<size_t>(tok) * static_cast<size_t>(octs_per_tok));
    const float sc = __ldg(act_scales + tok);
    const uint4 h8 = __ldg(reinterpret_cast<const uint4*>(y_tile_fp16) + idx);
    float4* dst4 = reinterpret_cast<float4*>(y_fp32 + static_cast<size_t>(tok) * N + row0 + oct * 8);
    dst4[0] = make_float4(__half2float(__ushort_as_half(static_cast<uint16_t>(h8.x & 0xFFFFu))) * sc,
                          __half2float(__ushort_as_half(static_cast<uint16_t>(h8.x >> 16))) * sc,
                          __half2float(__ushort_as_half(static_cast<uint16_t>(h8.y & 0xFFFFu))) * sc,
                          __half2float(__ushort_as_half(static_cast<uint16_t>(h8.y >> 16))) * sc);
    dst4[1] = make_float4(__half2float(__ushort_as_half(static_cast<uint16_t>(h8.z & 0xFFFFu))) * sc,
                          __half2float(__ushort_as_half(static_cast<uint16_t>(h8.z >> 16))) * sc,
                          __half2float(__ushort_as_half(static_cast<uint16_t>(h8.w & 0xFFFFu))) * sc,
                          __half2float(__ushort_as_half(static_cast<uint16_t>(h8.w >> 16))) * sc);
}

__global__ void __launch_bounds__(256)
unscale_tile_f16_to_f32_vec4_flat_kernel(const half* __restrict__ y_tile_fp16, const float* __restrict__ act_scales,
                                         float* __restrict__ y_fp32, size_t total_quads, int quads_per_tok,
                                         int N, int row0) {
    const size_t idx = static_cast<size_t>(blockIdx.x) * 256 + threadIdx.x;
    if (idx >= total_quads) return;
    const int tok = static_cast<int>(idx / static_cast<size_t>(quads_per_tok));
    const int quad = static_cast<int>(idx - static_cast<size_t>(tok) * static_cast<size_t>(quads_per_tok));
    const float sc = __ldg(act_scales + tok);
    const uint2 h4 = __ldg(reinterpret_cast<const uint2*>(y_tile_fp16) + idx);
    *reinterpret_cast<float4*>(y_fp32 + static_cast<size_t>(tok) * N + row0 + quad * 4) = make_float4(
        __half2float(__ushort_as_half(static_cast<uint16_t>(h4.x & 0xFFFFu))) * sc,
        __half2float(__ushort_as_half(static_cast<uint16_t>(h4.x >> 16))) * sc,
        __half2float(__ushort_as_half(static_cast<uint16_t>(h4.y & 0xFFFFu))) * sc,
        __half2float(__ushort_as_half(static_cast<uint16_t>(h4.y >> 16))) * sc);
}

__global__ void __launch_bounds__(256)
unscale_tile_f16_to_f32_scalar_kernel(const half* __restrict__ y_tile_fp16, const float* __restrict__ act_scales,
                                      float* __restrict__ y_fp32, int n_tokens, int N, int row0, int tile_rows) {
    const int tok = blockIdx.y, c = blockIdx.x * blockDim.x + threadIdx.x;
    if (tok >= n_tokens || c >= tile_rows) return;
    y_fp32[static_cast<size_t>(tok) * N + row0 + c] =
        __half2float(__ldg(y_tile_fp16 + static_cast<size_t>(tok) * tile_rows + c)) * __ldg(act_scales + tok);
}

__global__ void __launch_bounds__(256)
scale_tile_f32_kernel(float* __restrict__ y_fp32, const float* __restrict__ act_scales,
                      int n_tokens, int N, int row0, int tile_rows) {
    const int tok = blockIdx.y, c = blockIdx.x * blockDim.x + threadIdx.x;
    if (tok >= n_tokens || c >= tile_rows) return;
    y_fp32[static_cast<size_t>(tok) * N + row0 + c] *= __ldg(act_scales + tok);
}

void launch_unscale_tile_f16_to_f32(const half* y_tile_fp16, const float* act_scales, float* y_fp32,
                                    int n_tokens, int N, int row0, int tile_rows, cudaStream_t stream) {
    const bool vec8_ok = ((tile_rows & 7) == 0) && ((N & 3) == 0) && ((row0 & 3) == 0) &&
                         ((reinterpret_cast<uintptr_t>(y_tile_fp16) & 15u) == 0u) &&
                         ((reinterpret_cast<uintptr_t>(y_fp32) & 15u) == 0u);
    if (vec8_ok) {
        const int octs_per_tok = tile_rows >> 3;
        const size_t total_octs = static_cast<size_t>(n_tokens) * static_cast<size_t>(octs_per_tok);
        unscale_tile_f16_to_f32_vec8_flat_kernel<<<static_cast<unsigned>((total_octs + 255) / 256), 256, 0, stream>>>(
            y_tile_fp16, act_scales, y_fp32, total_octs, octs_per_tok, N, row0);
        ck(cudaGetLastError());
        return;
    }
    const bool vec4_ok = ((tile_rows & 3) == 0) && ((N & 3) == 0) && ((row0 & 3) == 0) &&
                         ((reinterpret_cast<uintptr_t>(y_tile_fp16) & 7u) == 0u) &&
                         ((reinterpret_cast<uintptr_t>(y_fp32) & 15u) == 0u);
    if (vec4_ok) {
        const int quads_per_tok = tile_rows >> 2;
        const size_t total_quads = static_cast<size_t>(n_tokens) * static_cast<size_t>(quads_per_tok);
        unscale_tile_f16_to_f32_vec4_flat_kernel<<<static_cast<unsigned>((total_quads + 255) / 256), 256, 0, stream>>>(
            y_tile_fp16, act_scales, y_fp32, total_quads, quads_per_tok, N, row0);
        ck(cudaGetLastError());
        return;
    }
    unscale_tile_f16_to_f32_scalar_kernel<<<dim3((tile_rows + 255) / 256, n_tokens), 256, 0, stream>>>(
        y_tile_fp16, act_scales, y_fp32, n_tokens, N, row0, tile_rows);
    ck(cudaGetLastError());
}

void launch_dequant_tile(int type, const uint8_t* src, half* dst, int tile_rows, int K,
                         size_t row_stride, cudaStream_t stream) {
    const bool k_aligned_256 = (K > 0) && ((K & 255) == 0);
    const bool dst_aligned_16 = ((reinterpret_cast<uintptr_t>(dst) & 15u) == 0u);
    const bool src_aligned_2 = ((reinterpret_cast<uintptr_t>(src) & 1u) == 0u) && ((row_stride & 1u) == 0u);
    const bool src_aligned_8 = ((reinterpret_cast<uintptr_t>(src) & 7u) == 0u) && ((row_stride & 7u) == 0u);

    if (k_aligned_256 && dst_aligned_16 && src_aligned_2) {
        const dim3 fg(K / 256, (tile_rows + 7) / 8), fb(256);
        if (type == 8)  { dequant_q8_0_256_kernel<<<fg, fb, 0, stream>>>(src, dst, tile_rows, K, row_stride); ck(cudaGetLastError()); return; }
        if (type == 12 && src_aligned_8) { dequant_qk45_256_kernel<false><<<fg, fb, 0, stream>>>(src, dst, tile_rows, K, row_stride); ck(cudaGetLastError()); return; }
        if (type == 13 && src_aligned_8) { dequant_qk45_256_kernel<true><<<fg, fb, 0, stream>>>(src, dst, tile_rows, K, row_stride); ck(cudaGetLastError()); return; }
        if (type == 14) { dequant_q6k_256_kernel<<<fg, fb, 0, stream>>>(src, dst, tile_rows, K, row_stride); ck(cudaGetLastError()); return; }
        if (type == 21) { dequant_iq3s_256_kernel<<<fg, fb, 0, stream>>>(src, dst, tile_rows, K, row_stride); ck(cudaGetLastError()); return; }
        if (type == 22) { dequant_iq2s_256_kernel<<<fg, fb, 0, stream>>>(src, dst, tile_rows, K, row_stride); ck(cudaGetLastError()); return; }
    }

    const dim3 grid((K + 255) / 256, tile_rows), block(256);
    switch (type) {
        case 0:   dequant_tile_kernel<0><<<grid, block, 0, stream>>>(src, dst, tile_rows, K, row_stride); break;
        case 1:   dequant_tile_kernel<1><<<grid, block, 0, stream>>>(src, dst, tile_rows, K, row_stride); break;
        case 8:   dequant_tile_kernel<8><<<grid, block, 0, stream>>>(src, dst, tile_rows, K, row_stride); break;
        case 11:  dequant_tile_kernel<11><<<grid, block, 0, stream>>>(src, dst, tile_rows, K, row_stride); break;
        case 12:  dequant_tile_kernel<12><<<grid, block, 0, stream>>>(src, dst, tile_rows, K, row_stride); break;
        case 13:  dequant_tile_kernel<13><<<grid, block, 0, stream>>>(src, dst, tile_rows, K, row_stride); break;
        case 14:  dequant_tile_kernel<14><<<grid, block, 0, stream>>>(src, dst, tile_rows, K, row_stride); break;
        case 20:  dequant_tile_kernel<20><<<grid, block, 0, stream>>>(src, dst, tile_rows, K, row_stride); break;
        case 21:  dequant_tile_kernel<21><<<grid, block, 0, stream>>>(src, dst, tile_rows, K, row_stride); break;
        case 22:  dequant_tile_kernel<22><<<grid, block, 0, stream>>>(src, dst, tile_rows, K, row_stride); break;
        case 23:  dequant_tile_kernel<23><<<grid, block, 0, stream>>>(src, dst, tile_rows, K, row_stride); break;
        case 30:  dequant_tile_kernel<30><<<grid, block, 0, stream>>>(src, dst, tile_rows, K, row_stride); break;
        case 39:  dequant_tile_kernel<39><<<grid, block, 0, stream>>>(src, dst, tile_rows, K, row_stride); break;
        case 40:  dequant_tile_kernel<40><<<grid, block, 0, stream>>>(src, dst, tile_rows, K, row_stride); break;
        case 142: dequant_tile_kernel<142><<<grid, block, 0, stream>>>(src, dst, tile_rows, K, row_stride); break;
        default:  throw std::runtime_error("unsupported matrix type in prefill_gemm");
    }
    ck(cudaGetLastError());
}

} // namespace

template<int TYPE> struct Q38WmmaDecoder {
    __device__ __forceinline__ float operator()(const uint8_t* row, int k) const { return dequant_elem(TYPE, row, k); }
};
template<int TYPE>
static void launch_fused_wmma_type(const StrataQuantTensor& weight, const half* x, float* y, int T, cudaStream_t stream) {
    const dim3 grid(wmma_layout::tiles(weight.ne1, wmma_layout::rows), wmma_layout::tiles(T, wmma_layout::tokens));
    detail::packed_wmma_kernel<<<grid, wmma_layout::threads, 0, stream>>>(
        static_cast<const uint8_t*>(weight.d_data), x, y, weight.ne0, weight.ne1, T, weight.row_bytes, Q38WmmaDecoder<TYPE>{});
}
void matmul_fused_wmma_fp16(const StrataQuantTensor& weight, const half* x, float* y, int n_tokens, cudaStream_t stream) {
    if (!weight.d_data || !x || !y || weight.ne0 <= 0 || weight.ne1 <= 0 || n_tokens <= 0 || n_tokens > 65535 * 32)
        throw std::runtime_error("invalid fused WMMA operands or grid size");
    const auto type = weight.type;
    const int block = static_cast<int>(StrataQuantTensor::block_elements(type));
    if (block <= 0 || weight.ne0 % block) throw std::runtime_error("invalid fused WMMA packed width");
    const size_t minimum_row = size_t(weight.ne0 / block) * StrataQuantTensor::block_bytes(type);
    if (weight.row_bytes < minimum_row) throw std::runtime_error("invalid fused WMMA row stride");
    switch (static_cast<int>(type)) {
        case 0: launch_fused_wmma_type<0>(weight, x, y, n_tokens, stream); break;
        case 1: launch_fused_wmma_type<1>(weight, x, y, n_tokens, stream); break;
        case 8: launch_fused_wmma_type<8>(weight, x, y, n_tokens, stream); break;
        case 12: launch_fused_wmma_type<12>(weight, x, y, n_tokens, stream); break;
        case 13: launch_fused_wmma_type<13>(weight, x, y, n_tokens, stream); break;
        case 14: launch_fused_wmma_type<14>(weight, x, y, n_tokens, stream); break;
        case 21: launch_fused_wmma_type<21>(weight, x, y, n_tokens, stream); break;
        case 22: launch_fused_wmma_type<22>(weight, x, y, n_tokens, stream); break;
        case 23: launch_fused_wmma_type<23>(weight, x, y, n_tokens, stream); break;
        case 30: launch_fused_wmma_type<30>(weight, x, y, n_tokens, stream); break;
        case 142: launch_fused_wmma_type<142>(weight, x, y, n_tokens, stream); break;
        default: throw std::runtime_error("unsupported fused WMMA GGUF type");
    }
    ck(cudaGetLastError());
}

struct LtPlanEntry {
    int tile_rows = 0, n_tokens = 0, K = 0, N = 0;
    cublasLtMatmulDesc_t desc = nullptr;
    cublasLtMatrixLayout_t a_desc = nullptr, b_desc = nullptr, c_desc = nullptr;
    cublasLtMatmulHeuristicResult_t heur{};
    bool valid = false;
};

struct PrefillGemmWorkspace::Impl {
    cublasHandle_t cublas = nullptr;
    cublasLtHandle_t cublas_lt = nullptr;
    half* tile_buf[2] = {nullptr, nullptr};
    size_t tile_buf_bytes = 0;
    half* act_fp16 = nullptr;
    size_t act_elems = 0;
    int max_tokens = 0, max_k = 0;
    size_t max_tile_rows_alloc = 0;
    half* y_tile_fp16 = nullptr;
    float* act_scales = nullptr;
    BlockQ8_1* act_q8 = nullptr;
    int act_q8_max_tokens = 0;

    __nv_fp8_e4m3* act_fp8 = nullptr;
    __nv_fp8_e4m3* w_fp8 = nullptr;
    float* w_scales = nullptr;
    void* lt_workspace = nullptr;
    size_t lt_workspace_bytes = 4ull * 1024 * 1024;
    bool use_fp8 = false;
    bool force_fp32_matvec = false;
    cudaStream_t dequant_stream = nullptr;
    cudaEvent_t tile_ready[2] = {nullptr, nullptr};
    cudaEvent_t tile_done[2] = {nullptr, nullptr};
    cudaEvent_t act_ready = nullptr;

    size_t persisting_max = 0, window_max = 0;
    cudaStream_t l2_stream = nullptr;
    bool l2_window_set = false;

    const float* prepared_x_fp32 = nullptr;
    int prepared_tokens = 0, prepared_k = 0;

    PrefillGemmStats stats{};
    std::vector<LtPlanEntry> lt_plans;
};

__global__ void __launch_bounds__(256)
scale_fp8_rows_kernel(float* __restrict__ y, const float* __restrict__ x_sc,
                      const float* __restrict__ w_sc, int n_tokens, int rows, int row0, int tile_rows) {
    const int tok = blockIdx.y, c = blockIdx.x * blockDim.x + threadIdx.x;
    if (tok >= n_tokens || c >= tile_rows) return;
    y[static_cast<size_t>(tok) * rows + row0 + c] *= (x_sc[tok] * w_sc[c]);
}

PrefillGemmWorkspace::~PrefillGemmWorkspace() { release(); }

size_t PrefillGemmWorkspace::estimate_bytes(int max_tokens, int max_k, size_t tile_buf_bytes) {
    if (max_tokens <= 0 || max_k <= 0 || tile_buf_bytes < static_cast<size_t>(max_k) * sizeof(half)) return 0;
    const bool use_fp8 = fp8_prefill_requested();
    const size_t act_elems = static_cast<size_t>(max_tokens) * static_cast<size_t>(max_k);
    const size_t act_fp16_bytes = act_elems * sizeof(half);
    const size_t act_scales_bytes = static_cast<size_t>(max_tokens) * sizeof(float);
    if (!use_fp8) {
        // The FP16 output tile needs at most tile_buf_bytes/(min_K*sizeof(half)) rows;
        // every admitted projection has K >= 5120. Using 512 here overallocates 20x.
        const size_t max_tile_rows = std::max<size_t>(16, tile_buf_bytes / (5120u * sizeof(half)));
        return tile_buf_bytes + act_fp16_bytes + static_cast<size_t>(max_tokens) * max_tile_rows * sizeof(half) + act_scales_bytes;
    }
    const size_t max_tile_rows_est = (tile_buf_bytes / 2) / 512 + 1024;
    constexpr size_t lt_workspace_bytes = 4ull * 1024 * 1024;
    return tile_buf_bytes * 2 + act_fp16_bytes + act_elems * sizeof(__nv_fp8_e4m3)
         + (tile_buf_bytes / 2) + lt_workspace_bytes + act_scales_bytes + max_tile_rows_est * sizeof(float);
}

int PrefillGemmWorkspace::cublas_version() {
    cublasHandle_t h = nullptr;
    int ver = 0;
    if (cublasCreate(&h) == CUBLAS_STATUS_SUCCESS) { cublasGetVersion(h, &ver); cublasDestroy(h); }
    return ver;
}

PrefillGemmStats PrefillGemmWorkspace::stats() const { return impl_ ? impl_->stats : PrefillGemmStats{}; }
void PrefillGemmWorkspace::reset_stats() { if (impl_) impl_->stats = PrefillGemmStats{}; }

void PrefillGemmWorkspace::init(int max_tokens, int max_k, size_t tile_buf_bytes) {
    release();
    if (max_tokens <= 0 || max_k <= 0 || tile_buf_bytes < static_cast<size_t>(max_k) * sizeof(half))
        throw std::runtime_error("invalid PrefillGemmWorkspace dimensions");
    impl_ = new Impl();
    try {
        impl_->use_fp8 = fp8_prefill_requested();
        impl_->force_fp32_matvec = fp32_prefill_requested();
        impl_->max_tokens = max_tokens;
        impl_->max_k = max_k;
        impl_->tile_buf_bytes = tile_buf_bytes;
        impl_->act_elems = static_cast<size_t>(max_tokens) * static_cast<size_t>(max_k);
        ck_cublas(cublasCreate(&impl_->cublas));
        ck_cublas(cublasSetMathMode(impl_->cublas, CUBLAS_DEFAULT_MATH));
        size_t allocated = 0;
        auto alloc_dev = [&](auto** ptr, size_t bytes) { ck(cudaMalloc(reinterpret_cast<void**>(ptr), bytes)); allocated += bytes; };
        alloc_dev(&impl_->tile_buf[0], tile_buf_bytes);
        alloc_dev(&impl_->act_fp16, impl_->act_elems * sizeof(half));
        alloc_dev(&impl_->act_scales, static_cast<size_t>(max_tokens) * sizeof(float));
        impl_->act_q8_max_tokens = std::min(max_tokens, 16);
        if (impl_->act_q8_max_tokens >= 2 && (max_k / 32) > 0) {
            alloc_dev(&impl_->act_q8, static_cast<size_t>(impl_->act_q8_max_tokens) * static_cast<size_t>(max_k / 32) * sizeof(BlockQ8_1));
        }
        if (!impl_->use_fp8) {
            impl_->max_tile_rows_alloc = std::max<size_t>(16, tile_buf_bytes / (5120u * sizeof(half)));
            alloc_dev(&impl_->y_tile_fp16, static_cast<size_t>(max_tokens) * impl_->max_tile_rows_alloc * sizeof(half));
            int dev = 0, persisting_max = 0, window_max = 0;
            ck(cudaGetDevice(&dev));
            ck(cudaDeviceGetAttribute(&persisting_max, cudaDevAttrMaxPersistingL2CacheSize, dev));
            ck(cudaDeviceGetAttribute(&window_max, cudaDevAttrMaxAccessPolicyWindowSize, dev));
            impl_->persisting_max = (persisting_max > 0) ? static_cast<size_t>(persisting_max) : 0;
            impl_->window_max = (window_max > 0) ? static_cast<size_t>(window_max) : 0;
            if (impl_->persisting_max > 0 && tile_buf_bytes <= impl_->persisting_max)
                ck(cudaDeviceSetLimit(cudaLimitPersistingL2CacheSize, tile_buf_bytes));
        } else {
            ck_cublas(cublasLtCreate(&impl_->cublas_lt));
            alloc_dev(&impl_->tile_buf[1], tile_buf_bytes);
            alloc_dev(&impl_->act_fp8, impl_->act_elems * sizeof(__nv_fp8_e4m3));
            alloc_dev(&impl_->w_fp8, tile_buf_bytes / 2);
            alloc_dev(&impl_->w_scales, ((tile_buf_bytes / 2) / 512 + 1024) * sizeof(float));
            alloc_dev(&impl_->lt_workspace, impl_->lt_workspace_bytes);
            ck(cudaStreamCreateWithFlags(&impl_->dequant_stream, cudaStreamNonBlocking));
            for (int i = 0; i < 2; ++i) {
                ck(cudaEventCreateWithFlags(&impl_->tile_ready[i], cudaEventDisableTiming));
                ck(cudaEventCreateWithFlags(&impl_->tile_done[i], cudaEventDisableTiming));
            }
            ck(cudaEventCreateWithFlags(&impl_->act_ready, cudaEventDisableTiming));
        }
        total_bytes_ = allocated;
    } catch (...) { release(); throw; }
}

void PrefillGemmWorkspace::reset_l2_window(cudaStream_t stream) {
    if (!impl_ || !impl_->l2_window_set) return;
    cudaStreamAttrValue attr{};
    attr.accessPolicyWindow.hitProp = cudaAccessPropertyNormal;
    attr.accessPolicyWindow.missProp = cudaAccessPropertyNormal;
    ck(cudaStreamSetAttribute(impl_->l2_stream, cudaStreamAttributeAccessPolicyWindow, &attr));
    if (stream != impl_->l2_stream) ck(cudaStreamSetAttribute(stream, cudaStreamAttributeAccessPolicyWindow, &attr));
    ck(cudaCtxResetPersistingL2Cache());
    impl_->l2_window_set = false;
}

void PrefillGemmWorkspace::release() {
    if (!impl_) return;
    if (impl_->l2_window_set) {
        cudaStreamAttrValue attr{};
        attr.accessPolicyWindow.hitProp = cudaAccessPropertyNormal;
        attr.accessPolicyWindow.missProp = cudaAccessPropertyNormal;
        cudaStreamSetAttribute(impl_->l2_stream, cudaStreamAttributeAccessPolicyWindow, &attr);
        cudaCtxResetPersistingL2Cache();
        cudaGetLastError();
        impl_->l2_window_set = false;
    }
    for (auto& plan : impl_->lt_plans) {
        if (plan.c_desc) cublasLtMatrixLayoutDestroy(plan.c_desc);
        if (plan.b_desc) cublasLtMatrixLayoutDestroy(plan.b_desc);
        if (plan.a_desc) cublasLtMatrixLayoutDestroy(plan.a_desc);
        if (plan.desc) cublasLtMatmulDescDestroy(plan.desc);
    }
    impl_->lt_plans.clear();
    if (impl_->act_ready) cudaEventDestroy(impl_->act_ready);
    for (int i = 0; i < 2; ++i) {
        if (impl_->tile_done[i]) cudaEventDestroy(impl_->tile_done[i]);
        if (impl_->tile_ready[i]) cudaEventDestroy(impl_->tile_ready[i]);
    }
    if (impl_->dequant_stream) cudaStreamDestroy(impl_->dequant_stream);
    for (void* p : {impl_->lt_workspace, static_cast<void*>(impl_->w_scales), static_cast<void*>(impl_->act_scales),
                    static_cast<void*>(impl_->act_q8),
                    static_cast<void*>(impl_->w_fp8), static_cast<void*>(impl_->act_fp8),
                    static_cast<void*>(impl_->y_tile_fp16), static_cast<void*>(impl_->act_fp16),
                    static_cast<void*>(impl_->tile_buf[0]), static_cast<void*>(impl_->tile_buf[1])}) {
        if (p) cudaFree(p);
    }
    if (impl_->cublas_lt) cublasLtDestroy(impl_->cublas_lt);
    if (impl_->cublas) cublasDestroy(impl_->cublas);
    delete impl_;
    impl_ = nullptr;
    total_bytes_ = 0;
}

void dequant_rows_fp16(const StrataQuantTensor& weight, int row_start, int tile_rows, half* dst_fp16, cudaStream_t stream) {
    if (!weight.d_data || !dst_fp16 || row_start < 0 || tile_rows <= 0 || weight.ne0 <= 0 || row_start + tile_rows > weight.ne1)
        throw std::runtime_error("invalid dequant_rows_fp16 arguments");
    const size_t be = StrataQuantTensor::block_elements(weight.type), bb = StrataQuantTensor::block_bytes(weight.type);
    if (be == 0 || bb == 0 || (static_cast<size_t>(weight.ne0) % be) != 0)
        throw std::runtime_error("invalid dequant_rows_fp16 block alignment");
    if (weight.row_bytes < (static_cast<size_t>(weight.ne0) / be) * bb)
        throw std::runtime_error("invalid dequant_rows_fp16 row stride");
    const auto* src = static_cast<const uint8_t*>(weight.d_data) + static_cast<size_t>(row_start) * weight.row_bytes;
    launch_dequant_tile(static_cast<int>(weight.type), src, dst_fp16, tile_rows, weight.ne0, weight.row_bytes, stream);
}

void PrefillGemmWorkspace::prepare_activation(const float* x_fp32, int n_tokens, int K, cudaStream_t stream) {
    if (!impl_) throw std::runtime_error("PrefillGemmWorkspace is not initialized");
    if (!x_fp32 || n_tokens <= 0 || K <= 0) throw std::runtime_error("invalid prepare_activation arguments");
    const size_t x_elems = static_cast<size_t>(n_tokens) * static_cast<size_t>(K);
    if (n_tokens > impl_->max_tokens || x_elems > impl_->act_elems)
        throw std::runtime_error("PrefillGemmWorkspace activation buffer too small");
    impl_->prepared_x_fp32 = x_fp32;
    impl_->prepared_tokens = n_tokens;
    impl_->prepared_k = K;
    if (impl_->act_q8 && !impl_->force_fp32_matvec && !fp16_gemv_requested() &&
        n_tokens >= 2 && n_tokens <= impl_->act_q8_max_tokens && (K % 32) == 0) {
        quantize_q8_1_batch(x_fp32, impl_->act_q8, n_tokens, K, stream);
    }
    if (n_tokens < 4 || impl_->force_fp32_matvec) return;

    if (!impl_->use_fp8) {
        const bool aligned_16 = ((reinterpret_cast<uintptr_t>(x_fp32) & 15u) == 0u) &&
                                ((reinterpret_cast<uintptr_t>(impl_->act_fp16) & 15u) == 0u);
        if (aligned_16 && K == 4096)
            prepare_act_scaled_fp16_reg_kernel<4096><<<n_tokens, 256, 0, stream>>>(x_fp32, impl_->act_fp16, impl_->act_scales, n_tokens);
        else if (aligned_16 && K == 5120)
            prepare_act_scaled_fp16_reg_kernel<5120><<<n_tokens, 256, 0, stream>>>(x_fp32, impl_->act_fp16, impl_->act_scales, n_tokens);
        else if (aligned_16 && K == 6144)
            prepare_act_scaled_fp16_reg_kernel<6144><<<n_tokens, 256, 0, stream>>>(x_fp32, impl_->act_fp16, impl_->act_scales, n_tokens);
        else if (aligned_16 && K == 12288)
            prepare_act_scaled_fp16_reg_kernel<12288><<<n_tokens, 256, 0, stream>>>(x_fp32, impl_->act_fp16, impl_->act_scales, n_tokens);
        else if (aligned_16 && ((K & 7) == 0))
            prepare_act_scaled_fp16_kernel<true><<<n_tokens, 256, 0, stream>>>(x_fp32, impl_->act_fp16, impl_->act_scales, n_tokens, K);
        else
            prepare_act_scaled_fp16_kernel<false><<<n_tokens, 256, 0, stream>>>(x_fp32, impl_->act_fp16, impl_->act_scales, n_tokens, K);
        ck(cudaGetLastError());
        return;
    }

    f32_to_fp16_kernel<<<static_cast<unsigned>((x_elems + 255) / 256), 256, 0, stream>>>(x_fp32, impl_->act_fp16, x_elems);
    ck(cudaGetLastError());
    if ((n_tokens >= 16) && (n_tokens % 16 == 0) && (K % 16 == 0)) {
        spin8_fp8_hadamard_quant_kernel<<<n_tokens, 256, 0, stream>>>(impl_->act_fp16, impl_->act_fp8, impl_->act_scales, n_tokens, K);
        ck(cudaGetLastError());
    }
}

void PrefillGemmWorkspace::matmul_batch_prepared(const StrataQuantTensor& weight, float* y_fp32,
                                                 int n_tokens, cudaStream_t stream) {
    if (!impl_) throw std::runtime_error("PrefillGemmWorkspace is not initialized");
    if (!weight.d_data || !y_fp32 || n_tokens <= 0 || weight.ne0 <= 0 || weight.ne1 <= 0) {
        throw std::runtime_error("invalid matmul_batch_prepared arguments");
    }
    const int K = weight.ne0, N = weight.ne1;
    if (!impl_->prepared_x_fp32 || n_tokens != impl_->prepared_tokens || K != impl_->prepared_k) {
        throw std::runtime_error("matmul_batch_prepared shape mismatch with prepare_activation");
    }

    const int wtype = static_cast<int>(weight.type);
    if (impl_->act_q8 && !impl_->force_fp32_matvec && !fp16_gemv_requested() &&
        n_tokens >= 2 && n_tokens <= impl_->act_q8_max_tokens && matvec_q8_1_batch_supported(wtype)) {
        const bool aligned_ok = (wtype == 8)
            ? ((K % 32) == 0)
            : ((wtype == 14) ? ((K % 256) == 0)
                             : (((K % 256) == 0) && ((reinterpret_cast<uintptr_t>(weight.d_data) & 15u) == 0u) && ((weight.row_bytes & 15u) == 0u)));
        if (aligned_ok) {
            matvec_q8_1_batch(wtype, weight.d_data, impl_->act_q8, y_fp32, K, N, n_tokens, stream);
            return;
        }
    }

    if ((n_tokens < 4 || impl_->force_fp32_matvec) && !fp16_gemv_requested()) {
        impl_->stats.matvec_fallback_calls += static_cast<uint64_t>(n_tokens);
        for (int t = 0; t < n_tokens; ++t) {
            matvec(static_cast<int>(weight.type), weight.d_data,
                   impl_->prepared_x_fp32 + static_cast<size_t>(t) * K,
                   y_fp32 + static_cast<size_t>(t) * N, K, N, stream);
        }
        return;
    }

#if defined(STRATA_Q38_FUSED_WMMA) && STRATA_Q38_FUSED_WMMA
    if (wmma_layout::eligible(K, N, n_tokens, impl_->use_fp8)) {
        matmul_fused_wmma_fp16(weight, impl_->act_fp16, y_fp32, n_tokens, stream);
        if (!impl_->use_fp8) {
            scale_tile_f32_kernel<<<dim3((N + 255) / 256, n_tokens), 256, 0, stream>>>(
                y_fp32, impl_->act_scales, n_tokens, N, 0, N);
            ck(cudaGetLastError());
        }
        ++impl_->stats.fused_wmma_calls;
        return;
    }
#endif

    ck_cublas(cublasSetStream(impl_->cublas, stream));

    if (!impl_->use_fp8) {
        if (impl_->persisting_max > 0 && impl_->tile_buf_bytes <= impl_->persisting_max &&
            (impl_->window_max == 0 || impl_->tile_buf_bytes <= impl_->window_max) &&
            (!impl_->l2_window_set || impl_->l2_stream != stream)) {
            if (impl_->l2_window_set) reset_l2_window(impl_->l2_stream);
            cudaStreamAttrValue attr{};
            attr.accessPolicyWindow.base_ptr = impl_->tile_buf[0];
            attr.accessPolicyWindow.num_bytes = impl_->tile_buf_bytes;
            attr.accessPolicyWindow.hitRatio = 1.0f;
            attr.accessPolicyWindow.hitProp = cudaAccessPropertyPersisting;
            attr.accessPolicyWindow.missProp = cudaAccessPropertyStreaming;
            ck(cudaStreamSetAttribute(stream, cudaStreamAttributeAccessPolicyWindow, &attr));
            impl_->l2_stream = stream;
            impl_->l2_window_set = true;
        }

        const int wtype = static_cast<int>(weight.type);
        // Opt-in diagnostics: FP32 accumulation writes FP32 directly and can
        // beat the FP16-accumulate + rescale path on some SM120 shapes.
        const char* accum_env = std::getenv("STRATA_Q38_FP32_ACCUM");
        const bool use_fp32_accum = (wtype == 0 || wtype == 1 || wtype == 30) ||
            (accum_env && accum_env[0] == '1');
        size_t max_tile_rows_sz = impl_->tile_buf_bytes / (static_cast<size_t>(K) * sizeof(half));
        if (max_tile_rows_sz == 0) throw std::runtime_error("PrefillGemmWorkspace tile buffer smaller than one row");
        if (!use_fp32_accum && max_tile_rows_sz > impl_->max_tile_rows_alloc) max_tile_rows_sz = impl_->max_tile_rows_alloc;
        if (max_tile_rows_sz >= 16) max_tile_rows_sz = (max_tile_rows_sz / 16) * 16;

        const int cap = static_cast<int>(std::min<size_t>(max_tile_rows_sz, static_cast<size_t>(N)));
        const int init_tiles = (N + cap - 1) / cap;
        int max_tile_rows = cap;
        if (init_tiles > 1 && cap >= 16) {
            const int even16 = (((N + init_tiles - 1) / init_tiles + 15) / 16) * 16;
            if (even16 <= cap) max_tile_rows = even16;
        }
        const int n_tiles = (N + max_tile_rows - 1) / max_tile_rows;
        const half half_one = __float2half(1.0f), half_zero = __float2half(0.0f);
        const float f32_one = 1.0f, f32_zero = 0.0f;

        for (int ti = 0; ti < n_tiles; ++ti) {
            const int row0 = ti * max_tile_rows, tile_rows = std::min(max_tile_rows, N - row0);
            dequant_rows_fp16(weight, row0, tile_rows, impl_->tile_buf[0], stream);
            if (use_fp32_accum) {
                ck_cublas(cublasGemmEx(impl_->cublas, CUBLAS_OP_T, CUBLAS_OP_N, tile_rows, n_tokens, K, &f32_one,
                    impl_->tile_buf[0], CUDA_R_16F, K, impl_->act_fp16, CUDA_R_16F, K, &f32_zero,
                    y_fp32 + row0, CUDA_R_32F, N, CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP));
                scale_tile_f32_kernel<<<dim3((tile_rows + 255) / 256, n_tokens), 256, 0, stream>>>(
                    y_fp32, impl_->act_scales, n_tokens, N, row0, tile_rows);
                ck(cudaGetLastError());
            } else {
                if (static_cast<size_t>(tile_rows) > impl_->max_tile_rows_alloc)
                    throw std::runtime_error("PrefillGemmWorkspace y_tile_fp16 capacity exceeded");
                ck_cublas(cublasGemmEx(impl_->cublas, CUBLAS_OP_T, CUBLAS_OP_N, tile_rows, n_tokens, K, &half_one,
                    impl_->tile_buf[0], CUDA_R_16F, K, impl_->act_fp16, CUDA_R_16F, K, &half_zero,
                    impl_->y_tile_fp16, CUDA_R_16F, tile_rows, CUBLAS_COMPUTE_16F, CUBLAS_GEMM_DEFAULT_TENSOR_OP));
                launch_unscale_tile_f16_to_f32(impl_->y_tile_fp16, impl_->act_scales, y_fp32, n_tokens, N, row0, tile_rows, stream);
            }
            ++impl_->stats.fp16_gemm_calls;
        }
        return;
    }

    // Optional STRATA_FP8_PREFILL=1 path
    const bool fp8_eligible = (n_tokens >= 16) && (n_tokens % 16 == 0) && (K % 16 == 0);
    int max_tile_rows = static_cast<int>(impl_->tile_buf_bytes / (static_cast<size_t>(K) * sizeof(half)));
    if (max_tile_rows <= 0) throw std::runtime_error("PrefillGemmWorkspace tile buffer smaller than one row");
    if (fp8_eligible && max_tile_rows >= 16) max_tile_rows = (max_tile_rows / 16) * 16;

    auto get_lt_plan = [&](int tile_rows) -> const LtPlanEntry* {
        for (const auto& entry : impl_->lt_plans) {
            if (entry.tile_rows == tile_rows && entry.n_tokens == n_tokens && entry.K == K && entry.N == N) return &entry;
        }
        LtPlanEntry entry{};
        entry.tile_rows = tile_rows; entry.n_tokens = n_tokens; entry.K = K; entry.N = N;
        cublasOperation_t op_a = CUBLAS_OP_T, op_b = CUBLAS_OP_N;
        ck_cublas(cublasLtMatmulDescCreate(&entry.desc, CUBLAS_COMPUTE_32F, CUDA_R_32F));
        ck_cublas(cublasLtMatmulDescSetAttribute(entry.desc, CUBLASLT_MATMUL_DESC_TRANSA, &op_a, sizeof(op_a)));
        ck_cublas(cublasLtMatmulDescSetAttribute(entry.desc, CUBLASLT_MATMUL_DESC_TRANSB, &op_b, sizeof(op_b)));
        ck_cublas(cublasLtMatrixLayoutCreate(&entry.a_desc, CUDA_R_8F_E4M3, K, tile_rows, K));
        ck_cublas(cublasLtMatrixLayoutCreate(&entry.b_desc, CUDA_R_8F_E4M3, K, n_tokens, K));
        ck_cublas(cublasLtMatrixLayoutCreate(&entry.c_desc, CUDA_R_32F, tile_rows, n_tokens, N));
        cublasLtMatmulPreference_t pref = nullptr;
        int found = 0;
        ck_cublas(cublasLtMatmulPreferenceCreate(&pref));
        ck_cublas(cublasLtMatmulPreferenceSetAttribute(pref, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES,
                                                        &impl_->lt_workspace_bytes, sizeof(impl_->lt_workspace_bytes)));
        ck_cublas(cublasLtMatmulAlgoGetHeuristic(impl_->cublas_lt, entry.desc, entry.a_desc, entry.b_desc,
                                                entry.c_desc, entry.c_desc, pref, 1, &entry.heur, &found));
        cublasLtMatmulPreferenceDestroy(pref);
        entry.valid = (found > 0);
        if (!entry.valid) {
            cublasLtMatrixLayoutDestroy(entry.c_desc); entry.c_desc = nullptr;
            cublasLtMatrixLayoutDestroy(entry.b_desc); entry.b_desc = nullptr;
            cublasLtMatrixLayoutDestroy(entry.a_desc); entry.a_desc = nullptr;
            cublasLtMatmulDescDestroy(entry.desc); entry.desc = nullptr;
        }
        impl_->lt_plans.push_back(entry);
        return &impl_->lt_plans.back();
    };

    const float alpha = 1.0f, beta = 0.0f;
    const int n_tiles = (N + max_tile_rows - 1) / max_tile_rows;
    if (n_tiles > 1) {
        ck(cudaEventRecord(impl_->act_ready, stream));
        ck(cudaStreamWaitEvent(impl_->dequant_stream, impl_->act_ready, 0));
    }
    int slot = 0;
    for (int ti = 0; ti < n_tiles; ++ti) {
        const int row0 = ti * max_tile_rows, tile_rows = std::min(max_tile_rows, N - row0);
        half* w_fp16 = impl_->tile_buf[slot];
        if (n_tiles > 1) {
            if (ti >= 2) ck(cudaStreamWaitEvent(impl_->dequant_stream, impl_->tile_done[slot], 0));
            dequant_rows_fp16(weight, row0, tile_rows, w_fp16, impl_->dequant_stream);
            ck(cudaEventRecord(impl_->tile_ready[slot], impl_->dequant_stream));
            ck(cudaStreamWaitEvent(stream, impl_->tile_ready[slot], 0));
        } else {
            dequant_rows_fp16(weight, row0, tile_rows, w_fp16, stream);
        }
        bool did_fp8 = false;
        if (fp8_eligible && (tile_rows % 16 == 0)) {
            const LtPlanEntry* plan = get_lt_plan(tile_rows);
            if (plan->valid) {
                spin8_fp8_hadamard_quant_kernel<<<tile_rows, 256, 0, stream>>>(w_fp16, impl_->w_fp8, impl_->w_scales, tile_rows, K);
                ck(cudaGetLastError());
                ck_cublas(cublasLtMatmul(impl_->cublas_lt, plan->desc, &alpha, impl_->w_fp8, plan->a_desc,
                                         impl_->act_fp8, plan->b_desc, &beta, y_fp32 + row0, plan->c_desc,
                                         y_fp32 + row0, plan->c_desc, &plan->heur.algo,
                                         impl_->lt_workspace, impl_->lt_workspace_bytes, stream));
                scale_fp8_rows_kernel<<<dim3((tile_rows + 255) / 256, n_tokens), 256, 0, stream>>>(
                    y_fp32, impl_->act_scales, impl_->w_scales, n_tokens, N, row0, tile_rows);
                ck(cudaGetLastError());
                ++impl_->stats.fp8_gemm_calls;
                did_fp8 = true;
            } else {
                ++impl_->stats.fp8_fallback_no_algo;
            }
        } else if (fp8_eligible) {
            ++impl_->stats.fp8_fallback_shape;
        }
        if (!did_fp8) {
            ck_cublas(cublasGemmEx(impl_->cublas, CUBLAS_OP_T, CUBLAS_OP_N, tile_rows, n_tokens, K, &alpha,
                w_fp16, CUDA_R_16F, K, impl_->act_fp16, CUDA_R_16F, K, &beta,
                y_fp32 + row0, CUDA_R_32F, N, CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP));
            ++impl_->stats.fp16_gemm_calls;
        }
        if (n_tiles > 1) ck(cudaEventRecord(impl_->tile_done[slot], stream));
        slot ^= 1;
    }
}

void PrefillGemmWorkspace::matmul_batch(const StrataQuantTensor& weight, const float* x_fp32, float* y_fp32,
                                        int n_tokens, cudaStream_t stream) {
    if (!weight.d_data || !x_fp32 || !y_fp32 || n_tokens <= 0 || weight.ne0 <= 0 || weight.ne1 <= 0) {
        throw std::runtime_error("invalid matmul_batch arguments");
    }
    const int K = weight.ne0, N = weight.ne1;
    if (n_tokens < 2 || !impl_ || impl_->force_fp32_matvec) {
        if (impl_) impl_->stats.matvec_fallback_calls += static_cast<uint64_t>(n_tokens);
        for (int t = 0; t < n_tokens; ++t) {
            matvec(static_cast<int>(weight.type), weight.d_data,
                   x_fp32 + static_cast<size_t>(t) * K,
                   y_fp32 + static_cast<size_t>(t) * N, K, N, stream);
        }
        return;
    }
    prepare_activation(x_fp32, n_tokens, K, stream);
    matmul_batch_prepared(weight, y_fp32, n_tokens, stream);
}

} // namespace strata::qwen38
