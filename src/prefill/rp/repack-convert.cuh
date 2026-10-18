// Vendored from mx-llama.cpp (github.com/mxxm-t/mx-llama.cpp, MIT) ggml/src/ggml-cuda/q8_repack/buffer.cu at c4d12e2c,
// lines 123-367 unchanged: the device-side repack kernels (GGUF blocks -> the repacked planes) for Q8_0, MXFP4,
// IQ4_NL, Q6_K, Q4_K, Q5_1 and Q5_K.  Launch arithmetic: buffer.cu's set_tensor (mirrored in rp_moe.cu).
#pragma once
#include "repack-common.cuh"


// Device-side repack: one thread per destination sub-block. Reads canonical
// block_q8_0 (34 B: half d, then 32 int8) and writes the two planes. Padding
// sub-blocks (blk >= n_blocks) are zeroed, matching repack_q8_0_host().
static __global__ void repack_q8_0_kernel(
        const uint8_t * __restrict__ src, uint8_t * __restrict__ dst,
        const int64_t ne1, const int64_t n_blocks, const int64_t qs_str,
        const int64_t qs_len, const int64_t src_stride, const int64_t dst_stride,
        const int64_t total) {
    for (int64_t i = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
         i < total; i += (int64_t) gridDim.x * blockDim.x) {
        const int64_t blk = i % n_blocks;
        const int64_t row = (i / n_blocks) % ne1;
        const int64_t e   = i / (n_blocks * ne1);

        uint8_t * d_qs = dst + e * dst_stride + row * qs_str + blk * 32;
        uint8_t * d_d  = dst + e * dst_stride + qs_len + (row * n_blocks + blk) * 2;

        const uint8_t * sb = src + e * src_stride + (row * n_blocks + blk) * 34;
#pragma unroll
        for (int k = 0; k < 32; k++) {
            d_qs[k] = sb[2 + k];
        }
        d_d[0] = sb[0];
        d_d[1] = sb[1];
    }
}

// Device-side MXFP4 repack: canonical block_mxfp4 (17 B: e8m0 byte, 16 nibble
// bytes) to de-aliased nibble rows + a 1-byte e plane. The de-alias gap bytes
// are never read (the kernels guard with sb < n_sub), matching the Q8_0 kernel.
static __global__ void repack_mxfp4_kernel(
        const uint8_t * __restrict__ src, uint8_t * __restrict__ dst,
        const int64_t ne1, const int64_t n_blocks, const int64_t qs_str,
        const int64_t qs_len, const int64_t src_stride, const int64_t dst_stride,
        const int64_t total) {
    for (int64_t i = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
         i < total; i += (int64_t) gridDim.x * blockDim.x) {
        const int64_t blk = i % n_blocks;
        const int64_t row = (i / n_blocks) % ne1;
        const int64_t e   = i / (n_blocks * ne1);

        uint8_t * d_qs = dst + e * dst_stride + row * qs_str + blk * 16;
        uint8_t * d_e  = dst + e * dst_stride + qs_len + row * n_blocks + blk;

        const uint8_t * sb = src + e * src_stride + (row * n_blocks + blk) * 17;
#pragma unroll
        for (int k = 0; k < 16; k++) {
            d_qs[k] = sb[1 + k];
        }
        d_e[0] = sb[0];
    }
}

// Device-side IQ4_NL repack: canonical block_iq4_nl (18 B: f16 d, 16 nibble bytes) to de-aliased nibble rows + an f16 plane, the MXFP4 kernel with a 2-byte scale.
static __global__ void repack_iq4nl_kernel(
        const uint8_t * __restrict__ src, uint8_t * __restrict__ dst,
        const int64_t ne1, const int64_t n_blocks, const int64_t qs_str,
        const int64_t qs_len, const int64_t src_stride, const int64_t dst_stride,
        const int64_t total) {
    for (int64_t i = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
         i < total; i += (int64_t) gridDim.x * blockDim.x) {
        const int64_t blk = i % n_blocks;
        const int64_t row = (i / n_blocks) % ne1;
        const int64_t e   = i / (n_blocks * ne1);

        uint8_t * d_qs = dst + e * dst_stride + row * qs_str + blk * 16;
        uint8_t * d_d  = dst + e * dst_stride + qs_len + (row * n_blocks + blk) * 2;

        const uint8_t * sb = src + e * src_stride + (row * n_blocks + blk) * 18;
#pragma unroll
        for (int k = 0; k < 16; k++) {
            d_qs[k] = sb[2 + k];
        }
        d_d[0] = sb[0];
        d_d[1] = sb[1];
    }
}

// Device-side Q6_K repack, one thread per 32-value sub-block: the host converter's de-interleave, lows on the de-aliased row stride, highs, scale pair, and the f16 d plane written by the sub-block that owns k == 0 of its super-block.
static __global__ void repack_q6k_kernel(
        const uint8_t * __restrict__ src, uint8_t * __restrict__ dst,
        const int64_t ne1, const int64_t n_sub, const int64_t qs_str,
        const int64_t hoff, const int64_t soff, const int64_t doff, const int64_t nds,
        const int64_t src_stride, const int64_t dst_stride, const int64_t total) {
    for (int64_t i = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
         i < total; i += (int64_t) gridDim.x * blockDim.x) {
        const int64_t sb  = i % n_sub;
        const int64_t row = (i / n_sub) % ne1;
        const int64_t e   = i / (n_sub * ne1);
        const block_q6_K * b = reinterpret_cast<const block_q6_K *>(src + e * src_stride) + row * (n_sub / 8) + (sb >> 3);
        const int k = (int)(sb & 7);
        uint8_t raw[32];
#pragma unroll
        for (int j = 0; j < 32; j++) {
            const int v = k * 32 + j;
            const int h = v >> 7;
            const int l = v & 127;
            const uint8_t * ql = b->ql + 64 * h;
            const uint8_t * qh = b->qh + 32 * h;
            uint8_t r6;
            if      (l <  32) { r6 = (ql[l]      & 0xF) | (((qh[l]      >> 0) & 3) << 4); }
            else if (l <  64) { r6 = (ql[l]      & 0xF) | (((qh[l - 32] >> 2) & 3) << 4); }
            else if (l <  96) { r6 = (ql[l - 64] >>  4) | (((qh[l - 64] >> 4) & 3) << 4); }
            else              { r6 = (ql[l - 64] >>  4) | (((qh[l - 96] >> 6) & 3) << 4); }
            raw[j] = r6;
        }
        uint8_t * base  = dst + e * dst_stride;
        const int64_t wi = row * n_sub + sb;
        uint8_t * lows  = base + row * qs_str + sb * 16;
        uint8_t * highs = base + hoff + wi * 8;
#pragma unroll
        for (int j = 0; j < 16; j++) {
            lows[j] = (uint8_t)((raw[j] & 0xF) | ((raw[j + 16] & 0xF) << 4));
        }
#pragma unroll
        for (int m = 0; m < 8; m++) {
            const int bs = (m & 3) * 4 + (m >> 2) * 16;
            highs[m] = (uint8_t)( ((raw[bs + 0] >> 4) << 0) | ((raw[bs + 1] >> 4) << 2)
                                | ((raw[bs + 2] >> 4) << 4) | ((raw[bs + 3] >> 4) << 6));
        }
        base[soff + wi * 2 + 0] = (uint8_t) b->scales[2 * k + 0];
        base[soff + wi * 2 + 1] = (uint8_t) b->scales[2 * k + 1];
        if (k == 0) {
            const uint8_t * dp = reinterpret_cast<const uint8_t *>(&b->d);
            base[doff + (row * nds + (sb >> 3)) * 2 + 0] = dp[0];
            base[doff + (row * nds + (sb >> 3)) * 2 + 1] = dp[1];
        }
    }
}

// Device-side Q4_K repack, one thread per 32-value sub-block: the host converter's nibble pick, packed on the de-aliased row stride, the scale record and the half2 {d, dmin} written by the sub-block that owns k == 0 of its super-block.
static __global__ void repack_q4k_kernel(
        const uint8_t * __restrict__ src, uint8_t * __restrict__ dst,
        const int64_t ne1, const int64_t n_sub, const int64_t qs_str,
        const int64_t soff, const int64_t doff, const int64_t nsb,
        const int64_t src_stride, const int64_t dst_stride, const int64_t total) {
    for (int64_t i = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
         i < total; i += (int64_t) gridDim.x * blockDim.x) {
        const int64_t sb  = i % n_sub;
        const int64_t row = (i / n_sub) % ne1;
        const int64_t e   = i / (n_sub * ne1);
        const block_q4_K * b = reinterpret_cast<const block_q4_K *>(src + e * src_stride) + row * (n_sub / 8) + (sb >> 3);
        const int k = (int)(sb & 7);
        const uint8_t * q = b->qs + 32 * (k >> 1);
        const int shift = (k & 1) * 4;
        uint8_t * base = dst + e * dst_stride;
        uint8_t * lows = base + row * qs_str + sb * 16;
#pragma unroll
        for (int j = 0; j < 16; j++) {
            const uint8_t v0 = (q[j]      >> shift) & 0xF;
            const uint8_t v1 = (q[j + 16] >> shift) & 0xF;
            lows[j] = (uint8_t)(v0 | (v1 << 4));
        }
        if (k == 0) {
            const int64_t sbk = row * nsb + (sb >> 3);
            const uint8_t * bp = reinterpret_cast<const uint8_t *>(b);
#pragma unroll
            for (int j = 0; j < 12; j++) {
                base[soff + sbk * 12 + j] = b->scales[j];
            }
#pragma unroll
            for (int j = 0; j < 4; j++) {
                base[doff + sbk * 4 + j] = bp[j];
            }
        }
    }
}

// Device-side Q5_1 repack, one thread per 32-value block: the canonical nibble bytes, the fifth-bit word and the half2 {d, m} are copied into their planes.
static __global__ void repack_q51_kernel(
        const uint8_t * __restrict__ src, uint8_t * __restrict__ dst,
        const int64_t ne1, const int64_t n_sub, const int64_t qs_str,
        const int64_t hoff, const int64_t doff,
        const int64_t src_stride, const int64_t dst_stride, const int64_t total) {
    for (int64_t i = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
         i < total; i += (int64_t) gridDim.x * blockDim.x) {
        const int64_t sb  = i % n_sub;
        const int64_t row = (i / n_sub) % ne1;
        const int64_t e   = i / (n_sub * ne1);
        const block_q5_1 * b = reinterpret_cast<const block_q5_1 *>(src + e * src_stride) + row * n_sub + sb;
        uint8_t * base = dst + e * dst_stride;
        const int64_t wi = row * n_sub + sb;
        uint8_t * lows = base + row * qs_str + sb * 16;
#pragma unroll
        for (int j = 0; j < 16; j++) {
            lows[j] = b->qs[j];
        }
#pragma unroll
        for (int j = 0; j < 4; j++) {
            base[hoff + wi * 4 + j] = b->qh[j];
        }
        const uint8_t * bp = reinterpret_cast<const uint8_t *>(b);
#pragma unroll
        for (int j = 0; j < 4; j++) {
            base[doff + wi * 4 + j] = bp[j];
        }
    }
}

// Device-side Q5_K repack, one thread per 32-value sub-block: Q4_K's nibble pick plus the fifth-bit word gathered from qh, the scale record and {d, dmin} by k == 0.
static __global__ void repack_q5k_kernel(
        const uint8_t * __restrict__ src, uint8_t * __restrict__ dst,
        const int64_t ne1, const int64_t n_sub, const int64_t qs_str,
        const int64_t hoff, const int64_t soff, const int64_t doff, const int64_t nsb,
        const int64_t src_stride, const int64_t dst_stride, const int64_t total) {
    for (int64_t i = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
         i < total; i += (int64_t) gridDim.x * blockDim.x) {
        const int64_t sb  = i % n_sub;
        const int64_t row = (i / n_sub) % ne1;
        const int64_t e   = i / (n_sub * ne1);
        const block_q5_K * b = reinterpret_cast<const block_q5_K *>(src + e * src_stride) + row * (n_sub / 8) + (sb >> 3);
        const int k = (int)(sb & 7);
        const uint8_t * q = b->qs + 32 * (k >> 1);
        const int shift = (k & 1) * 4;
        const int hbit  = 2 * (k >> 1) + (k & 1);
        uint8_t * base = dst + e * dst_stride;
        const int64_t wi = row * n_sub + sb;
        uint8_t * lows = base + row * qs_str + sb * 16;
        uint32_t h = 0;
#pragma unroll
        for (int j = 0; j < 16; j++) {
            const uint8_t v0 = (q[j]      >> shift) & 0xF;
            const uint8_t v1 = (q[j + 16] >> shift) & 0xF;
            lows[j] = (uint8_t)(v0 | (v1 << 4));
        }
#pragma unroll
        for (int j = 0; j < 32; j++) {
            h |= (uint32_t)((b->qh[j] >> hbit) & 1) << j;
        }
        *reinterpret_cast<uint32_t *>(base + hoff + wi * 4) = h;
        if (k == 0) {
            const int64_t sbk = row * nsb + (sb >> 3);
            const uint8_t * bp = reinterpret_cast<const uint8_t *>(b);
#pragma unroll
            for (int j = 0; j < 12; j++) {
                base[soff + sbk * 12 + j] = b->scales[j];
            }
#pragma unroll
            for (int j = 0; j < 4; j++) {
                base[doff + sbk * 4 + j] = bp[j];
            }
        }
    }
}
