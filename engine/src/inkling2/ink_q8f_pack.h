// ink_q8f_pack.h - GGUF Q8_0 rows -> the fragment order of the CUDA int8 tensor-core GEMM (format 105, k_ink_gemm_q8f in cuda/ink_kernels.cuh).
// Host only, no CUDA types: the loader packs with it and the GPU test packs with it.
//
// Input: `rows` rows of a Q8_0 tensor, `cols` columns each (cols % 32 == 0), `rows` % 16 == 0; a row is cols/32 blocks of 34 bytes (fp16 d, 32 int8).
// Output (rows * cols int8 bytes, then rows * cols/32 fp16 scales):
//   qpack [rows/16][cols/32][32 lanes][16 B]: lane (g = lane >> 2, t = lane & 3) holds the four A registers of mma.m16n8k32 (s8)
//         a0 = row g, k t*4..+3   a1 = row g+8, k t*4..+3   a2 = row g, k 16+t*4..+3   a3 = row g+8, k 16+t*4..+3
//   scales [rows/16][cols/32][16] fp16: the block scale of each row of the tile.
#pragma once

#include <cstddef>
#include <cstdint>
#include <cstring>

namespace ink2 {

inline size_t ink_q8f_bytes(size_t rows, size_t cols) { return rows * cols + rows * (cols / 32) * 2; }

inline void ink_q8f_pack(const uint8_t* src, size_t rows, size_t cols, uint8_t* out) {
    const size_t nblk = cols / 32;
    const size_t rb = nblk * 34;
    uint8_t* q_out = out;
    uint8_t* sc_out = out + rows * cols;
    for (size_t rt = 0; rt < rows / 16; ++rt) {
        for (size_t kb = 0; kb < nblk; ++kb) {
            uint8_t* tile = q_out + (rt * nblk + kb) * 512;
            for (int lane = 0; lane < 32; ++lane) {
                const int g = lane >> 2, t = lane & 3;
                const uint8_t* s0 = src + (rt * 16 + g) * rb + kb * 34 + 2;
                const uint8_t* s1 = src + (rt * 16 + g + 8) * rb + kb * 34 + 2;
                uint8_t* o = tile + lane * 16;
                std::memcpy(o + 0, s0 + t * 4, 4);
                std::memcpy(o + 4, s1 + t * 4, 4);
                std::memcpy(o + 8, s0 + 16 + t * 4, 4);
                std::memcpy(o + 12, s1 + 16 + t * 4, 4);
            }
            for (int rr = 0; rr < 16; ++rr) {
                std::memcpy(sc_out + ((rt * nblk + kb) * 16 + rr) * 2, src + (rt * 16 + rr) * rb + kb * 34, 2);
            }
        }
    }
}

}  // namespace ink2
