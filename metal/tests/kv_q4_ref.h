// metal/tests/kv_q4_ref.h - the CPU reference both kv_q4 tests share (tests/kv_q4_parity.cpp's kernel path and
// tests/kv_q4_engine.cpp's engine path).  It replicates the DEVICE arithmetic exactly: the same fwht register
// butterfly, the same group-maximum tie rule (a tie in |x| resolves to the LARGER value in every lane - not the
// first strict maximum of src/kernels/kv_q4_parity.cpp:68), the same int(fma(x, id, 8.5f)) contraction and the
// same software round-to-nearest-even f32 -> f16 bits.
#pragma once

#include "../../include/strata/kernels/f16_bits.hpp"

#include <cmath>
#include <cstdint>
#include <vector>

namespace kvq4ref {

// include/strata/kernels/kv_q4.hpp's block (18 packed bytes), a local copy like kv_q4.metal's: this header's
// only shared include is the one it exercises (f16_bits.hpp, by relative path), and kv_q4.hpp itself would drag
// in the whole qsa.hpp contract for an 18-byte layout
#pragma pack(push, 1)
struct block_q4_0 {
    uint16_t d;
    uint8_t qs[16];
};
#pragma pack(pop)
static_assert(sizeof(block_q4_0) == 18, "block_q4_0 must be 18 bytes");

// the fixture geometry both tests use: kv heads, head dim, page size, blocks per head, bytes per (head, cell) row
constexpr int H = 2, D = 256, P = 64, BPH = D / 32, ROW_BYTES = BPH * 18;

// the device fwht, one row per (imaginary) simdgroup: 32 lanes x 8 registers, the low 5 index bits across lanes
// (the xor exchange reads every lane's pre-step values), the high 3 within each lane's registers - the kernel's
// exact order, so equal floats in give equal bits out.
inline void fwht256_ref(const float* src, float* dst, int64_t n_rows) {
    constexpr float scale = 1.0f / 16.0f;
    for (int64_t r = 0; r < n_rows; ++r) {
        float reg[32][8];
        for (int lane = 0; lane < 32; ++lane)
            for (int i = 0; i < 8; ++i) reg[lane][i] = src[r * 256 + i * 32 + lane] * scale;
        for (int h = 1; h < 32; h *= 2) {
            for (int j = 0; j < 8; ++j) {
                float vals[32];
                for (int lane = 0; lane < 32; ++lane) vals[lane] = reg[lane][j];   // what the shuffle reads
                for (int lane = 0; lane < 32; ++lane) {
                    const float val = vals[lane], val2 = vals[lane ^ h];
                    reg[lane][j] = (lane & h) == 0 ? val + val2 : val2 - val;
                }
            }
        }
        for (int h = 32; h < 256; h *= 2) {
            const int step = h / 32;
            for (int j = 0; j < 8; j += 2 * step)
                for (int kk = 0; kk < step; ++kk)
                    for (int lane = 0; lane < 32; ++lane) {
                        const float x = reg[lane][j + kk], y = reg[lane][j + kk + step];
                        reg[lane][j + kk] = x + y;
                        reg[lane][j + kk + step] = x - y;
                    }
        }
        for (int lane = 0; lane < 32; ++lane)
            for (int i = 0; i < 8; ++i) dst[r * 256 + i * 32 + lane] = reg[lane][i];
    }
}

// one 32-value block, the device tie rule: butterflies over xor offsets 16, 8, 4, 2, 1 of (amax, mval), taking
// the partner's pair when it is strictly larger in |x|, or equal in |x| and larger in value.
inline void q4_block_ref(const float* x, block_q4_0& blk) {
    float amax[32], mval[32];
    for (int t = 0; t < 32; ++t) {
        amax[t] = std::fabs(x[t]);
        mval[t] = x[t];
    }
    for (int o = 16; o > 0; o >>= 1) {
        float a2[32], v2[32];
        for (int t = 0; t < 32; ++t) {
            a2[t] = amax[t ^ o];
            v2[t] = mval[t ^ o];
        }
        for (int t = 0; t < 32; ++t)
            if (a2[t] > amax[t] || (a2[t] == amax[t] && v2[t] > mval[t])) {
                amax[t] = a2[t];
                mval[t] = v2[t];
            }
    }
    const float d = mval[0] / -8.0f;   // every lane converged to the same pair
    const float id = d != 0.0f ? 1.0f / d : 0.0f;
    blk.d = strata::kernels::f16_from_f32(d);
    for (int t = 0; t < 16; ++t) {
        const int q0 = (int) std::fma(x[t], id, 8.5f);
        const int q1 = (int) std::fma(x[t + 16], id, 8.5f);
        const uint8_t c0 = (uint8_t) (q0 < 0 ? 0 : (q0 > 15 ? 15 : q0));
        const uint8_t c1 = (uint8_t) (q1 < 0 ? 0 : (q1 > 15 ? 15 : q1));
        blk.qs[t] = (uint8_t) (c0 | (c1 << 4));
    }
}

// one cell's K row and V row (H * D floats each, already rotated) into their pools at `pos`: through the page
// table for the device pools, the identity layout for the host / stage copies.  A null row leaves its pool alone.
inline void append_cell_ref(const float* kcell, const float* vcell, const std::vector<int32_t>& table,
                            std::vector<uint8_t>& kpool, std::vector<uint8_t>& vpool, bool identity_rows,
                            int64_t pos) {
    for (int h = 0; h < H; ++h)
        for (int b = 0; b < BPH; ++b) {
            const int64_t page = table[pos / P];
            const int64_t row = identity_rows ? ((pos / P) * H + h) * P + pos % P
                                              : ((page * H + h) * P + pos % P);
            if (kcell) q4_block_ref(kcell + (int64_t) h * D + b * 32, *(block_q4_0*) (kpool.data() + row * ROW_BYTES + b * 18));
            if (vcell) q4_block_ref(vcell + (int64_t) h * D + b * 32, *(block_q4_0*) (vpool.data() + row * ROW_BYTES + b * 18));
        }
}

}  // namespace kvq4ref
