// include/strata/program/kv_q8_quant.hpp - header-only, CPU-mirror of the int8 K/V quant writer.
// (Phase 3: lives in the program/ tree, alongside the splice helpers it serves, rather than
// the kernels/ tree which is sparse-checkout in this repo and only touched by the GPU parity
// suite.  The file is header-only with no CUDA includes, so a CUDA TU can pull it in.)
//
// WHY THIS EXISTS.  The engine's INT8 K/V pool writer (src/kernels/cuda/kv_q8.cu:47-67
// for K, mirrored for V) stores per group of 64 values:
//     amax  = max|x| over the 64 values
//     sf    = f32_from_f16(f16_from_f32(amax / 127))                  // quantize against the STORED scale
//     code  = clamp(rint(x / sf), -127, 127) when sf > 0, else 0
//     block = (codes[64], scale_bits)
//
// Phase 3 (int8 splice) needs the EXACT same quant math to: (a) dequant a cached block
// back to fp32, (b) apply the splice's RoPE delta rotation, (c) requant the rotated image
// back to the same block format.  If the requant differs from the writer by a rounding
// choice, the splice pays a per-group half-step on every rotated cell, which the existing
// fresh-append path already pays and the model's downstream tolerance is sized for.
//
// The helpers here are inline so T11 (CPU test) and the runtime splice (host loop) call
// the SAME code: no separate "host reference" copy.  This is a header-only quant module
// with no CUDA includes, identical conventions to the host reference in
// src/kernels/kv_hybrid_parity.cpp:99-116 (which is BIT-EXACT with the kernel writer;
// the GPU parity suite pins that).  The function inlines we use here match the same
// rounding rules: f16_from_f32 = round-to-nearest-even, rint = ties-to-even
// (std::nearbyint == rint; both round half-to-even on the host).
//
// Layout invariant (kv_q8.hpp:6, kvr_q8_layers: 32 B of scales per cell): one scale per
// group of 64 values, stored next to the codes by the writer thread t==0
// (src/kernels/cuda/kv_q8.cu:67).  The mirror here writes one scale per group too.
#pragma once

#include "strata/kernels/f16_bits.hpp"     // f16_from_f32 / f32_from_f16 (kv_q8.cu uses the same)
#include "strata/kernels/kv_q8.hpp"       // KV_Q8_GROUP (64)

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>

namespace strata::kernels {

/// One quantized int8 group (64 values + one fp16 scale bit pattern), matching the
/// writer's `block_q8_K`-style layout.  Packed because every splice call needs to read
/// or write 64 + 2 bytes in one place; the writer stores these in two separate arrays
/// (`k_q` codes and `k_scale` scales), the dequant/requant here takes them as a
/// transient struct and unpacks/repacks at the call site.
struct KvQ8Group {
    int8_t  codes[KV_Q8_GROUP] = {};   ///> 64 int8 codes (clamped to [-127, 127])
    uint16_t scale_bits = 0;            ///> fp16 of (amax / 127) for the group (kv_q8.cu:55)
};

/// Quantize 64 fp32 values into a KvQ8Group.  Mirrors src/kernels/cuda/kv_q8.cu:47-67
/// exactly: amax over the group, scale = f16(f32(amax/127)), each code = clamp(rint(x/sf)).
inline void kv_q8_quant_group(const float* x, KvQ8Group& out) {
    // amax over the group (the writer: kv_q8.cu:50-54, with __shfl_xor reduction over
    // 64 threads; the group is small enough that a single linear scan is equivalent).
    float amax = 0.0f;
    for (int t = 0; t < KV_Q8_GROUP; ++t) {
        const float a = std::fabs(x[t]);
        if (a > amax) amax = a;
    }
    const uint16_t sb = f16_from_f32(amax / 127.0f);
    const float sf = f32_from_f16(sb);                                // quantize against STORED scale
    out.scale_bits = sb;
    for (int t = 0; t < KV_Q8_GROUP; ++t) {
        int q = 0;
        if (sf > 0.0f) {
            // kv_q8.cu:59: __float2int_rn = round-to-nearest-even (RTNE).  std::nearbyint
            // is RTNE too; std::rint is RTNE; the names are aliases on x86_64.  Using
            // std::nearbyint explicitly to match the GPU kernel's __float2int_rn.
            q = (int) std::nearbyint(x[t] / sf);
            if (q < -127) q = -127;
            else if (q > 127) q = 127;
        }
        out.codes[t] = (int8_t) q;
    }
}

/// Dequantize one KvQ8Group to 64 fp32 values.  Symmetric to the writer: x' = f32(code * sf)
/// (kv_q8.cu:100-101 mirror: the gather kernel dequantizes into fp16; the splice runs in
/// fp32 and writes through f16 at the end of the rotation step).
inline void kv_q8_dequant_group(const KvQ8Group& g, float* out) {
    const float sf = f32_from_f16(g.scale_bits);
    for (int t = 0; t < KV_Q8_GROUP; ++t)
        out[t] = (float) g.codes[t] * sf;
}

/// Read a KvQ8Group from a (codes, scale_bits) buffer pair (the writer's storage shape
/// in QsaState: int8_t* k_q, uint16_t* k_scale).  The (row, group) addressing is:
///   - codes:   row-major, head_dim per row, KV_Q8_GROUP per group
///   - scales:  row-major, head_dim / KV_Q8_GROUP per row, 1 per group
inline void kv_q8_read_group(const int8_t* codes, const uint16_t* scale_bits,
                              int head_dim, int row, int group, KvQ8Group& out) {
    out.scale_bits = scale_bits[row * (head_dim / KV_Q8_GROUP) + group];
    for (int t = 0; t < KV_Q8_GROUP; ++t)
        out.codes[t] = codes[row * head_dim + group * KV_Q8_GROUP + t];
}

/// Write a KvQ8Group back to the writer's storage shape.  Counterpart to
/// `kv_q8_read_group`.  The two helpers in this form let the splice reuse the same
/// dequant/rotate/requant pipeline as the parity test's host reference without a
/// second copy of the addressing arithmetic.
inline void kv_q8_write_group(int8_t* codes, uint16_t* scale_bits,
                               int head_dim, int row, int group, const KvQ8Group& in) {
    scale_bits[row * (head_dim / KV_Q8_GROUP) + group] = in.scale_bits;
    for (int t = 0; t < KV_Q8_GROUP; ++t)
        codes[row * head_dim + group * KV_Q8_GROUP + t] = in.codes[t];
}

}  // namespace strata::kernels
