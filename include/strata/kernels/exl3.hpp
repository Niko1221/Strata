// include/strata/kernels/exl3.hpp - GPU EXL3 reconstruct (docs/EXL3.md).
//
// Reconstructs the original-basis weight W = diag(suh) . H128 . W_hat . H128 . diag(svh) from a
// packed EXL3 trellis, on the GPU.  The codebook is procedural (no lookup table), as in ExLlamaV3.
#pragma once

#include <cstdint>

namespace strata::kernels {

// trellis: [ki, nj, 256*bits/16] packed; suh: ki*16 fp16; svh: nj*16 fp16.  cb: 0=3inst, 1=mcg, 2=mul1.
// out: ki*16 x nj*16 fp16, original basis.  stream may be null.
void exl3_reconstruct_weight(const uint16_t* trellis, int ki, int nj, int bits, int cb,
                             const uint16_t* suh, const uint16_t* svh, uint16_t* out, void* stream);

// Fused decode-time GEMV: y = H(x . suh) @ W_hat . H . svh, without ever materializing W.
// x: ki*16 fp16 (one token); y: nj*16 fp16.  stream may be null.
void exl3_gemv(const uint16_t* x, const uint16_t* suh, const uint16_t* svh, const uint16_t* trellis,
               int ki, int nj, int bits, int cb, uint16_t* y, void* stream);

// The same GEMV for the engine's f32 activations (converts x f32->fp16 and y fp16->f32).  This is the
// entry point `gemv_quantized` uses for an EXL3 dense linear.  stream may be null.
void exl3_gemv_f32(const float* x, const uint16_t* suh, const uint16_t* svh, const uint16_t* trellis,
                   int ki, int nj, int bits, int cb, float* y, void* stream);

// One EXL3 linear's tensors (see docs/EXL3.md).  cb: 0=3inst, 1=mcg, 2=mul1.
struct Exl3Mat {
    const uint16_t* trellis = nullptr;
    const uint16_t* suh = nullptr;
    const uint16_t* svh = nullptr;
    int ki = 0, nj = 0, bits = 0, cb = 2;
};

// One routed expert's FFN for a single token: out = down(silu(gate(x)) * up(x)), accumulated over
// `n_experts` experts weighted by `weights`.  x: ki*16 fp16; out: down[].nj*16 fp16.  stream may be null.
void exl3_moe_ffn(const Exl3Mat* gate, const Exl3Mat* up, const Exl3Mat* down, const float* weights,
                  int n_experts, const uint16_t* x, uint16_t* out, void* stream);

}  // namespace strata::kernels
