// src/prefill/moe_f16_kq.hpp - gfx11: the K-quant prompt experts on the F16 matrix cores (moe_f16_kq.cu; the switch,
// STRATA_PF_F16=1 with STRATA_PF_FUSED_KQ=1, is moe_f16_common.cuh's).
// Internal to moe_fused_iq.cu, which owns the native experts' tables and launches.
#pragma once

#include <cstdint>

#include <cuda_runtime.h>

#include "strata/prefill/moe_fused.hpp"
#include "strata/prefill/moe_fused_iq.hpp"

namespace strata::prefill::fused::f16kq {

#if defined(STRATA_HIP_GFX906)
// gfx906 compiles the CUDA targets' source list as HIP, which leaves moe_f16_kq.cu out (its kernels are gfx11's, and
// gfx906 never takes the native experts' HIP launch): nothing is covered there.
inline bool covers(const Batch&, const NativeGeom&) { return false; }
inline bool covers_down(const Batch&, const NativeGeom&) { return false; }
inline void gate_up(const Batch&, const NativeGeom&, const int32_t*, const int2*, const int32_t*, int64_t, int, int64_t,
                    const uint8_t*, const int32_t*, uint8_t*, cudaStream_t) { __builtin_trap(); }
inline void down(const Batch&, const NativeGeom&, const int32_t*, const int2*, const int32_t*, int64_t, int, int64_t,
                 const uint8_t*, float*, cudaStream_t) { __builtin_trap(); }
#else
/// The gate/up type has a kernel here, and the batch's weights are where its
/// 16-byte reads can take them: every blob, g.up_off and g.gu_row 16-byte aligned.  The native blobs are only promised
/// 2-byte alignment (moe_fused_iq.hpp); otherwise the caller runs native_w11_kernel, which reads 16-bit words.
bool covers(const Batch& b, const NativeGeom& g);

/// Gate/up, SwiGLU and H for experts [b.e0, b.e1): the same tiles, activations and H as native_w11_kernel<WT, true>.
/// `ts`, `tiles`, `off` are fused::group's tables (at most `work_bound` tiles); `n` the layer's routed rows over
/// `n_expert` experts; `act` the int8 activations (quantize_act_native); `out` H.
void gate_up(const Batch& b, const NativeGeom& g, const int32_t* ts, const int2* tiles, const int32_t* off, int64_t n,
             int n_expert, int64_t work_bound, const uint8_t* act, const int32_t* src, uint8_t* out, cudaStream_t s);

/// The down type has a kernel here (Q5_1), and every blob, g.down_off and g.d_row are 8-byte aligned (its reads are 8 bytes).
bool covers_down(const Batch& b, const NativeGeom& g);
/// Down for the same experts and tiles: H (int8, `h`) to the routed rows' FP32 outputs in `dm`.
void down(const Batch& b, const NativeGeom& g, const int32_t* ts, const int2* tiles, const int32_t* off, int64_t n, int n_expert,
          int64_t work_bound, const uint8_t* h, float* dm, cudaStream_t s);
#endif

}  // namespace strata::prefill::fused::f16kq
