// src/qwen35/memt_kernels.cuh - MemT-1.0's Native Memory Block on CUDA.
//
// The engine does the three linear projections (in_proj on the native MMVQ, f_down
// and g_down on the FP32 GEMM) and hands this header the pieces; the kernels here
// are the gates, the snapshot read and the state update.  They are the CUDA
// transcription of engine/src/qwen35/memt_math.h; memt_pool_test and the Python
// oracle pin the same arithmetic on the CPU.
//
// All buffers are FP32 and row-major.  `proj` is [tokens][5*H*D + 2*R]:
// q | k | v | erase_raw | write_raw | decay_latent | out_gate_latent.
#pragma once

#include <cstdint>

namespace strata::qwen35::memt {

/// decay[t, h, d] = -exp(decay_scale_log[h]) * softplus(fdown[t, h, d] + bias[h, d]).
/// `fdown` comes from the caller's f_down GEMM; `decay` may be the same buffer.
void gates(const float* fdown, const float* bias, const float* decay_scale_log, float* decay,
           int tokens, int heads, int dim, void* stream);

/// The decay and out-gate latents, gathered out of proj's wide rows into
/// contiguous [tokens][rank] blocks for the two FP32 GEMMs.
void latents(const float* proj, float* decay_latent, float* og_latent, int tokens, int64_t proj_stride,
             int64_t head_dim, int rank, void* stream);

/// k_eff[t, h, :] = l2norm(k[t, h, :]) * erase, erase = exp(0.5 * logsigmoid(erase_raw)).
void key_effective(const float* proj, float* k_eff, int tokens, int heads, int dim, int64_t proj_stride,
                   void* stream);

/// att[t, h, :] += gated_rmsnorm((l2norm(q[t, h, :]) @ state[h]) * dim^-0.5, gate[t, h, :]).
/// `state` is [H][dim][dim], FP32; `proj` supplies q, `og` is the g_down result
/// and `o_norm` the read norm's weight; `att` is the o_proj input.
void read(const float* proj, const float* og, const float* o_norm, const float* state, float* att,
          int tokens, int heads, int dim, int64_t proj_stride, float eps, float softcap, void* stream);

/// One token at a time: S <- diag(exp(log_decay)) S; kv = S^T k_eff;
/// delta = write*v - kv; S += k_eff delta^T.  `state` is advanced in place.
void update(const float* proj, const float* k_eff, const float* decay, float* state, int tokens, int heads,
            int dim, int64_t proj_stride, void* stream);

}  // namespace strata::qwen35::memt
