#pragma once

#include "strata/hy4flash_gguf/mat.hpp"
#include <cstddef>

/// The inter-block hidden state consists of four parallel residual streams.
namespace strata::hy4_flash {

/// Scratch one hc_pre needs: the normalised row in F32 plus its Q8_1 form for the matvec.
/// `hc_width` is hc * hidden.
std::size_t hc_scratch_bytes(int hc_width, int n_tok);

/// Normalize each column out of place so the original residual stays available.
void hc_flat_rms(const float* x, float* out, int hc_width, int n_tok, float eps, void* stream);

/// Split the projected mixes into contiguous pre- and post-gates for each token.
void hc_gates(const float* mixes, const float* scale, const float* base, int hc, int n_tok, int n_out,
              float hc_eps, float magnitude, float* pre, float* post, void* stream);

/// reduced [hidden, n_tok] = sum over the hc streams of x[:, ih, :] * w[ih, :]. `x` is the ORIGINAL state.
void hc_reduce(const float* x, const float* w, int hidden, int hc, int n_tok, float* reduced, void* stream);

/// out [hidden, hc, n_tok] = residual[:, ih, :] + y[:, :] * post[ih, :]. `out` may alias `residual`.
void hc_post(const float* y, const float* residual, const float* post, int hidden, int hc, int n_tok,
             float* out, void* stream);

/// Enable fused hyper-connection normalization and quantization.
bool hc_fuse_on();

/// Fuse flat RMS normalization and activation quantization before computing the gates.
void hc_pre_rms_quant_fused(const float* x, void* q8, int hc_width, int n_tok, float eps, void* stream);

/// The whole pre half: rms -> quantize -> Q4_K matvec -> gates -> reduce. `fn` is the Q4_K matrix on the device,
/// `scratch` at least hc_scratch_bytes(hc * hidden, n_tok). `mixes` is [n_out, n_tok] of caller-owned device
/// floats (kept separate so a test can read it). Pass `post == nullptr` for the output reduce.
void hc_pre(const float* x, const void* fn, const float* scale, const float* base, int hidden, int hc,
            int n_tok, int n_out, float rms_eps, float hc_eps, float magnitude, void* scratch, float* mixes,
            float* pre, float* post, float* reduced, void* stream, MatCtx* mc = nullptr, int fn_type = 12);

}  // namespace strata::hy4_flash
