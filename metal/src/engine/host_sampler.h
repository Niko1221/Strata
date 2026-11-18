// metal/src/engine/host_sampler.h - the request's sampling on the Mac, on the CPU over the window's head logits.
//
// The CUDA engine's sampler (src/kernels/cuda/sampler.cu, sample_tokens) row for row: on unified memory the head
// logits are host memory once the window has finished, so the Mac picks the token where it reads it.  Same chain,
// same order (llama.cpp's): logit_bias, penalties, top_k (1..64; 0 or wider = 64), top_p, min_p, temperature, one
// Philox 4x32-10 draw at (seed, counter).  The greedy pick (temperature <= 0) is the argmax after the bias and the
// penalties, ties to the lowest id.  metal/tests/engine/host_sampler_parity.cpp pins it against a serial
// transcription of the CUDA kernel, draw for draw, and times it.
#pragma once

#include "strata/kernels/sampler.hpp"

#include <cstdint>

namespace strata::metal_engine {

/// One row's token.  `logits`: n_vocab floats (host memory).  `history`: the row's penalty history, `history_len`
/// ids, most recent last, -1 for unused front slots (strata::kernels::penalty_rows); only the last
/// min(p.penalty_last_n, history_len) are counted; null or 0 = no penalties.  `logit_bias`: n_vocab floats or null
/// (-inf bans).  `counter`: the draw's Philox counter, the position the row predicts from (the solo window's
/// pos0 + t, a slot row's position).  p.logit_bias and p.counter are not read (device pointer / per-call).
int sample_row(const float* logits, int n_vocab, const int32_t* history, int history_len,
               const strata::kernels::SamplerParams& p, const float* logit_bias, uint64_t counter);

/// The Philox 4x32-10 uniform in [0, 1) of the CUDA sampler (24 bits), for the tests.
float philox_uniform(uint64_t seed, uint64_t counter);

}  // namespace strata::metal_engine
