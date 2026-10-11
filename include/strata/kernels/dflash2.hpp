// Experimental DFlash2 primitives. Layout follows llama.cpp a9d27ac / z-lab 07ebd93.
#pragma once
#include <cstdint>

namespace strata::kernels {
// Out-of-place grouped causal convolution. Each block of block_rows rows has its own left padding.
// dynamic: [rows][2][kernel][hidden/group], base: [2][kernel][hidden].
void dflash2_convolve(const float* x, const float* dynamic, const float* base, float* out,
                     int rows, int hidden, int kernel, int group, int side, int block_rows, void* stream);
// Top-k unary candidates for each row, with deterministic smaller-token tie breaking.
void dflash2_topk(const float* logits, int32_t* ids, float* unary, int rows, int vocab, int top_k, void* stream);
// Greedy predecessor-conditioned walk over rows 1..rows-1 (row 0 is the anchor).
// Codebooks [vocab][rank], hidden projection [rows][rank]. Writes rows-1 proposals and optional
// conditional top-k softmax confidence (a gate, not the target sampling distribution).
void dflash2_select(const int32_t* candidates, const float* unary, const float* hidden,
                    const float* predecessor, const float* successor, const int32_t* anchor,
                    int32_t* out, float* confidence, int rows, int top_k, int rank, void* stream);
}  // namespace strata::kernels
