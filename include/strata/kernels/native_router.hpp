#pragma once
#include <cstdint>

namespace strata::kernels {
// Configure before session capture. Existing graphs retain their selected kernels.
// This experiment is off by default and does not modify the legacy router.
void native_router_set_enabled(bool enabled);
bool native_router_enabled();

// Pinned CUDA topk-moe contract for ONE token, top 10, softmax,
// no selection bias, lower normalization clamp 2^-14, and scale 1.
// Reads `n_expert` finite F32 logits (32..512 in multiples of 32); writes 10 I32 IDs and 10 F32 weights. Equal
// computed probabilities select the lower expert index. All spans must be
// four-byte aligned and outputs disjoint from each other and the input.
// Requires a nonnull ordered CUDA stream. No allocation or synchronization.
//
// `n_expert` exists because the Coder release is Qwen3.8-Flash-Next with half of its
// routed experts pruned away (256 of 512, ISTA-DASLab's RCO), so a hard-coded 512
// would read past the router's `[n_embd, 256]` weight: the token's own row plus the
// next token's logits. The count is the model's, and the caller has it in `ModelGeometry`.
void native_router_top10(const float* logits, int32_t* ids, float* weights, int n_expert, void* stream);
/// n_tok rows at once (logits [n,512], ids/weights [n,10]); each row exactly as the single call.
void native_router_top10_multi(const float* logits, int32_t* ids, float* weights, int n_tok, void* stream);
}
