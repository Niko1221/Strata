#pragma once
#include <cstdint>

namespace strata::kernels {
// Configure before session capture. Existing graphs retain their selected kernels.
// This experiment is off by default and does not modify the legacy router.
void native_router_set_enabled(bool enabled);
bool native_router_enabled();

// Pinned CUDA topk-moe contract for ONE token, 512 experts, top 10, softmax,
// no selection bias, lower normalization clamp 2^-14, and scale 1.
// Reads 512 finite F32 logits; writes 10 I32 IDs and 10 F32 weights. Equal
// computed probabilities select the lower expert index. All spans must be
// four-byte aligned and outputs disjoint from each other and the input.
// Requires a nonnull ordered CUDA stream. No allocation or synchronization.
void native_router_top10(const float* logits, int32_t* ids, float* weights, void* stream);
/// n_tok rows at once (logits [n,512], ids/weights [n,10]); each row exactly as the single call.
void native_router_top10_multi(const float* logits, int32_t* ids, float* weights, int n_tok, void* stream);
/// STRATA_ROUTE_RESIDENT (opt-in, changes output): swap non-resident tail-rank experts for resident ones within
/// `margin` logits. Supports only (n_expert,k) = (512,10) or (256,8). Reads logits [n_tok,n_expert] and residency
/// [n_expert] (negative = non-resident); updates IDs/weights [n_tok,k]. IDs must be distinct and in [0,n_expert).
/// Ranks are zero-based and require 0 <= rank_lo <= rank_hi < k; margin must be finite and nonnegative.
/// No swap leaves IDs/weights bit-identical. Requires aligned, disjoint buffers and a nonnull ordered stream.
/// `stats` is nullptr or 4 device uint64 counters: tail entries seen, swaps, non-resident before, non-resident after.
void native_route_resident(const float* logits, int32_t* ids, float* weights, const int32_t* res_layer, int n_tok,
                           int n_expert, int k, float margin, int rank_lo, int rank_hi,
                           unsigned long long* stats, void* stream);
}
