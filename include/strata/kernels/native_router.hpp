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
/// STRATA_ROUTE_FAST (default on): the DPP router kernels and the fast publish (x null: the rows are on the host).
bool native_router_fast();
/// `seq_value` (STRATA_ROUTE_FAST): the step number the doorbell stores; 0 reads it and adds one (the old way).
/// STRATA_ROUTE_PUB: native_router_top10_multi (1-8 rows) and doorbell_publish_res in one launch: the same ids and
/// weights bit for bit, then the ids (and the n_x floats of x when an expert is not resident) to the mapped
/// buffers and the sequence number incremented.
void native_router_top10_multi_publish_res(const float* logits, int32_t* ids, float* weights, int n_tok, const float* x,
                                           const int32_t* d_res, int n_expert, int64_t n_x, float* x_out,
                                           int32_t* ids_out, uint32_t* d_seq, void* stream,
                                           uint32_t seq_value = 0);
/// STRATA_ROUTE_RESIDENT: swap non-resident tail-rank experts for resident ones within `margin` logits (see the .cu).
/// `stats` is nullptr or 4 device uint64 counters: tail entries seen, swaps, non-resident before, non-resident after.
void native_route_resident(const float* logits, int32_t* ids, float* weights, const int32_t* res_layer, int n_tok, float margin,
                           int rank_lo, int rank_hi, unsigned long long* stats, void* stream);
}
