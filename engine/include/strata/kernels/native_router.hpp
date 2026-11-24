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
/// The same contract for a model family's (experts, top-k): 512 / 10 (Flash-Next: the two calls above, unchanged) and
/// 256 / 8 (Qwen3.6-35B-A3B).  logits [n_tok, n_expert], ids / weights [n_tok, k].  CUDA / HIP.
bool native_router_topk_supported(int n_expert, int k);
void native_router_topk(const float* logits, int32_t* ids, float* weights, int n_expert, int k, int n_tok, void* stream);
/// Selection bias affects expert ranking. Combine weights use the original sigmoid scores.
bool native_router_sigmoid_supported(int n_expert, int k);
void native_router_sigmoid(const float* logits, const float* bias, int32_t* ids, float* weights, int n_expert,
                           int k, int n_tok, float scale, void* stream);
// Private T1 router/publication fusion. Plain pointer fields avoid a hy4-flash engine dependency.
struct NativeRouterPublishPlan {
    volatile uint32_t* request;
    volatile uint32_t* fast_epoch;
    volatile uint32_t* error;
    volatile int32_t* host_ids;
    volatile int32_t* host_hit;
    const int32_t* slot_of;
    int32_t* hit;
    int32_t* miss;
    int n_expert;
    int permanent_slots;
    uint32_t* device_epoch = nullptr; // optional device-owned [request, fast_epoch]
};
void native_router_sigmoid_publish(const float* logits, const float* bias, int32_t* ids,
    float* weights, float scale, const NativeRouterPublishPlan& plan, void* stream);
/// STRATA_ROUTE_RESIDENT: swap non-resident tail-rank experts for resident ones within `margin` logits (see the .cu).
/// `stats` is nullptr or 4 device uint64 counters: tail entries seen, swaps, non-resident before, non-resident after.
void native_route_resident(const float* logits, int32_t* ids, float* weights, const int32_t* res_layer, int n_tok, float margin,
                           int rank_lo, int rank_hi, unsigned long long* stats, void* stream);
}
