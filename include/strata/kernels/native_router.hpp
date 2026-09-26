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
// `native_router_top10` for n_tok tokens in one launch: logits (n_tok, 512) -> ids, weights (n_tok, 10).
void native_router_top10_multi(const float* logits, int32_t* ids, float* weights, int n_tok, void* stream);

// Plan v0.3 P6: a verify window layer's router in one kernel: bf16_gemv_fp32_mmvf_multi's logits,
// native_router_top10_multi's ids and weights and doorbell_publish's ring, bitwise, and verify_hit_plan's plan.  The
// mapped copy of x is written while the logits run; the ids and weights (and their mapped copies) once every token's
// logits are in; then the sequence number; then the plan.
struct VerifyRouterArgs {
    const float* x = nullptr;                          // (n_tok, n_embd)
    const uint16_t* w = nullptr;                       // the router: bf16 (n_expert, n_embd)
    float* logits = nullptr;                           // (n_tok, n_expert) scratch
    int32_t* ids = nullptr;                            // (n_tok, 10)
    float* weights = nullptr;                          // (n_tok, 10)
    float* x_out = nullptr;                            // mapped copy of x, or null
    int32_t* ids_out = nullptr;                        // mapped copies of ids and weights, or null
    float* w_out = nullptr;
    uint32_t* seq = nullptr;                           // mapped: set to `ring` once those copies are visible
    uint32_t ring = 0;                                 // (doorbell_publish's increment: the rings so far, fixed at capture)
    const int32_t* res = nullptr;                      // verify_hit_plan's arguments; plan null: no plan
    const unsigned long long* slot_ptr = nullptr;
    int32_t* plan = nullptr;
    int cap = 0, ptr_off = 0;
    unsigned* counter = nullptr;                       // device, zero before the first launch (each launch leaves it so)
    int n_tok = 0, n_embd = 0, n_expert = 0;           // 1..8 tokens, 512 experts, n_embd a multiple of 512
};
void verify_router(const VerifyRouterArgs& a, void* stream);
}
