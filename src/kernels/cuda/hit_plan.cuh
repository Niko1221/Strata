// src/kernels/cuda/hit_plan.cuh - the verify window's main-GPU hit plan (verify_kernels.hpp, verify_hit_plan) as the
// work of one block, for verify_kernels.cu's kernel and the router kernel that rings the doorbell (native_router.cu).
#pragma once

#include <cuda_runtime.h>

#include <cstdint>

namespace strata::kernels {
namespace {

constexpr int kHitPlanMax = 128;

// A block of at least kHitPlanMax threads, thread i = routed entry i (n <= kHitPlanMax).  A group per distinct resident
// expert, in the order of their first entries; a group's entries in routing order.  `ids` must be visible to the whole
// block.
__device__ __forceinline__ void hit_plan_block(const int32_t* __restrict__ ids, int n, int k,
                                               const int32_t* __restrict__ res, int n_expert,
                                               const unsigned long long* __restrict__ slot_ptr,
                                               int32_t* __restrict__ plan, int cap, int ptr_off) {
    __shared__ int32_t s_id[kHitPlanMax], s_size[kHitPlanMax], s_start[kHitPlanMax];
    const int i = threadIdx.x;
    int32_t e = -1, slot = -1;
    if (i < n) {
        e = ids[i];
        if (e >= 0 && e < n_expert) slot = res[e];
    }
    if (i < kHitPlanMax) s_id[i] = e;
    __syncthreads();
    const bool hit = slot >= 0;
    int first = i;
    if (hit)
        for (int j = 0; j < i; ++j)
            if (s_id[j] == e) { first = j; break; }
    const bool lead = hit && first == i;
    int size = 0;
    if (lead)
        for (int j = i; j < n; ++j) size += s_id[j] == e;
    if (i < kHitPlanMax) s_size[i] = size;   // > 0 at a group's first entry only
    const int groups = __syncthreads_count(lead);
    const int entries = __syncthreads_count(hit);
    int32_t* start = plan + 4;
    int32_t* dst = start + cap + 1;
    int32_t* tok = dst + cap;
    auto* ptr = (unsigned long long*) (plan + ptr_off);
    if (lead) {
        int grp = 0, at = 0;
        for (int j = 0; j < i; ++j)
            if (s_size[j] > 0) { ++grp; at += s_size[j]; }
        s_start[i] = at;
        start[grp] = at;
        ptr[grp] = slot_ptr[slot];
    }
    __syncthreads();
    if (hit) {
        int rank = 0;
        for (int j = first; j < i; ++j) rank += s_id[j] == e;
        dst[s_start[first] + rank] = i;
        tok[s_start[first] + rank] = i / k;
    }
    if (i == 0) {
        plan[0] = groups;
        plan[1] = entries;
        plan[2] = 0;
        plan[3] = 0;
        start[groups] = entries;
    }
}

}  // namespace
}  // namespace strata::kernels
