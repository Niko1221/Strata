// SYCL port: experts mirrored in pinned host memory (GgufExpertSource::mirror) in the device-built verify plan.
#pragma once
#include <cstdint>
namespace strata::kernels {
/// From now on (and in every graph captured after this call) the device-built plan (resident_plan) treats an expert
/// that is not in the VRAM cache but has a non-zero entry in `mirror_table` ([n_layers][n_expert] device-readable
/// addresses, 0 = none) as planned: the expert kernels read it from that address, over PCIe. `d_res` is the residency
/// table whose per-layer slices resident_plan receives, so the layer of a call is found from its pointer.
/// Adaptive swaps without host assistance must retain a mirror for every evicted expert.
void resident_plan_set_mirror(const int32_t* d_res, const unsigned long long* mirror_table);
/// From now on (and in every graph captured after this call) resident_plan counts, in `usage_table`
/// ([n_layers][n_expert] device uint32 counters, indexed like the residency table), how often each expert is routed to.
/// The host reads and clears it to feed the adaptive tier when no host pool sees the routing (STRATA_VERIFY_NO_HOST).
/// Needs resident_plan_set_mirror to have been called first: the layer of a call is found from the residency table.
void resident_plan_set_usage(uint32_t* usage_table);
}
