// SYCL port: experts mirrored in pinned host memory (GgufExpertSource::mirror) in the device-built verify plan.
#pragma once
#include <cstdint>
namespace strata::kernels {
/// From now on (and in every graph captured after this call) the device-built plan (resident_plan) treats an expert
/// that is not in the VRAM cache but has a non-zero entry in `mirror_table` ([n_layers][n_expert] device-readable
/// addresses, 0 = none) as planned: the expert kernels read it from that address, over PCIe. `d_res` is the residency
/// table whose per-layer slices resident_plan receives, so the layer of a call is found from its pointer; `n_layers` is
/// how many layers `mirror_table` holds, and a slice outside it gets no mirror (a layer-split stage's own residency
/// table can sit above this one, and the pointer difference would then index far out of range).
void resident_plan_set_mirror(const int32_t* d_res, const unsigned long long* mirror_table, int n_layers);
}
