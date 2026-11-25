// sycl/include/strata/core/verify_env.hpp - STRATA_VERIFY_NO_HOST needs STRATA_VERIFY_DEVICE_PLAN (#6250, #4698).
// With STRATA_VERIFY_NO_HOST=1 the host never answers a layer's doorbell (verify.cpp: no host stepping loop).  A layer whose experts are not all in VRAM is
// then planned only by the device-built plan (STRATA_VERIFY_DEVICE_PLAN=1, device_plan_); without it the GPU waits for a plan that nobody makes and the window
// ends "the GPU stopped waiting for the plan for its GPU share (flag A) ... the CPU took 0 ms" on an xe card (#6250), or the output is "!!!" from the first token
// (#4698).  strata-sycl.sh sets both; an engine started without that wrapper (the packaged component, a source run) with only NO_HOST=1, as docs/INTEL.md
// says an xe card needs, hit it.  For a stage with every expert in VRAM the device plan is off by itself (device_plan_ = !all_resident_ && ...), so asking for it
// changes nothing there.  A pure function: no queue, no GPU.
#pragma once

#include <cstring>

namespace strata {

/// True when STRATA_VERIFY_DEVICE_PLAN should be set to 1 by the engine: NO_HOST is on (set, not empty, not "0") and DEVICE_PLAN is not set at all.  A value the
/// caller set, "0" included, is kept.
inline bool no_host_wants_device_plan(const char* no_host, const char* device_plan) {
    const bool nh = no_host != nullptr && no_host[0] != '\0' && std::strcmp(no_host, "0") != 0;
    return nh && device_plan == nullptr;
}

/// #4698: the combination to refuse at the start.  With STRATA_VERIFY_NO_HOST on (set, not empty, not "0") the host never serves a layer, and the device plan
/// reads every routed expert from VRAM or from a USM mirror (--stream-experts).  A host arena (no --stream-experts) is ordinary pageable memory on SYCL
/// (PinnedArena::register_slices registers nothing): the experts the VRAM cache does not hold have no answer, and a reply came out as '!' x 256 (Arc Pro B70,
/// #4698) or the window hung.  True when NO_HOST is on, the experts come from the host arena, and the VRAM cache holds fewer than all of them.  With every
/// expert in VRAM the arena is never read, so nothing is refused.  A pure function: no queue, no GPU.
inline bool arena_refuses_no_host(const char* no_host, bool host_arena, long long experts_total, long long experts_in_vram) {
    const bool nh = no_host != nullptr && no_host[0] != '\0' && std::strcmp(no_host, "0") != 0;
    return nh && host_arena && experts_in_vram < experts_total;
}

}  // namespace strata
