// ink_hip_promote_policy.h - may the HIP hybrid engine use an expert whose async copy into the VRAM cache ring is still running? (no HIP, unit-tested)
//
// Same policy and setting as the SYCL hybrid engine (ink_sy_promote_policy.h, INK_SYCL_PROMOTE_ASYNC).  The hybrid engine copies a hot host
// expert into an elastic VRAM cache slot on a DMA stream.  Once the copy is done the expert is computed from VRAM by the GPU kernel; before
// that it is computed on the CPU.  The two paths do not round alike, so WHICH path an expert takes changes the logits.
//
// probe_ready() used to look at the copy with a non-blocking hipEventQuery: finished, use the VRAM copy; running, use the CPU.  That depends on
// timing, so two runs of the same prompt placed experts differently.  The victim choice of the next prefetch skipped slots whose copy was still
// running, which also depends on timing.  The default is now to WAIT for the copy (hipEventSynchronize) where the poll was, so the placement
// depends only on the routing history.  INK_HIP_PROMOTE_ASYNC=1 brings the poll back (speed experiments, not answers that must repeat).
#pragma once

#include <cstdlib>

namespace ink_hip {

// true: poll the copy (timing dependent); false (default): wait for it.
inline bool ink_hip_promote_async_poll(const char* env_value) {
    return env_value != nullptr && env_value[0] == '1';
}

inline bool ink_hip_promote_async_poll() { return ink_hip_promote_async_poll(std::getenv("INK_HIP_PROMOTE_ASYNC")); }

// Is a copy that may still be running treated as usable at this point?  `copy_complete` is what a status poll would say right now.
inline bool ink_hip_promote_usable_now(bool async_poll, bool copy_complete) {
    return async_poll ? copy_complete : true;  // deterministic mode waits for the copy, so it is complete by definition
}

}  // namespace ink_hip
