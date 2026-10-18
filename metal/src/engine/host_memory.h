// metal/src/engine/host_memory.h - the memory macOS can give the engine now, read one way for every user: the
// streaming arena (window.cpp) and the memory guard of parking and session files
// (src/core/conversation_memory.cpp, sample_host_memory).
//
// The rule is arena_budget.h's arena_avail: free and purgeable pages and the clean file cache (inactive and speculative
// pages, at most as many as are file-backed).  Other programs' inactive anonymous memory is left out: macOS frees it
// only by compressing or swapping it.
#pragma once

namespace strata::metal_engine {

/// `available`: bytes macOS can give now without paging anything out (the rule above); `phys`: hw.memsize.  False
/// (both 0) when the VM statistics or the RAM size cannot be read.
bool host_memory_now(double& available, double& phys);

}  // namespace strata::metal_engine
