// include/strata/platform/numa_os.hpp - the OS-side NUMA answers platform/numa.hpp's decisions need.
//
// The entire portability surface of the NUMA staging feature lives behind these four functions: node
// count, device -> node, thread -> node, and node-preferred virtual allocation.  Every function answers
// "-1/false/1/nullptr" on every failure so the caller takes its legacy path: a machine without the
// query, a driver that does not fill the property, or a VM that hides NUMA all degrade to the
// pre-switch behaviour rather than to an error.  The implementation (platform/numa_os.cpp) is
// Windows-only today; non-Windows builds answer "unknown" everywhere, which the decisions translate
// into the legacy path.  A port fills in the same functions (Linux: sysfs numa_node / mbind /
// sched_setaffinity; get_mempolicy for the audit probe) and nothing else needs to change.
#pragma once

#include <cstdint>

namespace strata::platform {

/// How many NUMA nodes the machine has (1 when the OS cannot say or has one).
int numa_node_count_os();

/// The NUMA node a CUDA device hangs off (-1 = the OS cannot name it).
int device_numa_node(int dev);

/// Bind the calling thread to the CPUs of `node`.  false = could not (the caller falls back to the
/// placement it would have used anyway, so a failed bind is never worse than the pre-switch behaviour).
bool bind_thread_to_node(int node);

/// Whether node-preferred virtual allocation exists on this system at all.
bool numa_va_supported();

/// Node-preferred virtual allocation (reserve+commit; `large` asks for large pages).  The node is a
/// preference on the VAD - pages a node cannot hold spill to others at fault time.  `addr` pins an
/// exact address (the arena domains' placeholder dance re-reserves inside a released span); nullptr
/// lets the OS choose.  nullptr return = API unavailable or the allocation was refused; the caller
/// falls back to its old allocator either way.
void* numa_virtual_alloc(void* addr, uint64_t bytes, bool large, int node);

}  // namespace strata::platform
