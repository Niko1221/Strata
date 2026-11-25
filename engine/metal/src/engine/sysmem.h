// metal/src/engine/sysmem.h - what this Mac has and what this process holds, for the arena rules
// (arena_budget.h takes these as its inputs).
//
// WHY A HEADER OF ITS OWN.  Three mach/sysctl probes feed `ArenaMem`, and both Metal engines need the same three:
// `window.cpp` (Qwen) and hy4-flash (engine/metal/src/hy4flash_gguf/experts.cpp).  One copy here, rather than two that can drift.
//
// Header-only on purpose: they are three leaf calls, and this way no target's source list changes.
#pragma once

#include "arena_budget.h"

#include <mach/mach.h>
#include <mach/mach_host.h>
#include <mach/task_info.h>
#include <sys/sysctl.h>

namespace strata::metal_engine {

/// What the process holds now, all of it (task_info TASK_VM_INFO): phys_footprint is its anonymous + wired +
/// compressed memory (not the clean file pages it only maps), resident_size is the RSS (those pages too).  Bytes;
/// 0 when the call fails.
struct ProcMem {
    double footprint = 0, rss = 0;
};
inline ProcMem proc_mem() {
    task_vm_info_data_t ti;
    mach_msg_type_number_t cnt = TASK_VM_INFO_COUNT;
    ProcMem r;
    if (task_info(mach_task_self(), TASK_VM_INFO, (task_info_t) &ti, &cnt) == KERN_SUCCESS) {
        r.footprint = (double) ti.phys_footprint;
        r.rss = (double) ti.resident_size;
    }
    return r;
}

/// Bytes of RAM (sysctl hw.memsize).
inline double sys_phys() {
    uint64_t phys = 0;
    size_t len = sizeof phys;
    sysctlbyname("hw.memsize", &phys, &len, nullptr, 0);
    return (double) phys;
}

/// The file cache macOS keeps before it compresses anonymous memory (a 4-byte count of pages, read into a zeroed
/// 8).  0 when the sysctl is absent.
inline double sys_filecache_min() {
    unsigned long long pages = 0;
    size_t pl = sizeof pages;
    if (sysctlbyname("vm.vm_page_filecache_min", &pages, &pl, nullptr, 0) != 0) return 0;
    return (double) pages * (double) vm_page_size;
}

/// What macOS gives now without paging anything out (arena_avail: free, purgeable, the clean file cache; not other
/// programs' inactive memory, which it frees only by compressing or swapping it).  `fallback` when the call fails.
inline double sys_avail(double fallback) {
    vm_statistics64_data_t vs;
    mach_msg_type_number_t cnt = HOST_VM_INFO64_COUNT;
    if (host_statistics64(mach_host_self(), HOST_VM_INFO64, (host_info64_t) &vs, &cnt) == KERN_SUCCESS) {
        const double pg = (double) vm_page_size;
        return arena_avail(vs.free_count * pg, vs.purgeable_count * pg, vs.inactive_count * pg,
                           vs.speculative_count * pg, vs.external_page_count * pg);
    }
    return fallback;
}

}  // namespace strata::metal_engine
