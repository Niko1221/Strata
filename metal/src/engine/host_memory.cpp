// metal/src/engine/host_memory.cpp - see host_memory.h.
#include "host_memory.h"

#include "arena_budget.h"

#include <cstdint>
#include <mach/mach.h>
#include <sys/sysctl.h>

namespace strata::metal_engine {

bool host_memory_now(double& available, double& phys) {
    available = 0;
    phys = 0;
    uint64_t total = 0;
    size_t len = sizeof total;
    if (sysctlbyname("hw.memsize", &total, &len, nullptr, 0) != 0 || total == 0) return false;
    vm_statistics64_data_t vs{};
    mach_msg_type_number_t cnt = HOST_VM_INFO64_COUNT;
    if (host_statistics64(mach_host_self(), HOST_VM_INFO64, (host_info64_t) &vs, &cnt) != KERN_SUCCESS) return false;
    const double pg = (double) vm_page_size;
    phys = (double) total;
    available = arena_avail(vs.free_count * pg, vs.purgeable_count * pg, vs.inactive_count * pg, vs.speculative_count * pg,
                            vs.external_page_count * pg);
    return true;
}

}  // namespace strata::metal_engine
