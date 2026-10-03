#include "strata/kernels/cpu/cpu_topology.hpp"

#include <algorithm>
#include <limits>

namespace strata::kernels::cpu::detail {

CpuTopology linux_cpu_layout(const std::vector<CpuCore>& cores, bool skip_first,
                             PoolAffinity affinity, bool group_arm_capacities) {
    CpuTopology topo;
    long max_cap = 0, min_cap = std::numeric_limits<long>::max();
    bool complete_capacities = true;
    for (const auto& core : cores) {
        if (core.capacity > 0) {
            max_cap = (std::max)(max_cap, core.capacity);
            min_cap = (std::min)(min_cap, core.capacity);
        } else {
            complete_capacities = false;
        }
    }

    long performance_threshold = max_cap;
    if (group_arm_capacities) {
        // GB10's identical X925 cores report 997/1017/1024; A725 reports 718/731.
        // Treat a spread below 10% as homogeneous. Otherwise split at the midpoint.
        // Incomplete sysfs data cannot establish performance classes safely.
        topo.is_hybrid = complete_capacities && max_cap > 0 &&
                         static_cast<double>(max_cap) >= 1.1 * static_cast<double>(min_cap);
        if (topo.is_hybrid) performance_threshold = min_cap + (max_cap - min_cap) / 2;
    } else {
        // Preserve the existing x86 capacity policy.
        topo.is_hybrid = max_cap > 0 && max_cap > min_cap;
    }

    for (const auto& core : cores) {
        if (!topo.is_hybrid || core.capacity >= performance_threshold) {
            if (!core.is_sibling) ++topo.p_cores;
            ++topo.p_threads;
        } else if (!core.is_sibling) {
            ++topo.e_cores;
        }
    }

    if (affinity == PoolAffinity::All || !topo.is_hybrid) {
        for (const auto& core : cores) {
            if (!core.is_sibling) topo.worker_cores.push_back(core.cpu);
        }
        if (skip_first && !topo.worker_cores.empty()) {
            topo.host_core = topo.worker_cores.front();
            topo.worker_cores.erase(topo.worker_cores.begin());
        }
        return topo;
    }

    std::vector<int> p_primaries, p_siblings, e_cores;
    for (const auto& core : cores) {
        if (core.capacity >= performance_threshold) {
            if (core.is_sibling) p_siblings.push_back(core.cpu);
            else p_primaries.push_back(core.cpu);
        } else {
            e_cores.push_back(core.cpu);
        }
    }
    if (skip_first && !p_primaries.empty()) {
        topo.host_core = p_primaries.front();
        p_primaries.erase(p_primaries.begin());
    }
    topo.worker_cores.insert(topo.worker_cores.end(), p_primaries.begin(), p_primaries.end());
    topo.worker_cores.insert(topo.worker_cores.end(), p_siblings.begin(), p_siblings.end());
    if (affinity != PoolAffinity::PCores) {
        topo.worker_cores.insert(topo.worker_cores.end(), e_cores.begin(), e_cores.end());
    }
    return topo;
}
}  // namespace strata::kernels::cpu::detail
