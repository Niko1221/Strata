#pragma once

#include <vector>

namespace strata::kernels::cpu {

/// Worker placement across physical/logical cores (#272). `All` is the default;
/// hybrid policies are opt-in through --pool-affinity auto|p-cores.
enum class PoolAffinity {
    Auto,      ///< Prefer physical P-cores, then SMT, then E-cores (defaults to P-core count)
    PCores,    ///< Use P-cores and their SMT siblings only
    All,       ///< One worker per physical core in OS order
};

struct CpuTopology {
    bool is_hybrid = false;
    int p_cores = 0;                ///< Physical performance cores
    int p_threads = 0;              ///< Logical threads on performance cores
    int e_cores = 0;                ///< Physical efficient cores
    std::vector<int> worker_cores;  ///< Ordered allowed CPU IDs, excluding the reserved host
    int host_core = -1;             ///< Logical CPU reserved when skip_first is true
};

CpuTopology detect_cpu_topology(bool skip_first, PoolAffinity affinity = PoolAffinity::All);

namespace detail {
/// An allowed logical CPU; sysfs discovery marks repeated physical cores as siblings.
struct CpuCore {
    int cpu = -1;
    long capacity = -1;
    bool is_sibling = false;
};

/// Linux layout policy, separated from sysfs discovery for fixture tests.
/// ARM capacity grouping tolerates small differences between identical cores.
CpuTopology linux_cpu_layout(const std::vector<CpuCore>& cores, bool skip_first,
                             PoolAffinity affinity, bool group_arm_capacities);
}  // namespace detail
}  // namespace strata::kernels::cpu
