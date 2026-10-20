// include/strata/kernels/cpu/pool_plan.hpp - how many CPU expert workers the engine starts by default, and when a
// configured count deserves a warning.  Pure functions of a CPU description (no OS calls), so the choice is unit
// tested on mock topologies (pool_plan_test.cpp: 8c/16t, 16c/32t, 64c/128t, a 2-socket 128c/256t box, ...).
//
// Why a plan and not "every physical core but the host's":
//   * the expert pool's workers spin while they wait.  A worker without a core of its own (more workers than physical
//     cores) takes cycles from a worker that has one, so a big count is slower than a small one, not just useless.
//     A report from a 256-thread machine: ~256 workers, speed collapsed.
//   * the kernel is DRAM-bound; past the memory system's width more cores add nothing, only the barrier at the end of
//     every layer waits for more threads (kPoolWorkersCap, measured: phaseA-tests/notes-152-users.md).
//   * on a 2-socket box the workers of the other socket read the experts across the interconnect (35 workers measured
//     slower than 17-18 on 2-socket Xeons, bench #674 #707): the default stays on the host thread's socket.
//   * a container limited to a few CPUs by a CFS quota (docker --cpus) still shows every core of the machine; spinning
//     threads beyond the quota are throttled together.
//
// A count the user configures (--pool-workers) always runs as given; `pool_workers_warning` says in plain words when it
// is above what this CPU can use well (recommend, never force).
#pragma once

#include <algorithm>
#include <string>
#include <vector>

namespace strata::kernels::cpu {

/// The most default workers: past this the layer barrier costs more than the extra cores read (see the header comment).
/// Measured on 6c/12t (z440), 20c (P2003), 32c/64t (r9700) and 64c (l620): notes-152-users.md.
inline constexpr int kPoolWorkersCap = 32;
/// Cores kept free for the other threads of the engine (copy / IO threads, the GPU driver's, the OS) on a CPU with at
/// least `kPoolReserveFrom` cores that the pool may use (besides the host's).  Small CPUs keep every core.
inline constexpr int kPoolReserveFrom = 16;
inline constexpr int kPoolReserve = 1;

struct PoolPlanInput {
    int logical = 0;                    ///< logical processors this process may run on
    /// physical cores this process may use per CPU package (socket), the host thread's package FIRST; the host
    /// thread's own core is counted in it.  One entry on a single-socket CPU.  Empty: the topology is not known.
    std::vector<int> cores_per_package;
    double cpu_quota = 0;               ///< CPUs a container quota allows (cgroup cpu.max); 0: no quota
    /// Physical performance / efficiency cores of a hybrid CPU (both 0 on an ordinary one).  With more E-cores than
    /// P-cores the default is the P-cores but the host's plus half of the E-cores (#642: i9-14900KF 8P+16E, 15 workers
    /// 165 / 116 tok/s against 106 / 84 with all 23; Core Ultra 7 265, 8P+12E, RTX 3060 decode at 4K: 12-16 workers
    /// +12.6% over the 19 of one-per-core, 24 workers -44%: notes-152-users.md).
    int p_cores = 0;
    int e_cores = 0;
};

/// #3262: is a CPU an efficiency-class core?  `pmu_core` / `pmu_atom` / `pmu_lowpower` are Intel's hybrid PMU lists
/// (/sys/devices/cpu_core/cpus, cpu_atom/cpus, cpu_lowpower/cpus; the last exists on Meteor Lake and Arrow Lake-H, where
/// the LP-E cores of the SoC tile are in neither of the other two).  A CPU in the P list is a P-core; one in the atom
/// or low-power list is an E-core (the LP-E cores sort last by capacity).  A CPU in no list falls back to its capacity
/// (an E-core under 90% of the largest).  `pmu`: both the core and the atom list were read.
inline bool hybrid_cpu_is_efficiency(int cpu, long cap, long max_cap, bool pmu, const std::vector<int>& pmu_core,
                                     const std::vector<int>& pmu_atom, const std::vector<int>& pmu_lowpower) {
    auto in = [&](const std::vector<int>& v) { return std::find(v.begin(), v.end(), cpu) != v.end(); };
    if (pmu) {
        if (in(pmu_atom) || in(pmu_lowpower)) return true;
        if (in(pmu_core)) return false;
    }
    return max_cap > 0 && cap > 0 && cap * 10 < max_cap * 9;
}

struct PoolPlan {
    int workers = 1;                   ///< the default count of pool workers (the host thread works beside them)
    int cores = 0;                      ///< physical cores counted (all packages)
    int cores_local = 0;                ///< ... on the host's package
    int packages = 1;
    bool topology_known = true;
    const char* limit = "";             ///< what bounded the count (for the start-up line)
};

inline PoolPlan plan_pool_workers(const PoolPlanInput& in) {
    PoolPlan p;
    int cores = 0;
    for (int c : in.cores_per_package) cores += std::max(0, c);
    int local = in.cores_per_package.empty() ? 0 : std::max(0, in.cores_per_package.front());
    p.packages = std::max<int>(1, (int) in.cores_per_package.size());
    p.topology_known = !in.cores_per_package.empty() && cores > 0;
    if (!p.topology_known) {
        // The OS did not say which logical processors share a core.  Counting every one as a core is what put a
        // 256-thread machine at 255 workers; a 2-way SMT is the usual case on a many-core CPU.
        cores = in.logical >= 16 ? in.logical / 2 : in.logical;
        local = cores;
        p.packages = 1;
    }
    p.cores = cores;
    p.cores_local = local;
    p.limit = "every physical core but the host's";
    int n = std::max(1, local - 1);                       // the host thread has the first core
    if (!p.topology_known) p.limit = "half the logical processors (the OS gave no core layout)";
    else if (p.packages > 1) p.limit = "the host's CPU socket";
    if (n >= kPoolReserveFrom) {                          // a big CPU keeps a core for the engine's other threads
        n -= kPoolReserve;
        if (p.topology_known && p.packages == 1) p.limit = "physical cores, one kept free";
    }
    if (in.p_cores > 0 && in.e_cores > in.p_cores) {      // a hybrid CPU with more E-cores than P-cores
        const int h = std::max(1, in.p_cores - 1 + in.e_cores / 2);
        if (h < n) {
            n = h;
            p.limit = "the performance cores and half of the efficiency cores";
        }
    }
    if (in.cpu_quota > 0.0) {
        const int q = std::max(1, (int) in.cpu_quota - 1);
        if (q < n) {
            n = q;
            p.limit = "the container's CPU quota";
        }
    }
    if (n > kPoolWorkersCap) {
        n = kPoolWorkersCap;
        p.limit = "the measured cap";
    }
    p.workers = std::max(1, n);
    return p;
}

/// The warning for a configured worker count `requested` (> 0), or "" when it is fine.  `plan` is plan_pool_workers(in).
inline std::string pool_workers_warning(int requested, const PoolPlanInput& in, const PoolPlan& plan) {
    if (requested <= 0 || requested <= plan.workers) return "";
    const std::string rec = "Recommended: --pool-workers " + std::to_string(plan.workers) +
                            " (or leave it out: the engine picks that by itself). It still runs as configured.";
    const std::string asked = std::to_string(requested) + " CPU expert workers are configured";
    if (requested + 1 > in.logical && in.logical > 0)
        return asked + ", but this process can run only " + std::to_string(in.logical) +
               " CPU threads. The extra workers spin and take time from the others. " + rec;
    if (plan.topology_known && requested + 1 > plan.cores)
        return asked + ", but this CPU has " + std::to_string(plan.cores) +
               " physical cores. Two workers on one core share its memory bandwidth and slow each other down. " + rec;
    if (plan.packages > 1 && requested + 1 > plan.cores_local)
        return asked + ", more than the " + std::to_string(plan.cores_local) +
               " cores of the CPU socket the engine's main thread runs on. Workers on the other socket read the experts " +
               "across the link and were measured slower than fewer local ones. " + rec;
    if (in.cpu_quota > 0.0 && (double) requested + 1 > in.cpu_quota)
        return asked + ", but a container limit allows about " + std::to_string((int) in.cpu_quota) +
               " CPUs. Threads beyond the limit are throttled together. " + rec;
    return asked + ", more than this CPU can use well (the speed stops rising at about " +
           std::to_string(plan.workers) + " workers and the end of each layer waits for every one). " + rec;
}

}  // namespace strata::kernels::cpu
