#include "strata/kernels/cpu/cpu_topology.hpp"

#include <cstdio>
#include <cstdlib>
#include <vector>

using strata::kernels::cpu::CpuTopology;
using strata::kernels::cpu::PoolAffinity;
using strata::kernels::cpu::detail::CpuCore;
using strata::kernels::cpu::detail::linux_cpu_layout;

namespace {
void check(bool ok, const char* what) {
    if (!ok) {
        std::fprintf(stderr, "cpu_topology_test: FAIL: %s\n", what);
        std::exit(1);
    }
}

void check_layout(const CpuTopology& topo, int host, const std::vector<int>& workers,
                  const char* what) {
    check(topo.host_core == host && topo.worker_cores == workers, what);
    for (int worker : topo.worker_cores) {
        check(worker != topo.host_core, "reserved host CPU must not be a worker CPU");
    }
}
}  // namespace

int main() {
    // Actual scheduler capacities from a 20-core GB10 DGX Spark.
    const long capacities[] = {
        718, 718, 718, 718, 718, 997, 997, 997, 997, 997,
        731, 731, 731, 731, 731, 1017, 1017, 1017, 1017, 1024,
    };
    std::vector<CpuCore> gb10;
    for (int cpu = 0; cpu < 20; ++cpu) gb10.push_back({cpu, capacities[cpu], false});
    const std::vector<int> p_workers = {6, 7, 8, 9, 15, 16, 17, 18, 19};
    const std::vector<int> auto_workers = {
        6, 7, 8, 9, 15, 16, 17, 18, 19, 0, 1, 2, 3, 4, 10, 11, 12, 13, 14,
    };
    const auto automatic = linux_cpu_layout(gb10, true, PoolAffinity::Auto, true);
    check(automatic.is_hybrid && automatic.p_cores == 10 && automatic.p_threads == 10 &&
          automatic.e_cores == 10, "GB10 has ten P-cores and ten E-cores");
    check_layout(automatic, 5, auto_workers, "Auto reserves X925 CPU 5");
    check_layout(linux_cpu_layout(gb10, true, PoolAffinity::PCores, true), 5, p_workers,
                 "PCores excludes all A725 CPUs");
    std::vector<int> all_workers;
    for (int cpu = 1; cpu < 20; ++cpu) all_workers.push_back(cpu);
    check_layout(linux_cpu_layout(gb10, true, PoolAffinity::All, true), 0, all_workers,
                 "All preserves OS order and reserves CPU 0");

    auto without_reservation = auto_workers;
    without_reservation.insert(without_reservation.begin(), 5);
    check_layout(linux_cpu_layout(gb10, false, PoolAffinity::Auto, true), -1,
                 without_reservation, "skip_first=false does not reserve a host");

    // A restricted affinity mask is represented by only the allowed CPUs.
    const std::vector<CpuCore> restricted = {gb10[1], gb10[8], gb10[16]};
    const auto restricted_topo = linux_cpu_layout(restricted, true, PoolAffinity::Auto, true);
    check(restricted_topo.p_cores == 2 && restricted_topo.e_cores == 1,
          "restricted mask counts only allowed cores");
    check_layout(restricted_topo, 8, {16, 1}, "restricted mask reserves an allowed P-core");

    const std::vector<CpuCore> noisy = {{2, 997, false}, {7, 1017, false}, {9, 1024, false}};
    const auto homogeneous = linux_cpu_layout(noisy, true, PoolAffinity::PCores, true);
    check(!homogeneous.is_hybrid && homogeneous.p_cores == 3,
          "small ARM capacity differences do not invent E-cores");
    check_layout(homogeneous, 2, {7, 9}, "homogeneous ARM retains OS order");

    const std::vector<CpuCore> missing = {{3, -1, false}, {6, 1024, false}, {8, 718, false}};
    const auto incomplete = linux_cpu_layout(missing, true, PoolAffinity::PCores, true);
    check(!incomplete.is_hybrid && incomplete.p_cores == 3,
          "missing capacity falls back to physical-core order");
    check_layout(incomplete, 3, {6, 8}, "unknown capacity never silently drops an allowed CPU");
    const auto absent = linux_cpu_layout({{3, -1, false}, {6, -1, false}}, true,
                                         PoolAffinity::Auto, true);
    check(!absent.is_hybrid && absent.p_cores == 2, "absent sysfs still counts physical cores");
    check_layout(absent, 3, {6}, "absent capacity preserves affinity order");

    const std::vector<CpuCore> smt = {
        {0, 1024, false}, {1, 512, false}, {2, 1024, false},
        {4, 1024, true}, {5, 512, true}, {6, 1024, true},
    };
    const auto x86 = linux_cpu_layout(smt, true, PoolAffinity::Auto, false);
    check(x86.is_hybrid && x86.p_cores == 2 && x86.p_threads == 4 && x86.e_cores == 1,
          "x86 retains physical counts and logical P-thread counts");
    check_layout(x86, 0, {2, 4, 6, 1, 5}, "x86 retains P-primary/SMT/E ordering");
    check_layout(linux_cpu_layout(smt, true, PoolAffinity::All, false), 0, {1, 2},
                 "All deduplicates SMT siblings");
    check_layout(linux_cpu_layout(smt, true, PoolAffinity::PCores, false), 0, {2, 4, 6},
                 "x86 PCores still includes P SMT siblings");
    const auto exact_x86 = linux_cpu_layout(noisy, true, PoolAffinity::Auto, false);
    check(exact_x86.is_hybrid && exact_x86.p_cores == 1 && exact_x86.e_cores == 2,
          "x86 exact-capacity classification remains unchanged");
    check_layout(exact_x86, 9, {2, 7}, "x86 maximum-capacity host policy remains unchanged");

    const auto empty = linux_cpu_layout({}, true, PoolAffinity::Auto, true);
    check(!empty.is_hybrid && empty.p_cores == 0 && empty.p_threads == 0 && empty.e_cores == 0,
          "empty affinity set has zero counts");
    check_layout(empty, -1, {}, "empty affinity set has no host or workers");
    check_layout(linux_cpu_layout({{19, 1024, false}}, true, PoolAffinity::Auto, true),
                 19, {}, "one allowed CPU can be reserved without workers");
    std::puts("cpu_topology_test: PASS");
    return 0;
}
