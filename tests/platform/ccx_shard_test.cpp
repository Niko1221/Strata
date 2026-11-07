#include "strata/kernels/cpu/ccx_shard.hpp"
#include <array>
#include <cstdio>
#include <vector>
#include <pthread.h>
#include <sched.h>

int main() {
    using namespace strata::kernels::cpu;
    // The host is on logical CPU 24, workers on 25..29; 3+3 L3 IDs.
    const std::vector<int> cpus{25, 26, 27, 28, 29, 24};
    const std::vector<int> cache{1, 1, 2, 2, 2, 1};
    std::array<int, 6> slice{};
    if (!ccx_shard_plan(cpus, cache, slice)) return 1;
    for (int id = 0; id < 256; ++id) {
        const int owner = ccx_owner(id);
        if (owner != (id & 1) || slice[0] != 0 || slice[2] != 1) return 2;
    }
    auto bad = cache;
    bad[2] = 1;
    if (ccx_shard_plan(cpus, bad, slice)) return 3;
    auto dup = cpus;
    dup[5] = dup[0];
    if (ccx_shard_plan(dup, cache, slice)) return 4;

    // Exercise the kernel affinity syscall on CPUs the test process can use;
    // no assumed logical CPU number leaks into the scheduler.
    cpu_set_t original;
    if (pthread_getaffinity_np(pthread_self(), sizeof original, &original) != 0) return 5;
    for (int cpu = 0; cpu < CPU_SETSIZE; ++cpu) {
        if (!CPU_ISSET(cpu, &original)) continue;
        cpu_set_t one;
        CPU_ZERO(&one);
        CPU_SET(cpu, &one);
        if (pthread_setaffinity_np(pthread_self(), sizeof one, &one) != 0) return 6;
        cpu_set_t got;
        if (pthread_getaffinity_np(pthread_self(), sizeof got, &got) != 0 || !CPU_ISSET(cpu, &got)) return 7;
        if (pthread_setaffinity_np(pthread_self(), sizeof original, &original) != 0) return 8;
        std::printf("CCX plan 3+3 and pthread affinity OK (CPU %d)\n", cpu);
        return 0;
    }
    return 77;
}