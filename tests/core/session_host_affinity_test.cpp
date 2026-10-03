#include "strata/core/session.hpp"
#include "strata/kernels/cpu/pool.hpp"

#include <cstdio>
#include <sched.h>
#include <string>

using strata::core::SessionLoopScratch;
using strata::kernels::cpu::ExpertPool;
using strata::kernels::cpu::PoolAffinity;
using strata::kernels::cpu::detect_cpu_topology;

int main() {
    int devices = 0;
    if (cudaGetDeviceCount(&devices) != cudaSuccess || devices == 0) {
        std::puts("session_host_affinity_test: SKIP: no CUDA device");
        return 77;
    }
    cpu_set_t original;
    if (sched_getaffinity(0, sizeof original, &original) != 0) return 1;
    struct Restore {
        cpu_set_t mask;
        ~Restore() { sched_setaffinity(0, sizeof mask, &mask); }
    } restore{original};

    // pin_current_thread's existing save/restore API stores a 64-bit mask.
    // Limit this process to representable allowed IDs while testing that API.
    cpu_set_t allowed;
    CPU_ZERO(&allowed);
    for (int cpu = 0; cpu < 63; ++cpu) {
        if (CPU_ISSET(cpu, &original)) CPU_SET(cpu, &allowed);
    }
    if (CPU_COUNT(&allowed) == 0) {
        std::puts("session_host_affinity_test: SKIP: no allowed CPU below 63");
        return 77;
    }
    if (sched_setaffinity(0, sizeof allowed, &allowed) != 0) return 1;

    for (PoolAffinity affinity : {PoolAffinity::All, PoolAffinity::Auto, PoolAffinity::PCores}) {
        const int expected = detect_cpu_topology(true, affinity).host_core;
        ExpertPool pool(1, true, true, affinity);
        if (pool.host_core() != expected || expected < 0) return 1;
        SessionLoopScratch scratch;
        std::string err;
        if (!scratch.init(64, err, pool.host_core())) {
            std::fprintf(stderr, "session_host_affinity_test: init failed: %s\n", err.c_str());
            return 1;
        }
        cpu_set_t pinned;
        const bool host_ok = sched_getaffinity(0, sizeof pinned, &pinned) == 0 &&
                             CPU_COUNT(&pinned) == 1 && CPU_ISSET(expected, &pinned);
        scratch.free();
        cpu_set_t after;
        const bool restored = sched_getaffinity(0, sizeof after, &after) == 0 &&
                              CPU_EQUAL(&after, &allowed);
        if (!host_ok || !restored) {
            std::fprintf(stderr, "session_host_affinity_test: mode %d host/restoration failed\n",
                         static_cast<int>(affinity));
            return 1;
        }
        std::printf("session_host_affinity_test: mode=%d host=%d restored=yes\n",
                    static_cast<int>(affinity), expected);
    }

    // Existing two-argument callers keep the default All-policy host.
    const int default_host = detect_cpu_topology(true).host_core;
    SessionLoopScratch legacy;
    std::string err;
    if (!legacy.init(64, err)) return 1;
    cpu_set_t pinned;
    const bool legacy_ok = sched_getaffinity(0, sizeof pinned, &pinned) == 0 &&
                           CPU_COUNT(&pinned) == 1 && CPU_ISSET(default_host, &pinned);
    legacy.free();
    cpu_set_t after;
    if (!legacy_ok || sched_getaffinity(0, sizeof after, &after) != 0 ||
        !CPU_EQUAL(&after, &allowed)) return 1;
    std::puts("session_host_affinity_test: PASS");
    return 0;
}
