// Linux affinity and SMT selection, without model fixtures.
#include "strata/kernels/cpu/pool.hpp"
#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <sched.h>
#include <set>
#include <utility>

namespace cpu = strata::kernels::cpu;
static void require(bool ok) { if (!ok) { std::fputs("pool topology failed\n", stderr); std::exit(1); } }
static long topology(int core, const char* field) {
    char path[128]; std::snprintf(path, sizeof(path), "/sys/devices/system/cpu/cpu%d/topology/%s", core, field);
    FILE* file = std::fopen(path, "r"); long value = -1;
    if (file) { const int read = std::fscanf(file, "%ld", &value); std::fclose(file); require(read == 1); }
    return value;
}
int main() {
    cpu_set_t original; require(sched_getaffinity(0, sizeof(original), &original) == 0);
    const auto cores = cpu::physical_cores(false); require(!cores.empty());
    std::set<std::pair<long, long>> seen;
    for (int core : cores) {
        require(CPU_ISSET(core, &original));
        const auto package = topology(core, "physical_package_id"), id = topology(core, "core_id");
        if (package >= 0 && id >= 0) require(seen.emplace(package, id).second);
    }
    { cpu::ExpertPool pool; require(pool.workers() == std::max(1, (int) cores.size() - 1)); }
    // Restrict to the last allowed logical CPU: an excluded first SMT sibling
    // must not make its physical core disappear.
    int last = -1;
    for (int i = 0; i < CPU_SETSIZE; ++i) if (CPU_ISSET(i, &original)) last = i;
    cpu_set_t restricted; CPU_ZERO(&restricted); CPU_SET(last, &restricted);
    require(sched_setaffinity(0, sizeof(restricted), &restricted) == 0);
    const auto one = cpu::physical_cores(false);
    require(one.size() == 1 && one[0] == last);
    require(cpu::physical_cores(true).empty());
    { cpu::ExpertPool pool; require(pool.workers() == 1); }
    require(sched_setaffinity(0, sizeof(original), &original) == 0);
    std::printf("pool topology: %zu physical cores; restricted sibling selection PASS\n", cores.size());
}
