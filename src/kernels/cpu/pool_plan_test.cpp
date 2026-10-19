// pool_plan_test: the default CPU expert worker count on mock CPUs (include/strata/kernels/cpu/pool_plan.hpp).
// Pure arithmetic, no threads and no OS calls: it runs on any machine.
#include "strata/kernels/cpu/pool_plan.hpp"

#include <cstdio>
#include <string>

namespace cpu = strata::kernels::cpu;

static int g_fail = 0;

static void expect(bool ok, const char* what) {
    if (!ok) {
        ++g_fail;
        std::fprintf(stderr, "FAIL: %s\n", what);
    }
}

static cpu::PoolPlanInput cpu_of(int logical, std::vector<int> per_package, double quota = 0) {
    cpu::PoolPlanInput in;
    in.logical = logical;
    in.cores_per_package = std::move(per_package);
    in.cpu_quota = quota;
    return in;
}

static void line(const char* name, const cpu::PoolPlanInput& in) {
    const cpu::PoolPlan p = cpu::plan_pool_workers(in);
    std::printf("  %-34s logical %3d  cores %3d (host socket %3d, %d socket%s)  -> %2d workers  [%s]\n", name, in.logical,
                p.cores, p.cores_local, p.packages, p.packages == 1 ? "" : "s", p.workers, p.limit);
}

int main() {
    std::printf("pool_plan_test: cap %d, reserve %d from %d workers\n", cpu::kPoolWorkersCap, cpu::kPoolReserve,
                cpu::kPoolReserveFrom);

    // 8 cores / 16 threads: every physical core but the host's, never the SMT siblings
    const auto c8 = cpu_of(16, {8});
    line("8c/16t", c8);
    expect(cpu::plan_pool_workers(c8).workers == 7, "8c/16t -> 7 workers");

    // 6 cores / 12 threads (the small test PCs) and a CPU without SMT
    expect(cpu::plan_pool_workers(cpu_of(12, {6})).workers == 5, "6c/12t -> 5 workers");
    expect(cpu::plan_pool_workers(cpu_of(4, {4})).workers == 3, "4c/4t (no SMT) -> 3 workers");
    expect(cpu::plan_pool_workers(cpu_of(1, {1})).workers == 1, "1 core -> 1 worker (never 0)");
    expect(cpu::plan_pool_workers(cpu_of(2, {2})).workers == 1, "2 cores -> 1 worker");

    // 16 cores / 32 threads
    const auto c16 = cpu_of(32, {16});
    line("16c/32t", c16);
    {
        const int w = cpu::plan_pool_workers(c16).workers;
        expect(w == 15 - (15 >= cpu::kPoolReserveFrom ? cpu::kPoolReserve : 0) || w == cpu::kPoolWorkersCap,
               "16c/32t -> physical cores minus the host (and the reserve)");
        expect(w <= 15, "16c/32t never above the physical cores minus the host");
    }

    // 64 cores / 128 threads: the cap, far below the logical count
    const auto c64 = cpu_of(128, {64});
    line("64c/128t", c64);
    expect(cpu::plan_pool_workers(c64).workers == cpu::kPoolWorkersCap, "64c/128t -> the cap");
    expect(cpu::plan_pool_workers(c64).workers < 64, "64c/128t is not near the logical count");

    // 2 sockets x 64 cores / 256 threads: the user report - never 255, one socket's cores at most, under the cap
    const auto c2s = cpu_of(256, {64, 64});
    line("2-socket 128c/256t", c2s);
    expect(cpu::plan_pool_workers(c2s).workers == cpu::kPoolWorkersCap, "2-socket 128c/256t -> the cap");
    expect(cpu::plan_pool_workers(c2s).packages == 2, "2-socket: two packages");
    expect(cpu::plan_pool_workers(c2s).cores == 128, "2-socket: 128 physical cores counted");

    // 2 sockets x 20 cores (the Xeons of bench #674 #707: 17-18 workers beat 35): the host's socket only
    const auto xeon = cpu_of(80, {20, 20});
    line("2-socket 40c/80t (Xeon)", xeon);
    expect(cpu::plan_pool_workers(xeon).workers == 18, "2x20c: 18 workers (bench #674 #707: 17-18 beat 35): the host socket only, minus the host and the reserve");
    // ... and a smaller second socket does not matter; the host's socket comes first in the input
    expect(cpu::plan_pool_workers(cpu_of(48, {6, 18})).workers == 5, "host socket of 6 cores: 5 workers");

    // an unknown core layout (no sysfs, a container): half the logical processors for a many-thread CPU, never all
    const auto unk = cpu_of(256, {});
    line("256 logical, layout unknown", unk);
    expect(!cpu::plan_pool_workers(unk).topology_known, "unknown layout is flagged");
    expect(cpu::plan_pool_workers(unk).workers <= cpu::kPoolWorkersCap, "unknown 256 -> at most the cap");
    expect(cpu::plan_pool_workers(unk).workers < 255, "unknown 256 -> never 255");
    expect(cpu::plan_pool_workers(cpu_of(8, {})).workers == 7, "unknown, 8 logical -> 7 workers");

    // hybrid CPUs: with more E-cores than P-cores, the P-cores but the host's plus half of the E-cores
    {
        auto h = cpu_of(20, {20});
        h.p_cores = 8;
        h.e_cores = 12;
        line("hybrid 8P+12E (Core Ultra 7 265)", h);
        expect(cpu::plan_pool_workers(h).workers == 13, "8P+12E -> 13 workers (measured: 12-16 beat the 19 of one per core)");
        auto h2 = cpu_of(32, {24});
        h2.p_cores = 8;
        h2.e_cores = 16;
        expect(cpu::plan_pool_workers(h2).workers == 15, "8P+16E -> 15 workers (#642)");
        auto h3 = cpu_of(24, {16});
        h3.p_cores = 8;
        h3.e_cores = 8;
        expect(cpu::plan_pool_workers(h3).workers == 15, "8P+8E: not more E-cores than P-cores, every core but the host's");
    }

    // a container quota
    const auto q = cpu_of(128, {64}, 4.0);
    line("64c, container quota 4 CPUs", q);
    expect(cpu::plan_pool_workers(q).workers == 3, "quota 4 CPUs -> 3 workers");
    expect(cpu::plan_pool_workers(cpu_of(128, {64}, 200.0)).workers == cpu::kPoolWorkersCap, "a big quota changes nothing");

    // warnings: only above the plan, and each says what to do
    const cpu::PoolPlan p2s = cpu::plan_pool_workers(c2s);
    expect(cpu::pool_workers_warning(0, c2s, p2s).empty(), "0 (the default) never warns");
    expect(cpu::pool_workers_warning(p2s.workers, c2s, p2s).empty(), "the plan itself never warns");
    expect(cpu::pool_workers_warning(4, c2s, p2s).empty(), "fewer than the plan never warns");
    {
        const std::string w = cpu::pool_workers_warning(256, c2s, p2s);
        expect(!w.empty(), "256 workers on 256 threads warns");
        expect(w.find("--pool-workers " + std::to_string(p2s.workers)) != std::string::npos, "the warning names the recommendation");
        expect(w.find("still runs") != std::string::npos, "the warning says it still runs");
    }
    expect(cpu::pool_workers_warning(200, c2s, p2s).find("physical cores") != std::string::npos,
           "200 workers on 128 cores: more than the physical cores");
    expect(cpu::pool_workers_warning(300, c2s, p2s).find("256 CPU threads") != std::string::npos,
           "300 workers on 256 threads: more than the CPU threads");
    expect(cpu::pool_workers_warning(30, xeon, cpu::plan_pool_workers(xeon)).find("socket") != std::string::npos,
           "30 workers on 2x20c: crosses the socket");
    expect(!cpu::pool_workers_warning(40, c64, cpu::plan_pool_workers(c64)).empty(), "40 workers on 64c is above the cap: warns");
    expect(cpu::pool_workers_warning(8, q, cpu::plan_pool_workers(q)).find("container") != std::string::npos,
           "8 workers under a 4-CPU quota: names the container limit");

    if (g_fail) {
        std::fprintf(stderr, "pool_plan_test: %d failure(s)\n", g_fail);
        return 1;
    }
    std::printf("pool_plan_test: OK\n");
    return 0;
}
