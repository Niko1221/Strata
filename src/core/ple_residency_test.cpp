#include "strata/core/ple_residency.hpp"

#include <cstdio>
#include <cstdlib>
#include <limits>

using namespace strata::core;
namespace {
int checks = 0;
void check(bool ok, const char* label) {
    ++checks;
    if (!ok) { std::fprintf(stderr, "FAIL: %s\n", label); std::exit(1); }
}
constexpr uint64_t GiB = 1ull << 30;
}

int main() {
    const uint64_t table = 26ull * GiB + 870ull * (1ull << 20);   // IQ3_S's table, 26.8 GiB
    const uint64_t arena = 51ull * GiB + 100ull * (1ull << 20);   // IQ3_S's 51.1 GiB of experts
    // the owner PC, 2026-10-26: 45.2 GiB available when the table is opened, 63 GiB total.  The 0.2.7 first policy
    // (table + 8 GiB) admitted it and the expert load then failed with error 1453.  Not admitted now.
    check(!ple_resident_admit(45ull * GiB + (1ull << 28), table, arena, 8 * GiB), "64 GB PC with the IQ3_S arena: not admitted");
    check(!ple_resident_admit(40ull * GiB, table, arena, 8 * GiB), "40 GiB: not admitted");
    // STRATA_DEFAULTS=026: the policy does not run (v0.2.6 had none); an explicit STRATA_PLE_AUTO_RAM wins either way
    check(ple_auto_ram_allowed(nullptr, false), "policy on by default");
    check(!ple_auto_ram_allowed(nullptr, true), "policy off under STRATA_DEFAULTS=026");
    check(ple_auto_ram_allowed("1", true), "STRATA_PLE_AUTO_RAM=1 wins over 026");
    check(!ple_auto_ram_allowed("0", false), "STRATA_PLE_AUTO_RAM=0 is the old off switch");
    check(!ple_auto_ram_allowed("0", true), "STRATA_PLE_AUTO_RAM=0 under 026");
    // a PC that holds the table, the arena and 8 GiB
    check(ple_resident_admit(table + arena + 8 * GiB, table, arena, 8 * GiB), "exactly table + arena + headroom is admitted");
    check(!ple_resident_admit(table + arena + 8 * GiB - 1, table, arena, 8 * GiB), "one byte below is not");
    check(ple_resident_admit(240ull * GiB, table, arena, 8 * GiB), "a 256 GB box: admitted");
    // a smaller model's arena on a mid PC (Q2_0: 34 GiB arena): 26.8 + 34 + 8 = 68.8
    check(ple_resident_admit(80ull * GiB, table, 34ull * GiB, 8 * GiB), "Q2_0 on 96 GB: admitted");
    check(!ple_resident_admit(60ull * GiB, table, 34ull * GiB, 8 * GiB), "Q2_0 with 60 GiB available: not admitted");
    // unknowns fail closed
    check(!ple_resident_admit({}, table, arena, 8 * GiB), "unknown telemetry");
    check(!ple_resident_admit(240ull * GiB, 0, arena, 8 * GiB), "unknown table size");
    check(!ple_resident_admit(240ull * GiB, table, 0, 8 * GiB), "unknown arena size");
    check(!ple_resident_admit(std::numeric_limits<uint64_t>::max(), table, std::numeric_limits<uint64_t>::max(), 8 * GiB),
          "sizes cannot overflow");
    check(!ple_resident_admit(0, table, arena, 0), "zero available");
    std::printf("ple_residency_test: %d checks passed\n", checks);
    return 0;
}
