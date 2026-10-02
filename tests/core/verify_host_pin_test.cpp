// tests/core/verify_host_pin_test.cpp - P1: the verify host thread pins to a core no pool worker uses, and never
// under a layer split (one reserved core, one stage's host - otherwise the stages serialize).
#include "strata/core/verify.hpp"
#include "strata/kernels/cpu/pool.hpp"

#include <algorithm>
#include <cstdio>
#include <vector>

int main() {
    using strata::core::Verifier;
    const std::vector<int> workers = strata::kernels::cpu::physical_cores(true);
    const std::vector<int> all = strata::kernels::cpu::physical_cores(false);
    const int core = Verifier::host_pin_core(false);
    if (!all.empty() && core < 0) {
        std::fprintf(stderr, "verify_host_pin: no reserved host core among %zu physical cores\n", all.size());
        return 1;
    }
    if (core >= 0 && std::find(workers.begin(), workers.end(), core) != workers.end()) {
        std::fprintf(stderr, "verify_host_pin: chosen host core %d is a pool worker's\n", core);
        return 1;
    }
    if (Verifier::host_pin_core(true) != -1) {
        std::fprintf(stderr, "verify_host_pin: a layer split must not pin\n");
        return 1;
    }
    std::printf("verify_host_pin: ok (host core %d; %zu worker cores, %zu physical cores)\n", core, workers.size(),
                all.size());
    return 0;
}
