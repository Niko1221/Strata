// ink_sy_intelarc_gate_test.cpp - host-only test of the Battlemage refusal of the fast Intel Arc set (no device needed)
//   icpx -fsycl -std=c++17 ink_sy_intelarc_gate_test.cpp -o ink_sy_intelarc_gate_test && ./ink_sy_intelarc_gate_test
// MIT License - Copyright (c) 2026 Strata contributors
#include <cstdio>

#include "ink_sy_intelarc_cfg.h"

int main() {
    int fails = 0;
    auto check = [&](const char* what, bool got, bool want) {
        if (got != want) { std::printf("FAIL %s: got %d want %d\n", what, (int)got, (int)want); ++fails; }
    };
    check("battlemage fast is refused", ink_sycl::intelarc_fast_refused(true, false), true);
    check("battlemage exact is kept", ink_sycl::intelarc_fast_refused(true, true), false);
    check("alchemist fast is kept", ink_sycl::intelarc_fast_refused(false, false), false);
    check("alchemist exact is kept", ink_sycl::intelarc_fast_refused(false, true), false);
    check("name helper still lower-cases", ink_sycl::to_lower_ascii("Arc Pro B70") == "arc pro b70", true);
    std::printf(fails ? "FAILED (%d)\n" : "ALL PASS\n", fails);
    return fails ? 1 : 0;
}
