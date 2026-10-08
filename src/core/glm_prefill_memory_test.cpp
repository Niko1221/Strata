#include "glm_prefill_memory.hpp"

#include <cstdio>
#include <vector>

int main() {
    using namespace strata::core::glm_prefill_memory;
    int checks = 0;
    const auto check = [&](bool okay) {
        ++checks;
        if (!okay) { std::fprintf(stderr, "glm_prefill_memory_test: failed check %d\n", checks); std::exit(1); }
    };
    check(landing_slots(0, 1024, nullptr) == 12);
    check(landing_slots(1600000, 1000, nullptr) == 48);
    check(landing_slots(INT64_MAX, 1024, nullptr) == 96);
    check(landing_slots(1024, 0, nullptr) == 12);
    check(landing_slots(0, 1024, "1") == 12);
    check(landing_slots(0, 1024, "64") == 64);
    check(landing_slots(0, 1024, "128") == 128);
    std::vector<int> attempts;
    check(allocate_landing(64, [&](int n) { attempts.push_back(n); return n == 16; }) == 16);
    check(attempts == std::vector<int>({64, 32, 16}));
    attempts.clear();
    // Match the CUDA callback contract: failed pinning resets its output before retry/fallback.
    void* staging = nullptr;
    check(allocate_landing(64, [&](int n) {
        attempts.push_back(n);
        staging = nullptr;
        return false;
    }) == 0);
    check(attempts == std::vector<int>({64, 32, 16, 12}) && staging == nullptr);
    attempts.clear();
    check(allocate_landing(1, [&](int n) { attempts.push_back(n); return true; }) == 12);
    check(attempts == std::vector<int>({12}));
    check(latent_floats(512, 8193, true) == 512 * 8193 / 2);
    check(latent_floats(3, 3, true) == 5);  // odd element count rounds up to an F32 arena unit
    check(latent_floats(3, 3, false) == 9);
    check(latent_start(5, true) == 8);
    check(latent_start(8, true) == 8);
    check(latent_start(5, false) == 5);  // diagnostic layout remains unchanged
    std::printf("glm_prefill_memory_test: PASS (%d checks)\n", checks);
}
