#include "strata/core/hip_stability.hpp"
#include <cstdio>
#include <cstdlib>
#include <cstring>

static void env(const char* name, const char* value) {
#ifdef _WIN32
    _putenv_s(name, value ? value : "");
#else
    if (value) setenv(name, value, 1); else unsetenv(name);
#endif
}

int main() {
    env("STRATA_HIP_STABLE", "0");
    env("HSA_ENABLE_SDMA", nullptr);
    if (!strata::core::configure_hip_stability() || std::getenv("HSA_ENABLE_SDMA")) return 1;
    env("STRATA_HIP_STABLE", "1");
    env("STRATA_SH_STREAM", "1"); // deliberate one-variable override
    if (!strata::core::configure_hip_stability()) return 2;
    if (std::strcmp(std::getenv("HSA_ENABLE_SDMA"), "0") ||
        std::strcmp(std::getenv("STRATA_SH_STREAM"), "1")) return 3;
    using strata::core::hip_stability_chunk;
    // Disabled and explicitly small prefill stay small; auto and very large
    // chunks cannot defeat the conservative buffer reservation.
    if (hip_stability_chunk(0, false) != 0 || hip_stability_chunk(512, false) != 512 ||
        hip_stability_chunk(32768, false) != 2048 || hip_stability_chunk(0, true) != 2048) return 4;
    if (strata::core::hip_stability_full_kv()) return 5;
    env("STRATA_HIP_STABLE", "2");
    if (!strata::core::hip_stability_requested() || !strata::core::hip_stability_full_kv()) return 6;
    return 0;
}
