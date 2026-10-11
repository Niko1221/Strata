#include "strata/core/hip_verify_policy.hpp"
#include <cstdlib>

static bool initial(const char* name) {
    const char* value = std::getenv(name);
    return value && value[0] && value[0] != '0';
}
// Observe the inherited environment before main, then deliberately change it
// as the engine's profile does. An eager policy initializer fails this test.
const bool coherent_before_main = initial("STRATA_VERIFY_COHERENT");
const bool store_before_main = initial("STRATA_DOORBELL_STORE");

static void set(const char* name, bool enabled) {
#ifdef _WIN32
    _putenv_s(name, enabled ? "1" : "0");
#else
    setenv(name, enabled ? "1" : "0", 1);
#endif
}

int main() {
    set("STRATA_VERIFY_COHERENT", !coherent_before_main);
    set("STRATA_DOORBELL_STORE", !store_before_main);
    if (strata::core::hip_verify_policy::coherent() == coherent_before_main ||
        strata::core::hip_verify_policy::doorbell_store() == store_before_main) return 1;
    // A captured verifier must keep its allocation/graph policy thereafter.
    set("STRATA_VERIFY_COHERENT", coherent_before_main);
    set("STRATA_DOORBELL_STORE", store_before_main);
    if (strata::core::hip_verify_policy::coherent() == coherent_before_main ||
        strata::core::hip_verify_policy::doorbell_store() == store_before_main) return 2;
    return 0;
}
