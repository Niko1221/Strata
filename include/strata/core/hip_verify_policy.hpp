#pragma once
#include <cstdlib>

namespace strata::core::hip_verify_policy {

// Read at first use, not during static initialization: the HIP stability
// profile sets these in main before verifier allocation/capture begins.
inline bool coherent() {
    static const bool enabled = [] {
        const char* value = std::getenv("STRATA_VERIFY_COHERENT");
        return value && value[0] && value[0] != '0';
    }();
    return enabled;
}

inline bool doorbell_store() {
    static const bool enabled = [] {
        const char* value = std::getenv("STRATA_DOORBELL_STORE");
        return value && value[0] && value[0] != '0';
    }();
    return enabled;
}

}  // namespace strata::core::hip_verify_policy
