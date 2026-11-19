#pragma once
// adapters/upstream/dsv41_steering_bridge.hpp - Isolated steering bridge for DeepSeek architecture
#include <cstdint>
#include <string>
#include <vector>

namespace strata::dsv41 {

struct SteeringBridgeConfig {
    bool enabled                    = false;
    std::string target_architecture  = "dsv41";
    uint32_t hidden_dim             = 2048;
    std::string profile_hash        = "";
};

inline bool validate_deepseek_vector_compatibility(
    const std::string& model_arch,
    uint32_t vector_dim,
    const SteeringBridgeConfig& cfg
) {
    // Rule: Width match is not enough; architecture family must strictly match dsv41!
    if (model_arch != "dsv41" && model_arch != "deepseek") {
        return false;
    }
    if (vector_dim != cfg.hidden_dim) {
        return false;
    }
    return true;
}

} // namespace strata::dsv41
