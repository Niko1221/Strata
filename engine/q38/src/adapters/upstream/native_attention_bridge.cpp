// adapters/upstream/native_attention_bridge.cpp - Native attention & cache bridge isolating Cl(8,0) Spin8 & TurboQuant
#include <cstdint>
#include <string>

namespace strata::adapters {

struct AttentionFeatureMatrix {
    bool supports_spin8_triality    = false;
    bool supports_turboquant_tq4    = false;
    bool supports_dsv41_shared_attn = false;
};

inline AttentionFeatureMatrix get_backend_feature_matrix(const std::string& backend_id) {
    AttentionFeatureMatrix matrix;
    if (backend_id == "strata-native") {
        matrix.supports_spin8_triality = true;
        matrix.supports_turboquant_tq4 = true;
        matrix.supports_dsv41_shared_attn = false;
    } else if (backend_id == "strata-dsv41") {
        // DeepSeek uses native multi-head latent attention (MLA); Spin8 is not substituted
        matrix.supports_spin8_triality = false;
        matrix.supports_turboquant_tq4 = false;
        matrix.supports_dsv41_shared_attn = true;
    } else if (backend_id == "prism") {
        matrix.supports_spin8_triality = false;
        matrix.supports_turboquant_tq4 = true; // PQ2_0 Bonsai support
        matrix.supports_dsv41_shared_attn = false;
    }
    return matrix;
}

} // namespace strata::adapters
