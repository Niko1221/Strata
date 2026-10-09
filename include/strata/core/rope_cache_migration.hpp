#pragma once

#include "strata/core/conversation_file.hpp"
#include "strata/kernels/rope_scaling.hpp"
#include <string>

namespace strata::core {

// This operation is approximate state migration, not a fresh YaRN prefill.
// Profiles come from the resolved engine configuration, never a context-size guess.
struct RopeCacheProfile {
    SessionConfig config;
    kernels::RopeScaling rope;
    uint64_t model_identity = 0; // full-file model fingerprint in migration mode
    bool full_model_identity = false;
};

// Transform one NEOX pair using the backend's table coefficients. Kept general
// internally; the session entry point below only admits ordinary -> YaRN 4x.
void convert_rope_pair(double& x, double& y, int64_t position, int pair, int n_rot,
                       const kernels::RopeScaling& source, const kernels::RopeScaling& target);

// Stages only changed host K/indexer data. Values and canonical token IDs never
// move. All validation/allocation/conversion precedes noexcept commit. On refusal
// cache is byte-for-byte unchanged. No GPU allocation, no dense page expansion.
bool migrate_rope_cache_to_yarn4(SavedConversation& cache, const RopeCacheProfile& source,
                                 const RopeCacheProfile& target, std::string& error);

} // namespace strata::core
