// include/strata/core/steer.hpp - per-layer control vectors applied to the residual stream.
//
// A control-vector file is a GGUF of `direction.<L>` tensors, one n_embd-wide F32/F16/BF16 vector
// per layer (the abliterate format: `general.architecture` is `control-vector` or `controlvector`).
// Each GGUF `direction.<L>` is normalized, then the scaled component along it is projected out of
// each hyper-connection residual stream after that layer's FFN write. `@scale` controls projection
// strength; `--control-vector-layer-range` restricts which layer directions are loaded.
//
// The plan is static for the process: vectors are uploaded once and the per-layer projections are
// ordinary stream work inside captured graphs. A plan that is not loaded makes the call return
// before touching anything, leaving runs without control vectors unchanged.
#pragma once

#include "strata/core/layout.hpp"

#include <string>
#include <vector>

namespace strata::core {

struct SteeringVector {
    int64_t layer = 0;   ///< the `direction.<layer>` applied to the layer residual
    float alpha = 1.0f;  ///< the `@scale` from the spec
    const float* dev = nullptr;  ///< device, n_embd floats (one arena, offsets into it)
    std::string spec;    ///< the spec it came from, for messages
};

class SteeringPlan {
public:
    /// Parses `path.gguf[:scale]` specs, loads each file's `direction.<L>` tensors, checks
    /// the width against the geometry and uploads the vectors to one device arena.  Static for
    /// the process: call once, before session capture.
    ///
    /// `layer_first`/`layer_last` are the llama.cpp `--control-vector-layer-range` pair, INCLUSIVE
    /// on both ends: directions whose layer falls outside are skipped entirely (they cost nothing,
    /// they are never uploaded).  Negative first = no limit.
    bool load(const std::vector<std::string>& specs, const ModelGeometry& g, std::string& err,
              int64_t layer_first = -1, int64_t layer_last = -1);
    void free();
    ~SteeringPlan() { free(); }

    const std::vector<SteeringVector>& entries() const { return entries_; }
    bool empty() const { return entries_.empty(); }

private:
    std::vector<SteeringVector> entries_;
    std::vector<std::vector<float>> host_;  ///< staging while the arena is still being collected
    void* arena_ = nullptr;   ///< one device allocation, the vectors back to back
    int64_t arena_bytes_ = 0;
};

/// The process-global plan (null = no steering).  Set before capture; the same pattern as
/// `set_native_embed`.
void steer_set_active(const SteeringPlan* plan);
const SteeringPlan* steer_active();

/// Apply llama.cpp-style cvec operation to every hyper-connection stream after the layer's FFN write.
bool cvec_residual(float* residual, int64_t tokens, int64_t hc, int64_t layer, int64_t n_embd,
                   void* stream, std::string& err);

}  // namespace strata::core
