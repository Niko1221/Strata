// include/strata/artifact/exl3_pack.hpp - build the engine's WeightTable from an EXL3 model (docs/EXL3.md).
//
// The engine consumes a name -> WeightRef map.  This builds one directly from a turboderp EXL3 model
// (safetensors + index.json), instead of the canonical `<pack>/index.txt`:
//   * every EXL3 linear is uploaded to the GPU as a `strata::kernels::Exl3Mat` and attached via
//     `WeightRef::exl3` (the `gemv_quantized` seam runs it);
//   * every non-EXL3 tensor (norms, router, hyper-connections, GDN conv/A/dt, shared-expert gate, PLE
//     weights, embedding) is staged into a device arena in the ENGINE form the kernels read
//     (F32 / BF16 / F16), applying the HF -> engine transform (transpose, conv1d reshape, indexer split).
//
// The HF module names and the transforms are the table below; `tools/exl3/roles.py` verifies the same
// mapping against the model (1078/1078 roles).
#pragma once

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "strata/core/weights.hpp"

namespace strata {

class Exl3Model;

namespace core {

class Exl3Pack {
public:
    explicit Exl3Pack(const std::string& dir);
    ~Exl3Pack();
    Exl3Pack(const Exl3Pack&) = delete;
    Exl3Pack& operator=(const Exl3Pack&) = delete;

    /// Populate `wt` (engine names).  Uploads EXL3 linears and stages the rest into a device arena this owns.
    /// `stream` may be null.  `layer_lo/hi` restrict the layer range for a split (default: all).
    bool build(WeightTable& wt, std::string& err, void* stream = nullptr);

    /// Device bytes for the EXL3 linears + the staged arena (for the engine's budget/report).
    uint64_t exl3_bytes() const { return exl3_bytes_; }
    uint64_t arena_bytes() const { return arena_bytes_; }
    /// Bytes of GPU memory held by all per-layer expert stores, once `experts_ready`.
    uint64_t expert_bytes() const { return expert_bytes_; }

    struct Impl;

private:
    std::unique_ptr<Impl> impl_;
    uint64_t exl3_bytes_ = 0;
    uint64_t arena_bytes_ = 0;
    uint64_t expert_bytes_ = 0;
};

}  // namespace core
}  // namespace strata
