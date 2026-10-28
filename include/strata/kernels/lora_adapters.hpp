// include/strata/kernels/lora_adapters.hpp - llama.cpp's adapter GGUFs read into the tables `lora_upload` takes.
//
// The loader is parameterised by the model, not by the engine: `strata::core::ModelGeometry` is what the GGUF reader
// fills for either family, and the adapter itself names the projections it carries (`blk.<l>.ssm_out.weight` on a
// GDN layer, `blk.<l>.attn_output.weight` on a full-attention one).  So the one loader serves both engine paths -
// Flash-Next's generate/verify/prefill and Qwen3.6's own driver (src/qwen35) - and a file built for another
// architecture is refused by name instead of being half-applied.
#pragma once

#include <cstdint>
#include <string>
#include <utility>
#include <vector>

#include "strata/core/geometry.hpp"

namespace strata::kernels {

/// `--lora FILE` / `--lora-scaled FILE:SCALE`, one entry per adapter; several files' ranks are summed.
using LoraAdapterFiles = std::vector<std::pair<std::string, float>>;

/// Read `files` and upload their tables (`lora_upload`).  `summary` is what the start-up INFO reports (the number of
/// adapted layers), `digest` identifies the uploaded tables (0 means none was loaded).  Fails on a tensor the engine
/// does not adapt, on a shape that does not match layer l, and on a file built for another architecture - an adapter
/// applied to part of a model gives neither model.
bool load_lora_adapters(const LoraAdapterFiles& files, const core::ModelGeometry& g, std::string& summary,
                        uint64_t& digest, std::string& err);

}  // namespace strata::kernels
