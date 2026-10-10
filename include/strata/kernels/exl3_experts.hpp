// include/strata/kernels/exl3_experts.hpp - one layer's routed EXL3 experts resident on the GPU.
//
// The engine's CPU expert pool is blob/quantized-activation oriented; EXL3 experts are per-expert
// trellis/suh/svh tensors (docs/EXL3.md), so this is the GPU-side counterpart: upload a layer's experts
// once and run the MoE (`exl3_moe_ffn`) for a token's routed experts.
#pragma once

#include <cstdint>
#include <string>

#include "strata/kernels/exl3.hpp"

namespace strata::kernels {

class Exl3ExpertStore {
public:
    // Uploads `n_experts` experts of `layer` from the EXL3 model at `model_dir` to the GPU.  `stream` may
    // be null (default stream).  `layer` is the transformer layer index; expert keys are
    // `model.language_model.layers.<layer>.mlp.experts.<e>.{gate,up,down}_proj`.
    Exl3ExpertStore(const std::string& model_dir, int layer, int n_experts, void* stream);
    ~Exl3ExpertStore();
    Exl3ExpertStore(const Exl3ExpertStore&) = delete;
    Exl3ExpertStore& operator=(const Exl3ExpertStore&) = delete;

    int n_experts() const { return n_experts_; }
    // Bytes of GPU memory the layer's experts occupy (for the engine's budget).
    size_t bytes() const { return bytes_; }

    // out = sum_i weights[i] * ffn(ids[i], x); x: 2560 fp16, out: 2560 fp16.  ids are expert indices in
    // [0, n_experts), k of them.  `stream` may be null.
    void run(const uint16_t* x, const int* ids, const float* weights, int k, uint16_t* out, void* stream) const;

private:
    struct Impl;
    Impl* impl_ = nullptr;
    int n_experts_ = 0;
    size_t bytes_ = 0;
};

// Streaming EXL3 experts for the engine's pool: reads a token's routed experts straight from the model's
// safetensors (mmap), uploads just those, runs them, and hands back the k UNWEIGHTED rows in fp32 (the
// engine's `moe_combine` applies the router weights).  One instance serves all layers.
class Exl3ExpertStream {
public:
    Exl3ExpertStream(const std::string& model_dir, int n_embd, void* stream);
    ~Exl3ExpertStream();
    Exl3ExpertStream(const Exl3ExpertStream&) = delete;
    Exl3ExpertStream& operator=(const Exl3ExpertStream&) = delete;

    // out[i*n_embd] = ffn(layer, expert ids[i], x) for i in [0,k).  `x_f32` and `out` are HOST fp32.
    bool run(int layer, const float* x_f32, const int* ids, int k, float* out, void* stream);
    size_t bytes() const;

private:
    struct Impl;
    Impl* impl_ = nullptr;
};

}  // namespace strata::kernels
