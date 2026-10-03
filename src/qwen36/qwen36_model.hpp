// qwen36_model.hpp - Qwen3.6-35B-A3B (arch qwen35moe) decoder on one GPU, one token at a time (Phase 1).
#pragma once

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <memory>
#include <string>
#include <vector>

#include "qwen36_config.hpp"

namespace q36 {

struct Mat {
    int type = -1;                 // ggml type id
    const void* d = nullptr;       // device pointer
    int n_in = 0, n_out = 0;
    int n_expert = 1;
    size_t expert_stride = 0;      // bytes between consecutive experts (3-D tensors)
};

struct LayerWeights {
    bool gdn = false;
    float *attn_norm = nullptr, *post_norm = nullptr;
    // GDN
    Mat qkv, z, alpha, beta, ssm_out;
    float *conv = nullptr, *ssm_a = nullptr, *dt = nullptr, *ssm_norm = nullptr;
    // attention
    Mat q, k, v, o;
    float *qn = nullptr, *kn = nullptr;
    // MoE
    Mat gate_inp, gate_exps, up_exps, down_exps, sh_gate, sh_up, sh_down;
    float* sh_gate_inp = nullptr;  // [n_embd] vector
};

class Model {
  public:
    Model(const std::string& gguf_path, int max_ctx);
    ~Model();
    Model(const Model&) = delete;
    Model& operator=(const Model&) = delete;

    const Config& cfg() const { return cfg_; }
    int max_ctx() const { return max_ctx_; }
    int pos() const { return pos_; }
    void reset();                                   // forget everything (zero GDN state / conv history, pos = 0)
    // Feed one token at position pos(); then pos()++.  With want_logits the logits are computed and copied to the host.
    void forward(int token, bool want_logits);
    // Block prefill: feed n tokens (1 <= n <= kMaxBatch) at positions pos() .. pos()+n-1 in one pass.  Equivalent to n calls
    // of forward() (checked by --selftest-batch); with want_logits only the LAST token's logits are produced.
    static constexpr int kMaxBatch = 128;
    void forward_batch(const int* tokens, int n, bool want_logits);
    const std::vector<float>& logits() const { return h_logits_; }
    size_t vram_bytes() const { return vram_; }

  private:
    struct Impl;
    Config cfg_;
    int max_ctx_ = 0, pos_ = 0;
    size_t vram_ = 0;
    std::vector<float> h_logits_;
    std::unique_ptr<Impl> p_;
};

}  // namespace q36
