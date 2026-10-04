// dsv4/config.hpp - DeepSeek-V4 config read from the GGUF header, and the per-layer expert byte inventory.
#pragma once
#include "dsv4/gguf_header.hpp"

#include <cstdint>
#include <string>
#include <vector>

namespace dsv4 {

struct Dsv4Config {
    std::string arch;
    int64_t n_ctx_train = 0;
    int n_layer = 0, n_embd = 0, n_head = 0, n_head_kv = 0;
    int key_len = 0, val_len = 0, rope_dim = 0;
    int q_lora = 0, o_lora = 0, o_groups = 0, sliding_window = 0;
    int n_expert = 0, n_expert_used = 0, n_expert_shared = 0, ff_exp = 0;
    int gating_func = 0;          // GGUF id as stored; 4 is ASSUMED to be sqrtsoftplus (HF scoring_func) - see docs
    float exp_w_scale = 1.0f;
    bool exp_w_norm = false;
    int n_hash_layers = 0;
    int idx_heads = 0, idx_key_len = 0, idx_topk = 0;
    int hc_count = 0, hc_sinkhorn_iters = 0;
    float hc_eps = 0, rms_eps = 0;
    float rope_base = 0, rope_base_compress = 0;
    float yarn_factor = 0, yarn_beta_fast = 0, yarn_beta_slow = 0;
    int64_t yarn_orig_ctx = 0;
    std::vector<int> compress_ratios;       // n_layer (+ extra layers, e.g. MTP) entries
    std::vector<float> swiglu_clamp_exp;    // per layer
    std::vector<float> swiglu_clamp_shexp;  // per layer
    int n_extra_layers = 0;                 // compress_ratios.size() - n_layer  (INFERRED: the MTP layer)
    int64_t vocab = 0;
    int bos = -1, eos = -1, pad = -1;
};

bool config_from_gguf(const GgufHeader& h, Dsv4Config& c, std::string& err);
std::string config_summary(const Dsv4Config& c);

/// Bytes of the routed-expert tensors (every 3D tensor whose name contains "_exps" and whose last dim is
/// n_expert), per layer, taken from the REAL tensor shapes and ggml types - never from metadata.
struct ExpertInventory {
    std::vector<uint64_t> bytes_per_expert;   // [n_layer]: gate+up+down for ONE expert; 0 if the layer has none
    std::vector<std::string> types;           // [n_layer]: e.g. "IQ1_M/IQ1_M/IQ2_XXS" (gate/up/down)
    std::vector<uint64_t> bytes_per_expert_extra;  // layers >= n_layer (MTP), index = layer - n_layer
    uint64_t total_expert_bytes = 0;          // layers [0, n_layer) only
    uint64_t extra_expert_bytes = 0;          // MTP / extra layers
    uint64_t other_bytes = 0;                 // every other tensor
    int layers_without_experts = 0;
    int unknown_type_tensors = 0;
};

bool inventory_from_gguf(const GgufHeader& h, const Dsv4Config& c, ExpertInventory& inv, std::string& err);

}  // namespace dsv4
