#pragma once

#include "strata/core/layer.hpp"

#include <cstdint>
#include <string>
#include <vector>

namespace strata::core {

/// Geometry of the dense Qwen3.5 27B text model in
/// unsloth/Qwen3.8-27B-GGUF.
///
/// 64 layers, with every fourth layer using gated full attention.
/// The other 48 layers use the same Gated DeltaNet geometry already
/// implemented by Strata's GDN kernels.
struct Qwen35Geometry {
    int64_t n_embd = 5120;
    int64_t n_layers = 64;
    int64_t full_attention_interval = 4;

    int64_t ssm_state_size = 128;
    int64_t ssm_k_heads = 16;
    int64_t ssm_v_heads = 48;
    int64_t ssm_d_conv = 4;
    int64_t ssm_conv_channels = 10240;
    int64_t ssm_value_dim = 6144;

    int64_t n_head = 24;
    int64_t n_head_kv = 4;
    int64_t head_dim = 256;
    int64_t n_rot = 64;

    int64_t n_ff = 17408;
    int64_t vocab_size = 248320;

    bool is_full_attention(int64_t layer) const {
        return layer % full_attention_interval == full_attention_interval - 1;
    }
    int64_t n_full_attention_layers() const { return n_layers / full_attention_interval; }
    int64_t n_gdn_layers() const { return n_layers - n_full_attention_layers(); }
};

/// Native Strata engine for the dense Qwen3.5 27B GGUF.
///
/// The class owns the resident model state and is intentionally separate
/// from the existing Flash-Next engine: the latter is an MoE/QSA runtime,
/// while Qwen3.5 is dense/GDN/full-attention.
class Qwen35Runtime {
public:
    Qwen35Runtime() = default;
    ~Qwen35Runtime();

    Qwen35Runtime(const Qwen35Runtime&) = delete;
    Qwen35Runtime& operator=(const Qwen35Runtime&) = delete;

    bool init(const std::string& pack_dir, int64_t max_context, int64_t resident_layers,
              std::string& err);
    bool generate(const std::vector<int64_t>& prompt, int max_new, float temperature,
                  float top_p, int top_k, float min_p, uint64_t seed,
                  std::vector<int32_t>& output, std::string& finish, std::string& err);

    int64_t max_context() const { return max_context_; }
    int64_t vocab_size() const { return g_.vocab_size; }

private:
    struct LayerSlot {
        void* arena = nullptr;
        uint64_t bytes = 0;
        WeightTable * weights = nullptr;
        bool resident = false;
    };

    bool load_global_weights(std::string& err);
    bool prepare_layers(int64_t resident_layers, std::string& err);
    bool load_layer(int64_t layer, WeightTable*& table, void*& arena, uint64_t& bytes,
                    std::string& err);
    bool eval_token(int64_t token, int64_t pos, float* hidden, std::string& err);
    bool eval_layer(int64_t layer, float* hidden, std::string& err);
    bool output_logits(float* hidden, float* logits, std::string& err);

    std::string pack_dir_;
    Qwen35Geometry qg_;
    ModelGeometry g_; // adapter geometry for the existing GDN/weight helpers
    int64_t max_context_ = 0;

    WeightTable global_weights_;
    void* global_arena_ = nullptr;
    uint64_t global_bytes_ = 0;

    std::vector<std::string> index_names_;
    std::vector<LayerSlot> layers_;
    void* transient_arena_ = nullptr;
    WeightTable * transient_weights_ = nullptr;
    uint64_t transient_bytes_ = 0;

    GdnBuffers gdn_;
    void* gdn_arena_ = nullptr;
    float* gdn_states_ = nullptr;
    uint64_t gdn_state_bytes_ = 0;

    void* attn_cache_ = nullptr;
    float* attn_scores_ = nullptr;
    float* attn_q_ = nullptr;
    float* attn_k_ = nullptr;
    float* attn_v_ = nullptr;
    float* attn_gate_ = nullptr;
    float* attn_out_ = nullptr;
    float* attn_qfull_ = nullptr;
    uint8_t* attn_q8_0_ = nullptr;

    float* ffn_gate_ = nullptr;
    float* ffn_up_ = nullptr;
    float* ffn_hidden_ = nullptr;
    uint8_t* ffn_q8_0_ = nullptr;
    uint8_t* ffn_up_q8_0_ = nullptr;

    float* hidden_scratch_ = nullptr;
    float* residual_ = nullptr;
    float* logits_dev_ = nullptr;
    int* token_dev_ = nullptr;

    float* rope_cos_ = nullptr;
    float* rope_sin_ = nullptr;
    int* rope_pos_ = nullptr;

    void* stream_ = nullptr;

    // Host-side copies used only for the resident-layer index and sampling.
    std::vector<int32_t> history_;
    uint64_t sample_counter_ = 0;
    int64_t current_position_ = 0;
};

/// Entry point used by src/program/generate.cpp when `--family qwen35` is present.
int qwen35_main(int argc, char ** argv);

} // namespace strata::core
