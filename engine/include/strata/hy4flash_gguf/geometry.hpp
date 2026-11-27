// Model geometry decoded from GGUF metadata.
#pragma once

#include <cstdint>
#include <string>
#include <vector>

namespace strata {
class GgufFile;
struct TensorInfo;
}

namespace strata::hy4_flash {

enum class Attention {
    Sliding,   ///< 64 heads, rope theta 1e4, a window of `sliding_window` positions
    Sparse,    ///< 32 heads, rope theta 1e7, full history selected by the compressed indexer
};

/// One block of the model, as the file describes it.
struct LayerSpec {
    int32_t block_id = -1;                      ///< 0..block_count-1; the last one is the MTP block
    Attention attention = Attention::Sparse;
    int32_t n_heads = 0;                        ///< from the per-block array, not from a scalar
    int32_t expert_ordinal = -1;                ///< -1 = the dense FFN block; else its index in the expert pack
    int32_t engram_table = -1;                  ///< -1 = no table; else 0..n_engram_tables-1
    bool has_indexer = false;                   ///< the compressed indexer, on the sparse blocks
    bool is_mtp = false;                        ///< the draft block: no hyper-connections, plain hidden residual
};

/// One layer's view into the merged per_layer_token_embd.weight table.
struct EngramTable {
    int32_t block_id = -1;
    uint64_t row_base = 0;                      ///< this table's first row in the merged tensor
    uint64_t rows = 0;                          ///< its own row count (the four differ)
    std::vector<uint64_t> head_offsets;         ///< columns entries, global rows
    std::vector<uint64_t> head_bucket_counts;   ///< columns entries; a column's hash is taken modulo this
    std::vector<uint64_t> multipliers;          ///< ngram_size entries, from the file (never re-derived: the
                                                ///< converter draws them from numpy's PCG64)
};

struct Geometry {
    // Widths
    int32_t hidden_dim = 0;
    int32_t dense_ff = 0;                       ///< feed_forward_length: the leading dense block's FFN (13824)
    int32_t vocab_size = 0;
    int32_t hc_mult = 0;                        ///< hyper_connection.count; 0 means the file declared none
    float hc_eps = 0, hc_magnitude = 0;
    int32_t expert_dim = 0;                     ///< the routed experts' subspace (expert_latent_length)
    int32_t expert_ff = 0;
    int32_t n_experts = 0, top_k = 0, n_shared_experts = 0;
    float swiglu_clamp = 0;                     ///< swiglu_clamp_exp, which is one value per block in the file
    int32_t rope_dim = 0;
    float rope_theta_sparse = 0, rope_theta_sliding = 0;
    float routed_scale = 0, rms_norm_eps = 0;
    bool routed_scale_normalized = false;       ///< expert_weights_norm: renormalise the top-k before scaling
    int32_t gating_func = 0;                    ///< 2 = sigmoid (llama.cpp's LLAMA_EXPERT_GATING_FUNC_TYPE)

    /// Read turn-ending token IDs from the model's tokenizer metadata.
    std::vector<int32_t> stop_tokens;
    int32_t eos_token_id = -1;

    // MLA
    int32_t q_lora_rank = 0, kv_lora_rank = 0;
    int32_t qk_head_dim = 0;                    ///< attention.key_length_mla (256 = 192 nope + 64 rope)
    int32_t v_head_dim = 0;                     ///< attention.value_length_mla
    int32_t kv_a_width = 0;                     ///< attention.key_length: kv_lora_rank + rope_dim, one shared key

    // The compressed indexer, on the sparse blocks
    int32_t idx_heads = 0, idx_key_dim = 0, idx_kpool = 0, idx_top_k = 0;
    bool idx_select_tail = false;

    // Blocks
    int32_t block_count = 0;                    ///< including the MTP draft block
    int32_t trunk_blocks = 0;                   ///< excluding it
    int32_t n_mtp_blocks = 0;
    int32_t dense_blocks = 0;                   ///< the leading blocks with a plain FFN
    int32_t sliding_window = 0;
    std::vector<LayerSpec> layers;              ///< block_count entries

    // Engram
    int32_t engram_heads = 0;                   ///< columns per token per table ((ngram_size - 1) * heads_per_ngram)
    int32_t engram_head_dim = 0;                ///< 96
    int32_t engram_ngram_size = 0;
    int32_t engram_heads_per_ngram = 0;
    int32_t engram_pad_token = -1;
    uint64_t engram_total_rows = 0;             ///< the merged tensor's row count
    uint32_t engram_table_type = 0;             ///< the merged tensor's ggml type, when this shard holds it
    uint32_t engram_row_bytes = 0;              ///< one row's size in the file; 0 when the type is not known here
    std::vector<EngramTable> engram_tables;

    // The engine's own limit, not the file's
    int32_t max_context = 0;

    /// Predecessor tokens a request must keep for the hash: ngram_size - 1.
    int32_t engram_predecessors() const { return engram_ngram_size > 0 ? engram_ngram_size - 1 : 0; }
    /// The width of the state carried between blocks. Use this, never hidden_dim, to size a buffer.
    int32_t hc_width() const { return hidden_dim * (hc_mult > 0 ? hc_mult : 1); }
    /// How many entries the expert pack holds (the trunk's MoE blocks plus the MTP block).
    int32_t n_expert_ordinals() const { return trunk_blocks - dense_blocks + n_mtp_blocks; }
    /// Lookup features per token per Engram block: engram_heads x engram_head_dim (16 x 96 = 1536).
    int32_t engram_lookup_dim() const { return engram_heads * engram_head_dim; }
    /// Heads of the block whose spec is `s`, as its own entry in the per-block array gave it.
    const LayerSpec& block(int32_t id) const { return layers[(size_t) id]; }
};

/// Return an empty string on success or the first invalid metadata field.
std::string read_geometry(const GgufFile& f, Geometry& g, int32_t max_context,
                          const TensorInfo* engram_tensor = nullptr);

/// A one-line-per-fact summary for `--help`-style output and for the refusal message.
std::string describe(const Geometry& g);

}  // namespace strata::hy4_flash
