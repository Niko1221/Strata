/**
 * @file config.hpp
 * @brief DeepSeek-V4 configuration extracted from GGUF metadata and per-layer expert byte inventory.
 *
 * This module parses the GGUF header of a DeepSeek-V4-Flash model file to extract all architectural
 * hyperparameters (layers, embeddings, attention heads, MoE experts, rope parameters, compression
 * ratios, etc.) and computes the exact byte size of each routed-expert tensor per layer from the
 * real tensor shapes — never from hardcoded metadata.
 *
 * @par Architecture
 *   - `config_from_gguf()` reads all `deepseek.*` keys from a `GgufHeader` and validates them.
 *   - `inventory_from_gguf()` walks every 3D tensor whose name contains `_exps` and last dimension
 *     equals `n_expert`, summing gate+up+down bytes per expert, per layer.
 * @par Usage
 *   @code
 *   GgufHeader h;
 *   gguf_read_header({"shard-00001-of-00003.gguf"}, h, err);
 *   Dsv4Config c;
 *   config_from_gguf(h, c, err);
 *   ExpertInventory inv;
 *   inventory_from_gguf(h, c, inv, err);
 *   @endcode
 */
#pragma once

#include "dsv4/gguf_header.hpp"

#include <cstdint>
#include <string>
#include <vector>

namespace dsv4 {

/**
 * @struct Dsv4Config
 * @brief Complete architectural configuration of a DeepSeek-V4-Flash model, parsed from GGUF keys.
 *
 * Every field maps directly to a `deepseek.*` key in the GGUF file. Fields not present in the
 * model receive safe defaults (0 or -1). The architecture string is validated against `"deepseek4"`.
 *
 * @see config_from_gguf() for parsing, config_summary() for human-readable output.
 */
struct Dsv4Config {
    std::string arch;                     ///< "deepseek4" — validated on parse.
    int64_t n_ctx_train = 0;              ///< Training context length (tokenizer.ggml.context_length).
    int n_layer = 0, n_embd = 0, n_head = 0, n_head_kv = 0;   ///< Transformer dimensions.
    int key_len = 0, val_len = 0, rope_dim = 0;                 ///< Attention head / RoPE dimensions.
    int q_lora = 0, o_lora = 0, o_groups = 0, sliding_window = 0;  ///< LoRA ranks & attention window.
    int n_expert = 0, n_expert_used = 0, n_expert_shared = 0, ff_exp = 0;  ///< MoE expert counts & FFN width.
    int gating_func = 0;                  ///< GGUF id for gating: 4 = sqrtsoftplus (HF scoring_func).
    float exp_w_scale = 1.0f;             ///< Expert weight scale factor.
    bool exp_w_norm = false;              ///< Whether expert weights are normalized.
    int n_hash_layers = 0;                ///< Number of layers with predetermined hash-based experts.
    int idx_heads = 0, idx_key_len = 0, idx_topk = 0;  ///< Sparse attention indexer parameters.
    int hc_count = 0, hc_sinkhorn_iters = 0;           ///< Hyper-connection streams & Sinkhorn iterations.
    float hc_eps = 0, rms_eps = 0;         ///< Hyper-connection epsilon & RMSNorm epsilon.
    float rope_base = 0, rope_base_compress = 0;  ///< RoPE base frequency (plain and compressed layers).
    float yarn_factor = 0, yarn_beta_fast = 0, yarn_beta_slow = 0;  ///< YaRN extrapolation parameters.
    int64_t yarn_orig_ctx = 0;             ///< YaRN original context length.
    std::vector<int> compress_ratios;      ///< Per-layer compression ratios (n_layer + extra MTP entries).
    std::vector<float> swiglu_clamp_exp;   ///< Per-layer SwiGLU clamp exponent.
    std::vector<float> swiglu_clamp_shexp; ///< Per-layer SwiGLU clamp shift-exponent.
    int n_extra_layers = 0;                ///< MTP (next-token prediction) layers: compress_ratios.size() - n_layer.
    int64_t vocab = 0;                     ///< Vocabulary size from tokenizer.ggml.tokens array length.
    int bos = -1, eos = -1, pad = -1;      ///< Special token ids (BOS, EOS, padding).
};

/**
 * @brief Parse Dsv4Config from a GGUF header.
 *
 * Validates that `general.architecture == "deepseek4"` and reads all `deepseek.*` keys.
 * Returns false on missing required keys or validation failures.
 *
 * @param h  Parsed GGUF header (metadata only, no tensor data).
 * @param c  Output: populated configuration struct.
 * @param err  Output error message on failure.
 * @return true if all keys present and dimensions valid; false otherwise.
 */
bool config_from_gguf(const GgufHeader& h, Dsv4Config& c, std::string& err);

/**
 * @brief Generate a human-readable summary of the model configuration.
 *
 * Outputs architecture, layer count, embedding size, MoE parameters, attention details,
 * hyper-connection settings, vocabulary size, and context length in a multi-line string.
 *
 * @param c  Configuration to summarize.
 * @return Formatted summary string.
 */
std::string config_summary(const Dsv4Config& c);

/**
 * @struct ExpertInventory
 * @brief Byte-level inventory of routed-expert tensors per layer, derived from real GGUF tensor shapes.
 *
 * Every 3D tensor whose name contains `_exps` and last dimension equals `n_expert` is classified as
 * an expert tensor (gate, up, or down). Bytes are summed per expert (total / n_expert) and grouped
 * by layer index extracted from the tensor name prefix (`blk.N_`).
 *
 * @par Structure
 *   - `bytes_per_expert[l]` = gate + up + down bytes for ONE expert at layer l.
 *   - `types[l]` = concatenated quantization types, e.g. `"IQ1_M/IQ2_XXS/IQ3_S"`.
 *   - `bytes_per_expert_extra[]` = same for MTP/extra layers (index = layer - n_layer).
 */
struct ExpertInventory {
    std::vector<uint64_t> bytes_per_expert;       ///< [n_layer]: gate+up+down for ONE expert; 0 if none.
    std::vector<std::string> types;                ///< [n_layer]: e.g. "IQ1_M/IQ2_XXS/IQ3_S".
    std::vector<uint64_t> bytes_per_expert_extra;  ///< MTP/extra layers, index = layer - n_layer.
    uint64_t total_expert_bytes = 0;               ///< Sum of expert bytes for layers [0, n_layer).
    uint64_t extra_expert_bytes = 0;               ///< Expert bytes for MTP / extra layers.
    uint64_t other_bytes = 0;                      ///< All non-expert tensor bytes.
    int layers_without_experts = 0;                ///< Layers with zero routed experts.
    int unknown_type_tensors = 0;                  ///< Tensors with unrecognized ggml quantization type.
};

/**
 * @brief Build the expert byte inventory from GGUF tensor shapes and config.
 *
 * Walks all tensors, identifies expert tensors by name pattern (`_exps`) and shape (last dim == n_expert),
 * extracts layer index from `blk.N.` prefix, and accumulates bytes per expert. Validates that every
 * expert tensor's byte count is divisible by n_expert.
 *
 * @param h  Parsed GGUF header.
 * @param c  Model configuration (provides n_expert, n_layer).
 * @param inv  Output: populated expert inventory.
 * @param err  Output error message on failure.
 * @return true if all tensors processed successfully; false otherwise.
 */
bool inventory_from_gguf(const GgufHeader& h, const Dsv4Config& c, ExpertInventory& inv, std::string& err);

}  // namespace dsv4
