// include/strata/dsv4/geometry.hpp - DeepSeek V4-Flash's geometry: one struct for the shape a model file
// declares and the shape the dsv4 kernels are compiled for.
//
// The dsv4 counterpart of core/geometry.hpp, family-scoped on purpose: V4's attention stack (MLA + DSA
// compressor/indexer) shares no dimension with qwen4exp's GDN/QSA stack, so appending fields to ModelGeometry
// would make that struct three models wide.  The contract is the same: the numbers a kernel depends on are
// pinned here with a static_assert next to the kernel, `check_architecture_deepseek4` (gguf_reader.hpp)
// refuses a file whose metadata disagrees with `kDsv4CompiledGeometry`, and the layer split is derived, never
// read from a second list.
//
// Every value was established from the real checkpoint's metadata and tensor shapes (the 164 GB
// dsv4-flash-nvfp4.gguf, 2026-10-20), not from the config comments: the compress_ratios array is 46 entries
// (43 layers + 3 dspark stage zeros), layer 2 is the hash+DSA hybrid, and the ratio/indexer mapping is
// r=4 -> compressor + indexer, r=128 -> compressor only.
#pragma once

#include <cstdint>

namespace strata::dsv4 {

/// Every field is a number a dsv4 kernel depends on; the GGUF key each is read from is named beside it.
struct Dsv4Geometry {
    int64_t n_embd = 4096;          ///< embedding_length
    int64_t n_embd_out = 16384;     ///< embedding_length_out: the hyper-connection streams, hc * n_embd
    int64_t n_layers = 43;          ///< block_count (the trunk; dspark stages are extra zero ratios, not layers)
    int64_t n_vocab = 129280;       ///< the length of tokenizer.ggml.tokens
    double rms_eps = 1e-6;          ///< attention.layer_norm_rms_epsilon
    double hc_eps = 1e-6;           ///< hyper_connection.epsilon

    // MLA attention (every layer)
    int64_t n_head = 64;            ///< attention.head_count
    int64_t head_dim = 512;         ///< attention.key_length (= attention.value_length), the per-head width
    int64_t q_lora_rank = 1024;     ///< attention.q_lora_rank (attn_q_a 4096->1024, attn_q_b 1024->64*512)
    int64_t kv_lora_rank = 512;     ///< attn_kv 4096->512 (the KV latent)
    int64_t n_rot = 64;             ///< rope.dimension_count: the rotated slice of head_dim
    int64_t sliding_window = 128;   ///< attention.sliding_window
    int64_t o_groups = 8;           ///< attention.output_group_count
    int64_t o_lora_rank = 1024;     ///< attention.output_lora_rank (attn_output_a 4096->8192 = o_groups*o_lora)

    // rope / yarn
    double rope_theta = 10000.0;         ///< rope.freq_base
    double compress_rope_theta = 160000.0;  ///< attention.compress_rope_freq_base
    double yarn_factor = 16.0;           ///< rope.scaling.factor
    int64_t yarn_orig_ctx = 65536;       ///< rope.scaling.original_context_length
    double yarn_beta_fast = 32.0;        ///< rope.scaling.yarn_beta_fast
    double yarn_beta_slow = 1.0;         ///< rope.scaling.yarn_beta_slow

    // DSA indexer (the even, ratio-4 layers)
    int64_t idx_heads = 64;         ///< attention.indexer.head_count
    int64_t idx_key_dim = 128;      ///< attention.indexer.key_length
    int64_t idx_top_k = 512;        ///< attention.indexer.top_k

    // hyper-connections (every layer, and the head)
    int64_t hc = 4;                     ///< hyper_connection.count
    int64_t hc_sinkhorn_iters = 20;     ///< hyper_connection.sinkhorn_iterations

    // MoE (every layer)
    int64_t n_expert = 256;         ///< expert_count
    int64_t n_expert_used = 6;      ///< expert_used_count
    int64_t n_expert_shared = 1;    ///< expert_shared_count
    int64_t n_ff = 2048;            ///< expert_feed_forward_length
    double route_scale = 1.5;       ///< expert_weights_scale
    int64_t n_hash_layers = 3;      ///< hash_layer_count: layers 0,1 pure hash, layer 2 the hash+DSA hybrid
    double swiglu_clamp = 10.0;     ///< swiglu_clamp_exp / swiglu_clamp_shexp (per-layer arrays, all 10.0)

    constexpr int64_t hc_dim() const { return hc * n_embd; }
    constexpr int64_t q_rows() const { return n_head * head_dim; }        ///< 32768, attn_q_b's out rows
    constexpr int64_t o_rows() const { return o_groups * o_lora_rank; }   ///< 8192, attn_output_a's out rows
};

/// The shape the dsv4 kernels are compiled for.  A kernel that fixes a dimension asserts it here.
inline constexpr Dsv4Geometry kDsv4CompiledGeometry{};

/// The layer split, derived (the GGUF's compress_ratios is checked against this by the gate):
/// layers 0-1 pure hash (no compressor); layer 2 the hash + DSA hybrid; even layers 2..42 carry the
/// compressor AND the indexer (ratio 4); odd layers 3..41 the compressor only (ratio 128 - a 128-wide
/// pooling block needs no indexer pass, attention runs over the compressed KV directly).
inline constexpr int64_t dsv4_compress_ratio(int64_t layer) {
    return layer < 2 ? 0 : (layer % 2 == 0 ? 4 : 128);
}
inline constexpr bool dsv4_is_hash_layer(int64_t layer) { return layer < 3; }
inline constexpr bool dsv4_has_indexer(int64_t layer) { return dsv4_compress_ratio(layer) == 4; }

}  // namespace strata::dsv4
