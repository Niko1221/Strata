// include/strata/core/qwen35.hpp - the Qwen35MoE (Ornith-1.5) architecture: identity, geometry and guards.
//
// This is the Qwen4Exp split applied to the other family.  It is deliberately SEPARATE from `layout.hpp`:
// `ModelGeometry` stays the Qwen4Exp geometry, and none of its fields (hc, PLE, QSA indexer, 512 experts,
// 2560 hidden) are allowed to leak into the Qwen35 path.  The reference for every number here is
// `src/models/qwen35moe.cpp` in the llama.cpp revision this repository pins (`third_party/ggml/VERSION.txt`)
// and the artifact's own GGUF metadata; `docs/ornith/gguf-layout.txt` records the artifact it was written
// against.
//
// The checks are not a restatement of `general.architecture == "qwen35moe"`.  They are the shapes the
// kernels depend on, validated at LOAD time so a mismatch names the tensor, what it has and what is
// required - the same contract `layout.cpp`'s `check_layer`/`check_all` gives Qwen4Exp.
#pragma once

#include "strata/artifact/gguf_reader.hpp"
#include "strata/core/model_kind.hpp"

#include <cstdint>
#include <string>

namespace strata::core {

/// Which architecture a metadata shard is.  Reads `general.architecture`; Qwen4Exp and Qwen35MoE are the only
/// two this engine admits, anything else is `Unknown` and must be refused by the caller.
ModelKind detect_model_kind(const GgufFile& meta_shard);

/// The Qwen35MoE geometry, taken from the artifact's metadata.  Every field is a number a kernel depends on.
struct Qwen35Geometry {
    int64_t n_layers = 0;
    int64_t n_embd = 0;               ///< residual width
    int64_t n_expert = 0;             ///< routed experts per layer
    int64_t n_expert_used = 0;        ///< routed experts selected per token
    int64_t n_ff_exp = 0;             ///< one routed expert's intermediate width
    int64_t n_ff_shexp = 0;           ///< the shared expert's intermediate width
    int64_t full_attention_interval = 4;
    int64_t n_head = 0;               ///< full-attention query heads
    int64_t n_head_kv = 0;
    int64_t head_dim = 0;             ///< key/value head width
    int64_t rope_dim = 0;             ///< rotary width (a partial-rotary model has rope_dim < head_dim)
    int64_t rope_sections[4] = {0, 0, 0, 0};   ///< MRoPE dimension split
    double rope_freq_base = 0.0;
    int64_t context_length = 0;
    // Gated DeltaNet (the recurrent layers)
    int64_t ssm_conv_kernel = 0;
    int64_t ssm_inner = 0;            ///< value dimension (n_v_heads * head_v_dim)
    int64_t ssm_state = 0;            ///< head_k_dim == head_v_dim
    int64_t ssm_dt_rank = 0;          ///< number of value heads (num_v_heads)
    int64_t ssm_groups = 0;           ///< number of key heads (num_k_heads)
    float rms_eps = 0.0f;
    int64_t n_vocab = 0;              ///< from token_embd.weight's output dimension
    bool output_tied = false;         ///< true when output.weight is absent (the embedding is the head)

    /// GDN projected dimensions, derived (llama.cpp: key_dim = head_k_dim*n_k_heads, value_dim = inner_size).
    int64_t key_dim() const { return ssm_state * ssm_groups; }
    int64_t value_dim() const { return ssm_inner; }
    /// The qkv projection's output width: q + k + v = 2*key_dim + value_dim (llama.cpp: key_dim*2 + value_dim).
    int64_t qkv_dim() const { return 2 * key_dim() + value_dim(); }
    int64_t conv_channels() const { return qkv_dim(); }

    /// llama.cpp marks layers `(i+1) % interval != 0` recurrent ("linear attention"), the rest full attention.
    bool is_recurrent(int64_t layer) const { return (layer + 1) % full_attention_interval != 0; }
    int64_t n_attention_layers() const { return n_layers / full_attention_interval; }
    int64_t n_recurrent_layers() const { return n_layers - n_attention_layers(); }
};

/// Parses and validates the metadata.  Returns false with a specific reason on any missing or inconsistent
/// key - it never fills a field with a default when the artifact is silent.
bool qwen35_geometry(const GgufFile& meta_shard, Qwen35Geometry& g, std::string& err);

/// Every shape and type the Qwen35 kernels depend on, asserted for the whole trunk.  `model` may be split;
/// tensors are looked up across all shards.  On failure `err` names the tensor, the actual shape/type and
/// the required shape/type.  It also refuses Qwen4Exp-only tensors appearing in a Qwen35 block.
bool check_qwen35_tensors(const GgufModel& model, const Qwen35Geometry& g, std::string& err);

/// `detect_model_kind` + `qwen35_geometry` + `check_qwen35_tensors`.  The entry point a loader calls before
/// it allocates anything.
bool check_qwen35_all(const GgufModel& model, Qwen35Geometry& g, std::string& err);

}  // namespace strata::core
