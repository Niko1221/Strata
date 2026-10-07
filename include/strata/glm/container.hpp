// include/strata/glm/container.hpp - the GLM-5.3 int4-g64 container: config.json and the safetensors shards.
//
// The engine reads this container IN PLACE.  tools/glm53_pack.py describes a repacked experts.bin, but on the
// measured checkpoint that is a 408 GB copy of a 419 GB folder, and nothing is gained by it: safetensors orders a
// shard's tensors by dtype and then by name, so one expert's three code planes (down, gate, up) are usually ONE
// contiguous 18,874,368 B span and its three scale planes ONE contiguous 2,359,296 B span: an expert is two reads
// (measured on out-00060.safetensors, layer 40 expert 17).  Experts that straddle a shard boundary take more
// (`ExpertSpan` groups whatever is adjacent).
//
// The format, measured (docs/GLM53.md):
//   - a quantized tensor is `name` (U8, 4-bit codes, two per byte, LSB first, stored as q + 8) and `name.qs`
//     (F32, one scale per group of 64 along the input dimension);
//   - `model.embed_tokens.weight` and `lm_head.weight` are int8: signed bytes in a U8 tensor, one F32 scale per row;
//   - norms, the router weight and its e_score_correction_bias are F32.
#pragma once

#include "strata/glm/json.hpp"

#include <cstdint>
#include <string>
#include <unordered_map>
#include <vector>

namespace strata::glm {

/// What config.json says, restricted to what the forward pass reads.  Every field is checked against a range in
/// `load_config`: the folder is user-supplied, and a hostile number must not reach an allocation.
struct GlmConfig {
    int hidden = 0;              ///< 6144
    int n_layers = 0;            ///< 78 main layers (the NextN block, when present, is not one of them)
    int n_heads = 0;             ///< 64
    int n_experts = 0;           ///< 256 routed experts per sparse layer
    int topk = 0;                ///< 8
    int moe_inter = 0;           ///< 2048
    int dense_inter = 0;         ///< 12288
    int first_dense = 0;         ///< 3: layers [0, first_dense) have a dense MLP
    int n_shared = 0;            ///< 1 shared expert, weight 1 (no shared-expert gate in this family)
    int q_lora = 0;              ///< 2048
    int kv_lora = 0;             ///< 512
    int qk_nope = 0;             ///< 192
    int qk_rope = 0;             ///< 64
    int v_head = 0;              ///< 256
    int vocab = 0;               ///< 154880
    bool norm_topk = false;      ///< true: the eight weights are renormalised to sum 1
    float routed_scale = 1.0f;   ///< 2.5, applied after the renormalisation
    float eps = 1e-5f;           ///< rms_norm_eps
    double rope_theta = 10000.0; ///< 8,000,000 on GLM-5.3
    std::vector<int> eos;        ///< config.json eos_token_id, union generation_config.json's
    // DSA indexer (absent from this container: its weights are not converted, see docs/GLM53.md)
    int index_topk = 0;
    int index_n_heads = 0;
    int index_head_dim = 0;

    int qk_head() const { return qk_nope + qk_rope; }
    bool sparse(int layer) const { return layer >= first_dense; }
};

bool load_config(const std::string& dir, GlmConfig& c, std::string& err);

/// One tensor entry of a shard header: where its bytes are, absolutely, in which shard.
struct TensorSpan {
    int shard = -1;
    uint64_t offset = 0;   ///< absolute byte offset in the shard file (8 + header length + data_offsets[0])
    uint64_t bytes = 0;
    std::string dtype;     ///< "U8", "F32", ...
    std::vector<int64_t> shape;
};

/// Every shard of a model folder, and every tensor in them by name.
class Container {
public:
    bool open(const std::string& dir, std::string& err);

    const std::string& dir() const { return dir_; }
    const std::vector<std::string>& shard_paths() const { return shards_; }
    const TensorSpan* find(const std::string& name) const;
    size_t tensor_count() const { return index_.size(); }

private:
    std::string dir_;
    std::vector<std::string> shards_;
    std::unordered_map<std::string, TensorSpan> index_;
};

/// A routed expert's place on disk: its six planes - codes of down, gate, up (planes 0..2), then their scales (3..5) -
/// grouped into contiguous runs.  Almost every expert is two runs (the three code planes, the three scale planes);
/// an expert that straddles a shard boundary has more (measured: layer 3 expert 27 does).
struct ExpertSpan {
    struct Run {
        int shard = -1;
        uint64_t off = 0, bytes = 0;
        int first = 0, count = 0;   ///< planes [first, first + count), adjacent on disk in that order
    };
    Run runs[6];
    int n_runs = 0;
};

/// Resolve and check expert `e` of `layer`: the six planes, their sizes, and the runs they form.
bool expert_span(const Container& ct, const GlmConfig& c, int layer, int e, ExpertSpan& out, std::string& err);

}  // namespace strata::glm
