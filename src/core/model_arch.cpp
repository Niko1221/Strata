// src/core/model_arch.cpp - reading a model file's own geometry.
//
// The engine used to hold its geometry as compile-time defaults plus two keys read by hand from the GGUF
// (`expert_count`, `expert_used_count`), because there was one model and its shape was a constant.  There are
// two families now, and the honest rule is the one the MoE shape already followed: THE FILE IS THE AUTHORITY.
// The factory supplies a family's shape so that a key the file omits still has a value, and every key the
// file carries is then read.
//
// The checks below are not decoration either.  `attention.head_count_kv` is an ARRAY on glm5-next whose
// entries are the layer type (0 = KDA, 1 = MLA) - reading it as a scalar silently yields 0, and a geometry
// built from that would run every layer as the wrong mixer.  So the array is checked for length and for the
// positions it implies, and a mismatch is refused rather than run.
#include "strata/core/layout.hpp"

#include "strata/artifact/gguf_reader.hpp"

#include <string>

namespace strata::core {

std::string apply_model_geometry(const GgufFile& f, ModelGeometry& g, int64_t& K) {
    const MetaValue* a = f.get("general.architecture");
    if (a == nullptr) return "missing general.architecture";
    Arch arch = Arch::Unknown;
    if (!arch_from_string(a->s, arch))
        return "architecture is '" + a->s + "', this engine runs " + arch_list();
    g = model_geometry_for(arch);

    const std::string p = std::string(arch_meta_prefix(arch)) + ".";

    // An integer key, when present.  A key the file writes as an array is NOT read here: the caller has to
    // say what the array means (see head_count_kv below), and reading `.u` off an array silently gives 0.
    auto i64 = [&](const char* name, int64_t& out) {
        const MetaValue* v = f.get(p + name);
        if (v != nullptr && v->is_num()) out = (int64_t) v->num();
    };
    auto f64 = [&](const char* name, double& out) {
        const MetaValue* v = f.get(p + name);
        if (v != nullptr && v->is_num()) out = v->num();
    };

    i64("embedding_length", g.n_embd);
    i64("vocab_size", g.n_vocab);
    i64("feed_forward_length", g.n_ff_dense);
    i64("expert_count", g.n_expert);
    i64("expert_used_count", K);
    i64("expert_feed_forward_length", g.n_ff);
    i64("expert_shared_count", g.n_expert_shared);
    i64("leading_dense_block_count", g.n_layer_dense_lead);
    i64("attention.head_count", g.n_head);
    f64("attention.layer_norm_rms_epsilon", g.rms_eps);
    i64("q_lora_rank", g.q_lora_rank);
    i64("attention.q_lora_rank", g.q_lora_rank);
    i64("attention.kv_lora_rank", g.kv_lora_rank);
    i64("attention.key_length_mla", g.mla_head_dim);
    i64("attention.indexer.head_count", g.idx_q_heads);
    i64("attention.indexer.key_length", g.idx_key_dim);
    i64("attention.indexer.top_k", g.idx_top_k);
    i64("attention.indexer.kpool", g.idx_kpool);
    i64("hyper_connection.count", g.hc);
    i64("kda.head_dim", g.kda_head_dim);
    f64("kda.gate_lower_bound", g.kda_gate_floor);
    // Both families spell their linear layer's conv kernel `ssm.conv_kernel`, and each has its own field for
    // it, so the key is read into the one that belongs to the family the file declares.
    if (arch == Arch::Glm5Next) i64("ssm.conv_kernel", g.kda_conv_kernel);
    else i64("ssm.conv_kernel", g.ssm_d_conv);
    f64("expert_weights_scale", g.expert_weights_scale);

    // The SwiGLU clamp is a PER-LAYER array (one float per block).  The engine applies one value, so the file
    // is only usable if they agree - and a fine-tune that clamps layer 20 differently would otherwise be run
    // with layer 0's limit, which is a wrong answer that looks like a rounding difference.
    //
    // TWO KEYS, because glm5-next clamps two different FFNs: `swiglu_clamp_exp` for the routed experts and
    // `swiglu_clamp_shexp` for the dense-lead layers and the shared expert.  They happen to be equal (10.0) on
    // every published artifact, which is exactly why reading only one of them would go unnoticed.
    const auto read_clamp = [&](const char* key, double& out) -> std::string {
        const MetaValue* v = f.get(p + key);
        if (v == nullptr || v->type != MetaType::ARRAY || v->count == 0) return std::string();
        const double first = v->items[0].num();
        for (uint64_t i = 1; i < v->count; ++i) {
            if (v->items[i].num() != first)
                return p + key + " is not uniform (layer 0 is " + std::to_string(first) + ", layer " +
                       std::to_string(i) + " is " + std::to_string(v->items[i].num()) +
                       "); this engine applies one clamp to every layer";
        }
        out = first;
        return std::string();
    };
    if (std::string e = read_clamp("swiglu_clamp_exp", g.swiglu_clamp); !e.empty()) return e;
    if (std::string e = read_clamp("swiglu_clamp_shexp", g.swiglu_clamp_shexp); !e.empty()) return e;

    // ---- the block count, and what counts as a trunk layer ----
    // `block_count` counts the MTP/NextN block; the trunk the engine runs does not include it.  The file says
    // how many there are, so this is read rather than assumed, and a file whose block count cannot be split
    // into a trunk plus its MTP blocks is refused instead of run one layer short.
    int64_t blocks = g.n_layers + g.n_nextn;
    i64("block_count", blocks);
    i64("nextn_predict_layers", g.n_nextn);
    if (blocks <= g.n_nextn)
        return p + "block_count is " + std::to_string(blocks) + " with " + std::to_string(g.n_nextn) +
               " MTP block(s): no trunk layers left";
    g.n_layers = blocks - g.n_nextn;

    // ---- the layer split ----
    // glm5-next declares the mixer per layer as an ARRAY under a key whose name (`head_count_kv`) says "KV
    // heads".  On the linear layers it is 0 and on the full-attention layers 1, so the array IS the layer map
    // and this is the one place it is read.
    if (const MetaValue* v = f.get(p + "attention.head_count_kv"); v != nullptr) {
        if (v->type != MetaType::ARRAY) {
            g.n_head_kv = (int64_t) v->u;   // a scalar here means "the same on every layer"
        } else {
            if ((int64_t) v->count != blocks)
                return p + "attention.head_count_kv has " + std::to_string(v->count) + " entries for " +
                       std::to_string(blocks) + " blocks";
            if (v->items[0].u != 0)
                return p + "attention.head_count_kv expects layer 0 to be a linear layer (kv 0)";
            for (int64_t il = 0; il < g.n_layers; ++il) {
                const bool full = v->items[il].u != 0;
                if (full != is_qsa_layer(g, il)) {
                    char buf[192];
                    std::snprintf(buf, sizeof buf,
                                  "%sattention.head_count_kv marks layer %lld as %s, but the layer rule "
                                  "(`layer %% %lld == %lld`) says otherwise",
                                  p.c_str(), (long long) il, full ? "full attention" : "linear",
                                  (long long) g.qsa_interval, (long long) (g.qsa_interval - 1));
                    return buf;
                }
            }
            // One latent per token, not a KV head count: the cache is `kv_lora_rank` wide and K == V.
            g.n_head_kv = 1;
        }
    }

    // Derived, not read: the cache on the MLA layers is the kv_lora latent.
    if (g.kv_lora_rank > 0) g.head_dim = g.kv_lora_rank;

    return {};
}

}  // namespace strata::core
