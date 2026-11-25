#pragma once

#include "strata/core/family_layout.hpp"

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <string>
#include <vector>

namespace strata::qwen35 {

struct MemtConfig {
    int64_t heads = 0;
    int64_t head_dim = 0;
    int64_t gate_rank = 0;
    double softcap = 0.0;
    double rms_eps = 0.0;
    std::string snapshot_identity;

    constexpr std::size_t state_elements() const { return (std::size_t) heads * head_dim * head_dim; }
};

inline bool read_memt_config(const strata::GgufFile& meta, const core::ModelGeometry& g,
                             std::optional<MemtConfig>& memt, std::string& err) {
    err.clear();
    memt.reset();
    const strata::MetaValue* version = meta.get("memt.version");
    if (!version) return true;
    auto bad = [&](const char* message) {
        err = message;
        return false;
    };
    if (!strata::meta_is_int_type(version->type) || (int64_t) version->u != 1)
        return bad("memt.version must be the integer 1");
    if (g.family != strata::ModelFamily::Qwen35Moe)
        return bad("memt.* is only supported on qwen35moe");
    MemtConfig config;
    auto memt_int = [&](const char* key, int64_t& dst) {
        const strata::MetaValue* v = meta.get(key);
        if (!v || !strata::meta_is_int_type(v->type) || (int64_t) v->u <= 0) {
            err = std::string("memt: ") + key + " is missing, not an integer, or not positive";
            return false;
        }
        dst = (int64_t) v->u;
        return true;
    };
    if (!memt_int("memt.head_count", config.heads)) return false;
    if (!memt_int("memt.head_dim", config.head_dim)) return false;
    if (!memt_int("memt.gate_rank", config.gate_rank)) return false;
    int64_t chunk_tokens = 0;
    if (!memt_int("memt.chunk_tokens", chunk_tokens)) return false;
    if (config.heads != g.n_head)
        return bad("memt.head_count != attention.head_count");
    if (config.head_dim != g.head_dim)
        return bad("memt.head_dim != attention.key_length");
    if (config.gate_rank != g.head_dim)
        return bad("memt.gate_rank != attention.key_length");
    auto memt_float = [&](const char* key, double& dst) {
        const strata::MetaValue* v = meta.get(key);
        if (!v || !v->is_num() || !std::isfinite(v->num()) || v->num() <= 0.0) {
            err = std::string("memt: ") + key + " is missing, not a number, or not positive";
            return false;
        }
        dst = v->num();
        return true;
    };
    if (!memt_float("memt.output_gate_softcap_limit", config.softcap)) return false;
    if (!memt_float("memt.rms_norm_epsilon", config.rms_eps)) return false;
    if (const strata::MetaValue* fi = meta.get("memt.full_attention_interval")) {
        if (!strata::meta_is_int_type(fi->type) || (int64_t) fi->u != g.qsa_interval)
            return bad("memt.full_attention_interval != qwen35moe.full_attention_interval");
    }
    if (const strata::MetaValue* id = meta.get("memt.identity")) {
        const std::string& s = id->s;
        const bool hex = id->type == strata::MetaType::STRING && s.size() == 64 &&
                         s.find_first_not_of("0123456789abcdefABCDEF") == std::string::npos;
        if (!hex)
            return bad("memt.identity must be 64 hex characters (a memt-memory-v1 identity)");
        config.snapshot_identity = s;
    }
    if (const strata::MetaValue* ml = meta.get("memt.memory_layers")) {
        const int64_t want = g.n_qsa_layers();
        if (ml->type != strata::MetaType::ARRAY || !strata::meta_is_int_type(ml->elem) ||
            ml->count != (uint64_t) want)
            return bad("memt.memory_layers must list the full-attention layers");
        for (int64_t i = 0; i < want; ++i) {
            const int64_t expect = (i + 1) * g.qsa_interval - 1;
            if ((int64_t) ml->items[(size_t) i].u != expect)
                return bad("memt.memory_layers does not match full_attention_interval");
        }
    }
    memt = config;
    return true;
}

inline std::vector<core::TensorWant> memt_tensors(const core::ModelGeometry& g, const MemtConfig& config) {
    const int64_t width = config.heads * config.head_dim;
    const int64_t rank = config.gate_rank;
    std::vector<core::TensorWant> tensors;
    for (int64_t layer = 0; layer < g.n_layers; ++layer) {
        if (!core::is_qsa_layer(g, layer)) continue;
        const std::string prefix = "blk." + std::to_string(layer) + ".memt.";
        tensors.push_back({prefix + "in_proj.weight", {g.n_embd, 5 * width + 2 * rank}});
        tensors.push_back({prefix + "f_down.weight", {rank, width}});
        tensors.push_back({prefix + "g_down.weight", {rank, width}});
        tensors.push_back({prefix + "decay_scale_log", {config.heads}});
        tensors.push_back({prefix + "decay_bias", {width}});
        tensors.push_back({prefix + "o_norm.weight", {config.head_dim}});
    }
    return tensors;
}

}  // namespace strata::qwen35
