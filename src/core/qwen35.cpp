// src/core/qwen35.cpp - the Qwen35MoE architecture guard.  See include/strata/core/qwen35.hpp for why this
// is separate from layout.cpp.
//
// The reference is llama.cpp's src/models/qwen35moe.cpp at the revision this repository pins.  Every
// dimension below is read from that file's `load_arch_tensors` (the `create_tensor(...)` shapes) and checked
// against the artifact's own metadata; the validation is therefore "the shape the reference builds", not
// "the shape this repository happens to have seen".
#include "strata/core/qwen35.hpp"

#include <cctype>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

namespace strata::core {
namespace {

std::string shape_str(const strata::TensorInfo& t) {
    std::string s = "[";
    for (size_t i = 0; i < t.shape.size(); ++i) {
        if (i) s += ", ";
        s += std::to_string(t.shape[i]);
    }
    return s + "]";
}

std::string want_str(const std::vector<int64_t>& dims) {
    std::string s = "[";
    bool first = true;
    for (int64_t d : dims) {
        if (!first) s += ", ";
        first = false;
        s += std::to_string(d);
    }
    return s + "]";
}

/// A required scalar integer metadata value.
bool meta_i64(const GgufFile& g, const char* key, int64_t& out, std::string& err) {
    const MetaValue* v = g.get(key);
    if (!v) { err = std::string("qwen35moe: missing metadata ") + key; return false; }
    if (!v->is_num()) { err = std::string("qwen35moe: ") + key + " is not a number"; return false; }
    out = (int64_t) v->num();
    return true;
}

bool meta_f64(const GgufFile& g, const char* key, double& out, std::string& err) {
    const MetaValue* v = g.get(key);
    if (!v) { err = std::string("qwen35moe: missing metadata ") + key; return false; }
    if (!v->is_num()) { err = std::string("qwen35moe: ") + key + " is not a number"; return false; }
    out = v->num();
    return true;
}

/// `expert_feed_forward_length` is written per layer by some quantizers (llama.cpp's `get_key_or_arr`).  The
/// kernels need ONE number, so an array is accepted only when every entry agrees.
bool meta_i64_uniform(const GgufFile& g, const char* key, int64_t& out, std::string& err) {
    const MetaValue* v = g.get(key);
    if (!v) { err = std::string("qwen35moe: missing metadata ") + key; return false; }
    if (v->type != MetaType::ARRAY) return meta_i64(g, key, out, err);
    if (v->count == 0) { err = std::string("qwen35moe: ") + key + " is an empty array"; return false; }
    if (v->count > v->items.size())
        { err = std::string("qwen35moe: ") + key + " is longer than this reader samples"; return false; }
    out = (int64_t) v->items[0].num();
    for (const MetaValue& e : v->items)
        if ((int64_t) e.num() != out)
            { err = std::string("qwen35moe: ") + key + " varies by layer; the kernels need one value"; return false; }
    return true;
}

enum class TypePolicy { F32, Weight };

bool check_type(const strata::TensorInfo& t, TypePolicy policy, std::string& err) {
    if (policy == TypePolicy::F32 && t.type != 0) {
        err = "qwen35moe: tensor " + t.name + " is " + t.type_name() + ", the kernel reads it as F32";
        return false;
    }
    if (policy == TypePolicy::Weight) {
        int be = 0, bb = 0;
        if (!strata::block_geometry(t.type, be, bb)) {
            err = "qwen35moe: tensor " + t.name + " is " + t.type_name() + ", for which this engine has no kernel";
            return false;
        }
        if (t.shape.empty() || t.shape[0] % (uint64_t) be != 0) {
            err = "qwen35moe: tensor " + t.name + " has " + shape_str(t) + " " + t.type_name() +
                  ", whose contiguous dimension is not a whole number of " + std::to_string(be) + "-value blocks";
            return false;
        }
    }
    return true;
}

/// One required trunk tensor.  `ne1 < 0` means 1-D; `ne0..ne1` with `ne2 < 0` means 2-D; all three means 3-D.
struct Want {
    const char* suffix;
    int64_t ne0, ne1, ne2;
    TypePolicy type;
    bool recurrent_only;
    bool attention_only;
};

bool check_one(const GgufModel& m, const Qwen35Geometry& g, int64_t layer, std::string& err) {
    const bool recr = g.is_recurrent(layer);
    const std::vector<Want> wants = {
        // Every layer: norms, router, routed experts, shared expert.
        {"attn_norm.weight", g.n_embd, -1, -1, TypePolicy::F32, false, false},
        {"post_attention_norm.weight", g.n_embd, -1, -1, TypePolicy::F32, false, false},
        {"ffn_gate_inp.weight", g.n_embd, g.n_expert, -1, TypePolicy::F32, false, false},
        {"ffn_gate_inp_shexp.weight", g.n_embd, -1, -1, TypePolicy::F32, false, false},
        {"ffn_gate_shexp.weight", g.n_embd, g.n_ff_shexp, -1, TypePolicy::Weight, false, false},
        {"ffn_up_shexp.weight", g.n_embd, g.n_ff_shexp, -1, TypePolicy::Weight, false, false},
        {"ffn_down_shexp.weight", g.n_ff_shexp, g.n_embd, -1, TypePolicy::Weight, false, false},
        {"ffn_gate_exps.weight", g.n_embd, g.n_ff_exp, g.n_expert, TypePolicy::Weight, false, false},
        {"ffn_up_exps.weight", g.n_embd, g.n_ff_exp, g.n_expert, TypePolicy::Weight, false, false},
        {"ffn_down_exps.weight", g.n_ff_exp, g.n_embd, g.n_expert, TypePolicy::Weight, false, false},

        // Recurrent (gated delta net) layers.
        {"attn_qkv.weight", g.n_embd, g.qkv_dim(), -1, TypePolicy::Weight, true, false},
        {"attn_gate.weight", g.n_embd, g.value_dim(), -1, TypePolicy::Weight, true, false},
        {"ssm_conv1d.weight", g.ssm_conv_kernel, g.conv_channels(), -1, TypePolicy::F32, true, false},
        {"ssm_dt.bias", g.ssm_dt_rank, -1, -1, TypePolicy::F32, true, false},
        {"ssm_a", g.ssm_dt_rank, -1, -1, TypePolicy::F32, true, false},
        {"ssm_beta.weight", g.n_embd, g.ssm_dt_rank, -1, TypePolicy::Weight, true, false},
        {"ssm_alpha.weight", g.n_embd, g.ssm_dt_rank, -1, TypePolicy::Weight, true, false},
        {"ssm_norm.weight", g.ssm_state, -1, -1, TypePolicy::F32, true, false},
        {"ssm_out.weight", g.value_dim(), g.n_embd, -1, TypePolicy::Weight, true, false},

        // Full-attention layers: Q carries the query half AND the output gate half.
        {"attn_q.weight", g.n_embd, 2 * g.n_head * g.head_dim, -1, TypePolicy::Weight, false, true},
        {"attn_k.weight", g.n_embd, g.n_head_kv * g.head_dim, -1, TypePolicy::Weight, false, true},
        {"attn_v.weight", g.n_embd, g.n_head_kv * g.head_dim, -1, TypePolicy::Weight, false, true},
        {"attn_output.weight", g.n_head * g.head_dim, g.n_embd, -1, TypePolicy::Weight, false, true},
        {"attn_q_norm.weight", g.head_dim, -1, -1, TypePolicy::F32, false, true},
        {"attn_k_norm.weight", g.head_dim, -1, -1, TypePolicy::F32, false, true},
    };
    for (const Want& w : wants) {
        if (w.recurrent_only && !recr) continue;
        if (w.attention_only && recr) continue;
        const std::string name = "blk." + std::to_string(layer) + "." + w.suffix;
        const strata::TensorInfo* t = m.find(name);
        if (!t) { err = "qwen35moe: layer " + std::to_string(layer) + ": missing " + name; return false; }
        std::vector<int64_t> want = {w.ne0};
        if (w.ne1 >= 0) want.push_back(w.ne1);
        if (w.ne2 >= 0) want.push_back(w.ne2);
        bool shape_ok = t->shape.size() == want.size();
        for (size_t i = 0; shape_ok && i < want.size(); ++i) shape_ok = t->shape[i] == (uint64_t) want[i];
        if (!shape_ok) {
            err = "qwen35moe: layer " + std::to_string(layer) + ": " + name + " is " + shape_str(*t) +
                  " " + t->type_name() + ", the kernel requires " + want_str(want);
            return false;
        }
        if (!check_type(*t, w.type, err)) return false;
    }

    // A Qwen35 layer must not carry the OTHER architecture's tensors.  A recurrent block has no dense
    // attention projections and no QSA indexer; a full-attention block has no GDN state tensors.
    auto forbid = [&](const char* suffix) -> bool {
        const std::string name = "blk." + std::to_string(layer) + "." + suffix;
        if (m.find(name)) {
            err = "qwen35moe: layer " + std::to_string(layer) + ": " + name +
                  " belongs to a " + (recr ? std::string("full-attention") : std::string("recurrent")) +
                  " layer, but this is a " + (recr ? std::string("recurrent") : std::string("full-attention")) + " layer";
            return false;
        }
        return true;
    };
    if (recr) {
        if (!forbid("attn_q.weight")) return false;
        if (!forbid("attn_k.weight")) return false;
        if (!forbid("attn_v.weight")) return false;
        if (!forbid("attn_output.weight")) return false;
        if (!forbid("attn_q_norm.weight")) return false;
        if (!forbid("attn_k_norm.weight")) return false;
    } else {
        if (!forbid("attn_qkv.weight")) return false;
        if (!forbid("attn_gate.weight")) return false;
        if (!forbid("ssm_conv1d.weight")) return false;
        if (!forbid("ssm_dt.bias")) return false;
        if (!forbid("ssm_a")) return false;
        if (!forbid("ssm_beta.weight")) return false;
        if (!forbid("ssm_alpha.weight")) return false;
        if (!forbid("ssm_norm.weight")) return false;
        if (!forbid("ssm_out.weight")) return false;
    }
    // Hyper-connection, QSA-indexer and PLE tensors are Qwen4Exp-only; check_qwen35_tensors scans for
    // them separately so the message names the OTHER architecture rather than a crossing layer family.
    return true;
}

}  // namespace

ModelKind detect_model_kind(const GgufFile& meta_shard) {
    const MetaValue* a = meta_shard.get("general.architecture");
    if (!a || a->type != MetaType::STRING) return ModelKind::Unknown;
    if (a->s == "qwen4exp") return ModelKind::Qwen4Exp;
    if (a->s == "qwen35moe") return ModelKind::Qwen35Moe;
    return ModelKind::Unknown;
}

bool qwen35_geometry(const GgufFile& g, Qwen35Geometry& out, std::string& err) {
    const MetaValue* arch = g.get("general.architecture");
    if (!arch || arch->type != MetaType::STRING)
        { err = "qwen35moe: missing general.architecture"; return false; }
    if (arch->s != "qwen35moe")
        { err = "qwen35moe: architecture is '" + arch->s + "', this guard requires 'qwen35moe'"; return false; }

    Qwen35Geometry q;
    if (!meta_i64(g, "qwen35moe.block_count", q.n_layers, err)) return false;
    if (!meta_i64(g, "qwen35moe.embedding_length", q.n_embd, err)) return false;
    if (!meta_i64(g, "qwen35moe.expert_count", q.n_expert, err)) return false;
    if (!meta_i64(g, "qwen35moe.expert_used_count", q.n_expert_used, err)) return false;
    if (!meta_i64_uniform(g, "qwen35moe.expert_feed_forward_length", q.n_ff_exp, err)) return false;
    if (!meta_i64(g, "qwen35moe.expert_shared_feed_forward_length", q.n_ff_shexp, err)) return false;
    if (!meta_i64(g, "qwen35moe.attention.head_count", q.n_head, err)) return false;
    if (!meta_i64(g, "qwen35moe.attention.head_count_kv", q.n_head_kv, err)) return false;
    if (!meta_i64(g, "qwen35moe.attention.key_length", q.head_dim, err)) return false;
    int64_t value_length = 0;
    if (!meta_i64(g, "qwen35moe.attention.value_length", value_length, err)) return false;
    {
        double eps = 0.0;
        if (!meta_f64(g, "qwen35moe.attention.layer_norm_rms_epsilon", eps, err)) return false;
        q.rms_eps = (float) eps;
    }
    if (!meta_i64(g, "qwen35moe.rope.dimension_count", q.rope_dim, err)) return false;
    if (!meta_f64(g, "qwen35moe.rope.freq_base", q.rope_freq_base, err)) return false;
    if (!meta_i64(g, "qwen35moe.context_length", q.context_length, err)) return false;
    if (!meta_i64(g, "qwen35moe.ssm.conv_kernel", q.ssm_conv_kernel, err)) return false;
    if (!meta_i64(g, "qwen35moe.ssm.inner_size", q.ssm_inner, err)) return false;
    if (!meta_i64(g, "qwen35moe.ssm.state_size", q.ssm_state, err)) return false;
    if (!meta_i64(g, "qwen35moe.ssm.time_step_rank", q.ssm_dt_rank, err)) return false;
    if (!meta_i64(g, "qwen35moe.ssm.group_count", q.ssm_groups, err)) return false;
    {
        const MetaValue* v = g.get("qwen35moe.rope.dimension_sections");
        if (!v || v->type != MetaType::ARRAY || v->count != 4)
            { err = "qwen35moe: rope.dimension_sections must be a 4-element array"; return false; }
        for (int i = 0; i < 4; ++i) q.rope_sections[i] = (int64_t) v->items[(size_t) i].num();
    }
    // The interval falls back to llama.cpp's default of 4 only when the key is absent.
    {
        int64_t interval = 4;
        if (const MetaValue* v = g.get("qwen35moe.full_attention_interval")) {
            if (!v->is_num()) { err = "qwen35moe: full_attention_interval is not a number"; return false; }
            interval = (int64_t) v->num();
        }
        q.full_attention_interval = interval;
    }

    // ---- cross-field consistency, checked here so a malformed file fails on ONE readable line
    if (q.n_layers <= 0 || q.n_embd <= 0 || q.head_dim <= 0 || q.n_head <= 0 || q.n_head_kv <= 0)
        { err = "qwen35moe: non-positive layer/width/head geometry"; return false; }
    if (value_length != q.head_dim)
        { err = "qwen35moe: attention.value_length (" + std::to_string(value_length) +
                ") != attention.key_length (" + std::to_string(q.head_dim) + ")"; return false; }
    if (q.rope_dim <= 0 || q.rope_dim > q.head_dim || q.rope_dim % 2 != 0)
        { err = "qwen35moe: rope.dimension_count (" + std::to_string(q.rope_dim) +
                ") must be positive, even and no larger than head_length"; return false; }
    if (q.full_attention_interval <= 0 || q.n_layers % q.full_attention_interval != 0)
        { err = "qwen35moe: full_attention_interval must divide block_count"; return false; }
    if (q.n_head % q.n_head_kv != 0)
        { err = "qwen35moe: attention.head_count must be a multiple of head_count_kv"; return false; }
    if (q.ssm_groups <= 0 || q.ssm_dt_rank <= 0 || q.ssm_dt_rank % q.ssm_groups != 0)
        { err = "qwen35moe: ssm.time_step_rank must be a positive multiple of ssm.group_count"; return false; }
    if (q.ssm_state <= 0 || q.ssm_inner != q.ssm_dt_rank * (q.ssm_inner / q.ssm_dt_rank))
        { err = "qwen35moe: ssm.inner_size must be a multiple of ssm.time_step_rank"; return false; }
    if (q.ssm_conv_kernel <= 0)
        { err = "qwen35moe: ssm.conv_kernel must be positive"; return false; }
    if (q.n_expert <= 0 || q.n_expert_used <= 0 || q.n_expert_used > q.n_expert)
        { err = "qwen35moe: expert_count / expert_used_count are not a valid routing geometry"; return false; }
    if (q.rms_eps <= 0.f)
        { err = "qwen35moe: layer_norm_rms_epsilon must be positive"; return false; }

    // The embedding's output dimension is the vocabulary the head must match.
    if (const strata::TensorInfo* t = g.find("token_embd.weight")) {
        if (t->shape.size() == 2 && t->shape[0] == (uint64_t) q.n_embd) q.n_vocab = (int64_t) t->shape[1];
    }
    if (q.n_vocab <= 0)
        { err = "qwen35moe: token_embd.weight is missing or its shape is not [embedding_length, vocab]"; return false; }
    q.output_tied = g.find("output.weight") == nullptr;

    out = q;
    return true;
}

bool check_qwen35_tensors(const GgufModel& m, const Qwen35Geometry& g, std::string& err) {
    if (detect_model_kind(m.meta()) != ModelKind::Qwen35Moe)
        { err = "qwen35moe: metadata shard is not a qwen35moe model"; return false; }

    // ---- the global tensors
    {
        const strata::TensorInfo* emb = m.find("token_embd.weight");
        if (!emb) { err = "qwen35moe: missing token_embd.weight"; return false; }
        if (emb->shape.size() != 2 || emb->shape[0] != (uint64_t) g.n_embd || emb->shape[1] != (uint64_t) g.n_vocab) {
            err = "qwen35moe: token_embd.weight is " + shape_str(*emb) + " " + emb->type_name() +
                  ", the kernel requires " + want_str({g.n_embd, g.n_vocab});
            return false;
        }
        if (!check_type(*emb, TypePolicy::Weight, err)) return false;

        const strata::TensorInfo* on = m.find("output_norm.weight");
        if (!on) { err = "qwen35moe: missing output_norm.weight"; return false; }
        if (on->shape.size() != 1 || on->shape[0] != (uint64_t) g.n_embd) {
            err = "qwen35moe: output_norm.weight is " + shape_str(*on) + ", the kernel requires " + want_str({g.n_embd});
            return false;
        }
        if (!check_type(*on, TypePolicy::F32, err)) return false;

        if (const strata::TensorInfo* out = m.find("output.weight")) {
            if (out->shape.size() != 2 || out->shape[0] != (uint64_t) g.n_embd ||
                out->shape[1] != (uint64_t) g.n_vocab) {
                err = "qwen35moe: output.weight is " + shape_str(*out) + " " + out->type_name() +
                      ", the kernel requires " + want_str({g.n_embd, g.n_vocab});
                return false;
            }
            if (!check_type(*out, TypePolicy::Weight, err)) return false;
        }
    }

    // ---- every trunk layer
    int64_t n_recr = 0, n_attn = 0;
    for (int64_t l = 0; l < g.n_layers; ++l) {
        if (!check_one(m, g, l, err)) return false;
        if (g.is_recurrent(l)) ++n_recr; else ++n_attn;
    }
    if (n_recr != g.n_recurrent_layers() || n_attn != g.n_attention_layers()) {
        err = "qwen35moe: the layer split is " + std::to_string(n_recr) + " recurrent / " + std::to_string(n_attn) +
              " attention, the geometry says " + std::to_string(g.n_recurrent_layers()) + " / " +
              std::to_string(g.n_attention_layers());
        return false;
    }

    // ---- any Qwen4Exp-only tensor under a trunk block is a mis-labeled artifact, not a Qwen35 one
    for (size_t s = 0; s < m.size(); ++s) {
        for (const strata::TensorInfo& t : m.shard(s).tensors()) {
            if (t.name.rfind("blk.", 0) != 0) continue;
            const size_t dot = t.name.find('.', 4);
            if (dot == std::string::npos) continue;
            bool numeric = dot > 4;
            for (size_t i = 4; i < dot; ++i) numeric = numeric && std::isdigit((unsigned char) t.name[i]);
            if (!numeric) continue;
            const int64_t layer = std::atoll(t.name.substr(4, dot - 4).c_str());
            if (layer < 0 || layer >= g.n_layers) continue;   // MTP layers live past the trunk
            const std::string suffix = t.name.substr(dot + 1);
            if (suffix.rfind("hc_", 0) == 0 || suffix.rfind("indexer.", 0) == 0 || suffix.rfind("ple_", 0) == 0) {
                err = "qwen35moe: " + t.name + " is a Qwen4Exp tensor in a qwen35moe block";
                return false;
            }
        }
    }
    return true;
}

bool check_qwen35_all(const GgufModel& model, Qwen35Geometry& g, std::string& err) {
    if (detect_model_kind(model.meta()) != ModelKind::Qwen35Moe) {
        err = std::string("qwen35moe: general.architecture is not 'qwen35moe'");
        return false;
    }
    if (!qwen35_geometry(model.meta(), g, err)) return false;
    if (!check_qwen35_tensors(model, g, err)) return false;
    return true;
}

}  // namespace strata::core
