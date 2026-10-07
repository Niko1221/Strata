// src/core/layout.cpp - the shape checks.  See the header for why this exists.
#include "strata/core/layout.hpp"

#include "strata/artifact/gguf_reader.hpp"

#include <cstdio>

namespace strata::core {
namespace {

/// A required 2-D shape.  `ne0` is the CONTIGUOUS axis, matching the manifest and `s_gemv`'s convention
/// (`y[o] = sum_i x[i]*W[i][o]`, row o contiguous of length ne0).
struct Want2 {
    const char* suffix;
    int64_t ne0;
    int64_t ne1;
    bool qsa_only;
    bool gdn_only;
};

/// A required element count and ENGINE FORM for a 1-D tensor.
///
/// `kind` is not decoration.  The engine form of a 1-D tensor is whatever `WeightKind` the loader applied -
/// `F32` copies 4 B/elem, `Bf16InF32` re-rounds to 2 - and a consumer that casts either one to `const float*`
/// reads 2x its length.  **THAT IS NOT HYPOTHETICAL: `ffn_gate_inp_shexp.weight` is BF16, so the arena holds
/// 5120 B for 2560 elements, and `shared_expert` read it as f32.  Every one of the 2560 MoE outputs came out
/// non-finite** - a wrong answer loud enough to notice, which is the lucky version.  Reading 5120 B past the
/// end of a tensor inside a 4.5 GiB arena does not fault, so the same mistake on a tensor followed by
/// plausible bytes would have produced plausible logits.
struct Want1 {
    const char* suffix;
    int64_t elements;
    WeightKind kind;
    bool qsa_only;
    bool gdn_only;
};

bool fail(std::string& err, const LayerView& v, const char* suffix, const char* what, int64_t got,
          int64_t want) {
    char buf[512];
    std::snprintf(buf, sizeof buf, "layer %lld: %s %s is %lld, the kernels require %lld",
                  (long long) v.layer(), v.name(suffix).c_str(), what, (long long) got, (long long) want);
    err = buf;
    return false;
}

bool check_one(const WeightTable& t, const ModelGeometry& g, int64_t layer, std::string& err) {
    const LayerView v(t, layer);
    const bool qsa = is_qsa_layer(g, layer);

    // ---- the 2-D tensor set, per layer family
    const Want2 want2[] = {
        // gated residual, EVERY layer - `gr_read` takes w_down (hc_lr, hc_dim) and w_up (hc_lr, hc_dim)
        // after its own transpose, so the pack's orientation is [hc_dim, hc_lr] and [hc_lr, hc_dim].
        {"hc_attn_down.weight", g.hc_dim(), g.hc_lr, false, false},
        {"hc_attn_up.weight", g.hc_lr, g.hc_dim(), false, false},
        {"hc_attn_inject.weight", g.hc_dim(), g.hc, false, false},
        {"hc_ffn_down.weight", g.hc_dim(), g.hc_lr, false, false},
        {"hc_ffn_up.weight", g.hc_lr, g.hc_dim(), false, false},
        {"hc_ffn_inject.weight", g.hc_dim(), g.hc, false, false},
        // MoE, EVERY layer
        {"ffn_gate_inp.weight", g.n_embd, g.n_expert, false, false},
        {"ffn_gate_shexp.weight", g.n_embd, g.n_ff, false, false},
        {"ffn_up_shexp.weight", g.n_embd, g.n_ff, false, false},
        {"ffn_down_shexp.weight", g.n_ff, g.n_embd, false, false},
        // GDN only
        {"attn_qkv.weight", g.n_embd, g.ssm_conv_channels, false, true},
        {"attn_gate.weight", g.n_embd, g.ssm_value_dim, false, true},
        {"ssm_out.weight", g.ssm_value_dim, g.n_embd, false, true},
        {"ssm_conv1d.weight", g.ssm_d_conv, g.ssm_conv_channels, false, true},
        {"ssm_alpha.weight", g.n_embd, g.ssm_v_heads, false, true},
        {"ssm_beta.weight", g.n_embd, g.ssm_v_heads, false, true},
        // QSA only
        {"attn_q.weight", g.n_embd, 2 * g.n_head * g.head_dim, true, false},
        {"attn_k.weight", g.n_embd, g.n_head_kv * g.head_dim, true, false},
        {"attn_v.weight", g.n_embd, g.n_head_kv * g.head_dim, true, false},
        {"attn_output.weight", g.n_head * g.head_dim, g.n_embd, true, false},
        // THE INDEXER.  `q_proj` is the QUERY count and `k_proj` is the KEY width, and they are different
        // numbers - conflating them is what made the planner's indexer term 4x too big in round 194.
        {"indexer.q_proj.weight", g.n_embd, g.idx_q_heads * g.idx_key_dim, true, false},
        {"indexer.k_proj.weight", g.n_embd, g.idx_key_dim, true, false},
    };
    for (const Want2& w : want2) {
        if (w.qsa_only && !qsa) continue;
        if (w.gdn_only && qsa) continue;
        const WeightRef* r = v.get(w.suffix);
        if (!r) {
            err = "layer " + std::to_string(layer) + ": missing " + v.name(w.suffix);
            return false;
        }
        if (r->ne0 != w.ne0) return fail(err, v, w.suffix, "ne0", r->ne0, w.ne0);
        if (r->ne1 != w.ne1) return fail(err, v, w.suffix, "ne1", r->ne1, w.ne1);
    }

    // ---- the 1-D set.  EVERY ONE OF THESE IS `F32` EXCEPT the shared expert's scalar gate, which is BF16 -
    // and that single exception is the one a reader would not guess, so it is written down rather than
    // inferred from the tensor count.
    const Want1 want1[] = {
        {"hc_attn_norm.weight", g.hc_dim(), WeightKind::F32, false, false},
        {"hc_ffn_norm.weight", g.hc_dim(), WeightKind::F32, false, false},
        {"ffn_gate_inp_shexp.weight", g.n_embd, WeightKind::Bf16InF32, false, false},
        {"ssm_a", g.ssm_v_heads, WeightKind::F32, false, true},
        {"ssm_dt.bias", g.ssm_v_heads, WeightKind::F32, false, true},
        {"ssm_norm.weight", g.ssm_state_size, WeightKind::F32, false, true},
        {"attn_q_norm.weight", g.head_dim, WeightKind::F32, true, false},
        {"attn_k_norm.weight", g.head_dim, WeightKind::F32, true, false},
        {"indexer.q_norm.weight", g.idx_key_dim, WeightKind::F32, true, false},
        {"indexer.k_norm.weight", g.idx_key_dim, WeightKind::F32, true, false},
    };
    for (const Want1& w : want1) {
        if (w.qsa_only && !qsa) continue;
        if (w.gdn_only && qsa) continue;
        const WeightRef* r = v.get(w.suffix);
        if (!r) {
            err = "layer " + std::to_string(layer) + ": missing " + v.name(w.suffix);
            return false;
        }
        if (r->elements != w.elements) return fail(err, v, w.suffix, "elements", r->elements, w.elements);
        if (r->kind != w.kind) {
            // The byte count is what makes this a REAL check and not a label: a 4 B/elem tensor holds
            // `elements * 4`, a 2 B/elem one holds `elements * 2`, and a consumer reading it as the wrong one
            // walks off the end.
            const uint64_t want_bytes = (uint64_t) w.elements *
                                        (w.kind == WeightKind::Bf16InF32 ? 2u : 4u);
            char buf[512];
            std::snprintf(buf, sizeof buf,
                          "layer %lld: %s is engine form %d (%llu B), the kernels read it as form %d (%llu B)",
                          (long long) layer, v.name(w.suffix).c_str(), (int) r->kind,
                          (unsigned long long) r->bytes, (int) w.kind, (unsigned long long) want_bytes);
            err = buf;
            return false;
        }
    }
    return true;
}

}  // namespace

const char* arch_name(ModelArch arch) {
    return arch == ModelArch::kQwen35Moe ? "qwen35moe" : "qwen4exp";
}

bool geometry_from_gguf(const GgufFile& f, ModelGeometry& g, std::string& err) {
    const MetaValue* arch = f.get("general.architecture");
    if (arch == nullptr) {
        err = "the model file has no general.architecture";
        return false;
    }
    if (arch->s == "qwen4exp") {
        // Flash-Next: the defaults, with the MoE shape from the file as before (a pruned Coder ships 256 experts)
        ModelGeometry q;
        if (const MetaValue* v = f.get("qwen4exp.expert_count")) q.n_expert = (int64_t) v->u;
        if (const MetaValue* v = f.get("qwen4exp.expert_used_count")) q.n_expert_used = (int64_t) v->u;
        g = q;
        return true;
    }
    if (arch->s != "qwen35moe") {
        err = "the model's architecture is '" + arch->s + "'; this engine runs 'qwen4exp' (Qwen3.8-Flash-Next) and "
              "'qwen35moe' (Qwen3.6-35B-A3B)";
        return false;
    }
    std::string missing;
    auto key = [&](const char* name, int64_t& out, bool required = true) {
        const MetaValue* v = f.get(std::string("qwen35moe.") + name);
        if (v == nullptr || !v->is_num()) {
            if (required && missing.empty()) missing = std::string("qwen35moe.") + name;
            return;
        }
        out = (int64_t) v->u;
    };
    ModelGeometry q;
    q.arch = ModelArch::kQwen35Moe;
    int64_t block_count = 0, head_dim = 0, value_len = 0, n_rot = 0, n_ff_shexp = 0;
    key("block_count", block_count);
    key("nextn_predict_layers", q.n_mtp_layers, false);
    key("embedding_length", q.n_embd);
    key("full_attention_interval", q.qsa_interval);
    key("attention.head_count", q.n_head);
    key("attention.head_count_kv", q.n_head_kv);
    key("attention.key_length", head_dim);
    key("attention.value_length", value_len);
    key("rope.dimension_count", n_rot);
    key("ssm.state_size", q.ssm_state_size);
    key("ssm.group_count", q.ssm_k_heads);
    key("ssm.time_step_rank", q.ssm_v_heads);
    key("ssm.conv_kernel", q.ssm_d_conv);
    key("ssm.inner_size", q.ssm_value_dim);
    key("expert_count", q.n_expert);
    key("expert_used_count", q.n_expert_used);
    key("expert_feed_forward_length", q.n_ff);
    key("expert_shared_feed_forward_length", n_ff_shexp);
    if (!missing.empty()) {
        err = "the model file has no " + missing;
        return false;
    }
    q.n_layers = block_count - q.n_mtp_layers;
    q.ssm_conv_channels = 2 * q.ssm_state_size * q.ssm_k_heads + q.ssm_value_dim;
    q.hc = 1;                       // a plain residual stream
    q.hc_lr = 0;
    q.idx_q_heads = 0;              // no indexer: every cached cell is attended
    q.idx_key_dim = 0;
    q.has_ple = false;
    q.has_indexer = false;
    q.gdn_gate_silu = true;

    // What the kernels support.  The widths must be the ones the kernels are written for (head 256 with 64 rotated
    // dims, GDN state 128 and conv 4), and every count at most Flash-Next's: buffers sized by its compile-time
    // numbers then hold the smaller model unchanged.
    const ModelGeometry cap;
    char buf[256];
    auto bad = [&](const char* what, int64_t got, const char* want) {
        std::snprintf(buf, sizeof buf, "qwen35moe: %s is %lld; this engine supports %s", what, (long long) got, want);
        err = buf;
        return false;
    };
    if (head_dim != cap.head_dim || value_len != cap.head_dim) return bad("attention.key_length", head_dim, "256");
    if (n_rot != 64) return bad("rope.dimension_count", n_rot, "64");
    if (q.ssm_state_size != cap.ssm_state_size) return bad("ssm.state_size", q.ssm_state_size, "128");
    if (q.ssm_d_conv != cap.ssm_d_conv) return bad("ssm.conv_kernel", q.ssm_d_conv, "4");
    if (q.ssm_value_dim != q.ssm_state_size * q.ssm_v_heads)
        return bad("ssm.inner_size", q.ssm_value_dim, "state_size * time_step_rank");
    if (q.ssm_k_heads <= 0 || q.ssm_v_heads % q.ssm_k_heads != 0)
        return bad("ssm.time_step_rank", q.ssm_v_heads, "a multiple of ssm.group_count");
    if (q.n_head_kv <= 0 || q.n_head % q.n_head_kv != 0)
        return bad("attention.head_count", q.n_head, "a multiple of head_count_kv");
    if (q.qsa_interval != cap.qsa_interval) return bad("full_attention_interval", q.qsa_interval, "4");
    if (q.n_layers <= 0 || q.n_layers > cap.n_layers || q.n_layers % q.qsa_interval != 0)
        return bad("block_count - nextn_predict_layers", q.n_layers, "a multiple of 4, at most 48");
    if (q.n_mtp_layers < 0 || q.n_mtp_layers > 1) return bad("nextn_predict_layers", q.n_mtp_layers, "0 or 1");
    if (q.n_embd <= 0 || q.n_embd > cap.n_embd || q.n_embd % 256 != 0)
        return bad("embedding_length", q.n_embd, "a multiple of 256, at most 2560");
    if (q.n_head > cap.n_head) return bad("attention.head_count", q.n_head, "at most 24");
    if (q.n_head_kv != cap.n_head_kv) return bad("attention.head_count_kv", q.n_head_kv, "2");
    if (q.ssm_k_heads > cap.ssm_k_heads) return bad("ssm.group_count", q.ssm_k_heads, "at most 16");
    if (q.ssm_v_heads > cap.ssm_v_heads) return bad("ssm.time_step_rank", q.ssm_v_heads, "at most 48");
    if (q.n_expert <= 0 || q.n_expert > cap.n_expert) return bad("expert_count", q.n_expert, "at most 512");
    if (q.n_expert_used <= 0 || q.n_expert_used > cap.n_expert_used)
        return bad("expert_used_count", q.n_expert_used, "at most 10");
    if (q.n_ff <= 0 || q.n_ff > cap.n_ff || q.n_ff % 256 != 0)
        return bad("expert_feed_forward_length", q.n_ff, "a multiple of 256, at most 640");
    if (n_ff_shexp != q.n_ff)
        return bad("expert_shared_feed_forward_length", n_ff_shexp, "expert_feed_forward_length");
    g = q;
    return true;
}

std::string LayerView::name(const char* suffix) const {
    return "blk." + std::to_string(layer_) + "." + suffix;
}

bool check_layer(const WeightTable& table, const ModelGeometry& g, int64_t layer, std::string& err) {
    if (layer < 0 || layer >= g.n_layers) {
        err = "layer " + std::to_string(layer) + " is outside 0.." + std::to_string(g.n_layers - 1);
        return false;
    }
    return check_one(table, g, layer, err);
}

bool check_all(const WeightTable& table, const ModelGeometry& g, std::string& err) {
    int64_t n_qsa = 0, n_gdn = 0;
    for (int64_t l = 0; l < g.n_layers; ++l) {
        if (!check_one(table, g, l, err)) return false;
        if (is_qsa_layer(g, l)) {
            ++n_qsa;
        } else {
            ++n_gdn;
        }
    }
    // The split is derived twice over in `docs/semantics.md` - the interval and an explicit ratio array -
    // and both give 36 GDN and 12 QSA.  Counting them here is the check that the LAYER TYPE PREDICATE and
    // the pack agree, which no per-tensor shape check can see.
    if (n_qsa != g.n_qsa_layers() || n_gdn != g.n_gdn_layers()) {
        char buf[256];
        std::snprintf(buf, sizeof buf, "layer split is %lld QSA / %lld GDN, the geometry says %lld / %lld",
                      (long long) n_qsa, (long long) n_gdn, (long long) g.n_qsa_layers(),
                      (long long) g.n_gdn_layers());
        err = buf;
        return false;
    }
    return true;
}

}  // namespace strata::core
