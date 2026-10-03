// qwen36_config.hpp - the model's dimensions, read from the GGUF.  Header-only and CUDA-free so it can be tested on a
// CPU-only machine (tests/config_test.cpp).
//
// Dimensions come from TENSOR SHAPES (they cannot disagree with the data); metadata is only used for what a tensor
// cannot say: rope base, rotary dims, rms eps and how many experts are used per token.
#pragma once

#include <cstdint>
#include <cstdio>
#include <stdexcept>
#include <string>
#include <vector>

#include "strata/artifact/gguf_reader.hpp"

#if defined(__GNUC__) && !defined(__clang__)
#pragma GCC diagnostic ignored "-Wdangling-reference"  // need() returns a reference into the model, not into its argument
#endif

namespace q36 {

struct Config {
    std::string arch;
    int n_layer = 0, n_embd = 0, n_vocab = 0;
    int n_head = 0, n_head_kv = 0, head_dim = 0, rot_dim = 0;
    float rope_theta = 1e7f, eps = 1e-6f;
    int n_expert = 0, n_used = 8, n_ff_exp = 0, n_ff_shexp = 0;
    int ssm_hk = 0, ssm_hv = 0, ssm_S = 0, ssm_conv = 4, ssm_qkv = 0, ssm_inner = 0;
    std::vector<uint8_t> is_gdn;  // per layer
    int n_gdn = 0, n_attn = 0;
    int context_length = 0;       // from metadata, 0 if absent

    std::string describe() const {
        char b[512];
        std::snprintf(b, sizeof b,
                      "arch=%s layers=%d (gdn=%d attn=%d) hidden=%d vocab=%d | attn heads=%d kv=%d hd=%d rot=%d theta=%g"
                      " | gdn hk=%d hv=%d S=%d conv=%d | moe experts=%d used=%d ff=%d shared_ff=%d eps=%g",
                      arch.c_str(), n_layer, n_gdn, n_attn, n_embd, n_vocab, n_head, n_head_kv, head_dim, rot_dim,
                      (double)rope_theta, ssm_hk, ssm_hv, ssm_S, ssm_conv, n_expert, n_used, n_ff_exp, n_ff_shexp,
                      (double)eps);
        return b;
    }
};

inline std::string blk(int il, const char* what) { return "blk." + std::to_string(il) + "." + what; }

inline const strata::TensorInfo& need(const strata::GgufModel& m, std::string name) {
    const strata::TensorInfo* t = m.find(name);
    if (!t) throw std::runtime_error("tensor not found in the GGUF: " + name);
    return *t;
}

inline Config load_config(const strata::GgufModel& m) {
    Config c;
    const strata::MetaValue* a = m.meta().get("general.architecture");
    c.arch = a ? a->s : "";
    auto num = [&](const char* key, double def) {
        const strata::MetaValue* v = m.meta().get(c.arch + "." + key);
        return v && v->is_num() ? v->num() : def;
    };

    const strata::TensorInfo& emb = need(m, "token_embd.weight");
    if (emb.shape.size() != 2) throw std::runtime_error("token_embd.weight is not 2-D");
    c.n_embd = (int)emb.shape[0];
    c.n_vocab = (int)emb.shape[1];

    const int n_block = (int)num("block_count", 0);
    for (int il = 0; il < n_block || (n_block == 0 && il < 512); ++il) {
        const bool has_moe = m.find(blk(il, "ffn_gate_inp.weight")) != nullptr;
        const bool gdn = m.find(blk(il, "attn_qkv.weight")) != nullptr;
        const bool att = m.find(blk(il, "attn_q.weight")) != nullptr;
        const bool mtp = m.find(blk(il, "nextn.eh_proj.weight")) != nullptr;  // multi-token-prediction block: not a layer
        if (!has_moe || mtp || (!gdn && !att)) break;
        c.is_gdn.push_back(gdn ? 1 : 0);
    }
    c.n_layer = (int)c.is_gdn.size();
    if (c.n_layer == 0) throw std::runtime_error("no decoder layers found (expected blk.N.ffn_gate_inp.weight and "
                                                 "blk.N.attn_qkv.weight or blk.N.attn_q.weight); arch='" + c.arch + "'");
    for (uint8_t g : c.is_gdn) (g ? c.n_gdn : c.n_attn)++;

    // MoE (every layer is the same shape)
    const strata::TensorInfo& gi = need(m, blk(0, "ffn_gate_inp.weight"));
    if (gi.shape.size() != 2 || (int)gi.shape[0] != c.n_embd) throw std::runtime_error("ffn_gate_inp.weight shape");
    c.n_expert = (int)gi.shape[1];
    const strata::TensorInfo& ge = need(m, blk(0, "ffn_gate_exps.weight"));
    if (ge.shape.size() != 3 || (int)ge.shape[2] != c.n_expert) throw std::runtime_error("ffn_gate_exps.weight shape");
    c.n_ff_exp = (int)ge.shape[1];
    c.n_ff_shexp = (int)need(m, blk(0, "ffn_gate_shexp.weight")).shape[1];
    c.n_used = (int)num("expert_used_count", 8);
    if (c.n_used < 1 || c.n_used > c.n_expert || c.n_used > 64) throw std::runtime_error("expert_used_count out of range");

    for (int il = 0; il < c.n_layer; ++il) {
        if (c.is_gdn[il]) {
            if (c.ssm_S) continue;
            const auto& qkv = need(m, blk(il, "attn_qkv.weight"));
            const auto& alpha = need(m, blk(il, "ssm_alpha.weight"));
            const auto& nrm = need(m, blk(il, "ssm_norm.weight"));
            const auto& conv = need(m, blk(il, "ssm_conv1d.weight"));
            c.ssm_qkv = (int)qkv.shape[1];
            c.ssm_hv = (int)alpha.shape[1];
            c.ssm_S = (int)nrm.shape[0];
            c.ssm_conv = (int)conv.shape[0];
            c.ssm_inner = c.ssm_hv * c.ssm_S;
            const int rest = c.ssm_qkv - c.ssm_inner;
            if (rest <= 0 || rest % (2 * c.ssm_S)) throw std::runtime_error("attn_qkv width is not q+k+v for these heads");
            c.ssm_hk = rest / (2 * c.ssm_S);
            if (c.ssm_hv % c.ssm_hk) throw std::runtime_error("value heads are not a multiple of key heads");
            if (c.ssm_S != 128) throw std::runtime_error("the GDN kernels need head size 128");
            if (c.ssm_conv != 4) throw std::runtime_error("the GDN conv kernel needs kernel size 4");
        } else if (!c.head_dim) {
            const auto& q = need(m, blk(il, "attn_q.weight"));
            const auto& k = need(m, blk(il, "attn_k.weight"));
            const auto& qn = need(m, blk(il, "attn_q_norm.weight"));
            c.head_dim = (int)qn.shape[0];
            c.n_head = (int)q.shape[1] / (2 * c.head_dim);
            c.n_head_kv = (int)k.shape[1] / c.head_dim;
            if (c.n_head % c.n_head_kv) throw std::runtime_error("attention heads are not a multiple of kv heads");
            if (c.head_dim != 128 && c.head_dim != 256) throw std::runtime_error("attention kernel needs head_dim 128 or 256");
        }
    }
    c.rot_dim = (int)num("rope.dimension_count", c.head_dim ? c.head_dim / 4 : 0);
    c.rope_theta = (float)num("rope.freq_base", 1e7);
    c.eps = (float)num("attention.layer_norm_rms_epsilon", 1e-6);
    c.context_length = (int)num("context_length", 0);
    if (c.head_dim && (c.rot_dim <= 0 || (c.rot_dim & 1) || c.rot_dim > c.head_dim))
        throw std::runtime_error("rope.dimension_count must be even and <= head_dim");
    return c;
}

}  // namespace q36
