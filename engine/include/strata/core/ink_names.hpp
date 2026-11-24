// Inkling-2-Mini tensor names that every backend looks up (strata-inkling): the DSpark drafter tensors are always
// "dspark.<name>" (the .strata v1.1 manifest spells them that way, and the CUDA / HIP / SYCL / Metal runners ask for
// "dspark.layers.0.self_attn.o_proj.weight", "dspark.fuse_weight", ...). A drafter that comes from an Unsloth
// dspark-*.gguf or from mtp.safetensors carries the plain names ("layers.0...", "fuse_weight"), so a loader must put the
// prefix on before it hands the tensor to a backend (#6762, #6784, #6717: "missing tensor 'dspark.layers.0...'").
#pragma once

#include <string>

namespace strata::core {

constexpr const char* kInkDsparkPrefix = "dspark.";

// "layers.0.x" -> "dspark.layers.0.x"; a name that already starts with "dspark." is returned unchanged.
inline std::string ink_dspark_name(const std::string& n) {
    if (n.compare(0, 7, kInkDsparkPrefix) == 0) return n;
    return std::string(kInkDsparkPrefix) + n;
}

// "dspark.layers.0.x" -> "layers.0.x"; a plain name is returned unchanged.
inline std::string ink_dspark_plain(const std::string& n) {
    if (n.compare(0, 7, kInkDsparkPrefix) == 0) return n.substr(7);
    return n;
}

// The name a tensor of Unsloth's dspark-Inkling-2-Mini-*.gguf (arch "dflash", 54 tensors) has in the .strata drafter, WITHOUT the
// "dspark." prefix: blk.N.attn_output.weight -> layers.N.self_attn.o_proj.weight, dspark_fuse.weight -> fuse_weight, ...
inline std::string ink_dspark_plain_from_gguf(const std::string& n) {
    if (n == "dspark_fuse.weight" || n == "dspark.fuse_weight" || n == "fuse_weight") return "fuse_weight";
    if (n == "mask_embd.weight" || n == "dspark.mask_embed" || n == "mask_embed") return "mask_embed";
    if (n == "output_norm.weight" || n == "dspark.output_norm.weight" || n == "norm.weight") return "norm.weight";
    if (n == "dspark_norm.weight") return "markov_head.dflare_norm.weight";
    if (n == "dspark_hidden_out.weight") return "markov_head.hidden_out.weight";
    if (n == "dspark_hidden_out.bias") return "markov_head.hidden_out.bias";
    if (n == "markov_w1.weight") return "markov_head.markov_w1.weight";
    if (n == "joint_proj.weight") return "markov_head.rnn_proj.weight";
    if (n == "joint_proj.bias") return "markov_head.rnn_proj.bias";
    if (n.compare(0, 4, "blk.") == 0) {
        const size_t dot = n.find('.', 4);
        if (dot != std::string::npos) {
            const std::string idx = n.substr(4, dot - 4);
            const std::string rest = n.substr(dot + 1);
            const std::string lp = "layers." + idx + ".";
            if (rest == "dflare_norm.weight") return lp + "dflare_norm.weight";
            if (rest == "attn_norm.weight") return lp + "input_layernorm.weight";
            if (rest == "ffn_norm.weight") return lp + "post_attention_layernorm.weight";
            if (rest == "ffn_down.weight") return lp + "mlp.down_proj.weight";
            if (rest == "ffn_gate.weight") return lp + "mlp.gate_proj.weight";
            if (rest == "ffn_up.weight") return lp + "mlp.up_proj.weight";
            if (rest == "attn_q.weight") return lp + "self_attn.q_proj.weight";
            if (rest == "attn_k.weight") return lp + "self_attn.k_proj.weight";
            if (rest == "attn_v.weight") return lp + "self_attn.v_proj.weight";
            if (rest == "attn_k_ctx.weight") return lp + "self_attn.k_proj_ctx.weight";
            if (rest == "attn_v_ctx.weight") return lp + "self_attn.v_proj_ctx.weight";
            if (rest == "attn_output.weight") return lp + "self_attn.o_proj.weight";
            if (rest == "attn_q_norm.weight") return lp + "self_attn.q_norm.weight";
            if (rest == "attn_k_norm.weight") return lp + "self_attn.k_norm.weight";
            if (rest == "attn_k_ctx_norm.weight") return lp + "self_attn.k_norm_ctx.weight";
        }
    }
    if (n.compare(0, 7, kInkDsparkPrefix) == 0) return n.substr(7);
    return n;
}

// The name every backend finds the drafter tensor under (the .strata spelling): "dspark." + the plain name.
inline std::string ink_dspark_name_from_gguf(const std::string& n) {
    return ink_dspark_name(ink_dspark_plain_from_gguf(n));
}

}  // namespace strata::core
