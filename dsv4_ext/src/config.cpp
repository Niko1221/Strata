#include "dsv4/config.hpp"

#include <cstdio>
#include <cstdlib>
#include <sstream>

namespace dsv4 {
namespace {

const char* P = "deepseek4.";

struct Getter {
    const GgufHeader& h;
    std::string err;
    const GgufValue* req(const std::string& key) {
        const GgufValue* v = h.find(key);
        if (!v && err.empty()) err = "missing GGUF key: " + key;
        return v;
    }
    int64_t i(const std::string& key) {
        const GgufValue* v = req(P + key);
        return v ? v->i : 0;
    }
    double f(const std::string& key) {
        const GgufValue* v = req(P + key);
        return v ? v->f : 0.0;
    }
    template <class T> std::vector<T> arr(const std::string& key) {
        std::vector<T> out;
        const GgufValue* v = req(P + key);
        if (!v) return out;
        if (v->type != GGUF_ARR || !v->nums_complete()) { if (err.empty()) err = "key is not a complete numeric array: " + key; return out; }
        for (double d : v->nums) out.push_back((T) d);
        return out;
    }
};

}  // namespace

bool config_from_gguf(const GgufHeader& h, Dsv4Config& c, std::string& err) {
    c = Dsv4Config();
    const GgufValue* a = h.find("general.architecture");
    if (!a || a->s != "deepseek4") { err = "general.architecture is not 'deepseek4' (got '" + (a ? a->s : std::string("<none>")) + "')"; return false; }
    c.arch = a->s;
    Getter g{h, ""};
    c.n_ctx_train = g.i("context_length");
    c.n_layer = (int) g.i("block_count");
    c.n_embd = (int) g.i("embedding_length");
    c.n_head = (int) g.i("attention.head_count");
    c.n_head_kv = (int) g.i("attention.head_count_kv");
    c.key_len = (int) g.i("attention.key_length");
    c.val_len = (int) g.i("attention.value_length");
    c.rope_dim = (int) g.i("rope.dimension_count");
    c.q_lora = (int) g.i("attention.q_lora_rank");
    c.o_lora = (int) g.i("attention.output_lora_rank");
    c.o_groups = (int) g.i("attention.output_group_count");
    c.sliding_window = (int) g.i("attention.sliding_window");
    c.n_expert = (int) g.i("expert_count");
    c.n_expert_used = (int) g.i("expert_used_count");
    c.n_expert_shared = (int) g.i("expert_shared_count");
    c.ff_exp = (int) g.i("expert_feed_forward_length");
    c.gating_func = (int) g.i("expert_gating_func");
    c.exp_w_scale = (float) g.f("expert_weights_scale");
    c.exp_w_norm = g.i("expert_weights_norm") != 0;
    c.n_hash_layers = (int) g.i("hash_layer_count");
    c.idx_heads = (int) g.i("attention.indexer.head_count");
    c.idx_key_len = (int) g.i("attention.indexer.key_length");
    c.idx_topk = (int) g.i("attention.indexer.top_k");
    c.hc_count = (int) g.i("hyper_connection.count");
    c.hc_sinkhorn_iters = (int) g.i("hyper_connection.sinkhorn_iterations");
    c.hc_eps = (float) g.f("hyper_connection.epsilon");
    c.rms_eps = (float) g.f("attention.layer_norm_rms_epsilon");
    c.rope_base = (float) g.f("rope.freq_base");
    c.rope_base_compress = (float) g.f("attention.compress_rope_freq_base");
    c.yarn_factor = (float) g.f("rope.scaling.factor");
    c.yarn_beta_fast = (float) g.f("rope.scaling.yarn_beta_fast");
    c.yarn_beta_slow = (float) g.f("rope.scaling.yarn_beta_slow");
    c.yarn_orig_ctx = g.i("rope.scaling.original_context_length");
    c.compress_ratios = g.arr<int>("attention.compress_ratios");
    c.swiglu_clamp_exp = g.arr<float>("swiglu_clamp_exp");
    c.swiglu_clamp_shexp = g.arr<float>("swiglu_clamp_shexp");
    if (!g.err.empty()) { err = g.err; return false; }

    if (c.n_layer <= 0 || c.n_embd <= 0 || c.n_expert <= 0 || c.ff_exp <= 0) { err = "non-positive core dimension"; return false; }
    if (c.n_expert_used <= 0 || c.n_expert_used > c.n_expert) { err = "expert_used_count out of range"; return false; }
    if ((int) c.compress_ratios.size() < c.n_layer) { err = "compress_ratios shorter than block_count"; return false; }
    if ((int) c.swiglu_clamp_exp.size() != c.n_layer || (int) c.swiglu_clamp_shexp.size() != c.n_layer) {
        err = "swiglu_clamp arrays must have block_count entries"; return false;
    }
    if (c.n_hash_layers < 0 || c.n_hash_layers > c.n_layer) { err = "hash_layer_count out of range"; return false; }
    c.n_extra_layers = (int) c.compress_ratios.size() - c.n_layer;

    if (const GgufValue* t = h.find("tokenizer.ggml.tokens")) c.vocab = (int64_t) t->arr_len;
    if (const GgufValue* v = h.find("tokenizer.ggml.bos_token_id")) c.bos = (int) v->i;
    if (const GgufValue* v = h.find("tokenizer.ggml.eos_token_id")) c.eos = (int) v->i;
    if (const GgufValue* v = h.find("tokenizer.ggml.padding_token_id")) c.pad = (int) v->i;
    return true;
}

std::string config_summary(const Dsv4Config& c) {
    std::ostringstream o;
    o << "arch " << c.arch << ": " << c.n_layer << " layers (+" << c.n_extra_layers << " extra), n_embd " << c.n_embd
      << ", heads " << c.n_head << "/" << c.n_head_kv << " kv, head_dim " << c.key_len << " (rope " << c.rope_dim << ")\n"
      << "  moe: " << c.n_expert << " experts, top-" << c.n_expert_used << ", shared " << c.n_expert_shared << ", ff "
      << c.ff_exp << ", gating_func " << c.gating_func << ", scale " << c.exp_w_scale << ", norm " << c.exp_w_norm
      << ", hash layers " << c.n_hash_layers << "\n"
      << "  attn: q_lora " << c.q_lora << ", o_lora " << c.o_lora << " x" << c.o_groups << " groups, window "
      << c.sliding_window << ", indexer " << c.idx_heads << "x" << c.idx_key_len << " top" << c.idx_topk << "\n"
      << "  hc: " << c.hc_count << " streams, sinkhorn " << c.hc_sinkhorn_iters << " iters, eps " << c.hc_eps << "\n"
      << "  vocab " << c.vocab << ", ctx " << c.n_ctx_train << "\n";
    return o.str();
}

bool inventory_from_gguf(const GgufHeader& h, const Dsv4Config& c, ExpertInventory& inv, std::string& err) {
    inv = ExpertInventory();
    inv.bytes_per_expert.assign((size_t) c.n_layer, 0);
    inv.types.assign((size_t) c.n_layer, std::string());
    inv.bytes_per_expert_extra.assign((size_t) (c.n_extra_layers > 0 ? c.n_extra_layers : 0), 0);
    std::vector<int> seen((size_t) c.n_layer, 0);
    for (const GgufTensor& t : h.tensors) {
        if (!t.known_type) { ++inv.unknown_type_tensors; continue; }
        const bool expert = t.name.find("_exps") != std::string::npos && t.dims.size() == 3 &&
                            t.dims.back() == (uint64_t) c.n_expert;
        if (!expert) { inv.other_bytes += t.nbytes; continue; }
        if (t.nbytes % (uint64_t) c.n_expert != 0) { err = "expert tensor bytes not divisible by n_expert: " + t.name; return false; }
        const size_t b = t.name.find("blk.");
        if (b == std::string::npos) { err = "expert tensor without 'blk.N.' prefix: " + t.name; return false; }
        const int layer = std::atoi(t.name.c_str() + b + 4);
        const uint64_t per = t.nbytes / (uint64_t) c.n_expert;
        if (layer >= 0 && layer < c.n_layer) {
            inv.bytes_per_expert[(size_t) layer] += per;
            inv.total_expert_bytes += t.nbytes;
            inv.types[(size_t) layer] += (seen[(size_t) layer]++ ? "/" : "") + ggml_type_str(t.type);
        } else if (layer >= c.n_layer && layer - c.n_layer < c.n_extra_layers) {
            inv.bytes_per_expert_extra[(size_t) (layer - c.n_layer)] += per;
            inv.extra_expert_bytes += t.nbytes;
        } else {
            err = "expert tensor layer index out of range: " + t.name; return false;
        }
    }
    for (int l = 0; l < c.n_layer; ++l) if (inv.bytes_per_expert[(size_t) l] == 0) ++inv.layers_without_experts;
    return true;
}

}  // namespace dsv4
