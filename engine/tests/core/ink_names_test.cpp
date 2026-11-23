// strata-inkling: the DSpark drafter tensors a GGUF / safetensors drafter hands to a backend carry the "dspark." prefix, the
// spelling the CUDA / HIP / SYCL / Metal runners look up (#6762, #6784, #6717: "missing tensor 'dspark.layers.0.self_attn.o_proj.weight'").
// The 54 tensor names of Unsloth's dspark-Inkling-2-Mini-Q8_0.gguf (arch "dflash") are checked against the names the runners ask for.
#include "strata/core/ink_names.hpp"

#include <cstdio>
#include <set>
#include <string>
#include <vector>

using namespace strata::core;

namespace {
int failed = 0;
#define CHECK(x) do { if (!(x)) { std::fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #x); ++failed; } } while (0)
}  // namespace

int main() {
    // prefix helpers: idempotent both ways
    CHECK(ink_dspark_name("layers.0.self_attn.o_proj.weight") == "dspark.layers.0.self_attn.o_proj.weight");
    CHECK(ink_dspark_name("dspark.fuse_weight") == "dspark.fuse_weight");
    CHECK(ink_dspark_name("fuse_weight") == "dspark.fuse_weight");
    CHECK(ink_dspark_plain("dspark.fuse_weight") == "fuse_weight");
    CHECK(ink_dspark_plain("fuse_weight") == "fuse_weight");
    CHECK(ink_dspark_name(ink_dspark_plain("dspark.mask_embed")) == "dspark.mask_embed");

    // the 9 global tensors and the 15 per-block tensors of the real dspark GGUF
    std::vector<std::string> gguf = {
        "dspark_fuse.weight", "dspark_hidden_out.bias", "dspark_hidden_out.weight", "dspark_norm.weight", "joint_proj.bias",
        "joint_proj.weight", "markov_w1.weight", "mask_embd.weight", "output_norm.weight"};
    const char* blk[] = {"attn_k.weight", "attn_k_ctx.weight", "attn_k_ctx_norm.weight", "attn_k_norm.weight", "attn_norm.weight",
                         "attn_output.weight", "attn_q.weight", "attn_q_norm.weight", "attn_v.weight", "attn_v_ctx.weight",
                         "dflare_norm.weight", "ffn_down.weight", "ffn_gate.weight", "ffn_norm.weight", "ffn_up.weight"};
    for (int il = 0; il < 3; ++il)
        for (const char* b : blk) gguf.push_back("blk." + std::to_string(il) + "." + b);
    CHECK(gguf.size() == 54);

    // what ink_dspark.cuh / the other runners ask for (dspark.<...>)
    std::set<std::string> wanted = {
        "dspark.fuse_weight", "dspark.mask_embed", "dspark.norm.weight", "dspark.markov_head.dflare_norm.weight",
        "dspark.markov_head.hidden_out.bias", "dspark.markov_head.hidden_out.weight", "dspark.markov_head.markov_w1.weight",
        "dspark.markov_head.rnn_proj.bias", "dspark.markov_head.rnn_proj.weight"};
    const char* lw[] = {"dflare_norm.weight", "input_layernorm.weight", "post_attention_layernorm.weight", "mlp.down_proj.weight",
                        "mlp.gate_proj.weight", "mlp.up_proj.weight", "self_attn.q_proj.weight", "self_attn.k_proj.weight",
                        "self_attn.v_proj.weight", "self_attn.k_proj_ctx.weight", "self_attn.v_proj_ctx.weight",
                        "self_attn.o_proj.weight", "self_attn.q_norm.weight", "self_attn.k_norm.weight", "self_attn.k_norm_ctx.weight"};
    for (int il = 0; il < 3; ++il)
        for (const char* w : lw) wanted.insert("dspark.layers." + std::to_string(il) + "." + w);
    CHECK(wanted.size() == 54);

    std::set<std::string> got;
    for (const auto& n : gguf) {
        const std::string m = ink_dspark_name_from_gguf(n);
        CHECK(m.compare(0, 7, "dspark.") == 0);
        got.insert(m);
    }
    CHECK(got == wanted);
    for (const auto& w : wanted) {
        if (!got.count(w)) std::fprintf(stderr, "not produced: %s\n", w.c_str());
    }
    // already-canonical names pass through unchanged
    for (const auto& w : wanted) CHECK(ink_dspark_name_from_gguf(w) == w);

    if (failed) {
        std::fprintf(stderr, "ink_names_test: %d failed\n", failed);
        return 1;
    }
    std::printf("ink_names_test: ok\n");
    return 0;
}
