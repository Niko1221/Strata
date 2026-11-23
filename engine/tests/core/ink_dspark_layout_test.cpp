// strata-inkling: a drafter file in a layout the runners do not read is dropped, never fatal (strata/core/ink_dspark_layout.hpp).
// The two real files of unsloth/Inkling-2-Mini-GGUF: dspark-Inkling-2-Mini-Q8_0_Old.gguf (402,278,560 bytes, the layout the engines read) and the
// dspark-Inkling-2-Mini-Q8_0.gguf of 2026-11-23 14:54 UTC (503,889,952 bytes, llama.cpp mainline "dflash": ctx_mix / ctx_norm / markov_w2 ...).
#include "strata/core/ink_dspark_layout.hpp"
#include "strata/core/ink_names.hpp"

#include <cstdio>
#include <set>
#include <string>
#include <vector>

using namespace strata::core;

namespace {
int failed = 0;
#define CHECK(x) do { if (!(x)) { std::fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #x); ++failed; } } while (0)

const std::vector<std::string> kOld = {
    "dspark_fuse.weight",
    "dspark_hidden_out.bias",
    "dspark_hidden_out.weight",
    "dspark_norm.weight",
    "joint_proj.bias",
    "joint_proj.weight",
    "markov_w1.weight",
    "mask_embd.weight",
    "output_norm.weight",
    "blk.0.attn_k.weight",
    "blk.0.attn_k_ctx.weight",
    "blk.0.attn_k_ctx_norm.weight",
    "blk.0.attn_k_norm.weight",
    "blk.0.attn_norm.weight",
    "blk.0.attn_output.weight",
    "blk.0.attn_q.weight",
    "blk.0.attn_q_norm.weight",
    "blk.0.attn_v.weight",
    "blk.0.attn_v_ctx.weight",
    "blk.0.dflare_norm.weight",
    "blk.0.ffn_down.weight",
    "blk.0.ffn_gate.weight",
    "blk.0.ffn_norm.weight",
    "blk.0.ffn_up.weight",
    "blk.1.attn_k.weight",
    "blk.1.attn_k_ctx.weight",
    "blk.1.attn_k_ctx_norm.weight",
    "blk.1.attn_k_norm.weight",
    "blk.1.attn_norm.weight",
    "blk.1.attn_output.weight",
    "blk.1.attn_q.weight",
    "blk.1.attn_q_norm.weight",
    "blk.1.attn_v.weight",
    "blk.1.attn_v_ctx.weight",
    "blk.1.dflare_norm.weight",
    "blk.1.ffn_down.weight",
    "blk.1.ffn_gate.weight",
    "blk.1.ffn_norm.weight",
    "blk.1.ffn_up.weight",
    "blk.2.attn_k.weight",
    "blk.2.attn_k_ctx.weight",
    "blk.2.attn_k_ctx_norm.weight",
    "blk.2.attn_k_norm.weight",
    "blk.2.attn_norm.weight",
    "blk.2.attn_output.weight",
    "blk.2.attn_q.weight",
    "blk.2.attn_q_norm.weight",
    "blk.2.attn_v.weight",
    "blk.2.attn_v_ctx.weight",
    "blk.2.dflare_norm.weight",
    "blk.2.ffn_down.weight",
    "blk.2.ffn_gate.weight",
    "blk.2.ffn_norm.weight",
    "blk.2.ffn_up.weight"
};
const std::vector<std::string> kNew = {
    "joint_proj.bias",
    "joint_proj.weight",
    "markov_ctx_mix.weight",
    "markov_ctx_norm.weight",
    "markov_w1.weight",
    "markov_w2.bias",
    "markov_w2.weight",
    "mask_embd.weight",
    "output_norm.weight",
    "token_embd_norm.weight",
    "blk.0.attn_k.weight",
    "blk.0.attn_k_ctx.weight",
    "blk.0.attn_k_ctx_norm.weight",
    "blk.0.attn_k_norm.weight",
    "blk.0.attn_norm.weight",
    "blk.0.attn_output.weight",
    "blk.0.attn_q.weight",
    "blk.0.attn_q_norm.weight",
    "blk.0.attn_v.weight",
    "blk.0.attn_v_ctx.weight",
    "blk.0.ctx_mix.weight",
    "blk.0.ctx_norm.weight",
    "blk.0.ffn_down.weight",
    "blk.0.ffn_gate.weight",
    "blk.0.ffn_norm.weight",
    "blk.0.ffn_up.weight",
    "blk.1.attn_k.weight",
    "blk.1.attn_k_ctx.weight",
    "blk.1.attn_k_ctx_norm.weight",
    "blk.1.attn_k_norm.weight",
    "blk.1.attn_norm.weight",
    "blk.1.attn_output.weight",
    "blk.1.attn_q.weight",
    "blk.1.attn_q_norm.weight",
    "blk.1.attn_v.weight",
    "blk.1.attn_v_ctx.weight",
    "blk.1.ctx_mix.weight",
    "blk.1.ctx_norm.weight",
    "blk.1.ffn_down.weight",
    "blk.1.ffn_gate.weight",
    "blk.1.ffn_norm.weight",
    "blk.1.ffn_up.weight",
    "blk.2.attn_k.weight",
    "blk.2.attn_k_ctx.weight",
    "blk.2.attn_k_ctx_norm.weight",
    "blk.2.attn_k_norm.weight",
    "blk.2.attn_norm.weight",
    "blk.2.attn_output.weight",
    "blk.2.attn_q.weight",
    "blk.2.attn_q_norm.weight",
    "blk.2.attn_v.weight",
    "blk.2.attn_v_ctx.weight",
    "blk.2.ctx_mix.weight",
    "blk.2.ctx_norm.weight",
    "blk.2.ffn_down.weight",
    "blk.2.ffn_gate.weight",
    "blk.2.ffn_norm.weight",
    "blk.2.ffn_up.weight"
};

std::set<std::string> plain(const std::vector<std::string>& gguf) {
    std::set<std::string> s;
    for (const auto& n : gguf) s.insert(ink_dspark_plain(ink_dspark_name_from_gguf(n)));
    return s;
}
}  // namespace

int main() {
    std::string why;
    // the old layout is read
    CHECK(kOld.size() == 54);
    CHECK(ink_dspark_layout_check(plain(kOld), &why));
    CHECK(why.empty());
    // the mainline layout is not, and the reason names it
    CHECK(kNew.size() == 58);
    CHECK(!ink_dspark_layout_check(plain(kNew), &why));
    CHECK(why.find("markov_head.hidden_out") != std::string::npos || why.find("fuse_weight") != std::string::npos);
    CHECK(why.find("mainline") != std::string::npos);
    const std::string msg = ink_dspark_layout_message("dspark-Inkling-2-Mini-Q8_0.gguf", why);
    CHECK(msg.find("drafting is off for this start and the model starts normally") != std::string::npos);
    CHECK(msg.find("Q8_0_Old.gguf") != std::string::npos);
    // a file with one required tensor missing is refused, the others still count
    for (const char* gone : {"fuse_weight", "mask_embed", "markov_head.hidden_out.weight", "markov_head.hidden_out.bias", "markov_head.dflare_norm.weight"}) {
        std::set<std::string> s = plain(kOld);
        s.erase(gone);
        std::string w;
        CHECK(!ink_dspark_layout_check(s, &w));
        CHECK(w.find(gone) != std::string::npos);
    }
    // names that arrive with the dspark. prefix (the .strata and mtp.safetensors spelling) pass the same check
    std::set<std::string> s2;
    for (const auto& n : kOld) s2.insert(ink_dspark_plain(ink_dspark_name_from_gguf(n)));
    CHECK(ink_dspark_layout_check(s2, nullptr));
    if (failed) { std::fprintf(stderr, "%d check(s) failed\n", failed); return 1; }
    std::printf("ink_dspark_layout_test: all checks passed\n");
    return 0;
}
