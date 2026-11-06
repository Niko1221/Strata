// cann/tests/port_check.cpp - what this engine would need versus what a GGUF actually has.
//
// Reads only the header, so it works on a partially downloaded model. Prints every shape the
// engine hardcodes next to the file's own value, then the tensors the loader asks for that the
// file does not contain.
#include <cstdio>
#include <initializer_list>
#include <string>
#include <vector>

#include "strata/cann/gguf.hpp"

using namespace strata::cann;

namespace {

struct Expect {
    const char * what;   // human name
    const char * key;    // metadata key suffix, or nullptr for a derived value
    int64_t      ours;   // what the engine has today
    int64_t      file;   // value read from the file
};

void row(const char * what, int64_t ours, int64_t file) {
    const bool same = ours == file;
    printf("%-26s ours %-8lld file %-8lld %s\n", what, (long long) ours, (long long) file,
           same ? "ok" : "DIFFERS");
}

} // namespace

int main(int argc, char ** argv) {
    if (argc < 2) {
        printf("usage: port_check <model.gguf>\n");
        return 2;
    }
    const Gguf g(argv[1]);
    const std::string a = g.arch();
    auto kv = [&](const char * suffix) {
        return g.metadata_int(a + "." + suffix);
    };

    printf("file   %s\narch   %s\n", argv[1], a.c_str());
    printf("=== the engine's constants against this file ===\n");

    const int64_t f_layers = kv("block_count");
    const int64_t f_hidden = kv("embedding_length");
    const int64_t f_ssm_groups = kv("ssm.group_count");
    const int64_t f_ssm_rank = kv("ssm.time_step_rank");
    const int64_t f_ssm_state = kv("ssm.state_size");
    const int64_t f_conv = kv("ssm.conv_kernel");
    const int64_t f_heads = kv("attention.head_count");
    const int64_t f_kv_heads = kv("attention.head_count_kv");
    const int64_t f_klen = kv("attention.key_length");
    const int64_t f_rot = kv("rope.dimension_count");
    const int64_t f_inter = kv("expert_feed_forward_length");
    const int64_t f_topk = kv("expert_used_count");
    const int64_t f_experts = kv("expert_count");

    row("hidden",          2048, f_hidden);
    row("layers",            40, f_layers);
    row("linear k heads",    16, f_ssm_groups);
    row("linear v heads",    32, f_ssm_rank);
    row("linear state dim", 128, f_ssm_state);
    row("conv kernel",        4, f_conv);
    row("attn q heads",      16, f_heads);
    row("attn kv heads",      2, f_kv_heads);
    row("attn head dim",    256, f_klen);
    row("rope dim",          64, f_rot);
    row("expert inter",     512, f_inter);
    row("experts used",       8, f_topk);
    row("experts total",      0, f_experts);
    printf("\n");
    row("qkv dim (derived)", 8192, 2 * f_ssm_groups * f_ssm_state + f_ssm_rank * f_ssm_state);
    printf("rms eps ours %.1e file %.1e\n", 1e-6,
           g.metadata_flt(a + ".attention.layer_norm_rms_epsilon"));
    printf("rope base ours %.3g file %.3g\n", 1e7, g.metadata_flt(a + ".rope.freq_base"));

    // the tensors the qwen35moe loader asks for, per layer kind
    static const char * linear[] = { "attn_qkv.weight", "attn_gate.weight", "ssm_out.weight",
                                     "ssm_norm.weight", "ssm_beta.weight", "ssm_alpha.weight",
                                     "ssm_conv1d.weight", "ssm_dt.bias", "ssm_a" };
    static const char * attn[]   = { "attn_q.weight", "attn_k.weight", "attn_v.weight",
                                     "attn_output.weight", "attn_q_norm.weight", "attn_k_norm.weight" };
    static const char * shared[] = { "ffn_gate_inp.weight", "ffn_gate_inp_shexp.weight",
                                     "ffn_gate_shexp.weight", "ffn_up_shexp.weight",
                                     "ffn_down_shexp.weight", "ffn_gate_exps.weight",
                                     "ffn_up_exps.weight", "ffn_down_exps.weight" };

    printf("\n=== tensors the loader asks for ===\n");
    int missing = 0;
    for (int il = 0; il < 4; ++il) {
        const bool is_linear = (il % 4) != 3;
        printf("blk.%d (%s)\n", il, is_linear ? "linear" : "attention");
        auto check = [&](const char * suffix) {
            const std::string name = "blk." + std::to_string(il) + "." + suffix;
            const GgufTensor * t = g.find(name);
            printf("   %-28s %s", suffix, t ? "present" : "MISSING");
            if (t) {
                printf("  type %u  dims", t->type);
                for (size_t d = 0; d < t->dims.size() && d < 3; ++d) {
                    printf(" %llu", (unsigned long long) t->dims[d]);
                }
            }
            printf("\n");
            if (!t) {
                ++missing;
            }
        };
        for (const char * s : shared) {
            check(s);
        }
        if (is_linear) {
            for (const char * s : linear) {
                check(s);
            }
        } else {
            for (const char * s : attn) {
                check(s);
            }
            for (const char * s : { "indexer.q_proj.weight", "indexer.k_proj.weight",
                                    "indexer.q_norm.weight", "indexer.k_norm.weight" }) {
                check(s);
            }
        }
        for (const char * s : { "hc_attn_norm.weight", "hc_attn_down.weight", "hc_attn_up.weight",
                                "hc_attn_inject.weight" }) {
            check(s);
        }
        if (il == 1) {
            // the per-layer embedding lands on the second block, not at top level
            for (const char * s : { "ple_key.weight", "ple_value.weight", "ple_conv1d.weight",
                                    "ple_norm_key.weight", "ple_norm_query.weight",
                                    "ple_norm_conv.weight" }) {
                check(s);
            }
        }
    }

    printf("\n=== globals (shard 1 carries all but the n-gram table) ===\n");
    for (const char * s : { "token_embd.weight", "output.weight",
                            "output_hc_down.weight", "output_hc_up.weight",
                            "output_hc_norm.weight", "output_norm.weight",
                            "per_layer_token_embd.weight" }) {
        const GgufTensor * t = g.find(s);
        printf("   %-28s %s\n", s, t ? "present" : "MISSING");
    }

    printf("\ntensors missing from the four blocks sampled: %d\n", missing);
    return 0;
}
