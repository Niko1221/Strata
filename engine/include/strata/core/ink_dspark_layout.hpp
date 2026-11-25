// strata-inkling: a drafter file whose layout this engine cannot read must never stop the model from starting.
//
// 2026-11-23 14:54 UTC Unsloth re-uploaded dspark-Inkling-2-Mini-Q8_0.gguf in llama.cpp mainline's "dflash" layout (503,889,952 bytes; the old
// padded-vocab layout is still in the repo as dspark-Inkling-2-Mini-Q8_0_Old.gguf, 402,278,560 bytes). Same drafter (3 blocks, 2304 wide, block 8,
// taps 5 / 23 / 41 / 48, Markov head) written differently: fuse_weight and the per-block dflare_norm split into per-block ctx_mix / ctx_norm,
// the Markov head's hidden_out projection folded through the output head into markov_w2 (+ bias) over the real 200058-token vocabulary,
// mask_embd with 7 rows. The engines look the old names up (dspark.markov_head.hidden_out.weight ...) and stopped the whole start with
// "missing tensor 'dspark.markov_head.hidden_out.weight'".
//
// ink_dspark_layout_check() tells, from the plain tensor names a loader collected, whether the layout is one the runners read. A loader that
// gets false drops the drafter (drafting off for this start), prints ink_dspark_layout_message() once and starts the model normally.
#pragma once

#include <set>
#include <string>

namespace strata::core {

inline bool ink_dspark_layout_check(const std::set<std::string>& plain_names, std::string* why = nullptr) {
    static const char* const kRequired[] = {
        "fuse_weight", "mask_embed", "markov_head.hidden_out.weight", "markov_head.hidden_out.bias", "markov_head.dflare_norm.weight",
        "markov_head.markov_w1.weight", "layers.0.self_attn.q_proj.weight", "layers.0.dflare_norm.weight"};
    for (const char* r : kRequired) {
        if (!plain_names.count(r)) {
            if (why) {
                const bool mainline = plain_names.count("markov_w2.weight") || plain_names.count("blk.0.ctx_mix.weight") || plain_names.count("markov_ctx_mix.weight");
                *why = std::string("missing '") + r + "'" + (mainline ? " (the llama.cpp mainline dflash layout: ctx_mix / markov_w2)" : "");
            }
            return false;
        }
    }
    return true;
}

inline std::string ink_dspark_layout_message(const std::string& file, const std::string& why) {
    return "strata-inkling: the drafter file " + file + " is in a layout this engine cannot read yet (" + why +
           "); drafting is off for this start and the model starts normally. A drafter this engine reads: Unsloth's "
           "dspark-Inkling-2-Mini-Q8_0_Old.gguf, or the original mtp.safetensors.";
}

}  // namespace strata::core
