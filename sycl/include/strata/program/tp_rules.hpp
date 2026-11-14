// include/strata/program/tp_rules.hpp - when a tensor-parallel run may start, as rules that need no GPU.
//
// The layer split's own refusals live inline in `generate.cpp` because each one needs the values that were
// parsed; these do not.  Every rule here is a property of the FLAGS, so it can be tested without a model, a
// card or a command line - and the engine's job is only to print the returned line and continue on one card.
//
// WHY EACH RULE EXISTS, in one line each (the message says the same thing to the user):
//
//   * TWO RANKS.  `n_head_kv` is 2 (`core/geometry.hpp`), so a third rank has no KV head to own and would
//     have to replicate one - the shape of split that kvmem measured going wrong.
//   * `--serve` is where the layer split already lives (`generate.cpp:2376`): the split needs the window loop,
//     and TP needs the same loop with both ranks in lockstep.
//   * NOT with `--layer-split`: the two are different ways to spend a second card - one cuts the layers, the
//     other cuts every layer - and running both is a four-card shape this engine does not have.
//   * ONE BACKEND.  The collective is staged through mapped pinned memory and both ranks must be able to see
//     it; a CUDA/HIP pair is not a pair this code has ever run on.
//   * COMPUTE CAPABILITY 7.5.  The engine's own rule, and TP inherits it: pre-Turing kernels take a different
//     path that has never been run in a two-rank window.
//   * SYMMETRY.  The kernels wait for each other on an arrival token, so both ranks must execute the SAME
//     number of reduces in the SAME order in every window.  `--batch` holds independent sequences (their row
//     counts differ per slot), `--pipeline-windows` runs two windows per card at once, and
//     `--peer-device`/`--expert-cache-remote` let one card compute another's work.  Each of those makes the
//     two ranks' call sequences differ, and a rank that skips a call leaves its peer spinning.
//   * NOT YET, and the message says so rather than pretending: session files (a split stage's state is not
//     saved - `conversation_file.cpp:1032` already refuses a layer split for the same reason), `--vision`
//     (per-card image tables), `--mmap-experts` (a per-card view of the file), `--kv-resident` (each card
//     streams its own session's KV) and `--prompt-cache-every` (each card saves its own part).
#pragma once

#include <string>

namespace strata::program::tp_rules {

/// The smallest compute capability the engine runs on at all (`CMakeLists.txt:178-182`).
inline constexpr int kMinCcMajor = 7;
inline constexpr int kMinCcMinor = 5;

inline bool cc_at_least(int major, int minor) {
    return major > kMinCcMajor || (major == kMinCcMajor && minor >= kMinCcMinor);
}

/// Every flag that decides whether TP may start.  One field per flag the engine already has.
struct Options {
    int tp = 0;                      ///< `--tp N`; 0 = not asked for
    int gpus = 1;                    ///< how many cards this run was given
    /// `--tp-device D`: the second rank's card.  -1 = not given (the next visible one); 0 = both ranks on one
    /// card: the A/B of the split with no second card's VRAM.  NOT bit-exact - a contraction split
    /// reassociates the sum, so the cut tensors agree to fp32 accumulation error and nothing else changes.
    /// Anything at or past `gpus` names no card this run has.
    int second_device = -1;
    bool serve = false;              ///< running under `--serve`
    bool layer_split = false;        ///< `--layer-split` (or a config's `"layer_split"`)
    bool same_backend = true;        ///< every chosen card is the same backend
    int weakest_cc_major = 8;        ///< the weakest chosen card's compute capability
    int weakest_cc_minor = 6;
    bool batch = false;              ///< `--batch` / `--batch-groups`
    bool pipeline_windows = false;   ///< `--pipeline-windows` (non-zero)
    bool peer_device = false;        ///< `--peer-device`
    bool remote_experts = false;     ///< `--expert-cache-remote*` / `--remote-expert-opt`
    bool session_files = false;      ///< a conversation/session file was asked for
    bool vision = false;             ///< `--vision`
    bool mmap_experts = false;       ///< `--mmap-experts`
    bool kv_resident = false;        ///< `--kv-resident`
    bool prompt_cache_every = false; ///< `--prompt-cache-every`
};

/// "" when a tensor-parallel run may start, else ONE line naming what to change.  The engine prints it
/// verbatim and then continues with one card, which is what makes this feature opt-in rather than a mode that
/// fails to start.
inline std::string refusal(const Options& o) {
    if (o.tp <= 0) return "";   // not asked for: nothing to refuse, and the default path is untouched
    if (o.tp < 2 || o.tp > 4) {
        return "--tp is 2, 3 or 4 on this engine: the shard table splits query/GDN heads across the ranks and "
               "replicates the KV heads when there are fewer of them than ranks";
    }
    if (!o.serve) return "--tp needs --serve (it is the same window loop the layer split runs in)";
    if (o.gpus < o.tp) {
        return "--tp " + std::to_string(o.tp) + " needs " + std::to_string(o.tp) +
               " cards, and this run was given " + std::to_string(o.gpus);
    }
    // A flag nobody reads is worse than no flag: `--tp-device` used to be parsed and then ignored, so a typo
    // like `--tp-device 5` on a two-card run looked accepted and changed nothing.
    if (o.second_device >= o.gpus) {
        return "--tp-device names card " + std::to_string(o.second_device) + ", and this run was given " +
               std::to_string(o.gpus) + " cards";
    }
    if (o.layer_split) {
        return "--tp and --layer-split are two different ways to use a second card; pick one "
               "(see docs/MULTI_GPU.md)";
    }
    if (!o.same_backend) {
        return "--tp needs both cards on one backend (two CUDA or two HIP): the collective stages through "
               "mapped pinned memory that each backend sees differently";
    }
    if (!cc_at_least(o.weakest_cc_major, o.weakest_cc_minor)) {
        return "--tp needs compute capability 7.5 or newer on EVERY card, and the weakest one is " +
               std::to_string(o.weakest_cc_major) + "." + std::to_string(o.weakest_cc_minor);
    }
    // NO FEATURE IS REFUSED FOR ASKING FOR TENSOR PARALLELISM, and that is a requirement rather than an
    // omission.  Both ranks run the SAME window on the same tokens, so everything else works exactly as it does
    // on one card: a batch window's slots are the same rows on both ranks, images go through the same vision
    // path, each rank streams its own share of the KV, and a session file saves what each rank holds.  This list
    // used to refuse `--batch`, `--pipeline-windows`, `--peer-device`, the remote caches, `--vision`,
    // `--mmap-experts`, `--kv-resident` and the mid-prompt cache - which was a symptom of a design where the two
    // ranks could drift apart, not a property of tensor parallelism.  llama.cpp's tensor parallelism refuses
    // none of it, and neither does this one.
    return "";
}

}  // namespace strata::program::tp_rules
