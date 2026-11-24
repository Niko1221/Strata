// Serve requests over the native engine protocol.
#pragma once

#include "strata/hy4flash_gguf/dfly_policy.hpp"
#include "strata/hy4flash_gguf/forward.hpp"
#include "strata/hy4flash_gguf/geometry.hpp"
#include "strata/hy4flash_gguf/kv_cache.hpp"
#include "strata/hy4flash_gguf/sampler.hpp"

#include <cstdint>
#include <string>
#include <vector>

namespace strata::hy4_flash {
class Mtp;
struct DflyRuntime;

/// Everything the loop needs from the engine's own setup, borrowed. The loop owns none of it.
struct ServeState {
    const std::string* gguf = nullptr;
    const std::string* pack = nullptr;
    const Geometry* g = nullptr;
    KvCache* cache = nullptr;
    ForwardOptions* fo = nullptr;          ///< the loop sets `history`, `want_logits` and `logits_last_only`
    std::string vision_shard;
    int max_context = 0;                   ///< the cache's reserved positions; a request past it gets ERR
    std::int64_t vocab = 0;
    SamplerConfig sampler;                 ///< the command line's, which a GEN line's keys may override
    /// The ids that end a turn, already resolved by the engine (the file's metadata, `--stop-ids`,
    /// `--ignore-eos`). Empty means a request runs to its `max_new` whatever the model produces - which is
    /// what this loop did before it had them, and why `READY <ctx> stop` was a promise it could not keep.
    std::vector<std::int32_t> stop_tokens;
    Mtp* mtp = nullptr;
    int draft_tokens = 3;
    /// `--draft dfly`: every request, greedy or sampled, runs through `dfly_generate` (dfly_loop.hpp) with
    /// this runtime, its drafter already loaded. Null: off. Never together with `mtp`.
    DflyRuntime* dfly = nullptr;
    const std::string* dfly_model = nullptr;
    int dfly_k_max = 3;
    DflyWindow dfly_window = DflyWindow::fixed;   ///< `--draft-window`: how much of the window a round uses
    int conversation_cache_mib = 0;
    int conversation_cache_slots = 4;
};

/// Read requests until QUIT or end of input. Returns the process's exit code.
int serve_loop(const ServeState& s);

}  // namespace strata::hy4_flash
