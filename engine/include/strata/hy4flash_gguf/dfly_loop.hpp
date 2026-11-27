#pragma once
// Run DFly proposal, target verification and rollback against one shared target cache.
#include "strata/hy4flash_gguf/dfly.hpp"
#include "strata/hy4flash_gguf/dfly_policy.hpp"
#include "strata/hy4flash_gguf/forward.hpp"
#include "strata/hy4flash_gguf/sampler.hpp"

#include <cstdint>
#include <functional>
#include <map>
#include <string>
#include <utility>
#include <vector>

namespace strata::hy4_flash {

struct Geometry;
struct KvCache;
struct DenseResidency;

struct DflyStats {
    // Rounds with kv > 0 drafts verified whose verify and rewind completed (the acceptance rate's definition).
    long rounds = 0, offered = 0, accepted = 0, round_outputs = 0;
    long drafted = 0;      ///< drafts proposed; more than `offered` when the confidence gate cut a verify
    long gated = 0;        ///< rounds whose verify the confidence gate cut below the drafts (kv < k)
    long ar_steps = 0;     ///< one-row verifies: k = 0 by budget, by the engaged boundary or after a
                           ///< fallback, or kv = 0 by the window
    long fallbacks = 0;    ///< proposals that came back safe_fallback
    long kv0_chosen = 0;   ///< of `ar_steps`: k > 0 drafts were made and the window verified none
    std::vector<long> k_rounds;                ///< committed rounds by the kv they verified (0..kMaxK)
    std::vector<long> reach, hit;              ///< per draft position: compared, matched
    std::map<std::pair<int, int>, long> kr;    ///< (kv, r) of each counted round
    double prefill_ms = 0, decode_ms = 0, propose_ms = 0, verify_ms = 0;
};

enum class DflyFinish { stop, length, nonfinite, cancel, error, restart_worker };
const char* dfly_finish_name(DflyFinish f);

/// What one process keeps across requests: the drafter and the two capture buffers.
struct DflyRuntime {
    DflyDrafter drafter;
    DflyCapture tail;      ///< one row, the prompt's last position
    DflyCapture win;       ///< up to kMaxK + 1 rows, a verify window
    /// The confidence window's measured cost of a round minus its proposal, by kv. It describes the card, so
    /// requests share it.
    DflyWindowCosts verify_costs;
    /// false after a failure (`dfly_generate` reset the target to 0, or the worker must restart). A request
    /// then needs `cache.len == 0`; a caller that has put the cache back into a known state some other way
    /// (serve restoring a parked conversation whole) may set it true again.
    bool trusted = true;
};
/// Allocate the two captures (every sparse block's selection is kept; the drafter picks its five by block id).
bool dfly_runtime_init(DflyRuntime& rt, const Geometry& g, std::string& err);

struct DflyRequest {
    const std::vector<int32_t>* prompt = nullptr;
    int max_new = 1;
    int k_max = DflyDrafter::kMaxK;
    /// OPT-IN: how many of the k drafts a round verifies (dfly_policy.hpp). The output is the same for every
    /// choice; only the speed differs. fixed: all of them, every round.
    DflyWindow window = DflyWindow::fixed;
    const std::vector<int32_t>* stop_tokens = nullptr;
    /// The plain path's sampler settings, any of them. Used to build a host `Sampler` when `pick` is empty.
    SamplerConfig sampler;
    /// Sample the first token from the prompt's last row and subsequent tokens from verification rows.
    std::function<int64_t(const float* row, const std::vector<int32_t>& history)> pick;
    /// The rows `pick` and `emit` are given live on the device (`ForwardOptions::device_logits_out`) and
    /// are valid until the next pass; false: host copies.
    bool device_logits = false;
    /// The final prompt row supplies the first token's logits.
    bool tail_last_row_only = false;
    std::string draft_path;                                      ///< loaded and bound after the first prefill
    /// One committed token and the target logits row it was picked from (see `device_logits`). false: the
    /// caller wants to stop.
    std::function<bool(int32_t, const float*)> emit;
    /// Checked between prompt chunks and at each commit boundary; true ends the request with `cancel`.
    std::function<bool()> should_stop;
    /// After each whole prompt chunk before the last: the positions now in the cache and the prompt's length.
    std::function<void(int, int)> on_prompt_chunk;
};

/// Reuse only a pool-aligned prefix shorter than the prompt. Replay the final prompt token to obtain its
/// prediction.
DflyFinish dfly_generate(const std::string& gguf, const Geometry& g, const std::string& pack,
                         const ForwardOptions& fo, DenseResidency& dense, KvCache& cache, DflyRuntime& rt,
                         const DflyRequest& req, std::vector<int32_t>& produced, DflyStats& st, std::string& err);

}  // namespace strata::hy4_flash
