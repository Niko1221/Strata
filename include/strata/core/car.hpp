// include/strata/core/car.hpp - cache-aware routing: the decode path's expert substitution decision.
//
// **WHAT THIS IS, AND WHY IT IS A PURE FUNCTION.**  The router picks ten experts per layer per token.  The
// VRAM expert cache holds a small part of each layer (on a 12 GiB card, ~21 of 512), so most picks are
// computed by the CPU pool or fetched over PCIe - per-expert costs that a substitution deletes.  Cache-aware
// routing replaces an uncached pick `E` with the best *resident* expert `C` that this token did not already
// select, when the router's own score ratio `p_C / p_E` clears a threshold.  The substituted expert is a
// resident hit: the GPU computes it from a slot that is already in VRAM.
//
// The decision is kept here, apart from the engine, for the reason `expert_cache.hpp` gives about the cache
// itself: a rule that decides what the model computes has to be checkable on its own, on the host, with no
// GPU and no session.  `tests/core/car_substitute_test.cpp` is the authority on the rules below.
//
// **THE RATIO IS EXACT, AND THE WEIGHT IS NOT TOUCHED (IN THIS INCREMENT).**  Both the engine's router
// weights and this test are softmaxes of the SAME logit row, and softmax is monotone and normalized per
// token, so `p_C / p_E = exp(l_C - l_E)` - the ratio is computable from raw logits with no softmax pass and
// no renormalization.  A substituted entry keeps the router's own weight for that position: the engine
// combines `w_E * (the substitute's FFN output)`.  That is a documented, measurable rule (the ratio test
// bounds `w_E` by `w_C / threshold`), and it needs no device-side weight patch.  Dampening the weight by the
// ratio - fomoe's `car.h`/`car.c` `skip_renorm` mode - is planned as a second increment and is refused by
// the engine rather than silently ignored (`--car-dampen`).
#pragma once

#include <cstdint>
#include <vector>

namespace strata::core::car {

/// `kNotResident` is the cache's own -1 (`expert_cache.hpp`); the residency row passed in is the same table
/// the tier decision already uses, so there is one definition of "resident" in the process.
inline constexpr int32_t kNotResident = -1;

struct Config {
    /// 1.0 = OFF: no candidate's ratio can be >= 1.0 except by rounding, and the engine refuses to run the
    /// scan at all when this is 1.0.  Values above 1.0 are treated as "off" too.
    float threshold = 1.0f;
    /// CAR is force-off for the session's first `warmup_tokens` tokens (token index < this).  A cache seeded
    /// from the profile is already warm; this is for a run that starts cold, where a substitution would be
    /// decided against a cache that holds almost nothing the router wants.
    int64_t warmup_tokens = 0;
    /// Maximum substitutions per TOKEN (0 = unlimited).  A quality budget: fomoe's `car.h` carries the same
    /// knob (`max_subs_per_layer`/budget pool) because a window that substitutes every miss is a window whose
    /// hidden state has drifted.
    int64_t budget_per_token = 0;
    /// Substitutions whose ratio is at least this are "free" and do not spend the budget (0 = none are free).
    float free_ratio = 0.0f;
};

/// Everything the engine reports about the scan.  Counted per call and accumulated by the caller.
struct Stats {
    int64_t groups = 0;        ///< calls (one per layer per window)
    int64_t entries = 0;       ///< entries considered = n_tok * k
    int64_t misses = 0;        ///< entries whose expert was not resident (the substitutable set)
    int64_t with_candidate = 0;///< misses that had at least one resident candidate at all
    int64_t substitutions = 0;
    int64_t free_subs = 0;
    int64_t budget_skipped = 0;///< misses that cleared the ratio but had no budget left
    int64_t warmup_skipped = 0;///< entries left alone because the session is inside the warmup window
    double ratio_sum = 0.0;    ///< sum of the ratios of the accepted substitutions
    void accum(const Stats& o);
    void reset();
    /// Mean accepted ratio, or 0 when nothing was substituted.
    double mean_ratio() const { return substitutions > 0 ? ratio_sum / (double) substitutions : 0.0; }
};

/// **CALLER-OWNED, RESIZED, NEVER ALLOCATED ON THE TOKEN PATH.**  The engine reuses one of these per session
/// (`SessionLoopScratch` exists for the same reason); a per-layer allocation on the verify path would be a
/// driver call inside the window.
struct Scratch {
    /// [n_tok * k] the best ratio found per entry, or -1 when none was found.  Doubles as the sort key.
    std::vector<float> ratio;
    /// [n_tok * k] the chosen substitute per entry, -1 when none.
    std::vector<int32_t> cand;
    /// [n_tok * k] entry indices ordered by decreasing ratio (only the ones that have a candidate).
    std::vector<int32_t> order;
    /// [n_tok][n_expert] "already used by this token" marks: the token's own picks, and every substitute
    /// accepted for it.  Cleared per call, never reallocated.
    std::vector<uint8_t> used;
    /// [n_tok] substitutions accepted so far for each token (the `budget_per_token` counter).
    std::vector<int32_t> per_token;
    void reserve(int64_t n_tok, int64_t k, int64_t n_expert);
};

/// The threshold in force for a session token index: `cfg.threshold`, or 1.0 (off) inside the warmup window.
/// Exposed so the engine's warmup handling is testable without a GPU.
float effective_threshold(const Config& cfg, int64_t token_index);

/// **THE DECISION FOR ONE GROUP (ONE LAYER, ONE WINDOW).**  `ids` is modified in place: an accepted
/// substitution writes the substitute's id over the picked one, so every consumer that reads this array
/// afterwards - the CPU pool's miss list, the GPU plan, the tier split - sees one list, not two.
///
/// - `ids`      [n_tok * k]  the routed ids; entries are >= 0 and < n_expert (an out-of-range id is left
///                           alone and never becomes a candidate).
/// - `logits`   [n_tok * n_expert] the router's raw scores, row-major (row t = token t).
/// - `res`      [n_expert]   this layer's residency row: `res[e]` is the expert's slot, or kNotResident.
/// - `out_ratio`[n_tok * k]  the accepted ratio per entry (1.0 where nothing was substituted); may be null.
///
/// Returns the number of substitutions.  Deterministic: candidates are the resident, non-selected experts in
/// increasing id order (a tie in logits goes to the lower id), and entries are served best-ratio-first with a
/// (token, entry) tie-break, so the same input always produces the same output.
int64_t substitute(const Config& cfg, int64_t token_index, int64_t n_tok, int64_t k, int64_t n_expert,
                   int32_t* ids, const float* logits, const int32_t* res, float* out_ratio, Scratch& scratch,
                   Stats& stats);

}  // namespace strata::core::car
