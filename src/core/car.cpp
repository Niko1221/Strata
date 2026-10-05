// src/core/car.cpp - the cache-aware routing decision.  The rules are in `include/strata/core/car.hpp`.
//
// **THE TWO THINGS THAT MAKE THIS SAFE TO RUN ON A TOKEN PATH.**  It allocates nothing (the caller's
// `Scratch` is resized once) and it is deterministic (best ratio first, ties by lower expert id and then by
// (token, entry)), so the same window always makes the same decision and a test can pin it.
#include "strata/core/car.hpp"

#include <algorithm>
#include <cmath>
#include <cfloat>

namespace strata::core::car {

void Stats::accum(const Stats& o) {
    groups += o.groups;
    entries += o.entries;
    misses += o.misses;
    with_candidate += o.with_candidate;
    substitutions += o.substitutions;
    free_subs += o.free_subs;
    budget_skipped += o.budget_skipped;
    warmup_skipped += o.warmup_skipped;
    ratio_sum += o.ratio_sum;
}

void Stats::reset() { *this = Stats{}; }

void Scratch::reserve(int64_t n_tok, int64_t k, int64_t n_expert) {
    const size_t n = (size_t) (n_tok > 0 ? n_tok : 1) * (size_t) (k > 0 ? k : 1);
    if (ratio.size() < n) ratio.resize(n);
    if (cand.size() < n) cand.resize(n);
    if (order.size() < n) order.resize(n);
    const size_t m = (size_t) (n_tok > 0 ? n_tok : 1) * (size_t) (n_expert > 0 ? n_expert : 1);
    if (used.size() < m) used.resize(m);
    if (per_token.size() < (size_t) (n_tok > 0 ? n_tok : 1)) per_token.resize((size_t) (n_tok > 0 ? n_tok : 1));
}

float effective_threshold(const Config& cfg, int64_t token_index) {
    if (token_index < cfg.warmup_tokens) return 1.0f;              // warmup: off
    if (!(cfg.threshold < 1.0f)) return 1.0f;                       // 1.0 (or above, or NaN): off
    return cfg.threshold;
}

namespace {

/// `p_C / p_E` from raw logits, and it is exact for the engine's own weights: both are softmaxes of this row,
/// so the normalization cancels.  Clamped at both ends because `exp` of a float difference can overflow a
/// float, and an `inf` in the stats' sum would poison every report that includes it.
float ratio_of(double l_cand, double l_orig) {
    const double d = l_cand - l_orig;
    if (!(d == d)) return 0.0f;                    // NaN logits: no candidacy
    if (d > 88.0) return 3.0e38f;                  // saturate rather than produce inf
    if (d < -88.0) return 0.0f;
    return (float) std::exp(d);
}

bool finite(float v) { return std::isfinite(v); }

}  // namespace

int64_t substitute(const Config& cfg, int64_t token_index, int64_t n_tok, int64_t k, int64_t n_expert,
                   int32_t* ids, const float* logits, const int32_t* res, float* out_ratio, Scratch& scratch,
                   Stats& stats) {
    if (n_tok <= 0 || k <= 0 || n_expert <= 0 || ids == nullptr) return 0;

    ++stats.groups;
    stats.entries += n_tok * k;
    if (out_ratio != nullptr)
        for (int64_t i = 0; i < n_tok * k; ++i) out_ratio[i] = 1.0f;

    const float tau = effective_threshold(cfg, token_index);
    const bool warmup_off = token_index < cfg.warmup_tokens;
    if (ids == nullptr || res == nullptr || logits == nullptr || warmup_off || !(cfg.threshold < 1.0f)) {
        // Still count what was there: "how many misses did this window have" is worth reporting even when the
        // scan does not run, and a test asserts that nothing was written.
        if (res != nullptr)
            for (int64_t i = 0; i < n_tok * k; ++i) {
                const int32_t e = ids[i];
                if (e >= 0 && e < n_expert && res[e] == kNotResident) ++stats.misses;
            }
        if (warmup_off) stats.warmup_skipped += stats.misses;   // the warmup window, not the threshold, held it off
        (void) tau;
        return 0;
    }

    scratch.reserve(n_tok, k, n_expert);
    int32_t* const cand = scratch.cand.data();
    float* const ratio = scratch.ratio.data();
    // The per-token "already used" mask spans the whole window: a substitute chosen for one entry must not be
    // chosen for another entry of the SAME token, and the acceptance pass below is where that is enforced.
    uint8_t* const used = scratch.used.data();
    std::fill(used, used + (size_t) n_tok * (size_t) n_expert, (uint8_t) 0);
    int32_t* const per_token = scratch.per_token.data();
    for (int64_t t = 0; t < n_tok; ++t) per_token[t] = 0;

    // ---- pass 1: for every miss, this token's best resident expert that the token did not select.
    for (int64_t t = 0; t < n_tok; ++t) {
        uint8_t* const tu = used + (size_t) t * (size_t) n_expert;
        const float* const row = logits + (size_t) t * (size_t) n_expert;
        for (int64_t j = 0; j < k; ++j) {
            const int32_t e = ids[t * k + j];
            if (e >= 0 && e < n_expert) tu[e] = 1;         // selected by this token (or already a substitute)
        }
        for (int64_t j = 0; j < k; ++j) {
            const int64_t i = t * k + j;
            cand[i] = -1;
            ratio[i] = -1.0f;
            const int32_t e = ids[i];
            if (e < 0 || e >= n_expert) continue;          // out of range: the pool refuses it, not CAR
            if (res[e] != kNotResident) continue;          // resident: nothing to substitute
            ++stats.misses;
            if (!finite(row[e])) continue;
            int32_t best = -1;
            float best_l = 0.0f;
            for (int64_t c = 0; c < n_expert; ++c) {       // increasing id: an exact tie keeps the lower id
                if (res[c] == kNotResident) continue;
                if (tu[c]) continue;                       // already selected by this token
                const float lc = row[c];
                if (!finite(lc)) continue;
                if (best < 0 || lc > best_l) { best = (int32_t) c; best_l = lc; }
            }
            if (best < 0) continue;
            ++stats.with_candidate;
            cand[i] = best;
            ratio[i] = ratio_of((double) best_l, (double) row[e]);
        }
    }

    // ---- pass 2: serve the best ratios first, under the threshold and the per-token budget.
    int32_t* const order = scratch.order.data();
    int64_t n_ord = 0;
    for (int64_t i = 0; i < n_tok * k; ++i)
        if (cand[i] >= 0) order[n_ord++] = (int32_t) i;
    std::stable_sort(order, order + n_ord, [&](int32_t a, int32_t b) {
        if (ratio[a] != ratio[b]) return ratio[a] > ratio[b];
        return a < b;                                      // (token, entry): the input order
    });

    int64_t subs = 0;
    for (int64_t o = 0; o < n_ord; ++o) {
        const int64_t i = order[o];
        const int64_t t = i / k;
        uint8_t* const tu = used + (size_t) t * (size_t) n_expert;
        int32_t c = cand[i];
        if (tu[c]) {
            // The best candidate went to another entry of this token.  Re-scan for this entry with the mask as
            // it stands now; if the next-best no longer clears the threshold, this entry stays a miss.
            const int32_t e = ids[i];
            const float* const row = logits + (size_t) t * (size_t) n_expert;
            int32_t best = -1;
            float best_l = 0.0f;
            for (int64_t q = 0; q < n_expert; ++q) {
                if (res[q] == kNotResident || tu[q]) continue;
                const float lq = row[q];
                if (!finite(lq)) continue;
                if (best < 0 || lq > best_l) { best = (int32_t) q; best_l = lq; }
            }
            if (best < 0) continue;
            c = best;
            cand[i] = c;
            ratio[i] = ratio_of((double) best_l, (double) row[e]);
        }
        if (!(ratio[i] >= tau)) continue;                  // below the threshold (or NaN): a miss stays a miss
        const bool is_free = cfg.free_ratio > 0.0f && ratio[i] >= cfg.free_ratio;
        if (cfg.budget_per_token > 0 && per_token[t] >= cfg.budget_per_token && !is_free) {
            ++stats.budget_skipped;
            continue;
        }
        ids[i] = c;                                        // in place: ONE list, every consumer sees it
        tu[c] = 1;                                         // and this token cannot use it twice
        // A free substitution does NOT spend the budget - that is what "free" means (`car.hpp`), and the
        // counter below is the budget's, not a count of everything substituted.
        if (is_free) ++stats.free_subs;
        else ++per_token[t];
        if (out_ratio != nullptr) out_ratio[i] = ratio[i];
        ++subs;
        ++stats.substitutions;
        stats.ratio_sum += (double) ratio[i];
    }
    return subs;
}

}  // namespace strata::core::car
