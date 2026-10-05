// car_substitute_test - the cache-aware routing decision's rules, on the host, with no GPU.
//
// **THIS IS THE AUTHORITY ON WHAT CAR DOES.**  The engine wires the function into a path that produces
// tokens, so a wrong rule here is a wrong token there, and the failure mode this project keeps paying for is a
// plausible token that is not the model's.  Three layers of coverage:
//
//   1. named cases, one per rule in `car.hpp`, with hand-computed expectations and the negative assertions
//      (nothing written when the threshold is off, no duplicate expert within a token, out-of-range ids
//      untouched);
//   2. a **differential** test: an independent reference implementation of the documented rule, compared
//      against the engine's on thousands of randomized windows (deterministic PRNG), including ties, NaN
//      logits, saturated ratios, every k the engine uses and a 512-expert/21-resident layer;
//   3. `--emit-golden PATH`, one window written for `tools/test_car_estimate.py`, so the offline estimator's
//      copy of the rule is checked against the engine's too.
#include "strata/core/car.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

using strata::core::car::Config;
using strata::core::car::Scratch;
using strata::core::car::Stats;
using strata::core::car::effective_threshold;
using strata::core::car::substitute;

namespace {

int g_failures = 0;

void check(bool ok, const char* what) {
    if (!ok) {
        std::fprintf(stderr, "FAIL: %s\n", what);
        ++g_failures;
    }
}

constexpr int32_t kMiss = strata::core::car::kNotResident;

std::vector<int32_t> residency(int64_t n_expert, const std::vector<int32_t>& resident) {
    std::vector<int32_t> res((size_t) n_expert, kMiss);
    for (size_t i = 0; i < resident.size(); ++i) res[(size_t) resident[i]] = (int32_t) i;
    return res;
}

/// A logit row: `n_expert` values from a sparse list; everything else sits far below the rest, so a case only
/// has to state the experts it means.
std::vector<float> logits_row(int64_t n_expert, const std::vector<std::pair<int32_t, float>>& v, float fill = -50.0f) {
    std::vector<float> row((size_t) n_expert, fill);
    for (const auto& p : v) row[(size_t) p.first] = p.second;
    return row;
}

/// `p_C / p_E` with the same clamping the implementation documents (the test's own arithmetic, deliberately
/// written from the header's formula rather than copied from the source).
float ratio_ref(double l_cand, double l_orig) {
    const double d = l_cand - l_orig;
    if (!(d == d)) return 0.0f;
    if (d > 88.0) return 3.0e38f;
    if (d < -88.0) return 0.0f;
    return (float) std::exp(d);
}

/// **THE REFERENCE IMPLEMENTATION.**  Written from `car.hpp`'s rules, not from `car.cpp`'s structures: it keeps
/// no scratch, sorts a copy, and rescans the whole layer each time.  If the two disagree anywhere in the
/// randomized sweep, one of them is wrong and the cases above say which rule is at stake.
struct RefResult {
    std::vector<int32_t> ids;
    Stats stats;
};

RefResult reference(const Config& cfg, int64_t token_index, int64_t n_tok, int64_t k, int64_t n_expert,
                    std::vector<int32_t> ids, const std::vector<float>& logits, const std::vector<int32_t>& res) {
    RefResult out;
    Stats s;
    s.groups = 1;
    s.entries = n_tok * k;
    const float tau = effective_threshold(cfg, token_index);
    const bool warmup_off = token_index < cfg.warmup_tokens;
    if (warmup_off || !(cfg.threshold < 1.0f)) {
        for (int64_t i = 0; i < n_tok * k; ++i) {
            const int32_t e = ids[i];
            if (e >= 0 && e < n_expert && res[(size_t) e] == kMiss) ++s.misses;
        }
        if (warmup_off) s.warmup_skipped = s.misses;
        out.ids = ids;
        out.stats = s;
        return out;
    }

    std::vector<uint8_t> used((size_t) n_tok * (size_t) n_expert, 0);
    std::vector<float> ratio((size_t) n_tok * (size_t) k, -1.0f);
    std::vector<int32_t> cand((size_t) n_tok * (size_t) k, -1);
    std::vector<int32_t> per_token((size_t) n_tok, 0);

    auto best_for = [&](int64_t t, int64_t i) -> int32_t {
        const int32_t e = ids[i];
        if (e < 0 || e >= n_expert) return -1;
        const float* row = logits.data() + (size_t) t * (size_t) n_expert;
        if (!std::isfinite(row[e])) return -1;
        int32_t best = -1;
        float best_l = 0.0f;
        for (int64_t c = 0; c < n_expert; ++c) {
            if (res[(size_t) c] == kMiss) continue;
            if (used[(size_t) t * (size_t) n_expert + (size_t) c]) continue;
            const float lc = row[c];
            if (!std::isfinite(lc)) continue;
            if (best < 0 || lc > best_l) { best = (int32_t) c; best_l = lc; }
        }
        if (best >= 0) ratio[(size_t) i] = ratio_ref((double) best_l, (double) row[e]);
        return best;
    };

    for (int64_t t = 0; t < n_tok; ++t) {
        for (int64_t j = 0; j < k; ++j) {
            const int32_t e = ids[t * k + j];
            if (e >= 0 && e < n_expert) used[(size_t) t * (size_t) n_expert + (size_t) e] = 1;
        }
        for (int64_t j = 0; j < k; ++j) {
            const int64_t i = t * k + j;
            const int32_t e = ids[i];
            if (e < 0 || e >= n_expert) continue;
            if (res[(size_t) e] != kMiss) continue;
            ++s.misses;
            cand[(size_t) i] = best_for(t, i);
            if (cand[(size_t) i] >= 0) ++s.with_candidate;
        }
    }

    std::vector<int32_t> order;
    for (int64_t i = 0; i < n_tok * k; ++i)
        if (cand[(size_t) i] >= 0) order.push_back((int32_t) i);
    std::stable_sort(order.begin(), order.end(), [&](int32_t a, int32_t b) {
        if (ratio[(size_t) a] != ratio[(size_t) b]) return ratio[(size_t) a] > ratio[(size_t) b];
        return a < b;
    });

    for (int32_t i : order) {
        const int64_t t = i / k;
        const int32_t e = ids[(size_t) i];
        if (used[(size_t) t * (size_t) n_expert + (size_t) cand[(size_t) i]]) {
            const int32_t again = best_for(t, i);
            if (again < 0) continue;
            cand[(size_t) i] = again;
        }
        if (!(ratio[(size_t) i] >= tau)) continue;
        const bool is_free = cfg.free_ratio > 0.0f && ratio[(size_t) i] >= cfg.free_ratio;
        if (cfg.budget_per_token > 0 && per_token[(size_t) t] >= cfg.budget_per_token && !is_free) {
            ++s.budget_skipped;
            continue;
        }
        ids[(size_t) i] = cand[(size_t) i];
        used[(size_t) t * (size_t) n_expert + (size_t) cand[(size_t) i]] = 1;
        if (is_free) ++s.free_subs;
        else ++per_token[(size_t) t];
        ++s.substitutions;
        s.ratio_sum += (double) ratio[(size_t) i];
    }
    out.ids = ids;
    out.stats = s;
    return out;
}

/// What a run produced, so a case reads as one line.
struct Run {
    std::vector<int32_t> ids;
    std::vector<float> ratio;
    Stats stats;
    int64_t subs = 0;
};

Run run(const Config& cfg, int64_t token_index, int64_t n_tok, int64_t k, const std::vector<int32_t>& ids_in,
        const std::vector<float>& logits, const std::vector<int32_t>& res_in) {
    Run r;
    r.ids = ids_in;
    r.ratio.assign(ids_in.size(), 0.0f);
    static Scratch scratch;  // reused, exactly as the engine does
    r.subs = substitute(cfg, token_index, n_tok, k, (int64_t) res_in.size(), r.ids.data(), logits.data(),
                        res_in.data(), r.ratio.data(), scratch, r.stats);
    return r;
}

bool same_stats(const Stats& a, const Stats& b) {
    return a.groups == b.groups && a.entries == b.entries && a.misses == b.misses &&
           a.with_candidate == b.with_candidate && a.substitutions == b.substitutions &&
           a.free_subs == b.free_subs && a.budget_skipped == b.budget_skipped &&
           a.warmup_skipped == b.warmup_skipped && a.ratio_sum == b.ratio_sum;
}

// =========================================================================================================
// 1. off is off: threshold 1.0 (and above) changes nothing at all
void case_off_is_off() {
    const int64_t NE = 16, K = 4;
    const auto res = residency(NE, {0, 1, 2, 3});
    const auto row = logits_row(NE, {{0, 10.0f}, {1, 9.0f}, {2, 8.0f}, {3, 7.0f},
                                     {12, 8.0f}, {13, 7.0f}, {14, 6.0f}, {15, 5.0f}});
    const std::vector<int32_t> ids{12, 13, 14, 15};

    const Run r = run(Config{}, 100, 1, K, ids, row, res);
    check(r.subs == 0, "threshold 1.0: no substitution");
    check(r.ids == ids, "threshold 1.0: the id list is untouched");
    check(r.stats.substitutions == 0, "threshold 1.0: no substitution counted");
    check(r.stats.misses == 4, "threshold 1.0: the misses are still reported");
    for (float v : r.ratio) check(v == 1.0f, "threshold 1.0: every ratio is 1.0");

    Config above;
    above.threshold = 1.5f;
    const Run r2 = run(above, 100, 1, K, ids, row, res);
    check(r2.subs == 0 && r2.ids == ids, "threshold above 1.0 is off, not the opposite");
}

// 2. the ratio test: a miss whose best resident is scored below it keeps its expert
void case_ratio_threshold() {
    const int64_t NE = 16, K = 2;
    const auto res = residency(NE, {0});
    // resident 0 at 10.0.  Miss 9 is scored 12.0 -> ratio exp(-2) = 0.135 (below 0.35, stays a miss).
    // Miss 4 is scored 9.0      -> ratio exp(1)  = 2.718 (above, substituted by 0).
    const auto row = logits_row(NE, {{0, 10.0f}, {4, 9.0f}, {9, 12.0f}});
    const std::vector<int32_t> ids{9, 4};
    Config cfg;
    cfg.threshold = 0.35f;
    const Run r = run(cfg, 0, 1, K, ids, row, res);
    check(r.subs == 1, "one substitution: exactly the entry whose ratio clears the threshold");
    check(r.ids[0] == 9, "the entry above the threshold keeps its expert");
    check(r.ids[1] == 0, "the entry below it is substituted by the resident expert");
    check(std::fabs(r.stats.mean_ratio() - std::exp(1.0)) < 1e-6, "the reported ratio is exp(1)");
}

// 3. the boundary is ">=", checked against the ratio the function itself computed
void case_boundary_is_ge() {
    const int64_t NE = 16, K = 4;
    // Residents 0..3 at 3.0, 2.0, 1.0, 0.5.  Token picks 1, 2 and 3 (all resident) and 5 (a miss at 3.5),
    // so the miss's best candidate is 0 and its ratio is exp(3.0 - 3.5) = 0.6065 - between 0.5 and 1.
    const auto res = residency(NE, {0, 1, 2, 3});
    const auto row = logits_row(NE, {{0, 3.0f}, {1, 2.0f}, {2, 1.0f}, {3, 0.5f}, {5, 3.5f}});
    const std::vector<int32_t> ids{1, 2, 3, 5};

    Config learn;
    learn.threshold = 0.5f;
    const Run first = run(learn, 0, 1, K, ids, row, res);
    check(first.subs == 1, "the entry is substitutable at 0.5");
    const float r = first.ratio[3];
    check(r > 0.5f && r < 1.0f, "the ratio sits between the threshold and 1");

    Config at;
    at.threshold = r;
    const Run exact = run(at, 0, 1, K, ids, row, res);
    check(exact.subs == 1, "a ratio exactly at the threshold substitutes (>=)");

    Config just_above;
    just_above.threshold = std::nextafterf(r, 2.0f);
    const Run over = run(just_above, 0, 1, K, ids, row, res);
    check(over.subs == 0, "a ratio just below the threshold does not");
}

// 4. no candidate: nothing written, and the misses are still misses
void case_no_candidate() {
    const int64_t NE = 16, K = 4;
    const auto res = residency(NE, {});
    const auto row = logits_row(NE, {{0, 10.0f}, {1, 9.0f}, {2, 8.0f}, {3, 7.0f}});
    const std::vector<int32_t> ids{0, 1, 2, 3};
    Config cfg;
    cfg.threshold = 0.0f;  // the loosest possible setting: even so, there is nothing to substitute
    const Run r = run(cfg, 0, 1, K, ids, row, res);
    check(r.subs == 0 && r.ids == ids, "an empty cache substitutes nothing");
    check(r.stats.misses == 4 && r.stats.with_candidate == 0, "misses without candidates are counted apart");
}

// 5. an expert the token already selected is not a candidate
void case_selected_is_not_a_candidate() {
    const int64_t NE = 8, K = 2;
    const auto res = residency(NE, {0, 1});
    const auto row = logits_row(NE, {{0, 10.0f}, {1, 9.0f}, {2, 9.5f}, {3, 0.0f}});
    const std::vector<int32_t> ids{0, 2};  // 0 is resident AND selected; 2 is a miss
    Config cfg;
    cfg.threshold = 0.0f;
    const Run r = run(cfg, 0, 1, K, ids, row, res);
    check(r.ids[0] == 0, "a resident pick is left alone");
    check(r.ids[1] == 1, "the miss takes the resident expert the token did not select (1, not 0)");
}

// 6. no expert twice in one token, even when it is the best candidate for two misses
void case_no_duplicate_within_token() {
    const int64_t NE = 8, K = 2;
    const auto res = residency(NE, {0, 1});
    // Both misses want resident 0.  Miss 4 is scored 4.5 against 0's 10.0 (ratio exp(5.5)) and miss 5 is
    // scored 5.0 (ratio exp(5)), so miss 4 is served first; miss 5 rescans and takes 1.
    const auto row = logits_row(NE, {{0, 10.0f}, {1, 9.0f}, {4, 4.5f}, {5, 5.0f}});
    const std::vector<int32_t> ids{4, 5};
    Config cfg;
    cfg.threshold = 0.0f;
    const Run r = run(cfg, 0, 1, K, ids, row, res);
    check(r.subs == 2, "two resident experts, two substitutions");
    check(r.ids[0] == 0, "the higher ratio takes the first resident expert");
    check(r.ids[1] == 1, "the second miss rescans and takes the other one");
    check(r.ids[0] != r.ids[1], "no expert appears twice in a token's list");
}

// 7. best ratio first: a lower-ratio entry can lose its candidate to a higher-ratio one
void case_best_ratio_first() {
    const int64_t NE = 8, K = 2;
    const auto res = residency(NE, {0});
    // Both misses would take resident 0.  Miss 4's ratio exp(10-10.5) = 0.607 is the higher one, so it is
    // served first; miss 5 then has no resident left and stays a miss.
    const auto row = logits_row(NE, {{0, 10.0f}, {4, 10.5f}, {5, 11.0f}});
    const std::vector<int32_t> ids{4, 5};
    Config cfg;
    cfg.threshold = 0.35f;
    const Run r = run(cfg, 0, 1, K, ids, row, res);
    check(r.ids[0] == 0, "the higher ratio takes the only resident expert");
    check(r.ids[1] == 5, "the lower ratio rescans, finds nothing above the threshold, and stays a miss");
    check(r.subs == 1, "one substitution");
}

// 8. all resident: nothing to do
void case_all_resident() {
    const int64_t NE = 8, K = 4;
    const auto res = residency(NE, {0, 1, 2, 3, 4, 5, 6, 7});
    const std::vector<float> row((size_t) NE, 1.0f);
    const std::vector<int32_t> ids{0, 1, 2, 3};
    Config cfg;
    cfg.threshold = 0.0f;
    const Run r = run(cfg, 0, 1, K, ids, row, res);
    check(r.subs == 0 && r.ids == ids && r.stats.misses == 0, "a fully resident token has no misses");
}

// 9. the per-token budget, and the free ratio that does not spend it
void case_budget() {
    const int64_t NE = 8, K = 4;
    const auto res = residency(NE, {0, 1, 2});
    const auto row = logits_row(NE, {{0, 10.0f}, {1, 9.9f}, {2, 9.8f}, {4, 5.0f}, {5, 4.9f}, {6, 4.8f}, {7, 4.7f}});
    const std::vector<int32_t> ids{4, 5, 6, 7};

    Config cfg;
    cfg.threshold = 0.0f;
    cfg.budget_per_token = 1;
    const Run r = run(cfg, 0, 1, K, ids, row, res);
    check(r.subs == 1, "a budget of 1 allows exactly one substitution");
    // Every one of the three remaining misses clears the ratio and finds a resident expert (1 is never marked
    // used, because its acceptance was the one the budget refused), so all three are reported as skipped.
    check(r.stats.budget_skipped == 3, "each refused miss is reported as budget-skipped");

    Config free_cfg;
    free_cfg.threshold = 0.0f;
    free_cfg.budget_per_token = 1;
    free_cfg.free_ratio = 0.9f;   // every candidate here is ~1e2 above its miss, so all are free
    const Run fr = run(free_cfg, 0, 1, K, ids, row, res);
    check(fr.stats.free_subs >= 1, "a substitution at or above free_ratio is counted as free");
    check(fr.subs >= 2, "free substitutions do not spend the budget");
}

// 10. warmup: off for the first N tokens, on afterwards
void case_warmup() {
    const int64_t NE = 8, K = 2;
    const auto res = residency(NE, {0});
    const auto row = logits_row(NE, {{0, 10.0f}, {4, 5.0f}, {5, 5.0f}});
    const std::vector<int32_t> ids{4, 5};
    Config cfg;
    cfg.threshold = 0.0f;
    cfg.warmup_tokens = 10;
    const Run warm = run(cfg, 0, 1, K, ids, row, res);
    check(warm.subs == 0 && warm.ids == ids, "inside the warmup window nothing is substituted");
    check(warm.stats.warmup_skipped == 2, "the warmup window's misses are reported");
    const Run after = run(cfg, 10, 1, K, ids, row, res);
    check(after.subs >= 1, "the token at the warmup boundary is live");
    check(effective_threshold(cfg, 9) >= 1.0f, "effective_threshold is 1.0 inside warmup");
    check(effective_threshold(cfg, 10) == 0.0f, "and the configured value after it");
}

// 11. bad input never writes anything
void case_robustness() {
    const int64_t NE = 8, K = 2;
    const auto res = residency(NE, {0});
    const auto row = logits_row(NE, {{0, 1.0f}, {4, 0.0f}, {5, 0.0f}});
    Config cfg;
    cfg.threshold = 0.0f;

    {   // out-of-range ids: left alone, not counted as misses
        const std::vector<int32_t> ids{-1, (int32_t) NE};
        const Run r = run(cfg, 0, 1, K, ids, row, res);
        check(r.ids == ids && r.stats.misses == 0, "an out-of-range id is left alone and not counted");
    }
    {   // a null logits buffer, and a null residency row: no crash, no write
        std::vector<int32_t> ids{1, 2};
        const std::vector<int32_t> before = ids;
        Scratch scratch;
        Stats stats;
        const int64_t s = substitute(cfg, 0, 1, K, NE, ids.data(), nullptr, res.data(), nullptr, scratch, stats);
        check(s == 0 && ids == before, "a null logits buffer is a no-op");
        Stats s2;
        Scratch sc2;
        const int64_t s2n = substitute(cfg, 0, 1, K, NE, ids.data(), row.data(), nullptr, nullptr, sc2, s2);
        check(s2n == 0 && ids == before, "a null residency row is a no-op (nothing is resident)");
    }
    {   // non-finite logits are neither candidates nor substitutable
        auto bad = row;
        bad[0] = std::nanf("");
        const std::vector<int32_t> ids{4, 5};
        const Run r = run(cfg, 0, 1, K, ids, bad, res);
        check(r.subs == 0, "a NaN logit is never used");
    }
    {   // a saturated ratio stays finite in the stats
        const auto far = residency(NE, {0});
        const auto huge = logits_row(NE, {{0, 100.0f}, {4, -100.0f}, {5, -100.0f}});
        const std::vector<int32_t> ids{4, 5};
        const Run r = run(cfg, 0, 1, K, ids, huge, far);
        check(r.subs == 1, "an enormous score difference still substitutes");
        check(std::isfinite(r.stats.ratio_sum), "the ratio sum stays finite");
    }
}

// 12. a window: two tokens decide independently, and the same resident expert may serve both
void case_two_tokens() {
    const int64_t NE = 8, K = 2, T = 2;
    const auto res = residency(NE, {0});
    std::vector<float> rows;
    const auto r0 = logits_row(NE, {{0, 10.0f}, {4, 5.0f}, {5, 5.0f}});
    const auto r1 = logits_row(NE, {{0, 10.0f}, {4, 6.0f}, {5, 6.0f}});
    rows.insert(rows.end(), r0.begin(), r0.end());
    rows.insert(rows.end(), r1.begin(), r1.end());
    const std::vector<int32_t> ids{4, 5, 4, 5};
    Config cfg;
    cfg.threshold = 0.0f;
    const Run r = run(cfg, 0, T, K, ids, rows, res);
    check(r.subs == 2, "one substitution per token");
    check(r.ids[0] == 0 && r.ids[2] == 0, "the same resident expert may serve two different tokens");
}

// 13. the Coder's k = 8 and 256 experts take the same path
void case_coder_k8() {
    const int64_t NE = 256, K = 8;
    const auto res = residency(NE, {7, 11, 13});
    std::vector<std::pair<int32_t, float>> v{{7, 10.0f}, {11, 9.0f}, {13, 8.0f}};
    for (int32_t e = 100; e < 108; ++e) v.emplace_back(e, 5.0f);
    const auto row = logits_row(NE, v);
    const std::vector<int32_t> ids{100, 101, 102, 103, 104, 105, 106, 107};
    Config cfg;
    cfg.threshold = 0.0f;
    const Run r = run(cfg, 0, 1, K, ids, row, res);
    check(r.subs == 3, "three resident experts serve three of the eight misses");
    for (int64_t j = 0; j < K; ++j)
        for (int64_t q = j + 1; q < K; ++q) check(r.ids[j] != r.ids[q], "still no duplicate within the token");
}

// 14. the shape this machine actually has: 48 layers, 5 tokens, 10 experts, ~21 residents
void case_real_shape() {
    const int64_t NE = 512, K = 10, T = 5;
    std::vector<int32_t> resident;
    for (int32_t e = 0; e < 21; ++e) resident.push_back(e);
    const auto res = residency(NE, resident);
    std::vector<float> rows;
    std::vector<int32_t> ids;
    for (int64_t t = 0; t < T; ++t) {
        std::vector<std::pair<int32_t, float>> v;
        for (int32_t e = 0; e < 21; ++e) v.emplace_back(e, 6.0f - 0.1f * (float) e);
        for (int64_t j = 0; j < K; ++j) {
            const int32_t e = (int32_t) (200 + t * 20 + j);
            v.emplace_back(e, 5.0f - 0.2f * (float) j);
            ids.push_back(e);
        }
        const auto row = logits_row(NE, v);
        rows.insert(rows.end(), row.begin(), row.end());
    }
    Config cfg;
    cfg.threshold = 0.35f;
    const Run r = run(cfg, 0, T, K, ids, rows, res);
    check(r.subs > 0, "the realistic shape substitutes something (21 residents, ranks 4..10)");
    int64_t changed = 0;
    for (size_t i = 0; i < ids.size(); ++i)
        if (r.ids[i] != ids[i]) {
            ++changed;
            check(res[(size_t) r.ids[i]] != kMiss, "every change is to a resident expert");
        }
    check(changed == r.subs, "exactly the accepted entries changed");
    check(r.stats.misses == T * K, "every routed expert was a miss in a 21-slot cache");

    // ... and the same window 48 times, as a layer stack would run it: no allocation growth, same result.
    Scratch scratch;
    Stats total;
    for (int layer = 0; layer < 48; ++layer) {
        std::vector<int32_t> ids2 = ids;
        Stats st;
        const int64_t subs = substitute(cfg, layer, T, K, NE, ids2.data(), rows.data(), res.data(), nullptr,
                                        scratch, st);
        total.accum(st);
        if (layer == 0) check(subs == r.subs, "a reused scratch buffer decides the same window the same way");
    }
    check(total.groups == 48, "one group per layer");
    check(total.substitutions == 48 * r.subs, "48 layers substitute 48 windows' worth");
}

void case_stats_accumulate() {
    Stats a, b;
    a.groups = 1; a.entries = 10; a.misses = 9; a.substitutions = 4; a.ratio_sum = 2.0;
    b.groups = 2; b.entries = 20; b.misses = 18; b.substitutions = 5; b.ratio_sum = 1.5;
    a.accum(b);
    check(a.groups == 3 && a.entries == 30 && a.substitutions == 9, "accum sums the counters");
    check(std::fabs(a.mean_ratio() - (3.5 / 9.0)) < 1e-12, "mean_ratio is the sum over the substitutions");
    a.reset();
    check(a.groups == 0 && a.mean_ratio() == 0.0, "reset clears everything");
}

// =========================================================================================================
// 15. the differential sweep
uint64_t rng_state = 0x9E3779B97F4A7C15ull;
uint64_t rnd() {
    rng_state ^= rng_state << 13;
    rng_state ^= rng_state >> 7;
    rng_state ^= rng_state << 17;
    return rng_state;
}
double rnd01() { return (double) (rnd() % 1000000ull) / 1000000.0; }

void case_differential() {
    const int64_t expert_choices[] = {8, 16, 64, 512};
    const int64_t k_choices[] = {1, 2, 4, 8, 10};
    const int64_t tok_choices[] = {1, 2, 5};
    const float taus[] = {0.0f, 0.1f, 0.35f, 0.5f, 1.0f};

    int64_t cases = 0, subs_total = 0, refusals = 0;
    for (int iter = 0; iter < 4000; ++iter) {
        const int64_t NE = expert_choices[rnd() % 4];
        const int64_t K = k_choices[rnd() % 5];
        const int64_t T = tok_choices[rnd() % 3];
        if (K > NE) continue;
        Config cfg;
        cfg.threshold = taus[rnd() % 5];
        cfg.warmup_tokens = (int64_t) (rnd() % 4);
        cfg.budget_per_token = (int64_t) (rnd() % 3);            // 0, 1 or 2
        cfg.free_ratio = (rnd() % 2) ? 0.0f : 0.6f;

        // a random cache: up to a quarter of the layer resident
        const int64_t n_res = (int64_t) (rnd() % (uint64_t) (NE / 4 + 1));
        std::vector<int32_t> resident;
        for (int64_t e = 0; e < NE && (int64_t) resident.size() < n_res; ++e)
            if (rnd() % 4 == 0) resident.push_back((int32_t) e);
        const auto res = residency(NE, resident);

        // random logits, with duplicates and a NaN now and then
        std::vector<float> logits((size_t) (T * NE));
        for (auto& v : logits) {
            const double u = rnd01();
            v = (float) (-2.0 + u * 8.0);
        }
        if (rnd() % 8 == 0) logits[(size_t) (rnd() % logits.size())] = std::nanf("");

        // A token's routing has DISTINCT experts (the router picks k different ones) and may occasionally
        // carry an invalid id, which is what the pool's own guard sees.
        std::vector<int32_t> ids((size_t) (T * K));
        for (int64_t t = 0; t < T; ++t) {
            std::vector<int32_t> picked;
            for (int64_t j = 0; j < K; ++j) {
                int32_t e = -2;
                for (int attempt = 0; attempt < 64; ++attempt) {
                    const int64_t r = (int64_t) (rnd() % 12);
                    if (r == 11) { e = -1; break; }          // an invalid id: left alone by every rule
                    const int32_t c = (int32_t) (rnd() % (uint64_t) NE);
                    if (std::find(picked.begin(), picked.end(), c) == picked.end()) { e = c; break; }
                }
                if (e == -2) e = 0;                          // the loop gave up: a distinct id is not needed
                picked.push_back(e);
                ids[(size_t) (t * K + j)] = e;
            }
        }

        const int64_t token_index = (int64_t) (rnd() % 6);

        // The engine's function, on a copy.
        std::vector<int32_t> got = ids;
        std::vector<float> got_ratio(ids.size(), 0.0f);
        Scratch scratch;
        Stats st;
        const int64_t subs = substitute(cfg, token_index, T, K, NE, got.data(), logits.data(), res.data(),
                                        got_ratio.data(), scratch, st);

        // The reference, on the same input.
        const RefResult ref = reference(cfg, token_index, T, K, NE, ids, logits, res);

        ++cases;
        subs_total += subs;
        if (subs == 0) ++refusals;
        if (got != ref.ids) {
            std::fprintf(stderr, "FAIL: differential case %d (NE=%lld k=%lld T=%lld tau=%g warm=%lld budget=%lld "
                                 "free=%g token=%lld): the id lists differ\n",
                         iter, (long long) NE, (long long) K, (long long) T, (double) cfg.threshold,
                         (long long) cfg.warmup_tokens, (long long) cfg.budget_per_token, (double) cfg.free_ratio,
                         (long long) token_index);
            ++g_failures;
            return;
        }
        if (!same_stats(st, ref.stats)) {
            std::fprintf(stderr, "FAIL: differential case %d: the stats differ (subs %lld/%lld, misses %lld/%lld)\n",
                         iter, (long long) st.substitutions, (long long) ref.stats.substitutions,
                         (long long) st.misses, (long long) ref.stats.misses);
            ++g_failures;
            return;
        }
        // invariants that hold whatever the rule: each substitution is a resident expert, an invalid id was
        // never touched, and a substitution never collides with another entry of its token.
        for (int64_t t = 0; t < T; ++t) {
            for (int64_t j = 0; j < K; ++j) {
                const int64_t i = t * K + j;
                if (got[(size_t) i] != ids[(size_t) i]) check(res[(size_t) got[(size_t) i]] != kMiss,
                                                              "differential: a change is to a resident expert");
                if (ids[(size_t) i] < 0 || ids[(size_t) i] >= NE) check(got[(size_t) i] == ids[(size_t) i],
                                                                        "differential: an invalid id is untouched");
                // The generator gives each token DISTINCT valid ids, so any repeat of a valid id is the
                // substitution's doing.  Invalid ids (-1, twice in a token) are the input's and are skipped.
                if (got[(size_t) i] >= 0 && got[(size_t) i] < NE)
                    for (int64_t q = 0; q < K; ++q)
                        if (q != j)
                            check(got[(size_t) i] != got[(size_t) (t * K + q)],
                                  "differential: no expert appears twice in a token");
            }
        }
    }
    std::printf("car_substitute_test: %lld randomized windows agreed with the reference (%lld substitutions, "
                "%lld with none)\n",
                (long long) cases, (long long) subs_total, (long long) refusals);
}

// =========================================================================================================
// the golden case for tools/test_car_estimate.py: one small window, written in plain text
void emit_golden(const char* path) {
    const int64_t NE = 24, K = 4, T = 2;
    const auto res = residency(NE, {0, 1, 2, 3, 4});
    std::vector<float> rows;
    const auto r0 = logits_row(NE, {{0, 10.0f}, {1, 9.5f}, {2, 9.0f}, {3, 8.0f}, {4, 7.0f},
                                    {12, 9.8f}, {13, 9.6f}, {14, 7.0f}, {15, 6.0f}});
    const auto r1 = logits_row(NE, {{0, 10.0f}, {1, 9.5f}, {2, 9.0f}, {3, 8.0f}, {4, 7.0f},
                                    {16, 10.5f}, {17, 9.0f}, {18, 8.5f}, {19, 8.4f}});
    rows.insert(rows.end(), r0.begin(), r0.end());
    rows.insert(rows.end(), r1.begin(), r1.end());
    const std::vector<int32_t> ids{12, 13, 14, 15, 16, 17, 18, 19};
    Config cfg;
    cfg.threshold = 0.35f;
    const Run r = run(cfg, 0, T, K, ids, rows, res);

    std::FILE* f = std::fopen(path, "w");
    if (f == nullptr) {
        std::fprintf(stderr, "cannot write %s\n", path);
        ++g_failures;
        return;
    }
    std::fprintf(f, "car-golden 1\n");
    std::fprintf(f, "n_expert %lld\nk %lld\nn_tok %lld\nthreshold %.6f\n", (long long) NE, (long long) K,
                 (long long) T, (double) cfg.threshold);
    std::fprintf(f, "res");
    for (int64_t e = 0; e < NE; ++e) std::fprintf(f, " %d", res[(size_t) e]);
    std::fprintf(f, "\nlogits");
    for (float x : rows) std::fprintf(f, " %.6f", (double) x);
    std::fprintf(f, "\nids");
    for (int32_t x : ids) std::fprintf(f, " %d", x);
    std::fprintf(f, "\nexpected");
    for (int32_t x : r.ids) std::fprintf(f, " %d", x);
    std::fprintf(f, "\n");
    std::fclose(f);
    std::printf("car_substitute_test: golden case written (%lld substitutions)\n", (long long) r.subs);
}

}  // namespace

int main(int argc, char** argv) {
    for (int i = 1; i < argc; ++i)
        if (std::strcmp(argv[i], "--emit-golden") == 0 && i + 1 < argc) {
            emit_golden(argv[i + 1]);
            return g_failures == 0 ? 0 : 1;
        }

    case_off_is_off();
    case_ratio_threshold();
    case_boundary_is_ge();
    case_no_candidate();
    case_selected_is_not_a_candidate();
    case_no_duplicate_within_token();
    case_best_ratio_first();
    case_all_resident();
    case_budget();
    case_warmup();
    case_robustness();
    case_two_tokens();
    case_coder_k8();
    case_real_shape();
    case_stats_accumulate();
    case_differential();

    if (g_failures == 0) puts("car_substitute_test: all cases passed");
    return g_failures == 0 ? 0 : 1;
}
