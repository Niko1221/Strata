// prefill_tol.h - the TOLERANCE judge of --prefill-consistency.
//
// An MMA prefill sums in a different fp32 order than the decode path, so its logits are no longer bitwise the
// decode path's.  The judge compares, position by position, the prefill run's logits against the decode path
// run over the same tokens (the reference):
//   KL(P_decode || P_prefill) over the whole vocabulary, in float64 with a shifted log-softmax, and whether the
//   argmax agrees.  Positions are the prompt's (every S-th and the last 64) and a few CONTINUATION positions
//   decoded after the prompt from the state the prompt left behind - the state a real generation reads, which
//   the prompt's own logits do not cover for the last chunk's KV.
// The thresholds come from a file (`key value` lines: kl_mean, kl_p99, tail01_max, tail1_max, top1_min,
// next_margin), derived beforehand from a perturbation known to be harmless
// (tmp/tests/hy4_flash/pyref/prefill_tol_calibrate.py), never tuned on the candidate.  Header-only so
// tmp/tests/hy4_flash/engine/prefill_tol_test.cpp can test it without a GPU.
#pragma once

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <string>
#include <utility>
#include <vector>

namespace strata::hy4_flash {

struct TolPos {
    int pos = 0;
    int cont = 0;          ///< 0 = a prompt position, 1 = a continuation position
    double kl = 0;         ///< KL(P_ref || P_test), nats
    int32_t am_ref = 0, am_test = 0;
    double max_abs = 0;    ///< the largest |ref - test| over the vocabulary
    double gap12 = 0;      ///< the reference's top-1 minus its top-2 logit: how near a tie its pick is
    double deficit = 0;    ///< ref[am_ref] - ref[am_test]: how much worse the test's pick is BY THE REFERENCE (0 = same)
};

/// KL(softmax(ref) || softmax(test)) in float64.  A non-finite logit on either side gives +inf (a FAIL).
inline double tol_kl(const float* ref, const float* test, int32_t n) {
    double mr = -INFINITY, mt = -INFINITY;
    for (int32_t v = 0; v < n; ++v) {
        if (!std::isfinite(ref[v]) || !std::isfinite(test[v])) return INFINITY;
        mr = std::max(mr, (double) ref[v]);
        mt = std::max(mt, (double) test[v]);
    }
    double sr = 0, st = 0;
    for (int32_t v = 0; v < n; ++v) {
        sr += std::exp((double) ref[v] - mr);
        st += std::exp((double) test[v] - mt);
    }
    const double lr = std::log(sr), lt = std::log(st);
    double kl = 0;
    for (int32_t v = 0; v < n; ++v) {
        const double a = (double) ref[v] - mr - lr, b = (double) test[v] - mt - lt;
        kl += std::exp(a) * (a - b);
    }
    return std::max(0.0, kl);   // >= 0 mathematically; a -1e-17 from rounding is reported as 0
}

inline int32_t tol_argmax(const float* x, int32_t n) {
    int32_t b = 0;
    for (int32_t v = 1; v < n; ++v)
        if (x[v] > x[b]) b = v;
    return b;
}

inline TolPos tol_compare(int pos, int cont, const float* ref, const float* test, int32_t n) {
    TolPos r;
    r.pos = pos;
    r.cont = cont;
    r.kl = tol_kl(ref, test, n);
    r.am_ref = tol_argmax(ref, n);
    r.am_test = tol_argmax(test, n);
    double second = -INFINITY;
    for (int32_t v = 0; v < n; ++v) {
        r.max_abs = std::max(r.max_abs, std::fabs((double) ref[v] - (double) test[v]));
        if (v != r.am_ref) second = std::max(second, (double) ref[v]);
    }
    r.gap12 = (double) ref[r.am_ref] - second;
    r.deficit = (double) ref[r.am_ref] - (double) ref[r.am_test];
    return r;
}

struct TolSummary {
    size_t n = 0;
    double kl_mean = 0, kl_p99 = 0, kl_max = 0, top1 = 1;   ///< top1 = the fraction of positions whose argmax agrees
    double next_kl = 0;    ///< the last prompt position's KL (the next-token logits of a generation)
    bool next_top1 = true;
    double next_gap12 = 0, next_deficit = 0;   ///< that position's TolPos::gap12 / deficit
    double max_deficit = 0;                     ///< the largest deficit over every compared position
    /// The FRACTION of compared positions whose KL exceeds a fixed bound.  These are what the verdict uses instead
    /// of kl_max: a maximum grows with the number of positions compared even when the per-position distribution is
    /// unchanged, so a limit derived at one prompt length rejects the same kernel at a longer one.  A rate estimates
    /// a property of the distribution, so it transfers.  kl_max stays reported.
    double tail_01 = 0;   ///< KL > kTail01 (0.1 nats: the reference's own pick is ~10% less likely under the test)
    double tail_1 = 0;    ///< KL > kTail1 (1 nat: a position where the two distributions really disagree)
};

constexpr double kTail01 = 0.1, kTail1 = 1.0;

/// p99 is nearest-rank, as tmp/tests/hy4_flash/golden/logits_stats.py does it.
inline TolSummary tol_summarize(const std::vector<TolPos>& v, int last_prompt_pos) {
    TolSummary s;
    s.n = v.size();
    if (v.empty()) return s;
    std::vector<double> k;
    size_t agree = 0;
    for (const TolPos& p : v) {
        k.push_back(p.kl);
        s.kl_mean += p.kl;
        s.kl_max = std::max(s.kl_max, p.kl);
        if (p.am_ref == p.am_test) ++agree;
        s.max_deficit = std::max(s.max_deficit, p.deficit);
        if (!p.cont && p.pos == last_prompt_pos) {
            s.next_kl = p.kl;
            s.next_top1 = p.am_ref == p.am_test;
            s.next_gap12 = p.gap12;
            s.next_deficit = p.deficit;
        }
    }
    s.kl_mean /= (double) v.size();
    size_t t01 = 0, t1 = 0;
    for (const TolPos& p : v) {
        if (p.kl > kTail01) ++t01;
        if (p.kl > kTail1) ++t1;
    }
    s.tail_01 = (double) t01 / (double) v.size();
    s.tail_1 = (double) t1 / (double) v.size();
    std::sort(k.begin(), k.end());
    const size_t r = (size_t) std::max<double>(1.0, std::ceil(0.99 * (double) k.size()));
    s.kl_p99 = k[r - 1];
    s.top1 = (double) agree / (double) v.size();
    return s;
}

struct TolThresholds {
    double kl_mean = -1, kl_p99 = -1, top1_min = -1;
    /// The limits on TolSummary::tail_01 / tail_1 (fractions).  They are rates rather than a kl_max limit, which
    /// does not transfer between prompt lengths.
    double tail01_max = -1, tail1_max = -1;
    /// The next token's argmax may differ only at a near-tie: when the reference rates the test's pick at most
    /// `next_margin` logits below its own (benign reorderings flip near-ties).
    double next_margin = -1;
    std::string source;
    /// `key value` lines; '#' starts a comment.  All four keys are required (a missing one is an error, not a
    /// default: a judge with an unset threshold passes everything).
    bool load(const std::string& path, std::string& err) {
        source = path;
        FILE* f = std::fopen(path.c_str(), "r");
        if (f == nullptr) { err = "cannot read the thresholds " + path; return false; }
        char line[512];
        while (std::fgets(line, sizeof line, f)) {
            char key[64];
            double val = 0;
            if (line[0] == '#' || std::sscanf(line, "%63s %lf", key, &val) != 2) continue;
            const std::string k = key;
            if (k == "kl_mean") kl_mean = val;
            else if (k == "kl_p99") kl_p99 = val;
            else if (k == "tail01_max") tail01_max = val;
            else if (k == "tail1_max") tail1_max = val;
            else if (k == "kl_max") continue;   // an older file's key: reported, never a limit
            else if (k == "top1_min") top1_min = val;
            else if (k == "next_margin") next_margin = val;
        }
        std::fclose(f);
        // A non-finite limit would pass everything: inf <= inf is true, so an `inf` in the file would accept even
        // a NaN logit's infinite KL.  Rejected here, with the missing-key check.
        for (const double v : {kl_mean, kl_p99, tail01_max, tail1_max, top1_min, next_margin})
            if (!std::isfinite(v)) {
                err = path + " has a threshold that is not a finite number";
                return false;
            }
        if (kl_mean < 0 || kl_p99 < 0 || tail01_max < 0 || tail1_max < 0 || top1_min < 0 || next_margin < 0) {
            err = path + " must set kl_mean, kl_p99, tail01_max, tail1_max, top1_min and next_margin";
            return false;
        }
        return true;
    }
};

/// PASS iff every statistic is within its threshold, the next-token argmax agrees or differs only at a near-tie
/// (deficit <= next_margin), and nothing was non-finite.
/// `why` lists every limit that was exceeded.
inline bool tol_pass(const TolSummary& s, const TolThresholds& t, std::string& why) {
    why.clear();
    char b[160];
    auto over = [&](const char* name, double v, double lim) {
        std::snprintf(b, sizeof b, "%s%s %.3g > %.3g", why.empty() ? "" : "; ", name, v, lim);
        why += b;
    };
    if (s.n == 0) { why = "no positions compared"; return false; }
    // The thresholds too, not only what load() read: a hand-built TolThresholds with an infinite limit would
    // otherwise accept anything.
    for (const double v : {t.kl_mean, t.kl_p99, t.tail01_max, t.tail1_max, t.top1_min, t.next_margin})
        if (!std::isfinite(v)) { why = "a threshold is not a finite number"; return false; }
    // A non-finite statistic is a FAIL on its own, whatever the limits say: tol_kl returns inf for a non-finite
    // logit, and a limit of inf would otherwise accept it.
    for (const auto& st : {std::pair<const char*, double>{"KL mean", s.kl_mean},
                           {"KL p99", s.kl_p99},
                           {"KL max", s.kl_max},
                           {"the KL > 0.1 fraction", s.tail_01},
                           {"the KL > 1 fraction", s.tail_1},
                           {"top-1 agreement", s.top1},
                           {"the next-token deficit", s.next_deficit}})
        if (!std::isfinite(st.second)) {
            std::snprintf(b, sizeof b, "%s%s is not finite", why.empty() ? "" : "; ", st.first);
            why += b;
        }
    if (!why.empty()) return false;
    if (!(s.kl_mean <= t.kl_mean)) over("KL mean", s.kl_mean, t.kl_mean);
    if (!(s.kl_p99 <= t.kl_p99)) over("KL p99", s.kl_p99, t.kl_p99);
    if (!(s.tail_01 <= t.tail01_max)) over("the KL > 0.1 fraction", s.tail_01, t.tail01_max);
    if (!(s.tail_1 <= t.tail1_max)) over("the KL > 1 fraction", s.tail_1, t.tail1_max);
    if (!(s.top1 >= t.top1_min)) {
        std::snprintf(b, sizeof b, "%stop-1 agreement %.4f < %.4f", why.empty() ? "" : "; ", s.top1, t.top1_min);
        why += b;
    }
    if (!s.next_top1 && !(s.next_deficit <= t.next_margin)) {
        std::snprintf(b, sizeof b, "%sthe next-token argmax differs by %.3g logits (margin %.3g)", why.empty() ? "" : "; ",
                      s.next_deficit, t.next_margin);
        why += b;
    }
    return why.empty();
}

}  // namespace strata::hy4_flash
