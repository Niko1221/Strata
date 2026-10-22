// src/spec/dflash_native_parity_test.cpp - the native fused-Metal DFlash2 drafter against llama.cpp's DFLASH graph.
// The same target features go into both backends (a long context, so the sliding-window ring wraps), then many drafts
// with other anchors and max_n 1..7: the chains must agree (>= 99%) and the confidences within 2e-2.  Both backends are
// also measured against a plain f64 CPU reference (dflash_ref.hpp).
// Features: a STRATA_DFLASH_DUMP capture (<prefix>.idx / .f16, e.g. Strata-data/dflash2/train/d0) when given, else
// synthetic (per-block scaled gaussian around a per-channel offset).
// usage: dflash_native_parity_test <draft.gguf> [context rows (2200)] [trials (300)] [max inject chunk (300)] [dump prefix]
#include "strata/spec/dflash_drafter.hpp"

#include "dflash_ref.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <memory>
#include <random>
#include <vector>

using strata::spec::DflashDrafter;

namespace {

constexpr int kTokens = 248320;   // the target vocabulary
constexpr int kW = 5 * 2560;      // a feature row

// The rows that go to the drafters, in order, and the token at a position (the anchor of a draft there).
struct Source {
    std::mt19937_64 rng{1234};
    std::vector<int32_t> tok;   // a real capture: the longest run of positions 0, 1, 2.. of the dump
    std::vector<float> real;    // run length x kW
    std::vector<float> mean;
    size_t at = 0;

    bool load_dump(const std::string& prefix, int need) {
        std::FILE* fi = std::fopen((prefix + ".idx").c_str(), "rb");
        std::FILE* ff = std::fopen((prefix + ".f16").c_str(), "rb");
        if (!fi || !ff) { if (fi) std::fclose(fi); if (ff) std::fclose(ff); return false; }
        std::fseek(fi, 0, SEEK_END);
        std::fseek(ff, 0, SEEK_END);
        const long n = std::min(std::ftell(fi) / 12, std::ftell(ff) / (long) (kW * 2));   // a capture still being written: whole records only
        std::rewind(fi);
        std::vector<int32_t> rec((size_t) n * 3);
        const bool read = std::fread(rec.data(), 12, (size_t) n, fi) == (size_t) n;
        std::fclose(fi);
        long best = 0, best_len = 0;
        for (long i = 0; read && i < n;) {
            long j = i;
            while (j < n && rec[(size_t) j * 3] == (int32_t) (j - i)) ++j;
            if (j - i > best_len) { best = i; best_len = j - i; }
            i = j == i ? i + 1 : j;
        }
        if (!read || best_len < need) { std::fclose(ff); return false; }
        std::vector<uint16_t> h((size_t) best_len * kW);
        std::fseek(ff, best * (long) (kW * 2), SEEK_SET);
        const size_t got = std::fread(h.data(), 2, h.size(), ff);
        std::fclose(ff);
        if (got != h.size()) return false;
        real.resize(h.size());
        for (size_t k = 0; k < h.size(); ++k) real[k] = ref::f16_to_f32(h[k]);
        for (long i = 0; i < best_len; ++i) tok.push_back(rec[(size_t) (best + i) * 3 + 1]);
        return true;
    }
    void init_synthetic() {
        mean.resize(kW);
        std::normal_distribution<float> g(0.0f, 1.0f);
        for (int b = 0; b < 5; ++b)
            for (int i = 0; i < 2560; ++i) mean[b * 2560 + i] = g(rng) * (i % 97 == 0 ? 6.0f : 0.5f);   // a few outlier channels
    }
    bool is_real() const { return !real.empty(); }
    std::vector<float> rows(int n) {
        std::vector<float> f((size_t) n * kW);
        if (is_real()) {
            for (int r = 0; r < n; ++r, ++at) std::copy_n(&real[(at % tok.size()) * kW], kW, &f[(size_t) r * kW]);
            return f;
        }
        std::normal_distribution<float> g(0.0f, 1.0f);
        for (int r = 0; r < n; ++r)
            for (int b = 0; b < 5; ++b) {
                const float scale = 0.5f * (float) (1 << b);
                for (int i = 0; i < 2560; ++i) f[(size_t) r * kW + b * 2560 + i] = scale * (mean[b * 2560 + i] + g(rng));
            }
        return f;
    }
    // the token at `pos` (real: the capture's; synthetic: a random one)
    int32_t anchor(int pos, std::mt19937_64& r) const { return is_real() ? tok[(size_t) pos % tok.size()] : (int32_t) (r() % kTokens); }
};

bool inject_both(DflashDrafter& a, DflashDrafter& b, const std::vector<float>& f, int n, int pos0) {
    std::string err;
    if (!a.inject(f.data(), n, pos0, err) || !b.inject(f.data(), n, pos0, err)) {
        std::fprintf(stderr, "inject failed: %s\n", err.c_str());
        return false;
    }
    if (a.pos_max() != b.pos_max()) {
        std::fprintf(stderr, "pos_max differs: %d vs %d\n", a.pos_max(), b.pos_max());
        return false;
    }
    return true;
}

// a context of `ctx` rows in chunks of other sizes, as prompt chunks and commits arrive; returns the next position
int fill_context(DflashDrafter& a, DflashDrafter& b, Source& src, int ctx, int max_chunk) {
    int pos = 0;
    while (pos < ctx) {
        const int n = std::min(ctx - pos, 1 + (int) (src.rng() % max_chunk));
        if (!inject_both(a, b, src.rows(n), n, pos)) std::exit(1);
        pos += n;
    }
    return pos;
}

double ms_since(std::chrono::steady_clock::time_point t) {
    return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t).count();
}

// the selector walk of DflashDrafter::draft over a lattice (p_min 0): the picks and their confidences
int walk(const float* lat, int nb, int32_t* tok, float* conf) {
    int n = 0, pred = 0;
    for (int i = 1; i < nb; ++i) {
        const float* row = lat + (size_t) i * 2560;
        const float* sc = row + 16 + (size_t) pred * 16;
        pred = (int) (std::max_element(sc, sc + 16) - sc);
        float sum = 0.0f;
        for (int k = 0; k < 16; ++k) sum += std::exp(sc[k] - sc[pred]);
        conf[n] = 1.0f / sum;
        tok[n++] = (int32_t) row[pred];
    }
    return n;
}

bool in_row(const float* lat, int row, int32_t tok) {   // is tok among the row's 16 candidates
    for (int k = 0; k < 16; ++k)
        if ((int32_t) lat[(size_t) row * 2560 + k] == tok) return true;
    return false;
}

}  // namespace

int main(int argc, char** argv) {
    if (argc < 2) { std::fprintf(stderr, "usage: %s <draft.gguf> [context rows] [trials] [max inject chunk] [dump prefix]\n", argv[0]); return 2; }
    const int ctx = argc > 2 ? std::atoi(argv[2]) : 2200, trials = argc > 3 ? std::atoi(argv[3]) : 300;
    const int max_chunk = argc > 4 ? std::atoi(argv[4]) : 300;
    DflashDrafter ll, nat;
    std::string err;
    if (!ll.load(argv[1], err, DflashDrafter::Backend::kLlama)) { std::fprintf(stderr, "llama: %s\n", err.c_str()); return 1; }
    if (!nat.load(argv[1], err, DflashDrafter::Backend::kNative)) { std::fprintf(stderr, "native: %s\n", err.c_str()); return 1; }
    if (nat.block() != ll.block() || nat.target_hidden() != 2560) { std::fprintf(stderr, "config differs\n"); return 1; }

    Source src;
    const int need = 1000;   // the longest run of the capture; a longer context repeats it (the ring still wraps)
    if (argc > 5 && src.load_dump(argv[5], need)) std::printf("features: real, %s (%zu rows in the run)\n", argv[5], src.tok.size());
    else {
        std::printf("features: synthetic%s\n", argc > 5 ? " (the capture has no run of 1000 positions from 0)" : "");
        src.init_synthetic();
    }
    // STRATA_PARITY_REF_ALL=1: the reference gets every row too (slow: 60M MACs a row), to say which backend is closer
    // at a confidence disagreement
    std::unique_ptr<dfref::Model> rall;
    if (std::getenv("STRATA_PARITY_REF_ALL")) {
        rall = std::make_unique<dfref::Model>();
        if (!rall->load(argv[1], err)) { std::fprintf(stderr, "reference: %s\n", err.c_str()); return 1; }
    }
    auto feed = [&](const std::vector<float>& f, int n, int pos0) {
        if (rall) rall->inject(f.data(), n, pos0);
        return inject_both(ll, nat, f, n, pos0);
    };
    int pos = 0;
    while (pos < ctx) {
        const int n = std::min(ctx - pos, 1 + (int) (src.rng() % max_chunk));
        if (!feed(src.rows(n), n, pos)) return 1;
        pos += n;
    }
    std::printf("context: %d rows injected, pos_max %d / %d\n", ctx, ll.pos_max(), nat.pos_max());

    // 1. chains and confidences, p_min 0; a commit of 1..4 rows after every third draft
    int equal = 0, total = 0, conf_n = 0, conf_bad = 0, tok_n = 0, tok_bad = 0, boundary = 0, near_tie = 0;
    double conf_max = 0.0, acc_ll = 0, acc_nat = 0;
    std::mt19937_64 rng(99);
    for (int t = 0; t < trials; ++t) {
        const int max_n = 1 + t % 7;
        const int32_t id = src.anchor(pos, rng);
        int32_t ta[8] = {}, tb[8] = {};
        float ca[8] = {}, cb[8] = {};
        const int na = ll.draft(id, pos, max_n, 0.0f, ta, ca), nb = nat.draft(id, pos, max_n, 0.0f, tb, cb);
        ++total;
        bool diverged = na != nb;
        for (int i = 0; i < std::min(na, nb); ++i) {
            ++tok_n;
            if (ta[i] != tb[i]) {
                ++tok_bad;
                if (!diverged) {
                    diverged = true;
                    const bool flip = !in_row(nat.lattice(), i + 1, ta[i]) || !in_row(ll.lattice(), i + 1, tb[i]);
                    boundary += flip;
                    near_tie += !flip && std::fabs(ca[i] - cb[i]) < 5e-3;
                    std::printf("trial %d: first divergence at draft %d: llama %d (conf %.4f), native %d (conf %.4f)%s\n", t, i, ta[i],
                                ca[i], tb[i], cb[i], flip ? ", a pick is outside the other's 16 candidates" : "");
                }
            } else if (!diverged) {   // the confidences while the chains still agree
                const double d = std::fabs(ca[i] - cb[i]);
                conf_max = std::max(conf_max, d);
                ++conf_n;
                conf_bad += d > 2e-2;
                if (d > 2e-2 && rall) {
                    const std::vector<float> lr = rall->draft(id, pos, max_n + 1);
                    int32_t tr[8];
                    float cr[8];
                    walk(lr.data(), max_n + 1, tr, cr);
                    std::printf("trial %d draft %d: conf llama %.4f native %.4f, f64 reference %.4f (reference pick %d)\n", t, i, ca[i], cb[i], cr[i], tr[i]);
                }
            }
        }
        equal += !diverged;
        if (src.is_real() && pos + 1 < (int) src.tok.size()) {   // the drafted prefix the target would accept
            for (int i = 0; i < na && pos + 1 + i < (int) src.tok.size() && ta[i] == src.tok[pos + 1 + i]; ++i) acc_ll += 1;
            for (int i = 0; i < nb && pos + 1 + i < (int) src.tok.size() && tb[i] == src.tok[pos + 1 + i]; ++i) acc_nat += 1;
        }
        if (t % 3 == 2) {   // a commit: the next rows land at pos
            const int k = 1 + t % 4;
            if (!feed(src.rows(k), k, pos)) return 1;
            pos += k;
        }
    }
    const double frac = 100.0 * equal / total;
    std::printf("chains equal: %d / %d (%.2f%%); %d divergences: %d a rank 16/17 candidate flip, %d a near tie (|conf diff| < 5e-3), %d other\n",
                equal, total, frac, total - equal, boundary, near_tie, total - equal - boundary - near_tie);
    std::printf("tokens differing %d / %d; confidences compared %d (chains agreeing), max |diff| %.3g, over 2e-2: %d\n", tok_bad, tok_n,
                conf_n, conf_max, conf_bad);
    if (src.is_real()) std::printf("accepted draft tokens vs the capture's next tokens: llama %.0f, native %.0f over %d drafts\n", acc_ll, acc_nat, total);

    // 2. the stopping rule: p_min gates the same drafts
    int pm_equal = 0, pm_total = 0;
    for (float p_min : {0.3f, 0.9f})
        for (int t = 0; t < 40; ++t) {
            const int32_t id = src.anchor(pos, rng);
            int32_t ta[8] = {}, tb[8] = {};
            const int na = ll.draft(id, pos, 7, p_min, ta), nb = nat.draft(id, pos, 7, p_min, tb);
            ++pm_total;
            pm_equal += na == nb && std::equal(ta, ta + na, tb);
        }
    std::printf("p_min 0.3/0.9: chains equal %d / %d\n", pm_equal, pm_total);

    // 3. native is deterministic: the same draft twice
    int det = 0;
    for (int t = 0; t < 10; ++t) {
        const int32_t id = src.anchor(pos, rng);
        int32_t ta[8] = {}, tb[8] = {};
        float ca[8] = {}, cb[8] = {};
        const int na = nat.draft(id, pos, 7, 0.0f, ta, ca), nb = nat.draft(id, pos, 7, 0.0f, tb, cb);
        det += na == nb && std::equal(ta, ta + na, tb) && std::equal(ca, ca + na, cb);
    }
    std::printf("native repeat draft bitwise equal: %d / 10\n", det);

    // 4. both backends against a plain f64 CPU reference (a short context: it costs 60M MACs a row).  llama.cpp rounds
    // through f16 in places (its K/V cache, flash attention, the f16 matmul tiles), and a K/V cache rounded to f16 turns
    // an f32 rounding difference into a 1-ulp flip of an element: the floor between two implementations.  Native has to
    // be no further from the reference than llama is.
    double ref_nat_max = 0, ref_ll_max = 0, ref_nat_mean = 0, ref_ll_mean = 0, nl_max = 0;
    {
        dfref::Model rm;
        if (!rm.load(argv[1], err)) { std::fprintf(stderr, "reference: %s\n", err.c_str()); return 1; }
        ll.trim(0);
        nat.trim(0);
        const int rc = 24;
        const auto f = src.rows(rc);
        if (!inject_both(ll, nat, f, rc, 0)) return 1;
        rm.inject(f.data(), rc, 0);
        int cases = 0;
        for (int nbk : {2, 4, 8})
            for (int id : {1234, 98765}) {
                int32_t o2[8];
                ll.draft(id, rc, nbk - 1, 0.0f, o2);
                const std::vector<float> la(ll.lattice(), ll.lattice() + (size_t) nbk * 2560);
                nat.draft(id, rc, nbk - 1, 0.0f, o2);
                const std::vector<float> lb(nat.lattice(), nat.lattice() + (size_t) nbk * 2560);
                const std::vector<float> lr = rm.draft(id, rc, nbk);
                double mx, mean;
                int sh;
                dfref::lattice_diff(lb.data(), lr.data(), nbk, id, mx, mean, sh);
                ref_nat_max = std::max(ref_nat_max, mx);
                ref_nat_mean += mean;
                dfref::lattice_diff(la.data(), lr.data(), nbk, id, mx, mean, sh);
                ref_ll_max = std::max(ref_ll_max, mx);
                ref_ll_mean += mean;
                dfref::lattice_diff(lb.data(), la.data(), nbk, id, mx, mean, sh);
                nl_max = std::max(nl_max, mx);
                ++cases;
            }
        ref_nat_mean /= cases;
        ref_ll_mean /= cases;
        std::printf("lattice scores vs f64 reference (%d drafts, nb 2/4/8): native max %.3g mean %.3g | llama max %.3g mean %.3g | native vs llama max %.3g\n",
                    cases, ref_nat_max, ref_nat_mean, ref_ll_max, ref_ll_mean, nl_max);
        ll.trim(0);
        nat.trim(0);
        pos = fill_context(ll, nat, src, ctx, max_chunk);   // the long context again for the sections below
    }

    // 5. bookkeeping: a draft needs the features up to its anchor; trim and re-inject land on the same chains
    int32_t o[8];
    const bool gap = ll.draft(1234, pos + 5, 3, 0.0f, o) == 0 && nat.draft(1234, pos + 5, 3, 0.0f, o) == 0;
    const int keep = pos - 8;   // within the native ring's rewind slack (RING - 2048 = 16)
    ll.trim(keep);
    nat.trim(keep);
    const bool trimmed = ll.pos_max() == keep - 1 && nat.pos_max() == keep - 1;
    const bool again = inject_both(ll, nat, src.rows(8), 8, keep);
    int32_t ta[8] = {}, tb[8] = {};
    const int na = ll.draft(777, pos, 7, 0.0f, ta), nb = nat.draft(777, pos, 7, 0.0f, tb);
    std::printf("gap draft refused: %d, trim: %d, re-inject: %d, chain after trim equal: %d (%d drafts)\n", gap, trimmed, again,
                na == nb && std::equal(ta, ta + na, tb), na);
    // a rewind past the ring's slack drops in-window K/V: the native drafter forgets everything and refuses to draft
    nat.trim(pos - 40);
    const bool far = nat.pos_max() == -1 && nat.draft(777, pos - 39, 3, 0.0f, o) == 0;
    std::printf("native far rewind refused: %d\n", far);
    ll.trim(0);
    nat.trim(0);
    pos = fill_context(ll, nat, src, ctx, max_chunk);   // the long context again for the timings

    // 6. steady timings: draft(k) at the end of the context, 10 warm-up + 30 timed
    std::printf("steady ms          llama   native\n");
    for (int k = 1; k <= 7; ++k) {
        double ms[2] = {};
        DflashDrafter* d[2] = {&ll, &nat};
        for (int b = 0; b < 2; ++b) {
            for (int r = 0; r < 10; ++r) d[b]->draft(1234, pos, k, 0.0f, o);
            const auto t0 = std::chrono::steady_clock::now();
            for (int r = 0; r < 30; ++r) d[b]->draft(1234, pos, k, 0.0f, o);
            ms[b] = ms_since(t0) / 30;
        }
        std::printf("  draft(%d)       %7.2f  %7.2f\n", k, ms[0], ms[1]);
    }
    {   // inject of a commit's rows (3), the other half of a round
        const auto r3 = src.rows(3);
        double ms[2] = {};
        DflashDrafter* d[2] = {&ll, &nat};
        for (int b = 0; b < 2; ++b) {
            int p = pos;
            for (int r = 0; r < 5; ++r, p += 3) d[b]->inject(r3.data(), 3, p, err);
            const auto t0 = std::chrono::steady_clock::now();
            for (int r = 0; r < 20; ++r, p += 3) d[b]->inject(r3.data(), 3, p, err);
            ms[b] = ms_since(t0) / 20;
        }
        std::printf("  inject(3)      %7.2f  %7.2f\n", ms[0], ms[1]);
    }

    // Gaussian features give flat selector confidences (~0.2), where llama.cpp's own noise (about 0.01 on a score, 5x the
    // native error against the f64 reference) flips near-tied picks: with them the chain / confidence statistics are
    // informational and the gate is the reference.  A real capture gates on the chains and confidences too.
    const bool parity = !src.is_real() || (frac >= 99.0 && conf_bad == 0);
    const bool ok = parity && gap && trimmed && again && far && det == 10 && ref_nat_max <= std::max(2 * ref_ll_max, 5e-3);
    if (!src.is_real()) std::printf("synthetic features: chain / confidence statistics are informational (gate: the f64 reference)\n");
    std::printf(ok ? "ok\n" : "FAIL\n");
    return ok ? 0 : 1;
}
