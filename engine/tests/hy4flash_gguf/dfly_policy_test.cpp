// tests/hy4flash_gguf/dfly_policy_test.cpp - the confidence-gated Dfly window, host only (no device, no model).
#include "strata/hy4flash_gguf/dfly_policy.hpp"

#include <cmath>
#include <cstdio>
#include <string>

namespace {

int failures = 0;
void check(bool ok, const char* what) {
    if (!ok) {
        std::printf("FAIL %s\n", what);
        ++failures;
    }
}

}  // namespace

int main() {
    using namespace strata::hy4_flash;

    // --draft-window's names.
    {
        DflyWindow w = DflyWindow::fixed;
        check(dfly_window_parse("confidence", w) && w == DflyWindow::confidence, "confidence parses");
        check(dfly_window_parse("fixed", w) && w == DflyWindow::fixed, "fixed parses");
        check(!dfly_window_parse("history", w) && w == DflyWindow::fixed, "an unknown window is refused");
        check(std::string(dfly_window_name(DflyWindow::confidence)) == "confidence", "names round-trip");
    }

    // The cost estimate. A cost outlier (a round that captured a graph) moves it by at most alpha_up * (hi - 1).
    {
        DflyCostConfig c;
        DflyWindowCosts costs;
        for (int i = 0; i < 20; ++i) costs.observe(c, 3, 34.0);
        const double before = costs.estimate(c, 3);
        costs.observe(c, 3, 340.0);
        const double after = costs.estimate(c, 3);
        check(std::fabs(before - 34.0) < 1e-6, "a steady cost is its own estimate");
        check(after <= before * (1.0 + c.cost_alpha_up * (c.cost_clamp_hi - 1.0)) + 1e-9, "an outlier is clamped");
        // A width never run is priced from the ratio and the scale of the ones that were.
        const double k1 = costs.estimate(c, 1);
        check(k1 > 0.6 * 34.0 && k1 < 34.0, "an unmeasured width follows the measured scale");
        // Slow first samples do not price a width out: the H20 kv = 1 sequence that stuck near 27 ms with one
        // weight (its steady rounds are 22.5-23 ms) ends near the steady cost.
        DflyWindowCosts s;
        for (int i = 0; i < 10; ++i) s.observe(c, 3, 31.5);
        const double seq[] = {24.9, 24.2, 39.0, 24.9, 24.3, 23.8, 30.5, 24.9, 31.3, 22.8, 22.9, 23.1, 22.7};
        for (double v : seq) s.observe(c, 1, v);
        check(s.estimate(c, 1) < 24.0, "early slow samples do not keep a width priced high");
        // ... while a cost that really rises is followed, if slowly.
        DflyWindowCosts r;
        for (int i = 0; i < 10; ++i) r.observe(c, 2, 28.0);
        for (int i = 0; i < 100; ++i) r.observe(c, 2, 33.0);
        check(r.estimate(c, 2) > 32.5, "a lasting rise is followed");
    }

    // The gate. Verify costs of the H20 model by kv (the rest of the round after the proposal).
    {
        DflyGateConfig c;
        c.eta = 0.0;
        DflyWindowCosts vc;
        DflyConfidenceGate g(c, &vc);
        const float sure[3] = {0.999f, 0.999f, 0.999f}, unsure[3] = {0.05f, 0.05f, 0.05f};
        const float mixed[3] = {0.999f, 0.05f, 0.999f};
        check(g.choose(unsure, 3) == 3, "the first round verifies everything (no rate yet)");
        const double rest[4] = {16.4, 22.8, 26.8, 31.8};
        for (int i = 0; i < 10; ++i) g.observe(sure, 3, 4, 3, rest[3], rest[3] + 3.2);   // 4 tokens a 35 ms round
        check(g.rate() > 0.1 && g.rate() < 0.12, "lambda is the request's tokens per ms");
        for (int kv = 0; kv < 3; ++kv) vc.observe(c.costs, kv, rest[kv]);
        check(g.choose(sure, 3) == 3, "sure drafts: verify all");
        check(g.choose(unsure, 3) == 0, "unsure drafts: verify none");
        check(g.choose(mixed, 3) == 1, "a doubtful second draft cuts the verify there, whatever follows");
        check(g.choose(sure, 2) <= 2 && g.choose(sure, 0) == 0, "never past the drafts");
        check(g.p(0, 0.2f) < g.p(0, 0.5f) && g.p(0, 0.5f) < g.p(0, 0.9f), "p rises with the confidence");
        // A slower request (lambda lower) verifies more of the same doubtful drafts, never less.
        DflyConfidenceGate slow(c, &vc);
        for (int i = 0; i < 10; ++i) slow.observe(sure, 3, 1, 0, rest[3], rest[3] + 3.2);   // 1 token a round
        const float half[3] = {0.6f, 0.6f, 0.6f};
        check(slow.choose(half, 3) >= g.choose(half, 3), "a lower rate makes a draft worth its rows sooner");
    }

    // The per-request bias: learned from the compared slots only.
    {
        DflyGateConfig c;
        c.eta = 0.1;
        DflyConfidenceGate g(c, nullptr);
        const float conf[3] = {0.9f, 0.9f, 0.9f};
        for (int i = 0; i < 20; ++i) g.observe(conf, 3, 1, 0, 30.0, 33.0);   // slot 1 always misses
        check(g.bias() < -0.5, "misses at high confidence lower the bias");
        check(g.bias() >= -c.bias_limit, "the bias stays inside its limit");
        DflyConfidenceGate h(c, nullptr);
        h.observe(conf, 3, 2, 1, 30.0, 33.0);   // slot 1 hit, slot 2 miss (the correction), slot 3 never compared
        DflyConfidenceGate h2(c, nullptr);
        h2.observe(conf, 2, 2, 1, 30.0, 33.0);   // the same round with the verify cut after slot 2
        check(h.bias() == h2.bias(), "a slot after the first mismatch labels nothing");
        DflyConfidenceGate one(c, nullptr);
        one.observe(conf, 3, 2, 1, 30.0, 33.0);
        one.observe(conf, 0, 1, 0, 16.0, 19.0);   // kv = 0: no slot compared, the bias does not move
        check(h.bias() == one.bias(), "a round that verified no draft labels nothing");
        DflyConfidenceGate stop(c, nullptr);
        stop.observe(conf, 3, 1, 1, 30.0, 33.0);   // slot 1 matched and was a stop token: the round ended there
        DflyConfidenceGate hit(c, nullptr);
        hit.observe(conf, 1, 2, 1, 22.0, 25.0);    // slot 1 matched, the bonus row followed
        check(stop.bias() == hit.bias() && stop.bias() > 0.0, "a stop after a match labels only that slot");
    }

    // The override string.
    {
        DflyGateConfig c;
        std::string err;
        check(dfly_gate_config_parse("a=1.5/0.5,b=-1,eta=0,rate_decay=0.8,bias_limit=1,alpha=0.1,alpha_up=0.02", c,
                                     err),
              "a gate override parses");
        check(c.a[0] == 1.5 && c.a[1] == 0.5 && c.a[7] == 0.5 && c.b[0] == -1 && c.b[5] == -1 && c.eta == 0 &&
                  c.rate_decay == 0.8 && c.bias_limit == 1 && c.costs.cost_alpha == 0.1 &&
                  c.costs.cost_alpha_up == 0.02,
              "per-slot lists fill, the last value repeats, every key lands");
        const DflyGateConfig keep = c;
        check(!dfly_gate_config_parse("a=1/2/3/4/5/6/7/8/9", c, err), "too many slots are refused");
        check(!dfly_gate_config_parse("eta=-1", c, err) && c.eta == keep.eta, "a negative eta is refused");
        check(!dfly_gate_config_parse("gain=1", c, err), "an unknown gate key is refused");
        check(!dfly_gate_config_parse("eta", c, err), "a key without a value is refused");
        check(dfly_gate_config_parse("", c, err), "an empty override changes nothing");
    }

    if (failures == 0) std::printf("dfly_policy_test: all passed\n");
    return failures == 0 ? 0 : 1;
}
