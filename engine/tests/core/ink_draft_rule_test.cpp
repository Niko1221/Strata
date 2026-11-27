// Unit test of the ONE drafter on/off rule (strata/core/ink_draft_rule.hpp): residency first at load time, plain steps by context on hybrid cards.
#include <cstdio>
#include <cstring>
#include <string>

#include "strata/core/ink_draft_rule.hpp"

using namespace strata::core;

#define CHECK(c) do { if (!(c)) { std::printf("FAIL line %d: %s\n", __LINE__, #c); return 1; } } while (0)

int main() {
    // full residency without the drafter, hybrid with it: vetoed (the RX 7900 XT case: 536 MiB)
    {
        InkDraftLoadIn in; in.full_without = true; in.full_with = false; in.drafter_mib = 536; in.tier_budget_mib = 0;
        const InkDraftLoadOut o = ink_draft_load_rule(in);
        CHECK(!o.load && o.residency_veto && o.to_hybrid);
        // asked for by name: loaded
        in.user_on = true;
        const InkDraftLoadOut f = ink_draft_load_rule(in);
        CHECK(f.load && !f.residency_veto);
        // --draft off always wins
        in.user_on = false; in.user_off = true;
        CHECK(!ink_draft_load_rule(in).load && !ink_draft_load_rule(in).residency_veto);
    }
    // full residency with the drafter as well: loaded
    {
        InkDraftLoadIn in; in.full_without = true; in.full_with = true; in.drafter_mib = 432;
        const InkDraftLoadOut o = ink_draft_load_rule(in);
        CHECK(o.load && !o.residency_veto);
    }
    // hybrid card: a small tier cut is accepted, a cut of 10% or more is vetoed
    {
        InkDraftLoadIn in; in.full_without = false; in.full_with = false; in.tier_budget_mib = 6000; in.drafter_mib = 432;   // 7.2%
        CHECK(ink_draft_load_rule(in).load);
        in.tier_budget_mib = 3000;                                                                                           // 14.4%
        const InkDraftLoadOut o = ink_draft_load_rule(in);
        CHECK(!o.load && o.residency_veto && !o.to_hybrid && o.tier_cut > 0.14 && o.tier_cut < 0.15);
        in.user_on = true;
        CHECK(ink_draft_load_rule(in).load);
    }
    // below sm_80 (principle 1b): a hybrid load never gets the default drafter, a fully resident card keeps it, --draft on loads it, --draft off wins
    {
        InkDraftLoadIn in; in.below_sm80 = true; in.full_without = false; in.full_with = false; in.tier_budget_mib = 9000; in.drafter_mib = 450;   // a 5% tier cut: fine above sm_80
        const InkDraftLoadOut o = ink_draft_load_rule(in);
        CHECK(!o.load && o.residency_veto);
        in.below_sm80 = false;
        CHECK(ink_draft_load_rule(in).load);                              // the same load above sm_80 is accepted
        in.below_sm80 = true; in.user_on = true;
        CHECK(ink_draft_load_rule(in).load);                              // --draft on
        in.user_on = false; in.user_off = true;
        CHECK(!ink_draft_load_rule(in).load && !ink_draft_load_rule(in).residency_veto);
        InkDraftLoadIn res; res.below_sm80 = true; res.full_without = true; res.full_with = true; res.drafter_mib = 450;   // a V100 32 GB
        CHECK(ink_draft_load_rule(res).load && !ink_draft_load_rule(res).residency_veto);
    }
    // the Tesla P100 (sm_60) exception of principle 1b: a hybrid load keeps the default drafter; sm_61 (GTX 10), sm_70 hybrid and sm_75 (RTX 20) stay vetoed; the other principles still apply
    {
        InkDraftLoadIn p; p.below_sm80 = true; p.sm60 = true; p.full_without = false; p.full_with = false; p.tier_budget_mib = 10500; p.drafter_mib = 728;   // the P100 SQ-20 case: a 6.9% tier cut
        CHECK(ink_draft_load_rule(p).load && !ink_draft_load_rule(p).residency_veto);
        InkDraftLoadIn g = p; g.sm60 = false;                              // GTX 10 (sm_61), hybrid V100 (sm_70) or RTX 20 (sm_75): vetoed as before
        CHECK(!ink_draft_load_rule(g).load && ink_draft_load_rule(g).residency_veto);
        InkDraftLoadIn big = p; big.tier_budget_mib = 3000;                // principle 1 still applies to the P100: a 24% tier cut is vetoed
        CHECK(!ink_draft_load_rule(big).load && ink_draft_load_rule(big).residency_veto);
        InkDraftLoadIn tohy = p; tohy.full_without = true; tohy.full_with = false;   // a resident model the drafter would push to hybrid: vetoed
        CHECK(!ink_draft_load_rule(tohy).load && ink_draft_load_rule(tohy).to_hybrid);
        InkDraftLoadIn off = p; off.user_off = true;                       // --draft off wins
        CHECK(!ink_draft_load_rule(off).load);
        InkDraftLoadIn res = p; res.full_without = true; res.full_with = true;       // a resident P100 load: loaded
        CHECK(ink_draft_load_rule(res).load);
        CHECK(ink_draft_plain_step(true, kDraftPlainCtxHybrid, false));    // principle 2 is unchanged on the hybrid P100
    }
    // the printed line: the same sentence start on every backend, the override named
    {
        const std::string s = ink_draft_veto_line("[InkCUDA]", "DFlash 2", 19.3, 14.2, 536, 14.7);
        CHECK(s.find("DFlash 2 not loaded: it would move part of the model off the card") != std::string::npos);
        CHECK(s.find("`--draft on` loads it anyway.") != std::string::npos);
        CHECK(s.find("536 MiB") != std::string::npos);
    }
    // plain steps by context: only on hybrid cards, only from the limit on, never with --draft on
    {
        CHECK(!ink_draft_plain_step(false, 100000, false));                 // fully resident: the window governor decides alone
        CHECK(!ink_draft_plain_step(true, kDraftPlainCtxHybrid - 1, false));
        CHECK(ink_draft_plain_step(true, kDraftPlainCtxHybrid, false));
        CHECK(ink_draft_plain_step(true, 32768, false));
        CHECK(!ink_draft_plain_step(true, 32768, true));
        // the measured cases: a 62-token prompt keeps drafting on a hybrid card, a 4K document does not (RTX 5070 12 GB: 34.5 against 59.2 tok/s)
        CHECK(!ink_draft_plain_step(true, 62, false));
        CHECK(ink_draft_plain_step(true, 4094, false));
        // an emulated-draft architecture (sm < 80) that is fully resident: plain from kDraftPlainCtxSlowArch; sm_80 and newer resident: never
        CHECK(!ink_draft_plain_step(false, kDraftPlainCtxSlowArch - 1, false, 70));
        CHECK(ink_draft_plain_step(false, kDraftPlainCtxSlowArch, false, 70));
        CHECK(!ink_draft_plain_step(false, 100000, false, 80));
        CHECK(!ink_draft_plain_step(false, 100000, false, 120));
        CHECK(!ink_draft_plain_step(false, 100000, true, 70));              // --draft on
    }
    std::printf("ink_draft_rule_test: ok\n");
    return 0;
}
