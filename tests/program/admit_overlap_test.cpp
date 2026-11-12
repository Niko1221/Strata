// admit_overlap.hpp: which batch admissions read beside the decoding slot groups (STRATA_ADMIT_OVERLAP), the RAM-copy
// coverage rule and the counters (synthetic: no model, no GPU).
#include "strata/program/admit_overlap.hpp"

#include <cstdio>
#include <cstring>
#include <utility>
#include <vector>

namespace ao = strata::program::admit_overlap;

static int fails = 0;
#define CHECK(c) do { if (!(c)) { std::printf("FAIL %s:%d %s\n", __FILE__, __LINE__, #c); ++fails; } } while (0)

static ao::Read ready() {   // everything that lets a read go beside the slots
    ao::Read r;
    r.on = true;
    r.piped = true;
    r.admission = true;
    r.slots_decoding = true;
    r.tokens = 1300;
    return r;
}

int main() {
    // the env
    CHECK(!ao::env_on(nullptr));
    CHECK(!ao::env_on(""));
    CHECK(!ao::env_on("0"));
    CHECK(ao::env_on("1"));
    CHECK(ao::env_on("2"));
    CHECK(ao::env_min_tokens(nullptr) == ao::kDefaultMinTokens);
    CHECK(ao::env_min_tokens("") == ao::kDefaultMinTokens);
    CHECK(ao::env_min_tokens("0") == 0);
    CHECK(ao::env_min_tokens("512") == 512);
    CHECK(ao::env_min_tokens("-5") == 0);

    // off is off whatever else holds (the serial path, p56/ref's bytes)
    {
        ao::Read r = ready();
        r.on = false;
        CHECK(ao::decide(r) == ao::Why::off);
        ao::Read z;   // a default Read is off
        CHECK(ao::decide(z) == ao::Why::off);
    }
    CHECK(ao::decide(ready()) == ao::Why::none);

    // each fact alone sends the read the serial way, named
    struct Case { void (*set)(ao::Read&); ao::Why want; };
    const Case cases[] = {
        {[](ao::Read& r) { r.piped = false; }, ao::Why::not_piped},
        {[](ao::Read& r) { r.admission = false; }, ao::Why::solo},
        {[](ao::Read& r) { r.legacy = true; }, ao::Why::legacy},
        {[](ao::Read& r) { r.setup_off = true; }, ao::Why::setup},
        {[](ao::Read& r) { r.pad_rows = true; }, ao::Why::pad_rows},
        {[](ao::Read& r) { r.runtime_off = true; }, ao::Why::file_reads},
        {[](ao::Read& r) { r.slots_decoding = false; }, ao::Why::idle},
        {[](ao::Read& r) { r.tokens = 10; }, ao::Why::short_read},
        {[](ao::Read& r) { r.tokens = 0; r.min_tokens = 0; }, ao::Why::short_read},
        {[](ao::Read& r) { r.pictures = true; }, ao::Why::pictures},
        {[](ao::Read& r) { r.co_read = true; }, ao::Why::co_read},
        {[](ao::Read& r) { r.elastic_kv = true; }, ao::Why::elastic_kv},
        {[](ao::Read& r) { r.cpu_share = true; }, ao::Why::cpu_share},
        {[](ao::Read& r) { r.swaps_pending = true; }, ao::Why::swaps_pending},
        {[](ao::Read& r) { r.loan_out = true; }, ao::Why::loan},
        {[](ao::Read& r) { r.source_ok = false; }, ao::Why::source},
    };
    for (const Case& c : cases) {
        ao::Read r = ready();
        c.set(r);
        CHECK(ao::decide(r) == c.want);
        CHECK(std::strlen(ao::why_text(c.want)) > 0);
    }
    // the minimum is inclusive; 0 lets any non-empty read go
    {
        ao::Read r = ready();
        r.tokens = r.min_tokens = 64;
        CHECK(ao::decide(r) == ao::Why::none);
        r.tokens = 63;
        CHECK(ao::decide(r) == ao::Why::short_read);
        r.min_tokens = 0;
        r.tokens = 1;
        CHECK(ao::decide(r) == ao::Why::none);
    }
    // the loan wins over a later reason (the loan is the one that changes the windows' bits)
    {
        ao::Read r = ready();
        r.loan_out = true;
        r.source_ok = false;
        CHECK(ao::decide(r) == ao::Why::loan);
    }

    // coverage: 3 layers x 4 experts; slots 0..9 per layer's cache; layer 1's lend floor at slot 5
    {
        const int64_t L = 3, E = 4;
        std::vector<int32_t> res = {0, 1, -1, 2,    // layer 0: expert 2 not resident
                                    3, 6, 4, -1,    // layer 1: expert 1 in slot 6 (>= floor 5: lendable), expert 3 not resident
                                    0, 1, 2, 3};    // layer 2: all resident
        std::vector<int32_t> floor = {-1, 5, -1};
        std::vector<char> ram(L * E, 0);
        auto has = [&](int64_t l, int64_t e) { return ram[(size_t) (l * E + e)] != 0; };
        // nothing in RAM: the first pair asked for is (0, 2)
        ao::Coverage c = ao::covered(res.data(), L, E, floor.data(), has);
        CHECK(!c.covers && c.at_l == 0 && c.at_e == 2);
        // the three that can be asked for in RAM: covered, need 3
        ram[0 * E + 2] = ram[1 * E + 1] = ram[1 * E + 3] = 1;
        c = ao::covered(res.data(), L, E, floor.data(), has);
        CHECK(c.covers && c.need == 3);
        // the lendable one missing: not covered (a loan would ask for it)
        ram[1 * E + 1] = 0;
        c = ao::covered(res.data(), L, E, floor.data(), has);
        CHECK(!c.covers && c.at_l == 1 && c.at_e == 1);
        // no floors: a resident expert is never asked for, whatever its slot
        c = ao::covered(res.data(), L, E, nullptr, has);
        CHECK(c.covers && c.need == 2);
        // everything resident, nothing lendable: need 0
        std::vector<int32_t> all = {0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3};
        c = ao::covered(all.data(), L, E, nullptr, [](int64_t, int64_t) { return false; });
        CHECK(c.covers && c.need == 0);
    }

    // the counters and their line
    {
        ao::Stats s;
        CHECK(s.serial_total() == 0);
        s.note_serial(ao::Why::none);   // not a serial read
        s.note_serial(ao::Why::off);    // not counted in the total (the mode was off)
        s.note_serial(ao::Why::loan);
        s.note_serial(ao::Why::loan);
        s.note_serial(ao::Why::idle);
        CHECK(s.serial_total() == 3);
        s.note_beside(1300, 1100.0, 220);
        s.note_beside(1500, 900.0, 180);
        CHECK(s.beside == 2 && s.beside_tokens == 2800 && s.beside_rows == 400);
        const std::string t = s.text();
        CHECK(t.find("2 reads beside the slots") != std::string::npos);
        CHECK(t.find("200.0 rows/s") != std::string::npos);
        CHECK(t.find("lent cache slots 2") != std::string::npos);
        CHECK(t.find("no slot decoding 1") != std::string::npos);
        std::printf("%s\n", t.c_str());
    }
    // the summary's cadence: 1, 2, 4, ..., 64, then every 64th
    {
        std::vector<int64_t> at;
        for (int64_t n = 0; n <= 300; ++n)
            if (ao::say_now(n)) at.push_back(n);
        const std::vector<int64_t> want = {1, 2, 4, 8, 16, 32, 64, 128, 192, 256};
        CHECK(at == want);
    }

    // the verifiers' stream priority (verify.cpp make_stream): STRATA_PIPED_CONCURRENT as upstream, STRATA_ADMIT_OVERLAP
    // alone never (the solo path's windows share those streams), STRATA_ADMIT_OVERLAP_PRIO=1 as an A/B
    {
        auto top = [](const char* pc, const char* pcp, const char* ao_, const char* aop) {
            return ao::windows_top_priority(pc, pcp, ao_, aop);
        };
        CHECK(!top(nullptr, nullptr, nullptr, nullptr));       // nothing set: the default priority
        CHECK(!top(nullptr, nullptr, "1", nullptr));           // the regression of 03.11: off now
        CHECK(!top(nullptr, nullptr, "1", "0"));
        CHECK(!top(nullptr, nullptr, "1", ""));
        CHECK(top(nullptr, nullptr, "1", "1"));                // the A/B arm
        CHECK(!top(nullptr, nullptr, nullptr, "1"));           // PRIO alone: no mode, nothing raised
        CHECK(!top(nullptr, nullptr, "0", "1"));
        CHECK(!top(nullptr, "0", "1", "1"));                   // STRATA_PIPED_CONCURRENT_PRIO=0 wins
        CHECK(top("1", nullptr, nullptr, nullptr));            // upstream's STRATA_PIPED_CONCURRENT, unchanged
        CHECK(top("1", nullptr, "1", nullptr));
        CHECK(top("1", "1", nullptr, nullptr));
        CHECK(!top("1", "0", nullptr, nullptr));
        CHECK(!top("1", "", nullptr, nullptr));                // (upstream: an empty PRIO is 0)
        CHECK(!top("0", nullptr, nullptr, nullptr));
        CHECK(!top("", nullptr, nullptr, nullptr));
    }

    // STRATA_ADMIT_OVERLAP_LONG: the env, the decision with a loan out, the lend floors and covered()'s lower bound
    {
        CHECK(ao::env_long_min_tokens(nullptr) == ao::kDefaultLongMinTokens);
        CHECK(ao::env_long_min_tokens("") == ao::kDefaultLongMinTokens);
        CHECK(ao::env_long_min_tokens("32768") == 32768);
        CHECK(ao::env_long_min_tokens("-1") == 0);
        ao::Read r = ready();
        r.loan_out = true;
        CHECK(ao::decide(r) == ao::Why::loan);                  // off: the loan keeps it serial, as before
        r.long_on = true;
        r.tokens = ao::kDefaultLongMinTokens - 1;
        CHECK(ao::decide(r) == ao::Why::loan);                  // a short read with a (kept) loan: still serial
        r.tokens = ao::kDefaultLongMinTokens;
        CHECK(ao::decide(r) == ao::Why::none);                  // long, covered: beside the slots
        r.loan_ok = false;
        CHECK(ao::decide(r) == ao::Why::loan_uncovered);        // long, a lent expert without its RAM copy
        CHECK(std::strlen(ao::why_text(ao::Why::loan_uncovered)) > 0);
        r.loan_ok = true;
        r.loan_out = false;
        r.tokens = 100;
        CHECK(ao::decide(r) == ao::Why::none);                  // no loan: the long switch changes nothing
        // the other reasons still come first (the order of the rule)
        ao::Read q = ready();
        q.loan_out = true;
        q.long_on = true;
        q.tokens = 200000;
        q.loan_ok = false;
        q.cpu_share = true;
        CHECK(ao::decide(q) == ao::Why::cpu_share);
        q.cpu_share = false;
        q.swaps_pending = true;
        CHECK(ao::decide(q) == ao::Why::swaps_pending);
        q.swaps_pending = false;
        q.slots_decoding = false;
        CHECK(ao::decide(q) == ao::Why::idle);
        q.slots_decoding = true;
        q.loan_ok = true;
        q.source_ok = false;
        CHECK(ao::decide(q) == ao::Why::source);                // the non-lent source rule after the loan's
        q.long_min = 0;
        q.tokens = 0;
        q.min_tokens = 0;
        CHECK(ao::decide(q) == ao::Why::short_read);
        CHECK(!ao::long_loan_allowed(q));                        // an empty read is never "long"

        // loan_floor: each lent part's first slot on its layers, -1 elsewhere (a part with no loan, out of range)
        const std::vector<int32_t> f = ao::loan_floor(6, {{0, 2, 100}, {2, 4, -1}, {4, 9, 50}});
        CHECK((f == std::vector<int32_t>{100, 100, -1, -1, 50, 50}));
        CHECK(ao::loan_floor(0, {{0, 2, 5}}).empty());
        // covered() with the floors: a resident expert at or above its layer's floor is lent (needs RAM), below is not;
        // a lent expert marked missing by lend() (host_res -1) needs RAM too - the bound is inclusive
        const int64_t L = 3, E = 4;
        std::vector<int32_t> res = {10, 99, 100, -1,    // layer 0 floor 100: 100 lent, -1 missing
                                    5, 6, 7, 8,         // layer 1 floor -1: all resident, never lent
                                    49, 50, 51, 0};     // layer 2 floor 50: 50, 51 lent
        const std::vector<int32_t> fl = ao::loan_floor(L, {{0, 1, 100}, {2, 3, 50}});
        std::vector<std::pair<int64_t, int64_t>> asked;
        auto rec = [&](int64_t l, int64_t e) { asked.emplace_back(l, e); return true; };
        ao::Coverage c = ao::covered(res.data(), L, E, fl.data(), rec);
        CHECK(c.covers && c.need == 4);
        CHECK((asked == std::vector<std::pair<int64_t, int64_t>>{{0, 2}, {0, 3}, {2, 1}, {2, 2}}));
        c = ao::covered(res.data(), L, E, fl.data(), [](int64_t l, int64_t e) { return !(l == 2 && e == 2); });
        CHECK(!c.covers && c.at_l == 2 && c.at_e == 2);         // the first lent expert without RAM is named
        c = ao::covered(res.data(), L, E, nullptr, [](int64_t l, int64_t e) { return !(l == 2 && e == 2); });
        CHECK(c.covers && c.need == 1);                          // without the floors only the missing one is asked
        // after lend() the lent ones are -1 already: with or without floors the same answer
        std::vector<int32_t> lent = res;
        for (int64_t l = 0; l < L; ++l)
            for (int64_t e = 0; e < E; ++e) {
                int32_t& x = lent[(size_t) (l * E + e)];
                if (fl[(size_t) l] >= 0 && x >= fl[(size_t) l]) x = -1;
            }
        CHECK(ao::covered(lent.data(), L, E, nullptr, [](int64_t, int64_t) { return true; }).need == 4);
        CHECK(ao::covered(lent.data(), L, E, fl.data(), [](int64_t, int64_t) { return true; }).need == 4);
    }

    if (fails == 0) std::printf("admit_overlap_test: OK\n");
    return fails == 0 ? 0 : 1;
}
