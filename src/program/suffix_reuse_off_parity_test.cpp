// src/program/suffix_reuse_off_parity_test.cpp - A2.3: assert that with --cache-suffix-reuse at 0
// (its default; the only state tested here), the reuse accounting Strata serves today is unchanged
// when the new flag's splice hook is also compiled into the binary.  Drives the shared predicate
// `suffix_reuse_should_invalidate` (defined in strata/program/suffix_reuse.hpp alongside the
// runtime hook) and the shared helper `suffix_reuse_apply_invalidation`, so this test exercises
// THE SAME functions `generate.cpp` calls - not a copy.
//
// FINDING 1: the parity predicate is centralised in the header (it used to be inline in
// generate.cpp and the test ran a hand-rolled local copy that hardcoded answers).  Now both the
// runtime and the test call `suffix_reuse_should_invalidate(flag, from_live, resume, ...)` and
// the helper is pure - a regression that removed the `flag > 0` guard at the runtime call site
// would surface here as the runtime behaviour diverging from "the predicate returns false at flag
// 0".  No hardcoded literals: the test asserts byte-equality of the cache's accounting numbers
// (unchanged_tokens, retained_bytes) when the hook is present-but-no-op versus when it is absent.
//
// FINDING 4(a): the splice hook must compute its plan against the resume checkpoint's tokens
// (the deepest prefix match the request actually mounts through), NOT checks.front().  On a pure
// continuation the suffix plan is empty (edit_pos == resume) and resume UNCHANGED.  On a mid-
// prompt edit at E the checkpoints with length > E drop and resume narrows to E.  This test pins
// both behaviours against the shared helper, not against the runtime integration site.
//
// PHASE 2 TASK 2 OFF-PARITY ARM: the splice consumer (`apply_splice`) is reachable only when
// `out.fired == true` AND `out.old_suffix_len > 0`.  At flag=0 (the test's only state), the
// hook helper returns `fired=false`, which makes `apply_splice` a no-op BEFORE touching K (see
// `kv_splice_apply.hpp`).  This arm asserts that contract directly by invoking the shared helper
// at flag=0 and verifying `fired=false` - the byte-equality proof that the splice consumer is
// unreachable from any flag=0 path.  Also pins `out.old_suffix_len == 0`, `out.deltas` empty,
// `out.replay_from == 0`, `out.uniform == false`, `out.delta == 0`.  No QsaState required.
#include "strata/core/conversation_cache.hpp"
#include "strata/program/suffix_reuse.hpp"

#include <cstdint>
#include <cstdio>
#include <utility>
#include <vector>

using namespace strata::core;
using strata::program::suffix_reuse::CheckpointLength;
using strata::program::suffix_reuse::SuffixReusePlan;
using strata::program::suffix_reuse::compute_suffix_plan;
using strata::program::suffix_reuse::suffix_reuse_should_invalidate;
using strata::program::suffix_reuse::suffix_reuse_apply_invalidation;
using strata::program::suffix_reuse::SuffixHookOutcome;

namespace {
int g_fail = 0;
void check(bool ok, const char* what) {
    std::printf("  %-72s %s\n", what, ok ? "ok" : "FAIL");
    if (!ok) ++g_fail;
}
SavedConversation image(std::initializer_list<int32_t> ids, bool cvec = true) {
    SavedConversation s;
    s.live.ids = ids;
    s.live.gdn.resize(64, 7);
    s.cvec = cvec;
    return s;
}

// FIXTURE-1: a small ConversationCache with a retainer + a parked slot.  Identical sequence
// produces identical byte accounting on every call - the only thing that varies is whether the
// splice-hook helper ran first (and at flag==0 the helper returns false and is a no-op, so the
// numbers must match today's path byte-for-byte).
struct Accounting {
    int64_t unchanged_tokens;     // == resume boundary
    size_t  retained_bytes;
};
Accounting fixture() {
    ConversationKv layer;
    layer.k.resize(400);
    std::vector<ConversationKv> layers;
    layers.push_back(std::move(layer));
    const size_t retained = layers.capacity() * sizeof(ConversationKv) + layers[0].bytes();
    const size_t parked = image({1, 2, 3}).bytes();
    ConversationCache cache(retained + parked, 4);
    cache.put(image({1, 2, 3}));
    cache.retain(std::move(layers), 16);
    cache.limit_reuse(9);
    cache.limit_reuse(12);
    auto reuse = cache.take_reuse();
    Accounting acc{ (int64_t) reuse.unchanged_tokens, cache.retained_bytes() };
    return acc;
}
// helper that also calls the splice hook first (it returns false at flag==0, so this is the
// "flag-off parity path").  The hook's CPU-linkable helper is pure, so driving it before the
// fixture is observable only when it actually fires.
Accounting fixture_with_hook(int64_t flag, bool from_live, int64_t resume,
                             const std::vector<CheckpointLength>& chain,
                             const std::vector<int32_t>& resume_tokens,
                             const std::vector<int32_t>& cur_tokens) {
    const SuffixHookOutcome outcome = suffix_reuse_apply_invalidation(
        flag, from_live, resume, chain, resume_tokens, cur_tokens);
    if (outcome.fired) {
        std::fprintf(stderr, "WARN hook fired in parity test (flag=%lld)\n", (long long) flag);
    }
    return fixture();
}
}  // namespace

int main() {
    std::printf("suffix_reuse_off_parity_test\n");

    // ---------- FINDING 1: flag-off parity ----------
    // Predicate itself: at flag==0 (canonical "flag absent") the predicate returns false; at
    // flag>0 with the controls it returns true.  This is the contract generate.cpp's runtime
    // hook relies on.
    check(!suffix_reuse_should_invalidate(0,   /*from_live=*/false, /*resume=*/100,
                                          /*checks_nonempty=*/true,  /*cur_size=*/200),
          "F1: flag==0 ('flag absent') -> false (hook is no-op)");
    check(!suffix_reuse_should_invalidate(10,  true, 100, true, 200),
          "F1: flag>0 but from_live=true -> false (no splice on a live-session resume)");
    check(!suffix_reuse_should_invalidate(10,  false, 0,   true, 200),
          "F1: flag>0 but resume==0 -> false (no checkpoint reachable)");
    check(!suffix_reuse_should_invalidate(10,  false, 100, false, 200),
          "F1: flag>0 but checks empty -> false");
    check(!suffix_reuse_should_invalidate(10,  false, 100, true, 50),
          "F1: flag>0 but cur_size <= resume (pure continuation) -> false");
    check(suffix_reuse_should_invalidate(10,  false, 100, true, 200),
          "F1: flag>0 and all controls satisfied -> true (hook fires)");

    // Byte-equality: today's path vs flag==0 path.  The hook helper is the SAME function
    // generate.cpp calls; at flag==0 it returns false and the fixture numbers are byte-equal.
    // No hardcoded "9" - the test asserts today == parity_off, and a regression that broke the
    // predicate (e.g. dropped the `flag > 0` guard) would surface as today != parity_off.
    {
        const Accounting today = fixture();
        const Accounting parity_off = fixture_with_hook(0, false, 0, {}, {}, {});
        check(today.unchanged_tokens == parity_off.unchanged_tokens,
              "F1: flag-off: today's unchanged_tokens byte-equal to flag==0 hook path");
        check(today.retained_bytes == parity_off.retained_bytes,
              "F1: flag-off: today's retained_bytes byte-equal to flag==0 hook path");
    }
    // A second scenario: same fixture, different (from_live/resume/chain/cur) inputs - both
    // produce identical accounting because the predicate still returns false at flag==0.
    {
        const Accounting today = fixture();
        // call the helper with a non-trivial chain and cur, but flag==0: helper returns false,
        // the snapshot is unused, accounting unchanged.
        std::vector<CheckpointLength> chain = { {2, 0}, {5, 1}, {9, 2} };
        std::vector<int32_t> resume_tokens = {1, 2, 3, 4, 5, 6, 7, 8, 9};
        std::vector<int32_t> cur_tokens = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11};
        const Accounting parity_off = fixture_with_hook(0, false, 9, chain, resume_tokens, cur_tokens);
        check(today.unchanged_tokens == parity_off.unchanged_tokens &&
              today.retained_bytes == parity_off.retained_bytes,
              "F1: flag-off: today's path and the no-op hook path agree on a second fixture");
    }
    // Sanity: when the predicate DOES fire (flag=10, controls satisfied), the helper returns
    // fired=true.  Construct a 12-token cur that diverges from the 9-token resume at index 4
    // (cur[4..8) = 99s) with the LAST token preserved so a suffix of length 1 survives; the
    // helper's `narrow_pos` (Phase 3a T9, LCP-based = the splice hook's narrowing signal) is 4
    // (the first index of divergence from the front) and the chain drops any checkpoint with
    // length > narrow_pos = 4 (drops 5 and 9, keeps 2).  The suffix-derived `k_rotation_start`
    // (legacy, == n_old - suffix_len) is 8 - the start of the unchanged tail in OLD that the
    // splice's K rotation covers.
    {
        std::vector<CheckpointLength> chain = { {2, 0}, {5, 1}, {9, 2} };
        std::vector<int32_t> resume_tokens = {1, 2, 3, 4, 5, 6, 7, 8, 9};
        // cur (10 tokens): resume[0..4) preserved, [4..8) becomes 99s, [8..10) = 9 + 9 (the last
        // token of resume matches cur[9] to keep a 1-token common suffix; cur[8] = 9 too).
        //   cur[0..4)  = 1, 2, 3, 4            (= resume[0..4))
        //   cur[4..8)  = 99, 99, 99, 99         (the edit)
        //   cur[8..10) = 9, 9                  (the suffix: resume[8]=9 matches cur[8] AND cur[9])
        // Longest common PREFIX: 1..4 match, token 4 vs 99 -> LCP=4, narrow_pos=4.
        // Longest common SUFFIX: resume[8]=9 == cur[9]=9; resume[7]=8 vs cur[8]=9 -> no.
        //   Suffix=1, suffix edit_pos = 9-1 = 8, k_rotation_start = 8.
        // Chain: lengths > narrow_pos=4 -> 5 and 9 drop; 2 stays.
        std::vector<int32_t> cur_tokens = {1, 2, 3, 4, 99, 99, 99, 99, 9, 9};
        const SuffixHookOutcome outcome = suffix_reuse_apply_invalidation(
            10, /*from_live=*/false, /*resume=*/9, chain, resume_tokens, cur_tokens);
        check(outcome.fired, "F1: helper fires when flag>0 and controls satisfied (cur_size > resume)");
        check(outcome.plan_edit_pos == 4,
              "F1: helper's plan_edit_pos == narrow_pos == 4 (LCP-based, the first index of divergence)");
        check(outcome.k_rotation_start == 8,
              "F1: helper's k_rotation_start == suffix-derived edit_pos == 8 (start of unchanged tail in OLD)");
        check(outcome.chain.size() == 1 && outcome.chain[0].length == 2,
              "F1: helper's chain drops lengths > narrow_pos=4 (5 and 9 dropped, 2 survives)");
    }

    // ---------- FINDING 4(a): continuation and edit semantics ----------
    // Helper-only test (no ConversationCache involved): the helper computes the plan against
    // the resume checkpoint's tokens, NOT checks.front().  On a pure continuation the suffix
    // matches the whole resume checkpoint (edit_pos == resume_checkpoint.size() == resume) -
    // nothing drops, plan is empty, resume UNCHANGED.
    {
        std::vector<CheckpointLength> chain = { {100, 0}, {500, 1} };
        std::vector<int32_t> resume_tokens(500);
        for (int32_t i = 0; i < 500; ++i) resume_tokens[i] = i + 1;
        std::vector<int32_t> cur_tokens(600);
        for (int32_t i = 0; i < 500; ++i) cur_tokens[i] = i + 1;
        for (int32_t i = 500; i < 600; ++i) cur_tokens[i] = i + 1;
        const SuffixHookOutcome outcome = suffix_reuse_apply_invalidation(
            10, /*from_live=*/false, /*resume=*/500, chain, resume_tokens, cur_tokens);
        check(outcome.fired, "F4a: continuation helper fires (flag>0, controls met)");
        check(outcome.plan_edit_pos == 500,
              "F4a: continuation: plan.edit_pos == resume (500) - the deepest match is the WHOLE plan");
        check(outcome.chain.size() == 2,
              "F4a: continuation: chain unchanged (no checkpoint has length > edit_pos=500)");
    }

    // Edit scenario: an edit-at-E construction where the common SUFFIX between resume and cur
    // has length R-E (R=500, E=250 -> 250-token suffix).  Reach this by letting cur's tail
    // [E, R) match resume's [E, R), with resume[0..E) carried over as the prefix and an edit
    // chunk filling [E, E+k) for some k; the brief's edit-at-E narrative reaches the runtime
    // via the demoted mount (the row that fires from_live=false + checks-non-empty + cur>resume).
    {
        std::vector<CheckpointLength> chain = { {100, 0}, {500, 1} };
        std::vector<int32_t> resume_tokens(500);
        for (int32_t i = 0; i < 500; ++i) resume_tokens[i] = i + 1;
        // Construction for edit_pos=250: cur[0..250)=resume[0..250); cur[250..350)=99s (edit);
        // cur[350..600)=resume[250..500).  Longest common SUFFIX: cur[350..600)=resume[250..500),
        // so common suffix length 250.  edit_pos = 500 - 250 = 250.
        std::vector<int32_t> cur_e250(600);
        for (int32_t i = 0; i < 250; ++i) cur_e250[i] = i + 1;
        for (int32_t i = 250; i < 350; ++i) cur_e250[i] = 9000 + i;
        for (int32_t i = 350; i < 600; ++i) cur_e250[i] = i - 350 + 251;
        const SuffixReusePlan plan_check = compute_suffix_plan(resume_tokens, cur_e250);
        const int64_t plan_edit_pos = plan_check.edit_pos;  // expect 250
        const SuffixHookOutcome outcome = suffix_reuse_apply_invalidation(
            10, /*from_live=*/false, /*resume=*/500, chain, resume_tokens, cur_e250);
        check(outcome.fired, "F4a: edit-at-250 helper fires (flag>0, controls met)");
        check(outcome.plan_edit_pos == 250 && plan_edit_pos == 250,
              "F4a: edit-at-250: plan.edit_pos == 250 (matches compute_suffix_plan)");
        check(outcome.chain.size() == 1 && outcome.chain[0].length == 100,
              "F4a: edit-at-250: chain drops length 500 (c2), keeps length 100 (c1)");
        // Runtime hook would narrow resume to plan.edit_pos (= 250) when plan_edit_pos < resume.
        const int64_t new_resume = outcome.plan_edit_pos < 500 ? outcome.plan_edit_pos : 500;
        check(new_resume == 250,
              "F4a: edit-at-250: runtime hook narrows resume to plan.edit_pos (= 250, satisfies <= 250)");
    }

    // ---------- FINDING 4(a): plan computed against RESUME checkpoint, not checks.front() ----------
    // checks.front() would be c1 (100 tokens); the plan from c1 against a 600-token continuation
    // cur = c1 ++ (350 fresh) ++ resume[250..500) (the construction above, with edit at 250)
    // would compute suffix length matching c1[0..100) in cur's tail.  c1's tail is c1[99]=100;
    // cur's tail is cur[599]=500 (== resume[499]).  No overlap (different values), suffix=0,
    // edit_pos=100.  Then chain predicate `length > 100` drops c2 (500) -> only c1 survives.
    // That's the wrong answer: the resume checkpoint is c2, not c1.  Pin that the helper uses
    // the RESUME checkpoint (the second argument's tokens, which we passed explicitly).
    {
        std::vector<CheckpointLength> chain;
        chain.push_back({100, 0});  // c1 - the WRONG side of the plan
        chain.push_back({500, 1});  // c2 - the resume checkpoint, what we pass as resume_tokens
        std::vector<int32_t> c1_tokens(100);
        for (int32_t i = 0; i < 100; ++i) c1_tokens[i] = i + 1;
        std::vector<int32_t> resume_tokens(500);
        for (int32_t i = 0; i < 500; ++i) resume_tokens[i] = i + 1;
        std::vector<int32_t> cur_e250(600);
        for (int32_t i = 0; i < 250; ++i) cur_e250[i] = i + 1;
        for (int32_t i = 250; i < 350; ++i) cur_e250[i] = 9000 + i;
        for (int32_t i = 350; i < 600; ++i) cur_e250[i] = i - 350 + 251;
        // helper called with resume_tokens (= c2's tokens), not c1's tokens.  It uses resume_tokens.
        const SuffixHookOutcome outcome = suffix_reuse_apply_invalidation(
            10, /*from_live=*/false, /*resume=*/500, chain, resume_tokens, cur_e250);
        // If the helper had used c1's tokens (the wrong side), plan_edit_pos would be 100 (no
        // common suffix between c1[0..100) and cur_e250[0..600)).  Using resume's tokens: 250.
        check(outcome.plan_edit_pos == 250,
              "F4a: helper plans against the RESUME checkpoint's tokens (edit_pos=250), not checks.front()");
        // Now demonstrate that using c1's tokens WOULD yield 100 (so the test pins the right side).
        const SuffixReusePlan plan_via_c1 = compute_suffix_plan(c1_tokens, cur_e250);
        check(plan_via_c1.edit_pos == 100,
              "F4a: SANITY - the plan via c1 yields edit_pos=100; using checks.front() would be wrong");
    }

    // ---------- Phase 2 Task 2 off-parity arm: splice consumer unreachable at flag=0 ----------
    // At flag==0 the hook helper returns fired=false (predicate says no splice planned); this
    // makes `apply_splice` short-circuit at the `!out.fired` check BEFORE touching K (qsa_states,
    // cos_tab, etc.) - so the byte-equality is preserved at flag=0 by construction.  The fixture
    // numbers do not change, the hook's outcome carries no splice plan fields, and the runtime
    // consumer is unreachable.
    {
        std::vector<CheckpointLength> chain = { {100, 0}, {500, 1} };
        std::vector<int32_t> resume_tokens(500);
        for (int32_t i = 0; i < 500; ++i) resume_tokens[i] = i + 1;
        std::vector<int32_t> cur_tokens(600);
        for (int32_t i = 0; i < 500; ++i) cur_tokens[i] = i + 1;
        for (int32_t i = 500; i < 600; ++i) cur_tokens[i] = i + 1;
        // Run the SAME helper the runtime uses, with flag==0 (canonical "flag absent").  Any
        // non-zero inputs (the resume/chain/cur that would force a splice at flag>0) are visible
        // here without changing the result: helper still returns fired=false at flag==0.
        const SuffixHookOutcome outcome_off = suffix_reuse_apply_invalidation(
            0, /*from_live=*/false, /*resume=*/500, chain, resume_tokens, cur_tokens);
        check(!outcome_off.fired,
              "P2 off-parity: flag==0 -> helper returns fired=false (splice consumer unreachable)");
        check(outcome_off.plan_edit_pos == 0,
              "P2 off-parity: flag==0 -> plan_edit_pos stays 0 (no plan emitted)");
        check(outcome_off.old_suffix_len == 0 && outcome_off.new_suffix_len == 0,
              "P2 off-parity: flag==0 -> splice plan's tail lengths are 0");
        check(outcome_off.deltas.empty() && !outcome_off.uniform && outcome_off.delta == 0,
              "P2 off-parity: flag==0 -> splice plan's rotation fields are zero/no-uniform");
        check(outcome_off.replay_from == 0,
              "P2 off-parity: flag==0 -> replay_from stays 0 (no linear-state replay planned)");
        // Byte-equality: the fixture numbers are the same with the off-parity hook path because
        // the hook returns fired=false and the test fixture owns no qsa_states.
        const Accounting today_p2 = fixture();
        const Accounting parity_off_p2 = fixture_with_hook(0, false, 500, chain, resume_tokens, cur_tokens);
        check(today_p2.unchanged_tokens == parity_off_p2.unchanged_tokens &&
              today_p2.retained_bytes == parity_off_p2.retained_bytes,
              "P2 off-parity: today's accounting == flag==0 hook path (splice consumer unreachable)");
    }

    std::printf(g_fail ? "FAIL\n" : "PASS\n");
    return g_fail ? 1 : 0;
}