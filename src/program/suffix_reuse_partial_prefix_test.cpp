// src/program/suffix_reuse_partial_prefix_test.cpp - phase 3b: T12-T15.  The partial-prefix
// mount's pure helpers and the flag-gated `best_with_partial` path.  Header-only / CPU - no
// CUDA, no <cuda_runtime.h>, no QsaState.  Pins the exact behaviour the runtime integration
// site in generate.cpp relies on.
//
// CONTEXT.  Phase 3a shipped the strict-mount splice (T9 continuation parity + T10 anchor
// engagement).  On ContextBench traffic the strict mount never fires for the 44/71 mid-file-
// edit requests because the previous conversation's END is no longer a strict prefix of the
// new prompt.  Phase 3b adds a flag-gated partial-prefix mount: when the strict mount fails,
// find the checkpoint C with the longest LCP P (0 < P < L) and mount C.  The K/V cells for
// [0, P) are valid from the restore, the chunk loop reads [P, n), and the existing splice
// machinery handles the K rotation for the unchanged tail [P, L) and the prefill of [P, n).
//
// OFF-FLAG BYTE-IDENTITY.  `ConversationCache::best_with_partial(0, ...)` MUST return the
// same Match as `best()` (the same loop, same `conversation_prefix` call, same tie-break).
// The new `partial` and `checkpoint_length` fields default to false / 0 on the strict path,
// so downstream consumers that don't read them see no change.  T13 pins this directly.
//
// T12 PARTIAL-PREFIX MATCH.  Checkpoint {A B C D} (length 4), prompt {A B X Y} (length 4).
// Strict match: 0.  Partial match: LCP = 2 (matches A, B; diverges at index 2).  Asserts:
//   - best_with_partial(flag>0, prompt, ...) returns Match{index, tokens=2, partial=true,
//     checkpoint_length=4, live=appropriate}
//   - suffix_reuse_partial_apply(C.tokens, prompt, L=4, P=2, chain) returns:
//       fired = true
//       plan_edit_pos = 2 (= the LCP)
//       restore_anchor_length = 4 (= the full checkpoint's length)
//       old_suffix_len = 0 (C's tail [C D] vs prompt's tail [X Y] - no common suffix)
//       chain drops any entry with length > 2
//
// T12 variant: partial match WITH a common tail (the K rotation engages - pure insert of 1
// token at position 2: OLD = {1 2 3 4 5} (5), NEW = {1 2 90 3 4 5} (6), LCP=2, suffix=3,
// uniform=true, delta=+1).
//
// T13 OFF-FLAG PARITY.  Same inputs as T12 with flag = 0:
//   - best_with_partial(0, ...) returns the same as best() - strict search, 0 tokens
//   - the partial helper is NOT called (the runtime only calls it when the cache returned
//     a partial match - at flag=0 the cache never returns a partial match)
//
// T14 BEST-OF-N.  Multiple checkpoints with different LCPs - the longest LCP wins.  Strict
// match still preferred over partial (strict > partial always, even if strict is shorter).
//
// T15 IMAGES / GEOMETRY MISMATCH.  Partial match is REFUSED when images mismatch at positions
// below the LCP: a checkpoint with an image at position 1 cannot mount a prompt that doesn't
// have that image (the LCP is 2 but the image is at position 1 which is below the LCP).  The
// fail-closed case: no mount, no error.

#include "strata/core/conversation_cache.hpp"
#include "strata/program/suffix_reuse.hpp"

#include <cstdint>
#include <cstdio>
#include <utility>
#include <vector>

using namespace strata::core;
using strata::program::suffix_reuse::CheckpointLength;
using strata::program::suffix_reuse::SuffixHookOutcome;
using strata::program::suffix_reuse::suffix_reuse_partial_apply;
using strata::program::suffix_reuse::compute_suffix_plan;

namespace {
int g_fail = 0;
int g_pass = 0;
void check(bool ok, const char* what) {
    std::printf("  %-90s %s\n", what, ok ? "ok" : "FAIL");
    if (ok) ++g_pass; else ++g_fail;
}
SavedConversation image(std::initializer_list<int32_t> ids, bool cvec = true) {
    SavedConversation s;
    s.live.ids = ids;
    s.live.gdn.resize(64, 7);
    s.cvec = cvec;
    return s;
}
ConversationCheckpoint ckpt(std::initializer_list<int32_t> ids) {
    ConversationCheckpoint c;
    c.ids = ids;
    c.gdn.resize(64, 1);
    return c;
}
}  // namespace

int main() {
    std::printf("suffix_reuse_partial_prefix_test (T12 partial-prefix mount, T13 off-flag parity, T14 best-of-N, T15 image mismatch)\n");
    // ============================================================
    // T12: PARTIAL-PREFIX MATCH
    // ============================================================
    // Checkpoint {A B C D} (length 4).  Prompt {A B X Y} (length 4).
    // Strict match: 0 (prompt[2]=X != C, not a prefix).
    // Partial match: LCP = 2 (A, B match; index 2 diverges: C vs X).
    // Common tail from back: [C, D] vs [X, Y] -> 0 (no match).
    // Splice plan: narrow_pos = 2, old_suffix_len = 0, uniform = false (no deltas to compute).
    {
        auto s = image({1, 2, 3, 4});
        s.checkpoints = {ckpt({1, 2, 3, 4})};
        ConversationCache cache(1 << 20, 4);
        cache.put(std::move(s));

        const std::vector<int32_t> prompt = {1, 2, 90, 91};   // [A B X Y]
        // best() (strict): no match (the live and checkpoint are full of {1,2,3,4}, prompt diverges at 2).
        const auto strict = cache.best(prompt, {}, true);
        check(!strict.partial && strict.tokens == 0,
              "T12 setup: strict best() returns no match for partial-prefix shape");

        // best_with_partial(flag=64, ...): partial match, LCP=2, ckpt length=4.
        const auto m = cache.best_with_partial(64, prompt, {}, true);
        check(m.partial,
              "T12.1: best_with_partial(flag>0) returns partial=true for {A B C D} / {A B X Y}");
        check(m.tokens == 2,
              "T12.2: match.tokens == 2 (the LCP)");
        check(m.checkpoint_length == 4,
              "T12.3: match.checkpoint_length == 4 (the full checkpoint's length)");
        check(m.tokens < m.checkpoint_length,
              "T12.4: LCP < checkpoint_length (the partial-mount invariant)");

        // The partial helper: fire, plan, anchor.
        const std::vector<int32_t> ck_tokens = {1, 2, 3, 4};
        std::vector<CheckpointLength> chain = { {4, 0} };
        const SuffixHookOutcome outcome = suffix_reuse_partial_apply(
            ck_tokens, prompt, /*L=*/4, /*P=*/2, chain);
        check(outcome.fired,
              "T12.5: partial_apply fired (valid partial shape)");
        check(outcome.plan_edit_pos == 2,
              "T12.6: outcome.plan_edit_pos == 2 (the LCP)");
        check(outcome.restore_anchor_length == 4,
              "T12.7: outcome.restore_anchor_length == 4 (the partial-mount anchor)");
        check(outcome.old_suffix_len == 0,
              "T12.8: outcome.old_suffix_len == 0 ([C D] vs [X Y] - no common tail)");
        check(outcome.new_suffix_len == 0,
              "T12.9: outcome.new_suffix_len == 0 (no common tail)");
        check(!outcome.uniform,
              "T12.10: outcome.uniform == false (no deltas to compute; gate is on old_suffix_len > 0)");
        check(outcome.chain.size() == 1 && outcome.chain[0].length == 4,
              "T12.11: chain keeps the anchor (length 4 == L); non-anchor entries with length > P are dropped");

        // The runtime's narrowing predicate at generate.cpp:9074: `if (outcome.plan_edit_pos < resume)`.
        // For the partial case, resume is set to P (= 2) by the runtime before the splice hook runs;
        // plan_edit_pos == 2 == resume, so the predicate is false.  The runtime uses
        // `restore_anchor_length` to set resume = L (= 4) AFTER the splice hook returns.
        // (The runtime's flow: cache.best_with_partial -> set resume=P -> call splice -> read restore_anchor_length -> set resume=L -> restore C -> chunk loop reads P.)
        // We pin the helper's contract: plan_edit_pos == P, restore_anchor_length == L.
        check(outcome.plan_edit_pos != outcome.restore_anchor_length,
              "T12.12: partial case - plan_edit_pos (P) != restore_anchor_length (L) [sentinel]");

        // The K rotation gate at generate.cpp:9089: `if (outcome.old_suffix_len > 0)`.
        // For this shape, old_suffix_len == 0, so apply_splice is NOT called - no K rotation.
        // The prefill recomputes K+V for [P, n) anyway.  The K rotation is wasted in this shape.
        const bool runtime_would_rotate = outcome.old_suffix_len > 0;
        check(!runtime_would_rotate,
              "T12.13: runtime does NOT call apply_splice (old_suffix_len == 0)");
    }

    // ============================================================
    // T12 variant: partial match WITH a common tail (K rotation engages - pure insert of 1)
    // ============================================================
    // OLD = {A B C D E} (length 5), NEW = {A B X C D E} (length 6 - pure insert at position 2).
    // LCP = 2 (A, B match; index 2 diverges: C vs X).
    // Common tail from back: [C, D, E] vs [C, D, E] -> 3.
    // The K rotation rotates K rows at OLD positions 2, 3, 4 to NEW positions 3, 4, 5 (delta=+1).
    {
        const std::vector<int32_t> ck_tokens = {1, 2, 3, 4, 5};     // OLD: [A B C D E]
        const std::vector<int32_t> prompt    = {1, 2, 90, 3, 4, 5}; // NEW: [A B X C D E]
        std::vector<CheckpointLength> chain;
        const SuffixHookOutcome outcome = suffix_reuse_partial_apply(
            ck_tokens, prompt, /*L=*/5, /*P=*/2, chain);
        check(outcome.fired, "T12v.1: partial_apply fired (LCP=2, suffix=3)");
        check(outcome.plan_edit_pos == 2, "T12v.2: plan_edit_pos == 2 (LCP)");
        check(outcome.restore_anchor_length == 5, "T12v.3: restore_anchor_length == 5 (anchor)");
        check(outcome.old_suffix_len == 3,
              "T12v.4: old_suffix_len == 3 ([C D E] common tail in OLD)");
        check(outcome.new_suffix_len == 3,
              "T12v.5: new_suffix_len == 3 ([C D E] common tail in NEW)");
        check(outcome.uniform,
              "T12v.6: uniform == true (pure insert of 1, all deltas == +1)");
        check(outcome.delta == 1,
              "T12v.7: delta == +1 (insert of X shifts C,D,E from OLD [2,4) to NEW [3,5))");
        check(outcome.deltas.empty(),
              "T12v.8: uniform -> deltas table empty (single rotation suffices)");
    }

    // ============================================================
    // T12 variant: partial match WITH a common tail of a different size (non-uniform)
    // ============================================================
    // OLD = {A B C D E F} (6), NEW = {A B X Y C D E F} (8 - insert of 2 at position 2).
    // LCP = 2, suffix = 4 ([C D E F]).
    // uniform_delta = n_new - n_old = 2.  For each suffix token: new_pos = 8-4+i, old_pos = 6-4+i.
    // deltas = [2, 2, 2, 2].  uniform=true, delta=+2.
    {
        const std::vector<int32_t> ck_tokens = {1, 2, 3, 4, 5, 6};
        const std::vector<int32_t> prompt    = {1, 2, 90, 91, 3, 4, 5, 6};
        std::vector<CheckpointLength> chain;
        const SuffixHookOutcome outcome = suffix_reuse_partial_apply(
            ck_tokens, prompt, /*L=*/6, /*P=*/2, chain);
        check(outcome.fired && outcome.uniform && outcome.delta == 2,
              "T12v2.1: pure insert of 2 -> delta=+2");
    }

    // ============================================================
    // T13: OFF-FLAG BYTE-IDENTITY
    // ============================================================
    // Same shape as T12, flag=0: best_with_partial returns the same as best() (no partial).
    {
        auto s = image({1, 2, 3, 4});
        s.checkpoints = {ckpt({1, 2, 3, 4})};
        ConversationCache cache(1 << 20, 4);
        cache.put(std::move(s));

        const std::vector<int32_t> prompt = {1, 2, 90, 91};   // [A B X Y]
        // flag=0: byte-identical to best() - no partial match, no partial fields touched.
        const auto m0 = cache.best_with_partial(0, prompt, {}, true);
        const auto mb = cache.best(prompt, {}, true);
        check(!m0.partial,
              "T13.1: flag=0 -> partial=false (byte-identical to best()'s default)");
        check(m0.tokens == mb.tokens && m0.index == mb.index && m0.live == mb.live,
              "T13.2: flag=0 -> match fields == best() (index, tokens, live)");
        check(m0.tokens == 0,
              "T13.3: flag=0 + no strict match -> tokens == 0 (the runtime sees no mount)");

        // flag<0 (canonical "flag absent"): same as flag=0.
        const auto m_neg = cache.best_with_partial(-1, prompt, {}, true);
        check(!m_neg.partial && m_neg.tokens == 0,
              "T13.4: flag<0 (canonical 'flag absent') -> byte-identical to flag=0");

        // The runtime never calls the partial helper when the cache returned a strict or no match,
        // so the partial helper's behaviour at flag=0 is not exercised by the runtime.  The pin is:
        // the cache never returns a partial match at flag=0 -> the runtime sees a strict-or-empty
        // Match, identical to today.  T13.5 below proves the helper is invariant to the flag
        // (the flag is a cache-side concern; the helper is a pure token+shape function that fires
        // on valid partial inputs regardless of how the cache was called).
        const SuffixHookOutcome fired = suffix_reuse_partial_apply(
            {1, 2, 3, 4}, prompt, 4, 2, std::vector<CheckpointLength>{{4, 0}});
        check(fired.fired && fired.plan_edit_pos == 2 && fired.restore_anchor_length == 4,
              "T13.5: partial helper is flag-agnostic (pure token+shape function; fires on valid partial)");

        // T13.6: the helper IS fail-closed on invalid shapes (any of: L<=0, P<=0, P>=L, L>n, LCP
        // inconsistent, tokens.size()!=L).  See T15v below for the exhaustive set.
    }

    // ============================================================
    // T14: BEST-OF-N (longest LCP wins; strict > partial)
    // ============================================================
    {
        // Three checkpoints with different LCPs against prompt {A B X Y}:
        //   ck A: {A B C D}     -> LCP = 2, depth = 4
        //   ck B: {A B C D E F} -> LCP = 2, depth = 6  (longer, same LCP - prefer deeper)
        //   ck C: {A B}         -> strict prefix (length 2) - PREFERRED OVER PARTIAL
        // Expected: best_with_partial returns ck C (strict match at 2, beats any partial).
        auto s = image({});
        s.checkpoints = {ckpt({1, 2, 3, 4}), ckpt({1, 2, 3, 4, 5, 6}), ckpt({1, 2})};
        ConversationCache cache(1 << 20, 4);
        cache.put(std::move(s));

        // T14 under the FW-1 splice-engagement contract (2026-10-13): the OLD
        // pin was "strict > partial always".  The dominance rule replaces it:
        // partial is chosen when its anchor covers strictly more (L > s) and it
        // never reads shallower (p >= s).  Here ck C {A B} is strict at s=2;
        // ck A {A B C D} has LCP p=2 == s and L=4 > s -> partial WINS, reading
        // from the same point 2 the strict match would, with a deeper anchor
        // (the splice branch decides downstream whether the tail is recoverable;
        // this prompt's tail {C D} vs {X Y} has no common suffix, so it degrades
        // to the same gap-read the strict match would do - never worse).
        const std::vector<int32_t> prompt = {1, 2, 90, 91};   // [A B X Y]
        const auto m = cache.best_with_partial(64, prompt, {}, true);
        check(m.partial,
              "T14.1: deeper partial (L=4) beats equal-depth strict (s=2, p=2) - splice engagement");
        check(m.tokens == 2,
              "T14.2: reads from the LCP (2) - same point as the strict match, never shallower");
        check(m.checkpoint_length == 6,
              "T14.3: the deepest checkpoint (6) is the anchor - T17 lifted the size guard, so the "
              "6-long entry (LCP 2, previously size-refused) wins the deeper-anchor tie-break over 4");

        // Now drop ck C and re-test: partial mode, longest LCP wins, deeper anchor wins ties.
        // Use a prompt long enough for both checkpoints to be partial (L <= n).
        auto s2 = image({});
        s2.checkpoints = {ckpt({1, 2, 3, 4}), ckpt({1, 2, 3, 4, 5, 6})};
        ConversationCache cache2(1 << 20, 4);
        cache2.put(std::move(s2));
        const std::vector<int32_t> prompt_long = {1, 2, 90, 91, 92, 93, 94, 95};  // length 8
        const auto m2 = cache2.best_with_partial(64, prompt_long, {}, true);
        check(m2.partial,
              "T14.4: no strict match -> partial mode engages");
        check(m2.tokens == 2 && m2.checkpoint_length == 6,
              "T14.5: deepest anchor wins ties (ck B {1..6} beats ck A {1..4} at LCP=2)");

        // Three checkpoints with different LCPs: longest LCP wins.
        auto s3 = image({});
        s3.checkpoints = {ckpt({1, 2, 3, 4}), ckpt({1, 2, 30, 31, 32})};
        ConversationCache cache3(1 << 20, 4);
        cache3.put(std::move(s3));
        const std::vector<int32_t> prompt2 = {1, 2, 3, 90, 91, 92};   // matches ck A for 3 tokens, ck B for 2
        const auto m3 = cache3.best_with_partial(64, prompt2, {}, true);
        check(m3.partial && m3.tokens == 3 && m3.checkpoint_length == 4,
              "T14.6: longest LCP wins (3 > 2, ck A {1..4} at LCP=3 over ck B at LCP=2)");

        // Live state beats checkpoints of the same length (the existing best() tie-break).  Use a
        // prompt long enough for both the live (length 6) and the checkpoint (length 4) to be
        // partial candidates.
        auto s4 = image({1, 2, 3, 4, 5, 6});
        s4.checkpoints = {ckpt({1, 2, 3, 4})};
        ConversationCache cache4(1 << 20, 4);
        cache4.put(std::move(s4));
        const std::vector<int32_t> prompt_live = {1, 2, 90, 91, 92, 93, 94, 95};  // length 8
        const auto m4 = cache4.best_with_partial(64, prompt_live, {}, true);
        check(m4.partial && m4.tokens == 2 && m4.checkpoint_length == 6 && m4.live,
              "T14.7: live state matches at LCP=2 (deeper anchor); live=true");
    }

    // ============================================================
    // T15: IMAGE / GEOMETRY MISMATCH (refuse, full re-read)
    // ============================================================
    // The checkpoint has an image at position 1; the request's images don't include it.  The
    // LCP-based image check refuses the partial mount even if the tokens match.
    {
        auto s = image({});
        s.checkpoints = {ckpt({1, 2, 3, 4})};
        s.checkpoints[0].imgs = {{1, 0xCAFE}};                   // image at position 1
        ConversationCache cache(1 << 20, 4);
        cache.put(std::move(s));

        const std::vector<int32_t> prompt = {1, 2, 90, 91};
        // strict search: image mismatch at position 1 (< 4) -> strict returns 0.
        const std::vector<ConversationImageKey> req_imgs = {{2, 0xDEAD}};  // request has an image at pos 2
        const auto strict = cache.best(prompt, req_imgs, true);
        check(strict.tokens == 0,
              "T15.1 setup: strict refuses on image mismatch at position < strict-length");

        // partial search: the LCP image check (bounded by P) must also refuse.  Here the LCP
        // could be 2 (tokens 0,1 match) but the checkpoint has an image at position 1 which is
        // below the LCP.  The request's image at position 2 is at P (not below).  So the
        // checkpoint has an image below P that the request does NOT have -> refuse.
        const auto m = cache.best_with_partial(64, prompt, req_imgs, true);
        check(!m.partial,
              "T15.2: partial refuses on image mismatch below the LCP (no fabricated mount)");

        // The partial helper itself is a pure token+shape check; the cache is the image gate.
        // In this case, the cache refuses BEFORE the helper is called.  Pin the helper's
        // token-level correctness: same shape, tokens match, helper fires.
        const SuffixHookOutcome ok = suffix_reuse_partial_apply(
            {1, 2, 3, 4}, prompt, /*L=*/4, /*P=*/2, std::vector<CheckpointLength>{});
        check(ok.fired,
              "T15.3: helper's token-level check fires (the cache is the image gate, not the helper)");
    }

    // ============================================================
    // T15 variant: PARTIAL helper fail-closed on inconsistent inputs
    // ============================================================
    {
        // L <= 0
        const SuffixHookOutcome a = suffix_reuse_partial_apply(
            {1, 2, 3, 4}, {1, 2, 90, 91}, /*L=*/0, /*P=*/2, {});
        check(!a.fired, "T15v.1: L <= 0 -> fired=false (fail-closed)");

        // P <= 0
        const SuffixHookOutcome b = suffix_reuse_partial_apply(
            {1, 2, 3, 4}, {1, 2, 90, 91}, /*L=*/4, /*P=*/0, {});
        check(!b.fired, "T15v.2: P <= 0 -> fired=false (fail-closed)");

        // P >= L (not partial, would be strict)
        const SuffixHookOutcome c = suffix_reuse_partial_apply(
            {1, 2, 3, 4}, {1, 2, 3, 4, 5}, /*L=*/4, /*P=*/4, {});
        check(!c.fired, "T15v.3: P >= L -> fired=false (not a partial shape)");

        // L > cur.size() - T17 (2026-10-13): this is now a VALID partial shape
        // (the parked-live anchor is typically longer than the next prompt).
        // The helper fires and narrows to P; the runtime restores the deepest
        // checkpoint <= P, never state@L.  The old pin refused it (three guards);
        // see T17.6 in this suite.
        const SuffixHookOutcome d = suffix_reuse_partial_apply(
            {1, 2, 3, 4, 5, 6}, {1, 2}, /*L=*/6, /*P=*/2, {});
        check(d.fired && d.plan_edit_pos == 2 && d.restore_anchor_length == 6,
              "T15v.4: L(6) > cur(2) fires (T17 live-anchor shape; anchor 6, narrow 2)");

        // Inconsistent tokens (length mismatch)
        const SuffixHookOutcome e = suffix_reuse_partial_apply(
            {1, 2, 3, 4, 5}, {1, 2, 90, 91}, /*L=*/4, /*P=*/2, {});
        check(!e.fired, "T15v.5: tokens.size() != L -> fired=false (inconsistent)");

        // LCP claim inconsistent with the actual tokens
        const SuffixHookOutcome f = suffix_reuse_partial_apply(
            {1, 2, 3, 4}, {1, 90, 3, 4}, /*L=*/4, /*P=*/2, {});   // claim P=2 but token 1 mismatches
        check(!f.fired, "T15v.6: LCP claim inconsistent with tokens -> fired=false");

        // Empty chain + valid partial = fired with empty chain (chain filter is a no-op).
        const SuffixHookOutcome g = suffix_reuse_partial_apply(
            {1, 2, 3, 4}, {1, 2, 90, 91}, /*L=*/4, /*P=*/2, {});
        check(g.fired && g.chain.empty(),
              "T15v.7: empty chain + valid partial = fired, chain stays empty");

        // Chain with the anchor + a non-anchor entry above the LCP: keep the anchor, drop the other.
        {
            std::vector<CheckpointLength> chain = { {2, 0}, {4, 1}, {6, 2} };  // L=4 is the anchor
            const SuffixHookOutcome a = suffix_reuse_partial_apply(
                {1, 2, 3, 4}, {1, 2, 90, 91}, /*L=*/4, /*P=*/2, chain);
            check(a.fired && a.chain.size() == 2 && a.chain[0].length == 2 && a.chain[1].length == 4,
                  "T15v.9: chain with anchor L=4 + non-anchor 6 > P=2 -> keep [2, 4], drop 6");
        }
        // Chain with the anchor + an entry exactly at the LCP: keep both (the LCP entry is below).
        {
            std::vector<CheckpointLength> chain = { {2, 0}, {4, 1} };
            const SuffixHookOutcome a = suffix_reuse_partial_apply(
                {1, 2, 3, 4}, {1, 2, 90, 91}, /*L=*/4, /*P=*/2, chain);
            check(a.fired && a.chain.size() == 2,
                  "T15v.10: chain with anchor L=4 + entry at P=2 -> both kept (entry length <= P)");
        }

        // Token at the divergence point matches -> LCP == L (strict, not partial).  The helper refuses
        // because the divergence check `partial_ck[P] != cur[P]` fails (they're equal -> LCP > P).
        // The strict path handles this; the partial helper must not fire.
        const SuffixHookOutcome strict = suffix_reuse_partial_apply(
            {1, 2, 3, 4}, {1, 2, 3, 90}, /*L=*/4, /*P=*/2, {});   // claim P=2 but token 2 matches (LCP=3)
        check(!strict.fired,
              "T15v.8: claim P=2 but tokens show LCP=3 -> fired=false (strict path handles)");

        // The LCP verification loop catches the inconsistency: a P=2 claim on tokens where the
        // first 2 match but the 3rd also matches means the caller mis-computed P.  The helper's
        // LCP verification walks [0, P) only; it does NOT walk beyond P (the divergence check at
        // [P] is the next guard).  When P==L (which the L>=P guard catches above), it's strict.
        // For P < L but token[P] == cur[P], the LCP would be > P, not == P - the divergence
        // check catches it: the helper returns fired=false.
    }

    // ============================================================
    // T16: SPLICE ENGAGEMENT - a deeper partial beats a shallower strict
    // ============================================================
    // The M1 steady-state finding (2026-10-13): best_with_partial returns at the
    // FIRST strict match and never evaluates partial, so when the checkpoints hold
    // both a strict anchor s (s <= D, the divergence point) and a DEEPER checkpoint
    // L > D that would splice, the strict one wins and the splice never engages
    // ([SFX-DBG] partial-branch enter=0 in every steady trace).
    //
    // Dominance rule (safe because one conversation's checkpoints are nested
    // prefixes of ONE history: the history diverges from the request at a single
    // point D, so every strict anchor s <= D = the partial LCP p - a partial mount
    // never reads from a shallower point, and its splice may recover up to L):
    // prefer the partial match when its checkpoint covers strictly more than the
    // strict anchor (L > s); otherwise keep strict (no splice cost when it cannot
    // recover more).
    {
        // One conversation: checkpoints {A B} (2, the future strict anchor) and
        // {A B C D E} (5, crosses the edit).  Request {A B C X E ...}: history
        // diverges at D=3 (C vs X at index 3).  Strict anchor {A B} mounts 2;
        // partial on {A B C D E}: LCP=3, L=5 -> splice recovers [P=3, L=5) tail E.
        auto s = image({1, 2, 3, 4, 5});
        s.checkpoints = { ckpt({1, 2}), ckpt({1, 2, 3, 4, 5}) };
        ConversationCache cache(1 << 20, 4);
        cache.put(std::move(s));

        const std::vector<int32_t> prompt = {1, 2, 3, 90, 5};   // edits token 3 (X=90)
        const auto strict = cache.best(prompt, {}, true);
        check(strict.tokens == 2 && !strict.partial,
              "T16.1: strict best() mounts the {A B} anchor (2)");

        const auto m = cache.best_with_partial(64, prompt, {}, true);
        check(m.partial,
              "T16.2: partial preferred over shallower strict (splice engagement)");
        check(m.tokens == 3,
              "T16.3: partial LCP == 3 (== the strict anchor's 2... history diverges at 3)");
        check(m.checkpoint_length == 5,
              "T16.4: the deeper checkpoint (5) is the splice anchor");

        // The neutral case: strict anchor AT the divergence depth (s == p == 3).
        // The dominance rule still prefers the deeper checkpoint (L=5 covers the
        // common tail {E}), and the mount reads from p=3 == s: NEVER shallower
        // than the strict anchor, so the swap is safe even when the tail turns
        // out recoverable or not (old_suffix_len=0 handles itself downstream).
        auto s2 = image({1, 2, 3, 4, 5});
        s2.checkpoints = { ckpt({1, 2, 3}), ckpt({1, 2, 3, 4, 5}) };
        ConversationCache cache2(1 << 20, 4);
        cache2.put(std::move(s2));
        const auto m2 = cache2.best_with_partial(64, prompt, {}, true);
        check(m2.tokens >= strict.tokens,
              "T16.5: partial never reads shallower than the strict anchor (p >= s)");
        check(!m2.partial || m2.checkpoint_length > strict.tokens,
              "T16.5b: partial only chosen when its anchor covers strictly more (L > s)");

        // Off-flag: byte-identical strict (the early-return never changes behavior).
        const auto m0 = cache.best_with_partial(0, prompt, {}, true);
        check(!m0.partial && m0.tokens == 2,
              "T16.6: flag=0 stays byte-identical strict (2)");
    }

    // ============================================================
    // T17: LIVE-ANCHOR PARTIAL - a parked live LONGER than the prompt
    // ============================================================
    // The M1 probe finding (2026-10-13, chain /tmp/dt-steady4.log, C-turn):
    //   parked-scan: e0.live:L=5786:skip   (prompt_len=5736)
    // The parked LIVE (prompt+response of the previous turn) is almost always
    // slightly LONGER than the next turn's prompt, and three L > prompt.size()
    // guards refused it without computing the LCP: the cache's consider()
    // size-guard, conversation_lcp's n > prompt.size() early-out, and
    // suffix_reuse_partial_apply's L > n_new refusal.  The live anchor is the
    // single most valuable partial candidate in every edit chain - lifting the
    // guards for the PARTIAL PASS ONLY (strict stays byte-identical) lets the
    // partial mount recover [P, L) K/V the strict anchor cannot reach.
    //
    // Safety: the LCP walk stops at min(L, prompt.size()) - tokens past the
    // prompt are irrelevant (the partial mount discards them); the strict pass
    // keeps its own guard (a longer-than-prompt checkpoint can NEVER be a
    // strict prefix).  The runtime routes a live-anchor partial through the
    // narrowed_resume_target restore (state@L is stale for the edited history;
    // see the v3 fix note in generate.cpp) - restore the deepest checkpoint
    // <= P, arena covers [ckpt, P) by the LCP, read [P, n).
    {
        // Parked live {A B C D E} (5) + shallow checkpoints {A B}(2), {A B C}(3).
        // Request {A B C X} (4) - SHORTER than the live, edits index 3.
        // Strict: {A B C} mounts 3 (live and {A B C D E} fail: longer/edited).
        // Partial on the live: LCP = 3, L = 5 > strict 3 -> dominance fires.
        auto s = image({1, 2, 3, 4, 5});
        s.checkpoints = { ckpt({1, 2}), ckpt({1, 2, 3}) };
        ConversationCache cache(1 << 20, 4);
        cache.put(std::move(s));

        const std::vector<int32_t> prompt = {1, 2, 3, 90};   // len 4 < live 5
        const auto strict = cache.best(prompt, {}, true);
        check(strict.tokens == 3 && !strict.partial,
              "T17.1: strict best() mounts {A B C} (3); the longer live is refused");

        const auto m = cache.best_with_partial(64, prompt, {}, true);
        check(m.partial,
              "T17.2: live anchor (L=5 > prompt=4) is now a partial candidate");
        check(m.tokens == 3,
              "T17.3: live-anchor partial LCP == 3 (divergence at index 3)");
        check(m.checkpoint_length == 5,
              "T17.4: the parked LIVE (5) is the splice anchor");
        check(m.live,
              "T17.5: Match.live == true (the anchor is the parked live)");

        // The helper accepts L > n_new for the partial path (fail-closed kept
        // for every OTHER shape).  Fire with L=5, P=3, cur len 4: plan narrows
        // to P=3, anchor 5, no common tail ({D E} vs {X}).
        const std::vector<int32_t> live_tokens = {1, 2, 3, 4, 5};
        std::vector<CheckpointLength> chain = { {2, 0}, {3, 1} };
        const SuffixHookOutcome outcome = suffix_reuse_partial_apply(
            live_tokens, prompt, /*L=*/5, /*P=*/3, chain);
        check(outcome.fired,
              "T17.6: partial_apply fires with L(5) > n_new(4)");
        check(outcome.fired && outcome.plan_edit_pos == 3,
              "T17.7: plan_edit_pos == 3 (the LCP)");
        check(outcome.fired && outcome.restore_anchor_length == 5,
              "T17.8: restore_anchor_length == 5 (the live anchor)");
        check(outcome.fired && outcome.old_suffix_len == 0,
              "T17.9: old_suffix_len == 0 ({D E} vs {X} - no common tail)");

        // conversation_lcp direct: live longer than prompt walks to the prompt's
        // end and reports the prompt-bounded LCP (not 0).
        ConversationCheckpoint lc;
        lc.ids = {1, 2, 3, 4, 5};
        check(strata::core::conversation_lcp(lc, prompt, {}) == 3,
              "T17.10: conversation_lcp walks a longer-than-prompt checkpoint (3, not 0)");

        // Off-flag: byte-identical strict - the live-anchor lift is flag-gated.
        const auto m0 = cache.best_with_partial(0, prompt, {}, true);
        check(!m0.partial && m0.tokens == 3,
              "T17.11: flag=0 stays byte-identical strict (3)");

        // A strict-longer guard REMAINS: a candidate longer than the prompt can
        // never produce p >= L (strict); the partial pass alone evaluates it.
        // Pin that conversation_lcp reports the FULL checkpoint length when the
        // checkpoint is a strict prefix (shorter or equal) - unchanged.
        ConversationCheckpoint sc;
        sc.ids = {1, 2};
        check(strata::core::conversation_lcp(sc, prompt, {}) == 2,
              "T17.12: shorter strict-prefix checkpoint still reports its full length (2)");
    }

    std::printf("suffix_reuse_partial_prefix_test: %d passed, %d failed\n", g_pass, g_fail);
    return g_fail == 0 ? 0 : 1;
}
