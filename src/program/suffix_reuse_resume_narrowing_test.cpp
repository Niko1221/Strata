// src/program/suffix_reuse_resume_narrowing_test.cpp - T8: the resume-narrowing wiring
// regression for the 2026-10-12 wiring bug 2 on `feat/suffix-reuse`.
//
// THE BUG.  The phase-2 splice hook (generate.cpp:9034-9062) erases checkpoints past
// `plan_edit_pos` AND narrows `resume` to `plan_edit_pos`.  The checkpoint-restore search
// later (generate.cpp:9294-9325, the `else if (!from_live)` branch) requires a checkpoint
// with `ids.size() == resume` EXACTLY.  On a re-edited request the splice hook's own erase
// removed the exact-length candidate (the resume checkpoint is at the OLD resume, e.g.
// 3240, and the narrowed resume is the edit position, e.g. 3239), so the search returns
// `c == nullptr` and the engine prints "ERR restoring a conversation checkpoint failed"
// and returns 1, which the upstream serve loop turns into a 400.  Live evidence
// (GPU-verified by orchestrator): `checks`={3240, 3244}, `resume` narrowed to 3239 ->
// c==nullptr -> 400.  Off-arm (flag absent) identical traffic: mounts 3240, serves ~1s.
//
// THE FIX (route c - the gap-read / replay path).  The header `suffix_reuse.hpp` gains a
// pure helper `narrowed_resume_target` that decides, given a narrowed resume and the
// live `checks` chain, what the request's next move is:
//
//   - exact-length candidate exists:  restore it; read_from = narrowed resume
//                                     (no gap; byte-equivalent to today's restore).
//   - no exact-length, deepest-below: restore the deepest ck with length <= narrowed;
//                                     read_from = narrowed resume (the [ck, narrowed)
//                                     gap is re-read by the existing prefill path - the
//                                     "replay" / gap-read path phase 2 was designed for).
//   - no candidate at or below:       restore_length = 0, read_from = 0
//                                     (no usable mount; full re-prefill, NO error).
//
// In every case the request proceeds.  generate.cpp's `else if (!from_live)` branch
// calls this helper and uses its two return values (restore length, read_from) to drive
// the existing restore + chunk-loop paths unchanged.
//
// WHAT THIS TEST PINS.  The helper is pure, header-only, and is exercised on the live
// failing inputs (resume_narrowed=3239, checks={3240, 3244}) plus the three adjacent
// shapes the fix has to remain correct on:
//
//   T8.1 LIVE RED    3239, {3240, 3244}                 -> restore=0,    read_from=0
//   T8.2 EXACT-LEN   3239, {200, 3239, 3240}            -> restore=3239, read_from=3239
//   T8.3 DEEPEST-BEL 3239, {100,200,1500,3240,3244}     -> restore=1500, read_from=3239
//   T8.4 CONTINUATN  3000, {200,1000,2000,3000,3240}    -> restore=3000, read_from=3000
//   T8.5 NO CHECKS   3239, {}                           -> restore=0,    read_from=0
//
// CPU-ONLY.  Header-only; no CUDA, no <cuda_runtime.h>.  RED on 37dec54: the helper
// does not exist (compile error).  GREEN after the fix: the helper exists and matches
// every case.  generate.cpp's call site is exercised by the orchestrator's GPU probe
// (t1 -> t2 splice-eligible -> expect 200, splice/replay marker, reused>0 or read<full).

#include "strata/program/suffix_reuse.hpp"

#include <cstdint>
#include <cstdio>
#include <vector>

namespace {

// Result tally
int g_fail = 0;
int g_pass = 0;
void check(bool ok, const char* what) {
    std::printf("  %-78s %s\n", what, ok ? "ok" : "FAIL");
    if (ok) ++g_pass; else ++g_fail;
}

}  // namespace

int main() {
    std::printf("suffix_reuse_resume_narrowing_test (T8: wiring bug 2 on feat/suffix-reuse)\n");

    using strata::program::suffix_reuse::NarrowedResumeTarget;
    using strata::program::suffix_reuse::narrowed_resume_target;

    // ---- T8.1 LIVE RED: 3239, {3240, 3244} ----
    // The splice hook erased the resume checkpoint and narrowed resume; no exact
    // candidate survives.  The fix must NOT error: it must fall back to a full
    // re-prefill from 0.
    {
        const std::vector<int64_t> chain{3240, 3244};
        const NarrowedResumeTarget t = narrowed_resume_target(3239, chain);
        check(t.restore_length == 0,
              "T8.1 LIVE: no ck <= 3239 survives; restore_length=0 (no 'restoring failed' error)");
        check(t.read_from == 0,
              "T8.1 LIVE: read_from=0 (full re-prefill; gap-read path engaged)");
    }

    // ---- T8.2 EXACT-LENGTH: 3239, {200, 3239, 3240} ----
    {
        const std::vector<int64_t> chain{200, 3239, 3240};
        const NarrowedResumeTarget t = narrowed_resume_target(3239, chain);
        check(t.restore_length == 3239,
              "T8.2 exact-length: ck@3239 is the resume target; restore_length=3239");
        check(t.read_from == 3239,
              "T8.2 exact-length: no gap to re-read; read_from=3239");
    }

    // ---- T8.3 DEEPEST-BELOW: 3239, {100, 200, 1500, 3240, 3244} ----
    // No exact-length; deepest surviving ck below the narrowed resume is 1500.
    // Restore 1500 and re-read the [1500, 3239) gap via the existing prefill path.
    {
        const std::vector<int64_t> chain{100, 200, 1500, 3240, 3244};
        const NarrowedResumeTarget t = narrowed_resume_target(3239, chain);
        check(t.restore_length == 1500,
              "T8.3 deepest-below: ck@1500 is deepest <= 3239; restore_length=1500");
        check(t.read_from == 3239,
              "T8.3 deepest-below: gap [1500, 3239) is re-read; read_from=3239");
    }

    // ---- T8.4 PURE CONTINUATION: 3000, {200, 1000, 2000, 3000, 3240} ----
    // Splice hook did NOT narrow (plan.edit_pos == resume, the common case); the
    // exact-length candidate is there; the helper is a no-op find.
    {
        const std::vector<int64_t> chain{200, 1000, 2000, 3000, 3240};
        const NarrowedResumeTarget t = narrowed_resume_target(3000, chain);
        check(t.restore_length == 3000,
              "T8.4 continuation: ck@3000 is exact-length; restore_length=3000");
        check(t.read_from == 3000,
              "T8.4 continuation: no gap, no re-read; read_from=3000");
    }

    // ---- T8.5 NO CHECKS: 3239, {} ----
    // No mounted conversation; the request is a fresh prefill from 0; no error.
    {
        const std::vector<int64_t> chain{};
        const NarrowedResumeTarget t = narrowed_resume_target(3239, chain);
        check(t.restore_length == 0,
              "T8.5 no-checks: empty chain; restore_length=0 (fresh read, no error)");
        check(t.read_from == 0,
              "T8.5 no-checks: read_from=0 (full re-prefill)");
    }

    std::printf("\nSummary: %d passed, %d failed\n", g_pass, g_fail);
    return g_fail ? 1 : 0;
}
