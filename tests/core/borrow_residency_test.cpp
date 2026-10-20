// The prompt loan's residency bookkeeping (#796 part B): the residency table is built whenever the prompt path can
// borrow expert-cache storage, whether or not the token graph - its other reader - is captured.  Before the fix the
// table existed only inside the token-graph setup, so --no-token-graph or --no-capture silently turned a borrowed
// prefill into owned buffers the cache sizing had not reserved room for.  The decision is one pure predicate
// (strata/prefill/prefill.hpp), and this pins its mode matrix; the lend/refill behavior itself is exercised on the
// hardware by the two-request serve run named in the PR.
#include "strata/prefill/prefill.hpp"

#include <cstdio>

namespace {
int fails = 0;
void check(bool ok, const char* what) {
    if (!ok) {
        std::fprintf(stderr, "FAIL: %s\n", what);
        ++fails;
    }
}
}  // namespace

int main() {
    // borrowing capability (an expert profile, not --no-prefill-borrow) x the prompt path's chunk
    // 1: the capability alone turns the bookkeeping on - token graph or not (the graph never decides this)
    check(strata::prefill::borrow_residency(true, 8192), "borrowing with a chunk builds the residency map");
    check(strata::prefill::borrow_residency(true, 256), "the smallest chunk still borrows");
    // 2: --no-prefill-borrow (or no expert profile) builds nothing extra - the owned path needs no loan bookkeeping
    check(!strata::prefill::borrow_residency(false, 8192), "--no-prefill-borrow keeps the map off");
    check(!strata::prefill::borrow_residency(false, 0), "no profile and no chunk keeps the map off");
    // 3: no prompt path (no --prefill) borrows nothing and allocates no residency state
    check(!strata::prefill::borrow_residency(true, 0), "a chunkless run builds no loan bookkeeping");
    check(!strata::prefill::borrow_residency(false, -1), "a negative chunk is no prompt path");

    // 4: a staging failure of the table keeps its old severity where the token graph needs it, and is only a
    // warn-and-fall-back where the loan does - borrowing-only must not gain a new hard refusal
    check(strata::prefill::residency_staging_failure_is_fatal(true),
          "the graph's residency staging failure stays fatal (the engine's own refusal)");
    check(!strata::prefill::residency_staging_failure_is_fatal(false),
          "a borrowing-only staging failure falls back: borrowing disabled, owned buffers, no process exit");
    // the combined case (graph + borrowing) is the graph's: its hit path genuinely cannot run without the table
    check(strata::prefill::residency_staging_failure_is_fatal(true && true),
          "graph + borrowing staging failure is the graph's, fatal");

    if (fails == 0) std::fprintf(stderr, "borrow_residency_test: all checks passed\n");
    return fails == 0 ? 0 : 1;
}
