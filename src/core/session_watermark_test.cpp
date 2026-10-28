// CPU only: the immutability watermark is host-side bookkeeping on the session state - no device, no CUDA calls.
// The bump sites (M1-M9) live in the engine and core restore paths; this file pins the WATERMARK ITSELF:
// the low-water semantics, the whole-cells rounding, and the mount-case refusals they exist to drive.
#include "strata/core/session.hpp"

#include <cstdio>
#include <cstdlib>
#include <limits>

using namespace strata::core;
namespace {
int checks = 0;
void check(bool ok, const char* label) {
    ++checks;
    if (!ok) { std::fprintf(stderr, "FAIL: %s\n", label); std::exit(1); }
}
}

int main() {
    // A fresh state is append-eligible for any saved boundary.
    {
        SessionState s;
        check(s.kv_first_dirty == std::numeric_limits<int64_t>::max(), "fresh state has nothing dirty");
        check(s.kv_append_ok(64, 0), "fresh state is append-eligible at boundary 0");
        check(s.kv_append_ok(64, (int64_t) 1 << 20), "fresh state is append-eligible at any boundary");
    }

    // Dirty below the saved boundary refuses; at or above it stays eligible.
    {
        SessionState s;
        s.mark_kv_dirty(1100);
        check(s.kv_append_ok(64, 1024), "a write above the saved boundary keeps the append eligible");
    }
    {
        SessionState s;
        s.mark_kv_dirty(1024);  // a write starting exactly AT the boundary never touches the committed region
        check(s.kv_append_ok(64, 1024), "a write starting at the boundary stays eligible");
    }
    {
        SessionState s;
        s.mark_kv_dirty(1023);
        check(!s.kv_append_ok(64, 1024), "a write below the saved boundary refuses the append");
    }

    // Rounding (S2/A3 partial-page caveat): the watermark counts from its whole-page start - the shared
    // partial page is never treated as committed, so a write inside it refuses even when it is only a
    // little below the boundary.
    {
        SessionState s;
        s.mark_kv_dirty(1000);  // inside the page [960, 1024): whole cells 960 < 1024
        check(!s.kv_append_ok(64, 1024), "a dirty write inside the shared partial page refuses");
    }
    {
        SessionState s;
        s.mark_kv_dirty(960);  // exactly at a page start below the boundary
        check(!s.kv_append_ok(64, 1024), "a dirty page start below the boundary refuses");
    }
    {
        SessionState s;
        s.mark_kv_dirty(1088);  // a whole page start above the boundary
        check(s.kv_append_ok(64, 1024), "a dirty page start above the boundary stays eligible");
    }

    // Low-water semantics: a mark can only move the watermark down.
    {
        SessionState s;
        s.mark_kv_dirty(2000);
        s.mark_kv_dirty(500);
        s.mark_kv_dirty(3000);
        check(s.kv_first_dirty == 500, "the watermark is a low-water mark");
    }
    {
        SessionState s;
        s.mark_kv_dirty(-1);  // not a token index: refuses to invent dirt out of nothing
        check(s.kv_first_dirty == std::numeric_limits<int64_t>::max(), "a negative mark changes nothing");
    }

    // No page geometry: raw comparison.
    {
        SessionState s;
        s.mark_kv_dirty(1030);
        check(s.kv_append_ok(0, 1024), "without page geometry the raw index compares");
    }

    // The M8 mount case (the drill's mount rule): mounting N tokens rewrites the spare pooled row at
    // floor(N/idx_block)*idx_block - strictly below the saved boundary even for a SAME-tokens mount, so
    // a save landing between the mount and the next block completion must not append.
    {
        const int64_t n = 65, idx_block = 4;
        SessionState s;
        s.mark_kv_dirty((n / idx_block) * idx_block);
        check(s.kv_first_dirty == 64, "the spare-row bump marks its row boundary");
        check(!s.kv_append_ok(64, n), "a same-tokens mount refuses the append (spare row rewritten)");
    }

    // The whole-cells helper the engine gate shares (trailer saved_tokens -> committed prefix).
    check(session_whole_cells(65, 64) == 64, "saved 65 tokens commit one whole page at page 64");
    check(session_whole_cells(1100, 64) == 1088, "whole cells round down to the page");
    check(session_whole_cells(1024, 64) == 1024, "a boundary page start is whole");
    check(session_whole_cells(700, 0) == 700, "page 0 means no rounding");

    std::printf("session_watermark_test: %d checks passed\n", checks);
    return 0;
}
