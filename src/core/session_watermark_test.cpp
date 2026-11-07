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

    // Rounding: the watermark counts from its whole-page start.  At an aligned boundary the committed
    // prefix ends on the page line, so a rewrite inside the last full page - a little below the boundary -
    // sits below the prefix and refuses.
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

    // The M8 mount case: mounting N tokens rewrites the spare pooled row at floor(N/idx_block)*idx_block -
    // strictly below the saved token count even for a SAME-tokens mount.  The file's trusted region is its
    // whole-page prefix (the format re-emits the partial page every generation), so the mount rewrite lands
    // on re-emitted rows and the save stays append-eligible; only rewrites below the committed prefix refuse.
    {
        const int64_t n = 65, idx_block = 4;
        SessionState s;
        s.mark_kv_dirty((n / idx_block) * idx_block);
        check(s.kv_first_dirty == 64, "the spare-row bump marks its row boundary");
        check(s.kv_append_ok(64, n), "a same-tokens mount rewrites only re-emitted rows, so it stays eligible");
    }

    // Unaligned saved boundary (measured on live momo, where every save was refusing its own next append):
    // the file trusts whole_cells(saved_tokens), and writes at or above it - inside the re-emitted partial
    // page - stay eligible.
    {
        SessionState s;
        s.mark_kv_dirty(1000);  // a resume rewriting from exactly the saved token count
        check(s.kv_append_ok(64, 1000), "an unaligned save earns its own next append");
        s.mark_kv_dirty(960);  // the whole-page start of the shared partial page (re-emitted per generation)
        check(s.kv_append_ok(64, 1000), "a mount rewriting the unaligned partial page stays eligible");
        s.mark_kv_dirty(959);  // one row below the committed whole-page prefix
        check(!s.kv_append_ok(64, 1000), "a write below an unaligned file's committed prefix refuses");
    }

    // The whole-cells helper the engine gate shares (trailer saved_tokens -> committed prefix).
    check(session_whole_cells(65, 64) == 64, "saved 65 tokens commit one whole page at page 64");
    check(session_whole_cells(1100, 64) == 1088, "whole cells round down to the page");
    check(session_whole_cells(1024, 64) == 1024, "a boundary page start is whole");
    check(session_whole_cells(700, 0) == 700, "page 0 means no rounding");

    // #4552 (upstream-confirmed): a resume (file restore / park restore / slot return / borrow) rewrites
    // [0, restored) with the conversation's own rows, so its watermark marks the RESTORED PREFIX, not
    // token 0: everything the earlier save committed below the restored count is still byte-faithful, and
    // the conversation earns its next append instead of forcing a full rewrite every switch.  The zeroing
    // sites (M2) keep marking 0 - nothing was restored.
    {
        SessionState s;
        s.kv_first_dirty = 2048;             // a save raised the watermark to its committed boundary
        s.mark_kv_dirty(1500);               // a resume that restored [0, 1500)
        check(s.kv_first_dirty == 1500, "a resume prefix mark stops at the restored count, not 0");
        check(!s.kv_append_ok(64, 2048), "a partial restore still refuses an append over its own prefix");
        check(s.kv_append_ok(64, 1024), "a restore covering the saved prefix keeps it append-eligible");
    }
    {
        SessionState s;
        s.mark_kv_dirty(1000);               // the live shape: restore of the whole file at its saved count
        check(s.kv_append_ok(64, 1000), "a whole-file restore earns the same file's next append");
        s.mark_kv_dirty(0);                  // an M2 zeroing after that is still absolute
        check(!s.kv_append_ok(64, 1000), "the zeroing sites keep marking from token 0");
    }

    // #4552 per-stage gate: a split save writes rows from EVERY carve into one file, so the gate must
    // consult the fold of all carves' watermarks - the most-dirty one governs.  Before this the engine
    // read only the main carve's, and a dirty stage could ride an append.
    {
        SessionState main_, stage_, clean_;
        main_.kv_first_dirty = 4096;                                        // main carve: clean above the boundary
        stage_.kv_first_dirty = 900;                                        // a stage rewrote below 1024
        clean_.kv_first_dirty = std::numeric_limits<int64_t>::max();       // a stage nothing touched
        std::vector<SessionState*> all{&main_, &stage_, &clean_};
        check(session_gate_watermark(all) == 900, "the fold returns the most-dirty carve's watermark");
        SessionState gate;
        gate.kv_first_dirty = session_gate_watermark(all);
        check(!gate.kv_append_ok(64, 1024), "a stage dirty below the boundary refuses the split save's append");
        stage_.kv_first_dirty = 1024;                                      // the stage rewrite started AT the boundary
        gate.kv_first_dirty = session_gate_watermark(all);
        check(gate.kv_append_ok(64, 1024), "stage writes at or above the committed prefix stay eligible");
        check(session_gate_watermark(std::vector<SessionState*>{&main_, &clean_}) == 4096,
              "clean stages leave the fold at the dirty main carve");
        check(session_gate_watermark(std::vector<SessionState*>{&clean_}) == std::numeric_limits<int64_t>::max(),
              "a fold of untouched sessions is INT64_MAX (a single-device save behaves as before)");
    }

    std::printf("session_watermark_test: %d checks passed\n", checks);
    return 0;
}
