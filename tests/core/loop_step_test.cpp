// tests/core/loop_step_test.cpp - #7295 M6: `loop_next_layer` (include/strata/core/loop_step.hpp), the
// recurrent-depth loop's layer-stepping decision pulled out of `session_loop` so it is testable without a
// GPU. No model, no CUDA - `session_capture`/`session_loop` have no dedicated test anywhere in this repo
// (only the one production call site in src/program/generate.cpp), so this is the mechanics-level coverage
// M6 settles for: the regression property ("off is byte-identical") and the synthetic property ("on" visits
// the right layers, the right number of times, with the right graph variant and gate-firing signal) - NOT a
// quality check, because no model trained for this loop exists yet (see the plan's §9).
#include "strata/core/loop_step.hpp"

#include <cstdio>
#include <string>
#include <vector>

namespace {

int g_fail = 0;

void check(bool ok, const std::string& what) {
    std::printf("  %-90s %s\n", what.c_str(), ok ? "ok" : "FAIL");
    if (!ok) ++g_fail;
}

strata::core::ModelGeometry base_geometry(int64_t n_layers, int64_t qsa_interval) {
    strata::core::ModelGeometry g{};
    g.n_layers = n_layers;
    g.qsa_interval = qsa_interval;
    return g;
}

/// Runs the whole layer order `session_loop` would, using ONLY `loop_next_layer` - the same function the
/// real loop calls - and records what happened at every step.
struct Trace {
    std::vector<int64_t> visited;     // every `l` the simulated loop's body ran for, in order
    std::vector<bool> used_noKV;      // per visited layer: true if session_loop would launch execs_noKV
    int64_t pass_boundaries = 0;      // how many times the gate would fire
    int64_t final_loop_pass = 0;
};

Trace run(const strata::core::ModelGeometry& g) {
    using strata::core::loop_next_layer;
    Trace t;
    int64_t loop_pass = 0;
    bool repeat_pass = false;   // whether THIS iteration's `l` was reached via execs_noKV (layer 0: never)
    for (int64_t l = 0; l < g.n_layers; ++l) {
        t.visited.push_back(l);
        t.used_noKV.push_back(repeat_pass);
        const auto step = loop_next_layer(g, l, loop_pass);
        loop_pass = step.loop_pass;
        if (step.pass_boundary) ++t.pass_boundaries;
        repeat_pass = step.repeat_pass;   // applies to `step.next`, which becomes `l` next iteration
        l = step.next - 1;   // mirrors session_loop's "`l = next - 1` so the for's `++l` lands on `next`"
    }
    t.final_loop_pass = loop_pass;
    return t;
}

}  // namespace

int main() {
    std::printf("loop_step_test\n");

    // ---- Regression: the loop OFF (num_passes <= 1, the geometry default) is byte-identical to no loop at
    // all - every layer visited exactly once, in order, never execs_noKV, one call per layer.
    {
        auto g = base_geometry(8, 4);   // num_passes defaults to 1 => has_loop() is false
        check(!g.has_loop(), "off: has_loop() false by default");
        const Trace t = run(g);
        check(t.visited.size() == 8, "off: visits all 8 layers");
        bool in_order = true;
        for (size_t i = 0; i < t.visited.size(); ++i) in_order &= (t.visited[i] == (int64_t) i);
        check(in_order, "off: visited 0..7 in order, no repeats");
        bool no_noKV = true;
        for (bool b : t.used_noKV) no_noKV &= !b;
        check(no_noKV, "off: execs_noKV never selected");
        check(t.pass_boundaries == 0, "off: the gate never fires");
    }

    // ---- Explicit off via loop_core_end <= loop_core_start (num_passes > 1 alone does not mean "on" if the
    // rest of the config is degenerate) is covered by has_loop() itself (`loop_num_passes > 1`), which is
    // geometry.hpp's own property - not retested here, this file only exercises loop_next_layer AFTER
    // validation (session_capture/the GGUF reader are what refuse a bad range before this ever runs).

    // ---- Synthetic: QSA-only range [4, 8) of a 12-layer model (qsa_interval 4: layers 3,7,11 are QSA),
    // 3 passes. Expect: layers 0-3 once, then 4..7 three times (passes), then 8-11 once.
    {
        auto g = base_geometry(12, 4);
        g.loop_core_start = 4;
        g.loop_core_end = 8;
        g.loop_num_passes = 3;
        check(g.has_loop(), "on: has_loop() true");
        const Trace t = run(g);
        const std::vector<int64_t> want = {0, 1, 2, 3, 4, 5, 6, 7, 4, 5, 6, 7, 4, 5, 6, 7, 8, 9, 10, 11};
        check(t.visited == want, "on: visits prelude once, core 3x, coda once, in that order");
        check(t.pass_boundaries == 3, "on: the gate fires exactly num_passes (3) times");
        check(t.final_loop_pass == 2, "on: loop_pass ends at num_passes - 1 (0-indexed)");
        // `used_noKV[i]` says whether `visited[i]` was launched as execs_noKV - recorded when it was
        // `next`, one entry per transition, same indexing as `visited` because every `next` here is <
        // n_layers. execs_noKV must be used for every entry into the core range EXCEPT the very first
        // (pass 0 writes KV normally): positions 4-7 (pass 0's core) stay execs; 8-15 (the two repeats)
        // are execs_noKV; 0-3 (prelude) and 16-19 (coda) are never execs_noKV.
        check(t.used_noKV.size() == t.visited.size(), "on: one noKV flag per transition (none ran off the end)");
        if (t.used_noKV.size() == want.size()) {
            const bool pass0_core_is_normal = !t.used_noKV[4] && !t.used_noKV[5] && !t.used_noKV[6] && !t.used_noKV[7];
            const bool repeats_are_noKV = t.used_noKV[8] && t.used_noKV[9] && t.used_noKV[10] && t.used_noKV[11] &&
                                         t.used_noKV[12] && t.used_noKV[13] && t.used_noKV[14] && t.used_noKV[15];
            const bool prelude_coda_normal = !t.used_noKV[0] && !t.used_noKV[1] && !t.used_noKV[2] && !t.used_noKV[3] &&
                                            !t.used_noKV[16] && !t.used_noKV[17] && !t.used_noKV[18] && !t.used_noKV[19];
            check(pass0_core_is_normal, "on: pass 0's core range uses execs (writes KV), not execs_noKV");
            check(repeats_are_noKV, "on: every repeat pass's core range uses execs_noKV (reads pass-0 KV, doesn't overwrite)");
            check(prelude_coda_normal, "on: the prelude and coda never use execs_noKV");
        } else {
            check(false, "on: used_noKV sized as expected (see above)");
        }
    }

    // ---- Edge: the core range starts at layer 0 (session_loop snapshots R0 before pre[0] in this case,
    // not inside the per-layer loop - loop_next_layer itself does not distinguish this case, which is the
    // point: the snapshot timing is session_loop's concern, not the stepping decision's).
    {
        auto g = base_geometry(4, 1);   // qsa_interval 1: every layer is QSA, so [0, 2) is a valid range
        g.loop_core_start = 0;
        g.loop_core_end = 2;
        g.loop_num_passes = 2;
        const Trace t = run(g);
        const std::vector<int64_t> want = {0, 1, 0, 1, 2, 3};
        check(t.visited == want, "core_start=0: visits 0,1 twice then 2,3 once");
        check(t.pass_boundaries == 2, "core_start=0: gate fires num_passes (2) times");
    }

    // ---- Edge: num_passes = 1 with an otherwise-valid range is "off" (has_loop() is false), so the range
    // is never consulted at all - geometry.hpp's own contract, exercised here end to end.
    {
        auto g = base_geometry(8, 4);
        g.loop_core_start = 4;
        g.loop_core_end = 8;
        g.loop_num_passes = 1;
        check(!g.has_loop(), "num_passes=1: has_loop() false even with a core range set");
        const Trace t = run(g);
        check(t.pass_boundaries == 0, "num_passes=1: the gate never fires");
        bool in_order = true;
        for (size_t i = 0; i < t.visited.size(); ++i) in_order &= (t.visited[i] == (int64_t) i);
        check(in_order, "num_passes=1: visits 0..7 once each, in order");
    }

    std::printf(g_fail ? "loop_step_test: %d FAILED\n" : "loop_step_test: all passed\n", g_fail);
    return g_fail ? 1 : 0;
}
