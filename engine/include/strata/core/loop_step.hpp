// include/strata/core/loop_step.hpp - #7295: the recurrent-depth loop's layer-stepping decision, pulled
// out of `session_loop` so it can be unit-tested without a GPU or a model.
//
// `session_loop` (src/core/session.cpp) is almost entirely CUDA graph launches and doorbell polling -
// nothing in that is cheap to exercise in a test. But WHICH layer runs next, whether it is a pass
// boundary (where `session_loop` resets `R` from `loop_R0` and rotates the message bank - see
// LOOP_TRANSFORMER_PLAN.md §2a/§2b), and which graph variant to launch (`execs[next]` vs.
// `execs_pass[pass - 1][next]`) is pure arithmetic over `ModelGeometry`, with no CUDA dependency at all -
// so it is factored out here, and `session_loop` calls it directly. A test against THIS function
// exercises the real decision, not a re-implementation of it.
#pragma once

#include "strata/core/geometry.hpp"

namespace strata::core {

/// What `session_loop` should do after finishing layer `l` (`loop_pass` is the pass counter BEFORE this
/// step).
struct LoopStep {
    int64_t next;          ///< the layer whose `pre` graph launches next (`g.n_layers` means "done")
    int64_t loop_pass;     ///< `loop_pass` AFTER this step (unchanged unless this step crossed a boundary)
    bool pass_boundary;    ///< true for exactly one step per pass: reset `R` from `loop_R0`, rotate the
                           ///< message bank (`loop_bank_write` -> `loop_bank_read`/`loop_bank_out`), then
                           ///< launch `next` via the NEW pass's graph
    bool repeat_pass;      ///< true: `next`'s graph is `execs_pass[loop_pass - 1][next]`; false: `execs[next]`
};

/// **BYTE-IDENTICAL WHEN OFF**: `g.has_loop() == false` always returns `{l + 1, 0, false, false}` - the
/// one `if` below never taking its true branch is the entire difference the loop adds to the layer order.
inline LoopStep loop_next_layer(const ModelGeometry& g, int64_t l, int64_t loop_pass) {
    LoopStep r{l + 1, loop_pass, false, false};
    if (!g.has_loop()) return r;
    const bool at_range_end = (l == g.loop_core_end - 1);
    if (at_range_end && loop_pass + 1 < g.loop_num_passes) {
        r.next = g.loop_core_start;
        r.loop_pass = loop_pass + 1;
        r.pass_boundary = true;
        r.repeat_pass = true;
    } else if (loop_pass > 0 && r.next >= g.loop_core_start && r.next < g.loop_core_end) {
        r.repeat_pass = true;   // still inside the pass that is already under way
    }
    return r;
}

}  // namespace strata::core
