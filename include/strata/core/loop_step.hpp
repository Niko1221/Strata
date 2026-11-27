// include/strata/core/loop_step.hpp - #7295 M6: the recurrent-depth loop's layer-stepping decision, pulled
// out of `session_loop` so it can be unit-tested without a GPU.
//
// `session_loop` (src/core/session.cpp) is 95% CUDA graph launches and doorbell polling - nothing in THAT a
// host-only test can exercise cheaply, and the project has no existing harness that builds a synthetic
// WeightTable + captured graphs to run one end to end (M6 looked: `session_capture`/`session_loop` have zero
// dedicated tests anywhere in this repo today, only the one production call site in generate.cpp). But the
// one part of the loop that is pure arithmetic over `ModelGeometry` - which layer runs next, whether it reads
// `execs_noKV` or `execs`, and whether this is a pass boundary (where M5's gate fires) - has NO CUDA
// dependency at all, so it is factored out here and `session_loop` calls it directly. A test against THIS
// function exercises the real decision `session_loop` makes, not a re-implementation of it.
#pragma once

#include "strata/core/geometry.hpp"

namespace strata::core {

/// What `session_loop` should do after finishing layer `l` (`loop_pass` is the pass counter BEFORE this step).
struct LoopStep {
    int64_t next;        ///< the layer whose `pre` graph launches next (`g.n_layers` means "done")
    int64_t loop_pass;   ///< `loop_pass` AFTER this step (unchanged unless this step crossed a pass boundary)
    bool repeat_pass;    ///< true: `next`'s graph is `execs_noKV[next]`; false: `execs[next]`
    bool pass_boundary;  ///< true: layer `l` was the core range's last layer of the CURRENT pass (M5's gate
                         ///< fires here, once per pass, whether or not another pass follows)
};

/// **BYTE-IDENTICAL WHEN OFF**: `g.has_loop() == false` always returns `{l + 1, 0, false, false}` - the one
/// `if` below never taking its true branch is the entire difference the loop adds to the layer order.
inline LoopStep loop_next_layer(const ModelGeometry& g, int64_t l, int64_t loop_pass) {
    LoopStep r{l + 1, loop_pass, false, false};
    if (!g.has_loop()) return r;
    r.pass_boundary = (l == g.loop_core_end - 1);
    if (r.pass_boundary && loop_pass + 1 < g.loop_num_passes) {
        r.next = g.loop_core_start;
        r.loop_pass = loop_pass + 1;
        r.repeat_pass = true;
    } else if (loop_pass > 0 && r.next >= g.loop_core_start && r.next < g.loop_core_end) {
        r.repeat_pass = true;   // still inside the pass that is already under way
    }
    return r;
}

}  // namespace strata::core
