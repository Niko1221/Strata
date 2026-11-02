// include/strata/flint/engine.hpp - the Flint 40B Preview (GGUF architecture flint) engine path:
// `strata` hands a flint model to it at the top of main() (the family switch), and strata-flint is a thin wrapper
// around the same entry point. See src/flint/flint_engine.cu.
#pragma once

namespace strata::flint {

/// The engine's command line (--model / --native, --max-context, --prefill / --chunk, --serve,
/// --expert-vram, --expert-slots, --hybrid-rows, --cpu-threads, --tokens, --max-new, --no-graph).
/// Returns the process exit code.
int run(int argc, char** argv);

}  // namespace strata::flint
