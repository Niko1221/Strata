// include/strata/dsv4/engine.hpp - the DeepSeek V4-Flash (deepseek4) engine path's entry point.
//
// Mirrors include/strata/qwen35/engine.hpp: the family switch at the top of generate.cpp's main() hands the
// command line over unchanged, and the standalone strata-dsv4 executable wraps the same call.  The engine itself
// is src/dsv4/dsv4_engine.cu; its geometry comes from include/strata/dsv4/geometry.hpp, never the shared
// core/kCompiledGeometry (Qwen's).
#pragma once

namespace strata::dsv4 {

/// Runs the DeepSeek V4-Flash engine on strata's own command line (--model/--native GGUF, --pack DIR, ...).
int run(int argc, char** argv);

}  // namespace strata::dsv4
