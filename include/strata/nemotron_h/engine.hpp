// include/strata/nemotron_h/engine.hpp - the engine path for NVIDIA Nemotron-H MoE files (GGUF architecture
// nemotron_h_moe: Nemotron 3.5 Super VL 120B-A12B), src/nemotron_h/nh_engine.cu.  `strata` switches to it from the
// file's architecture (the family switch at the top of its main); strata-nh runs it on its own.
#pragma once

namespace strata::nemotron_h {

/// The whole program: strata's command line (the options this path takes; --serve passes the config's), its exit code.
int run(int argc, char** argv);

}  // namespace strata::nemotron_h
