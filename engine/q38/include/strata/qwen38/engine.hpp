// Official engine API entry point for the fork's dense Q38 CUDA engine.
//
// generate.cpp's family switch calls this when the model's GGUF architecture
// is `qwen35` with the dense 64-layer geometry (ModelFamily::Qwen38Dense).
// All Q38 state stays inside this function; nothing is exported besides it.
#pragma once

namespace strata {
namespace qwen38 {

int run(int argc, char** argv);

}  // namespace qwen38
}  // namespace strata
