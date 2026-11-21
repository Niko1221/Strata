// Based on Project Maya (https://github.com/mw00/project-maya)
// Copyright (c) 2026 Peasant Smith, OscarSheng, alesha-pro, cobra91 and the Project Maya contributors
// Licensed under the MIT License.

// include/strata/ik2/engine.hpp - the Inkling 2 (GGUF architecture inkling2) engine path as a library:
// `strata` hands an inkling2 model to it at the top of main() (the family switch), and strata-ik2 is a thin wrapper
// around the same entry point. See src/ik2/ik2_engine.cu.
#pragma once

namespace strata::ik2 {

int run(int argc, char** argv);

}  // namespace strata::ik2
