// Native engine entry point for the hy_v4_flash GGUF architecture.
#pragma once

namespace strata::hy4_flash {

/// The engine's command line (strata's own flag names where they mean the same thing). Returns the process exit
/// code. Build-guarded: `strata` only calls this when STRATA_HAVE_HY4_FLASH is defined.
int run(int argc, char** argv);

}  // namespace strata::hy4_flash
