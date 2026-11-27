// engine/metal/src/hy4flash_gguf/dispatch.cpp - see dispatch.h.
#include "dispatch.h"

#include "mtl_rt.h"

#include <cstdlib>
#include <string>

namespace strata::hy4_flash {

std::string rejected_env() {
    // The arithmetic switches: ONE list, in mtl_rt, because `Context::lock_safe_math` ends the process on the same
    // set at compile time and two lists would drift.  This is the host-side face of it - a message rather than an
    // exit, so the engine can refuse before it has a device and so a test can exercise the path.
    if (const char* bad = mtl::deterministic_env_violation())
        return std::string(bad) +
               " is set, and this engine is frozen on safe math and f32 activations with a reduction order that "
               "does not depend on the window size (ADR 0001 §2, §4.2): it refuses to run rather than change the "
               "numerics silently";
    // The weight-corrupting one.  Read exactly as `stream.cpp` reads it (atoi != 0), so the values that would
    // really turn it on are the values refused here: with it on, every expert that is not resident is pointed at
    // slot 0 and nothing is read, so the answers are whatever that slot held.  That is a LOSSY path, and no lossy
    // path is allowed outside the MTP layer, so it is refused rather than tolerated.  (The Qwen engine keeps it:
    // there it is an explicit "wrong answers, measure the schedule" switch.)
    if (const char* f = std::getenv("STRATA_METAL_STREAM_FAKE"); f != nullptr && *f && std::atoi(f) != 0)
        return "STRATA_METAL_STREAM_FAKE=" + std::string(f) +
               " is set, which makes the expert arena point every missing expert at slot 0 and read nothing "
               "(metal/src/engine/stream.cpp): the expert kernels would then multiply by whatever that slot holds. "
               "It is a timing path with WRONG answers, and rule R0 allows no lossy path outside the MTP layer, so "
               "this engine refuses to start";
    return {};
}

}  // namespace strata::hy4_flash
