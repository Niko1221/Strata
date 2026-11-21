// include/strata/core/engine_version.hpp - the engine's own version string.
//
// The project version (CMakeLists.txt) says the release; the flavour says which build of it: this tree's engine
// carries the measured q4u32 default and calls itself "0.1.63-q4u32", an upstream build has no suffix.  The numbers
// in front are what versions are compared by (strata_control/kv_default.py reads them out of the string), so the
// suffix never changes which engine counts as newer.
#pragma once

#ifndef STRATA_ENGINE_VERSION
#define STRATA_ENGINE_VERSION STRATA_VERSION
#endif
