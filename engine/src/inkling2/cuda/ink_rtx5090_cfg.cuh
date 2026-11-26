// engine/src/inkling2/cuda/ink_rtx5090_cfg.cuh - kept so the 5090 hand-over's includes keep working.
//
// The RTX 5090 switches are no longer decided here: STRATA_INK_RTX5090 / STRATA_INK_RTX5090_EXACT are aliases of the shared sm_120 switches
// STRATA_INK_SM120 / STRATA_INK_SM120_EXACT, and the card is matched by name, SM count and VRAM (not by "cc 12.0"), see ink_sm120_cfg.cuh
// and strata/core/ink_sm120_table.hpp. strata::ink::rtx5090::*_on() are wrappers over that decision.
#pragma once

#include "ink_sm120_cfg.cuh"
