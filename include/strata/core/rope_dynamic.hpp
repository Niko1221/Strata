// include/strata/core/rope_dynamic.hpp - --rope-scaling dynamic: no scaling while a dialog fits the trained
// context, yarn with a factor by its length past it (rounded up to a step).  The cached keys follow a switch by one
// more rotation (kernels/rope_dynamic.hpp), the table is rebuilt in place (graphs keep their pointers).
#pragma once

#include "strata/core/layer.hpp"
#include "strata/kernels/rope_scaling.hpp"

#include <cmath>
#include <cstdint>
#include <string>

namespace strata::core {

struct RopeDynamic {
    bool on = false;
    strata::kernels::RopeScaling base;   ///< the yarn knobs; `factor` is set per switch
    double orig = 262144;                ///< the trained context (yarn's original context)
    double step = 0.125;                 ///< factors are multiples of this
    double max_factor = 2.0;
    bool reread = false;                 ///< STRATA_ROPE_DYNAMIC_REREAD=1: a change reads the prompt again
    double live = 1.0;                   ///< the factor the session's K/V and checkpoints were written with
    double table = 1.0;                  ///< the factor the rope table holds now
    int64_t switches = 0, rotated_cells = 0;

    /// 1 up to `orig` positions, else ceil(positions / orig / step) * step, at most max_factor, on_grid.
    double factor_for(int64_t positions) const;
    /// f on the 2^-20 grid the session files store it on (conversation_file.cpp), so a factor read back from a parked
    /// conversation or a file is the same double the engine chose and an unchanged factor turns nothing
    static double on_grid(double f) { return std::ldexp(std::round(std::ldexp(f, 20)), -20); }
    /// The configuration of factor f: none at 1 (the startup table, bit for bit), else `base` with factor f.
    strata::kernels::RopeScaling config(double f) const;
    /// Positions the current table covers without extrapolation.
    int64_t limit() const { return (int64_t) (table * orig); }
};

/// Cells [0, cells) of a QSA state from `from` to `to`: K (int8 or fp16, with or without H) and, with `index`, the
/// indexer's pooled rows, its spare row and `dead`.  A streamed layer (--kv-resident) turns its host copy and drops
/// its VRAM slots; a ring (the drafter's) turns its host copy and the caller refills it.  Asynchronous on `stream`.
bool rope_rerotate_state(const QsaState& st, int64_t cells, bool index, const strata::kernels::RopeScaling& from,
                         const strata::kernels::RopeScaling& to, void* stream, std::string& error);

/// Rebuilds the table `owner` holds for `to`, in place, and makes `to` the process's scaling (rope_scaling_set and
/// the table registry).  Synchronizes the device first.
bool rope_table_rebuild(const QsaState& owner, const strata::kernels::RopeScaling& to, std::string& error);

}  // namespace strata::core
