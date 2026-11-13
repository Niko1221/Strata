// include/strata/core/tp_runtime.hpp - the seam between the layer code and the two-card runtime.
//
// A TENSOR-PARALLEL RANK COMPUTES PARTIAL SUMS.  `ssm_out` and `attn_output` are cut on the contraction axis,
// so each rank's projection covers half the columns of its activation and the two halves have to be added
// before the next layer can use the result.  The layer code therefore OWES an all-reduce at exactly those two
// points - every layer, twice - and the default path has to stay byte-identical while it does.
//
// This class is that switch, and it is deliberately a null test rather than a flag threaded through every
// signature: `current()` is null on a single-card run, so a call site reads
//
//     if (strata::core::TpRuntime* rt = strata::core::TpRuntime::current()) {
//         if (!rt->reduce(y, n, stream, err)) return false;
//     }
//
// and a null means "nothing is owed" rather than "nothing happened".  The all-reduce itself is the tested
// primitive in `kernels/allreduce.hpp` (gate: `tp_allreduce_test`, two cards, token lockstep 62/62); what does
// not exist yet is the object that owns the two devices, the `ArPair` and the host threads that drive the two
// ranks.  Until it does, `--tp` is refused by the gate in `program/generate.cpp` - so this seam is dormant in
// every run that can currently start, and cannot produce a half sum by accident.
#pragma once

#include <cstdint>
#include <cstdlib>
#include <string>

#include "strata/core/shard.hpp"   // tp_ranks_current

namespace strata::core {

/// STRATA_GDN_SPLIT=1 asks for the 4-block re-cut of each head's state columns in `gdn_step_norm_multi`;
/// a tensor-parallel PAIR refuses it by default - the re-cut is not the geometry a rank owns (24 heads,
/// its own state block).  Measured on a pair with the long answer as the judge: with the split on, a
/// telegraphic word salad ('user ask: say hello to the person. user ask: take create annotation chat not
/// append system do review model nuclear programming challenge...'); with it off, complete sentences.
/// Turning off the neighbouring GDN kernels as well made it worse (3 distinct of 8 against 2), so this
/// one is the difference.  One card is unaffected: tp_ranks_current() is 1 there and the split still runs.
inline bool allow_gdn_split() {
    const char* v = std::getenv("STRATA_GDN_SPLIT");
    return v != nullptr && v[0] == '1' && tp_ranks_current() <= 1;
}

/// TPDIAG (temporary): the stream the current verify window runs on, published by `Verifier::record_window`.
/// The no-MoE diagnostic zeroes a buffer INSIDE that window, and a capture only accepts work on its own stream -
/// a blocking `cudaMemset` there hangs the request (measured).
inline void*& tpdiag_window_stream_cell() { static void* s = nullptr; return s; }
inline void tpdiag_set_window_stream(void* s) { tpdiag_window_stream_cell() = s; }
inline void* tpdiag_window_stream() { return tpdiag_window_stream_cell(); }


class TpRuntime {
public:
    virtual ~TpRuntime() = default;
    /// Add the peer rank's partial sum INTO `y` (`n` floats) on `stream`.  False with `err` set when the two
    /// ranks could not meet: a rank that cannot reduce must not carry on with half a sum, because the result
    /// would look plausible.
    virtual bool reduce(float* y, int64_t n, void* stream, std::string& err) = 0;

    static TpRuntime* current() { return current_; }
    /// The engine installs one for a TP run and clears it (nullptr) for a single-card one.
    static void install(TpRuntime* rt) { current_ = rt; }

private:
    /// PER THREAD, NOT PER PROCESS.  Both ranks of a tensor-parallel run are threads in one process (that is
    /// how the two windows stay in lockstep while each drives its own card), so a process-wide pointer would
    /// have rank 1's install shadow rank 0's - and rank 0 would then reduce into rank 1's collective.  A
    /// single-card run has no thread that installs anything, so every thread still reads null.
    inline static thread_local TpRuntime* current_ = nullptr;
};

/// The all-reduce a tensor-parallel rank OWES at a contraction point (`ssm_out`, `attn_output`): add the
/// peer's partial sum INTO `y` (`n` floats) on `stream`.  True with nothing done on a single-card run,
/// where `current()` is null and `y` is already the whole projection.  False means "fail this request":
/// a rank that cannot reduce must not carry on with half a sum, because the result would look plausible.
inline bool tp_owed_reduce(float* y, int64_t n, void* stream, std::string& err) {
    TpRuntime* tp = TpRuntime::current();
    return tp == nullptr || tp->reduce(y, n, stream, err);
}

}  // namespace strata::core
