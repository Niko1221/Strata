// include/strata/tp/tp_diag.hpp - TPDIAG: TEMPORARY tensor-parallel divergence instrumentation.
//
// This is the TP-owned home of the TPDIAG DEVICE LOG that the two-card hunt added to `src/core/verify.cpp`:
// the log and its reserve/dump/record helpers.  It lives here, not in the core file, so a rebase onto an
// upstream Strata release conflicts only at the probe seams that stay in `verify.cpp` - when the question is
// answered, deleting this header, `src/tp/tp_diag.cpp`, and those seams is the whole removal.  Nothing
// outside a TP run compiles into the default path beyond a call that returns at its first gate check.
//
// The gates: STRATA_TP_DIAG_LAYERS=1 arms the log and the minimal probe set, STRATA_TP_DIAG_FULL=1 adds the
// full set (about 430 records per window; with them on the GPU stage hung at 587 s against 83 s clean,
// measured on two cards).  Every function here checks its gate BEFORE touching any CUDA, so with the
// environment off the default path stays byte-identical to a build without this file.
//
// The `tpdiag_*` names keep their original spelling so the log lines and a `git grep TPDIAG` still match.
// The scattered probes themselves (the record CALLS and their inline staging blocks inside
// `Verifier::record_window`/`run`) stay in verify.cpp for now: collapsing them into one-line calls here is
// the follow-up the N-5 hunt (docs/TP2-HANDOVER.md section 0a) will want anyway.
#pragma once

#include <cstddef>
#include <cstdint>

namespace strata::tp {

/// Arm the log for a window of `dn = T * HC * N` floats (560 slots).  Called from `Verifier::init`'s carve.
void tpdiag_log_reserve(size_t dn);

/// Read the recorded slots back and print them.  Blocking D2H by design: a diagnostic, not a path.
void tpdiag_log_dump(int T, int64_t HC, int64_t N, const char* tag, int64_t win);

/// One capture-legal device-to-device copy of `n` floats from `buf` into `slot` of this rank's log.
void tpdiag_reduce_record(const float* buf, int slot, int64_t n, int64_t stride, void* cs);

/// The residual ENTERING the layer loop (slot 192).
void tpdiag_input_record(const float* buf, int T, int64_t HC, int64_t N, void* cs);

/// The residual at the top of `pre(l)` (the full-probe slots 0..95 / 96..191).
void tpdiag_pre_record(const float* buf, int64_t l, int grp, int T, int64_t HC, int64_t N, void* cs);

/// The env gates (cached statics, checked before any CUDA touch).
bool tpdiag_layer_on();
bool tpdiag_full_on();

/// The log and its capacity, per rank: a probe that stages bytes INTO the log (the conv filter) reads these
/// directly, exactly as it did inside verify.cpp.
extern float* g_tpdiag_log[2];
extern size_t g_tpdiag_cap[2];

}  // namespace strata::tp
