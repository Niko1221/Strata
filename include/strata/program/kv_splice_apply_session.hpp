// include/strata/program/kv_splice_apply_session.hpp - Phase 2 Task 2: SessionState wrapper for
// `apply_splice`.  The kernel-only helper (`rotate_qsa_tail_inplace`) lives in `kv_splice_apply.hpp`
// (header-only, no CUDA include) so T6 can exercise it directly.  This header pulls in
// `SessionState`/`cuda_runtime.h` and provides the production runtime wrapper that walks every
// owned QSA layer of the session and invokes the helper.
#pragma once

#include "strata/core/session.hpp"        // SessionState (CUDA-runtime include, runtime-only)
#include "strata/core/layout.hpp"         // ModelGeometry + is_qsa_layer
#include "strata/kernels/qsa.hpp"         // qsa_real_shapes().page_size (the pool's page granule)
#include "strata/program/kv_splice_apply.hpp"        // SpliceStatus + rotate_qsa_tail_inplace
#include "strata/program/kv_splice_apply_int8.hpp"   // Phase 3: int8 splice (dequant->rotate->requant)

#include <cstdio>

namespace strata::program::suffix_reuse {

/// Walk the QSA K pools of the session and rotate, in place, the rows corresponding to positions
/// `[plan_edit_pos, plan_edit_pos + old_suffix_len)` of every owned QSA layer by the splice plan's
/// per-position delta.  V is untouched.
///
/// **GUARDS BEFORE TOUCHING K (each returns SpliceStatus::Unsupported* WITHOUT side effects and
/// logs `SPLICE_UNSUPPORTED: <format|kv_resident|layer_split>` to stderr):**
///   1. KV format != fp16 (`qsa_kv_int8() / qsa_kv_q4() / qsa_kv_hybrid()` returns true)
///   2. `qsa_kv_resident() > 0` (slot-ring / streamed mode)
///   3. layer split active (`g.layer_lo != 0 || g.layer_hi != g.n_layers`)
///
/// The runtime guard complements Task 4's startup guard (which refuses the same combination at
/// flag parse), forming the two-layer A1.3 fail-closed pattern.  When the guard fires the caller
/// falls through to today's re-read path byte-identically.
inline SpliceStatus apply_splice(const SuffixHookOutcome& out,
                                 strata::core::SessionState& ss,
                                 const strata::core::ModelGeometry& g,
                                 void* /*stream*/) {
    if (!out.fired) return SpliceStatus::NoOp;
    if (out.old_suffix_len <= 0) return SpliceStatus::NoOp;
    // ---- Phase 3 dispatch: which splice path?  fp16 K (the v1 path) or int8 K (Phase 3
    //      dequant->rotate->requant).  Read the format ONCE so the two paths' guards stay
    //      independent and a future format (q4_0, hybrid) still gets refused uniformly.
    const bool kv_int8_now   = strata::core::qsa_kv_int8() || strata::core::qsa_kv_hybrid();
    const bool kv_other      = strata::core::qsa_kv_q4();           // --kv q4_0 (Phase 3 follow-up)
    const int64_t kv_res_now = strata::core::qsa_kv_resident();
    const bool layer_split_active = (ss.layer_lo != 0) || (ss.layer_hi != g.n_layers);
    // Hoisted above both format branches: the device-pointer safety checks inside each
    // branch use n_qsa before its old (post-branch) declaration site.  Same class as the
    // e51880c n_qsa hoist in generate.cpp - only the engine TU (generate.cpp) compiles
    // this header, so the tests-only CPU tree never catches the use-before-decl.
    const int64_t n_qsa = g.n_qsa_layers();
    if (kv_int8_now) {
        // ---- int8 path (Phase 3).  Uses the int8-aware guard; refuses if STRATA_KV_ROT=1
        //      put K through a Hadamard rotation (out of scope for the daily preset).
        const SpliceStatus g_status = splice_scope_guard_int8(
            /*kv_int8=*/true, /*kv_q4=*/kv_other, /*kv_hybrid=*/strata::core::qsa_kv_hybrid(),
            kv_res_now, layer_split_active, /*kv_rot=*/strata::core::qsa_kv_int8_rotate());
        if (g_status != SpliceStatus::NoOp) {
            const char* reason = (g_status == SpliceStatus::UnsupportedResident)
                ? "int8 + kv_resident > 0 (streaming-ring splice is Phase 3)"
                : (g_status == SpliceStatus::UnsupportedSplit)
                  ? "int8 + layer_split active (per-stage is Phase 3)"
                  : (strata::core::qsa_kv_int8_rotate()
                      ? "int8 + STRATA_KV_ROT=1 (Hadamard K domain is a follow-up)"
                      : "int8 + q4_0 / hybrid / kv_resident / layer_split are Phase 3");
            std::fprintf(stderr, "[strata] suffix-reuse: SPLICE_UNSUPPORTED: %s.  "
                         "Falling through to today's re-read path (degrade, never crash).\n", reason);
            return g_status;
        }
        // device-pointer safety check (same hotfix as the fp16 path below)
        if (n_qsa > 0) {
            const auto& st0 = ss.qsa_states[ss.qsa_ord0];
            if (transport::pointer_is_device(st0.k_q) && !transport::is_active()) {
                std::fprintf(stderr, "[strata] suffix-reuse: SPLICE_UNSUPPORTED: device buffers "
                             "without transport (int8 path).  Falling through to the re-read path.\n");
                return SpliceStatus::UnsupportedFormat;
            }
        }
    } else {
        // ---- fp16 path (the v1 path).  Existing guard: q4 / hybrid / resident / split.
        if (strata::core::qsa_kv_q4() || strata::core::qsa_kv_hybrid()) {
            std::fprintf(stderr, "[strata] suffix-reuse: SPLICE_UNSUPPORTED: kv format != fp16 "
                         "(q4_0/hybrid).  Dequant-rotate-requant for q4_0 is a follow-up.\n");
            return SpliceStatus::UnsupportedFormat;
        }
        if (kv_res_now > 0) {
            std::fprintf(stderr, "[strata] suffix-reuse: SPLICE_UNSUPPORTED: kv_resident=%lld "
                         "(slot-ring/streamed mode).  Streaming-ring splice is Phase 3.\n",
                         (long long) kv_res_now);
            return SpliceStatus::UnsupportedResident;
        }
        if (layer_split_active) {
            std::fprintf(stderr, "[strata] suffix-reuse: SPLICE_UNSUPPORTED: layer_split "
                         "(layer_lo=%lld, layer_hi=%lld of %lld).  Per-stage splice is Phase 3.\n",
                         (long long) ss.layer_lo, (long long) ss.layer_hi, (long long) g.n_layers);
            return SpliceStatus::UnsupportedSplit;
        }
        // device-pointer safety check (the 2026-10-12 SIGSEGV hotfix - fp16 path)
        if (n_qsa > 0) {
            const auto& st0 = ss.qsa_states[ss.qsa_ord0];
            if (transport::pointer_is_device(st0.k_pool) && !transport::is_active()) {
                std::fprintf(stderr, "[strata] suffix-reuse: SPLICE_UNSUPPORTED: device buffers "
                             "without transport.  k_pool is a device address but the splice "
                             "transport is not active on this build.  Falling through to the "
                             "re-read path (degrade, never crash).\n");
                return SpliceStatus::UnsupportedFormat;
            }
        }
    }
    // Range: [k_rotation_start, k_rotation_start + old_suffix_len).  Phase 3a T9: the
    // runtime hook at generate.cpp:9050 narrows the local `resume` AFTER this function
    // returns, so the splice ranges are derived directly from the outcome (not from the
    // narrowed `resume`).  `k_rotation_start` is the suffix-walk-derived start of the
    // unchanged tail in OLD (= plan.edit_pos = n_old - old_suffix_len).  This is
    // DIFFERENT from `out.plan_edit_pos` (the splice hook's narrowing signal = LCP).
    // The two values agree for mid-prompt edits but diverge for continuations (where
    // old_suffix_len == 0 gates apply_splice out anyway).
    const int64_t edit_pos = out.k_rotation_start;
    const int64_t old_suffix_len = out.old_suffix_len;
    const int head_dim = (int) g.head_dim;
    const int n_rot    = (head_dim >= 64) ? 64 : head_dim;
    const int n_head_kv = (int) g.n_head_kv;
    const int max_pos  = (int) (ss.qsa_states[ss.qsa_ord0].max_cells);
    // `page_size` is the KV page granule qsa_state_init sizes the pool with:
    // `qsa_shapes(g).page_size`, which is `qsa_real_shapes().page_size` with the model's
    // head counts substituted (layer.cpp:507).  It equals idx_block (4) in this artifact
    // only by construction; the value is taken from the same source the pool layout uses,
    // NOT from qsa_interval (r1's "page-size coincidence" - the two 4s are different facts:
    // one in every 4th LAYER is full attention; one page = 4 CELLS).
    const int64_t page_size = strata::kernels::qsa_real_shapes().page_size > 0
                                  ? strata::kernels::qsa_real_shapes().page_size
                                  : 1;
    for (int64_t j = 0; j < n_qsa; ++j) {
        strata::core::QsaState& st = ss.qsa_states[ss.qsa_ord0 + j];
        if (st.cos_tab == nullptr || st.sin_tab == nullptr) continue;
        if (st.kv_mode != 0) continue;
        // The pool's row count: n_slots pages of n_head_kv*page_size rows.  kv_mode==0 is
        // identity-mapped, so position p's rows are exactly the rows the splice walks.
        const int64_t pool_rows = st.n_slots > 0 ? st.n_slots * n_head_kv * page_size : 0;
        if (kv_int8_now) {
            // Phase 3 int8 path: dequant per group, rotate, requant (kv_splice_apply_int8.hpp).
            if (st.k_q == nullptr || st.k_scale == nullptr) continue;
            (void) rotate_qsa_tail_inplace_int8(
                st, head_dim, n_rot, n_head_kv, max_pos, page_size,
                edit_pos, old_suffix_len,
                out.uniform, (int) out.delta,
                out.deltas.empty() ? nullptr : out.deltas.data(),
                pool_rows);
        } else {
            // fp16 path: rotate in place (kv_splice_apply.hpp).
            if (st.k_pool == nullptr) continue;
            (void) rotate_qsa_tail_inplace(st, head_dim, n_rot, n_head_kv, max_pos, page_size,
                                           edit_pos, old_suffix_len,
                                           out.uniform, (int) out.delta,
                                           out.deltas.empty() ? nullptr : out.deltas.data(),
                                           /*skip_rotation=*/false, pool_rows);
        }
    }
    return SpliceStatus::Rotated;
}

}  // namespace strata::program::suffix_reuse
