// include/strata/program/kv_splice_apply.hpp - Phase 2 Task 2: the runtime `apply_splice` that walks
// the QSA K pool and rotates the unchanged tail by the splice plan's per-position delta.  V is
// untouched.  This is a SIBLING header to `suffix_reuse.hpp` (which holds the PURE plan computation
// and the runtime hook); the split keeps `suffix_reuse.hpp` lightweight for tests and call sites
// that only need the plan, while `apply_splice` brings in `QsaState`, `ModelGeometry`, the rope
// helpers, and the splice functions.  Nothing here allocates - the session owns `k_pool`,
// `cos_tab`, `sin_tab`, and `max_cells`.
//
// v1 SCOPE (review-r2 NEW-2 / r1 Blockers 1+2): the splice SUPPORTS ONLY fp16 K, no kv-resident
// ring, no layer-split.  Anything else - int8 codes, q4_0 rotated codes, hybrid k8v4, a streamed
// slot-ring, a multi-stage layer split - logs `SPLICE_UNSUPPORTED: <format|kv_resident|layer_split>`
// to stderr and returns WITHOUT touching K.  The runtime hook at generate.cpp falls through to
// today's re-read path, which is byte-identical to before.  The startup guard (Task 4) refuses
// the same combination at flag-parse time, so this runtime guard is the second layer of the
// A1.3 fail-closed pattern.  Dequant-rotate-requant (int8/q4_0) and streaming-ring splice = Phase 3.
//
// `void* stream` is a future CUDA seam (review-r2 NEW-3): today's path is a HOST LOOP, no kernel
// launch (T3 already established the host port of `rotate_row_by_delta` and pinned it to 1e-6).
// The parameter is reserved for a future GPU kernel that operates on the same per-row layout;
// leaving it in the signature now keeps the call site from churning when that lands.  The
// SUPPORTS guard checks happen BEFORE the stream is consulted.
//
// FP16 STORAGE NOTES: `QsaState.k_pool` is documented `uint16_t* fp16 (null in INT8 mode)` at
// include/strata/core/layer.hpp:210 - one fp16 per cell of head_dim.  The math runs in fp32
// (T3's tolerance is 1e-6, fp32 ULP), so each row is converted fp16->fp32 on entry, rotated, and
// written back to fp16 on exit.  The fp16 rounding is the SAME on the splice path and the fresh-
// compute path (both produce the same fp16-stored cell), so the round-trip preserves the cache's
// numerical ground truth.  No re-quantization; the fp16 stored K is rotated and stored.
//
// POOLED-LAYOUT NOTES: K is laid out `[page][kv_head][page_size][head_dim]` (qsa.cu:130) - a
// single position `p` has its K split across `n_head_kv` different rows, one per kv-head.  Phase 2
// v1 supports the guard-checked mode (fp16, kv_resident==0, no layer-split) where `st.kv_mode==0`
// is identity: position `p` lives in page `p/page_size`, offset `p%page_size`, and row index
// `page_idx * n_head_kv * page_size + h * page_size + offset_within_page`.  The splice walks
// every kv_head's row at each position in the tail.  V is untouched (separate `st.v_pool`).
//
// TWO-TIER API: `rotate_qsa_tail_inplace(st, plan, head_dim)` is the lower-level helper that
// operates on a single QsaState (T6's unit test target).  `apply_splice(outcome, ss, g, stream)`
// is the SessionState-level wrapper that walks the layer range and calls the helper per owned
// QSA layer.
#pragma once

#include "strata/kernels/kv_stream.hpp" // for kKvF16 / kKvInt8 / kKvQ4 / kKvHybrid enum (kv_stream.hpp:48)
#include "strata/core/layer.hpp"        // QsaState, qsa_kv_int8/q4/hybrid/resident (header-only, no CUDA)
#include "strata/core/layout.hpp"        // ModelGeometry + is_qsa_layer
#include "strata/program/kv_splice.hpp"  // rotate_k_by_delta + build_rope_table_host
#include "strata/program/kv_splice_transport.hpp"  // host/device staging (identity on CPU builds)
#include "strata/program/suffix_reuse.hpp"  // SuffixHookOutcome

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstring>

namespace strata::program::suffix_reuse {

/// Splice outcome tag for the runtime caller - what happened.  `Rotated` is the success path;
/// `Unsupported*` are the SPLICE_UNSUPPORTED log lines (caller falls through to today's re-read);
/// `NoOp` is the contract-false path (fired=false or old_suffix_len<=0, no log).
enum class SpliceStatus {
    NoOp = 0,            ///< contract not satisfied - fired=false or empty tail.  No log.
    Rotated,             ///< K rotated in place.  V untouched.
    UnsupportedFormat,   ///< kv != fp16 (int8 / q4_0 / hybrid).  Logged + no-op.
    UnsupportedResident, ///< kv_resident > 0 (streaming ring).  Logged + no-op.
    UnsupportedSplit,    ///< layer split active.  Logged + no-op.
};

/// Pure predicate for the runtime scope guard (review-r2 NEW-2 / r1 Blockers 1+2).  When this
/// returns Rotated/`NoOp` we proceed; otherwise we log SPLICE_UNSUPPORTED and return without
/// touching K.  Factored out of `apply_splice` so T6 (and the test suite) can pin the guard
/// against pure inputs without linking to layer.cpp (where `qsa_kv_int8()`/etc. live as globals).
inline SpliceStatus splice_scope_guard(bool kv_int8, bool kv_q4, bool kv_hybrid,
                                       int64_t kv_resident, bool layer_split_active) {
    if (kv_int8 || kv_q4 || kv_hybrid) return SpliceStatus::UnsupportedFormat;
    if (kv_resident > 0) return SpliceStatus::UnsupportedResident;
    if (layer_split_active) return SpliceStatus::UnsupportedSplit;
    return SpliceStatus::NoOp;  // contract-false (not enough signal to say Rotated yet) - the
                                 // caller also checks fired and old_suffix_len
}

namespace detail {
/// fp16 <-> fp32 round-trip (IEEE 754 binary16, no NaN preservation - K never holds NaN, and
/// `__half` is not portable here so we hand-roll the conversions; they round to nearest even).
inline uint16_t fp32_to_fp16_bits(float f) {
    uint32_t x = 0u; std::memcpy(&x, &f, 4);
    uint32_t sign = (x >> 16) & 0x8000u;
    int32_t exp = (int32_t) ((x >> 23) & 0xff) - 127 + 15;
    uint32_t mant = x & 0x7fffffu;
    if (exp <= 0) {
        if (exp < -10) return (uint16_t) sign;
        mant = (mant | 0x800000u) >> (uint32_t) (1 - exp);
        if (mant & 0x1000u) mant += 0x2000u;
        return (uint16_t) (sign | (mant >> 13));
    }
    if (exp >= 31) return (uint16_t) (sign | 0x7bf0u | (((x & 0x7fffffu) != 0u) ? 0x1u : 0u));
    if (mant & 0x1000u) { mant += 0x2000u; if (mant & 0x800000u) { mant = 0u; ++exp; if (exp >= 31) return (uint16_t) (sign | 0x7bf0u); } }
    return (uint16_t) (sign | ((uint32_t) exp << 10) | (mant >> 13));
}
inline float fp16_bits_to_fp32(uint16_t h) {
    uint32_t sign = ((uint32_t) h & 0x8000u) << 16;
    uint32_t exp  = ((uint32_t) h >> 10) & 0x1fu;
    uint32_t mant = (uint32_t) h & 0x3ffu;
    uint32_t bits;
    if (exp == 0) {
        if (mant == 0) bits = sign;
        else {
            uint32_t e = 0;
            while ((mant & 0x400u) == 0) { mant <<= 1; ++e; }
            mant &= 0x3ffu;
            bits = sign | ((uint32_t) (113 - e) << 23) | (mant << 13);
        }
    } else if (exp == 31) {
        bits = sign | 0x7f800000u | (mant << 13);
    } else {
        bits = sign | ((exp + 112u) << 23) | (mant << 13);
    }
    float f = 0.0f; std::memcpy(&f, &bits, 4); return f;
}
}  // namespace detail

/// T6's test target: rotate the k_pool rows of ONE QsaState that correspond to positions
/// `[edit_pos, R)` in the OLD prompt, where `R = edit_pos + old_suffix_len`.  This is the
/// pure-CPU loop the SessionState wrapper invokes per owned Qsa layer.
///
/// MEMORY TRANSPORT (the 2026-10-12 SIGSEGV hotfix).  In a CUDA build `st.k_pool`,
/// `st.cos_tab` and `st.sin_tab` are DEVICE addresses (the session arena is cudaMalloc'd,
/// and qsa_state_init uploads the rope table INTO st.cos_tab).  This loop therefore never
/// dereferences those pointers directly: `stage_for_readwrite` brings a contiguous range
/// into a host staging buffer (identity on a CPU-only build, cudaMemcpy otherwise) and
/// `commit` writes it back.  Three ranges are staged per call:
///   1. the K rows of `[edit_pos, R)` - read, rotated, written back;
///   2. the rope-table SLICE the rotation reads - `[0, min(max_pos, p_delta_max + 1))`
///      rows of cos/sin, which for a mid-prompt edit is a small prefix of the 64 MiB table;
///   3. nothing else - V is untouched.
///
/// Parameters:
///   - `st`: the QsaState whose `k_pool` is rotated (must be fp16, kv_mode==0).
///   - `head_dim`: 256 for the artifact.
///   - `n_rot`: 64 for the artifact.
///   - `n_head_kv`: 2 for the artifact.
///   - `max_pos`: the rope-table extent (== the state's max_cells).
///   - `page_size`: the kv_stream page_size (qsa_real_shapes().page_size; 4 in the
///     artifact - the SAME source qsa_state_init sizes the pool with).
///   - `pool_rows`: the k_pool's ROW count (st.n_slots * n_head_kv * page_size).  The
///     row index of every touched position is checked against it; an out-of-range index
///     (a plan the pool cannot hold) refuses the rotation instead of writing past the
///     pool.  0 disables the check (T6 synthesises states without n_slots set).
///   - `edit_pos`: the splice's plan_edit_pos.
///   - `old_suffix_len`: the splice's plan.old_suffix_len.
///   - `uniform`: true for single delta, false for per-token deltas.
///   - `delta`: the uniform position offset when `uniform` is true.
///   - `deltas`: per-row deltas when `uniform` is false (size == old_suffix_len; may be
///     null ONLY when `uniform` is true - a non-uniform plan without a table is refused
///     rather than dereferenced, the empty-deltas trap the live plan can hit when the
///     suffix is empty).
///   - `skip_rotation`: T6's RED hook - skips the `rotate_k_by_delta` call so the test proves
///     a regression that drops the rotation surfaces as `diff_count > 0`.  Production callers
///     pass `false`.
///
/// Returns true on success (K rotated).  No allocation on the CPU path; the CUDA path
/// stages through the shared transport buffer (kv_splice_transport.hpp).
inline bool rotate_qsa_tail_inplace(strata::core::QsaState& st,
                                     int head_dim, int n_rot, int n_head_kv, int max_pos,
                                     int64_t page_size,
                                     int64_t edit_pos, int64_t old_suffix_len,
                                     bool uniform, int delta, const int64_t* deltas,
                                     bool skip_rotation = false,
                                     int64_t pool_rows = 0) {
    if (st.k_pool == nullptr) return false;
    if (st.cos_tab == nullptr || st.sin_tab == nullptr) return false;
    if (st.kv_mode != 0) return false;                         // v1: identity-page mode only
    if (edit_pos < 0 || old_suffix_len <= 0) return false;
    if (!uniform && deltas == nullptr) return false;           // no table to walk: refuse, don't fault
    if (page_size <= 0) page_size = 1;
    // Bounds: every position the loop touches must land inside the pool's rows.  Checked
    // BEFORE the first staging copy so a bad plan never reads or writes anything.
    const int64_t last_pos = edit_pos + old_suffix_len - 1;
    if (pool_rows > 0) {
        const int64_t last_page = last_pos / page_size;
        const int64_t last_row = last_page * n_head_kv * page_size + (int64_t) (n_head_kv - 1) * page_size
                               + last_pos % page_size;
        if (last_row >= pool_rows) return false;               // plan exceeds the pool: no-op, caller re-reads
    }
    // The rope-table slice: the rotation reads rows [0, p_delta_max] where p_delta_max is
    // the largest |effective delta| (rotate_row_by_delta clamps to max_pos-1 itself, so
    // staging min(max_pos, max|delta|+1) rows covers every read).
    int64_t max_abs_delta = 0;
    for (int64_t k = 0; k < old_suffix_len; ++k)
        max_abs_delta = std::max<int64_t>(max_abs_delta,
            uniform ? (delta < 0 ? -(int64_t) delta : (int64_t) delta)
                    : (deltas[k] < 0 ? -deltas[k] : deltas[k]));
    const int half = n_rot / 2;
    const int64_t tab_rows = std::min<int64_t>(max_pos, max_abs_delta + 1);
    if (tab_rows <= 0 || half <= 0) return false;
    // Stage the rope-table slice and the K rows (transport = identity on a CPU build).
    const size_t tab_floats = (size_t) tab_rows * (size_t) half;
    transport::Staged cos_s = transport::stage_for_readwrite(st.cos_tab, tab_floats * 4);
    transport::Staged sin_s = transport::stage_for_readwrite(st.sin_tab, tab_floats * 4);
    if (cos_s.host == nullptr || sin_s.host == nullptr) return false;   // staging failed: refuse the layer
    const size_t first_row = (size_t) (edit_pos / page_size) * (size_t) n_head_kv * (size_t) page_size;
    const size_t row_count = (size_t) ((last_pos / page_size) - (edit_pos / page_size) + 1)
                           * (size_t) n_head_kv * (size_t) page_size;
    const size_t k_elems = row_count * (size_t) head_dim;
    transport::Staged k_s = transport::stage_for_readwrite(
        st.k_pool + first_row * (size_t) head_dim, k_elems * 2);
    if (k_s.host == nullptr) return false;
    const float* cos_tab = (const float*) cos_s.host;
    const float* sin_tab = (const float*) sin_s.host;
    // The staged table covers rows [0, tab_rows); rotate_row_by_delta clamps its p_delta
    // into [0, max_pos-1] - with the slice staged, every in-slice row it picks is present.
    // Production reads fp16 -> fp32, rotates in fp32, writes back to fp16.  The fp16 round-trips
    // add 2 quantizations (cached image at OLD_pos stored fp16, splice-path output stored fp16).
    // The fresh-comparison path does 2 quantizations of its own (raw -> fp16, rope -> fp16), so
    // both paths have the same structural fp16 footprint.  T6 compares the two paths using the
    // SAME fp16 round-trip step, so the test tolerates ~fp16 ULP errors between them (T3 pinned
    // the underlying rotation math to 1e-6 absolute tolerance; this function is fp16-correct).
    float row_scratch[ /*head_dim*/ 4096];
    if (head_dim > 4096) return false;                          // stack scratch bound; artifact is 256
    for (int64_t k = 0; k < old_suffix_len; ++k) {
        const int64_t p = edit_pos + k;                       // position in OLD prompt
        const int64_t page = p / page_size;
        const int64_t off  = p % page_size;
        const int delta_row = uniform ? delta : (int) deltas[(size_t) k];
        for (int h = 0; h < n_head_kv; ++h) {
            const int64_t row_idx = page * n_head_kv * page_size + (int64_t) h * page_size + off;
            uint16_t* row = (uint16_t*) k_s.host
                          + (size_t)(row_idx - (int64_t) first_row) * (size_t) head_dim;
            for (int i = 0; i < head_dim; ++i) row_scratch[i] = detail::fp16_bits_to_fp32(row[i]);
            // Production rotation.  The `skip_rotation=true` arg lets T6 (and any future
            // RED-armed regression test) skip the rotation step to prove the assertion fails
            // without it.  The brief's RED proof was exercised during build: masking this call
            // makes T6.y's `ulp == strict+ulp+bad` and the regression-guardrail checks fail.
            if (!skip_rotation) {
                strata::program::suffix_reuse::rotate_k_by_delta(
                    row_scratch, 1, head_dim, n_rot, (int) p, delta_row,
                    cos_tab, sin_tab, max_pos);
            }
            for (int i = 0; i < head_dim; ++i) row[i] = detail::fp32_to_fp16_bits(row_scratch[i]);
        }
    }
    transport::commit(k_s, k_elems * 2);
    return true;
}

/// Walk the QSA K pools of the session and rotate, in place, the rows corresponding to positions
/// `[plan_edit_pos, plan_edit_pos + old_suffix_len)` of every owned QSA layer by the splice plan's
/// per-position delta.  V is untouched.
///
/// Defined in `kv_splice_apply_session.hpp` (separate header that pulls `SessionState`/CUDA-runtime
/// include); the kernel-only helper above + the runtime's `apply_splice` are the two layers, with
/// T6 covering the helper directly without paying the CUDA include cost.

}  // namespace strata::program::suffix_reuse
