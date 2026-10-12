// include/strata/program/kv_splice_apply_int8.hpp - Phase 3: the INT8 K/V splice.
//
// WHAT THIS IS.  The fp16 splice (kv_splice_apply.hpp) rotates the cached K rows of
// [edit_pos, edit_pos + old_suffix_len) by a per-position RoPE delta, leaving V
// untouched.  INT8 K cannot be rotated in place: each cell is 64 int8 codes + 1 fp16
// scale, and rotating the codes would break the [code * scale] contract the dequant
// kernel relies on.  The fix: dequant the 64 codes to fp32, apply the rotation in fp32
// (the splice's RoPE math already runs in fp32: kv_splice_apply.hpp:228-238), requant
// the rotated image with the EXACT writer math (src/kernels/cuda/kv_q8.cu:47-67, mirrored
// in include/strata/kernels/kv_q8_quant.hpp:kv_q8_quant_group), write the codes + scale
// back.  V is untouched.
//
// MATH-MIRROR PROPERTY.  For the UNCHANGED range (cells outside the splice), the helper
// is a no-op: cells OUTSIDE [edit_pos, R) are not visited, and their codes+scales are
// BIT-IDENTICAL.  For the CHANGED range, requant(dequant(dequant->rotate->requant))
// equals dequant->rotate->requant on the original stored image (the dequant/requant
// pair round-trips BIT-EXACTLY because the writer's `rint(x/sf) * sf = x` is exact for
// the 254 quantable values: the only difference is the second quant's rounding of the
// ROTATED fp32, which pays the same per-group half-step the fresh-append path pays when
// it writes the rotated image to the cache).  T11.c pins this bit-for-bit; T11.b bounds
// the rotation-requant error to the same half-step the engine's other write paths pay.
//
// TRANSPORT.  The cached K is staged through the existing splice transport
// (kv_splice_transport.hpp): in a CUDA build the k_q and k_scale buffers are device
// pointers and the host loop must go through `stage_for_readwrite` / `commit`.  The
// transport's identity path (CPU build) aliases the buffers directly.  V is never
// touched; this header only stages K codes + K scales.
//
// FAIL-CLOSED.  Anything outside the supported config (kv_hybrid / kv_q4 / kv_resident >
// 0 / layer_split / kv_rot -- the opt-in STRATA_KV_ROT=1 -- K is stored through a
// Hadamard rotation that the splice's RoPE-only math does not undo) returns
// SpliceStatus::Unsupported* WITHOUT side effects, matching the A1.3 fail-closed pattern.
// The startup guard in suffix_reuse.hpp:188-204 keeps refusing q4_0 / k8v4 / kv_resident /
// layer_split at flag-parse; the runtime guard below is the second layer.
#pragma once

#include "strata/core/layer.hpp"                  // QsaState (kv_int8 / k_q / k_scale / kv_mode)
#include "strata/core/layout.hpp"                 // ModelGeometry
#include "strata/kernels/f16_bits.hpp"           // f16_from_f32 / f32_from_f16
#include "strata/kernels/kv_q8.hpp"              // KV_Q8_GROUP
#include "strata/program/kv_q8_quant.hpp"         // KvQ8Group + kv_q8_{quant,dequant,read,write}_group
#include "strata/program/kv_splice.hpp"          // rotate_k_by_delta + build_rope_table_host
#include "strata/program/kv_splice_apply.hpp"    // SpliceStatus, fp32/fp16 helpers
#include "strata/program/kv_splice_transport.hpp"  // Staged + stage_for_readwrite + commit
#include "strata/program/suffix_reuse.hpp"       // SuffixHookOutcome

#include <algorithm>
#include <cstdint>
#include <cstdio>

namespace strata::program::suffix_reuse {

/// Host FWHT-256 (Sylvester construction, 1/sqrt(256) normalized) - bit-exact same math
/// as the engine's kv_q4.hpp fwht256_cuda (kv_q4_parity.cpp:46-62 is the reference the
/// parity test pins it to).  H is its own inverse (H*H = I), which is what the kv_rot
/// splice sandwich relies on.  The splice is a host-loop helper by design (a tail of a
/// few thousand rows), so the row is already staged on the host when this runs.
inline void fwht256_row_host(float* x) {
    constexpr int n = 256;
    for (int s = 1; s < n; s *= 2)
        for (int i = 0; i < n; i += 2 * s)
            for (int j = 0; j < s; ++j) {
                const float u = x[i + j];
                const float v = x[i + j + s];
                x[i + j]     = u + v;
                x[i + j + s] = u - v;
            }
    const float inv = 1.0f / 16.0f;   // 1/sqrt(256)
    for (int i = 0; i < n; ++i) x[i] *= inv;
}

/// Phase 3 runtime scope guard for the int8 splice.  Returns Rotated/`NoOp` when the
/// config is supported; otherwise one of the `Unsupported*` codes WITHOUT side effects.
/// This is the int8-aware version of `splice_scope_guard` (kv_splice_apply.hpp:72-79):
/// plain `--kv int8` (kv_rot == false) is now ACCEPTED.  Hybrid K8V4 still refuses
/// (V is rotated Q4_0, the splice does not apply Hadamard to V); q4_0 / kv-resident /
/// layer-split keep the same refusals.  STRATA_KV_ROT=1 is refused: when K is stored
/// through a Hadamard rotation, the splice must undo it before applying RoPE - that
/// scope is "int8 + Hadamard" and is a follow-up (per the dequant-rotate-requant
/// section in plans/2026-10-11_184000-strata-suffix-reuse-phase2-v2.md).
inline SpliceStatus splice_scope_guard_int8(bool kv_int8, bool kv_q4, bool kv_hybrid,
                                            int64_t kv_resident, bool layer_split_active,
                                            bool kv_rot) {
    (void) kv_rot;   // accepted since the H-sandwich landed; head_dim!=256 refuses inside the helper
    (void) kv_hybrid;   // accepted (k8v4): the hybrid K half is int8; V is untouched
    if (!kv_int8) {
        // fp16 path: never comes here (the dispatch is the caller's); the q4/hybrid/resident
        // /split refusals are still enforced for safety.
        if (kv_q4 || kv_hybrid) return SpliceStatus::UnsupportedFormat;
        if (kv_resident > 0) return SpliceStatus::UnsupportedResident;
        if (layer_split_active) return SpliceStatus::UnsupportedSplit;
        return SpliceStatus::NoOp;   // fp16 K is the OTHER helper's domain
    }
    // int8 path.  kv_rot on == STRATA_KV_ROT=1: K rows are stored Hadamard-rotated; the
    // splice now handles it with the H -> R(delta) -> H sandwich (kv_splice_apply_int8.hpp;
    // H*H = I so the math is exact, and the requant count is unchanged).  The only refusal
    // left for kv_rot is head_dim != 256, enforced fail-closed INSIDE the rotation helper
    // (fwht256 is order-256 only) - not a config-level refusal.
    if (kv_q4) return SpliceStatus::UnsupportedFormat;
    if (kv_resident > 0) return SpliceStatus::UnsupportedResident;
    if (layer_split_active) return SpliceStatus::UnsupportedSplit;
    return SpliceStatus::NoOp;   // contract gate; the caller also checks fired + old_suffix_len
}

/// Phase 3 T11.c's test target: rotate the K blocks of ONE QsaState that correspond to
/// positions `[edit_pos, R)` in the OLD prompt, where `R = edit_pos + old_suffix_len`.
/// Mirrors `rotate_qsa_tail_inplace` (kv_splice_apply.hpp:162-243) in the int8 layout:
///
///   1. Bounds-check the splice range against the pool's row count (the same out-of-pool
///      refusal the fp16 path has).
///   2. Compute the rope-table slice the rotation reads (p_delta_max based on deltas).
///   3. Stage cos_tab + sin_tab (transport: identity on CPU, cudaMemcpy on device).
///   4. Stage the K codes range [first_row, last_row) AND the K scales range covering
///      the same rows.  The two are separate transport views (same Staged pool), so the
///      pair is read-write consistent: the host rotates codes+scales in lockstep.
///   5. For each (position, kv_head) the splice touches:
///        a. dequant every 64-value group to fp32
///        b. apply the splice's RoPE delta rotation in fp32 (rotate_k_by_delta, fp16-input
///           row staged via the local helper so the rope math stays the same as the fp16
///           splice path; the difference is the read/dequant before and the requant write
///           after).
///        c. requant every 64-value group with the writer's math (kv_q8_quant_group).
///   6. commit the two staged views (no-op on CPU, cudaMemcpy on device).
///
/// Parameters mirror `rotate_qsa_tail_inplace` (kv_splice_apply.hpp:162-168) for symmetry;
/// the only added constraint is `head_dim % KV_Q8_GROUP == 0` (the engine's qsa_state_init
/// already enforces this, kv_q8.cu:22-26 validate()).  V is untouched.  Returns true on
/// success; false on a guard failure (null pool, no table, page out of pool, etc.) with
/// NO side effects on the staged buffers (the staging slots are only mutated after a
/// successful loop).
inline bool rotate_qsa_tail_inplace_int8(strata::core::QsaState& st,
                                          int head_dim, int n_rot, int n_head_kv, int max_pos,
                                          int64_t page_size,
                                          int64_t edit_pos, int64_t old_suffix_len,
                                          bool uniform, int delta, const int64_t* deltas,
                                          int64_t pool_rows = 0) {
    // ---- Guard: identical to the fp16 path's fail-closed pattern ----
    if (st.k_q == nullptr || st.k_scale == nullptr) return false;
    if (st.cos_tab == nullptr || st.sin_tab == nullptr) return false;
    if (st.kv_mode != 0) return false;                                  // v1: identity-page mode only
    if (edit_pos < 0 || old_suffix_len <= 0) return false;
    if (!uniform && deltas == nullptr) return false;
    if (page_size <= 0) page_size = 1;
    if (head_dim % strata::kernels::KV_Q8_GROUP != 0) return false;        // writer's invariant (kv_q8.cu:22-26)
    // STRATA_KV_ROT=1: K rows are stored Hadamard-rotated (fwht256, kv_q4.hpp).  The splice
    // must undo H, apply the RoPE delta, re-apply H (H*H = I).  fwht256 is order-256 only:
    // a head_dim other than 256 in the rotated domain is refused fail-closed (the splice
    // helper would mix wrong).  The engine's rotated configs are all head_dim 256 today.
    if (st.kv_rot && head_dim != 256) return false;
    if ((int64_t) n_rot > head_dim) return false;
    const int groups = head_dim / strata::kernels::KV_Q8_GROUP;
    // Bounds: every position the loop touches must land inside the pool's rows.
    const int64_t last_pos = edit_pos + old_suffix_len - 1;
    if (pool_rows > 0) {
        const int64_t last_page = last_pos / page_size;
        const int64_t last_row = last_page * n_head_kv * page_size + (int64_t) (n_head_kv - 1) * page_size
                               + last_pos % page_size;
        if (last_row >= pool_rows) return false;
    }
    // ---- Rope-table slice (the rotation reads cos/sin at |p_delta|) ----
    int64_t max_abs_delta = 0;
    for (int64_t k = 0; k < old_suffix_len; ++k)
        max_abs_delta = std::max<int64_t>(max_abs_delta,
            uniform ? (delta < 0 ? -(int64_t) delta : (int64_t) delta)
                    : (deltas[k] < 0 ? -deltas[k] : deltas[k]));
    const int half = n_rot / 2;
    const int64_t tab_rows = std::min<int64_t>(max_pos, max_abs_delta + 1);
    if (tab_rows <= 0 || half <= 0) return false;
    const size_t tab_floats = (size_t) tab_rows * (size_t) half;
    transport::Staged cos_s = transport::stage_for_readwrite(st.cos_tab, tab_floats * 4);
    transport::Staged sin_s = transport::stage_for_readwrite(st.sin_tab, tab_floats * 4);
    if (cos_s.host == nullptr || sin_s.host == nullptr) return false;
    // ---- Stage the K codes + K scales ranges covering the splice rows ----
    // The first/last touched ROWS in the pool (row-major across pages and kv_heads).
    const size_t first_row = (size_t) (edit_pos / page_size) * (size_t) n_head_kv * (size_t) page_size;
    const size_t row_count = (size_t) ((last_pos / page_size) - (edit_pos / page_size) + 1)
                           * (size_t) n_head_kv * (size_t) page_size;
    const size_t k_codes_bytes = row_count * (size_t) head_dim * sizeof(int8_t);
    const size_t k_scales_elems = row_count * (size_t) groups;
    const size_t k_scales_bytes = k_scales_elems * sizeof(uint16_t);
    transport::Staged codes_s  = transport::stage_for_readwrite(
        st.k_q + first_row * (size_t) head_dim, k_codes_bytes);
    transport::Staged scales_s = transport::stage_for_readwrite(
        st.k_scale + first_row * (size_t) groups, k_scales_bytes);
    if (codes_s.host == nullptr || scales_s.host == nullptr) return false;
    int8_t*   codes  = (int8_t*)  codes_s.host;
    uint16_t* scales = (uint16_t*) scales_s.host;
    const float* cos_tab = (const float*) cos_s.host;
    const float* sin_tab = (const float*) sin_s.host;
    // ---- Rotate: dequant per group, apply RoPE delta, requant per group ----
    // Scratch: one fp32 row of head_dim.  The fp16 splice uses an fp16 stack scratch and
    // does the convert in-line; the int8 splice does TWO conversions (dequant + requant)
    // per group, so the scratch is fp32.  head_dim == 256 in the artifact, < 4096 stack
    // scratch limit (same as the fp16 helper, kv_splice_apply.hpp:218).
    float row_scratch[4096];
    if (head_dim > 4096) return false;
    for (int64_t k = 0; k < old_suffix_len; ++k) {
        const int64_t p = edit_pos + k;                       // position in OLD prompt
        const int64_t page = p / page_size;
        const int64_t off  = p % page_size;
        const int delta_row = uniform ? delta : (int) deltas[(size_t) k];
        for (int h = 0; h < n_head_kv; ++h) {
            const int64_t row_idx = page * n_head_kv * page_size + (int64_t) h * page_size + off;
            const size_t local_row = (size_t)(row_idx - (int64_t) first_row);
            // 1. Dequant: every group -> fp32.  The dequant math is `x' = code * sf`
            //    (mirror of src/kernels/cuda/kv_q8.cu:100-101).
            for (int g = 0; g < groups; ++g) {
                const float sf = strata::kernels::f32_from_f16(scales[local_row * groups + g]);
                for (int t = 0; t < strata::kernels::KV_Q8_GROUP; ++t)
                    row_scratch[g * strata::kernels::KV_Q8_GROUP + t] =
                        (float) codes[local_row * head_dim + g * strata::kernels::KV_Q8_GROUP + t] * sf;
            }
            // 2. Splice rotation (fp32, same as the fp16 helper's math but no fp16 round-trip
            //    around the rotation - the dequant is fp32, the rotation is fp32, the requant
            //    is fp32, the dequant in the gather is fp16.  The fp16 path does 2 fp16
            //    round-trips; the int8 path does 1 int8 round-trip on each end.  Bounded by T11.b).
            //    STRATA_KV_ROT=1: the stored row is H*R(p)*k (queries are H'd too, H^T*H = I),
            //    so dequant gives H*R(p)*k; undo H, rotate by delta, re-apply H to get
            //    H*R(delta)*H * (H*R(p)*k) = H*R(p+delta)*k - the exact stored form at the
            //    new position.  SAME single requant as the unrotated path (no extra noise).
            if (st.kv_rot) fwht256_row_host(row_scratch);
            strata::program::suffix_reuse::rotate_k_by_delta(
                row_scratch, 1, head_dim, n_rot, (int) p, delta_row,
                cos_tab, sin_tab, max_pos);
            if (st.kv_rot) fwht256_row_host(row_scratch);
            // 3. Requant: writer's math.  Per group, amax over the 64 rotated fp32 values
            //    (fabs), scale = f16(f32(amax/127)) (the round-to-nearest-even bit pattern
            //    the writer stores), code = clamp(rint(x / sf), -127, 127).  Mirrors
            //    src/kernels/cuda/kv_q8.cu:47-67 EXACTLY.
            for (int g = 0; g < groups; ++g) {
                strata::kernels::KvQ8Group qg{};
                strata::kernels::kv_q8_quant_group(&row_scratch[g * strata::kernels::KV_Q8_GROUP], qg);
                scales[local_row * groups + g] = qg.scale_bits;
                for (int t = 0; t < strata::kernels::KV_Q8_GROUP; ++t)
                    codes[local_row * head_dim + g * strata::kernels::KV_Q8_GROUP + t] = qg.codes[t];
            }
        }
    }
    // ---- Commit: copy the host-staged codes + scales back to the device on a CUDA build ----
    transport::commit(codes_s,  k_codes_bytes);
    transport::commit(scales_s, k_scales_bytes);
    return true;
}

}  // namespace strata::program::suffix_reuse
