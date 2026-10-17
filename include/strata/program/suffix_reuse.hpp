// include/strata/program/suffix_reuse.hpp - the suffix-reuse plan computation: the longest common tail
// of an old and a new prompt and the per-token RoPE delta that re-uses the old prompt cache for it.
//
// On a mid-prompt edit at position E, the tokens [E, n_old) and [E, n_new) are the unchanged SUFFIX;
// the request path can splice the cached K/V for them with a rotation of K (V is untouched - it is the
// same content), and replay the linear state (GDN recurrence, PLE history, QSA indexer tails) from E.
// Today, `limit_reuse(first_dirty)` at include/strata/core/conversation_cache.hpp:221 discards the
// tail past `first_dirty`; this header ADDS the splice-plan call path that the flag `--cache-suffix-reuse`
// enables - by default off (P0 invariant: flag absent or 0 -> today's behaviour byte-identical).
//
// The plan struct is the contract the request path consumes.  `uniform == true` means one rotation
// (`delta`) handles the whole tail; `uniform == false` means the per-token table (`deltas`) carries
// the per-position offsets `rope_neox_pair` walks.  `replay_from` is set to the edit position so the
// deferred GDN replay can pick it up (A1.5: deliverable is the field + routing, no stub; with replay
// unimplemented, routing falls through to the existing re-read path unchanged).
//
// The plan is pure and deterministic, header-only, so `suffix_reuse_test.cpp` walks it by hand.
//
// QSA INDEXER TAILS ARE REBUILD-ON-REREAD, NOT MAINTAINED ON THE SPLICE PATH.  The splice rotates
// K rows in place and leaves V untouched - positions for the tail are re-derived per prefill step
// from `pos_base + pos` (src/core/layer.cpp:1058 uploads `host_pos[h] = pos_base + pos` every
// step), so no per-cell positions table exists.  The indexer's per-side state (`tails/dead/block_pos`,
// `include/strata/kernels/qsa.hpp:185-209`) is snapshot-and-restored via ConversationCheckpoint
// (`src/core/conversation_state.cpp:154-193`); on a re-read the indexer's `idx_tail` is refilled
// for [E, L), `block_pos` is set fresh, and `idx_dead` is a per-sequence constant.  None of it
// requires splice-specific work beyond running the prefill [E, L).  Phase 2 splice: no-op here.
#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <utility>
#include <vector>

namespace strata::program::suffix_reuse {

/// A checkpoint candidate's identity for invalidation: just its token length (the chain is sorted) and
/// the caller's stable index (a pointer back to the underlying `ConversationCheckpoint` or its slot).
/// `length` is what `invalidate_checkpoints` drops on; the index lets the caller remove the live entry.
struct CheckpointLength {
    int64_t length = 0;    ///< the checkpoint's token length (sorted ascending forms the chain)
    size_t  index  = 0;    ///< caller's stable index into the underlying checkpoint array
};

/// The splice plan for a re-edited prompt.
///
/// `edit_pos` is the position in the OLD prompt at which the cache stops being valid - the splice
/// reuses the cache cells with index `>= edit_pos`, the prefix path (conversation_cache.hpp:217-218
/// `ConversationKvReuse.unchanged_tokens`) keeps `edit_pos` valid as the unused prefix.  When the
/// prompts are identical the common tail is the whole prompt and `edit_pos == 0` (the path that
/// uses `edit_pos` short-circuits on `edit_pos == old_suffix_len`).
///
/// `old_suffix_len` and `new_suffix_len` are the lengths of the common tail beyond `edit_pos` - the
/// splice reuses these from the old cache.  On a uniform delta they are equal; on a per-token delta
/// table they can differ (e.g. an insert further right than a delete moves the boundary).
///
/// `uniform == true`: one rotation of K by `delta` (e.g. +k for a pure insert of k tokens).  `deltas`
/// stays empty.  `uniform == false`: per-token position offsets - each `deltas[i]` is the offset to
/// rotate cached token (old index `edit_pos + i`) by, to land it at new position `edit_pos + i + offset`.
///
/// `replay_from` mirrors `edit_pos`: the position the deferred linear-state replay (GDN recurrence,
/// PLE history, QSA indexer tail) starts from.  With replay unimplemented (A1.5), the call site falls
/// through to the existing re-read path unchanged.
struct SuffixReusePlan {
    int64_t edit_pos = 0;            ///< smallest index of divergence (LEGACY: suffix-walk-derived; the start of the K rotation range in OLD).  For backward compat with the original T1/T4 tests and the K rotation range.
    int64_t narrow_pos = 0;          ///< Phase 3a T9: longest common PREFIX (first index of divergence from the FRONT).  The splice hook's narrowing signal - where to narrow resume and which checkpoints to keep.  For a pure continuation, narrow_pos == n_old == resume (the splice is a no-op).  For a mid-prompt edit at E, narrow_pos == E.
    int64_t old_suffix_len = 0;      ///< common-tail tokens in the OLD prompt (the K rotation range in OLD is [n_old - old_suffix_len, n_old))
    int64_t new_suffix_len = 0;      ///< common-tail tokens in the NEW prompt
    bool    uniform = false;         ///< true -> single rotation by `delta`
    int64_t delta = 0;               ///< the uniform position offset (0 when `uniform` is false)
    std::vector<int64_t> deltas;     ///< per-token offsets when `uniform` is false (size = new_suffix_len)
    int64_t replay_from = 0;         ///< position the linear-state replay starts from (== narrow_pos, the splice hook's narrowing position)
};

/// Compute the splice plan for a re-edited prompt.  `old_tokens` is the prompt the cache holds; `new_tokens`
/// is the request's prompt.  The longest common PREFIX sets `narrow_pos` (the first index of divergence
/// from the front - the splice hook's narrowing signal).  The longest common SUFFIX sets
/// `old_suffix_len`/`new_suffix_len` and the per-position rotation (the unchanged tail from the back - the
/// K rows the splice rotates in place).  `edit_pos` (legacy) is the suffix-walk-derived start of the K
/// rotation range in OLD (= n_old - old_suffix_len).  V is untouched.  Pure, header-only.
/// O(n + m) on the lengths.
///
/// Phase 3a T9 (the live 0% vs 98% off-arm regression): `narrow_pos` (LCP-based) is the right
/// notion of "edit position" for the splice hook.  The longest common SUFFIX walk the previous
/// version used mis-IDed a pure continuation as a 1-token edit whenever the last token of `old`
/// happened to equal the last token of `new` (the live probe5 case: the EOS token 248046 at the
/// end of the resume checkpoint also ends the appended CONT).  The suffix walk produced
/// `suffix=1, edit_pos = n_old - 1`, the splice hook narrowed `resume` to `n_old - 1`, erased
/// the resume checkpoint, and the mount was destroyed.  The LCP-based `narrow_pos` is `n_old`
/// for a pure continuation (the resume checkpoint IS a prefix of the new prompt by definition -
/// the mount condition), so the splice hook does not narrow and the mount survives.
///
/// The continuation special case (LCP == n_old): the splice is a no-op.  `old_suffix_len` and
/// `new_suffix_len` are forced to 0 (any common-suffix tokens past the LCP are by definition
/// coincidental, e.g. the EOS token - they don't represent real content match worth rotating).
/// The runtime then sees `narrow_pos == n_old == resume` (no narrowing) and `old_suffix_len == 0`
/// (no K rotation), byte-identical to flag-off.
template <class T>
SuffixReusePlan compute_suffix_plan(const std::vector<T>& old_tokens, const std::vector<T>& new_tokens) {
    SuffixReusePlan plan;
    const size_t n_old = old_tokens.size();
    const size_t n_new = new_tokens.size();
    // ---- narrow_pos: longest common PREFIX from the FRONT (Phase 3a T9) ----
    // The first index of divergence is the position in the OLD prompt at which the cache
    // stops being valid.  For a pure continuation (`old` is a prefix of `new`) the LCP is
    // n_old and narrow_pos == n_old - the splice hook's narrowing predicate `narrow_pos < resume`
    // is false (narrow_pos == resume), no narrowing.  For a mid-prompt edit at E, LCP = E
    // (the first E tokens match, token E differs), narrow_pos = E, the splice hook narrows
    // resume to E and the chain is filtered.
    size_t prefix = 0;
    while (prefix < n_old && prefix < n_new &&
           old_tokens[prefix] == new_tokens[prefix]) ++prefix;
    plan.narrow_pos = (int64_t) prefix;
    plan.replay_from = plan.narrow_pos;
    // ---- old_suffix_len / new_suffix_len: longest common SUFFIX from the BACK (the K rotation range) ----
    size_t suffix = 0;
    while (suffix < n_old && suffix < n_new &&
           old_tokens[n_old - 1 - suffix] == new_tokens[n_new - 1 - suffix]) ++suffix;
    plan.old_suffix_len = (int64_t) suffix;
    plan.new_suffix_len = (int64_t) suffix;
    // ---- edit_pos (LEGACY): suffix-walk-derived start of the K rotation range in OLD ----
    // = n_old - suffix.  Kept for backward compat with the T1/T4 tests and any consumer
    // that used the old semantics.  The splice hook should use `narrow_pos` instead.
    plan.edit_pos = (int64_t) n_old - (int64_t) suffix;
    // ---- Phase 3a T9: continuation special case ----
    // The resume checkpoint is a prefix of cur (LCP == n_old): the splice is a no-op.  Any
    // common-suffix tokens are coincidental (e.g. the EOS token at the turn boundary that
    // appears at the end of both old and new); they do not represent content the splice
    // can usefully rotate - rotating a single EOS K row by +54 lands it past the new prompt's
    // last position, and the prefill overwrites it anyway.  Set tail lengths to 0 so the
    // runtime's `apply_splice` is gated out (line 9088 in generate.cpp: `if (old_suffix_len > 0)`).
    // narrow_pos stays at n_old (= resume) so the runtime does not narrow.  edit_pos (legacy)
    // stays at n_old - suffix = 0 if the EOS matches, or n_old if not - the legacy field's
    // value is irrelevant when old_suffix_len == 0.
    if (prefix == n_old) {
        plan.old_suffix_len = 0;
        plan.new_suffix_len = 0;
        plan.uniform = true;            // a no-op plan is trivially uniform (delta unused)
        plan.delta = 0;
        plan.deltas.clear();
        return plan;
    }
    // uniform offset: only when every old-position-to-new-position pair in the tail adds the same delta
    // (a pure insert or a pure delete; a per-token table is the mixed case)
    if (suffix > 0) {
        const int64_t uniform_delta = (int64_t) n_new - (int64_t) n_old;
        bool ok = true;
        plan.deltas.assign(suffix, 0);
        for (size_t i = 0; i < suffix; ++i) {
            const int64_t new_pos = (int64_t) (n_new - suffix + i);
            const int64_t old_pos = (int64_t) (n_old - suffix + i);
            plan.deltas[i] = new_pos - old_pos;
            if (plan.deltas[i] != uniform_delta) ok = false;
        }
        if (ok) {
            plan.uniform = true;
            plan.delta = uniform_delta;
            plan.deltas.clear();
        }
    }
    return plan;
}

/// Drop every checkpoint candidate with `length > edit_pos`.  Returns how many were dropped.  `chain`
/// is sorted ascending by `length`; this is pure and the caller owns the live `ConversationCheckpoint`
/// array (the loop on the call site uses the returned `index` to identify survivors).  The empty-edit
/// case (`edit_pos == 0`) drops the entire chain.
inline size_t invalidate_checkpoints(std::vector<CheckpointLength>& chain, int64_t edit_pos) {
    size_t dropped = 0;
    size_t i = 0;
    while (i < chain.size()) {
        if (chain[i].length > edit_pos) { chain.erase(chain.begin() + (ptrdiff_t) i); ++dropped; }
        else ++i;
    }
    return dropped;
}

/// A1.3 / A2.1 fail-closed guards: the splice refuses the same combination
/// `strata generate` refuses at startup.  Kept here as a CPU-testable mirror of the run-time check
/// (the runtime check is in `strata/program/generate.cpp` post-flag-parse; this is the
/// same predicate, separated only so `suffix_reuse_test` can exercise it without GPU).
/// Returns the (non-empty) reason the combination is refused, or empty when all conditions
/// are met.
///
/// Phase 2 Task 4 (review-r2 NEW-2): the startup guard (v1.1) refuses q4_0 / hybrid K
/// (`--kv q4_0` and `--kv k8v4`), the kv-resident slot ring (`--kv-resident > 0`), and
/// the layer split when the splice flag is on.  Phase 3 (int8 KV splice): `--kv int8` is
/// NOW SUPPORTED (the daily preset's mode).  The int8 K is dequantized, rotated by the
/// splice's RoPE delta, and requantized in place (kv_splice_apply_int8.hpp).  Fail-closed:
/// STRATA_KV_ROT=1 (K through a Hadamard rotation) is still a Phase 3 follow-up and the
/// runtime hook (kv_splice_apply_session.hpp) refuses it explicitly when the flag is on.
inline std::string validate_suffix_reuse_options(int64_t cache_suffix_reuse, int prompt_cache,
                                                 const std::string& kv_format,
                                                 int64_t kv_resident, bool layer_split_active,
                                                 int64_t prompt_cache_every = 0,
                                                 int64_t dense_tail = 0) {
    if (cache_suffix_reuse <= 0) return {};
    if (prompt_cache < 2)
        return "--cache-suffix-reuse N requires --prompt-cache >= 2";
    // q4_0: ACCEPTED (same as int8/k8v4).  The Q4 helper does the exact H-sandwich with
    // Q4_0 dequant/requant (suffix_reuse_int8_test T11.f measures the codec noise); falls
    // through to the kv_resident / layer_split refusals below like every quant format.
    if (kv_format != "fp16" && kv_format != "int8" && kv_format != "k8v4" && kv_format != "q4_0")
        return "--cache-suffix-reuse N requires --kv fp16 or --kv int8 (see plans/2026-10-11_184000-strata-suffix-reuse-phase2-v2.md)";
    if (kv_resident > 0)
        return "--cache-suffix-reuse N is unsupported with --kv-resident (streaming-ring splice "
               "is Phase 3)";
    if (layer_split_active)
        return "--cache-suffix-reuse N is unsupported with --layer-split (per-stage splice is "
               "follow-up work; see plans/2026-10-11_130800-strata-suffix-reuse.md A2.1)";
    if (dense_tail > 0 && prompt_cache_every <= 0)
        return "--prompt-cache-dense-tail N requires --prompt-cache-every > 0 (dense checkpoints "
               "are a cadence on the same periodic mechanism)";
    return {};
}

/// FW-1: position-dependent periodic-checkpoint cadence.  PURE so it is CPU-testable
/// without a GPU (suffix_reuse_dense_tail_test).  Returns the `done` threshold at/after
/// which the NEXT periodic checkpoint fires, given the position just checkpointed (`done`,
/// 0 when starting), the prompt's total length AS KNOWN at decision time (`pp_total` - the
/// prompt pass discovers it incrementally, so callers re-ask as it grows), the periodic
/// cadence (`every`, > 0), and the dense-tail width (`dense_tail`; 0 = OFF).
///
///   dense_tail == 0  -> next = done + every  (EXACTLY today's cadence, generate.cpp's
///                       `pp_next_check = done + o.prompt_cache_every`; byte-identical
///                       behavior when the flag is off).
///   dense_tail > 0   -> the last `dense_tail` tokens of the prompt are the DENSE zone:
///                       cadence `every` there (mid-prompt edits land near the tail - the
///                       splice needs an anchor at or before the edit).  Before the zone
///                       the cadence is max(every, 8192) (sparse head: the cold read pays
///                       for far fewer checkpoint pauses; one deep anchor survives for the
///                       deep-prefix rescue).  A sparse next that would overshoot the
///                       dense zone CLAMPS to the zone start - the zone is never skipped,
///                       whatever the head cadence stepped over.
///
/// Fail-closed by construction: every path returns a threshold > done (or the zone start
/// when clamping; the caller's `done >= threshold` check fires there immediately).
inline int64_t dense_tail_next_check(int64_t done, int64_t pp_total, int64_t every,
                                     int64_t dense_tail) {
    if (every <= 0) return INT64_MAX;                 // periodic checkpoints off
    if (dense_tail <= 0) return done + every;         // OFF: today's exact cadence
    const int64_t zone_start = pp_total > dense_tail ? pp_total - dense_tail : 0;
    if (done >= zone_start) return done + every;      // dense zone (or a tiny prompt)
    const int64_t sparse = std::max<int64_t>(every, 8192);
    const int64_t next = done + sparse;
    return next > zone_start ? zone_start : next;     // never skip the dense zone
}

/// FW-1 (dynamic sizing): the chunk size the prompt pass should use for the chunk
/// starting at c0, so that CHUNK BOUNDARIES (the only places checkpoints can fire)
/// land at dense-tail cadence inside the zone while the head reads in big chunks.
///
///   dense_tail <= 0            -> min(big, n - c0)            (today's uniform sizing)
///   c0 before the dense zone   -> min(big, zone_start - c0)   (a big chunk that ends
///                                 EXACTLY at the zone start: the first anchor)
///   c0 inside the dense zone   -> min(every, n - c0)          (anchor-sized chunks)
///
/// Pure; pinned by suffix_reuse_dense_tail_test DTT8-10.  `big` is the --prefill chunk
/// size (auto = the whole prompt).  The dense_tail_next_check cadence aligns with these
/// boundaries by construction (zone start + every-steps), so generate.cpp's checkpoint
/// gate needs no change.
inline int64_t dense_tail_chunk_size(int64_t c0, int64_t n, int64_t big, int64_t every,
                                     int64_t dense_tail) {
    if (n - c0 <= 0) return 0;
    if (dense_tail <= 0 || every <= 0) return std::min(big, n - c0);
    const int64_t zone_start = n > dense_tail ? n - dense_tail : 0;
    if (c0 < zone_start) {
        const int64_t head = std::min(big, zone_start - c0);   // land exactly at zone start
        return std::max<int64_t>(head, 1);
    }
    return std::max<int64_t>(std::min(every, n - c0), 1);      // anchor-sized zone chunks
}

/// FINDING 1: the parity predicate (the runtime hook's fire decision).  Centralised here so
/// `generate.cpp` and the off-parity test call THE SAME function.  At flag==0 (or "flag absent",
/// which is 0 by default) the predicate returns false and the hook is a no-op - today's reuse
/// accounting is byte-identical.  The function is pure (no side effects) so a header test can
/// pin byte-equality against today's path without GPU.
inline bool suffix_reuse_should_invalidate(int64_t flag, bool from_live, int64_t resume,
                                           bool checks_nonempty, int64_t cur_size) {
    if (flag <= 0) return false;                     // the canonical "flag absent" or "flag off"
    if (from_live) return false;                     // live sessions do not need a splice
    if (resume <= 0) return false;                   // no checkpoint reachable
    if (!checks_nonempty) return false;              // nothing to invalidate
    if (cur_size <= resume) return false;            // a strict continuation - no edit past the mounted prefix
    return true;
}

/// FINDING 4(a): the splice hook's invalidation result.  When the predicate says fire, we compute
/// the plan against the resume checkpoint's tokens (NOT checks.front() - the deepest match is
/// the one the request actually mounts through, and the shortest checkpoint would always yield
/// edit_pos=0 on a pure continuation, narrowing `resume` and pessimising the common case).  We
/// then drop checkpoints with `length > plan.edit_pos` from `chain` and surface the plan.
///
/// Phase 2 Task 1: the outcome also carries the splice plan's fields so the runtime `apply_splice`
/// in this file can read them.  The names mirror `SuffixReusePlan` (old_suffix_len, new_suffix_len,
/// deltas, replay_from) so a single struct describes both decisions: which checkpoints to keep and
/// which K rows to rotate.  Fields are populated on the `fired` path only; at `!fired` the runtime
/// sees zeros, an empty `deltas`, and the doc says "no splice planned".  This is a pure struct
/// extension - no behaviour change; the runtime consumer lands in Task 2.
struct SuffixHookOutcome {
    std::vector<CheckpointLength> chain;   ///< the chain after invalidation (== input when not fired)
    bool fired = false;                     ///< the predicate's verdict (== false -> chain untouched)
    int64_t plan_edit_pos = 0;             ///< Phase 3a T9: the splice hook's narrowing signal (LCP-based, == SuffixReusePlan::narrow_pos).  Where to narrow resume and which checkpoints to keep.
    int64_t k_rotation_start = 0;          ///< Phase 3a T9: the K rotation start in OLD (suffix-walk-derived, == n_old - old_suffix_len, == SuffixReusePlan::edit_pos).  The unchanged tail from the back - the K rows the splice rotates in place.  For a pure continuation this is 0 (irrelevant; old_suffix_len == 0 gates apply_splice out).
    // ---- Phase 2 Task 1: splice plan fields surfaced for `apply_splice` (Task 2) ----
    int64_t old_suffix_len = 0;            ///< common-tail tokens in the OLD prompt (== SuffixReusePlan::old_suffix_len)
    int64_t new_suffix_len = 0;            ///< common-tail tokens in the NEW prompt (== SuffixReusePlan::new_suffix_len)
    std::vector<int64_t> deltas;            ///< per-token position offsets (uniform case: empty, single delta suffices)
    int64_t replay_from = 0;                ///< position the linear-state replay (Task 5) starts from; == plan.narrow_pos
    bool    uniform = false;                ///< true -> single rotation by `delta`; false -> per-token `deltas`
    int64_t delta = 0;                      ///< uniform position offset when `uniform` is true
    // ---- Phase 3b: partial-prefix mount anchor ----
    // When > 0, the splice path mounted a partial-prefix checkpoint (its LCP is `plan_edit_pos` <
    // `restore_anchor_length`): the runtime restores the checkpoint's running state at length
    // `restore_anchor_length` and the chunk loop reads from `plan_edit_pos` (= the LCP).  The K
    // rotation operates on [plan_edit_pos, restore_anchor_length) in OLD.  When 0 (the default
    // and the strict-mount / off-flag path), the runtime uses `narrowed_resume_target` as before.
    int64_t restore_anchor_length = 0;
};
inline SuffixHookOutcome suffix_reuse_apply_invalidation(
        int64_t flag, bool from_live, int64_t resume,
        const std::vector<CheckpointLength>& chain,
        const std::vector<int32_t>& resume_checkpoint_tokens,
        const std::vector<int32_t>& cur_tokens) {
    SuffixHookOutcome out;
    out.chain = chain;
    if (!suffix_reuse_should_invalidate(flag, from_live, resume, !chain.empty(),
                                        (int64_t) cur_tokens.size())) {
        return out;
    }
    out.fired = true;
    // F4(a) + Phase 3a T9: plan against the resume checkpoint's tokens, not checks.front().
    // The plan now uses the longest common PREFIX (`narrow_pos`) for the splice hook's
    // narrowing signal instead of the suffix-walk-derived `edit_pos`.  For a pure continuation
    // (resume_ck.tokens is a prefix of cur_tokens, LCP == n_old), the plan is a no-op:
    // narrow_pos == n_old == resume (the splice hook's narrowing predicate `narrow_pos < resume`
    // is false), old_suffix_len == 0 (apply_splice is gated out, no K rotation).  The chain
    // drop predicate `length > narrow_pos` drops only the post-resume periodic checkpoints
    // (e.g. 3244 > 3240), exactly the same drops the off-flag canonical erase at
    // generate.cpp:9283-9285 makes - byte-identical.  For a mid-prompt edit at E, LCP = E,
    // narrow_pos = E < resume, and the chain is filtered to checkpoints with length <= E
    // (the anchor below the edit survives).
    SuffixReusePlan plan = compute_suffix_plan(resume_checkpoint_tokens, cur_tokens);
    // Phase 3a T9: outcome.plan_edit_pos is the splice hook's narrowing signal (narrow_pos,
    // LCP-based).  outcome.k_rotation_start is the K rotation start in OLD (edit_pos,
    // suffix-based) - the unchanged tail from the back.  The legacy suffix-derived
    // `plan.edit_pos` is still surfaced for the T1/T4 backward-compat tests.
    out.plan_edit_pos = plan.narrow_pos;
    out.k_rotation_start = plan.edit_pos;
    // Phase 2 Task 1: surface the splice plan's fields so the runtime `apply_splice` can read
    // them straight off the outcome (no second pass over resume_checkpoint_tokens / cur_tokens).
    // Names mirror `SuffixReusePlan` to keep both structs in lockstep (r1 Minor 5: name match).
    out.old_suffix_len = plan.old_suffix_len;
    out.new_suffix_len = plan.new_suffix_len;
    out.uniform       = plan.uniform;
    out.delta         = plan.delta;
    out.deltas        = plan.deltas;
    out.replay_from   = plan.replay_from;
    // the canonical invalidation: drop checkpoints with length > narrow_pos (= the splice
    // hook's narrowing signal).  For a pure continuation narrow_pos == n_old == resume, so the
    // predicate drops nothing past resume (the off-flag path's canonical erase at
    // generate.cpp:9283-9285 then drops anything > resume, byte-identical).  For a mid-prompt
    // edit at E, narrow_pos < resume and the chain is filtered to length <= E.
    size_t dropped = 0;
    size_t i = 0;
    while (i < out.chain.size()) {
        if (out.chain[i].length > out.plan_edit_pos) {
            out.chain.erase(out.chain.begin() + (ptrdiff_t) i);
            ++dropped;
        } else ++i;
    }
    (void) dropped;
    return out;
}

/// Phase 2 wiring bug 2 (T8): the splice hook narrows `resume` to `plan_edit_pos`, and
/// its own erase (in `suffix_reuse_apply_invalidation`'s caller at generate.cpp:9034-9043)
/// removes the exact-length checkpoint the restore search later wants.  This pure
/// helper decides, given a narrowed resume and the live `checks` chain, where the
/// request continues from - the gap-read / replay path phase 2 was designed for.
///
/// `chain_lengths` carries only the candidate lengths (the helper compares sizes
/// only); generate.cpp projects its `ConvCheckpoint::ids.size()` into a vector at the
/// call site so this stays header-only / CPU-testable.
///
/// `NarrowedResumeTarget::restore_length` is the checkpoint length the request
/// restores through; `read_from` is the position the chunk loop reads from (== 0 when
/// no usable mount survives - a fresh prefill from the start, NO error).
///
/// Cases:
///   - exact-length candidate exists:  restore_length = that length, read_from = narrowed_resume
///                                     (no gap; byte-equivalent to today's restore path).
///   - no exact-length, deepest-below: restore_length = deepest length <= narrowed_resume,
///                                     read_from = narrowed_resume (the [restore, narrowed)
///                                     gap is re-read by the existing prefill path).
///   - no candidate at or below:       restore_length = 0, read_from = 0
///                                     (no usable mount; full re-prefill, no error).
///
/// In every case the request proceeds; the helper never returns a state that maps to
/// the pre-fix "ERR restoring a conversation checkpoint failed" path.
struct NarrowedResumeTarget {
    int64_t restore_length = 0;   ///< checkpoint length the request restores through (0 -> fresh read)
    int64_t read_from      = 0;   ///< position the chunk loop reads from (= 0 when restore_length == 0)
    // Phase 3b: partial-prefix mount anchor.  When > 0, the runtime restores a checkpoint of
    // length `restore_anchor_length` (the partial-match candidate C, length L > P) and the
    // chunk loop reads from `read_from` (= the LCP P).  The strict-mount path leaves this at
    // 0 and uses `restore_length` as before.  Set by the partial-mount branch in generate.cpp
    // when `parked.partial` is true and the partial helper fires.
    int64_t restore_anchor_length = 0;
};
inline NarrowedResumeTarget narrowed_resume_target(int64_t narrowed_resume,
                                                   const std::vector<int64_t>& chain_lengths) {
    NarrowedResumeTarget t{};
    if (narrowed_resume <= 0 || chain_lengths.empty()) {
        t.restore_length = 0;
        t.read_from      = 0;
        return t;
    }
    int64_t best = -1;
    for (int64_t L : chain_lengths) {
        if (L <= narrowed_resume && L > best) best = L;
    }
    if (best < 0) {
        // No surviving check at or below the narrowed resume: the splice hook erased
        // the resume checkpoint and there is no deeper mount to fall back to.  Phase
        // 2 design's gap-read path reduces here to a fresh prefill from 0 - we do
        // NOT synthesise a fake checkpoint (a fake checkpoint with mismatched KV
        // would be silently wrong).  The request proceeds; no error.
        t.restore_length = 0;
        t.read_from      = 0;
        return t;
    }
    t.restore_length = best;
    t.read_from      = narrowed_resume;   // gap [best, narrowed_resume) is re-read by the chunk loop
    return t;
}

/// FW-2b (T22): the two-pool restore pick.  `checks` are the LRU-capped periodic
/// checkpoints (deep anchors); `state_checks` are fine-cadence state-only
/// checkpoints (checkpoint_save copies gdn/ple/tails/dead/block_pos only - no
/// K/V - so each is ~100-150ms, T19).  The deepest checkpoint <= P across BOTH
/// pools is restored; the chunk loop still reads [P, n) - fail-close semantics
/// unchanged, the gap [restore, P) just shrinks.  Equal lengths are dedup-free:
/// deepest is deepest, whichever pool it came from.
inline NarrowedResumeTarget narrowed_resume_target_state(
        int64_t narrowed_resume, const std::vector<int64_t>& chain_lengths,
        const std::vector<int64_t>& state_lengths) {
    NarrowedResumeTarget t = narrowed_resume_target(narrowed_resume, chain_lengths);
    for (int64_t L : state_lengths) {
        if (L <= narrowed_resume && L > t.restore_length) {
            t.restore_length = L;
            t.read_from      = narrowed_resume;
        }
    }
    return t;
}

/// FW-2b (T22): the fine-cadence capture ladder.  Returns the next state-checkpoint
/// capture position after `done`: done + every, clamped to n (a capture at exactly
/// n is the live conversation, not a checkpoint; never past it).  every <= 0 = off
/// (INT64_MAX: the gate never fires) - byte-identical to today when the flag is 0.
inline int64_t state_cadence_next(int64_t done, int64_t n, int64_t every) {
    if (every <= 0) return INT64_MAX;
    const int64_t next = done + every;
    return next >= n ? n : next;
}

/// Phase 3a T10: the runtime's `read_from` resolution at generate.cpp:9394.  When the splice
/// hook fired AND narrowed resume (splice_narrowed == true), the chunk loop's read start is
/// `t.read_from` (the splice-narrowed resume = the edit pos E), NOT `resume` (the restored
/// anchor C).  Otherwise the legacy rules apply: STRATA_CKPT_REREAD rebuilds the whole
/// prefill from 0, and the normal path starts at `resume` (the deepest usable mount).
///
/// Without this helper the runtime at line 9394 used `reread_to > 0 ? 0 : resume`, which
/// discarded `t.read_from` and re-read [anchor, n) - the rotated region [C, E) the splice
/// just set up AND the gap [E, n) the splice's gap-read was supposed to handle.  The
/// gap-read IS the splice's saved work; discarding it makes the splice a no-op on the
/// prefill path (GLM review MAJOR-1: the K rotation is then redundantly overwritten by
/// the fresh prefill, but the prefill re-reads C..n instead of E..n, which is the actual
/// regression).
///
/// The helper is pure and header-only so the T10 test pins it.  generate.cpp's call site
/// is `const int64_t read_from = wire_splice_read_from(splice_narrowed, t, resume, reread_to);`
/// (replacing the pre-3a `reread_to > 0 ? 0 : resume`).
inline int64_t wire_splice_read_from(bool splice_narrowed,
                                     const NarrowedResumeTarget& t,
                                     int64_t resume,
                                     int64_t reread_to) {
    if (reread_to > 0) return 0;                          // STRATA_CKPT_REREAD: rebuild from 0
    if (splice_narrowed) return t.read_from;              // gap-read: [edit pos E, n) only
    return resume;                                         // legacy: re-read from the restored mount
}

/// Phase 3b T12: the partial-prefix mount's splice outcome.  When the strict mount fails and
/// `ConversationCache::best_with_partial(flag>0, ...)` returns a Match with `partial = true`,
/// the engine has a checkpoint C of length L whose first P tokens (= LCP) match the new
/// prompt's first P tokens and diverge at P (a mid-file edit past P).  The runtime mounts C
/// (restores its running state at length L), and the chunk loop reads from P.  The splice
/// plan (LCP-based `narrow_pos = P`, suffix-based `old_suffix_len` / `new_suffix_len` for the
/// unchanged tail from the back) drives the K rotation in [P, L) - the same machinery the
/// strict-mount splice path uses, but the resume and the chunk-loop read start differ.
///
/// This helper centralises the outcome so the runtime integration test can pin the exact
/// field values without GPU.  Inputs:
///   `partial_ck_tokens`  the full checkpoint C's tokens (length L)
///   `cur_tokens`         the new prompt's tokens
///   `partial_ck_length`  L (== partial_ck_tokens.size())
///   `partial_resume`     P (the LCP; the chunk loop's read start)
/// Returns a `SuffixHookOutcome` with:
///   fired = true
///   plan_edit_pos = P (== narrow_pos of the LCP-based plan)
///   restore_anchor_length = L (the runtime restores C's running state, not P's)
///   old_suffix_len / new_suffix_len / deltas / uniform / delta = the K-rotation plan
///   chain = the input chain with every entry whose length > P dropped (the runtime's
///           post-splice erase keeps anchors at or below the LCP; the anchor C of length L
///           is also dropped by the chain filter but the runtime uses `restore_anchor_length`
///           to restore C directly - see generate.cpp's `else if (!from_live)` branch).
///
/// Fail-closed: any unsupported input (L > cur.size(), P not in (0, L), LCP inconsistent with
/// the token equality check) returns `fired = false` and the runtime degrades to today's full
/// re-read.  Never crashes, never fabricates a checkpoint.
inline SuffixHookOutcome suffix_reuse_partial_apply(
        const std::vector<int32_t>& partial_ck_tokens,
        const std::vector<int32_t>& cur_tokens,
        int64_t partial_ck_length,
        int64_t partial_resume,
        const std::vector<CheckpointLength>& chain) {
    SuffixHookOutcome out;
    out.chain = chain;
    // fail-closed guards: every shape that would make the splice wrong is refused.
    const int64_t L = partial_ck_length;
    const int64_t P = partial_resume;
    const size_t n_new = cur_tokens.size();
    if (L <= 0 || P <= 0 || P >= L) return out;               // not a valid partial match
    if ((int64_t) partial_ck_tokens.size() != L) return out;  // inconsistent: tokens vs length
    // T17 (2026-10-13): L > n_new is now a VALID partial shape - the parked live
    // (prompt+response) is typically longer than the next turn's prompt, and this
    // refusal was the third guard that refused the live anchor.  The LCP below is
    // prompt-bounded, the tail walk in compute_suffix_plan is min-bounded, and the
    // runtime restores the deepest checkpoint <= P (never state@L).  The strict
    // path keeps its own refusal (a longer checkpoint is never a strict prefix).
    // verify the LCP claim: partial_ck[0..P) == cur[0..P) and partial_ck[P] != cur[P] (or P == L).
    for (int64_t i = 0; i < P; ++i)
        if (partial_ck_tokens[(size_t) i] != cur_tokens[(size_t) i]) return out;
    if (P < (int64_t) partial_ck_tokens.size() &&
        P < (int64_t) n_new &&
        partial_ck_tokens[(size_t) P] == cur_tokens[(size_t) P]) {
        // LCP == L means strict (cur[0..L) == ck[0..L)); the strict path handles it.
        return out;
    }
    // The plan: same compute_suffix_plan the strict path uses, but `old` is C's full tokens
    // (length L) and `new` is the new prompt.  LCP-based narrow_pos = P (verified above);
    // suffix-based edit_pos = L - old_suffix_len; the K rotation operates on [edit_pos, L).
    SuffixReusePlan plan = compute_suffix_plan(partial_ck_tokens, cur_tokens);
    if (plan.narrow_pos != P) return out;                     // defensive: LCP claim failed
    out.fired = true;
    out.plan_edit_pos = plan.narrow_pos;
    out.k_rotation_start = plan.edit_pos;
    out.old_suffix_len = plan.old_suffix_len;
    out.new_suffix_len = plan.new_suffix_len;
    out.uniform       = plan.uniform;
    out.delta         = plan.delta;
    out.deltas        = plan.deltas;
    out.replay_from   = plan.replay_from;
    out.restore_anchor_length = L;                            // the partial-mount anchor
    // chain filter: drop entries with length > P (the LCP) - the runtime's post-splice erase
    // at generate.cpp:9283-9285 (predicate `length > resume`) keeps anchors at or below P.
    // The anchor C of length L MUST be kept (the runtime restores it via `restore_anchor_length`),
    // so we explicitly preserve the entry with `length == L` regardless of the length > P filter.
    size_t i = 0;
    while (i < out.chain.size()) {
        const int64_t len = out.chain[i].length;
        if (len > P && len != L) { out.chain.erase(out.chain.begin() + (ptrdiff_t) i); }
        else ++i;
    }
    return out;
}

/// T18 (2026-10-13): the partial mount's read start, replay-gated.  The live
/// measurement (chain dt-steady5, /tmp/dt-steady5.log) proved that reading
/// [P, n) on state@checkpoint leaves the gap [ckpt, P) out of the GDN/PLE
/// recurrence: attention sees the gap's rows through the (valid, LCP-shared)
/// arena K/V, but the running state never ingested them - a count query over
/// the gap comes back under-counted (C answered wrong, gap 826 tokens; the
/// same signature as the v3 state@L probe).  Until FW-2 replay (ring splice /
/// recurrence replay over the gap) exists, the partial mount fail-closes to
/// reading from the deepest surviving checkpoint at or below P - byte-equal to
/// the strict-path semantics - keeping the park/restore/chain bookkeeping
/// intact and exercised.  `replay_available` is the engine-side gate (env
/// STRATA_SFX_REPLAY, default off): with it on, the gap-read [P, n) returns.
/// Pure: the T18 tests pin both arms plus the no-checkpoint degenerate.
inline int64_t partial_gap_read_from(bool replay_available, int64_t P,
                                     const std::vector<int64_t>& chain_lengths) {
    if (replay_available) return P;                 // FW-2: the gap is replayed, read [P, n)
    int64_t best = 0;                               // fail-closed: read from the checkpoint
    for (const int64_t L : chain_lengths)
        if (L <= P && L > best) best = L;
    return best;
}

}  // namespace strata::program::suffix_reuse
