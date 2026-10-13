# FW-2c design notes — session-local partial K/V splice (drafted 2026-10-13, awaiting review)

## Goal
When a mid-prompt edit finds only a partial (LCP) match, mount the unchanged prefix's K/V directly from the PARKED conversation image instead of re-reading those tokens. Potential (FW-2 survey): 91.2% session-local on deep edits vs 59.7% parked / current state-cadence gap-read.

## Recon facts (verified in source today)
1. `ConversationKvReuse` (conversation_cache.hpp:57) — parking ALREADY retains real K/V:
   `kv` (per-layer ConversationKv) + `captured_tokens` / `unchanged_tokens` extents
   + per-stage variants. This is live infrastructure, not new.
2. Park path: `park_current_body` (generate.cpp:7889) captures stage snapshots via
   `conversation_snapshot_capture_bytes` with a byte budget; retained K/V must fit it.
3. `best_with_partial` (conversation_cache.hpp:315) returns Match{tokens, partial,
   checkpoint_length} — the LCP `P` and anchor depth; FW-2b's two-pool pick feeds
   `narrowed_resume_target_state`.
4. Fail-close invariant (T18): gap [anchor, P) must be READ through the normal chunk
   loop. FW-2c does NOT touch that — it replaces the *mount source* for [0, anchor)
   from "arena re-read" to "parked K/V splice" only when the parked image covers it.

## Design sketch (T24–T26, TDD)
- T24 pure seam (suffix_reuse.hpp): `kv_splice_plan(P, anchor, captured_tokens,
  unchanged_tokens, n)` -> {splice_from_parked: [0, min(anchor, unchanged)), read:
  [anchor, n), gap: [min(anchor, unchanged), anchor)}; fail-closed default OFF;
  env gate `STRATA_SFX_KV_SPLICE=1` for the first probes (same pattern as REPLAY).
- T25 runtime: in the partial-mount branch, if the parked entry's KvReuse covers
  [0, anchor): splice K/V into the arena (fp16 path first; int8 needs the Phase-3
  dequant->rotate->requant kernels), read [anchor, n) as today. Budget: splice bytes
  count against conversation-cache-mib.
- T26 gates: dt-steady chain oracle (160/161/162 must hold), kv-store cert cell,
  CUDA parity probe; kill criterion: splice cost > re-read cost at 4096-token depth
  (T19 measured piece restarts ~8s — splice must beat that).

## Risks / unknowns
- Int8 splicing kernel reuse from Phase 3 (exists, needs rewiring to park extents).
- Layer-split stages: KvReuse.stages must splice per-stage (park splits images).
- Host RAM: retained K/V already budgeted at park time — splice adds none.
