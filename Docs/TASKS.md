# TASKS

## NOW
- (idle) Stage 1 complete; PR open. No active work.

## NEXT
- Watch Niko1221/Strata#3 for review/CI; apply requested changes on `feature/v100-moe`, push to `origin` (noorazman/Strata).

## LATER
- Stage 2 candidates (only after Stage 1 merges / user direction): dense-model path, multi-GPU, additional quants, scheduler/memory rewrites, new API frameworks.
- If a 16 GB deploy is wanted long-term: consider a smaller explicit `--expert-cache` or int8 KV at 8192+ ctx to grow the VRAM headroom.

## FUTURE
- Re-verify when the upstream hit path is fixed (ROUND 328) — the token-match comparison in `Docs/v100-testing.md` is the regression check.
- Q2_0 fixtures for the `ple_parity` ctest (currently red by design).
