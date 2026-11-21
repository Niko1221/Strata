# 9B preview engine and worktree consolidation — implementation log

## Overview
Investigated "9B won't run" and "0.6.3 is broken": the 9B checkpoints were
not loadable because the signed 0.6.3 engine (64 layers) rejects the 32-layer
9B GGUF. A rebuilt 9B-capable engine already exists in another build tree.
Per user decision, ship it as an UNSIGNED local preview component (no signing
key exists), keep the signed 27B component intact, consolidate all worktrees
into main, then build/publish v0.7.1.

## 9B findings
- GGUF: `H:/Strata/models/新しいフォルダー/DeepSeek-V4-Pro-Qwen3.5-9B-UNCENSORED-heretic.i1-Q4_K_M.gguf`
  (5.63 GB, arch qwen35, 427 tensors, 32 layers).
- Signed 0.6.3 `strata-q38.exe --inspect` rejects it: "requires 64 trunk layers".
- Rebuilt engines under `H:/Strata/builds/q38-9b-*` DO recognise it:
  `profile=q38-9b layers=32 tensors=427 hidden=4096 ffn=12288`. Selected
  `q38-9b-cvec-20261120` (latest, control-vector support). It also recognises
  the 27B GGUF (`profile=q38-27b layers=64`), so it is backward compatible.
- 9B source support exists in main (`admission.hpp` blob 007c186 matches the
  5a0df8e4 9B commit; README documents two dense qwen35 profiles).

## 9B preview component (unsigned, local only)
- Built `H:/Strata/preview-components/9b-unsigned-local-preview` from the signed
  0.6.3 component: replaced `strata-q38.exe` with the 9B build, wrote a manifest
  with `signature: null`, `local: true`, `variant: cuda13`, `backend: cuda`,
  `name: strata-engine` (all required by engine_component.py).
- Validated with staged Control: `read_manifest_any` + `verify(full=True)` =>
  `local=True signed=False VERIFY_OK files=152`, exit 0.
- NOT shipped in the installer. To use it, the user copies the folder into
  `%LOCALAPPDATA%/Strata/components/engine/` (STRATA_REQUIRE_SIGNED_COMPONENTS
  must not be 1); the signed 0.6.3 27B component is left untouched.

## Run configs (already applied)
- 2 Q38 run configs pointed at stale GGUF paths; rewritten to the real file.
  `strata models list` now shows 3/3 Q38 models `present` (was 1/3).
  Backups: `H:/Strata/evidence/runconfig-backup-20261122/`.

## Worktree consolidation
- 18 extra worktrees existed; 3 had "ahead" commits (swarm-prefill 3,
  native-q38-27b 4, q38-preview-pages 10). patch-id scan showed the engine
  commits are all present elsewhere; swarm-prefill's `fused_wmma_calls` fix is
  in main as `3bd6191b4`. No unique work needed merging.
- Dirty files (44) from 6 worktrees backed up to
  `H:/Strata/evidence/worktree-backup-20261122/` before removal.
- All 17 removable worktrees removed + pruned; only `main` remains.

## Next
- Full v0.7.1 installer build (C-drive target to avoid the H: NTFS
  permission-file access-denied seen on v0.7.0), NSIS overwrite check, then
  v0.7.1 tag/release/publication.
