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

## v0.7.1 build and verification (completed)
- Full build: `BUILD_EXIT=0` (C-drive `v071-target`, 5 jobs, 10m22s).
  `verify-release.mjs` passed: signed Q38 companion 152 files, payload
  Hakua 0.7.1 / engine 0.6.3 / Control 0.6.3.
  Installer `Strata_0.7.1_x64-setup.exe` 127,138,637 bytes,
  SHA-256 `b724b70b07af2db4f87d024e7b82a3b0a8ae1716b6fdfaa5f9ba710676852ed2`.
  tar + Copy-mode tar.7z built; tar.7z SHA-256
  `037a1d26654c176772071f08e3957fcaf5f52b9fb9c9cd4621552dc1aed4d528`.
- NSIS upgrade fixtures (6 scenarios): exit 0, all ok=true.
- Real installer overwrite (isolated fixture, registry backed up and restored):
  exit 0; exe replaced (old `4260de50...` -> new `4812b396...`), user-data
  sentinel preserved, uninstall registry restored to 0.6.9 / real path.
  Real install was never touched. Evidence: `H:/Strata/evidence/v071-*-fixture/`.

## v0.7.1 publication (completed)
- `git tag -a v0.7.1` at main HEAD; pushed main and tag (origin/main=1152af728,
  refs/tags/v0.7.1=924e2a85).
- `gh release create v0.7.1 --repo zapabob/Strata --latest` with both assets;
  published 2026-11-21T16:22:56Z, isPrerelease=false, GitHub Latest=v0.7.1.
- Public readback: installer 127,138,637 bytes SHA-256 `b724b70b...ed2` and
  tar.7z 127,141,538 bytes SHA-256 `037a1d26...528` both match local exactly;
  state=uploaded. Release URL: https://github.com/zapabob/Strata/releases/tag/v0.7.1
