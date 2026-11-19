# v0.6.7: new-layout release pipeline port and NSIS build

Date: 2026-11-20 JST / OpenCode. Worktree:
`H:\Strata\worktrees\upstream-oneway-20261119` (branch
`codex/upstream-oneway-20261119` = origin/main).

## Overview

The user asked (1) to make the fork work well on 32 GB RAM like Sync-Strata,
and (2) to finish the v0.6.7 release on the new `terminal/` layout after the
upstream 0.2.16 merge relocated the tree. The RAM32 work was landed earlier
(`c144e65ef`/`7588dc407`); this log covers the build-pipeline port and the
v0.6.7 build.

## Background / requirements

`verify-release.mjs`, `stage-control.py` and `tools/packaging/package_terminal.py`
still pointed at the old layout (`apps/`, root `strata.py`, root `CMakeLists.txt`,
`packaging/locks`). v0.6.5/v0.6.6 recorded repeated failures of exactly these
staging checks on the new layout. The user chose "fix the new-layout pipeline
completely" and, for the native-source identity wall, "use the signed preview
branch". A mid-course correction from the user: do NOT treat the staged Harness
as expendable — it ships in the installer and must stay verified.

## Assumptions / decisions

- `package_terminal.py` moved from `packaging/` to `tools/packaging/`, so its
  `ROOT = parents[1]` silently became `tools/` instead of the repo root. Fixed
  to `parents[2]`; added `PKG = ROOT/tools/packaging` for `LOCKS` and the
  hakua-memory vendor wheel (both also moved).
- Application payload moved under `terminal/`, engine-side helpers under
  `engine/` (e.g. `engine/metal/tools/metal_repack.py`). Added `app_path()`
  (terminal/ → engine/ → legacy root) instead of rewriting every constant, so
  both layouts still build.
- `verify-release.mjs`: Terminal sources now read from `terminal/` (strata_cli,
  strata_control, serve, strata.py, setup.py, tools, control-vector data);
  engine component version now read from the staged `engine-component.json`
  itself, because `engine/CMakeLists.txt` carries the independent upstream
  version (0.1.62), not the component version (0.6.3).
- Native-source identity wall: the 0.6.3 `native-build.json` record (1103 files)
  is bound to the v0.6.6-era source state; no existing worktree matches and the
  ed25519 private key does not exist (user-confirmed), so a re-signed engine is
  impossible. Per user decision, the existing signed Q38 preview branch
  (`STRATA_Q38_PREVIEW_COMPONENT_DIR`, manifest SHA-256 `1408136b…`, key
  `9c835e4839a4c6f3`, 152 files) is now used for any product version, not only
  0.6.4.
- Staged inputs reused from the verified v0.6.6 tree (provenance preserved,
  bytes untouched): Control rebuilt fresh via the fixed `stage-control.py`
  (sha256 `20150598…`, `--check` exit 0, 22 models/6 families), Harness copied
  from the pinned revision (23,419 files, BUNDLE commit `571902028a75…` as
  pinned by verify-release), notices copied with `libstrix-NOTICE.md` and
  `README.md` refreshed to the new-layout source bytes (content identical,
  line-ending-only hash drift; NOTICES.json entries updated to match), and
  local-backends (40 files) + engine-bundle (155 files) copied.
- Harness and git: the root `.gitignore` harness rules were still on `apps/`
  paths; updated to `app/` (`c12a5fe11`) and the extra local `.gitignore` I had
  added was removed, restoring the old-layout shape. Clarified: git-ignoring a
  staged build artifact does not exclude it from the installer —
  `tauri.conf.json` bundles `resources/harness` and verify-release pins its
  BUNDLE commit, imports and RustXNG payload.

## Changed files (pushed)

- `5d9ac0a8d` [Build] Port release pipeline to new terminal/ layout —
  `app/desktop/scripts/verify-release.mjs`, `app/desktop/scripts/stage-control.py`,
  `tools/packaging/package_terminal.py`.
- `c12a5fe11` [Build] Point harness ignore rules at the new app/ layout — `.gitignore`.

## Commands run

- `py -3 app/desktop/scripts/stage-control.py` (r1–r6; r6 exit 0) and `--check` (exit 0).
- `node app/desktop/scripts/verify-release.mjs` →
  `Release payload verified: Hakua 0.6.7, engine 0.6.3, Control 0.6.3` with
  `STRATA_Q38_PREVIEW_COMPONENT_DIR` set and `Verified the signed Q38 preview
  companion (152 files)`.
- Host rebuilt with the pinned Node 24.22.3 runtime and the pinned public
  assets: `STRATA_PUBLIC_PACKAGE=1 STRATA_PUBLIC_ASSETS=…\inputs\public-assets
  STRATA_PUBLIC_ASSETS_RECEIPT=…\inputs\lucide-sources\provenance.json
  node scripts/build.mjs` (recipe: publicPackage true, nodeVersion 24.22.3,
  publicAssetCommit `03f11e2d…`).
- NSIS build started: `node node_modules/@tauri-apps/cli/tauri.js build` in
  `app/desktop` with Node 24.22.3 first on PATH, the same public-asset and
  preview env, `CARGO_TARGET_DIR=H:\Strata\releases\v067-main-build\cargo-target`,
  `CARGO_BUILD_JOBS=5` (PC five-job baseline), log
  `H:\Strata\releases\v067-main-build\build.log`.

## Verification results

- `stage-control.py --check`: exit 0 (`installer ok: 22 models, 6 families`).
- `verify-release.mjs`: exit 0 with the signed preview env (full payload:
  host sources, Control trees vs terminal/, notices membership and hashes,
  Harness bundle + RustXNG, local backends PrismML/TurboQuant, signed Q38
  preview signature and `--version` probe `SM120 cudart=13050 cublas=131000`).
- Pre-existing failures on clean origin/main (2 in test_no_debugger_children)
  were shown earlier to be unrelated to this work.

## Residual risks

- The engine component shipped remains the signed 0.6.3 preview companion
  (152 files); the fork's KV batch-store and other engine changes live in
  source and in the separately built `strata.exe`, not in this signed payload
  (no private key exists — accepted constraint).
- The NSIS build result is pending at the time of writing; the installer
  version string, overwrite install and real-device E2E follow in the next log.
- `verify-release.mjs` now trusts the signed preview for any product version;
  the manifest hash and key id are pinned in the script, so the trust anchor
  stays explicit.

## Recommended next actions

1. Collect the NSIS build result; attach `Strata_0.6.7_x64-setup.exe` to the
   `v0.6.7` release with SHA-256.
2. Overwrite-install on this PC and run the resident `--serve` E2E with the
   real Qwen3.8-27B GGUF (exit 0, READY/GEN/QUIT).
3. Record the matched reversed-order warm A/B numbers for decode/prefill
   against Prism in the release notes (decode parity measured 30.0 vs
   30.5–30.9 step/s; prefill KV cost −27% from the batched store).
