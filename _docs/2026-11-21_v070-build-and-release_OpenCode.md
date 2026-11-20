# v0.7.0 build, install verification and release — implementation log

## Overview
Build the Windows NSIS installer from the main checkout after the 0.2.17 intake,
verify an overwrite install, publish the v0.7.0 tag and GitHub release assets, and
push main to origin. (User decision 2026-11-20/21: full scope including publication.)

## Prerequisites completed
- main commits: 8a3e9866 (0.2.17 intake merge), 618318d8 (v0.7.0 bump),
  7a1c633a (upstream v0.2.17 metadata), 2c8b6bad (rustxng path fix), plus earlier merges.
- `tools/release_manifest.py --check` passes: `ok (0.7.0, engine 0.6.3)`.
- Staged resources copied from `H:/Strata/worktrees/upstream-oneway-20261119`:
  control 5,811 files, engine-bundle 155, local-backends 40, node 3, notices 102,
  harness 23,419. RustXNG re-staged from main (`a509798f...`, SHA verified).
- Signed Q38 companion: `H:/Strata/evidence/q38-063-short-prefill-package-a22a8a35/
  0.6.3-windows-x86_64-cuda-cuda13` (engine-component.json SHA matches the
  pinned `1408136b...` in verify-release.mjs).

## Build plan
- Env: `CARGO_BUILD_JOBS=5`, `CMAKE_BUILD_PARALLEL_LEVEL=5`; keep the 3 GiB RAM /
  4 GiB commit / disk floors.
- `STRATA_Q38_PREVIEW_COMPONENT_DIR` = signed Q38 companion folder above.
- `npm run tauri build` from `app/desktop` (runs host build + verify-release,
  then cargo release build, then NSIS). Reuse `H:/Strata/releases/v067-main-build/
  cargo-target` as CARGO_TARGET_DIR for incremental build.
- Output: `Strata_0.7.0_x64-setup.exe`; copy to `H:/Strata/releases/v070-main-build/`.

## Verification
- build attempt 1 failed: `tauri` not on PATH → `npm ci` in `app/host` and
  `app/desktop` installed the pinned toolchain (@tauri-apps/cli 2.12.3).
- build attempt 2 failed at verify-release: Node v26.10.0 vs required 24.22.3.
  Rebuilt the host UI with the pinned Node 24.22.3
  (`H:/Strata/evidence/upstream-sync-20261116/node-24.22.3-v1/node.exe`) and
  the approved public assets + provenance receipt
  (`H:/Strata/releases/stable-060-20261116-codex/inputs/{public-assets,lucide-sources/provenance.json}`).
  `app/host/dist/release-build.json` now reports nodeVersion 24.22.3,
  publicPackage true, publicAssetCommit 03f11e2d (verified).
- build attempt 3 failed at verify-release: "Stale staged membership" for
  `resources/control/strata_cli` — the Control bundle copied from the old
  upstream-oneway worktree does not match the current 0.2.17-intaken source
  membership. Copy-based staging is not viable. Re-staging Control from the
  current source with `stage-control.py` (in progress at time of writing).
- build attempt 3 follow-ups: re-staged Control from current source
  (`stage-control.py`, exit 0); restored exact upstream bytes of the control-vector
  LICENSE (`a0dc4225...`, the 0.2.17 three-way merge had CRLF-corrupted it) and the
  libstrix NOTICE (`775926b...`); refreshed the staged notices README hash inventory.
- `verify-release.mjs`: **exit 0** — "Verified the signed Q38 preview companion
  (152 files); Release payload verified: Hakua 0.7.0, engine 0.6.3, Control 0.6.3".
- Full `npx tauri build` attempts 1-5 all failed in tauri-build's custom build
  script at `PERMISSION_FILES_PATH ... __app__-permission-files` with
  "access denied, E(os error 5)" — before any Rust compilation of the app. Tried:
  shared H: target (v067), fresh H: target (v070), v066 target that built 0.6.9,
  creating the app-manifest parent dir, removing a stale permission file, and
  C:-drive-local target. Capabilities dir is empty in both the prior successful
  worktree and here; tauri.conf.json is byte-equivalent apart from CRLF.
- Root cause of "access denied os error 5": the `__app__-permission-files`
  write by tauri-utils `write_if_changed` failed on the H: NTFS-compressed
  target dir. Moving CARGO_TARGET_DIR to C:\Users\downl\AppData\Local\Temp\opencode\
  v070-target let the permission files be written (confirmed len=2, Archive) and
  the build progressed into compiling strata-desktop v0.7.0.
- Full build: `BUILD_EXIT=0` (C:-drive target, 5 jobs, 3 GiB RAM / 4 GiB commit /
  disk floors kept). `strata-desktop.exe` built in 7m20s; NSIS produced
  `Strata_0.7.0_x64-setup.exe` (121.15 MiB / 127,039,448 bytes).
  Installer SHA-256: `04ea502271edfa02ca7922f09fb70f42330b2fdf1fa40b8bcce343ad7d76d124`.
  Copied to `H:/Strata/releases/v070-main-build/`; tar + Copy-mode tar.7z built
  with the installer and SHA256SUMS.txt.
- overwrite install: `tools/test_nsis_upgrade.ps1` (fixed to the app/ layout)
  ran against the real production `installer.nsi` cleanup macros in 6 isolated
  fixtures (upgrade, fresh, empty, locked, junction, root-junction): **exit 0,
  all ok=true**; user-sentinel data preserved, old engine retained on upgrade,
  engine-bundle/notices/host/node/harness cleanup exercised. Evidence:
  `H:/Strata/evidence/v070-nsis-upgrade-fixture/` and
  `H:/Strata/releases/v070-main-build/nsis-upgrade-test.log`.
- tag / release / push:
  - `git tag -a v0.7.0` at main HEAD; `git push origin main` and
    `git push origin v0.7.0` succeeded (origin/main=017804d8e,
    refs/tags/v0.7.0=8e1f7b2a).
  - `gh release create v0.7.0 --repo zapabob/Strata --latest` with both assets;
    published 2026-11-20T16:19:29Z, isPrerelease=false, GitHub Latest=v0.7.0.
  - Public readback: installer 127,039,448 bytes SHA-256
    `04ea5022...d76d124` (matches local); tar.7z 127,042,210 bytes SHA-256
    `ecaaf22f...e4187e2` (matches local); both state=uploaded.
  - Release URL: https://github.com/zapabob/Strata/releases/tag/v0.7.0

## Residual risk
- Windows Authenticode signing remains absent; component signatures verified with
  bundled public keys. `releaseQualified=false` unchanged.
- Full Terminal/app suites and GPU inference were not run in this session.

## Commands
- node app/desktop/scripts/stage-rustxng.mjs --execute   (exit 0, SHA verified)
- (build/install/tag/push commands appended after run)
