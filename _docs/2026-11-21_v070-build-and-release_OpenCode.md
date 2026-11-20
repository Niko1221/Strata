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
- build exit / installer SHA: (appended after run)
- overwrite install: (appended after run)
- tag / release / push: (appended after run)

## Residual risk
- Windows Authenticode signing remains absent; component signatures verified with
  bundled public keys. `releaseQualified=false` unchanged.
- Full Terminal/app suites and GPU inference were not run in this session.

## Commands
- node app/desktop/scripts/stage-rustxng.mjs --execute   (exit 0, SHA verified)
- (build/install/tag/push commands appended after run)
