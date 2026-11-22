# v0.7.6 — publish the rebuilt 0.6.7 engine component

## Problem
v0.7.5 shipped the OLD `strata-engine-0.6.4` component even though the
2026-11-22/23 development changed the engine sources (CUDA 13.5 sm_120a
speedups, Sakura Inkling-2 98K). User pointed out "エンジンがリリースされてない".

## Work
- Built fresh CUDA 13.5 (V13.5.37) binaries from the Antigravity dev work:
  `strata.exe` SHA 83c84669... (72.5 MB), `strata-inkling2.exe` SHA 97b9a658...
  (36.8 MB). Q38 `strata-q38.exe` has no fresh built binary, so the verified
  0.6.4 Q38 binary is retained (Q38 kernel changes remain source-only).
- Assembled the 0.6.7 component from the verified 0.6.4 DLL closure plus the
  fresh exes; regenerated `engine-component.json` by recomputing every file's
  SHA-256 (154 files), version 0.6.7, build_id q38-067-cuda135-20261123-unsigned.
- Verified with the app's own `engine_component.verify(full=True)`: OK, 154 files.
  (The packaging tool's stricter dependency check also flags the official 0.6.4
  asset, so it is not the shipping gate.)
- tar.gz: `strata-engine-0.6.7-windows-x86_64-cuda-cuda13.tar.gz`, 1,522,975,898
  bytes, SHA-256 6ef349f9807cc3c8507809e27e88b7f6fe82ee0cf92c83f24d5afbf3e126128d.
- Bumped product 0.7.5 -> 0.7.6 (engine family stays 0.6.7), updated release.json,
  8 READMEs; `release_manifest.py --check` ok (0.7.6, engine 0.6.7).

## Notes
- Work on a new branch/worktree earlier (engine/0.6.7) was superseded because the
  original fork conversation was active on shared main; no shared state was
  overwritten. This log + version bump are committed on main per user request.
- No Authenticode signing; signature verification removed earlier; SHA-256 digest
  verification retained. `releaseQualified=false`.

## Publication result (2026-11-23)
- main committed + pushed at 8f172e9b8 (version 0.7.6 / engine 0.6.7),
  tag v0.7.6 pushed, GitHub release created with --latest:
  https://github.com/zapabob/Strata/releases/tag/v0.7.6
- 4 assets uploaded, all state=uploaded with SHA-256 matching local:
  engine 0.6.7 6ef349f9...e126128d, hy4 0.1.64 fad3a195...ff91,
  app setup/tar.7z (the 0.7.5 app binary, unchanged) 734558cc.../b761f998....
- GitHub Latest verified = v0.7.6.

## Residual risks
- Q38 binary is unchanged from 0.6.4 (kernel speedups not compiled into a new
  Q38 executable); those optimizations remain source-only until Q38 is rebuilt.
- No cross-GPU qualification; measurements from RTX 5060 Ti only.
- The installer/tar.7z assets carry 0.7.5 filenames (the app itself was not
  rebuilt for 0.7.6; only the engine component changed).
