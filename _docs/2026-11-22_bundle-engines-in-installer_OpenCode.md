# Bundle engines into installer and tar.7z — implementation log

## Overview
User decision 2026-11-22: after installing 0.7.3, both engines were absent
(expected: engines are separate opt-in components). User requested the
"official distribution that bundles python/node" — i.e. engine runtimes
bundled like python/node — and explicitly chose "1 and 2":
1. Bundle engines into the NSIS installer (so a desktop install includes them).
2. Bundle engines into the tar.7z release (setup + engine components together).

## Findings
- `app/desktop/src-tauri/tauri.conf.json` bundle.resources includes node, control,
  harness, notices, host/dist — but NOT engine-bundle / local-backends.
- `app/host/scripts/verify-release.mjs:36` explicitly throws on Windows if
  `resources/engine-bundle` or `resources/local-backends` are in bundle.resources:
  "The Windows app must distribute GPU runtimes as a separate engine component".
  This guard must be removed (or narrowed) to bundle engines.
- Staged `resources/engine-bundle/` holds an old 0.6.3 Q38 engine; its
  native-build.json sources no longer match the ported kernel sources
  (this is why verify previously used STRATA_Q38_PREVIEW_COMPONENT_DIR).
- Precedent: fork 0.5.x tar.7z bundled the NSIS installer together with the
  signed native CUDA component — that is the "official" combined bundle.
- Public engine assets available locally (SHA verified 2026-11-22):
  - strata-engine-0.6.4-windows-x86_64-cuda-cuda13.tar.gz
    SHA 607a7bd1ecbf747a568a1dcdfe8c6731d642fb3f0b4a386f80804dc9078ace9d
  - strata-hy4-0.1.64-windows-x86_64-cuda-cuda13.tar.gz
    SHA fad3a1952f6fe6304eeddd57db76cb74f87b34a04b78f5fb83e895898f93ff91

## Local install state (important)
- The user installed 0.7.3 via GUI, but my earlier rollback restored the real
  install to 0.7.1. Current real install is 0.7.1 (exe SHA 4812b396..., reg 0.7.1).
  After the new bundled installer is built and published, the user should install
  it fresh (0.7.3-with-engines) so the engines land together.

## Plan
1. Remove the Windows engine-bundle guard in verify-release.mjs (keep digest checks).
2. Re-stage resources/engine-bundle with Q38 0.6.4 (replacing old 0.6.3) and
   add Hy4 0.1.64 as a second engine in the same bundle.
3. Add resources/engine-bundle (+ local-backends) to tauri.conf.json bundle.resources.
4. Make verify-release accept the bundled engines (digest-based, no signature).
5. Rebuild full Tauri/NSIS installer; verify; NSIS 6-scenario; overwrite E2E.
6. Build tar.7z containing setup.exe + engine 0.6.4 + hy4 0.1.64.
7. Publish v0.7.4 (new version, since published v0.7.3 assets are immutable);
   update READMEs/release.json/manifest; set GitHub Latest.
8. Restore local real install to the new release via GUI (user) or verified path.

## Decision: official-conformant (2026-11-22)
User chose "official-conformant" over the ~2 GB bundled installer. Result:
- Do NOT remove the verify-release.mjs:36 Windows guard; do NOT bundle engines
  into the NSIS installer. Keep python/node/control/harness/notices bundled as
  before (that is already the official layout).
- Do NOT publish v0.7.4. The published v0.7.3 release and its 4 assets stay as-is.
- Engines remain separate opt-in components, which is the official upstream design:
  `strata-engine-0.6.4-...tar.gz` (Q38 9B/27B) and `strata-hy4-0.1.64-...tar.gz`.
- No code/build changes were made; this log records the evaluation and decision.

## Local install
- Real install was rolled back to 0.7.1 during the earlier overwrite attempt.
  User should install v0.7.3 from GitHub directly (double-click) and then add the
  two engine components from the release page if models for them are needed.

## Installed both engines on the real machine (2026-11-22)
User asked to put both engines into the running 0.7.3 install (screenshot showed
"エンジン: バージョンを利用できません"). Steps:
- Found engine components dir `C:/Users/downl/AppData/Local/Strata/components/engine`
  held 0.5.9-0.6.3 but not 0.6.4 / hy4 0.1.64. App scans this folder for
  `engine-component.json` (id/version naming `<id>-<version>-<platform>-<backend>-<variant>`).
- C: was full (0.10 GB free); first extraction produced 0-byte files. Removed
  incomplete dirs plus old build caches v070-target/v071-target (7.5 GB free).
- Extracted `strata-engine-0.6.4-windows-x86_64-cuda-cuda13` (153 files) and
  `strata-hy4-0.1.64-windows-x86_64-cuda-cuda13` (8 files, strata-hy4.exe 32 MB)
  from the SHA-verified v0.7.3 release assets into the components/engine dir.
- Verified with `engine_component.verify(full=True)` (digest, no signature):
  strata-engine 0.6.4 -> 152 files OK; strata-hy4 0.1.64 -> 7 files OK.
- Deterministic: the engine dirs are the official opt-in component layout, so
  the GUI "Engines" settings should show versions after an app restart.
- Note: v072-target C: build cache was preserved (used by the 0.7.3 build).

## Residual risks
- `releaseQualified=false` remains; Windows Authenticode signing absent;
  signature verification removed by user decision; SHA-256 digest checks retained.
- No Inkling-2 GGUF or Hy4 model locally; engine binaries installed but inference
  untested on this machine.
