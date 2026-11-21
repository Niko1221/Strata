# Real overwrite install of 0.7.1 and 9B preview deployment — implementation log

## Overview
Applied the published v0.7.1 installer to the real Windows install at
`C:\Users\downl\AppData\Local\Programs\Strata` (was 0.6.9), then deployed the
unsigned 9B-capable engine as a local preview component. Per user decision,
9B load/chat E2E was NOT run — component deployment and Control recognition
are the stopping point.

## Real overwrite install (0.6.9 -> 0.7.1)
- Backed up `HKCU\...\Uninstall\Strata` (reg), old exe, and SHA-256 inventory
  of 440 user-data files to `H:/Strata/evidence/real-overwrite-backup-20261122/`.
- Pitfall: prior isolated-fixture tests had left `InstallLocation` /
  `UninstallString` / `DisplayIcon` pointing at
  `H:\Strata\evidence\v071-overwrite-fixture\app`. Plain `/S` installs kept
  writing to that stale path (registry was rewritten back to the fixture path
  by the installer on each run). Fix: set the registry values to the real path
  AND pass `/D=C:\Users\downl\AppData\Local\Programs\Strata` explicitly.
- Result: `INSTALLER_EXIT=0`, `DisplayVersion=0.7.1`,
  `InstallLocation=C:\Users\downl\AppData\Local\Programs\Strata`.
  Real exe replaced: `4260de50...` (0.6.9) -> `4812b396...` (0.7.1).
  Control BUNDLE 0.6.3 staged. User data: 440/440 files present and
  byte-identical (missing=0, changed=0). Real install was the target this time
  (unlike the earlier isolated-fixture verifications).

## 9B preview component deployment
- Copied `H:/Strata/preview-components/9b-unsigned-local-preview` (153 files)
  to `C:\Users\downl\AppData\Local\Strata\components\engine\
  0.6.3-windows-x86_64-cuda-cuda13-9bpreview` (robocopy /MIR).
  The signed 27B-only `0.6.3-windows-x86_64-cuda-cuda13` component is left
  untouched.
- The preview engine recognises the 9B GGUF
  (`profile=q38-9b layers=32 tensors=427`) and the 27B GGUF
  (`profile=q38-27b layers=64`).
- Control `scan()` lists both 0.6.3 components with `problems: []`, so the
  unsigned local preview is accepted without weakening signature verification
  (`signature: null`, `local: true`, STRATA_REQUIRE_SIGNED_COMPONENTS not set).

## Residual risks
- 9B load/chat E2E not run (user chose deployment-only). GPU showed ~11 GiB
  used by system processes at check time; the Q38 engine keeps most 9B weights
  on host (~5 GiB GPU weights) so it is expected to fit, but this is unverified.
- WebView2 0x80070578 (0.6.9-era OS-side GUI fault) is unchanged; the
  Playwright probe showed the UI renders with zero page errors.
- One model (qwen36 35B MoE) remains missing_or_invalid (no GGUF on disk).
