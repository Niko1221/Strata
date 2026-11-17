# Hakua / Strata 0.6.2 Implementation & Release Log

## Overview
Bumped Hakua / Strata to `0.6.2` (App `0.6.2`, Terminal `0.6.2`, Control `0.6.2`, Native Q38 `0.6.2` CUDA 13.5 SM120), packaged `Strata-0.6.2-windows-x64-setup.exe`, `strata-engine-0.6.2-windows-x86_64-cuda-cuda13.tar.gz`, `Strata-0.6.2-windows-x64.tar.7z`, and `SHA256SUMS.txt`, overwrite-installed locally, verified on real GPU hardware (`RTX 5060 Ti 16 GB`), updated all 8 language READMEs, and published `v0.6.2` (`--latest`) on `zapabob/Strata`.

## Version & Packaging Changes
- Updated version metadata to `0.6.2` across `VERSION`, `CONTROL_VERSION`, `CMakeLists.txt`, `metal/CMakeLists.txt`, `strata_cli/__init__.py`, `apps/desktop/package.json`, `apps/desktop/package-lock.json`, `apps/desktop/src-tauri/Cargo.toml`, `apps/desktop/src-tauri/Cargo.lock`, `apps/host/package.json`, `apps/host/package-lock.json`, `src/qwen38/engine.cu`, `docs/data/release.json`, and `AGENTS.md`.
- Rewrote all 8 language READMEs (`README.md`, `README.ja.md`, `README.de.md`, `README.es.md`, `README.fr.md`, `README.pl.md`, `README.pt-BR.md`, `README.zh-CN.md`) for `0.6.2` with Stage 1–6 and 65K/96K/131K real-device measurements.
- Built `strata-q38.exe` (`strata-q38 own-native 0.6.2 CUDA13 SM120 cudart=13050 cublas=131000`) and `Strata-0.6.2-windows-x64-setup.exe` (`129,686,405` bytes).
- Packaged `strata-engine-0.6.2-windows-x86_64-cuda-cuda13.tar.gz` (`1,512,250,140` bytes), `Strata-0.6.2-windows-x64.tar.7z` (`1,633,076,782` bytes), and `SHA256SUMS.txt` (`322` bytes) under `H:\Strata\releases\stable-062\assets\`.

## Verification & Real-Device Results
- `tools/check --only docs`: `6 passed, 0 failed` (`0m04s`).
- Local overwrite-install: `NSIS_ExitCode=0` (`C:\Users\downl\AppData\Local\Programs\Strata\strata-desktop.exe` ProductVersion `0.6.2`, `C:\Users\downl\AppData\Local\Strata\components\engine\0.6.2-windows-x86_64-cuda-cuda13\strata-q38.exe`).
- Stage 1–6 & Long-Context (`qwen3.8-27b-abliterated-3.69bpw-12GB-MTP.gguf`, `--kv spin8_k8v4`, RTX 5060 Ti 16 GB):
  - `140`-token prefill (`--ctx-size 4096`): `1,558.3 ms` (`89.8 tok/s` warm, `23.6x` faster than `36,760.7 ms` sequential baseline).
  - Steady-state decode (`GEN 32`, CUDA Graph + 8-layer STOP/NaN fences): `85.13 ms/step` (`11.75 step/s`).
  - `65K` (`65,536`): PASS (`1,024`-tok prefill `10,813.5–11,220.7 ms`, `91.3–94.7 tok/s`, decode `670.4 ms` / 7 steps).
  - `96K` (`98,304`): PASS (`required_with_reserve=15,093,582,572 B`, `140`-tok prefill `1,968.744 ms`).
  - `131K` (`131,072`): Guard-rejected (`required=15,968,094,956 B > available=15,906,897,920 B`, reserve `805,306,368 B`).
- **Limits**: `releaseQualified=false`; Windows Authenticode signing and component/feed signing remain absent.
