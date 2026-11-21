# Hakua 0.7.1 — fix release over upstream 0.2.17

This release supersedes v0.7.0. It fixes a regression introduced during the
official upstream v0.2.17 intake that left 175 `terminal/tools/*.py` files as
zero-length stubs, which broke Control's `models list` and the GUI model view.
All fork-only features (Q38 dense engine, DS4/DeepSeek, RAM32 squeeze, Hakua UI,
web search, 8-language READMEs) and the official 0.2.17 API adaptation are kept.

## Fixed
- Restored all 175 empty `terminal/tools/*.py` files from the official upstream
  blobs. `strata models list` now returns the installed Q38 models correctly
  (3/3 present on the packaged Control).
- The official 0.2.17 API is followed: `/v1/models` reports `max_model_len` /
  `context_length` / `max_context_length` and supports `GET /v1/models/{id}`;
  33 new official `STRATA_*` settings are merged with 56 fork-only settings
  (1,181 total).
- Re-staged Control and the bundled tools so the installer carries the restored
  files.

## Verification
- `release_manifest.py --check`: ok (0.7.1, engine 0.6.3).
- `verify-release.mjs`: signed Q38 preview companion (152 files) verified;
  payload Hakua 0.7.1 / engine 0.6.3 / Control 0.6.3.
- Playwright real-device probe: the packaged host renders the UI with zero page
  errors; the `ui-flows` English-text failures were only a Japanese-vs-English
  UI language mismatch, not a GUI regression.
- Windows NSIS installer build and overwrite-install check: (appended after run).

## 9B preview (separate, unsigned)
The rebuilt Q38 engine that supports both the 32-layer 9B and 64-layer 27B
GGUFs is available as an **unsigned local preview component** at
`H:/Strata/preview-components/9b-unsigned-local-preview`. It is NOT shipped in
this installer. To try 9B, copy that folder into
`%LOCALAPPDATA%/Strata/components/engine/` (with
`STRATA_REQUIRE_SIGNED_COMPONENTS` not set to 1); the signed 27B-only 0.6.3
component can remain. There is no signing key, so this preview is local-only.

## Limits
- Windows Authenticode signing remains absent; component/feed signatures are
  verified with bundled public keys. `releaseQualified=false` is unchanged:
  full-window generation, general model quality and GPU concurrency are not
  qualified by this release.
