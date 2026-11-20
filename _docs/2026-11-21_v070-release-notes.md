# Hakua 0.7.0 — official upstream v0.2.17 intake

This release absorbs official upstream `Niko1221/Strata` v0.2.17 (app and
Terminal 0.2.17, engine/Control 0.1.63) into the Hakua fork while preserving all
fork-only features: the Q38 dense CUDA engine, DS4/DeepSeek work, RAM32 squeeze,
Hakua UI, web search, agent workflows, and the 8-language READMEs.

## Official API followed
- HTTP API: `GET /v1/models` now reports `max_model_len`, `context_length` and
  `max_context_length`, and `GET /v1/models/{id}` retrieves one model (404 on a
  miss), matching upstream v0.2.17.
- Settings register: 1,181 entries — all 33 new official `STRATA_*` settings
  (for example `STRATA_POOL_DEP`, `STRATA_SLOT_READ_WIDE`, `STRATA_P2P_HANDOFF`,
  `STRATA_ROUTE_TAIL_SKIP`) plus 56 fork-only settings (DS4/Q38/Hakua).
- Tool-call parser, error responses and thinking-budget fixes from the official
  0.2.17 release notes were adopted where the one-way merge allowed a clean
  three-way merge; the critical API files were manually resolved.
- Documentation and `docs/data/release.json` follow the official v0.2.17 release
  metadata and its 34 public assets (from the official GitHub API, read 2026-11-20).

## Fork features retained
- Q38 dense engine on the official engine API: `strata::qwen38::run` entry,
  `STRATA_Q38_STANDALONE=1` build option, signed 0.6.3 CUDA companion (152 files
  verified, CUDA 13 / SM120).
- RAM32 squeeze, DS4 speed switches, Hakua memory/Hermes stack, and all 8
  translated READMEs are present.

## Verification
- One-way intake: 16/16 fork anchors, 13/13 upstream anchors, 5,978/5,978 applied
  file digests matched, 0 conflict markers.
- `pytest terminal/serve/test_server.py -k 'model or models'`: 8 passed, 15 subtests.
- `release_manifest.py --check`: ok (0.7.0, engine 0.6.3).
- `verify-release.mjs`: signed Q38 companion verified, payload Hakua 0.7.0 /
  engine 0.6.3 / Control 0.6.3.
- Windows installer build and overwrite-install check: (appended after build).

## Limits
- Windows Authenticode signing remains absent; component/feed signatures are
  verified with bundled public keys. `releaseQualified=false` is unchanged:
  full-window generation, general model quality and GPU concurrency are not
  qualified by this release.
