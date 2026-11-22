# Hakua 0.7.2 — upstream 0.2.18 full intake, engine 0.6.4 family

This release absorbs all of official upstream v0.2.18 while keeping every
fork-only feature. The native engine is now the 0.6.4 family with three
separate, opt-in components. Product version follows semantic versioning 0.7.2.

## Official upstream 0.2.18 integrated
- Hy4-Flash 220B MoE support (`strata-hy4` 0.1.64 shape engine, Tree-8 CUDA
  Graphs, VNNI 1.5x, full vcache) — source integrated as opt-in.
- Official API, settings and docs follow v0.2.18 (control/engines 0.1.64).
- 16/16 fork anchors verified intact: Q38 27B/9B, DS4/DeepSeek, RAM32 squeeze,
  Hakua UI/memory, 8-language READMEs.

## Engine 0.6.4 family (separate components, signed verification removed)
Per user decision 2026-11-22, ed25519 component signature verification was
removed from verify-release (no signing key exists); SHA-256 digest checks
remain as the integrity gate. The 0.6.4 family ships as three opt-in components:
- `strata-engine-0.6.4-windows-x86_64-cuda-cuda13.tar.gz` (1.49 GB): rebuilt Q38
  dense engine supporting both 9B (32-layer) and 27B (64-layer) GGUFs, control
  vectors and cvec. SHA-256 `607a7bd1...ace9d`.
- `strata-hy4-0.1.64-windows-x86_64-cuda-cuda13.tar.gz` (426 MB): official signed
  Hy4-Flash engine, SHA-256 `fad3a195...93ff91` (matches upstream v0.2.18).
- Inkling-2 Mini: `yskn_inkling2.dll` + `yskn.dll` bundled inside the installer
  under `resources/inkling2/` with Apache-2.0 LICENSE/NOTICE/THIRD_PARTY_NOTICES.
  SHA-256 of yskn_inkling2.dll `b10248cd...1c078` matches YASAKANI BUILDING.md.
  Recognised as GGUF family `inkling2_loop` for metadata/inspect; generation
  uses the separate DLL. Requires a 24 GB+ NVIDIA GPU.

## Research incorporated (documentation, no third-party code copied)
- TurboQuant-CUDA (Apache-2.0): Qwen3.5-9B KV-cache TurboQuant methodology.
- Elastic Looped Transformer (README Apache-2.0): looped-weight ideas related
  to Inkling-2's loop core.
- See `research/inkling2-turboquant-elt-findings.md`.

## Verification
- One-way 0.2.18 intake: 8697 paths, 0 errors; 16/16 fork + 13/13 upstream anchors.
- `release_manifest.py --check`: ok (0.7.2, engine 0.6.4).
- `verify-release.mjs`: passed (digest-based; 152-file preview payload).
- Full build: exit 0 (20m18s); NSIS installer 141.37 MiB.
- NSIS upgrade fixtures: 6/6 ok; real installer overwrite E2E passed with
  user-data preserved and uninstall registry restored.

## Limits
- Windows Authenticode signing absent; component signature verification removed.
  `releaseQualified=false`: Hy4-Flash needs 64 GiB+ RAM, Inkling-2 needs 24 GB+
  GPU — neither can run on a 32 GB / 16 GB PC. Full-window inference, general
  model quality and GPU concurrency remain unqualified.
