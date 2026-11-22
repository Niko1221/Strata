# v0.2.18 full intake + Inkling-2 + engine 0.6.4 — implementation log

## Overview
User decision 2026-11-22: absorb ALL of official upstream v0.2.18 plus the
YASAKANI/zapabob-yasakani Inkling-2 Mini support, keep all fork-only features,
follow the official API, bump the native engine to 0.6.4 and the semantic
product version, then build -> overwrite install -> publish a release.

## DeepResearch facts (primary sources, read 2026-11-22)
- Official v0.2.18 (tag 6e38f588, published 2026-11-21): 227 commits,
  373 files, +85k lines. Main feature: Tencent Hy4-Flash 220B MoE
  (`strata-hy4` 0.1.64 shape engine, SQ-64/96/128 packs) needing 64 GiB+ RAM;
  plus CPU VNNI 1.5x, CUDA Graphs/Tree-8, many build/crash fixes.
- `hiruhiru2005/sync-strata`: MIT; RAM32 uses Sync-S + RAM squeeze (already
  ported to this fork as `c144e65ef`). Not compatible with Q38 GGUF.
- `yasakani-dev/yasakani` and `zapabob/yasakani`: Apache-2.0 (NOT MIT).
  Inkling-2 Mini = 26.4B MoE, needs 24 GB+ GPU and a proprietary `.yskn`
  weight format + dedicated CUDA kernels; cannot run on this 32GB/16GB PC.
- User explicitly chose "put it ALL in" despite these constraints.

## Selected standards / skills
- `deepresearch-defense-standard`, `milspec-codex-standard`.
- SOPs: Common, Implementation Start Gate, Skill Selection, DeepResearch,
  Security, Python. Primary sources: upstream git + GitHub API + release notes.

## Plan
1. One-way intake of v0.2.18 into a fresh worktree (reuse
   `tools/oneway_upstream_merge.py`), base = v0.2.17 adopting tip, preserving
   fork-only Q38/DS4/Hakua/RAM32/9B features. Resolve conflicts; verify anchors.
2. Port Inkling-2 Mini support from zapabob/yasakani with Apache-2.0 NOTICE
   retained (as a distinct shape engine; do NOT weaken signature verification).
3. Bump native engine to 0.6.4 (unsigned; no signing key exists) and the fork
   semantic product version; sync docs/data/release.json upstream ref to 0.2.18.
4. Full build (C-drive target), NSIS + real overwrite install, tag/release/push.

## Residual risks
- Hy4-Flash and Inkling-2 cannot actually run on this 32GB/16GB PC; they are
  integrated as opt-in support only.
- The one-way tool has a history of creating empty rename stubs; after the
  merge, audit for zero-byte files with non-empty upstream blobs and repair.
- No signing key; engine 0.6.4 will be unsigned/local like the 9B preview.
