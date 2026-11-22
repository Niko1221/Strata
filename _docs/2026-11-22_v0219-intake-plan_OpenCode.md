# Intake official upstream v0.2.19 while preserving fork features — plan/log

## Overview
User request 2026-11-22: preserve fork-only features while absorbing the latest
official upstream features, security/vulnerability fixes and bug fixes; follow
official API; where an official feature and a fork feature are equivalent, merge
the fork's advantages into the official feature; bump semantic version, install
engine+GUI on this PC via overwrite, publish release artifacts, rewrite all 8
localized READMEs.

## Primary source (fetched 2026-11-22)
- `git fetch upstream --tags`: new tag v0.2.19 (upstream/main c6f6427d9).
- v0.2.18..v0.2.19: 55 commits, 278 files, +81,605 / -2,035 lines.
- Release: Control and engines 0.1.65 (API/settings update from 0.1.64).
- Notable content:
  - Inkling-2 integrated into official upstream across CUDA/HIP/Metal/SYCL,
    TUI, Desktop GUI, GGUF, DSpark/Beam-Tree, no-thinking mode, EP dual-GPU.
    Our fork already has an Inkling-2 port (kuraneko1, MIT) from 0.7.3 — this is
    the "equivalent feature" case: prefer official upstream Inkling-2 and carry
    over our fork's credit/advantages where compatible.
  - Security/bug fixes: sm_60 dp4a fallback in vhead_bitplane, libstdc++-11 JVal
    map, VRAM cache eviction race, thinking-channel leaks in host, MSVC atomic
    fence, stale Hy4-Flash assertions, SYCL compile errors.
  - Flash-Next sm_120/sm_89/sm_86 optimizations, RDNA3 WMMA, Metal 4.

## Fork features to verify intact (16 anchors from 0.2.18 intake)
Q38 27B/9B dense, DS4/DeepSeek, RAM32 squeeze, Hakua UI/memory, workflow skills,
8-language READMEs,kuraneko1 Inkling-2 credit, Hy4 opt-in, etc.

## Plan (stepwise, evidence-based)
1. One-way intake via tools/oneway_upstream_merge.py: base v0.2.18, fork HEAD,
   upstream v0.2.19. Verify ok=true, 16/16 fork anchors, upstream anchors.
2. Resolve equivalent-feature overlap: official Inkling-2 vs fork Inkling-2 port —
   keep official code as primary, retain CREDITS.md + README credit to kuraneko1
   where the ported code remains; ensure model_family arch `inkling2` present.
3. Follow official API: bump control/engines 0.1.64 -> 0.1.65 references in docs,
   release.json, READMEs, code where pinned (preserve fork defaults only where
   measured faster). Verify release_manifest --check.
4. Semantic version bump: product 0.7.3 -> 0.7.4; engine family 0.6.5 -> 0.6.6;
   upstream reference v0.2.18 -> v0.2.19.
5. Rewrite all 8 READMEs (en/ja/de/es/fr/pl/pt-BR/zh-CN) for new versions +
   upstream features + security fixes + fork features preserved.
6. Commit + push main.
7. Full build (C: target, 5 jobs), verify-release (Node 24), NSIS 6-scenario,
   overwrite-install verification on this PC, publish new tag/release assets,
   set GitHub Latest, public SHA readback.
8. Consolidate worktrees (already only main), cleanup build artifacts as decided.
9. Implementation log + residual risks (releaseQualified=false, no signing).

## Residual risks
- Large 278-file intake may conflict with fork Q38/DS4/RAM32 kernel edits; rely
  on one-way intake 3-way resolution + anchor verification, fix concretely.
- Official Inkling-2 may supersede the kuraneko1 port; preserve credit and any
  fork-specific defaults only where compatible with official API.
- Windows Authenticode signing absent; SHA-256 digest verification retained;
  no real model for Inkling-2/Hy4 locally (32 GB / 16 GB PC limits).
