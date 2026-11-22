# Hakua 0.7.3 — Inkling-2 engine integrated (kuraneko1, MIT), engine family 0.6.5

Product version 0.7.3 (semantic versioning); native engine family 0.6.5;
upstream reference stays Niko1221/Strata v0.2.18. This release re-integrates
Inkling-2 Mini support as an in-tree CUDA engine, crediting its author.

## Inkling-2 Mini engine (source integrated, credited)
- Based on `strata-inkling2` from **kuraneko1/Strata** (MIT) by **kuraneko1
  (_ryu15_)**, https://github.com/kuraneko1/Strata. Credit is recorded in code
  headers, `docs/third-party/inkling2-kuraneko/CREDITS.md` and all 8 READMEs.
- New engine path `strata-inkling2` (GGUF arch `inkling2`): CUDA + AVX2 hybrid
  with SITU activation, FP16 WMMA grouped MoE and BF16 AVX2 CPU rows.
- Dependent kernel API ported additively (SITU members, `native_expert_grouped_wmma`,
  `Gemm::reserve_bf16_tc`, `bf16_avx2`), with defaults preserving Q38/DS4 behavior.
- CMake target `strata-inkling2` and `ModelFamily::Inkling2` added.
- Built successfully with CUDA 13.5 / sm_120: `strata-inkling2.exe` (36.9 MB);
  `--help` smoke test passed. **No Inkling-2 GGUF is present locally**, so real
  model inference, full-window decoding and quality remain unqualified.

## Fixed during the port
- `Prefill::set_card_expert_share` 5th argument (upstream 0.2.18 BW-H) added to
  the fork CUDA definition, which had kept the old 4-argument signature.

## Distribution note
- The publicly downloadable engine components remain the already-verified
  `strata-engine-0.6.4` (Q38 9B/27B) and `strata-hy4-0.1.64` packages. The
  version bump to engine family 0.6.5 denotes the source-level addition of the
  Inkling-2 engine; the Inkling-2 binary is shipped in-tree (built from source)
  and is not yet published as a standalone signed/opt-in component.
- ed25519 component signature verification was removed earlier by user decision
  (no signing key); SHA-256 digest verification remains the integrity gate.

## Verification
- `release_manifest.py --check`: ok (0.7.3, engine 0.6.5).
- `verify-release.mjs` (Node 24.22.3, digest-based, 152-file preview payload): passed.
- Inkling-2 engine built with CMake/Ninja + MSVC + CUDA 13.5 on sm_120: exit 0.
- Host UI rebuilt with pinned Node 24.22.3 and prepared public assets.

## Limits / residual risk
- Windows Authenticode signing absent; Inkling-2 engine untested on real GGUF.
  Q38/DS4 full regression build and GPU inference not run in this session.
- `releaseQualified=false` unchanged. Full-window inference, general model
  quality and GPU concurrency remain unqualified.
