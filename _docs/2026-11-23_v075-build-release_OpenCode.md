# v0.7.5 — rebuild, overwrite-install and release the 2026-11-22/23 dev work

## Overview
User request 2026-11-23: using the implementation logs as reference, run the
full chain again — development history -> build -> overwrite install -> release.

## Development history to ship (from Antigravity logs, in working tree)
- `_docs/2026-11-22_rtx5060ti-cuda135-speedup_Antigravity.md`: CUDA 13.5 /
  sm_120a optimizations across Inkling-2 (INT8 TC, sm_120a MoE, branchless
  router), Q38 (pinned error, batched Hadamard/MMVQ, PQ2_0 warp kernel, 9B
  GEMM specialization), build fixes. Measured: DS4-9B prefill 2.84x faster,
  decode +13.4%, E2E +22.7% with exact token match.
- `_docs/2026-11-23_sakura-inkling2-98k-hermes_Antigravity.md`: Sakura
  Inkling-2 (inkling2_loop) 24->48 virtual layers, loop_res query rebuild
  (cosine 1.000000), WDDM 2048 MiB auto-reserve; 98K context with
  `--expert-vram 7.8` gave 22.44 t/s decode (12.7x vs 1.77 t/s) and 32/32
  token match; Hermes Agent provider + serve GGUF tokenizer/template/output
  parser. `configs/strata-sakura-inkling2-{65k,98k}.json` and
  `scripts/start-sakura-inkling2-hermes.ps1` added.
- 11 changed engine/serve files (+740/-196) + 3 new config/script files.

## Plan (repeat of the v0.7.4 chain)
1. Commit + push the development work to main.
2. Bump semantic version 0.7.4 -> 0.7.5 (engine family 0.6.6 -> 0.6.7);
   update VERSION, package.json/lock, Cargo.toml/lock, release.json, all 8
   READMEs; `release_manifest.py --check` ok.
3. Commit + push; create tag v0.7.5.
4. Full Tauri/NSIS build (C: target, Node 24, 5 jobs, preview component).
5. NSIS 6-scenario check; tar.7z bundle with SHA-256.
6. Publish v0.7.5 with 4 assets, --latest; verify public SHA readback.
7. Overwrite-install on the real machine from the published installer payload
   (7-Zip unpack direct copy — GUI/silent NSIS automation is unreliable here),
   set registry to 0.7.5, verify exe/reg/user-data/engines/GUI launch.
8. Implementation log + residual risks.

## Evidence to collect
Every step's exact command output: manifest check, BUILD_EXIT, installer SHA,
NSIS 6/6, tar.7z SHA, GitHub Latest + 4 asset digests, real exe SHA + reg +
settings SHA + GUI launch title.

## Provenance review (2026-11-23, user asked for opinion on logs + deep research)
- Finding: the perf numbers in the 8 READMEs and release body came from two
  different sources and were initially presented together without labels:
  - Antigravity dev logs (this session did NOT reproduce): DS4 2.84x, FLA 4.86x,
    Sakura 96K 22.44 t/s, etc. — measured by the Antigravity agent.
  - This session's own subagent measurement (reproduced, SHA-verified GGUF):
    9.09 t/s prefill / 11.78 t/s decode at 32K, evidence
    H:/Strata/evidence/inkling2-benchmark-result.txt.
- Fix: added an explicit provenance note to all 8 READMEs and v0.7.5 release
  body (commit 80868c840), per deepresearch-defense-standard "separate facts,
  inferences and others' claims; do not present unverified claims as your own".
- Quick search for the GGUF source (Hugging Face unsloth) was acceptable: the
  task was locating a model file, not a CVE/supply-chain trust decision; the
  LFS OID/SHA-256 was cross-checked against the downloaded bytes.

## Residual risks
- Windows Authenticode signing absent; SHA-256 digest verification retained;
  `releaseQualified=false`.
- Some optimizations measured only on RTX 5060 Ti (and by another agent, not
  reproduced here); no cross-GPU qualification.
- Long-context (65K-98K) E2E and ELT sweep still not run.
