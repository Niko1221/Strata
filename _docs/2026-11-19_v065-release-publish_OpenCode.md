# v0.6.5 release publish — implementation log

## Overview
Published the 180 s Control model-scan timeout fix as GitHub release `v0.6.5`,
with the NSIS installer that was verified in the normal installed profile.

## Background / requirements
- Published `v0.6.4` still had the normal-profile picker defect:
  `strata models list` takes 78-81 s on this host, but the host used a 60 s
  timeout, so `/api/native` returned `available:false, models:[]` and the
  Qwen3.8 model did not appear in the chat picker.
- The source fix (60 s -> 180 s in `apps/host/server/control-runtime.mjs`)
  was committed on `main` (`e5b0ea4af`) and tagged `v0.6.5`, but no Release
  asset existed.
- User chose performance **policy A** (strict byte-identical FP32 logits;
  accept native slower than Prism; stop new kernel rewrites). Release/UX
  integrity work is orthogonal and was completed.

## Assumptions / decisions
- The NSIS built at `q38-control-timeout-7b00b103-r2` (SHA-256
  `2976941f...c77`) is the artifact actually overwrite-installed and verified
  in the normal profile. Its shell version string is 0.6.4; the only delta
  from v0.6.4 is the timeout fix. Rather than rebuild a 0.6.5-labelled NSIS
  (blocked by stale staged Control/BUNDLE metadata that would break signed
  component provenance), this verified binary was attached to `v0.6.5` and
  the version-string caveat was stated plainly in the release notes.
- Repeated clean-worktree rebuild attempts (r1-r5) failed on
  `verify-release.mjs` staging checks (stale `pack_space.py`, stale
  `tools/qwen35/q36_phase_metrics.py`, then Control package version
  mismatch). No source or signed-component metadata was altered to force a
  build.

## Changed files / artifacts
- `VERSION`, `apps/host/package*.json`, `apps/desktop/package*.json`,
  `apps/desktop/src-tauri/Cargo.toml`, `Cargo.lock`: 0.6.4 -> 0.6.5.
- `apps/host/server/control-runtime.mjs`: 180 s `installedModels()` timeout.
- `apps/host/tests/component-updates.test.mjs`: test asserting 180_000.
- `_docs/2026-11-19_q38-control-timeout_OpenCode.md`,
  `_docs/2026-11-19_q38-picker-e2e-pass_OpenCode.md`,
  `_docs/2026-11-19_q38-064-normal-profile-pass_OpenCode.md`,
  `_docs/2026-11-19_q38-performance-assessment_OpenCode.md`.
- GitHub release `v0.6.5` + asset `Strata_0.6.4_x64-setup.exe`.

## Commands run
- `node --test tests/component-updates.test.mjs` -> 23/23 pass.
- `npm run typecheck` -> pass.
- Overwrite install of r2 NSIS -> installed host `control-runtime.mjs` hash
  `BECA8AEE...`, `timeoutMs: 180_000` present.
- `node q38-picker-diag.mjs` (normal profile) -> `available:true, models:4`
  in 64.9 s; picker showed `qwen3.8-27b-abliterated-3.69bpw-12GB-MTP`.
- `gh release create v0.6.5 ... --verify-tag <asset>`.
- Anonymous `Invoke-WebRequest` download -> HTTP 200, 132362730 bytes,
  SHA-256 `2976941F...C77` (matches attached digest).

## Verification results
- Release URL: https://github.com/zapabob/Strata/releases/tag/v0.6.5
- Attached asset digest: `sha256:2976941f545517ab8681584e6f4f6c24f7c338d58fee5283efb94ae409b6fc77`
- Anonymous download byte count and SHA-256 match.
- `v0.6.5` is now GitHub Latest.

## Residual risks
- The installed shell reports 0.6.4 while the release/tag is 0.6.5; disclosed
  in release notes. A future full Control rebuild could align the strings.
- Performance parity remains unmet (policy A): native prefill 312-388 ms vs
  Prism 208-211 ms; decode 27.10-27.33 vs 30.52-30.91 step/s. Full-window
  inference, general model quality and GPU concurrency unqualified;
  `releaseQualified=false`.

## Recommended next actions
- No further kernel work under policy A. If parity is ever required, revisit
  the INT8 MMQ gate/up option under a relaxed numerical-tolerance contract.
- Keep `releaseQualified=false` and the measured numbers in all README/docs.
