# Hakua / Strata 0.6.0 Release Completion Log

## Overview
Completed the remaining verification, packaging, installation, GitHub publication, public readback, and documentation updates for Hakua / Strata `0.6.0` (App G `21013e19b1e3c43c42d6f13e84792501f26c87ff`, Component/Control F `b1cabf92628c64a4015a2ced606399c52698e0ed`, Native Q38 A `8b0b3c74514d26e5ca73496f0b268d66a0cb99a5`).

## Background / Requirements
- Complete Q38 real-model continuation trial (`installed-q38-r8`) under GPU admission baseline `r9`.
- Complete exact-G hosted CI acceptance (`acceptance-ci-r4`).
- Complete normal-profile startup and signed 0.6.0 CUDA engine import/selection (`finished-original-g-r3`).
- Clean up old engine versions `0.5.1`–`0.5.8` from `%LOCALAPPDATA%\Strata\components\engine`.
- Package all 17 G-scope release assets (`finished-artifacts-g-r1`), publish `v0.6.0` as GitHub Latest, update `refs/tags/latest`, and verify all 17 assets via anonymous streaming HTTP readback.
- Publish the real-model Japanese chat screenshot (`native-qwen38-japanese-chat.png`) in `README.md` and `docs/index.html` (GitHub Pages), and update `AGENTS.md`.

## Verification & Receipts
- **GPU Baseline r9 Preflight**: `H:/Strata/evidence/upstream-sync-20261116/gpu-desktop-baseline-r9.json` (`7976e07d6b2a5800457015e86d7c5692cf01c1b7ff7726b796c5f2d34bd5450e`), `freeMiB: 14544`, `unapprovedPIDs: []`.
- **Installed Q38 Real-Model Trial (`installed-q38-r8`)**:
  - `H:/Strata/releases/stable-060-20261116-codex/installed-q38-r8/result.json` (`04d48bceed5022b1bdcd20be5d20f3414dad8cdf537b748235cc388f95804119`).
  - `actualWebView=true`, `resumedComponentIdentity=true`, `registeredModelAvailable=true`, `actualNativeChild=true`, `sampledJapanese=true`, all Japanese chat turns passed, `nativeExited=true`, `modelStatUnchanged=true`.
  - Screenshot: `native-qwen38-japanese-chat.png` (`0d3dd3971b9cb7d9bde0f3c1aa302e8c12bb03080e958747d8e464b998881e84`), visually verified with no PII.
  - Cleanup note: `shellExited` timed out at 30s while Playwright CDP was still connected.
- **Exact-G CI (`acceptance-ci-r4`)**:
  - Rerun of failed `Windows / Python tools/` in `Checks` (`44035693629`) succeeded; all 9 workflows / 13 Checks jobs passed at `21013e19b1e3c43c42d6f13e84792501f26c87ff`.
  - `acceptance.json` SHA-256: `60021479bb624a687f835c05f41cb50717dc75255220bc634ddd0a312731b701`.
- **Normal-Profile Startup & Engine Import (`finished-original-g-r3`)**:
  - `normal-start-observation.json`: `223561d5d8f4cc9a5196ac36c6c8fc40d080d322349b612a21eeaae4935a1530`.
  - `original-engine-observation.json`: `abb8a08dd7d4b6cc89f42d55d9a2ed0864e2d994295f4f4a293ad31a2dde8dfb`.
- **Packaging (`finished-artifacts-g-r1`)**:
  - `finished-artifact-delivery.json`: `11f0c491c007cc9f2d0496a388c88fe39815405f117b3856ab8b357aefec17b4`.
- **Publication & Anonymous Readback**:
  - `v0.6.0` published as GitHub Latest (`publish-r1`: `04757259c58c75c553b637f660ba50bab6424ad2c2ca64d29f18cbae861a0f57`).
  - `refs/tags/latest` updated with lease to `eb94e50c4a0784125cd0ff3e54363cb8a3b9c8e1`.
  - All 17 public assets streamed anonymously from `v0.6.0` and matched frozen SHA-256 values (`finished-publication-g-readback-r1/readback.json`: `1ce2c6f16130eeefca54d1c0363a13d718cd14c8f9ac545c7d7f34595d563547`).

## Residual Risks
- `releaseQualified=false`: full-window inference, general model quality, and GPU concurrency remain unqualified.
- Windows Authenticode signing is absent; component and feed signatures remain verified against bundled production public keys.
