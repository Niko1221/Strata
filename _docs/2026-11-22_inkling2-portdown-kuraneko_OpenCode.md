# Re-integrate Inkling-2 from kuraneko1/Strata with credit — implementation log

## Overview
User decision 2026-11-22: after removing the YASAKANI-derived DLL approach,
re-integrate Inkling-2 support based on `kuraneko1/Strata` (MIT), crediting the
author kuraneko1 (_ryu15_) in code and docs.

## Source (primary, read 2026-11-22)
- `kuraneko1/Strata` fork of Niko1221/Strata, MIT License
  (Copyright (c) 2026 Niko1221 and the Strata contributors).
- Author: GitHub `kuraneko1`, display name `_ryu15_`.
- Branch `inkling2` adds `strata-inkling2`, a CUDA+AVX2 hybrid engine for
  Inkling-2-Mini (GGUF arch `inkling2`), tested on RTX 4070 12 GB (sm_89)
  65K-262K context and V100 32 GB (sm_70) up to 819K-1M.
- Files (MIT, ported verbatim with credit):
  engine/src/inkling2/{inkling2_engine.cu,inkling2_kernels.cuh,
  inkling2_dspark.cuh,inkling2_main.cpp},
  engine/include/strata/inkling2/{engine.hpp,profile_rules.hpp},
  engine/tests/program/inkling2_profile_rules_test.cpp,
  docs/INKLING2.md.
- Credit line (code header + docs): "Based on strata-inkling2 by kuraneko1
  (_ryu15_), https://github.com/kuraneko1/Strata (MIT License)."

## Plan
1. Port the above files into this fork's engine/ tree with credit headers.
2. Wire the CMake target `strata-inkling2` into engine/CMakeLists.txt.
3. Add `inkling2` arch to model_family.hpp (metadata/inspect) with credit.
4. Add attribution to docs/third-party/inkling2-kuraneko/ (MIT LICENSE).
5. Update all 8 READMEs with an Inkling-2 note crediting kuraneko1.
6. Rebuild, overwrite-install, publish release; consolidate worktrees; delete
   build artifacts.

## Commands run (verification)
- Configure: `cmake -S engine -B H:/Strata/builds/inkling2-configure-test -G Ninja
  -DCMAKE_MAKE_PROGRAM=<VS Insiders ninja> -DCMAKE_BUILD_TYPE=Release
  -DSTRATA_ENABLE_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=120 -DSTRATA_BUILD_TESTS=OFF`
  (via vcvars64.bat) — exit 0.
- Build: `cmake --build ... --target strata-inkling2 -j 5` — iterative fixes below;
  final exit 0, produced `strata-inkling2.exe` (36,869,120 bytes).
- `--help`: prints usage (`--model`, `--draft`, `--serve`, `--bench`, ...) — OK.
- `release_manifest.py --check`: `ok (0.7.3, engine 0.6.5)`.

## Build fixes applied (incompatibility causes)
1. `prefill.cpp(3414) set_card_expert_share` signature mismatch (upstream 0.2.18
   BW-H added a 5th `int pool_workers` arg to the header; fork's CUDA definition
   still had 4). Added `int g_sp_pool_workers` + 5th parameter + assignment.
2. Inkling-2 kernel depends on kernel API absent from fork: `NativeExpertLayout::
   situ/situ_beta/situ_linear_beta`, `NativeFmt::situ/...`, `Gemm::reserve_bf16_tc`,
   `native_expert_grouped_wmma` (declarations added manually).
3. `.cu/.cpp` implementations ported via `git apply -p2` from kuraneko1 diffs:
   native_expert.cpp (+SITU dispatch), iq_kernels.cu (+708 lines WMMA/SITU),
   gemm.cu (+reserve_bf16_tc), nvfp4_avx2.{hpp,cpp} (+nvfp4_256_gu_rows_situ).
4. New files copied: `bf16_avx2.hpp`, `bf16_avx2.cpp` (with CMake source + `/arch:AVX2`
   MSVC and `-mavx2;-mfma;-mf16c` GNU compile options in cpu.cmake).

## Version bump
- Product 0.7.2 -> 0.7.3; native engine family 0.6.4 -> 0.6.5; upstream stays v0.2.18.
- All 8 localized READMEs updated: heading version, engine 0.6.5, three opt-in
  components, and an Inkling-2 note crediting kuraneko1 (_ryu15_) / kuraneko1/Strata (MIT).

## Residual risks
- 12 GB+ GPU required for the hybrid engine; this PC is 16 GB (RTX 5060 Ti),
  so it may be testable here (unlike the YASAKANI 24 GB DLL). No Inkling-2 GGUF
  is present locally, so only `--help` was smoke-tested; real model inference untested.
- Kernel API port touched Q38-adjacent headers (iq/native_expert/gemm); added APIs are
  additive with defaults, but full Q38/DS4 regression build and tests are not yet run.
- Windows Authenticode signing absent; ed25519 component signature checks removed by
  user decision; SHA-256 digest verification retained. `releaseQualified=false`.

## Full build, install verification and publication (2026-11-22)
- Host UI rebuilt with pinned Node 24.22.3 + prepared public assets (STRATA_PUBLIC_PACKAGE=1,
  STRATA_PUBLIC_ASSETS, STRATA_PUBLIC_ASSETS_RECEIPT) — exit 0.
- `verify-release.mjs` (Node 24.22.3, digest-based, STRATA_Q38_PREVIEW_COMPONENT_DIR
  = 0.6.4 preview, 152 files): exit 0, "Release payload verified: Hakua 0.7.3".
- Full `npx tauri build --bundles nsis` with C:-drive target (v072-target), 5 jobs:
  BUILD_EXIT=0, NSIS `Strata_0.7.3_x64-setup.exe` 121.24 MiB / 127,127,684 bytes,
  SHA-256 `baf25ef97cc5c6336426faeeed0448e24c99af37297f7e442dae2a43eceba3e5`.
- tar (Copy-mode) + tar.7z: `Strata-0.7.3-windows-x64.tar.7z` 127,130,786 bytes,
  SHA-256 `9ca7651ecddb0f684372899239ef83afe3b6efdfda3b17a0fa541062acb62517`.
- NSIS upgrade fixtures (`tools/test_nsis_upgrade.ps1`): 6/6 ok=true, exit 0.
- Real installer overwrite E2E (isolated fixture, `/S /D=<fixture>\app`, uninstall
  registry backed up then restored): exit 0; strata-desktop.exe replaced
  (old 2519bd44... -> new 05618eb0...), user-data sentinel preserved byte-identical,
  Control bundle staged; registry restored to 0.7.1 / real install path.
- Tag `v0.7.3` annotated + pushed; main pushed at 2f1061fc3.
- GitHub release created with `--latest`, 4 assets uploaded/readback `state=uploaded`
  and SHA-256 matching local:
  - Strata-0.7.3-windows-x64-setup.exe  baf25ef9...e5
  - Strata-0.7.3-windows-x64.tar.7z     9ca7651e...17
  - strata-engine-0.6.4-...tar.gz       607a7bd1...9d (reused, verified)
  - strata-hy4-0.1.64-...tar.gz         fad3a195...91 (reused, verified)
  GitHub Latest = v0.7.3.
- Worktree consolidation: removed `integrate-upstream-0218` (clean) and
  `v065-app-only-correction` (13 uncommitted files backed up and byte-verified to
  H:/Strata/evidence/v065-worktree-uncommitted-backup-20261122 before --force removal).
  Only C:/Users/downl/Documents/ChatGPT/Strata on main remains.
- Build-artifact cleanup: user chose "large cleanup including old build caches";
  plan prepared at H:/Strata/evidence/cleanup-plan-20261122.ps1. Removed 98 dirs /
  ~148 GB (61 build caches, 33 old release dirs, 2 temp patch dirs, current-session
  inkling2 build + v073 E2E fixture + published-local copy). 0 failures.
- Cleanup-script bug (recorded honestly): the preserve comparison used forward-slash
  paths against `Get-ChildItem.FullName` (backslashes), so two intended-preserved
  dirs were deleted: `stable-060 inputs` (verify public assets + provenance) and
  `v072-component` (reused engine/hy4 source). Both were restored:
  - v072-component: re-downloaded engine 0.6.4 (SHA 607a7bd1...) and hy4 0.1.64
    (SHA fad3a195...) from the published v0.7.3 GitHub release; SHA matched.
  - stable-060 inputs: `public-assets` (20 files) reconstructed from app/host/dist
    using release-build.json recipe/publicAssets SHA (all 20 matched), and the
    provenance receipt regenerated as
    `{lucide_commit: 03f11e2d..., files: <recipe.publicAssets>}` at
    inputs/lucide-sources/provenance.json via H:/Strata/evidence/restore-verify-inputs.py.
  - `verify-release.mjs` re-run with restored inputs: exit 0 (verify-073-restored.log).
  Preserved intact: preview-components/0.6.4-9b27b-unsigned (verify companion),
  all research/, all _docs, evidence receipts/logs, model weights, the git repo,
  and the v073-regbak / v065-uncommitted-backup safety dirs.
