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
