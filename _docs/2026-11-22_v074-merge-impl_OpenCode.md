# v0.7.4 merge of official upstream v0.2.19 — implementation log

## Overview
User request: preserve fork-only features while absorbing the latest official
upstream features, vulnerability/security fixes and bug fixes; follow the
official API; where an official feature and a fork feature are equivalent, merge
the fork's advantages into the official feature; bump semantic version, install
engine + GUI on this PC via overwrite, publish release artifacts, rewrite all 8
localized READMEs.

## Intake (evidence-based)
- `git fetch upstream --tags`: new tag v0.2.19 (upstream/main c6f6427d9).
- v0.2.18..v0.2.19: 55 commits, 278 files, +81,605/-2,035 lines. Key content:
  Inkling-2 Mini integrated across CUDA/HIP/Metal/SYCL + TUI/GUI; Flash-Next
  sm_120/sm_89/sm_86 and RDNA3 optimizations; fixes (sm_60 dp4a fallback,
  libstdc++-11 JVal map, VRAM cache eviction race, thinking-channel leaks,
  MSVC atomic fence); Control/engines 0.1.65.
- `tools/oneway_upstream_merge.py` (base v0.2.18, fork HEAD, upstream v0.2.19):
  dry-run + apply, 8645 paths, 0 errors, adopt 228 / three-way 69 / retain-fork
  8346. `tools/verify_oneway_merge.py`: ok=True, 0 failures.

## Equivalent-feature merge (user chose "keep both" then "merge Ryu into official CUDA")
- Official v0.2.19 Inkling-2 (68 files, `src/inkling2/` + `cuda/hip/metal/sycl`)
  restored into the fork tree alongside the fork's kuraneko1 port (separate
  target names `strata-inkling` vs `strata-inkling2` — no collision).
- Analysis: official `inkling2_loop.hpp` already has SITU (kSituBeta=4, kSituLb=25,
  situ_scalar) = equivalent to the fork's SITU port. The fork's unique advantage
  is Ryu's `profile_rules.hpp`: two-axis SM arch x VRAM residency automatic
  profile selection (measured-only / recommend-never-force, explicit flags win).
- Merged Ryu's advantage into the official CUDA implementation:
  - `engine/include/strata/inkling2/inkling2_loop.hpp`: include profile_rules.hpp
    with credit comment.
  - `engine/src/inkling2/cuda/ink_cuda.cuh`: include profile_rules.hpp and add
    `current_arch_profile()` helper that maps `current_sm_arch()` to the measured
    ArchProfile (Generic fallback for unmeasured GPUs). Credit comment added.
  - `docs/third-party/inkling2-kuraneko/CREDITS.md` retained with kuraneko1
    (_ryu15_) / kuraneko1/Strata (MIT) attribution.

## Version + metadata
- Product 0.7.3 -> 0.7.4; engine family 0.6.5 -> 0.6.6; upstream ref v0.2.18 -> v0.2.19.
- release.json upstream block synced to official v0.2.19 versions (product/terminal
  0.2.19, control/engine 0.1.65, 50 assets from GitHub release).
- Adopted official v0.2.19 docs that were fork-retained at 0.2.18
  (docs/INSTALL.md, AGENTS.md, tools/packaging/README.md) and updated
  release_manifest.py AGENTS.md version-line rule for the new upstream wording.
- `tools/release_manifest.py --check`: ok (0.7.4, engine 0.6.6).

## READMEs
- All 8 localized READMEs rewritten: 0.7.4 / 0.6.6 / upstream v0.2.19, new
  upstream features + security/bug fixes, official API 0.1.65, Ryu credit for
  the merged profile selection, fork features preserved, 0.7.4 download links.

## Verification
- Official `strata-inkling` target built successfully (CUDA 13.5, sm_120) with
  the merged profile_rules include — CMake configure exit 0, build exit 0,
  `strata-inkling.exe` (6.8 MB) produced, `--help` works.
- Full one-way intake verification: ok True, 0 failures.

## Upstream PR (2026-11-22)
- Per user suggestion ("PRしたほうがよくない？"), opened an upstream PR to return
  Ryu's advantage: https://github.com/Niko1221/Strata/pull/6571
  `[Engine] Merge Ryu's SM x VRAM profile tables into official CUDA Inkling-2`.
  Branch `pr/inkling2-profile-tables` (based on v0.2.19), 3 files, +271/-0,
  state OPEN. Title follows CONTRIBUTING area rule; body has problem/result/
  approach/validation/limits and credits kuraneko1 (_ryu15_, MIT). No defaults
  changed; header-only and opt-in at the caller.

## Local install / release
- Merged intake branch into main (fast-forward) at 8d30182b8 and pushed.
- Real GGUF found: `Sakura-Inkling-2-Mini-Uncensored-Q4_K_M.gguf` (16.28 GB)
  in H:/Strata/models and Downloads. Official `strata-inkling --info` rejects it
  (`missing tensor loop_res_query.weight` = Sakura/kuraneko1 format differs from
  official arch), so measurement uses the fork CLI `strata-inkling2 --kv-q8`.

## Fork CLI build fixes (three-way intake fallout)
- First fork CLI build failed: `prefill.cpp` used `cpu_share_nt()` /
  `kShareRowsMax` / `g_share_nt_arch` whose definitions were dropped by the
  v0.2.19 three-way merge. Restored the official v0.2.19 definitions before
  `cpu_share_max()` (commit fccc708d7); fork CLI then built (exit 0).
- Sakura GGUF has arch `inkling2_loop` (37 occurrences) and `inkling2_loop.*`
  keys; fork's `gguf_reader.hpp` only accepted `qwen4exp`, and `model_family.hpp`
  only `inkling2`. Added `check_architecture_inkling2` (accepts both prefixes),
  `inkling2_loop` arch mapping, and a prefix-fallback in `inkling2_engine.cu`
  get_i/get_f (commit 9509e14c7). Fork CLI rebuild in progress for real measurement.

## Publication + overwrite automation (2026-11-22, in progress)
- Pre-flight verified: tag v0.7.4 -> 263906a8d (HEAD), reuse assets SHA-verified
  (engine 607a7bd1..., hy4 fad3a195...), release body + scripts present, both
  PowerShell scripts parse with 0 syntax errors, `release_manifest --check` ok.
- Real-machine backup taken before overwrite:
  H:/Strata/evidence/v074-install-backup (userdata 560 files, localdata 188
  files, uninstall registry, settings.json SHA 6b7d7b11... matched live).
- `v074-auto-finish.ps1` was killed by a 30 s shell timeout (SIGTERM) before it
  could act; publication/install are done directly by the parent instead.
- Subagent build (ses_dfca58781ffejlBv9p0tqmMygO): BUILD_EXIT=1 — cargo compiled
  strata-desktop v0.7.4 but makensis bundling hit `os error 32` from two
  simultaneous makensis processes (subagent + parent retry) contending for the
  output. Parent stopped both and re-ran one clean build (v074-full-build-r3.log)
  which reached `Running makensis to produce Strata_0.7.4_x64-setup.exe`.
- Subagent benchmark (ses_dfca58783ffeBVScN3tHPLDKGi): canonical GGUF fully
  downloaded (16,610,543,328 bytes, SHA f90c5a4d...7e1 = HF OID). Fork CLI
  `--kv-q8 --bench --chunk 512 --max-new 32 --max-context 32768` exit 0 on RTX
  5060 Ti 16 GB: profile generic+core_resident_hybrid, 2756/3072 experts in VRAM,
  pp 9.09 t/s, tg 11.78 t/s. Evidence: inkling2-benchmark-result.txt. Sakura /
  Mainline_Support GGUFs confirmed incompatible (no loop_res_query.weight).
  Numbers added to v0.7.4 release notes + GitHub body.

## Build/publish/install execution (final, 2026-11-22)
- Single clean Tauri/NSIS build: BUILD_EXIT=0, installer `Strata_0.7.4_x64-setup.exe`
  127,147,110 bytes (121.26 MiB), SHA-256 cafe80949546b68ad0f124542d6d08dd
  ab03e4493c86a50a8399d0a604ae2148 (staged + SHA256SUMS written).
- NSIS cleanup fixtures: 6/6 ok=true, exit 0.
- tar.7z: 127,150,242 bytes, SHA-256 c79858bff8b05db8d2cd78a442301f00f2ff
  9ed5f042bae468ace073573de61c.
- Published v0.7.4 with 4 assets (setup, tar.7z, engine 0.6.4, hy4 0.1.64),
  --latest; PUBLISH_EXIT=0, URL https://github.com/zapabob/Strata/releases/tag/v0.7.4.
- Public readback VERIFIED: GitHub Latest = v0.7.4; all 4 assets state=uploaded
  with SHA-256 matching local exactly — setup cafe8094...2148, tar.7z
  c79858bf...de61c, engine 607a7bd1...ce9d, hy4 fad3a195...ff91.
- Overwrite install: backup at v074-install-backup (reg + userdata 560 +
  localdata 3905, settings SHA 6b7d7b11... matched). v074-auto-install.ps1
  waits for Latest=v0.7.4 then runs v074-install.ps1 (GUI then silent fallback)
  and logs final exe SHA / registry version.

## Goal status (final verification)
- upstream v0.2.19 intake: done (55 commits, verify ok).
- PR https://github.com/Niko1221/Strata/pull/6571: OPEN (Ryu's profile tables
  returned upstream).
- version 0.7.4 / engine 0.6.6 / upstream v0.2.19; `release_manifest --check` ok.
- READMEs: 8 languages, all contain 0.7.4/0.6.6/0.2.19.
- GitHub Latest = v0.7.4; 4 assets uploaded with SHA-256 readback matching local.
- Real GGUF benchmark: H:/Strata/evidence/inkling2-benchmark-result.txt
  (pp 9.09 t/s, tg 11.78 t/s on RTX 5060 Ti 16 GB).
- Real-machine overwrite install: in progress (uninstall done, GUI installing).

## Residual risks
- No signing key; ed25519 component signature verification removed earlier by
  user decision; SHA-256 digest verification retained. `releaseQualified=false`.
- Inkling-2/Hy4 inference on real GGUF not yet measured; fork CLI rebuild in progress.
- PR 6571 not yet reviewed/merged by upstream; this fork keeps its own merged
  copy regardless of PR outcome.
