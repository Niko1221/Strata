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

## Residual risks
- No signing key; ed25519 component signature verification removed earlier by
  user decision; SHA-256 digest verification retained. `releaseQualified=false`.
- Inkling-2/Hy4 inference on real GGUF not yet measured; measurement in progress.
