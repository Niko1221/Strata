# Hakua 0.7.4 — upstream 0.2.19 intake + Ryu's Inkling-2 profile tables merged

Product version **0.7.4** (semantic versioning); native engine family **0.6.6**;
upstream reference **Niko1221/Strata v0.2.19** (Control/engines 0.1.65).
Fork-only features (Q38 27B/9B dense, DS4/DeepSeek, RAM32 squeeze, Hakua
UI/memory, workflow skills, 8-language READMEs) are preserved.

## Official upstream 0.2.19 integrated
- One-way intake: 55 commits / 278 files absorbed; 8,645 paths classified,
  0 errors; verify ok (all fork anchors + upstream anchors intact).
- Inkling-2 Mini now spans CUDA/HIP/Metal/SYCL, TUI and Desktop GUI in upstream.
- Flash-Next sm_120/sm_89/sm_86 and RDNA3 WMMA optimizations.
- Bug/security fixes: sm_60 dp4a fallback, libstdc++-11 JVal map, VRAM cache
  eviction race, thinking-channel leaks, MSVC atomic fence, stale Hy4 assertions.
- Official API/settings follow Control and engines **0.1.65**.

## Ryu's advantage merged into the official CUDA Inkling-2 (and upstream PR)
- kuraneko1's (MIT, kuraneko1/_ryu15_) two-axis SM arch x VRAM residency
  automatic profile selection (`profile_rules.hpp`) merged into the official
  CUDA Inkling-2 path: `inkling2_loop.hpp` includes the table, `ink_cuda.cuh`
  adds `current_arch_profile()` mapping the current SM arch to the measured
  profile, with a conservative Generic fallback for unmeasured GPUs.
- Header-only, opt-in at the caller, measured-only / recommend-never-force;
  explicit CLI flags and env overrides still win. No defaults changed.
- Returned to upstream as PR https://github.com/Niko1221/Strata/pull/6571
  (`[Engine] Merge Ryu's SM x VRAM profile tables into official CUDA Inkling-2`).
- Credit inlined in code comments; `docs/third-party/inkling2-kuraneko/CREDITS.md`
  retained in this fork.

## Engine compatibility fix
- Restored CPU-share definitions (`cpu_share_nt`, `kShareRowsMax`,
  `g_share_nt_arch`) dropped by the three-way intake so the fork `strata-inkling2`
  CLI builds again.
- `gguf_reader.hpp` / `model_family.hpp` now accept both `inkling2` and
  `inkling2_loop` GGUF architectures, and the engine falls back between the
  two key prefixes, so kuraneko1-style GGUFs load.

## Distribution
- Per maintainer decision the Windows app keeps the official layout: python/node/
  Control bundled in the installer; GPU engines as separate opt-in release
  components. The public engine assets remain `strata-engine-0.6.4` (Q38
  9B/27B) and `strata-hy4-0.1.64`; engine family label 0.6.6 denotes the
  source-level Inkling-2 addition.
- ed25519 component signature verification was removed earlier (no signing key);
  SHA-256 digest verification remains the integrity gate.

## Verification
- `release_manifest.py --check`: ok (0.7.4, engine 0.6.6).
- Official `strata-inkling` target built with the merged profile tables:
  CMake configure + build exit 0, `strata-inkling.exe` produced, `--help` OK.
- Fork `strata-inkling2` CLI built (CUDA 13.5 / sm_120).

## Real GGUF benchmark (measured 2026-11-22)
- Model: `unsloth/Inkling-2-Mini-GGUF` root `Inkling-2-Mini-UD-Q4_K_M.gguf`
  (16,610,543,328 bytes, SHA-256 f90c5a4d...7e1 = HF LFS OID). The canonical
  root file (arch `inkling2`, 734 tensors) has `loop_res_query.weight`,
  `loop_kv_res_query.weight`, `output_res_query.weight`, matching the CLI.
  The `Mainline_Support` (`inkling2_loop`) variant has none of these and cannot
  be used by this CLI. The Sakura re-quantized GGUF is likewise incompatible.
- Fork CLI `strata-inkling2 --kv-q8 --bench --chunk 512 --max-new 32
  --max-context 32768`, exit 0, on RTX 5060 Ti 16 GB (sm_120, 13.5 GiB free):
  `profile=generic+core_resident_hybrid (cc=12.0 sm=36)`; placed 2756/3072
  experts in VRAM (12.25 GiB, tier1 128/128/layer, free=0 MiB) in 132.82 s;
  `pp 2 tok in 219.9 ms (9.09 t/s) | tg 31 tok in 2630.7 ms (11.78 t/s)`.
- Caveat: generated token IDs repeat; output quality not assessed. No OOM or
  hang. `--info`'s implicit default config (ctx 65536 / chunk 1024) was much
  slower (pp 0.03 t/s) — the reported numbers are for the requested config.
  Full evidence: H:/Strata/evidence/inkling2-benchmark-result.txt.

## Limits
- Windows Authenticode signing absent; `releaseQualified=false`.
- These are short 2-token prefill / 31-token decode numbers on one consumer GPU;
  not a general model-quality, full-window or GPU-concurrency qualification.
