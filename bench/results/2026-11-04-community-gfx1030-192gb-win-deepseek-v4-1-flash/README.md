# Community benchmark on Radeon PRO W6800

Measured on 2026-11-04 by @yukieiji. DeepSeek-V4.1-Flash 2.00 bpw on a gfx1030
card with an engine built from source - the release feed carries no gfx1030
DeepSeek engine, so these are, as far as I know, the first DeepSeek numbers for
gfx1030. Main limitation: the machine keeps the whole expert set in its 192 GB
of RAM; a 64 GB machine would not reach these decode numbers.

## Hardware and software

- GPU and VRAM: AMD Radeon PRO W6800, 32 GB GDDR6 (gfx1030). PCIe link: not
  recorded. An integrated Radeon (gfx1036) is present and takes HIP ordinal 0
  on Windows (#325); the W6800 serves as ordinal 1. The suite's own auto-filled
  hardware line names the integrated GPU - that line is wrong for this machine;
  the README overrides it.
- CPU: Ryzen 9 9900X (12 cores / 24 threads); installed RAM: 192 GB; storage:
  SanDisk Ultra 3D 4 TB, SATA - the model and the engine read from this one
  drive.
- OS: Windows 11 (10.0.26300); GPU driver: 32.0.21043.5001; ROCm: TheRock 10.2
  wheels from `tools/hip/build_windows.bat`, exact wheel versions in
  `rocm-toolchain-versions.txt`.
- Strata commit `ffa80d73133ea5c548aefe57889a16e8085a938f` (project 0.1.56);
  engine `strata-dsv41`, **source build for gfx1030, code unmodified**:
  `tools/dsv41/win/build_hip_win.bat` with `STRATA_HIP_ARCHS=gfx1030`, rest of
  the defaults; CMake `-DSTRATA_ENABLE_HIP=ON -DSTRATA_ENABLE_CUDA=OFF
  -DSTRATA_BUILD_TESTS=OFF -DSTRATA_BUILD_DSV41=ON -DSTRATA_NATIVE_EXPERTS=ON
  -DSTRATA_PORTABLE=ON -DCMAKE_HIP_ARCHITECTURES=gfx1030`; clang from the ROCm
  toolchain, MSVC headers/linker via vcvars64; build finished without errors.
  Kernel parity: `ds4_iq_test` on the W6800: passed.
- Windows source-build staging (worth a docs note): the built exe needs the
  ROCm runtime DLLs **and** the rocBLAS kernel library
  (`rocblas/library/Kernels.so-000-gfx1030.hsaco` and the `TensileLibrary_*`
  files) beside it, copied from the same wheel that built it.
- Background workloads: not recorded; no power limit set.

## Model and configuration

- Model: DeepSeek-V4.1-Flash 1017; quantization: ISTA-DASLab 2.00 bpw (GSQ-RCO),
  downloaded 2026-11-03 (257 GB), revision not recorded as a commit.
- Vision encoder: built into the engine (`--vision on`), no separate reader.
  Custom packs: the setup pack with the shipped 2.00 hot-expert profile
  (`deepseek-v4.1-flash-ist-das-2.00bpw`).
- Context 131,072; KV and expert tiers are managed by the engine: 110-119 GB of
  experts copied into RAM at load (`--mmap-experts auto`), 1,990-2,535 experts
  of 15,360 kept in VRAM, the rest of the set read from the model files;
  prompt chunks 32,768; conversation cache is the engine's own `ds4-slots`
  (always on for this engine).
- Drafting: DSpark, changed per run (`--dspark auto|off|on`),
  `--dspark-conf 0.65`; `--experts 5` in all three runs. Thinking off for the
  suite; sampling temperature 0 (the suite fixes both). No calibration, no
  speed projection.

```text
<repo>\build-dsv41-hip\strata-dsv41.exe --serve --pack <data>\packs\deepseek-v4.1-flash-ist-das-2.00bpw
  --max-context 131072 --eos 1 --vision on
  --dspark auto|off|on --dspark-conf 0.65 --reserve-mib 900 --experts 5
```

## Method

`python tools/strata_bench.py --long` (community quick suite), three complete
runs on one day about 30 minutes apart; each run started its own fresh engine
from the config and stopped it afterwards, loading and the ~200 s model load
are not timed. One warm-up chat and a 1K prompt are discarded; every fresh
prompt carries a unique tag, so the conversation cache cannot help. 3 runs per
test (9 chat requests), medians with low-high ranges; output caps: 100 tokens
chat, 24 tokens prompt tests, 64 tokens per agent turn. Speeds are the engine's
own timings; TTFT measured by the client over HTTP. Expert-cache state as in
the configuration section, unchanged by the runs. To reproduce: the command
above once per `--dspark` value.

## Results

| Configuration | Actual prompt tokens | Reused tokens | Generated tokens | Runs | Prompt tok/s median and range | Decode tok/s median and range | TTFT seconds median and range |
| --- | ---: | ---: | ---: | ---: | --- | --- | --- |
| auto, short chat | 32 (29-38) | 0 | 100 (100-100) | 9 | not measured | 18.1 (16.2-18.8) | 0.85 (0.73-1.31) |
| auto, 4K prompt, fresh | 4125 (4125-4125) | 0 (0-0) | 24 cap | 3 | 107.8 (103.4-107.9) | not measured | 38.28 (38.24-39.92) |
| auto, 16K prompt, fresh | 16562 (16562-16562) | 0 (0-0) | 24 cap | 3 | 192.7 (174.9-193.0) | not measured | 85.98 (85.85-94.70) |
| auto, 32K prompt, fresh | 33126 (33126-33126) | 0 (0-0) | 24 cap | 3 | 213.1 (211.5-213.9) | not measured | 155.50 (154.91-156.67) |
| auto, agent follow-up turns | 12875 (12834-12885) at the last turn | 69.5 % over the turns | up to 64 | 3 sessions | see the per-run tables | 16.6 (9.8-18.0) | 30.45 (20.27-36.95) |
| off, short chat | 32 (29-38) | 0 | 100 (100-100) | 9 | not measured | 17.1 (14.9-21.8) | 0.82 (0.64-0.99) |
| off, 4K prompt, fresh | 4125 (4125-4125) | 0 (0-0) | 24 cap | 3 | 114.2 (107.1-115.9) | not measured | 36.13 (35.59-38.54) |
| off, 16K prompt, fresh | 16562 (16562-16562) | 0 (0-0) | 24 cap | 3 | 189.7 (175.7-192.9) | not measured | 87.32 (85.88-94.31) |
| off, 32K prompt, fresh | 33126 (33126-33126) | 0 (0-0) | 24 cap | 3 | 214.5 (211.7-215.8) | not measured | 154.45 (153.52-156.48) |
| off, agent follow-up turns | 12857 (12843-12877) at the last turn | 69.5 % over the turns | up to 64 | 3 sessions | see the per-run tables | 19.4 (11.4-21.5) | 31.59 (18.88-36.86) |
| on, short chat | 32 (29-38) | 0 | 100 (100-100) | 9 | not measured | 20.2 (13.6-30.8) | 0.75 (0.60-1.01) |
| on, 4K prompt, fresh | 4125 (4125-4125) | 0 (0-0) | 24 cap | 3 | 112.3 (103.9-113.0) | not measured | 36.75 (36.52-39.72) |
| on, 16K prompt, fresh | 16562 (16562-16562) | 0 (0-0) | 24 cap | 3 | 186.0 (170.1-186.5) | not measured | 89.08 (88.81-97.42) |
| on, 32K prompt, fresh | 33126 (33126-33126) | 0 (0-0) | 24 cap | 3 | 207.4 (204.3-208.3) | not measured | 159.79 (159.03-162.17) |
| on, agent follow-up turns | 12867 (12859-12871) at the last turn | 69.5 % over the turns | up to 64 | 3 sessions | see the per-run tables | 19.6 (11.4-26.3) | 32.49 (19.14-38.01) |

Per-run data and engine log excerpts: `results.json` and
`engine-log-excerpt.txt` under `dspark-auto/`, `dspark-off/`, `dspark-on/`;
the per-run READMEs have the agent turn tables. Total suite time per mode:
about 1,500 s. Memory observations: 110.3 GB (auto, off) or 118.3 GB (on) of
experts in RAM, plus 1,990-2,535 experts in VRAM.

Failures and skipped: none in the three recorded runs.

`--dspark auto` chose off ("the routed experts (132 GB) are far more than the
GPU holds (30 GB), where its verify rounds read 3.3x a token's bytes and gain
nothing"), and the auto and off suites match within their ranges. On against
off is +18 % at the short-chat median with a wide range, level elsewhere: in
the suite the drafter is a modest, noisy win.

A separate live observation, not a suite run: with `--dspark on`, one answer
of 3,764 tokens ran at 24.90 tok/s, drafts kept 1991 of 2468, per-position
acceptance 84-93 % (engine log); earlier the same day, drafter off, 748
tokens at 15.68 tok/s. Not repeats of one conversation, so this suggests the
drafter pays on long answers rather than proving it. For a card holding about
18 % of the experts, auto's "gain nothing" rule may give up too early.

## Correctness and limitations

No correctness check was run; this is a speed suite. No standard-v1 result:
the standard protocol has no DeepSeek profile. Answers can differ between runs
of the same prompt because the experts kept on the card change over time (a
property of this engine, documented for the model). Thinking off for the
suite, on as served in the live observation above. Loading from the SATA drive
takes about 200 s and is outside all timings. Not tested: `--dspark on` with
`--mmap-experts`, other `--experts` values, needle or coding quality checks.
