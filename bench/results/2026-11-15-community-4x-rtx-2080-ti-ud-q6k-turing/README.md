# Community benchmark on 4x NVIDIA GeForce RTX 2080 Ti (sm_75)

Measured on 2026-11-15 by michaljj. Three engine arms on one machine, one model and one configuration:
the 0.1.59 prebuilt package (llama.cpp MMQ prompt path), and the 0.1.60.1 source build of `baf16363b` (0.2.14.1) with
`-DSTRATA_MMQ=OFF` (the 0.2.14 default: Strata's own prompt kernels) and with `-DSTRATA_MMQ=ON` (the old path).
Prompt throughput is the finding: the default build reads prompts 15-18% slower than 0.1.59 here, and the same source
built with MMQ on reads them 14-23% faster than 0.1.59. Decode is faster than 0.1.59 in both 0.1.60 arms.

This is a serving-suite and long-suite comparison, not a `standard-v1` protocol run: no sanity score, no quality check.

## Hardware and software

- GPU and VRAM: 4x NVIDIA GeForce RTX 2080 Ti, 22 GiB each (22,528 MiB), sm_75, 68 SMs per card; selected GPUs 0,1,2,3
  with `--layer-split auto`
- CPU: Intel Xeon E5-2699 v4 (22 cores, 44 threads); installed RAM 121.4 GiB; storage: Lexar NM620 2 TB NVMe
- OS: Linux 7.0.0-43-generic; GPU driver 595.91.07; CUDA 13.4 (build toolkit), engine built for `sm_75` only
- Power limit raised to 280 W on all four cards (`sudo nvidia-smi -pl 280`); the default is 250 W and it returns after a
  reboot. All arms measured at 280 W, box otherwise idle.
- Strata commit `baf16363b` (v0.2.14.1); engine 0.1.60.1, source build. Baseline engine 0.1.59, prebuilt package
  `0.1.59-linux-x86_64-cuda-cuda13`, build `712fec6b4297+cuda13.0`.
- No other workload on the GPUs during the runs; the monitor TUI was attached for part of them (it only reads telemetry).

## Model and configuration

- Qwen3.8-Flash-Next, Unsloth UD-Q6_K_XL (`Qwen3.8-Flash-Next-UD-Q6_K_XL-00001-of-00006.gguf`), served through a native
  expert pack (`packs/unsloth-ud-q6_k_xl`): Q6_K gate/up (ggml type 14) with a Q8_0 down (type 8), 512 experts
- Vision encoder compiled for CUDA, `strata-vision` from the same source (not exercised: no image request in the suites)
- Context 524,288 (`--rope-scaling yarn --rope-scale 2`), `--kv int8 --kv-resident 32768`,
  `--resident-budget-gib 96`, `--expert-cache auto`, `--prefill auto`, MTP draft `--spec 4 --spec-min-p 0.70`,
  `--pcie-frac 0.05`, 2 parallel slots, no calibration run, no speed projection, thinking off for the suites
- Environment: `STRATA_MMQ_KQ=1 STRATA_SM75_PERF=1 STRATA_ROUTE_TAIL_SKIP=0`
- The applied tuning (`--pcie-frac 0.05 --spec-min-p 0.70`) is the one measured on 0.1.59 on 2026-11-14; it was kept
  identical in all three arms. Strata's own preset recheck on the 0.1.60.1 build reported 'Still as fast as when it was
  saved. Kept.'

Configs: `config-0159.json` (package paths) and `config-0160.json` (both source arms; they differ only in the build
flag). The API key and the user's home path are removed.

## Method

- `strata bench run --suite serving-smoke --repeat 3` for the short phases: 2 serial requests, then 4 mixed requests,
  plus the suite's own warm-up pass. One engine start per arm; rounds run back to back on the same engine.
- `tools/strata_bench.py --runs 5 --long` for the long suite: three short chats (100 tokens), fresh 4K / 16K / 32K word
  list prompts with a 24-token cap, and the 5-turn agent session (64 tokens per turn). Medians with ranges over 5 runs.
- Every prompt carries a unique tag, so the engine reports `0 reused` tokens for all prompt rows here. Model loading is
  not timed.
- Warm-up: the first pass after a start is discarded (the engine's expert cache and the prompt path warm up). The smoke
  files under `smoke-0160-*` are `warm` (discarded) plus the measured rounds.
- The 0.1.59 arm was served with `strata serve --profile config-0159.json --parallel 2`, the same slot count the managed
  runtime uses for the two 0.1.60 arms; the suite reports `Config match unverified` for that arm because the run config
  names the package path instead of the install's engine.
- No page cache was dropped between arms; the suites' fresh prompts are unique text, and the engine's own start-up plan
  reported `about 0.0 GB read from the drive per 32768-token prompt (4 chunks)` in every arm, so all three arms read the
  same expert set from RAM/VRAM.

## Results

Long suite, median (range over 5 runs):

| metric | 0.1.59, MMQ path | 0.1.60.1, own int8 (default) | 0.1.60.1, MMQ |
|---|---:|---:|---:|
| short chat decode, tok/s | 43.2 (36.4-50.7) | **52.2 (40.9-68.1)** | 51.5 (41.5-67.0) |
| short chat TTFT, s | 0.58 (0.52-0.61) | 0.57 (0.52-0.63) | 0.57 (0.52-0.69) |
| 4K prompt, tok/s | 589.5 (583.3-590.2) | 486.0 (482.4-487.1) -17.5% | **725.0 (712.1-731.5) +23.0%** |
| 16K prompt, tok/s | 1099.3 (1097.4-1107.2) | 926.6 (923.8-933.1) -15.7% | **1280.9 (1275.2-1290.7) +16.5%** |
| 32K prompt, tok/s | 1439.9 (1434.5-1446.0) | 1231.5 (1223.7-1236.4) -14.6% | **1642.2 (1634.8-1653.9) +14.2%** |
| 32K prompt TTFT, s | 22.33 | 26.09 | **19.59** |
| agent turn decode, tok/s | 64.2 (44.9-74.8) | 63.9 (49.3-72.7) | **65.1 (51.1-76.6)** |
| agent turn TTFT, s | 6.43 | 7.24 | **5.13** |
| agent cache hit | 69.21% | 69.19% | 69.25% |

Prompt rows are 4019 / 16058 / 32049 fresh tokens in every arm (identical prompts). Percentages are against the
0.1.59 column of the same table.

Serving smoke, `output_tok_s` of each round (the `warm` pass is discarded):

| arm | serial, tok/s | mixed, tok/s |
|---|---|---|
| 0.1.59 | 33.98 / 41.95 / 46.19 | 39.40 / 70.45 / 42.69 |
| 0.1.60.1 own int8 | 46.78 / 46.56 / 48.92 | 48.61 / 75.39 / 78.12 |
| 0.1.60.1 MMQ | 49.98 / 54.58 | 71.68 / 50.12 |

The mixed phase is bimodal on this box (~40-50 against ~70-78 tok/s), so its median is not a stable number; the modes
are. The 0.1.59 serial rounds climb 34 -> 42 -> 46 as the cache settles, so its median understates its steady state.

Boot logs for the three arms: `start-lines.txt`. The prompt path each arm took:

- 0.1.59: `K-quant prompt kernels from <home>/.../0.1.59-.../libstrata_kq.so (STRATA_MMQ_KQ=1)`
- 0.1.60.1 default: `prompt kernels: every chunk runs 48 of 48 layers on the int8 kernels.`
- 0.1.60.1 MMQ: `K-quant prompt kernels from <home>/Strata/engine/libstrata_kq.so (STRATA_MMQ_KQ=1)`

## What this shows

- On this hardware and this pack, the 0.2.14 default (`STRATA_MMQ=OFF`) reads prompts 14.6-17.5% slower than 0.1.59.
- The same source built with `-DSTRATA_MMQ=ON` reads prompts 14.2-23.0% faster than 0.1.59. So the regression is the
  prompt-kernel switch, not the 0.2.14 source: prefill improves over 0.1.59 when the MMQ path is present.
- Decode improves in both 0.1.60 arms (+20-22% short chat, +9-11% serial smoke, agent turn decode about equal).
- The one place the default build wins is the high mode of the concurrent (mixed) phase: 75.4-78.1 against 71.7 tok/s.

## Failures and skipped cases

- No failure in any request of any arm; every suite passed (`passed` in each `results.json`).
- The `--parallel 1` attempt at the 0.1.59 baseline is not included: it queued the mixed phase and is not comparable.
- Vision was not exercised (text-only suites).

## What was not tested

- Quality or KL between the three prompt paths; only speed.
- Other quantizations (only UD-Q6_K_XL with Q6_K gate/up and Q8_0 down), other cards, Windows, the app.
- Thermal steady state over a long session: each arm ran 10-20 minutes. GPU temperatures during the runs were 57-70 C
  and power draw 98-107 W per card, so the 280 W limit was not the binding constraint.
- Page-cache state beyond the engine's own start-up plan line, and any other workload on the machine.

## Reading the code, not measured here (unconfirmed static analysis)

- `src/prefill/fused_tile_rule.hpp:198` (`kq_takes`) and `src/core/nv_defaults.cpp:135` keep the 40-79 SM class and
  below-sm_80 cards on the FP16/MMQ routes; the K-quant fused pairs it documents are Q4_K/Q5_K gate/up with a Q5_1 or
  Q8_0 down, while this pack's gate/up is Q6_K. That is consistent with the `int8 kernels` line this build printed, but
  it is a reading of the source, not a measurement.

## Evidence

- `long-0159/results.json`, `long-0160-ownint8/results.json`, `long-0160-mmq/results.json` - `tools/strata_bench.py`
  output, one file per arm
- `smoke-0159/results.json`, `smoke-0160-ownint8/*.json`, `smoke-0160-mmq/*.json` - `strata bench run` serving-smoke
  output per round
- `start-lines.txt` - the engine start line, prompt-kernel line and prefill plan of each arm
- `config-0159.json`, `config-0160.json` - the run configs, credentials and home path removed

Related: Niko1221/Strata#6053 (6x V100, sm_70: own-kernel prompt path ~11% slower than 0.1.58; maintainer reply cites
Turing Titan RTX at -7..-10% in the 0.2.14 notes).
