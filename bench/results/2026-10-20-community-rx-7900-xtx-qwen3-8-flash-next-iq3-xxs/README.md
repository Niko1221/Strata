# Community benchmark on AMD Radeon RX 7900 XTX 24 GB (gfx1100), Windows 10, Ryzen 9 5900X

Measured on 2026-10-20 on the reporter's own machine. One model and one profile, two runs on
one quiet box: the legacy `tools/strata_bench.py --long` suite (thinking off) and the
`standard-v1` protocol (thinking on). The tuned setup reads 32K prompts at about 1,820 tok/s
and decodes near 70 tok/s, and the v0.2.1 serve layer changes those numbers by less than the
run-to-run spread against the 2026-10-19 record. The main limitation: the standard-v1 sanity
checks scored 0 of 12 on this run, traced to a per-start engine state that matches upstream
issue #2960 (see Correctness and limitations).

## Hardware and software

- **GPU:** AMD Radeon RX 7900 XTX, 24 GB (gfx1100; the engine reports 24,560 MiB, the
  standard runner 24.0 GiB), stock settings, power limit not recorded. **PCIe:** engine probe
  `pcie_frac 0.55`; link speed and width not recorded.
- **CPU / RAM:** AMD Ryzen 9 5900X, 12 cores / 24 threads. 64 GB installed (63.9 GiB in the
  machine readings, 106 GiB Windows commit limit).
- **Storage:** NVMe SSD (KINGSTON SKC3000D2048G). It holds the GGUF shards and the expert
  offload tier.
- **OS / driver:** Windows 10, build 19044 (10.0.19044). AMD display driver 32.0.31036.15
  (Adrenalin, updated 2026-10-13).
- **Strata:** source checkout of release tag v0.2.1, commit 90767252. The serve layer's own
  version string reads 0.1.46 (the control-package version, shipped alongside release 0.2.1).
  Engine: 0.1.45 prebuilt binary, HIP runtime `engine\amdhip64_7.dll` from the 0.1.45 release
  engine zip `strata-windows-x64-hip.zip` (SHA-256 starts `75a2ff9c`). Python 3.14.6 ran the
  server and the tools.
- **Background:** no other workloads during either run. Production serving was stopped. No
  power or clock capture.

## Model and configuration

- **Model:** ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF, IQ3_XXS (setup pins repository
  revision `ed59f92082b1e93c0e96d60a8b11aab089b52f09`), two shards:
  - `Qwen3.8-Flash-Next-GSQ-RCO-IQ3_XXS-00001-of-00002.gguf`, SHA-256
    `219ea929900dfa9ef091f3aa473fdba6874b65fcb36526d7d851ac9e95856d15`
  - `Qwen3.8-Flash-Next-GSQ-RCO-IQ3_XXS-00002-of-00002.gguf`, SHA-256
    `316b46f3a2dbd68c900f43136ab9449f9dcc3725dfd8c794847c204bc161e113`
- **Custom pack and profiles:** pack `iq3_xxs` built from the two shards, expert profile
  `expert-profile.bin` (SHA-256
  `8f59b4aa8873209dff11c11e37bcda9529a1335b724a1afeea37bf6388975baf`). MTP draft layer from
  Qwen/Qwen3.8-Flash-Next rev `de4b8e4`, prepared under `strata-data/mtp/rt`.
- **Context and cache:** context 131,072. KV int8 with 32,768 cells resident in VRAM. Experts
  split across VRAM, pinned RAM and the SSD tier: expert cache auto to 8,769 slots / 14.2 GiB
  of VRAM, 11 CPU pool workers.
- **Prefill:** `--prefill auto:16384` (Correctness and limitations records why not 32,768).
- **MTP / reasoning / sampling:** `--spec 4 --spec-min-p 0.5` with the MTP draft layer. The
  legacy suite runs thinking off. standard-v1 runs thinking on with the model card's sampling
  (temperature 1.0, top_p 0.95, top_k 20). No calibration, no experimental speed projection.
- **Issue 2274 prompt tuning** (present in every tuned run below): `STRATA_PF_FUSED=1`,
  `STRATA_PF_GEMM=1`, `STRATA_PF_SWITCH_MIN_T=4096`, `STRATA_HIP_WMMA=1`, `STRATA_PA_FAST=1`,
  plus the gfx1100 hipBLASLt tuning table (`STRATA_HIPBLASLT_TUNING=gfx1100-hipblaslt-100500.txt`).
- **Engine version note:** this machine runs engine 0.1.45, not the 0.1.46 the v0.2.1 release
  ships. setup.py's `WIN_HIP_ASSET` (`strata-windows-x64-hip.zip`) matches no asset of the
  v0.2.1 release (the release ships `strata-engine-0.1.46-windows-x86_64-hip-rdna3.tar.gz`),
  so a source install cannot fetch a Windows HIP engine and the earlier 0.1.45 zip stays in
  use.

Engine arguments, local paths and env flags as used: [config-strata-iq3_xxs.json](config-strata-iq3_xxs.json)
(no secrets, machine paths only).

```text
python serve\server.py --engine strata --config strata-iq3_xxs.json --port <port>
full strata.exe launch line and env: config-strata-iq3_xxs.json
```

## Method

Two runs on the same day, same profile, one quiet box. Both commands are in
[repeat.sh](repeat.sh).

1. **Legacy suite** ([results.json](results.json) in this folder):
   `tools/strata_bench.py --config strata-iq3_xxs.json --long --runs 3 --port 28987`. The
   tool started the engine itself ([engine-log-excerpt.txt](engine-log-excerpt.txt)) and sent
   three repetitions per cell with fresh prompts. Every prompt carries a unique tag, so no
   measured run reused tokens (reused tokens 0 everywhere). Thinking off. Output caps: 100
   new tokens (chat), 24 (prompt cells), 64 (agent turns). Prompt and decode tok/s are the
   engine's own timings, TTFT is client-measured. Loading is not in any timing (the engine
   was ready before the first request). This is the same tool and the same cells as the
   2026-10-13 and 2026-10-19 records compared below.
2. **standard-v1**
   ([../2026-10-20-standard-rx-7900-xtx-5b4dbf77](../2026-10-20-standard-rx-7900-xtx-5b4dbf77/results.json)):
   `strata bench run --profile strata-iq3_xxs.json --suite standard-v1 --track default`.
   Draft protocol `strata-qwen38-thinking-v1-draft.1`: one warm-up (excluded), 9 performance
   requests (short, prompt-4k, prompt-16k, three seeds 1-3 each) and 12 sanity checks, output
   caps 512/2048 (caps include reasoning). The engine was freshly loaded for this run.
   `strata bench report` validation passed, run status `complete`. Time after load: 5 min 4 s.

Runs were sequential and state was not reset between requests, so the expert cache was warm
and evolving in both runs. These are not cold-model measurements.

## Results

Medians and ranges (min-max) of 3 runs per cell. Prompt and decode are the engine's own
timings, TTFT is client-measured. The files record no decode aggregate for the prompt cells,
so that decode median is computed from the three per-request engine values. In every session
the first repetition of a cell reads slower than the other two, and the ranges include it.

| Configuration | Actual prompt tokens | Reused tokens | Generated tokens | Runs | Prompt tok/s median (range) | Decode tok/s median (range) | TTFT s median (range) |
| --- | ---: | ---: | ---: | ---: | --- | --- | --- |
| 0.1.43.1, untuned, 4K cell (2026-10-13) | 4,019 | 0 | 24 | 3 | 729.2 (442.6-735.4) | 75.8 (57.5-85.4) | 5.533 (5.486-9.103) |
| 0.1.43.1, untuned, 16K cell (2026-10-13) | 16,058 | 0 | 24 | 3 | 804.9 (774.1-822.9) | 61.6 (57.4-77.9) | 20.004 (19.549-20.773) |
| 0.1.43.1, untuned, 32K cell (2026-10-13) | 32,049 | 0 | 24 | 3 | 846.7 (846.7-850.9) | 64.8 (55.8-76.3) | 37.907 (37.719-37.910) |
| 0.1.43.1 + issue 2274 tuning, 4K (2026-10-13) | 4,019 | 0 | 24 | 3 | 847.5 (491.8-854.5) | 75.9 (57.5-86.1) | 4.764 (4.726-8.194) |
| 0.1.43.1 + issue 2274 tuning, 16K (2026-10-13) | 16,058 | 0 | 24 | 3 | 1,777.2 (1,109.5-1,796.4) | 73.1 (51.0-78.9) | 9.086 (8.985-14.505) |
| 0.1.43.1 + issue 2274 tuning, 32K (2026-10-13) | 32,049 | 0 | 24 | 3 | 1,789.8 (1,785.0-1,791.5) | 59.1 (58.0-76.5) | 17.969 (17.939-17.998) |
| 0.1.45 + issue 2274 tuning, 4K (2026-10-19) | 4,019 | 0 | 24 | 3 | 867.8 (514.3-869.1) | 66.9 (48.8-74.0) | 4.678 (4.671-7.866) |
| 0.1.45 + issue 2274 tuning, 16K (2026-10-19) | 16,058 | 0 | 24 | 3 | 1,824.2 (1,146.5-1,826.9) | 77.9 (68.4-86.2) | 8.833 (8.822-14.040) |
| 0.1.45 + issue 2274 tuning, 32K (2026-10-19) | 32,049 | 0 | 24 | 3 | 1,813.5 (1,789.3-1,825.8) | 65.1 (55.1-74.8) | 17.715 (17.593-17.963) |
| v0.2.1 serve layer, 0.1.45 engine, tuning, 4K (2026-10-20) | 4,019 | 0 | 24 | 3 | 842.3 (510.3-856.7) | 66.6 (49.1-73.2) | 4.804 (4.726-7.911) |
| v0.2.1 serve layer, 0.1.45 engine, tuning, 16K (2026-10-20) | 16,058 | 0 | 24 | 3 | 1,821.9 (1,149.1-1,828.8) | 78.3 (71.3-83.4) | 8.863 (8.827-14.023) |
| v0.2.1 serve layer, 0.1.45 engine, tuning, 32K (2026-10-20) | 32,049 | 0 | 24 | 3 | 1,817.7 (1,813.7-1,817.9) | 69.3 (54.2-79.8) | 17.686 (17.682-17.737) |

Changed settings between the rows:

- untuned -> tuned: the five issue 2274 env flags, `--prefill auto:16384`, the hipBLASLt
  tuning table. Nothing else changed.
- 0.1.43.1 -> 0.1.45: serve layer and engine updated (2026-10-19 record, same tuning).
- 0.1.45 -> v0.2.1 serve layer: serve-layer source update only (commit 90767252), same 0.1.45
  engine and same tuning.

Chat and agent from the same four legacy runs (chat n=9, agent decode n=15 turns, agent cache
hit per session n=3):

| Configuration | Chat decode tok/s median (range) | Agent decode tok/s median (range) | Agent cache hit % median (range) |
| --- | --- | --- | --- |
| 0.1.43.1, untuned | 70.4 (52.7-82.3) | 81.2 (46.4-107.1) | 68.784 (68.780-68.791) |
| 0.1.43.1 + issue 2274 tuning | 70.1 (54.9-83.1) | 71.6 (40.3-109.0) | 68.793 (68.790-68.797) |
| 0.1.45 + issue 2274 tuning | 71.4 (53.1-80.0) | 78.7 (44.2-99.3) | 68.788 (68.757-68.895) |
| v0.2.1 serve layer, 0.1.45 engine, tuning | 70.0 (52.7-81.0) | 82.1 (50.3-107.0) | 69.429 (69.153-69.457) |

standard-v1 (thinking on, n=3 per case; full data in the standard folder's results.json):

| Case | Prompt tok/s median (range) | Decode tok/s median (range) | TTFT s median (range) |
| --- | --- | --- | --- |
| short | 80.2 (79.6-81.2) | 63.3 (54.6-108.9) | 0.991 (0.981-1.000) |
| prompt-4k | 847.3 (768.5-864.8) | 85.4 (73.5-109.0) | 4.855 (4.755-5.348) |
| prompt-16k | 1,822.6 (1,714.7-1,825.0) | 80.6 (76.2-84.6) | 9.038 (9.026-9.600) |

Eight of the nine standard-v1 performance requests reached the output cap (finish reason
`truncated`, caps include reasoning), so their decode numbers sit near the cap. The legacy
prompt cells end after 24 tokens. Sanity: 0 of 12, see the next section.

Memory (engine report, everything loaded): expert arena 40.9 GiB, expert cache 8,769 slots /
14.2 GiB of VRAM, 2.1 GiB VRAM free, KV 32,768 cells in VRAM and the rest streamed from pinned
RAM. Peak RAM and VRAM per request, clocks, power and thermals: not measured.

Per-run data: [results.json](results.json) is the 2026-10-20 legacy run.
[comparisons/](comparisons/) holds the three earlier `results.json` files this report compares
against, copied unchanged and renamed:

- `results-0.1.43.1-baseline.json` - 2026-10-13, Strata 0.1.43.1 (commit a0116b27), untuned
- `results-0.1.43.1-tuned.json` - 2026-10-13, 0.1.43.1 with the issue 2274 tuning
- `results-0.1.45-tuned.json` - 2026-10-19, 0.1.45 with the same tuning

The two 0.1.43.1 files predate the tool's GPU and driver machine facts, so those fields are
empty in them.

## Correctness and limitations

- Two manual smokes through the same serve layer answered correctly: 12 x 34 returned 408
  (thinking on) and 15 + 27 returned 42 (thinking off). The smokes ran in the terminal and are
  not part of the copied files.
- The standard-v1 sanity checks scored 0 of 12 (every case is listed in the standard folder's
  README). The model answered questions the runner never sent: I4 returns an essay titled
  "Top 5 Most Popular Programming Languages", the three short performance cases loop in
  degenerate Chinese reasoning, and I1's reasoning describes "a series of system tags". The
  raw text is in the standard folder's outputs.jsonl.
- Diagnosis. The evidence excludes everything above the engine:
  1. The runner posted the exact frozen fixture messages.
  2. A direct render with the pack's `chat_template.jinja` (in the standard folder's
     `provenance/`) reproduces the intended prompt byte for byte.
  3. The recorded prompt token counts equal the frozen fixture counts on all 22 requests
     (checked against `bench/prompts/standard-v1`; the API and engine counts agree).
  A follow-up probe replay on the same day replayed the exact recorded I1 and short-r1
  requests against five fresh engine starts across four configurations: the issue 2274
  tuning present and stripped, and `--spec 4` and `--spec 2`. Every probe answered both
  cases correctly. The corruption never reproduced, because the bad state belongs to one
  engine start, not to a configuration.
- Match with upstream issue #2960. The failed run's engine session logged
  `KV streaming: 100.00% of 0 block reads hit VRAM, 0.0 MiB read from RAM` on every
  request. That marker, the wrong-language answers to unrelated topics, and the
  whole-run persistence match open upstream issue #2960 (Windows HIP, single RX 7900 XTX,
  roughly one bad start in ten). None of the five probe logs shows the marker. One
  caveat: this native pack rejects `--spec` below 2, so a fully speculative-off start was
  not testable. The performance cells still measure the intended fixtures: the frozen
  token counts match on all 22 requests, including the 9 performance requests. A 0-of-12
  score is a valid standard result. Note for reproduction attempts: this server does not
  replay seeded requests byte-identically (same seed and engine start, different text on
  an immediate resend).
- The serve layer issued 4 invisible loop-recovery continuation turns after R3/R4/T3/T4 that
  the runner cannot see (the standard README notes the runner cannot see the server's
  repeat-stop and loop-recovery switches).
- The runner's stop_engine kills the engine with `taskkill` but never checks the return
  code, so a denied kill is silent. In this run the engine start came up elevated or
  stuck, survived the kill, and left two python processes on port 8080 until a manual
  elevated `taskkill`. Five freshly started probe engines on the same machine died to the
  first `taskkill`, so the runner's silence is upstream and the denied kill itself came
  with the bad engine start. Separately, the run folder's engine.log is 0 bytes by
  design: the serve layer writes its log to the run config's own log file, and the
  runner's capture records nothing on Windows.
- `--prefill auto:32768` fails on this machine: the host buffer allocation dies against the
  106 GiB commit limit of the 64 GB machine. `STRATA_WDDM_BUDGET=0` with
  `--vram-reserve-mib 1000` fails mapped staging (HIP error 719). Both verified to fail here.
  The issue's 128 GB machine does not hit this.
- Windows AMD HIP serves text-only. No vision was tested.
- Not measured: GPU clocks, power, thermals, peak RAM and VRAM per request, the optional
  concurrency stage, and the model and source revisions the runner asks for with
  `--model-revision` and `--source-revision`.

This report was drafted with AI assistance, but every number comes from the machine-recorded
result files copied into this folder, none was typed by hand or generated.
