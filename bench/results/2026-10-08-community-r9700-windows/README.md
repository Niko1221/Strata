# Community benchmark on AMD Radeon AI PRO R9700 (gfx1201), Windows

Measured on 2026-10-08 by a Strata user on Windows with an AMD Radeon AI PRO R9700. This report does two
things: it publishes decode and prompt throughput for Qwen3.8-Flash-Next at IQ3_XXS over 17,154 / 99,696 /
132,886 / 199,316 actual prompt tokens, and it reports a single-variable A/B in which the environment variable
`STRATA_SH_STREAM=0` was worth **+25% to +42% decode throughput**, with the mechanism visible in the engine's
own per-window timing.

The main limitation is that this is **one user's n=3 measurements on a GPU that was shared with other sessions
part of the day**. Absolute throughput on this machine drifts by double-digit percentages within minutes, so
every comparison here is made against a base arm measured in the same window, and two of the five windows are
explicitly reported as invalid. Nothing here has been verified by a maintainer, and **no correctness claim is
made** — the engine is not deterministic run to run on this build, which is documented below.

## Hardware and software

- **GPU:** 1x AMD Radeon AI PRO R9700, 32 GB VRAM, gfx1201 (RDNA4, wave32). Single card; no multi-GPU, no
  layer split. PCIe link speed and width: not recorded. GPU power limit and clocks: not recorded. The engine
  reported `PCIe 0.00` expert fetches per layer-window in every window of every arm in this report, so the
  host-device link was not on the critical path.
- **CPU and RAM:** AMD Ryzen 9 9950X, 16 cores / 32 threads. 48 GB installed RAM (Windows reports 50.58 GB).
  The engine ran 8 expert-pool workers plus its host thread.
- **Storage:** UNIS SSD S5 Ultra 2 TB, NVMe. The engine kept the SSD awake between reads and reported 0 expert
  blob reads from the model file during every decode request in this report.
- **OS and driver:** Windows (exact build not recorded). AMD display driver `32.0.31041.5005`.
- **ROCm / HIP:** ROCm `10.2.0a20260930` from AMD's TheRock wheels, installed inside the project at
  `.rocm-win`. HIP `7.17.26391`, hipBLASLt `100500`. hipBLASLt tuning table
  `tools/hip/gfx1201-hipblaslt-100500.txt`.
- **Strata:** commit `6f32ec070f23ced9f50e704d854d775da52591ab`, tag `v0.1.39`, engine version `0.1.39`.
- **Binary:** the **release prebuilt** `engine/strata.exe`, not a local build. `engine/BUILD.json` at that
  commit reads `"source": "prebuilt"`, `"backend": "hip"`, `"version": "0.1.39"`, `archs` including `gfx1201`,
  `"rocm": "10.2.0a20260930"`, `"hipblaslt_version": 100500`.
- **Source build, separately verified:** the same commit also compiled from source with the ROCm clang
  24.0.0git in 40.17 s of `cmake --build` with zero errors, producing a binary with SHA-256
  `5C38865E48CA620C4B4D56FB38F3F79B490BD384DC05AD15012146922FB45825`. **No performance number in this report
  comes from that binary**; it was only checked for `--help`. See limitations.
- **Background workloads:** **the GPU was not exclusively this report's for the whole day.** The measurement
  driver waits for a free card before every arm (a `tools/opt/gpu-window.json` lock file, and the harness's
  `exit=3` meaning "gpu window busy"), and one arm in the 21:31-21:47 band was rejected outright with
  `exit=3` because another session held the card. Three further arms in the same band died with `exit=1` /
  `exit=-1`, which the driver names a "known VRAM-release crash". Other CPU, disk and network load during the
  session was not recorded.

## Model and configuration

- **Model:** `Qwen3.8-Flash-Next-GSQ-RCO`, **IQ3_XXS**, two GGUF shards. Repository and revision: not recorded
  (the files were already on the machine).
  - `Qwen3.8-Flash-Next-GSQ-RCO-IQ3_XXS-00001-of-00002.gguf` — 47,039,860,096 B
  - `Qwen3.8-Flash-Next-GSQ-RCO-IQ3_XXS-00002-of-00002.gguf` — 28,800,138,432 B
  - SHA-256 of the shards: not recorded.
- **Geometry** (from `include/strata/core/layout.hpp` at this commit): 48 layers, `qsa_interval = 4`, so 12 QSA
  full-attention layers (3, 7, ... 47) and 36 GDN layers. QSA `n_head = 24`, `n_head_kv = 2`, `head_dim = 256`.
  MoE `n_expert = 512`, i.e. 24,576 expert pairs; the engine log confirms `24576 ranked pairs`.
- **Vision encoder:** none. **Reasoning:** off. **Sampling:** greedy (temperature 0). **Calibration:** not
  enabled. **Experimental speed projection:** not enabled. **Control vectors:** not used.
- **Custom pack / profile / MTP:** a **custom native pack** built with `tools/iq_pack.py` (contains
  `experts.bin`), the repository's `data/expert-profile.bin`, and the MTP runtime from `E:\Strata-data\mtp\rt`
  (the engine reported the draft layer loading 1,098 MiB of VRAM in 0.31 s). Pack hashes and the exact
  `iq_pack.py` invocation: not recorded — this is a real reproducibility gap.
- **Context:** `--max-context 262144`. **KV:** `--kv int8`. **Expert cache:** `--expert-cache auto`, which
  resolved to 12,391 resident slots (20.12 GiB of VRAM), pre-filled 12,391 of 12,391 from the profile, with
  `--resident-experts` and `--pcie-frac 0.55`. `--prefill auto` resolved to an 8,192-token prompt chunk with a
  96-slot ring, and the prompt path borrowed 2,063 VRAM cache slots (3.37 GiB).
- **MTP / speculation:** `--spec 4`, `--spec-min-p 0.70`, `--pool-workers 8`. The verify window was 6 tokens,
  70.4 MiB of device buffers.
- **Engine environment** (identical in every arm except the one variable under test):

```text
engine\strata.exe --pack E:\Strata-data\packs\iq3_xxs
  --native E:\Strata-data\models\IQ3_XXS\Qwen3.8-Flash-Next-GSQ-RCO-IQ3_XXS-00001-of-00002.gguf
  --ple-gguf E:\Strata-data\models\IQ3_XXS\Qwen3.8-Flash-Next-GSQ-RCO-IQ3_XXS-00002-of-00002.gguf
  --expert-profile D:\Workstation\Strata\data\expert-profile.bin
  --expert-cache auto --prefill auto --spec 4 --mtp E:\Strata-data\mtp\rt
  --max-context 262144 --kv int8 --resident-experts --pcie-frac 0.55
  --spec-min-p 0.70 --pool-workers 8

env:
  STRATA_HIPBLASLT_TUNING = D:\Workstation\Strata\tools\hip\gfx1201-hipblaslt-100500.txt
  STRATA_RESIDENT_PIN     = 0
  STRATA_HIP_WMMA         = 1
  STRATA_DECODE_TIMING    = 1     # in every arm, including both sides of every A/B
  STRATA_SH_STREAM        = 0     # the variable under test; absent (default 1) in the base arms
```

Server: `python serve/server.py --engine strata --config strata-iq3_xxs.json --port 8080`. No API key was used,
so there is nothing to redact. The config file itself is git-ignored upstream (`/strata-*.json`); a copy is in
[`scripts/strata-iq3_xxs.json`](scripts/strata-iq3_xxs.json), and the per-arm variants in
[`scripts/cfg-ab/`](scripts/cfg-ab/) differ from it only in the `log` path and the one `env` key under test.

**What `STRATA_SH_STREAM=0` actually does.** At `src/core/verify.cpp:903-907` the engine reads the variable once
and folds it into

```cpp
const bool sh_fork = sh_stream_env && !prof_on_ && sh_cs_ != nullptr && ev_fork_ != nullptr && ev_join_ != nullptr;
```

When true, each layer records `ev_fork_` on the main compute stream and makes the second stream `sh_cs_` wait
on it (`:909-911`), runs the shared expert on `sh_cs_` (`:981`), records `ev_join_` on `sh_cs_` (`:986`), and
rejoins the main stream on it (`:1078-1079`). With 48 layers that is **2 event records + 2 stream waits per
layer, 192 synchronisation calls per verify window**. Setting the variable to `0` disables that fork and puts
the shared expert back on the main stream. It changes **only the execution stream and the synchronisation
points** — it does not change any math, dtype, kernel or precision. That is the structural argument in
"Correctness and limitations".

## Method

All measurements come from [`scripts/bench_decode.py`](scripts/bench_decode.py), a **local probe script written
for this measurement session**. It is not part of the upstream repository (`tools/bench_decode.py` does not exist
at this commit) and it was never committed; it is included here in full so a reviewer can check exactly how each
number was parsed. It starts and stops the engine itself, parses the engine's unbuffered stderr timing lines,
writes one JSON per arm, and exits non-zero rather than report a median from fewer than three repeats. It also
excludes any cancelled, failed, truncated or unparseable request from the statistics. See
[`raw/FIELDS.md`](raw/FIELDS.md) for every field and its unit.

- **Workload.** Synthetic prompts from the harness's own generator — no real conversation, no agent loop, no
  code task. `--seed 1234`, greedy, `--max-tokens 128`, `--repeats 3`, **engine restarted between arms**. The
  requested-prompt-size knob is not a token count; `--prompt-tokens 4096 / 24576 / 32768 / 49152` produced
  actual prompts of 17,154 / 99,696 / 132,886 / 199,316 tokens. Every request generated the full 128 tokens.
- **Prompt reuse and cache state.** Repeat 1 of every arm is a fresh prompt (`reused = 0`) against a cold
  expert cache; repeats 2 and 3 reuse the conversation prefix (`reused = prompt_tokens - 5`) against a warm
  cache. Both states are in the JSON per repeat. `stats.decode_tps` mixes them; `stats.decode_tps_warm` (n=2) is
  the steady state and is what the deltas below use.
- **Warm-up and loading.** Model loading is **excluded** from every number — the engine loads, then the harness
  starts timing. Each arm gets one cold repeat to warm the caches.
- **Timing boundaries.** Prompt and decode throughput are taken verbatim from the engine's own
  `strata serve: prompt ... generated ... tok/s` lines. Decode throughput is `generated / decode_ms`, never
  `generated / total request time`. **TTFT was not measured** and total latency was not measured either.
- **Anchored windows.** Because this machine drifts (below), each variant is compared only against a `base`
  arm from the same window, in a `base -> variant -> base` design. **A window whose leading and trailing base
  anchors differ by more than 5% is reported as invalid and its deltas are not used to support a claim.**
  Deltas are `variant / base_anchor - 1`; the anchor used is named every time.
- **Mechanism evidence.** `STRATA_DECODE_TIMING=1` was on in every arm, so the engine also emitted its
  per-window split (`verify`, `GPU-reach wait`, `host`, `draft`, ...). This is host-side accounting of
  work already being done; unlike `STRATA_VERIFY_PROFILE=1`, it does **not** disable the shared-stream fork
  (`sh_fork = sh_stream_env && !prof_on_ && ...`, `verify.cpp:907`), so decode is still the production code
  path. No reported arm ran with `STRATA_VERIFY_PROFILE=1`; `stage_profile` is `null` everywhere.
- **Reproducing.** [`scripts/drivers/`](scripts/drivers/) holds the four PowerShell drivers that produced these
  arms, unmodified, including the card-idle guards and the retry policy.

## Results

Every repeat in every arm below was usable (3 of 3, 0 rejected). Prompt tok/s is repeat 1's fresh-prefill
figure, marked `n=1` because there is exactly one cold repeat per arm; repeats 2-3 re-read 5 tokens and are not
a throughput measurement (see [`raw/FIELDS.md`](raw/FIELDS.md)). TTFT was not measured.

| Configuration | Actual prompt tokens | Reused tokens | Generated tokens | Runs | Prompt tok/s median and range | Decode tok/s median and range | TTFT seconds median and range |
| --- | ---: | ---: | ---: | ---: | --- | --- | --- |
| base, 17,154 tok | 17,154 | 0 / 17,149 (rep 1 / rep 2-3) | 128 | 3 | 922.3 (n=1, cold) | 55.8 [42.3-58.7] | not measured |
| base + `SH_STREAM=0`, 17,154 tok | 17,154 | 0 / 17,149 | 128 | 3 | 943.9 (n=1, cold) | 77.3 [62.9-82.1] | not measured |
| base, 99,696 tok (sweep A) | 99,696 | 0 / 99,691 | 128 | 3 | 1,101.2 (n=1, cold) | 47.7 [43.8-50.9] | not measured |
| base + `SH_STREAM=0`, 99,696 tok (sweep A) | 99,696 | 0 / 99,691 | 128 | 3 | 1,107.6 (n=1, cold) | 64.3 [62.4-72.1] | not measured |
| base + `HC_SPLIT=0`, 99,696 tok (sweep A) | 99,696 | 0 / 99,691 | 128 | 3 | 1,099.7 (n=1, cold) | 46.9 [42.9-51.4] | not measured |
| base + `DOORBELL_STORE=1`, 99,696 tok (sweep A) | 99,696 | 0 / 99,691 | 128 | 3 | 1,096.8 (n=1, cold) | 48.3 [44.7-51.4] | not measured |
| base + `VERIFY_COHERENT=1`, 99,696 tok (sweep A) | 99,696 | 0 / 99,691 | 128 | 3 | 1,089.3 (n=1, cold) | 48.2 [45.4-48.4] | not measured |
| base + `VERIFY_DEVICE_PLAN=1`, 99,696 tok (sweep A) | 99,696 | 0 / 99,691 | 128 | 3 | 1,085.7 (n=1, cold) | 47.8 [42.3-51.2] | not measured |
| base, 99,696 tok (sweep A trailing anchor) | 99,696 | 0 / 99,691 | 128 | 3 | 1,092.3 (n=1, cold) | 48.3 [45.9-48.5] | not measured |
| base, 99,696 tok (sweep B) | 99,696 | 0 / 99,691 | 128 | 3 | 1,110.5 (n=1, cold) | 51.0 [41.8-57.7] | not measured |
| base + `SH_STREAM=0`, 99,696 tok (sweep B) | 99,696 | 0 / 99,691 | 128 | 3 | 1,094.3 (n=1, cold) | 66.8 [55.8-69.4] | not measured |
| base + `HC_SPLIT=0`, 99,696 tok (sweep B) | 99,696 | 0 / 99,691 | 128 | 3 | 1,307.8 (n=1, cold) | 51.1 [44.2-58.1] | not measured |
| base + `DOORBELL_STORE=1`, 99,696 tok (sweep B) | 99,696 | 0 / 99,691 | 128 | 3 | 1,301.9 (n=1, cold) | 52.3 [44.6-52.8] | not measured |
| base, 99,696 tok (sweep B trailing anchor) | 99,696 | 0 / 99,691 | 128 | 3 | 1,309.8 (n=1, cold) | 54.5 [45.3-55.2] | not measured |
| base + `VERIFY_COHERENT=1`, 99,696 tok (sweep B) | 99,696 | 0 / 99,691 | 128 | 3 | 1,285.4 (n=1, cold) | 48.7 [47.5-54.8] | not measured |
| base + `SH_STREAM=0` + `--spec-min-p 0`, 99,696 tok (21:35, unanchored) | 99,696 | 0 / 99,691 | 128 | 3 | 1,193.2 (n=1, cold) | 69.6 [59.8-72.0] | not measured |
| base, 99,696 tok (final) | 99,696 | 0 / 99,691 | 128 | 3 | 1,100.2 (n=1, cold) | 40.8 [37.5-43.7] | not measured |
| base + `SH_STREAM=0`, 99,696 tok (final) | 99,696 | 0 / 99,691 | 128 | 3 | 1,107.5 (n=1, cold) | 54.6 [49.1-55.5] | not measured |
| base + `SH_STREAM=0` + `--spec-min-p 0`, 99,696 tok (final) | 99,696 | 0 / 99,691 | 128 | 3 | 1,112.4 (n=1, cold) | 50.3 [44.6-50.6] | not measured |
| base, 99,696 tok (final trailing anchor) | 99,696 | 0 / 99,691 | 128 | 3 | 1,113.4 (n=1, cold) | 47.2 [45.1-52.0] | not measured |
| base, 132,886 tok (long) | 132,886 | 0 / 132,881 | 128 | 3 | 1,096.6 (n=1, cold) | 48.3 [42.9-50.2] | not measured |
| base + `SH_STREAM=0`, 132,886 tok (long) | 132,886 | 0 / 132,881 | 128 | 3 | 1,106.2 (n=1, cold) | 68.7 [55.7-71.6] | not measured |
| base, 132,886 tok (long trailing anchor) | 132,886 | 0 / 132,881 | 128 | 3 | 1,106.1 (n=1, cold) | 47.4 [42.8-50.0] | not measured |
| base + `SH_STREAM=0`, 199,316 tok (longest) | 199,316 | 0 / 199,311 | 128 | 3 | 1,124.9 (n=1, cold) | 63.0 [54.7-67.0] | not measured |
| base + `SH_STREAM=0`, 99,696 tok (late re-check) | 99,696 | 0 / 99,691 | 128 | 3 | 1,249.2 (n=1, cold) | 71.9 [61.5-76.1] | not measured |
| base + `SH_STREAM=0` + `--spec-min-p 0`, 99,696 tok (late re-check) | 99,696 | 0 / 99,691 | 128 | 3 | 1,152.5 (n=1, cold) | 49.6 [47.4-50.7] | not measured |

Per-arm JSON and the harness's own summary tables are in [`raw/`](raw/). Nothing was edited by hand.

### Which file is which

Every row above is one JSON in `raw/` (plus the harness's `<name>.md` summary of the same run). Arm start
times are the `run.timestamp` of each file.

| Row label in the table above | File in `raw/` | Started |
| --- | --- | --- |
| base, 17,154 tok | `ctx-short-base.json` | 21:27:24 |
| base + `SH_STREAM=0`, 17,154 tok | `ctx-short-shstream0.json` | 21:29:01 |
| base, 99,696 tok (sweep A) | `w2-base.json` | 19:31:32 |
| base + `SH_STREAM=0`, 99,696 tok (sweep A) | `w2-shstream0-2.json` | 19:35:15 |
| base + `HC_SPLIT=0`, 99,696 tok (sweep A) | `w2-hcplain-2.json` | 19:48:28 |
| base + `DOORBELL_STORE=1`, 99,696 tok (sweep A) | `w2-dbstore-2.json` | 19:51:18 |
| base + `VERIFY_COHERENT=1`, 99,696 tok (sweep A) | `w2-coherent-2.json` | 19:54:02 |
| base + `VERIFY_DEVICE_PLAN=1`, 99,696 tok (sweep A) | `w2-devplan-2.json` | 19:56:53 |
| base, 99,696 tok (sweep A trailing anchor) | `w2-base-2.json` | 19:59:44 |
| base, 99,696 tok (sweep B) | `w2-base-a.json` | 20:27:50 |
| base + `SH_STREAM=0`, 99,696 tok (sweep B) | `w2-shstream0.json` | 20:30:39 |
| base + `HC_SPLIT=0`, 99,696 tok (sweep B) | `w2-hcplain.json` | 20:33:27 |
| base + `DOORBELL_STORE=1`, 99,696 tok (sweep B) | `w2-dbstore.json` | 20:36:02 |
| base, 99,696 tok (sweep B trailing anchor) | `w2-base-b.json` | 20:38:32 |
| base + `VERIFY_COHERENT=1`, 99,696 tok (sweep B) | `w2-coherent.json` | 20:41:03 |
| base + `SH_STREAM=0` + `--spec-min-p 0`, 99,696 tok (21:35, unanchored) | `w2-shstream0-minp0.json` | 21:35:38 |
| base, 99,696 tok (final) | `final-base.json` | 20:58:51 |
| base + `SH_STREAM=0`, 99,696 tok (final) | `final-shstream0.json` | 21:01:43 |
| base + `SH_STREAM=0` + `--spec-min-p 0`, 99,696 tok (final) | `final-shstream0-minp0.json` | 21:04:32 |
| base, 99,696 tok (final trailing anchor) | `final-base-b.json` | 21:07:16 |
| base, 132,886 tok (long) | `final-long-base.json` | 21:17:18 |
| base + `SH_STREAM=0`, 132,886 tok (long) | `final-long-shstream0.json` | 21:20:44 |
| base, 132,886 tok (long trailing anchor) | `final-long-base-b.json` | 21:24:01 |
| base + `SH_STREAM=0`, 199,316 tok (longest) | `w2-shstream0-longctx.json` | 21:53:57 |
| base + `SH_STREAM=0`, 99,696 tok (late re-check) | `ab-shstream0.json` | 21:47:49 |
| base + `SH_STREAM=0` + `--spec-min-p 0`, 99,696 tok (late re-check) | `ab-minp0.json` | 21:50:56 |

### Which windows are trustworthy

This is the most important table in the report, because it invalidates two of the five windows.

| Window | Prompt tokens | Leading base anchor (median / warm) | Trailing base anchor | Drift, median | Drift, warm | Verdict |
| --- | ---: | --- | --- | ---: | ---: | --- |
| ctx-short | 17,154 | 55.8 / 57.25 | **none run** | not measured | not measured | **incomplete** - provisional only |
| sweep A | 99,696 | 47.7 / 49.3 | 48.3 / 48.4 | +1.26% | -1.83% | **valid** |
| sweep B | 99,696 | 51.0 / 54.35 | 54.5 / 54.85 | +6.86% | +0.92% | **median invalid**, warm-column usable |
| final | 99,696 | 40.8 / 42.25 | 47.2 / 49.6 | **+15.69%** | **+17.40%** | **invalid** |
| long | 132,886 | 48.3 / 49.25 | 47.4 / 48.7 | -1.86% | -1.12% | **valid** |

The `final` window drifted **+17.4% in 8.4 minutes** (base arm started 20:58:51, trailing base arm started
21:07:16) with no change to the configuration. Any absolute tok/s from that window, and any delta in it, should
be read with that in mind. This is also why the three-arm `--spec-min-p` comparison below is reported but not
relied on.

### `STRATA_SH_STREAM=0`

Warm medians (steady state, n=2 each), against the base anchor of the same window:

| Prompt tokens | Window | Base warm | `SH_STREAM=0` warm | Change | Anchor status |
| ---: | --- | ---: | ---: | ---: | --- |
| 17,154 | ctx-short | 57.25 | 79.7 | **+39.2%** | no trailing anchor |
| 99,696 | sweep A | 49.3 | 68.2 | **+38.3%** | valid |
| 99,696 | sweep B | 54.35 | 68.1 | **+25.3%** | warm column valid |
| 99,696 | 21:35, no base arm in that window | not measured | 70.8 | not measurable | **no anchor at all** |
| 99,696 | late re-check | not measured | 74.0 | not measurable | **no base arm at all** |
| 132,886 | long | 49.25 | 70.15 | **+42.4%** | valid |
| 199,316 | longest | not measured | 65.0 | not measurable | **no base arm at all** |

Every anchor-valid window, at every prompt length measured, shows a gain between **+25% and +42%**. On overall
medians (cold repeat included) the three anchor-valid windows give +38.5% at 17,154 tokens, +34.8% in sweep A
and +42.2% at 132,886 tokens. Sweep B's median column is the one invalidated by its +6.86% anchor drift; it
reads +31.0%.

### Mechanism

The gain is not an artefact of the draft statistics. At 132,886 prompt tokens in the valid `long` window,
across all six requests (two base arms, one variant arm):

| Per-window metric | Base (6 requests) | `SH_STREAM=0` (3 requests) |
| --- | --- | --- |
| `wait_ms` — host ms blocked per window | 27.42 - 28.82 | 16.53 - 17.88 |
| `verify_ms` | 32.50 - 35.10 | 23.12 - 23.42 |
| `tokens_per_window` | 1.73 - 1.94 | 1.73 - 2.00 |
| `draft_ms` | 3.04 - 3.31 | 3.02 - 3.20 |
| Draft acceptance (median, %) | 77.5 / 76.5 | 75.0 |
| Expert-cache hit rate (median, %) | 94.8 / 97.5 | 97.4 |

The two `wait_ms` ranges **do not overlap at all**; the token yield per window, the draft cost and the
acceptance rate are unchanged. The saving is ~10 ms per window in exactly the term that measures the host
waiting on the GPU, which is what removing 192 stream/event synchronisation calls per window should do. On this
card the shared expert evidently did **not** have enough independent work to hide behind the main stream, so the
fork cost more than it saved.

The same pattern appears at 17,154 prompt tokens (`wait_ms` 24.79-26.89 -> 14.29-16.93) and at 99,696
(26.66-27.44 -> 15.92-18.01, taken from the invalid `final` window, so indicative only).

### `--spec-min-p 0` — mechanism reproduces, speed sign does not

`--spec-min-p 0` removes the early exit at `pj >= min_p`, so every window runs the full 4 speculative steps.
That part is unambiguous and reproduced in all three pairs:

| Pair (arm start times) | `SH_STREAM=0` alone | `+ --spec-min-p 0` | Change, warm | Drafts per request | Acceptance | Gap between the two arms |
| --- | ---: | ---: | ---: | --- | ---: | ---: |
| `final` window, 21:01:43 -> 21:04:32 | 55.05 | 50.45 | **-8.4%** | 72 / 84 / 81 -> 173 / 165 / 171 | 77.4% -> 40.9% | 2.8 min |
| 21:35:38, against the 20:30:39 sweep B arm | 68.1 | 70.8 | **+4.0%** | 77 / 72 / 79 -> 169 / 154 / 171 | 72.2% -> 42.0% | **65 min** |
| late re-check, 21:47:49 -> 21:50:56 | 74.0 | 50.15 | **-32.2%** | 77 / 83 / 78 -> 162 / 168 / 177 | 74.4% -> 42.9% | 3.1 min |

Draft volume roughly doubles and acceptance collapses to ~41-43% every time — that is a property of the
setting, not of the environment. **The throughput effect is not established.** The two arms in each pair differ
in no setting other than `--spec-min-p`, but **no pair has a base anchor taken between its two arms**: the
21:35 arm sits 57 minutes past sweep B's trailing anchor and the 21:47 arm has no anchor at all, so all three
comparisons are exposed to exactly the drift documented above. One of the three gaps is 65 minutes. Reported as
inconclusive rather than as a negative result.

### Switches with no measurable effect

`STRATA_HC_SPLIT=0`, `STRATA_DOORBELL_STORE=1`, `STRATA_VERIFY_COHERENT=1` and `STRATA_VERIFY_DEVICE_PLAN=1`
were each measured on the same 99,696-token workload — the first three twice, `VERIFY_DEVICE_PLAN=1` once.
Sweep A is the anchor-valid window:

| Switch | Sweep A (valid) vs base 47.7 / 49.3 | Sweep B vs base 51.0 / 54.35 | Sweep B vs trailing anchor 54.5 / 54.85 | Verdict |
| --- | ---: | ---: | ---: | --- |
| `HC_SPLIT=0` | -1.7% / -0.3% | +0.2% / +0.5% | -6.2% / -0.5% | **no effect** |
| `DOORBELL_STORE=1` | +1.3% / +1.1% | +2.6% / -3.3% | -4.0% / -4.2% | **no effect** |
| `VERIFY_COHERENT=1` | +1.0% / -2.0% | -4.5% / -4.8% | -10.6% / -5.7% | **no effect** |
| `VERIFY_DEVICE_PLAN=1` | +0.2% / +0.4% | not run | not run | **no effect** |

Every valid-window delta is smaller than the window's own anchor drift would tolerate to mean anything, and in
the sweep B column the sign flips depending on which anchor you pick. At n=3 on this machine the honest
conclusion is that **no effect could be established**, not that these toggles are exactly neutral. They are
included so nobody spends another day on them.

### Memory

- **VRAM at start-up only.** The engine logged `strata serve: <N> MiB of VRAM free with everything loaded` at
  every engine start: **268 to 524 MiB free** across the 43 starts behind these arms (the value varies with the
  arm's extra buffers). This is a start-up snapshot. **Peak VRAM during inference was not measured.**
- **RAM:** 23.10 - 23.25 GiB of experts resident in pageable system RAM, logged per request. Virtually no
  change between arms and repeats.
- **Expert cache:** 12,391 resident slots, 20.12 GiB of VRAM, pre-filled 12,391 of 12,391 from the profile at
  start, 0 evictions needed; `PCIe 0.00` per layer-window throughout; 0 expert blob reads from the model file
  in every decode request.
- Paging or out-of-memory failures: **none during decode.** The failures listed below happened at engine
  start, during prompt prefill, or at request validation.

### Failures and skipped cases

These are outside the throughput summary and are listed for completeness:

| Time | Arm | Prompt | Outcome |
| --- | --- | --- | --- |
| 20:05:50 | sweep A `SH_STREAM=0` | 99,696 | `exit=-1`, driver calls it a known VRAM-release crash; re-run succeeded at 20:08:54 |
| 21:13:44 | `final-long-base` | ~199,316 | `exit=-1` during prefill, at 163,840 of 199,316 tokens read; the arm was then re-run at 132,886 and succeeded at 21:17:18 |
| 21:31 / 21:35 / 21:39 | `ctx-long-base` | intended 99,696, actually sent ~444,000 | `exit=1` three times in a row; **all three repeats returned `HTTP Error 400: Bad Request`**. See limitation 7: this was an input-format bug, not a prefill failure |
| 21:45:34 | `ctx-long-shstream0` | 99,696 | **`exit=3`, rejected because the GPU was busy with another session** |
| 21:46:04 | `ctx-long-shstream0` | 99,696 | `exit=-1` after the retry |
| 21:38:25 | long-context probe | ~200k | engine log is 136 bytes: a banner and nothing else, no prompt line |
| 21:47:55 | `ctx-long-base` | 99,696 | aborted, exit code empty |

The last successful long-context arm is therefore the 21:53 one in the table above, not the 21:38 probe.

## Correctness and limitations

**No correctness check was run, and no correctness claim is made.** The repository's
`tools/needle_bench.py` was **not** executed; there is no recall result, no coding task, no tool-call test and
no image question in this report. Only throughput was measured.

**The engine is not deterministic run to run on this build, so token-for-token equality is not a usable
criterion.** Two cold `base` engine starts, same commit, same configuration, same greedy request, same seed,
produced different text: **389 words versus 383 words** (2,450 and 2,407 bytes as stored; those counts include
the files' 12 and 10 CRLF line breaks, so the newline-normalised character counts are 2,438 and 2,397),
diverging at **word index 72, 453 characters in**. The repository's own `docs/AMD_HIP.md` states that roughly
1 in 10 HIP starts produces different greedy output for reasons not explained. Both texts are kept in
[`raw/ab-text-base.txt`](raw/ab-text-base.txt) and [`raw/ab-text-base2.txt`](raw/ab-text-base2.txt).

Consequently `STRATA_SH_STREAM=0` is argued for on two grounds that survive non-determinism, **neither of
which is a numerical-equivalence proof**:

1. **Structural.** It sets `sh_fork` to false at `src/core/verify.cpp:907`. It removes stream forks and event
   waits. It touches no weight, no dtype, no kernel, no precision setting and no accumulation order within a
   kernel. The arithmetic performed per window is unchanged.
2. **Distributional.** Draft acceptance (75.0% vs 77.5% / 76.5%), expert-cache hit rate (97.4% vs 94.8% /
   97.5%), tokens per window (1.73-2.00 vs 1.73-1.94) and generated length (128 in every request) are
   unchanged, while only `wait_ms` moves. A numerical change would be expected to disturb the distribution, not
   just the synchronisation term.

**The second shstream0 greedy run never completed**, so this report **cannot** say whether `SH_STREAM=0`
changes the output text at all — only that its distributions agree. That question is open.

Limits of this report, in rough order of how much they should worry a reader:

1. **The GPU was shared.** One arm was rejected outright because another session held the card, three arms died
   in the same band with the "VRAM-release crash", and the base anchor in the `final` window drifted **+17.4%
   in 8.4 minutes**. Contention with a peer session is the most likely cause and it was not controlled for.
2. **Absolute tok/s across windows are not comparable.** Only same-window deltas mean anything. Two of the five
   windows failed the 5% anchor test and a third (`ctx-short`) has no trailing anchor at all.
3. **n=3 repeats per arm, no confidence intervals.** Only median and range are reported. With n=2 for the warm
   column there is no interval at all.
4. **An unexplained two-mode `draft_ms`.** Across the 78 timed requests, per-window draft cost was either
   **2.57-4.15 ms** (66 requests) or **8.72-14.56 ms** (12 requests), with every other component agreeing. The
   high mode hit *all three* arms of the `final` window (base, shstream0 and min-p0 alike) plus the late
   min-p0 arm, so it is window-wide rather than arm-specific. Two of those four arms (`final-base` and
   `final-shstream0`) drafted only 77-84 tokens, unchanged from their normal mode, so their ~3x draft cost is
   **not** explained by the setting and its cause is unknown. Every absolute tok/s here therefore carries an
   unquantified environment component. (The min-p0 arms' own high mode *is* separately explicable: they draft
   about 170 tokens per request.)
5. **Synthetic prompts only.** No real conversation, no agent or tool loop, no code, no multi-turn. Output cap
   was 128 tokens, which is short enough that a 4-token draft window yields ~1.8 tokens per window; longer
   outputs and different acceptance rates could change the balance.
6. **TTFT and total latency not measured.** Only the engine's own prompt and decode tok/s exist in this data.
7. **Long-context behaviour is only partly established.** 199,316 actual prompt tokens (context 199,444) did
   prefill fresh in 177.2 s at 1,124.9 tok/s and then decoded at 63.0 tok/s median, so 132,886 is *not* the
   ceiling — but that arm had **no base anchor at all**, so its decode number cannot be compared to anything.
   `--max-context 262144` remains **unverified**: no prompt larger than 199,316 was ever attempted
   successfully. Two failures in this band are easy to conflate, so they are separated here:

   - The ~200k probe ([`scripts/drivers/probe-199k.ps1`](scripts/drivers/probe-199k.ps1)) died before emitting
     a timing line. That one **was** a real prefill failure: the engine log simply stops mid-prompt.
   - The `ctx-long-base` arm failed three times in a row but **not** during prefill: all three repeats returned
     `HTTP Error 400: Bad Request`. Its token file stores one vocabulary id per line as plain text, so the
     text re-tokenizes to roughly 4.18 tokens per line — 106,225 lines is about 444,000 tokens, well past
     `--max-context 262144`. The request was rejected before the engine read it. **This failure therefore
     says nothing about whether a ~200k prompt is feasible, in either direction**, and the driver's own
     "known VRAM-release crash" label is a hard-coded string it prints for any non-zero exit, not a diagnosis.

   The evidence that 199,316 works is the successful arm in the table above, not the absence of failures
   around it. And the "163,840 = 10 x prompt-cache-every" reading of the earlier crash is **not supported**:
   that run happened while another session held the card, and the same length succeeded three times when run
   exclusively.
8. **One GPU, one model, one quantization, one configuration.** No multi-GPU, no layer split, no vision, no
   other quantization, no concurrency, no sustained thermal run.
9. **The source build was never benchmarked.** It was only proven to compile (40.17 s, zero errors) and to
   answer `--help`. No speed comparison against the prebuilt binary exists, so it is unknown whether the
   shipped binary matches a local build of this commit.
10. **Prompt throughput is n=1 per arm.** One cold prefill per arm, with no repetition, so it has no range.
11. **Pack provenance is incomplete.** GGUF SHA-256s and the exact `tools/iq_pack.py` invocation were not
    recorded, so the model preparation is not bit-reproducible from this report.
12. **PCIe link, power limit, clocks and background CPU/disk load were not recorded.**

**What a maintainer could reuse from this:** the anchored-window method and its 5% invalidation rule, the
per-window `wait_ms` evidence in section "Mechanism", the four falsified toggles, and the non-determinism
measurements. What needs independent confirmation before anyone changes the engine: the `STRATA_SH_STREAM=0`
gain, which was measured by one user on one shared card and which this report deliberately does not assert as
a verified result.