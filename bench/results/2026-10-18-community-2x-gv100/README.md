# Community benchmark: 2x Quadro GV100 (sm_70) vs 1x Quadro GV100, Unsloth UD-Q4_K_XL

Measured on 2026-10-18. Three independent benchmark sessions per GPU configuration
(`tools/strata_bench.py`, suite 1, 3 runs per test per session; 9 sessions' worth of
data in total). Speeds are the engine's own timings; first-token times are client-side
over HTTP. Speed only: no correctness check was run.

Headline: decode scales **1.59x** from one GV100 to two (42.3 -> 67.1 tok/s median);
prefill scales ~1.26–1.34x. A single GV100 keeps 42 tok/s decode on the Q4_K_XL pack
with 125 GiB system RAM. Both cards run at a **150 W power cap** (GV100 TDP is 250 W),
so these numbers are conservative.

## Hardware and software

- GPU and VRAM: 2x Quadro GV100, 32 GiB each (compute capability 7.0); tested as a
  pair (`GPUS=0,1`) and alone (`GPU=0`, GPU 0 only). PCIe Gen3 x16 on both slots.
  The cards have NVLink connectors but **no bridge installed** (all links inactive).
- CPU: Intel Xeon E5-1650 v4 @ 3.60 GHz (6 cores / 12 threads, AVX-512);
  installed RAM: 125.7 GiB DDR4 (8x 16 GB).
- Storage: NVMe (U.2/M.2 adapter, Gen3 x4 link) holds the model pack; timings
  exclude model loading (server already running).
- OS: Ubuntu 24.04.4 LTS, kernel 6.8.0-117; GPU driver 580.159.03;
  engine is a CUDA 12 build (driver advertises CUDA 13.0).
- Strata: project 0.1.45, commit `477ac13c`, engine 0.1.45, official Docker image,
  container limited to the selected GPU(s) via `device_ids`.
- Background workloads: none during the runs (idle machine; benchmark client on the
  same host).
- **Power limits: both GV100s capped at 150 W** via a systemd service
  (`nvidia-power-limit.service`). Default TDP is 250 W. Not measured whether the
  cap is actually reached during these workloads.

## Model and configuration

- Model: Qwen3.8-Flash-Next, Unsloth **UD-Q4_K_XL**; `--pack
  /data/packs/unsloth-ud-q4_k_xl` + native `Qwen3.8-Flash-Next-UD-Q4_K_XL-00001-of-00004.gguf`.
- Expert profile: learned profile (`expert_profile_save`), persisted across restarts;
  the expert cache was **warm from daily use** before every session.
- Context: 131,072 tokens; `--kv int8`, `--kv-resident 32768`; `--expert-cache auto`;
  `--prefill auto`; `--spec 4 --spec-min-p 0.5`; `--mtp /data/mtp/rt`;
  `layer_split: auto` (dual-GPU run); parallel slots: 1.
- Env: `STRATA_HC_Q8=1`, `STRATA_ROUTE_TAIL_SKIP=0`.
- Reasoning effort configured `low`; thinking off for the suite (tool default);
  temperature 0.

```text
strata --pack /data/packs/unsloth-ud-q4_k_xl \
  --native /data/models/unsloth-UD-Q4_K_XL/Qwen3.8-Flash-Next-UD-Q4_K_XL-00001-of-00004.gguf \
  --expert-profile /opt/strata/data/expert-profile.bin --expert-cache auto \
  --prefill auto --spec 4 --spec-min-p 0.5 --mtp /data/mtp/rt \
  --max-context 131072 --kv int8 --kv-resident 32768
```

(Docker entrypoint builds this from the saved config; the API key was removed.)

## Method

`python tools/strata_bench.py --url http://127.0.0.1:8086` against the already-running
server; model loading not timed. The suite sends three short chat questions (up to 100
new tokens), fresh ~4K and ~16K word-list prompts (24 new tokens, unique tag so the
conversation cache cannot help), and a 6-turn agent session with four tools.

Per configuration we ran **three separate benchmark sessions** on an idle machine
(same engine process, sessions spaced apart; a 20 s pause between sessions). The GPU
configuration was changed by recreating the container (`GPU=0` + `device_ids: ['0']`
vs `GPUS=0,1` + `device_ids: ['0', '1']`), so each configuration was benchmarked after
a fresh engine start with a warm, persisted expert cache.

## Results

Session medians (n=3 runs per test per session), then the overall median.

### Short chat (decode)

| Config | Session medians (tok/s) | Decode median (range over all 9 runs) | TTFT median (s) |
| --- | --- | --- | --- |
| 2x GV100 | 70.0 / 65.6 / 67.1 | **67.1** (51.3–89.9) | **0.45** (0.38–0.51) |
| 1x GV100 | 42.3 / 41.5 / 43.0 | **42.3** (32.3–48.8) | **0.76** (0.59–1.05) |

Decode spread within a session is wide (short 100-token runs, warm-up effects);
session medians agree within ~4%.

### Prompt throughput (fresh prompts)

| Config | 4K prompt tok/s (sessions) | 4K median | 4K TTFT (s) | 16K prompt tok/s (sessions) | 16K median | 16K TTFT (s) |
| --- | --- | --- | --- | --- | --- | --- |
| 2x GV100 | 1070.8 / 1070.6 / 1070.8 | **1070.8** | 3.78 | 1751.8 / 1708.4 / 1709.2 | **1709.2** | 9.44 |
| 1x GV100 | 799.5 / 797.7 / 797.8 | **797.8** | 5.07 | 1357.6 / 1354.3 / 1354.2 | **1354.3** | 11.89 |

Prompt throughput is extremely stable (±0.1% across sessions). The first 2x session
measured 1751.8 tok/s at 16K; the two later sessions 1708–1709 (unexplained, ~2.5%).

### Agent session (conversation cache)

| Config | Decode tok/s (sessions) | Cache hit % | Turn TTFT median (s) |
| --- | --- | --- | --- |
| 2x GV100 | 68.6 / 68.5 / 68.8 | ~68.7 | 4.29 |
| 1x GV100 | 55.0 / 54.9 / 52.5 | ~68.6 | 5.39 |

### Dual vs single (median-to-median)

| Metric | 1x GV100 | 2x GV100 | Gain |
| --- | ---: | ---: | ---: |
| Chat decode tok/s | 42.3 | 67.1 | **1.59x** |
| 4K prompt tok/s | 797.8 | 1070.8 | 1.34x |
| 16K prompt tok/s | 1354.3 | 1709.2 | 1.26x |
| Chat TTFT (s) | 0.76 | 0.45 | 0.59x |
| Agent decode tok/s | 54.9 | 68.6 | 1.25x |

Memory: both configurations run near-full VRAM (~32 GiB per card on the dual run;
~31 GiB on the single-card run with all 77 GB of experts in system RAM). No paging
or OOM failures observed.

Per-request data for all six sessions: `results-2x-r1.json` ... `results-1x-r3.json`
(`results-<n>x-r<session>.json`), with the tool-generated session reports in
`session-2x-r1.md` ... `session-1x-r3.md`.

## Correctness and limitations

- Speed suite only; no needle/recall or coding checks were run.
- 150 W power cap on both cards (see above); NVLink not connected.
- Short-chat decode has high per-request variance (32–90 tok/s); medians over 9 runs
  and three sessions are the reliable figure.
- The expert cache was warm (learned profile + daily use) in every session; a
  cold-cache run would start slower.
- Single-GPU run used GPU 0 only; the second card was fully idle.
- Total suite time per session: ~170 s.
