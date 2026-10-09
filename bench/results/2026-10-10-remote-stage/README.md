# Remote stage: two and three PCs (2026-10-09 and 2026-10-10)

The measurements behind [docs/REMOTE_STAGE.md](../../../docs/REMOTE_STAGE.md) and
[docs/remote-stage/ENGINEERING.md](../../../docs/remote-stage/ENGINEERING.md): a model's layers split across PCs over
TCP (`--remote-stage`, `--stage-worker`, relay workers). One run per configuration; every number below is in `data/`.

## Hardware

| | PC A (main) | PC B | PC C |
|---|---|---|---|
| GPU | RTX 3080 10 GB, PCIe 4.0 slot at x8 (13.0 GB/s probe) | RTX 2080 Ti 11 GB, PCIe 3.0 x16 (13.2 GB/s) | RTX 3070 8 GB, PCIe 4.0 x16 (26.3 GB/s); drives the desktop too, its worker kept to ~6.5 GB of VRAM |
| CPU | Ryzen 9 5900XT, 16 cores | Ryzen 9 5900X, 12 cores | Ryzen 7 5700X3D, 8 cores |
| RAM | 128 GB DDR4-3200 (96 GB for the container) | 64 GB DDR4-2400 | 64 GB DDR4-3600 |
| storage | not recorded | not recorded | NVMe, a ReFS volume for the shipped model files |

Links: A-B 1 GbE (117 MB/s measured) and 10 GbE (390-400 MB/s measured; PC B's card in a slot negotiating PCIe
2.5 GT/s x2); B-C and A-C 1 GbE (114 MB/s B to C). PC A's host ran CI jobs during some of the 2026-10-09 runs; the
two-PC sweep's table in ENGINEERING.md has the load per row.

## Software

- This branch, built per PC: CUDA 13.0 on Linux (driver 580.178.04 on PC A); on PC C MSVC 2022 + CUDA 13.0 (driver
  591.86) for the native Windows runs and the Linux build under WSL2 for the WSL2 runs.
- PC A: Ubuntu 24.04 in an LXC container. PC B: a Linux host, kernel 6.14. PC C: Windows 11 (build 26200), WSL2 with
  mirrored networking.

## Models

- RVN-Qwen3.8-Flash-Next IQ3_S: `0bserverx/RVN-Qwen3.8-Flash-Next-Abliterated-Uncensored-GGUF`, revision
  `75358160c1d7152de10e43611c5e0be25d8e4638`, 8 shards.
- Huihui-Qwen3.8-Flash-Next UD-Q4_K_XL: `huihui-ai/Huihui-Qwen3.8-Flash-Next-abliterated-GGUF`, revision
  `1e1bd8216ba46a590d8168cca08f68de9536e4ac`, folder `UD-Q4_K_XL`, 4 shards.
- Each with a native pack from `tools/iq_pack.py` (experts read from the GGUF), the stock expert profile and the stock
  draft layer. PC B's RVN IQ3_S worker had a full copy of the model; every other worker ran from files shipped with
  `tools/stage_ship.py` (sparse: their own layers' bytes only).

## Settings

`--kv int8 --kv-resident 32768 --max-context 131072 --spec 4 --spec-min-p 0.5` with the draft layer,
`--expert-cache auto` (the 3070: a fixed count, 2,400 slots for IQ3_S and 1,800 asked / 1,766 given for UD-Q4_K_XL),
no conversation cache, `STRATA_REMOTE_TIMING=1`. The prompt chunk (`--prefill`) is in each label: `c5888` etc.; labels
without one are 2048. `k24-40` means layers 0-23 on PC A, 24-39 on PC B, 40-47 on PC C. "lock": PC C's clocks locked
with `nvidia-smi -lgc 1500,1905 -lmc 7001,7001`. The UD-Q4_K_XL single-card runs: `q4xl-single-auto-1009` with the
served config's `--resident-budget-gib 70`, `q4xl-single-auto-nobudget-1010` without it.

## Workload

- `bench.py LABEL URL corpus.txt`: nine short prompts (code, prose, Chinese; 200-256-token answers), a reasoning
  prompt, a refusal probe and two ~4K / ~7K summaries. "Short decode" = the mean of the nine.
- `bench_ctx.py LABEL corpus.txt URL 8000 32000 64000 100000`: prompts of 7,521 / 30,167 / 57,383 / 90,921 tokens (a
  different slice of the corpus each) with a 320-token answer; the A/B runs used one 29,802-token prompt.
- The corpus: Strata's own Markdown documentation concatenated (744,408 bytes of English text, 69 top-level
  headings; not shipped here, any long English text of that size works). Requests greedy (`temperature 0`), reasoning off,
  the engine's own `timings` recorded. `STRATA_API_KEY` in the environment when the server needs a key.

## Results (prefill / decode tok/s)

| file | configuration | short | 8K | 32K | 64K | 100K |
|---|---|---:|---:|---:|---:|---:|
| `rvn-single-auto-1009` | RVN, PC A alone, auto chunk (5888) | 39.9 | 791 / 38.1 | 1,118 / 37.2 | 1,146 / 39.0 | 1,120 / 38.0 |
| `rvn-remote-k26-c5888-1009` | RVN, A + B, 1 GbE, K=26 | 53.5 | 655 / 48.1 | 1,231 / 49.7 | 1,474 / 47.7 | 1,521 / 45.6 |
| `rvn-remote-k26-c8192-1009-r1200` | the same at chunk 8192, reserve 1200 | 52.5 | 589 / 46.0 | 1,225 / 47.5 | 1,485 / 47.2 | 1,633 / 45.4 |
| `rvn-2stage-3080-3070-k40-c5888-1009` | RVN, A + C (WSL2, not locked), K=40 | 36.2 | 685 / 33.7 | 1,018 / 31.8 | 1,138 / 32.4 | 1,189 / 33.0 |
| `rvn-3stage-k24-40-c5888-1009` | RVN, A + B + C, all 1 GbE, C WSL2 | 40.9 | 497 / 41.3 | 706 / 37.3 | 974 / 36.8 | 1,248 / 36.0 |
| `rvn-3stage-10g-k24-40-c5888-1009` | the same, A-B 10 GbE | 42.1 | 635 / 39.9 | 1,035 / 35.3 | 1,418 / 34.8 | 1,576 / 34.1 |
| `rvn-3stage-10g-lock-k24-40-c5888-1009` | the same, C native Windows, locked | 51.3 | 658 / 47.8 | 1,110 / 46.5 | 1,451 / 44.1 | 1,640 / 44.7 |
| `q4xl-single-auto-1009` | UD-Q4_K_XL, PC A alone, RAM budget 70 GiB | 23.9 | 302 / 22.2 | 324 / 21.8 | 328 / 20.9 | 326 / 21.8 |
| `q4xl-single-auto-nobudget-1010` | UD-Q4_K_XL, PC A alone, no budget | 27.3 | 206 / 25.4 | 212 / 24.5 | 209 / 25.3 | 209 / 23.7 |
| `q4xl-2stage-10g-k26-c5888-1010` | UD-Q4_K_XL, A + B, 10 GbE, K=26 | 38.8 | 659 / 34.4 | 1,147 / 36.9 | 1,192 / 37.5 | 1,164 / 33.6 |
| `q4xl-3stage-10g-lock-k24-40-c5888-1010` | UD-Q4_K_XL, A + B + C, C locked | 38.4 | 549 / 35.3 | 945 / 36.6 | 1,210 / 33.2 | 1,334 / 35.2 |

The other `rvn-remote-*` files are the rest of the two-PC sweep (K=20-28, chunks 2048-8192; `rvn-remote-k26-c8192-1009`
ran out of VRAM after three prompts). The `ab-*` files: the same last stage (layers 40-47, 2,400 slots, chunk 5888,
1 GbE) on the 3070 under WSL2, natively, natively locked, and on the 2080 Ti (twice):

| file | short | 30K prefill / decode |
|---|---:|---:|
| `ab-3070wsl-k40-1009` | 33.0 | 1,033 / 30.6 |
| `ab-3070win-k40-1009` | 33.8 | 995 / 31.7 |
| `ab-3070win-k40-1009lock` | 41.8 | 1,000 / 35.7 |
| `ab-2080ti-k40-1009`, `-1009b` | 41.8, 41.6 | 1,042 / 37.7, 1,041 / 39.0 |

`data/timing-lines.txt`: each run's per-stage timing lines (round trip, the main process between windows, each
worker's own time split into GPU wait, pool + plan and host staging, and its prompt chunk times). `data/clocks-*.csv`:
`nvidia-smi --query-gpu=timestamp,pstate,clocks.sm,clocks.mem,utilization.gpu,power.draw[,temperature.gpu] -lms 100`
during the A/B runs (PC C's clock is in local time, PC B's too); `data/clk_stats.py FILE HH:MM:SS HH:MM:SS` counts
P-states and clocks in a time window. The short-prompt part of each run: 3070 unlocked 23:09:22-23:11:16, 3070 locked
23:27:16-23:28:40, 2080 Ti 23:15:31-23:16:55.

`data/tuner-q4xl-3pc-report.md` and `-console.txt`: `tools/stage_tune.py`'s report and console output for UD-Q4_K_XL on
A + B + C (8 searched splits, two workload-profile runs, the max-context check). `data/tuner-rvn-2pc-report.md` and
`-console.txt`: the first tuner run, RVN IQ3_S on A + B (9 configurations, a 32,000-token goal).

`data/nic-pc-b-1gbe-3pc.txt`: PC B's NIC counters once a second during one long prompt of the all-1 GbE three-PC run;
`data/nic_busy.py FILE` prints the busy stretch's length, bytes and peak rate. `data/tools-and-engine-lines.txt`: the
shipper's summary lines, the shipped files' logical and on-disk sizes on PC C, PC A's start-up lines for UD-Q4_K_XL
alone, and PC C's GPU memory counters at three `--expert-cache` values.

`default_path_check.py URL OUT.json [API_KEY_FILE]` and `data/default-path-{fork1,fork2,base}.json`: greedy answers
to five prompts from one single-GPU config (RVN IQ3_S on PC A, auto chunk) served by the feature's build twice and by
upstream main's (the merge base) once. The three files are byte-identical.

## Not tested

Other quantizations of the family; Windows as the main PC; an in-process split on Windows with and without locked
clocks; more than one run per configuration (decode moved by up to ~14% between two runs of one configuration on
2026-10-09); AMD and Intel GPUs.
