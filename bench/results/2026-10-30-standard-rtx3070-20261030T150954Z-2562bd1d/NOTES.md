# RTX 3070, one engine

Runner status: **complete** (see README.md).

## Setup

- RTX 3070, one engine.
- Run config: `engine-config.json` (as used, paths redacted).
- Extra environment: `none`.

## Files besides the runner's own

- `runner-console.txt`: the runner's console output (paths redacted).
- `engine-config.json`.
- Engine log: **not kept** for this run (the run script did not save it yet); `missing` in results.json applies.
- `long-prompts/`: an extension, not part of the protocol. [`long_standard.py`](https://github.com/MarcoTool/Strata/blob/bench/0.2.7-split-gfx906/bench/split-gfx906/long_standard.py) sent the benchmark's record lines repeated to 32K, 64K, 128K, 192K and 258K with its summary question and the runner's request settings, three times each, on the same engine after the standard run. `records.jsonl` per request, `summary.tsv`, `stderr.txt` when a request failed.

## Machine (not visible to the runner)

- Host: Ubuntu 22.04, kernel 6.8, ROCm 7.14 with amdgpu DKMS 6.10.5, NVIDIA driver 580 / CUDA 13.0. The engines ran in
  Ubuntu 24.04 containers, which is the OS the runner reports.
- CPU AMD Threadripper 1920X, 96 GB RAM (the OS reports 94.3 GiB), PCIe 3.0, model files on a SATA SSD.
- GPUs: 2x AMD Instinct MI50 32 GB (gfx906); NVIDIA RTX 3070 on a modified 16 GB board (8 GB from the factory), FP16/BF16
  issue rate raised with Zagitta/gpu-unlock, clocks locked to 1,500 to 2,100 MHz. MI50 clocks capped at 1,339 MHz and
  170 W by the host.
- Background: one request at a time; no other inference process on the GPUs during the run.

## Build

Strata 0.2.7 (`5e8a4cca`) with #3986 (gfx906 build fix), commit `93f866a7` as the runner reports; #3987 was not in
this build. How to rebuild both engines, with the configs and the launcher, is on
[`bench/0.2.7-split-gfx906`](https://github.com/MarcoTool/Strata/tree/bench/0.2.7-split-gfx906/bench/split-gfx906). Track `custom`; the runner's track note names the setup by an internal label,
described in words above.
